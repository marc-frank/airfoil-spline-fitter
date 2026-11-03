"""The app's optimizer adapters: one error measure, progress from every
optimizer, settings that reach the library, and configurations that fit."""
import bspline_varpro
import numpy as np
import pytest
from bspline_varpro.chain import chain_data, locate
from scipy.interpolate import BSpline

from spline_fitter.optimization.engine import (
    get_evaluation_count,
    reset_iteration_tracking,
    run_optimization,
)
from spline_fitter.optimization.models import (
    AirfoilData,
    ContinuityMode,
    OptimizationConfig,
    OptimizerType,
    SplineConfig,
)
from spline_fitter.optimization.optimizers import gn_varpro
from spline_fitter.optimization.optimizers.gn_varpro import GNVarProConfig, GNVarProOptimizer
from spline_fitter.spline.spline import create_open_uniform_knots


def small_airfoil(noise=0.0, n=19):
    t = np.linspace(0, np.pi, n)
    upper = np.column_stack(((1 + np.cos(t)) / 2, 0.09 * np.sin(t)))
    lower = np.column_stack(((1 - np.cos(t)) / 2, -0.06 * np.sin(t)))
    rng = np.random.default_rng(82)
    for sec in [upper, lower]:
        sec[1:-1, 1] += rng.normal(0, noise, len(sec) - 2)
    sections = tuple(tuple(map(tuple, sec)) for sec in [upper, lower])
    return AirfoilData(sections[0] + sections[1][1:], sections, (n - 1,), 2)


def test_direct_lsq_reports_rms_error_like_varpro():
    """Both optimizers' error is the RMS distance of the points to their curve
    points, so their runs compare (Direct LSQ used to report the mean)."""
    airfoil = small_airfoil(0.003)
    result = run_optimization(SplineConfig(degree=3, num_control_points=6),
                              OptimizationConfig(optimizer=OptimizerType.DIRECT_LSQ), airfoil)
    data = chain_data([np.asarray(sec) for sec in airfoil.sections])
    section, local = locate(result.parameters, 2)
    knots = create_open_uniform_knots(3, 6)
    curves = [BSpline(knots, np.asarray(cp), 3) for cp in result.section_control_points]
    distances = np.linalg.norm(
        np.array([curves[s](t) for s, t in zip(section, local, strict=True)]) - data, axis=1)
    assert result.fun == pytest.approx(np.sqrt(np.mean(distances ** 2)))
    assert result.fun != pytest.approx(np.mean(distances))


def test_direct_lsq_reports_progress():
    """Direct LSQ counts its solves and hands its result to the callback, so
    a run gets an evaluation count, a loss history and a run log."""
    reset_iteration_tracking()
    seen = []

    def callback(x, error, preview=None):
        seen.append((x.shape, error, preview))
        return False

    result = run_optimization(SplineConfig(degree=3, num_control_points=6),
                              OptimizationConfig(optimizer=OptimizerType.DIRECT_LSQ),
                              small_airfoil(0.002), callback=callback)
    assert get_evaluation_count() >= 2
    assert len(seen) == 1
    shape, error, (control_points, knots, degree) = seen[0]
    assert shape == result.x.shape and error == result.fun and degree == 3
    assert [np.asarray(c).tolist() for c in control_points] == [
        [list(p) for p in cp] for cp in result.section_control_points]
    assert len(knots) == 2


def test_optimizers_package_shares_the_evaluation_counter():
    from spline_fitter.optimization import optimizers, tracking
    tracking.reset_iteration_tracking()
    tracking.increment_evaluation()
    assert optimizers.get_evaluation_count() == 1
    optimizers.reset_iteration_tracking()
    assert tracking.get_evaluation_count() == 0


@pytest.mark.parametrize("library_call", ["fit", "fit_to_tolerance"])
def test_fits_have_no_evaluation_limit(monkeypatch, library_call):
    """Fits run until they converge: no limit reaches the library."""
    captured = {}
    original = getattr(bspline_varpro, library_call)

    def capture(*args, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(gn_varpro.bspline_varpro, library_call, capture)
    airfoil = small_airfoil(0.001)
    config = SplineConfig(degree=3, num_control_points=6)
    tolerance = 1e-2 if library_call == "fit_to_tolerance" else None
    GNVarProOptimizer().fit(config, OptimizationConfig(tolerance=tolerance), airfoil)
    assert "max_nfev" not in captured and "max_iterations" not in captured
    assert not {"max_nfev", "max_iterations"} & set(vars(GNVarProConfig()))
    assert not hasattr(OptimizationConfig(), "max_iterations")


def test_fixed_count_steps_cover_the_whole_fit(monkeypatch):
    """Growing the count and moving the knots are several solves; the steps
    count all of them, not only the last solve's evaluations."""
    last = {}
    original = bspline_varpro.fit

    def capture(*args, **kwargs):
        last["result"] = original(*args, **kwargs)
        return last["result"]

    monkeypatch.setattr(gn_varpro.bspline_varpro, "fit", capture)
    reset_iteration_tracking()
    result = GNVarProOptimizer().fit(
        SplineConfig(degree=3, num_control_points=8),
        OptimizationConfig(grow_control_points=True, knot_placement="free"),
        small_airfoil(0.002), callback=lambda x, e: False)
    assert result.nit == get_evaluation_count()
    assert result.nit > last["result"].nfev


def test_one_section_ignores_the_junction_continuity():
    """One section has no junction, so a continuity its degree cannot carry
    (left over from more sections) does not stop the fit."""
    config = SplineConfig(degree=2, num_control_points=5, num_sections=1,
                          continuity=ContinuityMode.G3)
    assert config.junction_specs() == []
    with pytest.raises(ValueError, match="G3 continuity requires degree"):
        SplineConfig(degree=2, num_control_points=5, num_sections=2, continuity=ContinuityMode.G3)
    airfoil = small_airfoil(0.001)
    single = AirfoilData(coordinates=airfoil.coordinates, sections=(airfoil.coordinates,),
                         junction_indices=(), num_sections=1)
    for optimizer in OptimizerType:
        result = run_optimization(config, OptimizationConfig(optimizer=optimizer), single)
        assert len(result.section_control_points) == 1


def test_per_junction_entries_free_the_degree_from_the_continuity_setting():
    """The Continuity setting governs no junction when every junction has its
    own entry, so a continuity the degree cannot carry does not stop the fit;
    the entries themselves still must fit the degree."""
    config = SplineConfig(degree=2, num_control_points=5, continuity=ContinuityMode.G3,
                          junction_continuity=("C1",))
    assert config.junction_specs() == ["C1"]
    with pytest.raises(ValueError, match="C3 continuity requires degree"):
        SplineConfig(degree=2, continuity=ContinuityMode.G1, junction_continuity=("C3",))
    result = run_optimization(config, OptimizationConfig(), small_airfoil(0.001))
    assert len(result.section_control_points) == 2


def test_two_sections_keep_two_points_each():
    """When the farthest point from the trailing edge is an end point (data
    that is no airfoil), the split still leaves both sections a segment, so
    VarPro LM fits instead of failing and Direct LSQ does not fit a
    one-point section to nothing."""
    from spline_fitter.geometry import create_airfoil_data
    for coords in ([(0.0, 0.0), (1.0, 0.1), (1.2, -0.1), (2.0, 0.0)],
                   [(2.0, 0.0), (1.2, -0.1), (1.0, 0.1), (0.0, 0.0)]):
        airfoil = create_airfoil_data(coords, 2)
        assert [len(s) for s in airfoil.sections] in ([2, 3], [3, 2])
        assert airfoil.sections[0][-1] == airfoil.sections[1][0]
        for optimizer in OptimizerType:
            result = run_optimization(SplineConfig(degree=2, num_control_points=3),
                                      OptimizationConfig(optimizer=optimizer), airfoil)
            controls = np.vstack(result.section_control_points)
            assert np.all(np.abs(controls) < 5), optimizer


def test_every_run_records_its_rms_error():
    """A tolerance goal's fun is its max deviation; the RMS error is recorded
    beside it, so its runs compare with the others."""
    airfoil = small_airfoil(0.001)
    config = SplineConfig(degree=3, num_control_points=6)
    tolerance = GNVarProOptimizer().fit(config, OptimizationConfig(tolerance=1e-3), airfoil)
    assert tolerance.rms_error < tolerance.fun
    fixed = GNVarProOptimizer().fit(config, OptimizationConfig(), airfoil)
    direct = run_optimization(config, OptimizationConfig(optimizer=OptimizerType.DIRECT_LSQ),
                              airfoil)
    assert fixed.rms_error == fixed.fun and direct.rms_error == direct.fun


def test_runs_compare_by_one_measure():
    from spline_fitter.optimization.models import FitResult
    result = FitResult([[(0.0, 0.0), (1.0, 0.0)]], 1, final_error=3e-4, num_iterations=1,
                       num_evaluations=1, elapsed_seconds=0.0)
    assert result.comparison_error == 3e-4  # recorded nothing else
    result.rms_error = 1e-4
    assert result.comparison_error == 1e-4
    result.report = {"max_deviation": 3e-4, "rms_deviation": 1.2e-4}
    assert result.comparison_error == 1.2e-4


def test_projects_with_an_evaluation_limit_still_load(tmp_path):
    """Projects from when fits had an evaluation limit store "max_iterations";
    it limits nothing now, so loading ignores it and saving leaves it out."""
    import json

    from spline_fitter.geometry import create_airfoil_data
    from spline_fitter.optimization.models import FitResult
    from spline_fitter.project_io import (
        create_optimization_run,
        create_project,
        load_project,
        save_project,
    )
    airfoil = small_airfoil()
    options = OptimizationConfig(smoothing_weight=0.02)
    project = create_project(create_airfoil_data(list(airfoil.coordinates), 2))
    project.optimization_runs.append(create_optimization_run(
        SplineConfig(degree=3, num_control_points=6), options,
        FitResult([list(s) for s in airfoil.sections], 3, 1e-4, 3, 9, 0.5), []))
    path = tmp_path / "old.splineproj"
    save_project(project, str(path))
    data = json.loads(path.read_text(encoding="utf-8"))

    def stored_options(node):
        if isinstance(node, dict):
            if "optimization_config" in node:
                yield node["optimization_config"]
            for value in node.values():
                yield from stored_options(value)
        elif isinstance(node, list):
            for value in node:
                yield from stored_options(value)

    found = list(stored_options(data))
    assert found and not any("max_iterations" in o for o in found)
    for o in found:
        o["max_iterations"] = 500
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_project(str(path)).optimization_runs[0].optimization_config == options
