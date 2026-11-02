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


def test_tolerance_mode_passes_the_evaluation_limit(monkeypatch):
    captured = {}
    original = bspline_varpro.fit_to_tolerance

    def capture(*args, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(gn_varpro.bspline_varpro, "fit_to_tolerance", capture)
    airfoil = small_airfoil(0.001)
    config = SplineConfig(degree=3, num_control_points=6)
    GNVarProOptimizer().fit(config, OptimizationConfig(tolerance=1e-2, max_iterations=0), airfoil)
    assert "max_nfev" not in captured  # the library's own per-refit limit
    GNVarProOptimizer().fit(config, OptimizationConfig(tolerance=1e-2, max_iterations=50), airfoil)
    assert captured["max_nfev"] == 50
    GNVarProOptimizer(GNVarProConfig(max_nfev=9)).fit(
        config, OptimizationConfig(tolerance=1e-2, max_iterations=50), airfoil)
    assert captured["max_nfev"] == 9


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
        OptimizationConfig(max_iterations=300, grow_control_points=True, knot_placement="free"),
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
        result = run_optimization(config, OptimizationConfig(optimizer=optimizer,
                                                             max_iterations=300), single)
        assert len(result.section_control_points) == 1


@pytest.mark.parametrize("t0", [0.45, 0.2])
@pytest.mark.parametrize("method", ["newton", "robust"])
def test_footpoint_never_climbs_to_a_distance_maximum(t0, method):
    """A point past the centre of curvature: full Newton steps head for the
    farthest curve point, so the projection must only take steps that bring
    the point closer."""
    from spline_fitter.optimization.footpoint import (
        iterative_footpoint_projection,
        robust_footpoint_projection,
    )
    knots = create_open_uniform_knots(2, 3)
    spline_x = BSpline(knots, np.array([0.0, 0.5, 1.0]), 2)
    spline_y = BSpline(knots, np.array([0.0, 1.0, 0.0]), 2)
    point = np.array([[0.5, -0.6]])
    project = iterative_footpoint_projection if method == "newton" else robust_footpoint_projection
    t, distance = project(np.array([t0]), point, spline_x, spline_y, max_iter=50)

    ts = np.linspace(0, 1, 100001)
    closest = np.hypot(spline_x(ts) - 0.5, spline_y(ts) + 0.6).min()
    assert distance[0] == pytest.approx(closest, abs=1e-9)
    assert t[0] == pytest.approx(0.0, abs=1e-6)
