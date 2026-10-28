"""App integration of the bspline-varpro library."""
import bspline_varpro
import numpy as np
import pytest
from scipy.interpolate import BSpline

from spline_fitter.optimization.models import (
    AirfoilData,
    ContinuityMode,
    FairnessWeights,
    OptimizationConfig,
    OptimizerType,
    SplineConfig,
)
from spline_fitter.optimization.optimizers.gn_varpro import GNVarProConfig, GNVarProOptimizer


def small_airfoil(noise=0.0, n=19):
    t = np.linspace(0, np.pi, n)
    upper = np.column_stack(((1 + np.cos(t)) / 2, 0.09 * np.sin(t)))
    lower = np.column_stack(((1 - np.cos(t)) / 2, -0.06 * np.sin(t)))
    rng = np.random.default_rng(82)
    for sec in [upper, lower]:
        sec[1:-1, 1] += rng.normal(0, noise, len(sec) - 2)
    sections = tuple(tuple(map(tuple, sec)) for sec in [upper, lower])
    return AirfoilData(sections[0] + sections[1][1:], sections, (n - 1,), 2)


def test_adapter_returns_app_result_with_data_only_error():
    airfoil = small_airfoil(0.003)
    config = SplineConfig(degree=3, num_control_points=6)
    result = GNVarProOptimizer(GNVarProConfig(smoothing_weight=0.1)).fit(
        config, OptimizationConfig(max_iterations=500), airfoil)
    assert result.success
    assert [len(cp) for cp in result.section_control_points] == [6, 6]
    assert result.x.shape == (2 * 6 * 2,)
    error = []
    for sec, cp, t, knots in zip(airfoil.sections, result.section_control_points, result.parameters,
                                 result.section_knots, strict=False):
        curve = BSpline(knots, cp, 3)
        error.extend(np.sum((curve(t) - sec) ** 2, axis=1))
    assert result.fun == pytest.approx(np.sqrt(np.mean(error)))


def test_adapter_passes_settings_through_to_library(monkeypatch):
    from spline_fitter.optimization.models import ContinuityMode
    from spline_fitter.optimization.optimizers import gn_varpro
    captured = {}
    original_fit = bspline_varpro.fit

    def capture(points, **kwargs):
        captured.update(kwargs, points=points)
        return original_fit(points, **kwargs)

    monkeypatch.setattr(gn_varpro.bspline_varpro, "fit", capture)
    airfoil = small_airfoil(0.002)
    t0 = [np.linspace(0, 1, 19)] * 2
    GNVarProOptimizer(GNVarProConfig(smoothing_weight=1e-4, regularization=1e-11, tolerance=1e-9,
                                     use_exact_jacobian=False)).fit(
        SplineConfig(degree=5, num_control_points=10, continuity=ContinuityMode.G2),
        OptimizationConfig(max_iterations=300, optimize_le_position=True, custom_parameters=t0),
        airfoil)
    assert [np.asarray(s).tolist() for s in airfoil.sections] == [p.tolist() for p in captured["points"]]
    assert captured["num_control_points"] == 10 and captured["degree"] == 5
    assert captured["continuity"] == "C2" and captured["fix_junctions"] is False
    assert captured["smoothing"] == 1e-4 and captured["regularization"] == 1e-11
    assert captured["tolerance"] == 1e-9 and captured["exact_jacobian"] is False
    assert captured["max_nfev"] == 300 and captured["initial_parameters"] is t0
    GNVarProOptimizer(GNVarProConfig(max_iterations=7)).fit(
        SplineConfig(degree=3, num_control_points=6), OptimizationConfig(max_iterations=0), airfoil)
    assert captured["max_nfev"] == 7 and captured["fix_junctions"] is True


def test_single_section_fits_in_both_modes():
    """One section has no junctions, so no per-junction continuity list."""
    airfoil = small_airfoil(0.001)
    single = AirfoilData(coordinates=airfoil.coordinates, sections=(airfoil.coordinates,),
                         junction_indices=(), num_sections=1)
    config = SplineConfig(degree=5, num_control_points=10, num_sections=1,
                          continuity=ContinuityMode.G3, geometric_continuity=True)
    fixed = GNVarProOptimizer().fit(config, OptimizationConfig(max_iterations=300), single)
    assert fixed.section_control_points is not None and len(fixed.section_control_points) == 1
    tolerance = GNVarProOptimizer().fit(
        config, OptimizationConfig(max_iterations=300, tolerance=1e-3), single)
    assert tolerance.section_control_points is not None and len(tolerance.section_control_points) == 1


def test_callback_gets_flat_control_points_counts_evaluations_and_can_stop():
    from spline_fitter.optimization.engine import get_evaluation_count, reset_iteration_tracking
    reset_iteration_tracking()
    seen = []

    def callback(x, error):
        seen.append(x.shape)
        return len(seen) == 4

    result = GNVarProOptimizer().fit(SplineConfig(degree=3, num_control_points=8),
                                     OptimizationConfig(max_iterations=300, knot_placement="uniform",
                                                        grow_control_points=False),
                                     small_airfoil(0.002), callback=callback)
    assert seen == [(32,)] * 4
    assert get_evaluation_count() == 4
    assert not result.success and result.nit == 4


def test_engine_uses_configured_smoothing_weight():
    from spline_fitter.optimization.engine import run_optimization
    airfoil = small_airfoil(0.003)
    config = SplineConfig(degree=3, num_control_points=8)
    results = [run_optimization(config, OptimizationConfig(optimizer=OptimizerType.GN_VARPRO,
                                                           max_iterations=300, smoothing_weight=w,
                                                           knot_placement="uniform",
                                                           grow_control_points=False),
                                airfoil, FairnessWeights(), [])
               for w in (0.0, 1.0)]
    assert all(r.success for r in results)
    assert results[0].fun < results[1].fun


def test_retired_hierarchical_runs_load_as_varpro():
    from spline_fitter.project_io import _decode_optimization_config
    assert not hasattr(OptimizerType, "HIERARCHICAL_LSQ")
    options = _decode_optimization_config({"optimizer": "hierarchical-lsq", "metric": "avg_point"})
    assert options.optimizer == OptimizerType.GN_VARPRO


def test_projects_saved_with_fairing_still_load():
    from spline_fitter.project_io import _decode_optimization_config, _encode_optimization_config
    for fairing in (True, False):
        options = _decode_optimization_config({
            "optimizer": "gn-varpro", "metric": "avg_point", "tolerance": 1e-5, "fairing": fairing,
        })
        assert options.optimizer == OptimizerType.GN_VARPRO and options.tolerance == 1e-5
        assert "fairing" not in _encode_optimization_config(options)


def test_subsampling_setting_round_trips_through_projects():
    from spline_fitter.project_io import _decode_optimization_config, _encode_optimization_config
    options = OptimizationConfig(optimizer=OptimizerType.GN_VARPRO, max_points_per_section=150)
    assert _decode_optimization_config(_encode_optimization_config(options)) == options
    old = _decode_optimization_config({"optimizer": "gn-varpro", "metric": "avg_point"})
    assert old.max_points_per_section is None
    with pytest.raises(ValueError, match="at least 2"):
        OptimizationConfig(max_points_per_section=1)


def test_knot_settings_round_trip_and_old_projects_keep_their_behaviour():
    from spline_fitter.project_io import _decode_optimization_config, _encode_optimization_config
    options = OptimizationConfig(optimizer=OptimizerType.GN_VARPRO, knot_placement="uniform",
                                 grow_control_points=False)
    assert _decode_optimization_config(_encode_optimization_config(options)) == options
    fixed = _decode_optimization_config({"optimizer": "gn-varpro", "metric": "avg_point"})
    assert fixed.knot_placement == "uniform" and not fixed.grow_control_points
    tolerance = _decode_optimization_config({"optimizer": "gn-varpro", "metric": "avg_point",
                                             "tolerance": 1e-5})
    assert tolerance.knot_placement == "free" and tolerance.grow_control_points
    with pytest.raises(ValueError, match="knot_placement"):
        OptimizationConfig(knot_placement="auto")



def test_projects_saved_with_automatic_selection_still_load():
    from spline_fitter.project_io import (
        _decode_fit_result,
        _decode_optimization_config,
        _encode_optimization_config,
    )
    options = _decode_optimization_config({
        "optimizer": "gn-varpro", "metric": "avg_point", "smoothing_weight": 3e-5,
        "auto_select": True,
    })
    assert options.optimizer == OptimizerType.GN_VARPRO
    assert options.smoothing_weight == 3e-5
    assert _decode_optimization_config(_encode_optimization_config(options)) == options
    result = _decode_fit_result({
        "degree": 5, "final_error": 1e-6, "num_iterations": 3, "num_evaluations": 3,
        "elapsed_seconds": 1.0, "model_selection": {"candidates": []},
    }, [[(0.0, 0.0)]])
    assert result.final_error == 1e-6


def test_gui_fits_with_varpro(monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui import fitting
    from spline_fitter.gui.ui import SplineFitterWindow
    app = QApplication.instance() or QApplication([])
    window = SplineFitterWindow()
    monkeypatch.setattr(fitting.OptimizerErrorLogger, "save", lambda *a, **k: None)
    monkeypatch.setattr(window, "_play_completion_chime", lambda: None)
    window._airfoil = small_airfoil(0.001)
    window._spin_degree.setValue(3)
    window._spin_ctrl.setValue(8)
    window._combo_opt.setCurrentIndex(window._combo_opt.findData(OptimizerType.GN_VARPRO))
    window._seg_goal.setCurrentIndex(window._seg_goal.findData(False))
    window._combo_cont.setCurrentIndex(window._combo_cont.findData(ContinuityMode.G2))
    window._spin_maxiter.setValue(300)
    window._seg_live.setCurrentIndex(2)
    assert window._sweep_ctrl.isEnabled()
    assert window._sweep_smoothing_weight.isVisible() or not window.isVisible()
    window._fitting_controller.on_fit()
    assert window._fit_result is not None and window._fit_result.converged
    assert [len(cp) for cp in window._fit_result.section_control_points] == [8, 8]
    assert "VarPro LM" in window.statusBar().currentMessage()
    assert not window._is_fitting
    window.deleteLater()
    app.processEvents()
