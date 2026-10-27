"""Tolerance mode, geometric continuity, knots, report and CAD export in the app."""
import numpy as np
import pytest
from scipy.interpolate import BSpline

import bspline_varpro
from bspline_varpro import diagnostics
from bspline_varpro.airfoil import NACA4

from spline_fitter.analysis import default_tolerance, fit_report, format_report_html, scipy_curves
from spline_fitter.geometry import create_airfoil_data
from spline_fitter.io import export_spline_data, parse_spline_data
from spline_fitter.optimization.models import (
    ContinuityMode, FairnessWeights, FitResult, OptimizationConfig, OptimizerType, SplineConfig,
)
from spline_fitter.optimization.optimizers.gn_varpro import GNVarProOptimizer, accepts_preview


def naca_airfoil(code="2412", n=61, decimals=6, sections=2):
    pts = np.round(NACA4(code).points(n), decimals)
    return create_airfoil_data([tuple(p) for p in pts], filename=f"naca{code}", name=f"NACA {code}",
                               precision=decimals, num_sections=sections)


@pytest.fixture(scope="module")
def tolerance_fit():
    airfoil = naca_airfoil()
    result = GNVarProOptimizer().fit(
        SplineConfig(degree=5, num_control_points=10, continuity=ContinuityMode.G2,
                     geometric_continuity=True),
        OptimizationConfig(optimizer=OptimizerType.GN_VARPRO, tolerance=1e-5), airfoil)
    return airfoil, result


def test_tolerance_mode_returns_knots_within_tolerance(tolerance_fit):
    airfoil, result = tolerance_fit
    assert result.success
    assert result.fun <= 1e-5 * (1 + 1e-9)
    curves = [BSpline(np.array(k), np.array(c), 5)
              for k, c in zip(result.section_knots, result.section_control_points)]
    dev = diagnostics.max_deviation([np.array(s) for s in airfoil.sections], curves)
    assert dev <= 1e-5 * (1 + 1e-9)
    interior = [np.unique(np.array(k)[6:-6]) for k in result.section_knots]
    assert any(len(i) > 0 for i in interior)
    j = diagnostics.junctions(curves)[0]
    assert j["tangent_angle"] < 1e-6 and j["curvature_jump"] < 1e-3 * 60


def test_adapter_passes_per_junction_continuity(monkeypatch):
    from spline_fitter.optimization.optimizers import gn_varpro
    captured = {}
    original = bspline_varpro.fit_to_tolerance

    def capture(points, tolerance, degree, **kwargs):
        captured.update(kwargs, tolerance=tolerance, degree=degree)
        return original(points, tolerance, degree, **kwargs)

    monkeypatch.setattr(gn_varpro.bspline_varpro, "fit_to_tolerance", capture)
    airfoil = naca_airfoil(n=41, decimals=4, sections=3)
    config = SplineConfig(degree=4, num_control_points=8, num_sections=3,
                          junction_continuity=("g2", "C1"))
    result = GNVarProOptimizer().fit(
        config, OptimizationConfig(optimizer=OptimizerType.GN_VARPRO, tolerance=2e-4, fairing=False),
        airfoil)
    assert captured["continuity"] == ["G2", "C1"]
    assert captured["tolerance"] == 2e-4 and captured["degree"] == 4
    assert captured["fairing"] is False and captured["fix_junctions"] is True
    assert len(result.section_knots) == 3


def test_fixed_mode_geometric_continuity_uses_labels(monkeypatch):
    from spline_fitter.optimization.optimizers import gn_varpro
    captured = {}
    original = bspline_varpro.fit

    def capture(points, **kwargs):
        captured.update(kwargs)
        return original(points, **kwargs)

    monkeypatch.setattr(gn_varpro.bspline_varpro, "fit", capture)
    config = SplineConfig(degree=5, num_control_points=9, continuity=ContinuityMode.G2,
                          geometric_continuity=True)
    result = GNVarProOptimizer().fit(config, OptimizationConfig(max_iterations=200), naca_airfoil(n=31))
    assert captured["continuity"] == ["G2"]
    assert result.section_knots is not None and [len(k) for k in result.section_knots] == [15, 15]


def test_spline_config_validation():
    assert SplineConfig(num_sections=3, junction_continuity=(" g1", "c2 ")).junction_specs() == ["G1", "C2"]
    assert SplineConfig(num_sections=3, continuity=ContinuityMode.G2).junction_specs() == ["C2", "C2"]
    assert SplineConfig(geometric_continuity=True, continuity=ContinuityMode.G3).junction_specs() == ["G3"]
    with pytest.raises(ValueError, match="junction"):
        SplineConfig(num_sections=3, junction_continuity=("G2",))
    with pytest.raises(ValueError, match="Invalid"):
        SplineConfig(junction_continuity=("G5",))
    with pytest.raises(ValueError, match="degree"):
        SplineConfig(degree=2, junction_continuity=("G3",))
    with pytest.raises(ValueError, match="tolerance"):
        OptimizationConfig(tolerance=0.0)


def test_engine_dispatches_tolerance_mode():
    from spline_fitter.optimization.engine import run_optimization
    result = run_optimization(
        SplineConfig(degree=5, geometric_continuity=True, continuity=ContinuityMode.G2),
        OptimizationConfig(optimizer=OptimizerType.GN_VARPRO, tolerance=1e-4),
        naca_airfoil(n=41, decimals=5), FairnessWeights(), [])
    assert result.success and result.fun <= 1e-4 * (1 + 1e-9)


def test_preview_callbacks_get_curves_and_can_stop():
    seen = []

    def callback(x, error, preview=None):
        ctrl, knots, degree = preview
        assert degree == 5 and len(ctrl) == len(knots) == 2
        assert x.shape == (2 * sum(len(c) for c in ctrl),)
        seen.append(error)
        return len(seen) == 2

    assert accepts_preview(callback) and not accepts_preview(lambda x, e: False)
    result = GNVarProOptimizer().fit(
        SplineConfig(degree=5, geometric_continuity=True, continuity=ContinuityMode.G2),
        OptimizationConfig(tolerance=1e-7), naca_airfoil(), callback=callback)
    assert len(seen) == 2 and not result.success


def test_app_fit_result_curves_use_knots(tolerance_fit):
    airfoil, raw = tolerance_fit
    result = FitResult(raw.section_control_points, 5, raw.fun, raw.nit, raw.nit, 1.0,
                       section_knots=raw.section_knots)
    assert result.num_control_points == sum(len(c) for c in raw.section_control_points) - 1
    for curve, sc, knots, ctrl in zip(result.curves(), scipy_curves(result), raw.section_knots,
                                      raw.section_control_points):
        assert np.array_equal(curve.knot_vector, knots)
        u = np.linspace(0, 1, 17)
        x, y = curve.evaluate_array(u)
        assert np.allclose(np.column_stack([x, y]), sc(u))
    uniform = FitResult([[(0, 0), (1, 1), (2, 0), (3, 1)]], 3, 0.0, 1, 1, 0.0)
    assert list(uniform.knots(0)) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_report(tolerance_fit):
    airfoil, raw = tolerance_fit
    result = FitResult(raw.section_control_points, 5, raw.fun, raw.nit, raw.nit, 1.0,
                       section_knots=raw.section_knots)
    report = fit_report(airfoil.sections, result, precision=6, tolerance=1e-5)
    assert report["max_deviation"] == pytest.approx(raw.fun, rel=1e-6)
    assert report["control_points"] == result.num_control_points
    assert report["leading_edge_radius"] == pytest.approx(1 / NACA4("2412").leading_edge_curvature(), rel=0.03)
    assert len(report["junctions"]) == 1 and report["curvature_extrema"] >= 0
    page = format_report_html(report, "NACA <2412>")
    assert "NACA &lt;2412&gt;" in page and "met" in page
    assert default_tolerance(airfoil.coordinates, 6) == pytest.approx(1e-6)


def test_spline_file_round_trip(tmp_path, tolerance_fit):
    _, raw = tolerance_fit
    path = tmp_path / "a.deg5.spline.dat"
    export_spline_data(path, 5, raw.section_control_points, name="NACA 2412", precision=15,
                       section_knots=raw.section_knots, continuity=["G2"])
    data = parse_spline_data(path)
    assert data["name"] == "NACA 2412" and data["degree"] == 5 and data["continuity"] == ["G2"]
    assert [len(c) for c in data["section_control_points"]] == [len(c) for c in raw.section_control_points]
    for a, b in zip(data["section_control_points"], raw.section_control_points):
        assert np.allclose(a, b, atol=1e-14)
    assert data["section_knots"] == [list(map(float, k)) for k in raw.section_knots]
    from spline_fitter.io import parse_airfoil_data
    _, _, coords, _ = parse_airfoil_data(str(path))
    assert len(coords) == sum(len(c) for c in raw.section_control_points) - 1


def test_old_spline_files_still_parse(tmp_path):
    path = tmp_path / "old.deg3.spline.dat"
    path.write_text("NACA 0012\n1.0 0.0\n0.5 0.1\n0.0 0.0\n0.5 -0.1\n1.0 0.0\n")
    data = parse_spline_data(path)
    assert data["degree"] == 3 and data["section_knots"] is None
    assert len(data["section_control_points"][0]) == 5
    with pytest.raises(ValueError, match="Knot vector length"):
        export_spline_data(tmp_path / "bad.dat", 3, [[(0, 0), (1, 1), (2, 0), (3, 1)]],
                           section_knots=[[0, 0, 0, 1, 1, 1]])


def test_project_round_trip_keeps_knots_and_settings(tmp_path, tolerance_fit):
    from spline_fitter.project_io import create_optimization_run, create_project, load_project, save_project
    airfoil, raw = tolerance_fit
    result = FitResult(raw.section_control_points, 5, raw.fun, raw.nit, raw.nit, 1.0,
                       section_knots=raw.section_knots)
    result.report = fit_report(airfoil.sections, result, precision=6, tolerance=1e-5)
    config = SplineConfig(num_sections=2, geometric_continuity=True, junction_continuity=("G3",))
    options = OptimizationConfig(optimizer=OptimizerType.GN_VARPRO, tolerance=1e-5, fairing=False)
    project = create_project(airfoil)
    project.optimization_runs.append(create_optimization_run(config, options, FairnessWeights(), result, []))
    save_project(project, str(tmp_path / "p.splineproj"))
    run = load_project(str(tmp_path / "p.splineproj")).optimization_runs[0]
    assert run.spline_config == config and run.optimization_config == options
    assert run.result.section_knots == result.section_knots
    assert run.result.report["max_deviation"] == result.report["max_deviation"]
    from spline_fitter.gui.project import ProjectManager
    x, y, label = ProjectManager._detect_sweep_axis([run])
    assert label == "Control Points" and x == [result.num_control_points] and y == [raw.fun]


def _window(monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from spline_fitter.gui.ui import SplineFitterWindow
    from spline_fitter.gui import fitting
    app = QApplication.instance() or QApplication([])
    window = SplineFitterWindow()
    monkeypatch.setattr(fitting.OptimizerErrorLogger, "save", lambda *a, **k: None)
    monkeypatch.setattr(window, "_play_completion_chime", lambda: None)
    return app, window


def test_gui_tolerance_mode_fit_report_and_cad_export(monkeypatch, tmp_path):
    app, window = _window(monkeypatch)
    window._airfoil = naca_airfoil(decimals=5)
    window._combo_opt.setCurrentIndex(window._combo_opt.findData(OptimizerType.GN_VARPRO))
    assert window._check_tolerance_mode.isChecked() and window._check_auto_tolerance.isChecked()
    assert not window._sweep_ctrl.isEnabled()
    window._slider_live.setValue(2)
    window._check_knots.setChecked(True)
    window._fitting_controller.on_fit()
    result = window._fit_result
    assert result is not None and result.converged
    assert window._spin_tolerance.value() == pytest.approx(1e-5)
    assert result.section_knots is not None
    assert result.report["max_deviation"] <= 1e-5 * (1 + 1e-9)
    assert "max dev" in window.statusBar().currentMessage()
    assert "knots" in window._plot_manager._knot_items
    assert window._action_export_cad.isEnabled() and window._action_fit_report.isEnabled()
    run = window._project_manager._project.optimization_runs[-1]
    assert run.optimization_config.tolerance == pytest.approx(1e-5)
    assert run.spline_config.junction_specs() == ["G2"]
    for suffix, marker in [(".step", "B_SPLINE_CURVE_WITH_KNOTS"), (".igs", "126,"), (".dxf", "SPLINE")]:
        path = window.export_cad(str(tmp_path / f"a{suffix}"), units="mm", scale=100.0, merge=True)
        assert marker in open(path).read()
    window._edit_junctions.setText("G5")
    window._fitting_controller.on_fit()
    assert "Invalid junction continuity" in window.statusBar().currentMessage()
    window.deleteLater()
    app.processEvents()


def test_gui_dialogs_save_export_and_report(monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QDialog, QFileDialog, QInputDialog
    from bspline_varpro.airfoil import read_dat
    app, window = _window(monkeypatch)
    window._airfoil = naca_airfoil(n=41, decimals=5)
    window._combo_opt.setCurrentIndex(window._combo_opt.findData(OptimizerType.GN_VARPRO))
    window._fitting_controller.on_fit()
    assert window._fit_result is not None
    window._fit_result.report = None

    shown = []
    monkeypatch.setattr(QDialog, "exec", lambda self: shown.append(self.windowTitle())
                        or QDialog.DialogCode.Accepted)
    monkeypatch.setattr(QInputDialog, "getInt", staticmethod(lambda *a, **k: (50, True)))
    target = {}
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / target["name"]), "")))

    target["name"] = "fitted.dat"
    window._on_save_fitted()
    assert "Saved" in window.statusBar().currentMessage()
    name, points = read_dat(tmp_path / "fitted.dat")
    assert name == "NACA 2412" and len(points) == 99

    target["name"] = "a.deg5.spline.dat"
    window._on_save_spline()
    data = parse_spline_data(tmp_path / "a.deg5.spline.dat")
    assert data["continuity"] == ["G2"] and data["section_knots"] == [
        list(map(float, k)) for k in window._fit_result.section_knots]

    target["name"] = "a.step"
    window._on_export_cad()
    assert "Exported" in window.statusBar().currentMessage()
    assert "B_SPLINE_CURVE_WITH_KNOTS" in (tmp_path / "a.step").read_text()

    window._on_fit_report()
    assert shown[-1] == "Fit Report" and window._fit_result.report is not None
    window.deleteLater()
    app.processEvents()
