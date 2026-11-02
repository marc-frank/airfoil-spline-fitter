"""Thickness and camber along the chord, in the fit report."""
import os

import numpy as np
import pytest
from bspline_varpro import airfoil
from scipy.interpolate import BSpline

from spline_fitter.analysis import airfoil_shape, fit_report, format_report_html
from spline_fitter.geometry import create_airfoil_data
from spline_fitter.optimization.models import FitResult


def _fit(points: np.ndarray) -> FitResult:
    result = airfoil.fit_airfoil(points)
    return FitResult([[tuple(p) for p in c] for c in result.control_points], 5, 0.0, 0, 0, 0.0,
                     section_knots=[list(k) for k in result.knots])


def _curves(fit: FitResult) -> list[BSpline]:
    return [BSpline(np.asarray(fit.knots(i)), np.asarray(c), fit.degree)
            for i, c in enumerate(fit.section_control_points)]


@pytest.fixture(scope="module")
def cambered():
    """A shape whose thickness and camber at each chord station are known:
    the NACA 0012 thickness about a parabolic camber line of 3% at mid-chord,
    with both surfaces given at the same stations."""
    x = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, 121)))
    half = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                  + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    camber = 0.12 * x * (1 - x)
    upper = np.column_stack([x, camber + half])[::-1]
    lower = np.column_stack([x, camber - half])[1:]
    xs = np.linspace(0.0, 1.0, 200001)
    halfs = 0.6 * (0.2969 * np.sqrt(xs) - 0.1260 * xs - 0.3516 * xs ** 2
                   + 0.2843 * xs ** 3 - 0.1036 * xs ** 4)
    j = int(np.argmax(halfs))
    return _fit(np.round(np.vstack([upper, lower]), 7)), 2 * halfs[j], xs[j]


def test_known_thickness_and_camber(cambered):
    fit, thickness, thickness_x = cambered
    shape = airfoil_shape(fit)
    assert shape["chord"] == pytest.approx(1.0, abs=1e-9)
    assert shape["max_thickness"] == pytest.approx(thickness, abs=2e-6)
    assert shape["max_thickness_x"] == pytest.approx(thickness_x, abs=2e-3)
    # Within the fit's own deviation from the data
    assert shape["max_camber"] == pytest.approx(0.03, abs=1e-5)
    assert shape["max_camber_x"] == pytest.approx(0.5, abs=2e-3)
    x = shape["x"]
    np.testing.assert_allclose(shape["camber"], 0.12 * x * (1 - x), atol=2e-5)
    assert x[0] == 0.0 and x[-1] == 1.0 and np.all(np.diff(x) > 0)


def _moved(fit: FitResult, turn: np.ndarray, shift: np.ndarray) -> FitResult:
    return FitResult([[tuple(turn @ p + shift) for p in np.asarray(c)]
                      for c in fit.section_control_points], 5, 0.0, 0, 0, 0.0,
                     section_knots=fit.section_knots)


def test_measured_in_chords_whatever_the_scale_and_position(cambered):
    fit, *_ = cambered
    a = airfoil_shape(fit)
    b = airfoil_shape(_moved(fit, 250.0 * np.eye(2), np.array([40.0, -3.0])))
    assert b["chord"] == pytest.approx(250.0, rel=1e-9)
    for key in ("max_thickness", "max_thickness_x", "max_camber", "max_camber_x"):
        assert b[key] == pytest.approx(a[key], abs=1e-9)


def test_the_x_axis_is_the_chord_line(cambered):
    """A turned airfoil is measured as it lies, not turned back: with the
    trailing edge 2 degrees down, the camber line, measured from the
    trailing edge, rises by about half a chord times tan(2 degrees) in the
    middle."""
    fit, *_ = cambered
    angle = np.radians(-2.0)
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    level, turned = airfoil_shape(fit), airfoil_shape(_moved(fit, turn, np.zeros(2)))
    assert turned["max_camber"] > level["max_camber"] + 0.01


@pytest.mark.parametrize("code, camber", [("2412", 0.02), ("4415", 0.04)])
def test_naca_camber_is_the_nominal_one(code, camber):
    """Measured at the same x on both surfaces, the camber of a NACA section
    is its nominal maximum (NACA lays the thickness normal to the camber
    line, which moves the maximum a little aft of 40% chord)."""
    fit = _fit(np.round(airfoil.NACA4(code).points(121), 6))
    shape = airfoil_shape(fit)
    assert shape["max_camber"] == pytest.approx(camber, abs=5e-5)
    assert shape["max_camber_x"] == pytest.approx(0.40, abs=0.01)


def test_naca_0012_thickness_is_exact():
    fit = _fit(np.round(airfoil.NACA4("0012").points(121), 6))
    shape = airfoil.thickness_and_camber(_curves(fit))
    sigma = np.linspace(0.0, 1.0, 200001)
    exact = 2 * airfoil.NACA4("0012")._thickness(sigma)
    assert shape["max_thickness"] == pytest.approx(exact.max(), abs=1e-6)
    assert shape["max_thickness_x"] == pytest.approx(sigma[np.argmax(exact)] ** 2, abs=2e-3)
    assert abs(shape["max_camber"]) < 1e-6


def test_report_has_thickness_and_camber(cambered):
    fit, thickness, _ = cambered
    points = np.vstack([np.asarray(c) for c in fit.section_control_points])
    report = fit_report([points[:3], points[2:5]], fit)
    assert report["max_thickness"] == pytest.approx(thickness, abs=2e-6)
    page = format_report_html(report)
    assert "Max thickness" in page and "Max camber" in page
    assert "% chord" in page


def test_no_thickness_for_a_chain_that_is_not_an_airfoil():
    line = FitResult([[(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]], 2, 0.0, 0, 0, 0.0)
    assert airfoil_shape(line) is None
    page = format_report_html(fit_report([[(0.0, 0.0), (1.0, 0.0)]], line))
    assert "Max thickness" not in page


@pytest.fixture
def window(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui import fitting
    from spline_fitter.gui.ui import SplineFitterWindow
    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    monkeypatch.setattr(fitting.OptimizerErrorLogger, "save", lambda *a, **k: None)
    monkeypatch.setattr(w, "_play_completion_chime", lambda: None)
    coords = [tuple(p) for p in np.round(airfoil.NACA4("2412").points(61), 5)]
    w._airfoil = create_airfoil_data(coords, filename="n", name="n", precision=5, num_sections=2)
    w._plot_manager.plot_airfoil(coords)
    yield w
    w.deleteLater()
    app.processEvents()


def test_report_window_shows_the_chart_and_fills_in_older_reports(window, monkeypatch):
    from spline_fitter.gui import file_actions
    from spline_fitter.gui.theme import FramedPlot
    shown = {}
    monkeypatch.setattr(file_actions, "show_document",
                        lambda parent, title, html, w, h, extra=None: shown.update(html=html,
                                                                                   extra=extra))
    window._fitting_controller.on_fit()
    result = window._fit_result
    # A report saved before thickness and camber were measured
    for key in ("max_thickness", "max_thickness_x", "max_camber", "max_camber_x"):
        result.report.pop(key)
    window._on_fit_report()
    assert result.report["max_camber"] == pytest.approx(0.02, abs=5e-5)
    assert "Max camber" in shown["html"]
    assert isinstance(shown["extra"], FramedPlot)
    items = shown["extra"].getPlotItem().listDataItems()
    assert len(items) == 4  # thickness, camber and their maxima
