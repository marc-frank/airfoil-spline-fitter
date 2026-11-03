"""The plot manager: comb scale and sides, and the convergence plot."""
import os

import numpy as np
import pytest
from bspline_varpro import airfoil

from spline_fitter.spline import BSplineCurve2D

# The application, kept for the rest of the session: one that is collected
# with the fixture takes Qt objects of later tests with it
_APP = []


@pytest.fixture
def manager():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui.plotting import PlotManager

    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    plots = [pg.PlotWidget() for _ in range(3)]
    yield PlotManager(*plots)
    for plot in plots:
        plot.deleteLater()
    _APP[0].processEvents()


def _airfoil_curves(sections: int, chord: float = 1.0) -> list:
    """A NACA 0012 as cubic curves through its points, in ``sections`` pieces
    that share their end points (Selig order: upper trailing edge first)."""
    points = airfoil.NACA4("0012").points(41) * chord
    bounds = np.linspace(0, len(points) - 1, sections + 1).round().astype(int)
    return [BSplineCurve2D(3, [tuple(p) for p in points[a:b + 1]])
            for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


@pytest.mark.parametrize("quantity", [0, 1, 2, 3])
def test_comb_looks_the_same_at_any_size_of_the_data(quantity):
    from spline_fitter.gui.plotting import PlotManager

    reach = []
    for chord in (1.0, 100.0):
        curve = _airfoil_curves(1, chord)[0]
        base, tip = PlotManager._curvature_comb(curve, 0.05, 80, quantity, chord)
        reach.append(np.hypot(*(tip - base).T) / chord)
    np.testing.assert_allclose(reach[1], reach[0], rtol=1e-6, atol=1e-12 * np.max(reach[0]))


@pytest.mark.parametrize("sections", [1, 2, 3, 4])
def test_comb_sides_split_at_the_leading_edge(manager, sections):
    """Upper comb on the upper surface and lower on the lower, as the
    deviation is split, whatever the section count."""
    manager._draw_combs(_airfoil_curves(sections), 0.05, 60, 0)
    for side, sign in (("upper", 1), ("lower", -1)):
        x, y = manager._comb_items[side, True].getData()
        assert x is not None and len(x) > 10
        bx, by = (a[0::3] for a in manager._comb_items[side, False].getData())
        # Away from the leading edge, whose tooth both sides share, and the
        # closed trailing edge, on the chord
        away = np.isfinite(bx) & (np.hypot(bx, by) > 0.01) & (bx < 0.99)
        assert np.all(sign * by[away] > 0), (side, by[away].min(), by[away].max())


def test_a_new_airfoil_clears_the_convergence(manager):
    manager.update_loss_data([1, 2, 3], [0.1, 0.2, 0.3], [1e-1, 1e-2, 1e-3])
    manager.set_loss_curve_data("iters")
    manager.plot_airfoil([(1, 0), (0, 0), (1, 0.01)])
    x, _ = manager._loss_curve.getData()
    assert x is None or len(x) == 0
    assert not manager.has_loss_data


@pytest.mark.parametrize("n", [10, 150_000, 250_001])
def test_convergence_keeps_at_most_the_cap_and_the_latest(manager, n, monkeypatch):
    from spline_fitter.gui import plotting

    monkeypatch.setattr(plotting, "MAX_PLOT_POINTS", 100)
    manager.update_loss_data(list(range(1, n + 1)), list(range(n)), [1.0] * n)
    manager.set_loss_curve_data("iters")
    x, _ = manager._loss_curve.getData()
    assert len(x) == min(n, 100)
    assert x[0] == 1 and x[-1] == n


def test_switching_the_convergence_axes_ranges_anew(manager):
    view = manager._loss_plot.getPlotItem().getViewBox()
    manager.update_loss_data([1, 2, 3], [0.1, 0.2, 0.3], [1e-1, 1e-2, 1e-3])
    manager.set_loss_curve_data("iters")
    view.scaleBy((0.5, 0.5))  # as a wheel zoom does: auto range off
    assert view.autoRangeEnabled() == [False, False]
    manager.set_loss_x_label("Time (s)")
    assert view.autoRangeEnabled() == [True, True]
    view.scaleBy((0.5, 0.5))
    manager.set_loss_log_scale(True, True)
    assert view.autoRangeEnabled() == [True, True]


def test_convergence_empties_without_data(manager):
    """A run without history (or no run) shows no trace, not the last run's."""
    manager.update_loss_data([1, 2, 3], [0.1, 0.2, 0.3], [1e-1, 1e-2, 1e-3])
    manager.set_loss_curve_data("iters")
    manager.update_loss_data([], [], [])
    manager.set_loss_curve_data("iters")
    x, _ = manager._loss_curve.getData()
    assert x is None or len(x) == 0


def test_a_fit_that_fails_to_draw_leaves_no_curve(manager):
    from spline_fitter.optimization.models import FitResult

    curves = _airfoil_curves(1)
    manager.plot_fitted_spline(FitResult([curves[0].control_points], 3, 0.0, 0, 0, 0.0))
    assert len(manager._curve_item.getData()[0]) > 0

    class Broken(FitResult):
        def curves(self):
            raise ValueError("no curve")

    manager.plot_fitted_spline(Broken([], 3, 0.0, 0, 0, 0.0))
    for item in (manager._curve_item, manager._polygon_item):
        x, _ = item.getData()
        assert x is None or len(x) == 0


def test_shape_chart_follows_a_theme_change(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui import theme
    from spline_fitter.gui.plotting import shape_plot

    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(theme, "_settings", lambda: QSettings(ini, QSettings.Format.IniFormat))
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    theme.apply_theme(_APP[0])
    before = (theme.Scheme.MODE, theme.Scheme.TEMPERATURE)
    x = np.linspace(0, 1, 20)
    plot = shape_plot({"x": x, "thickness": 0.1 * x, "camber": 0.02 * x,
                       "max_thickness": 0.1, "max_thickness_x": 1.0,
                       "max_camber": 0.02, "max_camber_x": 1.0})
    try:
        for mode in ("light", "dark"):
            theme.set_theme(mode, 40)
            assert plot.backgroundBrush().color().name().lower() == theme.Palette.CANVAS.lower()
            axis = plot.getPlotItem().getAxis("bottom")
            assert axis.textPen().color().name().lower() == theme.Palette.TEXT_2.lower()
            assert axis.labelStyle["color"].lower() == theme.Palette.TEXT_2.lower()
            line, peak = plot._items["thickness"]
            trace = theme.brush(theme.Ink.TRACE).color()
            assert line.opts["pen"].color() == trace
            assert peak.opts["symbolBrush"].color() == trace
            assert peak.opts["symbolPen"].color().name().lower() == theme.Palette.CANVAS.lower()
            camber, _ = plot._items["camber"]
            assert camber.opts["pen"].color() == theme.brush(theme.Ink.CONSTRUCTION).color()
            label = plot._legend.items[0][1]
            assert label.opts["color"].name().lower() == theme.Palette.TEXT_2.lower()
    finally:
        theme.set_theme(*before)
        plot.deleteLater()
        _APP[0].processEvents()


def test_a_deleted_shape_chart_leaves_no_theme_connection():
    """A closed report's chart does not stay connected to theme changes."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QCoreApplication, QEvent
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui.plotting import shape_plot
    from spline_fitter.gui.theme import theme_changed

    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    receivers = theme_changed.receivers(theme_changed.changed)
    x = np.linspace(0, 1, 20)
    for _ in range(3):
        plot = shape_plot({"x": x, "thickness": 0.1 * x, "camber": 0.02 * x,
                           "max_thickness": 0.1, "max_thickness_x": 1.0,
                           "max_camber": 0.02, "max_camber_x": 1.0})
        assert theme_changed.receivers(theme_changed.changed) == receivers + 1
        plot.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert theme_changed.receivers(theme_changed.changed) == receivers
