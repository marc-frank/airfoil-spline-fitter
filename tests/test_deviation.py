"""Deviation: signed distance of the data from the curve, magnified on the airfoil."""
import os

import numpy as np
import pytest
from bspline_varpro import airfoil, diagnostics

from spline_fitter.analysis import foot_points, scipy_curves, signed_deviations
from spline_fitter.geometry import create_airfoil_data
from spline_fitter.optimization.models import FitResult


@pytest.fixture(scope="module")
def fit():
    points = np.round(airfoil.NACA4("2412").points(81), 4)
    result = airfoil.fit_airfoil(points)
    fit = FitResult([[tuple(p) for p in c] for c in result.control_points], 5, 0.0, 0, 0, 0.0,
                    section_knots=[list(k) for k in result.knots])
    return points, fit


def test_deviation_size_is_the_orthogonal_distance(fit):
    points, result = fit
    distance = diagnostics.distance_to_curves(points, scipy_curves(result))
    np.testing.assert_allclose(np.abs(signed_deviations(points, result)), distance, atol=1e-15)


def test_point_is_its_foot_plus_the_deviation_along_the_normal(fit):
    points, result = fit
    dev, foot, normal, _, _ = foot_points(points, result)
    np.testing.assert_allclose(np.hypot(*normal.T), 1.0)
    np.testing.assert_allclose(foot + dev[:, None] * normal, points, atol=1e-12)


def test_refining_from_a_nearby_fit_finds_the_same_feet(fit):
    points, result = fit
    exact = foot_points(points, result)
    # The previous step: the same fit, its control points moved a little
    nearby = FitResult([[(x + 2e-3, y - 1e-3) for x, y in c] for c in result.section_control_points],
                       5, 0.0, 0, 0, 0.0, section_knots=result.section_knots)
    start = foot_points(points, nearby)[3]
    refined = foot_points(points, result, start=start)
    # Within a thousandth of the data's rounding, far below a pixel; the next
    # step refines it further
    np.testing.assert_allclose(refined[0], exact[0], rtol=0, atol=5e-8)
    np.testing.assert_allclose(refined[1], exact[1], rtol=0, atol=1e-5)  # along the curve


def test_a_start_for_another_section_count_is_ignored(fit):
    points, result = fit
    start = np.zeros((len(points), 5))
    np.testing.assert_allclose(foot_points(points, result, start=start)[0],
                               signed_deviations(points, result))


def test_deviation_is_positive_outside(fit):
    points, result = fit
    moved = points.copy()
    upper, lower = 20, 120
    assert moved[upper, 1] > 0 > moved[lower, 1]
    moved[upper, 1] += 1e-3
    moved[lower, 1] -= 1e-3
    dev = signed_deviations(moved, result)
    assert dev[upper] == pytest.approx(1e-3, rel=0.1)
    assert dev[lower] == pytest.approx(1e-3, rel=0.1)
    moved[upper, 1] -= 2e-3
    assert signed_deviations(moved, result)[upper] == pytest.approx(-1e-3, rel=0.1)


def _empty(item) -> bool:
    x, _ = item.getData()
    return x is None or len(x) == 0


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
    w.show()
    coords = [tuple(p) for p in np.round(airfoil.NACA4("2412").points(81), 5)]
    w._airfoil = create_airfoil_data(coords, filename="n", name="n", precision=5, num_sections=2)
    w._plot_manager.plot_airfoil(coords)
    yield w
    w.deleteLater()
    app.processEvents()


def _reach(window, result):
    """Length of every whisker, in data order, after the whiskers start at the feet."""
    manager = window._plot_manager
    # Two sections: the first in the even color, the second in the odd
    tips = [np.column_stack(manager._deviation_items[parity, True].getData())
            for parity in (0, 1)]
    # The point where the nearest section changes is on both
    assert sum(len(t) for t in tips) == len(window._airfoil.coordinates) + 1
    dev, foot, _, _, section = foot_points(window._airfoil.coordinates, result)
    assert np.all(np.diff(section) >= 0) and set(section) == {0, 1}
    return dev, np.hypot(*(np.concatenate([tips[0], tips[1][1:]]) - foot).T)


def test_whiskers_reach_the_height_at_the_tolerance(window):
    manager = window._plot_manager
    window._seg_goal.setCurrentIndex(window._seg_goal.findData(True))  # to a tolerance
    window._fitting_controller.on_fit()
    result = window._fit_result
    assert result is not None
    assert _empty(manager._deviation_items[0, True])

    window._check_deviation.setChecked(True)
    tolerance = result.report["tolerance"]
    chord = np.ptp([p[0] for p in window._airfoil.coordinates])
    height = window._spin_deviation_height.value()
    dev, reach = _reach(window, result)
    np.testing.assert_allclose(reach, height * chord * np.abs(dev) / tolerance, atol=1e-12)
    assert np.max(reach) <= height * chord * (1 + 1e-9)
    assert not _empty(manager._tolerance_item)

    window._spin_deviation_height.setValue(2 * height)
    _, doubled = _reach(window, result)
    np.testing.assert_allclose(doubled, 2 * reach, atol=1e-12)

    window._check_deviation.setChecked(False)
    assert _empty(manager._deviation_items[0, True])
    assert _empty(manager._tolerance_item)


def test_without_a_tolerance_the_largest_deviation_reaches_the_height(window):
    window._fitting_controller.on_fit()  # a fixed control-point count
    assert not window._fit_result.report.get("tolerance")
    window._check_deviation.setChecked(True)
    _, reach = _reach(window, window._fit_result)
    chord = np.ptp([p[0] for p in window._airfoil.coordinates])
    assert np.max(reach) == pytest.approx(window._spin_deviation_height.value() * chord)
    assert _empty(window._plot_manager._tolerance_item)


def test_live_preview_draws_the_deviation(window, monkeypatch):
    from spline_fitter.gui.plotting import PlotManager
    drawn = []
    update = PlotManager.update_live_plot

    def recording(self, *args, **kwargs):
        update(self, *args, **kwargs)
        x, _ = self._deviation_items[0, True].getData()
        drawn.append((kwargs.get("tolerance"), None if x is None else len(x)))

    monkeypatch.setattr(PlotManager, "update_live_plot", recording)
    window._seg_goal.setCurrentIndex(window._seg_goal.findData(True))  # to a tolerance
    window._combo_live.setCurrentIndex(2)  # every step
    window._check_deviation.setChecked(True)
    window._fitting_controller.on_fit()
    assert len(drawn) > 3
    # The first section's whiskers, which run to about the leading edge
    le = int(np.argmin(np.hypot(*np.asarray(window._airfoil.coordinates).T)))
    assert all(tol and abs(n - (le + 1)) <= 1 for tol, n in drawn)


def test_default_height_is_five_percent_of_the_chord(window):
    assert window._spin_deviation_height.value() == pytest.approx(0.05)


def test_a_new_fit_clears_the_deviation(window):
    window._fitting_controller.on_fit()
    window._check_deviation.setChecked(True)
    window._plot_manager.clear_plot_items()
    assert _empty(window._plot_manager._deviation_items[1, False])
