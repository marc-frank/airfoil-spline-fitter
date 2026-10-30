"""The Curve settings never allow a degree or control point count the
continuity cannot be fitted with."""
import os

import pytest

from spline_fitter.optimization.models import ContinuityMode


@pytest.fixture
def window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui.ui import SplineFitterWindow
    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    yield w
    w.deleteLater()
    app.processEvents()


def test_control_point_minimum_applies_from_the_start(window):
    mode = window._combo_cont.currentData()
    degree = window._spin_degree.value()
    assert window._spin_ctrl.minimum() == mode.minimum_control_points(degree)
    window._spin_ctrl.setValue(2)
    assert window._spin_ctrl.value() >= degree + 1


def test_degree_raises_the_control_point_minimum(window):
    window._combo_cont.setCurrentIndex(window._combo_cont.findData(ContinuityMode.G0))
    window._spin_degree.setValue(3)
    window._spin_ctrl.setValue(4)
    window._spin_degree.setValue(5)
    assert window._spin_ctrl.minimum() == 6
    assert window._spin_ctrl.value() == 6


def test_continuity_raises_the_degree_minimum(window):
    window._combo_cont.setCurrentIndex(window._combo_cont.findData(ContinuityMode.G0))
    window._spin_degree.setValue(1)
    assert window._spin_degree.value() == 1
    window._combo_cont.setCurrentIndex(window._combo_cont.findData(ContinuityMode.G3))
    assert window._spin_degree.minimum() == 3
    assert window._spin_degree.value() == 3
    assert window._spin_ctrl.value() >= ContinuityMode.G3.minimum_control_points(3)


def test_sweep_upper_end_follows_the_minimum(window):
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_ctrl._spin_max.setValue(window._spin_ctrl.minimum())
    window._spin_degree.setValue(window._spin_degree.value() + 2)
    low, high = window._sweep_ctrl.sweep_range
    assert min(low, high) >= window._spin_ctrl.minimum()


def _g0_degree_3_with_6_control_points(window):
    window._combo_cont.setCurrentIndex(window._combo_cont.findData(ContinuityMode.G0))
    window._spin_degree.setValue(3)
    window._spin_ctrl.setValue(6)


def test_degree_sweep_stops_at_what_a_fixed_count_allows(window):
    _g0_degree_3_with_6_control_points(window)
    window._sweep_degree._btn_sweep.click()
    window._sweep_degree._spin_max.setValue(10)
    assert window._sweep_degree.sweep_range == (3, 5)
    assert window._spin_ctrl.value() == 6
    # Every swept degree must fit the count, so it cannot go below the top's need
    window._spin_ctrl.setValue(4)
    assert window._spin_ctrl.value() == 6


def test_degree_and_count_both_swept_open_the_limits(window):
    _g0_degree_3_with_6_control_points(window)
    window._sweep_degree._btn_sweep.click()
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_degree._spin_max.setValue(10)
    assert window._sweep_degree.sweep_range == (3, 10)
    window._spin_ctrl.setValue(4)
    assert window._spin_ctrl.value() == 4
    # Back to a fixed count: it stays, and the degree sweep is capped again
    window._sweep_ctrl._btn_sweep.click()
    assert window._spin_ctrl.value() == 4
    assert window._sweep_degree.sweep_range == (3, 3)


def test_applying_a_stored_run_keeps_its_degree(window):
    from types import SimpleNamespace

    from spline_fitter.optimization.models import (
        FairnessWeights,
        OptimizationConfig,
        SplineConfig,
    )

    window._combo_cont.setCurrentIndex(window._combo_cont.findData(ContinuityMode.G3))
    cfg = SplineConfig(degree=2, num_control_points=4, continuity=ContinuityMode.G2)
    run = SimpleNamespace(spline_config=cfg, optimization_config=OptimizationConfig(),
                          fairness_weights=FairnessWeights())
    window._apply_run_settings(run)
    assert window._spin_degree.value() == 2
    assert window._spin_ctrl.value() == 4
