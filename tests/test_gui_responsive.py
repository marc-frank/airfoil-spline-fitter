"""The GUI keeps processing events while fits run.

A 20 ms timer records the gaps between its ticks while a fit runs; the
event loop must never stall for FREEZE_S or longer.
"""
import os
import time

import numpy as np
import pytest
from bspline_varpro.airfoil import NACA4

from spline_fitter.geometry import create_airfoil_data

FREEZE_S = 1.0


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
    pts = np.round(NACA4("2412").points(301), 6)
    w._airfoil = create_airfoil_data([tuple(p) for p in pts], filename="n", name="n",
                                     precision=6, num_sections=2)
    w._check_comb.setChecked(True)
    w._check_knots.setChecked(True)
    w._combo_live.setCurrentIndex(2)
    yield w
    w.deleteLater()
    app.processEvents()


def _longest_stall(fit) -> float:
    from PyQt6.QtCore import QTimer
    gaps, last = [], [time.perf_counter()]

    def tick():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now

    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(tick)
    timer.start()
    start = time.perf_counter()
    fit()
    total = time.perf_counter() - start
    timer.stop()
    # No tick at all means the whole fit blocked the event loop
    return max(gaps) if gaps else total


def test_fit_keeps_gui_responsive(window):
    stall = _longest_stall(window._fitting_controller.on_fit)
    assert window._fit_result is not None
    assert stall < FREEZE_S, f"GUI stalled {stall:.2f} s during a VarPro fit"


def test_sweep_keeps_gui_responsive(window):
    window._spin_ctrl.setValue(10)
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_ctrl._spin_max.setValue(12)
    stall = _longest_stall(window._fitting_controller.on_fit)
    assert len(window._fitting_controller._sweep_results) == 3
    assert stall < FREEZE_S, f"GUI stalled {stall:.2f} s during a sweep"


def test_cancel_reaches_a_running_fit(window):
    from PyQt6.QtCore import QTimer
    window._seg_goal.setCurrentIndex(window._seg_goal.findData(True))
    window._combo_knot_placement.setCurrentIndex(window._combo_knot_placement.findData("free"))
    window._check_grow.setChecked(True)
    pressed = []
    QTimer.singleShot(1000, lambda: (pressed.append(time.perf_counter()),
                                     window._btn_cancel.click()))
    stall = _longest_stall(window._fitting_controller.on_fit)
    assert pressed, "Cancel was not clicked while the fit ran"
    # The tolerance search checks for Cancel inside its refits too
    waited = time.perf_counter() - pressed[0]
    assert waited < FREEZE_S, f"the fit ran on {waited:.2f} s after Cancel"
    assert window.statusBar().currentMessage().startswith("Cancelled")
    assert stall < FREEZE_S, f"GUI stalled {stall:.2f} s while cancelling"
