"""Live preview: Detailed draws every optimizer step, sweeps show every fit."""
import os

import numpy as np
import pytest
from bspline_varpro.airfoil import NACA4

from spline_fitter.geometry import create_airfoil_data


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
    pts = np.round(NACA4("2412").points(81), 6)
    w._airfoil = create_airfoil_data([tuple(p) for p in pts], filename="n", name="n",
                                     precision=6, num_sections=2)
    w._check_knots.setChecked(True)
    w._spin_ctrl.setValue(10)
    yield w
    w.deleteLater()
    app.processEvents()


@pytest.fixture
def counted(monkeypatch):
    """Counts optimizer callbacks and live drawings."""
    from spline_fitter.gui import live
    counts = {"steps": 0, "drawn": 0, "step_flags": 0}
    callback, draw = live.LiveProgress.callback, live.LiveProgress._draw

    def counting_callback(self, x, error, preview=None, step=False):
        counts["steps"] += 1
        counts["step_flags"] += step
        return callback(self, x, error, preview=preview, step=step)

    def counting_draw(self):
        counts["drawn"] += 1
        draw(self)

    monkeypatch.setattr(live.LiveProgress, "callback", counting_callback)
    monkeypatch.setattr(live.LiveProgress, "_draw", counting_draw)
    return counts


def test_detailed_draws_every_step(window, counted):
    window._combo_live.setCurrentIndex(2)
    window._fitting_controller.on_fit()
    assert window._fit_result is not None
    assert counted["steps"] > 5
    assert counted["drawn"] == counted["steps"]


def test_detailed_draws_every_step_of_a_tolerance_search(window, counted):
    window._combo_live.setCurrentIndex(2)
    window._seg_goal.setCurrentIndex(window._seg_goal.findData(True))
    window._check_auto_tolerance.setChecked(False)
    window._spin_tolerance.setValue(1e-4)
    window._fitting_controller.on_fit()
    assert window._fit_result is not None
    # Steps inside the refits are reported, not only the refits
    assert counted["step_flags"] > 10
    refits = counted["steps"] - counted["step_flags"]
    assert counted["drawn"] == counted["step_flags"] > refits
    # The loss curve has a point per step
    assert len(window._plot_manager._loss_y) == counted["step_flags"]


def test_minimal_draws_only_the_result(window, counted):
    window._combo_live.setCurrentIndex(0)
    window._fitting_controller.on_fit()
    assert counted["steps"] > 5 and counted["drawn"] == 0


def test_result_shows_sections_times_control_points(window):
    window._combo_live.setCurrentIndex(0)
    window._fitting_controller.on_fit()
    window._set_save_enabled(True)
    assert window._lbl_result_ctrl.text() == "2 sections of 10"
    assert window._lbl_result_ctrl.toolTip() == "10, 10 per section"


def test_live_setting_can_change_while_fitting(window, counted, monkeypatch):
    """Switching from Detailed to Minimal mid-fit stops the drawing, and the
    worker waiting for a Detailed drawing is released."""
    from spline_fitter.gui import live
    on_tick = live.LiveProgress.on_tick

    enabled = []

    def switch_after_three(self):
        enabled.append(window._combo_live.isEnabled())
        if counted["drawn"] >= 3:
            window._combo_live.setCurrentIndex(0)
        on_tick(self)

    monkeypatch.setattr(live.LiveProgress, "on_tick", switch_after_three)
    window._combo_live.setCurrentIndex(2)
    window._fitting_controller.on_fit()
    assert window._fit_result is not None
    assert enabled and all(enabled)
    assert counted["drawn"] == 3 < counted["steps"]


def test_sweep_toggles_are_locked_while_fitting(window, monkeypatch):
    from spline_fitter.gui import live
    on_tick = live.LiveProgress.on_tick
    toggles = (window._sweep_degree._btn_sweep, window._sweep_ctrl._btn_sweep)
    states = []

    def record(self):
        states.append([t.isEnabled() for t in toggles])
        on_tick(self)

    monkeypatch.setattr(live.LiveProgress, "on_tick", record)
    window._fitting_controller.on_fit()
    assert states and not any(any(s) for s in states)
    assert all(t.isEnabled() for t in toggles)


def test_sweep_shows_every_fit_and_its_comparison_point(window, counted, monkeypatch):
    window._combo_live.setCurrentIndex(2)
    sizes = []
    plots = window._plot_manager
    set_data = plots.set_comparison_data
    monkeypatch.setattr(plots, "set_comparison_data",
                        lambda x, y, **kw: (sizes.append(len(x)), set_data(x, y, **kw)))
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_ctrl._spin_max.setValue(12)
    window._fitting_controller.on_fit()
    assert len(window._fitting_controller._sweep_results) == 3
    assert counted["drawn"] == counted["steps"] > 0
    # Cleared at the start, one more point after each fit, then the final view
    assert sizes == [0, 1, 2, 3, 3]


def test_sweep_fit_in_progress_replaces_the_previous_fit(window, counted, monkeypatch):
    """While a sweep fit is drawn live, the previous fit's curve is gone: the
    control polygon holds the points of the fit in progress only."""
    from spline_fitter.gui import live
    draw = live.LiveProgress._draw
    stale = []

    def record(self):
        draw(self)
        polygon = window._plot_manager._polygon_item.getData()[0]
        drawn = len(polygon) - (self._config.num_sections - 1)
        stale.append(drawn != self._config.num_sections * self._config.num_control_points)

    monkeypatch.setattr(live.LiveProgress, "_draw", record)
    window._combo_live.setCurrentIndex(2)
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_ctrl._spin_max.setValue(11)
    window._fitting_controller.on_fit()
    assert len(window._fitting_controller._sweep_results) == 2
    assert stale and not any(stale)



def test_second_sweep_shows_the_whole_project_range(window, monkeypatch):
    """The comparison plot of a sweep keeps the runs of earlier sweeps of the
    same configuration, but not runs whose other settings differ."""
    window._combo_live.setCurrentIndex(0)
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_ctrl._spin_max.setValue(11)
    window._fitting_controller.on_fit()
    # A run with another smoothing weight is a different series
    window._sweep_ctrl.stop_sweep()
    window._spin_smoothing_weight.setValue(window._spin_smoothing_weight.value() * 10)
    window._fitting_controller.on_fit()
    window._spin_smoothing_weight.setValue(window._spin_smoothing_weight.value() / 10)

    window._spin_ctrl.setValue(14)
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_ctrl._spin_max.setValue(15)
    window._fitting_controller.on_fit()
    x, _ = window._plot_manager._comparison_curve.getData()
    assert list(x) == [10, 11, 14, 15]

def test_batch_shows_every_file_live(window, counted, tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    for code in ("0012", "2412"):
        pts = np.round(NACA4(code).points(81), 6)
        lines = [f"NACA {code}"] + [f"{x:.6f} {y:.6f}" for x, y in pts]
        (src / f"naca{code}.dat").write_text("\n".join(lines) + "\n")
    window._combo_live.setCurrentIndex(2)
    batch = window._batch_controller
    batch._batch_src_dir, batch._batch_dst_dir = str(src), str(dst)
    batch._on_batch_convert()
    assert len(list(dst.glob("*.spline.dat"))) == 2
    assert counted["drawn"] == counted["steps"] > 0
    assert window.statusBar().currentMessage().startswith("Batch complete: 2 files")


def test_sections_sweep_splits_the_airfoil_per_count(window):
    window._combo_live.setCurrentIndex(0)
    window._sweep_sections._btn_sweep.click()
    window._sweep_sections._spin_max.setValue(3)
    window._fitting_controller.on_fit()
    results = window._fitting_controller._sweep_results
    assert [v for v, _ in results] == [(2,), (3,)]
    assert [len(r.section_control_points) for _, r in results] == [2, 3]
    x, _ = window._plot_manager._comparison_curve.getData()
    assert list(x) == [2, 3]
    # The best count is applied, with the airfoil split to match it
    assert window._airfoil.num_sections == window._spin_sections.value()


def test_sweep_toggles_add_up(window):
    """Turning on a second sweep keeps the first one."""
    window._sweep_degree._btn_sweep.click()
    window._sweep_ctrl._btn_sweep.click()
    assert window._get_active_sweeps() == [window._sweep_degree, window._sweep_ctrl]


def test_combined_sweep_fits_every_combination(window):
    """Two swept settings: one fit per pair of values, those with too few
    control points for their degree left out, and the best pair applied."""
    window._combo_live.setCurrentIndex(0)
    window._spin_degree.setValue(3)
    window._sweep_degree._btn_sweep.click()
    window._sweep_degree._spin_max.setValue(4)
    window._spin_ctrl.setValue(10)
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_ctrl._spin_max.setValue(12)
    window._fitting_controller.on_fit()
    results = window._fitting_controller._sweep_results
    assert [v for v, _ in results] == [(3, 10), (3, 11), (3, 12), (4, 10), (4, 11), (4, 12)]
    assert [(r.degree, len(r.section_control_points[0])) for _, r in results] == [
        (3, 10), (3, 11), (3, 12), (4, 10), (4, 11), (4, 12)]
    # Along control points (the most values), two fits per value: dots, not a line
    x, _ = window._plot_manager._comparison_curve.getData()
    assert list(x) == [10, 10, 11, 11, 12, 12]
    best, _ = min(results, key=lambda vr: vr[1].final_error)
    assert (window._spin_degree.value(), window._spin_ctrl.value()) == best
    assert window.statusBar().currentMessage().startswith("Sweep complete: 6 fits")


def test_smoothing_sweep_steps_by_step_size(window):
    sweep = window._sweep_smoothing_weight
    window._spin_smoothing_weight.setValue(0.002)
    sweep._btn_sweep.click()
    sweep._spin_max.setValue(0.01)
    sweep._spin_step.setValue(0.004)
    assert sweep.sweep_values() == [0.002, 0.006, 0.01]
    sweep._spin_step.setValue(0.003)
    assert sweep.sweep_values() == [0.002, 0.005, 0.008]


def test_double_clicking_a_splitter_handle_restores_the_default_width(window):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    from spline_fitter.gui.layout import EXPLORER_WIDTH

    splitter = window._main_splitter
    window.resize(2200, 950)
    QTest.qWait(50)
    sizes = splitter.sizes()
    sizes[0] += 120
    sizes[2] += 90
    sizes[1] -= 210
    splitter.setSizes(sizes)
    for index in (1, 2):
        handle = splitter.handle(index)
        QTest.mouseDClick(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
    width = splitter.sizes()
    assert width[0] == window._sidebar_width and width[0] % 10 == 0
    assert width[2] == EXPLORER_WIDTH
    assert sum(width) == sum(sizes)
