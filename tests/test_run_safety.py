"""Fits, sweeps and batches: what runs while they run, how they fail, and
what they leave in the project and the settings."""
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from bspline_varpro.airfoil import NACA4

from spline_fitter.geometry import create_airfoil_data

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])


@pytest.fixture
def window(monkeypatch):
    from spline_fitter.gui import fitting
    from spline_fitter.gui.ui import SplineFitterWindow
    w = SplineFitterWindow()
    monkeypatch.setattr(fitting.OptimizerErrorLogger, "save", lambda *a, **k: None)
    monkeypatch.setattr(w, "_play_completion_chime", lambda: None)
    monkeypatch.setattr(w, "_play_error_chime", lambda: None)
    monkeypatch.setattr(w, "_add_recent_file", lambda *a: None)
    w.show()
    w._airfoil = _naca("2412")
    w._spin_ctrl.setValue(10)
    w._combo_live.setCurrentIndex(0)
    yield w
    w.deleteLater()
    _APP.processEvents()


def _naca(code, sections=2):
    pts = np.round(NACA4(code).points(81), 6)
    return create_airfoil_data([tuple(p) for p in pts], filename=code, name=f"NACA {code}",
                               precision=6, num_sections=sections)


def _slow_fits(monkeypatch, seconds=0.5):
    """Make every fit take at least ``seconds`` (worker thread)."""
    from spline_fitter.gui import fitting
    real = fitting.run_optimization

    def slow(*a, **k):
        time.sleep(seconds)
        return real(*a, **k)
    monkeypatch.setattr(fitting, "run_optimization", slow)


def _during_fit(window, monkeypatch, action):
    """Run ``action`` on the GUI thread once the next fit has started."""
    set_fitting_ui = window._set_fitting_ui

    def start(fitting):
        set_fitting_ui(fitting)
        if fitting:
            QTimer.singleShot(0, action)
    monkeypatch.setattr(window, "_set_fitting_ui", start)


def _fail(*a, **k):
    raise RuntimeError("solver blew up")


# -- Project ----------------------------------------------------------------

def test_cancelled_rerun_keeps_the_finished_run(window, monkeypatch):
    from spline_fitter.gui import live
    window._fitting_controller.on_fit()
    runs = window._project_manager.project.optimization_runs
    finished = runs[0].result
    assert finished.converged

    callback = live.LiveProgress.callback

    def cancel_at_once(self, *a, **k):
        window._cancel_requested = True
        return callback(self, *a, **k)
    monkeypatch.setattr(live.LiveProgress, "callback", cancel_at_once)
    window._fitting_controller.on_fit()

    assert window._fit_result.cancelled
    assert [r.result for r in runs] == [finished]
    assert "keeps its finished run" in window.statusBar().currentMessage()


def test_cancelled_run_is_kept_when_nothing_finished(window, monkeypatch):
    from spline_fitter.gui import live
    callback = live.LiveProgress.callback

    def cancel_at_once(self, *a, **k):
        window._cancel_requested = True
        return callback(self, *a, **k)
    monkeypatch.setattr(live.LiveProgress, "callback", cancel_at_once)
    window._fitting_controller.on_fit()
    runs = window._project_manager.project.optimization_runs
    assert len(runs) == 1 and runs[0].result.cancelled


def test_project_cannot_be_opened_during_a_fit(window, monkeypatch, tmp_path):
    from spline_fitter.project_io import create_project, save_project
    path = str(tmp_path / "other.asfproj")
    save_project(create_project(_naca("0012")), path)
    _slow_fits(monkeypatch)
    seen = {}

    def open_now():
        seen["enabled"] = window._action_open_project.isEnabled()
        seen["opened"] = window._project_manager.open_project(path)
    _during_fit(window, monkeypatch, open_now)
    window._fitting_controller.on_fit()

    assert seen == {"enabled": False, "opened": False}
    assert window._airfoil.name == "NACA 2412"
    assert window._project_manager.project.airfoil.name == "NACA 2412"
    assert window._action_open_project.isEnabled()


def test_explorer_neither_loads_nor_applies_runs_during_a_fit(window, monkeypatch):
    window._fitting_controller.on_fit()
    old = window._project_manager.project.optimization_runs[0]
    window._spin_ctrl.setValue(11)
    window._fitting_controller.on_fit()   # the current fit is not the old run's
    window._spin_ctrl.setValue(12)
    monkeypatch.setattr(window._run_table, "selected_runs", lambda: [old])
    _slow_fits(monkeypatch)
    seen = {}

    def use_explorer():
        manager = window._project_manager
        manager.on_table_selection_changed()
        manager.on_run_activated(old)
        manager.on_delete_selected_runs()
        seen.update(ctrl=window._spin_ctrl.value(),
                    old_fit_shown=window._fit_result is old.result,
                    save=window._btn_save_spline.isEnabled(),
                    report=window._action_fit_report.isEnabled(),
                    load=window._btn_load_run.isEnabled(),
                    delete=window._btn_delete_run.isEnabled())
    _during_fit(window, monkeypatch, use_explorer)
    window._fitting_controller.on_fit()

    assert seen == dict(ctrl=12, old_fit_shown=False, save=False, report=False,
                        load=False, delete=False)
    assert len(window._project_manager.project.optimization_runs) == 3
    # Afterwards the buttons follow the selection again
    assert window._btn_load_run.isEnabled() and window._btn_delete_run.isEnabled()


# -- Failures -----------------------------------------------------------------

def test_unexpected_fit_error_is_reported(window, monkeypatch):
    from spline_fitter.gui import fitting
    monkeypatch.setattr(fitting, "run_optimization", _fail)
    window._fitting_controller.on_fit()
    assert window.statusBar().currentMessage() == "Fitting error: solver blew up"
    assert not window._is_fitting


def test_fit_without_result_keeps_the_current_fit(window, monkeypatch):
    from spline_fitter.analysis import summary_line
    from spline_fitter.gui import fitting
    window._fitting_controller.on_fit()
    current = window._fit_result
    monkeypatch.setattr(fitting, "run_optimization", lambda *a, **k: SimpleNamespace(
        x=None, nit=0, fun=float("inf"), success=False, message="no result"))
    window._fitting_controller.on_fit()
    assert window._fit_result is current
    assert summary_line(current.report) not in window.statusBar().currentMessage()
    # It is drawn again after the plots were cleared for the run
    x, _ = window._plot_manager._curve_item.getData()
    assert x is not None and len(x)


def _curve_points(window) -> int:
    x, _ = window._plot_manager._curve_item.getData()
    return 0 if x is None else len(x)


def test_failed_fit_keeps_the_current_fit_drawn(window, monkeypatch):
    from spline_fitter.gui import fitting
    window._fitting_controller.on_fit()
    current = window._fit_result
    monkeypatch.setattr(fitting, "run_optimization", _fail)
    window._fitting_controller.on_fit()
    assert window._fit_result is current
    # It is drawn again after the plots were cleared for the run
    assert _curve_points(window)


def test_cancel_while_the_finished_fit_is_measured_is_no_cancel(window, monkeypatch):
    """The result and the status agree on whether a fit was cancelled."""
    from spline_fitter.gui import fitting
    measure = fitting.fit_report

    def cancel_then_measure(*a, **k):
        window._cancel_requested = True
        return measure(*a, **k)
    monkeypatch.setattr(fitting, "fit_report", cancel_then_measure)
    window._fitting_controller.on_fit()
    assert window._fit_result.converged and not window._fit_result.cancelled
    assert window.statusBar().currentMessage().startswith("Completed")


def test_cancelled_sweep_neither_wins_nor_sets_the_settings(window, monkeypatch):
    from spline_fitter.gui import live
    callback = live.LiveProgress.callback
    calls = []

    def cancel_in_the_last_fit(self, *a, **k):
        if self._config.num_control_points == 12:
            calls.append(1)
            if len(calls) == 5:
                window._cancel_requested = True
        return callback(self, *a, **k)
    monkeypatch.setattr(live.LiveProgress, "callback", cancel_in_the_last_fit)
    box = window._sweep_ctrl
    box._btn_sweep.click()
    box._spin_max.setValue(12)
    window._fitting_controller.on_fit()

    results = window._fitting_controller._sweep_results
    assert [values for values, _ in results] == [(10,), (11,)]
    assert not window._fit_result.cancelled
    assert window.statusBar().currentMessage().startswith("Sweep cancelled: 2 fits")
    # The sweep stays, to be run again
    assert box.is_sweep_active and window._spin_ctrl.value() == 10


def test_sweep_fit_whose_drawing_fails_counts_as_a_fit(window, monkeypatch):
    controller = window._fitting_controller
    show = controller._show_sweep_comparison
    calls = []

    def fail_once():
        calls.append(1)
        if len(calls) == 1:
            _fail()
        return show()
    monkeypatch.setattr(controller, "_show_sweep_comparison", fail_once)
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_ctrl._spin_max.setValue(11)
    controller.on_fit()
    assert len(controller._sweep_results) == 2
    assert "failed" not in window.statusBar().currentMessage()


def test_sweep_fit_that_cannot_be_kept_counts_once(window, monkeypatch):
    project = window._project_manager
    add = project.add_optimization_run

    def fail_at_11(spline_config, **k):
        if spline_config.num_control_points == 11:
            _fail()
        return add(spline_config=spline_config, **k)
    monkeypatch.setattr(project, "add_optimization_run", fail_at_11)
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_ctrl._spin_max.setValue(11)
    window._fitting_controller.on_fit()
    assert [v for v, _ in window._fitting_controller._sweep_results] == [(10,)]
    message = window.statusBar().currentMessage()
    assert message.startswith("Sweep complete: 1 fits") and message.endswith("| 1 failed")


def test_sweep_without_success_names_the_combinations_left_out(window, monkeypatch):
    from spline_fitter.gui import fitting
    window._fitting_controller.on_fit()
    monkeypatch.setattr(fitting, "run_optimization", _fail)
    # Two control points: the higher degrees of the sweep have too few
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_degree._btn_sweep.click()
    window._spin_ctrl.setValue(window._spin_ctrl.minimum())
    window._sweep_ctrl._spin_max.setValue(window._spin_ctrl.minimum())
    window._sweep_degree._spin_max.setValue(window._spin_degree.value() + 2)
    window._fitting_controller.on_fit()
    message = window.statusBar().currentMessage()
    assert message.startswith("Sweep failed: no fit succeeded | Left out ")
    assert "too few control points" in message
    # The current fit stays drawn
    assert _curve_points(window)


def test_closing_the_window_stops_a_running_fit(window, monkeypatch):
    """Fits run until they converge, so closing must stop one at once."""
    monkeypatch.setattr(window._project_manager, "confirm_discard", lambda: True)
    quits = _APP.quitOnLastWindowClosed()
    _APP.setQuitOnLastWindowClosed(False)
    window._seg_goal.setCurrentIndex(window._seg_goal.findData(True))
    window._combo_knot_placement.setCurrentIndex(window._combo_knot_placement.findData("free"))
    window._check_grow.setChecked(True)
    closed = []
    QTimer.singleShot(1000, lambda: (closed.append(time.perf_counter()), window.close()))
    try:
        window._fitting_controller.on_fit()
    finally:
        _APP.setQuitOnLastWindowClosed(quits)
    assert closed, "the window was not closed while the fit ran"
    assert time.perf_counter() - closed[0] < 1.0
    assert window.statusBar().currentMessage().startswith("Cancelled")


def test_failed_sweep_fits_are_reported(window, monkeypatch):
    from spline_fitter.gui import fitting
    real = fitting.run_optimization

    def fail_at_11(config, *a, **k):
        if config.num_control_points == 11:
            _fail()
        return real(config, *a, **k)
    monkeypatch.setattr(fitting, "run_optimization", fail_at_11)
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_ctrl._spin_max.setValue(11)
    window._fitting_controller.on_fit()
    message = window.statusBar().currentMessage()
    assert message.startswith("Sweep complete: 1 fits") and message.endswith("| 1 failed")


def test_sweep_unlocks_the_window_after_an_error(window, monkeypatch):
    controller = window._fitting_controller
    monkeypatch.setattr(controller, "_show_sweep_comparison", _fail)
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_ctrl._spin_max.setValue(11)
    with pytest.raises(RuntimeError):
        controller.on_fit()
    assert not window._is_fitting and window._btn_fit.isEnabled()


def _batch_folders(tmp_path, codes=("0012",)):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    for code in codes:
        pts = np.round(NACA4(code).points(81), 6)
        lines = [f"NACA {code}"] + [f"{x:.6f} {y:.6f}" for x, y in pts]
        (src / f"naca{code}.dat").write_text("\n".join(lines) + "\n")
    return src, dst


def test_batch_reports_files_it_skips(window, tmp_path):
    from spline_fitter.gui.batch import output_name
    src, dst = _batch_folders(tmp_path, ("0012", "2412"))
    name = output_name("naca0012", window._get_spline_config(),
                       window._get_optimization_config())
    (dst / name).write_text("earlier\n")
    batch = window._batch_controller
    batch._batch_src_dir, batch._batch_dst_dir = str(src), str(dst)
    batch._on_batch_convert()
    message = window.statusBar().currentMessage()
    assert message.startswith("Batch complete: 1 files")
    assert "Skipped 1 already in the target folder" in message


def test_batch_output_names_tell_settings_apart():
    """A new window's settings keep the plain name; every setting that
    changes the fit changes the name, so no two runs share an output."""
    from dataclasses import replace

    from spline_fitter.gui.batch import output_name
    from spline_fitter.gui.ui import SplineFitterWindow
    from spline_fitter.optimization.models import OptimizerType

    fresh = SplineFitterWindow()
    config, opt = fresh._get_spline_config(), fresh._get_optimization_config()
    fresh.deleteLater()
    assert output_name("naca0012", config, opt) == "naca0012.deg5.spline.dat"

    variants = {
        "degree": (replace(config, degree=4), opt),
        "count": (replace(config, num_control_points=30), opt),
        "sections": (replace(config, num_sections=3), opt),
        "parametric": (replace(config, geometric_continuity=False), opt),
        "junctions": (replace(config, num_sections=3, junction_continuity=("G2", "C1")), opt),
        "tolerance": (config, replace(opt, tolerance=1e-4)),
        "auto tolerance": (config, replace(opt, tolerance=2e-4)),
        "free knots": (config, replace(opt, knot_placement="free")),
        "smoothing": (config, replace(opt, smoothing_weight=0.01)),
        "subsample": (config, replace(opt, max_points_per_section=200)),
        "leading edge": (config, replace(opt, optimize_le_position=True)),
        "direct lsq": (config, replace(opt, optimizer=OptimizerType.DIRECT_LSQ)),
    }
    names = {key: output_name("naca0012", c, o, auto_tolerance=key == "auto tolerance")
             for key, (c, o) in variants.items()}
    assert len(set(names.values()) | {"naca0012.deg5.spline.dat"}) == len(names) + 1
    assert names["junctions"] == "naca0012.deg5.s3.G2-C1.spline.dat"
    assert names["tolerance"] == "naca0012.deg5.tol1e-4.spline.dat"
    assert names["auto tolerance"] == "naca0012.deg5.tol-auto.spline.dat"
    assert names["parametric"] == "naca0012.deg5.C3.spline.dat"
    # Numbers are written the same in every locale
    assert names["smoothing"] == "naca0012.deg5.smooth0.01.spline.dat"


def test_batch_unlocks_the_window_after_an_error(window, tmp_path, monkeypatch):
    src, dst = _batch_folders(tmp_path)
    monkeypatch.setattr(window._plot_manager, "plot_airfoil", _fail)
    batch = window._batch_controller
    batch._batch_src_dir, batch._batch_dst_dir = str(src), str(dst)
    with pytest.raises(RuntimeError):
        batch._on_batch_convert()
    assert not window._is_fitting and window._btn_fit.isEnabled()


def test_batch_without_an_airfoil_leaves_the_plot_empty(window, tmp_path):
    src, dst = _batch_folders(tmp_path)
    window._airfoil = None
    batch = window._batch_controller
    batch._batch_src_dir, batch._batch_dst_dir = str(src), str(dst)
    batch._on_batch_convert()
    assert window.statusBar().currentMessage().startswith("Batch complete: 1 files")
    x, _ = window._plot_manager._data_item.getData()
    assert (x is None or not len(x)) and not _curve_points(window)


# -- Sweep settings -------------------------------------------------------------

def test_sweep_applies_its_best_values_as_an_undoable_step(window):
    box = window._sweep_ctrl
    box._btn_sweep.click()
    box._spin_max.setValue(12)
    window._fitting_controller.on_fit()
    best, _ = min(window._fitting_controller._sweep_results, key=lambda vr: vr[1].final_error)
    # The sweep is over: its range is not narrowed to start at the best value
    assert not box.is_sweep_active and window._spin_ctrl.value() == best[0]
    window._settings_history.undo()
    assert box.is_sweep_active and box.sweep_range == (10, 12)


# -- Background work and the live preview ------------------------------------------

def test_detailed_step_drawn_before_the_wait_does_not_hang():
    """A GUI tick that draws a step right after it is published, before the
    worker starts waiting for it, must not leave the worker waiting."""
    from spline_fitter.gui.live import DETAILED, LiveProgress
    from spline_fitter.optimization import OptimizerErrorLogger

    plots = SimpleNamespace(update_loss_data=lambda *a: None, has_loss_data=False,
                            update_live_plot=lambda *a, **k: None)
    status = SimpleNamespace(showMessage=lambda *a: None)
    w = SimpleNamespace(_cancel_requested=False, _live_mode=DETAILED, _plot_manager=plots,
                        statusBar=lambda: status, _combo_loss_x=None)
    live = LiveProgress(w, None, SimpleNamespace(tolerance=None),
                        OptimizerErrorLogger(run_name="t"), time.perf_counter(),
                        airfoil=object())

    class TickRightAfter(dict):
        def update(self, *a, **k):
            super().update(*a, **k)
            live.on_tick()
    live._latest = TickRightAfter()

    stopped = []
    worker = threading.Thread(target=lambda: stopped.append(live.callback(np.zeros(4), 1.0)))
    worker.start()
    worker.join(2)
    hung = worker.is_alive()
    w._cancel_requested = True
    worker.join()
    assert not hung and stopped == [False]


def test_nested_background_work_keeps_the_outer_tick_requests(monkeypatch):
    """Work started during other work (a fit report during a fit) leaves the
    outer work's tick requests working when it ends."""
    from spline_fitter.gui import background
    monkeypatch.setattr(background, "TICK_MS", 1500)
    ticked, inner_done = threading.Event(), threading.Event()

    def outer():
        inner_done.wait(5)
        ticked.clear()
        background.request_tick()
        return ticked.wait(0.8)

    def inner():
        background.run_in_background(lambda: None)
        inner_done.set()
    QTimer.singleShot(0, inner)
    assert background.run_in_background(outer, ticked.set)


def test_a_failing_tick_keeps_the_outcome_of_the_work():
    from spline_fitter.gui.background import run_in_background

    def tick():
        raise RuntimeError("drawing failed")
    assert run_in_background(lambda: 5, tick) == 5

    def work():
        raise ValueError("work failed")
    with pytest.raises(ValueError, match="work failed"):
        run_in_background(work, tick)


# -- Status and run log ---------------------------------------------------------

def test_status_counts_evaluations_not_iterations(window):
    window._fitting_controller.on_fit()
    message = window.statusBar().currentMessage()
    assert message.startswith("Completed (VarPro LM): ")
    assert " evaluations in " in message and "iteration" not in message.lower()


def test_status_counts_refits_of_a_tolerance_search(window):
    window._seg_goal.setCurrentIndex(window._seg_goal.findData(True))
    window._check_auto_tolerance.setChecked(False)
    window._spin_tolerance.setValue(1e-4)
    window._fitting_controller.on_fit()
    assert " refits, " in window.statusBar().currentMessage()


def test_run_log_names_the_continuity_as_fitted(window, monkeypatch):
    from spline_fitter.gui import fitting
    from spline_fitter.optimization.models import ContinuityMode, SplineConfig
    infos = []
    monkeypatch.setattr(fitting.OptimizerErrorLogger, "save",
                        lambda self, **k: infos.append(k["extra_info"]))
    window._fitting_controller.on_fit()
    assert infos == ["continuity=G3, degree=5"]

    text = fitting.continuity_text
    assert text(SplineConfig(continuity=ContinuityMode.G2, geometric_continuity=True)) == "G2"
    assert text(SplineConfig(continuity=ContinuityMode.G2)) == "C2"
    assert text(SplineConfig(num_sections=3, junction_continuity=("G2", "C1"))) == "G2/C1"
    assert text(SplineConfig(num_sections=1)) == "none"


def test_status_counts_the_solves_of_direct_lsq(window):
    from spline_fitter.optimization import get_evaluation_count
    from spline_fitter.optimization.models import OptimizerType
    window._combo_opt.setCurrentIndex(window._combo_opt.findData(OptimizerType.DIRECT_LSQ))
    window._fitting_controller.on_fit()
    solves = get_evaluation_count()
    assert solves >= 1
    noun = "solve" if solves == 1 else "solves"
    assert f": {solves} {noun} in " in window.statusBar().currentMessage()
