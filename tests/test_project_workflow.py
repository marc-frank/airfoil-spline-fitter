"""The project's file workflow: unsaved runs are not lost without asking, the
explorer follows the open project, and a fit is not mixed into another one."""
import os

import numpy as np
import pytest
from PyQt6.QtWidgets import QMessageBox

from spline_fitter.optimization.models import (
    ContinuityMode,
    FitResult,
    OptimizationConfig,
    OptimizerType,
    SplineConfig,
)
from spline_fitter.project_io import create_project, save_project

Button = QMessageBox.StandardButton


@pytest.fixture
def window(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui.ui import SplineFitterWindow
    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    w._settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(w, "_play_error_chime", lambda: None)
    # The question about unsaved runs answers from this list (and fails the
    # test when asked unexpectedly)
    w.answers = []
    w.asked = 0

    def ask():
        w.asked += 1
        return w.answers.pop(0)
    monkeypatch.setattr(w._project_manager, "_ask_to_save", ask)
    # Deleting runs asks first: yes, unless a test says otherwise
    w.delete_answers = []
    monkeypatch.setattr(w._project_manager, "_confirm_delete",
                        lambda count: w.delete_answers.pop(0) if w.delete_answers else True)
    w.show()
    assert w.new_naca_airfoil("2412", 41)
    yield w
    w._project_manager._project_modified = False  # closes without asking
    w.deleteLater()
    app.processEvents()


def _add(window, ctrl, error=1e-4, sections=2, **opt):
    upper = [(1 - i / (ctrl - 1), 0.05 * np.sin(np.pi * i / (ctrl - 1))) for i in range(ctrl)]
    lower = [(x, -y) for x, y in reversed(upper)]
    curves = [upper, lower] if sections == 2 else [upper + lower[1:]]
    window._project_manager.add_optimization_run(
        SplineConfig(num_control_points=ctrl, num_sections=sections, continuity=ContinuityMode.G2),
        OptimizationConfig(optimizer=OptimizerType.GN_VARPRO, max_iterations=0, **opt),
        FitResult(curves, 5, error, 3, 9, 0.5), [])


def _numbers(window):
    """The Run column, in table order."""
    from spline_fitter.gui.explorer import COLUMN_NAMES
    proxy = window._run_table.proxy
    col = COLUMN_NAMES.index("Run")
    return [proxy.index(r, col).data() for r in range(proxy.rowCount())]


def _saved(window, path):
    """Save the project to ``path``."""
    window._project_manager._project_path = str(path)
    window._project_manager.on_save_project()
    assert not window._project_manager.is_modified


# -- Unsaved runs --------------------------------------------------------------

def test_a_new_airfoil_asks_before_dropping_unsaved_runs(window):
    _add(window, 10)
    window.answers = [Button.Cancel]
    assert not window.new_naca_airfoil("0012")
    assert window._airfoil.name == "NACA 2412"
    assert len(window._project_manager.project.optimization_runs) == 1

    window.answers = [Button.Discard]
    assert window.new_naca_airfoil("0012")
    assert window._project_manager.project is None


def test_saving_when_asked_keeps_the_runs(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    _add(window, 10)
    target = tmp_path / "kept.asfproj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window.answers = [Button.Save]
    assert window.new_naca_airfoil("0012")
    assert target.is_file()

    # A cancelled save cancels the new airfoil as well
    _add(window, 10)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    window.answers = [Button.Save]
    assert not window.new_naca_airfoil("2412")
    assert window._airfoil.name == "NACA 0012"


def test_closing_asks_before_dropping_unsaved_runs(window):
    from PyQt6.QtGui import QCloseEvent
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() and window.asked == 0  # nothing unsaved: no question

    _add(window, 10)
    window.answers = [Button.Cancel]
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()


def test_opening_a_project_asks_only_once_the_file_has_loaded(window, tmp_path):
    broken = tmp_path / "broken.asfproj"
    broken.write_text("{")
    _add(window, 10)
    assert not window.open_path(str(broken))
    assert window.asked == 0

    other = tmp_path / "other.asfproj"
    save_project(create_project(window._airfoil), str(other))
    window.answers = [Button.Cancel]
    assert not window.open_path(str(other))
    assert len(window._project_manager.project.optimization_runs) == 1


# -- The explorer follows the project ---------------------------------------------

def test_a_new_airfoil_empties_the_explorer(window):
    _add(window, 10)
    _add(window, 12)
    old = window._run_table.model.run(0)
    window._run_table.view.selectRow(0)
    window.answers = [Button.Discard]
    window.new_naca_airfoil("0012")
    assert window._run_table.model.rowCount() == 0
    assert not window._btn_delete_run.isEnabled() and not window._btn_load_run.isEnabled()
    # A run of the closed project is not applied
    window._project_manager.on_run_activated(old)
    assert window._fit_result is None and not window._action_save_dat.isEnabled()


def test_a_replaced_run_leaves_the_selection(window):
    _add(window, 10, error=1e-3)
    window._run_table.view.selectRow(0)
    _add(window, 10, error=1e-5)  # the same settings: replaces the selected run
    assert window._project_manager._selected_runs == []
    assert not window._btn_delete_run.isEnabled()


def test_delete_removes_the_selected_runs_only(window, tmp_path):
    _add(window, 10)
    _add(window, 12)
    path = tmp_path / "p.asfproj"
    _saved(window, path)
    assert window.open_path(str(path))
    window._run_table.view.selectRow(0)
    # Reopened, the runs are new objects equal to the old ones; none is selected
    assert window.open_path(str(path))
    assert not window._btn_delete_run.isEnabled()
    window._project_manager.on_delete_selected_runs()
    assert len(window._project_manager.project.optimization_runs) == 2
    assert not window._project_manager.is_modified


def test_deleting_the_shown_run_takes_its_fit_away(window):
    _add(window, 10)
    _add(window, 12)
    window._run_table.view.selectRow(0)
    assert window._fit_result is not None
    window._project_manager.on_delete_selected_runs()
    assert len(window._project_manager.project.optimization_runs) == 1
    assert window._fit_result is None and not window._action_save_dat.isEnabled()


def test_a_project_without_runs_drops_the_previous_fit(window, tmp_path):
    path = tmp_path / "empty.asfproj"
    save_project(create_project(window._airfoil), str(path))
    _add(window, 10)
    window._run_table.view.selectRow(0)
    assert window._fit_result is not None
    window.answers = [Button.Discard]
    assert window.open_path(str(path))
    assert window._fit_result is None and not window._action_save_dat.isEnabled()


# -- During a fit --------------------------------------------------------------------

def test_a_fit_locks_opening_projects_and_changing_runs(window, tmp_path):
    path = tmp_path / "p.asfproj"
    save_project(create_project(window._airfoil), str(path))
    _add(window, 10)
    _add(window, 12)
    window._run_table.view.selectRow(0)
    shown = window._fit_result
    window._set_fitting_ui(True)
    try:
        assert not window._action_open_project.isEnabled()
        assert not window._btn_delete_run.isEnabled() and not window._btn_load_run.isEnabled()
        assert not window._project_manager.open_project(str(path))
        # Selecting shows details but leaves the plots to the fit
        window._run_table.view.selectRow(1)
        assert window._fit_result is shown
        window._project_manager.on_run_activated(window._run_table.model.run(1))
        assert window._fit_result is shown
    finally:
        window._set_fitting_ui(False)
    assert window._action_open_project.isEnabled() and window._btn_delete_run.isEnabled()


# -- Saving ------------------------------------------------------------------------------

def test_save_as_adds_the_project_suffix(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    _add(window, 10)
    target = tmp_path / "myproject"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window._project_manager.on_save_project_as()
    saved = str(target) + ".asfproj"
    assert os.path.isfile(saved) and not target.exists()
    assert window._recent_files()[0] == os.path.abspath(saved)
    assert window.open_path(saved)


def test_a_failed_save_keeps_the_previous_file(window, tmp_path, monkeypatch):
    from spline_fitter import project_io
    _add(window, 10)
    path = tmp_path / "p.asfproj"
    _saved(window, path)
    good = path.read_text()
    _add(window, 12)

    def broken_dump(obj, f, **kw):
        f.write("{")
        raise TypeError("not serializable")
    monkeypatch.setattr(project_io.json, "dump", broken_dump)
    window._project_manager.on_save_project()
    assert path.read_text() == good
    assert not (tmp_path / "p.asfproj.tmp").exists()
    assert window._project_manager.is_modified
    assert "Save failed" in window.statusBar().currentMessage()


def test_save_as_asks_before_replacing_the_file_with_the_suffix(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    _add(window, 10)
    existing = tmp_path / "myproject.asfproj"
    existing.write_text("keep")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "myproject"), ""))
    asked = []
    monkeypatch.setattr(window._project_manager, "_confirm_replace",
                        lambda path: asked.append(path) or False)
    window._project_manager.on_save_project_as()
    assert asked == [str(existing)] and existing.read_text() == "keep"
    assert window._project_manager.project_path is None

    monkeypatch.setattr(window._project_manager, "_confirm_replace", lambda path: True)
    window._project_manager.on_save_project_as()
    assert window._project_manager.project_path == str(existing)
    assert existing.read_text() != "keep"


# -- Opening is all or nothing ------------------------------------------------------------

def _state(window):
    pm = window._project_manager
    return (pm.project, pm.project_path, pm.is_modified, window._airfoil, window._fit_result,
            window._spin_sections.value(), window._spin_ctrl.value(), window._spin_degree.value(),
            window._lbl_file_name.text(), window.windowTitle(),
            [window._run_table.model.run(r) for r in range(window._run_table.model.rowCount())])


def test_a_project_that_cannot_be_shown_changes_nothing(window, tmp_path):
    import json
    _add(window, 10)
    path = tmp_path / "p.asfproj"
    _saved(window, path)
    data = json.loads(path.read_text())
    data["optimization_runs"][0]["timestamp"] = None  # the Date column cannot show it
    broken = tmp_path / "broken.asfproj"
    broken.write_text(json.dumps(data))
    before = _state(window)
    assert not window.open_path(str(broken))
    assert _state(window) == before
    assert "Failed to load project" in window.statusBar().currentMessage()


def test_a_failure_while_showing_a_project_restores_the_open_one(window, tmp_path, monkeypatch):
    _add(window, 10)
    _add(window, 12, sections=1)
    other = tmp_path / "other.asfproj"
    _saved(window, other)
    _add(window, 14)  # unsaved
    window._run_table.view.selectRow(0)
    window._spin_ctrl.setValue(20)
    before = _state(window)

    def fail(run):
        raise RuntimeError("cannot apply")
    monkeypatch.setattr(window, "_apply_run_settings", fail)
    window.answers = [Button.Discard]
    assert not window.open_path(str(other))
    assert _state(window) == before
    assert window._run_table.selected_runs() == [window._run_table.model.run(0)]
    assert "cannot apply" in window.statusBar().currentMessage()


# -- Deleting runs -----------------------------------------------------------------------------

def test_delete_asks_and_the_delete_key_deletes(window):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    _add(window, 10)
    _add(window, 12)
    window._run_table.view.selectRow(0)
    window.delete_answers = [False]
    window._project_manager.on_delete_selected_runs()
    assert len(window._project_manager.project.optimization_runs) == 2

    window.activateWindow()
    window._run_table.view.setFocus()
    QApplication.processEvents()
    QTest.keyClick(window._run_table.view, Qt.Key.Key_Delete)
    assert len(window._project_manager.project.optimization_runs) == 1

    # Not while typing in the filter
    window._run_table.view.selectRow(0)
    window._run_table.filter_edit.setFocus()
    QApplication.processEvents()
    QTest.keyClick(window._run_table.filter_edit, Qt.Key.Key_Delete)
    assert len(window._project_manager.project.optimization_runs) == 1


def test_restore_brings_deleted_runs_back_in_place(window):
    for ctrl in (10, 12, 14, 16):
        _add(window, ctrl)
    runs = list(window._project_manager.project.optimization_runs)
    assert not window._action_restore_runs.isEnabled()
    window._run_table.select_runs([runs[1], runs[3]])
    window._project_manager.on_delete_selected_runs()
    window._run_table.select_runs([runs[0]])
    window._project_manager.on_delete_selected_runs()
    assert window._project_manager.project.optimization_runs == [runs[2]]
    assert window._action_restore_runs.isEnabled()

    window._action_restore_runs.trigger()
    assert window._project_manager.project.optimization_runs == [runs[0], runs[2]]
    window._action_restore_runs.trigger()
    assert all(a is b for a, b in zip(window._project_manager.project.optimization_runs,
                                      runs, strict=True))
    assert window._run_table.selected_runs() == [runs[1], runs[3]]
    assert not window._action_restore_runs.isEnabled()


def test_run_numbers_stay_when_runs_are_deleted(window, tmp_path):
    for ctrl in (10, 12, 14):
        _add(window, ctrl)
    assert _numbers(window) == ["1", "2", "3"]
    window._run_table.select_runs(window._project_manager.project.optimization_runs[1:])
    window._project_manager.on_delete_selected_runs()
    assert _numbers(window) == ["1"]
    _add(window, 16)  # a number not given before: 2 and 3 could be restored
    assert _numbers(window) == ["1", "4"]
    _add(window, 10, error=1e-6)  # replaces run 1, and takes its number
    assert _numbers(window) == ["1", "4"]

    path = tmp_path / "p.asfproj"
    _saved(window, path)
    assert window.open_path(str(path))
    assert _numbers(window) == ["1", "4"]
    _add(window, 18)
    assert _numbers(window) == ["1", "4", "5"]


def test_runs_of_files_without_numbers_are_numbered_in_order(tmp_path):
    import json

    from bspline_varpro.airfoil import NACA4

    from spline_fitter.geometry import create_airfoil_data
    from spline_fitter.project_io import create_optimization_run, load_project
    pts = [tuple(p) for p in np.round(NACA4("2412").points(41), 6)]
    project = create_project(create_airfoil_data(pts, filename="n", name="n", precision=6))
    for ctrl in (10, 12):
        project.optimization_runs.append(create_optimization_run(
            SplineConfig(num_control_points=ctrl),
            OptimizationConfig(optimizer=OptimizerType.GN_VARPRO),
            FitResult([[(0, 0), (1, 1)]], 1, 1.0, 1, 1, 0.1), []))
    path = tmp_path / "old.asfproj"
    save_project(project, str(path))
    data = json.loads(path.read_text())
    for run in data["optimization_runs"]:
        del run["number"]
    path.write_text(json.dumps(data))
    assert [run.number for run in load_project(str(path)).optimization_runs] == [1, 2]


# -- Title, shortcuts ------------------------------------------------------------------------------

def test_the_window_title_marks_unsaved_changes(window, tmp_path):
    assert window.windowTitle() == "Airfoil Spline Fitter" and not window.isWindowModified()
    _add(window, 10)
    assert window.windowTitle() == "Unsaved project[*] - Airfoil Spline Fitter"
    assert window.isWindowModified()
    _saved(window, tmp_path / "p.asfproj")
    assert window.windowTitle() == "p.asfproj[*] - Airfoil Spline Fitter"
    assert not window.isWindowModified()


def test_open_project_has_a_shortcut_of_its_own(window):
    from PyQt6.QtGui import QAction
    assert window._action_open_project.shortcut().toString() == "Ctrl+Shift+O"
    keys = [key.toString() for action in window.findChildren(QAction)
            for key in action.shortcuts() if not key.isEmpty()]
    assert keys.count("Ctrl+Shift+O") == 1


# -- A run of another section count, and tolerance runs ------------------------------------

def test_save_spline_takes_the_continuity_of_the_run_shown(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    from spline_fitter.gui import file_actions
    window._spin_sections.setValue(1)
    _add(window, 10, sections=2)
    window._run_table.view.selectRow(0)
    written = {}
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "s.spline.dat"), ""))
    monkeypatch.setattr(file_actions, "export_spline_data",
                        lambda *a, **k: written.update(k))
    window._on_save_spline()
    run = window._project_manager.project.optimization_runs[0]
    assert written["continuity"] == run.spline_config.junction_specs()
    assert len(written["continuity"]) == 1  # the junction of its two sections


def test_tolerance_runs_ignore_the_smoothing_weight(window):
    _add(window, 10, tolerance=1e-4, smoothing_weight=0.01)
    _add(window, 10, tolerance=1e-4, smoothing_weight=0.02)  # the same fit: replaces it
    assert len(window._project_manager.project.optimization_runs) == 1
    _add(window, 10, smoothing_weight=0.01)
    _add(window, 10, smoothing_weight=0.02)  # a fixed count uses the weight
    assert len(window._project_manager.project.optimization_runs) == 3
