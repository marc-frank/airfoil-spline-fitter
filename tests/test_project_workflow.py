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
    w.show()
    assert w.new_naca_airfoil("2412", 41)
    yield w
    w._project_manager._project_modified = False  # closes without asking
    w.deleteLater()
    app.processEvents()


def _add(window, ctrl, error=1e-4):
    upper = [(1 - i / (ctrl - 1), 0.05 * np.sin(np.pi * i / (ctrl - 1))) for i in range(ctrl)]
    lower = [(x, -y) for x, y in reversed(upper)]
    window._project_manager.add_optimization_run(
        SplineConfig(num_control_points=ctrl, continuity=ContinuityMode.G2),
        OptimizationConfig(optimizer=OptimizerType.GN_VARPRO, max_iterations=0),
        FitResult([upper, lower], 5, error, 3, 9, 0.5), [])


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
