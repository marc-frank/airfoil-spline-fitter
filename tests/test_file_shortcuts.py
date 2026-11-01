"""Opening airfoils without a file dialog (drop, recent files, clipboard, NACA
generator) and undoing settings changes."""
import os

import numpy as np
import pytest
from bspline_varpro.airfoil import NACA4

from spline_fitter.io import parse_airfoil_data, parse_airfoil_text


@pytest.fixture
def window(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui.ui import SplineFitterWindow
    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    # Recent files and dialog defaults go to a file of the test's own
    w._settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(w, "_play_error_chime", lambda: None)
    yield w
    w.deleteLater()
    app.processEvents()


def _write_dat(path, code="2412"):
    pts = np.round(NACA4(code).points(41), 6)
    path.write_text(f"NACA {code}\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in pts))
    return path


def test_text_parses_like_a_file(tmp_path):
    path = _write_dat(tmp_path / "naca2412.dat")
    _, name, coords, precision = parse_airfoil_data(str(path))
    assert parse_airfoil_text(path.read_text()) == (None, name, coords, precision)
    assert name == "NACA 2412" and precision == 6 and len(coords) == 81


def test_recent_files_lists_opened_files_newest_first(window, tmp_path):
    a, b = _write_dat(tmp_path / "a.dat"), _write_dat(tmp_path / "b.dat", "0012")
    assert window.open_path(str(a)) and window.open_path(str(b)) and window.open_path(str(a))
    assert window._recent_files() == [str(a), str(b)]

    window._populate_recent_menu()
    entries = [act.text() for act in window._menu_recent.actions() if act.text()]
    assert entries == ["&1 a.dat", "&2 b.dat", "Clear recent files"]

    # A file that is gone leaves the list when it fails to open
    b.unlink()
    assert not window.open_path(str(b))
    assert window._recent_files() == [str(a)]


def test_dropped_file_opens_wherever_it_lands(window, tmp_path):
    from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PyQt6.QtGui import QDragEnterEvent, QDropEvent
    from PyQt6.QtWidgets import QApplication

    path = _write_dat(tmp_path / "dropped.dat")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    target = window._plot.viewport()  # it ignores the drag, which so reaches the window
    action = Qt.DropAction.CopyAction
    enter = QDragEnterEvent(QPointF(5, 5).toPoint(), action, mime,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(target, enter)
    assert enter.isAccepted()
    QApplication.sendEvent(target, QDropEvent(QPointF(5, 5), action, mime,
                                              Qt.MouseButton.LeftButton,
                                              Qt.KeyboardModifier.NoModifier))
    assert window._airfoil is not None and window._airfoil.name == "NACA 2412"

    # Other files pass by
    other = QMimeData()
    other.setUrls([QUrl.fromLocalFile(str(tmp_path / "notes.txt"))])
    enter = QDragEnterEvent(QPointF(5, 5).toPoint(), action, other,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    enter.ignore()
    QApplication.sendEvent(window._lbl_file_name, enter)
    assert not enter.isAccepted()


def test_paste_airfoil_from_the_clipboard(window):
    from PyQt6.QtWidgets import QApplication

    pts = np.round(NACA4("4412").points(31), 5)
    QApplication.clipboard().setText(
        "NACA 4412 Airfoil M=4.0% P=40.0% T=12.0%\n"
        + "\n".join(f"  {x:.5f}\t{y:.5f}" for x, y in pts))
    window._on_paste_airfoil()
    assert window._airfoil.name.startswith("NACA 4412")
    assert window._airfoil.num_points == len(pts) and window._airfoil.precision == 5

    QApplication.clipboard().setText("Hello, 1 2")
    window._on_paste_airfoil()
    assert window._airfoil.name.startswith("NACA 4412")  # unchanged
    assert "No airfoil coordinates" in window.statusBar().currentMessage()


def test_new_naca_airfoil(window):
    assert window.new_naca_airfoil("0012", n_per_side=51)
    assert window._airfoil.name == "NACA 0012"
    assert window._airfoil.num_points == 101
    assert not window.new_naca_airfoil("12")


def test_undo_and_redo_settings_changes(window):
    history = window._settings_history
    degree, ctrl, sections = window._spin_degree, window._spin_ctrl, window._spin_sections
    before = (degree.value(), ctrl.value(), sections.value())
    assert not window._action_undo.isEnabled()

    degree.setValue(before[0] + 1)
    ctrl.setValue(before[1] + 3)  # within the merge time: one step with the degree
    history.commit()
    sections.setValue(before[2] + 1)
    window._sweep_ctrl._btn_sweep.click()
    history.commit()
    assert window._sweep_ctrl.is_sweep_active
    assert window._action_undo.isEnabled()

    window._action_undo.trigger()
    assert (sections.value(), window._sweep_ctrl.is_sweep_active) == (before[2], False)
    assert degree.value() == before[0] + 1
    window._action_undo.trigger()
    assert (degree.value(), ctrl.value(), sections.value()) == before
    assert not window._action_undo.isEnabled() and window._action_redo.isEnabled()

    window._action_redo.trigger()
    window._action_redo.trigger()
    assert (degree.value(), ctrl.value(), sections.value()) == (
        before[0] + 1, before[1] + 3, before[2] + 1)
    assert window._sweep_ctrl.is_sweep_active

    # A new change after an undo drops the steps that were undone
    window._action_undo.trigger()
    window._spin_smoothing_weight.setValue(window._spin_smoothing_weight.value() * 2)
    history.commit()
    assert not window._action_redo.isEnabled()


def test_automatic_tolerance_of_a_new_airfoil_is_not_a_step(window):
    window._seg_goal.setCurrentIndex(window._seg_goal.findData(True))
    window._check_auto_tolerance.setChecked(True)
    window._settings_history.commit()
    window.new_naca_airfoil("2412")
    tolerance = window._spin_tolerance.value()
    window._settings_history.commit()
    window._action_undo.trigger()  # undoes the goal and auto, not the airfoil's tolerance
    window._action_redo.trigger()
    assert window._spin_tolerance.value() == tolerance


def test_ctrl_z_in_a_focused_spin_box_undoes_the_setting(window):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtWidgets import QApplication

    event = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Z,
                      Qt.KeyboardModifier.ControlModifier)
    event.ignore()
    QApplication.sendEvent(window._spin_degree.lineEdit(), event)
    assert not event.isAccepted()  # left to the Edit > Undo shortcut
