"""The Curve settings always form a combination that can be fitted: the
setting just changed wins, and the degree, control points and continuity
follow it in either direction."""
import os

import pytest

from spline_fitter.optimization.models import (
    ContinuityMode,
    OptimizerType,
    minimum_control_points,
)


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


def test_showing_the_window_leaves_nothing_to_undo(window):
    """Measuring the sidebar at the first show toggles every sweep: that
    changes no setting, so Edit > Undo stays disabled."""
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui.history import MERGE_MS

    states = [box.sweep_state() for box in window._sweep_group]
    window.show()
    QTest.qWait(MERGE_MS + 200)
    QApplication.processEvents()
    assert not window._action_undo.isEnabled()
    assert [box.sweep_state() for box in window._sweep_group] == states


def test_closed_file_dialogs_are_freed(window, monkeypatch):
    """New NACA airfoil, batch convert and export CAD are freed on close,
    not kept as children of the window, and leave no theme connection."""
    from PyQt6.QtCore import QCoreApplication, QEvent, QTimer
    from PyQt6.QtWidgets import QApplication, QDialog

    from spline_fitter.gui.theme import theme_changed

    real_exec = QDialog.exec
    titles = []

    def exec_closing(dialog):
        titles.append(dialog.windowTitle())
        QTimer.singleShot(0, dialog.reject)
        return real_exec(dialog)

    monkeypatch.setattr(QDialog, "exec", exec_closing)

    def settle():
        QApplication.processEvents()
        # What the main loop does for objects freed with deleteLater
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    settle()
    receivers = theme_changed.receivers(theme_changed.changed)
    # Export CAD opens only with a fit to export
    monkeypatch.setattr(window, "_fit_result", object())
    monkeypatch.setattr(window, "_airfoil", object())
    for _ in range(2):
        window._on_new_naca()
        window._batch_controller.on_batch_dialog()
        window._on_export_cad()
    settle()
    assert titles == ["New NACA airfoil", "Batch convert", "Export CAD"] * 2
    assert not window.findChildren(QDialog)
    assert theme_changed.receivers(theme_changed.changed) == receivers


def _continuity(window, mode):
    window._combo_cont.setCurrentIndex(window._combo_cont.findData(mode))


def _setup(window, mode=ContinuityMode.G0, degree=3, count=6, sections=2):
    _continuity(window, ContinuityMode.G0)
    window._spin_degree.setValue(1)
    window._spin_ctrl.setValue(count)
    window._spin_sections.setValue(sections)
    window._spin_degree.setValue(degree)
    _continuity(window, mode)


def _state(window):
    return (window._combo_cont.currentData(), window._spin_degree.value(),
            window._spin_ctrl.value())


def test_minimum_control_points_follows_the_library():
    from bspline_varpro.constraints import Junction
    from bspline_varpro.constraints import minimum_control_points as library
    for orders in ([], [3], [3, 3], [1, 3, 0], [2, 2, 2, 2]):
        for degree in (1, 3, 5):
            junctions = [Junction(order=o) for o in orders]
            assert minimum_control_points(orders, degree) == max(library(junctions, degree))


def test_the_default_settings_fit(window):
    mode, degree, count = _state(window)
    assert count >= minimum_control_points([mode.order], degree)


def test_raising_the_degree_raises_the_count(window):
    _setup(window, degree=3, count=4)
    window._spin_degree.setValue(5)
    assert _state(window) == (ContinuityMode.G0, 5, 6)


def test_lowering_the_count_lowers_the_degree(window):
    _setup(window, degree=5, count=6)
    window._spin_ctrl.setValue(2)
    assert _state(window) == (ContinuityMode.G0, 1, 2)


def test_lowering_the_count_lowers_the_continuity_last(window):
    _setup(window, ContinuityMode.G3, degree=5, count=8)
    window._spin_ctrl.setValue(4)
    # Degree 3 still carries G3, but its junction needs 5 points: G2 needs 4
    assert _state(window) == (ContinuityMode.G2, 3, 4)


def test_raising_the_continuity_raises_degree_and_count(window):
    _setup(window, degree=1, count=2)
    _continuity(window, ContinuityMode.G3)
    assert _state(window) == (ContinuityMode.G3, 3, 5)


def test_lowering_the_degree_lowers_the_continuity(window):
    _setup(window, ContinuityMode.G3, degree=3, count=6)
    window._spin_degree.setValue(2)
    assert _state(window) == (ContinuityMode.G2, 2, 6)


def test_middle_sections_need_both_junctions(window):
    _setup(window, ContinuityMode.G3, degree=3, count=5, sections=2)
    window._spin_sections.setValue(3)
    assert window._spin_ctrl.value() == 8


def test_per_junction_entries_set_hard_minimums(window):
    window._combo_opt.setCurrentIndex(window._combo_opt.findData(OptimizerType.GN_VARPRO))
    _setup(window, ContinuityMode.G0, degree=5, count=10)
    window._edit_junctions.setText("G3")
    window._edit_junctions.editingFinished.emit()
    assert window._spin_degree.minimum() == 3
    assert window._spin_ctrl.minimum() == 5
    window._spin_ctrl.setValue(2)
    assert _state(window) == (ContinuityMode.G0, 4, 5)


def test_degree_sweep_stops_at_what_a_fixed_count_allows(window):
    _setup(window, degree=3, count=6)
    window._sweep_degree._btn_sweep.click()
    window._sweep_degree._spin_max.setValue(10)
    assert window._sweep_degree.sweep_range == (3, 5)
    # Fewer control points lower the swept degrees with them
    window._spin_ctrl.setValue(4)
    assert window._sweep_degree.sweep_range == (3, 3)


def test_section_sweep_stops_before_middle_sections_a_fixed_count_lacks(window):
    _setup(window, ContinuityMode.G3, degree=3, count=6)
    window._sweep_sections._btn_sweep.click()
    window._sweep_sections._spin_max.setValue(5)
    assert window._sweep_sections.sweep_range == (2, 2)
    window._spin_ctrl.setValue(8)
    window._sweep_sections._spin_max.setValue(5)
    assert window._sweep_sections.sweep_range == (2, 5)


def test_count_swept_with_the_degree_opens_the_limits(window):
    _setup(window, degree=3, count=6)
    window._sweep_degree._btn_sweep.click()
    window._sweep_ctrl._btn_sweep.click()
    window._sweep_degree._spin_max.setValue(10)
    assert window._sweep_degree.sweep_range == (3, 10)
    window._spin_ctrl.setValue(4)
    assert window._spin_ctrl.value() == 4
    # Back to a fixed count: it stays, and the swept degrees follow it
    window._sweep_ctrl._btn_sweep.click()
    assert window._spin_ctrl.value() == 4
    assert window._sweep_degree.sweep_range == (3, 3)


def test_a_grown_tolerance_fit_ignores_the_count(window):
    window._combo_opt.setCurrentIndex(window._combo_opt.findData(OptimizerType.GN_VARPRO))
    window._seg_goal.setCurrentIndex(1)
    window._check_grow.setChecked(True)
    _setup(window, degree=3, count=4)
    window._sweep_degree._btn_sweep.click()
    window._sweep_degree._spin_max.setValue(9)
    assert window._sweep_degree.sweep_range == (3, 9)


def test_applying_a_stored_run_keeps_its_settings(window):
    from types import SimpleNamespace

    from spline_fitter.optimization.models import OptimizationConfig, SplineConfig

    _continuity(window, ContinuityMode.G3)
    cfg = SplineConfig(degree=2, num_control_points=4, continuity=ContinuityMode.G2)
    run = SimpleNamespace(spline_config=cfg, optimization_config=OptimizationConfig())
    window._apply_run_settings(run)
    assert _state(window) == (ContinuityMode.G2, 2, 4)


def test_curve_settings_have_no_fixed_upper_limit(window):
    _setup(window, degree=3, count=6)
    window._spin_ctrl.setValue(5000)
    window._spin_degree.setValue(40)
    window._spin_sections.setValue(500)
    assert (window._spin_ctrl.value(), window._spin_degree.value(),
            window._spin_sections.value()) == (5000, 40, 500)


def test_sections_stop_one_short_of_the_airfoil_points(window, tmp_path):
    path = tmp_path / "five.dat"
    path.write_text("five\n1 0\n0.5 0.05\n0 0\n0.5 -0.05\n1 0\n")
    window._spin_sections.setValue(8)
    window._load_airfoil(str(path))
    assert window._spin_sections.maximum() == 4
    assert window._airfoil.num_sections == window._spin_sections.value() == 4


def test_a_sweep_of_sections_is_open_above_three(window):
    # From three sections on, more sections need no more control points
    _setup(window, mode=ContinuityMode.G1, degree=1, count=4, sections=1)
    window._sweep_sections._btn_sweep.click()
    window._sweep_sections._spin_max.setValue(50)
    assert window._sweep_sections.sweep_range == (1, 50)
