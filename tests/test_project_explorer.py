"""The project explorer: a filterable, sortable table of every run, whose
filtered rows the comparison plot shows and whose rows apply their settings."""
import os

import numpy as np
import pytest
from bspline_varpro.airfoil import NACA4
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from spline_fitter.geometry import create_airfoil_data
from spline_fitter.optimization.models import (
    ContinuityMode,
    ErrorMetric,
    FairnessWeights,
    FitResult,
    OptimizationConfig,
    OptimizerType,
    SplineConfig,
)


@pytest.fixture
def window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui.ui import SplineFitterWindow
    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    w.show()
    pts = np.round(NACA4("2412").points(81), 6)
    w._airfoil = create_airfoil_data([tuple(p) for p in pts], filename="n", name="n",
                                     precision=6, num_sections=2)
    yield w
    w.deleteLater()
    app.processEvents()


def _add(window, ctrl, error, optimizer=OptimizerType.GN_VARPRO, **opt):
    upper = [(1 - i / (ctrl - 1), 0.05 * np.sin(np.pi * i / (ctrl - 1))) for i in range(ctrl)]
    lower = [(x, -y) for x, y in reversed(upper)]
    if optimizer == OptimizerType.GN_VARPRO:
        opt.setdefault("max_iterations", 0)  # VarPro's own limit, as the sidebar stores it
    result = FitResult([upper, lower], 5, error, 3, 9, 0.5)
    window._project_manager.add_optimization_run(
        SplineConfig(num_control_points=ctrl, continuity=ContinuityMode.G2),
        OptimizationConfig(optimizer=optimizer, **opt),
        FairnessWeights(), result, [])


def _column(window, name):
    from spline_fitter.gui.explorer import COLUMN_NAMES
    proxy = window._run_table.proxy
    col = COLUMN_NAMES.index(name)
    return [proxy.index(r, col).data() for r in range(proxy.rowCount())]


def test_every_sidebar_setting_and_fit_metric_has_a_column():
    from spline_fitter.gui.explorer import COLUMN_NAMES
    for name in ("Optimizer", "Goal", "Tolerance", "Knots", "Parameterization", "Grow points",
                 "Error metric", "Degree", "Control points", "Sections", "Continuity", "Type",
                 "Per junction", "Subsample", "Smoothing", "Iteration limit", "Seed",
                 "Search scale", "Optimize junctions", "Even edges", "Polygon bend",
                 "Bunching", "End handles", "Bending", "Wiggle",
                 "Error", "Max deviation", "RMS deviation", "Fitted control points",
                 "Iterations", "Evaluations", "Time", "Converged"):
        assert name in COLUMN_NAMES


def test_filter_invert_and_sort(window):
    for ctrl, error in ((10, 3e-4), (14, 1e-5), (12, 5e-5)):
        _add(window, ctrl, error)
    _add(window, 8, 2e-3, optimizer=OptimizerType.DE, seed=4)
    table = window._run_table
    assert _column(window, "Run") == ["1", "2", "3", "4"]
    # Blank where the optimizer does not read the setting
    assert _column(window, "Seed") == ["", "", "", "4"]

    table.filter_edit.setText("varpro")
    assert _column(window, "Control points") == ["10", "14", "12"]
    table.invert_button.setChecked(True)
    assert _column(window, "Control points") == ["8"]
    table.invert_button.setChecked(False)

    # A column's own condition; every word must match
    table.filter_edit.clear()
    table.set_column_filter("Control points", "> 9 <13")
    assert _column(window, "Control points") == ["10", "12"]

    # Several column filters and the global filter at once
    table.set_column_filter("Control points", ">9")
    table.set_column_filter("Error", "<1e-4")
    assert _column(window, "Control points") == ["14", "12"]
    table.filter_edit.setText("varpro")
    table.set_column_filter("Optimizer", excluded={"Differential evolution"})
    assert _column(window, "Control points") == ["14", "12"]
    assert table.header.filtered == {_index("Control points"), _index("Error"),
                                     _index("Optimizer")}
    table.invert_button.setChecked(True)
    assert _column(window, "Control points") == ["10", "8"]
    table.invert_button.setChecked(False)

    table.clear_filters()
    assert _column(window, "Control points") == ["10", "14", "12", "8"]
    table.view.sortByColumn(_index("Error"), Qt.SortOrder.AscendingOrder)
    assert _column(window, "Control points") == ["14", "12", "10", "8"]


def _index(name):
    from spline_fitter.gui.explorer import COLUMN_NAMES
    return COLUMN_NAMES.index(name)


def test_plot_shows_the_filtered_runs_along_the_chosen_column(window):
    for ctrl, error in ((10, 3e-4), (14, 1e-5), (12, 5e-5)):
        _add(window, ctrl, error)
    _add(window, 8, 2e-3, optimizer=OptimizerType.DE)
    curve = window._plot_manager._comparison_curve
    assert list(curve.getData()[0]) == [8, 10, 12, 14]

    window._run_table.filter_edit.setText("varpro")
    assert list(curve.getData()[0]) == [10, 12, 14]
    window._run_table.invert_button.setChecked(True)
    assert list(curve.getData()[0]) == [8]
    window._run_table.invert_button.setChecked(False)

    # A text column: one position per value, named by ticks
    window._combo_runs_x.setCurrentText("Optimizer")
    assert list(curve.getData()[0]) == [0, 0, 0]
    window._run_table.filter_edit.clear()
    assert sorted(set(curve.getData()[0])) == [0, 1]
    # The y axis takes any fit metric; the header menu picks columns too
    window._run_table.plot_column_requested.emit("x", "Control points")
    window._run_table.plot_column_requested.emit("y", "Evaluations")
    assert window._combo_runs_x.currentText() == "Control points"
    assert list(curve.getData()[1]) == [9, 9, 9, 9]

    # Selected runs are marked
    window._run_table.select_runs([window._project_manager.project.optimization_runs[1]])
    marker = window._plot_manager._comparison_marker
    assert list(marker.getData()[0]) == [14]


def test_double_click_applies_the_run_settings(window):
    _add(window, 11, 1e-4, optimizer=OptimizerType.DE, seed=7, max_iterations=500,
         metric=ErrorMetric.ORTHOGONAL)
    _add(window, 13, 1e-5, smoothing_weight=0.02, knot_placement="free")
    runs = window._project_manager.project.optimization_runs
    table = window._run_table
    for run in runs:
        row = next(r for r in range(table.proxy.rowCount())
                   if table.model.run(table.proxy.mapToSource(table.proxy.index(r, 0)).row())
                   is run)
        table.view.doubleClicked.emit(table.proxy.index(row, 0))
        assert window._get_optimization_config() == run.optimization_config
        spline = window._get_spline_config()
        assert (spline.num_control_points, spline.continuity) == (
            run.spline_config.num_control_points, run.spline_config.continuity)
        assert window._fit_result is run.result


def test_header_drop_down_filters_by_values_and_condition(window):
    for ctrl, error in ((10, 3e-4), (14, 1e-5), (12, 5e-5)):
        _add(window, ctrl, error)
    _add(window, 8, 2e-3, optimizer=OptimizerType.DE)
    table = window._run_table
    popup = table.open_filter(_index("Control points"))
    labels = [popup.values.item(i).text() for i in range(popup.values.count())]
    assert labels == ["8", "10", "12", "14"]  # by value, not by text

    # Unchecking values filters them out; Select all brings them back
    popup.values.item(0).setCheckState(Qt.CheckState.Unchecked)
    popup.values.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert _column(window, "Control points") == ["14", "12"]
    assert popup.select_all.checkState() == Qt.CheckState.PartiallyChecked
    popup.select_all.click()
    assert _column(window, "Control points") == ["10", "14", "12", "8"]
    def checked():
        return [popup.values.item(i).text() for i in range(popup.values.count())
                if popup.values.item(i).checkState() == Qt.CheckState.Checked]

    # A typed condition previews: what it rules out shows unchecked
    popup.condition.setText("<12")
    assert checked() == ["8", "10"]
    assert _column(window, "Control points") == ["10", "8"]
    # Emptying the field by hand drops it
    popup.condition.clear()
    assert checked() == ["8", "10", "12", "14"]
    # The field's clear button keeps it, as does Enter
    from PyQt6.QtGui import QAction
    from PyQt6.QtTest import QTest
    popup.condition.setText(">9")
    popup.condition.findChild(QAction, "_q_qlineeditclearaction").trigger()
    assert popup.condition.text() == "" and checked() == ["10", "12", "14"]
    popup.condition.setText("<13")
    QTest.keyClick(popup.condition, Qt.Key.Key_Return)
    assert popup.condition.text() == "" and checked() == ["10", "12"]
    assert popup.isVisible()
    popup.select_all.click()
    assert checked() == ["8", "10", "12", "14"]

    # Enter keeps it, and the next condition narrows from there
    popup.condition.setText("<12")
    popup.keep()
    assert popup.condition.text() == "" and checked() == ["8", "10"]
    popup.condition.setText(">9")
    assert checked() == ["10"]
    popup.keep()
    assert checked() == ["10"]
    # A value outside the condition can be checked by hand
    popup.values.item(3).setCheckState(Qt.CheckState.Checked)
    assert checked() == ["10", "14"]
    assert _column(window, "Control points") == ["10", "14"]
    # Add matching checks what a condition matches, leaving the rest
    popup.condition.setText("=8")
    popup.keep(add=True)
    assert checked() == ["8", "10", "14"]
    # Clicking a value while a condition is typed keeps the condition first
    popup.condition.setText(">9")
    popup.values.item(2).setCheckState(Qt.CheckState.Checked)
    assert popup.condition.text() == "" and checked() == ["10", "12", "14"]
    # So does closing; the filter stays when the drop-down opens again
    popup.condition.setText("<13")
    popup.close()
    popup = table.open_filter(_index("Control points"))
    assert checked() == ["10", "12"]
    popup.close()
    table.set_column_filter("Control points")
    table.set_column_filter("Control points", "<12")

    # The values offered are those the other filters let through
    popup = table.open_filter(_index("Optimizer"))
    assert [popup.values.item(i).text() for i in range(popup.values.count())] == [
        "Differential evolution", "VarPro LM"]
    popup.values.item(1).setCheckState(Qt.CheckState.Unchecked)
    popup.close()
    assert _column(window, "Control points") == ["8"]

    # The header's glyph opens the drop-down instead of sorting
    header = table.header
    col = _index("Degree")
    x = header.sectionViewportPosition(col) + header.sectionSize(col) - 10
    opened = []
    header.filter_requested.connect(opened.append)
    from PyQt6.QtCore import QPoint
    from PyQt6.QtTest import QTest
    QTest.mouseClick(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(x, 8))
    assert opened == [col]


def test_drop_down_of_the_last_column_stays_in_the_window(window, monkeypatch):
    from PyQt6.QtCore import QPoint, QRect
    _add(window, 10, 1e-4)
    table, header = window._run_table, window._run_table.header

    class Screen:  # Room for the whole window
        def availableGeometry(self):
            return QRect(0, 0, 5000, 3000)
    monkeypatch.setattr(table, "screen", lambda: Screen())
    window.resize(2000, 900)
    window._main_splitter.setSizes([330, 770, 900])
    QApplication.processEvents()

    def edges(col):
        """A section's left and right edges on the screen, as far as it shows."""
        x = header.sectionViewportPosition(col)
        right = min(x + header.sectionSize(col), header.width())
        return header.mapToGlobal(QPoint(x, 0)).x(), header.mapToGlobal(QPoint(right, 0)).x()

    # The last column's opens from its right edge, inside the window
    last = _index("Date")
    table.view.scrollTo(table.proxy.index(0, last))
    pos = table.drop_position(last, 300)
    assert pos.x() + 300 == edges(last)[1] <= window.frameGeometry().right() + 1
    # The first column's from its left edge
    table.view.horizontalScrollBar().setValue(0)
    assert table.drop_position(0, 300).x() == edges(0)[0]

    # And neither leaves the screen
    monkeypatch.setattr(table, "screen", lambda: type(
        "Small", (), {"availableGeometry": lambda self: QRect(0, 0, 400, 300)})())
    assert table.drop_position(0, 300).x() + 300 <= 400
