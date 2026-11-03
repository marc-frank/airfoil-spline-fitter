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
    w._project_manager._confirm_delete = lambda count: True
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
    result = FitResult([upper, lower], 5, error, 3, 9, 0.5)
    window._project_manager.add_optimization_run(
        SplineConfig(num_control_points=ctrl, continuity=ContinuityMode.G2),
        OptimizationConfig(optimizer=optimizer, **opt),
        result, [])


def _column(window, name):
    from spline_fitter.gui.explorer import COLUMN_NAMES
    proxy = window._run_table.proxy
    col = COLUMN_NAMES.index(name)
    return [proxy.index(r, col).data() for r in range(proxy.rowCount())]


def test_every_sidebar_setting_and_fit_metric_has_a_column():
    from spline_fitter.gui.explorer import COLUMN_NAMES
    for name in ("Optimizer", "Goal", "Tolerance", "Knots", "Parameterization", "Grow points",
                 "Degree", "Control points", "Sections", "Continuity", "Type",
                 "Per junction", "Subsample", "Smoothing", "Optimize junctions",
                 "Error", "Max deviation", "RMS deviation", "Fitted control points",
                 "Iterations", "Evaluations", "Time", "Converged"):
        assert name in COLUMN_NAMES


def test_filter_invert_and_sort(window):
    for ctrl, error in ((10, 3e-4), (14, 1e-5), (12, 5e-5)):
        _add(window, ctrl, error)
    _add(window, 8, 2e-3, optimizer=OptimizerType.DIRECT_LSQ, parameterization="arc_length")
    table = window._run_table
    assert _column(window, "Run") == ["1", "2", "3", "4"]
    # Blank where the optimizer does not read the setting
    assert _column(window, "Parameterization") == ["", "", "", "Arc length"]

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
    table.set_column_filter("Optimizer", excluded={"Direct LSQ"})
    assert _column(window, "Control points") == ["14", "12"]
    assert table.header.filtered == {_index("Control points"), _index("Error"),
                                     _index("Optimizer")}
    table.invert_button.setChecked(True)
    assert _column(window, "Control points") == ["10", "8"]
    table.invert_button.setChecked(False)

    table.clear_filters()
    assert _column(window, "Control points") == ["10", "14", "12", "8"]
    table.sort_by(_index("Error"), Qt.SortOrder.AscendingOrder)
    assert _column(window, "Control points") == ["14", "12", "10", "8"]


def _index(name):
    from spline_fitter.gui.explorer import COLUMN_NAMES
    return COLUMN_NAMES.index(name)


def test_plot_shows_the_filtered_runs_along_the_chosen_column(window):
    for ctrl, error in ((10, 3e-4), (14, 1e-5), (12, 5e-5)):
        _add(window, ctrl, error)
    _add(window, 8, 2e-3, optimizer=OptimizerType.DIRECT_LSQ)
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


def test_switching_runs_reuses_the_plot_items(window):
    """Switching runs only changes data, also to a run with another section
    count: the plots and details keep their items, and the comparison plot
    only moves its marker."""
    for ctrl in (8, 10):
        _add(window, ctrl, 1e-3 / ctrl)
    runs = window._project_manager.project.optimization_runs
    window._check_comb.setChecked(True)
    table = window._run_table
    table.select_runs([runs[0]])
    plots = window._plot_manager
    items = list(window._plot.getPlotItem().items)
    cells = [label for pair in window._run_details._cells for label in pair]
    curve_data = plots._comparison_curve.getData()

    table.select_runs([runs[1]])
    assert window._plot.getPlotItem().items == items
    assert [label for pair in window._run_details._cells for label in pair] == cells
    # Both sections' control polygons, a gap between them
    polygon = plots._polygon_item.getData()[0]
    assert len(polygon) == 2 * 10 + 1 and np.isnan(polygon[10])
    assert plots._comparison_curve.getData()[0] is curve_data[0]
    assert list(plots._comparison_marker.getData()[0]) == [10]

    # A run with one section
    upper, lower = runs[1].result.section_control_points
    window._project_manager.add_optimization_run(
        SplineConfig(num_control_points=19, num_sections=1, continuity=ContinuityMode.G2),
        OptimizationConfig(optimizer=OptimizerType.GN_VARPRO),
        FitResult([upper + lower[1:]], 5, 1e-6, 3, 9, 0.5), [])
    table.select_runs([runs[2]])
    assert window._plot.getPlotItem().items == items
    polygon = plots._polygon_item.getData()[0]
    assert len(polygon) == 19 and not np.isnan(polygon).any()

    # A deleted run leaves the plot
    window._project_manager.on_delete_selected_runs()
    assert list(plots._comparison_curve.getData()[0]) == [8, 10]


def test_moving_the_marker_keeps_the_comparison_range(window):
    """The marker is larger than the trace's dots: on a run at the end of the
    trace it must not widen the plot (and redraw its axes)."""
    for ctrl in (8, 10, 12):
        _add(window, ctrl, 1e-3 / ctrl)
    runs = window._project_manager.project.optimization_runs
    view_box = window._comparison_plot.getPlotItem().getViewBox()
    ranges = []
    for run in (runs[1], runs[0], runs[2]):
        window._run_table.select_runs([run])
        ranges.append(view_box.viewRange())
    assert ranges[0] == ranges[1] == ranges[2]


def test_the_table_scrolls_without_repainting_all_of_it(window, tmp_path, monkeypatch):
    """Filled with its background, the table is opaque, so Qt scrolls it by
    moving its pixels; the stylesheet turns the fill off, also on a new theme."""
    from PyQt6.QtCore import QSettings

    from spline_fitter.gui import theme
    viewport = window._run_table.view.viewport()
    assert viewport.autoFillBackground()
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(theme, "_settings", lambda: QSettings(ini, QSettings.Format.IniFormat))
    before = (theme.Scheme.MODE, theme.Scheme.TEMPERATURE)
    try:
        theme.set_theme("light" if theme.Scheme.DARK else "dark")
        QApplication.processEvents()
        assert viewport.autoFillBackground()
        assert viewport.palette().base().color().name().lower() == theme.Palette.PANEL.lower()
    finally:
        theme.set_theme(*before)


def test_double_click_applies_the_run_settings(window):
    _add(window, 11, 1e-4, optimizer=OptimizerType.DIRECT_LSQ, parameterization="arc_length")
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


def test_load_settings_button_and_details_spread_with_the_width(window):
    _add(window, 11, 1e-4, optimizer=OptimizerType.DIRECT_LSQ)
    _add(window, 13, 1e-5)
    table, details = window._run_table, window._run_details
    assert not window._btn_load_run.isEnabled()

    table.view.selectAll()
    assert not window._btn_load_run.isEnabled()
    assert "2 runs selected" in details.text()

    table.view.selectRow(0)
    run = table.selected_runs()[0]
    assert window._btn_load_run.isEnabled()
    assert "Control points" in details.text() and "Converged" in details.text()
    window._btn_load_run.click()
    assert window._get_optimization_config() == run.optimization_config

    # All pairs in one line when wide, one per line when narrow
    details.resize(10_000, details.height())
    assert details.columns() == len(details._rows)
    details.resize(10, details.height())
    assert details.columns() == 1
    details.resize(details._widths(2), details.height())
    assert details.columns() == 2


def test_header_drop_down_filters_by_values_and_condition(window):
    for ctrl, error in ((10, 3e-4), (14, 1e-5), (12, 5e-5)):
        _add(window, ctrl, error)
    _add(window, 8, 2e-3, optimizer=OptimizerType.DIRECT_LSQ)
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
        "Direct LSQ", "VarPro LM"]
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


def test_pivot_keeps_selection_filters_and_sorting(window):
    for ctrl, error in ((10, 3e-4), (14, 1e-5), (12, 5e-5)):
        _add(window, ctrl, error)
    _add(window, 8, 2e-3, optimizer=OptimizerType.DIRECT_LSQ)
    runs = window._project_manager.project.optimization_runs
    table = window._run_table
    table.set_column_filter("Optimizer", excluded={"Direct LSQ"})
    table.sort_by(_index("Error"), Qt.SortOrder.AscendingOrder)
    table.header.moveSection(table.header.visualIndex(_index("Error")), 1)
    table.select_runs([runs[0], runs[2]])

    table.pivot_button.setChecked(True)
    view = table.view
    # One column per run that passes the filter, in sort order; one row per setting
    assert view.model().columnCount() == 3
    assert [view.model().index(_index("Run"), c).data() for c in range(3)] == ["2", "3", "1"]
    assert not view.horizontalHeader().isVisible()
    assert view.model().index(0, 0).data(Qt.ItemDataRole.TextAlignmentRole) == (
        Qt.AlignmentFlag.AlignCenter)
    row = view.model().index(_index("Control points"), 0).row()
    assert [view.model().index(row, c).data() for c in range(3)] == ["14", "12", "10"]
    # The settings' header is now the row names, with the same state
    header = table.header
    assert header.orientation() == Qt.Orientation.Vertical and view.verticalHeader() is header
    assert header.filtered == {_index("Optimizer")}
    assert header.sortIndicatorSection() == _index("Error")
    assert header.visualIndex(_index("Error")) == 1
    assert table.selected_runs() == [runs[2], runs[0]]

    # Sorting, filtering and double-clicking work along the rows
    table.sort_by(_index("Control points"), Qt.SortOrder.DescendingOrder)
    assert [view.model().index(row, c).data() for c in range(3)] == ["14", "12", "10"]
    table.set_column_filter("Control points", "<13")
    assert [view.model().index(row, c).data() for c in range(2)] == ["12", "10"]
    view.doubleClicked.emit(view.model().index(0, 1))
    assert window._fit_result is runs[0].result
    popup = table.open_filter(_index("Degree"))
    assert popup.values.count() == 1
    popup.close()
    # The plot still follows the filter
    assert list(window._plot_manager._comparison_curve.getData()[0]) == [10, 12]

    # And back
    table.select_runs([runs[2]])
    table.pivot_button.setChecked(False)
    assert view.horizontalHeader() is table.header
    assert table.header.sortIndicatorSection() == _index("Control points")
    assert table.header.filtered == {_index("Optimizer"), _index("Control points")}
    assert _column(window, "Control points") == ["12", "10"]
    assert table.selected_runs() == [runs[2]]


def test_clear_filter_drops_a_typed_condition(window):
    for ctrl, error in ((10, 3e-4), (14, 1e-5), (12, 5e-5)):
        _add(window, ctrl, error)
    _add(window, 8, 2e-3, optimizer=OptimizerType.DIRECT_LSQ)
    table = window._run_table
    popup = table.open_filter(_index("Control points"))
    popup.condition.setText(">11")
    assert _column(window, "Control points") == ["14", "12"]
    popup._clear()
    QApplication.processEvents()
    assert _column(window, "Control points") == ["10", "14", "12", "8"]
    assert table.column_filters == {} and table.header.filtered == set()


def test_global_number_comparison_compares_the_error(window):
    """The run number or the degree would pass nearly every run: a number
    comparison in the global filter compares the error; words still match
    any column."""
    for ctrl, error in ((10, 3e-4), (14, 1e-5), (12, 5e-5)):
        _add(window, ctrl, error)
    _add(window, 8, 2e-3, optimizer=OptimizerType.DIRECT_LSQ)
    table = window._run_table
    table.filter_edit.setText(">1e-5")
    assert _column(window, "Control points") == ["10", "12", "8"]
    table.filter_edit.setText("<=3e-4")
    assert _column(window, "Control points") == ["10", "14", "12"]
    table.filter_edit.setText("varpro <1e-4")
    assert _column(window, "Control points") == ["14", "12"]
    table.filter_edit.setText("=free")
    assert _column(window, "Control points") == ["10", "14", "12"]


@pytest.mark.parametrize("order", [Qt.SortOrder.AscendingOrder, Qt.SortOrder.DescendingOrder])
def test_blanks_sort_last_in_either_order(window, order):
    for ctrl, error in ((10, 3e-4), (14, 1e-5)):
        _add(window, ctrl, error)
    _add(window, 8, 2e-3, optimizer=OptimizerType.DIRECT_LSQ)
    _add(window, 12, 5e-5)
    window._run_table.sort_by(_index("Parameterization"), order)
    assert _column(window, "Parameterization")[0] != ""
    window._run_table.sort_by(_index("Knots"), order)
    assert _column(window, "Knots")[-1] == ""
    window._run_table.sort_by(_index("Control points"), order)
    numbers = [int(c) for c in _column(window, "Control points")]
    assert numbers == sorted(numbers, reverse=order == Qt.SortOrder.DescendingOrder)


def test_a_closed_filter_drop_down_leaves_no_theme_connection(window):
    from PyQt6.QtCore import QCoreApplication, QEvent

    from spline_fitter.gui.theme import theme_changed

    _add(window, 10, 3e-4)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    receivers = theme_changed.receivers(theme_changed.changed)
    for _ in range(3):
        popup = window._run_table.open_filter(_index("Control points"))
        assert theme_changed.receivers(theme_changed.changed) == receivers + 1
        popup.close()  # deleted on close
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert theme_changed.receivers(theme_changed.changed) == receivers
