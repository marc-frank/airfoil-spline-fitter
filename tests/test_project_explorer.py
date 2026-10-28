"""The project explorer: a filterable, sortable table of every run, whose
filtered rows the comparison plot shows and whose rows apply their settings."""
import os

import numpy as np
import pytest
from bspline_varpro.airfoil import NACA4
from PyQt6.QtCore import Qt

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

    # Every word must match
    table.filter_column.setCurrentText("Control points")
    table.filter_edit.setText("> 9 <13")
    assert _column(window, "Control points") == ["10", "12"]

    # A comparison in one column only
    table.filter_edit.setText("<1e-4")
    table.filter_column.setCurrentText("Error")
    assert _column(window, "Control points") == ["14", "12"]

    table.filter_edit.clear()
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
