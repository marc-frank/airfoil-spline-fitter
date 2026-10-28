"""
Project Explorer
================

The project explorer is a table of every optimization run ever done in the
open project, one row per run:

- **Columns.** Every setting of the sidebar has its own column (method,
  curve, advanced and regularization, in sidebar order), followed by the
  fit metrics of the run (error, deviations, fitted control points,
  iterations, evaluations, time, convergence) and its date. A setting
  the run's optimizer does not read stays blank, as the sidebar hides it.
- **Filter.** One filter field narrows the rows, in all columns or in the
  one chosen beside it. Every word must match: plain text matches a part
  of the cell, a number with a comparison (``>1e-5``, ``<=12``, ``=5``,
  ``!=3``) compares the value. *Invert* shows the rows the filter does not
  match instead.
- **Sorting.** Clicking a header sorts by that column, numbers by value.
- **Plot.** The error plot at the lower right shows only the rows that pass
  the filter: any column can become its x axis, and any fit metric its y
  axis (in the plot's toolbar, or from a column header's context menu).
  Selected runs are marked on it.
- **Apply.** Double-clicking a row applies that run's settings to the
  sidebar, so it can be refitted or varied.

This module holds the table itself (columns, model, filter, view);
:class:`~spline_fitter.gui.project.ProjectManager` fills it and connects
it to the plot and the sidebar.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMenu,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..formatting import decimal_point, format_duration, number
from ..optimization.models import OptimizationRun, OptimizerType
from .theme import Size, mono_font

# =============================================================================
# Columns
# =============================================================================


@dataclass(frozen=True)
class Column:
    """One column of the run table.

    ``value`` reads the raw value of a run (a number, text or bool; None
    when it has none), ``applies`` whether the run's optimizer reads the
    setting at all. ``spec`` formats numbers (a ``formatting.number`` spec,
    or ``"duration"``); ``log`` plots the column on a log axis when all its
    values are positive; ``metric`` marks the fit metrics, which the plot
    can take as its y axis.
    """
    name: str
    value: Callable[[OptimizationRun], object]
    spec: str = ""
    log: bool = False
    metric: bool = False
    applies: Callable[[OptimizationRun], bool] = lambda run: True
    tip: str = ""

    def cell(self, run: OptimizationRun):
        """The run's value in this column, None where it does not apply."""
        return self.value(run) if self.applies(run) else None

    def text(self, value) -> str:
        """The value as shown in the table."""
        if value is None:
            return ""
        if isinstance(value, bool):
            return "yes" if value else "no"
        if self.spec == "duration":
            return format_duration(value)
        if isinstance(value, float) and self.spec:
            return number(value, self.spec)
        return str(value)


def is_number(value) -> bool:
    """A value that compares and plots as a number (bools do not)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _varpro(run: OptimizationRun) -> bool:
    return run.optimization_config.optimizer == OptimizerType.GN_VARPRO


def _tolerance_goal(run: OptimizationRun) -> bool:
    return _varpro(run) and run.optimization_config.tolerance is not None


def _uses_metric(run: OptimizationRun) -> bool:
    return run.optimization_config.optimizer.uses_error_metric


def _iterates(run: OptimizationRun) -> bool:
    opt = run.optimization_config.optimizer
    return not opt.is_algebraic and opt != OptimizerType.GN_VARPRO


def _has_junctions(run: OptimizationRun) -> bool:
    return run.spline_config.num_sections > 1


def _report(key: str) -> Callable[[OptimizationRun], float | None]:
    return lambda run: (run.result.report or {}).get(key)


def _fairness(name: str) -> Callable[[OptimizationRun], float]:
    return lambda run: getattr(run.fairness_weights, name)


def _cfg(run):
    return run.spline_config


def _opt(run):
    return run.optimization_config


COLUMNS: tuple[Column, ...] = (
    Column("Run", lambda run: run.index, tip="Order in which the runs were done"),
    # Method
    Column("Optimizer", lambda run: _opt(run).optimizer.display_name),
    Column("Goal", lambda run: "Tolerance" if _opt(run).tolerance is not None else "Fixed count",
           applies=_varpro),
    Column("Tolerance", lambda run: _opt(run).tolerance, ".3g", log=True,
           applies=_tolerance_goal, tip="Max deviation the tolerance goal had to meet"),
    Column("Knots", lambda run: _opt(run).knot_placement.capitalize(), applies=_varpro),
    Column("Parameterization",
           lambda run: _opt(run).parameterization.replace("_", " ").capitalize(),
           applies=lambda run: _opt(run).optimizer == OptimizerType.DIRECT_LSQ),
    Column("Grow points", lambda run: _opt(run).grow_control_points, applies=_varpro),
    Column("Error metric", lambda run: _opt(run).metric.display_name, applies=_uses_metric),
    # Curve
    Column("Degree", lambda run: _cfg(run).degree),
    Column("Control points", lambda run: _cfg(run).num_control_points,
           tip="Control points per section as set (the start count of a tolerance goal)"),
    Column("Sections", lambda run: _cfg(run).num_sections),
    Column("Continuity", lambda run: _cfg(run).continuity.name, applies=_has_junctions),
    Column("Type", lambda run: "Geometric" if _cfg(run).geometric_continuity else "Parametric",
           applies=lambda run: _varpro(run) and _has_junctions(run)),
    # Advanced
    Column("Per junction", lambda run: ", ".join(_cfg(run).junction_continuity),
           applies=_varpro),
    Column("Subsample", lambda run: _opt(run).max_points_per_section, applies=_varpro,
           tip="Points per section of the subsample; blank when every point is used"),
    Column("Smoothing", lambda run: _opt(run).smoothing_weight, ".3g", log=True,
           applies=lambda run: _varpro(run) and not _tolerance_goal(run)),
    Column("Iteration limit", lambda run: _opt(run).max_iterations, applies=_iterates),
    Column("Seed", lambda run: _opt(run).seed,
           applies=lambda run: _opt(run).optimizer.is_global),
    Column("Search scale", lambda run: _cfg(run).bbox_scale, ".3g", applies=_uses_metric),
    Column("Optimize junctions", lambda run: _opt(run).optimize_le_position,
           applies=_has_junctions),
    # Regularization
    Column("Even edges", _fairness("length_variance"), ".4g", applies=_uses_metric),
    Column("Polygon bend", _fairness("polygon_bend"), ".4g", applies=_uses_metric),
    Column("Bunching", _fairness("anti_bunching"), ".4g", applies=_uses_metric),
    Column("End handles", _fairness("end_handles"), ".4g", applies=_uses_metric),
    Column("Bending", _fairness("curvature_squared"), ".4g", applies=_uses_metric),
    Column("Wiggle", _fairness("curvature_gradient"), ".4g", applies=_uses_metric),
    # Fit metrics
    Column("Error", lambda run: run.result.final_error, ".3e", log=True, metric=True),
    Column("Max deviation", _report("max_deviation"), ".3e", log=True, metric=True),
    Column("RMS deviation", _report("rms_deviation"), ".3e", log=True, metric=True),
    Column("Fitted control points", lambda run: run.result.num_control_points, metric=True,
           tip="Distinct control points of the fitted chain, junction points counted once"),
    Column("Iterations", lambda run: run.result.num_iterations, metric=True),
    Column("Evaluations", lambda run: run.result.num_evaluations, metric=True),
    Column("Time", lambda run: run.result.elapsed_seconds, "duration", log=True, metric=True),
    Column("Converged", lambda run: run.result.converged),
    Column("Date", lambda run: run.timestamp[:16].replace("T", " ")),
)

COLUMN_NAMES = [c.name for c in COLUMNS]


def column(name: str) -> Column:
    """The column of that name."""
    return COLUMNS[COLUMN_NAMES.index(name)]


# =============================================================================
# Model
# =============================================================================


@dataclass(frozen=True)
class _Row:
    """A run with its number in the project, which the Run column shows."""
    run: OptimizationRun
    index: int

    def __getattr__(self, name):
        return getattr(self.run, name)


class RunTableModel(QAbstractTableModel):
    """The runs of a project, one per row, one column per ``COLUMNS`` entry.

    The display role holds the formatted text, the user role the raw value
    (for sorting, filtering and plotting).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[_Row] = []
        self._cells: list[list] = []

    def set_runs(self, runs: list[OptimizationRun]) -> None:
        self.beginResetModel()
        self._rows = [_Row(run, i + 1) for i, run in enumerate(runs)]
        self._cells = [[c.cell(row) for c in COLUMNS] for row in self._rows]
        self.endResetModel()

    def run(self, row: int) -> OptimizationRun:
        return self._rows[row].run

    def value(self, row: int, col: int):
        return self._cells[row][col]

    def rowCount(self, parent=None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._rows)

    def columnCount(self, parent=None) -> int:
        return 0 if parent is not None and parent.isValid() else len(COLUMNS)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._cells[index.row()][index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[index.column()].text(value)
        if role == Qt.ItemDataRole.UserRole:
            return value
        # Numbers line up: monospaced and right-aligned
        if role == Qt.ItemDataRole.FontRole and is_number(value):
            return mono_font()
        if role == Qt.ItemDataRole.TextAlignmentRole:
            horizontal = (Qt.AlignmentFlag.AlignRight if is_number(value)
                          else Qt.AlignmentFlag.AlignLeft)
            return horizontal | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section].name
        if role == Qt.ItemDataRole.ToolTipRole:
            return COLUMNS[section].tip or None
        return None


# =============================================================================
# Filter
# =============================================================================

# A comparison with a number: >1e-5, <=12, =5, !=3
_COMPARISON = re.compile(r"^(<=|>=|!=|<|>|=)(.+)$")
_OPERATORS = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
    "=": lambda a, b: a == b, "!=": lambda a, b: a != b,
}


def _parse_number(text: str) -> float | None:
    """A number typed with the locale's decimal point or a dot."""
    try:
        return float(text.replace(decimal_point(), "."))
    except ValueError:
        return None


def parse_filter(text: str) -> list[tuple[str, object]]:
    """The words of a filter as (operator, operand): ("", text) matches a part
    of the cell, an operator with a number compares the value. A space after
    the operator is allowed (``> 10``)."""
    text = re.sub(r"(<=|>=|!=|<|>|=)\s+", r"\1", text.strip())
    terms = []
    for word in text.split():
        m = _COMPARISON.match(word)
        operand = _parse_number(m.group(2)) if m else None
        if m and operand is not None:
            terms.append((m.group(1), operand))
        elif m and m.group(1) in ("=", "!="):  # =Free: the whole cell, any case
            terms.append((m.group(1), m.group(2).casefold()))
        else:
            terms.append(("", word.casefold()))
    return terms


def term_matches(term: tuple[str, object], value, text: str) -> bool:
    """Whether one filter word matches a cell (its raw value and its text)."""
    op, operand = term
    if not op:
        return operand in text.casefold()
    if isinstance(operand, str):
        return (text.casefold() == operand) == (op == "=")
    return is_number(value) and _OPERATORS[op](value, operand)


class RunFilterModel(QSortFilterProxyModel):
    """Filters the run table by the filter words, in one column or all of
    them, optionally inverted; sorts by the raw values."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._terms: list = []
        self._column: int | None = None
        self._inverted = False

    def set_filter(self, text: str, column: int | None, inverted: bool) -> None:
        self._terms = parse_filter(text)
        self._column = column
        self._inverted = inverted
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        if not self._terms:  # No filter: every run, inverted or not
            return True
        model = self.sourceModel()
        cols = range(len(COLUMNS)) if self._column is None else (self._column,)
        cells = [(model.value(source_row, c), COLUMNS[c].text(model.value(source_row, c)))
                 for c in cols]
        matches = all(any(term_matches(t, v, s) for v, s in cells) for t in self._terms)
        return matches != self._inverted

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        a = left.data(Qt.ItemDataRole.UserRole)
        b = right.data(Qt.ItemDataRole.UserRole)
        # Blanks last, then numbers by value, everything else by its text
        if a is None or b is None:
            return b is None and a is not None
        if is_number(a) and is_number(b):
            return a < b
        return str(left.data()).casefold() < str(right.data()).casefold()


# =============================================================================
# View
# =============================================================================


class RunTable(QWidget):
    """The filter row above the sortable run table.

    Signals: ``filter_changed`` when other runs pass the filter,
    ``selection_changed`` when the selected runs change, ``run_activated``
    with the run of a double-clicked row, and ``plot_column_requested``
    with a column index from the header's context menu (as x axis, or as
    y axis for a fit metric).
    """

    filter_changed = pyqtSignal()
    selection_changed = pyqtSignal()
    run_activated = pyqtSignal(object)
    plot_column_requested = pyqtSignal(str, str)  # axis ("x" or "y"), column name

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Size.GAP)

        # Filter row: the words, the column they search, and Invert
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setToolTip(
            "Show the runs that match every word.\n"
            "Text matches a part of a cell, for example varpro or tolerance.\n"
            "A number with a comparison compares the value:\n"
            ">1e-5, <=12, =5 or !=3. =free matches the whole cell."
        )
        self.filter_column = QComboBox()
        self.filter_column.addItem("All columns", userData=None)
        for i, c in enumerate(COLUMNS):
            self.filter_column.addItem(c.name, userData=i)
        self.filter_column.setToolTip("The column the filter searches")
        self.invert_button = QToolButton()
        self.invert_button.setText("Invert")
        self.invert_button.setCheckable(True)
        self.invert_button.setToolTip("Show the runs the filter does not match")

        layout.addWidget(self.filter_edit)
        row = QHBoxLayout()
        row.setSpacing(Size.GAP)
        row.addWidget(self.filter_column, stretch=1)
        row.addWidget(self.invert_button)
        layout.addLayout(row)

        # The table
        self.model = RunTableModel(self)
        self.proxy = RunFilterModel(self)
        self.proxy.setSourceModel(self.model)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.view.setShowGrid(False)
        self.view.setWordWrap(False)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header = self.view.horizontalHeader()
        header.setSectionsMovable(True)
        header.setHighlightSections(False)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._header_menu)
        layout.addWidget(self.view, stretch=1)

        self.filter_edit.textChanged.connect(self._apply_filter)
        self.filter_column.currentIndexChanged.connect(self._apply_filter)
        self.invert_button.toggled.connect(self._apply_filter)
        self.view.selectionModel().selectionChanged.connect(
            lambda *_: self.selection_changed.emit())
        self.view.doubleClicked.connect(
            lambda index: self.run_activated.emit(
                self.model.run(self.proxy.mapToSource(index).row())))

    # -- Contents -------------------------------------------------------------

    def set_runs(self, runs: list[OptimizationRun]) -> None:
        """Show these runs, keeping the selected ones selected."""
        selected = self.selected_runs()
        self.view.selectionModel().blockSignals(True)
        self.model.set_runs(runs)
        self.view.resizeColumnsToContents()
        self.select_runs([r for r in selected if r in runs])
        self.view.selectionModel().blockSignals(False)

    def select_runs(self, runs: list[OptimizationRun]) -> None:
        sel = self.view.selectionModel()
        sel.clearSelection()
        flags = sel.SelectionFlag.Select | sel.SelectionFlag.Rows
        for row in range(self.model.rowCount()):
            if any(self.model.run(row) is r for r in runs):
                index = self.proxy.mapFromSource(self.model.index(row, 0))
                if index.isValid():
                    sel.select(index, flags)

    def visible_rows(self) -> list[int]:
        """Model rows that pass the filter, in table order."""
        return [self.proxy.mapToSource(self.proxy.index(r, 0)).row()
                for r in range(self.proxy.rowCount())]

    def visible_runs(self) -> list[OptimizationRun]:
        return [self.model.run(r) for r in self.visible_rows()]

    def selected_runs(self) -> list[OptimizationRun]:
        rows = sorted(index.row() for index in self.view.selectionModel().selectedRows())
        return [self.model.run(self.proxy.mapToSource(self.proxy.index(r, 0)).row())
                for r in rows]

    def values(self, name: str, rows: list[int] | None = None) -> list:
        """Raw values of a column, for the given model rows (default: the
        rows that pass the filter)."""
        col = COLUMN_NAMES.index(name)
        rows = self.visible_rows() if rows is None else rows
        return [self.model.value(r, col) for r in rows]

    # -- Filter ---------------------------------------------------------------

    def _apply_filter(self, *_):
        self.proxy.set_filter(self.filter_edit.text(), self.filter_column.currentData(),
                              self.invert_button.isChecked())
        self.filter_changed.emit()

    def _header_menu(self, pos) -> None:
        """Plot along the column under the cursor."""
        section = self.view.horizontalHeader().logicalIndexAt(pos)
        if section < 0:
            return
        c = COLUMNS[section]
        menu = QMenu(self)
        menu.addAction("Plot as x axis",
                       lambda: self.plot_column_requested.emit("x", c.name))
        if c.metric:
            menu.addAction("Plot as y axis",
                           lambda: self.plot_column_requested.emit("y", c.name))
        menu.addSeparator()
        menu.addAction("Filter this column", lambda: self.filter_column.setCurrentIndex(
            self.filter_column.findData(section)))
        menu.exec(self.view.horizontalHeader().mapToGlobal(pos))
