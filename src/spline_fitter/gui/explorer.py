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
- **Filter.** A global filter field searches all columns, and every column
  has its own filter field under its header; any number of them filter at
  the same time, and a run must match them all. Every word must match:
  plain text matches a part of the cell, a number with a comparison
  (``>1e-5``, ``<=12``, ``=5``, ``!=3``) compares the value. *Invert*
  shows the rows the filters do not match instead.
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
    """Filters the run table, optionally inverted, and sorts by the raw values.

    A run passes when it matches the global filter (each word in some
    column) and every column filter (each word in that column); Invert
    shows the runs that do not.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._terms: list = []
        self._column_terms: dict[int, list] = {}
        self._inverted = False

    def set_filter(self, text: str, column_texts: dict[int, str], inverted: bool) -> None:
        self._terms = parse_filter(text)
        self._column_terms = {c: terms for c, t in column_texts.items()
                              if (terms := parse_filter(t))}
        self._inverted = inverted
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        if not self._terms and not self._column_terms:  # No filter: every run
            return True
        model = self.sourceModel()

        def cell(c: int):
            value = model.value(source_row, c)
            return value, COLUMNS[c].text(value)

        cells = [cell(c) for c in range(len(COLUMNS))]
        matches = (all(any(term_matches(t, v, s) for v, s in cells) for t in self._terms)
                   and all(term_matches(t, *cells[c])
                           for c, terms in self._column_terms.items() for t in terms))
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

_FILTER_TIP = (
    "Every word must match.\n"
    "Text matches a part of a cell, for example varpro or tolerance.\n"
    "A number with a comparison compares the value:\n"
    ">1e-5, <=12, =5 or !=3. =free matches the whole cell."
)


class FilterHeader(QHeaderView):
    """A column header with a filter field under every section.

    The fields follow their sections as they are resized, moved, hidden
    or scrolled; ``edited`` is emitted when one of them changes.
    """

    edited = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.editors: list[QLineEdit] = []
        self.sectionResized.connect(self._place_editors)
        self.sectionMoved.connect(self._place_editors)

    def set_columns(self, names: list[str]) -> None:
        for name in names:
            edit = QLineEdit(self)
            edit.setClearButtonEnabled(True)
            edit.setToolTip(f"Filter the {name} column. {_FILTER_TIP}")
            edit.textChanged.connect(lambda *_: self.edited.emit())
            self.editors.append(edit)
        self.updateGeometries()

    def _filter_height(self) -> int:
        return Size.CONTROL_HEIGHT + Size.TIGHT if self.editors else 0

    def sizeHint(self):
        size = super().sizeHint()
        size.setHeight(size.height() + self._filter_height())
        return size

    def updateGeometries(self):
        # The section labels above, the filter fields in the margin below
        self.setViewportMargins(0, 0, 0, self._filter_height())
        super().updateGeometries()
        self._place_editors()

    def _place_editors(self, *_):
        top = self.viewport().geometry().bottom() + 1 + Size.TIGHT // 2
        for logical, edit in enumerate(self.editors):
            if self.isSectionHidden(logical):
                edit.hide()
                continue
            edit.setGeometry(self.sectionViewportPosition(logical), top,
                             self.sectionSize(logical) - 1, Size.CONTROL_HEIGHT)
            edit.show()


class RunTable(QWidget):
    """The global filter above the sortable run table, whose header has a
    filter field per column.

    Signals: ``filter_changed`` when other runs pass the filter,
    ``selection_changed`` when the selected runs change, ``run_activated``
    with the run of a double-clicked row, and ``plot_column_requested``
    with a column name from the header's context menu (as x axis, or as
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

        # Global filter: words found in any column; Invert applies to all filters
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter all columns")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setToolTip(
            "Show the runs that match in any column, and in every column filter "
            "under the headers.\n" + _FILTER_TIP)
        self.invert_button = QToolButton()
        self.invert_button.setText("Invert")
        self.invert_button.setCheckable(True)
        self.invert_button.setToolTip("Show the runs the filters do not match")
        row = QHBoxLayout()
        row.setSpacing(Size.GAP)
        row.addWidget(self.filter_edit, stretch=1)
        row.addWidget(self.invert_button)
        layout.addLayout(row)

        # The table
        self.model = RunTableModel(self)
        self.proxy = RunFilterModel(self)
        self.proxy.setSourceModel(self.model)
        self.view = QTableView()
        self.header = FilterHeader(self.view)
        self.view.setHorizontalHeader(self.header)
        self.view.setModel(self.proxy)
        self.header.set_columns(COLUMN_NAMES)
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
        self.header.setSectionsMovable(True)
        self.header.setSectionsClickable(True)
        self.header.setHighlightSections(False)
        # Wide enough for a comparison in the column's filter field
        self.header.setMinimumSectionSize(64)
        self.header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.header.customContextMenuRequested.connect(self._header_menu)
        self.view.horizontalScrollBar().valueChanged.connect(self.header._place_editors)
        layout.addWidget(self.view, stretch=1)

        self.filter_edit.textChanged.connect(self._apply_filter)
        self.header.edited.connect(self._apply_filter)
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

    def column_filter(self, name: str) -> QLineEdit:
        """The filter field of a column."""
        return self.header.editors[COLUMN_NAMES.index(name)]

    def clear_filters(self) -> None:
        """Empty the global filter and every column filter."""
        for edit in (self.filter_edit, *self.header.editors):
            edit.blockSignals(True)
            edit.clear()
            edit.blockSignals(False)
        self._apply_filter()

    def _apply_filter(self, *_):
        self.proxy.set_filter(
            self.filter_edit.text(),
            {c: edit.text() for c, edit in enumerate(self.header.editors)},
            self.invert_button.isChecked())
        self.filter_changed.emit()

    def _header_menu(self, pos) -> None:
        """Plot along the column under the cursor, or filter it."""
        section = self.header.logicalIndexAt(pos)
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
        menu.addAction("Filter this column", self.header.editors[section].setFocus)
        menu.addAction("Clear all filters", self.clear_filters)
        menu.exec(self.header.mapToGlobal(pos))
