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
  has its own filter, as in a spreadsheet: the drop-down at the right of
  its header holds a checklist of the column's values, and a condition that
  checks or unchecks them (one after another, to narrow or widen the
  selection; see ``ColumnFilterPopup``). Any number of columns filter at the
  same time, and a run must pass them all; a filtered column shows a
  funnel. In a condition or the global filter every word must match: plain
  text matches a part of the cell, a number with a comparison (``>1e-5``,
  ``<=12``, ``=5``, ``!=3``) compares the value. *Invert* shows the rows
  the filters do not match instead.
- **Sorting.** Clicking a header (or its drop-down) sorts by that column,
  numbers by value.
- **Plot.** The error plot at the lower right shows only the rows that pass
  the filter: any column can become its x axis, and any fit metric its y
  axis (in the plot's toolbar, or from a column header's context menu).
  Selected runs are marked on it.
- **Apply.** Double-clicking a row applies that run's settings to the
  sidebar, so it can be refitted or varied.
- **Pivot.** *Pivot* swaps rows and columns: one column per run, one row
  per setting or metric, to compare a few runs side by side. The selected
  runs stay selected, and the filters, sorting and order of the columns
  travel with their names to the rows (whose names now hold the drop-downs
  and sort the runs when clicked).

This module holds the table itself (columns, model, filter, view);
:class:`~spline_fitter.gui.project.ProjectManager` fills it and connects
it to the plot and the sidebar.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import (
    QEvent,
    QModelIndex,
    QObject,
    QPoint,
    QRect,
    QSortFilterProxyModel,
    Qt,
    QTransposeProxyModel,
    pyqtSignal,
)
from PyQt6.QtGui import QAction, QBrush, QRegion, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..formatting import decimal_point, format_duration, number
from ..optimization.models import OptimizationRun, OptimizerType
from .theme import Palette, Size, brush, icon, mono_font, theme_changed

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


def _has_junctions(run: OptimizationRun) -> bool:
    return run.spline_config.num_sections > 1


def _report(key: str) -> Callable[[OptimizationRun], float | None]:
    return lambda run: (run.result.report or {}).get(key)


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
    Column("Optimize junctions", lambda run: _opt(run).optimize_le_position,
           applies=_has_junctions),
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


class RunTableModel(QStandardItemModel):
    """The runs of a project, one per row, one column per ``COLUMNS`` entry.

    Every cell is an item holding its formatted text, font and alignment, so
    painting the table (on every click in it) runs no Python; ``value`` gives
    the raw value (for sorting, filtering and plotting).
    """

    def __init__(self, parent=None):
        super().__init__(0, len(COLUMNS), parent)
        self._rows: list[_Row] = []
        self._cells: list[list] = []
        # Pivoted, a column per run: every cell centered under the others
        self.centered = False
        for col, column_ in enumerate(COLUMNS):
            item = QStandardItem(column_.name)
            item.setToolTip(column_.tip)
            self.setHorizontalHeaderItem(col, item)

    def set_runs(self, runs: list[OptimizationRun]) -> None:
        self.beginResetModel()
        self._rows = [_Row(run, i + 1) for i, run in enumerate(runs)]
        self._cells = [[c.cell(row) for c in COLUMNS] for row in self._rows]
        # Filled quietly: the reset tells the views once
        self.blockSignals(True)
        self.setRowCount(len(runs))
        font = mono_font()
        for r, values in enumerate(self._cells):
            for col, (value, column_) in enumerate(zip(values, COLUMNS, strict=True)):
                item = QStandardItem(column_.text(value))
                if is_number(value):
                    item.setFont(font)
                item.setTextAlignment(self._alignment(value))
                self.setItem(r, col, item)
        self.blockSignals(False)
        self.endResetModel()

    def set_centered(self, centered: bool) -> None:
        """Center every cell (pivoted), or align numbers right and text left."""
        self.centered = centered
        for r, values in enumerate(self._cells):
            for col, value in enumerate(values):
                self.item(r, col).setTextAlignment(self._alignment(value))

    def _alignment(self, value) -> Qt.AlignmentFlag:
        # Numbers line up: monospaced and right-aligned
        if self.centered:
            return Qt.AlignmentFlag.AlignCenter
        horizontal = (Qt.AlignmentFlag.AlignRight if is_number(value)
                      else Qt.AlignmentFlag.AlignLeft)
        return horizontal | Qt.AlignmentFlag.AlignVCenter

    def run(self, row: int) -> OptimizationRun:
        return self._rows[row].run

    def value(self, row: int, col: int):
        return self._cells[row][col]


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
    column) and its value in every column is checked in that column's
    filter; Invert shows the runs that do not.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._terms: list = []
        self._excluded: dict[int, frozenset[str]] = {}
        self._inverted = False

    def set_filter(self, text: str, excluded: dict[int, frozenset[str]], inverted: bool) -> None:
        self._terms = parse_filter(text)
        self._excluded = {c: texts for c, texts in excluded.items() if texts}
        self._inverted = inverted
        self.invalidateFilter()

    def matches(self, source_row: int, skip: int | None = None) -> bool:
        """Whether a run matches the filters (all but column ``skip``'s),
        regardless of Invert."""
        model = self.sourceModel()
        values = [model.value(source_row, c) for c in range(len(COLUMNS))]
        cells = [(v, c.text(v)) for v, c in zip(values, COLUMNS, strict=True)]
        if not all(any(term_matches(t, v, s) for v, s in cells) for t in self._terms):
            return False
        return all(cells[c][1] not in texts
                   for c, texts in self._excluded.items() if c != skip)

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        if not self._terms and not self._excluded:  # No filter: every run
            return True
        return self.matches(source_row) != self._inverted

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        model = self.sourceModel()
        a = model.value(left.row(), left.column())
        b = model.value(right.row(), right.column())
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

# How the value list names an empty cell: a setting the run's optimizer does not read
_BLANK = "Not used"


class FilterHeader(QHeaderView):
    """A column header (or, pivoted, the row names) with a drop-down glyph at
    the right of every section, as in a spreadsheet. Clicking the glyph emits
    ``filter_requested`` with the column instead of sorting; a filtered column
    shows a funnel. The sorted column has an arrow left of it (in place of the
    style's sort indicator, which would cover the glyph)."""

    filter_requested = pyqtSignal(int)

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setObjectName("FilterHeader")
        self.filtered: set[int] = set()

    def visualRegionForSelection(self, selection) -> QRegion:
        # Sections are not highlighted: selecting runs leaves the header as it
        # was (Qt would repaint all of it, as every setting is in the selection)
        return QRegion()

    def _glyph_rect(self, section: QRect) -> QRect:
        side = Size.GLYPH
        return QRect(section.right() - side - Size.GAP, section.center().y() - side // 2 + 1,
                     side, side)

    def paintSection(self, painter, rect: QRect, logical: int) -> None:
        painter.save()
        super().paintSection(painter, rect, logical)
        painter.restore()
        glyph = self._glyph_rect(rect)
        name, role = ("filter", "TEXT") if logical in self.filtered else ("chevron_down", "TEXT_3")
        icon(name, role).paint(painter, glyph)
        if logical == self.sortIndicatorSection():
            arrow = ("sort_up" if self.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
                     else "sort_down")
            icon(arrow).paint(painter, glyph.translated(-Size.GLYPH - Size.TIGHT, 0))

    def mousePressEvent(self, event) -> None:
        pos = event.position().toPoint()
        logical = self.logicalIndexAt(pos)
        if logical >= 0 and event.button() == Qt.MouseButton.LeftButton:
            start, size = self.sectionViewportPosition(logical), self.sectionSize(logical)
            section = (QRect(start, 0, size, self.height())
                       if self.orientation() == Qt.Orientation.Horizontal
                       else QRect(0, start, self.width(), size))
            # The whole height beside the glyph, clear of the resize grip
            target = self._glyph_rect(section)
            target.setTop(section.top())
            target.setBottom(section.bottom())
            if target.adjusted(-Size.TIGHT, 0, Size.TIGHT // 2, 0).contains(pos):
                self.filter_requested.emit(logical)
                return
        super().mousePressEvent(event)


class ColumnFilterPopup(QFrame):
    """The drop-down of a column header: sort the table by the column, or
    filter it by checking its values (those of the runs the other filters
    let through; all checked: no filter).

    The condition is a quick way to set the checkboxes. While it is typed,
    the values it rules out show unchecked and greyed (and the table follows).
    Enter (Keep matching) keeps that: the values it ruled out stay unchecked
    and the field empties, so the next condition narrows further. Shift+Enter
    (Add matching) checks the values it matches instead, leaving the others
    as they were. The field's clear button, a click on a value or Select all,
    and closing the drop-down keep it as well; emptying the field by hand
    drops it. Every value can be checked by hand at any time. Changes apply
    at once; a click outside or Escape closes the drop-down.
    """

    def __init__(self, table: RunTable, col: int):
        super().__init__(table, Qt.WindowType.Popup)
        self.setObjectName("DropDownPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._table, self._col = table, col
        # The unchecked values, without the condition being typed
        self._excluded = set(table.column_filters.get(col, ()))
        self._terms: list = []
        self._last_text = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Size.PAD, Size.PAD, Size.PAD, Size.PAD)
        layout.setSpacing(Size.GAP)
        caption = QLabel(COLUMNS[col].name.upper())
        caption.setObjectName("SectionCaption")
        layout.addWidget(caption)

        sort_row = QHBoxLayout()
        sort_row.setSpacing(Size.GAP)
        for text, order in (("Sort ascending", Qt.SortOrder.AscendingOrder),
                            ("Sort descending", Qt.SortOrder.DescendingOrder)):
            button = QPushButton(text)
            button.clicked.connect(lambda _, o=order: self._sort(o))
            sort_row.addWidget(button)
        layout.addLayout(sort_row)

        self.condition = QLineEdit()
        self.condition.setPlaceholderText("Condition")
        self.condition.setClearButtonEnabled(True)
        self.condition.setToolTip(
            "Unchecks the values it does not match. Enter keeps the result,\n"
            "Shift+Enter checks the values it matches instead.\n" + _FILTER_TIP)
        layout.addWidget(self.condition)
        condition_row = QHBoxLayout()
        condition_row.setSpacing(Size.GAP)
        self.keep_button = QPushButton("Keep matching")
        self.keep_button.setToolTip("Uncheck the values the condition does not match (Enter)")
        self.add_button = QPushButton("Add matching")
        self.add_button.setToolTip("Check the values the condition matches (Shift+Enter)")
        condition_row.addWidget(self.keep_button)
        condition_row.addWidget(self.add_button)
        layout.addLayout(condition_row)

        self.select_all = QCheckBox("Select all")
        layout.addWidget(self.select_all)
        self.values = QListWidget()
        self._cells: list[tuple[str, object]] = table.column_values(col)
        for text, _ in self._cells:
            item = QListWidgetItem(text or _BLANK)
            item.setData(Qt.ItemDataRole.UserRole, text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.values.addItem(item)
        self._show_values()
        rows = min(max(self.values.count(), 1), 10)
        self.values.setFixedHeight(rows * (self.values.sizeHintForRow(0) + 1) + 4)
        layout.addWidget(self.values)

        clear = QPushButton("Clear filter")
        clear.clicked.connect(self._clear)
        layout.addWidget(clear)

        self.condition.textChanged.connect(self._on_condition_changed)
        self.condition.returnPressed.connect(
            lambda: self.keep(add=bool(QApplication.keyboardModifiers()
                                       & Qt.KeyboardModifier.ShiftModifier)))
        # The clear button keeps what the condition shows, then empties the field
        clear_action = self.condition.findChild(QAction, "_q_qlineeditclearaction")
        if clear_action is not None:
            clear_action.triggered.connect(lambda: self.keep(text=self._last_text))
        self.keep_button.clicked.connect(lambda: self.keep())
        self.add_button.clicked.connect(lambda: self.keep(add=True))
        self.values.itemChanged.connect(self._on_value_changed)
        self.select_all.clicked.connect(self._on_select_all)
        self.setFixedWidth(max(Size.POPUP_MIN_WIDTH, self.sizeHint().width()))
        # The greyed values take TEXT_3 of the theme in effect
        theme_changed.changed.connect(self._show_values)

    def _items(self) -> list[QListWidgetItem]:
        return [self.values.item(i) for i in range(self.values.count())]

    def _ruled_out(self, terms: list) -> set[str]:
        """The values a condition does not match."""
        return {text for text, value in self._cells
                if not all(term_matches(t, value, text) for t in terms)}

    def _shown_excluded(self) -> set[str]:
        """The unchecked values, with what the typed condition rules out."""
        return self._excluded | self._ruled_out(self._terms)

    def _show_values(self) -> None:
        """Check the values that are not unchecked; grey what the typed
        condition rules out."""
        excluded, ruled_out = self._shown_excluded(), self._ruled_out(self._terms)
        self.values.blockSignals(True)
        for item in self._items():
            text = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Unchecked if text in excluded
                               else Qt.CheckState.Checked)
            item.setForeground(brush(Palette.TEXT_3) if text in ruled_out else QBrush())
        self.values.blockSignals(False)
        checked = [i.checkState() == Qt.CheckState.Checked for i in self._items()]
        self.select_all.setCheckState(
            Qt.CheckState.Checked if checked and all(checked)
            else Qt.CheckState.PartiallyChecked if any(checked)
            else Qt.CheckState.Unchecked)
        for button in (self.keep_button, self.add_button):
            button.setEnabled(bool(self._terms))

    def _on_condition_changed(self, text: str) -> None:
        self._terms = parse_filter(text)
        if text:
            self._last_text = text
        self._show_values()
        self._apply()

    def keep(self, add: bool = False, text: str | None = None) -> None:
        """Make what a condition (default: the typed one) shows the checked
        values, and empty the field. ``add`` checks the values it matches
        instead, leaving the others as they were."""
        terms = parse_filter(self.condition.text() if text is None else text)
        if terms:
            ruled_out = self._ruled_out(terms)
            if add:
                self._excluded -= {t for t, _ in self._cells} - ruled_out
            else:
                self._excluded |= ruled_out
        self._terms, self._last_text = [], ""
        self.condition.blockSignals(True)
        self.condition.clear()
        self.condition.blockSignals(False)
        self._show_values()
        self._apply()

    def _on_select_all(self) -> None:
        # Checks every value, or unchecks them all when all are checked
        self.keep()
        texts = {t for t, _ in self._cells}
        self._excluded = set() if self._excluded & texts else texts
        self._show_values()
        self._apply()

    def _on_value_changed(self, item: QListWidgetItem) -> None:
        checked = item.checkState() == Qt.CheckState.Checked
        self.keep()
        text = item.data(Qt.ItemDataRole.UserRole)
        if checked:
            self._excluded.discard(text)
        else:
            self._excluded.add(text)
        self._show_values()
        self._apply()

    def _apply(self) -> None:
        self._table.set_column_filter(COLUMNS[self._col].name, excluded=self._shown_excluded())

    def _sort(self, order) -> None:
        self._table.sort_by(self._col, order)
        self.close()

    def _clear(self) -> None:
        self._terms, self._excluded = [], set()
        self._table.set_column_filter(COLUMNS[self._col].name)
        self.close()

    def closeEvent(self, event) -> None:
        self.keep()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not self.condition.hasFocus()):
            self.close()
        else:
            super().keyPressEvent(event)


class _OpaqueFill(QObject):
    """Keeps a widget filled with its background color (the panel color, as
    the stylesheet would paint it), which the stylesheet turns off whenever
    it polishes the widget. Filled, the table is opaque, so scrolling moves
    its pixels and paints only the rows scrolled in, not the whole table."""

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.Type.Polish, QEvent.Type.StyleChange):
            obj.setAutoFillBackground(True)
        return False


class RunTable(QWidget):
    """The global filter above the sortable run table, whose header opens
    a filter for each column.

    Signals: ``filter_changed`` when other runs pass the filters,
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
        # The unchecked values of each filtered column, by column index
        self.column_filters: dict[int, frozenset[str]] = {}

        # Global filter: words found in any column; Invert applies to all filters
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter all columns")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setToolTip(
            "Show the runs that match in any column, and pass the filter of every\n"
            "column (the drop-down at the right of a column header).\n" + _FILTER_TIP)
        self.invert_button = QToolButton()
        self.invert_button.setText("Invert")
        self.invert_button.setCheckable(True)
        self.invert_button.setToolTip("Show the runs the filters do not match")
        self.pivot_button = QToolButton()
        self.pivot_button.setText("Pivot")
        self.pivot_button.setCheckable(True)
        self.pivot_button.setToolTip(
            "Swap rows and columns: one column per run, one row per setting")
        row = QHBoxLayout()
        row.setSpacing(Size.GAP)
        row.addWidget(self.filter_edit, stretch=1)
        row.addWidget(self.invert_button)
        row.addWidget(self.pivot_button)
        layout.addLayout(row)

        # The table: runs in rows, or in columns when pivoted
        self.model = RunTableModel(self)
        self.proxy = RunFilterModel(self)
        self.proxy.setSourceModel(self.model)
        self.transpose = QTransposeProxyModel(self)
        self.transpose.setSourceModel(self.proxy)
        self.pivoted = False
        self.sort_column, self.sort_order = 0, Qt.SortOrder.AscendingOrder
        self.view = QTableView()
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.view.setShowGrid(False)
        self.view.viewport().installEventFilter(_OpaqueFill(self))
        self.view.setWordWrap(False)
        self._arrange()
        self.sort_by(0, Qt.SortOrder.AscendingOrder)
        layout.addWidget(self.view, stretch=1)

        self.filter_edit.textChanged.connect(self._apply_filter)
        self.invert_button.toggled.connect(self._apply_filter)
        self.pivot_button.toggled.connect(self.set_pivoted)
        self.view.doubleClicked.connect(
            lambda index: self.run_activated.emit(self.model.run(self._model_row(index))))

    def _arrange(self) -> None:
        """Lay the table out as it is pivoted or not: its model, the header
        of the settings (with the drop-downs) and that of the runs."""
        horizontal, vertical = Qt.Orientation.Horizontal, Qt.Orientation.Vertical
        self.model.set_centered(self.pivoted)
        self.view.setModel(self.transpose if self.pivoted else self.proxy)
        self.header = FilterHeader(vertical if self.pivoted else horizontal)
        runs = QHeaderView(horizontal if self.pivoted else vertical)
        if self.pivoted:
            self.view.setVerticalHeader(self.header)
            self.view.setHorizontalHeader(runs)
        else:
            self.view.setHorizontalHeader(self.header)
            self.view.setVerticalHeader(runs)
        # Reparented into a shown view, a new header stays hidden until shown;
        # the runs need no header (the Run row numbers them when pivoted)
        self.header.setVisible(True)
        runs.setVisible(False)
        runs.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        runs.setHighlightSections(False)
        runs.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if self.pivoted:
            self.header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.header.setSortIndicatorShown(False)  # FilterHeader draws its own
        self.header.setSectionsMovable(True)
        self.header.setSectionsClickable(True)
        self.header.setHighlightSections(False)
        self.header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.header.customContextMenuRequested.connect(self._header_menu)
        self.header.filter_requested.connect(self.open_filter)
        self.header.filtered = set(self.column_filters)
        self.header.blockSignals(True)
        self.header.setSortIndicator(self.sort_column, self.sort_order)
        self.header.blockSignals(False)
        # A click on a section flips its sort indicator: sort by it
        self.header.sortIndicatorChanged.connect(self.sort_by)
        self.view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectColumns if self.pivoted
            else QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.selectionModel().selectionChanged.connect(
            lambda *_: self.selection_changed.emit())

    def set_pivoted(self, pivoted: bool) -> None:
        """Show the runs as columns (or rows again). The selected runs stay
        selected; the filters, sorting and order of the columns travel with
        their names."""
        if pivoted == self.pivoted:
            return
        selected = self.selected_runs()
        order = [self.header.logicalIndex(v) for v in range(self.header.count())]
        self.pivoted = pivoted
        self.pivot_button.blockSignals(True)
        self.pivot_button.setChecked(pivoted)
        self.pivot_button.blockSignals(False)
        self._arrange()
        for visual, logical in enumerate(order):
            self.header.moveSection(self.header.visualIndex(logical), visual)
        self.view.selectionModel().blockSignals(True)
        self.select_runs(selected)
        self.view.selectionModel().blockSignals(False)
        self.view.resizeColumnsToContents()
        if selected:
            self.view.scrollTo(self.view.selectionModel().selectedIndexes()[0])

    def sort_by(self, col: int, order: Qt.SortOrder) -> None:
        """Sort the runs by a column (a row when pivoted)."""
        self.sort_column, self.sort_order = col, order
        self.header.blockSignals(True)
        self.header.setSortIndicator(col, order)
        self.header.blockSignals(False)
        self.proxy.sort(col, order)
        self.header.viewport().update()

    def _model_row(self, index: QModelIndex) -> int:
        """The model row of the run a table index shows."""
        if self.pivoted:
            index = self.transpose.mapToSource(index)
        return self.proxy.mapToSource(index).row()

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
        flags = sel.SelectionFlag.Select | (sel.SelectionFlag.Columns if self.pivoted
                                            else sel.SelectionFlag.Rows)
        for row in range(self.model.rowCount()):
            if any(self.model.run(row) is r for r in runs):
                index = self.proxy.mapFromSource(self.model.index(row, 0))
                if index.isValid():
                    sel.select(self.transpose.mapFromSource(index) if self.pivoted
                               else index, flags)

    def visible_rows(self) -> list[int]:
        """Model rows that pass the filter, in table order."""
        return [self.proxy.mapToSource(self.proxy.index(r, 0)).row()
                for r in range(self.proxy.rowCount())]

    def visible_runs(self) -> list[OptimizationRun]:
        return [self.model.run(r) for r in self.visible_rows()]

    def selected_runs(self) -> list[OptimizationRun]:
        sel = self.view.selectionModel()
        indexes = sel.selectedColumns() if self.pivoted else sel.selectedRows()
        return [self.model.run(self._model_row(index)) for index in
                sorted(indexes, key=lambda i: i.column() if self.pivoted else i.row())]

    def values(self, name: str, rows: list[int] | None = None) -> list:
        """Raw values of a column, for the given model rows (default: the
        rows that pass the filter)."""
        col = COLUMN_NAMES.index(name)
        rows = self.visible_rows() if rows is None else rows
        return [self.model.value(r, col) for r in rows]

    # -- Filter ---------------------------------------------------------------

    def column_values(self, col: int) -> list[tuple[str, object]]:
        """The distinct cells (text and value) of a column among the runs
        the other filters let through (and the ones unchecked now), in sort
        order."""
        excluded = self.column_filters.get(col, frozenset())
        cells = {}
        for row in range(self.model.rowCount()):
            value = self.model.value(row, col)
            text = COLUMNS[col].text(value)
            if text in excluded or self.proxy.matches(row, skip=col):
                cells.setdefault(text, value)

        def key(text):
            value = cells[text]
            return (value is None, not is_number(value),
                    value if is_number(value) else text.casefold())
        return [(text, cells[text]) for text in sorted(cells, key=key)]

    def set_column_filter(self, name: str, condition: str = "", excluded=()) -> None:
        """Filter a column: uncheck the given cell texts, and those of its
        values a condition does not match (as Keep matching does); nothing
        clears its filter."""
        col = COLUMN_NAMES.index(name)
        excluded = set(excluded)
        if terms := parse_filter(condition):
            for row in range(self.model.rowCount()):
                value = self.model.value(row, col)
                text = COLUMNS[col].text(value)
                if not all(term_matches(t, value, text) for t in terms):
                    excluded.add(text)
        if excluded:
            self.column_filters[col] = frozenset(excluded)
        else:
            self.column_filters.pop(col, None)
        self._apply_filter()

    def clear_filters(self) -> None:
        """Empty the global filter and every column filter."""
        self.column_filters.clear()
        self.filter_edit.blockSignals(True)
        self.filter_edit.clear()
        self.filter_edit.blockSignals(False)
        self._apply_filter()

    def open_filter(self, col: int) -> ColumnFilterPopup:
        """Drop the filter of a column down under its header section."""
        popup = ColumnFilterPopup(self, col)
        popup.move(self.drop_position(col, popup.width(), popup.sizeHint().height()))
        popup.show()
        popup.condition.setFocus()
        return popup

    def drop_position(self, col: int, width: int, height: int = 0) -> QPoint:
        """Where a drop-down of that size opens under a header section: at
        its left edge, or at its right edge (as far as it shows) where it
        would pass the right edge of the window (the last columns), and
        always on the screen. Pivoted, it opens right of the row name."""
        start = self.header.sectionViewportPosition(col)
        if self.pivoted:
            pos = self.header.mapToGlobal(QPoint(self.header.width(), start))
        else:
            pos = self.header.mapToGlobal(QPoint(start, self.header.height()))
            if pos.x() + width > self.window().frameGeometry().right() + 1:
                right = min(start + self.header.sectionSize(col), self.header.width())
                pos.setX(self.header.mapToGlobal(QPoint(right, 0)).x() - width)
        screen = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        pos.setX(max(screen.left(), min(pos.x(), screen.right() + 1 - width)))
        pos.setY(max(screen.top(), min(pos.y(), screen.bottom() + 1 - height)))
        return pos

    def _apply_filter(self, *_):
        self.header.filtered = set(self.column_filters)
        self.header.viewport().update()
        self.proxy.set_filter(self.filter_edit.text(), self.column_filters,
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
        menu.addAction("Filter this column...", lambda: self.open_filter(section))
        menu.addAction("Clear all filters", self.clear_filters)
        menu.exec(self.header.mapToGlobal(pos))
