"""
Reusable UI Widgets
===================

Custom PyQt6 widgets used by the spline fitter GUI.

Widgets carry object names that theme.py styles; they set no colors or
stylesheets of their own.

This module provides:
    - SidebarSection / CollapsibleSection: Captioned sidebar sections
    - SegmentedControl: Button row with a combo-box-like API
    - SweepableSpinBox: Spinbox with parameter sweep toggle
    - ReadoutList: Label and value pairs, as many to a line as fit
    - ReflowRow: Widgets side by side, stacked when the row is too narrow
    - ThemeDialog: Light / dark mode and the temperature of the greys
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QEvent, QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QSplitterHandle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .theme import (
    Motion,
    Scheme,
    Size,
    glyph_size,
    icon,
    readout_heading,
    set_motion,
    set_theme,
    text_browser,
    theme_changed,
)

# =============================================================================
# Constants
# =============================================================================

# Maximum number of points to plot in the loss graph (to prevent UI freeze)
MAX_PLOT_POINTS = 100_000

# Upper end of an integer setting with no limit of its own (Qt's largest int);
# a float setting with none takes math.inf
UNLIMITED = 2**31 - 1


# =============================================================================
# Sidebar Sections
# =============================================================================

class SidebarSection(QWidget):
    """A sidebar section: a small caption above its content, divider below."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("SidebarSection")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Size.PAD, Size.PAD - 2, Size.PAD, Size.PAD)
        main_layout.setSpacing(Size.GROUP)
        main_layout.addWidget(self._build_header(title))

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(Size.GAP)
        main_layout.addWidget(self._content)

    def _build_header(self, title: str) -> QWidget:
        caption = QLabel(title.upper())
        caption.setObjectName("SectionCaption")
        return caption

    def content_layout(self) -> QVBoxLayout:
        """Return the layout for adding child widgets."""
        return self._content_layout

    def content_widget(self) -> QWidget:
        """Return the widget holding the section's content."""
        return self._content

    def add_widget(self, widget: QWidget):
        """Add a widget to the content area."""
        self._content_layout.addWidget(widget)

    def add_layout(self, layout):
        """Add a layout to the content area."""
        self._content_layout.addLayout(layout)


class ReadoutList(QWidget):
    """A title over label and value pairs, laid out in as many columns as
    the width allows: all pairs in one line when wide, one per line when
    narrow. Pairs fill the lines left to right, labels and values each line
    up in their columns, and spare width goes between the pairs. Shows a
    plain message instead when there are no pairs.

    The title's rich text carries the theme's colors, so it is rendered
    again when the theme changes. The values are plain text, styled by
    object name: rich text takes long to measure, and every run shown
    measures them all.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Size.GAP)
        self._title = QLabel()
        self._title.setWordWrap(True)
        self._title.setObjectName("Muted")
        layout.addWidget(self._title)
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(0)
        self._grid.setVerticalSpacing(1)
        layout.addLayout(self._grid)

        self._title_text = ""
        self._rows: list[tuple[str, str]] = []
        self._cells: list[tuple[QLabel, QLabel]] = []
        self._hints: list[tuple[int, int]] = []  # their size hints' widths
        self._columns = 0
        theme_changed.changed.connect(self._render)

    def set_message(self, text: str) -> None:
        """Show a muted message and no pairs."""
        self._title_text, self._rows = text, []
        self._render()

    def set_rows(self, title: str, rows: list[tuple[str, str]]) -> None:
        """Show a title over (label, value) pairs."""
        self._title_text, self._rows = title, list(rows)
        self._render()

    def text(self) -> str:
        """The title and every pair, one per line (for tests)."""
        return "\n".join([self._title_text, *(f"{k} {v}" for k, v in self._rows)])

    def columns(self) -> int:
        """How many pairs share a line."""
        return self._columns

    def _render(self) -> None:
        self._title.setText(readout_heading(self._title_text) if self._rows
                            else self._title_text)
        # The labels are reused, so a new run of as many pairs only changes
        # text; a message hides them until then
        for label, value in self._cells:
            label.setVisible(bool(self._rows))
            value.setVisible(bool(self._rows))
        if not self._rows:
            return
        if len(self._cells) != len(self._rows):
            self._columns = 0
        for label, value in self._cells[len(self._rows):]:
            label.deleteLater()
            value.deleteLater()
        del self._cells[len(self._rows):]
        while len(self._cells) < len(self._rows):
            label = QLabel()
            label.setObjectName("Muted")
            value = QLabel()
            value.setObjectName("ReadoutValue")
            value.setTextFormat(Qt.TextFormat.PlainText)
            value.setContentsMargins(Size.GROUP, 0, 0, 0)
            self._cells.append((label, value))
        for (label, value), (key, val) in zip(self._cells, self._rows, strict=True):
            label.setText(key)
            value.setText(val)
        # Measured once: each width is looked up for every column count tried
        self._hints = [(label.sizeHint().width(), value.sizeHint().width())
                       for label, value in self._cells]
        self._reflow()

    def _widths(self, columns: int) -> int:
        """The width the pairs take in that many columns, with no spare."""
        total = 0
        for c in range(columns):
            column = self._hints[c::columns]
            total += max(label for label, _ in column) + max(value for _, value in column)
        return total + (columns - 1) * Size.PAD_LARGE

    def _reflow(self) -> None:
        """Place the pairs in as many columns as fit the width."""
        if not self._cells:
            return
        columns = next((n for n in range(len(self._cells), 1, -1)
                        if self._widths(n) <= self.width()), 1)
        if columns == self._columns:
            return
        self._columns = columns
        for label, value in self._cells:
            self._grid.removeWidget(label)
            self._grid.removeWidget(value)
        for c in range(self._grid.columnCount()):
            self._grid.setColumnStretch(c, 0)
            self._grid.setColumnMinimumWidth(c, 0)
        for i, (label, value) in enumerate(self._cells):
            row, c = divmod(i, columns)
            self._grid.addWidget(label, row, 3 * c)
            self._grid.addWidget(value, row, 3 * c + 1)
        # An empty column after each pair: at least PAD_LARGE between pairs,
        # and the spare width shared out
        for c in range(columns):
            self._grid.setColumnMinimumWidth(3 * c + 2, Size.PAD_LARGE if c < columns - 1 else 0)
            self._grid.setColumnStretch(3 * c + 2, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()


class CollapsibleSection(SidebarSection):
    """A sidebar section whose caption toggles the content, with an optional badge."""

    def __init__(self, title: str, parent=None, expanded: bool = True):
        self._title = title
        self._expanded = expanded
        super().__init__(title, parent)
        self._content.setVisible(expanded)

    def _build_header(self, title: str) -> QWidget:
        header = QWidget()
        lay = QHBoxLayout(header)
        lay.setContentsMargins(0, 0, 0, 0)
        self._toggle_btn = QPushButton(" " + self._title.upper())
        self._toggle_btn.setObjectName("SectionHeader")
        self._toggle_btn.setIconSize(glyph_size())
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.clicked.connect(self._on_toggle)
        self._update_chevron()
        lay.addWidget(self._toggle_btn, stretch=1)
        self._badge = QLabel()
        self._badge.setObjectName("SectionBadge")
        lay.addWidget(self._badge)
        return header

    def _update_chevron(self):
        self._toggle_btn.setIcon(icon("chevron_down" if self._expanded else "chevron_right"))

    def _on_toggle(self):
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool):
        """Programmatically expand or collapse."""
        self._expanded = expanded
        self._content.setVisible(expanded)
        self._update_chevron()

    def set_badge(self, text: str, tooltip: str = ""):
        """Show a short note at the right of the header (empty hides it)."""
        self._badge.setText(text)
        self._badge.setToolTip(tooltip)


# =============================================================================
# Drop-down Panel
# =============================================================================

class DropDownPanel(QFrame):
    """Settings that drop down under a checkable button, toggled by it.

    Unlike a QMenu it does not grab the mouse, so the rest of the window
    (zooming and panning the plots, other controls) stays usable while it is
    open. It closes on the button or on Escape.
    """

    def __init__(self, button: QToolButton, content: QWidget, window: QWidget):
        super().__init__(window, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("DropDownPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(content)
        self._button = button
        button.setCheckable(True)
        button.toggled.connect(self._set_open)
        # Follow the button when the window moves or resizes
        window.installEventFilter(self)

    def _place(self):
        self.move(self._button.mapToGlobal(QPoint(0, self._button.height())))

    def _set_open(self, open_: bool):
        if open_:
            self.adjustSize()
            self._place()
            self.show()
        else:
            self.hide()

    def eventFilter(self, obj, event):
        if self.isVisible() and event.type() in (QEvent.Type.Move, QEvent.Type.Resize):
            self._place()
        return False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._button.setChecked(False)
        else:
            super().keyPressEvent(event)


# =============================================================================
# Segmented Control
# =============================================================================

class SegmentedControl(QWidget):
    """A row of mutually exclusive buttons with the item API of a QComboBox.

    Supports addItem/count/itemData/itemText/setItemText/currentIndex/
    currentData/setCurrentIndex/findData and the currentIndexChanged signal,
    so it can stand in for a combo box whose options fit on one line.
    """

    currentIndexChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.idClicked.connect(self.setCurrentIndex)
        self._data: list = []
        self._current = -1

    def addItem(self, text: str, userData=None):
        btn = QPushButton(text)
        btn.setObjectName("Segment")
        btn.setCheckable(True)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._group.addButton(btn, len(self._data))
        self._layout.addWidget(btn)
        self._data.append(userData)
        self._mark_ends()
        if self._current < 0:
            self.setCurrentIndex(0)

    def _mark_ends(self):
        """Round the outer corners of the first and last segments."""
        buttons = self._group.buttons()
        for i, b in enumerate(buttons):
            b.setProperty("position", "first" if i == 0 else "last" if i == len(buttons) - 1 else "")
            b.style().unpolish(b)
            b.style().polish(b)

    def count(self) -> int:
        return len(self._data)

    def itemData(self, index: int):
        return self._data[index] if 0 <= index < len(self._data) else None

    def itemText(self, index: int) -> str:
        return self._group.button(index).text()

    def setItemText(self, index: int, text: str):
        self._group.button(index).setText(text)

    def setItemToolTip(self, index: int, text: str):
        self._group.button(index).setToolTip(text)

    def findData(self, data) -> int:
        return self._data.index(data) if data in self._data else -1

    def currentIndex(self) -> int:
        return self._current

    def currentData(self):
        return self.itemData(self._current)

    def setCurrentIndex(self, index: int):
        if not 0 <= index < len(self._data):
            return
        self._group.button(index).setChecked(True)
        if index != self._current:
            self._current = index
            self.currentIndexChanged.emit(index)


# =============================================================================
# Scientific SpinBox
# =============================================================================

class ScientificSpinBox(QDoubleSpinBox):
    """A spin box for values spanning decades (tolerances, weights).

    Shows the value in its shortest form ("1e-05", "0.001") instead of a
    fixed number of decimals, accepts e-notation, and steps along 1-2-5.
    """

    SIGNIFICANT = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        super().setDecimals(15)

    def setDecimals(self, decimals: int) -> None:
        """Ignored: the value shows its own significant digits."""

    def textFromValue(self, value: float) -> str:
        return f"{value:.{self.SIGNIFICANT}g}".replace(".", self.locale().decimalPoint())

    def _parse(self, text: str) -> float:
        """The value of a text with the locale's decimal point; ValueError if none."""
        return float(text.replace(self.locale().decimalPoint(), "."))

    def valueFromText(self, text: str) -> float:
        try:
            return self._parse(text)
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        try:
            value = self._parse(text)
        except ValueError:
            # Part of a number still being typed, such as "1e-"
            ok = all(c in "0123456789eE+-" + self.locale().decimalPoint() for c in text)
            return (QValidator.State.Intermediate if ok else QValidator.State.Invalid), text, pos
        if self.minimum() <= value <= self.maximum():
            return QValidator.State.Acceptable, text, pos
        return QValidator.State.Intermediate, text, pos

    def stepBy(self, steps: int) -> None:
        value = self.value()
        for _ in range(abs(steps)):
            value = self._next(value, up=steps > 0)
        self.setValue(min(max(value, self.minimum()), self.maximum()))

    def _next(self, value: float, up: bool) -> float:
        """The neighbouring value in the 1-2-5 sequence."""
        if value <= 0:
            return self.singleStep() if up else 0.0
        exponent = math.floor(math.log10(value))
        ladder = [m * 10.0 ** e for e in (exponent - 1, exponent, exponent + 1)
                  for m in (1.0, 2.0, 5.0)]
        if up:
            return min(v for v in ladder if v > value * (1 + 1e-9))
        lower = [v for v in ladder if v < value * (1 - 1e-9)]
        return max(lower) if lower and max(lower) >= self.minimum() else self.minimum()


# =============================================================================
# Sweepable SpinBox Widget
# =============================================================================

class SweepableSpinBox(QWidget):
    """A spinbox with an inline sweep-range toggle button.

    When the range button (a dimension line) is toggled, a [max] field appears, turning
    the single value into a range for parameter sweeps.  Several sweepable
    widgets of a *group* can be active at once: the sweep fits every
    combination of their values.
    """

    # Sweep mode toggled or the range changed (not the value itself)
    sweep_changed = pyqtSignal()

    def __init__(self, spin_box: QSpinBox, group: list | None = None, parent=None):
        super().__init__(parent)
        self._sweep_active = False
        self._group = group
        if group is not None:
            group.append(self)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(Size.TIGHT)

        self._spin = spin_box
        lay.addWidget(self._spin, stretch=1)

        self._btn_sweep = QPushButton()
        self._btn_sweep.setObjectName("SweepToggle")
        self._btn_sweep.setIcon(icon("range", checked_role="TEXT"))
        self._btn_sweep.setIconSize(QSize(Size.GLYPH_LARGE, Size.GLYPH_LARGE))
        self._btn_sweep.setFixedSize(Size.CONTROL_HEIGHT, Size.CONTROL_HEIGHT)
        self._btn_sweep.setToolTip("Sweep a range of values: one fit per value, or per combination "
                                   "with the other swept settings")
        self._btn_sweep.setCheckable(True)
        self._btn_sweep.clicked.connect(self._on_toggle_sweep)
        lay.addWidget(self._btn_sweep)

        self._spin_max = QSpinBox()
        self._spin_max.setRange(self._spin.minimum(), self._spin.maximum())
        self._spin_max.setSingleStep(self._spin.singleStep())
        self._spin_max.setValue(min(self._spin.value() * 2, self._spin.maximum()))
        self._spin_max.setVisible(False)
        self._spin_max.valueChanged.connect(self.sweep_changed)
        lay.addWidget(self._spin_max, stretch=1)

    # ------------------------------------------------------------------
    def _on_toggle_sweep(self, checked: bool):
        self._sweep_active = checked
        self._spin_max.setVisible(checked)
        if checked:
            self._spin_max.setMinimum(self._spin.minimum())
            self._spin_max.setValue(min(self._spin.value() * 2, self._spin.maximum()))
        self.sweep_changed.emit()

    def sweep_state(self) -> tuple:
        """Sweep mode and range end, for set_sweep_state()."""
        return (self._sweep_active, self._spin_max.value())

    def set_sweep_state(self, state: tuple) -> None:
        active, maximum = state
        if active != self._sweep_active:
            self._btn_sweep.setChecked(active)
            self._on_toggle_sweep(active)
        self._spin_max.setValue(maximum)

    def stop_sweep(self) -> None:
        """Leave sweep mode."""
        self._btn_sweep.setChecked(False)
        self._on_toggle_sweep(False)

    def set_sweep_allowed(self, allowed: bool) -> None:
        """Enable or disable the sweep toggle, leaving sweep mode if needed."""
        if not allowed and self._sweep_active:
            self.stop_sweep()
        self._btn_sweep.setEnabled(allowed)

    def set_minimum(self, minimum: int) -> None:
        """Set the lowest allowed value, of the value and the sweep's upper end
        alike; either one below it is raised to it."""
        self._spin.setMinimum(minimum)
        self._spin_max.setMinimum(minimum)

    def ends(self) -> list[QSpinBox]:
        """The value, and the sweep's upper end while sweeping."""
        return [self._spin, self._spin_max] if self._sweep_active else [self._spin]

    def set_sweep_maximum(self, maximum: int) -> None:
        """Set the highest allowed upper end of the sweep; one above it is
        lowered to it. The value itself keeps its own maximum."""
        self._spin_max.setMaximum(maximum)

    @property
    def is_sweep_active(self) -> bool:
        return self._sweep_active

    @property
    def sweep_range(self) -> tuple[int, int]:
        """Return (min_val, max_val) for the sweep."""
        return (self._spin.value(), self._spin_max.value())

    @property
    def spinbox(self) -> QSpinBox:
        return self._spin


# =============================================================================
# Sweepable Double SpinBox Widget (for float parameters)
# =============================================================================

class SweepableDoubleSpinBox(QWidget):
    """A double-spinbox with an inline sweep-range toggle and step size.

    When the range button (a dimension line) is toggled, [max] and Step fields appear,
    turning the single value into a range for parameter sweeps.  Several
    sweepable widgets of a *group* can be active at once (SweepableSpinBox
    and SweepableDoubleSpinBox alike).
    """

    # Sweep mode toggled or the range or step changed (not the value itself)
    sweep_changed = pyqtSignal()

    def __init__(
        self,
        spin_box: QDoubleSpinBox,
        group: list | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._sweep_active = False
        self._group = group
        if group is not None:
            group.append(self)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(Size.TIGHT)

        self._spin = spin_box
        lay.addWidget(self._spin, stretch=1)

        self._btn_sweep = QPushButton()
        self._btn_sweep.setObjectName("SweepToggle")
        self._btn_sweep.setIcon(icon("range", checked_role="TEXT"))
        self._btn_sweep.setIconSize(QSize(Size.GLYPH_LARGE, Size.GLYPH_LARGE))
        self._btn_sweep.setFixedSize(Size.CONTROL_HEIGHT, Size.CONTROL_HEIGHT)
        self._btn_sweep.setToolTip("Sweep a range of values: one fit per value, or per combination "
                                   "with the other swept settings")
        self._btn_sweep.setCheckable(True)
        self._btn_sweep.clicked.connect(self._on_toggle_sweep)
        lay.addWidget(self._btn_sweep)

        self._spin_max = type(self._spin)()
        self._spin_max.setRange(self._spin.minimum(), self._spin.maximum())
        self._spin_max.setDecimals(self._spin.decimals())
        self._spin_max.setSingleStep(self._spin.singleStep())
        self._spin_max.setValue(min(self._spin.value() * 10, self._spin.maximum()))
        self._spin_max.setVisible(False)
        lay.addWidget(self._spin_max, stretch=1)

        self._lbl_step = QLabel("Step")
        self._lbl_step.setObjectName("FormLabel")
        self._lbl_step.setVisible(False)
        lay.addWidget(self._lbl_step)

        self._spin_step = type(self._spin)()
        self._spin_step.setRange(self._spin.singleStep() / 1000, self._spin.maximum())
        self._spin_step.setDecimals(self._spin.decimals())
        self._spin_step.setSingleStep(self._spin.singleStep())
        self._spin_step.setToolTip("Distance between the swept values")
        self._spin_step.setVisible(False)
        lay.addWidget(self._spin_step, stretch=1)
        self._spin_max.valueChanged.connect(self.sweep_changed)
        self._spin_step.valueChanged.connect(self.sweep_changed)

    # ------------------------------------------------------------------
    def _on_toggle_sweep(self, checked: bool):
        self._sweep_active = checked
        for widget in (self._spin_max, self._lbl_step, self._spin_step):
            widget.setVisible(checked)
        if checked:
            # Ten values from the current one: up to ten times it, in steps of it
            start = self._spin.value() or self._spin.singleStep()
            self._spin_max.setMinimum(self._spin.minimum())
            self._spin_max.setValue(min(start * 10, self._spin.maximum()))
            self._spin_step.setValue(start)
        self.sweep_changed.emit()

    def sweep_state(self) -> tuple:
        """Sweep mode, range end and step, for set_sweep_state()."""
        return (self._sweep_active, self._spin_max.value(), self._spin_step.value())

    def set_sweep_state(self, state: tuple) -> None:
        active, maximum, step = state
        if active != self._sweep_active:
            self._btn_sweep.setChecked(active)
            self._on_toggle_sweep(active)
        self._spin_max.setValue(maximum)
        self._spin_step.setValue(step)

    def stop_sweep(self) -> None:
        """Leave sweep mode."""
        self._btn_sweep.setChecked(False)
        self._on_toggle_sweep(False)

    @property
    def is_sweep_active(self) -> bool:
        return self._sweep_active

    @property
    def sweep_range(self) -> tuple[float, float]:
        """Return (min_val, max_val) for the sweep."""
        return (self._spin.value(), self._spin_max.value())

    @property
    def sweep_step(self) -> float:
        """The distance between swept values."""
        return self._spin_step.value()

    def sweep_values(self, limit: int = 1000) -> list[float]:
        """The swept values: from the lower to the upper end of the range in
        steps of the step size, the upper end included when a step lands on
        it. At most ``limit`` values."""
        low, high = sorted(self.sweep_range)
        step = self.sweep_step
        count = min(int(math.floor((high - low) / step * (1 + 1e-9))) + 1, limit)
        # Rounded so 0.001 + 8 * 0.001 is 0.009, not 0.009000000000000001
        return [float(f"{low + i * step:.12g}") for i in range(count)]

    @property
    def spinbox(self) -> QDoubleSpinBox:
        return self._spin


# =============================================================================
# Reflowing Row
# =============================================================================

class ReflowRow(QWidget):
    """Widgets side by side while they fit the row's width, one above the
    other when it is narrower, so the row asks for no more width than its
    widest widget (the window can be made as narrow as the rest allows)."""

    def __init__(self, spacing: int, parent=None):
        super().__init__(parent)
        self._layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(spacing)

    def add_widget(self, widget: QWidget, stretch: int = 1) -> None:
        self._layout.addWidget(widget, stretch)

    def is_stacked(self) -> bool:
        return self._layout.direction() == QBoxLayout.Direction.TopToBottom

    def _widths(self) -> list[int]:
        items = (self._layout.itemAt(i) for i in range(self._layout.count()))
        return [item.minimumSize().width() for item in items if not item.isEmpty()]

    def minimumSizeHint(self) -> QSize:
        return QSize(max(self._widths(), default=0), self._layout.minimumSize().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        widths = self._widths()
        side_by_side = sum(widths) + self._layout.spacing() * max(len(widths) - 1, 0)
        direction = (QBoxLayout.Direction.LeftToRight if event.size().width() >= side_by_side
                     else QBoxLayout.Direction.TopToBottom)
        if direction != self._layout.direction():
            self._layout.setDirection(direction)
            # Stacked, the row needs more height
            self.updateGeometry()


# =============================================================================
# Splitter
# =============================================================================

class _ResettableHandle(QSplitterHandle):
    def mouseDoubleClickEvent(self, event):
        splitter = self.splitter()
        splitter.handleDoubleClicked.emit(splitter.indexOf(self))


class ResettableSplitter(QSplitter):
    """A splitter whose handles report a double-click (with the handle's
    index, 1 for the one after the first widget), so the panels beside it
    can go back to their default sizes."""

    handleDoubleClicked = pyqtSignal(int)

    def createHandle(self) -> QSplitterHandle:
        return _ResettableHandle(self.orientation(), self)


# =============================================================================
# Theme Dialog
# =============================================================================

def dialog_form(dialog: QDialog) -> QFormLayout:
    """The form of a settings dialog, with the dialog padding."""
    form = QFormLayout(dialog)
    form.setContentsMargins(Size.PAD_LARGE, Size.PAD_LARGE, Size.PAD_LARGE, Size.PAD_LARGE)
    form.setHorizontalSpacing(Size.GROUP)
    form.setVerticalSpacing(Size.GAP)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    return form


def add_form_row(form: QFormLayout, text: str, field) -> QLabel:
    """Add a row with the fixed-width form label and return the label."""
    label = QLabel(text)
    label.setObjectName("FormLabel")
    label.setFixedWidth(Size.LABEL_WIDTH)
    form.addRow(label, field)
    return label


def show_document(parent: QWidget, title: str, html: str, width: int, height: int) -> None:
    """Show rich text (a report) in a dialog with a Close button."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(width, height)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(0, 0, 0, Size.PAD_LARGE)
    layout.setSpacing(Size.PAD_LARGE)
    browser = text_browser()
    browser.setOpenExternalLinks(True)
    browser.setHtml(html)
    layout.addWidget(browser)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.accept)
    buttons.setContentsMargins(Size.PAD_LARGE, 0, Size.PAD_LARGE, 0)
    layout.addWidget(buttons)
    dialog.exec()


class ThemeDialog(QDialog):
    """Light or dark mode (or the system's) and the temperature of the greys,
    applied while they change and saved by theme.set_theme."""

    # Temperatures this close to 0 snap to neutral
    SNAP = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Theme")
        form = dialog_form(self)

        self._mode = SegmentedControl()
        for mode in Scheme.MODES:
            self._mode.addItem(mode.capitalize(), userData=mode)
        self._mode.setItemToolTip(0, "Follow the light or dark setting of the system")
        self._mode.setCurrentIndex(self._mode.findData(Scheme.MODE))
        self._mode.currentIndexChanged.connect(lambda _: set_theme(mode=self._mode.currentData()))
        add_form_row(form, "Mode", self._mode)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setObjectName("TemperatureSlider")
        self._slider.setRange(-100, 100)
        self._slider.setPageStep(10)
        self._slider.setMinimumWidth(Size.SLIDER_WIDTH)
        self._slider.setValue(Scheme.TEMPERATURE)
        self._slider.setToolTip("Tint the greys cool (blue) or warm (sand); neutral in the middle")
        cool, warm = QLabel("Cool"), QLabel("Warm")
        for label in (cool, warm):
            label.setObjectName("Muted")
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(Size.GAP)
        lay.addWidget(cool)
        lay.addWidget(self._slider, stretch=1)
        lay.addWidget(warm)
        add_form_row(form, "Temperature", row)

        self._btn_neutral = QPushButton("Neutral")
        self._btn_neutral.setToolTip("Untinted greys (the default)")
        self._btn_neutral.clicked.connect(lambda: self._slider.setValue(0))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.addButton(self._btn_neutral, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.rejected.connect(self.accept)
        form.addRow(buttons)

        # Restyling the whole app takes a moment: apply the slider when it pauses
        self._apply_timer = QTimer(self, singleShot=True, interval=60)
        self._apply_timer.timeout.connect(self._apply_temperature)
        self._slider.valueChanged.connect(self._on_slider)

    def _on_slider(self, value: int):
        if value and abs(value) <= self.SNAP:
            self._slider.setValue(0)
            return
        self._apply_timer.start()

    def _apply_temperature(self):
        set_theme(temperature=self._slider.value())


class MotionDialog(QDialog):
    """How long a wheel step glides when scrolling and when zooming a plot,
    applied at once and saved by theme.set_motion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scrolling and zooming")
        form = dialog_form(self)
        self._spin_scroll = self._duration(
            Motion.SCROLL_MS, "How long a wheel step glides in lists, tables, panels and documents",
            lambda ms: set_motion(scroll_ms=ms))
        add_form_row(form, "Scrolling", self._spin_scroll)
        self._spin_zoom = self._duration(
            Motion.ZOOM_MS, "How long a wheel step glides when zooming a plot",
            lambda ms: set_motion(zoom_ms=ms))
        add_form_row(form, "Zooming", self._spin_zoom)

        self._btn_default = QPushButton("Default")
        self._btn_default.setToolTip(
            f"{Motion.DEFAULT_SCROLL_MS} milliseconds scrolling, {Motion.DEFAULT_ZOOM_MS} zooming")
        self._btn_default.clicked.connect(self._restore_defaults)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.addButton(self._btn_default, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.rejected.connect(self.accept)
        form.addRow(buttons)

    @staticmethod
    def _duration(ms: int, tooltip: str, apply) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, Motion.MAX_MS)
        spin.setSingleStep(25)
        spin.setSuffix(" milliseconds")
        # 0: the wheel jumps, without a glide
        spin.setSpecialValueText("Off")
        spin.setValue(ms)
        spin.setToolTip(tooltip + "; off jumps at once")
        spin.valueChanged.connect(apply)
        return spin

    def _restore_defaults(self):
        self._spin_scroll.setValue(Motion.DEFAULT_SCROLL_MS)
        self._spin_zoom.setValue(Motion.DEFAULT_ZOOM_MS)
