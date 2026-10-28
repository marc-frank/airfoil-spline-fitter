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
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .theme import Size, glyph_size, icon

# =============================================================================
# Constants
# =============================================================================

# Maximum number of points to plot in the loss graph (to prevent UI freeze)
MAX_PLOT_POINTS = 100_000


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
# Sweepable SpinBox Widget
# =============================================================================

class SweepableSpinBox(QWidget):
    """A spinbox with an inline sweep-range toggle button.

    When the [\u2194] button is toggled, a \u2192 [max] field appears, turning
    the single value into a range for parameter sweeps.  Only one
    SweepableSpinBox in a *group* can be active at a time.
    """

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

        self._btn_sweep = QPushButton("\u2194")
        self._btn_sweep.setObjectName("SweepToggle")
        self._btn_sweep.setFixedSize(Size.CONTROL_HEIGHT, Size.CONTROL_HEIGHT)
        self._btn_sweep.setToolTip("Toggle parameter sweep range")
        self._btn_sweep.setCheckable(True)
        self._btn_sweep.clicked.connect(self._on_toggle_sweep)
        lay.addWidget(self._btn_sweep)

        self._lbl_arrow = QLabel("\u2192")
        self._lbl_arrow.setVisible(False)
        lay.addWidget(self._lbl_arrow)

        self._spin_max = QSpinBox()
        self._spin_max.setRange(self._spin.minimum(), self._spin.maximum())
        self._spin_max.setSingleStep(self._spin.singleStep())
        self._spin_max.setValue(min(self._spin.value() * 2, self._spin.maximum()))
        self._spin_max.setVisible(False)
        lay.addWidget(self._spin_max, stretch=1)

    # ------------------------------------------------------------------
    def _on_toggle_sweep(self, checked: bool):
        if checked and self._group:
            for other in self._group:
                if other is not self and other._sweep_active:
                    other._btn_sweep.setChecked(False)
                    other._sweep_active = False
                    other._lbl_arrow.setVisible(False)
                    other._spin_max.setVisible(False)
        self._sweep_active = checked
        self._lbl_arrow.setVisible(checked)
        self._spin_max.setVisible(checked)
        if checked:
            self._spin_max.setMinimum(self._spin.minimum())
            self._spin_max.setValue(min(self._spin.value() * 2, self._spin.maximum()))

    def stop_sweep(self) -> None:
        """Leave sweep mode."""
        self._btn_sweep.setChecked(False)
        self._on_toggle_sweep(False)

    def set_sweep_allowed(self, allowed: bool) -> None:
        """Enable or disable the sweep toggle, leaving sweep mode if needed."""
        if not allowed and self._sweep_active:
            self.stop_sweep()
        self._btn_sweep.setEnabled(allowed)

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
    """A double-spinbox with an inline sweep-range toggle and step count.

    When the [\u2194] button is toggled, \u2192 [max] and Steps fields appear,
    turning the single value into a range for parameter sweeps.  Only one
    sweepable widget in a *group* can be active at a time (works with
    both SweepableSpinBox and SweepableDoubleSpinBox in the same group).
    """

    def __init__(
        self,
        spin_box: QDoubleSpinBox,
        group: list | None = None,
        default_steps: int = 10,
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

        self._btn_sweep = QPushButton("\u2194")
        self._btn_sweep.setObjectName("SweepToggle")
        self._btn_sweep.setFixedSize(Size.CONTROL_HEIGHT, Size.CONTROL_HEIGHT)
        self._btn_sweep.setToolTip("Toggle parameter sweep range")
        self._btn_sweep.setCheckable(True)
        self._btn_sweep.clicked.connect(self._on_toggle_sweep)
        lay.addWidget(self._btn_sweep)

        self._lbl_arrow = QLabel("\u2192")
        self._lbl_arrow.setVisible(False)
        lay.addWidget(self._lbl_arrow)

        self._spin_max = QDoubleSpinBox()
        self._spin_max.setRange(self._spin.minimum(), self._spin.maximum())
        self._spin_max.setDecimals(self._spin.decimals())
        self._spin_max.setSingleStep(self._spin.singleStep())
        self._spin_max.setValue(min(self._spin.value() * 10, self._spin.maximum()))
        self._spin_max.setVisible(False)
        lay.addWidget(self._spin_max, stretch=1)

        self._lbl_steps = QLabel("Steps:")
        self._lbl_steps.setVisible(False)
        lay.addWidget(self._lbl_steps)

        self._spin_steps = QSpinBox()
        self._spin_steps.setRange(2, 100)
        self._spin_steps.setValue(default_steps)
        self._spin_steps.setToolTip("Number of evenly-spaced values in sweep")
        self._spin_steps.setVisible(False)
        lay.addWidget(self._spin_steps)

    # ------------------------------------------------------------------
    def _on_toggle_sweep(self, checked: bool):
        if checked and self._group:
            for other in self._group:
                if other is not self and other._sweep_active:
                    other._btn_sweep.setChecked(False)
                    other._sweep_active = False
                    other._lbl_arrow.setVisible(False)
                    if hasattr(other, '_spin_max'):
                        other._spin_max.setVisible(False)
                    if hasattr(other, '_lbl_steps'):
                        other._lbl_steps.setVisible(False)
                    if hasattr(other, '_spin_steps'):
                        other._spin_steps.setVisible(False)
        self._sweep_active = checked
        self._lbl_arrow.setVisible(checked)
        self._spin_max.setVisible(checked)
        self._lbl_steps.setVisible(checked)
        self._spin_steps.setVisible(checked)
        if checked:
            self._spin_max.setMinimum(self._spin.minimum())
            self._spin_max.setValue(min(self._spin.value() * 10, self._spin.maximum()))

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
    def sweep_steps(self) -> int:
        """Return the number of sweep steps."""
        return self._spin_steps.value()

    @property
    def spinbox(self) -> QDoubleSpinBox:
        return self._spin
