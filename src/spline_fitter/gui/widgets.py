"""
Reusable UI Widgets
===================

Custom PyQt6 widgets used by the spline fitter GUI.

This module provides:
    - Colors: Application color constants
    - CollapsibleSection: Expandable/collapsible section widget
    - SweepableSpinBox: Spinbox with parameter sweep toggle
"""

from __future__ import annotations

from typing import Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
)


# =============================================================================
# Constants
# =============================================================================

# Maximum number of points to plot in the loss graph (to prevent UI freeze)
MAX_PLOT_POINTS = 100_000


# =============================================================================
# UI Color Scheme
# =============================================================================

class Colors:
    """Application color constants."""

    # Data points
    AIRFOIL_POINTS = (255, 165, 0)  # Orange

    # Curves
    SPLINE_CURVE = (255, 165, 0)  # Orange
    CONTROL_POLYGON = (160, 160, 160)  # Gray
    KNOTS = (0, 200, 120)  # Green

    # Curvature comb
    COMB_UPPER = (200, 0, 200)  # Purple
    COMB_LOWER = (0, 120, 255)  # Blue
    COMB_TIPS_UPPER = (130, 0, 130)  # Dark purple
    COMB_TIPS_LOWER = (0, 70, 180)  # Dark blue

    # Loss plot
    LOSS_POINTS = (100, 200, 255)  # Light blue


# =============================================================================
# Collapsible Section Widget
# =============================================================================

class CollapsibleSection(QWidget):
    """A collapsible section with a clickable header that toggles content visibility."""

    def __init__(self, title: str, parent=None, expanded: bool = True):
        super().__init__(parent)
        self._title = title
        self._expanded = expanded

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header button
        self._toggle_btn = QPushButton(self._header_text())
        self._toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; padding: 6px 8px; "
            "border: none; border-radius: 3px; background: palette(midlight); }"
            "QPushButton:hover { background: palette(mid); }"
        )
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.clicked.connect(self._on_toggle)
        main_layout.addWidget(self._toggle_btn)

        # Content area
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 4, 4, 8)
        self._content_layout.setSpacing(6)
        self._content.setVisible(expanded)
        main_layout.addWidget(self._content)

    def _header_text(self) -> str:
        arrow = "\u25BC" if self._expanded else "\u25B6"
        return f"{arrow}  {self._title}"

    def _on_toggle(self):
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._toggle_btn.setText(self._header_text())

    def content_layout(self) -> QVBoxLayout:
        """Return the layout for adding child widgets."""
        return self._content_layout

    def add_widget(self, widget: QWidget):
        """Add a widget to the content area."""
        self._content_layout.addWidget(widget)

    def add_layout(self, layout):
        """Add a layout to the content area."""
        self._content_layout.addLayout(layout)

    def set_expanded(self, expanded: bool):
        """Programmatically expand or collapse."""
        self._expanded = expanded
        self._content.setVisible(expanded)
        self._toggle_btn.setText(self._header_text())


# =============================================================================
# Sweepable SpinBox Widget
# =============================================================================

class SweepableSpinBox(QWidget):
    """A spinbox with an inline sweep-range toggle button.

    When the [\u2194] button is toggled, a \u2192 [max] field appears, turning
    the single value into a range for parameter sweeps.  Only one
    SweepableSpinBox in a *group* can be active at a time.
    """

    def __init__(self, spin_box: QSpinBox, group: Optional[list] = None, parent=None):
        super().__init__(parent)
        self._sweep_active = False
        self._group = group
        if group is not None:
            group.append(self)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._spin = spin_box
        lay.addWidget(self._spin, stretch=1)

        self._btn_sweep = QPushButton("\u2194")
        self._btn_sweep.setFixedSize(28, 28)
        self._btn_sweep.setToolTip("Toggle parameter sweep range")
        self._btn_sweep.setCheckable(True)
        self._btn_sweep.setStyleSheet(
            "QPushButton { font-size: 14px; border: 1px solid palette(mid); border-radius: 3px; }"
            "QPushButton:checked { background: palette(highlight); color: palette(highlighted-text); }"
        )
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

    @property
    def is_sweep_active(self) -> bool:
        return self._sweep_active

    @property
    def sweep_range(self) -> Tuple[int, int]:
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
        group: Optional[list] = None,
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
        lay.setSpacing(4)

        self._spin = spin_box
        lay.addWidget(self._spin, stretch=1)

        self._btn_sweep = QPushButton("\u2194")
        self._btn_sweep.setFixedSize(28, 28)
        self._btn_sweep.setToolTip("Toggle parameter sweep range")
        self._btn_sweep.setCheckable(True)
        self._btn_sweep.setStyleSheet(
            "QPushButton { font-size: 14px; border: 1px solid palette(mid); border-radius: 3px; }"
            "QPushButton:checked { background: palette(highlight); color: palette(highlighted-text); }"
        )
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
    def sweep_range(self) -> Tuple[float, float]:
        """Return (min_val, max_val) for the sweep."""
        return (self._spin.value(), self._spin_max.value())

    @property
    def sweep_steps(self) -> int:
        """Return the number of sweep steps."""
        return self._spin_steps.value()

    @property
    def spinbox(self) -> QDoubleSpinBox:
        return self._spin
