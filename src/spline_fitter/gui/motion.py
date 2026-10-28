"""
Motion: smooth wheel scrolling and zooming
==========================================

A mouse wheel moves in notches; drawn as they come, each notch jumps. Here
every notch starts a short eased animation instead, and notches that arrive
while one runs extend it, so a quick spin glides to where it would have
jumped to.

    - install_smooth_scrolling(app): every scroll bar the wheel moves, in
      lists, tables, panels and documents alike, scrolls smoothly
    - SmoothViewBox: a plot view box whose wheel zoom is animated; pass
      ``viewBox=SmoothViewBox()`` to ``pg.PlotWidget``

Durations are the ``Motion`` tokens in theme.py, read at each wheel step,
so a change in View > Scrolling and zooming... applies at once; with 0 the
wheel jumps as it did without this module.
"""

from __future__ import annotations

import math

import pyqtgraph as pg
from pyqtgraph import functions as fn
from PyQt6.QtCore import QEasingCurve, QEvent, QObject, Qt, QVariantAnimation
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QApplication,
    QGraphicsView,
    QScrollBar,
    QWidget,
)

from .theme import Motion

# One wheel notch, in eighths of a degree
_NOTCH = 120


class _ScrollAnimation(QVariantAnimation):
    """Eases one scroll bar towards a target that each wheel notch moves on."""

    def __init__(self, bar: QScrollBar):
        super().__init__(bar)
        self._bar = bar
        self.target = 0.0
        self.setEasingCurve(QEasingCurve.Type.OutCubic)
        # Start and end are both floats, or the animation cannot interpolate
        self.valueChanged.connect(lambda value: bar.setValue(round(value)))
        # Dragging the handle takes over from the animation
        bar.sliderPressed.connect(self.stop)

    def scroll_by(self, distance: float) -> bool:
        """Start (or extend) the glide; False if the bar is already at that end."""
        bar = self._bar
        running = self.state() == QVariantAnimation.State.Running
        current = float(self.currentValue()) if running else float(bar.value())
        start = self.target if running else current
        target = float(min(max(start + distance, bar.minimum()), bar.maximum()))
        if round(target) == round(current) and not running:
            return False
        self.stop()
        self.setDuration(Motion.SCROLL_MS)  # read now: the user may have changed it
        self.target = target
        self.setStartValue(current)
        self.setEndValue(target)
        self.start()
        return True


class SmoothScroller(QObject):
    """Application event filter that turns wheel notches into glides.

    Item views scroll by pixel rather than by row, so they can glide too.
    Wheel events with modifiers (page steps, text zoom), from touchpads that
    already send pixel deltas, and over plots (which zoom) keep Qt's own
    handling, as do bars already at the end they would move to, so the wheel
    passes on to an enclosing scroll area.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind == QEvent.Type.Polish and isinstance(obj, QAbstractItemView):
            obj.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
            obj.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        elif kind == QEvent.Type.Wheel and isinstance(obj, QWidget):
            return self._wheel(obj, event)
        return False

    def _wheel(self, widget: QWidget, event) -> bool:
        if event.modifiers() != Qt.KeyboardModifier.NoModifier:
            return False
        if not event.pixelDelta().isNull():
            return False
        delta = event.angleDelta()
        vertical = abs(delta.y()) >= abs(delta.x())
        notches = (delta.y() if vertical else delta.x()) / _NOTCH
        if not notches:
            return False
        bar = _scroll_bar(widget, vertical)
        if bar is None or not bar.isEnabled() or bar.maximum() == bar.minimum():
            return False
        if bar.orientation() == Qt.Orientation.Horizontal and widget.isRightToLeft():
            notches = -notches
        distance = -notches * QApplication.wheelScrollLines() * bar.singleStep()
        animation = bar.findChild(_ScrollAnimation) or _ScrollAnimation(bar)
        if not animation.scroll_by(distance):
            return False
        event.accept()
        return True


def _scroll_bar(widget: QWidget, vertical: bool) -> QScrollBar | None:
    """The scroll bar a wheel event on ``widget`` moves, if any."""
    if isinstance(widget, QScrollBar):
        return widget
    area = widget.parent()
    if not (isinstance(area, QAbstractScrollArea) and area.viewport() is widget):
        return None
    if isinstance(area, QGraphicsView):  # plots zoom with the wheel
        return None
    return area.verticalScrollBar() if vertical else area.horizontalScrollBar()


def install_smooth_scrolling(app: QApplication | None = None) -> None:
    """Make wheel scrolling smooth throughout the app; safe to repeat."""
    app = app or QApplication.instance()
    if app is None or app.findChild(SmoothScroller):
        return
    app.installEventFilter(SmoothScroller(app))


class SmoothViewBox(pg.ViewBox):
    """A view box that zooms to the wheel in a short glide, about the cursor.

    The total zoom of a notch is pyqtgraph's own; it is applied in eased
    steps, and notches during a glide add to what is left of it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zoom = QVariantAnimation(self)
        self._zoom.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._zoom.setStartValue(0.0)
        self._zoom.setEndValue(1.0)
        self._zoom.valueChanged.connect(self._zoom_step)
        self._zoom_log = 0.0       # log of this glide's total scale
        self._zoom_done = 0.0      # eased progress already applied
        self._zoom_mask = [True, True]
        self._zoom_center = pg.Point(0, 0)

    def wheelEvent(self, ev, axis=None):
        if axis in (0, 1):
            mask = [False, False]
            mask[axis] = self.state['mouseEnabled'][axis]
        else:
            mask = self.state['mouseEnabled'][:]
        log = math.log(1.02) * ev.delta() * self.state['wheelScaleFactor']
        remaining = 0.0
        if self._zoom.state() == QVariantAnimation.State.Running:
            remaining = self._zoom_log * (1.0 - self._zoom_done)
        self._zoom.stop()
        self._zoom_log = remaining + log
        self._zoom_done = 0.0
        self._zoom_mask = mask
        self._zoom_center = pg.Point(
            fn.invertQTransform(self.childGroup.transform()).map(ev.pos()))
        self._resetTarget()
        self._zoom.setDuration(Motion.ZOOM_MS)
        self._zoom.start()
        ev.accept()
        self.sigRangeChangedManually.emit(mask)

    def _zoom_step(self, progress: float) -> None:
        step = progress - self._zoom_done
        if not step:
            return
        self._zoom_done = progress
        s = math.exp(self._zoom_log * step)
        self.scaleBy([None if m is False else s for m in self._zoom_mask], self._zoom_center)
