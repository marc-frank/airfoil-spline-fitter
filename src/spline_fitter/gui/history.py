"""
Settings History
================

Undo and redo of the sidebar's fit settings (Edit > Undo, Ctrl+Z).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager

from PyQt6.QtCore import QEvent, QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QWidget,
)

from .widgets import SegmentedControl, SweepableDoubleSpinBox, SweepableSpinBox

# Changes closer together than this are one step: typing a number, holding
# a spin box arrow, or the settings of a run applied at once
MERGE_MS = 500
# Steps kept
LIMIT = 200

_SWEEPABLE = (SweepableSpinBox, SweepableDoubleSpinBox)


def widget_value(w: QWidget):
    """The value of a settings widget."""
    if isinstance(w, QLineEdit):
        return w.text()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, (QComboBox, SegmentedControl)):
        return w.currentIndex()
    if isinstance(w, QAbstractSpinBox):
        return w.value()
    if isinstance(w, _SWEEPABLE):
        return w.sweep_state()
    raise TypeError(type(w))


def set_widget_value(w: QWidget, value) -> None:
    """Set a settings widget to a value from widget_value()."""
    if isinstance(w, QLineEdit):
        w.setText(value)
    elif isinstance(w, QCheckBox):
        w.setChecked(value)
    elif isinstance(w, (QComboBox, SegmentedControl)):
        w.setCurrentIndex(value)
    elif isinstance(w, QAbstractSpinBox):
        w.setValue(value)
    elif isinstance(w, _SWEEPABLE):
        w.set_sweep_state(value)
    else:
        raise TypeError(type(w))


def _value_signal(w: QWidget):
    if isinstance(w, QLineEdit):
        return w.textChanged
    if isinstance(w, QCheckBox):
        return w.toggled
    if isinstance(w, (QComboBox, SegmentedControl)):
        return w.currentIndexChanged
    if isinstance(w, _SWEEPABLE):
        return w.sweep_changed
    return w.valueChanged


class SettingsHistory(QObject):
    """The states of a list of settings widgets, one per step.

    Widgets are restored in list order, so a setting that limits another
    (the degree limits the control point count) comes first. Changes made
    while ``is_quiet()`` holds, or inside ``quiet()``, are not steps of their
    own: they become part of the current state (the automatic tolerance of a
    new airfoil). Quiet and user changes close together stay apart: the user
    change is a step either way.
    """

    changed = pyqtSignal()

    def __init__(self, widgets: Sequence[QWidget], is_quiet: Callable[[], bool] = lambda: False,
                 after_restore: Callable[[], None] = lambda: None, parent=None):
        super().__init__(parent)
        self._widgets = list(widgets)
        self._is_quiet = is_quiet
        self._after_restore = after_restore
        self._restoring = False
        self._quiet_depth = 0
        self._pending_quiet = False
        # The state after the latest change, while a step is pending
        self._latest_state: tuple | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(MERGE_MS)
        self._timer.timeout.connect(self.commit)
        self._states = [self._capture()]
        self._index = 0
        for w in self._widgets:
            _value_signal(w).connect(self._on_change)
        # A focused spin box or line edit would take Ctrl+Z for its own text
        QApplication.instance().installEventFilter(self)

    def _capture(self) -> tuple:
        return tuple(widget_value(w) for w in self._widgets)

    def _on_change(self, *_):
        if self._restoring:
            return
        quiet = bool(self._quiet_depth or self._is_quiet())
        pending = self._timer.isActive()
        if pending and quiet != self._pending_quiet and self._latest_state is not None:
            # The other kind of change is pending: settle it first, as it
            # was before this change
            self._settle(self._latest_state, self._pending_quiet)
        self._pending_quiet = quiet
        self._latest_state = self._capture()
        self._timer.start()
        self.changed.emit()

    @contextmanager
    def quiet(self):
        """Changes made inside are part of the current state, not a step."""
        self._quiet_depth += 1
        try:
            yield
        finally:
            self._quiet_depth -= 1

    def commit(self) -> None:
        """Make the changes since the last step a step now."""
        self._timer.stop()
        quiet, self._pending_quiet = self._pending_quiet, False
        self._latest_state = None
        self._settle(self._capture(), quiet)
        self.changed.emit()

    def _settle(self, state: tuple, quiet: bool) -> None:
        """Make ``state`` part of the current state (quiet) or a new step."""
        if quiet:
            self._states[self._index] = state
        elif state != self._states[self._index]:
            del self._states[self._index + 1:]
            self._states.append(state)
            del self._states[:-LIMIT]
            self._index = len(self._states) - 1

    def checkpoint(self) -> tuple[list, int]:
        """The steps as they are now, to go back to with :meth:`rollback`."""
        self.commit()
        return list(self._states), self._index

    def rollback(self, checkpoint: tuple[list, int]) -> None:
        """Forget every change since ``checkpoint``: an action that failed
        and put the widgets back leaves no step."""
        self._timer.stop()
        self._pending_quiet, self._latest_state = False, None
        states, self._index = checkpoint
        self._states = list(states)
        self.changed.emit()

    def can_undo(self) -> bool:
        return self._index > 0 or self._timer.isActive() and not self._pending_quiet

    def can_redo(self) -> bool:
        return self._index < len(self._states) - 1 and not self._timer.isActive()

    def undo(self) -> None:
        self.commit()
        if self._index > 0:
            self._index -= 1
            self._restore()

    def redo(self) -> None:
        self.commit()
        if self._index < len(self._states) - 1:
            self._index += 1
            self._restore()

    def _restore(self) -> None:
        self._restoring = True
        try:
            for w, value in zip(self._widgets, self._states[self._index], strict=True):
                set_widget_value(w, value)
            self._after_restore()
        finally:
            self._restoring = False
        # What the widgets settled on (a limit may clamp a value)
        self._states[self._index] = self._capture()
        self.changed.emit()

    def eventFilter(self, obj, event):
        """Keep Ctrl+Z and its redo keys for the settings while one has focus."""
        if (event.type() == QEvent.Type.ShortcutOverride
                and (event.matches(QKeySequence.StandardKey.Undo)
                     or event.matches(QKeySequence.StandardKey.Redo))
                and isinstance(obj, QWidget)
                and any(w is obj or w.isAncestorOf(obj) for w in self._widgets)):
            return True
        return super().eventFilter(obj, event)
