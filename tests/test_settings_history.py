"""Undo steps of the settings: quiet changes and the user's stay apart."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QSpinBox  # noqa: E402

from spline_fitter.gui.history import SettingsHistory  # noqa: E402

_APP = QApplication.instance() or QApplication([])


def _history():
    user, auto = QSpinBox(), QSpinBox()
    return user, auto, SettingsHistory([user, auto])


def test_quiet_change_right_after_a_user_change_keeps_the_users_step():
    user, auto, history = _history()
    user.setValue(1)
    with history.quiet():
        auto.setValue(5)
    history.commit()
    assert history.can_undo()
    history.undo()
    assert user.value() == 0


def test_user_change_right_after_a_quiet_change_is_a_step():
    user, auto, history = _history()
    with history.quiet():
        auto.setValue(5)
    user.setValue(1)
    history.commit()
    history.undo()
    # Back to before the user's change, with the quiet change kept
    assert (user.value(), auto.value()) == (0, 5)
    assert not history.can_undo()
