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


def test_a_group_is_one_step_even_when_the_timer_expires_inside():
    user, auto, history = _history()
    user.setValue(1)  # pending: the group commits it as a step of its own
    with history.group():
        user.setValue(2)
        assert not history._timer.isActive()  # no merge timer to expire mid-group
        auto.setValue(3)
        with history.group():  # an inner group is part of the outer one
            auto.setValue(4)
        assert len(history._states) == 2
    assert len(history._states) == 3
    history.undo()
    assert (user.value(), auto.value()) == (1, 0)
    history.undo()
    assert (user.value(), auto.value()) == (0, 0)
