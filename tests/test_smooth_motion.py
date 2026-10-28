"""Wheel scrolling and plot zooming glide instead of jumping (gui/motion.py)."""
import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QWheelEvent  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QListWidget, QScrollArea, QWidget  # noqa: E402

from spline_fitter.gui.motion import SmoothViewBox, install_smooth_scrolling  # noqa: E402
from spline_fitter.gui.theme import Motion  # noqa: E402


# Kept for the whole session: later tests share this application
_APP = QApplication.instance() or QApplication([])


@pytest.fixture
def app():
    install_smooth_scrolling(_APP)
    return _APP


def _wheel(widget, notches=-1, modifiers=Qt.KeyboardModifier.NoModifier):
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    event = QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(), QPoint(0, 120 * notches),
                        Qt.MouseButton.NoButton, modifiers, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


def test_list_scrolls_by_pixel_and_glides(app):
    view = QListWidget()
    view.addItems([f"Run {i}" for i in range(500)])
    view.resize(200, 200)
    view.show()
    QTest.qWait(10)
    assert view.verticalScrollMode() == view.ScrollMode.ScrollPerPixel
    bar = view.verticalScrollBar()
    _wheel(view.viewport(), -1)
    QTest.qWait(Motion.SCROLL_MS // 4)
    midway = bar.value()
    QTest.qWait(Motion.SCROLL_MS * 2)
    final = bar.value()
    assert 0 < midway < final
    assert final == QApplication.wheelScrollLines() * bar.singleStep()
    view.close()


def test_notches_during_a_glide_add_up(app):
    area = QScrollArea()
    content = QWidget()
    content.setMinimumHeight(5000)
    area.setWidget(content)
    area.resize(200, 200)
    area.show()
    QTest.qWait(10)
    bar = area.verticalScrollBar()
    for _ in range(3):
        _wheel(area.viewport(), -1)
        QTest.qWait(10)
    QTest.qWait(Motion.SCROLL_MS * 2)
    assert bar.value() == 3 * QApplication.wheelScrollLines() * bar.singleStep()
    # Back past the top stops at the top
    _wheel(area.viewport(), 20)
    QTest.qWait(Motion.SCROLL_MS * 2)
    assert bar.value() == 0
    area.close()


def test_modified_wheel_keeps_qt_handling(app):
    view = QListWidget()
    view.addItems([f"Run {i}" for i in range(500)])
    view.resize(200, 200)
    view.show()
    QTest.qWait(10)
    bar = view.verticalScrollBar()
    _wheel(view.viewport(), -1, Qt.KeyboardModifier.ShiftModifier)
    # Qt scrolls at once (by a page), with no glide
    assert bar.value() > 0
    view.close()


class _FakeWheel:
    def __init__(self, delta, pos):
        self._delta, self._pos = delta, pos

    def delta(self):
        return self._delta

    def pos(self):
        return self._pos

    def accept(self):
        pass


def test_plot_zoom_glides_to_pyqtgraph_scale(app):
    import pyqtgraph as pg

    plot = pg.PlotWidget(viewBox=SmoothViewBox())
    plot.resize(400, 400)
    plot.show()
    vb = plot.getViewBox()
    vb.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)
    QTest.qWait(10)
    width = vb.viewRange()[0][1] - vb.viewRange()[0][0]
    center = vb.mapViewToScene(pg.Point(5, 5))
    center = vb.mapFromScene(center)
    vb.wheelEvent(_FakeWheel(120, center))
    QTest.qWait(Motion.ZOOM_MS // 4)
    midway = vb.viewRange()[0][1] - vb.viewRange()[0][0]
    QTest.qWait(Motion.ZOOM_MS * 2)
    final = vb.viewRange()[0][1] - vb.viewRange()[0][0]
    expected = width * 1.02 ** (120 * vb.state["wheelScaleFactor"])
    assert final < midway < width
    assert math.isclose(final, expected, rel_tol=1e-6)
    plot.close()
