"""
Background Work
===============

Runs long computations (fits, reports) on a worker thread while the GUI
keeps processing events, so the window never freezes.

The worker must not touch Qt widgets. It hands progress to the GUI thread
through plain Python state, which ``on_tick`` (run on the GUI thread)
reads and draws.
"""

from __future__ import annotations

import sys
import threading
import traceback
from collections.abc import Callable
from typing import TypeVar

from PyQt6.QtCore import QEventLoop, QObject, QTimer, pyqtSignal, pyqtSlot

T = TypeVar("T")

# How often on_tick runs while the work is in progress
TICK_MS = 50

# GIL switch interval (s) while work runs; Python's default is 0.005
SWITCH_INTERVAL = 0.0005


class _Ticker(QObject):
    """Lives on the GUI thread; a request from the worker runs the tick there."""

    requested = pyqtSignal()

    def __init__(self, tick: Callable[[], None]):
        super().__init__()
        self._tick = tick
        self.active = True
        self.requested.connect(self._run)

    @pyqtSlot()
    def _run(self):
        if self.active:
            self._tick()


# The tickers of the running background work, outermost first: work started
# while other work runs (a fit report during a fit) nests its event loop
_tickers: list[_Ticker] = []


def request_tick() -> None:
    """Run ``on_tick`` of the running background work as soon as the GUI
    thread is free, instead of at the next timer tick (callable from any thread)."""
    for ticker in list(_tickers):
        ticker.requested.emit()


def _run_tick(on_tick: Callable[[], None]) -> None:
    """Call ``on_tick``; a drawing error is reported, not raised, so it can
    neither stop the event loop from ending nor replace the work's outcome."""
    try:
        on_tick()
    except Exception:
        traceback.print_exc()


def run_in_background(work: Callable[[], T], on_tick: Callable[[], None] | None = None) -> T:
    """Run ``work()`` on a worker thread and return its result.

    Blocks the caller but not the GUI: a local event loop keeps handling
    paint, input and timer events until the worker finishes. ``on_tick`` is
    called on the GUI thread every ``TICK_MS`` and once more at the end;
    its exceptions are printed. Exceptions raised by ``work`` are re-raised here.
    """
    outcome: dict = {}

    def target():
        try:
            outcome["value"] = work()
        except BaseException as e:  # re-raised on the GUI thread
            outcome["error"] = e

    thread = threading.Thread(target=target, name="spline-fitter-work", daemon=True)
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(TICK_MS)

    def tick():
        if on_tick is not None:
            _run_tick(on_tick)
        if not thread.is_alive():
            loop.quit()

    timer.timeout.connect(tick)
    ticker = _Ticker(tick)
    _tickers.append(ticker)
    # Painting calls into Python many times (pyqtgraph items), and each call
    # waits for the worker to hand over the GIL; a short switch interval
    # keeps those waits small so repaints stay fast
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_INTERVAL)
    try:
        thread.start()
        timer.start()
        if thread.is_alive():
            loop.exec()
        timer.stop()
        thread.join()
    finally:
        sys.setswitchinterval(switch_interval)
        ticker.active = False
        _tickers.remove(ticker)
    if on_tick is not None:
        _run_tick(on_tick)

    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
