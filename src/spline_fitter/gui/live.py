"""
Live Preview
============

Progress of one running fit, drawn while it runs.

The optimizer callback runs on the worker thread and must not touch widgets:
``LiveProgress.callback`` only records, ``LiveProgress.on_tick`` (GUI thread)
draws. The Live preview setting picks what is drawn:

    Minimal   only the status bar; the plots show the finished fit
    Moderate  the latest state once a second
    Detailed  every step: the worker waits until each step is drawn
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from ..formatting import format_duration, format_error
from ..optimization import OptimizerErrorLogger, get_evaluation_count, increment_iteration
from ..optimization.models import AirfoilData, OptimizationConfig, SplineConfig
from .background import request_tick

if TYPE_CHECKING:
    from .ui import SplineFitterWindow

MINIMAL, MODERATE, DETAILED = 0, 1, 2

# Seconds between drawings in Moderate mode, and between status updates
MODERATE_INTERVAL = 1.0
STATUS_INTERVAL = 0.5


class LiveProgress:
    """Records the progress of one fit and draws it on the GUI thread."""

    def __init__(
        self,
        window: SplineFitterWindow,
        config: SplineConfig,
        opt_config: OptimizationConfig,
        error_logger: OptimizerErrorLogger,
        start_time: float,
        status_prefix: str = "",
        airfoil: AirfoilData | None = None,
    ):
        """``airfoil`` is the one being fitted, if not the window's (batch)."""
        self._w = window
        self._airfoil = airfoil or window._airfoil
        self._config = config
        self._opt_config = opt_config
        self._logger = error_logger
        self._start = start_time
        self._prefix = status_prefix

        # VarPro tolerance mode: solver steps reported, last refit's max deviation
        self._steps = False
        self._deviation: float | None = None

        self._latest: dict = {}
        self._drawn = threading.Event()
        self._last_plot = start_time
        self._last_status = start_time

    @property
    def mode(self) -> int:
        """The live preview setting, which can change while fitting."""
        return self._w._live_mode

    # -- Worker thread -----------------------------------------------------

    def callback(self, x: np.ndarray, error: float, preview=None, step: bool = False) -> bool:
        """Optimizer progress (worker thread); True stops the fit.

        ``step=True`` marks a solver step inside a VarPro tolerance refit
        (error = RMS error). Once steps arrive, they are the loss curve, and
        the refits between them (error = max deviation, the same curve as
        the last step) only update the status.
        """
        w = self._w
        if w._cancel_requested:
            return True

        if step:
            self._steps = True
        elif self._steps:
            self._deviation = error
            return False
        self._record(error)

        elapsed = time.perf_counter() - self._start
        status = (f"{self._prefix}Fitting... {format_duration(elapsed)} | "
                  f"Iteration {len(self._logger.errors)} | "
                  f"Evaluations {get_evaluation_count()} | ")
        if self._steps:
            status += f"RMS error {format_error(error)}"
            if self._deviation is not None:
                status += f" | Max deviation of last refit {format_error(self._deviation)}"
        else:
            status += f"Best error {format_error(min(self._logger.errors))}"
        # Cleared before the step is published: a tick that draws it in
        # between sets the event after this, never before
        self._drawn.clear()
        self._latest.update(x=np.array(x, copy=True), preview=preview, new=True, status=status)

        if self.mode == DETAILED:
            # Hand the step to the GUI and wait until it is drawn
            request_tick()
            while not self._drawn.wait(0.05):
                if w._cancel_requested:
                    return True
        return False

    def _record(self, error: float) -> None:
        """Log one iteration."""
        iteration = increment_iteration()
        self._logger.log(iteration, get_evaluation_count(), error,
                         time.perf_counter() - self._start)

    # -- GUI thread --------------------------------------------------------

    def on_tick(self):
        """Draw the latest progress as the mode asks (GUI thread)."""
        if not self._latest.get("new"):
            return
        now = time.perf_counter()
        if self.mode == DETAILED or (
                self.mode == MODERATE and now - self._last_plot >= MODERATE_INTERVAL):
            self._last_plot = now
            try:
                self._draw()
            finally:
                self._drawn.set()
        else:
            # Release a worker still waiting from Detailed mode
            self._drawn.set()
            if now - self._last_status >= STATUS_INTERVAL:
                self._last_status = now
                self._w.statusBar().showMessage(self._latest["status"])

    def _draw(self):
        w = self._w
        self._latest["new"] = False
        self.show_loss()
        w._plot_manager.update_live_plot(
            self._latest["x"], self._config, preview=self._latest["preview"],
            tolerance=self._opt_config.tolerance)
        w.statusBar().showMessage(self._latest["status"])

    def show_loss(self):
        """Put the logged iterations on the loss plot (GUI thread)."""
        w = self._w
        w._plot_manager.update_loss_data(
            list(self._logger.iterations),
            list(self._logger.elapsed_times),
            list(self._logger.errors),
        )
        if w._plot_manager.has_loss_data:
            w._plot_manager.set_loss_curve_data(w._combo_loss_x.currentData())
