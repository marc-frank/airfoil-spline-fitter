"""
Fitting Controller
==================

Manages spline fitting operations: single fit, parameter sweep, and cancellation.

Extracted from SplineFitterWindow to isolate optimization orchestration logic.
All window widget access is routed through ``self._window``.
"""

from __future__ import annotations

import itertools
import time
import traceback
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QApplication

from ..analysis import fit_report, summary_line
from ..formatting import format_duration, format_error, number
from ..geometry import create_airfoil_data
from ..optimization import (
    OptimizerErrorLogger,
    get_evaluation_count,
    reset_iteration_tracking,
    run_optimization,
)
from ..optimization.models import (
    AirfoilData,
    FitResult,
    IterationRecord,
    OptimizationConfig,
    SplineConfig,
)
from .background import run_in_background
from .live import LiveProgress

if TYPE_CHECKING:
    from .ui import SplineFitterWindow


def continuity_text(config: SplineConfig) -> str:
    """The continuity of every junction, as fitted: ``G3``, or per junction
    ``G2/C1/G3``; ``none`` for a single section."""
    specs = config.junction_specs()
    if not specs:
        return "none"
    return specs[0] if len(set(specs)) == 1 else "/".join(specs)


def _count(value: int, noun: str) -> str:
    return f"{value} {noun}" + ("" if value == 1 else "s")


def fit_effort(result, opt_config: OptimizationConfig, evaluations: int) -> str:
    """What a fit took: its residual evaluations, and the refits of a
    tolerance search; Direct LSQ evaluates nothing, it solves once."""
    if opt_config.tolerance is not None:
        return f"{_count(result.nit, 'refit')}, {_count(evaluations, 'evaluation')}"
    if evaluations:
        return _count(evaluations, "evaluation")
    return _count(result.nit, "solve")


class FittingController:
    """Manages single-fit and parameter-sweep optimization operations."""

    def __init__(self, window: SplineFitterWindow):
        self._window = window
        # (the swept values, one per swept parameter; the fit)
        self._sweep_results: list[tuple[tuple, FitResult]] = []
        self._sweep_mode = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_fit(self):
        """Run spline fitting on the current airfoil."""
        w = self._window

        if w._airfoil is None:
            w.statusBar().showMessage("Load an airfoil first.")
            return

        try:
            config = w._get_spline_config()
            opt_config = w._get_optimization_config()
        except ValueError as e:
            w.statusBar().showMessage(str(e))
            return

        # Check for active parameter sweeps
        active_sweeps = w._get_active_sweeps()
        if active_sweeps:
            self._run_sweep_fit(config, opt_config, active_sweeps)
            return

        # Clear sweep state when doing single fit
        self._sweep_results = []
        self._sweep_mode = False

        # Show setup message
        n_ctrl = config.num_control_points
        degree = config.degree
        mode = ", ".join(config.junction_specs()) or config.continuity.name
        size = (f"tolerance {number(opt_config.tolerance, '.2e')}" if opt_config.tolerance
                else f"{n_ctrl} control points")
        w.statusBar().showMessage(
            f"Preparing {size}, degree {degree}, {mode} continuity..."
        )
        QApplication.processEvents()

        # Reset state
        w._cancel_requested = False
        w._plot_manager.clear_plot_items()
        self._reset_loss_plot()
        reset_iteration_tracking()

        w._set_fitting_ui(True)
        start_time = time.perf_counter()

        # Initialize logger
        run_name = w._airfoil.name or w._airfoil.filename
        error_logger = OptimizerErrorLogger(run_name=run_name)

        # Show initial status
        opt_name = opt_config.optimizer.display_name
        w.statusBar().showMessage(f"Starting {opt_name}...")

        live = LiveProgress(w, config, opt_config, error_logger, start_time)

        airfoil = w._airfoil

        def work():
            """Run the optimization and measure the result (worker thread)."""
            result = run_optimization(
                config,
                opt_config,
                airfoil,
                callback=live.callback,
                should_stop=lambda: w._cancel_requested,
            )
            elapsed = time.perf_counter() - start_time
            fit_result = self._fit_result(result, config, opt_config, airfoil, elapsed)
            return result, fit_result, elapsed

        try:
            # Run optimization off the GUI thread
            result, fit_result, elapsed = run_in_background(work, live.on_tick)

            kept = False
            if fit_result is not None:
                w._fit_result = fit_result

                # Add optimization run to project
                kept = not w._project_manager.add_optimization_run(
                    spline_config=config,
                    opt_config=opt_config,
                    result=w._fit_result,
                    loss_history=self._loss_history(error_logger),
                )

                # Plot final result
                w._plot_manager.plot_fitted_spline(w._fit_result)
                w._refresh_curvature_comb()
            elif w._fit_result is not None:
                # No fit to show: the current fit stays, and is drawn again
                w._plot_manager.plot_fitted_spline(w._fit_result)
                w._refresh_curvature_comb()

            # Always update loss plot at completion (for Minimal mode)
            live.show_loss()

            # Save log
            extra_info = f"continuity={continuity_text(config)}, degree={config.degree}"
            log_path = error_logger.save(
                optimizer_key=opt_config.optimizer.value,
                extra_info=extra_info,
            )
            if log_path:
                print(f"Log saved: {log_path}")

            # Status message with detailed info
            effort = fit_effort(result, opt_config, get_evaluation_count())
            opt_name = opt_config.optimizer.display_name

            if w._cancel_requested:
                msg = (
                    f"Cancelled: {effort} "
                    f"in {format_duration(elapsed)} | Best error {format_error(result.fun)}"
                )
            else:
                msg = (
                    f"Completed ({opt_name}): {effort} "
                    f"in {format_duration(elapsed)} | Error {format_error(result.fun)}"
                )

            # The summary of this run's fit only, never of an earlier one
            if fit_result is not None and fit_result.report:
                msg += " | " + summary_line(fit_result.report)
            if not result.success and not w._cancel_requested:
                msg += " | Solver did not converge: " + result.message
            if kept:
                msg += " | The project keeps its finished run of these settings"
            w.statusBar().showMessage(msg)

            # Play completion chime (only if not cancelled)
            if not w._cancel_requested:
                w._play_completion_chime()

        except Exception as e:
            traceback.print_exc()
            w.statusBar().showMessage(f"Fitting error: {e}")
            w._play_error_chime()

        finally:
            w._set_fitting_ui(False)

    def _reset_loss_plot(self):
        """Empty the loss plot with the selected axis and scale."""
        w = self._window
        x_label = "Iterations" if w._combo_loss_x.currentData() == "iters" else "Time (s)"
        scale = w._combo_log_scale.currentData()
        w._plot_manager.reset_loss_plot(x_label, scale in ("both", "x"), scale in ("both", "y"))

    def _fit_result(self, result, config: SplineConfig, opt_config: OptimizationConfig,
                    airfoil: AirfoilData, elapsed: float) -> FitResult | None:
        """The measured FitResult of an optimizer result (worker thread)."""
        w = self._window
        if result.x is None:
            return None
        fit_result = FitResult(
            section_control_points=result.section_control_points,
            degree=config.degree,
            final_error=result.fun,
            num_iterations=result.nit,
            num_evaluations=get_evaluation_count(),
            elapsed_seconds=elapsed,
            converged=result.success and not w._cancel_requested,
            cancelled=w._cancel_requested,
            section_knots=result.section_knots,
        )
        self._attach_report(fit_result, opt_config, airfoil)
        return fit_result

    @staticmethod
    def _loss_history(error_logger: OptimizerErrorLogger) -> list[IterationRecord]:
        return [
            IterationRecord(iteration_number=it, error_value=err, elapsed_seconds=t)
            for it, err, t in zip(
                error_logger.iterations,
                error_logger.errors,
                error_logger.elapsed_times, strict=False,
            )
        ]

    def _attach_report(self, fit_result: FitResult, opt_config: OptimizationConfig,
                       airfoil: AirfoilData) -> None:
        """Measure deviation, curvature and junctions of a fit (never fails the fit)."""
        try:
            fit_result.report = fit_report(
                airfoil.sections, fit_result,
                precision=airfoil.precision, tolerance=opt_config.tolerance,
            )
        except Exception as e:
            print(f"Fit report failed: {e}")
            fit_result.report = None

    def on_cancel(self):
        """Handle cancel button click."""
        self._window._cancel_requested = True
        self._window.statusBar().showMessage("Cancelling... please wait")

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------

    def _sweep_axis_of(self, box) -> tuple[str, list, object] | None:
        """A swept parameter: its name, its values and how one value changes
        (spline config, optimization config)."""
        w = self._window
        if box is w._sweep_smoothing_weight:
            return ("Smoothing weight", box.sweep_values(),
                    lambda c, o, v: (c, dc_replace(o, smoothing_weight=v)))
        low, high = sorted(box.sweep_range)
        values = list(range(int(low), int(high) + 1))
        if box is w._sweep_ctrl:
            return ("Control points", values,
                    lambda c, o, v: (dc_replace(c, num_control_points=v), o))
        if box is w._sweep_degree:
            return "Degree", values, lambda c, o, v: (dc_replace(c, degree=v), o)
        if box is w._sweep_sections:
            def with_sections(c, o, v):
                # Per-junction overrides fit one section count only
                junctions = c.junction_continuity
                if junctions and len(junctions) != v - 1:
                    junctions = ()
                return dc_replace(c, num_sections=v, junction_continuity=junctions), o
            return "Sections", values, with_sections
        return None

    def _run_sweep_fit(
        self,
        config: SplineConfig,
        opt_config: OptimizationConfig,
        sweep_boxes: list,
    ):
        """Run one fit per combination of the swept parameters' values.

        Each fit is shown live like a single fit, and the comparison plot
        gains a point (and the best marker moves) after every fit.
        """
        w = self._window

        axes = [(box, axis) for box in sweep_boxes
                if (axis := self._sweep_axis_of(box)) is not None]
        if not axes:
            return
        names = [name for _, (name, _, _) in axes]

        def make_configs(values):
            c, o = config, opt_config
            for (_, (_, _, apply)), v in zip(axes, values, strict=True):
                c, o = apply(c, o, v)
            return c, o

        # Every combination, less those with too few control points for
        # their degree and junctions (the sidebar allows them only when the
        # count is swept along with the degree or the sections)
        combinations = []
        too_few = 0
        for values in itertools.product(*(vals for _, (_, vals, _) in axes)):
            c, _ = make_configs(values)
            if c.num_control_points < c.minimum_control_points:
                too_few += 1
            else:
                combinations.append(values)

        def describe(values):
            return ", ".join(
                f"{name} {number(v, '.5g') if isinstance(v, float) else v}"
                for name, v in zip(names, values, strict=True))

        # Reset state
        w._cancel_requested = False
        self._sweep_results = []
        self._sweep_mode = True
        # The comparison plot shows this sweep among the project's earlier
        # runs of the same configuration, along the parameter with the most
        # values (the first of them in the sidebar on a tie)
        x_axis = max(range(len(axes)), key=lambda i: (len(axes[i][1][1]), -i))
        self._sweep_axis = (config, opt_config, tuple(names), names[x_axis])
        self._sweep_x = x_axis
        w._plot_manager.clear_plot_items()
        w._project_manager.show_axis(names[x_axis])
        w._plot_manager.set_comparison_data([], [])
        w._plot_manager.set_comparison_marker([], [])

        msg = f"Sweep: fitting {len(combinations)} splines ({', '.join(names)})"
        if too_few:
            msg += (f" | Left out {too_few} combinations with too few control points "
                    "for their degree and junctions")

        w._set_fitting_ui(True)
        try:
            elapsed, failed = self._sweep(combinations, make_configs, describe, msg)
        finally:
            w._set_fitting_ui(False)

        if self._sweep_results:
            best_values, best_result = self._show_sweep_comparison()

            # The best values become the settings, as one step Undo takes
            # back to the sweep: the sweep is over, and its lower ends are
            # these values' boxes, so leaving it on would narrow it
            self._apply_sweep_values(axes, best_values)

            w._fit_result = best_result
            w._plot_manager.plot_fitted_spline(w._fit_result)
            w._refresh_curvature_comb()
            w._set_save_enabled(True)

            msg = (
                f"Sweep complete: {len(self._sweep_results)} fits "
                f"in {format_duration(elapsed)} | "
                f"Best: {describe(best_values)}, "
                f"error {format_error(best_result.final_error)}"
            )
            if failed:
                msg += f" | {failed} failed"
        else:
            msg = "Sweep failed: no fit succeeded"

        w.statusBar().showMessage(msg)

        if failed:
            w._play_error_chime()
        elif not w._cancel_requested and self._sweep_results:
            w._play_completion_chime()

    def _sweep(self, combinations: list, make_configs, describe, start_message: str
               ) -> tuple[float, int]:
        """Fit every combination (UI locked); the time taken and the fits that failed."""
        w = self._window
        airfoil = w._airfoil
        sweep_start_time = time.perf_counter()
        total_fits = len(combinations)
        failed = 0

        w.statusBar().showMessage(start_message + "...")
        QApplication.processEvents()

        # Perform sweep
        for idx, values in enumerate(combinations):
            if w._cancel_requested:
                break

            label = describe(values)
            fit_result = None
            error = None
            try:
                sweep_config, sweep_opt_config = make_configs(values)
                if sweep_config.num_sections == airfoil.num_sections:
                    sweep_airfoil = airfoil
                else:
                    sweep_airfoil = create_airfoil_data(
                        list(airfoil.coordinates), filename=airfoil.filename,
                        name=airfoil.name, precision=airfoil.precision,
                        num_sections=sweep_config.num_sections,
                    )

                # The live preview replaces the previous fit, as in a single fit
                w._plot_manager.clear_plot_items()
                reset_iteration_tracking()
                self._reset_loss_plot()
                fit_start_time = time.perf_counter()

                run_name = airfoil.name or airfoil.filename
                error_logger = OptimizerErrorLogger(run_name=run_name)
                live = LiveProgress(
                    w, sweep_config, sweep_opt_config, error_logger, fit_start_time,
                    status_prefix=f"Sweep {idx + 1}/{total_fits} | {label} | ",
                    airfoil=sweep_airfoil,
                )

                def sweep_work(sweep_config=sweep_config, sweep_opt_config=sweep_opt_config,
                               fit_start_time=fit_start_time, live=live,
                               airfoil=sweep_airfoil):
                    """Run one sweep fit and measure it (worker thread)."""
                    result = run_optimization(
                        sweep_config,
                        sweep_opt_config,
                        airfoil,
                        callback=live.callback,
                        should_stop=lambda: w._cancel_requested,
                    )
                    return self._fit_result(result, sweep_config, sweep_opt_config, airfoil,
                                            time.perf_counter() - fit_start_time)

                fit_result = run_in_background(sweep_work, live.on_tick)
                live.show_loss()

                if fit_result is not None:
                    self._sweep_results.append((values, fit_result))

                    # Add to project
                    w._project_manager.add_optimization_run(
                        spline_config=sweep_config,
                        opt_config=sweep_opt_config,
                        result=fit_result,
                        loss_history=self._loss_history(error_logger),
                        replot=False,
                    )

                    # Show the finished fit and its point on the comparison plot
                    w._plot_manager.plot_fitted_spline(fit_result)
                    w._fit_result = fit_result
                    w._refresh_curvature_comb()
                    self._show_sweep_comparison()

            except Exception as e:
                traceback.print_exc()
                error = e
                failed += 1

            # Update progress
            progress = (idx + 1) / total_fits * 100
            elapsed = time.perf_counter() - sweep_start_time
            done = f"Sweep: {idx + 1}/{total_fits} ({number(progress, '.0f')}%) | {label}"
            if error is not None:
                done += f" | Failed: {error}"
            elif fit_result is not None:
                done += f", error {format_error(fit_result.final_error)}"
            w.statusBar().showMessage(f"{done} | {format_duration(elapsed)}")
            QApplication.processEvents()

        return time.perf_counter() - sweep_start_time, failed

    def _apply_sweep_values(self, axes: list, values: tuple) -> None:
        """Leave sweep mode and set the swept settings to ``values``: the
        section count first, then the degree, as each limits the next."""
        order = {self._window._sweep_sections: 0, self._window._sweep_degree: 1}
        boxes = sorted(zip((box for box, _ in axes), values, strict=True),
                       key=lambda bv: order.get(bv[0], 2))
        for box, _ in boxes:
            box.stop_sweep()
        for box, v in boxes:
            box.spinbox.setValue(v)

    def _show_sweep_comparison(self) -> tuple[tuple, FitResult]:
        """Plot the errors of all project runs along the sweep's x axis and
        mark the best of this sweep; returns it."""
        plots = self._window._plot_manager
        x, y = self._window._project_manager.sweep_series(*self._sweep_axis)
        # Several swept parameters put several fits at one x value
        plots.set_comparison_data(x, y, connected=len(set(x)) == len(x))
        best_values, best_result = min(self._sweep_results,
                                       key=lambda vr: vr[1].final_error)
        plots.set_comparison_marker([best_values[self._sweep_x]], [best_result.final_error])
        return best_values, best_result
