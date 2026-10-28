"""
Fitting Controller
==================

Manages spline fitting operations: single fit, parameter sweep, and cancellation.

Extracted from SplineFitterWindow to isolate optimization orchestration logic.
All window widget access is routed through ``self._window``.
"""

from __future__ import annotations

import time
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QApplication

from ..analysis import fit_report, summary_line
from ..formatting import format_duration, format_error
from ..geometry import compute_optimization_bounds
from ..optimization import (
    OptimizerErrorLogger,
    construct_control_points,
    get_evaluation_count,
    reset_iteration_tracking,
    run_optimization,
)
from ..optimization.models import (
    AirfoilData,
    FairnessWeights,
    FitResult,
    IterationRecord,
    OptimizationConfig,
    SplineConfig,
)
from .background import run_in_background
from .live import LiveProgress

if TYPE_CHECKING:
    from .ui import SplineFitterWindow


class FittingController:
    """Manages single-fit and parameter-sweep optimization operations."""

    def __init__(self, window: SplineFitterWindow):
        self._window = window
        self._sweep_results: list[tuple[int | float, FitResult]] = []
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
            fairness_weights = w._get_fairness_weights()
        except ValueError as e:
            w.statusBar().showMessage(str(e))
            return

        # Check for active parameter sweep
        active_sweep = w._get_active_sweep()
        if active_sweep is not None:
            self._run_sweep_fit(config, opt_config, fairness_weights, active_sweep)
            return

        # Clear sweep state when doing single fit
        self._sweep_results = []
        self._sweep_mode = False

        # Show setup message
        n_ctrl = config.num_control_points
        degree = config.degree
        mode = ", ".join(config.junction_specs()) or config.continuity.name
        size = (f"tolerance {opt_config.tolerance:.2e}" if opt_config.tolerance
                else f"{n_ctrl} control points")
        w.statusBar().showMessage(
            f"Preparing {size}, degree {degree}, {mode} continuity..."
        )
        QApplication.processEvents()

        # Compute bounds
        bounds = compute_optimization_bounds(
            w._airfoil, config,
            optimize_le_position=opt_config.optimize_le_position,
        )
        n_vars = len(bounds)

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
        details = f"{n_vars} variables"
        if opt_config.optimizer.uses_error_metric:
            details += f", {opt_config.metric.display_name} error"
        w.statusBar().showMessage(f"Starting {opt_name} ({details})...")

        live = LiveProgress(w, config, opt_config, error_logger, start_time)

        # Get param optimizer settings
        param_opt_outer = w._get_param_opt_settings()
        orth_local_bounds = w._check_orth_lsq_local_bounds.isChecked()
        airfoil = w._airfoil

        def work():
            """Run the optimization and measure the result (worker thread)."""
            result = run_optimization(
                config,
                opt_config,
                airfoil,
                fairness_weights,
                bounds,
                callback=live.callback,
                param_opt_outer_optimizer=param_opt_outer,
                orth_lsq_use_local_bounds=orth_local_bounds,
                should_stop=lambda: w._cancel_requested,
            )
            elapsed = time.perf_counter() - start_time
            fit_result = self._fit_result(result, config, opt_config, airfoil, elapsed, live)
            return result, fit_result, elapsed

        try:
            # Run optimization off the GUI thread
            result, fit_result, elapsed = run_in_background(work, live.on_tick)

            if fit_result is not None:
                w._fit_result = fit_result

                # Add optimization run to project
                w._project_manager.add_optimization_run(
                    spline_config=config,
                    opt_config=opt_config,
                    fairness_weights=fairness_weights,
                    result=w._fit_result,
                    loss_history=self._loss_history(error_logger),
                )

                # Plot final result
                w._plot_manager.plot_fitted_spline(w._fit_result)
                w._refresh_curvature_comb()

            # Always update loss plot at completion (for Minimal mode)
            live.show_loss()

            # Save log
            cont_level = config.continuity.name[1]
            extra_info = f"mode=C{cont_level}, degree={config.degree}"
            log_path = error_logger.save(
                optimizer_key=opt_config.optimizer.value,
                extra_info=extra_info,
            )
            if log_path:
                print(f"Log saved: {log_path}")

            # Status message with detailed info
            eval_count = get_evaluation_count()
            iterations = result.nit
            opt_name = opt_config.optimizer.display_name

            if w._cancel_requested:
                msg = (
                    f"Cancelled: {iterations} iters, {eval_count} evals "
                    f"in {format_duration(elapsed)} | Best ˜ {format_error(result.fun)}"
                )
            elif live.early_stopped:
                msg = (
                    f"Converged ({opt_name}): {iterations} iters, {eval_count} evals "
                    f"in {format_duration(elapsed)} | Error ˜ {format_error(result.fun)}"
                )
            else:
                msg = (
                    f"Completed ({opt_name}): {iterations} iters, {eval_count} evals "
                    f"in {format_duration(elapsed)} | Error ˜ {format_error(result.fun)}"
                )

            if w._fit_result is not None and w._fit_result.report:
                msg += " | " + summary_line(w._fit_result.report)
            if not result.success and not w._cancel_requested:
                msg += " | Solver did not converge: " + result.message
            w.statusBar().showMessage(msg)

            # Play completion chime (only if not cancelled)
            if not w._cancel_requested:
                w._play_completion_chime()

        except ValueError as e:
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
                    airfoil: AirfoilData, elapsed: float, live: LiveProgress) -> FitResult | None:
        """The measured FitResult of an optimizer result (worker thread)."""
        w = self._window
        if result.x is None:
            return None
        if result.section_control_points is not None:
            section_ctrl_pts = result.section_control_points
        else:
            section_ctrl_pts = construct_control_points(
                result.x, config,
                [list(sec) for sec in airfoil.sections],
                optimize_le_position=opt_config.optimize_le_position,
            )
        fit_result = FitResult(
            section_control_points=section_ctrl_pts,
            degree=config.degree,
            final_error=result.fun,
            num_iterations=result.nit,
            num_evaluations=get_evaluation_count(),
            elapsed_seconds=elapsed,
            converged=result.success and not (w._cancel_requested or live.early_stopped),
            early_stopped=live.early_stopped,
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

    def _run_sweep_fit(
        self,
        config: SplineConfig,
        opt_config: OptimizationConfig,
        fairness_weights: FairnessWeights,
        sweep_box=None,
    ):
        """Run sweep of fits with varying parameter values.

        Each fit is shown live like a single fit, and the comparison plot
        gains a point (and the best marker moves) after every fit.
        """
        w = self._window

        if sweep_box is None:
            return

        min_val, max_val = sweep_box.sweep_range

        # Get param optimizer settings
        param_opt_outer = w._get_param_opt_settings()
        orth_local_bounds = w._check_orth_lsq_local_bounds.isChecked()
        airfoil = w._airfoil

        # Ensure min <= max
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        # Determine which parameter is being swept
        is_opt_config_sweep = False
        if sweep_box is w._sweep_ctrl:
            param_name = "Control Points"
            actual_min = config.continuity.minimum_control_points(config.degree)
            if min_val < actual_min:
                min_val = actual_min
                w.statusBar().showMessage(
                    f"Adjusted min to {actual_min} "
                    f"(required for degree {config.degree}, {config.continuity.name})"
                )
                QApplication.processEvents()
            sweep_values = list(range(int(min_val), int(max_val) + 1))

            def make_config(v):
                return dc_replace(config, num_control_points=v), opt_config
        elif sweep_box is w._sweep_degree:
            param_name = "Degree"
            sweep_values = list(range(int(min_val), int(max_val) + 1))

            def make_config(v):
                return dc_replace(config, degree=v), opt_config
        elif sweep_box is w._sweep_smoothing_weight:
            param_name = "Smoothing Weight"
            is_opt_config_sweep = True
            n_steps = sweep_box.sweep_steps
            if n_steps < 2:
                n_steps = 2
            sweep_values = [
                min_val + (max_val - min_val) * i / (n_steps - 1)
                for i in range(n_steps)
            ]

            def make_config(v):
                return config, dc_replace(opt_config, smoothing_weight=v)
        else:
            return

        # Reset state
        w._cancel_requested = False
        self._sweep_results = []
        self._sweep_mode = True
        # The comparison plot shows this sweep among the project's earlier
        # runs of the same configuration
        self._sweep_axis = (config, opt_config, param_name)
        w._plot_manager.clear_plot_items()
        w._plot_manager.set_comparison_label(param_name)
        w._plot_manager.set_comparison_data([], [])
        w._plot_manager.set_comparison_marker([], [])

        w._set_fitting_ui(True)
        sweep_start_time = time.perf_counter()

        total_fits = len(sweep_values)

        w.statusBar().showMessage(
            f"Sweep: fitting {total_fits} splines "
            f"({param_name} {min_val}–{max_val})..."
        )
        QApplication.processEvents()

        # Perform sweep
        for idx, val in enumerate(sweep_values):
            if w._cancel_requested:
                break

            sweep_config, sweep_opt_config = make_config(val)
            val_str = f"{val:.5g}" if is_opt_config_sweep else str(val)

            bounds = compute_optimization_bounds(
                w._airfoil, sweep_config,
                optimize_le_position=sweep_opt_config.optimize_le_position,
            )

            reset_iteration_tracking()
            self._reset_loss_plot()
            fit_start_time = time.perf_counter()

            run_name = w._airfoil.name or w._airfoil.filename
            error_logger = OptimizerErrorLogger(run_name=run_name)
            live = LiveProgress(
                w, sweep_config, sweep_opt_config, error_logger, fit_start_time,
                status_prefix=f"Sweep {idx + 1}/{total_fits} | {param_name}={val_str} | ",
            )

            def sweep_work(sweep_config=sweep_config, sweep_opt_config=sweep_opt_config,
                           bounds=bounds, fit_start_time=fit_start_time, live=live):
                """Run one sweep fit and measure it (worker thread)."""
                result = run_optimization(
                    sweep_config,
                    sweep_opt_config,
                    airfoil,
                    fairness_weights,
                    bounds,
                    callback=live.callback,
                    param_opt_outer_optimizer=param_opt_outer,
                    orth_lsq_use_local_bounds=orth_local_bounds,
                    should_stop=lambda: w._cancel_requested,
                )
                return self._fit_result(result, sweep_config, sweep_opt_config, airfoil,
                                        time.perf_counter() - fit_start_time, live)

            try:
                fit_result = run_in_background(sweep_work, live.on_tick)
                live.show_loss()

                if fit_result is not None:
                    self._sweep_results.append((val, fit_result))

                    # Add to project
                    w._project_manager.add_optimization_run(
                        spline_config=sweep_config,
                        opt_config=sweep_opt_config,
                        fairness_weights=fairness_weights,
                        result=fit_result,
                        loss_history=self._loss_history(error_logger),
                    )

                    # Show the finished fit and its point on the comparison plot
                    w._plot_manager.plot_fitted_spline(fit_result)
                    w._fit_result = fit_result
                    w._refresh_curvature_comb()
                    self._show_sweep_comparison()

            except Exception as e:
                print(f"Sweep fit failed for {param_name}={val}: {e}")

            # Update progress
            progress = (idx + 1) / total_fits * 100
            elapsed = time.perf_counter() - sweep_start_time
            if self._sweep_results and self._sweep_results[-1][0] == val:
                last_error = self._sweep_results[-1][1].final_error
                w.statusBar().showMessage(
                    f"Sweep: {idx + 1}/{total_fits} ({progress:.0f}%) | "
                    f"{param_name}={val_str}, err={format_error(last_error)} | "
                    f"t={format_duration(elapsed)}"
                )
            else:
                w.statusBar().showMessage(
                    f"Sweep: {idx + 1}/{total_fits} ({progress:.0f}%) | "
                    f"{param_name}={val_str} | t={format_duration(elapsed)}"
                )
            QApplication.processEvents()

        elapsed = time.perf_counter() - sweep_start_time

        if self._sweep_results:
            best_val, best_result = self._show_sweep_comparison()

            sweep_box.spinbox.setValue(best_val)

            w._fit_result = best_result
            w._plot_manager.plot_fitted_spline(w._fit_result)
            w._refresh_curvature_comb()
            w._set_save_enabled(True)

            best_val_str = f"{best_val:.5g}" if is_opt_config_sweep else str(best_val)
            msg = (
                f"Sweep complete: {len(self._sweep_results)} fits "
                f"in {format_duration(elapsed)} | "
                f"Best: {param_name}={best_val_str}, "
                f"error ˜ {format_error(best_result.final_error)}"
            )
        else:
            msg = "Sweep failed - no successful fits"

        w.statusBar().showMessage(msg)

        if not w._cancel_requested and self._sweep_results:
            w._play_completion_chime()

        w._set_fitting_ui(False)

    def _show_sweep_comparison(self) -> tuple[int | float, FitResult]:
        """Plot the errors of all project runs along the sweep axis and mark
        the best of this sweep; returns it."""
        plots = self._window._plot_manager
        x, y = self._window._project_manager.sweep_series(*self._sweep_axis)
        plots.set_comparison_data(x, y)
        best_val, best_result = min(self._sweep_results, key=lambda vr: vr[1].final_error)
        plots.set_comparison_marker([best_val], [best_result.final_error])
        return best_val, best_result
