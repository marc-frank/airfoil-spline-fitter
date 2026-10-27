"""
Fitting Controller
==================

Manages spline fitting operations: single fit, parameter sweep, and cancellation.

Extracted from SplineFitterWindow to isolate optimization orchestration logic.
All window widget access is routed through ``self._window``.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtWidgets import QApplication

from ..analysis import fit_report, summary_line
from ..formatting import format_duration, format_error
from ..geometry import compute_optimization_bounds
from ..optimization import (
    OptimizerErrorLogger,
    construct_control_points,
    get_evaluation_count,
    increment_iteration,
    reset_iteration_tracking,
    run_optimization,
)
from ..optimization.models import (
    FairnessWeights,
    FitResult,
    IterationRecord,
    OptimizationConfig,
    SplineConfig,
)

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
        mode = w._combo_loss_x.currentData()
        x_label = "Iterations" if mode == "iters" else "Time (s)"
        scale = w._combo_log_scale.currentData()
        log_x = scale in ("both", "x")
        log_y = scale in ("both", "y")
        w._plot_manager.reset_loss_plot(x_label, log_x, log_y)
        reset_iteration_tracking()

        w._set_fitting_ui(True)
        start_time = time.perf_counter()

        # Early stopping state
        best_error = float('inf')
        patience = max(50, opt_config.max_iterations // 20)
        no_improvement = 0
        early_stopped = False

        # Progress tracking for non-live mode
        last_status_update = start_time
        status_update_interval = 0.5

        # Initialize logger
        run_name = w._airfoil.name or w._airfoil.filename
        error_logger = OptimizerErrorLogger(run_name=run_name)

        # Show initial status
        opt_name = opt_config.optimizer.display_name
        metric_name = opt_config.metric.display_name
        w.statusBar().showMessage(
            f"Starting {opt_name} ({n_vars} variables, {metric_name} error)..."
        )
        QApplication.processEvents()

        def iteration_callback(x: np.ndarray, error: float, preview=None) -> bool:
            """Handle optimizer iteration updates."""
            nonlocal best_error, no_improvement, early_stopped, last_status_update

            # Check cancellation
            if w._cancel_requested:
                return True

            # Get current state
            eval_count = get_evaluation_count()
            iteration = increment_iteration()

            # Calculate elapsed time
            current_time = time.perf_counter()
            elapsed = current_time - start_time

            # Log error with elapsed time
            error_logger.log(iteration, eval_count, error, elapsed)

            # Update loss plot data from logger
            w._plot_manager.update_loss_data(
                list(error_logger.iterations),
                list(error_logger.elapsed_times),
                error_logger.errors,
            )

            # Find best error
            best_so_far = min(error_logger.errors) if error_logger.errors else error

            # Check for improvement
            if math.isfinite(best_so_far) and best_so_far < best_error:
                if math.isfinite(best_error) and abs(best_error) > 0:
                    improvement = (best_error - best_so_far) / abs(best_error)
                else:
                    improvement = 1.0
                if improvement > 1e-6:
                    no_improvement = 0
                else:
                    no_improvement += 1
                best_error = best_so_far
            else:
                no_improvement += 1

            # Early stopping
            if no_improvement >= patience and opt_config.optimizer.value != "gn-varpro":
                early_stopped = True
                return True

            # Get live preview mode: 0=Minimal, 1=Moderate, 2=Detailed
            live_mode = w._combo_live.currentIndex()

            # Check if optimizer uses direct control points (iterative algebraic)
            is_direct_cp = opt_config.optimizer.is_iterative_algebraic

            # Build status message
            status_msg = (
                f"Fitting... t={format_duration(elapsed)} | "
                f"Iter {iteration} | Evals {eval_count} | Best \u02dc {format_error(best_so_far)}"
            )

            if live_mode == 2:
                # Detailed: update plots on every callback
                w._plot_manager.update_live_plot(
                    x, config, w._airfoil.sections,
                    opt_config.optimize_le_position, is_direct_cp, preview=preview,
                )
                w._plot_manager.set_loss_curve_data(w._combo_loss_x.currentData())
                w.statusBar().showMessage(status_msg)

            elif live_mode == 1:
                # Moderate: update plots every 1 second
                if current_time - last_status_update >= 1.0:
                    last_status_update = current_time
                    w._plot_manager.update_live_plot(
                        x, config, w._airfoil.sections,
                        opt_config.optimize_le_position, is_direct_cp, preview=preview,
                    )
                    w._plot_manager.set_loss_curve_data(w._combo_loss_x.currentData())
                    w.statusBar().showMessage(status_msg)

            else:
                # Minimal: only status bar updates (every 0.5s)
                if current_time - last_status_update >= status_update_interval:
                    last_status_update = current_time
                    w.statusBar().showMessage(status_msg)

            QApplication.processEvents()
            return False

        # Get param optimizer settings
        param_opt_outer = w._get_param_opt_settings()

        try:
            # Run optimization
            result = run_optimization(
                config,
                opt_config,
                w._airfoil,
                fairness_weights,
                bounds,
                callback=iteration_callback,
                param_opt_outer_optimizer=param_opt_outer,
                orth_lsq_use_local_bounds=w._check_orth_lsq_local_bounds.isChecked(),
            )

            elapsed = time.perf_counter() - start_time

            if result.x is not None:
                # Build fit result
                if result.section_control_points is not None:
                    section_ctrl_pts = result.section_control_points
                else:
                    section_ctrl_pts = construct_control_points(
                        result.x, config,
                        [list(sec) for sec in w._airfoil.sections],
                        optimize_le_position=opt_config.optimize_le_position,
                    )

                w._fit_result = FitResult(
                    section_control_points=section_ctrl_pts,
                    degree=config.degree,
                    final_error=result.fun,
                    num_iterations=result.nit,
                    num_evaluations=get_evaluation_count(),
                    elapsed_seconds=elapsed,
                    converged=result.success and not (w._cancel_requested or early_stopped),
                    early_stopped=early_stopped,
                    cancelled=w._cancel_requested,
                    section_knots=result.section_knots,
                )
                self._attach_report(w._fit_result, opt_config)

                # Build loss history from logger
                loss_history = [
                    IterationRecord(
                        iteration_number=it,
                        error_value=err,
                        elapsed_seconds=t,
                    )
                    for it, err, t in zip(
                        error_logger.iterations,
                        error_logger.errors,
                        error_logger.elapsed_times, strict=False,
                    )
                ]

                # Add optimization run to project
                w._project_manager.add_optimization_run(
                    spline_config=config,
                    opt_config=opt_config,
                    fairness_weights=fairness_weights,
                    result=w._fit_result,
                    loss_history=loss_history,
                )

                # Plot final result
                w._plot_manager.plot_fitted_spline(w._fit_result)
                w._refresh_curvature_comb()

            # Always update loss plot at completion (for Minimal mode)
            if w._plot_manager.has_loss_data:
                w._plot_manager.set_loss_curve_data(w._combo_loss_x.currentData())

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
                    f"in {format_duration(elapsed)} | Best \u02dc {format_error(result.fun)}"
                )
            elif early_stopped:
                msg = (
                    f"Converged ({opt_name}): {iterations} iters, {eval_count} evals "
                    f"in {format_duration(elapsed)} | Error \u02dc {format_error(result.fun)}"
                )
            else:
                msg = (
                    f"Completed ({opt_name}): {iterations} iters, {eval_count} evals "
                    f"in {format_duration(elapsed)} | Error \u02dc {format_error(result.fun)}"
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

    def _attach_report(self, fit_result: FitResult, opt_config: OptimizationConfig) -> None:
        """Measure deviation, curvature and junctions of a fit (never fails the fit)."""
        w = self._window
        try:
            fit_result.report = fit_report(
                w._airfoil.sections, fit_result,
                precision=w._airfoil.precision, tolerance=opt_config.tolerance,
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
        """Run sweep of fits with varying parameter values."""
        w = self._window

        if sweep_box is None:
            return

        min_val, max_val = sweep_box.sweep_range

        # Get param optimizer settings
        param_opt_outer = w._get_param_opt_settings()

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
        w._plot_manager.clear_plot_items()

        w._set_fitting_ui(True)
        sweep_start_time = time.perf_counter()

        total_fits = len(sweep_values)

        w.statusBar().showMessage(
            f"Sweep: fitting {total_fits} splines "
            f"({param_name} {min_val}\u2013{max_val})..."
        )
        QApplication.processEvents()

        # Perform sweep
        for idx, val in enumerate(sweep_values):
            if w._cancel_requested:
                break

            sweep_config, sweep_opt_config = make_config(val)

            bounds = compute_optimization_bounds(
                w._airfoil, sweep_config,
                optimize_le_position=sweep_opt_config.optimize_le_position,
            )

            reset_iteration_tracking()
            fit_start_time = time.perf_counter()

            run_name = w._airfoil.name or w._airfoil.filename
            error_logger = OptimizerErrorLogger(run_name=run_name)

            # Early stopping state for this fit
            best_error = float('inf')
            patience = max(50, opt_config.max_iterations // 20)
            no_improvement = 0
            early_stopped = False

            def sweep_iteration_callback(x: np.ndarray, error: float) -> bool:
                """Handle optimizer iteration updates during sweep."""
                nonlocal best_error, no_improvement, early_stopped

                if w._cancel_requested:
                    return True

                eval_count = get_evaluation_count()
                iteration = increment_iteration()

                current_time = time.perf_counter()
                elapsed = current_time - fit_start_time

                error_logger.log(iteration, eval_count, error, elapsed)

                best_so_far = min(error_logger.errors) if error_logger.errors else error

                if math.isfinite(best_so_far) and best_so_far < best_error:
                    improvement = (
                        (best_error - best_so_far) / abs(best_error)
                        if abs(best_error) > 0 else 1.0
                    )
                    if improvement > 1e-6:
                        no_improvement = 0
                    else:
                        no_improvement += 1
                    best_error = best_so_far
                else:
                    no_improvement += 1

                if no_improvement >= patience and opt_config.optimizer.value != "gn-varpro":
                    early_stopped = True
                    return True

                return False

            try:
                result = run_optimization(
                    sweep_config,
                    sweep_opt_config,
                    w._airfoil,
                    fairness_weights,
                    bounds,
                    callback=sweep_iteration_callback,
                    param_opt_outer_optimizer=param_opt_outer,
                    orth_lsq_use_local_bounds=w._check_orth_lsq_local_bounds.isChecked(),
                )

                fit_elapsed = time.perf_counter() - fit_start_time

                if result.x is not None:
                    if result.section_control_points is not None:
                        section_ctrl_pts = result.section_control_points
                    else:
                        section_ctrl_pts = construct_control_points(
                            result.x, sweep_config,
                            [list(sec) for sec in w._airfoil.sections],
                            optimize_le_position=opt_config.optimize_le_position,
                        )

                    fit_result = FitResult(
                        section_control_points=section_ctrl_pts,
                        degree=sweep_config.degree,
                        final_error=result.fun,
                        num_iterations=result.nit,
                        num_evaluations=get_evaluation_count(),
                        elapsed_seconds=fit_elapsed,
                        converged=result.success and not (w._cancel_requested or early_stopped),
                        early_stopped=early_stopped,
                        cancelled=w._cancel_requested,
                        section_knots=result.section_knots,
                    )
                    self._attach_report(fit_result, sweep_opt_config)
                    self._sweep_results.append((val, fit_result))

                    # Build loss history
                    loss_history = [
                        IterationRecord(
                            iteration_number=it,
                            error_value=err,
                            elapsed_seconds=t,
                        )
                        for it, err, t in zip(
                            error_logger.iterations,
                            error_logger.errors,
                            error_logger.elapsed_times, strict=False,
                        )
                    ]

                    # Add to project
                    w._project_manager.add_optimization_run(
                        spline_config=sweep_config,
                        opt_config=sweep_opt_config,
                        fairness_weights=fairness_weights,
                        result=fit_result,
                        loss_history=loss_history,
                    )

            except Exception as e:
                print(f"Sweep fit failed for {param_name}={val}: {e}")

            # Update progress
            progress = (idx + 1) / total_fits * 100
            elapsed = time.perf_counter() - sweep_start_time
            val_str = f"{val:.5g}" if is_opt_config_sweep else str(val)
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
            # Update comparison plot
            w._plot_manager.set_comparison_label(param_name)
            x_data = [r[0] for r in self._sweep_results]
            y_data = [r[1].final_error for r in self._sweep_results]
            w._plot_manager.set_comparison_data(x_data, y_data)

            # Select best fit
            best_idx = min(
                range(len(self._sweep_results)),
                key=lambda i: self._sweep_results[i][1].final_error,
            )
            best_val = self._sweep_results[best_idx][0]
            best_result = self._sweep_results[best_idx][1]

            w._plot_manager.set_comparison_marker(
                [best_val], [best_result.final_error]
            )

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
                f"error \u02dc {format_error(best_result.final_error)}"
            )
        else:
            msg = "Sweep failed - no successful fits"

        w.statusBar().showMessage(msg)

        if not w._cancel_requested and self._sweep_results:
            w._play_completion_chime()

        w._set_fitting_ui(False)
