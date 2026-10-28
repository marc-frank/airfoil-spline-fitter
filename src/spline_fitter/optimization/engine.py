"""
Optimization Engine
===================

Core optimization infrastructure for spline fitting.

This module provides:
    - Control point construction from optimization variables
    - Objective function for spline fitting
    - Unified optimizer interface supporting multiple algorithms
    - Early stopping and convergence detection

Control point construction, constrained least-squares fitting, evaluation
tracking, and run logging live in their own modules and are re-exported here.

The optimization engine handles the translation between the optimizer's
view (a flat vector of real-valued variables) and the spline's view
(structured control points with continuity constraints).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..geometry import compute_arc_length_parameters
from ..spline.fairness import compute_fairness_penalty
from ..spline.spline import BSplineCurve2D
from .control_points import construct_control_points, generate_initial_guess
from .models import (
    AirfoilData,
    Bounds,
    ErrorMetric,
    FairnessWeights,
    OptimizationConfig,
    OptimizerType,
    PointList,
    SplineConfig,
)
from .objective_functions.orthogonal import compute_orthogonal_error
from .objective_functions.parametric import compute_parametric_error
from .optimizers import (
    RawOptimResult,
    get_optimizer,
)
from .run_log import OptimizerErrorLogger  # noqa: F401

# Re-exported so existing ``from .engine import ...`` imports keep working
from .tracking import (  # noqa: F401
    get_evaluation_count,
    increment_evaluation,
    increment_iteration,
    reset_evaluation_tracking,
    reset_iteration_tracking,
)

# =============================================================================
# Objective Function
# =============================================================================

def compute_fitting_error(
    variables: np.ndarray,
    config: SplineConfig,
    opt_config: OptimizationConfig,
    sections: list[PointList],
    t_params: list[list[float]],
    fairness_weights: FairnessWeights,
    optimize_le_position: bool = False,
) -> float:
    """
    Compute the spline fitting objective function for n sections.
    
    This is the generalized objective function for n curve sections.
    It combines:
        1. Data fitting error (distance from data points to curves)
        2. Fairness regularization (optional smoothness penalties)
    
    Args:
        variables: Current optimization variable vector
        config: Spline configuration
        opt_config: Optimization configuration (metric selection)
        sections: List of n section point lists
        t_params: Parameter values for each section
        fairness_weights: Weights for regularization terms
        optimize_le_position: If True, LE position is in variables
        
    Returns:
        Total objective function value
    """
    increment_evaluation()

    try:
        # Build curves from current variables
        section_ctrl_pts = construct_control_points(
            variables, config, sections,
            optimize_le_position=optimize_le_position,
        )

        # Create BSpline curves for each section
        curves = [BSplineCurve2D(config.degree, ctrl_pts) for ctrl_pts in section_ctrl_pts]

        # Compute data fitting error for each section
        metric = opt_config.metric
        section_errors = []
        total_points = 0

        for section_idx, (section, t_sec, curve) in enumerate(zip(sections, t_params, curves, strict=False)):
            section_arr = np.array(section, dtype=np.float64)
            t_arr = np.array(t_sec, dtype=np.float64)
            n_pts = len(section_arr)
            total_points += n_pts

            if metric in (ErrorMetric.ORTHOGONAL, ErrorMetric.ORTHOGONAL_LOCAL):
                # Orthogonal (closest-point) distance
                local_search = (metric == ErrorMetric.ORTHOGONAL_LOCAL)
                error_sum, _ = compute_orthogonal_error(
                    curve, section_arr, t_arr if local_search else None, local_search
                )
                section_errors.append((error_sum, n_pts))
            else:
                # Parametric (corresponding-point) distance
                error_sum, _ = compute_parametric_error(curve, section_arr, t_arr)
                section_errors.append((error_sum, n_pts))

        # Compute weighted average error
        if total_points > 0:
            total_error_sum = sum(e[0] for e in section_errors)
            data_error = total_error_sum / total_points
        else:
            data_error = 1e6

        # Compute fairness regularization
        regularization = 0.0
        if fairness_weights.is_enabled:
            try:
                regularization = compute_fairness_penalty(
                    section_ctrl_pts, curves, fairness_weights
                )
            except Exception:
                regularization = 0.0

        result = float(data_error) + float(regularization)

        return result

    except Exception:
        return 1e6


# =============================================================================
# Optimizer Dispatch
# =============================================================================

def run_optimization(
    config: SplineConfig,
    opt_config: OptimizationConfig,
    airfoil: AirfoilData,
    fairness_weights: FairnessWeights,
    bounds: Sequence[Bounds],
    callback: Callable[[np.ndarray, float], bool] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> RawOptimResult:
    """
    Run the optimization algorithm to fit splines to airfoil data.
    
    This is the main entry point for the optimization engine. It:
        1. Prepares parameters based on configuration
        2. Dispatches to the appropriate optimizer
        3. Handles early stopping and callbacks
    
    Args:
        config: Spline configuration
        opt_config: Optimization configuration
        airfoil: Airfoil data to fit
        fairness_weights: Regularization weights
        bounds: Variable bounds
        callback: Optional callback(x, error) returning True to stop
        should_stop: Optional cancel check for optimizers that can stop
            between callbacks (VarPro LM tolerance mode)
        
    Returns:
        RawOptimResult with best solution found
    """
    optimizer_type = opt_config.optimizer

    # Get the optimizer instance
    optimizer = get_optimizer(optimizer_type)

    # -------------------------------------------------------------------------
    # LSQ-Based Methods
    # -------------------------------------------------------------------------

    if optimizer_type == OptimizerType.GN_VARPRO:
        optimizer.config.smoothing_weight = opt_config.smoothing_weight
        return optimizer.fit(
            config,
            opt_config,
            airfoil,
            callback=callback,
            should_stop=should_stop,
        )

    if optimizer_type == OptimizerType.DIRECT_LSQ:
        return optimizer.fit(config, opt_config, airfoil)

    # -------------------------------------------------------------------------
    # Iterative Optimizers (Global and Local)
    # -------------------------------------------------------------------------

    # Convert sections to lists and compute arc-length parameters
    sections = [list(sec) for sec in airfoil.sections]
    t_params = [compute_arc_length_parameters(sec) for sec in sections]

    optimize_le = opt_config.optimize_le_position

    def objective(x):
        return compute_fitting_error(
            x, config, opt_config,
            sections, t_params,
            fairness_weights,
            optimize_le_position=optimize_le,
        )

    # Generate initial guess for local optimizers
    x0 = generate_initial_guess(
        config, sections, t_params, bounds,
        optimize_le_position=optimize_le,
    )

    # Run the optimizer
    return optimizer.optimize(
        objective=objective,
        x0=x0,
        bounds=bounds,
        max_iterations=opt_config.max_iterations,
        callback=callback,
        seed=opt_config.random_seed,
    )
