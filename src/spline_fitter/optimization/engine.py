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
from dataclasses import replace

import numpy as np

from ..geometry import compute_arc_length_parameters
from ..spline.fairness import compute_penalty_scales
from ..spline.spline import BSplineCurve2D
from .control_points import (
    construct_control_points,
    generate_initial_guess,
    initial_guess_from_fit,
)
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
from .objective_functions.chain import ChainPoints
from .objective_functions.fitting_error import compute_data_error
from .objective_functions.fitting_error import compute_fitting_error as _fitting_error
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
    chain: ChainPoints,
    fairness_weights: FairnessWeights,
    optimize_le_position: bool = False,
    penalty_scales: FairnessWeights | None = None,
) -> float:
    """
    Compute the spline fitting objective function for n sections.

    The data error is measured over the whole chain, every data point once
    (objective_functions.fitting_error), plus the optional fairness penalties.

    Args:
        variables: Current optimization variable vector
        config: Spline configuration
        opt_config: Optimization configuration (metric selection)
        sections: List of n section point lists (the split and junction points)
        chain: Data points of the whole chain with their chain parameters
        fairness_weights: Weights for regularization terms
        optimize_le_position: If True, junction positions are in variables
        penalty_scales: Multiplier per penalty term (compute_penalty_scales)

    Returns:
        Total objective function value
    """
    return _fitting_error(
        variables, config, opt_config, sections, chain, fairness_weights,
        construct_control_points, optimize_le_position=optimize_le_position,
        penalty_scales=penalty_scales,
    )


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

    # The chain is fitted as a whole: every data point once, with a chain
    # parameter from arc length within its starting section
    sections = [list(sec) for sec in airfoil.sections]
    t_params = [compute_arc_length_parameters(sec) for sec in sections]
    chain = ChainPoints.from_sections(sections, t_params)

    optimize_le = opt_config.optimize_le_position

    # Start from the plain least-squares fit: it already follows the shape and
    # meets the junction conditions. For the parametric metric it is solved on
    # the objective's own parameters, the best fit that metric can reach
    # without penalties; for the orthogonal metrics the points' parameters move
    # to their closest chain points between solves, so no point stays tied to
    # a section. It is also the yardstick for the fairness penalties, so their
    # weights mean the same for every shape, unit and degree. The polyline
    # guess stands in when it fails.
    penalty_scales = None
    try:
        reference_parameters = (chain.parameters if opt_config.metric == ErrorMetric.PARAMETRIC
                                else None)
        reference = get_optimizer(OptimizerType.DIRECT_LSQ).fit(
            config, replace(opt_config, custom_parameters=reference_parameters), airfoil)
        reference_pts = reference.section_control_points
        x0 = initial_guess_from_fit(config, reference_pts, bounds, optimize_le_position=optimize_le)
        if fairness_weights.is_enabled:
            reference_curves = [BSplineCurve2D(config.degree, pts) for pts in reference_pts]
            penalty_scales = compute_penalty_scales(
                reference_pts, reference_curves,
                compute_data_error(reference_curves, chain, opt_config.metric))
    except Exception:
        x0 = generate_initial_guess(
            config, sections, t_params, bounds,
            optimize_le_position=optimize_le,
        )

    # The best point evaluated: a stopped run reports its last iterate, and an
    # iterate can end above the starting point
    best = {"x": x0, "fun": float("inf")}

    def objective(x):
        value = compute_fitting_error(
            x, config, opt_config,
            sections, chain,
            fairness_weights,
            optimize_le_position=optimize_le,
            penalty_scales=penalty_scales,
        )
        if value < best["fun"]:
            best["x"], best["fun"] = np.array(x, dtype=np.float64), value
        return value

    # Every control point fixed (degree 1 with ends and junctions only): the
    # curve is already determined, and the optimizers reject an empty vector
    if len(x0) == 0:
        return RawOptimResult(x=x0, fun=objective(x0), nit=0)

    objective(x0)

    # Run the optimizer
    result = optimizer.optimize(
        objective=objective,
        x0=x0,
        bounds=bounds,
        max_iterations=opt_config.max_iterations,
        callback=callback,
        seed=opt_config.random_seed,
    )
    if not result.fun <= best["fun"]:
        result.x, result.fun = best["x"], best["fun"]
    return result
