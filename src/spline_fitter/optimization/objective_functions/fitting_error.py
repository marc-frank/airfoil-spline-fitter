"""
Fitting Error Objective Function
================================

Computes the spline fitting objective function that combines
data fitting error with optional fairness regularization.

The data error is measured over the whole chain (``chain.py``): every data
point once, compared with whichever section it lies on or is closest to.

This module orchestrates the three error metrics:
    - PARAMETRIC: Fast, corresponding-point distance (parametric.py)
    - ORTHOGONAL: Accurate, closest-point distance (orthogonal.py)
    - ORTHOGONAL_LOCAL: Balanced, local closest-point search (orthogonal.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..models import (
        FairnessWeights,
        OptimizationConfig,
        PointList,
        SplineConfig,
    )

from ...spline.fairness import compute_fairness_penalty
from ...spline.spline import BSplineCurve2D
from ..models import ErrorMetric
from ..tracking import get_evaluation_count, increment_evaluation
from ..tracking import reset_evaluation_tracking as reset_evaluation_count
from .chain import ChainPoints
from .orthogonal import compute_orthogonal_error
from .parametric import compute_parametric_error

__all__ = [
    "compute_data_error",
    "compute_fitting_error",
    "get_evaluation_count",
    "increment_evaluation",
    "reset_evaluation_count",
]


def compute_data_error(
    curves: list[BSplineCurve2D],
    chain: ChainPoints,
    metric: ErrorMetric,
) -> float:
    """Average distance of the data points to the chain under ``metric``."""
    if len(chain.data) == 0:
        return 1e6
    if metric in (ErrorMetric.ORTHOGONAL, ErrorMetric.ORTHOGONAL_LOCAL):
        error_sum, n_pts = compute_orthogonal_error(
            curves, chain, local_search=metric == ErrorMetric.ORTHOGONAL_LOCAL)
    else:
        error_sum, n_pts = compute_parametric_error(curves, chain)
    return error_sum / n_pts


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
    construct_control_points_fn,
    optimize_le_position: bool = False,
    penalty_scales: FairnessWeights | None = None,
) -> float:
    """
    Compute the spline fitting objective function for n sections.

    This is the generalized objective function for n curve sections.
    It combines:
        1. Data fitting error (distance from data points to the chain)
        2. Fairness regularization (optional smoothness penalties)

    Args:
        variables: Current optimization variable vector
        config: Spline configuration
        opt_config: Optimization configuration (metric selection)
        sections: List of n section point lists (the split and junction points)
        chain: Data points of the whole chain with their chain parameters
        fairness_weights: Weights for regularization terms
        construct_control_points_fn: Function to build control points from variables
        optimize_le_position: If True, junction positions are in variables
        penalty_scales: Multiplier per penalty term (compute_penalty_scales)

    Returns:
        Total objective function value
    """
    increment_evaluation()

    try:
        # Build curves from current variables
        section_ctrl_pts = construct_control_points_fn(
            variables, config, sections,
            optimize_le_position=optimize_le_position,
        )

        # Create BSpline curves for each section
        curves = [BSplineCurve2D(config.degree, ctrl_pts) for ctrl_pts in section_ctrl_pts]

        data_error = compute_data_error(curves, chain, opt_config.metric)

        # Compute fairness regularization
        regularization = 0.0
        if fairness_weights.is_enabled:
            try:
                regularization = compute_fairness_penalty(
                    section_ctrl_pts, curves, fairness_weights, penalty_scales
                )
            except Exception:
                regularization = 0.0

        return float(data_error) + float(regularization)

    except Exception:
        return 1e6
