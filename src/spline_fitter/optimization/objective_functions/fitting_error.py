"""
Fitting Error Objective Function
================================

Computes the spline fitting objective function that combines
data fitting error with optional fairness regularization.

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
from .orthogonal import compute_orthogonal_error
from .parametric import compute_parametric_error

# =============================================================================
# Module State for Evaluation Tracking
# =============================================================================

_evaluation_count: int = 0


def reset_evaluation_count() -> None:
    """Reset the evaluation counter."""
    global _evaluation_count
    _evaluation_count = 0


def get_evaluation_count() -> int:
    """Return the current evaluation count."""
    return _evaluation_count


def increment_evaluation() -> int:
    """Increment and return the evaluation counter."""
    global _evaluation_count
    _evaluation_count += 1
    return _evaluation_count


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
    construct_control_points_fn,
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
        construct_control_points_fn: Function to build control points from variables
        optimize_le_position: If True, LE position is in variables
        
    Returns:
        Total objective function value
    """
    global _evaluation_count
    _evaluation_count += 1

    try:
        # Build curves from current variables
        section_ctrl_pts = construct_control_points_fn(
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
