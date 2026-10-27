"""
Parametric Distance Objective
=============================

Fast error metric that compares spline curve points to data points
at corresponding arc-length parameter values.

This is the simplest and fastest error metric - it samples the curve
at the same parameter values used to parameterize the original data
points, then computes the Euclidean distance between corresponding
point pairs.

Pros:
    - Very fast (direct evaluation, no search)
    - Smooth objective function (good for gradient-based optimizers)

Cons:
    - Parameter correspondence may not reflect geometric distance
    - Can miss local shape deviations between sample points
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from ...spline import BSplineCurve2D

# Type aliases
PointList = List[Tuple[float, float]]


def compute_parametric_error(
    curve: BSplineCurve2D,
    section_points: np.ndarray,
    t_params: np.ndarray,
) -> Tuple[float, int]:
    """
    Compute parametric (corresponding-point) distance error.
    
    Evaluates the curve at the given parameter values and computes
    the sum of Euclidean distances to the corresponding data points.
    
    Args:
        curve: The B-spline curve to evaluate
        section_points: Data points as (N, 2) array
        t_params: Parameter values at which to evaluate curve
        
    Returns:
        Tuple of (total_error_sum, num_points) for averaging
    """
    n_pts = len(section_points)
    
    # Evaluate curve at all parameter values
    sx, sy = curve.evaluate_array(t_params)
    
    # Compute Euclidean distances
    err = np.hypot(sx - section_points[:, 0], sy - section_points[:, 1])
    err_sum = float(np.sum(err))
    
    if np.isfinite(err_sum):
        return (err_sum, n_pts)
    else:
        return (1e6, n_pts)
