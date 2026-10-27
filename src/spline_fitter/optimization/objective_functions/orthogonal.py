"""
Orthogonal Distance Objective
=============================

Accurate error metric that finds the closest point on the curve
to each data point, computing true geometric distance.

This metric searches the entire curve (or a local neighborhood) to
find the parameter value that minimizes distance to each data point,
providing the most accurate measure of fitting quality.

Two variants:
    - ORTHOGONAL (full): Search entire curve [0, 1] for each point
    - ORTHOGONAL_LOCAL: Search local neighborhood around expected parameter

The computation is parallelized across CPU cores.

Pros:
    - Most accurate geometric error measure
    - Correctly captures shape deviations anywhere on curve

Cons:
    - Expensive (requires scalar optimization per point)
    - Non-smooth objective (can cause optimizer difficulties)
"""

from __future__ import annotations

import math

import numpy as np

from ...spline import BSplineCurve2D
from ..parallel import (
    OrthogonalDistanceArgs,
    orthogonal_distance_worker,
    run_parallel,
)

# Type aliases
PointList = list[tuple[float, float]]


def _compute_local_bounds(
    i: int,
    section_points: np.ndarray,
) -> tuple[float, float]:
    """
    Compute local search bounds for point i.
    
    For local search, restricts to neighborhood between adjacent
    parameter values with padding to handle noise displacement.
    
    Args:
        i: Index of current point
        section_points: Data points as (N, 2) array
        
    Returns:
        (t_lo, t_hi) search bounds
    """
    n_points = len(section_points)
    if n_points < 2:
        return (0.0, 1.0)

    # Compute cumulative chord-length parameters from data points
    deltas = np.diff(section_points, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    total_length = float(np.sum(seg_lengths))
    if total_length <= 0.0:
        return (0.0, 1.0)

    cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))

    if i == 0:
        # First point: search from curve start to next point's param
        t_lo = 0.0
        t_hi = float(cumulative[1] / total_length) if n_points > 1 else 1.0
    elif i == n_points - 1:
        # Last point: search from previous point's param to curve end
        t_lo = float(cumulative[-2] / total_length)
        t_hi = 1.0
    else:
        # Middle points: search between adjacent params
        t_lo = float(cumulative[i - 1] / total_length)
        t_hi = float(cumulative[i + 1] / total_length)

    # Ensure proper ordering
    if t_hi < t_lo:
        t_lo, t_hi = t_hi, t_lo

    # Keep bounds strictly between adjacent data points (no fixed padding)
    return (max(0.0, t_lo), min(1.0, t_hi))


def _prepare_tasks(
    curve: BSplineCurve2D,
    section_points: np.ndarray,
    t_params: np.ndarray | None,
    local_search: bool,
    tolerance: float = 1e-4,
) -> list[OrthogonalDistanceArgs]:
    """
    Prepare worker task arguments for orthogonal distance computation.
    
    Args:
        curve: The B-spline curve
        section_points: Data points as (N, 2) array
        t_params: Parameter values for local search (optional)
        local_search: If True, use local search bounds
        tolerance: Minimizer tolerance
        
    Returns:
        List of task argument tuples for parallel execution
    """
    knots = curve.knot_vector
    cx = np.array([p[0] for p in curve.control_points])
    cy = np.array([p[1] for p in curve.control_points])
    degree = curve.degree

    n_points = len(section_points)
    tasks = []

    for i in range(n_points):
        px = float(section_points[i, 0])
        py = float(section_points[i, 1])

        if local_search and t_params is not None and len(t_params) >= n_points:
            bounds = _compute_local_bounds(i, section_points)
        else:
            bounds = (0.0, 1.0)

        tasks.append((knots, cx, cy, degree, px, py, bounds, tolerance))

    return tasks


def compute_orthogonal_error(
    curve: BSplineCurve2D,
    section_points: np.ndarray,
    t_params: np.ndarray | None = None,
    local_search: bool = False,
) -> tuple[float, int]:
    """
    Compute orthogonal (closest-point) distance error.
    
    Finds the closest point on the curve to each data point and
    computes the sum of these minimum distances.
    
    Uses multiprocessing to parallelize the expensive closest-point
    searches across available CPU cores.
    
    Args:
        curve: The B-spline curve to evaluate
        section_points: Data points as (N, 2) array
        t_params: Parameter values for local search bounds (optional)
        local_search: If True, restrict search to neighborhood of t_params
        
    Returns:
        Tuple of (total_error_sum, num_points) for averaging
    """
    n_pts = len(section_points)

    # Prepare task arguments
    tasks = _prepare_tasks(curve, section_points, t_params, local_search)

    # Execute in parallel
    dists = run_parallel(orthogonal_distance_worker, tasks)

    # Validate results
    if not dists or any(not math.isfinite(d) for d in dists):
        return (1e6, n_pts)

    return (sum(dists), n_pts)


def compute_orthogonal_error_full(
    curve: BSplineCurve2D,
    section_points: np.ndarray,
) -> tuple[float, int]:
    """
    Compute orthogonal error searching the entire curve.
    
    Convenience wrapper for compute_orthogonal_error with full curve search.
    
    Args:
        curve: The B-spline curve to evaluate
        section_points: Data points as (N, 2) array
        
    Returns:
        Tuple of (total_error_sum, num_points) for averaging
    """
    return compute_orthogonal_error(
        curve,
        section_points,
        t_params=None,
        local_search=False,
    )


def compute_orthogonal_error_local(
    curve: BSplineCurve2D,
    section_points: np.ndarray,
    t_params: np.ndarray,
) -> tuple[float, int]:
    """
    Compute orthogonal error searching local parameter neighborhoods.
    
    Convenience wrapper for compute_orthogonal_error with local search.
    This restricts the search to the neighborhood between adjacent
    parameter values, providing a balance of accuracy and speed.
    
    Args:
        curve: The B-spline curve to evaluate
        section_points: Data points as (N, 2) array
        t_params: Parameter values defining local search neighborhoods
        
    Returns:
        Tuple of (total_error_sum, num_points) for averaging
    """
    return compute_orthogonal_error(
        curve,
        section_points,
        t_params=t_params,
        local_search=True,
    )
