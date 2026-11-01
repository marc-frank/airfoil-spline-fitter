"""
Parametric Distance Objective
=============================

Fast error metric that compares the chain to the data points at the points'
fixed chain parameters (arc length within each starting section).

Each data point is compared with the chain point at its own parameter, so a
point keeps its place along the chain: the fastest metric, but tied to the
parameterization.

Pros:
    - Very fast (direct evaluation, no search)
    - Smooth objective function (good for gradient-based optimizers)

Cons:
    - Parameter correspondence may not reflect geometric distance
    - Points cannot slide along the chain or across a junction
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from bspline_varpro.chain import locate

from ...spline import BSplineCurve2D
from .chain import ChainPoints


def compute_parametric_error(
    curves: Sequence[BSplineCurve2D],
    chain: ChainPoints,
) -> tuple[float, int]:
    """
    Compute parametric (corresponding-point) distance error.

    Evaluates each point's section at the point's local parameter and sums the
    Euclidean distances to the data points.

    Args:
        curves: One curve per section
        chain: Data points with their chain parameters

    Returns:
        Tuple of (total_error_sum, num_points) for averaging
    """
    n_pts = len(chain.data)
    section, local = locate(chain.parameters, len(curves))
    err_sum = 0.0
    for s, curve in enumerate(curves):
        rows = section == s
        if not np.any(rows):
            continue
        sx, sy = curve.evaluate_array(local[rows])
        err_sum += float(np.sum(np.hypot(sx - chain.data[rows, 0], sy - chain.data[rows, 1])))

    if np.isfinite(err_sum):
        return (err_sum, n_pts)
    return (1e6, n_pts)
