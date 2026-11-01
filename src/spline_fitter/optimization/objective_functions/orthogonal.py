"""
Orthogonal Distance Objective
=============================

Accurate error metric: the distance from each data point to the closest
point of the chain, on whichever section it lies.

Two variants:
    - ORTHOGONAL (full): Search the whole chain for each point
    - ORTHOGONAL_LOCAL: Search the chain between the neighbouring points'
      chain parameters, across a junction where they span one

The closest points are found on a dense sample of each knot span, refined by
Newton's method (``bspline_varpro.diagnostics.project``), vectorized over all
points.

Pros:
    - Most accurate geometric error measure
    - Points are not tied to a section: the chain is fitted as a whole

Cons:
    - Slower than the parametric metric
    - The full search can pick a distant part of the chain that passes close
      (the other side of a thin trailing edge); the local search cannot
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from bspline_varpro.chain import evaluate_chain, project_between
from bspline_varpro.diagnostics import distance_to_curves

from ...spline import BSplineCurve2D
from .chain import ChainPoints, scipy_splines

# Samples per knot span of the full search's grid, before Newton refinement
SAMPLES_PER_SPAN = 8


def compute_orthogonal_error(
    curves: Sequence[BSplineCurve2D],
    chain: ChainPoints,
    local_search: bool = False,
) -> tuple[float, int]:
    """
    Compute orthogonal (closest-point) distance error over the whole chain.

    Args:
        curves: One curve per section
        chain: Data points with their chain parameters (the local search
            window of a point runs between its neighbours' parameters)
        local_search: If True, search only between the neighbours' parameters

    Returns:
        Tuple of (total_error_sum, num_points) for averaging
    """
    n_pts = len(chain.data)
    splines = scipy_splines(curves)
    u = chain.parameters
    if local_search:
        lower = np.concatenate([[0.0], u[:-1]])
        upper = np.concatenate([u[1:], [float(len(curves))]])
        foot = project_between(chain.data, splines, lower, upper, start=u)
        dists = np.linalg.norm(evaluate_chain(splines, foot) - chain.data, axis=1)
    else:
        # The chain point at a point's own parameter bounds its closest one:
        # a sampled search can miss the closest point of a curve that
        # oscillates between the data points
        own = np.linalg.norm(evaluate_chain(splines, u) - chain.data, axis=1)
        dists = np.minimum(distance_to_curves(chain.data, splines, SAMPLES_PER_SPAN), own)

    err_sum = float(np.sum(dists))
    if not np.isfinite(err_sum):
        return (1e6, n_pts)
    return (err_sum, n_pts)
