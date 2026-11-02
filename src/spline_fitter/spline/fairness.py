"""
Fairness Regularization
=======================

Smoothness metrics and regularization terms for spline fitting.

Fairness regularization encourages control point distributions that
produce aesthetically pleasing, professionally smooth curves. This is
essential for CAD applications where visual quality matters as much
as numerical accuracy.

The regularization terms fall into two categories:

Control Polygon Metrics:
    - Length variance: Penalizes uneven control polygon edge lengths
    - Polygon bend: Penalizes sharp angle changes in the control polygon
    - Anti-bunching: Penalizes control points that are too close together
    - End handles: Penalizes unusual tangent vector lengths at endpoints

Curve Energy Terms:
    - Curvature squared: Bending energy (integral of κ²)
    - Curvature gradient: Wiggle energy (integral of (dκ/ds)²)

All terms are normalized to be scale-independent, so the same weights
work for airfoils of any chord length.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..optimization.models import Point2D
from .spline import BSplineCurve2D

# =============================================================================
# Control Polygon Metrics
# =============================================================================

def compute_polygon_metrics(control_points: Sequence[Point2D]) -> dict[str, float]:
    """
    Compute fairness metrics for a control polygon.
    
    These metrics measure properties of the control polygon itself,
    independent of the resulting curve shape.
    
    Args:
        control_points: Sequence of (x, y) control point coordinates
        
    Returns:
        Dictionary with metrics:
            - length_variance: Coefficient of variation squared
            - anti_bunching: Mean squared penalty for short edges
            - polygon_bend: Normalized second-difference energy
            - end_handles: Deviation of end edge lengths from mean
    """
    arr = np.asarray(control_points, dtype=np.float64)

    # Default return for degenerate cases
    defaults = {
        'length_variance': 0.0,
        'anti_bunching': 0.0,
        'polygon_bend': 0.0,
        'end_handles': 0.0,
    }

    if arr.shape[0] < 2:
        return defaults

    # Compute edge vectors and lengths
    edges = arr[1:] - arr[:-1]
    edge_lengths = np.hypot(edges[:, 0], edges[:, 1])

    # Filter out degenerate edges
    valid_mask = np.isfinite(edge_lengths)
    if not np.any(valid_mask):
        return defaults

    valid_lengths = edge_lengths[valid_mask]
    mean_length = float(np.mean(valid_lengths))

    if mean_length < 1e-12 or not np.isfinite(mean_length):
        mean_length = 1.0

    # Length variance: (std / mean)^2
    coeff_var = float(np.std(valid_lengths) / mean_length)
    length_variance = coeff_var * coeff_var

    # Anti-bunching: penalize edges shorter than half the mean
    normalized_lengths = valid_lengths / mean_length
    shortness_penalty = np.maximum(0.0, 0.5 - normalized_lengths) ** 2
    anti_bunching = float(np.mean(shortness_penalty))

    # Polygon bend: second-difference energy (measures polygon smoothness)
    polygon_bend = 0.0
    if arr.shape[0] >= 3:
        second_diff = arr[:-2] - 2.0 * arr[1:-1] + arr[2:]
        magnitude_sq = np.sum(second_diff * second_diff, axis=1)
        valid_mag = magnitude_sq[np.isfinite(magnitude_sq)]
        if valid_mag.size > 0:
            polygon_bend = float(np.mean(valid_mag) / (mean_length * mean_length))

    # End handles: deviation of first/last edges from mean
    end_handles = 0.0
    if valid_lengths.size >= 1:
        front_ratio = valid_lengths[0] / mean_length
        back_ratio = valid_lengths[-1] / mean_length
        end_handles = 0.5 * ((front_ratio - 1.0) ** 2 + (back_ratio - 1.0) ** 2)

    return {
        'length_variance': length_variance,
        'anti_bunching': anti_bunching,
        'polygon_bend': polygon_bend,
        'end_handles': end_handles,
    }


# =============================================================================
# Gauss-Legendre Quadrature Cache
# =============================================================================

_gauss_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _get_gauss_nodes_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Get Gauss-Legendre quadrature nodes and weights (cached).
    
    Args:
        order: Number of quadrature points
        
    Returns:
        (nodes, weights) arrays on interval [-1, 1]
    """
    order = max(2, int(order))

    if order in _gauss_cache:
        return _gauss_cache[order]

    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = nodes.astype(np.float64)
    weights = weights.astype(np.float64)

    _gauss_cache[order] = (nodes, weights)
    return nodes, weights


# =============================================================================
# Curve Energy Terms
# =============================================================================

def compute_curve_energy(
    curve: BSplineCurve2D | None,
    quadrature_order: int = 5,
) -> dict[str, float]:
    """
    Compute arc-length-normalized energy integrals over a curve.
    
    Uses Gauss-Legendre quadrature per knot span for accurate
    integration of the curvature-related quantities.
    
    Args:
        curve: The B-spline curve to analyze
        quadrature_order: Number of quadrature points per span
        
    Returns:
        Dictionary with:
            - length: Total arc length
            - curvature_squared: ∫κ² ds / L (normalized bending energy)
            - curvature_gradient: ∫(dκ/ds)² ds / L (normalized wiggle energy)
    """
    defaults = {'length': 0.0, 'curvature_squared': 0.0, 'curvature_gradient': 0.0}

    if curve is None:
        return defaults

    knots = np.asarray(curve.knot_vector, dtype=np.float64)

    if knots.size < 2 or not np.all(np.isfinite(knots)):
        return defaults

    nodes, weights = _get_gauss_nodes_weights(quadrature_order)
    degree = int(curve.degree)

    # Get spline objects for derivatives
    spline_x = curve._spline_x
    spline_y = curve._spline_y

    d1x_spline = spline_x.derivative(1)
    d1y_spline = spline_y.derivative(1)

    has_second_deriv = degree >= 2
    has_third_deriv = degree >= 3

    if has_second_deriv:
        d2x_spline = spline_x.derivative(2)
        d2y_spline = spline_y.derivative(2)

    if has_third_deriv:
        d3x_spline = spline_x.derivative(3)
        d3y_spline = spline_y.derivative(3)

    # Integrate over each knot span
    total_length = 0.0
    curvature_integral = 0.0
    gradient_integral = 0.0

    for a, b in zip(knots[:-1], knots[1:], strict=False):
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if b <= a + 1e-12:
            continue

        # Transform quadrature points to [a, b]
        half_span = 0.5 * (b - a)
        mid_point = 0.5 * (a + b)
        t_values = mid_point + half_span * nodes

        # Evaluate first derivatives (velocity)
        vx = d1x_spline(t_values)
        vy = d1y_spline(t_values)
        speed = np.hypot(vx, vy)
        speed = np.maximum(speed, 1e-12)  # Avoid division by zero

        # Weighted integration
        w = weights * half_span
        total_length += float(np.sum(w * speed))

        if not has_second_deriv:
            continue

        # Second derivatives (acceleration)
        ax = d2x_spline(t_values)
        ay = d2y_spline(t_values)

        # Curvature: κ = (v × a) / |v|³
        cross = vx * ay - vy * ax
        kappa = cross / (speed ** 3)

        # Curvature squared integral: ∫κ² |v| dt
        curvature_integral += float(np.sum(w * (kappa ** 2) * speed))

        # Third derivatives (jerk); zero for a quadratic, whose curvature
        # still changes along it
        if has_third_deriv:
            jx = d3x_spline(t_values)
            jy = d3y_spline(t_values)
        else:
            jx = jy = np.zeros_like(t_values)

        # Curvature derivative with respect to arc length
        cross_prime = vx * jy - vy * jx
        dot = vx * ax + vy * ay
        speed_sq = speed ** 2

        # dκ/ds = (cross' * |v|² - 3 * cross * dot) / |v|⁶
        numerator = cross_prime * speed_sq - 3.0 * cross * dot
        dk_ds = numerator / (speed ** 6)

        # Gradient squared integral: ∫(dκ/ds)² |v| dt
        gradient_integral += float(np.sum(w * (dk_ds ** 2) * speed))

    # Normalize by arc length
    if total_length <= 1e-12 or not math.isfinite(total_length):
        return defaults

    curvature_squared = curvature_integral / total_length if curvature_integral > 0 else 0.0
    curvature_gradient = gradient_integral / total_length if gradient_integral > 0 else 0.0

    return {
        'length': total_length,
        'curvature_squared': curvature_squared,
        'curvature_gradient': curvature_gradient,
    }

