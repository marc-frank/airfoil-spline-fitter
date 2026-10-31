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
from dataclasses import fields

import numpy as np

from ..optimization.models import FairnessWeights, Point2D, PointList
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

    try:
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

    except Exception:
        return defaults


# =============================================================================
# Combined Regularizer
# =============================================================================

def compute_fairness_penalty(
    section_control_points: Sequence[PointList],
    section_curves: Sequence[BSplineCurve2D | None],
    weights: FairnessWeights,
    scales: FairnessWeights | None = None,
) -> float:
    """
    Compute the total fairness regularization penalty for n sections.
    
    Generalizes compute_fairness_penalty for arbitrary number of curve sections.
    Computes metrics for each section and averages them.
    
    Args:
        section_control_points: List of control point lists for each section
        section_curves: List of B-spline curves (for energy terms)
        weights: Weights for each regularization term
        scales: Multiplier per term (see compute_penalty_scales); 1 when None
        
    Returns:
        Total weighted regularization penalty (averaged over sections)
    """
    if not weights.is_enabled:
        return 0.0

    num_sections = len(section_control_points)
    if num_sections == 0:
        return 0.0

    # Compute metrics for each section
    polygon_metrics_list = [compute_polygon_metrics(ctrl_pts) for ctrl_pts in section_control_points]
    energy_metrics_list = [compute_curve_energy(curve) for curve in section_curves]

    # Compute scale factor for normalizing energy terms
    all_points = []
    for ctrl_pts in section_control_points:
        all_points.extend(ctrl_pts)

    scale = 1.0

    # Use arc length average
    lengths = [e.get('length', 0.0) for e in energy_metrics_list]
    valid_lengths = [L for L in lengths if L > 1e-12 and math.isfinite(L)]
    if valid_lengths:
        scale = max(scale, float(np.mean(valid_lengths)))

    # Also consider bounding box
    if all_points:
        arr = np.asarray(all_points, dtype=np.float64)
        if arr.size >= 2 and np.all(np.isfinite(arr)):
            x_span = float(np.max(arr[:, 0]) - np.min(arr[:, 0]))
            y_span = float(np.max(arr[:, 1]) - np.min(arr[:, 1]))
            scale = max(scale, x_span, y_span)

    scale = max(scale, 1e-6)

    # Average polygon metrics across all sections
    length_variance = float(np.mean([m['length_variance'] for m in polygon_metrics_list]))
    polygon_bend = float(np.mean([m['polygon_bend'] for m in polygon_metrics_list]))
    anti_bunching = float(np.mean([m['anti_bunching'] for m in polygon_metrics_list]))
    end_handles = float(np.mean([m['end_handles'] for m in polygon_metrics_list]))

    # Average energy terms (scaled for dimensionlessness)
    curvature_sq_values = [e.get('curvature_squared', 0.0) for e in energy_metrics_list]
    curvature_grad_values = [e.get('curvature_gradient', 0.0) for e in energy_metrics_list]

    curvature_sq = float(np.mean(curvature_sq_values)) * (scale ** 2)
    curvature_grad = float(np.mean(curvature_grad_values)) * (scale ** 4)

    # Sanitize values
    terms = [length_variance, polygon_bend, anti_bunching,
             end_handles, curvature_sq, curvature_grad]
    for i, val in enumerate(terms):
        if not math.isfinite(val) or val < 0.0:
            terms[i] = 0.0

    length_variance, polygon_bend, anti_bunching, end_handles, curvature_sq, curvature_grad = terms

    # Compute weighted sum
    values = {
        'length_variance': length_variance,
        'polygon_bend': polygon_bend,
        'anti_bunching': anti_bunching,
        'end_handles': end_handles,
        'curvature_squared': curvature_sq,
        'curvature_gradient': curvature_grad,
    }
    penalty = 0.0
    for name, value in values.items():
        scale = 1.0 if scales is None else getattr(scales, name)
        penalty += getattr(weights, name) * value * scale

    return float(penalty)


def compute_penalty_scales(
    section_control_points: Sequence[PointList],
    section_curves: Sequence[BSplineCurve2D | None],
    reference_error: float,
) -> FairnessWeights:
    """
    Multipliers that put every penalty term on the scale of the fit.
    
    Unscaled, the terms are dimensionless while the fitting error is a length,
    and bending and wiggle grow with the curvature of the shape (on an
    airfoil's leading edge, wiggle is many decades above the fitting error),
    so a weight would mean something different for every shape, unit and
    degree. Scaled, each term at weight 1 equals ``reference_error`` on the
    given reference curve.
    
    Args:
        section_control_points: Control points of the reference curve
        section_curves: B-spline curves of the reference curve
        reference_error: Fitting error of the reference curve
        
    Returns:
        Multiplier per term for compute_fairness_penalty. A term that is zero
        on the reference curve (bunching on an even polygon) counts as 1,
        the natural size of a dimensionless term.
    """
    if not math.isfinite(reference_error):
        return FairnessWeights(**{f.name: 1.0 for f in fields(FairnessWeights)})
    scales = {}
    for f in fields(FairnessWeights):
        term = compute_fairness_penalty(
            section_control_points, section_curves, FairnessWeights(**{f.name: 1.0}))
        usable = term > 0 and math.isfinite(term)
        scales[f.name] = reference_error / (term if usable else 1.0)
    return FairnessWeights(**scales)
