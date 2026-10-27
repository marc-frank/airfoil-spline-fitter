"""
Spline Package
==============

B-spline mathematics and curve operations.

This package provides:
    - spline: Core B-spline curve construction and evaluation
    - fairness: Smoothness metrics and regularization terms
"""

from .spline import (
    BSplineCurve2D,
    create_open_uniform_knots,
    validate_knot_vector,
    compute_signed_curvature,
    compute_curvature_derivative,
    compute_arc_length,
)

from .fairness import (
    compute_polygon_metrics,
    compute_curve_energy,
    compute_fairness_penalty,
)

__all__ = [
    # Spline math
    "BSplineCurve2D",
    "create_open_uniform_knots",
    "validate_knot_vector",
    "compute_signed_curvature",
    "compute_curvature_derivative",
    "compute_arc_length",
    # Fairness
    "compute_polygon_metrics",
    "compute_curve_energy",
    "compute_fairness_penalty",
]
