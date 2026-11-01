"""
Spline Package
==============

B-spline mathematics and curve operations.

This package provides:
    - spline: Core B-spline curve construction and evaluation
    - fairness: Smoothness metrics and regularization terms
"""

from .fairness import (
    compute_curve_energy,
    compute_polygon_metrics,
)
from .spline import (
    BSplineCurve2D,
    compute_arc_length,
    compute_curvature_derivative,
    compute_signed_curvature,
    create_open_uniform_knots,
    validate_knot_vector,
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
]
