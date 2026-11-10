"""
Spline Package
==============

B-spline mathematics and curve operations.

This package provides:
    - spline: B-spline curve evaluation and curvature helpers

Knot vectors come from the bspline-varpro library (``open_uniform_knots``,
``validate_knots``).
"""

from .spline import (
    BSplineCurve2D,
    compute_arc_length,
    compute_curvature_derivative,
    compute_signed_curvature,
)

__all__ = [
    "BSplineCurve2D",
    "compute_signed_curvature",
    "compute_curvature_derivative",
    "compute_arc_length",
]
