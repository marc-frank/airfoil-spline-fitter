"""
Objective Functions Package
===========================

Objective functions for spline fitting optimization.

Modules:
    fitting_error: Main orchestrator combining error metrics with fairness
    parametric: Fast corresponding-point distance metric
    orthogonal: Accurate closest-point distance metrics (parallel)
"""

from .fitting_error import (
    compute_fitting_error,
    get_evaluation_count,
    increment_evaluation,
    reset_evaluation_count,
)
from .orthogonal import (
    compute_orthogonal_error,
    compute_orthogonal_error_full,
    compute_orthogonal_error_local,
)
from .parametric import compute_parametric_error

__all__ = [
    # Main objective function
    "compute_fitting_error",
    "reset_evaluation_count",
    "get_evaluation_count",
    "increment_evaluation",
    # Individual error metrics
    "compute_parametric_error",
    "compute_orthogonal_error",
    "compute_orthogonal_error_full",
    "compute_orthogonal_error_local",
]
