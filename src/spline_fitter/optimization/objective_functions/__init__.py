"""
Objective Functions Package
===========================

Objective functions for spline fitting optimization.

Modules:
    fitting_error: Main orchestrator combining error metrics with fairness
    parametric: Fast corresponding-point distance metric
    orthogonal: Accurate closest-point distance metrics
    chain: The data points of the whole chain with their chain parameters
"""

from .chain import ChainPoints
from .fitting_error import (
    compute_data_error,
    compute_fitting_error,
    get_evaluation_count,
    increment_evaluation,
    reset_evaluation_count,
)
from .orthogonal import compute_orthogonal_error
from .parametric import compute_parametric_error

__all__ = [
    # Main objective function
    "ChainPoints",
    "compute_data_error",
    "compute_fitting_error",
    "reset_evaluation_count",
    "get_evaluation_count",
    "increment_evaluation",
    # Individual error metrics
    "compute_parametric_error",
    "compute_orthogonal_error",
]
