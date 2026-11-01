"""
Optimizers Package
==================

Individual optimizer implementations for spline fitting.

This package provides:
    - Base classes for optimizer interface
    - VarPro LM (bspline-varpro)
    - Algebraic methods (Direct LSQ)

Usage:
    >>> from spline_fitter.optimization.optimizers import get_optimizer, OptimizerType
    >>> optimizer = get_optimizer(OptimizerType.DIRECT_LSQ)
    >>> result = optimizer.fit(config, opt_config, airfoil)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    AlgebraicOptimizer,
    BaseOptimizer,
    RawOptimResult,
)
from .constrained_lsq import ConstrainedLSQOptimizer
from .gn_varpro import GNVarProOptimizer

if TYPE_CHECKING:
    from ..models import OptimizerType


# Module-level state for iteration tracking
_evaluation_count: int = 0
_iteration_count: int = 0


def reset_iteration_tracking() -> None:
    """Reset the global evaluation counter and iteration counter."""
    global _evaluation_count, _iteration_count
    _evaluation_count = 0
    _iteration_count = 0


def get_evaluation_count() -> int:
    """Return the current evaluation count."""
    return _evaluation_count


def increment_evaluation() -> int:
    """Increment and return the evaluation counter."""
    global _evaluation_count
    _evaluation_count += 1
    return _evaluation_count


def increment_iteration() -> int:
    """Increment and return the iteration counter."""
    global _iteration_count
    _iteration_count += 1
    return _iteration_count


def get_optimizer(optimizer_type: OptimizerType) -> BaseOptimizer:
    """
    Get an optimizer instance for the given type.
    
    Args:
        optimizer_type: The type of optimizer to create
        
    Returns:
        An optimizer instance
        
    Raises:
        ValueError: If optimizer type is not recognized
    """
    from ..models import OptimizerType

    optimizer_map = {
        OptimizerType.DIRECT_LSQ: ConstrainedLSQOptimizer,
        OptimizerType.GN_VARPRO: GNVarProOptimizer,
    }

    optimizer_class = optimizer_map.get(optimizer_type)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer_class()


__all__ = [
    # Base classes
    "BaseOptimizer",
    "AlgebraicOptimizer",
    "RawOptimResult",
    # Optimizers
    "ConstrainedLSQOptimizer",
    "GNVarProOptimizer",
    # Factory function
    "get_optimizer",
    # Iteration tracking
    "reset_iteration_tracking",
    "get_evaluation_count",
    "increment_evaluation",
    "increment_iteration",
]
