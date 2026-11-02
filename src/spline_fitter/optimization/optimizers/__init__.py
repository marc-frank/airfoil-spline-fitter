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

# The adapters count evaluations in ..tracking; one set of counters
from ..tracking import (
    get_evaluation_count,
    increment_evaluation,
    increment_iteration,
    reset_iteration_tracking,
)
from .base import (
    AlgebraicOptimizer,
    BaseOptimizer,
    RawOptimResult,
)
from .constrained_lsq import ConstrainedLSQOptimizer
from .gn_varpro import GNVarProOptimizer

if TYPE_CHECKING:
    from ..models import OptimizerType


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
