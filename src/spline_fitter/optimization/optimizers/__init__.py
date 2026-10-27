"""
Optimizers Package
==================

Individual optimizer implementations for spline fitting.

This package provides:
    - Base classes for optimizer interface
    - Global optimizers (DE, DA, Basin-Hopping)
    - Local optimizers (BFGS, L-BFGS-B, CG, SLSQP, etc.)
    - Algebraic methods (Constrained LSQ)

Usage:
    >>> from spline_fitter.optimization.optimizers import get_optimizer, OptimizerType
    >>> optimizer = get_optimizer(OptimizerType.BFGS)
    >>> result = optimizer.optimize(objective, x0, bounds, max_iter)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    BaseOptimizer,
    AlgebraicOptimizer,
    RawOptimResult,
    EarlyStopException,
)
from .differential_evolution import DifferentialEvolutionOptimizer
from .dual_annealing import DualAnnealingOptimizer
from .basin_hopping import BasinHoppingOptimizer
from .local import (
    LocalOptimizer,
    BFGSOptimizer,
    LBFGSBOptimizer,
    CGOptimizer,
    SLSQPOptimizer,
    TNCOptimizer,
    PowellOptimizer,
    NelderMeadOptimizer,
    COBYLAOptimizer,
)
from .constrained_lsq import ConstrainedLSQOptimizer, Parameterization
from .orthogonal_lsq import OrthogonalLSQOptimizer
from .param_optimizer import ParamOptimizer
from .gn_varpro import GNVarProOptimizer

if TYPE_CHECKING:
    from ..models import OptimizerType


# Module-level state for iteration tracking (used by local optimizers)
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


def get_optimizer(optimizer_type: "OptimizerType") -> BaseOptimizer:
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
    from .constrained_lsq import Parameterization
    
    # DIRECT_LSQ is ConstrainedLSQ with centripetal parameterization
    if optimizer_type == OptimizerType.DIRECT_LSQ:
        return ConstrainedLSQOptimizer(parameterization=Parameterization.CENTRIPETAL)
    
    optimizer_map = {
        # Global optimizers
        OptimizerType.DE: DifferentialEvolutionOptimizer,
        OptimizerType.DA: DualAnnealingOptimizer,
        OptimizerType.BH: BasinHoppingOptimizer,
        # Local optimizers
        OptimizerType.BFGS: BFGSOptimizer,
        OptimizerType.L_BFGS_B: LBFGSBOptimizer,
        OptimizerType.CG: CGOptimizer,
        OptimizerType.SLSQP: SLSQPOptimizer,
        OptimizerType.TNC: TNCOptimizer,
        OptimizerType.POWELL: PowellOptimizer,
        OptimizerType.NELDER_MEAD: NelderMeadOptimizer,
        OptimizerType.COBYLA: COBYLAOptimizer,
        # Algebraic
        OptimizerType.CONSTRAINED_LSQ: ConstrainedLSQOptimizer,
        OptimizerType.ORTHOGONAL_LSQ: OrthogonalLSQOptimizer,
        OptimizerType.GN_VARPRO: GNVarProOptimizer,
        # Two-stage parameterization
        OptimizerType.PARAM_OPT: ParamOptimizer,
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
    "EarlyStopException",
    # Global optimizers
    "DifferentialEvolutionOptimizer",
    "DualAnnealingOptimizer",
    "BasinHoppingOptimizer",
    # Local optimizers
    "LocalOptimizer",
    "BFGSOptimizer",
    "LBFGSBOptimizer",
    "CGOptimizer",
    "SLSQPOptimizer",
    "TNCOptimizer",
    "PowellOptimizer",
    "NelderMeadOptimizer",
    "COBYLAOptimizer",
    # Algebraic
    "ConstrainedLSQOptimizer",
    "Parameterization",
    "OrthogonalLSQOptimizer",
    "GNVarProOptimizer",
    # Two-stage parameterization
    "ParamOptimizer",
    # Factory function
    "get_optimizer",
    # Iteration tracking
    "reset_iteration_tracking",
    "get_evaluation_count",
    "increment_evaluation",
    "increment_iteration",
]
