"""
Optimization Package
====================

Optimization infrastructure for spline fitting.

This package provides:
    - models: Data structures (AirfoilData, SplineConfig, etc.)
    - optimizers: VarPro LM and Direct LSQ
"""

from .engine import (
    OptimizerErrorLogger,
    get_evaluation_count,
    increment_iteration,
    reset_iteration_tracking,
    run_optimization,
)
from .models import (
    # Data structures
    AirfoilData,
    BoundingBox,
    Bounds,
    # Enumerations
    ContinuityMode,
    FitResult,
    IterationRecord,
    OptimizationConfig,
    OptimizationRun,
    OptimizerType,
    # Type aliases
    Point2D,
    PointList,
    ProjectFile,
    ProjectMetadata,
    SplineConfig,
)
from .optimizers import (
    AlgebraicOptimizer,
    BaseOptimizer,
    RawOptimResult,
    get_optimizer,
)

__all__ = [
    # Type aliases
    "Point2D",
    "PointList",
    "Bounds",
    # Enumerations
    "ContinuityMode",
    "OptimizerType",
    # Data structures
    "AirfoilData",
    "SplineConfig",
    "OptimizationConfig",
    "FitResult",
    "BoundingBox",
    "IterationRecord",
    "OptimizationRun",
    "ProjectFile",
    "ProjectMetadata",
    # Optimization functions
    "run_optimization",
    "OptimizerErrorLogger",
    "reset_iteration_tracking",
    "get_evaluation_count",
    "increment_iteration",
    # Optimizer classes
    "RawOptimResult",
    "BaseOptimizer",
    "AlgebraicOptimizer",
    "get_optimizer",
]
