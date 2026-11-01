"""
Optimization Package
====================

Optimization infrastructure for spline fitting.

This package provides:
    - models: Data structures (AirfoilData, SplineConfig, etc.)
    - objective_functions: Optimizer wrappers and objective functions
    - optimizers: Individual optimizer implementations
"""

from .engine import (
    OptimizerErrorLogger,
    compute_fitting_error,
    construct_control_points,
    generate_initial_guess,
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
    ErrorMetric,
    EvaluationRecord,
    FairnessWeights,
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
    EarlyStopException,
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
    "ErrorMetric",
    # Data structures
    "AirfoilData",
    "SplineConfig",
    "OptimizationConfig",
    "FairnessWeights",
    "FitResult",
    "BoundingBox",
    "IterationRecord",
    "EvaluationRecord",
    "OptimizationRun",
    "ProjectFile",
    "ProjectMetadata",
    # Optimization functions
    "run_optimization",
    "OptimizerErrorLogger",
    "construct_control_points",
    "generate_initial_guess",
    "compute_fitting_error",
    "reset_iteration_tracking",
    "get_evaluation_count",
    "increment_iteration",
    # Optimizer classes
    "RawOptimResult",
    "EarlyStopException",
    "BaseOptimizer",
    "AlgebraicOptimizer",
    "get_optimizer",
]
