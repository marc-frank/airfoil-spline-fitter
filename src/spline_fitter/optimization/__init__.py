"""
Optimization Package
====================

Optimization infrastructure for spline fitting.

This package provides:
    - models: Data structures (AirfoilData, SplineConfig, etc.)
    - parallel: Multiprocessing utilities for orthogonal distance computation
    - objective_functions: Optimizer wrappers and objective functions
    - optimizers: Individual optimizer implementations
"""

from .models import (
    # Type aliases
    Point2D,
    PointList,
    Bounds,
    # Enumerations
    ContinuityMode,
    OptimizerType,
    ErrorMetric,
    # Data structures
    AirfoilData,
    SplineConfig,
    OptimizationConfig,
    FairnessWeights,
    FitResult,
    BoundingBox,
    IterationRecord,
    EvaluationRecord,
    OptimizationRun,
    ProjectFile,
    ProjectMetadata,
)

from .parallel import (
    get_pool,
    shutdown_pool,
    run_parallel,
    orthogonal_distance_worker,
    OrthogonalDistanceArgs,
)

from .engine import (
    run_optimization,
    OptimizerErrorLogger,
    construct_control_points,
    generate_initial_guess,
    compute_fitting_error,
    reset_iteration_tracking,
    get_evaluation_count,
    increment_iteration,
)

from .optimizers import (
    RawOptimResult,
    EarlyStopException,
    BaseOptimizer,
    AlgebraicOptimizer,
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
    # Parallel utilities
    "get_pool",
    "shutdown_pool",
    "run_parallel",
    "orthogonal_distance_worker",
    "OrthogonalDistanceArgs",
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
