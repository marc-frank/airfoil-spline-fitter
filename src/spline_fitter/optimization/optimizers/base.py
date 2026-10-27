"""
Base Optimizer Interface
========================

Abstract base class and common utilities for all optimizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..models import (
        AirfoilData,
        Bounds,
        OptimizationConfig,
        PointList,
        SplineConfig,
    )


class RawOptimResult:
    """
    Raw result from an optimizer before post-processing.
    
    For iterative optimizers:
        - x: flat optimization variable vector
        - section_control_points: None (computed via construct_control_points)
        
    For algebraic methods:
        - x: flattened control points (for logging compatibility)
        - section_control_points: directly computed control points
    
    Attributes:
        x: Solution variable vector
        fun: Objective function value at solution
        nit: Number of iterations performed
        section_control_points: Direct control points (for algebraic methods)
        section_knots: Knot vector per section (None = open uniform)
    """

    __slots__ = ("x", "fun", "nit", "section_control_points", "parameters", "success",
                 "message", "section_knots")

    def __init__(
        self,
        x: np.ndarray,
        fun: float,
        nit: int,
        section_control_points: list[PointList] | None = None,
        parameters=None,
        success: bool = True,
        message: str = "",
        section_knots: list[list[float]] | None = None,
    ) -> None:
        self.x = x
        self.fun = fun
        self.nit = nit
        self.section_control_points = section_control_points
        self.parameters = parameters
        self.success = success
        self.message = message
        self.section_knots = section_knots


class EarlyStopException(Exception):
    """Raised to signal early termination of optimization."""
    pass


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    Each optimizer implementation must provide:
        - optimize(): Main optimization method
        - supports_bounds: Property indicating bound support
        - is_global: Property indicating if global optimizer
        - is_algebraic: Property indicating if non-iterative method
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the optimizer name for display/logging."""
        pass

    @property
    def supports_bounds(self) -> bool:
        """Return True if this optimizer supports variable bounds."""
        return True

    @property
    def is_global(self) -> bool:
        """Return True if this is a global optimizer."""
        return False

    @property
    def is_algebraic(self) -> bool:
        """Return True if this is an algebraic (non-iterative) method."""
        return False

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: Sequence[Bounds],
        max_iterations: int,
        callback: Callable[[np.ndarray, float], bool] | None = None,
        seed: int | None = None,
    ) -> RawOptimResult:
        """
        Run the optimization algorithm.
        
        Args:
            objective: Objective function to minimize
            x0: Initial guess (may be ignored by global optimizers)
            bounds: Variable bounds as sequence of (lower, upper) tuples
            max_iterations: Maximum number of iterations
            callback: Optional callback(x, error) returning True to stop
            seed: Random seed for reproducibility
            
        Returns:
            RawOptimResult with best solution found
        """
        pass


class AlgebraicOptimizer(BaseOptimizer):
    """
    Base class for algebraic (non-iterative) optimization methods.
    
    These methods solve for the solution directly rather than iterating.
    """

    @property
    def is_algebraic(self) -> bool:
        return True

    @abstractmethod
    def fit(
        self,
        config: SplineConfig,
        opt_config: OptimizationConfig,
        airfoil: AirfoilData,
    ) -> RawOptimResult:
        """
        Fit splines using an algebraic method.
        
        Args:
            config: Spline configuration
            opt_config: Optimization configuration
            airfoil: Airfoil data to fit
            
        Returns:
            RawOptimResult with computed control points
        """
        pass

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: Sequence[Bounds],
        max_iterations: int,
        callback: Callable[[np.ndarray, float], bool] | None = None,
        seed: int | None = None,
    ) -> RawOptimResult:
        """
        Not used for algebraic methods. Use fit() instead.
        """
        raise NotImplementedError(
            "Algebraic optimizers should use fit() instead of optimize()"
        )
