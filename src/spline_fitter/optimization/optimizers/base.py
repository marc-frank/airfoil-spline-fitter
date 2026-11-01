"""
Base Optimizer Interface
========================

Abstract base class and common utilities for all optimizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..models import (
        AirfoilData,
        OptimizationConfig,
        PointList,
        SplineConfig,
    )


class RawOptimResult:
    """
    Raw result from an optimizer before post-processing.
    
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


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    Each optimizer implementation must provide:
        - name: Name for display/logging
        - is_algebraic: Property indicating if non-iterative method
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the optimizer name for display/logging."""
        pass

    @property
    def is_algebraic(self) -> bool:
        """Return True if this is an algebraic (non-iterative) method."""
        return False


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
