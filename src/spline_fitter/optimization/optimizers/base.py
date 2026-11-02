"""
Base Optimizer Interface
========================

Abstract base class and common utilities for all optimizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
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
        fun: Error of the fit: the RMS distance of the data points to their
            curve points, or the maximum orthogonal deviation in VarPro LM
            tolerance mode (whose goal it is)
        nit: Steps performed: residual evaluations of the whole fit (VarPro
            LM fixed count), refits (tolerance mode), or 1 (Direct LSQ)
        section_control_points: Direct control points (for algebraic methods)
        section_knots: Knot vector per section (None = open uniform)
        rms_error: RMS distance of the data points to their curve points,
            whatever ``fun`` is
    """

    __slots__ = ("x", "fun", "nit", "section_control_points", "parameters", "success",
                 "message", "section_knots", "rms_error")

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
        rms_error: float | None = None,
    ) -> None:
        self.x = x
        self.fun = fun
        self.nit = nit
        self.section_control_points = section_control_points
        self.parameters = parameters
        self.success = success
        self.message = message
        self.section_knots = section_knots
        self.rms_error = fun if rms_error is None else rms_error


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
        callback: Callable[..., bool] | None = None,
    ) -> RawOptimResult:
        """
        Fit splines using an algebraic method.
        
        Args:
            config: Spline configuration
            opt_config: Optimization configuration
            airfoil: Airfoil data to fit
            callback: Optional progress callback(x, error)
            
        Returns:
            RawOptimResult with computed control points
        """
        pass
