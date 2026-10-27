"""
Variable Projection Optimizer
=============================

Adapter from the app's optimizer interface to the ``bspline-varpro`` library,
which fits the B-spline sections by Levenberg-Marquardt variable projection.

Two modes:
    - fixed: the configured number of control points per section on uniform knots
    - tolerance: the fewest control points (non-uniform knots, optional fairing)
      that keep every data point within the tolerance
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, List, Optional, TYPE_CHECKING

import numpy as np
import bspline_varpro

from .base import AlgebraicOptimizer, RawOptimResult

if TYPE_CHECKING:
    from ..models import AirfoilData, OptimizationConfig, SplineConfig


@dataclass
class GNVarProConfig:
    """
    Configuration for the variable projection optimizer.

    Attributes:
        max_iterations: Residual evaluation limit when the optimization config sets none
        regularization: Tikhonov regularization for numerical stability
        smoothing_weight: Weight for bending energy regularization
        use_exact_jacobian: If True, compute exact Jacobian; else use finite diff
        verbose: Verbosity level (0=silent, 1=summary, 2=per-iteration)
        max_nfev: Residual evaluation limit overriding both iteration settings
        tolerance: Solver ftol/xtol/gtol
        tolerance_smoothing: Bending energy weight in tolerance mode (kept tiny;
            smoothness comes from the fairing stage)
        max_points_per_section: Subsample size for the variable projection
            stage of dense data in tolerance mode
    """
    max_iterations: int = 50
    regularization: float = 1e-12
    smoothing_weight: float = 1e-3
    use_exact_jacobian: bool = True
    verbose: int = 0
    max_nfev: Optional[int] = None
    tolerance: float = 1e-10
    tolerance_smoothing: float = 1e-5
    max_points_per_section: int = 200


def accepts_preview(callback: Optional[Callable]) -> bool:
    """True if ``callback`` takes a ``preview`` keyword with live curves."""
    if callback is None:
        return False
    try:
        return "preview" in inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return False


def _continuity(spline_config: "SplineConfig") -> List[str]:
    specs = spline_config.junction_specs()
    return specs if specs else ["G1"]


def _flat(control_points: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(c, dtype=np.float64).ravel() for c in control_points])


@dataclass
class GNVarProOptimizer(AlgebraicOptimizer):
    """
    Levenberg-Marquardt variable projection optimizer for B-spline fitting.

    Example:
        >>> optimizer = GNVarProOptimizer(config=GNVarProConfig(max_iterations=30))
        >>> result = optimizer.fit(spline_config, opt_config, airfoil)
    """

    config: GNVarProConfig = field(default_factory=GNVarProConfig)

    @property
    def name(self) -> str:
        return "VarPro LM"

    def fit(
        self,
        spline_config: "SplineConfig",
        opt_config: "OptimizationConfig",
        airfoil: "AirfoilData",
        callback: Optional[Callable[..., bool]] = None,
    ) -> RawOptimResult:
        """
        Fit splines using variable projection.

        Args:
            spline_config: Spline configuration
            opt_config: Optimization configuration (``tolerance`` selects
                tolerance mode)
            airfoil: Airfoil data to fit
            callback: Optional callback(x, error) with the flattened control
                points. Called on every residual evaluation (fixed mode) or
                after every refit (tolerance mode, error = max deviation).
                If it accepts a ``preview`` keyword, it also receives
                ``(section_control_points, section_knots, degree)``.
                Return True to stop early.

        Returns:
            RawOptimResult with fitted control points, knots and error
        """
        sections = [np.array(sec, dtype=np.float64) for sec in airfoil.sections]
        if opt_config.tolerance is not None:
            return self._fit_tolerance(spline_config, opt_config, sections, callback)
        return self._fit_fixed(spline_config, opt_config, sections, callback)

    def _fit_fixed(self, spline_config, opt_config, sections, callback) -> RawOptimResult:
        from ..engine import increment_evaluation

        with_preview = accepts_preview(callback)

        def on_evaluation(control_points: List[np.ndarray], rms_error: float) -> bool:
            increment_evaluation()
            if callback is None:
                return False
            if with_preview:
                return bool(callback(_flat(control_points), rms_error,
                                     preview=(control_points, None, spline_config.degree)))
            return bool(callback(_flat(control_points), rms_error))

        max_iters = (opt_config.max_iterations if opt_config.max_iterations > 0
                     else self.config.max_iterations)
        specs = _continuity(spline_config)
        continuity = specs if len(specs) > 1 or specs[0][0] == "G" else int(specs[0][1])
        result = bspline_varpro.fit(
            sections,
            num_control_points=spline_config.num_control_points,
            degree=spline_config.degree,
            continuity=continuity,
            fix_junctions=not opt_config.optimize_le_position,
            smoothing=self.config.smoothing_weight,
            regularization=self.config.regularization,
            initial_parameters=opt_config.custom_parameters,
            max_nfev=self.config.max_nfev or max_iters,
            tolerance=self.config.tolerance,
            exact_jacobian=self.config.use_exact_jacobian,
            verbose=self.config.verbose,
            callback=on_evaluation,
        )
        return self._raw(result, result.rms_error, result.nfev)

    def _fit_tolerance(self, spline_config, opt_config, sections, callback) -> RawOptimResult:
        from ..engine import increment_evaluation

        with_preview = accepts_preview(callback)
        refits = 0

        def on_refit(result: bspline_varpro.FitResult, deviation: float) -> bool:
            nonlocal refits
            refits += 1
            increment_evaluation()
            if callback is None:
                return False
            if with_preview:
                return bool(callback(_flat(result.control_points), deviation,
                                     preview=(result.control_points, result.knots, result.degree)))
            return bool(callback(_flat(result.control_points), deviation))

        specs = _continuity(spline_config)
        result = bspline_varpro.fit_to_tolerance(
            sections,
            opt_config.tolerance,
            spline_config.degree,
            continuity=specs if len(specs) > 1 else specs[0],
            fix_junctions=not opt_config.optimize_le_position,
            smoothing=self.config.tolerance_smoothing,
            regularization=self.config.regularization,
            initial_parameters=opt_config.custom_parameters,
            fairing=opt_config.fairing,
            max_points_per_section=self.config.max_points_per_section,
            callback=on_refit,
        )
        return self._raw(result, result.max_deviation, max(refits, 1))

    @staticmethod
    def _raw(result: bspline_varpro.FitResult, error: float, nit: int) -> RawOptimResult:
        return RawOptimResult(
            x=_flat(result.control_points),
            fun=float(error),
            nit=nit,
            section_control_points=[
                [(float(x), float(y)) for x, y in section] for section in result.control_points
            ],
            parameters=result.parameters,
            success=result.success,
            message=result.message,
            section_knots=[np.asarray(k, dtype=np.float64).tolist() for k in result.knots],
        )
