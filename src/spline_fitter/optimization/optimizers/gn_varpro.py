"""
Variable Projection Optimizer
=============================

Adapter from the app's optimizer interface to the ``bspline-varpro`` library,
which fits the B-spline sections by Levenberg-Marquardt variable projection.

Two modes:
    - fixed: the configured number of control points per section on uniform knots
    - tolerance: the fewest control points (non-uniform knots)
      that keep every data point within the tolerance
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import bspline_varpro
import numpy as np

from .base import AlgebraicOptimizer, RawOptimResult

if TYPE_CHECKING:
    from ..models import AirfoilData, OptimizationConfig, SplineConfig


@dataclass
class GNVarProConfig:
    """
    Configuration for the variable projection optimizer.

    Attributes:
        max_iterations: Residual evaluation limit when the optimization config sets
            none: a safety stop; the solver ends on convergence long before
        regularization: Tikhonov regularization for numerical stability
        smoothing_weight: Weight for bending energy regularization
        use_exact_jacobian: If True, compute exact Jacobian; else use finite diff
        verbose: Verbosity level (0=silent, 1=summary, 2=per-iteration)
        max_nfev: Residual evaluation limit overriding both iteration settings
        tolerance: Solver ftol/xtol/gtol
        tolerance_smoothing: Bending energy weight in tolerance mode (1e-7 gave
            the fewest control points and curvature extrema in the
            bspline-varpro benchmark)
    """
    max_iterations: int = 10000
    regularization: float = 1e-12
    smoothing_weight: float = 1e-3
    use_exact_jacobian: bool = True
    verbose: int = 0
    max_nfev: int | None = None
    tolerance: float = 1e-10
    tolerance_smoothing: float = 1e-7


def _accepts(callback: Callable | None, keyword: str) -> bool:
    if callback is None:
        return False
    try:
        return keyword in inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return False


def accepts_preview(callback: Callable | None) -> bool:
    """True if ``callback`` takes a ``preview`` keyword with live curves."""
    return _accepts(callback, "preview")


_TOLERANCE_PARAMETERS = inspect.signature(bspline_varpro.fit_to_tolerance).parameters
# bspline-varpro 2.1 can check for a cancel request inside tolerance refits,
# 2.2 reports every solver step of them
_TOLERANCE_STOP = "should_stop" in _TOLERANCE_PARAMETERS
_TOLERANCE_STEPS = "step_callback" in _TOLERANCE_PARAMETERS


def _continuity(spline_config: SplineConfig) -> str | list[str]:
    """Continuity argument for the library: a list with one spec per junction,
    or a single spec when there is at most one junction (a single section
    has none, and the library rejects a list whose length does not match)."""
    specs = spline_config.junction_specs()
    if len(specs) > 1:
        return specs
    return specs[0] if specs else "G1"


def _flat(control_points: list[np.ndarray]) -> np.ndarray:
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
        spline_config: SplineConfig,
        opt_config: OptimizationConfig,
        airfoil: AirfoilData,
        callback: Callable[..., bool] | None = None,
        should_stop: Callable[[], bool] | None = None,
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
                ``(section_control_points, section_knots, degree)``; if it
                accepts a ``step`` keyword, tolerance mode also calls it on
                every residual evaluation inside the refits with
                ``step=True`` (error = RMS error). Return True to stop early.
            should_stop: Optional check for a cancel request; tolerance mode
                calls it on every residual evaluation, also inside refits,
                where ``callback`` is only reached between refits.

        Returns:
            RawOptimResult with fitted control points, knots and error
        """
        sections = [np.array(sec, dtype=np.float64) for sec in airfoil.sections]
        if opt_config.tolerance is not None:
            return self._fit_tolerance(spline_config, opt_config, sections, callback, should_stop)
        return self._fit_fixed(spline_config, opt_config, sections, callback)

    def _fit_fixed(self, spline_config, opt_config, sections, callback) -> RawOptimResult:
        from ..engine import increment_evaluation

        with_preview = accepts_preview(callback)
        evaluations = 0

        def on_evaluation(control_points: list[np.ndarray], rms_error: float,
                          knots: list[np.ndarray] | None = None) -> bool:
            nonlocal evaluations
            evaluations += 1
            increment_evaluation()
            if callback is None:
                return False
            if with_preview:
                # Knots move with free placement and growth (library 2.2 passes them)
                return bool(callback(_flat(control_points), rms_error,
                                     preview=(control_points, knots, spline_config.degree)))
            return bool(callback(_flat(control_points), rms_error))

        max_iters = (opt_config.max_iterations if opt_config.max_iterations > 0
                     else self.config.max_iterations)
        result = bspline_varpro.fit(
            sections,
            num_control_points=spline_config.num_control_points,
            degree=spline_config.degree,
            continuity=_continuity(spline_config),
            fix_junctions=not opt_config.optimize_le_position,
            smoothing=self.config.smoothing_weight,
            regularization=self.config.regularization,
            initial_parameters=opt_config.custom_parameters,
            max_nfev=self.config.max_nfev or max_iters,
            max_points_per_section=opt_config.max_points_per_section,
            knot_placement=opt_config.knot_placement,
            grow=opt_config.grow_control_points,
            tolerance=self.config.tolerance,
            exact_jacobian=self.config.use_exact_jacobian,
            verbose=self.config.verbose,
            callback=on_evaluation,
        )
        # Every evaluation of the fit: growing and moving knots and dense
        # refinement too, where result.nfev counts only the last stage
        return self._raw(result, result.rms_error, max(evaluations, result.nfev))

    def _fit_tolerance(self, spline_config, opt_config, sections, callback,
                       should_stop=None) -> RawOptimResult:
        from ..engine import increment_evaluation

        with_preview = accepts_preview(callback)
        with_steps = _TOLERANCE_STEPS and _accepts(callback, "step")
        refits = 0

        def on_step(control_points: list[np.ndarray], rms_error: float,
                    knots: list[np.ndarray]) -> bool:
            increment_evaluation()
            preview = (control_points, knots, spline_config.degree) if with_preview else None
            return bool(callback(_flat(control_points), rms_error, preview=preview, step=True))

        def on_refit(result: bspline_varpro.FitResult, deviation: float) -> bool:
            nonlocal refits
            refits += 1
            if not with_steps:
                increment_evaluation()
            if callback is None:
                return False
            if with_preview:
                return bool(callback(_flat(result.control_points), deviation,
                                     preview=(result.control_points, result.knots, result.degree)))
            return bool(callback(_flat(result.control_points), deviation))

        result = bspline_varpro.fit_to_tolerance(
            sections,
            opt_config.tolerance,
            spline_config.degree,
            continuity=_continuity(spline_config),
            fix_junctions=not opt_config.optimize_le_position,
            smoothing=self.config.tolerance_smoothing,
            regularization=self.config.regularization,
            initial_parameters=opt_config.custom_parameters,
            max_points_per_section=opt_config.max_points_per_section,
            knot_placement=opt_config.knot_placement,
            initial_control_points=(None if opt_config.grow_control_points
                                    else spline_config.num_control_points),
            callback=on_refit,
            **self._tolerance_nfev(opt_config),
            **({"should_stop": should_stop} if _TOLERANCE_STOP and should_stop else {}),
            **({"step_callback": on_step} if with_steps else {}),
        )
        return self._raw(result, result.max_deviation, max(refits, 1))

    def _tolerance_nfev(self, opt_config) -> dict:
        """Residual evaluation limit per refit: the configured one, else the
        library's (the fixed-count default suits a whole fit, not a refit)."""
        limit = self.config.max_nfev or (opt_config.max_iterations
                                         if opt_config.max_iterations > 0 else None)
        return {"max_nfev": limit} if limit else {}

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
