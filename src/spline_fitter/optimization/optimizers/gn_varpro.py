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

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import bspline_varpro
import numpy as np

from .base import AlgebraicOptimizer, RawOptimResult

if TYPE_CHECKING:
    from ..models import AirfoilData, OptimizationConfig, SplineConfig


# Tikhonov regularization of the inner solve, for numerical stability
REGULARIZATION = 1e-12
# Solver ftol/xtol/gtol of fixed-count fits (the library's default, which the
# refits of a tolerance search use). At 1e-10 a fit with a flat valley, such
# as four sections at the fewest control points, never ended
LM_TOLERANCE = 1e-8
# Bending energy weight in tolerance mode (1e-7 gave the fewest control points
# and curvature extrema in the bspline-varpro benchmark)
TOLERANCE_SMOOTHING = 1e-7


def continuity_arg(spline_config: SplineConfig) -> str | list[str]:
    """Continuity argument for the library: a list with one spec per junction,
    or a single spec when there is at most one junction (a single section
    has none, and the library rejects a list whose length does not match)."""
    specs = spline_config.junction_specs()
    if len(specs) > 1:
        return specs
    return specs[0] if specs else "G1"


def _flat(control_points: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(c, dtype=np.float64).ravel() for c in control_points])


class GNVarProOptimizer(AlgebraicOptimizer):
    """
    Levenberg-Marquardt variable projection optimizer for B-spline fitting.

    Example:
        >>> result = GNVarProOptimizer().fit(spline_config, opt_config, airfoil)
    """

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
                tolerance mode; fixed mode reads ``smoothing_weight``)
            airfoil: Airfoil data to fit
            callback: Optional callback, always called as
                ``callback(x, error, preview=(control points per section,
                knots per section or None, degree), step=<bool>)`` with the
                flattened control points ``x``. Fixed mode calls it on every
                residual evaluation (``step=False``, error = RMS error).
                Tolerance mode calls it on every solver step inside a refit
                (``step=True``, error = RMS error) and after every refit
                (``step=False``, error = max deviation). Return True to stop.
            should_stop: Optional check for a cancel request; tolerance mode
                calls it on every residual evaluation, also inside refits.

        Returns:
            RawOptimResult with fitted control points, knots and error
        """
        sections = [np.array(sec, dtype=np.float64) for sec in airfoil.sections]
        if opt_config.tolerance is not None:
            return self._fit_tolerance(spline_config, opt_config, sections, callback, should_stop)
        return self._fit_fixed(spline_config, opt_config, sections, callback)

    def _fit_fixed(self, spline_config, opt_config, sections, callback) -> RawOptimResult:
        from ..tracking import increment_evaluation

        evaluations = 0

        def on_evaluation(control_points: list[np.ndarray], rms_error: float,
                          knots: list[np.ndarray] | None = None) -> bool:
            nonlocal evaluations
            evaluations += 1
            increment_evaluation()
            if callback is None:
                return False
            # Knots move with free placement and growth (library 2.2 passes them)
            return bool(callback(_flat(control_points), rms_error,
                                 preview=(control_points, knots, spline_config.degree),
                                 step=False))

        # No evaluation limit: the fit runs until the solver converges
        result = bspline_varpro.fit(
            sections,
            num_control_points=spline_config.num_control_points,
            degree=spline_config.degree,
            continuity=continuity_arg(spline_config),
            fix_junctions=not opt_config.optimize_le_position,
            smoothing=opt_config.smoothing_weight,
            regularization=REGULARIZATION,
            initial_parameters=opt_config.custom_parameters,
            knot_placement=opt_config.knot_placement,
            grow=opt_config.grow_control_points,
            min_knot_gap=opt_config.min_knot_gap,
            tolerance=LM_TOLERANCE,
            exact_jacobian=True,
            verbose=0,
            callback=on_evaluation,
        )
        # Every evaluation of the fit: with growth result.nfev counts only
        # the last refit
        return self._raw(result, result.rms_error, max(evaluations, result.nfev))

    def _fit_tolerance(self, spline_config, opt_config, sections, callback,
                       should_stop=None) -> RawOptimResult:
        from ..tracking import get_evaluation_count, increment_evaluation

        refits = 0
        # Every fit the search went through: (result, deviation, evaluations, seconds)
        fits: list[tuple] = []
        start = time.perf_counter()

        def on_step(control_points: list[np.ndarray], rms_error: float,
                    knots: list[np.ndarray]) -> bool:
            # Evaluations are counted here only, with or without a callback
            increment_evaluation()
            if callback is None:
                return False
            return bool(callback(_flat(control_points), rms_error,
                                 preview=(control_points, knots, spline_config.degree),
                                 step=True))

        def on_refit(result: bspline_varpro.FitResult, deviation: float) -> bool:
            nonlocal refits
            refits += 1
            fits.append((result, deviation, get_evaluation_count(), time.perf_counter() - start))
            if callback is None:
                return False
            return bool(callback(_flat(result.control_points), deviation,
                                 preview=(result.control_points, result.knots, result.degree),
                                 step=False))

        result = bspline_varpro.fit_to_tolerance(
            sections,
            opt_config.tolerance,
            spline_config.degree,
            continuity=continuity_arg(spline_config),
            fix_junctions=not opt_config.optimize_le_position,
            smoothing=TOLERANCE_SMOOTHING,
            regularization=REGULARIZATION,
            initial_parameters=opt_config.custom_parameters,
            knot_placement=opt_config.knot_placement,
            initial_control_points=(None if opt_config.grow_control_points
                                    else spline_config.num_control_points),
            callback=on_refit,
            **({"should_stop": should_stop} if should_stop is not None else {}),
            step_callback=on_step,
            min_knot_gap=opt_config.min_knot_gap,
        )
        raw = self._raw(result, result.max_deviation, max(refits, 1))
        # The search is a sequence of fits like a sweep of the control point
        # count: the ones before the last are handed on too
        for number, (fit, deviation, evaluations, seconds) in enumerate(fits, 1):
            if fit is not result:
                earlier = self._raw(fit, deviation, number)
                earlier.success = deviation <= opt_config.tolerance
                earlier.evaluations, earlier.seconds = evaluations, seconds
                raw.fits.append(earlier)
        return raw

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
            rms_error=float(result.rms_error),
        )
