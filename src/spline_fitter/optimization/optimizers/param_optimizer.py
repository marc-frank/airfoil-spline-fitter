"""
Parameterization Optimizer
==========================

Two-stage optimization: outer optimizer finds optimal parameterization (t values),
inner constrained LSQ fits control points algebraically.

Optimizes for minimum bending energy while maintaining data fidelity.
For orthogonal distance fitting, use OrthogonalLSQOptimizer instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .base import BaseOptimizer, EarlyStopException, RawOptimResult
from .constrained_lsq import ConstrainedLSQOptimizer

if TYPE_CHECKING:
    from ..models import (
        AirfoilData,
        Bounds,
        FairnessWeights,
        OptimizationConfig,
        OptimizerType,
        PointList,
        SplineConfig,
    )


class ParamOptimizer(BaseOptimizer):
    """
    Two-stage parameterization optimizer for minimum bending energy.
    
    Stage 1 (outer): Optimize parameter (t) values for each data point
    Stage 2 (inner): Constrained LSQ fits control points given t values
    
    The objective minimizes bending energy (curvature-based fairness)
    while maintaining curve fidelity to the data points.
    
    Variables are t values for each data point, bounded to [0, 1].
    """

    def __init__(
        self,
        outer_optimizer_type: OptimizerType | None = None,
        fairness_weights: FairnessWeights | None = None,
    ):
        """
        Initialize the parameterization optimizer.
        
        Args:
            outer_optimizer_type: Optimizer for the outer loop (default: L-BFGS-B)
            fairness_weights: Weights for bending energy computation
        """
        self._outer_optimizer_type = outer_optimizer_type
        self._fairness_weights = fairness_weights

        # Will be set during fit()
        self._inner_optimizer = ConstrainedLSQOptimizer()
        self._config: SplineConfig | None = None
        self._opt_config: OptimizationConfig | None = None
        self._airfoil: AirfoilData | None = None
        self._sections: list[list] | None = None
        self._section_sizes: list[int] | None = None

    @property
    def name(self) -> str:
        return "Param Optimizer"

    @property
    def supports_bounds(self) -> bool:
        return True

    @property
    def is_global(self) -> bool:
        return False

    @property
    def is_algebraic(self) -> bool:
        # Not purely algebraic - uses iterative outer loop
        return False

    def fit(
        self,
        config: SplineConfig,
        opt_config: OptimizationConfig,
        airfoil: AirfoilData,
        outer_optimizer_type: OptimizerType | None = None,
        fairness_weights: FairnessWeights | None = None,
        callback: Callable[[np.ndarray, float], bool] | None = None,
    ) -> RawOptimResult:
        """
        Fit splines using two-stage parameterization optimization.
        
        Args:
            config: Spline configuration
            opt_config: Optimization configuration
            airfoil: Airfoil data with sections
            outer_optimizer_type: Optimizer for outer loop
            fairness_weights: Weights for bending energy
            callback: Optional callback(x, error) returning True to stop
            
        Returns:
            RawOptimResult with computed control points
        """
        from ...geometry import compute_arc_length_parameters
        from ..engine import increment_evaluation
        from ..models import OptimizationConfig, OptimizerType

        # Store state for objective function
        self._config = config
        self._airfoil = airfoil
        self._fairness_weights = fairness_weights

        # Use provided or stored outer optimizer type
        outer_type = outer_optimizer_type or self._outer_optimizer_type
        if outer_type is None:
            outer_type = OptimizerType.L_BFGS_B

        # Convert sections to lists
        self._sections = [list(sec) for sec in airfoil.sections]
        self._section_sizes = [len(sec) for sec in self._sections]
        total_points = sum(self._section_sizes)

        # Initialize t values with arc-length parameterization
        t_init = []
        t_init_per_section = []
        for section in self._sections:
            t_params = compute_arc_length_parameters(section)
            t_init.extend(t_params)
            t_init_per_section.append(np.array(t_params, dtype=np.float64))
        t_init = np.array(t_init, dtype=np.float64)

        # Create opt_config for inner LSQ (without custom_parameters initially)
        self._opt_config = OptimizationConfig(
            optimizer=OptimizerType.CONSTRAINED_LSQ,
            metric=opt_config.metric,
            max_iterations=1,
            optimize_le_position=opt_config.optimize_le_position,
            custom_parameters=None,  # Will be set per evaluation
        )

        # Define bounds for t values (full range)
        bounds = [(0.0, 1.0)] * total_points

        # Track best result
        best_result: list[RawOptimResult | None] = [None]
        best_error: list[float] = [float('inf')]
        best_t_flat: list[np.ndarray | None] = [None]
        last_result: list[RawOptimResult | None] = [None]
        last_error: list[float] = [float('inf')]
        last_t_flat: list[np.ndarray | None] = [None]
        iteration_count: list[int] = [0]

        def objective(t_flat: np.ndarray) -> float:
            """Objective function: fit with given t values, return bending energy."""
            # Increment global evaluation counter
            increment_evaluation()

            # Unpack t values per section
            custom_params = []
            offset = 0
            for size in self._section_sizes:
                section_t = t_flat[offset:offset + size].tolist()
                custom_params.append(section_t)
                offset += size

            # Create new opt_config with custom parameters
            inner_opt_config = OptimizationConfig(
                optimizer=OptimizerType.CONSTRAINED_LSQ,
                metric=self._opt_config.metric,
                max_iterations=1,
                optimize_le_position=self._opt_config.optimize_le_position,
                custom_parameters=custom_params,
            )

            # Run inner LSQ fit
            try:
                inner_result = self._inner_optimizer.fit(
                    self._config, inner_opt_config, self._airfoil
                )
            except Exception:
                return 1e10

            if inner_result.section_control_points is None:
                return 1e10

            # Compute bending energy objective
            error = self._compute_bending_objective(
                inner_result.section_control_points,
                custom_params,
            )

            # Track last result for iteration callback
            last_result[0] = inner_result
            last_error[0] = error
            last_t_flat[0] = t_flat.copy()

            # Track best
            if error < best_error[0]:
                best_error[0] = error
                best_result[0] = inner_result
                best_t_flat[0] = t_flat.copy()

            return error

        # Wrapper callback for outer optimizer - called once per iteration
        def iteration_wrapper(t_flat: np.ndarray, obj_value: float) -> bool:
            """Pass control points (not t-values) to user callback."""
            iteration_count[0] += 1
            if callback is None:
                return False
            if last_result[0] is None or last_result[0].x is None:
                return False
            try:
                if callback(last_result[0].x, last_error[0]):
                    raise EarlyStopException("Callback requested stop")
            except EarlyStopException:
                raise
            except Exception:
                pass
            return False

        # Get outer optimizer
        from . import get_optimizer
        outer_optimizer = get_optimizer(outer_type)

        # Run outer optimization
        try:
            outer_optimizer.optimize(
                objective=objective,
                x0=t_init,
                bounds=bounds,
                max_iterations=opt_config.max_iterations,
                callback=iteration_wrapper,  # Called once per iteration with control points
                seed=opt_config.random_seed,
            )
        except EarlyStopException:
            pass

        # Return best result found
        if best_result[0] is not None:
            result = best_result[0]
            return RawOptimResult(
                x=result.x,
                fun=best_error[0],
                nit=iteration_count[0],  # Use actual iteration count
                section_control_points=result.section_control_points,
            )

        # Fallback: run with initial arc-length params
        fallback_result = self._inner_optimizer.fit(
            self._config, self._opt_config, self._airfoil
        )
        return fallback_result

    def _compute_bending_objective(
        self,
        section_control_points: list[PointList],
        t_params: list[list[float]],
    ) -> float:
        """Compute bending energy + fitting error."""
        from ...spline.fairness import compute_fairness_penalty
        from ...spline.spline import BSplineCurve2D, create_open_uniform_knots
        from ..models import FairnessWeights

        # Build curves
        curves = []
        for ctrl_pts in section_control_points:
            knots = create_open_uniform_knots(self._config.degree, len(ctrl_pts))
            curve = BSplineCurve2D(self._config.degree, ctrl_pts, knots)
            curves.append(curve)

        # Compute bending energy
        weights = self._fairness_weights
        if weights is None:
            # Default weights emphasizing bending
            weights = FairnessWeights(
                curvature_squared=1.0,
                curvature_gradient=0.1,
            )

        bending_penalty = compute_fairness_penalty(
            section_control_points, curves, weights
        )

        # Also add small parametric fitting error to keep curve close to data
        fitting_error = 0.0
        total_points = 0

        for section_idx, curve in enumerate(curves):
            section_pts = np.array(self._sections[section_idx], dtype=np.float64)
            section_t = np.array(t_params[section_idx], dtype=np.float64)

            # Evaluate curve at t values
            for i, t in enumerate(section_t):
                pt = curve.evaluate(float(t))
                dx = pt[0] - section_pts[i, 0]
                dy = pt[1] - section_pts[i, 1]
                fitting_error += np.hypot(dx, dy)
                total_points += 1

        avg_fitting_error = fitting_error / total_points if total_points > 0 else 0.0

        # Combine: bending + fitting (weighted to keep both relevant)
        return bending_penalty + 10.0 * avg_fitting_error**2

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
        Not used directly - use fit() instead.
        """
        raise NotImplementedError(
            "ParamOptimizer should be invoked via fit() method, "
            "which handles two-stage optimization internally."
        )
