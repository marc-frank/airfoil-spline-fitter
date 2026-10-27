"""
Orthogonal LSQ Optimizer
========================

Alternating least-squares for orthogonal distance minimization.

Algorithm (Point Distance Minimization):
    1. Initialize t-values with arc-length parameterization
    2. Solve constrained LSQ for control points (linear)
    3. Find closest curve point for each data point → new t-values
    4. Repeat 2-3 until convergence

This converges in 3-10 iterations typically, giving orthogonal
distance accuracy with LSQ speed per iteration.
"""

from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING

import numpy as np
from scipy.interpolate import BSpline as ScipyBSpline

from .base import AlgebraicOptimizer, RawOptimResult
from .constrained_lsq import ConstrainedLSQOptimizer
from ..footpoint import iterative_footpoint_projection

if TYPE_CHECKING:
    from ..models import AirfoilData, OptimizationConfig, PointList, SplineConfig


class OrthogonalLSQOptimizer(AlgebraicOptimizer):
    """
    Alternating LSQ optimizer for orthogonal distance fitting.
    
    Iterates between:
    - LSQ solve for control points (fast, linear)
    - Vectorized Newton-Raphson foot-point projection to update t-values
    
    Uses the new footpoint.py module for fast, vectorized foot-point
    computation instead of expensive per-point minimize_scalar calls.
    
    Early stopping is handled by the GUI callback.
    """
    
    def __init__(self, max_outer_iterations: int = 20):
        self._max_outer_iterations = max_outer_iterations
        self._lsq = ConstrainedLSQOptimizer()
    
    @property
    def name(self) -> str:
        return "Orthogonal LSQ"
    
    def fit(
        self,
        config: "SplineConfig",
        opt_config: "OptimizationConfig",
        airfoil: "AirfoilData",
        callback: Optional[Callable[[np.ndarray, float], bool]] = None,
    ) -> RawOptimResult:
        """Fit using alternating LSQ + vectorized foot-point projection.
        
        Args:
            config: Spline configuration
            opt_config: Optimization configuration  
            airfoil: Airfoil data to fit
            callback: Optional callback(x, error) called each iteration.
                      Return True to stop early.
        """
        from ..models import OptimizationConfig, OptimizerType
        from ...geometry import compute_arc_length_parameters
        from ...spline.spline import create_open_uniform_knots
        
        sections = [list(sec) for sec in airfoil.sections]
        n_ctrl = config.num_control_points
        degree = config.degree
        knots = create_open_uniform_knots(degree, n_ctrl)
        
        # Initialize t-values with arc-length parameterization
        t_params = [
            np.array(compute_arc_length_parameters(sec), dtype=np.float64)
            for sec in sections
        ]
        
        best_result = None
        best_error = float('inf')
        best_t_params = None
        
        # Use opt_config.max_iterations, fall back to default
        max_iters = opt_config.max_iterations if opt_config.max_iterations > 0 else self._max_outer_iterations
        
        # Lazy import to avoid circular dependency
        from ..engine import increment_evaluation
        
        for iteration in range(max_iters):
            # Track evaluation count for GUI display
            increment_evaluation()
            
            # Stage 1: LSQ solve with current t-values
            inner_config = OptimizationConfig(
                optimizer=OptimizerType.CONSTRAINED_LSQ,
                metric=opt_config.metric,
                max_iterations=1,
                optimize_le_position=opt_config.optimize_le_position,
                custom_parameters=[t.tolist() for t in t_params],
            )
            
            result = self._lsq.fit(config, inner_config, airfoil)
            
            if result.section_control_points is None:
                break
            
            # Stage 2: Update t-values via vectorized foot-point projection
            new_t_params = []
            total_error = 0.0
            total_points = 0
            
            for sec_idx, ctrl_pts in enumerate(result.section_control_points):
                cx = np.array([p[0] for p in ctrl_pts])
                cy = np.array([p[1] for p in ctrl_pts])
                sec_pts = np.array(sections[sec_idx], dtype=np.float64)
                sec_t = t_params[sec_idx]
                
                # Create SciPy BSpline objects for this section
                spline_x = ScipyBSpline(knots, cx, degree, extrapolate=False)
                spline_y = ScipyBSpline(knots, cy, degree, extrapolate=False)
                
                # Vectorized Newton-Raphson foot-point projection
                # max_iter=3 is typically sufficient for convergence
                new_t, dists = iterative_footpoint_projection(
                    t_init=sec_t,
                    data_points=sec_pts,
                    spline_x=spline_x,
                    spline_y=spline_y,
                    max_iter=3,
                    tol=1e-8,
                    use_second_derivative=True,
                )
                
                new_t_params.append(new_t)
                total_error += np.sum(dists)
                total_points += len(dists)
            
            avg_error = total_error / total_points if total_points > 0 else float('inf')
            
            # Track best result
            if avg_error < best_error:
                best_error = avg_error
                best_result = result
                best_t_params = [t.copy() for t in t_params]
            
            # Call callback with current control points and error
            if callback is not None and result.x is not None:
                try:
                    if callback(result.x, avg_error):
                        # Early stop requested by callback (GUI handles early stopping)
                        break
                except Exception:
                    pass
            
            t_params = new_t_params
        
        if best_result is None:
            # Fallback to plain LSQ
            return self._lsq.fit(config, opt_config, airfoil)
        
        return RawOptimResult(
            x=best_result.x,
            fun=best_error,
            nit=iteration + 1,
            section_control_points=best_result.section_control_points,
        )
