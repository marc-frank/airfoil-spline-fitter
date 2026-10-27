"""
Basin-Hopping Optimizer
=======================

Global optimizer using basin-hopping with random restarts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import basinhopping

from ..models import Bounds
from .base import BaseOptimizer, RawOptimResult


class BasinHoppingOptimizer(BaseOptimizer):
    """
    Basin-Hopping global optimizer.
    
    Combines a local optimizer with random perturbations to escape
    local minima. Uses SLSQP as the local minimizer.
    
    Benchmark notes:
        - Slow but more thorough than single local optimization
        - Use only if local optimizers fail
    """

    @property
    def name(self) -> str:
        return "Basin-Hopping"

    @property
    def is_global(self) -> bool:
        return True

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
        Run basin-hopping optimization.
        
        Args:
            objective: Objective function to minimize
            x0: Initial guess (if None, uses midpoint of bounds)
            bounds: Variable bounds as sequence of (lower, upper) tuples
            max_iterations: Maximum number of basin-hopping iterations
            callback: Optional callback(x, f, accept) returning True to stop
            seed: Random seed for reproducibility
            
        Returns:
            RawOptimResult with best solution found
        """
        def bh_callback(x, f, accept):
            if callback is None:
                return False
            try:
                return bool(callback(np.asarray(x), float(f)))
            except Exception:
                return False

        # Use provided x0 or start from midpoint of bounds
        if x0 is None or len(x0) == 0:
            x0 = np.array([(lo + hi) * 0.5 for lo, hi in bounds], dtype=np.float64)

        result = basinhopping(
            objective,
            x0=x0,
            niter=max_iterations,
            callback=bh_callback,
            seed=seed,
            minimizer_kwargs={"method": "SLSQP", "bounds": bounds},
            disp=False,
        )

        return RawOptimResult(
            x=np.asarray(result.x, dtype=np.float64),
            fun=float(result.fun) if np.isfinite(result.fun) else float('inf'),
            nit=int(getattr(result, "niter", 0)),
        )
