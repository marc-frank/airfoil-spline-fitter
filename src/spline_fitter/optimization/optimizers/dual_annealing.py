"""
Dual Annealing Optimizer
========================

Global optimizer using simulated annealing variant.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import dual_annealing

from ..models import Bounds
from .base import BaseOptimizer, RawOptimResult


class DualAnnealingOptimizer(BaseOptimizer):
    """
    Dual Annealing global optimizer.
    
    Simulated annealing variant that combines classical simulated
    annealing with a local search strategy.
    
    Benchmark notes:
        - Very slow
        - Use only if local optimizers fail
    """

    @property
    def name(self) -> str:
        return "Dual Annealing"

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
        Run dual annealing optimization.
        
        Args:
            objective: Objective function to minimize
            x0: Initial guess (may be used as starting point)
            bounds: Variable bounds as sequence of (lower, upper) tuples
            max_iterations: Maximum number of iterations
            callback: Optional callback(x, f, context) returning True to stop
            seed: Random seed for reproducibility
            
        Returns:
            RawOptimResult with best solution found
        """
        def da_callback(x, f, context):
            if callback is None:
                return False
            try:
                return bool(callback(np.asarray(x), float(f)))
            except Exception:
                return False

        result = dual_annealing(
            objective,
            bounds=bounds,
            callback=da_callback,
            maxiter=max_iterations,
            seed=seed,
        )

        return RawOptimResult(
            x=np.asarray(result.x, dtype=np.float64),
            fun=float(result.fun) if np.isfinite(result.fun) else float('inf'),
            nit=int(result.nit),
        )
