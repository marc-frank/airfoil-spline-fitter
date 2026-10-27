"""
Differential Evolution Optimizer
================================

Global optimizer using differential evolution algorithm.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import differential_evolution

from ..models import Bounds
from .base import BaseOptimizer, RawOptimResult


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """
    Differential Evolution global optimizer.
    
    Explores the entire search space using evolutionary strategies.
    Very slow but robust for finding global optima.
    
    Benchmark notes:
        - Use only if local optimizers fail
        - Very slow compared to gradient-based methods
    """

    @property
    def name(self) -> str:
        return "Differential Evolution"

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
        Run differential evolution optimization.
        
        Args:
            objective: Objective function to minimize
            x0: Initial guess (ignored - DE explores full bounds)
            bounds: Variable bounds as sequence of (lower, upper) tuples
            max_iterations: Maximum number of generations
            callback: Optional callback(x, error) returning True to stop
            seed: Random seed for reproducibility
            
        Returns:
            RawOptimResult with best solution found
        """
        def de_callback(xk, convergence):
            if callback is None:
                return False
            try:
                err = objective(xk)
                return bool(callback(xk, err))
            except Exception:
                return False

        result = differential_evolution(
            objective,
            bounds=bounds,
            callback=de_callback,
            maxiter=max_iterations,
            seed=seed,
            workers=1,  # Parallelism handled in objective
            updating="deferred",
            polish=False,
            tol=0,
            disp=False,
        )

        return RawOptimResult(
            x=np.asarray(result.x, dtype=np.float64),
            fun=float(result.fun) if np.isfinite(result.fun) else float('inf'),
            nit=int(result.nit),
        )
