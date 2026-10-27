"""
Local Optimizers
================

Local optimization methods using scipy.optimize.minimize.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from ..models import Bounds
from .base import BaseOptimizer, EarlyStopException, RawOptimResult


class LocalOptimizer(BaseOptimizer):
    """
    Local optimizer using scipy.optimize.minimize.
    
    Supports multiple methods via the scipy.optimize.minimize interface:
        - BFGS: Best speed/accuracy tradeoff (recommended)
        - L-BFGS-B: Fastest, lower accuracy
        - CG: Conjugate gradient
        - SLSQP: Highest accuracy, slower
        - TNC: Truncated Newton
        - Powell: Direction-set method (high timeout rate)
        - Nelder-Mead: Simplex method (poor accuracy)
        - COBYLA: Constrained optimization (poor accuracy)
    
    Attributes:
        method: The scipy.optimize.minimize method name
    """

    # Method name mapping from OptimizerType values
    METHOD_MAP = {
        "nelder-mead": "Nelder-Mead",
        "powell": "Powell",
        "cg": "CG",
        "bfgs": "BFGS",
        "l-bfgs-b": "L-BFGS-B",
        "tnc": "TNC",
        "cobyla": "COBYLA",
        "slsqp": "SLSQP",
    }

    def __init__(self, method: str = "BFGS"):
        """
        Initialize local optimizer with specified method.
        
        Args:
            method: scipy.optimize.minimize method name or OptimizerType value
        """
        # Handle both scipy method names and our internal names
        self._method = self.METHOD_MAP.get(method.lower(), method)

    @property
    def name(self) -> str:
        return self._method

    @property
    def supports_bounds(self) -> bool:
        """Some methods don't support bounds."""
        return self._method in ("L-BFGS-B", "TNC", "SLSQP", "COBYLA")

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
        Run local optimization.
        
        Args:
            objective: Objective function to minimize
            x0: Initial guess
            bounds: Variable bounds (may be ignored by some methods)
            max_iterations: Maximum number of iterations
            callback: Optional callback(x, error) returning True to stop
            seed: Random seed (not used by most local methods)
            
        Returns:
            RawOptimResult with best solution found
        """
        last_x = x0.copy()
        iteration_count = 0

        def local_callback(xk):
            nonlocal last_x, iteration_count
            last_x = np.asarray(xk, dtype=np.float64)
            iteration_count += 1
            if callback is None:
                return
            try:
                fval = objective(last_x)
                if callback(last_x, fval):
                    raise EarlyStopException("Early stop requested")
            except EarlyStopException:
                raise
            except Exception:
                pass

        try:
            options = {"maxiter": max_iterations, "disp": False}
            if self._method == "SLSQP":
                options["ftol"] = 0

            # Only pass bounds if method supports them
            min_kwargs = {
                "fun": objective,
                "x0": x0,
                "method": self._method,
                "callback": local_callback,
                "tol": 0,
                "options": options,
            }

            if self.supports_bounds:
                min_kwargs["bounds"] = bounds

            result = minimize(**min_kwargs)

            return RawOptimResult(
                x=np.asarray(result.x, dtype=np.float64),
                fun=float(result.fun) if np.isfinite(result.fun) else objective(result.x),
                nit=int(getattr(result, "nit", 0)),
            )

        except EarlyStopException:
            # Return best known solution
            try:
                f_best = objective(last_x)
            except Exception:
                f_best = float('inf')

            return RawOptimResult(x=last_x, fun=f_best, nit=iteration_count)


# Convenience classes for specific methods

class BFGSOptimizer(LocalOptimizer):
    """
    BFGS optimizer (recommended).
    
    Best speed/accuracy tradeoff for spline fitting.
    
    Benchmark (41 airfoils):
        - ~9s average
        - 7.8×10⁻⁵ error
        - 100% reliable
    """

    def __init__(self):
        super().__init__("BFGS")


class LBFGSBOptimizer(LocalOptimizer):
    """
    Limited-memory BFGS with bounds.
    
    Fastest optimizer, lower accuracy than BFGS.
    
    Benchmark (41 airfoils):
        - ~3s average
        - 1.7×10⁻⁴ error
        - 100% reliable
    """

    def __init__(self):
        super().__init__("L-BFGS-B")


class CGOptimizer(LocalOptimizer):
    """
    Conjugate Gradient optimizer.
    
    Moderate performance, no clear advantage over BFGS.
    
    Benchmark (41 airfoils):
        - ~10s average
        - 1.8×10⁻⁴ error
        - 100% reliable
    """

    def __init__(self):
        super().__init__("CG")


class SLSQPOptimizer(LocalOptimizer):
    """
    Sequential Least Squares Programming.
    
    Highest accuracy but significantly slower.
    
    Benchmark (41 airfoils):
        - ~220s average (26x slower than BFGS)
        - 7.0×10⁻⁵ error (best accuracy)
        - 100% reliable
    """

    def __init__(self):
        super().__init__("SLSQP")


class TNCOptimizer(LocalOptimizer):
    """
    Truncated Newton method.
    
    Not benchmarked. Use BFGS instead.
    """

    def __init__(self):
        super().__init__("TNC")


class PowellOptimizer(LocalOptimizer):
    """
    Powell's direction-set method.
    
    Not recommended - high timeout rate.
    
    Benchmark (41 airfoils):
        - 68% timeout rate (>5 min)
    """

    def __init__(self):
        super().__init__("Powell")


class NelderMeadOptimizer(LocalOptimizer):
    """
    Nelder-Mead simplex method.
    
    Not recommended - poor accuracy.
    
    Benchmark (41 airfoils):
        - 74× worse accuracy than SLSQP
    """

    def __init__(self):
        super().__init__("Nelder-Mead")


class COBYLAOptimizer(LocalOptimizer):
    """
    Constrained Optimization BY Linear Approximation.
    
    Not recommended - poor accuracy.
    
    Benchmark (41 airfoils):
        - 41× worse accuracy than SLSQP
    """

    def __init__(self):
        super().__init__("COBYLA")
