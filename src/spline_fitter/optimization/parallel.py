"""
Multiprocessing Utilities
=========================

Process pool management for parallelizing expensive computations.

This module provides generic parallel infrastructure - a managed process
pool that can execute worker functions across CPU cores.

The pool is lazily initialized and reused across operations.
Proper cleanup is registered via atexit to ensure clean shutdown.
"""

from __future__ import annotations

import atexit
import math
import multiprocessing as mp
import os
from collections.abc import Callable
from multiprocessing.pool import Pool
from typing import Any

import numpy as np
from scipy.interpolate import BSpline as ScipyBSpline
from scipy.optimize import minimize_scalar

# =============================================================================
# Module State
# =============================================================================

_pool: Pool | None = None
_pool_size: int | None = None


# =============================================================================
# Pool Management
# =============================================================================

def get_pool(num_processes: int | None = None) -> Pool:
    """
    Get or create the shared process pool.
    
    The pool is lazily created on first access and reused for all
    subsequent calls. Pool size can be controlled via the
    AIRFOIL_MP_PROCS environment variable.
    
    Args:
        num_processes: Number of worker processes. If None, uses
            AIRFOIL_MP_PROCS env var or defaults to CPU count.
            
    Returns:
        The shared multiprocessing Pool instance
    """
    global _pool, _pool_size

    if _pool is not None:
        return _pool

    # Determine pool size
    if num_processes is None:
        env_val = os.environ.get("AIRFOIL_MP_PROCS", "").strip()
        if env_val:
            try:
                num_processes = max(1, int(env_val))
            except ValueError:
                num_processes = None

    if num_processes is None:
        try:
            num_processes = max(1, mp.cpu_count() - 2)
            print(f"Using {num_processes} parallel processes.")
        except Exception:
            num_processes = 1

    _pool_size = num_processes
    _pool = mp.Pool(processes=num_processes)

    return _pool


def shutdown_pool() -> None:
    """
    Shutdown the process pool cleanly.
    
    This is called automatically at program exit via atexit,
    but can also be called manually for explicit cleanup.
    """
    global _pool, _pool_size

    if _pool is not None:
        try:
            _pool.close()
            _pool.join()
        except Exception:
            pass
        finally:
            _pool = None
            _pool_size = None


def get_pool_size() -> int:
    """
    Return the current pool size, or 0 if pool not initialized.
    """
    return _pool_size if _pool_size is not None else 0


# Register cleanup on interpreter exit
atexit.register(shutdown_pool)


def run_parallel(
    worker_fn: Callable[[Any], Any],
    task_args: list[Any],
) -> list[Any]:
    """
    Execute a worker function on multiple tasks in parallel.
    
    Args:
        worker_fn: Function to execute for each task
        task_args: List of argument tuples for the worker
        
    Returns:
        List of results corresponding to each input task
    """
    if not task_args:
        return []

    pool = get_pool()

    try:
        return pool.map(worker_fn, task_args)
    except Exception:
        # Fallback to sequential execution on pool failure
        return [worker_fn(args) for args in task_args]


# =============================================================================
# Orthogonal Distance Worker
# =============================================================================

# Type for orthogonal distance task arguments
OrthogonalDistanceArgs = tuple[
    np.ndarray,           # knots
    np.ndarray,           # cx (x coefficients)
    np.ndarray,           # cy (y coefficients)
    int,                  # degree
    float,                # px (query point x)
    float,                # py (query point y)
    tuple[float, float],  # bounds (t_min, t_max)
    float,                # tolerance
]


def orthogonal_distance_worker(args: OrthogonalDistanceArgs) -> float:
    """
    Compute orthogonal distance from a point to a B-spline curve.
    
    This is the worker function executed in child processes.
    It reconstructs the spline from passed data and finds the
    closest point via bounded scalar minimization.
    
    Args:
        args: Tuple containing:
            - knots: Knot vector (1D array)
            - cx: X control point coefficients
            - cy: Y control point coefficients  
            - degree: Spline degree
            - px, py: Query point coordinates
            - bounds: (t_min, t_max) parameter search bounds
            - tolerance: Minimizer tolerance
            
    Returns:
        Euclidean distance from point to closest curve point
    """
    try:
        knots, cx, cy, degree, px, py, bounds, tolerance = args

        # Reconstruct splines in worker process
        spline_x = ScipyBSpline(knots, cx, degree, extrapolate=False)
        spline_y = ScipyBSpline(knots, cy, degree, extrapolate=False)

        def squared_distance(t: float) -> float:
            """Squared distance from curve point at t to query point."""
            sx = spline_x(t)
            sy = spline_y(t)
            dx = sx - px
            dy = sy - py
            return dx * dx + dy * dy

        # Find minimum distance using bounded Brent's method
        result = minimize_scalar(
            squared_distance,
            bounds=bounds,
            method='bounded',
            options={'xatol': tolerance}
        )

        if getattr(result, 'success', False):
            dist_sq = result.fun
        else:
            # Fallback: evaluate at result.x anyway
            t_best = min(max(getattr(result, 'x', 0.5), bounds[0]), bounds[1])
            dist_sq = squared_distance(t_best)

        if not math.isfinite(dist_sq):
            return float('inf')

        return math.sqrt(max(0.0, dist_sq))

    except Exception:
        return float('inf')
