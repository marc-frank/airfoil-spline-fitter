"""
Vectorized Foot-Point Projection
================================

Fast foot-point computation for B-spline curves using Newton-Raphson iteration.

This module replaces expensive `minimize_scalar` calls with a vectorized
Taylor expansion approach. The key insight is that for reasonably good
initial parameter estimates, a few Newton-Raphson iterations converge
to the true foot-point (closest point on curve).

Mathematical Background
-----------------------
Given a parametric curve C(t) = (x(t), y(t)) and a data point P,
we want to find t* such that the distance ||C(t*) - P|| is minimized.

The necessary condition for a minimum is that the error vector is
orthogonal to the curve tangent:

    f(t) = dot(C(t) - P, C'(t)) = 0

Newton-Raphson update:
    t_new = t - f(t) / f'(t)

Where:
    f(t)  = (C(t) - P) · C'(t)
    f'(t) = ||C'(t)||² + (C(t) - P) · C''(t)

For well-behaved curves (no cusps, moderate curvature), this converges
in 2-4 iterations from arc-length initialization.

Simplified Update (Gauss-Newton approximation)
----------------------------------------------
Neglecting the second derivative term gives:
    t_new = t - dot(C(t) - P, C'(t)) / ||C'(t)||²

This is simpler and often sufficient when starting close to the solution.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline as ScipyBSpline

# =============================================================================
# Vectorized Foot-Point Updates
# =============================================================================

def vectorized_footpoint_update(
    t_values: np.ndarray,
    data_points: np.ndarray,
    spline_x: ScipyBSpline,
    spline_y: ScipyBSpline,
    use_second_derivative: bool = True,
) -> np.ndarray:
    """
    Perform one vectorized Newton-Raphson step for foot-point projection.
    
    Computes the update for all points simultaneously using NumPy broadcasting.
    This replaces N separate minimize_scalar calls with a single vectorized
    evaluation.
    
    The update formula is:
        t_new = t - f(t) / f'(t)
    
    Where:
        f(t)  = (C(t) - P) · C'(t)        (orthogonality condition)
        f'(t) = ||C'(t)||² + (C(t) - P) · C''(t)  (full Newton)
        
    Or simplified (Gauss-Newton):
        f'(t) ≈ ||C'(t)||²
    
    Args:
        t_values: Parameter values for each data point, shape (N,).
                  Should be in [0, 1].
        data_points: Data points to project, shape (N, 2).
                     Each row is (x, y).
        spline_x: SciPy BSpline object for x-coordinate.
        spline_y: SciPy BSpline object for y-coordinate.
        use_second_derivative: If True, use full Newton with C''(t).
                               If False, use Gauss-Newton approximation.
                               Default True for better convergence.
    
    Returns:
        Updated parameter values, shape (N,), clamped to [0, 1].
        
    Notes:
        - Handles degenerate cases where ||C'(t)|| ≈ 0 by not updating.
        - All operations are vectorized - no Python loops over data points.
        - The spline derivatives are computed via scipy's derivative method.
    
    Example:
        >>> t = np.array([0.2, 0.5, 0.8])
        >>> pts = np.array([[0.1, 0.05], [0.5, 0.02], [0.9, -0.01]])
        >>> t_new = vectorized_footpoint_update(t, pts, spline_x, spline_y)
    """
    t_values = np.asarray(t_values, dtype=np.float64)
    data_points = np.asarray(data_points, dtype=np.float64)

    n_points = len(t_values)

    if n_points == 0:
        return t_values.copy()

    # Ensure t is within valid domain (with small epsilon for numerical safety)
    t_clamped = np.clip(t_values, 0.0, 1.0)

    # Evaluate curve position: C(t) = (x(t), y(t))
    # Shape: (N,) for each coordinate
    cx = spline_x(t_clamped)
    cy = spline_y(t_clamped)

    # Evaluate first derivative: C'(t) = (x'(t), y'(t))
    spline_dx = spline_x.derivative(1)
    spline_dy = spline_y.derivative(1)
    dx = spline_dx(t_clamped)
    dy = spline_dy(t_clamped)

    # Error vector: C(t) - P
    # Shape: (N,) for each coordinate
    ex = cx - data_points[:, 0]
    ey = cy - data_points[:, 1]

    # f(t) = (C(t) - P) · C'(t) = ex * dx + ey * dy
    # Shape: (N,)
    f = ex * dx + ey * dy

    # ||C'(t)||² = dx² + dy²
    # Shape: (N,)
    tangent_norm_sq = dx * dx + dy * dy

    if use_second_derivative:
        # Full Newton: f'(t) = ||C'(t)||² + (C(t) - P) · C''(t)
        spline_ddx = spline_x.derivative(2)
        spline_ddy = spline_y.derivative(2)
        ddx = spline_ddx(t_clamped)
        ddy = spline_ddy(t_clamped)

        # (C(t) - P) · C''(t)
        error_dot_curvature = ex * ddx + ey * ddy

        # f'(t) = ||C'(t)||² + (C(t) - P) · C''(t)
        f_prime = tangent_norm_sq + error_dot_curvature
    else:
        # Gauss-Newton approximation: f'(t) ≈ ||C'(t)||²
        f_prime = tangent_norm_sq

    # Safeguard against division by zero or near-zero tangent
    # This can happen at cusps or if spline is degenerate
    MIN_TANGENT_NORM_SQ = 1e-14

    # Create mask for valid updates
    valid = np.abs(f_prime) > MIN_TANGENT_NORM_SQ

    # Newton update: t_new = t - f(t) / f'(t)
    t_new = t_clamped.copy()
    t_new[valid] = t_clamped[valid] - f[valid] / f_prime[valid]

    # Clamp to valid parameter range
    t_new = np.clip(t_new, 0.0, 1.0)

    return t_new


def iterative_footpoint_projection(
    t_init: np.ndarray,
    data_points: np.ndarray,
    spline_x: ScipyBSpline,
    spline_y: ScipyBSpline,
    max_iter: int = 3,
    tol: float = 1e-8,
    use_second_derivative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively project data points onto the spline curve.
    
    Starting from initial parameter estimates (e.g., from arc-length
    parameterization), perform Newton-Raphson iterations until convergence
    or max iterations reached.
    
    Args:
        t_init: Initial parameter values, shape (N,).
        data_points: Data points to project, shape (N, 2).
        spline_x: SciPy BSpline for x-coordinate.
        spline_y: SciPy BSpline for y-coordinate.
        max_iter: Maximum number of Newton iterations. Default 3.
                  Usually 2-4 iterations suffice for good convergence.
        tol: Convergence tolerance on parameter change. Default 1e-8.
             Iteration stops when max(|t_new - t|) < tol.
        use_second_derivative: Use full Newton (True) or Gauss-Newton (False).
    
    Returns:
        Tuple of:
            - t_converged: Final parameter values, shape (N,)
            - distances: Euclidean distances from each point to curve, shape (N,)
    
    Notes:
        - All operations are fully vectorized.
        - For typical airfoil fitting with arc-length initialization,
          3 iterations achieve machine-precision convergence.
        - Early exit if all points converge before max_iter.
    
    Example:
        >>> t_init = arc_length_parameters(data_points)
        >>> t_final, dists = iterative_footpoint_projection(
        ...     t_init, data_points, spline_x, spline_y, max_iter=3
        ... )
    """
    t_init = np.asarray(t_init, dtype=np.float64)
    data_points = np.asarray(data_points, dtype=np.float64)

    if len(t_init) == 0:
        return t_init.copy(), np.array([], dtype=np.float64)

    t_current = np.clip(t_init, 0.0, 1.0)

    for iteration in range(max_iter):
        t_new = vectorized_footpoint_update(
            t_current,
            data_points,
            spline_x,
            spline_y,
            use_second_derivative=use_second_derivative,
        )

        # Check convergence
        max_change = np.max(np.abs(t_new - t_current))
        t_current = t_new

        if max_change < tol:
            break

    # Compute final distances
    distances = compute_distances(t_current, data_points, spline_x, spline_y)

    return t_current, distances


def compute_distances(
    t_values: np.ndarray,
    data_points: np.ndarray,
    spline_x: ScipyBSpline,
    spline_y: ScipyBSpline,
) -> np.ndarray:
    """
    Compute Euclidean distances from data points to curve at given parameters.
    
    Args:
        t_values: Parameter values, shape (N,).
        data_points: Data points, shape (N, 2).
        spline_x: SciPy BSpline for x-coordinate.
        spline_y: SciPy BSpline for y-coordinate.
    
    Returns:
        Distances, shape (N,).
    """
    t_values = np.asarray(t_values, dtype=np.float64)
    data_points = np.asarray(data_points, dtype=np.float64)

    if len(t_values) == 0:
        return np.array([], dtype=np.float64)

    t_clamped = np.clip(t_values, 0.0, 1.0)

    cx = spline_x(t_clamped)
    cy = spline_y(t_clamped)

    dx = cx - data_points[:, 0]
    dy = cy - data_points[:, 1]

    return np.sqrt(dx * dx + dy * dy)


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def batch_footpoint_projection(
    t_init: np.ndarray,
    data_points: np.ndarray,
    knots: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    degree: int,
    max_iter: int = 3,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project data points onto a B-spline curve specified by coefficients.
    
    This is a convenience wrapper that constructs the SciPy BSpline objects
    from raw arrays, useful when coefficients change during optimization.
    
    Args:
        t_init: Initial parameter values, shape (N,).
        data_points: Data points to project, shape (N, 2).
        knots: Knot vector, shape (n_knots,).
        cx: X-coordinate control points / coefficients, shape (n_ctrl,).
        cy: Y-coordinate control points / coefficients, shape (n_ctrl,).
        degree: Spline degree.
        max_iter: Maximum Newton iterations.
        tol: Convergence tolerance.
    
    Returns:
        Tuple of (t_converged, distances).
    """
    spline_x = ScipyBSpline(knots, cx, degree, extrapolate=False)
    spline_y = ScipyBSpline(knots, cy, degree, extrapolate=False)

    return iterative_footpoint_projection(
        t_init, data_points, spline_x, spline_y,
        max_iter=max_iter, tol=tol
    )


def compute_orthogonal_rms_error(
    t_init: np.ndarray,
    data_points: np.ndarray,
    spline_x: ScipyBSpline,
    spline_y: ScipyBSpline,
    max_iter: int = 3,
) -> float:
    """
    Compute RMS orthogonal distance error after foot-point projection.
    
    This is the primary metric for orthogonal distance fitting quality.
    
    Args:
        t_init: Initial parameter estimates, shape (N,).
        data_points: Data points, shape (N, 2).
        spline_x: SciPy BSpline for x-coordinate.
        spline_y: SciPy BSpline for y-coordinate.
        max_iter: Newton iterations for foot-point projection.
    
    Returns:
        RMS (root mean square) orthogonal distance error.
    """
    _, distances = iterative_footpoint_projection(
        t_init, data_points, spline_x, spline_y, max_iter=max_iter
    )

    if len(distances) == 0:
        return 0.0

    return float(np.sqrt(np.mean(distances ** 2)))


def compute_orthogonal_max_error(
    t_init: np.ndarray,
    data_points: np.ndarray,
    spline_x: ScipyBSpline,
    spline_y: ScipyBSpline,
    max_iter: int = 3,
) -> float:
    """
    Compute maximum orthogonal distance error after foot-point projection.
    
    Args:
        t_init: Initial parameter estimates, shape (N,).
        data_points: Data points, shape (N, 2).
        spline_x: SciPy BSpline for x-coordinate.
        spline_y: SciPy BSpline for y-coordinate.
        max_iter: Newton iterations for foot-point projection.
    
    Returns:
        Maximum orthogonal distance error.
    """
    _, distances = iterative_footpoint_projection(
        t_init, data_points, spline_x, spline_y, max_iter=max_iter
    )

    if len(distances) == 0:
        return 0.0

    return float(np.max(distances))


# =============================================================================
# Damped Newton for Difficult Cases
# =============================================================================

def damped_footpoint_update(
    t_values: np.ndarray,
    data_points: np.ndarray,
    spline_x: ScipyBSpline,
    spline_y: ScipyBSpline,
    damping: float = 0.5,
) -> np.ndarray:
    """
    Perform a damped Newton step for more robust convergence.
    
    Uses the update: t_new = t - damping * f(t) / f'(t)
    
    This is useful when starting far from the solution or when the
    curve has high curvature regions where full Newton steps might
    overshoot.
    
    Args:
        t_values: Current parameter values, shape (N,).
        data_points: Data points to project, shape (N, 2).
        spline_x: SciPy BSpline for x-coordinate.
        spline_y: SciPy BSpline for y-coordinate.
        damping: Step size damping factor in (0, 1]. Default 0.5.
    
    Returns:
        Updated parameter values, shape (N,), clamped to [0, 1].
    """
    t_values = np.asarray(t_values, dtype=np.float64)
    data_points = np.asarray(data_points, dtype=np.float64)

    if len(t_values) == 0:
        return t_values.copy()

    t_clamped = np.clip(t_values, 0.0, 1.0)

    # Evaluate curve and derivatives
    cx = spline_x(t_clamped)
    cy = spline_y(t_clamped)

    spline_dx = spline_x.derivative(1)
    spline_dy = spline_y.derivative(1)
    dx = spline_dx(t_clamped)
    dy = spline_dy(t_clamped)

    spline_ddx = spline_x.derivative(2)
    spline_ddy = spline_y.derivative(2)
    ddx = spline_ddx(t_clamped)
    ddy = spline_ddy(t_clamped)

    # Error vector
    ex = cx - data_points[:, 0]
    ey = cy - data_points[:, 1]

    # f(t) = (C(t) - P) · C'(t)
    f = ex * dx + ey * dy

    # f'(t) = ||C'(t)||² + (C(t) - P) · C''(t)
    tangent_norm_sq = dx * dx + dy * dy
    error_dot_curvature = ex * ddx + ey * ddy
    f_prime = tangent_norm_sq + error_dot_curvature

    # Safeguard
    MIN_TANGENT_NORM_SQ = 1e-14
    valid = np.abs(f_prime) > MIN_TANGENT_NORM_SQ

    # Damped Newton update
    t_new = t_clamped.copy()
    t_new[valid] = t_clamped[valid] - damping * f[valid] / f_prime[valid]

    return np.clip(t_new, 0.0, 1.0)


def robust_footpoint_projection(
    t_init: np.ndarray,
    data_points: np.ndarray,
    spline_x: ScipyBSpline,
    spline_y: ScipyBSpline,
    max_iter: int = 5,
    tol: float = 1e-8,
    initial_damping: float = 0.5,
    damping_increase: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust foot-point projection with adaptive damping.
    
    Starts with damped Newton steps and progressively increases
    the step size as convergence improves. This provides better
    global convergence properties while maintaining fast local
    convergence.
    
    Args:
        t_init: Initial parameter values, shape (N,).
        data_points: Data points to project, shape (N, 2).
        spline_x: SciPy BSpline for x-coordinate.
        spline_y: SciPy BSpline for y-coordinate.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        initial_damping: Starting damping factor.
        damping_increase: Factor to increase damping each iteration.
    
    Returns:
        Tuple of (t_converged, distances).
    """
    t_init = np.asarray(t_init, dtype=np.float64)
    data_points = np.asarray(data_points, dtype=np.float64)

    if len(t_init) == 0:
        return t_init.copy(), np.array([], dtype=np.float64)

    t_current = np.clip(t_init, 0.0, 1.0)
    damping = initial_damping

    for iteration in range(max_iter):
        # Use damped update for first iterations, then switch to full Newton
        if damping < 1.0:
            t_new = damped_footpoint_update(
                t_current, data_points, spline_x, spline_y, damping=damping
            )
            damping = min(1.0, damping * damping_increase)
        else:
            t_new = vectorized_footpoint_update(
                t_current, data_points, spline_x, spline_y,
                use_second_derivative=True
            )

        max_change = np.max(np.abs(t_new - t_current))
        t_current = t_new

        if max_change < tol:
            break

    distances = compute_distances(t_current, data_points, spline_x, spline_y)

    return t_current, distances
