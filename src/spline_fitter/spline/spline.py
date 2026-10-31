"""
B-Spline Mathematics
====================

Core mathematical operations for B-spline curve construction and evaluation.

This module provides:
    - Open uniform knot vector generation
    - 2D B-spline curve evaluation using SciPy's BSpline
    - Curve derivative computation for curvature analysis
    - Signed curvature calculation

The implementation wraps SciPy's BSpline class for efficient vectorized
evaluation while providing a clean, domain-specific interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import numpy as np
from scipy.interpolate import BSpline as ScipyBSpline

from ..optimization.models import Point2D

# =============================================================================
# Knot Vector Construction
# =============================================================================

def create_open_uniform_knots(degree: int, num_control_points: int) -> np.ndarray:
    """
    Create an open uniform (clamped) knot vector on [0, 1].
    
    An open uniform knot vector has:
        - (degree + 1) repeated knots at each end
        - Uniformly spaced interior knots
    
    This ensures the curve starts at the first control point
    and ends at the last control point.
    
    Args:
        degree: Polynomial degree of the B-spline
        num_control_points: Number of control points (n)
        
    Returns:
        Numpy array of length (n + degree + 1) containing the knot vector
        
    Raises:
        ValueError: If num_control_points <= degree
        
    Example:
        >>> create_open_uniform_knots(3, 5)
        array([0. , 0. , 0. , 0. , 0.5, 1. , 1. , 1. , 1. ])
    """
    if num_control_points <= degree:
        raise ValueError(
            f"Number of control points ({num_control_points}) must be greater "
            f"than degree ({degree})"
        )

    # Number of interior knots
    num_interior = num_control_points - degree - 1

    # Build knot vector: [0,0,...,0, interior, 1,1,...,1]
    knots = [0.0] * (degree + 1)

    if num_interior > 0:
        step = 1.0 / (num_control_points - degree)
        knots.extend(i * step for i in range(1, num_interior + 1))

    knots.extend([1.0] * (degree + 1))

    return np.array(knots, dtype=np.float64)


def validate_knot_vector(knots: np.ndarray, degree: int, num_control_points: int) -> bool:
    """
    Validate that a knot vector is properly formed.
    
    Args:
        knots: The knot vector to validate
        degree: Expected spline degree
        num_control_points: Expected number of control points
        
    Returns:
        True if valid, False otherwise
    """
    expected_length = num_control_points + degree + 1

    if len(knots) != expected_length:
        return False

    # Knots must be non-decreasing
    if not np.all(np.diff(knots) >= 0):
        return False

    # For open-uniform: first (degree+1) should be equal, same for last
    if not np.allclose(knots[:degree + 1], knots[0]):
        return False
    if not np.allclose(knots[-(degree + 1):], knots[-1]):
        return False

    return True


# =============================================================================
# B-Spline Curve Class
# =============================================================================

class BSplineCurve2D:
    """
    A 2D B-spline curve defined by control points and a knot vector.
    
    This class wraps two SciPy BSpline objects (one for x, one for y)
    to provide efficient vectorized evaluation of 2D parametric curves.
    
    Attributes:
        degree: Polynomial degree of the curve
        control_points: List of (x, y) control point coordinates
        knot_vector: The knot vector defining the basis functions
        
    Example:
        >>> ctrl_pts = [(0, 0), (0.5, 0.5), (1, 0)]
        >>> curve = BSplineCurve2D(degree=2, control_points=ctrl_pts)
        >>> x, y = curve.evaluate(0.5)
        >>> print(f"Midpoint: ({x:.3f}, {y:.3f})")
        Midpoint: (0.500, 0.250)
    """

    def __init__(
        self,
        degree: int,
        control_points: Sequence[Point2D],
        knot_vector: np.ndarray | None = None,
    ):
        """
        Create a new B-spline curve.
        
        Args:
            degree: Polynomial degree (order = degree + 1)
            control_points: Sequence of (x, y) control point coordinates
            knot_vector: Custom knot vector, or None for open uniform
        """
        self.degree = degree
        self.control_points = [(float(p[0]), float(p[1])) for p in control_points]

        # Create or validate knot vector
        num_ctrl = len(self.control_points)
        if knot_vector is None:
            self.knot_vector = create_open_uniform_knots(degree, num_ctrl)
        else:
            self.knot_vector = np.asarray(knot_vector, dtype=np.float64)
            if not validate_knot_vector(self.knot_vector, degree, num_ctrl):
                raise ValueError("Invalid knot vector for given degree and control points")

        # Extract x and y coefficients for SciPy BSpline
        cx = np.array([p[0] for p in self.control_points], dtype=np.float64)
        cy = np.array([p[1] for p in self.control_points], dtype=np.float64)

        # Create SciPy BSpline objects (extrapolate=False for safety)
        self._spline_x = ScipyBSpline(self.knot_vector, cx, degree, extrapolate=False)
        self._spline_y = ScipyBSpline(self.knot_vector, cy, degree, extrapolate=False)

    @property
    def num_control_points(self) -> int:
        """Return the number of control points."""
        return len(self.control_points)

    @property
    def parameter_range(self) -> tuple[float, float]:
        """Return the valid parameter range (t_min, t_max)."""
        return (float(self.knot_vector[self.degree]),
                float(self.knot_vector[-self.degree - 1]))

    def evaluate(self, t: float) -> tuple[float, float]:
        """
        Evaluate the curve at a single parameter value.
        
        Args:
            t: Parameter value in [0, 1]
            
        Returns:
            (x, y) coordinates at parameter t
        """
        return (float(self._spline_x(t)), float(self._spline_y(t)))

    def evaluate_array(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the curve at multiple parameter values.
        
        Args:
            t: 1D numpy array of parameter values in [0, 1]
            
        Returns:
            Tuple of (x_array, y_array) with same shape as t
        """
        t = np.asarray(t, dtype=np.float64)
        return (self._spline_x(t), self._spline_y(t))

    def sample(self, num_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample the curve uniformly in parameter space.
        
        Args:
            num_points: Number of sample points
            
        Returns:
            Tuple of (x_array, y_array) of length num_points
        """
        t = np.linspace(0.0, 1.0, num_points)
        return self.evaluate_array(t)

    def derivative(self, order: int = 1) -> BSplineCurve2D:
        """
        Return a new curve representing the derivative.
        
        Args:
            order: Derivative order (1 = first derivative, etc.)
            
        Returns:
            New BSplineCurve2D representing the derivative curve
            
        Note:
            The returned curve has degree = original_degree - order
        """
        if order < 1:
            raise ValueError("Derivative order must be >= 1")
        if order > self.degree:
            raise ValueError(f"Cannot compute order-{order} derivative of degree-{self.degree} spline")

        # Use SciPy's derivative method
        dx = self._spline_x.derivative(order)
        dy = self._spline_y.derivative(order)

        # Create wrapper curve (hacky but works)
        new_degree = self.degree - order
        # Extract control points from derivative splines
        new_ctrl = list(zip(dx.c.tolist(), dy.c.tolist(), strict=False))

        result = object.__new__(BSplineCurve2D)
        result.degree = new_degree
        result.control_points = new_ctrl
        result.knot_vector = dx.t
        result._spline_x = dx
        result._spline_y = dy

        return result

    def evaluate_derivatives(
        self,
        t: np.ndarray,
        max_order: int = 2,
        min_order: int = 0,
    ) -> dict:
        """
        Evaluate the curve and its derivatives at given parameters.
        
        This is more efficient than calling derivative() multiple times
        when you need position and multiple derivative orders.
        
        Args:
            t: Parameter values to evaluate at
            max_order: Maximum derivative order to compute
            min_order: Lowest order to compute (1 leaves out the position)
            
        Returns:
            Dictionary with keys:
                'x', 'y': Position coordinates (if min_order is 0)
                'dx', 'dy': First derivatives (if max_order >= 1)
                'd2x', 'd2y': Second derivatives (if max_order >= 2)
                'd3x', 'd3y': Third derivatives (if max_order >= 3)
            Derivatives above the degree are zero.
        """
        t = np.asarray(t, dtype=np.float64)
        result = {}
        names = (('x', 'y'), ('dx', 'dy'), ('d2x', 'd2y'), ('d3x', 'd3y'))
        for order in range(min_order, min(max_order, 3) + 1):
            if order > self.degree:
                # Derivatives above the degree vanish
                result[names[order][0]] = result[names[order][1]] = np.zeros_like(t)
                continue
            # x and y in one call, each order straight from the curve's own
            # coefficients (no derivative spline built per call)
            values = self._spline_xy(t, order)
            result[names[order][0]], result[names[order][1]] = values[..., 0], values[..., 1]
        return result

    @cached_property
    def _spline_xy(self) -> ScipyBSpline:
        """x and y as one spline with two columns of coefficients."""
        return ScipyBSpline(self._spline_x.t, np.stack([self._spline_x.c, self._spline_y.c], axis=-1),
                            self._spline_x.k, extrapolate=False)


# =============================================================================
# Curvature Computation
# =============================================================================

def compute_signed_curvature(
    dx: np.ndarray,
    dy: np.ndarray,
    d2x: np.ndarray,
    d2y: np.ndarray,
) -> np.ndarray:
    """
    Compute signed curvature from first and second derivatives.
    
    The signed curvature κ is defined as:
        κ = (x'·y'' - y'·x'') / (x'² + y'²)^(3/2)
    
    Positive curvature indicates counter-clockwise turning.
    
    Args:
        dx: First derivative of x (x')
        dy: First derivative of y (y')
        d2x: Second derivative of x (x'')
        d2y: Second derivative of y (y'')
        
    Returns:
        Array of signed curvature values
    """
    numerator = dx * d2y - dy * d2x
    denominator = np.power(dx * dx + dy * dy, 1.5)

    # Avoid division by zero at stationary points
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = np.where(
            denominator > 1e-12,
            numerator / denominator,
            0.0
        )

    return curvature


def compute_curvature_derivative(
    dx: np.ndarray,
    dy: np.ndarray,
    d2x: np.ndarray,
    d2y: np.ndarray,
    d3x: np.ndarray,
    d3y: np.ndarray,
) -> np.ndarray:
    """
    Compute the derivative of curvature with respect to arc length.
    
    This measures how rapidly the curvature is changing along the curve,
    which is useful for detecting local curvature extrema and assessing
    smoothness.
    
    Args:
        dx, dy: First derivatives
        d2x, d2y: Second derivatives  
        d3x, d3y: Third derivatives
        
    Returns:
        Array of curvature derivative values (dκ/ds)
    """
    speed_sq = dx * dx + dy * dy

    # Cross products
    cross = dx * d2y - dy * d2x
    cross_prime = dx * d3y - dy * d3x

    # Dot product of velocity and acceleration
    dot = dx * d2x + dy * d2y

    # Numerator of dκ/ds formula
    numerator = cross_prime * speed_sq - 3.0 * cross * dot
    denominator = np.power(speed_sq, 3.0)  # speed^6

    with np.errstate(divide='ignore', invalid='ignore'):
        dk_ds = np.where(
            denominator > 1e-24,
            numerator / denominator,
            0.0
        )

    return dk_ds


def compute_arc_length(curve: BSplineCurve2D, num_samples: int = 1000) -> float:
    """
    Estimate the arc length of a curve by numerical integration.
    
    Uses the trapezoidal rule on uniformly sampled speed values.
    
    Args:
        curve: The B-spline curve to measure
        num_samples: Number of sample points for integration
        
    Returns:
        Approximate arc length
    """
    t = np.linspace(0.0, 1.0, num_samples)
    derivs = curve.evaluate_derivatives(t, max_order=1)

    speed = np.hypot(derivs['dx'], derivs['dy'])
    dt = 1.0 / (num_samples - 1)

    # Trapezoidal integration
    arc_length = np.trapezoid(speed, dx=dt)

    return float(arc_length)
