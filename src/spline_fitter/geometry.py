"""
Geometry Utilities
==================

Functions for processing airfoil coordinates and geometric computations.

This module provides:
    - Splitting airfoil data into upper/lower surfaces
    - Arc-length parameterization
    - Bounding box computation
    - Polyline interpolation
    - Control point bounds generation for optimization
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .optimization.models import (
    AirfoilData,
    BoundingBox,
    Bounds,
    Point2D,
    PointList,
    SplineConfig,
)

# =============================================================================
# Coordinate Splitting
# =============================================================================

def find_leading_edge_index(coordinates: Sequence[Point2D]) -> int:
    """
    Find the index of the leading edge point in airfoil coordinates.
    
    The leading edge is identified as the point closest to the origin,
    since coordinates should be normalized with LE at (0, 0).
    
    Args:
        coordinates: List of (x, y) airfoil coordinates in Selig format
        
    Returns:
        Index of the leading edge point
        
    Raises:
        ValueError: If coordinates is empty
    """
    if not coordinates:
        raise ValueError("Cannot find leading edge in empty coordinate list")

    # Find point with minimum distance to origin
    min_dist_sq = float('inf')
    le_index = 0

    for i, (x, y) in enumerate(coordinates):
        dist_sq = x * x + y * y
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            le_index = i

    return le_index


def split_at_leading_edge(coordinates: Sequence[Point2D]) -> tuple[PointList, PointList, int]:
    """
    Split airfoil coordinates into upper and lower surfaces.
    
    Assumes Selig format: coordinates run from upper trailing edge,
    around the leading edge, to lower trailing edge.
    
    This is a convenience wrapper around split_into_sections(n=2).
    
    Args:
        coordinates: List of (x, y) airfoil coordinates
        
    Returns:
        Tuple of (upper_surface, lower_surface, le_index) where:
            - upper_surface: Points from upper TE to LE (inclusive)
            - lower_surface: Points from LE to lower TE (inclusive)
            - le_index: Index of leading edge in original coordinates
            
    Raises:
        ValueError: If coordinates has fewer than 3 points
    """
    sections, junction_indices = split_into_sections(coordinates, num_sections=2)
    return sections[0], sections[1], junction_indices[0]


def split_into_sections(
    coordinates: Sequence[Point2D],
    num_sections: int,
) -> tuple[tuple[tuple[Point2D, ...], ...], tuple[int, ...]]:
    """
    Split airfoil coordinates into n curve sections using equal arc-length divisions.
    
    For n=1: Returns full wrap-around curve (all coordinates as single section)
    For even n: Splits at leading edge, then subdivides upper/lower surfaces equally.
                E.g., n=2 gives upper/lower split at LE; n=4 gives 2 upper + 2 lower.
    For odd n: Divides entire curve into n equal arc-length sections (no LE junction).
               The center section wraps around the LE.
    
    Args:
        coordinates: List of (x, y) airfoil coordinates in Selig format
        num_sections: Number of curve sections to create
        
    Returns:
        Tuple of (sections, junction_indices) where:
            - sections: Tuple of n curve sections, each as tuple of points
            - junction_indices: Tuple of n-1 junction point indices (where sections meet)
            
    Raises:
        ValueError: If coordinates has fewer than num_sections+1 points
        ValueError: If num_sections < 1
    """
    if num_sections < 1:
        raise ValueError(f"num_sections must be >= 1, got {num_sections}")

    if len(coordinates) < num_sections + 1:
        raise ValueError(
            f"Need at least {num_sections + 1} points for {num_sections} sections, "
            f"got {len(coordinates)}"
        )

    # Special case: n=1 means single wrap-around curve
    if num_sections == 1:
        return (tuple(coordinates),), ()

    # For n=2, use LE as the junction point
    if num_sections == 2:
        le_index = find_leading_edge_index(coordinates)
        upper = tuple(coordinates[:le_index + 1])
        lower = tuple(coordinates[le_index:])
        return (upper, lower), (le_index,)

    # For odd n: Split entire curve into equal arc-length sections (no LE junction)
    # This allows a center section to wrap around the LE naturally
    if num_sections % 2 == 1:
        arc_params = compute_arc_length_parameters(list(coordinates))
        junction_indices_list = []

        for i in range(1, num_sections):
            target_param = i / num_sections
            closest_idx = min(range(len(arc_params)),
                            key=lambda idx: abs(arc_params[idx] - target_param))
            junction_indices_list.append(closest_idx)

        # Build sections
        sections = []
        start_idx = 0
        for j_idx in junction_indices_list:
            section = list(coordinates[start_idx:j_idx + 1])
            sections.append(tuple(section))
            start_idx = j_idx
        # Last section
        sections.append(tuple(coordinates[start_idx:]))

        return tuple(sections), tuple(junction_indices_list)

    # For even n>2: Ensure LE is at center junction
    # Upper gets n/2 sections, lower gets n/2 sections
    le_index = find_leading_edge_index(coordinates)

    # Compute arc-length parameters for upper and lower surfaces separately
    upper_coords = list(coordinates[:le_index + 1])
    lower_coords = list(coordinates[le_index:])

    n_upper = num_sections // 2
    n_lower = num_sections // 2

    sections = []
    junction_indices = []

    # Split upper surface into n_upper sections
    if n_upper == 1:
        sections.append(tuple(upper_coords))
    else:
        upper_arc_params = compute_arc_length_parameters(upper_coords)
        upper_junctions = []
        for i in range(1, n_upper):
            target_param = i / n_upper
            closest_idx = min(range(len(upper_arc_params)),
                            key=lambda idx: abs(upper_arc_params[idx] - target_param))
            upper_junctions.append(closest_idx)

        # Build upper sections
        start_idx = 0
        for j_idx in upper_junctions:
            section = list(upper_coords[start_idx:j_idx + 1])
            sections.append(tuple(section))
            junction_indices.append(j_idx)  # Index in original coordinates
            start_idx = j_idx
        # Last upper section ends at LE
        sections.append(tuple(upper_coords[start_idx:]))

    # LE junction
    junction_indices.append(le_index)

    # Split lower surface into n_lower sections
    if n_lower == 1:
        sections.append(tuple(lower_coords))
    else:
        lower_arc_params = compute_arc_length_parameters(lower_coords)
        lower_junctions = []
        for i in range(1, n_lower):
            target_param = i / n_lower
            closest_idx = min(range(len(lower_arc_params)),
                            key=lambda idx: abs(lower_arc_params[idx] - target_param))
            lower_junctions.append(closest_idx)

        # Build lower sections
        start_idx = 0
        for j_idx in lower_junctions:
            section = list(lower_coords[start_idx:j_idx + 1])
            sections.append(tuple(section))
            # Convert to original coordinate index
            junction_indices.append(le_index + j_idx)
            start_idx = j_idx
        # Last lower section ends at TE
        sections.append(tuple(lower_coords[start_idx:]))

    return tuple(sections), tuple(junction_indices)


def create_airfoil_data(
    coordinates: Sequence[Point2D],
    num_sections: int = 2,
    filename: str = "",
    name: str = "",
    precision: int = 6,
) -> AirfoilData:
    """
    Create an AirfoilData object from raw coordinates.
    
    Handles the coordinate splitting and packages everything into
    an immutable data structure.
    
    Args:
        coordinates: List of (x, y) normalized airfoil coordinates
        num_sections: Number of curve sections to create
        filename: Original filename (without path)
        name: Airfoil name from file content
        precision: Number of decimal places in original data
        
    Returns:
        AirfoilData object with n split sections
    """
    sections, junction_indices = split_into_sections(coordinates, num_sections)

    return AirfoilData(
        coordinates=tuple(coordinates),
        sections=sections,
        junction_indices=junction_indices,
        num_sections=num_sections,
        filename=filename,
        name=name,
        precision=precision,
    )


# =============================================================================
# Parameterization
# =============================================================================

def compute_arc_length_parameters(points: Sequence[Point2D]) -> list[float]:
    """
    Compute normalized arc-length parameters for a sequence of points.
    
    Each point is assigned a parameter value proportional to the
    cumulative arc length from the first point. The parameters
    are normalized to [0, 1].
    
    Args:
        points: Sequence of (x, y) coordinates forming a polyline
        
    Returns:
        List of parameter values with same length as points,
        ranging from 0.0 (first point) to 1.0 (last point)
    """
    if not points:
        return []

    if len(points) == 1:
        return [0.0]

    # Compute cumulative distances
    cumulative = [0.0]
    total = 0.0

    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        segment_length = math.hypot(dx, dy)
        total += segment_length
        cumulative.append(total)

    # Normalize to [0, 1]
    if total <= 0:
        # Degenerate case: all points coincident
        return [i / (len(points) - 1) for i in range(len(points))]

    return [c / total for c in cumulative]


def centripetal_parametrize(points: Sequence[Point2D], alpha: float = 0.5) -> list[float]:
    """
    Compute parametrization using centripetal or chord-length method.
    
    This is an alternative to arc-length parametrization that can provide
    better results for curves with sharp corners or high curvature regions.
    
    Args:
        points: Sequence of (x, y) coordinates forming a polyline
        alpha: Exponent controlling the parametrization:
               - alpha=0.0: Uniform parametrization
               - alpha=0.5: Centripetal parametrization (recommended for most cases)
               - alpha=1.0: Chord-length parametrization (same as arc-length)
               
    Returns:
        List of parameter values with same length as points,
        ranging from 0.0 (first point) to 1.0 (last point)
    """
    if not points:
        return []

    if len(points) == 1:
        return [0.0]

    # Convert to numpy for efficient computation
    pts = np.array(points, dtype=np.float64)

    # Compute segment lengths
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)

    # Apply alpha exponent
    dt = np.power(dists, alpha)

    # Cumulative sum with leading zero
    t = np.concatenate([[0.0], np.cumsum(dt)])

    # Normalize to [0, 1]
    if t[-1] <= 0:
        # Degenerate case: all points coincident
        return [i / (len(points) - 1) for i in range(len(points))]

    return (t / t[-1]).tolist()


def interpolate_along_polyline(
    points: Sequence[Point2D],
    t_params: Sequence[float],
    query_t: Sequence[float],
) -> list[Point2D]:
    """
    Interpolate positions along a polyline at specified parameter values.
    
    Uses linear interpolation between consecutive points based on
    the arc-length parameterization.
    
    Args:
        points: The polyline vertices
        t_params: Parameter value for each vertex (same length as points)
        query_t: Parameter values at which to interpolate
        
    Returns:
        List of interpolated (x, y) positions
    """
    if not points:
        return [(0.0, 0.0) for _ in query_t]

    if len(points) == 1:
        return [points[0] for _ in query_t]

    n = len(points)
    t_arr = np.asarray(t_params, dtype=np.float64)

    results = []
    for t_val in query_t:
        # Handle boundary cases
        if t_val <= t_arr[0]:
            results.append((float(points[0][0]), float(points[0][1])))
            continue

        if t_val >= t_arr[-1]:
            results.append((float(points[-1][0]), float(points[-1][1])))
            continue

        # Find the segment containing t_val
        idx = int(np.searchsorted(t_arr, t_val)) - 1
        idx = max(0, min(idx, n - 2))

        # Linear interpolation within segment
        t0, t1 = t_arr[idx], t_arr[idx + 1]
        if t1 <= t0:
            weight = 0.0
        else:
            weight = (t_val - t0) / (t1 - t0)

        x0, y0 = points[idx]
        x1, y1 = points[idx + 1]

        x = x0 + weight * (x1 - x0)
        y = y0 + weight * (y1 - y0)

        results.append((float(x), float(y)))

    return results


# =============================================================================
# Bounding Box
# =============================================================================

def compute_bounding_box(points: Sequence[Point2D]) -> BoundingBox:
    """
    Compute the axis-aligned bounding box of a point set.
    
    Args:
        points: Sequence of (x, y) coordinates
        
    Returns:
        BoundingBox containing all points
        
    Raises:
        ValueError: If points is empty
    """
    if not points:
        raise ValueError("Cannot compute bounding box of empty point set")

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    return BoundingBox(
        x_min=min(x_coords),
        x_max=max(x_coords),
        y_min=min(y_coords),
        y_max=max(y_coords),
    )


# =============================================================================
# Optimization Bounds
# =============================================================================

def compute_optimization_bounds(
    airfoil: AirfoilData,
    config: SplineConfig,
    optimize_le_position: bool = False,
) -> list[Bounds]:
    """
    Compute the variable bounds for optimization.
    
    The bounds are derived from the airfoil bounding box, scaled by
    the configuration's bbox_scale factor. Different continuity modes
    require different variables with different bounds.
    
    Args:
        airfoil: The airfoil data being fitted
        config: Spline configuration specifying continuity mode, etc.
        optimize_le_position: If True, add bounds for every junction (x, y) at the end
        
    Returns:
        List of (lower, upper) bounds for each optimization variable
    """
    # Compute scaled bounding box
    bbox = compute_bounding_box(airfoil.coordinates)
    scaled_bbox = bbox.scaled(config.bbox_scale)

    x_lo, x_hi = scaled_bbox.x_bounds()
    y_lo, y_hi = scaled_bbox.y_bounds()

    # Ensure positive dimensions for derivative bounds
    base_w = max(bbox.width, 1e-6)
    base_h = max(bbox.height, 1e-6)

    bounds: list[Bounds] = []

    def add_point_bounds(count: int) -> None:
        """Add bounds for control point (x, y) pairs."""
        for _ in range(count):
            bounds.append((x_lo, x_hi))
            bounds.append((y_lo, y_hi))

    # Get parameters for n-section architecture
    num_sections = config.num_sections
    n_ctrl = config.num_control_points
    mode = config.continuity
    n_constrained = mode.constrained_control_points

    # Compute free control points per section
    def get_n_free_for_section(section_idx: int) -> int:
        if num_sections == 1:
            return n_ctrl - 2  # Just endpoints
        elif section_idx == 0 or section_idx == num_sections - 1:
            # End sections: 1 endpoint + n_free + n_constrained junction CPs
            return max(0, n_ctrl - 1 - n_constrained)
        else:
            # Interior: n_constrained at each junction
            return max(0, n_ctrl - 2 * n_constrained)

    # Add bounds for free control points in all sections
    for section_idx in range(num_sections):
        n_free = get_n_free_for_section(section_idx)
        add_point_bounds(n_free)

    # Add bounds for junction parameters (n-1 junctions for n sections)
    num_junctions = max(0, num_sections - 1)

    # Parametric derivatives grow with the knot density: the k-th by about
    # (n_ctrl - degree)^k times the airfoil's size, along either axis (the
    # tangent is vertical at the leading edge)
    span = max(base_w, base_h)
    density = max(1, n_ctrl - config.degree)
    order = mode.constrained_control_points - 1
    for _junction_idx in range(num_junctions):
        for k, factor in zip((1, 2, 3), (3.0, 10.0, 50.0), strict=False):
            if k <= order:
                limit = factor * span * density ** k
                bounds.append((-limit, limit))  # Dk x
                bounds.append((-limit, limit))  # Dk y

    # Optional junction positions (appended at end): anywhere a free control
    # point may go. The chain is fitted to all points as a whole, so a junction
    # is not tied to its split point, which is only where it starts
    if optimize_le_position and num_sections >= 2:
        add_point_bounds(num_sections - 1)

    return bounds


def clamp_to_bounds(
    values: Sequence[float],
    bounds: Sequence[Bounds],
) -> list[float]:
    """
    Clamp values to their corresponding bounds.
    
    Args:
        values: The values to clamp
        bounds: Sequence of (lower, upper) bound pairs
        
    Returns:
        List of clamped values
    """
    result = []
    for value, (lo, hi) in zip(values, bounds, strict=False):
        lo = lo if lo is not None else float('-inf')
        hi = hi if hi is not None else float('inf')
        result.append(min(max(float(value), float(lo)), float(hi)))
    return result
