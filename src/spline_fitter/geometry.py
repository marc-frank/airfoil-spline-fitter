"""
Geometry Utilities
==================

Functions for processing airfoil coordinates and geometric computations.

This module provides:
    - Splitting airfoil data into upper/lower surfaces
    - Arc-length parameterization
    - Bounding box computation
    - Polyline interpolation
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .optimization.models import (
    AirfoilData,
    BoundingBox,
    Point2D,
    PointList,
)

# =============================================================================
# Coordinate Splitting
# =============================================================================

def find_leading_edge_index(coordinates: Sequence[Point2D]) -> int:
    """
    Find the index of the leading edge point in airfoil coordinates.
    
    The leading edge is the point farthest from the trailing edge (the
    midpoint of the first and last points), the rule the file reader
    normalizes by, so it is found wherever the data sits: shifted, scaled,
    rotated or in millimetres. The first of equally far points is taken.

    Args:
        coordinates: List of (x, y) airfoil coordinates in Selig format

    Returns:
        Index of the leading edge point

    Raises:
        ValueError: If coordinates is empty
    """
    if len(coordinates) == 0:
        raise ValueError("Cannot find leading edge in empty coordinate list")

    te_x = (coordinates[0][0] + coordinates[-1][0]) / 2.0
    te_y = (coordinates[0][1] + coordinates[-1][1]) / 2.0
    max_dist_sq = -1.0
    le_index = 0

    for i, (x, y) in enumerate(coordinates):
        dist_sq = (x - te_x) ** 2 + (y - te_y) ** 2
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
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
        junction_indices_list = _junctions(arc_params, num_sections)

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
    if len(upper_coords) < n_upper + 1 or len(lower_coords) < n_lower + 1:
        raise ValueError(
            f"Need at least {n_upper + 1} points on each side of the leading edge for "
            f"{num_sections} sections, got {len(upper_coords)} and {len(lower_coords)}"
        )

    sections = []
    junction_indices = []

    # Split upper surface into n_upper sections
    if n_upper == 1:
        sections.append(tuple(upper_coords))
    else:
        upper_junctions = _junctions(compute_arc_length_parameters(upper_coords), n_upper)

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
        lower_junctions = _junctions(compute_arc_length_parameters(lower_coords), n_lower)

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


def _junctions(arc_params: Sequence[float], num_sections: int) -> list[int]:
    """Indices splitting a polyline into ``num_sections`` of about equal arc
    length: the point nearest each division, moved where needed so every
    section keeps at least one segment (two points)."""
    last = len(arc_params) - 1
    junctions = [min(range(len(arc_params)), key=lambda idx: abs(arc_params[idx] - i / num_sections))
                 for i in range(1, num_sections)]
    for k in range(len(junctions)):
        junctions[k] = max(junctions[k], junctions[k - 1] + 1 if k else 1)
    for k in reversed(range(len(junctions))):
        junctions[k] = min(junctions[k], junctions[k + 1] - 1 if k + 1 < len(junctions) else last - 1)
    return junctions


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
