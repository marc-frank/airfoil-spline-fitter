"""
Control Point Construction
==========================

Translation between the optimizer's flat variable vector and the
per-section control points, including junction continuity and the
initial guess.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..geometry import interpolate_along_polyline
from ..spline.spline import BSplineCurve2D, create_open_uniform_knots
from .constraints import compute_derivative_constraint_coeffs
from .models import (
    Bounds,
    ContinuityMode,
    Point2D,
    PointList,
    SplineConfig,
)

# Use shared implementation from constraints module
_compute_derivative_constraint_coeffs = compute_derivative_constraint_coeffs


# =============================================================================
# Control Point Construction
# =============================================================================

def construct_control_points(
    variables: np.ndarray,
    config: SplineConfig,
    sections: list[PointList],
    optimize_le_position: bool = False,
) -> list[PointList]:
    """
    Construct control points for n curve sections from optimization variables.
    
    This function interprets the flat optimization variable vector
    according to the specified continuity mode and constructs the
    full set of control points for all n sections.
    
    For n=1 (single wrap-around curve):
        - No junctions, just free control points + endpoints
        
    For n>=2:
        - Each section has free control points
        - n-1 junctions with continuity constraint parameters
        - Constraints applied at each junction based on ContinuityMode
    
    Variable layout:
        [section_0_free_pts, section_1_free_pts, ..., section_{n-1}_free_pts,
         junction_0_params, junction_1_params, ..., junction_{n-2}_params]
    
    Args:
        variables: Flat array of optimization variables
        config: Spline configuration with continuity mode
        sections: List of n point lists (original section data for endpoints)
        optimize_le_position: If True, every junction's (x, y) is at the end of
            the variables (only for n>=2)
        
    Returns:
        List of n control point lists, one per section
    """
    degree = config.degree
    n_ctrl = config.num_control_points
    mode = config.continuity
    num_sections = len(sections)

    # Create knot vector for derivative coefficient computation
    knots = create_open_uniform_knots(degree, n_ctrl)

    def unpack_points(flat: Sequence[float]) -> PointList:
        """Convert flat array to list of (x, y) tuples."""
        return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]

    # -------------------------------------------------------------------------
    # Special case: n=1 (single wrap-around curve, no junctions)
    # -------------------------------------------------------------------------
    if num_sections == 1:
        # For n=1, all control points are free except endpoints
        section = sections[0]
        start_pt = section[0]
        end_pt = section[-1]

        n_free = max(0, n_ctrl - 2)  # Exclude start and end
        f2 = 2 * n_free

        free_pts = unpack_points(variables[:f2])
        ctrl_pts = [start_pt] + free_pts + [end_pt]

        return [ctrl_pts]

    # -------------------------------------------------------------------------
    # General case: n>=2 (junctions with continuity constraints)
    # -------------------------------------------------------------------------
    # Number of free control points per section
    # Each section has: [start_endpoint] + [free_pts] + [constrained_junction_pts]
    # For interior sections, both ends have junction constraints
    # For end sections, one end is free (airfoil endpoint), one has junction constraints

    n_constrained = mode.constrained_control_points  # CPs constrained at each junction side

    # For end sections (first and last):
    #   n_ctrl = 1 (endpoint) + n_free + n_constrained (junction side)
    # For interior sections (if n > 2):
    #   n_ctrl = n_constrained (left junction) + n_free + n_constrained (right junction)
    #   But the shared junction point counts once, so effectively:
    #   n_ctrl = 1 (shared point) + (n_constrained-1) + n_free + (n_constrained-1) + 1

    # Simplified: for uniform treatment, each section has some free CPs
    # n_free_per_section = n_ctrl - 1 (endpoint) - (n_constrained - 1) (junction side)
    # But for sections in the middle, both ends are junction-constrained

    # Let's use a simpler approach:
    # - First section: 1 endpoint + n_free + (n_constrained-1) junction CPs + 1 shared junction point
    # - Interior sections: 1 shared junction + (n_constrained-1) + n_free + (n_constrained-1) + 1 shared
    # - Last section: 1 shared junction + (n_constrained-1) + n_free + 1 endpoint

    # For simplicity, let's compute n_free for each position
    def get_n_free_for_section(section_idx: int) -> int:
        """Get number of free control points for a section."""
        if num_sections == 1:
            return n_ctrl - 2  # Just endpoints
        elif section_idx == 0:
            # First section: 1 endpoint + n_free + n_constrained junction CPs
            return max(0, n_ctrl - 1 - n_constrained)
        elif section_idx == num_sections - 1:
            # Last section: n_constrained junction CPs + n_free + 1 endpoint
            return max(0, n_ctrl - 1 - n_constrained)
        else:
            # Interior: n_constrained left + n_free + n_constrained right
            return max(0, n_ctrl - 2 * n_constrained)

    # Calculate total free point variables
    total_free_pts = sum(get_n_free_for_section(i) for i in range(num_sections))
    free_vars_end = 2 * total_free_pts  # x, y for each point

    # Extract free points for each section
    section_free_pts = []
    var_idx = 0
    for i in range(num_sections):
        n_free = get_n_free_for_section(i)
        f2 = 2 * n_free
        section_free_pts.append(unpack_points(variables[var_idx:var_idx + f2]))
        var_idx += f2

    # Junction parameters start after free points
    junction_var_start = free_vars_end

    # Determine junction parameter count based on continuity mode
    # (Only parametric/C^n continuity is supported)
    if mode == ContinuityMode.G0:
        params_per_junction = 0  # Junction point is just the shared endpoint
    elif mode == ContinuityMode.G1:
        params_per_junction = 2  # D1x, D1y (same derivative both sides)
    elif mode == ContinuityMode.G2:
        params_per_junction = 4  # D1x, D1y, D2x, D2y
    else:  # G3
        params_per_junction = 6  # D1x, D1y, D2x, D2y, D3x, D3y

    # Junction points: the split points, or one moving (x, y) per junction at
    # the end of the variables
    junction_pts = [section[-1] for section in sections[:-1]]
    if optimize_le_position:
        tail = len(variables) - 2 * len(junction_pts)
        junction_pts = [(float(variables[tail + 2 * j]), float(variables[tail + 2 * j + 1]))
                        for j in range(len(junction_pts))]

    # Build control points for each section
    result = []

    for section_idx in range(num_sections):
        section = sections[section_idx]
        free_pts = section_free_pts[section_idx]

        # Determine endpoints and junction constraints
        if section_idx == 0:
            # First section: starts at airfoil start, ends at first junction
            start_pt = section[0]
            junction_idx = 0

            junction_pt = junction_pts[junction_idx]

            # Compute constrained CPs at junction end
            if mode == ContinuityMode.G0:
                # Just position continuity
                ctrl_pts = [start_pt] + free_pts + [junction_pt]
            else:
                # Higher continuity - extract junction parameters
                jp_start = junction_var_start + junction_idx * params_per_junction
                junction_params = variables[jp_start:jp_start + params_per_junction]

                constrained_end = _compute_junction_cps_before(
                    junction_pt, junction_params, mode, knots, n_ctrl, degree
                )
                ctrl_pts = [start_pt] + free_pts + constrained_end + [junction_pt]

        elif section_idx == num_sections - 1:
            # Last section: starts at last junction, ends at airfoil end
            junction_idx = section_idx - 1
            end_pt = section[-1]

            junction_pt = junction_pts[junction_idx]

            # Compute constrained CPs at junction start
            if mode == ContinuityMode.G0:
                ctrl_pts = [junction_pt] + free_pts + [end_pt]
            else:
                jp_start = junction_var_start + junction_idx * params_per_junction
                junction_params = variables[jp_start:jp_start + params_per_junction]

                constrained_start = _compute_junction_cps_after(
                    junction_pt, junction_params, mode, knots, n_ctrl, degree
                )
                ctrl_pts = [junction_pt] + constrained_start + free_pts + [end_pt]
        else:
            # Interior section: junction at both ends
            left_junction_idx = section_idx - 1
            right_junction_idx = section_idx

            left_junction_pt = junction_pts[left_junction_idx]
            right_junction_pt = junction_pts[right_junction_idx]

            if mode == ContinuityMode.G0:
                ctrl_pts = [left_junction_pt] + free_pts + [right_junction_pt]
            else:
                # Left junction constrained CPs (after junction)
                jp_left_start = junction_var_start + left_junction_idx * params_per_junction
                left_params = variables[jp_left_start:jp_left_start + params_per_junction]
                constrained_left = _compute_junction_cps_after(
                    left_junction_pt, left_params, mode, knots, n_ctrl, degree
                )

                # Right junction constrained CPs (before junction)
                jp_right_start = junction_var_start + right_junction_idx * params_per_junction
                right_params = variables[jp_right_start:jp_right_start + params_per_junction]
                constrained_right = _compute_junction_cps_before(
                    right_junction_pt, right_params, mode, knots, n_ctrl, degree
                )

                ctrl_pts = [left_junction_pt] + constrained_left + free_pts + constrained_right + [right_junction_pt]

        result.append(ctrl_pts)

    return result


def _compute_junction_cps_before(
    junction_pt: Point2D,
    params: np.ndarray,
    mode: ContinuityMode,
    knots: np.ndarray,
    n_ctrl: int,
    degree: int,
) -> PointList:
    """
    Compute constrained control points BEFORE a junction (approaching it).
    
    These are the control points on the incoming curve side, 
    ordered from free region toward the junction.
    
    Uses correct B-spline derivative formulas that account for knot spacing.
    Uses parametric (C^n) continuity where derivatives match exactly.
    
    Args:
        junction_pt: The junction point (x, y)
        params: Junction parameters from optimizer [D1x, D1y, ...]
        mode: Continuity mode (G1/G2/G3)
        knots: B-spline knot vector
        n_ctrl: Number of control points
        degree: B-spline degree
    """
    if mode == ContinuityMode.G1:
        # Parametric G1: params = [D1x, D1y]
        d1x, d1y = params
        D1 = (d1x, d1y)
        return _compute_junction_cps_from_derivatives(
            junction_pt, D1, None, None, knots, n_ctrl, degree, mode, is_before=True
        )

    elif mode == ContinuityMode.G2:
        # Parametric G2: params = [D1x, D1y, D2x, D2y]
        d1x, d1y, d2x, d2y = params
        D1 = (d1x, d1y)
        D2 = (d2x, d2y)
        return _compute_junction_cps_from_derivatives(
            junction_pt, D1, D2, None, knots, n_ctrl, degree, mode, is_before=True
        )

    else:  # G3
        # Parametric G3: params = [D1x, D1y, D2x, D2y, D3x, D3y]
        d1x, d1y, d2x, d2y, d3x, d3y = params
        D1 = (d1x, d1y)
        D2 = (d2x, d2y)
        D3 = (d3x, d3y)
        return _compute_junction_cps_from_derivatives(
            junction_pt, D1, D2, D3, knots, n_ctrl, degree, mode, is_before=True
        )


def _compute_junction_cps_after(
    junction_pt: Point2D,
    params: np.ndarray,
    mode: ContinuityMode,
    knots: np.ndarray,
    n_ctrl: int,
    degree: int,
) -> PointList:
    """
    Compute constrained control points AFTER a junction (leaving it).
    
    These are the control points on the outgoing curve side,
    ordered from junction toward free region.
    
    Uses parametric (C^n) continuity where derivatives match exactly on both sides.
    
    Uses correct B-spline derivative formulas that account for knot spacing.
    
    Args:
        junction_pt: The junction point (x, y)
        params: Junction parameters from optimizer [D1x, D1y, ...]
        mode: Continuity mode (G1/G2/G3)
        knots: B-spline knot vector
        n_ctrl: Number of control points
        degree: B-spline degree
    """
    if mode == ContinuityMode.G1:
        # Parametric G1: params = [D1x, D1y] - same derivative both sides
        d1x, d1y = params
        D1 = (d1x, d1y)
        return _compute_junction_cps_from_derivatives(
            junction_pt, D1, None, None, knots, n_ctrl, degree, mode, is_before=False
        )

    elif mode == ContinuityMode.G2:
        # Parametric G2: params = [D1x, D1y, D2x, D2y] - same derivatives both sides
        d1x, d1y, d2x, d2y = params
        D1_after = (d1x, d1y)
        D2_after = (d2x, d2y)
        return _compute_junction_cps_from_derivatives(
            junction_pt, D1_after, D2_after, None, knots, n_ctrl, degree, mode, is_before=False
        )

    else:  # G3
        # Parametric G3: params = [D1x, D1y, D2x, D2y, D3x, D3y] - same derivatives both sides
        d1x, d1y, d2x, d2y, d3x, d3y = params
        D1_after = (d1x, d1y)
        D2_after = (d2x, d2y)
        D3_after = (d3x, d3y)
        return _compute_junction_cps_from_derivatives(
            junction_pt, D1_after, D2_after, D3_after, knots, n_ctrl, degree, mode, is_before=False
        )


# =============================================================================
# Initial Guess Generation
# =============================================================================

def generate_initial_guess(
    config: SplineConfig,
    sections: list[PointList],
    t_params: list[list[float]],
    bounds: Sequence[Bounds],
    optimize_le_position: bool = False,
) -> np.ndarray:
    """
    Generate a data-driven initial guess for n-section optimization.
    
    Args:
        config: Spline configuration
        sections: List of n section point lists
        t_params: Arc-length parameters for each section
        bounds: Variable bounds for clamping
        optimize_le_position: If True, append every junction's (x, y) (n>=2 only)
        
    Returns:
        Initial guess as a numpy array
    """
    degree = config.degree
    n_ctrl = config.num_control_points
    mode = config.continuity
    num_sections = len(sections)

    # Knot scaling factors
    delta_u = 1.0 / (n_ctrl - degree)
    lambda_end = delta_u / degree if degree > 0 else 0.0
    alpha_2 = (delta_u ** 2) / (degree * (degree - 1)) if degree > 1 else 0.0

    def flatten_points(points: PointList) -> list[float]:
        result = []
        for x, y in points:
            result.extend([float(x), float(y)])
        return result

    def get_vector(point_a: Point2D, point_b: Point2D) -> np.ndarray:
        return np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])

    def safe_normalize(v: np.ndarray) -> np.ndarray:
        mag = float(np.hypot(v[0], v[1]))
        return v / mag if mag > 1e-12 else np.array([1.0, 0.0])

    # Number of constrained CPs at each junction side
    n_constrained = mode.constrained_control_points

    def get_n_free_for_section(section_idx: int) -> int:
        if num_sections == 1:
            return n_ctrl - 2
        elif section_idx == 0 or section_idx == num_sections - 1:
            return max(0, n_ctrl - 1 - n_constrained)
        else:
            return max(0, n_ctrl - 2 * n_constrained)

    x0 = []

    # Generate free control points for each section
    for section_idx in range(num_sections):
        section = sections[section_idx]
        t_sec = t_params[section_idx]
        n_free = get_n_free_for_section(section_idx)

        if n_free > 0:
            # Sample interior points along section polyline
            t_interior = [i / (n_ctrl - 1) for i in range(1, n_free + 1)]
            free_pts = interpolate_along_polyline(section, t_sec, t_interior)
            x0.extend(flatten_points(free_pts))

    # Generate junction constraint parameters for n-1 junctions
    if num_sections >= 2:
        for junction_idx in range(num_sections - 1):
            # Junction is at the end of section[junction_idx]
            section_before = sections[junction_idx]
            junction_pt = np.array(section_before[-1])

            if mode == ContinuityMode.G0:
                # No extra parameters for G0
                pass
            elif mode == ContinuityMode.G1:
                # Estimate tangent direction at junction
                v_before = get_vector(section_before[-2], junction_pt) if len(section_before) >= 2 else np.array([1.0, 0.0])

                # Parametric G1: D1x, D1y (same derivative both sides)
                D1 = v_before / lambda_end if lambda_end > 1e-12 else v_before
                x0.extend([D1[0], D1[1]])

            elif mode == ContinuityMode.G2:
                # Estimate derivatives
                v_before = get_vector(section_before[-2], junction_pt) if len(section_before) >= 2 else np.array([1.0, 0.0])
                D1 = v_before / lambda_end if lambda_end > 1e-12 else v_before

                if alpha_2 > 1e-12 and len(section_before) >= 3:
                    A = np.array(section_before[-3])
                    D2 = (A - junction_pt + 2 * lambda_end * D1) / alpha_2
                else:
                    D2 = np.array([0.0, 0.0])

                # Parametric G2: D1x, D1y, D2x, D2y
                x0.extend([D1[0], D1[1], D2[0], D2[1]])

            else:  # G3
                v_before = get_vector(section_before[-2], junction_pt) if len(section_before) >= 2 else np.array([1.0, 0.0])
                D1 = v_before / lambda_end if lambda_end > 1e-12 else v_before

                if alpha_2 > 1e-12 and len(section_before) >= 3:
                    A = np.array(section_before[-3])
                    D2 = (A - junction_pt + 2 * lambda_end * D1) / alpha_2
                else:
                    D2 = np.array([0.0, 0.0])

                D3 = np.array([0.0, 0.0])  # Default
                # Parametric G3: D1, D2, D3
                x0.extend([D1[0], D1[1], D2[0], D2[1], D3[0], D3[1]])

    # Junction positions start at the split points
    if optimize_le_position and num_sections >= 2:
        for section in sections[:-1]:
            x0.extend([float(section[-1][0]), float(section[-1][1])])

    return _clamp_to_bounds(x0, bounds)


def initial_guess_from_fit(
    config: SplineConfig,
    section_control_points: list[PointList],
    bounds: Sequence[Bounds],
    optimize_le_position: bool = False,
) -> np.ndarray:
    """
    The variables that reproduce a fitted curve, the inverse of construct_control_points.
    
    The fit must have the configured degree and count in every section, on open
    uniform knots, with the configured continuity at its junctions (as the
    constrained least-squares fit has). Free control points are copied; each
    junction's parameters are the curve's derivatives there, averaged over both
    sides.
    
    Args:
        config: Spline configuration
        section_control_points: Control points of the fit, one list per section
        bounds: Variable bounds for clamping
        optimize_le_position: If True, append every junction's (x, y) (n>=2 only)
        
    Returns:
        Initial guess as a numpy array
    """
    degree = config.degree
    n_ctrl = config.num_control_points
    mode = config.continuity
    num_sections = len(section_control_points)
    n_constrained = mode.constrained_control_points
    if any(len(pts) != n_ctrl for pts in section_control_points):
        raise ValueError(f"Every section needs {n_ctrl} control points")

    x0: list[float] = []
    for section_idx, pts in enumerate(section_control_points):
        if num_sections == 1:
            free_pts = pts[1:-1]
        elif section_idx == 0:
            free_pts = pts[1:max(1, n_ctrl - n_constrained)]
        else:
            # After the junction point and its constrained neighbours
            end = n_ctrl - 1 if section_idx == num_sections - 1 else n_ctrl - n_constrained
            free_pts = pts[n_constrained:max(n_constrained, end)]
        for x, y in free_pts:
            x0.extend([float(x), float(y)])

    order = n_constrained - 1
    if num_sections >= 2 and order > 0:
        names = (('dx', 'dy'), ('d2x', 'd2y'), ('d3x', 'd3y'))
        curves = [BSplineCurve2D(degree, pts) for pts in section_control_points]
        for before, after in zip(curves[:-1], curves[1:], strict=False):
            d_before = before.evaluate_derivatives(np.array([1.0]), max_order=order, min_order=1)
            d_after = after.evaluate_derivatives(np.array([0.0]), max_order=order, min_order=1)
            for name_x, name_y in names[:order]:
                x0.extend([0.5 * float(d_before[name_x][0] + d_after[name_x][0]),
                           0.5 * float(d_before[name_y][0] + d_after[name_y][0])])

    if optimize_le_position and num_sections >= 2:
        for pts in section_control_points[:-1]:
            x0.extend([float(pts[-1][0]), float(pts[-1][1])])

    return _clamp_to_bounds(x0, bounds)


def _clamp_to_bounds(x0: Sequence[float], bounds: Sequence[Bounds]) -> np.ndarray:
    """The variables clamped to their bounds (open where a bound is None)."""
    x0_clamped = []
    for i, val in enumerate(x0):
        if i < len(bounds):
            lo, hi = bounds[i]
            lo = lo if lo is not None else float('-inf')
            hi = hi if hi is not None else float('inf')
            x0_clamped.append(min(max(float(val), float(lo)), float(hi)))
        else:
            x0_clamped.append(float(val))

    return np.array(x0_clamped, dtype=np.float64)


def _compute_junction_cps_from_derivatives(
    junction_pt: Point2D,
    D1: tuple[float, float],
    D2: tuple[float, float] | None,
    D3: tuple[float, float] | None,
    knots: np.ndarray,
    n_ctrl: int,
    degree: int,
    mode: ContinuityMode,
    is_before: bool,
) -> PointList:
    """
    Compute constrained control points from parametric derivatives.
    
    This function correctly inverts the B-spline derivative formulas to find
    control points that produce the specified derivatives at the junction.
    
    For B-splines with clamped knot vectors, the relationship between
    control points and derivatives depends on the knot spacing:
    
        d¹C/du = coef · [P_boundary, P_adjacent, ...]
        
    We solve for the unknown control points given the known junction point
    and derivative values.
    
    Args:
        junction_pt: The junction point (x, y)
        D1: First derivative (dx/du, dy/du) at junction
        D2: Second derivative (if mode >= G2)
        D3: Third derivative (if mode == G3)
        knots: Knot vector
        n_ctrl: Number of control points
        degree: Spline degree
        mode: Continuity mode
        is_before: True if computing CPs approaching junction (u→1),
                   False if leaving junction (u→0)
    
    Returns:
        List of control points (excluding junction point itself) in order
        from junction toward interior of curve
    """
    jx, jy = junction_pt

    if mode == ContinuityMode.G1:
        # Only need first derivative
        coef_end_c1, coef_start_c1 = _compute_derivative_constraint_coeffs(
            knots, n_ctrl, degree, order=1
        )

        if is_before:
            # At u=1: d1 = coef_end[0]*P[-2] + coef_end[1]*P[-1]
            # P[-1] is junction point, solve for P[-2]:
            # P[-2] = (d1 - coef_end[1]*P[-1]) / coef_end[0]
            coef = coef_end_c1
            p_adjacent = (
                (D1[0] - coef[1] * jx) / coef[0],
                (D1[1] - coef[1] * jy) / coef[0]
            )
            return [p_adjacent]
        else:
            # At u=0: d1 = coef_start[0]*P[0] + coef_start[1]*P[1]
            # P[0] is junction point, solve for P[1]:
            # P[1] = (d1 - coef_start[0]*P[0]) / coef_start[1]
            coef = coef_start_c1
            p_adjacent = (
                (D1[0] - coef[0] * jx) / coef[1],
                (D1[1] - coef[0] * jy) / coef[1]
            )
            return [p_adjacent]

    elif mode == ContinuityMode.G2:
        # Need first and second derivatives
        coef_end_c1, coef_start_c1 = _compute_derivative_constraint_coeffs(
            knots, n_ctrl, degree, order=1
        )
        coef_end_c2, coef_start_c2 = _compute_derivative_constraint_coeffs(
            knots, n_ctrl, degree, order=2
        )

        if is_before:
            # At u=1: P[-1] is junction (known)
            # d1 = c1[0]*P[-2] + c1[1]*P[-1]  => P[-2] = (d1 - c1[1]*P[-1]) / c1[0]
            # d2 = c2[0]*P[-3] + c2[1]*P[-2] + c2[2]*P[-1]  => P[-3]
            c1 = coef_end_c1
            c2 = coef_end_c2

            p_nm1_x = (D1[0] - c1[1] * jx) / c1[0]
            p_nm1_y = (D1[1] - c1[1] * jy) / c1[0]

            p_nm2_x = (D2[0] - c2[2] * jx - c2[1] * p_nm1_x) / c2[0]
            p_nm2_y = (D2[1] - c2[2] * jy - c2[1] * p_nm1_y) / c2[0]

            return [(p_nm2_x, p_nm2_y), (p_nm1_x, p_nm1_y)]
        else:
            # At u=0: P[0] is junction (known)
            # d1 = c1[0]*P[0] + c1[1]*P[1]  => P[1]
            # d2 = c2[0]*P[0] + c2[1]*P[1] + c2[2]*P[2]  => P[2]
            c1 = coef_start_c1
            c2 = coef_start_c2

            p_1_x = (D1[0] - c1[0] * jx) / c1[1]
            p_1_y = (D1[1] - c1[0] * jy) / c1[1]

            p_2_x = (D2[0] - c2[0] * jx - c2[1] * p_1_x) / c2[2]
            p_2_y = (D2[1] - c2[0] * jy - c2[1] * p_1_y) / c2[2]

            return [(p_1_x, p_1_y), (p_2_x, p_2_y)]

    else:  # G3
        # Need first, second, and third derivatives
        coef_end_c1, coef_start_c1 = _compute_derivative_constraint_coeffs(
            knots, n_ctrl, degree, order=1
        )
        coef_end_c2, coef_start_c2 = _compute_derivative_constraint_coeffs(
            knots, n_ctrl, degree, order=2
        )
        coef_end_c3, coef_start_c3 = _compute_derivative_constraint_coeffs(
            knots, n_ctrl, degree, order=3
        )

        if is_before:
            # At u=1: P[-1] is junction (known)
            c1 = coef_end_c1
            c2 = coef_end_c2
            c3 = coef_end_c3

            p_nm1_x = (D1[0] - c1[1] * jx) / c1[0]
            p_nm1_y = (D1[1] - c1[1] * jy) / c1[0]

            p_nm2_x = (D2[0] - c2[2] * jx - c2[1] * p_nm1_x) / c2[0]
            p_nm2_y = (D2[1] - c2[2] * jy - c2[1] * p_nm1_y) / c2[0]

            p_nm3_x = (D3[0] - c3[3] * jx - c3[2] * p_nm1_x - c3[1] * p_nm2_x) / c3[0]
            p_nm3_y = (D3[1] - c3[3] * jy - c3[2] * p_nm1_y - c3[1] * p_nm2_y) / c3[0]

            return [(p_nm3_x, p_nm3_y), (p_nm2_x, p_nm2_y), (p_nm1_x, p_nm1_y)]
        else:
            # At u=0: P[0] is junction (known)
            c1 = coef_start_c1
            c2 = coef_start_c2
            c3 = coef_start_c3

            p_1_x = (D1[0] - c1[0] * jx) / c1[1]
            p_1_y = (D1[1] - c1[0] * jy) / c1[1]

            p_2_x = (D2[0] - c2[0] * jx - c2[1] * p_1_x) / c2[2]
            p_2_y = (D2[1] - c2[0] * jy - c2[1] * p_1_y) / c2[2]

            p_3_x = (D3[0] - c3[0] * jx - c3[1] * p_1_x - c3[2] * p_2_x) / c3[3]
            p_3_y = (D3[1] - c3[0] * jy - c3[1] * p_1_y - c3[2] * p_2_y) / c3[3]

            return [(p_1_x, p_1_y), (p_2_x, p_2_y), (p_3_x, p_3_y)]
