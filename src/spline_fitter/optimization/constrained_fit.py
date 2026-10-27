"""
Constrained Least-Squares Fitting
=================================

Algebraic (non-iterative) spline fitting with fixed trailing-edge
endpoints and continuity constraints at the junctions.
"""

from __future__ import annotations

from typing import (
    List, Tuple, Any, Dict
)
import numpy as np

from .models import (
    PointList, AirfoilData, SplineConfig, OptimizationConfig,
    ContinuityMode,
)
from .constraints import compute_derivative_constraint_coeffs
from ..spline.spline import create_open_uniform_knots
from ..geometry import compute_arc_length_parameters


# Constrained Least-Squares Spline Fitting
# =============================================================================

def run_constrained_lsq_fit(
    config: SplineConfig,
    opt_config: OptimizationConfig,
    airfoil: AirfoilData,
) -> Tuple[List[PointList], float]:
    """
    Fit splines using constrained least-squares.
    
    Solves for control points that:
    1. Minimize least-squares fitting error
    2. Fix TE endpoints (first CP of first section, last CP of last section)
    3. Satisfy continuity constraints at junctions (G0/G1/G2)
    
    Junction positions can be either:
    - Fixed to data points (optimize_le_position=False)
    - Free to optimize (optimize_le_position=True)
    
    Args:
        config: Spline configuration
        opt_config: Optimization configuration  
        airfoil: Airfoil data with sections
        
    Returns:
        Tuple of (section_control_points, final_error)
    """
    from scipy.interpolate import BSpline as ScipyBSpline
    
    sections = [list(sec) for sec in airfoil.sections]
    degree = config.degree
    n_ctrl = config.num_control_points
    mode = config.continuity
    num_sections = len(sections)
    free_junctions = opt_config.optimize_le_position
    
    # Create knot vector (same for all sections)
    knots = create_open_uniform_knots(degree, n_ctrl)
    
    # Precompute basis functions for each section
    section_data = []
    for section_idx, section in enumerate(sections):
        pts = np.array(section, dtype=np.float64)
        n_pts = len(pts)
        
        # Arc-length parameterization
        t_params = compute_arc_length_parameters(section)
        u = np.array(t_params, dtype=np.float64)
        
        # Build basis matrix N[j,i] = N_i(u_j)
        N = np.zeros((n_pts, n_ctrl))
        for i in range(n_ctrl):
            c = np.zeros(n_ctrl)
            c[i] = 1.0
            basis = ScipyBSpline(knots, c, degree)
            N[:, i] = basis(u)
        
        section_data.append({
            'pts': pts,
            'u': u,
            'N': N,
            'n_pts': n_pts,
        })
    
    # Solve based on number of sections
    if num_sections == 1:
        # Single section: just fix endpoints
        section_control_points = _solve_single_section(
            section_data[0], sections[0], n_ctrl
        )
    else:
        # Multiple sections: coupled system with continuity
        section_control_points = _solve_coupled_sections_lsq(
            section_data, sections, config, knots, free_junctions
        )
    
    # Compute final error
    total_error = 0.0
    total_points = 0
    
    for section_idx, (section, ctrl_pts) in enumerate(zip(sections, section_control_points)):
        pts = np.array(section, dtype=np.float64)
        N = section_data[section_idx]['N']
        
        # Predicted points
        ctrl_arr = np.array(ctrl_pts)
        pred_x = N @ ctrl_arr[:, 0]
        pred_y = N @ ctrl_arr[:, 1]
        
        err = np.hypot(pred_x - pts[:, 0], pred_y - pts[:, 1])
        total_error += float(np.sum(err))
        total_points += len(pts)
    
    final_error = total_error / total_points if total_points > 0 else 0.0
    
    return section_control_points, final_error


def _solve_single_section(
    data: Dict[str, Any],
    section: PointList,
    n_ctrl: int,
) -> List[PointList]:
    """Solve single section with fixed endpoints."""
    from scipy.linalg import solve
    
    pts = data['pts']
    N = data['N']
    
    # Endpoint constraints: P[0] = pts[0], P[-1] = pts[-1]
    C = np.zeros((2, n_ctrl))
    C[0, 0] = 1.0
    C[1, -1] = 1.0
    
    c_vals_x = np.array([pts[0, 0], pts[-1, 0]])
    c_vals_y = np.array([pts[0, 1], pts[-1, 1]])
    
    # Build KKT system
    NtN = N.T @ N + 1e-10 * np.eye(n_ctrl)
    n_c = 2
    
    KKT = np.zeros((n_ctrl + n_c, n_ctrl + n_c))
    KKT[:n_ctrl, :n_ctrl] = NtN
    KKT[:n_ctrl, n_ctrl:] = C.T
    KKT[n_ctrl:, :n_ctrl] = C
    
    rhs_x = np.concatenate([N.T @ pts[:, 0], c_vals_x])
    rhs_y = np.concatenate([N.T @ pts[:, 1], c_vals_y])
    
    sol_x = solve(KKT, rhs_x)
    sol_y = solve(KKT, rhs_y)
    
    ctrl_x = sol_x[:n_ctrl]
    ctrl_y = sol_y[:n_ctrl]
    
    ctrl_pts = [(float(ctrl_x[i]), float(ctrl_y[i])) for i in range(n_ctrl)]
    return [ctrl_pts]


# Use shared implementation from constraints module
_compute_derivative_constraint_coeffs = compute_derivative_constraint_coeffs


def _solve_coupled_sections_lsq(
    section_data: List[Dict[str, Any]],
    sections: List[PointList],
    config: SplineConfig,
    knots: np.ndarray,
    free_junctions: bool,
) -> List[PointList]:
    """
    Solve all sections as a coupled system with continuity constraints.
    
    Uses parametric (C^n) continuity where exact derivative matching is enforced:
    - C1: d¹C/du¹ matches exactly on both sides
    - C2: d²C/du² matches exactly on both sides
    - C3: d³C/du³ matches exactly on both sides
    
    Args:
        section_data: Precomputed basis matrices and data for each section
        sections: Original point data for each section
        config: Spline configuration
        knots: Knot vector
        free_junctions: If True, junction positions are optimized; if False, fixed to data
    """
    from scipy.linalg import solve
    
    degree = config.degree
    n_ctrl = config.num_control_points
    mode = config.continuity
    num_sections = len(sections)
    
    # Total unknowns: n_ctrl per section
    total_ctrl = n_ctrl * num_sections
    
    # Build block-diagonal basis matrix
    total_pts = sum(d['n_pts'] for d in section_data)
    N_full = np.zeros((total_pts, total_ctrl))
    
    pts_offset = 0
    for section_idx, data in enumerate(section_data):
        n_pts = data['n_pts']
        ctrl_offset = section_idx * n_ctrl
        N_full[pts_offset:pts_offset + n_pts, ctrl_offset:ctrl_offset + n_ctrl] = data['N']
        pts_offset += n_pts
    
    # Collect all data points
    all_pts = np.vstack([d['pts'] for d in section_data])
    
    # Build constraint equations
    C_rows = []
    c_x = []
    c_y = []
    
    # ----- Constraint: First section P[0] = TE start (always fixed) -----
    row = np.zeros(total_ctrl)
    row[0] = 1.0
    C_rows.append(row)
    c_x.append(sections[0][0][0])
    c_y.append(sections[0][0][1])
    
    # ----- Constraint: Last section P[-1] = TE end (always fixed) -----
    row = np.zeros(total_ctrl)
    row[-1] = 1.0
    C_rows.append(row)
    c_x.append(sections[-1][-1][0])
    c_y.append(sections[-1][-1][1])
    
    # ----- Junction constraints -----
    for j in range(num_sections - 1):
        junction_pt = np.array(sections[j][-1])  # Same as sections[j+1][0]
        
        ctrl_offset_j = j * n_ctrl
        ctrl_offset_j1 = (j + 1) * n_ctrl
        
        # Index of last CP in section j and first CP in section j+1
        idx_end_j = ctrl_offset_j + n_ctrl - 1
        idx_start_j1 = ctrl_offset_j1
        
        # Position continuity: P_j[-1] = P_{j+1}[0]
        row = np.zeros(total_ctrl)
        row[idx_end_j] = 1.0
        row[idx_start_j1] = -1.0
        C_rows.append(row)
        c_x.append(0.0)
        c_y.append(0.0)
        
        if not free_junctions:
            # Fix junction to data point
            row = np.zeros(total_ctrl)
            row[idx_end_j] = 1.0
            C_rows.append(row)
            c_x.append(junction_pt[0])
            c_y.append(junction_pt[1])
        
        # =====================================================================
        # TANGENT CONTINUITY (G1+)
        # =====================================================================
        # Parametric (C1): exact derivative matching - d¹C/du¹ equal on both sides
        # =====================================================================
        
        if mode in (ContinuityMode.G1, ContinuityMode.G2, ContinuityMode.G3):
            idx_nm1_j = ctrl_offset_j + n_ctrl - 2
            idx_1_j1 = ctrl_offset_j1 + 1
            
            # Parametric C1: exact derivative matching
            coef_end_c1, coef_start_c1 = _compute_derivative_constraint_coeffs(
                knots, n_ctrl, degree, order=1
            )
            row = np.zeros(total_ctrl)
            row[idx_nm1_j] = coef_end_c1[0]  # P[-2] coef
            row[idx_end_j] = coef_end_c1[1]   # P[-1] coef
            row[idx_start_j1] = -coef_start_c1[0]  # -P[0] coef
            row[idx_1_j1] = -coef_start_c1[1]      # -P[1] coef
            C_rows.append(row)
            c_x.append(0.0)
            c_y.append(0.0)
        
        # =====================================================================
        # CURVATURE CONTINUITY (G2+) 
        # =====================================================================
        # Parametric (C2): d²C/du² matches exactly on both sides
        # =====================================================================
        
        if mode in (ContinuityMode.G2, ContinuityMode.G3):
            if n_ctrl >= 3:
                idx_nm2_j = ctrl_offset_j + n_ctrl - 3
                idx_2_j1 = ctrl_offset_j1 + 2
                
                # C2 constraint using proper coefficients  
                coef_end_c2, coef_start_c2 = _compute_derivative_constraint_coeffs(
                    knots, n_ctrl, degree, order=2
                )
                row = np.zeros(total_ctrl)
                row[idx_nm2_j] = coef_end_c2[0]   # P[-3] coef
                row[idx_nm1_j] = coef_end_c2[1]   # P[-2] coef
                row[idx_end_j] = coef_end_c2[2]   # P[-1] coef
                row[idx_start_j1] = -coef_start_c2[0]  # -P[0] coef
                row[idx_1_j1] = -coef_start_c2[1]      # -P[1] coef
                row[idx_2_j1] = -coef_start_c2[2]      # -P[2] coef
                C_rows.append(row)
                c_x.append(0.0)
                c_y.append(0.0)
        
        # =====================================================================
        # CURVATURE RATE CONTINUITY (G3)
        # =====================================================================
        
        if mode == ContinuityMode.G3:
            if n_ctrl >= 4:
                idx_nm3_j = ctrl_offset_j + n_ctrl - 4
                idx_3_j1 = ctrl_offset_j1 + 3
                
                # C3 constraint using proper coefficients
                coef_end_c3, coef_start_c3 = _compute_derivative_constraint_coeffs(
                    knots, n_ctrl, degree, order=3
                )
                row = np.zeros(total_ctrl)
                row[idx_nm3_j] = coef_end_c3[0]   # P[-4] coef
                row[idx_nm2_j] = coef_end_c3[1]   # P[-3] coef
                row[idx_nm1_j] = coef_end_c3[2]   # P[-2] coef
                row[idx_end_j] = coef_end_c3[3]   # P[-1] coef
                row[idx_start_j1] = -coef_start_c3[0]  # -P[0] coef
                row[idx_1_j1] = -coef_start_c3[1]      # -P[1] coef
                row[idx_2_j1] = -coef_start_c3[2]      # -P[2] coef
                row[idx_3_j1] = -coef_start_c3[3]      # -P[3] coef
                C_rows.append(row)
                c_x.append(0.0)
                c_y.append(0.0)
    
    # Convert to arrays
    C = np.array(C_rows)
    c_x = np.array(c_x)
    c_y = np.array(c_y)
    n_c = len(C_rows)
    
    # Build and solve KKT system
    NtN = N_full.T @ N_full + 1e-10 * np.eye(total_ctrl)
    
    KKT = np.zeros((total_ctrl + n_c, total_ctrl + n_c))
    KKT[:total_ctrl, :total_ctrl] = NtN
    KKT[:total_ctrl, total_ctrl:] = C.T
    KKT[total_ctrl:, :total_ctrl] = C
    
    rhs_x = np.concatenate([N_full.T @ all_pts[:, 0], c_x])
    rhs_y = np.concatenate([N_full.T @ all_pts[:, 1], c_y])
    
    sol_x = solve(KKT, rhs_x)
    sol_y = solve(KKT, rhs_y)
    
    # Extract control points per section
    section_control_points = []
    for section_idx in range(num_sections):
        ctrl_offset = section_idx * n_ctrl
        ctrl_x = sol_x[ctrl_offset:ctrl_offset + n_ctrl]
        ctrl_y = sol_y[ctrl_offset:ctrl_offset + n_ctrl]
        ctrl_pts = [(float(ctrl_x[i]), float(ctrl_y[i])) for i in range(n_ctrl)]
        section_control_points.append(ctrl_pts)
    
    return section_control_points
