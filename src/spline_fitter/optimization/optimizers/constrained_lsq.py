"""
Constrained Least-Squares Optimizer
===================================

Algebraic (non-iterative) spline fitting using constrained LSQ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import BSpline as ScipyBSpline
from scipy.linalg import solve

from ..constraints import compute_derivative_constraint_coeffs
from .base import AlgebraicOptimizer, RawOptimResult

if TYPE_CHECKING:
    from ..models import (
        AirfoilData,
        OptimizationConfig,
        PointList,
        SplineConfig,
    )


class ConstrainedLSQOptimizer(AlgebraicOptimizer):
    """
    Constrained Least-Squares spline fitting.
    
    Solves for control points algebraically (non-iterative) that:
    1. Minimize least-squares fitting error
    2. Fix TE endpoints (first CP of first section, last CP of last section)
    3. Satisfy continuity constraints at junctions (G0/G1/G2)
    
    Junction positions can be either:
    - Fixed to data points (optimize_le_position=False)
    - Free to optimize (optimize_le_position=True)
    
    Data points get their parameters from opt_config.custom_parameters, or
    else by opt_config.parameterization (centripetal or arc length).
    """

    @property
    def name(self) -> str:
        return "Direct LSQ"

    def fit(
        self,
        config: SplineConfig,
        opt_config: OptimizationConfig,
        airfoil: AirfoilData,
    ) -> RawOptimResult:
        """
        Fit splines using constrained least-squares.
        
        Args:
            config: Spline configuration
            opt_config: Optimization configuration
            airfoil: Airfoil data with sections
            
        Returns:
            RawOptimResult with computed control points
        """
        from ...geometry import centripetal_parametrize, compute_arc_length_parameters
        from ...spline.spline import create_open_uniform_knots

        sections = [list(sec) for sec in airfoil.sections]
        degree = config.degree
        n_ctrl = config.num_control_points
        num_sections = len(sections)
        free_junctions = opt_config.optimize_le_position

        # Create knot vector (same for all sections)
        knots = create_open_uniform_knots(degree, n_ctrl)

        # Precompute basis functions for each section
        section_data = []
        for section_idx, section in enumerate(sections):
            pts = np.array(section, dtype=np.float64)
            n_pts = len(pts)

            # Parameterization: custom, centripetal, or arc-length
            if opt_config.custom_parameters is not None:
                if section_idx >= len(opt_config.custom_parameters):
                    raise ValueError(
                        f"custom_parameters has {len(opt_config.custom_parameters)} sections, "
                        f"but section {section_idx} requested"
                    )
                custom_t = opt_config.custom_parameters[section_idx]
                if len(custom_t) != n_pts:
                    raise ValueError(
                        f"custom_parameters[{section_idx}] has {len(custom_t)} values, "
                        f"but section has {n_pts} points"
                    )
                u = np.array(custom_t, dtype=np.float64)
            elif opt_config.parameterization == "centripetal":
                t_params = centripetal_parametrize(section, alpha=0.5)
                u = np.array(t_params, dtype=np.float64)
            else:
                t_params = compute_arc_length_parameters(section)
                u = np.array(t_params, dtype=np.float64)

            # Build basis matrix N[j,i] = N_i(u_j)
            N = ScipyBSpline.design_matrix(u, knots, degree).toarray()

            section_data.append({
                'pts': pts,
                'u': u,
                'N': N,
                'n_pts': n_pts,
            })

        # Solve based on number of sections
        if num_sections == 1:
            section_control_points = self._solve_single_section(
                section_data[0], sections[0], n_ctrl
            )
        else:
            section_control_points = self._solve_coupled_sections(
                section_data, sections, config, knots, free_junctions
            )

        # Compute final error
        total_error = 0.0
        total_points = 0

        for section_idx, (section, ctrl_pts) in enumerate(zip(sections, section_control_points, strict=False)):
            pts = np.array(section, dtype=np.float64)
            N = section_data[section_idx]['N']

            ctrl_arr = np.array(ctrl_pts)
            pred_x = N @ ctrl_arr[:, 0]
            pred_y = N @ ctrl_arr[:, 1]

            err = np.hypot(pred_x - pts[:, 0], pred_y - pts[:, 1])
            total_error += float(np.sum(err))
            total_points += len(pts)

        final_error = total_error / total_points if total_points > 0 else 0.0

        # Flatten control points to variable vector for logging compatibility
        flat_vars = []
        for ctrl_pts in section_control_points:
            for x, y in ctrl_pts:
                flat_vars.extend([x, y])

        return RawOptimResult(
            x=np.array(flat_vars, dtype=np.float64),
            fun=final_error,
            nit=1,  # Single "iteration" for algebraic solve
            section_control_points=section_control_points,
        )

    def _solve_single_section(
        self,
        data: dict[str, Any],
        section: PointList,
        n_ctrl: int,
    ) -> list[PointList]:
        """Solve single section with fixed endpoints."""
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

    def _solve_coupled_sections(
        self,
        section_data: list[dict[str, Any]],
        sections: list[PointList],
        config: SplineConfig,
        knots: np.ndarray,
        free_junctions: bool,
    ) -> list[PointList]:
        """
        Solve all sections as a coupled system with continuity constraints.
        
        Uses parametric (C^n) continuity where derivatives match exactly.
        """
        from ..models import ContinuityMode

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

        # First section P[0] = TE start (always fixed)
        row = np.zeros(total_ctrl)
        row[0] = 1.0
        C_rows.append(row)
        c_x.append(sections[0][0][0])
        c_y.append(sections[0][0][1])

        # Last section P[-1] = TE end (always fixed)
        row = np.zeros(total_ctrl)
        row[-1] = 1.0
        C_rows.append(row)
        c_x.append(sections[-1][-1][0])
        c_y.append(sections[-1][-1][1])

        # Junction constraints
        coef_end_c1 = coef_start_c1 = None
        if mode in (ContinuityMode.G1, ContinuityMode.G2, ContinuityMode.G3):
            coef_end_c1, coef_start_c1 = compute_derivative_constraint_coeffs(knots, n_ctrl, degree, 1)

        for j in range(num_sections - 1):
            junction_pt = np.array(sections[j][-1])

            ctrl_offset_j = j * n_ctrl
            ctrl_offset_j1 = (j + 1) * n_ctrl

            idx_end_j = ctrl_offset_j + n_ctrl - 1
            idx_start_j1 = ctrl_offset_j1
            idx_nm1_j = ctrl_offset_j + n_ctrl - 2
            idx_1_j1 = ctrl_offset_j1 + 1

            # Position continuity: P_j[-1] = P_{j+1}[0]
            row = np.zeros(total_ctrl)
            row[idx_end_j] = 1.0
            row[idx_start_j1] = -1.0
            C_rows.append(row)
            c_x.append(0.0)
            c_y.append(0.0)

            if not free_junctions:
                row = np.zeros(total_ctrl)
                row[idx_end_j] = 1.0
                C_rows.append(row)
                c_x.append(junction_pt[0])
                c_y.append(junction_pt[1])

            # Tangent continuity (G1+)
            if mode in (ContinuityMode.G1, ContinuityMode.G2, ContinuityMode.G3):
                row = np.zeros(total_ctrl)
                row[idx_nm1_j] = coef_end_c1[0]
                row[idx_end_j] = coef_end_c1[1]
                row[idx_start_j1] = -coef_start_c1[0]
                row[idx_1_j1] = -coef_start_c1[1]
                C_rows.append(row)
                c_x.append(0.0)
                c_y.append(0.0)

            # Curvature continuity (G2+)
            if mode in (ContinuityMode.G2, ContinuityMode.G3):
                if n_ctrl >= 3:
                    idx_nm2_j = ctrl_offset_j + n_ctrl - 3
                    idx_2_j1 = ctrl_offset_j1 + 2

                    coef_end_c2, coef_start_c2 = compute_derivative_constraint_coeffs(knots, n_ctrl, degree, 2)
                    row = np.zeros(total_ctrl)
                    row[idx_nm2_j] = coef_end_c2[0]
                    row[idx_nm1_j] = coef_end_c2[1]
                    row[idx_end_j] = coef_end_c2[2]
                    row[idx_start_j1] = -coef_start_c2[0]
                    row[idx_1_j1] = -coef_start_c2[1]
                    row[idx_2_j1] = -coef_start_c2[2]
                    C_rows.append(row)
                    c_x.append(0.0)
                    c_y.append(0.0)

            # Curvature rate continuity (G3)
            if mode == ContinuityMode.G3:
                if n_ctrl >= 4:
                    idx_nm2_j = ctrl_offset_j + n_ctrl - 3
                    idx_nm3_j = ctrl_offset_j + n_ctrl - 4
                    idx_2_j1 = ctrl_offset_j1 + 2
                    idx_3_j1 = ctrl_offset_j1 + 3

                    coef_end_c3, coef_start_c3 = compute_derivative_constraint_coeffs(knots, n_ctrl, degree, 3)
                    row = np.zeros(total_ctrl)
                    row[idx_nm3_j] = coef_end_c3[0]
                    row[idx_nm2_j] = coef_end_c3[1]
                    row[idx_nm1_j] = coef_end_c3[2]
                    row[idx_end_j] = coef_end_c3[3]
                    row[idx_start_j1] = -coef_start_c3[0]
                    row[idx_1_j1] = -coef_start_c3[1]
                    row[idx_2_j1] = -coef_start_c3[2]
                    row[idx_3_j1] = -coef_start_c3[3]
                    C_rows.append(row)
                    c_x.append(0.0)
                    c_y.append(0.0)

        # Solve standard system
        C = np.array(C_rows)
        c_x_arr = np.array(c_x)
        c_y_arr = np.array(c_y)
        n_c = len(C_rows)

        NtN = N_full.T @ N_full + 1e-10 * np.eye(total_ctrl)

        KKT = np.zeros((total_ctrl + n_c, total_ctrl + n_c))
        KKT[:total_ctrl, :total_ctrl] = NtN
        KKT[:total_ctrl, total_ctrl:] = C.T
        KKT[total_ctrl:, :total_ctrl] = C

        rhs_x = np.concatenate([N_full.T @ all_pts[:, 0], c_x_arr])
        rhs_y = np.concatenate([N_full.T @ all_pts[:, 1], c_y_arr])

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
