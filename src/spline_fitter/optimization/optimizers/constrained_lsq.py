"""
Constrained Least-Squares Optimizer
===================================

Algebraic (non-iterative) spline fitting using constrained LSQ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    
    Solves for control points algebraically that:
    1. Minimize least-squares fitting error
    2. Fix TE endpoints (first CP of first section, last CP of last section)
    3. Satisfy continuity constraints at junctions (G0/G1/G2)
    
    Junction positions can be either:
    - Fixed to data points (optimize_le_position=False)
    - Free to optimize (optimize_le_position=True)
    
    The chain is fitted as a whole: every data point once, on the section its
    chain parameter lies on (bspline_varpro.chain). Points get their
    parameters from opt_config.custom_parameters, kept fixed (one linear
    solve), or else start from opt_config.parameterization (centripetal or
    arc length) within each section; after the linear solve the parameters
    and control points are refined together by Gauss-Newton
    (bspline_varpro.joint), so a point slides to its closest chain point and
    can cross a junction.
    """

    # Gauss-Newton iterations of the joint refinement
    MAX_ITERATIONS = 100

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
            RawOptimResult with computed control points and the points' chain
            parameters
        """
        from bspline_varpro.chain import (
            chain_data,
            chain_parameters,
            evaluate_chain,
            validate_chain_parameters,
        )
        from bspline_varpro.joint import JointProblem, solve_joint

        from ...geometry import centripetal_parametrize, compute_arc_length_parameters
        from ...spline.spline import create_open_uniform_knots

        sections = [list(sec) for sec in airfoil.sections]
        arrays = [np.array(sec, dtype=np.float64) for sec in sections]
        degree = config.degree
        n_ctrl = config.num_control_points
        num_sections = len(sections)
        knots = create_open_uniform_knots(degree, n_ctrl)
        data = chain_data(arrays)

        fixed_parameters = opt_config.custom_parameters is not None
        if fixed_parameters:
            u = validate_chain_parameters(opt_config.custom_parameters, arrays, data)
        elif opt_config.parameterization == "centripetal":
            u = chain_parameters([centripetal_parametrize(sec, alpha=0.5) for sec in sections], arrays)
        else:
            u = chain_parameters([compute_arc_length_parameters(sec) for sec in sections], arrays)

        free = opt_config.optimize_le_position and num_sections > 1
        C, d = self._constraints(sections, config, knots, free)
        if fixed_parameters:
            P = self._solve(self._basis(u, knots, degree, n_ctrl, num_sections), data, C, d)
        else:
            # Free junctions start from the refined fit with pinned ones, which
            # meets their constraints too: freeing them can only lower the error
            stages = [self._constraints(sections, config, knots, False)] if free else []
            C0, d0 = stages[0] if free else (C, d)
            P = self._solve(self._basis(u, knots, degree, n_ctrl, num_sections), data, C0, d0)
            for C_stage, d_stage in stages + [(C, d)]:
                problem = JointProblem(arrays, [knots] * num_sections, degree, C_stage, d_stage,
                                       regularization=0.0)
                u, z, _, _, _ = solve_joint(problem, u, problem.z_from(P), self.MAX_ITERATIONS)
                P = problem.control_points(z)

        section_control_points = [
            [(float(x), float(y)) for x, y in P[s * n_ctrl:(s + 1) * n_ctrl]]
            for s in range(num_sections)
        ]
        splines = [ScipyBSpline(knots, np.asarray(c), degree) for c in section_control_points]
        final_error = float(np.mean(np.linalg.norm(evaluate_chain(splines, u) - data, axis=1)))

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
            parameters=u,
        )

    @staticmethod
    def _basis(u: np.ndarray, knots: np.ndarray, degree: int, n_ctrl: int,
               num_sections: int) -> np.ndarray:
        """Basis matrix of the chain: each point's row in its section's block."""
        from bspline_varpro.chain import groups

        N = np.zeros((len(u), n_ctrl * num_sections))
        for s, (rows, local) in enumerate(groups(u, num_sections)):
            if len(rows):
                N[rows, s * n_ctrl:(s + 1) * n_ctrl] = ScipyBSpline.design_matrix(
                    local, knots, degree).toarray()
        return N

    @staticmethod
    def _solve(N_full: np.ndarray, all_pts: np.ndarray, C: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Stacked control points minimizing |N P - D| subject to C P = d (KKT)."""
        total_ctrl = N_full.shape[1]
        n_c = len(C)
        KKT = np.zeros((total_ctrl + n_c, total_ctrl + n_c))
        KKT[:total_ctrl, :total_ctrl] = N_full.T @ N_full + 1e-10 * np.eye(total_ctrl)
        KKT[:total_ctrl, total_ctrl:] = C.T
        KKT[total_ctrl:, :total_ctrl] = C
        rhs = np.vstack([N_full.T @ all_pts, d])
        return solve(KKT, rhs)[:total_ctrl]

    def _constraints(
        self,
        sections: list[PointList],
        config: SplineConfig,
        knots: np.ndarray,
        free_junctions: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Constraints C P = d on the stacked control points of all sections.

        Fixes the chain's end points, joins the sections, pins each junction
        to its split point unless free_junctions, and uses parametric (C^n)
        continuity where derivatives match exactly. d has one (x, y) row per
        constraint.
        """
        from ..models import ContinuityMode

        degree = config.degree
        n_ctrl = config.num_control_points
        mode = config.continuity
        num_sections = len(sections)

        # Total unknowns: n_ctrl per section
        total_ctrl = n_ctrl * num_sections

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

        return np.array(C_rows), np.column_stack([c_x, c_y])
