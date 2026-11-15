"""
Constrained Least-Squares Optimizer
===================================

Direct LSQ: a single constrained linear least-squares solve, no iteration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import BSpline as ScipyBSpline
from scipy.linalg import solve

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

    One linear solve for the control points that:
    1. Minimize the least-squares error at fixed data parameters
    2. Fix the trailing-edge end points (first control point of the first
       section, last control point of the last section)
    3. Satisfy parametric (C^k) continuity constraints at the junctions

    Junction positions can be either:
    - Fixed to data points (optimize_le_position=False)
    - Free (optimize_le_position=True)

    The chain is fitted as a whole: every data point once, on the section its
    chain parameter lies on (bspline_varpro.chain). Points get their
    parameters from opt_config.custom_parameters or else from
    opt_config.parameterization (centripetal or arc length) within each
    section, and keep them: nothing is refined after the solve.
    """

    @property
    def name(self) -> str:
        return "Direct LSQ"

    def fit(
        self,
        config: SplineConfig,
        opt_config: OptimizationConfig,
        airfoil: AirfoilData,
        callback: Callable[..., bool] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> RawOptimResult:
        """
        Fit splines using one constrained least-squares solve.

        Args:
            config: Spline configuration
            opt_config: Optimization configuration
            airfoil: Airfoil data with sections
            callback: Optional callback, called once with the result as
                ``callback(x, error, preview=(section_control_points,
                section_knots, degree), step=False)`` (error = RMS error);
                its return value is ignored. The solve counts as one
                evaluation.
            should_stop: Accepted so that both optimizers are called alike;
                never read (there is nothing to cancel).

        Returns:
            RawOptimResult with computed control points and the points' chain
            parameters
        """
        from bspline_varpro import centripetal_parameters, open_uniform_knots
        from bspline_varpro.chain import (
            chain_data,
            chain_parameters,
            evaluate_chain,
            validate_chain_parameters,
        )

        from ...geometry import compute_arc_length_parameters
        from ..tracking import increment_evaluation

        sections = [list(sec) for sec in airfoil.sections]
        arrays = [np.array(sec, dtype=np.float64) for sec in sections]
        degree = config.degree
        n_ctrl = config.num_control_points
        num_sections = len(sections)
        knots = open_uniform_knots(degree, n_ctrl)
        data = chain_data(arrays)

        if opt_config.custom_parameters is not None:
            u = validate_chain_parameters(opt_config.custom_parameters, arrays, data)
        elif opt_config.parameterization == "centripetal":
            u = chain_parameters([centripetal_parameters(sec, 0.5) for sec in arrays], arrays)
        else:
            u = chain_parameters([compute_arc_length_parameters(sec) for sec in sections], arrays)

        free = opt_config.optimize_le_position and num_sections > 1
        C, d = self.constraints(sections, config, knots, free)
        P = self._solve(self._basis(u, knots, degree, n_ctrl, num_sections), data, C, d)
        increment_evaluation()

        section_control_points = [
            [(float(x), float(y)) for x, y in P[s * n_ctrl:(s + 1) * n_ctrl]]
            for s in range(num_sections)
        ]
        splines = [ScipyBSpline(knots, np.asarray(c), degree) for c in section_control_points]
        # RMS error at the data parameters
        final_error = float(np.sqrt(np.mean(np.sum((evaluate_chain(splines, u) - data) ** 2, axis=1))))

        x = np.array([v for ctrl_pts in section_control_points for pt in ctrl_pts for v in pt],
                     dtype=np.float64)
        if callback is not None:
            callback(x, final_error,
                     preview=([np.asarray(c) for c in section_control_points],
                              [knots] * num_sections, degree),
                     step=False)

        return RawOptimResult(
            x=x,
            fun=final_error,
            nit=1,
            section_control_points=section_control_points,
            parameters=u,
            success=True,
            message="",
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

    @staticmethod
    def constraints(
        sections: list[PointList],
        config: SplineConfig,
        knots: np.ndarray,
        free: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Constraints C P = d on the stacked control points of all sections.

        Fixes the chain's end points, joins the sections, pins each junction
        to its split point unless ``free``, and matches the derivatives of
        orders 1 .. continuity order exactly (parametric continuity), with
        the coefficients of ``bspline_varpro.end_derivative_coefficients``.
        d has one (x, y) row per constraint.
        """
        from bspline_varpro import end_derivative_coefficients

        degree = config.degree
        n_ctrl = config.num_control_points
        order = config.continuity.order
        num_sections = len(sections)
        total_ctrl = n_ctrl * num_sections

        rows: list[np.ndarray] = []
        values: list[tuple[float, float]] = []

        def add(entries: dict[int, float], value=(0.0, 0.0)) -> None:
            row = np.zeros(total_ctrl)
            for index, coefficient in entries.items():
                row[index] += coefficient
            rows.append(row)
            values.append((float(value[0]), float(value[1])))

        # The chain's end points (always fixed)
        add({0: 1.0}, sections[0][0])
        add({total_ctrl - 1: 1.0}, sections[-1][-1])

        # Only junctions have derivative rows: one section needs no coefficients
        # (and its degree may be below the continuity order)
        coefficients = {r: end_derivative_coefficients(knots, degree, r)
                        for r in range(1, order + 1)} if num_sections > 1 else {}
        for j in range(num_sections - 1):
            e = j * n_ctrl + n_ctrl - 1
            b = (j + 1) * n_ctrl
            # Position continuity: P_e = P_b
            add({e: 1.0, b: -1.0})
            if not free:
                add({e: 1.0}, sections[j][-1])
            for r in range(1, order + 1):
                if n_ctrl < r + 1:
                    break
                coef_end, coef_start = coefficients[r]
                entries = {e - r + i: float(c) for i, c in enumerate(coef_end)}
                for i, c in enumerate(coef_start):
                    entries[b + i] = entries.get(b + i, 0.0) - float(c)
                add(entries)

        return np.array(rows), np.array(values, dtype=np.float64).reshape(-1, 2)
