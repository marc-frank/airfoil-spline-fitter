"""Variable projection formulation of constrained B-spline curve fitting.

For data D, parameters t, and control points P, the separable problem

    min_{t, P}  ||N(t) P - D||² + P' (lambda S + reg I) P   s.t.  C(theta) P = d(theta)

is linear in P. Variable projection (Golub & Pereyra 1973) eliminates P by
solving the equality-constrained least-squares (KKT) system

    [N'N + lambda S + reg I   C'] [P*]   [N'D]
    [C                        0 ] [mu] = [d  ]

for every (t, theta), leaving a reduced problem in the outer variables alone.
Its residual is ``[N(t) P*(t) - D ; R P*(t)]`` with ``R'R = lambda S + reg I``,
so the outer objective equals the inner one exactly.

Every data point has a chain parameter t in [0, S] for S sections
(:mod:`~bspline_varpro.chain`): its row of N belongs to the section the
parameter lies on, so points are not assigned to sections for good.

The exact reduced Jacobian w.r.t. t has a direct term (moving t_j moves point j
along the curve) and a coupling term through dP*/dt_j, obtained by
differentiating the KKT system. Because t_j only affects row j of N, all n
sensitivities come from one batched solve with the already factored KKT matrix.
The constraint shape parameters theta (geometric continuity, tangent
magnitudes) only enter C and d; differentiating the KKT system gives

    [A  C'] [dP ]   [-dC' mu     ]
    [C  0 ] [dmu] = [dd - dC P*  ].

References:
    Golub & Pereyra (1973), "The Differentiation of Pseudo-Inverses and
    Nonlinear Least Squares Problems Whose Variables Separate".
    Kaufman (1975), "A variable projection method for solving separable
    nonlinear least squares problems".
    O'Leary & Rust (2013), "Variable projection for nonlinear least squares
    problems".
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import block_diag, lu_factor, lu_solve

from .basis import basis_derivative_matrix, basis_matrix, bending_energy_matrix, matrix_root, validate_knots
from .chain import chain_data, groups
from .constraints import ChainConstraints, ContinuitySpec, End, parse_junctions


class KKTSolver:
    """LU-factored KKT matrix ``[[A, C'], [C, 0]]`` for repeated solves."""

    def __init__(self, A: np.ndarray, C: np.ndarray, regularization: float = 1e-12):
        self.n_ctrl = A.shape[0]
        self.n_c = C.shape[0]
        size = self.n_ctrl + self.n_c
        K = np.zeros((size, size))
        K[:self.n_ctrl, :self.n_ctrl] = A + regularization * np.eye(self.n_ctrl)
        K[:self.n_ctrl, self.n_ctrl:] = C.T
        K[self.n_ctrl:, :self.n_ctrl] = C
        self._lu = lu_factor(K)

    def solve_full(self, rhs: np.ndarray, d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(control block, multiplier block)`` of the KKT solution."""
        full = np.zeros((self.n_ctrl + self.n_c,) + rhs.shape[1:])
        full[:self.n_ctrl] = rhs
        if d is not None:
            full[self.n_ctrl:] = d
        solution = lu_solve(self._lu, full)
        return solution[:self.n_ctrl], solution[self.n_ctrl:]

    def solve(self, rhs: np.ndarray, d: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve with control-point block ``rhs`` (n_ctrl, k) and constraint values ``d``.

        ``d = None`` means homogeneous constraints. Returns the (n_ctrl, k)
        control-point block of the solution.
        """
        return self.solve_full(rhs, d)[0]


class VarProProblem:
    """Reduced residual and exact Jacobian as functions of t and shape parameters.

    Args:
        sections: Ordered data points per section, each shaped (n_s, dim). They
            give the starting split and the junction and end positions; the
            chain is fitted to all points (:func:`~bspline_varpro.chain.chain_data`).
        knots: Clamped knot vector on [0, 1] shared by all sections, or one per section.
        degree: Spline degree.
        continuity: Junction continuity, one spec for all junctions or a list:
            ``"G2"`` (geometric), ``"C1"`` or ``1`` (parametric), or a
            :class:`~bspline_varpro.constraints.Junction`.
        fix_junctions: Pin junctions to the last point of their section (for
            specs that are not :class:`Junction` objects).
        smoothing: Weight lambda of the bending-energy penalty.
        regularization: Ridge ``reg`` that keeps the KKT system nonsingular.
        start, end: :class:`~bspline_varpro.constraints.End` conditions of the chain.
    """

    def __init__(
        self,
        sections: Sequence[np.ndarray],
        knots: Union[np.ndarray, Sequence[np.ndarray]],
        degree: int,
        continuity: Union[ContinuitySpec, Sequence[ContinuitySpec]] = 1,
        fix_junctions: bool = True,
        smoothing: float = 0.0,
        regularization: float = 1e-12,
        start: End = End(),
        end: End = End(),
    ):
        self.sections = [np.asarray(sec, dtype=np.float64) for sec in sections]
        self.data = chain_data(self.sections)
        self.num_sections = len(self.sections)
        if np.ndim(knots[0]) == 0:
            knots = [knots] * self.num_sections
        if len(knots) != self.num_sections:
            raise ValueError("Need one knot vector per section.")
        self.knots = [validate_knots(k, degree) for k in knots]
        self.degree = degree
        self.n_ctrl = [len(k) - degree - 1 for k in self.knots]
        self.offsets = np.concatenate([[0], np.cumsum(self.n_ctrl)]).astype(int)
        self.dim = self.data.shape[1]
        self.total_pts = len(self.data)
        self.total_ctrl = int(self.offsets[-1])
        self.smoothing = smoothing
        self.regularization = regularization

        self.bending_energy: Optional[np.ndarray] = None
        penalty = regularization * np.eye(self.total_ctrl)
        if smoothing > 0:
            self.bending_energy = block_diag(*[bending_energy_matrix(k, degree) for k in self.knots])
            penalty += smoothing * self.bending_energy
        self.penalty_root = matrix_root(penalty)

        self.junctions = parse_junctions(continuity, self.num_sections - 1, fix_junctions)
        self.constraints = ChainConstraints(self.sections, self.knots, degree, self.junctions, start, end)
        self.num_params = self.constraints.num_params
        self.theta = self.constraints.initial_params()

        self._constraint_key: Optional[bytes] = None
        self._constraint_cache: Tuple = ()
        self._cache_key: Optional[bytes] = None
        self._cache: Tuple = ()
        self.last_control_points: Optional[np.ndarray] = None
        self.last_rms_error = float("nan")

    @property
    def C(self) -> np.ndarray:
        return self._constraints(self.theta)[0]

    @property
    def d(self) -> np.ndarray:
        return self._constraints(self.theta)[1]

    def split_control_points(self, P: np.ndarray) -> List[np.ndarray]:
        """Split stacked control points into per-section arrays."""
        return [P[a:b] for a, b in zip(self.offsets[:-1], self.offsets[1:])]

    def _constraints(self, theta: np.ndarray):
        key = theta.tobytes()
        if self._constraint_key != key:
            self._constraint_cache = self.constraints.build(theta.copy(), with_derivatives=True)
            self._constraint_key = key
        return self._constraint_cache

    def _block_matrix(self, t_flat: np.ndarray, derivative: bool) -> np.ndarray:
        M = np.zeros((self.total_pts, self.total_ctrl))
        for s, (rows, local) in enumerate(groups(t_flat, self.num_sections)):
            if len(rows) == 0:
                continue
            block = (basis_derivative_matrix(local, self.knots[s], self.degree) if derivative
                     else basis_matrix(local, self.knots[s], self.degree))
            M[rows, self.offsets[s]:self.offsets[s + 1]] = block
        return M

    def solve(
        self, t_flat: np.ndarray, with_derivatives: bool = False, theta: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], KKTSolver]:
        """Return ``(N, P*, dN or None, kkt)`` at (t, theta), cached for the last call."""
        # Copy: solvers may pass views of buffers they free later, and theta is cached.
        theta = self.theta if theta is None else np.array(theta, dtype=np.float64)
        key = t_flat.tobytes() + b"|" + theta.tobytes()
        if self._cache_key == key:
            if with_derivatives and self._cache[2] is None:
                dN = self._block_matrix(t_flat, derivative=True)
                self._cache = self._cache[:2] + (dN,) + self._cache[3:]
            return self._cache[:4]

        N = self._block_matrix(t_flat, derivative=False)
        dN = self._block_matrix(t_flat, derivative=True) if with_derivatives else None
        A = N.T @ N
        if self.bending_energy is not None:
            A = A + self.smoothing * self.bending_energy
        C, d, _, _ = self._constraints(theta)
        kkt = KKTSolver(A, C, self.regularization)
        P, mu = kkt.solve_full(N.T @ self.data, d)

        self._cache_key = key
        self._cache = (N, P, dN, kkt, mu, theta)
        return self._cache[:4]

    def control_points(self, t_flat: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Optimal stacked control points P*(t), shape (total_ctrl, dim)."""
        return self.solve(t_flat, theta=theta)[1]

    def residual(self, t_flat: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Data residuals then penalty residuals, each flattened point-major.

        Shape ``(dim * (total_pts + total_ctrl),)``. Also records
        ``last_control_points`` and the data-only ``last_rms_error``.
        """
        N, P, _, _ = self.solve(t_flat, theta=theta)
        data_residual = N @ P - self.data
        self.last_control_points = P
        self.last_rms_error = float(np.sqrt(np.sum(data_residual ** 2) / self.total_pts))
        return np.concatenate((data_residual.ravel(), (self.penalty_root @ P).ravel()))

    def _stack(self, N: np.ndarray, dP_by_coord: List[np.ndarray]) -> np.ndarray:
        J_data = np.empty((self.total_pts * self.dim, dP_by_coord[0].shape[1]))
        J_penalty = np.empty((self.total_ctrl * self.dim, dP_by_coord[0].shape[1]))
        for k, dP in enumerate(dP_by_coord):
            J_data[k::self.dim] = N @ dP
            J_penalty[k::self.dim] = self.penalty_root @ dP
        return np.vstack((J_data, J_penalty))

    def jacobian(self, t_flat: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Exact derivative of :meth:`residual` w.r.t. t, shape (len(residual), total_pts).

        ``dr_i/dt_j = [i == j] dN_i P* + N_i dP*/dt_j`` for data rows and
        ``R dP*/dt_j`` for penalty rows.
        """
        N, P, dN, kkt = self.solve(t_flat, with_derivatives=True, theta=theta)
        tangent = dN @ P
        misfit = self.data - N @ P
        # Column j is d(N'D - A P*)/dt_j per coordinate (constraints do not depend on t).
        dPs = [kkt.solve(dN.T * misfit[:, k] - N.T * tangent[:, k]) for k in range(self.dim)]
        J = self._stack(N, dPs)
        diagonal = np.arange(self.total_pts)
        for k in range(self.dim):
            J[diagonal * self.dim + k, diagonal] += tangent[:, k]
        return J

    def parameter_jacobian(self, t_flat: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Exact derivative of :meth:`residual` w.r.t. theta, shape (len(residual), num_params)."""
        if self.num_params == 0:
            return np.zeros((self.dim * (self.total_pts + self.total_ctrl), 0))
        N, P, _, kkt = self.solve(t_flat, theta=theta)
        mu, used_theta = self._cache[4], self._cache[5]
        _, _, dC, dd = self._constraints(used_theta)
        dPs = []
        for k in range(self.dim):
            rhs = np.stack([-dC_i.T @ mu[:, k] for dC_i in dC], axis=1)
            d_rhs = np.stack([dd_i[:, k] - dC_i @ P[:, k] for dC_i, dd_i in zip(dC, dd)], axis=1)
            dPs.append(kkt.solve(rhs, d_rhs))
        return self._stack(N, dPs)
