"""Variable projection formulation of constrained B-spline curve fitting.

For data D, parameters t, and control points P, the separable problem

    min_{t, P}  ||N(t) P - D||² + (P - c)' (lambda S + reg I) (P - c)   s.t.  C(theta) P = d(theta)

is linear in P. The ridge pulls the control points towards the centroid c of
the data, not the origin, so a fit does not change when the data are moved
(the bending energy of a constant curve is zero, so S c = 0). Variable
projection (Golub & Pereyra 1973) eliminates P by solving the
equality-constrained least-squares (KKT) system

    [N'N + lambda S + reg I   C'] [P*]   [N'D + reg c]
    [C                        0 ] [mu] = [d          ]

for every (t, theta), leaving a reduced problem in the outer variables alone.
Its residual is ``[N(t) P*(t) - D ; R (P*(t) - c)]`` with ``R'R = lambda S + reg I``,
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

import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import LinAlgWarning, block_diag, lu_factor, lu_solve

from .basis import (
    basis_derivative_matrix,
    basis_matrix,
    bending_energy_rows,
    centripetal_parameters,
    validate_knots,
)
from .chain import chain_data, chain_parameters, groups
from .constraints import ChainConstraints, ContinuitySpec, End, parse_junctions

#: Weight of the pull of each geometric junction's speed ratio alpha towards
#: its natural value, relative to the smoothing weight (see VarProProblem):
#: the weakest that keeps alpha clear of its range limits (0.84 of the way at
#: worst in the tests, against 0.97 without), so it changes fits little.
ALPHA_PULL = 3e-5

#: Weight, relative to the extent of the data, of the rows that stop the curve's
#: speed between consecutive data points (parameter gap over chord) from jumping
#: between neighbouring pairs (see VarProProblem). Strong: the rows only act
#: beyond SPEED_SLACK.
SPEED_PULL = 1e-3

#: Change of log speed from one pair of consecutive points to the next, beyond
#: its change in the centripetal parameters, that costs nothing. Converged fits
#: of NACA sections stay below 0.63; a parameter gap opened into a bulge between
#: two points jumps by over 4.
SPEED_SLACK = 1.0


class KKTSolver:
    """LU-factored KKT matrix ``[[A, C'], [C, 0]]`` for repeated solves.

    A singular matrix (no ridge, and control points the data, penalty and
    constraints leave free) is solved in the least-squares sense with the
    pseudo-inverse instead: the minimum-norm solution, which leaves the
    residual unchanged because the free directions do not reach it.
    """

    def __init__(self, A: np.ndarray, C: np.ndarray, regularization: float = 1e-12):
        self.n_ctrl = A.shape[0]
        self.n_c = C.shape[0]
        size = self.n_ctrl + self.n_c
        K = np.zeros((size, size))
        K[:self.n_ctrl, :self.n_ctrl] = A + regularization * np.eye(self.n_ctrl)
        K[:self.n_ctrl, self.n_ctrl:] = C.T
        K[self.n_ctrl:, :self.n_ctrl] = C
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            self._lu = lu_factor(K, check_finite=False)
        pivots = np.abs(np.diag(self._lu[0]))
        self._pinv = None
        if not (np.all(np.isfinite(pivots)) and np.min(pivots, initial=1.0)
                > size * np.finfo(float).eps * np.max(pivots, initial=1.0)):
            self._pinv = np.linalg.pinv(K, rcond=1e-12)

    def solve_full(self, rhs: np.ndarray, d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(control block, multiplier block)`` of the KKT solution."""
        full = np.zeros((self.n_ctrl + self.n_c,) + rhs.shape[1:])
        full[:self.n_ctrl] = rhs
        if d is not None:
            full[self.n_ctrl:] = d
        solution = self._pinv @ full if self._pinv is not None else lu_solve(self._lu, full)
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
            :class:`~bspline_varpro.constraints.Junction`. Defaults as in
            :func:`~bspline_varpro.fitting.fit`.
        fix_junctions: Pin junctions to the last point of their section (for
            specs that are not :class:`Junction` objects).
        smoothing: Weight lambda of the bending-energy penalty.
        regularization: Ridge ``reg`` (towards the data centroid) that keeps the
            KKT system nonsingular.
        start, end: :class:`~bspline_varpro.constraints.End` conditions of the chain.
    """

    def __init__(
        self,
        sections: Sequence[np.ndarray],
        knots: Union[np.ndarray, Sequence[np.ndarray]],
        degree: int,
        continuity: Union[ContinuitySpec, Sequence[ContinuitySpec]] = "G1",
        fix_junctions: bool = True,
        smoothing: float = 1e-3,
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

        # Penalty residual rows R with R'R = lambda S + reg I: the weighted
        # second-derivative samples of the bending energy, then the ridge
        self.bending_energy: Optional[np.ndarray] = None
        root = [np.sqrt(regularization) * np.eye(self.total_ctrl)]
        if smoothing > 0:
            rows = block_diag(*[bending_energy_rows(k, degree) for k in self.knots])
            self.bending_energy = rows.T @ rows
            root.insert(0, np.sqrt(smoothing) * rows)
        self.penalty_root = np.vstack(root)
        # Centroid the penalty pulls towards
        self.center = np.mean(self.data, axis=0)

        self.junctions = parse_junctions(continuity, self.num_sections - 1, fix_junctions)
        self.constraints = ChainConstraints(self.sections, self.knots, degree, self.junctions, start, end)
        self.num_params = self.constraints.num_params
        self.theta = self.constraints.initial_params()
        # The bending energy is measured per unit of each section's parameter, so
        # with smoothing a geometric junction can lower it by changing alpha alone,
        # which runs alpha to the end of its range. A weak pull of ln(alpha / ratio)
        # towards 0, scaled like the bending energy (smoothing times the squared
        # extent of the data) and absent without smoothing, keeps it inside.
        spread = float(np.mean(np.sum((self.data - self.center) ** 2, axis=1)))
        self.alpha_pull = float(np.sqrt(smoothing * ALPHA_PULL * spread)) if smoothing > 0 else 0.0
        self.num_pull_rows = len(self.constraints.alpha_deviations(self.theta)[0]) if self.alpha_pull > 0 else 0
        # The data residual only sees the curve at the data points, so a fit run to
        # convergence can open one parameter gap wide and spend it on a bulge between
        # two neighbouring points, to give the rest of the curve more freedom. Then the
        # speed (gap over chord) jumps there; it otherwise changes smoothly from pair to
        # pair however the points are spaced, as it does in the centripetal parameters.
        # Rows on the change of log speed beyond that and beyond SPEED_SLACK rule the
        # bulge out and leave regular fits alone. Pairs of coincident points have no
        # speed, and across a junction the speed changes with the section lengths and
        # a geometric join's alpha: those pairs are left out.
        chords = np.linalg.norm(np.diff(self.data, axis=0), axis=1)
        moving = chords > 1e-12 * max(float(np.max(chords, initial=0.0)), 1e-300)
        reference = chain_parameters([centripetal_parameters(sec) for sec in self.sections],
                                     self.sections)
        section = np.minimum(np.floor(0.5 * (reference[:-1] + reference[1:])),
                             self.num_sections - 1)
        self._speed_pairs = np.flatnonzero(moving[:-1] & moving[1:]
                                           & (section[:-1] == section[1:]))
        self._log_chords = np.log(np.where(moving, chords, 1.0))
        self._speed_reference = self._speed_change(reference)[0]
        self.speed_pull = SPEED_PULL * float(np.sqrt(spread))
        self.num_speed_rows = len(self._speed_pairs) if self.speed_pull > 0 else 0

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
        # Solved for the offset from the centroid (N'D + R'R c - (N'N + R'R) c = N'(D - c),
        # the basis summing to one), so a singular system's minimum-norm solution
        # does not depend on where the data are either
        center = np.tile(self.center, (self.total_ctrl, 1))
        offset, mu = kkt.solve_full(N.T @ (self.data - self.center), d - C @ center)
        P = offset + center

        self._cache_key = key
        self._cache = (N, P, dN, kkt, mu, theta)
        return self._cache[:4]

    def control_points(self, t_flat: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Optimal stacked control points P*(t), shape (total_ctrl, dim)."""
        return self.solve(t_flat, theta=theta)[1]

    def residual(self, t_flat: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Data residuals then penalty residuals, each flattened point-major.

        Shape ``(dim * (total_pts + len(penalty_root)) + num_pull_rows + num_speed_rows,)``:
        then come the rows pulling the speed ratio of each geometric junction
        towards its natural value, and last those keeping the curve's speed
        between consecutive data points changing smoothly. Also records
        ``last_control_points`` and the data-only ``last_rms_error``.
        """
        N, P, _, _ = self.solve(t_flat, theta=theta)
        data_residual = N @ P - self.data
        self.last_control_points = P
        self.last_rms_error = float(np.sqrt(np.sum(data_residual ** 2) / self.total_pts))
        return np.concatenate((data_residual.ravel(), (self.penalty_root @ (P - self.center)).ravel(),
                               self._pull(theta)[0], self._speed(t_flat)[0]))

    def _speed_change(self, t_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Change of log speed ``log(g_{i+1} / c_{i+1}) - log(g_i / c_i)`` per speed
        pair, for gaps g and chords c, and the gaps."""
        gaps = np.maximum(np.diff(t_flat), 1e-300)
        log_speed = np.log(gaps) - self._log_chords
        i = self._speed_pairs
        return log_speed[i + 1] - log_speed[i], gaps

    def _speed(self, t_flat: np.ndarray
               ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Speed rows, then for the Jacobian whether each row is active and the
        weighted inverse gaps ``pull / g``."""
        if self.num_speed_rows == 0:
            return np.zeros(0), None, None
        change, gaps = self._speed_change(t_flat)
        excess = change - self._speed_reference
        active = np.abs(excess) > SPEED_SLACK
        rows = np.where(active, excess - np.sign(excess) * SPEED_SLACK, 0.0)
        return self.speed_pull * rows, active, self.speed_pull / gaps

    def _pull(self, theta: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Residual rows pulling alpha towards its natural value, and their theta derivative."""
        if self.num_pull_rows == 0:
            return np.zeros(0), np.zeros((0, self.num_params))
        theta = self.theta if theta is None else np.asarray(theta, dtype=np.float64)
        values, jacobian = self.constraints.alpha_deviations(theta)
        return self.alpha_pull * values, self.alpha_pull * jacobian

    def _stack(self, N: np.ndarray, dP_by_coord: List[np.ndarray]) -> np.ndarray:
        J_data = np.empty((self.total_pts * self.dim, dP_by_coord[0].shape[1]))
        J_penalty = np.empty((len(self.penalty_root) * self.dim, dP_by_coord[0].shape[1]))
        for k, dP in enumerate(dP_by_coord):
            J_data[k::self.dim] = N @ dP
            J_penalty[k::self.dim] = self.penalty_root @ dP
        # The pull rows depend on theta alone (added by parameter_jacobian), the speed
        # rows on t alone (added by jacobian): zero here
        extra = self.num_pull_rows + self.num_speed_rows
        return np.vstack((J_data, J_penalty, np.zeros((extra, J_data.shape[1]))))

    def jacobian(self, t_flat: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Exact derivative of :meth:`residual` w.r.t. t, shape (len(residual), total_pts).

        ``dr_i/dt_j = [i == j] dN_i P* + N_i dP*/dt_j`` for data rows and
        ``R dP*/dt_j`` for penalty rows.

        A point exactly on a junction (an integer t) lies on the section that
        starts there, so its column is the derivative from the right. Across a
        geometric junction with alpha != 1 the residual has a kink there (the
        speed on the left is alpha times that on the right), and the column is
        one-sided; a parametric junction has no kink.
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
        if self.num_speed_rows:
            # Row of pair i: log g_{i+1} - log g_i (with g_i = t_{i+1} - t_i) where active
            _, active, inverse = self._speed(t_flat)
            i = self._speed_pairs[active]
            rows = len(J) - self.num_speed_rows + np.flatnonzero(active)
            J[rows, i] += inverse[i]
            J[rows, i + 1] -= inverse[i] + inverse[i + 1]
            J[rows, i + 2] += inverse[i + 1]
        return J

    def matrices(self, t_flat: np.ndarray, theta: Optional[np.ndarray] = None):
        """``(N, R, C, d)``: the basis, penalty-root and constraint matrices of the
        residual at (t, theta)."""
        theta = self.theta if theta is None else np.asarray(theta, dtype=np.float64)
        C, d, _, _ = self._constraints(theta)
        return self._block_matrix(t_flat, derivative=False), self.penalty_root, C, d

    def matrix_jacobian(self, t_flat: np.ndarray, theta: Optional[np.ndarray],
                        changes: Sequence[Tuple[np.ndarray, ...]]) -> np.ndarray:
        """Derivative of :meth:`residual` w.r.t. outer variables the matrices depend on.

        Each change is ``(dN, dR, dC, dd)``, the derivative of :meth:`matrices`
        w.r.t. one variable (a knot position) at fixed (t, theta). The KKT
        solve is differentiated exactly, so the matrices alone may come from
        finite differences: differencing the residual itself would amplify
        the rounding of the solve by the conditioning of the normal equations.
        Returns one column per change.
        """
        N, P, _, kkt = self.solve(t_flat, theta=theta)
        mu = self._cache[4]
        R = self.penalty_root
        misfit = self.data - N @ P
        shifted = P - self.center
        penalty_residual = R @ shifted
        columns = []
        for dN, dR, dC, dd in changes:
            rhs = dN.T @ misfit - N.T @ (dN @ P) - dR.T @ penalty_residual - R.T @ (dR @ shifted) - dC.T @ mu
            dP = kkt.solve(rhs, dd - dC @ P)
            columns.append(np.concatenate(((dN @ P + N @ dP).ravel(), (dR @ shifted + R @ dP).ravel(),
                                           np.zeros(self.num_pull_rows + self.num_speed_rows))))
        return np.stack(columns, axis=1)

    def parameter_jacobian(self, t_flat: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Exact derivative of :meth:`residual` w.r.t. theta, shape (len(residual), num_params)."""
        if self.num_params == 0:
            return np.zeros((self.dim * (self.total_pts + len(self.penalty_root))
                             + self.num_pull_rows + self.num_speed_rows, 0))
        N, P, _, kkt = self.solve(t_flat, theta=theta)
        mu, used_theta = self._cache[4], self._cache[5]
        _, _, dC, dd = self._constraints(used_theta)
        dPs = []
        for k in range(self.dim):
            rhs = np.stack([-dC_i.T @ mu[:, k] for dC_i in dC], axis=1)
            d_rhs = np.stack([dd_i[:, k] - dC_i @ P[:, k] for dC_i, dd_i in zip(dC, dd)], axis=1)
            dPs.append(kkt.solve(rhs, d_rhs))
        J = self._stack(N, dPs)
        if self.num_pull_rows:
            start = len(J) - self.num_speed_rows - self.num_pull_rows
            J[start:start + self.num_pull_rows] = self._pull(used_theta)[1]
        return J
