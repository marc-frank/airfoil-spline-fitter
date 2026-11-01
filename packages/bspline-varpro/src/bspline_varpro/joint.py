"""Joint optimization of data parameters and control points.

Unlike variable projection, the control points are kept as unknowns. The
constraints ``C P = d`` (with junction shape parameters frozen) are eliminated
by a null-space parameterization ``P = P0 + Z z``, and the data parameters t
are free per point: chain parameters in [0, S] (:mod:`~bspline_varpro.chain`),
so a point can move across a junction. Each data residual depends on one t_i only, so the normal
equations have a diagonal t-block that is eliminated by a Schur complement:
one Levenberg-Marquardt step costs O(n * k^2) for n points and k free control
coordinates, which handles dense data.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import block_diag

from .basis import basis_derivative_matrix, basis_matrix, bending_energy_matrix, matrix_root
from .chain import chain_data, groups


class JointProblem:
    """Residual and Jacobian blocks of the joint (t, z) problem.

    Args:
        sections: Data points per section (the starting split; the chain is
            fitted to all of them).
        knots: Knot vector per section.
        degree: Spline degree.
        C, d: Linear constraints on the stacked control points.
        smoothing: Weight of the parametric bending-energy penalty.
        regularization: Ridge on the control points.
    """

    def __init__(
        self,
        sections: Sequence[np.ndarray],
        knots: Sequence[np.ndarray],
        degree: int,
        C: np.ndarray,
        d: np.ndarray,
        smoothing: float = 0.0,
        regularization: float = 1e-12,
    ):
        self.sections = [np.asarray(s, dtype=np.float64) for s in sections]
        self.data = chain_data(self.sections)
        self.num_sections = len(self.sections)
        self.knots = [np.asarray(k, dtype=np.float64) for k in knots]
        self.degree = degree
        self.dim = self.data.shape[1]
        self.n_ctrl = [len(k) - degree - 1 for k in self.knots]
        self.offsets = np.concatenate([[0], np.cumsum(self.n_ctrl)]).astype(int)
        self.total_ctrl = int(self.offsets[-1])

        _, s, Vt = np.linalg.svd(C)
        rank = int(np.sum(s > 1e-10 * max(s[0], 1.0))) if len(s) else 0
        self.Z = Vt[rank:].T
        self.P0 = np.linalg.lstsq(C, d, rcond=None)[0]

        penalty = regularization * np.eye(self.total_ctrl)
        if smoothing > 0:
            penalty += smoothing * block_diag(*[bending_energy_matrix(k, degree) for k in self.knots])
        root = matrix_root(penalty)
        self.RZ = root @ self.Z
        self.RP0 = root @ self.P0

        # Every point but the chain's ends carries a free parameter
        self.free = np.arange(1, len(self.data) - 1)

    @property
    def num_z(self) -> int:
        return self.Z.shape[1]

    def control_points(self, z: np.ndarray) -> np.ndarray:
        """Stacked control points for flat or (num_z, dim) coordinates z."""
        return self.P0 + self.Z @ np.reshape(z, (self.num_z, self.dim))

    def z_from(self, P: np.ndarray) -> np.ndarray:
        """Flat null-space coordinates of control points that satisfy the constraints."""
        return (self.Z.T @ (P - self.P0)).ravel()

    def _basis(self, t_full: np.ndarray, derivative: bool) -> np.ndarray:
        M = np.zeros((len(t_full), self.total_ctrl))
        for s, (rows, local) in enumerate(groups(t_full, self.num_sections)):
            if len(rows) == 0:
                continue
            block = (basis_derivative_matrix(local, self.knots[s], self.degree) if derivative
                     else basis_matrix(local, self.knots[s], self.degree))
            M[rows, self.offsets[s]:self.offsets[s + 1]] = block
        return M

    def evaluate(self, t_full: np.ndarray, z: np.ndarray, with_jacobian: bool = False):
        """Residual parts and (optionally) Jacobian blocks at (t, z).

        Returns ``(r, blocks)`` with r the concatenated residual (data
        point-major, then penalty) and blocks
        ``(tangent, Jz)`` where ``Jz`` is the dense z-Jacobian of all rows.
        """
        P = self.control_points(z)
        N = self._basis(t_full, derivative=False)
        data_r = N @ P - self.data
        penalty_r = self.RP0 + self.RZ @ np.reshape(z, (self.num_z, self.dim))
        r = np.concatenate([data_r.ravel(), penalty_r.ravel()])
        if not with_jacobian:
            return r, None

        tangent = self._basis(t_full, derivative=True) @ P
        identity = np.eye(self.dim)
        Jz = np.vstack([np.kron(N @ self.Z, identity), np.kron(self.RZ, identity)])
        return r, (tangent, Jz)


def solve_joint(
    problem: JointProblem,
    t_full: np.ndarray,
    z: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
    on_step: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, int, bool, str]:
    """Levenberg-Marquardt with Schur elimination of the per-point parameters.

    ``on_step(t, z)`` is called after every accepted step; returning True
    stops with that state and the message ``"Stopped by callback."``.

    Returns ``(t, z, iterations, converged, message)``.
    """
    dim = problem.dim
    free = problem.free
    n_data_rows = len(problem.data) * dim
    t = t_full.copy()
    z = np.ravel(z).copy()
    r, (tangent, Jz) = problem.evaluate(t, z, with_jacobian=True)
    cost = 0.5 * r @ r
    mu = None
    nu = 2.0
    message = "Maximum iterations reached."
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        data_r = r[:n_data_rows].reshape(-1, dim)
        T = tangent[free]
        g_t = np.sum(T * data_r[free], axis=1)
        U = np.sum(T * T, axis=1)
        # W[i] = sum_k T[i, k] * Jz[data row (free[i], k)]
        Jz_data = Jz[:n_data_rows].reshape(len(problem.data), dim, -1)
        W = np.einsum("ik,ikj->ij", T, Jz_data[free])
        V = Jz.T @ Jz
        g_z = Jz.T @ r
        gradient_norm = max(np.max(np.abs(g_t), initial=0.0), np.max(np.abs(g_z), initial=0.0))
        if gradient_norm < tolerance * max(1.0, cost):
            converged, message = True, "Gradient below tolerance."
            break
        diag_V = np.maximum(np.diag(V), 1e-12 * max(np.max(np.diag(V), initial=0.0), 1e-300))
        U_floor = np.maximum(U, 1e-12 * max(np.max(U, initial=1.0), 1e-300))
        if mu is None:
            mu = 1e-3
        while True:
            U_d = U + mu * U_floor
            V_d = V + mu * np.diag(diag_V)
            S = V_d - W.T @ (W / U_d[:, None])
            rhs = -g_z + W.T @ (g_t / U_d)
            try:
                dz = np.linalg.solve(S, rhs)
            except np.linalg.LinAlgError:
                mu *= nu
                nu *= 2
                if mu > 1e16:
                    return t, z, iteration, False, "Singular normal equations."
                continue
            dt = -(g_t + W @ dz) / U_d
            t_new = t.copy()
            t_new[free] = np.clip(t[free] + dt, 0.0, problem.num_sections)
            z_new = z + dz
            r_new, _ = problem.evaluate(t_new, z_new)
            cost_new = 0.5 * r_new @ r_new
            predicted = -(g_t @ dt + g_z @ dz) - 0.5 * (dt @ (U * dt) + 2 * dt @ (W @ dz) + dz @ (V @ dz))
            rho = (cost - cost_new) / predicted if predicted > 0 else -1.0
            if rho > 0:
                mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                break
            mu *= nu
            nu *= 2.0
            if mu > 1e16:
                return t, z, iteration, True, "No further decrease possible."
        decrease = cost - cost_new
        t, z = t_new, z_new
        r, (tangent, Jz) = problem.evaluate(t, z, with_jacobian=True)
        cost = cost_new
        if on_step is not None and on_step(t, z):
            return t, z, iteration, False, "Stopped by callback."
        if decrease <= tolerance * cost:
            converged, message = True, "Relative decrease below tolerance."
            break
    return t, z, iteration, converged, message
