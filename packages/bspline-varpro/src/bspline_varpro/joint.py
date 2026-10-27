"""Joint optimization of data parameters and control points, with optional fairing.

Unlike variable projection, the control points are kept as unknowns. The
constraints ``C P = d`` (with junction shape parameters frozen) are eliminated
by a null-space parameterization ``P = P0 + Z z``, and the data parameters t
are free per point. Each data residual depends on one t_i only, so the normal
equations have a diagonal t-block that is eliminated by a Schur complement:
one Levenberg-Marquardt step costs O(n * k^2) for n points and k free control
coordinates. This handles dense data, and it allows residuals that are not
linear in P, such as the fairness energy

    integral (dkappa/ds)^2 ds  =  integral kappa_u^2 / |C'| du,

sampled by Gauss-Legendre quadrature on every knot span. ``dkappa/ds`` measures
curvature variation independent of the parameterization, which is what a
curvature comb shows.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.linalg import block_diag

from .basis import basis_derivative_matrix, basis_matrix, bending_energy_matrix, gauss_legendre


def fairness_terms(d1: np.ndarray, d2: np.ndarray, d3: np.ndarray):
    """``f = kappa_u / sqrt(|C'|)`` and its partials w.r.t. (x1, y1, x2, y2, x3, y3).

    With a = x1 y2 - y1 x2, b = x1 y3 - y1 x3, c = x1 x2 + y1 y2, V = |C'|^2:
    kappa_u = (b V - 3 a c) / V^(5/2), so f = (b V - 3 a c) V^(-11/4).
    """
    x1, y1, x2, y2, x3, y3 = d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1], d3[:, 0], d3[:, 1]
    V = x1 * x1 + y1 * y1
    a = x1 * y2 - y1 * x2
    b = x1 * y3 - y1 * x3
    c = x1 * x2 + y1 * y2
    g = b * V - 3 * a * c
    scale = V ** -2.75
    f = g * scale
    dg = np.stack([
        V * y3 + 2 * b * x1 - 3 * (c * y2 + a * x2),
        -V * x3 + 2 * b * y1 + 3 * c * x2 - 3 * a * y2,
        3 * c * y1 - 3 * a * x1,
        -3 * (c * x1 + a * y1),
        -V * y1,
        V * x1,
    ], axis=1)
    dV = np.zeros_like(dg)
    dV[:, 0], dV[:, 1] = 2 * x1, 2 * y1
    df = scale[:, None] * dg - 2.75 * (g * V ** -3.75)[:, None] * dV
    return f, df


class JointProblem:
    """Residual and Jacobian blocks of the joint (t, z) problem.

    Args:
        sections: Data points per section.
        knots: Knot vector per section.
        degree: Spline degree.
        C, d: Linear constraints on the stacked control points.
        smoothing: Weight of the parametric bending-energy penalty.
        regularization: Ridge on the control points.
        fairness: Weight of the integral of (dkappa/ds)^2 ds (planar curves only).
        quad_order: Gauss points per knot span for the fairness integral.
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
        fairness: float = 0.0,
        quad_order: int = 6,
    ):
        self.sections = [np.asarray(s, dtype=np.float64) for s in sections]
        self.data = np.vstack(self.sections)
        self.knots = [np.asarray(k, dtype=np.float64) for k in knots]
        self.degree = degree
        self.dim = self.data.shape[1]
        self.sizes = [len(s) for s in self.sections]
        self.n_ctrl = [len(k) - degree - 1 for k in self.knots]
        self.offsets = np.concatenate([[0], np.cumsum(self.n_ctrl)]).astype(int)
        self.total_ctrl = int(self.offsets[-1])
        if fairness > 0 and (self.dim != 2 or degree < 3):
            raise ValueError("Fairing needs planar curves of degree 3 or higher.")

        _, s, Vt = np.linalg.svd(C)
        rank = int(np.sum(s > 1e-10 * max(s[0], 1.0))) if len(s) else 0
        self.Z = Vt[rank:].T
        self.P0 = np.linalg.lstsq(C, d, rcond=None)[0]

        penalty = regularization * np.eye(self.total_ctrl)
        if smoothing > 0:
            penalty += smoothing * block_diag(*[bending_energy_matrix(k, degree) for k in self.knots])
        eigenvalues, eigenvectors = np.linalg.eigh(penalty)
        self.RZ = (np.sqrt(np.maximum(eigenvalues, 0.0))[:, None] * eigenvectors.T) @ self.Z
        self.RP0 = (np.sqrt(np.maximum(eigenvalues, 0.0))[:, None] * eigenvectors.T) @ self.P0

        self.fairness = fairness
        self.B = None
        if fairness > 0:
            nodes, weights = gauss_legendre(quad_order)
            blocks = [[], [], []]
            w_all = []
            for s, k in enumerate(self.knots):
                breaks = np.unique(k)
                u = np.concatenate([0.5 * (a + b) + 0.5 * (b - a) * nodes for a, b in zip(breaks[:-1], breaks[1:])])
                w = np.concatenate([0.5 * (b - a) * weights for a, b in zip(breaks[:-1], breaks[1:])])
                w_all.append(w)
                for order in (1, 2, 3):
                    M = np.zeros((len(u), self.total_ctrl))
                    M[:, self.offsets[s]:self.offsets[s + 1]] = basis_derivative_matrix(u, k, degree, order)
                    blocks[order - 1].append(M)
            self.B = [np.vstack(b) for b in blocks]
            self.BZ = [b @ self.Z for b in self.B]
            self.fair_scale = np.sqrt(fairness * np.concatenate(w_all))

        # Interior points of every section carry free parameters; section ends stay at 0 and 1.
        free = []
        start = 0
        for n in self.sizes:
            free.extend(range(start + 1, start + n - 1))
            start += n
        self.free = np.array(free, dtype=int)

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
        row = 0
        for s, n in enumerate(self.sizes):
            t = t_full[row:row + n]
            block = (basis_derivative_matrix(t, self.knots[s], self.degree) if derivative
                     else basis_matrix(t, self.knots[s], self.degree))
            M[row:row + n, self.offsets[s]:self.offsets[s + 1]] = block
            row += n
        return M

    def evaluate(self, t_full: np.ndarray, z: np.ndarray, with_jacobian: bool = False):
        """Residual parts and (optionally) Jacobian blocks at (t, z).

        Returns ``(r, blocks)`` with r the concatenated residual (data
        point-major, fairness, penalty) and blocks
        ``(tangent, N, Jz)`` where ``Jz`` is the dense z-Jacobian of all rows.
        """
        P = self.control_points(z)
        N = self._basis(t_full, derivative=False)
        data_r = N @ P - self.data
        parts = [data_r.ravel()]
        fair_df = None
        if self.B is not None:
            d1, d2, d3 = (b @ P for b in self.B)
            f, fair_df = fairness_terms(d1, d2, d3)
            parts.append(self.fair_scale * f)
        parts.append((self.RP0 + self.RZ @ np.reshape(z, (self.num_z, self.dim))).ravel())
        r = np.concatenate(parts)
        if not with_jacobian:
            return r, None

        dim, nz = self.dim, self.num_z
        tangent = self._basis(t_full, derivative=True) @ P
        NZ = N @ self.Z
        rows = [np.kron(NZ, np.eye(dim))]
        if fair_df is not None:
            Jf = np.zeros((len(fair_df), nz * dim))
            for k in range(dim):
                Jf[:, k::dim] = (fair_df[:, k:k + 1] * self.BZ[0] + fair_df[:, 2 + k:3 + k] * self.BZ[1]
                                 + fair_df[:, 4 + k:5 + k] * self.BZ[2])
            rows.append(self.fair_scale[:, None] * Jf)
        rows.append(np.kron(self.RZ, np.eye(dim)))
        return r, (tangent, np.vstack(rows))

    def fairness_energy(self, z: np.ndarray) -> float:
        """Integral of (dkappa/ds)^2 ds (requires ``fairness > 0``)."""
        if self.B is None:
            return float("nan")
        P = self.control_points(z)
        f, _ = fairness_terms(*(b @ P for b in self.B))
        return float(np.sum((self.fair_scale * f) ** 2) / self.fairness)


def solve_joint(
    problem: JointProblem,
    t_full: np.ndarray,
    z: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, int, bool, str]:
    """Levenberg-Marquardt with Schur elimination of the per-point parameters.

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
        diag_V = np.maximum(np.diag(V), 1e-12 * max(np.max(np.diag(V)), 1e-300))
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
            t_new[free] = np.clip(t[free] + dt, 0.0, 1.0)
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
        if decrease <= tolerance * cost:
            converged, message = True, "Relative decrease below tolerance."
            break
    return t, z, iteration, converged, message
