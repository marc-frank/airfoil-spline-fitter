"""B-spline basis matrices, knot vectors, and initial data parameterization."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.interpolate import BSpline


@lru_cache(maxsize=None)
def _gauss_legendre(order: int):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.flags.writeable = False
    weights.flags.writeable = False
    return nodes, weights


def gauss_legendre(order: int):
    """Cached Gauss-Legendre nodes and weights on [-1, 1] (read-only arrays)."""
    return _gauss_legendre(int(order))


def open_uniform_knots(degree: int, num_control_points: int) -> np.ndarray:
    """Clamped knot vector on [0, 1] with uniformly spaced interior knots.

    The curve starts at the first and ends at the last control point.

    >>> open_uniform_knots(3, 5)
    array([0. , 0. , 0. , 0. , 0.5, 1. , 1. , 1. , 1. ])
    """
    if degree < 1:
        raise ValueError(f"Degree must be at least 1, got {degree}.")
    if num_control_points <= degree:
        raise ValueError(
            f"Number of control points ({num_control_points}) must be greater "
            f"than degree ({degree})."
        )
    num_interior = num_control_points - degree - 1
    interior = np.arange(1, num_interior + 1) * (1.0 / (num_control_points - degree))
    return np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])


def clamped_knots(degree: int, interior=()) -> np.ndarray:
    """Clamped knot vector on [0, 1] with the given interior knots (sorted)."""
    interior = np.sort(np.asarray(interior, dtype=np.float64).ravel())
    return np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])


def interior_knots(knots: np.ndarray, degree: int) -> np.ndarray:
    """Interior knots of a clamped knot vector."""
    return np.asarray(knots, dtype=np.float64)[degree + 1:len(knots) - degree - 1]


def validate_knots(knots: np.ndarray, degree: int) -> np.ndarray:
    """Check a clamped knot vector on [0, 1]; returns it as a float array."""
    knots = np.asarray(knots, dtype=np.float64)
    if knots.ndim != 1 or len(knots) < 2 * (degree + 1) or not np.all(np.isfinite(knots)):
        raise ValueError(f"Knot vector needs at least {2 * (degree + 1)} finite values.")
    if np.any(np.diff(knots) < 0):
        raise ValueError("Knot vector must be nondecreasing.")
    if np.any(knots[:degree + 1] != 0.0) or np.any(knots[-degree - 1:] != 1.0):
        raise ValueError("Knot vector must be clamped on [0, 1] (degree + 1 zeros and ones).")
    inner = interior_knots(knots, degree)
    if len(inner) and (inner[0] <= 0.0 or inner[-1] >= 1.0):
        raise ValueError("Interior knots must lie strictly inside (0, 1).")
    if len(inner):
        _, counts = np.unique(inner, return_counts=True)
        if np.max(counts) > degree:
            raise ValueError("Interior knot multiplicity must not exceed the degree.")
    return knots


def averaged_knots(t: np.ndarray, num_control_points: int, degree: int) -> np.ndarray:
    """Clamped knots placed by the data parameters (Piegl & Tiller eq. 9.68-9.69).

    Every knot span then contains data, which keeps least-squares fits well
    posed (Schoenberg-Whitney) when the data are unevenly spaced.
    """
    t = np.asarray(t, dtype=np.float64)
    n, m = num_control_points, len(t)
    if n <= degree:
        raise ValueError("Control-point count must exceed the degree.")
    num_interior = n - degree - 1
    if num_interior == 0:
        return clamped_knots(degree)
    ratio = m / (n - degree)
    j = np.arange(1, num_interior + 1)
    i = np.floor(j * ratio).astype(int)
    alpha = j * ratio - i
    i = np.clip(i, 1, m - 1)
    interior = (1 - alpha) * t[i - 1] + alpha * t[i]
    eps = 1e-9
    interior = np.clip(interior, eps, 1 - eps)
    return clamped_knots(degree, np.maximum.accumulate(interior))


def insert_knot(
    knots: np.ndarray, control_points: np.ndarray, degree: int, u: float,
) -> tuple:
    """Insert knot ``u`` once without changing the curve (Boehm's algorithm).

    Returns:
        ``(new_knots, new_control_points)``.

    Raises:
        ValueError: If ``u`` is not strictly inside the domain or would exceed
            multiplicity ``degree``.
    """
    knots = np.asarray(knots, dtype=np.float64)
    P = np.asarray(control_points, dtype=np.float64)
    if not knots[degree] < u < knots[-degree - 1]:
        raise ValueError(f"Knot {u} must lie strictly inside ({knots[degree]}, {knots[-degree - 1]}).")
    if np.count_nonzero(knots == u) >= degree:
        raise ValueError(f"Knot {u} already has multiplicity {degree}.")
    k = int(np.searchsorted(knots, u, side="right")) - 1
    k = min(max(k, degree), len(P) - 1)
    new_P = np.empty((len(P) + 1,) + P.shape[1:])
    new_P[:k - degree + 1] = P[:k - degree + 1]
    new_P[k + 1:] = P[k:]
    for i in range(k - degree + 1, k + 1):
        a = (u - knots[i]) / (knots[i + degree] - knots[i])
        new_P[i] = (1 - a) * P[i - 1] + a * P[i]
    return np.insert(knots, k + 1, u), new_P


def centripetal_parameters(points: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Monotone parameter values in [0, 1] for an ordered polyline.

    ``alpha`` = 0.5 gives centripetal, 1.0 chord-length, and 0.0 uniform
    parameterization. Coincident points fall back to uniform spacing.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros(len(pts))
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1) ** alpha
    t = np.concatenate([[0.0], np.cumsum(steps)])
    if t[-1] <= 0:
        return np.linspace(0.0, 1.0, len(pts))
    return t / t[-1]


def basis_matrix(t: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Dense matrix ``N[i, j] = B_j(t_i)``, shape (len(t), n_ctrl)."""
    t_clamped = np.clip(t, 0.0, 1.0)
    return BSpline.design_matrix(t_clamped, knots, degree).toarray()


def basis_derivative_matrix(
    t: np.ndarray, knots: np.ndarray, degree: int, order: int = 1,
) -> np.ndarray:
    """Dense matrix ``dN[i, j] = d^order B_j / dt^order (t_i)``."""
    n_ctrl = len(knots) - degree - 1
    if order > degree:
        return np.zeros((len(t), n_ctrl))
    t_clamped = np.clip(t, 0.0, 1.0)
    identity_spline = BSpline(knots, np.eye(n_ctrl), degree, extrapolate=False)
    return identity_spline.derivative(nu=order)(t_clamped)


def bending_energy_matrix(knots: np.ndarray, degree: int, quad_order: int = 8) -> np.ndarray:
    """Gram matrix ``S[i, j] = integral_0^1 B_i''(t) B_j''(t) dt``.

    ``P.T @ S @ P`` is the integrated squared second parametric derivative of
    the curve. The matrix is scaled so that ``trace(S) == n_ctrl``, which makes
    the smoothing weight roughly independent of the control-point count.
    Integrated exactly by Gauss-Legendre quadrature on each knot span.
    """
    knots = np.asarray(knots, dtype=np.float64)
    return _bending_energy(knots.tobytes(), int(degree), int(quad_order)).copy()


def bending_energy_rows(knots: np.ndarray, degree: int, quad_order: int = 8) -> np.ndarray:
    """``L`` with ``L.T @ L == bending_energy_matrix(knots, degree)``.

    Row q holds the second derivatives of the basis at Gauss node q, weighted
    by the root of its quadrature weight, so ``L @ P`` are the residuals whose
    squares sum to the bending energy. Unlike a matrix root of ``S``, they
    change smoothly with the knots without singular derivatives where ``S``
    is singular (straight lines carry no bending energy).
    """
    knots = np.asarray(knots, dtype=np.float64)
    return _bending_rows(knots.tobytes(), int(degree), int(quad_order)).copy()


@lru_cache(maxsize=1024)
def _bending_rows(knot_bytes: bytes, degree: int, quad_order: int) -> np.ndarray:
    knots = np.frombuffer(knot_bytes, dtype=np.float64)
    n_ctrl = len(knots) - degree - 1
    if degree < 2:
        return np.zeros((0, n_ctrl))

    nodes, weights = gauss_legendre(quad_order)
    d2_basis = BSpline(knots, np.eye(n_ctrl), degree, extrapolate=False).derivative(nu=2)

    rows = []
    breaks = np.unique(knots)
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b - a < 1e-14:
            continue
        half = 0.5 * (b - a)
        values = d2_basis(0.5 * (a + b) + half * nodes)
        rows.append(values * np.sqrt(weights * half)[:, np.newaxis])
    L = np.vstack(rows) if rows else np.zeros((0, n_ctrl))

    trace = float(np.sum(L * L))
    if trace > 1e-15:
        L *= np.sqrt(n_ctrl / trace)
    return L


@lru_cache(maxsize=1024)
def _bending_energy(knot_bytes: bytes, degree: int, quad_order: int) -> np.ndarray:
    L = _bending_rows(knot_bytes, degree, quad_order)
    return L.T @ L


def matrix_root(A: np.ndarray) -> np.ndarray:
    """``R`` with ``R.T @ R == A`` for a symmetric positive semidefinite ``A``.

    Uses the Cholesky factor, which is unique and changes smoothly with ``A``
    (an eigendecomposition root does not: repeated eigenvalues let the
    eigenvectors rotate freely, so finite differences across nearby problems
    would jump). Falls back to the eigendecomposition for singular ``A``.
    """
    try:
        return np.linalg.cholesky(A).T
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        return np.sqrt(np.maximum(eigenvalues, 0.0))[:, None] * eigenvectors.T
