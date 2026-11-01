"""Airfoil helpers: .dat input/output, default tolerances, leading- and
trailing-edge handling, and analytic NACA sections for testing.

Airfoil coordinates are expected in Selig order: upper trailing edge, over the
leading edge, to the lower trailing edge. :func:`read_dat` converts Lednicer
files to that order.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

from .export import CurveInput, _arc_table, as_curves


class NACA4:
    """Analytic NACA 4-digit section with a smooth leading-edge parameterization.

    Each surface is parameterized by ``sigma = sqrt(x)`` in [0, 1], which makes the
    thickness distribution a polynomial and the surfaces analytic at the leading
    edge. ``closed_te`` selects the modified coefficient (-0.1036) for a sharp
    trailing edge; otherwise the original open coefficient (-0.1015) is used.
    """

    def __init__(self, code: str = "0012", closed_te: bool = True):
        if len(code) != 4 or not code.isdigit():
            raise ValueError("NACA 4-digit code must have four digits.")
        self.code = code
        self.m = int(code[0]) / 100.0
        self.p = int(code[1]) / 10.0
        self.t = int(code[2:]) / 100.0
        self.closed_te = closed_te
        self._a4 = -0.1036 if closed_te else -0.1015

    def _thickness(self, sigma: np.ndarray) -> np.ndarray:
        x = sigma * sigma
        return 5 * self.t * (0.2969 * sigma - 0.1260 * x - 0.3516 * x ** 2
                             + 0.2843 * x ** 3 + self._a4 * x ** 4)

    def _camber(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, p = self.m, self.p
        if m == 0 or p == 0:
            return np.zeros_like(x), np.zeros_like(x)
        front = x < p
        yc = np.where(front, m / p ** 2 * (2 * p * x - x ** 2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        slope = np.where(front, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
        return yc, slope

    def surface(self, sigma: np.ndarray, upper: bool = True) -> np.ndarray:
        """Points on the upper or lower surface at ``sigma = sqrt(x)``."""
        sigma = np.asarray(sigma, dtype=np.float64)
        x = sigma * sigma
        yt = self._thickness(sigma) * (1.0 if upper else -1.0)
        yc, slope = self._camber(x)
        theta = np.arctan(slope)
        return np.stack([x - yt * np.sin(theta), yc + yt * np.cos(theta)], axis=-1)

    def points(self, n_per_side: int = 81, spacing: str = "cosine") -> np.ndarray:
        """Selig-ordered coordinates: upper TE -> LE -> lower TE (LE point shared)."""
        if spacing == "cosine":
            beta = np.linspace(0.0, np.pi, n_per_side)
            sigma = np.sin(beta / 2)
        elif spacing == "uniform":
            sigma = np.sqrt(np.linspace(0.0, 1.0, n_per_side))
        else:
            raise ValueError("spacing must be 'cosine' or 'uniform'.")
        upper = self.surface(sigma[::-1], upper=True)
        lower = self.surface(sigma[1:], upper=False)
        return np.vstack([upper, lower])

    def leading_edge_curvature(self, upper: bool = True) -> float:
        """Curvature magnitude at the leading edge (x = 0), as the one-sided limit.

        Each surface polynomial continues analytically to negative ``sigma``, so
        central differences there give the exact one-sided limit.
        """
        h = 1e-4
        pts = self.surface(np.array([-h, 0.0, h]), upper=upper)
        d1 = (pts[2] - pts[0]) / (2 * h)
        d2 = (pts[2] - 2 * pts[1] + pts[0]) / h ** 2
        return float(abs(d1[0] * d2[1] - d1[1] * d2[0]) / np.linalg.norm(d1) ** 3)


def read_dat(path: Union[str, Path]) -> Tuple[str, np.ndarray]:
    """``(name, points)`` from a Selig or Lednicer airfoil file, in Selig order.

    A Lednicer file starts with the upper and lower point counts, then lists both
    surfaces from the leading to the trailing edge. The counts split the surfaces
    when they account for every point; otherwise the split is where x jumps back
    from the trailing to the leading edge. The name is the first line unless that
    line is a coordinate pair, then the file's stem.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    name = path.stem
    rows: List[List[float]] = []
    for i, line in enumerate(lines):
        parts = line.replace(",", " ").split()
        try:
            rows.append([float(parts[0]), float(parts[1])])
        except (ValueError, IndexError):
            if i == 0:
                name = line.strip()
    if not rows:
        raise ValueError(f"No coordinates found in {path}.")
    points = np.array(rows)
    n_upper, n_lower = points[0]
    if len(points) > 4 and min(n_upper, n_lower) >= 2 and np.all(points[0] == np.round(points[0])):
        points = points[1:]
        if n_upper + n_lower != len(points):
            n_upper = np.argmin(np.diff(points[:, 0])) + 1
        upper, lower = points[:int(n_upper)], points[int(n_upper):]
        if np.allclose(upper[0], lower[0]):
            lower = lower[1:]
        points = np.vstack([upper[::-1], lower])
    return name, points


def write_dat(path: Union[str, Path], points: np.ndarray, name: str = "airfoil", decimals: int = 6) -> Path:
    """Write Selig-format coordinates."""
    path = Path(path)
    body = "\n".join(f"{x: .{decimals}f} {y: .{decimals}f}" for x, y in np.asarray(points))
    path.write_text(f"{name}\n{body}\n", encoding="ascii", newline="\n")
    return path


def decimal_places(points: np.ndarray, max_decimals: int = 10) -> Optional[int]:
    """Decimals the coordinates were rounded to, or None for unrounded data."""
    values = np.asarray(points, dtype=np.float64).ravel()
    scale = max(1.0, float(np.max(np.abs(values))))
    for d in range(max_decimals + 1):
        if np.all(np.abs(values - np.round(values, d)) <= 1e-12 * scale):
            return d
    return None


def noise_level(points: np.ndarray, window: int = 9, degree: int = 6) -> float:
    """Robust estimate of the normal noise standard deviation of ordered points.

    A polynomial is fitted in a local frame to every window of consecutive
    points and the leverage-corrected residual at the window center is kept; the
    median absolute residual ignores the few windows where the polynomial cannot
    follow the shape (leading edge, trailing edge corner).
    """
    pts = np.asarray(points, dtype=np.float64)
    half = window // 2
    if len(pts) < window:
        return 0.0
    idx = np.arange(len(pts) - window + 1)[:, None] + np.arange(window)
    w = pts[idx]
    center = w[:, half:half + 1]
    chord = w[:, -1] - w[:, 0]
    length = np.linalg.norm(chord, axis=1)
    keep = length > 0
    w, center, chord, length = w[keep], center[keep], chord[keep], length[keep]
    e = chord / length[:, None]
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)
    x = np.einsum("wkd,wd->wk", w - center, e) / length[:, None]
    y = np.einsum("wkd,wd->wk", w - center, n)
    monotone = np.all(np.diff(x, axis=1) > 0, axis=1)
    x, y = x[monotone], y[monotone]
    if len(x) == 0:
        return 0.0
    V = x[:, :, None] ** np.arange(degree + 1)
    H = V @ np.linalg.pinv(V)
    r = y[:, half] - np.einsum("wk,wk->w", H[:, half], y)
    r = r / np.sqrt(np.maximum(1.0 - H[:, half, half], 1e-12))
    return float(1.4826 * np.median(np.abs(r)))


def precision_floor(points: np.ndarray) -> float:
    """Largest orthogonal error that rounding alone can cause (0 for unrounded data)."""
    d = decimal_places(points)
    return 0.0 if d is None else 0.5 * math.sqrt(2.0) * 10.0 ** -d


def default_tolerance(points: np.ndarray, relative_floor: float = 1e-6,
                      decimals: Optional[int] = None) -> float:
    """Tolerance a fit can meet without chasing rounding or noise.

    The largest of: one unit in the last decimal of rounded data, the expected
    maximum of the estimated noise over all points (unless the noise is
    explained by the rounding), and ``relative_floor`` times the chord.

    Args:
        decimals: Decimal places of the source file, for coordinates that were
            transformed after reading (detected from the values when omitted).
    """
    pts = np.asarray(points, dtype=np.float64)
    chord = float(np.max(np.ptp(pts, axis=0)))
    tol = relative_floor * chord
    sigma = noise_level(pts)
    d = decimal_places(pts) if decimals is None else decimals
    if d is not None:
        tol = max(tol, 10.0 ** -d)
        if sigma <= 3.0 * 10.0 ** -d / math.sqrt(12.0):
            return float(tol)
    expected_max = math.sqrt(2.0 * math.log(max(len(pts), 2))) + 0.5
    return float(max(tol, expected_max * sigma))


def leading_edge_index(points: np.ndarray) -> int:
    """Index of the point farthest from the trailing-edge midpoint."""
    pts = np.asarray(points, dtype=np.float64)
    te = 0.5 * (pts[0] + pts[-1])
    return int(np.argmax(np.linalg.norm(pts - te, axis=1)))


def split_at_leading_edge(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Upper (TE -> LE) and lower (LE -> TE) sections sharing the leading-edge point."""
    pts = np.asarray(points, dtype=np.float64)
    i = leading_edge_index(pts)
    if i == 0 or i == len(pts) - 1:
        raise ValueError("Leading edge found at an end; expected Selig order (TE -> LE -> TE).")
    return pts[:i + 1].copy(), pts[i:].copy()


def trailing_edge_gap(points_or_curves) -> float:
    """Distance between the first and last point of the airfoil."""
    if isinstance(points_or_curves, np.ndarray):
        pts = np.asarray(points_or_curves, dtype=np.float64)
        return float(np.linalg.norm(pts[0] - pts[-1]))
    curves = as_curves(points_or_curves)
    a = curves[0](curves[0].t[curves[0].k])
    b = curves[-1](curves[-1].t[-curves[-1].k - 1])
    return float(np.linalg.norm(a - b))


def fit_airfoil(
    points: np.ndarray,
    tolerance: Optional[float] = None,
    degree: int = 5,
    *,
    le_continuity="G2",
    fix_leading_edge: bool = True,
    sections: Optional[Sequence[np.ndarray]] = None,
    **kwargs,
):
    """Tolerance-driven fit of a Selig-ordered airfoil split at the leading edge.

    Args:
        points: Selig-ordered coordinates (upper TE -> LE -> lower TE).
        tolerance: Maximum deviation; :func:`default_tolerance` when omitted.
        degree: Spline degree (5 gives continuous curvature slope within sections).
        le_continuity: Continuity at the leading-edge junction ("G2" default).
        fix_leading_edge: Interpolate the leading-edge data point.
        sections: Custom sections instead of the leading-edge split; then
            ``le_continuity`` may be a per-junction list.
        **kwargs: Passed to :func:`bspline_varpro.fit_to_tolerance`.
    """
    from .tolerance import fit_to_tolerance

    pts = np.asarray(points, dtype=np.float64)
    if tolerance is None:
        tolerance = default_tolerance(pts)
    floor = precision_floor(pts)
    if tolerance < floor:
        warnings.warn(f"Tolerance {tolerance:g} is below the rounding error of the data ({floor:g}).",
                      stacklevel=2)
    chain = list(sections) if sections is not None else list(split_at_leading_edge(pts))
    return fit_to_tolerance(chain, tolerance, degree, continuity=le_continuity,
                            fix_junctions=fix_leading_edge, **kwargs)


def _chain_table(curves: List[BSpline]):
    """Arc length table ``(section, u, s)`` over a whole chain."""
    sec, us, ss = [], [], []
    offset = 0.0
    for i, curve in enumerate(curves):
        u, s = _arc_table(curve)
        sec.append(np.full(len(u), i))
        us.append(u)
        ss.append(s + offset)
        offset += s[-1]
    return np.concatenate(sec), np.concatenate(us), np.concatenate(ss)


def _evaluate_chain(curves: List[BSpline], table, s_values: np.ndarray) -> np.ndarray:
    # Sections repeat their boundary arc length, so a bracket never spans two sections.
    sec, u, s = table
    idx = np.clip(np.searchsorted(s, s_values, side="right") - 1, 0, len(s) - 2)
    ds = s[idx + 1] - s[idx]
    frac = np.clip(np.where(ds > 0, (s_values - s[idx]) / np.where(ds > 0, ds, 1.0), 0.0), 0.0, 1.0)
    params = u[idx] + frac * (u[idx + 1] - u[idx])
    out = np.empty((len(s_values), np.asarray(curves[0].c).shape[1]))
    for i, curve in enumerate(curves):
        mask = sec[idx] == i
        if np.any(mask):
            out[mask] = curve(params[mask])
    return out


def leading_edge(curves: CurveInput) -> Tuple[int, float, np.ndarray]:
    """``(section, parameter, point)`` of the curve point farthest from the TE midpoint."""
    curves = as_curves(curves)
    te = 0.5 * (curves[0](curves[0].t[curves[0].k]) + curves[-1](curves[-1].t[-curves[-1].k - 1]))
    best = None
    for i, curve in enumerate(curves):
        k = curve.k
        u = np.linspace(curve.t[k], curve.t[-k - 1], 2001)
        d = np.linalg.norm(curve(u) - te, axis=1)
        j = int(np.argmax(d))
        if best is None or d[j] > best[0]:
            best = (d[j], i, u, j)
    _, i, u, j = best
    lo, hi = u[max(j - 1, 0)], u[min(j + 1, len(u) - 1)]
    res = minimize_scalar(lambda v: -np.linalg.norm(curves[i](v) - te), bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-14})
    v = float(res.x) if -res.fun >= np.linalg.norm(curves[i](u[j]) - te) else float(u[j])
    return i, v, curves[i](v)


def leading_edge_radius(curves: CurveInput) -> float:
    """Radius of curvature at the leading edge of a fitted airfoil."""
    curves = as_curves(curves)
    i, u, _ = leading_edge(curves)
    d1, d2 = curves[i](u, nu=1), curves[i](u, nu=2)
    kappa = abs(d1[0] * d2[1] - d1[1] * d2[0]) / np.linalg.norm(d1) ** 3
    return float(1.0 / kappa) if kappa > 0 else math.inf


def _peak(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """``(x, y)`` of the largest |y|, refined by a parabola through its neighbours."""
    j = int(np.argmax(np.abs(y)))
    if 0 < j < len(y) - 1:
        x0, x1, x2 = x[j - 1:j + 2]
        y0, y1, y2 = y[j - 1:j + 2]
        a = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)
        if a[0] != 0:
            xv = -a[1] / (2 * a[0])
            if x0 <= xv <= x2:
                return float(xv), float(np.polyval(a, xv))
    return float(x[j]), float(y[j])


def thickness_and_camber(curves: CurveInput, stations: int = 201,
                         samples: int = 4000) -> dict:
    """Thickness and camber of a fitted airfoil along its chord.

    The x axis is the chord line, as in airfoil databases: the airfoil is
    not turned, only moved and scaled so the leading edge (the smallest x)
    is at 0 and the trailing-edge midpoint at (1, 0). The two surfaces are
    compared at the same x: the thickness is ``y_upper - y_lower`` and the
    camber their mean. (The NACA definitions measure thickness normal to
    the camber line instead, which differs by a little where it slopes.)

    Args:
        stations: Chordwise stations of the returned distributions, cosine
            spaced.
        samples: Points per surface the surfaces are sampled with.

    Returns:
        ``x``, ``thickness`` and ``camber`` at the stations, and
        ``max_thickness``, ``max_thickness_x``, ``max_camber`` and
        ``max_camber_x`` (the camber of largest magnitude, with its sign),
        all in fractions of the ``chord``.

    Raises:
        ValueError: The leading edge is at an end of the chain (not an airfoil).
    """
    curves = as_curves(curves)
    section, u_le, le = leading_edge(curves)
    start, end = curves[0].t[curves[0].k], curves[-1].t[-curves[-1].k - 1]
    if (section == 0 and u_le <= start) or (section == len(curves) - 1 and u_le >= end):
        raise ValueError("Leading edge found at an end; expected an airfoil (TE -> LE -> TE).")
    te = 0.5 * (curves[0](start) + curves[-1](end))
    pts = resample(curves, samples)
    front = int(np.argmin(pts[:, 0]))
    chord = float(te[0] - pts[front, 0])
    if chord <= 0:
        raise ValueError("Trailing edge not behind the leading edge along x.")
    local = (pts - [pts[front, 0], te[1]]) / chord
    # Split at the smallest x, so each surface has one y at every x
    upper, lower = local[:front + 1], local[front:]

    def surface(side: np.ndarray, x: np.ndarray) -> np.ndarray:
        order = np.argsort(side[:, 0], kind="stable")
        return np.interp(x, side[order, 0], side[order, 1])

    def distributions(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        yu, yl = surface(upper, x), surface(lower, x)
        return yu - yl, 0.5 * (yu + yl)

    fine = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, 4 * samples)))
    t_fine, c_fine = distributions(fine)
    t_x, t_max = _peak(fine, t_fine)
    c_x, c_max = _peak(fine, c_fine)
    x = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, stations)))
    thickness, camber = distributions(x)
    return {"x": x, "thickness": thickness, "camber": camber, "chord": chord,
            "max_thickness": t_max, "max_thickness_x": t_x,
            "max_camber": c_max, "max_camber_x": c_x}


def trailing_edge_segment(curves: CurveInput) -> Optional[BSpline]:
    """Straight B-spline closing a blunt trailing edge (lower TE -> upper TE).

    It has the degree of the airfoil curves so it can be exported alongside or
    merged. Returns None for a sharp trailing edge.
    """
    curves = as_curves(curves)
    start = curves[-1](curves[-1].t[-curves[-1].k - 1])
    end = curves[0](curves[0].t[curves[0].k])
    if np.linalg.norm(end - start) <= 1e-12 * max(1.0, float(np.max(np.abs(start)))):
        return None
    k = curves[0].k
    ctrl = start + np.linspace(0.0, 1.0, k + 1)[:, None] * (end - start)
    return BSpline(np.r_[np.zeros(k + 1), np.ones(k + 1)], ctrl, k)


def resample(curves: CurveInput, n_per_side: int = 100, spacing: str = "cosine") -> np.ndarray:
    """Selig-ordered points on a fitted airfoil, clustered at the LE and TE.

    Works for any number of sections: the chain is split at its geometric
    leading edge and each side gets ``n_per_side`` points (the LE point shared).

    Args:
        spacing: ``"cosine"`` (arc length, clustered at LE and TE) or ``"arc"``.
    """
    curves = as_curves(curves)
    if n_per_side < 2:
        raise ValueError("Need at least two points per side.")
    fractions = {
        "cosine": 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_per_side))),
        "arc": np.linspace(0, 1, n_per_side),
    }
    if spacing not in fractions:
        raise ValueError("spacing must be 'cosine' or 'arc'.")
    table = _chain_table(curves)
    i, u_le, p_le = leading_edge(curves)
    sec, u, s = table
    rows = np.flatnonzero(sec == i)
    s_le = float(np.interp(u_le, u[rows], s[rows]))
    f = fractions[spacing]
    s_values = np.concatenate([f * s_le, s_le + f[1:] * (s[-1] - s_le)])
    out = _evaluate_chain(curves, table, s_values)
    out[n_per_side - 1] = p_le
    out[0] = curves[0](curves[0].t[curves[0].k])
    out[-1] = curves[-1](curves[-1].t[-curves[-1].k - 1])
    return out

