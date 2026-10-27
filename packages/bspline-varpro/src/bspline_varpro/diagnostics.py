"""Geometric quality measures: orthogonal deviation, curvature, fairness, continuity."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import BSpline


def _span_grid(curve: BSpline, samples_per_span: int) -> np.ndarray:
    k = curve.k
    breaks = np.unique(curve.t[k:len(curve.t) - k])
    pieces = [np.linspace(a, b, samples_per_span + 1)[:-1] for a, b in zip(breaks[:-1], breaks[1:])]
    return np.concatenate(pieces + [breaks[-1:]])


def project(
    points: np.ndarray,
    curve: BSpline,
    samples_per_span: int = 32,
    candidates: int = 3,
    chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closest curve parameter and orthogonal distance for every point.

    The global minimum is located on a dense grid of ``samples_per_span`` samples
    per knot span. Up to ``candidates`` local minima of the sampled distance are
    then refined to machine precision with a bracketed, safeguarded Newton
    iteration, and the best one is kept. Curve end points are always candidates.

    Returns:
        ``(u, distance)``, each shaped (n,).
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    grid = _span_grid(curve, samples_per_span)
    samples = curve(grid)
    lo_domain, hi_domain = grid[0], grid[-1]
    u_best = np.empty(len(points))
    d_best = np.empty(len(points))

    for start in range(0, len(points), chunk):
        q = points[start:start + chunk]
        d2 = np.sum((samples[None, :, :] - q[:, None, :]) ** 2, axis=2)
        padded = np.pad(d2, ((0, 0), (1, 1)), constant_values=np.inf)
        local = (d2 <= padded[:, :-2]) & (d2 <= padded[:, 2:])
        ranked = np.argsort(np.where(local, d2, np.inf), axis=1)[:, :candidates]
        valid = np.take_along_axis(local, ranked, axis=1)

        rows = np.repeat(np.arange(len(q)), ranked.shape[1])
        j = ranked.ravel()
        lo = grid[np.maximum(j - 1, 0)]
        hi = grid[np.minimum(j + 1, len(grid) - 1)]
        u = grid[j].copy()
        target = q[rows]
        for _ in range(40):
            c0, c1, c2 = curve(u), curve(u, nu=1), curve(u, nu=2)
            diff = c0 - target
            g = np.sum(diff * c1, axis=1)
            h = np.sum(c1 * c1, axis=1) + np.sum(diff * c2, axis=1)
            hi = np.where(g > 0, u, hi)
            lo = np.where(g < 0, u, lo)
            with np.errstate(divide="ignore", invalid="ignore"):
                newton = u - g / h
            ok = (h > 0) & (newton > lo) & (newton < hi)
            u_new = np.where(ok, newton, 0.5 * (lo + hi))
            if np.all(np.abs(u_new - u) <= 1e-15 * (1 + np.abs(u))):
                u = u_new
                break
            u = u_new
        u = np.clip(u, lo_domain, hi_domain)
        dist2 = np.sum((curve(u) - target) ** 2, axis=1)
        dist2 = np.where(valid.ravel(), dist2, np.inf).reshape(ranked.shape)

        ends = np.array([lo_domain, hi_domain])
        end_d2 = np.sum((curve(ends)[None, :, :] - q[:, None, :]) ** 2, axis=2)
        all_u = np.concatenate([u.reshape(ranked.shape), np.broadcast_to(ends, (len(q), 2))], axis=1)
        all_d2 = np.concatenate([dist2, end_d2], axis=1)
        pick = np.argmin(all_d2, axis=1)
        u_best[start:start + chunk] = all_u[np.arange(len(q)), pick]
        d_best[start:start + chunk] = np.sqrt(all_d2[np.arange(len(q)), pick])
    return u_best, d_best


def distance_to_curves(
    points: np.ndarray, curves: Sequence[BSpline], samples_per_span: int = 32,
) -> np.ndarray:
    """Distance from each point to the nearest of several curves."""
    return np.min([project(points, c, samples_per_span)[1] for c in curves], axis=0)


def section_deviations(
    sections: Sequence[np.ndarray], curves: Sequence[BSpline], samples_per_span: int = 32,
) -> List[np.ndarray]:
    """Orthogonal distance of every point to the curve of its own section."""
    return [project(np.asarray(sec, dtype=np.float64), c, samples_per_span)[1]
            for sec, c in zip(sections, curves)]


def _planar(curve: BSpline) -> None:
    if np.asarray(curve.c).ndim != 2 or np.asarray(curve.c).shape[1] != 2:
        raise ValueError("Curvature measures require planar (2D) curves.")


def curvature(curve: BSpline, u: np.ndarray) -> np.ndarray:
    """Signed curvature of a planar curve, positive for counter-clockwise turning."""
    _planar(curve)
    d1, d2 = curve(u, nu=1), curve(u, nu=2)
    speed = np.hypot(d1[..., 0], d1[..., 1])
    return (d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]) / speed ** 3


def curvature_slope(curve: BSpline, u: np.ndarray) -> np.ndarray:
    """Derivative of signed curvature with respect to arc length, dkappa/ds."""
    _planar(curve)
    d1, d2 = curve(u, nu=1), curve(u, nu=2)
    d3 = curve(u, nu=3) if curve.k >= 3 else np.zeros_like(d1)
    x1, y1, x2, y2, x3, y3 = d1[..., 0], d1[..., 1], d2[..., 0], d2[..., 1], d3[..., 0], d3[..., 1]
    speed2 = x1 * x1 + y1 * y1
    a = x1 * y2 - y1 * x2
    b = x1 * y3 - y1 * x3
    c = x1 * x2 + y1 * y2
    return (b * speed2 - 3 * a * c) / speed2 ** 3


def _span_quadrature(curve: BSpline, order: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    k = curve.k
    breaks = np.unique(curve.t[k:len(curve.t) - k])
    u, w = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        u.append(0.5 * (a + b) + 0.5 * (b - a) * nodes)
        w.append(0.5 * (b - a) * weights)
    return np.concatenate(u), np.concatenate(w)


def _count_turns(values: np.ndarray, threshold: float) -> int:
    """Interior extrema of a sequence, ignoring reversals smaller than ``threshold``."""
    turns = 0
    direction = 0
    lo = hi = values[0]
    for v in values[1:]:
        if direction == 0:
            if v - lo > threshold:
                direction, hi = 1, v
            elif hi - v > threshold:
                direction, lo = -1, v
            else:
                lo, hi = min(lo, v), max(hi, v)
        elif direction == 1:
            if v > hi:
                hi = v
            elif hi - v > threshold:
                turns, direction, lo = turns + 1, -1, v
        else:
            if v < lo:
                lo = v
            elif v - lo > threshold:
                turns, direction, hi = turns + 1, 1, v
    return turns


def fairness(
    curves: Sequence[BSpline], samples_per_span: int = 64, min_amplitude: float = 1e-5,
) -> Dict[str, float]:
    """Curvature-comb quality of a chain of planar curves.

    Args:
        min_amplitude: Curvature reversals smaller than this fraction of the
            chain's maximum |curvature| are not counted as extrema.

    Returns:
        ``curvature_extrema``: interior curvature extrema of all sections
        (wiggles in the curvature comb; junctions are reported separately).
        ``roughness``: integral of (dkappa/ds)^2 ds over the chain.
        ``max_curvature_slope``: max |dkappa/ds|.
        ``max_curvature``: max |kappa|.
    """
    grids = [_span_grid(c, samples_per_span) for c in curves]
    kappas = [curvature(c, g) for c, g in zip(curves, grids)]
    max_kappa = float(max(np.max(np.abs(k)) for k in kappas))
    extrema = sum(_count_turns(k, min_amplitude * max_kappa) for k in kappas)
    roughness = 0.0
    max_slope = 0.0
    for curve, grid in zip(curves, grids):
        u, w = _span_quadrature(curve)
        speed = np.linalg.norm(curve(u, nu=1), axis=1)
        roughness += float(np.sum(w * curvature_slope(curve, u) ** 2 * speed))
        max_slope = max(max_slope, float(np.max(np.abs(curvature_slope(curve, grid)))))
    return {"curvature_extrema": extrema, "roughness": roughness,
            "max_curvature_slope": max_slope, "max_curvature": max_kappa}


def junctions(curves: Sequence[BSpline]) -> List[Dict[str, float]]:
    """Continuity measured at each junction between consecutive planar curves.

    Returns one dict per junction with the position gap, the tangent angle jump
    (radians), and the jumps in curvature and in dkappa/ds.
    """
    report = []
    for a, b in zip(curves[:-1], curves[1:]):
        ua, ub = a.t[-1], b.t[0]
        ta, tb = a(ua, nu=1), b(ub, nu=1)
        ta, tb = ta / np.linalg.norm(ta), tb / np.linalg.norm(tb)
        report.append({
            "gap": float(np.linalg.norm(a(ua) - b(ub))),
            "tangent_angle": float(2 * np.arctan2(np.linalg.norm(ta - tb), np.linalg.norm(ta + tb))),
            "curvature_jump": float(abs(curvature(a, ua) - curvature(b, ub))),
            "curvature_slope_jump": float(abs(curvature_slope(a, ua) - curvature_slope(b, ub))),
        })
    return report


def max_deviation(
    sections: Sequence[np.ndarray], curves: Sequence[BSpline], samples_per_span: int = 32,
) -> float:
    """Largest orthogonal distance of any point to its section's curve."""
    return float(max(np.max(d) for d in section_deviations(sections, curves, samples_per_span)))


def point_on_curves(
    point: np.ndarray, curves: Sequence[BSpline],
) -> Tuple[int, float, float]:
    """``(section index, parameter, distance)`` of the closest point on a chain."""
    best: Optional[Tuple[int, float, float]] = None
    for s, c in enumerate(curves):
        u, d = project(np.asarray(point, dtype=np.float64)[None, :], c)
        if best is None or d[0] < best[2]:
            best = (s, float(u[0]), float(d[0]))
    return best
