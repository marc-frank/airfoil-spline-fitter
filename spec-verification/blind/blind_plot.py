"""Plot geometry of the curvature comb (19.14.3) and deviation (19.14.5, 12.1, 12.5).

Curves are (knot vector, control points (n, 2), degree) tuples. Written only from spec.md.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:  # pragma: no cover
    from scipy.integrate import cumtrapz as cumulative_trapezoid

COMB_FINE_SAMPLES = 4000
COMB_TURNING_SHARE = 0.5


def _spline(curve):
    t, c, p = curve
    return BSpline(np.asarray(t, float), np.asarray(c, float), int(p))


def _domain(spl):
    return spl.t[spl.k], spl.t[-spl.k - 1]


def _ev(spl, u, nu=0):
    u = np.asarray(u, float)
    if nu > spl.k:
        return np.zeros(u.shape + (spl.c.shape[1],))
    return spl(u, nu=nu)


def _empty():
    return np.array([], float), np.array([], float)


def _join(pieces):
    """Join (k, 2) arrays with a NaN row between consecutive pieces."""
    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return _empty()
    out = []
    for i, p in enumerate(pieces):
        if i:
            out.append(np.full((1, 2), np.nan))
        out.append(p)
    a = np.vstack(out)
    return a[:, 0].copy(), a[:, 1].copy()


def _segments(base, tip):
    """[base0, tip0, NaN, ..., baseK, tipK, NaN] (19.14.3 step 4: ends with a NaN)."""
    k = len(base)
    if k == 0:
        return np.zeros((0, 2))
    arr = np.full((3 * k, 2), np.nan)
    arr[0::3] = base
    arr[1::3] = tip
    return arr


# ---------------------------------------------------------------- 19.14.3 comb
def _kappa_comb(d1, d2, d3):
    x1, y1 = d1[:, 0], d1[:, 1]
    x2, y2 = d2[:, 0], d2[:, 1]
    x3, y3 = d3[:, 0], d3[:, 1]
    v2 = x1 ** 2 + y1 ** 2
    a = x1 * y2 - y1 * x2
    b = x1 * y3 - y1 * x3
    c = x1 * x2 + y1 * y2
    ok = v2 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.sqrt(v2)
        k = np.where(ok, a / v ** 3, 0.0)
        dk = np.where(ok, (b * v2 - 3 * a * c) / v ** 6, 0.0)
    return k, dk


def _kappa_app(d1, d2):
    # 2.4 application guard (compute_signed_curvature): 0 where v^3 <= 1e-12.
    x1, y1 = d1[:, 0], d1[:, 1]
    v3 = (x1 ** 2 + y1 ** 2) ** 1.5
    a = x1 * d2[:, 1] - y1 * d2[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v3 <= 1e-12, 0.0, a / v3)


def _fine_grid(n):
    return np.linspace(0.0, 1.0, max(COMB_FINE_SAMPLES, 20 * n))


def comb_parameters(spl, n):
    grid = _fine_grid(n)
    d1 = _ev(spl, grid, 1)
    d2 = _ev(spl, grid, 2)
    speed = np.linalg.norm(d1, axis=1)
    kappa = _kappa_app(d1, d2)
    s = cumulative_trapezoid(speed, grid, initial=0.0)
    turn_int = np.abs(np.nan_to_num(kappa, nan=0.0)) * speed
    turn = cumulative_trapezoid(turn_int, grid, initial=0.0)
    if s[-1] <= 0:
        return np.linspace(0, 1, n)
    measure = s / s[-1]
    if turn[-1] > 0:
        measure = (1 - COMB_TURNING_SHARE) * measure + COMB_TURNING_SHARE * turn / turn[-1]
    return np.interp(np.linspace(0, 1, n), measure, grid)


def comb_values(spl, t, q):
    t = np.asarray(t, float)
    if q <= 1:
        k, dk = _kappa_comb(_ev(spl, t, 1), _ev(spl, t, 2), _ev(spl, t, 3))
        return k if q == 0 else dk
    grid = _fine_grid(len(t))
    d1 = _ev(spl, grid, 1)
    _k, dk = _kappa_comb(d1, _ev(spl, grid, 2), _ev(spl, grid, 3))
    s = cumulative_trapezoid(np.linalg.norm(d1, axis=1), grid, initial=0.0)
    # 19.14.3: compared with the neighbour on the full grid.
    keep = np.concatenate([[0], np.nonzero(s[1:] > s[:-1])[0] + 1])
    g, ss, vals = grid[keep], s[keep], dk[keep]
    for _ in range(q - 1):
        vals = np.gradient(vals, ss, edge_order=2)
    return np.interp(t, g, vals)


def _all_chord(spls):
    xs = []
    for spl in spls:
        xs.append(_ev(spl, np.linspace(0.0, 1.0, 200))[:, 0])
    x = np.concatenate(xs)
    chord = float(np.max(x) - np.min(x))
    return chord if chord != 0 else 1.0


def comb(curves, scale, density, q):
    height = float(scale)
    q = int(q)
    density = max(5, int(density))
    spls = [_spline(c) for c in curves]
    chord = _all_chord(spls)
    teeth = {0: [], 1: []}
    env = {0: [], 1: []}
    for i, spl in enumerate(spls):
        t = comb_parameters(spl, density)
        val = comb_values(spl, t, q)
        base = _ev(spl, t)
        d1 = _ev(spl, t, 1)
        sp = np.linalg.norm(d1, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            nrm = np.where((sp > 1e-12)[:, None],
                           np.column_stack([d1[:, 1], -d1[:, 0]]) / sp[:, None], 0.0)
        tip = base + (val * height * chord ** (q + 2))[:, None] * nrm
        teeth[i % 2].append(_segments(base, tip))
        env[i % 2].append(tip)
    out = {}
    for par in (0, 1):
        out[(par, False)] = _join(teeth[par])
        out[(par, True)] = _join(env[par])
    return out


# ---------------------------------------------------------------- 12.1 projection
def project(points, spl, samples_per_span=32, candidates=3, chunk=4096):
    pts = np.asarray(points, float)
    p = spl.k
    lo_d, hi_d = _domain(spl)
    breaks = np.unique(spl.t[p: len(spl.t) - p])
    grid = np.concatenate(
        [np.linspace(a, b, samples_per_span + 1)[:-1] for a, b in zip(breaks[:-1], breaks[1:])]
        + [breaks[-1:]]
    )
    samples = _ev(spl, grid)
    M = len(grid)
    u_out = np.empty(len(pts))
    d_out = np.empty(len(pts))
    for c0 in range(0, len(pts), chunk):
        Q = pts[c0: c0 + chunk]
        d2 = ((Q[:, None, :] - samples[None, :, :]) ** 2).sum(axis=2)
        left = np.concatenate([np.full((len(Q), 1), np.inf), d2[:, :-1]], axis=1)
        right = np.concatenate([d2[:, 1:], np.full((len(Q), 1), np.inf)], axis=1)
        is_min = (d2 <= left) & (d2 <= right)
        ranked = np.argsort(np.where(is_min, d2, np.inf), axis=1, kind="stable")[:, :candidates]
        cand_u = []
        for r in range(ranked.shape[1]):
            j = ranked[:, r]
            valid = is_min[np.arange(len(Q)), j]
            lo = grid[np.clip(j - 1, 0, M - 1)].copy()
            hi = grid[np.clip(j + 1, 0, M - 1)].copy()
            u = grid[j].copy()
            active = valid.copy()
            for _ in range(40):
                if not active.any():
                    break
                ua = u[active]
                C = _ev(spl, ua)
                C1 = _ev(spl, ua, 1)
                C2 = _ev(spl, ua, 2)
                diff = C - Q[active]
                g = (diff * C1).sum(axis=1)
                h = (C1 * C1).sum(axis=1) + (diff * C2).sum(axis=1)
                la, ha = lo[active], hi[active]
                ha = np.where(g > 0, ua, ha)
                la = np.where(g < 0, ua, la)
                with np.errstate(divide="ignore", invalid="ignore"):
                    newton = ua - g / h
                take = (h > 0) & (newton > la) & (newton < ha)
                un = np.where(take, newton, (la + ha) / 2)
                done = np.abs(un - ua) <= 1e-15 * (1 + np.abs(ua))
                lo[active], hi[active] = la, ha
                u[active] = un
                idx = np.nonzero(active)[0]
                active[idx[done]] = False
            u = np.clip(u, lo_d, hi_d)
            cand_u.append((u, valid))
        # foot: refined candidates (rank order), then start, then end; first on ties.
        best_u = np.full(len(Q), lo_d)
        best_d = np.full(len(Q), np.inf)
        options = cand_u + [(np.full(len(Q), lo_d), np.ones(len(Q), bool)),
                            (np.full(len(Q), hi_d), np.ones(len(Q), bool))]
        for u, valid in options:
            d = np.linalg.norm(_ev(spl, u) - Q, axis=1)
            better = valid & (d < best_d)
            best_u = np.where(better, u, best_u)
            best_d = np.where(better, d, best_d)
        u_out[c0: c0 + chunk] = best_u
        d_out[c0: c0 + chunk] = best_d
    return u_out, d_out


# ---------------------------------------------------------------- 12.5 foot points
def foot_points(points, spls):
    pts = np.asarray(points, float)
    n, S = len(pts), len(spls)
    u = np.empty((n, S))
    for s, spl in enumerate(spls):
        u[:, s] = project(pts, spl)[0]
    best = np.full(n, np.inf)
    foot = np.zeros((n, 2))
    tang = np.zeros((n, 2))
    dists = np.empty((n, S))
    for s, spl in enumerate(spls):
        at = _ev(spl, u[:, s])
        d = np.linalg.norm(pts - at, axis=1)
        dists[:, s] = d
        better = d < best
        best = np.where(better, d, best)
        foot[better] = at[better]
        tang[better] = _ev(spl, u[better, s], 1)
    nrm = np.column_stack([tang[:, 1], -tang[:, 0]])
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
    signed = np.copysign(best, ((pts - foot) * nrm).sum(axis=1))
    rng = float(np.max(np.ptp(pts, axis=0))) if n else 0.0
    tie = 1e-9 * max(rng, 1e-300)
    section = np.argmax(dists <= (best + tie)[:, None], axis=1)
    section = np.maximum.accumulate(section)
    return signed, foot, nrm, u, section


# ---------------------------------------------------------------- 19.14.5 deviation
def deviation(points, curves, height, tolerance):
    hidden = {(p, t): _empty() for p in (0, 1) for t in (False, True)}
    hidden["band"] = _empty()
    pts = np.asarray(points, float)
    if height is None or len(pts) < 2:
        return hidden
    spls = [_spline(c) for c in curves]
    signed, foot, nrm, _u, section = foot_points(pts, spls)
    reference = tolerance or float(np.max(np.abs(signed)))
    chord = float(np.ptp(pts[:, 0]))
    if reference <= 0 or chord <= 0:
        return hidden
    scale = float(height) * chord / reference
    tips = foot + (scale * signed)[:, None] * nrm
    whisk = {0: [], 1: []}
    env = {0: [], 1: []}
    start = 0
    n = len(pts)
    runs = []
    while start < n:
        end = start
        while end + 1 < n and section[end + 1] == section[start]:
            end += 1
        runs.append((start, end))
        start = end + 1
    for r, (a, b) in enumerate(runs):
        first = a - 1 if r > 0 else a
        par = int(section[a]) % 2
        idx = np.arange(first, b + 1)
        whisk[par].append(_segments(foot[idx], tips[idx]))
        env[par].append(tips[idx])
    out = {}
    for par in (0, 1):
        out[(par, False)] = _join(whisk[par])
        out[(par, True)] = _join(env[par])
    if tolerance:
        lines = []
        off = scale * float(tolerance)
        for spl in spls:
            a, b = _domain(spl)
            uu = np.linspace(a, b, 400)
            C = _ev(spl, uu)
            d1 = _ev(spl, uu, 1)
            nn = np.column_stack([d1[:, 1], -d1[:, 0]])
            nn = nn / np.maximum(np.linalg.norm(nn, axis=1), 1e-300)[:, None]
            # 19.14.5 step 5: section 1 +, section 1 -, section 2 +, ... one NaN between.
            lines.append(C + off * nn)
            lines.append(C - off * nn)
        out["band"] = _join(lines)
    else:
        out["band"] = _empty()
    return out
