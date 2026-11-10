"""Blind reimplementation of selected spline_fitter / bspline_varpro functions.

Written only from spec.md. Ambiguities are marked with ``# SPEC-GAP:`` comments.
"""
from __future__ import annotations

import math
import re

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve

EPS = np.finfo(float).eps


# ---------------------------------------------------------------- 2.1 knots
def open_uniform_knots(p, n):
    # SPEC-GAP: 2.1 two message variants (library with period, application without);
    # the task names the library behaviour, so the library messages (and p < 1 check) are used.
    if p < 1:
        raise ValueError(f"Degree must be at least 1, got {p}.")
    if n <= p:
        raise ValueError(f"Number of control points ({n}) must be greater than degree ({p}).")
    k = n - p - 1
    step = 1.0 / (n - p)
    interior = [i * step for i in range(1, k + 1)]
    return np.array([0.0] * (p + 1) + interior + [1.0] * (p + 1), dtype=float)


def clamped_knots(p, interior=()):
    inner = sorted(float(v) for v in interior)
    return np.array([0.0] * (p + 1) + inner + [1.0] * (p + 1), dtype=float)


def _interior(t, p):
    t = np.asarray(t, float)
    return t[p + 1: len(t) - p - 1]


def _basis(u, t, p, nu=0):
    t = np.asarray(t, float)
    n = len(t) - p - 1
    u = np.clip(np.atleast_1d(np.asarray(u, float)), 0.0, 1.0)
    if nu > p:
        return np.zeros((len(u), n))
    spl = BSpline(t, np.eye(n), p)
    return np.asarray(spl(u, nu=nu)).reshape(len(u), n)


# ---------------------------------------------------------------- 2.3 bending energy
def bending_energy(t, p):
    t = np.asarray(t, float)
    n = len(t) - p - 1
    S = np.zeros((n, n))
    if p < 2:
        return S
    xi, w = np.polynomial.legendre.leggauss(8)
    spl = BSpline(t, np.eye(n), p)
    d2 = spl.derivative(2)
    vals = np.unique(t)
    for a, b in zip(vals[:-1], vals[1:]):
        if b - a < 1e-14:
            continue
        u = (a + b) / 2 + (b - a) / 2 * xi
        B = np.asarray(d2(u)).reshape(len(u), n)
        S += B.T @ (B * (w * (b - a) / 2)[:, None])
    tr = np.trace(S)
    if tr > 1e-15:
        S = S * (n / tr)
    return S


# ---------------------------------------------------------------- 2.6 parameterizations
def centripetal(P, a=0.5):
    P = np.asarray(P, float)
    N = len(P)
    if N < 2:
        return np.zeros(N)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1) ** a
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] <= 0:
        return np.linspace(0, 1, N)
    return t / t[-1]


def arc_params(P):
    P = list(P)
    N = len(P)
    if N == 0:
        return []
    if N == 1:
        return [0.0]
    cum = [0.0]
    for i in range(1, N):
        cum.append(cum[-1] + math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]))
    total = cum[-1]
    if total <= 0:
        return [i / (N - 1) for i in range(N)]
    return [c / total for c in cum]


# ---------------------------------------------------------------- 3 reading
def get_precision(s):
    s = str(s).lower().replace("d", "e")
    if "e" in s:
        mant, expo = s.split("e", 1)
    else:
        mant, expo = s, ""
    decimals = len(mant.split(".", 1)[1]) if "." in mant else 0
    if expo == "":
        return decimals
    try:
        return decimals - int(expo)
    except ValueError:
        return decimals


_SPLIT_SIGN = re.compile(r"(?<=[0-9.])(?=[+-][0-9.])")
_FORTRAN_D = re.compile(r"(?<=[0-9.])[dD](?=[+-]?[0-9])")


def _number_fields(content):
    s = content.replace(";", " ")
    s = _SPLIT_SIGN.sub(" ", s)
    if "," in s:
        if len(s.split()) >= 2 and "." not in s:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", " ")
    out = []
    for f in s.split():
        f = f.strip("()\"'")
        f = _FORTRAN_D.sub("e", f)
        out.append(f)
    return out


def _name_line(name):
    name = str(name).strip()
    if re.match(r"^\\*#", name):
        name = "\\" + name
    return name


def _name_of_line(line):
    if re.match(r"^\\+#", line):
        return line[1:]
    return line


def _is_point_counts(x, y):
    # Short-circuit order per 3.5; int(inf) raises OverflowError on purpose (not caught).
    return x >= 2 and y >= 2 and x == int(x) and y == int(y)


def _is_lednicer(rows):
    if len(rows) <= 4:
        return False
    x0, y0 = rows[0][0], rows[0][1]
    if not _is_point_counts(x0, y0):
        return False
    m = len(rows) - 1
    if not abs(x0 + y0 - m) <= max(2, m // 10):
        return False
    xs = [r[0] for r in rows[1:]]
    return xs[0] - min(xs) < max(xs) - xs[0]


def _is_stray_first_row(rows):
    if len(rows) <= 4:
        return False
    xs = [r[0] for r in rows[1:]]
    ys = [r[1] for r in rows[1:]]
    m = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys))
    x, y = rows[0][0], rows[0][1]
    inside = (min(xs) - m <= x <= max(xs) + m) and (min(ys) - m <= y <= max(ys) + m)
    return not inside


def _lednicer_to_selig(coords, n_upper, n_lower):
    if n_upper + n_lower != len(coords):
        steps = [coords[i + 1][0] - coords[i][0] for i in range(len(coords) - 1)]
        n_upper = int(np.argmin(steps)) + 1
    upper = coords[:n_upper]
    lower = coords[n_upper:]
    if lower and lower[0] == upper[0]:
        lower = lower[1:]
    return list(reversed(upper)) + list(lower)


def _file_precision(texts):
    plain, sci = [], []
    for s in texts:
        low = s.lower().replace("d", "e")
        (sci if "e" in low else plain).append(get_precision(s))
    cands = []
    if plain:
        cands.append(max(plain))
    if sci:
        cands.append(min(sci))
    return max(0, min(cands)) if cands else 0


def _filter_outliers(coords, threshold_factor=5.0):
    if len(coords) < 4:
        return coords
    srt = sorted(coords)
    med = srt[len(srt) // 2]
    d = [math.hypot(x - med[0], y - med[1]) for x, y in coords]
    sd = sorted(d)
    q1 = sd[len(sd) // 4]
    q3 = sd[3 * len(sd) // 4]
    thr = q3 + threshold_factor * (q3 - q1)
    kept = [c for c, di in zip(coords, d) if di <= thr]
    if len(kept) < 0.8 * len(coords):
        return coords
    return kept


def _normalize(coords):
    if len(coords) < 3:
        return coords, 1.0
    scale = 1.0
    big = sum(1 for x, y in coords if abs(x) > 2 or abs(y) > 2)
    if big > len(coords) / 2:
        coords = [(x / 100, y / 100) for x, y in coords]
        scale = 0.01
    te_x = (coords[0][0] + coords[-1][0]) / 2
    te_y = (coords[0][1] + coords[-1][1]) / 2
    le_x = min(x for x, _ in coords)
    chord = te_x - le_x
    if chord < 1e-6:
        return coords, scale
    if abs(le_x) <= 5e-3 and abs(te_x - 1) <= 5e-3 and abs(te_y) <= 5e-3:
        return coords, scale
    coords = [((x - le_x) / chord, (y - te_y) / chord) for x, y in coords]
    return coords, scale / chord


def _scaled_precision(d, scale):
    if not math.isfinite(scale) or scale <= 0:
        return d
    v = d - math.log10(scale)
    r = round(v)
    return max(0, int(r if abs(v - r) < 0.05 else math.floor(v)))


def parse_airfoil_text(text, fname=None):
    filename_name = fname
    text = text.lstrip("﻿")
    kept = []
    for line in text.splitlines():
        content = line.split("#", 1)[0].strip()
        if content:
            kept.append((content, line.strip()))
    if not kept:
        return (None, None, None, 0)
    content_name = filename_name
    name_line = kept[0][1]
    rows = []
    first_is_row = False
    for i, (content, _full) in enumerate(kept):
        parts = _number_fields(content)
        row = None
        if len(parts) >= 2:
            try:
                row = (float(parts[0]), float(parts[1]), parts[0], parts[1])
            except ValueError:
                row = None
        if row is not None:
            rows.append(row)
            if i == 0:
                first_is_row = True
        elif i == 0:
            content_name = _name_of_line(name_line)
    if first_is_row and not _is_lednicer(rows) and (
        _is_lednicer(rows[1:]) or _is_stray_first_row(rows)
    ):
        content_name = name_line
        rows = rows[1:]
    if _is_lednicer(rows):
        nu, nl = int(rows[0][0]), int(rows[0][1])
        rows = rows[1:]
        coords = _lednicer_to_selig([(r[0], r[1]) for r in rows], nu, nl)
    else:
        coords = [(r[0], r[1]) for r in rows]
    if not coords:
        return (None, None, None, 0)
    texts = []
    for r in rows:
        texts.extend([r[2], r[3]])
    prec = _file_precision(texts)
    coords = _filter_outliers(coords)
    coords, scale = _normalize(coords)
    coords = [(float(x), float(y)) for x, y in coords]
    return (filename_name, content_name, coords, _scaled_precision(prec, scale))


# ---------------------------------------------------------------- 4 geometry
def le_index(P):
    P = list(P)
    if not P:
        raise ValueError("Cannot find leading edge in empty coordinate list")
    tx = (P[0][0] + P[-1][0]) / 2
    ty = (P[0][1] + P[-1][1]) / 2
    best, idx = -1, 0
    for i, (x, y) in enumerate(P):
        d = (x - tx) ** 2 + (y - ty) ** 2
        if d > best:
            best, idx = d, i
    return idx


def _junctions(u, S):
    u = list(u)
    J = []
    for i in range(1, S):
        target = i / S
        best, bi = None, 0
        for k, v in enumerate(u):
            d = abs(v - target)
            if best is None or d < best:
                best, bi = d, k
        J.append(bi)
    for k in range(len(J)):
        lo = 1 if k == 0 else J[k - 1] + 1
        J[k] = max(J[k], lo)
    for k in range(len(J) - 1, -1, -1):
        hi = len(u) - 2 if k == len(J) - 1 else J[k + 1] - 1
        J[k] = min(J[k], hi)
    return J


def _cut(points, J):
    secs = []
    prev = 0
    for j in J:
        secs.append(list(points[prev: j + 1]))
        prev = j
    secs.append(list(points[prev:]))
    return secs


def split(P, S):
    coords = [tuple(p) for p in P]
    if S < 1:
        raise ValueError(f"num_sections must be >= 1, got {S}")
    if len(coords) < S + 1:
        raise ValueError(f"Need at least {S + 1} points for {S} sections, got {len(coords)}")
    if S == 1:
        return [list(coords)], []
    if S == 2:
        le = min(max(le_index(coords), 1), len(coords) - 2)
        return [list(coords[: le + 1]), list(coords[le:])], [le]
    if S % 2 == 1:
        J = _junctions(arc_params(coords), S)
        return _cut(coords, J), J
    le = le_index(coords)
    upper = coords[: le + 1]
    lower = coords[le:]
    h = S // 2
    if len(upper) < h + 1 or len(lower) < h + 1:
        raise ValueError(
            f"Need at least {h + 1} points on each side of the leading edge for {S} sections, "
            f"got {len(upper)} and {len(lower)}"
        )
    Ju = _junctions(arc_params(upper), h)
    Jl = _junctions(arc_params(lower), h)
    secs = _cut(upper, Ju) + _cut(lower, Jl)
    return secs, Ju + [le] + [j + le for j in Jl]


# ---------------------------------------------------------------- 5.2
def min_ctrl(orders, p):
    seq = [0, *orders, 0]
    return max(p + 1, max(a + b + 2 for a, b in zip(seq[:-1], seq[1:])))


# ---------------------------------------------------------------- 7.1
def end_coefs(t, p, r):
    if r == 0:
        return np.array([1.0]), np.array([1.0])
    t = np.asarray(t, float)
    n = len(t) - p - 1
    spl = BSpline(t, np.eye(n), p)
    if r > p:
        raise ValueError(f"Derivative order {r} exceeds the spline degree {p}.")
    d = spl.derivative(r)
    end = np.asarray(d(t[-1])).ravel()
    start = np.asarray(d(t[0])).ravel()
    return end[-r - 1:].copy(), start[: r + 1].copy()


# ---------------------------------------------------------------- 7.3
ALPHA_RANGE = 10


def _alpha_range(ratio):
    lr = math.log(ratio)
    low = min(lr, 0.0) - math.log(ALPHA_RANGE)
    high = max(lr, 0.0) + math.log(ALPHA_RANGE)
    return (low + high) / 2, (high - low) / 2


def alpha(a, ratio):
    c, h = _alpha_range(ratio)
    th = math.tanh(a / h)
    al = math.exp(c + h * th)
    return al, al * (1 - th * th)


def alpha_param(al, ratio):
    c, h = _alpha_range(ratio)
    return h * math.atanh(float(np.clip((math.log(al) - c) / h, -0.999, 0.999)))


# ---------------------------------------------------------------- 8.2
def softplus(x):
    x = np.asarray(x, float)
    with np.errstate(over="ignore"):
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -700, 20))))


def sigmoid(x):
    x = np.asarray(x, float)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def inv_softplus(y):
    y = np.maximum(np.asarray(y, float), 1e-15)
    return np.where(y > 20, y, np.log(np.expm1(np.clip(y, 1e-15, 20))))


def t_to_eta(u):
    u = np.asarray(u, float)
    g = np.diff(u)
    floor = max(1e-3 * float(np.mean(g)), 1e-15) if len(g) else 1e-15
    return inv_softplus(np.maximum(g, floor))


def eta_to_t(eta, S):
    eta = np.nan_to_num(np.asarray(eta, float), nan=0.0)
    eta = np.clip(eta, -700, 700)
    c = np.cumsum(softplus(eta))
    if len(c) == 0:
        return np.zeros(1)
    return np.concatenate([[0.0], (S * c) / c[-1]])


# ---------------------------------------------------------------- 8.1 chain helpers
def _shared(sections):
    allp = np.vstack(sections)
    E = float(np.max(np.ptp(allp, axis=0)))
    out = [False]
    for s in range(1, len(sections)):
        diff = np.max(np.abs(sections[s - 1][-1] - sections[s][0]))
        out.append(bool(diff <= 1e-12 * max(E, 1e-300)))
    return out


def _chain_data(sections):
    sh = _shared(sections)
    parts = [sections[0]]
    for s in range(1, len(sections)):
        parts.append(sections[s][1:] if sh[s] else sections[s])
    return np.vstack(parts)


def _chain_parameters(t_sections, sections):
    sh = _shared(sections)
    out = []
    for s, t in enumerate(t_sections):
        t = np.array(t, float)
        if s > 0 and sh[s]:
            t = t[1:]
        elif s > 0 and len(t) > 1:
            t[0] = t[1] / 2
        out.append(t + s)
    return np.concatenate(out)


def _locate(u, S):
    u = np.clip(np.asarray(u, float), 0, S)
    sec = np.minimum(np.floor(u), S - 1).astype(int)
    return sec, u - sec


def _polyline_length(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


# ---------------------------------------------------------------- 7.4 build
def _theta_slots(juncs):
    slots = []  # (junction, kind)
    for j, (order, geometric, _fixed) in enumerate(juncs):
        if geometric and order >= 1:
            slots.append((j, "a"))
            if order >= 2:
                slots.append((j, "b"))
            if order >= 3:
                slots.append((j, "g"))
    return slots


def _ratios(sections):
    L = []
    for sec in sections:
        l = _polyline_length(sec)
        L.append(l if l != 0 else 1.0)
    return [L[j] / L[j + 1] for j in range(len(sections) - 1)]


def build(sections, knots, p, juncs, theta):
    sections = [np.asarray(s, float) for s in sections]
    knots = [np.asarray(t, float) for t in knots]
    S = len(sections)
    if len(juncs) != S - 1:
        raise ValueError("Need one junction per pair of consecutive sections.")
    ns = [len(t) - p - 1 for t in knots]
    offs = np.concatenate([[0], np.cumsum(ns)]).astype(int)
    T = int(offs[-1])
    dim = sections[0].shape[1]
    ends = [0] + [int(o) for o, _, _ in juncs] + [0]
    for j, (order, _, _) in enumerate(juncs):
        if order > p:
            raise ValueError(f"Junction {j}: continuity order exceeds the degree.")
    for s in range(S):
        need = max(p + 1, ends[s] + ends[s + 1] + 2)
        if ns[s] < need:
            raise ValueError(f"Section {s} needs at least {need} control points for its end conditions.")
    ratios = _ratios(sections)
    slots = _theta_slots(juncs)
    theta = np.asarray(theta, float)
    shape = [(1.0, 0.0, 0.0)] * (S - 1)
    shape = [list(v) for v in shape]
    for k, (j, kind) in enumerate(slots):
        if kind == "a":
            shape[j][0] = alpha(float(theta[k]), ratios[j])[0]
        elif kind == "b":
            shape[j][1] = float(theta[k])
        else:
            shape[j][2] = float(theta[k])
    rows, vals = [], []

    def row():
        return np.zeros(T)

    # 1, 2 fixed ends
    r = row(); r[offs[0]] = 1.0; rows.append(r); vals.append(sections[0][0])
    r = row(); r[T - 1] = 1.0; rows.append(r); vals.append(sections[-1][-1])
    for j, (order, geometric, fixed) in enumerate(juncs):
        a, b = j, j + 1
        ea = offs[a + 1] - 1
        sb = offs[b]
        r = row(); r[ea] = 1.0; r[sb] = -1.0; rows.append(r); vals.append(np.zeros(dim))
        if fixed:
            r = row(); r[ea] = 1.0; rows.append(r); vals.append(sections[a][-1])
        if order >= 1:
            al, be, ga = shape[j] if geometric else (1.0, 0.0, 0.0)
            e_r, s_r = {}, {}
            for rr in range(1, order + 1):
                ce, _ = end_coefs(knots[a], p, rr)
                _, cs = end_coefs(knots[b], p, rr)
                e = row(); e[ea - rr: ea + 1] = ce
                s_ = row(); s_[sb: sb + rr + 1] = cs
                e_r[rr], s_r[rr] = e, s_
            rows.append(e_r[1] - al * s_r[1]); vals.append(np.zeros(dim))
            if order >= 2:
                rows.append(e_r[2] - al ** 2 * s_r[2] - be * s_r[1]); vals.append(np.zeros(dim))
            if order >= 3:
                rows.append(e_r[3] - al ** 3 * s_r[3] - 3 * al * be * s_r[2] - ga * s_r[1])
                vals.append(np.zeros(dim))
    C = np.array(rows) if rows else np.zeros((0, T))
    d = np.array(vals, float) if vals else np.zeros((0, dim))
    return C, d, slots


# ---------------------------------------------------------------- 8.4 residual
def _matrix_root(Q):
    try:
        L = np.linalg.cholesky(Q)
        return L.T
    except np.linalg.LinAlgError:
        lam, V = np.linalg.eigh(Q)
        return np.diag(np.sqrt(np.maximum(lam, 0))) @ V.T


def _kkt_solve(A, C, reg, rhs, dvals):
    T = A.shape[0]
    m = C.shape[0]
    K = np.zeros((T + m, T + m))
    K[:T, :T] = A + reg * np.eye(T)
    K[:T, T:] = C.T
    K[T:, :T] = C
    full = np.vstack([rhs, dvals])
    lu, piv = lu_factor(K, check_finite=False)
    dg = np.abs(np.diag(lu))
    size = K.shape[0]
    mn = dg.min() if dg.size else 1.0
    mx = dg.max() if dg.size else 1.0
    if not np.all(np.isfinite(dg)) or mn <= size * EPS * mx:
        sol = np.linalg.pinv(K, rcond=1e-12) @ full
    else:
        sol = lu_solve((lu, piv), full)
    return sol[:T], sol[T:]


def varpro_residual(sections, knots, p, juncs, smoothing, reg, u, theta):
    sections = [np.asarray(s, float) for s in sections]
    knots = [np.asarray(t, float) for t in knots]
    S = len(sections)
    D = _chain_data(sections)
    N_pts, dim = D.shape
    ns = [len(t) - p - 1 for t in knots]
    offs = np.concatenate([[0], np.cumsum(ns)]).astype(int)
    T = int(offs[-1])
    Sb = np.zeros((T, T))
    for s in range(S):
        Sb[offs[s]:offs[s + 1], offs[s]:offs[s + 1]] = bending_energy(knots[s], p)
    Q = reg * np.eye(T)
    if smoothing > 0:
        Q = Q + smoothing * Sb
    R = _matrix_root(Q)
    c = D.mean(axis=0)
    spread = float(np.mean(np.sum((D - c) ** 2, axis=1)))
    alpha_pull = math.sqrt(smoothing * 3e-5 * spread) if smoothing > 0 else 0.0
    # speed rows
    chord = np.linalg.norm(np.diff(D, axis=0), axis=1)
    moving = chord > 1e-12 * max(chord.max() if chord.size else 0.0, 1e-300)
    rref = _chain_parameters([centripetal(sec) for sec in sections], sections)
    secg = np.minimum(np.floor((rref[:-1] + rref[1:]) / 2), S - 1).astype(int)
    pairs = [i for i in range(N_pts - 2)
             if moving[i] and moving[i + 1] and secg[i] == secg[i + 1]]
    pairs = np.array(pairs, int)
    log_chord = np.where(moving, np.log(np.where(moving, chord, 1.0)), 0.0)

    def change(uu):
        g = np.maximum(np.diff(uu), 1e-300)
        ls = np.log(g) - log_chord
        return ls[pairs + 1] - ls[pairs]

    reference = change(rref)
    speed_pull = 1e-3 * math.sqrt(spread)

    # solve
    u = np.asarray(u, float)
    sec, loc = _locate(u, S)
    Nm = np.zeros((N_pts, T))
    for s in range(S):
        idx = np.nonzero(sec == s)[0]
        if idx.size:
            Nm[np.ix_(idx, np.arange(offs[s], offs[s + 1]))] = _basis(loc[idx], knots[s], p)
    A = Nm.T @ Nm + smoothing * Sb
    C, d, slots = build(sections, knots, p, juncs, theta)
    oneC = np.tile(c, (T, 1))
    rhs = Nm.T @ (D - c)
    dv = d - C @ oneC
    O, _mu = _kkt_solve(A, C, reg, rhs, dv)
    P = O + oneC
    res = [(Nm @ P - D).ravel(), (R @ (P - oneC)).ravel()]
    if alpha_pull > 0:
        ratios = _ratios(sections)
        th = np.asarray(theta, float)
        pull = [alpha_pull * math.log(alpha(float(th[k]), ratios[j])[0] / ratios[j])
                for k, (j, kind) in enumerate(slots) if kind == "a"]
        res.append(np.array(pull, float))
    if speed_pull > 0 and len(pairs):
        excess = change(u) - reference
        rows = np.where(np.abs(excess) > 1.0, speed_pull * (excess - np.sign(excess) * 1.0), 0.0)
        res.append(rows)
    return np.concatenate(res)


# ---------------------------------------------------------------- 8.5 LM
_LM_MESSAGES = {
    -1: "The residual or Jacobian is not finite.",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
}


def _lm_run(residual, jacobian, x0, max_nfev, tol):
    x = np.array(x0, float)
    r = np.asarray(residual(x), float)
    nfev = 1
    cost = 0.5 * float(r @ r)
    fd = isinstance(jacobian, str)

    def affordable(e):
        return max_nfev is None or nfev + e <= max_nfev

    def linearize(x, r):
        nonlocal nfev
        if fd:
            J = np.empty((len(r), len(x)))
            for j in range(len(x)):
                h = math.sqrt(EPS) * max(1.0, abs(x[j]))
                xp = x.copy(); xp[j] += h
                J[:, j] = (np.asarray(residual(xp), float) - r) / h
            nfev += len(x)
            return J
        return np.asarray(jacobian(x), float)

    if fd and not affordable(len(x)):
        return x, cost, r, nfev, 0
    J = linearize(x, r)
    A = J.T @ J
    g = J.T @ r
    scale = np.maximum(np.diag(A), 1e-300)
    mu, nu = 1e-3, 2.0
    status = 0
    while affordable(1):
        if not (np.isfinite(cost) and np.all(np.isfinite(A)) and np.all(np.isfinite(g))):
            status = -1
            break
        rn = math.sqrt(2 * cost)
        coln = np.sqrt(np.maximum(np.diag(A), 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.where(coln > 0, np.abs(g) / (coln * rn), 0.0) if rn > 0 else np.zeros_like(g)
        if rn == 0 or (cos.max() if cos.size else 0.0) <= tol:
            status = 1
            break
        try:
            cf = cho_factor(A + mu * np.diag(scale), lower=False)
            delta = -cho_solve(cf, g)
            ok = np.all(np.isfinite(delta))
        except np.linalg.LinAlgError:
            ok = False
        if not ok:
            mu *= nu
            nu *= 2
            if mu > 1e20:
                status = 3
                break
            continue
        ss = np.sqrt(scale)
        if np.linalg.norm(ss * delta) <= tol * (tol + np.linalg.norm(ss * x)):
            status = 3
            break
        xn = x + delta
        rnew = np.asarray(residual(xn), float)
        nfev += 1
        costn = 0.5 * float(rnew @ rnew)
        pred = -(float(g @ delta) + 0.5 * float(delta @ (A @ delta)))
        rho = (cost - costn) / pred if pred > 0 else -1.0
        if rho > 0 and np.isfinite(costn):
            actual = (cost - costn) / cost if cost > 0 else 0.0
            x, r, cost = xn, rnew, costn
            if actual <= tol and pred / max(costn, 1e-300) <= tol:
                status = 2
                break
            if fd and not affordable(len(x)):
                status = 0
                break
            J = linearize(x, r)
            A = J.T @ J
            g = J.T @ r
            scale = np.maximum(scale, np.diag(A))
            mu *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
            nu = 2.0
        else:
            mu *= nu
            nu *= 2
            if mu > 1e20:
                status = 3
                break
    return x, cost, r, nfev, status


def levenberg_marquardt(res, jac, x0, max_nfev=None, tol=1e-10, max_restarts=8):
    if max_nfev is not None and max_nfev < 1:
        raise ValueError("max_nfev must be at least 1 or None.")
    if not tol > 0:
        raise ValueError("tolerance must be positive.")
    used = 0
    best = math.inf
    x = np.array(x0, float)
    out = None
    for _ in range(1 + max_restarts):
        budget = None if max_nfev is None else max_nfev - used
        x, cost, _r, nfev, status = _lm_run(res, jac, x, budget, tol)
        used += nfev
        improved = cost < 0.99 * best
        best = min(best, cost)
        out = (x, cost, status)
        if not (status == 3 and improved and (max_nfev is None or used < max_nfev)):
            break
    x, cost, status = out
    return x, cost, used, status, _LM_MESSAGES[status]


# ---------------------------------------------------------------- 9.3
def _spans(t, u):
    breaks = np.unique(np.asarray(t, float))
    idx = np.clip(np.searchsorted(breaks, u, "right") - 1, 0, len(breaks) - 2)
    return breaks, idx


def new_knots(t, u, dev, tol, min_span):
    u = np.asarray(u, float)
    dev = np.asarray(dev, float)
    breaks, sp = _spans(t, u)
    keep = []
    for i in range(len(breaks) - 1):
        a, b = breaks[i], breaks[i + 1]
        m = sp == i
        if not np.any(dev[m] > tol):
            continue
        if m.sum() < 2 or b - a < 2 * min_span:
            continue
        us, ds = u[m], dev[m]
        j = int(np.argmax(ds))
        v = float(np.clip(us[j], a + (b - a) / 4, b - (b - a) / 4))
        if not (np.any(us < v) and np.any(us > v)):
            v = float(np.median(us))
        if v - a >= min_span and b - v >= min_span:
            keep.append((float(ds.max()), v))
    keep.sort(reverse=True)
    return [v for _, v in keep]


def removal_order(t, p, u, dev, tol):
    u = np.asarray(u, float)
    dev = np.asarray(dev, float)
    breaks, sp = _spans(t, u)
    out = []
    for k in np.unique(_interior(t, p)):
        i = int(np.searchsorted(breaks, k))
        m = (sp == i - 1) | (sp == i)
        score = (float(dev[m].max()) if m.any() else 0.0) / tol
        out.append((score, float(k)))
    out.sort()
    return [k for _, k in out]


# ---------------------------------------------------------------- 9.7 PAV
def pav(v):
    blocks = []
    for x in np.asarray(v, float):
        blocks.append([float(x), 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            m2, c2 = blocks.pop()
            m1, c1 = blocks.pop()
            blocks.append([(m1 * c1 + m2 * c2) / (c1 + c2), c1 + c2])
    out = []
    for m, c in blocks:
        out.extend([m] * c)
    return np.array(out, float)


# ---------------------------------------------------------------- 12.6
def decimal_places(P):
    v = np.asarray(P, float).ravel()
    lim = 1e-12 * max(1.0, float(np.max(np.abs(v))) if v.size else 0.0)
    for d in range(11):
        if np.all(np.abs(v - np.round(v, d)) <= lim):
            return d
    return None


def _noise_level(P):
    P = np.asarray(P, float)
    if len(P) < 9:
        return 0.0
    rs = []
    for s in range(len(P) - 8):
        w = P[s: s + 9]
        centre = w[4]
        vec = w[8] - w[0]
        length = float(np.linalg.norm(vec))
        if length == 0:
            continue
        e = vec / length
        nrm = np.array([-e[1], e[0]])
        x = ((w - centre) @ e) / length
        y = (w - centre) @ nrm
        if not np.all(np.diff(x) > 0):
            continue
        V = np.vander(x, 7, increasing=True)
        H = V @ np.linalg.pinv(V)
        rs.append((y[4] - H[4] @ y) / math.sqrt(max(1 - H[4, 4], 1e-12)))
    if not rs:
        return 0.0
    return 1.4826 * float(np.median(np.abs(rs)))


def default_tolerance(P, decimals=None):
    P = np.asarray(P, float)
    chord = float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1])))
    tol = 1e-6 * chord
    sigma = _noise_level(P)
    d = decimals if decimals is not None else decimal_places(P)
    if d is not None:
        tol = max(tol, 10.0 ** -d)
        if sigma <= 3 * 10.0 ** -d / math.sqrt(12):
            return tol
    return max(tol, (math.sqrt(2 * math.log(max(len(P), 2))) + 0.5) * sigma)


# ---------------------------------------------------------------- 13.1 NACA
def naca(code, n=81, closed=True):
    code = str(code)
    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA 4-digit code must have four digits.")
    m = int(code[0]) / 100
    pp = int(code[1]) / 10
    t = int(code[2:]) / 100
    a4 = -0.1036 if closed else -0.1015
    sig = np.sin(np.linspace(0, np.pi, n) / 2)

    def surface(s, upper):
        x = s ** 2
        yt = 5 * t * (0.2969 * s - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 + a4 * x ** 4)
        if m == 0 or pp == 0:
            yc = np.zeros_like(x)
            slope = np.zeros_like(x)
        else:
            front = x < pp
            yc = np.where(front, m / pp ** 2 * (2 * pp * x - x ** 2),
                          m / (1 - pp) ** 2 * ((1 - 2 * pp) + 2 * pp * x - x ** 2))
            slope = np.where(front, 2 * m / pp ** 2 * (pp - x), 2 * m / (1 - pp) ** 2 * (pp - x))
        th = np.arctan(slope)
        sgn = 1.0 if upper else -1.0
        return np.column_stack([x - sgn * yt * np.sin(th), yc + sgn * yt * np.cos(th)])

    up = surface(sig, True)[::-1]
    lo = surface(sig[1:], False)
    return np.vstack([up, lo])


# ---------------------------------------------------------------- 14.2
def spline_text(degree, secs, name=None, precision=None, knots=None, continuity=None):
    secs = [list(s) for s in (secs or [])]
    if not secs or all(len(s) == 0 for s in secs):
        raise ValueError("No control points to export.")
    d = 8 if precision is None else precision
    d = max(0, min(int(d), 16))
    if knots is None:
        knots = [open_uniform_knots(int(degree), len(s)) for s in secs]
    if len(knots) != len(secs):
        raise ValueError("Need one knot vector per section.")
    for k, s in zip(knots, secs):
        if len(k) != len(s) + degree + 1:
            raise ValueError("Knot vector length must be control points + degree + 1.")
    lines = []
    if name:
        lines.append(_name_line(name))
    lines.append("# spline-fitter B-spline v2")
    lines.append(f"# degree {int(degree)}")
    lines.append(f"# sections {len(secs)}")
    if continuity:
        lines.append("# continuity " + " ".join(str(c) for c in continuity))
    for i, (s, k) in enumerate(zip(secs, knots), start=1):
        lines.append(f"# section {i} control_points {len(s)}")
        lines.append(f"# knots {i} " + " ".join(repr(float(v)) for v in k))
    for i, s in enumerate(secs):
        pts = s if i == 0 else s[1:]
        for x, y in pts:
            lines.append(f"{x:.{d}f} {y:.{d}f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 14.4
def step_real(v):
    s = repr(float(v)).upper()
    if "E" in s:
        mant, expo = s.split("E", 1)
        if "." not in mant:
            mant += "."
        return mant + "E" + str(int(expo))
    if "." not in s:
        s += "."
    return s


def step_string(s):
    out = []
    for ch in str(s):
        o = ord(ch)
        if ch == "'":
            out.append("''")
        elif ch == "\\":
            out.append("\\\\")
        elif 32 <= o <= 126:
            out.append(ch)
        elif o <= 0xFFFF:
            out.append(f"\\X2\\{o:04X}\\X0\\")
        else:
            out.append(f"\\X4\\{o:08X}\\X0\\")
    return "'" + "".join(out) + "'"


# ---------------------------------------------------------------- 16
def num(v):
    if v != 0 and abs(v) < 1e-2:
        mant, expo = f"{v:.2e}".split("e")
        return mant.rstrip("0").rstrip(".") + "e" + str(int(expo))
    return f"{v:.3g}"


def output_name(stem, c, o, auto=False):
    parts = [str(stem), f"deg{c.degree}"]
    n = c.num_control_points
    if (o.tolerance is None or not o.grow_control_points) and n != 25:
        parts.append(f"n{n}")
    if c.num_sections != 2:
        parts.append(f"s{c.num_sections}")
    specs = list(c.junction_specs())
    if specs and not all(s == "G3" for s in specs):
        parts.append(specs[0] if len(set(specs)) == 1 else "-".join(specs))
    opt = getattr(o.optimizer, "value", o.optimizer)
    if opt != "gn-varpro":
        parts.append("lsq" if opt == "direct-lsq" else str(opt))
        if o.parameterization != "centripetal":
            parts.append(o.parameterization.replace("_", "-"))
    if o.tolerance is not None:
        parts.append("tol-auto" if auto else f"tol{num(o.tolerance)}")
    if o.knot_placement != "uniform":
        parts.append(f"knots-{o.knot_placement}")
    if o.grow_control_points:
        parts.append("grow")
    if o.smoothing_weight != 1e-3:
        parts.append(f"smooth{num(o.smoothing_weight)}")
    if o.smoothing_factor is not None:
        parts.append(f"sfactor{num(o.smoothing_factor)}")
    if o.max_points_per_section is not None:
        parts.append(f"sub{o.max_points_per_section}")
    if o.optimize_le_position:
        parts.append("free-le")
    if o.early_stop_patience != 50:
        parts.append(f"patience{o.early_stop_patience}")
    if o.relative_tolerance != 1e-6:
        parts.append(f"rtol{num(o.relative_tolerance)}")
    return ".".join(parts) + ".spline.dat"


# ---------------------------------------------------------------- 18
_DECIMAL_POINT = "."


def _number(v, spec="g"):
    return format(v, spec).replace(".", _DECIMAL_POINT)


def format_error(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return str(f)
    if 0 < abs(f) < 1e-3 or abs(f) >= 1e4:
        return _number(f, ".3e")
    return _number(f, ".6f")


def format_duration(s):
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.0f} ms"
    if s < 60:
        return _number(s, ".2f") + " s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {int(sec)}s"
    h, m = divmod(int(m), 60)
    return f"{h}h {m}m {int(sec)}s"


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9\-_.]", "_", str(s))


def file_stem(s, default="airfoil"):
    out = "".join("_" if (ch in '<>:"/\\|?*' or ord(ch) < 32) else ch for ch in str(s))
    out = out.strip().rstrip(". ")
    return out or default


# ---------------------------------------------------------------- 19.17.2
_OPS = r"(<=|>=|!=|<|>|=)"


def parse_filter(text):
    s = re.sub(_OPS + r"\s+", r"\1", str(text).strip())
    terms = []
    for w in s.split():
        m = re.match(r"^" + _OPS + r"(.+)$", w)
        if m:
            op, operand = m.group(1), m.group(2)
            try:
                val = float(operand.replace(_DECIMAL_POINT, "."))
                terms.append((op, val))
                continue
            except ValueError:
                pass
            if op in ("=", "!="):
                terms.append((op, operand.casefold()))
                continue
        terms.append(("", w.casefold()))
    return terms


# ---------------------------------------------------------------- 19.21
def ladder_next(v, up, single, minimum, maximum=math.inf):
    # SPEC-GAP: 19.21 item 6 "clamped to the range": the maximum is not an input of
    # this function (task signature); +inf assumed. Single ladder step (k = 1).
    if v <= 0:
        nv = single if up else 0.0
    else:
        e = math.floor(math.log10(v))
        ladder = sorted(m * 10.0 ** k for k in (e - 1, e, e + 1) for m in (1.0, 2.0, 5.0))
        if up:
            nv = min(x for x in ladder if x > v * (1 + 1e-9))
        else:
            below = [x for x in ladder if x < v * (1 - 1e-9)]
            nv = max(below) if below else minimum
            if nv < minimum:
                nv = minimum
    return float(min(max(nv, minimum), maximum))


def sweep_values(v, end, step, limit=1000):
    low, high = sorted((v, end))
    count = min(math.floor((high - low) / step * (1 + 1e-9)) + 1, limit)
    return [float(f"{low + i * step:.12g}") for i in range(count)]


# ---------------------------------------------------------------- 19.19.1
def grey(L, T):
    hue = math.radians(95 if T >= 0 else 255)
    chroma = abs(T) * 0.035 * L * min(1, (1 - L) * 20)
    a, b = chroma * math.cos(hue), chroma * math.sin(hue)
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )
    out = []
    for ch in rgb:
        ch = min(max(ch, 0.0), 1.0)
        ch = 12.92 * ch if ch <= 0.0031308 else 1.055 * ch ** (1 / 2.4) - 0.055
        out.append(round(ch * 255))
    return "#" + "".join(f"{v:02X}" for v in out)
