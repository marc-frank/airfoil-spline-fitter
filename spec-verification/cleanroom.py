"""Clean-room implementations written from spec.md only (sections cited)."""
import math, re
import numpy as np
from scipy.interpolate import BSpline


# ---------------- 2.1 / 2.3 / 2.6 ----------------
def open_uniform_knots(p, n):
    if p < 1:
        raise ValueError(f"Degree must be at least 1, got {p}.")
    if n <= p:
        raise ValueError(f"Number of control points ({n}) must be greater than degree ({p}).")
    k = n - p - 1
    return np.array([0.0] * (p + 1) + [i * (1.0 / (n - p)) for i in range(1, k + 1)] + [1.0] * (p + 1))


def clamped_knots(p, interior=()):
    return np.concatenate([np.zeros(p + 1), np.sort(np.asarray(interior, float).ravel()), np.ones(p + 1)])


def bending_energy(t, p):
    t = np.asarray(t, float); n = len(t) - p - 1
    S = np.zeros((n, n))
    if p < 2:
        return S
    xi, w = np.polynomial.legendre.leggauss(8)
    d2 = BSpline(t, np.eye(n), p, extrapolate=False).derivative(2)
    br = np.unique(t)
    for a, b in zip(br[:-1], br[1:]):
        if b - a < 1e-14:
            continue
        u = (a + b) / 2 + (b - a) / 2 * xi
        B = d2(u)
        S += B.T @ (B * (w * (b - a) / 2)[:, None])
    tr = np.trace(S)
    if tr > 1e-15:
        S *= n / tr
    return S


def centripetal(P, a=0.5):
    P = np.asarray(P, float)
    if len(P) < 2:
        return np.zeros(len(P))
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1) ** a)])
    return np.linspace(0, 1, len(P)) if t[-1] <= 0 else t / t[-1]


def arc_params(P):
    if not P:
        return []
    if len(P) == 1:
        return [0.0]
    c = [0.0]; tot = 0.0
    for a, b in zip(P[:-1], P[1:]):
        tot += math.hypot(b[0] - a[0], b[1] - a[1]); c.append(tot)
    if tot <= 0:
        return [i / (len(P) - 1) for i in range(len(P))]
    return [x / tot for x in c]


# ---------------- 3 parser ----------------
def get_precision(s):
    s = s.lower().replace("d", "e")
    m, _, e = s.partition("e")
    dec = len(m.split(".")[1]) if "." in m else 0
    try:
        return dec - int(e or 0)
    except ValueError:
        return dec


def number_fields(c):
    c = c.replace(";", " ")
    c = re.sub(r"(?<=[0-9.])(?=[+-][0-9.])", " ", c)
    if "," in c:
        if len(c.split()) >= 2 and "." not in c:
            c = c.replace(",", ".")
        else:
            c = c.replace(",", " ")
    return [re.sub(r"(?<=[0-9.])[dD](?=[+-]?[0-9])", "e", f.strip("()\"'")) for f in c.split()]


def is_counts(x, y):
    return x >= 2 and y >= 2 and x == int(x) and y == int(y)


def is_lednicer(rows):
    if len(rows) <= 4 or not is_counts(rows[0][0], rows[0][1]):
        return False
    N = len(rows) - 1
    if abs(rows[0][0] + rows[0][1] - N) > max(2, N // 10):
        return False
    xs = [r[0] for r in rows[1:]]
    return xs[0] - min(xs) < max(xs) - xs[0]


def is_stray(rows):
    if len(rows) <= 4:
        return False
    xs = [r[0] for r in rows[1:]]; ys = [r[1] for r in rows[1:]]
    m = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys))
    x, y = rows[0][0], rows[0][1]
    return not (min(xs) - m <= x <= max(xs) + m and min(ys) - m <= y <= max(ys) + m)


def name_of_line(line):
    return line[1:] if re.match(r"^\\+#", line) else line


def parse_airfoil_text(text, fname=None):
    lines = []
    for line in text.lstrip("﻿").splitlines():
        c = line.split("#", 1)[0].strip()
        if c:
            lines.append((c, line.strip()))
    if not lines:
        return None, None, None, 0
    cname = fname; name_line = lines[0][1]
    rows = []; first = False
    for i, (c, _) in enumerate(lines):
        parts = number_fields(c)
        try:
            rows.append((float(parts[0]), float(parts[1]), parts[0], parts[1]))
            if i == 0:
                first = True
        except (ValueError, IndexError):
            if i == 0:
                cname = name_of_line(name_line)
    if first and not is_lednicer(rows) and (is_lednicer(rows[1:]) or is_stray(rows)):
        cname = name_line; rows = rows[1:]
    if is_lednicer(rows):
        nu, nl = int(rows[0][0]), int(rows[0][1]); rows = rows[1:]
        co = [(r[0], r[1]) for r in rows]
        if nu + nl != len(co):
            steps = [co[i + 1][0] - co[i][0] for i in range(len(co) - 1)]
            nu = steps.index(min(steps)) + 1
        up, lo = co[:nu], co[nu:]
        if lo and lo[0] == up[0]:
            lo = lo[1:]
        coords = list(reversed(up)) + lo
    else:
        coords = [(r[0], r[1]) for r in rows]
    if not coords:
        return None, None, None, 0
    texts = [v for r in rows for v in (r[2], r[3])]
    plain = [get_precision(v) for v in texts if "e" not in v.lower().replace("d", "e")]
    sci = [get_precision(v) for v in texts if "e" in v.lower().replace("d", "e")]
    cand = ([max(plain)] if plain else []) + ([min(sci)] if sci else [])
    prec = max(0, min(cand)) if cand else 0
    # outliers
    if len(coords) >= 4:
        mp = sorted(coords)[len(coords) // 2]
        d = [math.sqrt((p[0] - mp[0]) ** 2 + (p[1] - mp[1]) ** 2) for p in coords]
        sd = sorted(d); q1, q3 = sd[len(d) // 4], sd[3 * len(d) // 4]
        keep = [p for p, di in zip(coords, d) if di <= q3 + 5 * (q3 - q1)]
        if len(keep) >= 0.8 * len(coords):
            coords = keep
    scale = 1.0
    if len(coords) >= 3:
        if sum(1 for x, y in coords if abs(x) > 2 or abs(y) > 2) > len(coords) / 2:
            coords = [(x / 100.0, y / 100.0) for x, y in coords]; scale = 0.01
        tex = (coords[0][0] + coords[-1][0]) / 2; tey = (coords[0][1] + coords[-1][1]) / 2
        le = min(x for x, _ in coords); ch = tex - le
        if ch >= 1e-6 and not (abs(le) <= 5e-3 and abs(tex - 1) <= 5e-3 and abs(tey) <= 5e-3):
            coords = [((x - le) / ch, (y - tey) / ch) for x, y in coords]; scale = scale / ch
    if scale > 0 and math.isfinite(scale):
        v = prec - math.log10(scale); r = round(v)
        prec = max(0, int(r if abs(v - r) < 0.05 else math.floor(v)))
    return fname, cname, coords, prec


# ---------------- 4.1 / 4.2 ----------------
def le_index(P):
    tx = (P[0][0] + P[-1][0]) / 2; ty = (P[0][1] + P[-1][1]) / 2
    best, bi = -1.0, 0
    for i, (x, y) in enumerate(P):
        d = (x - tx) ** 2 + (y - ty) ** 2
        if d > best:
            best, bi = d, i
    return bi


def junctions(u, S):
    J = [min(range(len(u)), key=lambda k: abs(u[k] - i / S)) for i in range(1, S)]
    for k in range(len(J)):
        J[k] = max(J[k], J[k - 1] + 1 if k else 1)
    for k in reversed(range(len(J))):
        J[k] = min(J[k], J[k + 1] - 1 if k + 1 < len(J) else len(u) - 2)
    return J


def split(P, S):
    if S < 1 or len(P) < S + 1:
        raise ValueError
    if S == 1:
        return [list(P)], []
    if S == 2:
        le = min(max(le_index(P), 1), len(P) - 2)
        return [P[:le + 1], P[le:]], [le]
    def cut(Q, J, off):
        secs, s0 = [], 0
        for j in J:
            secs.append(Q[s0:j + 1]); s0 = j
        secs.append(Q[s0:])
        return secs, [j + off for j in J]
    if S % 2:
        return cut(P, junctions(arc_params(P), S), 0)
    le = le_index(P); h = S // 2; up, lo = P[:le + 1], P[le:]
    if len(up) < h + 1 or len(lo) < h + 1:
        raise ValueError
    s1, j1 = (([up], []) if h == 1 else cut(up, junctions(arc_params(up), h), 0))
    s2, j2 = (([lo], []) if h == 1 else cut(lo, junctions(arc_params(lo), h), le))
    return s1 + s2, j1 + [le] + j2


# ---------------- 5.2 ----------------
def min_ctrl(orders, p):
    e = [0, *orders, 0]
    return max(p + 1, *(a + b + 2 for a, b in zip(e, e[1:])))


# ---------------- 7 constraints ----------------
def end_coefs(t, p, r):
    if r == 0:
        return np.ones(1), np.ones(1)
    n = len(t) - p - 1
    d = BSpline(t, np.eye(n), p).derivative(r)
    return d(t[-1])[n - r - 1:], d(t[0])[:r + 1]


def alpha_range(ratio):
    lo = min(math.log(ratio), 0) - math.log(10); hi = max(math.log(ratio), 0) + math.log(10)
    return (lo + hi) / 2, (hi - lo) / 2


def alpha(a, ratio):
    c, h = alpha_range(ratio); x = math.tanh(a / h); al = math.exp(c + h * x)
    return al, al * (1 - x * x)


def alpha_param(al, ratio):
    c, h = alpha_range(ratio)
    return h * math.atanh(float(np.clip((math.log(al) - c) / h, -0.999, 0.999)))


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -700, 20.0))))


def sigmoid(x):
    e = np.exp(-np.abs(x)); return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def inv_softplus(y):
    y = np.maximum(y, 1e-15); return np.where(y > 20, y, np.log(np.expm1(np.clip(y, 1e-15, 20))))


def build(sections, knots, p, juncs, theta, fix_ends=True):
    """juncs: list of (order, geometric, fixed). No tangents. Returns C, d, layout."""
    n = [len(t) - p - 1 for t in knots]; o = np.concatenate([[0], np.cumsum(n)]).astype(int); T = o[-1]
    L = lambda s: float(np.sum(np.linalg.norm(np.diff(sections[s], axis=0), axis=1))) or 1.0
    slots = {}; k = 0
    for j, (ordr, geo, _) in enumerate(juncs):
        if geo and ordr >= 1:
            for nm in ("alpha", "beta", "gamma")[:ordr]:
                slots[(j, nm)] = k; k += 1
    rows, vals = [], []
    def place(row, s, r, at_end, sc=1.0):
        ce, cs = end_coefs(knots[s], p, r)
        if at_end:
            row[o[s + 1] - r - 1:o[s + 1]] += sc * ce
        else:
            row[o[s]:o[s] + r + 1] += sc * cs
        return row
    z = lambda: np.zeros(T)
    if fix_ends:
        rows.append(place(z(), 0, 0, False)); vals.append(sections[0][0])
        rows.append(place(z(), len(n) - 1, 0, True)); vals.append(sections[-1][-1])
    for j, (ordr, geo, fixed) in enumerate(juncs):
        a, b = j, j + 1
        rows.append(place(place(z(), a, 0, True), b, 0, False, -1.0)); vals.append(np.zeros(2))
        if fixed:
            rows.append(place(z(), a, 0, True)); vals.append(sections[a][-1])
        if ordr == 0:
            continue
        if geo:
            al = alpha(theta[slots[(j, "alpha")]], L(j) / L(j + 1))[0]
            be = theta[slots[(j, "beta")]] if (j, "beta") in slots else 0.0
            ga = theta[slots[(j, "gamma")]] if (j, "gamma") in slots else 0.0
        else:
            al, be, ga = 1.0, 0.0, 0.0
        s1 = place(z(), b, 1, False); s2 = place(z(), b, 2, False) if ordr >= 2 else None
        s3 = place(z(), b, 3, False) if ordr >= 3 else None
        rows.append(place(z(), a, 1, True) - al * s1); vals.append(np.zeros(2))
        if ordr >= 2:
            rows.append(place(z(), a, 2, True) - al ** 2 * s2 - be * s1); vals.append(np.zeros(2))
        if ordr >= 3:
            rows.append(place(z(), a, 3, True) - al ** 3 * s3 - 3 * al * be * s2 - ga * s1); vals.append(np.zeros(2))
    return np.array(rows), np.array(vals), slots


# ---------------- 8.1/8.2/8.4 residual ----------------
def chain_parameters(ts, sections):
    pts = np.vstack(sections); E = max(float(np.max(np.ptp(pts, axis=0))), 1e-300)
    shared = [False] + [bool(np.max(np.abs(a[-1] - b[0])) <= 1e-12 * E) for a, b in zip(sections[:-1], sections[1:])]
    parts = []
    for s, (t, sh) in enumerate(zip(ts, shared)):
        t = np.array(t, float)
        if sh:
            t = t[1:]
        elif s > 0 and len(t) > 1:
            t[0] = t[1] / 2
        parts.append(t + s)
    return np.concatenate(parts), shared


def chain_data(sections, shared):
    return np.vstack([sections[0]] + [s[1:] if sh else s for s, sh in zip(sections[1:], shared[1:])])


def eta_to_t(eta, S):
    eta = np.clip(np.nan_to_num(eta, nan=0.0), -700, 700); c = np.cumsum(softplus(eta))
    return np.concatenate([[0.0], S * c / c[-1]])


def t_to_eta(u):
    g = np.diff(u); fl = max(1e-3 * float(np.mean(g)), 1e-15) if len(g) else 1e-15
    return inv_softplus(np.maximum(g, fl))


def matrix_root(A):
    try:
        return np.linalg.cholesky(A).T
    except np.linalg.LinAlgError:
        l, V = np.linalg.eigh(A); return np.sqrt(np.maximum(l, 0))[:, None] * V.T


def varpro_residual(sections, knots, p, juncs, smoothing, reg, u, theta):
    sections = [np.asarray(s, float) for s in sections]
    r0, shared = chain_parameters([centripetal(s) for s in sections], sections)
    D = chain_data(sections, shared); S = len(sections); Np = len(D)
    n = [len(t) - p - 1 for t in knots]; o = np.concatenate([[0], np.cumsum(n)]).astype(int); T = o[-1]
    Q = reg * np.eye(T); Sb = None
    if smoothing > 0:
        from scipy.linalg import block_diag
        Sb = block_diag(*[bending_energy(t, p) for t in knots]); Q = Q + smoothing * Sb
    R = matrix_root(Q); c = D.mean(axis=0)
    uu = np.clip(u, 0, S); sec = np.minimum(np.floor(uu).astype(int), S - 1); loc = uu - sec
    N = np.zeros((Np, T))
    for s in range(S):
        m = sec == s
        if m.any():
            N[np.ix_(np.flatnonzero(m), np.arange(o[s], o[s + 1]))] = BSpline.design_matrix(np.clip(loc[m], 0, 1), knots[s], p).toarray()
    A = N.T @ N + (smoothing * Sb if Sb is not None else 0)
    C, d, slots = build(sections, knots, p, juncs, theta)
    K = np.zeros((T + len(C),) * 2); K[:T, :T] = A + reg * np.eye(T); K[:T, T:] = C.T; K[T:, :T] = C
    one_c = np.tile(c, (T, 1))
    sol = np.linalg.solve(K, np.vstack([N.T @ (D - c), d - C @ one_c]))
    P = sol[:T] + one_c
    L = lambda s: float(np.sum(np.linalg.norm(np.diff(sections[s], axis=0), axis=1))) or 1.0
    spread = float(np.mean(np.sum((D - c) ** 2, axis=1)))
    ap = math.sqrt(smoothing * 3e-5 * spread) if smoothing > 0 else 0.0
    pull = []
    if ap > 0:
        for (j, nm), i in sorted(slots.items(), key=lambda kv: kv[1]):
            if nm == "alpha":
                ratio = L(j) / L(j + 1); pull.append(ap * math.log(alpha(theta[i], ratio)[0] / ratio))
    ch = np.linalg.norm(np.diff(D, axis=0), axis=1); mv = ch > 1e-12 * max(float(ch.max(initial=0)), 1e-300)
    gs = np.minimum(np.floor((r0[:-1] + r0[1:]) / 2), S - 1)
    pairs = np.flatnonzero(mv[:-1] & mv[1:] & (gs[:-1] == gs[1:]))
    lch = np.log(np.where(mv, ch, 1.0))
    def change(v):
        g = np.maximum(np.diff(v), 1e-300); ls = np.log(g) - lch; return ls[pairs + 1] - ls[pairs]
    sp = 1e-3 * math.sqrt(spread); ex = change(u) - change(r0)
    speed = np.where(np.abs(ex) > 1.0, ex - np.sign(ex) * 1.0, 0.0) * sp if sp > 0 else np.zeros(0)
    return np.concatenate([(N @ P - D).ravel(), (R @ (P - one_c)).ravel(), pull, speed])


# ---------------- 8.5 LM ----------------
MSG = {-1: "The residual or Jacobian is not finite.", 0: "The maximum number of function evaluations is exceeded.",
       1: "`gtol` termination condition is satisfied.", 2: "`ftol` termination condition is satisfied.",
       3: "`xtol` termination condition is satisfied."}


def lm_run(res, jac, x0, max_nfev, tol):
    from scipy.linalg import cho_factor, cho_solve
    x = np.array(x0, float); r = res(x); nfev = 1; cost = 0.5 * float(r @ r)
    aff = lambda e: max_nfev is None or nfev + e <= max_nfev
    J = jac(x); A = J.T @ J; g = J.T @ r; scale = np.maximum(np.diag(A), 1e-300); mu, nu = 1e-3, 2.0; st = 0
    while aff(1):
        if not (np.isfinite(cost) and np.all(np.isfinite(A)) and np.all(np.isfinite(g))):
            st = -1; break
        cn = np.sqrt(np.maximum(np.diag(A), 0)); rn = math.sqrt(2 * cost)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.where(cn > 0, np.abs(g) / (cn * rn), 0.0)
        if rn == 0 or np.max(cos, initial=0) <= tol:
            st = 1; break
        try:
            step = -cho_solve(cho_factor(A + mu * np.diag(scale)), g)
        except np.linalg.LinAlgError:
            step = None
        if step is None or not np.all(np.isfinite(step)):
            mu *= nu; nu *= 2
            if mu > 1e20:
                st = 3; break
            continue
        if np.linalg.norm(np.sqrt(scale) * step) <= tol * (tol + np.linalg.norm(np.sqrt(scale) * x)):
            st = 3; break
        xn = x + step; rnew = res(xn); nfev += 1; cn_ = 0.5 * float(rnew @ rnew)
        pred = -(g @ step + 0.5 * step @ (A @ step)); rho = (cost - cn_) / pred if pred > 0 else -1.0
        if rho > 0 and np.isfinite(cn_):
            act = (cost - cn_) / cost if cost > 0 else 0.0
            x, r, cost = xn, rnew, cn_
            if act <= tol and pred / max(cost, 1e-300) <= tol:
                st = 2; break
            J = jac(x); A = J.T @ J; g = J.T @ r; scale = np.maximum(scale, np.diag(A))
            mu *= max(1 / 3, 1 - (2 * rho - 1) ** 3); nu = 2.0
        else:
            mu *= nu; nu *= 2
            if mu > 1e20:
                st = 3; break
    return x, cost, nfev, st


def levenberg_marquardt(res, jac, x0, max_nfev=None, tol=1e-10, restarts=8):
    x = np.asarray(x0, float); used = 0; best = np.inf; out = None
    for _ in range(restarts + 1):
        x, cost, nf, st = lm_run(res, jac, x, None if max_nfev is None else max_nfev - used, tol)
        used += nf; imp = cost < 0.99 * best; best = min(best, cost); out = (x, cost, st)
        if st != 3 or not imp or (max_nfev is not None and used >= max_nfev):
            break
    return out[0], out[1], used, out[2], MSG[out[2]]


# ---------------- 9.3 / 9.7 ----------------
def new_knots(t, u, dev, tol, ms):
    br = np.unique(t); idx = np.clip(np.searchsorted(br, u, "right") - 1, 0, len(br) - 2); new = []
    for i in range(len(br) - 1):
        m = idx == i
        if not np.any(dev[m] > tol):
            continue
        a, b = br[i], br[i + 1]; ts, ds = u[m], dev[m]
        if len(ts) < 2 or b - a < 2 * ms:
            continue
        v = float(np.clip(ts[np.argmax(ds)], a + (b - a) / 4, b - (b - a) / 4))
        if not (np.any(ts < v) and np.any(ts > v)):
            v = float(np.median(ts))
        if v - a >= ms and b - v >= ms:
            new.append((float(np.max(ds)), v))
    return [v for _, v in sorted(new, reverse=True)]


def removal_order(t, p, u, dev, tol):
    br = np.unique(t); idx = np.clip(np.searchsorted(br, u, "right") - 1, 0, len(br) - 2); c = []
    for k in np.unique(np.asarray(t)[p + 1:len(t) - p - 1]):
        i = int(np.searchsorted(br, k)); m = (idx == i - 1) | (idx == i)
        c.append((float(np.max(dev[m], initial=0.0)) / tol, float(k)))
    return [k for _, k in sorted(c)]


def pav(v):
    blocks = []
    for x in v:
        blocks.append([float(x), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (m1, c1), (m2, c2) = blocks[-2], blocks[-1]
            blocks[-2:] = [[(m1 * c1 + m2 * c2) / (c1 + c2), c1 + c2]]
    return np.repeat([b[0] for b in blocks], [b[1] for b in blocks])


# ---------------- 12.6 ----------------
def decimal_places(P):
    v = np.asarray(P, float).ravel(); sc = max(1.0, float(np.max(np.abs(v))))
    for d in range(11):
        if np.all(np.abs(v - np.round(v, d)) <= 1e-12 * sc):
            return d
    return None


def noise_level(P):
    P = np.asarray(P, float)
    if len(P) < 9:
        return 0.0
    W = P[np.arange(len(P) - 8)[:, None] + np.arange(9)]
    ch = W[:, -1] - W[:, 0]; L = np.linalg.norm(ch, axis=1); k = L > 0
    W, ch, L = W[k], ch[k], L[k]; cen = W[:, 4:5]; e = ch / L[:, None]; nv = np.stack([-e[:, 1], e[:, 0]], 1)
    x = np.einsum("wkd,wd->wk", W - cen, e) / L[:, None]; y = np.einsum("wkd,wd->wk", W - cen, nv)
    mono = np.all(np.diff(x, axis=1) > 0, axis=1); x, y = x[mono], y[mono]
    if len(x) == 0:
        return 0.0
    V = x[:, :, None] ** np.arange(7); H = V @ np.linalg.pinv(V)
    r = (y[:, 4] - np.einsum("wk,wk->w", H[:, 4], y)) / np.sqrt(np.maximum(1 - H[:, 4, 4], 1e-12))
    return float(1.4826 * np.median(np.abs(r)))


def default_tolerance(P, decimals=None):
    P = np.asarray(P, float); tol = 1e-6 * float(np.max(np.ptp(P, axis=0))); s = noise_level(P)
    d = decimal_places(P) if decimals is None else decimals
    if d is not None:
        tol = max(tol, 10.0 ** -d)
        if s <= 3 * 10.0 ** -d / math.sqrt(12):
            return float(tol)
    return float(max(tol, (math.sqrt(2 * math.log(max(len(P), 2))) + 0.5) * s))


# ---------------- 13.1 NACA ----------------
def naca(code, n=81, closed=True):
    m, pp, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100; a4 = -0.1036 if closed else -0.1015
    def surf(sig, up):
        x = sig * sig; yt = 5 * t * (0.2969 * sig - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 + a4 * x ** 4) * (1 if up else -1)
        if m == 0 or pp == 0:
            yc = sl = np.zeros_like(x)
        else:
            f = x < pp
            yc = np.where(f, m / pp ** 2 * (2 * pp * x - x ** 2), m / (1 - pp) ** 2 * ((1 - 2 * pp) + 2 * pp * x - x ** 2))
            sl = np.where(f, 2 * m / pp ** 2 * (pp - x), 2 * m / (1 - pp) ** 2 * (pp - x))
        th = np.arctan(sl)
        return np.stack([x - yt * np.sin(th), yc + yt * np.cos(th)], -1)
    sig = np.sin(np.linspace(0, np.pi, n) / 2)
    return np.vstack([surf(sig[::-1], True), surf(sig[1:], False)])


# ---------------- 14 writers ----------------
def name_line(n):
    n = str(n).strip(); return "\\" + n if re.match(r"^\\*#", n) else n


def spline_text(degree, secs, name=None, precision=None, knots=None, continuity=None):
    d = 8 if precision is None else max(0, min(int(precision), 16))
    if knots is None:
        knots = [open_uniform_knots(degree, len(c)) for c in secs]
    out = []
    if name:
        out.append(name_line(name))
    out += ["# spline-fitter B-spline v2", f"# degree {int(degree)}", f"# sections {len(secs)}"]
    if continuity:
        out.append("# continuity " + " ".join(continuity))
    for i, (c, k) in enumerate(zip(secs, knots), 1):
        out.append(f"# section {i} control_points {len(c)}")
        out.append(f"# knots {i} " + " ".join(repr(float(v)) for v in k))
    for i, c in enumerate(secs):
        for x, y in c[(1 if i else 0):]:
            out.append(f"{float(x):.{d}f} {float(y):.{d}f}")
    return "\n".join(out) + "\n"


def step_real(v):
    s = repr(float(v)).upper()
    if "E" in s:
        m, e = s.split("E")
        return (m if "." in m else m + ".") + "E" + str(int(e))
    return s if "." in s else s + "."


def step_string(text):
    out = []
    for ch in text:
        c = ord(ch)
        if ch == "'":
            out.append("''")
        elif ch == "\\":
            out.append("\\\\")
        elif 32 <= c < 127:
            out.append(ch)
        elif c <= 0xFFFF:
            out.append("\\X2\\" + f"{c:04X}" + "\\X0\\")
        else:
            out.append("\\X4\\" + f"{c:08X}" + "\\X0\\")
    return "'" + "".join(out) + "'"


# ---------------- 16 / 18 ----------------
def num(v):
    if v != 0 and abs(v) < 1e-2:
        m, e = f"{v:.2e}".split("e"); return m.rstrip("0").rstrip(".") + "e" + str(int(e))
    return f"{v:.3g}"


def format_error(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(v):
        return str(v)
    return format(v, ".3e") if (0 < abs(v) < 1e-3 or abs(v) >= 1e4) else format(v, ".6f")


def format_duration(s):
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.0f} ms"
    if s < 60:
        return f"{s:.2f} s"
    m, sec = divmod(s, 60.0)
    if m < 60:
        return f"{int(m)}m {int(sec)}s"
    h, m = divmod(int(m), 60)
    return f"{h}h {m}m {int(sec)}s"


def file_stem(s, default="airfoil"):
    st = "".join("_" if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in s)
    return st.strip().rstrip(". ") or default


def sanitize(s):
    ok = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return "".join(c if c in ok else "_" for c in s)


# ---------------- 19.17.2 filters ----------------
def parse_filter(text, dp="."):
    text = re.sub(r"(<=|>=|!=|<|>|=)\s+", r"\1", text.strip()); terms = []
    for w in text.split():
        m = re.match(r"^(<=|>=|!=|<|>|=)(.+)$", w)
        num_ = None
        if m:
            try:
                num_ = float(m.group(2).replace(dp, "."))
            except ValueError:
                num_ = None
        if m and num_ is not None:
            terms.append((m.group(1), num_))
        elif m and m.group(1) in ("=", "!="):
            terms.append((m.group(1), m.group(2).casefold()))
        else:
            terms.append(("", w.casefold()))
    return terms


# ---------------- 19.21 widgets arithmetic ----------------
def ladder_next(v, up, single, minimum):
    """One step of the spin box, then clamped to the range (19.21 item 6); no maximum here."""
    return max(_ladder(v, up, single, minimum), minimum)


def _ladder(v, up, single, minimum):
    if v <= 0:
        return single if up else 0.0
    e = math.floor(math.log10(v)); L = [m * 10.0 ** x for x in (e - 1, e, e + 1) for m in (1.0, 2.0, 5.0)]
    if up:
        return min(x for x in L if x > v * (1 + 1e-9))
    lo = [x for x in L if x < v * (1 - 1e-9)]
    return max(lo) if lo and max(lo) >= minimum else minimum


def sweep_values(v, end, step, limit=1000):
    lo, hi = sorted((v, end)); cnt = min(int(math.floor((hi - lo) / step * (1 + 1e-9))) + 1, limit)
    return [float(f"{lo + i * step:.12g}") for i in range(cnt)]


def grey(L, T):
    hue = math.radians(95.0 if T >= 0 else 255.0); ch = abs(T) * 0.035 * L * min(1.0, (1 - L) * 20)
    a, b = ch * math.cos(hue), ch * math.sin(hue)
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3; m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = (4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_, -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
           -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_)
    def chn(c):
        c = min(max(c, 0.0), 1.0); c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055; return round(c * 255)
    return "#{:02X}{:02X}{:02X}".format(*(chn(c) for c in lin))
