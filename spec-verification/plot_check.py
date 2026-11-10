"""spec.md 19.14.3 (curvature comb), 12.1 (orthogonal projection), 12.5 (foot points)
and 19.14.5 (deviation whiskers, envelopes and tolerance band), implemented following
the text and compared with what the real PlotManager draws for random fits.
"""
import math, os, random, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "packages", "bspline-varpro", "src")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from scipy.interpolate import BSpline

rng = random.Random(611); nrng = np.random.default_rng(611)


# ---------- clean-room, from the spec ----------
def ev(c, u, nu=0):
    return BSpline(c[0], c[1], c[2], extrapolate=False)(u, nu=nu) if nu <= c[2] else np.zeros((len(np.atleast_1d(u)), 2))


def curv_slope(d1, d2, d3):  # 2.4, comb guard: 0 where v^2 <= 0
    v = d1[:, 0] ** 2 + d1[:, 1] ** 2; a = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    b = d1[:, 0] * d3[:, 1] - d1[:, 1] * d3[:, 0]; c = d1[:, 0] * d2[:, 0] + d1[:, 1] * d2[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, a / v ** 1.5, 0.0), np.where(v > 0, (b * v - 3 * a * c) / v ** 3, 0.0)


def cumtrap(f, t):
    return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])


def comb_parameters(c, n):
    t = np.linspace(0, 1, max(4000, 20 * n)); d1, d2 = ev(c, t, 1), ev(c, t, 2)
    sp = np.hypot(d1[:, 0], d1[:, 1])
    den = sp ** 3
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(den > 1e-12, (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / den, 0.0)   # 2.4 application guard
    s = cumtrap(sp, t); turn = cumtrap(np.abs(np.nan_to_num(k)) * sp, t)
    if s[-1] <= 0:
        return np.linspace(0, 1, n)
    m = s / s[-1]
    if turn[-1] > 0:
        m = 0.5 * m + 0.5 * turn / turn[-1]
    return np.interp(np.linspace(0, 1, n), m, t)


def comb_values(c, t, q):
    if q <= 1:
        k, sl = curv_slope(ev(c, t, 1), ev(c, t, 2), ev(c, t, 3)); return k if q == 0 else sl
    f = np.linspace(0, 1, max(4000, 20 * len(t))); d1 = ev(c, f, 1)
    _, val = curv_slope(d1, ev(c, f, 2), ev(c, f, 3)); s = cumtrap(np.hypot(d1[:, 0], d1[:, 1]), f)
    keep = np.concatenate([[True], np.diff(s) > 0]); f, s, val = f[keep], s[keep], val[keep]
    for _ in range(q - 1):
        val = np.gradient(val, s, edge_order=2)
    return np.interp(t, f, val)


def joined(lines):
    xs, ys = [], []
    for x, y in lines:
        if xs:
            xs.append([np.nan]); ys.append([np.nan])
        xs.append(np.asarray(x, float)); ys.append(np.asarray(y, float))
    return (np.concatenate(xs), np.concatenate(ys)) if xs else (np.empty(0), np.empty(0))


def teeth(base, tip):
    gap = np.full(len(base), np.nan)
    return np.column_stack([base[:, 0], tip[:, 0], gap]).ravel(), np.column_stack([base[:, 1], tip[:, 1], gap]).ravel()


def comb(curves, scale, density, q):
    density = max(5, density)
    chord = float(np.ptp(np.concatenate([ev(c, np.linspace(0, 1, 200))[:, 0] for c in curves]))) or 1.0
    pieces = {0: [], 1: []}
    for i, c in enumerate(curves):
        t = comb_parameters(c, density); base = ev(c, t); d1 = ev(c, t, 1); mag = np.hypot(d1[:, 0], d1[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            nrm = np.column_stack([np.where(mag > 1e-12, d1[:, 1] / mag, 0), np.where(mag > 1e-12, -d1[:, 0] / mag, 0)])
        val = comb_values(c, t, q) * scale * chord ** (q + 2)
        pieces[i % 2].append((base, base + val[:, None] * nrm))
    return {(p, tips): (joined([tp.T for _, tp in parts]) if tips else joined([teeth(b, tp) for b, tp in parts]))
            for p, parts in pieces.items() for tips in (False, True)}


def project(pts, c):  # 12.1
    t, P, k = c
    br = np.unique(t[k:len(t) - k])
    grid = np.concatenate([np.linspace(a, b, 33)[:-1] for a, b in zip(br[:-1], br[1:])] + [br[-1:]])
    S = ev(c, grid); lo_d, hi_d = grid[0], grid[-1]
    out_u, out_d = np.empty(len(pts)), np.empty(len(pts))
    for i, q in enumerate(pts):
        d2 = np.sum((S - q) ** 2, axis=1); pad = np.concatenate([[np.inf], d2, [np.inf]])
        loc = (d2 <= pad[:-2]) & (d2 <= pad[2:])
        ranked = np.argsort(np.where(loc, d2, np.inf), kind="quicksort")[:3]
        cands = []
        for j in ranked:
            if not loc[j]:
                continue
            lo, hi, u = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)], grid[j]
            for _ in range(40):
                c0, c1, c2 = ev(c, [u])[0], ev(c, [u], 1)[0], ev(c, [u], 2)[0]
                g = (c0 - q) @ c1; h = c1 @ c1 + (c0 - q) @ c2
                if g > 0:
                    hi = u
                elif g < 0:
                    lo = u
                with np.errstate(divide="ignore", invalid="ignore"):
                    nw = u - g / h
                un = nw if (h > 0 and lo < nw < hi) else 0.5 * (lo + hi)
                done = abs(un - u) <= 1e-15 * (1 + abs(u)); u = un
                if done:
                    break
            u = min(max(u, lo_d), hi_d); cands.append((float(np.sum((ev(c, [u])[0] - q) ** 2)), u))
        cands += [(float(np.sum((ev(c, [lo_d])[0] - q) ** 2)), lo_d), (float(np.sum((ev(c, [hi_d])[0] - q) ** 2)), hi_d)]
        best = min(range(len(cands)), key=lambda m: cands[m][0])
        out_u[i], out_d[i] = cands[best][1], math.sqrt(cands[best][0])
    return out_u, out_d


def foot_points(pts, curves):  # 12.5 (search form)
    u = np.column_stack([project(pts, c)[0] for c in curves])
    best = np.full(len(pts), np.inf); foot = np.zeros_like(pts); tan = np.zeros_like(pts); D = np.empty((len(pts), len(curves)))
    for i, c in enumerate(curves):
        at = ev(c, u[:, i]); d = D[:, i] = np.hypot(*(pts - at).T); cl = d < best
        best = np.where(cl, d, best); foot[cl] = at[cl]; tan[cl] = ev(c, u[cl, i], 1)
    nrm = np.column_stack([tan[:, 1], -tan[:, 0]]); nrm /= np.maximum(np.hypot(nrm[:, 0], nrm[:, 1]), 1e-300)[:, None]
    signed = np.copysign(best, np.sum((pts - foot) * nrm, axis=1))
    tie = 1e-9 * max(float(np.max(np.ptp(pts, axis=0), initial=0)), 1e-300)
    sec = np.maximum.accumulate(np.argmax(D <= best[:, None] + tie, axis=1))
    return signed, foot, nrm, sec


def deviation(pts, curves, height, tol):  # 19.14.5
    dev, foot, nrm, sec = foot_points(pts, curves)
    ref = tol or float(np.max(np.abs(dev))); chord = float(np.ptp(pts[:, 0]))
    scale = height * chord / ref; tips = foot + (scale * dev)[:, None] * nrm
    br = np.flatnonzero(np.diff(sec)) + 1; starts = np.concatenate([[0], br - 1]); ends = np.concatenate([br, [len(sec)]])
    pieces = {0: [], 1: []}
    for lo, hi in zip(starts, ends):
        pieces[int(sec[hi - 1]) % 2].append((foot[lo:hi], tips[lo:hi]))
    out = {(p, False): joined([teeth(b, t) for b, t in parts]) for p, parts in pieces.items()}
    out.update({(p, True): joined([t.T for _, t in parts]) for p, parts in pieces.items()})
    band = []
    if tol:
        for c in curves:
            u = np.linspace(c[0][c[2]], c[0][-c[2] - 1], 400); at, tg = ev(c, u), ev(c, u, 1)
            r = np.column_stack([tg[:, 1], -tg[:, 0]]); r /= np.maximum(np.hypot(r[:, 0], r[:, 1]), 1e-300)[:, None]
            band += [(at + scale * tol * r).T, (at - scale * tol * r).T]
    out["band"] = joined(band) if tol else (np.empty(0), np.empty(0))
    return out


# Expected geometry: this file's own (default), or comb(curves, scale, density, q) and
# deviation(points, curves, height, tolerance) of the module named by IMPL_PLOT (a path)
if os.environ.get("IMPL_PLOT"):
    import importlib.util as _u
    _s = _u.spec_from_file_location("impl_plot", os.environ["IMPL_PLOT"]); _impl = _u.module_from_spec(_s); _s.loader.exec_module(_impl)
    comb, deviation = _impl.comb, _impl.deviation

# ---------- the real plots ----------
from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from spline_fitter.gui.ui import SplineFitterWindow
from spline_fitter.optimization.models import FitResult, SplineConfig, OptimizationConfig, OptimizerType, ContinuityMode
from spline_fitter.optimization import run_optimization
from spline_fitter.geometry import create_airfoil_data
from bspline_varpro.airfoil import NACA4

w = SplineFitterWindow(); pm = w._plot_manager
results = []


def same(a, b, rtol=1e-9, atol=1e-12):
    # pyqtgraph returns None for an item without data
    a = tuple(np.empty(0) if x is None else np.asarray(x, float) for x in a)
    b = tuple(np.empty(0) if x is None else np.asarray(x, float) for x in b)
    # Elementwise, with an absolute floor of 1e-9 of the array's largest value: higher comb
    # derivatives are differentiated numerically and span many decades
    return all(x.shape == y.shape and np.allclose(
        x, y, rtol=rtol, atol=max(atol, 1e-9 * float(np.nanmax(np.abs(y), initial=0.0))), equal_nan=True)
        and np.array_equal(np.isnan(x), np.isnan(y)) for x, y in zip(a, b))


for case in range(24):
    code = rng.choice(["0012", "2412", "4415"]); S = rng.randint(1, 4); p = rng.choice([3, 4, 5])
    pts = [tuple(x) for x in np.round(NACA4(code).points(rng.randint(15, 30)), 6)]
    af = create_airfoil_data(pts, S, "f", "f", 6)
    opt = rng.choice(list(OptimizerType)); tol_goal = opt == OptimizerType.GN_VARPRO and rng.random() < 0.0
    r = run_optimization(SplineConfig(degree=p, num_control_points=rng.randint(p + 3, 14), num_sections=S,
                                      continuity=ContinuityMode.G2), OptimizationConfig(optimizer=opt), af)
    tol = rng.choice([None, 2e-4, 1e-5])
    fit = FitResult(r.section_control_points, p, r.fun, 1, 1, 0.0, section_knots=r.section_knots,
                    report=None if tol is None else {"tolerance": tol})
    curves = [(fit.knots(i), np.asarray(c, float), p) for i, c in enumerate(fit.section_control_points)]
    pm.plot_airfoil(pts); pm.comb_settings = None
    scale, density, q = rng.choice([0.05, 0.3, 1.0]), rng.choice([5, 37, 250]), rng.randrange(4)
    pm.refresh_curvature_comb(fit, scale, density, q)
    want = comb(curves, scale, density, q)
    ok = all(same(pm._comb_items[k].getOriginalDataset(), want[k], rtol=1e-7, atol=1e-10) for k in want)
    results.append((f"19.14.3 comb case {case} (S={S}, q={q}, density={density})", ok))
    height = rng.choice([0.05, 0.2, 0.001]); pm.deviation_height = height
    pm.refresh_deviation(fit)
    dw = deviation(np.asarray(pts, float), curves, height, tol)
    ok = all(same(pm._deviation_items[k].getOriginalDataset(), dw[k], rtol=1e-7, atol=1e-9) for k in dw if k != "band")
    ok = ok and same(pm._tolerance_item.getOriginalDataset(), dw["band"], rtol=1e-7, atol=1e-9)
    results.append((f"19.14.5 deviation case {case} (S={S}, tolerance={tol})", ok))

w.deleteLater(); app.processEvents()
for name, ok in results:
    print(("ok  " if ok else "FAIL"), name)
print(f"\n{sum(ok for _, ok in results)} of {len(results)} plot geometry checks agree with spec.md")
sys.exit(0 if all(ok for _, ok in results) else 1)
