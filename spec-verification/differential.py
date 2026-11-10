"""Differential test: clean-room (from spec.md) vs the source, on random inputs."""
import math, os, random, sys, tempfile, traceback
ROOT = r"C:\Users\MSFra\Desktop\spline_fitter"
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "packages", "bspline-varpro", "src"), os.path.dirname(__file__)]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import importlib, importlib.util
# Implementation under test: cleanroom.py by default, or a module named by IMPL (a path)
if os.environ.get('IMPL'):
    _spec = importlib.util.spec_from_file_location('impl', os.environ['IMPL']); cr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cr)
else:
    import cleanroom as cr
import importlib.util

rng = random.Random(20261106); nrng = np.random.default_rng(20261106)
stats = {}


def compare(name, ref_fn, cr_fn, eq=lambda a, b: a == b):
    s = stats.setdefault(name, [0, 0, None])
    s[0] += 1
    try:
        a = ("ok", ref_fn())
    except Exception as e:
        a = ("err", type(e).__name__)
    try:
        b = ("ok", cr_fn())
    except Exception as e:
        b = ("err", type(e).__name__)
    same = a[0] == b[0] and (a[0] == "err" and a[1] == b[1] or a[0] == "ok" and eq(a[1], b[1]))
    if not same:
        s[1] += 1
        if s[2] is None:
            s[2] = (a, b)


def arr_eq(rtol=0, atol=0):
    def f(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    return f


# ---------- parser ----------
from spline_fitter import io as sio
from bspline_varpro.airfoil import NACA4


def random_airfoil_text():
    code = rng.choice(["0012", "2412", "4415", "6409", "0006"])
    n = rng.randint(4, 40)
    P = NACA4(code, rng.random() < 0.7).points(n)
    if rng.random() < 0.3:
        P = P * rng.choice([1, 100, 37.5]) + [rng.uniform(-3, 3), rng.uniform(-1, 1)]
    dec = rng.randint(0, 8)
    sci = rng.random() < 0.15; fort = sci and rng.random() < 0.5
    comma = rng.random() < 0.1; semi = rng.random() < 0.1; glued = rng.random() < 0.1; paren = rng.random() < 0.05
    def fmt(v):
        s = f"{v:.{dec}e}" if sci else f"{v:.{dec}f}"
        if fort:
            s = s.replace("e", rng.choice("dD"))
        if comma:
            s = s.replace(".", ",")
        return s
    lines = []
    led = rng.random() < 0.2
    if led:
        le = int(np.argmin(P[:, 0])); up, lo = P[:le + 1][::-1], P[le:]
        if rng.random() < 0.5:
            lines.append(f"{len(up)} {len(lo)}")
        else:
            lines.append(f"{len(up)+rng.randint(-5,5)}. {len(lo)}.")
        rows = list(up) + list(lo)
    else:
        rows = list(P)
    body = []
    for x, y in rows:
        a, b = fmt(x), fmt(y)
        if glued and not comma and b.startswith("-"):
            sep = ""
        elif semi:
            sep = ";"
        elif comma:
            sep = " "
        else:
            sep = rng.choice([" ", "  ", "\t", ", "])
        if comma and sep == ", ":
            sep = " "
        ln = f"{a}{sep}{b}"
        if paren:
            ln = f"({a}, {b})"
        if rng.random() < 0.05:
            ln += " # c"
        body.append(ln)
    name = rng.choice(["", "NACA x", "#hash", "\\#esc", "0012 64", "12 9", "name # c", "  sp  "])
    out = ([name] if name else []) + lines + body
    if rng.random() < 0.1:
        out.insert(rng.randint(0, len(out)), "# comment")
    if rng.random() < 0.05:
        out.insert(rng.randint(1, len(out)), "garbage line")
    if rng.random() < 0.05 and len(body) > 6:
        out.append("999 999")
    t = "\n".join(out) + rng.choice(["\n", "", "\r\n"])
    if rng.random() < 0.05:
        t = "\ufeff" + t
    return t


def parse_eq(a, b):
    if a[0] != b[0] or a[1] != b[1] or a[3] != b[3]:
        return False
    if a[2] is None or b[2] is None:
        return a[2] is b[2]
    return len(a[2]) == len(b[2]) and all(x == y for p, q in zip(a[2], b[2]) for x, y in zip(p, q))


for _ in range(3000):
    t = random_airfoil_text()
    compare("3 parse_airfoil_text", lambda: sio.parse_airfoil_text(t, "f"), lambda: cr.parse_airfoil_text(t, "f"), parse_eq)
for s in ["1", "1.5", "0.00", "1e5", "1.2E-03", "1d2", "-0.0", "12.5e+1", "3.", ".5"]:
    compare("3.6 get_precision", lambda: sio.get_precision(s), lambda: cr.get_precision(s))

# ---------- geometry ----------
from spline_fitter import geometry as geo
for _ in range(1500):
    n = rng.randint(1, 30)
    P = [(round(rng.uniform(-1, 1), 3), round(rng.uniform(-1, 1), 3)) for _ in range(n)]
    if rng.random() < 0.5:
        P = [tuple(p) for p in NACA4("2412").points(rng.randint(2, 15))]
    S = rng.randint(0, 8)
    compare("4.2 split_into_sections", lambda: tuple(map(list, geo.split_into_sections(P, S))),
            lambda: (lambda s: (tuple(tuple(x) for x in s[0]), tuple(s[1])))(cr.split(P, S)),
            lambda a, b: tuple(map(tuple, a[0])) == b[0] and tuple(a[1]) == b[1])
    compare("4.1 leading edge", lambda: geo.find_leading_edge_index(P), lambda: cr.le_index(P))
    compare("2.6 arc length params", lambda: geo.compute_arc_length_parameters(P), lambda: cr.arc_params(P))

# ---------- knots, basis, models ----------
from bspline_varpro import basis
from spline_fitter.optimization.models import minimum_control_points
for _ in range(400):
    p, n = rng.randint(0, 7), rng.randint(0, 14)
    compare("2.1 open_uniform_knots", lambda: basis.open_uniform_knots(p, n), lambda: cr.open_uniform_knots(p, n), arr_eq())
    if 1 <= p < n:
        inner = np.sort(nrng.uniform(0.01, 0.99, rng.randint(0, 6)))
        t = cr.clamped_knots(p, inner)
        compare("2.3 bending energy", lambda: basis.bending_energy_matrix(t, p), lambda: cr.bending_energy(t, p), arr_eq(1e-12, 1e-12))
        for r in range(0, min(p, 3) + 1):
            if len(t) - p - 1 > r:
                from bspline_varpro.constraints import end_derivative_coefficients
                compare("7.1 end coefficients", lambda: np.concatenate(end_derivative_coefficients(t, p, r)),
                        lambda: np.concatenate(cr.end_coefs(t, p, r)), arr_eq(1e-12, 1e-12))
    orders = [rng.randint(0, 3) for _ in range(rng.randint(0, 5))]
    compare("5.2 minimum control points", lambda: minimum_control_points(orders, p), lambda: cr.min_ctrl(orders, p))
for _ in range(300):
    P = nrng.normal(size=(rng.randint(0, 12), 2))
    if rng.random() < 0.2 and len(P):
        P[:] = P[0]
    compare("2.6 centripetal", lambda: basis.centripetal_parameters(P), lambda: cr.centripetal(P), arr_eq(1e-15, 0))

# ---------- reparam, alpha ----------
from bspline_varpro import reparam, constraints as bc
for _ in range(300):
    x = nrng.normal(scale=rng.choice([1, 30, 800]), size=20)
    compare("8.2 softplus", lambda: reparam.softplus(x), lambda: cr.softplus(x), arr_eq())
    compare("8.2 inverse_softplus", lambda: reparam.inverse_softplus(np.abs(x)), lambda: cr.inv_softplus(np.abs(x)), arr_eq())
    u = np.sort(nrng.uniform(0, 3, 15)); u[0], u[-1] = 0, 3
    compare("8.2 t_to_eta", lambda: reparam.t_to_eta(u), lambda: cr.t_to_eta(u), arr_eq())
    compare("8.2 eta_to_t", lambda: reparam.eta_to_t(x, 3), lambda: cr.eta_to_t(x, 3), arr_eq())
    ratio, a = math.exp(rng.uniform(-4, 4)), rng.uniform(-10, 10)
    compare("7.3 alpha", lambda: bc._alpha(a, ratio), lambda: cr.alpha(a, ratio), lambda p, q: np.allclose(p, q, rtol=1e-14))
    al = math.exp(rng.uniform(-6, 6))
    compare("7.3 alpha inverse", lambda: bc._alpha_parameter(al, ratio), lambda: cr.alpha_param(al, ratio), lambda p, q: math.isclose(p, q, rel_tol=1e-13, abs_tol=1e-13))

# ---------- constraints and residual ----------
from bspline_varpro.problem import VarProProblem


def random_problem():
    S = rng.randint(1, 4)
    pts = [tuple(p) for p in NACA4(rng.choice(["0012", "2412", "4415"])).points(rng.randint(15, 30))]
    secs = [np.array(s, float) for s in cr.split(pts, S)[0]]
    p = rng.randint(2, 5)
    juncs = [(rng.randint(0, min(3, p)), rng.random() < 0.6, rng.random() < 0.7) for _ in range(S - 1)]
    e = [0] + [j[0] for j in juncs] + [0]
    knots = []
    for s in range(S):
        need = max(p + 1, e[s] + e[s + 1] + 2)
        ni = need + rng.randint(0, 4) - p - 1
        knots.append(cr.clamped_knots(p, np.sort(nrng.uniform(0.05, 0.95, ni))) if rng.random() < 0.5
                     else cr.open_uniform_knots(p, need + rng.randint(0, 4)))
    return secs, knots, p, juncs


for _ in range(250):
    secs, knots, p, juncs = random_problem()
    specs = [bc.Junction(o, g, f) for o, g, f in juncs]
    chain = bc.ChainConstraints(secs, knots, p, specs)
    theta = nrng.normal(size=chain.num_params)
    compare("7.4 constraint rows", lambda: np.hstack([*chain.build(theta)[:2]]) if chain.build(theta)[0].size else np.zeros(0),
            lambda: (lambda C, d, _: np.hstack([C, d]) if C.size else np.zeros(0))(*cr.build(secs, knots, p, juncs, theta)),
            arr_eq(1e-11, 1e-11))
    sm = rng.choice([0.0, 1e-3, 1e-1]); reg = rng.choice([1e-12, 1e-6])
    prob = VarProProblem(secs, knots, p, continuity=specs, smoothing=sm, regularization=reg)
    u0 = prob.constraints and None
    from bspline_varpro.fitting import initial_chain_parameters
    u = initial_chain_parameters(None, secs)
    u = np.sort(np.clip(u + nrng.normal(scale=0.02, size=len(u)), 0, len(secs))); u[0], u[-1] = 0, len(secs)
    compare("8.4 VarPro residual", lambda: prob.residual(u, theta),
            lambda: cr.varpro_residual(secs, knots, p, juncs, sm, reg, u, theta), arr_eq(1e-6, 1e-9))

# ---------- LM ----------
from bspline_varpro.fitting import levenberg_marquardt
for _ in range(200):
    k = rng.randint(2, 6); xs = np.linspace(0, 3, 25); truth = nrng.normal(size=k)
    noise = nrng.normal(scale=rng.choice([0, 1e-3, 0.3]), size=25)
    def model(c):
        return sum(c[i] * np.exp(-(i + 1) * c[(i + 1) % k] * xs * 0.3) for i in range(k))
    y = model(truth) + noise
    res = lambda c: model(c) - y
    def jac(c, h=1e-7):
        r0 = res(c); return np.column_stack([(res(c + h * np.eye(k)[j]) - r0) / h for j in range(k)])
    x0 = truth + nrng.normal(scale=0.5, size=k); mx = rng.choice([None, 5, 30])
    def ref():
        s = levenberg_marquardt(res, jac, x0, max_nfev=mx); return (s.x, s.nfev, s.status, s.message)
    def cln():
        x, c, nf, st, m = cr.levenberg_marquardt(res, jac, x0, max_nfev=mx); return (x, nf, st, m)
    compare("8.5 Levenberg-Marquardt", ref, cln, lambda a, b: np.allclose(a[0], b[0], rtol=1e-9, atol=1e-12) and a[1:] == b[1:])

# ---------- knot helpers, PAV ----------
from bspline_varpro import tolerance as tl, joint
for _ in range(500):
    p = rng.randint(1, 5); t = cr.clamped_knots(p, np.round(np.sort(nrng.uniform(0, 1, rng.randint(0, 6))), 2).clip(0.01, 0.99))
    u = np.sort(nrng.uniform(0, 1, rng.randint(1, 30))); dev = nrng.exponential(size=len(u)); tol = rng.uniform(0, 2)
    ms = rng.choice([1e-4, 0.05])
    compare("9.3 new_knots", lambda: tl._new_knots(t, u, dev, tol, ms), lambda: cr.new_knots(t, u, dev, tol, ms))
    compare("9.3 removal_order", lambda: tl._removal_order(t, p, u, dev, tol + 0.1), lambda: cr.removal_order(t, p, u, dev, tol + 0.1))
    v = nrng.normal(size=rng.randint(1, 20))
    compare("9.7 PAV", lambda: joint._nondecreasing(v), lambda: cr.pav(v), arr_eq(1e-14, 1e-14))

# ---------- default tolerance, NACA ----------
from bspline_varpro import airfoil as ba
for _ in range(150):
    P = NACA4(rng.choice(["0012", "2412", "4415"])).points(rng.randint(3, 60))
    if rng.random() < 0.5:
        P = P + nrng.normal(scale=rng.choice([1e-6, 1e-4]), size=P.shape)
    if rng.random() < 0.6:
        P = np.round(P, rng.randint(3, 8))
    dec = rng.choice([None, 4, 6])
    compare("12.6 default tolerance", lambda: ba.default_tolerance(P, decimals=dec), lambda: cr.default_tolerance(P, dec),
            lambda a, b: math.isclose(a, b, rel_tol=1e-9))
    compare("12.6 decimal_places", lambda: ba.decimal_places(P), lambda: cr.decimal_places(P))
for code in ["0012", "2412", "4415", "6409", "9999", "0000", "1234"]:
    for closed in (True, False):
        n = rng.randint(2, 100)
        compare("13.1 NACA4 points", lambda: NACA4(code, closed).points(n), lambda: cr.naca(code, n, closed), arr_eq(1e-15, 1e-15))

# ---------- writers ----------
from bspline_varpro import export as ex
d = tempfile.mkdtemp()
for i in range(200):
    p = rng.randint(1, 5); S = rng.randint(1, 4)
    secs = [[(nrng.normal(), nrng.normal()) for _ in range(rng.randint(p + 1, p + 6))] for _ in range(S)]
    for s in range(1, S):
        secs[s][0] = secs[s - 1][-1]
    knots = None if rng.random() < 0.3 else [cr.clamped_knots(p, np.sort(nrng.uniform(0.01, 0.99, len(c) - p - 1))).tolist() for c in secs]
    name = rng.choice([None, "", "NACA", "#x", "\\#y", " sp "]); prec = rng.choice([None, 0, 5, 12, 20, -3])
    cont = rng.choice([None, [], ["G2"] * (S - 1)])
    path = os.path.join(d, f"s{i}.spline.dat")
    def ref():
        sio.export_spline_data(path, p, secs, name=name, precision=prec, section_knots=knots, continuity=cont)
        return open(path, encoding="utf-8").read()
    compare("14.2 spline file text", ref, lambda: cr.spline_text(p, secs, name, prec, knots, cont))
for _ in range(2000):
    v = float(nrng.choice([nrng.normal() * 10 ** rng.randint(-30, 30), rng.randint(-5, 5), 0.0, -0.0, 1e16, 1e-5]))
    compare("14.4 STEP real", lambda: ex._step_real(v), lambda: cr.step_real(v))
for s in ["abc", "it's", "a\\b", "é", "名前", "\U0001F600", "", "x'y\\z"]:
    compare("14.4 STEP string", lambda: ex._step_string(s), lambda: cr.step_string(s))

# ---------- naming, formatting ----------
from spline_fitter.gui import batch
from spline_fitter.optimization.models import SplineConfig, OptimizationConfig, ContinuityMode, OptimizerType
from spline_fitter import formatting as fm


def cr_output_name(stem, c, o, auto):
    parts = [stem, f"deg{c.degree}"]
    if (o.tolerance is None or not o.grow_control_points) and c.num_control_points != 25:
        parts.append(f"n{c.num_control_points}")
    if c.num_sections != 2:
        parts.append(f"s{c.num_sections}")
    sp = c.junction_specs()
    if sp and sp != ["G3"] * len(sp):
        parts.append("-".join(sp) if len(set(sp)) > 1 else sp[0])
    if o.optimizer != OptimizerType.GN_VARPRO:
        parts.append("lsq" if o.optimizer == OptimizerType.DIRECT_LSQ else o.optimizer.value)
        if o.parameterization != "centripetal":
            parts.append(o.parameterization.replace("_", "-"))
    if o.tolerance is not None:
        parts.append("tol-auto" if auto else f"tol{cr.num(o.tolerance)}")
    if o.knot_placement != "uniform":
        parts.append(f"knots-{o.knot_placement}")
    if o.grow_control_points:
        parts.append("grow")
    if o.smoothing_weight != 1e-3:
        parts.append(f"smooth{cr.num(o.smoothing_weight)}")
    if o.smoothing_factor is not None:
        parts.append(f"sfactor{cr.num(o.smoothing_factor)}")
    if o.max_points_per_section is not None:
        parts.append(f"sub{o.max_points_per_section}")
    if o.optimize_le_position:
        parts.append("free-le")
    if o.early_stop_patience != 50:
        parts.append(f"patience{o.early_stop_patience}")
    if o.relative_tolerance != 1e-6:
        parts.append(f"rtol{cr.num(o.relative_tolerance)}")
    return ".".join(parts) + ".spline.dat"


for _ in range(3000):
    S = rng.randint(1, 4); deg = rng.randint(3, 7)
    jc = tuple(rng.choice(["G3", "G2", "C1", "C0"]) for _ in range(S - 1)) if rng.random() < 0.3 else ()
    c = SplineConfig(degree=deg, num_control_points=rng.choice([25, 30, 12]), num_sections=S,
                     continuity=rng.choice(list(ContinuityMode)), geometric_continuity=rng.random() < 0.5, junction_continuity=jc)
    o = OptimizationConfig(optimizer=rng.choice(list(OptimizerType)), parameterization=rng.choice(["centripetal", "arc_length"]),
                           tolerance=rng.choice([None, 1e-4, 2.5e-5, 0.02]), knot_placement=rng.choice(["uniform", "free"]),
                           grow_control_points=rng.random() < 0.5, smoothing_weight=rng.choice([1e-3, 0.0, 0.05, 3.3e-7]),
                           smoothing_factor=rng.choice([None, 0.5]), max_points_per_section=rng.choice([None, 150]),
                           optimize_le_position=rng.random() < 0.3, early_stop_patience=rng.choice([50, 10]),
                           relative_tolerance=rng.choice([1e-6, 1e-8]))
    auto = rng.random() < 0.5
    compare("16 batch output_name", lambda: batch.output_name("x", c, o, auto), lambda: getattr(cr, "output_name", cr_output_name)("x", c, o, auto))
for _ in range(3000):
    v = rng.choice([nrng.normal() * 10 ** rng.randint(-8, 8), 0.0, float("inf"), float("nan"), 1e-3, 1e4, "abc"])
    compare("18 format_error", lambda: fm.format_error(v), lambda: cr.format_error(v))
    sec = abs(float(nrng.normal() * 10 ** rng.randint(-6, 5)))
    compare("18 format_duration", lambda: fm.format_duration(sec), lambda: cr.format_duration(sec))
    txt = "".join(rng.choice("ab <>:\"/\\|?*.\t\x01 é") for _ in range(rng.randint(0, 8)))
    compare("18 file_stem", lambda: fm.file_stem(txt), lambda: cr.file_stem(txt))
    compare("18 sanitize_filename", lambda: fm.sanitize_filename(txt), lambda: cr.sanitize(txt))
    vv = rng.choice([1e-4, 2.5e-5, 0.0123, 0.5, 3.33333e-7, -1e-3, 12345.0, 0.0])
    compare("16 num", lambda: batch._num(vv), lambda: cr.num(vv))

# ---------- GUI pure logic ----------
from spline_fitter.gui import explorer as exp
for _ in range(3000):
    words = [rng.choice(["varpro", ">1e-5", "<= 12", "=5", "!=3", "=Free", "!=uniform", ">", "=", "<x", "1,5", ">=2,5", "G3", "  "])
             for _ in range(rng.randint(0, 4))]
    text = " ".join(words)
    compare("19.17.2 parse_filter", lambda: exp.parse_filter(text), lambda: cr.parse_filter(text))
from PyQt6.QtWidgets import QApplication, QDoubleSpinBox
app = QApplication.instance() or QApplication([])
from spline_fitter.gui.widgets import ScientificSpinBox, SweepableDoubleSpinBox
from spline_fitter.gui.theme import grey
for _ in range(1000):
    sb = ScientificSpinBox(); mn = rng.choice([0.0, 1e-9, 1e-6]); sb.setRange(mn, math.inf); sb.setSingleStep(rng.choice([1e-6, 1e-3]))
    v = rng.choice([0.0, 1e-5, 2e-5, 5e-3, 1.0, 3.7e-4, 1e-9, 0.1]); up = rng.random() < 0.5
    sb.setValue(v); v = sb.value()   # the box holds its value clamped to the range
    compare("19.21 ScientificSpinBox step", lambda: (sb.setValue(v), sb.stepBy(1 if up else -1), sb.value())[2], lambda: cr.ladder_next(v, up, sb.singleStep(), sb.minimum()),
            lambda a, b: math.isclose(a, b, rel_tol=1e-12))
    compare("19.21 ScientificSpinBox text", lambda: sb.textFromValue(v * rng.choice([1, 1.234567891])) if False else sb.textFromValue(v),
            lambda: f"{v:.6g}".replace(".", sb.locale().decimalPoint()))
for _ in range(500):
    spin = ScientificSpinBox(); spin.setRange(0.0, math.inf); spin.setSingleStep(0.001)
    w = SweepableDoubleSpinBox(spin); lo = rng.choice([0.0, 0.001, 0.02]); spin.setValue(lo)
    w._btn_sweep.click(); w._spin_max.setValue(rng.choice([0.01, 0.1, 1.0, 0.0])); w._spin_step.setValue(rng.choice([0.001, 0.003, 0.05]))
    compare("19.21 sweep_values", lambda: w.sweep_values(), lambda: cr.sweep_values(spin.value(), w._spin_max.value(), w._spin_step.value()))
for _ in range(2000):
    L, T = rng.random(), rng.uniform(-4, 4)
    compare("19.19.1 grey", lambda: grey(L, T), lambda: cr.grey(L, T))

total = sum(s[0] for s in stats.values()); bad = sum(s[1] for s in stats.values())
for k, (n, f, ex1) in stats.items():
    print(f"{'FAIL' if f else 'ok  '} {k}: {n - f}/{n}" + (f"   first mismatch: {str(ex1)[:400]}" if f else ""))
print(f"\n{total - bad} of {total} comparisons agree across {len(stats)} components")
sys.exit(1 if bad else 0)
