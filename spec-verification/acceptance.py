"""Check every item of spec.md section 21 against the implementation.

Usage: python acceptance.py [--fits-only]
--fits-only prints a SHA-256 of the item-11 fit outputs (for bit-for-bit runs).
"""
import hashlib, json, os, re, sys, tempfile, time

ROOT = r"C:\Users\MSFra\Desktop\spline_fitter"
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "packages", "bspline-varpro", "src")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["SPLINE_FITTER_LOG_DIR"] = tempfile.mkdtemp()
import numpy as np

results = []


def check(item, ok, detail=""):
    results.append((item, bool(ok), detail))
    print(("PASS" if ok else "FAIL"), item, detail if not ok else "")


def close(a, b, rel=1e-3):
    return abs(a - b) <= rel * abs(b)


from bspline_varpro.airfoil import NACA4
from spline_fitter.geometry import create_airfoil_data
from spline_fitter.optimization.models import *
from spline_fitter.optimization import run_optimization
from spline_fitter.analysis import fit_report, default_tolerance, scipy_curves

pts = [(float(x), float(y)) for x, y in np.round(NACA4("2412", closed_te=True).points(81), 6)]
af = create_airfoil_data(pts, 2, "naca2412", "NACA 2412", 6)
REF_SPLINE = SplineConfig(continuity=ContinuityMode.G3, geometric_continuity=True)
REF_OPT = OptimizationConfig(knot_placement="uniform", grow_control_points=False)


def fit(cfg, opt):
    r = run_optimization(cfg, opt, af)
    fr = FitResult(r.section_control_points, cfg.degree, r.fun, r.nit, 0, 0.0,
                   section_knots=r.section_knots, rms_error=r.rms_error)
    return r, fr


if "--fits-only" in sys.argv:
    h = hashlib.sha256()
    for cfg, opt in ((REF_SPLINE, REF_OPT),
                     (SplineConfig(continuity=ContinuityMode.G2), OptimizationConfig(optimizer=OptimizerType.DIRECT_LSQ)),
                     (SplineConfig(continuity=ContinuityMode.G2, geometric_continuity=True),
                      OptimizationConfig(tolerance=1e-6))):
        r, _ = fit(cfg, opt)
        h.update(np.asarray(r.x, dtype=np.float64).tobytes())
        h.update(repr((r.fun, r.nit, r.section_knots, r.message)).encode())
    print("FITS", h.hexdigest())
    sys.exit(0)

# 1-4
from bspline_varpro.basis import open_uniform_knots
from spline_fitter.formatting import format_error, format_duration, sanitize_filename, file_stem, set_decimal_point
set_decimal_point(".")
check(1, list(open_uniform_knots(3, 5)) == [0, 0, 0, 0, 0.5, 1, 1, 1, 1])
check(2, [format_error(v) for v in (0.0000123, 0.00456, 12345.6)] == ["1.230e-05", "0.004560", "1.235e+04"])
check(3, [format_duration(v) for v in (0.000123, 0.456, 12.34, 123.4, 3723.0)]
      == ["123 µs", "456 ms", "12.34 s", "2m 3s", "1h 2m 3s"])
from spline_fitter.io import get_precision, _number_fields, parse_airfoil_text, _scaled_precision
check(4, [get_precision(s) for s in ("0.0123", "1.2346e-03", "1.23e+01")] == [4, 7, 1])
# 5
check(5, [_number_fields(s) for s in ("0.95000-0.01200", "0,5 0,01", "1.0, 0.5", "1.0D-03 2.5d+00", '(1.0; "2.0")')]
      == [["0.95000", "-0.01200"], ["0.5", "0.01"], ["1.0", "0.5"], ["1.0e-03", "2.5e+00"], ["1.0", "2.0"]])
# 6
P5 = [(1, 0), (0.5, 0.05), (0, 0), (0.5, -0.05), (1, 0)]
r = parse_airfoil_text("NACA test\n1.0000 0.0000\n0.5000 0.0500\n0.0000 0.0000\n0.5000 -0.0500\n1.0000 0.0000\n", "f")
check("6a", r == ("f", "NACA test", P5, 4), r)
r = parse_airfoil_text("NACA test\n100 0\n50 5\n0 0\n50 -5\n100 0\n", "f")
check("6b", r == ("f", "NACA test", P5, 2), r)
r = parse_airfoil_text("T\n3 3\n0 0\n0.5 0.06\n1 0\n0 0\n0.5 -0.05\n1 0\n", "f")
check("6c", r == ("f", "T", [(1, 0), (0.5, 0.06), (0, 0), (0.5, -0.05), (1, 0)], 2), r)
r = parse_airfoil_text("0012 64\n1 0\n0.5 0.06\n0 0\n0.5 -0.06\n1 0\n", "f")
check("6d", r[1] == "0012 64" and len(r[2]) == 5 and r[3] == 2, r)
r = parse_airfoil_text("#c\n\\#name\n1 0\n0.5 0.06\n0 0\n0.5 -0.06\n1 0\n", "f")
check("6e", r[1] == "#name", r)
r = parse_airfoil_text("x\n2 0.2\n1 0.1\n0 0\n1 -0.1\n2 -0.2\n", "f")
check("6f", r == ("f", "x", [(1, 0.1), (0.5, 0.05), (0, 0), (0.5, -0.05), (1, -0.1)], 1), r)
# 7
check(7, [_scaled_precision(4, 0.01), _scaled_precision(5, 0.5), _scaled_precision(6, 1 / 0.98)] == [6, 5, 6])
# 8, 9
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from spline_fitter.gui.batch import output_name, _num
names = [
    output_name("naca0012", SplineConfig(degree=5, num_control_points=30, num_sections=3, junction_continuity=("G2", "C1")),
                OptimizationConfig(tolerance=1e-4, knot_placement="uniform", grow_control_points=False)),
    output_name("naca0012", REF_SPLINE, REF_OPT),
    output_name("naca0012", REF_SPLINE, OptimizationConfig(tolerance=1e-5), True),
    output_name("x", SplineConfig(degree=3, continuity=ContinuityMode.G2),
                OptimizationConfig(optimizer=OptimizerType.DIRECT_LSQ, parameterization="arc_length", knot_placement="uniform",
                                   grow_control_points=False, smoothing_weight=0.01, optimize_le_position=True)),
    output_name("x", SplineConfig(num_sections=1, continuity=ContinuityMode.G3, geometric_continuity=True), REF_OPT),
]
check(8, names == ["naca0012.deg5.n30.s3.G2-C1.tol1e-4.spline.dat", "naca0012.deg5.spline.dat",
                   "naca0012.deg5.tol-auto.knots-free.grow.spline.dat",
                   "x.deg3.C2.lsq.arc-length.smooth0.01.free-le.spline.dat", "x.deg5.s1.spline.dat"], names)
check(9, [_num(v) for v in (1e-4, 2.5e-5, 0.001, 0.0123, 0.5, 1e-6, 3.33333e-7)]
      == ["1e-4", "2.5e-5", "1e-3", "0.0123", "0.5", "1e-6", "3.33e-7"])
# 10
check(10, sanitize_filename("a b/c") == "a_b_c" and file_stem("NACA 2412 t/c 12%") == "NACA 2412 t_c 12%"
      and file_stem("") == "airfoil")
# 11
check("11 deftol", default_tolerance(pts, 6) == 1e-6)
r, fr = fit(REF_SPLINE, REF_OPT)
rep = fit_report(af.sections, fr, 6)
check("11 varpro", close(r.fun, 8.978e-7) and r.fun == r.rms_error and r.nit == 33 and r.success
      and r.message == "`ftol` termination condition is satisfied." and [len(s) for s in r.section_control_points] == [25, 25]
      and close(rep["max_deviation"], 4.587e-6) and close(rep["rms_deviation"], 8.978e-7)
      and close(rep["max_thickness"], 0.12004) and close(rep["max_thickness_x"], 0.2988)
      and close(rep["max_camber"], 0.020000) and close(rep["max_camber_x"], 0.4065)
      and close(rep["leading_edge_radius"], 0.015698) and rep["curvature_extrema"] == 5, (r.fun, r.nit, rep))
varpro_fit = fr
r2, fr2 = fit(SplineConfig(continuity=ContinuityMode.G2), OptimizationConfig(optimizer=OptimizerType.DIRECT_LSQ))
rep2 = fit_report(af.sections, fr2, 6)
check("11 lsq", close(r2.fun, 2.942e-7) and r2.nit == 1 and [len(s) for s in r2.section_control_points] == [25, 25]
      and close(rep2["max_deviation"], 1.270e-6) and close(rep2["rms_deviation"], 2.933e-7)
      and rep2["curvature_extrema"] == 12, (r2.fun, rep2))
t0 = time.time()
r3, _ = fit(SplineConfig(continuity=ContinuityMode.G2, geometric_continuity=True), OptimizationConfig(tolerance=1e-6))
check("11 tolerance", r3.success and r3.fun <= 1e-6 and r3.nit == 9 and close(r3.fun, 7.9338e-7)
      and [len(s) for s in r3.section_control_points] == [11, 11], (r3.fun, r3.nit, [len(s) for s in r3.section_control_points],
                                                     round(time.time() - t0)))
print("   tolerance search:", r3.nit, "refits,", [len(s) for s in r3.section_control_points], "control points,",
      f"max deviation {r3.fun:.4e},", round(time.time() - t0), "s")
# 12, 13
from bspline_varpro.export import to_step, to_iges, to_dxf, merge_curves, _step_real
c = scipy_curves(varpro_fit)
ig = to_iges(c, "NACA 2412").splitlines()
m = merge_curves(c)
check(12, to_step(c).count("B_SPLINE_CURVE_WITH_KNOTS") == 2 and {len(x) for x in ig} == {80}
      and ig[-1] == "S      1G      4D      4P    112" + " " * 40 + "T      1"
      and to_dxf(c).count("SPLINE\n") == 2 and m.k == 5 and len(m.t) == 54, ig[-1])
check(13, [_step_real(v) for v in (1e-7, 3.0, 0.5, 2.5e16, -1e16)] == ["1.E-7", "3.0", "0.5", "2.5E16", "-1.E16"])
# 14
from spline_fitter.io import export_spline_data, parse_spline_data
from bspline_varpro.basis import clamped_knots
d = tempfile.mkdtemp()
k = [clamped_knots(3, [0.3, 0.3, 0.7]).tolist(), clamped_knots(3, [0.5]).tolist()]
cp = [[(0.1 * i, 0.2 * i) for i in range(7)], [(0.6, 1.2)] + [(1 + i / 3, -i / 7) for i in range(4)]]
p = os.path.join(d, "t.deg3.spline.dat")
export_spline_data(p, 3, cp, name="#n", precision=12, section_knots=k, continuity=["G2"])
s = parse_spline_data(p)
check(14, s["name"] == "#n" and s["degree"] == 3 and s["section_knots"] == k and s["continuity"] == ["G2"]
      and all(np.allclose(a, b, atol=5e-13, rtol=0) for a, b in zip(s["section_control_points"], cp))
      and open(p, encoding="utf-8").read().splitlines()[0] == "\\#n")
# 15
from spline_fitter.project_io import create_project, create_optimization_run, number_runs, save_project, load_project
fr15 = FitResult(cp, 3, 1e-3, 5, 7, 0.1, section_knots=k, rms_error=2e-3)
run = create_optimization_run(SplineConfig(3, 7, 2), OptimizationConfig(optimizer=OptimizerType.DIRECT_LSQ), fr15,
                              [IterationRecord(1, 1e-3, 0.1)])
pr = create_project(af); pr.optimization_runs.append(run); number_runs(pr.optimization_runs)
q = os.path.join(d, "t.asfproj"); save_project(pr, q)
l1 = load_project(q).optimization_runs[0]
ok1 = (l1.number == 1 and l1.spline_config == run.spline_config and l1.optimization_config.optimizer == OptimizerType.DIRECT_LSQ
       and l1.section_control_points == [list(map(tuple, s_)) for s_ in cp] and l1.result.section_knots == k
       and l1.result.final_error == 1e-3 and l1.result.rms_error == 2e-3 and l1.loss_history[0].error_value == 1e-3)
data = json.load(open(q, encoding="utf-8")); data["optimization_runs"][0]["optimization_config"]["optimizer"] = "bfgs"
json.dump(data, open(q, "w", encoding="utf-8"))
check(15, ok1 and load_project(q).optimization_runs[0].optimization_config.optimizer == OptimizerType.GN_VARPRO)
# 16: the grey table as written in spec.md
from spline_fitter.gui.theme import grey
spec = open(os.path.join(ROOT, "spec.md"), encoding="utf-8").read()
rows = re.findall(r"^\| `(\w+)`(?: \(plot\))? \| ([\d.]+) \| ([\d.]+) \| `(#[0-9A-F]{6})` \| `(#[0-9A-F]{6})` \|$", spec, re.M)
check(16, len(rows) == 13 and all(grey(float(dl), 0) == dh and grey(float(ll), 0) == lh for _, dl, ll, dh, lh in rows)
      and grey(0.213, 1) == "#1A1915" and grey(0.970, 1) == "#F9F5E6" and grey(0.213, -1) == "#17191C"
      and grey(0.950, -1) == "#E0F0FF", len(rows))

# 17, 18: the window
from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from spline_fitter.gui.ui import SplineFitterWindow
w = SplineFitterWindow(); set_decimal_point(".")
w.show(); app.processEvents()
msg = lambda: w.statusBar().currentMessage()
T = lambda s: s.itemText(s.currentIndex())
ok = msg() == "Ready" and w.windowTitle() == "Airfoil Spline Fitter"
ok &= (T(w._combo_opt), T(w._seg_goal), w._spin_degree.value(), w._spin_ctrl.value(), w._spin_sections.value(),
       T(w._combo_cont), T(w._combo_cont_kind), T(w._combo_knot_placement), w._check_grow.isChecked(),
       w._check_auto_tolerance.isChecked(), w._combo_live.currentText()) == (
    "VarPro LM", "Fixed count", 5, 25, 2, "G3", "Geometric", "Uniform", False, True, "Moderate")
w.new_naca_airfoil("2412", 81, True)
ok &= msg() == "Created: NACA 2412 (161 points, 2 sections: [81, 81])" and w._spin_tolerance.value() == 1e-6
w._fitting_controller.on_fit()
s1 = msg()
ok &= bool(re.fullmatch(r"Completed \(VarPro LM\): 33 evaluations in .+ \| Error 8\.978e-07 \| max deviation 4\.59e-06, "
                        r"RMS 8\.98e-07, 49 control points", s1))
ok &= w._lbl_project.text() == "Unsaved project*\n1 optimization run" and w.windowTitle() == "Unsaved project[*] - Airfoil Spline Fitter"
logs = os.listdir(os.environ["SPLINE_FITTER_LOG_DIR"])
ok &= len(logs) == 1 and bool(re.fullmatch(r"\d{8}_\d{6}_NACA_2412_gn-varpro\.csv", logs[0]))
ok &= (w._lbl_result_caption_dev.text(), w._lbl_result_dev.text(), w._lbl_result_ctrl.text()) == ("Max deviation", "4.59e-06", "2 × 25")
w._combo_opt.setCurrentIndex(1); w._combo_cont.setCurrentIndex(2)
w._fitting_controller.on_fit(); s2 = msg()
ok &= bool(re.fullmatch(r"Completed \(Direct LSQ\): 2 solves in .+ \| Error 2\.942e-07 \| max deviation 1\.27e-06, "
                        r"RMS 2\.93e-07, 49 control points", s2)) and len(w._project_manager.project.optimization_runs) == 2
w._settings_history.commit(); w._settings_history.undo()
ok &= (T(w._combo_opt), T(w._combo_cont)) == ("VarPro LM", "G3")
w._sweep_ctrl._btn_sweep.click(); w._sweep_ctrl._spin_max.setValue(26)
w._fitting_controller.on_fit(); s3 = msg()
ok &= bool(re.fullmatch(r"Sweep complete: 2 fits in .+ \| Best: Control points 25, error 8\.978e-07", s3))
ok &= not w._sweep_ctrl.is_sweep_active and w._spin_ctrl.value() == 25
check(17, ok, (s1, s2, s3))
# 18: batch, reference settings
src, dst = tempfile.mkdtemp(), tempfile.mkdtemp()
for code in ("0012", "2412"):
    with open(os.path.join(src, f"naca{code}.dat"), "w") as f:
        f.write(f"NACA {code}\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in NACA4(code).points(41)) + "\n")
w._settings_history.commit()
b = w._batch_controller; b._batch_src_dir, b._batch_dst_dir = src, dst
b._on_batch_convert(); first = msg(); out = sorted(os.listdir(dst))
b._on_batch_convert(); second = msg()
check(18, out == ["naca0012.deg5.spline.dat", "naca2412.deg5.spline.dat"]
      and first.startswith("Batch complete: 2 files in ") and first.endswith("(0 failed)")
      and second.startswith("Batch complete: 0 files in ")
      and second.endswith("(0 failed) | Skipped 2 already in the target folder with these settings"), (out, first, second))
w._project_manager._project_modified = False
w.close()
w.deleteLater(); app.processEvents(); del w  # avoid the exit crash of section 22 item 14

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)} of {len(results)} checks passed")
sys.exit(1 if failed else 0)
