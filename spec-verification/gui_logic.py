"""Model tests of spec.md 19.8 (settings history), 19.17.1-19.17.3 (explorer columns,
row filter, sorting), 19.16.4 (run matching) and 19.13 (sweep comparison series).

Each rule is implemented from the spec text and compared with the source on random input.
"""
import os, random, sys, types
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "packages", "bspline-varpro", "src"), os.path.dirname(__file__)]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cleanroom as cr
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QCheckBox, QLineEdit, QSpinBox
app = QApplication.instance() or QApplication([])
from spline_fitter.formatting import set_decimal_point, format_duration
set_decimal_point(".")
from spline_fitter.gui.history import SettingsHistory
from spline_fitter.gui import explorer as ex
from spline_fitter.gui.project import ProjectManager
from spline_fitter.optimization.models import (ContinuityMode, FitResult, OptimizationConfig, OptimizationRun,
                                              OptimizerType, SplineConfig)

rng = random.Random(1106)
report = {}


def tally(name, ok, detail=None):
    r = report.setdefault(name, [0, 0, None]); r[0] += 1
    if not ok:
        r[1] += 1
        if r[2] is None:
            r[2] = detail


# ---------------- 19.8 settings history ----------------
class HistoryModel:
    def __init__(self, cap):
        self.cap = cap; self.states = [cap()]; self.idx = 0; self.timer = False; self.pq = False; self.latest = None
    def on_change(self, quiet):
        if self.timer and quiet != self.pq and self.latest is not None:
            self.settle(self.latest, self.pq)
        self.pq = quiet; self.latest = self.cap(); self.timer = True
    def settle(self, s, q):
        if q:
            self.states[self.idx] = s
        elif s != self.states[self.idx]:
            del self.states[self.idx + 1:]; self.states.append(s); del self.states[:-200]; self.idx = len(self.states) - 1
    def commit(self):
        self.timer = False; q, self.pq = self.pq, False; self.latest = None; self.settle(self.cap(), q)
    def can_undo(self):
        return self.idx > 0 or (self.timer and not self.pq)
    def can_redo(self):
        return self.idx < len(self.states) - 1 and not self.timer
    # undo/redo are split around the real call: commit (before the real widgets change),
    # then step and check the widgets the real history restored
    def step(self, delta, setter):
        if 0 <= self.idx + delta < len(self.states):
            self.idx += delta; setter(self.states[self.idx]); self.states[self.idx] = self.cap()


def history_test(rounds=6, steps=700):
    for _ in range(rounds):
        edit, check, spin = QLineEdit(), QCheckBox(), QSpinBox(); spin.setRange(0, 9)
        widgets = [edit, check, spin]
        fitting = [False]
        h = SettingsHistory(widgets, is_quiet=lambda: fitting[0])
        h._timer.setInterval(10 ** 8)   # time never passes: commits happen only when called
        cap = lambda: (edit.text(), check.isChecked(), spin.value())
        m = HistoryModel(cap)
        def setter(state):
            # The real history's undo/redo has already set the widgets: they must hold this state
            tally("19.8 restored widget values", cap() == state, (cap(), state))
        m_restoring = [False]
        quiet_ctx = [0]
        def changed_via(fn, quiet_flag):
            before = cap(); fn()
            if cap() != before:
                pass
        # Mirror every value change into the model through the same signals
        def hook(*_):
            if not m_restoring[0] and not h._restoring:
                m.on_change(bool(quiet_ctx[0] or fitting[0]))
        for sig in (edit.textChanged, check.toggled, spin.valueChanged):
            sig.connect(hook)
        cp = None
        for _ in range(steps):
            k = rng.randrange(9)
            if k <= 2:
                fitting[0] = rng.random() < 0.2
                w = rng.choice(widgets)
                if w is edit:
                    w.setText(rng.choice(["", "a", "b", "G2"]))
                elif w is check:
                    w.setChecked(rng.random() < 0.5)
                else:
                    w.setValue(rng.randint(0, 9))
                fitting[0] = False
            elif k == 3:
                quiet_ctx[0] += 1
                with h.quiet():
                    spin.setValue(rng.randint(0, 9))
                quiet_ctx[0] -= 1
            elif k == 4:
                h.commit(); m.commit()
            elif k == 5:
                m.commit(); h.undo(); m.step(-1, setter)
            elif k == 6:
                m.commit(); h.redo(); m.step(+1, setter)
            elif k == 7:
                cp = h.checkpoint(); m.commit(); mcp = (list(m.states), m.idx)
            elif cp is not None:
                h.rollback(cp); m.timer = False; m.pq = False; m.latest = None; m.states, m.idx = list(mcp[0]), mcp[1]
            ok = (h._states == m.states and h._index == m.idx and h.can_undo() == m.can_undo()
                  and h.can_redo() == m.can_redo())
            tally("19.8 settings history", ok, (h._states[-3:], h._index, m.states[-3:], m.idx))
        for w in widgets:
            w.deleteLater()


# ---------------- random runs ----------------
def random_run(number):
    S = rng.randint(1, 4); opt = rng.choice(list(OptimizerType)); tol = rng.choice([None, 1e-5, 2e-4])
    jc = tuple(rng.choice(["G2", "C1", "G3"]) for _ in range(S - 1)) if (rng.random() < 0.3 and S > 1) else ()
    c = SplineConfig(degree=rng.choice([3, 5]), num_control_points=rng.choice([10, 25]), num_sections=S,
                     continuity=rng.choice([ContinuityMode.G1, ContinuityMode.G2, ContinuityMode.G3]),
                     geometric_continuity=rng.random() < 0.5, junction_continuity=jc)
    o = OptimizationConfig(optimizer=opt, tolerance=tol, knot_placement=rng.choice(["uniform", "free"]),
                           parameterization=rng.choice(["centripetal", "arc_length"]),
                           grow_control_points=rng.random() < 0.5, smoothing_weight=rng.choice([1e-3, 1e-2]),
                           max_points_per_section=rng.choice([None, 100]), optimize_le_position=rng.random() < 0.3)
    secs = [[(0.0, 0.0)] * rng.choice([8, 12]) for _ in range(S)]
    rep = None if rng.random() < 0.3 else {"max_deviation": rng.choice([1e-6, 3e-5, 2e-4]),
                                             "rms_deviation": rng.choice([5e-7, 1e-5, None])}
    res = FitResult(secs, c.degree, rng.choice([1e-6, 4e-5]), rng.randint(1, 50), rng.randint(1, 500),
                    rng.choice([0.05, 2.0, 75.0]), converged=rng.random() < 0.8, report=rep,
                    rms_error=rng.choice([None, 2e-6]))
    return OptimizationRun(f"id{number}", f"2026-11-0{rng.randint(1, 6)}T1{rng.randint(0, 9)}:30:00.5", c, o, res, [], secs, number)


# ---------------- 19.17.1 columns, from the spec table ----------------
def spec_cells(run):
    c, o, r = run.spline_config, run.optimization_config, run.result
    vp = o.optimizer == OptimizerType.GN_VARPRO; tg = vp and o.tolerance is not None; hj = c.num_sections > 1
    rep = r.report or {}
    cells = [
        (run.number, ""), (o.optimizer.display_name, ""), ("Tolerance" if o.tolerance is not None else "Fixed count", "") if vp else (None, ""),
        (o.tolerance, ".3g") if tg else (None, ""), (o.knot_placement.capitalize(), "") if vp else (None, ""),
        ({"centripetal": "Centripetal", "arc_length": "Arc length"}[o.parameterization], "") if not vp else (None, ""),
        (o.grow_control_points, "") if vp else (None, ""), (c.degree, ""), (c.num_control_points, ""), (c.num_sections, ""),
        (c.continuity.name, "") if hj else (None, ""), ("Geometric" if c.geometric_continuity else "Parametric", "") if (vp and hj) else (None, ""),
        (", ".join(c.junction_continuity), "") if vp else (None, ""), (o.max_points_per_section, "") if vp else (None, ""),
        (o.smoothing_weight, ".3g") if (vp and not tg) else (None, ""), (o.optimize_le_position, "") if hj else (None, ""),
        (r.comparison_error, ".3e"), (rep.get("max_deviation"), ".3e"), (rep.get("rms_deviation"), ".3e"),
        (r.num_control_points, ""), (r.num_iterations, ""), (r.num_evaluations, ""), (r.elapsed_seconds, "duration"),
        (r.converged, ""), (run.timestamp[:16].replace("T", " "), ""),
    ]
    def text(v, spec):
        if v is None:
            return ""
        if isinstance(v, bool):
            return "yes" if v else "no"
        if spec == "duration":
            return format_duration(v)
        if isinstance(v, float) and spec:
            return format(v, spec)
        return str(v)
    return [(v, text(v, s)) for v, s in cells]


NAMES = ["Run", "Optimizer", "Goal", "Tolerance", "Knots", "Parameterization", "Grow points", "Degree", "Control points",
         "Sections", "Continuity", "Type", "Per junction", "Subsample", "Smoothing", "Optimize junctions", "Error",
         "Max deviation", "RMS deviation", "Fitted control points", "Iterations", "Evaluations", "Time", "Converged", "Date"]
METRIC = {"Error", "Max deviation", "RMS deviation", "Fitted control points", "Iterations", "Evaluations", "Time"}
LOG = {"Tolerance", "Smoothing", "Error", "Max deviation", "RMS deviation", "Time"}
isnum = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)


def spec_row_passes(cells, terms, excluded, inverted):
    if not terms and not excluded:
        return True
    ok = all(cr_term(t, *cells[16]) if isnum(t[1]) else any(cr_term(t, v, s) for v, s in cells) for t in terms)
    ok = ok and all(cells[c][1] not in tx for c, tx in excluded.items())
    return ok != inverted


def cr_term(term, value, text):
    op, operand = term
    if not op:
        return operand in text.casefold()
    if isinstance(operand, str):
        return (text.casefold() == operand) == (op == "=")
    return isnum(value) and {"<": value < operand if isnum(value) else 0, "<=": isnum(value) and value <= operand,
                             ">": isnum(value) and value > operand, ">=": isnum(value) and value >= operand,
                             "=": isnum(value) and value == operand, "!=": isnum(value) and value != operand}[op]


def explorer_test(rounds=150):
    tally("19.17.1 column names, metric and log flags",
          [c.name for c in ex.COLUMNS] == NAMES and {c.name for c in ex.COLUMNS if c.metric} == METRIC
          and {c.name for c in ex.COLUMNS if c.log} == LOG, [c.name for c in ex.COLUMNS])
    table = ex.RunTable()
    for _ in range(rounds):
        runs = [random_run(i + 1) for i in range(rng.randint(1, 15))]
        table.set_runs(runs)
        cells = [spec_cells(r) for r in runs]
        for r, cl in zip(range(len(runs)), cells):
            tally("19.17.1 cell values and texts", all(table.model.value(r, c) == cl[c][0] and table.model.item(r, c).text() == cl[c][1]
                                                       for c in range(25)), (r, [(table.model.value(r, c), table.model.item(r, c).text(), cl[c]) for c in range(25)
                                                                                 if table.model.value(r, c) != cl[c][0] or table.model.item(r, c).text() != cl[c][1]]))
        for _ in range(8):
            text = " ".join(rng.choice(["varpro", "lsq", ">1e-5", "<=2e-5", "=5", "!=3", "=free", "!=uniform", "yes", "g2", ">3", "2026"])
                            for _ in range(rng.randint(0, 3)))
            excluded = {}
            for _ in range(rng.randint(0, 2)):
                col = rng.randrange(25); vals = {cl[col][1] for cl in cells}
                excluded[col] = frozenset(rng.sample(sorted(vals), rng.randint(0, len(vals))))
            inv = rng.random() < 0.3
            table.column_filters = {c: t for c, t in excluded.items() if t}
            table.filter_edit.setText(text); table.invert_button.setChecked(inv); table._apply_filter()
            terms = cr.parse_filter(text)
            want = {i for i, cl in enumerate(cells) if spec_row_passes(cl, terms, {c: t for c, t in excluded.items() if t}, inv)}
            got = set(table.visible_rows())
            tally("19.17.2 row filter", got == want, (text, excluded, inv, sorted(got), sorted(want)))
            col = rng.randrange(25); order = rng.choice([Qt.SortOrder.AscendingOrder, Qt.SortOrder.DescendingOrder])
            table.sort_by(col, order)
            rows = table.visible_rows()
            def key_ok(a, b):   # spec 19.17.3: blanks last in both orders, numbers by value, else case-folded text
                va, vb = cells[a][col][0], cells[b][col][0]
                if va is None or vb is None:
                    return not (va is None and vb is not None)
                if isnum(va) and isnum(vb):
                    return va <= vb if order == Qt.SortOrder.AscendingOrder else va >= vb
                ta, tb = cells[a][col][1].casefold(), cells[b][col][1].casefold()
                return ta <= tb if order == Qt.SortOrder.AscendingOrder else ta >= tb
            tally("19.17.3 sorting", all(key_ok(a, b) for a, b in zip(rows, rows[1:])), (col, order, [cells[r][col] for r in rows]))
        table.clear_filters()


# ---------------- 19.16.4 matching, 19.13 sweep series ----------------
def spec_match(e, c, o):
    ec, eo = e.spline_config, e.optimization_config
    same = all(getattr(ec, f) == getattr(c, f) for f in ("degree", "num_control_points", "num_sections", "continuity",
                                                         "geometric_continuity", "junction_continuity"))
    same = same and all(getattr(eo, f) == getattr(o, f) for f in ("tolerance", "max_points_per_section", "knot_placement",
                                                                 "parameterization", "optimize_le_position",
                                                                 "grow_control_points", "optimizer"))
    return same and (o.tolerance is not None or eo.smoothing_weight == o.smoothing_weight)


def axis_value(run, name):
    if name == "Smoothing weight":
        return run.optimization_config.smoothing_weight
    if name == "Degree":
        return run.spline_config.degree
    if name == "Sections":
        return run.spline_config.num_sections
    return run.result.num_control_points if run.optimization_config.tolerance is not None else run.spline_config.num_control_points


def configured(run_c, run_o, name):
    return {"Control points": run_c.num_control_points, "Degree": run_c.degree, "Sections": run_c.num_sections,
            "Smoothing weight": run_o.smoothing_weight}[name]


def spec_series(runs, c, o, swept, axis):
    by = {}
    for run in runs:
        rc, ro = run.spline_config, run.optimization_config
        sj = ((rc.continuity == c.continuity and rc.geometric_continuity == c.geometric_continuity) if "Sections" in swept
              else rc.junction_specs() == c.junction_specs())
        if (ro.optimizer != o.optimizer or not sj or (ro.tolerance is None) != (o.tolerance is None)
                or ro.knot_placement != o.knot_placement or ro.parameterization != o.parameterization
                or any(configured(rc, ro, n) != configured(c, o, n) for n in ("Control points", "Degree", "Sections", "Smoothing weight")
                       if n not in swept)):
            continue
        by[tuple(axis_value(run, n) for n in swept)] = (axis_value(run, axis), run.result.comparison_error)
    pts = sorted(by.values())
    return [p[0] for p in pts], [p[1] for p in pts]


def project_test(rounds=3000):
    axes = ["Control points", "Degree", "Sections", "Smoothing weight"]
    for _ in range(rounds):
        runs = [random_run(i + 1) for i in range(rng.randint(1, 12))]
        new = random_run(99)
        e = rng.choice(runs)
        tally("19.16.4 run matching", ProjectManager._configs_match(None, e, new.spline_config, new.optimization_config)
              == spec_match(e, new.spline_config, new.optimization_config))
        tally("19.16.4 run matching (equal configs)", ProjectManager._configs_match(None, e, e.spline_config, e.optimization_config))
        swept = tuple(sorted(rng.sample(axes, rng.randint(1, 3)), key=axes.index)); axis = rng.choice(swept)
        dummy = types.SimpleNamespace(_project=types.SimpleNamespace(optimization_runs=runs),
                                      _get_run_axis_value=ProjectManager._get_run_axis_value)
        got = ProjectManager.sweep_series(dummy, new.spline_config, new.optimization_config, swept, axis)
        tally("19.13 sweep series", got == spec_series(runs, new.spline_config, new.optimization_config, swept, axis),
              (swept, axis, got, spec_series(runs, new.spline_config, new.optimization_config, swept, axis)))



# Rules under test: this file's own (default), or those of IMPL_GUI (a path) defining
# History(cap), spec_cells(run), spec_row_passes(cells, terms, excluded, inverted),
# spec_match(existing, c, o) and spec_series(runs, c, o, swept, axis); IMPL (a path) supplies
# parse_filter for the row filter terms
if os.environ.get("IMPL_GUI"):
    import importlib.util as _u
    _s = _u.spec_from_file_location("impl_gui", os.environ["IMPL_GUI"]); _impl = _u.module_from_spec(_s); _s.loader.exec_module(_impl)
    HistoryModel = _impl.History
    spec_cells, spec_row_passes = _impl.spec_cells, _impl.spec_row_passes
    spec_match, spec_series = _impl.spec_match, _impl.spec_series
if os.environ.get("IMPL"):
    import importlib.util as _u
    _s = _u.spec_from_file_location("impl_core", os.environ["IMPL"]); cr = _u.module_from_spec(_s); _s.loader.exec_module(cr)

if __name__ == "__main__":
    history_test(); explorer_test(); project_test()
    total = bad = 0
    for k, (n, f, d) in report.items():
        total += n; bad += f
        print(f"{'FAIL' if f else 'ok  '} {k}: {n - f}/{n}" + (f"   first mismatch: {str(d)[:500]}" if f else ""))
    print(f"\n{total - bad} of {total} comparisons agree")
    sys.exit(1 if bad else 0)
