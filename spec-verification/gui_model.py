"""Model test of spec.md 19.6 (visibility and enabling) and 19.7 (curve validity rule).

A model written from the spec text keeps its own copy of the sidebar state. Random
user actions are applied to the model and to a real SplineFitterWindow; after every
action the relevant widget states must agree.
"""
import os, random, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "packages", "bspline-varpro", "src")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication

UNL = 2 ** 31 - 1


def min_ctrl(orders, p):
    e = [0, *orders, 0]
    return max(p + 1, *(a + b + 2 for a, b in zip(e, e[1:])))


class Spin:
    """Qt QSpinBox value semantics (clamping on range changes)."""
    def __init__(self, m, name, v, lo, hi):
        self.m, self.name, self.v, self.lo, self.hi = m, name, v, lo, hi
    def _emit(self, old):
        if self.v != old:
            self.m.signal(self.name)
    def set(self, v):
        old = self.v; self.v = min(max(v, self.lo), self.hi); self._emit(old)
    def set_min(self, lo):
        old = self.v; self.lo = lo; self.hi = max(self.hi, lo); self.v = min(max(self.v, self.lo), self.hi); self._emit(old)
    def set_max(self, hi):
        old = self.v; self.hi = hi; self.lo = min(self.lo, hi); self.v = min(max(self.v, self.lo), self.hi); self._emit(old)


class Sweep:
    """19.21 item 3."""
    def __init__(self, m, key, spin, end):
        self.m, self.key, self.spin, self.end, self.active, self.allowed = m, key, spin, end, False, True
    def ends(self):
        return [self.spin, self.end] if self.active else [self.spin]
    def toggle(self, on):
        self.active = on
        if on:
            self.end.set_min(self.spin.lo); self.end.set(min(self.spin.v * 2, self.spin.hi))
        self.m.signal(self.end.name + "!")
    def stop(self):
        self.toggle(False)
    def set_allowed(self, allowed):
        if not allowed and self.active:
            self.stop()
        self.allowed = allowed
    def set_minimum(self, lo):
        self.spin.set_min(lo); self.end.set_min(lo)
    def set_sweep_maximum(self, hi):
        self.end.set_max(hi)


class Model:
    def __init__(self, w):
        self.guard = False
        g = lambda s: (s.value(), s.minimum(), s.maximum())
        self.varpro = w._combo_opt.currentIndex() == 0
        self.tol_goal = bool(w._seg_goal.currentData()); self.grow = w._check_grow.isChecked()
        self.auto = w._check_auto_tolerance.isChecked(); self.cont = w._combo_cont.currentIndex()
        self.geometric = bool(w._combo_cont_kind.currentData()); self.junctions = w._edit_junctions.text()
        self.fitting = False
        mk = lambda name, s: Spin(self, name, *g(s))
        self.degree = Sweep(self, "degree", mk("degree", w._spin_degree), mk("degree", w._sweep_degree._spin_max))
        self.count = Sweep(self, "count", mk("count", w._spin_ctrl), mk("count", w._sweep_ctrl._spin_max))
        self.sections = Sweep(self, "sections", mk("sections", w._spin_sections), mk("sections", w._sweep_sections._spin_max))
        self.sections.spin.name = "sections_value"
        self.smooth_active = False
        for s in (self.degree, self.count, self.sections):
            s.active = getattr(w, {"degree": "_sweep_degree", "count": "_sweep_ctrl", "sections": "_sweep_sections"}[s.key]).is_sweep_active

    # signal wiring (19.6, 19.7 triggers; 19.21 item 3)
    def signal(self, name):
        if name in ("degree", "count", "sections"):           # value or sweep end changed
            self.keep(name)
        elif name in ("degree!", "count!", "sections!"):      # sweep toggled
            self.keep(name[:-1])
        elif name == "sections_value":
            self.keep("sections"); self.on_optimizer_changed()
        elif name == "continuity":
            self.keep("continuity")

    def tol_mode(self):
        return self.varpro and self.tol_goal

    def count_used(self):
        return not self.tol_mode() or not self.grow

    def junction_orders(self, S):
        if self.varpro:
            specs = self.junctions.replace(",", " ").replace(";", " ").split()
            if specs and len(specs) == S - 1 and all(len(s) == 2 and s[0].upper() in "CG" and s[1] in "0123" for s in specs):
                return [int(s[1]) for s in specs], True
        return [self.cont] * (S - 1), False

    def set_cont(self, i):
        if i != self.cont:
            self.cont = i; self.signal("continuity")

    def on_optimizer_changed(self):
        if self.tol_mode():
            if self.count.active:
                self.count.stop()
            self.smooth_active = False
        self.count.set_allowed(not self.tol_mode())
        self.keep("")

    def keep(self, changed):
        if self.guard:
            return
        self.guard = True
        try:
            self.resolve(changed)
        finally:
            self.guard = False

    def resolve(self, changed):
        deg, cnt, sec = self.degree, self.count, self.sections
        open_sweep = cnt.active and (deg.active or sec.active)
        low = lambda b: [min(b.ends(), key=lambda s: s.v)] if open_sweep else b.ends()
        high = lambda b: [max(b.ends(), key=lambda s: s.v)] if open_sweep else b.ends()
        def move(spins, limit, up):
            for s in spins:
                if (s.v < limit) if up else (s.v > limit):
                    s.set(limit)
        orders = lambda: self.junction_orders(max(s.v for s in low(sec)))
        need = lambda: min_ctrl(orders()[0], max(s.v for s in low(deg)))
        lower = lambda k: self.set_cont(min(k, self.cont))
        lowest = min(s.v for s in deg.ends())
        if max(orders()[0], default=0) > lowest:
            if changed in ("degree", "count") and not orders()[1]:
                lower(lowest)
            else:
                move(deg.ends(), max(orders()[0]), True)
        if self.count_used():
            have = min(s.v for s in high(cnt))
            if need() > have:
                if changed == "count":
                    move(low(deg), have - 1, False)
                    if need() > have and sec.active:
                        move(sec.ends()[-1:], max(sec.spin.v, 2), False)
                    while need() > have and not orders()[1] and self.cont:
                        lower(self.cont - 1)
                else:
                    move(high(cnt), need(), True)
        fixed_orders, fixed = self.junction_orders(sec.spin.v)
        dmin = max(fixed_orders, default=1) if fixed else 1
        deg.set_minimum(dmin)
        cnt.set_minimum(min_ctrl(fixed_orders, dmin) if fixed else 2)
        dcap, scap = deg.spin.hi, sec.spin.hi
        if self.count_used() and not cnt.active:
            have = cnt.spin.v
            dcap = max(have - 1, deg.spin.lo)
            fitting = [k for k in range(1, min(scap, 3) + 1) if min_ctrl([self.cont] * (k - 1), 1) <= have]
            if fitting[-1] < 3:
                scap = fitting[-1]
            scap = max(scap, sec.spin.v)
        deg.set_sweep_maximum(dcap); sec.set_sweep_maximum(scap)

    # 19.6 expected widget flags
    def flags(self):
        v, t, hj, ed = self.varpro, self.tol_mode(), self.sections.spin.v > 1, not self.fitting
        return {
            "goal shown": v, "parameterization shown": not v, "type shown": v, "per junction shown": v,
            "tolerance label enabled": t, "tolerance spin enabled": ed and t and not self.auto,
            "auto enabled": ed and t, "count label": "Start count" if t else "Control points",
            "count label enabled": self.count_used(), "count sweep enabled": ed and self.count_used(),
            "count sweep allowed": not t, "continuity enabled": ed and hj, "continuity label enabled": hj,
            "continuity texts": [("G" if (v and self.geometric) else "C") + str(i) for i in range(4)],
            "junction field enabled": hj, "smoothing enabled": not t, "optimize junctions enabled": hj,
        }

    def state(self):
        s = lambda x: (x.v, x.lo, x.hi)
        return {"degree": s(self.degree.spin), "degree end": s(self.degree.end), "count": s(self.count.spin),
                "count end": s(self.count.end), "sections": s(self.sections.spin), "sections end": s(self.sections.end),
                "continuity": self.cont, "sweeps": (self.degree.active, self.count.active, self.sections.active,
                                                    self.smooth_active), **self.flags()}


# Rules under test: this file's own (default), or those of a module named by IMPL_GUI (a path)
# that defines resolve(m, changed), on_optimizer_changed(m), junction_orders(m, S) and flags(m)
if os.environ.get("IMPL_GUI"):
    import importlib.util as _u
    _s = _u.spec_from_file_location("impl_gui", os.environ["IMPL_GUI"]); _impl = _u.module_from_spec(_s); _s.loader.exec_module(_impl)
    Model.resolve = lambda self, changed: _impl.resolve(self, changed)
    Model.on_optimizer_changed = lambda self: _impl.on_optimizer_changed(self)
    Model.junction_orders = lambda self, S: _impl.junction_orders(self, S)
    Model.flags = lambda self: _impl.flags(self)


def window_state(w):
    s = lambda x: (x.value(), x.minimum(), x.maximum())
    vis = lambda x: not x.isHidden()  # its own shown flag (Advanced may be collapsed)
    return {"degree": s(w._spin_degree), "degree end": s(w._sweep_degree._spin_max), "count": s(w._spin_ctrl),
            "count end": s(w._sweep_ctrl._spin_max), "sections": s(w._spin_sections),
            "sections end": s(w._sweep_sections._spin_max), "continuity": w._combo_cont.currentIndex(),
            "sweeps": (w._sweep_degree.is_sweep_active, w._sweep_ctrl.is_sweep_active, w._sweep_sections.is_sweep_active,
                       w._sweep_smoothing_weight.is_sweep_active),
            "goal shown": vis(w._seg_goal), "parameterization shown": vis(w._seg_parameterization),
            "type shown": vis(w._combo_cont_kind), "per junction shown": vis(w._edit_junctions),
            "tolerance label enabled": w._lbl_tolerance.isEnabled(), "tolerance spin enabled": w._spin_tolerance.isEnabled(),
            "auto enabled": w._check_auto_tolerance.isEnabled(), "count label": w._lbl_ctrl.text(),
            "count label enabled": w._lbl_ctrl.isEnabled(), "count sweep enabled": w._sweep_ctrl.isEnabled(),
            "count sweep allowed": w._sweep_ctrl._btn_sweep.isEnabled(), "continuity enabled": w._combo_cont.isEnabled(),
            "continuity label enabled": w._lbl_cont.isEnabled(),
            "continuity texts": [w._combo_cont.itemText(i) for i in range(4)],
            "junction field enabled": w._edit_junctions.isEnabled(), "smoothing enabled": w._sweep_smoothing_weight.isEnabled(),
            "optimize junctions enabled": w._check_optimize_le.isEnabled()}


def main(steps=4000, seed=7):
    app = QApplication.instance() or QApplication([])
    from spline_fitter.gui.ui import SplineFitterWindow
    w = SplineFitterWindow(); w.show(); app.processEvents()
    w.new_naca_airfoil("2412", 30, True)   # 59 points: sections at most 58
    m = Model(w)
    rng = random.Random(seed)
    mismatches, n = [], 0

    def act():
        k = rng.randrange(13)
        if k == 0:
            i = rng.randrange(2); w._combo_opt.setCurrentIndex(i)
            if (i == 0) != m.varpro:
                m.varpro = i == 0; m.on_optimizer_changed()
            return f"optimizer {i}"
        if k == 1:
            i = rng.randrange(2); w._seg_goal.setCurrentIndex(i)
            if bool(i) != m.tol_goal:
                m.tol_goal = bool(i); m.on_optimizer_changed()
            return f"goal {i}"
        if k == 2:
            b = rng.random() < 0.5; w._check_grow.setChecked(b)
            if b != m.grow:
                m.grow = b; m.on_optimizer_changed()
            return f"grow {b}"
        if k == 3:
            b = rng.random() < 0.5; w._check_auto_tolerance.setChecked(b)
            if b != m.auto:
                m.auto = b; m.on_optimizer_changed()
            return f"auto {b}"
        if k == 4:
            v = rng.randint(1, 9); w._spin_degree.setValue(v); m.degree.spin.set(v); return f"degree {v}"
        if k == 5:
            v = rng.randint(2, 30); w._spin_ctrl.setValue(v); m.count.spin.set(v); return f"count {v}"
        if k == 6:
            v = rng.randint(1, 7); w._spin_sections.setValue(v); m.sections.spin.set(v); return f"sections {v}"
        if k == 7:
            i = rng.randrange(4); w._combo_cont.setCurrentIndex(i); m.set_cont(i); return f"continuity {i}"
        if k == 8:
            i = rng.randrange(2); w._combo_cont_kind.setCurrentIndex(i); m.geometric = i == 0; return f"type {i}"
        if k == 9:
            txt = rng.choice(["", "G2", "G3", "C1 G2", "G1, C3", "g2", "X9", "C0;C0;C0", "G3 G3 G3 G3"])
            w._edit_junctions.setText(txt); w._edit_junctions.editingFinished.emit(); m.junctions = txt; m.keep("")
            return f"junctions {txt!r}"
        if k == 10:
            which = rng.choice(["degree", "count", "sections"])
            box = {"degree": w._sweep_degree, "count": w._sweep_ctrl, "sections": w._sweep_sections}[which]
            sw = getattr(m, which)
            if not box._btn_sweep.isEnabled() or not box.isEnabled():
                return f"sweep {which} (not allowed)"
            box._btn_sweep.click(); sw.toggle(not sw.active); return f"toggle sweep {which}"
        if k == 11:
            which = rng.choice(["degree", "count", "sections"])
            box = {"degree": w._sweep_degree, "count": w._sweep_ctrl, "sections": w._sweep_sections}[which]
            sw = getattr(m, which)
            if not sw.active:
                return f"sweep end {which} (inactive)"
            v = rng.randint(1, 40); box._spin_max.setValue(v); sw.end.set(v); return f"sweep end {which} {v}"
        b = rng.random() < 0.5
        if w._sweep_smoothing_weight.isEnabled():
            if b != m.smooth_active:
                w._sweep_smoothing_weight._btn_sweep.click(); m.smooth_active = b
        return f"smoothing sweep {b}"

    trail = []
    for _ in range(steps):
        trail.append(act()); trail = trail[-6:]
        app.processEvents(); n += 1
        a, b = window_state(w), m.state()
        diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
        if diff:
            mismatches.append((list(trail), diff))
            m = Model(w)   # resynchronise and continue
    w._project_manager._project_modified = False
    w.close(); w.deleteLater(); app.processEvents(); del w
    print(f"{n - len(mismatches)} of {n} random actions left window and model in the same state")
    for t, d in mismatches[:8]:
        print("  after", t, "->", d)
    return len(mismatches)


if __name__ == "__main__":
    sys.exit(1 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000) else 0)
