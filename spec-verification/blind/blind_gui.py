"""Blind reimplementation of GUI rules (spec.md 19.6, 19.7, 19.8, 19.13, 19.16.4, 19.17).

Written only from spec.md. Ambiguities are marked with ``# SPEC-GAP:`` comments.
"""
from __future__ import annotations

import operator

DECIMAL_POINT = "."


# ---------------------------------------------------------------- helpers (5.2, 18)
def minimum_control_points(orders, p):
    seq = [0, *orders, 0]
    return max(p + 1, max(a + b + 2 for a, b in zip(seq[:-1], seq[1:])))


def number(v, spec="g"):
    return format(v, spec).replace(".", DECIMAL_POINT)


def format_duration(s):
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.0f} ms"
    if s < 60:
        return number(s, ".2f") + " s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {int(sec)}s"
    h, m = divmod(int(m), 60)
    return f"{h}h {m}m {int(sec)}s"


# ---------------------------------------------------------------- 19.7
def junction_orders(m, S):
    S = int(S)
    if m.varpro:
        words = str(m.junctions).replace(",", " ").replace(";", " ").split()
        if len(words) == S - 1 and all(
            len(w) == 2 and w[0] in "CGcg" and w[1] in "0123" for w in words
        ):
            return [int(w[1]) for w in words], True
    return [int(m.cont)] * (S - 1), False


def _open_sweep(m):
    return m.count.active and (m.degree.active or m.sections.active)


def _low_end(m, w):
    ends = list(w.ends())
    if _open_sweep(m):
        return [min(ends, key=lambda b: b.v)]
    return ends


def _high_end(m, w):
    ends = list(w.ends())
    if _open_sweep(m):
        return [max(ends, key=lambda b: b.v)]
    return ends


def _orders(m):
    return junction_orders(m, max(b.v for b in _low_end(m, m.sections)))


def _need(m):
    orders, _ = _orders(m)
    return minimum_control_points(orders, max(b.v for b in _low_end(m, m.degree)))


def _lower_continuity(m, k):
    m.set_cont(min(k, m.cont))


def _move(boxes, limit, up):
    for b in boxes:
        if up and b.v < limit:
            b.set(limit)
        elif not up and b.v > limit:
            b.set(limit)


def resolve(m, changed):
    # 1. continuity vs degree
    lowest = min(b.v for b in m.degree.ends())
    orders, from_pj = _orders(m)
    top = max(orders) if orders else 0
    if top > lowest:
        if changed in ("degree", "count") and not from_pj:
            _lower_continuity(m, lowest)
        else:
            _move(m.degree.ends(), top, up=True)
    # 2. control points vs need
    if m.count_used():
        have = min(b.v for b in _high_end(m, m.count))
        if _need(m) > have:
            if changed == "count":
                _move(_low_end(m, m.degree), have - 1, up=False)
                if _need(m) > have and m.sections.active:
                    _move([m.sections.ends()[-1]], max(m.sections.spin.v, 2), up=False)
                while True:
                    _o, pj = _orders(m)
                    if not (_need(m) > have and not pj and m.cont > 0):
                        break
                    before = m.cont
                    _lower_continuity(m, m.cont - 1)
                    if m.cont == before:
                        break
            else:
                _move(_high_end(m, m.count), _need(m), up=True)
    # 3. hard minimums
    fixed_orders, fixed = junction_orders(m, m.sections.spin.v)
    if fixed:
        # SPEC-GAP: 19.7 step 3 with one section and VarPro, an empty Per junction text
        # gives ([], True) and max([]) is undefined in the revised text; 1 is used.
        dmin = max(fixed_orders) if fixed_orders else 1
        cmin = minimum_control_points(fixed_orders, dmin)
    else:
        dmin, cmin = 1, 2
    m.degree.set_minimum(dmin)
    m.count.set_minimum(cmin)
    # 4. sweep caps
    degree_cap = m.degree.spin.hi
    sections_cap = m.sections.spin.hi
    if m.count_used() and not m.count.active:
        have = m.count.spin.v
        degree_cap = max(have - 1, dmin)
        order = int(m.cont)
        fitting = [
            k for k in range(1, int(min(sections_cap, 3)) + 1)
            if minimum_control_points([order] * (k - 1), 1) <= have
        ]
        if fitting and fitting[-1] < 3:
            sections_cap = fitting[-1]
        sections_cap = max(sections_cap, m.sections.spin.v)
    m.degree.set_sweep_maximum(degree_cap)
    m.sections.set_sweep_maximum(sections_cap)


# ---------------------------------------------------------------- 19.6
def on_optimizer_changed(m):
    tol = m.tol_mode()
    if tol:
        # 19.6 step 3: only sweeps that are on are stopped; then the toggle's allowed state.
        if m.count.active:
            m.count.stop()
        if m.smooth_active:
            m.smooth_active = False
    m.count.set_allowed(not tol)
    # Advanced badge (19.9) is not modelled here.
    m.keep("")


def flags(m):
    varpro = bool(m.varpro)
    tol = bool(m.tol_mode())
    has_j = m.sections.spin.v > 1
    editable = not m.fitting
    count_used = bool(m.count_used())
    # 19.6 step 5: the Advanced content is disabled while fitting, so its widgets also
    # require "editable".
    return {
        "goal shown": varpro,
        "parameterization shown": not varpro,
        "type shown": varpro,
        "per junction shown": varpro,
        "tolerance label enabled": tol,
        "tolerance spin enabled": editable and tol and not m.auto,
        "auto enabled": editable and tol,
        "count label": "Start count" if tol else "Control points",
        "count label enabled": count_used,
        "count sweep enabled": editable and count_used,
        "count sweep allowed": not tol,
        "continuity enabled": editable and has_j,
        "continuity label enabled": has_j,
        "continuity texts": [("G" if varpro and m.geometric else "C") + str(i) for i in range(4)],
        "junction field enabled": has_j and editable,
        "smoothing enabled": (not tol) and editable,
        "optimize junctions enabled": has_j and editable,
    }


# ---------------------------------------------------------------- 19.8
class History:
    LIMIT = 200

    def __init__(self, cap):
        self.cap = cap
        self.states = [cap()]
        self.idx = 0
        self.timer = False
        self.pq = False
        self.latest = None
        self.restoring = False

    def _settle(self, state, quiet):
        if quiet:
            self.states[self.idx] = state
        elif state != self.states[self.idx]:
            del self.states[self.idx + 1:]
            self.states.append(state)
            if len(self.states) > self.LIMIT:
                del self.states[: len(self.states) - self.LIMIT]
            self.idx = len(self.states) - 1

    def on_change(self, quiet):
        if self.restoring:
            return
        if self.timer and quiet != self.pq:
            self._settle(self.latest, self.pq)
        self.pq = quiet
        self.latest = self.cap()
        self.timer = True

    def commit(self):
        self.timer = False
        self._settle(self.cap(), self.pq)
        self.pq = False
        self.latest = None

    def can_undo(self):
        return self.idx > 0 or (self.timer and not self.pq)

    def can_redo(self):
        return self.idx < len(self.states) - 1 and not self.timer

    def step(self, delta, setter):
        target = self.idx + delta
        if delta < 0 and self.idx <= 0:
            return
        if delta > 0 and self.idx >= len(self.states) - 1:
            return
        self.idx = target
        self.restoring = True
        try:
            setter(self.states[self.idx])
        finally:
            self.restoring = False
        self.states[self.idx] = self.cap()


# ---------------------------------------------------------------- 19.17.1
_DURATION = "duration"


def _cell_text(v, fmt):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if fmt == _DURATION:
        return format_duration(v)
    if fmt and isinstance(v, float):
        return number(v, fmt)
    return str(v)


def spec_cells(run):
    c, o, r = run.spline_config, run.optimization_config, run.result
    varpro = getattr(o.optimizer, "value", o.optimizer) == "gn-varpro"
    tol = o.tolerance
    S = c.num_sections
    report = r.report if isinstance(r.report, dict) else None
    param_text = {"centripetal": "Centripetal", "arc_length": "Arc length"}
    rows = [
        (run.number, None),
        (o.optimizer.display_name, None),
        (("Tolerance" if tol is not None else "Fixed count") if varpro else None, None),
        (tol if (varpro and tol is not None) else None, ".3g"),
        (str(o.knot_placement).capitalize() if varpro else None, None),
        (param_text.get(o.parameterization, str(o.parameterization)) if not varpro else None, None),
        (bool(o.grow_control_points) if varpro else None, None),
        (c.degree, None),
        (c.num_control_points, None),
        (S, None),
        (c.continuity.name if S > 1 else None, None),
        (("Geometric" if c.geometric_continuity else "Parametric") if (varpro and S > 1) else None, None),
        (", ".join(c.junction_continuity) if varpro else None, None),
        (o.max_points_per_section if varpro else None, None),
        (o.smoothing_weight if (varpro and tol is None) else None, ".3g"),
        (bool(o.optimize_le_position) if S > 1 else None, None),
        (r.comparison_error, ".3e"),
        (report.get("max_deviation") if report else None, ".3e"),
        (report.get("rms_deviation") if report else None, ".3e"),
        (r.num_control_points, None),
        (r.num_iterations, None),
        (r.num_evaluations, None),
        (r.elapsed_seconds, _DURATION),
        (r.converged, None),
        (str(run.timestamp)[:16].replace("T", " "), None),
    ]
    return [(v, _cell_text(v, f)) for v, f in rows]


# ---------------------------------------------------------------- 19.17.2
ERROR_COLUMN = 16  # 0-based index of "Error"

_CMP = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "=": operator.eq, "!=": operator.ne,
}


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _term_matches(term, value, text):
    op, operand = term
    if op == "":
        return operand in text.casefold()
    if isinstance(operand, str):
        return (text.casefold() == operand) == (op == "=")
    return _is_number(value) and _CMP[op](value, operand)


def spec_row_passes(cells, terms, excluded, inverted):
    active_cols = {k: v for k, v in (excluded or {}).items() if v}
    if not terms and not active_cols:
        return True
    ok = True
    for term in terms:
        op, operand = term
        if op != "" and not isinstance(operand, str):
            value, text = cells[ERROR_COLUMN]
            if not _term_matches(term, value, text):
                ok = False
                break
        else:
            if not any(_term_matches(term, v, t) for v, t in cells):
                ok = False
                break
    if ok:
        for col, ex in active_cols.items():
            if cells[col][1] in ex:
                ok = False
                break
    return (not ok) if inverted else ok


# ---------------------------------------------------------------- 19.16.4
def spec_match(existing_run, c, o):
    ec, eo = existing_run.spline_config, existing_run.optimization_config
    same = (
        ec.degree == c.degree
        and ec.num_control_points == c.num_control_points
        and ec.num_sections == c.num_sections
        and ec.continuity == c.continuity
        and ec.geometric_continuity == c.geometric_continuity
        and tuple(ec.junction_continuity) == tuple(c.junction_continuity)
        and eo.tolerance == o.tolerance
        and eo.max_points_per_section == o.max_points_per_section
        and eo.knot_placement == o.knot_placement
        and eo.parameterization == o.parameterization
        and eo.optimize_le_position == o.optimize_le_position
        and eo.grow_control_points == o.grow_control_points
        and eo.optimizer == o.optimizer
    )
    if same and o.tolerance is None:
        same = eo.smoothing_weight == o.smoothing_weight
    return same


# ---------------------------------------------------------------- 19.13
def _axis_value(run, name):
    rc, ro = run.spline_config, run.optimization_config
    if name == "Smoothing weight":
        return ro.smoothing_weight
    if name == "Degree":
        return rc.degree
    if name == "Sections":
        return rc.num_sections
    if name == "Control points":
        if ro.tolerance is not None:
            return run.result.num_control_points
        return rc.num_control_points
    raise ValueError(name)


_CONFIGURED = {
    "Control points": lambda c, o: c.num_control_points,
    "Degree": lambda c, o: c.degree,
    "Sections": lambda c, o: c.num_sections,
    "Smoothing weight": lambda c, o: o.smoothing_weight,
}


def spec_series(runs, c, o, swept, axis):
    swept = tuple(swept)
    points = {}
    for run in runs:
        rc, ro = run.spline_config, run.optimization_config
        if ro.optimizer != o.optimizer:
            continue
        if "Sections" in swept:
            if rc.continuity != c.continuity or rc.geometric_continuity != c.geometric_continuity:
                continue
        elif list(rc.junction_specs()) != list(c.junction_specs()):
            continue
        if (ro.tolerance is None) != (o.tolerance is None):
            continue
        if ro.knot_placement != o.knot_placement or ro.parameterization != o.parameterization:
            continue
        if any(
            _CONFIGURED[name](rc, ro) != _CONFIGURED[name](c, o)
            for name in _CONFIGURED if name not in swept
        ):
            continue
        key = tuple(_axis_value(run, name) for name in swept)
        # Later runs replace earlier ones with the same key (dict keeps first position,
        # but order is irrelevant because the points are sorted).
        points[key] = (_axis_value(run, axis), run.result.comparison_error)
    # 19.13: sorted as (x, error) tuples.
    pts = sorted(points.values())
    return [p[0] for p in pts], [p[1] for p in pts]
