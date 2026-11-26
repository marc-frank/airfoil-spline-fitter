//! The sidebar's settings, their limits and the rules that keep them
//! fittable.

use spline_fitter::bspline_varpro::KnotPlacement;
use spline_fitter::formatting::number;
use spline_fitter::models::{
    minimum_control_points, placement_from_value, placement_value, ContinuityMode, OptimizationConfig, OptimizerType, Parameterization,
    SplineConfig, ToleranceMetric,
};
use spline_fitter::project::OptimizationRun;
use spline_fitter::Result;

/// Maximum of integer fields without a limit.
pub const UNLIMITED: usize = i32::MAX as usize;

/// An integer field with its range; values are clamped into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntField {
    pub value: usize,
    pub min: usize,
    pub max: usize,
}

impl IntField {
    pub const fn new(value: usize, min: usize, max: usize) -> Self {
        IntField { value, min, max }
    }

    pub fn set(&mut self, value: usize) {
        self.value = value.clamp(self.min, self.max);
    }

    pub fn set_min(&mut self, min: usize) {
        self.min = min;
        self.max = self.max.max(min);
        self.value = self.value.max(min);
    }

    pub fn set_max(&mut self, max: usize) {
        self.max = max;
        self.min = self.min.min(max);
        self.value = self.value.min(max);
    }
}

/// An integer setting that can sweep a range of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntSweep {
    pub value: IntField,
    pub active: bool,
    pub end: IntField,
    /// Whether the sweep toggle can be turned on.
    pub allowed: bool,
}

impl IntSweep {
    pub fn new(value: usize, min: usize, max: usize) -> Self {
        IntSweep { value: IntField::new(value, min, max), active: false, end: IntField::new((2 * value).min(max), min, max), allowed: true }
    }

    /// Turns the sweep on (end at twice the value) or off.
    pub fn toggle(&mut self, on: bool) {
        if on == self.active || (on && !self.allowed) {
            return;
        }
        self.active = on;
        if on {
            self.end.min = self.value.min;
            self.end.set((2 * self.value.value).min(self.value.max));
        }
    }

    pub fn set_allowed(&mut self, allowed: bool) {
        if !allowed {
            self.toggle(false);
        }
        self.allowed = allowed;
    }

    /// Both boxes' minimum.
    pub fn set_min(&mut self, min: usize) {
        self.value.set_min(min);
        self.end.set_min(min);
    }

    /// The value box and, while sweeping, the end box.
    fn ends_mut(&mut self) -> Vec<&mut IntField> {
        if self.active {
            vec![&mut self.value, &mut self.end]
        } else {
            vec![&mut self.value]
        }
    }

    fn ends(&self) -> Vec<usize> {
        if self.active {
            vec![self.value.value, self.end.value]
        } else {
            vec![self.value.value]
        }
    }

    /// The swept values, ascending.
    pub fn values(&self) -> Vec<usize> {
        let (a, b) = (self.value.value, if self.active { self.end.value } else { self.value.value });
        (a.min(b)..=a.max(b)).collect()
    }
}

/// A floating-point setting that can sweep a range with a step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatSweep {
    pub value: f64,
    pub active: bool,
    pub end: f64,
    pub step: f64,
    pub single_step: f64,
}

impl FloatSweep {
    pub fn new(value: f64, single_step: f64) -> Self {
        FloatSweep { value, active: false, end: value, step: single_step, single_step }
    }

    /// Turns the sweep on (from the value, or one step, to ten times it) or off.
    pub fn toggle(&mut self, on: bool) {
        if on == self.active {
            return;
        }
        self.active = on;
        if on {
            let start = if self.value != 0.0 { self.value } else { self.single_step };
            self.end = 10.0 * start;
            self.step = start;
        }
    }

    /// The swept values: from the smaller end by the step, at most `limit`.
    pub fn values(&self) -> Vec<f64> {
        if !self.active {
            return vec![self.value];
        }
        let (low, high) = if self.value <= self.end { (self.value, self.end) } else { (self.end, self.value) };
        let count = (((high - low) / self.step * (1.0 + 1e-9)).floor() as usize + 1).min(1000);
        (0..count)
            .map(|i| {
                let v = low + i as f64 * self.step;
                round12(v)
            })
            .collect()
    }
}

/// `float(f"{v:.12g}")`.
fn round12(v: f64) -> f64 {
    spline_fitter::formatting::format_spec(v, ".12g").parse().unwrap_or(v)
}

/// What a change touched, for the curve rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    Degree,
    Count,
    Sections,
    Continuity,
    Other,
}

/// Every setting of the sidebar.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub optimizer: OptimizerType,
    /// Tolerance goal (else a fixed count).
    pub tolerance_goal: bool,
    pub auto_tolerance: bool,
    pub tolerance: f64,
    /// What the tolerance limits.
    pub tolerance_metric: ToleranceMetric,
    pub knot_placement: KnotPlacement,
    pub parameterization: Parameterization,
    pub grow: bool,
    pub sections: IntSweep,
    pub degree: IntSweep,
    pub continuity: usize,
    pub geometric: bool,
    /// Join the sections at the corners of the data with G0.
    pub corners: bool,
    pub control_points: IntSweep,
    pub per_junction: String,
    /// Smallest spacing of free knots, as a fraction of a section.
    pub knot_gap: f64,
    pub smoothing: FloatSweep,
    pub optimize_junctions: bool,
}

impl Default for Settings {
    /// The defaults of the models (one set of defaults).
    fn default() -> Self {
        let (c, o) = (SplineConfig::default(), OptimizationConfig::default());
        let mut s = Settings {
            optimizer: o.optimizer,
            tolerance_goal: o.tolerance.is_some(),
            auto_tolerance: true,
            tolerance: o.tolerance.unwrap_or(1e-5),
            tolerance_metric: o.tolerance_metric,
            knot_placement: o.knot_placement,
            parameterization: o.parameterization,
            grow: o.grow_control_points,
            sections: IntSweep::new(c.num_sections, 1, UNLIMITED),
            degree: IntSweep::new(c.degree, 1, UNLIMITED),
            continuity: c.continuity.order(),
            geometric: c.geometric_continuity,
            corners: c.split_at_corners,
            control_points: IntSweep::new(c.num_control_points, 2, UNLIMITED),
            per_junction: c.junction_continuity.join(", "),
            knot_gap: o.min_knot_gap,
            smoothing: FloatSweep::new(o.smoothing_weight, 1e-3),
            optimize_junctions: o.optimize_le_position,
        };
        s.optimizer_changed();
        s
    }
}

/// The values of the settings, without their limits: what undo restores.
#[derive(Debug, Clone, PartialEq)]
pub struct Values {
    optimizer: OptimizerType,
    tolerance_goal: bool,
    auto_tolerance: bool,
    tolerance: f64,
    tolerance_metric: ToleranceMetric,
    knot_placement: KnotPlacement,
    parameterization: Parameterization,
    grow: bool,
    sections: usize,
    degree: usize,
    continuity: usize,
    geometric: bool,
    corners: bool,
    control_points: usize,
    per_junction: String,
    knot_gap: f64,
    smoothing: f64,
    optimize_junctions: bool,
    degree_sweep: (bool, usize),
    count_sweep: (bool, usize),
    sections_sweep: (bool, usize),
    smoothing_sweep: (bool, f64, f64),
}

impl Values {
    /// The values as kept between sessions (without the sweeps, which
    /// start off).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "optimizer": self.optimizer.value(),
            "tolerance_goal": self.tolerance_goal,
            "auto_tolerance": self.auto_tolerance,
            "tolerance": self.tolerance,
            "tolerance_metric": self.tolerance_metric.value(),
            "knot_placement": placement_value(self.knot_placement),
            "parameterization": self.parameterization.value(),
            "grow": self.grow,
            "sections": self.sections,
            "degree": self.degree,
            "continuity": self.continuity,
            "geometric": self.geometric,
            "corners": self.corners,
            "control_points": self.control_points,
            "per_junction": self.per_junction,
            "knot_gap": self.knot_gap,
            "smoothing": self.smoothing,
            "optimize_junctions": self.optimize_junctions,
        })
    }

    /// Values read back from [`Values::to_json`]: a missing or unreadable
    /// entry keeps its value in `defaults`, and every sweep is off.
    pub fn from_json(json: &serde_json::Value, defaults: &Values) -> Values {
        let d = defaults;
        let get = |key: &str| json.get(key);
        let flag = |key: &str, default: bool| get(key).and_then(serde_json::Value::as_bool).unwrap_or(default);
        let count = |key: &str, default: usize| get(key).and_then(serde_json::Value::as_u64).map_or(default, |v| v.min(UNLIMITED as u64) as usize);
        let float = |key: &str, default: f64, ok: fn(f64) -> bool| get(key).and_then(serde_json::Value::as_f64).filter(|v| ok(*v)).unwrap_or(default);
        let text = |key: &str| get(key).and_then(serde_json::Value::as_str);
        Values {
            optimizer: text("optimizer").map_or(d.optimizer, |t| OptimizerType::from_value(t).0),
            tolerance_goal: flag("tolerance_goal", d.tolerance_goal),
            auto_tolerance: flag("auto_tolerance", d.auto_tolerance),
            tolerance: float("tolerance", d.tolerance, |v| v.is_finite() && v > 0.0),
            tolerance_metric: text("tolerance_metric").and_then(|t| ToleranceMetric::from_value(t).ok()).unwrap_or(d.tolerance_metric),
            knot_placement: text("knot_placement").and_then(|t| placement_from_value(t).ok()).unwrap_or(d.knot_placement),
            parameterization: text("parameterization").and_then(|t| Parameterization::from_value(t).ok()).unwrap_or(d.parameterization),
            grow: flag("grow", d.grow),
            sections: count("sections", d.sections),
            degree: count("degree", d.degree),
            continuity: count("continuity", d.continuity).min(3),
            geometric: flag("geometric", d.geometric),
            corners: flag("corners", d.corners),
            control_points: count("control_points", d.control_points),
            per_junction: text("per_junction").map_or_else(|| d.per_junction.clone(), str::to_string),
            knot_gap: float("knot_gap", d.knot_gap, |v| v.is_finite() && v >= 0.0),
            smoothing: float("smoothing", d.smoothing, |v| v.is_finite() && v >= 0.0),
            optimize_junctions: flag("optimize_junctions", d.optimize_junctions),
            degree_sweep: (false, d.degree_sweep.1),
            count_sweep: (false, d.count_sweep.1),
            sections_sweep: (false, d.sections_sweep.1),
            smoothing_sweep: (false, d.smoothing_sweep.1, d.smoothing_sweep.2),
        }
    }
}

/// Which controls show and which are enabled (19.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct View {
    pub varpro: bool,
    pub tolerance_mode: bool,
    pub has_junctions: bool,
    pub count_used: bool,
    pub editable: bool,
    pub tolerance_enabled: bool,
    pub auto_enabled: bool,
    pub count_enabled: bool,
    pub junction_controls_enabled: bool,
    pub smoothing_enabled: bool,
    /// The Knot spacing label: VarPro with free knots.
    pub free_knots: bool,
    /// The Knot spacing field: also editable.
    pub knot_gap_enabled: bool,
}

impl Settings {
    pub fn varpro(&self) -> bool {
        self.optimizer == OptimizerType::VarPro
    }

    pub fn tolerance_mode(&self) -> bool {
        self.varpro() && self.tolerance_goal
    }

    /// Whether the control-point count is used: not in a growing tolerance fit.
    pub fn count_used(&self) -> bool {
        !self.tolerance_mode() || !self.grow
    }

    pub fn has_junctions(&self) -> bool {
        self.sections.value.value > 1
    }

    pub fn view(&self, fitting: bool) -> View {
        let editable = !fitting;
        let tolerance_mode = self.tolerance_mode();
        let has_junctions = self.has_junctions();
        let free_knots = self.varpro() && self.knot_placement == KnotPlacement::Free;
        View {
            varpro: self.varpro(),
            tolerance_mode,
            has_junctions,
            count_used: self.count_used(),
            editable,
            tolerance_enabled: editable && tolerance_mode && !self.auto_tolerance,
            auto_enabled: editable && tolerance_mode,
            count_enabled: editable && self.count_used(),
            junction_controls_enabled: editable && has_junctions,
            smoothing_enabled: true,
            free_knots,
            knot_gap_enabled: editable && free_knots,
        }
    }

    /// The continuity labels: `G0`..`G3` for geometric VarPro, else `C0`..`C3`.
    pub fn continuity_letter(&self) -> char {
        if self.varpro() && self.geometric {
            'G'
        } else {
            'C'
        }
    }

    /// Runs after the optimizer, goal, Grow points, Auto or section count
    /// change, after undo and redo, and after applying a run: a sweep of the
    /// count stops in tolerance mode, then the curve rule runs.
    pub fn optimizer_changed(&mut self) {
        let tolerance_mode = self.tolerance_mode();
        if tolerance_mode && self.control_points.active {
            self.control_points.toggle(false);
            self.keep_curve_valid(Change::Count);
        }
        self.control_points.set_allowed(!tolerance_mode);
        self.keep_curve_valid(Change::Other);
    }

    /// Per-junction orders from the Per junction text for `sections`
    /// sections, and whether they come from it.
    pub fn junction_orders(&self, sections: usize) -> (Vec<usize>, bool) {
        let entries = per_junction_entries(&self.per_junction);
        let valid = |e: &String| {
            let b = e.as_bytes();
            b.len() == 2 && b"CGcg".contains(&b[0]) && b"0123".contains(&b[1])
        };
        if self.varpro() && !entries.is_empty() && entries.len() + 1 == sections && entries.iter().all(valid) {
            (entries.iter().map(|e| (e.as_bytes()[1] - b'0') as usize).collect(), true)
        } else {
            (vec![self.continuity; sections.saturating_sub(1)], false)
        }
    }

    /// Keeps degree, control points, sections and continuity fittable (19.7).
    pub fn keep_curve_valid(&mut self, changed: Change) {
        let open_sweep = self.control_points.active && (self.degree.active || self.sections.active);
        let low = |w: &IntSweep| -> Vec<usize> { pick_end(w, open_sweep, |a, b| a < b) };
        let high = |w: &IntSweep| -> Vec<usize> { pick_end(w, open_sweep, |a, b| a > b) };
        let orders = |s: &Settings| s.junction_orders(low(&s.sections).iter().map(|&i| s.sections.ends()[i]).max().unwrap()).0;
        let need = |s: &Settings| {
            let degree = low(&s.degree).iter().map(|&i| s.degree.ends()[i]).max().unwrap();
            minimum_control_points(&orders(s), degree)
        };
        let from_text = self.junction_orders(low(&self.sections).iter().map(|&i| self.sections.ends()[i]).max().unwrap()).1;

        // 1. Degree against the continuity.
        let lowest = *self.degree.ends().iter().min().unwrap();
        let top = orders(self).into_iter().max().unwrap_or(0);
        if top > lowest {
            if matches!(changed, Change::Degree | Change::Count) && !from_text {
                self.continuity = self.continuity.min(lowest);
            } else {
                for end in self.degree.ends_mut() {
                    if end.value < top {
                        end.set(top);
                    }
                }
            }
        }
        // 2. Control points against degree and continuity.
        if self.count_used() {
            let have = high(&self.control_points).iter().map(|&i| self.control_points.ends()[i]).min().unwrap();
            if need(self) > have {
                if changed == Change::Count {
                    let lowered = low(&self.degree);
                    let mut ends = self.degree.ends_mut();
                    for i in lowered {
                        if ends[i].value > have.saturating_sub(1) {
                            ends[i].set(have.saturating_sub(1));
                        }
                    }
                    if need(self) > have && self.sections.active {
                        let floor = self.sections.value.value.max(2);
                        if self.sections.end.value > floor {
                            self.sections.end.set(floor);
                        }
                    }
                    while need(self) > have && !from_text && self.continuity > 0 {
                        self.continuity -= 1;
                    }
                } else {
                    let need = need(self);
                    let raised = high(&self.control_points);
                    let mut ends = self.control_points.ends_mut();
                    for i in raised {
                        if ends[i].value < need {
                            ends[i].set(need);
                        }
                    }
                }
            }
        }
        // 3. Hard minimums.
        let (fixed_orders, fixed) = self.junction_orders(self.sections.value.value);
        let degree_min = if fixed { fixed_orders.iter().copied().max().unwrap_or(0) } else { 1 };
        self.degree.set_min(degree_min);
        let count_min = if fixed { minimum_control_points(&fixed_orders, degree_min) } else { 2 };
        self.control_points.set_min(count_min);
        // 4. Sweep caps.
        let mut degree_cap = self.degree.value.max;
        let mut sections_cap = self.sections.value.max;
        if self.count_used() && !self.control_points.active {
            let have = self.control_points.value.value;
            degree_cap = have.saturating_sub(1).max(degree_min);
            let order = self.continuity;
            let fitting = (1..=sections_cap.min(3)).rev().find(|&k| minimum_control_points(&vec![order; k - 1], 1) <= have).unwrap_or(1);
            if fitting < 3 {
                sections_cap = fitting;
            }
            sections_cap = sections_cap.max(self.sections.value.value);
        }
        self.degree.end.set_max(degree_cap);
        self.sections.end.set_max(sections_cap);
    }

    /// Settings after a user change from `before`: the rules for what changed.
    pub fn settle(&mut self, before: &Settings) {
        let sweep_changed =
            |a: &IntSweep, b: &IntSweep| a.value.value != b.value.value || a.active != b.active || a.end.value != b.end.value;
        if sweep_changed(&self.degree, &before.degree) {
            self.keep_curve_valid(Change::Degree);
        }
        if sweep_changed(&self.control_points, &before.control_points) {
            self.keep_curve_valid(Change::Count);
        }
        let sections_changed = sweep_changed(&self.sections, &before.sections);
        if sections_changed {
            self.keep_curve_valid(Change::Sections);
        }
        if self.continuity != before.continuity {
            self.keep_curve_valid(Change::Continuity);
        }
        if self.optimizer != before.optimizer
            || self.tolerance_goal != before.tolerance_goal
            || self.grow != before.grow
            || self.auto_tolerance != before.auto_tolerance
            || self.sections.value.value != before.sections.value.value
        {
            self.optimizer_changed();
        }
    }

    /// The spline settings of the sidebar.
    pub fn spline_config(&self) -> Result<SplineConfig> {
        let junctions = if self.varpro() { per_junction_entries(&self.per_junction) } else { vec![] };
        SplineConfig {
            degree: self.degree.value.value,
            num_control_points: self.control_points.value.value,
            num_sections: self.sections.value.value,
            continuity: ContinuityMode::from_order(self.continuity),
            geometric_continuity: self.varpro() && self.geometric,
            junction_continuity: junctions,
            split_at_corners: self.corners,
        }
        .validated()
    }

    /// The fit settings of the sidebar (the tolerance as set; the caller
    /// updates an automatic tolerance first).
    pub fn optimization_config(&self) -> Result<OptimizationConfig> {
        OptimizationConfig {
            optimizer: self.optimizer,
            optimize_le_position: self.optimize_junctions && self.has_junctions(),
            smoothing_weight: self.smoothing.value,
            parameterization: self.parameterization,
            tolerance: self.tolerance_mode().then_some(self.tolerance),
            tolerance_metric: self.tolerance_metric,
            knot_placement: self.knot_placement,
            grow_control_points: self.grow,
            min_knot_gap: self.knot_gap,
        }
        .validated()
    }

    /// Sets the sidebar to a run's settings (19.10); the next fit splits
    /// the airfoil for the section count.
    pub fn apply_run(&mut self, run: &OptimizationRun) {
        let (c, o) = (&run.spline_config, &run.optimization_config);
        self.optimizer = o.optimizer;
        self.tolerance_goal = o.tolerance.is_some();
        if let Some(t) = o.tolerance {
            self.auto_tolerance = false;
            self.tolerance = t.max(1e-9);
            self.tolerance_metric = o.tolerance_metric;
        }
        self.knot_placement = o.knot_placement;
        self.parameterization = o.parameterization;
        self.grow = o.grow_control_points;
        self.sections.value.set(c.num_sections);
        self.continuity = c.continuity.order();
        self.degree.value.set(c.degree);
        self.geometric = c.geometric_continuity;
        self.corners = c.split_at_corners;
        self.control_points.value.set(c.num_control_points);
        self.per_junction = c.junction_continuity.join(", ");
        self.knot_gap = o.min_knot_gap;
        self.smoothing.value = o.smoothing_weight.max(0.0);
        self.optimize_junctions = o.optimize_le_position;
        self.optimizer_changed();
    }

    pub fn values(&self) -> Values {
        Values {
            optimizer: self.optimizer,
            tolerance_goal: self.tolerance_goal,
            auto_tolerance: self.auto_tolerance,
            tolerance: self.tolerance,
            tolerance_metric: self.tolerance_metric,
            knot_placement: self.knot_placement,
            parameterization: self.parameterization,
            grow: self.grow,
            sections: self.sections.value.value,
            degree: self.degree.value.value,
            continuity: self.continuity,
            geometric: self.geometric,
            corners: self.corners,
            control_points: self.control_points.value.value,
            per_junction: self.per_junction.clone(),
            knot_gap: self.knot_gap,
            smoothing: self.smoothing.value,
            optimize_junctions: self.optimize_junctions,
            degree_sweep: (self.degree.active, self.degree.end.value),
            count_sweep: (self.control_points.active, self.control_points.end.value),
            sections_sweep: (self.sections.active, self.sections.end.value),
            smoothing_sweep: (self.smoothing.active, self.smoothing.end, self.smoothing.step),
        }
    }

    /// Sets every value in order, then runs the rules (limits may clamp).
    pub fn restore(&mut self, v: &Values) {
        self.optimizer = v.optimizer;
        self.tolerance_goal = v.tolerance_goal;
        self.auto_tolerance = v.auto_tolerance;
        self.tolerance = v.tolerance;
        self.tolerance_metric = v.tolerance_metric;
        self.knot_placement = v.knot_placement;
        self.parameterization = v.parameterization;
        self.grow = v.grow;
        self.sections.value.set(v.sections);
        self.degree.value.set(v.degree);
        self.continuity = v.continuity;
        self.geometric = v.geometric;
        self.corners = v.corners;
        self.control_points.value.set(v.control_points);
        self.per_junction = v.per_junction.clone();
        self.knot_gap = v.knot_gap;
        self.smoothing.value = v.smoothing;
        self.optimize_junctions = v.optimize_junctions;
        let int = |w: &mut IntSweep, (active, end): (bool, usize)| {
            if active && !w.allowed {
                w.allowed = true;
            }
            w.toggle(active);
            if active {
                w.end.set(end);
            }
        };
        int(&mut self.degree, v.degree_sweep);
        int(&mut self.control_points, v.count_sweep);
        int(&mut self.sections, v.sections_sweep);
        let (active, end, step) = v.smoothing_sweep;
        self.smoothing.toggle(active);
        if active {
            self.smoothing.end = end;
            self.smoothing.step = step;
        }
        self.optimizer_changed();
    }

    /// Turns every sweep off.
    pub fn stop_sweeps(&mut self) {
        self.sections.toggle(false);
        self.degree.toggle(false);
        self.control_points.toggle(false);
        self.smoothing.toggle(false);
    }

    /// Whether any setting sweeps.
    pub fn sweeping(&self) -> bool {
        self.degree.active || self.control_points.active || self.sections.active || self.smoothing.active
    }
}

/// Indices into `ends()`: all of them, or with `open` only the one that
/// `better` picks (the first on ties).
fn pick_end(w: &IntSweep, open: bool, better: impl Fn(usize, usize) -> bool) -> Vec<usize> {
    let ends = w.ends();
    if !open {
        return (0..ends.len()).collect();
    }
    vec![(0..ends.len()).fold(0, |b, i| if better(ends[i], ends[b]) { i } else { b })]
}

/// The entries of a Per junction text: `,` and `;` separate like spaces.
pub fn per_junction_entries(text: &str) -> Vec<String> {
    text.replace([',', ';'], " ").split_whitespace().map(str::to_string).collect()
}

/// One field the Advanced badge tracks.
#[derive(Debug, Clone, PartialEq)]
struct Tracked {
    name: &'static str,
    value: String,
    text: String,
}

/// Values of the Advanced section at a baseline, to count changes (19.9).
#[derive(Debug, Clone, PartialEq)]
pub struct AdvancedBaseline(Vec<Tracked>);

fn advanced_fields(s: &Settings) -> Vec<(Tracked, bool)> {
    let on_off = |b: bool| if b { "on" } else { "off" }.to_string();
    let varpro = s.varpro();
    vec![
        (Tracked { name: "Per junction", value: s.per_junction.trim().into(), text: s.per_junction.clone() }, varpro),
        (Tracked { name: "Smoothing", value: s.smoothing.value.to_string(), text: number(s.smoothing.value, ".6g") }, varpro),
        (Tracked { name: "Junctions", value: on_off(s.optimize_junctions), text: on_off(s.optimize_junctions) }, true),
    ]
}

impl AdvancedBaseline {
    pub fn of(s: &Settings) -> Self {
        AdvancedBaseline(advanced_fields(s).into_iter().map(|(t, _)| t).collect())
    }

    /// The badge text (`2 changed`, empty when none) and its tooltip.
    pub fn badge(&self, s: &Settings) -> (String, String) {
        let changes: Vec<String> = advanced_fields(s)
            .into_iter()
            .zip(&self.0)
            .filter(|((now, visible), base)| *visible && now.value != base.value)
            .map(|((now, _), base)| {
                let shown = |t: &str| if t.is_empty() { "(blank)".to_string() } else { t.to_string() };
                format!("{}: {} → {}", now.name, shown(&base.text), shown(&now.text))
            })
            .collect();
        if changes.is_empty() {
            return (String::new(), String::new());
        }
        (format!("{} changed", changes.len()), format!("Changed this session:\n{}", changes.join("\n")))
    }
}

/// A swept setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Degree,
    ControlPoints,
    Sections,
    Smoothing,
}

impl Axis {
    pub fn name(self) -> &'static str {
        match self {
            Axis::Degree => "Degree",
            Axis::ControlPoints => "Control points",
            Axis::Sections => "Sections",
            Axis::Smoothing => "Smoothing weight",
        }
    }
}

/// A value of a swept setting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AxisValue {
    Int(usize),
    Float(f64),
}

impl AxisValue {
    pub fn as_f64(self) -> f64 {
        match self {
            AxisValue::Int(i) => i as f64,
            AxisValue::Float(f) => f,
        }
    }

    pub fn label(self) -> String {
        match self {
            AxisValue::Int(i) => i.to_string(),
            AxisValue::Float(f) => number(f, ".5g"),
        }
    }
}

/// The swept axes with their values, in sidebar order.
pub fn sweep_axes(s: &Settings) -> Vec<(Axis, Vec<AxisValue>)> {
    let mut axes = Vec::new();
    let ints = |w: &IntSweep| w.values().into_iter().map(AxisValue::Int).collect::<Vec<_>>();
    if s.degree.active {
        axes.push((Axis::Degree, ints(&s.degree)));
    }
    if s.control_points.active {
        axes.push((Axis::ControlPoints, ints(&s.control_points)));
    }
    if s.sections.active {
        axes.push((Axis::Sections, ints(&s.sections)));
    }
    if s.smoothing.active {
        axes.push((Axis::Smoothing, s.smoothing.values().into_iter().map(AxisValue::Float).collect()));
    }
    axes
}

/// The settings of one sweep combination.
pub fn combination_configs(
    base: &SplineConfig,
    opt: &OptimizationConfig,
    values: &[(Axis, AxisValue)],
) -> (SplineConfig, OptimizationConfig) {
    let mut c = base.clone();
    let mut o = opt.clone();
    for &(axis, value) in values {
        match (axis, value) {
            (Axis::Degree, AxisValue::Int(v)) => c.degree = v,
            (Axis::ControlPoints, AxisValue::Int(v)) => c.num_control_points = v,
            (Axis::Sections, AxisValue::Int(v)) => {
                c.num_sections = v;
                if c.junction_continuity.len() + 1 != v {
                    c.junction_continuity.clear();
                }
            }
            (Axis::Smoothing, AxisValue::Float(v)) => o.smoothing_weight = v,
            _ => unreachable!("axis and value kinds match"),
        }
    }
    (c, o)
}

/// Every combination of the axes in order.
pub fn combinations(axes: &[(Axis, Vec<AxisValue>)]) -> Vec<Vec<(Axis, AxisValue)>> {
    axes.iter().fold(vec![vec![]], |acc, (axis, values)| {
        acc.iter()
            .flat_map(|prefix| {
                values.iter().map(move |v| {
                    let mut c = prefix.clone();
                    c.push((*axis, *v));
                    c
                })
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let s = Settings::default();
        assert_eq!((s.degree.value.value, s.control_points.value.value, s.sections.value.value, s.continuity), (5, 25, 2, 3));
        assert!(s.geometric && s.auto_tolerance && !s.grow);
        assert_eq!(s.degree.end.max, 24);
        assert_eq!(s.spline_config().unwrap().junction_specs(), vec!["G3"]);
    }

    #[test]
    fn lowering_the_degree_lowers_the_continuity() {
        let mut s = Settings::default();
        let before = s.clone();
        s.degree.value.set(2);
        s.settle(&before);
        assert_eq!(s.continuity, 2);
        assert_eq!(s.degree.value.value, 2);
    }

    #[test]
    fn raising_continuity_raises_degree_and_count() {
        let mut s = Settings::default();
        let before = s.clone();
        s.degree.value.set(1);
        s.control_points.value.set(2);
        s.settle(&before);
        let before = s.clone();
        s.continuity = 3;
        s.settle(&before);
        assert_eq!(s.degree.value.value, 3);
        assert!(s.control_points.value.value >= 5);
    }

    #[test]
    fn tolerance_mode_stops_count_sweeps() {
        let mut s = Settings::default();
        s.control_points.toggle(true);
        assert_eq!(s.control_points.end.value, 50);
        let before = s.clone();
        s.tolerance_goal = true;
        s.settle(&before);
        assert!(!s.control_points.active && !s.control_points.allowed);
    }

    #[test]
    fn values_are_kept_without_their_sweeps() {
        let defaults = Settings::default().values();
        let mut s = Settings { tolerance_goal: true, tolerance: 2e-6, ..Default::default() };
        s.smoothing.value = 1e-7;
        s.control_points.value.set(6);
        s.per_junction = "G2".into();
        s.smoothing.toggle(true);
        let before = s.clone();
        s.settle(&before);
        let kept = Values::from_json(&s.values().to_json(), &defaults);
        let mut t = Settings::default();
        t.restore(&kept);
        assert_eq!((t.tolerance_goal, t.tolerance, t.smoothing.value), (true, 2e-6, 1e-7));
        assert_eq!((t.control_points.value.value, t.per_junction.as_str()), (6, "G2"));
        assert!(!t.sweeping());
        // Unreadable entries keep their defaults.
        let odd = serde_json::json!({"degree": "five", "smoothing": -1.0, "optimizer": "gn-varpro", "knot_placement": "odd"});
        assert_eq!(Values::from_json(&odd, &defaults), defaults);
    }

    #[test]
    fn smoothing_stays_usable_in_tolerance_mode() {
        let mut s = Settings::default();
        s.smoothing.toggle(true);
        let before = s.clone();
        s.tolerance_goal = true;
        s.settle(&before);
        assert!(s.tolerance_mode());
        assert!(s.smoothing.active);
        assert!(s.view(false).smoothing_enabled);
        s.smoothing.value = 1e-7;
        assert_eq!(s.optimization_config().unwrap().smoothing_weight, 1e-7);
    }

    #[test]
    fn per_junction_minimums() {
        let mut s = Settings::default();
        s.sections.value.set(3);
        s.per_junction = "G3, C1".into();
        s.keep_curve_valid(Change::Other);
        assert_eq!(s.degree.value.min, 3);
        assert_eq!(s.control_points.value.min, 6);
    }

    #[test]
    fn float_sweep_values() {
        let mut f = FloatSweep::new(1e-3, 1e-3);
        f.toggle(true);
        assert_eq!((f.end, f.step), (1e-2, 1e-3));
        let v = f.values();
        assert_eq!(v.len(), 10);
        assert_eq!(v[2], 0.003);
    }

    #[test]
    fn badge() {
        let s = Settings::default();
        let base = AdvancedBaseline::of(&s);
        let mut t = s.clone();
        t.optimize_junctions = true;
        t.per_junction = "G2".into();
        let (text, tip) = base.badge(&t);
        assert_eq!(text, "2 changed");
        assert_eq!(tip, "Changed this session:\nPer junction: (blank) → G2\nJunctions: off → on");
    }

    #[test]
    fn sweep_combinations() {
        let mut s = Settings::default();
        s.degree.toggle(true);
        s.degree.end.set(6);
        s.control_points.toggle(true);
        s.control_points.end.set(26);
        let axes = sweep_axes(&s);
        assert_eq!(combinations(&axes).len(), 4);
    }
}
