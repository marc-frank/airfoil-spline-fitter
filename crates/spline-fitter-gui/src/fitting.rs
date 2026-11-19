//! Fits, parameter sweeps and batch conversion, driven by the window: each
//! fit runs on a worker; the window starts the next one when it ends.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use spline_fitter::batch::{airfoil_files, prepare, write_result, BatchItem};
use spline_fitter::bspline_varpro::KnotPlacement;
use spline_fitter::formatting::{format_count, format_duration, format_error, number};
use spline_fitter::geometry::AirfoilData;
use spline_fitter::models::{FitResult, IterationRecord, OptimizationConfig, SplineConfig};
use spline_fitter::BSpline;

use crate::app::App;
use crate::job::{FitJob, Live, Outcome};
use crate::settings::{combination_configs, combinations, sweep_axes, Axis, AxisValue};

/// A sweep in progress.
pub struct Sweep {
    combos: Vec<Vec<(Axis, AxisValue)>>,
    index: usize,
    x_axis: Axis,
    base: SplineConfig,
    opt: OptimizationConfig,
    results: Vec<(Vec<(Axis, AxisValue)>, FitResult)>,
    left_out: usize,
    failed: usize,
    cancelled: bool,
    started: Instant,
}

/// A batch conversion in progress.
pub struct Batch {
    files: Vec<String>,
    index: usize,
    source: PathBuf,
    target: PathBuf,
    config: SplineConfig,
    opt: OptimizationConfig,
    auto: bool,
    item: Option<BatchItem>,
    done: usize,
    failed: usize,
    skipped: usize,
    started: Instant,
}

/// What runs in the background.
pub enum Running {
    Fit(FitJob),
    Sweep(FitJob, Box<Sweep>),
    Batch(FitJob, Box<Batch>),
}

impl Running {
    fn job(&mut self) -> &mut FitJob {
        match self {
            Running::Fit(j) | Running::Sweep(j, _) | Running::Batch(j, _) => j,
        }
    }

    pub fn live(&self) -> Arc<Live> {
        match self {
            Running::Fit(j) | Running::Sweep(j, _) | Running::Batch(j, _) => j.live.clone(),
        }
    }

    pub fn cancel(&self) {
        self.live().cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn plural_noun(k: usize, noun: &str) -> String {
    format!("{k} {noun}{}", if k == 1 { "" } else { "s" })
}

/// Curves of a live preview.
fn preview_curves(preview: &crate::job::Preview) -> Vec<BSpline> {
    preview
        .control_points
        .iter()
        .zip(&preview.knots)
        .map(|(c, k)| BSpline { degree: preview.degree, knots: k.clone(), control_points: c.clone() })
        .collect()
}

impl App {
    /// The sidebar's settings, with an automatic tolerance updated first.
    fn configs(&mut self) -> spline_fitter::Result<(SplineConfig, OptimizationConfig)> {
        let config = self.settings.spline_config()?;
        if self.settings.tolerance_mode() {
            self.quiet = true;
            self.update_auto_tolerance();
            self.accept_settings_quietly();
        }
        Ok((config, self.settings.optimization_config()?))
    }

    /// The Fit spline button. The split belongs to the fit: the airfoil is
    /// split for the section count here, not when the count changes.
    pub fn on_fit(&mut self) {
        let Some(mut airfoil) = self.airfoil.clone() else {
            return self.set_status("Load an airfoil first.");
        };
        let (config, opt) = match self.configs() {
            Ok(c) => c,
            Err(e) => return self.set_status(e.0),
        };
        if airfoil.num_sections() != config.num_sections {
            airfoil = match airfoil.resplit(config.num_sections) {
                Ok(a) => a,
                Err(e) => return self.set_status(e.0),
            };
            self.airfoil = Some(airfoil.clone());
        }
        if self.settings.sweeping() {
            return self.start_sweep(config, opt);
        }
        // What is about to be fitted shows for a frame (as Python's status
        // while it prepared), then the fit starts.
        let size = match opt.tolerance {
            Some(t) => format!("tolerance {}", number(t, ".2e")),
            None => format!("{} control points", config.num_control_points),
        };
        let specs = config.junction_specs();
        let mode = if specs.is_empty() { format!("G{}", config.continuity.order()) } else { specs.join(", ") };
        self.set_status(format!("Preparing {size}, degree {}, {mode} continuity...", config.degree));
        self.preparing = Some(Box::new((config, opt, airfoil)));
    }

    /// Starts the fit `on_fit` prepared.
    fn start_fit(&mut self, (config, opt, airfoil): (SplineConfig, OptimizationConfig, AirfoilData)) {
        self.main_plot.clear_items();
        self.loss.reset();
        self.set_status(format!("Starting {}...", opt.optimizer.display_name()));
        let live = Live::new(airfoil.display_name(), self.live_mode);
        let job = FitJob::start(config, opt, airfoil, live, "");
        self.running = Some(Running::Fit(job));
    }

    /// The refits of the running tolerance search in the runs plot.
    fn show_live_refits(&mut self, refits: &[[f64; 2]], tolerance: f64) {
        let mut points = refits.to_vec();
        let latest = *points.last().expect("at least one refit");
        points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
        let c = &mut self.comparison;
        c.x_label = "Control points".into();
        c.y_label = "Max deviation".into();
        c.log_x = false;
        c.log_y = points.iter().all(|p| p[1] > 0.0) && tolerance > 0.0;
        c.ticks = None;
        c.tolerance = Some(tolerance);
        c.labels.clear();
        let unique = points.windows(2).all(|w| w[0][0] != w[1][0]);
        c.set(points, unique);
        c.marked = vec![latest];
        self.explorer.forget_points();
    }

    /// Follows the running job: live drawing, and its end.
    pub fn tick_job(&mut self, ctx: &eframe::egui::Context) {
        if let Some(prepared) = self.preparing.take() {
            self.start_fit(*prepared);
            ctx.request_repaint();
            return;
        }
        let mode = self.live_mode;
        let Some(running) = self.running.as_mut() else { return };
        let job = running.job();
        let tolerance = job.options.tolerance;
        let (preview, status) = job.tick(mode);
        let history = job.live.history();
        if let Some(p) = preview {
            self.main_plot.show_curves(preview_curves(&p), tolerance);
            self.loss.set_data(history);
        }
        if let Some(s) = status {
            self.status = s;
        }
        // A tolerance search draws its refits in the runs plot as they
        // finish, the latest marked, with the tolerance line; at its end the
        // plot goes back to the project's runs.
        if let (Some(Running::Fit(job)), Some(tolerance)) = (self.running.as_mut(), tolerance) {
            let refits = job.live.refits();
            if refits.len() != job.shown_refits {
                job.shown_refits = refits.len();
                self.show_live_refits(&refits, tolerance);
            }
        }
        if !self.running.as_mut().unwrap().job().finished() {
            return;
        }
        match self.running.take().unwrap() {
            Running::Fit(mut job) => self.finish_fit(&mut job),
            Running::Sweep(mut job, sweep) => self.finish_sweep_step(&mut job, sweep),
            Running::Batch(mut job, batch) => self.finish_batch_step(&mut job, batch),
        }
    }

    fn effort(&self, job: &FitJob, outcome: &Outcome) -> String {
        let evaluations = plural_noun(outcome.raw.evaluations, "evaluation");
        if job.options.tolerance.is_some() {
            format!("{}, {evaluations}", plural_noun(outcome.raw.iterations, "refit"))
        } else {
            evaluations
        }
    }

    /// Stores the earlier fits of a tolerance search as runs of their own,
    /// each with the loss history of its own refit (from the fit before it);
    /// the search's last fit keeps the whole history. A search adds one
    /// control point a round, so histories up to each fit's time would hold
    /// the search's evaluations many times over.
    fn add_earlier_runs(&mut self, job: &FitJob, outcome: &Outcome, history: &[IterationRecord]) {
        let mut since = f64::NEG_INFINITY;
        for fit in &outcome.earlier {
            let own = history.iter().filter(|r| r.elapsed_seconds > since && r.elapsed_seconds <= fit.elapsed_seconds).copied().collect();
            since = fit.elapsed_seconds;
            self.add_run(&job.config, &job.options, fit.clone(), own, false);
        }
    }

    fn finish_fit(&mut self, job: &mut FitJob) {
        let log = job.live.log();
        self.loss.set_data(log.history().collect());
        match job.join() {
            Err(e) => {
                eprintln!("Fitting error: {e}");
                self.fail(format!("Fitting error: {e}"));
                self.show_current_fit();
                // The runs plot may show the search's refits.
                self.explorer.forget_points();
                self.refresh_comparison();
            }
            Ok(outcome) => {
                let history: Vec<IterationRecord> =
                    log.history().map(|(i, t, e)| IterationRecord { iteration: i, error: e, elapsed_seconds: t }).collect();
                self.add_earlier_runs(job, &outcome, &history);
                let kept = !self.add_run(&job.config, &job.options, outcome.fit.clone(), history, true);
                if kept {
                    // Not added, so not replotted: the runs plot may show the search's refits.
                    self.explorer.forget_points();
                    self.refresh_comparison();
                }
                self.fit = Some(outcome.fit.clone());
                self.show_current_fit();
                let specs = job.config.junction_specs();
                let continuity = if specs.is_empty() {
                    "none".to_string()
                } else if specs.iter().all(|s| *s == specs[0]) {
                    specs[0].clone()
                } else {
                    specs.join("/")
                };
                match log.save(job.options.optimizer.value(), &format!("continuity={continuity}, degree={}", job.config.degree)) {
                    Ok(Some(path)) => println!("Log saved: {}", path.display()),
                    Ok(None) => {}
                    Err(e) => println!("Failed to save optimizer log: {e}"),
                }
                let effort = self.effort(job, &outcome);
                let duration = format_duration(outcome.elapsed);
                let mut status = if outcome.cancelled {
                    format!("Cancelled: {effort} in {duration} | Best error {}", format_error(outcome.raw.error))
                } else {
                    format!(
                        "Completed ({}): {effort} in {duration} | Error {}",
                        job.options.optimizer.display_name(),
                        format_error(outcome.raw.error)
                    )
                };
                if let Some(r) = &outcome.fit.report {
                    status += &format!(" | {}", r.summary_line());
                }
                if !outcome.raw.success && !outcome.cancelled {
                    // A search that misses its tolerance converged; its
                    // message says what stopped it.
                    let message = &outcome.raw.message;
                    if message.starts_with("Tolerance not reached") {
                        status += &format!(" | {message}");
                    } else {
                        status += &format!(" | Solver did not converge: {message}");
                    }
                }
                if kept {
                    status += " | The project keeps its finished run of these settings";
                }
                self.set_status(status);
                if !outcome.cancelled {
                    crate::chime::completion();
                }
            }
        }
    }

    /// Draws the current fit (or nothing) on the airfoil.
    pub fn show_current_fit(&mut self) {
        match &self.fit {
            Some(fit) => {
                let tolerance = fit.report.as_ref().and_then(|r| r.tolerance);
                self.main_plot.show_curves(fit.curves(), tolerance);
            }
            None => self.main_plot.clear_items(),
        }
    }

    // -----------------------------------------------------------------
    // Sweeps

    fn start_sweep(&mut self, base: SplineConfig, opt: OptimizationConfig) {
        let axes = sweep_axes(&self.settings);
        let names: Vec<Axis> = axes.iter().map(|(a, _)| *a).collect();
        let x_axis = axes
            .iter()
            .fold(None::<&(Axis, Vec<AxisValue>)>, |best, a| match best {
                Some(b) if b.1.len() >= a.1.len() => Some(b),
                _ => Some(a),
            })
            .map(|a| a.0)
            .unwrap();
        let mut left_out = 0;
        let combos: Vec<_> = combinations(&axes)
            .into_iter()
            .filter(|combo| {
                let (c, _) = combination_configs(&base, &opt, combo);
                let fits = c.num_control_points >= c.minimum_control_points();
                left_out += usize::from(!fits);
                fits
            })
            .collect();
        self.comparison.x_label = x_axis.name().into();
        self.comparison.y_label = "Error".into();
        self.comparison.log_x = false;
        self.comparison.log_y = true;
        self.comparison.ticks = None;
        self.comparison.tolerance = None;
        self.comparison.labels.clear();
        self.comparison.set(vec![], false);
        self.comparison.marked.clear();
        self.explorer.forget_points();
        self.main_plot.clear_items();
        let names_text: Vec<&str> = names.iter().map(|a| a.name()).collect();
        let left_text = if left_out > 0 {
            format!(" | Left out {left_out} combinations with too few control points for their degree and junctions")
        } else {
            String::new()
        };
        self.set_status(format!("Sweep: fitting {} splines ({}){left_text}...", combos.len(), names_text.join(", ")));
        let sweep =
            Sweep { combos, index: 0, x_axis, base, opt, results: vec![], left_out, failed: 0, cancelled: false, started: Instant::now() };
        self.next_sweep_step(Box::new(sweep));
    }

    fn sweep_label(combo: &[(Axis, AxisValue)]) -> String {
        combo.iter().map(|(a, v)| format!("{} {}", a.name(), v.label())).collect::<Vec<_>>().join(", ")
    }

    fn next_sweep_step(&mut self, sweep: Box<Sweep>) {
        if sweep.cancelled || sweep.index >= sweep.combos.len() {
            return self.finish_sweep(*sweep);
        }
        let combo = &sweep.combos[sweep.index];
        let (config, opt) = combination_configs(&sweep.base, &sweep.opt, combo);
        let airfoil: AirfoilData = match &self.airfoil {
            Some(a) if a.num_sections() == config.num_sections => a.clone(),
            Some(a) => match a.resplit(config.num_sections) {
                Ok(a) => a,
                Err(e) => {
                    let mut sweep = sweep;
                    sweep.failed += 1;
                    sweep.index += 1;
                    self.set_status(format!(
                        "Sweep: {}/{} | {} | Failed: {e}",
                        sweep.index,
                        sweep.combos.len(),
                        Self::sweep_label(&sweep.combos[sweep.index - 1])
                    ));
                    return self.next_sweep_step(sweep);
                }
            },
            None => return self.finish_sweep(*sweep),
        };
        self.main_plot.clear_items();
        self.loss.reset();
        let total = sweep.combos.len();
        let prefix = format!("Sweep {}/{total} | {} | ", sweep.index + 1, Self::sweep_label(combo));
        let live = Live::new(airfoil.display_name(), self.live_mode);
        let job = FitJob::start(config, opt, airfoil, live, &prefix);
        self.running = Some(Running::Sweep(job, sweep));
    }

    fn finish_sweep_step(&mut self, job: &mut FitJob, mut sweep: Box<Sweep>) {
        let log = job.live.log();
        self.loss.set_data(log.history().collect());
        let combo = sweep.combos[sweep.index].clone();
        let total = sweep.combos.len();
        let label = Self::sweep_label(&combo);
        sweep.index += 1;
        let mut status =
            format!("Sweep: {}/{total} ({}%) | {label}", sweep.index, number(100.0 * sweep.index as f64 / total as f64, ".0f"));
        match job.join() {
            Err(e) => {
                eprintln!("Sweep fit failed: {e}");
                sweep.failed += 1;
                status += &format!(" | Failed: {e}");
            }
            Ok(outcome) => {
                let history: Vec<IterationRecord> =
                    log.history().map(|(i, t, e)| IterationRecord { iteration: i, error: e, elapsed_seconds: t }).collect();
                self.add_earlier_runs(job, &outcome, &history);
                self.add_run(&job.config, &job.options, outcome.fit.clone(), history, false);
                if outcome.cancelled {
                    sweep.cancelled = true;
                    status += " | Cancelled";
                } else {
                    status += &format!(", error {}", format_error(outcome.fit.comparison_error()));
                    sweep.results.push((combo, outcome.fit.clone()));
                    self.fit = Some(outcome.fit);
                    self.show_current_fit();
                    self.draw_sweep_comparison(&sweep);
                }
            }
        }
        status += &format!(" | {}", format_duration(sweep.started.elapsed().as_secs_f64()));
        self.set_status(status);
        self.next_sweep_step(sweep);
    }

    /// The sweep's series on the comparison plot, the best marked.
    /// The sweep's series on the comparison plot: every project run of the
    /// sweep's settings (all but the swept ones), so earlier sweeps over
    /// other ranges show beside it; the best of this sweep marked.
    fn draw_sweep_comparison(&mut self, sweep: &Sweep) {
        let swept: Vec<Axis> = sweep.combos.first().map(|c| c.iter().map(|(a, _)| *a).collect()).unwrap_or_default();
        let points = self.sweep_series(&sweep.base, &sweep.opt, &swept, sweep.x_axis);
        let unique = points.windows(2).all(|w| w[0][0] != w[1][0]);
        let best = sweep.results.iter().fold(None::<&(Vec<(Axis, AxisValue)>, FitResult)>, |b, r| match b {
            Some(b) if b.1.comparison_error() <= r.1.comparison_error() => Some(b),
            _ => Some(r),
        });
        let x_of = |combo: &[(Axis, AxisValue)]| combo.iter().find(|(a, _)| *a == sweep.x_axis).map_or(0.0, |(_, v)| v.as_f64());
        self.comparison.log_y = points.iter().all(|p| p[1] > 0.0);
        self.comparison.set(points, unique);
        self.comparison.marked = best.map(|(c, f)| vec![[x_of(c), f.comparison_error()]]).unwrap_or_default();
    }

    /// `(axis value, comparison error)` of the project runs that match the
    /// configuration in everything but the swept settings, sorted; of runs
    /// at the same swept values the latest counts.
    fn sweep_series(&self, base: &SplineConfig, opt: &OptimizationConfig, swept: &[Axis], axis: Axis) -> Vec<[f64; 2]> {
        let fixed = |c: &SplineConfig, o: &OptimizationConfig, a: Axis| match a {
            Axis::ControlPoints => c.num_control_points as f64,
            Axis::Degree => c.degree as f64,
            Axis::Sections => c.num_sections as f64,
            Axis::Smoothing => o.smoothing_weight,
        };
        let value_of = |run: &spline_fitter::project::OptimizationRun, a: Axis| match a {
            Axis::ControlPoints if run.optimization_config.tolerance.is_some() => run.result.num_control_points() as f64,
            _ => fixed(&run.spline_config, &run.optimization_config, a),
        };
        let mut by_values: Vec<(Vec<f64>, [f64; 2])> = Vec::new();
        for run in self.projects.runs() {
            let (c, o) = (&run.spline_config, &run.optimization_config);
            // Along the sections axis the junction count changes: compare the
            // continuity setting instead of the junctions themselves.
            let same_junctions = if swept.contains(&Axis::Sections) {
                c.continuity == base.continuity && c.geometric_continuity == base.geometric_continuity
            } else {
                c.junction_specs() == base.junction_specs()
            };
            let others_equal = [Axis::ControlPoints, Axis::Degree, Axis::Sections, Axis::Smoothing]
                .into_iter()
                .filter(|a| !swept.contains(a))
                .all(|a| fixed(c, o, a) == fixed(base, opt, a));
            let belongs = o.optimizer == opt.optimizer
                && same_junctions
                && o.tolerance.is_some() == opt.tolerance.is_some()
                && o.knot_placement == opt.knot_placement
                && (o.knot_placement != KnotPlacement::Free || o.min_knot_gap == opt.min_knot_gap)
                && o.parameterization == opt.parameterization
                && others_equal;
            if !belongs {
                continue;
            }
            let key: Vec<f64> = swept.iter().map(|a| value_of(run, *a)).collect();
            let point = [value_of(run, axis), run.result.comparison_error()];
            match by_values.iter_mut().find(|(k, _)| *k == key) {
                Some(entry) => entry.1 = point,
                None => by_values.push((key, point)),
            }
        }
        let mut points: Vec<[f64; 2]> = by_values.into_iter().map(|(_, p)| p).collect();
        points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
        points
    }

    fn finish_sweep(&mut self, sweep: Sweep) {
        let duration = format_duration(sweep.started.elapsed().as_secs_f64());
        let left = if sweep.left_out > 0 {
            format!(" | Left out {} combinations with too few control points for their degree and junctions", sweep.left_out)
        } else {
            String::new()
        };
        let best = sweep
            .results
            .iter()
            .fold(None::<&(Vec<(Axis, AxisValue)>, FitResult)>, |b, r| match b {
                Some(b) if b.1.comparison_error() <= r.1.comparison_error() => Some(b),
                _ => Some(r),
            })
            .cloned();
        let mut status = match &best {
            Some((combo, fit)) => {
                if !sweep.cancelled {
                    self.apply_best(combo);
                }
                self.fit = Some(fit.clone());
                self.show_current_fit();
                format!(
                    "Sweep {}: {} in {duration} | Best: {}, error {}",
                    if sweep.cancelled { "cancelled" } else { "complete" },
                    format_count(sweep.results.len(), "fit"),
                    Self::sweep_label(combo),
                    format_error(fit.comparison_error())
                )
            }
            None => {
                self.show_current_fit();
                if sweep.cancelled {
                    "Sweep cancelled: no fit finished".into()
                } else {
                    "Sweep failed: no fit succeeded".into()
                }
            }
        };
        status += &left;
        if sweep.failed > 0 {
            status += &format!(" | {} failed", sweep.failed);
        }
        self.set_status(status);
        if sweep.failed > 0 {
            crate::chime::error();
        } else if !sweep.cancelled && best.is_some() {
            crate::chime::completion();
        }
    }

    /// Sets the swept settings to the best values; sweeps stop. One undo
    /// step returns to the sweep.
    fn apply_best(&mut self, combo: &[(Axis, AxisValue)]) {
        self.as_one_step(|app| app.set_best(combo));
    }

    fn set_best(&mut self, combo: &[(Axis, AxisValue)]) {
        self.settings.stop_sweeps();
        for axis in [Axis::Sections, Axis::Degree, Axis::ControlPoints, Axis::Smoothing] {
            if let Some((_, v)) = combo.iter().find(|(a, _)| *a == axis) {
                match (axis, v) {
                    (Axis::Sections, AxisValue::Int(i)) => self.settings.sections.value.set(*i),
                    (Axis::Degree, AxisValue::Int(i)) => self.settings.degree.value.set(*i),
                    (Axis::ControlPoints, AxisValue::Int(i)) => self.settings.control_points.value.set(*i),
                    (Axis::Smoothing, AxisValue::Float(f)) => self.settings.smoothing.value = *f,
                    _ => {}
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Batch conversion

    pub fn start_batch(&mut self) {
        let (source, target) = (PathBuf::from(&self.batch_source), PathBuf::from(&self.batch_target));
        if self.batch_source.is_empty() || !source.is_dir() {
            return self.set_status("Choose a source folder first.");
        }
        if self.batch_target.is_empty() || !target.is_dir() {
            return self.set_status("Choose a target folder first.");
        }
        let files = match airfoil_files(&source) {
            Ok(f) if !f.is_empty() => f,
            _ => return self.set_status("No .dat files in the source folder."),
        };
        let (config, opt) = match (self.settings.spline_config(), self.settings.optimization_config()) {
            (Ok(c), Ok(o)) => (c, o),
            (Err(e), _) | (_, Err(e)) => return self.set_status(e.0),
        };
        let auto = self.settings.auto_tolerance;
        let batch = Batch {
            files,
            index: 0,
            source,
            target,
            config,
            opt,
            auto,
            item: None,
            done: 0,
            failed: 0,
            skipped: 0,
            started: Instant::now(),
        };
        self.next_batch_file(Box::new(batch));
    }

    fn next_batch_file(&mut self, mut batch: Box<Batch>) {
        loop {
            if batch.index >= batch.files.len() {
                return self.finish_batch(*batch, false);
            }
            let total = batch.files.len();
            let idx = batch.index + 1;
            let file = batch.files[batch.index].clone();
            batch.index += 1;
            let mut opt = batch.opt.clone();
            match prepare(&batch.source.join(&file), &batch.target, &batch.config, &batch.opt, batch.auto) {
                Err(e) => {
                    batch.failed += 1;
                    self.fail(format!("[{idx}/{total}] Failed: {file} - {e}"));
                }
                Ok(item) if item.output.exists() => {
                    batch.skipped += 1;
                    self.set_status(format!("[{idx}/{total}] Skip: {}", item.output.display()));
                }
                Ok(item) => {
                    if opt.tolerance.is_some() && batch.auto {
                        opt.tolerance = Some(spline_fitter::analysis::default_tolerance(&item.airfoil.coordinates, item.airfoil.precision));
                    }
                    self.set_status(format!("[{idx}/{total}] Fitting {file}..."));
                    self.main_plot.set_data(&item.airfoil.coordinates);
                    self.loss.reset();
                    let live = Live::new(item.airfoil.display_name(), self.live_mode);
                    let job = FitJob::start(batch.config.clone(), opt, item.airfoil.clone(), live, &format!("[{idx}/{total}] {file} | "));
                    batch.item = Some(item);
                    self.running = Some(Running::Batch(job, batch));
                    return;
                }
            }
        }
    }

    fn finish_batch_step(&mut self, job: &mut FitJob, mut batch: Box<Batch>) {
        let total = batch.files.len();
        let idx = batch.index;
        let file = batch.files[idx - 1].clone();
        self.loss.set_data(job.live.history());
        let item = batch.item.take().expect("a batch job has its item");
        let result = job.join().and_then(|outcome| {
            if outcome.cancelled {
                return Ok(None);
            }
            if !outcome.raw.success {
                return Err(spline_fitter::Error::new(format!("Optimizer did not converge: {}", outcome.raw.message)));
            }
            self.main_plot.show_curves(outcome.fit.curves(), job.options.tolerance);
            write_result(&item, &batch.config, &outcome.raw)?;
            Ok(Some(outcome))
        });
        match result {
            Ok(None) => return self.finish_batch(*batch, true),
            Ok(Some(outcome)) => {
                batch.done += 1;
                self.set_status(format!(
                    "[{idx}/{total}] Done: {file} | {} evaluations, error {} | {}",
                    outcome.raw.evaluations,
                    format_error(outcome.raw.error),
                    format_duration(outcome.elapsed)
                ));
            }
            Err(e) => {
                batch.failed += 1;
                self.fail(format!("[{idx}/{total}] Failed: {file} - {e}"));
            }
        }
        if job.live.cancelled() {
            return self.finish_batch(*batch, true);
        }
        self.next_batch_file(batch);
    }

    fn finish_batch(&mut self, batch: Batch, cancelled: bool) {
        let total = batch.files.len();
        let duration = format_duration(batch.started.elapsed().as_secs_f64());
        let mut status = if cancelled {
            format!("Batch cancelled after {duration}. Completed {}/{total}, failed {}.", batch.done, batch.failed)
        } else {
            format!("Batch complete: {} files in {duration} ({} failed)", batch.done, batch.failed)
        };
        if batch.skipped > 0 {
            status += &format!(" | Skipped {} already in the target folder with these settings", batch.skipped);
        }
        self.set_status(status);
        let coords = self.airfoil.as_ref().map(|a| a.coordinates.clone()).unwrap_or_default();
        self.main_plot.set_data(&coords);
        if self.airfoil.is_some() {
            self.show_current_fit();
        }
    }
}
