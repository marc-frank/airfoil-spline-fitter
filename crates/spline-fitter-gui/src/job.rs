//! Fits on a worker thread with a live preview the window polls.
//!
//! The worker never touches the window. It records the error history,
//! publishes the latest curves and status text, and, in the Detailed live
//! mode, waits until the window has drawn each step.
//!
//! In the browser the worker is a web worker (crates/spline-fitter-worker):
//! the fit goes to it as a message and its updates and result come back as
//! messages, which the window feeds through the same monitor each frame.
//! A web worker cannot be asked to stop, so Cancel ends it and keeps the
//! last step it reported. The Detailed mode cannot hold the worker at each
//! step there and draws as often as it can.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::thread::JoinHandle;
use std::time::Duration;
use web_time::Instant;

use spline_fitter::analysis::fit_report_for;
use spline_fitter::formatting::{format_duration, format_error};
use spline_fitter::geometry::AirfoilData;
use spline_fitter::models::{FitResult, OptimizationConfig, SplineConfig};
#[cfg(not(target_arch = "wasm32"))]
use spline_fitter::optimize::run_optimization;
use spline_fitter::optimize::{Monitor, RawResult, Update, UpdateKind};
use spline_fitter::run_log::ErrorLog;
use spline_fitter::{Result, Vec2};

/// How often the plots follow a running fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LiveMode {
    /// Only at the end.
    Minimal,
    /// Once a second.
    #[default]
    Moderate,
    /// Every iteration (slower).
    Detailed,
}

impl LiveMode {
    pub const ALL: [LiveMode; 3] = [Self::Minimal, Self::Moderate, Self::Detailed];
    pub const LABELS: [&'static str; 3] = ["Minimal", "Moderate", "Detailed"];
}

const MODERATE_INTERVAL: Duration = Duration::from_secs(1);
const STATUS_INTERVAL: Duration = Duration::from_millis(500);

/// The curves of the latest step.
#[derive(Debug, Clone, PartialEq)]
pub struct Preview {
    pub control_points: Vec<Vec<Vec2>>,
    pub knots: Vec<Vec<f64>>,
    pub degree: usize,
}

/// What the worker shares with the window.
#[derive(Default)]
struct Published {
    preview: Option<Preview>,
    status: String,
    fresh: bool,
    drawn: bool,
}

/// State shared between the window and a running fit.
pub struct Live {
    pub cancel: AtomicBool,
    mode: AtomicUsize,
    log: Mutex<ErrorLog>,
    published: Mutex<Published>,
    drawn: Condvar,
    /// The finished fits of a tolerance search: control points per
    /// section and largest deviation.
    refits: Mutex<Vec<[f64; 2]>>,
}

impl Live {
    pub fn new(run_name: &str, mode: LiveMode) -> Arc<Self> {
        Arc::new(Live {
            cancel: AtomicBool::new(false),
            mode: AtomicUsize::new(mode as usize),
            log: Mutex::new(ErrorLog::new(run_name)),
            published: Mutex::new(Published::default()),
            drawn: Condvar::new(),
            refits: Mutex::new(Vec::new()),
        })
    }

    pub fn set_mode(&self, mode: LiveMode) {
        self.mode.store(mode as usize, Ordering::Relaxed);
    }

    fn mode(&self) -> LiveMode {
        LiveMode::ALL[self.mode.load(Ordering::Relaxed)]
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// The finished refits so far: `[control points, largest deviation]`.
    pub fn refits(&self) -> Vec<[f64; 2]> {
        self.refits.lock().unwrap().clone()
    }

    /// The error history: `(iteration, seconds, error)`.
    pub fn history(&self) -> Vec<(usize, f64, f64)> {
        self.log.lock().unwrap().history().collect()
    }

    pub fn log(&self) -> ErrorLog {
        self.log.lock().unwrap().clone()
    }

    /// The latest step if it is new, marking it taken.
    fn take(&self) -> Option<(Option<Preview>, String)> {
        let mut p = self.published.lock().unwrap();
        if !p.fresh {
            return None;
        }
        p.fresh = false;
        Some((p.preview.clone(), p.status.clone()))
    }

    /// Releases a worker waiting for the drawing of its step.
    fn mark_drawn(&self) {
        self.published.lock().unwrap().drawn = true;
        self.drawn.notify_all();
    }
}

/// The monitor a worker hands to the optimizer.
struct Progress {
    live: Arc<Live>,
    start: Instant,
    prefix: String,
    tolerance_mode: bool,
    iteration: usize,
    best: f64,
    last_refit: Option<f64>,
}

impl Monitor for Progress {
    fn update(&mut self, u: &Update) -> bool {
        if self.live.cancelled() {
            return true;
        }
        if u.kind == UpdateKind::Refit {
            self.last_refit = Some(u.error);
            // Per section, as the runs plot shows the runs.
            let count = u.control_points.first().map_or(0, Vec::len);
            self.live.refits.lock().unwrap().push([count as f64, u.error]);
            return false;
        }
        self.iteration += 1;
        let elapsed = self.start.elapsed().as_secs_f64();
        self.live.log.lock().unwrap().log(self.iteration, u.evaluations, u.error, elapsed);
        self.best = self.best.min(u.error);
        let mut status = format!("{}Fitting... {} | Evaluations {} | ", self.prefix, format_duration(elapsed), u.evaluations);
        if self.tolerance_mode {
            // The search adds control points as it goes: show where it is.
            let counts: Vec<usize> = u.control_points.iter().map(Vec::len).collect();
            status += &format!("Control points {} | RMS error {}", crate::sidebar::count_text(&counts), format_error(u.error));
            if let Some(d) = self.last_refit {
                status += &format!(" | Max deviation of the last fit {}", format_error(d));
            }
        } else {
            status += &format!("Best RMS error {}", format_error(self.best));
        }
        let preview = Preview { control_points: u.control_points.to_vec(), knots: u.knots.to_vec(), degree: u.degree };
        let mut p = self.live.published.lock().unwrap();
        *p = Published { preview: Some(preview), status, fresh: true, drawn: false };
        if self.live.mode() == LiveMode::Detailed && !cfg!(target_arch = "wasm32") {
            while !p.drawn {
                let (guard, _) = self.live.drawn.wait_timeout(p, Duration::from_millis(50)).unwrap();
                p = guard;
                if self.live.cancelled() {
                    return true;
                }
            }
        }
        false
    }
}

/// What a finished fit hands back.
pub struct Outcome {
    pub raw: RawResult,
    pub fit: FitResult,
    /// The earlier fits of a tolerance search, each with its report.
    pub earlier: Vec<FitResult>,
    pub cancelled: bool,
    pub elapsed: f64,
}

/// A fit running on a worker thread.
pub struct FitJob {
    pub live: Arc<Live>,
    pub config: SplineConfig,
    pub options: OptimizationConfig,
    #[cfg(not(target_arch = "wasm32"))]
    handle: Option<JoinHandle<Result<Outcome>>>,
    #[cfg(target_arch = "wasm32")]
    handle: Option<browser::WorkerFit>,
    last_draw: Option<Instant>,
    last_status: Option<Instant>,
    /// Refits of a tolerance search already in the runs plot.
    pub shown_refits: usize,
}

impl FitJob {
    /// Starts fitting `airfoil`; status texts start with `prefix`.
    pub fn start(config: SplineConfig, options: OptimizationConfig, airfoil: AirfoilData, live: Arc<Live>, prefix: &str) -> Self {
        let monitor = Progress {
            live: live.clone(),
            start: Instant::now(),
            prefix: prefix.to_string(),
            tolerance_mode: options.tolerance.is_some(),
            iteration: 0,
            best: f64::INFINITY,
            last_refit: None,
        };
        let (c, o, a) = (config.clone(), options.clone(), airfoil);
        #[cfg(not(target_arch = "wasm32"))]
        let handle = std::thread::Builder::new()
            .name("spline-fitter-work".into())
            .spawn(move || work(c, o, a, monitor))
            .expect("the worker thread starts");
        #[cfg(target_arch = "wasm32")]
        let handle = browser::WorkerFit::start(c, o, a, monitor);
        FitJob { live, config, options, handle: Some(handle), last_draw: None, last_status: None, shown_refits: 0 }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn finished(&self) -> bool {
        self.handle.as_ref().is_none_or(|h| h.is_finished())
    }

    /// The browser: takes what the worker sent; true once it has finished
    /// (or was ended by Cancel).
    #[cfg(target_arch = "wasm32")]
    pub fn finished(&self) -> bool {
        self.handle.as_ref().is_none_or(|h| h.poll())
    }

    #[cfg(target_arch = "wasm32")]
    pub fn join(&mut self) -> Result<Outcome> {
        let handle = self.handle.take().expect("joined once");
        handle.poll();
        handle.take().unwrap_or_else(|| Err(spline_fitter::Error::new("the fit stopped unexpectedly")))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn join(&mut self) -> Result<Outcome> {
        let handle = self.handle.take().expect("joined once");
        self.live.mark_drawn();
        match handle.join() {
            Ok(result) => result,
            Err(panic) => {
                let message = panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()));
                Err(spline_fitter::Error::new(message.unwrap_or_else(|| "the fit stopped unexpectedly".into())))
            }
        }
    }

    /// The window's tick: a preview to draw and a status to show, if due.
    pub fn tick(&mut self, mode: LiveMode) -> (Option<Preview>, Option<String>) {
        let Some((preview, status)) = self.live.take() else { return (None, None) };
        let now = Instant::now();
        let due = match mode {
            LiveMode::Detailed => true,
            LiveMode::Moderate => self.last_draw.is_none_or(|t| now - t >= MODERATE_INTERVAL),
            LiveMode::Minimal => false,
        };
        self.live.mark_drawn();
        if due {
            self.last_draw = Some(now);
            self.last_status = Some(now);
            return (preview, Some(status));
        }
        if self.last_status.is_none_or(|t| now - t >= STATUS_INTERVAL) {
            self.last_status = Some(now);
            return (None, Some(status));
        }
        (None, None)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn work(config: SplineConfig, options: OptimizationConfig, airfoil: AirfoilData, mut monitor: Progress) -> Result<Outcome> {
    let start = Instant::now();
    let raw = run_optimization(&config, &options, &airfoil, &mut monitor)?;
    let cancelled = monitor.live.cancelled();
    Ok(outcome(&config, &options, &airfoil, raw, start.elapsed().as_secs_f64(), cancelled))
}

/// A finished fit with its report (and its earlier fits' reports).
fn outcome(config: &SplineConfig, options: &OptimizationConfig, airfoil: &AirfoilData, raw: RawResult, elapsed: f64, cancelled: bool) -> Outcome {
    let fit_of = |r: &RawResult, elapsed: f64, cancelled: bool| {
        let mut fit = FitResult {
            section_control_points: r.section_control_points.clone(),
            degree: config.degree,
            final_error: r.error,
            num_iterations: r.iterations,
            num_evaluations: r.evaluations,
            elapsed_seconds: elapsed,
            converged: r.success && !cancelled,
            cancelled,
            section_knots: r.section_knots.clone(),
            report: None,
            rms_error: Some(r.rms_error),
        };
        fit.report = Some(fit_report_for(&airfoil.sections, &fit, Some(airfoil.precision), options.tolerance, options.tolerance_metric));
        fit
    };
    let fit = fit_of(&raw, elapsed, cancelled);
    let earlier = raw.earlier.iter().map(|r| fit_of(r, r.seconds.unwrap_or(0.0), false)).collect();
    Outcome { raw, fit, earlier, cancelled, elapsed }
}

/// The browser's fit: a web worker and what it sent so far.
#[cfg(target_arch = "wasm32")]
mod browser {
    use std::cell::RefCell;
    use std::rc::Rc;

    use spline_fitter::project::wire::{self, Message, OwnedUpdate};
    use wasm_bindgen::prelude::*;

    use super::*;

    pub struct WorkerFit {
        worker: Option<web_sys::Worker>,
        inbox: Rc<RefCell<Vec<String>>>,
        _listeners: [Closure<dyn FnMut(web_sys::MessageEvent)>; 2],
        state: RefCell<State>,
    }

    struct State {
        monitor: Progress,
        config: SplineConfig,
        options: OptimizationConfig,
        airfoil: AirfoilData,
        last: Option<OwnedUpdate>,
        done: Option<Result<Outcome>>,
    }

    impl WorkerFit {
        /// Starts a worker (crates/spline-fitter-worker, built by Trunk next
        /// to the page) and gives it the fit once it listens.
        pub fn start(config: SplineConfig, options: OptimizationConfig, airfoil: AirfoilData, monitor: Progress) -> Self {
            let inbox: Rc<RefCell<Vec<String>>> = Rc::default();
            let request = wire::request(&config, &options, &airfoil);
            let worker = web_sys::Worker::new("./fit-worker_loader.js");
            let (box_, sender) = (inbox.clone(), worker.clone().ok());
            let onmessage = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |event: web_sys::MessageEvent| {
                let text = event.data().as_string().unwrap_or_default();
                if text == "ready" {
                    if let Some(sender) = &sender {
                        let _ = sender.post_message(&JsValue::from_str(&request));
                    }
                } else {
                    box_.borrow_mut().push(text);
                    crate::web::repaint();
                }
            });
            let box_ = inbox.clone();
            let onerror = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |_| {
                let error: Result<RawResult> = Err(spline_fitter::Error::new("the fit stopped unexpectedly"));
                box_.borrow_mut().push(wire::done(&error));
                crate::web::repaint();
            });
            let worker = match worker {
                Ok(worker) => {
                    worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
                    worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));
                    Some(worker)
                }
                Err(e) => {
                    let error: Result<RawResult> = Err(spline_fitter::Error::new(format!("the fitting worker did not start: {e:?}")));
                    inbox.borrow_mut().push(wire::done(&error));
                    None
                }
            };
            let state = State { monitor, config, options, airfoil, last: None, done: None };
            WorkerFit { worker, inbox, _listeners: [onmessage, onerror], state: RefCell::new(state) }
        }

        /// Feeds what arrived through the monitor; true once finished.
        pub fn poll(&self) -> bool {
            let mut s = self.state.borrow_mut();
            let messages: Vec<String> = std::mem::take(&mut *self.inbox.borrow_mut());
            for text in messages {
                if s.done.is_some() {
                    break;
                }
                match wire::parse(&text) {
                    Ok(Message::Update(mut u)) => {
                        // Updates between two drawings carry no curves: the last ones hold.
                        if u.control_points.is_empty() {
                            if let Some(last) = &s.last {
                                u.control_points = last.control_points.clone();
                                u.knots = last.knots.clone();
                            }
                        }
                        s.monitor.update(&u.as_update());
                        s.last = Some(u);
                    }
                    Ok(Message::Done(result)) => {
                        let elapsed = s.monitor.start.elapsed().as_secs_f64();
                        let done = result
                            .map(|raw| outcome(&s.config, &s.options, &s.airfoil, raw, elapsed, false))
                            .map_err(spline_fitter::Error::new);
                        s.done = Some(done);
                    }
                    Err(e) => s.done = Some(Err(e)),
                }
            }
            if s.done.is_none() && s.monitor.live.cancelled() {
                // Cancel: end the worker and keep its last step.
                if let Some(worker) = &self.worker {
                    worker.terminate();
                }
                let elapsed = s.monitor.start.elapsed().as_secs_f64();
                let done = match &s.last {
                    None => Err(spline_fitter::Error::new("cancelled before the first step")),
                    Some(u) => {
                        let raw = RawResult {
                            section_control_points: u.control_points.clone(),
                            section_knots: Some(u.knots.clone()),
                            error: u.error,
                            rms_error: u.error,
                            iterations: s.monitor.iteration,
                            evaluations: u.evaluations,
                            success: false,
                            message: "Cancelled".into(),
                            seconds: None,
                            earlier: vec![],
                            search: None,
                        };
                        Ok(outcome(&s.config, &s.options, &s.airfoil, raw, elapsed, true))
                    }
                };
                s.done = Some(done);
            }
            s.done.is_some()
        }

        pub fn take(&self) -> Option<Result<Outcome>> {
            self.state.borrow_mut().done.take()
        }
    }

    impl Drop for WorkerFit {
        fn drop(&mut self) {
            if let Some(worker) = &self.worker {
                worker.terminate();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tolerance_search_shows_its_control_points() {
        let live = Live::new("test", LiveMode::Moderate);
        let mut progress = Progress {
            live: live.clone(),
            start: Instant::now(),
            prefix: String::new(),
            tolerance_mode: true,
            iteration: 0,
            best: f64::INFINITY,
            last_refit: None,
        };
        let points = |n| vec![Vec2::new(0.0, 0.0); n];
        let mut update = |sections: Vec<Vec<Vec2>>| {
            let u = Update { kind: UpdateKind::Evaluation, control_points: &sections, knots: &[], degree: 5, error: 1e-4, evaluations: 3 };
            assert!(!progress.update(&u));
            live.take().unwrap().1
        };
        assert!(update(vec![points(23)]).contains("| Control points 23 | RMS error "));
        assert!(update(vec![points(10), points(10)]).contains("| Control points 2 × 10 | "));
        // Different counts: the curve's, the shared junction point once.
        assert!(update(vec![points(10), points(12)]).contains("| Control points 21 | "));
    }
}
