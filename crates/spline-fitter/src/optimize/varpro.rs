//! Variable projection fits through `bspline-varpro`: a fixed number of
//! control points per section, or the fewest within a tolerance.
//!
//! A tolerance search is a series of fixed-count fits, each exactly the fit
//! a single run with that count makes (fresh, from the same start), so any
//! of them can be reproduced as a run of its own.

use web_time::Instant;

use bspline_varpro::chain::{chain_data, chain_deviations, evaluate_chain};
use bspline_varpro::{ChainOptions, Continuity, FitOptions, FitResult, Observer, Progress, STOPPED};

use super::{Chain, Monitor, RawResult, SearchOutcome, Update, UpdateKind};
use crate::models::{OptimizationConfig, SplineConfig};
use crate::{bail, Result, Vec2};

const REGULARIZATION: f64 = 1e-12;
/// The most fits of one search. There is no rule that stops a search whose
/// deviation stays level: fresh fits stay level for many counts and then
/// drop (Darkstar tip at 1e-5, smoothing 1e-7: about 3e-5 from 19 to 57
/// control points, within the tolerance at 64), so any such rule stopped
/// searches that would have met it. Esc cancels one that takes too long.
const MAX_FITS: usize = 200;

/// Forwards library progress to an application monitor, counting
/// evaluations.
struct Relay<'a> {
    monitor: &'a mut dyn Monitor,
    degree: usize,
    evaluations: usize,
}

impl Observer for Relay<'_> {
    fn step(&mut self, progress: &Progress) -> bool {
        self.evaluations += 1;
        self.monitor.update(&Update {
            kind: UpdateKind::Evaluation,
            control_points: progress.control_points,
            knots: progress.knots,
            degree: self.degree,
            error: progress.rms,
            evaluations: self.evaluations,
        })
    }
}

impl Relay<'_> {
    /// Reports a finished fit of a search; true asks the search to stop.
    fn refit(&mut self, result: &FitResult, max_deviation: f64) -> bool {
        self.monitor.update(&Update {
            kind: UpdateKind::Refit,
            control_points: &result.control_points,
            knots: &result.knots,
            degree: self.degree,
            error: max_deviation,
            evaluations: self.evaluations,
        })
    }
}

fn chain_options(chain: &Chain, options: &OptimizationConfig, smoothing: f64) -> ChainOptions {
    ChainOptions {
        continuity: Continuity::Each(chain.junctions.clone()),
        fix_junctions: !options.optimize_le_position,
        fix_ends: true,
        smoothing,
        regularization: REGULARIZATION,
    }
}

/// A fixed-count fit as a run: judged by its RMS error, of the airfoil's
/// own points.
fn raw(chain: &Chain, result: FitResult, evaluations: usize) -> RawResult {
    let rms = chain.rms(result.rms_error, || {
        let (data, curves) = (chain_data(&chain.sections), result.curves());
        data.iter().zip(&result.parameters).map(|(d, &u)| (evaluate_chain(&curves, u, 0) - d).norm_squared()).collect()
    });
    RawResult {
        error: rms,
        rms_error: rms,
        iterations: evaluations.max(result.nfev),
        evaluations,
        success: result.success,
        message: result.message,
        section_control_points: result.control_points,
        section_knots: Some(result.knots),
        seconds: None,
        earlier: vec![],
        search: None,
    }
}

/// The options of a fixed-count fit (a search's fits never grow: each is
/// the fit of its own count).
fn fit_options(chain: &Chain, options: &OptimizationConfig, grow: bool) -> FitOptions {
    FitOptions {
        chain: chain_options(chain, options, options.smoothing_weight),
        knot_placement: options.knot_placement,
        grow,
        min_knot_gap: options.min_knot_gap,
        ..FitOptions::default()
    }
}

pub fn fit(config: &SplineConfig, options: &OptimizationConfig, chain: &Chain, monitor: &mut dyn Monitor) -> Result<RawResult> {
    let mut relay = Relay { monitor, degree: config.degree, evaluations: 0 };
    let Some(tolerance) = options.tolerance else {
        let fit_options = fit_options(chain, options, options.grow_control_points);
        let counts = chain.counts(config.num_control_points);
        let result = bspline_varpro::fit(&chain.sections, &counts, config.degree, &fit_options, &mut relay)?;
        let evaluations = relay.evaluations;
        return Ok(raw(chain, result, evaluations));
    };
    search(config, options, tolerance, chain, &mut relay)
}

/// A finished fit of a search.
struct Fitted {
    raw: RawResult,
    count: usize,
    /// The tolerance's metric of its deviations.
    deviation: f64,
    max_deviation: f64,
    /// The data point farthest from the curve.
    worst: Vec2,
    stopped: bool,
}

/// The fewest control points (the same count for every section; a section
/// at a corner takes of it what its points determine) whose fixed-count fit
/// keeps every point within `tolerance`: from the start count, one more at
/// a time until a fit meets it, or one fewer at a time while they do.
fn search(
    config: &SplineConfig,
    options: &OptimizationConfig,
    tolerance: f64,
    chain: &Chain,
    relay: &mut Relay,
) -> Result<RawResult> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        bail!("Tolerance must be positive.");
    }
    let degree = config.degree;
    let fit_options = fit_options(chain, options, false);
    // Name the section and the counts, so the user sees what to change.
    if let Some((s, points, n)) = chain.too_short() {
        bail!(
            "Section {} has {points} points, too few for its end conditions: they need {n} control points, so at least {} points.",
            s + 1,
            n + 1
        );
    }
    let low = chain.fewest();
    let limit = chain.most();
    let start = if options.grow_control_points { low } else { config.num_control_points.clamp(low, limit) };
    let data = chain.own(&chain_data(&chain.sections));
    let began = Instant::now();
    let mut fits: Vec<Fitted> = Vec::new();
    let one = |count: usize, relay: &mut Relay| -> Result<Fitted> {
        let before = relay.evaluations;
        let result = bspline_varpro::fit(&chain.sections, &chain.counts(count), degree, &fit_options, relay)?;
        let deviations = chain_deviations(&data, &result.curves());
        let worst = (0..deviations.len()).fold(0, |b, i| if deviations[i] > deviations[b] { i } else { b });
        let max_deviation = deviations.get(worst).copied().unwrap_or(0.0);
        let deviation = options.tolerance_metric.of(&deviations);
        let stopped = result.stopped || relay.refit(&result, deviation);
        let mut fit = raw(chain, result, relay.evaluations - before);
        fit.seconds = Some(began.elapsed().as_secs_f64());
        Ok(Fitted { raw: fit, count, deviation, max_deviation, worst: data[worst], stopped })
    };

    // The fit the search ends on, with whether it met the tolerance and why.
    let (chosen, met, message) = 'search: {
        let first = one(start, relay)?;
        let first_met = first.deviation <= tolerance;
        let first_stopped = first.stopped;
        fits.push(first);
        if first_stopped {
            break 'search (0, false, STOPPED.to_string());
        }
        let within = format!("Within tolerance {}.", bspline_varpro::numfmt::general6(tolerance));
        if first_met {
            // Fewer while the tolerance holds.
            let mut best = 0;
            for count in (low..start).rev() {
                let fit = one(count, relay)?;
                let (ok, stopped) = (fit.deviation <= tolerance, fit.stopped);
                fits.push(fit);
                if stopped {
                    break 'search (best, true, STOPPED.to_string());
                }
                if !ok {
                    break;
                }
                best = fits.len() - 1;
            }
            break 'search (best, true, within);
        }
        let mut closest = 0;
        // Where the closest fit is farthest from the data.
        let near = |f: &Fitted| format!("the closest fit, {} control points, is farthest near ({:.4}, {:.4})", f.count, f.worst.x, f.worst.y);
        for count in start + 1..=limit {
            if fits.len() >= MAX_FITS {
                break 'search (closest, false, format!("Tolerance not reached in {MAX_FITS} fits; {}.", near(&fits[closest])));
            }
            let fit = one(count, relay)?;
            let (deviation, stopped) = (fit.deviation, fit.stopped);
            fits.push(fit);
            let index = fits.len() - 1;
            if stopped {
                break 'search (closest, false, STOPPED.to_string());
            }
            if deviation <= tolerance {
                break 'search (index, true, within);
            }
            if deviation < fits[closest].deviation {
                closest = index;
            }
        }
        (
            closest,
            false,
            format!(
                "Tolerance not reached: {limit} control points are as many as the points of a section allow; {}.",
                near(&fits[closest])
            ),
        )
    };
    // A fit stopped part way is not one of the search's fits, unless it is
    // the only one.
    if fits.len() > 1 && fits.last().is_some_and(|f| f.stopped) && chosen != fits.len() - 1 {
        fits.pop();
    }
    let total = relay.evaluations;
    let found = fits.remove(chosen);
    let mut result = found.raw;
    result.search = Some(SearchOutcome {
        count: found.count,
        deviation: found.deviation,
        max_deviation: found.max_deviation,
        met,
        message,
        fits: fits.len() + 1,
        evaluations: total,
        seconds: began.elapsed().as_secs_f64(),
    });
    result.earlier = fits.into_iter().map(|f| f.raw).collect();
    Ok(result)
}
