//! Variable projection fits through `bspline-varpro`: a fixed number of
//! control points per section, or the fewest within a tolerance.
//!
//! A tolerance search is a series of fixed-count fits, each exactly the fit
//! a single run with that count makes (fresh, from the same start), so any
//! of them can be reproduced as a run of its own.

use std::time::Instant;

use bspline_varpro::chain::{chain_data, chain_deviations};
use bspline_varpro::{minimum_control_points, ChainOptions, Continuity, FitOptions, FitResult, Junction, Observer, Progress, STOPPED};

use super::{Monitor, RawResult, SearchOutcome, Update, UpdateKind};
use crate::models::{OptimizationConfig, SplineConfig};
use crate::{bail, Result, Vec2};

const REGULARIZATION: f64 = 1e-12;
/// Control points added without a gain of [`STALL_GAIN`] in the largest
/// deviation after which a search stops: this many, or three quarters of the
/// count of the last gain if that is more. The deviation of measured
/// airfoils stays level for up to 20 control points before it drops again.
const STALL_POINTS: usize = 10;
/// The relative drop of the largest deviation that counts as a gain.
const STALL_GAIN: f64 = 0.2;
/// The most fits of one search.
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

fn chain_options(config: &SplineConfig, options: &OptimizationConfig, smoothing: f64) -> Result<ChainOptions> {
    let junctions = config.junction_specs().iter().map(|s| s.parse::<Junction>()).collect::<Result<Vec<_>>>()?;
    Ok(ChainOptions {
        continuity: Continuity::Each(junctions),
        fix_junctions: !options.optimize_le_position,
        fix_ends: true,
        smoothing,
        regularization: REGULARIZATION,
    })
}

/// A fixed-count fit as a run: judged by its RMS error.
fn raw(result: FitResult, evaluations: usize) -> RawResult {
    RawResult {
        error: result.rms_error,
        rms_error: result.rms_error,
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
fn fit_options(config: &SplineConfig, options: &OptimizationConfig, grow: bool) -> Result<FitOptions> {
    Ok(FitOptions {
        chain: chain_options(config, options, options.smoothing_weight)?,
        knot_placement: options.knot_placement,
        grow,
        min_knot_gap: options.min_knot_gap,
        ..FitOptions::default()
    })
}

pub fn fit(config: &SplineConfig, options: &OptimizationConfig, sections: &[Vec<Vec2>], monitor: &mut dyn Monitor) -> Result<RawResult> {
    let mut relay = Relay { monitor, degree: config.degree, evaluations: 0 };
    let Some(tolerance) = options.tolerance else {
        let fit_options = fit_options(config, options, options.grow_control_points)?;
        let counts = vec![config.num_control_points; sections.len()];
        let result = bspline_varpro::fit(sections, &counts, config.degree, &fit_options, &mut relay)?;
        let evaluations = relay.evaluations;
        return Ok(raw(result, evaluations));
    };
    search(config, options, tolerance, sections, &mut relay)
}

/// A finished fit of a search.
struct Fitted {
    raw: RawResult,
    count: usize,
    deviation: f64,
    /// The data point farthest from the curve.
    worst: Vec2,
    stopped: bool,
}

/// The fewest control points (the same in every section) whose fixed-count
/// fit keeps every point within `tolerance`: from the start count, one more
/// at a time until a fit meets it, or one fewer at a time while they do.
fn search(
    config: &SplineConfig,
    options: &OptimizationConfig,
    tolerance: f64,
    sections: &[Vec<Vec2>],
    relay: &mut Relay,
) -> Result<RawResult> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        bail!("Tolerance must be positive.");
    }
    let degree = config.degree;
    let fit_options = fit_options(config, options, false)?;
    let need = minimum_control_points(&fit_options.chain.orders(sections.len())?, degree);
    // Name the section and the counts, so the user sees what to change.
    for (s, (points, &n)) in sections.iter().map(Vec::len).zip(&need).enumerate() {
        if n > points - 1 {
            bail!(
                "Section {} has {points} points, too few for its end conditions: they need {n} control points, so at least {} points.",
                s + 1,
                n + 1
            );
        }
    }
    let low = need.iter().copied().max().unwrap_or(2);
    let limit = sections.iter().map(|s| s.len() - 1).min().unwrap_or(low).max(low);
    let start = if options.grow_control_points { low } else { config.num_control_points.clamp(low, limit) };
    let data = chain_data(sections);
    let began = Instant::now();
    let mut fits: Vec<Fitted> = Vec::new();
    let one = |count: usize, relay: &mut Relay| -> Result<Fitted> {
        let before = relay.evaluations;
        let result = bspline_varpro::fit(sections, &vec![count; sections.len()], degree, &fit_options, relay)?;
        let deviations = chain_deviations(&data, &result.curves());
        let worst = (0..deviations.len()).fold(0, |b, i| if deviations[i] > deviations[b] { i } else { b });
        let deviation = deviations.get(worst).copied().unwrap_or(0.0);
        let stopped = result.stopped || relay.refit(&result, deviation);
        let mut fit = raw(result, relay.evaluations - before);
        fit.seconds = Some(began.elapsed().as_secs_f64());
        Ok(Fitted { raw: fit, count, deviation, worst: data[worst], stopped })
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
        // The largest deviation and count of the last fit that gained
        // STALL_GAIN on the one before.
        let mut gained = (fits[0].deviation, start);
        for count in start + 1..=limit {
            if fits.len() >= MAX_FITS {
                break 'search (closest, false, "Tolerance not reached within max_iterations.".into());
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
            // A kink or noise in the data holds the largest deviation where
            // more control points cannot lower it: stop with the closest fit.
            if deviation <= (1.0 - STALL_GAIN) * gained.0 {
                gained = (deviation, count);
            }
            let added = count - gained.1;
            if added >= STALL_POINTS.max(3 * gained.1 / 4) {
                let p = fits[closest].worst;
                break 'search (
                    closest,
                    false,
                    format!(
                        "Tolerance not reached: {added} more control points lowered the largest deviation by less than a fifth; it stays near ({:.4}, {:.4}); the data may have a kink or noise there.",
                        p.x, p.y
                    ),
                );
            }
        }
        (closest, false, format!("Tolerance not reached: {limit} control points are as many as the points of a section allow."))
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
        max_deviation: found.deviation,
        met,
        message,
        fits: fits.len() + 1,
        evaluations: total,
        seconds: began.elapsed().as_secs_f64(),
    });
    result.earlier = fits.into_iter().map(|f| f.raw).collect();
    Ok(result)
}
