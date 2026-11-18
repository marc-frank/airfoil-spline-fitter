//! Variable projection fits through `bspline-varpro`: a fixed number of
//! control points per section, or the fewest within a tolerance.

use std::time::Instant;

use bspline_varpro::{fit_to_tolerance, ChainOptions, Continuity, FitOptions, FitResult, Junction, Observer, Progress, ToleranceOptions};

use super::{Monitor, RawResult, Update, UpdateKind};
use crate::models::{OptimizationConfig, SplineConfig};
use crate::{Result, Vec2};

/// Bending-energy weight of tolerance fits (it gave the fewest control
/// points and curvature extrema in the library's benchmark).
const TOLERANCE_SMOOTHING: f64 = 1e-7;
const REGULARIZATION: f64 = 1e-12;

/// A finished refit of a tolerance search with its deviation, the
/// evaluations so far and the seconds since the search began.
struct Refit {
    result: FitResult,
    deviation: f64,
    evaluations: usize,
    seconds: f64,
}

/// Forwards library progress to an application monitor, counting
/// evaluations and keeping the refits of a search.
struct Relay<'a> {
    monitor: &'a mut dyn Monitor,
    degree: usize,
    evaluations: usize,
    refits: Vec<Refit>,
    start: Instant,
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

    fn refit(&mut self, result: &FitResult, max_deviation: f64) -> bool {
        self.refits.push(Refit {
            result: result.clone(),
            deviation: max_deviation,
            evaluations: self.evaluations,
            seconds: self.start.elapsed().as_secs_f64(),
        });
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

fn raw(result: FitResult, error: f64, iterations: usize, evaluations: usize) -> RawResult {
    RawResult {
        section_control_points: result.control_points,
        section_knots: Some(result.knots),
        error,
        rms_error: result.rms_error,
        iterations,
        evaluations,
        success: result.success,
        message: result.message,
        seconds: None,
        earlier: vec![],
    }
}

pub fn fit(config: &SplineConfig, options: &OptimizationConfig, sections: &[Vec<Vec2>], monitor: &mut dyn Monitor) -> Result<RawResult> {
    let mut relay = Relay { monitor, degree: config.degree, evaluations: 0, refits: vec![], start: Instant::now() };
    let Some(tolerance) = options.tolerance else {
        let fit_options = FitOptions {
            chain: chain_options(config, options, options.smoothing_weight)?,
            knot_placement: options.knot_placement,
            grow: options.grow_control_points,
            min_knot_gap: options.min_knot_gap,
            ..FitOptions::default()
        };
        let counts = vec![config.num_control_points; sections.len()];
        let result = bspline_varpro::fit(sections, &counts, config.degree, &fit_options, &mut relay)?;
        let iterations = relay.evaluations.max(result.nfev);
        let (error, evaluations) = (result.rms_error, relay.evaluations);
        return Ok(raw(result, error, iterations, evaluations));
    };
    let tolerance_options = ToleranceOptions {
        chain: chain_options(config, options, TOLERANCE_SMOOTHING)?,
        initial_control_points: (!options.grow_control_points).then_some(config.num_control_points),
        knot_placement: options.knot_placement,
        min_knot_gap: options.min_knot_gap,
        ..ToleranceOptions::default()
    };
    let result = fit_to_tolerance(sections, tolerance, config.degree, &tolerance_options, &mut relay)?;
    let error = result.max_deviation.unwrap_or(result.rms_error);
    let mut refits = std::mem::take(&mut relay.refits);
    let count = refits.len().max(1);
    // The search ends on one of its refits (the last, or the best when it
    // stalled); the others are fits of their own.
    if let Some(i) = refits.iter().rposition(|r| r.result.control_points == result.control_points && r.result.knots == result.knots) {
        refits.remove(i);
    }
    let mut final_fit = raw(result, error, count, relay.evaluations);
    final_fit.earlier = refits
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            let mut fit = raw(r.result, r.deviation, i + 1, r.evaluations);
            fit.success = r.deviation <= tolerance;
            fit.seconds = Some(r.seconds);
            fit
        })
        .collect();
    Ok(final_fit)
}
