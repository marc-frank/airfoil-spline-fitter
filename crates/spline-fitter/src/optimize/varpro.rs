//! Variable projection fits through `bspline-varpro`: a fixed number of
//! control points per section, or the fewest within a tolerance.

use bspline_varpro::{
    fit_to_tolerance, ChainOptions, Continuity, FitOptions, FitResult, Junction, Observer, Progress, ToleranceOptions,
};

use super::{Monitor, RawResult, Update, UpdateKind};
use crate::models::{OptimizationConfig, SplineConfig};
use crate::{Result, Vec2};

/// Bending-energy weight of tolerance fits (it gave the fewest control
/// points and curvature extrema in the library's benchmark).
const TOLERANCE_SMOOTHING: f64 = 1e-7;
const REGULARIZATION: f64 = 1e-12;

/// Forwards library progress to an application monitor, counting evaluations.
struct Relay<'a> {
    monitor: &'a mut dyn Monitor,
    degree: usize,
    evaluations: usize,
    refits: usize,
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
        self.refits += 1;
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

pub fn fit(config: &SplineConfig, options: &OptimizationConfig, sections: &[Vec<Vec2>], monitor: &mut dyn Monitor) -> Result<RawResult> {
    let mut relay = Relay { monitor, degree: config.degree, evaluations: 0, refits: 0 };
    let (result, error, iterations) = match options.tolerance {
        None => {
            let fit_options = FitOptions {
                chain: chain_options(config, options, options.smoothing_weight)?,
                max_points_per_section: options.max_points_per_section,
                knot_placement: options.knot_placement,
                grow: options.grow_control_points,
                ..FitOptions::default()
            };
            let counts = vec![config.num_control_points; sections.len()];
            let result = bspline_varpro::fit(sections, &counts, config.degree, &fit_options, &mut relay)?;
            let iterations = relay.evaluations.max(result.nfev);
            let error = result.rms_error;
            (result, error, iterations)
        }
        Some(tolerance) => {
            let tolerance_options = ToleranceOptions {
                chain: chain_options(config, options, TOLERANCE_SMOOTHING)?,
                initial_control_points: (!options.grow_control_points).then_some(config.num_control_points),
                max_points_per_section: options.max_points_per_section,
                knot_placement: options.knot_placement,
                ..ToleranceOptions::default()
            };
            let result = fit_to_tolerance(sections, tolerance, config.degree, &tolerance_options, &mut relay)?;
            let error = result.max_deviation.unwrap_or(result.rms_error);
            (result, error, relay.refits.max(1))
        }
    };
    Ok(RawResult {
        section_control_points: result.control_points,
        section_knots: Some(result.knots),
        error,
        rms_error: result.rms_error,
        iterations,
        evaluations: relay.evaluations,
        success: result.success,
        message: result.message,
    })
}
