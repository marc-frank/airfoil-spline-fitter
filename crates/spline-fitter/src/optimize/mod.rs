//! Running a fit with the configured method.

mod direct_lsq;
mod varpro;

use crate::geometry::AirfoilData;
use crate::models::{OptimizationConfig, OptimizerType, SplineConfig};
use crate::{Result, Vec2};

/// What a progress update reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateKind {
    /// A residual evaluation; the error is the RMS error.
    Evaluation,
    /// A finished fit of a tolerance search; the error is its maximum deviation.
    Refit,
}

/// The state of a running fit.
#[derive(Debug, Clone, Copy)]
pub struct Update<'a> {
    pub kind: UpdateKind,
    pub control_points: &'a [Vec<Vec2>],
    pub knots: &'a [Vec<f64>],
    pub degree: usize,
    pub error: f64,
    /// Evaluations (residual evaluations, or linear solves) so far.
    pub evaluations: usize,
}

/// Watches a fit; returning `true` asks it to stop (Direct LSQ cannot stop).
pub trait Monitor {
    fn update(&mut self, update: &Update) -> bool;
}

/// A monitor that watches nothing.
pub struct Unmonitored;

impl Monitor for Unmonitored {
    fn update(&mut self, _: &Update) -> bool {
        false
    }
}

/// The outcome of an optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct RawResult {
    pub section_control_points: Vec<Vec<Vec2>>,
    /// Knots of every section; open uniform when `None`.
    pub section_knots: Option<Vec<Vec<f64>>>,
    /// The error the run is judged by: the RMS error (Direct LSQ: of its
    /// solve).
    pub error: f64,
    pub rms_error: f64,
    /// Iterations: evaluations, refits or 1 for Direct LSQ.
    pub iterations: usize,
    pub evaluations: usize,
    pub success: bool,
    pub message: String,
    /// Seconds from the start of the search to the end of this fit (fits
    /// of a tolerance search only).
    pub seconds: Option<f64>,
    /// The other fits of a tolerance search, in the order they ran. Each,
    /// like the result, is the fixed-count fit of its own count.
    pub earlier: Vec<RawResult>,
    /// For the result of a tolerance search: how the search ended.
    pub search: Option<SearchOutcome>,
}

/// How a tolerance search ended.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchOutcome {
    /// Control points per section of the fit it ended on.
    pub count: usize,
    /// That fit's largest orthogonal deviation.
    pub max_deviation: f64,
    /// Whether that fit is within the tolerance.
    pub met: bool,
    pub message: String,
    /// Fixed-count fits it ran (the result among them).
    pub fits: usize,
    /// Evaluations of all its fits.
    pub evaluations: usize,
    pub seconds: f64,
}

/// Fits the airfoil's sections with the configured method.
pub fn run_optimization(
    config: &SplineConfig,
    options: &OptimizationConfig,
    airfoil: &AirfoilData,
    monitor: &mut dyn Monitor,
) -> Result<RawResult> {
    match options.optimizer {
        OptimizerType::VarPro => varpro::fit(config, options, &airfoil.sections, monitor),
        OptimizerType::DirectLsq => direct_lsq::fit(config, options, &airfoil.sections, monitor),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::models::ContinuityMode;
    use bspline_varpro::airfoil::Naca4;

    pub fn naca2412() -> AirfoilData {
        let points: Vec<Vec2> =
            Naca4::new("2412", true).unwrap().points(81).iter().map(|p| p.map(|v| (v * 1e6).round_ties_even() / 1e6)).collect();
        AirfoilData::new(points, 2, "naca2412", "NACA 2412", 6).unwrap()
    }

    /// Evaluations of the reference G3 fit (specification 21, item 11).
    pub const REFERENCE_EVALUATIONS: usize = 33;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-3 * b.abs()
    }

    /// Every fit of a tolerance search is the fixed-count run of its count,
    /// so double-clicking it reproduces it; the search ends on the fewest
    /// that meet the tolerance, going up from below and down from above.
    #[test]
    fn a_search_is_a_series_of_single_runs() {
        let airfoil = naca2412();
        let config = SplineConfig { num_control_points: 8, num_sections: 2, ..Default::default() };
        let tolerance = OptimizationConfig { tolerance: Some(2e-5), grow_control_points: false, ..Default::default() };
        let up = run_optimization(&config, &tolerance, &airfoil, &mut Unmonitored).unwrap();
        let search = up.search.clone().unwrap();
        assert!(search.met, "{}", search.message);
        let mut fits: Vec<&RawResult> = up.earlier.iter().chain([&up]).collect();
        fits.sort_by_key(|f| f.section_control_points[0].len());
        let counts: Vec<usize> = fits.iter().map(|f| f.section_control_points[0].len()).collect();
        assert_eq!(counts, (8..=search.count).collect::<Vec<_>>());
        assert_eq!(search.fits, counts.len());
        for fit in &fits {
            let n = fit.section_control_points[0].len();
            assert!(fit.section_control_points.iter().all(|s| s.len() == n));
            let single = run_optimization(
                &SplineConfig { num_control_points: n, ..config.clone() },
                &OptimizationConfig { tolerance: None, ..tolerance.clone() },
                &airfoil,
                &mut Unmonitored,
            )
            .unwrap();
            assert_eq!(single.section_control_points, fit.section_control_points, "{n} control points");
            assert_eq!((single.error, single.evaluations), (fit.error, fit.evaluations));
        }
        // From above, one fewer while the tolerance holds: the same end.
        let above = SplineConfig { num_control_points: search.count + 3, ..config.clone() };
        let down = run_optimization(&above, &tolerance, &airfoil, &mut Unmonitored).unwrap();
        let from_above = down.search.unwrap();
        assert!(from_above.met && from_above.count <= search.count, "{} > {}", from_above.count, search.count);
        assert_eq!(down.section_control_points[0].len(), from_above.count);
    }

    #[test]
    fn reference_fits() {
        let airfoil = naca2412();
        let config = SplineConfig::default();
        let options = OptimizationConfig::default();
        let r = run_optimization(&config, &options, &airfoil, &mut Unmonitored).unwrap();
        assert!(close(r.error, 8.978e-7), "{}", r.error);
        assert_eq!((r.iterations, r.evaluations), (REFERENCE_EVALUATIONS, REFERENCE_EVALUATIONS));
        assert_eq!(r.message, "`ftol` termination condition is satisfied.");

        let config = SplineConfig { continuity: ContinuityMode::G2, geometric_continuity: false, ..Default::default() };
        let options = OptimizationConfig { optimizer: OptimizerType::DirectLsq, ..options };
        let r = run_optimization(&config, &options, &airfoil, &mut Unmonitored).unwrap();
        assert!(close(r.error, 1.897e-5), "{}", r.error);
        assert_eq!((r.iterations, r.evaluations), (1, 1));
        assert_eq!(r.section_control_points.iter().map(Vec::len).collect::<Vec<_>>(), vec![25, 25]);
    }
}
