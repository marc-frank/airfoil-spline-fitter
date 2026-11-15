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
    /// A finished refit of a tolerance search; the error is its maximum deviation.
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
    /// The error the run is judged by: the RMS error, or the maximum
    /// deviation in tolerance mode.
    pub error: f64,
    pub rms_error: f64,
    /// Iterations: evaluations, refits or 1 for Direct LSQ.
    pub iterations: usize,
    pub evaluations: usize,
    pub success: bool,
    pub message: String,
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
    use bspline_varpro::KnotPlacement;

    pub fn naca2412() -> AirfoilData {
        let points: Vec<Vec2> =
            Naca4::new("2412", true).unwrap().points(81).iter().map(|p| p.map(|v| (v * 1e6).round_ties_even() / 1e6)).collect();
        AirfoilData::new(points, 2, "naca2412", "NACA 2412", 6).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-3 * b.abs()
    }

    #[test]
    fn reference_fits() {
        let airfoil = naca2412();
        let config = SplineConfig { continuity: ContinuityMode::G3, geometric_continuity: true, ..Default::default() };
        let options = OptimizationConfig { knot_placement: KnotPlacement::Uniform, grow_control_points: false, ..Default::default() };
        let r = run_optimization(&config, &options, &airfoil, &mut Unmonitored).unwrap();
        assert!(close(r.error, 8.978e-7), "{}", r.error);
        assert_eq!((r.iterations, r.evaluations), (40, 40));
        assert_eq!(r.message, "`ftol` termination condition is satisfied.");

        let config = SplineConfig { continuity: ContinuityMode::G2, ..Default::default() };
        let options = OptimizationConfig { optimizer: OptimizerType::DirectLsq, ..options };
        let r = run_optimization(&config, &options, &airfoil, &mut Unmonitored).unwrap();
        assert!(close(r.error, 2.942e-7), "{}", r.error);
        assert_eq!((r.iterations, r.evaluations), (1, 2));
        assert_eq!(r.section_control_points.iter().map(Vec::len).collect::<Vec<_>>(), vec![25, 25]);
    }
}
