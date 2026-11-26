//! Running a fit with the configured method.

mod direct_lsq;
mod varpro;

use bspline_varpro::chain::shared_junctions;
use bspline_varpro::corners::smoothest_through;
use bspline_varpro::{minimum_control_points, Junction};

use crate::geometry::AirfoilData;
use crate::models::{OptimizationConfig, OptimizerType, SplineConfig};
use crate::{bail, Result, Vec2};

/// What a progress update reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateKind {
    /// A residual evaluation; the error is the RMS error.
    Evaluation,
    /// A finished fit of a tolerance search; the error is the tolerance's
    /// metric of its deviations (by default the largest).
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
    /// That fit's value of the tolerance's metric.
    pub deviation: f64,
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

/// An airfoil's sections as an optimizer fits them: the continuity of
/// every junction, the points of every section and the control points it
/// gets.
///
/// A junction at a corner of the data is G0; the others have the
/// continuity of the layout junction they are. A section that ends at a
/// corner is as long as the data make it, so its control points follow its
/// points: the configured count, but no more than its points determine
/// (one fewer than it has) and no fewer than its end conditions need. Where
/// even those are more than its points determine, it is fitted to its
/// points and more on the smoothest curve through them
/// ([`smoothest_through`]), so no control point is left to a penalty. The
/// other sections get the configured count, as without corners.
pub(crate) struct Chain {
    /// The points each section is fitted to.
    pub sections: Vec<Vec<Vec2>>,
    /// Continuity of every junction (as a fixed one).
    pub junctions: Vec<Junction>,
    /// Fewest control points of every section, for its end conditions.
    pub need: Vec<usize>,
    /// For a section at a corner: the most control points its points
    /// determine (at least its `need`).
    most: Vec<Option<usize>>,
    /// Which points of the chain (a junction point once) are the airfoil's.
    own: Vec<bool>,
}

impl Chain {
    pub fn new(config: &SplineConfig, options: &OptimizationConfig, airfoil: &AirfoilData) -> Result<Self> {
        let junctions: Vec<Junction> = if options.optimizer == OptimizerType::VarPro {
            // The settings name the junctions of a layout: the airfoil's.
            let layout = config.junction_specs().len();
            if layout + 1 != airfoil.split.sections {
                bail!("Expected {} continuity entries, got {layout}.", airfoil.split.sections.saturating_sub(1));
            }
            config.seam_specs(&airfoil.seams).iter().map(|s| s.parse::<Junction>()).collect::<Result<_>>()?
        } else {
            // Direct LSQ joins every junction of the layout alike, parametrically.
            let order = |seam: &crate::geometry::Seam| if seam.corner { 0 } else { config.continuity.order() };
            airfoil.seams.iter().map(|seam| Junction::new(order(seam), false, true)).collect::<Result<_>>()?
        };
        let need = minimum_control_points(&junctions.iter().map(|j| j.order).collect::<Vec<_>>(), config.degree);
        let at_corner = airfoil.at_corner();
        let most: Vec<Option<usize>> =
            (0..need.len()).map(|i| at_corner[i].then(|| need[i].max(airfoil.sections[i].len().saturating_sub(1)))).collect();
        let sections: Vec<Vec<Vec2>> = airfoil
            .sections
            .iter()
            .enumerate()
            .map(|(i, points)| if at_corner[i] { smoothest_through(points, need[i] + 1) } else { points.clone() })
            .collect();
        // The airfoil's points among them, in the chain's order.
        let shared = shared_junctions(&sections);
        let mut own = Vec::new();
        for (i, (fitted, points)) in sections.iter().zip(&airfoil.sections).enumerate() {
            let mut given = points.iter().peekable();
            let flags: Vec<bool> = fitted.iter().map(|p| given.next_if(|q| *q == p).is_some()).collect();
            own.extend_from_slice(if i > 0 && shared[i - 1] { &flags[1..] } else { &flags });
        }
        Ok(Chain { sections, junctions, need, most, own })
    }

    /// Control points of every section for a configured count.
    pub fn counts(&self, count: usize) -> Vec<usize> {
        self.need.iter().zip(&self.most).map(|(&need, most)| most.map_or(count, |most| count.clamp(need, most))).collect()
    }

    /// The fewest control points a count can be: what the end conditions of
    /// every section allow.
    pub fn fewest(&self) -> usize {
        self.need.iter().copied().max().unwrap_or(2)
    }

    /// The largest count worth fitting: one fewer than the points of the
    /// smallest section that takes the count as it is; of sections at
    /// corners alone, the most any of them takes.
    pub fn most(&self) -> usize {
        let plain = self.sections.iter().zip(&self.most).filter(|(_, most)| most.is_none()).map(|(s, _)| s.len() - 1).min();
        plain.or_else(|| self.most.iter().flatten().copied().max()).unwrap_or(0).max(self.fewest())
    }

    /// The first section that takes the count as it is and has too few
    /// points for its end conditions: its index, points and need.
    pub fn too_short(&self) -> Option<(usize, usize, usize)> {
        (0..self.need.len()).find(|&i| self.most[i].is_none() && self.need[i] > self.sections[i].len() - 1).map(|i| (i, self.sections[i].len(), self.need[i]))
    }

    /// The airfoil's own points of the chain, in order.
    pub fn own<T: Copy>(&self, chain: &[T]) -> Vec<T> {
        chain.iter().zip(&self.own).filter(|(_, own)| **own).map(|(v, _)| *v).collect()
    }

    /// The RMS distance of the airfoil's own points to their curve points:
    /// `fitted` (the fit's own figure, exactly) when no points were added,
    /// else from the squared distance of every point of the chain.
    pub fn rms(&self, fitted: f64, squared: impl FnOnce() -> Vec<f64>) -> f64 {
        if self.own.iter().all(|own| *own) {
            return fitted;
        }
        let mine = self.own(&squared());
        (mine.iter().sum::<f64>() / mine.len() as f64).sqrt()
    }
}

/// Fits the airfoil's sections with the configured method.
pub fn run_optimization(
    config: &SplineConfig,
    options: &OptimizationConfig,
    airfoil: &AirfoilData,
    monitor: &mut dyn Monitor,
) -> Result<RawResult> {
    let chain = Chain::new(config, options, airfoil)?;
    match options.optimizer {
        OptimizerType::VarPro => varpro::fit(config, options, &chain, monitor),
        OptimizerType::DirectLsq => direct_lsq::fit(config, options, &chain, monitor),
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

    /// How far the curve's direction turns at every junction, in degrees.
    fn tangent_breaks(sections: &[Vec<Vec2>]) -> Vec<f64> {
        sections
            .windows(2)
            .map(|w| {
                let (u, v) = (w[0][w[0].len() - 1] - w[0][w[0].len() - 2], w[1][1] - w[1][0]);
                (u.x * v.y - u.y * v.x).atan2(u.dot(&v)).abs().to_degrees()
            })
            .collect()
    }

    fn cut(points: Vec<Vec2>, config: &SplineConfig) -> AirfoilData {
        AirfoilData::new(points, config.split(), "", "", 5).unwrap()
    }

    fn counts(r: &RawResult) -> Vec<usize> {
        r.section_control_points.iter().map(Vec::len).collect()
    }

    /// A corner of the data is joined with G0 and nothing else is: the
    /// round nose of Saratov keeps the continuity of the settings, its
    /// ridge gets its corner. The sections at the ridge take the control
    /// points their points determine.
    #[test]
    fn corners_are_g0_and_nothing_else_is() {
        use crate::geometry::tests::saratov;
        let config = SplineConfig { num_control_points: 8, ..Default::default() };
        let airfoil = cut(saratov(), &config);
        assert_eq!(airfoil.section_sizes(), "[15, 9, 7]");
        let options = OptimizationConfig::default();
        let chain = Chain::new(&config, &options, &airfoil).unwrap();
        assert_eq!(chain.junctions.iter().map(|j| j.label()).collect::<Vec<_>>(), ["G3", "G0"]);
        let r = run_optimization(&config, &options, &airfoil, &mut Unmonitored).unwrap();
        assert!(r.success, "{}", r.message);
        assert_eq!(counts(&r), [8, 8, 6]);
        let breaks = tangent_breaks(&r.section_control_points);
        // The data turn 23 degrees at the ridge.
        assert!(breaks[0] < 1e-6 && (18.0..28.0).contains(&breaks[1]), "{breaks:?}");
        // Without the corner the ridge is rounded: no break anywhere, and
        // the points farther from the curve.
        let plain = SplineConfig { split_at_corners: false, ..config.clone() };
        let rounded = run_optimization(&plain, &options, &cut(saratov(), &plain), &mut Unmonitored).unwrap();
        assert_eq!(counts(&rounded), [8, 8]);
        assert!(tangent_breaks(&rounded.section_control_points)[0] < 1e-6 && rounded.rms_error > 1.5 * r.rms_error, "{} {}", rounded.rms_error, r.rms_error);
        // Direct LSQ: the same junctions, parametric. Its curve is free to
        // break at the ridge (how far, its fixed parameters leave open on
        // so few points) and is nearer the points for it.
        let lsq = OptimizationConfig { optimizer: OptimizerType::DirectLsq, ..options.clone() };
        let chain = Chain::new(&config, &lsq, &airfoil).unwrap();
        assert_eq!(chain.junctions.iter().map(|j| j.label()).collect::<Vec<_>>(), ["C3", "C0"]);
        let d = run_optimization(&config, &lsq, &airfoil, &mut Unmonitored).unwrap();
        let breaks = tangent_breaks(&d.section_control_points);
        assert_eq!(counts(&d), [8, 8, 6]);
        assert!(breaks[0] < 1e-6 && breaks[1] > 10.0, "{breaks:?}");
        let rounded = run_optimization(&plain, &lsq, &cut(saratov(), &plain), &mut Unmonitored).unwrap();
        assert!(tangent_breaks(&rounded.section_control_points)[0] < 1e-6 && rounded.rms_error > 1.5 * d.rms_error, "{} {}", rounded.rms_error, d.rms_error);
        // The continuity of each junction of the layout stays its own: with
        // four sections the third junction is the ridge.
        let four = SplineConfig { num_sections: 4, junction_continuity: vec!["G2".into(), "G3".into(), "G1".into()], ..config.clone() };
        let chain = Chain::new(&four, &options, &cut(saratov(), &four)).unwrap();
        assert_eq!(chain.junctions.iter().map(|j| j.label()).collect::<Vec<_>>(), ["G2", "G3", "G0"]);
        // An airfoil cut for other settings is not fitted as if it were.
        let three = SplineConfig { num_sections: 3, ..config.clone() };
        assert_eq!(run_optimization(&three, &options, &airfoil, &mut Unmonitored).unwrap_err().0, "Expected 1 continuity entries, got 2.");
    }

    /// A section of very few points between two corners is fitted to its
    /// own points whatever the smoothing, by either method: its control
    /// points stay on the face they describe.
    #[test]
    fn a_few_points_between_corners_decide_their_section() {
        use crate::geometry::tests::cut_nose;
        let config = SplineConfig { num_control_points: 10, ..Default::default() };
        let airfoil = cut(cut_nose(), &config);
        assert_eq!(airfoil.section_sizes(), "[126, 3, 126]");
        let face = &airfoil.sections[1];
        for (optimizer, smoothing_weight) in [(OptimizerType::VarPro, 1e-3), (OptimizerType::VarPro, 0.0), (OptimizerType::DirectLsq, 0.0)] {
            let options = OptimizationConfig { optimizer, smoothing_weight, ..Default::default() };
            let chain = Chain::new(&config, &options, &airfoil).unwrap();
            // Six control points need seven points: four more than the face has.
            assert_eq!((chain.sections[1].len(), chain.own.iter().filter(|own| !**own).count()), (7, 4));
            assert_eq!(chain.own(&bspline_varpro::chain::chain_data(&chain.sections)), airfoil.coordinates);
            let r = run_optimization(&config, &options, &airfoil, &mut Unmonitored).unwrap();
            assert_eq!(counts(&r), [10, 6, 10], "{optimizer} {smoothing_weight}");
            // The face runs from corner to corner through its middle
            // point, never leaving the strip its three points span.
            let control = &r.section_control_points[1];
            assert!((control[0] - face[0]).norm() < 1e-12 && (control[5] - face[2]).norm() < 1e-12, "{control:?}");
            let curve = crate::BSpline { degree: 5, knots: bspline_varpro::open_uniform_knots(5, 6).unwrap(), control_points: control.clone() };
            let on_face: Vec<Vec2> = (0..=40).map(|i| curve.derivative(i as f64 / 40.0, 0)).collect();
            let within = |p: &Vec2| (-0.000201..=1e-9).contains(&p.x) && p.y.abs() <= 0.004 + 1e-9;
            assert!(on_face.iter().all(within), "{optimizer} {smoothing_weight}: {control:?}");
            assert!(on_face.windows(2).all(|w| w[1].y < w[0].y), "{optimizer} {smoothing_weight}: {control:?}");
            let middle = bspline_varpro::chain::chain_deviations(&face[1..2], &[curve])[0];
            assert!(middle < 2e-6, "{optimizer} {smoothing_weight}: {middle}");
            assert!(r.rms_error.is_finite() && r.rms_error < 1e-4, "{optimizer} {smoothing_weight}: {}", r.rms_error);
            let breaks = tangent_breaks(&r.section_control_points);
            assert!(breaks.iter().all(|b| (60.0..100.0).contains(b)), "{optimizer} {smoothing_weight}: {breaks:?}");
        }
    }

    /// A search with corners: every fit is the single run of its count, in
    /// which a section at a corner takes what its points determine; the
    /// count goes as far as the points of the other sections allow.
    #[test]
    fn a_search_sizes_sections_at_corners_by_their_points() {
        use crate::geometry::tests::{saratov, wedge};
        let config = SplineConfig { num_control_points: 6, ..Default::default() };
        let airfoil = cut(saratov(), &config);
        let tolerance = OptimizationConfig { tolerance: Some(1e-9), ..Default::default() };
        let chain = Chain::new(&config, &tolerance, &airfoil).unwrap();
        assert_eq!((chain.fewest(), chain.most(), chain.too_short()), (6, 14, None));
        assert_eq!((chain.counts(6), chain.counts(9), chain.counts(14)), (vec![6, 6, 6], vec![9, 8, 6], vec![14, 8, 6]));
        // The tolerance is out of reach: the search fits every count.
        let r = run_optimization(&config, &tolerance, &airfoil, &mut Unmonitored).unwrap();
        let search = r.search.clone().unwrap();
        assert!(!search.met && search.fits == 9, "{} fits: {}", search.fits, search.message);
        assert!(search.message.starts_with("Tolerance not reached: 14 control points are as many as the points of a section allow"), "{}", search.message);
        for fit in r.earlier.iter().chain([&r]) {
            let n = counts(fit).into_iter().max().unwrap();
            assert_eq!(counts(fit), chain.counts(n));
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
        // Both sections of a sharp nose end at its corner: each takes the
        // count up to its own points, the longer one further.
        let mut sharp = wedge(0.05, None);
        sharp.truncate(sharp.len() - 4);
        let airfoil = cut(sharp, &config);
        assert_eq!(airfoil.section_sizes(), "[21, 17]");
        let chain = Chain::new(&config, &tolerance, &airfoil).unwrap();
        assert_eq!((chain.fewest(), chain.most(), chain.counts(18)), (6, 20, vec![18, 16]));
    }

    /// Without corners in the data the setting changes nothing.
    #[test]
    fn airfoils_without_corners_are_fitted_as_before() {
        let with = SplineConfig::default();
        let without = SplineConfig { split_at_corners: false, ..with.clone() };
        assert!(with.split_at_corners);
        let (a, b) = (naca2412().resplit(with.split()).unwrap(), naca2412().resplit(without.split()).unwrap());
        assert_eq!((&a.sections, &a.junction_indices, &a.seams), (&b.sections, &b.junction_indices, &b.seams));
        for options in [OptimizationConfig::default(), OptimizationConfig { tolerance: Some(2e-5), ..Default::default() }] {
            let (x, y) = (run_optimization(&with, &options, &a, &mut Unmonitored).unwrap(), run_optimization(&without, &options, &b, &mut Unmonitored).unwrap());
            assert_eq!((&x.section_control_points, x.error, x.evaluations), (&y.section_control_points, y.error, y.evaluations));
        }
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

    /// With the RMS deviation as the target the search ends on the fewest
    /// whose RMS meets the tolerance: never more than for the largest
    /// deviation, which is at least the RMS.
    #[test]
    fn the_target_can_be_the_rms_deviation() {
        use crate::models::ToleranceMetric;
        let airfoil = naca2412();
        let config = SplineConfig { num_control_points: 8, num_sections: 2, ..Default::default() };
        let max = OptimizationConfig { tolerance: Some(2e-5), ..Default::default() };
        let rms = OptimizationConfig { tolerance_metric: ToleranceMetric::RmsDeviation, ..max.clone() };
        let by_max = run_optimization(&config, &max, &airfoil, &mut Unmonitored).unwrap().search.unwrap();
        let r = run_optimization(&config, &rms, &airfoil, &mut Unmonitored).unwrap();
        let by_rms = r.search.clone().unwrap();
        assert!(by_rms.met && by_rms.deviation <= 2e-5 && by_rms.deviation <= by_rms.max_deviation);
        assert!(by_rms.count < by_max.count, "{} >= {}", by_rms.count, by_max.count);
        // The search's value is the report's RMS deviation.
        let fit = crate::models::FitResult {
            section_control_points: r.section_control_points.clone(),
            degree: config.degree,
            final_error: r.error,
            num_iterations: 1,
            num_evaluations: 1,
            elapsed_seconds: 0.0,
            converged: true,
            cancelled: false,
            section_knots: r.section_knots.clone(),
            report: None,
            rms_error: Some(r.rms_error),
        };
        let report = crate::analysis::fit_report_for(&airfoil.sections, &fit, Some(6), Some(2e-5), ToleranceMetric::RmsDeviation);
        assert!((report.rms_deviation - by_rms.deviation).abs() <= 1e-3 * by_rms.deviation, "{} {}", report.rms_deviation, by_rms.deviation);
        assert_eq!((report.target(), report.within_tolerance()), (report.rms_deviation, Some(true)));
        assert!(report.rows().iter().any(|row| row.label == "RMS tolerance"));
        assert!(report.summary_line().ends_with("(RMS tolerance 2.00e-05)"), "{}", report.summary_line());
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
