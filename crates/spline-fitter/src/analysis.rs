//! Measuring a fit: orthogonal deviation, fairness, junction continuity,
//! leading-edge radius, thickness and camber.

use bspline_varpro::airfoil::{default_tolerance as library_default_tolerance, leading_edge_radius, thickness_and_camber};
use bspline_varpro::chain::{chain_data, largest_extent};
use bspline_varpro::diagnostics::{fairness, junctions, project};
use bspline_varpro::{interior_knots, BSpline};

use crate::formatting::number;
use crate::models::FitResult;
use crate::Vec2;

/// Curvature measures of a fit of degree 2 or more.
#[derive(Debug, Clone, PartialEq)]
pub struct FairnessReport {
    pub curvature_extrema: usize,
    pub roughness: f64,
    pub max_curvature_slope: f64,
    pub max_curvature: f64,
    pub junctions: Vec<JunctionReport>,
    /// `None` when it could not be measured.
    pub leading_edge_radius: Option<f64>,
}

/// Continuity measured at a junction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JunctionReport {
    pub gap: f64,
    pub tangent_angle_deg: f64,
    pub curvature_jump: f64,
    pub curvature_slope_jump: f64,
}

/// Thickness and camber in fractions of the chord.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeReport {
    pub max_thickness: f64,
    pub max_thickness_x: f64,
    pub max_camber: f64,
    pub max_camber_x: f64,
}

/// Quality measurements of a fit.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub max_deviation: f64,
    pub rms_deviation: f64,
    pub section_max_deviation: Vec<f64>,
    pub control_points: usize,
    pub section_control_points: Vec<usize>,
    pub section_interior_knots: Vec<usize>,
    pub degree: usize,
    pub tolerance: Option<f64>,
    /// Largest error the rounding of the data alone causes.
    pub precision_floor: Option<f64>,
    pub fairness: Option<FairnessReport>,
    /// `None` for a chain that is not an airfoil.
    pub shape: Option<ShapeReport>,
    /// The report carries the shape keys (reports saved before them do not;
    /// a project keeps them that way, and the fit report adds them).
    pub shape_keys: bool,
}

/// Measures a fit against the sections it was fitted to.
pub fn fit_report(sections: &[Vec<Vec2>], fit: &FitResult, precision: Option<i32>, tolerance: Option<f64>) -> Report {
    let curves = fit.curves();
    let data = chain_data(sections);
    let distances: Vec<Vec<f64>> = curves.iter().map(|c| project(&data, c).into_iter().map(|(_, d)| d).collect()).collect();
    let nearest: Vec<usize> =
        (0..data.len()).map(|i| (0..curves.len()).fold(0, |b, s| if distances[s][i] < distances[b][i] { s } else { b })).collect();
    let deviation: Vec<f64> = (0..data.len()).map(|i| distances[nearest[i]][i]).collect();
    let section_max_deviation =
        (0..curves.len()).map(|s| (0..data.len()).filter(|&i| nearest[i] == s).map(|i| deviation[i]).fold(0.0, f64::max)).collect();
    let degree = fit.degree;
    let fairness = (degree >= 2).then(|| {
        let f = fairness(&curves);
        FairnessReport {
            curvature_extrema: f.curvature_extrema,
            roughness: f.roughness,
            max_curvature_slope: f.max_curvature_slope,
            max_curvature: f.max_curvature,
            junctions: junctions(&curves)
                .into_iter()
                .map(|j| JunctionReport {
                    gap: j.gap,
                    tangent_angle_deg: j.tangent_angle.to_degrees(),
                    curvature_jump: j.curvature_jump,
                    curvature_slope_jump: j.curvature_slope_jump,
                })
                .collect(),
            leading_edge_radius: Some(leading_edge_radius(&curves)).filter(|r| !r.is_nan()),
        }
    });
    Report {
        max_deviation: deviation.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        rms_deviation: (deviation.iter().map(|d| d * d).sum::<f64>() / deviation.len() as f64).sqrt(),
        section_max_deviation,
        control_points: fit.num_control_points(),
        section_control_points: fit.section_counts(),
        section_interior_knots: (0..curves.len())
            .map(|i| bspline_varpro::basis::distinct(interior_knots(&fit.knots(i), degree)).len())
            .collect(),
        degree,
        tolerance,
        precision_floor: precision.map(|p| 0.5 * 2f64.sqrt() * 10f64.powi(-p)),
        fairness,
        shape: airfoil_shape(&curves),
        shape_keys: true,
    }
}

/// Thickness and camber of a fitted chain, or `None` if it is no airfoil.
pub fn airfoil_shape(curves: &[BSpline]) -> Option<ShapeReport> {
    let shape = thickness_and_camber(curves).ok()?;
    Some(ShapeReport {
        max_thickness: shape.max_thickness,
        max_thickness_x: shape.max_thickness_x,
        max_camber: shape.max_camber,
        max_camber_x: shape.max_camber_x,
    })
}

impl Report {
    /// One line for the status bar.
    pub fn summary_line(&self) -> String {
        let mut text = format!(
            "max deviation {}, RMS {}, {} control points",
            number(self.max_deviation, ".2e"),
            number(self.rms_deviation, ".2e"),
            self.control_points
        );
        if let Some(t) = self.tolerance.filter(|t| *t != 0.0) {
            text += &format!(" (tolerance {})", number(t, ".2e"));
        }
        text
    }

    /// The rows of the report window.
    pub fn rows(&self) -> Vec<ReportRow> {
        let join = |values: &[usize]| values.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
        let mut rows = vec![
            ReportRow::new("Max orthogonal deviation", number(self.max_deviation, ".3e")),
            ReportRow::new("RMS orthogonal deviation", number(self.rms_deviation, ".3e")),
            ReportRow::new(
                "Max deviation per section",
                self.section_max_deviation.iter().map(|d| number(*d, ".2e")).collect::<Vec<_>>().join(", "),
            ),
        ];
        let tolerance = self.tolerance.filter(|t| *t != 0.0);
        if let Some(t) = tolerance {
            let met = self.max_deviation <= t * (1.0 + 1e-9);
            rows.push(ReportRow {
                label: "Tolerance",
                value: number(t, ".3e"),
                note: Some(if met { Note::Ok("met") } else { Note::Warn("not met") }),
            });
        }
        if let Some(floor) = self.precision_floor.filter(|f| *f != 0.0) {
            let below = tolerance.is_some_and(|t| t < floor);
            rows.push(ReportRow {
                label: "Rounding error of the data",
                value: number(floor, ".2e"),
                note: below.then_some(Note::Warn("tolerance is below the data rounding")),
            });
        }
        rows.extend([
            ReportRow::new("Degree", self.degree.to_string()),
            ReportRow::new("Control points (chain)", self.control_points.to_string()),
            ReportRow::new("Control points per section", join(&self.section_control_points)),
            ReportRow::new("Interior knots per section", join(&self.section_interior_knots)),
        ]);
        if let Some(f) = &self.fairness {
            rows.extend([
                ReportRow::new("Max curvature", number(f.max_curvature, ".4g")),
                ReportRow::new("Curvature extrema (wiggles)", f.curvature_extrema.to_string()),
                ReportRow::new("Roughness ∫(dκ/ds)² ds", number(f.roughness, ".4g")),
            ]);
            if let Some(r) = f.leading_edge_radius.filter(|r| *r != 0.0) {
                rows.push(ReportRow::new("Leading-edge radius", number(r, ".6g")));
            }
        }
        if let Some(s) = &self.shape {
            let at_chord = |v: f64, x: f64| format!("{}% at {}% chord", number(100.0 * v, ".3f"), number(100.0 * x, ".1f"));
            rows.push(ReportRow::new("Max thickness", at_chord(s.max_thickness, s.max_thickness_x)));
            rows.push(ReportRow::new("Max camber", at_chord(s.max_camber, s.max_camber_x)));
        }
        rows
    }

    /// The report as text: `title`, then a line per row (label, tab,
    /// value and its note in parentheses), then the junction table with tab
    /// separated columns; it pastes into a spreadsheet as cells.
    pub fn text(&self, title: &str) -> String {
        let mut lines = vec![title.to_string()];
        for row in self.rows() {
            let note = match row.note {
                Some(Note::Ok(t) | Note::Warn(t)) => format!(" ({t})"),
                None => String::new(),
            };
            lines.push(format!("{}\t{}{note}", row.label, row.value));
        }
        let junctions = self.junction_rows();
        if !junctions.is_empty() {
            lines.push(String::new());
            lines.push(JUNCTION_COLUMNS.join("\t"));
            lines.extend(junctions.iter().map(|r| r.join("\t")));
        }
        lines.join("\n") + "\n"
    }

    /// The junction table: index, gap, tangent angle, Δκ, Δ dκ/ds.
    pub fn junction_rows(&self) -> Vec<[String; 5]> {
        let Some(f) = &self.fairness else { return vec![] };
        f.junctions
            .iter()
            .enumerate()
            .map(|(i, j)| {
                [
                    (i + 1).to_string(),
                    number(j.gap, ".1e"),
                    number(j.tangent_angle_deg, ".2e") + "°",
                    number(j.curvature_jump, ".2e"),
                    number(j.curvature_slope_jump, ".2e"),
                ]
            })
            .collect()
    }
}

/// Header of [`Report::junction_rows`].
pub const JUNCTION_COLUMNS: [&str; 5] = ["Junction", "Gap", "Tangent angle", "Δκ", "Δ dκ/ds"];

/// A remark after a report value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Note {
    Ok(&'static str),
    Warn(&'static str),
}

/// A labelled value of the report window.
#[derive(Debug, Clone, PartialEq)]
pub struct ReportRow {
    pub label: &'static str,
    pub value: String,
    pub note: Option<Note>,
}

impl ReportRow {
    fn new(label: &'static str, value: String) -> Self {
        ReportRow { label, value, note: None }
    }
}

/// Signed orthogonal distances of points to the nearest section of a fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FootPoints {
    /// Positive to the right of the direction of travel (outside for
    /// Selig-ordered data): `foot + distance * normal` is the point.
    pub distance: Vec<f64>,
    pub foot: Vec<Vec2>,
    pub normal: Vec<Vec2>,
    /// The nearest section of every point (nondecreasing along the chain).
    pub section: Vec<usize>,
}

/// Foot points of `points` on the curves, each searched on the whole of
/// every curve. Not refined from a previous fit: the solver's steps can move
/// a point's parameter anywhere on the curve, so a search started from the
/// previous step's parameters lands on the wrong part of it.
pub fn foot_points(points: &[Vec2], curves: &[BSpline]) -> FootPoints {
    let n = points.len();
    let per_curve: Vec<Vec<(f64, f64)>> = curves.iter().map(|c| project(points, c)).collect();
    let parameters: Vec<Vec<f64>> = (0..n).map(|i| per_curve.iter().map(|feet| feet[i].0).collect()).collect();
    let tie = 1e-9 * largest_extent(points).max(1e-300);
    let mut result = FootPoints { distance: vec![], foot: vec![], normal: vec![], section: vec![] };
    let mut running = 0;
    for (i, point) in points.iter().enumerate() {
        let candidates: Vec<(Vec2, f64)> = curves
            .iter()
            .zip(&parameters[i])
            .map(|(c, &u)| {
                let at = c.point(u);
                (at, (point - at).norm())
            })
            .collect();
        let best = (0..curves.len()).fold(0, |b, s| if candidates[s].1 < candidates[b].1 { s } else { b });
        let (foot, distance) = candidates[best];
        let tangent = curves[best].derivative(parameters[i][best], 1);
        let normal = Vec2::new(tangent.y, -tangent.x) / tangent.norm().max(1e-300);
        let signed = distance.copysign((point - foot).dot(&normal));
        let section = (0..curves.len()).find(|&s| candidates[s].1 <= distance + tie).unwrap_or(best);
        running = running.max(section);
        result.distance.push(signed);
        result.foot.push(foot);
        result.normal.push(normal);
        result.section.push(running);
    }
    result
}

/// A tolerance a fit can meet: from the decimals of the file (0 means
/// unknown) and the noise of the points.
pub fn default_tolerance(coordinates: &[Vec2], precision: i32) -> f64 {
    library_default_tolerance(coordinates, (precision != 0).then_some(precision))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ContinuityMode, OptimizationConfig, SplineConfig};
    use crate::optimize::{run_optimization, RawResult, Unmonitored};

    fn to_fit(raw: RawResult, degree: usize) -> FitResult {
        FitResult {
            section_control_points: raw.section_control_points,
            degree,
            final_error: raw.error,
            num_iterations: raw.iterations,
            num_evaluations: raw.evaluations,
            elapsed_seconds: 0.0,
            converged: raw.success,
            cancelled: false,
            section_knots: raw.section_knots,
            report: None,
            rms_error: Some(raw.rms_error),
        }
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn reference_reports() {
        let airfoil = crate::optimize::tests::naca2412();
        assert_eq!(default_tolerance(&airfoil.coordinates, 6), 1e-6);
        let config = SplineConfig::default();
        let options = OptimizationConfig::default();
        let fit = to_fit(run_optimization(&config, &options, &airfoil, &mut Unmonitored).unwrap(), 5);
        let r = fit_report(&airfoil.sections, &fit, Some(6), None);
        assert!(close(r.max_deviation, 4.587e-6, 1e-3), "{}", r.max_deviation);
        assert!(close(r.rms_deviation, 8.978e-7, 1e-3));
        let f = r.fairness.as_ref().unwrap();
        assert_eq!(f.curvature_extrema, 5);
        assert!(close(f.leading_edge_radius.unwrap(), 0.015698, 1e-3));
        let s = r.shape.unwrap();
        assert!(close(s.max_thickness, 0.12004, 1e-4) && close(s.max_thickness_x, 0.2988, 1e-3));
        assert!(close(s.max_camber, 0.020000, 1e-3) && close(s.max_camber_x, 0.4065, 1e-3));
        assert_eq!(r.summary_line(), "max deviation 4.59e-06, RMS 8.98e-07, 49 control points");
        // As text: the title, a tab between label and value, the junction table.
        let text = r.text("NACA 2412");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "NACA 2412");
        assert_eq!(lines[1], "Max orthogonal deviation\t4.587e-06");
        assert!(lines.contains(&"Control points (chain)\t49"), "{text}");
        assert!(lines.contains(&"Junction\tGap\tTangent angle\tΔκ\tΔ dκ/ds"), "{text}");
        assert!(text.ends_with('\n'));

        let config = SplineConfig { continuity: ContinuityMode::G2, geometric_continuity: false, ..Default::default() };
        let options = OptimizationConfig { optimizer: crate::models::OptimizerType::DirectLsq, ..options };
        let fit = to_fit(run_optimization(&config, &options, &airfoil, &mut Unmonitored).unwrap(), 5);
        let r = fit_report(&airfoil.sections, &fit, Some(6), None);
        assert!(close(r.max_deviation, 4.722e-5, 1e-3), "{}", r.max_deviation);
        assert!(close(r.rms_deviation, 8.807e-6, 1e-3), "{}", r.rms_deviation);
        assert_eq!(r.fairness.unwrap().curvature_extrema, 8);
    }
}
