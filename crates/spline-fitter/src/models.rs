//! Fit settings and results.

use std::fmt;

use bspline_varpro::{minimum_control_points as section_minimums, BSpline, KnotPlacement};

use crate::analysis::Report;
use crate::geometry::{Seam, Split};
use crate::{bail, Result, Vec2};

/// Continuity order linked across junctions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ContinuityMode {
    G0,
    G1,
    G2,
    G3,
}

impl ContinuityMode {
    pub const ALL: [ContinuityMode; 4] = [Self::G0, Self::G1, Self::G2, Self::G3];

    pub fn order(self) -> usize {
        self as usize
    }

    pub fn from_order(order: usize) -> Self {
        Self::ALL[order.min(3)]
    }

    /// Lowest degree that can carry this continuity.
    pub fn minimum_degree(self) -> usize {
        self.order().max(1)
    }

    /// The stored name: `G0`..`G3`.
    pub fn name(self) -> &'static str {
        ["G0", "G1", "G2", "G3"][self.order()]
    }

    /// `G0`..`G3` (case-insensitive), else `G1`.
    pub fn parse_or_default(text: &str) -> Self {
        Self::ALL.into_iter().find(|m| m.name().eq_ignore_ascii_case(text)).unwrap_or(Self::G1)
    }
}

/// Fewest control points per section for junctions of the given orders:
/// a section needs room for the conditions at both of its ends.
pub fn minimum_control_points(orders: &[usize], degree: usize) -> usize {
    section_minimums(orders, degree).into_iter().max().unwrap_or(degree + 1)
}

/// Fitting method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizerType {
    /// Levenberg-Marquardt variable projection.
    VarPro,
    /// A single constrained least-squares solve.
    DirectLsq,
}

impl OptimizerType {
    pub const ALL: [OptimizerType; 2] = [Self::VarPro, Self::DirectLsq];

    /// The stored value.
    pub fn value(self) -> &'static str {
        match self {
            Self::VarPro => "gn-varpro",
            Self::DirectLsq => "direct-lsq",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::VarPro => "VarPro LM",
            Self::DirectLsq => "Direct LSQ",
        }
    }

    pub fn tooltip(self) -> &'static str {
        match self {
            Self::VarPro => {
                "Levenberg-Marquardt Variable Projection (bspline-varpro).\n\
                 Tolerance mode: fewest control points within a maximum\n\
                 orthogonal deviation, non-uniform knots.\n\
                 Geometric (G) or parametric (C) continuity per junction.\n\
                 Recommended for CAD-quality results."
            }
            Self::DirectLsq => {
                "Direct LSQ: a single solve, no iteration.\n\
                 One constrained least-squares solve on uniform knots, with the\n\
                 data parameters taken from the point spacing (Parameterization).\n\
                 Parametric continuity; junctions fixed or free (Advanced).\n\
                 Fastest, but less accurate than VarPro LM."
            }
        }
    }

    /// The type of a stored value, mapping retired optimizers to VarPro.
    /// Returns the type and whether the value was known.
    pub fn from_value(value: &str) -> (Self, bool) {
        match value.to_lowercase().as_str() {
            "gn-varpro" | "hierarchical-lsq" => (Self::VarPro, true),
            "direct-lsq" | "constrained-lsq" => (Self::DirectLsq, true),
            "orthogonal-lsq" | "param-opt" | "bfgs" | "slsqp" | "l-bfgs-b" | "cg" | "tnc" | "powell" | "nelder-mead" | "cobyla" | "de"
            | "da" | "bh" => (Self::VarPro, true),
            _ => (Self::VarPro, false),
        }
    }
}

impl fmt::Display for OptimizerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.display_name())
    }
}

/// Spline shape settings.
#[derive(Debug, Clone, PartialEq)]
pub struct SplineConfig {
    pub degree: usize,
    pub num_control_points: usize,
    pub num_sections: usize,
    pub continuity: ContinuityMode,
    /// Geometric (G) instead of parametric (C) continuity.
    pub geometric_continuity: bool,
    /// Per-junction continuity (`G2`, `C1`, ...), overriding the above.
    pub junction_continuity: Vec<String>,
    /// Cut the outline at the clear corners of the data too and join the
    /// sections there with G0, whatever the continuity of the junctions the
    /// section count places.
    pub split_at_corners: bool,
}

impl Default for SplineConfig {
    fn default() -> Self {
        SplineConfig {
            degree: 5,
            num_control_points: 25,
            num_sections: 2,
            continuity: ContinuityMode::G3,
            geometric_continuity: true,
            junction_continuity: vec![],
            split_at_corners: true,
        }
    }
}

impl SplineConfig {
    /// Checks the settings and normalizes the per-junction entries.
    pub fn validated(mut self) -> Result<Self> {
        let s = self.num_sections;
        if s < 1 {
            bail!("num_sections must be >= 1, got {s}");
        }
        self.junction_continuity = self.junction_continuity.iter().map(|j| j.trim().to_uppercase()).collect();
        let minimum = self.continuity.minimum_degree();
        if s > 1 && self.junction_continuity.is_empty() && self.degree < minimum {
            bail!("{} continuity requires degree >= {minimum}, got {}", self.continuity.name(), self.degree);
        }
        if !self.junction_continuity.is_empty() {
            if self.junction_continuity.len() != s - 1 {
                bail!("{s} sections have {} junction(s), but {} continuity entries were given", s - 1, self.junction_continuity.len());
            }
            for spec in &self.junction_continuity {
                let bytes = spec.as_bytes();
                if bytes.len() != 2 || !b"CG".contains(&bytes[0]) || !b"0123".contains(&bytes[1]) {
                    bail!("Invalid junction continuity {}; use e.g. G2 or C1", bspline_varpro::numfmt::quoted(spec));
                }
                let digit = (bytes[1] - b'0') as usize;
                if digit > self.degree {
                    bail!("{spec} continuity requires degree >= {digit}");
                }
            }
        }
        Ok(self)
    }

    /// How an outline is cut into sections for these settings.
    pub fn split(&self) -> Split {
        Split { sections: self.num_sections, corners: self.split_at_corners }
    }

    /// The continuity of every junction of the layout (the `num_sections
    /// - 1` the section count places; a corner of the data is G0): `G2`,
    /// `C1`, ...
    pub fn junction_specs(&self) -> Vec<String> {
        if !self.junction_continuity.is_empty() {
            return self.junction_continuity.clone();
        }
        let kind = if self.geometric_continuity { 'G' } else { 'C' };
        vec![format!("{kind}{}", self.continuity.order()); self.num_sections.saturating_sub(1)]
    }

    /// The continuity of every junction of an outline cut for these
    /// settings: `G0` at a corner of the data (`C0` with parametric
    /// continuity), else that of the layout junction it is.
    pub fn seam_specs(&self, seams: &[Seam]) -> Vec<String> {
        let layout = self.junction_specs();
        let corner = format!("{}0", if self.geometric_continuity { 'G' } else { 'C' });
        seams.iter().map(|s| s.layout.filter(|_| !s.corner).and_then(|k| layout.get(k)).cloned().unwrap_or_else(|| corner.clone())).collect()
    }

    /// Orders of the junctions.
    pub fn junction_orders(&self) -> Vec<usize> {
        self.junction_specs().iter().map(|s| s[1..].parse().unwrap_or(0)).collect()
    }

    pub fn minimum_control_points(&self) -> usize {
        minimum_control_points(&self.junction_orders(), self.degree)
    }

    /// The continuity label of the configuration: the per-junction specs
    /// joined by `, `, else the mode name.
    pub fn continuity_label(&self) -> String {
        let specs = self.junction_specs();
        if specs.is_empty() {
            self.continuity.name().to_string()
        } else {
            specs.join(", ")
        }
    }
}

/// Start parameters of Direct LSQ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Parameterization {
    #[default]
    Centripetal,
    ArcLength,
}

impl Parameterization {
    pub fn value(self) -> &'static str {
        match self {
            Self::Centripetal => "centripetal",
            Self::ArcLength => "arc_length",
        }
    }

    pub fn from_value(value: &str) -> Result<Self> {
        match value {
            "centripetal" => Ok(Self::Centripetal),
            "arc_length" => Ok(Self::ArcLength),
            _ => bail!("parameterization must be 'centripetal' or 'arc_length', got {}", bspline_varpro::numfmt::quoted(value)),
        }
    }
}

pub fn placement_value(p: KnotPlacement) -> &'static str {
    match p {
        KnotPlacement::Uniform => "uniform",
        KnotPlacement::Free => "free",
    }
}

pub fn placement_from_value(value: &str) -> Result<KnotPlacement> {
    match value {
        "uniform" => Ok(KnotPlacement::Uniform),
        "free" => Ok(KnotPlacement::Free),
        _ => bail!("knot_placement must be 'uniform' or 'free', got {}", bspline_varpro::numfmt::quoted(value)),
    }
}

/// What a tolerance limits: the largest orthogonal deviation of a data
/// point from the curve, or their root mean square.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToleranceMetric {
    #[default]
    MaxDeviation,
    RmsDeviation,
}

impl ToleranceMetric {
    pub const ALL: [ToleranceMetric; 2] = [Self::MaxDeviation, Self::RmsDeviation];

    /// The stored value.
    pub fn value(self) -> &'static str {
        match self {
            Self::MaxDeviation => "max_deviation",
            Self::RmsDeviation => "rms_deviation",
        }
    }

    pub fn from_value(value: &str) -> Result<Self> {
        match value {
            "max_deviation" => Ok(Self::MaxDeviation),
            "rms_deviation" => Ok(Self::RmsDeviation),
            _ => bail!("tolerance_metric must be 'max_deviation' or 'rms_deviation', got {}", bspline_varpro::numfmt::quoted(value)),
        }
    }

    /// The name as the window shows it.
    pub fn name(self) -> &'static str {
        match self {
            Self::MaxDeviation => "Max deviation",
            Self::RmsDeviation => "RMS deviation",
        }
    }

    /// The metric of orthogonal deviations.
    pub fn of(self, deviations: &[f64]) -> f64 {
        match self {
            Self::MaxDeviation => deviations.iter().copied().fold(0.0, f64::max),
            Self::RmsDeviation if deviations.is_empty() => 0.0,
            Self::RmsDeviation => (deviations.iter().map(|d| d * d).sum::<f64>() / deviations.len() as f64).sqrt(),
        }
    }
}

/// Fitting method settings.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationConfig {
    pub optimizer: OptimizerType,
    /// Fit the junction positions (else they stay at the section split points).
    pub optimize_le_position: bool,
    /// Bending-energy weight of VarPro fits with a fixed count.
    pub smoothing_weight: f64,
    /// Direct LSQ start parameters.
    pub parameterization: Parameterization,
    /// The tolerance of the metric `tolerance_metric`; `None` fits a fixed
    /// count.
    pub tolerance: Option<f64>,
    /// What the tolerance limits.
    pub tolerance_metric: ToleranceMetric,
    /// Uniform knots, or free knots whose positions are unknowns of the fit.
    pub knot_placement: KnotPlacement,
    /// Start from the fewest control points and add them where the fit is worst.
    pub grow_control_points: bool,
    /// Smallest spacing of free knots, as a fraction of a section (0 to 0.5).
    pub min_knot_gap: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        OptimizationConfig {
            optimizer: OptimizerType::VarPro,
            optimize_le_position: false,
            smoothing_weight: 1e-3,
            parameterization: Parameterization::Centripetal,
            tolerance: None,
            tolerance_metric: ToleranceMetric::MaxDeviation,
            knot_placement: KnotPlacement::Uniform,
            grow_control_points: false,
            min_knot_gap: bspline_varpro::MIN_KNOT_GAP,
        }
    }
}

impl OptimizationConfig {
    pub fn validated(self) -> Result<Self> {
        if let Some(t) = self.tolerance.filter(|t| t.is_nan() || *t <= 0.0) {
            bail!("tolerance must be positive, got {}", bspline_varpro::numfmt::shortest(t));
        }
        if !(0.0..=0.5).contains(&self.min_knot_gap) {
            bail!("min_knot_gap must be between 0 and 0.5, got {}", bspline_varpro::numfmt::shortest(self.min_knot_gap));
        }
        Ok(self)
    }
}

/// A fitted spline chain as the application keeps it.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    pub section_control_points: Vec<Vec<Vec2>>,
    pub degree: usize,
    pub final_error: f64,
    pub num_iterations: usize,
    pub num_evaluations: usize,
    pub elapsed_seconds: f64,
    pub converged: bool,
    pub cancelled: bool,
    /// Knot vectors; open uniform per section when `None`.
    pub section_knots: Option<Vec<Vec<f64>>>,
    pub report: Option<Report>,
    pub rms_error: Option<f64>,
}

impl FitResult {
    /// Knot vector of section `i`.
    pub fn knots(&self, i: usize) -> Vec<f64> {
        match &self.section_knots {
            Some(k) => k[i].clone(),
            None => bspline_varpro::open_uniform_knots(self.degree, self.section_control_points[i].len())
                .expect("a fitted section has more control points than its degree"),
        }
    }

    pub fn all_knots(&self) -> Vec<Vec<f64>> {
        (0..self.section_control_points.len()).map(|i| self.knots(i)).collect()
    }

    /// One curve per section.
    pub fn curves(&self) -> Vec<BSpline> {
        self.section_control_points
            .iter()
            .enumerate()
            .map(|(i, p)| BSpline { degree: self.degree, knots: self.knots(i), control_points: p.clone() })
            .collect()
    }

    /// The RMS orthogonal deviation of the report (runs without one: the
    /// RMS error, else the final error), the measure runs are compared by.
    pub fn rms_deviation(&self) -> f64 {
        self.report.as_ref().map(|r| r.rms_deviation).or(self.rms_error).unwrap_or(self.final_error)
    }

    /// Control points per section, as the sidebar sets them; when the
    /// sections differ (tolerance runs of older projects), of the chain.
    pub fn control_points_per_section(&self) -> usize {
        match self.section_control_points.split_first() {
            Some((first, rest)) if rest.iter().all(|s| s.len() == first.len()) => first.len(),
            _ => self.num_control_points(),
        }
    }

    /// Control points of the chain, each junction point once.
    pub fn num_control_points(&self) -> usize {
        let sections = self.section_control_points.len();
        self.section_control_points.iter().map(Vec::len).sum::<usize>() + 1 - sections.max(1)
    }

    pub fn section_counts(&self) -> Vec<usize> {
        self.section_control_points.iter().map(Vec::len).collect()
    }
}

/// One record of a fit's error history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterationRecord {
    pub iteration: usize,
    pub error: f64,
    pub elapsed_seconds: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_config_rules() {
        let c = SplineConfig { continuity: ContinuityMode::G2, geometric_continuity: false, degree: 1, ..Default::default() };
        assert_eq!(c.validated().unwrap_err().0, "G2 continuity requires degree >= 2, got 1");
        let c = SplineConfig { num_sections: 3, junction_continuity: vec![" g2".into(), "c1".into()], ..Default::default() };
        let c = c.validated().unwrap();
        assert_eq!(c.junction_specs(), vec!["G2", "C1"]);
        assert_eq!(c.minimum_control_points(), 6);
        let bad = SplineConfig { num_sections: 3, junction_continuity: vec!["G2".into()], ..Default::default() };
        assert_eq!(bad.validated().unwrap_err().0, "3 sections have 2 junction(s), but 1 continuity entries were given");
        assert_eq!(minimum_control_points(&[], 1), 2);
        assert_eq!(minimum_control_points(&[3, 3], 1), 8);
    }
}
