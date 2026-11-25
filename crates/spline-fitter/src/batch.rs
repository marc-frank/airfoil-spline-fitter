//! Batch conversion: fitting every airfoil file of a folder.

use std::path::{Path, PathBuf};

use bspline_varpro::numfmt::{exponent, general};
use bspline_varpro::KnotPlacement;

use crate::analysis::default_tolerance;
use crate::geometry::AirfoilData;
use crate::io::{decode, parse_airfoil_file, parse_airfoil_text, spline_data_text, ParsedAirfoil, SplineFile};
use crate::models::{OptimizationConfig, OptimizerType, Parameterization, SplineConfig};
use crate::optimize::{run_optimization, Monitor, RawResult};
use crate::{bail, Error, Result};

/// The settings of a new window, against which output names are made: the
/// defaults of the models.
pub fn reference_settings() -> (SplineConfig, OptimizationConfig) {
    (SplineConfig::default(), OptimizationConfig::default())
}

/// A number for a file name, the same in every locale: small ones as
/// `1e-4` or `2.5e-5`, others with three significant digits.
pub fn num(v: f64) -> String {
    if v != 0.0 && v.abs() < 1e-2 {
        let text = exponent(v, 2);
        let (mantissa, exp) = text.split_once('e').unwrap();
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        format!("{mantissa}e{}", exp.parse::<i32>().unwrap())
    } else {
        general(v, 3)
    }
}

/// The batch output file of `stem` fitted with these settings: the degree
/// always, every other setting when it differs from a new window's, so two
/// different settings never write the same file
/// (`naca0012.deg5.n30.s3.G2-C1.tol1e-4.spline.dat`).
pub fn output_name(stem: &str, config: &SplineConfig, opt: &OptimizationConfig, auto_tolerance: bool) -> String {
    let (reference, reference_opt) = reference_settings();
    let mut parts = vec![stem.to_string(), format!("deg{}", config.degree)];
    let count_used = opt.tolerance.is_none() || !opt.grow_control_points;
    if count_used && config.num_control_points != reference.num_control_points {
        parts.push(format!("n{}", config.num_control_points));
    }
    if config.num_sections != reference.num_sections {
        parts.push(format!("s{}", config.num_sections));
    }
    let specs = config.junction_specs();
    if !specs.is_empty() && specs.iter().any(|s| s != "G3") {
        let all_equal = specs.iter().all(|s| *s == specs[0]);
        parts.push(if all_equal { specs[0].clone() } else { specs.join("-") });
    }
    if opt.optimizer != reference_opt.optimizer {
        parts.push(match opt.optimizer {
            OptimizerType::DirectLsq => "lsq".into(),
            other => other.value().into(),
        });
        if opt.parameterization != Parameterization::Centripetal {
            parts.push(opt.parameterization.value().replace('_', "-"));
        }
    }
    if let Some(t) = opt.tolerance {
        parts.push(if auto_tolerance { "tol-auto".into() } else { format!("tol{}", num(t)) });
        if opt.tolerance_metric == crate::models::ToleranceMetric::RmsDeviation {
            parts.push("rms".into());
        }
    }
    if opt.knot_placement != reference_opt.knot_placement {
        parts.push(format!("knots-{}", crate::models::placement_value(opt.knot_placement)));
    }
    if opt.knot_placement == KnotPlacement::Free && opt.min_knot_gap != reference_opt.min_knot_gap {
        parts.push(format!("gap{}", num(opt.min_knot_gap)));
    }
    if opt.grow_control_points {
        parts.push("grow".into());
    }
    if opt.smoothing_weight != reference_opt.smoothing_weight {
        parts.push(format!("smooth{}", num(opt.smoothing_weight)));
    }
    if opt.optimize_le_position {
        parts.push("free-le".into());
    }
    parts.join(".") + ".spline.dat"
}

/// The airfoil files of a folder (not recursive), sorted by name: names
/// ending in `.dat` that are not spline files.
pub fn airfoil_files(folder: &Path) -> Result<Vec<String>> {
    let entries = std::fs::read_dir(folder).map_err(|e| Error::new(e.to_string()))?;
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| is_airfoil_file(n))
        .collect();
    names.sort();
    Ok(names)
}

/// Whether a file name is a batch's airfoil: `.dat`, not a spline file.
pub fn is_airfoil_file(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.ends_with(".dat") && !lower.contains(".spline.")
}

/// What happened to one file of a batch.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The target file exists already.
    Skipped(PathBuf),
    /// Fitted and written.
    Done(RawResult),
    /// The monitor asked to stop.
    Cancelled,
}

/// A file read for a batch, ready to fit.
pub struct BatchItem {
    pub airfoil: AirfoilData,
    pub output: PathBuf,
    /// Mistakes the reader removed from the file (the corrected points are
    /// fitted, as the window's default).
    pub corrections: usize,
}

/// Reads one source file and names its output in `target`.
pub fn prepare(source: &Path, target: &Path, config: &SplineConfig, opt: &OptimizationConfig, auto_tolerance: bool) -> Result<BatchItem> {
    let file_name = source.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    prepare_parsed(parse_airfoil_file(source)?, &file_name, target, config, opt, auto_tolerance)
}

/// [`prepare`] for a file already read (the browser reads a batch's files
/// first): its name and contents.
pub fn prepare_bytes(
    name: &str,
    bytes: &[u8],
    target: &Path,
    config: &SplineConfig,
    opt: &OptimizationConfig,
    auto_tolerance: bool,
) -> Result<BatchItem> {
    let file_name = Path::new(name).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    prepare_parsed(parse_airfoil_text(&decode(bytes), Some(&file_name)), &file_name, target, config, opt, auto_tolerance)
}

fn prepare_parsed(
    parsed: Option<ParsedAirfoil>,
    file_name: &str,
    target: &Path,
    config: &SplineConfig,
    opt: &OptimizationConfig,
    auto_tolerance: bool,
) -> Result<BatchItem> {
    let Some(parsed) = parsed else { bail!("Failed to parse") };
    let file_name = file_name.to_string();
    let name = parsed.name.unwrap_or_else(|| file_name.clone());
    let corrections = parsed.cleanup.findings.len();
    let airfoil = AirfoilData::new(parsed.coords, config.num_sections, &file_name, &name, parsed.precision)?;
    let stem = if file_name.is_empty() { &name } else { &file_name };
    let output = target.join(output_name(stem, config, opt, auto_tolerance));
    Ok(BatchItem { airfoil, output, corrections })
}

/// Fits one prepared file and writes its spline.
pub fn convert(
    item: &BatchItem,
    config: &SplineConfig,
    opt: &OptimizationConfig,
    auto_tolerance: bool,
    monitor: &mut dyn Monitor,
    cancelled: impl Fn() -> bool,
) -> Result<Outcome> {
    if item.output.exists() {
        return Ok(Outcome::Skipped(item.output.clone()));
    }
    let mut opt = opt.clone();
    if opt.tolerance.is_some() && auto_tolerance {
        opt.tolerance = Some(default_tolerance(&item.airfoil.coordinates, item.airfoil.precision));
    }
    let result = run_optimization(config, &opt, &item.airfoil, monitor)?;
    if cancelled() {
        return Ok(Outcome::Cancelled);
    }
    if !result.success {
        bail!("Optimizer did not converge: {}", result.message);
    }
    write_result(item, config, &result)?;
    Ok(Outcome::Done(result))
}

/// Writes the fitted spline of a batch item (with 12 decimals).
pub fn write_result(item: &BatchItem, config: &SplineConfig, result: &RawResult) -> Result<()> {
    let text = result_text(item, config, result)?;
    std::fs::write(&item.output, text).map_err(|e| Error::new(e.to_string()))
}

/// The text [`write_result`] writes.
pub fn result_text(item: &BatchItem, config: &SplineConfig, result: &RawResult) -> Result<String> {
    let a = &item.airfoil;
    let spline = SplineFile {
        name: if a.name.is_empty() { a.filename.clone() } else { a.name.clone() },
        degree: Some(config.degree),
        control_points: result.section_control_points.clone(),
        knots: result.section_knots.clone(),
        continuity: Some(config.junction_specs()),
    };
    spline_data_text(&spline, Some(12))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        let cases = [
            (1e-4, "1e-4"),
            (2.5e-5, "2.5e-5"),
            (0.001, "1e-3"),
            (0.0123, "0.0123"),
            (0.5, "0.5"),
            (1e-6, "1e-6"),
            (3.33333e-7, "3.33e-7"),
        ];
        for (v, text) in cases {
            assert_eq!(num(v), text);
        }
    }

    #[test]
    fn output_names() {
        let (spline, opt) = reference_settings();
        assert_eq!(output_name("naca0012", &spline, &opt, false), "naca0012.deg5.spline.dat");
        let config = SplineConfig {
            num_control_points: 30,
            num_sections: 3,
            junction_continuity: vec!["G2".into(), "C1".into()],
            ..SplineConfig::default()
        };
        let o = OptimizationConfig {
            tolerance: Some(1e-4),
            knot_placement: KnotPlacement::Uniform,
            grow_control_points: false,
            ..Default::default()
        };
        assert_eq!(output_name("naca0012", &config, &o, false), "naca0012.deg5.n30.s3.G2-C1.tol1e-4.spline.dat");
        let o = OptimizationConfig {
            tolerance: Some(1e-5),
            knot_placement: KnotPlacement::Free,
            grow_control_points: true,
            ..Default::default()
        };
        assert_eq!(output_name("naca0012", &spline, &o, true), "naca0012.deg5.tol-auto.knots-free.grow.spline.dat");
        let gap = OptimizationConfig { knot_placement: KnotPlacement::Free, min_knot_gap: 0.02, ..Default::default() };
        assert_eq!(output_name("x", &spline, &gap, false), "x.deg5.knots-free.gap0.02.spline.dat");
        let config = SplineConfig {
            degree: 3,
            continuity: crate::models::ContinuityMode::G2,
            geometric_continuity: false,
            ..SplineConfig::default()
        };
        let o = OptimizationConfig {
            optimizer: OptimizerType::DirectLsq,
            parameterization: Parameterization::ArcLength,
            knot_placement: KnotPlacement::Uniform,
            grow_control_points: false,
            smoothing_weight: 0.01,
            optimize_le_position: true,
            ..Default::default()
        };
        assert_eq!(output_name("x", &config, &o, false), "x.deg3.C2.lsq.arc-length.smooth0.01.free-le.spline.dat");
        let one = SplineConfig { num_sections: 1, ..spline };
        assert_eq!(output_name("x", &one, &opt, false), "x.deg5.s1.spline.dat");
    }
}
