//! Python bindings of `bspline-varpro`: read an airfoil, fit it to a
//! tolerance or with fixed control-point counts, measure the fit and write
//! it to STEP, IGES or DXF, from scripts and notebooks.
//!
//! Points go in as any sequence of `(x, y)` rows (a NumPy array shaped
//! `(n, 2)` included) and come back as NumPy arrays (as lists without NumPy).

use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use varpro::{
    airfoil, chain, diagnostics, export, ChainOptions, Continuity, FitOptions, Junction, KnotPlacement, Silent,
    ToleranceOptions, Vec2,
};

fn value_error(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn to_points(rows: Vec<[f64; 2]>) -> Vec<Vec2> {
    rows.into_iter().map(|[x, y]| Vec2::new(x, y)).collect()
}

/// A NumPy array of `rows` (a list when NumPy is not installed).
fn array<'py, T: IntoPyObject<'py> + Clone>(py: Python<'py>, rows: &[T]) -> PyResult<Bound<'py, PyAny>> {
    let list = PyList::new(py, rows.iter().cloned())?.into_any();
    match py.import("numpy") {
        Ok(numpy) => numpy.call_method1("asarray", (list,)),
        Err(_) => Ok(list),
    }
}

fn point_array<'py>(py: Python<'py>, points: &[Vec2]) -> PyResult<Bound<'py, PyAny>> {
    array(py, &points.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>())
}

/// Continuity at the junctions: `"G2"` for all, or one entry per junction.
#[derive(FromPyObject)]
enum ContinuityArg {
    All(String),
    Each(Vec<String>),
}

impl ContinuityArg {
    fn parse(self) -> PyResult<Continuity> {
        let junction = |s: &str| s.trim().parse::<Junction>().map_err(value_error);
        Ok(match self {
            ContinuityArg::All(s) if s.contains(',') => {
                Continuity::Each(s.split(',').map(junction).collect::<PyResult<_>>()?)
            }
            ContinuityArg::All(s) => Continuity::All(junction(&s)?),
            ContinuityArg::Each(list) => Continuity::Each(list.iter().map(|s| junction(s)).collect::<PyResult<_>>()?),
        })
    }
}

/// Control points per section: one count for all, or one per section.
#[derive(FromPyObject)]
enum CountsArg {
    All(usize),
    Each(Vec<usize>),
}

fn placement(knots: &str) -> PyResult<KnotPlacement> {
    match knots {
        "uniform" => Ok(KnotPlacement::Uniform),
        "free" => Ok(KnotPlacement::Free),
        _ => Err(value_error(format!("knots must be 'uniform' or 'free', got {knots:?}."))),
    }
}

fn to_sections(sections: Vec<Vec<[f64; 2]>>) -> Vec<Vec<Vec2>> {
    sections.into_iter().map(to_points).collect()
}

/// A fitted chain of B-spline sections.
#[pyclass(name = "Fit", module = "bspline_varpro", frozen)]
struct Fit {
    inner: varpro::FitResult,
}

impl Fit {
    fn curves(&self) -> Vec<varpro::BSpline> {
        self.inner.curves()
    }
}

#[pymethods]
impl Fit {
    /// Control points of every section, each shaped `(n, 2)`.
    #[getter]
    fn control_points<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.inner.control_points.iter().map(|p| point_array(py, p)).collect()
    }

    /// Clamped knot vector of every section.
    #[getter]
    fn knots<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.inner.knots.iter().map(|k| array(py, k)).collect()
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree
    }

    /// Chain parameter of every data point, in `[0, sections]`.
    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        array(py, &self.inner.parameters)
    }

    /// RMS distance of the data to their curve points.
    #[getter]
    fn rms_error(&self) -> f64 {
        self.inner.rms_error
    }

    /// Largest orthogonal distance of a point to the chain, where measured.
    #[getter]
    fn max_deviation(&self) -> Option<f64> {
        self.inner.max_deviation
    }

    /// Whether the solver converged (or the tolerance search met its tolerance).
    #[getter]
    fn success(&self) -> bool {
        self.inner.success
    }

    #[getter]
    fn message(&self) -> &str {
        &self.inner.message
    }

    /// Residual evaluations.
    #[getter]
    fn nfev(&self) -> usize {
        self.inner.nfev
    }

    /// Control points per section.
    #[getter]
    fn num_control_points(&self) -> Vec<usize> {
        self.inner.num_control_points()
    }

    /// Points of the chain at chain parameters `u` in `[0, sections]`.
    fn evaluate<'py>(&self, py: Python<'py>, u: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
        let curves = self.curves();
        let points: Vec<Vec2> = u.iter().map(|&u| chain::evaluate_chain(&curves, u, 0)).collect();
        point_array(py, &points)
    }

    /// Selig-ordered points on the fitted airfoil, `n_per_side` per side
    /// with cosine spacing (the leading-edge point shared).
    #[pyo3(signature = (n_per_side = 100))]
    fn resample<'py>(&self, py: Python<'py>, n_per_side: usize) -> PyResult<Bound<'py, PyAny>> {
        point_array(py, &airfoil::resample(&self.curves(), n_per_side).map_err(value_error)?)
    }

    /// Orthogonal distance of every point to the nearest section.
    fn deviations<'py>(&self, py: Python<'py>, points: Vec<[f64; 2]>) -> PyResult<Bound<'py, PyAny>> {
        array(py, &diagnostics::distance_to_curves(&to_points(points), &self.curves()))
    }

    /// Quality measures of the fit: curvature fairness, the junctions, the
    /// leading-edge radius, maximum thickness and camber (for an airfoil,
    /// in chords, at chord positions), and with `points` the maximum and RMS
    /// deviation of those points.
    #[pyo3(signature = (points = None))]
    fn measure<'py>(&self, py: Python<'py>, points: Option<Vec<[f64; 2]>>) -> PyResult<Bound<'py, PyDict>> {
        let curves = self.curves();
        let out = PyDict::new(py);
        let fairness = diagnostics::fairness(&curves);
        out.set_item("max_curvature", fairness.max_curvature)?;
        out.set_item("curvature_extrema", fairness.curvature_extrema)?;
        out.set_item("roughness", fairness.roughness)?;
        out.set_item("max_curvature_slope", fairness.max_curvature_slope)?;
        let junctions = PyList::empty(py);
        for j in diagnostics::junctions(&curves) {
            let item = PyDict::new(py);
            item.set_item("gap", j.gap)?;
            item.set_item("tangent_angle", j.tangent_angle)?;
            item.set_item("curvature_jump", j.curvature_jump)?;
            item.set_item("curvature_slope_jump", j.curvature_slope_jump)?;
            junctions.append(item)?;
        }
        out.set_item("junctions", junctions)?;
        if curves.len() >= 2 {
            out.set_item("leading_edge_radius", airfoil::leading_edge_radius(&curves))?;
        }
        if let Ok(shape) = airfoil::thickness_and_camber(&curves) {
            out.set_item("chord", shape.chord)?;
            out.set_item("max_thickness", shape.max_thickness)?;
            out.set_item("max_thickness_x", shape.max_thickness_x)?;
            out.set_item("max_camber", shape.max_camber)?;
            out.set_item("max_camber_x", shape.max_camber_x)?;
        }
        if let Some(points) = points {
            let d = diagnostics::distance_to_curves(&to_points(points), &curves);
            if !d.is_empty() {
                out.set_item("max_deviation", d.iter().cloned().fold(0.0, f64::max))?;
                out.set_item("rms_deviation", (d.iter().map(|v| v * v).sum::<f64>() / d.len() as f64).sqrt())?;
            }
        }
        Ok(out)
    }

    /// Writes the curves to STEP (`.step`, `.stp`), IGES (`.igs`, `.iges`)
    /// or DXF (`.dxf`), by the path's suffix. `scale` multiplies the
    /// coordinates (the chord in `units`); `merge` writes one curve instead
    /// of one per section; `close_trailing_edge` adds a straight segment
    /// across a blunt trailing edge. Returns the path written.
    #[pyo3(signature = (path, scale = 1.0, units = "mm", merge = false, close_trailing_edge = false))]
    fn export(&self, path: &str, scale: f64, units: &str, merge: bool, close_trailing_edge: bool) -> PyResult<String> {
        let units: export::Units = units.parse().map_err(value_error)?;
        let mut curves = self.curves();
        if merge {
            curves = vec![export::merge_curves(&curves).map_err(value_error)?];
        }
        if close_trailing_edge {
            curves.extend(airfoil::trailing_edge_segment(&curves));
        }
        let written = export::write(Path::new(path), &curves, scale, units).map_err(value_error)?;
        Ok(written.to_string_lossy().into_owned())
    }

    fn __repr__(&self) -> String {
        let deviation = self.inner.max_deviation.map(|d| format!(", max_deviation={d:.3e}")).unwrap_or_default();
        format!(
            "Fit(degree={}, control_points={:?}, rms_error={:.3e}{deviation}, success={})",
            self.inner.degree,
            self.inner.num_control_points(),
            self.inner.rms_error,
            if self.inner.success { "True" } else { "False" },
        )
    }
}

/// Reads a Selig or Lednicer airfoil file: `(name, points, precision)`,
/// the points in Selig order at unit chord, keeping the file's chord line,
/// and the decimal places of the coordinates.
#[pyfunction]
fn read_airfoil<'py>(py: Python<'py>, path: &str) -> PyResult<(Option<String>, Bound<'py, PyAny>, i32)> {
    let parsed = app::io::parse_airfoil_file(Path::new(path))
        .map_err(value_error)?
        .ok_or_else(|| value_error(format!("No coordinates in {path}.")))?;
    Ok((parsed.name, point_array(py, &parsed.coords)?, parsed.precision))
}

/// Selig-ordered points of a NACA 4-digit section, `n_per_side` per side
/// with cosine spacing; `closed_te` gives a sharp trailing edge.
#[pyfunction]
#[pyo3(signature = (code, n_per_side = 81, closed_te = true))]
fn naca4<'py>(py: Python<'py>, code: &str, n_per_side: usize, closed_te: bool) -> PyResult<Bound<'py, PyAny>> {
    let naca = airfoil::Naca4::new(code, closed_te).map_err(value_error)?;
    point_array(py, &naca.points(n_per_side))
}

/// Splits Selig-ordered points into `count` sections that share their
/// junction points: two split at the leading edge. With `corners`, also at
/// the clear corners of the data (`corners` says where they are), as the
/// application cuts an airfoil: one more section for every corner away
/// from those junctions, to be joined there with G0.
#[pyfunction]
#[pyo3(signature = (points, count = 2, corners = false))]
fn split_sections<'py>(py: Python<'py>, points: Vec<[f64; 2]>, count: usize, corners: bool) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let split = app::geometry::Split { sections: count, corners };
    let (sections, _, _) = app::geometry::cut_into_sections(&to_points(points), split).map_err(value_error)?;
    sections.iter().map(|s| point_array(py, s)).collect()
}

/// The corners of ordered points, read from the points alone: a list of
/// `(index, turn, clear)`, the turn in degrees. A corner is a point where
/// the points turn at least 20 degrees and six times as much as at either
/// neighbour, beyond what their rounding can make of it; `clear` is false
/// where the points are too far apart to tell it from a tight bend. Any
/// ordered points: no chord line is assumed.
#[pyfunction]
fn corners(points: Vec<[f64; 2]>) -> Vec<(usize, f64, bool)> {
    varpro::corners::corners(&to_points(points)).iter().map(|c| (c.index, c.turn, !c.sparse)).collect()
}

/// Ordered points cut into sections at their clear corners (one section
/// without any). The sections share the corner points: fit them with
/// `continuity="G0"`.
#[pyfunction]
fn split_at_corners<'py>(py: Python<'py>, points: Vec<[f64; 2]>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let (sections, _) = varpro::corners::split_at_corners(&to_points(points));
    sections.iter().map(|s| point_array(py, s)).collect()
}

/// `count` points on the smoothest curve through `points` (a natural cubic
/// spline over their chord lengths; the straight line through two), the
/// given points among them. A section with fewer points than control
/// points is fitted to these, so that its points decide every control
/// point (`fit_airfoil` does so for sections that end at a corner).
#[pyfunction]
fn smoothest_through<'py>(py: Python<'py>, points: Vec<[f64; 2]>, count: usize) -> PyResult<Bound<'py, PyAny>> {
    point_array(py, &varpro::corners::smoothest_through(&to_points(points), count))
}

/// The tolerance the data support: their rounding error (from `decimals`,
/// or as read from the values), or their noise level for scanned data.
#[pyfunction]
#[pyo3(signature = (points, decimals = None))]
fn default_tolerance(points: Vec<[f64; 2]>, decimals: Option<i32>) -> f64 {
    airfoil::default_tolerance(&to_points(points), decimals)
}

fn chain_options(
    continuity: ContinuityArg,
    smoothing: f64,
    fix_junctions: bool,
    fix_ends: bool,
    defaults: ChainOptions,
) -> PyResult<ChainOptions> {
    Ok(ChainOptions { continuity: continuity.parse()?, smoothing, fix_junctions, fix_ends, ..defaults })
}

/// Least-squares fit with fixed control-point counts: `control_points` is
/// one count for every section or one per section. `continuity` is one
/// spec for all junctions (`"G2"`, `"C1"`) or one per junction (a list or
/// `"G2, C1"`). `knots` is `"uniform"` or `"free"` (knot positions become
/// unknowns). `grow` starts from the fewest control points and adds knots
/// where the fit is worst.
#[pyfunction]
#[pyo3(signature = (
    sections, control_points, degree = 5, continuity = ContinuityArg::All("G2".into()),
    smoothing = 1e-3, knots = "uniform", grow = false, fix_junctions = true, fix_ends = true,
    max_nfev = None,
))]
#[allow(clippy::too_many_arguments)]
fn fit(
    py: Python<'_>,
    sections: Vec<Vec<[f64; 2]>>,
    control_points: CountsArg,
    degree: usize,
    continuity: ContinuityArg,
    smoothing: f64,
    knots: &str,
    grow: bool,
    fix_junctions: bool,
    fix_ends: bool,
    max_nfev: Option<usize>,
) -> PyResult<Fit> {
    let sections = to_sections(sections);
    let counts = match control_points {
        CountsArg::All(n) => vec![n; sections.len()],
        CountsArg::Each(list) => list,
    };
    let options = FitOptions {
        chain: chain_options(continuity, smoothing, fix_junctions, fix_ends, ChainOptions::default())?,
        knot_placement: placement(knots)?,
        grow,
        max_nfev,
        ..FitOptions::default()
    };
    let result = py.detach(|| varpro::fit(&sections, &counts, degree, &options, &mut Silent));
    Ok(Fit { inner: result.map_err(value_error)? })
}

/// The fewest control points whose curve keeps every point within
/// `tolerance` (maximum orthogonal deviation). Arguments as in `fit`;
/// `initial_control_points` starts the search from that count per section,
/// `max_control_points` limits it.
#[pyfunction]
#[pyo3(signature = (
    sections, tolerance, degree = 5, continuity = ContinuityArg::All("G2".into()),
    smoothing = 1e-7, knots = "free", initial_control_points = None, max_control_points = None,
    fix_junctions = true, fix_ends = true,
))]
#[allow(clippy::too_many_arguments)]
fn fit_to_tolerance(
    py: Python<'_>,
    sections: Vec<Vec<[f64; 2]>>,
    tolerance: f64,
    degree: usize,
    continuity: ContinuityArg,
    smoothing: f64,
    knots: &str,
    initial_control_points: Option<usize>,
    max_control_points: Option<usize>,
    fix_junctions: bool,
    fix_ends: bool,
) -> PyResult<Fit> {
    let sections = to_sections(sections);
    let defaults = ToleranceOptions::default();
    let options = ToleranceOptions {
        chain: chain_options(continuity, smoothing, fix_junctions, fix_ends, defaults.chain.clone())?,
        knot_placement: placement(knots)?,
        initial_control_points,
        max_control_points,
        ..defaults
    };
    let result = py.detach(|| varpro::fit_to_tolerance(&sections, tolerance, degree, &options, &mut Silent));
    Ok(Fit { inner: result.map_err(value_error)? })
}

impl ContinuityArg {
    /// One entry per junction of `sections` sections.
    fn per_junction(self, sections: usize) -> Vec<String> {
        match self {
            ContinuityArg::All(s) if s.contains(',') => s.split(',').map(|j| j.trim().to_string()).collect(),
            ContinuityArg::All(s) => vec![s; sections.saturating_sub(1)],
            ContinuityArg::Each(list) => list,
        }
    }
}

/// Fits Selig-ordered airfoil points to a tolerance as the application
/// does, with its defaults: split into `sections` (two split at the leading
/// edge), then fixed-count fits with the same count in every section, from
/// `start_control_points` one fewer while the tolerance holds (or one more
/// until it does). Without a `tolerance`, the one the data support
/// (`default_tolerance`, with the data's `decimals` when given).
///
/// With `corners` (the default) the outline is also cut at the clear
/// corners of the data and joined there with G0, whatever `continuity`
/// says of the other junctions (one entry per junction of `sections`): a
/// sharp leading edge is a G0 junction, a ridge behind a round nose one
/// more section. A section that ends at a corner takes of the count what
/// its points determine.
#[pyfunction]
#[pyo3(signature = (
    points, tolerance = None, degree = 5, sections = 2, continuity = ContinuityArg::All("G3".into()),
    smoothing = 1e-3, knots = "uniform", start_control_points = 25, decimals = None, corners = true,
))]
#[allow(clippy::too_many_arguments)]
fn fit_airfoil(
    py: Python<'_>,
    points: Vec<[f64; 2]>,
    tolerance: Option<f64>,
    degree: usize,
    sections: usize,
    continuity: ContinuityArg,
    smoothing: f64,
    knots: &str,
    start_control_points: usize,
    decimals: Option<i32>,
    corners: bool,
) -> PyResult<Fit> {
    let points = to_points(points);
    let tolerance = tolerance.unwrap_or_else(|| airfoil::default_tolerance(&points, decimals));
    let config = app::models::SplineConfig {
        degree,
        num_control_points: start_control_points,
        num_sections: sections,
        junction_continuity: continuity.per_junction(sections),
        split_at_corners: corners,
        ..app::models::SplineConfig::default()
    }
    .validated()
    .map_err(value_error)?;
    let data = app::geometry::AirfoilData::new(points.clone(), config.split(), "", "", decimals.unwrap_or(0))
        .map_err(value_error)?;
    let options = app::models::OptimizationConfig {
        tolerance: Some(tolerance),
        smoothing_weight: smoothing,
        knot_placement: placement(knots)?,
        ..app::models::OptimizationConfig::default()
    };
    let raw = py
        .detach(|| app::optimize::run_optimization(&config, &options, &data, &mut app::optimize::Unmonitored))
        .map_err(value_error)?;
    let knots = match raw.section_knots {
        Some(knots) => knots,
        None => raw
            .section_control_points
            .iter()
            .map(|c| varpro::open_uniform_knots(degree, c.len()))
            .collect::<varpro::Result<_>>()
            .map_err(value_error)?,
    };
    let mut inner = varpro::FitResult {
        control_points: raw.section_control_points,
        parameters: vec![],
        knots,
        degree,
        rms_error: raw.rms_error,
        nfev: raw.evaluations,
        success: raw.success,
        message: raw.message,
        stopped: false,
        theta: vec![],
        max_deviation: None,
    };
    let deviations = diagnostics::distance_to_curves(&points, &inner.curves());
    inner.max_deviation = Some(deviations.iter().cloned().fold(0.0, f64::max));
    Ok(Fit { inner })
}

#[pymodule]
fn bspline_varpro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", varpro::VERSION)?;
    m.add_class::<Fit>()?;
    m.add_function(wrap_pyfunction!(read_airfoil, m)?)?;
    m.add_function(wrap_pyfunction!(naca4, m)?)?;
    m.add_function(wrap_pyfunction!(split_sections, m)?)?;
    m.add_function(wrap_pyfunction!(corners, m)?)?;
    m.add_function(wrap_pyfunction!(split_at_corners, m)?)?;
    m.add_function(wrap_pyfunction!(smoothest_through, m)?)?;
    m.add_function(wrap_pyfunction!(default_tolerance, m)?)?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(fit_to_tolerance, m)?)?;
    m.add_function(wrap_pyfunction!(fit_airfoil, m)?)?;
    Ok(())
}
