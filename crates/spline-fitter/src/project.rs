//! Project files (`.asfproj`): an airfoil and the fit runs made on it.
//!
//! UTF-8 JSON with two-space indentation, floats in shortest round-trip
//! form and non-finite numbers as `Infinity`/`NaN`, so files stay
//! interchangeable with earlier versions of the application.

use std::path::Path;

use bspline_varpro::numfmt::shortest;
use bspline_varpro::KnotPlacement;

use crate::analysis::{FairnessReport, JunctionReport, Report, ShapeReport};
use crate::geometry::AirfoilData;
use crate::models::{
    placement_from_value, placement_value, ContinuityMode, FitResult, IterationRecord, OptimizationConfig, OptimizerType, Parameterization,
    SplineConfig,
};
use crate::{bail, Error, Result, Vec2};

/// 1.1: a tolerance run's `smoothing_weight` is the weight it was fitted
/// with; before, tolerance fits ignored it and used
/// [`OLD_TOLERANCE_SMOOTHING`].
/// 1.2: tolerance searches (`tolerance_searches`), whose fits are
/// fixed-count runs with a `search` key; readers of 1.0 see those runs as
/// plain fixed-count runs.
pub const FORMAT_VERSION: &str = "1.2.0";
/// The smoothing weight of tolerance fits in projects before format 1.1.
pub const OLD_TOLERANCE_SMOOTHING: f64 = 1e-7;

/// Where and when a project was written.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectMetadata {
    pub format_version: String,
    pub created: String,
    pub modified: String,
    pub app_version: String,
    pub platform: String,
}

impl ProjectMetadata {
    /// Metadata of a project created now (or at `created`).
    pub fn new(created: Option<String>) -> Self {
        let now = now_iso();
        ProjectMetadata {
            format_version: FORMAT_VERSION.into(),
            created: created.unwrap_or_else(|| now.clone()),
            modified: now,
            app_version: crate::VERSION.into(),
            platform: platform(),
        }
    }
}

/// One fit with its settings.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationRun {
    pub run_id: String,
    /// 1, 2, ... in the order runs were added; 0 until numbered.
    pub number: usize,
    pub timestamp: String,
    pub spline_config: SplineConfig,
    pub optimization_config: OptimizationConfig,
    pub result: FitResult,
    pub loss_history: Vec<IterationRecord>,
    /// The tolerance search this run is a fit of (its `search_id`).
    pub search: Option<String>,
}

/// A tolerance search: its settings (the start count, the tolerance) and
/// how it ended. Its fits are runs of their own, fixed-count runs that name
/// it, each exactly the run its count makes on its own.
#[derive(Debug, Clone, PartialEq)]
pub struct ToleranceSearch {
    pub search_id: String,
    /// 1, 2, ... in the order searches were added.
    pub number: usize,
    pub timestamp: String,
    /// The sidebar's settings: `num_control_points` is the start count.
    pub spline_config: SplineConfig,
    /// With the tolerance.
    pub optimization_config: OptimizationConfig,
    /// The run it ended on.
    pub result_run: String,
    /// Control points per section of that run.
    pub control_points: usize,
    pub max_deviation: f64,
    pub met: bool,
    pub message: String,
    pub fits: usize,
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub cancelled: bool,
}

impl ToleranceSearch {
    pub fn tolerance(&self) -> f64 {
        self.optimization_config.tolerance.unwrap_or(f64::NAN)
    }
}

impl OptimizationRun {
    pub fn new(
        spline_config: SplineConfig,
        optimization_config: OptimizationConfig,
        result: FitResult,
        loss_history: Vec<IterationRecord>,
    ) -> Self {
        OptimizationRun {
            run_id: uuid::Uuid::new_v4().to_string(),
            number: 0,
            timestamp: now_iso(),
            spline_config,
            optimization_config,
            result,
            loss_history,
            search: None,
        }
    }
}

/// An airfoil, its runs and its tolerance searches.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectFile {
    pub metadata: ProjectMetadata,
    pub airfoil: AirfoilData,
    pub runs: Vec<OptimizationRun>,
    pub searches: Vec<ToleranceSearch>,
}

impl ProjectFile {
    pub fn new(airfoil: AirfoilData) -> Self {
        ProjectFile { metadata: ProjectMetadata::new(None), airfoil, runs: vec![], searches: vec![] }
    }

    pub fn search(&self, id: &str) -> Option<&ToleranceSearch> {
        self.searches.iter().find(|s| s.search_id == id)
    }

    /// The runs of a search, in the order they were added.
    pub fn search_runs<'a>(&'a self, id: &'a str) -> impl Iterator<Item = &'a OptimizationRun> + 'a {
        self.runs.iter().filter(move |r| r.search.as_deref() == Some(id))
    }
}

/// Numbers the runs without a number (`<= 0`) after the largest one (or
/// `after`); returns the last number.
pub fn number_runs(runs: &mut [OptimizationRun], after: usize) -> usize {
    let mut last = runs.iter().map(|r| r.number).max().unwrap_or(0).max(after);
    for run in runs.iter_mut().filter(|r| r.number == 0) {
        last += 1;
        run.number = last;
    }
    last
}

/// A new run or search id.
pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Numbers the searches without a number after the largest one.
pub fn number_searches(searches: &mut [ToleranceSearch]) {
    let mut last = searches.iter().map(|s| s.number).max().unwrap_or(0);
    for s in searches.iter_mut().filter(|s| s.number == 0) {
        last += 1;
        s.number = last;
    }
}

/// Local time like Python's `datetime.now().isoformat()`.
pub fn now_iso() -> String {
    let now = chrono::Local::now();
    if now.timestamp_subsec_micros() == 0 {
        now.format("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        now.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
    }
}

fn platform() -> String {
    match std::env::consts::OS {
        "windows" => "Windows",
        "macos" => "Darwin",
        "linux" => "Linux",
        other => other,
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// JSON values with the distinctions the file format keeps

/// A JSON value that keeps integers and floats apart and allows
/// non-finite floats.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

fn obj(fields: Vec<(&str, Json)>) -> Json {
    Json::Object(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn float(v: f64) -> Json {
    Json::Float(v)
}

fn int(v: usize) -> Json {
    Json::Int(v as i64)
}

fn opt_float(v: Option<f64>) -> Json {
    v.map_or(Json::Null, Json::Float)
}

fn string(s: &str) -> Json {
    Json::Str(s.to_string())
}

fn points(p: &[Vec2]) -> Json {
    Json::Array(p.iter().map(|p| Json::Array(vec![float(p.x), float(p.y)])).collect())
}

fn floats(v: &[f64]) -> Json {
    Json::Array(v.iter().map(|&x| float(x)).collect())
}

fn ints(v: &[usize]) -> Json {
    Json::Array(v.iter().map(|&x| int(x)).collect())
}

impl Json {
    /// Text as Python's `json.dumps(indent=2, ensure_ascii=False)` writes it.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0);
        out
    }

    fn write(&self, out: &mut String, depth: usize) {
        let indent = |out: &mut String, d: usize| {
            out.push('\n');
            out.push_str(&"  ".repeat(d));
        };
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => out.push_str(&i.to_string()),
            // Non-finite numbers are written as strings, so a project is standard JSON.
            Json::Float(f) if f.is_nan() => out.push_str("\"NaN\""),
            Json::Float(f) if f.is_infinite() => out.push_str(if *f > 0.0 { "\"Infinity\"" } else { "\"-Infinity\"" }),
            Json::Float(f) => out.push_str(&shortest(*f)),
            Json::Str(s) => out.push_str(&serde_json::to_string(s).unwrap()),
            Json::Array(items) if items.is_empty() => out.push_str("[]"),
            Json::Object(items) if items.is_empty() => out.push_str("{}"),
            Json::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    indent(out, depth + 1);
                    item.write(out, depth + 1);
                }
                indent(out, depth);
                out.push(']');
            }
            Json::Object(items) => {
                out.push('{');
                for (i, (key, value)) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    indent(out, depth + 1);
                    out.push_str(&serde_json::to_string(key).unwrap());
                    out.push_str(": ");
                    value.write(out, depth + 1);
                }
                indent(out, depth);
                out.push('}');
            }
        }
    }

    /// Parses JSON text, also accepting the bare `NaN`, `Infinity` and
    /// `-Infinity` of older projects.
    pub fn parse(text: &str) -> std::result::Result<Json, String> {
        let value: serde_json::Value = serde_json::from_str(&mark_non_finite(text)).map_err(|e| e.to_string())?;
        Ok(Json::from_value(value))
    }

    fn from_value(value: serde_json::Value) -> Json {
        use serde_json::Value;
        match value {
            Value::Null => Json::Null,
            Value::Bool(b) => Json::Bool(b),
            Value::Number(n) => n.as_i64().map_or_else(|| Json::Float(n.as_f64().unwrap_or(f64::NAN)), Json::Int),
            Value::String(s) => match s.as_str() {
                NAN_MARK => Json::Float(f64::NAN),
                INF_MARK => Json::Float(f64::INFINITY),
                NEG_INF_MARK => Json::Float(f64::NEG_INFINITY),
                _ => Json::Str(s),
            },
            Value::Array(items) => Json::Array(items.into_iter().map(Json::from_value).collect()),
            Value::Object(map) => Json::Object(map.into_iter().map(|(k, v)| (k, Json::from_value(v))).collect()),
        }
    }

    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
}

const NAN_MARK: &str = "\u{0}NaN";
const INF_MARK: &str = "\u{0}Infinity";
const NEG_INF_MARK: &str = "\u{0}-Infinity";

/// The text with bare `NaN` / `Infinity` / `-Infinity` (outside strings)
/// replaced by marker strings standard JSON can hold.
fn mark_non_finite(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if in_string {
            in_string = escaped || c != '"';
            escaped = !escaped && c == '\\';
        } else if c == '"' {
            in_string = true;
        } else {
            let replacement = [("-Infinity", NEG_INF_MARK), ("Infinity", INF_MARK), ("NaN", NAN_MARK)]
                .into_iter()
                .find(|(token, _)| rest.starts_with(token));
            if let Some((token, mark)) = replacement {
                out.push_str(&serde_json::to_string(mark).unwrap());
                rest = &rest[token.len()..];
                continue;
            }
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

// ---------------------------------------------------------------------------
// Encoding

fn encode_airfoil(a: &AirfoilData) -> Json {
    obj(vec![
        ("coordinates", points(&a.coordinates)),
        ("sections", Json::Array(a.sections.iter().map(|s| points(s)).collect())),
        ("junction_indices", ints(&a.junction_indices)),
        ("num_sections", int(a.num_sections())),
        ("filename", string(&a.filename)),
        ("name", string(&a.name)),
        ("precision", Json::Int(a.precision as i64)),
    ])
}

fn encode_spline_config(c: &SplineConfig) -> Json {
    obj(vec![
        ("degree", int(c.degree)),
        ("num_control_points", int(c.num_control_points)),
        ("num_sections", int(c.num_sections)),
        ("continuity", string(c.continuity.name())),
        ("geometric_continuity", Json::Bool(c.geometric_continuity)),
        ("junction_continuity", Json::Array(c.junction_continuity.iter().map(|s| string(s)).collect())),
    ])
}

fn encode_optimization_config(c: &OptimizationConfig) -> Json {
    obj(vec![
        ("optimizer", string(c.optimizer.value())),
        ("smoothing_weight", float(c.smoothing_weight)),
        ("optimize_le_position", Json::Bool(c.optimize_le_position)),
        ("parameterization", string(c.parameterization.value())),
        ("tolerance", opt_float(c.tolerance)),
        ("knot_placement", string(placement_value(c.knot_placement))),
        ("grow_control_points", Json::Bool(c.grow_control_points)),
        ("min_knot_gap", float(c.min_knot_gap)),
        // Kept for older readers: data parameters given by the caller, never set.
        ("custom_parameters", Json::Null),
    ])
}

fn encode_report(r: &Report) -> Json {
    let mut fields = vec![
        ("max_deviation", float(r.max_deviation)),
        ("rms_deviation", float(r.rms_deviation)),
        ("section_max_deviation", floats(&r.section_max_deviation)),
        ("control_points", int(r.control_points)),
        ("section_control_points", ints(&r.section_control_points)),
        ("section_interior_knots", ints(&r.section_interior_knots)),
        ("degree", int(r.degree)),
        ("tolerance", opt_float(r.tolerance)),
        ("precision_floor", opt_float(r.precision_floor)),
    ];
    if let Some(f) = &r.fairness {
        fields.extend([
            ("curvature_extrema", int(f.curvature_extrema)),
            ("roughness", float(f.roughness)),
            ("max_curvature_slope", float(f.max_curvature_slope)),
            ("max_curvature", float(f.max_curvature)),
            (
                "junctions",
                Json::Array(
                    f.junctions
                        .iter()
                        .map(|j| {
                            obj(vec![
                                ("gap", float(j.gap)),
                                ("tangent_angle_deg", float(j.tangent_angle_deg)),
                                ("curvature_jump", float(j.curvature_jump)),
                                ("curvature_slope_jump", float(j.curvature_slope_jump)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("leading_edge_radius", opt_float(f.leading_edge_radius)),
        ]);
    }
    let shape = r.shape.as_ref();
    if r.shape_keys {
        fields.extend([
            ("max_thickness", opt_float(shape.map(|s| s.max_thickness))),
            ("max_thickness_x", opt_float(shape.map(|s| s.max_thickness_x))),
            ("max_camber", opt_float(shape.map(|s| s.max_camber))),
            ("max_camber_x", opt_float(shape.map(|s| s.max_camber_x))),
        ]);
    }
    obj(fields)
}

fn encode_fit_result(r: &FitResult) -> Json {
    obj(vec![
        ("degree", int(r.degree)),
        ("final_error", float(r.final_error)),
        ("num_iterations", int(r.num_iterations)),
        ("num_evaluations", int(r.num_evaluations)),
        ("elapsed_seconds", float(r.elapsed_seconds)),
        ("converged", Json::Bool(r.converged)),
        ("cancelled", Json::Bool(r.cancelled)),
        ("section_knots", r.section_knots.as_ref().map_or(Json::Null, |k| Json::Array(k.iter().map(|v| floats(v)).collect()))),
        ("report", r.report.as_ref().map_or(Json::Null, encode_report)),
        ("rms_error", opt_float(r.rms_error)),
    ])
}

fn encode_run(run: &OptimizationRun) -> Json {
    let mut fields = vec![
        ("run_id", string(&run.run_id)),
        ("number", int(run.number)),
        ("timestamp", string(&run.timestamp)),
        ("spline_config", encode_spline_config(&run.spline_config)),
        ("optimization_config", encode_optimization_config(&run.optimization_config)),
        ("result", encode_fit_result(&run.result)),
        (
            "loss_history",
            Json::Array(
                run.loss_history
                    .iter()
                    .map(|r| obj(vec![("iter", int(r.iteration)), ("error", float(r.error)), ("time", float(r.elapsed_seconds))]))
                    .collect(),
            ),
        ),
        ("section_control_points", Json::Array(run.result.section_control_points.iter().map(|s| points(s)).collect())),
    ];
    if let Some(search) = &run.search {
        fields.push(("search", string(search)));
    }
    obj(fields)
}

fn encode_search(s: &ToleranceSearch) -> Json {
    obj(vec![
        ("search_id", string(&s.search_id)),
        ("number", int(s.number)),
        ("timestamp", string(&s.timestamp)),
        ("spline_config", encode_spline_config(&s.spline_config)),
        ("optimization_config", encode_optimization_config(&s.optimization_config)),
        ("result_run", string(&s.result_run)),
        ("control_points", int(s.control_points)),
        ("max_deviation", float(s.max_deviation)),
        ("met", Json::Bool(s.met)),
        ("message", string(&s.message)),
        ("fits", int(s.fits)),
        ("evaluations", int(s.evaluations)),
        ("elapsed_seconds", float(s.elapsed_seconds)),
        ("cancelled", Json::Bool(s.cancelled)),
    ])
}

fn encode_project(p: &ProjectFile) -> Json {
    let m = &p.metadata;
    let mut json = obj(vec![
        ("format_version", string(&m.format_version)),
        (
            "metadata",
            obj(vec![
                ("format_version", string(&m.format_version)),
                ("created", string(&m.created)),
                ("modified", string(&m.modified)),
                ("app_version", string(&m.app_version)),
                ("platform", string(&m.platform)),
            ]),
        ),
        ("airfoil", encode_airfoil(&p.airfoil)),
        ("optimization_runs", Json::Array(p.runs.iter().map(encode_run).collect())),
    ]);
    // Written only when there are searches, so other projects keep the
    // layout of format 1.0.
    if !p.searches.is_empty() {
        if let Json::Object(fields) = &mut json {
            fields.push(("tolerance_searches".into(), Json::Array(p.searches.iter().map(encode_search).collect())));
        }
    }
    json
}

/// The text of a project file.
pub fn project_text(project: &ProjectFile) -> String {
    encode_project(project).to_text()
}

/// Writes a project atomically (a temporary file in the target directory
/// replaces the target), with `modified` set to now. A failure leaves an
/// existing file untouched.
pub fn save_project(project: &mut ProjectFile, path: &Path) -> Result<()> {
    let io = |e: std::io::Error| Error::new(e.to_string());
    project.metadata.modified = now_iso();
    // Loading brought older runs to this format.
    project.metadata.format_version = FORMAT_VERSION.into();
    let text = project_text(project);
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.permissions().readonly() {
            bail!("{} is read-only.", path.display());
        }
    }
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let mut temp = tempfile::Builder::new().prefix(".").suffix(".tmp").tempfile_in(dir).map_err(io)?;
    use std::io::Write;
    temp.write_all(text.as_bytes()).map_err(io)?;
    temp.as_file().sync_all().map_err(io)?;
    if let Ok(meta) = std::fs::metadata(path) {
        temp.as_file().set_permissions(meta.permissions()).map_err(io)?;
    }
    temp.persist(path).map_err(|e| io(e.error))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Decoding

/// Reads one part of a project, naming it in errors.
struct Part<'a> {
    json: &'a Json,
    place: &'a str,
}

impl<'a> Part<'a> {
    fn error(&self, message: impl std::fmt::Display) -> Error {
        Error::new(format!("{}: {message}.", self.place))
    }

    fn field(&self, key: &str) -> Result<&'a Json> {
        match self.json {
            Json::Object(_) => self.json.get(key).ok_or_else(|| self.error(format_args!("the '{key}' field is missing"))),
            _ => Err(self.error("it is not an object")),
        }
    }

    fn optional(&self, key: &str) -> Option<&'a Json> {
        self.json.get(key).filter(|v| !v.is_null())
    }

    /// A number; also the strings "NaN", "Infinity" and "-Infinity".
    fn number_of(&self, key: &str, v: &Json) -> Result<f64> {
        match v {
            Json::Int(i) => Ok(*i as f64),
            Json::Float(f) => Ok(*f),
            Json::Str(s) if s == "NaN" => Ok(f64::NAN),
            Json::Str(s) if s == "Infinity" => Ok(f64::INFINITY),
            Json::Str(s) if s == "-Infinity" => Ok(f64::NEG_INFINITY),
            _ => Err(self.error(format_args!("'{key}' is not a number"))),
        }
    }

    fn float(&self, key: &str) -> Result<f64> {
        self.number_of(key, self.field(key)?)
    }

    fn float_or(&self, key: &str, default: f64) -> Result<f64> {
        self.optional(key).map_or(Ok(default), |v| self.number_of(key, v))
    }

    fn opt_float(&self, key: &str) -> Result<Option<f64>> {
        self.optional(key).map(|v| self.number_of(key, v)).transpose()
    }

    fn count_of(&self, key: &str, v: &Json) -> Result<usize> {
        match v {
            Json::Int(i) if *i >= 0 => Ok(*i as usize),
            Json::Float(f) if *f >= 0.0 && f.fract() == 0.0 => Ok(*f as usize),
            _ => Err(self.error(format_args!("'{key}' is not a count"))),
        }
    }

    fn count(&self, key: &str) -> Result<usize> {
        self.count_of(key, self.field(key)?)
    }

    fn count_or(&self, key: &str, default: usize) -> Result<usize> {
        self.optional(key).map_or(Ok(default), |v| self.count_of(key, v))
    }

    fn bool_or(&self, key: &str, default: bool) -> Result<bool> {
        match self.optional(key) {
            None => Ok(default),
            Some(Json::Bool(b)) => Ok(*b),
            Some(_) => Err(self.error(format_args!("'{key}' is not true or false"))),
        }
    }

    fn text_of(&self, key: &str, v: &'a Json) -> Result<&'a str> {
        match v {
            Json::Str(s) => Ok(s),
            _ => Err(self.error(format_args!("'{key}' is not a text"))),
        }
    }

    fn text(&self, key: &str) -> Result<&'a str> {
        self.text_of(key, self.field(key)?)
    }

    fn text_or(&self, key: &str, default: &'a str) -> Result<&'a str> {
        self.optional(key).map_or(Ok(default), |v| self.text_of(key, v))
    }

    fn list_of(&self, key: &str, v: &'a Json) -> Result<&'a [Json]> {
        match v {
            Json::Array(items) => Ok(items),
            _ => Err(self.error(format_args!("'{key}' is not a list"))),
        }
    }

    fn list(&self, key: &str) -> Result<&'a [Json]> {
        self.list_of(key, self.field(key)?)
    }

    fn points_of(&self, key: &str, v: &'a Json) -> Result<Vec<Vec2>> {
        self.list_of(key, v)?
            .iter()
            .map(|p| {
                let xy = self.list_of(key, p)?;
                if xy.len() < 2 {
                    return Err(self.error(format_args!("a point of '{key}' has fewer than two coordinates")));
                }
                Ok(Vec2::new(self.number_of(key, &xy[0])?, self.number_of(key, &xy[1])?))
            })
            .collect()
    }

    fn sections(&self, key: &str) -> Result<Vec<Vec<Vec2>>> {
        self.list(key)?.iter().map(|s| self.points_of(key, s)).collect()
    }

    fn floats_of(&self, key: &str, v: &'a Json) -> Result<Vec<f64>> {
        self.list_of(key, v)?.iter().map(|x| self.number_of(key, x)).collect()
    }

    fn counts(&self, key: &str) -> Result<Vec<usize>> {
        self.list(key)?.iter().map(|x| self.count_of(key, x)).collect()
    }
}

/// The airfoil; its sections are rebuilt from the coordinates and the
/// junction indices (the stored `sections` and `num_sections` are not read).
fn decode_airfoil(json: &Json) -> Result<AirfoilData> {
    let part = Part { json, place: "Airfoil" };
    let coordinates = part.points_of("coordinates", part.field("coordinates")?)?;
    let junction_indices = part.counts("junction_indices")?;
    let n = coordinates.len();
    let inside = junction_indices.iter().all(|&j| 1 <= j && j + 2 <= n);
    if !inside || junction_indices.windows(2).any(|w| w[1] <= w[0]) {
        return Err(part.error("the junction indices do not split the coordinates"));
    }
    let starts = std::iter::once(0).chain(junction_indices.iter().copied());
    let ends = junction_indices.iter().copied().chain(std::iter::once(n - 1));
    let sections = starts.zip(ends).map(|(a, b)| coordinates[a..=b].to_vec()).collect();
    Ok(AirfoilData {
        sections,
        coordinates,
        junction_indices,
        filename: part.text_or("filename", "")?.to_string(),
        name: part.text_or("name", "")?.to_string(),
        precision: part.float_or("precision", 6.0)? as i32,
    })
}

fn decode_spline_config(json: &Json, place: &str) -> Result<SplineConfig> {
    let part = Part { json, place };
    let continuity = ContinuityMode::parse_or_default(part.text("continuity")?);
    let geometric = match part.optional("geometric_continuity") {
        Some(Json::Bool(b)) => *b,
        Some(_) => return Err(part.error("'geometric_continuity' is not true or false")),
        // Older projects stored "continuity_type": "GEOMETRIC" or "PARAMETRIC".
        None => part.text_or("continuity_type", "")?.eq_ignore_ascii_case("GEOMETRIC"),
    };
    let junctions = match part.optional("junction_continuity") {
        None => vec![],
        Some(v) => part
            .list_of("junction_continuity", v)?
            .iter()
            .map(|s| part.text_of("junction_continuity", s).map(String::from))
            .collect::<Result<_>>()?,
    };
    SplineConfig {
        degree: part.count("degree")?,
        num_control_points: part.count("num_control_points")?,
        num_sections: part.count_or("num_sections", 2)?,
        continuity,
        geometric_continuity: geometric,
        junction_continuity: junctions,
    }
    .validated()
    .map_err(|e| part.error(e.0.trim_end_matches('.')))
}

/// Settings of a run; also returns a warning for a retired or unknown optimizer.
fn decode_optimization_config(json: &Json, place: &str) -> Result<(OptimizationConfig, Option<String>)> {
    let part = Part { json, place };
    let stored = part.text("optimizer")?;
    let (optimizer, known) = OptimizerType::from_value(stored);
    let warning = (stored.to_lowercase() != optimizer.value()).then(|| {
        let verb = if known { "was retired" } else { "is unknown" };
        format!("Optimizer '{stored}' {verb}; showing the run as {}.", optimizer.display_name())
    });
    let tolerance = part.opt_float("tolerance")?;
    let default_placement = if tolerance.is_none() { "uniform" } else { "free" };
    let knot_placement: KnotPlacement =
        placement_from_value(part.text_or("knot_placement", default_placement)?).map_err(|e| part.error(e.0.trim_end_matches('.')))?;
    let config = OptimizationConfig {
        optimizer,
        optimize_le_position: part.bool_or("optimize_le_position", false)?,
        smoothing_weight: part.float_or("smoothing_weight", 1e-3)?,
        parameterization: Parameterization::from_value(part.text_or("parameterization", "centripetal")?).map_err(|e| part.error(e.0))?,
        tolerance,
        knot_placement,
        grow_control_points: part.bool_or("grow_control_points", tolerance.is_some())?,
        // Projects from before this setting were fitted with 0.001.
        min_knot_gap: part.float_or("min_knot_gap", 1e-3)?,
    };
    Ok((config.validated().map_err(|e| part.error(e.0))?, warning))
}

fn decode_report(json: &Json, place: &str) -> Result<Report> {
    let part = Part { json, place };
    let fairness = match part.optional("max_curvature") {
        None => None,
        Some(_) => Some(FairnessReport {
            curvature_extrema: part.count("curvature_extrema")?,
            roughness: part.float("roughness")?,
            max_curvature_slope: part.float("max_curvature_slope")?,
            max_curvature: part.float("max_curvature")?,
            junctions: match part.optional("junctions") {
                None => vec![],
                Some(list) => part
                    .list_of("junctions", list)?
                    .iter()
                    .map(|j| {
                        let p = Part { json: j, place };
                        Ok(JunctionReport {
                            gap: p.float("gap")?,
                            tangent_angle_deg: p.float("tangent_angle_deg")?,
                            curvature_jump: p.float("curvature_jump")?,
                            curvature_slope_jump: p.float("curvature_slope_jump")?,
                        })
                    })
                    .collect::<Result<_>>()?,
            },
            leading_edge_radius: part.opt_float("leading_edge_radius")?,
        }),
    };
    let shape_keys = part.json.get("max_thickness").is_some();
    let shape = match part.opt_float("max_thickness")? {
        None => None,
        Some(max_thickness) => Some(ShapeReport {
            max_thickness,
            max_thickness_x: part.float("max_thickness_x")?,
            max_camber: part.float("max_camber")?,
            max_camber_x: part.float("max_camber_x")?,
        }),
    };
    Ok(Report {
        max_deviation: part.float("max_deviation")?,
        rms_deviation: part.float("rms_deviation")?,
        section_max_deviation: part.floats_of("section_max_deviation", part.field("section_max_deviation")?)?,
        control_points: part.count("control_points")?,
        section_control_points: part.counts("section_control_points")?,
        section_interior_knots: part.counts("section_interior_knots")?,
        degree: part.count("degree")?,
        tolerance: part.opt_float("tolerance")?,
        precision_floor: part.opt_float("precision_floor")?,
        fairness,
        shape,
        shape_keys,
    })
}

/// The result of a run; its degree is the run's spline degree (the stored
/// `degree` is not read).
fn decode_fit_result(json: &Json, place: &str, control_points: Vec<Vec<Vec2>>, degree: usize) -> Result<FitResult> {
    let part = Part { json, place };
    let section_knots = match part.optional("section_knots") {
        None => None,
        Some(v) => Some(part.list_of("section_knots", v)?.iter().map(|k| part.floats_of("section_knots", k)).collect::<Result<_>>()?),
    };
    Ok(FitResult {
        section_control_points: control_points,
        degree,
        final_error: part.float("final_error")?,
        num_iterations: part.count("num_iterations")?,
        num_evaluations: part.count("num_evaluations")?,
        elapsed_seconds: part.float("elapsed_seconds")?,
        converged: part.bool_or("converged", true)?,
        cancelled: part.bool_or("cancelled", false)?,
        section_knots,
        report: part.optional("report").map(|r| decode_report(r, place)).transpose()?,
        rms_error: part.opt_float("rms_error")?,
    })
}

fn decode_run(json: &Json, place: &str, warnings: &mut Vec<String>) -> Result<OptimizationRun> {
    let run = Part { json, place };
    let control_points = Part { json, place: &format!("{place} control points") }.sections("section_control_points")?;
    let spline_config = decode_spline_config(run.field("spline_config")?, &format!("{place} spline settings"))?;
    let (optimization_config, warning) = decode_optimization_config(run.field("optimization_config")?, &format!("{place} fit settings"))?;
    warnings.extend(warning);
    let result = decode_fit_result(run.field("result")?, &format!("{place} result"), control_points, spline_config.degree)?;
    let history_place = format!("{place} loss history");
    let history = Part { json, place: &history_place };
    let loss_history = history
        .list("loss_history")?
        .iter()
        .map(|r| {
            let p = Part { json: r, place: &history_place };
            Ok(IterationRecord { iteration: p.count("iter")?, error: p.float("error")?, elapsed_seconds: p.float_or("time", 0.0)? })
        })
        .collect::<Result<_>>()?;
    Ok(OptimizationRun {
        run_id: run.text("run_id")?.to_string(),
        number: run.count_or("number", 0)?,
        timestamp: run.text("timestamp")?.to_string(),
        spline_config,
        optimization_config,
        result,
        loss_history,
        search: run.optional("search").map(|v| run.text_of("search", v).map(str::to_string)).transpose()?,
    })
}

fn decode_search(json: &Json, place: &str, warnings: &mut Vec<String>) -> Result<ToleranceSearch> {
    let s = Part { json, place };
    let spline_config = decode_spline_config(s.field("spline_config")?, &format!("{place} spline settings"))?;
    let (optimization_config, warning) = decode_optimization_config(s.field("optimization_config")?, &format!("{place} fit settings"))?;
    warnings.extend(warning);
    if optimization_config.tolerance.is_none() {
        return Err(s.error("the search has no tolerance"));
    }
    Ok(ToleranceSearch {
        search_id: s.text("search_id")?.to_string(),
        number: s.count_or("number", 0)?,
        timestamp: s.text_or("timestamp", "")?.to_string(),
        spline_config,
        optimization_config,
        result_run: s.text("result_run")?.to_string(),
        control_points: s.count("control_points")?,
        max_deviation: s.float("max_deviation")?,
        met: s.bool_or("met", false)?,
        message: s.text_or("message", "")?.to_string(),
        fits: s.count_or("fits", 0)?,
        evaluations: s.count_or("evaluations", 0)?,
        elapsed_seconds: s.float_or("elapsed_seconds", 0.0)?,
        cancelled: s.bool_or("cancelled", false)?,
    })
}

/// A project read from a file, with warnings about retired settings.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedProject {
    pub project: ProjectFile,
    pub warnings: Vec<String>,
}

/// Reads a project file. Errors name the part that is wrong.
pub fn load_project(path: &Path) -> Result<LoadedProject> {
    let bytes = std::fs::read(path).map_err(|e| Error::new(e.to_string()))?;
    let body = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(&bytes);
    let text = std::str::from_utf8(body).map_err(|e| Error::new(format!("The file is not a project file: {e}.")))?;
    parse_project(text)
}

/// Reads the text of a project file.
pub fn parse_project(text: &str) -> Result<LoadedProject> {
    let json = Json::parse(text).map_err(|e| Error::new(format!("The file is not a project file: {e}.")))?;
    if !matches!(json, Json::Object(_)) {
        bail!("The file is not a project file: it holds no project.");
    }
    let version = match json.get("format_version") {
        None => FORMAT_VERSION.to_string(),
        Some(Json::Str(s)) => s.clone(),
        Some(other) => other.to_text(),
    };
    let major: u32 = version
        .split('.')
        .next()
        .and_then(|m| m.trim().parse().ok())
        .ok_or_else(|| Error::new(format!("Project file version {} is not valid.", bspline_varpro::numfmt::quoted(&version))))?;
    if major > 1 {
        bail!("Project file version {version} is not supported. Please update the application.");
    }
    let minor: u32 = version.split('.').nth(1).and_then(|m| m.trim().parse().ok()).unwrap_or(0);
    let mut warnings = Vec::new();
    let root = Part { json: &json, place: "Run list" };
    let runs = match root.optional("optimization_runs") {
        None => vec![],
        Some(v) => root.list_of("optimization_runs", v)?.to_vec(),
    };
    let mut runs: Vec<OptimizationRun> =
        runs.iter().enumerate().map(|(i, r)| decode_run(r, &format!("Run {}", i + 1), &mut warnings)).collect::<Result<_>>()?;
    number_runs(&mut runs, 0);
    if major == 1 && minor < 1 {
        for run in runs.iter_mut().filter(|r| r.optimization_config.tolerance.is_some()) {
            run.optimization_config.smoothing_weight = OLD_TOLERANCE_SMOOTHING;
        }
    }
    let empty = Json::Object(vec![]);
    let m = Part { json: json.get("metadata").unwrap_or(&empty), place: "Metadata" };
    let metadata = ProjectMetadata {
        format_version: m.text_or("format_version", FORMAT_VERSION)?.to_string(),
        created: m.text_or("created", "")?.to_string(),
        modified: m.text_or("modified", "")?.to_string(),
        app_version: m.text_or("app_version", "")?.to_string(),
        platform: m.text_or("platform", "")?.to_string(),
    };
    let searches = match root.optional("tolerance_searches") {
        None => vec![],
        Some(v) => root.list_of("tolerance_searches", v)?.to_vec(),
    };
    let mut searches: Vec<ToleranceSearch> = searches
        .iter()
        .enumerate()
        .map(|(i, s)| decode_search(s, &format!("Search {}", i + 1), &mut warnings))
        .collect::<Result<_>>()?;
    // A search keeps only runs it names; a run naming no search is a run
    // of its own.
    searches.retain(|s| runs.iter().any(|r| r.search.as_deref() == Some(s.search_id.as_str())));
    for run in &mut runs {
        if run.search.as_ref().is_some_and(|id| !searches.iter().any(|s| &s.search_id == id)) {
            run.search = None;
        }
    }
    number_searches(&mut searches);
    let airfoil = decode_airfoil(Part { json: &json, place: "Airfoil" }.field("airfoil")?)?;
    Ok(LoadedProject { project: ProjectFile { metadata, airfoil, runs, searches }, warnings })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project() -> ProjectFile {
        let pts = bspline_varpro::airfoil::Naca4::new("0012", true).unwrap().points(11);
        let airfoil = AirfoilData::new(pts, 2, "naca0012", "NACA 0012", 6).unwrap();
        let result = FitResult {
            section_control_points: vec![vec![Vec2::new(1.0, 0.0), Vec2::new(0.1, 0.06)], vec![Vec2::new(0.1, 0.06), Vec2::new(1.0, 0.0)]],
            degree: 5,
            final_error: 1.5e-7,
            num_iterations: 3,
            num_evaluations: 4,
            elapsed_seconds: 0.25,
            converged: true,
            cancelled: false,
            section_knots: Some(vec![vec![0.0, 0.0, 1.0, 1.0]; 2]),
            report: None,
            rms_error: Some(1e-7),
        };
        let mut p = ProjectFile::new(airfoil);
        p.runs.push(OptimizationRun::new(
            SplineConfig::default(),
            OptimizationConfig::default(),
            result,
            vec![IterationRecord { iteration: 1, error: 0.5, elapsed_seconds: 0.0 }],
        ));
        number_runs(&mut p.runs, 0);
        p
    }

    #[test]
    fn reports_keep_their_shape_keys_or_their_absence() {
        for (keys, shape) in [
            (false, None),
            (true, None),
            (true, Some(ShapeReport { max_thickness: 0.12, max_thickness_x: 0.3, max_camber: 0.02, max_camber_x: 0.4 })),
        ] {
            let mut p = project();
            p.runs[0].result.report = Some(Report {
                max_deviation: 1e-5,
                rms_deviation: 1e-6,
                section_max_deviation: vec![1e-5, 9e-6],
                control_points: 3,
                section_control_points: vec![2, 2],
                section_interior_knots: vec![0, 0],
                degree: 5,
                tolerance: None,
                precision_floor: None,
                fairness: None,
                shape,
                shape_keys: keys,
            });
            let text = project_text(&p);
            // A report saved before the shape keys (Python passes it through).
            assert_eq!(text.contains("\"max_thickness\""), keys, "{keys} {shape:?}");
            let loaded = parse_project(&text).unwrap().project;
            assert_eq!(loaded.runs[0].result.report, p.runs[0].result.report);
        }
    }

    #[test]
    fn round_trip() {
        let mut p = project();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.asfproj");
        save_project(&mut p, &path).unwrap();
        let loaded = load_project(&path).unwrap();
        assert_eq!(loaded.project, p);
        assert!(loaded.warnings.is_empty());
    }

    #[test]
    fn retired_optimizer_and_infinity() {
        let text = project_text(&project()).replace("\"gn-varpro\"", "\"bfgs\"").replace("\"rms_error\": 1e-07", "\"rms_error\": Infinity");
        let loaded = parse_project(&text).unwrap();
        assert_eq!(loaded.project.runs[0].optimization_config.optimizer, OptimizerType::VarPro);
        assert_eq!(loaded.warnings, vec!["Optimizer 'bfgs' was retired; showing the run as VarPro LM."]);
        assert_eq!(loaded.project.runs[0].result.rms_error, Some(f64::INFINITY));
    }

    #[test]
    fn non_finite_numbers_are_strings() {
        let mut p = project();
        p.runs[0].result.final_error = f64::NAN;
        p.runs[0].result.rms_error = Some(f64::NEG_INFINITY);
        let text = project_text(&p);
        assert!(text.contains("\"final_error\": \"NaN\"") && text.contains("\"rms_error\": \"-Infinity\""));
        assert!(serde_json::from_str::<serde_json::Value>(&text).is_ok());
        let r = &parse_project(&text).unwrap().project.runs[0].result;
        assert!(r.final_error.is_nan() && r.rms_error == Some(f64::NEG_INFINITY));
    }

    /// The project's text with `change` applied to its JSON.
    fn edited(change: impl Fn(&mut serde_json::Value)) -> String {
        let mut json: serde_json::Value = serde_json::from_str(&project_text(&project())).unwrap();
        change(&mut json);
        json.to_string()
    }

    #[test]
    fn sections_and_degree_are_rebuilt_not_read() {
        let text = edited(|j| {
            j["airfoil"]["num_sections"] = 7.into();
            j["airfoil"]["sections"] = serde_json::json!([]);
            j["optimization_runs"][0]["result"]["degree"] = 9.into();
        });
        let loaded = parse_project(&text).unwrap().project;
        assert_eq!(loaded.airfoil, project().airfoil);
        assert_eq!(loaded.runs[0].result.degree, 5);
        let junctions = |v: serde_json::Value| parse_project(&edited(|j| j["airfoil"]["junction_indices"] = v.clone()));
        for bad in [serde_json::json!([0]), serde_json::json!([20]), serde_json::json!([5, 5]), serde_json::json!([8, 4])] {
            assert_eq!(junctions(bad.clone()).unwrap_err().0, "Airfoil: the junction indices do not split the coordinates.", "{bad}");
        }
        assert_eq!(junctions(serde_json::json!([3, 6])).unwrap().project.airfoil.sections.len(), 3);
    }

    #[test]
    fn errors_name_the_part() {
        let text = project_text(&project()).replace("\"degree\": 5,", "");
        assert_eq!(parse_project(&text).unwrap_err().0, "Run 1 spline settings: the 'degree' field is missing.");
        assert_eq!(parse_project("[1]").unwrap_err().0, "The file is not a project file: it holds no project.");
        let newer = project_text(&project()).replacen("\"1.2.0\"", "\"2.0.0\"", 1);
        assert_eq!(parse_project(&newer).unwrap_err().0, "Project file version 2.0.0 is not supported. Please update the application.");
    }

    #[test]
    fn older_tolerance_runs_were_fitted_with_old_smoothing() {
        let tolerance_run = |version: &str| {
            edited(|j| {
                j["format_version"] = version.into();
                j["optimization_runs"][0]["optimization_config"]["tolerance"] = 1e-5.into();
                j["optimization_runs"][0]["optimization_config"]["smoothing_weight"] = 1e-3.into();
            })
        };
        let smoothing = |text: &str| parse_project(text).unwrap().project.runs[0].optimization_config.smoothing_weight;
        assert_eq!(smoothing(&tolerance_run("1.0.0")), OLD_TOLERANCE_SMOOTHING);
        assert_eq!(smoothing(&tolerance_run("1.1.0")), 1e-3);
        // A fixed-count run always used its own weight.
        let fixed = edited(|j| {
            j["format_version"] = "1.0.0".into();
            j["optimization_runs"][0]["optimization_config"]["tolerance"] = serde_json::Value::Null;
            j["optimization_runs"][0]["optimization_config"]["smoothing_weight"] = 1e-3.into();
        });
        assert_eq!(smoothing(&fixed), 1e-3);
        // Saving writes the current format, so the weight is kept as loaded.
        let mut p = parse_project(&tolerance_run("1.0.0")).unwrap().project;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.asfproj");
        save_project(&mut p, &path).unwrap();
        let again = load_project(&path).unwrap().project;
        assert_eq!(again.metadata.format_version, FORMAT_VERSION);
        assert_eq!(again.runs[0].optimization_config.smoothing_weight, OLD_TOLERANCE_SMOOTHING);
    }

    #[test]
    fn searches_and_their_runs_round_trip() {
        let mut p = project();
        let mut run = p.runs[0].clone();
        run.run_id = "fit-of-search".into();
        run.number = 0;
        run.search = Some("s1".into());
        p.runs.push(run);
        number_runs(&mut p.runs, 0);
        p.searches.push(ToleranceSearch {
            search_id: "s1".into(),
            number: 1,
            timestamp: "2026-11-19T10:00:00".into(),
            spline_config: SplineConfig::default(),
            optimization_config: OptimizationConfig { tolerance: Some(1e-4), ..OptimizationConfig::default() },
            result_run: "fit-of-search".into(),
            control_points: 2,
            max_deviation: 9e-5,
            met: true,
            message: "Within tolerance 0.0001.".into(),
            fits: 1,
            evaluations: 4,
            elapsed_seconds: 0.25,
            cancelled: false,
        });
        let loaded = parse_project(&project_text(&p)).unwrap();
        assert_eq!(loaded.project, p);
        assert_eq!(p.search_runs("s1").count(), 1);
        // A search without runs is dropped; a run naming a missing search
        // is a run of its own.
        let text = edited(|j| j["optimization_runs"][0]["search"] = "gone".into());
        let lone = parse_project(&text).unwrap().project;
        assert!(lone.runs[0].search.is_none() && lone.searches.is_empty());
    }

    #[test]
    fn python_layout() {
        let text = Json::Object(vec![("a".into(), Json::Array(vec![Json::Float(1e-7), Json::Int(2)])), ("b".into(), Json::Array(vec![]))])
            .to_text();
        assert_eq!(text, "{\n  \"a\": [\n    1e-07,\n    2\n  ],\n  \"b\": []\n}");
    }
}
