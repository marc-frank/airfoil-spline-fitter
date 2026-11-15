//! STEP AP214, IGES 5.3 and DXF R2000 export of B-spline curves, and
//! merging a chain into one curve.
//!
//! Planar curves are written in the z = 0 plane. Every file is ASCII with
//! `\n` line ends.

use std::path::{Path, PathBuf};

use nalgebra::DMatrix;
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

use crate::basis::{basis_matrix, distinct, linspace, runs};
use crate::curve::BSpline;
use crate::linalg::{lstsq, matrix_to_points};
use crate::numfmt::{exponent_upper, shortest};
use crate::quadrature::{gauss_legendre, on_interval};
use crate::{bail, Error, Result, Vec2};

/// Length unit of an exported file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Units {
    #[default]
    Millimetre,
    Metre,
    Inch,
}

impl std::str::FromStr for Units {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "mm" => Ok(Units::Millimetre),
            "m" => Ok(Units::Metre),
            "inch" => Ok(Units::Inch),
            _ => bail!("units must be 'mm', 'm', or 'inch'."),
        }
    }
}

/// CAD file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Step,
    Iges,
    Dxf,
}

impl Format {
    /// The format of a path's suffix.
    pub fn of(path: &Path) -> Result<Self> {
        let suffix = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        match suffix.as_str() {
            "step" | "stp" => Ok(Format::Step),
            "iges" | "igs" => Ok(Format::Iges),
            "dxf" => Ok(Format::Dxf),
            _ => {
                let shown = if suffix.is_empty() { String::new() } else { format!(".{suffix}") };
                bail!("Unsupported export format '{shown}'; use .step, .iges, or .dxf.")
            }
        }
    }
}

/// Writes curves in the format of the path's suffix; STEP and IGES name the
/// product after the file stem.
pub fn write(path: &Path, curves: &[BSpline], scale: f64, units: Units) -> Result<PathBuf> {
    let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let text = match Format::of(path)? {
        Format::Step => to_step(curves, &name, scale, units)?,
        Format::Iges => to_iges(curves, &name, scale, units)?,
        Format::Dxf => to_dxf(curves, scale, "0", units)?,
    };
    std::fs::write(path, text).map_err(|e| Error::new(e.to_string()))?;
    Ok(path.to_path_buf())
}

/// Knots and scaled 3-D control points of a curve.
fn curve_data(curve: &BSpline, scale: f64) -> Result<(Vec<f64>, Vec<[f64; 3]>)> {
    let control: Vec<[f64; 3]> = curve.control_points.iter().map(|p| [p.x * scale, p.y * scale, 0.0]).collect();
    if curve.knots.iter().chain(control.iter().flatten()).any(|v| !v.is_finite()) {
        bail!("Cannot export non-finite coordinates or knots.");
    }
    Ok((curve.knots.clone(), control))
}

/// Whether a clamped curve ends where it starts.
fn closed(control: &[[f64; 3]]) -> bool {
    let scale = control.iter().flatten().fold(1.0f64, |m, v| m.max(v.abs()));
    let (a, b) = (control[0], control[control.len() - 1]);
    let d = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
    d <= 1e-12 * scale
}

fn local_time(format: &str) -> String {
    chrono::Local::now().format(format).to_string()
}

/// STEP real: the shortest round-trip text in upper case, with a point in
/// the mantissa and an unpadded exponent (`1.E-7`, `2.5E16`, `0.5`).
pub fn step_real(v: f64) -> String {
    let s = shortest(v).to_uppercase();
    match s.split_once('E') {
        Some((mantissa, exp)) => {
            let point = if mantissa.contains('.') { "" } else { "." };
            format!("{mantissa}{point}E{}", exp.parse::<i32>().unwrap())
        }
        None if s.contains('.') => s,
        None => s + ".",
    }
}

/// STEP string literal: `'` and `\` doubled, non-ASCII as `\X2\hhhh\X0\`.
pub fn step_string(text: &str) -> String {
    let mut out = String::from("'");
    for ch in text.chars() {
        let code = ch as u32;
        match ch {
            '\'' => out.push_str("''"),
            '\\' => out.push_str("\\\\"),
            _ if (32..127).contains(&code) => out.push(ch),
            _ if code <= 0xFFFF => out.push_str(&format!("\\X2\\{code:04X}\\X0\\")),
            _ => out.push_str(&format!("\\X4\\{code:08X}\\X0\\")),
        }
    }
    out.push('\'');
    out
}

/// STEP AP214 text with one `B_SPLINE_CURVE_WITH_KNOTS` per curve.
pub fn to_step(curves: &[BSpline], name: &str, scale: f64, units: Units) -> Result<String> {
    let mut entities: Vec<String> = Vec::new();
    let mut add = |text: String| {
        entities.push(text);
        entities.len()
    };
    let product_name = step_string(name);
    let ctx = add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')".into());
    add(format!("APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{ctx})"));
    let pctx = add(format!("PRODUCT_CONTEXT('',#{ctx},'mechanical')"));
    let product = add(format!("PRODUCT({product_name},{product_name},'',(#{pctx}))"));
    add(format!("PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{product}))"));
    let pdf = add(format!("PRODUCT_DEFINITION_FORMATION('','',#{product})"));
    let pdctx = add(format!("PRODUCT_DEFINITION_CONTEXT('part definition',#{ctx},'design')"));
    let pd = add(format!("PRODUCT_DEFINITION('design','',#{pdf},#{pdctx})"));
    let pds = add(format!("PRODUCT_DEFINITION_SHAPE('','',#{pd})"));
    let millimetre = "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))";
    let length = match units {
        Units::Millimetre => add(millimetre.into()),
        Units::Metre => add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))".into()),
        Units::Inch => {
            let mm = add(millimetre.into());
            let factor = add(format!("LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#{mm})"));
            let dims = add("DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)".into());
            add(format!("(CONVERSION_BASED_UNIT('INCH',#{factor})LENGTH_UNIT()NAMED_UNIT(#{dims}))"))
        }
    };
    let angle = add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))".into());
    let solid = add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())".into());
    let uncertainty = add(format!(
        "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{length},'distance_accuracy_value','confusion accuracy')"
    ));
    let context = add(format!(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty}))\
         GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid}))\
         REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY'))"
    ));
    let origin = add("CARTESIAN_POINT('',(0.,0.,0.))".into());
    let z_axis = add("DIRECTION('',(0.,0.,1.))".into());
    let x_axis = add("DIRECTION('',(1.,0.,0.))".into());
    let placement = add(format!("AXIS2_PLACEMENT_3D('',#{origin},#{z_axis},#{x_axis})"));
    let mut curve_ids = Vec::new();
    for (i, curve) in curves.iter().enumerate() {
        let (knots, control) = curve_data(curve, scale)?;
        let points: Vec<String> = control
            .iter()
            .map(|p| format!("#{}", add(format!("CARTESIAN_POINT('',({}))", p.map(step_real).join(",")))))
            .collect();
        let (values, counts): (Vec<f64>, Vec<usize>) = runs(&knots).unzip();
        curve_ids.push(format!(
            "#{}",
            add(format!(
                "B_SPLINE_CURVE_WITH_KNOTS({},{},({}),.UNSPECIFIED.,{},.F.,({}),({}),.UNSPECIFIED.)",
                step_string(&format!("{name} {}", i + 1)),
                curve.degree,
                points.join(","),
                if closed(&control) { ".T." } else { ".F." },
                counts.iter().map(usize::to_string).collect::<Vec<_>>().join(","),
                values.iter().map(|&v| step_real(v)).collect::<Vec<_>>().join(","),
            ))
        ));
    }
    let curve_set = add(format!("GEOMETRIC_CURVE_SET('curves',({}))", curve_ids.join(",")));
    let representation = add(format!(
        "GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION({product_name},(#{curve_set},#{placement}),#{context})"
    ));
    add(format!("SHAPE_DEFINITION_REPRESENTATION(#{pds},#{representation})"));
    let mut lines = vec![
        "ISO-10303-21;".to_string(),
        "HEADER;".into(),
        "FILE_DESCRIPTION(('B-spline curves'),'2;1');".into(),
        format!(
            "FILE_NAME({product_name},'{}',(''),(''),'bspline-varpro','bspline-varpro','');",
            local_time("%Y-%m-%dT%H:%M:%S")
        ),
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));".into(),
        "ENDSEC;".into(),
        "DATA;".into(),
    ];
    lines.extend(entities.iter().enumerate().map(|(i, e)| format!("#{}={e};", i + 1)));
    lines.extend(["ENDSEC;".into(), "END-ISO-10303-21;".into(), String::new()]);
    Ok(lines.join("\n"))
}

/// Printable-ASCII approximation (accents dropped, other characters `?`).
fn ascii(text: &str) -> String {
    text.nfkd().filter(|c| !is_combining_mark(*c)).map(|c| if (32..127).contains(&(c as u32)) { c } else { '?' }).collect()
}

/// IGES Hollerith string of at most 60 characters; empty for empty text.
fn hollerith(text: &str) -> String {
    let text: String = ascii(text).chars().take(60).collect();
    if text.is_empty() { text } else { format!("{}H{text}", text.len()) }
}

fn iges_real(v: f64) -> String {
    exponent_upper(v, 16)
}

/// Delimited fields packed into fixed-width records without splitting a field.
fn iges_records(fields: &[String], letter: char, start: usize, pointer: &str) -> Vec<String> {
    let limit = if letter == 'P' { 64 } else { 72 };
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for field in fields {
        if !current.is_empty() && current.len() + field.len() > limit {
            lines.push(std::mem::take(&mut current));
        }
        current.push_str(field);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
        .iter()
        .enumerate()
        .map(|(i, text)| {
            if letter == 'P' {
                format!("{text:<64}{pointer:>8}{letter}{:7}", start + i)
            } else {
                format!("{text:<72}{letter}{:7}", start + i)
            }
        })
        .collect()
}

/// `,` after every field but the last, which gets `;`.
fn delimit(fields: Vec<String>) -> Vec<String> {
    let n = fields.len();
    fields.into_iter().enumerate().map(|(i, f)| if i + 1 < n { f + "," } else { f + ";" }).collect()
}

/// IGES 5.3 text with one entity 126 (B-spline curve) per curve.
pub fn to_iges(curves: &[BSpline], name: &str, scale: f64, units: Units) -> Result<String> {
    let (flag, unit_name) = match units {
        Units::Millimetre => (2, "MM"),
        Units::Metre => (6, "M"),
        Units::Inch => (1, "IN"),
    };
    let data = curves.iter().map(|c| curve_data(c, scale)).collect::<Result<Vec<_>>>()?;
    let extent = data.iter().flat_map(|(_, c)| c.iter().flatten()).fold(0.0f64, |m, v| m.max(v.abs()));
    let extent = if extent == 0.0 { 1.0 } else { extent };
    let stamp = local_time("%Y%m%d.%H%M%S");
    let start = vec![format!("{:<72}S{:7}", "B-spline curves written by bspline-varpro", 1)];
    let global: Vec<String> = [
        "1H,".to_string(),
        "1H;".into(),
        hollerith(name),
        hollerith(&format!("{name}.igs")),
        hollerith("bspline-varpro"),
        hollerith("bspline-varpro"),
        "32".into(),
        "38".into(),
        "6".into(),
        "308".into(),
        "15".into(),
        hollerith(name),
        "1.0".into(),
        flag.to_string(),
        hollerith(unit_name),
        "1".into(),
        "1.0".into(),
        hollerith(&stamp),
        "1.0E-07".into(),
        iges_real(extent),
        String::new(),
        String::new(),
        "11".into(),
        "0".into(),
        hollerith(&stamp),
    ]
    .into();
    let global = iges_records(&delimit(global), 'G', 1, "");
    let mut directory = Vec::new();
    let mut parameter: Vec<String> = Vec::new();
    for (i, ((knots, control), curve)) in data.iter().zip(curves).enumerate() {
        let de = 2 * i + 1;
        let p = curve.degree;
        let mut fields = vec![
            "126".to_string(),
            (control.len() - 1).to_string(),
            p.to_string(),
            "1".into(),
            u8::from(closed(control)).to_string(),
            "1".into(),
            "0".into(),
        ];
        fields.extend(knots.iter().map(|&k| iges_real(k)));
        fields.extend(std::iter::repeat("1.0".to_string()).take(control.len()));
        fields.extend(control.iter().flat_map(|c| c.map(iges_real)));
        fields.push(iges_real(knots[p]));
        fields.push(iges_real(knots[knots.len() - p - 1]));
        fields.extend(["0.0", "0.0", "1.0"].map(String::from));
        let p_start = parameter.len() + 1;
        let records = iges_records(&delimit(fields), 'P', p_start, &de.to_string());
        let label: String = format!("SPL{}", i + 1).chars().take(8).collect();
        directory.push(format!("{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}D{:7}", 126, p_start, 0, 0, 0, 0, 0, 0, "00000000", de));
        directory.push(format!("{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}D{:7}", 126, 0, 0, records.len(), 0, "", "", label, 0, de + 1));
        parameter.extend(records);
    }
    let terminate = format!(
        "S{:7}G{:7}D{:7}P{:7}{:<40}T{:7}",
        start.len(),
        global.len(),
        directory.len(),
        parameter.len(),
        "",
        1
    );
    let mut all = start;
    all.extend(global);
    all.extend(directory);
    all.extend(parameter);
    all.push(terminate);
    Ok(all.join("\n") + "\n")
}

const DXF_FORBIDDEN: &str = "<>/\\\":;?*|=`";

/// Symbol-table name for an ASCII DXF: non-ASCII characters as `\U+XXXX`.
fn dxf_name(text: &str) -> Result<String> {
    if text.is_empty() || text.chars().any(|c| DXF_FORBIDDEN.contains(c) || (c as u32) < 32) {
        bail!(
            "Invalid DXF layer name {}: it must not be empty or contain control characters or any of < > / \\ \" : ; ? * | = `.",
            crate::numfmt::quoted(text)
        );
    }
    Ok(text
        .chars()
        .map(|c| match c as u32 {
            0..=126 => c.to_string(),
            code @ 127..=0xFFFF => format!("\\U+{code:04X}"),
            _ => "_".into(),
        })
        .collect())
}

/// Group writer of a DXF file.
struct Dxf {
    lines: Vec<String>,
    next_handle: usize,
}

impl Dxf {
    fn pair(&mut self, code: i32, value: impl ToString) {
        self.lines.push(format!("{code:>3}"));
        self.lines.push(value.to_string());
    }

    fn handle(&mut self) -> String {
        self.next_handle += 1;
        format!("{:X}", self.next_handle)
    }

    fn record(&mut self, kind: &str, owner: &str, subclass: &str, name: &str, extra: &[(i32, &str)], handle_code: i32) -> String {
        let h = self.handle();
        self.pair(0, kind);
        self.pair(handle_code, &h);
        self.pair(330, owner);
        self.pair(100, "AcDbSymbolTableRecord");
        self.pair(100, subclass);
        self.pair(2, name);
        self.pair(70, 0);
        for (code, value) in extra {
            self.pair(*code, value);
        }
        h
    }

    /// A symbol table whose records `entries` writes; returns their handles.
    fn table(&mut self, name: &str, subclass: Option<&str>, entries: impl FnOnce(&mut Dxf, &str) -> Vec<String>) -> Vec<String> {
        let h = self.handle();
        self.pair(0, "TABLE");
        self.pair(2, name);
        self.pair(5, &h);
        self.pair(330, 0);
        self.pair(100, "AcDbSymbolTable");
        let count_at = self.lines.len() + 1;
        self.pair(70, 0);
        if let Some(subclass) = subclass {
            self.pair(100, subclass);
            self.pair(71, 0);
        }
        let written = entries(self, &h);
        self.lines[count_at] = written.len().to_string();
        self.pair(0, "ENDTAB");
        written
    }
}

/// DXF R2000 text with one `SPLINE` per curve, with the symbol tables,
/// blocks, root dictionary and owner handles AutoCAD needs for R2000.
pub fn to_dxf(curves: &[BSpline], scale: f64, layer: &str, units: Units) -> Result<String> {
    let insunits = match units {
        Units::Millimetre => 4,
        Units::Metre => 6,
        Units::Inch => 1,
    };
    let layer = dxf_name(layer)?;
    let data = curves.iter().map(|c| curve_data(c, scale)).collect::<Result<Vec<_>>>()?;
    let mut d = Dxf { lines: Vec::new(), next_handle: 0 };
    for (code, value) in [(0, "SECTION"), (2, "CLASSES"), (0, "ENDSEC"), (0, "SECTION"), (2, "TABLES")] {
        d.pair(code, value);
    }
    d.table("VPORT", None, |_, _| vec![]);
    d.table("LTYPE", None, |d, t| {
        [("ByBlock", ""), ("ByLayer", ""), ("Continuous", "Solid line")]
            .iter()
            .map(|(name, text)| {
                d.record("LTYPE", t, "AcDbLinetypeTableRecord", name, &[(3, text), (72, "65"), (73, "0"), (40, "0.0")], 5)
            })
            .collect()
    });
    let layers: Vec<String> = if layer == "0" { vec!["0".into()] } else { vec!["0".into(), layer.clone()] };
    d.table("LAYER", None, |d, t| {
        layers
            .iter()
            .map(|name| d.record("LAYER", t, "AcDbLayerTableRecord", name, &[(62, "7"), (6, "Continuous"), (370, "-3")], 5))
            .collect()
    });
    d.table("STYLE", None, |d, t| {
        let extra = [(40, "0.0"), (41, "1.0"), (50, "0.0"), (71, "0"), (42, "2.5"), (3, "txt"), (4, "")];
        vec![d.record("STYLE", t, "AcDbTextStyleTableRecord", "Standard", &extra, 5)]
    });
    d.table("VIEW", None, |_, _| vec![]);
    d.table("UCS", None, |_, _| vec![]);
    d.table("APPID", None, |d, t| vec![d.record("APPID", t, "AcDbRegAppTableRecord", "ACAD", &[], 5)]);
    d.table("DIMSTYLE", Some("AcDbDimStyleTable"), |d, t| {
        vec![d.record("DIMSTYLE", t, "AcDbDimStyleTableRecord", "Standard", &[], 105)]
    });
    let records = d.table("BLOCK_RECORD", None, |d, t| {
        ["*Model_Space", "*Paper_Space"].iter().map(|name| d.record("BLOCK_RECORD", t, "AcDbBlockTableRecord", name, &[], 5)).collect()
    });
    let (model, paper) = (records[0].clone(), records[1].clone());
    d.pair(0, "ENDSEC");
    d.pair(0, "SECTION");
    d.pair(2, "BLOCKS");
    for (owner, name, paper_space) in [(&model, "*Model_Space", false), (&paper, "*Paper_Space", true)] {
        let h = d.handle();
        d.pair(0, "BLOCK");
        d.pair(5, h);
        d.pair(330, owner);
        d.pair(100, "AcDbEntity");
        if paper_space {
            d.pair(67, 1);
        }
        for (code, value) in [(8, "0"), (100, "AcDbBlockBegin"), (2, name), (70, "0"), (10, "0.0"), (20, "0.0"), (30, "0.0"), (3, name), (1, ""), (0, "ENDBLK")] {
            d.pair(code, value);
        }
        let h = d.handle();
        d.pair(5, h);
        d.pair(330, owner);
        d.pair(100, "AcDbEntity");
        if paper_space {
            d.pair(67, 1);
        }
        d.pair(8, "0");
        d.pair(100, "AcDbBlockEnd");
    }
    d.pair(0, "ENDSEC");
    d.pair(0, "SECTION");
    d.pair(2, "ENTITIES");
    for ((knots, control), curve) in data.iter().zip(curves) {
        let h = d.handle();
        d.pair(0, "SPLINE");
        d.pair(5, h);
        d.pair(330, &model);
        d.pair(100, "AcDbEntity");
        d.pair(8, &layer);
        d.pair(100, "AcDbSpline");
        for (code, value) in [(210, "0.0"), (220, "0.0"), (230, "1.0"), (70, "8")] {
            d.pair(code, value);
        }
        d.pair(71, curve.degree);
        d.pair(72, knots.len());
        d.pair(73, control.len());
        d.pair(74, 0);
        d.pair(42, "1e-10");
        d.pair(43, "1e-10");
        for &k in knots {
            d.pair(40, shortest(k));
        }
        for c in control {
            d.pair(10, shortest(c[0]));
            d.pair(20, shortest(c[1]));
            d.pair(30, shortest(c[2]));
        }
    }
    d.pair(0, "ENDSEC");
    d.pair(0, "SECTION");
    d.pair(2, "OBJECTS");
    let (root, group) = (d.handle(), d.handle());
    for (code, value) in [(0, "DICTIONARY"), (5, root.as_str()), (330, "0"), (100, "AcDbDictionary"), (281, "1"), (3, "ACAD_GROUP"), (350, group.as_str())] {
        d.pair(code, value);
    }
    for (code, value) in [(0, "DICTIONARY"), (5, group.as_str()), (330, root.as_str()), (100, "AcDbDictionary"), (281, "1"), (0, "ENDSEC"), (0, "EOF")] {
        d.pair(code, value);
    }
    let body = std::mem::take(&mut d.lines);
    let seed = d.handle();
    for (code, value) in [(0, "SECTION"), (2, "HEADER"), (9, "$ACADVER"), (1, "AC1015"), (9, "$HANDSEED"), (5, seed.as_str()), (9, "$INSUNITS")] {
        d.pair(code, value);
    }
    d.pair(70, insunits);
    d.pair(0, "ENDSEC");
    d.lines.extend(body);
    Ok(d.lines.join("\n") + "\n")
}

/// Arc length of a curve between `a` and `b` (its domain by default) by
/// 16-point Gauss-Legendre quadrature on every knot span.
pub fn arc_length(curve: &BSpline) -> f64 {
    let (lo, hi) = curve.domain();
    let mut breaks: Vec<f64> = curve.knots.iter().copied().filter(|&k| k > lo && k < hi).collect();
    breaks.push(lo);
    breaks.push(hi);
    breaks.sort_by(f64::total_cmp);
    breaks.dedup();
    let (nodes, weights) = gauss_legendre(16);
    breaks.windows(2).flat_map(|w| on_interval(&nodes, &weights, w[0], w[1]).collect::<Vec<_>>()).map(|(u, w)| w * curve.derivative(u, 1).norm()).sum()
}

/// One B-spline on `[0, 1]` reproducing a position-continuous chain.
///
/// Parameter ranges are scaled so the parametric speed is equal on both
/// sides of every tangent-continuous junction, which makes the merged curve
/// C1 there; junction knots then keep only the multiplicity the actual
/// continuity needs, so CAD readers do not split the curve.
pub fn merge_curves(curves: &[BSpline]) -> Result<BSpline> {
    let p = curves[0].degree;
    if curves.iter().any(|c| c.degree != p) {
        bail!("All sections must have the same degree to be merged.");
    }
    let mut lengths = vec![arc_length(&curves[0])];
    for w in curves.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let (alo, ahi) = a.domain();
        let (blo, bhi) = b.domain();
        let da = a.derivative(ahi, 1) * (ahi - alo);
        let db = b.derivative(blo, 1) * (bhi - blo);
        let (na, nb) = (da.norm(), db.norm());
        let smooth = na > 0.0 && nb > 0.0 && (da / na - db / nb).norm() < 1e-7;
        let last = lengths[lengths.len() - 1];
        lengths.push(last * if smooth { nb / na } else { arc_length(b) / arc_length(a) });
    }
    let total: f64 = lengths.iter().sum();
    let mut bounds = vec![0.0];
    let mut sum = 0.0;
    for l in &lengths {
        sum += l;
        bounds.push(sum / total);
    }
    let last = curves.len() - 1;
    let mut knots: Vec<f64> = Vec::new();
    let mut control: Vec<Vec2> = Vec::new();
    for (s, curve) in curves.iter().enumerate() {
        let t = &curve.knots;
        let (lo, hi) = curve.domain();
        let mut mapped: Vec<f64> = t.iter().map(|v| bounds[s] + (v - lo) / (hi - lo) * (bounds[s + 1] - bounds[s])).collect();
        let n = mapped.len();
        mapped[..=p].iter_mut().for_each(|v| *v = bounds[s]);
        mapped[n - p - 1..].iter_mut().for_each(|v| *v = bounds[s + 1]);
        // Drop one end knot at every junction so it keeps multiplicity p.
        let head = if s == 0 { 0 } else { p + 1 };
        let tail = if s == last { n } else { n - 1 };
        knots.extend_from_slice(&mapped[head..tail]);
        let mut c = curve.control_points.clone();
        if s > 0 {
            let previous = control.last_mut().unwrap();
            *previous = 0.5 * (*previous + c[0]);
            c.remove(0);
        }
        control.extend(c);
    }
    let n = knots.len();
    knots[..=p].iter_mut().for_each(|v| *v = 0.0);
    knots[n - p - 1..].iter_mut().for_each(|v| *v = 1.0);
    let mut merged = BSpline { degree: p, knots, control_points: control };
    for &u in &bounds[1..bounds.len() - 1] {
        for _ in 0..p.saturating_sub(1) {
            match remove_knot(&merged, u) {
                Some(reduced) => merged = reduced,
                None => break,
            }
        }
    }
    Ok(merged)
}

/// The curve without one occurrence of knot `u`, if it stays the same
/// within 1e-10 relative.
fn remove_knot(curve: &BSpline, u: f64) -> Option<BSpline> {
    let k = curve.degree;
    let t = &curve.knots;
    let i = t.iter().position(|&v| v == u)?;
    let mut new_t = t.clone();
    new_t.remove(i);
    let breaks = distinct(&new_t[k..new_t.len() - k]);
    let nodes = crate::airfoil::cosine_spacing(2 * k + 3);
    let mut x: Vec<f64> = breaks.windows(2).flat_map(|w| nodes.iter().map(move |f| w[0] + (w[1] - w[0]) * f)).collect();
    x.sort_by(f64::total_cmp);
    x.dedup();
    let a = basis_matrix(&x, &new_t, k, 0);
    let target = curve.points(&x);
    let b = DMatrix::from_fn(x.len(), 2, |i, c| target[i][c]);
    let mut control = matrix_to_points(&lstsq(&a, &b));
    let candidate = BSpline { degree: k, knots: new_t.clone(), control_points: control.clone() };
    let check = linspace(new_t[0], new_t[new_t.len() - 1], 20 * new_t.len());
    let scale = target.iter().fold(1.0f64, |m, p| m.max(p.x.abs()).max(p.y.abs()));
    let error = check.iter().map(|&v| {
        let d = candidate.point(v) - curve.point(v);
        d.x.abs().max(d.y.abs())
    }).fold(0.0, f64::max);
    if error > 1e-10 * scale {
        return None;
    }
    let n = control.len();
    control[0] = curve.control_points[0];
    control[n - 1] = curve.control_points[t.len() - k - 2];
    Some(BSpline { degree: k, knots: new_t, control_points: control })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_reals() {
        assert_eq!(step_real(1e-7), "1.E-7");
        assert_eq!(step_real(3.0), "3.0");
        assert_eq!(step_real(0.5), "0.5");
        assert_eq!(step_real(2.5e16), "2.5E16");
        assert_eq!(step_real(-1e16), "-1.E16");
    }

    #[test]
    fn step_strings() {
        assert_eq!(step_string("it's"), "'it''s'");
        assert_eq!(step_string("éé"), "'\\X2\\00E9\\X0\\\\X2\\00E9\\X0\\'");
    }

    #[test]
    fn iges_records_are_80_columns() {
        let c = BSpline::uniform(3, (0..7).map(|i| Vec2::new(i as f64, (i * i) as f64 * 0.1)).collect()).unwrap();
        let text = to_iges(&[c.clone(), c], "naca", 1.0, Units::Millimetre).unwrap();
        assert!(text.lines().all(|l| l.len() == 80));
        assert!(dxf_name("a:b").is_err());
        assert_eq!(dxf_name("Läyer").unwrap(), "L\\U+00E4yer");
    }

    #[test]
    fn merge_keeps_geometry() {
        let a = BSpline::uniform(3, vec![Vec2::new(0., 0.), Vec2::new(1., 1.), Vec2::new(2., 1.), Vec2::new(3., 0.)]).unwrap();
        let b = BSpline::uniform(3, vec![Vec2::new(3., 0.), Vec2::new(4., -1.), Vec2::new(5., -1.), Vec2::new(6., 0.)]).unwrap();
        let merged = merge_curves(&[a.clone(), b.clone()]).unwrap();
        for v in linspace(0.0, 1.0, 11) {
            let feet = crate::diagnostics::project(&[a.point(v), b.point(v)], &merged);
            assert!(feet.iter().all(|(_, d)| *d < 1e-9));
        }
    }
}
