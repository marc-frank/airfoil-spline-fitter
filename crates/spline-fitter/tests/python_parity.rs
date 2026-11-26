//! Outputs of the Python implementation (frozen in `fixtures/`) reproduced:
//! the airfoil reader on fuzzed files, and the CAD, spline and airfoil
//! files byte for byte (timestamps aside).

use serde_json::Value;
use spline_fitter::bspline_varpro::export::{merge_curves, to_dxf, to_iges, to_step, Units};
use spline_fitter::io::{export_airfoil_data, export_spline_data, parse_airfoil_text, SplineFile};
use spline_fitter::{BSpline, Vec2};

fn points(v: &Value) -> Vec<Vec2> {
    v.as_array().unwrap().iter().map(|p| Vec2::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap())).collect()
}

fn curve(v: &Value) -> BSpline {
    BSpline {
        degree: v["degree"].as_u64().unwrap() as usize,
        knots: v["knots"].as_array().unwrap().iter().map(|k| k.as_f64().unwrap()).collect(),
        control_points: points(&v["control_points"]),
    }
}

#[test]
fn airfoil_reader_matches_python() {
    let cases: Vec<Value> = serde_json::from_str(include_str!("fixtures/airfoil_texts.json")).unwrap();
    for (i, case) in cases.iter().enumerate() {
        let text = case["text"].as_str().unwrap();
        let parsed = parse_airfoil_text(text, Some("file"));
        match (&case["coords"], parsed) {
            (Value::Null, None) => {}
            (Value::Null, Some(p)) => panic!("case {i}: Python found no coordinates, Rust {} points\n{text}", p.coords.len()),
            (_, None) => panic!("case {i}: Rust found no coordinates\n{text}"),
            (expected, Some(p)) => {
                assert_eq!(p.name.as_deref(), case["name"].as_str(), "case {i}\n{text}");
                // Rust drops repeated and stray points, which Python keeps:
                // the points as read are Python's.
                let (coords, precision) = p.as_read();
                if !p.cleanup.any() {
                    assert_eq!((&coords, precision), (&p.coords, p.precision), "case {i}\n{text}");
                }
                assert_eq!(precision as i64, case["precision"].as_i64().unwrap(), "case {i}\n{text}");
                let expected = points(expected);
                assert_eq!(coords.len(), expected.len(), "case {i}\n{text}");
                for (a, b) in coords.iter().zip(&expected) {
                    assert!((a - b).norm() <= 1e-12 * (1.0 + b.norm()), "case {i}: {a:?} != {b:?}\n{text}");
                }
            }
        }
    }
}

fn exports() -> Value {
    serde_json::from_str(include_str!("fixtures/exports.json")).unwrap()
}

fn without_stamps(text: &str) -> String {
    let re = regex::Regex::new(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d|\d{8}\.\d{6}").unwrap();
    re.replace_all(text, "STAMP").into_owned()
}

#[test]
fn cad_files_match_python() {
    let data = exports();
    let curves: Vec<BSpline> = data["curves"].as_array().unwrap().iter().map(curve).collect();
    for (name, units) in [("mm", Units::Millimetre), ("m", Units::Metre), ("inch", Units::Inch)] {
        let step = without_stamps(&to_step(&curves, "NACA 2412 é", 1.5, units).unwrap());
        assert_eq!(step, data[format!("step_{name}")].as_str().unwrap(), "STEP {name}");
        let iges = without_stamps(&to_iges(&curves, "NACA 2412 é", 1.5, units).unwrap());
        assert_eq!(iges, data[format!("iges_{name}")].as_str().unwrap(), "IGES {name}");
        let dxf = to_dxf(&curves, 1.5, "Profil Ä", units).unwrap();
        assert_eq!(dxf, data[format!("dxf_{name}")].as_str().unwrap(), "DXF {name}");
    }
    let merged = merge_curves(&curves).unwrap();
    let expected = curve(&data["merged"]);
    assert_eq!(merged.knots.len(), expected.knots.len());
    for (a, b) in merged.knots.iter().zip(&expected.knots) {
        assert!((a - b).abs() < 1e-12);
    }
    for (a, b) in merged.control_points.iter().zip(&expected.control_points) {
        assert!((a - b).norm() < 1e-9);
    }
}

#[test]
fn spline_and_airfoil_files_match_python() {
    let data = exports();
    let curves: Vec<BSpline> = data["curves"].as_array().unwrap().iter().map(curve).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.spline.dat");
    let spline = SplineFile {
        name: "#NACA 2412".into(),
        degree: Some(5),
        control_points: curves.iter().map(|c| c.control_points.clone()).collect(),
        knots: Some(curves.iter().map(|c| c.knots.clone()).collect()),
        continuity: Some(vec!["G3".into()]),
    };
    export_spline_data(&path, &spline, Some(12)).unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), data["spline_file"].as_str().unwrap());
    let dat = dir.path().join("a.dat");
    let pts: Vec<Vec2> = spline_fitter::bspline_varpro::airfoil::Naca4::new("2412", true)
        .unwrap()
        .points(41)
        .iter()
        .map(|p| p.map(|v| (v * 1e6).round_ties_even() / 1e6))
        .collect();
    export_airfoil_data(&dat, &pts, Some("NACA 2412"), Some(6)).unwrap();
    assert_eq!(std::fs::read_to_string(&dat).unwrap(), data["dat_file"].as_str().unwrap());
}

#[test]
fn projects_round_trip_with_python() {
    use spline_fitter::project::{parse_project, project_text, save_project};
    let text = include_str!("fixtures/python.asfproj");
    let loaded = parse_project(text).unwrap();
    assert!(loaded.warnings.is_empty());
    let project = &loaded.project;
    assert_eq!(project.runs.len(), 2);
    assert_eq!(project.runs.iter().map(|r| r.number).collect::<Vec<_>>(), vec![1, 2]);
    let report = project.runs[0].result.report.as_ref().unwrap();
    assert!(report.fairness.as_ref().unwrap().leading_edge_radius.is_some());
    assert!(report.shape.is_some());
    // Rust writes what it read, field for field and in Python's layout.
    let mut again = parse_project(&project_text(project)).unwrap().project;
    assert_eq!(&again, project);
    assert_eq!(project_text(project), text.replace("\r\n", "\n").trim_end());
    if let Some(path) = std::env::var_os("SPLINE_FITTER_PROJECT_OUT") {
        save_project(&mut again, std::path::Path::new(&path)).unwrap();
    }
}
