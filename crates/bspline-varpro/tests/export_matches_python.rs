//! The CAD writers produce the Python version's files byte for byte, apart
//! from the time stamp. Fixtures: tests/data/python_export (make_fixtures.py
//! writes them with the Python exporter).

use std::path::Path;

use bspline_varpro::export::{to_dxf, to_iges, to_step, Units};
use bspline_varpro::{BSpline, Vec2};

/// Two degree-5 sections of 10 control points with uneven knots, the second
/// starting at the first one's last control point.
fn curves() -> Vec<BSpline> {
    let mut curves: Vec<BSpline> = (0..2)
        .map(|s| {
            let (n, k) = (10usize, 5usize);
            let mut knots = vec![0.0; k + 1];
            knots.extend((0..n - k - 1).map(|i| 0.13 + 0.1 * i as f64 + 0.01 * s as f64));
            knots.extend(vec![1.0; k + 1]);
            let control_points = (0..n)
                .map(|i| {
                    Vec2::new((0.3 * i as f64 + s as f64).cos() * (1.0 + 0.1 * i as f64), (0.7 * i as f64).sin() / 3.0 + s as f64 * 0.05)
                })
                .collect();
            BSpline { degree: k, knots, control_points }
        })
        .collect();
    curves[1].control_points[0] = *curves[0].control_points.last().unwrap();
    curves
}

/// The text with its line ends as `\n` and every time stamp (STEP
/// `'YYYY-MM-DDTHH:MM:SS'`, IGES `15HYYYYMMDD.HHMMSS`) replaced by `#`s.
fn without_stamps(text: &str) -> String {
    let mut chars: Vec<char> = text.replace("\r\n", "\n").chars().collect();
    let digit_or = |c: char, extra: &[char]| c.is_ascii_digit() || extra.contains(&c);
    let mut i = 0;
    while i < chars.len() {
        // STEP: 19 characters in quotes, digits with '-', 'T' and ':'.
        if chars[i] == '\''
            && i + 20 < chars.len()
            && chars[i + 20] == '\''
            && chars[i + 1..i + 20].iter().all(|&c| digit_or(c, &['-', 'T', ':']))
            && chars[i + 11] == 'T'
        {
            chars[i + 1..i + 20].iter_mut().for_each(|c| *c = '#');
            i += 21;
            continue;
        }
        // IGES: 15H then 15 characters of digits and one '.'.
        if chars[i..].starts_with(&['1', '5', 'H']) && i + 18 <= chars.len() && chars[i + 3..i + 18].iter().all(|&c| digit_or(c, &['.'])) {
            chars[i + 3..i + 18].iter_mut().for_each(|c| *c = '#');
            i += 18;
            continue;
        }
        i += 1;
    }
    chars.into_iter().collect()
}

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/python_export").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn assert_same(name: &str, ours: String) {
    let (ours, theirs) = (without_stamps(&ours), without_stamps(&fixture(name)));
    for (line, (a, b)) in ours.lines().zip(theirs.lines()).enumerate() {
        assert_eq!(a, b, "{name}, line {}", line + 1);
    }
    assert_eq!(ours.lines().count(), theirs.lines().count(), "{name}: line count");
    assert_eq!(ours, theirs, "{name}: line ends");
}

#[test]
fn step_iges_and_dxf_match_the_python_files() {
    let curves = curves();
    for (tag, units) in [("mm", Units::Millimetre), ("in", Units::Inch)] {
        assert_same(&format!("two_sections_{tag}.step"), to_step(&curves, "Darkstar ä tip", 1000.0, units).unwrap());
        assert_same(&format!("two_sections_{tag}.igs"), to_iges(&curves, "Darkstar ä tip", 1000.0, units).unwrap());
        assert_same(&format!("two_sections_{tag}.dxf"), to_dxf(&curves, 1000.0, "Profil ä", units).unwrap());
    }
}

#[test]
fn stamps_are_masked_and_nothing_else() {
    let step = "FILE_NAME('a','2026-11-18T22:02:09',(''));";
    assert_eq!(without_stamps(step), "FILE_NAME('a','###################',(''));");
    let iges = "15H20261118.220209,1.0E-07,1.74E+03\r\n";
    assert_eq!(without_stamps(iges), "15H###############,1.0E-07,1.74E+03\n");
    assert_eq!(without_stamps("'1.5','x'"), "'1.5','x'");
}
