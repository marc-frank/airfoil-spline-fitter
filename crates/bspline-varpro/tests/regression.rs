//! Reference fits of the specification (section 21, item 11).

use bspline_varpro::airfoil::{default_tolerance, Naca4};
use bspline_varpro::{fit, ChainOptions, Continuity, FitOptions, Junction, KnotPlacement, Silent, Vec2};

fn naca2412() -> Vec<Vec<Vec2>> {
    let points: Vec<Vec2> = Naca4::new("2412", true)
        .unwrap()
        .points(81)
        .iter()
        .map(|p| p.map(|v| (v * 1e6).round_ties_even() / 1e6))
        .collect();
    vec![points[..81].to_vec(), points[80..].to_vec()]
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs()
}

#[test]
fn varpro_g3_fixed_count() {
    let sections = naca2412();
    assert_eq!(default_tolerance(&sections.concat(), Some(6)), 1e-6);
    let options = FitOptions {
        chain: ChainOptions { continuity: Continuity::All(Junction::new(3, true, true).unwrap()), ..ChainOptions::default() },
        knot_placement: KnotPlacement::Uniform,
        ..FitOptions::default()
    };
    let result = fit(&sections, &[25, 25], 5, &options, &mut Silent).unwrap();
    assert_eq!(result.num_control_points(), vec![25, 25]);
    assert!(result.success, "{}", result.message);
    assert_eq!(result.message, "`ftol` termination condition is satisfied.");
    assert!(close(result.rms_error, 8.978e-7, 1e-3), "rms {}", result.rms_error);
    assert_eq!(result.nfev, 40);
}
