//! Reference fits of the specification (section 21, item 11).

use bspline_varpro::airfoil::{default_tolerance, Naca4};
use bspline_varpro::{fit, ChainOptions, Continuity, FitOptions, Junction, KnotPlacement, Silent, Vec2};

fn naca2412() -> Vec<Vec<Vec2>> {
    let points: Vec<Vec2> =
        Naca4::new("2412", true).unwrap().points(81).iter().map(|p| p.map(|v| (v * 1e6).round_ties_even() / 1e6)).collect();
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

/// About a minute; run with `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn varpro_g2_tolerance_search() {
    use bspline_varpro::{fit_to_tolerance, FitResult, Observer, ToleranceOptions};

    struct Refits(usize);
    impl Observer for Refits {
        fn refit(&mut self, _: &FitResult, _: f64) -> bool {
            self.0 += 1;
            false
        }
    }

    let sections = naca2412();
    let options = ToleranceOptions {
        chain: ChainOptions {
            continuity: Continuity::All(Junction::new(2, true, true).unwrap()),
            smoothing: 1e-7,
            ..ChainOptions::default()
        },
        max_points_per_section: None,
        ..ToleranceOptions::default()
    };
    let mut refits = Refits(0);
    let result = fit_to_tolerance(&sections, 1e-6, 5, &options, &mut refits).unwrap();
    assert!(result.success, "{}", result.message);
    assert_eq!(result.num_control_points(), vec![10, 11]);
    assert_eq!(refits.0, 17);
    // Every refit matches the reference to about nine digits; the capped
    // knot polishes (60 evaluations, finite-difference knot columns) amplify
    // rounding differences of the linear algebra to a few tenths of a
    // percent in the final deviation.
    assert!(close(result.max_deviation.unwrap(), 9.1121e-7, 1e-2));
}
