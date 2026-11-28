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
    // 33 with the junction written from its ending section's side (the
    // Python version); the same fit.
    assert_eq!(result.nfev, 31);
}

/// A fit does not depend on the end its points are listed from, so the
/// fit of a symmetric outline is symmetric: nothing asks for it. Here to
/// rounding, for geometric and parametric junctions of every order.
#[test]
fn a_symmetric_outline_gets_a_symmetric_curve() {
    let points: Vec<Vec2> = Naca4::new("0010", true).unwrap().points(81).iter().map(|p| p.map(|v| (v * 1e6).round_ties_even() / 1e6)).collect();
    let sections = vec![points[..=80].to_vec(), points[80..].to_vec()];
    for geometric in [true, false] {
        for order in 0..=3 {
            let chain = ChainOptions { continuity: Continuity::All(Junction::new(order, geometric, true).unwrap()), ..ChainOptions::default() };
            let result = fit(&sections, &[25, 25], 5, &FitOptions { chain, ..FitOptions::default() }, &mut Silent).unwrap();
            let (upper, lower) = (&result.control_points[0], &result.control_points[1]);
            let apart = (0..25).map(|i| (Vec2::new(upper[24 - i].x, -upper[24 - i].y) - lower[i]).norm()).fold(0.0, f64::max);
            assert!(apart < 1e-10, "order {order}, geometric {geometric}: {apart:e}");
            // The speed ratio and the third-order shift are odd in the
            // direction: zero at a symmetric junction, as far as the fit converges.
            let odd = result.theta.iter().step_by(2).fold(0.0f64, |m, t| m.max(t.abs()));
            assert!(odd < 1e-6, "order {order}: {:?}", result.theta);
        }
    }
}

/// The same curve whichever end a cambered outline is listed from, as far
/// as the fits converge.
#[test]
fn the_direction_of_the_points_does_not_matter() {
    let points: Vec<Vec2> = Naca4::new("2412", true).unwrap().points(81).iter().map(|p| p.map(|v| (v * 1e6).round_ties_even() / 1e6)).collect();
    let reversed: Vec<Vec2> = points.iter().rev().copied().collect();
    let cut = |p: &[Vec2]| vec![p[..=80].to_vec(), p[80..].to_vec()];
    let chain = ChainOptions { continuity: Continuity::All(Junction::new(3, true, true).unwrap()), ..ChainOptions::default() };
    let options = FitOptions { chain, ..FitOptions::default() };
    let one = fit(&cut(&points), &[25, 25], 5, &options, &mut Silent).unwrap();
    let other = fit(&cut(&reversed), &[25, 25], 5, &options, &mut Silent).unwrap();
    assert_eq!(one.nfev, other.nfev);
    let back: Vec<Vec2> = other.control_points.iter().rev().flat_map(|s| s.iter().rev().copied()).collect();
    let apart = one.control_points.iter().flatten().zip(&back).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
    assert!(apart < 1e-6, "{apart:e}");
    // Speed ratio and third-order shift change sign, the second-order one stays.
    let (t, r) = (&one.theta, &other.theta);
    assert!((t[0] + r[0]).abs() < 1e-5 && (t[1] - r[1]).abs() < 1e-3 * t[1].abs() && (t[2] + r[2]).abs() < 1e-3 * t[2].abs(), "{t:?} {r:?}");
}

fn g2(smoothing: f64) -> ChainOptions {
    ChainOptions { continuity: Continuity::All(Junction::new(2, true, true).unwrap()), smoothing, ..ChainOptions::default() }
}

/// Free knots are unknowns of the one Levenberg-Marquardt loop. About 40
/// seconds; run with `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn varpro_g2_free_knots() {
    let sections = naca2412();
    let options = FitOptions { chain: g2(1e-7), knot_placement: KnotPlacement::Free, ..FitOptions::default() };
    let result = fit(&sections, &[12, 12], 5, &options, &mut Silent).unwrap();
    assert!(result.success, "{}", result.message);
    assert!(close(result.rms_error, 2.9394e-7, 1e-3), "rms {}", result.rms_error);
    // The first interior knot sits on the minimum gap.
    assert!(close(result.knots[0][6], 0.05, 1e-4), "{:?}", result.knots[0]);

    let grown = fit(&sections, &[12, 12], 5, &FitOptions { grow: true, ..options }, &mut Silent).unwrap();
    assert_eq!(grown.num_control_points(), vec![12, 12]);
    assert!(close(grown.rms_error, 3.2869e-7, 1e-3), "rms {}", grown.rms_error);
}

struct Refits(Vec<(Vec<usize>, f64)>);

impl bspline_varpro::Observer for Refits {
    fn refit(&mut self, result: &bspline_varpro::FitResult, deviation: f64) -> bool {
        self.0.push((result.num_control_points(), deviation));
        false
    }
}

fn search(placement: KnotPlacement) -> (bspline_varpro::FitResult, Vec<(Vec<usize>, f64)>) {
    use bspline_varpro::{fit_to_tolerance, ToleranceOptions};
    let options = ToleranceOptions { chain: g2(1e-7), knot_placement: placement, ..ToleranceOptions::default() };
    let mut refits = Refits(vec![]);
    let result = fit_to_tolerance(&naca2412(), 1e-6, 5, &options, &mut refits).unwrap();
    assert!(result.success, "{}", result.message);
    (result, refits.0)
}

/// About 20 seconds; run with `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn varpro_g2_tolerance_search_uniform() {
    let (result, refits) = search(KnotPlacement::Uniform);
    let counts: Vec<Vec<usize>> = refits.iter().map(|r| r.0.clone()).collect();
    assert_eq!(counts, [[6, 6], [7, 7], [9, 9], [13, 12], [16, 14], [18, 15], [20, 16], [20, 18], [19, 18]]);
    assert_eq!(result.num_control_points(), vec![19, 18]);
    assert!(close(result.max_deviation.unwrap(), 9.7673e-7, 1e-3), "{:?}", result.max_deviation);
}

/// Specification 21, item 11: about three minutes; run with
/// `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn varpro_g2_tolerance_search_free() {
    let (result, refits) = search(KnotPlacement::Free);
    let counts: Vec<Vec<usize>> = refits.iter().map(|r| r.0.clone()).collect();
    assert_eq!(counts, [[6, 6], [7, 7], [9, 9], [11, 10], [13, 11], [13, 12], [12, 12], [11, 12], [11, 11]]);
    assert_eq!(result.num_control_points(), vec![11, 11]);
    // The finite-difference knot columns amplify rounding differences of the
    // linear algebra over a few thousand evaluations: the deviations match
    // the reference to about 0.2 %.
    assert!(close(result.max_deviation.unwrap(), 7.9338e-7, 1e-2), "{:?}", result.max_deviation);
}
