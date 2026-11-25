//! Times the parts of one Levenberg-Marquardt step on the app's default
//! problem (NACA 2412, 81 points per side, 2 × 25 control points, G3).
//!
//! cargo run --release -p bspline-varpro --example profile_parts

use web_time::Instant;

use bspline_varpro::airfoil::Naca4;
use bspline_varpro::reparam::{apply_eta_jacobian, t_to_eta};
use bspline_varpro::{open_uniform_knots, ChainOptions, Continuity, Junction, Vec2};

fn time<T>(name: &str, repeats: usize, mut f: impl FnMut(usize) -> T) -> T {
    let mut out = f(0);
    let start = Instant::now();
    for i in 1..=repeats {
        out = f(i);
    }
    println!("{name:28} {:8.3} ms", start.elapsed().as_secs_f64() * 1e3 / repeats as f64);
    out
}

fn main() {
    // Upper and lower surface, sharing the leading edge.
    let points: Vec<Vec2> = Naca4::new("2412", true).unwrap().points(81);
    let sections = vec![points[..81].to_vec(), points[80..].to_vec()];
    let options = ChainOptions { continuity: Continuity::All(Junction { order: 3, geometric: true, fixed: true }), ..Default::default() };
    let knots = vec![open_uniform_knots(5, 25).unwrap(); 2];
    let problem = options.problem(&sections, knots, 5).unwrap();
    let u0 = problem.initial_parameters();
    let theta = problem.theta.clone();
    // A fresh u every call, so the cached KKT solve is never reused.
    let u_at = |i: usize| {
        let mut u = u0.clone();
        let last = u.len() - 2;
        u[1 + i % last] += 1e-9;
        u
    };
    println!("points {}, residuals {}, controls {}, θ {}", problem.num_points(), problem.num_residuals(), problem.total_controls(), theta.len());
    let r = 50;
    time("residual (KKT solve)", r, |i| problem.residual(&u_at(i), &theta).0);
    let u = u_at(0);
    let _ = problem.residual(&u, &theta);
    let ju = time("∂r/∂u (cached KKT)", r, |_| problem.jacobian(&u, &theta));
    time("∂r/∂θ (cached KKT)", r, |_| problem.parameter_jacobian(&u, &theta));
    let eta = t_to_eta(&u);
    let s = problem.num_sections() as f64;
    let j = time("apply ∂u/∂η", r, |_| apply_eta_jacobian(&ju, &eta, &u, s));
    let a = time("JᵀJ", r, |_| j.transpose() * &j);
    time("Jᵀ r", r, |_| j.transpose() * nalgebra::DVector::from_element(j.nrows(), 1.0));
    time("cholesky + solve", r, |_| a.clone().cholesky().map(|c| c.solve(&nalgebra::DVector::from_element(a.nrows(), 1.0))));
    time("JᵀJ via tr_mul", r, |_| j.tr_mul(&j));
}
