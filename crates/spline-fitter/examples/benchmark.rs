//! Times the optimizers on NACA airfoils: the cases of the Python
//! version's `scripts/benchmark_new_optimizers.py`, plus the app's default
//! fit (each run several times; the median time is printed), then tolerance
//! searches on a finely sampled airfoil.
//!
//! cargo run --release -p spline-fitter --example benchmark

use std::time::Instant;

use spline_fitter::bspline_varpro::airfoil::Naca4;
use spline_fitter::bspline_varpro::KnotPlacement;
use spline_fitter::geometry::AirfoilData;
use spline_fitter::models::{ContinuityMode, OptimizationConfig, OptimizerType, SplineConfig};
use spline_fitter::optimize::{run_optimization, Unmonitored};
use spline_fitter::Vec2;

fn airfoil(code: &str, per_side: usize) -> AirfoilData {
    let points: Vec<Vec2> =
        Naca4::new(code, true).unwrap().points(per_side).iter().map(|p| p.map(|v| (v * 1e6).round_ties_even() / 1e6)).collect();
    AirfoilData::new(points, 2, &format!("NACA{code}.dat"), &format!("NACA{code}"), 6).unwrap()
}

fn main() {
    let benchmark = SplineConfig { num_control_points: 20, continuity: ContinuityMode::G1, ..Default::default() };
    let cases = [
        ("NACA0012 G1 20", "0012", 80, benchmark.clone()),
        ("NACA2412 G1 20", "2412", 80, benchmark.clone()),
        ("NACA4415 G1 20", "4415", 100, benchmark),
        ("NACA2412 G3 25 (app default)", "2412", 81, SplineConfig::default()),
    ];
    for (name, code, per_side, config) in cases {
        let data = airfoil(code, per_side);
        for optimizer in [OptimizerType::DirectLsq, OptimizerType::VarPro] {
            let options = OptimizationConfig { optimizer, ..Default::default() };
            let mut times = Vec::new();
            let mut result = None;
            for _ in 0..7 {
                let start = Instant::now();
                result = Some(run_optimization(&config, &options, &data, &mut Unmonitored).unwrap());
                times.push(start.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(f64::total_cmp);
            let r = result.unwrap();
            println!("{name:30} | {:12} | {:9.2} ms | error {:.3e} | evals {}", optimizer.display_name(), times[3], r.error, r.evaluations);
        }
    }
    // Tolerance searches on a finely sampled airfoil (130 points a side, the
    // size of a measured section): thousands of evaluations, where the cost
    // of one solver step shows. Run once each.
    let fine = airfoil("2412", 130);
    let grow = |sections| SplineConfig { num_control_points: 6, num_sections: sections, ..Default::default() };
    let tolerance =
        |knot_placement| OptimizationConfig { tolerance: Some(1e-5), grow_control_points: true, knot_placement, ..Default::default() };
    let searches = [
        ("NACA2412 259 pts, 1 section", grow(1), tolerance(KnotPlacement::Uniform)),
        ("NACA2412 259 pts, 2 sections", grow(2), tolerance(KnotPlacement::Uniform)),
        ("NACA2412 259 pts, 2 sec, free", grow(2), tolerance(KnotPlacement::Free)),
    ];
    for (name, config, options) in searches {
        let data = AirfoilData::new(fine.coordinates.clone(), config.num_sections, "NACA2412.dat", "NACA2412", 6).unwrap();
        let start = Instant::now();
        let r = run_optimization(&config, &options, &data, &mut Unmonitored).unwrap();
        let seconds = start.elapsed().as_secs_f64();
        let counts: Vec<usize> = r.section_control_points.iter().map(Vec::len).collect();
        println!("{name:30} | tolerance    | {:9.2} s  | error {:.3e} | evals {} | control points {counts:?}", seconds, r.error, r.evaluations);
    }
}
