//! Times the optimizers on NACA airfoils: the cases of the Python
//! version's `scripts/benchmark_new_optimizers.py`, plus the app's default
//! fit. Each fit runs several times; the median time is printed.
//!
//! cargo run --release -p spline-fitter --example benchmark

use std::time::Instant;

use spline_fitter::bspline_varpro::airfoil::Naca4;
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
}
