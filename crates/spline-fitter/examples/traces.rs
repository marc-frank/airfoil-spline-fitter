//! Prints the error of every evaluation of VarPro fits on NACA airfoils,
//! marking the evaluations that are worse than the best before them.

use spline_fitter::bspline_varpro::airfoil::Naca4;
use spline_fitter::geometry::AirfoilData;
use spline_fitter::models::{ContinuityMode, OptimizationConfig, SplineConfig};
use spline_fitter::optimize::{run_optimization, Monitor, Update};
use spline_fitter::Vec2;

struct Trace(Vec<f64>);

impl Monitor for Trace {
    fn update(&mut self, update: &Update) -> bool {
        self.0.push(update.error);
        false
    }
}

fn main() {
    for code in ["0012", "2412", "4415", "6409", "2421"] {
        for (cp, continuity) in [(25, ContinuityMode::G3), (20, ContinuityMode::G2), (30, ContinuityMode::G3)] {
            let points: Vec<Vec2> =
                Naca4::new(code, true).unwrap().points(81).iter().map(|p| p.map(|v| (v * 1e6).round_ties_even() / 1e6)).collect();
            let data = AirfoilData::new(points, 2, "x", "x", 6).unwrap();
            let config = SplineConfig { num_control_points: cp, continuity, ..Default::default() };
            let mut trace = Trace(vec![]);
            run_optimization(&config, &OptimizationConfig::default(), &data, &mut trace).unwrap();
            let mut best = f64::INFINITY;
            let worse: Vec<String> = trace
                .0
                .iter()
                .enumerate()
                .filter_map(|(i, &e)| {
                    let out = (e > 2.0 * best).then(|| format!("{}:{:.1e}", i + 1, e));
                    best = best.min(e);
                    out
                })
                .collect();
            println!("NACA{code} {cp} {continuity:?} evals {} final {:.4e} worse {}", trace.0.len(), best, worse.join(" "));
        }
    }
}
