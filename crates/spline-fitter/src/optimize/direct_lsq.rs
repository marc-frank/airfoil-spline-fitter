//! Direct LSQ: one constrained linear least-squares solve on fixed data
//! parameters, then a joint refinement of parameters and control points.
//!
//! The chain is fitted as a whole with parametric continuity: the ends are
//! fixed, the sections joined and, unless the junctions are free, pinned to
//! their split points. After the linear solve every point slides to its
//! closest chain point (and may cross a junction).

use bspline_varpro::basis::SparseRows;
use bspline_varpro::chain::{chain_data, chain_parameters, evaluate_chain, locate};
use bspline_varpro::{
    centripetal_parameters, open_uniform_knots, solve_joint, BSpline, ChainConstraints, End, JointProblem, Junction, KktSolver,
};
use nalgebra::DMatrix;

use super::{Monitor, RawResult, Update, UpdateKind};
use crate::models::{OptimizationConfig, Parameterization, SplineConfig};
use crate::{Result, Vec2};

/// Gauss-Newton iterations of each joint refinement.
const MAX_ITERATIONS: usize = 100;
/// Ridge of the linear solve.
const REGULARIZATION: f64 = 1e-10;

pub fn fit(config: &SplineConfig, options: &OptimizationConfig, sections: &[Vec<Vec2>], monitor: &mut dyn Monitor) -> Result<RawResult> {
    let p = config.degree;
    let n = config.num_control_points;
    let s = sections.len();
    let knots = vec![open_uniform_knots(p, n)?; s];
    let data = chain_data(sections);
    let alpha = match options.parameterization {
        Parameterization::Centripetal => 0.5,
        Parameterization::ArcLength => 1.0,
    };
    let local: Vec<Vec<f64>> = sections.iter().map(|sec| centripetal_parameters(sec, alpha)).collect();
    let mut u = chain_parameters(&local, sections);
    let free = options.optimize_le_position && s > 1;
    let constraints = |fixed: bool| -> Result<(DMatrix<f64>, DMatrix<f64>)> {
        let junction = Junction::new(config.continuity.order(), false, fixed)?;
        let chain = ChainConstraints::new(sections, &knots, p, vec![junction; s - 1], End::default(), End::default())?;
        let built = chain.build(&[]);
        Ok((built.c, built.d))
    };
    // With free junctions the fit starts from the pinned ones.
    let mut stages = vec![constraints(true)?];
    if free {
        stages.push(constraints(false)?);
    }

    let offsets: Vec<usize> = (0..=s).map(|k| k * n).collect();
    let basis = SparseRows::new(&u, &knots, &offsets, p, 0, |v| locate(v, s));
    let (c0, d0) = &stages[0];
    let rhs = basis.tr_mul(&DMatrix::from_fn(data.len(), 2, |i, k| data[i][k]));
    let mut control = KktSolver::new(&basis.gram(), c0, REGULARIZATION).solve(&rhs, Some(d0));
    let mut evaluations = 1;
    for (c, d) in &stages {
        let mut problem = JointProblem::new(sections, knots.clone(), p, c, d, 0.0, 0.0);
        let z0 = problem.z_from(&control);
        let solution = solve_joint(&mut problem, &u, z0, Some(MAX_ITERATIONS), 1e-12, &mut |_, _, _| false)?;
        u = solution.u;
        control = problem.control_points(&solution.z);
        evaluations += 1;
    }

    let section_control_points: Vec<Vec<Vec2>> =
        (0..s).map(|k| (k * n..(k + 1) * n).map(|i| Vec2::new(control[(i, 0)], control[(i, 1)])).collect()).collect();
    let curves: Vec<BSpline> =
        section_control_points.iter().map(|c| BSpline { degree: p, knots: knots[0].clone(), control_points: c.clone() }).collect();
    let sum: f64 = data.iter().zip(&u).map(|(d, &v)| (evaluate_chain(&curves, v, 0) - d).norm_squared()).sum();
    let error = (sum / data.len() as f64).sqrt();
    monitor.update(&Update {
        kind: UpdateKind::Evaluation,
        control_points: &section_control_points,
        knots: &knots,
        degree: p,
        error,
        evaluations,
    });
    Ok(RawResult {
        section_control_points,
        section_knots: None,
        error,
        rms_error: error,
        iterations: 1,
        evaluations,
        success: true,
        message: String::new(),
    })
}
