//! Direct LSQ: one constrained linear least-squares solve on uniform knots
//! with the data parameters taken from the point spacing.
//!
//! The chain is fitted as a whole with parametric continuity (position
//! alone at a corner of the data): the ends are fixed, the sections joined
//! and, unless the junctions are free, pinned to their split points. One
//! solve, one evaluation, no iteration.

use bspline_varpro::basis::SparseRows;
use bspline_varpro::chain::{chain_data, chain_parameters, evaluate_chain, locate};
use bspline_varpro::{centripetal_parameters, open_uniform_knots, BSpline, ChainConstraints, End, Junction, KktSolver};
use nalgebra::DMatrix;

use super::{Chain, Monitor, RawResult, Update, UpdateKind};
use crate::models::{OptimizationConfig, Parameterization, SplineConfig};
use crate::{Result, Vec2};

/// Ridge of the linear solve.
const REGULARIZATION: f64 = 1e-10;

pub fn fit(config: &SplineConfig, options: &OptimizationConfig, chain: &Chain, monitor: &mut dyn Monitor) -> Result<RawResult> {
    let p = config.degree;
    let sections = &chain.sections;
    let counts = chain.counts(config.num_control_points);
    let s = sections.len();
    let knots = counts.iter().map(|&n| open_uniform_knots(p, n)).collect::<bspline_varpro::Result<Vec<_>>>()?;
    let data = chain_data(sections);
    let alpha = match options.parameterization {
        Parameterization::Centripetal => 0.5,
        Parameterization::ArcLength => 1.0,
    };
    let local: Vec<Vec<f64>> = sections.iter().map(|sec| centripetal_parameters(sec, alpha)).collect();
    let u = chain_parameters(&local, sections);
    let fixed = !(options.optimize_le_position && s > 1);
    let junctions: Vec<Junction> = chain.junctions.iter().map(|j| Junction { geometric: false, fixed, ..*j }).collect();
    let constraints = ChainConstraints::new(sections, &knots, p, junctions, End::default(), End::default())?.build(&[]);

    let mut offsets = vec![0];
    for n in &counts {
        offsets.push(offsets[offsets.len() - 1] + n);
    }
    let basis = SparseRows::new(&u, &knots, &offsets, p, 0, |v| locate(v, s));
    let rhs = basis.tr_mul(&DMatrix::from_fn(data.len(), 2, |i, k| data[i][k]));
    let control = KktSolver::new(&basis.gram(), &constraints.c, REGULARIZATION).solve(&rhs, Some(&constraints.d));

    let section_control_points: Vec<Vec<Vec2>> =
        offsets.windows(2).map(|w| (w[0]..w[1]).map(|i| Vec2::new(control[(i, 0)], control[(i, 1)])).collect()).collect();
    let curves: Vec<BSpline> = section_control_points
        .iter()
        .zip(&knots)
        .map(|(c, k)| BSpline { degree: p, knots: k.clone(), control_points: c.clone() })
        .collect();
    // Of the airfoil's own points.
    let squared: Vec<f64> = data.iter().zip(&u).map(|(d, &v)| (evaluate_chain(&curves, v, 0) - d).norm_squared()).collect();
    let error = chain.rms((squared.iter().sum::<f64>() / squared.len() as f64).sqrt(), || squared.clone());
    monitor.update(&Update {
        kind: UpdateKind::Evaluation,
        control_points: &section_control_points,
        knots: &knots,
        degree: p,
        error,
        evaluations: 1,
    });
    Ok(RawResult {
        section_control_points,
        section_knots: None,
        error,
        rms_error: error,
        iterations: 1,
        evaluations: 1,
        success: true,
        message: String::new(),
        seconds: None,
        earlier: vec![],
        search: None,
    })
}
