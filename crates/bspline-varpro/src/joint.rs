//! Joint optimization of data parameters and control points.
//!
//! Unlike variable projection, the control points stay unknowns. The
//! constraints `C P = d` (with junction shape parameters frozen) are
//! eliminated by a null-space parameterization `P = P0 + Z z`, and the chain
//! parameters are free per point. Each data residual depends on one
//! parameter only, so the normal equations have a diagonal parameter block
//! that a Schur complement eliminates: a step costs `O(n k²)` for `n` points
//! and `k` free control coordinates, which handles dense data.

use nalgebra::{DMatrix, DVector};

use crate::basis::{bending_energy_matrix, matrix_root, num_control_points, SparseRows};
use crate::chain::{chain_data, chain_deviations, locate};
use crate::fitting::{ChainOptions, FitResult};
use crate::linalg::{block_diag, lstsq, matrix_to_points, null_space, points_to_matrix};
use crate::{bail, Observer, Progress, Result, Vec2};

/// Residual and Jacobian blocks of the joint `(u, z)` problem.
pub struct JointProblem {
    data: Vec<Vec2>,
    knots: Vec<Vec<f64>>,
    degree: usize,
    offsets: Vec<usize>,
    z_basis: DMatrix<f64>,
    p0: DMatrix<f64>,
    rz: DMatrix<f64>,
    rp0: DMatrix<f64>,
    /// Residual evaluations so far.
    pub evaluations: usize,
}

struct Evaluation {
    r: DVector<f64>,
    /// `dN(u) P`, one row per point.
    tangent: DMatrix<f64>,
    /// `∂r/∂z`.
    jz: DMatrix<f64>,
}

impl JointProblem {
    pub fn new(
        sections: &[Vec<Vec2>],
        knots: Vec<Vec<f64>>,
        degree: usize,
        c: &DMatrix<f64>,
        d: &DMatrix<f64>,
        smoothing: f64,
        regularization: f64,
    ) -> Self {
        let data = chain_data(sections);
        let mut offsets = vec![0];
        for k in &knots {
            offsets.push(offsets.last().unwrap() + num_control_points(k, degree));
        }
        let total = *offsets.last().unwrap();
        let (z_basis, p0) =
            if c.nrows() > 0 { (null_space(c, 1e-10).0, lstsq(c, d)) } else { (DMatrix::identity(total, total), DMatrix::zeros(total, 2)) };
        let mut penalty = DMatrix::identity(total, total) * regularization;
        if smoothing > 0.0 {
            penalty += block_diag(&knots.iter().map(|k| bending_energy_matrix(k, degree)).collect::<Vec<_>>()) * smoothing;
        }
        let root = matrix_root(&penalty);
        let center = data.iter().sum::<Vec2>() / data.len() as f64;
        let centered = DMatrix::from_fn(total, 2, |i, k| p0[(i, k)] - center[k]);
        JointProblem { rz: &root * &z_basis, rp0: &root * centered, data, knots, degree, offsets, z_basis, p0, evaluations: 0 }
    }

    fn num_z(&self) -> usize {
        self.z_basis.ncols()
    }

    /// Stacked control points of flat coordinates `z` (`num_z × 2`, row-major).
    pub fn control_points(&self, z: &DVector<f64>) -> DMatrix<f64> {
        &self.p0 + &self.z_basis * self.z_matrix(z)
    }

    fn z_matrix(&self, z: &DVector<f64>) -> DMatrix<f64> {
        DMatrix::from_fn(self.num_z(), 2, |i, k| z[2 * i + k])
    }

    /// Flat null-space coordinates of control points that meet the constraints.
    pub fn z_from(&self, p: &DMatrix<f64>) -> DVector<f64> {
        let z = self.z_basis.transpose() * (p - &self.p0);
        DVector::from_fn(z.len(), |i, _| z[(i / 2, i % 2)])
    }

    fn basis(&self, u: &[f64], order: usize) -> SparseRows {
        let s = self.knots.len();
        SparseRows::new(u, &self.knots, &self.offsets, self.degree, order, |v| locate(v, s))
    }

    fn residual(&mut self, u: &[f64], z: &DVector<f64>) -> DVector<f64> {
        self.evaluate(u, z, false).r
    }

    fn evaluate(&mut self, u: &[f64], z: &DVector<f64>, with_jacobian: bool) -> Evaluation {
        self.evaluations += 1;
        let p = self.control_points(z);
        let n = self.basis(u, 0);
        let fitted = n.mul(&p);
        let penalty = &self.rp0 + &self.rz * self.z_matrix(z);
        let rows = 2 * (self.data.len() + penalty.nrows());
        let mut r = DVector::zeros(rows);
        for i in 0..self.data.len() {
            for k in 0..2 {
                r[2 * i + k] = fitted[(i, k)] - self.data[i][k];
            }
        }
        let first = 2 * self.data.len();
        for i in 0..penalty.nrows() {
            for k in 0..2 {
                r[first + 2 * i + k] = penalty[(i, k)];
            }
        }
        if !with_jacobian {
            return Evaluation { r, tangent: DMatrix::zeros(0, 0), jz: DMatrix::zeros(0, 0) };
        }
        let tangent = self.basis(u, 1).mul(&p);
        // kron(M, I₂): row 2i + k, column 2j + k holds M[i, j].
        let nz = n.mul(&self.z_basis);
        let mut jz = DMatrix::zeros(rows, 2 * self.num_z());
        for (offset, m) in [(0, &nz), (self.data.len(), &self.rz)] {
            for i in 0..m.nrows() {
                for j in 0..m.ncols() {
                    for k in 0..2 {
                        jz[(2 * (offset + i) + k, 2 * j + k)] = m[(i, j)];
                    }
                }
            }
        }
        Evaluation { r, tangent, jz }
    }
}

/// Least-squares nondecreasing projection (pool adjacent violators).
fn nondecreasing(values: &[f64]) -> Vec<f64> {
    let mut blocks: Vec<(f64, usize)> = Vec::new();
    for &v in values {
        blocks.push((v, 1));
        while blocks.len() > 1 && blocks[blocks.len() - 2].0 > blocks[blocks.len() - 1].0 {
            let (m2, c2) = blocks.pop().unwrap();
            let (m1, c1) = blocks.pop().unwrap();
            blocks.push(((m1 * c1 as f64 + m2 * c2 as f64) / (c1 + c2) as f64, c1 + c2));
        }
    }
    blocks.into_iter().flat_map(|(m, c)| std::iter::repeat_n(m, c)).collect()
}

/// Largest cosine between the residual and a Jacobian column.
fn gradient_cosine(g_t: &DVector<f64>, u_diag: &DVector<f64>, g_z: &DVector<f64>, v: &DMatrix<f64>, cost: f64) -> f64 {
    let r_norm = (2.0 * cost).sqrt();
    if r_norm == 0.0 {
        return 0.0;
    }
    let gradient = g_t.iter().chain(g_z.iter());
    let diagonal = v.diagonal();
    let norms = u_diag.iter().chain(diagonal.iter());
    gradient
        .zip(norms)
        .map(|(g, n2)| {
            let norm = n2.max(0.0).sqrt();
            if norm > 0.0 {
                g.abs() / (norm * r_norm)
            } else {
                0.0
            }
        })
        .fold(0.0, f64::max)
}

/// Called after every accepted step of [`solve_joint`]; true stops it.
pub type StepCallback<'a> = dyn FnMut(&JointProblem, &[f64], &DVector<f64>) -> bool + 'a;

/// End state of [`solve_joint`].
#[derive(Debug, Clone)]
pub struct JointSolution {
    pub u: Vec<f64>,
    pub z: DVector<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub message: &'static str,
}

/// Levenberg-Marquardt with Schur elimination of the per-point parameters.
///
/// Every step is projected onto nondecreasing parameters, so points never
/// pass each other. Convergence is tested on the cosine between the residual
/// and the Jacobian columns (MINPACK's gtol). `on_step` runs after every
/// accepted step and stops the solve by returning true.
pub fn solve_joint(
    problem: &mut JointProblem,
    u: &[f64],
    z: DVector<f64>,
    max_iterations: Option<usize>,
    tolerance: f64,
    on_step: &mut StepCallback,
) -> Result<JointSolution> {
    if max_iterations == Some(0) {
        bail!("max_iterations must be at least 1 or None.");
    }
    if tolerance.is_nan() || tolerance <= 0.0 {
        bail!("tolerance must be positive.");
    }
    let s = problem.knots.len() as f64;
    let n_points = problem.data.len();
    let free: Vec<usize> = (1..n_points.saturating_sub(1)).collect();
    let mut running = f64::NEG_INFINITY;
    let mut u: Vec<f64> = u
        .iter()
        .map(|&v| {
            running = running.max(v);
            running.clamp(0.0, s)
        })
        .collect();
    let mut z = z;
    let mut eval = problem.evaluate(&u, &z, true);
    let mut cost = 0.5 * eval.r.norm_squared();
    let mut mu: Option<f64> = None;
    let mut nu = 2.0;
    let solution =
        |u: Vec<f64>, z: DVector<f64>, iterations, converged, message| Ok(JointSolution { u, z, iterations, converged, message });
    let mut iteration = 0;
    let (mut converged, mut message) = (false, "Maximum iterations reached.");
    while max_iterations.is_none_or(|m| iteration < m) {
        iteration += 1;
        let nzc = eval.jz.ncols();
        let mut g_t = DVector::zeros(free.len());
        let mut u_diag = DVector::zeros(free.len());
        let mut w = DMatrix::zeros(free.len(), nzc);
        for (f, &i) in free.iter().enumerate() {
            for k in 0..2 {
                let t = eval.tangent[(i, k)];
                g_t[f] += t * eval.r[2 * i + k];
                u_diag[f] += t * t;
                for c in 0..nzc {
                    w[(f, c)] += t * eval.jz[(2 * i + k, c)];
                }
            }
        }
        let v = eval.jz.transpose() * &eval.jz;
        let g_z = eval.jz.transpose() * &eval.r;
        let cosine = gradient_cosine(&g_t, &u_diag, &g_z, &v, cost);
        if cosine <= tolerance {
            converged = true;
            message = "Gradient below tolerance.";
            break;
        }
        let v_max = v.diagonal().iter().copied().fold(0.0, f64::max);
        let d_v = v.diagonal().map(|x| x.max(1e-12 * v_max.max(1e-300)));
        let u_max = u_diag.iter().copied().fold(1.0, f64::max);
        let u_floor = u_diag.map(|x| x.max(1e-12 * u_max.max(1e-300)));
        let mut damping = mu.unwrap_or(1e-3);
        let (u_new, z_new, cost_new) = loop {
            let u_d = &u_diag + &u_floor * damping;
            let mut v_d = v.clone();
            for j in 0..nzc {
                v_d[(j, j)] += damping * d_v[j];
            }
            let mut w_scaled = w.clone();
            for (f, mut row) in w_scaled.row_iter_mut().enumerate() {
                row /= u_d[f];
            }
            let schur = v_d - w.transpose() * &w_scaled;
            let rhs = -&g_z + w.transpose() * g_t.component_div(&u_d);
            let Some(dz) = schur.lu().solve(&rhs) else {
                damping *= nu;
                nu *= 2.0;
                if damping > 1e16 {
                    return solution(u, z, iteration, false, "Singular normal equations.");
                }
                continue;
            };
            let dt = -(&g_t + &w * &dz).component_div(&u_d);
            let moved: Vec<f64> = free.iter().enumerate().map(|(f, &i)| u[i] + dt[f]).collect();
            let mut u_new = u.clone();
            for (&i, value) in free.iter().zip(nondecreasing(&moved)) {
                u_new[i] = value.clamp(0.0, s);
            }
            let dt = DVector::from_iterator(free.len(), free.iter().map(|&i| u_new[i] - u[i]));
            let z_new = &z + &dz;
            let cost_new = 0.5 * problem.residual(&u_new, &z_new).norm_squared();
            let wdz = &w * &dz;
            let predicted =
                -(g_t.dot(&dt) + g_z.dot(&dz)) - 0.5 * (dt.dot(&u_diag.component_mul(&dt)) + 2.0 * dt.dot(&wdz) + dz.dot(&(&v * &dz)));
            let rho = if predicted > 0.0 { (cost - cost_new) / predicted } else { -1.0 };
            if rho > 0.0 {
                damping *= (1.0 / 3.0f64).max(1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
                break (u_new, z_new, cost_new);
            }
            damping *= nu;
            nu *= 2.0;
            if damping > 1e16 {
                let stalled = cosine <= tolerance.sqrt();
                return solution(u, z, iteration, stalled, "No further decrease possible.");
            }
        };
        mu = Some(damping);
        let decrease = cost - cost_new;
        u = u_new;
        z = z_new;
        eval = problem.evaluate(&u, &z, true);
        cost = cost_new;
        if on_step(problem, &u, &z) {
            return solution(u, z, iteration, false, crate::STOPPED);
        }
        if decrease <= tolerance * cost {
            converged = true;
            message = "Relative decrease below tolerance.";
            break;
        }
    }
    solution(u, z, iteration, converged, message)
}

/// Fit state of a joint solution, with the knots and θ of `template`.
fn to_result(problem: &JointProblem, u: &[f64], z: &DVector<f64>, template: &FitResult) -> FitResult {
    let p = problem.control_points(z);
    let fitted = problem.basis(u, 0).mul(&p);
    let misfit: f64 = (0..problem.data.len()).map(|i| (fitted.row(i).transpose() - problem.data[i]).norm_squared()).sum();
    FitResult {
        control_points: problem.offsets.windows(2).map(|w| matrix_to_points(&p.rows(w[0], w[1] - w[0]).into_owned())).collect(),
        parameters: u.to_vec(),
        knots: template.knots.clone(),
        degree: template.degree,
        rms_error: (misfit / problem.data.len() as f64).sqrt(),
        nfev: problem.evaluations,
        success: false,
        message: String::new(),
        stopped: false,
        theta: template.theta.clone(),
        max_deviation: None,
    }
}

/// Joint Levenberg-Marquardt refinement of a fit on all points: the second
/// stage for dense data after variable projection on a subsample. Knots and
/// junction shape parameters stay fixed; the result has `max_deviation`.
pub fn refine(sections: &[Vec<Vec2>], result: &FitResult, options: &ChainOptions, observer: &mut dyn Observer) -> Result<FitResult> {
    let constraints = options.problem(sections, result.knots.clone(), result.degree)?.constraints;
    let theta = if result.theta.len() == constraints.num_params() { result.theta.clone() } else { constraints.initial_params() };
    let built = constraints.build(&theta);
    let mut problem =
        JointProblem::new(sections, result.knots.clone(), result.degree, &built.c, &built.d, options.smoothing, options.regularization);
    let z0 = problem.z_from(&points_to_matrix(&result.control_points.concat()));
    let mut on_step = |problem: &JointProblem, u: &[f64], z: &DVector<f64>| {
        let step = to_result(problem, u, z, result);
        observer.step(&Progress { control_points: &step.control_points, rms: step.rms_error, knots: &step.knots })
    };
    let solved = solve_joint(&mut problem, &result.parameters, z0, None, 1e-12, &mut on_step)?;
    let mut refined = to_result(&problem, &solved.u, &solved.z, result);
    refined.success = solved.converged;
    refined.stopped = solved.message == crate::STOPPED;
    refined.message = solved.message.to_string();
    refined.max_deviation = Some(chain_deviations(&problem.data, &refined.curves()).into_iter().fold(f64::NEG_INFINITY, f64::max));
    Ok(refined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_adjacent_violators() {
        assert_eq!(nondecreasing(&[1.0, 3.0, 2.0, 4.0]), vec![1.0, 2.5, 2.5, 4.0]);
        assert_eq!(nondecreasing(&[3.0, 2.0, 1.0]), vec![2.0, 2.0, 2.0]);
    }
}
