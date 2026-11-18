//! Variable projection formulation of constrained B-spline curve fitting.
//!
//! For data `D`, parameters `u` and control points `P`, the problem
//!
//! ```text
//! min ‖N(u)P - D‖² + (P - c)ᵀ(λS + reg·I)(P - c)   s.t.  C(θ)P = d
//! ```
//!
//! is linear in `P`. The ridge pulls towards the data centroid `c`, so a
//! fit does not change when the data move. Variable projection (Golub &
//! Pereyra 1973) eliminates `P` by solving the KKT system for every
//! `(u, θ)`, leaving a reduced problem in the outer variables whose residual
//! `[N P* - D ; R(P* - c)]` (`RᵀR = λS + reg·I`) equals the inner objective.
//!
//! The exact Jacobian with respect to `u` has a direct term (moving `u_j`
//! moves point `j` along the curve) and a coupling term through `∂P*/∂u_j`,
//! found by differentiating the KKT system; `u_j` only affects row `j` of
//! `N`, so all sensitivities come from one batched solve.

use std::cell::RefCell;

use nalgebra::{DMatrix, DVector};

use crate::basis::{bending_energy_matrix, matrix_root, num_control_points, validate_knots, SparseRows};
use crate::chain::{chain_data, chain_parameters, initial_chain_parameters, locate};
use crate::constraints::{ChainConstraints, Constraints, End, Junction};
use crate::kkt::KktSolver;
use crate::linalg::{block_diag, matrix_to_points, points_to_matrix};
use crate::{bail, centripetal_parameters, Result, Vec2};

/// Weight of the pull of each geometric junction's speed ratio towards its
/// natural value, relative to the smoothing weight: the weakest that keeps
/// `α` clear of its range limits.
pub const ALPHA_PULL: f64 = 3e-5;

/// Weight, relative to the data extent, of the rows that keep the curve's
/// speed between consecutive points from jumping between neighbouring pairs.
pub const SPEED_PULL: f64 = 1e-3;

/// Change of log speed from one pair of points to the next, beyond its
/// change in the centripetal parameters, that costs nothing.
pub const SPEED_SLACK: f64 = 1.0;

/// Everything solved at one `(u, θ)`.
struct Solution {
    u: Vec<f64>,
    theta: Vec<f64>,
    n: SparseRows,
    p: DMatrix<f64>,
    mu: DMatrix<f64>,
    kkt: KktSolver,
    constraints: Constraints,
}

/// Reduced residual and exact Jacobians as functions of the chain
/// parameters `u` and the junction shape parameters `θ`.
pub struct VarProProblem {
    pub sections: Vec<Vec<Vec2>>,
    pub data: Vec<Vec2>,
    pub knots: Vec<Vec<f64>>,
    pub degree: usize,
    pub constraints: ChainConstraints,
    /// Initial shape parameters.
    pub theta: Vec<f64>,
    data_matrix: DMatrix<f64>,
    offsets: Vec<usize>,
    smoothing: f64,
    regularization: f64,
    bending_energy: Option<DMatrix<f64>>,
    penalty_root: DMatrix<f64>,
    center: Vec2,
    alpha_pull: f64,
    num_pull_rows: usize,
    speed_pull: f64,
    speed_pairs: Vec<usize>,
    log_chords: Vec<f64>,
    speed_reference: Vec<f64>,
    cache: RefCell<Option<Solution>>,
}

/// Rows of the residual: data, penalty, α pull, speed.
#[derive(Debug, Clone, Copy)]
struct Layout {
    data: usize,
    penalty: usize,
    pull: usize,
    speed: usize,
}

impl VarProProblem {
    /// A problem on one knot vector per section.
    pub fn new(
        sections: &[Vec<Vec2>],
        knots: Vec<Vec<f64>>,
        degree: usize,
        junctions: Vec<Junction>,
        ends: (End, End),
        smoothing: f64,
        regularization: f64,
    ) -> Result<Self> {
        let num_sections = sections.len();
        if knots.len() != num_sections {
            bail!("Need one knot vector per section.");
        }
        for k in &knots {
            validate_knots(k, degree)?;
        }
        let data = chain_data(sections);
        let mut offsets = vec![0];
        for k in &knots {
            offsets.push(offsets.last().unwrap() + num_control_points(k, degree));
        }
        let total = *offsets.last().unwrap();

        let mut penalty = DMatrix::identity(total, total) * regularization;
        let bending_energy =
            (smoothing > 0.0).then(|| block_diag(&knots.iter().map(|k| bending_energy_matrix(k, degree)).collect::<Vec<_>>()));
        if let Some(s) = &bending_energy {
            penalty += s * smoothing;
        }
        let penalty_root = matrix_root(&penalty);
        let center = data.iter().sum::<Vec2>() / data.len() as f64;

        let constraints = ChainConstraints::new(sections, &knots, degree, junctions, ends.0, ends.1)?;
        let theta = constraints.initial_params();

        // With smoothing a geometric junction can lower the bending energy by
        // changing α alone, running it to the end of its range; a weak pull of
        // ln(α/ratio) towards 0, scaled like the bending energy, keeps it inside.
        let spread = data.iter().map(|p| (p - center).norm_squared()).sum::<f64>() / data.len() as f64;
        let alpha_pull = if smoothing > 0.0 { (smoothing * ALPHA_PULL * spread).sqrt() } else { 0.0 };
        let num_pull_rows = if alpha_pull > 0.0 { constraints.num_alpha() } else { 0 };

        // The data residual only sees the curve at the points, so a converged fit
        // can open one parameter gap into a bulge between two neighbours. Rows on
        // the change of log speed (gap over chord) beyond its change in the
        // centripetal parameters and beyond SPEED_SLACK rule such bulges out.
        // Coincident points and pairs across a junction are left out.
        let chords: Vec<f64> = data.windows(2).map(|w| (w[1] - w[0]).norm()).collect();
        let largest = chords.iter().copied().fold(0.0, f64::max);
        let moving: Vec<bool> = chords.iter().map(|&c| c > 1e-12 * largest.max(1e-300)).collect();
        let local: Vec<Vec<f64>> = sections.iter().map(|s| centripetal_parameters(s, 0.5)).collect();
        let reference = chain_parameters(&local, sections);
        let gap_section: Vec<f64> = reference.windows(2).map(|w| (0.5 * (w[0] + w[1])).floor().min(num_sections as f64 - 1.0)).collect();
        let speed_pairs: Vec<usize> =
            (0..chords.len().saturating_sub(1)).filter(|&i| moving[i] && moving[i + 1] && gap_section[i] == gap_section[i + 1]).collect();
        let log_chords = chords.iter().zip(&moving).map(|(c, &m)| if m { c.ln() } else { 0.0 }).collect();
        let speed_pull = SPEED_PULL * spread.sqrt();

        let mut problem = VarProProblem {
            sections: sections.to_vec(),
            data_matrix: points_to_matrix(&data),
            data,
            knots,
            degree,
            constraints,
            theta,
            offsets,
            smoothing,
            regularization,
            bending_energy,
            penalty_root,
            center,
            alpha_pull,
            num_pull_rows,
            speed_pull,
            speed_pairs,
            log_chords,
            speed_reference: vec![],
            cache: RefCell::new(None),
        };
        problem.speed_reference = problem.speed_change(&reference).0;
        Ok(problem)
    }

    pub fn num_sections(&self) -> usize {
        self.sections.len()
    }

    pub fn num_points(&self) -> usize {
        self.data.len()
    }

    pub fn num_params(&self) -> usize {
        self.constraints.num_params()
    }

    pub fn total_controls(&self) -> usize {
        *self.offsets.last().unwrap()
    }

    fn num_speed_rows(&self) -> usize {
        if self.speed_pull > 0.0 {
            self.speed_pairs.len()
        } else {
            0
        }
    }

    fn layout(&self) -> Layout {
        Layout {
            data: 2 * self.num_points(),
            penalty: 2 * self.penalty_root.nrows(),
            pull: self.num_pull_rows,
            speed: self.num_speed_rows(),
        }
    }

    /// Length of the residual vector.
    pub fn num_residuals(&self) -> usize {
        let l = self.layout();
        l.data + l.penalty + l.pull + l.speed
    }

    /// Initial chain parameters: centripetal per section.
    pub fn initial_parameters(&self) -> Vec<f64> {
        initial_chain_parameters(&self.sections)
    }

    /// Per-section slices of stacked control points.
    pub fn split_control_points(&self, p: &DMatrix<f64>) -> Vec<Vec<Vec2>> {
        self.offsets.windows(2).map(|w| matrix_to_points(&p.rows(w[0], w[1] - w[0]).into_owned())).collect()
    }

    /// The `(points, controls)` basis (or derivative) matrix of the chain.
    fn block_matrix(&self, u: &[f64], order: usize) -> SparseRows {
        let s = self.num_sections();
        SparseRows::new(u, &self.knots, &self.offsets, self.degree, order, |v| locate(v, s))
    }

    /// Solves the KKT system at `(u, θ)` unless that is the cached point.
    fn solve(&self, u: &[f64], theta: &[f64]) -> std::cell::Ref<'_, Solution> {
        let fresh = self.cache.borrow().as_ref().is_some_and(|s| s.u == u && s.theta == theta);
        if !fresh {
            let n = self.block_matrix(u, 0);
            let mut a = n.gram();
            if let Some(s) = &self.bending_energy {
                a += s * self.smoothing;
            }
            let constraints = self.constraints.build(theta);
            let kkt = KktSolver::new(&a, &constraints.c, self.regularization);
            // Solved for the offset from the centroid, so a singular system's
            // minimum-norm solution does not depend on where the data lie.
            let total = self.total_controls();
            let center = DMatrix::from_fn(total, 2, |_, k| self.center[k]);
            let centered = DMatrix::from_fn(self.num_points(), 2, |i, k| self.data[i][k] - self.center[k]);
            let rhs = n.tr_mul(&centered);
            let d = &constraints.d - &constraints.c * &center;
            let (offset, mu) = kkt.solve_full(&rhs, Some(&d));
            let p = offset + center;
            *self.cache.borrow_mut() = Some(Solution { u: u.to_vec(), theta: theta.to_vec(), n, p, mu, kkt, constraints });
        }
        std::cell::Ref::map(self.cache.borrow(), |s| s.as_ref().unwrap())
    }

    /// Optimal stacked control points `P*(u, θ)`, shape `(T, 2)`.
    pub fn control_points(&self, u: &[f64], theta: &[f64]) -> DMatrix<f64> {
        self.solve(u, theta).p.clone()
    }

    /// Change of log speed per speed pair, and the gaps.
    fn speed_change(&self, u: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let gaps: Vec<f64> = u.windows(2).map(|w| (w[1] - w[0]).max(1e-300)).collect();
        let log_speed: Vec<f64> = gaps.iter().zip(&self.log_chords).map(|(g, c)| g.ln() - c).collect();
        let change = self.speed_pairs.iter().map(|&i| log_speed[i + 1] - log_speed[i]).collect();
        (change, gaps)
    }

    /// Residual: data rows `N P - D` and penalty rows `R(P - c)` (point-major),
    /// then the `α` pull rows and the speed rows. Also returns the stacked
    /// control points and the RMS of the data residual.
    pub fn residual(&self, u: &[f64], theta: &[f64]) -> (DVector<f64>, DMatrix<f64>, f64) {
        let solution = self.solve(u, theta);
        let layout = self.layout();
        let mut r = DVector::zeros(self.num_residuals());
        let fitted = solution.n.mul(&solution.p);
        let mut sum = 0.0;
        for i in 0..self.num_points() {
            for k in 0..2 {
                let value = fitted[(i, k)] - self.data_matrix[(i, k)];
                r[2 * i + k] = value;
                sum += value * value;
            }
        }
        let center = DMatrix::from_fn(self.total_controls(), 2, |_, k| self.center[k]);
        let penalty = &self.penalty_root * (&solution.p - center);
        for i in 0..penalty.nrows() {
            for k in 0..2 {
                r[layout.data + 2 * i + k] = penalty[(i, k)];
            }
        }
        let mut row = layout.data + layout.penalty;
        if layout.pull > 0 {
            for (_, value, _) in self.constraints.alpha_deviations(theta) {
                r[row] = self.alpha_pull * value;
                row += 1;
            }
        }
        if layout.speed > 0 {
            let (change, _) = self.speed_change(u);
            for (c, reference) in change.iter().zip(&self.speed_reference) {
                let excess = c - reference;
                if excess.abs() > SPEED_SLACK {
                    r[row] = self.speed_pull * (excess - excess.signum() * SPEED_SLACK);
                }
                row += 1;
            }
        }
        let rms = (sum / self.num_points() as f64).sqrt();
        (r, solution.p.clone(), rms)
    }

    /// Data and penalty rows of `∂r/∂x` for control-point sensitivities
    /// `dP[k]` (one `(T, m)` matrix per coordinate); the other rows are zero.
    fn stack(&self, n: &SparseRows, dp: [DMatrix<f64>; 2]) -> DMatrix<f64> {
        let layout = self.layout();
        let columns = dp[0].ncols();
        let mut j = DMatrix::zeros(self.num_residuals(), columns);
        for (k, dpk) in dp.iter().enumerate() {
            let penalty = &self.penalty_root * dpk;
            for col in 0..columns {
                let source = dpk.column(col);
                let source = source.as_slice();
                let mut target = j.column_mut(col);
                for i in 0..self.num_points() {
                    let (start, values) = n.row(i);
                    target[2 * i + k] = values.iter().zip(&source[start..start + values.len()]).map(|(a, b)| a * b).sum();
                }
                for i in 0..penalty.nrows() {
                    target[layout.data + 2 * i + k] = penalty[(i, col)];
                }
            }
        }
        j
    }

    /// Exact `∂r/∂u`, shape `(residuals, N)`.
    ///
    /// A point exactly on a junction (an integer `u`) lies on the section
    /// that starts there, so its column is the derivative from the right.
    pub fn jacobian(&self, u: &[f64], theta: &[f64]) -> DMatrix<f64> {
        let solution = self.solve(u, theta);
        let dn = self.block_matrix(u, 1);
        let n = &solution.n;
        let tangent = dn.mul(&solution.p);
        let misfit = &self.data_matrix - n.mul(&solution.p);
        // dP_jk = H (dN_jᵀ M_jk - N_jᵀ Tg_jk), H the control block of K⁻¹.
        // The right-hand side is nonzero only on point j's knot span, so the
        // data and penalty rows of column j combine a few columns of N H and
        // R H instead of solving and multiplying out a dense column.
        let total = self.total_controls();
        let h = solution.kkt.solve(&DMatrix::identity(total, total), None);
        let nh = n.mul(&h);
        let rh = &self.penalty_root * &h;
        let layout = self.layout();
        let rows = self.num_residuals();
        let mut jac = DMatrix::zeros(rows, self.num_points());
        let (points, penalties) = (nh.nrows(), rh.nrows());
        let (nh, rh) = (nh.as_slice(), rh.as_slice());
        let mut weights = Vec::new();
        for (j, column) in jac.as_mut_slice().chunks_exact_mut(rows).enumerate() {
            let (d_start, d_values) = dn.row(j);
            let (n_start, n_values) = n.row(j);
            let first = d_start.min(n_start);
            let end = (d_start + d_values.len()).max(n_start + n_values.len());
            let (data, rest) = column.split_at_mut(layout.data);
            let penalty = &mut rest[..layout.penalty];
            for k in 0..2 {
                weights.clear();
                weights.resize(end - first, 0.0);
                for (c, v) in d_values.iter().enumerate() {
                    weights[d_start - first + c] += v * misfit[(j, k)];
                }
                for (c, v) in n_values.iter().enumerate() {
                    weights[n_start - first + c] -= v * tangent[(j, k)];
                }
                for (c, &w) in weights.iter().enumerate() {
                    let control = first + c;
                    let source = &nh[control * points..(control + 1) * points];
                    for (target, value) in data.iter_mut().skip(k).step_by(2).zip(source) {
                        *target += w * value;
                    }
                    let source = &rh[control * penalties..(control + 1) * penalties];
                    for (target, value) in penalty.iter_mut().skip(k).step_by(2).zip(source) {
                        *target += w * value;
                    }
                }
                data[2 * j + k] += tangent[(j, k)];
            }
        }
        if layout.speed > 0 {
            let first = layout.data + layout.penalty + layout.pull;
            let (change, gaps) = self.speed_change(u);
            for (row, (&i, (c, reference))) in self.speed_pairs.iter().zip(change.iter().zip(&self.speed_reference)).enumerate() {
                if (c - reference).abs() <= SPEED_SLACK {
                    continue;
                }
                let (a, b) = (self.speed_pull / gaps[i], self.speed_pull / gaps[i + 1]);
                jac[(first + row, i)] += a;
                jac[(first + row, i + 1)] -= a + b;
                jac[(first + row, i + 2)] += b;
            }
        }
        jac
    }

    /// Exact `∂r/∂θ`, shape `(residuals, num_params)`.
    pub fn parameter_jacobian(&self, u: &[f64], theta: &[f64]) -> DMatrix<f64> {
        let m = self.num_params();
        if m == 0 {
            return DMatrix::zeros(self.num_residuals(), 0);
        }
        let solution = self.solve(u, theta);
        let total = self.total_controls();
        let rows = solution.constraints.c.nrows();
        let dp = [0, 1].map(|k| {
            let mut rhs = DMatrix::zeros(total, m);
            let mut d_rhs = DMatrix::zeros(rows, m);
            for (i, dc) in solution.constraints.dc.iter().enumerate() {
                rhs.set_column(i, &(-(dc.transpose() * solution.mu.column(k))));
                d_rhs.set_column(i, &(-(dc * solution.p.column(k))));
            }
            solution.kkt.solve(&rhs, Some(&d_rhs))
        });
        let mut jac = self.stack(&solution.n, dp);
        let layout = self.layout();
        if layout.pull > 0 {
            let first = layout.data + layout.penalty;
            for (row, (slot, _, derivative)) in self.constraints.alpha_deviations(theta).into_iter().enumerate() {
                jac[(first + row, slot)] = self.alpha_pull * derivative;
            }
        }
        jac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::airfoil::Naca4;
    use crate::open_uniform_knots;

    fn sections() -> Vec<Vec<Vec2>> {
        let points = Naca4::new("2412", true).unwrap().points(21);
        vec![points[..21].to_vec(), points[20..].to_vec()]
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let sections = sections();
        let knots = vec![open_uniform_knots(5, 9).unwrap(); 2];
        let junction = Junction::new(3, true, true).unwrap();
        let problem = VarProProblem::new(&sections, knots, 5, vec![junction], (End::default(), End::default()), 1e-3, 1e-12).unwrap();
        let mut u = problem.initial_parameters();
        // move off the centripetal start so the speed rows are active somewhere
        u[5] += 0.02;
        let theta = vec![0.2, 0.1, -0.05];
        let (r0, _, _) = problem.residual(&u, &theta);
        let ju = problem.jacobian(&u, &theta);
        for j in [3, 5, 20, 30] {
            let mut v = u.clone();
            let h = 1e-7;
            v[j] += h;
            let fd = (problem.residual(&v, &theta).0 - &r0) / h;
            let err = (fd - ju.column(j)).abs().max();
            assert!(err < 1e-5 * (1.0 + ju.column(j).abs().max()), "u column {j}: {err}");
        }
        let jt = problem.parameter_jacobian(&u, &theta);
        for j in 0..3 {
            let mut t = theta.clone();
            let h = 1e-7;
            t[j] += h;
            let fd = (problem.residual(&u, &t).0 - &r0) / h;
            let err = (fd - jt.column(j)).abs().max();
            assert!(err < 1e-5 * (1.0 + jt.column(j).abs().max()), "theta column {j}: {err}");
        }
    }
}
