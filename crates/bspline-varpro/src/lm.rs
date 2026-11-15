//! Levenberg-Marquardt on the normal equations.
//!
//! Marquardt scaling with the running maximum of `diag(JᵀJ)`, Nielsen damping
//! updates and MINPACK's ftol / xtol / gtol tests.

use nalgebra::{DMatrix, DVector};

/// A progress observer asked the solve to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stopped;

/// A least-squares model: residual and Jacobian at `x`.
pub trait Model {
    fn residual(&mut self, x: &DVector<f64>) -> Result<DVector<f64>, Stopped>;
    fn jacobian(&mut self, x: &DVector<f64>) -> Result<DMatrix<f64>, Stopped>;
}

/// Why a solve ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    NotFinite,
    Budget,
    Gradient,
    Cost,
    Step,
}

impl Status {
    pub fn success(self) -> bool {
        matches!(self, Status::Gradient | Status::Cost | Status::Step)
    }

    pub fn message(self) -> &'static str {
        match self {
            Status::NotFinite => "The residual or Jacobian is not finite.",
            Status::Budget => "The maximum number of function evaluations is exceeded.",
            Status::Gradient => "`gtol` termination condition is satisfied.",
            Status::Cost => "`ftol` termination condition is satisfied.",
            Status::Step => "`xtol` termination condition is satisfied.",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub x: DVector<f64>,
    pub cost: f64,
    pub nfev: usize,
    pub status: Status,
}

const MAX_DAMPING: f64 = 1e20;
const MAX_RESTARTS: usize = 8;

/// One run until a convergence test, the damping limit or the budget.
fn run(model: &mut dyn Model, x0: DVector<f64>, max_nfev: Option<usize>, tolerance: f64) -> Result<Solution, Stopped> {
    let mut x = x0;
    let mut r = model.residual(&x)?;
    let mut nfev = 1;
    let mut cost = 0.5 * r.norm_squared();
    let affordable = |nfev: usize| max_nfev.is_none_or(|m| nfev < m);

    let mut jac = model.jacobian(&x)?;
    let mut a = jac.transpose() * &jac;
    let mut g = jac.transpose() * &r;
    let mut scale: DVector<f64> = a.diagonal().map(|v| v.max(1e-300));
    let (mut mu, mut nu) = (1e-3, 2.0);
    let mut status = Status::Budget;
    while affordable(nfev) {
        if !(cost.is_finite() && a.iter().all(|v| v.is_finite()) && g.iter().all(|v| v.is_finite())) {
            status = Status::NotFinite;
            break;
        }
        let r_norm = (2.0 * cost).sqrt();
        let max_cosine = (0..g.len())
            .map(|j| {
                let column = a[(j, j)].max(0.0).sqrt();
                if column > 0.0 {
                    g[j].abs() / (column * r_norm)
                } else {
                    0.0
                }
            })
            .fold(0.0, f64::max);
        if r_norm == 0.0 || max_cosine <= tolerance {
            status = Status::Gradient;
            break;
        }
        let mut damped = a.clone();
        for j in 0..scale.len() {
            damped[(j, j)] += mu * scale[j];
        }
        let step = damped.cholesky().map(|c| -c.solve(&g)).filter(|s| s.iter().all(|v| v.is_finite()));
        let Some(step) = step else {
            mu *= nu;
            nu *= 2.0;
            if mu > MAX_DAMPING {
                status = Status::Step;
                break;
            }
            continue;
        };
        let root = scale.map(f64::sqrt);
        if root.component_mul(&step).norm() <= tolerance * (tolerance + root.component_mul(&x).norm()) {
            status = Status::Step;
            break;
        }
        let x_new = &x + &step;
        let r_new = model.residual(&x_new)?;
        nfev += 1;
        let cost_new = 0.5 * r_new.norm_squared();
        let predicted = -(g.dot(&step) + 0.5 * step.dot(&(&a * &step)));
        let rho = if predicted > 0.0 { (cost - cost_new) / predicted } else { -1.0 };
        if rho > 0.0 && cost_new.is_finite() {
            let actual = if cost > 0.0 { (cost - cost_new) / cost } else { 0.0 };
            x = x_new;
            r = r_new;
            cost = cost_new;
            if actual <= tolerance && predicted / cost.max(1e-300) <= tolerance {
                status = Status::Cost;
                break;
            }
            jac = model.jacobian(&x)?;
            a = jac.transpose() * &jac;
            g = jac.transpose() * &r;
            scale = scale.zip_map(&a.diagonal(), f64::max);
            mu *= (1.0 / 3.0f64).max(1.0 - (2.0 * rho - 1.0).powi(3));
            nu = 2.0;
        } else {
            mu *= nu;
            nu *= 2.0;
            if mu > MAX_DAMPING {
                status = Status::Step;
                break;
            }
        }
    }
    Ok(Solution { x, cost, nfev, status })
}

/// Levenberg-Marquardt, restarted while it stops on the step test but still
/// improves: after a series of rejected steps the damping grows until the
/// step length triggers xtol in curved valleys, and a restart resets it.
/// `max_nfev` caps the residual evaluations of all runs together.
pub fn levenberg_marquardt(model: &mut dyn Model, x0: DVector<f64>, max_nfev: Option<usize>, tolerance: f64) -> Result<Solution, Stopped> {
    let mut x = x0;
    let mut used = 0;
    let mut best = f64::INFINITY;
    let mut solution;
    let mut restarts = 0;
    loop {
        solution = run(model, x, max_nfev.map(|m| m - used), tolerance)?;
        used += solution.nfev;
        let improved = solution.cost < 0.99 * best;
        best = best.min(solution.cost);
        x = solution.x.clone();
        let budget_left = max_nfev.is_none_or(|m| used < m);
        if solution.status != Status::Step || !improved || !budget_left || restarts == MAX_RESTARTS {
            break;
        }
        restarts += 1;
    }
    solution.nfev = used;
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rosenbrock;

    impl Model for Rosenbrock {
        fn residual(&mut self, x: &DVector<f64>) -> Result<DVector<f64>, Stopped> {
            Ok(DVector::from_vec(vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]))
        }
        fn jacobian(&mut self, x: &DVector<f64>) -> Result<DMatrix<f64>, Stopped> {
            Ok(DMatrix::from_row_slice(2, 2, &[-20.0 * x[0], 10.0, -1.0, 0.0]))
        }
    }

    #[test]
    fn solves_rosenbrock() {
        let s = levenberg_marquardt(&mut Rosenbrock, DVector::from_vec(vec![-1.2, 1.0]), None, 1e-10).unwrap();
        assert!(s.status.success());
        assert!((s.x[0] - 1.0).abs() < 1e-8 && (s.x[1] - 1.0).abs() < 1e-8);
    }
}
