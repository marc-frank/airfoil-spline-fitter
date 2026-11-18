//! Fitting chains of B-splines by Levenberg-Marquardt variable projection.

use nalgebra::{DMatrix, DVector};

use crate::basis::open_uniform_knots;
use crate::chain::{initial_chain_parameters, validate_chain_parameters, Parameters};
use crate::constraints::{End, Junction};
use crate::lm::{levenberg_marquardt, Model, Stopped};
use crate::problem::VarProProblem;
use crate::reparam::{apply_eta_jacobian, apply_eta_jacobian_transpose, eta_congruence, eta_to_t, t_to_eta};
use crate::tolerance::{grow_to_counts, Setup};
use crate::{bail, BSpline, Observer, Progress, Result, Vec2, STOPPED};

/// Convergence tolerance (ftol, xtol and gtol) of every fit. At 1e-10 a fit
/// in a valley without a minimum (a junction's β growing without limit for
/// ever smaller gains) never ended.
pub const LM_TOLERANCE: f64 = 1e-8;

/// Default smallest spacing of free knots, as a fraction of a section.
pub const MIN_KNOT_GAP: f64 = 0.05;

/// Checks a minimum knot gap: between 0 and 0.5.
pub(crate) fn validate_min_knot_gap(gap: f64) -> Result<()> {
    if !(gap.is_finite() && (0.0..=0.5).contains(&gap)) {
        bail!("min_knot_gap must be between 0 and 0.5.");
    }
    Ok(())
}

/// Continuity at the junctions: one for all, or one per junction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Continuity {
    All(Junction),
    Each(Vec<Junction>),
}

/// Constraint and penalty settings shared by every fit of one chain.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainOptions {
    pub continuity: Continuity,
    /// Pin each junction to the last point of its section (else it is fitted).
    pub fix_junctions: bool,
    /// Interpolate the first and last point of the chain.
    pub fix_ends: bool,
    /// Weight of the bending-energy penalty (0 disables it).
    pub smoothing: f64,
    /// Ridge towards the data centroid that keeps the solve well posed.
    pub regularization: f64,
}

impl Default for ChainOptions {
    fn default() -> Self {
        ChainOptions {
            continuity: Continuity::All(Junction { order: 1, geometric: true, fixed: true }),
            fix_junctions: true,
            fix_ends: true,
            smoothing: 1e-3,
            regularization: 1e-12,
        }
    }
}

impl ChainOptions {
    /// One junction per junction of a chain of `num_sections` sections.
    pub fn junctions(&self, num_sections: usize) -> Result<Vec<Junction>> {
        let count = num_sections - 1;
        let junctions = match &self.continuity {
            Continuity::All(j) => vec![*j; count],
            Continuity::Each(list) => {
                if list.len() != count {
                    bail!("Expected {count} continuity entries, got {}.", list.len());
                }
                list.clone()
            }
        };
        Ok(junctions.into_iter().map(|j| Junction { fixed: self.fix_junctions, ..j }).collect())
    }

    /// Junction orders of the chain.
    pub fn orders(&self, num_sections: usize) -> Result<Vec<usize>> {
        Ok(self.junctions(num_sections)?.iter().map(|j| j.order).collect())
    }

    pub fn problem(&self, sections: &[Vec<Vec2>], knots: Vec<Vec<f64>>, degree: usize) -> Result<VarProProblem> {
        let end = End { fixed: self.fix_ends };
        VarProProblem::new(sections, knots, degree, self.junctions(sections.len())?, (end, end), self.smoothing, self.regularization)
    }
}

/// Where knots go when a fit grows or settles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KnotPlacement {
    /// Evenly spaced in the curve parameter.
    #[default]
    Uniform,
    /// Moved to where they reduce the error most.
    Free,
}

/// Fitted curve sections.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Control points of every section.
    pub control_points: Vec<Vec<Vec2>>,
    /// Chain parameter of every data point, in `[0, S]`.
    pub parameters: Vec<f64>,
    /// Knot vector of every section.
    pub knots: Vec<Vec<f64>>,
    pub degree: usize,
    /// RMS distance of the data to their curve points (no penalty).
    pub rms_error: f64,
    /// Residual evaluations.
    pub nfev: usize,
    /// Whether the solver met a convergence test (or the tolerance search its tolerance).
    pub success: bool,
    pub message: String,
    /// Whether an observer stopped the fit.
    pub stopped: bool,
    /// Junction shape parameters, usable to warm-start another fit.
    pub theta: Vec<f64>,
    /// Largest orthogonal distance of a point to the chain, where measured.
    pub max_deviation: Option<f64>,
}

impl FitResult {
    pub fn num_control_points(&self) -> Vec<usize> {
        self.control_points.iter().map(Vec::len).collect()
    }

    /// One curve per section.
    pub fn curves(&self) -> Vec<BSpline> {
        self.knots
            .iter()
            .zip(&self.control_points)
            .map(|(k, p)| BSpline { degree: self.degree, knots: k.clone(), control_points: p.clone() })
            .collect()
    }

    pub(crate) fn stop(mut self) -> Self {
        self.success = false;
        self.stopped = true;
        self.message = STOPPED.to_string();
        self
    }
}

/// Settings of [`fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct FitOptions {
    pub chain: ChainOptions,
    /// Starting data parameters; centripetal per section by default.
    pub initial_parameters: Option<Parameters>,
    /// Residual evaluations per solve; `None` runs until converged.
    pub max_nfev: Option<usize>,
    pub knot_placement: KnotPlacement,
    /// Start from the fewest control points and add knots where the fit is
    /// worst until each section has its count.
    pub grow: bool,
    /// Smallest spacing of free knots, as a fraction of a section (0 to 0.5).
    pub min_knot_gap: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            chain: ChainOptions::default(),
            initial_parameters: None,
            max_nfev: None,
            knot_placement: KnotPlacement::default(),
            grow: false,
            min_knot_gap: MIN_KNOT_GAP,
        }
    }
}

/// Every section has at least two finite points.
pub(crate) fn validate_sections(sections: &[Vec<Vec2>]) -> Result<()> {
    if sections.is_empty() || sections.iter().any(|s| s.len() < 2 || s.iter().any(|p| !p.iter().all(|v| v.is_finite()))) {
        bail!("Each section needs at least two finite points shaped (n, dim).");
    }
    Ok(())
}

/// Starting chain parameters: the given ones, or centripetal per section.
pub(crate) fn start_parameters(given: Option<&Parameters>, sections: &[Vec<Vec2>]) -> Result<Vec<f64>> {
    match given {
        Some(p) => validate_chain_parameters(p, sections),
        None => Ok(initial_chain_parameters(sections)),
    }
}

/// Least-squares fit of a chain of clamped B-splines to ordered points.
///
/// Control points and the curve parameter of every point are optimized
/// jointly: the control points are eliminated by variable projection and the
/// parameters, the shape parameters of geometric junctions and, with free
/// knots, the interior knot positions are solved by one Levenberg-Marquardt
/// loop with the exact reduced Jacobian. The sections only give the
/// starting split; a point's parameter slides along the chain, across
/// junctions, staying strictly increasing.
///
/// `counts` holds the control points of every section. The observer sees
/// every residual evaluation and can stop the fit; the result then holds the
/// best state evaluated so far.
pub fn fit(
    sections: &[Vec<Vec2>],
    counts: &[usize],
    degree: usize,
    options: &FitOptions,
    observer: &mut dyn Observer,
) -> Result<FitResult> {
    validate_sections(sections)?;
    if counts.len() != sections.len() {
        bail!("Need one control-point count per section.");
    }
    if counts.iter().any(|&n| n < degree + 1) {
        bail!("Control-point count must be at least degree + 1.");
    }
    let chain = &options.chain;
    if !chain.smoothing.is_finite() || chain.smoothing < 0.0 {
        bail!("Smoothing weight must be finite and nonnegative.");
    }
    if !chain.regularization.is_finite() || chain.regularization < 0.0 {
        bail!("Regularization must be finite and nonnegative.");
    }
    if options.max_nfev == Some(0) {
        bail!("max_nfev must be at least 1 or None.");
    }
    validate_min_knot_gap(options.min_knot_gap)?;
    let knots = counts.iter().map(|&n| open_uniform_knots(degree, n)).collect::<Result<Vec<_>>>()?;
    let u0 = start_parameters(options.initial_parameters.as_ref(), sections)?;
    let setup = Setup { sections, degree, options: chain, max_nfev: options.max_nfev, min_knot_gap: options.min_knot_gap };
    if options.grow {
        return grow_to_counts(setup, counts, u0, options.knot_placement, observer);
    }
    setup.refit(knots, &u0, None, options.knot_placement == KnotPlacement::Free, observer)
}

/// The reduced residual of a [`VarProProblem`] over `x = [η, θ]`.
struct Reduced<'a> {
    problem: &'a VarProProblem,
    observer: &'a mut dyn Observer,
    nfev: usize,
    /// Lowest-cost point evaluated: the result if the observer stops the solve.
    best: Option<(f64, DVector<f64>)>,
}

impl Reduced<'_> {
    fn unpack(&self, x: &DVector<f64>) -> (Vec<f64>, Vec<f64>) {
        let n_eta = self.problem.num_points() - 1;
        let s = self.problem.num_sections() as f64;
        (eta_to_t(&x.as_slice()[..n_eta], s), x.as_slice()[n_eta..].to_vec())
    }
}

impl Model for Reduced<'_> {
    fn residual(&mut self, x: &DVector<f64>) -> Result<DVector<f64>, Stopped> {
        let (u, theta) = self.unpack(x);
        let (r, p, rms) = self.problem.residual(&u, &theta);
        self.nfev += 1;
        let cost = r.norm_squared();
        if self.best.as_ref().is_none_or(|(c, _)| cost < *c) {
            self.best = Some((cost, x.clone()));
        }
        let control_points = self.problem.split_control_points(&p);
        let progress = Progress { control_points: &control_points, rms, knots: &self.problem.knots };
        if self.observer.step(&progress) {
            return Err(Stopped);
        }
        Ok(r)
    }

    fn jacobian(&mut self, x: &DVector<f64>) -> Result<DMatrix<f64>, Stopped> {
        let n_eta = self.problem.num_points() - 1;
        Ok(reduced_jacobian(self.problem, &x.as_slice()[..n_eta], &x.as_slice()[n_eta..]))
    }

    fn normal_equations(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> Result<(DMatrix<f64>, DVector<f64>), Stopped> {
        let n_eta = self.problem.num_points() - 1;
        Ok(reduced_normal_equations(self.problem, &x.as_slice()[..n_eta], &x.as_slice()[n_eta..], r, None))
    }
}

/// `(JᵀJ, Jᵀr)` of the reduced residual with respect to `[η, θ]` (and the
/// unknowns behind the `extra` columns of `J`, after `θ`), the
/// `η` block from the structure of `∂r/∂u` (`J` is never formed: it is
/// most of the cost of a step when there are many points).
pub(crate) fn reduced_normal_equations(
    problem: &VarProProblem,
    eta: &[f64],
    theta: &[f64],
    r: &DVector<f64>,
    extra: Option<&DMatrix<f64>>,
) -> (DMatrix<f64>, DVector<f64>) {
    let s = problem.num_sections() as f64;
    let u = eta_to_t(eta, s);
    // The dense columns: ∂r/∂θ, then any the caller adds.
    let parameters = problem.parameter_jacobian(&u, theta);
    let jt = match extra {
        Some(extra) => {
            let mut both = DMatrix::zeros(r.len(), parameters.ncols() + extra.ncols());
            both.columns_mut(0, parameters.ncols()).copy_from(&parameters);
            both.columns_mut(parameters.ncols(), extra.ncols()).copy_from(extra);
            both
        }
        None => parameters,
    };
    let (n, m) = (eta.len(), jt.ncols());
    let mut x = DMatrix::zeros(r.len(), 1 + m);
    x.set_column(0, r);
    x.view_mut((0, 1), jt.shape()).copy_from(&jt);
    let (normal, cross) = problem.normal_equations(&u, theta, &x);
    let cross = apply_eta_jacobian_transpose(&cross, eta, &u, s);
    let mut a = DMatrix::zeros(n + m, n + m);
    a.view_mut((0, 0), (n, n)).copy_from(&eta_congruence(&normal, eta, &u, s));
    let eta_theta = cross.columns(1, m);
    a.view_mut((0, n), (n, m)).copy_from(&eta_theta);
    a.view_mut((n, 0), (m, n)).copy_from(&eta_theta.transpose());
    a.view_mut((n, n), (m, m)).copy_from(&(jt.transpose() * &jt));
    let mut g = DVector::zeros(n + m);
    g.rows_mut(0, n).copy_from(&cross.column(0));
    g.rows_mut(n, m).copy_from(&(jt.transpose() * r));
    (a, g)
}

/// Jacobian of the reduced residual with respect to `[η, θ]`.
pub(crate) fn reduced_jacobian(problem: &VarProProblem, eta: &[f64], theta: &[f64]) -> DMatrix<f64> {
    let s = problem.num_sections() as f64;
    let u = eta_to_t(eta, s);
    let ju = apply_eta_jacobian(&problem.jacobian(&u, theta), eta, &u, s);
    let jt = problem.parameter_jacobian(&u, theta);
    let mut j = DMatrix::zeros(ju.nrows(), eta.len() + theta.len());
    j.view_mut((0, 0), ju.shape()).copy_from(&ju);
    j.view_mut((0, eta.len()), jt.shape()).copy_from(&jt);
    j
}

/// Runs Levenberg-Marquardt on a prepared problem from `(u0, θ0)`.
pub(crate) fn solve_problem(
    problem: &VarProProblem,
    u0: &[f64],
    theta0: &[f64],
    max_nfev: Option<usize>,
    observer: &mut dyn Observer,
) -> FitResult {
    let x0 = DVector::from_iterator(u0.len() - 1 + theta0.len(), t_to_eta(u0).into_iter().chain(theta0.iter().copied()));
    let mut model = Reduced { problem, observer, nfev: 0, best: None };
    let outcome = levenberg_marquardt(&mut model, x0, max_nfev, LM_TOLERANCE);
    let (x, success, message, stopped) = match outcome {
        Ok(s) => (s.x, s.status.success(), s.status.message(), false),
        Err(Stopped) => (model.best.clone().unwrap().1, false, STOPPED, true),
    };
    let (u, theta) = model.unpack(&x);
    let result = result_from_problem(problem, &u, &theta, model.nfev, success, message);
    if stopped {
        result.stop()
    } else {
        result
    }
}

/// A [`FitResult`] from the problem state at `(u, θ)`.
pub(crate) fn result_from_problem(
    problem: &VarProProblem,
    u: &[f64],
    theta: &[f64],
    nfev: usize,
    success: bool,
    message: &str,
) -> FitResult {
    let (_, p, rms) = problem.residual(u, theta);
    FitResult {
        control_points: problem.split_control_points(&p),
        parameters: u.to_vec(),
        knots: problem.knots.clone(),
        degree: problem.degree,
        rms_error: rms,
        nfev,
        success,
        message: message.to_string(),
        stopped: false,
        theta: theta.to_vec(),
        max_deviation: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::airfoil::Naca4;

    #[test]
    #[ignore = "timing; cargo test --release -p bspline-varpro reduced_normal_equations_timing -- --ignored --nocapture"]
    fn reduced_normal_equations_timing() {
        // Two sections of 130 points and 25 controls, G3: a free-knot step has
        // 40 knot columns besides θ.
        let points = Naca4::new("2412", true).unwrap().points(130);
        let sections = vec![points[..130].to_vec(), points[129..].to_vec()];
        let options =
            ChainOptions { continuity: Continuity::All(Junction { order: 3, geometric: true, fixed: true }), ..Default::default() };
        let problem = options.problem(&sections, vec![open_uniform_knots(5, 25).unwrap(); 2], 5).unwrap();
        let u = problem.initial_parameters();
        let eta = t_to_eta(&u);
        let theta = problem.theta.clone();
        let r = problem.residual(&eta_to_t(&eta, 2.0), &theta).0;
        let knots = DMatrix::from_fn(r.len(), 40, |i, j| ((i * 7 + j * 3) % 11) as f64 * 1e-3);
        for (name, extra) in [("without knot columns", None), ("with 40 knot columns", Some(&knots))] {
            let start = std::time::Instant::now();
            for _ in 0..100 {
                std::hint::black_box(reduced_normal_equations(&problem, &eta, &theta, &r, extra));
            }
            println!("{name}: {:.3} ms", start.elapsed().as_secs_f64() * 10.0);
        }
        let u = eta_to_t(&eta, 2.0);
        let x = DMatrix::from_column_slice(r.len(), 1, r.as_slice());
        let time = |name: &str, f: &dyn Fn()| {
            let start = std::time::Instant::now();
            for _ in 0..100 {
                f();
            }
            println!("  {name}: {:.3} ms", start.elapsed().as_secs_f64() * 10.0);
        };
        time("parameter_jacobian", &|| drop(std::hint::black_box(problem.parameter_jacobian(&u, &theta))));
        time("problem.normal_equations", &|| drop(std::hint::black_box(problem.normal_equations(&u, &theta, &x))));
        let (normal, _) = problem.normal_equations(&u, &theta, &x);
        time("eta_congruence", &|| drop(std::hint::black_box(eta_congruence(&normal, &eta, &u, 2.0))));
    }

    #[test]
    fn reduced_normal_equations_match_the_dense_jacobian() {
        let points = Naca4::new("2412", true).unwrap().points(21);
        let sections = vec![points[..21].to_vec(), points[20..].to_vec()];
        let options =
            ChainOptions { continuity: Continuity::All(Junction { order: 3, geometric: true, fixed: true }), ..Default::default() };
        let problem = options.problem(&sections, vec![open_uniform_knots(5, 9).unwrap(); 2], 5).unwrap();
        let mut u = problem.initial_parameters();
        u[5] += 0.02;
        let eta = t_to_eta(&u);
        let theta = vec![0.2, 0.1, -0.05];
        let u = eta_to_t(&eta, problem.num_sections() as f64);
        let r = problem.residual(&u, &theta).0;
        let j = reduced_jacobian(&problem, &eta, &theta);
        let (a, g) = reduced_normal_equations(&problem, &eta, &theta, &r, None);
        let expected = j.transpose() * &j;
        assert!((&a - &expected).amax() <= 1e-10 * expected.amax(), "JᵀJ {}", (&a - &expected).amax());
        let expected = j.transpose() * &r;
        assert!((&g - &expected).amax() <= 1e-10 * expected.amax(), "Jᵀr {}", (&g - &expected).amax());
    }
}
