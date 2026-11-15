//! Fitting chains of B-splines by Levenberg-Marquardt variable projection.

use nalgebra::{DMatrix, DVector};

use crate::basis::open_uniform_knots;
use crate::chain::{validate_chain_parameters, initial_chain_parameters, Parameters};
use crate::constraints::{End, Junction};
use crate::lm::{levenberg_marquardt, Model, Stopped};
use crate::problem::VarProProblem;
use crate::reparam::{eta_jacobian, eta_to_t, t_to_eta};
use crate::tolerance::{fit_subsampled, grow_to_counts, polish_knots};
use crate::{bail, BSpline, Observer, Progress, Result, Vec2, STOPPED};

/// Convergence tolerance (ftol, xtol and gtol) of every variable-projection solve.
pub(crate) const LM_TOLERANCE: f64 = 1e-10;

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

    /// The same chain with every geometric junction made parametric.
    pub fn parametric(&self) -> ChainOptions {
        let continuity = match &self.continuity {
            Continuity::All(j) => Continuity::All(j.parametric()),
            Continuity::Each(list) => Continuity::Each(list.iter().map(|j| j.parametric()).collect()),
        };
        ChainOptions { continuity, ..self.clone() }
    }

    pub fn problem(&self, sections: &[Vec<Vec2>], knots: Vec<Vec<f64>>, degree: usize) -> Result<VarProProblem> {
        let end = End { fixed: self.fix_ends };
        VarProProblem::new(
            sections,
            knots,
            degree,
            self.junctions(sections.len())?,
            (end, end),
            self.smoothing,
            self.regularization,
        )
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
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FitOptions {
    pub chain: ChainOptions,
    /// Starting data parameters; centripetal per section by default.
    pub initial_parameters: Option<Parameters>,
    /// For dense data: fit an evenly spaced subsample of longer sections by
    /// variable projection, then refine jointly on all points.
    pub max_points_per_section: Option<usize>,
    /// Residual evaluations per solve; `None` runs until converged.
    pub max_nfev: Option<usize>,
    pub knot_placement: KnotPlacement,
    /// Start from the fewest control points and add knots where the fit is
    /// worst until each section has its count.
    pub grow: bool,
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
/// parameters (and the shape parameters of geometric junctions) are solved by
/// Levenberg-Marquardt with the exact reduced Jacobian. The sections only
/// give the starting split; a point's parameter slides along the chain,
/// across junctions, staying strictly increasing.
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
    if options.max_points_per_section.is_some_and(|m| m < 2) {
        bail!("max_points_per_section must be at least 2.");
    }
    let knots = counts.iter().map(|&n| open_uniform_knots(degree, n)).collect::<Result<Vec<_>>>()?;
    let u0 = start_parameters(options.initial_parameters.as_ref(), sections)?;
    let dense = options.max_points_per_section.is_some_and(|m| sections.iter().any(|s| s.len() > m));
    let (max_nfev, max_points) = (options.max_nfev, options.max_points_per_section);
    if options.grow {
        return grow_to_counts(sections, counts, degree, chain, u0, max_nfev, max_points, options.knot_placement, observer);
    }
    if let Some(m) = max_points.filter(|_| dense) {
        return fit_subsampled(sections, knots, degree, chain, &u0, None, max_nfev, m, observer);
    }
    let problem = chain.problem(sections, knots, degree)?;
    let result = solve_with_continuation(&problem, chain, sections, &u0, max_nfev, observer)?;
    if options.knot_placement == KnotPlacement::Free && !result.stopped {
        let polished = polish_knots(sections, &result, chain, observer)?;
        if polished.stopped {
            return Ok(polished);
        }
        if polished.rms_error <= result.rms_error {
            // Moving the knots only refines a converged fit: keep its status.
            return Ok(FitResult { nfev: polished.nfev + result.nfev, success: result.success, message: result.message, ..polished });
        }
    }
    Ok(result)
}

/// Solves geometric junctions starting from the converged parametric
/// solution: the geometric problem has flat directions (shape parameters
/// trade off against the parameterization) that make a cold start drift.
pub(crate) fn solve_with_continuation(
    problem: &VarProProblem,
    options: &ChainOptions,
    sections: &[Vec<Vec2>],
    u0: &[f64],
    max_nfev: Option<usize>,
    observer: &mut dyn Observer,
) -> Result<FitResult> {
    if !problem.constraints.has_shape_parameters() {
        return Ok(solve_problem(problem, u0, &problem.theta, max_nfev, observer));
    }
    let base_problem = options.parametric().problem(sections, problem.knots.clone(), problem.degree)?;
    let base = solve_problem(&base_problem, u0, &base_problem.theta, max_nfev, observer);
    let theta0 = problem.constraints.parametric_params();
    let remaining = max_nfev.map(|m| m as i64 - base.nfev as i64);
    if base.stopped || remaining.is_some_and(|r| r < 1) {
        // The same curve as a geometric join (α = 1).
        let result = result_from_problem(problem, &base.parameters, &theta0, base.nfev, false, crate::lm::Status::Budget.message());
        return Ok(if base.stopped { result.stop() } else { result });
    }
    let mut result = solve_problem(problem, &base.parameters, &theta0, remaining.map(|r| r as usize), observer);
    result.nfev += base.nfev;
    Ok(result)
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
}

/// Jacobian of the reduced residual with respect to `[η, θ]`.
pub(crate) fn reduced_jacobian(problem: &VarProProblem, eta: &[f64], theta: &[f64]) -> DMatrix<f64> {
    let s = problem.num_sections() as f64;
    let u = eta_to_t(eta, s);
    let ju = problem.jacobian(&u, theta) * eta_jacobian(eta, &u, s);
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
    if stopped { result.stop() } else { result }
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
