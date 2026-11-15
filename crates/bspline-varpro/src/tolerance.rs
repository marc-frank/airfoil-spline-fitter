//! Tolerance-driven fitting: few control points within a maximum orthogonal
//! deviation.
//!
//! Starting from the fewest control points the end conditions allow, a knot
//! goes into every knot span that holds a point farther than the tolerance
//! from the chain, and the chain is refitted warm-started from the previous
//! parameters. Once the tolerance is met, interior knots are moved to where
//! they reduce the error most ("free-knot polish") and knots whose removal
//! keeps the fit within tolerance are removed.

use nalgebra::{DMatrix, DVector};

use crate::basis::{clamped_knots, distinct, interior_knots, num_control_points, open_uniform_knots};
use crate::chain::{chain_data, chain_deviations, groups, project_to_chain, section_ranges, Parameters};
use crate::constraints::minimum_control_points;
use crate::fitting::{
    reduced_jacobian, result_from_problem, solve_problem, solve_with_continuation, start_parameters, validate_sections,
    ChainOptions, FitResult, KnotPlacement,
};
use crate::joint::refine;
use crate::lm::{levenberg_marquardt, Model, Stopped};
use crate::problem::VarProProblem;
use crate::reparam::{eta_to_t, inverse_softplus, softplus, t_to_eta};
use crate::{bail, Observer, Progress, Result, Vec2};

/// Smallest knot span a knot insertion may create.
const MIN_SPAN: f64 = 1e-4;
/// Residual evaluations of a knot polish.
const POLISH_BUDGET: usize = 60;
/// Smallest polished knot span (unless the data are closer).
const MIN_GAP: f64 = 1e-3;

/// Refits on new knots, subsampling dense sections.
#[allow(clippy::too_many_arguments)]
fn refit(
    sections: &[Vec<Vec2>],
    knots: Vec<Vec<f64>>,
    degree: usize,
    options: &ChainOptions,
    u: &[f64],
    theta: Option<&[f64]>,
    max_nfev: Option<usize>,
    max_points: Option<usize>,
    observer: &mut dyn Observer,
) -> Result<FitResult> {
    if let Some(m) = max_points.filter(|&m| sections.iter().any(|s| s.len() > m)) {
        return fit_subsampled(sections, knots, degree, options, u, theta, max_nfev, m, observer);
    }
    let problem = options.problem(sections, knots, degree)?;
    match theta.filter(|t| t.len() == problem.num_params()) {
        Some(theta) => Ok(solve_problem(&problem, u, theta, max_nfev, observer)),
        None => solve_with_continuation(&problem, options, sections, u, max_nfev, observer),
    }
}

/// `m` evenly spaced indices of `0..n`, both ends included.
fn subsample_indices(n: usize, m: usize) -> Vec<usize> {
    if n <= m {
        return (0..n).collect();
    }
    let mut indices: Vec<usize> = crate::basis::linspace(0.0, (n - 1) as f64, m).iter().map(|v| v.round_ties_even() as usize).collect();
    indices.dedup();
    indices
}

/// Variable projection on a subsample, then joint refinement on all points.
///
/// Variable projection costs `O(n³)` per iteration in the number of points,
/// the joint solver `O(n k²)` in the number of control coordinates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_subsampled(
    sections: &[Vec<Vec2>],
    knots: Vec<Vec<f64>>,
    degree: usize,
    options: &ChainOptions,
    u: &[f64],
    theta: Option<&[f64]>,
    max_nfev: Option<usize>,
    max_points: usize,
    observer: &mut dyn Observer,
) -> Result<FitResult> {
    let data = chain_data(sections);
    let indices: Vec<Vec<usize>> = section_ranges(sections)
        .into_iter()
        .map(|(first, last)| subsample_indices(last - first + 1, max_points).into_iter().map(|i| first + i).collect())
        .collect();
    let subsets: Vec<Vec<Vec2>> = indices.iter().map(|ix| ix.iter().map(|&i| data[i]).collect()).collect();
    let mut chosen: Vec<usize> = indices.concat();
    chosen.sort_unstable();
    chosen.dedup();
    let u_chosen: Vec<f64> = chosen.iter().map(|&i| u[i]).collect();
    let mut coarse = refit(&subsets, knots, degree, options, &u_chosen, theta, max_nfev, None, observer)?;
    coarse.parameters = project_to_chain(&data, &coarse.curves(), &chosen, &coarse.parameters);
    if coarse.stopped {
        return Ok(coarse);
    }
    let mut refined = refine(sections, &coarse, options, observer)?;
    refined.nfev += coarse.nfev;
    Ok(refined)
}

/// Distinct knots and the span index of every parameter.
fn spans(knots: &[f64], u: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let breaks = distinct(knots);
    let last = breaks.len() - 2;
    let index = u.iter().map(|&v| breaks.partition_point(|&b| b <= v).saturating_sub(1).min(last)).collect();
    (breaks, index)
}

/// One new knot per span holding a point beyond `tol`, worst span first.
fn new_knots(knots: &[f64], u: &[f64], dev: &[f64], tol: f64) -> Vec<f64> {
    let (breaks, index) = spans(knots, u);
    let mut new: Vec<(f64, f64)> = Vec::new();
    for i in 0..breaks.len() - 1 {
        let inside: Vec<usize> = (0..u.len()).filter(|&k| index[k] == i).collect();
        if !inside.iter().any(|&k| dev[k] > tol) {
            continue;
        }
        let (a, b) = (breaks[i], breaks[i + 1]);
        if inside.len() < 2 || b - a < 2.0 * MIN_SPAN {
            continue;
        }
        let ts: Vec<f64> = inside.iter().map(|&k| u[k]).collect();
        let ds: Vec<f64> = inside.iter().map(|&k| dev[k]).collect();
        let worst = argmax(&ds);
        let mut v = ts[worst].clamp(a + 0.25 * (b - a), b - 0.25 * (b - a));
        if !(ts.iter().any(|&t| t < v) && ts.iter().any(|&t| t > v)) {
            v = median(&ts);
        }
        if v - a >= MIN_SPAN && b - v >= MIN_SPAN {
            new.push((ds[worst], v));
        }
    }
    new.sort_by(|x, y| y.partial_cmp(x).unwrap());
    new.into_iter().map(|(_, v)| v).collect()
}

/// Interior knots ordered by the largest deviation in their two adjacent spans.
fn removal_order(knots: &[f64], degree: usize, u: &[f64], dev: &[f64], tol: f64) -> Vec<f64> {
    let (breaks, index) = spans(knots, u);
    let mut candidates: Vec<(f64, f64)> = distinct(interior_knots(knots, degree))
        .into_iter()
        .map(|k| {
            let i = breaks.partition_point(|&b| b < k);
            let near = (0..u.len()).filter(|&p| index[p] + 1 == i || index[p] == i).map(|p| dev[p]).fold(0.0, f64::max);
            (near / tol, k)
        })
        .collect();
    candidates.sort_by(|x, y| x.partial_cmp(y).unwrap());
    candidates.into_iter().map(|(_, k)| k).collect()
}

fn argmax(values: &[f64]) -> usize {
    (0..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best })
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn max_deviation(data: &[Vec2], result: &FitResult) -> f64 {
    chain_deviations(data, &result.curves()).into_iter().fold(f64::NEG_INFINITY, f64::max)
}

/// Knot gap parameters of the sections a polish moves.
struct KnotGaps {
    degree: usize,
    /// Per section: the gap floors, or `None` for a section that keeps its knots.
    floors: Vec<Option<Vec<f64>>>,
    fixed: Vec<Vec<f64>>,
}

impl KnotGaps {
    fn to_xi(&self, knots: &[Vec<f64>]) -> Vec<f64> {
        let mut xi = Vec::new();
        for (k, floor) in knots.iter().zip(&self.floors) {
            if let Some(floor) = floor {
                for (gap, f) in gaps(interior_knots(k, self.degree)).iter().zip(floor) {
                    xi.push(inverse_softplus((gap - f).max(1e-12)));
                }
            }
        }
        xi
    }

    fn knots(&self, xi: &[f64]) -> Vec<Vec<f64>> {
        let mut offset = 0;
        self.floors
            .iter()
            .zip(&self.fixed)
            .map(|(floor, fixed)| match floor {
                None => fixed.clone(),
                Some(floor) => {
                    let gaps: Vec<f64> = floor.iter().enumerate().map(|(i, f)| softplus(xi[offset + i]) + f).collect();
                    offset += floor.len();
                    let total: f64 = gaps.iter().sum();
                    let mut sum = 0.0;
                    let interior: Vec<f64> = gaps[..gaps.len() - 1].iter().map(|g| { sum += g; sum / total }).collect();
                    clamped_knots(self.degree, &interior)
                }
            })
            .collect()
    }
}

/// Gaps of `[0, interior, 1]`.
fn gaps(interior: &[f64]) -> Vec<f64> {
    let mut points = vec![0.0];
    points.extend_from_slice(interior);
    points.push(1.0);
    points.windows(2).map(|w| w[1] - w[0]).collect()
}

/// The reduced residual over `[η, θ, ξ]` with knot gaps `ξ`.
struct Polish<'a> {
    sections: &'a [Vec<Vec2>],
    options: &'a ChainOptions,
    gaps: KnotGaps,
    n_eta: usize,
    n_theta: usize,
    cached: Option<(Vec<f64>, VarProProblem)>,
    observer: &'a mut dyn Observer,
}

impl Polish<'_> {
    fn split<'x>(&self, x: &'x DVector<f64>) -> (&'x [f64], &'x [f64], &'x [f64]) {
        let s = x.as_slice();
        (&s[..self.n_eta], &s[self.n_eta..self.n_eta + self.n_theta], &s[self.n_eta + self.n_theta..])
    }

    fn problem(&mut self, xi: &[f64]) -> Result<&VarProProblem, Stopped> {
        if self.cached.as_ref().is_none_or(|(k, _)| k != xi) {
            let problem = self.build(xi).ok_or(Stopped)?;
            self.cached = Some((xi.to_vec(), problem));
        }
        Ok(&self.cached.as_ref().unwrap().1)
    }

    fn build(&self, xi: &[f64]) -> Option<VarProProblem> {
        self.options.problem(self.sections, self.gaps.knots(xi), self.gaps.degree).ok()
    }
}

impl Model for Polish<'_> {
    fn residual(&mut self, x: &DVector<f64>) -> Result<DVector<f64>, Stopped> {
        let (eta, theta, xi) = self.split(x);
        let (eta, theta, xi) = (eta.to_vec(), theta.to_vec(), xi.to_vec());
        let s = self.sections.len() as f64;
        let problem = self.problem(&xi)?;
        let (r, p, rms) = problem.residual(&eta_to_t(&eta, s), &theta);
        let control_points = problem.split_control_points(&p);
        let knots = problem.knots.clone();
        if self.observer.step(&Progress { control_points: &control_points, rms, knots: &knots }) {
            return Err(Stopped);
        }
        Ok(r)
    }

    fn jacobian(&mut self, x: &DVector<f64>) -> Result<DMatrix<f64>, Stopped> {
        let rows = self.residual(x)?.len();
        let (eta, theta, xi) = self.split(x);
        let (eta, theta, xi) = (eta.to_vec(), theta.to_vec(), xi.to_vec());
        let s = self.sections.len() as f64;
        let u = eta_to_t(&eta, s);
        let mut j = DMatrix::zeros(rows, x.len());
        let problem = self.problem(&xi)?;
        let exact = reduced_jacobian(problem, &eta, &theta);
        j.view_mut((0, 0), exact.shape()).copy_from(&exact);
        let base = problem.residual(&u, &theta).0;
        for i in 0..xi.len() {
            let h = 1e-7 * (1.0 + xi[i].abs());
            let mut shifted = xi.clone();
            shifted[i] += h;
            let other = self.build(&shifted).ok_or(Stopped)?;
            let column = (other.residual(&u, &theta).0 - &base) / h;
            j.set_column(self.n_eta + self.n_theta + i, &column);
        }
        Ok(j)
    }
}

/// Optimizes the interior knot positions (at fixed counts) together with
/// the fit.
///
/// Knots stay ordered with a minimum spacing through a normalized softplus
/// gap parameterization. A span's minimum is the section's closest spacing
/// of the data parameters (or [`MIN_GAP`]), or just under its starting
/// length if that is shorter: knots packed closer than any two points leave
/// spans without data where the curve can wobble unseen. The budget stays
/// capped: the error is so flat in the knot positions that a converged
/// polish takes thousands of evaluations for well under a percent.
pub fn polish_knots(
    sections: &[Vec<Vec2>],
    result: &FitResult,
    options: &ChainOptions,
    observer: &mut dyn Observer,
) -> Result<FitResult> {
    let degree = result.degree;
    let interiors: Vec<&[f64]> = result.knots.iter().map(|k| interior_knots(k, degree)).collect();
    let counts: Vec<usize> =
        interiors.iter().map(|inner| if inner.windows(2).all(|w| w[1] > w[0]) { inner.len() } else { 0 }).collect();
    if counts.iter().sum::<usize>() == 0 {
        return Ok(result.clone());
    }
    let floors = groups(&result.parameters, sections.len())
        .into_iter()
        .zip(&interiors)
        .zip(&counts)
        .map(|(((_, local), inner), &r)| {
            if r == 0 {
                return None;
            }
            let mut ts = vec![0.0];
            ts.extend(local);
            ts.push(1.0);
            ts.sort_by(f64::total_cmp);
            ts.dedup();
            let closest = ts.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
            let floor: Vec<f64> = gaps(inner).iter().map(|g| closest.max(MIN_GAP).min(0.999 * g)).collect();
            let factor = (0.5 / floor.iter().sum::<f64>()).min(1.0);
            Some(floor.into_iter().map(|f| f * factor).collect())
        })
        .collect();
    let gaps = KnotGaps { degree, floors, fixed: result.knots.clone() };
    let xi0 = gaps.to_xi(&result.knots);
    let n_eta = result.parameters.len() - 1;
    let x0 = DVector::from_iterator(
        n_eta + result.theta.len() + xi0.len(),
        t_to_eta(&result.parameters).into_iter().chain(result.theta.iter().copied()).chain(xi0),
    );
    let mut model =
        Polish { sections, options, gaps, n_eta, n_theta: result.theta.len(), cached: None, observer };
    match levenberg_marquardt(&mut model, x0, Some(POLISH_BUDGET), crate::fitting::LM_TOLERANCE) {
        Err(Stopped) => Ok(result.clone().stop()),
        Ok(solution) => {
            let (eta, theta, xi) = model.split(&solution.x);
            let Some(problem) = model.build(xi) else { return Ok(result.clone()) };
            let u = eta_to_t(eta, sections.len() as f64);
            Ok(result_from_problem(&problem, &u, theta, solution.nfev, solution.status.success(), solution.status.message()))
        }
    }
}

/// Grows from the fewest control points to `counts`, refitting each step.
///
/// Each round adds one knot to every span whose largest deviation is at
/// least half the section's largest, worst first; when none of those spans
/// can be split, any span holding two points is. Free placement finally
/// polishes the knots (unless subsampled); uniform placement respaces the
/// section evenly with the new count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grow_to_counts(
    sections: &[Vec<Vec2>],
    counts: &[usize],
    degree: usize,
    options: &ChainOptions,
    mut u: Vec<f64>,
    max_nfev: Option<usize>,
    max_points: Option<usize>,
    placement: KnotPlacement,
    observer: &mut dyn Observer,
) -> Result<FitResult> {
    let need = minimum_control_points(&options.orders(sections.len())?, degree);
    let mut knots: Vec<Vec<f64>> =
        need.iter().zip(counts).map(|(&n, &c)| open_uniform_knots(degree, n.min(c))).collect::<Result<_>>()?;
    let data = chain_data(sections);
    let mut theta: Option<Vec<f64>> = None;
    let mut result;
    loop {
        result = refit(sections, knots.clone(), degree, options, &u, theta.as_deref(), max_nfev, max_points, observer)?;
        if result.stopped {
            return Ok(result);
        }
        u = result.parameters.clone();
        theta = Some(result.theta.clone());
        let missing: Vec<i64> = counts.iter().zip(&knots).map(|(&c, k)| c as i64 - num_control_points(k, degree) as i64).collect();
        if missing.iter().all(|&m| m <= 0) {
            break;
        }
        let devs = chain_deviations(&data, &result.curves());
        let mut inserted = false;
        for (s, (rows, local)) in groups(&u, sections.len()).into_iter().enumerate() {
            let room = missing[s];
            if room <= 0 {
                continue;
            }
            let dev: Vec<f64> = rows.iter().map(|&i| devs[i]).collect();
            let worst = dev.iter().copied().fold(0.0, f64::max);
            let mut new = new_knots(&knots[s], &local, &dev, 0.5 * worst);
            if new.is_empty() {
                new = new_knots(&knots[s], &local, &dev, -1.0);
            }
            new.truncate(room as usize);
            match placement {
                KnotPlacement::Uniform => {
                    knots[s] = open_uniform_knots(degree, num_control_points(&knots[s], degree) + new.len().max(1))?;
                    inserted = true;
                }
                KnotPlacement::Free if !new.is_empty() => {
                    knots[s] = with_knots(&knots[s], degree, &new);
                    inserted = true;
                }
                KnotPlacement::Free => {}
            }
        }
        if !inserted {
            break;
        }
    }
    let all_used = max_points.is_none_or(|m| sections.iter().all(|s| s.len() <= m));
    if placement == KnotPlacement::Free && all_used {
        let polished = polish_knots(sections, &result, options, observer)?;
        if polished.stopped {
            return Ok(polished);
        }
        if polished.rms_error <= result.rms_error {
            return Ok(FitResult { nfev: polished.nfev + result.nfev, success: result.success, message: result.message, ..polished });
        }
    }
    Ok(result)
}

/// `knots` with the `new` interior values added.
fn with_knots(knots: &[f64], degree: usize, new: &[f64]) -> Vec<f64> {
    let mut interior = interior_knots(knots, degree).to_vec();
    interior.extend_from_slice(new);
    clamped_knots(degree, &interior)
}

/// Settings of [`fit_to_tolerance`].
#[derive(Debug, Clone, PartialEq)]
pub struct ToleranceOptions {
    pub chain: ChainOptions,
    /// Start from this many control points per section (raised to the
    /// fewest the end conditions allow) instead of the fewest.
    pub initial_control_points: Option<usize>,
    pub initial_parameters: Option<Parameters>,
    /// Upper limit of control points per section.
    pub max_control_points: Option<usize>,
    /// Polish the interior knot positions once the tolerance is met.
    pub optimize_knots: bool,
    /// Remove knots that are not needed once the tolerance is met.
    pub remove_knots: bool,
    /// Fit longer sections on a subsample, then refine on all points.
    pub max_points_per_section: Option<usize>,
    /// Residual evaluations per refit; `None` runs each until converged.
    pub max_nfev: Option<usize>,
    /// Maximum insertion rounds.
    pub max_iterations: usize,
    /// Free: insert knots where needed, then polish and remove. Uniform:
    /// keep evenly spaced knots and choose the count per section.
    pub knot_placement: KnotPlacement,
}

impl Default for ToleranceOptions {
    fn default() -> Self {
        ToleranceOptions {
            chain: ChainOptions { smoothing: 1e-7, ..ChainOptions::default() },
            initial_control_points: None,
            initial_parameters: None,
            max_control_points: None,
            optimize_knots: true,
            remove_knots: true,
            max_points_per_section: Some(200),
            max_nfev: None,
            max_iterations: 40,
            knot_placement: KnotPlacement::Free,
        }
    }
}

/// Wraps an observer and remembers whether a step asked to stop.
struct Steps<'a> {
    inner: &'a mut dyn Observer,
    stopped: bool,
}

impl Observer for Steps<'_> {
    fn step(&mut self, progress: &Progress) -> bool {
        let stop = self.inner.step(progress);
        self.stopped |= stop;
        stop
    }
}

/// Fits a chain with few control points so every point lies within
/// `tolerance` of it.
///
/// The observer's `step` sees every residual evaluation of every refit and
/// polish, its `refit` every finished refit with its maximum deviation;
/// either stops the search. The result carries `max_deviation`; `success`
/// is false when the tolerance could not be met.
pub fn fit_to_tolerance(
    sections: &[Vec<Vec2>],
    tolerance: f64,
    degree: usize,
    options: &ToleranceOptions,
    observer: &mut dyn Observer,
) -> Result<FitResult> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        bail!("Tolerance must be positive.");
    }
    if options.max_iterations < 1 {
        bail!("max_iterations must be at least 1.");
    }
    validate_sections(sections)?;
    let chain = &options.chain;
    let uniform = options.knot_placement == KnotPlacement::Uniform;
    let num_sections = sections.len();
    let need = minimum_control_points(&chain.orders(num_sections)?, degree);
    let mut knots: Vec<Vec<f64>> = need
        .iter()
        .map(|&n| open_uniform_knots(degree, options.initial_control_points.map_or(n, |c| c.max(n))))
        .collect::<Result<_>>()?;
    let limits: Vec<usize> =
        sections.iter().map(|s| (s.len() - 1).min(options.max_control_points.unwrap_or(s.len()))).collect();
    if need.iter().zip(&limits).any(|(n, l)| n > l) {
        bail!("Sections have too few points for the requested end conditions.");
    }
    let mut u = start_parameters(options.initial_parameters.as_ref(), sections)?;
    let data = chain_data(sections);
    let max_points = options.max_points_per_section;
    let mut theta: Option<Vec<f64>> = None;
    let mut steps = Steps { inner: observer, stopped: false };

    let finish = |mut result: FitResult, deviation: f64, success: bool, message: String| {
        result.max_deviation = Some(deviation);
        result.success = success;
        result.stopped = message == crate::STOPPED;
        result.message = message;
        Ok(result)
    };
    let stopped = |result: FitResult, deviation: f64| finish(result, deviation, false, crate::STOPPED.to_string());
    let done = |result: FitResult, deviation: f64| finish(result, deviation, true, format!("Within tolerance {}.", crate::numfmt::general6(tolerance)));

    let mut result = None;
    let mut deviation = f64::NAN;
    let mut met = false;
    for _ in 0..options.max_iterations {
        let fitted = refit(sections, knots.clone(), degree, chain, &u, theta.as_deref(), options.max_nfev, max_points, &mut steps)?;
        u = fitted.parameters.clone();
        theta = Some(fitted.theta.clone());
        let devs = chain_deviations(&data, &fitted.curves());
        deviation = devs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if fitted.stopped || steps.inner.refit(&fitted, deviation) {
            return stopped(fitted, deviation);
        }
        if deviation <= tolerance {
            result = Some(fitted);
            met = true;
            break;
        }
        let mut inserted = false;
        for (s, (rows, local)) in groups(&u, num_sections).into_iter().enumerate() {
            let dev: Vec<f64> = rows.iter().map(|&i| devs[i]).collect();
            if dev.iter().copied().fold(0.0, f64::max) <= tolerance {
                continue;
            }
            let count = num_control_points(&knots[s], degree);
            let room = limits[s] as i64 - count as i64;
            let mut new = new_knots(&knots[s], &local, &dev, tolerance);
            new.truncate(room.max(0) as usize);
            if uniform && room > 0 {
                knots[s] = open_uniform_knots(degree, count + new.len().max(1))?;
                inserted = true;
            } else if !uniform && !new.is_empty() {
                knots[s] = with_knots(&knots[s], degree, &new);
                inserted = true;
            }
        }
        if !inserted {
            return finish(fitted, deviation, false, "Tolerance not reached: no knot span can be refined further.".into());
        }
        result = Some(fitted);
    }
    let mut result = result.expect("at least one round");
    if !met {
        return finish(result, deviation, false, "Tolerance not reached within max_iterations.".into());
    }

    if uniform {
        for s in 0..num_sections {
            while num_control_points(&result.knots[s], degree) > need[s] {
                let mut trial_knots = result.knots.clone();
                trial_knots[s] = open_uniform_knots(degree, num_control_points(&result.knots[s], degree) - 1)?;
                let trial = refit(sections, trial_knots, degree, chain, &result.parameters, Some(&result.theta), options.max_nfev, max_points, &mut steps)?;
                if trial.stopped {
                    return stopped(result, deviation);
                }
                let trial_deviation = max_deviation(&data, &trial);
                if trial_deviation > tolerance {
                    break;
                }
                result = trial;
                deviation = trial_deviation;
                if steps.inner.refit(&result, deviation) {
                    return stopped(result, deviation);
                }
            }
        }
        return done(result, deviation);
    }

    let dense = max_points.is_some_and(|m| sections.iter().any(|s| s.len() > m));
    let polish = options.optimize_knots && !dense;
    // The polished fit if it stays within tolerance without raising the RMS error.
    let polished = |result: &FitResult, steps: &mut Steps| -> Result<Option<(FitResult, f64)>> {
        let candidate = polish_knots(sections, result, chain, steps)?;
        if candidate.stopped {
            return Ok(None);
        }
        let candidate_deviation = max_deviation(&data, &candidate);
        Ok((candidate_deviation <= tolerance && candidate.rms_error <= result.rms_error).then_some((candidate, candidate_deviation)))
    };
    if polish {
        if let Some((better, better_deviation)) = polished(&result, &mut steps)? {
            result = better;
            deviation = better_deviation;
            if steps.inner.refit(&result, deviation) {
                return stopped(result, deviation);
            }
        }
        if steps.stopped {
            return stopped(result, deviation);
        }
    }
    if options.remove_knots {
        let mut removed_any = false;
        for s in 0..num_sections {
            let mut removed = true;
            while removed && num_control_points(&result.knots[s], degree) > need[s] {
                removed = false;
                let devs = chain_deviations(&data, &result.curves());
                let (rows, local) = groups(&result.parameters, num_sections).swap_remove(s);
                let dev: Vec<f64> = rows.iter().map(|&i| devs[i]).collect();
                for k in removal_order(&result.knots[s], degree, &local, &dev, tolerance) {
                    let mut inner = interior_knots(&result.knots[s], degree).to_vec();
                    let nearest = (0..inner.len()).fold(0, |b, i| if (inner[i] - k).abs() < (inner[b] - k).abs() { i } else { b });
                    inner.remove(nearest);
                    let mut trial_knots = result.knots.clone();
                    trial_knots[s] = clamped_knots(degree, &inner);
                    let mut trial = refit(sections, trial_knots, degree, chain, &result.parameters, Some(&result.theta), options.max_nfev, max_points, &mut steps)?;
                    if trial.stopped {
                        return stopped(result, deviation);
                    }
                    let mut trial_deviation = max_deviation(&data, &trial);
                    if trial_deviation > tolerance && polish {
                        trial = polish_knots(sections, &trial, chain, &mut steps)?;
                        if trial.stopped {
                            return stopped(result, deviation);
                        }
                        trial_deviation = max_deviation(&data, &trial);
                    }
                    if trial_deviation <= tolerance {
                        result = trial;
                        deviation = trial_deviation;
                        removed = true;
                        removed_any = true;
                        if steps.inner.refit(&result, deviation) {
                            return stopped(result, deviation);
                        }
                        break;
                    }
                }
            }
        }
        if polish && removed_any && !steps.stopped {
            if let Some((better, better_deviation)) = polished(&result, &mut steps)? {
                result = better;
                deviation = better_deviation;
                steps.inner.refit(&result, deviation);
            }
        }
    }
    done(result, deviation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsample_keeps_ends() {
        assert_eq!(subsample_indices(5, 10), vec![0, 1, 2, 3, 4]);
        let s = subsample_indices(101, 11);
        assert_eq!((s[0], s[s.len() - 1], s.len()), (0, 100, 11));
    }

    #[test]
    fn knots_go_to_the_worst_span() {
        let knots = open_uniform_knots(3, 5).unwrap();
        let u = [0.1, 0.2, 0.3, 0.6, 0.7, 0.9];
        let dev = [0.0, 0.0, 0.0, 0.5, 0.2, 0.0];
        assert_eq!(new_knots(&knots, &u, &dev, 0.1), vec![0.625]);
    }
}
