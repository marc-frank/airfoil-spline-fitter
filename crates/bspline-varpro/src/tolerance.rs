//! Free knots, growth to control-point counts and the tolerance search.
//!
//! Every procedure here is a sequence of one kind of fit: the single
//! optimization of [`solve_problem`](crate::fitting) or, with free knots,
//! of [`solve_free_knots`], whose unknowns include the interior knot
//! positions. Nothing runs between or after those fits. The tolerance search
//! starts from the fewest control points the end conditions allow and adds
//! one control point per round to every section beyond the tolerance (a
//! knot in its worst span), refitting warm-started; once the tolerance is
//! met it removes the knots that are not needed.

use nalgebra::{DMatrix, DVector};

use crate::basis::{clamped_knots, distinct, interior_knots, num_control_points, open_uniform_knots};
use crate::chain::{chain_data, chain_deviations, groups, Parameters};
use crate::constraints::minimum_control_points;
use crate::fitting::{
    reduced_jacobian, reduced_normal_equations, result_from_problem, solve_problem, start_parameters, validate_min_knot_gap,
    validate_sections, ChainOptions, FitResult, KnotPlacement, LM_TOLERANCE, MIN_KNOT_GAP,
};
use crate::lm::{levenberg_marquardt, Model, Stopped};
use crate::problem::VarProProblem;
use crate::reparam::{eta_to_t, inverse_softplus, softplus, t_to_eta};
use crate::{bail, Observer, Progress, Result, Vec2, STOPPED};

/// Smallest knot span an insertion of uniform knots may create.
const MIN_SPAN: f64 = 1e-4;

/// What every fit of one procedure shares: the data, degree, constraints,
/// budget and the knot setting.
#[derive(Clone, Copy)]
pub(crate) struct Setup<'a> {
    pub sections: &'a [Vec<Vec2>],
    pub degree: usize,
    pub options: &'a ChainOptions,
    /// Residual evaluations per fit.
    pub max_nfev: Option<usize>,
    /// Smallest spacing of free knots, as a fraction of a section.
    pub min_knot_gap: f64,
}

impl Setup<'_> {
    /// The one fit every procedure repeats: from `(u, θ)` (the problem's
    /// initial θ when `None` or of another length), with `free` knots among
    /// the unknowns or the knots kept.
    pub(crate) fn refit(
        &self,
        knots: Vec<Vec<f64>>,
        u: &[f64],
        theta: Option<&[f64]>,
        free: bool,
        observer: &mut dyn Observer,
    ) -> Result<FitResult> {
        let problem = self.options.problem(self.sections, knots.clone(), self.degree)?;
        let theta = theta.filter(|t| t.len() == problem.num_params()).map_or_else(|| problem.theta.clone(), <[f64]>::to_vec);
        if free {
            solve_free_knots(self.sections, knots, self.degree, self.options, u, &theta, self.max_nfev, self.min_knot_gap, observer)
        } else {
            Ok(solve_problem(&problem, u, &theta, self.max_nfev, observer))
        }
    }
}

/// Distinct knots and the span index of every parameter.
fn spans(knots: &[f64], u: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let breaks = distinct(knots);
    let last = breaks.len() - 2;
    let index = u.iter().map(|&v| breaks.partition_point(|&b| b <= v).saturating_sub(1).min(last)).collect();
    (breaks, index)
}

/// One new knot per span holding a point beyond `tol`, worst span first; no
/// span closer than `min_span`.
fn new_knots(knots: &[f64], u: &[f64], dev: &[f64], tol: f64, min_span: f64) -> Vec<f64> {
    let (breaks, index) = spans(knots, u);
    let mut new: Vec<(f64, f64)> = Vec::new();
    for i in 0..breaks.len() - 1 {
        let inside: Vec<usize> = (0..u.len()).filter(|&k| index[k] == i).collect();
        if !inside.iter().any(|&k| dev[k] > tol) {
            continue;
        }
        let (a, b) = (breaks[i], breaks[i + 1]);
        if inside.len() < 2 || b - a < 2.0 * min_span {
            continue;
        }
        let ts: Vec<f64> = inside.iter().map(|&k| u[k]).collect();
        let ds: Vec<f64> = inside.iter().map(|&k| dev[k]).collect();
        let worst = argmax(&ds);
        let mut v = ts[worst].clamp(a + 0.25 * (b - a), b - 0.25 * (b - a));
        if !(ts.iter().any(|&t| t < v) && ts.iter().any(|&t| t > v)) {
            v = median(&ts);
        }
        if v - a >= min_span && b - v >= min_span {
            new.push((ds[worst], v));
        }
    }
    new.sort_by(|x, y| y.partial_cmp(x).unwrap());
    new.into_iter().map(|(_, v)| v).collect()
}

/// Whether a span holding a point beyond `tol` is too narrow to split at
/// `min_span`.
fn too_narrow(knots: &[f64], u: &[f64], dev: &[f64], tol: f64, min_span: f64) -> bool {
    let (breaks, index) = spans(knots, u);
    (0..breaks.len() - 1).any(|i| breaks[i + 1] - breaks[i] < 2.0 * min_span && (0..u.len()).any(|k| index[k] == i && dev[k] > tol))
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
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn max_deviation(data: &[Vec2], result: &FitResult) -> f64 {
    chain_deviations(data, &result.curves()).into_iter().fold(f64::NEG_INFINITY, f64::max)
}

/// `knots` with the `new` interior values added.
fn with_knots(knots: &[f64], degree: usize, new: &[f64]) -> Vec<f64> {
    let mut interior = interior_knots(knots, degree).to_vec();
    interior.extend_from_slice(new);
    clamped_knots(degree, &interior)
}

/// Gaps of `[0, interior, 1]`.
fn gaps(interior: &[f64]) -> Vec<f64> {
    let mut points = vec![0.0];
    points.extend_from_slice(interior);
    points.push(1.0);
    points.windows(2).map(|w| w[1] - w[0]).collect()
}

/// The interior knots of a chain as unconstrained unknowns.
///
/// Knots stay ordered with a minimum spacing through a normalized softplus
/// gap parameterization: gap `i` of a section is `floor_i + (1 - Σfloor)
/// softplus(ξ_i) / Σsoftplus(ξ)`, so the floor holds in the knots. A
/// span's floor is the section's closest spacing of the data parameters (or
/// the minimum knot gap, if larger), or just under the span's starting
/// length if that is shorter: knots packed closer than any two points leave
/// spans without data where the curve can wobble unseen. Sections with a
/// repeated interior knot keep their knots.
struct KnotGaps {
    degree: usize,
    /// Per section: the gap floors, or `None` for a section that keeps its knots.
    floors: Vec<Option<Vec<f64>>>,
    fixed: Vec<Vec<f64>>,
}

impl KnotGaps {
    fn new(sections: &[Vec<Vec2>], knots: &[Vec<f64>], degree: usize, parameters: &[f64], min_knot_gap: f64) -> Self {
        let floors = groups(parameters, sections.len())
            .into_iter()
            .zip(knots)
            .map(|((_, local), k)| {
                let inner = interior_knots(k, degree);
                if inner.is_empty() || !inner.windows(2).all(|w| w[1] > w[0]) {
                    return None;
                }
                let mut ts = vec![0.0];
                ts.extend(local);
                ts.push(1.0);
                ts.sort_by(f64::total_cmp);
                ts.dedup();
                let closest = ts.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
                let floor: Vec<f64> = gaps(inner).iter().map(|g| closest.max(min_knot_gap).min(0.999 * g)).collect();
                let factor = (0.5 / floor.iter().sum::<f64>()).min(1.0);
                Some(floor.into_iter().map(|f| f * factor).collect())
            })
            .collect();
        KnotGaps { degree, floors, fixed: knots.to_vec() }
    }

    /// Number of unknowns.
    fn count(&self) -> usize {
        self.floors.iter().flatten().map(Vec::len).sum()
    }

    /// The unknowns of the starting knots.
    fn start(&self) -> Vec<f64> {
        let mut xi = Vec::new();
        for (k, floor) in self.fixed.iter().zip(&self.floors) {
            if let Some(floor) = floor {
                for (gap, f) in gaps(interior_knots(k, self.degree)).iter().zip(floor) {
                    xi.push(inverse_softplus((gap - f).max(1e-12)));
                }
            }
        }
        xi
    }

    /// The knot vectors of unknowns `xi`.
    fn knots(&self, xi: &[f64]) -> Vec<Vec<f64>> {
        let mut offset = 0;
        self.floors
            .iter()
            .zip(&self.fixed)
            .map(|(floor, fixed)| match floor {
                None => fixed.clone(),
                Some(floor) => {
                    let free: Vec<f64> = (0..floor.len()).map(|i| softplus(xi[offset + i])).collect();
                    offset += floor.len();
                    let (free_sum, floor_sum) = (free.iter().sum::<f64>(), floor.iter().sum::<f64>());
                    let mut sum = 0.0;
                    let interior: Vec<f64> = floor[..floor.len() - 1]
                        .iter()
                        .zip(&free)
                        .map(|(f, s)| {
                            sum += f + (1.0 - floor_sum) * s / free_sum;
                            sum
                        })
                        .collect();
                    clamped_knots(self.degree, &interior)
                }
            })
            .collect()
    }
}

/// The reduced residual over `[η, θ, ξ]` with knot gaps `ξ`.
struct FreeKnots<'a> {
    sections: &'a [Vec<Vec2>],
    options: &'a ChainOptions,
    gaps: KnotGaps,
    n_eta: usize,
    n_theta: usize,
    cached: Option<(Vec<f64>, VarProProblem)>,
    observer: &'a mut dyn Observer,
    nfev: usize,
    /// Lowest-cost point evaluated: the result if the observer stops the fit.
    best: Option<(f64, DVector<f64>)>,
}

impl FreeKnots<'_> {
    fn split<'x>(&self, x: &'x DVector<f64>) -> (&'x [f64], &'x [f64], &'x [f64]) {
        let s = x.as_slice();
        (&s[..self.n_eta], &s[self.n_eta..self.n_eta + self.n_theta], &s[self.n_eta + self.n_theta..])
    }

    fn build(&self, xi: &[f64]) -> Result<VarProProblem> {
        self.options.problem(self.sections, self.gaps.knots(xi), self.gaps.degree)
    }

    /// `∂r/∂ξ` by forward differences.
    fn knot_columns(&mut self, x: &DVector<f64>) -> DMatrix<f64> {
        let (eta, theta, xi) = self.split(x);
        let (theta, xi) = (theta.to_vec(), xi.to_vec());
        let u = eta_to_t(eta, self.sections.len() as f64);
        let base = self.problem(&xi).residual(&u, &theta).0;
        // Each column builds and solves a problem of its own (most of a
        // free-knot step), independent of the others: split over the cores.
        let (sections, options, gaps) = (self.sections, self.options, &self.gaps);
        let column = |i: usize| {
            let h = 1e-7 * (1.0 + xi[i].abs());
            let mut shifted = xi.clone();
            shifted[i] += h;
            let other = options.problem(sections, gaps.knots(&shifted), gaps.degree).expect("knots of the gap parameterization are valid");
            (other.residual(&u, &theta).0 - &base) / h
        };
        // At least four columns a thread, so few knots do not pay for threads.
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, xi.len().div_ceil(4).max(1));
        let per_thread = xi.len().div_ceil(threads).max(1);
        let indices: Vec<usize> = (0..xi.len()).collect();
        let computed: Vec<DVector<f64>> = std::thread::scope(|scope| {
            let handles: Vec<_> =
                indices.chunks(per_thread).map(|chunk| scope.spawn(|| chunk.iter().map(|&i| column(i)).collect::<Vec<_>>())).collect();
            handles.into_iter().flat_map(|h| h.join().expect("a knot column panicked")).collect()
        });
        DMatrix::from_columns(&computed)
    }

    fn problem(&mut self, xi: &[f64]) -> &VarProProblem {
        if self.cached.as_ref().is_none_or(|(k, _)| k != xi) {
            let problem = self.build(xi).expect("knots of the gap parameterization are valid");
            self.cached = Some((xi.to_vec(), problem));
        }
        &self.cached.as_ref().unwrap().1
    }
}

impl Model for FreeKnots<'_> {
    fn residual(&mut self, x: &DVector<f64>) -> Result<DVector<f64>, Stopped> {
        let (eta, theta, xi) = self.split(x);
        let (eta, theta, xi) = (eta.to_vec(), theta.to_vec(), xi.to_vec());
        let s = self.sections.len() as f64;
        let problem = self.problem(&xi);
        let (r, p, rms) = problem.residual(&eta_to_t(&eta, s), &theta);
        let control_points = problem.split_control_points(&p);
        let knots = problem.knots.clone();
        self.nfev += 1;
        let cost = r.norm_squared();
        if self.best.as_ref().is_none_or(|(c, _)| cost < *c) {
            self.best = Some((cost, x.clone()));
        }
        if self.observer.step(&Progress { control_points: &control_points, rms, knots: &knots }) {
            return Err(Stopped);
        }
        Ok(r)
    }

    fn jacobian(&mut self, x: &DVector<f64>) -> Result<DMatrix<f64>, Stopped> {
        let (eta, theta, xi) = self.split(x);
        let (eta, theta, xi) = (eta.to_vec(), theta.to_vec(), xi.to_vec());
        let knots = self.knot_columns(x);
        let exact = reduced_jacobian(self.problem(&xi), &eta, &theta);
        let mut j = DMatrix::zeros(exact.nrows(), x.len());
        j.view_mut((0, 0), exact.shape()).copy_from(&exact);
        j.columns_mut(self.n_eta + self.n_theta, xi.len()).copy_from(&knots);
        Ok(j)
    }

    fn normal_equations(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> Result<(DMatrix<f64>, DVector<f64>), Stopped> {
        let (eta, theta, xi) = self.split(x);
        let (eta, theta, xi) = (eta.to_vec(), theta.to_vec(), xi.to_vec());
        let knots = self.knot_columns(x);
        Ok(reduced_normal_equations(self.problem(&xi), &eta, &theta, r, Some(&knots)))
    }
}

/// Levenberg-Marquardt with the interior knots among the unknowns.
///
/// The same solve as the fixed-knot fit, from the same start, with the knot
/// positions (at fixed counts) optimized together with the data parameters
/// and the junction shape parameters: one loop from the first step to
/// convergence. Knot columns of the Jacobian are forward differences; the
/// others are exact. A chain without a movable knot is solved with its
/// knots kept.
#[allow(clippy::too_many_arguments)]
pub fn solve_free_knots(
    sections: &[Vec<Vec2>],
    knots: Vec<Vec<f64>>,
    degree: usize,
    options: &ChainOptions,
    u0: &[f64],
    theta0: &[f64],
    max_nfev: Option<usize>,
    min_knot_gap: f64,
    observer: &mut dyn Observer,
) -> Result<FitResult> {
    let gaps = KnotGaps::new(sections, &knots, degree, u0, min_knot_gap);
    if gaps.count() == 0 {
        let problem = options.problem(sections, knots, degree)?;
        return Ok(solve_problem(&problem, u0, theta0, max_nfev, observer));
    }
    let n_eta = u0.len() - 1;
    let x0 = DVector::from_iterator(
        n_eta + theta0.len() + gaps.count(),
        t_to_eta(u0).into_iter().chain(theta0.iter().copied()).chain(gaps.start()),
    );
    let mut model = FreeKnots { sections, options, gaps, n_eta, n_theta: theta0.len(), cached: None, observer, nfev: 0, best: None };
    let (x, success, message, stopped) = match levenberg_marquardt(&mut model, x0, max_nfev, LM_TOLERANCE) {
        Ok(s) => (s.x, s.status.success(), s.status.message(), false),
        Err(Stopped) => (model.best.clone().unwrap().1, false, STOPPED, true),
    };
    let (eta, theta, xi) = model.split(&x);
    let problem = model.build(xi)?;
    let u = eta_to_t(eta, sections.len() as f64);
    let result = result_from_problem(&problem, &u, theta, model.nfev, success, message);
    Ok(if stopped { result.stop() } else { result })
}

/// Grows from the fewest control points to `counts`, refitting each step.
///
/// Each round adds one knot to every span whose largest deviation is at
/// least half the section's largest, worst first; when none of those spans
/// can be split, any span holding two points is. Free knots go into those
/// spans and are unknowns of every refit, not closer than the minimum gap
/// (or half the uniform spacing of the largest count, if that is less);
/// uniform placement respaces the section evenly with the new count.
pub(crate) fn grow_to_counts(
    setup: Setup,
    counts: &[usize],
    mut u: Vec<f64>,
    placement: KnotPlacement,
    observer: &mut dyn Observer,
) -> Result<FitResult> {
    let Setup { sections, degree, options, .. } = setup;
    let free = placement == KnotPlacement::Free;
    let need = minimum_control_points(&options.orders(sections.len())?, degree);
    let mut knots: Vec<Vec<f64>> = need.iter().zip(counts).map(|(&n, &c)| open_uniform_knots(degree, n.min(c))).collect::<Result<_>>()?;
    let largest = counts.iter().copied().max().unwrap_or(0);
    let min_span = if free { MIN_SPAN.max(setup.min_knot_gap.min(0.5 / (largest.saturating_sub(degree)).max(1) as f64)) } else { MIN_SPAN };
    let data = chain_data(sections);
    let mut theta: Option<Vec<f64>> = None;
    loop {
        let result = setup.refit(knots.clone(), &u, theta.as_deref(), free, observer)?;
        if result.stopped {
            return Ok(result);
        }
        u = result.parameters.clone();
        theta = Some(result.theta.clone());
        knots = result.knots.clone();
        let missing: Vec<i64> = counts.iter().zip(&knots).map(|(&c, k)| c as i64 - num_control_points(k, degree) as i64).collect();
        if missing.iter().all(|&m| m <= 0) {
            return Ok(result);
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
            let mut new = new_knots(&knots[s], &local, &dev, 0.5 * worst, min_span);
            if new.is_empty() {
                new = new_knots(&knots[s], &local, &dev, -1.0, min_span);
            }
            new.truncate(room as usize);
            if !free {
                knots[s] = open_uniform_knots(degree, num_control_points(&knots[s], degree) + new.len().max(1))?;
                inserted = true;
            } else if !new.is_empty() {
                knots[s] = with_knots(&knots[s], degree, &new);
                inserted = true;
            }
        }
        if !inserted {
            return Ok(result);
        }
    }
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
    /// Remove knots that are not needed once the tolerance is met.
    pub remove_knots: bool,
    /// Residual evaluations per refit; `None` runs each until converged.
    pub max_nfev: Option<usize>,
    /// Maximum insertion rounds.
    pub max_iterations: usize,
    /// Free: insert knots where needed and make their positions unknowns of
    /// every refit, then remove the ones not needed. Uniform: keep evenly
    /// spaced knots and choose the count per section.
    pub knot_placement: KnotPlacement,
    /// Smallest spacing of free knots, as a fraction of a section: knots
    /// are neither inserted nor moved closer together.
    pub min_knot_gap: f64,
}

impl Default for ToleranceOptions {
    fn default() -> Self {
        ToleranceOptions {
            chain: ChainOptions { smoothing: 1e-7, ..ChainOptions::default() },
            initial_control_points: None,
            initial_parameters: None,
            max_control_points: None,
            remove_knots: true,
            max_nfev: None,
            max_iterations: 200,
            knot_placement: KnotPlacement::Free,
            min_knot_gap: MIN_KNOT_GAP,
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
/// The observer's `step` sees every residual evaluation of every refit, its
/// `refit` every finished refit with its maximum deviation; either stops the
/// search. The result carries `max_deviation`; `success` is false when the
/// tolerance could not be met.
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
    validate_min_knot_gap(options.min_knot_gap)?;
    if options.max_iterations < 1 {
        bail!("max_iterations must be at least 1.");
    }
    validate_sections(sections)?;
    let chain = &options.chain;
    let uniform = options.knot_placement == KnotPlacement::Uniform;
    let num_sections = sections.len();
    let need = minimum_control_points(&chain.orders(num_sections)?, degree);
    let mut knots: Vec<Vec<f64>> =
        need.iter().map(|&n| open_uniform_knots(degree, options.initial_control_points.map_or(n, |c| c.max(n)))).collect::<Result<_>>()?;
    let limits: Vec<usize> = sections.iter().map(|s| (s.len() - 1).min(options.max_control_points.unwrap_or(s.len()))).collect();
    // Name the section and the counts, so the user sees what to change.
    if let Some((s, (&n, &limit))) = need.iter().zip(&limits).enumerate().find(|(_, (n, l))| n > l) {
        let points = sections[s].len();
        if limit == points - 1 {
            bail!(
                "Section {} has {points} points, too few for its end conditions: they need {n} control points, so at least {} points.",
                s + 1,
                n + 1
            );
        }
        bail!("Section {} needs {n} control points for its end conditions, more than the limit of {limit}.", s + 1);
    }
    let mut u = start_parameters(options.initial_parameters.as_ref(), sections)?;
    let start_u = u.clone();
    let data = chain_data(sections);
    let mut theta: Option<Vec<f64>> = None;
    let min_span = if uniform { MIN_SPAN } else { MIN_SPAN.max(options.min_knot_gap) };
    let setup = Setup { sections, degree, options: chain, max_nfev: options.max_nfev, min_knot_gap: options.min_knot_gap };
    let mut steps = Steps { inner: observer, stopped: false };

    let finish = |mut result: FitResult, deviation: f64, success: bool, message: String| {
        result.max_deviation = Some(deviation);
        result.success = success;
        result.stopped = message == STOPPED;
        result.message = message;
        Ok(result)
    };
    let stopped = |result: FitResult, deviation: f64| finish(result, deviation, false, STOPPED.to_string());
    let done = |result: FitResult, deviation: f64| {
        finish(result, deviation, true, format!("Within tolerance {}.", crate::numfmt::general6(tolerance)))
    };

    let mut met = None;
    // The closest fit so far: its largest deviation, worst point and fit.
    // There is no rule that stops a search whose deviation stays level: it
    // stays level for many control points and then drops, so such a rule
    // stopped searches that would have met the tolerance.
    let mut closest: Option<(f64, usize, FitResult)> = None;
    for _ in 0..options.max_iterations {
        let mut fitted = setup.refit(knots.clone(), &u, theta.as_deref(), !uniform, &mut steps)?;
        let mut devs = chain_deviations(&data, &fitted.curves());
        let mut deviation = devs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // A warm start that is no closer than the closest fit so far may be
        // held by its parameters: fit uniform knots from the start
        // parameters too and keep the closer one.
        if uniform && !fitted.stopped && closest.as_ref().is_some_and(|c| deviation >= c.0) && u != start_u {
            let fresh = setup.refit(knots.clone(), &start_u, None, false, &mut steps)?;
            let fresh_devs = chain_deviations(&data, &fresh.curves());
            let fresh_deviation = fresh_devs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if fresh.stopped || fresh_deviation < deviation {
                (fitted, devs, deviation) = (fresh, fresh_devs, fresh_deviation);
            }
        }
        u = fitted.parameters.clone();
        theta = Some(fitted.theta.clone());
        knots = fitted.knots.clone();
        if fitted.stopped || steps.inner.refit(&fitted, deviation) {
            return stopped(fitted, deviation);
        }
        if deviation <= tolerance {
            met = Some((fitted, deviation));
            break;
        }
        if closest.as_ref().is_none_or(|c| deviation < c.0) {
            closest = Some((deviation, argmax(&devs), fitted.clone()));
        }
        let mut inserted = false;
        let mut narrow = false;
        for (s, (rows, local)) in groups(&u, num_sections).into_iter().enumerate() {
            let dev: Vec<f64> = rows.iter().map(|&i| devs[i]).collect();
            if dev.iter().copied().fold(0.0, f64::max) <= tolerance {
                continue;
            }
            let count = num_control_points(&knots[s], degree);
            let room = limits[s] as i64 - count as i64;
            // One control point a round: the fits then do not depend on the
            // tolerance, which only says where the search stops, so a looser
            // tolerance never ends with more control points than a tighter one.
            let mut new = new_knots(&knots[s], &local, &dev, tolerance, min_span);
            narrow |= room > 0 && new.is_empty() && too_narrow(&knots[s], &local, &dev, tolerance, min_span);
            new.truncate(room.clamp(0, 1) as usize);
            if uniform && room > 0 {
                knots[s] = open_uniform_knots(degree, count + 1)?;
                inserted = true;
            } else if !uniform && !new.is_empty() {
                knots[s] = with_knots(&knots[s], degree, &new);
                inserted = true;
            }
        }
        // Uniform knots are respaced every round, so only the parameters
        // carry over: take those of the closest fit so far, as a poor fit's
        // parameters hold every later fit near its poor shape.
        if uniform {
            u.clone_from(&closest.as_ref().expect("set above").2.parameters);
        }
        if !inserted {
            // Free knots crowd where the curve is hard to fit, down to the
            // knot spacing the user chose: say so, it is the setting to change.
            let message = if !uniform && narrow && options.min_knot_gap > MIN_SPAN {
                format!(
                    "Tolerance not reached: the worst knot spans are as narrow as the knot spacing {} allows.",
                    crate::numfmt::general6(options.min_knot_gap)
                )
            } else {
                "Tolerance not reached: no knot span can be refined further.".into()
            };
            return finish(fitted, deviation, false, message);
        }
    }
    let Some((mut result, mut deviation)) = met else {
        // The rounds ran out: the closest fit.
        let (deviation, _, result) = closest.expect("at least one round");
        return finish(result, deviation, false, "Tolerance not reached within max_iterations.".into());
    };

    if uniform {
        // Fewer control points per section: 1, 2, 4, ... fewer while the
        // tolerance holds, then halving the gap to the count that failed. A
        // search that grew one at a time needs a step or two; one that met
        // the tolerance at a high start count gets down in a few refits
        // instead of one per control point.
        for s in 0..num_sections {
            let (mut low, mut step, mut halving) = (need[s], 1, false);
            while num_control_points(&result.knots[s], degree) > low {
                let count = num_control_points(&result.knots[s], degree);
                let target = if halving { (low + count) / 2 } else { count.saturating_sub(step).max(low) };
                let mut trial_knots = result.knots.clone();
                trial_knots[s] = open_uniform_knots(degree, target)?;
                let trial = setup.refit(trial_knots, &result.parameters, Some(&result.theta), false, &mut steps)?;
                if trial.stopped {
                    return stopped(result, deviation);
                }
                let trial_deviation = max_deviation(&data, &trial);
                if trial_deviation > tolerance {
                    (low, halving) = (target + 1, true);
                    continue;
                }
                if !halving {
                    step *= 2;
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
    if options.remove_knots {
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
                    let trial = setup.refit(trial_knots, &result.parameters, Some(&result.theta), true, &mut steps)?;
                    if trial.stopped {
                        return stopped(result, deviation);
                    }
                    let trial_deviation = max_deviation(&data, &trial);
                    if trial_deviation <= tolerance {
                        result = trial;
                        deviation = trial_deviation;
                        removed = true;
                        if steps.inner.refit(&result, deviation) {
                            return stopped(result, deviation);
                        }
                        break;
                    }
                }
            }
        }
    }
    done(result, deviation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knots_go_to_the_worst_span() {
        let knots = open_uniform_knots(3, 5).unwrap();
        let u = [0.1, 0.2, 0.3, 0.6, 0.7, 0.9];
        let dev = [0.0, 0.0, 0.0, 0.5, 0.2, 0.0];
        assert_eq!(new_knots(&knots, &u, &dev, 0.1, MIN_SPAN), vec![0.625]);
    }

    #[test]
    fn free_knot_normal_equations_match_the_dense_jacobian() {
        let points = crate::airfoil::Naca4::new("2412", true).unwrap().points(21);
        let sections = vec![points[..21].to_vec(), points[20..].to_vec()];
        let options = ChainOptions {
            continuity: crate::Continuity::All(crate::Junction { order: 3, geometric: true, fixed: true }),
            ..Default::default()
        };
        let knots = vec![open_uniform_knots(5, 9).unwrap(); 2];
        let mut u = crate::chain::initial_chain_parameters(&sections);
        u[5] += 0.02;
        let gaps = KnotGaps::new(&sections, &knots, 5, &u, 0.05);
        let theta = [0.2, 0.1, -0.05];
        let mut xi = gaps.start();
        xi[1] += 0.1;
        let x = DVector::from_iterator(u.len() - 1 + 3 + xi.len(), t_to_eta(&u).into_iter().chain(theta).chain(xi.iter().copied()));
        let mut silent = crate::Silent;
        let mut model = FreeKnots {
            sections: &sections,
            options: &options,
            gaps,
            n_eta: u.len() - 1,
            n_theta: 3,
            cached: None,
            observer: &mut silent,
            nfev: 0,
            best: None,
        };
        let r = model.residual(&x).unwrap();
        let j = model.jacobian(&x).unwrap();
        let (a, g) = model.normal_equations(&x, &r).unwrap();
        let expected = j.transpose() * &j;
        assert!((&a - &expected).amax() <= 1e-10 * expected.amax(), "JᵀJ {}", (&a - &expected).amax());
        let expected = j.transpose() * &r;
        assert!((&g - &expected).amax() <= 1e-10 * expected.amax(), "Jᵀr {}", (&g - &expected).amax());
    }

    #[test]
    fn too_few_points_name_the_section_and_the_counts() {
        let arc = |n: usize, from: f64| -> Vec<Vec2> {
            (0..n).map(|i| Vec2::new((from + i as f64 / 10.0).cos(), (from + i as f64 / 10.0).sin())).collect()
        };
        let first = arc(30, 0.0);
        let second = arc(4, first.len() as f64 / 10.0 - 0.1);
        let sections = [first, second];
        let options = ToleranceOptions::default();
        let error = fit_to_tolerance(&sections, 1e-4, 5, &options, &mut crate::Silent).unwrap_err().to_string();
        assert!(error.starts_with("Section 2 has 4 points, too few for its end conditions: they need "), "{error}");
        let limited = ToleranceOptions { max_control_points: Some(5), ..Default::default() };
        let error = fit_to_tolerance(&sections[..1], 1e-4, 5, &limited, &mut crate::Silent).unwrap_err().to_string();
        assert!(error.starts_with("Section 1 needs "), "{error}");
        assert!(error.ends_with("more than the limit of 5."), "{error}");
    }

    #[test]
    fn a_search_that_misses_ends_with_its_closest_fit() {
        // A quarter circle with alternating noise of 1e-3: no smooth curve
        // gets within 1e-4 of every point. The search runs all its rounds
        // and ends on the closest fit.
        let points: Vec<Vec2> = (0..201)
            .map(|i| {
                let a = std::f64::consts::FRAC_PI_2 * i as f64 / 200.0;
                let r = 1.0 + if i % 2 == 0 { 1e-3 } else { -1e-3 };
                Vec2::new(r * a.cos(), r * a.sin())
            })
            .collect();
        let options = ToleranceOptions { knot_placement: KnotPlacement::Uniform, max_iterations: 15, ..Default::default() };
        let mut counts = Counts(Vec::new());
        let result = fit_to_tolerance(&[points], 1e-4, 3, &options, &mut counts).unwrap();
        assert!(!result.success);
        assert_eq!(result.message, "Tolerance not reached within max_iterations.");
        assert_eq!(counts.0.len(), 15, "every round ran: {:?}", counts.0);
        assert!(result.max_deviation.unwrap() > 9e-4);
    }

    /// The control-point counts of every refit.
    struct Counts(Vec<usize>);

    impl Observer for Counts {
        fn refit(&mut self, result: &FitResult, _: f64) -> bool {
            self.0.push(result.num_control_points().iter().sum());
            false
        }
    }

    #[test]
    fn the_search_adds_one_control_point_a_round_whatever_the_tolerance() {
        let points = crate::airfoil::Naca4::new("2412", true).unwrap().points(60);
        let options = ToleranceOptions { knot_placement: KnotPlacement::Uniform, ..Default::default() };
        let mut ends = Vec::new();
        for tolerance in [1e-3, 3e-4, 1e-4] {
            let mut counts = Counts(Vec::new());
            let result = fit_to_tolerance(std::slice::from_ref(&points), tolerance, 5, &options, &mut counts).unwrap();
            assert!(result.success, "{tolerance}: {}", result.message);
            // Up one at a time until the tolerance is met, then down.
            let top = counts.0.iter().position(|&c| c == *counts.0.iter().max().unwrap()).unwrap();
            assert!(counts.0[..=top].windows(2).all(|w| w[1] == w[0] + 1), "{tolerance}: {:?}", counts.0);
            ends.push((counts.0[..=top].to_vec(), result.num_control_points()[0]));
        }
        // The fits do not depend on the tolerance, only where they stop: a
        // tighter one goes on from where a looser one stopped.
        for pair in ends.windows(2) {
            let ((loose, loose_end), (tight, tight_end)) = (&pair[0], &pair[1]);
            assert_eq!(loose[..], tight[..loose.len()]);
            assert!(loose_end <= tight_end, "{loose_end} > {tight_end}");
        }
    }

    #[test]
    fn a_span_narrower_than_twice_the_spacing_is_too_narrow() {
        let knots = clamped_knots(3, &[0.5, 0.515]);
        let u = [0.1, 0.505, 0.51, 0.9];
        // The worst point is in [0.5, 0.515], 0.015 wide: too narrow at 0.01.
        let dev = [0.0, 0.5, 0.2, 0.0];
        assert!(too_narrow(&knots, &u, &dev, 0.1, 0.01));
        assert!(!too_narrow(&knots, &u, &dev, 0.1, 0.005));
        // Within the tolerance it does not matter how narrow the span is.
        assert!(!too_narrow(&knots, &u, &dev, 0.6, 0.01));
    }

    #[test]
    fn knot_gaps_reproduce_their_start_and_keep_the_floor() {
        let pts: Vec<Vec2> = (0..41).map(|i| Vec2::new(i as f64 / 40.0, (i as f64 / 6.0).sin())).collect();
        let knots = vec![clamped_knots(3, &[0.2, 0.5, 0.52, 0.9])];
        let u = crate::chain::initial_chain_parameters(std::slice::from_ref(&pts));
        let gaps = KnotGaps::new(std::slice::from_ref(&pts), &knots, 3, &u, 0.05);
        let start = gaps.knots(&gaps.start());
        for (a, b) in start[0].iter().zip(&knots[0]) {
            assert!((a - b).abs() < 1e-12);
        }
        let squeezed = gaps.knots(&[-30.0, -30.0, 30.0, -30.0, -30.0]);
        let inner = interior_knots(&squeezed[0], 3);
        let floor = gaps.floors[0].as_ref().unwrap();
        for (g, f) in self::gaps(inner).iter().zip(floor) {
            assert!(g + 1e-12 >= *f);
        }
    }
}
