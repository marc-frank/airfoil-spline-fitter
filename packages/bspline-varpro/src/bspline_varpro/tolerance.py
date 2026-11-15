"""Tolerance-driven fitting: few control points within a maximum orthogonal deviation.

Starting from the fewest control points the end conditions allow, knots are
inserted in every knot span that holds a point farther than the tolerance from
the chain, and the chain is refitted warm-started from the previous parameters.
A point counts toward the section its chain parameter currently lies on
(:mod:`~bspline_varpro.chain`).
Once the tolerance is met, knots whose removal keeps the fit within tolerance
are removed. With free knot placement the interior knot positions are unknowns
of every one of these fits (:func:`solve_free_knots`); with uniform placement
only the count per section is chosen.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from .basis import clamped_knots, interior_knots, open_uniform_knots
from .chain import chain_data, chain_deviations, groups
from .constraints import ContinuitySpec, End, minimum_control_points, parse_junctions
from .fitting import (
    STOPPED, ChainOptions, FitResult, _Stop, _validate_sections, as_sections,
    initial_chain_parameters, levenberg_marquardt, reduced_jacobian, report, result_from_problem,
    solve_problem,
    MIN_KNOT_GAP, validate_min_knot_gap,
)
from .reparam import eta_to_t, inverse_softplus, softplus, t_to_eta

ProgressCallback = Callable[[FitResult, float], Optional[bool]]


def _refit(sections, knots, degree, options: ChainOptions, params, theta, max_nfev,
           callback=None, free: bool = False, min_knot_gap: float = MIN_KNOT_GAP) -> FitResult:
    """One fit at the given knots: with ``free`` they are its starting knots and unknowns."""
    problem = options.problem(sections, knots, degree)
    if theta is None or len(theta) != problem.num_params:
        theta = problem.theta
    if free:
        return solve_free_knots(sections, knots, degree, options, params, theta,
                                max_nfev=max_nfev, callback=callback, min_knot_gap=min_knot_gap)
    return solve_problem(problem, params, theta, max_nfev=max_nfev, callback=callback)


def _spans(knots: np.ndarray, t: np.ndarray):
    breaks = np.unique(knots)
    index = np.clip(np.searchsorted(breaks, t, side="right") - 1, 0, len(breaks) - 2)
    return breaks, index


def _new_knots(knots, t, dev, tol, min_span) -> List[float]:
    """One knot per knot span that holds a point outside the tolerance, worst span first."""
    breaks, index = _spans(knots, t)
    new = []
    for i in range(len(breaks) - 1):
        inside = index == i
        if not np.any(dev[inside] > tol):
            continue
        a, b = breaks[i], breaks[i + 1]
        ts, ds = t[inside], dev[inside]
        if len(ts) < 2 or b - a < 2 * min_span:
            continue
        u = float(np.clip(ts[np.argmax(ds)], a + 0.25 * (b - a), b - 0.25 * (b - a)))
        if not (np.any(ts < u) and np.any(ts > u)):
            u = float(np.median(ts))
        if u - a >= min_span and b - u >= min_span:
            new.append((float(np.max(ds)), u))
    return [u for _, u in sorted(new, reverse=True)]


def _removal_order(knots, degree, t, dev, tol) -> List[float]:
    """Interior knots ordered by the largest deviation in their two adjacent spans."""
    breaks, index = _spans(knots, t)
    candidates = []
    for k in np.unique(interior_knots(knots, degree)):
        i = int(np.searchsorted(breaks, k))
        near = (index == i - 1) | (index == i)
        candidates.append((float(np.max(dev[near], initial=0.0)) / tol, float(k)))
    return [k for _, k in sorted(candidates)]


def _max_deviation(sections, result: FitResult) -> float:
    return float(np.max(chain_deviations(chain_data(sections), result.curves())))


class _KnotGaps:
    """Interior knots of a chain as unconstrained unknowns.

    Knots stay ordered with a minimum spacing through a normalized softplus
    gap parameterization. The minimum of each knot span is the section's
    closest spacing of the data parameters (or ``min_knot_gap``, if larger), or
    just under the span's starting length if that is shorter: knots packed
    closer than any two points would leave spans without data, where the curve
    can wobble unseen. Sections with a repeated interior knot keep their
    knots: the gap parameterization would spread the repeat and change the
    continuity there.

    ``count`` is the number of unknowns, ``start`` their values for the given
    knots and ``knots(xi)`` the knot vectors for values ``xi``.
    """

    def __init__(self, sections, knots, degree: int, parameters, min_knot_gap: float):
        self.degree = degree
        self.fixed = list(knots)
        inner_knots = [interior_knots(k, degree) for k in knots]
        self.counts = [len(inner) if np.all(np.diff(inner) > 0) else 0 for inner in inner_knots]
        # Smallest length of every knot span: the section's closest data spacing (a
        # span may move to where the data are densest), or just under its starting
        # length if that is shorter, so the solve starts from the given knots and
        # repeated solves cannot pack them ever closer
        self.floors = []
        for (_, local), inner, r in zip(groups(parameters, len(sections)), inner_knots, self.counts):
            if r == 0:
                self.floors.append(None)
                continue
            # The ends count as points: a junction point lies on the next section only
            ts = np.unique(np.concatenate([[0.0], local, [1.0]]))
            floor = np.minimum(max(float(np.min(np.diff(ts))), min_knot_gap),
                               0.999 * np.diff(np.concatenate([[0.0], inner, [1.0]])))
            # The floors must leave room to move: at most half the section
            self.floors.append(floor * min(1.0, 0.5 / np.sum(floor)))
        self.count = sum(r + 1 for r in self.counts if r > 0)
        self.start = (np.concatenate([self._xi(k, floor)
                                      for k, r, floor in zip(knots, self.counts, self.floors) if r > 0])
                      if self.count else np.zeros(0))

    def _xi(self, knots, floor):
        gaps = np.diff(np.concatenate([[0.0], interior_knots(knots, self.degree), [1.0]]))
        return inverse_softplus(np.maximum(gaps - floor, 1e-12))

    def knots(self, xi):
        knots, offset = [], 0
        for s, r in enumerate(self.counts):
            if r == 0:
                knots.append(self.fixed[s])
                continue
            # The floors hold after normalizing: only the room above them is shared out
            free = softplus(xi[offset:offset + r + 1])
            floor = self.floors[s]
            gaps = floor + (1.0 - np.sum(floor)) * free / np.sum(free)
            offset += r + 1
            knots.append(clamped_knots(self.degree, np.cumsum(gaps)[:-1]))
        return knots


def solve_free_knots(
    sections: Sequence[np.ndarray],
    knots: Sequence[np.ndarray],
    degree: int,
    options: ChainOptions,
    params: np.ndarray,
    theta0: np.ndarray,
    *,
    max_nfev: Optional[int] = None,
    tolerance: float = 1e-8,
    exact_jacobian: bool = True,
    verbose: int = 0,
    callback=None,
    min_knot_gap: float = MIN_KNOT_GAP,
) -> FitResult:
    """Levenberg-Marquardt with the interior knots among the unknowns.

    The same solve as :func:`~bspline_varpro.fitting.solve_problem`, from the
    same start, with the knot positions (at fixed counts, see
    :class:`_KnotGaps`) optimized together with the data parameters and the
    junction shape parameters: one loop from the first step to convergence.
    ``knots`` are the starting knots. Knot columns of the Jacobian are
    forward differences; the other columns are exact (or forward differences
    too, with ``exact_jacobian`` False). ``callback`` is called with the
    moving knots on every residual evaluation of the loop (not on the extra
    ones of the differences); if it returns True the lowest-cost state seen
    is returned with the message ``STOPPED``. A chain without a movable knot
    is solved by ``solve_problem``.
    """
    theta0 = np.asarray(theta0, dtype=np.float64)
    gaps = _KnotGaps(sections, knots, degree, params, min_knot_gap)
    if gaps.count == 0:
        return solve_problem(options.problem(sections, list(knots), degree), params, theta0,
                             max_nfev=max_nfev, tolerance=tolerance, exact_jacobian=exact_jacobian,
                             verbose=verbose, callback=callback)
    span = len(sections)
    n_eta = len(chain_data(sections)) - 1
    n_theta = len(theta0)
    nfev = 0
    best_x: Optional[np.ndarray] = None
    best_cost = np.inf
    cache = {}

    def problem_for(xi):
        key = xi.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = options.problem(sections, gaps.knots(xi), degree)
        return cache[key]

    def split(x):
        return x[:n_eta], x[n_eta:n_eta + n_theta], x[n_eta + n_theta:]

    def residual(x):
        nonlocal nfev, best_x, best_cost
        eta, theta, xi = split(x)
        problem = problem_for(xi)
        r = problem.residual(eta_to_t(eta, span), theta)
        nfev += 1
        cost = float(r @ r)
        if best_x is None or cost < best_cost:
            best_x, best_cost = x.copy(), cost
        if report(callback, problem.split_control_points(problem.last_control_points),
                  problem.last_rms_error, problem.knots):
            raise _Stop
        return r

    def jacobian(x):
        eta, theta, xi = split(x)
        t_flat = eta_to_t(eta, span)
        base = problem_for(xi).residual(t_flat, theta)
        J = np.empty((len(base), len(x)))
        J[:, :n_eta + n_theta] = reduced_jacobian(problem_for(xi), eta, theta)
        for i in range(len(xi)):
            h = 1e-7 * (1.0 + abs(xi[i]))
            shifted = xi.copy()
            shifted[i] += h
            other = options.problem(sections, gaps.knots(shifted), degree)
            J[:, n_eta + n_theta + i] = (other.residual(t_flat, theta) - base) / h
        return J

    x0 = np.concatenate([t_to_eta(params), theta0, gaps.start])
    try:
        solution = levenberg_marquardt(residual, jacobian if exact_jacobian else "2-point", x0,
                                       max_nfev=max_nfev, tolerance=tolerance, verbose=verbose)
        x_final = solution.x
        success, message = bool(solution.success), str(solution.message)
    except _Stop:
        x_final = best_x
        success, message = False, STOPPED
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"VarPro fit failed: {exc}") from exc
    eta, theta, xi = split(x_final)
    problem = options.problem(sections, gaps.knots(xi), degree)
    return result_from_problem(problem, eta_to_t(eta, span), theta, nfev, success, message)


def fit_to_tolerance(
    points: Union[np.ndarray, Sequence[np.ndarray]],
    tolerance: float,
    degree: int = 5,
    *,
    continuity: Union[ContinuitySpec, Sequence[ContinuitySpec]] = "G2",
    fix_junctions: bool = True,
    fix_ends: bool = True,
    start_tangent: Optional[Sequence[float]] = None,
    end_tangent: Optional[Sequence[float]] = None,
    smoothing: float = 1e-7,
    regularization: float = 1e-12,
    initial_knots: Optional[Sequence[np.ndarray]] = None,
    initial_control_points: Union[int, Sequence[int], None] = None,
    initial_parameters: Optional[Sequence[np.ndarray]] = None,
    max_control_points: Optional[int] = None,
    remove_knots: bool = True,
    max_nfev: Optional[int] = None,
    max_iterations: int = 40,
    callback: Optional[ProgressCallback] = None,
    knot_placement: str = "free",
    should_stop: Optional[Callable[[], bool]] = None,
    step_callback=None,
    min_knot_gap: float = MIN_KNOT_GAP,
) -> FitResult:
    """Fit a chain with few control points so every point lies within ``tolerance``.

    Args:
        points: One (n, dim) array or a sequence of sections (see :func:`fit`).
        tolerance: Maximum allowed orthogonal distance of any point to the
            chain, in data units.
        degree: Spline degree (5 recommended for CAD).
        continuity, fix_junctions, fix_ends, start_tangent, end_tangent,
            smoothing, regularization: As for :func:`fit`.
        initial_knots: Starting knot vectors per section (default: the fewest
            control points the end conditions allow).
        initial_control_points: Instead of ``initial_knots``, start from this
            many control points per section (an int or one per section) with
            uniform knots, raised to the fewest the end conditions allow.
        initial_parameters: Starting data parameters (as for :func:`fit`).
        max_control_points: Upper limit per section.
        remove_knots: Try to remove knots once the tolerance is met.
        max_nfev: Residual evaluations per refit; None (the default) runs each
            refit until converged.
        max_iterations: Maximum insertion rounds.
        callback: ``callback(result, max_deviation)`` after every refit; return
            True to stop early (the last result is returned).
        knot_placement: ``"free"`` inserts knots in the spans that miss the
            tolerance and removes the ones that are not needed; the knot
            positions are unknowns of every refit, like the data parameters.
            ``"uniform"`` keeps uniformly spaced knots and only chooses the
            count per section (``remove_knots`` does not apply).
        should_stop: Checked at every residual evaluation, also inside
            refits; returning True stops the search with the best
            result so far (``callback`` is only reached between refits).
        step_callback: ``step_callback(control_points, rms_error)`` on every
            residual evaluation of every refit, as the
            callback of :func:`~bspline_varpro.fitting.fit` (with a ``knots``
            keyword it also gets the knot vectors); return True to stop like
            ``should_stop``.

        min_knot_gap: Smallest spacing of free knots, as a fraction of a
            section's parameter range (0 to 0.5): knots are neither
            inserted nor moved closer together. Without effect on
            uniform knots.

    Returns:
        :class:`FitResult` with ``max_deviation`` and a ``history`` of stages.
        ``success`` is False if the tolerance could not be met.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be positive.")
    if knot_placement not in ("free", "uniform"):
        raise ValueError("knot_placement must be 'free' or 'uniform'.")
    validate_min_knot_gap(min_knot_gap)
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")
    uniform = knot_placement == "uniform"
    sections = as_sections(points)
    _validate_sections(sections)
    options = ChainOptions(continuity, fix_junctions, fix_ends, start_tangent, end_tangent,
                           smoothing, regularization)
    junctions = parse_junctions(continuity, len(sections) - 1, fix_junctions)
    start = End(fix_ends, None if start_tangent is None else tuple(start_tangent))
    end = End(fix_ends, None if end_tangent is None else tuple(end_tangent))
    need = minimum_control_points(junctions, degree, start, end)
    if initial_knots is not None:
        knots = [np.asarray(k, dtype=np.float64) for k in initial_knots]
    elif initial_control_points is not None:
        start_counts = ([int(initial_control_points)] * len(sections) if np.ndim(initial_control_points) == 0
                        else [int(n) for n in initial_control_points])
        if len(start_counts) != len(sections):
            raise ValueError("Need one initial control-point count per section.")
        knots = [open_uniform_knots(degree, max(n, m)) for n, m in zip(start_counts, need)]
    else:
        knots = [open_uniform_knots(degree, n) for n in need]
    limits = [min(len(sec) - 1, max_control_points or len(sec)) for sec in sections]
    if any(n > limit for n, limit in zip(need, limits)):
        raise ValueError("Sections have too few points for the requested end conditions.")
    params = initial_chain_parameters(initial_parameters, sections)
    data = chain_data(sections)

    history: List[dict] = []
    theta = None
    # Free knots are not inserted closer than they may move
    min_span = 1e-4 if uniform else max(1e-4, min_knot_gap)

    def record(stage, result, deviation):
        history.append({"stage": stage, "control_points": result.num_control_points,
                        "max_deviation": deviation, "rms_error": result.rms_error})
        return callback is not None and bool(callback(result, deviation))

    # Per-evaluation progress and stop for the refits
    stopped = []
    step = None
    if should_stop is not None or step_callback is not None:
        def step(control_points, rms_error, knots):
            if ((should_stop is not None and should_stop())
                    or report(step_callback, control_points, rms_error, knots)):
                stopped.append(True)
                return True
            return False

    def halted(trial):
        return trial.message == STOPPED

    def finish(result, deviation, success, message):
        result.max_deviation = deviation
        result.history = history
        result.options = options
        result.success = success
        result.message = message
        return result

    result = None
    for _ in range(max_iterations):
        result = _refit(sections, knots, degree, options, params, theta, max_nfev, step, not uniform,
                        min_knot_gap)
        params, theta = result.parameters, result.theta
        # Free knots have moved: the next round starts from where they are
        knots = list(result.knots)
        devs = chain_deviations(data, result.curves())
        deviation = float(np.max(devs))
        if halted(result) or record("insert", result, deviation):
            return finish(result, deviation, False, STOPPED)
        if deviation <= tolerance:
            break
        inserted = False
        for s, (rows, local) in enumerate(groups(params, len(sections))):
            if np.max(devs[rows], initial=0.0) <= tolerance:
                continue
            count = len(knots[s]) - degree - 1
            room = limits[s] - count
            new = _new_knots(knots[s], local, devs[rows], tolerance, min_span)[:max(room, 0)]
            if uniform and room > 0:
                # Grow by as many knots as free insertion would add, spaced uniformly.
                knots[s] = open_uniform_knots(degree, count + max(len(new), 1))
                inserted = True
            elif new:
                knots[s] = clamped_knots(degree, np.concatenate([interior_knots(knots[s], degree), new]))
                inserted = True
        if not inserted:
            return finish(result, deviation, False,
                          "Tolerance not reached: no knot span can be refined further.")
    else:
        return finish(result, deviation, False, "Tolerance not reached within max_iterations.")

    if uniform:
        # Growth may overshoot: drop one uniform knot per section while the tolerance holds.
        for s in range(len(sections)):
            while len(result.knots[s]) - degree - 1 > need[s]:
                trial_knots = list(result.knots)
                trial_knots[s] = open_uniform_knots(degree, len(result.knots[s]) - degree - 2)
                trial = _refit(sections, trial_knots, degree, options, result.parameters, result.theta,
                               max_nfev, step)
                if halted(trial):
                    return finish(result, deviation, False, STOPPED)
                trial_dev = _max_deviation(sections, trial)
                if trial_dev > tolerance:
                    break
                result, deviation = trial, trial_dev
                if record("remove", result, deviation):
                    return finish(result, deviation, False, STOPPED)
        return finish(result, deviation, True, f"Within tolerance {tolerance:g}.")

    if remove_knots:
        for s in range(len(sections)):
            # Try every knot, least useful first; after a removal start over, since the
            # refitted curve can make knots removable that failed before.
            removed = True
            while removed and len(result.knots[s]) - degree - 1 > need[s]:
                removed = False
                devs = chain_deviations(data, result.curves())
                rows, local = groups(result.parameters, len(sections))[s]
                for k in _removal_order(result.knots[s], degree, local, devs[rows], tolerance):
                    trial_knots = list(result.knots)
                    inner = interior_knots(result.knots[s], degree)
                    trial_knots[s] = clamped_knots(degree, np.delete(inner, np.argmin(np.abs(inner - k))))
                    trial = _refit(sections, trial_knots, degree, options, result.parameters, result.theta,
                                   max_nfev, step, True, min_knot_gap)
                    if halted(trial):
                        return finish(result, deviation, False, STOPPED)
                    trial_dev = _max_deviation(sections, trial)
                    if trial_dev <= tolerance:
                        result, deviation, removed = trial, trial_dev, True
                        if record("remove", result, deviation):
                            return finish(result, deviation, False, STOPPED)
                        break

    return finish(result, deviation, True, f"Within tolerance {tolerance:g}.")


def grow_to_counts(sections, counts, degree, options: ChainOptions, params, theta, max_nfev,
                   callback=None, knot_placement: str = "free",
                   min_knot_gap: float = MIN_KNOT_GAP) -> FitResult:
    """Grow from the fewest control points to the given counts, refitting each step.

    Each round adds one knot to every span whose largest deviation is at least
    half the section's largest, worst span first, until each section has its
    count; when none of those spans can be split, any span holding two points
    is. With ``"free"`` placement the knots go into those spans and are
    unknowns of every refit, not closer than ``min_knot_gap`` (or than half
    the uniform spacing of the largest count, if that is less); with
    ``"uniform"`` the section is respaced evenly with the new count.
    """
    junctions = parse_junctions(options.continuity, len(sections) - 1, options.fix_junctions)
    start = End(options.fix_ends, None if options.start_tangent is None else tuple(options.start_tangent))
    end = End(options.fix_ends, None if options.end_tangent is None else tuple(options.end_tangent))
    need = minimum_control_points(junctions, degree, start, end)
    knots = [open_uniform_knots(degree, min(n, c)) for n, c in zip(need, counts)]
    # Free knots are not inserted closer than they may move, unless the counts
    # asked for need it: then half their uniform spacing is the limit
    min_span = 1e-4
    if knot_placement == "free":
        min_span = max(min_span, min(min_knot_gap, 0.5 / max(max(counts) - degree, 1)))
    data = chain_data(sections)
    while True:
        result = _refit(sections, knots, degree, options, params, theta, max_nfev, callback,
                        knot_placement == "free", min_knot_gap)
        if result.message == STOPPED:
            return result
        params, theta = result.parameters, result.theta
        knots = list(result.knots)
        missing = [c - (len(k) - degree - 1) for c, k in zip(counts, knots)]
        if max(missing) <= 0:
            break
        devs = chain_deviations(data, result.curves())
        inserted = False
        for s, (room, (rows, local)) in enumerate(zip(missing, groups(params, len(sections)))):
            if room <= 0:
                continue
            dev = devs[rows]
            new = _new_knots(knots[s], local, dev, 0.5 * float(np.max(dev, initial=0.0)), min_span)[:room]
            if not new:
                # The worst points can sit alone in their spans, which cannot be
                # split; any span holding two points can, worst first
                new = _new_knots(knots[s], local, dev, -1.0, min_span)[:room]
            if knot_placement == "uniform":
                knots[s] = open_uniform_knots(degree, len(knots[s]) - degree - 1 + max(len(new), 1))
                inserted = True
            elif new:
                knots[s] = clamped_knots(degree, np.concatenate([interior_knots(knots[s], degree), new]))
                inserted = True
        if not inserted:
            break
    return result
