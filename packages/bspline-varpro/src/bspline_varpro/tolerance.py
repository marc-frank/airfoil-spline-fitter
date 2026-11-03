"""Tolerance-driven fitting: few control points within a maximum orthogonal deviation.

Starting from the fewest control points the end conditions allow, knots are
inserted in every knot span that holds a point farther than the tolerance from
the chain, and the chain is refitted warm-started from the previous parameters.
A point counts toward the section its chain parameter currently lies on
(:mod:`~bspline_varpro.chain`).
Once the tolerance is met, interior knot positions are optimized ("free-knot
polish") and knots whose removal keeps the fit within tolerance are removed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from scipy.linalg import block_diag

from .basis import clamped_knots, interior_knots, open_uniform_knots
from .chain import chain_data, chain_deviations, groups, project_to_chain, section_ranges
from .constraints import ContinuitySpec, End, minimum_control_points, parse_junctions
from .fitting import (
    STOPPED, ChainOptions, FitResult, _Stop, _validate_sections, as_sections,
    initial_chain_parameters, levenberg_marquardt, reduced_jacobian, report, result_from_problem,
    solve_problem, solve_with_continuation,
)
from .reparam import eta_to_t, inverse_softplus, sigmoid, softplus, t_to_eta

ProgressCallback = Callable[[FitResult, float], Optional[bool]]

# Central-difference step for the knot derivatives of the problem matrices, as a
# fraction of the shorter neighboring knot span
_KNOT_STEP = 1e-3


def _refit(sections, knots, degree, options: ChainOptions, params, theta, max_nfev,
           max_points: Optional[int] = None, callback=None) -> FitResult:
    if max_points is not None and any(len(s) > max_points for s in sections):
        return fit_subsampled(sections, knots, degree, options, params, theta, max_nfev, max_points,
                              callback)
    problem = options.problem(sections, knots, degree)
    if theta is None or len(theta) != problem.num_params:
        return solve_with_continuation(problem, options, params, max_nfev=max_nfev, callback=callback)
    return solve_problem(problem, params, theta, max_nfev=max_nfev, callback=callback)


def subsample_indices(n: int, max_points: int) -> np.ndarray:
    """Evenly spaced indices including both ends."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_points)).astype(int))


def fit_subsampled(sections, knots, degree, options: ChainOptions, params, theta, max_nfev,
                   max_points: int, callback=None) -> FitResult:
    """Variable projection on a subsample, then joint refinement on all points.

    Variable projection costs O(n^3) per iteration in the number of points,
    the joint solver O(n k^2) in the number of control coordinates k.
    ``callback`` (as for :func:`~bspline_varpro.fitting.fit`) sees the
    subsample fit and the refinement; if it stops the subsample fit, the
    refinement is skipped.
    """
    from .refinement import refine

    # Each section's subsample keeps its end points, so shared junction points
    # stay shared and the subsets chain the same way
    data = chain_data(sections)
    indices = [first + subsample_indices(last - first + 1, max_points)
               for first, last in section_ranges(sections)]
    subsets = [data[i] for i in indices]
    chosen = np.unique(np.concatenate(indices))
    coarse = _refit(subsets, knots, degree, options, np.asarray(params)[chosen], theta, max_nfev,
                    callback=callback)
    coarse.parameters = project_to_chain(data, coarse.curves(), chosen, coarse.parameters)
    coarse.options = options
    if coarse.message == STOPPED:
        return coarse
    refined = refine(sections, coarse, options, callback=callback)
    refined.nfev += coarse.nfev
    return refined


# Knot choices must not depend on rounding, or on the direction of the data: the
# mirror images in a symmetric fit differ by rounding only. Deviations equal to
# this relative difference count as tied, and tied candidates are taken together.
_TIE = 1e-6
# Parameter distance within which a point lies on a knot, and so in both spans
_ON_KNOT = 1e-9


def _spans(knots: np.ndarray, t: np.ndarray):
    """Breaks and, per knot span, a mask of the points in it (a point on a knot in both)."""
    breaks = np.unique(knots)
    return breaks, [(t >= a - _ON_KNOT) & (t <= b + _ON_KNOT) for a, b in zip(breaks[:-1], breaks[1:])]


def _tied_groups(scored, descending: bool) -> List[List[float]]:
    """Values of ``(score, value)`` pairs, grouped where the scores tie, best group first."""
    groups: List[tuple] = []
    for score, value in sorted(scored, key=lambda pair: -pair[0] if descending else pair[0]):
        if groups and abs(score - groups[-1][0]) <= _TIE * max(abs(score), abs(groups[-1][0])):
            groups[-1][1].append(value)
        else:
            groups.append((score, [value]))
    return [values for _, values in groups]


def _take(groups: List[List[float]], room: int, force: bool = True) -> List[float]:
    """Whole groups, best first, skipping those that no longer fit in ``room``.

    If none fits and ``force`` is set, one value of the best group: a tie the
    count cannot keep.
    """
    taken: List[float] = []
    for group in groups:
        if len(taken) + len(group) <= room:
            taken += group
    if force and not taken and room > 0 and groups:
        taken = groups[0][:1]
    return taken


def _new_knots(knots, t, dev, tol, min_span) -> List[List[float]]:
    """One knot per knot span that holds a point outside the tolerance, grouped by
    tied worst deviation, worst group first.

    The knot goes to the worst point of the span, or the mean of the points tied
    for worst."""
    breaks, members = _spans(knots, t)
    new = []
    for i, inside in enumerate(members):
        if not np.any(dev[inside] > tol):
            continue
        a, b = breaks[i], breaks[i + 1]
        ts, ds = t[inside], dev[inside]
        if len(ts) < 2 or b - a < 2 * min_span:
            continue
        worst = float(np.max(ds))
        u = float(np.clip(np.mean(ts[ds >= worst - _TIE * abs(worst)]),
                          a + 0.25 * (b - a), b - 0.25 * (b - a)))
        if not (np.any(ts < u) and np.any(ts > u)):
            u = float(np.median(ts))
        if u - a >= min_span and b - u >= min_span:
            new.append((worst, u))
    return _tied_groups(new, descending=True)


def _removal_order(knots, degree, t, dev, tol) -> List[List[float]]:
    """Interior knots ordered by the largest deviation in their two adjacent spans,
    grouped where those tie."""
    breaks, members = _spans(knots, t)
    candidates = []
    for k in np.unique(interior_knots(knots, degree)):
        i = int(np.searchsorted(breaks, k))
        near = members[i - 1] | members[i]
        candidates.append((float(np.max(dev[near], initial=0.0)) / tol, float(k)))
    return _tied_groups(candidates, descending=False)


def _max_deviation(sections, result: FitResult) -> float:
    return float(np.max(chain_deviations(chain_data(sections), result.curves())))


def polish_knots(
    points: Union[np.ndarray, Sequence[np.ndarray]],
    result: FitResult,
    options: Optional[ChainOptions] = None,
    max_nfev: Optional[int] = 60,
    min_gap: float = 1e-3,
    should_stop: Optional[Callable[[], bool]] = None,
    callback=None,
) -> FitResult:
    """Optimize interior knot positions (at fixed counts) together with the fit.

    Knots stay ordered with a minimum spacing through a normalized softplus gap
    parameterization. The minimum of each knot span is the section's closest
    spacing of the data parameters (or ``min_gap``, if larger), or just under
    the span's starting length if that is shorter: knots packed closer than any
    two points would leave spans without data, where the curve can wobble
    unseen, since the tolerance is only checked at the points.
    The knot columns of the Jacobian differentiate the linear solve exactly,
    with central differences of the basis, penalty and constraint matrices
    only; the data-parameter and shape columns are exact. ``callback`` (as for
    :func:`~bspline_varpro.fitting.fit`, with the moving knots) is called on
    every residual evaluation. If it or ``should_stop()`` returns True, the
    unpolished ``result`` is returned with the message ``STOPPED``.

    ``max_nfev`` stays capped by default (None polishes until converged): the
    error is so flat in the knot positions that a converged polish takes
    thousands of evaluations, and it drifts off a symmetric fit's mirror
    symmetry for well under a percent of the error.
    """
    sections = as_sections(points)
    options = options or result.options or ChainOptions(continuity=result.continuity or "G1")
    degree = result.degree
    # Sections with a repeated interior knot keep their knots: the gap
    # parameterization would spread the repeat and change the continuity there
    inner_knots = [interior_knots(k, degree) for k in result.knots]
    counts = [len(inner) if np.all(np.diff(inner) > 0) else 0 for inner in inner_knots]
    if sum(counts) == 0:
        return result
    span = len(sections)
    # Smallest length of every knot span: the section's closest data spacing (a
    # span may move to where the data are densest), or just under its starting
    # length if that is shorter, so the polish starts from the given knots and
    # repeated polishes cannot pack them ever closer
    floors = []
    for (_, local), inner, r in zip(groups(result.parameters, span), inner_knots, counts):
        if r == 0:
            floors.append(None)
            continue
        # The ends count as points: a junction point lies on the next section only
        ts = np.unique(np.concatenate([[0.0], local, [1.0]]))
        floor = np.minimum(max(float(np.min(np.diff(ts))), min_gap),
                           0.999 * np.diff(np.concatenate([[0.0], inner, [1.0]])))
        # The floors must leave room to move: at most half the section
        floors.append(floor * min(1.0, 0.5 / np.sum(floor)))
    n_eta = len(chain_data(sections)) - 1
    n_theta = len(result.theta)

    def to_xi(knots, floor):
        gaps = np.diff(np.concatenate([[0.0], interior_knots(knots, degree), [1.0]]))
        return inverse_softplus(np.maximum(gaps - floor, 1e-12))

    def to_knots(xi):
        knots, offset = [], 0
        for s, r in enumerate(counts):
            if r == 0:
                knots.append(result.knots[s])
                continue
            gaps = softplus(xi[offset:offset + r + 1]) + floors[s]
            offset += r + 1
            knots.append(clamped_knots(degree, np.cumsum(gaps)[:-1] / np.sum(gaps)))
        return knots

    cache = {}

    def problem_for(xi):
        key = xi.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = options.problem(sections, to_knots(xi), degree)
        return cache[key]

    def split(x):
        return x[:n_eta], x[n_eta:n_eta + n_theta], x[n_eta + n_theta:]

    def residual_at(x):
        if should_stop is not None and should_stop():
            raise _Stop
        eta, theta, xi = split(x)
        problem = problem_for(xi)
        r = problem.residual(eta_to_t(eta, span), theta)
        if report(callback, problem.split_control_points(problem.last_control_points),
                  problem.last_rms_error, problem.knots):
            raise _Stop
        return r

    def knots_by_xi(xi):
        """Derivative of the interior knots of the polished sections w.r.t. xi."""
        blocks, offset = [], 0
        for r, floor in zip(counts, floors):
            if r == 0:
                continue
            x = xi[offset:offset + r + 1]
            offset += r + 1
            gaps = softplus(x) + floor
            total = np.sum(gaps)
            inner = np.cumsum(gaps)[:-1] / total
            blocks.append((np.tri(r, r + 1) - inner[:, None]) * (sigmoid(x) / total))
        return block_diag(*blocks)

    def jacobian(x):
        eta, theta, xi = split(x)
        problem = problem_for(xi)
        t_flat = eta_to_t(eta, span)
        # Central differences in the knot positions, with steps a fixed fraction of
        # the neighboring spans: differences in xi would lose precision where the
        # softplus flattens
        changes = []
        for s, r in enumerate(counts):
            inner = interior_knots(problem.knots[s], degree)
            edges = np.concatenate([[0.0], inner, [1.0]])
            for j in range(r):
                h = _KNOT_STEP * min(edges[j + 1] - edges[j], edges[j + 2] - edges[j + 1])
                shifted = []
                for step in (h, -h):
                    knots = list(problem.knots)
                    knots[s] = clamped_knots(degree, inner + step * (np.arange(r) == j))
                    shifted.append(options.problem(sections, knots, degree).matrices(t_flat, theta))
                changes.append(tuple((a - b) / (2.0 * h) for a, b in zip(*shifted)))
        return np.hstack([reduced_jacobian(problem, eta, theta),
                          problem.matrix_jacobian(t_flat, theta, changes) @ knots_by_xi(xi)])

    xi0 = np.concatenate([to_xi(k, floor) for k, r, floor in zip(result.knots, counts, floors) if r > 0])
    x0 = np.concatenate([t_to_eta(result.parameters), result.theta, xi0])
    try:
        solution = levenberg_marquardt(residual_at, jacobian, x0, max_nfev=max_nfev)
    except _Stop:
        return replace(result, success=False, message=STOPPED)
    except (ValueError, np.linalg.LinAlgError):
        return result
    eta, theta, xi = split(solution.x)
    problem = options.problem(sections, to_knots(xi), degree)
    return result_from_problem(problem, eta_to_t(eta, span), theta, solution.nfev,
                               bool(solution.success), str(solution.message))


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
    optimize_knots: bool = True,
    remove_knots: bool = True,
    max_points_per_section: Optional[int] = 200,
    max_nfev: Optional[int] = None,
    max_iterations: int = 40,
    callback: Optional[ProgressCallback] = None,
    knot_placement: str = "free",
    should_stop: Optional[Callable[[], bool]] = None,
    step_callback=None,
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
        optimize_knots: Polish interior knot positions once the tolerance is met.
        remove_knots: Try to remove knots once the tolerance is met.
        max_points_per_section: Larger sections are fitted by variable
            projection on an evenly spaced subsample, then refined jointly on
            all points (knot polishing is skipped). None uses all points.
        max_nfev: Residual evaluations per refit; None (the default) runs each
            refit until converged.
        max_iterations: Maximum insertion rounds.
        callback: ``callback(result, max_deviation)`` after every refit; return
            True to stop early (the last result is returned).
        knot_placement: ``"free"`` inserts knots in the spans that miss the
            tolerance, then moves them to where they reduce the error most and
            removes the ones that are not needed. ``"uniform"`` keeps uniformly
            spaced knots and only chooses the count per section
            (``optimize_knots`` and ``remove_knots`` do not apply).
        should_stop: Checked at every residual evaluation, also inside refits
            and knot polishing; returning True stops the search with the best
            result so far (``callback`` is only reached between refits).
        step_callback: ``step_callback(control_points, rms_error)`` on every
            residual evaluation of every refit and knot polish, as the
            callback of :func:`~bspline_varpro.fitting.fit` (with a ``knots``
            keyword it also gets the knot vectors); return True to stop like
            ``should_stop``.

    Returns:
        :class:`FitResult` with ``max_deviation`` and a ``history`` of stages.
        ``success`` is False if the tolerance could not be met.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be positive.")
    if knot_placement not in ("free", "uniform"):
        raise ValueError("knot_placement must be 'free' or 'uniform'.")
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
    min_span = 1e-4

    def record(stage, result, deviation):
        history.append({"stage": stage, "control_points": result.num_control_points,
                        "max_deviation": deviation, "rms_error": result.rms_error})
        return callback is not None and bool(callback(result, deviation))

    # Per-evaluation progress and stop for the refits and knot polishing
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

    def stop_requested():
        """True once a step asked to stop (a rejected polish hides that)."""
        return bool(stopped) or (should_stop is not None and should_stop())

    def finish(result, deviation, success, message):
        result.max_deviation = deviation
        result.history = history
        result.options = options
        result.success = success
        result.message = message
        return result

    result = None
    for _ in range(max_iterations):
        result = _refit(sections, knots, degree, options, params, theta, max_nfev,
                        max_points_per_section, step)
        params, theta = result.parameters, result.theta
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
            new = _take(_new_knots(knots[s], local, devs[rows], tolerance, min_span), room)
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
                               max_nfev, max_points_per_section, step)
                if halted(trial):
                    return finish(result, deviation, False, STOPPED)
                trial_dev = _max_deviation(sections, trial)
                if trial_dev > tolerance:
                    break
                result, deviation = trial, trial_dev
                if record("remove", result, deviation):
                    return finish(result, deviation, False, STOPPED)
        return finish(result, deviation, True, f"Within tolerance {tolerance:g}.")

    dense = max_points_per_section is not None and any(len(s) > max_points_per_section for s in sections)
    polish = optimize_knots and not dense

    def polished(result, deviation):
        """Knot-polished result if it stays within tolerance without raising the RMS error."""
        candidate = polish_knots(sections, result, options, callback=step)
        if halted(candidate):
            return result, deviation, False
        candidate_dev = _max_deviation(sections, candidate)
        if candidate_dev <= tolerance and candidate.rms_error <= result.rms_error:
            return candidate, candidate_dev, True
        return result, deviation, False

    # Polish before removing: better-placed knots make more of them redundant.
    if polish:
        result, deviation, improved = polished(result, deviation)
        if (improved and record("polish", result, deviation)) or stop_requested():
            return finish(result, deviation, False, STOPPED)

    if remove_knots:
        removed_any = False
        for s in range(len(sections)):
            # Try every knot, least useful first (tied knots together); after a removal
            # start over, since the refitted curve can make knots removable that failed before.
            removed = True
            while removed and len(result.knots[s]) - degree - 1 > need[s]:
                removed = False
                devs = chain_deviations(data, result.curves())
                rows, local = groups(result.parameters, len(sections))[s]
                for tied in _removal_order(result.knots[s], degree, local, devs[rows], tolerance):
                    if len(result.knots[s]) - degree - 1 - len(tied) < need[s]:
                        continue
                    trial_knots = list(result.knots)
                    inner = interior_knots(result.knots[s], degree)
                    drop = [int(np.argmin(np.abs(inner - k))) for k in tied]
                    trial_knots[s] = clamped_knots(degree, np.delete(inner, drop))
                    trial = _refit(sections, trial_knots, degree, options, result.parameters, result.theta,
                                   max_nfev, max_points_per_section, step)
                    if halted(trial):
                        return finish(result, deviation, False, STOPPED)
                    trial_dev = _max_deviation(sections, trial)
                    if trial_dev > tolerance and polish:
                        trial = polish_knots(sections, trial, options, callback=step)
                        if halted(trial):
                            return finish(result, deviation, False, STOPPED)
                        trial_dev = _max_deviation(sections, trial)
                    if trial_dev <= tolerance:
                        result, deviation, removed, removed_any = trial, trial_dev, True, True
                        if record("remove", result, deviation):
                            return finish(result, deviation, False, STOPPED)
                        break
        if polish and removed_any and not stop_requested():
            result, deviation, improved = polished(result, deviation)
            if improved:
                record("polish", result, deviation)

    return finish(result, deviation, True, f"Within tolerance {tolerance:g}.")


def grow_to_counts(sections, counts, degree, options: ChainOptions, params, theta, max_nfev,
                   max_points: Optional[int] = None, callback=None,
                   knot_placement: str = "free") -> FitResult:
    """Grow from the fewest control points to the given counts, refitting each step.

    Each round adds one knot to every span whose largest deviation is at least
    half the section's largest, worst span first, until each section has its
    count; when none of those spans can be split, any span holding two points
    is. With ``"free"`` placement the knots go into those spans and are
    finally moved to where they reduce the error most (unless subsampled);
    with ``"uniform"`` the section is respaced evenly with the new count.
    """
    junctions = parse_junctions(options.continuity, len(sections) - 1, options.fix_junctions)
    start = End(options.fix_ends, None if options.start_tangent is None else tuple(options.start_tangent))
    end = End(options.fix_ends, None if options.end_tangent is None else tuple(options.end_tangent))
    need = minimum_control_points(junctions, degree, start, end)
    knots = [open_uniform_knots(degree, min(n, c)) for n, c in zip(need, counts)]
    data = chain_data(sections)
    while True:
        result = _refit(sections, knots, degree, options, params, theta, max_nfev, max_points, callback)
        if result.message == STOPPED:
            return result
        params, theta = result.parameters, result.theta
        missing = [c - (len(k) - degree - 1) for c, k in zip(counts, knots)]
        if max(missing) <= 0:
            break
        devs = chain_deviations(data, result.curves())
        inserted = False
        for s, (room, (rows, local)) in enumerate(zip(missing, groups(params, len(sections)))):
            if room <= 0:
                continue
            dev = devs[rows]
            new = _take(_new_knots(knots[s], local, dev, 0.5 * float(np.max(dev, initial=0.0)), 1e-4),
                        room, force=False)
            if not new:
                # The worst points can sit alone in their spans, which cannot be
                # split, or their tied spans outnumber the room; any span holding
                # two points can be split, worst first
                new = _take(_new_knots(knots[s], local, dev, -1.0, 1e-4), room)
            if knot_placement == "uniform":
                knots[s] = open_uniform_knots(degree, len(knots[s]) - degree - 1 + max(len(new), 1))
                inserted = True
            elif new:
                knots[s] = clamped_knots(degree, np.concatenate([interior_knots(knots[s], degree), new]))
                inserted = True
        if not inserted:
            break
    if knot_placement == "free" and (max_points is None or all(len(s) <= max_points for s in sections)):
        polished = polish_knots(sections, result, options, callback=callback)
        if polished.message == STOPPED:
            return polished
        if polished.rms_error <= result.rms_error:
            polished.nfev += result.nfev
            polished.success, polished.message = result.success, result.message
            result = polished
    return result
