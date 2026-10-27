"""Tolerance-driven fitting: few control points within a maximum orthogonal deviation.

Starting from the fewest control points the end conditions allow, knots are
inserted in every knot span that holds a point farther than the tolerance from
the curve, and the chain is refitted warm-started from the previous parameters.
Once the tolerance is met, interior knot positions are optimized ("free-knot
polish") and knots whose removal keeps the fit within tolerance are removed.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from .basis import centripetal_parameters, clamped_knots, interior_knots, open_uniform_knots
from .constraints import ContinuitySpec, End, minimum_control_points, parse_junctions
from .diagnostics import section_deviations
from .fitting import (
    ChainOptions, FitResult, _validate_parameters, _validate_sections, as_sections,
    levenberg_marquardt, reduced_jacobian, result_from_problem, solve_problem, solve_with_continuation,
)
from .reparam import eta_to_t, inverse_softplus, softplus, t_to_eta

ProgressCallback = Callable[[FitResult, float], Optional[bool]]


def _refit(sections, knots, degree, options: ChainOptions, t_sections, theta, max_nfev,
           max_points: Optional[int] = None) -> FitResult:
    if max_points is not None and any(len(s) > max_points for s in sections):
        return fit_subsampled(sections, knots, degree, options, t_sections, theta, max_nfev, max_points)
    problem = options.problem(sections, knots, degree)
    if theta is None or len(theta) != problem.num_params:
        return solve_with_continuation(problem, options, t_sections, max_nfev=max_nfev)
    return solve_problem(problem, t_sections, theta, max_nfev=max_nfev)


def subsample_indices(n: int, max_points: int) -> np.ndarray:
    """Evenly spaced indices including both ends."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_points)).astype(int))


def fit_subsampled(sections, knots, degree, options: ChainOptions, t_sections, theta, max_nfev,
                   max_points: int) -> FitResult:
    """Variable projection on a subsample, then joint refinement on all points.

    Variable projection costs O(n^3) per iteration in the number of points,
    the joint solver O(n k^2) in the number of control coordinates k.
    """
    from .diagnostics import project
    from .refinement import refine

    indices = [subsample_indices(len(s), max_points) for s in sections]
    subsets = [s[i] for s, i in zip(sections, indices)]
    t_sub = [np.asarray(t)[i] for t, i in zip(t_sections, indices)]
    coarse = _refit(subsets, knots, degree, options, t_sub, theta, max_nfev)
    t_all = []
    for sec, curve in zip(sections, coarse.curves()):
        t = np.maximum.accumulate(np.clip(project(sec, curve)[0], 0.0, 1.0))
        t[0], t[-1] = 0.0, 1.0
        t_all.append(t)
    coarse.parameters = t_all
    coarse.options = options
    refined = refine(sections, coarse, options)
    refined.nfev += coarse.nfev
    return refined


def _spans(knots: np.ndarray, t: np.ndarray):
    breaks = np.unique(knots)
    index = np.clip(np.searchsorted(breaks, t, side="right") - 1, 0, len(breaks) - 2)
    return breaks, index


def _new_knots(knots, t, dev, tol, min_span) -> List[float]:
    """One knot per knot span that holds a point outside the tolerance."""
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
            new.append(u)
    return new


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
    return float(max(np.max(d) for d in section_deviations(sections, result.curves())))


def polish_knots(
    points: Union[np.ndarray, Sequence[np.ndarray]],
    result: FitResult,
    options: Optional[ChainOptions] = None,
    max_nfev: int = 60,
    min_gap: float = 1e-3,
) -> FitResult:
    """Optimize interior knot positions (at fixed counts) together with the fit.

    Knots stay ordered with a minimum spacing through a normalized softplus gap
    parameterization. Knot columns of the Jacobian are finite differences; the
    data-parameter and shape columns are exact.
    """
    sections = as_sections(points)
    options = options or result.options or ChainOptions(continuity=result.continuity or "G1")
    degree = result.degree
    counts = [len(interior_knots(k, degree)) for k in result.knots]
    if sum(counts) == 0:
        return result
    sizes = [len(s) for s in sections]
    n_eta = sum(n - 1 for n in sizes)
    n_theta = len(result.theta)

    def to_xi(knots):
        gaps = np.diff(np.concatenate([[0.0], interior_knots(knots, degree), [1.0]]))
        return inverse_softplus(np.maximum(gaps - min_gap, 1e-12))

    def to_knots(xi):
        knots, offset = [], 0
        for s, r in enumerate(counts):
            if r == 0:
                knots.append(result.knots[s])
                continue
            gaps = softplus(xi[offset:offset + r + 1]) + min_gap
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
        eta, theta, xi = split(x)
        return problem_for(xi).residual(eta_to_t(eta, sizes)[0], theta)

    def jacobian(x):
        eta, theta, xi = split(x)
        J = np.empty((len(residual_at(x)), len(x)))
        J[:, :n_eta + n_theta] = reduced_jacobian(problem_for(xi), eta, theta)
        t_flat = eta_to_t(eta, sizes)[0]
        base = problem_for(xi).residual(t_flat, theta)
        for i in range(len(xi)):
            h = 1e-7 * (1.0 + abs(xi[i]))
            shifted = xi.copy()
            shifted[i] += h
            other = options.problem(sections, to_knots(shifted), degree)
            J[:, n_eta + n_theta + i] = (other.residual(t_flat, theta) - base) / h
        return J

    xi0 = np.concatenate([to_xi(k) for k, r in zip(result.knots, counts) if r > 0])
    x0 = np.concatenate([t_to_eta(result.parameters), result.theta, xi0])
    try:
        solution = levenberg_marquardt(residual_at, jacobian, x0, max_nfev=max_nfev)
    except (ValueError, np.linalg.LinAlgError):
        return result
    eta, theta, xi = split(solution.x)
    problem = options.problem(sections, to_knots(xi), degree)
    return result_from_problem(problem, eta_to_t(eta, sizes)[0], theta, solution.nfev,
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
    initial_parameters: Optional[Sequence[np.ndarray]] = None,
    max_control_points: Optional[int] = None,
    optimize_knots: bool = True,
    remove_knots: bool = True,
    max_points_per_section: Optional[int] = 200,
    max_nfev: int = 200,
    max_iterations: int = 40,
    callback: Optional[ProgressCallback] = None,
) -> FitResult:
    """Fit a chain with few control points so every point lies within ``tolerance``.

    Args:
        points: One (n, dim) array or a sequence of sections (see :func:`fit`).
        tolerance: Maximum allowed orthogonal distance of any point to its
            section's curve, in data units.
        degree: Spline degree (5 recommended for CAD).
        continuity, fix_junctions, fix_ends, start_tangent, end_tangent,
            smoothing, regularization: As for :func:`fit`.
        initial_knots: Starting knot vectors per section (default: the fewest
            control points the end conditions allow).
        initial_parameters: Starting data parameters per section.
        max_control_points: Upper limit per section.
        optimize_knots: Polish interior knot positions once the tolerance is met.
        remove_knots: Try to remove knots once the tolerance is met.
        max_points_per_section: Larger sections are fitted by variable
            projection on an evenly spaced subsample, then refined jointly on
            all points (knot polishing is skipped). None uses all points.
        max_nfev: Residual evaluations per refit.
        max_iterations: Maximum insertion rounds.
        callback: ``callback(result, max_deviation)`` after every refit; return
            True to stop early (the last result is returned).

    Returns:
        :class:`FitResult` with ``max_deviation`` and a ``history`` of stages.
        ``success`` is False if the tolerance could not be met.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be positive.")
    sections = as_sections(points)
    _validate_sections(sections)
    options = ChainOptions(continuity, fix_junctions, fix_ends, start_tangent, end_tangent,
                           smoothing, regularization)
    junctions = parse_junctions(continuity, len(sections) - 1, fix_junctions)
    start = End(fix_ends, None if start_tangent is None else tuple(start_tangent))
    end = End(fix_ends, None if end_tangent is None else tuple(end_tangent))
    need = minimum_control_points(junctions, degree, start, end)
    if initial_knots is None:
        knots = [open_uniform_knots(degree, n) for n in need]
    else:
        knots = [np.asarray(k, dtype=np.float64) for k in initial_knots]
    limits = [min(len(sec) - 1, max_control_points or len(sec)) for sec in sections]
    if any(n > limit for n, limit in zip(need, limits)):
        raise ValueError("Sections have too few points for the requested end conditions.")
    if initial_parameters is None:
        t_sections = [centripetal_parameters(sec) for sec in sections]
    else:
        t_sections = _validate_parameters(initial_parameters, sections)

    history: List[dict] = []
    theta = None
    min_span = 1e-4

    def record(stage, result, deviation):
        history.append({"stage": stage, "control_points": result.num_control_points,
                        "max_deviation": deviation, "rms_error": result.rms_error})
        return callback is not None and bool(callback(result, deviation))

    def finish(result, deviation, success, message):
        result.max_deviation = deviation
        result.history = history
        result.options = options
        result.success = success
        result.message = message
        return result

    result = None
    for _ in range(max_iterations):
        result = _refit(sections, knots, degree, options, t_sections, theta, max_nfev, max_points_per_section)
        t_sections, theta = result.parameters, result.theta
        devs = section_deviations(sections, result.curves())
        deviation = float(max(np.max(d) for d in devs))
        if record("insert", result, deviation):
            return finish(result, deviation, False, "Stopped by callback.")
        if deviation <= tolerance:
            break
        inserted = False
        for s in range(len(sections)):
            if np.max(devs[s]) <= tolerance:
                continue
            room = limits[s] - (len(knots[s]) - degree - 1)
            new = _new_knots(knots[s], t_sections[s], devs[s], tolerance, min_span)[:max(room, 0)]
            if new:
                knots[s] = clamped_knots(degree, np.concatenate([interior_knots(knots[s], degree), new]))
                inserted = True
        if not inserted:
            return finish(result, deviation, False,
                          "Tolerance not reached: no knot span can be refined further.")
    else:
        return finish(result, deviation, False, "Tolerance not reached within max_iterations.")

    dense = max_points_per_section is not None and any(len(s) > max_points_per_section for s in sections)
    polish = optimize_knots and not dense

    def polished(result, deviation):
        """Knot-polished result if it stays within tolerance without raising the RMS error."""
        candidate = polish_knots(sections, result, options)
        candidate_dev = _max_deviation(sections, candidate)
        if candidate_dev <= tolerance and candidate.rms_error <= result.rms_error:
            return candidate, candidate_dev, True
        return result, deviation, False

    # Polish before removing: better-placed knots make more of them redundant.
    if polish:
        result, deviation, improved = polished(result, deviation)
        if improved and record("polish", result, deviation):
            return finish(result, deviation, False, "Stopped by callback.")

    if remove_knots:
        removed_any = False
        for s in range(len(sections)):
            # Try every knot, least useful first; after a removal start over, since the
            # refitted curve can make knots removable that failed before.
            removed = True
            while removed and len(result.knots[s]) - degree - 1 > need[s]:
                removed = False
                devs = section_deviations(sections, result.curves())
                for k in _removal_order(result.knots[s], degree, result.parameters[s], devs[s], tolerance):
                    trial_knots = list(result.knots)
                    inner = interior_knots(result.knots[s], degree)
                    trial_knots[s] = clamped_knots(degree, np.delete(inner, np.argmin(np.abs(inner - k))))
                    trial = _refit(sections, trial_knots, degree, options, result.parameters, result.theta,
                                   max_nfev, max_points_per_section)
                    trial_dev = _max_deviation(sections, trial)
                    if trial_dev > tolerance and polish:
                        trial = polish_knots(sections, trial, options)
                        trial_dev = _max_deviation(sections, trial)
                    if trial_dev <= tolerance:
                        result, deviation, removed, removed_any = trial, trial_dev, True, True
                        if record("remove", result, deviation):
                            return finish(result, deviation, False, "Stopped by callback.")
                        break
        if polish and removed_any:
            result, deviation, improved = polished(result, deviation)
            if improved:
                record("polish", result, deviation)

    return finish(result, deviation, True, f"Within tolerance {tolerance:g}.")
