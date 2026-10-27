"""Geometric fairing and joint refinement of fitted chains.

``fair`` minimizes ``sum |C(t_i) - D_i|^2 + weight * integral (dkappa/ds)^2 ds``
jointly over data parameters and control points, starting from a fit. Knots,
continuity type, and junction shape parameters stay fixed. With a tolerance,
the largest weight that keeps every point within the tolerance is searched,
which gives the smoothest curvature comb the tolerance allows.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .diagnostics import section_deviations
from .fitting import ChainOptions, FitResult, as_sections
from .joint import JointProblem, solve_joint


def _options(result: FitResult, options: Optional[ChainOptions]) -> ChainOptions:
    if options is not None:
        return options
    if result.options is not None:
        return result.options
    return ChainOptions(continuity=result.continuity or "G1")


def _joint_problem(sections, result: FitResult, options: ChainOptions, weight: float) -> JointProblem:
    constraints = options.problem(sections, result.knots, result.degree).constraints
    theta = result.theta if len(result.theta) == constraints.num_params else constraints.initial_params()
    C, d, _, _ = constraints.build(theta)
    return JointProblem(sections, result.knots, result.degree, C, d,
                        smoothing=options.smoothing, regularization=options.regularization,
                        fairness=weight)


def _to_result(problem: JointProblem, t, z, template: FitResult, iterations, success, message,
               options: ChainOptions) -> FitResult:
    P = problem.control_points(z)
    N = problem._basis(t, derivative=False)
    misfit = np.linalg.norm(N @ P - problem.data, axis=1)
    return FitResult(
        control_points=[P[a:b].copy() for a, b in zip(problem.offsets[:-1], problem.offsets[1:])],
        parameters=np.split(t.copy(), np.cumsum(problem.sizes)[:-1]),
        knots=[k.copy() for k in template.knots],
        degree=template.degree,
        rms_error=float(np.sqrt(np.mean(misfit ** 2))),
        max_residual=float(np.max(misfit)),
        nfev=iterations,
        success=success,
        message=message,
        continuity=list(template.continuity),
        shape_parameters=list(template.shape_parameters),
        theta=np.array(template.theta, copy=True),
        options=options,
    )


def _solve(sections, start: FitResult, options, weight, max_iterations) -> FitResult:
    problem = _joint_problem(sections, start, options, weight)
    t0 = np.concatenate(start.parameters)
    z0 = problem.z_from(np.vstack(start.control_points))
    t, z, iterations, converged, message = solve_joint(problem, t0, z0, max_iterations)
    result = _to_result(problem, t, z, start, iterations, converged, message, options)
    result.max_deviation = float(max(np.max(d) for d in section_deviations(sections, result.curves())))
    return result


def refine(
    points: Union[np.ndarray, Sequence[np.ndarray]],
    result: FitResult,
    options: Optional[ChainOptions] = None,
    max_iterations: int = 100,
) -> FitResult:
    """Joint Levenberg-Marquardt refinement of a fit (no fairing term).

    Useful for dense data: fit a subsample by variable projection, then refine
    on all points. Knots and junction shape parameters stay fixed.
    """
    sections = as_sections(points)
    return _solve(sections, result, _options(result, options), 0.0, max_iterations)


def fairness_energy(result: FitResult) -> float:
    """Integral of (dkappa/ds)^2 ds over a planar chain."""
    from .diagnostics import fairness
    return fairness(result.curves())["roughness"]


def fair(
    points: Union[np.ndarray, Sequence[np.ndarray]],
    result: FitResult,
    *,
    tolerance: Optional[float] = None,
    weight: Optional[float] = None,
    options: Optional[ChainOptions] = None,
    max_iterations: int = 100,
    search_steps: int = 8,
) -> FitResult:
    """Smooth the curvature comb of a planar fit while keeping close to the data.

    Args:
        points: The data the result was fitted to.
        result: Starting fit (from :func:`fit` or :func:`fit_to_tolerance`).
        tolerance: Keep every point within this orthogonal distance and use the
            largest fairing weight that does (bisection in log scale).
        weight: Fixed fairing weight instead of a tolerance search; it
            multiplies the integral of (dkappa/ds)^2 ds against the sum of
            squared point distances.
        options: Constraint settings (default: those stored in the result).
        max_iterations: Levenberg-Marquardt iterations per solve.
        search_steps: Bisection steps of the weight search.

    Returns:
        :class:`FitResult` with ``max_deviation`` set and the weight search
        recorded in ``history``.
    """
    if (tolerance is None) == (weight is None):
        raise ValueError("Give exactly one of tolerance or weight.")
    sections = as_sections(points)
    options = _options(result, options)
    if weight is not None:
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Fairing weight must be finite and nonnegative.")
        faired = _solve(sections, result, options, weight, max_iterations)
        faired.history = [{"stage": "fair", "weight": weight, "max_deviation": faired.max_deviation,
                           "roughness": fairness_energy(faired)}]
        return faired
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be positive.")

    history = []

    def attempt(w, start):
        candidate = _solve(sections, start, options, w, max_iterations)
        history.append({"stage": "fair", "weight": w, "max_deviation": candidate.max_deviation,
                        "roughness": fairness_energy(candidate)})
        return candidate

    base = attempt(0.0, result)
    if base.max_deviation > tolerance:
        base.history = history
        base.success = False
        base.message = "The starting fit is not within the tolerance."
        return base

    # Scale-free starting weight: the fairing term equals the data term at the start.
    data_cost = max(base.rms_error ** 2 * sum(len(s) for s in sections), 1e-30)
    w = data_cost / max(fairness_energy(base), 1e-300)
    best, best_w = base, 0.0
    low, high = None, None
    for _ in range(24):
        candidate = attempt(w, best)
        if candidate.max_deviation <= tolerance:
            best, best_w, low = candidate, w, w
            if high is not None:
                break
            w *= 10.0
        else:
            high = w
            if low is not None:
                break
            w /= 10.0
    if low is not None and high is not None:
        for _ in range(search_steps):
            w = np.sqrt(low * high)
            candidate = attempt(w, best)
            if candidate.max_deviation <= tolerance:
                best, best_w, low = candidate, w, w
            else:
                high = w
    best.history = history
    best.success = True
    best.message = f"Faired with weight {best_w:.3g} within tolerance {tolerance:g}."
    return best
