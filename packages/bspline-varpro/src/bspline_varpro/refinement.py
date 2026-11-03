"""Joint refinement of fitted chains on all data points.

``refine`` minimizes ``sum |C(t_i) - D_i|^2`` (plus the chain's smoothing
penalty) jointly over data parameters and control points, starting from a fit.
Knots, continuity type, and junction shape parameters stay fixed. It is the
second stage for dense data: variable projection on a subsample, then
refinement on every point.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .chain import chain_deviations
from .fitting import ChainOptions, FitResult, as_sections, report
from .joint import JointProblem, solve_joint


def _options(result: FitResult, options: Optional[ChainOptions]) -> ChainOptions:
    if options is not None:
        return options
    if result.options is not None:
        return result.options
    return ChainOptions(continuity=result.continuity or "G1")


def _joint_problem(sections, result: FitResult, options: ChainOptions) -> JointProblem:
    constraints = options.problem(sections, result.knots, result.degree).constraints
    theta = result.theta if len(result.theta) == constraints.num_params else constraints.initial_params()
    C, d, _, _ = constraints.build(theta)
    return JointProblem(sections, result.knots, result.degree, C, d,
                        smoothing=options.smoothing, regularization=options.regularization)


def _to_result(problem: JointProblem, t, z, template: FitResult, iterations, success, message,
               options: ChainOptions) -> FitResult:
    P = problem.control_points(z)
    N = problem._basis(t, derivative=False)
    misfit = np.linalg.norm(N @ P - problem.data, axis=1)
    return FitResult(
        control_points=[P[a:b].copy() for a, b in zip(problem.offsets[:-1], problem.offsets[1:])],
        parameters=t.copy(),
        knots=[k.copy() for k in template.knots],
        degree=template.degree,
        rms_error=float(np.sqrt(np.mean(misfit ** 2))),
        max_residual=float(np.max(misfit)),
        nfev=problem.evaluations,
        success=success,
        message=message,
        continuity=list(template.continuity),
        shape_parameters=list(template.shape_parameters),
        theta=np.array(template.theta, copy=True),
        options=options,
    )


def refine(
    points: Union[np.ndarray, Sequence[np.ndarray]],
    result: FitResult,
    options: Optional[ChainOptions] = None,
    max_iterations: Optional[int] = None,
    callback=None,
) -> FitResult:
    """Joint Levenberg-Marquardt refinement of a fit on all points.

    Useful for dense data: fit a subsample by variable projection, then refine
    on all points. Knots and junction shape parameters stay fixed. The result
    has ``max_deviation`` set. ``max_iterations`` None (the default) runs
    until converged. ``callback`` (as for
    :func:`~bspline_varpro.fitting.fit`) is called after every iteration.
    """
    sections = as_sections(points)
    options = _options(result, options)
    problem = _joint_problem(sections, result, options)
    t0 = np.array(result.parameters, dtype=np.float64)
    z0 = problem.z_from(np.vstack(result.control_points))
    on_step = None
    if callback is not None:
        def on_step(t, z):
            step = _to_result(problem, t, z, result, 0, False, "", options)
            return report(callback, step.control_points, step.rms_error, step.knots)
    t, z, iterations, converged, message = solve_joint(problem, t0, z0, max_iterations, on_step=on_step)
    refined = _to_result(problem, t, z, result, iterations, converged, message, options)
    refined.max_deviation = float(np.max(chain_deviations(problem.data, refined.curves())))
    return refined
