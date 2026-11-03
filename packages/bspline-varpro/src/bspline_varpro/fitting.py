"""Fit parametric B-spline curves by Levenberg-Marquardt variable projection."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import OptimizeResult

from .basis import centripetal_parameters, open_uniform_knots
from .chain import chain_data, chain_deviations, chain_parameters, validate_chain_parameters
from .constraints import ContinuitySpec, End, Junction
from .problem import VarProProblem
from .reparam import eta_jacobian, eta_to_t, t_to_eta

Callback = Callable[[List[np.ndarray], float], Optional[bool]]

# Message of a result whose fit a callback stopped
STOPPED = "Stopped by callback."


def _takes_knots(callback: Callable) -> bool:
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return False
    return "knots" in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD
                                        for p in parameters.values())


def report(callback: Optional[Callback], control_points: List[np.ndarray], rms_error: float,
           knots: Sequence[np.ndarray]) -> bool:
    """Call a progress callback; True if it asks to stop.

    A callback that accepts a ``knots`` keyword (or ``**kwargs``) also gets
    the knot vectors the control points belong to, which change while knots
    are inserted, moved or removed.
    """
    if callback is None:
        return False
    if _takes_knots(callback):
        return bool(callback(control_points, rms_error, knots=knots))
    return bool(callback(control_points, rms_error))


@dataclass
class FitResult:
    """Fitted curve sections.

    Attributes:
        control_points: One (n_ctrl_s, dim) array per section.
        parameters: Fitted chain parameter of every data point, in [0, S] for S
            sections (:mod:`~bspline_varpro.chain`): the point lies on section
            ``floor(u)``. The points are those of
            :func:`~bspline_varpro.chain.chain_data`, a shared junction point once.
        knots: Clamped knot vector on [0, 1], one per section.
        degree: Spline degree.
        rms_error: sqrt(mean squared distance) between each data point and its
            point on the chain. Excludes the smoothing penalty.
        max_residual: Largest distance between a data point and its curve point
            (an upper bound of the orthogonal deviation).
        nfev: Number of residual evaluations.
        success: True if the solver met a convergence tolerance.
        message: Solver termination message.
        continuity: Label per junction, e.g. ``"G2"``.
        shape_parameters: ``(alpha, beta, gamma)`` of the reparameterization
            linking each junction ((1, 0, 0) for parametric joins).
        theta: Raw constraint parameters (shape and tangent magnitudes), usable
            as ``initial_theta`` to warm start another fit.
        max_deviation: Largest orthogonal distance of a data point to the
            chain, when computed (tolerance fits and refinement).
        history: Progress records of multi-stage procedures.
        options: Constraint settings the fit was made with (reused by
            :func:`~bspline_varpro.refinement.refine` and knot polishing).
    """

    control_points: List[np.ndarray]
    parameters: np.ndarray
    knots: List[np.ndarray]
    degree: int
    rms_error: float
    max_residual: float
    nfev: int
    success: bool
    message: str
    continuity: List[str] = field(default_factory=list)
    shape_parameters: List[Tuple[float, float, float]] = field(default_factory=list)
    theta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    max_deviation: Optional[float] = None
    history: List[dict] = field(default_factory=list)
    options: Optional["ChainOptions"] = None

    @property
    def num_control_points(self) -> List[int]:
        return [len(cp) for cp in self.control_points]

    def curves(self) -> List[BSpline]:
        """One :class:`scipy.interpolate.BSpline` per section, defined on [0, 1]."""
        return [BSpline(k, cp, self.degree) for k, cp in zip(self.knots, self.control_points)]

    def deviations(self, sections: Sequence[np.ndarray]) -> np.ndarray:
        """Orthogonal distance of every point of the chain to the nearest curve."""
        return chain_deviations(chain_data(sections), self.curves())


@dataclass(frozen=True)
class ChainOptions:
    """Constraint and penalty settings shared by every fit of one chain."""

    continuity: Union[ContinuitySpec, Sequence[ContinuitySpec]] = "G1"
    fix_junctions: bool = True
    fix_ends: bool = True
    start_tangent: Optional[Sequence[float]] = None
    end_tangent: Optional[Sequence[float]] = None
    smoothing: float = 1e-3
    regularization: float = 1e-12

    def problem(self, sections, knots, degree: int) -> VarProProblem:
        return VarProProblem(
            sections, knots, degree,
            continuity=list(self.continuity) if isinstance(self.continuity, tuple) else self.continuity,
            fix_junctions=self.fix_junctions,
            smoothing=self.smoothing,
            regularization=self.regularization,
            start=End(self.fix_ends, None if self.start_tangent is None else tuple(self.start_tangent)),
            end=End(self.fix_ends, None if self.end_tangent is None else tuple(self.end_tangent)),
        )


class _Stop(Exception):
    """Aborts least_squares from inside the residual when the callback asks to stop."""


def as_sections(points) -> List[np.ndarray]:
    """Normalize one (n, dim) array or a sequence of arrays to a list of sections."""
    if isinstance(points, np.ndarray) and points.ndim == 2:
        return [np.asarray(points, dtype=np.float64)]
    points = list(points)
    if points and np.ndim(points[0]) == 1:
        return [np.asarray(points, dtype=np.float64)]
    return [np.asarray(sec, dtype=np.float64) for sec in points]


def _validate_sections(sections: List[np.ndarray]) -> None:
    if not sections or any(
        sec.ndim != 2 or len(sec) < 2 or not np.all(np.isfinite(sec)) for sec in sections
    ):
        raise ValueError("Each section needs at least two finite points shaped (n, dim).")
    if len({sec.shape[1] for sec in sections}) != 1:
        raise ValueError("All sections must have the same dimension.")


def initial_chain_parameters(initial_parameters, sections) -> np.ndarray:
    """Chain parameters to start from: the given ones, or centripetal per section."""
    if initial_parameters is None:
        return chain_parameters([centripetal_parameters(sec) for sec in sections], sections)
    return validate_chain_parameters(initial_parameters, sections)


def fit(
    points: Union[np.ndarray, Sequence[np.ndarray]],
    num_control_points: Union[int, Sequence[int], None] = None,
    degree: int = 3,
    *,
    knots: Union[np.ndarray, Sequence[np.ndarray], None] = None,
    continuity: Union[ContinuitySpec, Sequence[ContinuitySpec]] = "G1",
    fix_junctions: bool = True,
    fix_ends: bool = True,
    start_tangent: Optional[Sequence[float]] = None,
    end_tangent: Optional[Sequence[float]] = None,
    smoothing: float = 1e-3,
    regularization: float = 1e-12,
    initial_parameters: Optional[Sequence[np.ndarray]] = None,
    initial_theta: Optional[np.ndarray] = None,
    max_points_per_section: Optional[int] = None,
    max_nfev: Optional[int] = None,
    tolerance: float = 1e-10,
    exact_jacobian: bool = True,
    verbose: int = 0,
    callback: Optional[Callback] = None,
    knot_placement: str = "uniform",
    grow: bool = False,
) -> FitResult:
    """Least-squares fit of a chain of clamped B-spline curves to ordered points.

    Control points and the curve parameter of every data point are optimized
    jointly. Control points are eliminated by variable projection, and the
    parameters (plus the shape parameters of geometric junctions and tangent
    magnitudes) are optimized with Levenberg-Marquardt using the exact reduced
    Jacobian. The chain is fitted to all points as a whole: the sections only
    give the starting split, and a point's parameter slides along the chain,
    across junctions, staying strictly increasing through a softplus
    reparameterization.

    Args:
        points: Ordered points, shape (n, dim), for a single curve, or a sequence
            of such arrays for a chain of sections. Each section needs at least
            two points.
        num_control_points: Control points per section (one int for all sections
            or one per section), with uniformly spaced knots. Ignored if
            ``knots`` is given.
        degree: Spline degree.
        knots: Clamped knot vectors on [0, 1], one shared or one per section.
        continuity: Continuity at the junctions: one spec for all or a list with
            one per junction. ``"G1"``-``"G3"`` geometric, ``"C0"``-``"C3"`` or
            ``0``-``3`` parametric, or a
            :class:`~bspline_varpro.constraints.Junction` for full control.
        fix_junctions: Pin the junction between section j and j + 1 to the last
            point of section j. If False, the junction position is fitted.
        fix_ends: Interpolate the first and last point of the chain.
        start_tangent, end_tangent: Optional directions of travel at the chain ends.
        smoothing: Weight of the integrated squared second derivative penalty
            (bending energy, normalized to trace = control-point count). 0 disables it.
        regularization: Small ridge term keeping the linear solve well posed.
        initial_parameters: Starting parameter values: chain parameters
            (``FitResult.parameters``), or one increasing array from 0 to 1 per
            section. Defaults to centripetal parameterization per section.
        initial_theta: Starting raw constraint parameters (``FitResult.theta``).
        max_points_per_section: For dense data: sections with more points are
            fitted on an evenly spaced subsample, then refined jointly on all
            points (:func:`~bspline_varpro.refinement.refine`).
        max_nfev: Maximum number of residual evaluations of each solve (the
            parametric warm-up of geometric junctions shares it, and a
            finite-difference Jacobian counts against it). None, the default,
            runs until a convergence tolerance is met; the callback can still
            stop it.
        tolerance: ``ftol``, ``xtol`` and ``gtol`` for the solver.
        exact_jacobian: Use the analytic reduced Jacobian instead of finite differences.
        verbose: Verbosity passed to :func:`scipy.optimize.least_squares`.
        callback: Called after every residual evaluation as
            ``callback(control_points, rms_error)`` with one control-point array
            per section, also while knots are grown or moved and during dense
            refinement. A callback with a ``knots`` keyword also gets the knot
            vectors (as does one taking ``**kwargs``). Return True to stop; the
            result then holds the best state evaluated so far (or, while knots
            are moved, the fit before) and ``success`` is False.
        knot_placement: ``"uniform"`` keeps the knots where they start
            (evenly spaced with ``num_control_points``). ``"free"`` then moves
            the interior knots to where they reduce the error most
            (:func:`~bspline_varpro.tolerance.polish_knots`; skipped for
            subsampled dense data).
        grow: Start from the fewest control points the end conditions allow
            and add knots where the fit is worst until each section has its
            count (:func:`~bspline_varpro.tolerance.grow_to_counts`). With
            uniform placement the knots are respaced evenly after each step.
            Needs ``num_control_points``.

    Returns:
        :class:`FitResult`.
    """
    sections = as_sections(points)
    _validate_sections(sections)
    num_sections = len(sections)
    if knots is None:
        if num_control_points is None:
            raise ValueError("Give num_control_points or knots.")
        counts = ([int(num_control_points)] * num_sections if np.ndim(num_control_points) == 0
                  else [int(n) for n in num_control_points])
        if len(counts) != num_sections:
            raise ValueError("Need one control-point count per section.")
        if min(counts) < degree + 1:
            raise ValueError("Control-point count must be at least degree + 1.")
        knot_vectors = [open_uniform_knots(degree, n) for n in counts]
    else:
        knot_vectors = [knots] * num_sections if np.ndim(knots[0]) == 0 else list(knots)
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("Smoothing weight must be finite and nonnegative.")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("Regularization must be finite and nonnegative.")
    if max_nfev is not None and max_nfev < 1:
        raise ValueError("max_nfev must be at least 1 or None.")
    if not (np.isfinite(tolerance) and tolerance > 0):
        raise ValueError("tolerance must be finite and positive.")
    if knot_placement not in ("uniform", "free"):
        raise ValueError("knot_placement must be 'uniform' or 'free'.")
    if grow and knots is not None:
        raise ValueError("Growing the control-point count needs num_control_points, not knots.")

    params = initial_chain_parameters(initial_parameters, sections)

    options = ChainOptions(continuity, fix_junctions, fix_ends, start_tangent, end_tangent,
                           smoothing, regularization)
    if max_points_per_section is not None and max_points_per_section < 2:
        raise ValueError("max_points_per_section must be at least 2.")
    dense = max_points_per_section is not None and any(len(s) > max_points_per_section for s in sections)
    if grow:
        from .tolerance import grow_to_counts
        result = grow_to_counts(sections, counts, degree, options, params, initial_theta,
                                max_nfev, max_points_per_section, callback, knot_placement)
    elif dense:
        from .tolerance import fit_subsampled
        result = fit_subsampled(sections, knot_vectors, degree, options, params, initial_theta,
                                max_nfev, max_points_per_section, callback)
    else:
        result = _fit_once(sections, knot_vectors, degree, options, params, initial_theta,
                           max_nfev, tolerance, exact_jacobian, verbose, callback)
    result.options = options
    if knot_placement == "free" and not grow and not dense and result.message != STOPPED:
        from .tolerance import polish_knots
        polished = polish_knots(sections, result, options, callback=callback)
        if polished.message == STOPPED:
            result = polished
        elif polished.rms_error <= result.rms_error:
            # Moving the knots only refines a converged fit: keep its status.
            polished.nfev += result.nfev
            polished.success, polished.message = result.success, result.message
            polished.options = options
            result = polished
    return result


def _fit_once(sections, knot_vectors, degree, options: ChainOptions, params, initial_theta,
              max_nfev, tolerance, exact_jacobian, verbose, callback) -> FitResult:
    problem = options.problem(sections, knot_vectors, degree)
    theta0 = problem.theta
    if initial_theta is not None:
        theta0 = np.asarray(initial_theta, dtype=np.float64)
        if theta0.shape != (problem.num_params,):
            raise ValueError(f"initial_theta must have {problem.num_params} entries.")
        result = solve_problem(problem, params, theta0, max_nfev=max_nfev, tolerance=tolerance,
                               exact_jacobian=exact_jacobian, verbose=verbose, callback=callback)
    else:
        result = solve_with_continuation(problem, options, params, max_nfev=max_nfev,
                                         tolerance=tolerance, exact_jacobian=exact_jacobian,
                                         verbose=verbose, callback=callback)
    return result


def parametric_options(options: ChainOptions, problem: VarProProblem) -> ChainOptions:
    """The same chain with every geometric junction replaced by its parametric version."""
    junctions = [Junction(jn.order, geometric=False, fixed=jn.fixed, tangent=jn.tangent)
                 for jn in problem.junctions]
    return replace(options, continuity=junctions)


def solve_with_continuation(
    problem: VarProProblem, options: ChainOptions, params, **kwargs,
) -> FitResult:
    """Solve geometric junctions starting from the converged parametric solution.

    The geometric problem has flat directions (shape parameters trade off
    against the data parameterization) that make a cold start drift; the
    parametric C^k fit is a feasible, well-determined starting point.
    """
    if not problem.constraints.has_shape_parameters:
        return solve_problem(problem, params, problem.theta, **kwargs)
    base_problem = parametric_options(options, problem).problem(
        problem.sections, problem.knots, problem.degree)
    base = solve_problem(base_problem, params, base_problem.theta, **kwargs)
    theta0 = problem.constraints.theta_from_parametric(base.theta)
    # Both stages share one evaluation budget
    max_nfev = kwargs.get("max_nfev")
    remaining = None if max_nfev is None else max_nfev - base.nfev
    if base.message == STOPPED or (remaining is not None and remaining < 1):
        # The same curve as a geometric join (alpha = 1), labelled and shaped like the fit asked for
        message = STOPPED if base.message == STOPPED else _MESSAGES[0]
        return result_from_problem(problem, base.parameters, theta0, base.nfev, False, message)
    result = solve_problem(problem, base.parameters, theta0, **{**kwargs, "max_nfev": remaining})
    result.nfev += base.nfev
    return result


def solve_problem(
    problem: VarProProblem,
    params: np.ndarray,
    theta0: np.ndarray,
    *,
    max_nfev: Optional[int] = None,
    tolerance: float = 1e-10,
    exact_jacobian: bool = True,
    verbose: int = 0,
    callback: Optional[Callback] = None,
) -> FitResult:
    """Run Levenberg-Marquardt on a prepared :class:`VarProProblem`."""
    n_eta = problem.total_pts - 1
    nfev = 0
    # Lowest-cost point evaluated: the result if the callback stops the solve
    best_x: Optional[np.ndarray] = None
    best_cost = np.inf

    def unpack(x):
        return eta_to_t(x[:n_eta], problem.num_sections), x[n_eta:]

    def residual(x: np.ndarray) -> np.ndarray:
        nonlocal nfev, best_x, best_cost
        t_flat, theta = unpack(x)
        r = problem.residual(t_flat, theta)
        nfev += 1
        cost = float(r @ r)
        if best_x is None or cost < best_cost:
            best_x, best_cost = x.copy(), cost
        if report(callback, problem.split_control_points(problem.last_control_points),
                  problem.last_rms_error, problem.knots):
            raise _Stop
        return r

    def jacobian(x: np.ndarray) -> np.ndarray:
        return reduced_jacobian(problem, x[:n_eta], x[n_eta:])

    x0 = np.concatenate([t_to_eta(params), theta0])
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

    t_final, theta = unpack(x_final)
    return result_from_problem(problem, t_final, theta, nfev, success, message)


_MESSAGES = {
    -1: "The residual or Jacobian is not finite.",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
}


def _forward_difference(residual, x, r):
    J = np.empty((len(r), len(x)))
    for j in range(len(x)):
        h = np.sqrt(np.finfo(float).eps) * max(1.0, abs(x[j]))
        step = x.copy()
        step[j] += h
        J[:, j] = (residual(step) - r) / h
    return J


def _lm_run(residual, jacobian, x0, max_nfev, tolerance, verbose):
    """One Levenberg-Marquardt run on the normal equations (Cholesky, BLAS).

    Marquardt scaling with the running maximum of diag(J'J), Nielsen damping
    updates, and MINPACK's ftol / xtol / gtol tests. ``max_nfev`` None runs
    until one of them (or the damping limit) ends it; a number caps the
    residual evaluations, those of a finite-difference Jacobian included.
    """
    x = np.array(x0, dtype=np.float64)
    r = residual(x)
    nfev = 1
    cost = 0.5 * float(r @ r)
    finite_differences = jacobian == "2-point"

    def affordable(evaluations):
        return max_nfev is None or nfev + evaluations <= max_nfev

    def linearize(x, r):
        nonlocal nfev
        if finite_differences:
            J = _forward_difference(residual, x, r)
            nfev += len(x)
        else:
            J = jacobian(x)
        return J, J.T @ J, J.T @ r

    def result(status):
        return OptimizeResult(x=x, cost=cost, fun=r, nfev=nfev, status=status,
                              success=status > 0, message=_MESSAGES[status])

    if finite_differences and not affordable(len(x)):
        return result(0)
    J, A, g = linearize(x, r)
    scale = np.maximum(np.diag(A), 1e-300)
    mu, nu = 1e-3, 2.0
    status = 0
    while affordable(1):
        # A non-finite model never yields a valid step: stop instead of raising the damping forever.
        if not (np.isfinite(cost) and np.all(np.isfinite(A)) and np.all(np.isfinite(g))):
            status = -1
            break
        column_norms = np.sqrt(np.maximum(np.diag(A), 0.0))
        r_norm = np.sqrt(2 * cost)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosines = np.where(column_norms > 0, np.abs(g) / (column_norms * r_norm), 0.0)
        if r_norm == 0 or np.max(cosines, initial=0.0) <= tolerance:
            status = 1
            break
        try:
            factor = cho_factor(A + mu * np.diag(scale), check_finite=False)
            step = -cho_solve(factor, g, check_finite=False)
        except np.linalg.LinAlgError:
            step = None
        if step is None or not np.all(np.isfinite(step)):
            mu *= nu
            nu *= 2.0
            if mu > 1e20:
                status = 3
                break
            continue
        if np.linalg.norm(scale ** 0.5 * step) <= tolerance * (tolerance + np.linalg.norm(scale ** 0.5 * x)):
            status = 3
            break
        x_new = x + step
        r_new = residual(x_new)
        nfev += 1
        cost_new = 0.5 * float(r_new @ r_new)
        predicted = -(g @ step + 0.5 * step @ (A @ step))
        rho = (cost - cost_new) / predicted if predicted > 0 else -1.0
        if verbose > 1:
            print(f"LM nfev={nfev} cost={cost_new:.6e} mu={mu:.2e} rho={rho:.3f}")
        if rho > 0 and np.isfinite(cost_new):
            actual_reduction = (cost - cost_new) / cost if cost > 0 else 0.0
            x, r, cost = x_new, r_new, cost_new
            if actual_reduction <= tolerance and predicted / max(cost, 1e-300) <= tolerance:
                status = 2
                break
            if finite_differences and not affordable(len(x)):
                break
            J, A, g = linearize(x, r)
            scale = np.maximum(scale, np.diag(A))
            mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
        else:
            mu *= nu
            nu *= 2.0
            if mu > 1e20:
                status = 3
                break
    return result(status)


def levenberg_marquardt(residual, jacobian, x0, *, max_nfev=None, tolerance=1e-10, verbose=0,
                        max_restarts=8):
    """Levenberg-Marquardt, restarted while it stops on ``xtol`` but still improves.

    After a series of rejected steps the damping grows until the step length
    triggers the ``xtol`` test in strongly curved valleys; a restart resets
    the damping. ``jacobian`` may be ``"2-point"`` for forward differences.
    ``max_nfev`` caps the residual evaluations of all runs together; None
    runs until converged. Returns an ``OptimizeResult`` with ``nfev`` summed
    over restarts.
    """
    if max_nfev is not None and max_nfev < 1:
        raise ValueError("max_nfev must be at least 1 or None.")
    if not tolerance > 0:
        raise ValueError("tolerance must be positive.")
    x = np.asarray(x0, dtype=np.float64)
    used = 0
    cost = np.inf
    solution = None
    for _ in range(max_restarts + 1):
        solution = _lm_run(residual, jacobian, x, None if max_nfev is None else max_nfev - used,
                           tolerance, verbose)
        used += solution.nfev
        improved = solution.cost < 0.99 * cost
        cost = min(cost, solution.cost)
        x = solution.x
        if solution.status != 3 or not improved or (max_nfev is not None and used >= max_nfev):
            break
    solution.nfev = used
    return solution


def reduced_jacobian(problem: VarProProblem, eta: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Jacobian of the reduced residual w.r.t. ``[eta, theta]``."""
    t_flat = eta_to_t(eta, problem.num_sections)
    J_t = problem.jacobian(t_flat, theta)
    J = np.empty((J_t.shape[0], len(eta) + len(theta)))
    J[:, :len(eta)] = J_t @ eta_jacobian(eta, t_flat, problem.num_sections)
    J[:, len(eta):] = problem.parameter_jacobian(t_flat, theta)
    return J


def result_from_problem(
    problem: VarProProblem, t_final: np.ndarray, theta: np.ndarray,
    nfev: int, success: bool, message: str,
) -> FitResult:
    """Assemble a :class:`FitResult` from the problem state at (t, theta)."""
    problem.theta = np.asarray(theta, dtype=np.float64).copy()
    problem.residual(t_final, theta)
    P = problem.last_control_points
    N = problem.solve(t_final, theta=theta)[0]
    return FitResult(
        control_points=[cp.copy() for cp in problem.split_control_points(P)],
        parameters=np.array(t_final, dtype=np.float64),
        knots=[k.copy() for k in problem.knots],
        degree=problem.degree,
        rms_error=problem.last_rms_error,
        max_residual=float(np.max(np.linalg.norm(N @ P - problem.data, axis=1))),
        nfev=nfev,
        success=success,
        message=message,
        continuity=[jn.label for jn in problem.junctions],
        shape_parameters=problem.constraints.shape_values(theta),
        theta=theta.copy(),
    )
