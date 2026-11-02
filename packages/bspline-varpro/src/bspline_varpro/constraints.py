"""Linear equality constraints ``C(theta) @ P = d(theta)`` on stacked control points.

A chain of clamped B-spline sections (each with its own knot vector) is joined
at junctions. At a junction between section a (ending) and section b (starting)
the shared point coincides and derivatives up to the junction order are linked.

Parametric continuity C^k equates the derivatives. Geometric continuity G^k only
requires the curves to agree after a local reparameterization u = phi(v) with
phi' = alpha > 0, phi'' = beta, phi''' = gamma (Faa di Bruno)::

    G1:  C_a'   = alpha   C_b'
    G2:  C_a''  = alpha^2 C_b''  + beta C_b'
    G3:  C_a''' = alpha^3 C_b''' + 3 alpha beta C_b'' + gamma C_b'

The shape parameters (alpha, beta, gamma) enter C linearly per coordinate, so
the constraints remain the same for every coordinate and the fit stays separable.
They are optimized as extra outer variables. alpha is kept within a factor
ALPHA_RANGE of the interval between 1 (the parametric join) and its natural
value, the length ratio of the two sections: ln alpha = c + R tanh(a / R)
with c and R the centre and half width of that interval in ln alpha, so a = 0
is its middle. Unbounded, alpha can collapse towards 0, which stops the ending
section dead at the junction and makes the join degenerate (any curvature
satisfies "G2" at zero speed). The interval always holds alpha = 1, so a
geometric join can reproduce the parametric one it is started from.

Optional tangent directions fix a derivative up to a positive magnitude,
``C'(u) = tau * direction`` with tau = softplus(raw), which only moves d.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import BSpline

from .reparam import inverse_softplus, sigmoid, softplus

ALPHA_RANGE = 10.0
_LOG_RANGE = np.log(ALPHA_RANGE)


def _alpha_range(ratio: float) -> Tuple[float, float]:
    """``(centre, half width)`` of the allowed ln alpha for a section length ratio."""
    low = min(np.log(ratio), 0.0) - _LOG_RANGE
    high = max(np.log(ratio), 0.0) + _LOG_RANGE
    return 0.5 * (low + high), 0.5 * (high - low)


def _alpha(a: float, ratio: float) -> Tuple[float, float]:
    """``(alpha, d alpha / d a)`` for the bounded speed ratio of a geometric junction."""
    centre, half = _alpha_range(ratio)
    x = np.tanh(a / half)
    alpha = float(np.exp(centre + half * x))
    return alpha, alpha * (1.0 - x * x)


def _alpha_parameter(alpha: float, ratio: float) -> float:
    """Inverse of :func:`_alpha` (clipped just inside the allowed range)."""
    centre, half = _alpha_range(ratio)
    x = np.clip((np.log(alpha) - centre) / half, -0.999, 0.999)
    return float(half * np.arctanh(x))


def end_derivative_coefficients(
    knots: np.ndarray, degree: int, order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coefficients mapping control points to the ``order``-th derivative at the ends.

    For a clamped B-spline only the first/last ``order + 1`` control points
    affect the derivative at the start / end of the domain::

        C^(order)(end)   = coef_end   @ P[-order-1:]
        C^(order)(start) = coef_start @ P[:order+1]

    Returns:
        ``(coef_end, coef_start)``, each of length ``order + 1``.
    """
    knots = np.asarray(knots, dtype=np.float64)
    if order == 0:
        return np.ones(1), np.ones(1)
    coef_end, coef_start = _end_coefficients(knots.tobytes(), degree, order)
    return coef_end.copy(), coef_start.copy()


@lru_cache(maxsize=4096)
def _end_coefficients(knot_bytes: bytes, degree: int, order: int):
    knots = np.frombuffer(knot_bytes, dtype=np.float64)
    n_ctrl = len(knots) - degree - 1
    derivative = BSpline(knots, np.eye(n_ctrl), degree).derivative(nu=order)
    return derivative(knots[-1])[n_ctrl - order - 1:], derivative(knots[0])[:order + 1]


@dataclass(frozen=True)
class Junction:
    """Continuity requirements where one section ends and the next begins.

    Attributes:
        order: Derivative order linked across the junction (0-3).
        geometric: G^k (True) or parametric C^k (False) continuity.
        fixed: Pin the junction to the last data point of the ending section.
        tangent: Optional direction the curve must leave the junction in.
    """

    order: int = 1
    geometric: bool = True
    fixed: bool = True
    tangent: Optional[Tuple[float, ...]] = None

    @property
    def label(self) -> str:
        return f"{'G' if self.geometric else 'C'}{self.order}"


@dataclass(frozen=True)
class End:
    """Conditions at the start or end of the whole chain.

    Attributes:
        fixed: Interpolate the first / last data point.
        tangent: Optional direction of travel the curve must have there.
    """

    fixed: bool = True
    tangent: Optional[Tuple[float, ...]] = None


ContinuitySpec = Union[int, str, Junction]


def parse_junction(spec: ContinuitySpec, fixed: bool = True) -> Junction:
    """``2`` or ``"C2"`` -> parametric, ``"G2"`` -> geometric, Junction passes through."""
    if isinstance(spec, Junction):
        junction = spec
    elif isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        junction = Junction(int(spec), geometric=False, fixed=fixed)
    elif isinstance(spec, str) and len(spec) == 2 and spec[0].upper() in "CG" and spec[1].isdigit():
        junction = Junction(int(spec[1]), geometric=spec[0].upper() == "G", fixed=fixed)
    else:
        raise ValueError(f"Invalid continuity {spec!r}; use e.g. 'G2', 'C1', 2 or a Junction.")
    if not 0 <= junction.order <= 3:
        raise ValueError("Continuity order must be between 0 and 3.")
    return junction


def parse_junctions(
    continuity: Union[ContinuitySpec, Sequence[ContinuitySpec]], num_junctions: int, fixed: bool = True,
) -> List[Junction]:
    """One :class:`Junction` per junction from a single spec or a per-junction list."""
    if isinstance(continuity, (list, tuple)):
        if len(continuity) != num_junctions:
            raise ValueError(f"Expected {num_junctions} continuity entries, got {len(continuity)}.")
        return [parse_junction(c, fixed) for c in continuity]
    return [parse_junction(continuity, fixed)] * num_junctions


def minimum_control_points(
    junctions: Sequence[Junction], degree: int, start: End = End(), end: End = End(),
) -> List[int]:
    """Fewest control points per section so that end conditions do not overlap."""
    orders = [1 if start.tangent is not None else 0]
    orders += [max(jn.order, 1 if jn.tangent is not None else 0) for jn in junctions]
    orders += [1 if end.tangent is not None else 0]
    return [max(degree + 1, orders[s] + orders[s + 1] + 2) for s in range(len(junctions) + 1)]


def _unit(direction, dim: int) -> np.ndarray:
    v = np.asarray(direction, dtype=np.float64).ravel()
    norm = np.linalg.norm(v)
    if v.shape != (dim,) or not np.isfinite(norm) or norm == 0:
        raise ValueError(f"Tangent direction must be a nonzero vector of length {dim}.")
    return v / norm


class ChainConstraints:
    """Builds ``C(theta)``, ``d(theta)`` and their derivatives for a section chain.

    ``theta`` holds, in order, per geometric junction ``a`` (for order >= 1),
    ``beta`` (order >= 2) and ``gamma`` (order 3), then one raw magnitude per
    tangent condition.
    """

    def __init__(
        self,
        sections: Sequence[np.ndarray],
        knots: Sequence[np.ndarray],
        degree: int,
        junctions: Sequence[Junction],
        start: End = End(),
        end: End = End(),
    ):
        self.sections = [np.asarray(s, dtype=np.float64) for s in sections]
        self.knots = [np.asarray(k, dtype=np.float64) for k in knots]
        self.degree = degree
        self.junctions = list(junctions)
        self.start, self.end = start, end
        self.dim = self.sections[0].shape[1]
        self.n_ctrl = [len(k) - degree - 1 for k in self.knots]
        self.offsets = np.concatenate([[0], np.cumsum(self.n_ctrl)]).astype(int)
        self.total_ctrl = int(self.offsets[-1])
        if len(self.junctions) != len(self.sections) - 1:
            raise ValueError("Need one junction per pair of consecutive sections.")
        for j, junction in enumerate(self.junctions):
            if junction.order > degree:
                raise ValueError(f"Junction {j}: continuity order exceeds the degree.")

        for s, (n, need) in enumerate(zip(self.n_ctrl, minimum_control_points(self.junctions, degree, start, end))):
            if n < need:
                raise ValueError(f"Section {s} needs at least {need} control points for its end conditions.")

        self._coef = {}
        self.param_labels: List[str] = []
        self._layout = []  # (kind, index, slot)
        for j, jn in enumerate(self.junctions):
            if jn.geometric and jn.order >= 1:
                for name in ("alpha", "beta", "gamma")[:jn.order]:
                    self._layout.append(("shape", j, name))
                    self.param_labels.append(f"junction {j} {name}")
        self._tangents = []  # (node, side, unit)
        if start.tangent is not None:
            self._tangents.append((0, "start", _unit(start.tangent, self.dim)))
        for j, jn in enumerate(self.junctions):
            if jn.tangent is not None:
                u = _unit(jn.tangent, self.dim)
                self._tangents.append((j + 1, "start", u))
                if jn.order == 0:
                    self._tangents.append((j, "end", u))
        if end.tangent is not None:
            self._tangents.append((len(self.sections) - 1, "end", _unit(end.tangent, self.dim)))
        for node, side, _ in self._tangents:
            self._layout.append(("tangent", node, side))
            self.param_labels.append(f"section {node} {side} tangent magnitude")
        self.num_params = len(self._layout)

    def coefficients(self, section: int, order: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (section, order)
        if key not in self._coef:
            self._coef[key] = end_derivative_coefficients(self.knots[section], self.degree, order)
        return self._coef[key]

    def _length_ratio(self, junction: int) -> float:
        return self._section_length(junction) / self._section_length(junction + 1)

    def _section_length(self, s: int) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.sections[s], axis=0), axis=1))) or 1.0

    def initial_params(self) -> np.ndarray:
        """Shape parameters of a parametric C^k join with speeds from polyline lengths."""
        theta = np.zeros(self.num_params)
        for i, (kind, index, slot) in enumerate(self._layout):
            if kind == "shape" and slot == "alpha":
                theta[i] = _alpha_parameter(self._length_ratio(index), self._length_ratio(index))
            elif kind == "tangent":
                theta[i] = inverse_softplus(np.array(self._section_length(index)))
        return theta

    def theta_from_parametric(self, tangent_theta: np.ndarray) -> np.ndarray:
        """Theta of the parametric join (alpha = 1, beta = gamma = 0) with given tangent magnitudes."""
        theta = np.zeros(self.num_params)
        tangents = iter(np.asarray(tangent_theta, dtype=np.float64))
        for i, (kind, index, slot) in enumerate(self._layout):
            if kind == "tangent":
                theta[i] = next(tangents)
            elif slot == "alpha":
                theta[i] = _alpha_parameter(1.0, self._length_ratio(index))
        return theta

    @property
    def has_shape_parameters(self) -> bool:
        return any(kind == "shape" for kind, _, _ in self._layout)

    def shape_values(self, theta: np.ndarray) -> List[Tuple[float, float, float]]:
        """``(alpha, beta, gamma)`` per junction (1, 0, 0 for parametric joins)."""
        values = [[1.0, 0.0, 0.0] for _ in self.junctions]
        for i, (kind, index, slot) in enumerate(self._layout):
            if kind == "shape":
                slot_index = ("alpha", "beta", "gamma").index(slot)
                values[index][slot_index] = (_alpha(theta[i], self._length_ratio(index))[0]
                                             if slot == "alpha" else float(theta[i]))
        return [tuple(v) for v in values]

    def build(
        self, theta: Optional[np.ndarray] = None, with_derivatives: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Return ``(C, d, dC, dd)`` where ``dC[i] = dC/dtheta_i`` and ``dd[i] = dd/dtheta_i``."""
        theta = np.zeros(0) if theta is None else np.asarray(theta, dtype=np.float64)
        slots = {}
        for i, (kind, index, slot) in enumerate(self._layout):
            slots[(kind, index, slot)] = i

        rows: List[np.ndarray] = []
        values: List[np.ndarray] = []
        d_rows: List[dict] = []    # per row: {param index: derivative row}
        d_values: List[dict] = []  # per row: {param index: derivative value}

        def new_row():
            return np.zeros(self.total_ctrl)

        def add(row, value=0.0, drow=None, dval=None):
            rows.append(row)
            values.append(np.broadcast_to(np.asarray(value, dtype=np.float64), (self.dim,)).copy())
            d_rows.append(drow or {})
            d_values.append(dval or {})

        def place(row, section, order, at_end, scale=1.0):
            coef_end, coef_start = self.coefficients(section, order)
            if at_end:
                stop = self.offsets[section + 1]
                row[stop - order - 1:stop] += scale * coef_end
            else:
                first = self.offsets[section]
                row[first:first + order + 1] += scale * coef_start
            return row

        last = len(self.sections) - 1
        if self.start.fixed:
            add(place(new_row(), 0, 0, False), self.sections[0][0])
        if self.end.fixed:
            add(place(new_row(), last, 0, True), self.sections[-1][-1])

        for j, jn in enumerate(self.junctions):
            a, b = j, j + 1
            add(place(place(new_row(), a, 0, True), b, 0, False, -1.0))
            if jn.fixed:
                add(place(new_row(), a, 0, True), self.sections[a][-1])
            if jn.order == 0:
                continue
            if jn.geometric:
                ia = slots[("shape", j, "alpha")]
                alpha, dalpha = _alpha(theta[ia], self._length_ratio(j))
                ib = slots.get(("shape", j, "beta"))
                ig = slots.get(("shape", j, "gamma"))
                beta = theta[ib] if ib is not None else 0.0
                gamma = theta[ig] if ig is not None else 0.0
            else:
                alpha, dalpha, beta, gamma, ia, ib, ig = 1.0, 0.0, 0.0, 0.0, None, None, None

            s1 = place(new_row(), b, 1, False)
            s2 = place(new_row(), b, 2, False) if jn.order >= 2 else None
            s3 = place(new_row(), b, 3, False) if jn.order >= 3 else None

            row = place(new_row(), a, 1, True) - alpha * s1
            add(row, drow={ia: -dalpha * s1} if ia is not None else None)
            if jn.order >= 2:
                row = place(new_row(), a, 2, True) - alpha ** 2 * s2 - beta * s1
                drow = {}
                if ia is not None:
                    drow = {ia: -2 * alpha * dalpha * s2, ib: -s1}
                add(row, drow=drow)
            if jn.order >= 3:
                row = place(new_row(), a, 3, True) - alpha ** 3 * s3 - 3 * alpha * beta * s2 - gamma * s1
                drow = {}
                if ia is not None:
                    drow = {ia: -(3 * alpha ** 2 * s3 + 3 * beta * s2) * dalpha,
                            ib: -3 * alpha * s2, ig: -s1}
                add(row, drow=drow)

        for node, side, unit in self._tangents:
            i = slots[("tangent", node, side)]
            tau, dtau = float(softplus(theta[i])), float(sigmoid(theta[i]))
            add(place(new_row(), node, 1, side == "end"), tau * unit, dval={i: dtau * unit})

        # Shaped (0, total_ctrl) and (0, dim) when nothing is constrained
        C = np.array(rows).reshape(len(rows), self.total_ctrl)
        d = np.array(values).reshape(len(values), self.dim)
        dC: List[np.ndarray] = []
        dd: List[np.ndarray] = []
        if with_derivatives:
            for i in range(self.num_params):
                dC_i = np.zeros_like(C)
                dd_i = np.zeros_like(d)
                for r, (drow, dval) in enumerate(zip(d_rows, d_values)):
                    if i in drow:
                        dC_i[r] = drow[i]
                    if i in dval:
                        dd_i[r] = dval[i]
                dC.append(dC_i)
                dd.append(dd_i)
        return C, d, dC, dd


def build_constraints(
    sections: List[np.ndarray],
    knots,
    degree: int,
    continuity: Union[ContinuitySpec, Sequence[ContinuitySpec]] = "G1",
    fix_junctions: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(C, d)`` for a chain with the given continuity and default shape parameters.

    ``knots`` is one knot vector shared by all sections or one per section.
    Integer continuity means parametric C^k. Geometric junctions use the
    initial shape parameters of :meth:`ChainConstraints.initial_params`.
    """
    if np.ndim(knots[0]) == 0:
        knots = [knots] * len(sections)
    junctions = parse_junctions(continuity, len(sections) - 1, fix_junctions)
    chain = ChainConstraints(sections, knots, degree, junctions)
    C, d, _, _ = chain.build(chain.initial_params())
    return C, d
