"""The data of a chain as one ordered point sequence.

The sections passed to a fit only give the starting split: the chain is
fitted to all points as a whole, and no point belongs to a section for good.
Every point has a chain parameter u in [0, S] for S sections; it lies on
section ``floor(u)`` at local parameter ``u - floor(u)`` (the end u = S on
the last section at 1), so a point moves across a junction when its
parameter does. A point shared by consecutive sections (their junction) is
one data point.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import BSpline


def shared_junctions(sections: Sequence[np.ndarray]) -> List[bool]:
    """Whether each section after the first starts at the last point of the one before
    (equal to within rounding of the data's extent)."""
    points = np.vstack([np.asarray(s, dtype=np.float64) for s in sections])
    tolerance = 1e-12 * max(float(np.max(np.ptp(points, axis=0))), 1e-300)
    return [bool(np.max(np.abs(np.asarray(a[-1], dtype=np.float64) - b[0])) <= tolerance)
            for a, b in zip(sections[:-1], sections[1:])]


def chain_data(sections: Sequence[np.ndarray]) -> np.ndarray:
    """All points in order, a shared junction point once."""
    shared = shared_junctions(sections)
    parts = [np.asarray(sections[0], dtype=np.float64)]
    for sec, skip in zip(sections[1:], shared):
        sec = np.asarray(sec, dtype=np.float64)
        parts.append(sec[1:] if skip else sec)
    return np.vstack(parts)


def chain_parameters(t_sections: Sequence[np.ndarray], sections: Sequence[np.ndarray]) -> np.ndarray:
    """Chain parameters from local parameters in [0, 1] per section (section s adds s).

    A shared junction point is one point at the integer. Where sections do
    not share it, the first point of the next section starts halfway along its
    first step: two points at one parameter could never move apart.
    """
    shared = [False] + shared_junctions(sections)
    parts = []
    for s, (t, skip) in enumerate(zip(t_sections, shared)):
        t = np.array(t, dtype=np.float64)
        if skip:
            t = t[1:]
        elif s > 0 and len(t) > 1:
            t[0] = 0.5 * t[1]
        parts.append(t + s)
    return np.concatenate(parts)


def locate(u: np.ndarray, num_sections: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(section, local parameter)`` of every chain parameter."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, num_sections)
    section = np.minimum(np.floor(u).astype(int), num_sections - 1)
    return section, u - section


def groups(u: np.ndarray, num_sections: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """``(point indices, local parameters)`` of the points on each section."""
    section, local = locate(u, num_sections)
    return [(np.flatnonzero(section == s), local[section == s]) for s in range(num_sections)]


def chain_deviations(data: np.ndarray, curves: Sequence[BSpline], samples_per_span: int = 32) -> np.ndarray:
    """Orthogonal distance of every point to the nearest point of the whole chain."""
    from .diagnostics import distance_to_curves
    return distance_to_curves(data, curves, samples_per_span)


def evaluate_chain(curves: Sequence[BSpline], u: np.ndarray, nu: int = 0) -> np.ndarray:
    """Points (or derivatives w.r.t. u) of the chain at chain parameters."""
    section, local = locate(u, len(curves))
    out = np.empty((len(local), np.asarray(curves[0].c).shape[1]))
    for s, curve in enumerate(curves):
        rows = section == s
        if np.any(rows):
            out[rows] = curve(local[rows], nu=nu)
    return out


def validate_chain_parameters(
    parameters, sections: Sequence[np.ndarray], data: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Chain parameters from either form a caller may give.

    One nondecreasing array from 0 to S with a value per point of
    :func:`chain_data`, or one array from 0 to 1 per section (each with a
    value per point of that section).
    """
    data = chain_data(sections) if data is None else data
    num_sections = len(sections)
    if len(parameters) and np.ndim(parameters[0]) == 0:
        u = np.array(parameters, dtype=np.float64)
        if (u.ndim != 1 or len(u) != len(data) or not np.all(np.isfinite(u))
                or np.any(np.diff(u) < 0) or not np.isclose(u[0], 0)
                or not np.isclose(u[-1], num_sections)):
            raise ValueError(
                "Chain parameters need one value per point, increasing from 0 to the section count.")
        return u
    t_sections = [np.array(t, dtype=np.float64) for t in parameters]
    if len(t_sections) != num_sections or any(
        t.ndim != 1 or len(t) != len(sec) or not np.all(np.isfinite(t))
        or np.any(np.diff(t) < 0) or not np.isclose(t[0], 0) or not np.isclose(t[-1], 1)
        for t, sec in zip(t_sections, sections)
    ):
        raise ValueError("Parameters must match each section and increase from 0 to 1.")
    return chain_parameters(t_sections, sections)
