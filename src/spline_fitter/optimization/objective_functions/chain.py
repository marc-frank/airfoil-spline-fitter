"""
Chain Data
==========

The data points of the whole chain, fitted as one: the sections only give the
starting split and the junction positions. Every point has a chain parameter
u in [0, S] for S sections and lies on section ``floor(u)``, so no point
belongs to a section for good (see ``bspline_varpro.chain``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from bspline_varpro.chain import chain_data, chain_parameters
from scipy.interpolate import BSpline

from ...spline import BSplineCurve2D


@dataclass(frozen=True)
class ChainPoints:
    """All data points once (a shared junction point once), in order.

    Attributes:
        data: Points, shape (n, 2)
        parameters: Chain parameter of every point, from 0 to ``num_sections``
        num_sections: Number of sections
    """
    data: np.ndarray
    parameters: np.ndarray
    num_sections: int

    @classmethod
    def from_sections(cls, sections: Sequence[Sequence[tuple[float, float]]],
                      t_params: Sequence[Sequence[float]]) -> ChainPoints:
        """Chain of the given sections with local parameters in [0, 1] per section."""
        arrays = [np.asarray(sec, dtype=np.float64) for sec in sections]
        return cls(chain_data(arrays), chain_parameters(t_params, arrays), len(arrays))


def scipy_splines(curves: Sequence[BSplineCurve2D]) -> list[BSpline]:
    """The curves as scipy B-splines (for the chain projection)."""
    return [BSpline(np.asarray(c.knot_vector, dtype=np.float64),
                    np.asarray(c.control_points, dtype=np.float64), c.degree)
            for c in curves]
