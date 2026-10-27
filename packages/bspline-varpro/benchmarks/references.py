"""Reference airfoil data sets with known exact geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from bspline_varpro.airfoil import NACA4


@dataclass
class Case:
    name: str
    points: np.ndarray            # Selig order: upper TE -> LE -> lower TE
    le_index: int
    truth: np.ndarray             # dense exact samples of both surfaces
    le_curvature: float
    tolerances: List[float] = field(default_factory=list)

    @property
    def sections(self) -> List[np.ndarray]:
        return [self.points[: self.le_index + 1], self.points[self.le_index:]]


def _case(name, airfoil, n_per_side, tolerances, decimals=None, noise=0.0, seed=0):
    pts = airfoil.points(n_per_side)
    if noise:
        rng = np.random.default_rng(seed)
        pts = pts + rng.normal(scale=noise, size=pts.shape)
    if decimals is not None:
        pts = np.round(pts, decimals)
    beta = np.linspace(0.0, np.pi, 4001)
    sigma = np.sin(beta / 2)
    truth = np.vstack([airfoil.surface(sigma, True), airfoil.surface(sigma, False)])
    return Case(name, pts, n_per_side - 1, truth, airfoil.leading_edge_curvature(), tolerances)


def reference_cases(quick: bool = False) -> List[Case]:
    naca0012 = NACA4("0012")
    cases = [
        _case("naca0012 exact", naca0012, 81, [1e-4, 1e-5, 1e-6]),
        _case("naca0012 6 dec", naca0012, 81, [1e-4, 1e-5], decimals=6),
        _case("naca0012 5 dec", naca0012, 81, [1e-4, 2e-5], decimals=5),
        _case("naca0012 4 dec", naca0012, 81, [1e-4], decimals=4),
        _case("naca2412 5 dec", NACA4("2412"), 81, [1e-4, 2e-5], decimals=5),
        _case("naca4412 blunt 6 dec", NACA4("4412", closed_te=False), 81, [1e-4, 1e-5], decimals=6),
        _case("naca0012 noisy 1e-4", naca0012, 101, [3e-4], noise=1e-4),
        _case("naca0012 dense", naca0012, 1001, [1e-5]),
    ]
    if quick:
        return [cases[0], cases[2], cases[4]]
    return cases
