"""Benchmark airfoil fitting methods against exact NACA geometry.

For each reference case and tolerance, every method is asked for the fewest
control points whose maximum orthogonal deviation from the data points stays
within the tolerance. The resulting curves are then scored on deviation from the
exact airfoil, curvature-comb quality, and leading-edge curvature.

Usage:
    python benchmarks/run.py [--quick] [--methods a,b] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import BSpline, splprep

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bspline_varpro as bv  # noqa: E402
from bspline_varpro import diagnostics as dg  # noqa: E402
from references import Case, reference_cases  # noqa: E402

DEGREE = 5
Attempt = Tuple[List[BSpline], int]


def _chain_ctrl_count(curves: List[BSpline]) -> int:
    return sum(len(c.c) for c in curves) - (len(curves) - 1)


def _search(attempt: Callable[[int], Optional[Attempt]], sections, tol: float,
            n_min: int, n_max: int):
    """Smallest n (by galloping then bisection) whose fit meets the tolerance."""
    cache: Dict[int, Tuple[bool, Optional[Attempt]]] = {}

    def ok(n):
        if n not in cache:
            result = attempt(n)
            good = result is not None and dg.max_deviation(sections, result[0]) <= tol
            cache[n] = (good, result)
        return cache[n][0]

    lo, n = n_min - 1, n_min
    while not ok(n):
        lo = n
        if n >= n_max:
            return None
        n = min(n_max, int(np.ceil(n * 1.5)))
    hi = n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return cache[hi][1]


class Skip(Exception):
    """Method not run for this case."""


def uniform_varpro(case: Case, tol: float, smoothing: float = 1e-3,
                   continuity="C2") -> Optional[Attempt]:
    sections = case.sections
    if len(case.points) > 1000:
        raise Skip("dense LM on every point does not finish in reasonable time")

    def attempt(n):
        try:
            res = bv.fit(sections, n, DEGREE, continuity=continuity, smoothing=smoothing, max_nfev=300)
        except ValueError:
            return None
        return res.curves(), _chain_ctrl_count(res.curves())

    n_max = min(len(s) for s in sections)
    return _search(attempt, sections, tol, DEGREE + 1, n_max)


def scipy_splprep(case: Case, tol: float) -> Optional[Attempt]:
    pts = case.points
    u = bv.centripetal_parameters(pts)
    keep = np.concatenate([[True], np.diff(u) > 0])
    pts, u = pts[keep], u[keep]

    def curve_for(s):
        tck, _ = splprep(pts.T, u=u, k=DEGREE, s=s, quiet=2)
        t, c, k = tck
        return BSpline(t, np.column_stack(c)[: len(t) - k - 1], k)

    def meets(curve):
        return dg.max_deviation([pts], [curve]) <= tol

    # Largest smoothing factor s (fewest knots) that still meets the tolerance.
    s = len(pts) * tol ** 2
    best = curve_for(s)
    s_ok, s_fail = (s, None) if meets(best) else (None, s)
    for _ in range(12):
        if s_ok is None:
            s = s_fail / 10
        elif s_fail is None:
            s = s_ok * 10
        else:
            break
        c = curve_for(s)
        if meets(c):
            s_ok, best = s, c
        else:
            s_fail = s
    if s_ok is None:
        best = curve_for(0.0)
    elif s_fail is not None:
        for _ in range(10):
            s = np.sqrt(s_ok * s_fail)
            c = curve_for(s)
            if meets(c):
                s_ok, best = s, c
            else:
                s_fail = s
    return [best], len(best.c)


def tolerance_fit(case: Case, tol: float, fairing: bool = False) -> Optional[Attempt]:
    result = bv.fit_to_tolerance(case.sections, tol, DEGREE, continuity="G2", fairing=fairing)
    if not result.success:
        return None
    return result.curves(), _chain_ctrl_count(result.curves())


METHODS: Dict[str, Callable[[Case, float], Optional[Attempt]]] = {
    "splprep": scipy_splprep,
    "uniform-varpro": uniform_varpro,
    "tolerance": tolerance_fit,
    "tolerance+fair": lambda case, tol: tolerance_fit(case, tol, fairing=True),
}


def evaluate(case: Case, curves: List[BSpline], n_ctrl: int, seconds: float) -> dict:
    fair = dg.fairness(curves)
    section_index, u, _ = dg.point_on_curves(np.zeros(2), curves)
    le_kappa = abs(float(dg.curvature(curves[section_index], np.array([u]))[0]))
    sections = case.sections if len(curves) > 1 else [case.points]
    return {
        "control_points": n_ctrl,
        "max_dev_points": dg.max_deviation(sections, curves),
        "max_dev_truth": float(np.max(dg.distance_to_curves(case.truth, curves))),
        "curvature_extrema": fair["curvature_extrema"],
        "roughness": fair["roughness"],
        "le_curvature_error": le_kappa / case.le_curvature - 1.0,
        "junction_curvature_jump": max((j["curvature_jump"] for j in dg.junctions(curves)), default=0.0),
        "seconds": seconds,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--cases", default="", help="Comma-separated substrings of case names.")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    methods = [m for m in args.methods.split(",") if m]
    cases = reference_cases(args.quick)
    if args.cases:
        cases = [c for c in cases if any(key in c.name for key in args.cases.split(","))]
    rows = []
    header = (f"{'case':22} {'tol':>7} {'method':16} {'ctrl':>5} {'dev pts':>9} {'dev true':>9} "
              f"{'extr':>5} {'rough':>9} {'LE err':>8} {'jump':>8} {'sec':>6}")
    print(header)
    print("-" * len(header))
    for case in cases:
        for tol in case.tolerances:
            for name in methods:
                start = time.perf_counter()
                try:
                    outcome = METHODS[name](case, tol)
                except Skip as reason:
                    print(f"{case.name:22} {tol:7.0e} {name:16} skipped: {reason}")
                    rows.append({"case": case.name, "tolerance": tol, "method": name, "skipped": str(reason)})
                    continue
                seconds = time.perf_counter() - start
                if outcome is None:
                    print(f"{case.name:22} {tol:7.0e} {name:16} {'failed':>5}")
                    rows.append({"case": case.name, "tolerance": tol, "method": name, "failed": True})
                    continue
                metrics = evaluate(case, *outcome, seconds)
                rows.append({"case": case.name, "tolerance": tol, "method": name, **metrics})
                print(f"{case.name:22} {tol:7.0e} {name:16} {metrics['control_points']:5d} "
                      f"{metrics['max_dev_points']:9.2e} {metrics['max_dev_truth']:9.2e} "
                      f"{metrics['curvature_extrema']:5d} {metrics['roughness']:9.2e} "
                      f"{metrics['le_curvature_error']:8.1%} {metrics['junction_curvature_jump']:8.2e} "
                      f"{seconds:6.1f}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
