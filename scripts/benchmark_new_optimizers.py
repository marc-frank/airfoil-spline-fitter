#!/usr/bin/env python3
"""
Benchmark Optimizers
====================

Times the app's optimizers on NACA airfoils and measures each fit with the
app's fit report (orthogonal deviation of every data point):

- Direct LSQ (baseline linear fit with fixed parameters)
- VarPro LM (bspline-varpro)

Usage:
    python scripts/benchmark_new_optimizers.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Run from a source checkout without installing
_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "packages" / "bspline-varpro" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bspline_varpro.airfoil import NACA4  # noqa: E402

from spline_fitter.analysis import fit_report  # noqa: E402
from spline_fitter.geometry import create_airfoil_data  # noqa: E402
from spline_fitter.optimization.engine import run_optimization  # noqa: E402
from spline_fitter.optimization.models import (  # noqa: E402
    ContinuityMode,
    FitResult,
    OptimizationConfig,
    OptimizerType,
    SplineConfig,
)

TEST_CASES = [
    ("NACA0012", "0012", 80),   # Symmetric
    ("NACA2412", "2412", 80),   # Cambered
    ("NACA4415", "4415", 100),  # Higher camber
]

OPTIMIZERS = [
    (OptimizerType.DIRECT_LSQ, "Direct LSQ (baseline)"),
    (OptimizerType.GN_VARPRO, "VarPro LM"),
]


def run_benchmark() -> list[dict]:
    """Run every optimizer on every test airfoil and print the results."""
    config = SplineConfig(degree=5, num_control_points=20, num_sections=2,
                          continuity=ContinuityMode.G1)

    print("=" * 80)
    print("OPTIMIZER BENCHMARK: " + " vs ".join(name for _, name in OPTIMIZERS))
    print("=" * 80)

    results = []
    for airfoil_name, code, n_per_side in TEST_CASES:
        points = np.round(NACA4(code).points(n_per_side), 6)
        airfoil = create_airfoil_data([tuple(p) for p in points], num_sections=2,
                                      filename=f"{airfoil_name}.dat", name=airfoil_name,
                                      precision=6)
        print(f"\n{'-' * 80}\nAirfoil: {airfoil_name} ({airfoil.num_points} points)\n{'-' * 80}")

        for opt_type, opt_name in OPTIMIZERS:
            start = time.perf_counter()
            row = {"airfoil": airfoil_name, "optimizer": opt_name, "max_dev": float("inf"),
                   "rms_dev": float("inf"), "iterations": 0}
            try:
                result = run_optimization(config, OptimizationConfig(optimizer=opt_type,
                                                                     max_iterations=200),
                                          airfoil)
                row["time_ms"] = (time.perf_counter() - start) * 1000
                if result.section_control_points is not None:
                    fit = FitResult(result.section_control_points, config.degree, result.fun,
                                    result.nit, result.nit, 0.0,
                                    section_knots=result.section_knots)
                    report = fit_report(airfoil.sections, fit, precision=6)
                    row.update(max_dev=report["max_deviation"], rms_dev=report["rms_deviation"],
                               iterations=result.nit)
                    print(f"  {opt_name:25s} | {row['time_ms']:8.1f} ms | max {row['max_dev']:.2e}"
                          f" | RMS {row['rms_dev']:.2e} | iters {row['iterations']:4d}")
                else:
                    print(f"  {opt_name:25s} | FAILED (no control points)")
            except Exception as e:
                row["time_ms"] = (time.perf_counter() - start) * 1000
                print(f"  {opt_name:25s} | ERROR: {e}")
            results.append(row)

    print(f"\n{'=' * 80}\nAVERAGE BY OPTIMIZER\n{'=' * 80}")
    print(f"{'Optimizer':<25} | {'Time (ms)':>10} | {'Max dev':>10} | {'RMS dev':>10} | Success")
    stats = defaultdict(list)
    for row in results:
        if np.isfinite(row["max_dev"]):
            stats[row["optimizer"]].append(row)
    for _, opt_name in OPTIMIZERS:
        rows = stats[opt_name]
        if rows:
            print(f"{opt_name:<25} | {np.mean([r['time_ms'] for r in rows]):>10.1f} | "
                  f"{np.mean([r['max_dev'] for r in rows]):>10.2e} | "
                  f"{np.mean([r['rms_dev'] for r in rows]):>10.2e} | "
                  f"{len(rows)}/{len(TEST_CASES)}")
        else:
            print(f"{opt_name:<25} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | 0/{len(TEST_CASES)}")
    return results


if __name__ == "__main__":
    run_benchmark()
