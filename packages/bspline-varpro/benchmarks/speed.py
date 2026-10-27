"""Run time of tolerance fitting versus point count (NACA 0012, tolerance 1e-5).

Usage:
    python benchmarks/speed.py [--sizes 81,201,501,1001,3001]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import bspline_varpro as bv
from bspline_varpro.airfoil import NACA4


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="81,201,501,1001,3001")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--full-limit", type=int, default=501,
                        help="Largest size also run without subsampling.")
    args = parser.parse_args(argv)

    print(f"{'points':>7} {'mode':>12} {'ctrl':>5} {'max dev':>9} {'sec':>7}")
    for n in [int(s) for s in args.sizes.split(",")]:
        pts = NACA4("0012").points(n)
        sections = [pts[:n], pts[n - 1:]]
        modes = [("subsample", 300)] + ([("all points", None)] if n <= args.full_limit else [])
        for label, limit in modes:
            start = time.perf_counter()
            result = bv.fit_to_tolerance(sections, args.tolerance, 5, max_points_per_section=limit)
            seconds = time.perf_counter() - start
            ctrl = sum(result.num_control_points) - 1
            print(f"{len(pts):7d} {label:>12} {ctrl:5d} "
                  f"{result.max_deviation:9.2e} {seconds:7.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
