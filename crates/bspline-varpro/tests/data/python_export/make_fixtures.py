"""Writes the CAD export fixtures with the Python version's exporter.

    py -3.12 make_fixtures.py <python checkout> <this folder>

The two sections below are the ones tests/export_matches_python.rs builds.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(sys.argv[1]) / "packages" / "bspline-varpro" / "src"))
from scipy.interpolate import BSpline  # noqa: E402
from bspline_varpro import export  # noqa: E402


def curves():
    out = []
    for s in range(2):
        n, k = 10, 5
        interior = [0.13 + 0.1 * i + 0.01 * s for i in range(n - k - 1)]
        t = [0.0] * (k + 1) + interior + [1.0] * (k + 1)
        c = np.array([[np.cos(0.3 * i + s) * (1 + 0.1 * i), np.sin(0.7 * i) / 3 + s * 0.05] for i in range(n)])
        out.append(BSpline(np.array(t), c, k))
    out[1].c[0] = out[0].c[-1]  # the shared junction
    return out


cs = curves()
folder = Path(sys.argv[2])
for units, tag in [("mm", "mm"), ("inch", "in")]:
    name = "Darkstar ä tip"
    (folder / f"two_sections_{tag}.step").write_text(export.to_step(cs, name=name, scale=1000.0, units=units), encoding="utf-8", newline="")
    (folder / f"two_sections_{tag}.igs").write_text(export.to_iges(cs, name=name, scale=1000.0, units=units), encoding="utf-8", newline="")
    (folder / f"two_sections_{tag}.dxf").write_text(export.to_dxf(cs, scale=1000.0, layer="Profil ä", units=units), encoding="utf-8", newline="")
