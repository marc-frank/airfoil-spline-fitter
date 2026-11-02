"""Selig and Lednicer airfoil files load into the same Selig-ordered contour."""
import numpy as np
import pytest
from bspline_varpro.airfoil import NACA4, split_at_leading_edge

from spline_fitter.io import parse_airfoil_data

PTS = np.round(NACA4("2412").points(21), 6)
UPPER, LOWER = split_at_leading_edge(PTS)  # TE -> LE, LE -> TE


def _rows(points, scale=1.0):
    return [f" {scale * x:.6f} {scale * y:.6f}" for x, y in points]


def _load(tmp_path, lines):
    path = tmp_path / "a.dat"
    path.write_text("\n".join(lines) + "\n")
    _, name, coords, precision = parse_airfoil_data(str(path))
    return name, np.array(coords), precision


@pytest.mark.parametrize("lines, name", [
    (["NACA 2412", *_rows(PTS)], "NACA 2412"),
    (["NACA 2412", *_rows(PTS, 100)], "NACA 2412"),
    (["NACA 2412", "21. 21.", "", *_rows(UPPER[::-1]), "", *_rows(LOWER)], "NACA 2412"),
    (["NACA 2412", "21 21", *_rows(UPPER[::-1]), *_rows(LOWER)], "NACA 2412"),
    (["21. 21.", "", *_rows(UPPER[::-1]), "", *_rows(LOWER)], "a"),
    # Counts that do not match the rows: split where x jumps back to the leading edge
    (["NACA 2412", "22. 21.", "", *_rows(UPPER[::-1]), "", *_rows(LOWER)], "NACA 2412"),
    (["NACA 2412", "21. 20.", *_rows(UPPER[::-1]), *_rows(LOWER[1:])], "NACA 2412"),
], ids=["selig", "selig 0-100", "lednicer", "lednicer int counts no blanks", "lednicer no name",
        "lednicer wrong counts", "lednicer unshared leading edge"])
def test_formats_load_the_same_contour(tmp_path, lines, name):
    back_name, coords, precision = _load(tmp_path, lines)
    assert back_name == name
    assert coords.shape == PTS.shape
    assert np.allclose(coords, PTS, atol=1e-6)
    # The point counts do not count as coordinates; precision is at unit
    # chord, so 0-100 data with 6 decimals has 8
    assert precision == (8 if "100" in lines[1] else 6)
