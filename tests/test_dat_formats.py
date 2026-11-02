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


def test_import_keeps_unit_chord_coordinates_as_written():
    from spline_fitter.io import normalize_coords
    pts = [tuple(p) for p in np.round(NACA4("2412").points(41), 6)]
    assert normalize_coords(pts) == pts


def test_import_moves_and_scales_without_turning():
    """In millimetres, trailing edge 3 degrees down: the chord is scaled to
    1 and the leading edge moved to x = 0, but the airfoil is not turned
    back level."""
    from spline_fitter.io import normalize_coords
    pts = np.round(NACA4("2412").points(41), 6)
    angle = np.radians(-3.0)
    turn = 250.0 * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = [tuple(turn @ p + (50.0, 7.0)) for p in pts]
    out = np.array(normalize_coords(moved))
    te = 0.5 * (out[0] + out[-1])
    assert np.min(out[:, 0]) == pytest.approx(0.0, abs=1e-12)
    assert te == pytest.approx([1.0, 0.0], abs=1e-12)
    # Still turned: the leading edge stays above the trailing edge
    assert out[np.argmin(out[:, 0]), 1] > 0.04
