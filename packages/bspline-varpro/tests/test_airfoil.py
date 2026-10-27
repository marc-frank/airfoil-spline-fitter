import warnings

import numpy as np
import pytest

from bspline_varpro import airfoil, diagnostics, fit_to_tolerance
from bspline_varpro.airfoil import NACA4


@pytest.fixture(scope="module")
def naca0012_fit():
    pts = NACA4("0012").points(81)
    return pts, airfoil.fit_airfoil(pts, 1e-6)


def test_dat_round_trip(tmp_path):
    pts = np.round(NACA4("2412").points(41), 6)
    path = airfoil.write_dat(tmp_path / "a.dat", pts, name="NACA 2412")
    name, back = airfoil.read_dat(path)
    assert name == "NACA 2412"
    assert np.array_equal(back, pts)


def test_lednicer_is_converted_to_selig(tmp_path):
    pts = np.round(NACA4("2412").points(21), 5)
    upper, lower = airfoil.split_at_leading_edge(pts)
    rows = [f"{x:.5f} {y:.5f}" for x, y in upper[::-1]]
    rows_lower = [f"{x:.5f} {y:.5f}" for x, y in lower]
    text = "NACA 2412\n21. 21.\n\n" + "\n".join(rows) + "\n\n" + "\n".join(rows_lower) + "\n"
    (tmp_path / "l.dat").write_text(text)
    name, back = airfoil.read_dat(tmp_path / "l.dat")
    assert name == "NACA 2412"
    assert np.allclose(back, pts)


def test_decimal_places():
    pts = NACA4("0012").points(41)
    assert airfoil.decimal_places(pts) is None
    for d in (3, 5, 6):
        assert airfoil.decimal_places(np.round(pts, d)) == d


def test_noise_level_separates_noise_from_shape():
    exact = NACA4("0012").points(81)
    assert airfoil.noise_level(exact) < 1e-7
    noisy = exact + np.random.default_rng(0).normal(scale=1e-4, size=exact.shape)
    assert airfoil.noise_level(noisy) == pytest.approx(1e-4, rel=0.25)


def test_default_tolerance():
    exact = NACA4("0012").points(81)
    assert airfoil.default_tolerance(exact) == pytest.approx(1e-6)
    assert airfoil.default_tolerance(np.round(exact, 5)) == pytest.approx(1e-5)
    assert airfoil.default_tolerance(np.round(exact, 4)) == pytest.approx(1e-4)
    noisy = exact + np.random.default_rng(0).normal(scale=1e-4, size=exact.shape)
    assert 3e-4 < airfoil.default_tolerance(noisy) < 6e-4
    assert airfoil.precision_floor(np.round(exact, 5)) == pytest.approx(0.5 * np.sqrt(2) * 1e-5)
    assert airfoil.precision_floor(exact) == 0.0
    scaled = np.round(exact, 5) * 1.0000001
    assert airfoil.default_tolerance(scaled, decimals=5) == pytest.approx(1e-5)


def test_split_at_leading_edge():
    pts = NACA4("4412").points(31)
    upper, lower = airfoil.split_at_leading_edge(pts)
    assert len(upper) + len(lower) == len(pts) + 1
    assert np.array_equal(upper[-1], lower[0])
    assert np.allclose(upper[-1], [0.0, 0.0], atol=1e-12)
    with pytest.raises(ValueError, match="Selig"):
        airfoil.split_at_leading_edge(pts[:31])


def test_fit_airfoil_meets_the_default_tolerance():
    pts = np.round(NACA4("0012").points(81), 5)
    result = airfoil.fit_airfoil(pts)
    assert result.success
    assert result.max_deviation <= 1e-5 * (1 + 1e-6)
    assert result.degree == 5
    assert result.continuity == ["G2"]
    assert len(result.control_points) == 2


def test_fit_airfoil_warns_below_rounding():
    pts = np.round(NACA4("0012").points(41), 3)
    with pytest.warns(UserWarning, match="rounding"):
        airfoil.fit_airfoil(pts, 1e-4, max_iterations=1, optimize_knots=False)


def test_leading_edge_radius(naca0012_fit):
    _, result = naca0012_fit
    exact = 1.0 / NACA4("0012").leading_edge_curvature()
    assert airfoil.leading_edge_radius(result) == pytest.approx(exact, rel=0.01)
    i, u, point = airfoil.leading_edge(result)
    assert np.allclose(point, [0.0, 0.0], atol=1e-5)


def test_trailing_edge_segment():
    sharp = airfoil.fit_airfoil(NACA4("0012").points(41), 1e-4)
    assert airfoil.trailing_edge_segment(sharp) is None
    blunt_pts = NACA4("0012", closed_te=False).points(41)
    blunt = airfoil.fit_airfoil(blunt_pts, 1e-4)
    seg = airfoil.trailing_edge_segment(blunt)
    assert seg.k == blunt.degree
    curves = blunt.curves()
    assert np.allclose(seg(0.0), curves[-1](1.0)) and np.allclose(seg(1.0), curves[0](0.0))
    assert airfoil.trailing_edge_gap(blunt) == pytest.approx(airfoil.trailing_edge_gap(blunt_pts), abs=1e-12)
    mid = seg(np.linspace(0, 1, 11))
    direction = (mid[-1] - mid[0]) / np.linalg.norm(mid[-1] - mid[0])
    offsets = (mid - mid[0]) @ np.array([-direction[1], direction[0]])
    assert np.max(np.abs(offsets)) < 1e-14


def test_resample_lies_on_the_curve(naca0012_fit):
    _, result = naca0012_fit
    pts = airfoil.resample(result, 61)
    assert pts.shape == (121, 2)
    assert np.allclose(pts[60], [0.0, 0.0], atol=1e-5)
    assert np.allclose(pts[0], result.curves()[0](0.0)) and np.allclose(pts[-1], result.curves()[-1](1.0))
    assert np.max(diagnostics.distance_to_curves(pts, result.curves())) < 1e-12
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert steps[59] < 0.2 * steps[30] and steps[0] < 0.2 * steps[30]


def test_resample_with_four_sections():
    pts = NACA4("2412").points(81)
    sections = [pts[:40], pts[39:80], pts[79:120], pts[119:]]
    result = fit_to_tolerance(sections, 1e-4)
    out = airfoil.resample(result, 41, spacing="arc")
    assert out.shape == (81, 2)
    assert np.max(diagnostics.distance_to_curves(out, result.curves())) < 1e-12
    assert out[40, 0] == pytest.approx(0.0, abs=1e-3)
    assert np.all(np.diff(out[:41, 0]) < 0) and np.all(np.diff(out[40:, 0]) > 0)


def test_fit_airfoil_with_custom_sections():
    pts = NACA4("0012").points(61)
    sections = [pts[:40], pts[39:]]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = airfoil.fit_airfoil(pts, 1e-4, sections=sections, le_continuity="G1")
    assert result.continuity == ["G1"]
    assert result.success
