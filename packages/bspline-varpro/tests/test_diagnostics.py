import numpy as np
import pytest
from scipy.interpolate import BSpline, make_interp_spline

from bspline_varpro import diagnostics as dg
from bspline_varpro.airfoil import NACA4


def _arc(k=5):
    theta = np.linspace(0, np.pi, 60)
    return make_interp_spline(theta / np.pi, np.column_stack([np.cos(theta), np.sin(theta)]), k=k)


def test_projection_matches_brute_force():
    curve = _arc()
    q = np.random.default_rng(0).normal(size=(300, 2)) * 0.4 + [0.0, 0.5]
    u, d = dg.project(q, curve)
    samples = curve(np.linspace(0, 1, 400001))
    brute = np.array([np.min(np.linalg.norm(samples - p, axis=1)) for p in q])
    assert np.all(d <= brute + 1e-12)
    assert np.max(brute - d) < 1e-8
    np.testing.assert_allclose(np.linalg.norm(curve(u) - q, axis=1), d, atol=1e-15)


def test_projection_is_orthogonal_in_the_interior():
    curve = _arc()
    q = np.array([[0.3, 0.2], [-0.5, 1.4], [0.9, 0.9]])
    u, _ = dg.project(q, curve)
    tangent = curve(u, nu=1)
    assert np.max(np.abs(np.sum((curve(u) - q) * tangent, axis=1))) < 1e-12


def test_projection_handles_end_points():
    curve = _arc()
    u, d = dg.project(np.array([[2.0, -1.0]]), curve)
    assert u[0] == 0.0
    assert d[0] == pytest.approx(np.hypot(1.0, 1.0))


def test_circle_curvature_and_slope():
    curve = _arc()
    u = np.linspace(0.05, 0.95, 20)
    np.testing.assert_allclose(dg.curvature(curve, u), 1.0, atol=1e-6)
    assert np.max(np.abs(dg.curvature_slope(curve, u))) < 1e-3


def test_curvature_slope_matches_finite_difference():
    rng = np.random.default_rng(3)
    curve = BSpline(np.r_[np.zeros(6), 0.3, 0.6, np.ones(6)], rng.normal(size=(8, 2)), 5)
    u = np.linspace(0.1, 0.9, 7)
    h = 1e-6
    speed = np.linalg.norm(curve(u, nu=1), axis=1)
    fd = (dg.curvature(curve, u + h) - dg.curvature(curve, u - h)) / (2 * h) / speed
    np.testing.assert_allclose(dg.curvature_slope(curve, u), fd, rtol=1e-5, atol=1e-6)


def test_fairness_counts_wiggles():
    x = np.linspace(0, 1, 40)
    smooth = make_interp_spline(x, np.column_stack([x, 0.1 * x ** 2]), k=5)
    wiggly = make_interp_spline(x, np.column_stack([x, 0.1 * x ** 2 + 1e-3 * np.sin(12 * np.pi * x)]), k=5)
    assert dg.fairness([smooth])["curvature_extrema"] == 0
    assert dg.fairness([wiggly])["curvature_extrema"] >= 10
    assert dg.fairness([wiggly])["roughness"] > 100 * dg.fairness([smooth])["roughness"]


def test_junction_report():
    x = np.linspace(0, 1, 20)
    a = make_interp_spline(x, np.column_stack([x, x]), k=3)
    b = make_interp_spline(x, np.column_stack([1 + x, 1 + 2 * x]), k=3)
    (report,) = dg.junctions([a, b])
    assert report["gap"] < 1e-14
    assert report["tangent_angle"] == pytest.approx(np.arctan(2) - np.pi / 4)


def test_naca_leading_edge_curvature():
    assert NACA4("0012").leading_edge_curvature() == pytest.approx(2 / (5 * 0.12 * 0.2969) ** 2)
    cambered = NACA4("2412")
    assert cambered.leading_edge_curvature(True) == pytest.approx(cambered.leading_edge_curvature(False), rel=1e-8)


def test_naca_points_selig_order():
    pts = NACA4("0012").points(41)
    assert pts.shape == (81, 2)
    np.testing.assert_allclose(pts[40], 0.0, atol=1e-15)
    np.testing.assert_allclose(pts[0], pts[-1], atol=1e-15)
    assert pts[10, 1] > 0 > pts[70, 1]
    blunt = NACA4("0012", closed_te=False).points(41)
    assert blunt[0, 1] - blunt[-1, 1] > 1e-3
