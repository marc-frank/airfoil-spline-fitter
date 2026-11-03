import numpy as np
import pytest
from scipy.interpolate import BSpline

import bspline_varpro as bv
from bspline_varpro.airfoil import NACA4


def naca_sections(n=61, decimals=None):
    pts = NACA4("0012").points(n)
    if decimals is not None:
        pts = np.round(pts, decimals)
    return [pts[:n], pts[n - 1:]]


def test_clamped_and_interior_knots_round_trip():
    knots = bv.clamped_knots(3, [0.7, 0.2])
    np.testing.assert_array_equal(knots, [0, 0, 0, 0, 0.2, 0.7, 1, 1, 1, 1])
    np.testing.assert_array_equal(bv.interior_knots(knots, 3), [0.2, 0.7])
    np.testing.assert_array_equal(bv.clamped_knots(3), bv.open_uniform_knots(3, 4))


@pytest.mark.parametrize("knots, message", [
    ([0, 0, 0, 1, 1, 1], "at least"),
    ([0, 0, 0, 0, 0.5, 0.4, 1, 1, 1, 1], "nondecreasing"),
    ([0, 0, 0, 0.1, 0.5, 1, 1, 1, 1], "clamped"),
    ([0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1], "multiplicity"),
])
def test_validate_knots_rejects_bad_vectors(knots, message):
    with pytest.raises(ValueError, match=message):
        bv.validate_knots(knots, 3)


def test_averaged_knots_put_data_in_every_span():
    t = np.linspace(0, 1, 50) ** 3
    knots = bv.averaged_knots(t, 12, 3)
    bv.validate_knots(knots, 3)
    breaks = np.unique(knots)
    counts = np.histogram(t, bins=breaks)[0]
    assert np.all(counts > 0)


def test_insert_knot_preserves_the_curve():
    rng = np.random.default_rng(1)
    knots = bv.clamped_knots(5, [0.3, 0.55])
    P = rng.normal(size=(8, 2))
    new_knots, new_P = bv.insert_knot(knots, P, 5, 0.42)
    assert len(new_knots) == len(knots) + 1 and len(new_P) == 9
    u = np.linspace(0, 1, 101)
    np.testing.assert_allclose(BSpline(new_knots, new_P, 5)(u), BSpline(knots, P, 5)(u), atol=1e-13)
    twice_knots, twice_P = bv.insert_knot(new_knots, new_P, 5, 0.42)
    np.testing.assert_allclose(BSpline(twice_knots, twice_P, 5)(u), BSpline(knots, P, 5)(u), atol=1e-13)


def test_explicit_uniform_knots_match_control_point_count():
    sections = naca_sections(41, 5)
    by_count = bv.fit(sections, 9, 5, continuity=2, max_nfev=100)
    by_knots = bv.fit(sections, degree=5, knots=bv.open_uniform_knots(5, 9), continuity=2, max_nfev=100)
    assert by_knots.rms_error == pytest.approx(by_count.rms_error, rel=1e-9)
    for a, b in zip(by_count.control_points, by_knots.control_points):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_sections_can_have_different_knot_vectors():
    sections = naca_sections(41, 5)
    knots = [bv.clamped_knots(5, [0.5, 0.8, 0.9]), bv.clamped_knots(5, [0.05, 0.15, 0.4, 0.7])]
    result = bv.fit(sections, degree=5, knots=knots, continuity=2, max_nfev=200)
    assert result.num_control_points == [9, 10]
    first, second = result.curves()
    for order in range(3):
        a, b = first(1.0, nu=order), second(0.0, nu=order)
        np.testing.assert_allclose(a, b, atol=1e-8 * max(1.0, np.abs(a).max()))


@pytest.mark.parametrize("smoothing", [0.0, 0.01])
def test_nonuniform_jacobian_matches_finite_differences(smoothing):
    sections = naca_sections(21)
    knots = [bv.clamped_knots(3, [0.6, 0.85, 0.95]), bv.clamped_knots(3, [0.1, 0.3])]
    problem = bv.VarProProblem(sections, knots, 3, continuity=2, smoothing=smoothing)
    t = bv.chain.chain_parameters([bv.centripetal_parameters(s) for s in sections], sections)
    jac = problem.jacobian(t)
    for j in [3, 15, 25, 38]:
        step = np.zeros_like(t)
        step[j] = 1e-6
        numerical = (problem.residual(t + step) - problem.residual(t - step)) / 2e-6
        np.testing.assert_allclose(jac[:, j], numerical, rtol=3e-4, atol=1e-7)


def test_knots_clustered_at_the_leading_edge_beat_uniform_knots():
    sections = naca_sections(81)
    n = 10
    uniform = bv.fit(sections, n, 5, continuity=2, max_nfev=300)
    upper_interior = 1 - (1 - np.linspace(0, 1, n - 4)[1:-1]) ** 2
    clustered = [bv.clamped_knots(5, np.sort(upper_interior)),
                 bv.clamped_knots(5, np.sort(1 - upper_interior))]
    nonuniform = bv.fit(sections, degree=5, knots=clustered, continuity=2, max_nfev=300)
    assert nonuniform.max_residual < 0.7 * uniform.max_residual


@pytest.mark.parametrize("u", [0.0, 1.0, -0.1, 1.5])
def test_insert_knot_rejects_values_outside_the_domain(u):
    with pytest.raises(ValueError, match="strictly inside"):
        bv.insert_knot(bv.open_uniform_knots(3, 5), np.zeros((5, 2)), 3, u)


def test_insert_knot_rejects_excess_multiplicity():
    knots, P = bv.open_uniform_knots(3, 5), np.zeros((5, 2))
    for _ in range(2):
        knots, P = bv.insert_knot(knots, P, 3, 0.5)
    with pytest.raises(ValueError, match="multiplicity"):
        bv.insert_knot(knots, P, 3, 0.5)


@pytest.mark.parametrize("t", [
    np.array([0.0] + [0.2] * 10 + [1.0]),
    np.array([0.0] + [0.2] * 8 + [0.3] * 8 + [1.0]),
    np.array([0.0] * 8 + [0.5, 1.0]),
])
def test_averaged_knots_of_repeated_parameters_are_valid(t):
    knots = bv.averaged_knots(t, 12, 3)
    bv.validate_knots(knots, 3)
    assert len(knots) == 12 + 3 + 1
