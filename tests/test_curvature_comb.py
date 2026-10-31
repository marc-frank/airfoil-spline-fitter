import numpy as np
import pytest
from bspline_varpro import airfoil, diagnostics
from scipy.interpolate import BSpline

from spline_fitter.gui.plotting import comb_parameters, comb_values
from spline_fitter.spline import BSplineCurve2D


@pytest.fixture(scope="module")
def upper_surface():
    result = airfoil.fit_airfoil(np.round(airfoil.NACA4("2412").points(81), 5))
    cp, knots = result.control_points[0], result.knots[0]
    return BSplineCurve2D(5, [tuple(p) for p in cp], list(knots)), BSpline(knots, cp, 5)


def test_comb_teeth_cluster_where_the_curve_turns(upper_surface):
    curve, reference = upper_surface
    t = comb_parameters(curve, 120)
    assert len(t) == 120 and t[0] == 0.0 and t[-1] == 1.0
    assert np.all(np.diff(t) > 0)
    near_le = lambda u: int(np.sum(reference(u)[:, 0] < 0.05))  # noqa: E731
    assert near_le(t) >= 2 * near_le(np.linspace(0, 1, 120))

    def max_turn(u):
        angle = np.unwrap(np.arctan2(*reference(u, nu=1).T[::-1]))
        return np.max(np.abs(np.diff(angle)))

    assert max_turn(t) < 0.5 * max_turn(np.linspace(0, 1, 120))


def test_comb_curvature_and_slope_are_exact(upper_surface):
    curve, reference = upper_surface
    t = comb_parameters(curve, 80)
    np.testing.assert_allclose(comb_values(curve, t, 0), diagnostics.curvature(reference, t), rtol=1e-10)
    slope = diagnostics.curvature_slope(reference, t)
    np.testing.assert_allclose(comb_values(curve, t, 1), slope, rtol=1e-9, atol=1e-9 * np.max(np.abs(slope)))


@pytest.mark.parametrize("quantity", [2, 3])
def test_higher_comb_derivatives_are_finite(upper_surface, quantity):
    curve, _ = upper_surface
    values = comb_values(curve, comb_parameters(curve, 50), quantity)
    assert values.shape == (50,) and np.all(np.isfinite(values))


def test_derivatives_leave_out_the_orders_below_the_lowest(upper_surface):
    curve, exact = upper_surface
    t = np.linspace(0, 1, 50)
    all_orders = curve.evaluate_derivatives(t, max_order=2)
    derivatives = curve.evaluate_derivatives(t, max_order=2, min_order=1)
    assert sorted(derivatives) == ["d2x", "d2y", "dx", "dy"]
    for key, values in derivatives.items():
        assert np.array_equal(values, all_orders[key])
    assert np.allclose(all_orders["d2y"], exact.derivative(2)(t)[:, 1])


@pytest.mark.parametrize("degree", [1, 2])
def test_low_degree_combs_are_finite(degree):
    curve = BSplineCurve2D(degree, [(0, 0), (0.3, 0.2), (0.7, 0.1), (1, 0)])
    t = comb_parameters(curve, 40)
    for quantity in range(4):
        assert np.all(np.isfinite(comb_values(curve, t, quantity)))


def test_derivatives_above_the_degree_are_zero():
    curve = BSplineCurve2D(2, [(0, 0), (0.3, 0.2), (0.7, 0.1), (1, 0)])
    derivatives = curve.evaluate_derivatives(np.linspace(0, 1, 9), max_order=3)
    assert not np.any(derivatives["d3x"]) and not np.any(derivatives["d3y"])
    assert np.any(derivatives["d2y"])
