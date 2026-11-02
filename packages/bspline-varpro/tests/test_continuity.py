import numpy as np
import pytest

import bspline_varpro as bv
from bspline_varpro import diagnostics as dg
from bspline_varpro.airfoil import NACA4
from bspline_varpro.constraints import End, Junction, parse_junction


def naca_sections(n=41, decimals=5, code="2412"):
    pts = np.round(NACA4(code).points(n), decimals)
    return [pts[:n], pts[n - 1:]]


def three_sections():
    pts = np.round(NACA4("2412").points(41), 5)
    return [pts[:25], pts[24:41], pts[40:]]


def _fd_check(problem, t, theta):
    J_t = problem.jacobian(t, theta)
    J_p = problem.parameter_jacobian(t, theta)
    h = 1e-6
    for j in range(len(theta)):
        step = np.zeros_like(theta)
        step[j] = h
        numerical = (problem.residual(t, theta + step) - problem.residual(t, theta - step)) / (2 * h)
        np.testing.assert_allclose(J_p[:, j], numerical, rtol=3e-4, atol=1e-7)
    for j in [1, len(t) // 2 + 3, len(t) - 3]:
        step = np.zeros_like(t)
        step[j] = h
        numerical = (problem.residual(t + step, theta) - problem.residual(t - step, theta)) / (2 * h)
        np.testing.assert_allclose(J_t[:, j], numerical, rtol=3e-4, atol=1e-7)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("fixed", [True, False])
@pytest.mark.parametrize("smoothing", [0.0, 0.01])
def test_shape_parameter_jacobian_matches_finite_differences(order, fixed, smoothing):
    sections = naca_sections(21)
    problem = bv.VarProProblem(sections, bv.open_uniform_knots(5, 9), 5,
                               continuity=f"G{order}", fix_junctions=fixed, smoothing=smoothing)
    assert problem.num_params == order
    t = bv.chain.chain_parameters([bv.centripetal_parameters(s) for s in sections], sections)
    theta = problem.theta + np.array([0.3, -0.7, 1.1][:order])
    _fd_check(problem, t, theta)


def test_tangent_and_mixed_junction_jacobian():
    sections = three_sections()
    continuity = [Junction(2, geometric=True, tangent=(0.0, -1.0)), Junction(1, geometric=False, fixed=False)]
    problem = bv.VarProProblem(sections, bv.open_uniform_knots(4, 9), 4, continuity=continuity,
                               smoothing=0.01, start=End(tangent=(-1.0, 0.1)), end=End(fixed=False))
    assert problem.num_params == 2 + 2
    t = bv.chain.chain_parameters([bv.centripetal_parameters(s) for s in sections], sections)
    theta = problem.theta + np.linspace(-0.4, 0.5, problem.num_params)
    _fd_check(problem, t, theta)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_geometric_fit_is_continuous_to_the_requested_order(order):
    sections = naca_sections(41)
    result = bv.fit(sections, 10, 5, continuity=f"G{order}", max_nfev=300)
    (report,) = dg.junctions(result.curves())
    assert report["gap"] < 1e-12
    assert report["tangent_angle"] < 1e-8
    first, second = result.curves()
    if order >= 2:
        kappa = abs(dg.curvature(first, np.array([1.0]))[0])
        assert report["curvature_jump"] < 1e-7 * kappa
    if order >= 3:
        slope = abs(dg.curvature_slope(first, np.array([1.0]))[0])
        assert report["curvature_slope_jump"] < 1e-6 * max(slope, 1.0)
    alpha, beta, gamma = result.shape_parameters[0]
    assert alpha > 0
    assert result.continuity == [f"G{order}"]


def test_geometric_continuity_fits_better_than_parametric():
    sections = naca_sections(41)
    parametric = bv.fit(sections, 9, 5, continuity="C2", max_nfev=300)
    geometric = bv.fit(sections, 9, 5, continuity="G2", max_nfev=300)
    assert geometric.rms_error < parametric.rms_error


def test_parametric_join_has_unit_shape_parameters():
    result = bv.fit(naca_sections(31), 8, 3, continuity="C2", max_nfev=100)
    assert result.shape_parameters == [(1.0, 0.0, 0.0)]
    assert len(result.theta) == 0


def test_tangent_directions_are_met():
    sections = naca_sections(41, code="0012")
    down = (0.0, -1.0)
    start = (-1.0, 0.05)
    result = bv.fit(sections, 10, 5, continuity=Junction(2, tangent=down), start_tangent=start,
                    max_nfev=300)
    first, second = result.curves()
    for vector, direction in [(second(0.0, nu=1), down), (first(0.0, nu=1), start)]:
        unit = np.asarray(direction) / np.linalg.norm(direction)
        assert np.dot(vector, unit) > 0
        assert abs(vector[0] * unit[1] - vector[1] * unit[0]) < 1e-9 * np.linalg.norm(vector)
    np.testing.assert_allclose(first(1.0, nu=1) / np.linalg.norm(first(1.0, nu=1)), down, atol=1e-9)


def test_free_ends_are_not_interpolated():
    sections = naca_sections(41)
    sections[0][0] += [0.0, 0.002]
    free = bv.fit(sections, 9, 5, continuity="G2", fix_ends=False, max_nfev=300)
    assert np.linalg.norm(free.curves()[0](0.0) - sections[0][0]) > 1e-4
    fixed = bv.fit(sections, 9, 5, continuity="G2", max_nfev=300)
    np.testing.assert_allclose(fixed.curves()[0](0.0), sections[0][0], atol=1e-10)


def test_per_junction_continuity():
    result = bv.fit(three_sections(), 9, 5, continuity=["G2", "C1"], max_nfev=300)
    assert result.continuity == ["G2", "C1"]
    a, b, c = result.curves()
    np.testing.assert_allclose(b(1.0, nu=1), c(0.0, nu=1), atol=1e-8)
    report = dg.junctions([a, b, c])
    assert report[0]["curvature_jump"] < 1e-6 * abs(dg.curvature(a, np.array([1.0]))[0])


def test_warm_start_with_theta():
    sections = naca_sections(41)
    first = bv.fit(sections, 10, 5, continuity="G3", max_nfev=300)
    warm = bv.fit(sections, 10, 5, continuity="G3", max_nfev=300,
                  initial_parameters=first.parameters, initial_theta=first.theta)
    assert warm.nfev < first.nfev
    assert warm.rms_error <= first.rms_error * (1 + 1e-6)


@pytest.mark.parametrize("spec, expected", [
    (2, Junction(2, geometric=False)),
    ("g3", Junction(3, geometric=True)),
    ("C0", Junction(0, geometric=False)),
])
def test_parse_junction(spec, expected):
    assert parse_junction(spec) == expected


def test_overlapping_end_conditions_are_rejected():
    with pytest.raises(ValueError, match="needs at least"):
        bv.fit(three_sections(), 6, 3, continuity="G3")


def lopsided_sections(length=10.0):
    # Section length ratio about 20, beyond the factor 10 alpha may vary around it
    s = np.linspace(0, 1, 30)
    long = np.column_stack((length * s, 0.01 * np.sin(3 * s)))
    v = np.linspace(0, 0.5, 10)
    short = np.column_stack((length + v, 0.3 * v ** 2))
    short[0] = long[-1]
    return [long, short]


@pytest.mark.parametrize("continuity", ["G1", "G2"])
def test_geometric_fit_starts_from_the_parametric_one_at_any_length_ratio(continuity):
    sections = lopsided_sections()
    problem = bv.ChainOptions(continuity=continuity).problem(sections, bv.open_uniform_knots(3, 6), 3)
    assert problem.constraints._length_ratio(0) > 10
    theta = problem.constraints.theta_from_parametric(np.zeros(0))
    assert problem.constraints.shape_values(theta)[0] == pytest.approx((1.0, 0.0, 0.0))
    parametric = bv.fit(sections, 6, continuity="C" + continuity[1], smoothing=0.0)
    geometric = bv.fit(sections, 6, continuity=continuity, smoothing=0.0)
    # G^k includes C^k, so it fits at least as well
    assert geometric.rms_error <= parametric.rms_error * (1 + 1e-6)


def test_initial_alpha_is_the_length_ratio():
    sections = lopsided_sections()
    problem = bv.VarProProblem(sections, bv.open_uniform_knots(3, 6), 3, continuity="G1")
    alpha = problem.constraints.shape_values(problem.theta)[0][0]
    assert alpha == pytest.approx(problem.constraints._length_ratio(0))


def test_stop_during_the_parametric_stage_returns_a_geometric_result():
    calls = []

    def stop_early(control_points, rms):
        calls.append(rms)
        return len(calls) == 3

    sections = naca_sections()
    result = bv.fit(sections, 8, 5, continuity="G2", callback=stop_early)
    assert result.message == "Stopped by callback."
    assert result.continuity == ["G2"] and len(result.theta) == 2
    assert result.shape_parameters[0] == pytest.approx((1.0, 0.0, 0.0))
    # Usable as a warm start of the same fit
    warm = bv.fit(sections, 8, 5, continuity="G2", initial_theta=result.theta, max_nfev=5)
    assert np.isfinite(warm.rms_error)


def test_junction_point_column_is_the_derivative_from_the_right():
    sections = naca_sections(21)
    problem = bv.VarProProblem(sections, bv.open_uniform_knots(5, 9), 5, continuity="G1")
    t = bv.chain.chain_parameters([bv.centripetal_parameters(s) for s in sections], sections)
    theta = problem.theta + 0.5  # alpha != 1: a kink at the junction
    junction = int(np.flatnonzero(t == 1.0)[0])
    J = problem.jacobian(t, theta)
    h = 1e-7
    step = np.zeros_like(t)
    step[junction] = h
    right = (problem.residual(t + step, theta) - problem.residual(t, theta)) / h
    left = (problem.residual(t, theta) - problem.residual(t - step, theta)) / h
    np.testing.assert_allclose(J[:, junction], right, rtol=1e-4, atol=1e-6)
    assert not np.allclose(J[:, junction], left, rtol=1e-3, atol=1e-6)
