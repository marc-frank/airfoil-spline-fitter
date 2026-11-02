import threading

import numpy as np
import pytest
from scipy.interpolate import BSpline

import bspline_varpro as bv


def two_sections(noise=0.0, n=19):
    t = np.linspace(0, np.pi, n)
    upper = np.column_stack(((1 + np.cos(t)) / 2, 0.09 * np.sin(t)))
    lower = np.column_stack(((1 - np.cos(t)) / 2, -0.06 * np.sin(t)))
    rng = np.random.default_rng(82)
    for sec in (upper, lower):
        sec[1:-1, 1] += rng.normal(0, noise, n - 2)
    return [upper, lower]


@pytest.mark.parametrize("smoothing", [0.0, 0.03])
@pytest.mark.parametrize("continuity", [0, 1, 2, 3])
@pytest.mark.parametrize("fix_junctions", [True, False])
def test_reduced_jacobian_matches_finite_differences(smoothing, continuity, fix_junctions):
    sections = two_sections()
    problem = bv.VarProProblem(sections, bv.open_uniform_knots(3, 6), 3, continuity=continuity,
                               fix_junctions=fix_junctions, smoothing=smoothing)
    t = bv.chain.chain_parameters([np.linspace(0, 1, 19)] * 2, sections)
    jac = problem.jacobian(t)
    for j in [2, 12, 23, 35]:
        step = np.zeros_like(t)
        step[j] = 1e-6
        numerical = (problem.residual(t + step) - problem.residual(t - step)) / 2e-6
        np.testing.assert_allclose(jac[:, j], numerical, rtol=3e-4, atol=1e-7)


def test_residual_norm_equals_penalized_objective_and_constraints_hold():
    sections = two_sections()
    problem = bv.VarProProblem(sections, bv.open_uniform_knots(3, 6), 3, continuity=2,
                               smoothing=0.03)
    t = bv.chain.chain_parameters([np.linspace(0, 1, 19)] * 2, sections)
    r = problem.residual(t)
    N, P, _, _ = problem.solve(t)
    energy = (np.sum((N @ P - problem.data) ** 2) + problem.regularization * np.sum(P ** 2)
              + 0.03 * np.sum(P * (problem.bending_energy @ P)))
    assert r @ r == pytest.approx(energy, rel=1e-8)
    np.testing.assert_allclose(problem.C @ P, problem.d, atol=1e-8)


def test_three_dimensional_jacobian():
    s = np.linspace(0, 1, 15)
    helix = np.column_stack((np.cos(4 * s), np.sin(4 * s), s))
    sections = [helix[:8], helix[7:]]
    problem = bv.VarProProblem(sections, bv.open_uniform_knots(3, 6), 3, continuity=2, smoothing=0.01)
    t = bv.chain.chain_parameters([np.linspace(0, 1, 8), np.linspace(0, 1, 8)], sections)
    jac = problem.jacobian(t)
    # The shared junction point is one data point; the penalty has a row per
    # Gauss node (8 on each of the 3 spans of both sections) and per control point
    assert jac.shape == (3 * (15 + 2 * 3 * 8 + 12), 15)
    for j in [1, 6, 10]:
        step = np.zeros_like(t)
        step[j] = 1e-6
        numerical = (problem.residual(t + step) - problem.residual(t - step)) / 2e-6
        np.testing.assert_allclose(jac[:, j], numerical, rtol=3e-4, atol=1e-7)


@pytest.mark.parametrize("continuity", [0, 1, 2, 3])
def test_fit_enforces_endpoints_and_junction_continuity(continuity):
    sections = two_sections(0.002)
    result = bv.fit(sections, 8, 5, continuity=continuity, max_nfev=300)
    first, second = result.curves()
    np.testing.assert_allclose(first(0.0), sections[0][0], atol=1e-9)
    np.testing.assert_allclose(second(1.0), sections[1][-1], atol=1e-9)
    np.testing.assert_allclose(first(1.0), sections[0][-1], atol=1e-9)
    for order in range(continuity + 1):
        np.testing.assert_allclose(first(1.0, nu=order), second(0.0, nu=order),
                                   atol=1e-7 * max(1.0, np.abs(first(1.0, nu=order)).max()))


def test_free_junction_moves_off_the_data_point():
    sections = two_sections(0.002)
    sections[0][-1] += [0.0, 0.01]
    sections[1][0] += [0.0, 0.01]
    result = bv.fit(sections, 8, 3, fix_junctions=False, max_nfev=300)
    first, second = result.curves()
    np.testing.assert_allclose(first(1.0), second(0.0), atol=1e-9)
    assert np.linalg.norm(first(1.0) - sections[0][-1]) > 1e-4


def test_recovers_a_known_curve_from_a_nearby_parameterization():
    knots = bv.open_uniform_knots(3, 7)
    control = np.array([[0, 0], [1, 1], [2, 1.5], [3, 1.6], [4, 1.2], [5, 0.5], [6, 0.]])
    truth = BSpline(knots, control, 3)
    points = truth(np.linspace(0, 1, 60) ** 1.3)
    result = bv.fit(points, 7, 3, smoothing=0.0, max_nfev=500,
                    initial_parameters=[np.linspace(0, 1, 60) ** 1.2])
    assert len(result.control_points) == 1 and result.control_points[0].shape == (7, 2)
    assert result.success
    assert result.rms_error < 1e-8


def test_reported_error_is_data_only():
    sections = two_sections(0.003)
    result = bv.fit(sections, 6, 3, smoothing=0.1, max_nfev=500)
    data = bv.chain.chain_data(sections)
    section, local = bv.chain.locate(result.parameters, len(sections))
    curves = result.curves()
    on_chain = np.array([curves[s](t) for s, t in zip(section, local)])
    assert result.rms_error == pytest.approx(np.sqrt(np.mean(np.sum((on_chain - data) ** 2, axis=1))))
    u = result.parameters
    assert np.all(np.diff(u) >= 0) and u[0] == 0 and u[-1] == pytest.approx(len(sections))


def test_callback_can_stop_the_fit():
    calls = []

    def stop_after_three(control_points, rms):
        calls.append(([cp.shape for cp in control_points], rms))
        return len(calls) == 3

    result = bv.fit(two_sections(0.002), 8, 3, max_nfev=300, callback=stop_after_three)
    assert len(calls) == 3 and result.nfev == 3
    assert calls[0][0] == [(8, 2), (8, 2)]
    assert not result.success and result.message == "Stopped by callback."
    assert np.isfinite(result.rms_error)


def test_callback_follows_knot_polishing():
    knots_seen = []

    def watch(control_points, rms, knots):
        knots_seen.append(np.concatenate(knots))

    bv.fit(two_sections(0.002), 8, 3, max_nfev=300, knot_placement="free", callback=watch)
    # The knots move while they are polished
    assert len({k.tobytes() for k in knots_seen}) > 1


def test_finite_difference_jacobian_reaches_same_cost():
    sections = two_sections(0.002)
    exact = bv.fit(sections, 8, 3, max_nfev=300)
    approximate = bv.fit(sections, 8, 3, max_nfev=3000, exact_jacobian=False)
    assert approximate.rms_error == pytest.approx(exact.rms_error, rel=1e-3)


def test_initial_parameters_are_used():
    sections = two_sections(0.002)
    first = bv.fit(sections, 8, 3, max_nfev=300)
    warm = bv.fit(sections, 8, 3, max_nfev=300, initial_parameters=first.parameters,
                  initial_theta=first.theta)
    assert warm.nfev < first.nfev
    assert warm.rms_error <= first.rms_error * (1 + 1e-6)


@pytest.mark.parametrize("kwargs, message", [
    (dict(num_control_points=3, degree=3), "degree \\+ 1"),
    (dict(num_control_points=8, degree=2, continuity=3), "continuity order exceeds"),
    (dict(num_control_points=8, continuity="X2"), "Invalid continuity"),
    (dict(num_control_points=[8, 8, 8]), "one control-point count per section"),
    (dict(num_control_points=8, smoothing=-1.0), "Smoothing"),
    (dict(num_control_points=8, initial_parameters=[np.linspace(0, 1, 5)] * 2), "Parameters"),
])
def test_invalid_input_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bv.fit(two_sections(), **kwargs)


def test_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least two"):
        bv.fit([np.zeros((1, 2))], 4)


def test_end_derivative_coefficients_match_basis_derivatives():
    knots = bv.open_uniform_knots(5, 12)
    points = np.random.default_rng(0).normal(size=(12, 2))
    curve = BSpline(knots, points, 5)
    for order in (1, 2, 3):
        coef_end, coef_start = bv.end_derivative_coefficients(knots, 5, order)
        np.testing.assert_allclose(coef_start @ points[:order + 1], curve(0.0, nu=order))
        np.testing.assert_allclose(coef_end @ points[-order - 1:], curve(1.0, nu=order))


def test_levenberg_marquardt_stops_on_a_non_finite_jacobian():
    from bspline_varpro.fitting import levenberg_marquardt

    outcome = []

    def run():
        outcome.append(levenberg_marquardt(
            lambda x: np.array([x[0] - 1.0, x[1] + 2.0]),
            lambda x: np.array([[1.0, np.nan], [0.0, 1.0]]),
            np.zeros(2), max_nfev=50,
        ))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert outcome, "levenberg_marquardt did not return"
    assert not outcome[0].success
    assert "not finite" in outcome[0].message
