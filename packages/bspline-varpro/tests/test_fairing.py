import numpy as np
import pytest

import bspline_varpro as bv
from bspline_varpro import diagnostics as dg
from bspline_varpro.airfoil import NACA4
from bspline_varpro.joint import JointProblem, fairness_terms, solve_joint


def naca_sections(n=41, decimals=None):
    pts = NACA4("0012").points(n)
    if decimals is not None:
        pts = np.round(pts, decimals)
    return [pts[:n], pts[n - 1:]]


def test_fairness_partials_match_finite_differences():
    rng = np.random.default_rng(0)
    d = [rng.normal(size=(6, 2)) + [1.0, 0.0] for _ in range(3)]
    _, df = fairness_terms(*d)
    h = 1e-6
    for j in range(6):
        plus = [x.copy() for x in d]
        minus = [x.copy() for x in d]
        plus[j // 2][:, j % 2] += h
        minus[j // 2][:, j % 2] -= h
        numerical = (fairness_terms(*plus)[0] - fairness_terms(*minus)[0]) / (2 * h)
        np.testing.assert_allclose(df[:, j], numerical, rtol=1e-6, atol=1e-8)


def test_fairness_term_is_curvature_slope():
    result = bv.fit(naca_sections(), 9, 5, continuity="G2", max_nfev=300)
    curve = result.curves()[0]
    u = np.linspace(0.1, 0.9, 7)
    f, _ = fairness_terms(curve(u, nu=1), curve(u, nu=2), curve(u, nu=3))
    speed = np.linalg.norm(curve(u, nu=1), axis=1)
    np.testing.assert_allclose(f / np.sqrt(speed), dg.curvature_slope(curve, u), rtol=1e-10)


def _joint(result, sections, weight):
    problem = bv.ChainOptions(continuity="G2").problem(sections, result.knots, result.degree)
    C, d, _, _ = problem.constraints.build(result.theta)
    return JointProblem(sections, result.knots, result.degree, C, d, smoothing=1e-4, fairness=weight)


def test_joint_jacobian_matches_finite_differences():
    sections = naca_sections(21)
    result = bv.fit(sections, 8, 5, continuity="G2", max_nfev=200)
    problem = _joint(result, sections, 1e-6)
    t = np.concatenate(result.parameters)
    z = problem.z_from(np.vstack(result.control_points)) + 1e-3
    r, (tangent, Jz) = problem.evaluate(t, z, with_jacobian=True)
    h = 1e-7
    for j in [0, 5, len(z) - 1]:
        step = np.zeros_like(z)
        step[j] = h
        numerical = (problem.evaluate(t, z + step)[0] - problem.evaluate(t, z - step)[0]) / (2 * h)
        np.testing.assert_allclose(Jz[:, j], numerical, rtol=1e-5, atol=1e-7)
    i = problem.free[4]
    step = np.zeros_like(t)
    step[i] = h
    numerical = (problem.evaluate(t + step, z)[0] - problem.evaluate(t - step, z)[0]) / (2 * h)
    np.testing.assert_allclose(numerical[2 * i:2 * i + 2], tangent[i], rtol=1e-6)
    assert np.allclose(np.delete(numerical, [2 * i, 2 * i + 1]), 0.0, atol=1e-9)


def test_refine_keeps_constraints_and_does_not_increase_error():
    sections = naca_sections(41, 5)
    result = bv.fit(sections, 9, 5, continuity="G2", max_nfev=300)
    refined = bv.refine(sections, result)
    assert refined.rms_error <= result.rms_error * (1 + 1e-6)
    first, second = refined.curves()
    np.testing.assert_allclose(first(0.0), sections[0][0], atol=1e-12)
    np.testing.assert_allclose(second(1.0), sections[1][-1], atol=1e-12)
    (junction,) = dg.junctions(refined.curves())
    assert junction["tangent_angle"] < 1e-9
    assert junction["curvature_jump"] < 1e-8 * abs(dg.curvature(first, np.array([1.0]))[0])


def test_fairing_within_tolerance_removes_end_hooks():
    sections = naca_sections(81)
    hooked = bv.fit(sections, 9, 5, continuity="G2", smoothing=0.0, max_nfev=300)
    start = dg.fairness(hooked.curves())
    deviation = dg.max_deviation(sections, hooked.curves())
    faired = bv.fair(sections, hooked, tolerance=3 * deviation)
    assert faired.success
    assert faired.max_deviation <= 3 * deviation
    after = dg.fairness(faired.curves())
    assert after["roughness"] < 1e-3 * start["roughness"]
    assert after["max_curvature"] == pytest.approx(NACA4("0012").leading_edge_curvature(), rel=0.02)
    assert [h["weight"] for h in faired.history][0] == 0.0


def test_larger_weight_gives_smoother_curve():
    sections = naca_sections(41, 4)
    result = bv.fit(sections, 10, 5, continuity="G2", max_nfev=300)
    light = bv.fair(sections, result, weight=1e-12)
    heavy = bv.fair(sections, result, weight=1e-9)
    assert dg.fairness(heavy.curves())["roughness"] < dg.fairness(light.curves())["roughness"]
    assert heavy.rms_error > light.rms_error


def test_tolerance_fit_with_fairing():
    sections = naca_sections(41, 5)
    plain = bv.fit_to_tolerance(sections, 5e-5, 5, fairing=False)
    faired = bv.fit_to_tolerance(sections, 5e-5, 5)
    assert faired.success and faired.max_deviation <= 5e-5
    assert faired.num_control_points == plain.num_control_points
    assert dg.fairness(faired.curves())["roughness"] <= dg.fairness(plain.curves())["roughness"]
    assert faired.history[-1]["stage"] == "fair"


def test_fair_argument_validation():
    sections = naca_sections(21)
    result = bv.fit(sections, 8, 5, max_nfev=50)
    with pytest.raises(ValueError, match="exactly one"):
        bv.fair(sections, result)
    with pytest.raises(ValueError, match="exactly one"):
        bv.fair(sections, result, tolerance=1e-4, weight=1.0)
    s = np.linspace(0, 1, 30)
    helix = np.column_stack([np.cos(3 * s), np.sin(3 * s), s])
    fit3d = bv.fit(helix, 8, 5, max_nfev=50)
    with pytest.raises(ValueError, match="planar"):
        bv.fair(helix, fit3d, weight=1.0)


def test_solve_joint_without_fairing_supports_3d():
    s = np.linspace(0, 1, 40)
    helix = np.column_stack([np.cos(3 * s), np.sin(3 * s), s])
    result = bv.fit(helix, 8, 5, max_nfev=100)
    refined = bv.refine(helix, result)
    assert refined.rms_error <= result.rms_error * (1 + 1e-6)
