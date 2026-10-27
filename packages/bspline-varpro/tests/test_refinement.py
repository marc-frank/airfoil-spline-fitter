import numpy as np

import bspline_varpro as bv
from bspline_varpro import diagnostics as dg
from bspline_varpro.airfoil import NACA4
from bspline_varpro.joint import JointProblem


def naca_sections(n=41, decimals=None):
    pts = NACA4("0012").points(n)
    if decimals is not None:
        pts = np.round(pts, decimals)
    return [pts[:n], pts[n - 1:]]


def _joint(result, sections):
    problem = bv.ChainOptions(continuity="G2").problem(sections, result.knots, result.degree)
    C, d, _, _ = problem.constraints.build(result.theta)
    return JointProblem(sections, result.knots, result.degree, C, d, smoothing=1e-4)


def test_joint_jacobian_matches_finite_differences():
    sections = naca_sections(21)
    result = bv.fit(sections, 8, 5, continuity="G2", max_nfev=200)
    problem = _joint(result, sections)
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


def test_refine_supports_3d():
    s = np.linspace(0, 1, 40)
    helix = np.column_stack([np.cos(3 * s), np.sin(3 * s), s])
    result = bv.fit(helix, 8, 5, max_nfev=100)
    refined = bv.refine(helix, result)
    assert refined.rms_error <= result.rms_error * (1 + 1e-6)
