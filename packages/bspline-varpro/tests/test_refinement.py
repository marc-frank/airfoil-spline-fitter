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
    t = np.array(result.parameters)
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


def jittered_half_circle(n=400, seed=0):
    # Jitter along the arc larger than the point spacing: neighbours want to swap
    s = np.linspace(0, np.pi, n)
    s[1:-1] += np.random.default_rng(seed).normal(0, 0.02, n - 2)
    return np.column_stack((np.cos(s), np.sin(s)))


def test_refined_parameters_stay_ordered_and_warm_start_a_fit():
    points = jittered_half_circle()
    dense = bv.fit(points, 8, max_points_per_section=50)
    assert np.all(np.diff(dense.parameters) >= 0)
    refit = bv.fit(points, 8, initial_parameters=dense.parameters)
    assert np.isfinite(refit.rms_error)


def test_refine_counts_residual_evaluations():
    points = jittered_half_circle(100)
    fitted = bv.fit(points, 8, max_nfev=3)
    iterations = []
    result = bv.refine(points, fitted, max_iterations=5, callback=lambda cp, rms: iterations.append(rms))
    # One evaluation to start, then at least a trial and a Jacobian per iteration
    assert iterations and result.nfev >= 2 * len(iterations) + 1


def test_refine_convergence_does_not_depend_on_units():
    sections = naca_sections(41)
    result = bv.fit(sections, 8, 5, continuity="G2")
    scale = 1e-7
    small = [s * scale for s in sections]
    tiny = bv.fit(small, 8, 5, continuity="G2")
    rough = bv.chain.chain_parameters([np.linspace(0, 1, len(s)) for s in sections], sections)
    for fitted, data, factor in ((result, sections, 1.0), (tiny, small, scale)):
        fitted.parameters = rough.copy()
        refined = bv.refine(data, fitted, max_iterations=30)
        # The first iteration is not mistaken for convergence on tiny data
        assert refined.nfev > 3
        assert refined.rms_error / factor < 1e-3


def test_joint_problem_without_constraints():
    points = jittered_half_circle(60)
    knots = [bv.open_uniform_knots(3, 6)]
    problem = JointProblem([points], knots, 3, np.zeros((0, 6)), np.zeros((0, 2)), smoothing=1e-3)
    assert problem.num_z == 6 * 1
    result = bv.fit(points, 6, fix_ends=False)
    refined = bv.refine(points, result)
    assert refined.rms_error <= result.rms_error * (1 + 1e-6)
