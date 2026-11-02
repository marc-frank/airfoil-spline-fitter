import numpy as np
import pytest

import bspline_varpro as bv
from bspline_varpro import diagnostics as dg
from bspline_varpro.airfoil import NACA4
from bspline_varpro.fitting import levenberg_marquardt
from bspline_varpro.tolerance import subsample_indices


def dense_sections(n=601):
    pts = NACA4("0012").points(n)
    return [pts[:n], pts[n - 1:]]


def test_subsample_indices_keep_both_ends():
    idx = subsample_indices(1000, 37)
    assert idx[0] == 0 and idx[-1] == 999 and len(idx) == 37
    np.testing.assert_array_equal(subsample_indices(10, 37), np.arange(10))


def test_dense_tolerance_fit_checks_every_point():
    sections = dense_sections()
    result = bv.fit_to_tolerance(sections, 1e-5, 5, max_points_per_section=80)
    assert result.success
    assert len(result.parameters) == len(bv.chain.chain_data(sections))
    assert dg.max_deviation(sections, result.curves()) <= 1e-5 * (1 + 1e-9)


def test_fit_with_subsampling_returns_all_parameters():
    sections = dense_sections(401)
    result = bv.fit(sections, 10, 5, continuity="G2", max_points_per_section=60)
    assert len(result.parameters) == len(bv.chain.chain_data(sections))
    full = bv.fit(sections, 10, 5, continuity="G2")
    assert result.rms_error == pytest.approx(full.rms_error, rel=0.1)
    assert np.all(np.diff(result.parameters) >= 0)


def test_subsampled_fit_reports_progress_and_can_be_stopped():
    sections = dense_sections(401)
    seen = []

    def stop_after_three(control_points, rms_error):
        seen.append(rms_error)
        return len(seen) >= 3

    result = bv.fit(sections, 10, 5, continuity="G2", max_points_per_section=60, callback=stop_after_three)
    assert len(seen) == 3
    assert not result.success and result.message == "Stopped by callback."
    assert len(result.parameters) == len(bv.chain.chain_data(sections))


def test_refinement_reports_every_iteration_and_can_be_stopped():
    sections = dense_sections(401)
    # A subsample fit with uniform parameters on all points: far from converged
    coarse = bv.fit([s[::20] for s in sections], 10, 5, continuity="G2")
    coarse.parameters = bv.chain.chain_parameters([np.linspace(0, 1, len(s)) for s in sections],
                                                  sections)
    seen = []
    refined = bv.refine(sections, coarse, max_iterations=20, callback=lambda cp, rms: seen.append(rms))
    # One report per iteration; nfev counts residual evaluations (rejected trials too)
    assert len(seen) == 20 and refined.nfev >= 2 * 20 + 1
    assert seen[-1] < seen[0]
    stopped = bv.refine(sections, coarse, callback=lambda cp, rms: True)
    assert stopped.message == "Stopped by callback." and stopped.nfev >= 3


def test_levenberg_marquardt_solves_rosenbrock():
    def residual(x):
        return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])

    def jacobian(x):
        return np.array([[-20 * x[0], 10.0], [-1.0, 0.0]])

    for jac in (jacobian, "2-point"):
        solution = levenberg_marquardt(residual, jac, np.array([-1.2, 1.0]), max_nfev=500)
        assert solution.success
        np.testing.assert_allclose(solution.x, [1.0, 1.0], atol=1e-6)
