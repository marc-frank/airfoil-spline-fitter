import numpy as np
import pytest

import bspline_varpro as bv
from bspline_varpro import diagnostics as dg
from bspline_varpro.airfoil import NACA4


def naca_sections(n=61, decimals=None, code="0012", noise=0.0):
    pts = NACA4(code).points(n)
    if noise:
        pts = pts + np.random.default_rng(5).normal(scale=noise, size=pts.shape)
    if decimals is not None:
        pts = np.round(pts, decimals)
    return [pts[:n], pts[n - 1:]]


@pytest.mark.parametrize("tolerance, decimals", [(1e-4, None), (1e-5, None), (2e-5, 5)])
def test_tolerance_is_met(tolerance, decimals):
    sections = naca_sections(61, decimals)
    result = bv.fit_to_tolerance(sections, tolerance, 5)
    assert result.success, result.message
    assert result.max_deviation <= tolerance
    assert dg.max_deviation(sections, result.curves()) == pytest.approx(result.max_deviation, rel=1e-6)
    assert result.continuity == ["G2"]
    assert [h["stage"] for h in result.history][0] == "insert"


def test_needs_fewer_control_points_than_uniform_knots():
    sections = naca_sections(61)
    tolerance = 1e-5
    result = bv.fit_to_tolerance(sections, tolerance, 5)
    uniform_needed = None
    for n in range(6, 20):
        fit = bv.fit(sections, n, 5, continuity="C2", max_nfev=300)
        if dg.max_deviation(sections, fit.curves()) <= tolerance:
            uniform_needed = n
            break
    assert uniform_needed is not None
    assert max(result.num_control_points) < uniform_needed


def test_unreachable_tolerance_is_reported():
    sections = naca_sections(31, noise=2e-4)
    result = bv.fit_to_tolerance(sections, 1e-6, 5, max_control_points=12)
    assert not result.success
    assert "Tolerance not reached" in result.message
    assert result.max_deviation > 1e-6
    assert max(result.num_control_points) <= 12


def test_polish_does_not_increase_the_objective():
    sections = naca_sections(41, 5)
    knots = [bv.clamped_knots(5, [0.5, 0.8]), bv.clamped_knots(5, [0.2, 0.5])]
    options = bv.ChainOptions(continuity="G2", smoothing=1e-5)
    start = bv.fit(sections, degree=5, knots=knots, continuity="G2", smoothing=1e-5, max_nfev=300)
    polished = bv.polish_knots(sections, start, options)
    assert polished.rms_error <= start.rms_error * (1 + 1e-9)
    for k in polished.knots:
        bv.validate_knots(k, 5)
        assert len(k) == len(knots[0])


def test_callback_can_stop_tolerance_fit():
    seen = []

    def stop(result, deviation):
        seen.append(deviation)
        return True

    result = bv.fit_to_tolerance(naca_sections(41), 1e-7, 5, callback=stop)
    assert len(seen) == 1
    assert not result.success and result.message == "Stopped by callback."


def test_warm_start_does_not_stall():
    # A warm start from a coarser fit used to stop early on collapsed parameter gaps.
    sections = naca_sections(81)
    coarse = bv.fit(sections, 7, 5, continuity="C2", max_nfev=300)
    cold = bv.fit(sections, 9, 5, continuity="C2", max_nfev=400)
    warm = bv.fit(sections, 9, 5, continuity="C2", max_nfev=400, initial_parameters=coarse.parameters)
    assert warm.rms_error <= cold.rms_error * 1.01


def test_rejects_invalid_tolerance():
    with pytest.raises(ValueError, match="positive"):
        bv.fit_to_tolerance(naca_sections(21), 0.0)
