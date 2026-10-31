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


def test_step_callback_sees_every_refit_step_with_its_knots():
    steps, refits = [], []

    def step(control_points, rms, knots):
        assert [len(c) for c in control_points] == [len(k) - 6 for k in knots]
        steps.append(rms)

    result = bv.fit_to_tolerance(naca_sections(41), 1e-5, 5, step_callback=step,
                                 callback=lambda r, d: refits.append(d))
    assert result.success
    assert len(steps) > 5 * len(refits)


@pytest.mark.parametrize("placement", ["free", "uniform"])
def test_step_callback_can_stop_tolerance_fit(placement):
    calls = []

    def stop_after(control_points, rms):
        calls.append(rms)
        return len(calls) == 20

    result = bv.fit_to_tolerance(naca_sections(41), 1e-7, 5, step_callback=stop_after,
                                 knot_placement=placement)
    assert len(calls) == 20
    assert not result.success and result.message == "Stopped by callback."


def test_warm_start_does_not_stall():
    # A warm start from a coarser fit used to stop early on collapsed parameter gaps.
    sections = naca_sections(81)
    coarse = bv.fit(sections, 7, 5, continuity="C2", max_nfev=300)
    cold = bv.fit(sections, 9, 5, continuity="C2", max_nfev=400)
    warm = bv.fit(sections, 9, 5, continuity="C2", max_nfev=400, initial_parameters=coarse.parameters)
    assert warm.rms_error <= cold.rms_error * 1.01


def test_no_remaining_knot_can_be_removed():
    # NACA 2412 at 5 decimals used to keep removable knots: removal gave up after
    # two failures in a row and only polished the knots afterwards.
    pts = np.round(NACA4("2412").points(81), 5)
    sections = [pts[:81], pts[80:]]
    tolerance = 1e-5
    result = bv.fit_to_tolerance(sections, tolerance, 5, smoothing=1e-5)
    assert result.success
    assert sum(result.num_control_points) <= 21
    for s in range(len(sections)):
        inner = bv.interior_knots(result.knots[s], 5)
        for i in range(len(inner)):
            knots = list(result.knots)
            knots[s] = bv.clamped_knots(5, np.delete(inner, i))
            trial = bv.fit(sections, degree=5, knots=knots, continuity="G2", smoothing=1e-5,
                           initial_parameters=result.parameters, max_nfev=200)
            assert dg.max_deviation(sections, trial.curves()) > tolerance


def test_rejects_invalid_tolerance():
    with pytest.raises(ValueError, match="positive"):
        bv.fit_to_tolerance(naca_sections(21), 0.0)


def test_uniform_knot_placement_meets_tolerance_with_even_knots():
    sections = naca_sections(61)
    result = bv.fit_to_tolerance(sections, 1e-5, 5, knot_placement="uniform")
    assert result.success and result.max_deviation <= 1e-5
    for knots, n in zip(result.knots, result.num_control_points):
        np.testing.assert_allclose(knots, bv.open_uniform_knots(5, n))
    for s, n in enumerate(result.num_control_points):
        if n > 6:  # one fewer uniform control point must miss the tolerance
            knots = list(result.knots)
            knots[s] = bv.open_uniform_knots(5, n - 1)
            fewer = bv.fit(sections, degree=5, knots=knots, continuity="G2", smoothing=1e-7,
                           initial_parameters=result.parameters, max_nfev=200)
            assert dg.max_deviation(sections, fewer.curves()) > 1e-5


def test_fixed_count_placement_and_growth():
    sections = naca_sections(61, code="2412")

    def deviation(result):
        return dg.max_deviation(sections, result.curves())

    uniform = bv.fit(sections, 10, 5, continuity="G2", max_nfev=300)
    free = bv.fit(sections, 10, 5, continuity="G2", max_nfev=300, knot_placement="free")
    grown_free = bv.fit(sections, [10, 9], 5, continuity="G2", max_nfev=300,
                        knot_placement="free", grow=True)
    grown_uniform = bv.fit(sections, [10, 9], 5, continuity="G2", max_nfev=300, grow=True)
    assert free.num_control_points == [10, 10]
    assert grown_free.num_control_points == grown_uniform.num_control_points == [10, 9]
    assert deviation(free) < 0.5 * deviation(uniform)  # the knots really move
    assert deviation(grown_free) < deviation(uniform)
    for knots, n in zip(grown_uniform.knots, [10, 9]):
        np.testing.assert_allclose(knots, bv.open_uniform_knots(5, n))
    for k in free.knots + grown_free.knots:
        bv.validate_knots(k, 5)


@pytest.mark.parametrize("count", [28, 30])
def test_growth_reaches_dense_counts(count):
    # Near interpolation the worst points sit alone in their knot spans
    sections = naca_sections(31, code="2412")
    result = bv.fit(sections, count, 3, continuity="G1", max_nfev=100,
                    knot_placement="free", grow=True)
    assert result.num_control_points == [count, count]


def test_tolerance_fit_can_start_from_a_control_point_count():
    sections = naca_sections(61)
    result = bv.fit_to_tolerance(sections, 1e-4, 5, initial_control_points=9, remove_knots=False)
    assert result.success
    assert min(result.num_control_points) >= 9
    assert result.history[0]["control_points"] == [9, 9]


def test_knot_placement_is_validated():
    sections = naca_sections(21)
    with pytest.raises(ValueError, match="knot_placement"):
        bv.fit_to_tolerance(sections, 1e-4, knot_placement="adaptive")
    with pytest.raises(ValueError, match="knot_placement"):
        bv.fit(sections, 8, 5, knot_placement="adaptive")
    with pytest.raises(ValueError, match="num_control_points"):
        bv.fit(sections, degree=5, knots=bv.open_uniform_knots(5, 8), grow=True)
