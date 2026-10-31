"""The objective-based optimizers fit what the settings ask for: every
section gets the chosen number of control points, they start from the plain
least-squares fit, every junction can move, a curve with nothing left to move
is evaluated rather than handed to an optimizer, and a penalty weight means the
same for every shape, unit and degree."""
from dataclasses import fields

import numpy as np
import pytest
from bspline_varpro.airfoil import NACA4

from spline_fitter.geometry import (
    compute_arc_length_parameters,
    compute_optimization_bounds,
    create_airfoil_data,
)
from spline_fitter.optimization.control_points import (
    construct_control_points,
    generate_initial_guess,
    initial_guess_from_fit,
)
from spline_fitter.optimization.engine import run_optimization
from spline_fitter.optimization.models import (
    ContinuityMode,
    FairnessWeights,
    OptimizationConfig,
    OptimizerType,
    SplineConfig,
)
from spline_fitter.optimization.optimizers import get_optimizer
from spline_fitter.spline import BSplineCurve2D
from spline_fitter.spline.fairness import (
    compute_curve_energy,
    compute_fairness_penalty,
    compute_penalty_scales,
)

COORDS = [tuple(p) for p in np.round(NACA4("2412").points(61), 6)]


def _problem(degree, count, sections, mode):
    config = SplineConfig(degree=degree, num_control_points=count, num_sections=sections,
                          continuity=mode)
    airfoil = create_airfoil_data(COORDS, filename="n", name="n", precision=6,
                                  num_sections=sections)
    return config, airfoil


@pytest.mark.parametrize("mode", list(ContinuityMode))
def test_every_section_gets_the_chosen_count(mode):
    config, airfoil = _problem(3, 8, 3, mode)
    sections = [list(s) for s in airfoil.sections]
    t = [compute_arc_length_parameters(s) for s in sections]
    bounds = compute_optimization_bounds(airfoil, config)
    x0 = generate_initial_guess(config, sections, t, bounds)
    assert len(x0) == len(bounds)
    assert [len(c) for c in construct_control_points(x0, config, sections)] == [8, 8, 8]


@pytest.mark.parametrize("sections", [1, 2])
@pytest.mark.parametrize("optimizer", [OptimizerType.BFGS, OptimizerType.DE])
def test_a_curve_with_nothing_to_move_is_evaluated(sections, optimizer):
    config, airfoil = _problem(1, 2, sections, ContinuityMode.G0)
    bounds = compute_optimization_bounds(airfoil, config)
    assert bounds == []
    result = run_optimization(config, OptimizationConfig(optimizer=optimizer), airfoil,
                              FairnessWeights(), bounds)
    assert result.nit == 0 and 0 < result.fun < 1


def test_wiggle_is_measured_on_a_quadratic():
    curve = BSplineCurve2D(2, [(0, 0), (0.2, 0.3), (0.6, 0.1), (0.8, 0.4), (1, 0)])
    assert compute_curve_energy(curve)["curvature_gradient"] > 0


def _least_squares(config, airfoil, optimize_le=False):
    opt_config = OptimizationConfig(optimize_le_position=optimize_le)
    return get_optimizer(OptimizerType.DIRECT_LSQ).fit(config, opt_config, airfoil)


@pytest.mark.parametrize("optimize_le", [False, True])
@pytest.mark.parametrize("sections", [1, 2, 3])
@pytest.mark.parametrize("mode", list(ContinuityMode))
def test_the_least_squares_fit_is_a_starting_point(mode, sections, optimize_le):
    config, airfoil = _problem(5, 12, sections, mode)
    fit = _least_squares(config, airfoil, optimize_le).section_control_points
    bounds = compute_optimization_bounds(airfoil, config, optimize_le_position=optimize_le)
    x0 = initial_guess_from_fit(config, fit, bounds, optimize_le_position=optimize_le)
    assert np.array_equal(x0, initial_guess_from_fit(config, fit, [], optimize_le_position=optimize_le))
    rebuilt = construct_control_points(x0, config, [list(s) for s in airfoil.sections],
                                       optimize_le_position=optimize_le)
    np.testing.assert_allclose(np.concatenate(rebuilt), np.concatenate(fit), atol=1e-9)


def test_every_junction_moves_around_its_own_split_point():
    config, airfoil = _problem(3, 10, 3, ContinuityMode.G1)
    bounds = compute_optimization_bounds(airfoil, config, optimize_le_position=True)
    for (lo_x, hi_x), (lo_y, hi_y), section in zip(bounds[-4::2], bounds[-3::2], airfoil.sections):
        assert lo_x < section[-1][0] < hi_x and lo_y < section[-1][1] < hi_y


@pytest.mark.parametrize("degree", [2, 3, 5])
def test_each_penalty_at_weight_one_equals_the_least_squares_error(degree):
    config, airfoil = _problem(degree, 10, 2, ContinuityMode.G1)
    fit = _least_squares(config, airfoil)
    curves = [BSplineCurve2D(degree, pts) for pts in fit.section_control_points]
    scales = compute_penalty_scales(fit.section_control_points, curves, fit.fun)
    for f in fields(FairnessWeights):
        penalty = compute_fairness_penalty(fit.section_control_points, curves,
                                           FairnessWeights(**{f.name: 1.0}), scales)
        if penalty:
            assert penalty == pytest.approx(fit.fun)
