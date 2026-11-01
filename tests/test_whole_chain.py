"""The chain of sections is fitted to the data as a whole: the split only sets
where the junctions start, and points are not assigned to sections."""
import numpy as np
import pytest
from bspline_varpro.airfoil import NACA4

from spline_fitter.geometry import (
    compute_arc_length_parameters,
    compute_optimization_bounds,
    create_airfoil_data,
)
from spline_fitter.optimization.models import (
    ContinuityMode,
    ErrorMetric,
    OptimizationConfig,
    SplineConfig,
)
from spline_fitter.optimization.objective_functions import ChainPoints, compute_data_error
from spline_fitter.optimization.optimizers.constrained_lsq import ConstrainedLSQOptimizer
from spline_fitter.spline.spline import BSplineCurve2D


def _airfoil(sections=2, n=81):
    coords = [tuple(p) for p in np.round(NACA4("2412").points(n), 6)]
    return create_airfoil_data(coords, num_sections=sections)


def _chain(airfoil):
    sections = [list(s) for s in airfoil.sections]
    return ChainPoints.from_sections(sections, [compute_arc_length_parameters(s) for s in sections])


def test_a_shared_split_point_counts_once():
    airfoil = _airfoil()
    assert len(_chain(airfoil).data) == len(airfoil.coordinates)


def test_orthogonal_error_measures_the_whole_chain():
    # Data along a line split at x = 0.5; the curve's junction sits at x = 1
    xs = np.linspace(0, 2, 41)
    sections = [[(x, 0.0) for x in xs[:11]], [(x, 0.0) for x in xs[10:]]]
    chain = ChainPoints.from_sections(sections, [compute_arc_length_parameters(s) for s in sections])
    curves = [BSplineCurve2D(1, [(0.0, 0.0), (1.0, 0.0)]), BSplineCurve2D(1, [(1.0, 0.0), (2.0, 0.0)])]
    assert compute_data_error(curves, chain, ErrorMetric.ORTHOGONAL) == pytest.approx(0, abs=1e-12)
    # Pointwise keeps each point at its place in its starting section
    assert compute_data_error(curves, chain, ErrorMetric.PARAMETRIC) > 0.1


@pytest.mark.parametrize("count", [10, 40])
def test_closest_point_is_never_farther_than_the_own_parameter(count):
    airfoil = _airfoil()
    chain = _chain(airfoil)
    config = SplineConfig(degree=3, num_control_points=count, continuity=ContinuityMode.G2)
    fit = ConstrainedLSQOptimizer().fit(config, OptimizationConfig(custom_parameters=chain.parameters),
                                        airfoil)
    curves = [BSplineCurve2D(3, c) for c in fit.section_control_points]
    parametric = compute_data_error(curves, chain, ErrorMetric.PARAMETRIC)
    for metric in (ErrorMetric.ORTHOGONAL, ErrorMetric.ORTHOGONAL_LOCAL):
        assert compute_data_error(curves, chain, metric) <= parametric * (1 + 1e-12)


@pytest.mark.parametrize("sections", [2, 3])
@pytest.mark.parametrize("mode", [ContinuityMode.G0, ContinuityMode.G2])
def test_freeing_the_junctions_never_hurts_direct_lsq(sections, mode):
    airfoil = _airfoil(sections)
    config = SplineConfig(degree=3, num_control_points=10, num_sections=sections, continuity=mode)

    def rms(free):
        result = ConstrainedLSQOptimizer().fit(config, OptimizationConfig(optimize_le_position=free),
                                               airfoil)
        curves = [BSplineCurve2D(3, c) for c in result.section_control_points]
        chain = ChainPoints(_chain(airfoil).data, result.parameters, sections)
        section = np.minimum(np.floor(chain.parameters).astype(int), sections - 1)
        on_chain = np.array([curves[s].evaluate(u - s) for s, u in zip(section, chain.parameters)])
        return np.sqrt(np.mean(np.sum((on_chain - chain.data) ** 2, axis=1))), result

    pinned, pinned_result = rms(False)
    free, free_result = rms(True)
    assert free <= pinned * (1 + 1e-6)
    # Pinned junctions stay at their split points
    for s, split in enumerate(airfoil.junction_indices):
        assert pinned_result.section_control_points[s][-1] == pytest.approx(airfoil.coordinates[split])


@pytest.mark.parametrize("free", [False, True])
def test_direct_lsq_with_every_control_point_pinned(free):
    # Degree 1 with two control points per section: the constraints fix every
    # control point, so only the data parameters move
    result = ConstrainedLSQOptimizer().fit(
        SplineConfig(degree=1, num_control_points=2, continuity=ContinuityMode.G0),
        OptimizationConfig(optimize_le_position=free), _airfoil())
    assert np.isfinite(result.fun)
    assert [len(c) for c in result.section_control_points] == [2, 2]


def test_points_cross_a_junction_in_direct_lsq():
    airfoil = _airfoil()
    result = ConstrainedLSQOptimizer().fit(
        SplineConfig(degree=3, num_control_points=10, continuity=ContinuityMode.G0),
        OptimizationConfig(optimize_le_position=True), airfoil)
    split = airfoil.junction_indices[0]
    start = _chain(airfoil).parameters
    moved = np.floor(result.parameters) != np.floor(start)
    assert np.any(moved[split - 5:split + 6])


def test_junctions_may_go_anywhere_a_control_point_may():
    airfoil = _airfoil(3)
    config = SplineConfig(degree=3, num_control_points=8, num_sections=3, continuity=ContinuityMode.G1)
    bounds = compute_optimization_bounds(airfoil, config, optimize_le_position=True)
    free_point = bounds[:2]
    assert bounds[-4:] == [*free_point, *free_point]


def test_orthogonal_is_the_default_metric():
    assert OptimizationConfig().metric == ErrorMetric.ORTHOGONAL
