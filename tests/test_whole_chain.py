"""The chain of sections is fitted to the data as a whole: the split only sets
where the junctions start, and points are not assigned to sections."""
import numpy as np
import pytest
from bspline_varpro.airfoil import NACA4
from bspline_varpro.chain import chain_data, chain_parameters, locate

from spline_fitter.geometry import compute_arc_length_parameters, create_airfoil_data
from spline_fitter.optimization.models import ContinuityMode, OptimizationConfig, SplineConfig
from spline_fitter.optimization.optimizers.constrained_lsq import ConstrainedLSQOptimizer
from spline_fitter.spline.spline import BSplineCurve2D


def _airfoil(sections=2, n=81):
    coords = [tuple(p) for p in np.round(NACA4("2412").points(n), 6)]
    return create_airfoil_data(coords, num_sections=sections)


def _sections(airfoil):
    return [np.asarray(s, dtype=np.float64) for s in airfoil.sections]


def _start_parameters(airfoil):
    t = [np.asarray(compute_arc_length_parameters(list(s))) for s in airfoil.sections]
    return chain_parameters(t, _sections(airfoil))


def test_a_shared_split_point_counts_once():
    airfoil = _airfoil()
    assert len(chain_data(_sections(airfoil))) == len(airfoil.coordinates)


@pytest.mark.parametrize("sections", [2, 3])
@pytest.mark.parametrize("mode", [ContinuityMode.G0, ContinuityMode.G2])
def test_freeing_the_junctions_never_hurts_direct_lsq(sections, mode):
    airfoil = _airfoil(sections)
    config = SplineConfig(degree=3, num_control_points=10, num_sections=sections, continuity=mode)
    data = chain_data(_sections(airfoil))

    def rms(free):
        result = ConstrainedLSQOptimizer().fit(config, OptimizationConfig(optimize_le_position=free),
                                               airfoil)
        curves = [BSplineCurve2D(3, c) for c in result.section_control_points]
        section, local = locate(result.parameters, sections)
        on_chain = np.array([curves[s].evaluate(t) for s, t in zip(section, local)])
        return np.sqrt(np.mean(np.sum((on_chain - data) ** 2, axis=1))), result

    pinned, pinned_result = rms(False)
    free, _ = rms(True)
    assert free <= pinned * (1 + 1e-6)
    # Pinned junctions stay at their split points
    for s, split in enumerate(airfoil.junction_indices):
        assert pinned_result.section_control_points[s][-1] == pytest.approx(airfoil.coordinates[split])


@pytest.mark.parametrize("free", [False, True])
def test_direct_lsq_with_every_control_point_pinned(free):
    # Degree 1 with two control points per section: the constraints fix every
    # control point (with free junctions the junction point is the solve's)
    result = ConstrainedLSQOptimizer().fit(
        SplineConfig(degree=1, num_control_points=2, continuity=ContinuityMode.G0),
        OptimizationConfig(optimize_le_position=free), _airfoil())
    assert np.isfinite(result.fun)
    assert [len(c) for c in result.section_control_points] == [2, 2]


def test_direct_lsq_keeps_the_data_parameters():
    """Direct LSQ is one solve at fixed parameters: the points keep the
    parameters their spacing gave them, also with free junctions."""
    airfoil = _airfoil()
    result = ConstrainedLSQOptimizer().fit(
        SplineConfig(degree=3, num_control_points=10, continuity=ContinuityMode.G0),
        OptimizationConfig(optimize_le_position=True, parameterization="arc_length"), airfoil)
    assert np.array_equal(result.parameters, _start_parameters(airfoil))
