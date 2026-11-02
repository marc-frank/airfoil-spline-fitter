"""The curve energy measures every degree that can bend."""
from spline_fitter.spline import BSplineCurve2D
from spline_fitter.spline.fairness import compute_curve_energy


def test_wiggle_is_measured_on_a_quadratic():
    curve = BSplineCurve2D(2, [(0, 0), (0.2, 0.3), (0.6, 0.1), (0.8, 0.4), (1, 0)])
    assert compute_curve_energy(curve)["curvature_gradient"] > 0


def test_a_broken_curve_raises_instead_of_reading_as_fair():
    """An error is not reported as zero energy (a perfectly fair curve)."""
    import pytest

    class Broken:
        knot_vector = [0.0, 0.0, 1.0, 1.0]
        degree = 1

    with pytest.raises(AttributeError):
        compute_curve_energy(Broken())
