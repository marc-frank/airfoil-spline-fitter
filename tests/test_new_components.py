"""
Tests for newly refactored spline fitter components.

This module tests:
1. varpro_lsq.py - Variable Projection Gauss-Newton optimizer
2. centripetal_parameters - the library parametrization the application uses
3. end_derivative_coefficients - the library coefficients Direct LSQ uses
"""

import numpy as np
import pytest

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def airfoil_like_points():
    """Generate synthetic airfoil-like data points."""
    # Upper surface: cosine distribution from TE to LE
    n_upper = 30
    t_upper = np.linspace(0, np.pi, n_upper)
    x_upper = 0.5 * (1 + np.cos(t_upper))  # 1 -> 0
    y_upper = 0.12 * (0.2969 * np.sqrt(x_upper) - 0.126 * x_upper)  # NACA thickness

    # Lower surface: LE to TE
    n_lower = 30
    t_lower = np.linspace(np.pi, 2 * np.pi, n_lower)
    x_lower = 0.5 * (1 + np.cos(t_lower))  # 0 -> 1
    y_lower = -0.12 * (0.2969 * np.sqrt(x_lower + 1e-10) - 0.126 * x_lower)

    upper = list(zip(x_upper.tolist(), y_upper.tolist(), strict=False))
    lower = list(zip(x_lower.tolist(), y_lower.tolist(), strict=False))

    return upper, lower


# =============================================================================
# Tests for centripetal_parameters (bspline_varpro)
# =============================================================================

class TestCentripetalParametrize:
    """Tests for centripetal parametrization function."""

    def test_basic_parametrization(self):
        """Test basic output properties."""
        from bspline_varpro import centripetal_parameters

        points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]

        params = centripetal_parameters(points)

        # Check length matches
        assert len(params) == len(points)

        # Check range
        assert params[0] == pytest.approx(0.0)
        assert params[-1] == pytest.approx(1.0)

        # Check monotonicity
        for i in range(1, len(params)):
            assert params[i] >= params[i - 1]

    def test_alpha_variations(self):
        """Test different alpha values produce expected behavior."""
        from bspline_varpro import centripetal_parameters

        # Non-uniform spacing
        points = [(0.0, 0.0), (0.1, 0.0), (0.5, 0.0), (1.0, 0.0)]

        # alpha=0: uniform
        uniform = centripetal_parameters(points, alpha=0.0)
        expected_uniform = [0.0, 1/3, 2/3, 1.0]
        np.testing.assert_allclose(uniform, expected_uniform, atol=1e-10)

        # alpha=1: chord-length
        chord = centripetal_parameters(points, alpha=1.0)
        # Distances: 0.1, 0.4, 0.5 -> cumulative: 0, 0.1, 0.5, 1.0
        expected_chord = [0.0, 0.1, 0.5, 1.0]
        np.testing.assert_allclose(chord, expected_chord, atol=1e-10)

        # alpha=0.5: centripetal (default)
        centripetal = centripetal_parameters(points, alpha=0.5)
        # Centripetal uses sqrt of distances, so spacing is compressed for longer segments
        # sqrt(0.1) ≈ 0.316, sqrt(0.4) ≈ 0.632, sqrt(0.5) ≈ 0.707
        # Total ≈ 1.655, so param[1] ≈ 0.316/1.655 ≈ 0.191
        # This is LESS than uniform (0.333), not more
        assert centripetal[1] < uniform[1]  # Less than uniform for first short segment
        # Centripetal gives more spacing to short segments relative to chord-length
        assert centripetal[2] > chord[2]  # Centripetal spreads params differently

    def test_empty_and_single_point(self):
        """Test edge cases with empty or single point."""
        from bspline_varpro import centripetal_parameters

        assert centripetal_parameters(np.empty((0, 2))).tolist() == []
        assert centripetal_parameters([(1.0, 2.0)]).tolist() == [0.0]

    def test_coincident_points(self):
        """Test handling of coincident points."""
        from bspline_varpro import centripetal_parameters

        # All points at same location
        points = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
        params = centripetal_parameters(points)

        # Should handle gracefully (uniform fallback)
        assert len(params) == 3
        assert params[0] == pytest.approx(0.0)
        assert params[-1] == pytest.approx(1.0)


# =============================================================================
# Tests for end_derivative_coefficients (bspline_varpro), used by Direct LSQ
# =============================================================================

class TestConstraints:
    """Tests for constraint coefficient computation."""

    def test_first_derivative_coefficients(self):
        """Test G1 constraint coefficients are computed correctly."""
        from bspline_varpro import end_derivative_coefficients, open_uniform_knots

        degree = 5
        n_ctrl = 15
        knots = open_uniform_knots(degree, n_ctrl)

        coef_end, coef_start = end_derivative_coefficients(
            knots, degree, 1
        )

        # First derivative coefficients should have length 2
        assert len(coef_end) == 2
        assert len(coef_start) == 2

        # The form should be [-c, c] for some positive c
        # (derivative is proportional to P[1] - P[0])
        assert coef_start[0] < 0
        assert coef_start[1] > 0
        assert coef_start[0] == pytest.approx(-coef_start[1])

    def test_second_derivative_coefficients(self):
        """Test G2 constraint coefficients."""
        from bspline_varpro import end_derivative_coefficients, open_uniform_knots

        degree = 5
        n_ctrl = 15
        knots = open_uniform_knots(degree, n_ctrl)

        coef_end, coef_start = end_derivative_coefficients(
            knots, degree, 2
        )

        # Second derivative coefficients should have length 3
        assert len(coef_end) == 3
        assert len(coef_start) == 3

        # Verify they're not all zero
        assert np.any(coef_start != 0)
        assert np.any(coef_end != 0)

    def test_third_derivative_coefficients(self):
        """Test G3 constraint coefficients."""
        from bspline_varpro import end_derivative_coefficients, open_uniform_knots

        degree = 5
        n_ctrl = 15
        knots = open_uniform_knots(degree, n_ctrl)

        coef_end, coef_start = end_derivative_coefficients(
            knots, degree, 3
        )

        # Third derivative coefficients should have length 4
        assert len(coef_end) == 4
        assert len(coef_start) == 4

        # Verify they're not all zero
        assert np.any(coef_start != 0)
        assert np.any(coef_end != 0)

    def test_constraint_symmetry_for_uniform_knots(self):
        """Verify that end/start coefficients have expected symmetry."""
        from bspline_varpro import end_derivative_coefficients, open_uniform_knots

        degree = 5
        n_ctrl = 15
        knots = open_uniform_knots(degree, n_ctrl)

        for order in [1, 2, 3]:
            coef_end, coef_start = end_derivative_coefficients(
                knots, degree, order
            )

            # For uniform interior knots, end and start coefficients should
            # have the same magnitude pattern (possibly reversed sign)
            np.testing.assert_allclose(
                np.abs(coef_end),
                np.abs(coef_start[::-1]),  # Reversed order
                rtol=1e-10,
                err_msg=f"Order {order} coefficients not symmetric"
            )


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests verifying components work together."""

    def test_constrained_lsq_uses_design_matrix(self, airfoil_like_points):
        """Test ConstrainedLSQOptimizer runs without error."""
        from spline_fitter.optimization.models import (
            AirfoilData,
            OptimizationConfig,
            SplineConfig,
        )
        from spline_fitter.optimization.optimizers.constrained_lsq import ConstrainedLSQOptimizer

        upper, lower = airfoil_like_points

        # Combine coordinates in Selig format: upper TE->LE, lower LE->TE
        all_coords = list(upper) + list(lower[1:])  # Skip duplicate LE point

        # Junction is at the leading edge (end of upper surface)
        junction_idx = len(upper) - 1

        airfoil = AirfoilData(
            coordinates=tuple(all_coords),
            sections=(tuple(upper), tuple(lower)),
            junction_indices=(junction_idx,),
            num_sections=2,
            name="test_airfoil",
        )

        config = SplineConfig(degree=5, num_control_points=15, num_sections=2)
        opt_config = OptimizationConfig()

        optimizer = ConstrainedLSQOptimizer()
        result = optimizer.fit(config, opt_config, airfoil)

        # Should complete without error
        assert result is not None
        assert result.fun >= 0  # Error should be non-negative
        assert result.section_control_points is not None
        assert len(result.section_control_points) == 2

    @pytest.mark.parametrize("parameterization", ["centripetal", "arc_length"])
    @pytest.mark.parametrize("free_junctions", [False, True])
    def test_direct_lsq_parameterization_and_junctions(
        self, airfoil_like_points, parameterization, free_junctions
    ):
        """Direct LSQ honours the parameterization and the free-junction option."""
        from spline_fitter.optimization.models import (
            AirfoilData,
            OptimizationConfig,
            OptimizerType,
            SplineConfig,
        )
        from spline_fitter.optimization.optimizers import get_optimizer

        upper, lower = airfoil_like_points
        airfoil = AirfoilData(
            coordinates=tuple(list(upper) + list(lower[1:])),
            sections=(tuple(upper), tuple(lower)),
            junction_indices=(len(upper) - 1,),
            num_sections=2,
            name="test_airfoil",
        )
        config = SplineConfig(degree=5, num_control_points=15, num_sections=2)
        opt_config = OptimizationConfig(
            optimizer=OptimizerType.DIRECT_LSQ,
            parameterization=parameterization,
            optimize_le_position=free_junctions,
        )

        result = get_optimizer(OptimizerType.DIRECT_LSQ).fit(config, opt_config, airfoil)

        first, second = result.section_control_points
        np.testing.assert_allclose(first[-1], second[0], atol=1e-9)
        junction_on_data = np.allclose(first[-1], upper[-1], atol=1e-9)
        assert junction_on_data != free_junctions
        assert result.fun < 1e-2

    def test_direct_lsq_parameterization_changes_fit(self, airfoil_like_points):
        """Centripetal and arc-length parameters give different fits."""
        from spline_fitter.optimization.models import (
            AirfoilData,
            OptimizationConfig,
            OptimizerType,
            SplineConfig,
        )
        from spline_fitter.optimization.optimizers import get_optimizer

        upper, lower = airfoil_like_points
        airfoil = AirfoilData(
            coordinates=tuple(list(upper) + list(lower[1:])),
            sections=(tuple(upper), tuple(lower)),
            junction_indices=(len(upper) - 1,),
            num_sections=2,
            name="test_airfoil",
        )
        config = SplineConfig(degree=5, num_control_points=15, num_sections=2)
        fits = [
            get_optimizer(OptimizerType.DIRECT_LSQ).fit(
                config,
                OptimizationConfig(optimizer=OptimizerType.DIRECT_LSQ, parameterization=p),
                airfoil,
            ).x
            for p in ("centripetal", "arc_length")
        ]
        assert not np.allclose(fits[0], fits[1])


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
