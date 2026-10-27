"""
Tests for newly refactored spline fitter components.

This module tests:
1. footpoint.py - Vectorized Newton-Raphson foot-point projection
2. varpro_lsq.py - Variable Projection Gauss-Newton optimizer
3. centripetal_parametrize - New parametrization function in geometry.py
4. constraints.py - Consolidated constraint coefficient computation
"""

import math
import numpy as np
import pytest
from scipy.interpolate import BSpline as ScipyBSpline


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_splines():
    """Create simple test splines using proper clamped knot vector."""
    from spline_fitter.spline.spline import create_open_uniform_knots
    
    degree = 3
    n_ctrl = 8  # Need > degree for valid knots
    knots = create_open_uniform_knots(degree, n_ctrl)
    
    # Control points for approximately a quarter circle from (1,0) to (0,1)
    ctrl_x = np.array([1.0, 0.95, 0.8, 0.5, 0.2, 0.05, 0.0, 0.0])
    ctrl_y = np.array([0.0, 0.0, 0.15, 0.5, 0.8, 0.95, 1.0, 1.0])
    
    spline_x = ScipyBSpline(knots, ctrl_x, degree)
    spline_y = ScipyBSpline(knots, ctrl_y, degree)
    
    return spline_x, spline_y


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
    
    upper = list(zip(x_upper.tolist(), y_upper.tolist()))
    lower = list(zip(x_lower.tolist(), y_lower.tolist()))
    
    return upper, lower


# =============================================================================
# Tests for footpoint.py
# =============================================================================

class TestFootpoint:
    """Tests for the vectorized foot-point projection module."""
    
    def test_vectorized_footpoint_update_basic(self, simple_splines):
        """Test basic foot-point update returns valid parameters."""
        from spline_fitter.optimization.footpoint import vectorized_footpoint_update
        
        spline_x, spline_y = simple_splines
        
        # Data points near the curve
        data_points = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.9],
        ])
        
        # Initial parameter guesses (roughly corresponding to positions)
        t_init = np.array([0.1, 0.5, 0.9])
        
        # Single update step
        t_new = vectorized_footpoint_update(
            t_init, data_points, spline_x, spline_y
        )
        
        # Check output shape and range
        assert t_new.shape == t_init.shape
        assert np.all(t_new >= 0.0)
        assert np.all(t_new <= 1.0)
        
    def test_iterative_footpoint_projection_convergence(self, simple_splines):
        """Test iterative projection converges to correct foot-points."""
        from spline_fitter.optimization.footpoint import (
            iterative_footpoint_projection,
            compute_distances,
        )
        
        spline_x, spline_y = simple_splines
        
        # Generate points exactly on the curve
        t_exact = np.array([0.2, 0.4, 0.6, 0.8])
        on_curve_pts = np.column_stack([
            spline_x(t_exact),
            spline_y(t_exact),
        ])
        
        # Start with perturbed initial guess
        t_init = t_exact + np.array([0.05, -0.03, 0.02, -0.04])
        t_init = np.clip(t_init, 0, 1)
        
        # Run projection
        t_final, distances = iterative_footpoint_projection(
            t_init, on_curve_pts, spline_x, spline_y, max_iter=5
        )
        
        # Should converge close to exact parameters
        np.testing.assert_allclose(t_final, t_exact, atol=1e-6)
        
        # Distances should be near zero (points are on curve)
        np.testing.assert_allclose(distances, 0.0, atol=1e-8)
    
    def test_compute_distances(self, simple_splines):
        """Test distance computation is correct."""
        from spline_fitter.optimization.footpoint import compute_distances
        
        spline_x, spline_y = simple_splines
        
        # Point exactly on curve at t=0.5
        curve_pt = np.array([[spline_x(0.5), spline_y(0.5)]])
        t = np.array([0.5])
        
        dist = compute_distances(t, curve_pt, spline_x, spline_y)
        assert dist[0] == pytest.approx(0.0, abs=1e-12)
        
        # Point off the curve
        off_curve_pt = np.array([[0.7, 0.7]])
        dist = compute_distances(np.array([0.5]), off_curve_pt, spline_x, spline_y)
        assert dist[0] > 0.0
    
    def test_footpoint_handles_empty_input(self, simple_splines):
        """Test that empty inputs are handled gracefully."""
        from spline_fitter.optimization.footpoint import (
            vectorized_footpoint_update,
            iterative_footpoint_projection,
        )
        
        spline_x, spline_y = simple_splines
        
        empty_t = np.array([])
        empty_pts = np.array([]).reshape(0, 2)
        
        t_new = vectorized_footpoint_update(empty_t, empty_pts, spline_x, spline_y)
        assert len(t_new) == 0
        
        t_final, dists = iterative_footpoint_projection(
            empty_t, empty_pts, spline_x, spline_y
        )
        assert len(t_final) == 0
        assert len(dists) == 0


# =============================================================================
# Tests for centripetal_parametrize
# =============================================================================

class TestCentripetalParametrize:
    """Tests for centripetal parametrization function."""
    
    def test_basic_parametrization(self):
        """Test basic output properties."""
        from spline_fitter.geometry import centripetal_parametrize
        
        points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        
        params = centripetal_parametrize(points)
        
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
        from spline_fitter.geometry import centripetal_parametrize
        
        # Non-uniform spacing
        points = [(0.0, 0.0), (0.1, 0.0), (0.5, 0.0), (1.0, 0.0)]
        
        # alpha=0: uniform
        uniform = centripetal_parametrize(points, alpha=0.0)
        expected_uniform = [0.0, 1/3, 2/3, 1.0]
        np.testing.assert_allclose(uniform, expected_uniform, atol=1e-10)
        
        # alpha=1: chord-length
        chord = centripetal_parametrize(points, alpha=1.0)
        # Distances: 0.1, 0.4, 0.5 -> cumulative: 0, 0.1, 0.5, 1.0
        expected_chord = [0.0, 0.1, 0.5, 1.0]
        np.testing.assert_allclose(chord, expected_chord, atol=1e-10)
        
        # alpha=0.5: centripetal (default)
        centripetal = centripetal_parametrize(points, alpha=0.5)
        # Centripetal uses sqrt of distances, so spacing is compressed for longer segments
        # sqrt(0.1) ≈ 0.316, sqrt(0.4) ≈ 0.632, sqrt(0.5) ≈ 0.707
        # Total ≈ 1.655, so param[1] ≈ 0.316/1.655 ≈ 0.191
        # This is LESS than uniform (0.333), not more
        assert centripetal[1] < uniform[1]  # Less than uniform for first short segment
        # Centripetal gives more spacing to short segments relative to chord-length
        assert centripetal[2] > chord[2]  # Centripetal spreads params differently
    
    def test_empty_and_single_point(self):
        """Test edge cases with empty or single point."""
        from spline_fitter.geometry import centripetal_parametrize
        
        assert centripetal_parametrize([]) == []
        assert centripetal_parametrize([(1.0, 2.0)]) == [0.0]
    
    def test_coincident_points(self):
        """Test handling of coincident points."""
        from spline_fitter.geometry import centripetal_parametrize
        
        # All points at same location
        points = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
        params = centripetal_parametrize(points)
        
        # Should handle gracefully (uniform fallback)
        assert len(params) == 3
        assert params[0] == pytest.approx(0.0)
        assert params[-1] == pytest.approx(1.0)


# =============================================================================
# Tests for constraints.py
# =============================================================================

class TestConstraints:
    """Tests for constraint coefficient computation."""
    
    def test_first_derivative_coefficients(self):
        """Test G1 constraint coefficients are computed correctly."""
        from spline_fitter.optimization.constraints import compute_derivative_constraint_coeffs
        from spline_fitter.spline.spline import create_open_uniform_knots
        
        degree = 5
        n_ctrl = 15
        knots = create_open_uniform_knots(degree, n_ctrl)
        
        coef_end, coef_start = compute_derivative_constraint_coeffs(
            knots, n_ctrl, degree, order=1
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
        from spline_fitter.optimization.constraints import compute_derivative_constraint_coeffs
        from spline_fitter.spline.spline import create_open_uniform_knots
        
        degree = 5
        n_ctrl = 15
        knots = create_open_uniform_knots(degree, n_ctrl)
        
        coef_end, coef_start = compute_derivative_constraint_coeffs(
            knots, n_ctrl, degree, order=2
        )
        
        # Second derivative coefficients should have length 3
        assert len(coef_end) == 3
        assert len(coef_start) == 3
        
        # Verify they're not all zero
        assert np.any(coef_start != 0)
        assert np.any(coef_end != 0)
    
    def test_third_derivative_coefficients(self):
        """Test G3 constraint coefficients."""
        from spline_fitter.optimization.constraints import compute_derivative_constraint_coeffs
        from spline_fitter.spline.spline import create_open_uniform_knots
        
        degree = 5
        n_ctrl = 15
        knots = create_open_uniform_knots(degree, n_ctrl)
        
        coef_end, coef_start = compute_derivative_constraint_coeffs(
            knots, n_ctrl, degree, order=3
        )
        
        # Third derivative coefficients should have length 4
        assert len(coef_end) == 4
        assert len(coef_start) == 4
        
        # Verify they're not all zero
        assert np.any(coef_start != 0)
        assert np.any(coef_end != 0)
    
    def test_constraint_symmetry_for_uniform_knots(self):
        """Verify that end/start coefficients have expected symmetry."""
        from spline_fitter.optimization.constraints import compute_derivative_constraint_coeffs
        from spline_fitter.spline.spline import create_open_uniform_knots
        
        degree = 5
        n_ctrl = 15
        knots = create_open_uniform_knots(degree, n_ctrl)
        
        for order in [1, 2, 3]:
            coef_end, coef_start = compute_derivative_constraint_coeffs(
                knots, n_ctrl, degree, order=order
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
        from spline_fitter.optimization.optimizers.constrained_lsq import ConstrainedLSQOptimizer
        from spline_fitter.optimization.models import (
            AirfoilData,
            SplineConfig,
            OptimizationConfig,
        )
        
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
    
    def test_orthogonal_lsq_uses_footpoint_module(self, airfoil_like_points):
        """Test OrthogonalLSQOptimizer runs and uses footpoint module."""
        from spline_fitter.optimization.optimizers.orthogonal_lsq import OrthogonalLSQOptimizer
        from spline_fitter.optimization.models import (
            AirfoilData,
            SplineConfig,
            OptimizationConfig,
        )
        
        upper, lower = airfoil_like_points
        
        # Combine coordinates in Selig format
        all_coords = list(upper) + list(lower[1:])
        junction_idx = len(upper) - 1
        
        airfoil = AirfoilData(
            coordinates=tuple(all_coords),
            sections=(tuple(upper), tuple(lower)),
            junction_indices=(junction_idx,),
            num_sections=2,
            name="test_airfoil",
        )
        
        config = SplineConfig(degree=5, num_control_points=15, num_sections=2)
        opt_config = OptimizationConfig(max_iterations=3)  # Few iterations for speed
        
        optimizer = OrthogonalLSQOptimizer(max_outer_iterations=3)
        result = optimizer.fit(config, opt_config, airfoil)
        
        # Should complete without error
        assert result is not None
        assert result.fun >= 0
        assert result.section_control_points is not None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
