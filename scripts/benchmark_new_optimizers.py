#!/usr/bin/env python3
"""
Benchmark New Optimizers
========================

Compares performance of the new optimizers:
- constrained-lsq (baseline parametric fit)
- orthogonal-lsq (updated with vectorized footpoint module)
- varpro-lsq (new VARPRO Gauss-Newton solver)

Usage:
    python scripts/benchmark_new_optimizers.py
"""

from __future__ import annotations

import sys
import time
from typing import List, Tuple

import numpy as np
from scipy.interpolate import BSpline as ScipyBSpline

# =============================================================================
# NACA 4-Digit Airfoil Generator
# =============================================================================

def naca_4digit(code: str, n_points: int = 100) -> List[Tuple[float, float]]:
    """
    Generate NACA 4-digit airfoil coordinates in Selig format.
    
    Args:
        code: 4-digit NACA code (e.g., "0012", "2412", "4415")
        n_points: Number of points per surface (total = 2*n_points - 1)
        
    Returns:
        List of (x, y) coordinates from upper TE -> LE -> lower TE (Selig format)
    """
    if len(code) != 4:
        raise ValueError(f"NACA code must be 4 digits, got '{code}'")
    
    m = int(code[0]) / 100.0  # Maximum camber
    p = int(code[1]) / 10.0   # Position of max camber
    t = int(code[2:]) / 100.0 # Thickness
    
    # Cosine spacing for better LE resolution
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))
    
    # Thickness distribution (NACA)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    # Camber line
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    if p > 0 and m > 0:
        # Front portion (0 <= x < p)
        front = x < p
        yc[front] = (m / p**2) * (2 * p * x[front] - x[front]**2)
        dyc_dx[front] = (2 * m / p**2) * (p - x[front])
        
        # Rear portion (p <= x <= 1)
        rear = ~front
        yc[rear] = (m / (1 - p)**2) * ((1 - 2*p) + 2 * p * x[rear] - x[rear]**2)
        dyc_dx[rear] = (2 * m / (1 - p)**2) * (p - x[rear])
    
    # Compute angle of camber line
    theta = np.arctan(dyc_dx)
    
    # Upper and lower surfaces
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Combine in Selig format: upper (TE->LE) + lower (LE->TE, skip LE duplicate)
    upper = list(zip(xu[::-1], yu[::-1]))  # Reverse for TE->LE
    lower = list(zip(xl[1:], yl[1:]))      # Skip LE duplicate
    
    return upper + lower


def compute_orthogonal_rms(
    section_control_points: List,
    sections: List,
    degree: int,
    knots: np.ndarray,
) -> float:
    """Compute orthogonal distance RMS error using footpoint projection."""
    from spline_fitter.optimization.footpoint import iterative_footpoint_projection
    from spline_fitter.geometry import compute_arc_length_parameters
    
    total_sq_dist = 0.0
    total_points = 0
    
    for sec_idx, sec_cps in enumerate(section_control_points):
        cx = np.array([p[0] for p in sec_cps])
        cy = np.array([p[1] for p in sec_cps])
        sec_pts = np.array(sections[sec_idx], dtype=np.float64)
        
        spline_x = ScipyBSpline(knots, cx, degree, extrapolate=False)
        spline_y = ScipyBSpline(knots, cy, degree, extrapolate=False)
        
        # Initialize t-values with arc-length
        t_init = np.array(compute_arc_length_parameters(sections[sec_idx]), dtype=np.float64)
        
        # Find footpoints (closest points on curve)
        _, dists = iterative_footpoint_projection(
            t_init=t_init,
            data_points=sec_pts,
            spline_x=spline_x,
            spline_y=spline_y,
            max_iter=5,
            tol=1e-10,
        )
        
        total_sq_dist += np.sum(dists**2)
        total_points += len(dists)
    
    return np.sqrt(total_sq_dist / total_points) if total_points > 0 else float('inf')


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark():
    """Run the optimizer benchmark."""
    from spline_fitter.optimization.models import (
        AirfoilData,
        SplineConfig,
        OptimizationConfig,
        FairnessWeights,
        OptimizerType,
        ErrorMetric,
        ContinuityMode,
    )
    from spline_fitter.optimization.engine import run_optimization, reset_iteration_tracking
    from spline_fitter.geometry import (
        create_airfoil_data,
        compute_optimization_bounds,
        compute_arc_length_parameters,
    )
    from spline_fitter.spline.spline import BSplineCurve2D, create_open_uniform_knots
    
    print("=" * 80)
    print("OPTIMIZER BENCHMARK: constrained-lsq vs orthogonal-lsq vs varpro-lsq")
    print("=" * 80)
    print()
    
    # Generate test airfoils
    test_cases = [
        ("NACA0012", naca_4digit("0012", 80)),   # Symmetric
        ("NACA2412", naca_4digit("2412", 80)),   # Cambered
        ("NACA4415", naca_4digit("4415", 100)),  # Higher camber
    ]
    
    # Optimizer configurations to test
    optimizers = [
        (OptimizerType.DIRECT_LSQ, "Direct LSQ (baseline)"),
        (OptimizerType.GN_VARPRO, "VarPro LM"),
    ]
    
    # Spline configuration
    config = SplineConfig(
        degree=5,
        num_control_points=20,
        num_sections=2,
        continuity=ContinuityMode.G1,
    )
    
    # Fairness weights (none for clean comparison)
    fairness = FairnessWeights()
    
    results = []
    
    for airfoil_name, coords in test_cases:
        print(f"\n{'-' * 80}")
        print(f"Airfoil: {airfoil_name} ({len(coords)} points)")
        print(f"{'-' * 80}")
        
        # Split into upper/lower surfaces using the helper
        airfoil = create_airfoil_data(
            coordinates=coords,
            num_sections=2,
            filename=f"{airfoil_name}.dat",
            name=airfoil_name,
        )
        
        sections = [list(sec) for sec in airfoil.sections]
        
        # Compute bounds for iterative optimizers
        bounds = compute_optimization_bounds(
            airfoil, config, False
        )
        
        # Create knots once
        knots = create_open_uniform_knots(config.degree, config.num_control_points)
        
        for opt_type, opt_name in optimizers:
            # Reset tracking
            reset_iteration_tracking()
            
            # Configure optimizer
            opt_config = OptimizationConfig(
                optimizer=opt_type,
                metric=ErrorMetric.PARAMETRIC,
                max_iterations=50,  # Allow more iterations for convergence
                optimize_le_position=False,
            )
            
            # Run optimization with timing
            start_time = time.perf_counter()
            
            try:
                result = run_optimization(
                    config=config,
                    opt_config=opt_config,
                    airfoil=airfoil,
                    fairness_weights=fairness,
                    bounds=bounds,
                    callback=None,
                )
                elapsed = time.perf_counter() - start_time
                
                if result.section_control_points is not None:
                    # Compute parametric RMS error
                    total_error = 0.0
                    total_pts = 0
                    
                    for sec_idx, sec_cps in enumerate(result.section_control_points):
                        spline = BSplineCurve2D(config.degree, sec_cps, knots)
                        sec_pts = np.array(sections[sec_idx])
                        
                        t_vals = compute_arc_length_parameters(sections[sec_idx])
                        
                        for i, t in enumerate(t_vals):
                            t_clamped = max(0.0, min(1.0, t))
                            pt = spline.evaluate(t_clamped)
                            dx = pt[0] - sec_pts[i, 0]
                            dy = pt[1] - sec_pts[i, 1]
                            total_error += dx*dx + dy*dy
                        total_pts += len(t_vals)
                    
                    param_rms = np.sqrt(total_error / total_pts) if total_pts > 0 else float('inf')
                    
                    # Compute orthogonal RMS error
                    orth_rms = compute_orthogonal_rms(result.section_control_points, sections, config.degree, knots)
                    
                    n_iters = result.nit if hasattr(result, 'nit') else 1
                    
                    results.append({
                        'airfoil': airfoil_name,
                        'optimizer': opt_name,
                        'time_ms': elapsed * 1000,
                        'param_rms': param_rms,
                        'orth_rms': orth_rms,
                        'iterations': n_iters,
                    })
                    
                    print(f"  {opt_name:35s} | {elapsed*1000:7.1f} ms | param: {param_rms:.2e} | orth: {orth_rms:.2e} | iters: {n_iters:3d}")
                else:
                    print(f"  {opt_name:35s} | FAILED (no control points)")
                    results.append({
                        'airfoil': airfoil_name,
                        'optimizer': opt_name,
                        'time_ms': elapsed * 1000,
                        'param_rms': float('inf'),
                        'orth_rms': float('inf'),
                        'iterations': 0,
                    })
            
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                import traceback
                print(f"  {opt_name:35s} | ERROR: {e}")
                traceback.print_exc()
                results.append({
                    'airfoil': airfoil_name,
                    'optimizer': opt_name,
                    'time_ms': elapsed * 1000,
                    'param_rms': float('inf'),
                    'orth_rms': float('inf'),
                    'iterations': 0,
                })
    
    # Print summary table
    print("\n")
    print("=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Airfoil':<12} | {'Optimizer':<35} | {'Time (ms)':>10} | {'Param RMS':>12} | {'Orth RMS':>12} | {'Iters':>5}")
    print("-" * 100)
    
    for r in results:
        param_str = f"{r['param_rms']:.2e}" if r['param_rms'] < float('inf') else "FAILED"
        orth_str = f"{r['orth_rms']:.2e}" if r['orth_rms'] < float('inf') else "FAILED"
        print(f"{r['airfoil']:<12} | {r['optimizer']:<35} | {r['time_ms']:>10.1f} | {param_str:>12} | {orth_str:>12} | {r['iterations']:>5}")
    
    # Compute average performance by optimizer
    print("\n")
    print("=" * 100)
    print("AVERAGE PERFORMANCE BY OPTIMIZER")
    print("=" * 100)
    
    from collections import defaultdict
    opt_stats = defaultdict(lambda: {'times': [], 'param_errors': [], 'orth_errors': []})
    
    for r in results:
        if r['param_rms'] < float('inf'):
            opt_stats[r['optimizer']]['times'].append(r['time_ms'])
            opt_stats[r['optimizer']]['param_errors'].append(r['param_rms'])
            opt_stats[r['optimizer']]['orth_errors'].append(r['orth_rms'])
    
    print(f"{'Optimizer':<35} | {'Avg Time (ms)':>12} | {'Avg Param RMS':>14} | {'Avg Orth RMS':>14} | {'Success':>7}")
    print("-" * 100)
    
    for opt_name in [o[1] for o in optimizers]:
        stats = opt_stats[opt_name]
        if stats['times']:
            avg_time = np.mean(stats['times'])
            avg_param = np.mean(stats['param_errors'])
            avg_orth = np.mean(stats['orth_errors'])
            success = len(stats['times'])
            print(f"{opt_name:<35} | {avg_time:>12.1f} | {avg_param:>14.2e} | {avg_orth:>14.2e} | {success}/{len(test_cases):>5}")
        else:
            print(f"{opt_name:<35} | {'N/A':>12} | {'N/A':>14} | {'N/A':>14} | 0/{len(test_cases):>5}")
    
    print("\n")
    print("=" * 100)
    print("CONCLUSIONS")
    print("=" * 100)
    
    # Find fastest and most accurate
    valid = [r for r in results if r['param_rms'] < float('inf')]
    if valid:
        fastest = min(valid, key=lambda x: x['time_ms'])
        best_param = min(valid, key=lambda x: x['param_rms'])
        best_orth = min(valid, key=lambda x: x['orth_rms'])
        
        print(f"Fastest:              {fastest['optimizer']} ({fastest['time_ms']:.1f} ms on {fastest['airfoil']})")
        print(f"Best parametric fit:  {best_param['optimizer']} (RMS {best_param['param_rms']:.2e} on {best_param['airfoil']})")
        print(f"Best orthogonal fit:  {best_orth['optimizer']} (RMS {best_orth['orth_rms']:.2e} on {best_orth['airfoil']})")
    
    return results


if __name__ == "__main__":
    run_benchmark()
