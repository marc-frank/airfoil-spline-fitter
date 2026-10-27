#!/usr/bin/env python3
"""
Optimizer Benchmark Script
==========================

Runs all local optimizers against a set of test airfoils and logs results.

Usage:
    python scripts/benchmark_optimizers.py

Results are saved to optimizer_logs/ (project root) with timestamped filenames.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# Add project root to path so `import spline_fitter` resolves
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from spline_fitter.io import parse_airfoil_data
from spline_fitter.geometry import create_airfoil_data, compute_optimization_bounds
from spline_fitter.optimizers.models import (
    SplineConfig, OptimizationConfig, FairnessWeights,
    OptimizerType, ContinuityMode, ErrorMetric,
)
from spline_fitter.optimization.objective_functions.optimization import (
    run_optimization,
    reset_iteration_tracking, get_evaluation_count,
    OptimizerErrorLogger,
)


# =============================================================================
# Configuration
# =============================================================================

AIRFOIL_LIBRARY = Path("C:/Users/MSFra/Desktop/airfoil_library/airfoils")

# Test airfoils with diverse characteristics
# Organized by category for systematic coverage
# 40 airfoils × 7 optimizers × 5min timeout = ~23hrs worst case, ~10hrs realistic
TEST_AIRFOILS = [
    # === NACA 4-digit symmetric (5) - thickness variation ===
    "naca0006-uiuc.dat",    # 35 pts, 6% thick
    "naca0008-uiuc.dat",    # 35 pts, 8% thick
    "naca0012-uiuc.dat",    # 35 pts, 12% thick (different from bf version)
    "naca0015-uiuc.dat",    # 35 pts, 15% thick
    "naca0018-uiuc.dat",    # 35 pts, 18% thick
    
    # === NACA 4-digit cambered (8) - camber/thickness variation ===
    "naca1408-uiuc.dat",    # 35 pts, 1% camber, 8% thick
    "naca1412-uiuc.dat",    # 35 pts, 1% camber, 12% thick
    "naca2408-uiuc.dat",    # 35 pts, 2% camber, 8% thick
    "naca2412-uiuc.dat",    # 35 pts, 2% camber, 12% thick, GA classic
    "naca2415-uiuc.dat",    # 35 pts, 2% camber, 15% thick
    "naca4412-uiuc.dat",    # 35 pts, 4% camber, 12% thick
    "naca4418-uiuc.dat",    # 35 pts, 4% camber, 18% thick
    "naca6412-uiuc.dat",    # 62 pts, 6% camber, 12% thick
    
    # === NACA 5-digit series (1) ===
    "naca23012-uiuc.dat",   # 61 pts, 12% thick, reflex camber
    
    # === Eppler series (6) - glider/sailplane ===
    "e193-uiuc.dat",        # 61 pts, high-lift
    "e205-uiuc.dat",        # 61 pts, sailplane
    "e211-uiuc.dat",        # 61 pts, sailplane
    "e374-uiuc.dat",        # 61 pts, low drag
    "e395-uiuc.dat",        # 72 pts, sailplane
    "e423-uiuc.dat",        # 72 pts, high-lift sailplane
    
    # === Selig SD series (5) - RC/drone ===
    "sd7003-uiuc.dat",      # 61 pts, low Re classic
    "sd7037-uiuc.dat",      # 61 pts, popular RC
    "sd7062-uiuc.dat",      # 61 pts, RC glider
    "sd7080-uiuc.dat",      # 61 pts, RC (your original test!)
    "sd8020-uiuc.dat",      # 61 pts, symmetric aerobatic
    
    # === Selig S series (5) - high-lift ===
    "s1210-uiuc.dat",       # 81 pts, high-lift
    "s1221-uiuc.dat",       # 72 pts, high-lift
    "s2027-uiuc.dat",       # 62 pts
    "s3010-uiuc.dat",       # 61 pts
    "s6063-uiuc.dat",       # 61 pts
    
    # === Other designers (7) ===
    "rg15-uiuc.dat",        # 62 pts, Roncz sailplane
    "mh32-uiuc.dat",        # 68 pts, Hepperle
    "mh60-uiuc.dat",        # 68 pts, Hepperle flying wing
    "mh114-uiuc.dat",       # 68 pts, Hepperle
    "hq3510-uiuc.dat",      # 77 pts, Horstmann-Quast sailplane
    "sa7035-uiuc.dat",      # 81 pts, Selig-Ashok
    "s8036-uiuc.dat",       # 81 pts, Selig low-drag
    
    # === Transonic/special (4) ===
    "rae2822-uiuc.dat",     # 129 pts, transonic research
    "whitcomb-uiuc.dat",    # 73 pts, supercritical
    "ah93w145-uiuc.dat",    # 97 pts, Althaus wind turbine
    "goe795-uiuc.dat",      # 97 pts, Göttingen historical
]

# Optimizers to test (local optimizers only for fair comparison)
TEST_OPTIMIZERS = [
    OptimizerType.BFGS,
    OptimizerType.L_BFGS_B,
    OptimizerType.CG,
    OptimizerType.SLSQP,
    OptimizerType.POWELL,
    OptimizerType.NELDER_MEAD,
    OptimizerType.COBYLA,
]

# Common configuration
SPLINE_CONFIG = SplineConfig(
    degree=5,
    num_control_points=25,
    continuity=ContinuityMode.G1,
)

OPT_CONFIG_BASE = OptimizationConfig(
    max_iterations=10000,
    metric=ErrorMetric.PARAMETRIC,
    seed=42,
)

FAIRNESS_WEIGHTS = FairnessWeights()  # Default (all zeros)

# Timeout per optimizer (seconds)
# BFGS/L-BFGS-B/CG finish in <30s, SLSQP needs ~2-3min for full convergence
# Powell/Nelder-Mead take 30-80min and aren't competitive
TIMEOUT_SECONDS = 5 * 60  # 5 minutes


# =============================================================================
# Benchmark Functions
# =============================================================================

def load_airfoil(filename: str):
    """Load an airfoil from the library."""
    filepath = AIRFOIL_LIBRARY / filename
    if not filepath.exists():
        print(f"  ERROR: File not found: {filepath}")
        return None
    
    filename_name, content_name, coords, precision = parse_airfoil_data(str(filepath))
    if coords is None:
        print(f"  ERROR: Failed to parse: {filepath}")
        return None
    
    return create_airfoil_data(coords, filename_name, content_name or filename_name, precision)


def run_single_benchmark(
    airfoil,
    optimizer: OptimizerType,
    spline_config: SplineConfig,
    opt_config: OptimizationConfig,
    fairness_weights: FairnessWeights,
    timeout: float = 60.0,
) -> Tuple[float, float, int, bool]:
    """
    Run a single optimizer on a single airfoil.
    
    Returns:
        (final_error, elapsed_time, evaluations, timed_out)
    """
    reset_iteration_tracking()
    
    # Compute bounds
    bounds = compute_optimization_bounds(airfoil, spline_config)
    
    # Set up logger
    run_name = f"{airfoil.name} ({len(airfoil.coordinates)} pts)"
    logger = OptimizerErrorLogger(run_name=run_name)
    start_time = time.perf_counter()
    
    # Callback for logging and timeout
    best_error = [float('inf')]
    timed_out = [False]
    
    def callback(x, error):
        elapsed = time.perf_counter() - start_time
        
        # Check timeout
        if elapsed > timeout:
            timed_out[0] = True
            return True  # Signal early stop
        
        if error < best_error[0]:
            best_error[0] = error
            iteration = len(logger.iterations) + 1
            logger.log(iteration, get_evaluation_count(), error, elapsed)
        return False  # Continue
    
    # Run optimization
    opt_config_with_optimizer = OptimizationConfig(
        optimizer=optimizer,
        max_iterations=opt_config.max_iterations,
        metric=opt_config.metric,
        seed=opt_config.seed,
    )
    
    result = run_optimization(
        config=spline_config,
        opt_config=opt_config_with_optimizer,
        airfoil=airfoil,
        fairness_weights=fairness_weights,
        bounds=bounds,
        callback=callback,
    )
    
    elapsed = time.perf_counter() - start_time
    
    # Final log entry
    if result.fun < best_error[0]:
        logger.log(len(logger.iterations) + 1, get_evaluation_count(), result.fun, elapsed)
    
    # Save log
    timeout_note = " (TIMEOUT)" if timed_out[0] else ""
    extra_info = f"mode={spline_config.continuity.name}, degree={spline_config.degree}{timeout_note}"
    logger.save(optimizer.value, extra_info)
    
    return best_error[0], elapsed, get_evaluation_count(), timed_out[0]


def run_benchmark():
    """Run the full benchmark suite."""
    print("=" * 70)
    print("OPTIMIZER BENCHMARK")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Spline config: degree={SPLINE_CONFIG.degree}, "
          f"control_points={SPLINE_CONFIG.num_control_points}, "
          f"continuity={SPLINE_CONFIG.continuity.name}")
    print(f"Max iterations: {OPT_CONFIG_BASE.max_iterations}")
    print()
    
    # Results storage: results[airfoil][optimizer] = (error, time, evals)
    results = {}
    
    for airfoil_file in TEST_AIRFOILS:
        print(f"\n{'='*70}")
        print(f"AIRFOIL: {airfoil_file}")
        print("=" * 70)
        
        airfoil = load_airfoil(airfoil_file)
        if airfoil is None:
            continue
        
        print(f"  Points: {len(airfoil.coordinates)}")
        print(f"  Name: {airfoil.name}")
        print()
        
        results[airfoil_file] = {}
        
        for optimizer in TEST_OPTIMIZERS:
            print(f"  Testing {optimizer.display_name}...", end=" ", flush=True)
            
            try:
                error, elapsed, evals, timed_out = run_single_benchmark(
                    airfoil, optimizer,
                    SPLINE_CONFIG, OPT_CONFIG_BASE, FAIRNESS_WEIGHTS,
                    timeout=TIMEOUT_SECONDS
                )
                results[airfoil_file][optimizer.value] = (error, elapsed, evals, timed_out)
                timeout_str = " TIMEOUT" if timed_out else ""
                print(f"error={error:.6e}, time={elapsed:.1f}s, evals={evals}{timeout_str}")
            except Exception as e:
                print(f"FAILED: {e}")
                results[airfoil_file][optimizer.value] = (float('inf'), 0, 0, False)
    
    # Print summary
    print_summary(results)
    
    # Save summary CSV
    save_summary_csv(results)


def print_summary(results: dict):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Header
    header = f"{'Airfoil':<25}"
    for opt in TEST_OPTIMIZERS:
        header += f"{opt.value:>12}"
    print(header)
    print("-" * len(header))
    
    # Rows (error)
    print("\nFinal Error:")
    for airfoil_file, opt_results in results.items():
        row = f"{airfoil_file[:24]:<25}"
        for opt in TEST_OPTIMIZERS:
            result = opt_results.get(opt.value, (float('inf'), 0, 0, False))
            error = result[0]
            timed_out = result[3] if len(result) > 3 else False
            if error < float('inf'):
                suffix = "*" if timed_out else ""
                row += f"{error:>11.2e}{suffix}"
            else:
                row += f"{'FAIL':>12}"
        print(row)
    
    # Rows (time)
    print("\nTime (seconds):  (* = timed out)")
    for airfoil_file, opt_results in results.items():
        row = f"{airfoil_file[:24]:<25}"
        for opt in TEST_OPTIMIZERS:
            result = opt_results.get(opt.value, (float('inf'), 0, 0, False))
            elapsed = result[1]
            timed_out = result[3] if len(result) > 3 else False
            suffix = "*" if timed_out else ""
            row += f"{elapsed:>11.1f}{suffix}"
        print(row)
    
    # Winner analysis
    print("\n" + "-" * 70)
    print("WINNER BY AIRFOIL (lowest error):")
    for airfoil_file, opt_results in results.items():
        best_opt = None
        best_error = float('inf')
        for opt in TEST_OPTIMIZERS:
            result = opt_results.get(opt.value, (float('inf'), 0, 0, False))
            error = result[0]
            if error < best_error:
                best_error = error
                best_opt = opt.value
        print(f"  {airfoil_file}: {best_opt} ({best_error:.2e})")


def save_summary_csv(results: dict):
    """Save a summary CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = project_root / "optimizer_logs" / f"{timestamp}_benchmark_summary.csv"
    
    os.makedirs(summary_path.parent, exist_ok=True)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        # Header
        f.write("airfoil")
        for opt in TEST_OPTIMIZERS:
            f.write(f",{opt.value}_error,{opt.value}_time,{opt.value}_evals,{opt.value}_timeout")
        f.write("\n")
        
        # Data rows
        for airfoil_file, opt_results in results.items():
            f.write(airfoil_file)
            for opt in TEST_OPTIMIZERS:
                result = opt_results.get(opt.value, (float('inf'), 0, 0, False))
                error, elapsed, evals = result[0], result[1], result[2]
                timed_out = result[3] if len(result) > 3 else False
                f.write(f",{error:.12e},{elapsed:.3f},{evals},{1 if timed_out else 0}")
            f.write("\n")
    
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    run_benchmark()
