#!/usr/bin/env python3
"""
Compare Search Strategies for Finding Optimal Control Points
=============================================================

Analyzes project files to compare two strategies for finding the minimum
number of control points where error drops below 1e-5:

1. LINEAR: Start at minimum and increment by 1 until threshold reached
2. BINARY: Jump by 100, then binary search to find exact threshold

Usage:
    python scripts/compare_search_strategies.py <project.asfproj> [--threshold 1e-5]

Example:
    python scripts/compare_search_strategies.py my_project.asfproj
    python scripts/compare_search_strategies.py my_project.asfproj --threshold 1e-6
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

# colors for a dark theme
COLORS = {
    'bg': '#1e1e1e',
    'fg': '#d4d4d4',
    'grid': '#3c3c3c',
    'accent': '#007acc',
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor': COLORS['bg'],
    'axes.edgecolor': COLORS['fg'],
    'axes.labelcolor': COLORS['fg'],
    'text.color': COLORS['fg'],
    'xtick.color': COLORS['fg'],
    'ytick.color': COLORS['fg'],
    'grid.color': COLORS['grid'],
    'legend.facecolor': '#252526',
    'legend.edgecolor': COLORS['grid'],
})


@dataclass
class SearchResult:
    """Result of a search strategy simulation."""
    found: bool
    control_points: Optional[int]
    fits_required: int
    search_path: List[int]  # Control points tested in order
    total_time: float = 0.0  # Total time in seconds


def linear_search(
    runs: List[Tuple[int, float, float]],  # (control_points, error, time) sorted by control_points
    threshold: float,
    min_ctrl: int,
) -> SearchResult:
    """
    Simulate linear search starting from min_ctrl.
    
    Args:
        runs: List of (control_points, error, time) tuples, sorted by control_points
        threshold: Error threshold to find (e.g., 1e-5)
        min_ctrl: Minimum control points to start from
        
    Returns:
        SearchResult with number of fits required and total time
    """
    # Build lookup dicts
    error_by_ctrl = {ctrl: err for ctrl, err, _ in runs}
    time_by_ctrl = {ctrl: t for ctrl, _, t in runs}
    max_ctrl = max(ctrl for ctrl, _, _ in runs)
    
    search_path = []
    fits = 0
    total_time = 0.0
    
    for ctrl in range(min_ctrl, max_ctrl + 1):
        if ctrl in error_by_ctrl:
            fits += 1
            search_path.append(ctrl)
            total_time += time_by_ctrl[ctrl]
            if error_by_ctrl[ctrl] < threshold:
                return SearchResult(
                    found=True,
                    control_points=ctrl,
                    fits_required=fits,
                    search_path=search_path,
                    total_time=total_time,
                )
    
    return SearchResult(
        found=False,
        control_points=None,
        fits_required=fits,
        search_path=search_path,
        total_time=total_time,
    )


def binary_search_with_jumps(
    runs: List[Tuple[int, float, float]],  # (control_points, error, time) sorted by control_points
    threshold: float,
    min_ctrl: int,
    jump_size: int = 100,
) -> SearchResult:
    """
    Simulate binary search with initial large jumps.
    
    Strategy:
    1. Start at min_ctrl, jump by jump_size until we find error < threshold
    2. Once found, binary search between last "bad" and first "good" to find exact transition
    
    This stops as soon as we find a good point (like linear), then refines backward.
    
    Args:
        runs: List of (control_points, error, time) tuples, sorted by control_points
        threshold: Error threshold to find (e.g., 1e-5)
        min_ctrl: Minimum control points to start from
        jump_size: Initial jump size (default 100)
        
    Returns:
        SearchResult with number of fits required and total time
    """
    # Build lookup dicts and get available control point values
    error_by_ctrl = {ctrl: err for ctrl, err, _ in runs}
    time_by_ctrl = {ctrl: t for ctrl, _, t in runs}
    available_ctrls = sorted(error_by_ctrl.keys())
    max_ctrl = max(available_ctrls)
    min_available = min(available_ctrls)
    
    search_path = []
    tested = set()
    total_time = 0.0
    
    def test(ctrl: int) -> Tuple[int, Optional[float]]:
        """Test a control point count and record it. Returns (actual_ctrl, error)."""
        nonlocal total_time
        # Clamp to available range
        if ctrl < min_available:
            ctrl = min_available
        elif ctrl > max_ctrl:
            ctrl = max_ctrl
        # Find exact match or skip
        if ctrl not in error_by_ctrl:
            return ctrl, None
        if ctrl not in tested:
            tested.add(ctrl)
            search_path.append(ctrl)
            total_time += time_by_ctrl[ctrl]
        return ctrl, error_by_ctrl[ctrl]
    
    # Phase 1: Jump by jump_size until we find a good point (or exhaust range)
    current = max(min_ctrl, min_available)
    last_bad = None  # Last point where error >= threshold
    first_good = None  # First point where error < threshold
    
    while current <= max_ctrl:
        actual, err = test(current)
        if err is not None:
            if err < threshold:
                first_good = actual
                break  # Found a good point, stop jumping
            else:
                last_bad = actual
        current += jump_size
    
    # If we didn't find any good point in jumps, threshold not reachable
    if first_good is None:
        # Try max as last resort
        if max_ctrl not in tested:
            actual, err = test(max_ctrl)
            if err is not None and err < threshold:
                first_good = actual
        
        if first_good is None:
            return SearchResult(
                found=False,
                control_points=None,
                fits_required=len(search_path),
                search_path=search_path,
                total_time=total_time,
            )
    
    # If first test was already good, or no bad point before it
    if last_bad is None:
        last_bad = min_available
    
    # Phase 2: Binary search between last_bad and first_good to find exact transition
    low = last_bad
    high = first_good
    
    while high - low > 1:
        mid = (low + high) // 2
        
        actual, err = test(mid)
        
        if err is None:
            # No data at this point, find nearest available
            candidates = [c for c in available_ctrls if low < c < high and c not in tested]
            if not candidates:
                break
            mid = candidates[len(candidates) // 2]
            actual, err = test(mid)
            if err is None:
                break
        
        if err < threshold:
            high = actual
        else:
            low = actual
    
    # high is now the first control point where error < threshold
    return SearchResult(
        found=True,
        control_points=high,
        fits_required=len(search_path),
        search_path=search_path,
        total_time=total_time,
    )


def analyze_project(data: Dict[str, Any], threshold: float, jump_sizes: List[int] = None) -> Dict[str, Any]:
    """
    Analyze all run groups in a project.
    
    Groups runs by: optimizer -> degree -> sections -> continuity
    For each group, simulates linear search and binary search with multiple jump sizes.
    
    Args:
        data: Project data dict
        threshold: Error threshold
        jump_sizes: List of jump sizes to test (default: [10, 25, 50, 100, 150, 200])
    """
    if jump_sizes is None:
        jump_sizes = [10, 25, 50, 100, 150, 200]
    
    runs = data.get("optimization_runs", [])
    
    # Group runs
    groups = defaultdict(list)
    for run in runs:
        # Tolerance-mode runs choose their own count: not part of a sweep
        if run["optimization_config"].get("tolerance") is not None:
            continue
        opt = run["optimization_config"]["optimizer"]
        cfg = run["spline_config"]
        degree = cfg["degree"]
        sections = cfg.get("num_sections", 2)
        continuity = cfg["continuity"]
        # Projects before geometric_continuity stored continuity_type
        cont_type = ("GEOMETRIC" if cfg.get("geometric_continuity")
                     else cfg.get("continuity_type", "PARAMETRIC"))
        
        key = (opt, degree, sections, continuity, cont_type)
        ctrl = cfg["num_control_points"]
        error = run["result"]["final_error"]
        elapsed = run["result"].get("elapsed_seconds", 0.0)
        groups[key].append((ctrl, error, elapsed))
    
    results = []
    
    for key, run_data in sorted(groups.items()):
        opt, degree, sections, continuity, cont_type = key
        
        # Sort by control points
        run_data.sort(key=lambda x: x[0])
        
        # Calculate minimum control points for this config
        # G0=1, G1=2, G2=3, G3=4 constrained points
        cont_level = int(continuity[1]) if len(continuity) > 1 else 1
        min_ctrl = degree + 1 + cont_level
        
        # Run both strategies
        linear_result = linear_search(run_data, threshold, min_ctrl)
        
        # Test multiple jump sizes for binary search
        binary_results = {}
        for js in jump_sizes:
            binary_results[js] = binary_search_with_jumps(run_data, threshold, min_ctrl, jump_size=js)
        
        # Find best binary result by time (if found) or fits
        best_js = min(jump_sizes, key=lambda js: (
            not binary_results[js].found,  # Prefer found
            binary_results[js].total_time if binary_results[js].found else float('inf')
        ))
        best_binary = binary_results[best_js]
        
        # Find actual minimum (ground truth)
        actual_min = None
        for ctrl, err, _ in run_data:
            if err < threshold:
                actual_min = ctrl
                break
        
        results.append({
            "optimizer": opt,
            "degree": degree,
            "sections": sections,
            "continuity": f"{continuity} ({cont_type[0]})",
            "num_runs": len(run_data),
            "ctrl_range": f"{run_data[0][0]}-{run_data[-1][0]}",
            "actual_minimum": actual_min,
            "linear": {
                "found": linear_result.found,
                "control_points": linear_result.control_points,
                "fits_required": linear_result.fits_required,
                "total_time": linear_result.total_time,
            },
            "binary_by_jump": {
                js: {
                    "found": binary_results[js].found,
                    "control_points": binary_results[js].control_points,
                    "fits_required": binary_results[js].fits_required,
                    "total_time": binary_results[js].total_time,
                }
                for js in jump_sizes
            },
            "best_jump_size": best_js,
            "best_binary": {
                "found": best_binary.found,
                "control_points": best_binary.control_points,
                "fits_required": best_binary.fits_required,
                "total_time": best_binary.total_time,
            },
            "speedup": (
                linear_result.fits_required / best_binary.fits_required
                if best_binary.fits_required > 0 else 0
            ),
            "time_speedup": (
                linear_result.total_time / best_binary.total_time
                if best_binary.total_time > 0 else 0
            ),
        })
    
    return {"results": results, "jump_sizes": jump_sizes}


def format_time(seconds: float) -> str:
    """Format time in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_results(analysis: Dict[str, Any], threshold: float):
    """Print analysis results in a formatted table."""
    results = analysis["results"]
    jump_sizes = analysis["jump_sizes"]
    
    print(f"\n{'='*140}")
    print(f"SEARCH STRATEGY COMPARISON (threshold: {threshold:.0e})")
    print(f"Jump sizes tested: {jump_sizes}")
    print(f"{'='*140}")
    
    # Header
    print(f"\n{'Config':<40} {'Runs':>5} {'Min':>6} {'Linear':>8} {'Lin Time':>10} {'Best JS':>7} {'Binary':>8} {'Bin Time':>10} {'Fit Spd':>7} {'Time Spd':>8}")
    print("-" * 140)
    
    total_linear_fits = 0
    total_binary_fits = 0
    total_linear_time = 0.0
    total_binary_time = 0.0
    found_count = 0
    best_js_counts = defaultdict(int)
    
    for r in results:
        config = f"{r['optimizer'][:15]}, d{r['degree']}, {r['sections']}s, {r['continuity']}"
        
        actual = str(r['actual_minimum']) if r['actual_minimum'] else "N/A"
        linear_fits = r['linear']['fits_required']
        binary_fits = r['best_binary']['fits_required']
        linear_time = r['linear']['total_time']
        binary_time = r['best_binary']['total_time']
        best_js = r['best_jump_size']
        fit_speedup = r['speedup']
        time_speedup = r['time_speedup']
        
        if r['linear']['found']:
            total_linear_fits += linear_fits
            total_binary_fits += binary_fits
            total_linear_time += linear_time
            total_binary_time += binary_time
            found_count += 1
            best_js_counts[best_js] += 1
            fit_speedup_str = f"{fit_speedup:.1f}x"
            time_speedup_str = f"{time_speedup:.1f}x"
        else:
            fit_speedup_str = "N/A"
            time_speedup_str = "N/A"
        
        print(f"{config:<40} {r['num_runs']:>5} {actual:>6} {linear_fits:>8} {format_time(linear_time):>10} {best_js:>7} {binary_fits:>8} {format_time(binary_time):>10} {fit_speedup_str:>7} {time_speedup_str:>8}")
    
    print("-" * 140)
    
    if found_count > 0:
        avg_fit_speedup = total_linear_fits / total_binary_fits if total_binary_fits > 0 else 0
        avg_time_speedup = total_linear_time / total_binary_time if total_binary_time > 0 else 0
        print(f"{'TOTAL (where threshold found)':<40} {found_count:>5} {'':<6} {total_linear_fits:>8} {format_time(total_linear_time):>10} {'':<7} {total_binary_fits:>8} {format_time(total_binary_time):>10} {avg_fit_speedup:.1f}x {avg_time_speedup:>7.1f}x")
    
    # Jump size comparison table
    print(f"\n{'='*140}")
    print("JUMP SIZE COMPARISON (time for groups where threshold was found)")
    print(f"{'='*140}")
    
    # Build header for jump size columns
    js_header = "Config".ljust(40) + "Linear".rjust(10)
    for js in jump_sizes:
        js_header += f"JS={js}".rjust(10)
    js_header += "Best".rjust(8)
    print(js_header)
    print("-" * 140)
    
    js_totals = {js: 0.0 for js in jump_sizes}
    
    for r in results:
        if not r['linear']['found']:
            continue
            
        config = f"{r['optimizer'][:15]}, d{r['degree']}, {r['sections']}s, {r['continuity']}"
        row = config.ljust(40) + format_time(r['linear']['total_time']).rjust(10)
        
        best_time = float('inf')
        best_js = None
        for js in jump_sizes:
            t = r['binary_by_jump'][js]['total_time']
            if r['binary_by_jump'][js]['found'] and t < best_time:
                best_time = t
                best_js = js
            js_totals[js] += t
            row += format_time(t).rjust(10)
        
        row += f"{best_js}".rjust(8) if best_js else "N/A".rjust(8)
        print(row)
    
    print("-" * 140)
    totals_row = "TOTAL".ljust(40) + format_time(total_linear_time).rjust(10)
    for js in jump_sizes:
        totals_row += format_time(js_totals[js]).rjust(10)
    print(totals_row)
    
    # Summary
    print(f"\n{'='*140}")
    print("SUMMARY")
    print(f"{'='*140}")
    print(f"  Groups analyzed: {len(results)}")
    print(f"  Groups where threshold reached: {found_count}")
    
    if found_count > 0:
        print(f"\n  Best jump size frequency:")
        for js in sorted(best_js_counts.keys()):
            print(f"    JS={js}: {best_js_counts[js]} groups")
        
        # Find overall best jump size by total time
        best_overall_js = min(jump_sizes, key=lambda js: js_totals[js])
        print(f"\n  Best overall jump size by total time: {best_overall_js} ({format_time(js_totals[best_overall_js])})")
        
        print(f"\n  Linear vs Best Binary:")
        print(f"    Linear total time: {format_time(total_linear_time)} ({total_linear_time:.1f}s)")
        print(f"    Best binary total time: {format_time(total_binary_time)} ({total_binary_time:.1f}s)")
        time_diff = total_linear_time - total_binary_time
        if time_diff > 0:
            print(f"    Time saved with binary: {format_time(time_diff)}")
        else:
            print(f"    Linear is faster by: {format_time(-time_diff)}")


def plot_jump_size_analysis(analysis: Dict[str, Any], threshold: float, output_path: Optional[Path] = None):
    """Plot total time vs jump size for each optimizer in separate subplots."""
    results = analysis["results"]
    jump_sizes = analysis["jump_sizes"]
    
    # Collect data per optimizer
    optimizer_data = defaultdict(lambda: {"js_times": defaultdict(float), "linear_time": 0.0})
    
    for r in results:
        if not r['linear']['found']:
            continue
        
        opt = r['optimizer']
        optimizer_data[opt]["linear_time"] += r['linear']['total_time']
        
        for js in jump_sizes:
            optimizer_data[opt]["js_times"][js] += r['binary_by_jump'][js]['total_time']
    
    # Calculate totals
    total_linear = sum(d["linear_time"] for d in optimizer_data.values())
    total_by_js = {js: sum(d["js_times"][js] for d in optimizer_data.values()) for js in jump_sizes}
    
    # Create figure with subplots - one for total + one per optimizer
    num_optimizers = len(optimizer_data)
    num_plots = 1 + num_optimizers  # Total + each optimizer
    
    # 16:9 aspect ratio
    fig_width = 16
    fig_height = 9
    
    # Arrange subplots - prefer 2 rows
    if num_plots <= 3:
        ncols = num_plots
        nrows = 1
    else:
        ncols = (num_plots + 1) // 2
        nrows = 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes = np.array(axes).flatten()  # Flatten for easy indexing
    
    js_array = np.array(jump_sizes)
    
    # Plot 1: Total time by jump size
    ax = axes[0]
    times_array = np.array([total_by_js[js] / 60 for js in jump_sizes])  # Convert to minutes
    
    ax.plot(js_array, times_array, '#00BFFF', linewidth=1.5, markersize=3, label='Binary search')
    ax.axhline(y=total_linear / 60, color='#FF6B6B', linestyle='--', linewidth=2, label='Linear search')
    
    # Mark minimum
    min_idx = np.argmin(times_array)
    min_js = jump_sizes[min_idx]
    min_time = times_array[min_idx]
    ax.scatter([min_js], [min_time], color='#00FF7F', s=100, zorder=5, marker='*', label=f'Best: JS={min_js}')
    
    ax.set_xlabel('Jump Size', fontsize=10)
    ax.set_ylabel('Total Time (minutes)', fontsize=10)
    ax.set_title(f'TOTAL (threshold: {threshold:.0e})', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(jump_sizes) + 10)
    
    # Plot each optimizer in its own subplot
    colors = plt.cm.tab10(np.linspace(0, 1, num_optimizers))
    
    for idx, ((opt, data), color) in enumerate(zip(optimizer_data.items(), colors)):
        ax = axes[idx + 1]
        times = np.array([data["js_times"][js] / 60 for js in jump_sizes])
        linear_time = data["linear_time"] / 60
        
        ax.plot(js_array, times, '-', color='#00BFFF', linewidth=1.5, label='Binary search')
        ax.axhline(y=linear_time, color='#FF6B6B', linestyle='--', linewidth=2, label='Linear search')
        
        # Mark minimum
        min_idx = np.argmin(times)
        min_js = jump_sizes[min_idx]
        min_time = times[min_idx]
        ax.scatter([min_js], [min_time], color='#00FF7F', s=100, zorder=5, marker='*', label=f'Best: JS={min_js}')
        
        # Calculate speedup
        speedup = linear_time / min_time if min_time > 0 else 0
        
        ax.set_xlabel('Jump Size', fontsize=10)
        ax.set_ylabel('Time (minutes)', fontsize=10)
        ax.set_title(f'{opt}\n(best JS={min_js}, {speedup:.1f}x faster)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(jump_sizes) + 10)
    
    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Default output path if not specified
    if output_path is None:
        output_path = Path("jump_size_analysis.png")
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close(fig)


def main():
    # Project file from the command line
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(1)
    project_path = Path(args[0])
    threshold = 1e-5
    if "--threshold" in args:
        threshold = float(args[args.index("--threshold") + 1])
    
    if not project_path.exists():
        print(f"Error: File not found: {project_path}")
        sys.exit(1)
    
    print(f"Loading: {project_path}")
    with open(project_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Analyzing {len(data.get('optimization_runs', []))} runs...")
    
    # Use fine grid of jump sizes
    jump_sizes = list(range(1, 201, 1))  # 1, 2, 3, ..., 200
    
    analysis = analyze_project(data, threshold, jump_sizes)
    print_results(analysis, threshold)
    
    # Save plot
    output_path = project_path.with_name(project_path.stem + "_jump_size_analysis.png")
    plot_jump_size_analysis(analysis, threshold, output_path)


if __name__ == "__main__":
    main()
