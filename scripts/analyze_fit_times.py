#!/usr/bin/env python3
"""
Analyze Fit Times vs Control Points
====================================

Plots how long optimization fits take as a function of the number of control points.
This helps understand the computational cost scaling.

Control points are the fitted chain's (junction points counted once), so
tolerance-mode runs, which choose their own count, are placed correctly.

Usage:
    python scripts/analyze_fit_times.py <project.asfproj>
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# VS Code-like dark theme
VSCODE_COLORS = {
    'bg': '#1e1e1e',
    'fg': '#d4d4d4',
    'grid': '#3c3c3c',
    'accent': '#007acc',
}
plt.rcParams.update({
    'figure.facecolor': VSCODE_COLORS['bg'],
    'axes.facecolor': VSCODE_COLORS['bg'],
    'axes.edgecolor': VSCODE_COLORS['fg'],
    'axes.labelcolor': VSCODE_COLORS['fg'],
    'text.color': VSCODE_COLORS['fg'],
    'xtick.color': VSCODE_COLORS['fg'],
    'ytick.color': VSCODE_COLORS['fg'],
    'grid.color': VSCODE_COLORS['grid'],
    'legend.facecolor': '#252526',
    'legend.edgecolor': VSCODE_COLORS['grid'],
})


def load_project(project_path: Path) -> Dict[str, Any]:
    """Load project file."""
    with open(project_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_fit_times(data: Dict[str, Any]) -> Dict[str, List[tuple]]:
    """
    Extract fit times grouped by optimizer.
    
    Returns:
        Dict mapping optimizer name to list of (control_points, time_seconds) tuples
    """
    runs = data.get("optimization_runs", [])
    
    optimizer_data = defaultdict(list)
    
    for run in runs:
        opt = run["optimization_config"]["optimizer"]
        cfg = run["spline_config"]
        sections = run.get("section_control_points")
        if sections:
            ctrl = sum(len(s) for s in sections) - (len(sections) - 1)
        else:
            ctrl = cfg["num_control_points"]
        elapsed = run["result"].get("elapsed_seconds", 0.0)
        
        optimizer_data[opt].append((ctrl, elapsed))
    
    # Sort each optimizer's data by control points
    for opt in optimizer_data:
        optimizer_data[opt].sort(key=lambda x: x[0])
    
    return dict(optimizer_data)


def plot_fit_times(optimizer_data: Dict[str, List[tuple]], output_path: Path):
    """Plot fit time vs control points for each optimizer."""
    
    num_optimizers = len(optimizer_data)
    
    # 16:9 aspect ratio
    fig_width = 16
    fig_height = 9
    
    # Arrange subplots
    if num_optimizers <= 2:
        ncols = num_optimizers
        nrows = 1
    elif num_optimizers <= 4:
        ncols = 2
        nrows = 2
    else:
        ncols = 3
        nrows = (num_optimizers + 2) // 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    if num_optimizers == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    colors = ['#00BFFF', '#FF6B6B', '#00FF7F', '#FFD700', '#FF69B4', '#00CED1', '#FFA500', '#9370DB']
    
    for idx, ((opt, data), color) in enumerate(zip(optimizer_data.items(), colors)):
        ax = axes[idx]
        
        ctrls = np.array([d[0] for d in data])
        times = np.array([d[1] for d in data])
        
        # Convert to appropriate units
        if times.max() > 3600:
            times_plot = times / 3600
            time_unit = "hours"
        elif times.max() > 60:
            times_plot = times / 60
            time_unit = "minutes"
        else:
            times_plot = times
            time_unit = "seconds"
        
        ax.scatter(ctrls, times_plot, color=color, s=20, alpha=0.7)
        
        # Fit a polynomial trend line (quadratic often fits well for optimization)
        has_fit = False
        if len(ctrls) > 3:
            try:
                # Try quadratic fit
                coeffs = np.polyfit(ctrls, times_plot, 2)
                poly = np.poly1d(coeffs)
                x_smooth = np.linspace(ctrls.min(), ctrls.max(), 100)
                ax.plot(x_smooth, poly(x_smooth), '#FF6B6B', linewidth=2, alpha=0.7, label='Quadratic fit')
                has_fit = True
            except:
                pass
        
        # Statistics
        avg_time = times_plot.mean()
        min_time = times_plot.min()
        max_time = times_plot.max()
        
        ax.set_xlabel('Number of Control Points', fontsize=10)
        ax.set_ylabel(f'Fit Time ({time_unit})', fontsize=10)
        ax.set_title(f'{opt}\n({len(data)} runs, avg: {avg_time:.2f} {time_unit})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if has_fit:
            ax.legend(loc='upper left', fontsize=8)
        
        # Add range annotation
        ax.annotate(f'Range: {min_time:.2f} - {max_time:.2f} {time_unit}',
                   xy=(0.98, 0.02), xycoords='axes fraction',
                   ha='right', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='#252526', alpha=0.9, edgecolor='#3c3c3c'))
    
    # Hide unused subplots
    for idx in range(num_optimizers, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Optimization Fit Time vs Number of Control Points', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close(fig)


def plot_combined_fit_times(optimizer_data: Dict[str, List[tuple]], output_path: Path):
    """Plot all optimizers on a single log-scale plot for comparison."""
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    colors = ['#00BFFF', '#FF6B6B', '#00FF7F', '#FFD700', '#FF69B4', '#00CED1', '#FFA500', '#9370DB']
    
    for idx, (opt, data) in enumerate(optimizer_data.items()):
        color = colors[idx % len(colors)]
        ctrls = np.array([d[0] for d in data])
        times = np.array([d[1] for d in data])  # Keep in seconds for log scale
        
        ax.scatter(ctrls, times, color=color, s=30, alpha=0.6, label=opt)
        
        # Fit trend line
        if len(ctrls) > 3:
            try:
                # Use log-space for fitting on log scale
                log_times = np.log10(times + 0.001)  # Avoid log(0)
                coeffs = np.polyfit(ctrls, log_times, 2)
                poly = np.poly1d(coeffs)
                x_smooth = np.linspace(ctrls.min(), ctrls.max(), 100)
                ax.plot(x_smooth, 10**poly(x_smooth), '-', color=color, linewidth=2, alpha=0.7)
            except:
                pass
    
    ax.set_yscale('log')
    ax.set_xlabel('Number of Control Points', fontsize=12)
    ax.set_ylabel('Fit Time (seconds, log scale)', fontsize=12)
    ax.set_title('Optimization Fit Time vs Control Points (All Optimizers)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Combined plot saved to: {output_path}")
    plt.close(fig)


def print_summary(optimizer_data: Dict[str, List[tuple]]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("FIT TIME ANALYSIS SUMMARY")
    print("=" * 80)
    
    for opt, data in sorted(optimizer_data.items()):
        ctrls = [d[0] for d in data]
        times = [d[1] for d in data]
        
        print(f"\n{opt}:")
        print(f"  Runs: {len(data)}")
        print(f"  Control points range: {min(ctrls)} - {max(ctrls)}")
        print(f"  Time range: {min(times):.2f}s - {max(times):.2f}s")
        print(f"  Average time: {sum(times)/len(times):.2f}s")
        print(f"  Total time: {sum(times)/60:.1f} minutes")
        
        # Time per control point bucket
        buckets = defaultdict(list)
        for ctrl, t in data:
            bucket = (ctrl // 50) * 50  # 50-point buckets
            buckets[bucket].append(t)
        
        print(f"  Time by control point range:")
        for bucket in sorted(buckets.keys()):
            avg = sum(buckets[bucket]) / len(buckets[bucket])
            print(f"    {bucket}-{bucket+49}: avg {avg:.2f}s ({len(buckets[bucket])} runs)")


def main():
    # Project file from the command line
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(1)
    project_path = Path(sys.argv[1])
    
    if not project_path.exists():
        print(f"Error: File not found: {project_path}")
        sys.exit(1)
    
    print(f"Loading: {project_path}")
    data = load_project(project_path)
    print(f"Loaded {len(data.get('optimization_runs', []))} runs")
    
    # Analyze
    optimizer_data = analyze_fit_times(data)
    
    # Print summary
    print_summary(optimizer_data)
    
    # Generate plots
    output_dir = project_path.parent
    
    # Individual optimizer plots
    plot_fit_times(optimizer_data, output_dir / "fit_times_by_optimizer.png")
    
    # Combined comparison plot
    plot_combined_fit_times(optimizer_data, output_dir / "fit_times_combined.png")


if __name__ == "__main__":
    main()
