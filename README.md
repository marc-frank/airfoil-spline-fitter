# Spline Fitter

Professional B-spline curve fitting tool for airfoil coordinate data.

## Features

- **Interactive GUI** for real-time spline fitting visualization
- **Tolerance mode** (VarPro LM): the fewest control points, with non-uniform
  knots, whose curve stays within a maximum deviation of every point. The
  default tolerance follows the file's precision (or its noise level for
  scans).
- **Continuity per junction**: parametric C0-C3 or geometric G1-G3, one setting
  for all junctions or a list such as `G2, C1, G3` for any number of sections
- **CAD export**: exact STEP AP214, IGES 5.3, and DXF R2000 curves, per section
  or merged into one B-spline, optionally with a closing line at a blunt
  trailing edge
- **Fit report**: maximum and RMS deviation, control points and knots,
  curvature extrema, junction continuity, and leading-edge radius, with a
  warning when the tolerance is below the data's rounding error
- **Knot display**, an interactive curvature comb, and the **deviation** of
  every data point from the curve, magnified along the normal with the
  tolerance band
- **Two fitting methods**: VarPro Levenberg-Marquardt (tolerance mode, free
  knots, geometric continuity) and Direct LSQ (one constrained least-squares
  solve)
- **Batch processing**: convert entire airfoil libraries

VarPro fitting, tolerance search, and the CAD writers come from the
standalone [`bspline-varpro`](packages/bspline-varpro) library, which can be
used without the GUI.

## Installation

```bash
pip install spline-fitter
```

Or for development:

```bash
git clone https://github.com/airfoil-library/spline-fitter.git
cd spline-fitter
pip install -e packages/bspline-varpro
pip install -e ".[dev]"
```

## Usage

### GUI Application

```bash
spline-fitter
# or
python -m spline_fitter
```

To launch directly from this source checkout without installing it:

```bash
python -m src.spline_fitter
```

Typical workflow:

1. **File > Open Airfoil** loads a `.dat` file (Selig order).
2. Choose the **VarPro LM** algorithm. **Tolerance mode** and **Auto**
   tolerance are on by default. Set the number of sections, the degree (5 is
   recommended for CAD), and the continuity. Continuity is either one
   **Smoothness** order with a **Continuity Type** (Geometric or Parametric),
   or a comma-separated list in **Per Junction** with one entry per junction.
   **Knot placement** is Free (knots move to where they reduce the error
   most) or Uniform (evenly spaced). **Grow control points** starts from the
   fewest control points and adds them where the fit is worst; off, the fit
   starts from the Control Points setting (in tolerance mode the count can
   still grow or shrink from there).
3. **Fit Spline**. The status bar shows the maximum and RMS deviation and the
   control-point count. **Show Knots** marks the knots on the curve, and
   **Deviation** shows where the fit departs from the data, also while it
   fits (its arrow sets the height: 5% of the chord at the tolerance).
4. **View > Fit Report** (Ctrl+R) shows deviation, fairness, and junction
   measurements.
5. **File > Export CAD** (Ctrl+E) writes STEP, IGES, or DXF with units and a
   scale (for example the chord length). **Save Fitted** writes a resampled
   `.dat` with cosine spacing, and **Save Spline** writes control points with
   their knot vectors.

Without tolerance mode, VarPro uses the fixed control-point count and supports
parameter sweeps, like the other optimizers.

### Spline file format

`.spline.dat` files written since version 2 start with the airfoil name,
followed by comment lines that describe the curves exactly:

```
NACA 2412
# spline-fitter B-spline v2
# degree 5
# sections 2
# continuity G2
# section 1 control_points 13
# knots 1 0.0 0.0 0.0 0.0 0.0 0.0 0.259864810632751 0.3467311126423559 ... 1.0 1.0 1.0 1.0 1.0 1.0
# section 2 control_points 12
# knots 2 0.0 0.0 0.0 0.0 0.0 0.0 0.43274513071260046 ... 1.0 1.0 1.0 1.0 1.0 1.0
1.000000000000 0.000000000000
...
```

Sections are numbered from 1, and knots are written at full double precision.

The coordinate lines list the control points of all sections in order, with
each shared junction point written once, so tools that only read coordinates
still work. `spline_fitter.io.parse_spline_data` reads both this format and
older files, which had uniform knots and no header.

### Programmatic API

```python
from spline_fitter import (
    OptimizationConfig, OptimizerType, SplineConfig, FairnessWeights,
    create_airfoil_data, run_optimization,
)
from spline_fitter.io import parse_airfoil_data

filename, name, coords, precision = parse_airfoil_data("naca2412.dat")
airfoil = create_airfoil_data(coords, filename=filename, name=name, precision=precision)
result = run_optimization(
    SplineConfig(degree=5, geometric_continuity=True),          # G2 at the leading edge
    OptimizationConfig(optimizer=OptimizerType.GN_VARPRO, tolerance=1e-5),
    airfoil, FairnessWeights(), [],
)
result.section_control_points, result.section_knots
```

For scripts that do not need the GUI's data model, use `bspline-varpro`
directly (`bspline_varpro.airfoil.fit_airfoil`, `bspline_varpro.export.write`).

### Retired: Hierarchical LSQ

The Hierarchical LSQ optimizer has been removed. Its foot-point projection
could produce non-monotone parameters and fail, and tolerance mode covers its
purpose (adding control points where the error is large). Projects that
contain Hierarchical LSQ runs still open, and those runs load as VarPro LM
runs.

### Retired: Constrained LSQ, Orthogonal LSQ, Param Optimizer

Constrained LSQ was Direct LSQ with arc-length parameters; Direct LSQ now has
a Parameterization setting (centripetal or arc length) and the free-junction
option. Orthogonal LSQ and the two-stage Param Optimizer refined the data
parameters around an LSQ solve, which VarPro LM does better.

### Retired: general-purpose optimizers

The scipy optimizers (BFGS, SLSQP, L-BFGS-B, CG, TNC, Powell, Nelder-Mead,
COBYLA) and the global ones (Differential evolution, Dual annealing,
Basin-hopping) have been removed, with the settings only they read: error
metric, fairness penalties, iteration limit, seed and search scale. They
minimized a sampled error over raw control-point coordinates, which VarPro LM
and Direct LSQ solve directly and much faster. Projects that contain their
runs still open, and those runs load as VarPro LM runs.

## Architecture

The package is organized into focused modules:

| Module | Purpose |
| ------ | ------- |
| `optimization/models.py` | Data structures (AirfoilData, SplineConfig, FitResult, etc.) |
| `optimization/engine.py` | Optimizer dispatch (`run_optimization`) |
| `optimization/run_log.py` | Optimizer progress CSVs (`optimizer_logs/`, or `SPLINE_FITTER_LOG_DIR`) |
| `optimization/optimizers/` | Optimizer adapters, including `gn_varpro.py` for bspline-varpro |
| `spline.py` | B-spline mathematics and curve evaluation |
| `geometry.py` | Coordinate processing and geometric utilities |
| `analysis.py` | Fit report: deviation, curvature, junctions, leading-edge radius |
| `fairness.py` | Smoothness metrics and regularization |
| `gui/` | PyQt6 graphical user interface (`ui.py` window, `layout.py` widgets, `file_actions.py` file menu and export, `theme.py` colors and styles) |
| `io.py` | Airfoil and spline file parsing and export |
| `project_io.py` | Project files (`.splineproj`) |
| `formatting.py` | Display formatting utilities |

The GUI's look follows the **Loft** design language, described in
[docs/design-language.md](docs/design-language.md) and implemented in
`gui/theme.py`. New GUI code takes its colors, fonts and spacing from there;
`tests/test_design_language.py` checks it.

## License

MIT License
