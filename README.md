# Spline Fitter

B-spline fitting of airfoil coordinates for CAD: the fewest control points
that keep every point within a tolerance, with geometric G1-G3 continuity
where sections meet, written exactly to STEP, IGES or DXF.

## Features

- **Tolerance mode** (VarPro LM): the fewest control points, with non-uniform
  knots, whose curve stays within a maximum orthogonal deviation of every
  point. The automatic tolerance follows the file's decimals (or the noise
  level of scanned data).
- **Continuity per junction**: parametric C0-C3 or geometric G1-G3, one
  setting for all junctions or a list such as `G2, C1, G3`.
- **Whole-chain fitting**: the sections are fitted to all points at once. The
  split only sets where the junctions start; points move between sections as
  the curve fits them, and with Optimize positions the junctions move too.
- **Two methods**: VarPro Levenberg-Marquardt (variable projection with the
  exact reduced Jacobian; tolerance search, free knots as unknowns of the
  same loop, geometric continuity) and Direct LSQ (a single constrained
  least-squares solve).
- **CAD export**: STEP AP214, IGES 5.3 and DXF R2000, per section or merged
  into one curve, optionally closing a blunt trailing edge.
- **Fit report**: maximum and RMS deviation, control points and knots,
  curvature extrema, junction continuity, leading-edge radius, maximum
  thickness and camber with a chart along the chord.
- **Plots**: knots, a curvature comb (κ to d³κ/ds³) and the deviation of every
  point magnified along the normal with the tolerance band, live while a fit
  runs; convergence; and a comparison of the project's runs along any column.
- **Projects**: every fit is kept as a run in an `.asfproj` file (standard
  JSON), each refit of a tolerance search too; the explorer filters, sorts
  and pivots the runs and applies a run's settings.
- **Sweeps** of degree, control points, sections and smoothing, and **batch
  conversion** of a folder of airfoils.
- Reads Selig and Lednicer files in many number formats (decimal commas,
  Fortran exponents, fixed-width columns, 0-100 scaling), keeping the file's
  chord line.

## Building and running

Requires a Rust toolchain (1.80 or later).

```bash
cargo run --release
```

The binary is `target/release/spline-fitter`. Tests:

```bash
cargo test --workspace
```

The free-knot and tolerance-search reference fits take from seconds to
minutes and are ignored by default:

```bash
cargo test --release -p bspline-varpro -- --ignored
```

Building the workspace needs Python 3.9 or later installed, for the Python
module's crate.

## Python

The fitting library is also a Python module, `bspline_varpro`, built from
the same Rust code ([`crates/bspline-varpro-py`](crates/bspline-varpro-py)).
Install it into the active Python with [maturin](https://www.maturin.rs):

```bash
pip install maturin
cd crates/bspline-varpro-py
maturin develop --release
```

or build a wheel with `maturin build --release`. One wheel serves Python
3.9 and every later version.

```python
import bspline_varpro as bv

name, points, decimals = bv.read_airfoil("naca2412.dat")
fit = bv.fit_airfoil(points, decimals=decimals)   # as the app fits it
print(fit)                       # control points, RMS error, max deviation
report = fit.measure(points)     # deviation, curvature, junctions, LE radius,
                                 # max thickness and camber
fit.export("naca2412.step", scale=100.0, units="mm", merge=True)
```

| Function | Does |
|---|---|
| `read_airfoil(path)` | `(name, points, decimals)` of a Selig or Lednicer file |
| `naca4(code, n_per_side=81)` | Points of a NACA 4-digit section |
| `split_sections(points, count=2)` | Sections sharing their junction points |
| `default_tolerance(points, decimals=None)` | The tolerance the data support |
| `fit_airfoil(points, tolerance=None, ...)` | The app's tolerance search with its defaults |
| `fit(sections, control_points, ...)` | A fit with fixed control-point counts |
| `fit_to_tolerance(sections, tolerance, ...)` | The library's free-knot tolerance search |

A `Fit` holds `control_points`, `knots`, `degree`, `rms_error`,
`max_deviation`, `success` and `message`, and has `evaluate(u)`,
`resample(n)`, `deviations(points)`, `measure(points)` and
`export(path, ...)`. Points come back as NumPy arrays; `continuity` is one
spec for all junctions (`"G2"`) or one per junction (`["G2", "C1"]` or
`"G2, C1"`). Tests: `pytest crates/bspline-varpro-py/tests` after
`maturin develop`.

## Workspace

| Crate | Purpose |
|---|---|
| [`bspline-varpro`](crates/bspline-varpro) | The fitting library: knots and basis, variable projection, Levenberg-Marquardt, tolerance search, free knots, diagnostics, airfoil helpers, STEP/IGES/DXF export. No GUI. |
| [`spline-fitter`](crates/spline-fitter) | The application core: airfoil and spline files, section splitting, fit settings, the VarPro and Direct LSQ adapters, the fit report, project files, batch naming, optimizer logs. |
| [`spline-fitter-gui`](crates/spline-fitter-gui) | The desktop application (egui): sidebar, plots, explorer, dialogs. |
| [`bspline-varpro-py`](crates/bspline-varpro-py) | The Python module `bspline_varpro` (PyO3): reading, fitting, measuring and exporting from scripts and notebooks. |

## Spline file format

`.spline.dat` files start with the airfoil name, followed by comment lines
that describe the curves exactly:

```
NACA 2412
# spline-fitter B-spline v2
# degree 5
# sections 2
# continuity G2
# section 1 control_points 13
# knots 1 0.0 0.0 0.0 0.0 0.0 0.0 0.259864810632751 ... 1.0 1.0 1.0 1.0 1.0 1.0
# section 2 control_points 12
# knots 2 0.0 0.0 0.0 0.0 0.0 0.0 0.43274513071260046 ... 1.0 1.0 1.0 1.0 1.0 1.0
1.000000000000 0.000000000000
...
```

The coordinate lines hold the control points of all sections in order, each
shared junction point once, so tools that only read coordinates still work.
Older files without the header load as one section on uniform knots.

## Behaviour

[docs/spec.md](docs/spec.md) specifies the behaviour in full: algorithms,
constants, file formats, messages and the window, with the design changes
D1 to D24 of its revision 3. It was written for the earlier Python version,
which this workspace replaces. The Rust version follows it, reproduces its
reference fits and file outputs (the tests in `crates/*/tests` hold the
Python results), and differs on purpose where the behaviour was a defect:

- Progress goes through typed observers, not process-global counters and
  callback signature inspection; a stopped fit is a flag, not a message
  comparison.
- Direct LSQ builds its junction constraints with the library's chain
  constraints instead of a second hand-written copy.
- Changes the window makes as one action (applying a run, the best values
  of a sweep) are one undo step because the settings settle once per frame,
  not through an explicit history group.
- The fit report is a native table, not HTML.
- The deviation overlay draws every whisker outward, its length the size of
  the deviation, in sky blue for a point inside the curve and rose for one
  outside; Python drew signed whiskers on both sides in the section colors.
  The tolerance is one dashed line outside the curve, always at the height
  setting, not one on each side.
- The live deviation searches the foot points on each step's curve instead
  of refining the previous step's: the solver's steps move a point's
  parameter anywhere on the curve, and the refined feet then joined points to
  the wrong part of it (whiskers across the airfoil).
- Tolerance fits use the sidebar's Smoothing weight (Python used a fixed
  1e-7 and greyed the field out), and it can be swept in tolerance mode.
  Project files are format 1.1.0; tolerance runs of older files are loaded
  with the 1e-7 they were fitted with.
- A tolerance search is a series of fixed-count fits, each exactly the fit
  a single run of that count makes (fresh, from the same start, the same
  count in every section, with free knots too): from the start count one
  more at a time until a fit meets the tolerance, or, when the start count
  meets it, one fewer at a time while they still do. Double-clicking one of
  its runs sets the sidebar to that fixed-count run, which fits the same
  curve again. Python started each refit from the one before and added
  control points where the fit was worst (one per knot span beyond the
  tolerance, so the count nearly doubled each round: 6, 7, 9, 13, 21, ...),
  and no single run could repeat its fits. Measured on one section from 6
  with uniform knots and smoothing 0.001, the fresh fits end within 4
  control points of a warm-started search and as fast (HM51 at 1e-4: 28
  control points in 0.9 s; Darkstar root at 1e-5: 67, against 65). With
  smoothing 1e-7 fresh fits can need more (HM51 at 1e-5: 48, where warm
  starts reached 30).
- A tolerance search is a folder in the run table, "Search 1", "Search
  2", ...: its row shows the search's settings and the fit it ended on
  (Control points and Max deviation of that fit; Iterations counts its fits,
  Converged says whether it met the tolerance). The chevron, or Right and
  Left, opens and closes it; inside are its runs, plain fixed-count runs.
  Double-clicking the search's row sets the sidebar back to the tolerance
  search; Delete on it deletes it with its runs. A new search of the same
  settings replaces it. Project files are format 1.2.0: searches are under
  `tolerance_searches`, each run names its search (`search`); readers of
  format 1.0 see the runs as plain fixed-count runs.
- The run table has one Control points column: per section, of the fit
  (a search: of the fit it ended on). Python showed the count as set (for
  a tolerance run, its start count) and the fitted count of the whole
  curve in a second column, Fitted control points, which is gone.
- A tolerance can limit the largest deviation (as in Python) or the RMS
  deviation: Target in the sidebar, under Goal. The field below is named
  for it, and the search, the report ("RMS tolerance"), the readout under
  Fit spline, the run table's Target column and the runs plot's tolerance
  line follow it. The deviation overlay draws its tolerance line only for
  the largest deviation, a bound for every point. Projects and batch file
  names record an RMS target (`tolerance_metric`, `.rms`); others keep the
  layout of format 1.0.
- A tolerance search runs until a fit meets the tolerance, up to 200 fits
  (Python: 40 rounds) or as many control points as a section's points
  allow; one that ends without meeting it keeps the closest fit and names
  where that fit is farthest from the data. There is no rule that stops a
  search whose deviation stays level: fresh fits stay level for many counts
  and then drop (Darkstar tip at 1e-5, smoothing 1e-7: about 3e-5 from 19
  to 57 control points, within the tolerance at 64), so such a rule ended
  searches that would have met it. Esc cancels a search that takes too
  long and keeps the closest fit so far.
- With sections of different counts the control-point readout gives the
  curve's count, each junction point once, as the status text and the runs
  plot do; Python added the sections' counts, so it read 13 where the status
  said 12.
- A point in the runs plot names its run on hover, and a click selects that
  run, shows its fit and scrolls its row into view (in Python the points
  did nothing): after a tolerance search, click any count to see that fit
  (its search opens to show the run).
- While a tolerance search runs, the runs plot follows it: a point per
  finished fit (control points against max deviation), the latest one
  marked, under the tolerance line. Python showed the search's runs only
  at its end.
- The runs plot draws the tolerance as a dashed line when its y axis is the
  metric the tolerance limits and every plotted run belongs to a search of
  the same tolerance (or is a tolerance run of an older project).
- The run table has no Error column. Python's Error was the solver's error
  for a fixed-count run and the largest deviation for a tolerance run, so
  it compared different things; the runs are compared by Max deviation and
  RMS deviation. The runs plot plots the RMS deviation by default, a number
  in the filter (">1e-5") compares it, and the solver's error is called
  RMS error wherever it shows (status bar, convergence plot, run details).
- A Curvature points toggle beside Curvature comb marks on the curve the
  curvature extrema the fit report counts, each a dot in its section's comb
  colour, and the inflections (where the curvature changes sign, as on a
  reflexed or concave surface), each a ring in that colour (kept between
  sessions); Python only gave the number of extrema.
- Every plot autoscales as pyqtgraph's do: it follows its data until you
  zoom or pan; then a button in its lower left corner (the corners of a
  frame, where pyqtgraph shows "A") or a double-click shows all the data
  again and follows it. Before, a double-click only set the airfoil and
  runs plots to follow without fitting them, so only the convergence and
  shape plots rescaled.
- Ctrl+Enter fits (also from a field you are typing in: the typed value
  counts), Esc cancels a running fit; Python had no keys for either.
- The sidebar hides to a rail as the explorer does (the chevron on its Data
  caption, View > Sidebar or Ctrl+B), and both panels keep their width and
  whether they are shown between sessions.
- Pivot is a small icon toggle beside Invert and stays on or off between
  sessions.
- Unsaved runs, replacing a project file and deleting runs ask in a dialog
  of the app, in its theme, instead of the system's message box.
- The fit report is not a window: the bottom-left block has tabs for the
  convergence plot, the report and the thickness and camber chart of the
  shown fit (View > Fit report or Ctrl+R picks Report), and the chosen tab
  is kept between sessions. Copy puts the report on the clipboard as tab
  separated text, which pastes into a spreadsheet as cells.
- The empty window says how to start: the main plot names the ways to load
  an airfoil (with their keys), the empty run table what it will hold.
- A fit that missed its tolerance shows its Max deviation readout in the
  warning colour, with the tolerance in the tooltip.
- The plot toggles (Knots, Curvature comb, Deviation, with their settings),
  Live preview, the convergence axes and the runs plot's columns are kept
  between sessions, and so are the sidebar's fit settings (sweeps start
  off).
- Failures (a file not found, a fit that failed, a save that failed) show
  in the status bar in the error colour, as the error chime plays.
- The explorer keeps the column order you drag its headers into between
  sessions (Python started from the default order every time).
- Each run of a tolerance search keeps the loss history and the time of
  its own fit, as the run would alone; Python kept the history from the
  search's start in each, which made a project several megabytes per
  search.
- The runs plot ticks only whole numbers when every run's x value is one
  (control points, sections, degree); pyqtgraph also ticked 25.2, 25.4, ...
- A search that misses its tolerance ends its status with its own message
  ("Tolerance not reached: ..."); Python put "Solver did not converge:"
  before it, though every fit of the search converged.
- A tolerance fit of a section with too few points names the section and
  the counts ("Section 1 has 4 points, too few for its end conditions: they
  need 6 control points, so at least 7 points.").
- Enter in a dialog presses its accept button (or the push button that has
  the focus). In the Python window the segmented choices were push buttons
  that took the default: Enter chose "Closed" in New NACA airfoil and "mm"
  in Export CAD, and reset Scrolling and zooming.
- Library features nothing used (tangent end conditions, averaged knots, knot
  insertion, the second airfoil reader, all-finite-difference Jacobians) and the
  retired settings (`smoothing_factor`, `early_stop_patience`,
  `relative_tolerance`, `custom_parameters`) are gone; projects that hold
  them still open.

The GUI follows the **Loft** design language,
[docs/design-language.md](docs/design-language.md).

## License

MIT License
