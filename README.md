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

## Workspace

| Crate | Purpose |
|---|---|
| [`bspline-varpro`](crates/bspline-varpro) | The fitting library: knots and basis, variable projection, Levenberg-Marquardt, tolerance search, free knots, diagnostics, airfoil helpers, STEP/IGES/DXF export. No GUI. |
| [`spline-fitter`](crates/spline-fitter) | The application core: airfoil and spline files, section splitting, fit settings, the VarPro and Direct LSQ adapters, the fit report, project files, batch naming, optimizer logs. |
| [`spline-fitter-gui`](crates/spline-fitter-gui) | The desktop application (egui): sidebar, plots, explorer, dialogs. |

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
- A free-knot tolerance search that stops because the worst knot spans are
  already as narrow as the knot spacing allows says so ("the worst knot
  spans are as narrow as the knot spacing 0.05 allows") instead of Python's
  "no knot span can be refined further".
- A tolerance search adds one control point a round to each section beyond
  the tolerance (with free knots: one knot, in its worst span); Python added
  one per knot span holding a point beyond it, so the count nearly doubled
  each round (6, 7, 9, 13, 21, 37, ...) and jumped past counts that meet the
  tolerance. The fits no longer depend on the tolerance, only where the
  search stops, so a looser tolerance never ends with more control points
  than a tighter one: on Darkstar root (one section from 6, uniform) 1e-5
  ends at 30 control points, where Python ended at 35. Uniform knots start
  each refit from the parameters of the closest fit so far, and from the
  start parameters too when that does not get closer: a poor fit's
  parameters otherwise hold the following fits at its deviation.
- A tolerance search whose largest deviation drops by less than a fifth
  while 10 control points are added (or three quarters of the count of the
  last fit that did drop, if more) stops with the closest fit and names the
  point where the deviation stays (a kink or noise in the data); Python kept
  adding control points for all its rounds. The search has 200 rounds
  instead of 40, as each adds one control point.
- With sections of different counts the control-point readout gives the
  curve's count, each junction point once, as the status text and the runs
  plot do; Python added the sections' counts, so it read 13 where the status
  said 12.
- A point in the runs plot names its run on hover, and a click selects that
  run, shows its fit and scrolls its row into view (in Python the points
  did nothing): after a tolerance search, click any count to see that fit.
- The runs plot draws the tolerance as a dashed line when its y axis is the
  error or the max deviation and every plotted run is a tolerance run with
  the same tolerance.
- The explorer keeps the column order you drag its headers into between
  sessions (Python started from the default order every time).
- With uniform knots, a tolerance search that meets the tolerance removes
  control points 1, 2, 4, ... at a time while it still holds, then halves
  the gap to the count that failed; Python removed one at a time and
  stopped at the first count that failed. From a start count of 60 on
  Darkstar root at 1e-5: 31 control points in 6.6 s instead of 41 in 11.9 s.
- The earlier runs of a tolerance search keep the loss history of their own
  refit; Python kept the history from the search's start in each, which
  made a project several megabytes per search. The search's final run
  keeps the whole history.
- Along "Control points" the runs plot places a tolerance run at the count
  its search found, as the sweep plot does; Python placed every tolerance
  run at the start count they share. The column still shows the start count.
- The runs plot ticks only whole numbers when every run's x value is one
  (control points, sections, degree); pyqtgraph also ticked 25.2, 25.4, ...
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
