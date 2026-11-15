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
  exact reduced Jacobian; tolerance search, free knots, geometric continuity)
  and Direct LSQ (one constrained least-squares solve and a joint
  refinement).
- **CAD export**: STEP AP214, IGES 5.3 and DXF R2000, per section or merged
  into one curve, optionally closing a blunt trailing edge.
- **Fit report**: maximum and RMS deviation, control points and knots,
  curvature extrema, junction continuity, leading-edge radius, maximum
  thickness and camber with a chart along the chord.
- **Plots**: knots, a curvature comb (κ to d³κ/ds³) and the deviation of every
  point magnified along the normal with the tolerance band, live while a fit
  runs; convergence; and a comparison of the project's runs along any column.
- **Projects**: every fit is kept as a run in an `.asfproj` file; the
  explorer filters, sorts and pivots the runs and applies a run's settings.
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

The tolerance-search reference fit takes about a minute and is ignored by
default:

```bash
cargo test --release -p bspline-varpro -- --ignored
```

## Workspace

| Crate | Purpose |
|---|---|
| [`bspline-varpro`](crates/bspline-varpro) | The fitting library: knots and basis, variable projection, Levenberg-Marquardt, tolerance search, joint refinement, diagnostics, airfoil helpers, STEP/IGES/DXF export. No GUI. |
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
constants, file formats, messages and the window. It was written for the
earlier Python version, which this workspace replaces. The Rust version
follows it, reproduces its reference fits and file outputs (the tests in
`crates/*/tests` hold the frozen Python results), and differs on purpose
where the old behaviour was a defect:

- Progress goes through typed observers, not process-global counters and
  callback signature inspection; a stopped fit is a flag, not a message
  comparison.
- Direct LSQ builds its junction constraints with the library's chain
  constraints instead of a second hand-written copy.
- A coordinate line with an infinite or NaN value is not a point (Python
  failed with "cannot convert float infinity to integer").
- Help > About shows the real version.
- The fit report is a native table, not HTML.
- Library features nothing used (tangent end conditions, averaged knots, knot
  insertion, the second airfoil reader, finite-difference Jacobians) and the
  retired settings (`smoothing_factor`, `early_stop_patience`,
  `relative_tolerance`, `custom_parameters`) are gone; projects that hold
  them still open.

The GUI follows the **Loft** design language,
[docs/design-language.md](docs/design-language.md).

## License

MIT License
