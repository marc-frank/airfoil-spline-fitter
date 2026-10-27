# bspline-varpro

CAD-grade B-spline curve fitting. Given ordered points, `bspline-varpro` finds
the curve with the **fewest control points within a maximum deviation**, with
**geometric G1-G3 continuity** where sections meet, a **fair curvature comb**,
and writes it exactly to **STEP, IGES, or DXF**. It is built for airfoils but
works on any open planar or 3D curve chain.

The core solver optimizes the curve parameter of every point and the control
points jointly by **variable projection** with a **Levenberg-Marquardt** outer
loop and the exact reduced Jacobian. Knots are non-uniform: they are inserted
where the deviation is largest, removed where they are not needed, and then
optimized.

```bash
pip install bspline-varpro
```

Requires Python 3.9+, NumPy, and SciPy.

## Quick start: an airfoil to CAD

```python
from bspline_varpro import airfoil, export

name, points = airfoil.read_dat("naca2412.dat")   # Selig or Lednicer
result = airfoil.fit_airfoil(points)               # tolerance from the data's precision

print(result.num_control_points, result.max_deviation)   # e.g. [13, 11] 9.9e-06
curves = result.curves()                           # scipy.interpolate.BSpline per section

export.write("naca2412.step", curves, name=name, units="mm", scale=250.0)
airfoil.write_dat("naca2412_fitted.dat", airfoil.resample(curves, 120), name=name)
print(airfoil.leading_edge_radius(curves))
```

`fit_airfoil` splits the points at the leading edge (the point farthest from
the trailing-edge midpoint), joins the upper and lower sections with G2 by
default, and runs `fit_to_tolerance`. Without a tolerance it uses
`airfoil.default_tolerance`: one unit in the last decimal of rounded data, or
the expected maximum of the estimated noise for scanned data. It warns if you
ask for a tolerance below the rounding error of the file.

## Tolerance fitting

```python
import bspline_varpro as bv

result = bv.fit_to_tolerance(
    [upper, lower],          # ordered sections; the last point of one is the first of the next
    tolerance=1e-5,          # maximum orthogonal deviation of any point, in data units
    degree=5,
    continuity="G2",         # or one entry per junction: ["G2", "C1", "G3"]
    fairing=True,            # smoothest curvature comb that stays within the tolerance
)
result.success, result.max_deviation, result.num_control_points
result.knots                 # non-uniform clamped knot vector per section
result.history               # every insertion, removal, polish and fairing stage
```

The procedure:

1. Start with the fewest control points the end and junction conditions allow.
2. Fit, measure the exact orthogonal deviation of every point, and insert knots
   in the spans that exceed the tolerance. Repeat until every point is within it.
3. Try removing each knot (least useful first) and keep removals that stay
   within the tolerance.
4. Optimize the interior knot positions together with the fit.
5. Fair: minimize the curvature-slope energy ∫(dκ/ds)² ds plus the squared
   deviations, with the largest weight that keeps every point within the
   tolerance.

Dense data (more than `max_points_per_section`, default 200 per section) is
fitted on an evenly spaced subsample and then refined jointly on all points. A
6000-point NACA 2412 fits to 1e-5 with fairing in about 15 s
(`benchmarks/results/speed_0.7.txt`).

Useful options: `max_control_points` (per section), `fairing_tolerance`
(allow a different deviation after fairing), `optimize_knots`, `remove_knots`,
`start_tangent` / `end_tangent`, `fix_junctions=False` (fit the junction
position instead of interpolating it), and `callback(result, deviation)`, which
can return True to stop.

## Continuity

Every junction between consecutive sections takes its own spec:

| Spec | Meaning |
| --- | --- |
| `"C0"`-`"C3"` or `0`-`3` | Parametric: derivatives up to that order are equal |
| `"G1"` | Common tangent direction |
| `"G2"` | Common tangent and curvature |
| `"G3"` | Common tangent, curvature, and curvature slope dκ/ds |
| `Junction(order, geometric, fixed, tangent)` | Full control, including a prescribed tangent direction |

Geometric continuity is the CAD notion: the curves agree after a local
reparameterization, so the parametric speeds on both sides are free. It is
imposed exactly through shape parameters (see [Method](#method)). On fitted
airfoils, the measured tangent-angle jump at a G2 leading edge is about 2e-13 rad,
and the relative curvature jump is about 1e-12.

## Fixed control points: `fit`

```python
result = bv.fit(
    [upper, lower],
    num_control_points=12,   # per section (an int or one per section), uniform knots
    degree=5,
    continuity="G2",
    smoothing=1e-5,          # bending-energy weight, 0 disables it
)
# or with your own knot vectors:
result = bv.fit(points, knots=[my_knots], degree=5)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `num_control_points` / `knots` | required | Control-point count with uniform knots, or clamped knot vectors on [0, 1] |
| `degree` | 3 | Spline degree |
| `continuity` | `"G1"` | One spec for all junctions or a list with one per junction |
| `fix_junctions` | True | Interpolate the junction data point instead of fitting its position |
| `fix_ends` | True | Interpolate the first and last point of the chain |
| `start_tangent`, `end_tangent` | None | Prescribed directions of travel at the chain ends |
| `smoothing` | 1e-3 | Bending-energy weight (normalized to trace = control-point count) |
| `initial_parameters`, `initial_theta` | centripetal | Warm starts from a previous fit |
| `max_points_per_section` | None | Subsample dense sections, then refine on all points |
| `max_nfev` | 200 | Residual evaluation budget |
| `callback` | None | `callback(control_points, rms_error)` after every evaluation; True stops |

## `FitResult`

`control_points`, `knots`, `parameters` (fitted parameter of every point),
`degree`, `rms_error`, `max_residual`, `max_deviation` (exact orthogonal, for
tolerance fits and fairing), `continuity` labels, `shape_parameters` per
junction, `theta` (for warm starts), `history`, `success`, and `message`.
`curves()` returns one `scipy.interpolate.BSpline` per section and
`deviations(sections)` the orthogonal distance of every point.

## Fairing and refinement

```python
faired = bv.fair(points, result, tolerance=2e-5)   # or a fixed weight=...
refined = bv.refine(dense_points, result)          # joint LM on all points, no fairing
polished = bv.polish_knots(points, result)         # optimize interior knot positions
```

## Diagnostics

`bspline_varpro.diagnostics` measures what a CAD reviewer would check:

- `project(points, curve)`: closest parameter and orthogonal distance, refined
  to machine precision (global search per knot span, then safeguarded Newton).
- `max_deviation(sections, curves)`, `section_deviations(sections, curves)`.
- `curvature(curve, u)`, `curvature_slope(curve, u)` (dκ/ds).
- `fairness(curves)`: interior curvature extrema, ∫(dκ/ds)² ds, maximum
  curvature and curvature slope.
- `junctions(curves)`: position gap, tangent angle, curvature jump, and
  curvature-slope jump at every junction.

## Export

```python
from bspline_varpro import export

export.write("part.step", curves, name="wing root", units="mm", scale=1000.0)
export.write("part.igs", curves, units="inch")
export.write("part.dxf", curves)
text = export.to_step(curves)                      # or to_iges / to_dxf
merged = export.merge_curves(curves)               # one B-spline for the whole chain
points = export.sample(curve, 200, spacing="cosine")
```

- **STEP** AP214 `B_SPLINE_CURVE_WITH_KNOTS` in a wireframe shape
  representation (mm, m, or inch).
- **IGES** 5.3 rational B-spline entity 126 (with unit weights).
- **DXF** R2000 `SPLINE` entities.

Each section is written as its own curve with its exact knots and control
points. `merge_curves` joins a chain into a single B-spline: parameter ranges
are scaled so smooth junctions become C1, then every junction knot that
exact knot removal allows is removed. On unit-chord airfoils, the merged curve
stays within 1e-12 of the chain. `airfoil.trailing_edge_segment(curves)`
returns the straight line that closes a blunt trailing edge.

All three formats were read back with OpenCASCADE (STEP and IGES) and ezdxf
(DXF), using `benchmarks/verify_cad.py`. The test parts were written at chords
of 1 to 1000 in mm, m, and inch. The largest deviation from the fitted curves
was 6e-13 model units for STEP and IGES. For DXF it was 2e-8, which is ezdxf's
spline evaluation accuracy: the file stores full-precision values.

## Airfoil helpers

`bspline_varpro.airfoil` provides:

- `read_dat`, `write_dat`: Selig and Lednicer `.dat` files.
- `decimal_places`, `noise_level`, `precision_floor`, `default_tolerance`:
  what the data can support.
- `split_at_leading_edge`, `leading_edge_index`, `trailing_edge_gap`.
- `fit_airfoil(points, tolerance=None, degree=5, le_continuity="G2", sections=None)`.
  Pass `sections` for any other split, for example four sections with a
  per-junction continuity list.
- `leading_edge(curves)`, `leading_edge_radius(curves)`: on the fitted curve.
- `trailing_edge_segment(curves)`.
- `resample(curves, n_per_side, spacing="cosine")`: resampling by arc length,
  clustered at the leading and trailing edges.
- `NACA4`: exact 4-digit sections for testing.

## Benchmarks

`benchmarks/run.py` compares methods on exact NACA geometry, on rounded,
noisy, and dense data. For each tolerance, every method must find the fewest
control points that keep all data points within it. The table compares
`scipy.interpolate.splprep` (FITPACK, degree 5) with `fit_to_tolerance`
(degree 5, G2 at the leading edge, with fairing). The count is control points
of the whole curve. The full results, including the deviation from the true
airfoil, are in `benchmarks/results/benchmark_1.0.txt`.

| Data | Tolerance | splprep control points | tolerance fit control points | splprep curvature extrema | tolerance fit curvature extrema | splprep LE curvature error | tolerance fit LE curvature error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NACA 0012 exact | 1e-4 | 21 | 11 | 5 | 2 | -2.5% | -0.7% |
| NACA 0012 exact | 1e-5 | 34 | 13 | 8 | 4 | 4.0% | -0.3% |
| NACA 0012 exact | 1e-6 | 45 | 17 | 8 | 4 | 1.2% | -0.3% |
| NACA 0012, 5 decimals | 2e-5 | 30 | 12 | 6 | 3 | 3.8% | 0.1% |
| NACA 2412, 5 decimals | 2e-5 | 29 | 19 | 8 | 6 | 2.4% | -0.6% |
| NACA 4412 blunt TE, 6 decimals | 1e-5 | 43 | 23 | 10 | 6 | 0.4% | -0.1% |
| NACA 0012, 2002 points | 1e-5 | 32 | 12 | 7 | 3 | 3.3% | -0.1% |
| NACA 0012, noise σ = 1e-4 | 3e-4 | 43 | 17 | 23 | 10 | 0.6% | -4.8% |

"Curvature extrema" counts wiggles in the curvature comb. A symmetric section
has at least 2, and a cambered one more. A tolerance fit takes 0.7-5 s, and
splprep 0.2-2 s. On noisy data, the tolerance fit uses the whole tolerance
band, so its curve can sit that far from the true shape. Its deviation from the
true airfoil is 2.7e-4, against 1.2e-4 for splprep's 43-control-point
least-squares curve. For scans, a tolerance below the default brings the curve
closer to the true shape at the cost of more control points.

`benchmarks/speed.py` times dense data. `benchmarks/verify_cad.py` re-imports
exported files with OpenCASCADE and ezdxf.

## Method

For data `D`, parameters `t`, and control points `P`, the fit minimizes

```
||N(t) P - D||² + lambda * P' S P + reg * ||P||²    subject to  C(theta) P = d(theta)
```

where `N(t)` is the B-spline basis matrix, `S` the Gram matrix of basis second
derivatives, and `C P = d` encodes the end, junction, and continuity conditions.

- **Variable projection.** `P` enters linearly. For fixed `t` and `theta`, the
  optimal `P*(t, theta)` comes from one LU-factored KKT system, which leaves a
  reduced problem in `t` and `theta`. The reduced residual includes the penalty
  rows, so the outer objective equals the inner one.
- **Exact reduced Jacobian.** `dr/dt_j` contains the direct term (point `j`
  moving along the curve) and the coupling term through `dP*/dt_j`, obtained by
  differentiating the KKT system with the existing factorization.
- **Geometric continuity.** G^k is C^k after a reparameterization with
  derivatives `(alpha, beta, gamma)`:
  `C_a' = alpha C_b'`, `C_a'' = alpha² C_b'' + beta C_b'`,
  `C_a''' = alpha³ C_b''' + 3 alpha beta C_b'' + gamma C_b'`.
  These shape parameters enter `C` linearly per coordinate, so the problem stays
  separable, and they join `t` as outer variables. Geometric fits start from the
  converged parametric solution, because the shape parameters and the data
  parameterization can trade off along nearly flat directions.
- **Monotone parameters.** Each section's parameters are normalized cumulative
  sums of `softplus(eta)`, so they stay ordered and the section ends stay at 0 and 1.
- **Solver.** Levenberg-Marquardt on the normal equations (Cholesky), with
  Marquardt scaling, Nielsen damping updates, MINPACK stopping tests, and
  restarts when the damping stalls in curved valleys.
- **Fairing.** A joint Levenberg-Marquardt problem over parameters and control
  points adds the discretized ∫(dκ/ds)² ds (Gauss-Legendre per knot span) to
  the squared deviations. The weight is found by bisection in log scale as the
  largest one whose exact maximum deviation stays within the tolerance.

## Limitations

- The objective is nonconvex in the parameters. Fits converge to a local
  minimum near the starting parameterization. Centripetal starts, parametric
  continuation, and knot insertion make this reliable for airfoils and similar
  smooth curves, but self-intersecting or strongly looping data may need
  `initial_parameters`.
- Fairing is available for planar curves of degree 3 or higher.
- Curves are open and non-rational. Closed or periodic curves are not
  supported. A blunt trailing edge is closed with a separate line segment.
- Knot polishing uses finite differences for the knot columns and is skipped
  for subsampled dense data.

## References

- G. H. Golub and V. Pereyra, "The Differentiation of Pseudo-Inverses and
  Nonlinear Least Squares Problems Whose Variables Separate", SIAM J. Numer.
  Anal. 10 (1973).
- L. Kaufman, "A variable projection method for solving separable nonlinear
  least squares problems", BIT 15 (1975).
- D. P. O'Leary and B. W. Rust, "Variable projection for nonlinear least
  squares problems", Comput. Optim. Appl. 54 (2013).
- B. A. Barsky and T. D. DeRose, "Geometric Continuity of Parametric Curves",
  IEEE Computer Graphics and Applications 9 (1989).
- L. Piegl and W. Tiller, *The NURBS Book*, 2nd ed., Springer (1997): knot
  insertion and removal.

## License

MIT
