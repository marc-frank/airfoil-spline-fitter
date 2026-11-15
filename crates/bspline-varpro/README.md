# bspline-varpro

CAD-grade B-spline curve fitting. Given ordered points, it finds the curve
with the **fewest control points within a maximum deviation**, with
**geometric G1-G3 continuity** where sections meet, and writes it exactly to
**STEP, IGES or DXF**. Built for airfoils; works on any open planar chain.

The solver optimizes the curve parameter of every point and the control
points jointly by **variable projection**: the control points are eliminated
by an equality-constrained least-squares (KKT) solve, and the parameters and
the junction shape parameters are solved by **Levenberg-Marquardt** with the
exact reduced Jacobian. Knots are non-uniform: inserted where the deviation
is largest, moved to where they reduce the error most, and removed where not
needed.

## Example

```rust
use bspline_varpro::airfoil::{default_tolerance, Naca4};
use bspline_varpro::export::{write, Units};
use bspline_varpro::{fit_to_tolerance, ChainOptions, Continuity, Junction, Silent, ToleranceOptions};

let points = Naca4::new("2412", true)?.points(81);
let sections = vec![points[..81].to_vec(), points[80..].to_vec()];
let options = ToleranceOptions {
    chain: ChainOptions { continuity: Continuity::All(Junction::new(2, true, true)?), smoothing: 1e-7, ..Default::default() },
    ..Default::default()
};
let tolerance = default_tolerance(&points, None);
let result = fit_to_tolerance(&sections, tolerance, 5, &options, &mut Silent)?;
println!("{:?} {:?}", result.num_control_points(), result.max_deviation);
write("naca2412.step".as_ref(), &result.curves(), 250.0, Units::Millimetre)?;
# Ok::<(), bspline_varpro::Error>(())
```

## Tolerance search

1. Start with the fewest control points the junction conditions allow.
2. Fit, measure the orthogonal deviation of every point, and insert a knot in
   every span that holds a point beyond the tolerance. Repeat until every
   point is within it.
3. Optimize the interior knot positions together with the fit (free
   placement).
4. Try removing every knot, least useful first, refitting after each attempt;
   keep removals that stay within the tolerance. Optimize the knots again.

Sections longer than `max_points_per_section` are fitted on an evenly spaced
subsample, then refined jointly on all points.

## Modules

| Module | Purpose |
|---|---|
| `basis` | Knot vectors, basis functions (sparse rows), bending energy |
| `curve` | Planar B-spline evaluation and curvature |
| `chain` | The data of a chain of sections, chain parameters, projection |
| `constraints` | Junction and end constraints `C(θ) P = d` |
| `reparam` | Softplus reparameterization of monotone parameters |
| `fitting` | `fit`, continuation for geometric junctions |
| `tolerance` | `fit_to_tolerance`, knot growth, polishing, subsampling |
| `joint` | Joint refinement of parameters and control points |
| `diagnostics` | Orthogonal projection, fairness, junction measures |
| `airfoil` | NACA 4-digit, default tolerance, leading edge, thickness and camber, resampling |
| `export` | STEP, IGES, DXF, merging a chain into one curve |
| `numfmt` | Number formatting of the file formats |
