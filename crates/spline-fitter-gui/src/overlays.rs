//! Geometry of the plot overlays: curvature comb, deviation whiskers and
//! the tolerance band.

use spline_fitter::analysis::FootPoints;
use spline_fitter::bspline_varpro::basis::linspace;
use spline_fitter::{BSpline, Vec2};

/// Teeth per section of the fine grid the comb measures on, at least.
const COMB_FINE_SAMPLES: usize = 4000;
/// Share of the turning angle in the tooth spacing.
const COMB_TURNING_SHARE: f64 = 0.5;

/// Segments and polylines of one colour parity (even or odd section).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Strokes {
    /// Teeth or whiskers, base to tip.
    pub teeth: Vec<[Vec2; 2]>,
    /// Envelopes through the tips, one polyline per section or run.
    pub envelopes: Vec<Vec<Vec2>>,
}

/// An overlay drawn in the comb colours: index 0 for the first, third, ...
/// section, 1 for the others.
pub type ParityStrokes = [Strokes; 2];

/// Comb quantities.
pub const QUANTITIES: [&str; 4] = ["Curvature (κ)", "Gradient (dκ/ds)", "Acceleration (d²κ/ds²)", "Jerk (d³κ/ds³)"];

fn cumulative_trapezoid(values: &[f64], t: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    for i in 1..values.len() {
        out[i] = out[i - 1] + 0.5 * (values[i] + values[i - 1]) * (t[i] - t[i - 1]);
    }
    out
}

fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let i = xs.partition_point(|&v| v <= x) - 1;
    let (x0, x1) = (xs[i], xs[i + 1]);
    if x1 == x0 {
        ys[i]
    } else {
        ys[i] + (x - x0) / (x1 - x0) * (ys[i + 1] - ys[i])
    }
}

fn fine_grid(n: usize) -> Vec<f64> {
    linspace(0.0, 1.0, COMB_FINE_SAMPLES.max(20 * n))
}

/// `κ` and `dκ/ds`, 0 where the curve does not move.
fn curvature_and_slope(curve: &BSpline, u: f64) -> (f64, f64) {
    let d = curve.derivatives(u, 3);
    let v = d[1].norm_squared();
    if v <= 0.0 {
        return (0.0, 0.0);
    }
    spline_fitter::bspline_varpro::curve::curvature_and_slope(d[1], d[2], d[3])
}

/// Parameters of `n` teeth, evenly spaced in a blend of arc length and
/// turning angle, so a leading edge that turns through most of the angle in
/// a short distance gets a matching share of the teeth.
pub fn comb_parameters(curve: &BSpline, n: usize) -> Vec<f64> {
    let t = fine_grid(n);
    let mut speed = Vec::with_capacity(t.len());
    let mut turning = Vec::with_capacity(t.len());
    for &u in &t {
        let d = curve.derivatives(u, 2);
        let v = d[1].norm();
        let cubed = v * v * v;
        let kappa = if cubed <= 1e-12 { 0.0 } else { spline_fitter::bspline_varpro::curve::cross(d[1], d[2]) / cubed };
        speed.push(v);
        turning.push(if kappa.is_nan() { 0.0 } else { kappa.abs() } * v);
    }
    let s = cumulative_trapezoid(&speed, &t);
    let turn = cumulative_trapezoid(&turning, &t);
    let (s_last, turn_last) = (s[s.len() - 1], turn[turn.len() - 1]);
    if s_last <= 0.0 {
        return linspace(0.0, 1.0, n);
    }
    let measure: Vec<f64> = s
        .iter()
        .zip(&turn)
        .map(|(a, b)| {
            let arc = a / s_last;
            if turn_last > 0.0 {
                (1.0 - COMB_TURNING_SHARE) * arc + COMB_TURNING_SHARE * b / turn_last
            } else {
                arc
            }
        })
        .collect();
    linspace(0.0, 1.0, n).into_iter().map(|v| interp(v, &measure, &t)).collect()
}

/// NumPy's `gradient(values, x, edge_order=2)`.
fn gradient(f: &[f64], x: &[f64]) -> Vec<f64> {
    let n = f.len();
    if n < 3 {
        return vec![0.0; n];
    }
    let mut g = vec![0.0; n];
    for i in 1..n - 1 {
        let (hs, hd) = (x[i] - x[i - 1], x[i + 1] - x[i]);
        g[i] = (hs * hs * f[i + 1] + (hd * hd - hs * hs) * f[i] - hd * hd * f[i - 1]) / (hs * hd * (hd + hs));
    }
    let (d1, d2) = (x[1] - x[0], x[2] - x[1]);
    g[0] = -(2.0 * d1 + d2) / (d1 * (d1 + d2)) * f[0] + (d1 + d2) / (d1 * d2) * f[1] - d1 / (d2 * (d1 + d2)) * f[2];
    let (d1, d2) = (x[n - 2] - x[n - 3], x[n - 1] - x[n - 2]);
    g[n - 1] = d2 / (d1 * (d1 + d2)) * f[n - 3] - (d2 + d1) / (d1 * d2) * f[n - 2] + (2.0 * d2 + d1) / (d2 * (d1 + d2)) * f[n - 1];
    g
}

/// The comb quantity (κ, dκ/ds, d²κ/ds² or d³κ/ds³) at `t`. Orders above
/// the first are differentiated numerically on a dense grid.
pub fn comb_values(curve: &BSpline, t: &[f64], quantity: usize) -> Vec<f64> {
    if quantity <= 1 {
        return t
            .iter()
            .map(|&u| {
                let (k, s) = curvature_and_slope(curve, u);
                if quantity == 0 {
                    k
                } else {
                    s
                }
            })
            .collect();
    }
    let fine = fine_grid(t.len());
    let speed: Vec<f64> = fine.iter().map(|&u| curve.derivative(u, 1).norm()).collect();
    let slope: Vec<f64> = fine.iter().map(|&u| curvature_and_slope(curve, u).1).collect();
    let s = cumulative_trapezoid(&speed, &fine);
    let keep: Vec<usize> = (0..fine.len()).filter(|&i| i == 0 || s[i] > s[i - 1]).collect();
    let grid: Vec<f64> = keep.iter().map(|&i| fine[i]).collect();
    let s: Vec<f64> = keep.iter().map(|&i| s[i]).collect();
    let mut values: Vec<f64> = keep.iter().map(|&i| slope[i]).collect();
    for _ in 0..quantity - 1 {
        values = gradient(&values, &s);
    }
    t.iter().map(|&u| interp(u, &grid, &values)).collect()
}

/// The x extent of the curves (1 if zero).
fn chord_of_curves(curves: &[BSpline]) -> f64 {
    let xs = curves.iter().flat_map(|c| linspace(0.0, 1.0, 200).into_iter().map(move |u| c.point(u).x));
    let (lo, hi) = xs.fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), x| (l.min(x), h.max(x)));
    let range = hi - lo;
    if range > 0.0 {
        range
    } else {
        1.0
    }
}

/// The curvature comb: per section, teeth along the right normal whose
/// length is `height` chords where the quantity, on the airfoil scaled to a
/// unit chord, is 1.
pub fn comb(curves: &[BSpline], quantity: usize, height: f64, density: usize) -> ParityStrokes {
    let density = density.max(5);
    let chord = chord_of_curves(curves);
    let mut strokes = ParityStrokes::default();
    for (i, curve) in curves.iter().enumerate() {
        let t = comb_parameters(curve, density);
        let values = comb_values(curve, &t, quantity);
        let scale = height * chord.powi(quantity as i32 + 2);
        let teeth: Vec<[Vec2; 2]> = t
            .iter()
            .zip(&values)
            .map(|(&u, &v)| {
                let d = curve.derivatives(u, 1);
                let speed = d[1].norm();
                let normal = if speed > 1e-12 { Vec2::new(d[1].y, -d[1].x) / speed } else { Vec2::zeros() };
                [d[0], d[0] + normal * (v * scale)]
            })
            .collect();
        let target = &mut strokes[i % 2];
        target.envelopes.push(teeth.iter().map(|t| t[1]).collect());
        target.teeth.extend(teeth);
    }
    strokes
}

/// Index of the points inside the curve in [`Deviation`]'s strokes and dots.
pub const INSIDE: usize = 0;
/// Index of the points outside the curve.
pub const OUTSIDE: usize = 1;

/// The deviation overlay and the tolerance line.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Deviation {
    /// Whiskers and envelopes of the points inside and outside the curve.
    pub strokes: [Strokes; 2],
    /// Tips of the whiskers, one dot per data point, by side.
    pub dots: [Vec<Vec2>; 2],
    /// Dashed line at the tolerance outside every section.
    pub band: Vec<Vec<Vec2>>,
}

/// +1 when the points run counterclockwise (the right-hand normal points
/// out of the shape they enclose), else -1.
fn orientation(coords: &[Vec2]) -> f64 {
    let n = coords.len();
    let area: f64 = (0..n).map(|i| coords[i].x * coords[(i + 1) % n].y - coords[(i + 1) % n].x * coords[i].y).sum();
    if area < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Every data point's distance from the curve, drawn outward whatever its
/// sign: a whisker of length `scale · |distance|` along the outward normal,
/// in the color of its side (inside or outside the curve). The tolerance is
/// one dashed line `height` chords outside the curve, and the whiskers use
/// the same magnification, so a point beyond the line is beyond the
/// tolerance. Without a tolerance the largest deviation is drawn at
/// `height`.
pub fn deviation(coords: &[Vec2], curves: &[BSpline], feet: &FootPoints, height: f64, tolerance: Option<f64>) -> Option<Deviation> {
    if coords.len() < 2 {
        return None;
    }
    let largest = feet.distance.iter().fold(0.0, |m: f64, d| m.max(d.abs()));
    let reference = tolerance.unwrap_or(largest);
    let (lo, hi) = coords.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| (l.min(p.x), h.max(p.x)));
    let chord = hi - lo;
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if !positive(reference) || !positive(chord) {
        return None;
    }
    let scale = height * chord / reference;
    let out_sign = orientation(coords);
    // Signed toward the outside, so the side is the sign.
    let outward: Vec<f64> = feet.distance.iter().map(|d| d * out_sign).collect();
    let side = |v: f64| if v > 0.0 { OUTSIDE } else { INSIDE };
    let normal = |i: usize| feet.normal[i] * out_sign;
    let tips: Vec<Vec2> = (0..coords.len()).map(|i| feet.foot[i] + normal(i) * (scale * outward[i].abs())).collect();
    let mut out = Deviation::default();
    let mut run = vec![tips[0]];
    for i in 0..coords.len() {
        let s = side(outward[i]);
        out.strokes[s].teeth.push([feet.foot[i], tips[i]]);
        out.dots[s].push(tips[i]);
        let next = i + 1;
        if next == coords.len() {
            out.strokes[s].envelopes.push(std::mem::take(&mut run));
        } else if side(outward[next]) != s {
            // The envelope meets the curve where the deviation changes sign.
            let t = outward[i].abs() / (outward[i].abs() + outward[next].abs()).max(1e-300);
            let zero = feet.foot[i] + (feet.foot[next] - feet.foot[i]) * t;
            run.push(zero);
            out.strokes[s].envelopes.push(std::mem::replace(&mut run, vec![zero, tips[next]]));
        } else {
            run.push(tips[next]);
        }
    }
    if tolerance.is_some() {
        let reach = height * chord;
        for curve in curves {
            let (a, b) = curve.domain();
            out.band.push(
                linspace(a, b, 400)
                    .into_iter()
                    .map(|u| {
                        let d = curve.derivatives(u, 1);
                        let normal = Vec2::new(d[1].y, -d[1].x) / d[1].norm().max(1e-300);
                        d[0] + normal * (out_sign * reach)
                    })
                    .collect(),
            );
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arc() -> BSpline {
        BSpline::uniform(3, vec![Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.55), Vec2::new(0.55, 1.0), Vec2::new(0.0, 1.0)]).unwrap()
    }

    #[test]
    fn gradient_matches_numpy_on_a_quadratic() {
        let x = [0.0, 0.1, 0.3, 0.6, 1.0];
        let f: Vec<f64> = x.iter().map(|v| v * v).collect();
        for (g, v) in gradient(&f, &x).iter().zip(x) {
            assert!((g - 2.0 * v).abs() < 1e-12);
        }
    }

    #[test]
    fn comb_teeth_point_inward_on_a_counterclockwise_arc() {
        let c = arc();
        let strokes = comb(&[c], 0, 0.05, 20);
        assert_eq!(strokes[0].teeth.len(), 20);
        assert!(strokes[1].teeth.is_empty());
        let [base, tip] = strokes[0].teeth[10];
        // Positive curvature (counterclockwise) along the right normal: outward.
        assert!(tip.norm() > base.norm());
    }

    #[test]
    fn higher_quantities_are_finite() {
        let c = arc();
        let t = comb_parameters(&c, 30);
        assert!(comb_values(&c, &t, 3).iter().all(|v| v.is_finite()));
    }

    /// Points 1e-3 outside a unit arc, as foot points at its parameters.
    fn off_arc(c: &BSpline, offset: f64) -> (Vec<Vec2>, FootPoints) {
        let u = linspace(0.0, 1.0, 9);
        let foot: Vec<Vec2> = u.iter().map(|&t| c.point(t)).collect();
        let normal: Vec<Vec2> = u
            .iter()
            .map(|&t| {
                let d = c.derivative(t, 1);
                Vec2::new(d.y, -d.x) / d.norm()
            })
            .collect();
        let coords = foot.iter().zip(&normal).map(|(f, n)| f + n * offset).collect();
        let n = u.len();
        (coords, FootPoints { distance: vec![offset; n], foot, normal, section: vec![0; n] })
    }

    fn longest_whisker(d: &Deviation) -> f64 {
        d.strokes.iter().flat_map(|s| &s.teeth).map(|[a, b]| (b - a).norm()).fold(0.0, f64::max)
    }

    fn chord(coords: &[Vec2]) -> f64 {
        coords.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) - coords.iter().map(|p| p.x).fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn tolerance_line_stays_at_the_height_and_sets_the_magnification() {
        let c = arc();
        let (coords, feet) = off_arc(&c, 1e-3);
        let chord = chord(&coords);
        for tolerance in [1e-2, 1e-5] {
            let d = deviation(&coords, std::slice::from_ref(&c), &feet, 0.05, Some(tolerance)).unwrap();
            // One line, outside the curve, `height` chords away.
            assert_eq!(d.band.len(), 1);
            let on_curve = c.point(linspace(0.0, 1.0, 400)[200]);
            let line = d.band[0][200];
            assert!(((line - on_curve).norm() - 0.05 * chord).abs() < 1e-9);
            assert!(line.norm() > on_curve.norm());
            // A deviation of `tolerance` would reach the line.
            assert!((longest_whisker(&d) - 0.05 * chord * 1e-3 / tolerance).abs() < 1e-9);
        }
        // Without a tolerance the largest deviation is at the height.
        let d = deviation(&coords, std::slice::from_ref(&c), &feet, 0.05, None).unwrap();
        assert!(d.band.is_empty());
        assert!((longest_whisker(&d) - 0.05 * chord).abs() < 1e-9);
    }

    #[test]
    fn whiskers_point_outward_and_take_the_color_of_their_side() {
        let c = arc();
        for (offset, side) in [(1e-3, OUTSIDE), (-1e-3, INSIDE)] {
            let (coords, feet) = off_arc(&c, offset);
            let d = deviation(&coords, std::slice::from_ref(&c), &feet, 0.05, None).unwrap();
            assert_eq!(d.strokes[side].teeth.len(), coords.len());
            assert!(d.strokes[1 - side].teeth.is_empty() && d.dots[1 - side].is_empty());
            for [foot, tip] in &d.strokes[side].teeth {
                assert!(tip.norm() > foot.norm() + 1e-3, "{offset}: {foot} -> {tip}");
            }
        }
        // The same points in clockwise order still draw outward.
        let (mut coords, mut feet) = off_arc(&c, -1e-3);
        coords.reverse();
        feet.foot.reverse();
        feet.normal = feet.normal.iter().rev().map(|n| -n).collect();
        feet.distance = feet.distance.iter().map(|d| -d).collect();
        let d = deviation(&coords, std::slice::from_ref(&c), &feet, 0.05, None).unwrap();
        assert_eq!(d.dots[INSIDE].len(), coords.len());
        for [foot, tip] in &d.strokes[INSIDE].teeth {
            assert!(tip.norm() > foot.norm() + 1e-3);
        }
    }

    #[test]
    fn envelope_meets_the_curve_where_the_deviation_changes_sign() {
        let c = arc();
        let (_, mut feet) = off_arc(&c, 1e-3);
        // Outside for the first four points, inside for the rest.
        for (i, d) in feet.distance.iter_mut().enumerate() {
            *d = if i < 4 { 1e-3 } else { -3e-3 };
        }
        let coords: Vec<Vec2> = (0..feet.foot.len()).map(|i| feet.foot[i] + feet.normal[i] * feet.distance[i]).collect();
        let d = deviation(&coords, std::slice::from_ref(&c), &feet, 0.05, None).unwrap();
        let (outside, inside) = (&d.strokes[OUTSIDE].envelopes, &d.strokes[INSIDE].envelopes);
        assert_eq!((outside.len(), inside.len()), (1, 1));
        let zero = *outside[0].last().unwrap();
        assert_eq!(zero, inside[0][0]);
        // A quarter of the way from foot 3 to foot 4: |1e-3| / (|1e-3| + |-3e-3|).
        assert!((zero - (feet.foot[3] + (feet.foot[4] - feet.foot[3]) * 0.25)).norm() < 1e-12);
        assert_eq!(outside[0].len(), 5);
        assert_eq!(inside[0].len(), 6);
    }
}
