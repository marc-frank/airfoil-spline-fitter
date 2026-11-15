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

/// The deviation overlay and the tolerance band.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Deviation {
    pub strokes: ParityStrokes,
    /// Tips of the whiskers, one dot per data point and parity.
    pub dots: [Vec<Vec2>; 2],
    /// Dashed lines at the tolerance on both sides of every section.
    pub band: Vec<Vec<Vec2>>,
}

/// Every data point's distance from the curve, magnified along the normal
/// so the tolerance (or without one the largest deviation) is drawn
/// `height` chords from the curve.
pub fn deviation(coords: &[Vec2], curves: &[BSpline], feet: &FootPoints, height: f64, tolerance: Option<f64>) -> Option<Deviation> {
    if coords.len() < 2 {
        return None;
    }
    let reference = tolerance.unwrap_or_else(|| feet.distance.iter().fold(0.0, |m, d| m.max(d.abs())));
    let (lo, hi) = coords.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| (l.min(p.x), h.max(p.x)));
    let chord = hi - lo;
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if !positive(reference) || !positive(chord) {
        return None;
    }
    let scale = height * chord / reference;
    let tips: Vec<Vec2> = (0..coords.len()).map(|i| feet.foot[i] + feet.normal[i] * (scale * feet.distance[i])).collect();
    let mut out = Deviation::default();
    // Runs of equal nearest section; each run after the first starts at the
    // last point of the previous one.
    let mut start = 0;
    while start < coords.len() {
        let section = feet.section[start];
        let mut end = start;
        while end + 1 < coords.len() && feet.section[end + 1] == section {
            end += 1;
        }
        let first = if start > 0 { start - 1 } else { start };
        let parity = section % 2;
        let strokes = &mut out.strokes[parity];
        for i in first..=end {
            strokes.teeth.push([feet.foot[i], tips[i]]);
            out.dots[parity].push(tips[i]);
        }
        strokes.envelopes.push(tips[first..=end].to_vec());
        start = end + 1;
    }
    if let Some(t) = tolerance {
        let reach = scale * t;
        for curve in curves {
            let (a, b) = curve.domain();
            for sign in [1.0, -1.0] {
                out.band.push(
                    linspace(a, b, 400)
                        .into_iter()
                        .map(|u| {
                            let d = curve.derivatives(u, 1);
                            let normal = Vec2::new(d[1].y, -d[1].x) / d[1].norm().max(1e-300);
                            d[0] + normal * (sign * reach)
                        })
                        .collect(),
                );
            }
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
}
