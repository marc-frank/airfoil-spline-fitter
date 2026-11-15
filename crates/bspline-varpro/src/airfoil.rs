//! Airfoil helpers: default tolerances, leading and trailing edge, thickness
//! and camber, resampling, and analytic NACA 4-digit sections.
//!
//! Coordinates are in Selig order: upper trailing edge, over the leading
//! edge, to the lower trailing edge.

use std::f64::consts::PI;

use nalgebra::DMatrix;

use crate::basis::linspace;
use crate::chain::largest_extent;
use crate::curve::{cross, BSpline};
use crate::linalg::pinv;
use crate::{bail, Result, Vec2};

/// Analytic NACA 4-digit section parameterized by `σ = √x`, which makes the
/// thickness polynomial and the surfaces analytic at the leading edge.
#[derive(Debug, Clone, PartialEq)]
pub struct Naca4 {
    m: f64,
    p: f64,
    t: f64,
    a4: f64,
}

impl Naca4 {
    /// `closed_te` selects the modified coefficient (-0.1036) for a sharp
    /// trailing edge, otherwise the original (-0.1015).
    pub fn new(code: &str, closed_te: bool) -> Result<Self> {
        if code.chars().count() != 4 || !code.chars().all(|c| c.is_ascii_digit()) {
            bail!("NACA 4-digit code must have four digits.");
        }
        let digit = |i: usize| (code.as_bytes()[i] - b'0') as f64;
        Ok(Naca4 {
            m: digit(0) / 100.0,
            p: digit(1) / 10.0,
            t: code[2..].parse::<f64>().unwrap() / 100.0,
            a4: if closed_te { -0.1036 } else { -0.1015 },
        })
    }

    fn thickness(&self, sigma: f64) -> f64 {
        let x = sigma * sigma;
        5.0 * self.t * (0.2969 * sigma - 0.1260 * x - 0.3516 * x * x + 0.2843 * x.powi(3) + self.a4 * x.powi(4))
    }

    fn camber(&self, x: f64) -> (f64, f64) {
        let (m, p) = (self.m, self.p);
        if m == 0.0 || p == 0.0 {
            return (0.0, 0.0);
        }
        if x < p {
            (m / (p * p) * (2.0 * p * x - x * x), 2.0 * m / (p * p) * (p - x))
        } else {
            let q = (1.0 - p) * (1.0 - p);
            (m / q * ((1.0 - 2.0 * p) + 2.0 * p * x - x * x), 2.0 * m / q * (p - x))
        }
    }

    /// Point on the upper or lower surface at `σ = √x`.
    pub fn surface(&self, sigma: f64, upper: bool) -> Vec2 {
        let x = sigma * sigma;
        let yt = self.thickness(sigma) * if upper { 1.0 } else { -1.0 };
        let (yc, slope) = self.camber(x);
        let theta = slope.atan();
        Vec2::new(x - yt * theta.sin(), yc + yt * theta.cos())
    }

    /// Selig-ordered points, cosine spaced toward both edges: the upper
    /// surface from the trailing edge, then the lower (`2n - 1` points).
    pub fn points(&self, n_per_side: usize) -> Vec<Vec2> {
        let sigma: Vec<f64> = linspace(0.0, PI, n_per_side).iter().map(|b| (b / 2.0).sin()).collect();
        let upper = sigma.iter().rev().map(|&s| self.surface(s, true));
        let lower = sigma[1..].iter().map(|&s| self.surface(s, false));
        upper.chain(lower).collect()
    }
}

/// Decimals the coordinates were rounded to (0..=10), or `None`.
pub fn decimal_places(points: &[Vec2]) -> Option<i32> {
    let values: Vec<f64> = points.iter().flat_map(|p| [p.x, p.y]).collect();
    let scale = values.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    (0..=10).find(|&d| {
        let factor = 10f64.powi(d);
        values.iter().all(|v| (v - (v * factor).round_ties_even() / factor).abs() <= 1e-12 * scale)
    })
}

/// Robust estimate of the normal noise of ordered points: a degree-6
/// polynomial is fitted in a local frame to every window of 9 points and the
/// leverage-corrected residual at the window centre kept; the median
/// absolute residual ignores the few windows the polynomial cannot follow.
pub fn noise_level(points: &[Vec2]) -> f64 {
    const WINDOW: usize = 9;
    const DEGREE: usize = 6;
    let half = WINDOW / 2;
    if points.len() < WINDOW {
        return 0.0;
    }
    let mut residuals = Vec::new();
    for w in points.windows(WINDOW) {
        let chord = w[WINDOW - 1] - w[0];
        let length = chord.norm();
        if length <= 0.0 {
            continue;
        }
        let e = chord / length;
        let n = Vec2::new(-e.y, e.x);
        let center = w[half];
        let x: Vec<f64> = w.iter().map(|p| (p - center).dot(&e) / length).collect();
        let y: Vec<f64> = w.iter().map(|p| (p - center).dot(&n)).collect();
        if !x.windows(2).all(|p| p[1] > p[0]) {
            continue;
        }
        let v = DMatrix::from_fn(WINDOW, DEGREE + 1, |i, j| x[i].powi(j as i32));
        let hat = &v * pinv(&v, 1e-15);
        let fitted: f64 = (0..WINDOW).map(|k| hat[(half, k)] * y[k]).sum();
        let leverage = (1.0 - hat[(half, half)]).max(1e-12);
        residuals.push(((y[half] - fitted) / leverage.sqrt()).abs());
    }
    if residuals.is_empty() {
        return 0.0;
    }
    residuals.sort_by(f64::total_cmp);
    let n = residuals.len();
    let median = if n % 2 == 1 { residuals[n / 2] } else { 0.5 * (residuals[n / 2 - 1] + residuals[n / 2]) };
    1.4826 * median
}

/// Largest orthogonal error rounding alone can cause (0 for unrounded data).
pub fn precision_floor(points: &[Vec2]) -> f64 {
    decimal_places(points).map_or(0.0, |d| 0.5 * 2f64.sqrt() * 10f64.powi(-d))
}

/// A tolerance a fit can meet without chasing rounding or noise: the
/// largest of one unit in the last decimal of rounded data, the expected
/// maximum of the estimated noise (unless rounding explains it) and 1e-6
/// times the chord. `decimals` overrides the detected decimal places.
pub fn default_tolerance(points: &[Vec2], decimals: Option<i32>) -> f64 {
    let mut tol = 1e-6 * largest_extent(points);
    let sigma = noise_level(points);
    if let Some(d) = decimals.or_else(|| decimal_places(points)) {
        let unit = 10f64.powi(-d);
        tol = tol.max(unit);
        if sigma <= 3.0 * unit / 12f64.sqrt() {
            return tol;
        }
    }
    let expected_max = (2.0 * (points.len().max(2) as f64).ln()).sqrt() + 0.5;
    tol.max(expected_max * sigma)
}

fn chain_start(curves: &[BSpline]) -> Vec2 {
    curves[0].start()
}

fn chain_end(curves: &[BSpline]) -> Vec2 {
    curves[curves.len() - 1].end()
}

/// `(section, parameter, point)` of the chain point farthest from the
/// trailing-edge midpoint.
pub fn leading_edge(curves: &[BSpline]) -> (usize, f64, Vec2) {
    let te = 0.5 * (chain_start(curves) + chain_end(curves));
    let mut best: Option<(f64, usize, Vec<f64>, usize)> = None;
    for (i, curve) in curves.iter().enumerate() {
        let (lo, hi) = curve.domain();
        let u = linspace(lo, hi, 2001);
        let d: Vec<f64> = u.iter().map(|&v| (curve.point(v) - te).norm()).collect();
        let j = (0..d.len()).fold(0, |b, k| if d[k] > d[b] { k } else { b });
        if best.as_ref().is_none_or(|b| d[j] > b.0) {
            best = Some((d[j], i, u, j));
        }
    }
    let (_, i, u, j) = best.unwrap();
    let curve = &curves[i];
    let (lo, hi) = (u[j.saturating_sub(1)], u[(j + 1).min(u.len() - 1)]);
    let (v, f) = fminbound(|v| -(curve.point(v) - te).norm(), lo, hi, 1e-14);
    let v = if -f >= (curve.point(u[j]) - te).norm() { v } else { u[j] };
    (i, v, curve.point(v))
}

/// Radius of curvature at the leading edge (infinite for zero curvature).
pub fn leading_edge_radius(curves: &[BSpline]) -> f64 {
    let (i, u, _) = leading_edge(curves);
    let d = curves[i].derivatives(u, 2);
    let kappa = cross(d[1], d[2]).abs() / d[1].norm().powi(3);
    if kappa > 0.0 { 1.0 / kappa } else { f64::INFINITY }
}

/// Brent's bounded minimization of `f` on `[a, b]` (as SciPy's `fminbound`):
/// returns `(x, f(x))`.
pub fn fminbound(f: impl Fn(f64) -> f64, a: f64, b: f64, xatol: f64) -> (f64, f64) {
    const MAX_EVALUATIONS: usize = 500;
    let sqrt_eps = 2.2e-16f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5f64.sqrt());
    let (mut a, mut b) = (a, b);
    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let (mut rat, mut e) = (0.0f64, 0.0f64);
    let mut fx = f(xf);
    let mut evaluations = 1;
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;
    let sign = |v: f64| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 };
    while (xf - xm).abs() > tol2 - 0.5 * (b - a) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                let x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = sign(xm - xf) + if xm - xf == 0.0 { 1.0 } else { 0.0 };
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let si = sign(rat) + if rat == 0.0 { 1.0 } else { 0.0 };
        let x = xf + si * rat.abs().max(tol1);
        let fu = f(x);
        evaluations += 1;
        if fu <= fx {
            if x >= xf { a = xf } else { b = xf }
            (fulc, ffulc) = (nfc, fnfc);
            (nfc, fnfc) = (xf, fx);
            (xf, fx) = (x, fu);
        } else {
            if x < xf { a = x } else { b = x }
            if fu <= fnfc || nfc == xf {
                (fulc, ffulc) = (nfc, fnfc);
                (nfc, fnfc) = (x, fu);
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                (fulc, ffulc) = (x, fu);
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if evaluations >= MAX_EVALUATIONS {
            break;
        }
    }
    (xf, fx)
}

/// Arc-length table `(u, s)` of a curve on 4096 samples (trapezoid rule).
pub(crate) fn arc_table(curve: &BSpline) -> (Vec<f64>, Vec<f64>) {
    let (lo, hi) = curve.domain();
    let u = linspace(lo, hi, 4096);
    let speed: Vec<f64> = u.iter().map(|&v| curve.derivative(v, 1).norm()).collect();
    let mut s = vec![0.0];
    for i in 1..u.len() {
        s.push(s[i - 1] + 0.5 * (speed[i] + speed[i - 1]) * (u[i] - u[i - 1]));
    }
    (u, s)
}

/// Linear interpolation of `(xs, ys)` at `x`, constant beyond the ends.
pub(crate) fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
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
        return ys[i];
    }
    ys[i] + (x - x0) / (x1 - x0) * (ys[i + 1] - ys[i])
}

/// `0.5(1 - cos(linspace(0, π, n)))`: cosine spacing on `[0, 1]`.
pub fn cosine_spacing(n: usize) -> Vec<f64> {
    linspace(0.0, PI, n).iter().map(|b| 0.5 * (1.0 - b.cos())).collect()
}

/// Selig-ordered points on a fitted airfoil, `n_per_side` per side of the
/// geometric leading edge (shared), cosine spaced in arc length so they
/// cluster at both edges.
pub fn resample(curves: &[BSpline], n_per_side: usize) -> Result<Vec<Vec2>> {
    if n_per_side < 2 {
        bail!("Need at least two points per side.");
    }
    // Chain table; sections repeat their boundary arc length, so a bracket
    // never spans two sections.
    let (mut sections, mut us, mut ss) = (Vec::new(), Vec::new(), Vec::new());
    let mut offset = 0.0;
    for (i, curve) in curves.iter().enumerate() {
        let (u, s) = arc_table(curve);
        sections.extend(std::iter::repeat(i).take(u.len()));
        us.extend_from_slice(&u);
        let total = s[s.len() - 1];
        ss.extend(s.iter().map(|v| v + offset));
        offset += total;
    }
    let (i, u_le, p_le) = leading_edge(curves);
    let rows: Vec<usize> = (0..sections.len()).filter(|&k| sections[k] == i).collect();
    let row_u: Vec<f64> = rows.iter().map(|&k| us[k]).collect();
    let row_s: Vec<f64> = rows.iter().map(|&k| ss[k]).collect();
    let s_le = interp(u_le, &row_u, &row_s);
    let total = ss[ss.len() - 1];
    let f = cosine_spacing(n_per_side);
    let targets = f.iter().map(|v| v * s_le).chain(f[1..].iter().map(|v| s_le + v * (total - s_le)));
    let last = ss.len() - 2;
    let mut out: Vec<Vec2> = targets
        .map(|value| {
            let idx = (ss.partition_point(|&s| s <= value) as isize - 1).clamp(0, last as isize) as usize;
            let ds = ss[idx + 1] - ss[idx];
            let frac = if ds > 0.0 { ((value - ss[idx]) / ds).clamp(0.0, 1.0) } else { 0.0 };
            curves[sections[idx]].point(us[idx] + frac * (us[idx + 1] - us[idx]))
        })
        .collect();
    out[n_per_side - 1] = p_le;
    out[0] = chain_start(curves);
    let n = out.len();
    out[n - 1] = chain_end(curves);
    Ok(out)
}

/// Thickness and camber of a fitted airfoil (fractions of the chord).
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub chord: f64,
    pub max_thickness: f64,
    pub max_thickness_x: f64,
    /// The camber of largest magnitude, with its sign.
    pub max_camber: f64,
    pub max_camber_x: f64,
}

/// `(x, y)` of the largest `|y|`, refined by a parabola through its neighbours.
fn peak(x: &[f64], y: &[f64]) -> (f64, f64) {
    let j = (0..y.len()).fold(0, |b, k| if y[k].abs() > y[b].abs() { k } else { b });
    if j > 0 && j + 1 < y.len() {
        let (x0, x1, x2) = (x[j - 1], x[j], x[j + 1]);
        let (y0, y1, y2) = (y[j - 1], y[j], y[j + 1]);
        // Parabola a x² + b x + c through the three points.
        let denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
        let a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
        let b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom;
        let c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / denom;
        if a != 0.0 {
            let xv = -b / (2.0 * a);
            if x0 <= xv && xv <= x2 {
                return (xv, a * xv * xv + b * xv + c);
            }
        }
    }
    (x[j], y[j])
}

/// Thickness and camber along the chord line, as in airfoil databases: the
/// airfoil is moved and scaled so the leading edge (smallest x) is at 0 and
/// the trailing-edge midpoint at (1, 0), and the surfaces are compared at
/// the same x.
pub fn thickness_and_camber(curves: &[BSpline]) -> Result<Shape> {
    const SAMPLES: usize = 4000;
    let (section, u_le, _) = leading_edge(curves);
    let start = curves[0].domain().0;
    let end = curves[curves.len() - 1].domain().1;
    if (section == 0 && u_le <= start) || (section == curves.len() - 1 && u_le >= end) {
        bail!("Leading edge found at an end; expected an airfoil (TE -> LE -> TE).");
    }
    let te = 0.5 * (chain_start(curves) + chain_end(curves));
    let pts = resample(curves, SAMPLES)?;
    let front = (0..pts.len()).fold(0, |b, k| if pts[k].x < pts[b].x { k } else { b });
    let chord = te.x - pts[front].x;
    if chord <= 0.0 {
        bail!("Trailing edge not behind the leading edge along x.");
    }
    let local: Vec<Vec2> = pts.iter().map(|p| (p - Vec2::new(pts[front].x, te.y)) / chord).collect();
    let (mut upper, mut lower) = (&local[..=front], &local[front..]);
    let n = local.len();
    let area: f64 = (0..n).map(|i| local[i].x * local[(i + 1) % n].y - local[i].y * local[(i + 1) % n].x).sum();
    if area < 0.0 {
        std::mem::swap(&mut upper, &mut lower);
    }
    let sorted = |side: &[Vec2]| {
        let mut s = side.to_vec();
        s.sort_by(|a, b| a.x.total_cmp(&b.x));
        (s.iter().map(|p| p.x).collect::<Vec<_>>(), s.iter().map(|p| p.y).collect::<Vec<_>>())
    };
    let (ux, uy) = sorted(upper);
    let (lx, ly) = sorted(lower);
    let fine = cosine_spacing(4 * SAMPLES);
    let mut thickness = Vec::with_capacity(fine.len());
    let mut camber = Vec::with_capacity(fine.len());
    for &x in &fine {
        let (yu, yl) = (interp(x, &ux, &uy), interp(x, &lx, &ly));
        thickness.push(yu - yl);
        camber.push(0.5 * (yu + yl));
    }
    let (max_thickness_x, max_thickness) = peak(&fine, &thickness);
    let (max_camber_x, max_camber) = peak(&fine, &camber);
    Ok(Shape { chord, max_thickness, max_thickness_x, max_camber, max_camber_x })
}

/// Straight segment closing a blunt trailing edge (lower TE to upper TE),
/// of the airfoil's degree; `None` for a sharp trailing edge.
pub fn trailing_edge_segment(curves: &[BSpline]) -> Option<BSpline> {
    let start = chain_end(curves);
    let end = chain_start(curves);
    if (end - start).norm() <= 1e-12 * start.x.abs().max(start.y.abs()).max(1.0) {
        return None;
    }
    let k = curves[0].degree;
    let mut knots = vec![0.0; k + 1];
    knots.extend(std::iter::repeat(1.0).take(k + 1));
    let control_points = (0..=k).map(|i| start + (end - start) * (i as f64 / k as f64)).collect();
    Some(BSpline { degree: k, knots, control_points })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naca_points_and_tolerance() {
        let pts: Vec<Vec2> = Naca4::new("2412", true)
            .unwrap()
            .points(81)
            .iter()
            .map(|p| p.map(|v| (v * 1e6).round_ties_even() / 1e6))
            .collect();
        assert_eq!(pts.len(), 161);
        assert_eq!(pts[80], Vec2::new(0.0, 0.0));
        assert_eq!(decimal_places(&pts), Some(6));
        assert_eq!(default_tolerance(&pts, Some(6)), 1e-6);
        assert!(Naca4::new("24x2", true).is_err());
    }

    #[test]
    fn fminbound_finds_minimum() {
        let (x, _) = fminbound(|x| (x - 0.3) * (x - 0.3), 0.0, 1.0, 1e-14);
        assert!((x - 0.3).abs() < 1e-7);
    }
}
