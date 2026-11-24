//! Geometric quality measures: orthogonal deviation, curvature fairness and
//! junction continuity.

use crate::basis::{distinct, linspace};
use crate::chain::largest_extent;
use crate::curve::BSpline;
use crate::quadrature::{gauss_legendre, on_interval};
use crate::Vec2;

/// Samples on every knot span of the domain (`per_span` each) and the end.
fn span_grid(curve: &BSpline, per_span: usize) -> Vec<f64> {
    let p = curve.degree;
    let breaks = distinct(&curve.knots[p..curve.knots.len() - p]);
    let mut grid = Vec::with_capacity(per_span * breaks.len());
    for w in breaks.windows(2) {
        let piece = linspace(w[0], w[1], per_span + 1);
        grid.extend_from_slice(&piece[..per_span]);
    }
    grid.push(breaks[breaks.len() - 1]);
    grid
}

/// Closest curve parameter and orthogonal distance of every point.
///
/// The global minimum is located on 32 samples per knot span; up to three
/// local minima of the sampled distance are refined to machine precision by
/// a bracketed, safeguarded Newton iteration. The curve ends are always
/// candidates.
pub fn project(points: &[Vec2], curve: &BSpline) -> Vec<(f64, f64)> {
    const CANDIDATES: usize = 3;
    let grid = span_grid(curve, 32);
    let samples = curve.points(&grid);
    let (lo_domain, hi_domain) = (grid[0], grid[grid.len() - 1]);
    let ends = [(lo_domain, curve.point(lo_domain)), (hi_domain, curve.point(hi_domain))];
    let last = grid.len() - 1;
    let mut d2 = vec![0.0; grid.len()];
    points
        .iter()
        .map(|q| {
            for (d, s) in d2.iter_mut().zip(&samples) {
                *d = (s - q).norm_squared();
            }
            let is_min = |j: usize| (j == 0 || d2[j] <= d2[j - 1]) && (j == last || d2[j] <= d2[j + 1]);
            let mut minima: Vec<usize> = (0..grid.len()).filter(|&j| is_min(j)).collect();
            minima.sort_by(|&a, &b| d2[a].total_cmp(&d2[b]));
            let mut best: Option<(f64, f64)> = None;
            let mut consider = |u: f64, dist2: f64| {
                if best.is_none_or(|(_, b)| dist2 < b) {
                    best = Some((u, dist2));
                }
            };
            for &j in minima.iter().take(CANDIDATES) {
                let u = refine_foot(curve, *q, grid[j.saturating_sub(1)], grid[(j + 1).min(last)], grid[j]);
                let u = u.clamp(lo_domain, hi_domain);
                consider(u, (curve.point(u) - q).norm_squared());
            }
            for (u, p) in ends {
                consider(u, (p - q).norm_squared());
            }
            let (u, dist2) = best.unwrap();
            (u, dist2.sqrt())
        })
        .collect()
}

/// Bracketed Newton on the squared distance from `u` within `[lo, hi]`.
fn refine_foot(curve: &BSpline, q: Vec2, mut lo: f64, mut hi: f64, mut u: f64) -> f64 {
    for _ in 0..40 {
        let d = curve.derivatives(u, 2);
        let diff = d[0] - q;
        let g = diff.dot(&d[1]);
        let h = d[1].norm_squared() + diff.dot(&d[2]);
        if g > 0.0 {
            hi = u;
        }
        if g < 0.0 {
            lo = u;
        }
        let newton = u - g / h;
        let next = if h > 0.0 && newton > lo && newton < hi { newton } else { 0.5 * (lo + hi) };
        let done = (next - u).abs() <= 1e-15 * (1.0 + u.abs());
        u = next;
        if done {
            break;
        }
    }
    u
}

/// Distance of every point to the nearest of several curves.
pub fn distance_to_curves(points: &[Vec2], curves: &[BSpline]) -> Vec<f64> {
    let mut best = vec![f64::INFINITY; points.len()];
    for curve in curves {
        for (b, (_, d)) in best.iter_mut().zip(project(points, curve)) {
            *b = b.min(d);
        }
    }
    best
}

/// Curvature-comb quality of a chain of planar curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fairness {
    /// Interior curvature extrema of all sections (wiggles in the comb).
    pub curvature_extrema: usize,
    /// `∫(dκ/ds)² ds` over the chain.
    pub roughness: f64,
    pub max_curvature_slope: f64,
    pub max_curvature: f64,
}

/// Fairness of a chain. Curvature reversals smaller than 1e-5 of the
/// largest curvature, or at the level of rounding noise, are not counted.
pub fn fairness(curves: &[BSpline]) -> Fairness {
    let (grids, kappas, threshold, max_curvature) = curvature_samples(curves);
    let curvature_extrema = kappas.iter().map(|k| count_turns(k, threshold)).sum();
    let (nodes, weights) = gauss_legendre(8);
    let mut roughness = 0.0;
    let mut max_curvature_slope: f64 = 0.0;
    for (curve, grid) in curves.iter().zip(&grids) {
        let p = curve.degree;
        let breaks = distinct(&curve.knots[p..curve.knots.len() - p]);
        for w in breaks.windows(2) {
            for (u, weight) in on_interval(&nodes, &weights, w[0], w[1]) {
                let (_, slope) = slope_at(curve, u);
                roughness += weight * slope * slope * curve.derivative(u, 1).norm();
            }
        }
        for &u in grid {
            max_curvature_slope = max_curvature_slope.max(slope_at(curve, u).1.abs());
        }
    }
    Fairness { curvature_extrema, roughness, max_curvature_slope, max_curvature }
}

/// The curvature of every curve on a grid of 64 steps per knot span, the
/// smallest reversal that counts as an extremum (a 1e-5 part of the
/// largest curvature, above rounding) and that largest curvature.
#[allow(clippy::type_complexity)]
fn curvature_samples(curves: &[BSpline]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, f64, f64) {
    const MIN_AMPLITUDE: f64 = 1e-5;
    let grids: Vec<Vec<f64>> = curves.iter().map(|c| span_grid(c, 64)).collect();
    let kappas: Vec<Vec<f64>> = curves.iter().zip(&grids).map(|(c, g)| g.iter().map(|&u| c.curvature(u)).collect()).collect();
    let max_curvature = kappas.iter().flatten().map(|k| k.abs()).fold(f64::NEG_INFINITY, f64::max);
    let size = largest_extent(curves.iter().flat_map(|c| &c.control_points));
    let noise = 1e3 * f64::EPSILON / if size == 0.0 { 1.0 } else { size };
    let threshold = (MIN_AMPLITUDE * max_curvature).max(noise);
    (grids, kappas, threshold, max_curvature)
}

/// Where the curvature has the extrema [`fairness`] counts: `(curve,
/// parameter)` of each.
pub fn curvature_extrema(curves: &[BSpline]) -> Vec<(usize, f64)> {
    let (grids, kappas, threshold, _) = curvature_samples(curves);
    kappas.iter().zip(&grids).enumerate().flat_map(|(s, (k, g))| turns(k, threshold).into_iter().map(move |i| (s, g[i]))).collect()
}

/// Where the signed curvature changes sign along the chain (inflections):
/// `(curve, parameter)` of each. A sign counts once `|κ|` exceeds the
/// threshold of [`fairness`], so rounding noise around a straight stretch
/// is no inflection. The zero is found by bisection between the samples
/// where the sign flips; a flip across a junction (curvature jumping
/// through zero there) is placed at the start of the next curve.
pub fn curvature_inflections(curves: &[BSpline]) -> Vec<(usize, f64)> {
    let (grids, kappas, threshold, _) = curvature_samples(curves);
    // The chain's samples in order: (curve, parameter, κ).
    let samples: Vec<(usize, f64, f64)> =
        grids.iter().zip(&kappas).enumerate().flat_map(|(s, (g, k))| g.iter().zip(k).map(move |(&u, &k)| (s, u, k))).collect();
    let mut found = Vec::new();
    // The last sample whose sign counted, and that sign.
    let mut last: Option<(usize, f64)> = None;
    for (i, &(_, _, k)) in samples.iter().enumerate() {
        if k.abs() <= threshold {
            continue;
        }
        let sign = k.signum();
        if let Some((from, was)) = last {
            if sign != was {
                // The first raw sign change between the two signed samples.
                let j = (from..i).find(|&j| samples[j + 1].2.signum() != was).unwrap_or(i - 1);
                let (sa, ua, _) = samples[j];
                let (sb, ub, _) = samples[j + 1];
                found.push(if sa == sb { (sa, bisect_zero(&curves[sa], ua, ub, was)) } else { (sb, ub) });
            }
        }
        last = Some((i, sign));
    }
    found
}

/// The parameter in `[a, b]` where the curvature of `curve` leaves the sign
/// `sign` it has at `a`.
fn bisect_zero(curve: &BSpline, mut a: f64, mut b: f64, sign: f64) -> f64 {
    for _ in 0..60 {
        let m = 0.5 * (a + b);
        if curve.curvature(m) * sign > 0.0 {
            a = m;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

/// `κ` and `dκ/ds` (the third derivative is zero below degree 3).
fn slope_at(curve: &BSpline, u: f64) -> (f64, f64) {
    curve.curvature_and_slope(u)
}

/// Interior extrema of a sequence, ignoring reversals smaller than `threshold`.
pub fn count_turns(values: &[f64], threshold: f64) -> usize {
    turns(values, threshold).len()
}

/// The indices of the interior extrema [`count_turns`] counts: each
/// reversal larger than `threshold`, at the value it turns from.
pub fn turns(values: &[f64], threshold: f64) -> Vec<usize> {
    let mut turns = Vec::new();
    let mut direction = 0;
    let (mut lo, mut hi) = (values[0], values[0]);
    let (mut lo_at, mut hi_at) = (0, 0);
    for (i, &v) in values.iter().enumerate().skip(1) {
        match direction {
            0 => {
                if v - lo > threshold {
                    direction = 1;
                    (hi, hi_at) = (v, i);
                } else if hi - v > threshold {
                    direction = -1;
                    (lo, lo_at) = (v, i);
                } else {
                    if v < lo {
                        (lo, lo_at) = (v, i);
                    }
                    if v > hi {
                        (hi, hi_at) = (v, i);
                    }
                }
            }
            1 => {
                if v > hi {
                    (hi, hi_at) = (v, i);
                } else if hi - v > threshold {
                    turns.push(hi_at);
                    direction = -1;
                    (lo, lo_at) = (v, i);
                }
            }
            _ => {
                if v < lo {
                    (lo, lo_at) = (v, i);
                } else if v - lo > threshold {
                    turns.push(lo_at);
                    direction = 1;
                    (hi, hi_at) = (v, i);
                }
            }
        }
    }
    turns
}

/// Continuity measured where one curve ends and the next starts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JunctionMeasure {
    pub gap: f64,
    /// Angle between the unit tangents, radians.
    pub tangent_angle: f64,
    pub curvature_jump: f64,
    pub curvature_slope_jump: f64,
}

/// One measure per junction of consecutive curves.
pub fn junctions(curves: &[BSpline]) -> Vec<JunctionMeasure> {
    curves
        .windows(2)
        .map(|w| {
            let (a, b) = (&w[0], &w[1]);
            let (ua, ub) = (a.knots[a.knots.len() - 1], b.knots[0]);
            let ta = a.derivative(ua, 1).normalize();
            let tb = b.derivative(ub, 1).normalize();
            let (ka, sa) = a.curvature_and_slope(ua);
            let (kb, sb) = b.curvature_and_slope(ub);
            JunctionMeasure {
                gap: (a.point(ua) - b.point(ub)).norm(),
                tangent_angle: 2.0 * (ta - tb).norm().atan2((ta + tb).norm()),
                curvature_jump: (ka - kb).abs(),
                curvature_slope_jump: (sa - sb).abs(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reflexed S-curve inflects once, where its curvature is zero, also
    /// when the sign change falls on a junction; a curve that never changes
    /// sign has none, in either direction of travel.
    #[test]
    fn curvature_inflections_are_the_sign_changes() {
        // An S: bends left, then right (a reflex).
        let s = |sign: f64| {
            let points: Vec<Vec2> =
                (0..9).map(|i| { let x = i as f64 / 8.0; Vec2::new(x, sign * 0.1 * (std::f64::consts::TAU * x).sin()) }).collect();
            BSpline::uniform(5, points).unwrap()
        };
        for curve in [s(1.0), s(-1.0)] {
            let found = curvature_inflections(&[curve.clone()]);
            assert_eq!(found.len(), 1, "{found:?}");
            let (_, u) = found[0];
            assert!(curve.curvature(u).abs() < 1e-8 * curve.curvature(0.25).abs().max(1.0), "{}", curve.curvature(u));
            assert!(curve.curvature(u - 0.05) * curve.curvature(u + 0.05) < 0.0);
            // Reversed, the same point.
            let reversed = BSpline { control_points: curve.control_points.iter().rev().copied().collect(), ..curve.clone() };
            let back = curvature_inflections(&[reversed.clone()]);
            assert_eq!(back.len(), 1);
            assert!((reversed.point(back[0].1) - curve.point(u)).norm() < 1e-9);
        }
        // Two arcs meeting tangentially, bending right then left: the
        // curvature jumps through zero at the junction.
        let arc = |p: [[f64; 2]; 3]| BSpline::uniform(2, p.iter().map(|q| Vec2::new(q[0], q[1])).collect()).unwrap();
        let a = arc([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]]);
        let b = arc([[2.0, 1.0], [3.0, 1.0], [4.0, 2.0]]);
        let found = curvature_inflections(&[a, b.clone()]);
        assert_eq!(found, vec![(1, b.domain().0)]);
        // A plain arc-like bump bends one way only.
        let bump: Vec<Vec2> = (0..9).map(|i| { let x = i as f64 / 8.0; Vec2::new(x, 0.1 * (std::f64::consts::PI * x).sin()) }).collect();
        assert!(curvature_inflections(&[BSpline::uniform(5, bump).unwrap()]).is_empty());
    }

    /// The extrema found are the ones the fairness report counts, at the
    /// curvature's turns.
    #[test]
    fn curvature_extrema_are_where_the_report_counts_them() {
        // A wavy curve: its curvature turns several times.
        let points: Vec<Vec2> = (0..12).map(|i| Vec2::new(i as f64 / 11.0, if i % 3 == 0 { 0.05 } else { 0.0 })).collect();
        let curve = BSpline::uniform(5, points).unwrap();
        let curves = [curve.clone()];
        let found = curvature_extrema(&curves);
        assert!(found.len() >= 4, "{found:?}");
        assert_eq!(found.len(), fairness(&curves).curvature_extrema);
        let step = 1e-4;
        for &(s, u) in &found {
            assert_eq!(s, 0);
            // Not a monotone stretch: the curvature turns within a step of it.
            let k = |v: f64| curve.curvature(v.clamp(0.0, 1.0));
            let (before, at, after) = (k(u - 0.02), k(u), k(u + 0.02));
            assert!((at - before) * (after - at) <= step, "{u}: {before} {at} {after}");
        }
    }

    #[test]
    fn projects_onto_a_line() {
        let line = BSpline::uniform(3, (0..6).map(|i| Vec2::new(i as f64 / 5.0, 0.0)).collect()).unwrap();
        let feet = project(&[Vec2::new(0.3, 0.2), Vec2::new(1.5, -1.0)], &line);
        assert!((line.point(feet[0].0) - Vec2::new(0.3, 0.0)).norm() < 1e-12);
        assert!((feet[0].1 - 0.2).abs() < 1e-12);
        assert_eq!(feet[1].0, 1.0);
        assert!((feet[1].1 - (0.25f64 + 1.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn counts_turns() {
        assert_eq!(count_turns(&[0.0, 1.0, 0.0, 1.0, 0.0], 0.5), 3);
        assert_eq!(count_turns(&[0.0, 0.1, 0.0], 0.5), 0);
        // Each at the value it turns from.
        assert_eq!(turns(&[0.0, 1.0, 0.9, 1.2, 0.0, 0.2, 1.0], 0.5), vec![3, 4]);
    }
}
