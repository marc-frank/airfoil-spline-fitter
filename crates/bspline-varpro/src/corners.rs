//! Corners of a point sequence, read from the points alone.
//!
//! A corner is a point where the points turn far more than at the points
//! beside it. Two sections joined there with G0 follow it, where one smooth
//! curve squeezes the turn into a curvature spike. [`split_at_corners`] cuts
//! any ordered points into such sections. A section between two corners can
//! have very few points; [`smoothest_through`] gives the points a fit of it
//! needs.

use crate::airfoil::precision_floor;
use crate::chain::largest_extent;
use crate::Vec2;

/// A corner in the data: the points turn at one point far more than at the
/// points beside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corner {
    /// Index of the point.
    pub index: usize,
    /// How far the points turn there, in degrees.
    pub turn: f64,
    /// Its neighbours are too far apart to tell a corner from a tight bend.
    pub sparse: bool,
}

/// The points turn at least this much at a corner (degrees)...
pub const CORNER_TURN: f64 = 20.0;
/// ...and this many times as much as at either neighbour.
pub const CORNER_RATIO: f64 = 6.0;
/// Neighbours farther from the point than this (as a fraction of the points'
/// largest extent, the chord of an airfoil) are too sparse to tell a corner
/// from a rounded bend, unless the corner is sharp...
pub const SPARSE_CORNER_SPACING: f64 = 0.01;
/// ...its sides, extended, meet within this many point spacings of it.
/// Round noses sampled sparsely enough to pass the turn test meet 1.3 or
/// more spacings ahead of the nose point; sharp noses and cut-off noses in
/// a 12,407-file airfoil library meet within 0.5.
pub const SHARP_CORNER_MEET: f64 = 0.5;

/// The corners of ordered points (not within two points of the ends: a
/// corner is read from two points on each side), before any fit.
///
/// Rounded coordinates count for what they can show: the turn at a point
/// less what rounding alone can make of it, against the turns beside it
/// plus the same. The thresholds were chosen on 1,724 airfoils: at a
/// corner, a smooth curve through it peaks in curvature 5 to 200 times
/// higher than two sections joined there with G0. Of 12,407 airfoil files
/// 1,010 have a clear corner: at the nose in all but one airfoil, which has
/// a ridge on its lower side.
pub fn corners(points: &[Vec2]) -> Vec<Corner> {
    let n = points.len();
    if n < 7 {
        return vec![];
    }
    let extent = largest_extent(points);
    let floor = precision_floor(points);
    let turn = |i: usize| {
        let (u, v) = (points[i] - points[i - 1], points[i + 1] - points[i]);
        cross(u, v).atan2(u.dot(&v)).abs().to_degrees()
    };
    // How much rounding alone can change the turn at a point: the tilt it
    // can give each of the two segments.
    let doubt = |i: usize| {
        let tilt = |a: Vec2, b: Vec2| (2.0 * floor).atan2((b - a).norm()).to_degrees();
        tilt(points[i - 1], points[i]) + tilt(points[i], points[i + 1])
    };
    (2..n - 2)
        .filter_map(|i| {
            // The least the point turns, the most its neighbours may.
            let at = turn(i) - doubt(i);
            let beside = (turn(i - 1) + doubt(i - 1)).max(turn(i + 1) + doubt(i + 1));
            (at >= CORNER_TURN && at >= CORNER_RATIO * beside).then(|| {
                let spacing = (points[i] - points[i - 1]).norm().max((points[i + 1] - points[i]).norm());
                let sharp = sides_meet(points, i) <= SHARP_CORNER_MEET * spacing;
                Corner { index: i, turn: turn(i), sparse: spacing > SPARSE_CORNER_SPACING * extent && !sharp }
            })
        })
        .collect()
}

fn cross(u: Vec2, v: Vec2) -> f64 {
    u.x * v.y - u.y * v.x
}

/// How far from point `i` the lines through the two points before it and
/// the two after it meet (infinite when they are parallel).
fn sides_meet(points: &[Vec2], i: usize) -> f64 {
    let (a, r) = (points[i - 1], points[i - 1] - points[i - 2]);
    let (b, s) = (points[i + 1], points[i + 1] - points[i + 2]);
    let den = cross(r, s);
    if den.abs() < 1e-15 {
        return f64::INFINITY;
    }
    (a + cross(b - a, s) / den * r - points[i]).norm()
}

/// Indices of the corners sections can be joined at: the ones the points
/// show clearly.
pub fn clear_corners(points: &[Vec2]) -> Vec<usize> {
    corners(points).iter().filter(|c| !c.sparse).map(|c| c.index).collect()
}

/// Sections of ordered points joined at the given indices (increasing, not
/// the first or last point): each junction point ends one section and
/// starts the next.
pub fn split_at(points: &[Vec2], junctions: &[usize]) -> Vec<Vec<Vec2>> {
    let mut bounds = vec![0];
    bounds.extend_from_slice(junctions);
    bounds.push(points.len() - 1);
    bounds.windows(2).map(|w| points[w[0]..=w[1]].to_vec()).collect()
}

/// Ordered points cut into sections at their clear corners, to be joined
/// there with G0: the sections and the corner indices. One section without
/// corners.
pub fn split_at_corners(points: &[Vec2]) -> (Vec<Vec<Vec2>>, Vec<usize>) {
    let at = clear_corners(points);
    (split_at(points, &at), at)
}

/// `count` points on the smoothest curve through `points`, the given points
/// among them: a natural cubic spline over their chord lengths (through two
/// points, the straight line). A section with fewer points than it has
/// control points is fitted to these, so that its own points, and not a
/// penalty, decide every control point. The added points go where the
/// spacing is widest. Points that are `count` or more already, or fewer
/// than two, are returned as they are.
pub fn smoothest_through(points: &[Vec2], count: usize) -> Vec<Vec2> {
    let m = points.len();
    if m < 2 || count <= m {
        return points.to_vec();
    }
    let h: Vec<f64> = points.windows(2).map(|w| (w[1] - w[0]).norm()).collect();
    // Second derivatives of the spline at the points: zero at both ends,
    // the others from the tridiagonal system of a natural spline.
    let mut second = vec![Vec2::zeros(); m];
    if m > 2 && h.iter().all(|&v| v > 0.0) {
        let k = m - 2;
        let mut diagonal: Vec<f64> = (0..k).map(|j| 2.0 * (h[j] + h[j + 1])).collect();
        let mut right: Vec<Vec2> =
            (0..k).map(|j| 6.0 * ((points[j + 2] - points[j + 1]) / h[j + 1] - (points[j + 1] - points[j]) / h[j])).collect();
        for j in 1..k {
            let w = h[j] / diagonal[j - 1];
            diagonal[j] -= w * h[j];
            let before = right[j - 1];
            right[j] -= w * before;
        }
        for j in (0..k).rev() {
            let after = second[j + 2];
            second[j + 1] = (right[j] - h[j + 1] * after) / diagonal[j];
        }
    }
    // Each added point halves the widest spacing left.
    let mut added = vec![0usize; m - 1];
    for _ in m..count {
        let widest = (0..m - 1).fold(0, |b, g| if h[g] / (added[g] + 1) as f64 > h[b] / (added[b] + 1) as f64 { g } else { b });
        added[widest] += 1;
    }
    let mut out = Vec::with_capacity(count);
    for g in 0..m - 1 {
        out.push(points[g]);
        for q in 1..=added[g] {
            let b = q as f64 / (added[g] + 1) as f64;
            let a = 1.0 - b;
            let bend = ((a * a * a - a) * second[g] + (b * b * b - b) * second[g + 1]) * (h[g] * h[g] / 6.0);
            out.push(a * points[g] + b * points[g + 1] + bend);
        }
    }
    out.push(points[m - 1]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::airfoil::Naca4;

    /// A wedge outline: straight sides meeting at the nose, `step` apart in x,
    /// with a flap-like kink at x = `kink` on the lower side when given.
    fn wedge(step: f64, kink: Option<f64>) -> Vec<Vec2> {
        let xs: Vec<f64> = (0..=(1.0 / step).round() as usize).map(|i| 1.0 - i as f64 * step).collect();
        let half = |x: f64| 0.1 * x.min(1.0 - x) + 0.02 * x;
        let lower = |x: f64| -half(x) - kink.map_or(0.0, |k| 0.5 * (x - k).max(0.0));
        xs.iter().map(|&x| Vec2::new(x, half(x))).chain(xs.iter().rev().skip(1).map(|&x| Vec2::new(x, lower(x)))).collect()
    }

    fn points(xy: &[[f64; 2]]) -> Vec<Vec2> {
        xy.iter().map(|p| Vec2::new(p[0], p[1])).collect()
    }

    /// Corners are found wherever the points bend at one point; NACA
    /// outlines have none however finely sampled. Sparse points show a
    /// sharp nose clearly, but not a thin round one.
    #[test]
    fn corners_are_found_anywhere() {
        let at = |pts: &[Vec2]| corners(pts).iter().map(|c| (pts[c.index].x, c.sparse)).collect::<Vec<_>>();
        assert_eq!(at(&wedge(0.005, None)), [(0.0, false)]);
        assert_eq!(at(&wedge(0.05, None)), [(0.0, false)]);
        let stations = [0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let naca = Naca4::new("0006", true).unwrap();
        let upper = stations.iter().rev().map(|x: &f64| naca.surface(x.sqrt(), true));
        let sparse_round: Vec<Vec2> = upper.chain(stations[1..].iter().map(|x| naca.surface(x.sqrt(), false))).collect();
        assert_eq!(at(&sparse_round), [(0.0, true)]);
        // A nose cut off straight (rc0864c): a corner at each end of the cut.
        let cut = points(&[
            [0.09887, 0.03383],
            [0.08010, 0.03099],
            [0.07386, 0.02993],
            [0.05744, 0.02689],
            [0.04891, 0.02506],
            [0.03808, 0.02245],
            [0.02391, 0.01830],
            [0.01152, 0.01308],
            [0.01016, 0.01245],
            [0.00034, 0.00700],
            [0.0, 0.0],
            [0.00034, -0.00585],
            [0.01016, -0.00975],
            [0.01348, -0.01088],
            [0.02609, -0.01424],
            [0.03808, -0.01666],
            [0.05109, -0.01836],
        ]);
        assert_eq!(at(&cut), [(0.00034, false), (0.00034, false)]);
        // A round nose written with three decimals (mh34): its sharpest
        // bends are within what rounding can make of them; the same points,
        // not recognisably rounded, show two corners.
        let rounded = points(&[
            [0.101, 0.040],
            [0.073, 0.035],
            [0.050, 0.029],
            [0.031, 0.023],
            [0.016, 0.016],
            [0.006, 0.009],
            [0.001, 0.003],
            [0.0, 0.001],
            [0.0, 0.0],
            [0.0, -0.001],
            [0.001, -0.002],
            [0.002, -0.003],
            [0.006, -0.007],
            [0.017, -0.011],
            [0.033, -0.016],
            [0.054, -0.020],
            [0.080, -0.024],
            [0.110, -0.026],
            [0.144, -0.029],
        ]);
        assert_eq!(at(&rounded), []);
        let unrounded: Vec<Vec2> = rounded.iter().map(|p| p * std::f64::consts::PI).collect();
        assert_eq!(corners(&unrounded).iter().map(|c| c.index).collect::<Vec<_>>(), [9, 12]);
        let kinked = wedge(0.005, Some(0.75));
        let found = at(&kinked);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!((found[0].0 - 0.0).abs() < 1e-12 && (found[1].0 - 0.75).abs() < 1e-9, "{found:?}");
        for n in [21, 81, 201] {
            assert!(corners(&Naca4::new("2412", true).unwrap().points(n)).is_empty(), "{n} points per side");
        }
    }

    /// The corner test knows no chord line: the same corners whichever way
    /// the points run, turned or scaled.
    #[test]
    fn corners_are_those_of_any_curve() {
        let kinked = wedge(0.005, Some(0.75));
        let found = clear_corners(&kinked);
        let n = kinked.len();
        let reversed: Vec<Vec2> = kinked.iter().rev().copied().collect();
        assert_eq!(clear_corners(&reversed), found.iter().rev().map(|i| n - 1 - i).collect::<Vec<_>>());
        let (sin, cos) = 1.1f64.sin_cos();
        let turned: Vec<Vec2> =
            kinked.iter().map(|p| Vec2::new(250.0 * (cos * p.x - sin * p.y), 250.0 * (sin * p.x + cos * p.y) + 40.0)).collect();
        assert_eq!(clear_corners(&turned), found);
        // Cut there, every section starts where the one before ends.
        let (sections, at) = split_at_corners(&kinked);
        assert_eq!((sections.len(), &at), (3, &found));
        assert_eq!(sections.iter().map(Vec::len).sum::<usize>(), n + 2);
        assert!(sections.windows(2).all(|w| w[0].last() == w[1].first()));
        // No corners: one section.
        let smooth = Naca4::new("2412", true).unwrap().points(41);
        assert_eq!(split_at_corners(&smooth).0, [smooth.clone()]);
    }

    /// The points a section of very few points is fitted to: its own, with
    /// more on the smoothest curve through them where they are farthest
    /// apart.
    #[test]
    fn few_points_get_company_on_their_smoothest_curve() {
        // Two points: the straight line, evenly.
        let ends = points(&[[0.0, 0.0], [1.0, 2.0]]);
        let line = smoothest_through(&ends, 5);
        assert_eq!(line.len(), 5);
        for (i, p) in line.iter().enumerate() {
            assert!((p - Vec2::new(0.25 * i as f64, 0.5 * i as f64)).norm() < 1e-15, "{p:?}");
        }
        // Three in a line, unevenly spaced: all on the line, the widest gap
        // filled first.
        let three = points(&[[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]]);
        let filled = smoothest_through(&three, 6);
        assert_eq!(filled.len(), 6);
        assert!(filled.iter().all(|p| (p.x - p.y).abs() < 1e-12), "{filled:?}");
        assert!(filled.windows(2).all(|w| w[1].x > w[0].x), "{filled:?}");
        assert_eq!(filled.iter().filter(|p| p.x > 1.0 && p.x < 5.0).count(), 3);
        assert!(three.iter().all(|p| filled.contains(p)));
        // Four on a quarter circle: the added points stay near the arc,
        // less than half as far from it as the straight lines between the
        // points (the ends of a natural spline are flat, so not on it).
        let arc: Vec<Vec2> = (0..4).map(|i| (i as f64 * std::f64::consts::FRAC_PI_6).sin_cos()).map(|(s, c)| Vec2::new(c, s)).collect();
        let round = smoothest_through(&arc, 7);
        assert_eq!(round.len(), 7);
        assert!(arc.iter().all(|p| round.contains(p)));
        let off = round.iter().map(|p| (p.norm() - 1.0).abs()).fold(0.0, f64::max);
        let chords_off = 1.0 - (0.5 * std::f64::consts::FRAC_PI_6).cos();
        assert!(off > 0.0 && off < 0.5 * chords_off, "{off} {chords_off}");
        // Enough already, or nothing to go by: unchanged.
        assert_eq!(smoothest_through(&three, 3), three);
        assert_eq!(smoothest_through(&three[..1], 4), three[..1]);
    }
}
