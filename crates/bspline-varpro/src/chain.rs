//! The data of a chain as one ordered point sequence.
//!
//! The sections given to a fit are only the starting split: the chain is
//! fitted to all points as a whole. Every point has a chain parameter `u` in
//! `[0, S]`; it lies on section `floor(u)` at local parameter `u - floor(u)`
//! (`u = S` on the last section at 1), so a point moves across a junction
//! when its parameter does. A point shared by consecutive sections (their
//! junction) is one data point.

use crate::basis::centripetal_parameters;
use crate::curve::BSpline;
use crate::diagnostics::distance_to_curves;
use crate::{bail, Result, Vec2};

/// Whether each section after the first starts at the last point of the one
/// before (equal within rounding of the data's extent).
pub fn shared_junctions(sections: &[Vec<Vec2>]) -> Vec<bool> {
    let extent = largest_extent(sections.iter().flatten());
    let tolerance = 1e-12 * extent.max(1e-300);
    sections
        .windows(2)
        .map(|w| {
            let d = w[0][w[0].len() - 1] - w[1][0];
            d.x.abs().max(d.y.abs()) <= tolerance
        })
        .collect()
}

/// The largest per-axis range of a point set.
pub fn largest_extent<'a>(points: impl IntoIterator<Item = &'a Vec2>) -> f64 {
    let mut lo = Vec2::repeat(f64::INFINITY);
    let mut hi = Vec2::repeat(f64::NEG_INFINITY);
    for p in points {
        lo = lo.inf(p);
        hi = hi.sup(p);
    }
    let range = hi - lo;
    if range.x.is_finite() {
        range.x.max(range.y)
    } else {
        0.0
    }
}

/// All points in order, a shared junction point once.
pub fn chain_data(sections: &[Vec<Vec2>]) -> Vec<Vec2> {
    let shared = shared_junctions(sections);
    let mut data = sections[0].clone();
    for (section, &skip) in sections[1..].iter().zip(&shared) {
        data.extend_from_slice(if skip { &section[1..] } else { section });
    }
    data
}

/// `(first, last)` index in [`chain_data`] of every section's points.
pub fn section_ranges(sections: &[Vec<Vec2>]) -> Vec<(usize, usize)> {
    let shared = shared_junctions(sections);
    let mut ranges = Vec::with_capacity(sections.len());
    let mut start = 0;
    for (s, section) in sections.iter().enumerate() {
        let first = if s > 0 && shared[s - 1] { start - 1 } else { start };
        let last = first + section.len() - 1;
        ranges.push((first, last));
        start = last + 1;
    }
    ranges
}

/// Chain parameters from local parameters in `[0, 1]` per section.
///
/// A shared junction point is one point at the integer. Where sections do
/// not share it, the next section's first point starts halfway along its
/// first step: two points at one parameter could never move apart.
pub fn chain_parameters(local: &[Vec<f64>], sections: &[Vec<Vec2>]) -> Vec<f64> {
    let shared = shared_junctions(sections);
    let mut u = Vec::new();
    for (s, t) in local.iter().enumerate() {
        let mut t = t.clone();
        if s > 0 && shared[s - 1] {
            t.remove(0);
        } else if s > 0 && t.len() > 1 {
            t[0] = 0.5 * t[1];
        }
        u.extend(t.iter().map(|v| v + s as f64));
    }
    u
}

/// Centripetal chain parameters of the sections.
pub fn initial_chain_parameters(sections: &[Vec<Vec2>]) -> Vec<f64> {
    let local: Vec<Vec<f64>> = sections.iter().map(|s| centripetal_parameters(s, 0.5)).collect();
    chain_parameters(&local, sections)
}

/// Chain parameters a caller gives: one nondecreasing value per data point
/// from 0 to `S`, or one array from 0 to 1 per section.
#[derive(Debug, Clone, PartialEq)]
pub enum Parameters {
    Chain(Vec<f64>),
    Sections(Vec<Vec<f64>>),
}

fn isclose(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

fn increasing_from_to(t: &[f64], first: f64, last: f64) -> bool {
    !t.is_empty()
        && t.iter().all(|v| v.is_finite())
        && t.windows(2).all(|w| w[1] >= w[0])
        && isclose(t[0], first)
        && isclose(t[t.len() - 1], last)
}

/// Checks caller-given parameters and converts them to chain parameters.
pub fn validate_chain_parameters(parameters: &Parameters, sections: &[Vec<Vec2>]) -> Result<Vec<f64>> {
    let num_points = chain_data(sections).len();
    let count = sections.len() as f64;
    match parameters {
        Parameters::Chain(u) => {
            if u.len() != num_points || !increasing_from_to(u, 0.0, count) {
                bail!("Chain parameters need one value per point, increasing from 0 to the section count.");
            }
            Ok(u.clone())
        }
        Parameters::Sections(local) => {
            let valid = local.len() == sections.len()
                && local.iter().zip(sections).all(|(t, s)| t.len() == s.len() && increasing_from_to(t, 0.0, 1.0));
            if !valid {
                bail!("Parameters must match each section and increase from 0 to 1.");
            }
            Ok(chain_parameters(local, sections))
        }
    }
}

/// Section and local parameter of a chain parameter.
pub fn locate(u: f64, num_sections: usize) -> (usize, f64) {
    let u = u.clamp(0.0, num_sections as f64);
    let section = (u.floor() as usize).min(num_sections - 1);
    (section, u - section as f64)
}

/// Point indices and local parameters of the points on each section.
pub fn groups(u: &[f64], num_sections: usize) -> Vec<(Vec<usize>, Vec<f64>)> {
    let mut groups = vec![(Vec::new(), Vec::new()); num_sections];
    for (i, &value) in u.iter().enumerate() {
        let (s, local) = locate(value, num_sections);
        groups[s].0.push(i);
        groups[s].1.push(local);
    }
    groups
}

/// The chain point (or its `order`-th derivative) at a chain parameter.
pub fn evaluate_chain(curves: &[BSpline], u: f64, order: usize) -> Vec2 {
    let (s, local) = locate(u, curves.len());
    curves[s].derivative(local, order)
}

/// Orthogonal distance of every point to the nearest point of the chain.
pub fn chain_deviations(data: &[Vec2], curves: &[BSpline]) -> Vec<f64> {
    distance_to_curves(data, curves)
}

/// Chain parameter of the closest chain point to each point within
/// `[lower, upper]`.
///
/// Ordered data lie between their ordered neighbours, so the window keeps a
/// point from jumping to a distant part of the chain that passes close (the
/// other side of a thin trailing edge). `start` is an extra candidate.
pub fn project_between(
    data: &[Vec2],
    curves: &[BSpline],
    lower: &[f64],
    upper: &[f64],
    start: Option<&[f64]>,
) -> Vec<f64> {
    const SAMPLES: usize = 16;
    let n = data.len();
    let upper: Vec<f64> = upper.iter().zip(lower).map(|(u, l)| u.max(*l)).collect();
    let mut u = Vec::with_capacity(n);
    let mut best = Vec::with_capacity(n);
    for i in 0..n {
        let mut candidates: Vec<f64> =
            (0..=SAMPLES).map(|k| lower[i] + (upper[i] - lower[i]) * (k as f64 / SAMPLES as f64)).collect();
        if let Some(start) = start {
            candidates.push(start[i]);
        }
        let (mut best_u, mut best_d) = (candidates[0], f64::INFINITY);
        for c in candidates {
            let d = (evaluate_chain(curves, c, 0) - data[i]).norm_squared();
            if d < best_d {
                best_u = c;
                best_d = d;
            }
        }
        u.push(best_u);
        best.push(best_d);
    }
    // Newton on the squared distance, inside the window, only where it improves.
    for _ in 0..20 {
        let mut improved = false;
        for i in 0..n {
            let (s, local) = locate(u[i], curves.len());
            let d = curves[s].derivatives(local, 2);
            let diff = d[0] - data[i];
            let g = diff.dot(&d[1]);
            let h = d[1].norm_squared() + diff.dot(&d[2]);
            let mut trial = if h > 0.0 { (u[i] - g / h).clamp(lower[i], upper[i]) } else { u[i] };
            if !trial.is_finite() {
                trial = u[i];
            }
            let dist = (evaluate_chain(curves, trial, 0) - data[i]).norm_squared();
            if dist < best[i] {
                u[i] = trial;
                best[i] = dist;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    u
}

/// Chain parameters of all points from those of an ordered subset `chosen`.
pub fn project_to_chain(data: &[Vec2], curves: &[BSpline], chosen: &[usize], u_chosen: &[f64]) -> Vec<f64> {
    let last = chosen.len() - 1;
    let mut lower = Vec::with_capacity(data.len());
    let mut upper = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let left = chosen.partition_point(|&c| c < i);
        let right = chosen.partition_point(|&c| c <= i);
        let before = (left as isize - 1).clamp(0, last as isize) as usize;
        let after = right.min(last);
        lower.push(u_chosen[before]);
        upper.push(u_chosen[after]);
    }
    let s = curves.len() as f64;
    let mut u = project_between(data, curves, &lower, &upper, None);
    let mut running = f64::NEG_INFINITY;
    for v in &mut u {
        running = running.max(v.clamp(0.0, s));
        *v = running;
    }
    u[0] = 0.0;
    let n = u.len();
    u[n - 1] = s;
    u
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sections() -> Vec<Vec<Vec2>> {
        vec![
            vec![Vec2::new(1.0, 0.1), Vec2::new(0.5, 0.1), Vec2::new(0.0, 0.0)],
            vec![Vec2::new(0.0, 0.0), Vec2::new(0.5, -0.1), Vec2::new(1.0, -0.1)],
        ]
    }

    #[test]
    fn shared_junction_is_one_point() {
        let s = sections();
        assert_eq!(shared_junctions(&s), vec![true]);
        assert_eq!(chain_data(&s).len(), 5);
        assert_eq!(section_ranges(&s), vec![(0, 2), (2, 4)]);
        let u = initial_chain_parameters(&s);
        assert_eq!(u.len(), 5);
        assert_eq!((u[0], u[2], u[4]), (0.0, 1.0, 2.0));
    }

    #[test]
    fn locate_ends() {
        assert_eq!(locate(2.0, 2), (1, 1.0));
        assert_eq!(locate(1.0, 2), (1, 0.0));
        assert_eq!(locate(-1.0, 2), (0, 0.0));
    }
}
