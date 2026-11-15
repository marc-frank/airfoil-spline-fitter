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
