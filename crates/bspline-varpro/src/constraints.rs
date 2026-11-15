//! Linear equality constraints `C(θ) P = d` on stacked control points.
//!
//! A chain of clamped sections is joined at junctions. Where section `a`
//! ends and section `b` starts, the end points coincide and derivatives up
//! to the junction order are linked. Parametric continuity `C^k` equates the
//! derivatives; geometric continuity `G^k` lets the curves agree after a
//! local reparameterization with `φ' = α > 0`, `φ'' = β`, `φ''' = γ`:
//!
//! ```text
//! G1:  C_a'   = α  C_b'
//! G2:  C_a''  = α² C_b''  + β C_b'
//! G3:  C_a''' = α³ C_b''' + 3αβ C_b'' + γ C_b'
//! ```
//!
//! The shape parameters enter `C` linearly, so the fit stays separable per
//! coordinate. `α` is kept within a factor [`ALPHA_RANGE`] of the interval
//! between 1 (the parametric join) and the section length ratio:
//! `ln α = c + h·tanh(a/h)`. Unbounded, `α` could collapse to 0 and make the
//! join degenerate (any curvature satisfies "G2" at zero speed).

use std::fmt;
use std::str::FromStr;

use nalgebra::DMatrix;

use crate::basis::{basis_derivatives, find_span, num_control_points};
use crate::{bail, Error, Result, Vec2};

/// Factor by which a geometric junction's speed ratio may leave `[1, ratio]`.
pub const ALPHA_RANGE: f64 = 10.0;

/// Continuity where one section ends and the next begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Junction {
    /// Derivative order linked across the junction (0-3).
    pub order: usize,
    /// Geometric `G^k` (true) or parametric `C^k` (false).
    pub geometric: bool,
    /// Pin the junction to the last data point of the ending section.
    pub fixed: bool,
}

impl Junction {
    pub fn new(order: usize, geometric: bool, fixed: bool) -> Result<Self> {
        if order > 3 {
            bail!("Continuity order must be between 0 and 3.");
        }
        Ok(Junction { order, geometric, fixed })
    }

    /// The same junction with parametric continuity.
    pub fn parametric(self) -> Self {
        Junction { geometric: false, ..self }
    }

    /// `G2`, `C1`, ...
    pub fn label(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for Junction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", if self.geometric { 'G' } else { 'C' }, self.order)
    }
}

impl FromStr for Junction {
    type Err = Error;

    /// `"G2"` (geometric) or `"C1"` (parametric), case-insensitive; fixed.
    fn from_str(spec: &str) -> Result<Self> {
        let mut chars = spec.chars();
        match (chars.next().map(|c| c.to_ascii_uppercase()), chars.next(), chars.next()) {
            (Some(kind @ ('C' | 'G')), Some(digit), None) if digit.is_ascii_digit() => {
                Junction::new(digit.to_digit(10).unwrap() as usize, kind == 'G', true)
            }
            _ => bail!("Invalid continuity {spec:?}; use e.g. 'G2' or 'C1'."),
        }
    }
}

/// Conditions at the start or end of the whole chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct End {
    /// Interpolate the first / last data point.
    pub fixed: bool,
}

impl Default for End {
    fn default() -> Self {
        End { fixed: true }
    }
}

/// Fewest control points per section so that the end conditions of a
/// section do not overlap: `max(p + 1, order_left + order_right + 2)`.
pub fn minimum_control_points(junction_orders: &[usize], degree: usize) -> Vec<usize> {
    let mut orders = vec![0];
    orders.extend_from_slice(junction_orders);
    orders.push(0);
    orders.windows(2).map(|w| (degree + 1).max(w[0] + w[1] + 2)).collect()
}

/// Coefficients `(end, start)` with `C^(r)(1) = end · P[-r-1:]` and
/// `C^(r)(0) = start · P[:r+1]` of a clamped curve.
pub fn end_derivative_coefficients(knots: &[f64], degree: usize, order: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if order == 0 {
        return Ok((vec![1.0], vec![1.0]));
    }
    if order > degree {
        bail!("Derivative order {order} exceeds the spline degree {degree}.");
    }
    let p = degree;
    let n = num_control_points(knots, p);
    let last = knots[knots.len() - 1];
    let end_span = find_span(knots, p, last);
    let end = basis_derivatives(knots, p, end_span, last, order);
    let start_span = find_span(knots, p, knots[0]);
    let start = basis_derivatives(knots, p, start_span, knots[0], order);
    // Basis j of a span is control point span - p + j.
    let end_coef = (n - order - 1..n).map(|i| end[order][i + p - end_span]).collect();
    let start_coef = (0..=order).map(|i| start[order][i + p - start_span]).collect();
    Ok((end_coef, start_coef))
}

/// `(centre, half width)` of the allowed `ln α` for a length ratio.
fn alpha_range(ratio: f64) -> (f64, f64) {
    let log_range = ALPHA_RANGE.ln();
    let low = ratio.ln().min(0.0) - log_range;
    let high = ratio.ln().max(0.0) + log_range;
    (0.5 * (low + high), 0.5 * (high - low))
}

/// `(α, dα/da)` of a junction's speed-ratio parameter.
fn alpha(a: f64, ratio: f64) -> (f64, f64) {
    let (centre, half) = alpha_range(ratio);
    let x = (a / half).tanh();
    let alpha = (centre + half * x).exp();
    (alpha, alpha * (1.0 - x * x))
}

/// Inverse of [`alpha`], clipped just inside the allowed range.
fn alpha_parameter(alpha: f64, ratio: f64) -> f64 {
    let (centre, half) = alpha_range(ratio);
    half * ((alpha.ln() - centre) / half).clamp(-0.999, 0.999).atanh()
}

/// Which shape parameter a slot of `θ` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Alpha,
    Beta,
    Gamma,
}

/// `C(θ)`, `d` and the derivatives `∂C/∂θ_i` (`d` does not depend on `θ`).
#[derive(Debug, Clone)]
pub struct Constraints {
    pub c: DMatrix<f64>,
    pub d: DMatrix<f64>,
    pub dc: Vec<DMatrix<f64>>,
}

/// Builds the junction and end constraints of a section chain.
///
/// `θ` holds, for each geometric junction of order `k >= 1` in order, the
/// `α` parameter, then `β` (`k >= 2`) and `γ` (`k = 3`).
#[derive(Debug, Clone)]
pub struct ChainConstraints {
    degree: usize,
    junctions: Vec<Junction>,
    start: End,
    end: End,
    offsets: Vec<usize>,
    /// First and last data point of every section.
    ends: Vec<(Vec2, Vec2)>,
    /// Data polyline length of every section (1 if zero).
    lengths: Vec<f64>,
    /// `(end, start)` derivative coefficients per section and order 0..=3.
    coefficients: Vec<Vec<(Vec<f64>, Vec<f64>)>>,
    slots: Vec<(usize, Shape)>,
}

impl ChainConstraints {
    pub fn new(sections: &[Vec<Vec2>], knots: &[Vec<f64>], degree: usize, junctions: Vec<Junction>, start: End, end: End) -> Result<Self> {
        if junctions.len() + 1 != sections.len() {
            bail!("Need one junction per pair of consecutive sections.");
        }
        for (j, junction) in junctions.iter().enumerate() {
            if junction.order > degree {
                bail!("Junction {j}: continuity order exceeds the degree.");
            }
        }
        let counts: Vec<usize> = knots.iter().map(|k| num_control_points(k, degree)).collect();
        let orders: Vec<usize> = junctions.iter().map(|j| j.order).collect();
        for (s, (&n, need)) in counts.iter().zip(minimum_control_points(&orders, degree)).enumerate() {
            if n < need {
                bail!("Section {s} needs at least {need} control points for its end conditions.");
            }
        }
        let mut offsets = vec![0];
        for n in &counts {
            offsets.push(offsets.last().unwrap() + n);
        }
        let coefficients = knots
            .iter()
            .map(|k| (0..=degree.min(3)).map(|r| end_derivative_coefficients(k, degree, r)).collect::<Result<_>>())
            .collect::<Result<_>>()?;
        let lengths = sections
            .iter()
            .map(|s| {
                let length: f64 = s.windows(2).map(|w| (w[1] - w[0]).norm()).sum();
                if length == 0.0 {
                    1.0
                } else {
                    length
                }
            })
            .collect();
        let slots = junctions
            .iter()
            .enumerate()
            .filter(|(_, jn)| jn.geometric)
            .flat_map(|(j, jn)| [Shape::Alpha, Shape::Beta, Shape::Gamma].into_iter().take(jn.order).map(move |s| (j, s)))
            .collect();
        Ok(ChainConstraints {
            degree,
            junctions,
            start,
            end,
            offsets,
            ends: sections.iter().map(|s| (s[0], s[s.len() - 1])).collect(),
            lengths,
            coefficients,
            slots,
        })
    }

    pub fn junctions(&self) -> &[Junction] {
        &self.junctions
    }

    pub fn num_params(&self) -> usize {
        self.slots.len()
    }

    pub fn has_shape_parameters(&self) -> bool {
        !self.slots.is_empty()
    }

    fn ratio(&self, junction: usize) -> f64 {
        self.lengths[junction] / self.lengths[junction + 1]
    }

    /// Start values: each `α` at the section length ratio, `β = γ = 0`.
    pub fn initial_params(&self) -> Vec<f64> {
        self.slots
            .iter()
            .map(|&(j, shape)| if shape == Shape::Alpha { alpha_parameter(self.ratio(j), self.ratio(j)) } else { 0.0 })
            .collect()
    }

    /// `θ` of the parametric join: `α = 1`, `β = γ = 0`.
    pub fn parametric_params(&self) -> Vec<f64> {
        self.slots.iter().map(|&(j, shape)| if shape == Shape::Alpha { alpha_parameter(1.0, self.ratio(j)) } else { 0.0 }).collect()
    }

    /// `ln(α/ratio)` per `α` slot and its derivative `(dα/da)/α` with
    /// respect to that slot: `(slot index, value, derivative)`.
    pub fn alpha_deviations(&self, theta: &[f64]) -> Vec<(usize, f64, f64)> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, (_, shape))| *shape == Shape::Alpha)
            .map(|(i, &(j, _))| {
                let ratio = self.ratio(j);
                let (alpha, dalpha) = alpha(theta[i], ratio);
                (i, (alpha / ratio).ln(), dalpha / alpha)
            })
            .collect()
    }

    pub fn num_alpha(&self) -> usize {
        self.slots.iter().filter(|(_, s)| *s == Shape::Alpha).count()
    }

    /// `(α, β, γ)` per junction; `(1, 0, 0)` for parametric or order-0 joins.
    pub fn shape_values(&self, theta: &[f64]) -> Vec<[f64; 3]> {
        let mut values = vec![[1.0, 0.0, 0.0]; self.junctions.len()];
        for (i, &(j, shape)) in self.slots.iter().enumerate() {
            match shape {
                Shape::Alpha => values[j][0] = alpha(theta[i], self.ratio(j)).0,
                Shape::Beta => values[j][1] = theta[i],
                Shape::Gamma => values[j][2] = theta[i],
            }
        }
        values
    }

    fn slot(&self, junction: usize, shape: Shape) -> Option<usize> {
        self.slots.iter().position(|&s| s == (junction, shape))
    }

    /// Row with the order-`r` end (`at_end`) or start coefficients of a section.
    fn place(&self, row: &mut [f64], section: usize, order: usize, at_end: bool, scale: f64) {
        let (end, start) = &self.coefficients[section][order];
        if at_end {
            let stop = self.offsets[section + 1];
            for (k, c) in end.iter().enumerate() {
                row[stop - order - 1 + k] += scale * c;
            }
        } else {
            let first = self.offsets[section];
            for (k, c) in start.iter().enumerate() {
                row[first + k] += scale * c;
            }
        }
    }

    /// `C(θ)`, `d` and `∂C/∂θ`.
    pub fn build(&self, theta: &[f64]) -> Constraints {
        let total = *self.offsets.last().unwrap();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut values: Vec<Vec2> = Vec::new();
        let mut derivatives: Vec<Vec<(usize, Vec<f64>)>> = Vec::new();
        let new_row = || vec![0.0; total];
        let mut add = |row: Vec<f64>, value: Vec2, drow: Vec<(usize, Vec<f64>)>| {
            rows.push(row);
            values.push(value);
            derivatives.push(drow);
        };
        let last = self.ends.len() - 1;
        if self.start.fixed {
            let mut row = new_row();
            self.place(&mut row, 0, 0, false, 1.0);
            add(row, self.ends[0].0, vec![]);
        }
        if self.end.fixed {
            let mut row = new_row();
            self.place(&mut row, last, 0, true, 1.0);
            add(row, self.ends[last].1, vec![]);
        }
        for (j, jn) in self.junctions.iter().enumerate() {
            let (a, b) = (j, j + 1);
            let mut row = new_row();
            self.place(&mut row, a, 0, true, 1.0);
            self.place(&mut row, b, 0, false, -1.0);
            add(row, Vec2::zeros(), vec![]);
            if jn.fixed {
                let mut row = new_row();
                self.place(&mut row, a, 0, true, 1.0);
                add(row, self.ends[a].1, vec![]);
            }
            if jn.order == 0 {
                continue;
            }
            let (ia, ib, ig) = (self.slot(j, Shape::Alpha), self.slot(j, Shape::Beta), self.slot(j, Shape::Gamma));
            let (al, dal) = ia.map_or((1.0, 0.0), |i| alpha(theta[i], self.ratio(j)));
            let be = ib.map_or(0.0, |i| theta[i]);
            let ga = ig.map_or(0.0, |i| theta[i]);
            let start_row = |order: usize| {
                let mut row = new_row();
                self.place(&mut row, b, order, false, 1.0);
                row
            };
            let end_row = |order: usize| {
                let mut row = new_row();
                self.place(&mut row, a, order, true, 1.0);
                row
            };
            let combine = |base: Vec<f64>, terms: &[(f64, &Vec<f64>)]| -> Vec<f64> {
                let mut row = base;
                for (scale, other) in terms {
                    for (r, o) in row.iter_mut().zip(other.iter()) {
                        *r += scale * o;
                    }
                }
                row
            };
            let scaled = |scale: f64, v: &Vec<f64>| v.iter().map(|x| scale * x).collect::<Vec<f64>>();
            let s1 = start_row(1);
            let s2 = if jn.order >= 2 { start_row(2) } else { vec![] };
            let s3 = if jn.order >= 3 { start_row(3) } else { vec![] };

            let row = combine(end_row(1), &[(-al, &s1)]);
            let drow = ia.map(|i| vec![(i, scaled(-dal, &s1))]).unwrap_or_default();
            add(row, Vec2::zeros(), drow);
            if jn.order >= 2 {
                let row = combine(end_row(2), &[(-al * al, &s2), (-be, &s1)]);
                let drow = match (ia, ib) {
                    (Some(ia), Some(ib)) => vec![(ia, scaled(-2.0 * al * dal, &s2)), (ib, scaled(-1.0, &s1))],
                    _ => vec![],
                };
                add(row, Vec2::zeros(), drow);
            }
            if jn.order >= 3 {
                let row = combine(end_row(3), &[(-al * al * al, &s3), (-3.0 * al * be, &s2), (-ga, &s1)]);
                let drow = match (ia, ib, ig) {
                    (Some(ia), Some(ib), Some(ig)) => vec![
                        (ia, combine(scaled(-3.0 * al * al * dal, &s3), &[(-3.0 * be * dal, &s2)])),
                        (ib, scaled(-3.0 * al, &s2)),
                        (ig, scaled(-1.0, &s1)),
                    ],
                    _ => vec![],
                };
                add(row, Vec2::zeros(), drow);
            }
        }
        let c = DMatrix::from_fn(rows.len(), total, |r, k| rows[r][k]);
        let d = DMatrix::from_fn(values.len(), 2, |r, k| values[r][k]);
        let dc = (0..self.num_params())
            .map(|i| {
                let mut m = DMatrix::zeros(rows.len(), total);
                for (r, drow) in derivatives.iter().enumerate() {
                    for (slot, values) in drow {
                        if *slot == i {
                            m.row_mut(r).copy_from_slice(values);
                        }
                    }
                }
                m
            })
            .collect();
        Constraints { c, d, dc }
    }

    /// The stacked control-point offsets of the sections.
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub fn degree(&self) -> usize {
        self.degree
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{open_uniform_knots, BSpline};

    #[test]
    fn coefficients_reproduce_end_derivatives() {
        let t = crate::clamped_knots(5, &[0.1, 0.3, 0.3, 0.7]);
        let n = t.len() - 6;
        let p: Vec<Vec2> = (0..n).map(|i| Vec2::new(i as f64, ((i * i) % 7) as f64)).collect();
        let curve = BSpline::new(5, t.clone(), p.clone()).unwrap();
        for r in 0..=3 {
            let (end, start) = end_derivative_coefficients(&t, 5, r).unwrap();
            let e: Vec2 = end.iter().enumerate().map(|(k, c)| p[n - r - 1 + k] * *c).sum();
            let s: Vec2 = start.iter().enumerate().map(|(k, c)| p[k] * *c).sum();
            assert!((e - curve.derivative(1.0, r)).norm() < 1e-9 * (1.0 + e.norm()));
            assert!((s - curve.derivative(0.0, r)).norm() < 1e-9 * (1.0 + s.norm()));
        }
    }

    #[test]
    fn parsing_and_minimums() {
        let j: Junction = "g2".parse().unwrap();
        assert_eq!(j.label(), "G2");
        assert!("X2".parse::<Junction>().is_err());
        assert_eq!(minimum_control_points(&[3], 5), vec![6, 6]);
        assert_eq!(minimum_control_points(&[3, 3], 1), vec![5, 8, 5]);
        assert_eq!(minimum_control_points(&[], 2), vec![3]);
    }

    #[test]
    fn theta_derivatives_match_finite_differences() {
        let sections = vec![
            (0..10).map(|i| Vec2::new(1.0 - i as f64 / 9.0, 0.1 * (i as f64).sin())).collect::<Vec<_>>(),
            (0..14).map(|i| Vec2::new(i as f64 / 13.0, -0.05 * (i as f64).cos())).collect(),
        ];
        let mut sections = sections;
        sections[1][0] = sections[0][9];
        let knots = vec![open_uniform_knots(5, 9).unwrap(); 2];
        let chain =
            ChainConstraints::new(&sections, &knots, 5, vec![Junction::new(3, true, true).unwrap()], End::default(), End::default())
                .unwrap();
        let theta = vec![0.3, -0.2, 0.15];
        let base = chain.build(&theta);
        for i in 0..3 {
            let mut t = theta.clone();
            t[i] += 1e-7;
            let fd = (chain.build(&t).c - &base.c) / 1e-7;
            let scale = base.dc[i].abs().max().max(1.0);
            assert!((fd - &base.dc[i]).abs().max() < 1e-5 * scale, "slot {i}");
        }
    }
}
