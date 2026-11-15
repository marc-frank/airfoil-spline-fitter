//! Planar clamped B-spline curves.

use crate::basis::{basis_derivatives, find_span, num_control_points};
use crate::{Result, Vec2};

/// A planar B-spline `C(u) = Σ B_j(u) P_j` of degree `degree` on its knots.
#[derive(Debug, Clone, PartialEq)]
pub struct BSpline {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub control_points: Vec<Vec2>,
}

impl BSpline {
    /// A curve; the knot count must be control points + degree + 1.
    pub fn new(degree: usize, knots: Vec<f64>, control_points: Vec<Vec2>) -> Result<Self> {
        if knots.len() != control_points.len() + degree + 1 || control_points.len() <= degree {
            crate::bail!("Knot vector length must be control points + degree + 1.");
        }
        if knots.windows(2).any(|w| w[1] < w[0]) {
            crate::bail!("Knot vector must be nondecreasing.");
        }
        Ok(BSpline { degree, knots, control_points })
    }

    /// A curve on open uniform knots.
    pub fn uniform(degree: usize, control_points: Vec<Vec2>) -> Result<Self> {
        let knots = crate::open_uniform_knots(degree, control_points.len())?;
        Self::new(degree, knots, control_points)
    }

    /// The parameter domain `(t[p], t[-p-1])`.
    pub fn domain(&self) -> (f64, f64) {
        let p = self.degree;
        (self.knots[p], self.knots[self.knots.len() - p - 1])
    }

    pub fn start(&self) -> Vec2 {
        self.point(self.domain().0)
    }

    pub fn end(&self) -> Vec2 {
        self.point(self.domain().1)
    }

    /// `C(u)`; outside the domain the end polynomials extend the curve.
    pub fn point(&self, u: f64) -> Vec2 {
        self.derivative(u, 0)
    }

    /// The `order`-th derivative `C^(order)(u)` (zero above the degree).
    pub fn derivative(&self, u: f64, order: usize) -> Vec2 {
        self.derivatives(u, order)[order]
    }

    /// `[C(u), C'(u), ..., C^(order)(u)]`.
    pub fn derivatives(&self, u: f64, order: usize) -> Vec<Vec2> {
        let p = self.degree;
        let span = find_span(&self.knots, p, u);
        let ders = basis_derivatives(&self.knots, p, span, u, order);
        (0..=order)
            .map(|k| ders[k].iter().enumerate().map(|(j, &b)| self.control_points[span - p + j] * b).sum())
            .collect()
    }

    pub fn points(&self, u: &[f64]) -> Vec<Vec2> {
        u.iter().map(|&v| self.point(v)).collect()
    }

    pub fn num_control_points(&self) -> usize {
        num_control_points(&self.knots, self.degree)
    }

    /// Signed curvature `κ = (C' × C'') / |C'|³` (no guard: a zero speed gives inf/NaN).
    pub fn curvature(&self, u: f64) -> f64 {
        let d = self.derivatives(u, 2);
        cross(d[1], d[2]) / d[1].norm().powi(3)
    }

    /// `κ` and `dκ/ds` at `u` (no guard).
    pub fn curvature_and_slope(&self, u: f64) -> (f64, f64) {
        let d = self.derivatives(u, 3);
        curvature_and_slope(d[1], d[2], d[3])
    }
}

/// `a.x b.y - a.y b.x`.
pub fn cross(a: Vec2, b: Vec2) -> f64 {
    a.x * b.y - a.y * b.x
}

/// Signed curvature and its arc-length derivative from the first three
/// parametric derivatives: `κ = a/v³`, `dκ/ds = (b v² - 3ac)/v⁶`.
pub fn curvature_and_slope(d1: Vec2, d2: Vec2, d3: Vec2) -> (f64, f64) {
    let v2 = d1.norm_squared();
    let a = cross(d1, d2);
    let b = cross(d1, d3);
    let c = d1.dot(&d2);
    (a / v2.powf(1.5), (b * v2 - 3.0 * a * c) / (v2 * v2 * v2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_circle_like_curve() {
        let p = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0), Vec2::new(2.0, 0.0)];
        let c = BSpline::uniform(2, p).unwrap();
        assert_eq!(c.start(), Vec2::new(0.0, 0.0));
        assert_eq!(c.end(), Vec2::new(2.0, 0.0));
        assert!((c.point(0.5) - Vec2::new(1.0, 0.5)).norm() < 1e-15);
        assert!((c.derivative(0.0, 1) - Vec2::new(2.0, 2.0)).norm() < 1e-14);
        assert!((c.derivative(0.3, 2) - Vec2::new(0.0, -4.0)).norm() < 1e-13);
        assert_eq!(c.derivative(0.3, 3), Vec2::zeros());
    }
}
