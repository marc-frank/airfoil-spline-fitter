//! Gauss-Legendre quadrature.

use std::f64::consts::PI;

/// Nodes and weights of the `order`-point Gauss-Legendre rule on `[-1, 1]`,
/// nodes ascending.
pub fn gauss_legendre(order: usize) -> (Vec<f64>, Vec<f64>) {
    let n = order;
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        // Tricomi's initial guess, then Newton on P_n.
        let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut derivative = 0.0;
        for _ in 0..100 {
            let (p, dp) = legendre(n, x);
            derivative = dp;
            let step = p / dp;
            x -= step;
            if step.abs() <= 1e-16 {
                break;
            }
        }
        let (_, dp) = legendre(n, x);
        if dp != 0.0 {
            derivative = dp;
        }
        let w = 2.0 / ((1.0 - x * x) * derivative * derivative);
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    if n % 2 == 1 {
        nodes[n / 2] = 0.0;
    }
    (nodes, weights)
}

/// `P_n(x)` and `P_n'(x)` by the three-term recurrence.
fn legendre(n: usize, x: f64) -> (f64, f64) {
    let (mut p0, mut p1) = (1.0, x);
    if n == 0 {
        return (1.0, 0.0);
    }
    for k in 2..=n {
        let k = k as f64;
        let p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
        p0 = p1;
        p1 = p2;
    }
    let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}

/// Nodes and weights of the rule mapped to `[a, b]`.
pub fn on_interval<'a>(nodes: &'a [f64], weights: &'a [f64], a: f64, b: f64) -> impl Iterator<Item = (f64, f64)> + 'a {
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    nodes.iter().zip(weights).map(move |(&x, &w)| (mid + half * x, half * w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrates_polynomials_exactly() {
        for order in 1..=16 {
            let (x, w) = gauss_legendre(order);
            assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-14);
            let degree = 2 * order - 1;
            let integral: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(degree as i32 - 1)).sum();
            let exact = if (degree - 1) % 2 == 0 { 2.0 / degree as f64 } else { 0.0 };
            assert!((integral - exact).abs() < 1e-13, "order {order}");
        }
    }

    #[test]
    fn matches_known_nodes() {
        let (x, w) = gauss_legendre(2);
        assert!((x[1] - 1.0 / 3f64.sqrt()).abs() < 1e-15);
        assert!((w[0] - 1.0).abs() < 1e-15);
    }
}
