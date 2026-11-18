//! Monotone softplus reparameterization of the chain parameters.
//!
//! The chain parameters `0 = u_0 < u_1 < ... < u_{N-1} = S` are cumulative
//! sums of `softplus(η)` scaled to end at `S`, so any unconstrained `η`
//! gives strictly increasing parameters with fixed ends. A point crosses a
//! junction (an integer `u`) by sliding along the chain.

use nalgebra::DMatrix;

/// `η` beyond which the gap is held constant (its derivative is zero there).
pub const ETA_LIMIT: f64 = 700.0;

/// Smallest gap [`t_to_eta`] passes on, relative to the mean gap.
pub const MIN_RELATIVE_GAP: f64 = 1e-3;

/// Numerically stable `ln(1 + eˣ)`.
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.clamp(-ETA_LIMIT, 20.0).exp().ln_1p()
    }
}

/// Numerically stable `1 / (1 + e⁻ˣ)`, the derivative of softplus.
pub fn sigmoid(x: f64) -> f64 {
    let e = (-x.abs()).exp();
    if x >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    }
}

/// `ln(eʸ - 1)` for `y > 0` (`y` floored at 1e-15).
pub fn inverse_softplus(y: f64) -> f64 {
    let y = y.max(1e-15);
    if y > 20.0 {
        y
    } else {
        y.clamp(1e-15, 20.0).exp_m1().ln()
    }
}

/// `η` (one shorter than `u`) reproducing the chain parameters.
///
/// Gaps below [`MIN_RELATIVE_GAP`] times the mean gap are widened: a gap's
/// derivative with respect to its `η` is about the gap itself, so a collapsed
/// gap would otherwise stay collapsed in every warm-started fit.
pub fn t_to_eta(u: &[f64]) -> Vec<f64> {
    let gaps: Vec<f64> = u.windows(2).map(|w| w[1] - w[0]).collect();
    let floor = if gaps.is_empty() { 1e-15 } else { (MIN_RELATIVE_GAP * gaps.iter().sum::<f64>() / gaps.len() as f64).max(1e-15) };
    gaps.iter().map(|&g| inverse_softplus(g.max(floor))).collect()
}

fn sanitize(eta: f64) -> f64 {
    if eta.is_nan() {
        0.0
    } else {
        eta.clamp(-ETA_LIMIT, ETA_LIMIT)
    }
}

/// Chain parameters from 0 to `span` for an `η` vector.
pub fn eta_to_t(eta: &[f64], span: f64) -> Vec<f64> {
    let mut cumsum = Vec::with_capacity(eta.len());
    let mut total = 0.0;
    for &e in eta {
        total += softplus(sanitize(e));
        cumsum.push(total);
    }
    let mut u = Vec::with_capacity(eta.len() + 1);
    u.push(0.0);
    u.extend(cumsum.iter().map(|c| span * c / total));
    u
}

/// `slope_j = S·sigmoid(η_j)/Σsoftplus(η)` (0 where `|η_j|` exceeds
/// [`ETA_LIMIT`]), so that `∂u_k/∂η_j = slope_j·([j < k] - u_k/S)`.
pub fn eta_slopes(eta: &[f64], span: f64) -> Vec<f64> {
    let total: f64 = eta.iter().map(|&e| softplus(sanitize(e))).sum();
    eta.iter()
        .map(|&e| {
            let inside = e.is_nan() || e.abs() <= ETA_LIMIT;
            if inside {
                span * sigmoid(sanitize(e)) / total
            } else {
                0.0
            }
        })
        .collect()
}

/// `∂u/∂η`, shape `(N, N - 1)`.
pub fn eta_jacobian(eta: &[f64], u: &[f64], span: f64) -> DMatrix<f64> {
    let slope = eta_slopes(eta, span);
    DMatrix::from_fn(u.len(), eta.len(), |k, j| slope[j] * (f64::from(u8::from(j < k)) - u[k] / span))
}

/// `J ∂u/∂η` without forming `∂u/∂η`: column `j` is
/// `slope_j (Σ_{k>j} J_k - J u / S)`.
pub fn apply_eta_jacobian(j: &DMatrix<f64>, eta: &[f64], u: &[f64], span: f64) -> DMatrix<f64> {
    let slope = eta_slopes(eta, span);
    let weighted = j * nalgebra::DVector::from_iterator(u.len(), u.iter().map(|v| v / span));
    let mut out = DMatrix::zeros(j.nrows(), eta.len());
    let mut tail = nalgebra::DVector::zeros(j.nrows());
    for c in (0..eta.len()).rev() {
        tail += j.column(c + 1);
        let mut column = out.column_mut(c);
        column.copy_from(&tail);
        column.axpy(-slope[c], &weighted, slope[c]);
    }
    out
}

/// `(∂u/∂η)ᵀ V` without forming `∂u/∂η`.
pub fn apply_eta_jacobian_transpose(v: &DMatrix<f64>, eta: &[f64], u: &[f64], span: f64) -> DMatrix<f64> {
    apply_eta_jacobian(&v.transpose(), eta, u, span).transpose()
}

/// `(∂u/∂η)ᵀ M ∂u/∂η` without forming `∂u/∂η`.
pub fn eta_congruence(m: &DMatrix<f64>, eta: &[f64], u: &[f64], span: f64) -> DMatrix<f64> {
    apply_eta_jacobian_transpose(&apply_eta_jacobian(m, eta, u, span), eta, u, span)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_transpose_and_congruence_match() {
        let eta = [0.3, -1.2, 2.0, 0.1];
        let u = eta_to_t(&eta, 2.0);
        let e = eta_jacobian(&eta, &u, 2.0);
        let v = DMatrix::from_fn(5, 3, |r, c| (r * 3 + c) as f64 * 0.1 - 0.4);
        assert!((apply_eta_jacobian_transpose(&v, &eta, &u, 2.0) - e.transpose() * &v).amax() < 1e-14);
        let m = DMatrix::from_fn(5, 5, |r, c| ((r + 1) * (c + 1)) as f64 * 0.1 + if r == c { 1.0 } else { 0.0 });
        assert!((eta_congruence(&m, &eta, &u, 2.0) - e.transpose() * &m * &e).amax() < 1e-13);
    }

    #[test]
    fn round_trip() {
        let u = [0.0, 0.1, 0.35, 0.9, 1.4, 2.0];
        let back = eta_to_t(&t_to_eta(&u), 2.0);
        for (a, b) in u.iter().zip(&back) {
            assert!((a - b).abs() < 1e-14);
        }
        assert_eq!(back[5], 2.0);
    }

    #[test]
    fn structured_product_matches() {
        let eta = [0.3, -1.2, 2.0, 0.1];
        let u = eta_to_t(&eta, 2.0);
        let j = DMatrix::from_fn(3, 5, |r, c| (r * 5 + c) as f64 * 0.1 - 0.4);
        let direct = &j * eta_jacobian(&eta, &u, 2.0);
        assert!((apply_eta_jacobian(&j, &eta, &u, 2.0) - direct).abs().max() < 1e-14);
    }

    #[test]
    fn jacobian_matches_finite_differences() {
        let eta = [0.3, -1.2, 2.0, 0.1];
        let u = eta_to_t(&eta, 3.0);
        let j = eta_jacobian(&eta, &u, 3.0);
        for c in 0..eta.len() {
            let mut e = eta;
            e[c] += 1e-7;
            let up = eta_to_t(&e, 3.0);
            for r in 0..u.len() {
                assert!(((up[r] - u[r]) / 1e-7 - j[(r, c)]).abs() < 1e-6);
            }
        }
    }
}
