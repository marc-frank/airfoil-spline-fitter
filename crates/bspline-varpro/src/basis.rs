//! Knot vectors, basis matrices, bending energy and initial data parameters.
//!
//! Every knot vector is clamped on `[0, 1]`: `p + 1` zeros, the interior
//! knots, `p + 1` ones. On an interior knot the basis of the span to the
//! right is used; at the domain end the last span.

use nalgebra::DMatrix;

use crate::linalg;
use crate::quadrature::{gauss_legendre, on_interval};
use crate::{bail, Result, Vec2};

/// Clamped knots with `n - p - 1` evenly spaced interior knots.
pub fn open_uniform_knots(degree: usize, num_control_points: usize) -> Result<Vec<f64>> {
    if degree < 1 {
        bail!("Degree must be at least 1, got {degree}.");
    }
    if num_control_points <= degree {
        bail!("Number of control points ({num_control_points}) must be greater than degree ({degree}).");
    }
    let step = 1.0 / (num_control_points - degree) as f64;
    let interior: Vec<f64> = (1..num_control_points - degree).map(|i| i as f64 * step).collect();
    Ok(clamped_knots(degree, &interior))
}

/// Clamped knots around the given interior values (sorted).
pub fn clamped_knots(degree: usize, interior: &[f64]) -> Vec<f64> {
    let mut sorted = interior.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut knots = vec![0.0; degree + 1];
    knots.extend(sorted);
    knots.extend(std::iter::repeat(1.0).take(degree + 1));
    knots
}

/// The interior knots of a clamped knot vector.
pub fn interior_knots(knots: &[f64], degree: usize) -> &[f64] {
    &knots[degree + 1..knots.len() - degree - 1]
}

/// Number of control points of a knot vector.
pub fn num_control_points(knots: &[f64], degree: usize) -> usize {
    knots.len() - degree - 1
}

/// Checks a clamped knot vector on `[0, 1]`.
pub fn validate_knots(knots: &[f64], degree: usize) -> Result<()> {
    let p = degree;
    if knots.len() < 2 * (p + 1) || knots.iter().any(|k| !k.is_finite()) {
        bail!("Knot vector needs at least {} finite values.", 2 * (p + 1));
    }
    if knots.windows(2).any(|w| w[1] < w[0]) {
        bail!("Knot vector must be nondecreasing.");
    }
    if knots[..=p].iter().any(|&k| k != 0.0) || knots[knots.len() - p - 1..].iter().any(|&k| k != 1.0) {
        bail!("Knot vector must be clamped on [0, 1] (degree + 1 zeros and ones).");
    }
    let inner = interior_knots(knots, p);
    if let (Some(&first), Some(&last)) = (inner.first(), inner.last()) {
        if first <= 0.0 || last >= 1.0 {
            bail!("Interior knots must lie strictly inside (0, 1).");
        }
        if runs(inner).any(|(_, count)| count > p) {
            bail!("Interior knot multiplicity must not exceed the degree.");
        }
    }
    Ok(())
}

/// Runs of equal values of a sorted slice as `(value, count)`.
pub fn runs(values: &[f64]) -> impl Iterator<Item = (f64, usize)> + '_ {
    let mut i = 0;
    std::iter::from_fn(move || {
        let &value = values.get(i)?;
        let count = values[i..].iter().take_while(|&&v| v == value).count();
        i += count;
        Some((value, count))
    })
}

/// Distinct values of a sorted slice.
pub fn distinct(values: &[f64]) -> Vec<f64> {
    runs(values).map(|(v, _)| v).collect()
}

/// Index `i` of the knot span `[t_i, t_{i+1})` holding `u` (the last
/// nonempty span at the domain end and beyond; the first before it).
pub fn find_span(knots: &[f64], degree: usize, u: f64) -> usize {
    let n = knots.len() - degree - 1;
    if u >= knots[n] {
        return (degree..n).rev().find(|&i| knots[i] < knots[i + 1]).unwrap_or(n - 1);
    }
    if u <= knots[degree] {
        return (degree..n).find(|&i| knots[i] < knots[i + 1]).unwrap_or(degree);
    }
    // largest i in [p, n) with t_i <= u
    let (mut low, mut high) = (degree, n);
    while high - low > 1 {
        let mid = (low + high) / 2;
        if knots[mid] <= u {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

/// Values and derivatives up to `order` of the `p + 1` basis functions that
/// are nonzero on span `span` (Piegl & Tiller A2.3): `ders[k][j]` is the
/// `k`-th derivative of `B_{span - p + j}`. Orders above `p` are zero.
pub fn basis_derivatives(knots: &[f64], degree: usize, span: usize, u: f64, order: usize) -> Vec<Vec<f64>> {
    let p = degree;
    let mut ndu = vec![vec![0.0; p + 1]; p + 1];
    let mut left = vec![0.0; p + 1];
    let mut right = vec![0.0; p + 1];
    ndu[0][0] = 1.0;
    for j in 1..=p {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    let mut ders = vec![vec![0.0; p + 1]; order + 1];
    for j in 0..=p {
        ders[0][j] = ndu[j][p];
    }
    let top = order.min(p);
    let mut a = vec![vec![0.0; p + 1]; 2];
    for r in 0..=p {
        let (mut s1, mut s2) = (0usize, 1usize);
        a[0][0] = 1.0;
        for k in 1..=top {
            let mut d = 0.0;
            let rk = r as isize - k as isize;
            let pk = p - k;
            if r >= k {
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk as usize];
                d = a[s2][0] * ndu[rk as usize][pk];
            }
            let j1 = if rk >= -1 { 1 } else { (-rk) as usize };
            let j2 = if (r as isize - 1) <= pk as isize { k - 1 } else { p - r };
            for j in j1..=j2 {
                let idx = (rk + j as isize) as usize;
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][idx];
                d += a[s2][j] * ndu[idx][pk];
            }
            if r <= pk {
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                d += a[s2][k] * ndu[r][pk];
            }
            ders[k][r] = d;
            std::mem::swap(&mut s1, &mut s2);
        }
    }
    let mut factor = p as f64;
    for k in 1..=top {
        for value in &mut ders[k] {
            *value *= factor;
        }
        factor *= (p - k) as f64;
    }
    ders
}

/// Dense matrix of the `order`-th derivatives: `N[i, j] = B_j^(order)(clip(u_i, 0, 1))`.
pub fn basis_matrix(u: &[f64], knots: &[f64], degree: usize, order: usize) -> DMatrix<f64> {
    let n = num_control_points(knots, degree);
    let mut matrix = DMatrix::zeros(u.len(), n);
    if order > degree {
        return matrix;
    }
    for (row, &value) in u.iter().enumerate() {
        let value = value.clamp(0.0, 1.0);
        let span = find_span(knots, degree, value);
        let ders = basis_derivatives(knots, degree, span, value, order);
        for (j, &d) in ders[order].iter().enumerate() {
            matrix[(row, span - degree + j)] = d;
        }
    }
    matrix
}

/// Gram matrix `S_ij = ∫ B_i'' B_j'' du`, scaled to trace `n` (so the
/// smoothing weight hardly depends on the control-point count).
pub fn bending_energy_matrix(knots: &[f64], degree: usize) -> DMatrix<f64> {
    const QUAD_ORDER: usize = 8;
    let n = num_control_points(knots, degree);
    let mut s = DMatrix::zeros(n, n);
    if degree < 2 {
        return s;
    }
    let (nodes, weights) = gauss_legendre(QUAD_ORDER);
    let breaks = distinct(knots);
    for w in breaks.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b - a < 1e-14 {
            continue;
        }
        for (u, weight) in on_interval(&nodes, &weights, a, b) {
            let span = find_span(knots, degree, u);
            let d2 = &basis_derivatives(knots, degree, span, u, 2)[2];
            let first = span - degree;
            for (i, &bi) in d2.iter().enumerate() {
                for (j, &bj) in d2.iter().enumerate() {
                    s[(first + i, first + j)] += weight * bi * bj;
                }
            }
        }
    }
    let trace = s.trace();
    if trace > 1e-15 {
        s *= n as f64 / trace;
    }
    s
}

/// `R` with `RᵀR = A` for a symmetric positive semidefinite `A`: the
/// Cholesky factor, which changes smoothly with `A`, or an eigen-root when
/// `A` is singular.
pub fn matrix_root(a: &DMatrix<f64>) -> DMatrix<f64> {
    linalg::cholesky_upper(a).unwrap_or_else(|| linalg::eigen_root(a))
}

/// Monotone parameters in `[0, 1]` of an ordered polyline: `alpha` 0.5 is
/// centripetal, 1 chord length, 0 uniform. Coincident points fall back to
/// uniform spacing.
pub fn centripetal_parameters(points: &[Vec2], alpha: f64) -> Vec<f64> {
    if points.len() < 2 {
        return vec![0.0; points.len()];
    }
    let mut t = Vec::with_capacity(points.len());
    let mut total = 0.0;
    t.push(0.0);
    for w in points.windows(2) {
        total += (w[1] - w[0]).norm().powf(alpha);
        t.push(total);
    }
    if total <= 0.0 {
        return linspace(0.0, 1.0, points.len());
    }
    t.iter().map(|v| v / total).collect()
}

/// `n` evenly spaced values from `a` to `b` (NumPy `linspace`).
pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    match n {
        0 => vec![],
        1 => vec![a],
        _ => {
            let step = (b - a) / (n - 1) as f64;
            let mut v: Vec<f64> = (0..n).map(|i| a + i as f64 * step).collect();
            v[n - 1] = b;
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_uniform() {
        assert_eq!(open_uniform_knots(3, 5).unwrap(), vec![0., 0., 0., 0., 0.5, 1., 1., 1., 1.]);
        assert_eq!(
            open_uniform_knots(3, 3).unwrap_err().0,
            "Number of control points (3) must be greater than degree (3)."
        );
    }

    #[test]
    fn basis_partition_of_unity_and_derivatives() {
        let t = clamped_knots(4, &[0.2, 0.5, 0.5, 0.8]);
        let u = linspace(0.0, 1.0, 41);
        let n = basis_matrix(&u, &t, 4, 0);
        for i in 0..u.len() {
            assert!((n.row(i).sum() - 1.0).abs() < 1e-14);
        }
        let d = basis_matrix(&u, &t, 4, 1);
        let h = 1e-6;
        let up: Vec<f64> = u.iter().map(|x| (x + h).min(1.0)).collect();
        let dn: Vec<f64> = u.iter().map(|x| (x - h).max(0.0)).collect();
        let (np, nm) = (basis_matrix(&up, &t, 4, 0), basis_matrix(&dn, &t, 4, 0));
        for i in 1..u.len() - 1 {
            if [0.2, 0.5, 0.8].iter().any(|k| (u[i] - k).abs() < 1e-9) {
                continue;
            }
            for j in 0..n.ncols() {
                let fd = (np[(i, j)] - nm[(i, j)]) / (up[i] - dn[i]);
                assert!((fd - d[(i, j)]).abs() < 1e-5, "{i} {j}");
            }
        }
        assert!(basis_matrix(&u, &t, 4, 5).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn validation_messages() {
        assert_eq!(validate_knots(&[0., 0., 1., 0.5], 1).unwrap_err().0, "Knot vector must be nondecreasing.");
        assert_eq!(
            validate_knots(&[0., 0., 0.5, 0.5, 0.5, 1., 1.], 1).unwrap_err().0,
            "Interior knot multiplicity must not exceed the degree."
        );
        assert!(validate_knots(&open_uniform_knots(5, 25).unwrap(), 5).is_ok());
    }

    #[test]
    fn bending_energy_trace() {
        let t = open_uniform_knots(5, 12).unwrap();
        let s = bending_energy_matrix(&t, 5);
        assert!((s.trace() - 12.0).abs() < 1e-10);
        assert!((&s - s.transpose()).abs().max() < 1e-12);
    }
}
