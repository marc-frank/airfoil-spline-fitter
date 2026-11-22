//! Dense linear algebra helpers on top of nalgebra.

use nalgebra::{DMatrix, SymmetricEigen};

use crate::Vec2;

/// Upper Cholesky factor `R` (`RᵀR = A`), or `None` when `A` is not
/// positive definite.
pub fn cholesky_upper(a: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let cholesky = a.clone().cholesky()?;
    let r = cholesky.l().transpose();
    r.iter().all(|v| v.is_finite()).then_some(r)
}

/// `diag(sqrt(max(λ, 0))) Vᵀ` from `A = V diag(λ) Vᵀ`.
pub fn eigen_root(a: &DMatrix<f64>) -> DMatrix<f64> {
    let eigen = SymmetricEigen::new(a.clone());
    let mut root = eigen.eigenvectors.transpose();
    for (mut row, &lambda) in root.row_iter_mut().zip(eigen.eigenvalues.iter()) {
        row *= lambda.max(0.0).sqrt();
    }
    root
}

/// Moore-Penrose pseudo-inverse with singular values below
/// `rcond · σ_max` treated as zero.
pub fn pinv(a: &DMatrix<f64>, rcond: f64) -> DMatrix<f64> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return DMatrix::zeros(n, m);
    }
    let svd = a.clone().svd(true, true);
    let (u, v_t) = (svd.u.unwrap(), svd.v_t.unwrap());
    let cutoff = rcond * svd.singular_values.max();
    let mut result = DMatrix::zeros(n, m);
    for (k, &sigma) in svd.singular_values.iter().enumerate() {
        if sigma > cutoff {
            result += (v_t.row(k).transpose() / sigma) * u.column(k).transpose();
        }
    }
    result
}

/// Minimum-norm least-squares solution of `A X = B` (LAPACK `gelsd` with
/// the default cutoff `eps · max(m, n) · σ_max`).
pub fn lstsq(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let rcond = f64::EPSILON * a.nrows().max(a.ncols()) as f64;
    pinv(a, rcond) * b
}

/// Rows of points as an `(n, 2)` matrix.
pub fn points_to_matrix(points: &[Vec2]) -> DMatrix<f64> {
    DMatrix::from_fn(points.len(), 2, |i, k| points[i][k])
}

/// Rows of an `(n, 2)` matrix as points.
pub fn matrix_to_points(m: &DMatrix<f64>) -> Vec<Vec2> {
    m.row_iter().map(|r| Vec2::new(r[0], r[1])).collect()
}

/// Block-diagonal matrix of the given square blocks.
pub fn block_diag(blocks: &[DMatrix<f64>]) -> DMatrix<f64> {
    let size: usize = blocks.iter().map(|b| b.nrows()).sum();
    let mut result = DMatrix::zeros(size, size);
    let mut offset = 0;
    for block in blocks {
        result.view_mut((offset, offset), block.shape()).copy_from(block);
        offset += block.nrows();
    }
    result
}
