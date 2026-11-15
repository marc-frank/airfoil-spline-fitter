//! Equality-constrained least squares through the KKT system.

use nalgebra::{DMatrix, Dyn, LU};

use crate::linalg::pinv;

/// The LU-factored KKT matrix `[[A + reg·I, Cᵀ], [C, 0]]` for repeated solves.
///
/// A (nearly) singular matrix is solved with the pseudo-inverse instead: the
/// minimum-norm solution, which leaves the residual unchanged because the
/// free directions do not reach it.
pub struct KktSolver {
    size: usize,
    num_controls: usize,
    method: Method,
}

enum Method {
    Lu(LU<f64, Dyn, Dyn>),
    Pinv(DMatrix<f64>),
}

impl KktSolver {
    pub fn new(a: &DMatrix<f64>, c: &DMatrix<f64>, regularization: f64) -> Self {
        let n = a.nrows();
        let size = n + c.nrows();
        let mut k = DMatrix::zeros(size, size);
        k.view_mut((0, 0), (n, n)).copy_from(a);
        for i in 0..n {
            k[(i, i)] += regularization;
        }
        k.view_mut((0, n), (n, c.nrows())).copy_from(&c.transpose());
        k.view_mut((n, 0), c.shape()).copy_from(c);
        let lu = k.clone().lu();
        let pivots: Vec<f64> = lu.u().diagonal().iter().map(|v| v.abs()).collect();
        let largest = pivots.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let smallest = pivots.iter().copied().fold(f64::INFINITY, f64::min);
        let (largest, smallest) = if pivots.is_empty() { (1.0, 1.0) } else { (largest, smallest) };
        let regular = pivots.iter().all(|p| p.is_finite()) && smallest > size as f64 * f64::EPSILON * largest;
        let method = if regular { Method::Lu(lu) } else { Method::Pinv(pinv(&k, 1e-12)) };
        KktSolver { size, num_controls: n, method }
    }

    /// Solves `K [x; μ] = [rhs; d]` (`d = None` means zeros) for every column
    /// of `rhs`; returns `(x, μ)`.
    pub fn solve_full(&self, rhs: &DMatrix<f64>, d: Option<&DMatrix<f64>>) -> (DMatrix<f64>, DMatrix<f64>) {
        let n = self.num_controls;
        let mut full = DMatrix::zeros(self.size, rhs.ncols());
        full.view_mut((0, 0), rhs.shape()).copy_from(rhs);
        if let Some(d) = d {
            full.view_mut((n, 0), d.shape()).copy_from(d);
        }
        let solution = match &self.method {
            Method::Lu(lu) => lu.solve(&full).unwrap_or_else(|| DMatrix::from_element(self.size, rhs.ncols(), f64::NAN)),
            Method::Pinv(p) => p * full,
        };
        (solution.rows(0, n).into_owned(), solution.rows(n, self.size - n).into_owned())
    }

    /// The control-point block of [`solve_full`](Self::solve_full).
    pub fn solve(&self, rhs: &DMatrix<f64>, d: Option<&DMatrix<f64>>) -> DMatrix<f64> {
        self.solve_full(rhs, d).0
    }
}
