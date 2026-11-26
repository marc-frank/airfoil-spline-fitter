//! CAD-grade B-spline curve fitting by Levenberg-Marquardt variable projection.
//!
//! A chain of clamped planar B-spline sections is fitted to ordered points.
//! The data parameters are free unknowns (variable projection eliminates the
//! control points), junctions between sections carry geometric (G1-G3) or
//! parametric (C0-C3) continuity, and [`fit_to_tolerance`] searches for the
//! fewest control points that keep every point within a maximum orthogonal
//! deviation. [`corners`] finds the corners of the data, where sections join
//! with G0. Diagnostics, airfoil helpers and STEP/IGES/DXF export complete
//! the toolbox.

pub mod airfoil;
pub mod basis;
pub mod chain;
pub mod constraints;
pub mod corners;
pub mod curve;
pub mod diagnostics;
pub mod export;
mod fitting;
mod kkt;
mod linalg;
mod lm;
pub mod numfmt;
mod problem;
pub mod quadrature;
pub mod reparam;
mod tolerance;

pub use basis::{centripetal_parameters, clamped_knots, interior_knots, open_uniform_knots, validate_knots};
pub use chain::Parameters;
pub use constraints::{minimum_control_points, ChainConstraints, End, Junction};
pub use curve::BSpline;
pub use fitting::{fit, ChainOptions, Continuity, FitOptions, FitResult, KnotPlacement, LM_TOLERANCE, MIN_KNOT_GAP};
pub use kkt::KktSolver;
pub use tolerance::{fit_to_tolerance, solve_free_knots, ToleranceOptions};

/// Planar point or vector.
pub type Vec2 = nalgebra::Vector2<f64>;

/// The library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Message of a result whose fit an observer stopped.
pub const STOPPED: &str = "Stopped by callback.";

/// Invalid input or a failed solve; the message is meant for the user.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct Error(pub String);

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Error(message.into())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Return early with an [`Error`] built from a format string.
macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::Error(format!($($arg)*))) };
}
pub(crate) use bail;

/// A snapshot of a running fit.
#[derive(Debug, Clone, Copy)]
pub struct Progress<'a> {
    /// Control points of every section.
    pub control_points: &'a [Vec<Vec2>],
    /// Root mean square distance of the data to their curve points.
    pub rms: f64,
    /// Knot vector of every section (they change while knots move).
    pub knots: &'a [Vec<f64>],
}

/// Watches a fit; returning `true` from a method stops it, and the result
/// then holds the best state so far with `stopped` set.
pub trait Observer {
    /// After every residual evaluation.
    fn step(&mut self, _progress: &Progress) -> bool {
        false
    }

    /// After every refit of a tolerance search, with its maximum deviation.
    fn refit(&mut self, _result: &FitResult, _max_deviation: f64) -> bool {
        false
    }
}

/// An observer that watches nothing.
pub struct Silent;

impl Observer for Silent {}

impl<F: FnMut(&Progress) -> bool> Observer for F {
    fn step(&mut self, progress: &Progress) -> bool {
        self(progress)
    }
}
