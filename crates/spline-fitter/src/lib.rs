//! B-spline fitting of airfoil coordinates.
//!
//! Reading airfoil files of many flavours, splitting an airfoil into
//! sections, fitting the chain by variable projection or a direct
//! constrained least-squares solve, measuring the fit, and the spline, CAD,
//! project and log files the application writes.

pub mod analysis;
pub mod batch;
pub mod formatting;
pub mod geometry;
pub mod io;
pub mod models;
pub mod optimize;
pub mod project;
pub mod run_log;

pub use bspline_varpro::{BSpline, Error, Result, Vec2};

/// The application version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The application's window title and product name.
pub const APP_NAME: &str = "Airfoil Spline Fitter";

/// Return early with an [`Error`] built from a format string.
macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::Error(format!($($arg)*))) };
}
pub(crate) use bail;
