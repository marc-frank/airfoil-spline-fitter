//! CSV logs of a fit's error history.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use chrono::{DateTime, Local};

use crate::formatting::sanitize_filename;

/// The log folder: `SPLINE_FITTER_LOG_DIR` if set, else `optimizer_logs`
/// in the checkout when run from one, else `~/.spline_fitter/optimizer_logs`.
pub fn default_log_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SPLINE_FITTER_LOG_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if checkout.join("Cargo.toml").is_file() {
        return checkout.join("optimizer_logs");
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from).unwrap_or_default();
    home.join(".spline_fitter").join("optimizer_logs")
}

#[derive(Debug, Clone, Copy)]
struct Record {
    iteration: usize,
    evaluation: usize,
    error: f64,
    elapsed: f64,
}

/// Records a fit's error per iteration and saves it as CSV.
#[derive(Debug, Clone)]
pub struct ErrorLog {
    run_name: String,
    start: DateTime<Local>,
    dir: PathBuf,
    records: Vec<Record>,
}

impl ErrorLog {
    pub fn new(run_name: &str) -> Self {
        ErrorLog { run_name: run_name.to_string(), start: Local::now(), dir: default_log_dir(), records: vec![] }
    }

    pub fn in_dir(run_name: &str, dir: PathBuf) -> Self {
        ErrorLog { dir, ..Self::new(run_name) }
    }

    pub fn log(&mut self, iteration: usize, evaluation: usize, error: f64, elapsed: f64) {
        self.records.push(Record { iteration, evaluation, error, elapsed });
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// `(iteration, elapsed seconds, error)` of every record.
    pub fn history(&self) -> impl Iterator<Item = (usize, f64, f64)> + '_ {
        self.records.iter().map(|r| (r.iteration, r.elapsed, r.error))
    }

    pub fn errors(&self) -> impl Iterator<Item = f64> + '_ {
        self.records.iter().map(|r| r.error)
    }

    /// Writes `<start>_<run name>_<optimizer>.csv` (with `_2`, `_3`, ...
    /// when taken); `None` without records.
    pub fn save(&self, optimizer: &str, info: &str) -> std::io::Result<Option<PathBuf>> {
        let Some(last) = self.records.last() else { return Ok(None) };
        std::fs::create_dir_all(&self.dir)?;
        let stem = format!("{}_{}_{optimizer}", self.start.format("%Y%m%d_%H%M%S"), sanitize_filename(&self.run_name));
        let mut text = format!(
            "# Optimizer Error Log\n# Run: {}\n# Optimizer: {optimizer}\n# Start Time: {}\n",
            self.run_name,
            self.start.format("%Y-%m-%dT%H:%M:%S%.6f")
        );
        if !info.is_empty() {
            text += &format!("# Info: {info}\n");
        }
        let min = self.errors().fold(f64::INFINITY, f64::min);
        text += &format!(
            "# Total Iterations: {}\n# Total Evaluations: {}\n# Final Error: {}\n# Min Error: {}\n# Total Time: {:.3}s\n#\n",
            self.records.len(),
            last.evaluation,
            bspline_varpro::numfmt::exponent(last.error, 12),
            bspline_varpro::numfmt::exponent(min, 12),
            last.elapsed
        );
        text += "iteration,evaluation,error,elapsed_s\n";
        for r in &self.records {
            text += &format!("{},{},{},{:.3}\n", r.iteration, r.evaluation, bspline_varpro::numfmt::exponent(r.error, 12), r.elapsed);
        }
        for k in 1.. {
            let name = if k == 1 { format!("{stem}.csv") } else { format!("{stem}_{k}.csv") };
            let path = self.dir.join(name);
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    file.write_all(text.as_bytes())?;
                    return Ok(Some(path));
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_csv() {
        let dir = tempfile::tempdir().unwrap();
        let mut log = ErrorLog::in_dir("NACA 2412", dir.path().to_path_buf());
        assert_eq!(log.save("gn-varpro", "").unwrap(), None);
        log.log(1, 3, 0.5, 0.01);
        log.log(2, 5, 0.25, 0.02);
        let path = log.save("gn-varpro", "continuity=G3, degree=5").unwrap().unwrap();
        assert!(path.file_name().unwrap().to_string_lossy().ends_with("_NACA_2412_gn-varpro.csv"));
        let second = log.save("gn-varpro", "").unwrap().unwrap();
        assert!(second.file_name().unwrap().to_string_lossy().ends_with("_gn-varpro_2.csv"));
        let text = std::fs::read_to_string(path).unwrap();
        assert!(text.contains("# Final Error: 2.500000000000e-01\n# Min Error: 2.500000000000e-01\n"));
        assert!(text.ends_with("iteration,evaluation,error,elapsed_s\n1,3,5.000000000000e-01,0.010\n2,5,2.500000000000e-01,0.020\n"));
    }
}
