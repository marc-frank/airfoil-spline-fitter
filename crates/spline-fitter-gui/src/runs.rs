//! The columns of the run explorer, its filter syntax and sort order.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use spline_fitter::bspline_varpro::KnotPlacement;
use spline_fitter::formatting::{decimal_point, format_duration, number};
use spline_fitter::models::OptimizerType;
use spline_fitter::project::OptimizationRun;

/// A cell value.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Empty,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

impl Cell {
    /// The value as a number (bools are not numbers).
    pub fn number(&self) -> Option<f64> {
        match self {
            Cell::Int(i) => Some(*i as f64),
            Cell::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn is_number(&self) -> bool {
        self.number().is_some()
    }
}

/// How a column formats its numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Plain,
    Spec(&'static str),
    Duration,
}

/// A column of the run table.
pub struct Column {
    pub name: &'static str,
    pub tooltip: &'static str,
    /// Plotted on a log axis.
    pub log: bool,
    /// A fit metric (offered as the comparison plot's y axis).
    pub metric: bool,
    format: Format,
    value: fn(&OptimizationRun) -> Cell,
    /// The value only applies when this holds.
    applies: fn(&OptimizationRun) -> bool,
}

fn varpro(r: &OptimizationRun) -> bool {
    r.optimization_config.optimizer == OptimizerType::VarPro
}

fn always(_: &OptimizationRun) -> bool {
    true
}

fn junctions(r: &OptimizationRun) -> bool {
    r.spline_config.num_sections > 1
}

fn opt_float(v: Option<f64>) -> Cell {
    v.map_or(Cell::Empty, Cell::Float)
}

fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map(|c| c.to_uppercase().chain(chars.flat_map(|c| c.to_lowercase())).collect()).unwrap_or_default()
}

pub const COLUMNS: [Column; 25] = [
    Column {
        name: "Run",
        tooltip: "Order in which the runs were done; a number stays when other runs are deleted",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Int(r.number as i64),
        applies: always,
    },
    Column {
        name: "Optimizer",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Text(r.optimization_config.optimizer.display_name().into()),
        applies: always,
    },
    Column {
        name: "Goal",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Text(if r.optimization_config.tolerance.is_some() { "Tolerance" } else { "Fixed count" }.into()),
        applies: varpro,
    },
    Column {
        name: "Tolerance",
        tooltip: "Max deviation the tolerance goal had to meet",
        log: true,
        metric: false,
        format: Format::Spec(".3g"),
        value: |r| opt_float(r.optimization_config.tolerance),
        applies: |r| varpro(r) && r.optimization_config.tolerance.is_some(),
    },
    Column {
        name: "Knots",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| {
            Cell::Text(
                match r.optimization_config.knot_placement {
                    KnotPlacement::Uniform => "Uniform",
                    KnotPlacement::Free => "Free",
                }
                .into(),
            )
        },
        applies: varpro,
    },
    Column {
        name: "Parameterization",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Text(capitalized(&r.optimization_config.parameterization.value().replace('_', " "))),
        applies: |r| r.optimization_config.optimizer == OptimizerType::DirectLsq,
    },
    Column {
        name: "Grow points",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Bool(r.optimization_config.grow_control_points),
        applies: varpro,
    },
    Column {
        name: "Degree",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Int(r.spline_config.degree as i64),
        applies: always,
    },
    Column {
        name: "Control points",
        tooltip: "Control points per section as set (the start count of a tolerance goal)",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Int(r.spline_config.num_control_points as i64),
        applies: always,
    },
    Column {
        name: "Sections",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Int(r.spline_config.num_sections as i64),
        applies: always,
    },
    Column {
        name: "Continuity",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Text(r.spline_config.continuity.name().into()),
        applies: junctions,
    },
    Column {
        name: "Type",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Text(if r.spline_config.geometric_continuity { "Geometric" } else { "Parametric" }.into()),
        applies: |r| varpro(r) && junctions(r),
    },
    Column {
        name: "Per junction",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Text(r.spline_config.junction_continuity.join(", ")),
        applies: varpro,
    },
    Column {
        name: "Subsample",
        tooltip: "Points per section of the subsample; blank when every point is used",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| r.optimization_config.max_points_per_section.map_or(Cell::Empty, |m| Cell::Int(m as i64)),
        applies: varpro,
    },
    Column {
        name: "Smoothing",
        tooltip: "",
        log: true,
        metric: false,
        format: Format::Spec(".3g"),
        value: |r| Cell::Float(r.optimization_config.smoothing_weight),
        applies: |r| varpro(r) && r.optimization_config.tolerance.is_none(),
    },
    Column {
        name: "Optimize junctions",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Bool(r.optimization_config.optimize_le_position),
        applies: junctions,
    },
    Column {
        name: "Error",
        tooltip: "RMS distance of the points to the curve, the same measure for every run",
        log: true,
        metric: true,
        format: Format::Spec(".3e"),
        value: |r| Cell::Float(r.result.comparison_error()),
        applies: always,
    },
    Column {
        name: "Max deviation",
        tooltip: "",
        log: true,
        metric: true,
        format: Format::Spec(".3e"),
        value: |r| opt_float(r.result.report.as_ref().map(|x| x.max_deviation)),
        applies: always,
    },
    Column {
        name: "RMS deviation",
        tooltip: "",
        log: true,
        metric: true,
        format: Format::Spec(".3e"),
        value: |r| opt_float(r.result.report.as_ref().map(|x| x.rms_deviation)),
        applies: always,
    },
    Column {
        name: "Fitted control points",
        tooltip: "Distinct control points of the fitted chain, junction points counted once",
        log: false,
        metric: true,
        format: Format::Plain,
        value: |r| Cell::Int(r.result.num_control_points() as i64),
        applies: always,
    },
    Column {
        name: "Iterations",
        tooltip: "",
        log: false,
        metric: true,
        format: Format::Plain,
        value: |r| Cell::Int(r.result.num_iterations as i64),
        applies: always,
    },
    Column {
        name: "Evaluations",
        tooltip: "",
        log: false,
        metric: true,
        format: Format::Plain,
        value: |r| Cell::Int(r.result.num_evaluations as i64),
        applies: always,
    },
    Column {
        name: "Time",
        tooltip: "",
        log: true,
        metric: true,
        format: Format::Duration,
        value: |r| Cell::Float(r.result.elapsed_seconds),
        applies: always,
    },
    Column {
        name: "Converged",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Bool(r.result.converged),
        applies: always,
    },
    Column {
        name: "Date",
        tooltip: "",
        log: false,
        metric: false,
        format: Format::Plain,
        value: |r| Cell::Text(r.timestamp.chars().take(16).collect::<String>().replace('T', " ")),
        applies: always,
    },
];

pub const ERROR_COLUMN: usize = 16;
pub const CONTROL_POINTS_COLUMN: usize = 8;

impl Column {
    pub fn cell(&self, run: &OptimizationRun) -> Cell {
        if (self.applies)(run) {
            (self.value)(run)
        } else {
            Cell::Empty
        }
    }

    pub fn text(&self, cell: &Cell) -> String {
        match (cell, self.format) {
            (Cell::Empty, _) => String::new(),
            (Cell::Bool(b), _) => if *b { "yes" } else { "no" }.into(),
            (Cell::Float(v), Format::Duration) => format_duration(*v),
            (Cell::Float(v), Format::Spec(spec)) => number(*v, spec),
            (Cell::Int(i), _) => i.to_string(),
            (Cell::Float(v), Format::Plain) => spline_fitter::bspline_varpro::numfmt::shortest(*v),
            (Cell::Text(t), _) => t.clone(),
        }
    }
}

/// Every cell of a run with its text.
pub fn row(run: &OptimizationRun) -> Vec<(Cell, String)> {
    COLUMNS
        .iter()
        .map(|c| {
            let cell = c.cell(run);
            let text = c.text(&cell);
            (cell, text)
        })
        .collect()
}

/// Sort order of cells: empty cells last (in both directions, so the caller
/// sorts empty cells apart), numbers by value, others by case-folded text.
pub fn compare(a: &(Cell, String), b: &(Cell, String)) -> Ordering {
    match (a.0.number(), b.0.number()) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        _ => a.1.to_lowercase().cmp(&b.1.to_lowercase()),
    }
}

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Le,
    Ge,
    Ne,
    Lt,
    Gt,
    Eq,
}

/// One word of a filter.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// A part of a cell text (case-folded).
    Substring(String),
    /// A whole cell text, equal (`=`) or not (`!=`).
    Whole { equal: bool, text: String },
    /// A comparison with a number.
    Numeric { op: Op, value: f64 },
}

const OPERATORS: [(&str, Op); 6] = [("<=", Op::Le), (">=", Op::Ge), ("!=", Op::Ne), ("<", Op::Lt), (">", Op::Gt), ("=", Op::Eq)];

/// Parses a filter text into terms.
pub fn parse_filter(text: &str) -> Vec<Term> {
    // Whitespace after an operator belongs to its operand.
    let mut joined = String::new();
    let mut rest = text.trim();
    while !rest.is_empty() {
        if let Some((token, _)) = OPERATORS.iter().find(|(t, _)| rest.starts_with(t)) {
            joined.push_str(token);
            rest = rest[token.len()..].trim_start();
        } else {
            let c = rest.chars().next().unwrap();
            joined.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    joined
        .split_whitespace()
        .map(|word| {
            for (token, op) in OPERATORS {
                if let Some(operand) = word.strip_prefix(token).filter(|o| !o.is_empty()) {
                    let parsed = operand.replace(decimal_point(), ".").parse::<f64>();
                    if let Ok(value) = parsed {
                        return Term::Numeric { op, value };
                    }
                    if matches!(op, Op::Eq | Op::Ne) {
                        return Term::Whole { equal: op == Op::Eq, text: operand.to_lowercase() };
                    }
                    break;
                }
            }
            Term::Substring(word.to_lowercase())
        })
        .collect()
}

/// Whether a term matches a cell.
pub fn term_matches(term: &Term, cell: &Cell, text: &str) -> bool {
    match term {
        Term::Substring(s) => text.to_lowercase().contains(s.as_str()),
        Term::Whole { equal, text: t } => (text.to_lowercase() == *t) == *equal,
        Term::Numeric { op, value } => cell.number().is_some_and(|v| match op {
            Op::Le => v <= *value,
            Op::Ge => v >= *value,
            Op::Ne => v != *value,
            Op::Lt => v < *value,
            Op::Gt => v > *value,
            Op::Eq => v == *value,
        }),
    }
}

/// The explorer's filters: global terms and excluded cell texts per column.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Filters {
    pub text: String,
    pub excluded: BTreeMap<usize, BTreeSet<String>>,
    pub invert: bool,
}

impl Filters {
    pub fn is_set(&self) -> bool {
        !parse_filter(&self.text).is_empty() || !self.excluded.is_empty()
    }

    /// Whether a row matches the global terms and every column filter,
    /// leaving out the filter of column `skip`.
    pub fn matches_except(&self, cells: &[(Cell, String)], skip: Option<usize>) -> bool {
        let terms = parse_filter(&self.text);
        let global = terms.iter().all(|term| match term {
            Term::Numeric { .. } => term_matches(term, &cells[ERROR_COLUMN].0, &cells[ERROR_COLUMN].1),
            _ => cells.iter().any(|(c, t)| term_matches(term, c, t)),
        });
        global && self.excluded.iter().filter(|(c, _)| Some(**c) != skip).all(|(c, ex)| !ex.contains(&cells[*c].1))
    }

    /// Whether a row is shown.
    pub fn shows(&self, cells: &[(Cell, String)]) -> bool {
        if !self.is_set() {
            return true;
        }
        self.matches_except(cells, None) != self.invert
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_filters() {
        assert_eq!(
            parse_filter(" >= 1e-5 varpro =free !=x"),
            vec![
                Term::Numeric { op: Op::Ge, value: 1e-5 },
                Term::Substring("varpro".into()),
                Term::Whole { equal: true, text: "free".into() },
                Term::Whole { equal: false, text: "x".into() },
            ]
        );
        assert_eq!(parse_filter("<inf"), vec![Term::Numeric { op: Op::Lt, value: f64::INFINITY }]);
        assert!(term_matches(&Term::Substring("pro".into()), &Cell::Empty, "VarPro LM"));
        assert!(!term_matches(&Term::Numeric { op: Op::Gt, value: 1.0 }, &Cell::Bool(true), "yes"));
    }

    #[test]
    fn capitalizes_like_python() {
        assert_eq!(capitalized("arc length"), "Arc length");
        assert_eq!(capitalized("centripetal"), "Centripetal");
    }
}
