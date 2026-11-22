//! Display formatting of numbers, durations, counts and file names.
//!
//! Texts shown to the user use the display decimal point, which the GUI
//! takes from the system locale; files never do.

use std::sync::atomic::{AtomicBool, Ordering};

use bspline_varpro::numfmt;

static DECIMAL_COMMA: AtomicBool = AtomicBool::new(false);

/// Sets the display decimal point (`.` or `,`).
pub fn set_decimal_point(point: char) {
    DECIMAL_COMMA.store(point == ',', Ordering::Relaxed);
}

/// The display decimal point.
pub fn decimal_point() -> char {
    if DECIMAL_COMMA.load(Ordering::Relaxed) {
        ','
    } else {
        '.'
    }
}

/// A number in a format spec (`.2e`, `.6f`, `.4g`, `g`, ...) with the
/// display decimal point.
pub fn number(v: f64, spec: &str) -> String {
    let text = format_spec(v, spec);
    if decimal_point() == ',' {
        text.replace('.', ",")
    } else {
        text
    }
}

/// A number in a format spec: `.Ne`, `.Nf`, `.Ng` or `g`.
pub fn format_spec(v: f64, spec: &str) -> String {
    let (precision, kind) = spec.split_at(spec.len() - 1);
    let precision: Option<usize> = precision.strip_prefix('.').and_then(|p| p.parse().ok());
    match kind {
        "e" => numfmt::exponent(v, precision.unwrap_or(6)),
        "f" => numfmt::fixed(v, precision.unwrap_or(6)),
        "g" => numfmt::general(v, precision.unwrap_or(6)),
        _ => panic!("unsupported format spec {spec:?}"),
    }
}

/// An error value: exponent notation when small or large, else six decimals.
pub fn format_error(v: f64) -> String {
    if !v.is_finite() {
        return numfmt::shortest(v);
    }
    if (v != 0.0 && v.abs() < 1e-3) || v.abs() >= 1e4 {
        number(v, ".3e")
    } else {
        number(v, ".6f")
    }
}

/// A duration: `123 µs`, `456 ms`, `12.34 s`, `2m 3s`, `1h 2m 3s`.
pub fn format_duration(seconds: f64) -> String {
    if seconds < 1e-3 {
        format!("{:.0} µs", seconds * 1e6)
    } else if seconds < 1.0 {
        format!("{:.0} ms", seconds * 1e3)
    } else if seconds < 60.0 {
        number(seconds, ".2f") + " s"
    } else {
        let minutes = (seconds / 60.0).floor();
        let rest = seconds - 60.0 * minutes;
        if minutes < 60.0 {
            format!("{}m {}s", minutes as u64, rest as u64)
        } else {
            let (h, m) = (minutes as u64 / 60, minutes as u64 % 60);
            format!("{h}h {m}m {}s", rest as u64)
        }
    }
}

/// `1 point`, `2.500 points` (the thousands separator follows the decimal
/// point: `.` with a decimal comma, else `,`).
pub fn format_count(v: usize, singular: &str) -> String {
    format_count_with(v, singular, &format!("{singular}s"))
}

pub fn format_count_with(v: usize, singular: &str, plural: &str) -> String {
    let noun = if v == 1 { singular } else { plural };
    let separator = if decimal_point() == ',' { '.' } else { ',' };
    let digits = v.to_string();
    let mut grouped = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            grouped.push(separator);
        }
        grouped.push(c);
    }
    format!("{grouped} {noun}")
}

/// `12.3%`.
pub fn format_percentage(v: f64, precision: usize) -> String {
    number(100.0 * v, &format!(".{precision}f")) + "%"
}

/// Every character outside `A-Z a-z 0-9 - _ .` becomes `_`.
pub fn sanitize_filename(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.".contains(c) { c } else { '_' }).collect()
}

/// A name usable as a file stem: characters Windows forbids become `_`,
/// surrounding spaces and trailing dots go; empty gives `default`.
pub fn file_stem(s: &str, default: &str) -> String {
    let replaced: String = s.chars().map(|c| if "<>:\"/\\|?*".contains(c) || (c as u32) < 32 { '_' } else { c }).collect();
    let stem = replaced.trim().trim_end_matches(['.', ' ']);
    if stem.is_empty() {
        default.to_string()
    } else {
        stem.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_examples() {
        assert_eq!(format_error(0.0000123), "1.230e-05");
        assert_eq!(format_error(0.00456), "0.004560");
        assert_eq!(format_error(12345.6), "1.235e+04");
        assert_eq!(format_duration(0.000123), "123 µs");
        assert_eq!(format_duration(0.456), "456 ms");
        assert_eq!(format_duration(12.34), "12.34 s");
        assert_eq!(format_duration(123.4), "2m 3s");
        assert_eq!(format_duration(3723.0), "1h 2m 3s");
        assert_eq!(sanitize_filename("a b/c"), "a_b_c");
        assert_eq!(file_stem("NACA 2412 t/c 12%", "airfoil"), "NACA 2412 t_c 12%");
        assert_eq!(file_stem("", "airfoil"), "airfoil");
        assert_eq!(format_count(1, "point"), "1 point");
        assert_eq!(format_count(12345, "point"), "12,345 points");
        assert_eq!(format_percentage(0.12004, 1), "12.0%");
    }
}
