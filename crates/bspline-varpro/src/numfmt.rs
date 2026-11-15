//! Number formatting with the conventions of the file formats and the
//! application's texts: shortest round-trip text, `e` notation with a signed
//! two-digit exponent, and the "general" format that picks fixed or
//! exponent notation by magnitude.

/// Shortest text that reads back as the same double, in fixed notation for
/// exponents -4..15 and exponent notation otherwise (`1e-07`, `0.5`, `3.0`,
/// `1e+16`).
pub fn shortest(v: f64) -> String {
    if !v.is_finite() {
        return non_finite(v);
    }
    let (negative, digits, exponent) = decompose(&format!("{v:e}"));
    let sign = if negative { "-" } else { "" };
    if !(-4..16).contains(&exponent) {
        let mantissa = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        return format!("{sign}{mantissa}e{}", signed_exponent(exponent));
    }
    format!("{sign}{}", place_point(&digits, exponent, true))
}

/// `{v:.{precision}e}`: `1.230e-05`.
pub fn exponent(v: f64, precision: usize) -> String {
    if !v.is_finite() {
        return non_finite(v);
    }
    let text = format!("{v:.precision$e}");
    let (mantissa, exp) = text.split_once('e').unwrap();
    format!("{mantissa}e{}", signed_exponent(exp.parse().unwrap()))
}

/// `{v:.{precision}E}`: `1.2300000000000000E-05`.
pub fn exponent_upper(v: f64, precision: usize) -> String {
    exponent(v, precision).to_uppercase()
}

/// `{v:.{precision}f}`.
pub fn fixed(v: f64, precision: usize) -> String {
    if !v.is_finite() {
        return non_finite(v);
    }
    format!("{v:.precision$}")
}

/// `{v:.{precision}g}`: `precision` significant digits, trailing zeros
/// removed, exponent notation below 1e-4 or from `10^precision`.
pub fn general(v: f64, precision: usize) -> String {
    if !v.is_finite() {
        return non_finite(v);
    }
    let precision = precision.max(1);
    let rounded = format!("{v:.prec$e}", prec = precision - 1);
    let (_, exp) = rounded.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if exp < -4 || exp >= precision as i32 {
        let (mantissa, _) = rounded.split_once('e').unwrap();
        format!("{}e{}", strip_zeros(mantissa), signed_exponent(exp))
    } else {
        let decimals = (precision as i32 - 1 - exp).max(0) as usize;
        strip_zeros(&format!("{v:.decimals$}")).to_string()
    }
}

/// `{v:g}` (six significant digits).
pub fn general6(v: f64) -> String {
    general(v, 6)
}

/// A string literal as Python's `repr` writes it: single quotes unless the
/// text holds a single quote and no double quote, with escapes.
pub fn quoted(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ if c == quote => {
                out.push('\\');
                out.push(c);
            }
            _ if (c as u32) < 32 || c as u32 == 127 => out.push_str(&format!("\\x{:02x}", c as u32)),
            _ => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn non_finite(v: f64) -> String {
    if v.is_nan() {
        "nan".into()
    } else if v > 0.0 {
        "inf".into()
    } else {
        "-inf".into()
    }
}

fn signed_exponent(exp: i32) -> String {
    format!("{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

fn strip_zeros(text: &str) -> &str {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        text
    }
}

/// Sign, significant digits and decimal exponent of Rust's `{:e}` text.
fn decompose(text: &str) -> (bool, String, i32) {
    let (mantissa, exp) = text.split_once('e').unwrap();
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    (negative, digits, exp.parse().unwrap())
}

/// Digits `d.ddd × 10^exponent` in fixed notation (`.0` appended to integers
/// when `point` is set).
fn place_point(digits: &str, exponent: i32, point: bool) -> String {
    let n = digits.len() as i32;
    if exponent < 0 {
        format!("0.{}{digits}", "0".repeat((-exponent - 1) as usize))
    } else if exponent + 1 >= n {
        let zeros = "0".repeat((exponent + 1 - n) as usize);
        if point { format!("{digits}{zeros}.0") } else { format!("{digits}{zeros}") }
    } else {
        let split = (exponent + 1) as usize;
        format!("{}.{}", &digits[..split], &digits[split..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortest_matches_python_repr() {
        for (v, text) in [
            (1e-7, "1e-07"),
            (3.0, "3.0"),
            (0.5, "0.5"),
            (2.5e16, "2.5e+16"),
            (-1e16, "-1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00012345, "0.00012345"),
            (123.25, "123.25"),
            (0.1, "0.1"),
            (0.0, "0.0"),
            (1.0 / 3.0, "0.3333333333333333"),
        ] {
            assert_eq!(shortest(v), text);
        }
    }

    #[test]
    fn exponent_and_general() {
        assert_eq!(exponent(0.0000123, 3), "1.230e-05");
        assert_eq!(exponent(12345.6, 3), "1.235e+04");
        assert_eq!(exponent_upper(1e-7, 2), "1.00E-07");
        assert_eq!(general6(1e-5), "1e-05");
        assert_eq!(general6(0.0001), "0.0001");
        assert_eq!(general(0.0123, 3), "0.0123");
        assert_eq!(general(0.5, 3), "0.5");
        assert_eq!(general(1234567.0, 6), "1.23457e+06");
        assert_eq!(general(100.0, 3), "100");
        assert_eq!(general(0.015698, 6), "0.015698");
    }
}
