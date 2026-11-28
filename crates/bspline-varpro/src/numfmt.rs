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
    let digits = tie_to_even(v, digits, exponent);
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
/// Python's choice between two shortest digit strings that are equally close
/// to `v`: the one ending in an even digit (Rust takes the upper one).
/// `1157348994281339.25` is `1157348994281339.2` in Python.
fn tie_to_even(v: f64, digits: String, exponent: i32) -> String {
    let n = digits.len();
    // Halfway between two n-digit numbers, v is half a unit of their last
    // digit from the one that reads back as it, so that half unit is at most
    // half the gap between doubles (2^-52 of v at most): n is 16 or 17.
    if n < 16 {
        return digits;
    }
    // Halfway, the digits after the nth are a 5 and zeros: looked for in a
    // few digits first, as nearly no number is.
    const LOOK: usize = 8;
    let (_, some, _) = decompose(&format!("{v:.*e}", n + LOOK - 1));
    let after = &some.as_bytes()[n..];
    if after[0] != b'5' || after[1..].iter().any(|&d| d != b'0') {
        return digits;
    }
    exact_tie_to_even(v, digits, exponent)
}

/// [`tie_to_even`] from the exact value of `v`, for any `v`.
fn exact_tie_to_even(v: f64, digits: String, exponent: i32) -> String {
    let n = digits.len();
    // The exact value of v, digit by digit (a double has at most 767).
    let (_, exact, exact_exponent) = decompose(&format!("{v:.800e}"));
    let tail = &exact.as_bytes()[n..];
    let halfway = exact_exponent == exponent && tail.first() == Some(&b'5') && tail[1..].iter().all(|&d| d == b'0');
    let last = digits.as_bytes()[n - 1] - b'0';
    if !halfway || last % 2 == 0 {
        return digits;
    }
    // The two candidates are the exact digits cut at n, and that plus one in
    // the last place; ours is odd, so take the other if it also reads back.
    let lower = &exact[..n];
    let other = if digits == lower {
        if last == 9 {
            return digits;
        }
        format!("{}{}", &digits[..n - 1], last + 1)
    } else {
        lower.to_string()
    };
    let reads_back = format!("{}.{}e{exponent}", &other[..1], &other[1..]).parse::<f64>().is_ok_and(|w| w == v.abs());
    if reads_back {
        other
    } else {
        digits
    }
}

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
        if point {
            format!("{digits}{zeros}.0")
        } else {
            format!("{digits}{zeros}")
        }
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
    fn halfway_digits_round_to_even_as_in_python() {
        // The quick ways out of the tie rule leave no number out: as from
        // the exact value, for doubles of every size and for ties.
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut ties = 0;
        for i in 0..200_000 {
            let bits = next();
            let v = match i % 4 {
                0 => f64::from_bits(bits),
                // Sixteen digits before the point and a quarter or a half after.
                1 => (1e15 + (bits % 8_000_000_000_000_000) as f64) + [0.25, 0.5, 0.75][(bits >> 60) as usize % 3],
                // Seventeen digits: every other integer is a double.
                2 => 1e16 + (bits % 80_000_000_000_000_000) as f64,
                _ => (bits % 1_000_000_000) as f64 / 1e6,
            };
            if !v.is_finite() {
                continue;
            }
            let (_, digits, exponent) = decompose(&format!("{v:e}"));
            let exact = exact_tie_to_even(v, digits.clone(), exponent);
            ties += usize::from(exact != digits);
            assert_eq!(tie_to_even(v, digits, exponent), exact, "{v:e}");
        }
        assert!(ties > 1000, "{ties} ties met");
        // Exactly 1157348994281339.25: .2 and .3 are equally close.
        assert_eq!(shortest(1157348994281339.0 + 0.25), "1157348994281339.2");
        assert_eq!(shortest(-(1157348994281339.0 + 0.25)), "-1157348994281339.2");
        // Not halfway: unchanged.
        assert_eq!(shortest(1157348994281339.5), "1157348994281339.5");
        assert_eq!(shortest(0.1), "0.1");
        assert_eq!(shortest(2.0 / 3.0), "0.6666666666666666");
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
