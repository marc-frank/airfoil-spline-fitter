//! Plot axes as the Python version drew them (pyqtgraph 0.13): where the
//! ticks go, which of them get labels, how the labels read and how strong
//! each level of the grid is.
//!
//! An axis has up to three tick levels: major, minor and extra. Every level
//! draws a grid line, fainter for lower levels and denser ticks; labels are
//! drawn level by level while they fill little enough of the axis.

/// One level of ticks: its spacing (`None` for the minor ticks of a log
/// axis) and values.
#[derive(Debug, Clone, PartialEq)]
pub struct Level {
    pub spacing: Option<f64>,
    pub values: Vec<f64>,
}

/// Major, minor and extra tick spacings for a range shown on `size` pixels.
pub fn tick_spacing(min: f64, max: f64, size: f64) -> Vec<f64> {
    let dif = (max - min).abs();
    if dif == 0.0 || !dif.is_finite() || size <= 0.0 {
        return vec![];
    }
    // Axes longer than 300 px show more than the least number of major ticks.
    let intervals = (2.25 * (size / 300.0).sqrt()).max(2.25);
    let major_max = dif / intervals;
    let mut p10 = 10f64.powf(major_max.log10().floor() - 1.0);
    let major_factor = if 100.0 * p10 <= major_max {
        p10 *= 10.0;
        10.0
    } else {
        *[50.0, 20.0, 10.0].iter().find(|&&f| f * p10 <= major_max).unwrap_or(&10.0)
    };
    let major = major_factor * p10;
    // No more than one minor tick per two pixels.
    let min_spacing = 2.0 * dif / size;
    let trials: &[f64] = if major_factor == 10.0 { &[5.0, 10.0] } else { &[10.0, 20.0, 50.0] };
    let minor = trials.iter().map(|f| f * p10).find(|&m| m >= min_spacing).unwrap_or(trials[trials.len() - 1] * p10);
    let mut levels = vec![major, minor];
    let extra_trials: &[f64] = match major_factor as i64 {
        10 => &[1.0, 2.0, 5.0, 10.0],
        20 => &[2.0, 5.0, 10.0, 20.0],
        _ => &[5.0, 10.0, 50.0],
    };
    let extra = extra_trials.iter().map(|f| f * p10).find(|&e| e >= min_spacing || e == minor).unwrap_or(minor);
    if extra < minor {
        levels.push(extra);
    }
    levels
}

/// Tick levels of a linear or log axis (log: the range is in decades).
pub fn tick_values(min: f64, max: f64, size: f64, log: bool) -> Vec<Level> {
    let (min, max) = if min <= max { (min, max) } else { (max, min) };
    let mut seen: Vec<f64> = Vec::new();
    let mut levels = Vec::new();
    for spacing in tick_spacing(min, max, size) {
        let start = (min / spacing).ceil() * spacing;
        let count = ((max - start) / spacing).floor() as i64 + 1;
        let values: Vec<f64> = (0..count.max(0))
            .map(|k| k as f64 * spacing + start)
            .filter(|v| !seen.iter().any(|s| (s - v).abs() <= spacing * 0.01))
            .collect();
        seen.extend(&values);
        levels.push(Level { spacing: Some(spacing), values });
    }
    if !log {
        return levels;
    }
    // A log axis keeps the levels of whole decades and, with fewer than
    // three, adds the 1..9 multiples of every decade in view.
    let mut kept: Vec<Level> = levels.into_iter().filter(|l| l.spacing.is_some_and(|s| s >= 1.0)).collect();
    if kept.len() < 3 {
        let mut minor = Vec::new();
        for decade in (min.floor() as i64)..(max.ceil() as i64) {
            for k in 1..10 {
                let v = decade as f64 + (k as f64).log10();
                if v > min && v < max {
                    minor.push(v);
                }
            }
        }
        kept.push(Level { spacing: None, values: minor });
    }
    kept
}

const SUPERSCRIPT: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];

/// `%g` of C with `digits` significant digits.
fn g(v: f64, digits: usize) -> String {
    spline_fitter::formatting::format_spec(v, &format!(".{digits}g"))
}

/// Labels of the ticks of one level; the display decimal point.
pub fn tick_strings(values: &[f64], spacing: Option<f64>, log: bool, decade_in_view: bool) -> Vec<String> {
    let point = spline_fitter::formatting::decimal_point();
    if log {
        return values
            .iter()
            .map(|&v| {
                if decade_in_view && (v - v.round()).abs() >= 1e-9 {
                    return String::new();
                }
                let e = g(10f64.powf(v), 1);
                let text = match e.split_once('e') {
                    Some((mantissa, power)) => {
                        let sign = if power.starts_with('-') { "⁻" } else { "" };
                        let digits: String =
                            power[1..].trim_start_matches('0').chars().map(|c| SUPERSCRIPT[c.to_digit(10).unwrap_or(0) as usize]).collect();
                        let lead = if mantissa == "1" { String::new() } else { format!("{mantissa}·") };
                        format!("{lead}10{sign}{digits}")
                    }
                    None => e,
                };
                text.replace('.', &point.to_string())
            })
            .collect();
    }
    let spacing = spacing.unwrap_or(1.0);
    let places = (-spacing.log10()).ceil().max(0.0) as usize;
    values
        .iter()
        .map(|&v| {
            let text = if v.abs() < 0.001 || v.abs() >= 10000.0 { g(v, 6) } else { format!("{v:.places$}") };
            text.replace('.', &point.to_string())
        })
        .collect()
}

/// Whether a log range holds a whole decade.
pub fn decade_in_view(min: f64, max: f64) -> bool {
    let (low, high) = if min <= max { (min, max) } else { (max, min) };
    high.floor() >= low.ceil()
}

/// Opacity (0..255) of the grid lines of level `index` with `count` ticks
/// on an axis of `length` pixels, at the theme's grid alpha.
pub fn grid_alpha(index: usize, count: usize, length: f64, grid_alpha: f64) -> f64 {
    let fade = (0.05 * length / (count as f64 + 1.0)).clamp(0.0, 1.0);
    255.0 / (index as f64 + 1.0) * grid_alpha * fade
}

/// Which levels get labels: each level after the first only while the
/// labels so far (and its own) fill little of the axis. `labels` are, per
/// level, each label's extent along the axis and whether it fits on the
/// axis (labels that do not are measured but not drawn, and the limits
/// count only drawn ones).
pub fn labelled_levels(labels: &[Vec<(f32, bool)>], length: f32) -> usize {
    const FILL_LIMITS: [(usize, f32); 4] = [(0, 0.8), (2, 0.6), (4, 0.4), (6, 0.2)];
    let mut total = 0.0;
    let mut drawn = 0;
    for (i, level) in labels.iter().enumerate().take(3) {
        total += level.iter().map(|l| l.0).sum::<f32>();
        if i > 0 {
            let ratio = total / length;
            if FILL_LIMITS.iter().any(|&(n, limit)| drawn >= n && ratio >= limit) {
                return i;
            }
        }
        drawn += level.iter().filter(|l| l.1).count();
    }
    labels.len().min(3)
}

/// The space an axis keeps for its tick labels: it grows to the widest
/// label and shrinks only when that is 10 px narrower.
pub fn text_space(current: f32, measured: f32) -> f32 {
    if measured > current || measured < current - 10.0 {
        measured
    } else {
        current
    }
}

/// A range with `pad` of its width added on each side; a zero range is
/// one unit wide (as a view with no earlier range).
pub fn padded(min: f64, max: f64, pad: f64) -> (f64, f64) {
    if min == max {
        return (min - 0.5, max + 0.5);
    }
    let d = (max - min) * pad;
    (min - d, max + d)
}

/// The padding of an automatic range on an axis of `pixels`.
pub fn auto_padding(pixels: f64) -> f64 {
    if pixels <= 0.0 {
        return 0.02;
    }
    (1.0 / pixels.sqrt()).clamp(0.02, 0.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spacings_follow_pyqtgraph() {
        // The airfoil plot's y axis: 0.2, 0.1 and 0.02 (measured in the
        // Python window's grid).
        let s = tick_spacing(-0.565, 0.565, 430.0);
        assert!((s[0] - 0.2).abs() < 1e-12 && (s[1] - 0.1).abs() < 1e-12 && (s[2] - 0.02).abs() < 1e-12, "{s:?}");
        // Iterations 0..33: 10, 5 and 1.
        let s = tick_spacing(-1.6, 34.6, 420.0);
        assert_eq!(s, vec![10.0, 5.0, 1.0]);
    }

    #[test]
    fn labels_read_like_python() {
        assert_eq!(tick_strings(&[0.0, 0.2, 1.0], Some(0.2), false, false), ["0", "0.2", "1.0"]);
        assert_eq!(tick_strings(&[-0.5], Some(0.1), false, false), ["-0.5"]);
        assert_eq!(tick_strings(&[5.0, 10.0], Some(5.0), false, false), ["5", "10"]);
        assert_eq!(tick_strings(&[-5.0, -6.0, 2f64.log10() - 6.0], None, true, true), ["10⁻⁵", "10⁻⁶", ""]);
        assert_eq!(tick_strings(&[2f64.log10() - 6.0], None, true, false), ["2·10⁻⁶"]);
    }

    #[test]
    fn grid_alpha_fades_by_level() {
        // 0.1 of the grid color, a half and a third of it, thinned when dense.
        assert!((grid_alpha(0, 5, 430.0, 0.1) - 25.5).abs() < 1e-9);
        assert!((grid_alpha(1, 6, 430.0, 0.1) - 12.75).abs() < 1e-9);
        assert!(grid_alpha(2, 45, 430.0, 0.1) < 5.0);
    }

    #[test]
    fn log_axes_add_the_multiples() {
        let levels = tick_values(-6.1, -4.6, 70.0, true);
        assert_eq!(levels.len(), 1);
        assert!(levels[0].values.iter().any(|v| (v + 5.0).abs() < 1e-12));
        assert!(decade_in_view(-6.1, -4.6) && !decade_in_view(-5.9, -5.1));
    }

    /// An empty plot's x axis (0 to 1): the labels at the ends do not fit,
    /// so the 0.1 steps are labelled too, as in the Python window.
    #[test]
    fn labels_that_do_not_fit_do_not_crowd() {
        let major: Vec<(f32, bool)> = (0..6).map(|i| (14.0, i > 0 && i < 5)).collect();
        let minor = vec![(14.0, true); 5];
        assert_eq!(labelled_levels(&[major.clone(), minor.clone(), vec![(20.0, true); 40]], 415.0), 2);
        let all_fit: Vec<(f32, bool)> = major.iter().map(|m| (m.0, true)).collect();
        assert_eq!(labelled_levels(&[all_fit, minor, vec![(20.0, true); 40]], 415.0), 1);
    }

    #[test]
    fn single_values_get_a_unit_range() {
        assert_eq!(padded(25.0, 25.0, 0.05), (24.5, 25.5));
    }
}
