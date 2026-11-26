//! Airfoil coordinate files and spline files.
//!
//! Airfoil files come in Selig order (a name line, then the points from the
//! upper trailing edge over the leading edge to the lower trailing edge) or
//! Lednicer order (point counts, then both surfaces from the leading edge),
//! with many number formats. Coordinates are normalized without turning
//! them: data already in chords are kept as written, others are moved and
//! scaled so the leading edge (smallest x) is at 0 and the trailing-edge
//! midpoint at (1, 0).

use std::collections::BTreeMap;
use std::path::Path;

use bspline_varpro::numfmt::{fixed, shortest};
use bspline_varpro::open_uniform_knots;

use crate::{bail, Error, Result, Vec2};

/// The text of a data file: UTF-8 (with or without byte order mark),
/// UTF-16 (with a byte order mark, or told by its zero bytes), or else
/// Windows-1252, so no character of a name is lost.
pub fn read_text(path: &Path) -> Result<String> {
    let data = std::fs::read(path).map_err(|e| Error::new(e.to_string()))?;
    Ok(decode(&data))
}

/// Decodes file bytes by the rules of [`read_text`].
pub fn decode(data: &[u8]) -> String {
    use encoding_rs::{UTF_16BE, UTF_16LE, WINDOWS_1252};
    if data.starts_with(&[0xFF, 0xFE]) {
        return UTF_16LE.decode_with_bom_removal(&data[2..]).0.into_owned();
    }
    if data.starts_with(&[0xFE, 0xFF]) {
        return UTF_16BE.decode_with_bom_removal(&data[2..]).0.into_owned();
    }
    let head = &data[..data.len().min(200)];
    if head.contains(&0) {
        let even = head.iter().step_by(2).filter(|&&b| b == 0).count();
        let odd = head.iter().skip(1).step_by(2).filter(|&&b| b == 0).count();
        let encoding = if even > odd { UTF_16BE } else { UTF_16LE };
        return encoding.decode_without_bom_handling(data).0.into_owned();
    }
    let body = data.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(data);
    match std::str::from_utf8(body) {
        Ok(text) => text.to_string(),
        Err(_) => WINDOWS_1252.decode_without_bom_handling(data).0.into_owned(),
    }
}

/// Lines as Python's `str.splitlines` splits them.
pub fn split_lines(text: &str) -> Vec<&str> {
    let breaks = |c: char| matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}');
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if breaks(c) {
            lines.push(&text[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && chars.peek().is_some_and(|&(_, n)| n == '\n') {
                chars.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// A number as written: digits with `_` only between digits are accepted
/// (Python's float syntax); only finite values count.
fn parse_number(text: &str) -> Option<f64> {
    let bytes = text.as_bytes();
    let underscores_ok = bytes
        .iter()
        .enumerate()
        .all(|(i, &b)| b != b'_' || (i > 0 && i + 1 < bytes.len() && bytes[i - 1].is_ascii_digit() && bytes[i + 1].is_ascii_digit()));
    if !underscores_ok {
        return None;
    }
    let cleaned: String = text.chars().filter(|&c| c != '_').collect();
    cleaned.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

/// The fields of a coordinate line as number texts.
///
/// Fields are separated by whitespace, commas or semicolons, or by nothing
/// before a sign (fixed-width columns, `0.95000-0.01200`). A line without a
/// dot whose fields hold commas uses the decimal comma (`0,5 0,01`).
/// Fortran `D` exponents become `e`; parentheses and quotes are dropped.
pub fn number_fields(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.replace(';', " ").chars().collect();
    let numeric = |c: char| c.is_ascii_digit() || c == '.';
    let mut glued = String::with_capacity(chars.len() + 4);
    for (i, &c) in chars.iter().enumerate() {
        if (c == '+' || c == '-') && i > 0 && numeric(chars[i - 1]) && chars.get(i + 1).is_some_and(|&n| numeric(n)) {
            glued.push(' ');
        }
        glued.push(c);
    }
    let line = if glued.contains(',') {
        if glued.split_whitespace().count() >= 2 && !glued.contains('.') {
            glued.replace(',', ".")
        } else {
            glued.replace(',', " ")
        }
    } else {
        glued
    };
    line.split_whitespace().map(|field| fortran_exponent(field.trim_matches(['(', ')', '"', '\'']))).collect()
}

/// `1.0D-03` -> `1.0e-03`.
fn fortran_exponent(field: &str) -> String {
    let chars: Vec<char> = field.chars().collect();
    let mut out = String::with_capacity(field.len());
    for (i, &c) in chars.iter().enumerate() {
        let after_number = i > 0 && (chars[i - 1].is_ascii_digit() || chars[i - 1] == '.');
        let rest = &chars[i + 1..];
        let digit_follows = match rest.first() {
            Some('+' | '-') => rest.get(1).is_some_and(|d| d.is_ascii_digit()),
            Some(d) => d.is_ascii_digit(),
            None => false,
        };
        out.push(if (c == 'd' || c == 'D') && after_number && digit_follows { 'e' } else { c });
    }
    out
}

/// Decimal places of a number as written: `0.0123` has 4, `1.2346e-03`
/// has 7 and `1.23e+01` has 1.
pub fn get_precision(text: &str) -> i32 {
    let lower = text.to_lowercase().replace('d', "e");
    let (mantissa, exponent) = lower.split_once('e').unwrap_or((&lower, ""));
    let decimals = mantissa.split_once('.').map_or(0, |(_, frac)| frac.chars().count() as i32);
    if exponent.is_empty() {
        return decimals;
    }
    exponent.trim().parse::<i32>().map_or(decimals, |e| decimals - e)
}

/// Decimal places of a file's coordinates: plain numbers may drop trailing
/// zeros (the most decimals any has); numbers in e-notation keep their
/// significant digits (the fewest decimals any has).
fn file_precision(texts: &[&str]) -> i32 {
    let scientific = |t: &str| t.to_lowercase().replace('d', "e").contains('e');
    let plain = texts.iter().filter(|t| !scientific(t)).map(|t| get_precision(t)).max();
    let sci = texts.iter().filter(|t| scientific(t)).map(|t| get_precision(t)).min();
    plain.into_iter().chain(sci).min().map_or(0, |p| p.max(0))
}

/// Points without the outliers far from the median point (unless that would
/// drop more than a fifth of them).
pub fn filter_coordinate_outliers(coords: Vec<Vec2>) -> Vec<Vec2> {
    if coords.len() < 4 {
        return coords;
    }
    let mut sorted = coords.clone();
    sorted.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    let median = sorted[sorted.len() / 2];
    let distances: Vec<f64> = coords.iter().map(|p| (p - median).norm()).collect();
    let mut sd = distances.clone();
    sd.sort_by(f64::total_cmp);
    let (q1, q3) = (sd[sd.len() / 4], sd[3 * sd.len() / 4]);
    let threshold = q3 + 5.0 * (q3 - q1);
    let kept: Vec<Vec2> = coords.iter().zip(&distances).filter(|(_, &d)| d <= threshold).map(|(p, _)| *p).collect();
    if (kept.len() as f64) < coords.len() as f64 * 0.8 {
        coords
    } else {
        kept
    }
}

/// A mistake [`clean_outline`] found in a point list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mistake {
    /// The point before written again, `times` more times.
    Repeated { times: usize },
    /// The whole outline written again from here: `points` more points.
    RepeatedOutline { points: usize },
    /// A stray point after the outline came back to the trailing edge.
    AfterTrailingEdge,
    /// A single point far off the line through its neighbours, bent
    /// against the outline: mistyped.
    Spike,
}

/// A mistake and where it is.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    pub mistake: Mistake,
    /// Index of its (first) point in the list given to [`clean_outline`].
    pub index: usize,
    /// That point as read.
    pub point: Vec2,
    /// Its line in the text, from 1, when read by [`parse_airfoil_text`].
    pub line: Option<usize>,
    /// Indices of the points it removes, in the list given to [`clean_outline`].
    pub removes: Vec<usize>,
}

impl Finding {
    /// Points the mistake removed.
    pub fn removed(&self) -> usize {
        self.removes.len()
    }

    /// The mistake in words, for a report.
    pub fn description(&self) -> String {
        match self.mistake {
            Mistake::Repeated { times: 1 } => "Written twice in a row".into(),
            Mistake::Repeated { times } => format!("Written {} times in a row", times + 1),
            Mistake::RepeatedOutline { points } => format!("The whole outline again, {points} more points"),
            Mistake::AfterTrailingEdge => "A stray point after the closing trailing edge".into(),
            Mistake::Spike => "Sticks out of the outline: mistyped".into(),
        }
    }
}

/// What [`clean_outline`] removed from a point list, in the list's order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cleanup {
    pub findings: Vec<Finding>,
}

impl Cleanup {
    /// Whether anything was removed.
    pub fn any(&self) -> bool {
        !self.findings.is_empty()
    }

    /// Points removed in all.
    pub fn removed(&self) -> usize {
        self.findings.iter().map(Finding::removed).sum()
    }

    /// Points removed by the mistakes `kind` picks.
    pub fn removed_by(&self, kind: impl Fn(&Mistake) -> bool) -> usize {
        self.findings.iter().filter(|f| kind(&f.mistake)).map(Finding::removed).sum()
    }
}

/// At most this many points after the last trailing-edge point are taken
/// for strays (after repeated points are merged); more are an outline part.
const MAX_STRAY_TAIL: usize = 2;

/// A spike stands off the line through its neighbours by more than this
/// fraction of the outline's size.
const SPIKE_OFFSET: f64 = 1e-3;

/// Points without the mistakes airfoil files often hold, each of which makes
/// a fitted curve loop: a point written twice in a row (the fit has to stop
/// and turn back to pass it twice), the outline written twice, stray points
/// after the closing trailing-edge point (`0 0` lines, padding rows), and a
/// single mistyped point that sticks out against the outline's turn.
pub fn clean_outline(coords: Vec<Vec2>) -> (Vec<Vec2>, Cleanup) {
    let mut report = Cleanup::default();
    if coords.len() < 4 {
        return (coords, report);
    }
    let (lo, hi) = coords.iter().fold((Vec2::repeat(f64::INFINITY), Vec2::repeat(f64::NEG_INFINITY)), |(l, h), p| (l.inf(p), h.sup(p)));
    let size = (hi - lo).max();
    if !(size > 0.0) {
        return (coords, report);
    }
    let same = |a: usize, b: usize| (coords[a] - coords[b]).norm() <= 1e-9 * size;
    let found = |report: &mut Cleanup, mistake: Mistake, removes: Vec<usize>| {
        let index = removes[0];
        report.findings.push(Finding { mistake, index, point: coords[index], line: None, removes });
    };
    // The points kept, as indices into `coords`; repeats merge into the first.
    let dedup = |keep: &mut Vec<usize>, report: &mut Cleanup| {
        let mut kept: Vec<usize> = Vec::with_capacity(keep.len());
        let mut run: Vec<usize> = Vec::new();
        for &i in keep.iter() {
            if kept.last().is_some_and(|&last| same(last, i)) {
                run.push(i);
                continue;
            }
            if !run.is_empty() {
                found(report, Mistake::Repeated { times: run.len() }, std::mem::take(&mut run));
            }
            kept.push(i);
        }
        if !run.is_empty() {
            found(report, Mistake::Repeated { times: run.len() }, run);
        }
        *keep = kept;
    };
    let mut keep: Vec<usize> = (0..coords.len()).collect();
    dedup(&mut keep, &mut report);

    // The outline twice: the points from the first return to the start on repeat the start.
    let n = keep.len();
    if let Some(p) = (2..n.saturating_sub(2)).find(|&p| same(keep[p], keep[0]) && (0..n - p).all(|i| same(keep[i + p], keep[i]))) {
        found(&mut report, Mistake::RepeatedOutline { points: n - 1 - p }, keep[p + 1..].to_vec());
        keep.truncate(p + 1);
    }

    // Strays after the trailing edge: the last surface ends at its largest x,
    // or comes back from there towards the first point (closing a blunt
    // edge); a stray `0 0` is no leading edge, so the last points are not
    // searched for it.
    let pts: Vec<Vec2> = keep.iter().map(|&i| coords[i]).collect();
    let n = pts.len();
    let le = (0..n.saturating_sub(MAX_STRAY_TAIL).max(1)).fold(0, |b, i| if pts[i].x < pts[b].x { i } else { b });
    let te = (le..n).fold(le, |b, i| if pts[i].x >= pts[b].x { i } else { b });
    let tail = n - 1 - te;
    let te_gap = (pts[te] - pts[0]).norm();
    if n > MAX_STRAY_TAIL + 2 && tail > 0 && tail <= MAX_STRAY_TAIL && te > le && pts[te + 1..].iter().all(|p| (p - pts[0]).norm() > te_gap) {
        for &i in &keep[te + 1..] {
            found(&mut report, Mistake::AfterTrailingEdge, vec![i]);
        }
        keep.truncate(te + 1);
    }

    // Spikes: an outline turns one way all round (the sign of its area);
    // a point that turns sharply the other way, or leaves and comes back to
    // the same place, is a mistyped one.
    let pts: Vec<Vec2> = keep.iter().map(|&i| coords[i]).collect();
    let area: f64 = pts.windows(2).map(|w| w[0].x * w[1].y - w[1].x * w[0].y).sum::<f64>() + {
        let (a, b) = (pts[pts.len() - 1], pts[0]);
        a.x * b.y - b.x * a.y
    };
    let orientation = area.signum();
    let is_spike = |a: Vec2, p: Vec2, b: Vec2| {
        let (ab, ap, pb) = (b - a, p - a, b - p);
        let base = ab.norm();
        let offset = if base > 0.0 { (ab.x * ap.y - ab.y * ap.x).abs() / base } else { ap.norm() };
        if offset <= SPIKE_OFFSET * size {
            return false;
        }
        let out_and_back = ap.norm().min(pb.norm()) > 10.0 * base;
        let turn = (ap.x * pb.y - ap.y * pb.x).atan2(ap.dot(&pb));
        out_and_back || turn * orientation < -std::f64::consts::FRAC_PI_3
    };
    // The points next to the ends shape a blunt trailing edge and are kept.
    let spikes: Vec<usize> = (2..pts.len().saturating_sub(2)).filter(|&i| is_spike(pts[i - 1], pts[i], pts[i + 1])).collect();
    // Only lone ones: two spikes side by side are a feature of the shape.
    let lone: Vec<usize> = spikes.iter().copied().filter(|i| !spikes.contains(&(i + 1)) && !spikes.contains(&(i.wrapping_sub(1)))).collect();
    if !lone.is_empty() {
        for &k in &lone {
            found(&mut report, Mistake::Spike, vec![keep[k]]);
        }
        keep = keep.iter().enumerate().filter(|(k, _)| !lone.contains(k)).map(|(_, &i)| i).collect();
        dedup(&mut keep, &mut report);
    }
    report.findings.sort_by_key(|f| f.index);
    (keep.iter().map(|&i| coords[i]).collect(), report)
}

/// Leading edge and trailing-edge midpoint this close to (0, 0) and (1, 0),
/// in chords, mean the data are normalized already and kept as written.
const UNIT_CHORD_TOLERANCE: f64 = 5e-3;

/// Normalized coordinates and the factor lengths were scaled by.
fn normalize(coords: Vec<Vec2>) -> (Vec<Vec2>, f64) {
    if coords.len() < 3 {
        return (coords, 1.0);
    }
    let mut coords = coords;
    let mut scale = 1.0;
    let large = coords.iter().filter(|p| p.x.abs() > 2.0 || p.y.abs() > 2.0).count();
    if large as f64 > coords.len() as f64 / 2.0 {
        coords.iter_mut().for_each(|p| *p /= 100.0);
        scale = 0.01;
    }
    let te = 0.5 * (coords[0] + coords[coords.len() - 1]);
    let le_x = coords.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let chord = te.x - le_x;
    if chord < 1e-6 {
        return (coords, scale);
    }
    if le_x.abs() <= UNIT_CHORD_TOLERANCE && (te.x - 1.0).abs() <= UNIT_CHORD_TOLERANCE && te.y.abs() <= UNIT_CHORD_TOLERANCE {
        return (coords, scale);
    }
    let moved = coords.iter().map(|p| Vec2::new((p.x - le_x) / chord, (p.y - te.y) / chord)).collect();
    (moved, scale / chord)
}

/// Decimal places of data written with `precision` decimals once its
/// lengths are multiplied by `scale`, rounded down unless within a few
/// percent of a whole number.
pub fn scaled_precision(precision: i32, scale: f64) -> i32 {
    if !scale.is_finite() || scale <= 0.0 {
        return precision;
    }
    let value = precision as f64 - scale.log10();
    let nearest = value.round_ties_even();
    let result = if (value - nearest).abs() < 0.05 { nearest } else { value.floor() };
    (result as i32).max(0)
}

/// A coordinate line read as numbers, with the texts as written.
struct Row {
    point: Vec2,
    texts: [String; 2],
    /// Line in the text, from 1.
    line: usize,
}

/// Whether a pair is a Lednicer header: two whole point counts.
fn is_point_counts(p: Vec2) -> bool {
    p.x >= 2.0 && p.y >= 2.0 && p.x == p.x.trunc() && p.y == p.y.trunc()
}

/// Point counts that roughly match the rows after them, which start at the
/// leading edge (a Selig file starts at the trailing edge).
fn is_lednicer(rows: &[Row]) -> bool {
    if rows.len() <= 4 || !is_point_counts(rows[0].point) {
        return false;
    }
    let n = rows.len() - 1;
    if (rows[0].point.x + rows[0].point.y - n as f64).abs() > (n / 10).max(2) as f64 {
        return false;
    }
    let xs: Vec<f64> = rows[1..].iter().map(|r| r.point.x).collect();
    let (lo, hi) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
    xs[0] - lo < hi - xs[0]
}

/// A first row far outside the box of the others: a name that reads as
/// numbers (`0012 64`), not a point.
fn is_stray_first_row(rows: &[Row]) -> bool {
    if rows.len() <= 4 {
        return false;
    }
    let rest: Vec<Vec2> = rows[1..].iter().map(|r| r.point).collect();
    let lo = rest.iter().fold(Vec2::repeat(f64::INFINITY), |m, p| m.inf(p));
    let hi = rest.iter().fold(Vec2::repeat(f64::NEG_INFINITY), |m, p| m.sup(p));
    let margin = 0.5 * (hi.x - lo.x).max(hi.y - lo.y);
    let p = rows[0].point;
    !(lo.x - margin <= p.x && p.x <= hi.x + margin && lo.y - margin <= p.y && p.y <= hi.y + margin)
}

/// Lednicer surfaces (upper then lower, both from the leading edge) in
/// Selig order; the leading-edge point once.
fn lednicer_to_selig(coords: &[Vec2], n_upper: usize, n_lower: usize) -> Vec<usize> {
    let mut n_upper = n_upper;
    if n_upper + n_lower != coords.len() {
        let steps: Vec<f64> = coords.windows(2).map(|w| w[1].x - w[0].x).collect();
        n_upper = (0..steps.len()).fold(0, |b, i| if steps[i] < steps[b] { i } else { b }) + 1;
    }
    let n_upper = n_upper.min(coords.len());
    let mut lower_start = n_upper;
    if n_upper > 0 && n_upper < coords.len() && coords[n_upper] == coords[0] {
        lower_start += 1;
    }
    (0..n_upper).rev().chain(lower_start..coords.len()).collect()
}

/// A name line of an airfoil or spline file: one that starts with `#`
/// (after any backslashes) gets one backslash more, so it is no comment.
pub fn name_line(name: &str) -> String {
    let name = name.trim();
    if name.trim_start_matches('\\').starts_with('#') {
        format!("\\{name}")
    } else {
        name.to_string()
    }
}

/// The name a header line holds: undoes [`name_line`].
pub fn name_of_line(line: &str) -> String {
    let escaped = line.starts_with('\\') && line.trim_start_matches('\\').starts_with('#');
    if escaped {
        line[1..].to_string()
    } else {
        line.to_string()
    }
}

/// An airfoil read from text.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedAirfoil {
    /// The name the text holds, or the file name.
    pub name: Option<String>,
    /// Normalized coordinates in Selig order, without the file's mistakes.
    pub coords: Vec<Vec2>,
    /// Decimal places of the coordinates at unit chord.
    pub precision: i32,
    /// Mistakes in the file that were removed, with their lines.
    pub cleanup: Cleanup,
    /// The points as the file has them (in Selig order, not normalized);
    /// the findings' indices point into it.
    pub read_points: Vec<Vec2>,
    /// Decimal places the file writes.
    read_precision: i32,
}

impl ParsedAirfoil {
    /// Normalized coordinates and precision with only the corrections
    /// `accept` picks (one entry per finding; all true gives `coords`, all
    /// false the points as read).
    pub fn with_corrections(&self, accept: &[bool]) -> (Vec<Vec2>, i32) {
        let removed: std::collections::HashSet<usize> =
            self.cleanup.findings.iter().zip(accept).filter(|(_, &a)| a).flat_map(|(f, _)| f.removes.iter().copied()).collect();
        let kept = self.read_points.iter().enumerate().filter(|(i, _)| !removed.contains(i)).map(|(_, p)| *p).collect();
        normalized(kept, self.read_precision)
    }

    /// Normalized coordinates and precision of the points as read.
    pub fn as_read(&self) -> (Vec<Vec2>, i32) {
        self.with_corrections(&[])
    }
}

/// Coordinates without outliers, normalized, and the precision of the file's
/// `precision` decimals at that scale.
fn normalized(coords: Vec<Vec2>, precision: i32) -> (Vec<Vec2>, i32) {
    let (coords, scale) = normalize(filter_coordinate_outliers(coords));
    (coords, scaled_precision(precision, scale))
}

/// Reads an airfoil file; `None` when it holds no coordinates.
pub fn parse_airfoil_file(path: &Path) -> Result<Option<ParsedAirfoil>> {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned());
    Ok(parse_airfoil_text(&read_text(path)?, stem.as_deref()))
}

/// Reads airfoil coordinates from text (for example copied from a web
/// page); `file_name` is the name when the text has none.
pub fn parse_airfoil_text(text: &str, file_name: Option<&str>) -> Option<ParsedAirfoil> {
    let lines: Vec<(&str, &str, usize)> = split_lines(text.trim_start_matches('\u{feff}'))
        .into_iter()
        .enumerate()
        .filter_map(|(number, line)| {
            // The whole line is kept for a name, which may hold a # ("NACA 0012 #2").
            let content = line.split('#').next().unwrap().trim();
            (!content.is_empty()).then(|| (content, line.trim(), number + 1))
        })
        .collect();
    let first_line = lines.first()?.1;
    let mut name = file_name.map(str::to_string);
    let mut rows: Vec<Row> = Vec::new();
    let mut first_is_row = false;
    for (i, (content, _, line)) in lines.iter().enumerate() {
        let fields = number_fields(content);
        match (fields.first().and_then(|f| parse_number(f)), fields.get(1).and_then(|f| parse_number(f))) {
            (Some(x), Some(y)) => {
                rows.push(Row { point: Vec2::new(x, y), texts: [fields[0].clone(), fields[1].clone()], line: *line });
                first_is_row |= i == 0;
            }
            _ if i == 0 => name = Some(name_of_line(first_line)),
            _ => {}
        }
    }
    if first_is_row && !is_lednicer(&rows) && (is_lednicer(&rows[1..]) || is_stray_first_row(&rows)) {
        name = Some(first_line.to_string());
        rows.remove(0);
    }
    let order: Vec<usize> = if is_lednicer(&rows) {
        let header = rows.remove(0).point;
        let points: Vec<Vec2> = rows.iter().map(|r| r.point).collect();
        lednicer_to_selig(&points, header.x as usize, header.y as usize)
    } else {
        (0..rows.len()).collect()
    };
    if order.is_empty() {
        return None;
    }
    let coords: Vec<Vec2> = order.iter().map(|&i| rows[i].point).collect();
    let texts: Vec<&str> = rows.iter().flat_map(|r| r.texts.iter().map(String::as_str)).collect();
    let precision = file_precision(&texts);
    let (cleaned, mut cleanup) = clean_outline(coords.clone());
    for f in &mut cleanup.findings {
        f.line = Some(rows[order[f.index]].line);
    }
    let (corrected, corrected_precision) = normalized(cleaned, precision);
    Some(ParsedAirfoil { name, coords: corrected, precision: corrected_precision, cleanup, read_points: coords, read_precision: precision })
}

/// Number of decimals written: 8 by default, at most 16.
fn decimals(precision: Option<i32>) -> usize {
    precision.unwrap_or(8).clamp(0, 16) as usize
}

fn point_line(p: Vec2, d: usize) -> String {
    format!("{} {}\n", fixed(p.x, d), fixed(p.y, d))
}

fn write_file(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| Error::new(e.to_string()))?;
    }
    std::fs::write(path, text).map_err(|e| Error::new(e.to_string()))
}

/// Writes coordinates in Selig format with an optional name line.
pub fn export_airfoil_data(path: &Path, coords: &[Vec2], name: Option<&str>, precision: Option<i32>) -> Result<()> {
    write_file(path, &airfoil_data_text(coords, name, precision)?)
}

/// The text [`export_airfoil_data`] writes.
pub fn airfoil_data_text(coords: &[Vec2], name: Option<&str>, precision: Option<i32>) -> Result<String> {
    if coords.is_empty() {
        bail!("No coordinates to export.");
    }
    let d = decimals(precision);
    let mut text = String::new();
    if let Some(name) = name.filter(|n| !n.is_empty()) {
        text += &name_line(name);
        text.push('\n');
    }
    for &p in coords {
        text += &point_line(p, d);
    }
    Ok(text)
}

/// First line of the comment header of a spline file.
pub const SPLINE_FORMAT_TAG: &str = "spline-fitter B-spline v2";

/// A spline file: degree, control points and knots of every section.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SplineFile {
    pub name: String,
    pub degree: Option<usize>,
    pub control_points: Vec<Vec<Vec2>>,
    /// One knot vector per section (open uniform when absent).
    pub knots: Option<Vec<Vec<f64>>>,
    /// Junction continuity labels (`G2`, `C1`, ...).
    pub continuity: Option<Vec<String>>,
}

/// Writes the control points of all sections (each junction point once)
/// with the degree, counts, knots and continuity in `#` comment lines, so
/// the file still reads as a plain point list.
pub fn export_spline_data(path: &Path, spline: &SplineFile, precision: Option<i32>) -> Result<()> {
    write_file(path, &spline_data_text(spline, precision)?)
}

/// The text [`export_spline_data`] writes.
pub fn spline_data_text(spline: &SplineFile, precision: Option<i32>) -> Result<String> {
    let sections = &spline.control_points;
    if sections.iter().all(Vec::is_empty) {
        bail!("No control points to export.");
    }
    let Some(degree) = spline.degree else { bail!("The spline has no degree.") };
    let knots = match &spline.knots {
        Some(k) => k.clone(),
        None => sections.iter().map(|c| open_uniform_knots(degree, c.len())).collect::<Result<_>>()?,
    };
    if knots.len() != sections.len() {
        bail!("Need one knot vector per section.");
    }
    if knots.iter().zip(sections).any(|(k, c)| k.len() != c.len() + degree + 1) {
        bail!("Knot vector length must be control points + degree + 1.");
    }
    let d = decimals(precision);
    let mut text = String::new();
    if !spline.name.is_empty() {
        text += &format!("{}\n", name_line(&spline.name));
    }
    text += &format!("# {SPLINE_FORMAT_TAG}\n# degree {degree}\n# sections {}\n", sections.len());
    if let Some(labels) = spline.continuity.as_ref().filter(|c| !c.is_empty()) {
        text += &format!("# continuity {}\n", labels.join(" "));
    }
    for (i, (control, k)) in sections.iter().zip(&knots).enumerate() {
        let values: Vec<String> = k.iter().map(|&v| shortest(v)).collect();
        text += &format!("# section {} control_points {}\n# knots {} {}\n", i + 1, control.len(), i + 1, values.join(" "));
    }
    for (i, control) in sections.iter().enumerate() {
        for &p in &control[usize::from(i > 0)..] {
            text += &point_line(p, d);
        }
    }
    Ok(text)
}

/// Reads a file written by [`export_spline_data`]. Files from before the
/// header was written hold only points: one section, with the degree from
/// a `.deg{N}.` part of the file name if there is one.
pub fn parse_spline_data(path: &Path) -> Result<SplineFile> {
    let text = read_text(path)?;
    let mut spline = SplineFile::default();
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    let mut knots: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    let mut points: Vec<Vec2> = Vec::new();
    let mut first_line: Option<String> = None;
    for line in split_lines(&text) {
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        let is_first = first_line.is_none();
        if is_first {
            first_line = Some(text.to_string());
        }
        if let Some(comment) = text.strip_prefix('#') {
            let words: Vec<&str> = comment.split_whitespace().collect();
            let int = |w: &str| w.parse::<i64>().ok();
            match words.as_slice() {
                ["degree", d, ..] => spline.degree = d.parse().ok(),
                ["section", i, "control_points", n, ..] => {
                    if let (Some(i), Some(n)) = (int(i), n.parse().ok()) {
                        counts.insert(i, n);
                    }
                }
                ["knots", i, values @ ..] if !values.is_empty() => {
                    let parsed: Option<Vec<f64>> = values.iter().map(|v| v.parse().ok()).collect();
                    if let (Some(i), Some(parsed)) = (int(i), parsed) {
                        knots.insert(i, parsed);
                    }
                }
                ["continuity", labels @ ..] if !labels.is_empty() => {
                    spline.continuity = Some(labels.iter().map(|s| s.to_string()).collect());
                }
                _ => {}
            }
            continue;
        }
        let parts: Vec<&str> = text.split_whitespace().collect();
        match (parts.first().and_then(|v| parse_number(v)), parts.get(1).and_then(|v| parse_number(v))) {
            (Some(x), Some(y)) => points.push(Vec2::new(x, y)),
            _ if is_first => spline.name = name_of_line(text),
            _ => {}
        }
    }
    if spline.degree.is_none() {
        let file_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let re = regex::Regex::new(r"\.deg(\d+)\.").unwrap();
        spline.degree = re.captures(&file_name).and_then(|c| c[1].parse().ok());
    }
    if counts.is_empty() {
        spline.control_points = vec![points];
        return Ok(spline);
    }
    let expected = counts.values().sum::<usize>() + 1 - counts.len();
    let first_line = first_line.unwrap_or_default();
    if spline.name.is_empty() && points.len() == expected + 1 && !first_line.starts_with('#') {
        // A name that reads as a point, like "0012 64": one point too many.
        spline.name = first_line;
        points.remove(0);
    }
    let mut offset = 0;
    for (k, &n) in counts.values().enumerate() {
        let take = if k == 0 { n } else { n.saturating_sub(1) };
        let end = (offset + take).min(points.len());
        let mut section: Vec<Vec2> = match spline.control_points.last() {
            Some(previous) if k > 0 => vec![previous[previous.len() - 1]],
            _ => vec![],
        };
        section.extend_from_slice(&points[offset..end]);
        spline.control_points.push(section);
        offset += take;
    }
    if offset != points.len() {
        bail!("{}: expected {offset} control points, found {}.", path.display(), points.len());
    }
    if knots.len() == counts.len() && knots.keys().eq(counts.keys()) {
        spline.knots = Some(knots.into_values().collect());
    }
    Ok(spline)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Vec2 {
        Vec2::new(x, y)
    }

    #[test]
    fn precision_and_fields() {
        assert_eq!(get_precision("0.0123"), 4);
        assert_eq!(get_precision("1.2346e-03"), 7);
        assert_eq!(get_precision("1.23e+01"), 1);
        assert_eq!(get_precision("5"), 0);
        assert_eq!(number_fields("0.95000-0.01200"), ["0.95000", "-0.01200"]);
        assert_eq!(number_fields("0,5 0,01"), ["0.5", "0.01"]);
        assert_eq!(number_fields("1.0, 0.5"), ["1.0", "0.5"]);
        assert_eq!(number_fields("1.0D-03 2.5d+00"), ["1.0e-03", "2.5e+00"]);
        assert_eq!(number_fields("(1.0; \"2.0\")"), ["1.0", "2.0"]);
        assert_eq!(number_fields("() 1"), ["", "1"]);
        assert_eq!(scaled_precision(4, 0.01), 6);
        assert_eq!(scaled_precision(5, 0.5), 5);
        assert_eq!(scaled_precision(6, 1.0 / 0.98), 6);
    }

    fn parse(text: &str) -> ParsedAirfoil {
        parse_airfoil_text(text, Some("f")).unwrap()
    }

    #[test]
    fn acceptance_parsing() {
        let unit = vec![p(1.0, 0.0), p(0.5, 0.05), p(0.0, 0.0), p(0.5, -0.05), p(1.0, 0.0)];
        let a = parse("NACA test\n1.0000 0.0000\n0.5000 0.0500\n0.0000 0.0000\n0.5000 -0.0500\n1.0000 0.0000\n");
        assert_eq!((a.name.as_deref(), &a.coords, a.precision), (Some("NACA test"), &unit, 4));
        let b = parse("NACA test\n100 0\n50 5\n0 0\n50 -5\n100 0\n");
        assert_eq!((&b.coords, b.precision), (&unit, 2));
        let c = parse("T\n3 3\n0 0\n0.5 0.06\n1 0\n0 0\n0.5 -0.05\n1 0\n");
        assert_eq!(c.name.as_deref(), Some("T"));
        assert_eq!(c.coords, vec![p(1.0, 0.0), p(0.5, 0.06), p(0.0, 0.0), p(0.5, -0.05), p(1.0, 0.0)]);
        assert_eq!(c.precision, 2);
        let d = parse("0012 64\n1 0\n0.5 0.06\n0 0\n0.5 -0.06\n1 0\n");
        assert_eq!((d.name.as_deref(), d.coords.len(), d.precision), (Some("0012 64"), 5, 2));
        let e = parse("#c\n\\#name\n1 0\n0.5 0.06\n0 0\n0.5 -0.06\n1 0\n");
        assert_eq!(e.name.as_deref(), Some("#name"));
        let f = parse("x\n2 0.2\n1 0.1\n0 0\n1 -0.1\n2 -0.2\n");
        assert_eq!(f.coords, vec![p(1.0, 0.1), p(0.5, 0.05), p(0.0, 0.0), p(0.5, -0.05), p(1.0, -0.1)]);
        assert_eq!(f.precision, 1);
        assert!(parse_airfoil_text("# nothing\n\n", None).is_none());
    }

    /// A NACA 0012 outline in Selig order, the leading edge once.
    fn naca0012() -> Vec<Vec2> {
        let half = |x: f64| 0.6 * (0.2969 * x.sqrt() - 0.126 * x - 0.3516 * x * x + 0.2843 * x.powi(3) - 0.1036 * x.powi(4));
        let xs: Vec<f64> = (0..=15).map(|i| 0.5 * (1.0 - (std::f64::consts::PI * i as f64 / 15.0).cos())).collect();
        xs.iter().rev().map(|&x| p(x, half(x))).chain(xs[1..].iter().map(|&x| p(x, -half(x)))).collect()
    }

    #[test]
    fn clean_outline_removes_file_mistakes() {
        let good = naca0012();
        let le = good.len() / 2;
        assert_eq!(clean_outline(good.clone()), (good.clone(), Cleanup::default()));
        let with = |edit: &dyn Fn(&mut Vec<Vec2>)| {
            let mut pts = good.clone();
            edit(&mut pts);
            clean_outline(pts)
        };
        // The leading edge written twice (BigFoil files).
        let (pts, c) = with(&|v| v.insert(le, v[le]));
        assert_eq!((pts, c.removed_by(|m| matches!(m, Mistake::Repeated { .. }))), (good.clone(), 1));
        // A stray `0 0` after the closing trailing edge (hn374): only the
        // stray goes, and a closing point back at the trailing edge stays.
        let (pts, c) = with(&|v| v.push(p(0.0, 0.0)));
        assert_eq!((pts, c.removed_by(|m| *m == Mistake::AfterTrailingEdge)), (good.clone(), 1));
        // Padding rows after it, left of the leading edge (rs001t).
        let (pts, c) = with(&|v| v.extend([p(-0.1, 0.0); 30]));
        assert_eq!((pts, c.removed_by(|m| *m == Mistake::AfterTrailingEdge), c.removed_by(|m| matches!(m, Mistake::Repeated { .. }))), (good.clone(), 1, 29));
        // A stray `0 0` and a point stepping back (nm26-3), the nose just off x = 0.
        let shifted: Vec<Vec2> = good.iter().map(|q| q + p(1e-5, 0.0)).collect();
        let mut v = shifted.clone();
        v.extend([p(0.0, 0.0), p(0.9999, -1e-4)]);
        assert_eq!(clean_outline(v).0, shifted);
        // The whole outline twice (mh28).
        let (pts, c) = with(&|v| v.extend(good.clone()));
        assert_eq!((pts, c.removed_by(|m| matches!(m, Mistake::RepeatedOutline { .. }))), (good.clone(), good.len() - 1));
        // A far point between two copies of the leading edge (e850).
        let (pts, c) = with(&|v| v.splice(le + 1..le + 1, [p(1.006, -0.0015), v[le]]).for_each(drop));
        assert_eq!((pts, c.removed_by(|m| *m == Mistake::Spike), c.removed_by(|m| matches!(m, Mistake::Repeated { .. }))), (good.clone(), 1, 1));
        // A mistyped upper point near the nose (goe451: y written as 0).
        let (pts, c) = with(&|v| v[le - 2].y = 0.0);
        let mut expected = good.clone();
        expected.remove(le - 2);
        assert_eq!((pts, c.removed_by(|m| *m == Mistake::Spike)), (expected, 1));
        // A blunt trailing edge closed back to its middle keeps its closing point.
        let mut blunt = good.clone();
        let (first, last) = (blunt.len() - 1, 0);
        blunt[first].y = -0.005;
        blunt[last].y = 0.005;
        blunt.push(p(1.0, 0.0));
        assert_eq!(clean_outline(blunt.clone()).0, blunt);
        // The parser cleans what it reads, tells the lines, and keeps the
        // points as read.
        let a = parse("t\n1 0\n0.5 0.06\n0 0\n# c\n0 0\n0.5 -0.06\n1 0\n0 0\n");
        assert_eq!(a.coords.len(), 5);
        let found: Vec<(Mistake, Option<usize>)> = a.cleanup.findings.iter().map(|f| (f.mistake, f.line)).collect();
        assert_eq!(found, [(Mistake::Repeated { times: 1 }, Some(6)), (Mistake::AfterTrailingEdge, Some(9))]);
        assert_eq!(a.cleanup.findings[1].description(), "A stray point after the closing trailing edge");
        assert_eq!(a.as_read().0.len(), 7);
        assert_eq!(a.with_corrections(&[true, false]).0.len(), 6);
        assert_eq!(a.with_corrections(&[false, true]).0.len(), 6);
        assert_eq!(a.with_corrections(&[true, true]), (a.coords.clone(), a.precision));
        assert_eq!(a.cleanup.findings.iter().map(|f| f.removes.clone()).collect::<Vec<_>>(), [vec![3], vec![6]]);
        // Lednicer files tell the lines too (upper surface read backwards).
        let l = parse("t\n3 3\n0 0\n0.5 0.06\n1 0\n0 0\n0.5 -0.05\n0.5 -0.05\n1 0\n");
        assert_eq!(l.cleanup.findings.iter().map(|f| f.line).collect::<Vec<_>>(), [Some(8)]);
    }

    #[test]
    fn name_lines_round_trip() {
        assert_eq!(name_line("#name"), "\\#name");
        assert_eq!(name_line("\\#x"), "\\\\#x");
        assert_eq!(name_of_line("\\\\#x"), "\\#x");
        assert_eq!(name_of_line("plain"), "plain");
    }

    #[test]
    fn spline_file_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.deg5.spline.dat");
        let knots = vec![bspline_varpro::clamped_knots(3, &[0.3]), open_uniform_knots(3, 4).unwrap()];
        let spline = SplineFile {
            name: "0012 64".into(),
            degree: Some(3),
            control_points: vec![
                vec![p(1.0, 0.0), p(0.7, 0.1), p(0.3, 0.1), p(0.1, 0.05), p(0.0, 0.0)],
                vec![p(0.0, 0.0), p(0.3, -0.1), p(0.7, -0.1), p(1.0, 0.0)],
            ],
            knots: Some(knots),
            continuity: Some(vec!["G2".into()]),
        };
        export_spline_data(&path, &spline, Some(6)).unwrap();
        assert_eq!(parse_spline_data(&path).unwrap(), spline);
    }
}
