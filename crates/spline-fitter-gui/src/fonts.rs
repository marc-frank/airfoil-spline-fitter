//! The system's UI face, its semibold weight (captions, the file name and
//! the primary button) and a monospace face for numbers, when installed.

use std::sync::Arc;

use eframe::egui::{self, FontData, FontDefinitions, FontFamily, FontId};

/// The semibold family (falls back to the regular face).
pub fn semibold() -> FontFamily {
    FontFamily::Name("semibold".into())
}

/// The semibold face at `size` pixels.
pub fn semibold_id(size: f32) -> FontId {
    FontId::new(size, semibold())
}

/// Candidate font files, first found wins.
const PROPORTIONAL: &[&str] =
    &["C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/SFNS.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"];
const SEMIBOLD: &[&str] = &["C:/Windows/Fonts/seguisb.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"];
const MONOSPACE: &[&str] = &[
    "C:/Windows/Fonts/CascadiaMono.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
];

pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    for (family, candidates) in [(FontFamily::Proportional, PROPORTIONAL), (FontFamily::Monospace, MONOSPACE)] {
        if let Some((name, bytes)) = candidates.iter().find_map(|p| std::fs::read(p).ok().map(|b| (p.to_string(), b))) {
            if family == FontFamily::Proportional {
                SEGOE.store(name.ends_with("segoeui.ttf"), std::sync::atomic::Ordering::Relaxed);
            }
            fonts.font_data.insert(name.clone(), Arc::new(FontData::from_owned(bytes)));
            fonts.families.entry(family).or_default().insert(0, name);
        }
    }
    let regular = fonts.families[&FontFamily::Proportional].clone();
    let mut bold = regular.clone();
    if let Some((name, bytes)) = SEMIBOLD.iter().find_map(|p| std::fs::read(p).ok().map(|b| (p.to_string(), b))) {
        fonts.font_data.insert(name.clone(), Arc::new(FontData::from_owned(bytes)));
        bold.insert(0, name);
    }
    fonts.families.insert(semibold(), bold);
    ctx.set_fonts(fonts);
}

/// Whether the UI face is Segoe UI, whose whole-pixel advances are below.
static SEGOE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Advances of the printable ASCII characters (from the space) in Segoe UI
/// at 12 px, as Qt lays text out on Windows: hinted to whole pixels.
const SEGOE_REGULAR: [u8; 95] = [
    3, 3, 5, 7, 6, 10, 10, 3, 4, 4, 5, 8, 3, 5, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 8, 8, 8, 5, 11, 8, 7, 8, 8, 6, 6, 8, 9, 3, 4, 7,
    6, 11, 9, 9, 7, 9, 7, 6, 7, 8, 7, 11, 7, 7, 7, 4, 5, 4, 8, 5, 3, 6, 7, 6, 7, 6, 4, 7, 7, 3, 3, 6, 3, 11, 7, 7, 7, 7, 4, 5, 4, 7, 6, 9,
    5, 6, 5, 4, 3, 4, 8,
];
/// The same in bold, which Qt measures header sections in.
const SEGOE_BOLD: [u8; 95] = [
    3, 4, 6, 7, 7, 10, 10, 4, 4, 4, 5, 8, 3, 5, 3, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3, 3, 8, 8, 8, 5, 11, 8, 8, 7, 9, 6, 6, 9, 9, 4, 5, 8,
    6, 11, 9, 9, 7, 9, 8, 7, 7, 9, 8, 12, 8, 7, 7, 4, 5, 4, 8, 5, 4, 6, 7, 6, 7, 7, 5, 7, 7, 3, 3, 7, 3, 11, 7, 7, 7, 7, 5, 5, 5, 7, 7, 10,
    7, 6, 6, 4, 4, 4, 8,
];

/// Left and right bearings of the same characters (regular): how far the
/// ink stays inside the advance.
const SEGOE_LEFT: [f32; 95] = [
    0.00, 1.05, 0.95, 0.19, 0.94, 0.47, 0.69, 0.95, 0.81, 0.12, 0.44, 1.36, 0.22, 0.84, 0.66, -0.16, 0.50, 1.33, 0.62, 0.72, 0.08, 0.95,
    0.64, 0.56, 0.52, 0.55, 0.66, 0.22, 1.59, 1.36, 1.59, 0.77, 1.00, 0.12, 1.09, 0.55, 1.09, 1.09, 1.09, 0.55, 1.09, 1.09, 0.11, 1.09,
    1.09, 1.09, 1.09, 0.55, 1.09, 0.55, 1.09, 0.70, 0.23, 0.98, 0.09, 0.14, 0.14, 0.09, 0.19, 1.17, -0.14, 0.30, 1.34, 0.00, 0.47, 0.52,
    0.97, 0.56, 0.56, 0.56, 0.30, 0.56, 0.97, 0.84, -1.16, 0.97, 0.97, 0.97, 0.97, 0.56, 0.97, 0.56, 0.97, 0.61, 0.25, 0.84, 0.08, 0.14,
    0.14, 0.08, 0.19, 0.53, 1.00, 0.38, 1.22,
];
const SEGOE_RIGHT: [f32; 95] = [
    3.28, 1.05, 0.95, 0.25, 0.73, 0.44, 0.23, 0.95, 0.11, 0.80, 0.42, 1.33, 0.75, 0.75, 0.62, -0.08, 0.48, 2.22, 0.64, 0.88, 0.41, 0.78,
    0.47, 0.55, 0.48, 0.58, 0.62, 0.61, 1.56, 1.33, 1.56, 0.58, 0.98, 0.12, 0.59, 0.61, 0.55, 0.52, 0.48, 0.84, 1.09, 1.09, 1.03, 0.00,
    0.19, 1.09, 1.09, 0.55, 0.47, -0.16, 0.05, 0.56, 0.20, 0.98, 0.09, 0.14, 0.17, 0.06, 0.25, 0.28, -0.12, 1.16, 1.33, 0.00, 0.45, 0.83,
    0.56, 0.47, 0.97, 0.47, -0.17, 0.97, 0.83, 0.80, 0.80, 0.00, 0.97, 0.84, 0.83, 0.56, 0.56, 0.97, 0.06, 0.50, 0.30, 0.95, 0.05, 0.14,
    0.14, 0.05, 0.27, 0.36, 0.98, 0.52, 1.20,
];

/// The width of `text`'s ink as Qt bounds it (`QFontMetrics::boundingRect`):
/// the advances less the outer bearings, rounded. Without the table, the
/// advances.
pub fn ink_width(ui: &egui::Ui, text: &str, font: &FontId) -> f32 {
    let width = layout_width(ui, text, font, false);
    let segoe = font.family == FontFamily::Proportional
        && font.size == crate::theme::typeface::BASE
        && SEGOE.load(std::sync::atomic::Ordering::Relaxed);
    let bearing = |c: Option<char>, table: &[f32; 95]| c.and_then(|c| (c as usize).checked_sub(32)).and_then(|i| table.get(i)).copied();
    match (segoe, bearing(text.chars().next(), &SEGOE_LEFT), bearing(text.chars().last(), &SEGOE_RIGHT)) {
        (true, Some(left), Some(right)) => (width - left - right).round(),
        _ => width,
    }
}

/// The width Qt gives `text` in `font` (bold: as a header section measures
/// it): every glyph's advance in whole pixels. Exact for Segoe UI at the
/// base size; otherwise each glyph is measured and rounded.
pub fn layout_width(ui: &egui::Ui, text: &str, font: &FontId, bold: bool) -> f32 {
    let table = (font.family == FontFamily::Proportional
        && font.size == crate::theme::typeface::BASE
        && SEGOE.load(std::sync::atomic::Ordering::Relaxed))
    .then_some(if bold { &SEGOE_BOLD } else { &SEGOE_REGULAR });
    let painter = ui.painter();
    text.chars()
        .map(|c| match (table, (c as usize).checked_sub(32)) {
            (Some(t), Some(i)) if i < t.len() => t[i] as f32,
            _ => painter.layout_no_wrap(c.to_string(), font.clone(), ui.visuals().text_color()).size().x.round(),
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Widths the Python window's explorer measured (Segoe UI only).
    #[test]
    fn widths_follow_qt() {
        let ctx = egui::Context::default();
        install(&ctx);
        if !SEGOE.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let font = FontId::proportional(crate::theme::typeface::BASE);
        let mut out = ctx.run_ui(Default::default(), |ui| {
            assert_eq!(layout_width(ui, "2026-11-16 00:16", &font, false), 88.0);
            assert_eq!(layout_width(ui, "Centripetal", &font, false), 58.0);
            assert_eq!(layout_width(ui, "Fitted control points", &font, true), 113.0);
            assert_eq!(layout_width(ui, "Run", &font, true), 22.0);
            for (text, ink) in [("None", 27.0), ("Iterations", 47.0), ("Y only", 33.0), ("Control points", 75.0), ("Time", 26.0)] {
                assert_eq!(ink_width(ui, text, &font), ink, "{text}");
            }
        });
        out.textures_delta.clear();
    }
}
