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
