//! The system's UI face and a monospace face for numbers, when installed.

use std::sync::Arc;

use eframe::egui::{self, FontData, FontDefinitions, FontFamily};

/// Candidate font files, first found wins.
const PROPORTIONAL: &[&str] =
    &["C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/SFNS.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"];
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
    ctx.set_fonts(fonts);
}
