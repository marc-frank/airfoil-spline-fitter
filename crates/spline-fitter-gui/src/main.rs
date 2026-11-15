//! Airfoil Spline Fitter: fit B-spline chains to airfoil coordinates.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod actions;
mod app;
mod chime;
mod dialogs;
mod explorer;
mod fitting;
mod fonts;
mod history;
mod job;
mod locale;
mod overlays;
mod plots;
mod runs;
mod settings;
mod sidebar;
mod store;
mod theme;
#[cfg(test)]
mod ui_tests;
mod widgets;

use eframe::egui;

fn icon() -> Option<egui::IconData> {
    let image = image::load_from_memory(include_bytes!("../assets/app_icon.png")).ok()?.into_rgba8();
    let (width, height) = image.dimensions();
    Some(egui::IconData { rgba: image.into_raw(), width, height })
}

fn main() -> eframe::Result {
    let mut viewport = egui::ViewportBuilder::default()
        .with_title(spline_fitter::APP_NAME)
        .with_app_id("AirfoilSplineFitter")
        .with_maximized(true)
        .with_min_inner_size([960.0, 600.0])
        .with_drag_and_drop(true);
    if let Some(icon) = icon() {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions { viewport, persist_window: false, ..Default::default() };
    eframe::run_native(spline_fitter::APP_NAME, options, Box::new(|cc| Ok(Box::new(app::App::new(cc)))))
}
