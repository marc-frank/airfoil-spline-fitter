//! Airfoil Spline Fitter: fit B-spline chains to airfoil coordinates.
//!
//! A desktop application, and a web page built from the same code
//! (packaging/web: Trunk builds it to WebAssembly for GitHub Pages).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod actions;
mod app;
mod axes;
mod chime;
mod dialogs;
mod explorer;
mod fitting;
mod fonts;
mod history;
mod job;
mod locale;
mod overlays;
mod parent;
mod plots;
mod runs;
mod settings;
mod sidebar;
mod store;
mod theme;
#[cfg(test)]
mod ui_tests;
#[cfg(target_arch = "wasm32")]
mod web;
mod widgets;

#[cfg(not(target_arch = "wasm32"))]
use eframe::egui;

#[cfg(not(target_arch = "wasm32"))]
fn icon() -> Option<egui::IconData> {
    let image = image::load_from_memory(include_bytes!("../assets/app_icon.png")).ok()?.into_rgba8();
    let (width, height) = image.dimensions();
    Some(egui::IconData { rgba: image.into_raw(), width, height })
}

#[cfg(not(target_arch = "wasm32"))]
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

/// The web page: the app in the page's canvas (crates/spline-fitter-gui/index.html).
#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast;

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().and_then(|w| w.document()).expect("a web page");
        let canvas: web_sys::HtmlCanvasElement =
            document.get_element_by_id("app").expect("the page has a canvas with id app").unchecked_into();
        let started = eframe::WebRunner::new()
            .start(canvas, eframe::WebOptions::default(), Box::new(|cc| Ok(Box::new(app::App::new(cc)))))
            .await;
        // Replace the loading text with the app, or with what went wrong.
        if let Some(loading) = document.get_element_by_id("loading") {
            match started {
                Ok(()) => loading.remove(),
                Err(e) => loading.set_text_content(Some(&format!("The app could not start: {e:?}"))),
            }
        }
    });
}
