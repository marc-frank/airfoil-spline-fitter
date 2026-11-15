//! Dialogs, the fit report window, and the plot toolbar's drop-downs.

use std::time::{Duration, Instant};

use eframe::egui::{self, Align, Layout, RichText, Ui};
use spline_fitter::analysis::{Note, Report, JUNCTION_COLUMNS};
use spline_fitter::bspline_varpro::airfoil::Shape;

use crate::app::{App, DEFAULT_SCROLL_MS, DEFAULT_ZOOM_MS};
use crate::overlays::QUANTITIES;
use crate::plots::{shape_plot, View};
use crate::runs::COLUMNS;
use crate::store::Store;
use crate::theme::{size, Glyph, ThemeMode};
use crate::widgets::{self, form_row, muted};

/// CAD formats: filter text and suffix.
pub const CAD_FORMATS: [(&str, &str); 3] = [("STEP AP214 (*.step)", ".step"), ("IGES 5.3 (*.igs)", ".igs"), ("DXF R2000 (*.dxf)", ".dxf")];
const CAD_UNITS: [&str; 3] = ["mm", "m", "inch"];

/// The Export CAD choices.
#[derive(Debug, Clone, PartialEq)]
pub struct CadChoice {
    pub format: usize,
    pub units: &'static str,
    pub scale: f64,
    pub merge: bool,
    pub close_te: bool,
}

/// An open dialog.
pub enum Dialog {
    Naca { code: String, per_side: usize, closed: bool },
    Cad(CadChoice),
    Batch,
    Theme { pending: Option<(i32, Instant)>, slider: i32 },
    Motion,
    PointsPerSide { value: usize },
}

impl Dialog {
    pub fn naca(store: &Store) -> Self {
        Dialog::Naca {
            code: store.string("naca_code", "2412"),
            per_side: store.int("naca_points", 81).max(10) as usize,
            closed: store.bool("naca_closed_te", true),
        }
    }

    pub fn cad(store: &Store) -> Self {
        let units = store.string("cad_units", "mm");
        Dialog::Cad(CadChoice {
            format: (store.int("cad_format", 0).clamp(0, 2)) as usize,
            units: CAD_UNITS.into_iter().find(|u| *u == units).unwrap_or("mm"),
            scale: store.float("cad_scale", 1.0).max(1e-6),
            merge: store.bool("cad_merge", false),
            close_te: store.bool("cad_close_te", true),
        })
    }

    pub fn batch() -> Self {
        Dialog::Batch
    }

    pub fn theme(app: &App) -> Self {
        Dialog::Theme { pending: None, slider: app.temperature }
    }

    pub fn motion(_app: &App) -> Self {
        Dialog::Motion
    }

    pub fn points_per_side(value: usize) -> Self {
        Dialog::PointsPerSide { value }
    }
}

/// What a dialog's buttons asked for.
enum Answer {
    Open,
    Accept,
    Reject,
}

/// A modal dialog with a title; the content returns the answer.
fn modal(ctx: &egui::Context, title: &str, width: f32, content: impl FnOnce(&mut Ui) -> Answer) -> Answer {
    let frame = egui::Frame::popup(&ctx.global_style()).inner_margin(egui::Margin::same(size::PAD_LARGE as i8));
    let response = egui::Modal::new(egui::Id::unique(("dialog", title))).frame(frame).show(ctx, |ui| {
        ui.set_width(width);
        ui.heading(title);
        ui.add_space(size::GROUP);
        content(ui)
    });
    if response.should_close() {
        return Answer::Reject;
    }
    response.inner
}

/// Dialog buttons: reset-role at the left, then accept, then reject.
fn buttons(ui: &mut Ui, accept: Option<&str>, reject: &str, reset: Option<(&str, &str)>) -> (bool, bool, bool) {
    let (mut a, mut r, mut z) = (false, false, false);
    ui.add_space(size::GROUP);
    ui.horizontal(|ui| {
        if let Some((text, tip)) = reset {
            z = widgets::button(ui, text, tip).clicked();
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            r = widgets::button(ui, reject, "").clicked();
            if let Some(text) = accept {
                a = widgets::button(ui, text, "").clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter));
            }
        });
    });
    (a, r, z)
}

impl App {
    pub fn show_dialogs(&mut self, ctx: &egui::Context) {
        self.report_window(ctx);
        let Some(mut dialog) = self.dialog.take() else { return };
        let p = self.palette.clone();
        let keep = match &mut dialog {
            Dialog::Naca { code, per_side, closed } => {
                let valid = code.len() == 4 && code.chars().all(|c| c.is_ascii_digit());
                let answer = modal(ctx, "New NACA airfoil", 360.0, |ui| {
                    form_row(ui, &p, "Digits", true, "", |ui| {
                        ui.add(egui::TextEdit::singleline(code).char_limit(4).desired_width(120.0)).on_hover_text(
                            "Four digits: the camber in percent of the chord, its position in tenths\nof the chord, and the thickness in percent. 2412 has 2% camber at 40%\nof the chord and is 12% thick; 0012 is symmetric.",
                        );
                    });
                    form_row(ui, &p, "Points per side", true, "", |ui| {
                        widgets::int_field(ui, &p, "naca_points", per_side, 10, crate::settings::UNLIMITED, 120.0, "", None, 1, "Spaced closer toward the leading and trailing edge");
                    });
                    form_row(ui, &p, "Trailing edge", true, "", |ui| {
                        widgets::segmented(ui, closed, &[(true, "Closed", "Modified thickness coefficient: a sharp trailing edge"), (false, "Open", "Original thickness coefficient: a thin blunt trailing edge")], 160.0);
                    });
                    let mut accepted = false;
                    let mut rejected = false;
                    ui.add_space(size::GROUP);
                    ui.horizontal(|ui| {
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            rejected = widgets::button(ui, "Cancel", "").clicked();
                            accepted = ui.add_enabled(valid, egui::Button::new("Create").min_size(egui::vec2(0.0, size::CONTROL_HEIGHT))).clicked();
                        });
                    });
                    if accepted { Answer::Accept } else if rejected { Answer::Reject } else { Answer::Open }
                });
                match answer {
                    Answer::Accept if valid => {
                        self.store.set_string("naca_code", code);
                        self.store.set_int("naca_points", *per_side as i64);
                        self.store.set_bool("naca_closed_te", *closed);
                        let (code, n, c) = (code.clone(), *per_side, *closed);
                        self.new_naca_airfoil(&code, n, c);
                        false
                    }
                    Answer::Open | Answer::Accept => true,
                    Answer::Reject => false,
                }
            }
            Dialog::Cad(choice) => {
                let answer = modal(ctx, "Export CAD", 380.0, |ui| {
                    form_row(ui, &p, "Format", true, "", |ui| {
                        let labels: Vec<&str> = CAD_FORMATS.iter().map(|(l, _)| *l).collect();
                        widgets::combo(ui, "cad_format", &mut choice.format, &labels, 200.0, "");
                    });
                    form_row(ui, &p, "Units", true, "", |ui| {
                        let items: Vec<(&'static str, &str, &str)> = CAD_UNITS.iter().map(|u| (*u, *u, "")).collect();
                        widgets::segmented(ui, &mut choice.units, &items, 200.0);
                    });
                    form_row(ui, &p, "Scale", true, "", |ui| {
                        widgets::scientific_field(ui, &p, "cad_scale", &mut choice.scale, 1e-6, f64::INFINITY, 1.0, 200.0, "Coordinates are multiplied by this factor, for example the chord length.");
                    });
                    form_row(ui, &p, "Sections", true, "", |ui| {
                        ui.checkbox(&mut choice.merge, "Merge sections into one curve").on_hover_text("Write the whole chain as a single B-spline (junctions become knots).\nOtherwise every section is its own curve.");
                    });
                    form_row(ui, &p, "Trailing edge", true, "", |ui| {
                        ui.checkbox(&mut choice.close_te, "Close a blunt trailing edge with a line");
                    });
                    let (a, r, _) = buttons(ui, Some("Export..."), "Cancel", None);
                    if a { Answer::Accept } else if r { Answer::Reject } else { Answer::Open }
                });
                match answer {
                    Answer::Accept => {
                        let c = choice.clone();
                        self.export_cad(c);
                        false
                    }
                    Answer::Reject => {
                        self.set_status("Export cancelled.");
                        false
                    }
                    Answer::Open => true,
                }
            }
            Dialog::Batch => {
                let (mut source, mut target) = (self.batch_source.clone(), self.batch_target.clone());
                let answer = modal(ctx, "Batch convert", size::DIALOG_MIN_WIDTH, |ui| {
                    muted(ui, &p, "Fit every .dat file in the source folder with the current settings\nand save each as .spline.dat in the target folder.");
                    for (label, folder) in [("Source", &mut source), ("Target", &mut target)] {
                        form_row(ui, &p, label, true, "", |ui| {
                            ui.horizontal(|ui| {
                                ui.add(egui::Label::new(RichText::new(if folder.is_empty() { "Not chosen" } else { folder.as_str() }).color(p.text_3)).wrap());
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if widgets::button(ui, "Choose...", "").clicked() {
                                        let home = dirs::home_dir().unwrap_or_default();
                                        if let Some(dir) = rfd::FileDialog::new().set_title(format!("{label} folder")).set_directory(home).pick_folder() {
                                            *folder = dir.to_string_lossy().into_owned();
                                        }
                                    }
                                });
                            });
                        });
                    }
                    let (a, r, _) = buttons(ui, Some("Convert"), "Close", None);
                    if a { Answer::Accept } else if r { Answer::Reject } else { Answer::Open }
                });
                self.batch_source = source;
                self.batch_target = target;
                match answer {
                    Answer::Accept => {
                        self.start_batch();
                        false
                    }
                    Answer::Reject => false,
                    Answer::Open => true,
                }
            }
            Dialog::Theme { pending, slider } => {
                let mut mode = self.theme_mode;
                let answer = modal(ctx, "Theme", 380.0, |ui| {
                    form_row(ui, &p, "Mode", true, "", |ui| {
                        let items: Vec<(ThemeMode, &str, &str)> = ThemeMode::ALL.iter().map(|m| (*m, m.label(), if *m == ThemeMode::System { "Follow the light or dark setting of the system" } else { "" })).collect();
                        widgets::segmented(ui, &mut mode, &items, 240.0);
                    });
                    form_row(ui, &p, "Temperature", true, "", |ui| {
                        ui.label(RichText::new("Cool").color(p.text_2));
                        ui.spacing_mut().slider_width = size::SLIDER_WIDTH;
                        let r = ui.add(egui::Slider::new(slider, -100..=100).show_value(false)).on_hover_text("Tint the greys cool (blue) or warm (sand); neutral in the middle");
                        if r.dragged() && slider.abs() <= 4 {
                            *slider = 0;
                        }
                        if r.changed() {
                            *pending = Some((*slider, Instant::now()));
                        }
                        ui.label(RichText::new("Warm").color(p.text_2));
                    });
                    let (_, r, z) = buttons(ui, None, "Close", Some(("Neutral", "Untinted greys (the default)")));
                    if z {
                        *slider = 0;
                        *pending = Some((0, Instant::now() - Duration::from_millis(100)));
                    }
                    if r { Answer::Reject } else { Answer::Open }
                });
                if mode != self.theme_mode {
                    self.set_theme(Some(mode), None);
                }
                let due = pending.is_some_and(|(_, t)| t.elapsed() >= Duration::from_millis(60));
                if let (true, Some((v, _))) = (due || matches!(answer, Answer::Reject), *pending) {
                    if v != self.temperature {
                        self.set_theme(None, Some(v));
                    }
                    *pending = None;
                } else if pending.is_some() {
                    ctx.request_repaint_after(Duration::from_millis(60));
                }
                !matches!(answer, Answer::Reject)
            }
            Dialog::Motion => {
                let (mut scroll, mut zoom) = (self.scroll_ms as usize, self.zoom_ms as usize);
                let answer = modal(ctx, "Scrolling and zooming", 380.0, |ui| {
                    form_row(ui, &p, "Scrolling", true, "", |ui| {
                        widgets::int_field(ui, &p, "scroll_ms", &mut scroll, 0, 1000, 200.0, " milliseconds", Some("Off"), 25, "How long a wheel step glides in lists, tables, panels and documents; off jumps at once");
                    });
                    form_row(ui, &p, "Zooming", true, "", |ui| {
                        widgets::int_field(ui, &p, "zoom_ms", &mut zoom, 0, 1000, 200.0, " milliseconds", Some("Off"), 25, "How long a wheel step glides when zooming a plot; off jumps at once");
                    });
                    let (_, r, z) = buttons(ui, None, "Close", Some(("Default", "350 milliseconds scrolling, 650 zooming")));
                    if z {
                        scroll = DEFAULT_SCROLL_MS as usize;
                        zoom = DEFAULT_ZOOM_MS as usize;
                    }
                    if r { Answer::Reject } else { Answer::Open }
                });
                if (scroll as u64, zoom as u64) != (self.scroll_ms, self.zoom_ms) {
                    self.set_motion(scroll as u64, zoom as u64);
                }
                !matches!(answer, Answer::Reject)
            }
            Dialog::PointsPerSide { value } => {
                let answer = modal(ctx, "Save fitted airfoil", 380.0, |ui| {
                    ui.label("Points per side, spaced closer toward the leading and trailing edge");
                    widgets::int_field(ui, &p, "points_per_side", value, 3, 100_000, 160.0, "", None, 1, "");
                    let (a, r, _) = buttons(ui, Some("OK"), "Cancel", None);
                    if a { Answer::Accept } else if r { Answer::Reject } else { Answer::Open }
                });
                match answer {
                    Answer::Accept => {
                        let n = *value;
                        self.save_airfoil(n);
                        false
                    }
                    Answer::Reject => {
                        self.set_status("Save cancelled.");
                        false
                    }
                    Answer::Open => true,
                }
            }
        };
        if keep && self.dialog.is_none() {
            self.dialog = Some(dialog);
        }
    }

    fn report_window(&mut self, ctx: &egui::Context) {
        let Some(window) = self.report.as_mut() else { return };
        let p = self.palette.clone();
        let mut open = true;
        let height = if window.shape.is_some() { 720.0 } else { 520.0 };
        egui::Window::new("Fit report").open(&mut open).collapsible(false).resizable(true).default_size([560.0, height]).show(ctx, |ui| {
            egui::ScrollArea::vertical().max_height(if window.shape.is_some() { 380.0 } else { f32::INFINITY }).show(ui, |ui| {
                ui.label(RichText::new(&window.title).heading().color(p.text));
                ui.add_space(size::GROUP);
                egui::Grid::new("report_rows").num_columns(2).spacing([24.0, 4.0]).show(ui, |ui| {
                    for row in window.report.rows() {
                        ui.label(RichText::new(row.label).color(p.text_2));
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&row.value).monospace().color(p.text));
                            match row.note {
                                Some(Note::Ok(t)) => {
                                    ui.label(RichText::new(format!("({t})")).color(p.ok));
                                }
                                Some(Note::Warn(t)) => {
                                    ui.label(RichText::new(format!("({t})")).strong().color(p.warn));
                                }
                                None => {}
                            }
                        });
                        ui.end_row();
                    }
                });
                let junctions = window.report.junction_rows();
                if !junctions.is_empty() {
                    ui.add_space(size::PAD);
                    ui.label(RichText::new("Junctions").strong().color(p.text));
                    egui::Grid::new("report_junctions").num_columns(5).spacing([24.0, 4.0]).show(ui, |ui| {
                        for c in JUNCTION_COLUMNS {
                            ui.label(RichText::new(c).strong().color(p.text_3));
                        }
                        ui.end_row();
                        for row in junctions {
                            for cell in row {
                                ui.label(RichText::new(cell).monospace().color(p.text));
                            }
                            ui.end_row();
                        }
                    });
                }
            });
            if let Some(shape) = &window.shape {
                ui.add_space(size::PAD);
                ui.allocate_ui(egui::vec2(ui.available_width(), ui.available_height().max(size::SHAPE_PLOT_HEIGHT)), |ui| shape_plot(ui, &p, shape, &mut window.view));
            }
        });
        if !open {
            self.report = None;
        }
    }

    pub fn comparison_block(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                widgets::caption(ui, &p, "Runs");
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let metrics: Vec<usize> = (0..COLUMNS.len()).filter(|&c| COLUMNS[c].metric).collect();
                    let names: Vec<&str> = metrics.iter().map(|&c| COLUMNS[c].name).collect();
                    let mut y = metrics.iter().position(|&c| c == self.explorer.y_column).unwrap_or(0);
                    if widgets::combo(ui, "cmp_y", &mut y, &names, 110.0, "The fit metric plotted for each run") {
                        self.explorer.y_column = metrics[y];
                        self.explorer.forget_points();
                        self.refresh_comparison();
                    }
                    ui.label("Y axis");
                    let all: Vec<&str> = COLUMNS.iter().map(|c| c.name).collect();
                    let mut x = self.explorer.x_column;
                    if widgets::combo(ui, "cmp_x", &mut x, &all, 110.0, "The explorer column the runs are plotted along.\nOnly the runs that pass the explorer's filter are shown.") {
                        self.explorer.x_column = x;
                        self.explorer.forget_points();
                        self.refresh_comparison();
                    }
                    ui.label("X axis");
                });
            });
            self.comparison.show(ui, &p, self.zoom_ms);
        });
    }
}

/// The fit report window.
pub struct ReportWindow {
    title: String,
    report: Report,
    shape: Option<Shape>,
    view: View,
}

impl ReportWindow {
    pub fn new(title: String, report: Report, shape: Option<Shape>) -> Self {
        ReportWindow { title, report, shape, view: View::default() }
    }
}

/// A toggle with a drop-down of settings beside it.
fn split_toggle(ui: &mut Ui, p: &crate::theme::Palette, text: &str, on: &mut bool, tip: &str, arrow_tip: &str, id: &str, content: impl FnOnce(&mut Ui) -> bool) -> (bool, bool) {
    let mut toggled = false;
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        let r = ui.add(egui::Button::new(text).selected(*on).min_size(egui::vec2(0.0, size::CONTROL_HEIGHT))).on_hover_text(tip);
        if r.clicked() {
            *on = !*on;
            toggled = true;
        }
        let arrow = ui.add(egui::Button::new("").min_size(egui::vec2(18.0, size::CONTROL_HEIGHT))).on_hover_text(arrow_tip);
        Glyph::ChevronDown.paint(ui.painter(), arrow.rect.center(), size::GLYPH, p.text_2);
        egui::Popup::from_toggle_button_response(&arrow)
            .id(egui::Id::unique(id))
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                egui::Frame::new().inner_margin(egui::Margin::same(12)).show(ui, |ui| {
                    changed = content(ui);
                });
            });
    });
    (toggled, changed)
}

/// The Curvature comb toggle and its settings.
pub fn comb_toggle(app: &mut App, ui: &mut Ui) {
    let p = app.palette.clone();
    let plot = &mut app.main_plot;
    let mut settings = plot.comb;
    let (toggled, changed) = split_toggle(
        ui,
        &p,
        "Curvature comb",
        &mut plot.comb_on,
        "Display curvature as perpendicular lines from the curve,\nblue and pink on alternate spline sections.\nUse the arrow for quantity, height and density.",
        "Curvature comb settings",
        "comb_settings",
        |ui| {
            let mut changed = false;
            form_row(ui, &p, "Quantity", true, "", |ui| changed |= widgets::combo(ui, "comb_quantity", &mut settings.quantity, &QUANTITIES, 180.0, ""));
            form_row(ui, &p, "Height", true, "", |ui| {
                let mut h = settings.height;
                let fmt = |v: f64| spline_fitter::formatting::number(v, ".4f");
                let parse = |t: &str| widgets::parse_scientific(t);
                let step = |v: f64, k: i32| v + 0.01 * k as f64;
                let f = widgets::NumberFormat { format: &fmt, parse: &parse, step: &step };
                changed |= widgets::number_field(ui, &p, "comb_height", &mut h, (0.0, 1000.0), 120.0, &f, "Length of a tooth, in chords, where the quantity is 1 on the airfoil\nscaled to a unit chord. The comb looks the same at any size of the data.");
                settings.height = h;
            });
            form_row(ui, &p, "Density", true, "", |ui| changed |= widgets::int_field(ui, &p, "comb_density", &mut settings.density, 5, 2000, 120.0, "", None, 1, ""));
            changed
        },
    );
    plot.comb = settings;
    if toggled || (changed && plot.comb_on) {
        plot.refresh_comb();
    }
}

/// The Deviation toggle and its height.
pub fn deviation_toggle(app: &mut App, ui: &mut Ui) {
    let p = app.palette.clone();
    let plot = &mut app.main_plot;
    let mut height = plot.deviation_height;
    let (toggled, changed) = split_toggle(
        ui,
        &p,
        "Deviation",
        &mut plot.deviation_on,
        "Display the distance of every data point from the curve, magnified\nalong the normal, outward for a point outside, in the color of its\nnearest spline section (as the curvature comb). The dashed lines\nare the tolerance. Use the arrow for the height.",
        "Deviation settings",
        "deviation_settings",
        |ui| {
            let mut changed = false;
            form_row(ui, &p, "Height", true, "", |ui| {
                let fmt = |v: f64| spline_fitter::formatting::number(v, ".3f");
                let parse = |t: &str| widgets::parse_scientific(t);
                let step = |v: f64, k: i32| v + 0.01 * k as f64;
                let f = widgets::NumberFormat { format: &fmt, parse: &parse, step: &step };
                changed |= widgets::number_field(ui, &p, "deviation_height", &mut height, (0.001, 1.0), 120.0, &f, "How far from the curve the tolerance is drawn, as a share of the chord\n(without a tolerance, the largest deviation).");
            });
            changed
        },
    );
    plot.deviation_height = height;
    if toggled || changed {
        plot.refresh_deviation();
    }
}
