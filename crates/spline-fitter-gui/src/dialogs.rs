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
use crate::theme::{size, Glyph, Palette, ThemeMode};
use crate::widgets::{self, form_row, muted, Field, IntOptions};

/// Motion durations: steps of 25, 0 shown as Off.
const MILLISECONDS: IntOptions = IntOptions { suffix: " milliseconds", zero: Some("Off"), step: 25 };

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

/// Padding of a settings dialog, and of a Qt input dialog.
const DIALOG_MARGIN: f32 = size::PAD_LARGE;
const INPUT_DIALOG_MARGIN: f32 = 11.0;

/// The report window: as wide as the Python version's, taller with a chart.
const REPORT_WIDTH: f32 = 560.0;

/// A dialog in a window of its own, `width` wide; its height follows the
/// content. Where the platform has a single window (as under test) it is
/// a window inside the main one. The content returns the answer; closing
/// the window or Escape rejects.
fn open_dialog(ctx: &egui::Context, title: &str, width: f32, margin: f32, content: impl FnOnce(&mut Ui) -> Answer) -> Answer {
    let height_id = egui::Id::unique(("dialog_height", title));
    let height: f32 = ctx.data(|d| d.get_temp(height_id)).unwrap_or(120.0);
    let builder = egui::ViewportBuilder::default()
        .with_title(title)
        .with_inner_size([width, height])
        .with_resizable(false)
        .with_minimize_button(false)
        .with_maximize_button(false);
    let mut content = Some(content);
    ctx.show_viewport_immediate(egui::ViewportId::from_hash_of(("dialog", title)), builder, |ui, class| {
        let mut answer = Answer::Open;
        // An embedded window brings its own margin.
        let embedded = class == egui::ViewportClass::EmbeddedWindow;
        let inner = if embedded { 0 } else { margin as i8 };
        let frame = egui::Frame::new().fill(ui.visuals().window_fill).inner_margin(egui::Margin::same(inner));
        let used = frame
            .show(ui, |ui| {
                ui.set_width(width - 2.0 * margin);
                ui.spacing_mut().item_spacing.y = size::GAP;
                focus_first_control(ui, title);
                if let Some(content) = content.take() {
                    answer = content(ui);
                }
                ui.min_rect().height()
            })
            .inner;
        if class != egui::ViewportClass::EmbeddedWindow {
            let wanted = (used + 2.0 * margin).ceil();
            if (wanted - height).abs() > 0.5 {
                ui.ctx().data_mut(|d| d.insert_temp(height_id, wanted));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(width, wanted)));
            }
            if ui.input(|i| i.viewport().close_requested()) {
                answer = Answer::Reject;
            }
        }
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            answer = Answer::Reject;
        }
        if !matches!(answer, Answer::Open) {
            // Opened again, it focuses its first control again.
            ui.ctx().data_mut(|d| d.remove::<u8>(opened_id(title)));
        }
        answer
    })
}

/// Where a dialog keeps how many frames it has been open (up to three).
fn opened_id(title: &str) -> egui::Id {
    egui::Id::unique(("dialog_opened", title))
}

/// Qt gives a dialog's first control the focus when it opens: an anchor
/// before the content takes the focus on the first frame and passes it on
/// to the next control on the second.
fn focus_first_control(ui: &mut Ui, title: &str) {
    let id = opened_id(title);
    let frames = ui.ctx().data(|d| d.get_temp::<u8>(id)).unwrap_or(0);
    if frames >= 3 {
        return;
    }
    // The move is asked for before the anchor registers: the widget after
    // it then takes the focus.
    let anchor_id = id.with("anchor");
    if frames == 1 && ui.memory(|m| m.has_focus(anchor_id)) {
        ui.memory_mut(|m| m.move_focus(egui::FocusDirection::Next));
    }
    let anchor = egui::Rect::from_min_size(ui.cursor().min, egui::Vec2::ZERO);
    let anchor = ui.interact(anchor, anchor_id, egui::Sense::focusable_noninteractive());
    if frames == 0 {
        anchor.request_focus();
    }
    ui.ctx().data_mut(|d| d.insert_temp(id, frames + 1));
    ui.ctx().request_repaint();
}

/// A dialog's push button, 22 px wider than its text, as Qt's auto-default
/// buttons: the one with the focus is the default; without a push button
/// in focus, the `default` one is. The default is outlined in the selection
/// line and Enter presses it. True when pressed.
fn dialog_button(ui: &mut Ui, p: &Palette, text: &str, default: bool, enabled: bool) -> bool {
    let response = ui.add_enabled(enabled, egui::Button::new(text).min_size(egui::vec2(0.0, size::CONTROL_HEIGHT)));
    let last_focused = egui::Id::unique("dialog_focused_button");
    let focused = response.has_focus();
    if focused {
        ui.ctx().data_mut(|d| d.insert_temp(last_focused, response.id));
    }
    let other = ui.memory(|m| m.focused()).is_some_and(|f| f != response.id && ui.ctx().data(|d| d.get_temp::<egui::Id>(last_focused)) == Some(f));
    let is_default = enabled && (focused || (default && !other));
    if is_default {
        widgets::outline_checked(ui, p, &response, egui::CornerRadius::same(size::RADIUS));
    }
    // A focused button answers Enter itself (as a click).
    response.clicked() || (is_default && !focused && ui.input(|i| i.key_pressed(egui::Key::Enter)))
}

/// The button row: a reset-role button at the left; at the right the
/// accept button (the default) and the reject button, 6 px apart. Tab
/// visits them in that order, as in Qt.
fn buttons(ui: &mut Ui, p: &Palette, accept: Option<(&str, bool)>, reject: &str, reset: Option<(&str, &str)>) -> (bool, bool, bool) {
    let (mut a, mut r, mut z) = (false, false, false);
    let row = egui::vec2(ui.available_width(), size::CONTROL_HEIGHT);
    ui.allocate_ui_with_layout(row, Layout::left_to_right(Align::Center), |ui| {
        if let Some((text, tip)) = reset {
            ui.scope(|ui| {
                z = dialog_button(ui, p, text, false, true);
            })
            .response
            .on_hover_text(tip);
        }
        // The right-hand buttons laid out left to right after the space they leave.
        let font = egui::FontId::proportional(crate::theme::typeface::BASE);
        let width = |t: &str| ui.painter().layout_no_wrap(t.to_string(), font.clone(), p.text).size().x + 2.0 * ui.spacing().button_padding.x;
        let gap = ui.spacing().item_spacing.x;
        let right = accept.map_or(0.0, |(t, _)| width(t) + gap) + width(reject);
        ui.add_space((ui.available_width() - right).max(0.0));
        if let Some((text, enabled)) = accept {
            a = dialog_button(ui, p, text, true, enabled);
        }
        r = dialog_button(ui, p, reject, false, true);
    });
    (a, r, z)
}

/// The answer of a button row.
fn answer_of(accepted: bool, rejected: bool) -> Answer {
    if accepted {
        Answer::Accept
    } else if rejected {
        Answer::Reject
    } else {
        Answer::Open
    }
}

impl App {
    /// Whether a dialog or the report is open (the main window then takes
    /// no input, as behind a modal dialog).
    pub fn modal_open(&self) -> bool {
        self.dialog.is_some() || self.report.is_some()
    }

    pub fn show_dialogs(&mut self, ctx: &egui::Context) {
        self.report_window(ctx);
        let Some(mut dialog) = self.dialog.take() else { return };
        let p = self.palette.clone();
        let form = |width: f32| width - 2.0 * DIALOG_MARGIN - size::LABEL_WIDTH - size::GROUP;
        let keep = match &mut dialog {
            Dialog::Naca { code, per_side, closed } => {
                let valid = code.len() == 4 && code.chars().all(|c| c.is_ascii_digit());
                const WIDTH: f32 = 250.0;
                let answer = open_dialog(ctx, "New NACA airfoil", WIDTH, DIALOG_MARGIN, |ui| {
                    form_row(ui, &p, "Digits", true, "", |ui| {
                        let tip = "Four digits: the camber in percent of the chord, its position in tenths\nof the chord, and the thickness in percent. 2412 has 2% camber at 40%\nof the chord and is 12% thick; 0012 is symmetric.";
                        widgets::text_field(ui, &p, "naca_code", code, form(WIDTH), "", 4).on_hover_text(tip);
                    });
                    form_row(ui, &p, "Points per side", true, "", |ui| {
                        let f = Field::new(form(WIDTH), "Spaced closer toward the leading and trailing edge");
                        widgets::int_field(ui, &p, "naca_points", per_side, (10, crate::settings::UNLIMITED), f, IntOptions::default());
                    });
                    form_row(ui, &p, "Trailing edge", true, "", |ui| {
                        let items = [
                            (true, "Closed", "Modified thickness coefficient: a sharp trailing edge"),
                            (false, "Open", "Original thickness coefficient: a thin blunt trailing edge"),
                        ];
                        widgets::segmented(ui, closed, &items, form(WIDTH));
                    });
                    let (a, r, _) = buttons(ui, &p, Some(("Create", valid)), "Cancel", None);
                    answer_of(a, r)
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
                const WIDTH: f32 = 349.0;
                let answer = open_dialog(ctx, "Export CAD", WIDTH, DIALOG_MARGIN, |ui| {
                    form_row(ui, &p, "Format", true, "", |ui| {
                        let labels: Vec<&str> = CAD_FORMATS.iter().map(|(l, _)| *l).collect();
                        widgets::combo(ui, &p, "cad_format", &mut choice.format, &labels, 0.0, "");
                    });
                    form_row(ui, &p, "Units", true, "", |ui| {
                        let items: Vec<(&'static str, &str, &str)> = CAD_UNITS.iter().map(|u| (*u, *u, "")).collect();
                        widgets::segmented(ui, &mut choice.units, &items, form(WIDTH));
                    });
                    form_row(ui, &p, "Scale", true, "", |ui| {
                        let f = Field::new(80.0, "Coordinates are multiplied by this factor, for example the chord length.");
                        widgets::scientific_field(ui, &p, "cad_scale", &mut choice.scale, (1e-6, f64::INFINITY), 1.0, f);
                    });
                    widgets::check_row(ui, &p, "Sections", true, |ui| {
                        widgets::checkbox(ui, &p, &mut choice.merge, "Merge sections into one curve").on_hover_text(
                            "Write the whole chain as a single B-spline (junctions become knots).\nOtherwise every section is its own curve.",
                        );
                    });
                    widgets::check_row(ui, &p, "Trailing edge", true, |ui| {
                        widgets::checkbox(ui, &p, &mut choice.close_te, "Close a blunt trailing edge with a line");
                    });
                    let (a, r, _) = buttons(ui, &p, Some(("Export...", true)), "Cancel", None);
                    answer_of(a, r)
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
                let answer = open_dialog(ctx, "Batch convert", size::DIALOG_MIN_WIDTH, DIALOG_MARGIN, |ui| {
                    muted(ui, &p, "Fit every .dat file in the source folder with the current settings\nand save each as .spline.dat in the target folder.");
                    for (label, folder) in [("Source", &mut source), ("Target", &mut target)] {
                        form_row(ui, &p, label, true, "", |ui| {
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if dialog_button(ui, &p, "Choose...", false, true) {
                                    let home = dirs::home_dir().unwrap_or_default();
                                    if let Some(dir) =
                                        rfd::FileDialog::new().set_title(format!("{label} folder")).set_directory(home).pick_folder()
                                    {
                                        *folder = dir.to_string_lossy().into_owned();
                                    }
                                }
                                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                                    let text = if folder.is_empty() { "Not chosen" } else { folder.as_str() };
                                    ui.add(egui::Label::new(RichText::new(text).color(p.text_3)).truncate());
                                });
                            });
                        });
                    }
                    let (a, r, _) = buttons(ui, &p, Some(("Convert", true)), "Close", None);
                    answer_of(a, r)
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
                const WIDTH: f32 = 418.0;
                let answer = open_dialog(ctx, "Theme", WIDTH, DIALOG_MARGIN, |ui| {
                    form_row(ui, &p, "Mode", true, "", |ui| {
                        let items: Vec<(ThemeMode, &str, &str)> = ThemeMode::ALL
                            .iter()
                            .map(|m| {
                                (*m, m.label(), if *m == ThemeMode::System { "Follow the light or dark setting of the system" } else { "" })
                            })
                            .collect();
                        widgets::segmented(ui, &mut mode, &items, form(WIDTH));
                    });
                    form_row(ui, &p, "Temperature", true, "", |ui| {
                        ui.label(RichText::new("Cool").color(p.text_2));
                        let r = widgets::temperature_slider(ui, &p, slider, size::SLIDER_WIDTH)
                            .on_hover_text("Tint the greys cool (blue) or warm (sand); neutral in the middle");
                        if r.dragged() && slider.abs() <= 4 {
                            *slider = 0;
                        }
                        if r.changed() {
                            *pending = Some((*slider, Instant::now()));
                        }
                        ui.label(RichText::new("Warm").color(p.text_2));
                    });
                    let (_, r, z) = buttons(ui, &p, None, "Close", Some(("Neutral", "Untinted greys (the default)")));
                    if z {
                        *slider = 0;
                        *pending = Some((0, Instant::now() - Duration::from_millis(100)));
                    }
                    answer_of(false, r)
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
                const WIDTH: f32 = 299.0;
                let answer = open_dialog(ctx, "Scrolling and zooming", WIDTH, DIALOG_MARGIN, |ui| {
                    form_row(ui, &p, "Scrolling", true, "", |ui| {
                        let f = Field::new(
                            form(WIDTH),
                            "How long a wheel step glides in lists, tables, panels and documents; off jumps at once",
                        );
                        widgets::int_field(ui, &p, "scroll_ms", &mut scroll, (0, 1000), f, MILLISECONDS);
                    });
                    form_row(ui, &p, "Zooming", true, "", |ui| {
                        let f = Field::new(form(WIDTH), "How long a wheel step glides when zooming a plot; off jumps at once");
                        widgets::int_field(ui, &p, "zoom_ms", &mut zoom, (0, 1000), f, MILLISECONDS);
                    });
                    let (_, r, z) = buttons(ui, &p, None, "Close", Some(("Default", "350 milliseconds scrolling, 650 zooming")));
                    if z {
                        scroll = DEFAULT_SCROLL_MS as usize;
                        zoom = DEFAULT_ZOOM_MS as usize;
                    }
                    answer_of(false, r)
                });
                if (scroll as u64, zoom as u64) != (self.scroll_ms, self.zoom_ms) {
                    self.set_motion(scroll as u64, zoom as u64);
                }
                !matches!(answer, Answer::Reject)
            }
            Dialog::PointsPerSide { value } => {
                const WIDTH: f32 = 370.0;
                let answer = open_dialog(ctx, "Save fitted airfoil", WIDTH, INPUT_DIALOG_MARGIN, |ui| {
                    let text = RichText::new("Points per side, spaced closer toward the leading and trailing edge").color(p.text);
                    ui.add(egui::Label::new(text).extend());
                    let f = Field::new(WIDTH - 2.0 * INPUT_DIALOG_MARGIN, "");
                    widgets::int_field(ui, &p, "points_per_side", value, (3, 100_000), f, IntOptions::default());
                    let (a, r, _) = buttons(ui, &p, Some(("OK", true)), "Cancel", None);
                    answer_of(a, r)
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

    /// The fit report: a window of its own with the report as a document
    /// above the thickness and camber chart, and a Close button.
    fn report_window(&mut self, ctx: &egui::Context) {
        let Some(window) = self.report.as_mut() else { return };
        let p = self.palette.clone();
        let height = if window.shape.is_some() { 720.0 } else { 520.0 };
        let builder =
            egui::ViewportBuilder::default().with_title("Fit report").with_inner_size([REPORT_WIDTH, height]).with_minimize_button(false);
        let mut close = false;
        ctx.show_viewport_immediate(egui::ViewportId::from_hash_of("fit_report"), builder, |ui, class| {
            let frame = egui::Frame::new().fill(ui.visuals().window_fill).inner_margin(egui::Margin {
                left: 0,
                right: 0,
                top: 0,
                bottom: size::PAD_LARGE as i8,
            });
            frame.show(ui, |ui| {
                if class == egui::ViewportClass::EmbeddedWindow {
                    ui.set_width(REPORT_WIDTH);
                    ui.set_height(height - size::PAD_LARGE);
                }
                let pad = size::PAD_LARGE;
                let total = ui.available_height();
                // The document and the chart share the height above the buttons.
                let below = size::CONTROL_HEIGHT + pad;
                let parts = if window.shape.is_some() { 2.0 } else { 1.0 };
                let each = ((total - below - pad * (parts - 1.0)) / parts).max(80.0);
                let width = ui.available_width();
                let (rect, _) = ui.allocate_exact_size(egui::vec2(width, each), egui::Sense::hover());
                report_document(ui, &p, rect, &window.title, &window.report);
                let mut top = rect.bottom() + pad;
                if let Some(shape) = &window.shape {
                    let chart = egui::Rect::from_min_size(
                        egui::pos2(rect.left() + pad, top),
                        egui::vec2(width - 2.0 * pad, each.max(size::SHAPE_PLOT_HEIGHT)),
                    );
                    ui.allocate_rect(chart, egui::Sense::hover());
                    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(chart));
                    shape_plot(&mut child, &p, shape, &mut window.view);
                    top = chart.bottom() + pad;
                }
                let row =
                    egui::Rect::from_min_size(egui::pos2(rect.left() + pad, top), egui::vec2(width - 2.0 * pad, size::CONTROL_HEIGHT));
                ui.allocate_rect(row, egui::Sense::hover());
                let mut child = ui.new_child(egui::UiBuilder::new().max_rect(row).layout(Layout::right_to_left(Align::Center)));
                // The only button: the default (Enter closes).
                close |= dialog_button(&mut child, &p, "Close", true, true);
            });
            if class != egui::ViewportClass::EmbeddedWindow {
                close |= ui.input(|i| i.viewport().close_requested());
            }
            close |= ui.input(|i| i.key_pressed(egui::Key::Escape));
        });
        if close {
            self.report = None;
        }
    }

    pub fn comparison_block(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                widgets::caption(ui, &p, "Runs");
                // The two combos share the width left after the labels (and
                // the three gaps between the four).
                let font = egui::FontId::proportional(crate::theme::typeface::BASE);
                let labels: f32 = ["X axis", "Y axis"].iter().map(|t| crate::fonts::layout_width(ui, t, &font, false)).sum();
                let each = ((ui.available_width() - labels - 3.0 * ui.spacing().item_spacing.x) / 2.0).max(60.0);
                ui.label(egui::RichText::new("X axis").color(p.text_2));
                let all: Vec<&str> = COLUMNS.iter().map(|c| c.name).collect();
                let mut x = self.explorer.x_column;
                let tip = "The explorer column the runs are plotted along.\nOnly the runs that pass the explorer's filter are shown.";
                if widgets::combo(ui, &p, "cmp_x", &mut x, &all, each, tip) {
                    self.explorer.x_column = x;
                    self.explorer.forget_points();
                    self.refresh_comparison();
                }
                ui.label(egui::RichText::new("Y axis").color(p.text_2));
                let metrics: Vec<usize> = (0..COLUMNS.len()).filter(|&c| COLUMNS[c].metric).collect();
                let names: Vec<&str> = metrics.iter().map(|&c| COLUMNS[c].name).collect();
                let mut y = metrics.iter().position(|&c| c == self.explorer.y_column).unwrap_or(0);
                if widgets::combo(ui, &p, "cmp_y", &mut y, &names, each, "The fit metric plotted for each run") {
                    self.explorer.y_column = metrics[y];
                    self.explorer.forget_points();
                    self.refresh_comparison();
                }
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

/// The report as the Python version's text browser showed it: a framed,
/// scrolling document with a heading, a table of label and value rows and
/// the junction table.
fn report_document(ui: &mut Ui, p: &Palette, rect: egui::Rect, title: &str, report: &Report) {
    const ROW: f32 = 31.0;
    let radius = egui::CornerRadius::same(size::RADIUS);
    ui.painter().rect_filled(rect, radius, p.panel);
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink(1.0)));
    crate::theme::scroll_handles(&mut child, p);
    let body = egui::FontId::proportional(crate::theme::typeface::BASE);
    let mono = egui::FontId::monospace(crate::theme::typeface::BASE);
    let painter = child.painter().clone();
    let width = |text: &str, font: &egui::FontId| painter.layout_no_wrap(text.to_string(), font.clone(), p.text).size().x;
    let rows = report.rows();
    let label_w = rows.iter().map(|r| width(r.label, &body)).fold(0.0, f32::max);
    egui::ScrollArea::vertical().id_salt("report_document").auto_shrink([false, false]).show(&mut child, |ui| {
        egui::Frame::new().inner_margin(egui::Margin::same(size::PAD_LARGE as i8)).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            ui.label(RichText::new(title).font(crate::fonts::semibold_id(crate::theme::typeface::HEADING)).color(p.text));
            ui.add_space(size::PAD);
            for row in &rows {
                ui.allocate_ui_with_layout(egui::vec2(ui.available_width(), ROW), Layout::left_to_right(Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.add_space(3.0);
                    let (cell, _) = ui.allocate_exact_size(egui::vec2(label_w + 14.0, ROW), egui::Sense::hover());
                    ui.painter().text(cell.left_center(), egui::Align2::LEFT_CENTER, row.label, body.clone(), p.text);
                    ui.label(RichText::new(&row.value).font(mono.clone()).color(p.text));
                    match row.note {
                        Some(Note::Ok(t)) => {
                            ui.label(RichText::new(format!(" ({t})")).font(mono.clone()).color(p.ok));
                        }
                        Some(Note::Warn(t)) => {
                            ui.label(
                                RichText::new(format!(" ({t})"))
                                    .font(crate::fonts::semibold_id(crate::theme::typeface::BASE))
                                    .color(p.warn),
                            );
                        }
                        None => {}
                    }
                });
            }
            let junctions = report.junction_rows();
            if !junctions.is_empty() {
                ui.add_space(size::PAD);
                ui.label(RichText::new("Junctions").font(crate::fonts::semibold_id(crate::theme::typeface::BASE)).color(p.text));
                ui.add_space(size::GAP);
                let widths: Vec<f32> = (0..5)
                    .map(|c| {
                        let head = width(JUNCTION_COLUMNS[c], &body);
                        junctions.iter().map(|r| width(&r[c], &mono)).fold(head, f32::max) + 12.0
                    })
                    .collect();
                let line = |ui: &mut Ui, cells: &[String], header: bool| {
                    let (r, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW), egui::Sense::hover());
                    let mut x = r.left() + 3.0;
                    for (c, text) in cells.iter().enumerate() {
                        let (font, color) = if header {
                            (crate::fonts::semibold_id(crate::theme::typeface::BASE), p.text_3)
                        } else {
                            (mono.clone(), p.text)
                        };
                        ui.painter().text(egui::pos2(x, r.center().y), egui::Align2::LEFT_CENTER, text, font, color);
                        x += widths[c];
                    }
                    if header {
                        let right = r.left() + 3.0 + widths.iter().sum::<f32>();
                        ui.painter().hline(r.left()..=right, r.bottom() - 4.0, egui::Stroke::new(1.0, p.rule));
                    }
                };
                line(ui, &JUNCTION_COLUMNS.iter().map(|s| s.to_string()).collect::<Vec<_>>(), true);
                for row in &junctions {
                    line(ui, row, false);
                }
            }
        });
    });
    ui.painter().rect_stroke(rect, radius, egui::Stroke::new(1.0, p.rule), egui::StrokeKind::Inside);
}

/// The texts of a toggle with settings.
struct SplitToggle<'a> {
    text: &'a str,
    tooltip: &'a str,
    arrow_tooltip: &'a str,
    id: &'a str,
}

/// A toggle with a drop-down of settings beside it; returns whether it was
/// toggled and whether a setting changed.
fn split_toggle(
    ui: &mut Ui,
    p: &crate::theme::Palette,
    toggle: SplitToggle,
    on: &mut bool,
    content: impl FnOnce(&mut Ui) -> bool,
) -> (bool, bool) {
    let SplitToggle { text, tooltip: tip, arrow_tooltip: arrow_tip, id } = toggle;
    let mut toggled = false;
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        let left = egui::CornerRadius { nw: size::RADIUS, sw: size::RADIUS, ne: 0, se: 0 };
        ui.spacing_mut().button_padding.x = 0.0;
        let width = widgets::tool_button_width(ui, text);
        let r = ui
            .add(egui::Button::new(text).selected(*on).corner_radius(left).min_size(egui::vec2(width, size::CONTROL_HEIGHT)))
            .on_hover_text(tip);
        if *on {
            widgets::outline_checked(ui, p, &r, left);
        }
        if r.clicked() {
            *on = !*on;
            toggled = true;
        }
        let right = egui::CornerRadius { nw: 0, sw: 0, ne: size::RADIUS, se: size::RADIUS };
        ui.spacing_mut().button_padding.x = 0.0;
        // The drop-down stays open while the rest of the window is used; its
        // arrow (checked while it is open) or Escape closes it.
        let popup_id = egui::Id::unique(id);
        let open = egui::Popup::is_id_open(ui.ctx(), popup_id);
        let arrow = ui
            .add(
                egui::Button::new("")
                    .selected(open)
                    .corner_radius(right)
                    .min_size(egui::vec2(size::ARROW_BUTTON_WIDTH, size::CONTROL_HEIGHT)),
            )
            .on_hover_text(arrow_tip);
        if open {
            widgets::outline_checked(ui, p, &arrow, right);
        }
        Glyph::ChevronDown.paint(ui.painter(), arrow.rect.center(), size::GLYPH, if open { p.text } else { p.text_2 });
        let frame = egui::Frame::popup(ui.style()).inner_margin(0);
        egui::Popup::from_toggle_button_response(&arrow).id(popup_id).frame(frame).close_behavior(egui::PopupCloseBehavior::IgnoreClicks).show(|ui| {
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
    let toggle = SplitToggle {
        text: "Curvature comb",
        tooltip: "Display curvature as perpendicular lines from the curve,\nblue and pink on alternate spline sections.\nUse the arrow for quantity, height and density.",
        arrow_tooltip: "Curvature comb settings",
        id: "comb_settings",
    };
    let (toggled, changed) = split_toggle(ui, &p, toggle, &mut plot.comb_on, |ui| {
        let mut changed = false;
        form_row(ui, &p, "Quantity", true, "", |ui| {
            changed |= widgets::combo(ui, &p, "comb_quantity", &mut settings.quantity, &QUANTITIES, 0.0, "")
        });
        form_row(ui, &p, "Height", true, "", |ui| {
            let mut h = settings.height;
            let fmt = |v: f64| spline_fitter::formatting::number(v, ".4f");
            let parse = |t: &str| widgets::parse_scientific(t);
            let step = |v: f64, k: i32| v + 0.01 * k as f64;
            let f = widgets::NumberFormat { format: &fmt, parse: &parse, step: &step };
            changed |= widgets::number_field(ui, &p, "comb_height", &mut h, (0.0, 1000.0), Field::new(widgets::spin_width(ui, &[&fmt(0.0), &fmt(1000.0)]), "Length of a tooth, in chords, where the quantity is 1 on the airfoil\nscaled to a unit chord. The comb looks the same at any size of the data."), &f);
            settings.height = h;
        });
        form_row(ui, &p, "Density", true, "", |ui| {
            changed |=
                widgets::int_field(ui, &p, "comb_density", &mut settings.density, (5, 2000), Field::new(widgets::spin_width(ui, &["5", "2000"]), ""), IntOptions::default())
        });
        changed
    });
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
    let toggle = SplitToggle {
        text: "Deviation",
        tooltip: "Display the distance of every data point from the curve, magnified\nalong the normal, outward for a point outside, in the color of its\nnearest spline section (as the curvature comb). The dashed lines\nare the tolerance. Use the arrow for the height.",
        arrow_tooltip: "Deviation settings",
        id: "deviation_settings",
    };
    let (toggled, changed) = split_toggle(ui, &p, toggle, &mut plot.deviation_on, |ui| {
        let mut changed = false;
        form_row(ui, &p, "Height", true, "", |ui| {
            let fmt = |v: f64| spline_fitter::formatting::number(v, ".3f");
            let parse = |t: &str| widgets::parse_scientific(t);
            let step = |v: f64, k: i32| v + 0.01 * k as f64;
            let f = widgets::NumberFormat { format: &fmt, parse: &parse, step: &step };
            changed |= widgets::number_field(
                ui,
                &p,
                "deviation_height",
                &mut height,
                (0.001, 1.0),
                Field::new(
                    widgets::spin_width(ui, &[&fmt(0.001), &fmt(1.0)]),
                    "How far from the curve the tolerance is drawn, as a share of the chord\n(without a tolerance, the largest deviation).",
                ),
                &f,
            );
        });
        changed
    });
    plot.deviation_height = height;
    if toggled || changed {
        plot.refresh_deviation();
    }
}
