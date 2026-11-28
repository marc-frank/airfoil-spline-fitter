//! Dialogs, the fit report window, and the plot toolbar's drop-downs.

use std::time::Duration;
use web_time::Instant;

use eframe::egui::{self, Align, Layout, RichText, Ui};
use spline_fitter::analysis::{Note, Report, JUNCTION_COLUMNS};
use spline_fitter::bspline_varpro::airfoil::Shape;
use spline_fitter::formatting::number;
use spline_fitter::io::{Finding, ParsedAirfoil};

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

/// A question before an action: the accept button does it (after saving
/// the project, for `save_first`), `other` does it without saving, Cancel
/// or closing leaves things as they are with the status `cancelled`.
pub struct Confirm {
    pub title: String,
    pub message: String,
    pub accept: String,
    pub other: Option<String>,
    pub save_first: bool,
    /// Enter answers with the accept button; else with Cancel (replacing
    /// a file, as Qt's question defaulted to No).
    pub default_accept: bool,
    pub cancelled: String,
    pub then: Box<dyn FnOnce(&mut App)>,
}

/// The mistakes the reader found in an airfoil file, before the airfoil is
/// loaded: a check box per mistake chooses which corrections to make (all
/// by default), a preview shows the points they remove, Load loads the
/// airfoil with them and Cancel leaves the window as it was.
pub struct ReadReport {
    /// The file (or "The pasted airfoil") the mistakes are in.
    pub source: String,
    pub parsed: ParsedAirfoil,
    /// One per finding: make its correction.
    pub accept: Vec<bool>,
    /// The finding the preview is zoomed to.
    pub selected: Option<usize>,
    pub view: View,
    /// What `App::set_airfoil` takes besides.
    pub filename: String,
    pub name: String,
    pub path: String,
    pub verb: String,
    /// Adjusts the precision read (a pasted text without decimals gets 6).
    pub precision: fn(i32) -> i32,
    /// The check boxes of the last frame: Enter on one loads, as anywhere
    /// else in the window, instead of toggling it (Space toggles).
    checks: Vec<egui::Id>,
}

impl ReadReport {
    pub fn new(parsed: ParsedAirfoil, source: &str, filename: &str, name: &str, path: &str, verb: &str, precision: fn(i32) -> i32) -> Self {
        ReadReport {
            source: source.to_string(),
            accept: vec![true; parsed.cleanup.findings.len()],
            parsed,
            selected: None,
            view: View::default(),
            filename: filename.to_string(),
            name: name.to_string(),
            path: path.to_string(),
            verb: verb.to_string(),
            precision,
            checks: Vec::new(),
        }
    }

    pub fn findings(&self) -> &[Finding] {
        &self.parsed.cleanup.findings
    }

    /// Zooms the preview to finding `i`: its points and the neighbours
    /// within a tenth of the airfoil, at least 2 % of the airfoil across.
    /// For the selected one, back to the whole airfoil.
    pub fn toggle_selected(&mut self, i: usize) {
        self.selected = if self.selected == Some(i) { None } else { Some(i) };
        let points = &self.parsed.read_points;
        let Some(i) = self.selected else {
            self.view.follow = true;
            return;
        };
        let xy = |k: usize| [points[k].x, points[k].y];
        let Some(all) = crate::plots::bounds_of(points.iter().map(|q| [q.x, q.y])) else { return };
        let span = all.width().max(all.height());
        let f = &self.parsed.cleanup.findings[i];
        let own = crate::plots::bounds_of(f.removes.iter().map(|&k| xy(k))).expect("a finding removes points");
        let centre = own.center();
        let near = |k: usize| ((points[k].x - centre.x).hypot(points[k].y - centre.y)) <= 0.1 * span;
        let (first, last) = (f.removes[0], *f.removes.last().unwrap());
        let neighbours = (first.saturating_sub(3)..=(last + 3).min(points.len() - 1)).filter(|&k| near(k));
        let b = crate::plots::bounds_of(f.removes.iter().copied().chain(neighbours).map(xy)).expect("not empty");
        let c = b.center();
        let (w, h) = (b.width().max(0.02 * span) / 2.0, b.height().max(0.02 * span) / 2.0);
        self.view.fit(Some(egui_plot::PlotBounds::from_min_max([c.x - w, c.y - h], [c.x + w, c.y + h])));
        self.view.follow = false;
    }

    /// The status verb for the corrections chosen.
    fn verb(&self) -> String {
        let (made, all) = (self.accept.iter().filter(|a| **a).count(), self.accept.len());
        match made {
            0 => format!("{} as read", self.verb),
            m if m == all => format!("{} corrected", self.verb),
            m => format!("{} with {m} of {all} corrections", self.verb),
        }
    }
}

/// An open dialog.
pub enum Dialog {
    Confirm(Confirm),
    ReadReport(ReadReport),
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

/// A dialog in a window of its own, `width` wide; its height follows the
/// content. It opens centred on the main window and stays in front of it:
/// the main window takes no input behind it, so a click there brings the
/// dialog back to the front. Where the platform has a single window (as
/// under test) it is a window inside the main one. The content returns the
/// answer; closing the window or Escape rejects.
fn open_dialog(ctx: &egui::Context, title: &str, width: f32, margin: f32, content: impl FnOnce(&mut Ui) -> Answer) -> Answer {
    let height_id = egui::Id::unique(("dialog_height", title));
    let height: f32 = ctx.data(|d| d.get_temp(height_id)).unwrap_or(120.0);
    let (main_inner, main_outer, main_focused) =
        ctx.input(|i| (i.viewport().inner_rect, i.viewport().outer_rect, i.viewport().focused == Some(true)));
    let centred = |height: f32| centred_on(main_inner, main_outer, egui::vec2(width, height));
    let mut builder = egui::ViewportBuilder::default()
        .with_title(title)
        .with_inner_size([width, height])
        .with_resizable(false)
        .with_minimize_button(false)
        .with_maximize_button(false);
    if let Some(position) = centred(height) {
        builder = builder.with_position(position);
    }
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
                if let Some(position) = centred(wanted) {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::OuterPosition(position));
                }
            }
            if main_focused {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Focus);
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

/// The outer position that centres a window of `inner` size on the main
/// window, whose inner and outer rectangles are given (screen points). The
/// dialog's frame is taken to be as wide as the main window's.
fn centred_on(main_inner: Option<egui::Rect>, main_outer: Option<egui::Rect>, inner: egui::Vec2) -> Option<egui::Pos2> {
    let (main_inner, main_outer) = (main_inner?, main_outer?);
    let border = main_inner.min - main_outer.min;
    Some((main_inner.center() - inner / 2.0 - border).round())
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
    let response = widgets::push_button(ui, p, enabled, text);
    let last_focused = egui::Id::unique("dialog_focused_button");
    let focused = response.has_focus();
    if focused {
        ui.ctx().data_mut(|d| d.insert_temp(last_focused, response.id));
    }
    let other =
        ui.memory(|m| m.focused()).is_some_and(|f| f != response.id && ui.ctx().data(|d| d.get_temp::<egui::Id>(last_focused)) == Some(f));
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
        let width = |t: &str| widgets::push_button_width(ui, t);
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
        self.dialog.is_some()
    }

    pub fn show_dialogs(&mut self, ctx: &egui::Context) {
        let Some(mut dialog) = self.dialog.take() else { return };
        let p = self.palette.clone();
        let parent = self.parent.as_ref();
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
                        widgets::segmented(ui, &p, closed, &items, form(WIDTH));
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
                        widgets::segmented(ui, &p, &mut choice.units, &items, form(WIDTH));
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
                    muted(ui, &p, "Fit every .dat file in the source folder with the current settings,\nsave each as .spline.dat in the target folder and keep the fits in the project.");
                    for (label, folder) in [("Source", &mut source), ("Target", &mut target)] {
                        form_row(ui, &p, label, true, "", |ui| {
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if dialog_button(ui, &p, "Choose...", false, true) {
                                    if let Some(dir) = crate::parent::pick_folder(parent, label) {
                                        *folder = dir.to_string_lossy().into_owned();
                                    }
                                }
                                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                                    // A folder the browser picked shows by its name.
                                    let text = if folder.is_empty() { "Not chosen" } else { crate::store::shown_path(folder) };
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
                        widgets::segmented(ui, &p, &mut mode, &items, form(WIDTH));
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
            Dialog::Confirm(c) => {
                const WIDTH: f32 = 380.0;
                let mut other = false;
                let answer = open_dialog(ctx, &c.title, WIDTH, DIALOG_MARGIN, |ui| {
                    ui.add(egui::Label::new(RichText::new(&c.message).color(p.text)).wrap());
                    ui.add_space(size::GROUP - size::GAP);
                    // At the right, as Windows orders them: the accept button
                    // first (it takes the focus and answers Enter), the other
                    // choice, then Cancel.
                    let (mut a, mut r) = (false, false);
                    let row = egui::vec2(ui.available_width(), size::CONTROL_HEIGHT);
                    ui.allocate_ui_with_layout(row, Layout::left_to_right(Align::Center), |ui| {
                        let width = |t: &str| widgets::push_button_width(ui, t);
                        let gap = ui.spacing().item_spacing.x;
                        let right = width(&c.accept) + gap + c.other.as_deref().map_or(0.0, |o| width(o) + gap) + width("Cancel");
                        ui.add_space((ui.available_width() - right).max(0.0));
                        a = dialog_button(ui, &p, &c.accept, c.default_accept, true);
                        if let Some(o) = &c.other {
                            other = dialog_button(ui, &p, o, false, true);
                        }
                        r = dialog_button(ui, &p, "Cancel", !c.default_accept, true);
                    });
                    // Cancel as the default: the first button does not keep
                    // the focus the dialog gave it on opening.
                    let opened = ui.ctx().data(|d| d.get_temp::<u8>(opened_id(&c.title))).unwrap_or(0);
                    if !c.default_accept && opened == 3 {
                        ui.memory_mut(|m| {
                            if let Some(id) = m.focused() {
                                m.surrender_focus(id);
                            }
                        });
                        ui.ctx().data_mut(|d| d.insert_temp(opened_id(&c.title), 4u8));
                    }
                    answer_of(a, r)
                });
                match answer {
                    Answer::Open if !other => true,
                    Answer::Accept | Answer::Open => {
                        let Dialog::Confirm(c) = dialog else { unreachable!() };
                        if c.save_first && !other {
                            self.on_save_project();
                            if self.projects.modified {
                                // Not saved (Save as was cancelled): nothing goes on.
                                return;
                            }
                        }
                        (c.then)(self);
                        return;
                    }
                    Answer::Reject => {
                        if !c.cancelled.is_empty() {
                            self.set_status(c.cancelled.clone());
                        }
                        false
                    }
                }
            }
            Dialog::ReadReport(report) => {
                const WIDTH: f32 = 640.0;
                let zoom_ms = self.zoom_ms;
                let answer = open_dialog(ctx, "Mistakes in the airfoil file", WIDTH, DIALOG_MARGIN, |ui| {
                    let on_check = ui.memory(|m| m.focused()).is_some_and(|f| report.checks.contains(&f));
                    let enter = on_check && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));
                    read_report_content(ui, &p, report, zoom_ms);
                    ui.add_space(size::GROUP - size::GAP);
                    let (a, r, _) = buttons(ui, &p, Some(("Load", true)), "Cancel", None);
                    answer_of(a || enter, r)
                });
                match answer {
                    Answer::Open => true,
                    Answer::Reject => {
                        self.set_status("Cancelled: the airfoil was not loaded");
                        false
                    }
                    Answer::Accept => {
                        let Dialog::ReadReport(r) = dialog else { unreachable!() };
                        let (coords, precision) = r.parsed.with_corrections(&r.accept);
                        self.set_airfoil(coords, &r.filename, &r.name, (r.precision)(precision), &r.path, &r.verb());
                        return;
                    }
                }
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
    pub fn comparison_block(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        ui.vertical(|ui| {
            // As the block beside it: the caption at the left, the axis
            // choices compact at the right.
            let row = egui::vec2(ui.available_width(), size::CONTROL_HEIGHT);
            ui.allocate_ui_with_layout(row, Layout::left_to_right(Align::Center), |ui| {
                widgets::caption(ui, &p, "Runs");
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let metrics: Vec<usize> = (0..COLUMNS.len()).filter(|&c| COLUMNS[c].metric).collect();
                    let names: Vec<&str> = metrics.iter().map(|&c| COLUMNS[c].name).collect();
                    let mut y = metrics.iter().position(|&c| c == self.explorer.y_column).unwrap_or(0);
                    if widgets::combo(ui, &p, "cmp_y", &mut y, &names, 0.0, "The fit metric plotted for each run") {
                        self.explorer.y_column = metrics[y];
                        self.explorer.forget_points();
                        self.refresh_comparison();
                    }
                    widgets::toolbar_label(ui, &p, "Y axis");
                    let all: Vec<&str> = COLUMNS.iter().map(|c| c.name).collect();
                    let mut x = self.explorer.x_column;
                    let tip = "The explorer column the runs are plotted along.\nOnly the runs that pass the explorer's filter are shown.";
                    if widgets::combo(ui, &p, "cmp_x", &mut x, &all, 0.0, tip) {
                        self.explorer.x_column = x;
                        self.explorer.forget_points();
                        self.refresh_comparison();
                    }
                    widgets::toolbar_label(ui, &p, "X axis");
                });
            });
            if let Some(i) = self.comparison.show(ui, &p, self.zoom_ms) {
                self.click_plotted_run(i);
            }
        });
    }
}

/// The report of the shown fit, for the Report and Shape views of the
/// bottom block; `key` tells the fit it belongs to.
pub struct ReportPanel {
    pub key: u64,
    pub title: String,
    pub report: Report,
    pub shape: Option<Shape>,
    pub view: View,
}

impl App {
    /// The bottom-left block: the convergence plot, the fit report or the
    /// thickness and camber chart, chosen by the tabs in its header.
    pub fn loss_block(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        ui.vertical(|ui| {
            let row = egui::vec2(ui.available_width(), size::CONTROL_HEIGHT);
            ui.allocate_ui_with_layout(row, Layout::left_to_right(Align::Center), |ui| {
                let mut view = self.bottom_view;
                if widgets::caption_tabs(ui, &p, &mut view, &BOTTOM_VIEWS) {
                    self.set_bottom_view(view);
                }
                if self.bottom_view == 1 {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let enabled = self.report_panel.is_some() && !self.fitting();
                        let tip = "Copy the report as text: a line per value, tab separated, so it\npastes into a spreadsheet as cells";
                        if widgets::push_button(ui, &p, enabled, "Copy").on_hover_text(tip).clicked() {
                            self.copy_report(ui.ctx());
                        }
                    });
                }
                if self.bottom_view == 0 {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if widgets::combo(ui, &p, "loss_log", &mut self.loss.log_mode, &["None", "X only", "Y only", "Both"], 0.0, "") {
                            self.loss.view.follow = true;
                        }
                        widgets::toolbar_label(ui, &p, "Log scale");
                        if widgets::combo(ui, &p, "loss_x", &mut self.loss.x_axis, &["Iterations", "Time"], 0.0, "") {
                            self.loss.view.follow = true;
                        }
                        widgets::toolbar_label(ui, &p, "X axis");
                    });
                }
            });
            if self.bottom_view == 0 {
                self.loss.show(ui, &p, self.zoom_ms);
                return;
            }
            let rect = ui.available_rect_before_wrap();
            ui.allocate_rect(rect, egui::Sense::hover());
            let ready = !self.fitting() && self.ensure_report_panel();
            match (ready, self.report_panel.as_mut()) {
                (true, Some(panel)) if self.bottom_view == 1 => report_document(ui, &p, rect, &panel.title, &panel.report),
                (true, Some(panel)) if panel.shape.is_some() => {
                    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect));
                    shape_plot(&mut child, &p, panel.shape.as_ref().unwrap(), &mut panel.view);
                }
                _ => {
                    let radius = egui::CornerRadius::same(size::RADIUS);
                    ui.painter().rect_filled(rect, radius, p.canvas);
                    ui.painter().rect_stroke(rect, radius, egui::Stroke::new(1.0, p.rule), egui::StrokeKind::Inside);
                    let text = if self.fitting() {
                        "Shown when the fit is done"
                    } else if self.bottom_view == 1 {
                        "The report of the shown fit: fit a spline or pick a run"
                    } else {
                        "The thickness and camber of the shown fit: fit a spline or pick a run"
                    };
                    crate::plots::centred_hint(ui.painter(), rect, text, p.text_3, size::PAD_LARGE);
                }
            }
        });
    }

    /// Copies the shown report to the clipboard as text.
    pub fn copy_report(&mut self, ctx: &egui::Context) {
        let Some(panel) = &self.report_panel else { return };
        let text = panel.report.text(&panel.title);
        match crate::parent::copy_text(&ctx, text) {
            Ok(()) => self.set_status("Copied the fit report"),
            Err(e) => self.fail(format!("Copy failed: {e}")),
        }
    }

    /// Chooses the bottom block's view, kept between sessions.
    pub fn set_bottom_view(&mut self, view: usize) {
        self.bottom_view = view;
        self.store.set_int("view/bottom", view as i64);
    }

    /// The report panel of the shown fit, made when the fit changed; false
    /// without a fit.
    pub fn ensure_report_panel(&mut self) -> bool {
        let (Some(fit), Some(airfoil)) = (self.fit.as_mut(), &self.airfoil) else {
            self.report_panel = None;
            return false;
        };
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for c in fit.section_control_points.iter().flatten() {
                (c.x.to_bits(), c.y.to_bits()).hash(&mut h);
            }
            fit.final_error.to_bits().hash(&mut h);
            h.finish()
        };
        if self.report_panel.as_ref().is_some_and(|r| r.key == key) {
            return true;
        }
        let curves = fit.curves();
        if let Some(report) = &mut fit.report {
            // A report saved before the shape keys gets them now.
            if !report.shape_keys {
                report.shape = spline_fitter::analysis::airfoil_shape(&curves);
                report.shape_keys = true;
            }
        } else {
            fit.report = Some(spline_fitter::analysis::fit_report(&airfoil.sections, fit, Some(airfoil.precision), None));
        }
        let title = [&airfoil.name, &airfoil.filename].into_iter().find(|s| !s.is_empty()).cloned().unwrap_or_else(|| "Fit report".into());
        let report = fit.report.clone().expect("made above");
        let shape = spline_fitter::bspline_varpro::airfoil::thickness_and_camber(&curves).ok();
        self.report_panel = Some(ReportPanel { key, title, report, shape, view: View::default() });
        true
    }
}

/// The views of the bottom-left block.
pub const BOTTOM_VIEWS: [&str; 3] = ["Convergence", "Report", "Shape"];

/// The report as the Python version's text browser showed it: a framed,
/// scrolling document with a heading, a table of label and value rows and
/// the junction table.
fn report_document(ui: &mut Ui, p: &Palette, rect: egui::Rect, title: &str, report: &Report) {
    // Rows as close as the run details beside it: the block is short.
    const ROW: f32 = 24.0;
    let radius = egui::CornerRadius::same(size::RADIUS);
    // Framed as the plots beside it.
    ui.painter().rect_filled(rect, radius, p.canvas);
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

/// The explanation, the table of the mistakes in a read airfoil file with a
/// check box each, and the preview of the points the checked corrections
/// remove. Clicking a row zooms the preview to it; clicking it again shows
/// the whole airfoil.
fn read_report_content(ui: &mut Ui, p: &Palette, report: &mut ReadReport, motion_ms: u64) {
    const ROW: f32 = 24.0;
    const MAX_ROWS: usize = 6;
    const PREVIEW_HEIGHT: f32 = 280.0;
    let mistakes = report.findings().len();
    let removed: usize = report.findings().iter().zip(&report.accept).filter(|(_, a)| **a).map(|(f, _)| f.removed()).sum();
    let message = format!(
        "{} has {} {} that would make the fitted curve loop or turn back. With the checked corrections the airfoil leaves out {} of its {} points.",
        report.source,
        mistakes,
        if mistakes == 1 { "mistake" } else { "mistakes" },
        removed,
        report.parsed.read_points.len(),
    );
    ui.add(egui::Label::new(RichText::new(message).color(p.text)).wrap());
    ui.add_space(size::GROUP - size::GAP);

    let body = egui::FontId::proportional(crate::theme::typeface::BASE);
    let mono = egui::FontId::monospace(crate::theme::typeface::BASE);
    let header = crate::fonts::semibold_id(crate::theme::typeface::BASE);
    let g = |v: f64| number(v, "g");
    let rows: Vec<[String; 4]> = report
        .findings()
        .iter()
        .map(|f| [f.line.map_or(String::new(), |l| l.to_string()), g(f.point.x), g(f.point.y), f.description()])
        .collect();
    let painter = ui.painter().clone();
    let width = |text: &str, font: &egui::FontId| painter.layout_no_wrap(text.to_string(), font.clone(), p.text).size().x;
    let heads = ["Line", "x", "y", "Mistake"];
    // The check box, then line, x and y as wide as their widest text; the mistake takes the rest.
    let check = size::CHECK_BOX + size::PAD;
    let widths: Vec<f32> = (0..3).map(|c| rows.iter().map(|r| width(&r[c], &mono)).fold(width(heads[c], &header), f32::max) + size::PAD).collect();
    let cells = |ui: &Ui, r: egui::Rect, texts: [&str; 4], head: bool| {
        let mut x = r.left() + check;
        for (c, text) in texts.iter().enumerate() {
            let (font, color) = match (head, c) {
                (true, _) => (header.clone(), p.text_3),
                (false, 3) => (body.clone(), p.text),
                _ => (mono.clone(), p.text),
            };
            // Numbers right-aligned in their column, the words left.
            if c < 3 {
                let right = x + widths[c] - size::PAD;
                ui.painter().text(egui::pos2(right, r.center().y), egui::Align2::RIGHT_CENTER, *text, font, color);
                x += widths[c];
            } else {
                ui.painter().text(egui::pos2(x, r.center().y), egui::Align2::LEFT_CENTER, *text, font, color);
            }
        }
    };
    let check_rect = |r: egui::Rect| egui::Rect::from_min_size(egui::pos2(r.left() + 3.0, r.center().y - size::CHECK_BOX / 2.0), egui::Vec2::splat(size::CHECK_BOX));
    ui.spacing_mut().item_spacing.y = 0.0;

    // The header: a check box for all of them.
    let (head, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW), egui::Sense::hover());
    let mut all = report.accept.iter().all(|a| *a);
    let partly = !all && report.accept.iter().any(|a| *a);
    report.checks.clear();
    let all_box = ui.put(check_rect(head), |ui: &mut Ui| widgets::tristate_checkbox(ui, p, &mut all, partly, ""));
    report.checks.push(all_box.id);
    let toggled = all_box.on_hover_text("Make all corrections or none").changed();
    if toggled {
        report.accept.iter_mut().for_each(|a| *a = all);
    }
    cells(ui, head, heads, true);
    ui.painter().hline(head.left()..=head.right(), head.bottom() - 2.0, egui::Stroke::new(1.0, p.rule));

    let mut clicked = None;
    egui::ScrollArea::vertical().id_salt("read_report").max_height(ROW * MAX_ROWS as f32).auto_shrink([false, true]).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        for (i, r) in rows.iter().enumerate() {
            let (rect, row) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW), egui::Sense::click());
            let fill = if report.selected == Some(i) {
                Some(p.select)
            } else if row.hovered() {
                Some(p.field_hover)
            } else {
                None
            };
            if let Some(fill) = fill {
                ui.painter().rect_filled(rect, egui::CornerRadius::same(size::RADIUS), fill);
            }
            let check = ui.put(check_rect(rect), |ui: &mut Ui| widgets::checkbox(ui, p, &mut report.accept[i], ""));
            report.checks.push(check.id);
            check.on_hover_text("Make this correction");
            cells(ui, rect, [&r[0], &r[1], &r[2], &r[3]], false);
            if row.on_hover_text("Show it in the preview; click again for the whole airfoil").clicked() {
                clicked = Some(i);
            }
        }
    });
    ui.spacing_mut().item_spacing.y = size::GAP;
    ui.add_space(size::GROUP - size::GAP);

    // The preview, in the file's own coordinates.
    if let Some(i) = clicked {
        report.toggle_selected(i);
    }
    let points = report.parsed.read_points.clone();
    let xy = |i: &usize| [points[*i].x, points[*i].y];
    if report.view.follow {
        report.view.fit(crate::plots::bounds_of(points.iter().map(|q| [q.x, q.y])));
    }
    let removed: Vec<usize> = report.findings().iter().zip(&report.accept).filter(|(_, a)| **a).flat_map(|(f, _)| f.removes.iter().copied()).collect();
    let kept_back: Vec<usize> = report.findings().iter().zip(&report.accept).filter(|(_, a)| !**a).flat_map(|(f, _)| f.removes.iter().copied()).collect();
    let selected: Vec<usize> = report.selected.map_or(vec![], |i| report.findings()[i].removes.clone());
    let line: Vec<[f64; 2]> = (0..points.len()).filter(|i| !removed.contains(i)).map(|i| xy(&i)).collect();
    let axes = crate::plots::Axes {
        x_label: "x",
        y_label: "y",
        log_x: false,
        log_y: false,
        x_ticks: None,
        whole_x: false,
        equal_aspect: true,
        marker_pad: crate::plots::marker_pad(size::MARKER_SIZE, 1.5),
    };
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), PREVIEW_HEIGHT), egui::Sense::hover());
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect));
    crate::plots::show_plot(&mut child, p, "read_report_preview", &mut report.view, &axes, motion_ms, |c| {
        use crate::plots::Marker;
        // Drawing order: the outline that is loaded, the data, then the rings.
        c.polyline(line.iter().copied(), egui::Stroke::new(1.0, p.construction), false);
        c.markers(points.iter().map(|q| [q.x, q.y]), Marker::Circle, size::DATA_SIZE, p.data(), egui::Stroke::NONE);
        let ring = |stroke: egui::Color32, width: f32| egui::Stroke::new(width, stroke);
        c.markers(kept_back.iter().map(xy), Marker::Circle, size::EXTREMUM_SIZE + 1.0, egui::Color32::TRANSPARENT, ring(p.text_3, 1.5));
        c.markers(removed.iter().map(xy), Marker::Circle, size::EXTREMUM_SIZE + 1.0, egui::Color32::TRANSPARENT, ring(p.warn, 1.5));
        c.markers(selected.iter().map(xy), Marker::Circle, size::MARKER_SIZE, egui::Color32::TRANSPARENT, ring(p.text, 1.5));
    });
    muted(
        ui,
        p,
        "The line joins the points that are loaded. Rings mark the points the checked corrections remove, grey rings\nthe mistakes left in. Click a row to zoom to it; scroll to zoom, drag to pan.",
    );
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
        egui::Popup::from_toggle_button_response(&arrow)
            .id(popup_id)
            .frame(frame)
            .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
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
            changed |= widgets::int_field(
                ui,
                &p,
                "comb_density",
                &mut settings.density,
                (5, 2000),
                Field::new(widgets::spin_width(ui, &["5", "2000"]), ""),
                IntOptions::default(),
            )
        });
        changed
    });
    plot.comb = settings;
    if toggled || (changed && plot.comb_on) {
        plot.refresh_comb();
    }
}

/// The deviation scales the Scale field takes.
pub const DEVIATION_SCALES: (f64, f64) = (0.001, 1e9);

/// `k` steps along 1, 2, 5, 10, 20, ... from `v` (from the step below or
/// above `v` when it is between two).
pub fn step_1_2_5(v: f64, k: i32) -> f64 {
    if !(v.is_finite() && v > 0.0) || k == 0 {
        return v;
    }
    // Index on the 1-2-5 ladder: 3 per decade.
    let ladder = |i: i32| [1.0, 2.0, 5.0][i.rem_euclid(3) as usize] * 10f64.powi(i.div_euclid(3));
    let mut i = 3 * v.log10().floor() as i32;
    while ladder(i + 1) <= v * (1.0 + 1e-9) {
        i += 1;
    }
    // `i` is the rung at or below `v`; from between two rungs one step
    // either way takes the nearer rung on that side.
    let on_rung = (ladder(i) - v).abs() <= 1e-9 * v;
    ladder(if k < 0 && !on_rung { i + k + 1 } else { i + k })
}

/// The Deviation toggle and its scale.
pub fn deviation_toggle(app: &mut App, ui: &mut Ui) {
    let p = app.palette.clone();
    let plot = &mut app.main_plot;
    let mut scale = plot.deviation_scale;
    let toggle = SplitToggle {
        text: "Deviation",
        tooltip: "Display the distance of every data point from the curve, drawn
outward: rose for a point outside the curve, sky blue for one inside.
The dashed line is the tolerance. Use the arrow for the scale.",
        arrow_tooltip: "Deviation settings",
        id: "deviation_settings",
    };
    let (toggled, changed) = split_toggle(ui, &p, toggle, &mut plot.deviation_on, |ui| {
        let mut changed = false;
        form_row(ui, &p, "Scale", true, "", |ui| {
            let fmt = |v: f64| spline_fitter::formatting::number(v, "g");
            let parse = |t: &str| widgets::parse_scientific(t);
            let step = |v: f64, k: i32| step_1_2_5(v, k);
            let f = widgets::NumberFormat { format: &fmt, parse: &parse, step: &step };
            changed |= widgets::number_field(
                ui,
                &p,
                "deviation_scale",
                &mut scale,
                DEVIATION_SCALES,
                Field::new(
                    widgets::spin_width(ui, &[&fmt(0.001), &fmt(1e9)]),
                    "How many times the real distance the deviations and the
tolerance line are drawn: 1 is true to size.",
                ),
                &f,
            );
        });
        changed
    });
    plot.deviation_scale = scale;
    if changed {
        plot.refresh_tolerance_line();
    }
    if toggled || changed {
        plot.refresh_deviation();
    }
}

#[cfg(test)]
mod tests {
    use super::step_1_2_5;

    /// The wheel walks the deviation scale along 1, 2, 5, 10, ...; from
    /// between two rungs it lands on the nearer one in that direction.
    #[test]
    fn the_scale_steps_along_1_2_5() {
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b;
        for (v, k, want) in [(1.0, 1, 2.0), (2.0, 1, 5.0), (5.0, 1, 10.0), (10.0, -1, 5.0), (1.0, -1, 0.5), (1.0, 3, 10.0), (3.0, 1, 5.0), (3.0, -1, 2.0), (1000.0, 2, 5000.0), (0.002, -1, 0.001)] {
            assert!(close(step_1_2_5(v, k), want), "{v} {k}: {}", step_1_2_5(v, k));
        }
    }
}
