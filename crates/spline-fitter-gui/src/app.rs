//! The main window: menus, layout, status bar and the frame loop.

use std::path::PathBuf;
use std::time::Instant;

use eframe::egui::{self, Align, Key, KeyboardShortcut, Layout, Modifiers, RichText, Ui};
use spline_fitter::geometry::AirfoilData;
use spline_fitter::models::FitResult;

use crate::dialogs::{Dialog, ReportWindow};
use crate::explorer::{ExplorerState, Projects};
use crate::fitting::Running;
use crate::history::History;
use crate::job::LiveMode;
use crate::plots::{ComparisonPlot, LossPlot, MainPlot};
use crate::settings::{AdvancedBaseline, Settings, Values};
use crate::store::Store;
use crate::theme::{self, size, Palette, ThemeMode};
use crate::widgets;

pub const DEFAULT_SCROLL_MS: u64 = 350;
pub const DEFAULT_ZOOM_MS: u64 = 650;
pub const MAX_MOTION_MS: u64 = 1000;

/// The application window.
pub struct App {
    pub palette: Palette,
    pub theme_mode: ThemeMode,
    pub temperature: i32,
    pub scroll_ms: u64,
    pub zoom_ms: u64,
    applied: Option<(bool, i32, u64)>,
    pub store: Store,
    pub settings: Settings,
    /// The settings as of the end of the previous frame.
    before: Settings,
    pub history: History<Values>,
    /// Changes made now are the program's, not the user's (no undo step).
    pub quiet: bool,
    pub baseline: AdvancedBaseline,
    pub advanced_open: bool,
    pub airfoil: Option<AirfoilData>,
    /// Path of the airfoil file (empty for pasted or created airfoils).
    pub airfoil_path: String,
    pub fit: Option<FitResult>,
    pub status: String,
    pub running: Option<Running>,
    pub live_mode: LiveMode,
    pub main_plot: MainPlot,
    pub loss: LossPlot,
    pub comparison: ComparisonPlot,
    pub projects: Projects,
    pub explorer: ExplorerState,
    pub explorer_visible: bool,
    pub dialog: Option<Dialog>,
    pub report: Option<ReportWindow>,
    pub batch_source: String,
    pub batch_target: String,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        crate::fonts::install(&cc.egui_ctx);
        spline_fitter::formatting::set_decimal_point(crate::locale::decimal_point());
        let store = Store::open();
        let theme_mode = ThemeMode::from_value(&store.string("theme/mode", "system"));
        let temperature = store.int("theme/temperature", 0).clamp(-100, 100) as i32;
        let scroll_ms = (store.int("motion/scroll_ms", DEFAULT_SCROLL_MS as i64).clamp(0, MAX_MOTION_MS as i64)) as u64;
        let zoom_ms = (store.int("motion/zoom_ms", DEFAULT_ZOOM_MS as i64).clamp(0, MAX_MOTION_MS as i64)) as u64;
        let settings = Settings::default();
        let mut app = App {
            palette: Palette::new(true, temperature),
            theme_mode,
            temperature,
            scroll_ms,
            zoom_ms,
            applied: None,
            store,
            history: History::new(settings.values()),
            before: settings.clone(),
            baseline: AdvancedBaseline::of(&settings),
            settings,
            quiet: false,
            advanced_open: false,
            airfoil: None,
            airfoil_path: String::new(),
            fit: None,
            status: "Ready".into(),
            running: None,
            live_mode: LiveMode::Moderate,
            main_plot: MainPlot::default(),
            loss: LossPlot::new(),
            comparison: ComparisonPlot::default(),
            projects: Projects::default(),
            explorer: ExplorerState::default(),
            explorer_visible: true,
            dialog: None,
            report: None,
            batch_source: String::new(),
            batch_target: String::new(),
        };
        app.refresh_comparison();
        app
    }

    pub fn fitting(&self) -> bool {
        self.running.is_some()
    }

    pub fn set_status(&mut self, text: impl Into<String>) {
        self.status = text.into();
    }

    /// A failure: the status text and the error chime.
    pub fn fail(&mut self, text: impl Into<String>) {
        self.status = text.into();
        crate::chime::error();
    }

    /// The decimal point, theme and motion settings in effect.
    fn apply_theme(&mut self, ctx: &egui::Context) {
        let dark = match self.theme_mode {
            ThemeMode::Light => false,
            ThemeMode::Dark => true,
            ThemeMode::System => ctx.system_theme().is_none_or(|t| t == egui::Theme::Dark),
        };
        let key = (dark, self.temperature, self.scroll_ms);
        if self.applied == Some(key) {
            return;
        }
        self.applied = Some(key);
        self.palette = Palette::new(dark, self.temperature);
        theme::apply(ctx, &self.palette);
        let seconds = self.scroll_ms as f32 / 1000.0;
        ctx.all_styles_mut(|s| s.scroll_animation.duration = egui::Rangef::new(seconds * 0.3, seconds));
        ctx.set_theme(if dark { egui::Theme::Dark } else { egui::Theme::Light });
    }

    pub fn set_theme(&mut self, mode: Option<ThemeMode>, temperature: Option<i32>) {
        if let Some(m) = mode {
            self.theme_mode = m;
            self.store.set_string("theme/mode", m.value());
        }
        if let Some(t) = temperature {
            self.temperature = t.clamp(-100, 100);
            self.store.set_int("theme/temperature", self.temperature as i64);
        }
    }

    pub fn set_motion(&mut self, scroll_ms: u64, zoom_ms: u64) {
        self.scroll_ms = scroll_ms.min(MAX_MOTION_MS);
        self.zoom_ms = zoom_ms.min(MAX_MOTION_MS);
        self.store.set_int("motion/scroll_ms", self.scroll_ms as i64);
        self.store.set_int("motion/zoom_ms", self.zoom_ms as i64);
    }

    /// Settles the settings after this frame's edits: the rules and the undo
    /// history. A new section count changes nothing else; the next fit splits.
    fn settle_settings(&mut self) {
        if self.settings.values() == self.before.values() {
            return;
        }
        let before = self.before.clone();
        self.settings.settle(&before);
        let quiet = self.quiet || self.fitting();
        self.history.on_change(self.settings.values(), quiet, Instant::now());
        self.before = self.settings.clone();
    }

    /// Takes the settings as they are as settled, without rules or a step.
    pub fn accept_settings(&mut self) {
        self.before = self.settings.clone();
    }

    pub fn undo(&mut self) {
        if let Some(values) = self.history.undo(&self.settings.values()) {
            self.restore(&values);
        }
    }

    pub fn redo(&mut self) {
        if let Some(values) = self.history.redo(&self.settings.values()) {
            self.restore(&values);
        }
    }

    fn restore(&mut self, values: &Values) {
        self.settings.restore(values);
        self.history.restored(self.settings.values());
        self.before = self.settings.clone();
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        let ctrl = |key| KeyboardShortcut::new(Modifiers::COMMAND, key);
        let ctrl_shift = |key| KeyboardShortcut::new(Modifiers::COMMAND | Modifiers::SHIFT, key);
        let typing = ctx.egui_wants_keyboard_input();
        let pressed = |s: KeyboardShortcut| ctx.input_mut(|i| i.consume_shortcut(&s));
        if pressed(ctrl_shift(Key::O)) {
            self.on_open_project();
        }
        if pressed(ctrl_shift(Key::E)) {
            self.explorer_visible = !self.explorer_visible;
        }
        if pressed(ctrl(Key::N)) {
            self.on_new_naca();
        }
        if pressed(ctrl(Key::O)) {
            self.on_open_airfoil();
        }
        if pressed(ctrl(Key::S)) && self.projects.file.is_some() {
            self.on_save_project();
        }
        if pressed(ctrl(Key::E)) && self.fit.is_some() {
            self.on_export_cad();
        }
        if pressed(ctrl(Key::R)) && self.fit.is_some() {
            self.on_report();
        }
        if !typing && !self.fitting() {
            if pressed(ctrl_shift(Key::Z)) || pressed(ctrl(Key::Y)) {
                self.redo();
            }
            if pressed(ctrl(Key::Z)) {
                self.undo();
            }
        }
        if !typing && !self.fitting() {
            let pasted = ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Paste(_))));
            if pasted {
                self.on_paste();
            }
        }
    }

    fn dropped_files(&mut self, ctx: &egui::Context) {
        let files: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        if self.fitting() {
            return;
        }
        let supported =
            |p: &PathBuf| p.extension().map(|e| e.to_string_lossy().to_lowercase()).is_some_and(|e| e == "dat" || e == "asfproj");
        if let Some(path) = files.into_iter().find(supported) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            self.open_path(&path);
        }
    }

    fn menu_bar(&mut self, ui: &mut Ui) {
        let fitting = self.fitting();
        let has_fit = self.fit.is_some() && !fitting;
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                let item = |ui: &mut Ui, text: &str, shortcut: &str, enabled: bool| {
                    ui.add_enabled(enabled, egui::Button::new(text).shortcut_text(shortcut)).clicked()
                };
                if item(ui, "New NACA airfoil...", "Ctrl+N", !fitting) {
                    self.on_new_naca();
                }
                if item(ui, "Open airfoil...", "Ctrl+O", !fitting) {
                    self.on_open_airfoil();
                }
                ui.add_enabled_ui(!fitting, |ui| {
                    ui.menu_button("Open recent", |ui| self.recent_menu(ui));
                });
                ui.separator();
                if item(ui, "Save fitted airfoil (.dat)...", "", has_fit) {
                    self.on_save_airfoil();
                }
                if item(ui, "Save spline (.spline.dat)...", "", has_fit) {
                    self.on_save_spline();
                }
                if item(ui, "Export CAD (STEP, IGES, DXF)...", "Ctrl+E", has_fit) {
                    self.on_export_cad();
                }
                ui.separator();
                if item(ui, "Open project...", "Ctrl+Shift+O", !fitting) {
                    self.on_open_project();
                }
                let has_project = self.projects.file.is_some();
                if item(ui, "Save project", "Ctrl+S", has_project) {
                    self.on_save_project();
                }
                if item(ui, "Save project as...", "", has_project) {
                    self.on_save_project_as();
                }
                ui.separator();
                if item(ui, "Batch convert...", "", !fitting) {
                    self.dialog = Some(Dialog::batch());
                }
                ui.separator();
                if item(ui, "Exit", "Alt+F4", true) {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui.add_enabled(!fitting && self.history.can_undo(), egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                    self.undo();
                }
                if ui.add_enabled(!fitting && self.history.can_redo(), egui::Button::new("Redo").shortcut_text("Ctrl+Y")).clicked() {
                    self.redo();
                }
                if ui.add_enabled(!fitting && !self.projects.deleted.is_empty(), egui::Button::new("Restore deleted runs")).clicked() {
                    self.restore_deleted_runs();
                }
                ui.separator();
                if ui.add_enabled(!fitting, egui::Button::new("Paste airfoil").shortcut_text("Ctrl+V")).clicked() {
                    self.on_paste();
                }
            });
            ui.menu_button("View", |ui| {
                if ui.add_enabled(has_fit, egui::Button::new("Fit report...").shortcut_text("Ctrl+R")).clicked() {
                    self.on_report();
                }
                ui.separator();
                let mut visible = self.explorer_visible;
                if ui.add(egui::Button::new("Project explorer").selected(visible).shortcut_text("Ctrl+Shift+E")).clicked() {
                    visible = !visible;
                }
                self.explorer_visible = visible;
                ui.separator();
                if ui.button("Theme...").clicked() {
                    self.dialog = Some(Dialog::theme(self));
                }
                if ui.button("Scrolling and zooming...").clicked() {
                    self.dialog = Some(Dialog::motion(self));
                }
            });
            ui.menu_button("Help", |ui| {
                if ui.button("About").clicked() {
                    self.set_status(format!("{} {}", spline_fitter::APP_NAME, spline_fitter::VERSION));
                }
            });
        });
    }

    fn recent_menu(&mut self, ui: &mut Ui) {
        let files = self.store.recent_files();
        if files.is_empty() {
            ui.add_enabled(false, egui::Button::new("No recent files"));
            return;
        }
        for (i, file) in files.iter().enumerate() {
            let name = std::path::Path::new(file).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let text = if i < 9 { format!("{} {name}", i + 1) } else { name };
            if ui.button(text).on_hover_text(file).clicked() {
                self.open_path(std::path::Path::new(file));
            }
        }
        ui.separator();
        if ui.button("Clear recent files").clicked() {
            self.store.set_recent_files(&[]);
        }
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        egui::Panel::bottom("status").frame(egui::Frame::new().fill(self.palette.window).inner_margin(egui::Margin::symmetric(8, 3))).show(
            ui,
            |ui| {
                ui.label(RichText::new(&self.status).color(self.palette.text_2));
            },
        );
    }

    fn title(&self) -> String {
        match &self.projects.file {
            None => spline_fitter::APP_NAME.to_string(),
            Some(_) => format!("{}{} - {}", self.projects.name(), if self.projects.modified { "*" } else { "" }, spline_fitter::APP_NAME),
        }
    }

    fn central(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let frame = egui::Frame::new().fill(p.window).inner_margin(egui::Margin::same(8));
        egui::CentralPanel::default().frame(frame).show(ui, |ui| {
            self.plot_toolbar(ui);
            ui.add_space(size::GAP);
            let total = ui.available_height();
            let main_height = (total - size::GAP) * 0.6;
            ui.allocate_ui(egui::vec2(ui.available_width(), main_height), |ui| self.main_plot.show(ui, &p, self.zoom_ms));
            ui.add_space(size::GAP);
            let rest = ui.available_height();
            let wide = ui.available_width() >= 2.0 * 440.0 + size::GROUP;
            if wide {
                let half = (ui.available_width() - size::GROUP) / 2.0;
                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(half, rest), |ui| self.loss_block(ui));
                    ui.add_space(size::GROUP);
                    ui.allocate_ui(egui::vec2(half, rest), |ui| self.comparison_block(ui));
                });
            } else {
                let half = (rest - size::GROUP) / 2.0;
                ui.allocate_ui(egui::vec2(ui.available_width(), half), |ui| self.loss_block(ui));
                ui.add_space(size::GROUP);
                ui.allocate_ui(egui::vec2(ui.available_width(), half), |ui| self.comparison_block(ui));
            }
        });
    }

    fn loss_block(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                widgets::caption(ui, &p, "Convergence");
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if widgets::combo(ui, &p, "loss_log", &mut self.loss.log_mode, &["None", "X only", "Y only", "Both"], 0.0, "") {
                        self.loss.view.follow = true;
                    }
                    ui.label(RichText::new("Log scale").color(p.text_2));
                    if widgets::combo(ui, &p, "loss_x", &mut self.loss.x_axis, &["Iterations", "Time"], 0.0, "") {
                        self.loss.view.follow = true;
                    }
                    ui.label(RichText::new("X axis").color(p.text_2));
                });
            });
            self.loss.show(ui, &p, self.zoom_ms);
        });
    }

    fn plot_toolbar(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        // The live preview setting at the right of the display toggles, or
        // under them when the plot is too narrow for both.
        let font = egui::FontId::proportional(crate::theme::typeface::BASE);
        let measure = |ui: &Ui, t: &str| ui.painter().layout_no_wrap(t.to_string(), font.clone(), p.text).size().x;
        let longest = LiveMode::LABELS.iter().map(|t| measure(ui, t)).fold(0.0, f32::max);
        let live_width = measure(ui, "Live preview") + size::GAP + longest + size::COMBO_EXTRA;
        let mut wraps = false;
        ui.horizontal(|ui| {
            let knots =
                widgets::toggle_button(ui, &p, "Knots", self.main_plot.show_knots, "Mark the curve points at the knots of every section.");
            if knots.clicked() {
                self.main_plot.show_knots = !self.main_plot.show_knots;
                self.main_plot.refresh_knots();
            }
            crate::dialogs::comb_toggle(self, ui);
            crate::dialogs::deviation_toggle(self, ui);
            wraps = ui.available_width() < live_width + size::GAP;
            if !wraps {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| self.live_preview(ui, &p, true));
            }
        });
        if wraps {
            ui.add_space(size::GAP - ui.spacing().item_spacing.y);
            ui.horizontal(|ui| self.live_preview(ui, &p, false));
        }
    }

    /// The Live preview label and combo (`reversed` in a right-to-left row).
    fn live_preview(&mut self, ui: &mut Ui, p: &crate::theme::Palette, reversed: bool) {
        let label = |ui: &mut Ui| ui.label(RichText::new("Live preview").color(p.text_2));
        if !reversed {
            label(ui);
        }
        let mut mode = self.live_mode as usize;
        let tip = "How often the plots update while fitting.\nMinimal: only at the end. Moderate: once a second.\nDetailed: every iteration (slower).";
        if widgets::combo(ui, p, "live_mode", &mut mode, &LiveMode::LABELS, 0.0, tip) {
            self.live_mode = LiveMode::ALL[mode];
            if let Some(live) = self.running.as_ref().map(|r| r.live()) {
                live.set_mode(self.live_mode);
            }
        }
        if reversed {
            label(ui);
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.apply_theme(&ctx);
        self.tick_job(&ctx);
        self.shortcuts(&ctx);
        self.dropped_files(&ctx);
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(self.title()));

        let p = self.palette.clone();
        let menu = egui::Panel::top("menu").exact_size(size::MENU_BAR_HEIGHT).frame(egui::Frame::new().fill(p.window)).show(ui, |ui| {
            ui.spacing_mut().button_padding = egui::vec2(10.0, 4.0);
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.spacing_mut().interact_size.y = size::MENU_BAR_HEIGHT - 1.0;
            self.menu_bar(ui)
        });
        let r = menu.response.rect;
        ui.painter().hline(r.x_range(), r.bottom() - 0.5, egui::Stroke::new(1.0, p.rule));
        self.status_bar(ui);
        self.sidebar(ui);
        if self.explorer_visible {
            self.explorer_panel(ui);
        } else {
            self.explorer_rail(ui);
        }
        self.central(ui);
        self.show_dialogs(&ctx);

        self.settle_settings();
        let now = Instant::now();
        let current = self.settings.values();
        self.history.tick(now, &current);
        if let Some(left) = self.history.time_left(now) {
            ctx.request_repaint_after(left);
        }
        if self.fitting() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        if ctx.input(|i| i.viewport().close_requested()) && !self.confirm_close() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        }
    }
}
