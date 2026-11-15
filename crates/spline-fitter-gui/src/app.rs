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
use crate::theme::{self, size, typeface, Palette, ThemeMode};
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
        // Bar items with 10 px beside and 4 px above their text; menu items
        // 5 px above and about 22 px beside (Python: 20 left, 24 right).
        let bar = egui::style::StyleModifier::new(|style| {
            egui::containers::menu::menu_style(style);
            style.spacing.button_padding = egui::vec2(10.0, 4.0);
            style.spacing.item_spacing.x = 0.0;
        });
        // Menus on the field grey, 4 px above and below their items.
        let field = self.palette.field;
        let items = egui::style::StyleModifier::new(move |style| {
            egui::containers::menu::menu_style(style);
            style.spacing.button_padding = egui::vec2(22.0, 5.0);
            style.spacing.item_spacing.y = 0.0;
            style.spacing.menu_margin = egui::Margin::symmetric(0, 4);
            style.visuals.window_fill = field;
            style.visuals.menu_corner_radius = egui::CornerRadius::ZERO;
        });
        egui::MenuBar::new().style(bar).config(egui::containers::menu::MenuConfig::new().style(items)).ui(ui, |ui| {
            let p = self.palette.clone();
            ui.menu_button("File", |ui| {
                let has_project = self.projects.file.is_some();
                let entries = [
                    ("New NACA airfoil...", "Ctrl+N"),
                    ("Open airfoil...", "Ctrl+O"),
                    ("Open recent", ""),
                    ("Save fitted airfoil (.dat)...", ""),
                    ("Save spline (.spline.dat)...", ""),
                    ("Export CAD (STEP, IGES, DXF)...", "Ctrl+E"),
                    ("Open project...", "Ctrl+Shift+O"),
                    ("Save project", "Ctrl+S"),
                    ("Save project as...", ""),
                    ("Batch convert...", ""),
                    ("Exit", "Alt+F4"),
                ];
                let m = MenuLayout::of(ui, &entries, false);
                if m.item(ui, &p, entries[0], !fitting, None) {
                    self.on_new_naca();
                }
                if m.item(ui, &p, entries[1], !fitting, None) {
                    self.on_open_airfoil();
                }
                ui.add_enabled_ui(!fitting, |ui| {
                    ui.spacing_mut().button_padding.x = m.text_x;
                    ui.menu_button("Open recent", |ui| self.recent_menu(ui));
                });
                menu_separator(ui);
                if m.item(ui, &p, entries[3], has_fit, None) {
                    self.on_save_airfoil();
                }
                if m.item(ui, &p, entries[4], has_fit, None) {
                    self.on_save_spline();
                }
                if m.item(ui, &p, entries[5], has_fit, None) {
                    self.on_export_cad();
                }
                menu_separator(ui);
                if m.item(ui, &p, entries[6], !fitting, None) {
                    self.on_open_project();
                }
                if m.item(ui, &p, entries[7], has_project, None) {
                    self.on_save_project();
                }
                if m.item(ui, &p, entries[8], has_project, None) {
                    self.on_save_project_as();
                }
                menu_separator(ui);
                if m.item(ui, &p, entries[9], !fitting, None) {
                    self.dialog = Some(Dialog::batch());
                }
                menu_separator(ui);
                if m.item(ui, &p, entries[10], true, None) {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("Edit", |ui| {
                let entries = [("Undo", "Ctrl+Z"), ("Redo", "Ctrl+Y"), ("Restore deleted runs", ""), ("Paste airfoil", "Ctrl+V")];
                let m = MenuLayout::of(ui, &entries, false);
                if m.item(ui, &p, entries[0], !fitting && self.history.can_undo(), None) {
                    self.undo();
                }
                if m.item(ui, &p, entries[1], !fitting && self.history.can_redo(), None) {
                    self.redo();
                }
                if m.item(ui, &p, entries[2], !fitting && !self.projects.deleted.is_empty(), None) {
                    self.restore_deleted_runs();
                }
                menu_separator(ui);
                if m.item(ui, &p, entries[3], !fitting, None) {
                    self.on_paste();
                }
            });
            ui.menu_button("View", |ui| {
                let entries =
                    [("Fit report...", "Ctrl+R"), ("Project explorer", "Ctrl+Shift+E"), ("Theme...", ""), ("Scrolling and zooming...", "")];
                let m = MenuLayout::of(ui, &entries, true);
                if m.item(ui, &p, entries[0], has_fit, None) {
                    self.on_report();
                }
                menu_separator(ui);
                if m.item(ui, &p, entries[1], true, Some(self.explorer_visible)) {
                    self.explorer_visible = !self.explorer_visible;
                }
                menu_separator(ui);
                if m.item(ui, &p, entries[2], true, None) {
                    self.dialog = Some(Dialog::theme(self));
                }
                if m.item(ui, &p, entries[3], true, None) {
                    self.dialog = Some(Dialog::motion(self));
                }
            });
            ui.menu_button("Help", |ui| {
                let entries = [("About", "")];
                let m = MenuLayout::of(ui, &entries, false);
                if m.item(ui, &p, entries[0], true, None) {
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
        menu_separator(ui);
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

/// The columns of a menu as Qt lays them out: the text 20 px in (and a
/// check column more when an item can be checked), the shortcuts in a
/// column of their own at the right, 24 px from the edge.
struct MenuLayout {
    text_x: f32,
    shortcut_x: f32,
    width: f32,
}

impl MenuLayout {
    const PAD_LEFT: f32 = 20.0;
    const PAD_RIGHT: f32 = 24.0;
    const CHECK_COLUMN: f32 = 14.0;
    /// Between an item's text and the shortcut column.
    const TAB: f32 = 11.0;
    const ROW: f32 = 26.0;

    fn of(ui: &Ui, entries: &[(&str, &str)], checkable: bool) -> Self {
        let font = egui::FontId::proportional(typeface::BASE);
        let width = |t: &str| ui.painter().layout_no_wrap(t.to_string(), font.clone(), ui.visuals().text_color()).size().x.ceil();
        let text_x = Self::PAD_LEFT + if checkable { Self::CHECK_COLUMN } else { 0.0 };
        let text = entries.iter().map(|(t, _)| width(t)).fold(0.0, f32::max);
        let shortcut = entries.iter().map(|(_, s)| width(s)).fold(0.0, f32::max);
        let tab = if shortcut > 0.0 { Self::TAB + shortcut } else { 0.0 };
        let width = text_x + text + tab + Self::PAD_RIGHT;
        MenuLayout { text_x, shortcut_x: width - Self::PAD_RIGHT - shortcut, width }
    }

    /// An item (checked or not when `check` is given); true when clicked,
    /// which closes the menu.
    fn item(&self, ui: &mut Ui, p: &crate::theme::Palette, (text, shortcut): (&str, &str), enabled: bool, check: Option<bool>) -> bool {
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(self.width, Self::ROW), if enabled { egui::Sense::click() } else { egui::Sense::hover() });
        if enabled && response.hovered() {
            ui.painter().rect_filled(rect, 0.0, p.select);
        }
        let color = if enabled { p.text } else { p.text_3 };
        let font = egui::FontId::proportional(typeface::BASE);
        if check == Some(true) {
            crate::theme::Glyph::Check.paint(ui.painter(), egui::pos2(rect.left() + 10.0, rect.center().y), size::GLYPH, color);
        }
        ui.painter().text(egui::pos2(rect.left() + self.text_x, rect.center().y), egui::Align2::LEFT_CENTER, text, font.clone(), color);
        if !shortcut.is_empty() {
            ui.painter().text(egui::pos2(rect.left() + self.shortcut_x, rect.center().y), egui::Align2::LEFT_CENTER, shortcut, font, color);
        }
        let clicked = enabled && response.clicked();
        if clicked {
            ui.close();
        }
        clicked
    }
}

/// A menu separator: a 1 px rule across the menu, 4 px above and below.
fn menu_separator(ui: &mut Ui) {
    ui.add(egui::Separator::default().spacing(9.0).grow(0.0));
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // Behind a dialog (a window of its own) the main window takes no input.
        if self.modal_open() && !ctx.embed_viewports() {
            ctx.input_mut(|i| {
                i.pointer = Default::default();
                i.events.clear();
                i.keys_down.clear();
                i.smooth_scroll_delta = egui::Vec2::ZERO;
            });
        }
        self.apply_theme(&ctx);
        self.tick_job(&ctx);
        self.shortcuts(&ctx);
        self.dropped_files(&ctx);
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(self.title()));

        let p = self.palette.clone();
        let menu = egui::Panel::top("menu").exact_size(size::MENU_BAR_HEIGHT).frame(egui::Frame::new().fill(p.window)).show(ui, |ui| {
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
