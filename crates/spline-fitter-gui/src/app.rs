//! The main window: menus, layout, status bar and the frame loop.

use std::path::PathBuf;
use std::time::Instant;

use eframe::egui::{self, Align, Key, KeyboardShortcut, Layout, Modifiers, RichText, Ui};
use spline_fitter::geometry::AirfoilData;
use spline_fitter::models::FitResult;

use crate::dialogs::{Dialog, ReportPanel};
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
    pub sidebar_visible: bool,
    /// Where the sidebar was on the last frame (its fields are settings).
    pub sidebar_rect: egui::Rect,
    /// Put the side panels back to their default widths on the next frame.
    pub reset_sidebar: bool,
    pub reset_explorer: bool,
    /// A submenu button its access key chose: it opens on the next frame.
    open_submenu: Option<egui::Id>,
    /// Ctrl+Enter was pressed: fit once the sidebar has taken its fields.
    fit_requested: bool,
    /// The user chose to lose the unsaved runs for the action now repeated.
    pub discard_confirmed: bool,
    /// The user answered the unsaved-runs question of closing: close.
    pub close_now: bool,
    /// The display choices last written to the settings file.
    saved_display: Option<Display>,
    /// The sidebar settings last written to the settings file.
    saved_settings: Values,
    /// The status text tells of a failure (drawn in the error colour).
    pub status_failed: bool,
    /// A fit `on_fit` prepared: it starts on the next frame.
    pub preparing: Option<Box<(spline_fitter::models::SplineConfig, spline_fitter::models::OptimizationConfig, AirfoilData)>>,
    pub dialog: Option<Dialog>,
    /// The report of the shown fit, for the Report and Shape views.
    pub report_panel: Option<ReportPanel>,
    /// The bottom-left block's view: convergence, report or shape.
    pub bottom_view: usize,
    pub batch_source: String,
    pub batch_target: String,
    /// The main window, which owns the native file pickers (none under test).
    pub parent: Option<crate::parent::ParentWindow>,
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
        let mut explorer = ExplorerState::with_column_names(&store.string("explorer/columns", ""));
        explorer.pivot = store.bool("explorer/pivot", false);
        let (explorer_visible, sidebar_visible) = (store.bool("panels/explorer", true), store.bool("panels/sidebar", true));
        let bottom_view = store.int("view/bottom", 0).clamp(0, 2) as usize;
        let mut settings = Settings::default();
        if let Some(kept) = store.value("sidebar") {
            settings.restore(&Values::from_json(kept, &settings.values()));
        }
        let saved_settings = settings.values();
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
            explorer,
            explorer_visible,
            sidebar_visible,
            sidebar_rect: egui::Rect::NOTHING,
            reset_sidebar: false,
            reset_explorer: false,
            open_submenu: None,
            fit_requested: false,
            discard_confirmed: false,
            close_now: false,
            saved_display: None,
            saved_settings,
            status_failed: false,
            preparing: None,
            dialog: None,
            report_panel: None,
            bottom_view,
            batch_source: String::new(),
            batch_target: String::new(),
            parent: crate::parent::ParentWindow::of(cc),
        };
        app.load_display();
        app.refresh_comparison();
        app
    }

    /// The plot and view choices as kept between sessions.
    fn display(&self) -> Display {
        let m = &self.main_plot;
        Display {
            knots: m.show_knots,
            extrema: m.show_extrema,
            comb: m.comb_on,
            comb_quantity: m.comb.quantity,
            comb_height: m.comb.height,
            comb_density: m.comb.density,
            deviation: m.deviation_on,
            deviation_height: m.deviation_height,
            live: self.live_mode as usize,
            loss_x: self.loss.x_axis,
            loss_log: self.loss.log_mode,
            runs_x: crate::runs::COLUMNS[self.explorer.x_column].name,
            runs_y: crate::runs::COLUMNS[self.explorer.y_column].name,
        }
    }

    /// The display choices of the last session (defaults where none).
    pub(crate) fn load_display(&mut self) {
        let d = self.display();
        let s = &self.store;
        let m = &mut self.main_plot;
        m.show_knots = s.bool("display/knots", d.knots);
        m.show_extrema = s.bool("display/extrema", d.extrema);
        m.comb_on = s.bool("display/comb", d.comb);
        m.comb.quantity = (s.int("display/comb_quantity", d.comb_quantity as i64).max(0) as usize).min(crate::overlays::QUANTITIES.len() - 1);
        m.comb.height = s.float("display/comb_height", d.comb_height).max(0.0);
        m.comb.density = s.int("display/comb_density", d.comb_density as i64).clamp(10, 100_000) as usize;
        m.deviation_on = s.bool("display/deviation", d.deviation);
        m.deviation_height = s.float("display/deviation_height", d.deviation_height).max(0.0);
        self.live_mode = LiveMode::ALL[s.int("display/live", d.live as i64).clamp(0, 2) as usize];
        self.loss.x_axis = s.int("display/loss_x", d.loss_x as i64).clamp(0, 1) as usize;
        self.loss.log_mode = s.int("display/loss_log", d.loss_log as i64).clamp(0, 3) as usize;
        let column = |name: String, default: usize| crate::runs::COLUMNS.iter().position(|c| c.name == name).unwrap_or(default);
        self.explorer.x_column = column(s.string("display/runs_x", d.runs_x), self.explorer.x_column);
        let y = column(s.string("display/runs_y", d.runs_y), self.explorer.y_column);
        if crate::runs::COLUMNS[y].metric {
            self.explorer.y_column = y;
        }
        self.saved_display = Some(self.display());
    }

    /// Writes the sidebar settings when one changed (sweeps are not kept).
    fn save_settings(&mut self) {
        let values = self.settings.values();
        if values != self.saved_settings {
            self.store.set_value("sidebar", values.to_json());
            self.saved_settings = values;
        }
    }

    /// Writes the display choices when one changed.
    fn save_display(&mut self) {
        let d = self.display();
        if self.saved_display.as_ref() == Some(&d) {
            return;
        }
        let s = &mut self.store;
        s.set_bool("display/knots", d.knots);
        s.set_bool("display/extrema", d.extrema);
        s.set_bool("display/comb", d.comb);
        s.set_int("display/comb_quantity", d.comb_quantity as i64);
        s.set_float("display/comb_height", d.comb_height);
        s.set_int("display/comb_density", d.comb_density as i64);
        s.set_bool("display/deviation", d.deviation);
        s.set_float("display/deviation_height", d.deviation_height);
        s.set_int("display/live", d.live as i64);
        s.set_int("display/loss_x", d.loss_x as i64);
        s.set_int("display/loss_log", d.loss_log as i64);
        s.set_string("display/runs_x", d.runs_x);
        s.set_string("display/runs_y", d.runs_y);
        self.saved_display = Some(d);
    }

    pub fn fitting(&self) -> bool {
        self.running.is_some() || self.preparing.is_some()
    }

    pub fn set_status(&mut self, text: impl Into<String>) {
        self.status = text.into();
        self.status_failed = false;
    }

    /// A failure: the status text, in the error colour, and the error chime.
    pub fn fail(&mut self, text: impl Into<String>) {
        self.status = text.into();
        self.status_failed = true;
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

    /// Makes `change` one undo step of its own, however long it takes: a
    /// pending change of the user settles first as its own step.
    pub fn as_one_step(&mut self, change: impl FnOnce(&mut Self)) {
        self.settle_settings();
        self.history.commit(&self.settings.values());
        change(self);
        // The sidebar's rules run for what changed, as for any change.
        let before = self.before.clone();
        self.settings.settle(&before);
        let values = self.settings.values();
        let quiet = self.quiet || self.fitting();
        self.history.on_change(values.clone(), quiet, Instant::now());
        self.history.commit(&values);
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

    /// Keeps a side panel's width between sessions, once the mouse lets go
    /// of its edge (not on every frame of a drag).
    pub fn remember_width(&mut self, ctx: &egui::Context, key: &str, width: f32) {
        if ctx.input(|i| i.pointer.any_down()) || width <= 0.0 {
            return;
        }
        let stored = self.store.float(key, -1.0) as f32;
        if (stored - width).abs() > 0.5 {
            self.store.set_float(key, f64::from(width));
        }
    }

    /// Shows or hides the project explorer; kept between sessions.
    pub fn set_explorer_visible(&mut self, visible: bool) {
        self.explorer_visible = visible;
        self.store.set_bool("panels/explorer", visible);
    }

    /// Shows or hides the sidebar; kept between sessions.
    pub fn set_sidebar_visible(&mut self, visible: bool) {
        self.sidebar_visible = visible;
        if !visible {
            self.sidebar_rect = egui::Rect::NOTHING;
        }
        self.store.set_bool("panels/sidebar", visible);
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
            self.set_explorer_visible(!self.explorer_visible);
        }
        if pressed(ctrl(Key::B)) {
            self.set_sidebar_visible(!self.sidebar_visible);
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
        // Ctrl+Enter fits, also from a settings field: the field gives up
        // the focus, so its text is committed when the sidebar is drawn,
        // and the fit starts after that.
        if pressed(ctrl(Key::Enter)) && !self.fitting() {
            ctx.memory_mut(|m| {
                if let Some(id) = m.focused() {
                    m.surrender_focus(id);
                }
            });
            self.fit_requested = true;
        }
        if self.fitting() && ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
            self.on_cancel();
        }
        // Undo and redo reach the settings also from a focused settings
        // field (the field keeps its own text undo otherwise).
        let in_settings =
            ctx.memory(|m| m.focused()).and_then(|id| ctx.read_response(id)).is_some_and(|r| self.sidebar_rect.contains(r.rect.center()));
        if (!typing || in_settings) && !self.fitting() {
            let (redo, undo) = (pressed(ctrl_shift(Key::Z)) || pressed(ctrl(Key::Y)), pressed(ctrl(Key::Z)));
            if redo || undo {
                widgets::forget_edits(ctx);
                if redo {
                    self.redo();
                } else {
                    self.undo();
                }
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
        // Bar items with 10 px beside and 4 px above their text, 6 px apart
        // and from the edge; menu items 5 px above and about 22 px beside
        // (Python: 20 left, 24 right).
        let bar = egui::style::StyleModifier::new(|style| {
            egui::containers::menu::menu_style(style);
            style.spacing.button_padding = egui::vec2(10.0, 4.0);
            style.spacing.item_spacing.x = size::GAP;
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
        let p = self.palette.clone();
        let mut titles = Vec::new();
        egui::MenuBar::new().style(bar).config(egui::containers::menu::MenuConfig::new().style(items)).ui(ui, |ui| {
            ui.add_space(size::GAP);
            titles.push(bar_menu(ui, "&File", |ui| {
                let has_project = self.projects.file.is_some();
                let entries = [
                    ("&New NACA airfoil...", "Ctrl+N"),
                    ("&Open airfoil...", "Ctrl+O"),
                    ("Open &recent", ""),
                    ("Save fitted airfoil (.dat)...", ""),
                    ("Save spline (.spline.dat)...", ""),
                    ("Export CAD (STEP, IGES, DXF)...", "Ctrl+E"),
                    ("Open &project...", "Ctrl+Shift+O"),
                    ("&Save project", "Ctrl+S"),
                    ("Save project &as...", ""),
                    ("&Batch convert...", ""),
                    ("E&xit", "Alt+F4"),
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
                    let id = ui.next_auto_id();
                    if self.open_submenu == Some(id) {
                        // Chosen by its access key: as if Enter were pressed on it.
                        self.open_submenu = None;
                        ui.memory_mut(|m| m.request_focus(id));
                        ui.input_mut(|i| {
                            i.events.push(egui::Event::Key {
                                key: Key::Enter,
                                physical_key: None,
                                pressed: true,
                                repeat: false,
                                modifiers: Modifiers::NONE,
                            })
                        });
                    }
                    let text = access_key(entries[2].0).0;
                    let (button, _) =
                        egui::containers::menu::SubMenuButton::from_button(egui::Button::new(text)).ui(ui, |ui| self.recent_menu(ui));
                    underline_button(ui, &button, entries[2].0, false);
                    submenu_arrow(ui, &p, button.rect);
                    if ui.is_enabled() && access_key_pressed(ui, entries[2].0) {
                        self.open_submenu = Some(button.id);
                        ui.ctx().request_repaint();
                    }
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
            }));
            titles.push(bar_menu(ui, "&Edit", |ui| {
                let entries = [("&Undo", "Ctrl+Z"), ("&Redo", "Ctrl+Y"), ("Restore &deleted runs", ""), ("&Paste airfoil", "Ctrl+V")];
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
            }));
            titles.push(bar_menu(ui, "&View", |ui| {
                let entries = [
                    ("&Fit report", "Ctrl+R"),
                    ("Project &explorer", "Ctrl+Shift+E"),
                    ("&Sidebar", "Ctrl+B"),
                    ("&Theme...", ""),
                    ("&Scrolling and zooming...", ""),
                ];
                let m = MenuLayout::of(ui, &entries, true);
                if m.item(ui, &p, entries[0], has_fit, None) {
                    self.on_report();
                }
                menu_separator(ui);
                if m.item(ui, &p, entries[1], true, Some(self.explorer_visible)) {
                    self.set_explorer_visible(!self.explorer_visible);
                }
                if m.item(ui, &p, entries[2], true, Some(self.sidebar_visible)) {
                    self.set_sidebar_visible(!self.sidebar_visible);
                }
                menu_separator(ui);
                if m.item(ui, &p, entries[3], true, None) {
                    self.dialog = Some(Dialog::theme(self));
                }
                if m.item(ui, &p, entries[4], true, None) {
                    self.dialog = Some(Dialog::motion(self));
                }
            }));
            titles.push(bar_menu(ui, "&Help", |ui| {
                let entries = [("&About", "")];
                let m = MenuLayout::of(ui, &entries, false);
                if m.item(ui, &p, entries[0], true, None) {
                    self.set_status(format!("{} {}", spline_fitter::APP_NAME, spline_fitter::VERSION));
                }
            }));
        });
        // Alt and a menu's access key opens the menu.
        for (title, button) in titles {
            let pressed = access_key_code(title).is_some_and(|key| ui.input_mut(|i| i.consume_key(Modifiers::ALT, key)));
            if pressed {
                egui::Popup::open_id(ui.ctx(), egui::Popup::default_response_id(&button));
                ui.ctx().request_repaint();
            }
        }
    }

    fn recent_menu(&mut self, ui: &mut Ui) {
        let files = self.store.recent_files();
        if files.is_empty() {
            ui.add_enabled(false, egui::Button::new("No recent files"));
            return;
        }
        for (i, file) in files.iter().enumerate() {
            let name = std::path::Path::new(file).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            // The first nine are numbered, the number their access key.
            let name = name.replace('&', "&&");
            let text = if i < 9 { format!("&{} {name}", i + 1) } else { name };
            let button = ui.button(access_key(&text).0).on_hover_text(file);
            underline_button(ui, &button, &text, false);
            if button.clicked() || access_key_pressed(ui, &text) {
                ui.close();
                self.open_path(std::path::Path::new(file));
            }
        }
        menu_separator(ui);
        if ui.button("Clear recent files").clicked() {
            self.store.set_recent_files(&[]);
        }
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        let p = &self.palette;
        let margin = egui::Margin { left: 8, right: 8, top: 2, bottom: 3 };
        egui::Panel::bottom("status").frame(egui::Frame::new().fill(p.window).inner_margin(margin)).show(ui, |ui| {
            let color = if self.status_failed { p.error } else { p.text_2 };
            ui.label(RichText::new(&self.status).color(color));
            // The size grip in the window's corner (none when maximized).
            if !ui.input(|i| i.viewport().maximized.unwrap_or(false)) {
                let corner = ui.max_rect().right_bottom() + egui::vec2(margin.right as f32, margin.bottom as f32);
                widgets::size_grip(ui, p, corner);
            }
        });
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
            // Toolbar, main plot and the bottom plots a gap apart; the plots
            // share the height 3 : 2.
            ui.spacing_mut().item_spacing = egui::vec2(size::GAP, size::GAP);
            self.plot_toolbar(ui);
            let total = ui.available_height();
            let main_height = (total - size::GAP) * 0.6;
            ui.allocate_ui(egui::vec2(ui.available_width(), main_height), |ui| self.main_plot.show(ui, &p, self.zoom_ms));
            let rest = ui.available_height();
            let wide = ui.available_width() >= 2.0 * 440.0 + size::GROUP;
            // The bottom plots are a group's space apart.
            let between = size::GROUP - size::GAP;
            if wide {
                let half = (ui.available_width() - size::GROUP) / 2.0;
                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(half, rest), |ui| self.loss_block(ui));
                    ui.add_space(between);
                    ui.allocate_ui(egui::vec2(half, rest), |ui| self.comparison_block(ui));
                });
            } else {
                let half = (rest - size::GROUP) / 2.0;
                ui.allocate_ui(egui::vec2(ui.available_width(), half), |ui| self.loss_block(ui));
                ui.add_space(between);
                ui.allocate_ui(egui::vec2(ui.available_width(), half), |ui| self.comparison_block(ui));
            }
        });
    }

    fn plot_toolbar(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        // The live preview setting at the right of the display toggles, or
        // under them when the plot is too narrow for both.
        let font = egui::FontId::proportional(crate::theme::typeface::BASE);
        let live_width = crate::fonts::layout_width(ui, "Live preview", &font, false) + size::GAP + widgets::combo_width(ui, &LiveMode::LABELS);
        let mut wraps = false;
        ui.horizontal(|ui| {
            let knots =
                widgets::toggle_button(ui, &p, "Knots", self.main_plot.show_knots, "Mark the curve points at the knots of every section.");
            if knots.clicked() {
                self.main_plot.show_knots = !self.main_plot.show_knots;
                self.main_plot.refresh_knots();
            }
            crate::dialogs::comb_toggle(self, ui);
            let tip = "Mark where the curvature has a maximum or a minimum,\nthe curvature extrema the fit report counts.";
            if widgets::toggle_button(ui, &p, "Curvature extrema", self.main_plot.show_extrema, tip).clicked() {
                self.main_plot.show_extrema = !self.main_plot.show_extrema;
                self.main_plot.refresh_extrema();
            }
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
        let label = |ui: &mut Ui| widgets::toolbar_label(ui, p, "Live preview");
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

/// The plot and view choices kept between sessions: the plot toggles and
/// their settings, the live preview, the convergence axes and the runs
/// plot's columns.
#[derive(Debug, Clone, PartialEq)]
struct Display {
    knots: bool,
    extrema: bool,
    comb: bool,
    comb_quantity: usize,
    comb_height: f64,
    comb_density: usize,
    deviation: bool,
    deviation_height: f64,
    live: usize,
    loss_x: usize,
    loss_log: usize,
    runs_x: &'static str,
    runs_y: &'static str,
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
    /// The check column moves the text 14 px and widens the menu 18 px.
    const CHECK_COLUMN: f32 = 14.0;
    const CHECK_WIDTH: f32 = 18.0;
    /// An entry with a shortcut is this much wider (Qt's tab).
    const TAB: f32 = 12.0;
    const ROW: f32 = 26.0;

    /// Qt's rule: the widest entry (its text, padding, and the tab when it
    /// has a shortcut) and the widest shortcut.
    fn of(ui: &Ui, entries: &[(&str, &str)], checkable: bool) -> Self {
        let font = egui::FontId::proportional(typeface::BASE);
        let width = |t: &str| crate::fonts::layout_width(ui, &access_key(t).0, &font, false);
        let check = if checkable { Self::CHECK_WIDTH } else { 0.0 };
        let entry = entries
            .iter()
            .map(|(t, s)| Self::PAD_LEFT + width(t) + Self::PAD_RIGHT + check + if s.is_empty() { 0.0 } else { Self::TAB })
            .fold(0.0, f32::max);
        let shortcut = entries.iter().map(|(_, s)| width(s)).fold(0.0, f32::max);
        let width = entry + shortcut;
        let text_x = Self::PAD_LEFT + if checkable { Self::CHECK_COLUMN } else { 0.0 };
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
            crate::theme::Glyph::MenuCheck.paint(ui.painter(), egui::pos2(rect.left() + 7.0, rect.center().y + 0.5), 12.0, color);
        }
        paint_access_text(ui, egui::pos2(rect.left() + self.text_x, rect.center().y), text, color);
        response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, enabled, access_key(text).0));
        if !shortcut.is_empty() {
            ui.painter().text(egui::pos2(rect.left() + self.shortcut_x, rect.center().y), egui::Align2::LEFT_CENTER, shortcut, font, color);
        }
        let clicked = enabled && (response.clicked() || access_key_pressed(ui, text));
        if clicked {
            ui.close();
        }
        clicked
    }
}

/// A menu text with its access key marked by `&` ("E&xit"; "&&" is an
/// ampersand): the text without the marks, and the access key with its
/// index among the text's characters.
fn access_key(text: &str) -> (String, Option<(usize, char)>) {
    let mut plain = String::new();
    let mut key = None;
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '&' {
            plain.push(c);
            continue;
        }
        match chars.next() {
            Some('&') => plain.push('&'),
            Some(k) => {
                key = key.or(Some((plain.chars().count(), k)));
                plain.push(k);
            }
            None => {}
        }
    }
    (plain, key)
}

/// The key that presses an entry's access key.
fn access_key_code(text: &str) -> Option<Key> {
    access_key(text).1.and_then(|(_, c)| Key::from_name(&c.to_uppercase().to_string()))
}

/// Underlines the access key of a menu text laid out from `left_center`,
/// 1 px below the baseline as Qt draws it (always, in this window's style).
fn underline_access_key(ui: &Ui, left_center: egui::Pos2, text: &str, color: egui::Color32) {
    let (plain, Some((index, _))) = access_key(text) else { return };
    let galley = ui.painter().layout_no_wrap(plain, egui::FontId::proportional(typeface::BASE), color);
    let Some(row) = galley.rows.first() else { return };
    let Some(glyph) = row.glyphs.get(index) else { return };
    let top = left_center.y - galley.size().y / 2.0;
    let baseline = (top + row.pos.y + glyph.pos.y).round();
    let left = (left_center.x + row.pos.x + glyph.pos.x).round();
    let line = egui::Rect::from_min_size(egui::pos2(left, baseline + 1.0), egui::vec2(glyph.advance_width.round(), 1.0));
    ui.painter().rect_filled(line, 0.0, color);
}

/// Paints a menu text from `left_center`, its access key underlined.
fn paint_access_text(ui: &Ui, left_center: egui::Pos2, text: &str, color: egui::Color32) {
    let font = egui::FontId::proportional(typeface::BASE);
    ui.painter().text(left_center, egui::Align2::LEFT_CENTER, access_key(text).0, font, color);
    underline_access_key(ui, left_center, text, color);
}

/// A menu button (or submenu or entry button) showing `text` without its
/// marks: the access key's underline goes on top, the label to assistive
/// technology.
fn underline_button(ui: &Ui, response: &egui::Response, text: &str, centered: bool) {
    let plain = access_key(text).0;
    let color = if ui.is_enabled() { ui.visuals().text_color() } else { ui.visuals().weak_text_color() };
    let width = ui.painter().layout_no_wrap(plain.clone(), egui::FontId::proportional(typeface::BASE), color).size().x;
    let rect = response.rect;
    let left = if centered { rect.center().x - width / 2.0 } else { rect.left() + ui.spacing().button_padding.x };
    underline_access_key(ui, egui::pos2(left, rect.center().y), text, color);
}

/// Whether the access key of an entry of the open menu was pressed (alone
/// or with Alt); the key press is used up.
fn access_key_pressed(ui: &Ui, text: &str) -> bool {
    let Some(key) = access_key_code(text) else { return false };
    ui.input_mut(|i| i.consume_key(Modifiers::NONE, key) || i.consume_key(Modifiers::ALT, key))
}

/// A menu of the menu bar: its title (the access key marked) as wide as Qt
/// makes it; the title and the title's response.
fn bar_menu<'t>(ui: &mut Ui, title: &'t str, content: impl FnOnce(&mut Ui)) -> (&'t str, egui::Response) {
    let font = egui::FontId::proportional(typeface::BASE);
    let plain = access_key(title).0;
    let width = crate::fonts::layout_width(ui, &plain, &font, false) + 2.0 * ui.spacing().button_padding.x;
    let button = egui::Button::new(plain).min_size(egui::vec2(width, 0.0));
    let (response, _) = egui::containers::menu::MenuButton::from_button(button).ui(ui, content);
    underline_button(ui, &response, title, true);
    (title, response)
}

/// The arrow of a submenu entry: a small triangle near the menu's edge.
fn submenu_arrow(ui: &Ui, p: &crate::theme::Palette, rect: egui::Rect) {
    let color = if ui.is_enabled() { p.text_2 } else { p.text_3 };
    let tip = egui::pos2(rect.right() - 5.5, rect.center().y);
    let points = vec![tip + egui::vec2(-4.5, -4.5), tip + egui::vec2(-4.5, 4.5), tip];
    ui.painter().add(egui::Shape::convex_polygon(points, color, egui::Stroke::NONE));
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
        if self.sidebar_visible {
            self.sidebar(ui);
        } else {
            self.sidebar_rail(ui);
        }
        if self.explorer_visible {
            self.explorer_panel(ui);
        } else {
            self.explorer_rail(ui);
        }
        self.central(ui);
        self.show_dialogs(&ctx);

        self.settle_settings();
        self.save_settings();
        self.save_display();
        if std::mem::take(&mut self.fit_requested) {
            self.on_fit();
        }
        let now = Instant::now();
        let current = self.settings.values();
        self.history.tick(now, &current);
        if let Some(left) = self.history.time_left(now) {
            ctx.request_repaint_after(left);
        }
        if self.fitting() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        if std::mem::take(&mut self.close_now) {
            if let Some(r) = &self.running {
                r.cancel();
            }
            self.projects.modified = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else if ctx.input(|i| i.viewport().close_requested()) && !self.confirm_close() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        }
    }
}
