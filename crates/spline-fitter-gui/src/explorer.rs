//! The project explorer: the runs of the project in a filterable table,
//! their details, and the comparison plot of the shown runs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use eframe::egui::{self, Align, Layout, RichText, Sense, Ui};
use egui_extras::{Column as TableColumn, TableBuilder};
use spline_fitter::bspline_varpro::KnotPlacement;
use spline_fitter::formatting::{format_duration, number};
use spline_fitter::models::{FitResult, IterationRecord, OptimizationConfig, SplineConfig};
use spline_fitter::project::{number_runs, OptimizationRun, ProjectFile};

use crate::app::App;
use crate::runs::{compare, parse_filter, row, term_matches, Cell, Filters, Term, COLUMNS, CONTROL_POINTS_COLUMN, ERROR_COLUMN};
use crate::theme::{size, Glyph, Palette};
use crate::widgets::{self, caption, muted};

/// The project of the window.
#[derive(Default)]
pub struct Projects {
    pub file: Option<ProjectFile>,
    pub path: Option<PathBuf>,
    pub modified: bool,
    /// Deleted runs with their positions, latest last.
    pub deleted: Vec<Vec<(usize, OptimizationRun)>>,
    /// The highest number given to a run.
    pub last_number: usize,
    /// Selected runs by id.
    pub selected: BTreeSet<String>,
    /// The run whose fit is shown.
    pub shown_run: Option<String>,
}

impl Projects {
    pub fn runs(&self) -> &[OptimizationRun] {
        self.file.as_ref().map_or(&[], |f| &f.runs)
    }

    pub fn run(&self, id: &str) -> Option<&OptimizationRun> {
        self.runs().iter().find(|r| r.run_id == id)
    }

    pub fn shown(&self) -> Option<&OptimizationRun> {
        self.shown_run.as_deref().and_then(|id| self.run(id))
    }

    /// The file name, or `Unsaved project`.
    pub fn name(&self) -> String {
        self.path.as_ref().and_then(|p| p.file_name()).map_or_else(|| "Unsaved project".into(), |n| n.to_string_lossy().into_owned())
    }

    /// The explorer's label.
    pub fn label(&self) -> String {
        match &self.file {
            None => "No project".into(),
            Some(f) => {
                let k = f.runs.len();
                format!("{}{}\n{k} optimization run{}", self.name(), if self.modified { "*" } else { "" }, if k == 1 { "" } else { "s" })
            }
        }
    }
}

/// Whether a run has the settings of a new fit (19.16.4): with free knots
/// also the knot spacing, with a tolerance the same control points in every
/// section (each refit of a search is a run of its own).
fn same_settings(run: &OptimizationRun, config: &SplineConfig, opt: &OptimizationConfig, fit: &FitResult) -> bool {
    let (c, o) = (&run.spline_config, &run.optimization_config);
    c.degree == config.degree
        && c.num_control_points == config.num_control_points
        && c.num_sections == config.num_sections
        && c.continuity == config.continuity
        && c.geometric_continuity == config.geometric_continuity
        && c.junction_continuity == config.junction_continuity
        && o.tolerance == opt.tolerance
        && o.knot_placement == opt.knot_placement
        && o.parameterization == opt.parameterization
        && o.optimize_le_position == opt.optimize_le_position
        && o.grow_control_points == opt.grow_control_points
        && o.optimizer == opt.optimizer
        && (opt.knot_placement != KnotPlacement::Free || o.min_knot_gap == opt.min_knot_gap)
        && (opt.tolerance.is_some() || o.smoothing_weight == opt.smoothing_weight)
        && (opt.tolerance.is_none() || run.result.section_counts() == fit.section_counts())
}

/// View state of the explorer.
pub struct ExplorerState {
    pub filters: Filters,
    pub pivot: bool,
    /// Sort column and whether ascending.
    pub sort: (usize, bool),
    /// The comparison plot's axes (column indices).
    pub x_column: usize,
    pub y_column: usize,
    /// Run ids of the plotted points, in plot order, with their coordinates.
    points: Option<Vec<(String, [f64; 2])>>,
    anchor: Option<String>,
    popup: Option<ColumnPopup>,
    /// Height of the run details on the last frame (the table takes the rest).
    details_height: f32,
    /// The columns in the order shown (header sections can be dragged).
    order: Vec<usize>,
    /// Where each header section was on the last frame, in shown order.
    sections: Vec<(usize, egui::Rect)>,
}

impl Default for ExplorerState {
    fn default() -> Self {
        ExplorerState {
            filters: Filters::default(),
            pivot: false,
            sort: (0, true),
            x_column: CONTROL_POINTS_COLUMN,
            y_column: ERROR_COLUMN,
            points: None,
            anchor: None,
            popup: None,
            details_height: 15.0,
            order: (0..COLUMNS.len()).collect(),
            sections: Vec::new(),
        }
    }
}

impl ExplorerState {
    /// Closes a column's drop-down.
    #[cfg(test)]
    pub fn close_popup(&mut self) {
        self.popup = None;
    }

    /// Forgets the plotted points, so the next update redraws the runs.
    pub fn forget_points(&mut self) {
        self.points = None;
    }
}

/// Run table: header and row heights, and what a header and a cell add to
/// their text (padding, and the header's sort and filter glyphs). Qt sizes
/// a header section by its text in bold.
const HEADER_HEIGHT: f32 = 28.0;
const ROW_HEIGHT: f32 = 28.0;
const HEADER_EXTRA: f32 = 46.0;
const CELL_EXTRA: f32 = 18.0;
const EMPTY_COLUMN: f32 = 100.0;

/// The open column filter drop-down.
struct ColumnPopup {
    column: usize,
    excluded: BTreeSet<String>,
    condition: String,
    /// The header section it drops down from (a point: right of a row name).
    anchor: egui::Rect,
    /// The condition field has had the focus it gets on opening.
    focused: bool,
}

/// The column drop-down's content width: as wide as a Qt list box wants
/// (the drop-down at least `POPUP_MIN_WIDTH`).
const POPUP_CONTENT_WIDTH: f32 = 258.0;
/// A value row of its list: a line of text and 2 px above and below, and
/// the 1 px between rows.
const POPUP_ROW: f32 = 21.0;

impl App {
    /// The visible runs in table order, with their cells.
    fn visible_rows(&self) -> Vec<(String, Vec<(Cell, String)>)> {
        let mut rows: Vec<(String, Vec<(Cell, String)>)> = self
            .projects
            .runs()
            .iter()
            .map(|r| (r.run_id.clone(), row(r)))
            .filter(|(_, cells)| self.explorer.filters.shows(cells))
            .collect();
        let (column, ascending) = self.explorer.sort;
        rows.sort_by(|a, b| {
            let (x, y) = (&a.1[column], &b.1[column]);
            match (x.0 == Cell::Empty, y.0 == Cell::Empty) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => {
                    let o = compare(x, y);
                    if ascending {
                        o
                    } else {
                        o.reverse()
                    }
                }
            }
        });
        rows
    }

    /// Adds a fit as a run (replacing the run of the same settings); false
    /// when a cancelled fit would replace a finished run.
    pub fn add_run(
        &mut self,
        config: &SplineConfig,
        opt: &OptimizationConfig,
        fit: FitResult,
        history: Vec<IterationRecord>,
        replot: bool,
    ) -> bool {
        let Some(airfoil) = self.airfoil.clone() else { return false };
        let existing = self.projects.runs().iter().position(|r| same_settings(r, config, opt, &fit));
        if fit.cancelled && existing.is_some_and(|i| !self.projects.runs()[i].result.cancelled) {
            return false;
        }
        let file = self.projects.file.get_or_insert_with(|| ProjectFile::new(airfoil));
        let mut run = OptimizationRun::new(config.clone(), opt.clone(), fit, history);
        let id = run.run_id.clone();
        match existing {
            Some(i) => {
                run.number = file.runs[i].number;
                file.runs[i] = run;
            }
            None => {
                file.runs.push(run);
                self.projects.last_number = number_runs(&mut file.runs, self.projects.last_number);
            }
        }
        self.projects.shown_run = Some(id);
        self.projects.modified = true;
        self.retain_selection();
        if replot {
            self.explorer.forget_points();
            self.refresh_comparison();
        }
        true
    }

    fn retain_selection(&mut self) {
        let ids: BTreeSet<String> = self.projects.runs().iter().map(|r| r.run_id.clone()).collect();
        self.projects.selected.retain(|id| ids.contains(id));
    }

    /// Sets the sidebar to a run's settings (one undo step).
    pub fn apply_run_settings(&mut self, run: &OptimizationRun) {
        self.as_one_step(|app| app.settings.apply_run(run));
    }

    /// Shows a run's fit and history.
    pub fn show_run(&mut self, run: &OptimizationRun) {
        self.fit = Some(run.result.clone());
        self.projects.shown_run = Some(run.run_id.clone());
        self.loss.set_data(run.loss_history.iter().map(|r| (r.iteration, r.elapsed_seconds, r.error)).collect());
        self.show_current_fit();
    }

    fn load_settings(&mut self, id: &str) {
        if self.fitting() {
            return self.set_status("Cancel the running fit to load a run");
        }
        let Some(run) = self.projects.run(id).cloned() else { return };
        self.apply_run_settings(&run);
        self.show_run(&run);
        self.set_status(format!(
            "Applied the settings of run {} ({}, degree {}, {} control points, {} sections)",
            run.number,
            run.optimization_config.optimizer.display_name(),
            run.spline_config.degree,
            run.spline_config.num_control_points,
            run.spline_config.num_sections
        ));
    }

    pub fn delete_selected_runs(&mut self) {
        let Some(file) = self.projects.file.as_mut() else { return };
        let selected = std::mem::take(&mut self.projects.selected);
        let mut removed = Vec::new();
        let mut i = 0;
        while i < file.runs.len() {
            if selected.contains(&file.runs[i].run_id) {
                removed.push((i + removed.len(), file.runs.remove(i)));
            } else {
                i += 1;
            }
        }
        let k = removed.len();
        if self.projects.shown_run.as_ref().is_some_and(|id| selected.contains(id)) {
            self.fit = None;
            self.projects.shown_run = None;
            self.main_plot.clear_items();
            self.loss.reset();
        }
        self.projects.deleted.push(removed);
        self.projects.modified = true;
        self.explorer.forget_points();
        self.refresh_comparison();
        self.set_status(format!("Deleted {k} run{}; Edit > Restore deleted runs brings them back", if k == 1 { "" } else { "s" }));
    }

    pub fn restore_deleted_runs(&mut self) {
        let (Some(batch), Some(file)) = (self.projects.deleted.pop(), self.projects.file.as_mut()) else { return };
        let k = batch.len();
        self.projects.selected.clear();
        for (index, run) in batch {
            self.projects.selected.insert(run.run_id.clone());
            let at = index.min(file.runs.len());
            file.runs.insert(at, run);
        }
        self.projects.modified = true;
        self.explorer.forget_points();
        self.refresh_comparison();
        self.set_status(format!("Restored {k} run{}", if k == 1 { "" } else { "s" }));
    }

    /// Redraws the comparison plot from the visible runs.
    pub fn refresh_comparison(&mut self) {
        let (xc, yc) = (self.explorer.x_column, self.explorer.y_column);
        let rows = self.visible_rows();
        let mut points: Vec<(String, Cell, String, f64)> = rows
            .into_iter()
            .filter_map(|(id, cells)| {
                let (x, xt) = cells[xc].clone();
                let y = cells[yc].0.number()?;
                (x != Cell::Empty).then_some((id, x, xt, y))
            })
            .collect();
        let categorical = points.iter().any(|(_, x, _, _)| !x.is_number());
        let mut ticks = None;
        let mut xy: Vec<(String, [f64; 2])> = if categorical {
            let mut labels: Vec<String> = points.iter().map(|(_, _, t, _)| t.clone()).collect::<BTreeSet<_>>().into_iter().collect();
            labels.sort_by_key(|l| l.to_lowercase());
            ticks = Some(labels.iter().enumerate().map(|(i, l)| (i as f64, l.clone())).collect::<Vec<_>>());
            points.drain(..).map(|(id, _, t, y)| (id, [labels.iter().position(|l| *l == t).unwrap() as f64, y])).collect()
        } else {
            points.drain(..).map(|(id, x, _, y)| (id, [x.number().unwrap(), y])).collect()
        };
        xy.sort_by(|a, b| a.1[0].total_cmp(&b.1[0]).then(a.1[1].total_cmp(&b.1[1])));
        let c = &mut self.comparison;
        c.x_label = COLUMNS[xc].name.into();
        c.y_label = COLUMNS[yc].name.into();
        c.log_x = COLUMNS[xc].log && ticks.is_none() && xy.iter().all(|p| p.1[0] > 0.0);
        c.log_y = COLUMNS[yc].log && xy.iter().all(|p| p.1[1] > 0.0);
        c.ticks = ticks;
        let unique = xy.windows(2).all(|w| w[0].1[0] != w[1].1[0]);
        c.set(xy.iter().map(|p| p.1).collect(), unique);
        self.explorer.points = Some(xy);
        self.mark_selection();
    }

    fn mark_selection(&mut self) {
        if self.explorer.points.is_none() {
            return self.refresh_comparison();
        }
        let points = self.explorer.points.as_ref().unwrap();
        self.comparison.marked = points.iter().filter(|(id, _)| self.projects.selected.contains(id)).map(|(_, p)| *p).collect();
    }

    fn on_selection_changed(&mut self) {
        self.mark_selection();
        if self.projects.selected.len() == 1 && !self.fitting() {
            let id = self.projects.selected.iter().next().unwrap().clone();
            if let Some(run) = self.projects.run(&id).cloned() {
                self.show_run(&run);
            }
        }
    }

    pub fn explorer_rail(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        egui::Panel::right("explorer_rail")
            .resizable(false)
            .exact_size(size::CONTROL_HEIGHT + 8.0)
            .frame(egui::Frame::new().fill(p.panel).inner_margin(egui::Margin { left: 4, right: 4, top: 10, bottom: 12 }))
            .show(ui, |ui| {
                if widgets::flat_icon_button(ui, &p, Glyph::ChevronLeft, "Show the project explorer (Ctrl+Shift+E)").clicked() {
                    self.explorer_visible = true;
                }
            });
    }

    pub fn explorer_panel(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let mut panel = egui::Panel::right("explorer")
            .resizable(true)
            .default_size(size::EXPLORER_WIDTH + size::SPLITTER)
            .size_range(size::EXPLORER_MIN_WIDTH..=ui.available_width() * 0.6)
            .frame(egui::Frame::new().fill(p.panel).inner_margin(egui::Margin { left: 12, right: 12, top: 10, bottom: 12 }));
        if std::mem::take(&mut self.reset_explorer) {
            panel = panel.exact_size(size::EXPLORER_WIDTH + size::SPLITTER);
        }
        let shown = panel.show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = size::GROUP;
            ui.horizontal(|ui| {
                caption(ui, &p, "Project");
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if widgets::flat_icon_button(ui, &p, Glyph::ChevronRight, "Hide the project explorer (Ctrl+Shift+E)").clicked() {
                        self.explorer_visible = false;
                    }
                });
            });
            muted(ui, &p, self.projects.label());
            self.filter_bar(ui);
            ui.add_space(size::GAP - size::GROUP);
            // The table takes the height the details and buttons leave.
            let below = self.explorer.details_height + size::CONTROL_HEIGHT + 3.0 * size::GROUP;
            let table_height = (ui.available_height() - below).max(80.0);
            let width = ui.available_width();
            let (rect, _) = ui.allocate_exact_size(egui::vec2(width, table_height), Sense::hover());
            let radius = egui::CornerRadius::same(size::RADIUS);
            ui.painter().rect_filled(rect, radius, p.panel);
            let mut inside = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink(1.0)));
            inside.set_clip_rect(rect.shrink(1.0));
            crate::theme::scroll_handles(&mut inside, &p);
            if self.explorer.pivot {
                self.pivot_table(&mut inside);
            } else {
                egui::ScrollArea::horizontal().id_salt("runs_h").auto_shrink([false, false]).show(&mut inside, |ui| self.run_table(ui));
            }
            ui.painter().rect_stroke(rect, radius, egui::Stroke::new(1.0, p.rule), egui::StrokeKind::Inside);
            let top = ui.cursor().top();
            self.details(ui);
            self.explorer.details_height = (ui.cursor().top() - top - size::GROUP).max(0.0);
            let fitting = self.fitting();
            let k = self.projects.selected.len();
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = size::GROUP;
                if ui
                    .add_enabled(!fitting && k == 1, egui::Button::new("Load settings").min_size(egui::vec2(0.0, size::CONTROL_HEIGHT)))
                    .on_hover_text("Set the sidebar to the selected run's settings (or double-click the run)")
                    .clicked()
                {
                    let id = self.projects.selected.iter().next().unwrap().clone();
                    self.load_settings(&id);
                }
                if ui
                    .add_enabled(!fitting && k > 0, egui::Button::new("Delete").min_size(egui::vec2(0.0, size::CONTROL_HEIGHT)))
                    .on_hover_text("Delete the selected runs from the project (Delete key).\nEdit > Restore deleted runs brings them back.")
                    .clicked()
                {
                    self.confirm_delete_runs();
                }
            });
        });
        // Double-clicking the edge puts the default width back.
        let rect = shown.response.rect;
        if widgets::edge_double_clicked(ui.ctx(), rect, rect.left()) {
            self.reset_explorer = true;
            ui.ctx().request_repaint();
        }
        self.column_popup(ui.ctx());
    }

    fn filter_bar(&mut self, ui: &mut Ui) {
        let mut changed = false;
        let p = self.palette.clone();
        // Right to left: the toggles, then the filter in the width they leave.
        let row = egui::vec2(ui.available_width(), size::CONTROL_HEIGHT);
        ui.allocate_ui_with_layout(row, Layout::right_to_left(Align::Center), |ui| {
            let pivot = self.explorer.pivot;
            if widgets::toggle_button(ui, &p, "Pivot", pivot, "Swap rows and columns: one column per run, one row per setting").clicked() {
                self.explorer.pivot = !pivot;
            }
            let invert = self.explorer.filters.invert;
            if widgets::toggle_button(ui, &p, "Invert", invert, "Show the runs the filters do not match").clicked() {
                self.explorer.filters.invert = !invert;
                changed = true;
            }
            let tip = "Show the runs that match in any column, and pass the filter of every\ncolumn (the drop-down at the right of a column header).\nEvery word must match.\nText matches a part of a cell, for example varpro or tolerance.\nA number with a comparison compares the error:\n>1e-5 or <=3e-4. =free matches a whole cell.";
            let width = ui.available_width();
            let (edit, cleared) = widgets::clearable_field(ui, &p, "run_filter", &mut self.explorer.filters.text, width, "Filter all columns");
            changed |= edit.on_hover_text(tip).changed();
            if cleared {
                self.explorer.filters.text.clear();
                changed = true;
            }
        });
        if changed {
            self.explorer.forget_points();
            self.refresh_comparison();
        }
    }

    /// Selects a run by a click with the modifiers held.
    /// Selects one run, as a plain click on it does.
    #[cfg(test)]
    pub fn select_run(&mut self, id: &str) {
        self.click_run(id, &[id.to_string()], egui::Modifiers::NONE);
    }

    fn click_run(&mut self, id: &str, order: &[String], modifiers: egui::Modifiers) {
        let selected = &mut self.projects.selected;
        if modifiers.shift {
            if let Some(anchor) = &self.explorer.anchor {
                let (a, b) = (order.iter().position(|x| x == anchor), order.iter().position(|x| x == id));
                if let (Some(a), Some(b)) = (a, b) {
                    if !modifiers.command {
                        selected.clear();
                    }
                    selected.extend(order[a.min(b)..=a.max(b)].iter().cloned());
                }
            }
        } else if modifiers.command {
            if !selected.remove(id) {
                selected.insert(id.to_string());
            }
            self.explorer.anchor = Some(id.to_string());
        } else {
            selected.clear();
            selected.insert(id.to_string());
            self.explorer.anchor = Some(id.to_string());
        }
        self.on_selection_changed();
    }

    /// A column header: name (click sorts), sort arrow and filter glyph.
    fn header_cell(&mut self, ui: &mut Ui, p: &Palette, column: usize, vertical: bool) {
        let col = &COLUMNS[column];
        let filtered = self.explorer.filters.excluded.contains_key(&column);
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        self.explorer.sections.push((column, rect));
        // A section dragged onto another takes its place.
        if response.drag_stopped() {
            if let Some(pos) = response.interact_pointer_pos() {
                let across = |r: &egui::Rect| if vertical { r.y_range().contains(pos.y) } else { r.x_range().contains(pos.x) };
                if let Some(&(target, _)) = self.explorer.sections.iter().find(|(c, r)| *c != column && across(r)) {
                    let order = &mut self.explorer.order;
                    let to = order.iter().position(|&c| c == target).unwrap_or(0);
                    order.retain(|&c| c != column);
                    order.insert(to.min(order.len()), column);
                }
            }
        }
        if response.dragged() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        }
        let glyph_x = rect.right() - 6.0 - 5.0;
        ui.painter().text(
            rect.left_center() + egui::vec2(size::GAP, 0.0),
            egui::Align2::LEFT_CENTER,
            col.name,
            egui::FontId::proportional(crate::theme::typeface::BASE),
            p.text_3,
        );
        let glyph = if filtered { Glyph::Filter } else { Glyph::ChevronDown };
        glyph.paint(ui.painter(), egui::pos2(glyph_x, rect.center().y), size::GLYPH, if filtered { p.text } else { p.text_3 });
        if self.explorer.sort.0 == column {
            let arrow = if self.explorer.sort.1 { Glyph::SortUp } else { Glyph::SortDown };
            arrow.paint(ui.painter(), egui::pos2(glyph_x - 14.0, rect.center().y), size::GLYPH, p.text_2);
        }
        let response = if col.tooltip.is_empty() { response } else { response.on_hover_text(col.tooltip) };
        if response.clicked() {
            let x = response.interact_pointer_pos().map_or(0.0, |q| q.x);
            if x >= glyph_x - 9.0 {
                // Pivoted, the drop-down opens right of the row name.
                let anchor = if vertical { egui::Rect::from_min_size(rect.right_top(), egui::Vec2::ZERO) } else { rect };
                self.open_column_popup(column, anchor);
            } else {
                let (c, asc) = self.explorer.sort;
                self.explorer.sort = (column, if c == column { !asc } else { true });
                self.explorer.forget_points();
                self.refresh_comparison();
            }
        }
        response.context_menu(|ui| {
            if ui.button("Plot as x axis").clicked() {
                self.explorer.x_column = column;
                self.explorer.forget_points();
                self.refresh_comparison();
            }
            if col.metric && ui.button("Plot as y axis").clicked() {
                self.explorer.y_column = column;
                self.explorer.forget_points();
                self.refresh_comparison();
            }
            ui.separator();
            if ui.button("Filter this column...").clicked() {
                self.open_column_popup(column, rect);
            }
            if ui.button("Clear all filters").clicked() {
                self.explorer.filters = Filters { invert: self.explorer.filters.invert, ..Filters::default() };
                self.explorer.forget_points();
                self.refresh_comparison();
            }
        });
    }

    fn run_table(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let rows = self.visible_rows();
        let order: Vec<String> = rows.iter().map(|(id, _)| id.clone()).collect();
        let mut clicked: Option<(String, egui::Modifiers)> = None;
        let mut double: Option<String> = None;
        let delete = ui.input(|i| i.key_pressed(egui::Key::Delete));
        let columns: Vec<usize> = self.explorer.order.clone();
        self.explorer.sections.clear();
        // Columns fit their contents when the runs change: the header with
        // room for its glyphs, or the widest cell.
        let text = egui::FontId::proportional(crate::theme::typeface::BASE);
        let mono = egui::FontId::monospace(crate::theme::typeface::BASE);
        let measure = |s: &str, font: &egui::FontId| crate::fonts::layout_width(ui, s, font, false);
        let header_width = |c: usize| crate::fonts::layout_width(ui, COLUMNS[c].name, &text, true) + HEADER_EXTRA;
        let widths: Vec<f32> = columns
            .iter()
            .map(|&c| {
                let header = header_width(c);
                let cells = rows
                    .iter()
                    .map(|(_, cells)| measure(&cells[c].1, if cells[c].0.is_number() { &mono } else { &text }) + CELL_EXTRA)
                    .fold(0.0, f32::max);
                // Without runs, Qt's default section width.
                if rows.is_empty() {
                    EMPTY_COLUMN
                } else {
                    header.max(cells).ceil()
                }
            })
            .collect();
        let generation = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for (id, _) in &rows {
                id.hash(&mut h);
            }
            h.finish()
        };
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
        // As wide as its columns: wider than the panel, it scrolls.
        ui.set_min_width(widths.iter().sum());
        // Columns resize by dragging, without lines between them.
        ui.visuals_mut().widgets.noninteractive.bg_stroke.color = egui::Color32::TRANSPARENT;
        let mut builder = TableBuilder::new(ui)
            .id_salt(("runs", generation, columns.clone()))
            .striped(false)
            .resizable(true)
            .sense(Sense::click())
            .auto_shrink([false, false]);
        for w in &widths {
            builder = builder.column(TableColumn::initial(*w).at_least(24.0).clip(true));
        }
        builder
            .header(HEADER_HEIGHT, |mut header| {
                for &c in &columns {
                    header.col(|ui| {
                        self.header_cell(ui, &p, c, false);
                        let r = ui.max_rect();
                        ui.painter().hline(r.x_range(), r.bottom() - 0.5, egui::Stroke::new(1.0, p.rule));
                    });
                }
            })
            .body(|body| {
                body.rows(ROW_HEIGHT, rows.len(), |mut table_row| {
                    let (id, cells) = &rows[table_row.index()];
                    table_row.set_selected(self.projects.selected.contains(id));
                    for &c in &columns {
                        table_row.col(|ui| {
                            let (cell, text) = &cells[c];
                            let rich = if cell.is_number() { RichText::new(text).monospace() } else { RichText::new(text) };
                            let layout =
                                if cell.is_number() { Layout::right_to_left(Align::Center) } else { Layout::left_to_right(Align::Center) };
                            ui.with_layout(layout, |ui| {
                                ui.add_space(size::GAP);
                                ui.add(egui::Label::new(rich.color(p.text)).selectable(false).truncate())
                            });
                        });
                    }
                    let response = table_row.response();
                    if response.clicked() {
                        clicked = Some((id.clone(), response.ctx.input(|i| i.modifiers)));
                    }
                    if response.double_clicked() {
                        double = Some(id.clone());
                    }
                });
            });
        if let Some((id, modifiers)) = clicked {
            self.click_run(&id, &order, modifiers);
        }
        if let Some(id) = double {
            self.load_settings(&id);
        }
        if delete && !self.projects.selected.is_empty() && !ui.ctx().egui_wants_keyboard_input() {
            self.confirm_delete_runs();
        }
    }

    /// The table pivoted: one row per setting (the header sections, in the
    /// shown order, as row names that stay in view), one column per run.
    fn pivot_table(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let rows = self.visible_rows();
        let order: Vec<String> = rows.iter().map(|(id, _)| id.clone()).collect();
        let columns: Vec<usize> = self.explorer.order.clone();
        self.explorer.sections.clear();
        let text = egui::FontId::proportional(crate::theme::typeface::BASE);
        let mono = egui::FontId::monospace(crate::theme::typeface::BASE);
        let measure = |s: &str, font: &egui::FontId| crate::fonts::layout_width(ui, s, font, false);
        let header_width = |c: usize| crate::fonts::layout_width(ui, COLUMNS[c].name, &text, true) + HEADER_EXTRA;
        let name_width = columns.iter().map(|&c| header_width(c)).fold(0.0, f32::max);
        let widths: Vec<f32> = rows
            .iter()
            .map(|(_, cells)| {
                columns
                    .iter()
                    .map(|&c| measure(&cells[c].1, if cells[c].0.is_number() { &mono } else { &text }) + CELL_EXTRA)
                    .fold(0.0, f32::max)
            })
            .collect();
        let height = columns.len() as f32 * ROW_HEIGHT;
        let mut clicked: Option<(String, egui::Modifiers)> = None;
        let mut double: Option<String> = None;
        egui::ScrollArea::vertical().id_salt("pivot_v").auto_shrink([false, false]).show(ui, |ui| {
            ui.spacing_mut().item_spacing = egui::Vec2::ZERO;
            ui.horizontal_top(|ui| {
                // The row names, ruled off on the right.
                let (names, _) = ui.allocate_exact_size(egui::vec2(name_width, height), Sense::hover());
                for (i, &c) in columns.iter().enumerate() {
                    let row =
                        egui::Rect::from_min_size(names.min + egui::vec2(0.0, i as f32 * ROW_HEIGHT), egui::vec2(name_width, ROW_HEIGHT));
                    let mut cell = ui.new_child(egui::UiBuilder::new().max_rect(row));
                    self.header_cell(&mut cell, &p, c, true);
                }
                ui.painter().vline(names.right() - 0.5, names.y_range(), egui::Stroke::new(1.0, p.rule));
                // The runs, one column each, scrolling sideways.
                egui::ScrollArea::horizontal().id_salt("pivot_h").auto_shrink([false, true]).show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::Vec2::ZERO;
                    ui.horizontal_top(|ui| {
                        for ((id, cells), &width) in rows.iter().zip(&widths) {
                            let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), Sense::click());
                            if self.projects.selected.contains(id) {
                                ui.painter().rect_filled(rect, 0.0, p.select);
                            }
                            for (i, &c) in columns.iter().enumerate() {
                                let (cell, value) = &cells[c];
                                let font = if cell.is_number() { mono.clone() } else { text.clone() };
                                let at = egui::pos2(rect.center().x, rect.top() + (i as f32 + 0.5) * ROW_HEIGHT);
                                ui.painter().text(at, egui::Align2::CENTER_CENTER, value, font, p.text);
                            }
                            if response.clicked() {
                                clicked = Some((id.clone(), ui.input(|i| i.modifiers)));
                            }
                            if response.double_clicked() {
                                double = Some(id.clone());
                            }
                        }
                    });
                });
            });
        });
        if let Some((id, modifiers)) = clicked {
            self.click_run(&id, &order, modifiers);
        }
        if let Some(id) = double {
            self.load_settings(&id);
        }
    }

    fn details(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let k = self.projects.selected.len();
        if k == 0 {
            return widgets::readout_list(ui, &p, "Select a run to view details", &[]);
        }
        if k > 1 {
            return widgets::readout_list(ui, &p, &format!("{k} runs selected"), &[]);
        }
        let Some(run) = self.projects.run(self.projects.selected.iter().next().unwrap()) else { return };
        let (c, o, r) = (&run.spline_config, &run.optimization_config, &run.result);
        let count = if o.tolerance.is_some() { r.num_control_points() } else { c.num_control_points };
        let mut pairs = vec![
            ("Degree".to_string(), c.degree.to_string()),
            ("Control points".to_string(), count.to_string()),
            ("Continuity".to_string(), c.continuity_label()),
        ];
        if let Some(t) = o.tolerance {
            pairs.push(("Tolerance".into(), number(t, ".3e")));
        }
        pairs.push(("Error".into(), number(r.final_error, ".3e")));
        if let Some(rep) = &r.report {
            pairs.push(("Max deviation".into(), number(rep.max_deviation, ".3e")));
            pairs.push(("RMS deviation".into(), number(rep.rms_deviation, ".3e")));
        }
        pairs.extend([
            ("Iterations".into(), r.num_iterations.to_string()),
            ("Evaluations".into(), r.num_evaluations.to_string()),
            ("Time".into(), format_duration(r.elapsed_seconds)),
            ("Converged".into(), if r.converged { "yes" } else { "no" }.into()),
        ]);
        widgets::readout_list(ui, &p, o.optimizer.display_name(), &pairs);
    }

    /// The values of a column among the rows other filters let through,
    /// plus the excluded ones; blanks last, numbers first.
    fn column_values(&self, column: usize) -> Vec<(Cell, String)> {
        let mut seen: BTreeMap<String, Cell> = BTreeMap::new();
        for run in self.projects.runs() {
            let cells = row(run);
            if self.explorer.filters.matches_except(&cells, Some(column))
                || self.explorer.filters.excluded.get(&column).is_some_and(|e| e.contains(&cells[column].1))
            {
                seen.entry(cells[column].1.clone()).or_insert(cells[column].0.clone());
            }
        }
        let mut values: Vec<(Cell, String)> = seen.into_iter().map(|(t, c)| (c, t)).collect();
        values.sort_by(|a, b| {
            let rank = |v: &(Cell, String)| {
                if v.1.is_empty() {
                    2
                } else if v.0.is_number() {
                    0
                } else {
                    1
                }
            };
            rank(a).cmp(&rank(b)).then_with(|| compare(a, b))
        });
        values
    }

    pub fn open_column_popup(&mut self, column: usize, anchor: egui::Rect) {
        let excluded = self.explorer.filters.excluded.get(&column).cloned().unwrap_or_default();
        self.explorer.popup = Some(ColumnPopup { column, excluded, condition: String::new(), anchor, focused: false });
    }

    /// Applies the popup's exclusions to the column filter.
    fn apply_popup(&mut self) {
        let Some(popup) = &self.explorer.popup else { return };
        if popup.excluded.is_empty() {
            self.explorer.filters.excluded.remove(&popup.column);
        } else {
            self.explorer.filters.excluded.insert(popup.column, popup.excluded.clone());
        }
        self.explorer.forget_points();
        self.refresh_comparison();
    }

    fn column_popup(&mut self, ctx: &egui::Context) {
        let Some(popup) = &self.explorer.popup else { return };
        let p = self.palette.clone();
        let column = popup.column;
        // Under the section's left edge, or right-aligned to its right edge
        // (as far as it shows) where it would pass the window's edge.
        let width = POPUP_CONTENT_WIDTH.max(size::POPUP_MIN_WIDTH - 2.0 * size::PAD) + 2.0 * size::PAD;
        let screen = ctx.content_rect();
        let anchor = popup.anchor;
        let x = if anchor.left() + width > screen.right() { anchor.right().min(screen.right()) - width } else { anchor.left() };
        let position = egui::pos2(x.max(screen.left()), anchor.bottom());
        let values = self.column_values(column);
        let mut close = false;
        let area = egui::Area::new(egui::Id::unique("column_popup")).order(egui::Order::Foreground).fixed_pos(position).show(ctx, |ui| {
            let frame = egui::Frame::new()
                .fill(p.field)
                .stroke(egui::Stroke::new(1.0, p.rule_strong))
                .corner_radius(egui::CornerRadius::same(size::RADIUS))
                .inner_margin(egui::Margin::same(size::PAD as i8));
            frame.show(ui, |ui| {
                ui.set_width(POPUP_CONTENT_WIDTH.max(size::POPUP_MIN_WIDTH - 2.0 * size::PAD));
                ui.spacing_mut().item_spacing.y = size::GAP;
                let half = (POPUP_CONTENT_WIDTH - size::GAP) / 2.0;
                caption(ui, &p, COLUMNS[column].name);
                ui.horizontal(|ui| {
                    for (text, ascending) in [("Sort ascending", true), ("Sort descending", false)] {
                        if ui.add_sized([half, size::CONTROL_HEIGHT], egui::Button::new(text)).clicked() {
                            self.explorer.sort = (column, ascending);
                            close = true;
                        }
                    }
                });
                let popup = self.explorer.popup.as_mut().unwrap();
                let tip = "Unchecks the values it does not match. Enter keeps the result,\nShift+Enter checks the values it matches instead.\nEvery word must match.\nText matches a part of a cell, for example varpro or tolerance.\nA number with a comparison compares the value:\n>1e-5, <=12, =5 or !=3. =free matches the whole cell.";
                let (edit, cleared) = widgets::clearable_field(ui, &p, "popup_condition", &mut popup.condition, POPUP_CONTENT_WIDTH, "Condition");
                let edit = edit.on_hover_text(tip);
                if !popup.focused {
                    edit.request_focus();
                    popup.focused = true;
                }
                let terms: Vec<Term> = parse_filter(&popup.condition);
                let matches = |v: &(Cell, String)| terms.iter().all(|t| term_matches(t, &v.0, &v.1));
                let ruled_out: BTreeSet<String> = if terms.is_empty() { BTreeSet::new() } else { values.iter().filter(|v| !matches(v)).map(|v| v.1.clone()).collect() };
                let keep = |popup: &mut ColumnPopup, add: bool| {
                    if !terms.is_empty() {
                        if add {
                            for v in values.iter().filter(|v| matches(v)) {
                                popup.excluded.remove(&v.1);
                            }
                        } else {
                            popup.excluded.extend(ruled_out.iter().cloned());
                        }
                        popup.condition.clear();
                    }
                };
                // The clear button keeps what the condition showed, then empties it.
                if cleared {
                    keep(popup, false);
                    popup.condition.clear();
                }
                let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                // Enter elsewhere in the drop-down closes it.
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) && !edit.has_focus() && !edit.lost_focus() {
                    close = true;
                }
                if enter {
                    let shift = ui.input(|i| i.modifiers.shift);
                    keep(popup, shift);
                }
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(!terms.is_empty(), |ui| {
                        let keep_tip = "Uncheck the values the condition does not match (Enter)";
                        if ui.add_sized([half, size::CONTROL_HEIGHT], egui::Button::new("Keep matching")).on_hover_text(keep_tip).clicked() {
                            keep(popup, false);
                        }
                        let add_tip = "Check the values the condition matches (Shift+Enter)";
                        if ui.add_sized([half, size::CONTROL_HEIGHT], egui::Button::new("Add matching")).on_hover_text(add_tip).clicked() {
                            keep(popup, true);
                        }
                    });
                });
                let shown_excluded: BTreeSet<String> = popup.excluded.union(&ruled_out).cloned().collect();
                let all = shown_excluded.is_empty();
                let none = values.iter().all(|v| shown_excluded.contains(&v.1));
                let mut select_all = all;
                let r = widgets::tristate_checkbox(ui, &p, &mut select_all, !all && !none, "Select all");
                if r.clicked() {
                    keep(popup, false);
                    if popup.excluded.is_empty() {
                        popup.excluded = values.iter().map(|v| v.1.clone()).collect();
                    } else {
                        popup.excluded.clear();
                    }
                }
                // The values in a framed list, at most ten rows tall.
                let rows = values.len().clamp(1, 10) as f32;
                let (list, _) = ui.allocate_exact_size(egui::vec2(POPUP_CONTENT_WIDTH, rows * POPUP_ROW + 4.0), egui::Sense::hover());
                let radius = egui::CornerRadius::same(size::RADIUS);
                ui.painter().rect(list, radius, p.panel, egui::Stroke::new(1.0, p.rule), egui::StrokeKind::Inside);
                let mut inside = ui.new_child(egui::UiBuilder::new().max_rect(list.shrink(2.0)));
                crate::theme::scroll_handles(&mut inside, &p);
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(&mut inside, |ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    for v in &values {
                        let mut checked = !shown_excluded.contains(&v.1);
                        let text = if v.1.is_empty() { "Not used" } else { v.1.as_str() };
                        let color = if ruled_out.contains(&v.1) { p.text_3 } else { p.text };
                        let (row, hover) = ui.allocate_exact_size(egui::vec2(ui.available_width(), POPUP_ROW), egui::Sense::hover());
                        if hover.hovered() {
                            ui.painter().rect_filled(row, 0.0, p.field_hover);
                        }
                        let mut cell = ui.new_child(egui::UiBuilder::new().max_rect(row.shrink2(egui::vec2(size::TIGHT, 0.0))).layout(Layout::left_to_right(Align::Center)));
                        if widgets::checkbox(&mut cell, &p, &mut checked, RichText::new(text).color(color)).clicked() {
                            keep(popup, false);
                            if checked {
                                popup.excluded.remove(&v.1);
                            } else {
                                popup.excluded.insert(v.1.clone());
                            }
                        }
                    }
                });
                if ui.add_sized([POPUP_CONTENT_WIDTH, size::CONTROL_HEIGHT], egui::Button::new("Clear filter")).clicked() {
                    popup.excluded.clear();
                    close = true;
                }
            });
        });
        self.apply_popup();
        let rect = area.response.rect;
        let clicked_outside = ctx.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|q| !rect.contains(q)));
        let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        if close || escape || clicked_outside {
            if let Some(popup) = self.explorer.popup.as_mut() {
                popup.condition.clear();
            }
            self.apply_popup();
            self.explorer.popup = None;
        }
    }
}
