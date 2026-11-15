//! The project explorer: the runs of the project in a filterable table,
//! their details, and the comparison plot of the shown runs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use eframe::egui::{self, Align, Layout, RichText, Sense, Ui};
use egui_extras::{Column as TableColumn, TableBuilder};
use spline_fitter::formatting::{format_duration, number};
use spline_fitter::models::{FitResult, IterationRecord, OptimizationConfig, SplineConfig};
use spline_fitter::project::{number_runs, OptimizationRun, ProjectFile};

use crate::app::App;
use crate::runs::{column_index, compare, parse_filter, row, term_matches, Cell, Filters, Term, COLUMNS, CONTROL_POINTS_COLUMN, ERROR_COLUMN};
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

/// Whether a run has the settings of another (19.16.4).
fn same_settings(run: &OptimizationRun, config: &SplineConfig, opt: &OptimizationConfig) -> bool {
    let (c, o) = (&run.spline_config, &run.optimization_config);
    c.degree == config.degree
        && c.num_control_points == config.num_control_points
        && c.num_sections == config.num_sections
        && c.continuity == config.continuity
        && c.geometric_continuity == config.geometric_continuity
        && c.junction_continuity == config.junction_continuity
        && o.tolerance == opt.tolerance
        && o.max_points_per_section == opt.max_points_per_section
        && o.knot_placement == opt.knot_placement
        && o.parameterization == opt.parameterization
        && o.optimize_le_position == opt.optimize_le_position
        && o.grow_control_points == opt.grow_control_points
        && o.optimizer == opt.optimizer
        && (opt.tolerance.is_some() || o.smoothing_weight == opt.smoothing_weight)
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
        }
    }
}

impl ExplorerState {
    /// Forgets the plotted points, so the next update redraws the runs.
    pub fn forget_points(&mut self) {
        self.points = None;
    }
}

/// The open column filter drop-down.
struct ColumnPopup {
    column: usize,
    excluded: BTreeSet<String>,
    condition: String,
    position: egui::Pos2,
}

impl App {
    /// The visible runs in table order, with their cells.
    fn visible_rows(&self) -> Vec<(String, Vec<(Cell, String)>)> {
        let mut rows: Vec<(String, Vec<(Cell, String)>)> =
            self.projects.runs().iter().map(|r| (r.run_id.clone(), row(r))).filter(|(_, cells)| self.explorer.filters.shows(cells)).collect();
        let (column, ascending) = self.explorer.sort;
        rows.sort_by(|a, b| {
            let (x, y) = (&a.1[column], &b.1[column]);
            match (x.0 == Cell::Empty, y.0 == Cell::Empty) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => {
                    let o = compare(x, y);
                    if ascending { o } else { o.reverse() }
                }
            }
        });
        rows
    }

    /// Adds a fit as a run (replacing the run of the same settings); false
    /// when a cancelled fit would replace a finished run.
    pub fn add_run(&mut self, config: &SplineConfig, opt: &OptimizationConfig, fit: FitResult, history: Vec<IterationRecord>, replot: bool) -> bool {
        let Some(airfoil) = self.airfoil.clone() else { return false };
        let existing = self.projects.runs().iter().position(|r| same_settings(r, config, opt));
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
        let sections = self.settings.sections.value.value;
        self.settings.apply_run(run);
        if self.settings.sections.value.value != sections {
            self.resplit();
        }
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
        egui::Panel::right("explorer_rail").resizable(false).exact_size(size::CONTROL_HEIGHT + 8.0).frame(egui::Frame::new().fill(p.panel).inner_margin(egui::Margin { left: 4, right: 4, top: 10, bottom: 12 })).show(ui, |ui| {
            if widgets::icon_button(ui, &p, Glyph::ChevronLeft, false, "Show the project explorer (Ctrl+Shift+E)").clicked() {
                self.explorer_visible = true;
            }
        });
    }

    pub fn explorer_panel(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        egui::Panel::right("explorer")
            .resizable(true)
            .default_size(size::EXPLORER_WIDTH)
            .min_size(size::EXPLORER_MIN_WIDTH)
            .frame(egui::Frame::new().fill(p.panel).inner_margin(egui::Margin { left: 12, right: 12, top: 10, bottom: 12 }))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    caption(ui, &p, "Project");
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if widgets::icon_button(ui, &p, Glyph::ChevronRight, false, "Hide the project explorer (Ctrl+Shift+E)").clicked() {
                            self.explorer_visible = false;
                        }
                    });
                });
                muted(ui, &p, self.projects.label());
                ui.add_space(size::GAP);
                self.filter_bar(ui);
                let details_height = 150.0;
                let table_height = (ui.available_height() - details_height - size::CONTROL_HEIGHT - 3.0 * size::GROUP).max(80.0);
                ui.allocate_ui(egui::vec2(ui.available_width(), table_height), |ui| {
                    if self.explorer.pivot { self.pivot_table(ui) } else { self.run_table(ui) }
                });
                ui.add_space(size::GROUP);
                self.details(ui);
                ui.add_space(size::GROUP);
                let fitting = self.fitting();
                let k = self.projects.selected.len();
                ui.horizontal(|ui| {
                    if ui.add_enabled(!fitting && k == 1, egui::Button::new("Load settings").min_size(egui::vec2(0.0, size::CONTROL_HEIGHT))).on_hover_text("Set the sidebar to the selected run's settings (or double-click the run)").clicked() {
                        let id = self.projects.selected.iter().next().unwrap().clone();
                        self.load_settings(&id);
                    }
                    if ui.add_enabled(!fitting && k > 0, egui::Button::new("Delete").min_size(egui::vec2(0.0, size::CONTROL_HEIGHT))).on_hover_text("Delete the selected runs from the project (Delete key).\nEdit > Restore deleted runs brings them back.").clicked() {
                        self.confirm_delete_runs();
                    }
                });
            });
        self.column_popup(ui.ctx());
    }

    fn filter_bar(&mut self, ui: &mut Ui) {
        let mut changed = false;
        ui.horizontal(|ui| {
            let width = ui.available_width() - 120.0;
            let tip = "Show the runs that match in any column, and pass the filter of every\ncolumn (the drop-down at the right of a column header).\nEvery word must match.\nText matches a part of a cell, for example varpro or tolerance.\nA number with a comparison compares the error:\n>1e-5 or <=3e-4. =free matches a whole cell.";
            changed |= ui.add(egui::TextEdit::singleline(&mut self.explorer.filters.text).hint_text("Filter all columns").desired_width(width)).on_hover_text(tip).changed();
            let invert = self.explorer.filters.invert;
            if ui.add(egui::Button::new("Invert").selected(invert).min_size(egui::vec2(0.0, size::CONTROL_HEIGHT))).on_hover_text("Show the runs the filters do not match").clicked() {
                self.explorer.filters.invert = !invert;
                changed = true;
            }
            let pivot = self.explorer.pivot;
            if ui.add(egui::Button::new("Pivot").selected(pivot).min_size(egui::vec2(0.0, size::CONTROL_HEIGHT))).on_hover_text("Swap rows and columns: one column per run, one row per setting").clicked() {
                self.explorer.pivot = !pivot;
            }
        });
        if changed {
            self.explorer.forget_points();
            self.refresh_comparison();
        }
    }

    /// Selects a run by a click with the modifiers held.
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
    fn header_cell(&mut self, ui: &mut Ui, p: &Palette, column: usize) {
        let col = &COLUMNS[column];
        let filtered = self.explorer.filters.excluded.contains_key(&column);
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click());
        let glyph_x = rect.right() - 6.0 - 5.0;
        ui.painter().text(rect.left_center() + egui::vec2(2.0, 0.0), egui::Align2::LEFT_CENTER, col.name, egui::FontId::proportional(crate::theme::typeface::BASE), p.text_2);
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
                self.open_column_popup(column, rect.left_bottom());
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
                self.open_column_popup(column, rect.left_bottom());
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
        let columns: Vec<usize> = (1..COLUMNS.len()).collect();
        let mut builder = TableBuilder::new(ui).id_salt("runs").striped(false).resizable(true).sense(Sense::click()).auto_shrink([false, false]);
        for _ in &columns {
            builder = builder.column(TableColumn::auto().at_least(48.0).clip(true));
        }
        builder
            .header(size::CONTROL_HEIGHT, |mut header| {
                for &c in &columns {
                    header.col(|ui| self.header_cell(ui, &p, c));
                }
            })
            .body(|body| {
                body.rows(22.0, rows.len(), |mut table_row| {
                    let (id, cells) = &rows[table_row.index()];
                    table_row.set_selected(self.projects.selected.contains(id));
                    for &c in &columns {
                        table_row.col(|ui| {
                            let (cell, text) = &cells[c];
                            let rich = if cell.is_number() { RichText::new(text).monospace() } else { RichText::new(text) };
                            let layout = if cell.is_number() { Layout::right_to_left(Align::Center) } else { Layout::left_to_right(Align::Center) };
                            ui.with_layout(layout, |ui| ui.add(egui::Label::new(rich.color(p.text)).selectable(false).truncate()));
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

    /// The table with one column per run and one row per setting.
    fn pivot_table(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let rows = self.visible_rows();
        let order: Vec<String> = rows.iter().map(|(id, _)| id.clone()).collect();
        let mut clicked: Option<(String, egui::Modifiers)> = None;
        egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
            egui::Grid::new("pivot").striped(false).spacing([14.0, 4.0]).show(ui, |ui| {
                ui.label(RichText::new(COLUMNS[0].name).color(p.text_2));
                for (id, cells) in &rows {
                    let selected = self.projects.selected.contains(id);
                    let r = ui.add(egui::Button::new(RichText::new(&cells[0].1).monospace()).selected(selected).frame(selected));
                    if r.clicked() {
                        clicked = Some((id.clone(), ui.input(|i| i.modifiers)));
                    }
                }
                ui.end_row();
                for c in 1..COLUMNS.len() {
                    self.header_label(ui, &p, c);
                    for (_, cells) in &rows {
                        ui.with_layout(Layout::top_down(Align::Center), |ui| ui.label(RichText::new(&cells[c].1).color(p.text)));
                    }
                    ui.end_row();
                }
            });
        });
        if let Some((id, modifiers)) = clicked {
            self.click_run(&id, &order, modifiers);
        }
    }

    fn header_label(&mut self, ui: &mut Ui, p: &Palette, column: usize) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(COLUMNS[column].name).color(p.text_2));
            let filtered = self.explorer.filters.excluded.contains_key(&column);
            let r = widgets::icon_button(ui, p, if filtered { Glyph::Filter } else { Glyph::ChevronDown }, false, "");
            if r.clicked() {
                self.open_column_popup(column, r.rect.right_top());
            }
        });
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
            if self.explorer.filters.matches_except(&cells, Some(column)) || self.explorer.filters.excluded.get(&column).is_some_and(|e| e.contains(&cells[column].1)) {
                seen.entry(cells[column].1.clone()).or_insert(cells[column].0.clone());
            }
        }
        let mut values: Vec<(Cell, String)> = seen.into_iter().map(|(t, c)| (c, t)).collect();
        values.sort_by(|a, b| {
            let rank = |v: &(Cell, String)| if v.1.is_empty() { 2 } else if v.0.is_number() { 0 } else { 1 };
            rank(a).cmp(&rank(b)).then_with(|| compare(a, b))
        });
        values
    }

    fn open_column_popup(&mut self, column: usize, position: egui::Pos2) {
        let excluded = self.explorer.filters.excluded.get(&column).cloned().unwrap_or_default();
        self.explorer.popup = Some(ColumnPopup { column, excluded, condition: String::new(), position });
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
        let (column, position) = (popup.column, popup.position);
        let values = self.column_values(column);
        let mut close = false;
        let area = egui::Area::new(egui::Id::unique("column_popup")).order(egui::Order::Foreground).fixed_pos(position).show(ctx, |ui| {
            egui::Frame::popup(ui.style()).fill(p.field).inner_margin(egui::Margin::same(12)).show(ui, |ui| {
                ui.set_min_width(size::POPUP_MIN_WIDTH);
                caption(ui, &p, COLUMNS[column].name);
                ui.horizontal(|ui| {
                    if widgets::button(ui, "Sort ascending", "").clicked() {
                        self.explorer.sort = (column, true);
                        close = true;
                    }
                    if widgets::button(ui, "Sort descending", "").clicked() {
                        self.explorer.sort = (column, false);
                        close = true;
                    }
                });
                let popup = self.explorer.popup.as_mut().unwrap();
                let tip = "Unchecks the values it does not match. Enter keeps the result,\nShift+Enter checks the values it matches instead.\nEvery word must match.\nText matches a part of a cell, for example varpro or tolerance.\nA number with a comparison compares the value:\n>1e-5, <=12, =5 or !=3. =free matches the whole cell.";
                let edit = ui.add(egui::TextEdit::singleline(&mut popup.condition).hint_text("Condition").desired_width(f32::INFINITY)).on_hover_text(tip);
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
                let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if enter {
                    let shift = ui.input(|i| i.modifiers.shift);
                    keep(popup, shift);
                }
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(!terms.is_empty(), |ui| {
                        if widgets::button(ui, "Keep matching", "Uncheck the values the condition does not match (Enter)").clicked() {
                            keep(popup, false);
                        }
                        if widgets::button(ui, "Add matching", "Check the values the condition matches (Shift+Enter)").clicked() {
                            keep(popup, true);
                        }
                    });
                });
                let shown_excluded: BTreeSet<String> = popup.excluded.union(&ruled_out).cloned().collect();
                let all = shown_excluded.is_empty();
                let none = values.iter().all(|v| shown_excluded.contains(&v.1));
                let mut select_all = all;
                let r = ui.add(egui::Checkbox::new(&mut select_all, "Select all").indeterminate(!all && !none));
                if r.clicked() {
                    keep(popup, false);
                    if popup.excluded.is_empty() {
                        popup.excluded = values.iter().map(|v| v.1.clone()).collect();
                    } else {
                        popup.excluded.clear();
                    }
                }
                egui::ScrollArea::vertical().max_height(10.0 * 24.0).show(ui, |ui| {
                    for v in &values {
                        let mut checked = !shown_excluded.contains(&v.1);
                        let text = if v.1.is_empty() { "Not used" } else { v.1.as_str() };
                        let color = if ruled_out.contains(&v.1) { p.text_3 } else { p.text };
                        if ui.checkbox(&mut checked, RichText::new(text).color(color)).clicked() {
                            keep(popup, false);
                            if checked {
                                popup.excluded.remove(&v.1);
                            } else {
                                popup.excluded.insert(v.1.clone());
                            }
                        }
                    }
                });
                if widgets::button(ui, "Clear filter", "").clicked() {
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
        let _ = column_index;
    }
}
