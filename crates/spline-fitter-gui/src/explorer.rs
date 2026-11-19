//! The project explorer: the runs of the project in a filterable table,
//! their details, and the comparison plot of the shown runs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use eframe::egui::{self, Align, Key, Layout, RichText, Sense, Ui};
use egui_extras::{Column as TableColumn, TableBuilder};
use spline_fitter::bspline_varpro::KnotPlacement;
use spline_fitter::formatting::{format_duration, number};
use spline_fitter::models::{FitResult, IterationRecord, OptimizationConfig, SplineConfig, ToleranceMetric};
use spline_fitter::project::{number_runs, OptimizationRun, ProjectFile, ToleranceSearch};

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
    /// Deleted runs and searches with their positions, latest last.
    pub deleted: Vec<Deleted>,
    /// The highest number given to a run.
    pub last_number: usize,
    /// Selected runs by id.
    pub selected: BTreeSet<String>,
    /// The run whose fit is shown.
    pub shown_run: Option<String>,
}

/// What one deletion took away, to restore.
#[derive(Default)]
pub struct Deleted {
    runs: Vec<(usize, OptimizationRun)>,
    searches: Vec<(usize, ToleranceSearch)>,
}

/// A tolerance search's row id in the run table (a run's is its run id).
const SEARCH_ROW: &str = "search:";

pub fn search_row(search_id: &str) -> String {
    format!("{SEARCH_ROW}{search_id}")
}

/// The search of a row id, if it is a search's row.
pub fn row_search(row: &str) -> Option<&str> {
    row.strip_prefix(SEARCH_ROW)
}

impl Projects {
    pub fn runs(&self) -> &[OptimizationRun] {
        self.file.as_ref().map_or(&[], |f| &f.runs)
    }

    pub fn searches(&self) -> &[ToleranceSearch] {
        self.file.as_ref().map_or(&[], |f| &f.searches)
    }

    pub fn run(&self, id: &str) -> Option<&OptimizationRun> {
        self.runs().iter().find(|r| r.run_id == id)
    }

    pub fn search(&self, id: &str) -> Option<&ToleranceSearch> {
        self.searches().iter().find(|s| s.search_id == id)
    }

    /// The run a search ended on (or, deleted, its last).
    pub fn search_result(&self, search: &ToleranceSearch) -> Option<&OptimizationRun> {
        self.run(&search.result_run)
            .filter(|r| r.search.as_deref() == Some(search.search_id.as_str()))
            .or_else(|| self.runs().iter().rev().find(|r| r.search.as_deref() == Some(search.search_id.as_str())))
    }

    /// A search as a run: its settings, the fit it ended on, and the
    /// search's own effort and outcome. `reached`: the control points it
    /// reached (its row) rather than its start count (its settings).
    pub fn search_as_run(&self, search: &ToleranceSearch, reached: bool) -> Option<OptimizationRun> {
        let mut run = self.search_result(search)?.clone();
        run.spline_config = search.spline_config.clone();
        if reached {
            run.spline_config.num_control_points = search.control_points;
        }
        run.optimization_config = search.optimization_config.clone();
        run.timestamp = search.timestamp.clone();
        run.number = search.number;
        run.search = None;
        let r = &mut run.result;
        r.num_iterations = search.fits;
        r.num_evaluations = search.evaluations;
        r.elapsed_seconds = search.elapsed_seconds;
        r.converged = search.met;
        r.cancelled = search.cancelled;
        Some(run)
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
                let searches = match f.searches.len() {
                    0 => String::new(),
                    1 => ", 1 tolerance search".into(),
                    n => format!(", {n} tolerance searches"),
                };
                format!(
                    "{}{}\n{k} optimization run{}{searches}",
                    self.name(),
                    if self.modified { "*" } else { "" },
                    if k == 1 { "" } else { "s" }
                )
            }
        }
    }
}

/// Whether a run of its own has the settings of a new fit (19.16.4): with
/// free knots also the knot spacing. The fits of a tolerance search belong
/// to it and are replaced with it.
fn same_settings(run: &OptimizationRun, config: &SplineConfig, opt: &OptimizationConfig, fit: &FitResult) -> bool {
    let (c, o) = (&run.spline_config, &run.optimization_config);
    run.search.is_none()
        && c.degree == config.degree
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
        && o.smoothing_weight == opt.smoothing_weight
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
    /// Where a Shift-click or Shift-arrow selects from.
    anchor: Option<String>,
    /// The table's current run: the arrow keys move from it.
    current: Option<String>,
    popup: Option<ColumnPopup>,
    /// Height of the run details on the last frame (the table takes the rest).
    details_height: f32,
    /// The columns in the order shown (header sections can be dragged).
    order: Vec<usize>,
    /// Where each header section was on the last frame, in shown order.
    sections: Vec<(usize, egui::Rect)>,
    /// Where the run table's own scroll bar moved its rows to.
    scroll_to: Option<f32>,
    /// Bring the current run's row into view on the next frame.
    reveal: bool,
    /// The tolerance searches shown open, their runs below them.
    pub expanded: BTreeSet<String>,
}

/// A row of the run table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowKind {
    /// A run of its own.
    Run,
    /// A tolerance search, its runs shown below it when open.
    Search { open: bool },
    /// A run of an open search.
    InSearch,
}

/// What the empty run table says.
pub const EMPTY_TABLE_HINT: &str = "Every fit is kept here as a run, to compare, reload and save as a project";

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
            current: None,
            popup: None,
            details_height: 15.0,
            order: (0..COLUMNS.len()).collect(),
            sections: Vec::new(),
            scroll_to: None,
            reveal: false,
            expanded: BTreeSet::new(),
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

    /// The explorer with its columns in the order `names` (the stored
    /// [`column_names`](Self::column_names)); unknown names are skipped
    /// and columns not named follow in their usual order.
    pub fn with_column_names(names: &str) -> Self {
        let mut order: Vec<usize> = Vec::new();
        for name in names.split(',') {
            if let Some(c) = COLUMNS.iter().position(|col| col.name == name).filter(|c| !order.contains(c)) {
                order.push(c);
            }
        }
        let rest: Vec<usize> = (0..COLUMNS.len()).filter(|c| !order.contains(c)).collect();
        order.extend(rest);
        ExplorerState { order, ..ExplorerState::default() }
    }

    /// The columns' names in the order shown, to keep between sessions.
    pub fn column_names(&self) -> String {
        self.order.iter().map(|&c| COLUMNS[c].name).collect::<Vec<_>>().join(",")
    }
}

/// Run table: header and row heights, and what a header and a cell add to
/// their text. Every column is as narrow as its header and cells allow:
/// a cell keeps a gap on each side; a header its text's gap, then room for
/// the sort arrow and the filter chevron (kept on every column, so sorting
/// does not change the widths).
const HEADER_HEIGHT: f32 = 28.0;
const ROW_HEIGHT: f32 = 28.0;
const CELL_EXTRA: f32 = 2.0 * size::GAP;
/// Centers of the filter chevron and the sort arrow, from the right edge.
const CHEVRON_INSET: f32 = 9.0;
const ARROW_INSET: f32 = 21.0;
const HEADER_EXTRA: f32 = size::GAP + size::TIGHT + ARROW_INSET + size::GLYPH / 2.0;

/// The run table's keyboard focus (a click on a run gives it).
fn table_focus() -> egui::Id {
    egui::Id::unique("run_table_focus")
}

/// Registers the table as the focus's holder for this frame (egui drops
/// the focus of a widget it does not see).
/// `take`: a run was clicked, which gives the table the focus.
fn table_focus_target(ui: &Ui, take: bool) {
    let response = ui.interact(ui.max_rect(), table_focus(), Sense::focusable_noninteractive());
    if take {
        response.request_focus();
    }
}

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
    /// The cells of a tolerance search's row: its settings, the fit it
    /// ended on, its effort; the Run column names it.
    fn search_cells(&self, search: &ToleranceSearch) -> Option<Vec<(Cell, String)>> {
        let mut cells = row(&self.projects.search_as_run(search, true)?);
        let name = format!("Search {}", search.number);
        cells[0] = (Cell::Text(name.clone()), name);
        Some(cells)
    }

    /// The visible rows in table order, with their cells: the runs of their
    /// own and the tolerance searches, sorted; below an open search, its
    /// runs that pass the filters, sorted the same way. A search shows when
    /// it or one of its runs passes the filters.
    pub(crate) fn visible_rows(&self) -> Vec<(String, Vec<(Cell, String)>)> {
        let shows = |cells: &Vec<(Cell, String)>| self.explorer.filters.shows(cells);
        let runs_of = |id: &str| -> Vec<(String, Vec<(Cell, String)>)> {
            self.projects.runs().iter().filter(|r| r.search.as_deref() == Some(id)).map(|r| (r.run_id.clone(), row(r))).collect()
        };
        let mut rows: Vec<(String, Vec<(Cell, String)>)> = self
            .projects
            .runs()
            .iter()
            .filter(|r| r.search.is_none())
            .map(|r| (r.run_id.clone(), row(r)))
            .filter(|(_, cells)| shows(cells))
            .collect();
        for s in self.projects.searches() {
            let Some(cells) = self.search_cells(s) else { continue };
            if shows(&cells) || runs_of(&s.search_id).iter().any(|(_, c)| shows(c)) {
                rows.push((search_row(&s.search_id), cells));
            }
        }
        self.sort_rows(&mut rows);
        let mut all = Vec::with_capacity(rows.len());
        for (id, cells) in rows {
            let open = row_search(&id).filter(|s| self.explorer.expanded.contains(*s)).map(str::to_string);
            all.push((id, cells));
            if let Some(search) = open {
                let mut inside: Vec<_> = runs_of(&search).into_iter().filter(|(_, c)| shows(c)).collect();
                self.sort_rows(&mut inside);
                all.extend(inside);
            }
        }
        all
    }

    /// Every run that passes the filters, the runs of searches open or not
    /// (what the runs plot shows).
    fn filtered_runs(&self) -> Vec<(String, Vec<(Cell, String)>)> {
        let mut rows: Vec<_> =
            self.projects.runs().iter().map(|r| (r.run_id.clone(), row(r))).filter(|(_, c)| self.explorer.filters.shows(c)).collect();
        self.sort_rows(&mut rows);
        rows
    }

    /// Whether a row is a run inside a search.
    fn in_search(&self, id: &str) -> bool {
        self.projects.run(id).is_some_and(|r| r.search.is_some())
    }

    /// Opens or closes a search's row.
    pub fn toggle_search(&mut self, search: &str, open: bool) {
        if open {
            self.explorer.expanded.insert(search.to_string());
        } else {
            self.explorer.expanded.remove(search);
        }
    }

    fn sort_rows(&self, rows: &mut [(String, Vec<(Cell, String)>)]) {
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

    pub(crate) fn load_settings(&mut self, id: &str) {
        if self.fitting() {
            return self.set_status("Cancel the running fit to load a run");
        }
        // A search's row: the tolerance settings it ran with.
        if let Some(search) = row_search(id).and_then(|s| self.projects.search(s)).cloned() {
            let Some(settings) = self.projects.search_as_run(&search, false) else { return };
            let result = self.projects.search_result(&search).cloned();
            self.apply_run_settings(&settings);
            if let Some(run) = result {
                self.show_run(&run);
            }
            let (c, o) = (&search.spline_config, &search.optimization_config);
            let start = if o.grow_control_points { "the fewest control points".to_string() } else { format!("{} control points", c.num_control_points) };
            return self.set_status(format!(
                "Applied the settings of search {} (tolerance {}, from {start}, degree {}, {} sections)",
                search.number,
                number(search.tolerance(), ".2e"),
                c.degree,
                c.num_sections
            ));
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

    /// Deletes the selected runs, and the selected searches with their
    /// runs; a search left without runs goes too.
    pub fn delete_selected_runs(&mut self) {
        let Some(file) = self.projects.file.as_mut() else { return };
        let selected = std::mem::take(&mut self.projects.selected);
        let searches: BTreeSet<&str> = selected.iter().filter_map(|id| row_search(id)).collect();
        let mut removed = Deleted::default();
        let mut i = 0;
        while i < file.runs.len() {
            let r = &file.runs[i];
            if selected.contains(&r.run_id) || r.search.as_deref().is_some_and(|s| searches.contains(s)) {
                removed.runs.push((i + removed.runs.len(), file.runs.remove(i)));
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < file.searches.len() {
            let id = file.searches[i].search_id.as_str();
            if file.runs.iter().any(|r| r.search.as_deref() == Some(id)) {
                i += 1;
            } else {
                removed.searches.push((i + removed.searches.len(), file.searches.remove(i)));
            }
        }
        let k = removed.runs.len();
        let gone = |id: &String| !file.runs.iter().any(|r| &r.run_id == id);
        if self.projects.shown_run.as_ref().is_some_and(gone) {
            self.fit = None;
            self.projects.shown_run = None;
            self.main_plot.clear_items();
            self.loss.reset();
        }
        let n = removed.searches.len();
        self.projects.deleted.push(removed);
        self.projects.modified = true;
        self.explorer.forget_points();
        self.refresh_comparison();
        let searches = match n {
            0 => String::new(),
            1 => " and 1 tolerance search".into(),
            n => format!(" and {n} tolerance searches"),
        };
        self.set_status(format!("Deleted {k} run{}{searches}; Edit > Restore deleted runs brings them back", if k == 1 { "" } else { "s" }));
    }

    pub fn restore_deleted_runs(&mut self) {
        let (Some(batch), Some(file)) = (self.projects.deleted.pop(), self.projects.file.as_mut()) else { return };
        let k = batch.runs.len();
        self.projects.selected.clear();
        let restored: BTreeSet<String> = batch.searches.iter().map(|(_, s)| s.search_id.clone()).collect();
        for (index, search) in batch.searches {
            self.projects.selected.insert(search_row(&search.search_id));
            let at = index.min(file.searches.len());
            file.searches.insert(at, search);
        }
        for (index, run) in batch.runs {
            if !run.search.as_ref().is_some_and(|s| restored.contains(s)) {
                self.projects.selected.insert(run.run_id.clone());
            }
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
        // Every run, also those of searches not open in the table.
        let rows = self.filtered_runs();
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
        // The runs' shared tolerance (of their search), where the y axis is
        // their deviation (a tolerance run of an older project has its
        // largest deviation as its error).
        let y = COLUMNS[yc].name;
        let tolerance_of = |id: &str| {
            let run = self.projects.run(id)?;
            match run.search.as_deref().and_then(|s| self.projects.search(s)) {
                Some(search) => {
                    let o = &search.optimization_config;
                    o.tolerance.filter(|_| y == o.tolerance_metric.name())
                }
                None => run.optimization_config.tolerance.filter(|_| matches!(y, "Error" | "Max deviation")),
            }
        };
        let tolerances: BTreeSet<u64> = xy.iter().map(|(id, _)| tolerance_of(id).map_or(u64::MAX, f64::to_bits)).collect();
        c.tolerance = match tolerances.into_iter().collect::<Vec<_>>()[..] {
            [bits] if bits != u64::MAX => Some(f64::from_bits(bits)),
            _ => None,
        };
        // "Run 12: 30 control points, max deviation 8.71e-06"
        let label = |id: &str, y: f64| {
            let value = if y.fract() == 0.0 { format!("{y}") } else { spline_fitter::formatting::format_error(y) };
            self.projects.run(id).map_or_else(String::new, |r| {
                format!("Run {}: {} control points, {} {value}", r.number, r.result.control_points_per_section(), COLUMNS[yc].name.to_lowercase())
            })
        };
        c.labels = xy.iter().map(|(id, q)| label(id, q[1])).collect();
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
        // A selected search marks its runs.
        let selected = |id: &String| {
            self.projects.selected.contains(id)
                || self.projects.run(id).and_then(|r| r.search.as_deref()).is_some_and(|s| self.projects.selected.contains(&search_row(s)))
        };
        self.comparison.marked = points.iter().filter(|(id, _)| selected(id)).map(|(_, p)| *p).collect();
    }

    /// A click on the `index`th point of the runs plot: selects its run as
    /// a plain click on its row does, and brings the row into view.
    pub fn click_plotted_run(&mut self, index: usize) {
        let Some(id) = self.explorer.points.as_ref().and_then(|p| p.get(index)).map(|(id, _)| id.clone()) else { return };
        // Its search opens, to show the run.
        if let Some(search) = self.projects.run(&id).and_then(|r| r.search.clone()) {
            self.toggle_search(&search, true);
        }
        self.click_run(&id, std::slice::from_ref(&id), egui::Modifiers::NONE);
        self.explorer.reveal = true;
    }

    fn on_selection_changed(&mut self) {
        self.mark_selection();
        if self.projects.selected.len() == 1 && !self.fitting() {
            let id = self.projects.selected.iter().next().unwrap().clone();
            // A search shows the fit it ended on.
            let run = match row_search(&id).and_then(|s| self.projects.search(s)) {
                Some(search) => self.projects.search_result(search).cloned(),
                None => self.projects.run(&id).cloned(),
            };
            if let Some(run) = run {
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
                    self.set_explorer_visible(true);
                }
            });
    }

    pub fn explorer_panel(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let mut panel = egui::Panel::right("explorer")
            .resizable(true)
            .default_size(self.store.float("panels/explorer_width", f64::from(size::EXPLORER_WIDTH + size::SPLITTER)) as f32)
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
                        self.set_explorer_visible(false);
                    }
                });
            });
            muted(ui, &p, self.projects.label());
            self.filter_bar(ui);
            ui.add_space(size::GAP - size::GROUP);
            // The table takes the height the details and buttons leave (a
            // group's space before each).
            let below = self.explorer.details_height + size::CONTROL_HEIGHT + 2.0 * size::GROUP;
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
                let body = egui::ScrollArea::horizontal()
                    .id_salt("runs_h")
                    .auto_shrink([false, false])
                    .show(&mut inside, |ui| self.run_table(ui))
                    .inner;
                // The rows' scroll bar at the box's edge (the table itself is
                // as wide as its columns), from the header down.
                if let Some(body) = body {
                    let viewport = inside.max_rect();
                    let track = egui::Rect::from_min_max(
                        egui::pos2(viewport.right() - size::SCROLL_TRACK, viewport.top()),
                        egui::pos2(viewport.right(), body.inner_rect.bottom()),
                    );
                    let id = inside.scope_id().with("runs_v");
                    let bar = widgets::vertical_scroll_bar(
                        &mut inside,
                        &p,
                        id,
                        track,
                        body.state.clamped_offset().y,
                        body.content_size.y,
                        body.inner_rect.height(),
                    );
                    if let Some(y) = bar {
                        self.explorer.scroll_to = Some(y);
                        inside.ctx().request_repaint();
                    }
                }
            }
            // No runs yet: what will be here.
            if self.projects.runs().is_empty() {
                crate::plots::centred_hint(ui.painter(), rect, EMPTY_TABLE_HINT, p.text_3, size::PAD);
            }
            ui.painter().rect_stroke(rect, radius, egui::Stroke::new(1.0, p.rule), egui::StrokeKind::Inside);
            let top = ui.cursor().top();
            self.details(ui);
            self.explorer.details_height = (ui.cursor().top() - top - size::GROUP).max(0.0);
            let fitting = self.fitting();
            let k = self.projects.selected.len();
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = size::GROUP;
                if widgets::push_button(ui, &p, !fitting && k == 1, "Load settings")
                    .on_hover_text("Set the sidebar to the selected run's settings (or double-click the run)")
                    .clicked()
                {
                    let id = self.projects.selected.iter().next().unwrap().clone();
                    self.load_settings(&id);
                }
                if widgets::push_button(ui, &p, !fitting && k > 0, "Delete")
                    .on_hover_text("Delete the selected runs from the project (Delete key).\nEdit > Restore deleted runs brings them back.")
                    .clicked()
                {
                    self.confirm_delete_runs();
                }
            });
        });
        // Double-clicking the edge puts the default width back.
        let rect = shown.response.rect;
        self.remember_width(ui.ctx(), "panels/explorer_width", rect.width());
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
            if widgets::icon_button(ui, &p, Glyph::Pivot, pivot, "Pivot: swap rows and columns, one column per run and one row per setting").clicked() {
                self.explorer.pivot = !pivot;
                // Kept between sessions, as the theme is.
                self.store.set_bool("explorer/pivot", !pivot);
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
        self.explorer.current = Some(id.to_string());
        let selected = &mut self.projects.selected;
        if modifiers.shift {
            if self.explorer.anchor.is_none() {
                self.explorer.anchor = Some(id.to_string());
            }
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
                    // The order is kept between sessions.
                    let names = self.explorer.column_names();
                    self.store.set_string("explorer/columns", &names);
                }
            }
        }
        if response.dragged() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        }
        let glyph_x = rect.right() - CHEVRON_INSET;
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
            arrow.paint(ui.painter(), egui::pos2(rect.right() - ARROW_INSET, rect.center().y), size::GLYPH, p.text_2);
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

    /// The runs, one per row; the body's scroll state (no rows: none).
    fn run_table(&mut self, ui: &mut Ui) -> Option<egui::scroll_area::ScrollAreaOutput<()>> {
        let p = self.palette.clone();
        let rows = self.visible_rows();
        let order: Vec<String> = rows.iter().map(|(id, _)| id.clone()).collect();
        let mut clicked: Option<(String, egui::Modifiers)> = None;
        let mut double: Option<String> = None;
        let delete = ui.input(|i| i.key_pressed(egui::Key::Delete));
        let page = ((ui.available_height() - HEADER_HEIGHT) / ROW_HEIGHT).floor().max(1.0) as usize;
        let reveal = self.table_keys(ui, &order, false, page).or_else(|| self.take_reveal(&order));
        let columns: Vec<usize> = self.explorer.order.clone();
        self.explorer.sections.clear();
        // A search's row opens and closes at its chevron; its runs below it
        // are indented as far.
        let kinds: Vec<RowKind> = rows
            .iter()
            .map(|(id, _)| match row_search(id) {
                Some(s) => RowKind::Search { open: self.explorer.expanded.contains(s) },
                None if self.in_search(id) => RowKind::InSearch,
                None => RowKind::Run,
            })
            .collect();
        let indent = if kinds.iter().any(|k| *k != RowKind::Run) { size::GLYPH + size::TIGHT } else { 0.0 };
        let mut toggle: Option<(String, bool)> = None;
        // Columns fit their contents when the runs change: the header with
        // room for its glyphs, or the widest cell.
        let text = egui::FontId::proportional(crate::theme::typeface::BASE);
        let mono = egui::FontId::monospace(crate::theme::typeface::BASE);
        let measure = |s: &str, font: &egui::FontId| crate::fonts::layout_width(ui, s, font, false);
        let header_width = |c: usize| crate::fonts::layout_width(ui, COLUMNS[c].name, &text, false) + HEADER_EXTRA;
        let widths: Vec<f32> = columns
            .iter()
            .map(|&c| {
                let header = header_width(c);
                let cells = rows
                    .iter()
                    .map(|(_, cells)| measure(&cells[c].1, if cells[c].0.is_number() { &mono } else { &text }) + CELL_EXTRA)
                    .fold(0.0, f32::max);
                let first = columns.first() == Some(&c);
                header.max(cells + if first { indent } else { 0.0 }).ceil()
            })
            .collect();
        let generation = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for (id, _) in &rows {
                id.hash(&mut h);
            }
            indent.to_bits().hash(&mut h);
            h.finish()
        };
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
        // As wide as its columns: wider than the panel, it scrolls.
        ui.set_min_width(widths.iter().sum());
        // No lines between the columns.
        ui.visuals_mut().widgets.noninteractive.bg_stroke.color = egui::Color32::TRANSPARENT;
        let mut builder = TableBuilder::new(ui)
            .id_salt(("runs", generation, columns.clone()))
            .striped(false)
            .resizable(false)
            .sense(Sense::click())
            .auto_shrink([false, false])
            // Its own bar would be at the table's right end, out of view.
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden);
        if let Some(y) = self.explorer.scroll_to.take() {
            builder = builder.vertical_scroll_offset(y);
        }
        if let Some(row) = reveal {
            builder = builder.scroll_to_row(row, None);
        }
        for w in &widths {
            builder = builder.column(TableColumn::exact(*w).clip(true));
        }
        let body = builder
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
                    let index = table_row.index();
                    let (id, cells) = &rows[index];
                    let kind = kinds[index];
                    table_row.set_selected(self.projects.selected.contains(id));
                    for (k, &c) in columns.iter().enumerate() {
                        table_row.col(|ui| {
                            let (cell, text) = &cells[c];
                            let rich = if cell.is_number() { RichText::new(text).monospace() } else { RichText::new(text) };
                            let layout =
                                if cell.is_number() { Layout::right_to_left(Align::Center) } else { Layout::left_to_right(Align::Center) };
                            let first = k == 0;
                            if first {
                                if let (RowKind::Search { open }, Some(search)) = (kind, row_search(id)) {
                                    let r = ui.max_rect();
                                    let centre = egui::pos2(r.left() + CELL_EXTRA / 2.0 + size::GLYPH / 2.0, r.center().y);
                                    let hit = egui::Rect::from_center_size(centre, egui::vec2(size::GLYPH + size::TIGHT, ROW_HEIGHT));
                                    let chevron = ui.interact(hit, ui.scope_id().with(("open_search", search)), Sense::click());
                                    let glyph = if open { Glyph::ChevronDown } else { Glyph::ChevronRight };
                                    glyph.paint(ui.painter(), centre, size::GLYPH, if chevron.hovered() { p.text } else { p.text_2 });
                                    let chevron = chevron.on_hover_text(if open { "Hide the runs of this search" } else { "Show the runs of this search" });
                                    if chevron.clicked() {
                                        toggle = Some((search.to_string(), !open));
                                    }
                                }
                            }
                            ui.with_layout(layout, |ui| {
                                ui.add_space(CELL_EXTRA / 2.0);
                                if first && !cell.is_number() {
                                    ui.add_space(indent);
                                }
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
        table_focus_target(ui, clicked.is_some() || toggle.is_some());
        if let Some((search, open)) = toggle {
            self.toggle_search(&search, open);
            self.explorer.current = Some(search_row(&search));
        } else {
            if let Some((id, modifiers)) = clicked {
                self.click_run(&id, &order, modifiers);
            }
            if let Some(id) = double {
                self.load_settings(&id);
            }
        }
        if delete && !self.projects.selected.is_empty() && !ui.ctx().egui_wants_keyboard_input() {
            self.confirm_delete_runs();
        }
        Some(body)
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
        let header_width = |c: usize| crate::fonts::layout_width(ui, COLUMNS[c].name, &text, false) + HEADER_EXTRA;
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
        let reveal = self.table_keys(ui, &order, true, 1).or_else(|| self.take_reveal(&order));
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
                            if reveal.is_some_and(|i| order[i] == *id) {
                                ui.scroll_to_rect(rect, None);
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
        table_focus_target(ui, clicked.is_some());
        if let Some((id, modifiers)) = clicked {
            self.click_run(&id, &order, modifiers);
        }
        if let Some(id) = double {
            self.load_settings(&id);
        }
    }

    /// Qt's table keys while the run table has the focus. The runs are rows
    /// (pivoted: columns): Up and Down (pivoted: Left and Right) move to
    /// the next run and select it, Page Up and Page Down a page of rows,
    /// Ctrl+Home and Ctrl+End (pivoted: Home and End) to the first and
    /// last. Shift selects from the anchor to there; Ctrl moves without
    /// selecting, and Ctrl+Space then selects or deselects the run. The
    /// run moved to, to scroll into view.
    /// The current run's place in `order` when it is to be brought into view.
    fn take_reveal(&mut self, order: &[String]) -> Option<usize> {
        let current = self.explorer.current.as_ref();
        std::mem::take(&mut self.explorer.reveal).then(|| order.iter().position(|id| Some(id) == current)).flatten()
    }

    fn table_keys(&mut self, ui: &Ui, order: &[String], pivoted: bool, page: usize) -> Option<usize> {
        if !ui.memory(|m| m.has_focus(table_focus())) || order.is_empty() {
            return None;
        }
        // The arrows stay with the table instead of moving the focus.
        let filter = egui::EventFilter { horizontal_arrows: true, vertical_arrows: true, ..Default::default() };
        ui.memory_mut(|m| m.set_focus_lock_filter(table_focus(), filter));
        let (back, forward) = if pivoted { (Key::ArrowLeft, Key::ArrowRight) } else { (Key::ArrowUp, Key::ArrowDown) };
        let last = order.len() - 1;
        let current = self.explorer.current.as_ref().and_then(|c| order.iter().position(|x| x == c));
        // Right opens the current search, Left closes it (or, on one of its
        // runs, goes up to it).
        if let (false, Some(c)) = (pivoted, current) {
            let (right, left) = ui.input(|i| (i.key_pressed(Key::ArrowRight), i.key_pressed(Key::ArrowLeft)));
            let id = order[c].clone();
            if let Some(search) = row_search(&id) {
                let open = self.explorer.expanded.contains(search);
                if (right && !open) || (left && open) {
                    self.toggle_search(search, right);
                    ui.ctx().request_repaint();
                    return None;
                }
            } else if left {
                if let Some(search) = self.projects.run(&id).and_then(|r| r.search.clone()) {
                    let row = search_row(&search);
                    let at = order.iter().position(|x| *x == row);
                    self.click_run(&row, order, egui::Modifiers::NONE);
                    return at;
                }
            }
        }
        let (target, modifiers) = ui.input(|i| {
            let m = i.modifiers;
            let jump = |key| i.key_pressed(key) && (pivoted || m.command);
            // Without a current run, a move goes to the first.
            let at = |step: isize| current.map_or(0, |c| (c as isize + step).clamp(0, last as isize) as usize);
            let target = if i.key_pressed(back) {
                Some(at(-1))
            } else if i.key_pressed(forward) {
                Some(at(1))
            } else if !pivoted && i.key_pressed(Key::PageUp) {
                Some(at(-(page as isize)))
            } else if !pivoted && i.key_pressed(Key::PageDown) {
                Some(at(page as isize))
            } else if jump(Key::Home) {
                Some(0)
            } else if jump(Key::End) {
                Some(last)
            } else {
                None
            };
            (target, m)
        });
        if modifiers.command && ui.input(|i| i.key_pressed(Key::Space)) {
            if let Some(c) = current {
                self.click_run(&order[c], order, egui::Modifiers::COMMAND);
            }
            return None;
        }
        let target = target?;
        if modifiers.command && !modifiers.shift {
            self.explorer.current = Some(order[target].clone());
        } else {
            let modifiers = egui::Modifiers { shift: modifiers.shift, ..Default::default() };
            self.click_run(&order[target], order, modifiers);
        }
        ui.ctx().request_repaint();
        Some(target)
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
        let id = self.projects.selected.iter().next().unwrap();
        if let Some(search) = row_search(id).and_then(|s| self.projects.search(s)) {
            let (c, o) = (&search.spline_config, &search.optimization_config);
            let start = if o.grow_control_points { "fewest".to_string() } else { c.num_control_points.to_string() };
            let result = if search.cancelled {
                "cancelled"
            } else if search.met {
                "within the tolerance"
            } else {
                "tolerance not reached"
            };
            let mut pairs = vec![
                ("Target".to_string(), o.tolerance_metric.name().to_string()),
                ("Tolerance".to_string(), number(search.tolerance(), ".3e")),
                ("Start count".to_string(), start),
                ("Control points".to_string(), search.control_points.to_string()),
                ("Max deviation".to_string(), number(search.max_deviation, ".3e")),
                ("Result".to_string(), result.to_string()),
                ("Fits".to_string(), search.fits.to_string()),
                ("Evaluations".to_string(), search.evaluations.to_string()),
                ("Time".to_string(), format_duration(search.elapsed_seconds)),
            ];
            // With an RMS target, its value beside the largest deviation.
            if o.tolerance_metric == ToleranceMetric::RmsDeviation {
                let rms = self.projects.search_result(search).and_then(|r| r.result.report.as_ref()).map(|r| r.rms_deviation);
                if let Some(rms) = rms {
                    pairs.insert(4, ("RMS deviation".to_string(), number(rms, ".3e")));
                }
            }
            return widgets::readout_list(ui, &p, &format!("Tolerance search {}", search.number), &pairs);
        }
        let Some(run) = self.projects.run(id) else { return };
        let (c, o, r) = (&run.spline_config, &run.optimization_config, &run.result);
        let count = r.control_points_per_section();
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
        let searches = self.projects.searches().iter().filter_map(|s| self.search_cells(s));
        for cells in self.projects.runs().iter().map(row).chain(searches) {
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
