"""
Project Manager
===============

Manages project file state (.asfproj) and the project explorer: the table
of every run (explorer.py, whose docstring states what it is for), the
comparison plot of the runs that pass its filter, and applying a run's
settings to the sidebar.

Extracted from SplineFitterWindow to keep project-related logic self-contained.
All window widget access is routed through ``self._window``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PyQt6.QtCore import QDir
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ..analysis import scipy_curves
from ..formatting import file_stem, format_duration, number
from ..geometry import create_airfoil_data
from ..optimization.models import (
    FitResult,
    IterationRecord,
    OptimizationConfig,
    OptimizationRun,
    ProjectFile,
    SplineConfig,
)
from ..project_io import (
    create_optimization_run,
    create_project,
    load_project,
    number_runs,
    save_project,
)
from .explorer import COLUMNS, column, is_number
from .file_actions import PROJECT_SUFFIX
from .history import set_widget_value, widget_value

# The window title, after the project's name
APP_NAME = "Airfoil Spline Fitter"

if TYPE_CHECKING:
    from .ui import SplineFitterWindow


class ProjectManager:
    """Manages project file state and the project explorer table."""

    def __init__(self, window: SplineFitterWindow):
        self._window = window
        self._project: ProjectFile | None = None
        self._project_path: str | None = None
        self._project_modified = False
        self._selected_runs: list[OptimizationRun] = []
        # The runs of each delete, with their places, for Restore deleted runs
        self._deleted: list[list[tuple[int, OptimizationRun]]] = []
        # The highest run number given in this project: never given again,
        # also when that run is deleted (it may be restored)
        self._last_number = 0
        # (x, y, run) of the comparison plot's points; None while the plot
        # shows something else (a sweep) or the table changed since
        self._comparison_points: list | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def project(self) -> ProjectFile | None:
        return self._project

    @property
    def project_path(self) -> str | None:
        return self._project_path

    @property
    def is_modified(self) -> bool:
        return self._project_modified

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Reset project state (called when loading a new airfoil)."""
        self._project = None
        self._project_path = None
        self._project_modified = False
        self._deleted, self._last_number = [], 0
        self._show_project_label()
        self._window._set_project_save_enabled(False)
        # The explorer lists no runs of the previous airfoil
        self._populate_table()
        self.refresh_comparison_plot()
        self.update_run_buttons()

    def confirm_discard(self) -> bool:
        """Whether the open project may be closed: asks to save its unsaved
        runs first. False when the user cancels, or the save does not happen."""
        if self._project is None or not self._project_modified:
            return True
        answer = self._ask_to_save()
        if answer == QMessageBox.StandardButton.Save:
            self.on_save_project()
            return not self._project_modified
        return answer == QMessageBox.StandardButton.Discard

    def _ask_to_save(self) -> QMessageBox.StandardButton:
        """Ask whether to save the unsaved runs: Save, Discard or Cancel."""
        name = os.path.basename(self._project_path) if self._project_path else "the project"
        box = QMessageBox(self._window)
        box.setWindowTitle("Unsaved runs")
        box.setText(f"Save the changes to {name}?")
        box.setInformativeText("Runs that are not saved are lost.")
        Button = QMessageBox.StandardButton
        box.setStandardButtons(Button.Save | Button.Discard | Button.Cancel)
        box.button(Button.Discard).setText("Don't save")
        box.setDefaultButton(Button.Save)
        box.setEscapeButton(Button.Cancel)
        return Button(box.exec())

    # ------------------------------------------------------------------
    # Open / Save
    # ------------------------------------------------------------------

    def on_open_project(self):
        """Open a project file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self._window,
            "Open project",
            QDir.homePath(),
            "Spline project (*.asfproj);;All files (*)",
        )

        if file_path:
            self.open_project(file_path)

    def open_project(self, file_path: str) -> bool:
        """Open the project at ``file_path``; whether it opened.

        All or nothing: what showing the project computes from its file is
        computed first, and should showing it fail all the same, the project
        open before is restored as it was.
        """
        w = self._window
        if w._is_fitting:
            w.statusBar().showMessage("Cancel the running fit to open a file")
            return False
        try:
            project = load_project(file_path)
            self._check_project(project)
        except Exception as e:
            w.statusBar().showMessage(f"Failed to load project: {e}")
            w._play_error_chime()
            return False
        # Asked only once the file has loaded: a broken file closes nothing
        if not self.confirm_discard():
            w.statusBar().showMessage("Open cancelled.")
            return False

        before = self._capture()
        try:
            self._show_project(project, file_path)
        except Exception as e:
            self._restore(before)
            w.statusBar().showMessage(f"Failed to load project: {e}")
            w._play_error_chime()
            return False
        w._add_recent_file(file_path)
        return True

    @staticmethod
    def _check_project(project: ProjectFile) -> None:
        """Compute what showing a project computes from its file, before
        anything changes: the explorer's cells of every run, the airfoil
        split as it opens, and the curves of the run it shows. Raises on a
        project that cannot be shown."""
        for run in project.optimization_runs:
            for c in COLUMNS:
                c.text(c.cell(run))
        runs, airfoil = project.optimization_runs, project.airfoil
        splits = {airfoil.num_sections} | {run.spline_config.num_sections for run in runs[-1:]}
        for sections in splits:
            create_airfoil_data(list(airfoil.coordinates), filename=airfoil.filename,
                                name=airfoil.name, precision=airfoil.precision,
                                num_sections=sections)
        if runs:
            scipy_curves(runs[-1].result)

    def _show_project(self, project: ProjectFile, file_path: str) -> None:
        """Make ``project`` the open project: its airfoil, its last run's
        settings and fit, and its runs in the explorer."""
        w = self._window
        self._project = project
        self._project_path = file_path
        self._project_modified = False
        self._deleted = []
        self._last_number = max((run.number for run in project.optimization_runs), default=0)

        # Load airfoil from project; the previous airfoil's fit goes with it
        w._airfoil = project.airfoil
        w._fit_result = None
        w._set_save_enabled(False)
        w._limit_sections(project.airfoil.num_points)
        w._show_file_info(project.airfoil, file_path)
        w._spin_sections.setValue(project.airfoil.num_sections)
        self._show_project_label()
        w._set_project_save_enabled(True)

        w._plot_manager.clear_plot_items()
        w._plot_manager.plot_airfoil(w._airfoil.coordinates)

        # If project has runs, load the last one: its settings, then its fit
        # (the airfoil keeps its own split; the next fit applies the settings')
        if project.optimization_runs:
            last_run = project.optimization_runs[-1]
            w._apply_run_settings(last_run)
            self._load_run_to_plots(last_run)
            n_runs = len(project.optimization_runs)
            w.statusBar().showMessage(
                f"Loaded project with {n_runs} run{'s' if n_runs != 1 else ''}, "
                f"showing {last_run.optimization_config.optimizer.display_name}"
            )
        else:
            w.statusBar().showMessage(f"Loaded project: {os.path.basename(file_path)}")

        # Fill the explorer and plot its runs
        self._populate_table()
        self.refresh_comparison_plot()
        self.update_run_buttons()

    def _capture(self) -> dict:
        """What opening a project changes, to restore when it fails."""
        w = self._window
        return {
            "project": (self._project, self._project_path, self._project_modified,
                        self._deleted, self._last_number),
            "airfoil": w._airfoil,
            "fit": w._fit_result,
            "sections_maximum": w._spin_sections.maximum(),
            "settings": [widget_value(x) for x in w._settings_widgets],
            "history": w._settings_history.checkpoint(),
            "file_card": [(label.text(), label.toolTip())
                          for label in (w._lbl_file_name, w._lbl_path)],
            "selected": list(self._selected_runs),
        }

    def _restore(self, state: dict) -> None:
        """Put back what ``_capture`` took: the project, airfoil, fit,
        settings, file card, plots and explorer."""
        w = self._window
        (self._project, self._project_path, self._project_modified,
         self._deleted, self._last_number) = state["project"]
        # The section maximum before the settings, so the count fits under it
        w._spin_sections.setMaximum(state["sections_maximum"])
        with w._settings_history.quiet():
            for widget, value in zip(w._settings_widgets, state["settings"], strict=True):
                set_widget_value(widget, value)
        w._on_optimizer_changed(w._combo_opt.currentIndex())
        # The failed open's settings were never a step
        w._settings_history.rollback(state["history"])
        w._airfoil = state["airfoil"]
        for label, (text, tip) in zip((w._lbl_file_name, w._lbl_path), state["file_card"],
                                      strict=True):
            label.setText(text)
            label.setToolTip(tip)
        w._set_project_save_enabled(self._project is not None)
        self._show_project_label()

        # The explorer, its selection quietly: the fit below is the one shown
        self._populate_table()
        table = w._run_table
        table.view.selectionModel().blockSignals(True)
        table.select_runs(state["selected"])
        table.view.selectionModel().blockSignals(False)
        self._show_selection()
        self.refresh_comparison_plot()

        w._plot_manager.clear_plot_items()
        if w._airfoil is not None:
            w._plot_manager.plot_airfoil(w._airfoil.coordinates)
        fit, run = state["fit"], self.run_of(state["fit"])
        if run is not None:
            self._load_run_to_plots(run)
        else:
            w._fit_result = fit
            if fit is not None:
                w._plot_manager.plot_fitted_spline(fit)
                w._refresh_curvature_comb()
            w._set_save_enabled(fit is not None)

    def run_of(self, result: FitResult | None) -> OptimizationRun | None:
        """The run of the open project whose fit ``result`` is."""
        if self._project is None or result is None:
            return None
        return next((run for run in self._project.optimization_runs if run.result is result),
                    None)

    def on_save_project(self):
        """Save project to current path."""
        if self._project is None or self._project_path is None:
            self.on_save_project_as()
            return

        try:
            save_project(self._project, self._project_path)
            self._project_modified = False
            self._show_project_label()
            self._window.statusBar().showMessage(f"Saved: {self._project_path}")
        except Exception as e:
            self._window.statusBar().showMessage(f"Save failed: {e}")
            self._window._play_error_chime()

    def on_save_project_as(self):
        """Save project to a new path."""
        if self._window._airfoil is None:
            self._window.statusBar().showMessage("No airfoil loaded to save.")
            return

        # Create project if needed
        if self._project is None:
            self._project = create_project(self._window._airfoil)

        default_name = (
            self._window._airfoil.filename
            or self._window._airfoil.name
            or "project"
        )
        default_path = os.path.join(QDir.homePath(), f"{file_stem(default_name)}{PROJECT_SUFFIX}")

        save_path, _ = QFileDialog.getSaveFileName(
            self._window,
            "Save project as",
            default_path,
            "Spline project (*.asfproj)",
        )

        if not save_path:
            self._window.statusBar().showMessage("Save cancelled.")
            return
        # Only the suffix makes it open as a project again
        if not save_path.lower().endswith(PROJECT_SUFFIX):
            save_path += PROJECT_SUFFIX
            # The file dialog asked about the name without it
            if os.path.exists(save_path) and not self._confirm_replace(save_path):
                self._window.statusBar().showMessage("Save cancelled.")
                return

        try:
            save_project(self._project, save_path)
            self._project_path = save_path
            self._project_modified = False

            self._show_project_label()

            # Enable save button now that we have a path
            self._window._set_project_save_enabled(True)
            self._window._add_recent_file(save_path)

            self._window.statusBar().showMessage(f"Saved: {save_path}")
        except Exception as e:
            self._window.statusBar().showMessage(f"Save failed: {e}")
            self._window._play_error_chime()

    def _confirm_replace(self, path: str) -> bool:
        """Whether to replace the existing file at ``path``."""
        Button = QMessageBox.StandardButton
        answer = QMessageBox.question(
            self._window, "Save project as",
            f"{os.path.basename(path)} already exists. Replace it?",
            Button.Yes | Button.No, Button.No)
        return answer == Button.Yes

    # ------------------------------------------------------------------
    # Optimization run management
    # ------------------------------------------------------------------

    def add_optimization_run(
        self,
        spline_config: SplineConfig,
        opt_config: OptimizationConfig,
        result: FitResult,
        loss_history: list[IterationRecord],
        replot: bool = True,
    ) -> bool:
        """Add a completed optimization run to the current project; whether
        it was added.

        If a run with identical configuration exists, it will be replaced,
        unless the new run was cancelled and the existing one was not: a
        finished run is never lost to a partial one.
        ``replot`` redraws the comparison plot from the explorer; a sweep
        draws its own series there instead.
        """
        if self._window._airfoil is None:
            return False
        if result.cancelled and self._project is not None and any(
                not existing.result.cancelled
                and self._configs_match(existing, spline_config, opt_config, result)
                for existing in self._project.optimization_runs):
            return False

        # Create project if needed
        if self._project is None:
            self._project = create_project(self._window._airfoil)
            self._window._set_project_save_enabled(True)

        # Create new run
        run = create_optimization_run(
            spline_config=spline_config,
            optimization_config=opt_config,
            result=result,
            loss_history=loss_history,
        )

        # Check for duplicate configuration and replace if found; the new run
        # takes the number of the one it replaces
        replaced = False
        for i, existing in enumerate(self._project.optimization_runs):
            if self._configs_match(existing, spline_config, opt_config, result):
                run.number = existing.number
                self._project.optimization_runs[i] = run
                replaced = True
                break

        if not replaced:
            self._project.optimization_runs.append(run)
            self._last_number = number_runs(self._project.optimization_runs, self._last_number)

        self._project_modified = True
        self._show_project_label()

        self._populate_table()
        if replot:
            self.refresh_comparison_plot()
        return True

    def _show_project_label(self):
        """Name the project and count its runs, also in the window title;
        * marks unsaved changes."""
        w = self._window
        if self._project is None:
            w._lbl_project.setText("No project")
            w.setWindowTitle(APP_NAME)
            w.setWindowModified(False)
            return
        name = os.path.basename(self._project_path) if self._project_path else "Unsaved project"
        n_runs = len(self._project.optimization_runs)
        w._lbl_project.setText(
            f"{name}{'*' if self._project_modified else ''}\n"
            f"{n_runs} optimization run{'s' if n_runs != 1 else ''}"
        )
        # Qt shows the * in place of [*] while the window is modified
        w.setWindowTitle(f"{name}[*] - {APP_NAME}")
        w.setWindowModified(self._project_modified)

    def _configs_match(
        self,
        existing_run: OptimizationRun,
        spline_config: SplineConfig,
        opt_config: OptimizationConfig,
        result: FitResult | None = None,
    ) -> bool:
        """Check if an existing run has identical configuration. The fits
        of a tolerance search share their settings: with ``result`` they
        match only at the same control point counts."""
        ec = existing_run.spline_config
        eo = existing_run.optimization_config

        # Compare spline config
        if (
            ec.degree != spline_config.degree
            or ec.num_control_points != spline_config.num_control_points
            or ec.num_sections != spline_config.num_sections
            or ec.continuity != spline_config.continuity
            or ec.geometric_continuity != spline_config.geometric_continuity
            or ec.junction_continuity != spline_config.junction_continuity
        ):
            return False

        if (eo.tolerance != opt_config.tolerance
                or eo.knot_placement != opt_config.knot_placement
                or eo.parameterization != opt_config.parameterization
                or eo.optimize_le_position != opt_config.optimize_le_position
                or eo.grow_control_points != opt_config.grow_control_points):
            return False
        if opt_config.knot_placement == "free" and eo.min_knot_gap != opt_config.min_knot_gap:
            return False
        if (opt_config.tolerance is not None and result is not None
                and ([len(s) for s in existing_run.result.section_control_points]
                     != [len(s) for s in result.section_control_points])):
            return False

        # Compare optimization config
        if eo.optimizer != opt_config.optimizer:
            return False

        # Compare smoothing weight (distinguishes GN-VarPro sweep runs); a
        # tolerance goal uses its own fixed weight, not the setting
        if opt_config.tolerance is None and eo.smoothing_weight != opt_config.smoothing_weight:
            return False

        return True

    # ------------------------------------------------------------------
    # Explorer toggle
    # ------------------------------------------------------------------

    def on_toggle_explorer(self, visible: bool):
        """Show or hide the project explorer; the plots take its width, and
        the rail at the window edge offers it back."""
        self._window._explorer_panel.setVisible(visible)
        self._window._explorer_rail.setVisible(not visible)

    # ------------------------------------------------------------------
    # Run table
    # ------------------------------------------------------------------

    def _populate_table(self):
        """Show every run of the project in the explorer table."""
        runs = self._project.optimization_runs if self._project else []
        self._window._run_table.set_runs(runs)
        self._comparison_points = None
        # The table drops selected runs that are gone without a signal (and a
        # sweep must keep its plot): follow it in the buttons and details
        if not self._same_runs(self._window._run_table.selected_runs(), self._selected_runs):
            self._show_selection()

    @staticmethod
    def _same_runs(a: list[OptimizationRun], b: list[OptimizationRun]) -> bool:
        """The same run objects: runs compare equal by value, so a reloaded
        copy of a run would pass for it."""
        return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))

    def _show_selection(self):
        """Take the table's selected runs: enable the run buttons and show
        their details."""
        runs = self._window._run_table.selected_runs()
        self._selected_runs = runs
        self.update_run_buttons()
        if not runs:
            self._window._run_details.set_message("Select a run to view details")
        elif len(runs) == 1:
            self._show_run_details(runs[0])
        else:
            self._window._run_details.set_message(f"{len(runs)} runs selected")

    def update_run_buttons(self):
        """Delete and Load settings act on the selected runs, Restore deleted
        runs on the last delete; none during a fit."""
        idle = not self._window._is_fitting
        self._window._btn_delete_run.setEnabled(idle and bool(self._selected_runs))
        self._window._btn_load_run.setEnabled(idle and len(self._selected_runs) == 1)
        self._window._action_restore_runs.setEnabled(idle and bool(self._deleted))

    def on_table_selection_changed(self):
        """Show the selected run (details and plots); mark the selection on
        the comparison plot. During a fit the plots show the fit."""
        self._show_selection()
        if len(self._selected_runs) == 1 and not self._window._is_fitting:
            self._load_run_to_plots(self._selected_runs[0])
        self._mark_selected_runs()

    def on_load_selected_run(self):
        """Load settings: as a double-click on the one selected run."""
        if len(self._selected_runs) == 1:
            self.on_run_activated(self._selected_runs[0])

    def on_run_activated(self, run: OptimizationRun):
        """Double-click: apply the run's settings to the sidebar and show it."""
        if self._window._is_fitting:
            self._window.statusBar().showMessage("Cancel the running fit to load a run")
            return
        if self._project is None:
            return
        if not any(r is run for r in self._project.optimization_runs):
            return  # not a run of this project
        self._window._apply_run_settings(run)
        self._load_run_to_plots(run)
        cfg, opt = run.spline_config, run.optimization_config
        self._window.statusBar().showMessage(
            f"Applied the settings of run {run.number} "
            f"({opt.optimizer.display_name}, degree {cfg.degree}, "
            f"{cfg.num_control_points} control points, {cfg.num_sections} sections)"
        )

    # ------------------------------------------------------------------
    # Comparison plot: the runs that pass the explorer's filter
    # ------------------------------------------------------------------

    def on_plot_column_requested(self, axis: str, name: str):
        """A column header's context menu: plot along that column."""
        combo = self._window._combo_runs_x if axis == "x" else self._window._combo_runs_y
        combo.setCurrentText(name)

    def show_axis(self, name: str):
        """Name the swept column on the x axis (and the error on the y axis)
        without redrawing the runs; the sweep draws its own series."""
        for combo, text in ((self._window._combo_runs_x, name),
                            (self._window._combo_runs_y, "Error")):
            combo.blockSignals(True)
            combo.setCurrentText(text)
            combo.blockSignals(False)
        self._window._plot_manager.set_comparison_label(name)
        self._comparison_points = None

    def refresh_comparison_plot(self):
        """Plot the runs that pass the explorer's filter: the chosen fit
        metric along the chosen column, the selected runs marked.

        Numbers keep their value (on a log axis for columns spanning decades,
        while all values are positive); a column of text or yes/no gets one
        position per value, in alphabetical order. Runs without a value in
        either column are left out. The points are joined by a line unless
        several share an x value.
        """
        w = self._window
        table = w._run_table
        x_col = column(w._combo_runs_x.currentText())
        y_col = column(w._combo_runs_y.currentText())
        rows = table.visible_rows()
        points = [(x, y, table.model.run(r))
                  for r, x, y in zip(rows, table.values(x_col.name, rows),
                                     table.values(y_col.name, rows), strict=True)
                  if x is not None and is_number(y)]

        ticks = None
        if any(not is_number(x) for x, _, _ in points):
            labels = sorted({x_col.text(x) for x, _, _ in points}, key=str.casefold)
            position = {text: i for i, text in enumerate(labels)}
            points = [(position[x_col.text(x)], y, run) for x, y, run in points]
            ticks = list(enumerate(labels))
        log_x = x_col.log and not ticks and all(x > 0 for x, _, _ in points)
        log_y = y_col.log and all(y > 0 for _, y, _ in points)

        points.sort(key=lambda p: (p[0], p[1]))
        xs = [p[0] for p in points]
        plots = w._plot_manager
        plots.set_comparison_axes(x_col.name, y_col.name, log_x, log_y, ticks)
        plots.set_comparison_data(xs, [p[1] for p in points], connected=len(set(xs)) == len(xs))
        self._comparison_points = points
        self._mark_selected_runs()

    def _mark_selected_runs(self):
        """Mark the selected runs on the comparison plot; only the marker
        moves, unless the plot is not of the explorer's runs."""
        if self._comparison_points is None:
            self.refresh_comparison_plot()
            return
        selected = {id(run) for run in self._window._run_table.selected_runs()}
        marked = [(x, y) for x, y, run in self._comparison_points if id(run) in selected]
        self._window._plot_manager.set_comparison_marker([m[0] for m in marked],
                                                         [m[1] for m in marked])

    # ------------------------------------------------------------------
    # Run details / plot helpers
    # ------------------------------------------------------------------

    def _show_run_details(self, run: OptimizationRun):
        """Display details for a single run."""
        cfg = run.spline_config
        opt = run.optimization_config
        res = run.result

        continuity = ", ".join(cfg.junction_specs()) or cfg.continuity.name
        ctrl = res.num_control_points if opt.tolerance else cfg.num_control_points
        settings = [("Degree", str(cfg.degree)), ("Control points", str(ctrl)),
                    ("Continuity", continuity)]
        if opt.tolerance:
            settings.append(("Tolerance", f"{number(opt.tolerance, '.3e')}"))
        fit = [("Error", f"{number(res.final_error, '.3e')}")]
        if res.report:
            fit += [("Max deviation", f"{number(res.report['max_deviation'], '.3e')}"),
                    ("RMS deviation", f"{number(res.report['rms_deviation'], '.3e')}")]
        effort = [("Iterations", str(res.num_iterations)),
                  ("Evaluations", str(res.num_evaluations)),
                  ("Time", format_duration(res.elapsed_seconds)),
                  ("Converged", "yes" if res.converged else "no")]
        self._window._run_details.set_rows(opt.optimizer.display_name,
                                           settings + fit + effort)

    def _load_run_to_plots(self, run: OptimizationRun):
        """Load a run's data into the main plots."""
        # Update fit result
        self._window._fit_result = run.result

        # Update loss history
        self._window._plot_manager.update_loss_data(
            [r.iteration_number for r in run.loss_history],
            [r.elapsed_seconds for r in run.loss_history],
            [r.error_value for r in run.loss_history],
        )

        # Update plots
        self._window._plot_manager.plot_fitted_spline(self._window._fit_result)
        self._window._plot_manager.set_loss_curve_data(
            self._window._combo_loss_x.currentData()
        )
        self._window._refresh_curvature_comb()

        # Enable save buttons
        self._window._set_save_enabled(True)

    def _clear_fit(self):
        """Take the shown fit off the plots, as it is no longer a run."""
        w = self._window
        w._fit_result = None
        w._plot_manager.clear_plot_items()
        w._plot_manager.update_loss_data([], [], [])
        w._plot_manager.set_loss_curve_data(w._combo_loss_x.currentData())
        w._set_save_enabled(False)

    # ------------------------------------------------------------------
    # Sweep series
    # ------------------------------------------------------------------

    def sweep_series(self, spline_config: SplineConfig, opt_config: OptimizationConfig,
                     swept: tuple[str, ...], axis: str) -> tuple[list, list]:
        """Errors of all project runs along a sweep axis, as (x, y) sorted by x.

        A run belongs to the series when it matches the given configuration
        in everything but the swept parameters, so earlier sweeps over other
        ranges show next to the running one. With several swept parameters
        several runs share an x value. Of runs at the same values, the
        latest counts.
        """
        if self._project is None:
            return [], []
        fixed = {
            "Control points": lambda c, o: c.num_control_points,
            "Degree": lambda c, o: c.degree,
            "Sections": lambda c, o: c.num_sections,
            "Smoothing weight": lambda c, o: o.smoothing_weight,
        }
        by_values: dict = {}
        for run in self._project.optimization_runs:
            c, o = run.spline_config, run.optimization_config
            # Along the sections axis the junction count changes: compare
            # the continuity setting instead of the junctions themselves
            if "Sections" in swept:
                same_junctions = (c.continuity == spline_config.continuity
                                  and c.geometric_continuity == spline_config.geometric_continuity)
            else:
                same_junctions = c.junction_specs() == spline_config.junction_specs()
            if (o.optimizer != opt_config.optimizer
                    or not same_junctions
                    or (o.tolerance is None) != (opt_config.tolerance is None)
                    or o.knot_placement != opt_config.knot_placement
                    or (o.knot_placement == "free" and o.min_knot_gap != opt_config.min_knot_gap)
                    or o.parameterization != opt_config.parameterization
                    or any(get(c, o) != get(spline_config, opt_config)
                           for name, get in fixed.items() if name not in swept)):
                continue
            values = tuple(self._get_run_axis_value(run, name) for name in swept)
            by_values[values] = (self._get_run_axis_value(run, axis),
                                 run.result.comparison_error)
        points = sorted(by_values.values())
        return [p[0] for p in points], [p[1] for p in points]

    @staticmethod
    def _run_control_points(run: OptimizationRun) -> int:
        """Configured count per section, or the fitted chain's count for tolerance runs."""
        if run.optimization_config.tolerance is not None:
            return run.result.num_control_points
        return run.spline_config.num_control_points

    @staticmethod
    def _get_run_axis_value(run: OptimizationRun, label: str) -> float:
        """Return the x-axis value for a run given the detected axis label."""
        if label == "Smoothing weight":
            return run.optimization_config.smoothing_weight
        elif label == "Degree":
            return run.spline_config.degree
        elif label == "Sections":
            return run.spline_config.num_sections
        else:
            return ProjectManager._run_control_points(run)

    # ------------------------------------------------------------------
    # Delete runs
    # ------------------------------------------------------------------

    def on_delete_selected_runs(self):
        """Delete the selected runs from the project, once confirmed; Edit >
        Restore deleted runs brings them back."""
        if not self._selected_runs or self._project is None or self._window._is_fitting:
            return

        # These objects, not runs equal to them
        selected = {id(run) for run in self._selected_runs}
        runs = self._project.optimization_runs
        doomed = [(i, run) for i, run in enumerate(runs) if id(run) in selected]
        if not doomed:
            self._show_selection()
            return
        count = len(doomed)
        if not self._confirm_delete(count):
            return

        shown = any(run.result is self._window._fit_result for _, run in doomed)
        runs[:] = [run for run in runs if id(run) not in selected]
        self._deleted.append(doomed)
        self._project_modified = True
        if shown:
            self._clear_fit()

        self._populate_table()
        self._show_project_label()
        self.on_table_selection_changed()

        self._window.statusBar().showMessage(
            f"Deleted {count} run{'s' if count != 1 else ''}; "
            "Edit > Restore deleted runs brings them back")

    def _confirm_delete(self, count: int) -> bool:
        """Whether to delete ``count`` runs."""
        box = QMessageBox(self._window)
        box.setWindowTitle("Delete runs")
        box.setText(f"Delete {count} run{'s' if count != 1 else ''} from the project?")
        box.setInformativeText("Edit > Restore deleted runs brings them back.")
        delete = box.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(delete)
        box.exec()
        return box.clickedButton() is delete

    def on_restore_deleted_runs(self):
        """Put the runs of the last delete back in their places, selected."""
        if not self._deleted or self._project is None or self._window._is_fitting:
            return
        restored = self._deleted.pop()
        runs = self._project.optimization_runs
        # In the order they were in: each at its old place among the others
        for i, run in restored:
            runs.insert(min(i, len(runs)), run)
        self._project_modified = True
        self._populate_table()
        self._show_project_label()
        self._window._run_table.select_runs([run for _, run in restored])
        self.update_run_buttons()
        count = len(restored)
        self._window.statusBar().showMessage(
            f"Restored {count} run{'s' if count != 1 else ''}")
