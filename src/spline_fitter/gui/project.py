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
from PyQt6.QtWidgets import QFileDialog

from ..formatting import format_duration, number
from ..optimization.models import (
    FairnessWeights,
    FitResult,
    IterationRecord,
    OptimizationConfig,
    OptimizationRun,
    ProjectFile,
    SplineConfig,
)
from ..project_io import create_optimization_run, create_project, load_project, save_project
from .explorer import column, is_number
from .theme import readout_table

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
        self._show_project_label()
        self._window._set_project_save_enabled(False)

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

        if not file_path:
            return

        try:
            project = load_project(file_path)

            # Set project state
            self._project = project
            self._project_path = file_path
            self._project_modified = False

            # Load airfoil from project
            self._window._airfoil = project.airfoil

            # Update UI state
            self._window._show_file_info(project.airfoil, file_path)
            self._window._spin_sections.setValue(project.airfoil.num_sections)

            self._show_project_label()

            # Enable save buttons
            self._window._set_project_save_enabled(True)

            # Update visualization - plot airfoil points
            self._window._plot_manager.clear_plot_items()
            self._window._plot_manager.plot_airfoil(self._window._airfoil.coordinates)

            # If project has runs, load the last one: its settings first, as
            # a different section count re-splits the airfoil
            if project.optimization_runs:
                last_run = project.optimization_runs[-1]
                self._window._apply_run_settings(last_run)
                self._load_run_to_plots(last_run)

                n_runs = len(project.optimization_runs)
                self._window.statusBar().showMessage(
                    f"Loaded project with {n_runs} run{'s' if n_runs != 1 else ''}, "
                    f"showing {last_run.optimization_config.optimizer.display_name}"
                )
            else:
                self._window.statusBar().showMessage(
                    f"Loaded project: {os.path.basename(file_path)}"
                )

            # Fill the explorer and plot its runs
            self._populate_table()
            self.refresh_comparison_plot()

        except Exception as e:
            self._window.statusBar().showMessage(f"Failed to load project: {e}")
            self._window._play_error_chime()

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
            self._window._airfoil.name
            or self._window._airfoil.filename
            or "project"
        )
        default_path = os.path.join(QDir.homePath(), f"{default_name}.asfproj")

        save_path, _ = QFileDialog.getSaveFileName(
            self._window,
            "Save project as",
            default_path,
            "Spline project (*.asfproj)",
        )

        if not save_path:
            self._window.statusBar().showMessage("Save cancelled.")
            return

        try:
            save_project(self._project, save_path)
            self._project_path = save_path
            self._project_modified = False

            self._show_project_label()

            # Enable save button now that we have a path
            self._window._set_project_save_enabled(True)

            self._window.statusBar().showMessage(f"Saved: {save_path}")
        except Exception as e:
            self._window.statusBar().showMessage(f"Save failed: {e}")
            self._window._play_error_chime()

    # ------------------------------------------------------------------
    # Optimization run management
    # ------------------------------------------------------------------

    def add_optimization_run(
        self,
        spline_config: SplineConfig,
        opt_config: OptimizationConfig,
        fairness_weights: FairnessWeights,
        result: FitResult,
        loss_history: list[IterationRecord],
        replot: bool = True,
    ):
        """Add a completed optimization run to the current project.

        If a run with identical configuration exists, it will be replaced.
        ``replot`` redraws the comparison plot from the explorer; a sweep
        draws its own series there instead.
        """
        if self._window._airfoil is None:
            return

        # Create project if needed
        if self._project is None:
            self._project = create_project(self._window._airfoil)
            self._window._set_project_save_enabled(True)

        # Create new run
        run = create_optimization_run(
            spline_config=spline_config,
            optimization_config=opt_config,
            fairness_weights=fairness_weights,
            result=result,
            loss_history=loss_history,
        )

        # Check for duplicate configuration and replace if found
        replaced = False
        for i, existing in enumerate(self._project.optimization_runs):
            if self._configs_match(existing, spline_config, opt_config, fairness_weights):
                self._project.optimization_runs[i] = run
                replaced = True
                break

        if not replaced:
            self._project.optimization_runs.append(run)

        self._project_modified = True
        self._show_project_label()

        self._populate_table()
        if replot:
            self.refresh_comparison_plot()

    def _show_project_label(self):
        """Name the project and count its runs; * marks unsaved changes."""
        if self._project is None:
            self._window._lbl_project.setText("No project")
            return
        name = os.path.basename(self._project_path) if self._project_path else "Unsaved project"
        n_runs = len(self._project.optimization_runs)
        self._window._lbl_project.setText(
            f"{name}{'*' if self._project_modified else ''}\n"
            f"{n_runs} optimization run{'s' if n_runs != 1 else ''}"
        )

    def _configs_match(
        self,
        existing_run: OptimizationRun,
        spline_config: SplineConfig,
        opt_config: OptimizationConfig,
        fairness_weights: FairnessWeights,
    ) -> bool:
        """Check if an existing run has identical configuration."""
        ec = existing_run.spline_config
        eo = existing_run.optimization_config
        ef = existing_run.fairness_weights

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
                or eo.max_points_per_section != opt_config.max_points_per_section
                or eo.knot_placement != opt_config.knot_placement
                or eo.parameterization != opt_config.parameterization
                or eo.optimize_le_position != opt_config.optimize_le_position
                or eo.grow_control_points != opt_config.grow_control_points):
            return False

        # Compare optimization config
        if eo.optimizer != opt_config.optimizer or eo.metric != opt_config.metric:
            return False

        # Compare smoothing weight (distinguishes GN-VarPro sweep runs)
        if eo.smoothing_weight != opt_config.smoothing_weight:
            return False

        # Compare fairness weights
        if (
            ef.length_variance != fairness_weights.length_variance
            or ef.polygon_bend != fairness_weights.polygon_bend
            or ef.anti_bunching != fairness_weights.anti_bunching
            or ef.end_handles != fairness_weights.end_handles
            or ef.curvature_squared != fairness_weights.curvature_squared
            or ef.curvature_gradient != fairness_weights.curvature_gradient
        ):
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

    def on_table_selection_changed(self):
        """Show the selected run (details and plots); mark the selection on
        the comparison plot."""
        runs = self._window._run_table.selected_runs()
        self._selected_runs = runs
        self._window._btn_delete_run.setEnabled(bool(runs))

        if not runs:
            self._window._lbl_run_details.setText("Select a run to view details")
        elif len(runs) == 1:
            self._show_run_details(runs[0])
            self._load_run_to_plots(runs[0])
        else:
            self._window._lbl_run_details.setText(f"{len(runs)} runs selected")
        self.refresh_comparison_plot()

    def on_run_activated(self, run: OptimizationRun):
        """Double-click: apply the run's settings to the sidebar and show it."""
        self._window._apply_run_settings(run)
        self._load_run_to_plots(run)
        cfg, opt = run.spline_config, run.optimization_config
        run_number = self._project.optimization_runs.index(run) + 1
        self._window.statusBar().showMessage(
            f"Applied the settings of run {run_number} "
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
        selected = table.selected_runs()
        marked = [(x, y) for x, y, run in points if any(run is s for s in selected)]
        plots = w._plot_manager
        plots.set_comparison_axes(x_col.name, y_col.name, log_x, log_y, ticks)
        plots.set_comparison_data(xs, [p[1] for p in points], connected=len(set(xs)) == len(xs))
        plots.set_comparison_marker([m[0] for m in marked], [m[1] for m in marked])

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
        rows = [("Degree", str(cfg.degree)), ("Control points", str(ctrl)),
                ("Continuity", continuity)]
        if opt.tolerance:
            rows.append(("Tolerance", f"{number(opt.tolerance, '.3e')}"))
        if opt.optimizer.uses_error_metric:
            rows.append(("Error metric", opt.metric.display_name))
        rows.append(("Error", f"{number(res.final_error, '.3e')}"))
        if res.report:
            rows += [("Max deviation", f"{number(res.report['max_deviation'], '.3e')}"),
                     ("RMS deviation", f"{number(res.report['rms_deviation'], '.3e')}")]
        rows += [("Iterations", str(res.num_iterations)),
                 ("Evaluations", str(res.num_evaluations)),
                 ("Time", format_duration(res.elapsed_seconds)),
                 ("Converged", "yes" if res.converged else "no")]
        details = readout_table(rows, title=opt.optimizer.display_name)
        self._window._lbl_run_details.setText(details)

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

    # ------------------------------------------------------------------
    # Sweep series
    # ------------------------------------------------------------------

    def sweep_series(self, spline_config: SplineConfig, opt_config: OptimizationConfig,
                     axis: str) -> tuple[list, list]:
        """Errors of all project runs along a sweep axis, as (x, y) sorted by x.

        A run belongs to the series when it matches the given configuration
        in everything but the swept parameter, so earlier sweeps over other
        ranges show next to the running one. Of runs at the same value, the
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
        by_value: dict = {}
        for run in self._project.optimization_runs:
            c, o = run.spline_config, run.optimization_config
            # Along the sections axis the junction count changes: compare
            # the continuity setting instead of the junctions themselves
            if axis == "Sections":
                same_junctions = (c.continuity == spline_config.continuity
                                  and c.geometric_continuity == spline_config.geometric_continuity)
            else:
                same_junctions = c.junction_specs() == spline_config.junction_specs()
            if (o.optimizer != opt_config.optimizer
                    or not same_junctions
                    or (o.tolerance is None) != (opt_config.tolerance is None)
                    or o.knot_placement != opt_config.knot_placement
                    or o.parameterization != opt_config.parameterization
                    or any(get(c, o) != get(spline_config, opt_config)
                           for name, get in fixed.items() if name != axis)):
                continue
            by_value[self._get_run_axis_value(run, axis)] = run.result.final_error
        x = sorted(by_value)
        return x, [by_value[v] for v in x]

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
        """Delete selected runs from the project."""
        if not self._selected_runs or self._project is None:
            return

        count = len(self._selected_runs)

        # Remove runs from project
        for run in self._selected_runs:
            if run in self._project.optimization_runs:
                self._project.optimization_runs.remove(run)

        self._project_modified = True
        self._selected_runs = []

        self._populate_table()
        self._show_project_label()
        self.on_table_selection_changed()

        self._window.statusBar().showMessage(f"Deleted {count} run{'s' if count != 1 else ''}")
