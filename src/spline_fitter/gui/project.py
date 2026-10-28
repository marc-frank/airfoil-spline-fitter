"""
Project Manager
===============

Manages project file state (.asfproj) and the project explorer tree panel.

Extracted from SplineFitterWindow to keep project-related logic self-contained.
All window widget access is routed through ``self._window``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING

from PyQt6.QtCore import QDir, Qt
from PyQt6.QtWidgets import QFileDialog, QTreeWidgetItem

from ..optimization.models import (
    FairnessWeights,
    FitResult,
    IterationRecord,
    OptimizationConfig,
    OptimizationRun,
    OptimizerType,
    ProjectFile,
    SplineConfig,
)
from ..project_io import create_optimization_run, create_project, load_project, save_project

if TYPE_CHECKING:
    from .ui import SplineFitterWindow


class ProjectManager:
    """Manages project file state and the project explorer tree."""

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
        self._window._lbl_project.setText("No project")
        self._window._set_project_save_enabled(False)

    # ------------------------------------------------------------------
    # Open / Save
    # ------------------------------------------------------------------

    def on_open_project(self):
        """Open a project file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self._window,
            "Open Project",
            QDir.homePath(),
            "Spline Project (*.asfproj);;All Files (*)",
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

            # Update project info label
            n_runs = len(project.optimization_runs)
            self._window._lbl_project.setText(
                f"Project: {os.path.basename(file_path)}\n"
                f"{n_runs} optimization run(s)"
            )

            # Enable save buttons
            self._window._set_project_save_enabled(True)

            # Update visualization - plot airfoil points
            self._window._plot_manager.clear_plot_items()
            self._window._plot_manager.plot_airfoil(self._window._airfoil.coordinates)

            # If project has runs, load the last one
            if project.optimization_runs:
                last_run = project.optimization_runs[-1]
                self._window._fit_result = last_run.result

                # Restore config to UI
                cfg = last_run.spline_config
                self._window._spin_degree.setValue(cfg.degree)
                self._window._spin_ctrl.setValue(cfg.num_control_points)
                for i in range(self._window._combo_cont.count()):
                    if self._window._combo_cont.itemData(i) == cfg.continuity:
                        self._window._combo_cont.setCurrentIndex(i)
                        break
                self._window._apply_run_config(cfg, last_run.optimization_config)

                # Restore loss history
                self._window._plot_manager.update_loss_data(
                    [r.iteration_number for r in last_run.loss_history],
                    [r.elapsed_seconds for r in last_run.loss_history],
                    [r.error_value for r in last_run.loss_history],
                )

                # Update plots
                self._window._plot_manager.plot_fitted_spline(self._window._fit_result)
                self._window._plot_manager.set_loss_curve_data(
                    self._window._combo_loss_x.currentData()
                )
                self._window._set_save_enabled(True)

                self._window.statusBar().showMessage(
                    f"Loaded project with {n_runs} run(s). "
                    f"Showing: {last_run.optimization_config.optimizer.display_name}"
                )
            else:
                self._window.statusBar().showMessage(
                    f"Loaded project: {os.path.basename(file_path)}"
                )

            # Populate the project tree
            self._populate_tree()

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
            n_runs = len(self._project.optimization_runs)
            self._window._lbl_project.setText(
                f"Project: {os.path.basename(self._project_path)}\n"
                f"{n_runs} optimization run(s)"
            )
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
            "Save Project As",
            default_path,
            "Spline Project (*.asfproj)",
        )

        if not save_path:
            self._window.statusBar().showMessage("Save cancelled.")
            return

        try:
            save_project(self._project, save_path)
            self._project_path = save_path
            self._project_modified = False

            n_runs = len(self._project.optimization_runs)
            self._window._lbl_project.setText(
                f"Project: {os.path.basename(save_path)}\n"
                f"{n_runs} optimization run(s)"
            )

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
    ):
        """Add a completed optimization run to the current project.

        If a run with identical configuration exists, it will be replaced.
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

        # Update project info label
        n_runs = len(self._project.optimization_runs)
        proj_name = (
            os.path.basename(self._project_path) if self._project_path else "Unsaved"
        )
        self._window._lbl_project.setText(
            f"Project: {proj_name}*\n"
            f"{n_runs} optimization run(s)"
        )

        # Refresh the project tree
        self._populate_tree()

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
        """Show or hide the project explorer; the plots take its width."""
        self._window._explorer_panel.setVisible(visible)

    # ------------------------------------------------------------------
    # Project tree
    # ------------------------------------------------------------------

    def _populate_tree(self):
        """Populate the project tree with optimization runs grouped hierarchically."""
        self._window._project_tree.clear()

        if self._project is None or not self._project.optimization_runs:
            return

        # Build hierarchical structure (5 levels):
        #   optimizer -> degree -> sections -> continuity -> control points
        tree_data = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )

        for run in self._project.optimization_runs:
            opt_name = run.optimization_config.optimizer.display_name
            cfg = run.spline_config
            degree_key = f"Degree {cfg.degree}"
            sections_key = (
                f"{cfg.num_sections} section{'s' if cfg.num_sections > 1 else ''}"
            )
            cont_key = ", ".join(dict.fromkeys(cfg.junction_specs())) or cfg.continuity.name
            tree_data[opt_name][degree_key][sections_key][cont_key].append(run)

        # Build tree items
        for opt_name, degrees in sorted(tree_data.items()):
            opt_item = QTreeWidgetItem([f"\U0001f527 {opt_name}"])
            opt_item.setData(0, Qt.ItemDataRole.UserRole, ("optimizer", opt_name))

            for degree_key, sections in sorted(degrees.items()):
                degree_item = QTreeWidgetItem([f"\U0001f4d0 {degree_key}"])
                degree_item.setData(
                    0, Qt.ItemDataRole.UserRole, ("degree", degree_key)
                )

                for sections_key, continuities in sorted(sections.items()):
                    sections_item = QTreeWidgetItem([f"\U0001f4ca {sections_key}"])
                    sections_item.setData(
                        0, Qt.ItemDataRole.UserRole, ("sections", sections_key)
                    )

                    for cont_key, runs in sorted(continuities.items()):
                        cont_item = QTreeWidgetItem([f"\U0001f517 {cont_key}"])
                        cont_item.setData(
                            0,
                            Qt.ItemDataRole.UserRole,
                            ("continuity", cont_key, runs),
                        )

                        # Sort runs by control points
                        runs_sorted = sorted(
                            runs, key=lambda r: r.spline_config.num_control_points
                        )

                        for run in runs_sorted:
                            n_ctrl = run.spline_config.num_control_points
                            error = run.result.final_error
                            # Show smoothing weight for GN-VarPro runs
                            sw_suffix = ""
                            sw_tip = ""
                            tol = run.optimization_config.tolerance
                            if tol is not None:
                                n_ctrl = run.result.num_control_points
                                sw_suffix = f", tol={tol:.1e}"
                                sw_tip = f", Tolerance: {tol:.3e}"
                            elif run.optimization_config.optimizer == OptimizerType.GN_VARPRO:
                                sw = run.optimization_config.smoothing_weight
                                sw_suffix = f", sw={sw:.5g}"
                                sw_tip = f", Smoothing: {sw:.5g}"
                            run_item = QTreeWidgetItem(
                                [f"\U0001f4c4 {n_ctrl} pts ({error:.2e}{sw_suffix})"]
                            )
                            run_item.setData(
                                0, Qt.ItemDataRole.UserRole, ("run", run)
                            )
                            run_item.setToolTip(
                                0,
                                f"Time: {run.result.elapsed_seconds:.2f}s, "
                                f"Evals: {run.result.num_evaluations}"
                                f"{sw_tip}",
                            )
                            cont_item.addChild(run_item)

                        sections_item.addChild(cont_item)
                    degree_item.addChild(sections_item)
                opt_item.addChild(degree_item)
            self._window._project_tree.addTopLevelItem(opt_item)

        # Expand all by default
        self._window._project_tree.expandAll()

    # ------------------------------------------------------------------
    # Tree selection / interaction
    # ------------------------------------------------------------------

    def on_tree_selection_changed(self):
        """Handle tree selection changes."""
        selected_items = self._window._project_tree.selectedItems()

        if not selected_items:
            self._selected_runs = []
            self._window._btn_delete_run.setEnabled(False)
            self._window._lbl_run_details.setText("Select a run to view details")
            self._window._plot_manager.set_comparison_data([], [])
            self._window._plot_manager.set_comparison_marker([], [])
            return

        # Collect all selected runs (including from folder selections)
        runs: list[OptimizationRun] = []
        for item in selected_items:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data is None:
                continue

            if data[0] == "run":
                runs.append(data[1])
            elif data[0] == "continuity":
                # Clicking a continuity folder selects all its runs
                runs.extend(data[2])

        self._selected_runs = runs
        self._window._btn_delete_run.setEnabled(len(runs) > 0)

        if len(runs) == 0:
            self._window._lbl_run_details.setText("Select a run to view details")
            self._window._plot_manager.set_comparison_data([], [])
            self._window._plot_manager.set_comparison_marker([], [])
        elif len(runs) == 1:
            # Single run selected - show details and update main plots
            run = runs[0]
            self._show_run_details(run)
            self._load_run_to_plots(run)
            # Update comparison plot with siblings
            self._update_comparison_plot_for_run(run)
        else:
            # Multiple runs selected - show comparison
            self._window._lbl_run_details.setText(f"{len(runs)} runs selected")
            self._update_comparison_plot(runs)

    def on_tree_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click to load run settings into UI."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None or data[0] != "run":
            return

        run = data[1]
        cfg = run.spline_config
        opt = run.optimization_config

        # Load settings into UI controls
        self._window._spin_degree.setValue(cfg.degree)
        self._window._spin_ctrl.setMaximum(max(self._window._spin_ctrl.maximum(), cfg.num_control_points))
        self._window._spin_ctrl.setValue(cfg.num_control_points)
        self._window._spin_smoothing_weight.setValue(opt.smoothing_weight)

        for i in range(self._window._combo_cont.count()):
            if self._window._combo_cont.itemData(i) == cfg.continuity:
                self._window._combo_cont.setCurrentIndex(i)
                break

        for i in range(self._window._combo_opt.count()):
            if self._window._combo_opt.itemData(i) == opt.optimizer:
                self._window._combo_opt.setCurrentIndex(i)
                break
        self._window._apply_run_config(cfg, opt)

        self._window.statusBar().showMessage(
            f"Loaded settings from run: {cfg.num_control_points} ctrl pts, "
            f"{opt.optimizer.display_name}"
        )

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
        tolerance = f"Tolerance: {opt.tolerance:.3e}<br>" if opt.tolerance else ""
        deviation = ""
        if res.report:
            deviation = (f"Max deviation: {res.report['max_deviation']:.3e}<br>"
                         f"RMS deviation: {res.report['rms_deviation']:.3e}<br>")
        details = (
            f"<b>{opt.optimizer.display_name}</b><br>"
            f"Degree: {cfg.degree}, Ctrl pts: {ctrl}<br>"
            f"Continuity: {continuity}<br>"
            f"{tolerance}"
            f"Metric: {opt.metric.display_name}<br>"
            f"<br>"
            f"<b>Result:</b><br>"
            f"Error: {res.final_error:.4e}<br>"
            f"{deviation}"
            f"Iterations: {res.num_iterations}<br>"
            f"Evaluations: {res.num_evaluations}<br>"
            f"Time: {res.elapsed_seconds:.2f}s<br>"
            f"Converged: {'\u2713' if res.converged else '\u2717'}"
        )
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

    def _update_comparison_plot_for_run(self, selected_run: OptimizationRun):
        """Update comparison plot showing siblings of the selected run."""
        if self._project is None:
            return

        # Find all runs with same optimizer, degree, sections, and continuity
        siblings = []
        for run in self._project.optimization_runs:
            if (
                run.optimization_config.optimizer
                == selected_run.optimization_config.optimizer
                and run.spline_config.degree == selected_run.spline_config.degree
                and run.spline_config.num_sections
                == selected_run.spline_config.num_sections
                and run.spline_config.junction_specs()
                == selected_run.spline_config.junction_specs()
                and (run.optimization_config.tolerance is None)
                == (selected_run.optimization_config.tolerance is None)
            ):
                siblings.append(run)

        if len(siblings) > 1:
            # Detect which parameter varies among siblings
            x, y, label = self._detect_sweep_axis(siblings)
            self._window._plot_manager.set_comparison_label(label)
            self._window._plot_manager.set_comparison_data(x, y)

            # Mark selected run
            sel_x = self._get_run_axis_value(selected_run, label)
            sel_y = selected_run.result.final_error
            self._window._plot_manager.set_comparison_marker([sel_x], [sel_y])
        else:
            self._window._plot_manager.set_comparison_data([], [])
            self._window._plot_manager.set_comparison_marker([], [])

    def _update_comparison_plot(self, runs: list[OptimizationRun]):
        """Update comparison plot for multiple selected runs."""
        if not runs:
            self._window._plot_manager.set_comparison_data([], [])
            self._window._plot_manager.set_comparison_marker([], [])
            return

        x, y, label = self._detect_sweep_axis(runs)
        self._window._plot_manager.set_comparison_label(label)
        self._window._plot_manager.set_comparison_data(x, y)
        # No single selection when multiple selected
        self._window._plot_manager.set_comparison_marker([], [])

    # ------------------------------------------------------------------
    # Sweep axis detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_sweep_axis(
        runs: list[OptimizationRun],
    ) -> tuple:
        """Detect which parameter varies among runs and return (x, y, label).

        Checks smoothing_weight first (float sweep), then falls back to
        num_control_points (integer sweep), then degree.
        """
        cp = ProjectManager._run_control_points
        sw_values = {r.optimization_config.smoothing_weight for r in runs}
        cp_values = {cp(r) for r in runs}
        deg_values = {r.spline_config.degree for r in runs}

        if len(sw_values) > 1 and len(cp_values) == 1 and len(deg_values) == 1:
            label = "Smoothing Weight"
            sorted_runs = sorted(
                runs, key=lambda r: r.optimization_config.smoothing_weight
            )
            x = [r.optimization_config.smoothing_weight for r in sorted_runs]
        elif len(deg_values) > 1 and len(cp_values) == 1:
            label = "Degree"
            sorted_runs = sorted(runs, key=lambda r: r.spline_config.degree)
            x = [r.spline_config.degree for r in sorted_runs]
        else:
            label = "Control Points"
            sorted_runs = sorted(runs, key=cp)
            x = [cp(r) for r in sorted_runs]

        y = [r.result.final_error for r in sorted_runs]
        return x, y, label

    @staticmethod
    def _run_control_points(run: OptimizationRun) -> int:
        """Configured count per section, or the fitted chain's count for tolerance runs."""
        if run.optimization_config.tolerance is not None:
            return run.result.num_control_points
        return run.spline_config.num_control_points

    @staticmethod
    def _get_run_axis_value(run: OptimizationRun, label: str) -> float:
        """Return the x-axis value for a run given the detected axis label."""
        if label == "Smoothing Weight":
            return run.optimization_config.smoothing_weight
        elif label == "Degree":
            return run.spline_config.degree
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

        # Refresh tree
        self._populate_tree()

        # Update project label
        n_runs = len(self._project.optimization_runs)
        proj_name = (
            os.path.basename(self._project_path) if self._project_path else "Unsaved"
        )
        self._window._lbl_project.setText(
            f"Project: {proj_name}*\n"
            f"{n_runs} optimization run(s)"
        )

        self._window.statusBar().showMessage(f"Deleted {count} run(s)")
