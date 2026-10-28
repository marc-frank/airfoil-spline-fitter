"""
User Interface Components
=========================

PyQt6-based graphical user interface for the spline fitter.

This module provides:
    - SplineFitterWindow: Main application window
    - UI building utilities for consistent widget creation
    - Event handlers for user interactions
    - Live visualization during optimization
"""

from __future__ import annotations

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QMainWindow,
)

from ..analysis import default_tolerance
from ..formatting import format_duration, format_error, number
from ..geometry import create_airfoil_data
from ..optimization.models import (
    AirfoilData,
    FairnessWeights,
    FitResult,
    OptimizationConfig,
    OptimizationRun,
    OptimizerType,
    SplineConfig,
)
from .batch import BatchController
from .file_actions import FileActionsMixin
from .fitting import FittingController
from .layout import WindowLayoutMixin
from .motion import install_smooth_scrolling
from .plotting import PlotManager
from .project import ProjectManager
from .theme import apply_theme, style_plot, theme_changed
from .widgets import SweepableSpinBox, ThemeDialog

# =============================================================================
# Main Window
# =============================================================================

class SplineFitterWindow(WindowLayoutMixin, FileActionsMixin, QMainWindow):
    """
    Main application window for airfoil spline fitting.

    Provides:
        - File loading and saving
        - Spline parameter configuration
        - Optimization control
        - Real-time visualization
        - Batch processing
    """

    def __init__(self):
        # The Loft theme (theme.py) styles every widget and plot built below
        apply_theme()
        # Wheel scrolling glides in every list, table, panel and document
        install_smooth_scrolling()
        super().__init__()
        self.setWindowTitle("Airfoil Spline Fitter")

        # Application state
        self._airfoil: AirfoilData | None = None
        self._fit_result: FitResult | None = None
        self._is_fitting = False
        self._cancel_requested = False

        # Settings persistence
        self._settings = QSettings("MSFra", "AirfoilSplineFitter")

        # Sweep group (mutual exclusion for parameter sweep toggles)
        self._sweep_group: list = []

        # Build UI
        self._build_ui()

        # Create plot manager (after UI is built, owns all plot state)
        self._plot_manager = PlotManager(
            self._plot, self._loss_plot, self._comparison_plot,
        )

        # Create controllers
        self._project_manager = ProjectManager(self)
        self._fitting_controller = FittingController(self)
        self._batch_controller = BatchController(self)

        self._connect_signals()
        self._restore_settings()
        self._set_advanced_baseline()

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_sidebar_sized", False):
            self._sidebar_sized = True
            self.size_sidebar()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._panels_dragged:
            self.reset_panel_widths()

    def closeEvent(self, event):
        """Stop a running fit so its worker thread ends with the window."""
        if self._is_fitting:
            self._cancel_requested = True
        super().closeEvent(event)

    # -- Optimizer Visibility Logic ----------------------------------------

    def _on_optimizer_changed(self, index: int):
        """Show the settings the selected optimizer reads; disable the ones
        another setting rules out (goal, growth, section count)."""
        opt_type = self._combo_opt.currentData()
        if opt_type is None:
            return
        is_gn_varpro = opt_type == OptimizerType.GN_VARPRO
        uses_metric = opt_type.uses_error_metric
        tolerance_mode = is_gn_varpro and self._tolerance_mode_selected()
        has_junctions = self._spin_sections.value() > 1
        editable = not self._is_fitting

        def show(visible: bool, *widgets):
            for w in widgets:
                w.setVisible(visible)

        def enable(enabled: bool, *widgets):
            for w in widgets:
                w.setEnabled(enabled)

        # Method. With a tolerance goal the control point count is only the
        # starting count, and only when the points are not grown
        start_count = tolerance_mode and not self._check_grow.isChecked()
        show(is_gn_varpro, self._lbl_goal, self._seg_goal,
             self._lbl_tolerance, self._tolerance_row,
             self._lbl_knot_placement, self._combo_knot_placement,
             self._lbl_grow, self._check_grow)
        show(opt_type == OptimizerType.DIRECT_LSQ,
             self._lbl_parameterization, self._seg_parameterization)
        enable(tolerance_mode, self._lbl_tolerance)
        self._spin_tolerance.setEnabled(
            editable and tolerance_mode and not self._check_auto_tolerance.isChecked())
        self._check_auto_tolerance.setEnabled(editable and tolerance_mode)
        uses_count = not tolerance_mode or start_count
        self._lbl_ctrl.setText("Start count" if tolerance_mode else "Control points")
        enable(uses_count, self._lbl_ctrl)
        if tolerance_mode:
            for sweep_box in (self._sweep_ctrl, self._sweep_smoothing_weight):
                if sweep_box.is_sweep_active:
                    sweep_box.stop_sweep()
        self._sweep_ctrl.setEnabled(editable and uses_count)
        self._sweep_ctrl.set_sweep_allowed(not tolerance_mode)
        show(uses_metric, self._lbl_err, self._seg_err)

        # Curve. Continuity applies at junctions, which one section lacks;
        # geometric continuity is VarPro only
        show(is_gn_varpro, self._lbl_cont_kind, self._combo_cont_kind)
        enable(has_junctions, self._lbl_cont, self._lbl_cont_kind)
        enable(editable and has_junctions, self._combo_cont, self._combo_cont_kind)
        self._update_continuity_labels()

        # Advanced
        show(is_gn_varpro, self._lbl_junctions, self._edit_junctions,
             self._lbl_subsample, self._subsample_row,
             self._lbl_smoothing_weight, self._sweep_smoothing_weight)
        enable(has_junctions, self._lbl_junctions, self._edit_junctions)
        # Tolerance mode uses its own fixed smoothing weight
        enable(not tolerance_mode, self._lbl_smoothing_weight, self._sweep_smoothing_weight)

        # VarPro runs until it converges (Cancel stops it); the algebraic
        # methods do not iterate
        show(not opt_type.is_algebraic and not is_gn_varpro,
             self._lbl_maxiter, self._spin_maxiter)

        # Seed: the stochastic global optimizers
        show(opt_type.is_global, self._lbl_seed, self._spin_seed)

        # Search scale: the bounds, which only bounded optimizers receive
        show(uses_metric and self._optimizer_takes_bounds(opt_type),
             self._lbl_bbox, self._spin_bbox)

        # Junction position: every method can move it
        enable(has_junctions, self._lbl_optimize_le, self._check_optimize_le)

        # Fairness penalties: the objective of the general-purpose optimizers
        show(uses_metric, self._regularization_box)
        self._update_advanced_badge()

    @staticmethod
    def _optimizer_takes_bounds(opt_type: OptimizerType) -> bool:
        """Whether the optimizer receives the variable bounds (Search scale)."""
        from ..optimization.optimizers import get_optimizer
        try:
            return get_optimizer(opt_type).supports_bounds
        except Exception:
            return False

    def _tolerance_mode_selected(self) -> bool:
        """Whether the Goal is a tolerance (VarPro) rather than a fixed count."""
        return bool(self._seg_goal.currentData())

    def _update_continuity_labels(self, *_):
        """Label the continuity levels G0-G3 or C0-C3 by the continuity type."""
        geometric = (self._combo_opt.currentData() == OptimizerType.GN_VARPRO
                     and bool(self._combo_cont_kind.currentData()))
        prefix = "G" if geometric else "C"
        for i in range(self._combo_cont.count()):
            self._combo_cont.setItemText(i, f"{prefix}{i}")

    def _update_auto_tolerance(self):
        """Fill in the automatic tolerance for the loaded airfoil."""
        if self._airfoil is None or not self._check_auto_tolerance.isChecked():
            return
        try:
            tol = default_tolerance(self._airfoil.coordinates, self._airfoil.precision)
            self._spin_tolerance.setValue(tol)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def _connect_signals(self):
        """Connect UI signals to handlers."""
        # Menu bar actions
        self._action_open.triggered.connect(self._on_open_file)
        self._action_save_dat.triggered.connect(self._on_save_fitted)
        self._action_save_spline.triggered.connect(self._on_save_spline)
        self._action_open_project.triggered.connect(self._project_manager.on_open_project)
        self._action_save_project.triggered.connect(self._project_manager.on_save_project)
        self._action_save_project_as.triggered.connect(self._project_manager.on_save_project_as)
        self._action_batch.triggered.connect(self._batch_controller.on_batch_dialog)
        self._action_view_report.triggered.connect(self._on_view_report)
        self._action_export_cad.triggered.connect(self._on_export_cad)
        self._action_fit_report.triggered.connect(self._on_fit_report)

        # File card and save buttons
        self._btn_open.clicked.connect(self._on_open_file)
        self._btn_save_dat.clicked.connect(self._on_save_fitted)
        self._btn_save_spline.clicked.connect(self._on_save_spline)
        self._btn_export_cad.clicked.connect(self._on_export_cad)

        # Goal (tolerance or fixed count)
        self._seg_goal.currentIndexChanged.connect(
            lambda _: self._on_optimizer_changed(self._combo_opt.currentIndex())
        )
        self._check_grow.toggled.connect(
            lambda _: self._on_optimizer_changed(self._combo_opt.currentIndex())
        )
        self._check_auto_tolerance.toggled.connect(
            lambda _: (self._update_auto_tolerance(),
                       self._on_optimizer_changed(self._combo_opt.currentIndex()))
        )
        self._check_subsample.toggled.connect(self._spin_subsample.setEnabled)

        # Spline configuration
        self._spin_degree.valueChanged.connect(self._update_ctrl_minimum)
        self._combo_cont.currentIndexChanged.connect(self._update_ctrl_minimum)
        self._combo_cont_kind.currentIndexChanged.connect(self._update_continuity_labels)

        # Display options
        self._combo_loss_x.currentIndexChanged.connect(self._update_loss_plot_axis)
        self._combo_log_scale.currentIndexChanged.connect(self._update_loss_plot_log_scale)
        self._check_comb.toggled.connect(self._update_curvature_comb)
        self._check_knots.toggled.connect(self._update_knots)
        self._combo_comb_qty.currentIndexChanged.connect(self._refresh_curvature_comb)
        self._spin_comb_scale.valueChanged.connect(self._refresh_curvature_comb)
        self._spin_comb_density.valueChanged.connect(self._refresh_curvature_comb)

        # Section count changes (a single section has no junction settings)
        self._spin_sections.valueChanged.connect(self._on_sections_changed)
        self._spin_sections.valueChanged.connect(
            lambda _: self._on_optimizer_changed(self._combo_opt.currentIndex())
        )

        # Fairness
        self._check_fairness.toggled.connect(
            lambda checked: self._fairness_container.setEnabled(checked)
        )

        # Run controls
        self._btn_fit.clicked.connect(self._fitting_controller.on_fit)
        self._btn_cancel.clicked.connect(self._fitting_controller.on_cancel)

        # Settings persistence
        self._spin_bbox.valueChanged.connect(self._save_settings)

        # Theme: the stylesheet restyles the widgets, the plots restyle here
        self._action_theme.triggered.connect(lambda: ThemeDialog(self).exec())
        theme_changed.changed.connect(self._on_theme_changed)

        # Project explorer
        self._action_toggle_explorer.toggled.connect(self._project_manager.on_toggle_explorer)
        self._btn_toggle_explorer.clicked.connect(
            lambda: self._action_toggle_explorer.setChecked(False))
        self._btn_show_explorer.clicked.connect(
            lambda: self._action_toggle_explorer.setChecked(True))
        table = self._run_table
        table.selection_changed.connect(self._project_manager.on_table_selection_changed)
        table.run_activated.connect(self._project_manager.on_run_activated)
        table.filter_changed.connect(self._project_manager.refresh_comparison_plot)
        table.plot_column_requested.connect(self._project_manager.on_plot_column_requested)
        for combo in (self._combo_runs_x, self._combo_runs_y):
            combo.currentIndexChanged.connect(self._project_manager.refresh_comparison_plot)
        self._btn_delete_run.clicked.connect(self._project_manager.on_delete_selected_runs)

        # Initialize optimizer-dependent UI state
        self._on_optimizer_changed(self._combo_opt.currentIndex())

    def _on_theme_changed(self):
        """Restyle the plots and their items after a theme change."""
        for plot in (self._plot, self._loss_plot, self._comparison_plot):
            style_plot(plot)
        self._plot_manager.restyle()

    # -------------------------------------------------------------------------
    # Settings Persistence
    # -------------------------------------------------------------------------

    def _restore_settings(self):
        """Restore settings from persistent storage."""
        try:
            bbox = self._settings.value("bbox_scale", 3.0, type=float)
            if bbox is not None:
                self._spin_bbox.setValue(float(bbox))
        except Exception:
            pass

    def _save_settings(self):
        """Save current settings to persistent storage."""
        try:
            self._settings.setValue("bbox_scale", float(self._spin_bbox.value()))
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Configuration Helpers
    # -------------------------------------------------------------------------

    def _get_spline_config(self) -> SplineConfig:
        """Build SplineConfig from current UI state."""
        is_gn_varpro = self._combo_opt.currentData() == OptimizerType.GN_VARPRO
        junctions = ()
        if is_gn_varpro:
            text = self._edit_junctions.text().replace(",", " ").replace(";", " ")
            junctions = tuple(text.split())
        return SplineConfig(
            degree=self._spin_degree.value(),
            num_control_points=self._spin_ctrl.value(),
            num_sections=self._spin_sections.value(),
            continuity=self._combo_cont.currentData(),
            bbox_scale=self._spin_bbox.value(),
            geometric_continuity=is_gn_varpro and bool(self._combo_cont_kind.currentData()),
            junction_continuity=junctions,
        )

    def _get_optimization_config(self) -> OptimizationConfig:
        """Build OptimizationConfig from current UI state."""
        opt_type = self._combo_opt.currentData()

        tolerance = None
        if opt_type == OptimizerType.GN_VARPRO and self._tolerance_mode_selected():
            if self._check_auto_tolerance.isChecked() and self._airfoil is not None:
                self._update_auto_tolerance()
            tolerance = float(self._spin_tolerance.value())

        return OptimizationConfig(
            optimizer=opt_type,
            metric=self._seg_err.currentData(),
            # 0: VarPro's own limit, a safety stop it does not reach in practice
            max_iterations=0 if opt_type == OptimizerType.GN_VARPRO else self._spin_maxiter.value(),
            seed=self._spin_seed.value(),
            optimize_le_position=(self._check_optimize_le.isChecked()
                                  and self._spin_sections.value() > 1),
            smoothing_weight=self._spin_smoothing_weight.value(),
            tolerance=tolerance,
            max_points_per_section=(self._spin_subsample.value()
                                    if opt_type == OptimizerType.GN_VARPRO
                                    and self._check_subsample.isChecked() else None),
            knot_placement=self._combo_knot_placement.currentData(),
            parameterization=self._seg_parameterization.currentData(),
            grow_control_points=self._check_grow.isChecked(),
        )

    def _apply_run_settings(self, run: OptimizationRun) -> None:
        """Set every sidebar setting to that of a stored run (a double-click
        in the explorer, or opening a project)."""
        cfg, opt, weights = run.spline_config, run.optimization_config, run.fairness_weights

        def choose(control, data) -> None:
            index = control.findData(data)
            if index >= 0:
                control.setCurrentIndex(index)

        # Method; the optimizer first, as it decides which settings apply
        choose(self._combo_opt, opt.optimizer)
        choose(self._seg_goal, opt.tolerance is not None)
        if opt.tolerance is not None:
            self._check_auto_tolerance.setChecked(False)
            self._spin_tolerance.setValue(opt.tolerance)
        choose(self._combo_knot_placement, opt.knot_placement)
        choose(self._seg_parameterization, opt.parameterization)
        self._check_grow.setChecked(opt.grow_control_points)
        choose(self._seg_err, opt.metric)

        # Curve; degree and continuity before the count, whose minimum they set
        self._spin_sections.setValue(cfg.num_sections)
        self._spin_degree.setValue(cfg.degree)
        choose(self._combo_cont, cfg.continuity)
        choose(self._combo_cont_kind, cfg.geometric_continuity)
        self._spin_ctrl.setMaximum(max(self._spin_ctrl.maximum(), cfg.num_control_points))
        self._spin_ctrl.setValue(cfg.num_control_points)

        # Advanced
        self._edit_junctions.setText(", ".join(cfg.junction_continuity))
        self._check_subsample.setChecked(opt.max_points_per_section is not None)
        if opt.max_points_per_section is not None:
            self._spin_subsample.setValue(opt.max_points_per_section)
        self._spin_smoothing_weight.setValue(opt.smoothing_weight)
        if opt.max_iterations:  # 0: VarPro's own limit, not a setting
            self._spin_maxiter.setValue(opt.max_iterations)
        self._spin_seed.setValue(opt.seed)
        self._spin_bbox.setValue(cfg.bbox_scale)
        self._check_optimize_le.setChecked(opt.optimize_le_position)

        # Regularization
        self._check_fairness.setChecked(weights.is_enabled)
        for spin, weight in ((self._spin_w_lenvar, weights.length_variance),
                             (self._spin_w_bend, weights.polygon_bend),
                             (self._spin_w_short, weights.anti_bunching),
                             (self._spin_w_tangent, weights.end_handles),
                             (self._spin_w_curv, weights.curvature_squared),
                             (self._spin_w_grad, weights.curvature_gradient)):
            spin.setValue(weight)
        self._on_optimizer_changed(self._combo_opt.currentIndex())

    def _get_fairness_weights(self) -> FairnessWeights:
        """Build FairnessWeights from current UI state."""
        if not self._check_fairness.isChecked():
            return FairnessWeights()

        return FairnessWeights(
            length_variance=self._spin_w_lenvar.value(),
            polygon_bend=self._spin_w_bend.value(),
            anti_bunching=self._spin_w_short.value(),
            end_handles=self._spin_w_tangent.value(),
            curvature_squared=self._spin_w_curv.value(),
            curvature_gradient=self._spin_w_grad.value(),
        )

    def _on_sections_changed(self):
        """Handle section count change - re-split the loaded airfoil."""
        if self._airfoil is None:
            return

        try:
            num_sections = self._spin_sections.value()

            # Re-create airfoil data with new section count
            self._airfoil = create_airfoil_data(
                list(self._airfoil.coordinates),
                filename=self._airfoil.filename,
                name=self._airfoil.name,
                precision=self._airfoil.precision,
                num_sections=num_sections,
            )

            # Clear previous fit (no longer valid)
            self._fit_result = None
            self._set_save_enabled(False)

            # Update status
            section_sizes = [len(sec) for sec in self._airfoil.sections]
            self.statusBar().showMessage(
                f"Re-split: {self._airfoil.name or self._airfoil.filename} "
                f"({self._airfoil.num_points} points, {num_sections} sections: {section_sizes})"
            )

            # Refresh the plot to show the new split (clear fitted curves)
            self._plot_manager.plot_airfoil(self._airfoil.coordinates)

        except Exception as e:
            self.statusBar().showMessage(f"Failed to re-split airfoil: {e}")
            self._play_error_chime()

    def _update_ctrl_minimum(self):
        """Update minimum control points based on degree and continuity."""
        try:
            degree = self._spin_degree.value()
            mode = self._combo_cont.currentData()
            if mode is None:
                return

            min_ctrl = mode.minimum_control_points(degree)
            self._spin_ctrl.setMinimum(min_ctrl)

            if self._spin_ctrl.value() < min_ctrl:
                self._spin_ctrl.setValue(min_ctrl)
        except Exception:
            pass

    def _update_loss_plot_axis(self):
        """Update loss plot X-axis based on combo selection."""
        mode = self._combo_loss_x.currentData()
        if mode == "iters":
            self._plot_manager.set_loss_x_label("Iterations")
        else:  # time
            self._plot_manager.set_loss_x_label("Time (s)")
        self._plot_manager.set_loss_curve_data(mode)

    def _update_loss_plot_log_scale(self):
        """Update loss plot log scale based on combo selection."""
        scale = self._combo_log_scale.currentData()
        log_x = scale in ("both", "x")
        log_y = scale in ("both", "y")
        self._plot_manager.set_loss_log_scale(log_x, log_y)

    # -------------------------------------------------------------------------
    # UI State Management
    # -------------------------------------------------------------------------

    def _set_save_enabled(self, enabled: bool):
        """Enable/disable all save-related controls."""
        self._btn_save_dat.setEnabled(enabled)
        self._btn_save_spline.setEnabled(enabled)
        self._btn_export_cad.setEnabled(enabled)
        self._action_save_dat.setEnabled(enabled)
        self._action_save_spline.setEnabled(enabled)
        self._action_export_cad.setEnabled(enabled)
        self._action_fit_report.setEnabled(enabled)
        self._update_result_summary(enabled)

    def _update_result_summary(self, show: bool):
        """Show the current fit's key numbers in the run bar."""
        result = self._fit_result
        show = show and result is not None
        self._result_box.setVisible(show)
        if not show:
            return
        if result.report:
            self._lbl_result_caption_dev.setText("Max deviation")
            self._lbl_result_dev.setText(f"{number(result.report['max_deviation'], '.2e')}")
        else:
            self._lbl_result_caption_dev.setText("Error")
            self._lbl_result_dev.setText(format_error(result.final_error))
        self._lbl_result_ctrl.setText(
            " + ".join(str(len(cp)) for cp in result.section_control_points)
        )
        self._lbl_result_time.setText(format_duration(result.elapsed_seconds))

    def _show_file_info(self, airfoil: AirfoilData, path: str = ""):
        """Show the loaded airfoil in the sidebar's file card."""
        self._lbl_file_name.setText(airfoil.name or airfoil.filename or "Airfoil")
        self._lbl_path.setText(f"{airfoil.num_points} points \u00B7 {airfoil.precision} decimals")
        self._lbl_file_name.setToolTip(path)
        self._lbl_path.setToolTip(path)

    def _set_project_save_enabled(self, enabled: bool):
        """Enable/disable project save controls."""
        self._action_save_project.setEnabled(enabled)
        self._action_save_project_as.setEnabled(enabled)

    def _get_active_sweep(self) -> SweepableSpinBox | None:
        """Return the SweepableSpinBox that has sweep mode active, or None."""
        for sweep_box in self._sweep_group:
            if sweep_box.is_sweep_active:
                return sweep_box
        return None

    def _set_fitting_ui(self, fitting: bool):
        """Enable/disable UI elements during fitting.

        Every fit setting must be locked here while a fit runs; new sidebar
        controls belong in the list below (Advanced is locked as a whole).
        """
        self._is_fitting = fitting

        # Toggle buttons / menu actions (Cancel takes the Fit button's place)
        self._btn_fit.setEnabled(not fitting)
        self._btn_fit.setVisible(not fitting)
        self._btn_cancel.setEnabled(fitting)
        self._btn_cancel.setVisible(fitting)
        self._btn_open.setEnabled(not fitting)
        self._action_open.setEnabled(not fitting)
        self._action_batch.setEnabled(not fitting)

        # Toggle controls
        self._sweep_degree.setEnabled(not fitting)
        self._spin_ctrl.setEnabled(not fitting)
        self._sweep_sections.setEnabled(not fitting)
        self._combo_cont.setEnabled(not fitting)
        self._combo_cont_kind.setEnabled(not fitting)
        self._combo_opt.setEnabled(not fitting)
        self._seg_goal.setEnabled(not fitting)
        self._tolerance_row.setEnabled(not fitting)
        self._combo_knot_placement.setEnabled(not fitting)
        self._seg_parameterization.setEnabled(not fitting)
        self._check_grow.setEnabled(not fitting)
        self._seg_err.setEnabled(not fitting)
        self._advanced_section.content_widget().setEnabled(not fitting)
        self._on_optimizer_changed(self._combo_opt.currentIndex())

        # Save buttons depend on fit result
        has_result = self._fit_result is not None
        self._set_save_enabled(not fitting and has_result)

    # -------------------------------------------------------------------------
    # Visualization (thin wrappers — logic in PlotManager)
    # -------------------------------------------------------------------------

    def _update_curvature_comb(self):
        """Show or hide curvature comb based on checkbox."""
        if not self._check_comb.isChecked():
            self._plot_manager.comb_settings = None
            self._plot_manager.clear_comb_items()
        else:
            self._refresh_curvature_comb()

    def _refresh_curvature_comb(self):
        """Refresh the curvature comb visualization (also used by the live preview)."""
        if not self._check_comb.isChecked():
            return
        settings = (
            self._spin_comb_scale.value(),
            self._spin_comb_density.value(),
            self._combo_comb_qty.currentData() or 0,
        )
        self._plot_manager.comb_settings = settings
        self._plot_manager.refresh_curvature_comb(self._fit_result, *settings)
