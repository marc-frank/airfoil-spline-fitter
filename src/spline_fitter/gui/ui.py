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
from ..geometry import create_airfoil_data
from ..optimization.models import (
    AirfoilData,
    FairnessWeights,
    FitResult,
    OptimizationConfig,
    OptimizerType,
    SplineConfig,
)
from .batch import BatchController
from .file_actions import FileActionsMixin
from .fitting import FittingController
from .layout import WindowLayoutMixin
from .plotting import PlotManager
from .project import ProjectManager
from .widgets import SweepableSpinBox

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

    # -- Optimizer Visibility Logic ----------------------------------------

    def _on_optimizer_changed(self, index: int):
        """Show/hide controls that depend on which optimizer is selected."""
        opt_type = self._combo_opt.currentData()
        is_algebraic = opt_type.is_algebraic if opt_type else False
        is_param_opt = opt_type.is_param_optimizer if opt_type else False
        is_gn_varpro = opt_type == OptimizerType.GN_VARPRO if opt_type else False

        self._lbl_maxiter.setVisible(not is_algebraic)
        self._spin_maxiter.setVisible(not is_algebraic)
        self._lbl_smoothing.setVisible(is_algebraic)
        self._spin_smoothing.setVisible(is_algebraic)
        tolerance_mode = is_gn_varpro and self._check_tolerance_mode.isChecked()
        self._tolerance_container.setVisible(is_gn_varpro)
        self._spin_tolerance.setEnabled(tolerance_mode and not self._check_auto_tolerance.isChecked())
        self._check_auto_tolerance.setEnabled(tolerance_mode)
        self._lbl_smoothing_weight.setVisible(is_gn_varpro and not tolerance_mode)
        self._sweep_smoothing_weight.setVisible(is_gn_varpro and not tolerance_mode)
        if tolerance_mode:
            for sweep_box in (self._sweep_ctrl, self._sweep_smoothing_weight):
                if sweep_box.is_sweep_active:
                    sweep_box.stop_sweep()
        self._sweep_ctrl.setEnabled(not self._is_fitting and not tolerance_mode)
        self._sweep_smoothing_weight.setEnabled(not self._is_fitting)
        self._param_opt_container.setVisible(is_param_opt)
        self._check_orth_lsq_local_bounds.setVisible(
            opt_type == OptimizerType.ORTHOGONAL_LSQ
        )
        geometric_ok = is_gn_varpro
        self._combo_cont_kind.setEnabled(geometric_ok and not self._is_fitting)
        self._edit_junctions.setEnabled(geometric_ok and not self._is_fitting)

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

        # Save buttons
        self._btn_save_dat.clicked.connect(self._on_save_fitted)
        self._btn_save_spline.clicked.connect(self._on_save_spline)
        self._btn_export_cad.clicked.connect(self._on_export_cad)

        # Tolerance mode
        self._check_tolerance_mode.toggled.connect(
            lambda _: self._on_optimizer_changed(self._combo_opt.currentIndex())
        )
        self._check_auto_tolerance.toggled.connect(
            lambda _: (self._update_auto_tolerance(),
                       self._on_optimizer_changed(self._combo_opt.currentIndex()))
        )

        # Spline configuration
        self._spin_degree.valueChanged.connect(self._update_ctrl_minimum)
        self._combo_cont.currentIndexChanged.connect(self._update_ctrl_minimum)

        # Display options
        self._slider_live.valueChanged.connect(self._update_live_mode_label)
        self._combo_loss_x.currentIndexChanged.connect(self._update_loss_plot_axis)
        self._combo_log_scale.currentIndexChanged.connect(self._update_loss_plot_log_scale)
        self._check_comb.stateChanged.connect(self._update_curvature_comb)
        self._check_knots.toggled.connect(self._update_knots)
        self._combo_comb_qty.currentIndexChanged.connect(self._refresh_curvature_comb)
        self._spin_comb_scale.valueChanged.connect(self._refresh_curvature_comb)
        self._spin_comb_density.valueChanged.connect(self._refresh_curvature_comb)

        # Section count changes
        self._spin_sections.valueChanged.connect(self._on_sections_changed)

        # Fairness
        self._check_fairness.toggled.connect(
            lambda checked: self._fairness_container.setEnabled(checked)
        )

        # Run controls
        self._btn_fit.clicked.connect(self._fitting_controller.on_fit)
        self._btn_cancel.clicked.connect(self._fitting_controller.on_cancel)

        # Settings persistence
        self._spin_bbox.valueChanged.connect(self._save_settings)

        # Project explorer
        self._btn_toggle_explorer.clicked.connect(self._project_manager.on_toggle_explorer)
        self._project_tree.itemSelectionChanged.connect(self._project_manager.on_tree_selection_changed)
        self._project_tree.itemDoubleClicked.connect(self._project_manager.on_tree_item_double_clicked)
        self._btn_delete_run.clicked.connect(self._project_manager.on_delete_selected_runs)

        # Initialize optimizer-dependent UI state
        self._on_optimizer_changed(self._combo_opt.currentIndex())

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

        # Get smoothing factor for algebraic methods
        smoothing = None
        if opt_type.is_algebraic:
            smoothing = self._spin_smoothing.value()

        tolerance = None
        if opt_type == OptimizerType.GN_VARPRO and self._check_tolerance_mode.isChecked():
            if self._check_auto_tolerance.isChecked() and self._airfoil is not None:
                self._update_auto_tolerance()
            tolerance = float(self._spin_tolerance.value())

        return OptimizationConfig(
            optimizer=opt_type,
            metric=self._combo_err.currentData(),
            max_iterations=self._spin_maxiter.value(),
            seed=self._spin_seed.value(),
            optimize_le_position=self._check_optimize_le.isChecked(),
            smoothing_factor=smoothing,
            smoothing_weight=self._spin_smoothing_weight.value(),
            tolerance=tolerance,
        )

    def _apply_run_config(self, cfg: SplineConfig, opt: OptimizationConfig) -> None:
        """Show the continuity and tolerance settings of a stored run."""
        for i in range(self._combo_cont_kind.count()):
            if self._combo_cont_kind.itemData(i) == cfg.geometric_continuity:
                self._combo_cont_kind.setCurrentIndex(i)
                break
        self._edit_junctions.setText(", ".join(cfg.junction_continuity))
        self._check_tolerance_mode.setChecked(opt.tolerance is not None)
        if opt.tolerance is not None:
            self._check_auto_tolerance.setChecked(False)
            self._spin_tolerance.setValue(opt.tolerance)

    def _get_param_opt_settings(self) -> OptimizerType | None:
        """Get parameterization optimizer settings from UI state.

        Returns:
            Outer optimizer type for param optimizer
        """
        return self._combo_param_opt_outer.currentData()

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
                f"({self._airfoil.num_points} pts, {num_sections} sections: {section_sizes})"
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

    def _update_live_mode_label(self):
        """Update the live mode label when slider changes."""
        mode_names = ["Minimal", "Moderate", "Detailed"]
        value = self._slider_live.value()
        self._lbl_live_mode.setText(mode_names[value])

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
        """Enable/disable UI elements during fitting."""
        self._is_fitting = fitting

        # Toggle buttons / menu actions
        self._btn_fit.setEnabled(not fitting)
        self._btn_cancel.setEnabled(fitting)
        self._action_open.setEnabled(not fitting)
        self._action_batch.setEnabled(not fitting)

        # Toggle controls
        self._spin_degree.setEnabled(not fitting)
        self._spin_ctrl.setEnabled(not fitting)
        self._spin_sections.setEnabled(not fitting)
        self._combo_cont.setEnabled(not fitting)
        self._combo_opt.setEnabled(not fitting)
        self._slider_live.setEnabled(not fitting)
        self._tolerance_container.setEnabled(not fitting)
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
            self._plot_manager.clear_comb_items()
        else:
            self._refresh_curvature_comb()

    def _refresh_curvature_comb(self):
        """Refresh the curvature comb visualization."""
        if not self._check_comb.isChecked():
            return
        self._plot_manager.refresh_curvature_comb(
            self._fit_result,
            self._spin_comb_scale.value(),
            self._spin_comb_density.value(),
            self._combo_comb_qty.currentData() or 0,
        )
