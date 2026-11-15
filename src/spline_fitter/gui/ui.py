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

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow,
)

from ..analysis import default_tolerance
from ..formatting import format_duration, format_error, number
from ..optimization.models import (
    AirfoilData,
    FitResult,
    OptimizationConfig,
    OptimizationRun,
    OptimizerType,
    SplineConfig,
    minimum_control_points,
)
from .batch import BatchController
from .file_actions import FileActionsMixin
from .fitting import FittingController
from .history import SettingsHistory
from .layout import WindowLayoutMixin
from .motion import install_smooth_scrolling
from .plotting import PlotManager
from .project import APP_NAME, ProjectManager
from .theme import apply_theme, style_plot, theme_changed
from .widgets import MotionDialog, SweepableSpinBox, ThemeDialog

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
        self.setWindowTitle(APP_NAME)

        # Application state
        self._airfoil: AirfoilData | None = None
        self._fit_result: FitResult | None = None
        self._is_fitting = False
        self._cancel_requested = False

        # Settings persistence
        self._settings = QSettings("MSFra", "AirfoilSplineFitter")

        # Sweep group (mutual exclusion for parameter sweep toggles)
        self._sweep_group: list = []
        # Set while the curve settings are made to fit each other (and until
        # the signals are connected)
        self._resolving_curve = True

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
        self._set_advanced_baseline()

        # Undo and redo of the fit settings, in the order they are applied
        # (as in _apply_run_settings: what limits a setting comes first); a
        # project that fails to open restores them in the same order
        self._settings_widgets = [
            self._combo_opt, self._seg_goal, self._check_auto_tolerance,
            self._spin_tolerance, self._combo_knot_placement, self._spin_knot_gap,
            self._seg_parameterization, self._check_grow, self._spin_sections,
            self._spin_degree, self._combo_cont, self._combo_cont_kind, self._spin_ctrl,
            self._edit_junctions,
            self._spin_smoothing_weight, self._check_optimize_le, *self._sweep_group]
        self._settings_history = SettingsHistory(
            self._settings_widgets,
            is_quiet=lambda: self._is_fitting,
            after_restore=lambda: self._on_optimizer_changed(self._combo_opt.currentIndex()),
            parent=self,
        )
        self._settings_history.changed.connect(self._update_undo_actions)
        self._update_undo_actions()

        # Airfoil and project files dropped anywhere on the window open
        self.setAcceptDrops(True)

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
        """Ask to save unsaved runs; stop a running fit so its worker thread
        ends with the window."""
        if not self._project_manager.confirm_discard():
            event.ignore()
            return
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
        show(is_gn_varpro, self._lbl_goal, self._seg_goal,
             self._lbl_tolerance, self._tolerance_row,
             self._lbl_knot_placement, self._combo_knot_placement,
             self._lbl_knot_gap, self._spin_knot_gap,
             self._lbl_grow, self._check_grow)
        show(opt_type == OptimizerType.DIRECT_LSQ,
             self._lbl_parameterization, self._seg_parameterization)
        enable(tolerance_mode, self._lbl_tolerance)
        self._spin_tolerance.setEnabled(
            editable and tolerance_mode and not self._check_auto_tolerance.isChecked())
        self._check_auto_tolerance.setEnabled(editable and tolerance_mode)
        free_knots = is_gn_varpro and self._combo_knot_placement.currentData() == "free"
        enable(free_knots, self._lbl_knot_gap)
        self._spin_knot_gap.setEnabled(editable and free_knots)
        uses_count = self._count_is_used()
        self._lbl_ctrl.setText("Start count" if tolerance_mode else "Control points")
        enable(uses_count, self._lbl_ctrl)
        if tolerance_mode:
            for sweep_box in (self._sweep_ctrl, self._sweep_smoothing_weight):
                if sweep_box.is_sweep_active:
                    sweep_box.stop_sweep()
        self._sweep_ctrl.setEnabled(editable and uses_count)
        self._sweep_ctrl.set_sweep_allowed(not tolerance_mode)

        # Curve. Continuity applies at junctions, which one section lacks;
        # geometric continuity is VarPro only
        show(is_gn_varpro, self._lbl_cont_kind, self._combo_cont_kind)
        enable(has_junctions, self._lbl_cont, self._lbl_cont_kind)
        enable(editable and has_junctions, self._combo_cont, self._combo_cont_kind)
        self._update_continuity_labels()

        # Advanced
        show(is_gn_varpro, self._lbl_junctions, self._edit_junctions,
             self._lbl_smoothing_weight, self._sweep_smoothing_weight)
        enable(has_junctions, self._lbl_junctions, self._edit_junctions)
        # Tolerance mode uses its own fixed smoothing weight
        enable(not tolerance_mode, self._lbl_smoothing_weight, self._sweep_smoothing_weight)

        # Junction position: every method can move it
        enable(has_junctions, self._lbl_optimize_le, self._check_optimize_le)

        self._update_advanced_badge()
        # The count may have become used, and Per junction entries may apply
        self._keep_curve_valid()

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
        self._action_new_naca.triggered.connect(self._on_new_naca)
        self._menu_recent.aboutToShow.connect(self._populate_recent_menu)
        self._action_paste_airfoil.triggered.connect(self._on_paste_airfoil)
        self._action_undo.triggered.connect(lambda: self._settings_history.undo())
        self._action_redo.triggered.connect(lambda: self._settings_history.redo())
        self._action_save_dat.triggered.connect(self._on_save_fitted)
        self._action_save_spline.triggered.connect(self._on_save_spline)
        self._action_open_project.triggered.connect(self._project_manager.on_open_project)
        self._action_save_project.triggered.connect(self._project_manager.on_save_project)
        self._action_save_project_as.triggered.connect(self._project_manager.on_save_project_as)
        self._action_batch.triggered.connect(self._batch_controller.on_batch_dialog)
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
        self._combo_knot_placement.currentIndexChanged.connect(
            lambda _: self._on_optimizer_changed(self._combo_opt.currentIndex())
        )
        self._check_auto_tolerance.toggled.connect(
            lambda _: (self._update_auto_tolerance(),
                       self._on_optimizer_changed(self._combo_opt.currentIndex()))
        )

        # Spline configuration
        for changed, signals in (
            ("degree", (self._spin_degree.valueChanged, self._sweep_degree.sweep_changed)),
            ("count", (self._spin_ctrl.valueChanged, self._sweep_ctrl.sweep_changed)),
            ("sections", (self._spin_sections.valueChanged, self._sweep_sections.sweep_changed)),
            ("continuity", (self._combo_cont.currentIndexChanged,)),
        ):
            for signal in signals:
                signal.connect(lambda *_, changed=changed: self._keep_curve_valid(changed))
        self._edit_junctions.editingFinished.connect(self._keep_curve_valid)
        self._resolving_curve = False
        self._keep_curve_valid()
        self._combo_cont_kind.currentIndexChanged.connect(self._update_continuity_labels)

        # Display options
        self._combo_loss_x.currentIndexChanged.connect(self._update_loss_plot_axis)
        self._combo_log_scale.currentIndexChanged.connect(self._update_loss_plot_log_scale)
        self._check_comb.toggled.connect(self._update_curvature_comb)
        self._check_knots.toggled.connect(self._update_knots)
        self._check_deviation.toggled.connect(self._update_deviation)
        self._spin_deviation_height.valueChanged.connect(self._update_deviation)
        self._combo_comb_qty.currentIndexChanged.connect(self._refresh_curvature_comb)
        self._spin_comb_scale.valueChanged.connect(self._refresh_curvature_comb)
        self._spin_comb_density.valueChanged.connect(self._refresh_curvature_comb)

        # Section count changes (a single section has no junction settings);
        # the airfoil is split for the next fit, when it starts
        self._spin_sections.valueChanged.connect(
            lambda _: self._on_optimizer_changed(self._combo_opt.currentIndex())
        )

        # Run controls
        self._btn_fit.clicked.connect(self._fitting_controller.on_fit)
        self._btn_cancel.clicked.connect(self._fitting_controller.on_cancel)

        # Theme: the stylesheet restyles the widgets, the plots restyle here
        self._action_theme.triggered.connect(lambda: ThemeDialog(self).exec())
        theme_changed.changed.connect(self._on_theme_changed)
        self._action_motion.triggered.connect(lambda: MotionDialog(self).exec())

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
        self._btn_load_run.clicked.connect(self._project_manager.on_load_selected_run)
        self._btn_delete_run.clicked.connect(self._project_manager.on_delete_selected_runs)
        # The Delete key deletes the selected runs while the table has focus
        QShortcut(QKeySequence.StandardKey.Delete, table.view,
                  self._project_manager.on_delete_selected_runs,
                  context=Qt.ShortcutContext.WidgetShortcut)
        self._action_restore_runs.triggered.connect(self._project_manager.on_restore_deleted_runs)

        # Initialize optimizer-dependent UI state
        self._on_optimizer_changed(self._combo_opt.currentIndex())

    def _update_undo_actions(self):
        """Enable Edit > Undo and Redo when there is a settings change to undo or redo."""
        history = self._settings_history
        self._action_undo.setEnabled(not self._is_fitting and history.can_undo())
        self._action_redo.setEnabled(not self._is_fitting and history.can_redo())

    def _on_theme_changed(self):
        """Restyle the plots and their items after a theme change."""
        for plot in (self._plot, self._loss_plot, self._comparison_plot):
            style_plot(plot)
        self._plot_manager.restyle()

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
            optimize_le_position=(self._check_optimize_le.isChecked()
                                  and self._spin_sections.value() > 1),
            smoothing_weight=self._spin_smoothing_weight.value(),
            tolerance=tolerance,
            knot_placement=self._combo_knot_placement.currentData(),
            min_knot_gap=self._spin_knot_gap.value(),
            parameterization=self._seg_parameterization.currentData(),
            grow_control_points=self._check_grow.isChecked(),
        )

    def _apply_run_settings(self, run: OptimizationRun) -> None:
        """Set every sidebar setting to that of a stored run (a double-click
        in the explorer, or opening a project)."""
        # One undo step, however long setting the widgets takes
        with self._settings_history.group():
            cfg, opt = run.spline_config, run.optimization_config

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
            self._spin_knot_gap.setValue(opt.min_knot_gap)
            choose(self._seg_parameterization, opt.parameterization)
            self._check_grow.setChecked(opt.grow_control_points)

            # Curve; continuity before the degree and both before the count, as
            # each sets the minimum of the next
            self._spin_sections.setValue(cfg.num_sections)
            choose(self._combo_cont, cfg.continuity)
            self._spin_degree.setValue(cfg.degree)
            choose(self._combo_cont_kind, cfg.geometric_continuity)
            self._spin_ctrl.setValue(cfg.num_control_points)

            # Advanced
            self._edit_junctions.setText(", ".join(cfg.junction_continuity))
            self._spin_smoothing_weight.setValue(opt.smoothing_weight)
            self._check_optimize_le.setChecked(opt.optimize_le_position)

            self._on_optimizer_changed(self._combo_opt.currentIndex())

    def _limit_sections(self, num_points: int) -> None:
        """Allow at most one section fewer than the airfoil's points, the
        most it can be split into."""
        self._spin_sections.setMaximum(max(1, num_points - 1))

    def _junction_orders(self, sections: int) -> tuple[list[int], bool]:
        """The continuity order of each junction of *sections* sections, and
        whether the Per junction entries set them (VarPro LM, when they suit
        the section count) rather than the Continuity setting, which can yield
        to the degree and the control points; the entries cannot."""
        if self._combo_opt.currentData() == OptimizerType.GN_VARPRO:
            specs = self._edit_junctions.text().replace(",", " ").replace(";", " ").split()
            if specs and len(specs) == sections - 1 and all(
                    len(s) == 2 and s[0].upper() in "CG" and s[1] in "0123" for s in specs):
                return [int(s[1]) for s in specs], True
        return [self._combo_cont.currentData().order] * (sections - 1), False

    def _count_is_used(self) -> bool:
        """Whether the fit reads the control point count: not with a tolerance
        goal that grows the count from the fewest."""
        tolerance_mode = (self._combo_opt.currentData() == OptimizerType.GN_VARPRO
                          and self._tolerance_mode_selected())
        return not tolerance_mode or not self._check_grow.isChecked()

    def _keep_curve_valid(self, changed: str = "") -> None:
        """Keep degree, control points, sections and continuity a combination
        that can be fitted, after the user *changed* one of them ("degree",
        "count", "sections", "continuity"; "" for anything else).

        The setting just changed wins and the others follow: a higher degree,
        continuity or section count raises the control points; fewer control
        points lower the degree, then a swept section count, then the
        continuity; a lower degree lowers the continuity. Per junction entries
        do not yield, so they set hard minimums instead.

        A swept degree or section count with a fixed count must fit it at
        every swept value, so the sweep's upper end stops where the count
        does. With the count swept as well, the sweep leaves out the
        combinations that do not fit, and the settings only need one that
        does: the lowest degree and sections with the highest count."""
        if self._resolving_curve or self._combo_cont.currentData() is None:
            return
        self._resolving_curve = True
        try:
            self._resolve_curve(changed)
        finally:
            self._resolving_curve = False

    def _resolve_curve(self, changed: str) -> None:
        degree, count, sections = self._sweep_degree, self._sweep_ctrl, self._sweep_sections
        open_sweep = count.is_sweep_active and (degree.is_sweep_active or sections.is_sweep_active)

        def low_end(box):
            """The degree or sections ends that must fit: all of them, or
            with an open sweep the lowest."""
            ends = box.ends()
            return [min(ends, key=lambda s: s.value())] if open_sweep else ends

        def high_end(box):
            """The count ends they must fit: all, or with an open sweep the highest."""
            ends = box.ends()
            return [max(ends, key=lambda s: s.value())] if open_sweep else ends

        def move(spins, limit: int, up: bool) -> None:
            for spin in spins:
                if spin.value() < limit if up else spin.value() > limit:
                    spin.setValue(limit)

        def orders() -> tuple[list[int], bool]:
            return self._junction_orders(max(s.value() for s in low_end(sections)))

        def need() -> int:
            return minimum_control_points(orders()[0], max(s.value() for s in low_end(degree)))

        def lower_continuity(order: int) -> None:
            self._combo_cont.setCurrentIndex(min(order, self._combo_cont.currentData().order))

        # Degree and continuity: every degree must carry every junction's order
        lowest_degree = min(s.value() for s in degree.ends())
        if max(orders()[0], default=0) > lowest_degree:
            if changed in ("degree", "count") and not orders()[1]:
                lower_continuity(lowest_degree)
            else:
                move(degree.ends(), max(orders()[0]), up=True)

        # Control points: enough for the degree and the junctions
        if self._count_is_used():
            have = min(s.value() for s in high_end(count))
            if need() > have:
                if changed == "count":
                    move(low_end(degree), have - 1, up=False)
                    if need() > have and sections.is_sweep_active:
                        move(sections.ends()[-1:], max(self._spin_sections.value(), 2), up=False)
                    while (need() > have and not orders()[1]
                           and self._combo_cont.currentData().order):
                        lower_continuity(self._combo_cont.currentData().order - 1)
                else:
                    move(high_end(count), need(), up=True)

        # Hard minimums from Per junction entries, and the sweep caps
        fixed_orders, fixed = self._junction_orders(self._spin_sections.value())
        lowest_degree = max(fixed_orders, default=1) if fixed else 1
        degree.set_minimum(lowest_degree)
        count.set_minimum(minimum_control_points(fixed_orders, lowest_degree) if fixed else 2)
        degree_cap, sections_cap = self._spin_degree.maximum(), self._spin_sections.maximum()
        if self._count_is_used() and not count.is_sweep_active:
            have = self._spin_ctrl.value()
            degree_cap = max(have - 1, degree.spinbox.minimum())
            order = self._combo_cont.currentData().order
            fitting = [n for n in range(1, min(sections_cap, 3) + 1)
                       if minimum_control_points([order] * (n - 1), 1) <= have]
            # From three sections on, the need no longer grows: the middle
            # sections all have two junctions
            if fitting[-1] < 3:
                sections_cap = fitting[-1]
            sections_cap = max(sections_cap, self._spin_sections.value())
        degree.set_sweep_maximum(degree_cap)
        sections.set_sweep_maximum(sections_cap)

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
        # Sections times count, or the total when the counts differ (tolerance
        # mode); the tooltip lists every section's count
        counts = [len(cp) for cp in result.section_control_points]
        if len(counts) == 1:
            text = str(counts[0])
        elif len(set(counts)) == 1:
            text = f"{len(counts)} × {counts[0]}"
        else:
            text = str(sum(counts))
        self._lbl_result_ctrl.setText(text)
        self._lbl_result_ctrl.setToolTip(
            ", ".join(str(c) for c in counts) + " per section" if len(counts) > 1 else "")
        self._lbl_result_time.setText(format_duration(result.elapsed_seconds))

    def _show_file_info(self, airfoil: AirfoilData, path: str = ""):
        """Show the loaded airfoil in the sidebar's file card."""
        self._lbl_file_name.setText(airfoil.name or airfoil.filename or "Airfoil")
        self._lbl_path.setText(f"{airfoil.num_points} points, {airfoil.precision} decimals")
        self._lbl_file_name.setToolTip(path)
        self._lbl_path.setToolTip(path)

    def _set_project_save_enabled(self, enabled: bool):
        """Enable/disable project save controls."""
        self._action_save_project.setEnabled(enabled)
        self._action_save_project_as.setEnabled(enabled)

    def _get_active_sweeps(self) -> list[SweepableSpinBox]:
        """The sweepable widgets in sweep mode, in sidebar order."""
        return [box for box in self._sweep_group if box.is_sweep_active]

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
        self._action_open_project.setEnabled(not fitting)
        self._action_new_naca.setEnabled(not fitting)
        self._menu_recent.setEnabled(not fitting)
        self._action_paste_airfoil.setEnabled(not fitting)
        self._action_batch.setEnabled(not fitting)
        # A project opened mid-fit would receive the fit of the airfoil before
        self._action_open_project.setEnabled(not fitting)
        self._project_manager.update_run_buttons()
        if hasattr(self, "_settings_history"):
            self._update_undo_actions()
        # The explorer shows runs while fitting but loads, applies and deletes none
        runs = self._run_table.selected_runs()
        self._btn_load_run.setEnabled(not fitting and len(runs) == 1)
        self._btn_delete_run.setEnabled(not fitting and bool(runs))

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

    def _update_deviation(self):
        """Show, hide or rescale the deviation of the data from the curve."""
        self._plot_manager.deviation_height = (
            self._spin_deviation_height.value() if self._check_deviation.isChecked() else None)
        self._plot_manager.refresh_deviation(self._fit_result)

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
