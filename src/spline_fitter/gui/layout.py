"""
Window Layout
=============

Widget construction for the main window: menu bar, side panel sections,
plots, and the run controls.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from ..optimization.models import (
    ContinuityMode,
    ErrorMetric,
    OptimizerType,
)
from .widgets import CollapsibleSection, SweepableDoubleSpinBox, SweepableSpinBox


class WindowLayoutMixin:
    """
    Builds the main window's widgets.

    Mixed into SplineFitterWindow; methods use the window's widgets and
    state through ``self``.
    """

    # -------------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------------

    def _build_ui(self):
        """Build the main window layout."""
        # Menu bar
        self._build_menu_bar()

        # Main horizontal splitter: [Controls] | [Plots] | [Explorer]
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Left panel container (scroll area + fixed run controls)
        left_container = QWidget(self)
        left_container_layout = QVBoxLayout(left_container)
        left_container_layout.setContentsMargins(0, 0, 0, 0)
        left_container_layout.setSpacing(0)

        # Scrollable area for controls
        left_scroll = QScrollArea(self)
        left_scroll.setWidgetResizable(True)
        left_panel = QWidget()
        left_scroll.setWidget(left_panel)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(4)

        # File info bar (compact, always visible)
        left_layout.addWidget(self._build_file_info())

        # Collapsible sections
        left_layout.addWidget(self._build_curve_section())           # expanded
        left_layout.addWidget(self._build_optimizer_section())       # expanded
        left_layout.addWidget(self._build_regularization_section())  # collapsed
        left_layout.addWidget(self._build_visualization_section())   # collapsed
        left_layout.addStretch()

        # Add scroll area to container
        left_container_layout.addWidget(left_scroll, stretch=1)

        # Fixed run controls at bottom (outside scroll area)
        left_container_layout.addWidget(self._build_run_group())

        # Center panel: Plots
        center_panel = QWidget(self)
        center_layout = QVBoxLayout(center_panel)

        # Main plot (airfoil visualization) — full width at top
        self._plot = pg.PlotWidget(parent=center_panel)
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.getPlotItem().setAspectLocked(True, ratio=1)
        self._plot.setLabel("bottom", "x")
        self._plot.setLabel("left", "y")
        center_layout.addWidget(self._plot, stretch=3)

        # Bottom row: Loss plot (with inline toolbar) and Comparison plot
        bottom_plots = QWidget()
        bottom_plots_layout = QHBoxLayout(bottom_plots)
        bottom_plots_layout.setContentsMargins(0, 0, 0, 0)
        bottom_plots_layout.setSpacing(4)

        # Loss plot with inline axis controls
        loss_container = QWidget()
        loss_layout = QVBoxLayout(loss_container)
        loss_layout.setContentsMargins(0, 0, 0, 0)
        loss_layout.setSpacing(2)
        loss_layout.addLayout(self._build_loss_toolbar())

        self._loss_plot = pg.PlotWidget(parent=loss_container)
        self._loss_plot.showGrid(x=True, y=True, alpha=0.25)
        self._loss_plot.setLabel("bottom", "Evaluations")
        self._loss_plot.setLabel("left", "Error")
        self._loss_plot.getPlotItem().setLogMode(x=True, y=True)
        loss_layout.addWidget(self._loss_plot)
        bottom_plots_layout.addWidget(loss_container, stretch=1)

        # Comparison plot (error vs control points)
        self._comparison_plot = pg.PlotWidget(parent=bottom_plots)
        self._comparison_plot.showGrid(x=True, y=True, alpha=0.25)
        self._comparison_plot.setLabel("bottom", "Control Points")
        self._comparison_plot.setLabel("left", "Error")
        self._comparison_plot.getPlotItem().setLogMode(x=False, y=True)
        bottom_plots_layout.addWidget(self._comparison_plot, stretch=1)

        center_layout.addWidget(bottom_plots, stretch=2)

        # Right panel: Project Explorer
        explorer_panel = self._build_project_explorer()

        # Configure splitter
        self._main_splitter.addWidget(left_container)
        self._main_splitter.addWidget(center_panel)
        self._main_splitter.addWidget(explorer_panel)
        left_container.setMinimumWidth(300)
        explorer_panel.setMinimumWidth(0)
        self._main_splitter.setCollapsible(0, False)
        self._main_splitter.setCollapsible(1, False)
        self._main_splitter.setCollapsible(2, True)
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)
        self._main_splitter.setSizes([360, 700, 280])

        self.setCentralWidget(self._main_splitter)
        self.statusBar().showMessage("Ready")

    # -- Menu Bar ----------------------------------------------------------

    def _build_menu_bar(self):
        """Build the application menu bar (File / View / Help)."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        self._action_open = file_menu.addAction("&Open Airfoil...")
        self._action_open.setShortcut("Ctrl+O")

        file_menu.addSeparator()
        self._action_save_dat = file_menu.addAction("Save Fitted (.dat)...")
        self._action_save_dat.setEnabled(False)
        self._action_save_spline = file_menu.addAction("Save Spline (.spline.dat)...")
        self._action_save_spline.setEnabled(False)
        self._action_export_cad = file_menu.addAction("Export CAD (STEP/IGES/DXF)...")
        self._action_export_cad.setShortcut("Ctrl+E")
        self._action_export_cad.setEnabled(False)

        file_menu.addSeparator()
        self._action_open_project = file_menu.addAction("Open &Project...")
        self._action_save_project = file_menu.addAction("&Save Project")
        self._action_save_project.setShortcut("Ctrl+S")
        self._action_save_project.setEnabled(False)
        self._action_save_project_as = file_menu.addAction("Save Project &As...")
        self._action_save_project_as.setEnabled(False)

        file_menu.addSeparator()
        self._action_batch = file_menu.addAction("&Batch Convert...")

        file_menu.addSeparator()
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)

        # View menu
        view_menu = menubar.addMenu("&View")
        self._action_fit_report = view_menu.addAction("&Fit Report...")
        self._action_fit_report.setShortcut("Ctrl+R")
        self._action_fit_report.setEnabled(False)
        self._action_view_report = view_menu.addAction("&Benchmark Report...")

        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("&About").triggered.connect(
            lambda: self.statusBar().showMessage("Airfoil Spline Fitter v1.0")
        )

    # -- File Info Bar -----------------------------------------------------

    def _build_file_info(self) -> QWidget:
        """Build the compact file-info area at the top of the left panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 8)
        layout.setSpacing(4)

        self._lbl_path = QLabel("No file loaded \u2014 use File \u2192 Open Airfoil")
        self._lbl_path.setWordWrap(True)
        self._lbl_path.setStyleSheet("font-size: 11px; color: gray;")
        layout.addWidget(self._lbl_path)

        save_row = QHBoxLayout()
        save_row.setSpacing(4)
        self._btn_save_dat = QPushButton("Save Fitted")
        self._btn_save_dat.setEnabled(False)
        self._btn_save_dat.setToolTip("Save fitted airfoil coordinates (.dat)")
        save_row.addWidget(self._btn_save_dat)
        self._btn_save_spline = QPushButton("Save Spline")
        self._btn_save_spline.setEnabled(False)
        self._btn_save_spline.setToolTip("Save spline control points and knots (.spline.dat)")
        save_row.addWidget(self._btn_save_spline)
        self._btn_export_cad = QPushButton("Export CAD")
        self._btn_export_cad.setEnabled(False)
        self._btn_export_cad.setToolTip("Export the exact B-spline curves as STEP, IGES or DXF")
        save_row.addWidget(self._btn_export_cad)
        save_row.addStretch()
        layout.addLayout(save_row)

        return widget

    # -- Loss Toolbar ------------------------------------------------------

    def _build_loss_toolbar(self) -> QHBoxLayout:
        """Build the inline toolbar above the loss/convergence plot."""
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        toolbar.setContentsMargins(0, 0, 0, 0)

        toolbar.addWidget(QLabel("X:"))
        self._combo_loss_x = QComboBox()
        self._combo_loss_x.addItem("Iterations", userData="iters")
        self._combo_loss_x.addItem("Time (s)", userData="time")
        self._combo_loss_x.setCurrentIndex(0)
        self._combo_loss_x.setFixedWidth(90)
        toolbar.addWidget(self._combo_loss_x)

        toolbar.addWidget(QLabel("Scale:"))
        self._combo_log_scale = QComboBox()
        self._combo_log_scale.addItem("Y only", userData="y")
        self._combo_log_scale.addItem("Both", userData="both")
        self._combo_log_scale.addItem("None", userData="none")
        self._combo_log_scale.addItem("X only", userData="x")
        self._combo_log_scale.setCurrentIndex(0)
        self._combo_log_scale.setFixedWidth(70)
        toolbar.addWidget(self._combo_log_scale)

        toolbar.addStretch()
        return toolbar

    # -- Project Explorer --------------------------------------------------

    def _build_project_explorer(self) -> QWidget:
        """Build the project explorer panel with tree view."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Header with toggle button
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Project Explorer</b>"))
        header.addStretch()
        self._btn_toggle_explorer = QPushButton("\u25C0")
        self._btn_toggle_explorer.setFixedWidth(24)
        self._btn_toggle_explorer.setToolTip("Collapse explorer panel")
        header.addWidget(self._btn_toggle_explorer)
        layout.addLayout(header)

        # Project info label (was in file tab, now lives here)
        self._lbl_project = QLabel("No project")
        self._lbl_project.setWordWrap(True)
        self._lbl_project.setStyleSheet("color: gray; font-size: 10px; padding: 2px 4px;")
        layout.addWidget(self._lbl_project)

        # Tree view for runs
        self._project_tree = QTreeWidget()
        self._project_tree.setHeaderLabels(["Optimization Runs"])
        self._project_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self._project_tree.setRootIsDecorated(True)
        self._project_tree.setAnimated(True)
        layout.addWidget(self._project_tree, stretch=1)

        # Run details label
        self._lbl_run_details = QLabel("Select a run to view details")
        self._lbl_run_details.setWordWrap(True)
        self._lbl_run_details.setStyleSheet("color: gray; font-size: 10px; padding: 4px;")
        layout.addWidget(self._lbl_run_details)

        # Action buttons
        btn_row = QHBoxLayout()
        self._btn_delete_run = QPushButton("Delete")
        self._btn_delete_run.setEnabled(False)
        self._btn_delete_run.setToolTip("Delete selected run(s) from project")
        btn_row.addWidget(self._btn_delete_run)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return panel

    # -- Curve Properties Section ------------------------------------------

    def _build_curve_section(self) -> CollapsibleSection:
        """Build the Curve Properties collapsible section."""
        section = CollapsibleSection("Curve Properties", expanded=True)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Degree (sweepable)
        self._spin_degree = QSpinBox()
        self._spin_degree.setRange(1, 10)
        self._spin_degree.setValue(5)
        self._spin_degree.setToolTip(
            "Polynomial degree of B-spline basis functions.\n"
            "Typical: 3 (cubic), 4, or 5 (quintic)."
        )
        self._sweep_degree = SweepableSpinBox(self._spin_degree, group=self._sweep_group)
        form.addRow("Degree:", self._sweep_degree)

        # Control points (sweepable)
        self._spin_ctrl = QSpinBox()
        self._spin_ctrl.setRange(2, 200)
        self._spin_ctrl.setValue(25)
        self._spin_ctrl.setToolTip(
            "Control points per section.\n"
            "More = more flexibility but more variables."
        )
        self._sweep_ctrl = SweepableSpinBox(self._spin_ctrl, group=self._sweep_group)
        form.addRow("Control Points:", self._sweep_ctrl)

        # Sections
        self._spin_sections = QSpinBox()
        self._spin_sections.setRange(1, 10)
        self._spin_sections.setValue(2)
        self._spin_sections.setToolTip(
            "Number of spline segments.\n"
            "2 = upper/lower split at leading edge (default)"
        )
        form.addRow("Sections:", self._spin_sections)

        # Smoothness
        self._combo_cont = QComboBox()
        self._combo_cont.addItem("0 - Position", userData=ContinuityMode.G0)
        self._combo_cont.addItem("1 - Tangent", userData=ContinuityMode.G1)
        self._combo_cont.addItem("2 - Curvature", userData=ContinuityMode.G2)
        self._combo_cont.addItem("3 - Curvature rate", userData=ContinuityMode.G3)
        self._combo_cont.setCurrentIndex(2)
        self._combo_cont.setToolTip(
            "Smoothness at junctions:\n"
            "0 = Position only, 1 = Tangent,\n"
            "2 = Curvature, 3 = Curvature rate"
        )
        form.addRow("Smoothness:", self._combo_cont)

        self._combo_cont_kind = QComboBox()
        self._combo_cont_kind.addItem("Geometric (G)", userData=True)
        self._combo_cont_kind.addItem("Parametric (C)", userData=False)
        self._combo_cont_kind.setToolTip(
            "Geometric: tangent direction, curvature and curvature rate match\n"
            "across the junction (what CAD calls G1/G2/G3).\n"
            "Parametric: the derivative vectors themselves match (C1/C2/C3),\n"
            "which is stricter and needs more control points.\n"
            "Only VarPro LM supports geometric continuity; other optimizers\n"
            "always use parametric."
        )
        form.addRow("Continuity Type:", self._combo_cont_kind)

        self._edit_junctions = QLineEdit()
        self._edit_junctions.setPlaceholderText("e.g. G2, C1  (blank = same everywhere)")
        self._edit_junctions.setToolTip(
            "Continuity of each junction in order, overriding the two settings\n"
            "above. One entry per junction (sections - 1), each G0-G3 or C0-C3.\n"
            "VarPro LM only."
        )
        form.addRow("Per Junction:", self._edit_junctions)

        section.add_layout(form)

        # Advanced subsection (collapsed)
        adv = CollapsibleSection("Advanced", expanded=False)
        adv_form = QFormLayout()
        adv_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._check_optimize_le = QCheckBox("Optimize Junction Position")
        self._check_optimize_le.setChecked(False)
        self._check_optimize_le.setToolTip(
            "Allow the leading edge junction to move during optimization."
        )
        adv_form.addRow("", self._check_optimize_le)

        self._spin_bbox = QDoubleSpinBox()
        self._spin_bbox.setRange(1.0, 100.0)
        self._spin_bbox.setDecimals(2)
        self._spin_bbox.setSingleStep(0.1)
        self._spin_bbox.setValue(3.0)
        self._spin_bbox.setToolTip("Multiplier for optimization search bounds.")
        adv_form.addRow("Search Space Scale:", self._spin_bbox)

        adv.add_layout(adv_form)
        section.add_widget(adv)

        return section

    # -- Optimizer Section -------------------------------------------------

    def _build_optimizer_section(self) -> CollapsibleSection:
        """Build the Optimizer collapsible section (merged with param-opt)."""
        section = CollapsibleSection("Optimizer", expanded=True)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Algorithm
        self._combo_opt = QComboBox()
        for idx, opt_type in enumerate(OptimizerType):
            self._combo_opt.addItem(opt_type.display_name, userData=opt_type)
            self._combo_opt.setItemData(idx, opt_type.tooltip, Qt.ItemDataRole.ToolTipRole)
        self._combo_opt.setCurrentIndex(0)
        self._combo_opt.currentIndexChanged.connect(self._on_optimizer_changed)
        form.addRow("Algorithm:", self._combo_opt)

        # Error metric
        self._combo_err = QComboBox()
        for metric in ErrorMetric:
            self._combo_err.addItem(metric.display_name, userData=metric)
        self._combo_err.setCurrentIndex(0)
        form.addRow("Error Metric:", self._combo_err)

        section.add_layout(form)

        # VarPro tolerance mode
        self._tolerance_container = QWidget()
        tol_form = QFormLayout(self._tolerance_container)
        tol_form.setContentsMargins(0, 0, 0, 0)
        tol_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._check_tolerance_mode = QCheckBox("Tolerance mode")
        self._check_tolerance_mode.setChecked(True)
        self._check_tolerance_mode.setToolTip(
            "Find the fewest control points (with non-uniform knots) that keep\n"
            "every data point within the tolerance. The control point count\n"
            "setting is ignored."
        )
        tol_form.addRow("", self._check_tolerance_mode)

        tol_row = QHBoxLayout()
        tol_row.setSpacing(4)
        self._spin_tolerance = QDoubleSpinBox()
        self._spin_tolerance.setRange(1e-9, 1.0)
        self._spin_tolerance.setDecimals(9)
        self._spin_tolerance.setSingleStep(1e-6)
        self._spin_tolerance.setValue(1e-5)
        self._spin_tolerance.setToolTip("Maximum orthogonal distance of any data point to the curve.")
        tol_row.addWidget(self._spin_tolerance, stretch=1)
        self._check_auto_tolerance = QCheckBox("Auto")
        self._check_auto_tolerance.setChecked(True)
        self._check_auto_tolerance.setToolTip(
            "Derive the tolerance from the decimal precision of the file and\n"
            "the estimated noise of the points."
        )
        tol_row.addWidget(self._check_auto_tolerance)
        tol_form.addRow("Tolerance:", tol_row)

        dense_row = QHBoxLayout()
        dense_row.setSpacing(4)
        self._check_subsample = QCheckBox("Subsample to")
        self._check_subsample.setChecked(False)
        self._check_subsample.setToolTip(
            "For dense data: fit an evenly spaced subsample of each section, then\n"
            "refine the fit on all points. Much faster for thousands of points.\n"
            "Off: every point is used throughout."
        )
        dense_row.addWidget(self._check_subsample)
        self._spin_subsample = QSpinBox()
        self._spin_subsample.setRange(20, 100000)
        self._spin_subsample.setValue(200)
        self._spin_subsample.setSuffix(" pts/section")
        self._spin_subsample.setEnabled(False)
        self._spin_subsample.setToolTip("Sections with more points than this are subsampled.")
        dense_row.addWidget(self._spin_subsample, stretch=1)
        tol_form.addRow("Dense data:", dense_row)

        self._tolerance_container.setVisible(False)
        section.add_widget(self._tolerance_container)

        # Algorithm Settings (collapsed subsection)
        self._opt_advanced = CollapsibleSection("Algorithm Settings", expanded=False)
        adv_form = QFormLayout()
        adv_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Max iterations
        self._spin_maxiter = QSpinBox()
        self._spin_maxiter.setRange(10, 1_000_000_000)
        self._spin_maxiter.setValue(10000)
        self._spin_maxiter.setToolTip("Maximum optimizer iterations.")
        self._lbl_maxiter = QLabel("Iteration Limit:")
        adv_form.addRow(self._lbl_maxiter, self._spin_maxiter)

        # Random seed
        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(-1, 2**31 - 1)
        self._spin_seed.setValue(-1)
        self._spin_seed.setToolTip("Random seed (-1 = system random, \u22650 = deterministic).")
        adv_form.addRow("Random Seed:", self._spin_seed)

        # Smoothing factor (algebraic methods)
        self._lbl_smoothing = QLabel("LSQ Smoothing:")
        self._spin_smoothing = QDoubleSpinBox()
        self._spin_smoothing.setRange(0.0, 1.0)
        self._spin_smoothing.setDecimals(6)
        self._spin_smoothing.setSingleStep(0.0001)
        self._spin_smoothing.setValue(0.0)
        self._spin_smoothing.setToolTip("Smoothing factor for algebraic fitting (0 = exact).")
        self._lbl_smoothing.setVisible(False)
        self._spin_smoothing.setVisible(False)
        adv_form.addRow(self._lbl_smoothing, self._spin_smoothing)

        # Bending energy weight (GN-VarPro, sweepable)
        self._lbl_smoothing_weight = QLabel("Smoothing Weight:")
        self._spin_smoothing_weight = QDoubleSpinBox()
        self._spin_smoothing_weight.setRange(0.0, 1e12)
        self._spin_smoothing_weight.setDecimals(12)
        self._spin_smoothing_weight.setSingleStep(0.001)
        self._spin_smoothing_weight.setValue(0.001)
        self._spin_smoothing_weight.setToolTip("Bending energy weight for GN-VarPro (0=off, 1e-3=default).")
        self._sweep_smoothing_weight = SweepableDoubleSpinBox(
            self._spin_smoothing_weight, group=self._sweep_group, default_steps=10,
        )
        self._lbl_smoothing_weight.setVisible(False)
        self._sweep_smoothing_weight.setVisible(False)
        adv_form.addRow(self._lbl_smoothing_weight, self._sweep_smoothing_weight)

        # Orthogonal LSQ local bounds
        self._check_orth_lsq_local_bounds = QCheckBox("Use local t bounds")
        self._check_orth_lsq_local_bounds.setChecked(False)
        self._check_orth_lsq_local_bounds.setVisible(False)
        self._check_orth_lsq_local_bounds.setToolTip("Restrict foot-point search to local bounds.")
        adv_form.addRow("", self._check_orth_lsq_local_bounds)

        # Param optimizer outer (for PARAM_OPT)
        self._param_opt_container = QWidget()
        param_lay = QFormLayout(self._param_opt_container)
        param_lay.setContentsMargins(0, 0, 0, 0)

        from ..optimization.optimizers import get_optimizer
        self._combo_param_opt_outer = QComboBox()
        for idx, opt_type in enumerate(OptimizerType):
            if opt_type.is_param_optimizer or opt_type.is_algebraic or opt_type.is_iterative_algebraic:
                continue
            try:
                opt = get_optimizer(opt_type)
                if not opt.supports_bounds:
                    continue
            except Exception:
                continue
            self._combo_param_opt_outer.addItem(opt_type.display_name, userData=opt_type)
            self._combo_param_opt_outer.setItemData(
                self._combo_param_opt_outer.count() - 1,
                opt_type.tooltip, Qt.ItemDataRole.ToolTipRole,
            )
        for i in range(self._combo_param_opt_outer.count()):
            if self._combo_param_opt_outer.itemData(i) == OptimizerType.L_BFGS_B:
                self._combo_param_opt_outer.setCurrentIndex(i)
                break
        self._combo_param_opt_outer.setToolTip("Outer-loop optimizer for parameter optimisation.")
        param_lay.addRow("Outer Optimizer:", self._combo_param_opt_outer)

        self._param_opt_container.setVisible(False)
        adv_form.addRow(self._param_opt_container)

        self._opt_advanced.add_layout(adv_form)
        section.add_widget(self._opt_advanced)

        return section

    # -- Regularization Section --------------------------------------------

    def _build_regularization_section(self) -> CollapsibleSection:
        """Build the Regularization collapsible section (collapsed)."""
        section = CollapsibleSection("Regularization", expanded=False)
        layout = QVBoxLayout()
        layout.setSpacing(6)

        self._check_fairness = QCheckBox("Enable Regularization")
        self._check_fairness.setChecked(False)
        self._check_fairness.setToolTip("Add fairness penalties to the objective function.")
        layout.addWidget(self._check_fairness)

        self._fairness_container = QWidget()
        wf = QFormLayout(self._fairness_container)
        wf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        def _ws() -> QDoubleSpinBox:
            s = QDoubleSpinBox(); s.setRange(0.0, 1.0); s.setDecimals(4)
            s.setSingleStep(0.001); s.setValue(0.0); return s

        self._spin_w_lenvar = _ws()
        self._spin_w_lenvar.setToolTip("Penalize uneven spacing between control points")
        wf.addRow("Uniform Edge Length:", self._spin_w_lenvar)

        self._spin_w_bend = _ws()
        self._spin_w_bend.setToolTip("Penalize sharp bends in the control polygon")
        wf.addRow("Polygon Smoothness:", self._spin_w_bend)

        self._spin_w_short = _ws()
        self._spin_w_short.setToolTip("Penalize control points that are too close together")
        wf.addRow("Minimum Spacing:", self._spin_w_short)

        self._spin_w_tangent = _ws()
        self._spin_w_tangent.setToolTip("Penalize unusual edge lengths at curve endpoints")
        wf.addRow("End Handle Length:", self._spin_w_tangent)

        self._spin_w_curv = _ws()
        self._spin_w_curv.setToolTip("Minimize bending energy: \u222B\u03BA\u00B2 ds")
        wf.addRow("Bending Energy:", self._spin_w_curv)

        self._spin_w_grad = _ws()
        self._spin_w_grad.setToolTip("Minimize wiggle energy: \u222B(d\u03BA/ds)\u00B2 ds")
        wf.addRow("Wiggle Energy:", self._spin_w_grad)

        layout.addWidget(self._fairness_container)
        self._fairness_container.setEnabled(False)

        section.add_layout(layout)
        return section

    # -- Visualization Section ---------------------------------------------

    def _build_visualization_section(self) -> CollapsibleSection:
        """Build the Visualization collapsible section (collapsed)."""
        section = CollapsibleSection("Visualization", expanded=False)
        layout = QVBoxLayout()
        layout.setSpacing(6)

        # Preview frequency slider
        preview_row = QHBoxLayout()
        preview_row.setSpacing(6)
        preview_row.addWidget(QLabel("Preview:"))
        self._slider_live = QSlider(Qt.Orientation.Horizontal)
        self._slider_live.setRange(0, 2)
        self._slider_live.setValue(1)
        self._slider_live.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_live.setTickInterval(1)
        self._slider_live.setFixedWidth(80)
        preview_row.addWidget(self._slider_live)
        self._lbl_live_mode = QLabel("Moderate")
        self._lbl_live_mode.setFixedWidth(65)
        preview_row.addWidget(self._lbl_live_mode)
        preview_row.addStretch()
        layout.addLayout(preview_row)

        # Curvature comb toggle
        self._check_comb = QCheckBox("Show Curvature Comb")
        self._check_comb.setChecked(False)
        self._check_comb.setToolTip("Display curvature as perpendicular lines from curve.")
        layout.addWidget(self._check_comb)

        self._check_knots = QCheckBox("Show Knots")
        self._check_knots.setChecked(False)
        self._check_knots.setToolTip("Mark the curve points at the knots of every section.")
        layout.addWidget(self._check_knots)

        # Comb settings (nested, collapsed)
        comb_section = CollapsibleSection("Comb Settings", expanded=False)
        comb_form = QFormLayout()
        comb_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._combo_comb_qty = QComboBox()
        self._combo_comb_qty.addItem("Curvature (\u03BA)", userData=0)
        self._combo_comb_qty.addItem("Gradient (d\u03BA/ds)", userData=1)
        self._combo_comb_qty.addItem("Acceleration (d\u00B2\u03BA/ds\u00B2)", userData=2)
        self._combo_comb_qty.addItem("Jerk (d\u00B3\u03BA/ds\u00B3)", userData=3)
        self._combo_comb_qty.setCurrentIndex(0)
        comb_form.addRow("Quantity:", self._combo_comb_qty)

        self._spin_comb_scale = QDoubleSpinBox()
        self._spin_comb_scale.setRange(0.0, 1000.0)
        self._spin_comb_scale.setDecimals(4)
        self._spin_comb_scale.setSingleStep(0.01)
        self._spin_comb_scale.setValue(0.05)
        comb_form.addRow("Height:", self._spin_comb_scale)

        self._spin_comb_density = QSpinBox()
        self._spin_comb_density.setRange(5, 2000)
        self._spin_comb_density.setValue(120)
        comb_form.addRow("Density:", self._spin_comb_density)

        comb_section.add_layout(comb_form)
        layout.addWidget(comb_section)

        section.add_layout(layout)
        return section

    # -- Run Group ---------------------------------------------------------

    def _build_run_group(self) -> QGroupBox:
        """Build the run controls group."""
        group = QGroupBox("Run")
        layout = QHBoxLayout(group)
        layout.setSpacing(8)
        self._btn_fit = QPushButton("Fit Spline")
        layout.addWidget(self._btn_fit)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        layout.addWidget(self._btn_cancel)
        layout.addStretch()
        return group
