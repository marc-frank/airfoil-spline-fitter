"""
Window Layout
=============

Widget construction for the main window: menu bar, sidebar sections,
plots with their toolbars, and the run bar.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ..optimization.models import (
    ContinuityMode,
    ErrorMetric,
    OptimizerType,
)
from .widgets import (
    SIDEBAR_STYLE,
    CollapsibleSection,
    SegmentedControl,
    SidebarSection,
    SweepableDoubleSpinBox,
    SweepableSpinBox,
)

# Width of the label column in every sidebar form, so fields line up
# across sections.
LABEL_WIDTH = 92


def _form() -> QFormLayout:
    """A sidebar form layout."""
    form = QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setHorizontalSpacing(8)
    form.setVerticalSpacing(6)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return form


def _add_row(form: QFormLayout, text: str, field) -> QLabel:
    """Add a row with a fixed-width label and return the label."""
    label = QLabel(text)
    label.setFixedWidth(LABEL_WIDTH)
    form.addRow(label, field)
    return label


def _hbox(*widgets, stretch_first: bool = True) -> QWidget:
    """Put widgets side by side in one form field."""
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for i, w in enumerate(widgets):
        lay.addWidget(w, stretch=1 if (i == 0 and stretch_first) else 0)
    return row


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

        # Main horizontal splitter: [Sidebar] | [Plots] | [Explorer]
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Sidebar: scrolling sections above a fixed run bar, in workflow
        # order (data -> method -> curve -> advanced -> fit and results)
        sidebar = QWidget(self)
        sidebar.setObjectName("Sidebar")
        sidebar.setStyleSheet(SIDEBAR_STYLE)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sections = QWidget()
        sections.setObjectName("Sidebar")
        sidebar_scroll.setWidget(sections)
        sections_layout = QVBoxLayout(sections)
        sections_layout.setContentsMargins(0, 0, 0, 0)
        sections_layout.setSpacing(0)

        sections_layout.addWidget(self._build_file_card())
        sections_layout.addWidget(self._build_method_section())
        sections_layout.addWidget(self._build_curve_section())
        sections_layout.addWidget(self._build_advanced_section())
        sections_layout.addStretch()

        # Long item texts and wide spin box ranges must not widen the
        # sidebar past its width: fields take the width the form gives them
        for combo in sections.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(6)
        for field in sections.findChildren((QComboBox, QAbstractSpinBox)):
            field.setSizePolicy(QSizePolicy.Policy.Expanding, field.sizePolicy().verticalPolicy())
            field.setMinimumWidth(70)

        sidebar_layout.addWidget(sidebar_scroll, stretch=1)
        sidebar_layout.addWidget(self._build_run_bar())

        # Center panel: Plots
        center_panel = QWidget(self)
        center_layout = QVBoxLayout(center_panel)

        # Main plot (airfoil visualization) with its display toolbar
        center_layout.addLayout(self._build_plot_toolbar())
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
        self._main_splitter.addWidget(sidebar)
        self._main_splitter.addWidget(center_panel)
        self._main_splitter.addWidget(explorer_panel)
        sidebar.setMinimumWidth(300)
        explorer_panel.setMinimumWidth(0)
        self._main_splitter.setCollapsible(0, False)
        self._main_splitter.setCollapsible(1, False)
        self._main_splitter.setCollapsible(2, True)
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)
        self._main_splitter.setSizes([330, 730, 280])

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

    # -- File Card ---------------------------------------------------------

    def _build_file_card(self) -> QWidget:
        """Build the loaded-file summary at the top of the sidebar."""
        card = SidebarSection("Data")
        row = QHBoxLayout()
        row.setSpacing(8)

        text = QVBoxLayout()
        text.setSpacing(0)
        self._lbl_file_name = QLabel("No airfoil loaded")
        self._lbl_file_name.setObjectName("FileName")
        text.addWidget(self._lbl_file_name)
        self._lbl_path = QLabel("Open a .dat file or a project")
        self._lbl_path.setObjectName("Muted")
        text.addWidget(self._lbl_path)
        row.addLayout(text, stretch=1)

        self._btn_open = QPushButton("Open\u2026")
        self._btn_open.setToolTip("Open an airfoil coordinate file (Ctrl+O)")
        row.addWidget(self._btn_open, alignment=Qt.AlignmentFlag.AlignVCenter)

        card.add_layout(row)
        return card

    # -- Plot Toolbar ------------------------------------------------------

    def _build_plot_toolbar(self) -> QHBoxLayout:
        """Build the display toolbar above the main plot."""
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        toolbar.setContentsMargins(0, 0, 0, 0)

        self._check_knots = QToolButton()
        self._check_knots.setText("Knots")
        self._check_knots.setCheckable(True)
        self._check_knots.setToolTip("Mark the curve points at the knots of every section.")
        toolbar.addWidget(self._check_knots)

        # Curvature comb: toggle, with its settings in the drop-down
        self._check_comb = QToolButton()
        self._check_comb.setText("Curvature comb")
        self._check_comb.setCheckable(True)
        self._check_comb.setToolTip(
            "Display curvature as perpendicular lines from the curve.\n"
            "Use the arrow for quantity, height and density."
        )
        self._check_comb.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

        comb_settings = QWidget()
        comb_form = _form()
        comb_settings.setLayout(comb_form)
        comb_form.setContentsMargins(10, 8, 10, 8)

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

        comb_menu = QMenu(self._check_comb)
        comb_action = QWidgetAction(comb_menu)
        comb_action.setDefaultWidget(comb_settings)
        comb_menu.addAction(comb_action)
        self._check_comb.setMenu(comb_menu)
        toolbar.addWidget(self._check_comb)

        toolbar.addStretch()

        toolbar.addWidget(QLabel("Live preview:"))
        self._combo_live = QComboBox()
        self._combo_live.addItems(["Minimal", "Moderate", "Detailed"])
        self._combo_live.setCurrentIndex(1)
        self._combo_live.setToolTip(
            "How often the plots update while fitting.\n"
            "Minimal: only at the end. Moderate: once a second.\n"
            "Detailed: every iteration (slower)."
        )
        toolbar.addWidget(self._combo_live)
        return toolbar

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

    # -- Method Section ----------------------------------------------------

    def _build_method_section(self) -> SidebarSection:
        """Build the Method section: optimizer, goal, and error metric."""
        section = SidebarSection("Method")
        form = _form()

        # Algorithm (decides which other controls apply, so it comes first)
        self._combo_opt = QComboBox()
        for idx, opt_type in enumerate(OptimizerType):
            self._combo_opt.addItem(opt_type.display_name, userData=opt_type)
            self._combo_opt.setItemData(idx, opt_type.tooltip, Qt.ItemDataRole.ToolTipRole)
        self._combo_opt.setCurrentIndex(self._combo_opt.findData(OptimizerType.GN_VARPRO))
        self._combo_opt.currentIndexChanged.connect(self._on_optimizer_changed)
        _add_row(form, "Optimizer", self._combo_opt)

        # Goal (VarPro): meet a tolerance, or fit a fixed control point count
        self._seg_goal = SegmentedControl()
        self._seg_goal.addItem("Tolerance", userData=True)
        self._seg_goal.addItem("Fixed count", userData=False)
        self._seg_goal.setItemToolTip(
            0,
            "Find the fewest control points that keep\n"
            "every data point within the tolerance.",
        )
        self._seg_goal.setItemToolTip(1, "Fit the given number of control points per section.")
        self._seg_goal.setCurrentIndex(self._seg_goal.findData(False))
        self._lbl_goal = _add_row(form, "Goal", self._seg_goal)

        self._spin_tolerance = QDoubleSpinBox()
        self._spin_tolerance.setRange(1e-9, 1.0)
        self._spin_tolerance.setDecimals(9)
        self._spin_tolerance.setSingleStep(1e-6)
        self._spin_tolerance.setValue(1e-5)
        self._spin_tolerance.setToolTip("Maximum orthogonal distance of any data point to the curve.")
        self._check_auto_tolerance = QCheckBox("Auto")
        self._check_auto_tolerance.setChecked(True)
        self._check_auto_tolerance.setToolTip(
            "Derive the tolerance from the decimal precision of the file and\n"
            "the estimated noise of the points."
        )
        self._tolerance_row = _hbox(self._spin_tolerance, self._check_auto_tolerance)
        self._lbl_tolerance = _add_row(form, "Max deviation", self._tolerance_row)

        # Knot placement and growth (VarPro, both goals)
        self._combo_knot_placement = SegmentedControl()
        self._combo_knot_placement.addItem("Free", userData="free")
        self._combo_knot_placement.addItem("Uniform", userData="uniform")
        self._combo_knot_placement.setItemToolTip(
            0, "Knots move to where they reduce the error most.")
        self._combo_knot_placement.setItemToolTip(
            1, "Knots stay evenly spaced in the curve parameter.")
        self._combo_knot_placement.setCurrentIndex(self._combo_knot_placement.findData("uniform"))
        self._lbl_knot_placement = _add_row(form, "Knots", self._combo_knot_placement)

        self._check_grow = QCheckBox("Add where the fit is worst")
        self._check_grow.setChecked(False)
        self._check_grow.setToolTip(
            "On: start from the fewest control points and add them where the\n"
            "fit is worst, refitting after each step.\n"
            "Off: start from the Control pts setting. In tolerance mode the\n"
            "count can still grow or shrink from there to meet the tolerance."
        )
        self._lbl_grow = _add_row(form, "Grow points", self._check_grow)

        # Control points (sweepable); in tolerance mode only shown as the
        # starting count when not growing from the fewest
        self._spin_ctrl = QSpinBox()
        self._spin_ctrl.setRange(2, 200)
        self._spin_ctrl.setValue(25)
        self._spin_ctrl.setToolTip(
            "Control points per section.\n"
            "More = more flexibility but more variables."
        )
        self._sweep_ctrl = SweepableSpinBox(self._spin_ctrl, group=self._sweep_group)
        self._lbl_ctrl = _add_row(form, "Control pts", self._sweep_ctrl)

        # Error metric
        self._combo_err = QComboBox()
        for metric in ErrorMetric:
            self._combo_err.addItem(metric.display_name, userData=metric)
        self._combo_err.setCurrentIndex(0)
        self._lbl_err = _add_row(form, "Error metric", self._combo_err)

        section.add_layout(form)
        return section

    # -- Curve Section -----------------------------------------------------

    def _build_curve_section(self) -> SidebarSection:
        """Build the Curve section: degree, sections, and continuity."""
        section = SidebarSection("Curve")
        form = _form()

        # Degree (sweepable)
        self._spin_degree = QSpinBox()
        self._spin_degree.setRange(1, 10)
        self._spin_degree.setValue(5)
        self._spin_degree.setToolTip(
            "Polynomial degree of B-spline basis functions.\n"
            "Typical: 3 (cubic), 4, or 5 (quintic)."
        )
        self._sweep_degree = SweepableSpinBox(self._spin_degree, group=self._sweep_group)
        _add_row(form, "Degree", self._sweep_degree)

        # Sections
        self._spin_sections = QSpinBox()
        self._spin_sections.setRange(1, 10)
        self._spin_sections.setValue(2)
        self._spin_sections.setToolTip(
            "Number of spline segments.\n"
            "2 = upper/lower split at leading edge (default)"
        )
        _add_row(form, "Sections", self._spin_sections)

        # Continuity at the junctions; labels read G or C depending on the type
        self._combo_cont = SegmentedControl()
        cont_names = ["Position", "Tangent", "Curvature", "Curvature rate"]
        for i, mode in enumerate(
            (ContinuityMode.G0, ContinuityMode.G1, ContinuityMode.G2, ContinuityMode.G3)
        ):
            self._combo_cont.addItem(f"C{i}", userData=mode)
            self._combo_cont.setItemToolTip(i, f"{cont_names[i]} matches across junctions")
        self._combo_cont.setCurrentIndex(3)
        _add_row(form, "Continuity", self._combo_cont)

        self._combo_cont_kind = SegmentedControl()
        self._combo_cont_kind.addItem("Geometric", userData=True)
        self._combo_cont_kind.addItem("Parametric", userData=False)
        self._combo_cont_kind.setToolTip(
            "Geometric: tangent direction, curvature and curvature rate match\n"
            "across the junction (what CAD calls G1/G2/G3).\n"
            "Parametric: the derivative vectors themselves match (C1/C2/C3),\n"
            "which is stricter and needs more control points.\n"
            "Only VarPro LM supports geometric continuity; other optimizers\n"
            "always use parametric."
        )
        self._lbl_cont_kind = _add_row(form, "Type", self._combo_cont_kind)

        section.add_layout(form)
        return section

    # -- Advanced Section --------------------------------------------------

    def _build_advanced_section(self) -> CollapsibleSection:
        """Build the Advanced section: everything rarely changed, in one place."""
        section = CollapsibleSection("Advanced", expanded=False)
        self._advanced_section = section
        form = _form()

        # VarPro
        self._edit_junctions = QLineEdit()
        self._edit_junctions.setPlaceholderText("G2, C1  (blank = same everywhere)")
        self._edit_junctions.setToolTip(
            "Continuity of each junction in order, overriding the Continuity\n"
            "setting. One entry per junction (sections - 1), each G0-G3 or C0-C3.\n"
            "VarPro LM only."
        )
        self._lbl_junctions = _add_row(form, "Per junction", self._edit_junctions)

        self._check_subsample = QCheckBox("to")
        self._check_subsample.setChecked(False)
        self._check_subsample.setToolTip(
            "For dense data: fit an evenly spaced subsample of each section, then\n"
            "refine the fit on all points. Much faster for thousands of points.\n"
            "Off: every point is used throughout."
        )
        self._spin_subsample = QSpinBox()
        self._spin_subsample.setRange(20, 100000)
        self._spin_subsample.setValue(200)
        self._spin_subsample.setSuffix(" pts/section")
        self._spin_subsample.setEnabled(False)
        self._spin_subsample.setToolTip("Sections with more points than this are subsampled.")
        self._subsample_row = _hbox(self._check_subsample, self._spin_subsample, stretch_first=False)
        self._subsample_row.layout().setStretch(1, 1)
        self._lbl_subsample = _add_row(form, "Subsample", self._subsample_row)

        # Bending energy weight (GN-VarPro fixed count, sweepable)
        self._spin_smoothing_weight = QDoubleSpinBox()
        self._spin_smoothing_weight.setRange(0.0, 1e12)
        self._spin_smoothing_weight.setDecimals(12)
        self._spin_smoothing_weight.setSingleStep(0.001)
        self._spin_smoothing_weight.setValue(0.001)
        self._spin_smoothing_weight.setToolTip("Bending energy weight for GN-VarPro (0=off, 1e-3=default).")
        self._sweep_smoothing_weight = SweepableDoubleSpinBox(
            self._spin_smoothing_weight, group=self._sweep_group, default_steps=10,
        )
        self._lbl_smoothing_weight = _add_row(form, "Smoothing", self._sweep_smoothing_weight)

        # Smoothing factor (algebraic methods)
        self._spin_smoothing = QDoubleSpinBox()
        self._spin_smoothing.setRange(0.0, 1.0)
        self._spin_smoothing.setDecimals(6)
        self._spin_smoothing.setSingleStep(0.0001)
        self._spin_smoothing.setValue(0.0)
        self._spin_smoothing.setToolTip("Smoothing factor for algebraic fitting (0 = exact).")
        self._lbl_smoothing = _add_row(form, "LSQ smoothing", self._spin_smoothing)

        # Iterative optimizers
        self._spin_maxiter = QSpinBox()
        self._spin_maxiter.setRange(10, 1_000_000_000)
        self._spin_maxiter.setValue(10000)
        self._spin_maxiter.setToolTip("Maximum optimizer iterations.")
        self._lbl_maxiter = _add_row(form, "Iteration limit", self._spin_maxiter)

        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(-1, 2**31 - 1)
        self._spin_seed.setValue(-1)
        self._spin_seed.setSpecialValueText("Random")
        self._spin_seed.setToolTip("Random seed (Random = system random, \u22650 = deterministic).")
        _add_row(form, "Seed", self._spin_seed)

        # Outer optimizer (parameter optimisation)
        from ..optimization.optimizers import get_optimizer
        self._combo_param_opt_outer = QComboBox()
        for opt_type in OptimizerType:
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
        idx = self._combo_param_opt_outer.findData(OptimizerType.L_BFGS_B)
        if idx >= 0:
            self._combo_param_opt_outer.setCurrentIndex(idx)
        self._combo_param_opt_outer.setToolTip("Outer-loop optimizer for parameter optimisation.")
        self._lbl_param_opt_outer = _add_row(form, "Outer optimizer", self._combo_param_opt_outer)

        # Orthogonal LSQ local bounds
        self._check_orth_lsq_local_bounds = QCheckBox("Use local t bounds")
        self._check_orth_lsq_local_bounds.setChecked(False)
        self._check_orth_lsq_local_bounds.setToolTip("Restrict foot-point search to local bounds.")
        self._lbl_orth_lsq_local_bounds = _add_row(form, "Foot points", self._check_orth_lsq_local_bounds)

        # Search bounds and junction
        self._spin_bbox = QDoubleSpinBox()
        self._spin_bbox.setRange(1.0, 100.0)
        self._spin_bbox.setDecimals(2)
        self._spin_bbox.setSingleStep(0.1)
        self._spin_bbox.setValue(3.0)
        self._spin_bbox.setToolTip("Multiplier for optimization search bounds.")
        _add_row(form, "Search scale", self._spin_bbox)

        self._check_optimize_le = QCheckBox("Let it move")
        self._check_optimize_le.setChecked(False)
        self._check_optimize_le.setToolTip(
            "Allow the leading edge junction to move during optimization."
        )
        _add_row(form, "Junction", self._check_optimize_le)

        section.add_layout(form)
        section.add_widget(self._build_regularization_box())

        # "N changed" badge in the header, so non-default settings stay
        # noticeable while the section is collapsed
        self._advanced_defaults = [
            (w, self._widget_value(w)) for w in (
                self._edit_junctions, self._check_subsample, self._spin_subsample,
                self._spin_smoothing_weight, self._spin_smoothing, self._spin_maxiter,
                self._spin_seed, self._combo_param_opt_outer,
                self._check_orth_lsq_local_bounds, self._spin_bbox,
                self._check_optimize_le, self._check_fairness,
            )
        ]
        for w, _ in self._advanced_defaults:
            signal = (w.textChanged if isinstance(w, QLineEdit)
                      else w.toggled if isinstance(w, QCheckBox)
                      else w.currentIndexChanged if isinstance(w, QComboBox)
                      else w.valueChanged)
            signal.connect(self._update_advanced_badge)

        return section

    def _build_regularization_box(self) -> QWidget:
        """Build the fairness penalties (objective-based optimizers only)."""
        self._regularization_box = QWidget()
        layout = QVBoxLayout(self._regularization_box)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(6)

        caption = QLabel("REGULARIZATION")
        caption.setObjectName("SectionCaption")
        layout.addWidget(caption)

        self._check_fairness = QCheckBox("Add fairness penalties")
        self._check_fairness.setChecked(False)
        self._check_fairness.setToolTip("Add fairness penalties to the objective function.")
        layout.addWidget(self._check_fairness)

        self._fairness_container = QWidget()
        wf = _form()
        self._fairness_container.setLayout(wf)

        def _ws() -> QDoubleSpinBox:
            s = QDoubleSpinBox(); s.setRange(0.0, 1.0); s.setDecimals(4)
            s.setSingleStep(0.001); s.setValue(0.0); return s

        self._spin_w_lenvar = _ws()
        self._spin_w_lenvar.setToolTip("Penalize uneven spacing between control points")
        _add_row(wf, "Even edges", self._spin_w_lenvar)

        self._spin_w_bend = _ws()
        self._spin_w_bend.setToolTip("Penalize sharp bends in the control polygon")
        _add_row(wf, "Polygon bend", self._spin_w_bend)

        self._spin_w_short = _ws()
        self._spin_w_short.setToolTip("Penalize control points that are too close together")
        _add_row(wf, "Min spacing", self._spin_w_short)

        self._spin_w_tangent = _ws()
        self._spin_w_tangent.setToolTip("Penalize unusual edge lengths at curve endpoints")
        _add_row(wf, "End handles", self._spin_w_tangent)

        self._spin_w_curv = _ws()
        self._spin_w_curv.setToolTip("Minimize bending energy: \u222B\u03BA\u00B2 ds")
        _add_row(wf, "Bending", self._spin_w_curv)

        self._spin_w_grad = _ws()
        self._spin_w_grad.setToolTip("Minimize wiggle energy: \u222B(d\u03BA/ds)\u00B2 ds")
        _add_row(wf, "Wiggle", self._spin_w_grad)

        layout.addWidget(self._fairness_container)
        self._fairness_container.setEnabled(False)
        return self._regularization_box

    @staticmethod
    def _widget_value(w):
        """Current value of an input widget, for default comparison."""
        if isinstance(w, QLineEdit):
            return w.text().strip()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentIndex()
        if isinstance(w, QAbstractSpinBox):
            return w.value()
        raise TypeError(type(w))

    def _update_advanced_badge(self, *_):
        """Count the applicable Advanced settings that differ from their defaults."""
        content = self._advanced_section.content_widget()
        changed = sum(
            1 for w, default in self._advanced_defaults
            if w.isVisibleTo(content) and self._widget_value(w) != default
        )
        self._advanced_section.set_badge(f"{changed} changed" if changed else "")

    # -- Run Bar -----------------------------------------------------------

    def _build_run_bar(self) -> QWidget:
        """Build the fixed bar under the sidebar: fit, result, and export."""
        bar = QWidget()
        bar.setObjectName("RunBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self._btn_fit = QPushButton("Fit spline")
        self._btn_fit.setObjectName("PrimaryButton")
        self._btn_fit.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self._btn_fit)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setVisible(False)
        layout.addWidget(self._btn_cancel)

        # Result of the current fit, with the exports for it
        self._result_box = QWidget()
        result_layout = QVBoxLayout(self._result_box)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(8)

        stats = QGridLayout()
        stats.setHorizontalSpacing(8)
        stats.setVerticalSpacing(0)
        self._lbl_result_caption_dev = QLabel("Max dev")
        self._lbl_result_dev = QLabel()
        self._lbl_result_ctrl = QLabel()
        self._lbl_result_time = QLabel()
        for col, (caption, value) in enumerate((
            (self._lbl_result_caption_dev, self._lbl_result_dev),
            (QLabel("Ctrl pts"), self._lbl_result_ctrl),
            (QLabel("Time"), self._lbl_result_time),
        )):
            caption.setObjectName("Muted")
            stats.addWidget(caption, 0, col)
            stats.addWidget(value, 1, col)
        result_layout.addLayout(stats)

        save_row = QHBoxLayout()
        save_row.setSpacing(6)
        self._btn_save_dat = QPushButton("Save .dat")
        self._btn_save_dat.setEnabled(False)
        self._btn_save_dat.setToolTip("Save fitted airfoil coordinates (.dat)")
        save_row.addWidget(self._btn_save_dat)
        self._btn_save_spline = QPushButton("Save spline")
        self._btn_save_spline.setEnabled(False)
        self._btn_save_spline.setToolTip("Save spline control points and knots (.spline.dat)")
        save_row.addWidget(self._btn_save_spline)
        self._btn_export_cad = QPushButton("Export CAD")
        self._btn_export_cad.setEnabled(False)
        self._btn_export_cad.setToolTip("Export the exact B-spline curves as STEP, IGES or DXF")
        save_row.addWidget(self._btn_export_cad)
        result_layout.addLayout(save_row)

        self._result_box.setVisible(False)
        layout.addWidget(self._result_box)
        return bar
