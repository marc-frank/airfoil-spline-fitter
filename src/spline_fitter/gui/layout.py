"""
Window Layout
=============

Widget construction for the main window: menu bar, sidebar sections,
plots with their toolbars, and the run bar.
"""

from __future__ import annotations

import math

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
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..optimization.models import (
    ContinuityMode,
    ErrorMetric,
    OptimizerType,
)
from .explorer import COLUMNS, RunTable
from .motion import SmoothViewBox
from .theme import Size, glyph_size, icon, label_plot, style_plot
from .widgets import (
    CollapsibleSection,
    DropDownPanel,
    ResettableSplitter,
    ScientificSpinBox,
    SegmentedControl,
    SidebarSection,
    SweepableDoubleSpinBox,
    SweepableSpinBox,
)

# Default widths of the sidebar (at least: size_sidebar widens it to fit
# Advanced with the smoothing sweep open) and the project explorer, whose
# table scrolls sideways through its columns
SIDEBAR_WIDTH = 330
EXPLORER_WIDTH = 570

# Most sections the curve can be split into; each needs enough points
# for its own spline, which airfoil files of 50 or more points give.
MAX_SECTIONS = 20

# Width of the label column in every sidebar form, so fields line up
# across sections.
LABEL_WIDTH = Size.LABEL_WIDTH


def _form() -> QFormLayout:
    """A sidebar form layout."""
    form = QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setHorizontalSpacing(Size.GROUP)
    form.setVerticalSpacing(Size.GAP)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return form


def _add_row(form: QFormLayout, text: str, field) -> QLabel:
    """Add a row with a fixed-width label and return the label."""
    label = QLabel(text)
    label.setObjectName("FormLabel")
    label.setFixedWidth(LABEL_WIDTH)
    form.addRow(label, field)
    return label


def _hbox(*widgets, stretch_first: bool = True) -> QWidget:
    """Put widgets side by side in one form field."""
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(Size.GAP)
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
        self._main_splitter = ResettableSplitter(Qt.Orientation.Horizontal, self)
        # Double-clicking a handle puts the panel beside it back to its default width
        self._main_splitter.handleDoubleClicked.connect(self._reset_panel_width)
        # Until a handle is dragged, the side panels keep their default widths
        # as the window settles (it is maximized after it is first shown)
        self._panels_dragged = False
        self._main_splitter.splitterMoved.connect(
            lambda *_: setattr(self, "_panels_dragged", True))

        # Sidebar: scrolling sections above a fixed run bar, in workflow
        # order (data -> method -> curve -> advanced -> fit and results)
        sidebar = QWidget(self)
        sidebar.setObjectName("Sidebar")
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

        # NOTE: every setting added to these sections must be locked while a
        # fit runs: add it to _set_fitting_ui (ui.py), or put it in Advanced,
        # whose content is locked as a whole
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
        self._sidebar_sections, self._sidebar_scroll = sections, sidebar_scroll

        sidebar_layout.addWidget(sidebar_scroll, stretch=1)
        sidebar_layout.addWidget(self._build_run_bar())

        # Center panel: Plots
        center_panel = QWidget(self)
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(Size.GROUP, Size.GROUP, Size.GROUP, Size.GROUP)
        center_layout.setSpacing(Size.GAP)

        # Main plot (airfoil visualization) with its display toolbar
        center_layout.addLayout(self._build_plot_toolbar())
        self._plot = pg.PlotWidget(parent=center_panel, viewBox=SmoothViewBox())
        style_plot(self._plot)
        self._plot.getPlotItem().setAspectLocked(True, ratio=1)
        label_plot(self._plot, "bottom", "x")
        label_plot(self._plot, "left", "y")
        center_layout.addWidget(self._plot, stretch=3)

        # Bottom row: Loss plot (with inline toolbar) and Comparison plot
        bottom_plots = QWidget()
        bottom_plots_layout = QHBoxLayout(bottom_plots)
        bottom_plots_layout.setContentsMargins(0, 0, 0, 0)
        bottom_plots_layout.setSpacing(Size.GROUP)

        # Loss plot with inline axis controls
        loss_container = QWidget()
        loss_layout = QVBoxLayout(loss_container)
        loss_layout.setContentsMargins(0, 0, 0, 0)
        loss_layout.setSpacing(Size.GAP)
        loss_layout.addLayout(self._build_loss_toolbar())

        self._loss_plot = pg.PlotWidget(parent=loss_container, viewBox=SmoothViewBox())
        style_plot(self._loss_plot)
        label_plot(self._loss_plot, "bottom", "Evaluations")
        label_plot(self._loss_plot, "left", "Error")
        self._loss_plot.getPlotItem().setLogMode(x=True, y=True)
        loss_layout.addWidget(self._loss_plot)
        bottom_plots_layout.addWidget(loss_container, stretch=1)

        # Comparison plot: the runs that pass the explorer's filter, along
        # the columns chosen in its toolbar
        comparison_container = QWidget()
        comparison_layout = QVBoxLayout(comparison_container)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.setSpacing(Size.GAP)
        comparison_layout.addLayout(self._build_comparison_toolbar())
        self._comparison_plot = pg.PlotWidget(parent=comparison_container, viewBox=SmoothViewBox())
        style_plot(self._comparison_plot)
        label_plot(self._comparison_plot, "bottom", "Control points")
        label_plot(self._comparison_plot, "left", "Error")
        self._comparison_plot.getPlotItem().setLogMode(x=False, y=True)
        comparison_layout.addWidget(self._comparison_plot)
        bottom_plots_layout.addWidget(comparison_container, stretch=1)

        center_layout.addWidget(bottom_plots, stretch=2)

        # Right panel: Project Explorer
        explorer_panel = self._build_project_explorer()
        self._explorer_panel = explorer_panel

        # Configure splitter
        self._main_splitter.addWidget(sidebar)
        self._main_splitter.addWidget(center_panel)
        self._main_splitter.addWidget(explorer_panel)
        sidebar.setMinimumWidth(300)
        # The explorer is hidden with its button or the View menu, never by
        # dragging the splitter shut, which left no way to bring it back
        explorer_panel.setMinimumWidth(180)
        self._main_splitter.setCollapsible(0, False)
        self._main_splitter.setCollapsible(1, False)
        self._main_splitter.setCollapsible(2, False)
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)
        # Widened to fit its rows once shown (size_sidebar), when the styled
        # fields have their real sizes
        self._sidebar_width = SIDEBAR_WIDTH
        self._main_splitter.setSizes([SIDEBAR_WIDTH, 440, EXPLORER_WIDTH])

        # The splitter, and at the right edge the rail a hidden explorer
        # folds into: a chevron pointing back out shows it again
        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self._main_splitter, stretch=1)
        central_layout.addWidget(self._build_explorer_rail())
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def size_sidebar(self) -> None:
        """Widen the sidebar's default to what its widest row needs: Advanced
        open and each sweep range shown in turn (the smoothing sweep is the
        widest), rounded up to 10 px."""
        sections, advanced = self._sidebar_sections, self._advanced_section
        was_expanded = advanced._expanded
        advanced.set_expanded(True)
        widest = sections.minimumSizeHint().width()
        for box in self._sweep_group:
            if box.isEnabled() and not box.is_sweep_active:
                box._btn_sweep.click()
                widest = max(widest, sections.minimumSizeHint().width())
                box.stop_sweep()
        advanced.set_expanded(was_expanded)
        width = widest + self._sidebar_scroll.verticalScrollBar().sizeHint().width()
        self._sidebar_width = max(SIDEBAR_WIDTH, math.ceil(width / 10) * 10)
        self.reset_panel_widths()

    def reset_panel_widths(self) -> None:
        """Sidebar and explorer at their default widths."""
        self._reset_panel_width(1)
        self._reset_panel_width(2)

    def _reset_panel_width(self, handle: int) -> None:
        """Give the panel beside a splitter handle its default width (1: the
        sidebar, 2: the explorer), taking or giving the difference to the plots."""
        sizes = self._main_splitter.sizes()
        panel, default = ((0, self._sidebar_width) if handle == 1
                          else (2, EXPLORER_WIDTH))
        sizes[1] -= default - sizes[panel]
        sizes[panel] = default
        self._main_splitter.setSizes(sizes)

    # -- Menu Bar ----------------------------------------------------------

    def _build_menu_bar(self):
        """Build the application menu bar (File / View / Help)."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        self._action_open = file_menu.addAction("&Open airfoil...")
        self._action_open.setShortcut("Ctrl+O")

        file_menu.addSeparator()
        self._action_save_dat = file_menu.addAction("Save fitted airfoil (.dat)...")
        self._action_save_dat.setEnabled(False)
        self._action_save_spline = file_menu.addAction("Save spline (.spline.dat)...")
        self._action_save_spline.setEnabled(False)
        self._action_export_cad = file_menu.addAction("Export CAD (STEP, IGES, DXF)...")
        self._action_export_cad.setShortcut("Ctrl+E")
        self._action_export_cad.setEnabled(False)

        file_menu.addSeparator()
        self._action_open_project = file_menu.addAction("Open &project...")
        self._action_save_project = file_menu.addAction("&Save project")
        self._action_save_project.setShortcut("Ctrl+S")
        self._action_save_project.setEnabled(False)
        self._action_save_project_as = file_menu.addAction("Save project &as...")
        self._action_save_project_as.setEnabled(False)

        file_menu.addSeparator()
        self._action_batch = file_menu.addAction("&Batch convert...")

        file_menu.addSeparator()
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)

        # View menu
        view_menu = menubar.addMenu("&View")
        self._action_fit_report = view_menu.addAction("&Fit report...")
        self._action_fit_report.setShortcut("Ctrl+R")
        self._action_fit_report.setEnabled(False)
        self._action_view_report = view_menu.addAction("&Benchmark report...")
        view_menu.addSeparator()
        # Shows and hides the explorer; its own button can only hide it
        self._action_toggle_explorer = view_menu.addAction("Project &explorer")
        self._action_toggle_explorer.setCheckable(True)
        self._action_toggle_explorer.setChecked(True)
        self._action_toggle_explorer.setShortcut("Ctrl+Shift+E")
        view_menu.addSeparator()
        self._action_theme = view_menu.addAction("&Theme...")

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
        row.setSpacing(Size.GROUP)

        text = QVBoxLayout()
        text.setSpacing(0)
        self._lbl_file_name = QLabel("No airfoil loaded")
        self._lbl_file_name.setObjectName("FileName")
        text.addWidget(self._lbl_file_name)
        self._lbl_path = QLabel("Open a .dat file or a project")
        self._lbl_path.setObjectName("Muted")
        text.addWidget(self._lbl_path)
        row.addLayout(text, stretch=1)

        self._btn_open = QPushButton("Open...")
        self._btn_open.setToolTip("Open an airfoil coordinate file (Ctrl+O)")
        row.addWidget(self._btn_open, alignment=Qt.AlignmentFlag.AlignVCenter)

        card.add_layout(row)
        return card

    # -- Plot Toolbar ------------------------------------------------------

    def _build_plot_toolbar(self) -> QHBoxLayout:
        """Build the display toolbar above the main plot."""
        toolbar = QHBoxLayout()
        toolbar.setSpacing(Size.GAP)
        toolbar.setContentsMargins(0, 0, 0, 0)

        self._check_knots = QToolButton()
        self._check_knots.setText("Knots")
        self._check_knots.setCheckable(True)
        self._check_knots.setToolTip("Mark the curve points at the knots of every section.")
        toolbar.addWidget(self._check_knots)

        # Curvature comb: toggle, with its settings in the drop-down
        self._check_comb = QToolButton()
        self._check_comb.setText("Curvature comb")
        self._check_comb.setObjectName("SplitToggle")
        self._check_comb.setCheckable(True)
        self._check_comb.setToolTip(
            "Display curvature as perpendicular lines from the curve.\n"
            "Use the arrow for quantity, height and density."
        )

        comb_settings = QWidget()
        comb_form = _form()
        comb_settings.setLayout(comb_form)
        comb_form.setContentsMargins(Size.PAD, Size.PAD, Size.PAD, Size.PAD)

        self._combo_comb_qty = QComboBox()
        self._combo_comb_qty.addItem("Curvature (\u03BA)", userData=0)
        self._combo_comb_qty.addItem("Gradient (d\u03BA/ds)", userData=1)
        self._combo_comb_qty.addItem("Acceleration (d\u00B2\u03BA/ds\u00B2)", userData=2)
        self._combo_comb_qty.addItem("Jerk (d\u00B3\u03BA/ds\u00B3)", userData=3)
        self._combo_comb_qty.setCurrentIndex(0)
        _add_row(comb_form, "Quantity", self._combo_comb_qty)

        self._spin_comb_scale = QDoubleSpinBox()
        self._spin_comb_scale.setRange(0.0, 1000.0)
        self._spin_comb_scale.setDecimals(4)
        self._spin_comb_scale.setSingleStep(0.01)
        self._spin_comb_scale.setValue(0.05)
        _add_row(comb_form, "Height", self._spin_comb_scale)

        self._spin_comb_density = QSpinBox()
        self._spin_comb_density.setRange(5, 2000)
        self._spin_comb_density.setValue(250)
        _add_row(comb_form, "Density", self._spin_comb_density)

        # A drop-down panel rather than a menu, so the plots stay usable
        # while the settings are open
        self._btn_comb_settings = QToolButton()
        self._btn_comb_settings.setObjectName("ArrowButton")
        self._btn_comb_settings.setIcon(icon("chevron_down"))
        self._btn_comb_settings.setIconSize(glyph_size())
        self._btn_comb_settings.setToolTip("Curvature comb settings")
        self._comb_panel = DropDownPanel(self._btn_comb_settings, comb_settings, self)
        comb_row = QHBoxLayout()
        comb_row.setSpacing(0)
        comb_row.addWidget(self._check_comb)
        comb_row.addWidget(self._btn_comb_settings)
        toolbar.addLayout(comb_row)

        toolbar.addStretch()

        live_label = QLabel("Live preview")
        live_label.setObjectName("FormLabel")
        toolbar.addWidget(live_label)
        self._combo_live = QComboBox()
        self._combo_live.addItems(["Minimal", "Moderate", "Detailed"])
        self._combo_live.setCurrentIndex(1)
        # Read by the fit's worker thread, so kept as a plain int; it can
        # change while fitting
        self._live_mode = self._combo_live.currentIndex()
        self._combo_live.currentIndexChanged.connect(lambda i: setattr(self, "_live_mode", i))
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
        toolbar.setSpacing(Size.GAP)
        toolbar.setContentsMargins(0, 0, 0, 0)

        caption = QLabel("CONVERGENCE")
        caption.setObjectName("SectionCaption")
        toolbar.addWidget(caption)
        toolbar.addStretch()

        x_label = QLabel("X axis")
        x_label.setObjectName("FormLabel")
        toolbar.addWidget(x_label)
        self._combo_loss_x = QComboBox()
        self._combo_loss_x.addItem("Iterations", userData="iters")
        self._combo_loss_x.addItem("Time", userData="time")
        self._combo_loss_x.setFixedWidth(96)
        toolbar.addWidget(self._combo_loss_x)

        scale_label = QLabel("Log scale")
        scale_label.setObjectName("FormLabel")
        toolbar.addWidget(scale_label)
        self._combo_log_scale = QComboBox()
        for text, axes in (("None", "none"), ("X only", "x"), ("Y only", "y"), ("Both", "both")):
            self._combo_log_scale.addItem(text, userData=axes)
        self._combo_log_scale.setCurrentIndex(self._combo_log_scale.findData("y"))
        self._combo_log_scale.setFixedWidth(84)
        toolbar.addWidget(self._combo_log_scale)
        return toolbar

    def _build_comparison_toolbar(self) -> QHBoxLayout:
        """Build the toolbar above the comparison plot: which explorer
        columns the runs are plotted along."""
        toolbar = QHBoxLayout()
        toolbar.setSpacing(Size.GAP)
        toolbar.setContentsMargins(0, 0, 0, 0)

        caption = QLabel("RUNS")
        caption.setObjectName("SectionCaption")
        toolbar.addWidget(caption)
        toolbar.addStretch()

        x_label = QLabel("X axis")
        x_label.setObjectName("FormLabel")
        toolbar.addWidget(x_label)
        self._combo_runs_x = QComboBox()
        for c in COLUMNS:
            self._combo_runs_x.addItem(c.name)
        self._combo_runs_x.setCurrentText("Control points")
        self._combo_runs_x.setToolTip(
            "The explorer column the runs are plotted along.\n"
            "Only the runs that pass the explorer's filter are shown.")
        toolbar.addWidget(self._combo_runs_x, stretch=1)

        y_label = QLabel("Y axis")
        y_label.setObjectName("FormLabel")
        toolbar.addWidget(y_label)
        self._combo_runs_y = QComboBox()
        for c in COLUMNS:
            if c.metric:
                self._combo_runs_y.addItem(c.name)
        self._combo_runs_y.setCurrentText("Error")
        self._combo_runs_y.setToolTip("The fit metric plotted for each run")
        toolbar.addWidget(self._combo_runs_y, stretch=1)

        # Compact, as the plot is narrow; their lists open as wide as the
        # longest column name
        for combo in (self._combo_runs_x, self._combo_runs_y):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(4)
            combo.view().setMinimumWidth(combo.view().sizeHintForColumn(0) + 2 * Size.PAD)
        return toolbar

    # -- Project Explorer --------------------------------------------------

    def _build_explorer_rail(self) -> QWidget:
        """The strip a hidden explorer folds into, with a chevron to show it again."""
        rail = QWidget()
        rail.setObjectName("ExplorerRail")
        rail.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(Size.TIGHT, Size.PAD - 2, Size.TIGHT, Size.PAD)
        self._btn_show_explorer = QPushButton()
        self._btn_show_explorer.setObjectName("IconButton")
        self._btn_show_explorer.setIcon(icon("chevron_left"))
        self._btn_show_explorer.setIconSize(glyph_size())
        self._btn_show_explorer.setFixedWidth(Size.CONTROL_HEIGHT)
        self._btn_show_explorer.setToolTip("Show the project explorer (Ctrl+Shift+E)")
        layout.addWidget(self._btn_show_explorer)
        layout.addStretch()
        rail.setVisible(False)
        self._explorer_rail = rail
        return rail


    def _build_project_explorer(self) -> QWidget:
        """Build the project explorer panel: the table of runs (explorer.py)."""
        panel = QWidget()
        panel.setObjectName("Explorer")
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(Size.PAD, Size.PAD - 2, Size.PAD, Size.PAD)
        layout.setSpacing(Size.GROUP)

        # Header with toggle button
        header = QHBoxLayout()
        caption = QLabel("PROJECT")
        caption.setObjectName("SectionCaption")
        header.addWidget(caption)
        header.addStretch()
        # A sideways chevron: the explorer folds away to the right
        self._btn_toggle_explorer = QPushButton()
        self._btn_toggle_explorer.setObjectName("IconButton")
        self._btn_toggle_explorer.setIcon(icon("chevron_right"))
        self._btn_toggle_explorer.setIconSize(glyph_size())
        self._btn_toggle_explorer.setToolTip("Hide the project explorer (Ctrl+Shift+E)")
        header.addWidget(self._btn_toggle_explorer)
        layout.addLayout(header)

        # Project info label (was in file tab, now lives here)
        self._lbl_project = QLabel("No project")
        self._lbl_project.setWordWrap(True)
        self._lbl_project.setObjectName("Muted")
        layout.addWidget(self._lbl_project)

        # Every run of the project, filterable and sortable; double-click
        # applies a run's settings
        self._run_table = RunTable()
        layout.addWidget(self._run_table, stretch=1)

        # Run details label
        self._lbl_run_details = QLabel("Select a run to view details")
        self._lbl_run_details.setWordWrap(True)
        self._lbl_run_details.setObjectName("Muted")
        layout.addWidget(self._lbl_run_details)

        # Action buttons
        btn_row = QHBoxLayout()
        self._btn_delete_run = QPushButton("Delete")
        self._btn_delete_run.setEnabled(False)
        self._btn_delete_run.setToolTip("Delete the selected runs from the project")
        btn_row.addWidget(self._btn_delete_run)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return panel

    # -- Method Section ----------------------------------------------------

    def _build_method_section(self) -> SidebarSection:
        """Build the Method section: optimizer, goal, and error metric."""
        section = SidebarSection("Method")
        form = _form()

        # Algorithm (decides which other controls apply, so it comes first).
        # Grouped under captions: the least-squares methods (the default
        # VarPro LM first), then the general-purpose local and global optimizers
        self._combo_opt = QComboBox()
        lsq = [OptimizerType.GN_VARPRO, OptimizerType.DIRECT_LSQ]
        rest = [t for t in OptimizerType if t not in lsq]
        groups = (
            ("LEAST SQUARES", lsq),
            ("LOCAL", [t for t in rest if not t.is_global]),
            ("GLOBAL", [t for t in rest if t.is_global]),
        )
        for caption, opt_types in groups:
            if self._combo_opt.count():
                self._combo_opt.insertSeparator(self._combo_opt.count())
            self._combo_opt.addItem(caption)
            # A caption, not a choice: unselectable and skipped by the keys
            self._combo_opt.model().item(self._combo_opt.count() - 1).setEnabled(False)
            for opt_type in opt_types:
                self._combo_opt.addItem(opt_type.display_name, userData=opt_type)
                self._combo_opt.setItemData(
                    self._combo_opt.count() - 1, opt_type.tooltip, Qt.ItemDataRole.ToolTipRole)
        self._combo_opt.setCurrentIndex(self._combo_opt.findData(OptimizerType.GN_VARPRO))
        self._combo_opt.currentIndexChanged.connect(self._on_optimizer_changed)
        _add_row(form, "Optimizer", self._combo_opt)

        # Goal (VarPro): meet a tolerance, or fit a fixed control point count
        self._seg_goal = SegmentedControl()
        self._seg_goal.addItem("Fixed count", userData=False)
        self._seg_goal.addItem("Tolerance", userData=True)
        self._seg_goal.setItemToolTip(0, "Fit the given number of control points per section.")
        self._seg_goal.setItemToolTip(
            1,
            "Find the fewest control points that keep\n"
            "every data point within the tolerance.",
        )
        self._lbl_goal = _add_row(form, "Goal", self._seg_goal)

        self._spin_tolerance = ScientificSpinBox()
        self._spin_tolerance.setRange(1e-9, 1.0)
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
        self._combo_knot_placement.addItem("Uniform", userData="uniform")
        self._combo_knot_placement.addItem("Free", userData="free")
        self._combo_knot_placement.setItemToolTip(
            0, "Knots stay evenly spaced in the curve parameter.")
        self._combo_knot_placement.setItemToolTip(
            1, "Knots move to where they reduce the error most.")
        self._lbl_knot_placement = _add_row(form, "Knots", self._combo_knot_placement)

        # Parameterization (Direct LSQ)
        self._seg_parameterization = SegmentedControl()
        self._seg_parameterization.addItem("Centripetal", userData="centripetal")
        self._seg_parameterization.addItem("Arc length", userData="arc_length")
        self._seg_parameterization.setItemToolTip(
            0, "Space the data parameters by the square root of the point distances.\n"
               "Follows tight curvature, such as the leading edge, more closely.")
        self._seg_parameterization.setItemToolTip(
            1, "Space the data parameters by the distance along the data.")
        self._lbl_parameterization = _add_row(form, "Parameterization", self._seg_parameterization)

        self._check_grow = QCheckBox("Add where the fit is worst")
        self._check_grow.setChecked(False)
        self._check_grow.setToolTip(
            "On: start from the fewest control points and add them where the\n"
            "fit is worst, refitting after each step.\n"
            "Off: start from the Control points setting (Curve). With a tolerance\n"
            "goal the count can still grow or shrink from there to meet it."
        )
        self._lbl_grow = _add_row(form, "Grow points", self._check_grow)

        # Error metric
        self._seg_err = SegmentedControl()
        for metric, text, tip in (
            (ErrorMetric.PARAMETRIC, "Pointwise",
             "Average distance from each data point to the curve point at its parameter.\n"
             "Fast."),
            (ErrorMetric.ORTHOGONAL, "Orthogonal",
             "Average distance from each data point to the nearest curve point.\n"
             "Accurate but slower."),
            (ErrorMetric.ORTHOGONAL_LOCAL, "Local",
             "Orthogonal distance, searching only near each point's parameter.\n"
             "A balance of the two."),
        ):
            self._seg_err.addItem(text, userData=metric)
            self._seg_err.setItemToolTip(self._seg_err.count() - 1, tip)
        self._lbl_err = _add_row(form, "Error metric", self._seg_err)

        section.add_layout(form)
        return section

    # -- Curve Section -----------------------------------------------------

    def _build_curve_section(self) -> SidebarSection:
        """Build the Curve section: degree, control points, sections, continuity."""
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

        # Control points (sweepable); with a tolerance goal only the starting
        # count, and only when not growing from the fewest
        self._spin_ctrl = QSpinBox()
        self._spin_ctrl.setRange(2, 200)
        self._spin_ctrl.setValue(25)
        self._spin_ctrl.setToolTip(
            "Control points per section.\n"
            "More = more flexibility but more variables."
        )
        self._sweep_ctrl = SweepableSpinBox(self._spin_ctrl, group=self._sweep_group)
        self._lbl_ctrl = _add_row(form, "Control points", self._sweep_ctrl)

        # Sections
        self._spin_sections = QSpinBox()
        self._spin_sections.setRange(1, MAX_SECTIONS)
        self._spin_sections.setValue(2)
        self._spin_sections.setToolTip(
            "Number of sections, each its own spline, joined at junctions.\n"
            "2 (default) splits upper and lower surface at the leading edge."
        )
        self._sweep_sections = SweepableSpinBox(self._spin_sections, group=self._sweep_group)
        _add_row(form, "Sections", self._sweep_sections)

        # Continuity at the junctions; labels read G or C depending on the type
        self._combo_cont = SegmentedControl()
        cont_names = ["Position", "Tangent", "Curvature", "Curvature rate"]
        for i, mode in enumerate(
            (ContinuityMode.G0, ContinuityMode.G1, ContinuityMode.G2, ContinuityMode.G3)
        ):
            self._combo_cont.addItem(f"C{i}", userData=mode)
            self._combo_cont.setItemToolTip(i, f"{cont_names[i]} matches across junctions")
        self._combo_cont.setCurrentIndex(3)
        self._lbl_cont = _add_row(form, "Continuity", self._combo_cont)

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
        self._spin_subsample.setSuffix(" points")
        self._spin_subsample.setEnabled(False)
        self._spin_subsample.setToolTip(
            "Points per section. Sections with more points than this are subsampled.")
        self._subsample_row = _hbox(self._check_subsample, self._spin_subsample, stretch_first=False)
        self._subsample_row.layout().setStretch(1, 1)
        self._lbl_subsample = _add_row(form, "Subsample", self._subsample_row)

        # Bending energy weight (VarPro LM fixed count, sweepable)
        self._spin_smoothing_weight = ScientificSpinBox()
        self._spin_smoothing_weight.setRange(0.0, 1e12)
        self._spin_smoothing_weight.setSingleStep(0.001)
        self._spin_smoothing_weight.setValue(0.001)
        self._spin_smoothing_weight.setToolTip(
            "Bending energy weight for VarPro LM (0 turns it off; default 0.001).")
        self._sweep_smoothing_weight = SweepableDoubleSpinBox(
            self._spin_smoothing_weight, group=self._sweep_group,
        )
        self._lbl_smoothing_weight = _add_row(form, "Smoothing", self._sweep_smoothing_weight)

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
        self._spin_seed.setToolTip(
            "Random seed. Random: a new seed for every fit;\n0 or more: the same result every time.")
        self._lbl_seed = _add_row(form, "Seed", self._spin_seed)

        # Search bounds and junction
        self._spin_bbox = QDoubleSpinBox()
        self._spin_bbox.setRange(1.0, 100.0)
        self._spin_bbox.setDecimals(2)
        self._spin_bbox.setSingleStep(0.1)
        self._spin_bbox.setValue(3.0)
        self._spin_bbox.setToolTip("Multiplier for optimization search bounds.")
        self._lbl_bbox = _add_row(form, "Search scale", self._spin_bbox)

        self._check_optimize_le = QCheckBox("Optimize positions")
        self._check_optimize_le.setChecked(False)
        self._check_optimize_le.setToolTip(
            "Optimize the position of every junction along the data,\n"
            "instead of keeping the section split points fixed."
        )
        self._lbl_optimize_le = _add_row(form, "Junctions", self._check_optimize_le)

        section.add_layout(form)
        section.add_widget(self._build_regularization_box())

        # "N changed" badge in the header, so settings changed this session
        # stay noticeable while the section is collapsed; its tooltip names them
        self._advanced_fields = (
            ("Per junction", self._edit_junctions),
            ("Subsample", self._check_subsample),
            ("Subsample points", self._spin_subsample),
            ("Smoothing", self._spin_smoothing_weight),
            ("Iteration limit", self._spin_maxiter),
            ("Seed", self._spin_seed),
            ("Search scale", self._spin_bbox),
            ("Junctions", self._check_optimize_le),
            ("Fairness penalties", self._check_fairness),
        )
        self._set_advanced_baseline()
        for _, w in self._advanced_fields:
            signal = (w.textChanged if isinstance(w, QLineEdit)
                      else w.toggled if isinstance(w, QCheckBox)
                      else w.currentIndexChanged if isinstance(w, QComboBox)
                      else w.valueChanged)
            signal.connect(self._update_advanced_badge)

        return section

    def _set_advanced_baseline(self):
        """Take the current Advanced values as the ones the badge compares to.

        Called after building and again once the saved settings are
        restored, so remembered values do not count as changed.
        """
        self._advanced_defaults = [
            (name, w, self._widget_value(w), self._widget_text(w))
            for name, w in self._advanced_fields
        ]
        self._update_advanced_badge()

    def _build_regularization_box(self) -> QWidget:
        """Build the fairness penalties (objective-based optimizers only)."""
        self._regularization_box = QWidget()
        layout = QVBoxLayout(self._regularization_box)
        layout.setContentsMargins(0, Size.GAP, 0, 0)
        layout.setSpacing(Size.GAP)

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
        _add_row(wf, "Bunching", self._spin_w_short)

        self._spin_w_tangent = _ws()
        self._spin_w_tangent.setToolTip("Penalize unusual edge lengths at curve endpoints")
        _add_row(wf, "End handles", self._spin_w_tangent)

        self._spin_w_curv = _ws()
        self._spin_w_curv.setToolTip(
            "Penalize bending energy: the integral of \u03BA\u00B2 along the curve")
        _add_row(wf, "Bending", self._spin_w_curv)

        self._spin_w_grad = _ws()
        self._spin_w_grad.setToolTip(
            "Penalize wiggles: the integral of (d\u03BA/ds)\u00B2 along the curve")
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

    @staticmethod
    def _widget_text(w) -> str:
        """Current value of an input widget as shown to the user."""
        if isinstance(w, QCheckBox):
            return "on" if w.isChecked() else "off"
        if isinstance(w, QComboBox):
            return w.currentText()
        return w.text().strip() or "(blank)"

    def _update_advanced_badge(self, *_):
        """Count the applicable Advanced settings that differ from their defaults."""
        content = self._advanced_section.content_widget()
        changed = [
            f"{name}: {default_text} \u2192 {self._widget_text(w)}"
            for name, w, default, default_text in self._advanced_defaults
            if w.isVisibleTo(content) and self._widget_value(w) != default
        ]
        self._advanced_section.set_badge(
            f"{len(changed)} changed" if changed else "",
            "Changed this session:\n" + "\n".join(changed) if changed else "",
        )

    # -- Run Bar -----------------------------------------------------------

    def _build_run_bar(self) -> QWidget:
        """Build the fixed bar under the sidebar: fit, result, and export."""
        bar = QWidget()
        bar.setObjectName("RunBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(Size.PAD, Size.PAD, Size.PAD, Size.PAD)
        layout.setSpacing(Size.PAD)

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
        result_layout.setSpacing(Size.PAD)

        stats = QGridLayout()
        stats.setHorizontalSpacing(Size.GROUP)
        stats.setVerticalSpacing(0)
        self._lbl_result_caption_dev = QLabel("Max deviation")
        self._lbl_result_dev = QLabel()
        self._lbl_result_ctrl = QLabel()
        self._lbl_result_time = QLabel()
        for readout in (self._lbl_result_dev, self._lbl_result_ctrl, self._lbl_result_time):
            readout.setObjectName("Readout")
        for col, (caption, value) in enumerate((
            (self._lbl_result_caption_dev, self._lbl_result_dev),
            (QLabel("Control points"), self._lbl_result_ctrl),
            (QLabel("Time"), self._lbl_result_time),
        )):
            caption.setObjectName("Muted")
            stats.addWidget(caption, 0, col)
            stats.addWidget(value, 1, col)
        result_layout.addLayout(stats)

        save_row = QHBoxLayout()
        save_row.setSpacing(Size.GAP)
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
