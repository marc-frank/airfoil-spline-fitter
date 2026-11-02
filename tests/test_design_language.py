"""The GUI takes all its styling from theme.py (docs/design-language.md).

Other GUI modules must not define colors or stylesheets of their own, so
new features cannot drift from the Loft design language.
"""
import io
import os
import re
import sys
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "src" / "spline_fitter" / "gui"

# What only theme.py may contain (checked in code and strings, not comments)
FORBIDDEN = {
    "hex color": re.compile(r"#[0-9A-Fa-f]{8}\b|#[0-9A-Fa-f]{6}\b|#[0-9A-Fa-f]{3}\b"),
    "stylesheet": re.compile(r"\bsetStyleSheet\b|\bsetDefaultStyleSheet\b"),
    "QColor": re.compile(r"\bQColor\b"),
    "Qt or pyqtgraph color": re.compile(
        r"\bGlobalColor\b|\b(?:mkColor|intColor|hsvColor)\b"
        r"|\b(?:mkPen|mkBrush|setBackground|setForeground)\s*\(\s*\d"),
    "named color": re.compile(
        r"\b(?:mkPen|mkBrush|setBackground|setForeground)\s*\(\s*[\"'][a-z]"
        r"|\b(?:color|pen|brush|symbolPen|symbolBrush|background|foreground)\s*=\s*[\"'][a-z]+[\"']"),
    "palette() color": re.compile(r"palette\(\w"),
    "RGB tuple": re.compile(r"\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*\)"),
    # Four numbers in a tuple, not a call's arguments (setContentsMargins(0, 0, 0, 0));
    # the tokens of _code_lines are joined by spaces
    "RGBA tuple": re.compile(r"(?<![\w)\]] )(?<![\w)\]])"
                             r"\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*\)"),
    "float RGB tuple": re.compile(
        r"\(\s*[01]?\.\d+\s*,\s*[01]?\.\d+\s*,\s*[01]?\.\d+\s*(?:,\s*[01]?\.\d+\s*)?\)"),
    "inline CSS color": re.compile(r"(?:background-)?color\s*:"),
    "emoji": re.compile(r"\\U0001[0-9a-fA-F]{4}|[\U0001F300-\U0001FAFF]"),
}


@pytest.mark.parametrize("code", [
    "(255, 110, 16, 128)", '"#FF6E10AA"', "Qt.GlobalColor.red", "pg.mkPen('r')",
    'pen = "orange"', "(1.0, 0.43, 0.06)", "pg.mkColor(3)", "pg.mkBrush(255, 0, 0)",
    "x = (255, 110, 16, 128)",
])
def test_forbidden_patterns_catch_every_color_form(code):
    assert any(p.search(code) for p in FORBIDDEN.values()), code


def test_forbidden_patterns_leave_margins_alone():
    assert not any(p.search("lay . setContentsMargins ( 0 , 0 , 0 , 0 )")
                   for p in FORBIDDEN.values())


def _code_lines(path: Path) -> dict[int, str]:
    """Each line's code and string tokens, without comments."""
    lines: dict[int, str] = {}
    source = path.read_text(encoding="utf-8")
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE):
            continue
        lines[tok.start[0]] = lines.get(tok.start[0], "") + " " + tok.string
    return lines


@pytest.mark.parametrize(
    "path", sorted(p for p in GUI.glob("*.py") if p.name != "theme.py"), ids=lambda p: p.name
)
def test_gui_module_takes_styling_from_theme(path):
    problems = [
        f"{path.name}:{n}: {name}: {code.strip()}"
        for n, code in _code_lines(path).items()
        for name, pattern in FORBIDDEN.items()
        if pattern.search(code)
    ]
    assert not problems, (
        "Styling belongs in gui/theme.py (see docs/design-language.md):\n" + "\n".join(problems)
    )


def test_documented_tokens_exist():
    """Every token the design doc names is defined in theme.py."""
    from spline_fitter.gui import theme

    doc = (ROOT / "docs" / "design-language.md").read_text(encoding="utf-8")
    defined = {name for name in vars(theme) if name.isupper()}
    for cls in (theme.Palette, theme.Ink, theme.Z, theme.Size, theme.Type, theme.Scheme, theme.Motion):
        defined |= {name for name in vars(cls) if name.isupper()}
    missing = {t for t in re.findall(r"`(?:[A-Za-z]+\.)?([A-Z][A-Z_]{2,})`", doc)} - defined
    assert not missing, f"documented but not in theme.py: {sorted(missing)}"


def test_theme_styles_the_window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui.theme import Palette
    from spline_fitter.gui.ui import SplineFitterWindow

    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    try:
        assert app.property("loftTheme")
        assert app.palette().window().color().name().lower() == Palette.WINDOW.lower()
        assert "#PrimaryButton" in app.styleSheet()
        assert w._btn_fit.objectName() == "PrimaryButton"
        assert w._plot.backgroundBrush().color().name().lower() == Palette.CANVAS.lower()
    finally:
        w.deleteLater()
        app.processEvents()


def test_inputs_and_buttons_share_one_height():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import (
        QAbstractButton,
        QAbstractSpinBox,
        QApplication,
        QCheckBox,
        QComboBox,
        QLineEdit,
    )

    from spline_fitter.gui.theme import Size
    from spline_fitter.gui.ui import SplineFitterWindow
    from spline_fitter.gui.widgets import MotionDialog, ThemeDialog

    def controls(root):
        return [
            c for c in root.findChildren((QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit))
            if c.isVisible() and not isinstance(c, QCheckBox)
            and c.objectName() != "SectionHeader"
            and not isinstance(c.parent(), (QAbstractSpinBox, QComboBox))
        ]

    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    dialogs = [ThemeDialog(w), MotionDialog(w)]
    try:
        w.resize(1500, 900)
        w.show()
        w._advanced_section.set_expanded(True)
        # The fields that only show while sweeping, and the drop-down panels
        for box in w._sweep_group:
            if box._btn_sweep.isEnabled():
                box._btn_sweep.click()
        w._btn_comb_settings.setChecked(True)
        w._btn_deviation_settings.setChecked(True)
        for dialog in dialogs:
            dialog.show()
        app.processEvents()
        shown = controls(w) + [c for d in dialogs for c in controls(d)]
        assert len(shown) > 30
        assert any(c.window() is w._comb_panel for c in shown)
        wrong = [(type(c).__name__, c.objectName(), c.height()) for c in shown
                 if c.height() != Size.CONTROL_HEIGHT]
        assert not wrong, f"controls not {Size.CONTROL_HEIGHT} px tall: {wrong}"
        icon_buttons = [c for c in shown if c.objectName() == "IconButton"]
        assert icon_buttons
        assert all(c.width() == Size.CONTROL_HEIGHT for c in icon_buttons), \
            [c.size() for c in icon_buttons]
    finally:
        for dialog in dialogs:
            dialog.deleteLater()
        w.deleteLater()
        app.processEvents()


def _contrast(fg: str, bg: str) -> float:
    """WCAG contrast ratio of two hex colors."""
    def luminance(color: str) -> float:
        channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    hi, lo = sorted((luminance(fg), luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.fixture
def scheme():
    """Restore the tokens of the default scheme after a test changed them."""
    from spline_fitter.gui import theme

    yield theme
    theme._set_tokens(theme.Scheme.DARK, theme.Scheme.TEMPERATURE)


@pytest.mark.parametrize("dark", [True, False], ids=["dark", "light"])
def test_neutral_temperature_has_no_tint(scheme, dark):
    scheme._set_tokens(dark, 0)
    for name in scheme.GREY_LIGHTNESS:
        color = getattr(scheme.Palette, name, None)
        if color:
            assert color[1:3] == color[3:5] == color[5:7], f"{name} {color} is tinted"


@pytest.mark.parametrize("dark", [True, False], ids=["dark", "light"])
def test_temperature_tints_warm_and_cool(scheme, dark):
    scheme._set_tokens(dark, 100)
    warm = scheme.rgb(scheme.Palette.PANEL)
    scheme._set_tokens(dark, -100)
    cool = scheme.rgb(scheme.Palette.PANEL)
    assert warm[0] > warm[2] and cool[2] > cool[0]


@pytest.mark.parametrize("dark", [True, False], ids=["dark", "light"])
@pytest.mark.parametrize("temperature", [-100, 0, 100])
def test_text_stays_readable(scheme, dark, temperature):
    scheme._set_tokens(dark, temperature)
    P = scheme.Palette
    for surface in (P.CANVAS, P.WINDOW, P.PANEL, P.FIELD):
        assert _contrast(P.TEXT, surface) >= 7
        assert _contrast(P.TEXT_2, surface) >= 4.5
        assert _contrast(P.TEXT_3, surface) >= 2.5


def test_theme_setting_restyles_the_running_app(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui import theme
    from spline_fitter.gui.ui import SplineFitterWindow

    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(theme, "_settings", lambda: QSettings(ini, QSettings.Format.IniFormat))
    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    browser = theme.text_browser()
    before = (theme.Scheme.MODE, theme.Scheme.TEMPERATURE)
    try:
        w._plot_manager.plot_airfoil([(1, 0), (0, 0), (1, 0.01)])
        w._run_details.set_rows("Title", [("Label", "1")])
        browser.setHtml("<p class='ok'>Fine</p>")
        theme.set_theme("light", -40)
        app.processEvents()
        assert not theme.Scheme.DARK
        assert app.palette().window().color().name().lower() == theme.Palette.WINDOW.lower()
        assert w._plot.backgroundBrush().color().name().lower() == theme.Palette.CANVAS.lower()
        data = w._plot_manager._data_item.opts["symbolBrush"].color()
        assert (data.red(), data.green(), data.blue()) == theme.Ink.DATA
        saved = QSettings(ini, QSettings.Format.IniFormat)
        assert saved.value("theme/mode") == "light"
        assert int(saved.value("theme/temperature")) == -40

        # Every plot item, axis and rich text in the new colors
        pm = w._plot_manager
        items = [(pm._data_item, pm._data_style()), (pm._curve_item, pm._curve_style()),
                 (pm._polygon_item, pm._polygon_style()), (pm._knot_item, pm._knot_style()),
                 (pm._tolerance_item, pm._tolerance_style()), (pm._loss_curve, pm._loss_style()),
                 (pm._comparison_curve, pm._comparison_style(pm._comparison_connected)),
                 (pm._comparison_marker, pm._marker_style())]
        items += [(item, pm._comb_style(*key)) for key, item in pm._comb_items.items()]
        items += [(item, pm._deviation_style(*key)) for key, item in pm._deviation_items.items()]
        stale = [(item, key) for item, style in items
                 for key in ("pen", "symbolBrush", "symbolPen") if style.get(key) is not None
                 and item.opts[key].color() != style[key].color()]
        assert not stale, stale
        grid = theme.Ink.GRID
        for plot in (w._plot, w._loss_plot, w._comparison_plot):
            for name in ("left", "bottom"):
                axis = plot.getPlotItem().getAxis(name)
                assert (axis.pen().color().red(), axis.pen().color().green(),
                        axis.pen().color().blue()) == grid
                assert axis.textPen().color().name().lower() == theme.Palette.TEXT_2.lower()
                assert axis.labelStyle["color"].lower() == theme.Palette.TEXT_2.lower()
        assert theme.Palette.TEXT.lower() in w._run_details._title.text().lower()
        assert theme.Palette.OK.lower() in browser.toHtml().lower()
    finally:
        theme.set_theme(*before)
        browser.deleteLater()
        w.deleteLater()
        app.processEvents()


def test_filter_preview_follows_a_theme_change(tmp_path, monkeypatch):
    """The values a typed condition rules out stay greyed in the new TEXT_3."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui import theme
    from spline_fitter.gui.explorer import COLUMN_NAMES
    from spline_fitter.gui.ui import SplineFitterWindow
    from spline_fitter.optimization.models import (
        FitResult,
        OptimizationConfig,
        SplineConfig,
    )

    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(theme, "_settings", lambda: QSettings(ini, QSettings.Format.IniFormat))
    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    before = (theme.Scheme.MODE, theme.Scheme.TEMPERATURE)
    try:
        w.show()
        w.new_naca_airfoil("2412")
        for ctrl in (8, 12):
            upper = [(1 - i / (ctrl - 1), 0.05 * (i % 2)) for i in range(ctrl)]
            w._project_manager.add_optimization_run(
                SplineConfig(num_control_points=ctrl),
                OptimizationConfig(max_iterations=0),
                FitResult([upper, upper[::-1]], 5, 1e-4, 3, 9, 0.5), [])
        popup = w._run_table.open_filter(COLUMN_NAMES.index("Control points"))
        popup.condition.setText("<10")
        greyed = [popup.values.item(i) for i in range(popup.values.count())
                  if popup.values.item(i).text() == "12"]
        assert greyed
        theme.set_theme("light", 60)
        app.processEvents()
        assert greyed[0].foreground().color().name().lower() == theme.Palette.TEXT_3.lower()
    finally:
        theme.set_theme(*before)
        w.deleteLater()
        app.processEvents()


def test_plots_start_as_their_toolbars_set_them():
    """The convergence plot's axes match the toolbar before the first fit,
    and every main-plot item has a drawing order from Z."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui.theme import Z
    from spline_fitter.gui.ui import SplineFitterWindow

    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    try:
        item = w._loss_plot.getPlotItem()
        log = w._combo_log_scale.currentData()
        assert item.ctrl.logXCheck.isChecked() == (log in ("x", "both"))
        assert item.ctrl.logYCheck.isChecked() == (log in ("y", "both"))
        assert item.getAxis("bottom").labelText == w._combo_loss_x.currentText()
        orders = {v for k, v in vars(Z).items() if k.isupper()}
        unordered = [i for i in w._plot.getPlotItem().listDataItems() if i.zValue() not in orders]
        assert not unordered
    finally:
        w.deleteLater()
        app.processEvents()


def test_window_narrows_by_stacking_rows():
    """The plot toolbar's groups and the convergence and runs plots stack
    when the window is too narrow for them side by side, so the center asks
    for no more width than its widest group."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from spline_fitter.gui.ui import SplineFitterWindow

    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    try:
        w.show()
        rows = (w._plot_toolbar, w._bottom_plots)
        widths = [[row.layout().itemAt(i).minimumSize().width() for i in range(2)]
                  for row in rows]
        for row, groups in zip(rows, widths, strict=True):
            assert row.minimumSizeHint().width() == max(groups)
        center = w._main_splitter.widget(1)
        margins = center.layout().contentsMargins()
        assert center.minimumSizeHint().width() == (
            max(max(groups) for groups in widths) + margins.left() + margins.right())
        w.resize(w.minimumSizeHint().width(), 900)
        app.processEvents()
        assert all(row.is_stacked() for row in rows)
        w.resize(4000, 900)
        app.processEvents()
        assert not any(row.is_stacked() for row in rows)
    finally:
        w.deleteLater()
        app.processEvents()


@pytest.mark.skipif(sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY"),
                    reason="needs the platform's own fonts")
def test_window_fits_half_a_full_hd_screen():
    """With the explorer shown, the window is at most 960 px wide at its
    narrowest. Measured with the platform's own fonts in a process of its
    own (the offscreen platform's stand-in fonts are wider); nothing shows."""
    import subprocess

    code = (
        "import os\n"
        "from PyQt6.QtWidgets import QApplication\n"
        "app = QApplication([])\n"
        "from spline_fitter.gui.ui import SplineFitterWindow\n"
        "w = SplineFitterWindow()\n"
        "w._action_toggle_explorer.setChecked(True)\n"
        "w.ensurePolished()\n"
        "print(w.minimumSizeHint().width(), w._explorer_panel.isVisibleTo(w), flush=True)\n"
        "os._exit(0)\n")
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "packages" / "bspline-varpro" / "src")])
    done = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    width, explorer = done.stdout.split()
    assert explorer == "True"
    assert int(width) <= 960


def test_ink_is_scarce():
    """Orange is spent on the fitted curve, the current-run marker and the
    primary button, of which there is one."""
    import ast

    from spline_fitter.gui import theme

    allowed = {("plotting.py", "_curve_style"), ("plotting.py", "_marker_style")}
    ink = {"INK", "INK_HOVER", "INK_PRESSED", "INK_DEEP", "ON_INK"}
    uses = []

    def visit(node, path, function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and ((node.value.id == "Palette" and node.attr in ink)
                     or (node.value.id == "Ink" and node.attr in ("CURVE", "MARKER")))
                and (path.name, function) not in allowed):
            uses.append(f"{path.name}:{node.lineno} {node.value.id}.{node.attr} in {function}")
        for child in ast.iter_child_nodes(node):
            visit(child, path, function)

    for path in GUI.glob("*.py"):
        if path.name != "theme.py":
            visit(ast.parse(path.read_text(encoding="utf-8-sig")), path, "<module>")
    assert not uses, uses
    P = theme.Palette
    for line in theme._qss().splitlines():
        if any(color in line for color in (P.INK, P.INK_HOVER, P.INK_PRESSED, P.ON_INK)):
            assert "#PrimaryButton" in line, line
    assert P.INK not in theme.html_css()


def test_one_primary_button():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QWidget

    from spline_fitter.gui.ui import SplineFitterWindow

    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    try:
        assert len(w.findChildren(QWidget, "PrimaryButton")) == 1
    finally:
        w.deleteLater()
        app.processEvents()


def test_sidebar_labels_have_a_role():
    """Every label in the sidebar is styled by object name: a form label, a
    caption, a muted note, a readout..."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QLabel, QWidget

    from spline_fitter.gui.ui import SplineFitterWindow

    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    try:
        unnamed = [label.text() for label in w.findChild(QWidget, "Sidebar").findChildren(QLabel)
                   if not label.objectName()]
        assert not unnamed
    finally:
        w.deleteLater()
        app.processEvents()


# -- Wording (section 3) --------------------------------------------------------

# Names that keep their capitals inside a sentence-case label
PROPER_NOUNS = {"Powell", "Nelder-Mead"}
# The product name, a title in its own right
PRODUCT = "Airfoil Spline Fitter"
# Shortened words and stand-ins for spelled-out words
ABBREVIATIONS = re.compile(
    r"\b(?:Avg|avg|Min|dev|pts|iters?|Iter|evals?|Evals|err|tol|sw|LE|TE|Ctrl pts)\b"
    r"|\b[a-z]+/[a-z]+\b(?!\))")
# Symbols beyond the arrow, the Greek letters of math and the times sign
SYMBOLS = re.compile("[˜✓✗≥≤∫…≈·•]")


def _decoded(code: str) -> str:
    """Code with its \\u, \\U and \\x escapes written as the characters they stand for."""
    return re.sub(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})|\\x([0-9a-fA-F]{2})",
                  lambda m: chr(int(next(g for g in m.groups() if g), 16)), code)


def _not_sentence_case(text: str) -> list[str]:
    """Words after the first that are capitalized without being an acronym or a name."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9+\-]*", text.replace("&", ""))
    return [w for w in words[1:]
            if w[0].isupper() and w[1:].islower() and w not in PROPER_NOUNS]


def _wording_problems(kind: str, text: str) -> list[str]:
    text = text.replace("&", "").replace(PRODUCT, "Product")
    problems = []
    if text.rstrip().endswith(":"):
        problems.append("trailing colon")
    if m := ABBREVIATIONS.search(text):
        problems.append(f"abbreviation {m.group()!r}")
    if m := SYMBOLS.search(text):
        problems.append(f"symbol {m.group()!r}")
    if kind != "tooltip" and (words := _not_sentence_case(text)):
        problems.append(f"not sentence case: {words}")
    return [f"{kind} {text!r}: {p}" for p in problems]


def test_window_wording():
    """Every label, button, item, menu entry and tooltip of the window."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import (
        QAbstractButton,
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QLabel,
        QLineEdit,
        QSpinBox,
        QWidget,
    )

    from spline_fitter.gui.ui import SplineFitterWindow

    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    texts: list[tuple[str, str]] = [("title", w.windowTitle())]
    try:
        for widget in w.findChildren(QWidget):
            if isinstance(widget, QLabel) and widget.objectName() == "SectionCaption":
                assert widget.text().isupper(), f"caption {widget.text()!r} is not uppercase"
            elif isinstance(widget, QLabel) and not widget.text().startswith("<"):
                texts.append(("label", widget.text()))
            elif isinstance(widget, QAbstractButton) and widget.objectName() != "SectionHeader":
                texts.append(("button", widget.text()))
            elif isinstance(widget, QComboBox):
                for i in range(widget.count()):
                    if widget.model().item(i).isEnabled():
                        texts.append(("item", widget.itemText(i)))
                    else:  # a caption grouping the items
                        assert widget.itemText(i).isupper() or not widget.itemText(i)
            elif isinstance(widget, QLineEdit):
                texts.append(("placeholder", widget.placeholderText()))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                texts.append(("suffix", widget.suffix().strip()))
            if widget.toolTip():
                texts.append(("tooltip", widget.toolTip()))

        def walk(menu):
            for action in menu.actions():
                texts.append(("menu", action.text()))
                if action.menu():
                    walk(action.menu())
        walk(w.menuBar())
        # Axis labels and the status bar
        for plot in (w._plot, w._loss_plot, w._comparison_plot):
            for name in ("left", "bottom"):
                texts.append(("label", plot.getPlotItem().getAxis(name).labelText))
        texts.append(("tooltip", w.statusBar().currentMessage()))  # a sentence: no case check

        problems = [p for kind, text in texts if text for p in _wording_problems(kind, text)]
        assert not problems, "Wording (docs/design-language.md, section 3):\n" + "\n".join(problems)
    finally:
        w.deleteLater()
        app.processEvents()


def test_short_choices_are_segmented_controls():
    """In the sidebar, two to four short options are a SegmentedControl, not a
    combo box (the plot toolbars keep compact combo boxes)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QComboBox, QWidget

    from spline_fitter.gui.ui import SplineFitterWindow

    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    try:
        short = [
            [c.itemText(i) for i in range(c.count())]
            for c in w.findChild(QWidget, "Sidebar").findChildren(QComboBox)
            if 2 <= c.count() <= 4 and all(len(c.itemText(i)) <= 12 for i in range(c.count()))
        ]
        assert not short, f"use a SegmentedControl for {short}"
    finally:
        w.deleteLater()
        app.processEvents()


# A string literal, in double or single quotes: its text is group "text"
_STRING = r"f?(?P<q>[\"'])(?P<text>(?:(?!(?P=q)).)+)(?P=q)"
# Calls whose string argument is shown as a label, button, title, menu entry,
# axis label, placeholder, suffix or message box title
_LABEL_CALLS = re.compile(
    r"\b(?:setWindowTitle|addAction|addMenu|QLabel|QPushButton|QCheckBox|QToolButton|"
    r"_add_row\(\w+,|add_form_row\(\w+,|setText|addItem|getOpenFileName\(\s*[\w.]+,|"
    r"getSaveFileName\(\s*[\w.]+,|getExistingDirectory\(\s*\w+,|"
    r"label_plot\(\s*[\w.]+,\s*[\"']\w+[\"'],|setLabel\(\s*[\"']\w+[\"'],|set_loss_x_label|"
    r"setPlaceholderText|setSuffix|setSpecialValueText|setItemText\(\s*[\w.]+,|"
    r"QMessageBox\.\w+\(\s*[\w.]+,)\(?\s*" + _STRING)
_ITEM_LISTS = re.compile(r"\baddItems\(\s*\[([^\]]*)\]")
_MESSAGES = re.compile(r"\bshowMessage\(\s*" + _STRING)
_TOOLTIPS = re.compile(r"\b(?:setToolTip|setItemToolTip)\([^\"']*?" + _STRING)


@pytest.mark.parametrize(
    "path", sorted(p for p in GUI.glob("*.py") if p.name != "theme.py"), ids=lambda p: p.name
)
def test_source_wording(path):
    """Labels in dialogs the window tests cannot open, and status messages."""
    source = path.read_text(encoding="utf-8")
    problems = []
    for m in _LABEL_CALLS.finditer(source):
        problems += _wording_problems("label", m.group("text").strip())
    for m in _ITEM_LISTS.finditer(source):
        for item in re.finditer(_STRING, m.group(1)):
            problems += _wording_problems("item", item.group("text"))
    for m in _MESSAGES.finditer(source):
        problems += _wording_problems("tooltip", m.group("text"))  # sentences: no case check
    for m in _TOOLTIPS.finditer(source):
        problems += _wording_problems("tooltip", m.group("text"))
    for n, code in _code_lines(path).items():
        if m := SYMBOLS.search(_decoded(code)):
            problems.append(f"line {n}: symbol {m.group()!r}")
    assert not problems, f"Wording in {path.name}:\n" + "\n".join(sorted(set(problems)))


def test_decimals_follow_the_locale():
    """Shown numbers take the locale's decimal point; files keep a dot."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QLocale
    from PyQt6.QtWidgets import QApplication

    from spline_fitter import formatting
    from spline_fitter.gui import theme
    from spline_fitter.gui.widgets import ScientificSpinBox

    app = QApplication.instance() or QApplication([])  # noqa: F841
    before = formatting.decimal_point()
    try:
        formatting.set_decimal_point(",")
        assert formatting.format_error(1.234e-5) == "1,234e-05"
        assert formatting.format_duration(12.34) == "12,34 s"
        assert formatting.number(0.5, ".2f") == "0,50"
        assert theme._DecadeAxis("bottom").tickStrings([0.5, 1.0], 1, 0.5) == ["0,5", "1,0"]

        spin = ScientificSpinBox()
        spin.setLocale(QLocale(QLocale.Language.German, QLocale.Country.Germany))
        spin.setRange(0, 1)
        spin.setValue(0.001)
        assert spin.text() == "0,001"
        assert spin.valueFromText("2,5e-05") == 2.5e-05
    finally:
        formatting.set_decimal_point(before)
