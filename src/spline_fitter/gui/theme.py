"""
Theme: the Loft design language
================================

The one place the GUI's colors, fonts, sizes and widget styles are defined.
See ``docs/design-language.md`` for the rules behind them; every other GUI
module takes its styling from here (``tests/test_design_language.py``
checks that).

Loft is named after the lofting floor, where aircraft and ship lines were
drawn full size with flexible splines held by lead weights ("ducks"). The
window is the floor at night: dark graphite chrome around a blueprint
canvas, construction lines in pencil grey, and one ink color, the orange
of the app icon, for the curve and the one action that makes it.

This module provides:
    - Palette: UI color tokens
    - Ink: plot color tokens, by what is drawn
    - Z: drawing order of plot items
    - apply_theme / style_plot / app_icon / pen helpers
"""

from __future__ import annotations

import os

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QPalette
from PyQt6.QtWidgets import QApplication, QTextBrowser

ASSETS = os.path.join(os.path.dirname(__file__), "assets")


# =============================================================================
# Tokens
# =============================================================================

class Palette:
    """UI colors. Surfaces step up in lightness the closer they are to the hand."""

    # Surfaces, darkest (furthest back) first
    CANVAS = "#0B131A"        # plot area: the blueprint
    WINDOW = "#11171E"        # window, toolbars, status bar
    PANEL = "#151C24"         # sidebar and explorer
    FIELD = "#1B242E"         # inputs, secondary buttons
    FIELD_HOVER = "#222D39"
    RULE = "#283440"          # hairlines and borders
    RULE_STRONG = "#3A4856"   # borders of hovered fields

    # Text
    TEXT = "#E3E9EF"          # chalk: values and labels
    TEXT_2 = "#9AA7B4"        # secondary: field labels, axis text
    TEXT_3 = "#667585"        # muted: captions, hints, disabled

    # Ink: the icon's orange, for the fitted curve and the primary action only
    INK = "#FF6E10"
    INK_HOVER = "#FF8636"
    INK_PRESSED = "#E05E08"
    INK_DEEP = "#AD4200"      # the icon's shadowed underside
    ON_INK = "#1A0C02"        # text on ink

    # Selection: checked toggles, selected rows, focus
    SELECT = "#21496A"
    SELECT_LINE = "#4C9BD0"

    # Status
    OK = "#4FBF85"
    WARN = "#F2B84B"
    ERROR = "#F0605D"


def rgb(hex_color: str) -> tuple[int, int, int]:
    """A token color as an RGB tuple."""
    c = QColor(hex_color)
    return (c.red(), c.green(), c.blue())


class Ink:
    """Plot colors, named by what they draw (RGB tuples for pyqtgraph)."""

    # The airfoil: measured stations in chalk, the fitted batten in ink
    DATA = rgb(Palette.TEXT)
    CURVE = rgb(Palette.INK)
    # Construction: the control polygon, pencil grey and dashed
    CONSTRUCTION = rgb("#7D8C9B")
    # Knots: the ducks that hold the batten
    KNOTS = rgb("#F2C94C")
    # Curvature comb, by surface as on a pressure plot: suction side cool,
    # pressure side warm (but never the curve's orange)
    COMB_UPPER = rgb("#49B6E8")
    COMB_LOWER = rgb("#E5689A")
    COMB_TEETH_ALPHA = 90      # teeth are faint; the envelope through the tips is the reading
    # Convergence and comparison traces; the current run is marked in ink
    TRACE = rgb("#8FB3CF")
    MARKER = rgb(Palette.INK)
    # Blueprint grid
    GRID = rgb(Palette.SELECT_LINE)
    GRID_ALPHA = 0.13


class Z:
    """Drawing order of plot items. The curve covers the data it fits, so a
    data point that shows beside it is a deviation worth seeing."""

    COMB = -10
    CONSTRUCTION = 0
    DATA = 5
    CURVE = 10
    KNOTS = 30


class Size:
    """Spacing, radii and fixed widths. Use the named gaps, not numbers."""

    TIGHT = 4                 # parts of one composite field (spin box + sweep toggle)
    GAP = 6                   # form rows; fields side by side; toolbar items
    GROUP = 8                 # between groups: label above content, button rows
    PAD = 12                  # panel and section padding
    PAD_LARGE = 16            # dialogs and documents
    RADIUS = 3
    LABEL_WIDTH = 92          # label column of every sidebar form
    CURVE_WIDTH = 2
    LINE_WIDTH = 1
    DATA_SIZE = 4
    CONTROL_SIZE = 6
    KNOT_SIZE = 8
    TRACE_POINT = 4
    MARKER_SIZE = 13


class Type:
    """Fonts. UI text uses the system face; numbers use a monospace face."""

    MONO = "Cascadia Mono, Consolas, Menlo, monospace"
    MONO_FAMILIES = ["Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono"]
    BASE_PT = 9
    CAPTION_PX = 10
    READOUT_PX = 15


# =============================================================================
# Application style
# =============================================================================

def _qss() -> str:
    P, S = Palette, Size
    return f"""
QWidget {{ color: {P.TEXT}; }}
QMainWindow, QDialog {{ background: {P.WINDOW}; }}
QToolTip {{ background: {P.FIELD}; color: {P.TEXT}; border: 1px solid {P.RULE_STRONG};
            padding: 4px 6px; }}

/* Menus */
QMenuBar {{ background: {P.WINDOW}; border-bottom: 1px solid {P.RULE}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {P.FIELD_HOVER}; }}
QMenu {{ background: {P.FIELD}; border: 1px solid {P.RULE_STRONG}; padding: 4px 0; }}
QMenu::item {{ padding: 5px 24px 5px 20px; }}
QMenu::item:selected {{ background: {P.SELECT}; }}
QMenu::item:disabled {{ color: {P.TEXT_3}; }}
QMenu::separator {{ height: 1px; background: {P.RULE}; margin: 4px 0; }}

/* Status bar */
QStatusBar {{ background: {P.WINDOW}; color: {P.TEXT_2}; border-top: 1px solid {P.RULE}; }}
QStatusBar::item {{ border: none; }}

/* Splitters and scroll bars: hairlines */
QSplitter::handle {{ background: {P.RULE}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {P.RULE_STRONG}; border-radius: 3px; min-height: 24px; min-width: 24px;
                      margin: 2px; }}
QScrollBar::handle:hover {{ background: {P.TEXT_3}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* Buttons: secondary by default */
QPushButton, QToolButton {{ background: {P.FIELD}; border: 1px solid {P.RULE};
    border-radius: {S.RADIUS}px; padding: 4px 10px; }}
QPushButton:hover, QToolButton:hover {{ background: {P.FIELD_HOVER}; border-color: {P.RULE_STRONG}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {P.RULE}; }}
QPushButton:checked, QToolButton:checked {{ background: {P.SELECT}; border-color: {P.SELECT_LINE}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {P.TEXT_3}; background: {P.PANEL};
    border-color: {P.RULE}; }}
QPushButton:focus, QToolButton:focus {{ outline: none; }}
QPushButton:default {{ border-color: {P.SELECT_LINE}; }}
QToolButton[popupMode="1"] {{ padding-right: 18px; }}
QToolButton#ArrowButton {{ padding: 4px 3px; border-top-left-radius: 0; border-bottom-left-radius: 0;
    margin-left: -1px; }}
QToolButton#SplitToggle {{ border-top-right-radius: 0; border-bottom-right-radius: 0; }}

/* The primary action: ink, once per view */
QPushButton#PrimaryButton {{ background: {P.INK}; color: {P.ON_INK}; border: none;
    padding: 8px; font-weight: 600; }}
QPushButton#PrimaryButton:hover {{ background: {P.INK_HOVER}; }}
QPushButton#PrimaryButton:pressed {{ background: {P.INK_PRESSED}; }}
QPushButton#PrimaryButton:disabled {{ background: {P.RULE}; color: {P.TEXT_3}; }}

/* Icon-only buttons */
QPushButton#IconButton {{ background: transparent; border: none; color: {P.TEXT_3};
    padding: 2px 6px; }}
QPushButton#IconButton:hover {{ color: {P.TEXT}; background: {P.FIELD_HOVER}; }}

/* Fields */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {P.FIELD}; border: 1px solid {P.RULE};
    border-radius: {S.RADIUS}px; padding: 3px 6px; selection-background-color: {P.SELECT}; }}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border-color: {P.RULE_STRONG}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {P.SELECT_LINE}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {P.TEXT_3}; background: {P.PANEL}; }}
QSpinBox, QDoubleSpinBox {{ font-family: {Type.MONO}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 16px; border: none; background: transparent; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {P.FIELD_HOVER}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({_asset_url("chevron_up.svg")}); width: 8px; height: 8px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({_asset_url("chevron_down.svg")}); width: 8px; height: 8px; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: none; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{ image: url({_asset_url("chevron_down.svg")}); width: 8px; height: 8px; }}
QComboBox QAbstractItemView {{ background: {P.FIELD}; border: 1px solid {P.RULE_STRONG};
    selection-background-color: {P.SELECT}; outline: none; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox:disabled {{ color: {P.TEXT_3}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {P.RULE_STRONG};
    border-radius: 2px; background: {P.FIELD}; }}
QCheckBox::indicator:hover {{ border-color: {P.SELECT_LINE}; }}
QCheckBox::indicator:checked {{ background: {P.SELECT}; border-color: {P.SELECT_LINE};
    image: url({_asset_url("check.svg")}); }}
QCheckBox::indicator:disabled {{ background: {P.PANEL}; border-color: {P.RULE}; }}

/* Lists and text */
QTreeView, QListView, QTextBrowser {{ background: {P.PANEL}; border: 1px solid {P.RULE};
    border-radius: {S.RADIUS}px; outline: none; }}
QTreeView::item {{ padding: 2px 0; }}
QTreeView::item:hover {{ background: {P.FIELD}; }}
QTreeView::item:selected {{ background: {P.SELECT}; color: {P.TEXT}; }}
QHeaderView::section {{ background: {P.PANEL}; color: {P.TEXT_3}; border: none;
    border-bottom: 1px solid {P.RULE}; padding: 4px 6px; }}

/* Plots */
PlotWidget, QGraphicsView {{ border: 1px solid {P.RULE}; border-radius: {S.RADIUS}px; }}

/* Panels: sidebar, explorer, drop-downs */
#Sidebar, #Explorer {{ background: {P.PANEL}; }}
#SidebarSection {{ border-bottom: 1px solid {P.RULE}; }}
#SectionCaption, #SectionHeader {{ color: {P.TEXT_3}; font-size: {Type.CAPTION_PX}px;
    font-weight: 600; letter-spacing: 1px; }}
#SectionHeader {{ text-align: left; border: none; padding: 0; background: transparent; }}
#SectionHeader:hover {{ color: {P.TEXT}; background: transparent; }}
#SectionBadge {{ color: {P.SELECT_LINE}; font-size: {Type.CAPTION_PX}px; }}
#FileName {{ font-weight: 600; }}
#Muted {{ color: {P.TEXT_3}; font-size: {Type.CAPTION_PX + 1}px; }}
#Readout {{ font-family: {Type.MONO}; font-size: {Type.READOUT_PX}px; color: {P.TEXT}; }}
#RunBar {{ background: {P.WINDOW}; border-top: 1px solid {P.RULE}; }}
#Toolbar QLabel {{ color: {P.TEXT_2}; }}
#DropDownPanel {{ background: {P.FIELD}; border: 1px solid {P.RULE_STRONG};
    border-radius: {S.RADIUS}px; }}
#FormLabel {{ color: {P.TEXT_2}; }}

/* Segmented control: joined buttons, one checked */
QPushButton#Segment {{ border-radius: 0; margin-left: -1px; padding: 4px 6px; }}
QPushButton#Segment[position="first"] {{ border-top-left-radius: {S.RADIUS}px;
    border-bottom-left-radius: {S.RADIUS}px; margin-left: 0; }}
QPushButton#Segment[position="last"] {{ border-top-right-radius: {S.RADIUS}px;
    border-bottom-right-radius: {S.RADIUS}px; }}
QPushButton#Segment:checked {{ color: {P.TEXT}; }}
QPushButton#SweepToggle {{ padding: 0; font-size: 13px; color: {P.TEXT_2}; }}
QPushButton#SweepToggle:checked {{ color: {P.TEXT}; }}
"""


def _asset_url(name: str) -> str:
    return os.path.join(ASSETS, name).replace("\\", "/")


def _palette() -> QPalette:
    """The Qt palette, so widgets drawn by the style itself match the tokens."""
    P = Palette
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: P.WINDOW,
        QPalette.ColorRole.WindowText: P.TEXT,
        QPalette.ColorRole.Base: P.FIELD,
        QPalette.ColorRole.AlternateBase: P.PANEL,
        QPalette.ColorRole.Text: P.TEXT,
        QPalette.ColorRole.Button: P.FIELD,
        QPalette.ColorRole.ButtonText: P.TEXT,
        QPalette.ColorRole.Highlight: P.SELECT,
        QPalette.ColorRole.HighlightedText: P.TEXT,
        QPalette.ColorRole.ToolTipBase: P.FIELD,
        QPalette.ColorRole.ToolTipText: P.TEXT,
        QPalette.ColorRole.PlaceholderText: P.TEXT_3,
        QPalette.ColorRole.Link: P.SELECT_LINE,
        QPalette.ColorRole.Mid: P.RULE,
        QPalette.ColorRole.Midlight: P.RULE,
        QPalette.ColorRole.Dark: P.WINDOW,
        QPalette.ColorRole.Light: P.RULE_STRONG,
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(P.TEXT_3))
    return pal


def apply_theme(app: QApplication | None = None) -> None:
    """Style the application. Call before building any widget; safe to repeat."""
    app = app or QApplication.instance()
    if app is None or app.property("loftTheme"):
        return
    app.setProperty("loftTheme", True)
    app.setStyle("Fusion")
    app.setPalette(_palette())
    font = app.font()
    font.setPointSize(Type.BASE_PT)
    app.setFont(font)
    app.setStyleSheet(_qss())
    app.setWindowIcon(app_icon())
    pg.setConfigOptions(background=Palette.CANVAS, foreground=Palette.TEXT_2, antialias=True)


def app_icon() -> QIcon:
    """The application icon: the orange airfoil."""
    icon = QIcon()
    for name in ("app_icon.ico", "app_icon.png"):
        path = os.path.join(ASSETS, name)
        if os.path.exists(path):
            icon.addFile(path)
    return icon


def mono_font(point_size: float | None = None) -> QFont:
    """The monospace face for numbers."""
    font = QFont()
    font.setFamilies(Type.MONO_FAMILIES)
    font.setStyleHint(QFont.StyleHint.Monospace)
    if point_size:
        font.setPointSizeF(point_size)
    return font


# =============================================================================
# Plots
# =============================================================================

def style_plot(plot: pg.PlotWidget) -> None:
    """Blueprint styling for a plot: grid, axes and labels."""
    item = plot.getPlotItem()
    axis_pen = pg.mkPen(color=Ink.GRID, width=1)
    tick_font = QFont()
    tick_font.setPointSizeF(Type.BASE_PT - 1)
    for name in ("left", "bottom", "right", "top"):
        axis = item.getAxis(name)
        axis.setPen(axis_pen)
        axis.setTextPen(pg.mkPen(color=Palette.TEXT_2))
        axis.setTickFont(tick_font)
        axis.label.setDefaultTextColor(QColor(Palette.TEXT_2))
    plot.showGrid(x=True, y=True, alpha=Ink.GRID_ALPHA)
    plot.setBackground(Palette.CANVAS)


def label_plot(plot: pg.PlotWidget, side: str, text: str) -> None:
    """Set an axis label in the theme's label style."""
    plot.setLabel(side, text, color=Palette.TEXT_2, **{"font-size": f"{Type.BASE_PT}pt"})


def _qcolor(color, alpha: int = 255) -> QColor:
    """A token color (hex string or RGB tuple) as a QColor."""
    c = QColor(*color) if isinstance(color, tuple) else QColor(color)
    c.setAlpha(alpha)
    return c


def pen(color, width: float = Size.LINE_WIDTH, dashed: bool = False, alpha: int = 255):
    """A cosmetic pen of a token color."""
    style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
    return pg.mkPen(color=_qcolor(color, alpha), width=width, style=style)


def brush(color, alpha: int = 255):
    """A brush of a token color."""
    return pg.mkBrush(_qcolor(color, alpha))


# =============================================================================
# Rich text
# =============================================================================

HTML_CSS = f"""
body {{ color: {Palette.TEXT}; line-height: 1.45; }}
h1, h2, h3, h4 {{ color: {Palette.TEXT}; font-weight: 600; }}
td, th {{ padding: 3px 12px 3px 0; }}
th {{ color: {Palette.TEXT_3}; font-weight: 600; text-align: left; border-bottom: 1px solid {Palette.RULE}; }}
pre, code {{ font-family: {Type.MONO}; background-color: {Palette.FIELD}; }}
a {{ color: {Palette.SELECT_LINE}; }}
b {{ color: {Palette.TEXT}; }}
.muted {{ color: {Palette.TEXT_3}; }}
.ok {{ color: {Palette.OK}; }}
.warn {{ color: {Palette.WARN}; }}
.error {{ color: {Palette.ERROR}; }}
"""


def text_browser() -> QTextBrowser:
    """A text browser whose HTML is styled with the theme."""
    browser = QTextBrowser()
    browser.document().setDefaultStyleSheet(HTML_CSS)
    browser.document().setDocumentMargin(Size.PAD_LARGE)
    return browser
