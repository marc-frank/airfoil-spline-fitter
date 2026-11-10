"""Blind reimplementation of the theme (spec.md 19.19, 19.21 item 12, Appendix A)."""
from __future__ import annotations

import os
import re
import tempfile

from blind_impl import grey

# 19.19.1 lightness (dark, light)
GREY_LIGHTNESS = {
    "CANVAS": (0.213, 0.995),
    "WINDOW": (0.236, 0.970),
    "PANEL": (0.268, 0.950),
    "FIELD": (0.309, 1.000),
    "FIELD_HOVER": (0.344, 0.965),
    "RULE": (0.360, 0.870),
    "RULE_STRONG": (0.434, 0.780),
    "TEXT": (0.982, 0.250),
    "TEXT_2": (0.807, 0.430),
    "TEXT_3": (0.686, 0.580),
    "SELECT": (0.405, 0.895),
    "SELECT_LINE": (0.609, 0.600),
    "CONSTRUCTION": (0.639, 0.620),
}
MODE_COLOURS = {
    "OK": ("#4FBF85", "#1A7A4A"),
    "WARN": ("#F2B84B", "#8F5E00"),
    "ERROR": ("#F0605D", "#C02F2B"),
    "KNOTS": ("#F2C94C", "#C99A06"),
    "COMB_EVEN": ("#49B6E8", "#1E88C7"),
    "COMB_ODD": ("#E5689A", "#CC3F78"),
}
FIXED_INK = {
    "INK": "#FF6E10",
    "INK_HOVER": "#FF8636",
    "INK_PRESSED": "#E05E08",
    "INK_DEEP": "#AD4200",
    "ON_INK": "#1A0C02",
}
SLIDER_GRADIENT = "stop:0 #6288BA, stop:0.5 #868686, stop:1 #968645"
MONO = "Cascadia Mono, Consolas, Menlo, monospace"

GLYPHS = {
    "chevron_down": "M2.5 4.5 L6 8 L9.5 4.5",
    "chevron_up": "M2.5 7.5 L6 4 L9.5 7.5",
    "chevron_right": "M4.5 2.5 L8 6 L4.5 9.5",
    "chevron_left": "M7.5 2.5 L4 6 L7.5 9.5",
    "check": "M2.5 6.2 L5 8.6 L9.5 3.6",
    "dash": "M3 6 L9 6",
    "sort_up": "M6 10 L6 2 M3 5 L6 2 L9 5",
    "sort_down": "M6 2 L6 10 M3 7 L6 10 L9 7",
    "filter": "M1.5 2.5 L10.5 2.5 L7 6.5 L7 10 L5 9 L5 6.5 L1.5 2.5",
    "range": "M1 3 L1 9 M11 3 L11 9 M3 6 L9 6 M4.6 4.4 L3 6 L4.6 7.6 M7.4 4.4 L9 6 L7.4 7.6",
}


def palette(dark, temperature):
    """All Palette tokens (19.19.1) as '#RRGGBB'."""
    T = max(-100, min(100, int(temperature))) / 100
    # SPEC-GAP: 19.19.1 temperature is clamped to -100..100 when read from QSettings
    # (19.19.2); the same clamp is applied to the argument here.
    i = 0 if dark else 1
    tok = {name: grey(L[i], T) for name, L in GREY_LIGHTNESS.items()}
    for name, pair in MODE_COLOURS.items():
        tok[name] = pair[i]
    tok.update(FIXED_INK)
    return tok


def glyph_svg(name, colour_hex, pad=0):
    pad = int(pad)
    size = 12 + 2 * pad
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-pad} {-pad} {size} {size}">'
        f'<path d="{GLYPHS[name]}" fill="none" stroke="{colour_hex}" stroke-width="1.4" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def glyph_path(name, colour_hex, pad=0):
    folder = os.path.join(tempfile.gettempdir(), "spline_fitter_glyphs")
    path = os.path.join(folder, f"{name}_{colour_hex.lstrip('#').lower()}_{int(pad)}.svg")
    # SPEC-GAP: 19.21 item 12 the application writes the file if missing; this blind
    # function only computes the path (the test rules forbid writing outside the blind
    # folder).
    return path.replace("\\", "/")


TEMPLATE = r"""QWidget { color: {TEXT}; }
QMainWindow, QDialog { background: {WINDOW}; }
QToolTip { background: {FIELD}; color: {TEXT}; border: 1px solid {RULE_STRONG}; padding: 4px 6px; }
QMenuBar { background: {WINDOW}; border-bottom: 1px solid {RULE}; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected { background: {FIELD_HOVER}; }
QMenu { background: {FIELD}; border: 1px solid {RULE_STRONG}; padding: 4px 0; }
QMenu::item { padding: 5px 24px 5px 20px; }
QMenu::item:selected { background: {SELECT}; }
QMenu::item:disabled { color: {TEXT_3}; }
QMenu::separator { height: 1px; background: {RULE}; margin: 4px 0; }
QStatusBar { background: {WINDOW}; color: {TEXT_2}; border-top: 1px solid {RULE}; }
QStatusBar::item { border: none; }
QSplitter::handle { background: {RULE}; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle { background: {RULE_STRONG}; border-radius: 3px; min-height: 24px; min-width: 24px; margin: 2px; }
QScrollBar::handle:hover { background: {TEXT_3}; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QPushButton, QToolButton { background: {FIELD}; border: 1px solid {RULE}; border-radius: 3px; padding: 0 10px; {FIXED} }
QPushButton:hover, QToolButton:hover { background: {FIELD_HOVER}; border-color: {RULE_STRONG}; }
QPushButton:pressed, QToolButton:pressed { background: {RULE}; }
QPushButton:checked, QToolButton:checked { background: {SELECT}; border-color: {SELECT_LINE}; }
QPushButton:disabled, QToolButton:disabled { color: {TEXT_3}; background: {PANEL}; border-color: {RULE}; }
QPushButton:focus, QToolButton:focus { outline: none; }
QPushButton:default { border-color: {SELECT_LINE}; }
QToolButton#ArrowButton { padding: 0 5px; border-top-left-radius: 0; border-bottom-left-radius: 0; margin-left: -1px; }
QToolButton#SplitToggle { border-top-right-radius: 0; border-bottom-right-radius: 0; }
QLineEdit QToolButton { background: transparent; border: none; padding: 0; min-height: 0; max-height: 16777215px; }
QPushButton#PrimaryButton { background: {INK}; color: {ON_INK}; border-color: {INK}; font-weight: 600; }
QPushButton#PrimaryButton:hover { background: {INK_HOVER}; border-color: {INK_HOVER}; }
QPushButton#PrimaryButton:pressed { background: {INK_PRESSED}; border-color: {INK_PRESSED}; }
QPushButton#PrimaryButton:disabled { background: {RULE}; border-color: {RULE}; color: {TEXT_3}; }
QPushButton#IconButton { background: transparent; border-color: transparent; padding: 0; min-width: {H}px; max-width: {H}px; }
QPushButton#IconButton:hover { background: {FIELD_HOVER}; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: {FIELD}; border: 1px solid {RULE}; border-radius: 3px; padding: 0 6px; {FIXED} selection-background-color: {SELECT}; }
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover { border-color: {RULE_STRONG}; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: {SELECT_LINE}; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { color: {TEXT_3}; background: {PANEL}; }
QSpinBox, QDoubleSpinBox { font-family: {MONO}; }
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button { width: 16px; border: none; background: transparent; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: {FIELD_HOVER}; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url({glyph:chevron_up}); width: 9px; height: 9px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url({glyph:chevron_down}); width: 9px; height: 9px; }
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled, QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled { image: none; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { image: url({glyph:chevron_down}); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: {FIELD}; border: 1px solid {RULE_STRONG}; selection-background-color: {SELECT}; outline: none; }
QCheckBox { spacing: 6px; }
QCheckBox:disabled { color: {TEXT_3}; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid {RULE_STRONG}; border-radius: 2px; background: {FIELD}; }
QCheckBox::indicator:hover { border-color: {SELECT_LINE}; }
QCheckBox::indicator:checked { background: {SELECT}; border-color: {SELECT_LINE}; image: url({glyph:check:TEXT}); }
QCheckBox::indicator:indeterminate { background: {SELECT}; border-color: {SELECT_LINE}; image: url({glyph:dash:TEXT}); }
QCheckBox::indicator:disabled { background: {PANEL}; border-color: {RULE}; }
QListView::indicator { width: 14px; height: 14px; border: 1px solid {RULE_STRONG}; border-radius: 2px; background: {FIELD}; }
QListView::indicator:checked { background: {SELECT}; border-color: {SELECT_LINE}; image: url({glyph:check:TEXT}); }
QListView::indicator:disabled { background: {PANEL}; border-color: {RULE}; }
QListView::item { padding: 2px 0; }
QListView::item:disabled { color: {TEXT_3}; }
QListView::item:hover { background: {FIELD_HOVER}; }
QSlider { {FIXED} }
QSlider::groove:horizontal { height: 4px; background: {RULE}; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 7px; background: {TEXT_2}; border: 1px solid {CANVAS}; }
QSlider::handle:horizontal:hover { background: {TEXT}; }
QSlider#TemperatureSlider::groove:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {SLIDER_GRADIENT}); }
QTreeView, QListView, QTableView, QTextBrowser { background: {PANEL}; border: 1px solid {RULE}; border-radius: 3px; outline: none; }
QTreeView::item { padding: 2px 0; }
QTableView::item { padding: 0 6px; }
QTreeView::item:hover { background: {FIELD}; }
QTreeView::item:selected, QTableView::item:selected { background: {SELECT}; color: {TEXT}; }
QTreeView::branch { background: transparent; }
QTreeView::branch:has-children:closed { image: url({glyph:chevron_right:TEXT_2:6}); }
QTreeView::branch:has-children:open { image: url({glyph:chevron_down:TEXT_2:6}); }
QHeaderView { background: {PANEL}; border: none; border-bottom: 1px solid {RULE}; }
QHeaderView::section { background: {PANEL}; color: {TEXT_3}; border: none; padding: 4px 6px; }
QHeaderView:vertical { border-bottom: none; border-right: 1px solid {RULE}; }
QHeaderView#FilterHeader::section { padding: 4px 36px 4px 6px; }
PlotWidget, QGraphicsView { border: 1px solid {RULE}; border-radius: 3px; }
#Sidebar, #Explorer { background: {PANEL}; }
#ExplorerRail { background: {PANEL}; border-left: 1px solid {RULE}; }
#SidebarSection { border-bottom: 1px solid {RULE}; }
#SectionCaption, #SectionHeader { color: {TEXT_3}; font-size: 10px; font-weight: 600; letter-spacing: 1px; }
QPushButton#SectionHeader { text-align: left; border: none; padding: 0; background: transparent; min-height: 0; max-height: 16777215px; }
QPushButton#SectionHeader:hover { color: {TEXT}; background: transparent; }
#SectionBadge { color: {TEXT_2}; font-size: 10px; }
#FileName { font-weight: 600; }
#Muted { color: {TEXT_3}; font-size: 11px; }
#Readout { font-family: {MONO}; font-size: 15px; color: {TEXT}; }
#ReadoutValue { font-family: {MONO}; color: {TEXT}; }
#RunBar { background: {WINDOW}; border-top: 1px solid {RULE}; }
#DropDownPanel { background: {FIELD}; border: 1px solid {RULE_STRONG}; border-radius: 3px; }
#FormLabel { color: {TEXT_2}; }
QPushButton#Segment { border-radius: 0; margin-left: -1px; padding: 0 6px; }
QPushButton#Segment[position="first"] { border-top-left-radius: 3px; border-bottom-left-radius: 3px; margin-left: 0; }
QPushButton#Segment[position="last"] { border-top-right-radius: 3px; border-bottom-right-radius: 3px; }
QPushButton#Segment:checked { color: {TEXT}; }
QPushButton#SweepToggle { padding: 0; }
"""
# Appendix A / 23.2 item 6: whitespace is not significant; the block is reproduced as is.


def stylesheet(dark, temperature):
    tok = palette(dark, temperature)

    def sub(mt):
        key = mt.group(1)
        if key.startswith("glyph:"):
            parts = key.split(":")[1:]
            name = parts[0]
            token = parts[1] if len(parts) > 1 else "TEXT_2"
            pad = int(parts[2]) if len(parts) > 2 else 0
            return glyph_path(name, tok[token], pad)
        if key == "H":
            return "24"
        if key == "FIXED":
            return "min-height: 24px; max-height: 24px;"
        if key == "MONO":
            return MONO
        if key == "SLIDER_GRADIENT":
            return SLIDER_GRADIENT
        if key in tok:
            return tok[key]
        return mt.group(0)

    # Placeholders are upper-case tokens or glyph references; CSS braces are left alone.
    return re.sub(r"\{(glyph:[a-z_]+(?::[A-Z_0-9]+)?(?::\d+)?|[A-Z_0-9]+)\}", sub, TEMPLATE)


def palette_roles(dark, temperature):
    t = palette(dark, temperature)
    roles = {
        "Window": t["WINDOW"],
        "WindowText": t["TEXT"],
        "Base": t["FIELD"],
        "AlternateBase": t["PANEL"],
        "Text": t["TEXT"],
        "Button": t["FIELD"],
        "ButtonText": t["TEXT"],
        "Highlight": t["SELECT"],
        "HighlightedText": t["TEXT"],
        "ToolTipBase": t["FIELD"],
        "ToolTipText": t["TEXT"],
        "PlaceholderText": t["TEXT_3"],
        "Link": t["TEXT_2"],
        "Mid": t["RULE"],
        "Midlight": t["RULE"],
        "Dark": t["WINDOW"],
        "Light": t["RULE_STRONG"],
        "disabled WindowText": t["TEXT_3"],
        "disabled Text": t["TEXT_3"],
        "disabled ButtonText": t["TEXT_3"],
    }
    return roles
