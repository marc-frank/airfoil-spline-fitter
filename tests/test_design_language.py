"""The GUI takes all its styling from theme.py (docs/design-language.md).

Other GUI modules must not define colors or stylesheets of their own, so
new features cannot drift from the Loft design language.
"""
import io
import os
import re
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "src" / "spline_fitter" / "gui"

# What only theme.py may contain (checked in code and strings, not comments)
FORBIDDEN = {
    "hex color": re.compile(r"#[0-9A-Fa-f]{6}\b|#[0-9A-Fa-f]{3}\b"),
    "stylesheet": re.compile(r"\bsetStyleSheet\b|\bsetDefaultStyleSheet\b"),
    "QColor": re.compile(r"\bQColor\b"),
    "palette() color": re.compile(r"palette\(\w"),
    "RGB tuple": re.compile(r"\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*\)"),
    "inline CSS color": re.compile(r"(?:background-)?color\s*:"),
    "emoji": re.compile(r"\\U0001[0-9a-fA-F]{4}|[\U0001F300-\U0001FAFF]"),
}


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
    for cls in (theme.Palette, theme.Ink, theme.Z, theme.Size, theme.Type, theme.Scheme):
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

    app = QApplication.instance() or QApplication([])
    w = SplineFitterWindow()
    try:
        w.resize(1500, 900)
        w.show()
        w._advanced_section.set_expanded(True)
        app.processEvents()
        controls = [
            c for c in w.findChildren((QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit))
            if c.isVisible() and not isinstance(c, QCheckBox)
            and c.objectName() != "SectionHeader"
            and not isinstance(c.parent(), (QAbstractSpinBox, QComboBox))
        ]
        assert len(controls) > 20
        wrong = [(type(c).__name__, c.objectName(), c.height()) for c in controls
                 if c.height() != Size.CONTROL_HEIGHT]
        assert not wrong, f"controls not {Size.CONTROL_HEIGHT} px tall: {wrong}"
    finally:
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
    before = (theme.Scheme.MODE, theme.Scheme.TEMPERATURE)
    try:
        w._plot_manager.plot_airfoil([(1, 0), (0, 0), (1, 0.01)])
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
    finally:
        theme.set_theme(*before)
        w.deleteLater()
        app.processEvents()
