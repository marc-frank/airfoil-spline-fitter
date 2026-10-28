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
    defined = set()
    for cls in (theme.Palette, theme.Ink, theme.Z, theme.Size, theme.Type):
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
