"""spec.md 19.19 and Appendix A: the stylesheet, Qt palette and glyph files the running
application installs, against those built only from the text of spec.md, after real
theme switches (both modes, several temperatures). The user's saved theme is restored.
"""
import math, os, re, sys, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "packages", "bspline-varpro", "src"), os.path.dirname(__file__)]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cleanroom as cr

spec = open(os.path.join(ROOT, "spec.md"), encoding="utf-8").read()
greys = {n: (float(d), float(l)) for n, d, l in re.findall(
    r"^\| `(\w+)`(?: \(plot\))? \| ([\d.]+) \| ([\d.]+) \| `#[0-9A-F]{6}` \| `#[0-9A-F]{6}` \|$", spec, re.M)}
mode_colours = {n: (d, l) for n, d, l in re.findall(r"`(\w+) (#[0-9A-F]{6})/(#[0-9A-F]{6})`", spec)}
ink = dict(re.findall(r"`((?:INK\w*)|ON_INK) (#[0-9A-F]{6})`", spec))
glyphs = dict(re.findall(r"`(chevron_down|chevron_up|chevron_right|chevron_left|check|dash|sort_up|sort_down|filter|range) (M[^`]+)`", spec))
template = spec.split("## Appendix A")[1].split("```css\n")[1].split("```")[0]
gradient = re.search(r"`\{SLIDER_GRADIENT\}` is `([^`]+)`", spec).group(1)
roles = re.search(r"set the Qt palette \((.+?); disabled", spec).group(1)
assert len(greys) == 13 and len(mode_colours) == 6 and len(ink) == 5 and len(glyphs) == 10, (len(greys), mode_colours, ink, glyphs)

tmp = tempfile.gettempdir().replace("\\", "/")


def palette(dark, temperature):
    side = 0 if dark else 1
    p = {n: cr.grey(L[side], temperature / 100) for n, L in greys.items() if n != "CONSTRUCTION"}
    p.update({n: c[side] for n, c in mode_colours.items() if n in ("OK", "WARN", "ERROR")})
    p.update(ink)
    return p


def glyph_path(name, colour, pad):
    return f"{tmp}/spline_fitter_glyphs/{name}_{colour.lstrip('#').lower()}_{pad}.svg"


def glyph_svg(name, colour, pad):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-pad} {-pad} {12 + 2 * pad} {12 + 2 * pad}">'
            f'<path d="{glyphs[name]}" fill="none" stroke="{colour}" stroke-width="1.4" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def expected_qss(p):
    def g(m):
        parts = m.group(1).split(":")
        token = parts[1] if len(parts) > 1 else "TEXT_2"; pad = int(parts[2]) if len(parts) > 2 else 0
        return glyph_path(parts[0], p[token], pad)
    s = re.sub(r"\{glyph:([^}]+)\}", g, template)
    s = s.replace("{SLIDER_GRADIENT}", gradient).replace("{FIXED}", "min-height: 24px; max-height: 24px;")
    s = s.replace("{H}", "24").replace("{MONO}", "Cascadia Mono, Consolas, Menlo, monospace")
    return re.sub(r"\{([A-Z_0-9]+)\}", lambda m: p[m.group(1)], s)


def norm(css):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return re.sub(r"\s+", " ", css).replace("{ ", "{").replace(" }", "}").strip()


# Expected values: this file's own, built from the tables and template of spec.md (default),
# or stylesheet(dark, temperature), palette_roles(dark, temperature) and
# glyph_svg(name, colour_hex, pad) of the module named by IMPL_THEME (a path)
IMPL_THEME = None
if os.environ.get("IMPL_THEME"):
    import importlib.util as _u
    _s = _u.spec_from_file_location("impl_theme", os.environ["IMPL_THEME"]); IMPL_THEME = _u.module_from_spec(_s); _s.loader.exec_module(IMPL_THEME)

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
settings = QSettings("MSFra", "AirfoilSplineFitter")
saved = {k: settings.value(k) for k in ("theme/mode", "theme/temperature") if settings.contains(k)}
from spline_fitter.gui import theme
theme.apply_theme(app)
results = []
try:
    for mode in ("light", "dark"):
        for temperature in (-100, -40, 0, 25, 100):
            theme.set_theme(mode=mode, temperature=temperature)
            p = palette(mode == "dark", temperature)
            want_text = IMPL_THEME.stylesheet(mode == "dark", temperature) if IMPL_THEME else expected_qss(p)
            want, got = norm(want_text), norm(app.styleSheet())
            first = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
            results.append((f"stylesheet {mode} {temperature:+d}", want == got, (want[first - 60:first + 60], got[first - 60:first + 60])))
            pal = app.palette(); bad = []
            expected_roles = (IMPL_THEME.palette_roles(mode == "dark", temperature) if IMPL_THEME else
                              {**{role: p[token] for role, token in re.findall(r"(\w+) `(\w+)`", roles)},
                               **{"disabled " + r: p["TEXT_3"] for r in ("WindowText", "Text", "ButtonText")}})
            for role, colour in expected_roles.items():
                if role.startswith("disabled "):
                    c = pal.color(QPalette.ColorGroup.Disabled, getattr(QPalette.ColorRole, role.split()[1])).name().upper()
                else:
                    c = pal.color(getattr(QPalette.ColorRole, role)).name().upper()
                if c != colour.upper():
                    bad.append((role, c, colour))
            results.append((f"palette {mode} {temperature:+d}", not bad, bad))
            files_ok = []
            for m in re.finditer(r"url\(([^)]+)\)", app.styleSheet()):
                path = m.group(1); name, colour, pad = re.match(r".*/(\w+?)_([0-9a-f]{6})_(\d+)\.svg$", path).groups()
                name = name if name in glyphs else name.rsplit("_", 0)[0]
                svg = (IMPL_THEME.glyph_svg if IMPL_THEME else glyph_svg)(name, "#" + colour.upper(), int(pad))
                files_ok.append(open(path, encoding="utf-8").read() == svg)
            results.append((f"glyph files {mode} {temperature:+d}", all(files_ok) and len(files_ok) >= 6, len(files_ok)))
finally:
    for k in ("theme/mode", "theme/temperature"):
        settings.remove(k)
    for k, v in saved.items():
        settings.setValue(k, v)
    settings.sync()

for name, ok, detail in results:
    print(("ok  " if ok else "FAIL"), name, "" if ok else detail)
print(f"\n{sum(ok for _, ok, _ in results)} of {len(results)} theme checks agree with spec.md")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
