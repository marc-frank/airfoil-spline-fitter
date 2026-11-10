"""spec.md 19.11.2 (drag and drop), 19.17.3 (pivot), 19.17.4 (column filter drop-down),
19.20 (scroll and zoom animation), and the dialog layouts of 19.11.5, 19.15.3, 19.18,
19.19.4 and 19.20.3, compared with a real window. User settings touched are restored.
"""
import math, os, sys, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "packages", "bspline-varpro", "src"), os.path.dirname(__file__)]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QEasingCurve, QMimeData, QPoint, QPointF, QSettings, Qt, QUrl, QVariantAnimation
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QWheelEvent
from PyQt6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QLabel, QLineEdit, QListWidget, QPushButton, QSlider, QSpinBox)
app = QApplication.instance() or QApplication([])
from spline_fitter.formatting import set_decimal_point
from spline_fitter.gui import explorer as ex, file_actions, widgets
from spline_fitter.gui.theme import Motion
from spline_fitter.gui.ui import SplineFitterWindow
from spline_fitter.gui.widgets import SegmentedControl
import gui_logic as gl   # random runs and spec column model
import importlib.util as _u
_path = os.environ.get("IMPL_EXPECT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "expect_mine.py")
_s = _u.spec_from_file_location("expect", _path); E = _u.module_from_spec(_s); _s.loader.exec_module(E)
D = E.dialog_details()

qs = QSettings("MSFra", "AirfoilSplineFitter")
KEYS = ["naca_code", "naca_points", "naca_closed_te", "cad_format", "cad_units", "cad_scale", "cad_merge", "cad_close_te",
        "theme/mode", "theme/temperature", "motion/scroll_ms", "motion/zoom_ms", "recent_files"]
saved = {k: qs.value(k) for k in KEYS if qs.contains(k)}
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def form_rows(dialog):
    """(label text, field) of a dialog's form, and the texts of its buttons."""
    form = dialog.layout(); rows, buttons = [], []
    for r in range(form.rowCount()):
        lab = form.itemAt(r, QFormLayout.ItemRole.LabelRole); fld = form.itemAt(r, QFormLayout.ItemRole.FieldRole)
        span = form.itemAt(r, QFormLayout.ItemRole.SpanningRole)
        if lab and fld:
            rows.append((lab.widget().text(), fld.widget()))
        elif span and isinstance(span.widget(), QDialogButtonBox):
            box = span.widget(); role = lambda b: box.buttonRole(b).name.replace("Role", "").lower()
            buttons = sorted((b.text(), role(b)) for b in box.buttons())
        elif span:
            rows.append((None, span.widget()))
    return rows, buttons


captured = {}
_exec = QDialog.exec
QDialog.exec = lambda self: (captured.__setitem__(self.windowTitle(), form_rows(self)), QDialog.DialogCode.Rejected)[1]

try:
    w = SplineFitterWindow(); set_decimal_point("."); w.show(); app.processEvents()
    # ---------- 19.11.2 drag and drop ----------
    tmp = tempfile.mkdtemp(); dat = os.path.join(tmp, "A.DAT"); txt = os.path.join(tmp, "b.txt")
    open(dat, "w").write("drop\n" + "\n".join(f"{x} {y}" for x, y in [(1, 0), (0.5, 0.06), (0, 0), (0.5, -0.06), (1, 0)]) + "\n")
    open(txt, "w").write("x")
    def drag(paths):
        mime = QMimeData(); mime.setUrls([QUrl.fromLocalFile(p) for p in paths] + [QUrl("https://example.com/c.dat")])
        e = QDragEnterEvent(QPoint(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        e.ignore(); w.dragEnterEvent(e); return e.isAccepted(), mime
    acc, _ = drag([txt]); check("drag: only unsupported files refused", not acc)
    acc, mime = drag([txt, dat]); check("drag: a .DAT file accepted (case-insensitive)", acc)
    w._is_fitting = True; acc, _ = drag([dat]); w._is_fitting = False; check("drag: refused while fitting", not acc)
    d = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    w.dropEvent(d); check("drop: opens the first supported file", w._airfoil is not None and w._airfoil.name == "drop",
                          w.statusBar().currentMessage())

    # ---------- 19.17.3 pivot ----------
    t = ex.RunTable(); runs = [gl.random_run(i + 1) for i in range(6)]; t.set_runs(runs)
    t.sort_by(7, Qt.SortOrder.DescendingOrder); t.set_column_filter("Optimizer", excluded={"Direct LSQ"})
    t.header.moveSection(t.header.visualIndex(16), 1)
    vis_before = t.visible_runs(); t.select_runs(vis_before[:2]); sel = t.selected_runs()
    order = [t.header.logicalIndex(v) for v in range(t.header.count())]
    t.set_pivoted(True)
    check("pivot: runs become columns", t.view.model() is t.transpose and t.view.verticalHeader() is t.header
          and t.header.orientation() == Qt.Orientation.Vertical)
    check("pivot: selection kept", [id(r) for r in t.selected_runs()] == [id(r) for r in sel])
    check("pivot: setting order kept as row order", [t.header.logicalIndex(v) for v in range(t.header.count())] == order)
    check("pivot: filters and sorting kept", t.visible_runs() == vis_before and t.sort_column == 7
          and t.sort_order == Qt.SortOrder.DescendingOrder and 1 in t.column_filters)
    t.set_pivoted(False); check("pivot off: selection kept", [id(r) for r in t.selected_runs()] == [id(r) for r in sel])

    # ---------- 19.17.4 column filter drop-down ----------
    t.clear_filters(); runs = [gl.random_run(i + 1) for i in range(12)]; t.set_runs(runs)
    col = ex.COLUMN_NAMES.index("Knots"); p = t.open_filter(col)
    lw = p.findChild(QListWidget); items = [lw.item(i).text() for i in range(lw.count())]
    cells = sorted({gl.spec_cells(r)[col][1] for r in runs}, key=lambda s: (s == "", s.casefold()))
    check("popup: caption, width, buttons", p.findChild(QLabel).text() == "KNOTS" and p.width() >= D["popup_min_width"]
          and [b.text() for b in p.findChildren(QPushButton)] == D["popup_buttons"], [b.text() for b in p.findChildren(QPushButton)])
    check("popup: values, blanks as Not used, last", items == [c or D["popup_blank"] for c in cells], (items, cells))
    check("popup: keep/add disabled without condition", not p.keep_button.isEnabled() and not p.add_button.isEnabled())
    p.condition.setText("=free")
    check("popup: condition unchecks what it rules out, live",
          p.keep_button.isEnabled() and all((lw.item(i).checkState() == Qt.CheckState.Checked) == (items[i] == "Free") for i in range(lw.count()))
          and all(gl.spec_cells(r)[col][1] == "Free" for r in t.visible_runs()), items)
    p.keep()
    check("popup: Keep matching keeps the result and empties the field", p.condition.text() == ""
          and t.column_filters.get(col) == frozenset(c for c in cells if c != "Free"))
    p.condition.setText("=uniform"); p.keep(add=True)
    check("popup: Add matching checks what it matches", t.column_filters.get(col, frozenset()) == frozenset(c for c in cells if c not in ("Free", "Uniform")))
    sa = p.select_all; sa.click()
    check("popup: Select all includes every value when some are excluded", col not in t.column_filters
          and sa.checkState() == Qt.CheckState.Checked)
    sa.click(); check("popup: Select all again excludes every value", t.column_filters.get(col) == frozenset(cells)
                      and sa.checkState() == Qt.CheckState.Unchecked)
    lw.item(0).setCheckState(Qt.CheckState.Checked)
    check("popup: one value checked by hand", t.column_filters.get(col) == frozenset(cells) - {cells[0]}
          and sa.checkState() == Qt.CheckState.PartiallyChecked)
    [b for b in p.findChildren(QPushButton) if b.text() == "Clear filter"][0].click(); app.processEvents()
    from PyQt6 import sip
    check("popup: Clear filter removes the filter and closes (deleted on close)", col not in t.column_filters and (sip.isdeleted(p) or not p.isVisible()))

    # ---------- 19.20 animation ----------
    f = lambda q: 1 - (1 - q) ** 3
    check("easing: Qt OutCubic is 1-(1-p)^3", all(abs(QEasingCurve(QEasingCurve.Type.OutCubic).valueForProgress(q) - f(q)) < 1e-12
                                                  for q in [i / 50 for i in range(51)]))
    lst = QListWidget(); lst.addItems([str(i) for i in range(400)]); lst.resize(150, 150); lst.show(); app.processEvents()
    bar = lst.verticalScrollBar()
    ev = QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(), QPoint(0, -240), Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(lst.viewport(), ev)
    from spline_fitter.gui.motion import _ScrollAnimation, SmoothViewBox
    anim = bar.findChild(_ScrollAnimation)
    check("scroll: per-pixel, out-cubic, SCROLL_MS, distance -notches*lines*step",
          lst.verticalScrollMode() == lst.ScrollMode.ScrollPerPixel and anim is not None
          and anim.easingCurve().type() == QEasingCurve.Type.OutCubic and anim.duration() == Motion.SCROLL_MS
          and anim.target == min(2 * QApplication.wheelScrollLines() * bar.singleStep(), bar.maximum()),
          (anim and anim.target, QApplication.wheelScrollLines(), bar.singleStep()))
    vb = SmoothViewBox()
    check("zoom: out-cubic over ZOOM_MS, progress 0 to 1", vb._zoom.easingCurve().type() == QEasingCurve.Type.OutCubic
          and vb._zoom.startValue() == 0.0 and vb._zoom.endValue() == 1.0)
    lst.close()

    # ---------- 19.21 item 13: button box order ----------
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    box.addButton(QPushButton("Reset"), QDialogButtonBox.ButtonRole.ResetRole)
    box.addButton(QPushButton("Accept"), QDialogButtonBox.ButtonRole.AcceptRole); box.show(); app.processEvents()
    check("button box: reset, accept, reject from left to right (Fusion on Windows)",
          [b.text() for b in sorted(box.buttons(), key=lambda b: b.geometry().x())] == [{"reset": "Reset", "accept": "Accept", "reject": "Close"}[r] for r in D["button_box_order"]])
    box.deleteLater()
    # ---------- dialog layouts ----------
    def text_of(field):
        return field.text() if hasattr(field, "text") and not isinstance(field, QAbstractSpinBox) else None
    w._on_new_naca(); rows, buttons = captured["New NACA airfoil"]
    labels = [r[0] for r in rows]; edit, spin, seg = (r[1] for r in rows)
    check("NACA dialog", (labels, buttons) == E.dialog_layout("New NACA airfoil")
          and isinstance(edit, QLineEdit) and (spin.minimum(), spin.maximum()) == D["naca_points_range"]
          and [seg.itemText(i) for i in range(seg.count())] == D["naca_te_items"], (labels, buttons))
    w._fit_result = object(); w._airfoil = w._airfoil
    w._on_export_cad(); rows, buttons = captured["Export CAD"]
    fmt = rows[0][1]; units = rows[1][1]; scale = rows[2][1]
    check("Export CAD dialog", ([r[0] for r in rows], buttons) == E.dialog_layout("Export CAD")
          and [fmt.itemText(i) for i in range(3)] == D["cad_formats"]
          and [units.itemText(i) for i in range(3)] == D["cad_units"] and scale.minimum() == D["cad_scale_min"]
          and rows[3][1].text() == D["cad_merge_text"] and rows[4][1].text() == D["cad_te_text"],
          ([r[0] for r in rows], buttons))
    w._fit_result = None
    w._batch_controller.on_batch_dialog(); rows, buttons = captured["Batch convert"]
    check("Batch dialog", ([r[0] for r in rows], buttons) == E.dialog_layout("Batch convert")
          and rows[0][1].text() == D["batch_note"],
          ([r[0] for r in rows], buttons))
    dlg = widgets.ThemeDialog(w); rows, buttons = form_rows(dlg)
    mode = rows[0][1]; temp_row = rows[1][1]; slider = temp_row.findChild(QSlider)
    check("Theme dialog", dlg.windowTitle() == "Theme" and ([r[0] for r in rows], buttons) == E.dialog_layout("Theme")
          and [mode.itemText(i) for i in range(3)] == D["theme_modes"]
          and (slider.minimum(), slider.maximum(), slider.pageStep()) == D["theme_slider"]
          and [l.text() for l in temp_row.findChildren(QLabel)] == D["theme_ends"], ([r[0] for r in rows], buttons))
    dlg.deleteLater()
    dlg = widgets.MotionDialog(w); rows, buttons = form_rows(dlg)
    s1, s2 = rows[0][1], rows[1][1]
    check("Scrolling and zooming dialog", dlg.windowTitle() == "Scrolling and zooming"
          and ([r[0] for r in rows], buttons) == E.dialog_layout("Scrolling and zooming")
          and all((s.minimum(), s.maximum(), s.singleStep(), s.suffix(), s.specialValueText()) == D["motion_spin"] for s in (s1, s2)),
          ([r[0] for r in rows], buttons))
    dlg.deleteLater()
    w._project_manager._project_modified = False
    w.close(); w.deleteLater(); app.processEvents()
finally:
    QDialog.exec = _exec
    for k in KEYS:
        qs.remove(k)
    for k, v in saved.items():
        qs.setValue(k, v)
    qs.sync()

for name, ok, detail in results:
    print(("ok  " if ok else "FAIL"), name, "" if ok else detail)
print(f"\n{sum(ok for _, ok, _ in results)} of {len(results)} interaction and dialog checks agree with spec.md")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
