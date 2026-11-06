"""spec.md 19.11, 19.15, 19.16: file and project flows of a real window, with dialogs
stubbed to record their titles, defaults and filters. Status texts, dialog arguments
and written files are compared with the literals of spec.md. The user's QSettings
values touched here are saved first and restored afterwards.
"""
import os, sys, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "packages", "bspline-varpro", "src")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QDir, QSettings
from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox
app = QApplication.instance() or QApplication([])
from spline_fitter.formatting import set_decimal_point
from spline_fitter.gui import file_actions, project as project_mod
from spline_fitter.gui.ui import SplineFitterWindow
from spline_fitter.io import parse_spline_data
import importlib.util as _u
# Expected texts and values: expect_mine.py, or the module named by IMPL_EXPECT (a path)
_path = os.environ.get("IMPL_EXPECT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "expect_mine.py")
_s = _u.spec_from_file_location("expect", _path); E = _u.module_from_spec(_s); _s.loader.exec_module(E)

KEYS = ["recent_files", "cad_format", "cad_units", "cad_scale", "cad_merge", "cad_close_te"]
qs = QSettings("MSFra", "AirfoilSplineFitter")
saved = {k: qs.value(k) for k in KEYS if qs.contains(k)}
for _k in KEYS:
    qs.remove(_k)   # start from a fresh profile for these keys (restored at the end)
results = []
home = QDir.homePath()
tmp = tempfile.mkdtemp()


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


calls = {}
answers = {}


def stub_save(parent, title, default, flt, *a, **k):
    calls["save"] = (title, default.replace("\\", "/"), flt); return answers.pop("save", ("", ""))


def stub_open(parent, title, default, flt, *a, **k):
    calls["open"] = (title, default.replace("\\", "/"), flt); return answers.pop("open", ("", ""))


def stub_int(parent, title, label, value, lo, hi, *a, **k):
    calls["int"] = (title, label, value, lo, hi); return answers.pop("int", (value, False))


QFileDialog.getSaveFileName = staticmethod(stub_save)
QFileDialog.getOpenFileName = staticmethod(stub_open)
QInputDialog.getInt = staticmethod(stub_int)
_exec = QDialog.exec
QDialog.exec = lambda self: answers.pop("dialog", QDialog.DialogCode.Rejected)
QMessageBox.question = staticmethod(lambda *a, **k: (calls.__setitem__("question", a[1:3]), answers.pop("question"))[1])
documents = []
file_actions.show_document = lambda parent, title, html, w, h, extra=None: documents.append((title, w, h, extra is not None))

try:
    w = SplineFitterWindow(); set_decimal_point("."); w.show(); app.processEvents()
    msg = lambda: w.statusBar().currentMessage()
    M = E.message
    # Without a fit (19.15)
    w._on_save_fitted(); check("save .dat without fit", msg() == M("no_fit_save"), msg())
    w._on_save_spline(); check("save spline without fit", msg() == M("no_fit_save"), msg())
    w._on_export_cad(); check("export without fit", msg() == M("no_fit_export"), msg())
    w._on_fit_report(); check("report without fit", msg() == M("no_fit_report"), msg())
    w._project_manager.on_save_project_as(); check("save project as without airfoil", msg() == M("no_airfoil_project"), msg())
    # Opening (19.11)
    w._load_airfoil(os.path.join(tmp, "x.txt")); check("unsupported type", msg() == M("unsupported", ext=".txt"), msg())
    missing = os.path.join(tmp, "missing.dat"); w.open_path(missing)
    check("missing file", msg() == M("not_found", path=missing), msg())
    QApplication.clipboard().setText("1 2\n3 4\n"); w._on_paste_airfoil()
    check("paste without airfoil", msg() == M("paste_empty"), msg())
    w.new_naca_airfoil("2412", 81, True)   # 161 points, filename naca2412, name NACA 2412, precision 6
    w._combo_opt.setCurrentIndex(1); w._combo_cont.setCurrentIndex(2)   # Direct LSQ, C2: fast
    w._fitting_controller.on_fit()
    # 19.15.1 save fitted airfoil
    w._on_save_fitted()
    check("save .dat: points dialog", calls["int"] == E.points_dialog(161), calls["int"])
    check("save .dat: cancel points", msg() == M("save_cancelled"), msg())
    answers["int"] = (41, True); w._on_save_fitted()
    check("save .dat: file dialog", calls["save"] == E.save_dialog("fitted", home, name="NACA 2412", filename="naca2412"), calls["save"])
    check("save .dat: cancel file", msg() == M("save_cancelled"), msg())
    path = os.path.join(tmp, "fitted.dat"); answers["int"] = (41, True); answers["save"] = (path, ""); w._on_save_fitted()
    lines = open(path, encoding="utf-8").read().splitlines(); first, count, decimals = E.fitted_file(41, 6, "NACA 2412")
    check("save .dat: written", msg() == M("saved", path=path) and lines[0] == first and len(lines) == count
          and all(len(v.split(".")[1]) == decimals for v in lines[1].split()), (msg(), lines[:2], len(lines)))
    # 19.15.2 save spline
    w._on_save_spline()
    check("save spline: file dialog", calls["save"] == E.save_dialog("spline", home, name="NACA 2412", filename="naca2412", degree=5), calls["save"])
    check("save spline: cancel", msg() == M("save_cancelled"), msg())
    path = os.path.join(tmp, "s.spline.dat"); answers["save"] = (path, ""); w._on_save_spline()
    s = parse_spline_data(path); text = open(path, encoding="utf-8").read().splitlines()
    check("save spline: written", msg() == M("saved", path=path) and s["name"] == "NACA 2412" and s["degree"] == 5
          and s["continuity"] == ["C2"] and len(text[-1].split()[0].split(".")[1]) == 12, (msg(), s["name"], s["continuity"]))
    # 19.15.3 export CAD
    w._on_export_cad(); check("export: dialog cancel", msg() == M("export_cancelled"), msg())
    answers["dialog"] = QDialog.DialogCode.Accepted; w._on_export_cad()
    check("export: file dialog", calls["save"] == E.save_dialog("cad", home, name="NACA 2412", filename="naca2412", suffix=".step"), calls["save"])
    check("export: file cancel", msg() == M("export_cancelled"), msg())
    stored = [qs.value(k) for k in ("cad_format", "cad_units", "cad_scale", "cad_merge", "cad_close_te")]
    as_bool = lambda v: v if isinstance(v, bool) else str(v).lower() == "true"
    want = E.cad_settings_default_accept()
    check("export: settings stored", int(stored[0]) == want[0] and stored[1] == want[1] and float(stored[2]) == want[2]
          and as_bool(stored[3]) is want[3] and as_bool(stored[4]) is want[4], (stored, want))   # the registry returns strings
    path = os.path.join(tmp, "a.step"); answers["dialog"] = QDialog.DialogCode.Accepted; answers["save"] = (path, ""); w._on_export_cad()
    step = open(path, encoding="ascii").read()
    check("export: written", msg() == M("exported", path=path) and step.startswith("ISO-10303-21;\nHEADER;")
          and "PRODUCT('NACA 2412','NACA 2412','',(#3))" in step, msg())
    # 19.15.4 report
    w._on_fit_report(); check("report window", [d[:3] for d in documents] == [E.report_window(True)] and documents[0][3], documents)
    # 19.16.3 save project as
    base = os.path.join(tmp, "proj"); answers["save"] = (base, ""); w._project_manager.on_save_project_as()
    check("save as: dialog", calls["save"] == E.save_dialog("project", home, name="NACA 2412", filename="naca2412"), calls["save"])
    check("save as: suffix added", msg() == M("saved", path=base + ".asfproj") and os.path.isfile(base + ".asfproj")
          and w.windowTitle() == E.window_title("proj.asfproj") and w._lbl_project.text() == E.project_label("proj.asfproj", False, 1),
          (msg(), w.windowTitle(), w._lbl_project.text()))
    answers["save"] = (base, ""); answers["question"] = QMessageBox.StandardButton.No; w._project_manager.on_save_project_as()
    check("save as: replace declined", msg() == M("save_cancelled") and calls["question"] == E.replace_question("proj.asfproj"),
          (msg(), calls.get("question")))
    # 19.11.3 recent files menu (this check starts from an empty recent list)
    w._populate_recent_menu(); texts = [a.text() for a in w._menu_recent.actions() if not a.isSeparator()]
    check("recent menu", texts == E.recent_menu(["proj.asfproj"]), texts)
    # 19.16.6 delete and restore
    w._run_table.select_runs(list(w._project_manager.project.optimization_runs)); w._project_manager.on_table_selection_changed()
    w._project_manager._confirm_delete = lambda n: True; w._project_manager.on_delete_selected_runs()
    check("delete", msg() == M("deleted", k=1) and w._lbl_project.text() == E.project_label("proj.asfproj", True, 0),
          (msg(), w._lbl_project.text()))
    w._project_manager.on_restore_deleted_runs(); check("restore", msg() == M("restored", k=1), msg())
    # 19.16.1 unsaved changes, cancel
    w._project_manager._ask_to_save = lambda: QMessageBox.StandardButton.Cancel; w.new_naca_airfoil("0012", 41, True)
    check("unsaved: cancel keeps project", msg() == M("unsaved_cancelled") and w._airfoil.name == "NACA 2412", msg())
    # 19.16.2 broken project file
    bad = os.path.join(tmp, "bad.asfproj"); open(bad, "w").write("{not json")
    w._project_manager._ask_to_save = lambda: QMessageBox.StandardButton.Discard; w.open_path(bad)
    import json
    try:
        json.loads("{not json")
    except json.JSONDecodeError as exc:
        detail = str(exc)
    inner = M("not_project_file", detail=detail)
    # The interface given to other expectation modules left open whether "not_project_file" is the bare
    # error or the whole status line: either reading is accepted
    wanted = {M("project_load_failed", error=inner), inner}
    check("broken project", msg() in wanted and w._airfoil.name == "NACA 2412", (msg(), wanted))
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
print(f"\n{sum(ok for _, ok, _ in results)} of {len(results)} file and project flow checks agree with spec.md")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
