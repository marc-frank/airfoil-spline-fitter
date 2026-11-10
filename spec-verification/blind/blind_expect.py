"""What the window must show, written only from spec.md (19.x, 20)."""
from __future__ import annotations

from blind_impl import file_stem, _name_line

UNLIMITED = 2 ** 31 - 1
APP_TITLE = "Airfoil Spline Fitter"


def _s(k):
    return "" if k == 1 else "s"


def message(key, **kw):
    if key == "no_fit_save":
        return "No fitted spline to save."
    if key == "no_fit_export":
        return "No fitted spline to export."
    if key == "no_fit_report":
        return "No fitted spline."
    if key == "no_airfoil_project":
        return "No airfoil loaded to save."
    if key == "unsupported":
        return f"Unsupported file type: {kw['ext']}"
    if key == "not_found":
        return f"File not found: {kw['path']}"
    if key == "paste_empty":
        return "No airfoil coordinates on the clipboard: copy x y pairs, one point per line"
    if key == "save_cancelled":
        return "Save cancelled."
    if key == "saved":
        return f"Saved: {kw['path']}"
    if key == "export_cancelled":
        return "Export cancelled."
    if key == "exported":
        return f"Exported: {kw['path']}"
    if key == "unsaved_cancelled":
        # 19.11.1 set_airfoil step 1.
        return "Cancelled: the project stays open"
    if key == "project_load_failed":
        return f"Failed to load project: {kw['error']}"
    if key == "not_project_file":
        # SPEC-GAP: 15.4/19.16.2 "not_project_file" is returned as the full status the
        # window shows: "Failed to load project: " + the ProjectFormatError text.
        return f"Failed to load project: The file is not a project file: {kw['detail']}."
    if key == "deleted":
        k = kw["k"]
        return f"Deleted {k} run{_s(k)}; Edit > Restore deleted runs brings them back"
    if key == "restored":
        k = kw["k"]
        return f"Restored {k} run{_s(k)}"
    raise KeyError(key)


def points_dialog(n_points):
    return (
        "Save fitted airfoil",
        "Points per side, spaced closer toward the leading and trailing edge",
        max(20, (n_points + 1) // 2),
        3,
        100000,
    )


CAD_FORMATS = [
    ("STEP AP214 (*.step)", ".step"),
    ("IGES 5.3 (*.igs)", ".igs"),
    ("DXF R2000 (*.dxf)", ".dxf"),
]


def save_dialog(kind, home, name="", filename="", degree=0, suffix=""):
    home = str(home).rstrip("/\\")
    if kind == "fitted":
        return ("Save fitted airfoil", f"{home}/{file_stem(name or filename)}_fitted.dat",
                "Airfoil data (*.dat)")
    if kind == "spline":
        return ("Save spline control points",
                f"{home}/{file_stem(filename or name)}.deg{degree}.spline.dat",
                "Spline data (*.spline.dat)")
    if kind == "cad":
        flt = next(text for text, suf in CAD_FORMATS if suf == suffix)
        return ("Export CAD", f"{home}/{file_stem(filename or name or 'airfoil')}{suffix}", flt)
    if kind == "project":
        return ("Save project as", f"{home}/{file_stem(name or filename or 'project')}.asfproj",
                "Spline project (*.asfproj)")
    raise KeyError(kind)


def fitted_file(n_per_side, precision, name):
    # 19.15.1: export_airfoil_data(path, resample(curves, n), name, precision=max(6, precision));
    # 14.1: precision clamped to 0..16; name line only if name is truthy.
    d = max(0, min(int(max(6, precision)), 16))
    n_pts = 2 * n_per_side - 1
    if name:
        return (_name_line(name), n_pts + 1, d)
    # SPEC-GAP: 19.15.1 without a name the first line is the first point; its text depends
    # on the fit, so None is returned for it.
    return (None, n_pts, d)


def cad_settings_default_accept():
    # 19.15.3: format index, units text, scale, merge, close TE (defaults of 19.23).
    return [0, "mm", 1.0, False, True]


def replace_question(basename):
    return ("Save project as", f"{basename} already exists. Replace it?")


def project_label(name, modified, runs):
    if name is None:
        return "No project"
    return f"{name}{'*' if modified else ''}\n{runs} optimization run{_s(runs)}"


def window_title(name):
    if name is None:
        return APP_TITLE
    # SPEC-GAP: 19.16.5 the title is returned with Qt's "[*]" placeholder (windowTitle()),
    # not as displayed.
    return f"{name}[*] - {APP_TITLE}"


def recent_menu(basenames):
    names = list(basenames)[:10]
    if not names:
        return ["No recent files"]
    out = []
    for i, b in enumerate(names, start=1):
        b = b.replace("&", "&&")
        out.append(f"&{i} {b}" if i < 10 else b)
    out.append("Clear recent files")
    return out


def report_window(has_chart):
    return ("Fit report", 560, 720 if has_chart else 520)


_LAYOUTS = {
    "New NACA airfoil": (
        ["Digits", "Points per side", "Trailing edge"],
        [("Create", "accept"), ("Cancel", "reject")],
    ),
    "Export CAD": (
        ["Format", "Units", "Scale", "Sections", "Trailing edge"],
        [("Export...", "accept"), ("Cancel", "reject")],
    ),
    "Batch convert": (
        # SPEC-GAP: 19.18 the muted note is assumed to be a form row spanning both
        # columns (None), above Source and Target.
        [None, "Source", "Target"],
        [("Convert", "accept"), ("Close", "reject")],
    ),
    "Theme": (
        ["Mode", "Temperature"],
        [("Close", "reject"), ("Neutral", "reset")],
    ),
    "Scrolling and zooming": (
        ["Scrolling", "Zooming"],
        [("Close", "reject"), ("Default", "reset")],
    ),
}


def dialog_layout(name):
    rows, buttons = _LAYOUTS[name]
    return list(rows), sorted(buttons)


def dialog_details():
    return {
        "naca_points_range": (10, UNLIMITED),
        "naca_te_items": ["Closed", "Open"],
        "cad_formats": [t for t, _ in CAD_FORMATS],
        "cad_units": ["mm", "m", "inch"],
        "cad_scale_min": 1e-6,
        "cad_merge_text": "Merge sections into one curve",
        "cad_te_text": "Close a blunt trailing edge with a line",
        "batch_note": "Fit every .dat file in the source folder with the current settings\n"
                      "and save each as .spline.dat in the target folder.",
        "theme_modes": ["System", "Light", "Dark"],
        "theme_slider": (-100, 100, 10),
        "theme_ends": ["Cool", "Warm"],
        "motion_spin": (0, 1000, 25, " milliseconds", "Off"),
        "button_box_order": ["reset", "accept", "reject"],
        # SPEC-GAP: 19.17.4 "buttons": Select all is a check box and the value list is a
        # list, so only the push buttons are listed, in the order of the contents list.
        "popup_buttons": ["Sort ascending", "Sort descending", "Keep matching", "Add matching",
                          "Clear filter"],
        "popup_blank": "Not used",
        "popup_min_width": 260,
        "drag_suffixes": [".dat", ".asfproj"],
    }
