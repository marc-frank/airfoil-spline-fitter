"""Expected texts and values of checks 7-10 (file_flows_check.py, ui_extras_check.py),
written from spec.md. A module with the same functions can replace this one through the
environment variable IMPL_EXPECT (a path).

All paths use forward slashes; `home` is the user's home folder as Qt reports it.
"""


def file_stem(name, default="airfoil"):
    """spec.md 18."""
    stem = "".join("_" if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in name)
    return stem.strip().rstrip(". ") or default


def message(key, **kw):
    """Status-bar texts (19.11, 19.15, 19.16)."""
    return {
        "no_fit_save": "No fitted spline to save.",
        "no_fit_export": "No fitted spline to export.",
        "no_fit_report": "No fitted spline.",
        "no_airfoil_project": "No airfoil loaded to save.",
        "unsupported": "Unsupported file type: {ext}",
        "not_found": "File not found: {path}",
        "paste_empty": "No airfoil coordinates on the clipboard: copy x y pairs, one point per line",
        "save_cancelled": "Save cancelled.",
        "saved": "Saved: {path}",
        "export_cancelled": "Export cancelled.",
        "exported": "Exported: {path}",
        "unsaved_cancelled": "Cancelled: the project stays open",
        "project_load_failed": "Failed to load project: {error}",
        "not_project_file": "The file is not a project file: {detail}.",
    }[key].format(**kw) if key not in ("deleted", "restored") else _count_message(key, kw["k"])


def _count_message(key, k):
    s = "" if k == 1 else "s"
    return (f"Deleted {k} run{s}; Edit > Restore deleted runs brings them back" if key == "deleted"
            else f"Restored {k} run{s}")


def points_dialog(n_points):
    """(title, label, default, minimum, maximum) of the points-per-side dialog (19.15.1)."""
    return ("Save fitted airfoil", "Points per side, spaced closer toward the leading and trailing edge",
            max(20, (n_points + 1) // 2), 3, 100000)


def save_dialog(kind, home, name="", filename="", degree=0, suffix=""):
    """(title, default path, filter) of a save dialog."""
    if kind == "fitted":
        return ("Save fitted airfoil", f"{home}/{file_stem(name or filename)}_fitted.dat", "Airfoil data (*.dat)")
    if kind == "spline":
        return ("Save spline control points", f"{home}/{file_stem(filename or name)}.deg{degree}.spline.dat",
                "Spline data (*.spline.dat)")
    if kind == "cad":
        texts = {".step": "STEP AP214 (*.step)", ".igs": "IGES 5.3 (*.igs)", ".dxf": "DXF R2000 (*.dxf)"}
        return ("Export CAD", f"{home}/{file_stem(filename or name or 'airfoil')}{suffix}", texts[suffix])
    if kind == "project":
        return ("Save project as", f"{home}/{file_stem(name or filename or 'project')}.asfproj", "Spline project (*.asfproj)")
    raise KeyError(kind)


def fitted_file(n_per_side, precision, name):
    """(first line, number of lines, decimals) of a saved fitted airfoil (14.1, 19.15.1)."""
    return (name, 1 + 2 * n_per_side - 1, max(6, precision))


def cad_settings_default_accept():
    """QSettings values after accepting the Export CAD dialog unchanged on a fresh profile:
    cad_format, cad_units, cad_scale, cad_merge, cad_close_te."""
    return [0, "mm", 1.0, False, True]


def replace_question(basename):
    return ("Save project as", f"{basename} already exists. Replace it?")


def project_label(name, modified, runs):
    return f"{name}{'*' if modified else ''}\n{runs} optimization run{'' if runs == 1 else 's'}"


def window_title(name):
    return f"{name}[*] - Airfoil Spline Fitter"


def recent_menu(basenames):
    """Texts of File > Open recent for these files, separator excluded."""
    if not basenames:
        return ["No recent files"]
    items = [(f"&{i} " if i < 10 else "") + b.replace("&", "&&") for i, b in enumerate(basenames, 1)]
    return items + ["Clear recent files"]


def report_window(has_chart):
    return ("Fit report", 560, 720 if has_chart else 520)


def dialog_layout(name):
    """Form rows (label or None for a spanning row) and button (text, role) pairs, sorted."""
    return {
        "New NACA airfoil": (["Digits", "Points per side", "Trailing edge"], [("Cancel", "reject"), ("Create", "accept")]),
        "Export CAD": (["Format", "Units", "Scale", "Sections", "Trailing edge"], [("Cancel", "reject"), ("Export...", "accept")]),
        "Batch convert": ([None, "Source", "Target"], [("Close", "reject"), ("Convert", "accept")]),
        "Theme": (["Mode", "Temperature"], [("Close", "reject"), ("Neutral", "reset")]),
        "Scrolling and zooming": (["Scrolling", "Zooming"], [("Close", "reject"), ("Default", "reset")]),
    }[name]


def dialog_details():
    """Further dialog contents checked by ui_extras_check.py."""
    return {
        "naca_points_range": (10, 2 ** 31 - 1), "naca_te_items": ["Closed", "Open"],
        "cad_formats": ["STEP AP214 (*.step)", "IGES 5.3 (*.igs)", "DXF R2000 (*.dxf)"], "cad_units": ["mm", "m", "inch"],
        "cad_scale_min": 1e-6, "cad_merge_text": "Merge sections into one curve",
        "cad_te_text": "Close a blunt trailing edge with a line",
        "batch_note": "Fit every .dat file in the source folder with the current settings\nand save each as .spline.dat in the target folder.",
        "theme_modes": ["System", "Light", "Dark"], "theme_slider": (-100, 100, 10), "theme_ends": ["Cool", "Warm"],
        "motion_spin": (0, 1000, 25, " milliseconds", "Off"),
        "button_box_order": ["reset", "accept", "reject"],
        "popup_buttons": ["Sort ascending", "Sort descending", "Keep matching", "Add matching", "Clear filter"],
        "popup_blank": "Not used", "popup_min_width": 260,
        "drag_suffixes": [".dat", ".asfproj"],
    }
