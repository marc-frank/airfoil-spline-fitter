"""
File Actions
============

Open, save, export, and report actions for the main window: files opened
from the menu, File > Open recent, a drop on the window or the clipboard,
and airfoils generated from a NACA code.
"""

from __future__ import annotations

import os
import threading

import numpy as np
from PyQt6.QtCore import QDir, QMimeData, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QSpinBox,
)

from ..analysis import airfoil_shape, fit_report, format_report_html, scipy_curves, shape_report
from ..geometry import create_airfoil_data
from ..io import export_airfoil_data, export_spline_data, parse_airfoil_data, parse_airfoil_text
from .background import run_in_background
from .plotting import shape_plot
from .widgets import ScientificSpinBox, SegmentedControl, add_form_row, dialog_form, show_document

# Files the window opens: dropped, recent, or copied to the clipboard
AIRFOIL_SUFFIX = ".dat"
PROJECT_SUFFIX = ".asfproj"
# Entries in File > Open recent
RECENT_FILES = 10
# Fewer pasted points are taken for stray numbers, not an airfoil
MIN_PASTED_POINTS = 5
# Decimals of a generated NACA airfoil, as in published coordinate files
NACA_DECIMALS = 6


def _supported_files(mime: QMimeData | None) -> list[str]:
    """The local airfoil and project files in dragged or copied data."""
    if mime is None or not mime.hasUrls():
        return []
    return [url.toLocalFile() for url in mime.urls()
            if url.isLocalFile() and os.path.splitext(url.toLocalFile())[1].lower()
            in (AIRFOIL_SUFFIX, PROJECT_SUFFIX)]


class FileActionsMixin:
    """
    File menu, CAD export, fit report, and completion sounds.

    Mixed into SplineFitterWindow; methods use the window's widgets and
    state through ``self``.
    """

    # -------------------------------------------------------------------------
    # File Operations
    # -------------------------------------------------------------------------

    def _on_open_file(self):
        """Handle open file button click."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open airfoil",
            QDir.homePath(),
            "Airfoil data (*.dat)",
        )
        if path:
            self._load_airfoil(path)

    def _load_airfoil(self, path: str) -> bool:
        """Load and display an airfoil file; whether it loaded."""
        ext = os.path.splitext(path)[1].lower()
        if ext != AIRFOIL_SUFFIX:
            self.statusBar().showMessage(f"Unsupported file type: {ext}")
            return False

        self.statusBar().showMessage(f"Loading: {os.path.basename(path)}...")
        QApplication.processEvents()

        try:
            filename_name, content_name, coords, max_precision = parse_airfoil_data(path)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load: {e}")
            self._play_error_chime()
            return False
        if not coords:
            self.statusBar().showMessage("No coordinates found in the file")
            self._play_error_chime()
            return False
        loaded = self._set_airfoil(
            coords,
            filename=filename_name or os.path.basename(path),
            name=content_name or filename_name or "",
            precision=max_precision if isinstance(max_precision, int) else 6,
            path=path,
        )
        if loaded:
            self._add_recent_file(path)
        return loaded

    def _set_airfoil(self, coords, filename: str, name: str, precision: int,
                     path: str = "", verb: str = "Loaded") -> bool:
        """Make ``coords`` the airfoil to fit, from a file, the clipboard or
        the NACA generator; whether it worked."""
        try:
            # Create airfoil data object with current section count
            self._airfoil = create_airfoil_data(
                coords,
                filename=filename,
                name=name,
                precision=precision,
                num_sections=self._spin_sections.value(),
            )

            # Clear previous fit
            self._fit_result = None
            self._set_save_enabled(False)

            # Reset project state for new airfoil
            self._project_manager.reset()

            # Update file card
            self._show_file_info(self._airfoil, path)

            # Plot airfoil
            self._plot_manager.plot_airfoil(coords)
            # The automatic tolerance follows the airfoil: not a step to undo
            with self._settings_history.quiet():
                self._update_auto_tolerance()

            section_sizes = [len(sec) for sec in self._airfoil.sections]
            self.statusBar().showMessage(
                f"{verb}: {name or filename} ({len(coords)} points, "
                f"{len(section_sizes)} sections: {section_sizes})"
            )
            return True

        except Exception as e:
            self.statusBar().showMessage(f"Failed to load: {e}")
            self._play_error_chime()
            return False

    def open_path(self, path: str) -> bool:
        """Open an airfoil (.dat) or project (.asfproj) file; whether it opened."""
        if self._is_fitting:
            self.statusBar().showMessage("Cancel the running fit to open a file")
            return False
        if not os.path.isfile(path):
            self.statusBar().showMessage(f"File not found: {path}")
            self._remove_recent_file(path)
            self._play_error_chime()
            return False
        if os.path.splitext(path)[1].lower() == PROJECT_SUFFIX:
            return self._project_manager.open_project(path)
        return self._load_airfoil(path)

    # -- Drop -------------------------------------------------------------------
    # The widgets under the cursor ignore file drags, which so reach the window

    def dragEnterEvent(self, event):
        if not self._is_fitting and _supported_files(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = _supported_files(event.mimeData())
        if files:
            event.acceptProposedAction()
            self.activateWindow()
            self.open_path(files[0])

    # -- Recent files ----------------------------------------------------------

    def _recent_files(self) -> list[str]:
        value = self._settings.value("recent_files", [])
        # QSettings gives a list of one back as a plain string
        return [value] if isinstance(value, str) else list(value or [])

    def _add_recent_file(self, path: str) -> None:
        """Put ``path`` at the top of File > Open recent."""
        path = os.path.abspath(path)
        files = [path] + [f for f in self._recent_files()
                          if os.path.normcase(f) != os.path.normcase(path)]
        self._settings.setValue("recent_files", files[:RECENT_FILES])

    def _remove_recent_file(self, path: str) -> None:
        path = os.path.normcase(os.path.abspath(path))
        self._settings.setValue("recent_files", [
            f for f in self._recent_files() if os.path.normcase(f) != path])

    def _populate_recent_menu(self) -> None:
        """Fill File > Open recent as it opens."""
        menu = self._menu_recent
        menu.clear()
        files = self._recent_files()
        if not files:
            menu.addAction("No recent files").setEnabled(False)
            return
        for i, path in enumerate(files, start=1):
            # The first nine get an access key; && shows an ampersand
            name = os.path.basename(path).replace("&", "&&")
            action = menu.addAction(f"&{i} {name}" if i < 10 else name)
            action.setToolTip(path)
            action.triggered.connect(lambda _=False, p=path: self.open_path(p))
        menu.addSeparator()
        menu.addAction("Clear recent files").triggered.connect(
            lambda: self._settings.setValue("recent_files", []))

    # -- Clipboard and generator -----------------------------------------------

    def _on_paste_airfoil(self):
        """Load coordinates, or a copied airfoil or project file, from the clipboard."""
        if self._is_fitting:
            return
        mime = QApplication.clipboard().mimeData()
        files = _supported_files(mime)
        if files:
            self.open_path(files[0])
            return
        text = mime.text() if mime is not None and mime.hasText() else ""
        try:
            _, content_name, coords, max_precision = parse_airfoil_text(text)
        except Exception:
            coords = None
        if not coords or len(coords) < MIN_PASTED_POINTS:
            self.statusBar().showMessage(
                "No airfoil coordinates on the clipboard: copy x y pairs, one point per line")
            self._play_error_chime()
            return
        self._set_airfoil(coords, filename="", name=content_name or "Pasted airfoil",
                          precision=max_precision or 6, verb="Pasted")

    def _on_new_naca(self):
        """Ask for a NACA 4-digit code and make that airfoil the one to fit."""
        dialog = QDialog(self)
        dialog.setWindowTitle("New NACA airfoil")
        form = dialog_form(dialog)
        edit_code = QLineEdit(self._settings.value("naca_code", "2412"))
        edit_code.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{4}")))
        edit_code.setToolTip(
            "Four digits: the camber in percent of the chord, its position in tenths\n"
            "of the chord, and the thickness in percent. 2412 has 2% camber at 40%\n"
            "of the chord and is 12% thick; 0012 is symmetric.")
        add_form_row(form, "Digits", edit_code)
        spin_points = QSpinBox()
        spin_points.setRange(10, 5000)
        spin_points.setValue(self._settings.value("naca_points", 81, type=int))
        spin_points.setToolTip("Spaced closer toward the leading and trailing edge")
        add_form_row(form, "Points per side", spin_points)
        seg_te = SegmentedControl()
        seg_te.addItem("Closed", userData=True)
        seg_te.addItem("Open", userData=False)
        seg_te.setItemToolTip(0, "Modified thickness coefficient: a sharp trailing edge")
        seg_te.setItemToolTip(1, "Original thickness coefficient: a thin blunt trailing edge")
        seg_te.setCurrentIndex(
            max(0, seg_te.findData(self._settings.value("naca_closed_te", True, type=bool))))
        add_form_row(form, "Trailing edge", seg_te)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        btn_create = buttons.addButton("Create", QDialogButtonBox.ButtonRole.AcceptRole)
        btn_create.setDefault(True)
        edit_code.textChanged.connect(
            lambda _: btn_create.setEnabled(edit_code.hasAcceptableInput()))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        code, n_per_side, closed_te = edit_code.text(), spin_points.value(), seg_te.currentData()
        self._settings.setValue("naca_code", code)
        self._settings.setValue("naca_points", n_per_side)
        self._settings.setValue("naca_closed_te", closed_te)
        self.new_naca_airfoil(code, n_per_side, closed_te)

    def new_naca_airfoil(self, code: str, n_per_side: int = 81, closed_te: bool = True) -> bool:
        """Make a NACA 4-digit section the airfoil to fit; whether it worked."""
        from bspline_varpro.airfoil import NACA4

        if self._is_fitting:
            return False
        try:
            points = np.round(NACA4(code, closed_te=closed_te).points(n_per_side), NACA_DECIMALS)
        except ValueError as e:
            self.statusBar().showMessage(f"Cannot create the airfoil: {e}")
            self._play_error_chime()
            return False
        return self._set_airfoil([(float(x), float(y)) for x, y in points],
                                 filename=f"naca{code}", name=f"NACA {code}",
                                 precision=NACA_DECIMALS, verb="Created")

    def _on_save_fitted(self):
        """Save fitted airfoil as sampled points."""
        if self._fit_result is None or self._airfoil is None:
            self.statusBar().showMessage("No fitted spline to save.")
            return

        try:
            from bspline_varpro.airfoil import resample

            default_n = max(20, (self._airfoil.num_points + 1) // 2)
            n_per_side, ok = QInputDialog.getInt(
                self, "Save fitted airfoil",
                "Points per side, spaced closer toward the leading and trailing edge",
                default_n, 3, 100000,
            )
            if not ok:
                self.statusBar().showMessage("Save cancelled.")
                return
            points = resample(scipy_curves(self._fit_result), n_per_side, spacing="cosine")
            coords = [(float(x), float(y)) for x, y in points]

            # File dialog
            default_name = self._airfoil.name or self._airfoil.filename
            default_path = os.path.join(QDir.homePath(), f"{default_name}_fitted.dat")

            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save fitted airfoil",
                default_path,
                "Airfoil data (*.dat)",
            )

            if not save_path:
                self.statusBar().showMessage("Save cancelled.")
                return

            export_airfoil_data(
                save_path,
                coords,
                name=default_name,
                precision=max(6, self._airfoil.precision),
            )

            self.statusBar().showMessage(f"Saved: {save_path}")

        except Exception as e:
            self.statusBar().showMessage(f"Save failed: {e}")
            self._play_error_chime()

    def _on_save_spline(self):
        """Save spline control points."""
        if self._fit_result is None or self._airfoil is None:
            self.statusBar().showMessage("No fitted spline to save.")
            return

        try:
            base_name = self._airfoil.filename or self._airfoil.name
            header_name = self._airfoil.name or base_name
            degree = self._fit_result.degree

            default_path = os.path.join(
                QDir.homePath(),
                f"{base_name}.deg{degree}.spline.dat"
            )

            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save spline control points",
                default_path,
                "Spline data (*.spline.dat)",
            )

            if not save_path:
                self.statusBar().showMessage("Save cancelled.")
                return

            try:
                continuity = self._get_spline_config().junction_specs()
            except ValueError:
                continuity = None
            export_spline_data(
                save_path,
                degree,
                self._fit_result.section_control_points,
                name=header_name,
                precision=12,
                section_knots=[self._fit_result.knots(i).tolist()
                               for i in range(len(self._fit_result.section_control_points))],
                continuity=continuity,
            )

            self.statusBar().showMessage(f"Saved: {save_path}")

        except Exception as e:
            self.statusBar().showMessage(f"Save failed: {e}")
            self._play_error_chime()

    def _on_export_cad(self):
        """Export the exact B-spline curves to STEP, IGES or DXF."""
        if self._fit_result is None or self._airfoil is None:
            self.statusBar().showMessage("No fitted spline to export.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Export CAD")
        form = dialog_form(dialog)
        combo_format = QComboBox()
        combo_format.addItem("STEP AP214 (*.step)", userData=".step")
        combo_format.addItem("IGES 5.3 (*.igs)", userData=".igs")
        combo_format.addItem("DXF R2000 (*.dxf)", userData=".dxf")
        combo_format.setCurrentIndex(self._settings.value("cad_format", 0, type=int))
        add_form_row(form, "Format", combo_format)
        seg_units = SegmentedControl()
        for units in ("mm", "m", "inch"):
            seg_units.addItem(units, userData=units)
        units = self._settings.value("cad_units", "mm")
        seg_units.setCurrentIndex(max(0, seg_units.findData(units)))
        add_form_row(form, "Units", seg_units)
        spin_scale = ScientificSpinBox()
        spin_scale.setRange(1e-6, 1e9)
        spin_scale.setValue(self._settings.value("cad_scale", 1.0, type=float))
        spin_scale.setToolTip(
            "Coordinates are multiplied by this factor, for example the chord length.")
        add_form_row(form, "Scale", spin_scale)
        check_merge = QCheckBox("Merge sections into one curve")
        check_merge.setChecked(self._settings.value("cad_merge", False, type=bool))
        check_merge.setToolTip(
            "Write the whole chain as a single B-spline (junctions become knots).\n"
            "Otherwise every section is its own curve."
        )
        add_form_row(form, "Sections", check_merge)
        check_te = QCheckBox("Close a blunt trailing edge with a line")
        check_te.setChecked(self._settings.value("cad_close_te", True, type=bool))
        add_form_row(form, "Trailing edge", check_te)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.addButton("Export...", QDialogButtonBox.ButtonRole.AcceptRole).setDefault(True)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.statusBar().showMessage("Export cancelled.")
            return

        suffix = combo_format.currentData()
        self._settings.setValue("cad_format", combo_format.currentIndex())
        self._settings.setValue("cad_units", seg_units.currentData())
        self._settings.setValue("cad_scale", spin_scale.value())
        self._settings.setValue("cad_merge", check_merge.isChecked())
        self._settings.setValue("cad_close_te", check_te.isChecked())

        base_name = self._airfoil.filename or self._airfoil.name or "airfoil"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export CAD", os.path.join(QDir.homePath(), base_name + suffix),
            combo_format.currentText(),
        )
        if not save_path:
            self.statusBar().showMessage("Export cancelled.")
            return
        try:
            path = self.export_cad(save_path, units=seg_units.currentData(),
                                   scale=spin_scale.value(), merge=check_merge.isChecked(),
                                   close_trailing_edge=check_te.isChecked())
            self.statusBar().showMessage(f"Exported: {path}")
        except Exception as e:
            self.statusBar().showMessage(f"Export failed: {e}")
            self._play_error_chime()

    def export_cad(self, path: str, units: str = "mm", scale: float = 1.0,
                   merge: bool = False, close_trailing_edge: bool = True) -> str:
        """Write the current fit to ``path`` (format from the suffix)."""
        from bspline_varpro import export
        from bspline_varpro.airfoil import trailing_edge_segment

        curves = scipy_curves(self._fit_result)
        if merge:
            curves = [export.merge_curves(curves)]
        if close_trailing_edge:
            segment = trailing_edge_segment(curves)
            if segment is not None:
                curves.append(segment)
        name = self._airfoil.name or self._airfoil.filename or "airfoil"
        return str(export.write(path, curves, name=name, units=units, scale=scale))

    def _on_fit_report(self):
        """Show deviation, curvature and junction measurements of the current fit."""
        if self._fit_result is None or self._airfoil is None:
            self.statusBar().showMessage("No fitted spline.")
            return
        try:
            report = self._fit_result.report
            if report is None:
                self.statusBar().showMessage("Measuring the fit...")
                fit_result, airfoil = self._fit_result, self._airfoil
                report = run_in_background(lambda: fit_report(
                    airfoil.sections, fit_result, precision=airfoil.precision))
                fit_result.report = report
            elif "max_thickness" not in report:
                # A report from before thickness and camber were measured
                report.update(shape_report(self._fit_result))
            title = self._airfoil.name or self._airfoil.filename or "Fit report"
            shape = airfoil_shape(self._fit_result)
            chart = None if shape is None else shape_plot(shape)
            show_document(self, "Fit report", format_report_html(report, title), 560,
                          520 if chart is None else 720, extra=chart)
        except Exception as e:
            self.statusBar().showMessage(f"Report failed: {e}")
            self._play_error_chime()

    def _update_knots(self):
        """Show or hide the knot markers."""
        self._plot_manager.show_knots = self._check_knots.isChecked()
        self._plot_manager.refresh_knots(self._fit_result)

    def _play_completion_chime(self):
        """Play a chime sound to indicate optimization completion."""
        _play_chime("success")

    def _play_error_chime(self):
        """Play a chime sound to indicate an error occurred."""
        _play_chime("error")


def _play_chime(kind: str) -> None:
    """Play a chime on its own thread: loading the audio can take a second."""
    def play():
        try:
            import chime
            chime.theme('material')
            getattr(chime, kind)(sync=True)
        except Exception:
            pass  # Silently ignore if chime not installed or audio fails

    threading.Thread(target=play, name="chime", daemon=True).start()
