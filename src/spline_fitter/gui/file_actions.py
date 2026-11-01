"""
File Actions
============

Open, save, export, and report actions for the main window.
"""

from __future__ import annotations

import math
import os
import threading

from PyQt6.QtCore import QDir
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QInputDialog,
)

from ..analysis import fit_report, format_report_html, scipy_curves
from ..geometry import create_airfoil_data
from ..io import export_airfoil_data, export_spline_data, parse_airfoil_data
from .background import run_in_background
from .widgets import ScientificSpinBox, SegmentedControl, add_form_row, dialog_form, show_document


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

    def _load_airfoil(self, path: str):
        """Load and display an airfoil file."""
        ext = os.path.splitext(path)[1].lower()
        if ext != ".dat":
            self.statusBar().showMessage(f"Unsupported file type: {ext}")
            return

        self.statusBar().showMessage(f"Loading: {os.path.basename(path)}...")
        QApplication.processEvents()

        try:
            filename_name, content_name, coords, max_precision = parse_airfoil_data(path)

            if not coords:
                self.statusBar().showMessage("No coordinates found in the file")
                self._play_error_chime()
                return

            # Create airfoil data object with current section count
            self._limit_sections(len(coords))
            num_sections = self._spin_sections.value()
            self._airfoil = create_airfoil_data(
                coords,
                filename=filename_name or os.path.basename(path),
                name=content_name or filename_name or "",
                precision=max_precision if isinstance(max_precision, int) else 6,
                num_sections=num_sections,
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
            self._update_auto_tolerance()

            display_name = content_name or filename_name
            n_points = len(coords)
            section_sizes = [len(sec) for sec in self._airfoil.sections]
            self.statusBar().showMessage(
                f"Loaded: {display_name} ({n_points} points, {len(section_sizes)} sections: {section_sizes})"
            )

        except Exception as e:
            self.statusBar().showMessage(f"Failed to load: {e}")
            self._play_error_chime()

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
        spin_scale.setRange(1e-6, math.inf)
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
            title = self._airfoil.name or self._airfoil.filename or "Fit report"
            show_document(self, "Fit report", format_report_html(report, title), 560, 520)
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
