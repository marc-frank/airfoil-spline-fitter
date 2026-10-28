"""
File Actions
============

Open, save, export, and report actions for the main window.
"""

from __future__ import annotations

import os
import threading

from PyQt6.QtCore import QDir
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QPushButton,
    QVBoxLayout,
)

from ..analysis import fit_report, format_report_html, scipy_curves
from ..geometry import create_airfoil_data
from ..io import export_airfoil_data, export_spline_data, parse_airfoil_data
from .background import run_in_background
from .theme import text_browser


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
            "Open Airfoil .dat",
            QDir.homePath(),
            "Airfoil Data (*.dat)",
        )
        if path:
            self._load_airfoil(path)

    def _on_view_report(self):
        """Open the optimizer benchmark report in a rendered markdown viewer."""
        # Find report path relative to this module
        module_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        report_path = os.path.join(module_dir, "docs", "optimizer_benchmark_report.md")

        if not os.path.exists(report_path):
            self.statusBar().showMessage(f"Report not found: {report_path}")
            return

        try:
            with open(report_path, encoding="utf-8") as f:
                markdown_text = f.read()

            # Convert markdown to HTML
            html = self._markdown_to_html(markdown_text)

            # Create and show dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Optimizer Benchmark Report")
            dialog.resize(900, 700)

            layout = QVBoxLayout(dialog)

            browser = text_browser()
            browser.setOpenExternalLinks(True)
            browser.setHtml(html)
            layout.addWidget(browser)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)

            dialog.exec()

        except Exception as e:
            self.statusBar().showMessage(f"Failed to open report: {e}")
            self._play_error_chime()

    def _markdown_to_html(self, md: str) -> str:
        """Convert markdown to HTML for display in QTextBrowser."""
        import re

        # Start with basic HTML structure
        lines = md.split("\n")
        html_lines = []
        in_code_block = False
        in_table = False

        for line in lines:
            # Code blocks
            if line.startswith("```"):
                if in_code_block:
                    html_lines.append("</pre>")
                    in_code_block = False
                else:
                    html_lines.append("<pre>")
                    in_code_block = True
                continue

            if in_code_block:
                html_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
                continue

            # Tables
            if "|" in line and not line.strip().startswith("#"):
                cells = [c.strip() for c in line.split("|")]
                cells = [c for c in cells if c]  # Remove empty from edges

                # Skip separator rows
                if cells and all(set(c) <= set("-: ") for c in cells):
                    continue

                if not in_table:
                    html_lines.append('<table width="100%" cellspacing="0">')
                    in_table = True
                    # First row is header
                    html_lines.append("<tr>")
                    for cell in cells:
                        html_lines.append(f"<th>{self._inline_markdown(cell)}</th>")
                    html_lines.append("</tr>")
                else:
                    html_lines.append("<tr>")
                    for cell in cells:
                        html_lines.append(f"<td>{self._inline_markdown(cell)}</td>")
                    html_lines.append("</tr>")
                continue
            elif in_table:
                html_lines.append("</table>")
                in_table = False

            # Headings
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                level = min(level, 6)
                text = line.lstrip("# ").strip()
                sizes = {1: "24px", 2: "20px", 3: "16px", 4: "14px", 5: "12px", 6: "11px"}
                html_lines.append(f'<h{level} style="font-size: {sizes.get(level, "14px")}; margin: 15px 0 10px 0;">{self._inline_markdown(text)}</h{level}>')
                continue

            # Horizontal rules
            if line.strip() in ("---", "***", "___"):
                html_lines.append("<hr>")
                continue

            # Lists
            list_match = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
            if list_match:
                indent = len(list_match.group(1))
                content = list_match.group(3)
                marker = "\u2022" if list_match.group(2) in "-*+" else list_match.group(2)
                padding = indent * 10 + 20
                html_lines.append(f'<div style="padding-left: {padding}px; margin: 3px 0;">{marker} {self._inline_markdown(content)}</div>')
                continue

            # Empty lines
            if not line.strip():
                html_lines.append("<br>")
                continue

            # Regular paragraphs
            html_lines.append(f"<p style=\"margin: 5px 0;\">{self._inline_markdown(line)}</p>")

        if in_table:
            html_lines.append("</table>")

        body = "\n".join(html_lines)
        # Styled by the theme's text browser (theme.HTML_CSS)
        return f"<html><body>{body}</body></html>"

    def _inline_markdown(self, text: str) -> str:
        """Convert inline markdown (bold, italic, code, links) to HTML."""
        import re
        # Code (must come first to avoid conflicts)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        # Bold
        text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
        # Italic
        text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
        # Superscript for scientific notation (×10⁵ etc already in Unicode)
        return text

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
                self.statusBar().showMessage("Failed to parse airfoil data - no coordinates found")
                self._play_error_chime()
                return

            # Create airfoil data object with current section count
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
                f"Loaded: {display_name} ({n_points} pts, {len(section_sizes)} sections: {section_sizes})"
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
                self, "Save Fitted Airfoil",
                "Points per side (cosine spacing, clustered at LE and TE):",
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
                "Save Fitted Airfoil",
                default_path,
                "Airfoil Data (*.dat)",
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
                "Save Spline Control Points",
                default_path,
                "Spline Data (*.spline.dat)",
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
        form = QFormLayout(dialog)
        combo_format = QComboBox()
        combo_format.addItem("STEP AP214 (*.step)", userData=".step")
        combo_format.addItem("IGES 5.3 (*.igs)", userData=".igs")
        combo_format.addItem("DXF R2000 (*.dxf)", userData=".dxf")
        combo_format.setCurrentIndex(self._settings.value("cad_format", 0, type=int))
        form.addRow("Format", combo_format)
        combo_units = QComboBox()
        for units in ("mm", "m", "inch"):
            combo_units.addItem(units, userData=units)
        combo_units.setCurrentIndex(max(0, combo_units.findData(self._settings.value("cad_units", "mm"))))
        form.addRow("Units", combo_units)
        spin_scale = QDoubleSpinBox()
        spin_scale.setRange(1e-6, 1e9)
        spin_scale.setDecimals(6)
        spin_scale.setValue(self._settings.value("cad_scale", 1.0, type=float))
        spin_scale.setToolTip("Coordinates are multiplied by this factor (e.g. the chord length).")
        form.addRow("Scale (chord)", spin_scale)
        check_merge = QCheckBox("Merge sections into one curve")
        check_merge.setChecked(self._settings.value("cad_merge", False, type=bool))
        check_merge.setToolTip(
            "Write the whole chain as a single B-spline (junctions become knots).\n"
            "Otherwise every section is its own curve."
        )
        form.addRow("", check_merge)
        check_te = QCheckBox("Close blunt trailing edge with a line")
        check_te.setChecked(self._settings.value("cad_close_te", True, type=bool))
        form.addRow("", check_te)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.statusBar().showMessage("Export cancelled.")
            return

        suffix = combo_format.currentData()
        self._settings.setValue("cad_format", combo_format.currentIndex())
        self._settings.setValue("cad_units", combo_units.currentData())
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
            path = self.export_cad(save_path, units=combo_units.currentData(),
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
            dialog = QDialog(self)
            dialog.setWindowTitle("Fit Report")
            dialog.resize(560, 520)
            layout = QVBoxLayout(dialog)
            browser = text_browser()
            title = self._airfoil.name or self._airfoil.filename or "Fit report"
            browser.setHtml(format_report_html(report, title))
            layout.addWidget(browser)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            dialog.exec()
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
