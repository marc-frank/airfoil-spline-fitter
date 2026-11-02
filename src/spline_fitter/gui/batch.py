from __future__ import annotations

import os
import time
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from PyQt6.QtCore import QDir
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from ..analysis import default_tolerance
from ..formatting import format_duration, format_error
from ..geometry import create_airfoil_data
from ..io import export_spline_data, parse_airfoil_data
from ..optimization import (
    OptimizerErrorLogger,
    get_evaluation_count,
    reset_iteration_tracking,
    run_optimization,
)
from .background import run_in_background
from .live import LiveProgress
from .theme import Size
from .widgets import add_form_row, dialog_form

if TYPE_CHECKING:
    from .ui import SplineFitterWindow


class BatchController:
    """Handles batch conversion dialog and batch processing operations."""

    def __init__(self, window: SplineFitterWindow):
        self._window = window
        self._batch_src_dir = ""
        self._batch_dst_dir = ""

    def on_batch_dialog(self):
        """Open the batch-conversion dialog (File -> Batch Convert)."""
        dialog = QDialog(self._window)
        dialog.setWindowTitle("Batch convert")
        dialog.setMinimumWidth(Size.DIALOG_MIN_WIDTH)
        form = dialog_form(dialog)
        note = QLabel("Fit every .dat file in the source folder with the current settings\n"
                      "and save each as .spline.dat in the target folder.")
        note.setObjectName("Muted")
        form.addRow(note)

        def folder_row(text: str, attr: str):
            """A folder path and the button that chooses it."""
            path = QLabel(getattr(self, attr) or "Not chosen")
            path.setObjectName("Muted")
            path.setWordWrap(True)
            button = QPushButton("Choose...")

            def choose():
                p = QFileDialog.getExistingDirectory(dialog, f"{text} folder", QDir.homePath())
                if p:
                    setattr(self, attr, p)
                    path.setText(p)
            button.clicked.connect(choose)
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(Size.GAP)
            lay.addWidget(path, stretch=1)
            lay.addWidget(button)
            add_form_row(form, text, row)

        folder_row("Source", "_batch_src_dir")
        folder_row("Target", "_batch_dst_dir")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.addButton("Convert", QDialogButtonBox.ButtonRole.AcceptRole).setDefault(True)

        def _do_convert():
            dialog.accept()
            self._on_batch_convert()
        buttons.accepted.connect(_do_convert)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        dialog.exec()

    def _on_batch_convert(self):
        """Run batch conversion."""
        # Validate directories
        if not self._batch_src_dir or not os.path.isdir(self._batch_src_dir):
            self._window.statusBar().showMessage("Choose a source folder first.")
            return
        if not self._batch_dst_dir or not os.path.isdir(self._batch_dst_dir):
            self._window.statusBar().showMessage("Choose a target folder first.")
            return

        # Gather .dat files
        files = [
            f for f in sorted(os.listdir(self._batch_src_dir))
            if f.lower().endswith('.dat') and '.spline.' not in f.lower()
        ]

        if not files:
            self._window.statusBar().showMessage("No .dat files in the source folder.")
            return

        # Get configuration
        try:
            config = self._window._get_spline_config()
            opt_config = self._window._get_optimization_config()
        except ValueError as e:
            self._window.statusBar().showMessage(str(e))
            return

        self._window._cancel_requested = False
        self._window._set_fitting_ui(True)

        start_time = time.perf_counter()
        done = 0
        failed = 0

        for idx, fname in enumerate(files, 1):
            if self._window._cancel_requested:
                break

            fpath = os.path.join(self._batch_src_dir, fname)

            try:
                # Parse airfoil
                filename_name, content_name, coords, max_precision = parse_airfoil_data(fpath)

                if not coords:
                    raise RuntimeError("Failed to parse")

                airfoil = create_airfoil_data(
                    coords,
                    filename=filename_name or fname,
                    name=content_name or filename_name or "",
                    precision=max_precision if isinstance(max_precision, int) else 6,
                    num_sections=config.num_sections,
                )

                # Check for existing output
                base_name = airfoil.filename or airfoil.name
                out_path = os.path.join(
                    self._batch_dst_dir,
                    f"{base_name}.deg{config.degree}.spline.dat"
                )

                if os.path.exists(out_path):
                    self._window.statusBar().showMessage(f"[{idx}/{len(files)}] Skip: {out_path}")
                    QApplication.processEvents()
                    continue

                file_opt_config = opt_config
                if opt_config.tolerance is not None and self._window._check_auto_tolerance.isChecked():
                    file_opt_config = dc_replace(
                        opt_config, tolerance=default_tolerance(airfoil.coordinates, airfoil.precision),
                    )

                # Reset tracking
                reset_iteration_tracking()

                # Status update; the file's points replace the plot
                batch_start = time.perf_counter()
                self._window.statusBar().showMessage(f"[{idx}/{len(files)}] Fitting {fname}...")
                plots = self._window._plot_manager
                plots.plot_airfoil(airfoil.coordinates)
                self._window._fitting_controller._reset_loss_plot()
                QApplication.processEvents()

                # Each file is shown live like a single fit
                live = LiveProgress(
                    self._window, config, file_opt_config,
                    OptimizerErrorLogger(run_name=airfoil.name or airfoil.filename),
                    batch_start, status_prefix=f"[{idx}/{len(files)}] {fname} | ",
                    airfoil=airfoil,
                )

                def batch_work(airfoil=airfoil, file_opt_config=file_opt_config, live=live):
                    """Fit one file (worker thread)."""
                    return run_optimization(
                        config,
                        file_opt_config,
                        airfoil,
                        callback=live.callback,
                        should_stop=lambda: self._window._cancel_requested,
                    )

                # Run optimization off the GUI thread
                result = run_in_background(batch_work, live.on_tick)
                live.show_loss()

                if result.x is None:
                    raise RuntimeError("Optimization failed")
                if self._window._cancel_requested:
                    break
                if not result.success:
                    raise ValueError("Optimizer did not converge: " + result.message)

                section_ctrl_pts = result.section_control_points

                # Show the finished fit
                plots.update_live_plot(
                    result.x, config,
                    preview=(section_ctrl_pts, result.section_knots, config.degree),
                )

                # Export
                header_name = airfoil.name or airfoil.filename
                export_spline_data(
                    out_path,
                    config.degree,
                    section_ctrl_pts,
                    name=header_name,
                    precision=12,
                    section_knots=result.section_knots,
                    continuity=config.junction_specs(),
                )

                done += 1
                batch_elapsed = time.perf_counter() - batch_start
                eval_count = get_evaluation_count()
                self._window.statusBar().showMessage(
                    f"[{idx}/{len(files)}] Done: {fname} | {eval_count} evaluations, "
                    f"error {format_error(result.fun)} | {format_duration(batch_elapsed)}"
                )
                QApplication.processEvents()

            except Exception as e:
                failed += 1
                self._window.statusBar().showMessage(f"[{idx}/{len(files)}] Failed: {fname} - {e}")
                self._window._play_error_chime()
                QApplication.processEvents()

        elapsed = time.perf_counter() - start_time

        if self._window._cancel_requested:
            msg = f"Batch cancelled after {format_duration(elapsed)}. Completed {done}/{len(files)}, failed {failed}."
        else:
            msg = f"Batch complete: {done} files in {format_duration(elapsed)} ({failed} failed)"

        # Back to the window's own airfoil and fit
        w = self._window
        if w._airfoil is not None:
            w._plot_manager.plot_airfoil(w._airfoil.coordinates)
            if w._fit_result is not None:
                w._plot_manager.plot_fitted_spline(w._fit_result)
                w._refresh_curvature_comb()

        self._window.statusBar().showMessage(msg)
        self._window._set_fitting_ui(False)
