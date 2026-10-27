from __future__ import annotations
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING
import os
import time
import numpy as np
from PyQt6.QtCore import QDir
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog,
)
from ..optimization.models import SplineConfig
from ..optimization import (
    run_optimization, construct_control_points,
    reset_iteration_tracking, get_evaluation_count,
)
from ..geometry import create_airfoil_data, compute_optimization_bounds
from ..io import parse_airfoil_data, export_spline_data
from ..formatting import format_error, format_duration
from ..analysis import default_tolerance

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
        dialog.setWindowTitle("Batch Convert")
        dialog.setMinimumWidth(500)
        lay = QVBoxLayout(dialog)
        lay.addWidget(QLabel("Convert all .dat files in source to .spline.dat"))
        lay.addSpacing(8)

        btn_src = QPushButton("Choose Source Directory...")
        lay.addWidget(btn_src)
        lbl_src = QLabel(self._batch_src_dir or "Not set")
        lbl_src.setWordWrap(True)
        lay.addWidget(lbl_src)

        def _on_src():
            p = QFileDialog.getExistingDirectory(dialog, "Source directory", QDir.homePath())
            if p:
                self._batch_src_dir = p
                lbl_src.setText(p)
        btn_src.clicked.connect(_on_src)

        lay.addSpacing(4)
        btn_dst = QPushButton("Choose Target Directory...")
        lay.addWidget(btn_dst)
        lbl_dst = QLabel(self._batch_dst_dir or "Not set")
        lbl_dst.setWordWrap(True)
        lay.addWidget(lbl_dst)

        def _on_dst():
            p = QFileDialog.getExistingDirectory(dialog, "Target directory", QDir.homePath())
            if p:
                self._batch_dst_dir = p
                lbl_dst.setText(p)
        btn_dst.clicked.connect(_on_dst)

        lay.addSpacing(12)
        btn_row = QHBoxLayout()
        btn_convert = QPushButton("Convert")
        btn_convert.setDefault(True)
        def _do_convert():
            dialog.accept()
            self._on_batch_convert()
        btn_convert.clicked.connect(_do_convert)
        btn_row.addWidget(btn_convert)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.reject)
        btn_row.addWidget(btn_close)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        dialog.exec()

    def _on_batch_convert(self):
        """Run batch conversion."""
        # Validate directories
        if not self._batch_src_dir or not os.path.isdir(self._batch_src_dir):
            self._window.statusBar().showMessage("Set a valid source directory.")
            return
        if not self._batch_dst_dir or not os.path.isdir(self._batch_dst_dir):
            self._window.statusBar().showMessage("Set a valid target directory.")
            return

        # Gather .dat files
        files = [
            f for f in sorted(os.listdir(self._batch_src_dir))
            if f.lower().endswith('.dat') and '.spline.' not in f.lower()
        ]

        if not files:
            self._window.statusBar().showMessage("No .dat files found in source.")
            return

        # Get configuration
        try:
            config = self._window._get_spline_config()
            opt_config = self._window._get_optimization_config()
            fairness_weights = self._window._get_fairness_weights()
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

                # Compute bounds
                bounds = compute_optimization_bounds(
                    airfoil, config,
                    optimize_le_position=opt_config.optimize_le_position,
                )

                # Reset tracking
                reset_iteration_tracking()

                # Status update
                batch_start = time.perf_counter()
                self._window.statusBar().showMessage(f"[{idx}/{len(files)}] Fitting {fname}...")
                QApplication.processEvents()

                # Batch callback for progress updates
                batch_last_update = batch_start

                def batch_callback(x: np.ndarray, error: float) -> bool:
                    nonlocal batch_last_update
                    if self._window._cancel_requested:
                        return True

                    current_time = time.perf_counter()
                    if current_time - batch_last_update >= 1.0:  # Update every 1 second
                        batch_last_update = current_time
                        eval_count = get_evaluation_count()
                        batch_elapsed = current_time - batch_start
                        self._window.statusBar().showMessage(
                            f"[{idx}/{len(files)}] {fname}: {eval_count} evals, "
                            f"err \u02dc {format_error(error)}, t={format_duration(batch_elapsed)}"
                        )
                        QApplication.processEvents()
                    return False

                # Get param optimizer settings
                param_opt_outer = self._window._get_param_opt_settings()

                # Run optimization with progress callback
                result = run_optimization(
                    config,
                    file_opt_config,
                    airfoil,
                    fairness_weights,
                    bounds,
                    callback=batch_callback,
                    param_opt_outer_optimizer=param_opt_outer,
                    orth_lsq_use_local_bounds=self._window._check_orth_lsq_local_bounds.isChecked(),
                )

                if result.x is None:
                    raise RuntimeError("Optimization failed")
                if self._window._cancel_requested:
                    break
                if not result.success:
                    raise ValueError("Optimizer did not converge: " + result.message)

                # Construct control points for all sections
                section_ctrl_pts = result.section_control_points or construct_control_points(
                    result.x, config,
                    [list(sec) for sec in airfoil.sections],
                    optimize_le_position=opt_config.optimize_le_position,
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
                    f"[{idx}/{len(files)}] Done: {fname} | {eval_count} evals, "
                    f"err \u02dc {format_error(result.fun)}, t={format_duration(batch_elapsed)}"
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
            msg = f"Batch complete! {done} files in {format_duration(elapsed)} ({failed} failed)"

        self._window.statusBar().showMessage(msg)
        self._window._set_fitting_ui(False)
