from __future__ import annotations

import os
import time
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QDir
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..analysis import default_tolerance
from ..formatting import format_duration, format_error
from ..geometry import compute_optimization_bounds, create_airfoil_data
from ..io import export_spline_data, parse_airfoil_data
from ..optimization import (
    construct_control_points,
    get_evaluation_count,
    reset_iteration_tracking,
    run_optimization,
)
from .background import run_in_background

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

                # Progress for the GUI thread: the callback runs on the worker
                # thread and must not touch widgets
                last_error = [float("nan")]

                def batch_callback(x: np.ndarray, error: float, last_error=last_error) -> bool:
                    if self._window._cancel_requested:
                        return True
                    last_error[0] = error
                    return False

                def batch_tick(idx=idx, fname=fname, batch_start=batch_start,
                               last_error=last_error):
                    """Show the progress of the running file (GUI thread)."""
                    self._window.statusBar().showMessage(
                        f"[{idx}/{len(files)}] {fname}: {get_evaluation_count()} evals, "
                        f"err \u02dc {format_error(last_error[0])}, "
                        f"t={format_duration(time.perf_counter() - batch_start)}"
                    )

                # Get param optimizer settings
                param_opt_outer = self._window._get_param_opt_settings()
                orth_local_bounds = self._window._check_orth_lsq_local_bounds.isChecked()

                def batch_work(airfoil=airfoil, file_opt_config=file_opt_config, bounds=bounds,
                               batch_callback=batch_callback, param_opt_outer=param_opt_outer,
                               orth_local_bounds=orth_local_bounds):
                    """Fit one file (worker thread)."""
                    return run_optimization(
                        config,
                        file_opt_config,
                        airfoil,
                        fairness_weights,
                        bounds,
                        callback=batch_callback,
                        param_opt_outer_optimizer=param_opt_outer,
                        orth_lsq_use_local_bounds=orth_local_bounds,
                        should_stop=lambda: self._window._cancel_requested,
                    )

                # Run optimization off the GUI thread
                result = run_in_background(batch_work, batch_tick)

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
