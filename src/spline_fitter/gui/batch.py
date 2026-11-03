from __future__ import annotations

import os
import time
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

from PyQt6.QtCore import QDir, Qt
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
from ..optimization.models import (
    ContinuityMode,
    OptimizationConfig,
    OptimizerType,
    SplineConfig,
)
from .background import run_in_background
from .live import LiveProgress
from .theme import Size
from .widgets import add_form_row, dialog_form

if TYPE_CHECKING:
    from .ui import SplineFitterWindow


# The settings of a new window: their outputs keep the plain name
# <file>.deg<degree>.spline.dat, as before other settings were named
DEFAULT_SPLINE = SplineConfig(continuity=ContinuityMode.G3, geometric_continuity=True)
DEFAULT_OPTIMIZATION = OptimizationConfig(
    knot_placement="uniform", grow_control_points=False)


def _num(value: float) -> str:
    """A number for a file name: short, the same in every locale; small
    ones (tolerances) as ``1e-4`` or ``2.5e-5``."""
    if value and abs(value) < 1e-2:
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    return f"{value:.3g}"


def output_name(stem: str, config: SplineConfig, opt_config: OptimizationConfig,
                auto_tolerance: bool = False) -> str:
    """The batch output file of ``stem`` fitted with these settings.

    The degree is always named; every other setting only when it differs
    from a new window's, so two runs with different settings never write
    the same file: ``naca0012.deg5.n30.s3.G2-C1.tol1e-4.spline.dat``.
    """
    d_spline, d_opt = DEFAULT_SPLINE, DEFAULT_OPTIMIZATION
    parts = [stem, f"deg{config.degree}"]

    # Curve
    count_used = opt_config.tolerance is None or not opt_config.grow_control_points
    if count_used and config.num_control_points != d_spline.num_control_points:
        parts.append(f"n{config.num_control_points}")
    if config.num_sections != d_spline.num_sections:
        parts.append(f"s{config.num_sections}")
    specs = config.junction_specs()
    if specs and specs != [d_spline.junction_specs()[0]] * len(specs):
        parts.append("-".join(specs) if len(set(specs)) > 1 else specs[0])

    # Method
    o = opt_config
    if o.optimizer != d_opt.optimizer:
        parts.append("lsq" if o.optimizer == OptimizerType.DIRECT_LSQ else o.optimizer.value)
        if o.parameterization != d_opt.parameterization:
            parts.append(o.parameterization.replace("_", "-"))
    if o.tolerance is not None:
        parts.append("tol-auto" if auto_tolerance else f"tol{_num(o.tolerance)}")
    if o.knot_placement != d_opt.knot_placement:
        parts.append(f"knots-{o.knot_placement}")
    if o.grow_control_points != d_opt.grow_control_points:
        parts.append("grow" if o.grow_control_points else "no-grow")
    if o.smoothing_weight != d_opt.smoothing_weight:
        parts.append(f"smooth{_num(o.smoothing_weight)}")
    if o.smoothing_factor != d_opt.smoothing_factor and o.smoothing_factor is not None:
        parts.append(f"sfactor{_num(o.smoothing_factor)}")
    if o.max_points_per_section != d_opt.max_points_per_section:
        parts.append(f"sub{o.max_points_per_section}")
    if o.optimize_le_position != d_opt.optimize_le_position:
        parts.append("free-le")
    if o.early_stop_patience != d_opt.early_stop_patience:
        parts.append(f"patience{o.early_stop_patience}")
    if o.relative_tolerance != d_opt.relative_tolerance:
        parts.append(f"rtol{_num(o.relative_tolerance)}")
    return ".".join(parts) + ".spline.dat"


class BatchController:
    """Handles batch conversion dialog and batch processing operations."""

    def __init__(self, window: SplineFitterWindow):
        self._window = window
        self._batch_src_dir = ""
        self._batch_dst_dir = ""

    def on_batch_dialog(self):
        """Open the batch-conversion dialog (File -> Batch Convert)."""
        dialog = QDialog(self._window)
        # Freed on close, not kept as a child of the window
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
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
        try:
            self._convert(files, config, opt_config)
        finally:
            # Back to the window's own airfoil and fit
            w = self._window
            w._set_fitting_ui(False)
            # Without an airfoil of its own the window shows none: not the
            # batch's last file
            coords = w._airfoil.coordinates if w._airfoil is not None else []
            w._plot_manager.plot_airfoil(coords)
            if w._airfoil is not None and w._fit_result is not None:
                w._plot_manager.plot_fitted_spline(w._fit_result)
                w._refresh_curvature_comb()

    def _convert(self, files: list[str], config, opt_config) -> None:
        """Fit and export every file (UI locked), and report how it went."""
        start_time = time.perf_counter()
        done = 0
        failed = 0
        skipped = 0

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
                auto_tolerance = self._window._check_auto_tolerance.isChecked()
                out_path = os.path.join(self._batch_dst_dir, output_name(
                    airfoil.filename or airfoil.name, config, opt_config, auto_tolerance))

                if os.path.exists(out_path):
                    skipped += 1
                    self._window.statusBar().showMessage(f"[{idx}/{len(files)}] Skip: {out_path}")
                    QApplication.processEvents()
                    continue

                file_opt_config = opt_config
                if opt_config.tolerance is not None and auto_tolerance:
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
            msg = (f"Batch cancelled after {format_duration(elapsed)}. "
                   f"Completed {done}/{len(files)}, failed {failed}.")
        else:
            msg = f"Batch complete: {done} files in {format_duration(elapsed)} ({failed} failed)"
        if skipped:
            # Files fitted with these settings before: say so, or a rerun
            # would seem to do nothing
            msg += (f" | Skipped {skipped} already in the target folder "
                    "with these settings")
        self._window.statusBar().showMessage(msg)
