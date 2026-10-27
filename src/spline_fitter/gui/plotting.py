"""
Plot Manager
============

Manages all pyqtgraph visualization for the spline fitter.

This module provides:
    - PlotManager: Centralized management of airfoil, loss, and comparison plots
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from ..optimization import construct_control_points
from ..optimization.models import FitResult, SplineConfig
from ..spline import BSplineCurve2D, compute_signed_curvature
from .widgets import MAX_PLOT_POINTS, Colors


class PlotManager:
    """Manages all pyqtgraph plot visualizations.

    Owns plot item state (curve_items, comb_items, loss data)
    and delegates all drawing to pyqtgraph.  No back-reference
    to the main window — receives data through method parameters.
    """

    def __init__(
        self,
        main_plot: pg.PlotWidget,
        loss_plot: pg.PlotWidget,
        comparison_plot: pg.PlotWidget,
    ):
        self._plot = main_plot
        self._loss_plot = loss_plot
        self._comparison_plot = comparison_plot

        # Plot item tracking
        self._curve_items: dict = {}
        self._comb_items: dict = {}
        self._knot_items: dict = {}
        self.show_knots = False

        # Loss data
        self._loss_x_iters: list[float] = []
        self._loss_x_time: list[float] = []
        self._loss_y: list[float] = []

        # Loss curve item
        self._loss_curve = loss_plot.plot(
            [], [], pen=None, symbol='o', symbolSize=3,
            symbolBrush=Colors.LOSS_POINTS, symbolPen=None,
        )

        # Comparison plot items
        self._comparison_curve = comparison_plot.plot(
            [], [], pen=pg.mkPen(color=(100, 200, 255), width=2),
            symbol='o', symbolSize=8,
            symbolBrush=(100, 200, 255), symbolPen=None,
        )
        self._comparison_marker = comparison_plot.plot(
            [], [], pen=None, symbol='o', symbolSize=14,
            symbolBrush=(255, 165, 0),
            symbolPen=pg.mkPen(color=(255, 255, 255), width=2),
        )

    # -----------------------------------------------------------------
    # Loss data management
    # -----------------------------------------------------------------

    @property
    def has_loss_data(self) -> bool:
        return bool(self._loss_x_iters) and bool(self._loss_y)

    def update_loss_data(
        self,
        iters: list[float],
        times: list[float],
        errors: list[float],
    ) -> None:
        """Replace the loss history data."""
        self._loss_x_iters = list(iters)
        self._loss_x_time = list(times)
        self._loss_y = [max(e, 1e-12) for e in errors]

    def append_loss_point(
        self,
        iteration: float,
        elapsed: float,
        error: float,
    ) -> None:
        """Append a single loss data point (used during live fitting)."""
        self._loss_x_iters.append(iteration)
        self._loss_x_time.append(elapsed)
        self._loss_y.append(max(error, 1e-12))

    # -----------------------------------------------------------------
    # Comparison plot helpers
    # -----------------------------------------------------------------

    def set_comparison_data(self, x: list, y: list) -> None:
        """Update the comparison curve data."""
        self._comparison_curve.setData(x, y)

    def set_comparison_marker(self, x: list, y: list) -> None:
        """Update the comparison marker data."""
        self._comparison_marker.setData(x, y)

    def set_comparison_label(self, label: str) -> None:
        """Change the comparison plot X-axis label."""
        self._comparison_plot.setLabel("bottom", label)
        self._comparison_plot.setLabel("left", "Error")
        self._comparison_plot.setTitle("")

    # -----------------------------------------------------------------
    # Loss plot helpers
    # -----------------------------------------------------------------

    def set_loss_log_scale(self, log_x: bool, log_y: bool) -> None:
        """Update the loss plot log/linear scale."""
        self._loss_plot.getPlotItem().setLogMode(x=log_x, y=log_y)

    def set_loss_x_label(self, label: str) -> None:
        """Update the loss plot X-axis label."""
        self._loss_plot.setLabel("bottom", label)

    # -----------------------------------------------------------------
    # Main plot operations
    # -----------------------------------------------------------------

    def clear_plot_items(self) -> None:
        """Remove all curve and control point items from the plot."""
        for key in list(self._curve_items.keys()):
            item = self._curve_items.pop(key)
            try:
                self._plot.removeItem(item)
            except Exception:
                pass

        for key in list(self._comb_items.keys()):
            item = self._comb_items.pop(key)
            try:
                self._plot.removeItem(item)
            except Exception:
                pass
        self.clear_knot_items()

    def clear_knot_items(self) -> None:
        """Remove the knot markers from the plot."""
        for key in list(self._knot_items.keys()):
            item = self._knot_items.pop(key)
            try:
                self._plot.removeItem(item)
            except Exception:
                pass

    def refresh_knots(self, fit_result: FitResult | None) -> None:
        """Mark the curve points at the knots of every section (if enabled)."""
        self.clear_knot_items()
        if fit_result is None or not self.show_knots:
            return
        try:
            xs, ys = [], []
            for i, curve in enumerate(fit_result.curves()):
                knots = np.unique(fit_result.knots(i))
                kx, ky = curve.evaluate_array(knots)
                xs.extend(kx.tolist())
                ys.extend(ky.tolist())
            self._knot_items['knots'] = self._plot.plot(
                xs, ys, pen=None, symbol='d', symbolSize=8,
                symbolBrush=Colors.KNOTS, symbolPen=None,
            )
        except Exception:
            pass

    def clear_comb_items(self) -> None:
        """Remove all curvature comb items from the plot."""
        for key in list(self._comb_items.keys()):
            item = self._comb_items.pop(key)
            try:
                self._plot.removeItem(item)
            except Exception:
                pass

    def plot_airfoil(self, coords: list) -> None:
        """Plot airfoil coordinate points.

        Args:
            coords: List of (x, y) tuples.
        """
        self._plot.clear()
        x = [p[0] for p in coords]
        y = [p[1] for p in coords]
        self._plot.plot(
            x, y,
            pen=None,
            symbol='o',
            symbolSize=5,
            symbolBrush=Colors.AIRFOIL_POINTS,
            symbolPen=None,
        )
        self._plot.autoRange()

    def reset_loss_plot(self, x_label: str, log_x: bool, log_y: bool) -> None:
        """Clear the loss plot and reconfigure axes."""
        self._loss_x_iters = []
        self._loss_x_time = []
        self._loss_y = []
        self._loss_plot.clear()
        self._loss_plot.getPlotItem().setLogMode(x=log_x, y=log_y)
        self._loss_plot.setLabel("bottom", x_label)
        self._loss_curve = self._loss_plot.plot(
            [], [],
            pen=None,
            symbol='o',
            symbolSize=3,
            symbolBrush=Colors.LOSS_POINTS,
            symbolPen=None,
        )

    def set_loss_curve_data(self, mode: str) -> None:
        """Update loss curve with data based on X-axis mode.

        Args:
            mode: Either "iters" or "time".
        """
        if not self._loss_y:
            return

        if mode == "time":
            x_data = self._loss_x_time
        else:  # iters (default)
            x_data = self._loss_x_iters

        y_data = self._loss_y

        if not x_data or not y_data:
            return

        # Downsample if too many points to avoid UI freeze
        n = len(x_data)
        if n > MAX_PLOT_POINTS:
            step = n // MAX_PLOT_POINTS
            x_plot = x_data[::step]
            y_plot = y_data[::step]
        else:
            x_plot = x_data
            y_plot = y_data

        # Ensure same length
        min_len = min(len(x_plot), len(y_plot))
        self._loss_curve.setData(x_plot[:min_len], y_plot[:min_len])

    def update_live_plot(
        self,
        variables: np.ndarray,
        config: SplineConfig,
        sections: list,
        optimize_le_position: bool = False,
        is_direct_control_points: bool = False,
        preview=None,
    ) -> None:
        """Update the live preview during optimization.

        Args:
            variables: Either optimization variables OR flattened control points
            config: Spline configuration
            sections: Airfoil sections (list of point lists)
            optimize_le_position: Whether LE position is being optimized
            is_direct_control_points: If True, variables are flattened control points
                                      (for algebraic methods like Orthogonal LSQ)
            preview: Optional ``(section_control_points, section_knots, degree)``
                from optimizers whose control-point counts and knots change
                (VarPro tolerance mode); takes precedence over ``variables``.
        """
        try:
            section_knots = None
            degree = config.degree
            # Build curves for all sections
            if preview is not None:
                section_ctrl_pts, section_knots, degree = preview
            elif is_direct_control_points:
                # Variables are flattened control points: [x0,y0,x1,y1,...] per section
                n_ctrl = config.num_control_points
                num_sections = config.num_sections
                section_ctrl_pts = []
                offset = 0
                for _ in range(num_sections):
                    pts = []
                    for i in range(n_ctrl):
                        x = float(variables[offset + 2*i])
                        y = float(variables[offset + 2*i + 1])
                        pts.append((x, y))
                    section_ctrl_pts.append(pts)
                    offset += 2 * n_ctrl
            else:
                section_ctrl_pts = construct_control_points(
                    variables, config,
                    [list(sec) for sec in sections],
                    optimize_le_position=optimize_le_position,
                )

            # Drop items of sections that no longer exist
            for key in [k for k in self._curve_items if k.startswith('live_')]:
                if int(key.rsplit('_', 1)[1]) >= len(section_ctrl_pts):
                    try:
                        self._plot.removeItem(self._curve_items.pop(key))
                    except Exception:
                        pass

            # Plot each section
            for section_idx, ctrl_pts in enumerate(section_ctrl_pts):
                ctrl_pts = [(float(p[0]), float(p[1])) for p in ctrl_pts]
                knots = None if section_knots is None else section_knots[section_idx]
                curve = BSplineCurve2D(degree, ctrl_pts, knots)

                # Spline curve
                ux, uy = curve.sample(1000)

                key = f'live_curve_{section_idx}'
                if key in self._curve_items:
                    try:
                        self._plot.removeItem(self._curve_items[key])
                    except Exception:
                        pass

                pen = pg.mkPen(color=Colors.SPLINE_CURVE, width=2)
                self._curve_items[key] = self._plot.plot(ux, uy, pen=pen)

                # Control polygon
                cx = [p[0] for p in ctrl_pts]
                cy = [p[1] for p in ctrl_pts]

                key = f'live_ctrl_{section_idx}'
                if key in self._curve_items:
                    try:
                        self._plot.removeItem(self._curve_items[key])
                    except Exception:
                        pass

                ctrl_pen = pg.mkPen(
                    color=Colors.CONTROL_POLYGON,
                    width=1,
                    style=Qt.PenStyle.DashLine,
                )
                self._curve_items[key] = self._plot.plot(
                    cx, cy,
                    pen=ctrl_pen,
                    symbol='o',
                    symbolSize=5,
                    symbolBrush=Colors.CONTROL_POLYGON,
                    symbolPen=None,
                )

        except Exception:
            pass

    def plot_fitted_spline(self, fit_result: FitResult) -> None:
        """Plot the final fitted splines.

        Args:
            fit_result: The optimization result containing control points.
        """
        # Clear live items
        self.clear_plot_items()

        try:
            # Plot each section
            curves = fit_result.curves()
            for section_idx, ctrl_pts in enumerate(fit_result.section_control_points):
                curve = curves[section_idx]

                # Spline curve
                sx, sy = curve.sample(1000)

                pen = pg.mkPen(color=Colors.SPLINE_CURVE, width=2)
                self._curve_items[f'final_curve_{section_idx}'] = self._plot.plot(sx, sy, pen=pen)

                # Control polygon
                cx = [p[0] for p in ctrl_pts]
                cy = [p[1] for p in ctrl_pts]

                ctrl_pen = pg.mkPen(
                    color=Colors.CONTROL_POLYGON,
                    width=1,
                    style=Qt.PenStyle.DashLine,
                )
                self._curve_items[f'final_ctrl_{section_idx}'] = self._plot.plot(
                    cx, cy,
                    pen=ctrl_pen,
                    symbol='o',
                    symbolSize=5,
                    symbolBrush=Colors.CONTROL_POLYGON,
                    symbolPen=None,
                )

        except Exception:
            pass
        self.refresh_knots(fit_result)

    # -----------------------------------------------------------------
    # Curvature Comb
    # -----------------------------------------------------------------

    def refresh_curvature_comb(
        self,
        fit_result: FitResult | None,
        scale: float,
        density: int,
        qty: int,
    ) -> None:
        """Refresh the curvature comb visualization.

        Args:
            fit_result: Current fit result (if any).
            scale: Comb scale factor.
            density: Number of sample points.
            qty: Displayed quantity (0=curvature, 1=first deriv, etc.).
        """
        if fit_result is None:
            return

        # Clear existing comb items
        self.clear_comb_items()

        density = max(5, density)

        try:
            # Draw comb for each section
            for section_idx, curve in enumerate(fit_result.curves()):
                # Use alternating colors for different sections
                which = 'upper' if section_idx % 2 == 0 else 'lower'
                self._draw_curvature_comb(f'{which}_{section_idx}', curve, scale, density, qty)

        except Exception:
            pass

    def _draw_curvature_comb(
        self,
        which: str,
        curve: BSplineCurve2D,
        scale: float,
        density: int,
        quantity: int,
    ) -> None:
        """Draw curvature comb for a single curve."""
        t = np.linspace(0.0, 1.0, density)
        derivs = curve.evaluate_derivatives(t, max_order=min(3, curve.degree))

        x = derivs['x']
        y = derivs['y']
        dx = derivs.get('dx', np.zeros_like(x))
        dy = derivs.get('dy', np.zeros_like(y))
        d2x = derivs.get('d2x', np.zeros_like(x))
        d2y = derivs.get('d2y', np.zeros_like(y))

        # Compute curvature
        kappa = compute_signed_curvature(dx, dy, d2x, d2y)

        # Compute arc length
        speed = np.hypot(dx, dy)
        dt = 1.0 / (density - 1) if density > 1 else 1.0
        ds = np.maximum(0.5 * (speed[1:] + speed[:-1]) * dt, 0.0)
        s = np.zeros_like(x)
        s[1:] = np.cumsum(ds)

        # Ensure strictly increasing s
        mask = np.diff(s) <= 1e-14
        if np.any(mask):
            fix = np.maximum(np.diff(s), 1e-12)
            s[1:] = s[0] + np.cumsum(fix)

        # Compute displayed quantity
        if quantity >= 1:
            val = np.gradient(kappa, s, edge_order=2)
        else:
            val = kappa

        if quantity >= 2:
            val = np.gradient(val, s, edge_order=2)

        if quantity >= 3:
            val = np.gradient(val, s, edge_order=2)

        # Compute normals
        mag = np.hypot(dx, dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            nx = np.where(mag > 1e-12, dy / mag, 0.0)
            ny = np.where(mag > 1e-12, -dx / mag, 0.0)

        # Comb tips
        cx = x + nx * val * scale
        cy = y + ny * val * scale

        # Build comb segments
        seg_x = []
        seg_y = []
        for x0, y0, x1, y1 in zip(x, y, cx, cy, strict=False):
            seg_x.extend([x0, x1, np.nan])
            seg_y.extend([y0, y1, np.nan])

        # Colors based on which key (handles 'upper_0', 'lower_1', etc.)
        if which.startswith('upper'):
            comb_color = Colors.COMB_UPPER
            tips_color = Colors.COMB_TIPS_UPPER
        else:
            comb_color = Colors.COMB_LOWER
            tips_color = Colors.COMB_TIPS_LOWER

        # Draw comb lines
        pen = pg.mkPen(color=comb_color, width=1)
        self._comb_items[f'comb_{which}'] = self._plot.plot(seg_x, seg_y, pen=pen)

        # Draw tips
        tips_pen = pg.mkPen(color=tips_color, width=1, style=Qt.PenStyle.DashLine)
        self._comb_items[f'tips_{which}'] = self._plot.plot(cx, cy, pen=tips_pen)
