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

from ..optimization.models import FitResult, SplineConfig
from ..spline import BSplineCurve2D, compute_signed_curvature
from .theme import Ink, Palette, Size, Z, brush, label_plot, pen
from .widgets import MAX_PLOT_POINTS

# Curvature comb: dense grid for tooth placement and higher derivatives, and the
# share of teeth placed by turning angle (the rest by arc length).
COMB_FINE_SAMPLES = 4000
COMB_TURNING_SHARE = 0.5


def _cumulative_trapezoid(f: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])


def _curvature_and_slope(derivs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Signed curvature and its arc-length derivative dκ/ds from curve derivatives."""
    dx, dy, d2x, d2y = derivs['dx'], derivs['dy'], derivs['d2x'], derivs['d2y']
    d3x = derivs.get('d3x', np.zeros_like(dx))
    d3y = derivs.get('d3y', np.zeros_like(dy))
    v = dx * dx + dy * dy
    a = dx * d2y - dy * d2x
    b = dx * d3y - dy * d3x
    c = dx * d2x + dy * d2y
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(v > 0, a / v ** 1.5, 0.0)
        slope = np.where(v > 0, (b * v - 3 * a * c) / v ** 3, 0.0)
    return kappa, slope


def _fine_grid(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, max(COMB_FINE_SAMPLES, 20 * n))


def comb_parameters(curve: BSplineCurve2D, n: int,
                    turning_share: float = COMB_TURNING_SHARE) -> np.ndarray:
    """Parameters of ``n`` comb teeth, more of them where the curve turns quickly.

    The teeth are evenly spaced in a blend of normalized arc length and
    normalized turning angle (the integral of |κ| ds), so a leading edge that
    turns through most of the total angle in a short distance gets a matching
    share of the teeth.
    """
    t = _fine_grid(n)
    d = curve.evaluate_derivatives(t, max_order=2, min_order=1)
    speed = np.hypot(d['dx'], d['dy'])
    kappa = compute_signed_curvature(d['dx'], d['dy'], d['d2x'], d['d2y'])
    s = _cumulative_trapezoid(speed, t)
    turn = _cumulative_trapezoid(np.abs(np.nan_to_num(kappa)) * speed, t)
    if s[-1] <= 0:
        return np.linspace(0.0, 1.0, n)
    measure = s / s[-1]
    if turn[-1] > 0:
        measure = (1 - turning_share) * measure + turning_share * turn / turn[-1]
    return np.interp(np.linspace(0.0, 1.0, n), measure, t)


def comb_values(curve: BSplineCurve2D, t: np.ndarray, quantity: int) -> np.ndarray:
    """Comb quantity at parameters ``t``: κ, dκ/ds, d²κ/ds² or d³κ/ds³ (0-3).

    κ and dκ/ds are exact. Higher orders are differentiated numerically on a
    dense grid, not between the (sparse) teeth.
    """
    if quantity <= 1:
        kappa, slope = _curvature_and_slope(curve.evaluate_derivatives(t, max_order=3))
        return kappa if quantity == 0 else slope
    fine = _fine_grid(len(t))
    d = curve.evaluate_derivatives(fine, max_order=3)
    _, val = _curvature_and_slope(d)
    s = _cumulative_trapezoid(np.hypot(d['dx'], d['dy']), fine)
    keep = np.concatenate([[True], np.diff(s) > 0])
    fine, s, val = fine[keep], s[keep], val[keep]
    for _ in range(quantity - 1):
        val = np.gradient(val, s, edge_order=2)
    return np.interp(t, fine, val)


class _Trace(pg.PlotDataItem):
    """The comparison trace. It keeps room for the marker around every point,
    and the marker is left out of the range, so moving the marker to another
    run never rescales the plot (and redraws its axes)."""

    def __init__(self, marker: pg.PlotDataItem, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._marker = marker

    def pixelPadding(self):
        return max(super().pixelPadding(), self._marker.scatter.pixelPadding())


def _joined(lines: list) -> tuple[np.ndarray, np.ndarray]:
    """Several (x, y) lines as one, a gap (NaN) between them."""
    xs, ys = [], []
    for x, y in lines:
        if xs:
            xs.append([np.nan])
            ys.append([np.nan])
        xs.append(np.asarray(x, dtype=float))
        ys.append(np.asarray(y, dtype=float))
    if not xs:
        return np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys)


def _settle(plot: pg.PlotWidget) -> None:
    """Auto-range the plot now: a range found while painting, as pyqtgraph
    does, paints the plot a second time for the axes it changed."""
    plot.getPlotItem().getViewBox().prepareForPaint()


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

        # The main plot's items: one per kind of thing drawn, holding every
        # section (a gap between sections), so a fit with another section
        # count, a run switched to or a live step only brings other data.
        # Cleared, an item holds no data
        self._data_item = self._new_item(self._data_style(), symbol='o',
                                         symbolSize=Size.DATA_SIZE)
        self._curve_item = self._new_item(self._curve_style())
        self._polygon_item = self._new_item(self._polygon_style())
        self._knot_item = self._new_item(self._knot_style())
        # The comb of the upper (even) and lower (odd) sections, by
        # (side, tips): the teeth, and the envelope through their tips
        self._comb_items = {(side, tips): self._new_item(self._comb_style(side, tips))
                            for side in ('upper', 'lower') for tips in (False, True)}
        # The curves of the last fit drawn, as (fit, curves)
        self._last_curves: tuple = (None, [])
        self.show_knots = False
        # (scale, density, quantity) of the curvature comb; None = hidden
        self.comb_settings: tuple[float, int, int] | None = None

        # Loss data
        self._loss_x_iters: list[float] = []
        self._loss_x_time: list[float] = []
        self._loss_y: list[float] = []

        # Loss curve item
        self._loss_curve = self._new_loss_curve()

        # Comparison plot items: the trace of all runs, the current one in ink.
        # The trace is a line through the runs, or only dots when several
        # runs share an x value
        self._comparison_connected = True
        self._comparison_marker = pg.PlotDataItem(
            [], [], symbol='o', symbolSize=Size.MARKER_SIZE, **self._marker_style())
        self._comparison_curve = _Trace(
            self._comparison_marker, [], [], symbol='o', symbolSize=Size.TRACE_POINT + 2,
            **self._comparison_style())
        comparison_plot.addItem(self._comparison_curve)
        comparison_plot.addItem(self._comparison_marker, ignoreBounds=True)

    def _new_item(self, style: dict, **symbol) -> pg.PlotDataItem:
        """An empty main-plot item in a style; non-finite values break its line."""
        style = dict(style)
        z = style.pop('z', Z.DATA)
        item = self._plot.plot([], [], connect='finite', **style, **symbol)
        item.setZValue(z)
        return item

    def _new_loss_curve(self):
        """The convergence trace: one dot per step."""
        return self._loss_plot.plot(
            [], [], symbol='o', symbolSize=Size.TRACE_POINT, **self._loss_style())

    # -----------------------------------------------------------------
    # Styles: the colors are read when called, so restyle() can reapply
    # them after a theme change
    # -----------------------------------------------------------------

    @staticmethod
    def _loss_style() -> dict:
        return dict(pen=None, symbolBrush=brush(Ink.TRACE), symbolPen=None)

    @staticmethod
    def _comparison_style(connected: bool = True) -> dict:
        return dict(pen=pen(Ink.TRACE, 1.5) if connected else None,
                    symbolBrush=brush(Ink.TRACE), symbolPen=None)

    @staticmethod
    def _marker_style() -> dict:
        return dict(pen=None, symbolBrush=brush(Ink.MARKER), symbolPen=pen(Palette.CANVAS, 2))

    @staticmethod
    def _data_style() -> dict:
        return dict(pen=None, symbolBrush=brush(Ink.DATA, 220), symbolPen=None)

    @staticmethod
    def _curve_style() -> dict:
        """The fitted curve: a solid line in ink."""
        return dict(pen=pen(Ink.CURVE, Size.CURVE_WIDTH), z=Z.CURVE)

    @staticmethod
    def _polygon_style() -> dict:
        """The control polygon: dashed construction line, hollow square vertices."""
        return dict(
            pen=pen(Ink.CONSTRUCTION, dashed=True, alpha=170),
            symbol='s', symbolSize=Size.CONTROL_SIZE,
            symbolBrush=brush(Palette.CANVAS), symbolPen=pen(Ink.CONSTRUCTION),
            z=Z.CONSTRUCTION,
        )

    @staticmethod
    def _knot_style() -> dict:
        """Knots: diamonds ringed in the canvas color."""
        return dict(pen=None, symbol='d', symbolSize=Size.KNOT_SIZE,
                    symbolBrush=brush(Ink.KNOTS), symbolPen=pen(Palette.CANVAS), z=Z.KNOTS)

    @staticmethod
    def _comb_style(which: str, tips: bool) -> dict:
        """Comb of a section: faint teeth, and the envelope through their tips.
        Suction side cool, pressure side warm ('upper_0', 'lower_1', ...)."""
        color = Ink.COMB_UPPER if which.startswith('upper') else Ink.COMB_LOWER
        if tips:
            return dict(pen=pen(color, 1.5), z=Z.COMB)
        return dict(pen=pen(color, alpha=Ink.COMB_TEETH_ALPHA), z=Z.COMB)

    def restyle(self) -> None:
        """Reapply the theme's colors to every plot item (after a theme change)."""
        def apply(item, style: dict) -> None:
            if item is None:
                return
            item.setPen(style.get('pen'))
            if 'symbolBrush' in style:
                item.setSymbolBrush(style['symbolBrush'])
                item.setSymbolPen(style['symbolPen'])

        apply(self._data_item, self._data_style())
        apply(self._curve_item, self._curve_style())
        apply(self._polygon_item, self._polygon_style())
        apply(self._knot_item, self._knot_style())
        for (side, tips), item in self._comb_items.items():
            apply(item, self._comb_style(side, tips))
        apply(self._loss_curve, self._loss_style())
        apply(self._comparison_curve, self._comparison_style(self._comparison_connected))
        apply(self._comparison_marker, self._marker_style())

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

    def set_comparison_data(self, x: list, y: list, connected: bool = True) -> None:
        """Update the comparison curve data; ``connected`` draws a line
        through the points, otherwise only the dots."""
        if connected != self._comparison_connected:
            self._comparison_connected = connected
            self._comparison_curve.setPen(self._comparison_style(connected)['pen'])
        self._comparison_curve.setData(x, y)
        _settle(self._comparison_plot)

    def set_comparison_marker(self, x: list, y: list) -> None:
        """Update the comparison marker data."""
        self._comparison_marker.setData(x, y)
        _settle(self._comparison_plot)

    def set_comparison_label(self, label: str) -> None:
        """Change the comparison plot X-axis label."""
        self.set_comparison_axes(label, "Error", log_x=False, log_y=True)

    def set_comparison_axes(self, x_label: str, y_label: str, log_x: bool, log_y: bool,
                            x_ticks: list[tuple[float, str]] | None = None) -> None:
        """Label and scale the comparison axes. ``x_ticks`` names the x
        positions of a column of text (one tick per value); None restores
        numeric ticks."""
        label_plot(self._comparison_plot, "bottom", x_label)
        label_plot(self._comparison_plot, "left", y_label)
        item = self._comparison_plot.getPlotItem()
        item.setLogMode(x=log_x, y=log_y)
        item.getAxis("bottom").setTicks([x_ticks] if x_ticks else None)
        item.enableAutoRange()

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
        """Clear the curve, control polygon, comb and knots (not the measured points)."""
        for item in (self._curve_item, self._polygon_item):
            item.setData([], [])
        self.clear_comb_items()
        self.clear_knot_items()

    def clear_knot_items(self) -> None:
        """Clear the knot markers."""
        self._knot_item.setData([], [])

    def refresh_knots(self, fit_result: FitResult | None) -> None:
        """Mark the curve points at the knots of every section (if enabled)."""
        if fit_result is None or not self.show_knots:
            self.clear_knot_items()
            return
        self._draw_knots(self._curves(fit_result))

    def _draw_knots(self, curves: list) -> None:
        """Mark the curve points at the knots of the given curves."""
        try:
            xs, ys = [], []
            for curve in curves:
                knots = np.unique(curve.knot_vector)
                kx, ky = curve.evaluate_array(knots)
                xs.extend(kx.tolist())
                ys.extend(ky.tolist())
            self._knot_item.setData(xs, ys)
        except Exception:
            pass

    def clear_comb_items(self) -> None:
        """Clear the curvature comb."""
        for item in self._comb_items.values():
            item.setData([], [])

    def plot_airfoil(self, coords: list) -> None:
        """Plot airfoil coordinate points.

        Args:
            coords: List of (x, y) tuples.
        """
        self.clear_plot_items()
        self._data_item.setData([p[0] for p in coords], [p[1] for p in coords])
        self._plot.autoRange()

    def reset_loss_plot(self, x_label: str, log_x: bool, log_y: bool) -> None:
        """Clear the loss plot and reconfigure axes."""
        self._loss_x_iters = []
        self._loss_x_time = []
        self._loss_y = []
        self._loss_plot.clear()
        self._loss_plot.getPlotItem().setLogMode(x=log_x, y=log_y)
        self._loss_plot.setLabel("bottom", x_label)
        self._loss_curve = self._new_loss_curve()

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
        _settle(self._loss_plot)

    def update_live_plot(
        self,
        variables: np.ndarray,
        config: SplineConfig,
        preview=None,
    ) -> None:
        """Update the live preview during optimization.

        Args:
            variables: Flattened control points, [x0, y0, x1, y1, ...] per section
            config: Spline configuration
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
            else:
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

            curves = [
                BSplineCurve2D(degree, [(float(p[0]), float(p[1])) for p in ctrl_pts],
                               None if section_knots is None else section_knots[i])
                for i, ctrl_pts in enumerate(section_ctrl_pts)]
            self._draw_spline(curves)

            # Comb and knot markers of the curve in progress, if enabled
            if self.comb_settings is not None:
                self._draw_combs(curves, *self.comb_settings)
            if self.show_knots:
                self._draw_knots(curves)

        except Exception:
            pass

    def plot_fitted_spline(self, fit_result: FitResult) -> None:
        """Plot the final fitted splines.

        Args:
            fit_result: The optimization result containing control points.
        """
        try:
            self._draw_spline(self._curves(fit_result))
        except Exception:
            pass
        self.refresh_knots(fit_result)
        _settle(self._plot)

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

        self._draw_combs(self._curves(fit_result), scale, density, qty)
        _settle(self._plot)

    def _draw_combs(self, curves: list, scale: float, density: int, qty: int) -> None:
        """Draw the curvature comb of every section, alternating in color."""
        density = max(5, density)
        try:
            for start, side in enumerate(('upper', 'lower')):
                combs = [self._curvature_comb(curve, scale, density, qty)
                         for curve in curves[start::2]]
                self._comb_items[side, False].setData(*_joined([c[0] for c in combs]))
                self._comb_items[side, True].setData(*_joined([c[1] for c in combs]))
        except Exception:
            pass

    @staticmethod
    def _curvature_comb(
        curve: BSplineCurve2D,
        scale: float,
        density: int,
        quantity: int,
    ) -> tuple[tuple, tuple]:
        """The comb of a single curve: its teeth (base, tip, gap) and the
        envelope through their tips, each as (x, y)."""
        t = comb_parameters(curve, density)
        derivs = curve.evaluate_derivatives(t, max_order=1)
        x, y = derivs['x'], derivs['y']
        dx, dy = derivs['dx'], derivs['dy']
        val = comb_values(curve, t, quantity)

        # Compute normals
        mag = np.hypot(dx, dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            nx = np.where(mag > 1e-12, dy / mag, 0.0)
            ny = np.where(mag > 1e-12, -dx / mag, 0.0)

        # Comb tips
        cx = x + nx * val * scale
        cy = y + ny * val * scale

        # Build comb segments: base, tip, gap
        gap = np.full_like(x, np.nan)
        seg_x = np.column_stack([x, cx, gap]).ravel()
        seg_y = np.column_stack([y, cy, gap]).ravel()

        return (seg_x, seg_y), (cx, cy)

    def _draw_spline(self, curves: list) -> None:
        """Draw the curves and their control polygons."""
        self._curve_item.setData(*_joined([curve.sample(1000) for curve in curves]))
        self._polygon_item.setData(*_joined([np.array(curve.control_points).T
                                             for curve in curves]))

    def _curves(self, fit_result: FitResult) -> list:
        """The fit's curves, built once for its spline, knots and comb."""
        if self._last_curves[0] is not fit_result:
            self._last_curves = (fit_result, fit_result.curves())
        return self._last_curves[1]
