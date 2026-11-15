//! The plots: the airfoil with its fit and overlays, convergence, the
//! comparison of runs, and the thickness and camber of a report.
//!
//! egui_plot supplies the frame, axes, grid and panning; every item is
//! painted through the plot's transform, so lines break at gaps, markers
//! have the Loft shapes and items stack in drawing order. Log axes plot
//! `log10` of the values with decade labels. The wheel zooms about the
//! cursor in a short glide.

use std::time::Instant;

use eframe::egui::{self, Color32, Event, MouseWheelUnit, Pos2, Rect, Response, Shape, Stroke, Ui, Vec2 as EVec2};
use egui_plot::{GridMark, Plot, PlotBounds, PlotPoint, PlotTransform};
use spline_fitter::analysis::{foot_points, FootPoints};
use spline_fitter::bspline_varpro::basis::{distinct, linspace};
use spline_fitter::formatting::number;
use spline_fitter::{BSpline, Vec2};

use crate::overlays::{comb, deviation, Deviation, ParityStrokes, Strokes};
use crate::theme::{alpha, size, Palette, INK, TEETH_ALPHA};

/// A running zoom glide.
struct Glide {
    total: f64,
    done: f64,
    start: Instant,
    center: PlotPoint,
}

/// View state of one plot: the zoom glide, a pending range and whether it
/// follows its data.
pub struct View {
    glide: Option<Glide>,
    pending: Option<PlotBounds>,
    /// Fit the range to the data when it changes, until the user zooms or pans.
    pub follow: bool,
}

impl Default for View {
    fn default() -> Self {
        View { glide: None, pending: None, follow: true }
    }
}

impl View {
    /// Shows `bounds` (with a margin) on the next frame.
    pub fn fit(&mut self, bounds: Option<PlotBounds>) {
        if let Some(b) = bounds {
            let (min, max) = (b.min(), b.max());
            let pad = [(max[0] - min[0]).max(1e-12) * 0.05, (max[1] - min[1]).max(1e-12) * 0.05];
            self.pending = Some(PlotBounds::from_min_max([min[0] - pad[0], min[1] - pad[1]], [max[0] + pad[0], max[1] + pad[1]]));
        }
        self.glide = None;
    }
}

/// Bounds of points (in plot coordinates), skipping non-finite ones.
pub fn bounds_of(points: impl IntoIterator<Item = [f64; 2]>) -> Option<PlotBounds> {
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for p in points {
        if p[0].is_finite() && p[1].is_finite() {
            for k in 0..2 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
    }
    (min[0] <= max[0]).then(|| PlotBounds::from_min_max(min, max))
}

/// Axis setup of a plot.
pub struct Axes<'a> {
    pub x_label: &'a str,
    pub y_label: &'a str,
    pub log_x: bool,
    pub log_y: bool,
    /// Text labels at positions of the x axis (a categorical axis).
    pub x_ticks: Option<&'a [(f64, String)]>,
    pub equal_aspect: bool,
}

/// Tick text: the display decimal point; on a log axis the decades.
fn tick_label(mark: GridMark, log: bool) -> String {
    if log {
        let rounded = mark.value.round();
        if (mark.value - rounded).abs() > 1e-9 || mark.step_size < 0.99 {
            return String::new();
        }
        return number(10f64.powf(rounded), "g");
    }
    number(mark.value, "g")
}

/// Shows a plot frame and returns its transform; `paint` draws the items.
pub fn show_plot(
    ui: &mut Ui,
    p: &Palette,
    id: &str,
    view: &mut View,
    axes: &Axes,
    motion_ms: u64,
    paint: impl FnOnce(&Canvas),
) -> Response {
    let rect = ui.available_rect_before_wrap();
    ui.painter().rect_filled(rect, 0.0, p.canvas);
    let log_x = axes.log_x;
    let log_y = axes.log_y;
    let ticks = axes.x_ticks.map(|t| t.to_vec());
    let mut plot = Plot::new(id)
        .x_axis_label(egui::RichText::new(axes.x_label).color(p.text_2))
        .y_axis_label(egui::RichText::new(axes.y_label).color(p.text_2))
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .allow_double_click_reset(false)
        .show_background(false)
        .auto_bounds(false)
        .grid_color(alpha(p.text_2, (crate::theme::GRID_ALPHA * 255.0) as u8 + 10))
        .x_axis_formatter(move |mark, _| match &ticks {
            Some(ticks) => ticks.iter().find(|(v, _)| (v - mark.value).abs() < 1e-9).map(|(_, t)| t.clone()).unwrap_or_default(),
            None => tick_label(mark, log_x),
        })
        .y_axis_formatter(move |mark, _| tick_label(mark, log_y))
        .label_formatter(|_| None);
    if axes.equal_aspect {
        plot = plot.data_aspect(1.0);
    }
    let pending = view.pending.take();
    let glide = &mut view.glide;
    let mut user_moved = false;
    let response = plot.show(ui, |plot_ui| {
        if let Some(b) = pending {
            plot_ui.set_plot_bounds(b);
        }
        let response = plot_ui.response().clone();
        if response.dragged() {
            *glide = None;
            user_moved = true;
        }
        if response.double_clicked() {
            user_moved = false;
        }
        // Wheel: glide the zoom about the cursor.
        if response.hovered() {
            let notches: f64 = plot_ui.ctx().input(|i| {
                i.events
                    .iter()
                    .map(|e| match e {
                        Event::MouseWheel { unit: MouseWheelUnit::Line, delta, .. } => delta.y as f64,
                        Event::MouseWheel { unit: MouseWheelUnit::Point, delta, .. } => delta.y as f64 / 50.0,
                        Event::MouseWheel { unit: MouseWheelUnit::Page, delta, .. } => delta.y as f64 * 3.0,
                        _ => 0.0,
                    })
                    .sum()
            });
            if notches != 0.0 {
                let log = 1.02f64.ln() * 15.0 * notches;
                let remaining = glide.as_ref().map_or(0.0, |g| g.total * (1.0 - g.done));
                let center = plot_ui.pointer_coordinate().unwrap_or_else(|| {
                    let b = plot_ui.plot_bounds();
                    PlotPoint::new((b.min()[0] + b.max()[0]) / 2.0, (b.min()[1] + b.max()[1]) / 2.0)
                });
                *glide = Some(Glide { total: log + remaining, done: 0.0, start: Instant::now(), center });
                user_moved = true;
            }
        }
        if let Some(g) = glide {
            let progress = if motion_ms == 0 {
                1.0
            } else {
                let t = (g.start.elapsed().as_secs_f64() * 1000.0 / motion_ms as f64).min(1.0);
                1.0 - (1.0 - t).powi(3)
            };
            let factor = (g.total * (progress - g.done)).exp() as f32;
            plot_ui.zoom_bounds(EVec2::splat(factor), g.center);
            g.done = progress;
            if progress >= 1.0 {
                *glide = None;
            } else {
                plot_ui.ctx().request_repaint();
            }
        }
    });
    if user_moved {
        view.follow = false;
    }
    if response.response.double_clicked() {
        view.follow = true;
    }
    let canvas = Canvas { painter: ui.painter_at(*response.transform.frame()), transform: response.transform, log_x, log_y };
    paint(&canvas);
    response.response
}

/// Paints items through a plot's transform.
pub struct Canvas {
    painter: egui::Painter,
    transform: PlotTransform,
    log_x: bool,
    log_y: bool,
}

/// Marker shapes.
#[derive(Clone, Copy)]
pub enum Marker {
    Circle,
    Square,
    Diamond,
}

impl Canvas {
    fn screen(&self, p: [f64; 2]) -> Option<Pos2> {
        let axis = |v: f64, log: bool| {
            if log {
                if v > 0.0 {
                    Some(v.log10())
                } else {
                    None
                }
            } else {
                Some(v)
            }
        };
        let x = axis(p[0], self.log_x)?;
        let y = axis(p[1], self.log_y)?;
        (x.is_finite() && y.is_finite()).then(|| self.transform.position_from_point(&PlotPoint::new(x, y)))
    }

    /// A polyline, broken where points are missing.
    pub fn polyline(&self, points: impl IntoIterator<Item = [f64; 2]>, stroke: Stroke, dashed: bool) {
        let mut run: Vec<Pos2> = Vec::new();
        let flush = |run: &mut Vec<Pos2>| {
            if run.len() >= 2 {
                if dashed {
                    self.painter.extend(Shape::dashed_line(run, stroke, 6.0, 4.0));
                } else {
                    self.painter.add(Shape::line(std::mem::take(run), stroke));
                }
            }
            run.clear();
        };
        for p in points {
            match self.screen(p) {
                Some(s) => run.push(s),
                None => flush(&mut run),
            }
        }
        flush(&mut run);
    }

    pub fn segments(&self, segments: &[[Vec2; 2]], stroke: Stroke) {
        for [a, b] in segments {
            if let (Some(a), Some(b)) = (self.screen([a.x, a.y]), self.screen([b.x, b.y])) {
                self.painter.line_segment([a, b], stroke);
            }
        }
    }

    pub fn markers(&self, points: impl IntoIterator<Item = [f64; 2]>, marker: Marker, side: f32, fill: Color32, stroke: Stroke) {
        let r = side / 2.0;
        for p in points {
            let Some(c) = self.screen(p) else { continue };
            match marker {
                Marker::Circle => {
                    self.painter.circle(c, r, fill, stroke);
                }
                Marker::Square => {
                    self.painter.rect(Rect::from_center_size(c, EVec2::splat(side)), 0.0, fill, stroke, egui::StrokeKind::Middle);
                }
                Marker::Diamond => {
                    let pts = vec![c + EVec2::new(0.0, -r), c + EVec2::new(r, 0.0), c + EVec2::new(0.0, r), c + EVec2::new(-r, 0.0)];
                    self.painter.add(Shape::convex_polygon(pts, fill, stroke));
                }
            }
        }
    }

    pub fn text(&self, at: Pos2, text: &str, color: Color32) {
        self.painter.text(at, egui::Align2::RIGHT_TOP, text, egui::FontId::proportional(crate::theme::typeface::SMALL), color);
    }

    pub fn frame(&self) -> Rect {
        *self.transform.frame()
    }
}

fn xy(p: &Vec2) -> [f64; 2] {
    [p.x, p.y]
}

/// Comb settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CombSettings {
    pub quantity: usize,
    pub height: f64,
    pub density: usize,
}

impl Default for CombSettings {
    fn default() -> Self {
        CombSettings { quantity: 0, height: 0.05, density: 250 }
    }
}

/// The main plot: data, curve, control polygon, knots and the overlays.
pub struct MainPlot {
    pub view: View,
    pub show_knots: bool,
    pub comb_on: bool,
    pub comb: CombSettings,
    pub deviation_on: bool,
    pub deviation_height: f64,
    data: Vec<Vec2>,
    curves: Vec<BSpline>,
    sampled: Vec<Vec<Vec2>>,
    knots: Vec<Vec2>,
    comb_strokes: Option<ParityStrokes>,
    deviation: Option<Deviation>,
    tolerance: Option<f64>,
    /// Foot parameters of the previous live step, refined at the next one.
    live_feet: Option<Vec<Vec<f64>>>,
    /// Foot points of the shown finished fit.
    fit_feet: Option<FootPoints>,
    live: bool,
}

impl Default for MainPlot {
    fn default() -> Self {
        MainPlot {
            view: View::default(),
            show_knots: false,
            comb_on: false,
            comb: CombSettings::default(),
            deviation_on: false,
            deviation_height: 0.05,
            data: vec![],
            curves: vec![],
            sampled: vec![],
            knots: vec![],
            comb_strokes: None,
            deviation: None,
            tolerance: None,
            live_feet: None,
            fit_feet: None,
            live: false,
        }
    }
}

impl MainPlot {
    /// New data points: clears every fit item and fits the range.
    pub fn set_data(&mut self, coords: &[Vec2]) {
        self.clear_items();
        self.data = coords.to_vec();
        self.view.fit(bounds_of(coords.iter().map(xy)));
        self.view.follow = true;
    }

    /// Removes the curve, polygon, knots and overlays (not the data).
    pub fn clear_items(&mut self) {
        self.curves.clear();
        self.sampled.clear();
        self.knots.clear();
        self.comb_strokes = None;
        self.deviation = None;
        self.live_feet = None;
        self.fit_feet = None;
    }

    /// Shows a finished fit.
    pub fn show_fit(&mut self, curves: Vec<BSpline>, tolerance: Option<f64>) {
        self.live = false;
        self.live_feet = None;
        self.fit_feet = None;
        self.set_curves(curves, tolerance);
    }

    /// Shows a step of a running fit.
    pub fn show_live(&mut self, curves: Vec<BSpline>, tolerance: Option<f64>) {
        self.live = true;
        self.set_curves(curves, tolerance);
    }

    fn set_curves(&mut self, curves: Vec<BSpline>, tolerance: Option<f64>) {
        self.sampled = curves.iter().map(|c| c.points(&linspace(0.0, 1.0, 1000))).collect();
        self.curves = curves;
        self.tolerance = tolerance;
        self.refresh_knots();
        self.refresh_comb();
        self.refresh_deviation();
    }

    pub fn refresh_knots(&mut self) {
        self.knots = if self.show_knots {
            self.curves.iter().flat_map(|c| distinct(&c.knots).into_iter().map(|u| c.point(u)).collect::<Vec<_>>()).collect()
        } else {
            vec![]
        };
    }

    pub fn refresh_comb(&mut self) {
        self.comb_strokes =
            (self.comb_on && !self.curves.is_empty()).then(|| comb(&self.curves, self.comb.quantity, self.comb.height, self.comb.density));
    }

    pub fn refresh_deviation(&mut self) {
        self.deviation = None;
        if !self.deviation_on || self.curves.is_empty() || self.data.len() < 2 {
            return;
        }
        let feet = if self.live {
            let feet = foot_points(&self.data, &self.curves, self.live_feet.as_deref());
            self.live_feet = Some(feet.parameters.clone());
            feet
        } else {
            match &self.fit_feet {
                Some(f) => f.clone(),
                None => {
                    let f = foot_points(&self.data, &self.curves, None);
                    self.fit_feet = Some(f.clone());
                    f
                }
            }
        };
        self.deviation = deviation(&self.data, &self.curves, &feet, self.deviation_height, self.tolerance);
    }

    pub fn show(&mut self, ui: &mut Ui, p: &Palette, motion_ms: u64) {
        let axes = Axes { x_label: "x", y_label: "y", log_x: false, log_y: false, x_ticks: None, equal_aspect: true };
        show_plot(ui, p, "main_plot", &mut self.view, &axes, motion_ms, |c| {
            let draw_strokes = |strokes: &ParityStrokes, dots: Option<&[Vec<Vec2>; 2]>| {
                for (parity, s) in strokes.iter().enumerate() {
                    let color = p.comb(parity);
                    let Strokes { teeth, envelopes } = s;
                    c.segments(teeth, Stroke::new(1.0, alpha(color, TEETH_ALPHA)));
                    for env in envelopes {
                        c.polyline(env.iter().map(xy), Stroke::new(1.5, color), false);
                    }
                    if let Some(dots) = dots {
                        c.markers(dots[parity].iter().map(xy), Marker::Circle, size::DATA_SIZE, color, Stroke::NONE);
                    }
                }
            };
            if let Some(strokes) = &self.comb_strokes {
                draw_strokes(strokes, None);
            }
            if let Some(d) = &self.deviation {
                draw_strokes(&d.strokes, Some(&d.dots));
                for line in &d.band {
                    c.polyline(line.iter().map(xy), Stroke::new(1.0, alpha(p.construction, 110)), true);
                }
            }
            // Construction: the control polygon.
            for curve in &self.curves {
                c.polyline(curve.control_points.iter().map(xy), Stroke::new(1.0, alpha(p.construction, 170)), true);
                c.markers(
                    curve.control_points.iter().map(xy),
                    Marker::Square,
                    size::CONTROL_SIZE,
                    p.canvas,
                    Stroke::new(1.0, p.construction),
                );
            }
            // Measured data.
            c.markers(self.data.iter().map(xy), Marker::Circle, size::DATA_SIZE, alpha(p.data(), 220), Stroke::NONE);
            // The curve.
            for section in &self.sampled {
                c.polyline(section.iter().map(xy), Stroke::new(size::CURVE_WIDTH, INK), false);
            }
            c.markers(self.knots.iter().map(xy), Marker::Diamond, size::KNOT_SIZE, p.knots, Stroke::new(1.0, p.canvas));
        });
    }
}

/// The convergence plot.
#[derive(Default)]
pub struct LossPlot {
    pub view: View,
    /// 0: iterations, 1: time.
    pub x_axis: usize,
    /// 0: none, 1: x only, 2: y only, 3: both.
    pub log_mode: usize,
    data: Vec<(usize, f64, f64)>,
}

/// Points of a trace, at most this many.
const MAX_PLOT_POINTS: usize = 100_000;

impl LossPlot {
    pub fn new() -> Self {
        LossPlot { log_mode: 2, ..Default::default() }
    }

    pub fn reset(&mut self) {
        self.data.clear();
        self.view = View::default();
    }

    /// `(iteration, seconds, error)` records; errors floored at 1e-12.
    pub fn set_data(&mut self, data: Vec<(usize, f64, f64)>) {
        self.data = data.into_iter().map(|(i, t, e)| (i, t, e.max(1e-12))).collect();
    }

    fn points(&self) -> Vec<[f64; 2]> {
        let n = self.data.len();
        let shown = n.min(MAX_PLOT_POINTS);
        let mut idx: Vec<usize> = linspace(0.0, n.saturating_sub(1) as f64, shown).iter().map(|v| v.round() as usize).collect();
        idx.dedup();
        idx.iter()
            .map(|&i| {
                let (it, t, e) = self.data[i];
                [if self.x_axis == 0 { it as f64 } else { t }, e]
            })
            .collect()
    }

    pub fn show(&mut self, ui: &mut Ui, p: &Palette, motion_ms: u64) {
        let (log_x, log_y) = (self.log_mode == 1 || self.log_mode == 3, self.log_mode >= 2);
        let points = self.points();
        if self.view.follow {
            let logged = points.iter().filter_map(|q| {
                let x = if log_x { (q[0] > 0.0).then(|| q[0].log10())? } else { q[0] };
                let y = if log_y { (q[1] > 0.0).then(|| q[1].log10())? } else { q[1] };
                Some([x, y])
            });
            if let Some(b) = bounds_of(logged) {
                self.view.fit(Some(b));
            }
        }
        let x_label = if self.x_axis == 0 { "Iterations" } else { "Time (s)" };
        let axes = Axes { x_label, y_label: "Error", log_x, log_y, x_ticks: None, equal_aspect: false };
        show_plot(ui, p, "loss_plot", &mut self.view, &axes, motion_ms, |c| {
            c.markers(points.iter().copied(), Marker::Circle, size::TRACE_POINT, p.trace(), Stroke::NONE);
        });
    }
}

/// The comparison of runs (or of a sweep).
#[derive(Default)]
pub struct ComparisonPlot {
    pub view: View,
    pub x_label: String,
    pub y_label: String,
    pub log_x: bool,
    pub log_y: bool,
    pub ticks: Option<Vec<(f64, String)>>,
    pub connected: bool,
    /// `(x, y)` of the trace.
    pub points: Vec<[f64; 2]>,
    /// Marked points (the selected runs or the best of a sweep).
    pub marked: Vec<[f64; 2]>,
}

impl ComparisonPlot {
    pub fn set(&mut self, points: Vec<[f64; 2]>, connected: bool) {
        self.points = points;
        self.connected = connected;
        let lx = self.log_x;
        let ly = self.log_y;
        let logged = self.points.iter().map(|q| [if lx { q[0].log10() } else { q[0] }, if ly { q[1].log10() } else { q[1] }]);
        self.view.fit(bounds_of(logged));
        self.view.follow = true;
    }

    pub fn show(&mut self, ui: &mut Ui, p: &Palette, motion_ms: u64) {
        let ticks = self.ticks.clone();
        let axes = Axes {
            x_label: &self.x_label,
            y_label: &self.y_label,
            log_x: self.log_x,
            log_y: self.log_y,
            x_ticks: ticks.as_deref(),
            equal_aspect: false,
        };
        let (points, marked, connected) = (&self.points, &self.marked, self.connected);
        show_plot(ui, p, "comparison_plot", &mut self.view, &axes, motion_ms, |c| {
            if connected {
                c.polyline(points.iter().copied(), Stroke::new(1.5, p.trace()), false);
            }
            c.markers(points.iter().copied(), Marker::Circle, 6.0, p.trace(), Stroke::NONE);
            c.markers(marked.iter().copied(), Marker::Circle, size::MARKER_SIZE, INK, Stroke::new(2.0, p.canvas));
        });
    }
}

/// Thickness and camber along the chord, in percent, each maximum marked.
pub fn shape_plot(ui: &mut Ui, p: &Palette, shape: &spline_fitter::bspline_varpro::airfoil::Shape, view: &mut View) {
    let pct = |xs: &[f64], ys: &[f64]| xs.iter().zip(ys).map(|(x, y)| [100.0 * x, 100.0 * y]).collect::<Vec<_>>();
    let thickness = pct(&shape.stations, &shape.thickness);
    let camber = pct(&shape.stations, &shape.camber);
    if view.follow && view.pending.is_none() {
        view.fit(bounds_of(thickness.iter().chain(camber.iter()).copied()));
        view.follow = false;
    }
    let axes =
        Axes { x_label: "Chord (%)", y_label: "Percent of the chord", log_x: false, log_y: false, x_ticks: None, equal_aspect: false };
    show_plot(ui, p, "shape_plot", view, &axes, 0, |c| {
        c.polyline(thickness.iter().copied(), Stroke::new(1.5, p.trace()), false);
        c.polyline(camber.iter().copied(), Stroke::new(1.5, p.construction), true);
        c.markers(
            [[100.0 * shape.max_thickness_x, 100.0 * shape.max_thickness]],
            Marker::Circle,
            7.0,
            p.trace(),
            Stroke::new(2.0, p.canvas),
        );
        c.markers(
            [[100.0 * shape.max_camber_x, 100.0 * shape.max_camber]],
            Marker::Circle,
            7.0,
            p.construction,
            Stroke::new(2.0, p.canvas),
        );
        let top_right = c.frame().right_top() + EVec2::new(-12.0, 12.0);
        c.text(top_right, "Thickness", p.trace());
        c.text(top_right + EVec2::new(0.0, 16.0), "Camber", p.construction);
    });
}
