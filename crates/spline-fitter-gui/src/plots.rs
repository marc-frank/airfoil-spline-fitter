//! The plots: the airfoil with its fit and overlays, convergence, the
//! comparison of runs, and the thickness and camber of a report.
//!
//! A plot is a framed canvas with axes on the left and bottom, drawn as the
//! Python version drew them (see `axes`); egui_plot supplies the transform
//! and panning. Every item is painted through the transform, so lines
//! break at gaps, markers have the Loft shapes and items stack in drawing
//! order. Log axes plot `log10` of the values with decade labels. The
//! wheel zooms about the cursor in a short glide.

use std::time::Instant;

use eframe::egui::{self, Color32, Event, MouseWheelUnit, Pos2, Rect, Response, Shape, Stroke, Ui, Vec2 as EVec2};
use egui_plot::{Plot, PlotBounds, PlotPoint, PlotTransform};
use spline_fitter::analysis::{foot_points, FootPoints};
use spline_fitter::bspline_varpro::basis::{distinct, linspace};
use spline_fitter::{BSpline, Vec2};

use crate::axes;
use crate::overlays::{comb, deviation, Deviation, ParityStrokes, Strokes, OUTSIDE};
use crate::theme::{alpha, size, typeface, Palette, INK, TEETH_ALPHA};

/// A running zoom glide.
struct Glide {
    total: f64,
    done: f64,
    start: Instant,
    center: PlotPoint,
}

/// View state of one plot: the zoom glide, a pending range, whether it
/// follows its data and the space its axes keep for tick labels.
pub struct View {
    glide: Option<Glide>,
    /// Data bounds to show on the next frame, padded for the plot's size.
    pending: Option<PlotBounds>,
    /// Fit the range to the data when it changes, until the user zooms or pans.
    pub follow: bool,
    /// Width of the left axis's labels and height of the bottom axis's.
    left_text: f32,
    bottom_text: f32,
}

impl Default for View {
    fn default() -> Self {
        View { glide: None, pending: None, follow: true, left_text: 30.0, bottom_text: 18.0 }
    }
}

impl View {
    /// Back to following its data; the axes keep the room their labels
    /// took (pyqtgraph's axes outlive a cleared plot).
    pub fn reset(&mut self) {
        *self = View { left_text: self.left_text, bottom_text: self.bottom_text, ..View::default() };
    }

    /// Shows `bounds` (padded as an automatic range) on the next frame.
    pub fn fit(&mut self, bounds: Option<PlotBounds>) {
        if let Some(b) = bounds {
            self.pending = Some(b);
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
    /// The x values are whole numbers: no ticks between them.
    pub whole_x: bool,
    pub equal_aspect: bool,
    /// Pixels an automatic range keeps around the data for its largest
    /// marker (0.7072 of its outer width, as pyqtgraph pads a scatter).
    pub marker_pad: f32,
}

/// The pixel padding of a marker `side` pixels wide with a `pen` outline.
pub fn marker_pad(side: f32, pen: f32) -> f32 {
    0.7072 * (side + pen)
}

/// Space between the plot frame and the plot item (half the 3 px outline
/// of an export, rounded up, inside the 1 px border).
const FRAME_INSET: f32 = 3.0;
/// The plot item's margins: small at the axes, even above and right.
const AXIS_MARGIN: f32 = size::TIGHT;
const PLOT_MARGIN: f32 = 24.0;
/// What an axis label adds to its axis (0.8 of the label's 24 px box).
const LABEL_EXTENT: f32 = 19.2;
/// Gap between the axis line and its tick labels: left, bottom.
const TICK_OFFSET: [f32; 2] = [5.0, 2.0];
/// Tick labels a point below the base size; axis labels at the base size.
const TICK_FONT: f32 = typeface::TICK;

/// Ticks of one axis, per level: screen positions and labels (`None`: not drawn).
struct AxisTicks {
    levels: Vec<Vec<(f32, Option<String>)>>,
}

/// The ticks of an axis showing `range` on `length` pixels, or the given
/// labels of a categorical axis.
/// Where an axis puts its ticks.
enum TickSource<'a> {
    /// At these positions, with these labels.
    Labels(&'a [(f64, String)]),
    /// Chosen for the range; `whole` keeps only whole numbers.
    Auto { whole: bool },
}

fn axis_ticks(
    range: (f64, f64),
    length: f32,
    log: bool,
    source: TickSource,
    to_screen: impl Fn(f64) -> f32,
    measure: impl Fn(&str) -> f32,
    fits: impl Fn(f32, f32) -> bool,
) -> AxisTicks {
    let given = matches!(source, TickSource::Labels(_));
    let levels: Vec<(Vec<f64>, Vec<String>)> = match source {
        TickSource::Labels(given) => vec![(given.iter().map(|g| g.0).collect(), given.iter().map(|g| g.1.clone()).collect())],
        TickSource::Auto { whole } => {
            let decade = axes::decade_in_view(range.0, range.1);
            let levels = axes::tick_values(range.0, range.1, length as f64, log);
            let levels = if whole && !log { axes::whole_number_levels(levels, range.0, range.1) } else { levels };
            levels
                .into_iter()
                .map(|l| {
                    let strings = axes::tick_strings(&l.values, l.spacing, log, decade);
                    (l.values, strings)
                })
                .collect()
        }
    };
    let (a, b) = (to_screen(range.0), to_screen(range.1));
    let (low, high) = (a.min(b), a.max(b));
    let positioned: Vec<Vec<(f32, String)>> = levels
        .iter()
        .map(|(values, strings)| {
            values
                .iter()
                .zip(strings)
                .map(|(&v, s)| (to_screen(v), s.clone()))
                .filter(|(x, _)| *x >= low - 0.5 && *x <= high + 0.5)
                .collect()
        })
        .collect();
    let extents: Vec<Vec<(f32, bool)>> = positioned
        .iter()
        .map(|l| {
            l.iter()
                .filter(|(_, s)| !s.is_empty())
                .map(|(x, s)| {
                    let e = measure(s);
                    (e, fits(*x, e))
                })
                .collect()
        })
        .collect();
    let labelled = if given { positioned.len() } else { axes::labelled_levels(&extents, length) };
    AxisTicks {
        levels: positioned
            .into_iter()
            .enumerate()
            .map(|(i, l)| l.into_iter().map(|(x, s)| (x, (i < labelled && !s.is_empty()).then_some(s))).collect())
            .collect(),
    }
}

/// Shows a framed plot and returns its response; `paint` draws the items.
pub fn show_plot(
    ui: &mut Ui,
    p: &Palette,
    id: &str,
    view: &mut View,
    axes: &Axes,
    motion_ms: u64,
    paint: impl FnOnce(&Canvas),
) -> Response {
    let outer = ui.available_rect_before_wrap();
    let radius = egui::CornerRadius::same(size::RADIUS);
    ui.painter().rect_filled(outer, radius, p.canvas);
    ui.painter().rect_stroke(outer, radius, Stroke::new(1.0, p.rule), egui::StrokeKind::Inside);
    let item = outer.shrink(FRAME_INSET);
    let left = item.left() + AXIS_MARGIN + view.left_text + TICK_OFFSET[0] + LABEL_EXTENT;
    let bottom = item.bottom() - AXIS_MARGIN - (view.bottom_text + TICK_OFFSET[1] + LABEL_EXTENT);
    let frame = Rect::from_min_max(egui::pos2(left, item.top() + PLOT_MARGIN), egui::pos2(item.right() - PLOT_MARGIN, bottom));
    let frame = if frame.width() > 1.0 && frame.height() > 1.0 { frame } else { Rect::from_min_size(item.min, EVec2::splat(2.0)) };

    let log_x = axes.log_x;
    let log_y = axes.log_y;
    let mut plot = Plot::new(id)
        .show_axes(false)
        .show_grid(false)
        .show_x(false)
        .show_y(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .allow_double_click_reset(false)
        .show_background(false)
        .auto_bounds(false)
        // An empty plot shows 0 to 1 on both axes, as pyqtgraph's.
        .default_x_bounds(0.0, 1.0)
        .default_y_bounds(0.0, 1.0);
    if axes.equal_aspect {
        plot = plot.data_aspect(1.0);
    }
    // An automatic range is padded by 1/sqrt(pixels) of its width, 2 to 10 %.
    // The markers' pixels widen the range first, as in pyqtgraph.
    let pad = axes.marker_pad as f64;
    let range = |lo: f64, hi: f64, pixels: f32| {
        let px = if hi > lo { (hi - lo) / pixels.max(1.0) as f64 } else { 0.0 };
        axes::padded(lo - pad * px, hi + pad * px, axes::auto_padding(pixels as f64))
    };
    let pending = view.pending.take().map(|b| {
        let (x0, x1) = range(b.min()[0], b.max()[0], frame.width());
        let (y0, y1) = range(b.min()[1], b.max()[1], frame.height());
        PlotBounds::from_min_max([x0, y0], [x1, y1])
    });
    let glide = &mut view.glide;
    let mut user_moved = false;
    let response = ui
        .scope_builder(egui::UiBuilder::new().max_rect(frame), |ui| {
            plot.show(ui, |plot_ui| {
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
            })
        })
        .inner;
    ui.allocate_rect(outer, egui::Sense::hover());
    if user_moved {
        view.follow = false;
    }
    if response.response.double_clicked() {
        view.follow = true;
    }
    let transform = response.transform;
    let frame = *transform.frame();
    let bounds = *transform.bounds();
    let tick_font = egui::FontId::proportional(TICK_FONT);
    // Label boxes as Qt measures them: every glyph hinted to whole pixels.
    let line = ui.painter().layout_no_wrap("0".to_string(), tick_font.clone(), p.text_2).size().y.ceil();
    let measure = |text: &str| EVec2::new(crate::fonts::layout_width(ui, text, &tick_font, false), line);
    let x_ticks = axis_ticks(
        (bounds.min()[0], bounds.max()[0]),
        frame.width(),
        log_x,
        axes.x_ticks.map_or(TickSource::Auto { whole: axes.whole_x }, TickSource::Labels),
        |v| transform.position_from_point_x(v),
        |s| measure(s).x,
        // With a grid, a label shows only within the plot's extent.
        |x, w| x - w / 2.0 >= frame.left() - 0.5 && x + w / 2.0 <= frame.right() + 0.5,
    );
    let y_ticks = axis_ticks(
        (bounds.min()[1], bounds.max()[1]),
        frame.height(),
        log_y,
        TickSource::Auto { whole: false },
        |v| transform.position_from_point_y(v),
        |s| measure(s).y * 0.8,
        |y, h| y - h / 2.0 >= frame.top() - 0.5 && y + h / 2.0 <= frame.bottom() + 0.5,
    );

    // The grid: every tick level, fainter for lower levels.
    let painter = ui.painter_at(frame);
    let grid = |index: usize, count: usize, length: f32| {
        let a = axes::grid_alpha(index, count, length as f64, crate::theme::GRID_ALPHA as f64);
        alpha(p.text_2, a.round().clamp(0.0, 255.0) as u8)
    };
    for (i, level) in x_ticks.levels.iter().enumerate() {
        let color = grid(i, level.len(), frame.width());
        for (x, _) in level {
            painter.vline(x.round() + 0.5, frame.y_range(), Stroke::new(1.0, color));
        }
    }
    for (i, level) in y_ticks.levels.iter().enumerate() {
        let color = grid(i, level.len(), frame.height());
        for (y, _) in level {
            painter.hline(frame.x_range(), y.round() + 0.5, Stroke::new(1.0, color));
        }
    }

    let canvas = Canvas { painter: ui.painter_at(frame), transform, log_x, log_y };
    paint(&canvas);

    // Axis lines, tick labels and axis labels.
    let painter = ui.painter_at(outer);
    let line = Stroke::new(1.0, p.text_2);
    painter.vline(frame.left() - 0.5, frame.top() - 1.0..=frame.bottom() + 1.0, line);
    painter.hline(frame.left() - 1.0..=frame.right() + 1.0, frame.bottom() + 0.5, line);
    let mut widest: f32 = 0.0;
    for (y, text) in y_ticks.levels.iter().flatten() {
        let Some(text) = text else { continue };
        let galley = painter.layout_no_wrap(text.clone(), tick_font.clone(), p.text_2);
        let size = galley.size();
        let half = (size.y * 0.8).ceil() / 2.0;
        if y - half < frame.top() - 0.5 || y + half > frame.bottom() + 0.5 {
            continue;
        }
        // As wide as Qt lays the label out, right-aligned at the offset.
        let width = measure(text).x;
        widest = widest.max(width);
        painter.galley(egui::pos2(frame.left() - TICK_OFFSET[0] - width, y - size.y / 2.0), galley, p.text_2);
    }
    let mut tallest: f32 = 0.0;
    for (x, text) in x_ticks.levels.iter().flatten() {
        let Some(text) = text else { continue };
        let galley = painter.layout_no_wrap(text.clone(), tick_font.clone(), p.text_2);
        let size = galley.size();
        if x - size.x / 2.0 < frame.left() - 0.5 || x + size.x / 2.0 > frame.right() + 0.5 {
            continue;
        }
        tallest = tallest.max(size.y * 0.8);
        painter.galley(egui::pos2(x - size.x / 2.0, frame.bottom() + TICK_OFFSET[1] + 0.5), galley, p.text_2);
    }
    let label_font = egui::FontId::proportional(typeface::BASE);
    if !axes.x_label.is_empty() {
        let bottom_axis = view.bottom_text + TICK_OFFSET[1] + LABEL_EXTENT;
        let at = egui::pos2(frame.center().x, frame.bottom() + bottom_axis - 6.0);
        painter.text(at, egui::Align2::CENTER_CENTER, axes.x_label, label_font.clone(), p.text_2);
    }
    if !axes.y_label.is_empty() {
        let galley = painter.layout_no_wrap(axes.y_label.to_string(), label_font, p.text_2);
        let center = egui::pos2(item.left() + AXIS_MARGIN + 6.0, frame.center().y);
        let at = center + EVec2::new(-galley.size().y / 2.0, galley.size().x / 2.0);
        painter.add(egui::epaint::TextShape::new(at, galley, p.text_2).with_angle(-std::f32::consts::FRAC_PI_2));
    }
    let left_text = axes::text_space(view.left_text, widest);
    let bottom_text = axes::text_space(view.bottom_text, tallest.max(18.0));
    if (left_text, bottom_text) != (view.left_text, view.bottom_text) {
        view.left_text = left_text;
        view.bottom_text = bottom_text;
        ui.ctx().request_repaint();
    }
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

    /// A polyline, broken where points are missing. Only its part inside
    /// the frame is drawn: zoomed far in, the line is millions of pixels
    /// long, too long to cut into dashes. A dash pattern keeps its phase
    /// along the whole line, so it does not crawl as the view pans.
    pub fn polyline(&self, points: impl IntoIterator<Item = [f64; 2]>, stroke: Stroke, dashed: bool) {
        let clip = self.frame().expand(stroke.width + 2.0);
        // Visible runs with the distance along the line where each starts.
        let mut runs: Vec<(f64, Vec<Pos2>)> = Vec::new();
        let mut open = false;
        let mut previous: Option<Pos2> = None;
        let mut travelled = 0.0f64;
        for p in points {
            let Some(b) = self.screen(p) else {
                previous = None;
                open = false;
                continue;
            };
            if let Some(a) = previous {
                let length = f64::from((b - a).length());
                match clip_segment(a, b, clip) {
                    Some((t0, t1)) => {
                        let (from, to) = (a + (b - a) * t0, a + (b - a) * t1);
                        if !open || t0 > 0.0 {
                            runs.push((travelled + length * f64::from(t0), vec![from]));
                        }
                        runs.last_mut().unwrap().1.push(to);
                        open = t1 >= 1.0;
                    }
                    None => open = false,
                }
                travelled += length;
            }
            previous = Some(b);
        }
        for (start, run) in runs {
            if run.len() < 2 {
                continue;
            }
            if dashed {
                // The pattern: 6 px dash, 4 px gap.
                let offset = start.rem_euclid(10.0) as f32;
                self.painter.extend(Shape::dashed_line_with_offset(&run, stroke, &[6.0], &[4.0], offset));
            } else {
                self.painter.add(Shape::line(run, stroke));
            }
        }
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
                    // egui strokes outside the radius; pyqtgraph centers the pen on it.
                    self.painter.circle(c, (r - stroke.width / 2.0).max(0.5), fill, stroke);
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

    pub fn frame(&self) -> Rect {
        *self.transform.frame()
    }
}

/// The part of the segment `a`–`b` inside `rect`, as the segment's
/// fractions where it enters and leaves (Liang-Barsky); `None` if it misses.
fn clip_segment(a: Pos2, b: Pos2, rect: Rect) -> Option<(f32, f32)> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [(-d.x, a.x - rect.left()), (d.x, rect.right() - a.x), (-d.y, a.y - rect.top()), (d.y, rect.bottom() - a.y)] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }
    (t0 <= t1).then_some((t0, t1))
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
    /// Foot points of the data on the shown curves, measured when the
    /// deviation is first drawn.
    feet: Option<FootPoints>,
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
            feet: None,
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
        self.feet = None;
    }

    /// Shows a finished fit or a step of a running one.
    pub fn show_curves(&mut self, curves: Vec<BSpline>, tolerance: Option<f64>) {
        self.feet = None;
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
        let feet = self.feet.get_or_insert_with(|| foot_points(&self.data, &self.curves));
        self.deviation = deviation(&self.data, &self.curves, feet, self.deviation_height, self.tolerance);
    }

    pub fn show(&mut self, ui: &mut Ui, p: &Palette, motion_ms: u64) {
        let pad = marker_pad(size::CONTROL_SIZE, 1.0);
        let axes = Axes {
            x_label: "x",
            y_label: "y",
            log_x: false,
            log_y: false,
            x_ticks: None,
            whole_x: false,
            equal_aspect: true,
            marker_pad: pad,
        };
        show_plot(ui, p, "main_plot", &mut self.view, &axes, motion_ms, |c| {
            let draw_strokes = |strokes: &[Strokes; 2], dots: Option<&[Vec<Vec2>; 2]>, color: &dyn Fn(usize) -> Color32| {
                for (parity, s) in strokes.iter().enumerate() {
                    let color = color(parity);
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
                draw_strokes(strokes, None, &|parity| p.comb(parity));
            }
            if let Some(d) = &self.deviation {
                draw_strokes(&d.strokes, Some(&d.dots), &|side| p.deviation(side == OUTSIDE));
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
        self.view.reset();
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
        let pad = marker_pad(size::TRACE_POINT, 0.0);
        let axes = Axes { x_label, y_label: "Error", log_x, log_y, x_ticks: None, whole_x: false, equal_aspect: false, marker_pad: pad };
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
    /// The tolerance the plotted runs share, drawn as a dashed line when
    /// the y axis is their deviation.
    pub tolerance: Option<f64>,
    /// A tooltip for each point (the run it stands for); none in a sweep.
    pub labels: Vec<String>,
}

/// How near the pointer a point must be to be hovered or clicked.
const PICK_RADIUS: f32 = 8.0;

impl ComparisonPlot {
    pub fn set(&mut self, points: Vec<[f64; 2]>, connected: bool) {
        self.points = points;
        self.connected = connected;
        let lx = self.log_x;
        let ly = self.log_y;
        // The tolerance line is in view with the points.
        let line = self.tolerance.zip(self.points.first()).map(|(t, q)| [q[0], t]);
        let logged =
            self.points.iter().chain(line.as_ref()).map(|q| [if lx { q[0].log10() } else { q[0] }, if ly { q[1].log10() } else { q[1] }]);
        self.view.fit(bounds_of(logged));
        self.view.follow = true;
    }

    /// Draws the plot; the index of a labelled point that was clicked.
    pub fn show(&mut self, ui: &mut Ui, p: &Palette, motion_ms: u64) -> Option<usize> {
        let ticks = self.ticks.clone();
        let axes = Axes {
            x_label: &self.x_label,
            y_label: &self.y_label,
            log_x: self.log_x,
            log_y: self.log_y,
            x_ticks: ticks.as_deref(),
            whole_x: self.ticks.is_none() && !self.points.is_empty() && self.points.iter().all(|q| q[0].fract() == 0.0),
            equal_aspect: false,
            // Room for the run marker around every point, so moving the
            // marker never rescales the plot.
            marker_pad: marker_pad(size::MARKER_SIZE, 2.0),
        };
        let (points, marked, connected, tolerance) = (&self.points, &self.marked, self.connected, self.tolerance);
        let mut screen: Vec<(usize, Pos2)> = Vec::new();
        let response = show_plot(ui, p, "comparison_plot", &mut self.view, &axes, motion_ms, |c| {
            screen = points.iter().enumerate().filter_map(|(i, q)| c.screen(*q).filter(|s| c.frame().contains(*s)).map(|s| (i, s))).collect();
            // As in the main plot: a dashed construction line.
            if let Some(y) = tolerance.zip(points.first()).and_then(|(t, q)| c.screen([q[0], t])).map(|s| s.y) {
                let frame = c.frame();
                if frame.y_range().contains(y) {
                    let line = [egui::pos2(frame.left(), y), egui::pos2(frame.right(), y)];
                    c.painter.extend(Shape::dashed_line(&line, Stroke::new(1.0, alpha(p.construction, 110)), 6.0, 4.0));
                }
            }
            if connected {
                c.polyline(points.iter().copied(), Stroke::new(1.5, p.trace()), false);
            }
            c.markers(points.iter().copied(), Marker::Circle, 6.0, p.trace(), Stroke::NONE);
            c.markers(marked.iter().copied(), Marker::Circle, size::MARKER_SIZE, INK, Stroke::new(2.0, p.canvas));
        });
        // The run under the pointer: its tooltip, and a click selects it.
        let near = response.hover_pos().and_then(|at| {
            screen
                .iter()
                .filter(|(i, s)| *i < self.labels.len() && s.distance(at) <= PICK_RADIUS)
                .min_by(|a, b| a.1.distance(at).total_cmp(&b.1.distance(at)))
                .map(|&(i, _)| i)
        })?;
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        let clicked = response.clicked();
        response.on_hover_text_at_pointer(&self.labels[near]);
        clicked.then_some(near)
    }
}

/// The chart legend (pyqtgraph's, 12 px in from the top right): its
/// names end this far from the plot's right edge, the first row's centre
/// this far below its top, rows this far apart.
const LEGEND_INSET: f32 = 25.0;
const LEGEND_FIRST: f32 = 34.0;
const LEGEND_ROW: f32 = 24.0;

/// Thickness and camber along the chord, in percent, each maximum marked.
pub fn shape_plot(ui: &mut Ui, p: &Palette, shape: &spline_fitter::bspline_varpro::airfoil::Shape, view: &mut View) {
    let pct = |xs: &[f64], ys: &[f64]| xs.iter().zip(ys).map(|(x, y)| [100.0 * x, 100.0 * y]).collect::<Vec<_>>();
    let thickness = pct(&shape.stations, &shape.thickness);
    let camber = pct(&shape.stations, &shape.camber);
    if view.follow && view.pending.is_none() {
        view.fit(bounds_of(thickness.iter().chain(camber.iter()).copied()));
        view.follow = false;
    }
    let axes = Axes {
        x_label: "Chord (%)",
        y_label: "Percent of the chord",
        log_x: false,
        log_y: false,
        x_ticks: None,
        whole_x: false,
        equal_aspect: false,
        marker_pad: marker_pad(7.0, 2.0),
    };
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
        // The legend: a line sample beside each name, inset at the top right.
        let font = egui::FontId::proportional(typeface::BASE);
        let names = [("Thickness", p.trace(), false), ("Camber", p.construction, true)];
        let widest =
            names.iter().map(|(n, ..)| c.painter.layout_no_wrap(n.to_string(), font.clone(), p.text_2).size().x).fold(0.0, f32::max);
        let left = c.frame().right() - LEGEND_INSET - widest;
        for (i, (name, color, dashed)) in names.into_iter().enumerate() {
            let y = c.frame().top() + LEGEND_FIRST + i as f32 * LEGEND_ROW;
            let sample = [egui::pos2(left - 30.0, y), egui::pos2(left - 10.0, y)];
            let stroke = Stroke::new(1.5, color);
            if dashed {
                c.painter.extend(Shape::dashed_line(&sample, stroke, 6.0, 4.0));
            } else {
                c.painter.line_segment(sample, stroke);
            }
            c.painter.text(egui::pos2(left, y), egui::Align2::LEFT_CENTER, name, font.clone(), p.text_2);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cleared plot follows its data again but keeps the room its axis
    /// labels took: from there the hysteresis lets the axis shrink to
    /// narrower labels, as pyqtgraph's axis does.
    #[test]
    fn a_reset_view_keeps_its_label_room() {
        let mut view = View { left_text: 20.0, bottom_text: 18.0, follow: false, ..View::default() };
        view.reset();
        assert!(view.follow);
        assert_eq!((view.left_text, view.bottom_text), (20.0, 18.0));
    }
}
