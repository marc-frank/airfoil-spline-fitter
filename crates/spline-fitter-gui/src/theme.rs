//! The Loft design language: colour tokens, sizes and type.
//!
//! Greys are lightness values turned into colours in OKLab with a cool-to-
//! warm temperature, so a temperature change keeps every contrast. Colours
//! that differ between light and dark but are no greys are mode pairs.
//! Orange ink is reserved for the fitted curve, the primary button and the
//! current-run marker. This module is the only place colours, fonts and
//! sizes are defined.

use eframe::egui::{self, Color32, CornerRadius, FontFamily, FontId, Margin, Stroke, TextStyle, Vec2};

/// Light or dark mode choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThemeMode {
    #[default]
    System,
    Light,
    Dark,
}

impl ThemeMode {
    pub const ALL: [ThemeMode; 3] = [Self::System, Self::Light, Self::Dark];

    pub fn value(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Light => "light",
            Self::Dark => "dark",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::System => "System",
            Self::Light => "Light",
            Self::Dark => "Dark",
        }
    }

    pub fn from_value(value: &str) -> Self {
        Self::ALL.into_iter().find(|m| m.value() == value).unwrap_or_default()
    }
}

/// OKLab lightness of every grey token: (dark, light).
const GREY_LIGHTNESS: [(&str, f64, f64); 13] = [
    ("CANVAS", 0.213, 0.995),
    ("WINDOW", 0.236, 0.970),
    ("PANEL", 0.268, 0.950),
    ("FIELD", 0.309, 1.000),
    ("FIELD_HOVER", 0.344, 0.965),
    ("RULE", 0.360, 0.870),
    ("RULE_STRONG", 0.434, 0.780),
    ("TEXT", 0.982, 0.250),
    ("TEXT_2", 0.807, 0.430),
    ("TEXT_3", 0.686, 0.580),
    ("SELECT", 0.405, 0.895),
    ("SELECT_LINE", 0.609, 0.600),
    ("CONSTRUCTION", 0.639, 0.620),
];

/// A grey of OKLab lightness `l` tinted by temperature `t` in `[-1, 1]`
/// (cool blue below 0, warm sand above).
pub fn grey(l: f64, t: f64) -> Color32 {
    let hue = (if t >= 0.0 { 95.0f64 } else { 255.0 }).to_radians();
    let chroma = t.abs() * 0.035 * l * ((1.0 - l) * 20.0).min(1.0);
    let (a, b) = (chroma * hue.cos(), chroma * hue.sin());
    let l_ = (l + 0.3963377774 * a + 0.2158037573 * b).powi(3);
    let m_ = (l - 0.1055613458 * a - 0.0638541728 * b).powi(3);
    let s_ = (l - 0.0894841775 * a - 1.2914855480 * b).powi(3);
    let linear = [
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    ];
    let encode = |c: f64| {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0).round_ties_even() as u8
    };
    Color32::from_rgb(encode(linear[0]), encode(linear[1]), encode(linear[2]))
}

const fn hex(rgb: u32) -> Color32 {
    Color32::from_rgb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8)
}

/// The icon's orange: the fitted curve, the primary button, the current run.
pub const INK: Color32 = hex(0xFF6E10);
pub const INK_HOVER: Color32 = hex(0xFF8636);
pub const INK_PRESSED: Color32 = hex(0xE05E08);
pub const ON_INK: Color32 = hex(0x1A0C02);

/// Every colour token for one mode and temperature.
#[derive(Debug, Clone, PartialEq)]
pub struct Palette {
    pub dark: bool,
    pub canvas: Color32,
    pub window: Color32,
    pub panel: Color32,
    pub field: Color32,
    pub field_hover: Color32,
    pub rule: Color32,
    pub rule_strong: Color32,
    pub text: Color32,
    pub text_2: Color32,
    pub text_3: Color32,
    pub select: Color32,
    pub select_line: Color32,
    /// Plot: control polygon, tolerance band, camber line.
    pub construction: Color32,
    pub ok: Color32,
    pub warn: Color32,
    pub error: Color32,
    /// Plot: knot diamonds.
    pub knots: Color32,
    /// Plot: comb and deviation of the first, third, ... section.
    pub comb_even: Color32,
    /// Plot: comb and deviation of the second, fourth, ... section.
    pub comb_odd: Color32,
}

impl Palette {
    pub fn new(dark: bool, temperature: i32) -> Self {
        let t = temperature.clamp(-100, 100) as f64 / 100.0;
        let g = |name: &str| {
            let (_, d, l) = GREY_LIGHTNESS.iter().find(|(n, _, _)| *n == name).unwrap();
            grey(if dark { *d } else { *l }, t)
        };
        let pick = |d: u32, l: u32| hex(if dark { d } else { l });
        Palette {
            dark,
            canvas: g("CANVAS"),
            window: g("WINDOW"),
            panel: g("PANEL"),
            field: g("FIELD"),
            field_hover: g("FIELD_HOVER"),
            rule: g("RULE"),
            rule_strong: g("RULE_STRONG"),
            text: g("TEXT"),
            text_2: g("TEXT_2"),
            text_3: g("TEXT_3"),
            select: g("SELECT"),
            select_line: g("SELECT_LINE"),
            construction: g("CONSTRUCTION"),
            ok: pick(0x4FBF85, 0x1A7A4A),
            warn: pick(0xF2B84B, 0x8F5E00),
            error: pick(0xF0605D, 0xC02F2B),
            knots: pick(0xF2C94C, 0xC99A06),
            comb_even: pick(0x49B6E8, 0x1E88C7),
            comb_odd: pick(0xE5689A, 0xCC3F78),
        }
    }

    /// Plot: measured data points.
    pub fn data(&self) -> Color32 {
        self.text
    }

    /// Plot: convergence and comparison traces, grid and axes.
    pub fn trace(&self) -> Color32 {
        self.text_2
    }

    /// Comb (and deviation) colour of a section.
    pub fn comb(&self, section: usize) -> Color32 {
        if section % 2 == 0 {
            self.comb_even
        } else {
            self.comb_odd
        }
    }
}

/// `color` with alpha `a` (0-255).
pub fn alpha(color: Color32, a: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), a)
}

/// Named sizes in pixels.
pub mod size {
    pub const TIGHT: f32 = 4.0;
    pub const GAP: f32 = 6.0;
    pub const GROUP: f32 = 8.0;
    pub const PAD: f32 = 12.0;
    pub const PAD_LARGE: f32 = 16.0;
    pub const CONTROL_HEIGHT: f32 = 26.0;
    /// Space on each side of a button's text (a push button is 22 px wider
    /// than its text, a tool button 31 px).
    pub const BUTTON_PADDING: f32 = 11.0;
    pub const TOOL_BUTTON_PADDING: f32 = 15.5;
    /// The drop-down half of a split toggle.
    pub const ARROW_BUTTON_WIDTH: f32 = 24.0;
    /// A combo box is this much wider than its longest item's ink.
    pub const COMBO_EXTRA: f32 = 36.0;
    /// A combo box's text inset, and its chevron's centre from the right.
    pub const COMBO_PADDING: f32 = 8.0;
    pub const COMBO_CHEVRON_INSET: f32 = 11.5;
    /// A check box's box, its 1 px border included.
    pub const CHECK_BOX: f32 = 16.0;
    /// A form row holding only a check box: a line of text.
    pub const CHECK_ROW_HEIGHT: f32 = 16.0;
    /// A section caption's line.
    pub const CAPTION_LINE: f32 = 14.0;
    /// A scroll bar's track; its handle is 2 px in and at least 24 px long.
    pub const SCROLL_TRACK: f32 = 10.0;
    pub const SCROLL_HANDLE_MARGIN: f32 = 2.0;
    pub const SCROLL_HANDLE_MIN: f32 = 24.0;
    /// The status bar's size grip.
    pub const GRIP: f32 = 17.0;
    /// A run bar readout's value: a line of the readout face.
    pub const READOUT_LINE: f32 = 18.0;
    /// The menu bar: a line of text and 4 px above and below.
    pub const MENU_BAR_HEIGHT: f32 = 24.0;
    pub const GLYPH: f32 = 10.0;
    pub const GLYPH_LARGE: f32 = 16.0;
    pub const RADIUS: u8 = 3;
    pub const LABEL_WIDTH: f32 = 92.0;
    pub const FIELD_MIN_WIDTH: f32 = 70.0;
    /// The sidebar's default width: its widest row (the smoothing sweep of
    /// Advanced, two fields and the toggle) at the base type size.
    pub const SIDEBAR_WIDTH: f32 = 410.0;
    pub const SIDEBAR_MIN_WIDTH: f32 = 300.0;
    /// The rule between panels: Qt's splitter handle lies between them, an
    /// egui panel's size takes it in.
    pub const SPLITTER: f32 = 1.0;
    pub const EXPLORER_WIDTH: f32 = 570.0;
    pub const EXPLORER_MIN_WIDTH: f32 = 180.0;
    pub const SLIDER_WIDTH: f32 = 220.0;
    pub const DIALOG_MIN_WIDTH: f32 = 480.0;
    pub const POPUP_MIN_WIDTH: f32 = 260.0;
    pub const CURVE_WIDTH: f32 = 2.0;
    pub const DATA_SIZE: f32 = 4.0;
    pub const CONTROL_SIZE: f32 = 6.0;
    pub const KNOT_SIZE: f32 = 8.0;
    pub const TRACE_POINT: f32 = 4.0;
    pub const MARKER_SIZE: f32 = 13.0;
    pub const SHAPE_PLOT_HEIGHT: f32 = 200.0;
}

/// Type sizes in pixels.
pub mod typeface {
    /// 9 pt.
    pub const BASE: f32 = 12.0;
    pub const SMALL: f32 = 10.5;
    pub const CAPTION: f32 = 10.0;
    /// Muted notes: a caption plus one.
    pub const MUTED: f32 = 11.0;
    pub const READOUT: f32 = 15.0;
    pub const HEADING: f32 = 15.0;
}

/// Alpha of comb teeth and deviation whiskers.
pub const TEETH_ALPHA: u8 = 90;
/// Alpha of the plot grid.
pub const GRID_ALPHA: f32 = 0.10;

/// Colors scroll handles as the strong rule (the text grey when hovered or
/// grabbed) in `ui`; egui paints them with the button fills, so only for a
/// region without buttons.
pub fn scroll_handles(ui: &mut egui::Ui, p: &Palette) {
    ui.visuals_mut().extreme_bg_color = p.panel;
    let w = &mut ui.visuals_mut().widgets;
    w.inactive.bg_fill = p.rule_strong;
    w.hovered.bg_fill = p.text_3;
    w.active.bg_fill = p.text_3;
}

/// Applies the palette and the sizes to every style of the context.
pub fn apply(ctx: &egui::Context, p: &Palette) {
    let mut visuals = if p.dark { egui::Visuals::dark() } else { egui::Visuals::light() };
    let radius = CornerRadius::same(size::RADIUS);
    let widget = |bg: Color32, border: Color32, fg: Color32| egui::style::WidgetVisuals {
        bg_fill: bg,
        weak_bg_fill: bg,
        bg_stroke: Stroke::new(1.0, border),
        corner_radius: radius,
        fg_stroke: Stroke::new(1.0, fg),
        expansion: 0.0,
    };
    visuals.widgets.noninteractive = widget(p.panel, p.rule, p.text);
    visuals.widgets.inactive = widget(p.field, p.rule, p.text);
    visuals.widgets.hovered = widget(p.field_hover, p.rule_strong, p.text);
    visuals.widgets.active = widget(p.select, p.select_line, p.text);
    visuals.widgets.open = widget(p.field_hover, p.rule_strong, p.text);
    visuals.selection.bg_fill = p.select;
    visuals.selection.stroke = Stroke::new(1.0, p.text);
    visuals.hyperlink_color = p.text_2;
    visuals.faint_bg_color = p.panel;
    visuals.extreme_bg_color = p.field;
    visuals.text_edit_bg_color = Some(p.field);
    visuals.code_bg_color = p.field;
    visuals.warn_fg_color = p.warn;
    visuals.error_fg_color = p.error;
    visuals.window_fill = p.window;
    visuals.window_stroke = Stroke::new(1.0, p.rule_strong);
    visuals.window_shadow = egui::Shadow::NONE;
    visuals.popup_shadow = egui::Shadow::NONE;
    visuals.window_corner_radius = radius;
    visuals.menu_corner_radius = radius;
    visuals.panel_fill = p.window;
    visuals.weak_text_color = Some(p.text_3);
    visuals.striped = false;
    visuals.slider_trailing_fill = true;
    ctx.all_styles_mut(|style| {
        style.visuals = visuals.clone();
        style.spacing.interact_size = Vec2::new(40.0, size::CONTROL_HEIGHT);
        style.spacing.item_spacing = Vec2::splat(size::GAP);
        style.spacing.button_padding = Vec2::new(size::BUTTON_PADDING, 2.0);
        style.spacing.window_margin = Margin::same(size::PAD_LARGE as i8);
        style.spacing.menu_margin = Margin::same(4);
        style.spacing.combo_width = size::FIELD_MIN_WIDTH;
        // Scroll bars: a 10 px track, shown whenever the content overflows.
        style.spacing.scroll = egui::style::ScrollStyle {
            floating: false,
            // A 6 px handle with 2 px around it: a 10 px track.
            bar_width: 6.0,
            handle_min_length: 24.0,
            bar_inner_margin: 2.0,
            bar_outer_margin: 2.0,
            foreground_color: false,
            ..egui::style::ScrollStyle::solid()
        };
        style.text_styles = [
            (TextStyle::Body, FontId::new(typeface::BASE, FontFamily::Proportional)),
            (TextStyle::Button, FontId::new(typeface::BASE, FontFamily::Proportional)),
            (TextStyle::Small, FontId::new(typeface::SMALL, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(typeface::BASE, FontFamily::Monospace)),
            (TextStyle::Heading, FontId::new(typeface::HEADING, FontFamily::Proportional)),
        ]
        .into();
    });
}

/// Glyphs: stroked paths on a 12 × 12 grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glyph {
    ChevronDown,
    ChevronUp,
    ChevronRight,
    ChevronLeft,
    SortUp,
    SortDown,
    Filter,
    Range,
    Check,
    /// The check of a checked menu entry: Windows' heavier mark.
    MenuCheck,
    Dash,
    /// The clear button of a text field.
    Cross,
}

impl Glyph {
    /// Polylines of the glyph on the 12 × 12 grid.
    fn strokes(self) -> &'static [&'static [(f32, f32)]] {
        match self {
            Glyph::ChevronDown => &[&[(2.5, 4.5), (6.0, 8.0), (9.5, 4.5)]],
            Glyph::ChevronUp => &[&[(2.5, 7.5), (6.0, 4.0), (9.5, 7.5)]],
            Glyph::ChevronRight => &[&[(4.5, 2.5), (8.0, 6.0), (4.5, 9.5)]],
            Glyph::ChevronLeft => &[&[(7.5, 2.5), (4.0, 6.0), (7.5, 9.5)]],
            Glyph::SortUp => &[&[(6.0, 10.0), (6.0, 2.0)], &[(3.0, 5.0), (6.0, 2.0), (9.0, 5.0)]],
            Glyph::SortDown => &[&[(6.0, 2.0), (6.0, 10.0)], &[(3.0, 7.0), (6.0, 10.0), (9.0, 7.0)]],
            Glyph::Filter => &[&[(1.5, 2.5), (10.5, 2.5), (7.0, 6.5), (7.0, 10.0), (5.0, 9.0), (5.0, 6.5), (1.5, 2.5)]],
            Glyph::Check => &[&[(2.5, 6.2), (5.0, 8.6), (9.5, 3.6)]],
            Glyph::MenuCheck => &[&[(4.0, 6.0), (6.0, 8.2), (10.0, 3.5)]],
            Glyph::Dash => &[&[(3.0, 6.0), (9.0, 6.0)]],
            Glyph::Cross => &[&[(3.5, 3.5), (8.5, 8.5)], &[(8.5, 3.5), (3.5, 8.5)]],
            Glyph::Range => &[
                &[(1.0, 3.0), (1.0, 9.0)],
                &[(11.0, 3.0), (11.0, 9.0)],
                &[(3.0, 6.0), (9.0, 6.0)],
                &[(4.6, 4.4), (3.0, 6.0), (4.6, 7.6)],
                &[(7.4, 4.4), (9.0, 6.0), (7.4, 7.6)],
            ],
        }
    }

    /// Paints the glyph centred in `rect` at `side` pixels.
    pub fn paint(self, painter: &egui::Painter, center: egui::Pos2, side: f32, color: Color32) {
        let scale = side / 12.0;
        let origin = center - Vec2::splat(side / 2.0);
        let weight = if self == Glyph::MenuCheck { 1.9 } else { 1.4 };
        let stroke = Stroke::new(weight * scale.max(0.8), color);
        for line in self.strokes() {
            let points: Vec<egui::Pos2> = line.iter().map(|&(x, y)| origin + Vec2::new(x, y) * scale).collect();
            painter.add(egui::Shape::line(points, stroke));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_of(c: Color32) -> String {
        format!("#{:02X}{:02X}{:02X}", c.r(), c.g(), c.b())
    }

    #[test]
    fn neutral_greys_match_the_table() {
        let dark = Palette::new(true, 0);
        let light = Palette::new(false, 0);
        let pairs = [
            (dark.canvas, light.canvas, "#191919", "#FDFDFD"),
            (dark.window, light.window, "#1E1E1E", "#F5F5F5"),
            (dark.panel, light.panel, "#262626", "#EEEEEE"),
            (dark.field, light.field, "#303030", "#FFFFFF"),
            (dark.field_hover, light.field_hover, "#393939", "#F3F3F3"),
            (dark.rule, light.rule, "#3D3D3D", "#D4D4D4"),
            (dark.rule_strong, light.rule_strong, "#515151", "#B7B7B7"),
            (dark.text, light.text, "#F9F9F9", "#222222"),
            (dark.text_2, light.text_2, "#C0C0C0", "#505050"),
            (dark.text_3, light.text_3, "#9A9A9A", "#7A7A7A"),
            (dark.select, light.select, "#494949", "#DCDCDC"),
            (dark.select_line, light.select_line, "#838383", "#808080"),
            (dark.construction, light.construction, "#8C8C8C", "#868686"),
        ];
        for (d, l, dh, lh) in pairs {
            assert_eq!((hex_of(d).as_str(), hex_of(l).as_str()), (dh, lh));
        }
    }

    #[test]
    fn tinted_greys() {
        assert_eq!(hex_of(Palette::new(true, 100).canvas), "#1A1915");
        assert_eq!(hex_of(Palette::new(false, 100).window), "#F9F5E6");
        assert_eq!(hex_of(Palette::new(true, -100).canvas), "#17191C");
        assert_eq!(hex_of(Palette::new(false, -100).panel), "#E0F0FF");
    }
}
