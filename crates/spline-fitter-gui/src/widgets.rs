//! Controls of the Loft design: form rows, segmented controls, numeric
//! fields with step arrows, the sweep toggle, captions and headers.

use eframe::egui::{self, Align, Button, CornerRadius, FontId, Key, Layout, Response, RichText, Sense, TextEdit, Ui, Vec2};
use spline_fitter::formatting::{decimal_point, number};

use crate::theme::{size, typeface, Glyph, Palette, INK, INK_HOVER, ON_INK};

/// An uppercase caption naming a region.
pub fn caption(ui: &mut Ui, p: &Palette, text: &str) -> Response {
    let rich = RichText::new(text.to_uppercase()).font(crate::fonts::semibold_id(typeface::CAPTION)).color(p.text_3).extra_letter_spacing(1.0);
    let galley = egui::WidgetText::from(rich).into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, egui::TextStyle::Body);
    // A line of a caption's height, the text at its top (as a Qt label).
    let (rect, response) = ui.allocate_exact_size(Vec2::new(galley.size().x, size::CAPTION_LINE), Sense::hover());
    ui.painter().galley(rect.min, galley, p.text_3);
    response
}

/// Muted text: hints and notes, a size below the base.
pub fn muted(ui: &mut Ui, p: &Palette, text: impl Into<String>) -> Response {
    ui.label(RichText::new(text.into()).size(typeface::MUTED).color(p.text_3))
}

/// A form row: a label column of fixed width, then the field.
pub fn form_row<R>(ui: &mut Ui, p: &Palette, label: &str, enabled: bool, tooltip: &str, field: impl FnOnce(&mut Ui) -> R) -> R {
    form_row_of(ui, p, label, enabled, tooltip, size::CONTROL_HEIGHT, field)
}

/// A form row holding a check box: as tall as a line of text.
pub fn check_row<R>(ui: &mut Ui, p: &Palette, label: &str, enabled: bool, field: impl FnOnce(&mut Ui) -> R) -> R {
    form_row_of(ui, p, label, enabled, "", size::CHECK_ROW_HEIGHT, field)
}

fn form_row_of<R>(ui: &mut Ui, p: &Palette, label: &str, enabled: bool, tooltip: &str, height: f32, field: impl FnOnce(&mut Ui) -> R) -> R {
    // A horizontal layout starts as tall as an interactive item.
    let interact = ui.spacing().interact_size.y;
    ui.spacing_mut().interact_size.y = height;
    let inner = ui
        .horizontal(|ui| {
            ui.set_min_height(height);
            let color = if enabled { p.text_2 } else { p.text_3 };
            let (rect, response) = ui.allocate_exact_size(Vec2::new(size::LABEL_WIDTH, height), Sense::hover());
            ui.add_space(size::GROUP - ui.spacing().item_spacing.x);
            ui.painter().text(rect.left_center(), egui::Align2::LEFT_CENTER, label, FontId::proportional(typeface::BASE), color);
            if !tooltip.is_empty() {
                response.on_hover_text(tooltip);
            }
            field(ui)
        })
        .inner;
    ui.spacing_mut().interact_size.y = interact;
    inner
}

/// The primary action, in ink.
pub fn primary_button(ui: &mut Ui, text: &str, width: f32) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, size::CONTROL_HEIGHT), Sense::click());
    let fill = if response.is_pointer_button_down_on() {
        crate::theme::INK_PRESSED
    } else if response.hovered() {
        INK_HOVER
    } else {
        INK
    };
    ui.painter().rect_filled(rect, CornerRadius::same(size::RADIUS), fill);
    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, text, crate::fonts::semibold_id(typeface::BASE), ON_INK);
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// The outline of a checked button or segment: the selection line.
pub fn outline_checked(ui: &Ui, p: &Palette, response: &Response, radius: CornerRadius) {
    if ui.is_enabled() {
        ui.painter().rect_stroke(response.rect, radius, egui::Stroke::new(1.0, p.select_line), egui::StrokeKind::Inside);
    }
}

/// A tool button's width: Qt's text width and the padding beside it.
pub fn tool_button_width(ui: &Ui, text: &str) -> f32 {
    crate::fonts::layout_width(ui, text, &FontId::proportional(typeface::BASE), false) + 2.0 * size::TOOL_BUTTON_PADDING
}

/// A checkable tool button, outlined when checked.
pub fn toggle_button(ui: &mut Ui, p: &Palette, text: &str, checked: bool, tooltip: &str) -> Response {
    let padding = ui.spacing().button_padding.x;
    ui.spacing_mut().button_padding.x = 0.0;
    let width = tool_button_width(ui, text);
    let response = ui.add(Button::new(text).selected(checked).min_size(Vec2::new(width, size::CONTROL_HEIGHT)));
    ui.spacing_mut().button_padding.x = padding;
    if checked {
        outline_checked(ui, p, &response, CornerRadius::same(size::RADIUS));
    }
    if tooltip.is_empty() {
        response
    } else {
        response.on_hover_text(tooltip)
    }
}

/// A check box: a 14 px box (the selection grey with a check when
/// checked), 6 px, then its text; a line of text tall.
pub fn checkbox(ui: &mut Ui, p: &Palette, checked: &mut bool, text: impl Into<RichText>) -> Response {
    tristate_checkbox(ui, p, checked, false, text)
}

/// The width of a check box with `text`: its box, a gap and the text as Qt
/// measures it.
pub fn checkbox_width(ui: &Ui, text: &str) -> f32 {
    size::CHECK_BOX + size::GAP + crate::fonts::layout_width(ui, text, &FontId::proportional(typeface::BASE), false)
}

/// A check box that can show a dash: some of what it stands for checked.
pub fn tristate_checkbox(ui: &mut Ui, p: &Palette, checked: &mut bool, partly: bool, text: impl Into<RichText>) -> Response {
    const BOX: f32 = size::CHECK_BOX;
    let enabled = ui.is_enabled();
    let text: RichText = text.into();
    let width = checkbox_width(ui, text.text());
    let color = if enabled { p.text } else { p.text_3 };
    let galley = egui::WidgetText::from(text).into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, egui::TextStyle::Body);
    let height = galley.size().y.max(BOX);
    let (rect, mut response) = ui.allocate_exact_size(Vec2::new(width, height), Sense::click());
    if response.clicked() {
        *checked = !*checked;
        response.mark_changed();
    }
    let indicator = egui::Rect::from_center_size(egui::pos2(rect.left() + BOX / 2.0, rect.center().y), Vec2::splat(BOX));
    let (fill, border) = match (enabled, *checked || partly) {
        (false, _) => (p.panel, p.rule),
        (true, true) => (p.select, p.select_line),
        (true, false) if response.hovered() => (p.field, p.select_line),
        (true, false) => (p.field, p.rule_strong),
    };
    ui.painter().rect(indicator, CornerRadius::same(2), fill, egui::Stroke::new(1.0, border), egui::StrokeKind::Inside);
    // The glyph fills the box inside its border.
    if partly {
        Glyph::Dash.paint(ui.painter(), indicator.center(), BOX - 2.0, p.text);
    } else if *checked {
        Glyph::Check.paint(ui.painter(), indicator.center(), BOX - 2.0, p.text);
    }
    ui.painter().galley(egui::pos2(indicator.right() + size::GAP, rect.center().y - galley.size().y / 2.0), galley, color);
    response.on_hover_cursor(egui::CursorIcon::Default)
}

/// A vertical scroll bar in `track` for a view showing `view` of `content`
/// from `offset`, drawn as the theme's (a handle in the strong rule, the
/// text grey when hovered). Dragging the handle or clicking the track
/// (a page) returns the new offset.
pub fn vertical_scroll_bar(ui: &mut Ui, p: &Palette, id: egui::Id, track: egui::Rect, offset: f32, content: f32, view: f32) -> Option<f32> {
    let range = (content - view).max(0.0);
    if range <= 0.0 {
        return None;
    }
    let inner = track.shrink(size::SCROLL_HANDLE_MARGIN);
    let length = (inner.height() * view / content).max(size::SCROLL_HANDLE_MIN).min(inner.height());
    let travel = (inner.height() - length).max(1.0);
    let top = inner.top() + travel * (offset / range).clamp(0.0, 1.0);
    let handle = egui::Rect::from_min_size(egui::pos2(inner.left(), top), Vec2::new(inner.width(), length));
    let track_response = ui.interact(track, id.with("track"), Sense::click());
    let response = ui.interact(handle, id.with("handle"), Sense::drag());
    let color = if response.hovered() || response.dragged() { p.text_3 } else { p.rule_strong };
    ui.painter().rect_filled(handle, CornerRadius::same(3), color);
    if response.dragged() {
        return Some((offset + response.drag_delta().y * range / travel).clamp(0.0, range));
    }
    if track_response.clicked() {
        let y = track_response.interact_pointer_pos()?.y;
        let page = if y < handle.top() { -view } else { view };
        return Some((offset + page).clamp(0.0, range));
    }
    None
}

/// The width Qt gives a spin box that can show `texts` (its least and
/// greatest values): the widest with a space after it, in the number face,
/// and the room for the cursor, frame and arrows.
pub fn spin_width(ui: &Ui, texts: &[&str]) -> f32 {
    let font = FontId::monospace(typeface::BASE);
    texts.iter().map(|t| crate::fonts::layout_width(ui, &format!("{t} "), &font, false)).fold(0.0, f32::max) + size::SPIN_EXTRA
}

/// Qt selects a field's text when the focus comes by keyboard (or with a
/// dialog's opening), not by a click: typing then replaces it.
fn select_on_focus(ui: &Ui, response: &Response, chars: usize) {
    if !response.gained_focus() || ui.input(|i| i.pointer.any_pressed() || i.pointer.any_down()) {
        return;
    }
    if let Some(mut state) = TextEdit::load_state(ui.ctx(), response.id) {
        let all = egui::text::CCursorRange::two(egui::text::CCursor::new(0), egui::text::CCursor::new(chars));
        state.cursor.set_char_range(Some(all));
        state.store(ui.ctx(), response.id);
    }
}

/// A one-line text field in the Loft frame (as a spin box's, without
/// arrows); at most `limit` characters (0: any).
pub fn text_field(ui: &mut Ui, p: &Palette, id_salt: &str, text: &mut String, width: f32, hint: &str, limit: usize) -> Response {
    field_of(ui, p, id_salt, text, width, hint, limit, false).0
}

/// A text field with a clear button (a cross at the right while it holds
/// text); true with the response when the button was clicked. The caller
/// empties the field (it may first use what it held).
pub fn clearable_field(ui: &mut Ui, p: &Palette, id_salt: &str, text: &mut String, width: f32, hint: &str) -> (Response, bool) {
    field_of(ui, p, id_salt, text, width, hint, 0, true)
}

#[allow(clippy::too_many_arguments)]
fn field_of(
    ui: &mut Ui,
    p: &Palette,
    id_salt: &str,
    text: &mut String,
    width: f32,
    hint: &str,
    limit: usize,
    clearable: bool,
) -> (Response, bool) {
    const CLEAR: f32 = 20.0;
    let enabled = ui.is_enabled();
    let (outer, frame_response) = ui.allocate_exact_size(Vec2::new(width, size::CONTROL_HEIGHT), Sense::hover());
    let show_clear = clearable && enabled && !text.is_empty();
    let edit_rect = if clearable { egui::Rect::from_min_max(outer.min, egui::pos2(outer.right() - CLEAR, outer.bottom())) } else { outer };
    let background = ui.painter().add(egui::Shape::Noop);
    let mut edit = TextEdit::singleline(text)
        .id(ui.make_persistent_id(id_salt))
        .frame(egui::Frame::NONE.inner_margin(egui::Margin {
            left: size::FIELD_TEXT_INSET as i8,
            right: size::FIELD_TEXT_INSET as i8,
            top: 0,
            bottom: 0,
        }))
        .hint_text(hint)
        .align(egui::Align2::LEFT_CENTER)
        .desired_width(edit_rect.width() - 2.0 * size::FIELD_TEXT_INSET)
        .min_size(edit_rect.size());
    if limit > 0 {
        edit = edit.char_limit(limit);
    }
    let response = ui.new_child(egui::UiBuilder::new().max_rect(edit_rect)).add(edit);
    select_on_focus(ui, &response, text.chars().count());
    let mut cleared = false;
    if show_clear {
        let button = egui::Rect::from_min_max(egui::pos2(outer.right() - CLEAR, outer.top()), outer.max);
        let clear =
            ui.interact(button, ui.make_persistent_id((id_salt, "clear")), Sense::click()).on_hover_cursor(egui::CursorIcon::Default);
        let color = if clear.hovered() { p.text } else { p.text_3 };
        Glyph::Cross.paint(ui.painter(), button.center(), size::GLYPH, color);
        cleared = clear.clicked();
    }
    let border = if !enabled {
        p.rule
    } else if response.has_focus() {
        p.select_line
    } else if response.hovered() || frame_response.hovered() {
        p.rule_strong
    } else {
        p.rule
    };
    ui.painter().set(
        background,
        egui::Shape::Rect(egui::epaint::RectShape::new(
            outer,
            CornerRadius::same(size::RADIUS),
            if enabled { p.field } else { p.panel },
            egui::Stroke::new(1.0, border),
            egui::StrokeKind::Inside,
        )),
    );
    (response, cleared)
}

/// The temperature slider: a 4 px groove from cool to warm and a 12 px
/// round handle; values -100 to 100.
pub fn temperature_slider(ui: &mut Ui, p: &Palette, value: &mut i32, width: f32) -> Response {
    let height = 24.0;
    let (rect, mut response) = ui.allocate_exact_size(Vec2::new(width, height), Sense::click_and_drag());
    let handle = 12.0;
    let span = (rect.left() + handle / 2.0)..=(rect.right() - handle / 2.0);
    if let Some(pos) = response.interact_pointer_pos().filter(|_| response.dragged() || response.clicked()) {
        let t = ((pos.x - span.start()) / (span.end() - span.start())).clamp(0.0, 1.0);
        let v = (-100.0 + 200.0 * t).round() as i32;
        if v != *value {
            *value = v;
            response.mark_changed();
        }
    }
    // The groove, exaggerated from cool to warm so it can be read.
    let groove = egui::Rect::from_center_size(rect.center(), Vec2::new(width, 4.0));
    let steps = 32;
    let mut mesh = egui::Mesh::default();
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let color = crate::theme::grey(0.62, -4.0 + 8.0 * t as f64);
        let x = groove.left() + t * groove.width();
        mesh.colored_vertex(egui::pos2(x, groove.top()), color);
        mesh.colored_vertex(egui::pos2(x, groove.bottom()), color);
        if i > 0 {
            let k = 2 * i as u32;
            mesh.add_triangle(k - 2, k - 1, k);
            mesh.add_triangle(k - 1, k, k + 1);
        }
    }
    ui.painter().add(egui::Shape::mesh(mesh));
    let x = span.start() + (*value as f32 + 100.0) / 200.0 * (span.end() - span.start());
    let fill = if response.hovered() || response.dragged() { p.text } else { p.text_2 };
    ui.painter().circle(egui::pos2(x, rect.center().y), handle / 2.0, fill, egui::Stroke::new(1.0, p.canvas));
    response
}

/// A flat square button showing a glyph: no frame until hovered.
pub fn flat_icon_button(ui: &mut Ui, p: &Palette, glyph: Glyph, tooltip: &str) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(size::CONTROL_HEIGHT), Sense::click());
    if response.hovered() && ui.is_enabled() {
        ui.painter().rect_filled(rect, CornerRadius::same(size::RADIUS), p.field_hover);
    }
    glyph.paint(ui.painter(), rect.center(), size::GLYPH, if ui.is_enabled() { p.text_2 } else { p.text_3 });
    if tooltip.is_empty() {
        response
    } else {
        response.on_hover_text(tooltip)
    }
}

/// A square button showing a glyph.
pub fn icon_button(ui: &mut Ui, p: &Palette, glyph: Glyph, checked: bool, tooltip: &str) -> Response {
    let side = size::CONTROL_HEIGHT;
    let response = ui.add(Button::new("").min_size(Vec2::splat(side)).selected(checked));
    if checked {
        outline_checked(ui, p, &response, CornerRadius::same(size::RADIUS));
    }
    let color = if !ui.is_enabled() {
        p.text_3
    } else if checked {
        p.text
    } else {
        p.text_2
    };
    let glyph_side = if glyph == Glyph::Range { size::GLYPH_LARGE } else { size::GLYPH };
    glyph.paint(ui.painter(), response.rect.center(), glyph_side, color);
    if tooltip.is_empty() {
        response
    } else {
        response.on_hover_text(tooltip)
    }
}

/// Joined buttons for a choice of two to four short options; returns true
/// when the choice changed.
pub fn segmented<T: PartialEq + Copy>(ui: &mut Ui, value: &mut T, items: &[(T, &str, &str)], width: f32) -> bool {
    let select_line = ui.visuals().widgets.active.bg_stroke.color;
    let mut changed = false;
    let n = items.len() as f32;
    let each = (width / n).floor();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        for (i, (item, text, tooltip)) in items.iter().enumerate() {
            let r = size::RADIUS;
            let radius = CornerRadius {
                nw: if i == 0 { r } else { 0 },
                sw: if i == 0 { r } else { 0 },
                ne: if i + 1 == items.len() { r } else { 0 },
                se: if i + 1 == items.len() { r } else { 0 },
            };
            let checked = *value == *item;
            let mut response = ui.add_sized([each, size::CONTROL_HEIGHT], Button::new(*text).selected(checked).corner_radius(radius));
            // Outlined when checked, and while it has the focus (Qt's segments
            // are push buttons: the focused one is the default).
            if (checked || response.has_focus()) && ui.is_enabled() {
                ui.painter().rect_stroke(response.rect, radius, egui::Stroke::new(1.0, select_line), egui::StrokeKind::Inside);
            }
            if !tooltip.is_empty() {
                response = response.on_hover_text(*tooltip);
            }
            if response.clicked() && *value != *item {
                *value = *item;
                changed = true;
            }
        }
    });
    changed
}

/// Width and tooltip of a field.
#[derive(Debug, Clone, Copy, Default)]
pub struct Field<'a> {
    pub width: f32,
    pub tooltip: &'a str,
}

impl<'a> Field<'a> {
    pub fn new(width: f32, tooltip: &'a str) -> Self {
        Field { width, tooltip }
    }
}

/// How an integer field shows its value.
#[derive(Debug, Clone, Copy)]
pub struct IntOptions<'a> {
    /// Text after the number (` points`).
    pub suffix: &'a str,
    /// Text shown in place of 0 (`Off`).
    pub zero: Option<&'a str>,
    /// Change of one arrow step.
    pub step: usize,
}

impl Default for IntOptions<'_> {
    fn default() -> Self {
        IntOptions { suffix: "", zero: None, step: 1 }
    }
}

/// How a numeric field shows, reads and steps its value.
pub struct NumberFormat<'a> {
    pub format: &'a dyn Fn(f64) -> String,
    pub parse: &'a dyn Fn(&str) -> Option<f64>,
    pub step: &'a dyn Fn(f64, i32) -> f64,
}

/// Whether the edge at `x` of a resizable panel spanning `rect` was
/// double-clicked this frame.
pub fn edge_double_clicked(ctx: &egui::Context, rect: egui::Rect, x: f32) -> bool {
    let grab = ctx.global_style().interaction.resize_grab_radius_side;
    ctx.input(|i| {
        i.pointer.button_double_clicked(egui::PointerButton::Primary)
            && i.pointer.interact_pos().is_some_and(|p| (p.x - x).abs() <= grab && rect.y_range().contains(p.y))
    })
}

/// The text being typed into a numeric field, until it is committed.
#[derive(Clone, Default)]
struct EditBuffer(String);

/// Drops what is being typed into numeric fields, so they show their
/// values again (after an undo or redo changed them).
pub fn forget_edits(ctx: &egui::Context) {
    ctx.data_mut(|d| d.remove_by_type::<EditBuffer>());
}

/// A spin box: a text field holding a number, with step arrows inside its
/// frame (hidden while disabled); edits apply on Enter or when the field
/// loses focus. Returns true when the value changed.
pub fn number_field(
    ui: &mut Ui,
    p: &Palette,
    id_salt: impl std::hash::Hash + std::fmt::Debug,
    value: &mut f64,
    range: (f64, f64),
    field: Field,
    fmt: &NumberFormat,
) -> bool {
    let Field { width, tooltip } = field;
    let id = ui.make_persistent_id(id_salt);
    let buffer_id = id.with("buffer");
    let mut changed = false;
    let arrows = 16.0;
    let enabled = ui.is_enabled();
    let (outer, frame_response) = ui.allocate_exact_size(Vec2::new(width.max(40.0), size::CONTROL_HEIGHT), Sense::hover());
    let background = ui.painter().add(egui::Shape::Noop);
    let editing: Option<String> = ui.data(|d| d.get_temp::<EditBuffer>(buffer_id)).map(|b| b.0);
    let mut text = editing.clone().unwrap_or_else(|| (fmt.format)(*value));
    let text_rect = egui::Rect::from_min_max(outer.min, egui::pos2(outer.right() - arrows, outer.bottom()));
    let edit = TextEdit::singleline(&mut text)
        .id(id)
        .frame(egui::Frame::NONE.inner_margin(egui::Margin { left: size::FIELD_TEXT_INSET as i8, right: 0, top: 0, bottom: 0 }))
        .font(FontId::monospace(typeface::BASE))
        .align(egui::Align2::LEFT_CENTER)
        .desired_width(text_rect.width() - size::FIELD_TEXT_INSET)
        .min_size(text_rect.size());
    // In a child, so the layout cursor stays after the whole field.
    let response = ui.new_child(egui::UiBuilder::new().max_rect(text_rect)).add(edit);
    // A spin box selects its number, not the unit after it.
    let number = if text.starts_with(|c: char| c.is_ascii_digit() || c == '-') {
        text.chars().take_while(|c| !c.is_whitespace()).count()
    } else {
        text.chars().count()
    };
    select_on_focus(ui, &response, number);
    let mut response = if tooltip.is_empty() { response } else { response.on_hover_text(tooltip) };
    let commit = |text: &str, value: &mut f64| {
        if let Some(v) = (fmt.parse)(text) {
            let v = v.clamp(range.0, range.1);
            if v != *value {
                *value = v;
                return true;
            }
        }
        false
    };
    if response.has_focus() {
        let (up, down) = ui.input(|i| (i.key_pressed(Key::ArrowUp), i.key_pressed(Key::ArrowDown)));
        if up || down {
            let v = (fmt.step)(*value, if up { 1 } else { -1 }).clamp(range.0, range.1);
            changed |= v != *value;
            *value = v;
            text = (fmt.format)(*value);
        }
        ui.data_mut(|d| d.insert_temp(buffer_id, EditBuffer(text.clone())));
    } else if editing.is_some() {
        changed |= commit(&text, value);
        ui.data_mut(|d| d.remove::<EditBuffer>(buffer_id));
    }
    if response.lost_focus() && editing.is_some() {
        changed |= commit(&text, value);
        ui.data_mut(|d| d.remove::<EditBuffer>(buffer_id));
    }
    // Step arrows inside the frame, gone while disabled.
    let arrow_rect = egui::Rect::from_min_max(egui::pos2(outer.right() - arrows, outer.top()), outer.max);
    let half = arrow_rect.height() / 2.0;
    let mut hovered = frame_response.hovered() || response.hovered();
    if enabled {
        for (k, glyph) in [(1, Glyph::ChevronUp), (-1, Glyph::ChevronDown)] {
            let r = if k == 1 {
                egui::Rect::from_min_size(arrow_rect.min, Vec2::new(arrows, half))
            } else {
                egui::Rect::from_min_size(arrow_rect.min + Vec2::new(0.0, half), Vec2::new(arrows, half))
            };
            let arrow = ui.interact(r, id.with(k), Sense::click());
            hovered |= arrow.hovered();
            if arrow.hovered() {
                ui.painter().rect_filled(r.shrink(1.0), 0.0, p.field_hover);
            }
            glyph.paint(ui.painter(), r.center(), 9.0, p.text_2);
            if arrow.clicked() {
                let v = (fmt.step)(*value, k).clamp(range.0, range.1);
                changed |= v != *value;
                *value = v;
                ui.data_mut(|d| d.remove::<EditBuffer>(buffer_id));
            }
        }
    }
    let border = if !enabled {
        p.rule
    } else if response.has_focus() {
        p.select_line
    } else if hovered {
        p.rule_strong
    } else {
        p.rule
    };
    let radius = CornerRadius::same(size::RADIUS);
    ui.painter().set(
        background,
        egui::Shape::Rect(egui::epaint::RectShape::new(
            outer,
            radius,
            if enabled { p.field } else { p.panel },
            egui::Stroke::new(1.0, border),
            egui::StrokeKind::Inside,
        )),
    );
    if changed {
        response.mark_changed();
    }
    changed
}

/// An integer field (`suffix` after the number, `zero` in place of 0).
pub fn int_field(
    ui: &mut Ui,
    p: &Palette,
    id_salt: impl std::hash::Hash + std::fmt::Debug,
    value: &mut usize,
    range: (usize, usize),
    field: Field,
    options: IntOptions,
) -> bool {
    let IntOptions { suffix, zero, step } = options;
    let format = |v: f64| match zero {
        Some(z) if v == 0.0 => z.to_string(),
        _ => format!("{}{suffix}", v as usize),
    };
    let parse = |t: &str| {
        let t = t.trim().trim_end_matches(suffix.trim()).trim();
        if zero.is_some_and(|z| z.eq_ignore_ascii_case(t)) {
            return Some(0.0);
        }
        t.parse::<i64>().ok().map(|v| v.max(0) as f64)
    };
    let stepper = |v: f64, k: i32| (v + (k as f64) * step as f64).max(0.0);
    let fmt = NumberFormat { format: &format, parse: &parse, step: &stepper };
    let mut v = *value as f64;
    let changed = number_field(ui, p, id_salt, &mut v, (range.0 as f64, range.1 as f64), field, &fmt);
    *value = v as usize;
    changed
}

/// Parses a number written with the display decimal point.
pub fn parse_scientific(text: &str) -> Option<f64> {
    let point = decimal_point();
    let t = text.trim();
    if t.is_empty() || !t.chars().all(|c| c.is_ascii_digit() || "eE+-.".contains(c) || c == point) {
        return None;
    }
    t.replace(point, ".").parse::<f64>().ok().filter(|v| v.is_finite())
}

/// One ladder step (1, 2, 5 per decade) up or down from `v`.
pub fn ladder_step(v: f64, up: bool, single_step: f64, minimum: f64) -> f64 {
    if v <= 0.0 {
        return if up { single_step } else { 0.0 };
    }
    let e = v.log10().floor() as i32;
    let mut ladder: Vec<f64> = (e - 1..=e + 1).flat_map(|k| [1.0, 2.0, 5.0].map(|m| m * 10f64.powf(k as f64))).collect();
    ladder.sort_by(f64::total_cmp);
    if up {
        ladder.into_iter().find(|&x| x > v * (1.0 + 1e-9)).unwrap_or(v)
    } else {
        ladder.into_iter().rev().find(|&x| x < v * (1.0 - 1e-9)).filter(|&x| x >= minimum).unwrap_or(minimum)
    }
}

/// A field for values spanning decades: six significant digits, stepping
/// along 1-2-5 ladders.
pub fn scientific_field(
    ui: &mut Ui,
    p: &Palette,
    id_salt: impl std::hash::Hash + std::fmt::Debug,
    value: &mut f64,
    range: (f64, f64),
    single_step: f64,
    field: Field,
) -> bool {
    let min = range.0;
    let format = |v: f64| number(v, ".6g");
    let parse = |t: &str| parse_scientific(t);
    let step = |v: f64, k: i32| {
        let mut x = v;
        for _ in 0..k.unsigned_abs() {
            x = ladder_step(x, k > 0, single_step, min);
        }
        x
    };
    let fmt = NumberFormat { format: &format, parse: &parse, step: &step };
    number_field(ui, p, id_salt, value, range, field, &fmt)
}

/// The sweep toggle: a dimension line `|←→|`.
pub fn sweep_toggle(ui: &mut Ui, p: &Palette, active: &mut bool, allowed: bool) -> bool {
    const TIP: &str = "Sweep a range of values: one fit per value, or per combination with the other swept settings";
    let response = ui.add_enabled_ui(allowed && ui.is_enabled(), |ui| icon_button(ui, p, Glyph::Range, *active, TIP)).inner;
    if response.clicked() {
        *active = !*active;
        return true;
    }
    false
}

/// The header of a collapsible section: chevron, uppercase title, badge.
/// The chevron and title toggle the section and brighten on hover.
pub fn collapsible_header(ui: &mut Ui, p: &Palette, title: &str, open: &mut bool, badge: &str, badge_tip: &str) {
    let font = crate::fonts::semibold_id(typeface::CAPTION);
    let text = title.to_uppercase();
    let galley = ui.painter().layout_no_wrap(text.clone(), font.clone(), p.text_3);
    let height = galley.size().y.max(size::CAPTION_LINE);
    let indent = size::GLYPH + 9.0;
    let interact = ui.spacing().interact_size.y;
    ui.spacing_mut().interact_size.y = height;
    ui.horizontal(|ui| {
        let (rect, response) =
            ui.allocate_exact_size(Vec2::new(indent + galley.size().x + 1.0 * text.len() as f32, height), Sense::click());
        let response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
        let color = if response.hovered() { p.text } else { p.text_3 };
        let glyph = if *open { Glyph::ChevronDown } else { Glyph::ChevronRight };
        // Text and chevron on the caption's line, at the top as in a label.
        let middle = rect.top() + galley.size().y / 2.0;
        glyph.paint(ui.painter(), egui::pos2(rect.left() + size::GLYPH / 2.0, middle), size::GLYPH, color);
        let job = egui::text::LayoutJob::single_section(
            text,
            egui::TextFormat { font_id: font, color, extra_letter_spacing: 1.0, ..Default::default() },
        );
        let galley = ui.painter().layout_job(job);
        ui.painter().galley(egui::pos2(rect.left() + indent, rect.top()), galley, color);
        if response.clicked() {
            *open = !*open;
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if !badge.is_empty() {
                let r = ui.label(RichText::new(badge).color(p.text_2).size(typeface::CAPTION));
                r.on_hover_text(badge_tip);
            }
        });
    });
    ui.spacing_mut().interact_size.y = interact;
}

/// A thin horizontal rule.
/// A size grip with its bottom right at `corner`: a triangle of dots that
/// resizes the window when dragged.
pub fn size_grip(ui: &mut Ui, p: &Palette, corner: egui::Pos2) {
    let rect = egui::Rect::from_min_max(corner - Vec2::splat(size::GRIP), corner);
    let response = ui.interact(rect, ui.scope_id().with("size_grip"), Sense::drag()).on_hover_cursor(egui::CursorIcon::ResizeSouthEast);
    if response.drag_started() {
        ui.ctx().send_viewport_cmd(egui::ViewportCommand::BeginResize(egui::viewport::ResizeDirection::SouthEast));
    }
    // Four rows of 2 px dots, 3 px apart, 1 px in from the corner.
    for row in 0..4 {
        for col in 0..=row {
            let at = corner - Vec2::new(3.0 + 3.0 * col as f32, 3.0 + 3.0 * (3 - row) as f32);
            ui.painter().rect_filled(egui::Rect::from_min_size(at, Vec2::splat(2.0)), 0.0, p.rule_strong);
        }
    }
}

pub fn rule(ui: &mut Ui, p: &Palette) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 1.0), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, p.rule);
}

/// Label and value pairs in as many columns as fit.
pub fn readout_list(ui: &mut Ui, p: &Palette, title: &str, pairs: &[(String, String)]) {
    // The title (a message when there are no pairs), muted.
    let color = if pairs.is_empty() { p.text_3 } else { p.text };
    ui.add(egui::Label::new(RichText::new(title).size(typeface::MUTED).color(color)).wrap());
    if pairs.is_empty() {
        return;
    }
    ui.add_space(size::GAP - ui.spacing().item_spacing.y);
    let label_font = FontId::proportional(typeface::MUTED);
    let value_font = FontId::monospace(typeface::BASE);
    let painter = ui.painter().clone();
    let labels: Vec<_> = pairs.iter().map(|(l, _)| painter.layout_no_wrap(l.clone(), label_font.clone(), p.text_3)).collect();
    let values: Vec<_> = pairs.iter().map(|(_, v)| painter.layout_no_wrap(v.clone(), value_font.clone(), p.text)).collect();
    let label_w: Vec<f32> = labels.iter().map(|g| g.size().x).collect();
    let value_w: Vec<f32> = values.iter().map(|g| g.size().x + size::GROUP).collect();
    // Per column of pairs: the widest label and value.
    let columns_of = |n: usize| -> Vec<(f32, f32)> {
        (0..n)
            .map(|c| {
                let idx = (c..pairs.len()).step_by(n);
                (idx.clone().map(|i| label_w[i]).fold(0.0, f32::max), idx.map(|i| value_w[i]).fold(0.0, f32::max))
            })
            .collect()
    };
    let needed = |n: usize| columns_of(n).iter().map(|(l, v)| l + v).sum::<f32>() + (n as f32 - 1.0) * size::PAD_LARGE;
    let available = ui.available_width();
    // As many pairs to a line as fit, the spare width shared between them.
    let n = (2..=pairs.len()).rev().find(|&n| needed(n) <= available).unwrap_or(1);
    let columns = columns_of(n);
    let spare = ((available - needed(n)) / n as f32).max(0.0);
    let line = labels.iter().chain(&values).map(|g| g.size().y).fold(0.0, f32::max);
    let rows = pairs.len().div_ceil(n);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(available, rows as f32 * (line + 1.0) - 1.0), Sense::hover());
    for (i, (label, value)) in labels.into_iter().zip(values).enumerate() {
        let (row, c) = (i / n, i % n);
        let x = rect.left() + columns[..c].iter().map(|(l, v)| l + v + size::PAD_LARGE + spare).sum::<f32>();
        let y = rect.top() + row as f32 * (line + 1.0);
        let (lh, vh) = (label.size().y, value.size().y);
        painter.galley(egui::pos2(x, y + (line - lh) / 2.0), label, p.text_3);
        painter.galley(egui::pos2(x + columns[c].0 + size::GROUP, y + (line - vh) / 2.0), value, p.text);
    }
}

/// A combo box of texts; returns true when the choice changed.
pub fn combo(
    ui: &mut Ui,
    p: &Palette,
    id_salt: impl std::hash::Hash + std::fmt::Debug,
    selected: &mut usize,
    items: &[&str],
    width: f32,
    tooltip: &str,
) -> bool {
    let before = *selected;
    // As wide as the longest item, its padding and the drop-down glyph.
    let width = if width > 0.0 { width } else { combo_width(ui, items) };
    let glyph = p.text_2;
    // Qt's layout: the text 8 px in, the chevron centred 11.5 px from the
    // right edge; the gap between them gives way to keep the width.
    let text = items.get(*selected).copied().unwrap_or("");
    let font = FontId::proportional(typeface::BASE);
    let text_width = ui.painter().layout_no_wrap(text.to_string(), font, p.text).size().x;
    let spacing = ui.spacing().clone();
    let pad = size::COMBO_PADDING;
    let icon = 2.0 * (size::COMBO_CHEVRON_INSET - pad);
    let gap = (width - 2.0 * pad - icon - text_width).clamp(0.0, size::COMBO_EXTRA);
    ui.spacing_mut().button_padding.x = pad;
    ui.spacing_mut().icon_width = icon;
    ui.spacing_mut().icon_spacing = gap;
    let response = egui::ComboBox::from_id_salt(id_salt)
        .width(width)
        .selected_text(text)
        .icon(move |ui, rect, _visuals, _open| Glyph::ChevronDown.paint(ui.painter(), rect.center(), size::GLYPH, glyph))
        .show_ui(ui, |ui| {
            *ui.spacing_mut() = spacing.clone();
            for (i, item) in items.iter().enumerate() {
                ui.selectable_value(selected, i, *item);
            }
        })
        .response;
    *ui.spacing_mut() = spacing;
    // Focused, its border is the selection line (as a field's).
    if response.has_focus() {
        let stroke = egui::Stroke::new(1.0, ui.visuals().widgets.active.bg_stroke.color);
        ui.painter().rect_stroke(response.rect, CornerRadius::same(size::RADIUS), stroke, egui::StrokeKind::Inside);
    }
    if !tooltip.is_empty() {
        response.on_hover_text(tooltip);
    }
    *selected != before
}

/// The width of a combo box that fits `items`: as Qt sizes one, the
/// longest item's ink and the room around it.
pub fn combo_width(ui: &Ui, items: &[&str]) -> f32 {
    let font = FontId::proportional(typeface::BASE);
    items.iter().map(|t| crate::fonts::ink_width(ui, t, &font)).fold(0.0, f32::max) + size::COMBO_EXTRA
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder() {
        assert_eq!(ladder_step(1e-5, true, 1e-6, 1e-9), 2e-5);
        assert_eq!(ladder_step(2e-5, true, 1e-6, 1e-9), 5e-5);
        assert!((ladder_step(5e-5, true, 1e-6, 1e-9) - 1e-4).abs() < 1e-18);
        assert_eq!(ladder_step(1e-5, false, 1e-6, 1e-9), 5.0 * 10f64.powf(-6.0));
        assert_eq!(ladder_step(0.0, true, 0.001, 0.0), 0.001);
        assert_eq!(ladder_step(0.0, false, 0.001, 0.0), 0.0);
    }

    #[test]
    fn parses_numbers() {
        assert_eq!(parse_scientific("1e-5"), Some(1e-5));
        assert_eq!(parse_scientific("abc"), None);
        assert_eq!(parse_scientific("inf"), None);
    }
}
