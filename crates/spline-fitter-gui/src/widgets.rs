//! Controls of the Loft design: form rows, segmented controls, numeric
//! fields with step arrows, the sweep toggle, captions and headers.

use eframe::egui::{
    self, Align, Button, Color32, CornerRadius, FontId, Key, Layout, Response, RichText, Sense, TextEdit,
    Ui, Vec2, Widget,
};
use spline_fitter::formatting::{decimal_point, number};

use crate::theme::{size, typeface, Glyph, Palette, INK, INK_HOVER, ON_INK};

/// An uppercase caption naming a region.
pub fn caption(ui: &mut Ui, p: &Palette, text: &str) -> Response {
    ui.label(RichText::new(text.to_uppercase()).size(typeface::CAPTION).strong().color(p.text_3).extra_letter_spacing(0.8))
}

/// Muted text: hints and notes.
pub fn muted(ui: &mut Ui, p: &Palette, text: impl Into<String>) -> Response {
    ui.label(RichText::new(text.into()).color(p.text_3))
}

/// A form row: a label column of fixed width, then the field.
pub fn form_row<R>(ui: &mut Ui, p: &Palette, label: &str, enabled: bool, tooltip: &str, field: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.set_min_height(size::CONTROL_HEIGHT);
        let color = if enabled { p.text_2 } else { p.text_3 };
        let (rect, response) = ui.allocate_exact_size(Vec2::new(size::LABEL_WIDTH, size::CONTROL_HEIGHT), Sense::hover());
        ui.painter().text(rect.left_center(), egui::Align2::LEFT_CENTER, label, FontId::proportional(typeface::BASE), color);
        if !tooltip.is_empty() {
            response.on_hover_text(tooltip);
        }
        field(ui)
    })
    .inner
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
    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, text, FontId::proportional(typeface::BASE), ON_INK);
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A square button showing a glyph.
pub fn icon_button(ui: &mut Ui, p: &Palette, glyph: Glyph, checked: bool, tooltip: &str) -> Response {
    let side = size::CONTROL_HEIGHT;
    let response = ui.add(Button::new("").min_size(Vec2::splat(side)).selected(checked));
    let color = if !ui.is_enabled() {
        p.text_3
    } else if checked {
        p.text
    } else {
        p.text_2
    };
    let glyph_side = if glyph == Glyph::Range { size::GLYPH_LARGE } else { size::GLYPH };
    glyph.paint(ui.painter(), response.rect.center(), glyph_side, color);
    if tooltip.is_empty() { response } else { response.on_hover_text(tooltip) }
}

/// Joined buttons for a choice of two to four short options; returns true
/// when the choice changed.
pub fn segmented<T: PartialEq + Copy>(ui: &mut Ui, value: &mut T, items: &[(T, &str, &str)], width: f32) -> bool {
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
            let mut response = ui.add_sized([each, size::CONTROL_HEIGHT], Button::new(*text).selected(*value == *item).corner_radius(radius));
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

/// How a numeric field shows, reads and steps its value.
pub struct NumberFormat<'a> {
    pub format: &'a dyn Fn(f64) -> String,
    pub parse: &'a dyn Fn(&str) -> Option<f64>,
    pub step: &'a dyn Fn(f64, i32) -> f64,
}

/// A text field holding a number, with step arrows; edits apply on Enter
/// or when the field loses focus. Returns true when the value changed.
pub fn number_field(ui: &mut Ui, p: &Palette, id_salt: impl std::hash::Hash + std::fmt::Debug, value: &mut f64, range: (f64, f64), width: f32, fmt: &NumberFormat, tooltip: &str) -> bool {
    let id = ui.make_persistent_id(id_salt);
    let buffer_id = id.with("buffer");
    let mut changed = false;
    let arrows = 16.0;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        let editing: Option<String> = ui.data(|d| d.get_temp(buffer_id));
        let mut text = editing.clone().unwrap_or_else(|| (fmt.format)(*value));
        let edit = TextEdit::singleline(&mut text)
            .id(id)
            .font(FontId::monospace(typeface::BASE))
            .desired_width((width - arrows).max(30.0))
            .align(egui::Align2::LEFT_CENTER)
            .min_size(Vec2::new(0.0, size::CONTROL_HEIGHT));
        let response = ui.add(edit);
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
            ui.data_mut(|d| d.insert_temp(buffer_id, text.clone()));
        } else if editing.is_some() {
            changed |= commit(&text, value);
            ui.data_mut(|d| d.remove::<String>(buffer_id));
        }
        if response.lost_focus() && editing.is_some() {
            changed |= commit(&text, value);
            ui.data_mut(|d| d.remove::<String>(buffer_id));
        }
        // Step arrows.
        let (rect, _) = ui.allocate_exact_size(Vec2::new(arrows, size::CONTROL_HEIGHT), Sense::hover());
        let half = rect.height() / 2.0;
        for (k, glyph) in [(1, Glyph::ChevronUp), (-1, Glyph::ChevronDown)] {
            let r = if k == 1 { egui::Rect::from_min_size(rect.min, Vec2::new(arrows, half)) } else { egui::Rect::from_min_size(rect.min + Vec2::new(0.0, half), Vec2::new(arrows, half)) };
            let arrow = ui.interact(r, id.with(k), Sense::click());
            if arrow.hovered() && ui.is_enabled() {
                ui.painter().rect_filled(r, 0.0, p.field_hover);
            }
            glyph.paint(ui.painter(), r.center(), 8.0, if ui.is_enabled() { p.text_2 } else { p.text_3 });
            if arrow.clicked() && ui.is_enabled() {
                let v = (fmt.step)(*value, k).clamp(range.0, range.1);
                changed |= v != *value;
                *value = v;
                ui.data_mut(|d| d.remove::<String>(buffer_id));
            }
        }
        if changed {
            response.mark_changed();
        }
    });
    changed
}

/// An integer field (`suffix` after the number, `zero` in place of 0).
pub fn int_field(ui: &mut Ui, p: &Palette, id_salt: impl std::hash::Hash + std::fmt::Debug, value: &mut usize, min: usize, max: usize, width: f32, suffix: &str, zero: Option<&str>, step: usize, tooltip: &str) -> bool {
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
    let changed = number_field(ui, p, id_salt, &mut v, (min as f64, max as f64), width, &fmt, tooltip);
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
pub fn scientific_field(ui: &mut Ui, p: &Palette, id_salt: impl std::hash::Hash + std::fmt::Debug, value: &mut f64, min: f64, max: f64, single_step: f64, width: f32, tooltip: &str) -> bool {
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
    number_field(ui, p, id_salt, value, (min, max), width, &fmt, tooltip)
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
pub fn collapsible_header(ui: &mut Ui, p: &Palette, title: &str, open: &mut bool, badge: &str, badge_tip: &str) {
    ui.horizontal(|ui| {
        let glyph = if *open { Glyph::ChevronDown } else { Glyph::ChevronRight };
        let text = RichText::new(format!("      {}", title.to_uppercase())).size(typeface::CAPTION).strong().color(p.text_3).extra_letter_spacing(0.8);
        let response = ui.add(Button::new(text).frame(false)).on_hover_cursor(egui::CursorIcon::PointingHand);
        glyph.paint(ui.painter(), response.rect.left_center() + Vec2::new(4.0, 0.0), size::GLYPH, p.text_3);
        if response.clicked() {
            *open = !*open;
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if !badge.is_empty() {
                let r = ui.label(RichText::new(badge).color(p.text_2).size(typeface::SMALL));
                r.on_hover_text(badge_tip);
            }
        });
    });
}

/// A thin horizontal rule.
pub fn rule(ui: &mut Ui, p: &Palette) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 1.0), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, p.rule);
}

/// Label and value pairs in as many columns as fit.
pub fn readout_list(ui: &mut Ui, p: &Palette, title: &str, pairs: &[(String, String)]) {
    if pairs.is_empty() {
        muted(ui, p, title);
        return;
    }
    ui.label(RichText::new(title).color(p.text).strong());
    let mono = FontId::monospace(typeface::BASE);
    let body = FontId::proportional(typeface::BASE);
    let width = |text: &str, font: &FontId| ui.fonts_mut(|f| f.layout_no_wrap(text.to_string(), font.clone(), Color32::WHITE).size().x);
    let label_w: Vec<f32> = pairs.iter().map(|(l, _)| width(l, &body)).collect();
    let value_w: Vec<f32> = pairs.iter().map(|(_, v)| width(v, &mono) + 8.0).collect();
    let available = ui.available_width();
    let columns = (1..=pairs.len().max(1))
        .rev()
        .find(|&k| {
            let total: f32 = (0..k)
                .map(|c| {
                    let idx: Vec<usize> = (c..pairs.len()).step_by(k).collect();
                    idx.iter().map(|&i| label_w[i]).fold(0.0, f32::max) + idx.iter().map(|&i| value_w[i]).fold(0.0, f32::max)
                })
                .sum();
            total + (k as f32 - 1.0) * 16.0 <= available
        })
        .unwrap_or(1);
    egui::Grid::new(ui.next_auto_id()).num_columns(columns * 2).spacing([8.0, 2.0]).show(ui, |ui| {
        for (i, (label, value)) in pairs.iter().enumerate() {
            ui.label(RichText::new(label).color(p.text_3));
            ui.label(RichText::new(value).font(mono.clone()).color(p.text));
            if (i + 1) % columns == 0 {
                ui.end_row();
            } else {
                ui.add_space(16.0);
            }
        }
    });
}

/// A combo box of texts; returns true when the choice changed.
pub fn combo(ui: &mut Ui, id_salt: impl std::hash::Hash + std::fmt::Debug, selected: &mut usize, items: &[&str], width: f32, tooltip: &str) -> bool {
    let before = *selected;
    let response = egui::ComboBox::from_id_salt(id_salt)
        .width(width)
        .selected_text(items.get(*selected).copied().unwrap_or(""))
        .show_ui(ui, |ui| {
            for (i, item) in items.iter().enumerate() {
                ui.selectable_value(selected, i, *item);
            }
        })
        .response;
    if !tooltip.is_empty() {
        response.on_hover_text(tooltip);
    }
    *selected != before
}

/// A plain button of control height.
pub fn button(ui: &mut Ui, text: &str, tooltip: &str) -> Response {
    let r = Button::new(text).min_size(Vec2::new(0.0, size::CONTROL_HEIGHT)).ui(ui);
    if tooltip.is_empty() { r } else { r.on_hover_text(tooltip) }
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
