//! The sidebar: data, method, curve and advanced settings, and the run bar.

use eframe::egui::{self, Align, Layout, RichText, Ui};
use spline_fitter::bspline_varpro::KnotPlacement;
use spline_fitter::formatting::{format_duration, format_error, number};
use spline_fitter::models::{OptimizerType, Parameterization};

use crate::app::App;
use crate::settings::{Change, IntSweep};
use crate::theme::{size, typeface, Palette};
use crate::widgets::{self, caption, form_row, muted, Field, IntOptions};

/// A section's margins; the bottom one holds the rule below it (Qt draws
/// a border inside the margin).
const SECTION_MARGIN: egui::Margin = egui::Margin { left: 12, right: 12, top: 10, bottom: 11 };

/// A section of the sidebar: caption, content, bottom rule.
fn section(ui: &mut Ui, p: &Palette, title: &str, content: impl FnOnce(&mut Ui)) {
    egui::Frame::new().inner_margin(SECTION_MARGIN).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = size::GAP;
        let caption = caption(ui, p, title);
        // The caption takes a line of its own, a group's space above the content.
        ui.add_space(size::CAPTION_LINE - caption.rect.height() + size::GROUP - size::GAP);
        content(ui);
    });
    widgets::rule(ui, p);
}

/// The field width beside a form label.
fn field_width(ui: &Ui) -> f32 {
    (ui.available_width() - size::LABEL_WIDTH - size::GROUP).max(size::FIELD_MIN_WIDTH)
}

/// An integer spin box with its sweep toggle and end box.
fn sweep_field(ui: &mut Ui, p: &Palette, id: &str, sweep: &mut IntSweep, width: f32, tooltip: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = size::TIGHT;
        let toggle_w = size::CONTROL_HEIGHT + size::TIGHT;
        let boxes = if sweep.active { 2.0 } else { 1.0 };
        let each = ((width - toggle_w - (boxes - 1.0) * size::TIGHT) / boxes).max(48.0);
        let v = &mut sweep.value;
        widgets::int_field(ui, p, (id, "value"), &mut v.value, (v.min, v.max), Field::new(each, tooltip), IntOptions::default());
        let mut active = sweep.active;
        if widgets::sweep_toggle(ui, p, &mut active, sweep.allowed) {
            sweep.toggle(active);
        }
        if sweep.active {
            let e = &mut sweep.end;
            widgets::int_field(ui, p, (id, "end"), &mut e.value, (e.min, e.max), Field::new(each, ""), IntOptions::default());
        }
    });
}

impl App {
    pub fn sidebar(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let frame = egui::Frame::new().fill(p.panel);
        let mut panel = egui::Panel::left("sidebar")
            .resizable(true)
            .default_size(size::SIDEBAR_WIDTH + size::SPLITTER)
            .min_size(size::SIDEBAR_MIN_WIDTH)
            .frame(frame);
        if std::mem::take(&mut self.reset_sidebar) {
            panel = panel.exact_size(size::SIDEBAR_WIDTH + size::SPLITTER);
        }
        let shown = panel.show(ui, |ui| {
            let bar = egui::Panel::bottom("run_bar")
                .frame(egui::Frame::new().fill(p.window).inner_margin(egui::Margin::same(12)))
                .show(ui, |ui| self.run_bar(ui));
            let top = bar.response.rect;
            ui.painter().hline(top.x_range(), top.top(), egui::Stroke::new(1.0, p.rule));
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                // Sections and their rules follow without gaps.
                ui.spacing_mut().item_spacing.y = 0.0;
                self.data_card(ui);
                let fitting = self.fitting();
                ui.add_enabled_ui(!fitting, |ui| {
                    self.method_section(ui);
                    self.curve_section(ui);
                });
                self.advanced_section(ui);
            });
        });
        let rect = shown.response.rect;
        self.sidebar_rect = rect;
        // Double-clicking the edge puts the default width back.
        if widgets::edge_double_clicked(ui.ctx(), rect, rect.right()) {
            self.reset_sidebar = true;
            ui.ctx().request_repaint();
        }
    }

    fn data_card(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let fitting = self.fitting();
        section(ui, &p, "Data", |ui| {
            ui.horizontal(|ui| {
                let (name, detail) = match &self.airfoil {
                    Some(a) => (a.display_name().to_string(), format!("{} points, {} decimals", a.coordinates.len(), a.precision)),
                    None => ("No airfoil loaded".to_string(), "Open a .dat file or a project".to_string()),
                };
                let tip = self.airfoil_path.clone();
                ui.vertical(|ui| {
                    ui.set_width(ui.available_width() - 80.0);
                    ui.spacing_mut().item_spacing.y = 0.0;
                    let r = ui.add(
                        egui::Label::new(RichText::new(name).font(crate::fonts::semibold_id(typeface::BASE)).color(p.text)).truncate(),
                    );
                    let m = muted(ui, &p, detail);
                    if !tip.is_empty() {
                        r.on_hover_text(&tip);
                        m.on_hover_text(&tip);
                    }
                });
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if widgets::push_button(ui, &p, !fitting, "Open...").on_hover_text("Open an airfoil coordinate file (Ctrl+O)").clicked() {
                        self.on_open_airfoil();
                    }
                });
            });
        });
    }

    fn method_section(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        section(ui, &p, "Method", |ui| {
            let s = &mut self.settings;
            let width = field_width(ui);
            form_row(ui, &p, "Optimizer", true, "", |ui| {
                widgets::segmented(
                    ui,
                    &p,
                    &mut s.optimizer,
                    &[
                        (OptimizerType::VarPro, "VarPro LM", OptimizerType::VarPro.tooltip()),
                        (OptimizerType::DirectLsq, "Direct LSQ", OptimizerType::DirectLsq.tooltip()),
                    ],
                    width,
                );
            });
            let view = s.view(false);
            if view.varpro {
                form_row(ui, &p, "Goal", true, "", |ui| {
                    widgets::segmented(
                        ui,
                        &p,
                        &mut s.tolerance_goal,
                        &[
                            (false, "Fixed count", "Fit the given number of control points per section."),
                            (true, "Tolerance", "Find the fewest control points that keep\nevery data point within the tolerance."),
                        ],
                        width,
                    );
                });
                form_row(ui, &p, "Max deviation", view.tolerance_mode, "", |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = size::GAP;
                        let auto_w = widgets::checkbox_width(ui, "Auto");
                        ui.add_enabled_ui(view.tolerance_mode && !s.auto_tolerance, |ui| {
                            widgets::scientific_field(
                                ui,
                                &p,
                                "tolerance",
                                &mut s.tolerance,
                                (1e-9, f64::INFINITY),
                                1e-6,
                                Field::new(width - auto_w - size::GAP, "Maximum orthogonal distance of any data point to the curve."),
                            );
                        });
                        ui.add_enabled_ui(view.tolerance_mode, |ui| {
                            widgets::checkbox(ui, &p, &mut s.auto_tolerance, "Auto").on_hover_text(
                                "Derive the tolerance from the decimal precision of the file and\nthe estimated noise of the points.",
                            );
                        });
                    });
                });
                form_row(ui, &p, "Knots", true, "", |ui| {
                    widgets::segmented(
                        ui,
                        &p,
                        &mut s.knot_placement,
                        &[
                            (KnotPlacement::Uniform, "Uniform", "Knots stay evenly spaced in the curve parameter."),
                            (KnotPlacement::Free, "Free", "Knots move to where they reduce the error most."),
                        ],
                        width,
                    );
                });
                let view = s.view(false);
                form_row(ui, &p, "Knot spacing", view.free_knots, "", |ui| {
                    ui.add_enabled_ui(view.knot_gap_enabled, |ui| knot_gap_field(ui, &p, &mut s.knot_gap, width));
                });
            } else {
                form_row(ui, &p, "Parameterization", true, "", |ui| {
                    widgets::segmented(ui, &p, &mut s.parameterization, &[
                        (Parameterization::Centripetal, "Centripetal", "Space the data parameters by the square root of the point distances.\nFollows tight curvature, such as the leading edge, more closely."),
                        (Parameterization::ArcLength, "Arc length", "Space the data parameters by the distance along the data."),
                    ], width);
                });
            }
            if view.varpro {
                widgets::check_row(ui, &p, "Grow points", true, |ui| {
                    widgets::checkbox(ui, &p, &mut s.grow, "Add where the fit is worst").on_hover_text(
                        "On: start from the fewest control points and add them where the\nfit is worst, refitting after each step.\nOff: start from the Control points setting (Curve). With a tolerance\ngoal the count can still grow or shrink from there to meet it.",
                    );
                });
            }
        });
    }

    fn curve_section(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        section(ui, &p, "Curve", |ui| {
            let s = &mut self.settings;
            let view = s.view(false);
            let width = field_width(ui);
            form_row(ui, &p, "Degree", true, "", |ui| {
                sweep_field(
                    ui,
                    &p,
                    "degree",
                    &mut s.degree,
                    width,
                    "Polynomial degree of B-spline basis functions.\nTypical: 3 (cubic), 4, or 5 (quintic).",
                );
            });
            let count_label = if view.tolerance_mode { "Start count" } else { "Control points" };
            form_row(ui, &p, count_label, view.count_used, "", |ui| {
                ui.add_enabled_ui(view.count_used, |ui| {
                    sweep_field(
                        ui,
                        &p,
                        "count",
                        &mut s.control_points,
                        width,
                        "Control points per section.\nMore = more flexibility but more variables.",
                    );
                });
            });
            form_row(ui, &p, "Sections", true, "", |ui| {
                sweep_field(ui, &p, "sections", &mut s.sections, width, "Number of sections, each its own spline, joined at junctions.\n2 (default) splits upper and lower surface at the leading edge.");
            });
            let letter = s.continuity_letter();
            let labels: Vec<String> = (0..4).map(|i| format!("{letter}{i}")).collect();
            let tips = [
                "Position matches across junctions",
                "Tangent matches across junctions",
                "Curvature matches across junctions",
                "Curvature rate matches across junctions",
            ];
            form_row(ui, &p, "Continuity", view.has_junctions, "", |ui| {
                ui.add_enabled_ui(view.has_junctions, |ui| {
                    let items: Vec<(usize, &str, &str)> = (0..4).map(|i| (i, labels[i].as_str(), tips[i])).collect();
                    widgets::segmented(ui, &p, &mut s.continuity, &items, width);
                });
            });
            if view.varpro {
                form_row(ui, &p, "Type", view.has_junctions, "", |ui| {
                    ui.add_enabled_ui(view.has_junctions, |ui| {
                        let tip = "Geometric: tangent direction, curvature and curvature rate match\nacross the junction (what CAD calls G1/G2/G3).\nParametric: the derivative vectors themselves match (C1/C2/C3),\nwhich is stricter and needs more control points.\nOnly VarPro LM supports geometric continuity; other optimizers\nalways use parametric.";
                        widgets::segmented(ui, &p, &mut s.geometric, &[(true, "Geometric", tip), (false, "Parametric", tip)], width);
                    });
                });
            }
        });
    }

    fn advanced_section(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let (badge, badge_tip) = self.baseline.badge(&self.settings);
        let fitting = self.fitting();
        egui::Frame::new().inner_margin(SECTION_MARGIN).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = size::GAP;
            widgets::collapsible_header(ui, &p, "Advanced", &mut self.advanced_open, &badge, &badge_tip);
            if !self.advanced_open {
                return;
            }
            ui.add_space(size::GROUP - size::GAP);
            ui.add_enabled_ui(!fitting, |ui| {
                let s = &mut self.settings;
                let view = s.view(false);
                let width = field_width(ui);
                if view.varpro {
                    let response = form_row(ui, &p, "Per junction", view.has_junctions, "", |ui| {
                        ui.add_enabled_ui(view.has_junctions, |ui| {
                            widgets::text_field(ui, &p, "per_junction", &mut s.per_junction, width, "G2, C1  (blank = same everywhere)", 0)
                                .on_hover_text("Continuity of each junction in order, overriding the Continuity\nsetting. One entry per junction (sections - 1), each G0-G3 or C0-C3.\nVarPro LM only.")
                        })
                        .inner
                    });
                    if response.lost_focus() {
                        s.keep_curve_valid(Change::Other);
                    }
                    form_row(ui, &p, "Smoothing", view.smoothing_enabled, "", |ui| {
                        ui.add_enabled_ui(view.smoothing_enabled, |ui| {
                            ui.vertical(|ui| {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = size::TIGHT;
                                    let sm = &mut s.smoothing;
                                    let boxes = if sm.active { 2.0 } else { 1.0 };
                                    let each = ((width - size::CONTROL_HEIGHT - size::TIGHT * boxes) / boxes).max(56.0);
                                    widgets::scientific_field(ui, &p, "smoothing", &mut sm.value, (0.0, f64::INFINITY), 1e-3, Field::new(each, "Bending energy weight for VarPro LM (0 turns it off; default 0.001)."));
                                    let mut active = sm.active;
                                    if widgets::sweep_toggle(ui, &p, &mut active, sm.allowed) {
                                        sm.toggle(active);
                                    }
                                    if sm.active {
                                        widgets::scientific_field(ui, &p, "smoothing_end", &mut sm.end, (0.0, f64::INFINITY), 1e-3, Field::new(each, ""));
                                    }
                                });
                                if s.smoothing.active {
                                    ui.horizontal(|ui| {
                                        ui.label(RichText::new("Step").color(p.text_2));
                                        let sm = &mut s.smoothing;
                                        let min = sm.single_step / 1000.0;
                                        widgets::scientific_field(ui, &p, "smoothing_step", &mut sm.step, (min, f64::INFINITY), 1e-3, Field::new((width - 40.0).max(60.0), "Distance between the swept values"));
                                    });
                                }
                            });
                        });
                    });
                }
                widgets::check_row(ui, &p, "Junctions", view.has_junctions, |ui| {
                    ui.add_enabled_ui(view.has_junctions, |ui| {
                        widgets::checkbox(ui, &p, &mut s.optimize_junctions, "Optimize positions").on_hover_text("Optimize the position of every junction, instead of keeping the\nsection split points fixed. The split points are where the junctions\nstart; the data is fitted by the whole curve, so points move between\nsections with them.");
                    });
                });
            });
        });
        widgets::rule(ui, &p);
    }

    fn run_bar(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let width = ui.available_width();
        if self.fitting() {
            // Cancel takes the Fit button's place.
            if ui.add_sized([width, size::CONTROL_HEIGHT], egui::Button::new("Cancel")).clicked() {
                if let Some(r) = &self.running {
                    r.cancel();
                }
                self.set_status("Cancelling... please wait");
            }
        } else if widgets::primary_button(ui, "Fit spline", width).clicked() {
            self.on_fit();
        }
        let Some(fit) = self.fit.clone().filter(|_| !self.fitting()) else { return };
        ui.add_space(size::PAD - ui.spacing().item_spacing.y);
        let (error_caption, error_value) = match &fit.report {
            Some(r) => ("Max deviation", number(r.max_deviation, ".2e")),
            None => ("Error", format_error(fit.final_error)),
        };
        let counts = fit.section_counts();
        let count_text = if counts.len() == 1 {
            counts[0].to_string()
        } else if counts.iter().all(|&c| c == counts[0]) {
            format!("{} × {}", counts.len(), counts[0])
        } else {
            counts.iter().sum::<usize>().to_string()
        };
        let count_tip = if counts.len() > 1 {
            format!("{} per section", counts.iter().map(usize::to_string).collect::<Vec<_>>().join(", "))
        } else {
            String::new()
        };
        // Three readouts in equal columns, a caption above each value.
        let column = (width - 2.0 * size::GROUP) / 3.0;
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = size::GROUP;
            let readouts = [
                (error_caption, error_value, String::new()),
                ("Control points", count_text, count_tip),
                ("Time", format_duration(fit.elapsed_seconds), String::new()),
            ];
            for (caption, value, tip) in readouts {
                ui.allocate_ui_with_layout(egui::vec2(column, 0.0), Layout::top_down(Align::Min), |ui| {
                    ui.set_width(column);
                    ui.spacing_mut().item_spacing.y = 0.0;
                    muted(ui, &p, caption);
                    let r = ui.label(RichText::new(value).monospace().size(typeface::READOUT).color(p.text));
                    ui.add_space((size::READOUT_LINE - r.rect.height()).max(0.0));
                    if !tip.is_empty() {
                        r.on_hover_text(tip);
                    }
                });
            }
        });
        ui.add_space(size::PAD - ui.spacing().item_spacing.y);
        // The exports share the row in equal widths.
        let each = (width - 2.0 * size::GAP) / 3.0;
        ui.horizontal(|ui| {
            let exports: [Export; 3] = [
                ("Save .dat", "Save fitted airfoil coordinates (.dat)", App::on_save_airfoil),
                ("Save spline", "Save spline control points and knots (.spline.dat)", App::on_save_spline),
                ("Export CAD", "Export the exact B-spline curves as STEP, IGES or DXF", App::on_export_cad),
            ];
            for (text, tip, action) in exports {
                if ui.add_sized([each, size::CONTROL_HEIGHT], egui::Button::new(text)).on_hover_text(tip).clicked() {
                    action(self);
                }
            }
        });
    }
}

/// A run bar export: its text, tooltip and action.
type Export = (&'static str, &'static str, fn(&mut App));

/// The Knot spacing field: three decimals, 0 to 0.5 in steps of 0.01.
fn knot_gap_field(ui: &mut Ui, p: &Palette, value: &mut f64, width: f32) {
    let format = |v: f64| spline_fitter::formatting::number(v, ".3f");
    let parse = |t: &str| widgets::parse_scientific(t).map(|v| (v * 1000.0).round() / 1000.0);
    let step = |v: f64, k: i32| ((v + 0.01 * k as f64) * 1000.0).round() / 1000.0;
    let fmt = widgets::NumberFormat { format: &format, parse: &parse, step: &step };
    let tooltip = "Smallest spacing of free knots, as a fraction of a section.\nCloser knots follow the data more tightly but can bend the\ncurve sharply between the points, and fits take longer.";
    widgets::number_field(ui, p, "knot_gap", value, (0.0, 0.5), Field::new(width, tooltip), &fmt);
}
