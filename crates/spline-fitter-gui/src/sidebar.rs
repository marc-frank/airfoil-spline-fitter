//! The sidebar: data, method, curve and advanced settings, and the run bar.

use eframe::egui::{self, Align, Layout, RichText, Ui};
use spline_fitter::bspline_varpro::KnotPlacement;
use spline_fitter::formatting::{format_duration, format_error, number};
use spline_fitter::models::{OptimizerType, Parameterization};

use crate::app::App;
use crate::settings::{Change, IntSweep};
use crate::theme::{size, typeface, Palette};
use crate::widgets::{self, caption, form_row, muted, Field, IntOptions};

const OPTIMIZER_TIPS: [&str; 2] = [
    "Levenberg-Marquardt Variable Projection (bspline-varpro).\nTolerance mode: fewest control points within a maximum\northogonal deviation, non-uniform knots.\nGeometric (G) or parametric (C) continuity per junction.\nRecommended for CAD-quality results.",
    "Direct LSQ (single-step, no iteration).\nOne constrained least-squares solve, with fixed trailing-edge\nend points and junctions fixed or free (Advanced).\nFastest option (~1ms), moderate accuracy (~1e-4).\nIdeal for dense data or real-time preview.",
];

/// A section of the sidebar: caption, content, bottom rule.
fn section(ui: &mut Ui, p: &Palette, title: &str, content: impl FnOnce(&mut Ui)) {
    egui::Frame::new().inner_margin(egui::Margin { left: 12, right: 12, top: 10, bottom: 12 }).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = size::GAP;
        caption(ui, p, title);
        ui.add_space(2.0);
        content(ui);
    });
    widgets::rule(ui, p);
}

/// The field width beside a form label.
fn field_width(ui: &Ui) -> f32 {
    (ui.available_width() - size::LABEL_WIDTH - size::GAP).max(size::FIELD_MIN_WIDTH)
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
        egui::Panel::left("sidebar").resizable(true).default_size(size::SIDEBAR_WIDTH).min_size(size::SIDEBAR_MIN_WIDTH).frame(frame).show(
            ui,
            |ui| {
                egui::Panel::bottom("run_bar")
                    .frame(egui::Frame::new().fill(p.window).inner_margin(egui::Margin::same(12)))
                    .show(ui, |ui| self.run_bar(ui));
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    self.data_card(ui);
                    let fitting = self.fitting();
                    ui.add_enabled_ui(!fitting, |ui| {
                        self.method_section(ui);
                        self.curve_section(ui);
                    });
                    self.advanced_section(ui);
                });
            },
        );
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
                    let r = ui.add(egui::Label::new(RichText::new(name).strong().color(p.text)).truncate());
                    let m = muted(ui, &p, detail);
                    if !tip.is_empty() {
                        r.on_hover_text(&tip);
                        m.on_hover_text(&tip);
                    }
                });
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add_enabled(!fitting, egui::Button::new("Open...").min_size(egui::vec2(0.0, size::CONTROL_HEIGHT)))
                        .on_hover_text("Open an airfoil coordinate file (Ctrl+O)")
                        .clicked()
                    {
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
                    &mut s.optimizer,
                    &[(OptimizerType::VarPro, "VarPro LM", OPTIMIZER_TIPS[0]), (OptimizerType::DirectLsq, "Direct LSQ", OPTIMIZER_TIPS[1])],
                    width,
                );
            });
            let view = s.view(false);
            if view.varpro {
                form_row(ui, &p, "Goal", true, "", |ui| {
                    widgets::segmented(
                        ui,
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
                        let auto_w = 60.0;
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
                            ui.checkbox(&mut s.auto_tolerance, "Auto").on_hover_text(
                                "Derive the tolerance from the decimal precision of the file and\nthe estimated noise of the points.",
                            );
                        });
                    });
                });
                form_row(ui, &p, "Knots", true, "", |ui| {
                    widgets::segmented(
                        ui,
                        &mut s.knot_placement,
                        &[
                            (KnotPlacement::Uniform, "Uniform", "Knots stay evenly spaced in the curve parameter."),
                            (KnotPlacement::Free, "Free", "Knots move to where they reduce the error most."),
                        ],
                        width,
                    );
                });
            } else {
                form_row(ui, &p, "Parameterization", true, "", |ui| {
                    widgets::segmented(ui, &mut s.parameterization, &[
                        (Parameterization::Centripetal, "Centripetal", "Space the data parameters by the square root of the point distances.\nFollows tight curvature, such as the leading edge, more closely."),
                        (Parameterization::ArcLength, "Arc length", "Space the data parameters by the distance along the data."),
                    ], width);
                });
            }
            if view.varpro {
                form_row(ui, &p, "Grow points", true, "", |ui| {
                    ui.checkbox(&mut s.grow, "Add where the fit is worst").on_hover_text(
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
                    widgets::segmented(ui, &mut s.continuity, &items, width);
                });
            });
            if view.varpro {
                form_row(ui, &p, "Type", view.has_junctions, "", |ui| {
                    ui.add_enabled_ui(view.has_junctions, |ui| {
                        let tip = "Geometric: tangent direction, curvature and curvature rate match\nacross the junction (what CAD calls G1/G2/G3).\nParametric: the derivative vectors themselves match (C1/C2/C3),\nwhich is stricter and needs more control points.\nOnly VarPro LM supports geometric continuity; other optimizers\nalways use parametric.";
                        widgets::segmented(ui, &mut s.geometric, &[(true, "Geometric", tip), (false, "Parametric", tip)], width);
                    });
                });
            }
        });
    }

    fn advanced_section(&mut self, ui: &mut Ui) {
        let p = self.palette.clone();
        let (badge, badge_tip) = self.baseline.badge(&self.settings);
        egui::Frame::new().inner_margin(egui::Margin { left: 6, right: 12, top: 6, bottom: 6 }).show(ui, |ui| {
            widgets::collapsible_header(ui, &p, "Advanced", &mut self.advanced_open, &badge, &badge_tip);
        });
        if !self.advanced_open {
            widgets::rule(ui, &p);
            return;
        }
        let fitting = self.fitting();
        egui::Frame::new().inner_margin(egui::Margin { left: 12, right: 12, top: 0, bottom: 12 }).show(ui, |ui| {
            ui.add_enabled_ui(!fitting, |ui| {
                let s = &mut self.settings;
                let view = s.view(false);
                let width = field_width(ui);
                if view.varpro {
                    let response = form_row(ui, &p, "Per junction", view.has_junctions, "", |ui| {
                        ui.add_enabled_ui(view.has_junctions, |ui| {
                            ui.add(egui::TextEdit::singleline(&mut s.per_junction).hint_text("G2, C1  (blank = same everywhere)").desired_width(width))
                                .on_hover_text("Continuity of each junction in order, overriding the Continuity\nsetting. One entry per junction (sections - 1), each G0-G3 or C0-C3.\nVarPro LM only.")
                        })
                        .inner
                    });
                    if response.lost_focus() {
                        s.keep_curve_valid(Change::Other);
                    }
                    form_row(ui, &p, "Subsample", true, "", |ui| {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut s.subsample, "to").on_hover_text("For dense data: fit an evenly spaced subsample of each section, then\nrefine the fit on all points. Much faster for thousands of points.\nOff: every point is used throughout.");
                            ui.add_enabled_ui(s.subsample, |ui| {
                                let f = &mut s.subsample_points;
                                widgets::int_field(ui, &p, "subsample", &mut f.value, (f.min, f.max), Field::new((width - 50.0).max(80.0), "Points per section. Sections with more points than this are subsampled."), IntOptions { suffix: " points", ..IntOptions::default() });
                            });
                        });
                    });
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
                form_row(ui, &p, "Junctions", view.has_junctions, "", |ui| {
                    ui.add_enabled_ui(view.has_junctions, |ui| {
                        ui.checkbox(&mut s.optimize_junctions, "Optimize positions").on_hover_text("Optimize the position of every junction, instead of keeping the\nsection split points fixed. The split points are where the junctions\nstart; the data is fitted by the whole curve, so points move between\nsections with them.");
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
            if widgets::button(ui, "Cancel", "").clicked() {
                if let Some(r) = &self.running {
                    r.cancel();
                }
                self.set_status("Cancelling... please wait");
            }
        } else if widgets::primary_button(ui, "Fit spline", width).clicked() {
            self.on_fit();
        }
        let Some(fit) = self.fit.clone().filter(|_| !self.fitting()) else { return };
        ui.add_space(size::PAD);
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
        egui::Grid::new("readouts").num_columns(3).spacing([16.0, 2.0]).show(ui, |ui| {
            for c in [error_caption, "Control points", "Time"] {
                muted(ui, &p, c);
            }
            ui.end_row();
            let readout = |ui: &mut Ui, text: &str| ui.label(RichText::new(text).monospace().size(typeface::READOUT).color(p.text));
            readout(ui, &error_value);
            let r = readout(ui, &count_text);
            if !count_tip.is_empty() {
                r.on_hover_text(count_tip);
            }
            readout(ui, &format_duration(fit.elapsed_seconds));
            ui.end_row();
        });
        ui.add_space(size::GAP);
        ui.horizontal(|ui| {
            if widgets::button(ui, "Save .dat", "Save fitted airfoil coordinates (.dat)").clicked() {
                self.on_save_airfoil();
            }
            if widgets::button(ui, "Save spline", "Save spline control points and knots (.spline.dat)").clicked() {
                self.on_save_spline();
            }
            if widgets::button(ui, "Export CAD", "Export the exact B-spline curves as STEP, IGES or DXF").clicked() {
                self.on_export_cad();
            }
        });
    }
}
