//! The window driven headlessly: the reference session of the
//! specification (section 21, item 17), with screenshots on request.

use std::time::Duration;

use eframe::egui;
use egui_kittest::Harness;
use spline_fitter::models::{OptimizerType, ToleranceMetric};

use crate::app::App;

fn harness() -> Harness<'static, App> {
    Harness::builder().with_size([1500.0, 950.0]).wgpu().build_eframe(|cc| {
        let app = App::new(cc);
        spline_fitter::formatting::set_decimal_point('.');
        app
    })
}

/// Runs frames until the running fit (or sweep) ends.
fn finish(h: &mut Harness<'static, App>) {
    for _ in 0..3000 {
        h.step();
        if !h.state().fitting() {
            h.run_steps(3);
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("the fit did not end");
}

/// Saves a screenshot when `SPLINE_FITTER_SHOTS` names a folder.
fn shot(h: &mut Harness<'static, App>, name: &str) {
    if let Some(dir) = std::env::var_os("SPLINE_FITTER_SHOTS") {
        let image = h.render().expect("the window renders");
        std::fs::create_dir_all(&dir).unwrap();
        image.save(std::path::Path::new(&dir).join(format!("{name}.png"))).unwrap();
    }
}

#[test]
fn reference_session() {
    let mut h = harness();
    h.run_steps(3);
    {
        let app = h.state();
        assert_eq!(app.status, "Ready");
        let s = &app.settings;
        assert_eq!(s.optimizer, OptimizerType::VarPro);
        assert!(!s.tolerance_goal && s.auto_tolerance && !s.grow && s.geometric);
        assert_eq!((s.degree.value.value, s.control_points.value.value, s.sections.value.value, s.continuity), (5, 25, 2, 3));
    }
    shot(&mut h, "1_start");

    h.state_mut().new_naca_airfoil("2412", 81, true);
    h.run_steps(3);
    assert_eq!(h.state().status, "Created: NACA 2412 (161 points, 2 sections: [81, 81])");
    assert_eq!(h.state().settings.tolerance, 1e-6);
    shot(&mut h, "2_naca");

    h.state_mut().on_fit();
    // What is about to be fitted shows first; the fit starts a frame later.
    assert_eq!(h.state().status, "Preparing 25 control points, degree 5, G3 continuity...");
    assert!(h.state().fitting());
    h.step();
    assert_eq!(h.state().status, "Starting VarPro LM...");
    // While it fits, the settings are disabled (as Python draws them).
    h.step();
    shot(&mut h, "2a_fitting");
    finish(&mut h);
    {
        let app = h.state();
        assert!(app.status.starts_with("Completed (VarPro LM): 33 evaluations in "), "{}", app.status);
        assert!(app.status.ends_with(" | RMS error 8.978e-07 | max deviation 4.59e-06, RMS 8.98e-07, 49 control points"), "{}", app.status);
        assert_eq!(app.projects.label(), "Unsaved project*\n1 optimization run");
        assert_eq!(app.fit.as_ref().unwrap().section_counts(), vec![25, 25]);
    }
    h.state_mut().main_plot.comb_on = true;
    h.state_mut().main_plot.refresh_comb();
    h.run_steps(3);
    shot(&mut h, "3_varpro_fit");

    {
        let app = h.state_mut();
        app.settings.optimizer = OptimizerType::DirectLsq;
        app.settings.continuity = 2;
    }
    h.run_steps(2);
    h.state_mut().on_fit();
    finish(&mut h);
    {
        let app = h.state();
        assert!(app.status.starts_with("Completed (Direct LSQ): 1 evaluation in "), "{}", app.status);
        assert!(app.status.ends_with(" | RMS error 1.897e-05 | max deviation 4.72e-05, RMS 8.81e-06, 49 control points"), "{}", app.status);
        assert_eq!(app.projects.runs().len(), 2);
    }
    shot(&mut h, "4_direct_lsq");

    // One undo step returns the optimizer and the continuity.
    let current = h.state().settings.values();
    h.state_mut().history.commit(&current);
    h.state_mut().undo();
    h.run_steps(2);
    assert_eq!(h.state().settings.optimizer, OptimizerType::VarPro);
    assert_eq!(h.state().settings.continuity, 3);

    // A sweep of the control points 25..26.
    {
        let app = h.state_mut();
        app.settings.control_points.toggle(true);
        app.settings.control_points.end.set(26);
    }
    h.run_steps(2);
    h.state_mut().on_fit();
    finish(&mut h);
    {
        let app = h.state();
        assert!(app.status.starts_with("Sweep complete: 2 fits in "), "{}", app.status);
        assert!(app.status.ends_with(" | Best: Control points 25, RMS deviation 8.978e-07"), "{}", app.status);
        assert!(!app.settings.control_points.active);
        assert_eq!(app.settings.control_points.value.value, 25);
    }
    shot(&mut h, "5_sweep");

    // A run selected in the explorer, then Advanced open.
    let first = h.state().projects.runs()[0].run_id.clone();
    h.state_mut().select_run(&first);
    h.run_steps(3);
    shot(&mut h, "8_selected");
    h.state_mut().advanced_open = true;
    h.run_steps(3);
    shot(&mut h, "9_advanced");
    h.state_mut().explorer.pivot = true;
    h.run_steps(3);
    shot(&mut h, "10_pivot");
}

/// The empty window says how to start: the main plot names the ways to
/// load an airfoil, the run table what it will hold.
#[test]
fn the_empty_window_says_how_to_start() {
    let mut h = harness();
    h.run_steps(3);
    shot(&mut h, "1e_empty_hints");
    assert!(crate::plots::EMPTY_PLOT_HINT.contains("Ctrl+O") && crate::plots::EMPTY_PLOT_HINT.contains("Ctrl+N"));
    assert!(h.state().projects.runs().is_empty());
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.run_steps(2);
    h.state_mut().on_fit();
    finish(&mut h);
    assert_eq!(h.state().projects.runs().len(), 1, "the table's hint gives way to the run");
}

#[test]
fn light_theme_and_dialogs() {
    let mut h = harness();
    h.state_mut().set_theme(Some(crate::theme::ThemeMode::Light), Some(30));
    h.state_mut().new_naca_airfoil("0012", 61, false);
    h.run_steps(3);
    {
        let app = h.state_mut();
        app.settings.tolerance_goal = true;
        app.settings.auto_tolerance = false;
        app.settings.tolerance = 2e-6;
        // From the fewest control points, one more a fit: a run each.
        app.settings.grow = true;
        app.settings.sections.value.set(3);
        app.advanced_open = true;
    }
    h.run_steps(3);
    shot(&mut h, "6a_light_settings");
    // The section count runs the rules only; the next fit splits.
    assert_eq!(h.state().status, "Created: NACA 0012 (121 points, 2 sections: [61, 61])");
    assert_eq!(h.state().airfoil.as_ref().unwrap().num_sections(), 2);
    h.state_mut().settings.sections.value.set(2);
    h.run_steps(3);
    h.state_mut().on_fit();
    // While the search runs, the runs plot follows its refits: control
    // points against the largest deviation, the latest marked, with the
    // tolerance line.
    let mut live = 0;
    for _ in 0..3000 {
        h.step();
        if !h.state().fitting() {
            break;
        }
        let c = &h.state().comparison;
        if c.y_label == "Max deviation" {
            assert_eq!((c.x_label.as_str(), c.tolerance, c.marked.len()), ("Control points", Some(2e-6), 1));
            assert!(c.labels.is_empty(), "nothing to click while it runs");
            if c.points.len() > live {
                live = c.points.len();
                if live == 2 {
                    shot(&mut h, "6l_live_refits");
                }
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    h.run_steps(3);
    assert!(live >= 2, "the refits were drawn as they finished ({live})");
    {
        let app = h.state();
        assert!(app.status.starts_with("Completed (VarPro LM): "), "{}", app.status);
        // At its end the plot shows the project's runs again.
        assert_eq!(app.comparison.y_label, "RMS deviation");
        // The search is a folder; each of its fits a fixed-count run of its
        // own count, the same in both sections, told apart by it.
        let file = app.projects.file.as_ref().unwrap();
        assert_eq!(file.searches.len(), 1);
        let search = &file.searches[0];
        assert_eq!((search.number, search.optimization_config.tolerance), (1, Some(2e-6)));
        let runs = app.projects.runs();
        assert!(runs.iter().all(|r| r.search.as_deref() == Some(search.search_id.as_str())));
        assert!(runs.iter().all(|r| r.optimization_config.tolerance.is_none() && !r.optimization_config.grow_control_points));
        let counts: Vec<usize> = runs.iter().map(|r| r.spline_config.num_control_points).collect();
        assert!(runs.iter().all(|r| r.result.section_counts() == vec![r.spline_config.num_control_points; 2]), "{counts:?}");
        assert!(counts.windows(2).all(|w| w[1] == w[0] + 1), "one more a fit: {counts:?}");
        let result = app.projects.search_result(search).unwrap();
        assert_eq!(result.result.section_counts(), app.fit.as_ref().unwrap().section_counts());
        assert_eq!(search.control_points, result.spline_config.num_control_points);
        // Each run keeps only its own evaluations.
        assert!(runs.iter().all(|r| !r.loss_history.is_empty()));
        let own: usize = runs.iter().map(|r| r.loss_history.len()).sum();
        assert_eq!(own, search.evaluations);
        // In the table the search is one row, closed.
        let rows = app.visible_rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1[0].1, "Search 1");
    }
    // Open, its runs follow it.
    let search_id = h.state().projects.searches()[0].search_id.clone();
    h.state_mut().toggle_search(&search_id, true);
    h.run_steps(2);
    assert_eq!(h.state().visible_rows().len(), 1 + h.state().projects.runs().len());
    h.state_mut().main_plot.deviation_on = true;
    h.state_mut().main_plot.show_knots = true;
    h.state_mut().main_plot.refresh_knots();
    h.state_mut().main_plot.refresh_deviation();
    h.run_steps(3);
    shot(&mut h, "6_light_tolerance");
    // Along control points every fit of the search is a point, at its count
    // per section.
    {
        let app = h.state();
        let mut xs: Vec<f64> = app.comparison.points.iter().map(|p| p[0]).collect();
        let mut fitted: Vec<f64> = app.projects.runs().iter().map(|r| r.spline_config.num_control_points as f64).collect();
        xs.sort_by(f64::total_cmp);
        fitted.sort_by(f64::total_cmp);
        assert_eq!(xs, fitted);
        assert!(xs.first() != xs.last(), "{xs:?}");
        // Plotted by their RMS deviation: no tolerance line.
        assert_eq!(app.comparison.tolerance, None);
        // Each point names its run.
        assert_eq!(app.comparison.labels.len(), app.comparison.points.len());
        assert!(app.comparison.labels.iter().all(|l| l.starts_with("Run ") && l.contains(" control points, RMS deviation ")), "{:?}", app.comparison.labels);
    }
    // The search's tolerance is a line in the plot of their largest deviations.
    {
        let app = h.state_mut();
        app.explorer.y_column = crate::runs::COLUMNS.iter().position(|c| c.name == "Max deviation").unwrap();
        app.explorer.forget_points();
        app.refresh_comparison();
        assert_eq!(app.comparison.tolerance, Some(2e-6));
        app.explorer.y_column = crate::runs::RMS_DEVIATION_COLUMN;
        app.explorer.forget_points();
        app.refresh_comparison();
    }
    // A click on a point selects its run and shows that fit.
    {
        let app = h.state_mut();
        let last = app.comparison.points.len() - 1;
        let point = app.comparison.points[last];
        app.click_plotted_run(last);
        let id = app.projects.selected.iter().next().unwrap().clone();
        assert_eq!(app.projects.selected.len(), 1);
        assert_eq!(app.projects.shown_run.as_deref(), Some(id.as_str()));
        assert_eq!(app.comparison.marked, vec![point]);
        let run = app.projects.run(&id).unwrap();
        assert_eq!(run.spline_config.num_control_points as f64, point[0]);
        assert!(app.comparison.labels[last].starts_with(&format!("Run {}: ", run.number)));
    }
    // Double-clicking a run of the search sets the sidebar to that fixed-count
    // run; the search's own row, to the tolerance search it was.
    {
        let app = h.state_mut();
        let run = app.projects.runs()[0].clone();
        app.load_settings(&run.run_id);
        let s = &app.settings;
        assert!(!s.tolerance_mode() && !s.grow && s.control_points.value.value == run.spline_config.num_control_points);
        app.load_settings(&crate::explorer::search_row(&search_id));
        let s = &app.settings;
        assert!(s.tolerance_mode() && s.grow && !s.auto_tolerance && s.tolerance == 2e-6);
        assert!(app.status.starts_with("Applied the settings of search 1 (tolerance 2.00e-06, from the fewest control points"), "{}", app.status);
    }
    h.run_steps(3);
    // The report shows in the bottom block, beside the plots.
    h.state_mut().on_report();
    h.run_steps(4);
    shot(&mut h, "6r_report");
    assert_eq!(h.state().bottom_view, 1);
    assert!(h.state().report_panel.as_ref().is_some_and(|r| r.title == "NACA 0012"));
    assert!(!h.state().modal_open(), "no window of its own");
    h.state_mut().set_bottom_view(2);
    h.run_steps(4);
    shot(&mut h, "6s_shape");
    assert!(h.state().report_panel.as_ref().is_some_and(|r| r.shape.is_some()));
    h.state_mut().set_bottom_view(0);
    h.run_steps(2);
    h.state_mut().on_export_cad();
    h.run_steps(3);
    shot(&mut h, "7_cad_dialog");
    h.state_mut().dialog = None;

    // A fit splits the airfoil for its section count.
    {
        let app = h.state_mut();
        app.settings.optimizer = OptimizerType::DirectLsq;
        app.settings.sections.value.set(3);
    }
    h.run_steps(3);
    h.state_mut().on_fit();
    finish(&mut h);
    assert!(h.state().status.starts_with("Completed (Direct LSQ): 1 evaluation in "), "{}", h.state().status);
    assert_eq!(h.state().airfoil.as_ref().unwrap().num_sections(), 3);
}

/// Every dialog, for screenshots (the dialogs are windows of their own;
/// under test they show inside the main window).
#[test]
fn dialogs() {
    use crate::dialogs::Dialog;
    let mut h = harness();
    h.state_mut().dialog = Some(Dialog::naca(&h.state().store));
    h.run_steps(3);
    shot(&mut h, "d_naca");
    h.state_mut().dialog = None;
    h.state_mut().new_naca_airfoil("2412", 81, true);
    h.state_mut().on_fit();
    finish(&mut h);
    shot(&mut h, "d_base");
    for (name, dialog) in [
        ("d_points", Dialog::points_per_side(81)),
        ("d_cad", Dialog::cad(&h.state().store)),
        ("d_theme", Dialog::theme(h.state())),
        ("d_motion", Dialog::motion(h.state())),
        ("d_batch", Dialog::batch()),
    ] {
        h.state_mut().dialog = Some(dialog);
        h.run_steps(3);
        shot(&mut h, name);
        h.state_mut().dialog = None;
    }
    h.state_mut().on_report();
    h.run_steps(4);
    shot(&mut h, "d_report");
    assert_eq!(h.state().bottom_view, 1);
}

/// Clicks at a point of the window.
fn click_at(h: &mut Harness<'static, App>, x: f32, y: f32) {
    let pos = egui::pos2(x, y);
    h.input_mut().events.push(egui::Event::PointerMoved(pos));
    h.run_steps(1);
    for pressed in [true, false] {
        h.input_mut().events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        });
        h.run_steps(1);
    }
    h.run_steps(2);
}

/// The File menu, a column's filter drop-down and the comb settings.
#[test]
fn popups() {
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 81, true);
    h.state_mut().on_fit();
    finish(&mut h);
    click_at(&mut h, 18.0, 12.0);
    shot(&mut h, "p_file_menu");
    h.input_mut().events.push(egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Default::default(),
    });
    h.run_steps(3);
    h.state_mut().open_column_popup(5, egui::Rect::from_min_max(egui::pos2(1290.0, 140.0), egui::pos2(1406.0, 165.0)));
    h.run_steps(3);
    shot(&mut h, "p_column_popup");
    h.state_mut().explorer.close_popup();
    h.run_steps(3);
    click_at(&mut h, 615.0, 45.0);
    shot(&mut h, "p_comb_panel");
}

/// Presses a key with modifiers.
/// Alt and a menu's underlined letter opens the menu; an entry's letter
/// chooses it, a submenu's opens the submenu (Qt's access keys).
#[test]
fn menus_answer_their_access_keys() {
    use crate::dialogs::Dialog;
    use egui_kittest::kittest::Queryable;
    let mut h = harness();
    h.run_steps(2);
    key(&mut h, egui::Key::V, egui::Modifiers::ALT);
    h.run_steps(2);
    shot(&mut h, "p_view_by_key");
    assert!(h.query_by_label("Theme...").is_some(), "Alt+V opens the View menu");
    key(&mut h, egui::Key::T, egui::Modifiers::NONE);
    h.run_steps(2);
    assert!(matches!(h.state().dialog, Some(Dialog::Theme { .. })), "T chooses Theme...");
    assert!(h.query_by_label("Theme...").is_none(), "the menu closed");
    h.state_mut().dialog = None;
    h.state_mut().store.set_recent_files(&["C:/airfoils/naca0012.dat".into()]);
    h.run_steps(2);
    key(&mut h, egui::Key::F, egui::Modifiers::ALT);
    h.run_steps(2);
    key(&mut h, egui::Key::R, egui::Modifiers::NONE);
    h.run_steps(4);
    shot(&mut h, "p_recent_by_key");
    assert!(h.query_by_label("1 naca0012.dat").is_some(), "R opens Open recent");
}

fn key(h: &mut Harness<'static, App>, key: egui::Key, modifiers: egui::Modifiers) {
    h.input_mut().events.push(egui::Event::ModifiersChanged(modifiers));
    for pressed in [true, false] {
        h.input_mut().events.push(egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers });
    }
    h.run_steps(1);
    h.input_mut().events.push(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
    h.run_steps(1);
}

/// Double-clicks at a point (both clicks in one frame: the harness renders
/// each frame, slower than a double-click allows).
fn double_click_at(h: &mut Harness<'static, App>, x: f32, y: f32) {
    let pos = egui::pos2(x, y);
    h.input_mut().events.push(egui::Event::PointerMoved(pos));
    h.run_steps(1);
    for pressed in [true, false, true, false] {
        h.input_mut().events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        });
    }
    h.run_steps(1);
}

/// Drags the pointer from one point to another.
fn drag(h: &mut Harness<'static, App>, from: egui::Pos2, to: egui::Pos2) {
    let button =
        |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
    h.input_mut().events.push(egui::Event::PointerMoved(from));
    h.run_steps(1);
    h.input_mut().events.push(button(from, true));
    h.run_steps(1);
    for k in 1..=5 {
        h.input_mut().events.push(egui::Event::PointerMoved(from + (to - from) * (k as f32 / 5.0)));
        h.run_steps(1);
    }
    h.input_mut().events.push(button(to, false));
    h.run_steps(2);
}

/// Undo groups, Ctrl+Z in a settings field, and double-clicking a panel's
/// edge (specification 19.2, 19.8).
#[test]
fn settings_and_panels_behave_as_in_python() {
    let mut h = harness();
    h.run_steps(3);
    // Applying a run right after an edit: the edit is a step of its own.
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.state_mut().on_fit();
    finish(&mut h);
    let run = h.state().projects.runs()[0].clone();
    h.state_mut().settings.degree.value.set(4);
    h.run_steps(1);
    h.state_mut().apply_run_settings(&run);
    h.run_steps(2);
    assert_eq!(h.state().settings.degree.value.value, 5);
    h.state_mut().undo();
    h.run_steps(1);
    assert_eq!(h.state().settings.degree.value.value, 4, "the run's settings are one step");
    h.state_mut().undo();
    h.run_steps(1);
    assert_eq!(h.state().settings.degree.value.value, 5, "the edit is another");

    // Ctrl+Z with a settings field focused undoes the setting.
    h.state_mut().settings.control_points.value.set(30);
    h.run_steps(2);
    let current = h.state().settings.values();
    h.state_mut().history.commit(&current);
    click_at(&mut h, 200.0, 397.0);
    assert!(h.ctx.memory(|m| m.focused()).is_some(), "the control points field has the focus");
    key(&mut h, egui::Key::Z, egui::Modifiers::COMMAND);
    h.run_steps(2);
    assert_eq!(h.state().settings.control_points.value.value, 25);

    // Dragging the sidebar wider, then double-clicking its edge.
    let edge = h.state().sidebar_rect.right();
    drag(&mut h, egui::pos2(edge, 600.0), egui::pos2(edge + 80.0, 600.0));
    let wide = h.state().sidebar_rect.width();
    assert!(wide > crate::theme::size::SIDEBAR_WIDTH + 40.0, "{wide}");
    h.run_steps(2);
    // Kept for the next session once the mouse lets go.
    assert_eq!(h.state().store.float("panels/sidebar_width", 0.0) as f32, wide);
    let edge = h.state().sidebar_rect.right();
    double_click_at(&mut h, edge, 600.0);
    h.run_steps(4);
    let default = crate::theme::size::SIDEBAR_WIDTH + crate::theme::size::SPLITTER;
    assert_eq!(h.state().sidebar_rect.width(), default);
    assert_eq!(h.state().store.float("panels/sidebar_width", 0.0) as f32, default);

    // The comb settings stay open while the plot is used.
    click_at(&mut h, 615.0, 45.0);
    let open = |h: &Harness<'static, App>| egui::Popup::is_id_open(&h.ctx, egui::Id::unique("comb_settings"));
    assert!(open(&h));
    click_at(&mut h, 650.0, 300.0);
    assert!(open(&h), "a click in the plot leaves the comb settings open");
    key(&mut h, egui::Key::Escape, egui::Modifiers::NONE);
    assert!(!open(&h));
}

/// A second sweep shows the runs of the first beside its own (19.13).
#[test]
fn sweeps_show_the_whole_project_range() {
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.run_steps(2);
    let sweep = |h: &mut Harness<'static, App>, from: usize, to: usize| {
        let app = h.state_mut();
        app.settings.control_points.value.set(from);
        app.settings.control_points.toggle(true);
        app.settings.control_points.end.set(to);
        h.run_steps(2);
        h.state_mut().on_fit();
        finish(h);
    };
    sweep(&mut h, 20, 21);
    sweep(&mut h, 24, 25);
    let xs: Vec<f64> = h.state().comparison.points.iter().map(|p| p[0]).collect();
    assert_eq!(xs, [20.0, 21.0, 24.0, 25.0]);

    // A click on a run gives the table the arrow keys, as in Qt.
    use egui_kittest::kittest::Queryable;
    let ppp = h.ctx.pixels_per_point();
    let cell = h.query_all_by_label("1").map(|n| n.rect()).rfind(|r| r.min.x > 700.0 * ppp && r.min.x < 1010.0 * ppp);
    let cell = cell.expect("run 1 in the Run column");
    click_at(&mut h, cell.center().x / ppp, cell.center().y / ppp);
    // In the order shown.
    let ids: Vec<String> = h.state().visible_rows().into_iter().map(|(id, _)| id).collect();
    let selected = |h: &Harness<'static, App>| h.state().projects.selected.iter().cloned().collect::<Vec<_>>();
    assert_eq!(selected(&h), [ids[0].clone()]);
    key(&mut h, egui::Key::ArrowDown, egui::Modifiers::NONE);
    assert_eq!(selected(&h), [ids[1].clone()]);
    key(&mut h, egui::Key::ArrowDown, egui::Modifiers::SHIFT);
    let mut two = vec![ids[1].clone(), ids[2].clone()];
    two.sort();
    assert_eq!(selected(&h), two);
    key(&mut h, egui::Key::End, egui::Modifiers::COMMAND);
    assert_eq!(selected(&h), two, "Ctrl moves without selecting");
    key(&mut h, egui::Key::Space, egui::Modifiers::COMMAND);
    assert_eq!(selected(&h).len(), 3);
    key(&mut h, egui::Key::ArrowUp, egui::Modifiers::NONE);
    assert_eq!(selected(&h), [ids[2].clone()]);
    // Pivoted, the runs are columns: Left and Right move.
    h.state_mut().explorer.pivot = true;
    h.run_steps(2);
    key(&mut h, egui::Key::ArrowLeft, egui::Modifiers::NONE);
    assert_eq!(selected(&h), [ids[1].clone()]);
    key(&mut h, egui::Key::Home, egui::Modifiers::NONE);
    assert_eq!(selected(&h), [ids[0].clone()]);
}

/// Zoomed into a few millionths, the plot draws as quickly as zoomed out:
/// the dashed control polygon is only cut into dashes where it is seen.
#[test]
fn deep_zoom_draws_quickly() {
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.run_steps(2);
    h.state_mut().settings.optimizer = OptimizerType::DirectLsq;
    h.state_mut().main_plot.deviation_on = true;
    h.run_steps(2);
    h.state_mut().on_fit();
    finish(&mut h);
    let center = h.state().airfoil.as_ref().unwrap().coordinates[20];
    let bounds = egui_plot::PlotBounds::from_min_max([center.x - 2e-6, center.y - 1e-6], [center.x + 2e-6, center.y + 1e-6]);
    h.state_mut().main_plot.view.fit(Some(bounds));
    let start = std::time::Instant::now();
    h.run_steps(3);
    shot(&mut h, "z_deep_zoom");
    assert!(start.elapsed().as_secs_f64() < 2.0, "three frames took {:?}", start.elapsed());
}

/// A dialog gives its first control the focus with its text selected, so
/// typing replaces it; Enter presses the default button, Tab moves the
/// focus on and a focused push button becomes the default.
#[test]
fn dialogs_focus_their_first_control() {
    use crate::dialogs::Dialog;
    let mut h = harness();
    h.run_steps(2);
    h.state_mut().dialog = Some(Dialog::naca(&h.state().store));
    h.run_steps(4);
    h.input_mut().events.push(egui::Event::Text("0012".into()));
    h.run_steps(2);
    key(&mut h, egui::Key::Enter, egui::Modifiers::NONE);
    h.run_steps(3);
    assert!(h.state().dialog.is_none(), "Enter pressed Create");
    assert_eq!(h.state().airfoil.as_ref().map(|a| a.name.clone()), Some("NACA 0012".into()));
    // In Theme, Tab past the segments and the slider reaches Neutral, which
    // Enter then presses (Close is no longer the default).
    h.state_mut().dialog = Some(Dialog::theme(h.state()));
    h.run_steps(4);
    h.state_mut().temperature = 40;
    for _ in 0..4 {
        key(&mut h, egui::Key::Tab, egui::Modifiers::NONE);
    }
    shot(&mut h, "d_theme_neutral_focused");
    key(&mut h, egui::Key::Enter, egui::Modifiers::NONE);
    h.run_steps(2);
    assert_eq!(h.state().temperature, 0, "Enter pressed the focused Neutral");
    assert!(h.state().dialog.is_some(), "and not Close");
}

#[test]
fn the_column_order_is_kept_by_name() {
    use crate::explorer::ExplorerState;
    let default = ExplorerState::default().column_names();
    assert!(default.starts_with("Run,Optimizer,Goal,"), "{default}");
    assert_eq!(ExplorerState::with_column_names("").column_names(), default);
    // Named columns first, unknown and repeated names skipped, the rest after.
    let state = ExplorerState::with_column_names("Max deviation,Control points,Gone,Max deviation,Run");
    let names = state.column_names();
    assert!(names.starts_with("Max deviation,Control points,Run,Optimizer,Goal,"), "{names}");
    assert_eq!(names.split(',').count(), default.split(',').count());
    assert_eq!(ExplorerState::with_column_names(&names).column_names(), names);
}

/// The run table pages with its scroll bar, and a click on a point of the
/// runs plot brings that run's row into view.
#[test]
fn the_run_table_pages_and_follows_the_runs_plot() {
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.run_steps(2);
    {
        let app = h.state_mut();
        app.settings.optimizer = OptimizerType::DirectLsq;
        app.settings.control_points.value.set(10);
        app.settings.control_points.toggle(true);
        app.settings.control_points.end.set(45);
    }
    h.run_steps(2);
    h.state_mut().on_fit();
    finish(&mut h);
    h.state_mut().refresh_comparison();
    h.run_steps(2);
    use egui_kittest::kittest::Queryable;
    let ppp = h.ctx.pixels_per_point();
    // A run's number in the Run column (the left of the box).
    let in_run_column = |h: &Harness<'static, App>, number: &str| h.query_all_by_label(number).any(|n| n.rect().min.x < 1010.0 * ppp);
    // The rows' scroll bar is at the box's edge: clicking its track below
    // the handle pages down to the last runs.
    let runs = h.state().projects.runs().len().to_string();
    assert!(!in_run_column(&h, &runs), "the last run is out of view");
    for _ in 0..3 {
        click_at(&mut h, 1482.0, 790.0);
    }
    assert!(in_run_column(&h, &runs), "paged down to the last run");
    // The fewest control points, run 1, is now out of view; a click on its
    // point brings it back.
    assert!(!in_run_column(&h, "1"), "run 1 is out of view");
    assert_eq!(h.state().comparison.points[0][0], 10.0);
    h.state_mut().click_plotted_run(0);
    h.run_steps(3);
    assert!(in_run_column(&h, "1"), "run 1 brought into view");
}

/// A search that misses its tolerance says what stopped it, not that the
/// solver did not converge.
#[test]
fn a_missed_tolerance_is_not_a_solver_failure() {
    let mut h = harness();
    // Few points: the search reaches as many control points as they allow.
    h.state_mut().new_naca_airfoil("2412", 12, true);
    h.run_steps(2);
    {
        let app = h.state_mut();
        app.settings.tolerance_goal = true;
        app.settings.auto_tolerance = false;
        app.settings.tolerance = 1e-9;
        app.settings.grow = true;
    }
    h.run_steps(2);
    h.state_mut().on_fit();
    finish(&mut h);
    let status = &h.state().status;
    assert!(status.starts_with("Completed (VarPro LM): "), "{status}");
    assert!(status.contains(" | Tolerance not reached: "), "{status}");
    assert!(status.contains(" control points are as many as the points of a section allow; the closest fit, "), "{status}");
    assert!(!status.contains("did not converge"), "{status}");
    // The readout's report knows the tolerance it missed (drawn in warn).
    let report = h.state().fit.as_ref().unwrap().report.clone().unwrap();
    assert!(report.tolerance == Some(1e-9) && report.max_deviation > 1e-9);
    shot(&mut h, "m_missed_tolerance");
}

/// The Curvature points toggle marks on the curve the extrema the fit
/// report counts and the inflections, and is kept with the other display
/// choices. A symmetric airfoil bends one way only; a strongly cambered
/// one inflects where its lower surface turns concave.
#[test]
fn the_curvature_points_are_marked() {
    use egui_kittest::kittest::Queryable;
    let mut h = harness();
    h.state_mut().new_naca_airfoil("0012", 41, true);
    h.run_steps(2);
    h.state_mut().on_fit();
    finish(&mut h);
    assert!(h.state().main_plot.extrema.is_empty(), "off by default");
    h.get_by_label("Curvature points").click();
    h.run_steps(3);
    let app = h.state();
    let count = app.fit.as_ref().unwrap().report.as_ref().unwrap().fairness.as_ref().unwrap().curvature_extrema;
    assert!(count > 0);
    assert_eq!(app.main_plot.extrema.len(), count);
    assert!(app.main_plot.inflections.is_empty(), "{:?}", app.main_plot.inflections);
    assert!(app.store.bool("display/extrema", false));

    // A fresh window: replacing the fitted airfoil would ask first.
    let mut h = harness();
    h.state_mut().main_plot.show_extrema = true;
    h.state_mut().new_naca_airfoil("6412", 41, true);
    h.run_steps(2);
    h.state_mut().on_fit();
    finish(&mut h);
    // NACA 6412's lower surface turns concave near 20 % chord and stays so.
    let inflections = &h.state().main_plot.inflections;
    assert_eq!(inflections.len(), 1, "{inflections:?}");
    let (_, q) = inflections[0];
    assert!((q.x - 0.2).abs() < 0.03 && q.y < 0.0, "{q:?}");
    shot(&mut h, "x_curvature_points");
}

/// Zoomed with the wheel, a plot stops following its data; the autoscale
/// button in its lower left corner, or a double-click, shows the data again.
#[test]
fn plots_autoscale() {
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.run_steps(3);
    let centre = egui::pos2(880.0, 380.0);
    let zoom = |h: &mut Harness<'static, App>| {
        h.input_mut().events.push(egui::Event::PointerMoved(centre));
        h.run_steps(1);
        h.input_mut().events.push(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: egui::vec2(0.0, 3.0),
            modifiers: Default::default(),
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Wheel,
        });
        h.run_steps(30);
    };
    zoom(&mut h);
    assert!(!h.state().main_plot.view.follow, "zoomed");
    shot(&mut h, "z_autoscale_button");
    // The button, in the plot's lower left corner.
    let button = h.state().main_plot.view.autoscale_at.expect("an autoscale button");
    click_at(&mut h, button.x, button.y);
    h.run_steps(2);
    assert!(h.state().main_plot.view.follow, "the button autoscales");
    // A double-click does the same.
    zoom(&mut h);
    assert!(!h.state().main_plot.view.follow);
    double_click_at(&mut h, centre.x, centre.y);
    h.run_steps(2);
    assert!(h.state().main_plot.view.follow, "a double-click autoscales");
}

/// The tolerance can limit the RMS deviation instead of the largest: the
/// sidebar names the field for it, and the search, its folder and the
/// readout follow.
#[test]
fn the_target_can_be_the_rms_deviation() {
    use egui_kittest::kittest::Queryable;
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.run_steps(2);
    {
        let app = h.state_mut();
        app.settings.tolerance_goal = true;
        app.settings.auto_tolerance = false;
        app.settings.tolerance = 2e-5;
    }
    h.run_steps(2);
    assert!(h.query_by_label("Max deviation").is_some());
    h.get_by_label("RMS deviation").click();
    h.run_steps(2);
    assert_eq!(h.state().settings.tolerance_metric, ToleranceMetric::RmsDeviation);
    shot(&mut h, "t_rms_target");
    h.state_mut().on_fit();
    finish(&mut h);
    let app = h.state();
    let search = &app.projects.searches()[0];
    assert_eq!(search.optimization_config.tolerance_metric, ToleranceMetric::RmsDeviation);
    assert!(search.met, "{}", search.message);
    let report = app.fit.as_ref().unwrap().report.clone().unwrap();
    assert!(report.tolerance_metric == ToleranceMetric::RmsDeviation && report.rms_deviation <= 2e-5);
    let rows = app.visible_rows();
    let target = crate::runs::COLUMNS.iter().position(|c| c.name == "Target").unwrap();
    assert_eq!(rows[0].1[target].1, "RMS deviation");
    // Kept with the other sidebar settings.
    assert_eq!(app.store.value("sidebar").unwrap()["tolerance_metric"], "rms_deviation");
}

/// Ctrl+Enter fits, with the value still being typed into a field; Esc
/// cancels the running fit.
#[test]
fn ctrl_enter_fits_with_the_typed_value_and_esc_cancels() {
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 81, true);
    h.run_steps(2);
    {
        let app = h.state_mut();
        app.settings.tolerance_goal = true;
        app.settings.auto_tolerance = false;
        app.settings.tolerance = 1e-4;
    }
    h.run_steps(2);
    // Into the Max deviation field (below Target), a new value typed and not yet committed.
    click_at(&mut h, 220.0, 240.0);
    key(&mut h, egui::Key::A, egui::Modifiers::COMMAND);
    h.input_mut().events.push(egui::Event::Text("1e-7".into()));
    h.run_steps(1);
    assert_eq!(h.state().settings.tolerance, 1e-4, "still being typed");
    key(&mut h, egui::Key::Enter, egui::Modifiers::COMMAND);
    assert!(h.state().fitting(), "{}", h.state().status);
    assert_eq!(h.state().settings.tolerance, 1e-7);
    let fitting_with = match &h.state().running {
        Some(crate::fitting::Running::Fit(job)) => job.options.tolerance,
        _ => None,
    };
    assert_eq!(fitting_with, Some(1e-7));
    key(&mut h, egui::Key::Escape, egui::Modifiers::NONE);
    finish(&mut h);
    assert!(h.state().status.starts_with("Cancelled: "), "{}", h.state().status);
}

/// The wheel over a number field steps its value as the arrows do, without
/// focusing it first, one step per notch.
#[test]
fn the_wheel_steps_a_number_field() {
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.run_steps(2);
    {
        let app = h.state_mut();
        app.settings.tolerance_goal = true;
        app.settings.auto_tolerance = false;
        app.settings.tolerance = 1e-4;
    }
    h.run_steps(2);
    let wheel = |h: &mut Harness<'static, App>, notches: f32| {
        // Over the Max deviation field (below Target), not clicked.
        h.input_mut().events.push(egui::Event::PointerMoved(egui::pos2(220.0, 240.0)));
        h.run_steps(1);
        h.input_mut().events.push(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: egui::vec2(0.0, notches),
            modifiers: Default::default(),
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Wheel,
        });
        h.run_steps(2);
    };
    wheel(&mut h, 1.0);
    assert_eq!(h.state().settings.tolerance, 2e-4, "one notch up: one step of the 1-2-5 ladder");
    wheel(&mut h, -2.0);
    assert_eq!(h.state().settings.tolerance, 5e-5, "two notches down: two steps");
}

/// The sidebar hides to a rail and comes back, as the explorer does.
#[test]
fn the_sidebar_hides_to_a_rail() {
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 81, true);
    h.run_steps(3);
    let shown = |h: &Harness<'static, App>| h.state().sidebar_visible && h.state().sidebar_rect.width() > 200.0;
    assert!(shown(&h));
    shot(&mut h, "s1_sidebar_shown");
    key(&mut h, egui::Key::B, egui::Modifiers::COMMAND);
    h.run_steps(2);
    assert!(!h.state().sidebar_visible && h.state().sidebar_rect == egui::Rect::NOTHING);
    shot(&mut h, "s2_sidebar_rail");
    // The rail's button, at the window's left edge under the menu bar.
    click_at(&mut h, 16.0, 46.0);
    h.run_steps(2);
    assert!(shown(&h), "the rail button shows it again");
    // Its own hide button, on the Data caption line at the right.
    let right = h.state().sidebar_rect.right();
    click_at(&mut h, right - 12.0 - 12.0, 46.0);
    h.run_steps(2);
    assert!(!h.state().sidebar_visible, "the Data caption's button hides it");
}

/// Pivot is an icon toggle at the right of the filter row, kept between
/// sessions as the theme is.
#[test]
fn pivot_is_an_icon_toggle_that_is_kept() {
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.run_steps(3);
    assert!(!h.state().explorer.pivot);
    // The square button at the right end of the filter row (no project yet).
    click_at(&mut h, 1475.0, 104.0);
    h.run_steps(2);
    assert!(h.state().explorer.pivot);
    assert!(h.state().store.bool("explorer/pivot", false));
    click_at(&mut h, 1475.0, 104.0);
    h.run_steps(2);
    assert!(!h.state().explorer.pivot && !h.state().store.bool("explorer/pivot", true));
}

/// Unsaved runs, replacing a file and deleting runs ask in the app's own
/// dialog, and its answer finishes the action that asked.
#[test]
fn questions_are_dialogs_of_the_app() {
    use egui_kittest::kittest::Queryable;
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.run_steps(2);
    h.state_mut().on_fit();
    finish(&mut h);
    assert!(h.state().projects.modified);
    // A new airfoil would lose the unsaved run: asked first.
    h.state_mut().new_naca_airfoil("0012", 41, true);
    h.run_steps(3);
    assert!(matches!(h.state().dialog, Some(crate::dialogs::Dialog::Confirm(_))));
    shot(&mut h, "q_unsaved_runs");
    assert_eq!(h.state().airfoil.as_ref().unwrap().name, "NACA 2412", "nothing changed yet");
    h.get_by_label("Cancel").click();
    h.run_steps(3);
    assert!(h.state().dialog.is_none());
    assert_eq!(h.state().status, "Cancelled: the project stays open");
    assert_eq!(h.state().airfoil.as_ref().unwrap().name, "NACA 2412");
    // Don't save goes on with the new airfoil.
    h.state_mut().new_naca_airfoil("0012", 41, true);
    h.run_steps(3);
    h.get_by_label("Don't save").click();
    h.run_steps(3);
    assert!(h.state().dialog.is_none());
    assert_eq!(h.state().airfoil.as_ref().unwrap().name, "NACA 0012");
    assert!(h.state().projects.file.is_none(), "the old project is gone");
    // Deleting runs asks too; Delete deletes.
    h.state_mut().on_fit();
    finish(&mut h);
    let id = h.state().projects.runs()[0].run_id.clone();
    h.state_mut().select_run(&id);
    h.state_mut().confirm_delete_runs();
    h.run_steps(3);
    shot(&mut h, "q_delete_runs");
    assert_eq!(h.state().projects.runs().len(), 1, "asked first");
    // The dialog's button, drawn after the explorer's Delete button.
    h.get_all_by_label("Delete").last().unwrap().click();
    h.run_steps(3);
    assert!(h.state().projects.runs().is_empty());
    // Replacing a file: Enter answers Cancel, as Qt's question defaulted to No.
    h.state_mut().dialog = Some(crate::dialogs::Dialog::Confirm(crate::dialogs::Confirm {
        title: "Save project as".into(),
        message: "x.asfproj already exists. Replace it?".into(),
        accept: "Replace".into(),
        other: None,
        save_first: false,
        default_accept: false,
        cancelled: "Save cancelled.".into(),
        then: Box::new(|app: &mut App| app.set_status("Replaced")),
    }));
    h.run_steps(6);
    shot(&mut h, "q_replace");
    key(&mut h, egui::Key::Enter, egui::Modifiers::NONE);
    h.run_steps(2);
    assert!(h.state().dialog.is_none());
    assert_eq!(h.state().status, "Save cancelled.");
}

/// The plot toggles, the live preview, the convergence axes and the runs
/// plot's columns are written when they change and read on the next start.
#[test]
fn display_choices_are_kept() {
    let mut h = harness();
    h.run_steps(2);
    {
        let app = h.state_mut();
        app.main_plot.show_knots = true;
        app.main_plot.deviation_on = true;
        app.main_plot.deviation_scale = 20.0;
        app.live_mode = crate::job::LiveMode::Detailed;
        app.loss.log_mode = 3;
        app.explorer.y_column = crate::runs::COLUMNS.iter().position(|c| c.name == "Max deviation").unwrap();
    }
    h.run_steps(2);
    let store = &h.state().store;
    assert!(store.bool("display/knots", false) && store.bool("display/deviation", false));
    assert_eq!(store.float("display/deviation_scale", 0.0), 20.0);
    assert_eq!((store.int("display/live", 0), store.int("display/loss_log", 0)), (2, 3));
    assert_eq!(store.string("display/runs_y", ""), "Max deviation");
    // The next start reads them.
    {
        let app = h.state_mut();
        app.main_plot.show_knots = false;
        app.live_mode = crate::job::LiveMode::Minimal;
        app.explorer.y_column = 0;
        app.load_display();
        assert!(app.main_plot.show_knots && app.live_mode == crate::job::LiveMode::Detailed);
        assert_eq!(crate::runs::COLUMNS[app.explorer.y_column].name, "Max deviation");
    }
}

/// The sidebar settings are written to the settings file as they change
/// (without the sweeps) and read back at the next start.
#[test]
fn sidebar_settings_are_kept() {
    let mut h = harness();
    h.run_steps(2);
    {
        let app = h.state_mut();
        app.settings.tolerance_goal = true;
        app.settings.smoothing.value = 1e-7;
        app.settings.control_points.value.set(6);
        app.settings.smoothing.toggle(true);
    }
    h.run_steps(2);
    let kept = h.state().store.value("sidebar").cloned().expect("written");
    assert_eq!(kept["smoothing"], 1e-7);
    assert_eq!((kept["tolerance_goal"].as_bool(), kept["control_points"].as_u64()), (Some(true), Some(6)));
    let defaults = crate::settings::Settings::default();
    let mut next = defaults.clone();
    next.restore(&crate::settings::Values::from_json(&kept, &defaults.values()));
    assert!(next.tolerance_mode() && next.smoothing.value == 1e-7 && next.control_points.value.value == 6);
    assert!(!next.sweeping());
}

/// A failure's status is drawn in the error colour until the next message.
#[test]
fn failures_show_in_the_status_bar_in_the_error_colour() {
    let mut h = harness();
    h.run_steps(2);
    // Saving with nothing to save is a plain message.
    h.state_mut().on_fit();
    h.run_steps(2);
    assert_eq!(h.state().status, "Load an airfoil first.");
    assert!(!h.state().status_failed);
    h.state_mut().open_path(std::path::Path::new("Z:/no/such/airfoil.dat"));
    h.run_steps(2);
    assert!(h.state().status.starts_with("File not found: "), "{}", h.state().status);
    assert!(h.state().status_failed);
    shot(&mut h, "f_failed_status");
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.run_steps(2);
    assert!(!h.state().status_failed, "the next message is plain again");
}

/// The tolerance line comes from the data and the sidebar: it shows as soon
/// as the goal is a tolerance, before any fit, stays put while fitting, and
/// is the tolerance times the deviation scale out from every data point.
#[test]
fn the_tolerance_line_is_drawn_from_the_data() {
    let mut h = harness();
    h.state_mut().new_naca_airfoil("2412", 41, true);
    h.run_steps(2);
    assert!(h.state().main_plot.tolerance_line.is_empty(), "no tolerance goal yet");
    {
        let app = h.state_mut();
        app.settings.tolerance_goal = true;
        app.settings.auto_tolerance = false;
        app.settings.tolerance = 1e-4;
    }
    h.run_steps(2);
    let before = h.state().main_plot.tolerance_line.clone();
    let data = h.state().airfoil.as_ref().unwrap().coordinates.clone();
    assert_eq!(before.len(), data.len());
    let scale = h.state().main_plot.deviation_scale;
    for (q, on_line) in data.iter().zip(&before) {
        assert!(((on_line - q).norm() - scale * 1e-4).abs() < 1e-12);
    }
    h.state_mut().on_fit();
    finish(&mut h);
    assert_eq!(h.state().main_plot.tolerance_line, before, "the fit does not move it");
    h.state_mut().settings.tolerance_goal = false;
    h.run_steps(2);
    assert!(h.state().main_plot.tolerance_line.is_empty());
}
