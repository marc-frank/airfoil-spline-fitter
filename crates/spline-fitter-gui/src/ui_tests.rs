//! The window driven headlessly: the reference session of the
//! specification (section 21, item 17), with screenshots on request.

use std::time::Duration;

use eframe::egui;
use egui_kittest::Harness;
use spline_fitter::models::OptimizerType;

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
        assert!(app.status.ends_with(" | Error 8.978e-07 | max deviation 4.59e-06, RMS 8.98e-07, 49 control points"), "{}", app.status);
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
        assert!(app.status.ends_with(" | Error 1.897e-05 | max deviation 4.72e-05, RMS 8.81e-06, 49 control points"), "{}", app.status);
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
        assert!(app.status.ends_with(" | Best: Control points 25, error 8.978e-07"), "{}", app.status);
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
        app.settings.tolerance = 1e-4;
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
    finish(&mut h);
    {
        let app = h.state();
        assert!(app.status.starts_with("Completed (VarPro LM): "), "{}", app.status);
        // Every refit of the search is a run of its own, told apart by its counts.
        let counts: Vec<Vec<usize>> = app.projects.runs().iter().map(|r| r.result.section_counts()).collect();
        assert!(counts.iter().enumerate().all(|(i, c)| !counts[..i].contains(c)), "{counts:?}");
        assert_eq!(counts.last(), Some(&app.fit.as_ref().unwrap().section_counts()));
        // Each earlier refit keeps only its own evaluations; the last run
        // has the whole search.
        let runs = app.projects.runs();
        let (last, earlier) = runs.split_last().unwrap();
        let own: usize = earlier.iter().map(|r| r.loss_history.len()).sum();
        assert!(earlier.iter().all(|r| !r.loss_history.is_empty()));
        assert!(own <= last.loss_history.len(), "{own} > {}", last.loss_history.len());
    }
    h.state_mut().main_plot.deviation_on = true;
    h.state_mut().main_plot.show_knots = true;
    h.state_mut().main_plot.refresh_knots();
    h.state_mut().main_plot.refresh_deviation();
    h.run_steps(3);
    shot(&mut h, "6_light_tolerance");
    // Along control points the tolerance runs are plotted at the counts they
    // found, not at the start count they share.
    {
        let app = h.state();
        let mut xs: Vec<f64> = app.comparison.points.iter().map(|p| p[0]).collect();
        let mut fitted: Vec<f64> = app.projects.runs().iter().map(|r| r.result.num_control_points() as f64).collect();
        xs.sort_by(f64::total_cmp);
        fitted.sort_by(f64::total_cmp);
        assert_eq!(xs, fitted);
        assert!(xs.first() != xs.last(), "{xs:?}");
        // The search's tolerance is a line in the plot of its errors.
        assert_eq!(app.comparison.tolerance, Some(1e-4));
        // Each point names its run.
        assert_eq!(app.comparison.labels.len(), app.comparison.points.len());
        assert!(app.comparison.labels.iter().all(|l| l.starts_with("Run ") && l.contains(" control points, error ")), "{:?}", app.comparison.labels);
    }
    // The rows' scroll bar is at the box's edge: clicking its track below
    // the handle pages down to the last runs.
    {
        use egui_kittest::kittest::Queryable;
        let runs = h.state().projects.runs().len().to_string();
        // The last run's number in the Run column (the left of the box).
        let ppp = h.ctx.pixels_per_point();
        let in_run_column = |h: &Harness<'static, App>| h.query_all_by_label(&runs).any(|n| n.rect().min.x < 1010.0 * ppp);
        assert!(!in_run_column(&h), "the last run is out of view");
        for _ in 0..3 {
            click_at(&mut h, 1482.0, 790.0);
        }
        assert!(in_run_column(&h), "paged down to the last run");
    }
    // A click on a point selects its run, shows that fit and brings its
    // row into view: the most control points, the search's first run, is
    // out of view after paging down.
    let number = {
        let app = h.state_mut();
        let last = app.comparison.points.len() - 1;
        let point = app.comparison.points[last];
        app.click_plotted_run(last);
        let id = app.projects.selected.iter().next().unwrap().clone();
        assert_eq!(app.projects.selected.len(), 1);
        assert_eq!(app.projects.shown_run.as_deref(), Some(id.as_str()));
        assert_eq!(app.comparison.marked, vec![point]);
        let run = app.projects.run(&id).unwrap();
        assert_eq!(run.result.num_control_points() as f64, point[0]);
        assert!(app.comparison.labels[last].starts_with(&format!("Run {}: ", run.number)));
        run.number.to_string()
    };
    {
        use egui_kittest::kittest::Queryable;
        let ppp = h.ctx.pixels_per_point();
        let in_run_column = |h: &Harness<'static, App>| h.query_all_by_label(&number).any(|n| n.rect().min.x < 1010.0 * ppp);
        assert!(!in_run_column(&h), "run {number} starts out of view");
        h.run_steps(3);
        assert!(in_run_column(&h), "run {number} brought into view");
    }
    h.state_mut().on_report();
    h.run_steps(4);
    shot(&mut h, "6r_report");
    h.state_mut().report = None;
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
    assert!(h.state().modal_open());
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
    let edge = h.state().sidebar_rect.right();
    double_click_at(&mut h, edge, 600.0);
    h.run_steps(4);
    assert_eq!(h.state().sidebar_rect.width(), crate::theme::size::SIDEBAR_WIDTH + crate::theme::size::SPLITTER);

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

