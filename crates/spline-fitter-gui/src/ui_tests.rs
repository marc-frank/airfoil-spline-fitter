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
    }
    h.state_mut().main_plot.deviation_on = true;
    h.state_mut().main_plot.show_knots = true;
    h.state_mut().main_plot.refresh_knots();
    h.state_mut().main_plot.refresh_deviation();
    h.run_steps(3);
    shot(&mut h, "6_light_tolerance");
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
    h.state_mut().open_column_popup(5, egui::pos2(1300.0, 165.0));
    h.run_steps(3);
    shot(&mut h, "p_column_popup");
    h.state_mut().explorer.close_popup();
    h.run_steps(3);
    click_at(&mut h, 615.0, 45.0);
    shot(&mut h, "p_comb_panel");
}
