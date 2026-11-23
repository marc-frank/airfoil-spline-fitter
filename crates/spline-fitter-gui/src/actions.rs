//! Opening, creating, saving and exporting airfoils, fits and projects.

use std::path::{Path, PathBuf};

use spline_fitter::analysis::default_tolerance;
use spline_fitter::bspline_varpro::airfoil::{resample, trailing_edge_segment, Naca4};
use spline_fitter::bspline_varpro::export::{self, merge_curves, Units};
use spline_fitter::formatting::file_stem;
use spline_fitter::geometry::AirfoilData;
use spline_fitter::io::{export_airfoil_data, export_spline_data, parse_airfoil_file, parse_airfoil_text, SplineFile};
use spline_fitter::project::{load_project, save_project, ProjectFile};
use spline_fitter::Vec2;

use crate::app::App;
use crate::dialogs::{CadChoice, Dialog};

/// Fewest pasted coordinates that make an airfoil.
const MIN_PASTED_POINTS: usize = 5;

fn home() -> PathBuf {
    dirs::home_dir().unwrap_or_default()
}

fn extension(path: &Path) -> String {
    path.extension().map(|e| format!(".{}", e.to_string_lossy().to_lowercase())).unwrap_or_default()
}

fn plural(k: usize) -> &'static str {
    if k == 1 {
        ""
    } else {
        "s"
    }
}

impl App {
    /// File > Open airfoil.
    pub fn on_open_airfoil(&mut self) {
        if self.fitting() {
            return;
        }
        if let Some(path) =
            self.file_dialog().set_title("Open airfoil").set_directory(home()).add_filter("Airfoil data", &["dat"]).pick_file()
        {
            self.load_airfoil(&path);
        }
    }

    /// Opens an airfoil or project file (recent files, drag and drop).
    pub fn open_path(&mut self, path: &Path) {
        if self.fitting() {
            self.set_status("Cancel the running fit to open a file");
            return;
        }
        if !path.is_file() {
            self.store.remove_recent(&path.to_string_lossy());
            self.fail(format!("File not found: {}", path.display()));
            return;
        }
        if extension(path) == ".asfproj" {
            self.open_project(path);
        } else {
            self.load_airfoil(path);
        }
    }

    fn load_airfoil(&mut self, path: &Path) {
        let ext = extension(path);
        if ext != ".dat" {
            self.set_status(format!("Unsupported file type: {ext}"));
            return;
        }
        let parsed = match parse_airfoil_file(path) {
            Err(e) => return self.fail(format!("Failed to load: {e}")),
            Ok(None) => return self.fail("No coordinates found in the file"),
            Ok(Some(p)) => p,
        };
        let filename = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let name = parsed.name.unwrap_or_else(|| filename.clone());
        self.set_airfoil(parsed.coords, &filename, &name, parsed.precision, &path.to_string_lossy(), "Loaded");
    }

    /// Makes `coords` the airfoil (a file's `path` goes to the recent
    /// files); false when the coordinates cannot be split, or while the
    /// user is asked about unsaved runs (the answer finishes it).
    pub fn set_airfoil(&mut self, coords: Vec<Vec2>, filename: &str, name: &str, precision: i32, path: &str, verb: &str) -> bool {
        let args = (coords.clone(), filename.to_string(), name.to_string(), precision, path.to_string(), verb.to_string());
        let retry = move |app: &mut App| {
            let (coords, filename, name, precision, path, verb) = args;
            app.set_airfoil(coords, &filename, &name, precision, &path, &verb);
        };
        if !self.confirm_discard("Cancelled: the project stays open", retry) {
            return false;
        }
        // The airfoil is built before anything changes, so a failure leaves
        // the window as it was.
        let most = coords.len().saturating_sub(1).max(1);
        let sections = self.settings.sections.value.value.min(most);
        let airfoil = match AirfoilData::new(coords, sections, filename, name, precision) {
            Ok(a) => a,
            Err(e) => {
                self.fail(format!("Failed to load: {e}"));
                return false;
            }
        };
        self.quiet = true;
        self.settings.sections.value.set_max(most);
        self.fit = None;
        self.reset_project();
        self.airfoil_path = path.to_string();
        self.main_plot.set_data(&airfoil.coordinates);
        self.loss.reset();
        self.set_status(format!(
            "{verb}: {} ({} points, {} sections: {})",
            airfoil.display_name(),
            airfoil.coordinates.len(),
            airfoil.num_sections(),
            airfoil.section_sizes()
        ));
        self.airfoil = Some(airfoil);
        self.update_auto_tolerance();
        self.accept_settings_quietly();
        if !path.is_empty() {
            self.store.add_recent(Path::new(path));
        }
        true
    }

    /// Records this frame's program changes as quiet ones.
    pub fn accept_settings_quietly(&mut self) {
        self.history.on_change(self.settings.values(), true, std::time::Instant::now());
        self.history.commit(&self.settings.values());
        self.accept_settings();
        self.quiet = false;
    }

    /// Sets the tolerance from the airfoil when Auto is checked.
    pub fn update_auto_tolerance(&mut self) {
        if let (true, Some(a)) = (self.settings.auto_tolerance, &self.airfoil) {
            self.settings.tolerance = default_tolerance(&a.coordinates, a.precision).max(1e-9);
        }
    }

    /// Edit > Paste airfoil: a file on the clipboard, or coordinates as text.
    pub fn on_paste(&mut self) {
        if self.fitting() {
            return;
        }
        let text = arboard::Clipboard::new().and_then(|mut c| c.get_text()).unwrap_or_default();
        let as_path = PathBuf::from(text.trim().trim_matches('"'));
        let ext = extension(&as_path);
        if (ext == ".dat" || ext == ".asfproj") && as_path.is_file() {
            return self.open_path(&as_path);
        }
        match parse_airfoil_text(&text, None) {
            Some(parsed) if parsed.coords.len() >= MIN_PASTED_POINTS => {
                let name = parsed.name.unwrap_or_else(|| "Pasted airfoil".into());
                self.set_airfoil(parsed.coords, "", &name, if parsed.precision > 0 { parsed.precision } else { 6 }, "", "Pasted");
            }
            _ => self.fail("No airfoil coordinates on the clipboard: copy x y pairs, one point per line"),
        }
    }

    /// File > New NACA airfoil.
    pub fn on_new_naca(&mut self) {
        if !self.fitting() {
            self.dialog = Some(Dialog::naca(&self.store));
        }
    }

    pub fn new_naca_airfoil(&mut self, code: &str, per_side: usize, closed: bool) {
        if self.fitting() {
            return;
        }
        let points = match Naca4::new(code, closed) {
            Ok(naca) => naca.points(per_side).into_iter().map(|p| p.map(|v| (v * 1e6).round_ties_even() / 1e6)).collect(),
            Err(e) => return self.fail(format!("Cannot create the airfoil: {e}")),
        };
        self.set_airfoil(points, &format!("naca{code}"), &format!("NACA {code}"), 6, "", "Created");
    }

    /// File > Save fitted airfoil: asks for the points per side first.
    pub fn on_save_airfoil(&mut self) {
        let (Some(_), Some(airfoil)) = (&self.fit, &self.airfoil) else {
            return self.set_status("No fitted spline to save.");
        };
        let default = 20.max(airfoil.coordinates.len().div_ceil(2));
        self.dialog = Some(Dialog::points_per_side(default));
    }

    pub fn save_airfoil(&mut self, per_side: usize) {
        let (Some(fit), Some(airfoil)) = (self.fit.clone(), self.airfoil.clone()) else { return };
        let name = if airfoil.name.is_empty() { &airfoil.filename } else { &airfoil.name };
        let default = format!("{}_fitted.dat", file_stem(airfoil.file_base(), "airfoil"));
        let Some(path) = self.file_dialog()
            .set_title("Save fitted airfoil")
            .set_directory(home())
            .set_file_name(&default)
            .add_filter("Airfoil data", &["dat"])
            .save_file()
        else {
            return self.set_status("Save cancelled.");
        };
        let result = resample(&fit.curves(), per_side)
            .and_then(|points| export_airfoil_data(&path, &points, Some(name), Some(airfoil.precision.max(6))));
        match result {
            Ok(()) => self.set_status(format!("Saved: {}", path.display())),
            Err(e) => self.fail(format!("Save failed: {e}")),
        }
    }

    /// File > Save spline.
    pub fn on_save_spline(&mut self) {
        let (Some(fit), Some(airfoil)) = (self.fit.clone(), self.airfoil.clone()) else {
            return self.set_status("No fitted spline to save.");
        };
        let default = format!("{}.deg{}.spline.dat", file_stem(airfoil.file_base(), "airfoil"), fit.degree);
        let Some(path) = self.file_dialog()
            .set_title("Save spline control points")
            .set_directory(home())
            .set_file_name(&default)
            .add_filter("Spline data", &["spline.dat", "dat"])
            .save_file()
        else {
            return self.set_status("Save cancelled.");
        };
        let continuity = self
            .projects
            .shown()
            .map(|run| run.spline_config.junction_specs())
            .or_else(|| self.settings.spline_config().ok().map(|c| c.junction_specs()));
        let spline = SplineFile {
            name: if airfoil.name.is_empty() { airfoil.file_base().to_string() } else { airfoil.name.clone() },
            degree: Some(fit.degree),
            control_points: fit.section_control_points.clone(),
            knots: Some(fit.all_knots()),
            continuity,
        };
        match export_spline_data(&path, &spline, Some(12)) {
            Ok(()) => self.set_status(format!("Saved: {}", path.display())),
            Err(e) => self.fail(format!("Save failed: {e}")),
        }
    }

    /// File > Export CAD.
    pub fn on_export_cad(&mut self) {
        if self.fit.is_none() {
            return self.set_status("No fitted spline to export.");
        }
        self.dialog = Some(Dialog::cad(&self.store));
    }

    pub fn export_cad(&mut self, choice: CadChoice) {
        let (Some(fit), Some(airfoil)) = (self.fit.clone(), self.airfoil.clone()) else { return };
        self.store.set_int("cad_format", choice.format as i64);
        self.store.set_string("cad_units", choice.units);
        self.store.set_float("cad_scale", choice.scale);
        self.store.set_bool("cad_merge", choice.merge);
        self.store.set_bool("cad_close_te", choice.close_te);
        let (label, suffix) = crate::dialogs::CAD_FORMATS[choice.format];
        let Some(path) = self.file_dialog()
            .set_title("Export CAD")
            .set_directory(home())
            .set_file_name(format!("{}{suffix}", file_stem(airfoil.file_base(), "airfoil")))
            .add_filter(label, &[suffix.trim_start_matches('.')])
            .save_file()
        else {
            return self.set_status("Export cancelled.");
        };
        let result = (|| {
            let mut curves = fit.curves();
            if choice.merge {
                curves = vec![merge_curves(&curves)?];
            }
            if choice.close_te {
                if let Some(segment) = trailing_edge_segment(&curves) {
                    curves.push(segment);
                }
            }
            let units: Units = choice.units.parse()?;
            export::write(&path, &curves, choice.scale, units)
        })();
        match result {
            Ok(_) => self.set_status(format!("Exported: {}", path.display())),
            Err(e) => self.fail(format!("Export failed: {e}")),
        }
    }

    /// View > Fit report: the report of the shown fit in the bottom block.
    pub fn on_report(&mut self) {
        if !self.ensure_report_panel() {
            return self.set_status("No fitted spline.");
        }
        self.set_bottom_view(1);
    }

    // -----------------------------------------------------------------
    // Projects

    /// True when nothing unsaved is lost (or the user already said so).
    /// Otherwise asks, in the app's own dialog: Save (then goes on if it
    /// saved), Don't save (goes on) or Cancel (status `cancelled`); going
    /// on calls `retry`, which repeats the action that asked.
    pub fn confirm_discard(&mut self, cancelled: &str, retry: impl FnOnce(&mut App) + 'static) -> bool {
        if std::mem::take(&mut self.discard_confirmed) || self.projects.file.is_none() || !self.projects.modified {
            return true;
        }
        let name = self
            .projects
            .path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "the project".into());
        self.dialog = Some(Dialog::Confirm(crate::dialogs::Confirm {
            title: "Unsaved runs".into(),
            message: format!("Save the changes to {name}? Runs that are not saved are lost."),
            accept: "Save".into(),
            other: Some("Don't save".into()),
            save_first: true,
            default_accept: true,
            cancelled: cancelled.into(),
            then: Box::new(move |app: &mut App| {
                app.discard_confirmed = true;
                retry(app);
                app.discard_confirmed = false;
            }),
        }));
        false
    }

    /// The window may close: unsaved runs confirmed, a running fit cancelled.
    pub fn confirm_close(&mut self) -> bool {
        if !self.confirm_discard("Close cancelled.", |app| app.close_now = true) {
            return false;
        }
        if let Some(r) = &self.running {
            r.cancel();
        }
        true
    }

    pub fn reset_project(&mut self) {
        self.projects = crate::explorer::Projects::default();
        self.explorer.forget_points();
        self.refresh_comparison();
    }

    /// File > Open project.
    pub fn on_open_project(&mut self) {
        if self.fitting() {
            return;
        }
        if let Some(path) = self.file_dialog()
            .set_title("Open project")
            .set_directory(home())
            .add_filter("Spline project", &["asfproj"])
            .add_filter("All files", &["*"])
            .pick_file()
        {
            self.open_project(&path);
        }
    }

    pub fn open_project(&mut self, path: &Path) -> bool {
        if self.fitting() {
            self.set_status("Cancel the running fit to open a file");
            return false;
        }
        let loaded = match load_project(path).and_then(|l| {
            // Pre-check: the airfoil splits and the last run's curves build.
            l.project.airfoil.resplit(l.project.airfoil.num_sections())?;
            if let Some(last) = l.project.runs.last() {
                l.project.airfoil.resplit(last.spline_config.num_sections)?;
                for (i, c) in last.result.section_control_points.iter().enumerate() {
                    spline_fitter::BSpline::new(last.result.degree, last.result.knots(i), c.clone())?;
                }
            }
            Ok(l)
        }) {
            Ok(l) => l,
            Err(e) => {
                self.fail(format!("Failed to load project: {e}"));
                return false;
            }
        };
        let again = path.to_path_buf();
        if !self.confirm_discard("Open cancelled.", move |app| {
            app.open_project(&again);
        }) {
            return false;
        }
        let ProjectFile { metadata, airfoil, runs, searches } = loaded.project;
        let run_count = runs.len();
        self.quiet = true;
        self.fit = None;
        self.settings.sections.value.set_max(airfoil.coordinates.len().saturating_sub(1).max(1));
        self.settings.sections.value.set(airfoil.num_sections());
        self.airfoil_path = path.to_string_lossy().into_owned();
        self.main_plot.set_data(&airfoil.coordinates);
        self.loss.reset();
        self.airfoil = Some(airfoil.clone());
        self.projects = crate::explorer::Projects::default();
        self.projects.last_number = runs.iter().map(|r| r.number).max().unwrap_or(0);
        self.projects.file = Some(ProjectFile { metadata, airfoil, runs, searches });
        self.projects.path = Some(path.to_path_buf());
        self.accept_settings_quietly();
        if let Some(last) = self.projects.file.as_ref().and_then(|f| f.runs.last()).cloned() {
            self.apply_run_settings(&last);
            self.show_run(&last);
            self.set_status(format!(
                "Loaded project with {run_count} run{}, showing {}",
                plural(run_count),
                last.optimization_config.optimizer.display_name()
            ));
        } else {
            self.set_status(format!("Loaded project: {}", path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()));
        }
        for warning in loaded.warnings {
            eprintln!("{warning}");
        }
        self.explorer.forget_points();
        self.refresh_comparison();
        self.store.add_recent(path);
        true
    }

    /// File > Save project.
    pub fn on_save_project(&mut self) {
        let Some(path) = self.projects.path.clone().filter(|_| self.projects.file.is_some()) else {
            return self.on_save_project_as();
        };
        let result = save_project(self.projects.file.as_mut().unwrap(), &path);
        match result {
            Ok(()) => {
                self.projects.modified = false;
                self.set_status(format!("Saved: {}", path.display()));
            }
            Err(e) => self.fail(format!("Save failed: {e}")),
        }
    }

    /// File > Save project as.
    pub fn on_save_project_as(&mut self) {
        let Some(airfoil) = self.airfoil.clone() else {
            return self.set_status("No airfoil loaded to save.");
        };
        if self.projects.file.is_none() {
            self.projects.file = Some(ProjectFile::new(airfoil.clone()));
        }
        let Some(mut path) = self.file_dialog()
            .set_title("Save project as")
            .set_directory(home())
            .set_file_name(format!("{}.asfproj", file_stem(airfoil.file_base(), "project")))
            .add_filter("Spline project", &["asfproj"])
            .save_file()
        else {
            return self.set_status("Save cancelled.");
        };
        if extension(&path) != ".asfproj" {
            path = PathBuf::from(format!("{}.asfproj", path.display()));
            if path.exists() {
                let base = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                self.dialog = Some(Dialog::Confirm(crate::dialogs::Confirm {
                    title: "Save project as".into(),
                    message: format!("{base} already exists. Replace it?"),
                    accept: "Replace".into(),
                    other: None,
                    save_first: false,
                    default_accept: false,
                    cancelled: "Save cancelled.".into(),
                    then: Box::new(move |app: &mut App| app.save_project_to(path)),
                }));
                return;
            }
        }
        self.save_project_to(path);
    }

    /// Saves the project to `path`, which becomes its file.
    fn save_project_to(&mut self, path: PathBuf) {
        match save_project(self.projects.file.as_mut().unwrap(), &path) {
            Ok(()) => {
                self.projects.path = Some(path.clone());
                self.projects.modified = false;
                self.store.add_recent(&path);
                self.set_status(format!("Saved: {}", path.display()));
            }
            Err(e) => self.fail(format!("Save failed: {e}")),
        }
    }

    /// Delete runs: confirm, then remove the selected runs.
    pub fn confirm_delete_runs(&mut self) {
        let k = self.projects.selected.len();
        if k == 0 || self.fitting() {
            return;
        }
        self.dialog = Some(Dialog::Confirm(crate::dialogs::Confirm {
            title: "Delete runs".into(),
            message: format!("Delete {k} run{} from the project? Edit > Restore deleted runs brings them back.", plural(k)),
            accept: "Delete".into(),
            other: None,
            save_first: false,
            default_accept: true,
            cancelled: String::new(),
            then: Box::new(|app: &mut App| app.delete_selected_runs()),
        }));
    }
}
