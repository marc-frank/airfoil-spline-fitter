//! Opening, creating, saving and exporting airfoils, fits and projects.

use std::path::{Path, PathBuf};

use spline_fitter::analysis::default_tolerance;
use spline_fitter::bspline_varpro::airfoil::{resample, trailing_edge_segment, Naca4};
use spline_fitter::bspline_varpro::export::{self, merge_curves, Units};
use spline_fitter::formatting::file_stem;
use spline_fitter::geometry::AirfoilData;
use spline_fitter::io::{airfoil_data_text, decode, parse_airfoil_text, spline_data_text, ParsedAirfoil, SplineFile};
use spline_fitter::project::{load_project, parse_project, ProjectFile};
use spline_fitter::{Result, Vec2};

use crate::app::App;
use crate::dialogs::{CadChoice, Dialog};

/// Fewest pasted coordinates that make an airfoil.
const MIN_PASTED_POINTS: usize = 5;

/// Where a project comes from: a file, or the text of one the browser read.
#[derive(Clone)]
enum ProjectSource {
    Path(PathBuf),
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    Text {
        name: String,
        text: String,
    },
}

/// What a finished save says: the browser downloads the file.
const SAVED: &str = if cfg!(target_arch = "wasm32") { "Downloaded" } else { "Saved" };

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
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = self
            .file_dialog()
            .set_title("Open airfoil")
            .set_directory(crate::parent::home())
            .add_filter("Airfoil data", &["dat"])
            .pick_file()
        {
            self.load_airfoil(&path);
        }
        #[cfg(target_arch = "wasm32")]
        crate::web::pick_file("Open airfoil", "Airfoil data", &["dat"]);
    }

    /// Opens a file the browser read: an airfoil or a project.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub fn open_bytes(&mut self, name: &str, bytes: &[u8]) {
        if self.fitting() {
            self.set_status("Cancel the running fit to open a file");
            return;
        }
        let ext = extension(Path::new(name));
        if ext == ".asfproj" {
            self.open_project_from(ProjectSource::Text { name: name.to_string(), text: decode(bytes) });
        } else if ext == ".dat" {
            let stem = Path::new(name).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let parsed = parse_airfoil_text(&decode(bytes), Some(&stem));
            self.set_parsed_airfoil(Ok(parsed), &stem, "");
        } else {
            self.set_status(format!("Unsupported file type: {ext}"));
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
        let filename = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let parsed = spline_fitter::io::read_text(path).map(|text| parse_airfoil_text(&text, Some(&filename)));
        self.set_parsed_airfoil(parsed, &filename, &path.to_string_lossy());
    }

    /// Makes a read airfoil file the airfoil; `path` is empty for one the
    /// browser read.
    fn set_parsed_airfoil(&mut self, parsed: Result<Option<ParsedAirfoil>>, filename: &str, path: &str) {
        let parsed = match parsed {
            Err(e) => return self.fail(format!("Failed to load: {e}")),
            Ok(None) => return self.fail("No coordinates found in the file"),
            Ok(Some(p)) => p,
        };
        let name = parsed.name.unwrap_or_else(|| filename.to_string());
        self.set_airfoil(parsed.coords, filename, &name, parsed.precision, path, "Loaded");
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
        self.history.on_change(self.settings.values(), true, web_time::Instant::now());
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
        #[cfg(not(target_arch = "wasm32"))]
        self.paste_text(&arboard::Clipboard::new().and_then(|mut c| c.get_text()).unwrap_or_default());
        // The browser answers later (crate::web::Incoming::Text).
        #[cfg(target_arch = "wasm32")]
        crate::web::read_clipboard();
    }

    /// Pastes `text`: a file path, or coordinates.
    pub fn paste_text(&mut self, text: &str) {
        if self.fitting() {
            return;
        }
        let as_path = PathBuf::from(text.trim().trim_matches('"'));
        let ext = extension(&as_path);
        if (ext == ".dat" || ext == ".asfproj") && as_path.is_file() {
            return self.open_path(&as_path);
        }
        match parse_airfoil_text(text, None) {
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

    /// Asks where to save (the browser downloads `default` instead) and
    /// writes the text `make` returns for that file; `None` when cancelled.
    fn save_text_as(
        &mut self,
        title: &str,
        default: &str,
        filter: (&str, &[&str]),
        make: impl FnOnce(&Path) -> Result<String>,
    ) -> Option<Result<PathBuf>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = self
                .file_dialog()
                .set_title(title)
                .set_directory(crate::parent::home())
                .set_file_name(default)
                .add_filter(filter.0, filter.1)
                .save_file()?;
            Some(make(&path).and_then(|text| {
                std::fs::write(&path, text).map_err(|e| spline_fitter::Error::new(e.to_string()))?;
                Ok(path)
            }))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (title, filter);
            let path = PathBuf::from(default);
            Some(make(&path).and_then(|text| {
                crate::web::download(default, &text).map_err(spline_fitter::Error::new)?;
                Ok(path)
            }))
        }
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
        let make = |_: &Path| {
            resample(&fit.curves(), per_side).and_then(|points| airfoil_data_text(&points, Some(name), Some(airfoil.precision.max(6))))
        };
        match self.save_text_as("Save fitted airfoil", &default, ("Airfoil data", &["dat"]), make) {
            None => self.set_status("Save cancelled."),
            Some(Ok(path)) => self.set_status(format!("{SAVED}: {}", path.display())),
            Some(Err(e)) => self.fail(format!("Save failed: {e}")),
        }
    }

    /// File > Save spline.
    pub fn on_save_spline(&mut self) {
        let (Some(fit), Some(airfoil)) = (self.fit.clone(), self.airfoil.clone()) else {
            return self.set_status("No fitted spline to save.");
        };
        let default = format!("{}.deg{}.spline.dat", file_stem(airfoil.file_base(), "airfoil"), fit.degree);
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
        let make = |_: &Path| spline_data_text(&spline, Some(12));
        match self.save_text_as("Save spline control points", &default, ("Spline data", &["spline.dat", "dat"]), make) {
            None => self.set_status("Save cancelled."),
            Some(Ok(path)) => self.set_status(format!("{SAVED}: {}", path.display())),
            Some(Err(e)) => self.fail(format!("Save failed: {e}")),
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
        let default = format!("{}{suffix}", file_stem(airfoil.file_base(), "airfoil"));
        let make = |path: &Path| {
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
            Ok(export::text(path, &curves, choice.scale, units)?)
        };
        match self.save_text_as("Export CAD", &default, (label, &[suffix.trim_start_matches('.')]), make) {
            None => self.set_status("Export cancelled."),
            Some(Ok(path)) => self.set_status(format!("Exported: {}", path.display())),
            Some(Err(e)) => self.fail(format!("Export failed: {e}")),
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
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = self
            .file_dialog()
            .set_title("Open project")
            .set_directory(crate::parent::home())
            .add_filter("Spline project", &["asfproj"])
            .add_filter("All files", &["*"])
            .pick_file()
        {
            self.open_project(&path);
        }
        #[cfg(target_arch = "wasm32")]
        crate::web::pick_file("Open project", "Spline project", &["asfproj"]);
    }

    pub fn open_project(&mut self, path: &Path) -> bool {
        self.open_project_from(ProjectSource::Path(path.to_path_buf()))
    }

    fn open_project_from(&mut self, source: ProjectSource) -> bool {
        if self.fitting() {
            self.set_status("Cancel the running fit to open a file");
            return false;
        }
        let (read, path) = match &source {
            ProjectSource::Path(path) => (load_project(path), path.clone()),
            ProjectSource::Text { name, text } => (parse_project(text), PathBuf::from(name)),
        };
        let path = path.as_path();
        let loaded = match read.and_then(|l| {
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
        let again = source.clone();
        if !self.confirm_discard("Open cancelled.", move |app| {
            app.open_project_from(again);
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
        if matches!(source, ProjectSource::Path(_)) {
            self.store.add_recent(path);
        }
        true
    }

    /// File > Save project.
    pub fn on_save_project(&mut self) {
        let Some(path) = self.projects.path.clone().filter(|_| self.projects.file.is_some()) else {
            return self.on_save_project_as();
        };
        match write_project(self.projects.file.as_mut().unwrap(), &path) {
            Ok(()) => {
                self.projects.modified = false;
                self.set_status(format!("{SAVED}: {}", path.display()));
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
        let default = format!("{}.asfproj", file_stem(airfoil.file_base(), "project"));
        // The browser downloads the file under its default name.
        #[cfg(target_arch = "wasm32")]
        self.save_project_to(PathBuf::from(default));
        #[cfg(not(target_arch = "wasm32"))]
        self.ask_project_path(default);
    }

    /// Asks where to save the project, then saves it there.
    #[cfg(not(target_arch = "wasm32"))]
    fn ask_project_path(&mut self, default: String) {
        let Some(mut path) = self
            .file_dialog()
            .set_title("Save project as")
            .set_directory(crate::parent::home())
            .set_file_name(default)
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
        match write_project(self.projects.file.as_mut().unwrap(), &path) {
            Ok(()) => {
                self.projects.path = Some(path.clone());
                self.projects.modified = false;
                self.store.add_recent(&path);
                self.set_status(format!("{SAVED}: {}", path.display()));
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

/// Saves a project to `path` (the browser downloads it under that name).
fn write_project(project: &mut ProjectFile, path: &Path) -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    return spline_fitter::project::save_project(project, path);
    #[cfg(target_arch = "wasm32")]
    {
        let text = spline_fitter::project::project_text_to_save(project);
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        crate::web::download(&name, &text).map_err(spline_fitter::Error::new)
    }
}
