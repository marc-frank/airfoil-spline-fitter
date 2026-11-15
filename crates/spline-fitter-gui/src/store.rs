//! Settings kept between sessions: recent files, dialog choices, theme and
//! motion. Nothing else persists; the sidebar starts at its defaults.

use std::path::PathBuf;

use serde_json::{json, Map, Value};

/// A small JSON file in the user's configuration folder.
pub struct Store {
    path: Option<PathBuf>,
    values: Map<String, Value>,
}

impl Store {
    pub fn open() -> Self {
        if cfg!(test) {
            return Store { path: None, values: Map::new() };
        }
        let path = dirs::config_dir().map(|d| d.join("AirfoilSplineFitter").join("settings.json"));
        let values = path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        Store { path, values }
    }

    fn save(&self) {
        let Some(path) = &self.path else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(path, Value::Object(self.values.clone()).to_string());
    }

    fn set(&mut self, key: &str, value: Value) {
        self.values.insert(key.into(), value);
        self.save();
    }

    pub fn string(&self, key: &str, default: &str) -> String {
        self.values.get(key).and_then(Value::as_str).unwrap_or(default).to_string()
    }

    pub fn set_string(&mut self, key: &str, value: &str) {
        self.set(key, json!(value));
    }

    pub fn int(&self, key: &str, default: i64) -> i64 {
        self.values.get(key).and_then(Value::as_i64).unwrap_or(default)
    }

    pub fn set_int(&mut self, key: &str, value: i64) {
        self.set(key, json!(value));
    }

    pub fn float(&self, key: &str, default: f64) -> f64 {
        self.values.get(key).and_then(Value::as_f64).unwrap_or(default)
    }

    pub fn set_float(&mut self, key: &str, value: f64) {
        self.set(key, json!(value));
    }

    pub fn bool(&self, key: &str, default: bool) -> bool {
        self.values.get(key).and_then(Value::as_bool).unwrap_or(default)
    }

    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.set(key, json!(value));
    }

    /// The recent files, newest first.
    pub fn recent_files(&self) -> Vec<String> {
        match self.values.get("recent_files") {
            Some(Value::Array(items)) => items.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
            Some(Value::String(s)) => vec![s.clone()],
            _ => vec![],
        }
    }

    pub fn set_recent_files(&mut self, files: &[String]) {
        self.set("recent_files", json!(files));
    }

    /// Puts a path at the top of the recent files (at most ten).
    pub fn add_recent(&mut self, path: &std::path::Path) {
        let full = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()).to_string_lossy().into_owned();
        let key = full.to_lowercase();
        let mut files: Vec<String> = self.recent_files().into_iter().filter(|f| f.to_lowercase() != key).collect();
        files.insert(0, full);
        files.truncate(10);
        self.set_recent_files(&files);
    }

    pub fn remove_recent(&mut self, path: &str) {
        let key = path.to_lowercase();
        let files: Vec<String> = self.recent_files().into_iter().filter(|f| f.to_lowercase() != key).collect();
        self.set_recent_files(&files);
    }
}
