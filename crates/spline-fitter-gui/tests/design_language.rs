//! The rules of the Loft design language (docs/design-language.md) that the
//! sources can be checked for.

use std::fs;
use std::path::PathBuf;

/// Every GUI source file with its name, tests left out.
fn sources() -> Vec<(String, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files: Vec<(String, String)> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .map(|p| {
            let text = fs::read_to_string(&p).unwrap();
            let code = text.split("#[cfg(test)]").next().unwrap().to_string();
            (p.file_name().unwrap().to_string_lossy().into_owned(), code)
        })
        .filter(|(name, _)| name != "ui_tests.rs")
        .collect();
    files.sort();
    files
}

/// The string literals of a source (plain `"..."`, escapes kept).
fn literals(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = code.chars().peekable();
    let mut previous = ' ';
    while let Some(c) = chars.next() {
        if c == '"' && previous != '\'' && previous != '\\' {
            let mut s = String::new();
            let mut escaped = false;
            for d in chars.by_ref() {
                if escaped {
                    s.push(d);
                    escaped = false;
                } else if d == '\\' {
                    s.push(d);
                    escaped = true;
                } else if d == '"' {
                    break;
                } else {
                    s.push(d);
                }
            }
            out.push(s);
        }
        previous = c;
    }
    out
}

#[test]
fn colors_come_from_the_theme() {
    let constructors = [
        "Color32::from_rgb",
        "Color32::from_rgba",
        "Color32::from_gray",
        "Color32::WHITE",
        "Color32::BLACK",
        "Color32::RED",
        "Color32::GRAY",
    ];
    for (name, code) in sources() {
        if name == "theme.rs" {
            continue;
        }
        for c in constructors {
            assert!(!code.contains(c), "{name} makes its own color with {c}");
        }
        for lit in literals(&code) {
            let hex = lit.starts_with('#') && lit.len() == 7 && lit[1..].chars().all(|c| c.is_ascii_hexdigit());
            assert!(!hex, "{name} writes the color {lit}");
        }
    }
}

#[test]
fn ink_only_where_allowed() {
    // The fitted curve and the current-run marker (plots), the primary button (widgets).
    for (name, code) in sources() {
        let uses_ink = code.contains("INK") && !code.contains("INK_");
        if uses_ink || code.contains(" INK,") || code.contains("(INK") {
            assert!(["theme.rs", "plots.rs", "widgets.rs"].contains(&name.as_str()), "{name} uses the ink");
        }
    }
}

#[test]
fn labels_follow_the_wording_rules() {
    let abbreviations = [" pts", "Ctrl pts", " num ", " cfg", " opt "];
    for (name, code) in sources() {
        for lit in literals(&code) {
            if lit.contains('{') || lit.len() < 2 || lit.starts_with('/') || lit.contains("://") {
                continue;
            }
            let label = lit.trim_end_matches("\\n");
            let looks_like_text = label.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && label.contains(char::is_whitespace);
            if looks_like_text {
                assert!(!label.ends_with(':'), "{name}: label with a trailing colon: {lit:?}");
                for a in abbreviations {
                    assert!(!label.contains(a), "{name}: abbreviation in {lit:?}");
                }
            }
            assert!(!lit.chars().any(|c| matches!(c as u32, 0x1F300..=0x1FAFF | 0x2600..=0x27BF)), "{name}: emoji in {lit:?}");
            assert!(!lit.contains('·') && !lit.contains('•'), "{name}: bullet or middle dot in {lit:?}");
        }
    }
}

#[test]
fn every_control_is_control_height() {
    // Heights of controls come from the token, never a number.
    for (name, code) in sources() {
        for bad in ["min_size(egui::vec2(0.0, 24", "min_size(egui::vec2(0.0, 26", "min_size(egui::vec2(0.0, 28"] {
            assert!(!code.contains(bad), "{name} sets a control height by number");
        }
    }
}
