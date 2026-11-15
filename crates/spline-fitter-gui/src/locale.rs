//! The decimal point of the system locale.

/// Languages that write a decimal comma.
const COMMA_LANGUAGES: &[&str] = &[
    "bg", "ca", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hr", "hu", "id", "it", "lt", "lv", "nb", "nl", "nn", "no", "pl", "pt",
    "ro", "ru", "sk", "sl", "sr", "sv", "tr", "uk", "vi",
];

/// `,` for locales that write a decimal comma, else `.`.
pub fn decimal_point() -> char {
    let locale = sys_locale::get_locale().unwrap_or_default().to_lowercase();
    let language = locale.split(['-', '_']).next().unwrap_or("");
    let swiss = locale.ends_with("-ch") && language == "de";
    if COMMA_LANGUAGES.contains(&language) && !swiss { ',' } else { '.' }
}
