//! The browser version's file access (packaging/web, GitHub Pages).
//!
//! A web page cannot open or write files by path. Opening asks the browser's
//! file picker, which answers later: the chosen file lands in an inbox the
//! window empties each frame. Saving hands the text to the browser as a
//! download.

use std::cell::RefCell;

use wasm_bindgen::JsCast;

/// What arrived for the window: a chosen file or the clipboard's text.
pub enum Incoming {
    File { name: String, bytes: Vec<u8> },
    Text(String),
}

thread_local! {
    static INBOX: RefCell<Vec<Incoming>> = const { RefCell::new(Vec::new()) };
    static CONTEXT: RefCell<Option<eframe::egui::Context>> = const { RefCell::new(None) };
}

/// Remembers the window, so an answer arriving later repaints it.
pub fn set_context(ctx: &eframe::egui::Context) {
    CONTEXT.with(|c| *c.borrow_mut() = Some(ctx.clone()));
}

fn deliver(item: Incoming) {
    INBOX.with(|i| i.borrow_mut().push(item));
    CONTEXT.with(|c| {
        if let Some(ctx) = &*c.borrow() {
            ctx.request_repaint();
        }
    });
}

/// Everything that arrived since the last call.
pub fn take() -> Vec<Incoming> {
    INBOX.with(|i| std::mem::take(&mut *i.borrow_mut()))
}

/// Asks the browser's file picker for a file with one of `extensions`.
pub fn pick_file(title: &str, filter: &str, extensions: &[&str]) {
    let dialog = rfd::AsyncFileDialog::new().set_title(title).add_filter(filter, extensions);
    wasm_bindgen_futures::spawn_local(async move {
        if let Some(file) = dialog.pick_file().await {
            let bytes = file.read().await;
            deliver(Incoming::File { name: file.file_name(), bytes });
        }
    });
}

/// Reads a file dropped on the page.
pub fn read_dropped(file: eframe::egui::DroppedFileHandle) {
    wasm_bindgen_futures::spawn_local(async move {
        let name = file.path().to_string_lossy().into_owned();
        if let Ok(bytes) = file.bytes_async().await {
            deliver(Incoming::File { name, bytes });
        }
    });
}

/// Reads the clipboard's text (the browser may ask the user first).
pub fn read_clipboard() {
    let Some(window) = web_sys::window() else { return };
    let promise = window.navigator().clipboard().read_text();
    wasm_bindgen_futures::spawn_local(async move {
        let text = wasm_bindgen_futures::JsFuture::from(promise).await.ok().and_then(|v| v.as_string()).unwrap_or_default();
        deliver(Incoming::Text(text));
    });
}

/// Hands `text` to the browser as the download `name`.
pub fn download(name: &str, text: &str) -> Result<(), String> {
    let error = |e: wasm_bindgen::JsValue| e.as_string().unwrap_or_else(|| "the browser refused the download".into());
    let window = web_sys::window().ok_or("no browser window")?;
    let document = window.document().ok_or("no document")?;
    let parts = js_sys::Array::of1(&wasm_bindgen::JsValue::from_str(text));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("application/octet-stream");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options).map_err(error)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(error)?;
    let anchor: web_sys::HtmlAnchorElement = document.create_element("a").map_err(error)?.unchecked_into();
    anchor.set_href(&url);
    anchor.set_download(name);
    anchor.click();
    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(())
}

/// The browser's stored settings (the page's local storage).
pub fn load_settings(key: &str) -> Option<String> {
    web_sys::window()?.local_storage().ok()??.get_item(key).ok()?
}

pub fn save_settings(key: &str, value: &str) {
    if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = storage.set_item(key, value);
    }
}

/// The browser's language, as `de-DE`.
pub fn language() -> String {
    web_sys::window().and_then(|w| w.navigator().language()).unwrap_or_default()
}
