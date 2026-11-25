//! The browser version's file access (packaging/web, GitHub Pages).
//!
//! A web page cannot open or write files by path. The browser's pickers
//! answer later: what they bring lands in an inbox the window empties each
//! frame. In Chrome and Edge the File System Access API (web/fs.js) gives
//! files and folders that can be written back and reopened: the app knows
//! them by the path `fsa/<id>/<name>`. Elsewhere opening uses the file
//! picker and saving hands the text to the browser as a download.

use std::cell::RefCell;

use wasm_bindgen::{JsCast, JsValue};

/// What arrived for the window.
pub enum Incoming {
    /// A chosen or dropped file; `path` for one that can be written back.
    File { name: String, bytes: Vec<u8>, path: Option<String> },
    /// The clipboard's text.
    Text(String),
    /// A save finished; `project` when it was the project's file.
    Saved { path: String, project: bool },
    /// A status line (a cancelled save).
    Status(String),
    /// Something failed.
    Failed(String),
    /// A batch folder was chosen (`target`: the target folder).
    Folder { target: bool, path: String },
    /// A batch's source files and the names already in its target folder.
    Batch { sources: Vec<(String, Vec<u8>)>, existing: Vec<String> },
}

/// The path of a file or folder picked through web/fs.js.
fn fsa_path(id: &str, name: &str) -> String {
    format!("fsa/{id}/{name}")
}

/// The id in such a path.
pub fn fsa_id(path: &str) -> Option<&str> {
    path.strip_prefix("fsa/")?.split('/').next()
}

/// Whether files and folders can be written back (Chrome, Edge).
pub fn files_supported() -> bool {
    call_sync("supported", &[]).and_then(|v| v.as_bool()).unwrap_or(false)
}

fn files() -> Option<JsValue> {
    js_sys::Reflect::get(&web_sys::window()?.into(), &"splineFitterFiles".into()).ok().filter(|v| v.is_object())
}

fn call_sync(name: &str, args: &[JsValue]) -> Option<JsValue> {
    let files = files()?;
    let f: js_sys::Function = js_sys::Reflect::get(&files, &name.into()).ok()?.dyn_into().ok()?;
    f.apply(&files, &args.iter().collect::<js_sys::Array>()).ok()
}

/// Calls a function of web/fs.js and waits for its promise.
async fn call(name: &str, args: Vec<JsValue>) -> Result<JsValue, String> {
    let promise = call_sync(name, &args).ok_or_else(|| format!("the browser cannot {name}"))?;
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&promise))
        .await
        .map_err(|e| e.as_string().unwrap_or_else(|| format!("{e:?}")))
}

fn field(value: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(value, &key.into()).unwrap_or(JsValue::UNDEFINED)
}

fn text_field(value: &JsValue, key: &str) -> String {
    field(value, key).as_string().unwrap_or_default()
}

fn bytes(value: &JsValue) -> Vec<u8> {
    js_sys::Uint8Array::new(value).to_vec()
}

fn extensions(list: &[&str]) -> JsValue {
    list.iter().map(|e| JsValue::from_str(e)).collect::<js_sys::Array>().into()
}

/// A picked file `{ id, name, bytes }` as it arrives.
fn deliver_file(value: &JsValue) {
    let (id, name) = (text_field(value, "id"), text_field(value, "name"));
    let path = Some(fsa_path(&id, &name));
    deliver(Incoming::File { name, bytes: bytes(&field(value, "bytes")), path });
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
    repaint();
}

/// Draws the window again (something arrived for it).
pub fn repaint() {
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

/// Asks the browser's file picker for a file with one of `exts`.
pub fn pick_file(title: &str, filter: &str, exts: &[&str]) {
    if files_supported() {
        let args = vec![JsValue::from_str(filter), extensions(exts)];
        wasm_bindgen_futures::spawn_local(async move {
            match call("open", args).await {
                Ok(v) if v.is_object() => deliver_file(&v),
                Ok(_) => {}
                Err(e) => deliver(Incoming::Failed(format!("Open failed: {e}"))),
            }
        });
        return;
    }
    let dialog = rfd::AsyncFileDialog::new().set_title(title).add_filter(filter, exts);
    wasm_bindgen_futures::spawn_local(async move {
        if let Some(file) = dialog.pick_file().await {
            let bytes = file.read().await;
            deliver(Incoming::File { name: file.file_name(), bytes, path: None });
        }
    });
}

/// Opens a recent file again (the browser may ask for permission).
pub fn reopen(path: &str) {
    let Some(id) = fsa_id(path) else { return };
    let (args, path) = (vec![JsValue::from_str(id)], path.to_string());
    wasm_bindgen_futures::spawn_local(async move {
        match call("reopen", args).await {
            Ok(v) if v.is_object() => deliver_file(&v),
            _ => deliver(Incoming::Failed(format!("File not found: {}", name_of(&path)))),
        }
    });
}

/// The file name of a path.
pub fn name_of(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

/// How a save goes.
pub enum Saving {
    /// The browser asks where; the answer arrives later.
    Asked,
    /// Handed over as a download.
    Downloaded,
}

/// Saves `text`: asks for the place where the browser can (the answer
/// arrives as `Saved`, `Status` or `Failed`), else downloads `name`.
pub fn save_as(name: &str, filter: &str, exts: &[&str], text: &str, project: bool) -> Result<Saving, String> {
    if !files_supported() {
        return download(name, text).map(|()| Saving::Downloaded);
    }
    let args = vec![JsValue::from_str(name), JsValue::from_str(filter), extensions(exts), JsValue::from_str(text)];
    wasm_bindgen_futures::spawn_local(async move {
        deliver(match call("saveAs", args).await {
            Ok(v) if v.is_object() => Incoming::Saved { path: fsa_path(&text_field(&v, "id"), &text_field(&v, "name")), project },
            Ok(_) => Incoming::Status("Save cancelled.".into()),
            Err(e) => Incoming::Failed(format!("Save failed: {e}")),
        });
    });
    Ok(Saving::Asked)
}

/// Writes `text` over the file at `path` (picked before); the answer
/// arrives as `Saved` or `Failed`.
pub fn save_in_place(path: &str, text: &str, project: bool) {
    let Some(id) = fsa_id(path) else { return };
    let (args, path) = (vec![JsValue::from_str(id), JsValue::from_str(text)], path.to_string());
    wasm_bindgen_futures::spawn_local(async move {
        deliver(match call("save", args).await {
            Ok(v) if v.as_bool() == Some(true) => Incoming::Saved { path, project },
            Ok(_) => Incoming::Failed(format!("Save failed: the browser did not allow writing {}", name_of(&path))),
            Err(e) => Incoming::Failed(format!("Save failed: {e}")),
        });
    });
}

/// Asks for a batch folder; the answer arrives as `Folder`.
pub fn pick_folder(target: bool) {
    wasm_bindgen_futures::spawn_local(async move {
        match call("pickFolder", vec![]).await {
            Ok(v) if v.is_object() => {
                deliver(Incoming::Folder { target, path: fsa_path(&text_field(&v, "id"), &text_field(&v, "name")) })
            }
            Ok(_) => {}
            Err(e) => deliver(Incoming::Failed(format!("Choosing the folder failed: {e}"))),
        }
    });
}

/// Reads a batch's source files and the target's names; the answer
/// arrives as `Batch` or `Failed`.
pub fn read_batch(source: &str, target: &str) {
    let (Some(s), Some(t)) = (fsa_id(source), fsa_id(target)) else { return };
    let args = vec![JsValue::from_str(s), JsValue::from_str(t)];
    wasm_bindgen_futures::spawn_local(async move {
        deliver(match call("readBatch", args).await {
            Ok(v) => {
                let sources = js_sys::Array::from(&field(&v, "sources"))
                    .iter()
                    .map(|pair| {
                        let pair = js_sys::Array::from(&pair);
                        (pair.get(0).as_string().unwrap_or_default(), bytes(&pair.get(1)))
                    })
                    .collect();
                let existing = js_sys::Array::from(&field(&v, "existing")).iter().filter_map(|n| n.as_string()).collect();
                Incoming::Batch { sources, existing }
            }
            Err(e) => Incoming::Failed(format!("Batch failed: {e}")),
        });
    });
}

/// Writes a batch result into the target folder; a failure arrives as
/// `Failed`.
pub fn write_in_folder(folder: &str, name: &str, text: &str) {
    let Some(id) = fsa_id(folder) else { return };
    let args = vec![JsValue::from_str(id), JsValue::from_str(name), JsValue::from_str(text)];
    let name = name.to_string();
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(e) = call("writeInFolder", args).await {
            deliver(Incoming::Failed(format!("Writing {name} failed: {e}")));
        }
    });
}

/// Reads a file dropped on the page.
pub fn read_dropped(file: eframe::egui::DroppedFileHandle) {
    wasm_bindgen_futures::spawn_local(async move {
        let name = file.path().to_string_lossy().into_owned();
        if let Ok(bytes) = file.bytes_async().await {
            deliver(Incoming::File { name, bytes, path: None });
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

/// Whether leaving the page loses unsaved runs: the page then asks first
/// (index.html listens for `beforeunload`).
pub fn set_unsaved(unsaved: bool) {
    thread_local! {
        static LAST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }
    if LAST.with(|l| l.replace(Some(unsaved))) == Some(unsaved) {
        return;
    }
    if let Some(window) = web_sys::window() {
        let _ = js_sys::Reflect::set(&window, &"splineFitterUnsaved".into(), &unsaved.into());
    }
}
