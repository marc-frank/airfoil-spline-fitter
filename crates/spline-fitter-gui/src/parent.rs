//! The main window as the owner of the native file pickers: Windows shows
//! an owned dialog over its owner and keeps it in front. The browser has no
//! native window, folder picker or clipboard access by name: there the
//! functions below do without.

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;
#[cfg(target_arch = "wasm32")]
pub use web::*;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::PathBuf;

    use raw_window_handle::{
        DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, WindowHandle,
    };
    use rfd::FileDialog;

    use crate::app::App;

    /// The main window's handles, taken when the app starts. The window lives
    /// as long as the app, so the handles stay valid.
    pub struct ParentWindow {
        window: RawWindowHandle,
        display: RawDisplayHandle,
    }

    impl ParentWindow {
        /// The main window's handles, or none where there is no native window.
        pub fn of(cc: &eframe::CreationContext<'_>) -> Option<Self> {
            let window = cc.window_handle().ok()?.as_raw();
            let display = cc.display_handle().ok()?.as_raw();
            Some(ParentWindow { window, display })
        }
    }

    impl HasWindowHandle for ParentWindow {
        fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
            // SAFETY: the main window outlives the app that holds this handle.
            Ok(unsafe { WindowHandle::borrow_raw(self.window) })
        }
    }

    impl HasDisplayHandle for ParentWindow {
        fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
            // SAFETY: as above, the display outlives the app.
            Ok(unsafe { DisplayHandle::borrow_raw(self.display) })
        }
    }

    /// A native file picker owned by the main window, if there is one.
    pub fn file_dialog(parent: Option<&ParentWindow>) -> FileDialog {
        match parent {
            Some(parent) => FileDialog::new().set_parent(parent),
            None => FileDialog::new(),
        }
    }

    /// The user's home folder, where the file pickers start.
    pub fn home() -> PathBuf {
        dirs::home_dir().unwrap_or_default()
    }

    /// A folder chosen in the native picker.
    pub fn pick_folder(parent: Option<&ParentWindow>, title: &str) -> Option<PathBuf> {
        file_dialog(parent).set_title(title).set_directory(home()).pick_folder()
    }

    /// Puts `text` on the clipboard.
    pub fn copy_text(_ctx: &eframe::egui::Context, text: String) -> Result<(), String> {
        arboard::Clipboard::new().and_then(|mut c| c.set_text(text)).map_err(|e| e.to_string())
    }

    impl App {
        /// A native file picker in front of the main window, centred on it.
        pub fn file_dialog(&self) -> FileDialog {
            file_dialog(self.parent.as_ref())
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use std::path::PathBuf;

    /// No native window in the browser.
    pub struct ParentWindow;

    impl ParentWindow {
        pub fn of(_cc: &eframe::CreationContext<'_>) -> Option<Self> {
            None
        }
    }

    /// Folders are out of a web page's reach.
    pub fn pick_folder(_parent: Option<&ParentWindow>, _title: &str) -> Option<PathBuf> {
        None
    }

    /// Puts `text` on the clipboard (through the page).
    pub fn copy_text(ctx: &eframe::egui::Context, text: String) -> Result<(), String> {
        ctx.copy_text(text);
        Ok(())
    }
}
