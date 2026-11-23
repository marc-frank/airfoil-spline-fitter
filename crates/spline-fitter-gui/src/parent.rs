//! The main window as the owner of the native file pickers: Windows shows
//! an owned dialog over its owner and keeps it in front.

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

impl App {
    /// A native file picker in front of the main window, centred on it.
    pub fn file_dialog(&self) -> FileDialog {
        file_dialog(self.parent.as_ref())
    }
}
