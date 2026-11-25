//! Embeds the app icon in the Windows executable, so Explorer, the taskbar
//! and the installer's Start menu shortcut show it.

fn main() {
    println!("cargo:rerun-if-changed=assets/app_icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/app_icon.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=the app icon was not embedded: {e}");
        }
    }
}
