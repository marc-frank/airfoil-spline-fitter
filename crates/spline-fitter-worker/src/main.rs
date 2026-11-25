//! The browser version's fitting worker (a web worker): fits on a thread of
//! its own, so the page stays responsive while it runs.
//!
//! It says `ready` once it listens, then takes fit requests and answers each
//! with its updates and its result, in the messages of
//! `spline_fitter::project::wire`. The page stops a fit by ending the worker.

#[cfg(not(target_arch = "wasm32"))]
fn main() {}

#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::prelude::*;
    use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

    let scope: DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let poster = scope.clone();
    let onmessage = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
        let text = event.data().as_string().unwrap_or_default();
        let mut monitor = Poster { scope: &poster, last_points: None };
        let result = spline_fitter::project::wire::parse_request(&text)
            .and_then(|(config, options, airfoil)| spline_fitter::optimize::run_optimization(&config, &options, &airfoil, &mut monitor));
        let _ = poster.post_message(&JsValue::from_str(&spline_fitter::project::wire::done(&result)));
    });
    scope.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    onmessage.forget();
    let _ = scope.post_message(&JsValue::from_str("ready"));
}

/// Posts every update. The curves go along at most every 30 ms (and with
/// every refit): the page draws at most that often, and the error history
/// needs only the numbers.
#[cfg(target_arch = "wasm32")]
struct Poster<'a> {
    scope: &'a web_sys::DedicatedWorkerGlobalScope,
    last_points: Option<web_time::Instant>,
}

#[cfg(target_arch = "wasm32")]
impl spline_fitter::optimize::Monitor for Poster<'_> {
    fn update(&mut self, u: &spline_fitter::optimize::Update) -> bool {
        use spline_fitter::optimize::{Update, UpdateKind};
        let now = web_time::Instant::now();
        let curves = u.kind == UpdateKind::Refit || self.last_points.is_none_or(|t| (now - t).as_millis() >= 30);
        let text = if curves {
            self.last_points = Some(now);
            spline_fitter::project::wire::update(u)
        } else {
            spline_fitter::project::wire::update(&Update { control_points: &[], knots: &[], ..*u })
        };
        let _ = self.scope.post_message(&wasm_bindgen::JsValue::from_str(&text));
        false
    }
}
