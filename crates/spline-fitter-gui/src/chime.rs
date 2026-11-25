//! Short sounds at the end of a fit or on an error.

/// A finished fit: two rising notes.
pub fn completion() {
    play(&[(880.0, 0.09), (1318.5, 0.16)]);
}

/// A failure: two falling notes.
pub fn error() {
    play(&[(392.0, 0.11), (293.7, 0.2)]);
}

/// Plays notes on a background thread; any audio failure is ignored.
#[cfg(not(target_arch = "wasm32"))]
fn play(notes: &'static [(f32, f32)]) {
    use std::time::Duration;

    use rodio::source::{SineWave, Source};

    let _ = std::thread::Builder::new().name("chime".into()).spawn(move || {
        let Ok(sink) = rodio::DeviceSinkBuilder::open_default_sink() else { return };
        let player = rodio::Player::connect_new(sink.mixer());
        for &(frequency, seconds) in notes {
            let note =
                SineWave::new(frequency).take_duration(Duration::from_secs_f32(seconds)).fade_in(Duration::from_millis(8)).amplify(0.15);
            player.append(note);
        }
        player.sleep_until_end();
    });
}

/// The browser: the same notes through the page's audio (Web Audio). The
/// browser lets a page sound only after the user clicked or typed in it,
/// which a fit or an error always follows; any failure is ignored.
#[cfg(target_arch = "wasm32")]
fn play(notes: &'static [(f32, f32)]) {
    use web_sys::{AudioContext, OscillatorType};

    thread_local! {
        static AUDIO: Option<AudioContext> = AudioContext::new().ok();
    }
    AUDIO.with(|audio| {
        let Some(audio) = audio else { return };
        let _ = audio.resume();
        let mut at = audio.current_time();
        for &(frequency, seconds) in notes {
            let (frequency, seconds) = (frequency as f64, seconds as f64);
            let note = || -> Result<(), wasm_bindgen::JsValue> {
                let oscillator = audio.create_oscillator()?;
                oscillator.set_type(OscillatorType::Sine);
                oscillator.frequency().set_value(frequency as f32);
                let gain = audio.create_gain()?;
                // 8 ms fade in to 0.15, as the desktop's notes.
                gain.gain().set_value_at_time(0.0, at)?;
                gain.gain().linear_ramp_to_value_at_time(0.15, at + 0.008)?;
                gain.gain().set_value_at_time(0.15, at + seconds - 0.004)?;
                gain.gain().linear_ramp_to_value_at_time(0.0, at + seconds)?;
                oscillator.connect_with_audio_node(&gain)?;
                gain.connect_with_audio_node(&audio.destination())?;
                oscillator.start_with_when(at)?;
                oscillator.stop_with_when(at + seconds)?;
                Ok(())
            };
            let _ = note();
            at += seconds;
        }
    });
}
