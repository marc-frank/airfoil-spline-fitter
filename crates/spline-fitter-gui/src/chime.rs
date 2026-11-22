//! Short sounds at the end of a fit or on an error.

use std::time::Duration;

use rodio::source::{SineWave, Source};

/// A finished fit: two rising notes.
pub fn completion() {
    play(&[(880.0, 0.09), (1318.5, 0.16)]);
}

/// A failure: two falling notes.
pub fn error() {
    play(&[(392.0, 0.11), (293.7, 0.2)]);
}

/// Plays notes on a background thread; any audio failure is ignored.
fn play(notes: &'static [(f32, f32)]) {
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
