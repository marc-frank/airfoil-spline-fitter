# Spline Fitter

A Rust workspace: `crates/bspline-varpro` (fitting library), `crates/spline-fitter`
(application core), `crates/spline-fitter-gui` (egui desktop application).
[docs/spec.md](docs/spec.md) specifies the behaviour; README.md lists where the
Rust version deliberately differs from it.

## GUI design language

The GUI follows the **Loft** design language: read
[docs/design-language.md](docs/design-language.md) before adding or changing
anything in `crates/spline-fitter-gui/src/`.

- All colors, fonts and sizes live in `crates/spline-fitter-gui/src/theme.rs`
  (`Palette`, `size`, `typeface`, `Glyph`). Other modules take colors from the
  `Palette` they are given and never write `Color32::from_rgb` or hex colors.
- Orange ink (`theme::INK`) is reserved for the fitted curve, the one primary
  button and the current-run marker. The chrome is grey; selection is a
  lighter grey. Every input and button is `size::CONTROL_HEIGHT` tall.
- The user chooses light/dark/system and a cool-to-warm temperature (View >
  Theme...), applied live. Greys are lightness values in `GREY_LIGHTNESS`,
  never hex; read the palette at draw time.
- Labels: sentence case, spelled out (no "pts"), no trailing colons; captions
  uppercase; no emoji.
- `crates/spline-fitter-gui/tests/design_language.rs` checks this.

## Running

```bash
cargo test --workspace
cargo run --release
```

GUI tests drive the real window headlessly (`src/ui_tests.rs`, egui_kittest);
set `SPLINE_FITTER_SHOTS` to a folder to save screenshots of each step.
