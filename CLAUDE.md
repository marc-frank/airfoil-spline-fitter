# Spline Fitter

## GUI design language

The GUI follows the **Loft** design language: read
[docs/design-language.md](docs/design-language.md) before adding or changing
anything in `src/spline_fitter/gui/`.

- All colors, fonts, sizes and stylesheets live in `src/spline_fitter/gui/theme.py`.
  Other modules use its tokens (`Palette`, `Ink`, `Z`, `Size`, `Type`) and helpers
  (`style_plot`, `label_plot`, `pen`, `brush`, `text_browser`), and style widgets by
  object name (`PrimaryButton`, `SectionCaption`, `Muted`, `Readout`, ...), never with
  `setStyleSheet`, hex colors or RGB tuples.
- Orange ink is reserved for the fitted curve, the one primary button and the
  current-run marker. Selection is blue.
- Labels: sentence case, no trailing colons; captions uppercase; no emoji.
- `tests/test_design_language.py` enforces this; run it after GUI changes.

## Running

```bash
python -m pytest
python -m src.spline_fitter
```
