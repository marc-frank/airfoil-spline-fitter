# Loft: the Spline Fitter design language

Before the computer, aircraft and ship lines were drawn full size on a
**lofting floor**. A long, flexible strip of wood, the *spline*, was bent
through the measured stations and held in place by lead weights called
*ducks*. The shape the strip settled into was the fair curve. That is
exactly what this application does, so the interface borrows the floor's
look and vocabulary.

> The window is the lofting floor: graphite at night or paper by day, in
> greys you can tint cool or warm, construction lines in pencil, and one ink
> (the orange of the app icon) for the curve and for the one action that
> makes it.

Everything below is implemented in
[`crates/spline-fitter-gui/src/theme.rs`](../crates/spline-fitter-gui/src/theme.rs).
That module is the **only** place colors, fonts and sizes are defined.
[`tests/design_language.rs`](../crates/spline-fitter-gui/tests/design_language.rs)
fails if another GUI module sets its own.

---

## 1. Principles

1. **The curve is the hero.** Orange ink is spent on the fitted curve and
   on almost nothing else, so the eye always lands on the result.
2. **Measured, constructed, derived.** Every plot item belongs to one of
   three kinds and is drawn like it:
   - *measured* (data stations): chalk dots, the plain truth;
   - *constructed* (control polygon, knots): drafting marks, thin, dashed
     or small;
   - *derived* (curvature comb, convergence): colored by meaning, kept
     quieter than the curve.
3. **Engineering readout, not dashboard.** Numbers are monospaced and
   aligned, labels are plain and short, and there is no decoration that
   carries no information.
4. **Hairlines, not boxes.** Regions are separated by 1 px rules and steps
   in surface lightness, never by heavy frames or drop shadows.
5. **One theme, owned by the app.** The look does not depend on the OS
   style: `theme::apply` sets every widget color and size. Only light or
   dark follows the OS, and only while the user leaves it so.

---

## 2. Color

### Theme setting (View > Theme...)

The user picks two things; both are saved with the other session settings
(`store.rs`) and applied live:

- **Mode**: system (the default: follows the OS, also when it switches),
  light or dark.
- **Temperature**: -100 (cool, blue) .. 0 (neutral, the default) .. +100
  (warm, sand).

The greys are therefore **not fixed hex values**. Each is a lightness in
`GREY_LIGHTNESS` (one for dark, one for light), and `grey(lightness,
temperature)` turns it into a color in OKLab, so a temperature change keeps
every contrast. At temperature 0 the greys have no hue at all. Colors that
must differ between light and dark but are not greys (status, knots, comb)
are pairs in `Palette::new`. Never write a grey as a hex value anywhere: add
a lightness pair to `GREY_LIGHTNESS`.

### UI tokens (`Palette`)

Dark mode: surfaces get lighter the closer they are to the hand. Light
mode: paper-white canvas and fields on slightly darker panels. Lightness
values are OKLab (dark / light).

| Token | Lightness | Use |
|---|---|---|
| `canvas` | 0.213 / 0.995 | Plot area: the drafting film |
| `window` | 0.236 / 0.970 | Window, toolbars, run bar, status bar |
| `panel` | 0.268 / 0.950 | Sidebar, project explorer |
| `field` | 0.309 / 1.000 | Inputs, secondary buttons, menus, popups |
| `field_hover` | 0.344 / 0.965 | Hovered field or button |
| `rule` | 0.360 / 0.870 | Hairlines and borders |
| `rule_strong` | 0.434 / 0.780 | Borders of hovered fields, popups |
| `text` | 0.982 / 0.250 | Values, primary text |
| `text_2` | 0.807 / 0.430 | Field labels, axis text, status bar, glyphs, badges |
| `text_3` | 0.686 / 0.580 | Captions, hints, disabled text |
| `select` | 0.405 / 0.895 | Checked toggles, selected rows, menu highlight |
| `select_line` | 0.609 / 0.600 | Focus and checked borders |

Fixed colors, the same in both modes:

| Token | Hex | Use |
|---|---|---|
| `INK` | `#FF6E10` | The icon's orange: fitted curve, primary button |
| `INK_HOVER` / `INK_PRESSED` | `#FF8636` / `#E05E08` | Primary button states |
| `ON_INK` | `#1A0C02` | Text on ink |
| `ok` / `warn` / `error` | dark `#4FBF85` / `#F2B84B` / `#F0605D`, light `#1A7A4A` / `#8F5E00` / `#C02F2B` | Status only |

**Rules**

- **Ink is scarce.** Allowed uses: the fitted curve, the primary button,
  the marker for the current run in a trace. A new feature that wants
  orange needs a reason as strong as those.
- **Selection is a lighter grey, not a color.** Checked toggles, segmented
  controls, selected rows and focus rings use `select` / `select_line`. The
  chrome carries no hue at all; color is kept for the plots, the primary
  action and status.
- **Status colors are for status.** Use `ok`/`warn`/`error` only to say
  that something is fine, doubtful or failed, never to decorate.
- **Disabled is `text_3`.** Disabled things are not hidden.
- **Everything follows a theme change.** egui draws every frame from the
  current `Palette`: widgets take it from the style `theme::apply` sets, and
  plots, glyphs and readouts read it when they draw. Never keep a copy of a
  token color beyond one frame.
- **Readable at every setting.** `text` / `text_2` / `text_3` keep their
  contrast on every surface in both modes and at both temperature extremes.

### Plot tokens

Named by what they draw, never by hue.

| Token | Drawn as | Meaning |
|---|---|---|
| `data()` | chalk (or graphite) dots in `text`, `size::DATA_SIZE` | Measured stations |
| `INK` | solid line, `size::CURVE_WIDTH` | The fitted spline (the batten) |
| `construction` | dashed pencil grey line, hollow squares | Control polygon and control points (CAD convention: squares) |
| `knots` | sulfur diamonds with a canvas outline | Knots (the ducks that hold the batten) |
| `comb_even` | sky blue | Comb on the first, third, ... spline section (a single section is all sky blue) |
| `comb_odd` | rose | Comb on the second, fourth, ... spline section |
| `deviation(outside)` | the comb's colors by side: rose outside the curve, sky blue inside; faint whiskers drawn outward from the curve, an envelope through their tips with a dot at each tip | Size of each data point's deviation from the curve, magnified along the outward normal; the tolerance is one faint dashed `construction` line outside the curve, at the height setting |
| `trace()` | light grey dots or line | Convergence and comparison traces |
| `INK` marker | ink dot with a canvas ring | The current run in a trace |

The comb alternates cool and warm by spline section, so each section's
extent reads at a glance, and the deviation follows it. Its teeth are faint
(`TEETH_ALPHA`) because the envelope through their tips is what you read. A
tooth's length is in chords, for the quantity measured on the airfoil scaled
to a unit chord, so the comb looks the same whatever the size of the data.

### Drawing order

Comb and deviation < construction < data < curve < knots. The curve covers
the data it fits, so a data point that shows beside the curve is a deviation
worth seeing. Every new plot item takes its place in this order in its
plot's `show`.

---

## 3. Type

- **UI text**: the system face (Segoe UI when installed) at
  `typeface::BASE` (9 pt).
- **Numbers**: monospace (Cascadia Mono or Consolas when installed).
  Numeric fields and result readouts (`typeface::READOUT`) use it, so values
  line up and digits do not jump while a fit updates; label and value lists
  use `widgets::readout_list`. Every number shown (fields, readouts,
  messages, reports, plot ticks) takes the decimal point of the system locale
  and has no group separators: text is formatted with
  `spline_fitter::formatting::number`, never a bare `format!`. Files and logs
  keep a dot. Small or large values are written in e-notation (`1.23e-05`,
  `formatting::format_error`). Values spanning decades (tolerances, weights)
  use `widgets::scientific_field`, which steps along 1-2-5 ladders.
- **Captions**: uppercase, `typeface::CAPTION`, strong, letter-spaced,
  `text_3` (`widgets::caption`). One caption names each region: DATA,
  METHOD, CURVE, ADVANCED, PROJECT, CONVERGENCE, RUNS.
- **Wording**: sentence case everywhere ("Fit spline", "Optimization
  runs"); words are spelled out ("Control points", not "Ctrl pts");
  labels have **no trailing colons**; menu items that open a dialog end with
  an ellipsis ("Export CAD..."). This holds for menus, dialog and
  file-dialog titles, buttons and status messages too; only the product
  name, "Airfoil Spline Fitter", keeps its capitals.
- **Glyphs**: no emoji. Icons are the stroked glyphs of `theme::Glyph`
  (chevrons, `Range`, `SortUp`, `SortDown`, `Filter`), drawn in `text_2` at
  `size::GLYPH`. Text symbols are limited to → (from one value to the next),
  plus Greek letters for math (κ, dκ/ds) and × between numbers ("2 × 10"
  control points). No middle dots or bullets as separators: use a comma.
- **The sweep toggle is a dimension line** (`Glyph::Range`, at
  `size::GLYPH_LARGE`): |←→|, as a drawing marks a span. Glyphs on checkable
  buttons turn `text` when checked.
- **Chevrons mean disclosure**: pointing right = closed, down = open
  (collapsible sections, the comb settings drop-down). Sideways chevrons
  fold a side panel away: the explorer's hide button points right, toward
  the edge it folds into, and the rail it leaves there has a chevron
  pointing left to bring it back.

---

## 4. Space and shape (`theme::size`)

Use the named gaps, not numbers:

| Token | px | Between |
|---|---|---|
| `TIGHT` | 4 | Parts of one composite field (field + sweep toggle) |
| `GAP` | 6 | Form rows; fields side by side; toolbar items |
| `GROUP` | 8 | Groups: caption and content, button rows, plot panels |
| `PAD` | 12 | Padding inside panels and sidebar sections |
| `PAD_LARGE` | 16 | Dialogs and documents |

- **Every input and button is `size::CONTROL_HEIGHT` (26 px) tall**: text
  and numeric fields, combo boxes, buttons, segments, the sweep toggle and
  the primary button (`theme::apply` sets the interaction height); a square
  icon button is `CONTROL_HEIGHT` on both sides. The only exception is the
  caption-style header of a collapsible section.
- Corner radius `size::RADIUS` (3 px): crisp, drafted, not pill-shaped.
- Fixed and least widths are tokens too: `FIELD_MIN_WIDTH` (sidebar fields),
  `SIDEBAR_MIN_WIDTH` and `EXPLORER_MIN_WIDTH` (how narrow the side panels
  can be dragged), `SLIDER_WIDTH` (a dialog's slider), `DIALOG_MIN_WIDTH` (a
  dialog with folder fields), `POPUP_MIN_WIDTH` (a column filter popup).
- **The window narrows by stacking, not clipping.** The plot toolbar wraps,
  and the convergence and runs plots stack when the center is too narrow for
  them side by side. Wide content (the run table) scrolls inside its panel.
- Sidebar form labels are `size::LABEL_WIDTH` wide so fields line up across
  sections; add rows with `widgets::form_row`.

### Motion (View > Scrolling and zooming...)

The wheel glides rather than jumps, like a batten easing into place.

| Setting | ms | For |
|---|---|---|
| Scrolling | 350 | A wheel notch in any list, table, panel or document (egui's scroll animation) |
| Zooming | 650 | A wheel notch zooming a plot about the cursor (`plots::show_plot`) |

- Both range from 0 (off: the wheel jumps) to 1000 and are saved; Default
  restores them. They are read at each step, never once at construction.
- Nothing else animates: data, fits and theme changes appear at once.

---

## 5. Components

Every component is a function in `widgets.rs` (or a view in its module) that
takes the `Palette`; none sets colors of its own.

| Component | How |
|---|---|
| Sidebar section | `sidebar::section` (caption, content, rule); `widgets::collapsible_header` for Advanced |
| Form row | `widgets::form_row(ui, palette, "Label", enabled, tooltip, field)` |
| Caption | `widgets::caption` |
| Views of one block | `widgets::caption_tabs`: the block's captions as tabs, the chosen one in `text` (the bottom-left block: Convergence, Report, Shape) |
| Muted note | `widgets::muted` |
| Number readout | monospace at `typeface::READOUT` (the run bar) |
| Primary button | `widgets::primary_button`. **One per view** (it is "Fit spline" in the main window) |
| Secondary button | `widgets::button` |
| Icon-only button | `widgets::icon_button` with a `Glyph`; square |
| Toggle | `egui::Button::selected`; checked state is `select` |
| Toggle with settings | `dialogs::split_toggle`: a toggle and an arrow opening a popup |
| Choice of 2-4 short options | `widgets::segmented` (not a combo box) in the sidebar and dialogs; the plot toolbars keep compact combo boxes (`widgets::combo`) |
| Numeric range sweep | `widgets::sweep_toggle` beside the field and its end field |
| Numbers spanning decades | `widgets::scientific_field` |
| Plot | `plots::show_plot` with its `Axes`; items painted through the `Canvas` in palette colors, in drawing order |
| Dialog | `dialogs::modal` with form rows and `dialogs::buttons` (reset role left, then accept, then reject) |
| Question | `dialogs::Confirm`: one paragraph of text, the buttons at the right (accept, another choice, Cancel); never the system's message box |
| Report | the Report view of the bottom-left block: rows with `ok`/`warn` notes in the status colors; the Max deviation readout under Fit spline in `warn` when the fit missed its tolerance; the thickness and camber chart below (`plots::shape_plot`: thickness a `trace` line, the camber dashed `construction`, each maximum a dot ringed in the canvas color) |
| Table of records | the run table (`explorer.rs`): no grid lines, rows selected whole; numbers monospaced and right-aligned, blanks for what does not apply; Pivot swaps rows and columns |
| Group of records | a folder row in the table (a tolerance search): a `ChevronRight`/`ChevronDown` in `text_2` (`text` on hover) at the left of its first cell opens and closes it, its name in that cell ("Search 1"); the rows inside follow it, their first cell indented as far as the chevron; Right and Left open and close it from the keyboard |
| Column filter | as in a spreadsheet: a `ChevronDown` in `text_3` at the right of each header opens a popup (sort, condition, checklist of values, previewing which values a condition unchecks in `text_3`); a filtered column shows the `Filter` funnel in `text`; the sorted column has `SortUp`/`SortDown` in `text_2` |

**Layout of the main window**: sidebar (workflow order: data, method,
curve, advanced, then the fixed run bar with the primary action and the
result readouts) | plots (main canvas with its toolbar, convergence and
comparison below) | project explorer (the table of every run: filter it,
sort it, plot the filtered runs along any column in the comparison plot,
double-click a run to apply its settings; a tolerance search is a folder
of its runs).

---

## 6. Adding to the GUI: checklist

1. Need a color, size or font? Take it from `theme.rs`. If none fits, add
   a **named** token there (named by role, not hue) and document it here.
   A new grey is a lightness pair in `GREY_LIGHTNESS`; another color that
   must change with the mode is a pair in `Palette::new`.
2. Check it in light and dark, cool and warm (View > Theme...), and make
   sure it follows a switch while the window is open.
3. Need a new look for a widget kind? Add a function to `widgets.rs` that
   draws it from the palette.
4. Drawing something in a plot? Decide whether it is measured, constructed
   or derived, give it a palette token and paint it in drawing order in its
   plot's `show`.
5. Orange? Re-read section 2.
6. Labels: sentence case, no colons, captions uppercase.
7. Run `cargo test -p spline-fitter-gui`: `tests/design_language.rs` checks
   that colors come only from `theme.rs` and that labels follow the wording
   rules, and the window tests drive the flows and can save screenshots
   (`SPLINE_FITTER_SHOTS`) to check light and dark, cool and warm.

## 7. Assets

`crates/spline-fitter-gui/assets/` holds the app icon (`app_icon.png`,
`.ico`), from which the ink colors are sampled. Glyphs are not stored as
files: `Glyph::paint` draws them from their path data in the current token
color, so a theme change recolors every chevron.
