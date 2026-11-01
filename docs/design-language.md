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
[`src/spline_fitter/gui/theme.py`](../src/spline_fitter/gui/theme.py).
That module is the **only** place colors, fonts, sizes and stylesheets are
defined. [`tests/test_design_language.py`](../tests/test_design_language.py)
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
   style: the Fusion style and the palette are set by `apply_theme()`. Only
   light or dark follows the OS, and only while the user leaves it so.

---

## 2. Color

### Theme setting (`theme.Scheme`, View > Theme...)

The user picks two things; both are saved in QSettings and applied live:

- **Mode**: system (the default: follows the OS, also when it switches),
  light or dark.
- **Temperature**: -100 (cool, blue) .. 0 (neutral, the default) .. +100
  (warm, sand). +50 reproduces the Claude Code greys.

The greys are therefore **not fixed hex values**. Each is a lightness in
`GREY_LIGHTNESS` (one for dark, one for light), and `grey(lightness,
temperature)` turns it into a color in OKLab, so a temperature change keeps
every contrast. At temperature 0 the greys have no hue at all. Colors that
must differ between light and dark but are not greys (status, knots, comb)
are pairs in `MODE_COLORS`. Never write a grey as a hex value anywhere: add
a lightness pair to `GREY_LIGHTNESS`.

### UI tokens (`theme.Palette`)

Dark mode: surfaces get lighter the closer they are to the hand. Light
mode: paper-white canvas and fields on slightly darker panels. Lightness
values are OKLab (dark / light).

| Token | Lightness | Use |
|---|---|---|
| `CANVAS` | 0.213 / 0.995 | Plot area: the drafting film |
| `WINDOW` | 0.236 / 0.970 | Window, toolbars, run bar, status bar |
| `PANEL` | 0.268 / 0.950 | Sidebar, project explorer, disabled fields |
| `FIELD` | 0.309 / 1.000 | Inputs, secondary buttons, menus, drop-down panels |
| `FIELD_HOVER` | 0.344 / 0.965 | Hovered field or button |
| `RULE` | 0.360 / 0.870 | Hairlines and borders |
| `RULE_STRONG` | 0.434 / 0.780 | Borders of hovered fields, popups |
| `TEXT` | 0.982 / 0.250 | Values, primary text |
| `TEXT_2` | 0.807 / 0.430 | Field labels, axis text, status bar, glyphs, badges |
| `TEXT_3` | 0.686 / 0.580 | Captions, hints, disabled text |
| `SELECT` | 0.405 / 0.895 | Checked toggles, selected rows, menu highlight |
| `SELECT_LINE` | 0.609 / 0.600 | Focus and checked borders |

Fixed colors, the same in both modes:

| Token | Hex | Use |
|---|---|---|
| `INK` | `#FF6E10` | The icon's orange: fitted curve, primary button |
| `INK_HOVER` / `INK_PRESSED` | `#FF8636` / `#E05E08` | Primary button states |
| `INK_DEEP` | `#AD4200` | The icon's shadowed underside (reserved) |
| `ON_INK` | `#1A0C02` | Text on ink |
| `OK` / `WARN` / `ERROR` | dark `#4FBF85` / `#F2B84B` / `#F0605D`, light `#23905A` / `#B87A00` / `#D23B37` | Status only |

**Rules**

- **Ink is scarce.** Allowed uses: the fitted curve, the primary button,
  the marker for the current run in a trace. A new feature that wants
  orange needs a reason as strong as those.
- **Selection is a lighter grey, not a color.** Checked toggles, segmented
  controls, selected tree rows and focus rings use `SELECT` /
  `SELECT_LINE`. The chrome carries no hue at all; color is kept for the
  plots, the primary action and status.
- **Status colors are for status.** Use `OK`/`WARN`/`ERROR` only to say
  that something is fine, doubtful or failed, never to decorate.
- **Disabled is `TEXT_3` on `PANEL`.** Disabled things are not hidden or
  greyed by opacity.
- **Everything follows a theme change.** The stylesheet restyles widgets by
  itself. Whatever takes a color some other way must restyle on
  `theme_changed.changed`: plot items get their style from a `_..._style()`
  method of `PlotManager` that `restyle()` reapplies; icons come from
  `theme.icon()`, which paints in the current token color. Read tokens when
  drawing, never copy them into a constant at import time.
- **Readable at every setting.** Tests check, in both modes and at both
  temperature extremes, that `TEXT` / `TEXT_2` / `TEXT_3` keep a contrast of
  7 / 4.5 / 2.5 on every surface.

### Plot tokens (`theme.Ink`)

Named by what they draw, never by hue.

| Token | Drawn as | Meaning |
|---|---|---|
| `DATA` | chalk (or graphite) dots in `TEXT`, `Size.DATA_SIZE` | Measured stations |
| `CURVE` | solid ink line, `Size.CURVE_WIDTH` | The fitted spline (the batten) |
| `CONSTRUCTION` | dashed pencil grey line, hollow squares (a grey in `GREY_LIGHTNESS`) | Control polygon and control points (CAD convention: squares) |
| `KNOTS` | sulfur diamonds with a canvas outline (`MODE_COLORS`) | Knots (the ducks that hold the batten) |
| `COMB_UPPER` | sky blue (`MODE_COLORS`) | Comb on the suction side (upper surface, even sections) |
| `COMB_LOWER` | rose (`MODE_COLORS`) | Comb on the pressure side (lower surface, odd sections) |
| `DEVIATION_UPPER` / `DEVIATION_LOWER` | the comb's colors: faint whiskers from the curve, an envelope through their tips | Signed deviation of the data on the upper / lower surface, magnified along the normal; the tolerance band is faint dashed `CONSTRUCTION` |
| `TRACE` | light grey dots or line | Convergence and comparison traces |
| `MARKER` | ink dot with a canvas ring | The current run in a trace |
| `GRID` | `TEXT_2` at `GRID_ALPHA` | Grid and axes |

The comb follows the pressure-plot convention (suction cool, pressure
warm). Its teeth are faint (`COMB_TEETH_ALPHA`) because the envelope
through their tips is what you read.

### Drawing order (`theme.Z`)

`COMB` < `CONSTRUCTION` < `DATA` < `CURVE` < `KNOTS`.
The curve covers the data it fits, so a data point that shows beside the
curve is a deviation worth seeing. Every new plot item takes a `Z` value.

---

## 3. Type

- **UI text**: the system face at `Type.BASE_PT` (9 pt).
- **Numbers**: monospace (`Type.MONO`: Cascadia Mono, Consolas, ...).
  Spin boxes and result readouts (`#Readout`, `Type.READOUT_PX`) use it, so
  values line up and digits do not jump while a fit updates; label and value
  lists use `widgets.ReadoutList` (`theme.readout_value()`), reports the `num` class. Every number
  shown (fields, readouts, messages, reports, plot ticks) takes the decimal
  point of the system locale and has no group separators: `apply_theme()`
  sets the locale, and text is formatted with `formatting.number()`, never a
  bare f-string format. Files and logs keep a dot. Small or large values are
  written in e-notation (`1.23e-05`, `formatting.format_error`). Values
  spanning decades (tolerances, weights) use `ScientificSpinBox`, not a dozen
  fixed decimals.
- **Captions**: uppercase, `Type.CAPTION_PX`, weight 600, letter-spaced,
  `TEXT_3` (`#SectionCaption`). One caption names each region: DATA,
  METHOD, CURVE, ADVANCED, PROJECT, CONVERGENCE, ...
- **Wording**: sentence case everywhere ("Fit spline", "Optimization
  runs"); words are spelled out ("Control points", not "Ctrl pts");
  labels have **no trailing colons**; menu items that open a dialog
  end with an ellipsis ("Export CAD..."). This holds for menus, dialog and
  file-dialog titles, buttons and status messages too; only the product
  name, "Airfoil Spline Fitter", keeps its capitals.
- **Glyphs**: no emoji. Icons are the stroked glyphs from `theme.icon()`
  (`chevron_down`, `chevron_up`, `chevron_right`, `chevron_left`,
  `check`, `dash` (partly checked), `range`, `sort_up`, `sort_down`, `filter`), drawn in `TEXT_2` at `Size.GLYPH`. Text symbols are limited to
  → (from one value to the next), plus Greek letters for math (κ, dκ/ds).
- **The sweep toggle is a dimension line** (`range`, at `Size.GLYPH_LARGE`):
  |←→|, as a drawing marks a span. Glyphs on checkable buttons turn `TEXT` when checked
  (`icon(name, checked_role="TEXT")`).
- **Chevrons mean disclosure**: pointing right = closed, down = open
  (collapsible sections, tree branches, the comb settings drop-down).
  Sideways chevrons fold a side panel away: the explorer's hide button
  points right, toward the edge it folds into, and the rail it leaves
  there (`ExplorerRail`) has a chevron pointing left to bring it back.

---

## 4. Space and shape (`theme.Size`)

Use the named gaps, not numbers:

| Token | px | Between |
|---|---|---|
| `TIGHT` | 4 | Parts of one composite field (spin box + sweep toggle) |
| `GAP` | 6 | Form rows; fields side by side; toolbar items |
| `GROUP` | 8 | Groups: caption and content, button rows, plot panels |
| `PAD` | 12 | Padding inside panels and sidebar sections |
| `PAD_LARGE` | 16 | Dialogs and documents |
| `PLOT_MARGIN` | 24 | Inside a plot's frame: above the plot area and right of it (`style_plot`) |

- **Every input and button is `Size.CONTROL_HEIGHT` (26 px) tall**: line
  edits, spin boxes, combo boxes, push and tool buttons, segments, the sweep
  toggle and the primary button. The stylesheet fixes the height (1 px
  border, no vertical padding), so do not set heights or vertical padding on
  a control; a square icon button is `CONTROL_HEIGHT` on both sides. The
  only exception is the caption-style header of a collapsible section.
- Corner radius `Size.RADIUS` (3 px): crisp, drafted, not pill-shaped.
- Sidebar form labels are `Size.LABEL_WIDTH` wide so fields line up across
  sections; add rows with `_add_row()` in `layout.py`.

### Motion (`theme.Motion`, `motion.py`, View > Scrolling and zooming...)

The wheel glides rather than jumps, like a batten easing into place: each
notch starts a short ease-out, and notches during a glide extend it.

| Token | ms | For |
|---|---|---|
| `SCROLL_MS` | 350 (`DEFAULT_SCROLL_MS`) | A wheel notch in any list, table, panel or document |
| `ZOOM_MS` | 650 (`DEFAULT_ZOOM_MS`) | A wheel notch zooming a plot, about the cursor |

- The user sets both in `MotionDialog` (`widgets.py`), from `0` (off: the
  wheel jumps) to `MAX_MS`; `set_motion()` saves them and the next wheel
  step uses them. Default restores the defaults. Read the tokens at each
  step, never once at construction.

- Scrolling is app-wide: `install_smooth_scrolling()` (called by the main
  window) filters wheel events, and item views scroll by pixel. Nothing to
  do for a new list or panel.
- The wheel with a modifier, touchpads that send pixel deltas, and dragging
  a scroll bar keep Qt's own handling.
- Nothing else animates: data, fits and theme changes appear at once.

---

## 5. Components

Styled by object name in `theme._qss()`. Give a widget the object name;
never call `setStyleSheet` on it.

| Component | How |
|---|---|
| Sidebar section | `SidebarSection("Title")` / `CollapsibleSection` (`widgets.py`) |
| Form row | `_add_row(form, "Label", field)`; label gets `#FormLabel` |
| Caption | `QLabel("CAPTION")` with object name `SectionCaption` |
| Muted note | object name `Muted` |
| Number readout | object name `Readout` |
| Primary button | object name `PrimaryButton`. **One per view** (it is "Fit spline" in the main window) |
| Secondary button | plain `QPushButton` / `QToolButton` |
| Icon-only button | object name `IconButton` with a `theme.icon()` glyph |
| Toggle | checkable `QToolButton`; checked state is `SELECT` |
| Toggle with settings | `SplitToggle` button + `ArrowButton` opening a `DropDownPanel` |
| Choice of 2-4 short options | `SegmentedControl` (not a combo box) in the sidebar and dialogs; the plot toolbars keep compact combo boxes |
| Choice of many | `QComboBox` |
| Numeric range sweep | `SweepableSpinBox` / `SweepableDoubleSpinBox` |
| Plot | `FramedPlot(viewBox=SmoothViewBox())` + `style_plot(plot)` + `label_plot(plot, side, text)` |
| Plot pens and brushes | `theme.pen(Ink.X, width, dashed, alpha)`, `theme.brush(Ink.X, alpha)` |
| Numbers spanning decades | `ScientificSpinBox` (`widgets.py`) |
| Settings dialog | `dialog_form(dialog)` (dialog padding) + `add_form_row(form, "Label", field)`, buttons in a `QDialogButtonBox` |
| Rich text dialog | `show_document(parent, title, html, w, h)` around `theme.text_browser()`; mark status with `class='ok'`, `'warn'`, `'error'`, numbers with `class='num'`. A chart goes below the text as `extra` (the fit report's thickness and camber, `plotting.shape_plot`: thickness a `TRACE` line, the camber line dashed `CONSTRUCTION`, each maximum a dot ringed in the canvas color) |
| Slider | `QSlider`; the temperature slider is `TemperatureSlider` (cool-to-warm groove) |
| Theme setting | `ThemeDialog` (`widgets.py`), from View > Theme... |
| Scroll and zoom glide | `MotionDialog` (`widgets.py`), from View > Scrolling and zooming... |
| Table of records | `QTableView` without grid lines, rows selected whole; numbers monospaced and right-aligned, blanks for what does not apply; a *Pivot* toggle swaps rows and columns, the drop-downs moving to the row names (the explorer's `RunTable`, `explorer.py`) |
| Column filter | As in a spreadsheet: a `chevron_down` in `TEXT_3` at the right of each header section opens a `#DropDownPanel` (sort, checklist of values, and a condition that previews which values it unchecks, greying them in `TEXT_3`, until Enter keeps it); a filtered column shows the `filter` funnel in `TEXT`; the sorted column has `sort_up`/`sort_down` in `TEXT_2` left of it (`FilterHeader`, `ColumnFilterPopup`) |

**Layout of the main window**: sidebar (workflow order: data, method,
curve, advanced, then the fixed run bar with the primary action and the
result readouts) | plots (main canvas with its toolbar, convergence and
comparison below) | project explorer (the table of every run: filter it,
sort it, plot the filtered runs along any column in the comparison plot,
double-click a run to apply its settings).

---

## 6. Adding to the GUI: checklist

1. Need a color, size or font? Take it from `theme.py`. If none fits, add
   a **named** token there (named by role, not hue) and document it here.
   A new grey is a lightness pair in `GREY_LIGHTNESS`; another color that
   must change with the mode is a pair in `MODE_COLORS`.
2. Check it in light and dark, cool and warm (View > Theme...), and make
   sure it follows a switch while the window is open.
3. Need a new look for a widget kind? Add an object-name rule to
   `theme._qss()`, not a `setStyleSheet` call.
4. Drawing something in a plot? Decide whether it is measured, constructed
   or derived, give it an `Ink` token, a `Z` value and a `_..._style()`
   method that `PlotManager.restyle()` reapplies.
5. Orange? Re-read section 2.
6. Labels: sentence case, no colons, captions uppercase.
7. Run `pytest tests/test_design_language.py`. Besides colors and heights it
   checks the wording of every label, item, menu entry, tooltip and status
   message (sentence case, colons, abbreviations, symbols) and that short
   choices are segmented controls.

## 7. Assets

`src/spline_fitter/gui/assets/` holds the app icon (`app_icon.png`,
`.ico`), from which the ink colors are sampled. Glyphs are not stored as
files: `theme.icon()` paints them from their path data in the current token
color, and the stylesheet's glyphs (spin box and combo arrows, tree
branches, check marks) are SVGs `theme.py` writes in the token colors, so a
theme change recolors every chevron.
