# Loft: the Spline Fitter design language

Before the computer, aircraft and ship lines were drawn full size on a
**lofting floor**. A long, flexible strip of wood, the *spline*, was bent
through the measured stations and held in place by lead weights called
*ducks*. The shape the strip settled into was the fair curve. That is
exactly what this application does, so the interface borrows the floor's
look and vocabulary.

> The window is the lofting floor in graphite: warm neutral greys (the
> Claude Code greys), construction lines in pencil, and one ink (the orange
> of the app icon) for the curve and for the one action that makes it.

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
   theme: the Fusion style and the palette are set by `apply_theme()`.

---

## 2. Color

### UI tokens (`theme.Palette`)

All chrome is **warm neutral grey**, taken from the Claude Code dark
theme; there is no blue or other tint. Surfaces get lighter the closer
they are to the hand.

| Token | Hex | Use |
|---|---|---|
| `CANVAS` | `#191918` | Plot area: the drafting film |
| `WINDOW` | `#1F1E1D` | Window, toolbars, run bar, status bar |
| `PANEL` | `#262624` | Sidebar, project explorer, disabled fields |
| `FIELD` | `#30302E` | Inputs, secondary buttons, menus, drop-down panels |
| `FIELD_HOVER` | `#393936` | Hovered field or button |
| `RULE` | `#3E3D3A` | Hairlines and borders |
| `RULE_STRONG` | `#52514C` | Borders of hovered fields, popups |
| `TEXT` | `#FAF9F5` | Values, primary text |
| `TEXT_2` | `#C2C0B6` | Field labels, axis text, status bar, glyphs, badges |
| `TEXT_3` | `#9C9A92` | Captions, hints, disabled text |
| `INK` | `#FF6E10` | The icon's orange: fitted curve, primary button |
| `INK_HOVER` / `INK_PRESSED` | `#FF8636` / `#E05E08` | Primary button states |
| `INK_DEEP` | `#AD4200` | The icon's shadowed underside (reserved) |
| `ON_INK` | `#1A0C02` | Text on ink |
| `SELECT` | `#4A4945` | Checked toggles, selected rows, menu highlight |
| `SELECT_LINE` | `#85837B` | Focus and checked borders |
| `OK` / `WARN` / `ERROR` | `#4FBF85` / `#F2B84B` / `#F0605D` | Status only |

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

### Plot tokens (`theme.Ink`)

Named by what they draw, never by hue.

| Token | Drawn as | Meaning |
|---|---|---|
| `DATA` | chalk dots, `Size.DATA_SIZE` | Measured stations |
| `CURVE` | solid ink line, `Size.CURVE_WIDTH` | The fitted spline (the batten) |
| `CONSTRUCTION` | dashed pencil grey line, hollow squares | Control polygon and control points (CAD convention: squares) |
| `KNOTS` | sulfur diamonds with a canvas outline | Knots (the ducks that hold the batten) |
| `COMB_UPPER` | sky blue | Comb on the suction side (upper surface, even sections) |
| `COMB_LOWER` | rose | Comb on the pressure side (lower surface, odd sections) |
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
  values line up and digits do not jump while a fit updates.
- **Captions**: uppercase, `Type.CAPTION_PX`, weight 600, letter-spaced,
  `TEXT_3` (`#SectionCaption`). One caption names each region: DATA,
  METHOD, CURVE, ADVANCED, PROJECT, CONVERGENCE, ...
- **Wording**: sentence case everywhere ("Fit spline", "Optimization
  runs"); labels have **no trailing colons**; menu items that open a dialog
  end with an ellipsis ("Export CAD...").
- **Glyphs**: no emoji. Icons are the stroked glyphs from `theme.icon()`
  (`chevron_down`, `chevron_up`, `chevron_right`, `chevron_left`,
  `check`), drawn in `TEXT_2` at `Size.GLYPH`. Text symbols are limited to
  ↔ (sweep) and → (range), plus Greek letters for math (κ, dκ/ds).
- **Chevrons mean disclosure**: pointing right = closed, down = open
  (collapsible sections, tree branches, the comb settings drop-down).
  Sideways chevrons fold a side panel away: the explorer's hide button
  points right, toward the edge it folds into.

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

- **Every input and button is `Size.CONTROL_HEIGHT` (26 px) tall**: line
  edits, spin boxes, combo boxes, push and tool buttons, segments, the sweep
  toggle and the primary button. The stylesheet fixes the height (1 px
  border, no vertical padding), so do not set heights or vertical padding on
  a control; a square icon button is `CONTROL_HEIGHT` on both sides. The
  only exception is the caption-style header of a collapsible section.
- Corner radius `Size.RADIUS` (3 px): crisp, drafted, not pill-shaped.
- Sidebar form labels are `Size.LABEL_WIDTH` wide so fields line up across
  sections; add rows with `_add_row()` in `layout.py`.

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
| Choice of 2-4 short options | `SegmentedControl` (not a combo box) |
| Choice of many | `QComboBox` |
| Numeric range sweep | `SweepableSpinBox` / `SweepableDoubleSpinBox` |
| Plot | `pg.PlotWidget` + `style_plot(plot)` + `label_plot(plot, side, text)` |
| Plot pens and brushes | `theme.pen(Ink.X, width, dashed, alpha)`, `theme.brush(Ink.X, alpha)` |
| Rich text dialog | `theme.text_browser()`; mark status with `class='ok'`, `'warn'`, `'error'` |

**Layout of the main window**: sidebar (workflow order: data, method,
curve, advanced, then the fixed run bar with the primary action and the
result readouts) | plots (main canvas with its toolbar, convergence and
comparison below) | project explorer.

---

## 6. Adding to the GUI: checklist

1. Need a color, size or font? Take it from `theme.py`. If none fits, add
   a **named** token there (named by role, not hue) and document it here.
2. Need a new look for a widget kind? Add an object-name rule to
   `theme._qss()`, not a `setStyleSheet` call.
3. Drawing something in a plot? Decide whether it is measured, constructed
   or derived, give it an `Ink` token and a `Z` value.
4. Orange? Re-read section 2.
5. Labels: sentence case, no colons, captions uppercase.
6. Run `pytest tests/test_design_language.py`.

## 7. Assets

`src/spline_fitter/gui/assets/` holds the app icon (`app_icon.png`,
`.ico`), from which the ink colors are sampled. Glyphs are not stored as
files: `theme.py` writes them as SVGs from their path data in the token
colors, so changing a color token also recolors every chevron.
