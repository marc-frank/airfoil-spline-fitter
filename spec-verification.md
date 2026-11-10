# Verification record for `spec.md`

Results of the checks described in `spec.md` section 23.2. Checks 1-11, and the blind runs of checks 3 and 7-10, ran in one session after the last edit to `spec.md`, to the source and to the scripts; the SHA-256 of every input was recorded before the first check and after the last, and they were identical, as were the application's QSettings before and after. Check 12 (the test suites, which do not read `spec.md`) ran earlier the same day on the same source: `src/spline_fitter/__main__.py` and `tests/test_smooth_motion.py` had the hashes below, and no other source file differed from the commit (`git diff --stat` lists only those two).

## Inputs

| File | SHA-256 |
|---|---|
| `spec.md` | `aa65a0ae889d81e65ba2b7e9e2b65ce823084cec644b12fb0f2ff4e89a8d163e` |
| `spec-appendix-icons.md` | `c66801c17eb68fa25f74275c78126cdc7c81509c90d805e5768f99804d7db3f1` |
| `src/spline_fitter/__main__.py` (exit fix) | `a55572dfb0db979bcb5bf9f93df1224ab848d4bd46797ac12ff6dcc62454572b` |
| `tests/test_smooth_motion.py` (fixture fix) | `d31a65e20bec10f3d02a1c02d3f7d3d88556a728fc880bdbe7e9e07028799667` |
| `spec-verification/acceptance.py` | `f73af9a1f54218000e176a42d04ba4cbf1fc9069a5667add618c284b8dfc2342` |
| `spec-verification/cleanroom.py` | `7b10572189480bc571486b00ff0a0055b59a711bca2683704c9939fdb8c0caaf` |
| `spec-verification/differential.py` | `d21ecdfabd72b93badcb30854f9cd56fbdb6f39f7c9628367651bda627417524` |
| `spec-verification/exit_probe.py` | `55778304a5503d6390d5ff7a627686f8fa424ad700ae86817b18b0e8c0397d9a` |
| `spec-verification/expect_mine.py` | `920baefcbd7c779880b731379621b586013720bb482ab44d4dffa1c4cf765f00` |
| `spec-verification/file_flows_check.py` | `9720d74399d0f66870ec8634f9191c69f22900cf672e8f9bca19534f83fcf9ff` |
| `spec-verification/gui_logic.py` | `44b7f3dd6334a11b1b42321f0b73d1b7f65773827e299af4f45331f46d9b7c93` |
| `spec-verification/gui_model.py` | `e7e5a9bf23a138e75fe64616c686b03d7403768028260bffce91ca84a5ccccca` |
| `spec-verification/gui_mutants.py` | `4fba2c68e0e64c83cad76891fd5bbd2b56eadc8bceca9da1b4b46df3ae4e0434` |
| `spec-verification/plot_check.py` | `80dc2b0e0d82621387741d0e67b882083b977d8b31b50c3746496442734b3368` |
| `spec-verification/theme_check.py` | `4180774b9acb1934e585bdaa6849abc374985eba3e81e77b2e653ded632daa62` |
| `spec-verification/ui_extras_check.py` | `8507f7e8679c8a21d0028f8c79473bd7eb6c9f030e281d6cb9233e5f54201778` |
| `spec-verification/blind/blind_expect.py` (written by the blind agent) | `2bfd055ea2d43a65d6a0d5b1fb22bdf8f62c3f19f6017aa62e3e7fda4e5c2977` |
| `spec-verification/blind/blind_gui.py` (written by the blind agent) | `3d25b137ba6c0c8ae57b7d875fdf4e1483a5acc96818c80da2a79ddf84c95b8a` |
| `spec-verification/blind/blind_impl.py` (written by the blind agent) | `4fa987df7297fe8cd484a162f9dac96c2456b33c1dec0c2b68b9ef2171d3eb11` |
| `spec-verification/blind/blind_plot.py` (written by the blind agent) | `8f2f96f59e4e29b4ce496813dc75e7de5edcf4c20af748393b9c845af9759dfc` |
| `spec-verification/blind/blind_theme.py` (written by the blind agent) | `2ce9405bb186fd3a54ab070a06804b622327e4590639a4d35daf779e7b9a3153` |

Source: commit `2febd1353c26d7deac8b1803f6ee44afb1e228f1` plus the two changes above. Platform: Windows 11, Python 3.12, NumPy 2.2.5, SciPy 1.15.2 (OpenBLAS 0.3.28), PyQt6 / Qt 6.9.0, `QT_QPA_PLATFORM=offscreen` unless stated. `acceptance.py` and `differential.py` find the checkout through a hard-coded `ROOT` (`C:\Users\MSFra\Desktop\spline_fitter`); the other scripts use their own location. The scripts that touch the user's QSettings (`theme_check.py`, `file_flows_check.py`, `exit_probe.py`) restore or remove what they write.

## Summary

| # | Check (`spec.md` 23.2) | Result |
|---|---|---|
| 1 | Acceptance, section 21 | 26 of 26 |
| 2 | Clean-room differential test | 41,733 of 41,733 comparisons, 39 components |
| 3 | Blind reimplementation (agent given only `spec.md`) | core: 41,733 of 41,733 comparisons, 39 components; GUI rules: 9,000 of 9,000 actions and 17,353 of 17,353 comparisons; blind expectations for checks 7-10: 30 of 30, 48 of 48, 30 of 30, 28 of 28 |
| 4 | Bit-for-bit reproducibility | 3 of 3 processes identical |
| 5 | GUI rule model test (19.6, 19.7) | 9,000 of 9,000 actions; 3 of 3 mutants detected |
| 6 | GUI logic model test (19.8, 19.13, 19.16.4, 19.17) | 17,353 of 17,353 comparisons |
| 7 | Theme check (19.19, Appendix A) | 30 of 30 |
| 8 | Plot geometry check (12.1, 12.5, 19.14.3, 19.14.5) | 48 of 48 |
| 9 | File and project flow check (19.11, 19.15, 19.16) | 30 of 30 |
| 10 | Interaction and dialog check (19.11.2, 19.17.3, 19.17.4, 19.20, 19.21 item 13, dialog layouts) | 28 of 28 |
| 11 | Exit status (1.1) | 60 of 60 offscreen runs correct |
| 12 | Full test suites | 673 passed, 2 skipped, 0 failed |

## 1. Acceptance: `python spec-verification/acceptance.py`

26 of 26, exit status 0: items 1-5, 6a-6f, 7-10, 11 (default tolerance, VarPro, Direct LSQ, tolerance search), 12-18. Tolerance search: 17 refits, `[10, 11]` control points, max deviation `9.1121e-07`, 114 s. Item 16 parsed the 13-row grey table from `spec.md`. Items 17 and 18 drove a real window, including a two-file batch run twice.

## 2. Clean-room differential test: `python spec-verification/differential.py`

Seed `20261106`; 41,733 of 41,733 agree, exit status 0.

| Spec section | Component | Agreeing / compared |
|---|---|---|
| 3 | `parse_airfoil_text` (fuzzed files) | 3000 / 3000 |
| 3.6 | `get_precision` | 10 / 10 |
| 4.1, 4.2 | leading edge, `split_into_sections` | 1500 / 1500 each |
| 2.6 | arc-length, centripetal parameters | 1500 / 1500, 300 / 300 |
| 2.1, 2.3 | knots, bending energy | 400 / 400, 245 / 245 |
| 7.1, 7.3, 7.4 | end coefficients, alpha and inverse, constraint rows | 856 / 856, 300 / 300 each, 250 / 250 |
| 5.2 | minimum control points | 400 / 400 |
| 8.2 | softplus, inverse, `t_to_eta`, `eta_to_t` | 300 / 300 each |
| 8.4 | VarPro residual vector | 250 / 250 |
| 8.5 | Levenberg-Marquardt (x, nfev, status, message) | 200 / 200 |
| 9.3, 9.7 | `new_knots`, `removal_order`, PAV | 500 / 500 each |
| 12.6 | default tolerance, decimal places | 150 / 150 each |
| 13.1 | NACA 4-digit points | 14 / 14 |
| 14.2, 14.4 | spline file text, STEP real, STEP string | 200 / 200, 2000 / 2000, 8 / 8 |
| 16 | batch `output_name`, `num` | 3000 / 3000 each |
| 18 | `format_error`, `format_duration`, `file_stem`, `sanitize_filename` | 3000 / 3000 each |
| 19.17.2 | explorer `parse_filter` | 3000 / 3000 |
| 19.19.1 | `grey()` | 2000 / 2000 |
| 19.21 | scientific spin box step and text, `sweep_values` | 1000 / 1000 each, 500 / 500 |

## 3. Blind reimplementation: `IMPL=spec-verification/blind/blind_impl.py python spec-verification/differential.py`

A separate agent was given only `spec.md`, instructed to read no other file of the repository and to mark every unclear passage with a `SPEC-GAP` note, and asked to implement the 39 components of check 2 (it confirmed it read only `spec.md`; this was an instruction, not a technical restriction). Scored with the same harness and seed:

| Round | Result |
|---|---|
| First implementation, from the earlier `spec.md` | 41,315 of 41,733; failures in `eta_to_t` (299 of 300), Levenberg-Marquardt (63 of 200) and the spin-box step (56 of 1000) |
| After `spec.md` was revised, the agent's own revision from the text alone (it was not told its scores) | **41,733 of 41,733**, exit status 0 |

What the first round found, all fixed in `spec.md`: the evaluation order of `eta_to_t` (8.2) and of the predicted reduction in Levenberg-Marquardt (8.5); 13 further points the agent marked unclear (2.3 breakpoints, 3.3 empty fields, 3.5 behaviour for infinite values, which raises `OverflowError` in the source, 3.6, 4.2 indexing, 7.1 orders above the degree, 8.5 empty maximum and the Cholesky form, 9.3, 12.6 rounding, 14.4 per-character escapes, 16, 19.17.2, 19.21 item 6 clamping), each written down as measured in the source. The spin-box failures were in the harness, which compared the widget's internal step instead of the clamped step a user sees; the harness and `cleanroom.py` were corrected. Two notes remain, neither a gap in `spec.md`: 2.1 has two documented message variants (library and application), and the test interface `ladder_next` passes no maximum (`spec.md` clamps to the range).

### 3b. Blind GUI rules: `IMPL_GUI=spec-verification/blind/blind_gui.py` with `gui_model.py` and `gui_logic.py`

The same agent then implemented, again from `spec.md` alone, the GUI rules of 19.6, 19.7, 19.8, 19.13, 19.16.4, 19.17.1 and 19.17.2 (`blind_gui.py`). The harnesses keep only Qt's value clamping and the signal routing of 19.7 and take every application rule from that module.

| Round | Sidebar rules (`gui_model.py`, seeds 7, 101, 202) | Logic rules (`gui_logic.py`, with `blind_impl.py` for filter parsing) |
|---|---|---|
| First implementation | 2,996 of 3,000 (seed 7) | 17,353 of 17,353 |
| After `spec.md` was revised, the agent's own revision (not told its scores) | **3,000 of 3,000 for each seed** | **17,353 of 17,353** |

What the first round found, fixed in `spec.md`: 19.6 did not say that tolerance mode stops a sweep only when it is on (stopping emits a signal that reruns the curve rule; the 4 mismatches); 19.7 did not say that Per junction entries need at least one entry. Its other notes (ties of the sweep ends, the empty maximum, the sections end only lowered, Advanced disabled during a fit, the Parameterization text rule, `!=` in any column, sort order of the sweep series) were written down as the source behaves.

### 3c. Blind expectations for checks 7-10: `IMPL_THEME`, `IMPL_PLOT`, `IMPL_EXPECT`

The same agent wrote, from `spec.md` alone, the expected values that checks 7-10 compare the running application with: the stylesheet, Qt palette roles and glyph files (`blind_theme.py`), the comb and deviation geometry (`blind_plot.py`), and the status texts, dialog arguments and contents, file defaults and dialog layouts (`blind_expect.py`). Each harness was run with these in place of its own expectations (`expect_mine.py` and the formulas inside the harnesses).

| Check | First implementation | Final run |
|---|---|---|
| 7 Theme | 30 of 30 | **30 of 30** |
| 8 Plot geometry | mismatches in the comb and deviation | **48 of 48** |
| 9 File and project flows | 29 of 30 | **30 of 30** |
| 10 Interactions and dialogs | 28 of 28 | **28 of 28** |

What the first round found, fixed in `spec.md`: 19.14.3 did not say that the comb's `chord` samples `linspace(0, 1, 200)`, that NaN curvature is set to 0 before it is integrated, that "strictly increases" compares with the neighbouring grid point, or how teeth, whiskers and the tolerance band are laid out with NaN separators; 19.14.5 did not give the order of the band lines. The one flow mismatch was in the harness: the interface the agent was given left open whether `not_project_file` is the bare error or the whole status line, and the harness now accepts either reading (the shown text is otherwise compared exactly).

## 4. Bit-for-bit reproducibility: `acceptance.py --fits-only`, three processes

All three: `0aba644dc8828ed2662da234e0a49f674dc5630503ccb22ec28a103b381452b7` (SHA-256 of every output of the item-11 fits).

## 5. GUI rule model test: `gui_model.py`, `gui_mutants.py`

25 widget states compared after every random action: seeds 7, 101 and 202, 3000 of 3000 each. Mutants (600 actions, seed 11), each detected: `need` 468 of 600 agreed, `cap` 277, `label` 463.

## 6. GUI logic model test: `gui_logic.py`

Seed `1106`; 17,353 of 17,353 agree.

| Spec section | Rule | Agreeing / compared |
|---|---|---|
| 19.8 | history states, index, undo/redo availability after every action | 4200 / 4200 |
| 19.8 | widget values restored by each undo and redo | 573 / 573 |
| 19.17.1 | column names, metric and log flags; all 25 cell values and texts | 1 / 1; 1179 / 1179 |
| 19.17.2 | row filter | 1200 / 1200 |
| 19.17.3 | sort order | 1200 / 1200 |
| 19.16.4 | run matching (random pairs; equal configurations) | 3000 / 3000 each |
| 19.13 | sweep comparison series | 3000 / 3000 |

## 7. Theme check: `theme_check.py`

Light and dark at temperatures -100, -40, 0, 25, 100: stylesheet, Qt palette and glyph files each agree in all 10 states (30 of 30).

## 8. Plot geometry check: `plot_check.py`

24 random fits (1-4 sections, VarPro or Direct LSQ, with and without a tolerance): comb (random quantity 0-3, height, density) 24 of 24, deviation 24 of 24. Compared with the data the plot manager sets (`getOriginalDataset`), within 1e-7 relative and 1e-9 of each array's largest value (higher comb derivatives are differentiated numerically; the measured differences were at most about 2e-14 relative).

## 9. File and project flow check: `file_flows_check.py`

30 of 30: status texts, dialog titles, default paths, filters and written files of the flows listed in `spec.md` 23.2 item 8.

## 10. Interaction and dialog check: `ui_extras_check.py`

28 of 28: drag and drop (refusal of unsupported files and during a fit, case-insensitive `.DAT`, opening the first supported file), pivot on and off (selection, row order, filters, sorting), the column filter drop-down (caption, width, buttons, values with `Not used` last, live condition, Keep matching, Add matching, Select all both ways, one value by hand, Clear filter), Qt's OutCubic equal to `1 - (1 - p)^3` at 51 points, the scroll animation (per-pixel mode, easing, `SCROLL_MS`, target `2 notches x wheelScrollLines x singleStep`), the zoom animation (easing, progress 0 to 1), a live button box ordered reset, accept, reject, and the rows, items, ranges and button roles of the New NACA airfoil, Export CAD, Batch convert, Theme and Scrolling and zooming dialogs. This check corrected `spec.md` 19.21 item 13 and the dialog descriptions, which had given a fixed button order.

## 11. Exit status: `exit_probe.py`

20 rounds of `idle`, `fit` and `code3` (event loop ended with `QApplication.exit(3)`), offscreen: 60 of 60 correct. Each had the expected exit status (0, 0, 3), wrote a QSettings value just before closing that a new process read back, and never returned from `main()`. Before the fix, the same probe crashed (status 139) in 5 of 6 offscreen and 4 of 6 native Windows runs; with the fix, 9 of 9 native Windows runs were correct (earlier the same session).

## 12. Full test suites: `python -m pytest tests packages/bspline-varpro/tests -q -rs`

673 passed, 2 skipped, 0 failed (1319.83 s). Skips: `packages/bspline-varpro/tests/test_export.py:315`, `could not import 'ezdxf'` (optional DXF reader, not installed). Earlier the same day the suite failed `tests/test_smooth_motion.py::test_list_scrolls_by_pixel_and_glides` twice: the preceding test's window, freed with `deleteLater()`, was destroyed during the measured glide (about 8,000 deferred deletions), delaying it by about 50 ms. The motion tests' fixture now delivers pending deferred deletions first.

## Limits of this evidence

- The blind checks (3, 3b, 3c) cover the 39 components of check 2, the GUI rules of checks 5 and 6 and the expectations of checks 7-10. The harnesses themselves (which widgets they drive and how they read them) were written by someone who had read the source; the acceptance list (check 1), the exit probe (check 11) and the test suites (check 12) have no blind counterpart. The blind agent was instructed, not technically prevented, to read only `spec.md`; it confirmed this each time.
- No finite set of checks proves that a document leaves nothing unspecified; these checks show that an engineer working from `spec.md` alone reproduced every behaviour they compare.
- Not covered by a model or literal check: the live-preview timing (covered by the test suite), pixel rendering by Qt and pyqtgraph (outside the spec's scope, 0.2), and the animation frame timing (only its parameters are checked).
- Bit-for-bit results were checked on the reference build only (`spec.md` 23.3).
