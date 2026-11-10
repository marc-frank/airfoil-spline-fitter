# Airfoil Spline Fitter: Reimplementation Specification

Product: `spline-fitter` 2.0.0 (Python package `spline_fitter`, window title "Airfoil Spline Fitter"), with its fitting library `bspline-varpro` 3.0.0 (`packages/bspline-varpro`).
Baseline: source tree at commit `6db8ad3` on `main` (commit `2febd13` plus the exit fix in `src/spline_fitter/__main__.py`, section 1.1 step 5, and a test-fixture fix that changes no behaviour).
The specification is this file plus Appendix B, `spec-appendix-icons.md` (the two icon files, base64).
Revision: 2 (2026-11-10). The verification record `spec-verification.md` was run against revision 1; section 23.4 lists what revision 2 changed and which checks cover it.

This document lets an engineer who has never seen the source rebuild the system with the same behaviour: the algorithms, constants, defaults, file formats, message texts and user-visible rules that the code implements are stated as numbers, formulas, ordered procedures or literal strings. Where the code contradicts its own comments or README, section 22 states which behaviour this spec requires.

Contents:

| Sections | Subject |
|---|---|
| 0-1 | Scope, conformance, platform, startup, architecture |
| 2 | B-spline mathematics |
| 3-4 | Reading airfoil files; airfoil data and the section split |
| 5-6 | Configuration and result models; the application curve class |
| 7-9 | Library: junction constraints, the VarPro core, growth, polishing, tolerance search, joint refinement |
| 10-11 | Direct LSQ; optimizer dispatch and adapters |
| 12-13 | Measuring a fit; airfoil geometry of a fit |
| 14-17 | File formats: `.dat`, `.spline.dat`, STEP, IGES, DXF, `.asfproj`, batch names, CSV logs |
| 18 | Display formatting |
| 19 | Graphical user interface (19.1-19.23) |
| 20 | Error handling summary |
| 21 | Acceptance examples |
| 22 | Known inconsistencies and their resolution |
| 23 | Review and verification record (informative) |
| Appendix A, B | Application stylesheet; icon files (`spec-appendix-icons.md`) |

Normative and informative parts: sections 0 to 20, section 22 and the appendices are requirements. Section 21 is the conformance test (0.3 says which of its items are exact). Section 23 is informative: it records how this document was reviewed and checked and requires nothing.

Conventions:

- `p` degree, `n` control points of one section, `S` number of sections, `N` number of data points, `dim` coordinate dimension (2 in the application).
- A few letters are reused locally, always with a definition beside them, and the local meaning holds only in that section: `S` and `S_b` are also bending-energy matrices (2.3, 8.4); `N(u)` is also the basis matrix of 8.4, 9.7 and 10; `T` is the total control-point count (7.4, 8.4), the theme temperature (19.19) and a count of files or fits in status texts (19.13, 19.18); in 13.1 `m`, `p`, `t` are the NACA camber, camber position and thickness; `u` is a curve or chain parameter and `t` a knot vector, except in 8.1 and 9.7 where `t` also names local data parameters.
- Points are `(x, y)` pairs of IEEE double (`float64`). Arrays are row-major; "point-major flattening" of an `(m, dim)` array means `[r0x, r0y, r1x, r1y, ...]`.
- "Selig order": points run from the upper trailing edge (TE), over the leading edge (LE), to the lower trailing edge (counterclockwise for a normal airfoil).
- "Chain": the sections joined end to end. "Junction": where section `j` ends and section `j+1` starts (`S - 1` junctions).
- Literal strings (messages, labels, file text) are given in `"double quotes"` or code font and must be reproduced character for character, including Unicode characters (`κ`, `µ`, `×`, `→`, `∫`, `²`, `³`, `Δ`, `°`).
- `{name}` inside a literal is a substitution; `{v:.3e}` means Python format-spec semantics (`format(v, ".3e")`). `number(v, spec)` (section 18) is that format with the display decimal point.
- Python semantics are normative where a rule says "as Python's ...": `round` is round-half-to-even, `repr(float)` is the shortest round-trip text, `str.split()` splits on runs of whitespace, `casefold()` is Unicode case folding.
- References such as `io.py:_normalize` name the original code for traceability only; nothing in this document requires reading it.

---

## 0. Scope

### 0.1 In scope

1. Parsing airfoil coordinate text and files (`.dat`, Selig and Lednicer, many number formats) into normalized Selig-order coordinates.
2. Splitting an airfoil into `S` sections.
3. Fitting a chain of clamped B-spline sections by two methods:
   - **VarPro LM**: Levenberg-Marquardt variable projection with free data parameters, optional free knots, knot growth, a tolerance-driven search and geometric G1-G3 or parametric C0-C3 junction continuity (library `bspline-varpro`);
   - **Direct LSQ**: one constrained linear least-squares solve followed by joint parameter refinement.
4. Measuring a fit: orthogonal deviation, curvature fairness, junction continuity, leading-edge radius, thickness and camber.
5. Writing `.spline.dat`, resampled `.dat`, STEP AP214, IGES 5.3, DXF R2000 and CSV optimizer logs.
6. Project files (`.asfproj`).
7. Batch conversion of a folder.
8. A PyQt6 desktop GUI with live preview, parameter sweeps, undo/redo of settings, a project explorer, a fit report and a themable appearance.

### 0.2 Out of scope

- `scripts/`, `packages/bspline-varpro/benchmarks/` (developer tools) and the test suites (their role is in section 21).
- Publishing and distribution (building and uploading releases, installers). The package layout, dependencies and entry points that the behaviour depends on are in scope (section 1).
- Platform-native rendering details that Qt's "Fusion" style decides (anti-aliasing, exact text metrics). Everything the application itself sets (colours, sizes, stylesheet) is specified (section 19.19 and Appendix A).

### 0.3 Conformance

An implementation conforms when:

1. every rule of the normative sections (0 to 20, 22, the appendices) holds;
2. the acceptance examples of section 21 give the stated outputs: exactly for the items without a fit (1 to 10, 13 to 16, 18), and for the items that run a fit (11, 12, 17) as the next paragraph says;
3. files it writes can be read by this application and vice versa, and every writer of sections 14 to 17 produces the same bytes as the reference writer when given the same values. The values that differ from run to run or from machine to machine are excluded: timestamps, elapsed times, `run_id`, the `platform` field, and the line ends of the files that follow the platform (section 1, "Text files");
4. the GUI exposes every control, label, default, message and interaction listed in section 19.

A reimplementation in another language must reproduce the numerics of the named SciPy/NumPy routines (B-spline evaluation by de Boor, LU with partial pivoting, Cholesky, SVD, pseudo-inverse, Gauss-Legendre nodes, bounded Brent minimization). Bitwise identity of iterates is not required, so two levels apply to anything computed by an iterative fit:

- **Reference build** (the versions named in section 21): every number, count and status text of items 11, 12 and 17 is reproduced as written.
- **Any other build or language**: errors and deviations of item 11 agree within the stated relative tolerance of 1e-3; the control-point counts of the fixed-count fits (`[25, 25]`) and everything in items 12 and 17 that does not come out of an iteration are exact. The evaluation and refit counts (`nit = 40`, 17 refits), the counts found by the tolerance search (`[10, 11]`) and the digits of fit results inside status texts (item 17) may differ, because rounding in the last bits can move a stopping test by one step; an implementation whose tolerance search meets the tolerance with at most one control point more per section than the reference conforms.

---

## 1. Platform, packaging and entry points

| Item | Requirement |
|---|---|
| Language | Python >= 3.10 (library: >= 3.9). |
| Dependencies | `numpy>=1.20`, `scipy>=1.8`, `bspline-varpro>=3.0`, `PyQt6>=6.2`, `pyqtgraph>=0.13`, `chime>=0.7`. The library needs only NumPy (>= 1.22) and SciPy. |
| SciPy routines used | `interpolate.BSpline` (evaluation, `.derivative`, `.design_matrix`), `linalg.lu_factor/lu_solve`, `linalg.cho_factor/cho_solve`, `linalg.solve`, `linalg.block_diag`, `optimize.minimize_scalar(method="bounded")`. NumPy: `linalg.cholesky/eigh/svd/lstsq/pinv/solve`, `polynomial.legendre.leggauss`, `trapezoid` (NumPy 2; `trapz` in NumPy 1, same rule). |
| Build | Hatchling. Wheel contains `src/spline_fitter`. The library is a separate distribution. |
| Console script | `spline-fitter` -> `spline_fitter.__main__:main`. |
| GUI script | `spline-fitter-gui` -> `spline_fitter.__main__:main`. |
| Module run | `python -m spline_fitter`; from a checkout `python -m src.spline_fitter`. |
| Library path | On import, `spline_fitter/__init__.py` computes `<checkout> = parents[2]` of its own file. If `<checkout>/packages/bspline-varpro/src` is a directory and not already in `sys.path`, it is inserted at position 0. |
| Lazy GUI | The attribute `spline_fitter.SplineFitterWindow` imports the GUI on first access (module `__getattr__`), so all non-GUI APIs work without Qt. Any other unknown attribute raises `AttributeError("module 'spline_fitter' has no attribute '<name>'")`. |
| Versions | `spline_fitter.__version__ = "2.0.0"`, `__author__ = "Airfoil Library Project"`; `bspline_varpro.__version__ = "3.0.0"`. |
| Text files | The application's own writers (`.dat` 14.1, `.spline.dat` 14.2, `.asfproj` 15.3, CSV logs 17) write UTF-8 through Python's text mode with the default newline handling, so every `\n` this document shows in their content is written as the platform line end: `\r\n` on Windows (the reference build), `\n` elsewhere. The library's writers (CAD files 14.3 to 14.6, `write_dat` 13.6) write ASCII with `\n` on every platform. The glyph SVG files (19.21 item 12) contain no line break. All readers accept `\n`, `\r\n` and `\r` (section 22 item 15). |

### 1.1 Startup

`cli_entry()` (used by `python -m`): parse arguments with argparse (`prog="spline_fitter"`, description `"Interactive B-spline fitting tool for airfoil coordinates."`, epilog listing `python -m spline_fitter`). There are no options: `--help` prints help and exits 0; any argument exits 2 with a usage error. Then `sys.exit(main())`.

The installed scripts call `main()` directly; the application does not parse command-line arguments there (Qt still receives `sys.argv` in step 2 and consumes its own options, such as `-platform`).

`main()`:

1. On `sys.platform == "win32"`, call `SetCurrentProcessExplicitAppUserModelID("AirfoilSplineFitter")` (shell32); ignore any failure.
2. Create `QApplication(sys.argv)`.
3. Create `SplineFitterWindow()` (section 19), call `showMaximized()`.
4. Run the event loop; let `code` be its exit code (0 after the window closes normally, or the value passed to `QApplication.exit`).
5. Shut down without returning: destroy the window while the application still exists (`deleteLater()`, deliver pending deferred deletions, drop the reference, run a garbage collection), which also writes its pending QSettings values; call `logging.shutdown()`; flush stdout and stderr; end the process with `os._exit(code)`. Exit handlers (`atexit`) and interpreter finalization are skipped on purpose: tearing down Qt objects there crashed the process after a clean exit (segmentation fault, exit status 139, in most runs). Neither the application nor the library registers exit handlers or finalizers. `main()` therefore never returns (`NoReturn`); `cli_entry` and the console scripts end with the same exit code.

### 1.2 Environment

| Variable | Effect |
|---|---|
| `SPLINE_FITTER_LOG_DIR` | Directory for optimizer CSV logs (section 17); read when each logger is created. |

### 1.3 Architecture

| Module | Responsibility |
|---|---|
| `bspline_varpro.basis` | Knot vectors, basis matrices, bending energy, centripetal parameters, knot insertion. |
| `bspline_varpro.reparam` | Softplus reparameterization of data parameters. |
| `bspline_varpro.chain` | Chain data, chain parameters, projection onto a chain. |
| `bspline_varpro.constraints` | Junction and end constraints `C(θ) P = d(θ)`. |
| `bspline_varpro.problem` | Variable-projection residual and exact Jacobian; KKT solver. |
| `bspline_varpro.fitting` | `fit()`, Levenberg-Marquardt, continuation, `FitResult`. |
| `bspline_varpro.tolerance` | `fit_to_tolerance()`, knot growth, knot polishing, subsampling. |
| `bspline_varpro.joint`, `.refinement` | Joint (parameters + control points) Gauss-Newton/LM with Schur complement. |
| `bspline_varpro.diagnostics` | Orthogonal projection, curvature, fairness, junction measures. |
| `bspline_varpro.airfoil` | NACA 4-digit generator, default tolerance, leading edge, thickness/camber, resampling, TE segment. |
| `bspline_varpro.export` | STEP, IGES, DXF writers, merge, sampling, arc length. |
| `spline_fitter.io` | Airfoil and spline `.dat` files. |
| `spline_fitter.geometry` | Leading edge, section split, parameterizations, bounding box. |
| `spline_fitter.optimization.*` | Models, Direct LSQ, VarPro adapter, dispatch, evaluation counters, CSV logs. |
| `spline_fitter.analysis` | Fit report, foot points, default tolerance wrapper, report HTML. |
| `spline_fitter.project_io` | `.asfproj` encode/decode. |
| `spline_fitter.formatting` | Display formatting. |
| `spline_fitter.spline` | `BSplineCurve2D`, curvature helpers, fairness metrics. |
| `spline_fitter.gui.*` | Window, layout, widgets, theme, motion, plots, live preview, background worker, fitting, sweeps, project manager, explorer, file actions, batch, settings history. |

Threading: fits run on one worker thread (section 19.12). The worker never touches widgets. Module-level state (evaluation counters, theme tokens) is process-global.

---

## 2. B-spline mathematics

### 2.1 Knot vectors

A knot vector for `n` control points of degree `p` has `n + p + 1` entries. All knot vectors in the system are clamped on `[0, 1]`: `p + 1` zeros, interior knots, `p + 1` ones.

**`open_uniform_knots(p, n)`** (library `basis.open_uniform_knots`; application `spline.create_open_uniform_knots`, identical output):

1. Library: if `p < 1` raise `ValueError("Degree must be at least 1, got {p}.")`. Both: if `n <= p` raise `ValueError("Number of control points ({n}) must be greater than degree ({p}).")` (the application's message has no final period).
2. `k = n - p - 1` interior knots `i / (n - p)` for `i = 1..k` (computed as `i * (1.0 / (n - p))`).
3. Result: `[0]*(p+1) + interior + [1]*(p+1)`.

Example: `open_uniform_knots(3, 5) = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]`.

**`clamped_knots(p, interior=())`**: sort the interior values ascending and wrap them in `p + 1` zeros and ones.

**`interior_knots(t, p) = t[p+1 : len(t)-p-1]`**.

**`validate_knots(t, p)`** (library; required for every knot vector given to the VarPro problem). Raise `ValueError` with the quoted message when:

| Condition | Message |
|---|---|
| not 1-D, fewer than `2(p+1)` values, or any non-finite | `"Knot vector needs at least {2(p+1)} finite values."` |
| any `t[i+1] < t[i]` | `"Knot vector must be nondecreasing."` |
| `t[:p+1]` not all exactly `0.0` or `t[-p-1:]` not all exactly `1.0` | `"Knot vector must be clamped on [0, 1] (degree + 1 zeros and ones)."` |
| interior knots exist and first `<= 0` or last `>= 1` | `"Interior knots must lie strictly inside (0, 1)."` |
| an interior value repeated more than `p` times | `"Interior knot multiplicity must not exceed the degree."` |

**Application check** `validate_knot_vector(t, p, n)` (used only by `BSplineCurve2D`): true iff `len(t) == n + p + 1`, `diff(t) >= 0` everywhere, `t[:p+1]` all close (NumPy `allclose`, rtol 1e-5, atol 1e-8) to `t[0]` and `t[-(p+1):]` all close to `t[-1]`.

**`averaged_knots(t, n, p)`** (exported, not used by the application): `m = len(t)`; if `n <= p` raise `ValueError("Control-point count must exceed the degree.")`; with `k = n - p - 1`, if `k == 0` return `clamped_knots(p)`. `ratio = m / (n - p)`; for `j = 1..k`: `i = floor(j·ratio)`, `α = j·ratio - i`, `i = clip(i, 1, m-1)`, value `(1-α) t[i-1] + α t[i]`. Clip values to `[1e-9, 1 - 1e-9]`, apply running maximum, then `_limit_multiplicity`: for every distinct value occurring more than `p` times (with sorted distinct values `v` and the run at index `r`), `low = (v[r-1]+v[r])/2` (or 0 for the first), `high = (v[r]+v[r+1])/2` (or 1 for the last), and the run's `c` entries become `low + (high-low)·i/(c+1)` for `i = 1..c`.

**`insert_knot(t, P, p, u)`** (Boehm, exported): require `t[p] < u < t[-p-1]` else `ValueError("Knot {u} must lie strictly inside ({t[p]}, {t[-p-1]}).")`; require the count of `t == u` below `p` else `ValueError("Knot {u} already has multiplicity {p}.")`. `k = searchsorted(t, u, "right") - 1`, clamped to `[p, len(P)-1]`. New points: `Q[0 .. k-p] = P[0 .. k-p]`; `Q[k+1 ..] = P[k ..]`; for `i = k-p+1 .. k`: `a = (u - t[i]) / (t[i+p] - t[i])`, `Q[i] = (1-a) P[i-1] + a P[i]`. New knots: `u` inserted at position `k + 1`.

### 2.2 Evaluation

- A curve is `C(u) = Σ_j B_{j,p}(u) P_j` with B-spline basis functions of the knot vector (Cox-de Boor). Domain `[t[p], t[-p-1]] = [0, 1]`; `C(0) = P_0`, `C(1) = P_{n-1}`; the basis sums to 1.
- On a knot interior to the domain, evaluation uses the span to the right (SciPy convention); at `u = 1` the last span.
- **Basis matrix** `basis_matrix(u, t, p)`: `N[i, j] = B_j(clip(u_i, 0, 1))`, dense.
- **Derivative basis** `basis_derivative_matrix(u, t, p, r=1)`: `r`-th derivatives of each basis function at `clip(u, 0, 1)`; all zeros when `r > p`. Evaluated with `extrapolate=False`.
- Library curve objects are SciPy-style `BSpline(t, c, k)` with `t` knots, `c` coefficients `(n, dim)`, `k` degree; evaluation `curve(u, nu=r)` gives the `r`-th derivative.

### 2.3 Bending energy

`bending_energy_matrix(t, p, quad_order=8)` returns `n × n` `S` with `S_ij = ∫_0^1 B_i''(u) B_j''(u) du`:

1. If `p < 2`, return zeros.
2. For each pair of consecutive values `[a, b]` of the sorted distinct values of the whole knot vector, with `b - a >= 1e-14`: 8-point Gauss-Legendre nodes `ξ`, weights `w`; points `u = (a+b)/2 + (b-a)/2·ξ`; add `Bᵀ diag(w (b-a)/2) B` where `B` holds the second derivatives of all basis functions at `u`.
3. If `trace(S) > 1e-15`, scale `S` by `n / trace(S)`.

Results may be cached by `(t, p, order)`.

### 2.4 Curvature

For a planar curve with derivatives `(x', y')`, `(x'', y'')`, `(x''', y''')` with respect to its parameter, `v² = x'² + y'²`, `a = x'y'' - y'x''`, `b = x'y''' - y'x'''`, `c = x'x'' + y'y''`:

- signed curvature `κ = a / v³` (positive = counterclockwise turning);
- `dκ/ds = (b v² - 3 a c) / v⁶`.

Guards differ per caller and are part of the behaviour:

| Function | Guard |
|---|---|
| `spline.compute_signed_curvature(dx,dy,d2x,d2y)` | `0` where `v³ <= 1e-12` |
| `spline.compute_curvature_derivative(...)` | `0` where `v⁶ <= 1e-24` |
| `diagnostics.curvature`, `curvature_slope` | no guard (division by zero gives inf/nan); raise `ValueError("Curvature measures require planar (2D) curves.")` unless coefficients are `(n, 2)`. Third derivative taken as zero when `p < 3`. |
| GUI comb `_curvature_and_slope` | `0` where `v² <= 0` |

### 2.5 Gauss-Legendre quadrature

Nodes and weights of `numpy.polynomial.legendre.leggauss(order)` on `[-1, 1]`, mapped to `[a, b]` by `u = (a+b)/2 + (b-a)/2·ξ`, weight `(b-a)/2·w`. Orders: 8 (bending energy, roughness), 5 (application `compute_curve_energy`), 16 (export arc length).

### 2.6 Parameterizations of a polyline

- **Centripetal** (library `centripetal_parameters(P, α=0.5)`): fewer than 2 points -> zeros of the same length. `t_0 = 0`, `t_i = Σ_{k<=i} |P_k - P_{k-1}|^α`; if `t_last <= 0` return `linspace(0, 1, N)`, else `t / t_last`. `α = 1` is chord length, `α = 0` uniform.
- **Application** `geometry.centripetal_parametrize(points, alpha=0.5)`: same, returning a list; empty input -> `[]`; one point -> `[0.0]`.
- **Arc length** `geometry.compute_arc_length_parameters(points)`: empty -> `[]`; one point -> `[0.0]`; cumulative Euclidean segment lengths divided by the total; if the total is `<= 0`, `[i/(N-1)]`.

---

## 3. Reading airfoil files (`spline_fitter.io`)

Entry points: `parse_airfoil_data(path)` reads the file (3.1) and calls `parse_airfoil_text(text, filename_name)` with `filename_name = splitext(basename(path))[0]`. `parse_airfoil_text(text, filename_name=None)` returns `(filename_name, content_name, coords, precision)` or `(None, None, None, 0)` when nothing is found. `coords` is a list of `(float, float)`.

### 3.1 Decoding (`read_text`)

1. Read all bytes.
2. Starts with `FF FE` or `FE FF`: decode as UTF-16 (BOM-detected endianness), `errors="replace"`.
3. Else if a zero byte occurs in the first 200 bytes (`head`): big-endian if `head[0::2].count(0) > head[1::2].count(0)`, else little-endian; decode UTF-16-BE/LE with `errors="replace"`.
4. Else decode UTF-8 accepting a leading BOM (`utf-8-sig`); on a decoding error decode Windows-1252 with `errors="replace"`.

### 3.2 Lines

1. Remove leading `U+FEFF` characters from the start of the whole text (not per line). Split into lines (Python `splitlines`).
2. For each line: `content = line.split("#", 1)[0].strip()`; keep the line when `content` is non-empty, as the pair `(content, line.strip())`.
3. No kept line: return `(None, None, None, 0)`.
4. `content_name = filename_name`; `name_line` = the full stripped text of the first kept line.

### 3.3 Number fields (`_number_fields(content)`)

1. Replace every `;` by a space.
2. Insert a space at every position preceded by a digit or `.` and followed by `+` or `-` that is followed by a digit or `.` (regex lookaround `(?<=[0-9.])(?=[+-][0-9.])`). `0.95000-0.01200` becomes `0.95000 -0.01200`. An exponent sign follows `e`/`d`, a letter, and is not split.
3. If the text contains `,`: when it has at least two whitespace-separated fields and no `.`, replace every `,` by `.` (decimal comma, `0,5 0,01`); otherwise replace every `,` by a space.
4. Split on whitespace. From each field strip leading/trailing `(`, `)`, `"`, `'` (a field that becomes empty is kept as an empty string, so `() 1` gives `["", "1"]`, which is not a row). Replace a `d` or `D` that is preceded by a digit or `.` and followed by an optional sign and a digit by `e` (Fortran exponent).

### 3.4 Rows

For each kept line `i` (0-based), `parts = _number_fields(content)`. If `parts` has at least two entries and both `float(parts[0])` and `float(parts[1])` succeed (Python `float` syntax, which also accepts `inf`/`nan`), the line is a **row** `(x, y, x_text, y_text)`; if `i == 0`, set `first_is_row = True`. Otherwise, if `i == 0`, `content_name = _name_of_line(name_line)`; other non-rows are skipped.

### 3.5 Lednicer and numeric names

Definitions on the list `rows`:

- `is_point_counts(x, y)`: evaluated left to right with short-circuit: `x >= 2`, then `y >= 2`, then `x == int(x)`, then `y == int(y)`. Comparisons with NaN are false (IEEE). `int()` of an infinite value raises `OverflowError`, which is not caught: a first line whose two numbers are both `>= 2` with one of them infinite (for example `inf 5`, `5 inf`, `1e400 3`) makes `parse_airfoil_text` raise `OverflowError("cannot convert float infinity to integer")`; the window then shows `Failed to load: cannot convert float infinity to integer` (19.11.1).
- `is_lednicer(rows)`: `len(rows) > 4`, `is_point_counts(rows[0])`, `|rows[0].x + rows[0].y - (len(rows)-1)| <= max(2, (len(rows)-1) // 10)`, and with `xs` = x of `rows[1:]`: `xs[0] - min(xs) < max(xs) - xs[0]`.
- `is_stray_first_row(rows)`: `len(rows) > 4` and `rows[0]` lies outside the box `[min-m, max+m]` in x or y of `rows[1:]`, where `m = 0.5·max(x range, y range)` of `rows[1:]` (bounds inclusive: inside means `min-m <= v <= max+m` in both axes).

Procedure:

1. If `first_is_row` and not `is_lednicer(rows)` and (`is_lednicer(rows[1:])` or `is_stray_first_row(rows)`): `content_name = name_line` (unescaped full line) and drop `rows[0]`.
2. If `is_lednicer(rows)`: `n_upper, n_lower = int(rows[0].x), int(rows[0].y)`; drop `rows[0]`; `coords = lednicer_to_selig(points of rows, n_upper, n_lower)`:
   1. if `n_upper + n_lower != len(coords)`: `steps[i] = coords[i+1].x - coords[i].x`; `n_upper = (index of the first minimum of steps) + 1`;
   2. `upper = coords[:n_upper]`, `lower = coords[n_upper:]`;
   3. if `lower` is non-empty and `lower[0] == upper[0]` exactly, drop `lower[0]`;
   4. result `reversed(upper) + lower`.
3. Otherwise `coords` = the points of all rows.
4. If `coords` is empty, return `(None, None, None, 0)`.

### 3.6 Precision

`get_precision(s)`: lower-case, replace `d` by `e`, split at the first `e` into mantissa and exponent; `decimals` = number of characters after `.` in the mantissa (0 if no `.`); result `decimals - int(exponent)` (exponent `""` counts as 0; a non-integer exponent returns `decimals`). Examples: `"0.0123"` -> 4, `"1.2346e-03"` -> 7, `"1.23e+01"` -> 1, `"5"` -> 0.

`file_precision(texts)` (a text counts as e-notation when its lower-cased form with `d` replaced by `e` contains `e`) over the `x_text` and `y_text` of the rows that remain after 3.5 (header and name rows excluded): `plain` = precisions of texts without `e`/`d`; `scientific` = of those with; `candidates = [max(plain)] (if any) + [min(scientific)] (if any)`; result `max(0, min(candidates))`, or 0 without candidates.

### 3.7 Outliers and normalization

1. **Outliers** `filter_coordinate_outliers(coords, threshold_factor=5.0)`: fewer than 4 points -> unchanged. Sort the points lexicographically (by x then y); `median_point` = element `len // 2`. `d_i` = Euclidean distance of each point to it. With `sd = sorted(d)`: `q1 = sd[len//4]`, `q3 = sd[3·len//4]`, threshold `q3 + 5·(q3 - q1)`. Keep points (in original order) with `d_i <= threshold`. If fewer than `0.8·len` remain, return the original list.
2. **Normalize** (`_normalize`, returns coords and a length factor `scale`):
   1. fewer than 3 points: unchanged, `scale = 1`;
   2. if more than half the points have `|x| > 2` or `|y| > 2`: divide all coordinates by 100, `scale = 0.01`;
   3. `te = ((x_first + x_last)/2, (y_first + y_last)/2)`, `le_x = min x`, `chord = te_x - le_x`;
   4. `chord < 1e-6`: return as is (with the current `scale`);
   5. `|le_x| <= 5e-3` and `|te_x - 1| <= 5e-3` and `|te_y| <= 5e-3`: return as is (already unit chord, values kept exactly);
   6. else each point becomes `((x - le_x)/chord, (y - te_y)/chord)` and `scale = scale / chord`.

   The data are never rotated.
3. **Precision after scaling** `scaled_precision(d, scale)`: if `scale` is not finite or `<= 0`, return `d`; `v = d - log10(scale)`; `r = round(v)`; result `max(0, int(r if |v - r| < 0.05 else floor(v)))`.
4. Return `(filename_name, content_name, coords, scaled_precision(file_precision, scale))`.

### 3.8 Name lines

- `_name_line(name)` (writing): `name = str(name).strip()`; if it matches `^\\*#` (zero or more backslashes then `#`), prepend one backslash.
- `_name_of_line(line)` (reading): if it matches `^\\+#`, remove the first character.

### 3.9 Library reader

`bspline_varpro.airfoil.read_dat(path)` is a simpler reader the application never calls (section 22 item 5). It returns `(name, points)` with `points` an `(m, 2)` array; it does not strip comments, filter outliers or normalize.

1. Decode the file as UTF-8 with `errors="replace"`; split into lines (Python `splitlines`). `name` = the file name without its last suffix.
2. For each line `i` (0-based, counting every line, empty ones too): `parts = line.replace(",", " ").split()`; if `float(parts[0])` and `float(parts[1])` both succeed, it is a row (and if `i == 0`, `first_is_row = True`); otherwise, if `i == 0`, `name = line.strip()`.
3. No rows: `ValueError("No coordinates found in {path}.")`.
4. `(n_upper, n_lower)` = row 0; `integer_first` = both equal their rounded value; with `x` = the x of rows `1..`, `starts_at_le` = `x` is non-empty and `x[0] - min(x) < max(x) - x[0]`.
5. Lednicer, when there are more than 4 rows, `min(n_upper, n_lower) >= 2`, `integer_first` and `starts_at_le`: drop row 0; if `n_upper + n_lower` differs from the number of rows left, `n_upper = (index of the first minimum of diff(x)) + 1`; `upper` = the first `int(n_upper)` rows, `lower` = the rest; if `lower[0]` is close to `upper[0]` (NumPy `allclose` defaults, rtol 1e-5, atol 1e-8; the application's reader in 3.5 compares exactly), drop `lower[0]`; result `reversed(upper) + lower`.
6. Otherwise, when `first_is_row`, `integer_first` and there are more than 4 rows: with `lo`, `hi` the per-axis minimum and maximum of rows `1..` and `margin` = the larger of the two ranges `hi - lo`, if row 0 is below `lo - margin` or above `hi + margin` in either axis, `name` = line 0 stripped and row 0 is dropped (a numeric title).

---

## 4. Airfoil data and section split (`spline_fitter.geometry`)

### 4.1 Leading edge

`find_leading_edge_index(points)`: empty input raises `ValueError("Cannot find leading edge in empty coordinate list")`. `te` = midpoint of first and last points. Return the index of the largest squared distance to `te`; the first index wins ties (strict `>`; initial best `-1`).

Library `leading_edge_index(points)` uses `argmax` of the distance (same tie rule). Library `split_at_leading_edge(points)` raises `ValueError("Leading edge found at an end; expected Selig order (TE -> LE -> TE).")` when the index is 0 or last, else returns `(points[:i+1], points[i:])`.

### 4.2 `split_into_sections(coords, S)`

Returns `(sections, junction_indices)` (tuples of point tuples; indices into `coords`).

1. `S < 1`: `ValueError("num_sections must be >= 1, got {S}")`. `len(coords) < S + 1`: `ValueError("Need at least {S+1} points for {S} sections, got {len}")`.
2. `S = 1`: `((all,), ())`.
3. `S = 2`: `le = min(max(find_leading_edge_index(coords), 1), len - 2)`; sections `coords[:le+1]`, `coords[le:]`; junctions `(le,)`.
4. `S` odd `>= 3`: `u = compute_arc_length_parameters(coords)`; `J = junctions(u, S)`; sections are `coords[J_{k-1} : J_k + 1]` with `J_0 = 0` and the last section `coords[J_last:]`; junction indices `J`.
5. `S` even `>= 4`: `le = find_leading_edge_index(coords)` (no clamp); `upper = coords[:le+1]`, `lower = coords[le:]`, `h = S/2`. If `len(upper) < h+1` or `len(lower) < h+1`: `ValueError("Need at least {h+1} points on each side of the leading edge for {S} sections, got {len(upper)} and {len(lower)}")`. Upper sections: `junctions(arc params of upper, h)` splits `upper` as in 4 (indices are into `coords`); then the junction `le`; lower sections likewise, their junction indices offset by `le`.

`junctions(u, S)`:
1. For `i = 1..S-1`: the index whose `u` is nearest to `i/S` (first on ties).
2. Forward, for the junction list `J_1 .. J_{S-1}` from step 1 (this list's own numbering, unrelated to the `J_0 = 0` start index used in step 4 of `split_into_sections`): `J_1 = max(J_1, 1)`, then `J_k = max(J_k, J_{k-1} + 1)` for `k = 2..S-1`.
3. Backward: `J_{S-1} = min(J_{S-1}, len(u) - 2)`, then `J_k = min(J_k, J_{k+1} - 1)` for `k = S-2 .. 1`.

`split_at_leading_edge(coords)` (application) = `split_into_sections(coords, 2)` returned as `(upper, lower, le_index)`.

### 4.3 `AirfoilData`

Immutable record: `coordinates` (tuple of points), `sections` (tuple of tuples), `junction_indices`, `num_sections`, `filename = ""`, `name = ""`, `precision = 6`. `num_points = len(coordinates)`. `create_airfoil_data(coords, num_sections=2, filename="", name="", precision=6)` splits by 4.2.

### 4.4 Other geometry helpers

- `compute_bounding_box(points)`: empty raises `ValueError("Cannot compute bounding box of empty point set")`; else `BoundingBox(min x, max x, min y, max y)`.
- `BoundingBox`: `center`, `width = x_max - x_min`, `height`, `diagonal = hypot(width, height)`, `scaled(f)` (half sizes times `f` about the centre), `x_bounds()`, `y_bounds()`.
- `interpolate_along_polyline(points, t_params, query_t)`: empty points -> `(0.0, 0.0)` per query; one point -> that point per query; else per query: `<= t[0]` -> first point; `>= t[-1]` -> last point; else `idx = searchsorted(t, q) - 1` clamped to `[0, N-2]`, linear interpolation with weight `(q - t0)/(t1 - t0)` (0 if `t1 <= t0`).

---

## 5. Configuration and result models (`spline_fitter.optimization.models`)

### 5.1 `ContinuityMode`

Members `G0, G1, G2, G3` (Python `Enum` with `auto()` values 1..4; persisted by name). `order` = the digit. `minimum_degree`: G0 -> 1, G1 -> 1, G2 -> 2, G3 -> 3.

### 5.2 Minimum control points

`minimum_control_points(orders, p) = max(p + 1, max over consecutive pairs (a, b) of [0, *orders, 0] of (a + b + 2))`. With one section (`orders` empty) this is `max(p+1, 2)`.

Library version per section: `ends = [start, *junction orders, end]` where a junction's order is `max(order, 1 if it has a tangent else 0)` and an end counts 1 if it has a tangent else 0; section `s` needs `max(p + 1, ends[s] + ends[s+1] + 2)`. The application sets no tangents, so both agree.

### 5.3 `OptimizerType`

| Member | Value | `display_name` | `is_algebraic` |
|---|---|---|---|
| `DIRECT_LSQ` | `"direct-lsq"` | `Direct LSQ` | true |
| `GN_VARPRO` | `"gn-varpro"` | `VarPro LM` | false (`is_iterative_algebraic` true) |

Tooltips (shown on the optimizer segments, lines separated by `\n`):

- Direct LSQ: `Direct LSQ (single-step, no iteration).` / `One constrained least-squares solve, with fixed trailing-edge` / `end points and junctions fixed or free (Advanced).` / `Fastest option (~1ms), moderate accuracy (~1e-4).` / `Ideal for dense data or real-time preview.`
- VarPro LM: `Levenberg-Marquardt Variable Projection (bspline-varpro).` / `Tolerance mode: fewest control points within a maximum` / `orthogonal deviation, non-uniform knots.` / `Geometric (G) or parametric (C) continuity per junction.` / `Recommended for CAD-quality results.`

### 5.4 `SplineConfig` (immutable)

| Field | Default |
|---|---|
| `degree` | 5 |
| `num_control_points` | 25 |
| `num_sections` | 2 |
| `continuity` | `G1` |
| `geometric_continuity` | `False` |
| `junction_continuity` | `()` |

Construction (in this order):

1. `num_sections < 1`: `ValueError("num_sections must be >= 1, got {S}")`.
2. `junction_continuity` becomes a tuple of `s.strip().upper()`.
3. If `S > 1`, `junction_continuity` empty and `degree < continuity.minimum_degree`: `ValueError("{continuity name} continuity requires degree >= {continuity.minimum_degree}, got {degree}")` (for example `G2 continuity requires degree >= 2, got 1`).
4. If `junction_continuity` non-empty: length must be `S - 1`, else `ValueError("{S} sections have {S-1} junction(s), but {k} continuity entries were given")`; each entry must be 2 characters, first in `CG`, second in `0123`, else `ValueError("Invalid junction continuity {spec!r}; use e.g. G2 or C1")`; `int(spec[1]) > degree`: `ValueError("{spec} continuity requires degree >= {digit}")`.

`junction_specs()`: the entries if given; else `[("G" if geometric_continuity else "C") + str(order)] * (S - 1)`.
`minimum_control_points` property: 5.2 over the digits of `junction_specs()`.

### 5.5 `OptimizationConfig` (immutable)

| Field | Default | Meaning |
|---|---|---|
| `optimizer` | `GN_VARPRO` | method |
| `smoothing_factor` | `None` | retired; only in batch naming |
| `early_stop_patience` | 50 | retired; only in batch naming |
| `relative_tolerance` | `1e-6` | retired; only in batch naming |
| `optimize_le_position` | `False` | junction positions free (all junctions) |
| `smoothing_weight` | `1e-3` | bending-energy weight, VarPro fixed-count fits |
| `custom_parameters` | `None` | fixed data parameters (Direct LSQ) / initial parameters (VarPro); never set by the GUI |
| `parameterization` | `"centripetal"` | Direct LSQ start: `"centripetal"` or `"arc_length"` |
| `tolerance` | `None` | maximum deviation; `None` = fixed count |
| `max_points_per_section` | `None` | VarPro subsampling |
| `knot_placement` | `"free"` | `"uniform"` or `"free"` |
| `grow_control_points` | `True` | start from the fewest control points |

Validation: tolerance not `None` and not `> 0`: `ValueError("tolerance must be positive, got {t}")`; `max_points_per_section < 2`: `ValueError("max_points_per_section must be at least 2, got {v}")`; bad placement: `ValueError("knot_placement must be 'uniform' or 'free', got {v!r}")`; bad parameterization: `ValueError("parameterization must be 'centripetal' or 'arc_length', got {v!r}")`.

### 5.6 `FitResult` (application)

Fields: `section_control_points` (list of lists of `(x, y)`; tuples converted to lists), `degree`, `final_error`, `num_iterations`, `num_evaluations`, `elapsed_seconds`, `converged = True`, `early_stopped = False`, `cancelled = False`, `section_knots = None` (stored as lists of floats), `report = None`, `rms_error = None`.

- `knots(i)`: `section_knots[i]` as array, else `open_uniform_knots(degree, len(section i))`.
- `curves()`: `BSplineCurve2D(degree, ctrl_i, knots(i))` per section.
- `comparison_error`: `report["rms_deviation"]` if the report exists and has a non-`None` value; else `rms_error` if not `None`; else `final_error`.
- `num_control_points = Σ len(section) - (S - 1)`.

### 5.7 Other records

- `IterationRecord(iteration_number, error_value, elapsed_seconds=0.0)`; alias `EvaluationRecord`.
- `ProjectMetadata(format_version="1.0.0", created="", modified="", app_version="", platform="")`.
- `OptimizationRun(run_id, timestamp, spline_config, optimization_config, result, loss_history, section_control_points, number=0)`.
- `ProjectFile(metadata, airfoil, optimization_runs=[])`.

---

## 6. Application curve class and fairness metrics (`spline_fitter.spline`)

### 6.1 `BSplineCurve2D(degree, control_points, knot_vector=None)`

- Control points stored as float tuples. Without knots: `open_uniform_knots(p, n)`. With knots: `validate_knot_vector` (2.1) must pass, else `ValueError("Invalid knot vector for given degree and control points")`.
- `parameter_range = (t[p], t[-p-1])`.
- `evaluate(u)` -> `(x, y)`; `evaluate_array(u)` -> `(xs, ys)`; evaluation does not extrapolate (outside the domain gives NaN).
- `sample(m=100)`: `evaluate_array(linspace(0, 1, m))`.
- `derivative(r=1)`: `r < 1`: `ValueError("Derivative order must be >= 1")`; `r > p`: `ValueError("Cannot compute order-{r} derivative of degree-{p} spline")`; else a curve of degree `p - r` from the derivative splines.
- `evaluate_derivatives(u, max_order=2, min_order=0)`: dict with keys `x,y` (order 0), `dx,dy`, `d2x,d2y`, `d3x,d3y` for orders `min_order..min(max_order, 3)`; orders above `p` are zero arrays.
- `compute_arc_length(curve, m=1000)`: trapezoid rule on `|C'(u)|` at `linspace(0, 1, m)` with step `1/(m-1)`.

### 6.2 `compute_polygon_metrics(P)`

All four 0 for fewer than 2 points. `l_i = |P_{i+1} - P_i|` (finite ones only; none -> zeros). `m = mean(l)`; if `m < 1e-12` or not finite, `m = 1`.

- `length_variance = (std(l) / m)²` (population std);
- `anti_bunching = mean(max(0, 0.5 - l_i/m)²)`;
- `polygon_bend = mean(|P_i - 2P_{i+1} + P_{i+2}|²) / m²` over finite values (0 for fewer than 3 points);
- `end_handles = 0.5·((l_0/m - 1)² + (l_last/m - 1)²)`.

### 6.3 `compute_curve_energy(curve, quadrature_order=5)`

Returns `{length, curvature_squared, curvature_gradient}` (zeros for `None`, fewer than 2 knots, or non-finite knots). Gauss order is `max(2, order)`. For every consecutive knot pair `(a, b)` of the full knot vector with `b > a + 1e-12`: nodes on `[a, b]`; `v = max(|C'|, 1e-12)`; `L += Σ w v`; if `p >= 2`: `κ = (v × C'') / v³`, accumulate `Σ w κ² v`; jerk `C'''` (zero if `p < 3`); `dκ/ds` by 2.4 with `v` floored; accumulate `Σ w (dκ/ds)² v`. If `L <= 1e-12` or not finite return zeros; `curvature_squared = I_κ / L` (0 if `I_κ <= 0`), `curvature_gradient = I_g / L` (0 if `I_g <= 0`).

These metrics are a public API; the GUI does not show them.

---

## 7. Junction and end constraints (`bspline_varpro.constraints`)

### 7.1 End derivative coefficients

For a clamped curve, `C^(r)(1)` depends only on `P[-r-1:]` and `C^(r)(0)` only on `P[:r+1]`. `end_derivative_coefficients(t, p, r)` returns `(coef_end, coef_start)`, each of length `r+1`, such that `C^(r)(1) = coef_end · P[-r-1:]` and `C^(r)(0) = coef_start · P[:r+1]`. `r = 0` returns `([1], [1])`; `r > p` raises `ValueError` (the derivative order exceeds the spline order). Otherwise evaluate the `r`-th derivative of the spline whose coefficients are the identity matrix at `t[-1]` (take the last `r+1` entries) and at `t[0]` (first `r+1`).

### 7.2 Junction specs

`Junction(order=1, geometric=True, fixed=True, tangent=None)`; `label = ("G" if geometric else "C") + str(order)`. `End(fixed=True, tangent=None)`.

`parse_junction(spec, fixed=True)`: a `Junction` passes through; an integer (not bool) `k` -> `Junction(k, geometric=False, fixed)`; a 2-character string with first letter in `CGcg` and a digit -> geometric iff the letter is `G`/`g`; anything else: `ValueError("Invalid continuity {spec!r}; use e.g. 'G2', 'C1', 2 or a Junction.")`. Order outside `0..3`: `ValueError("Continuity order must be between 0 and 3.")`.

`parse_junctions(continuity, J, fixed)`: a list or tuple must have `J` entries (`ValueError("Expected {J} continuity entries, got {k}.")`); a single spec is repeated `J` times.

### 7.3 Shape parameters (geometric junctions)

A geometric junction of order `k >= 1` links the derivatives of ending section `a` at `u = 1` and starting section `b` at `u = 0` through a reparameterization with `φ' = α > 0`, `φ'' = β`, `φ''' = γ`:

- G1: `C_a' = α C_b'`
- G2: `C_a'' = α² C_b'' + β C_b'`
- G3: `C_a''' = α³ C_b''' + 3αβ C_b'' + γ C_b'`

`ratio_j = L_a / L_b`, where `L_s` is the polyline length of the data of section `s` (or 1 if zero). `ALPHA_RANGE = 10`.

- `alpha_range(ratio)`: `low = min(ln ratio, 0) - ln 10`, `high = max(ln ratio, 0) + ln 10`; centre `c = (low+high)/2`, half width `h = (high-low)/2`.
- `alpha(a) = exp(c + h tanh(a/h))`; `dα/da = α (1 - tanh²(a/h))`.
- Inverse `alpha_parameter(α) = h · atanh(clip((ln α - c)/h, -0.999, 0.999))`.
- `β`, `γ` are raw unbounded parameters. Tangent magnitudes (unused by the application) are `τ = softplus(raw)`, `dτ = sigmoid(raw)`.

Parameter vector `θ` layout, in order: for each junction `j` in order that is geometric with order `>= 1`: `α`-parameter, then `β` if order `>= 2`, then `γ` if order 3; then one raw magnitude per tangent condition (start tangent: node 0 side "start"; each junction with a tangent: node `j+1` side "start", plus node `j` side "end" when the junction order is 0; end tangent: last node side "end").

- `initial_params()`: each `α`-parameter = `alpha_parameter(ratio_j)` (alpha starts at the length ratio); `β = γ = 0`; tangent raw = `inverse_softplus(L_node)`, where a tangent condition's "node" is the index of the section it acts on and `L_node` is that section's polyline length as defined above (1 if zero).
- `theta_from_parametric(tangent_theta)`: `α`-parameters at `alpha_parameter(1.0)`, `β = γ = 0`, tangents from the given values in order.
- `shape_values(θ)`: `(α, β, γ)` per junction; `(1, 0, 0)` for parametric or order-0 junctions.
- `alpha_deviations(θ)`: per `α` slot, `ln(α/ratio)` and its row derivative `dα/da / α` in the slot's column.

### 7.4 Constraint matrix `ChainConstraints.build(θ)`

Columns: all control points of all sections stacked (section offsets `o_s = Σ_{r<s} n_r`, total `T`). Each row has a coefficient vector of length `T` and a value of length `dim`. Rows, in this order:

1. if `start.fixed`: `P_{o_0} = first data point of section 0`;
2. if `end.fixed`: `P_{T-1} = last data point of the last section`;
3. for each junction `j` (`a = j`, `b = j+1`):
   1. `P_end(a) - P_start(b) = 0`;
   2. if `fixed`: `P_end(a) = last data point of section a`;
   3. if order `>= 1`, with `s_r` = the row placing `coef_start` of order `r` on section `b` and `e_r` = the row placing `coef_end` of order `r` on section `a` (for parametric junctions `α = 1, β = γ = 0`):
      - order 1 row: `e_1 - α s_1 = 0`;
      - order 2 row (if order `>= 2`): `e_2 - α² s_2 - β s_1 = 0`;
      - order 3 row (if order 3): `e_3 - α³ s_3 - 3αβ s_2 - γ s_1 = 0`;
4. for each tangent condition: the order-1 coefficients at that node and side `= τ · unit direction`.

Derivatives with respect to `θ` (`dC_i`, `dd_i`; zero elsewhere): G1 row: `∂/∂a = -α' s_1`; G2 row: `∂/∂a = -2αα' s_2`, `∂/∂β = -s_1`; G3 row: `∂/∂a = -(3α² s_3 + 3β s_2) α'`, `∂/∂β = -3α s_2`, `∂/∂γ = -s_1`; tangent value: `∂/∂raw = sigmoid(raw) · unit`. (`α' = dα/da`.)

Construction checks: `len(junctions) == S - 1` (`ValueError("Need one junction per pair of consecutive sections.")`); junction order `> p`: `ValueError("Junction {j}: continuity order exceeds the degree.")`; section `s` with fewer control points than 5.2 (library rule): `ValueError("Section {s} needs at least {need} control points for its end conditions.")`; tangent directions must be nonzero finite vectors of length `dim` (`ValueError("Tangent direction must be a nonzero vector of length {dim}.")`).

`build_constraints(sections, knots, p, continuity="G1", fix_junctions=True)` builds `(C, d)` with `initial_params()`; a single knot vector is shared by all sections.

### 7.5 Direct LSQ coefficients

Direct LSQ (section 10) uses hand-derived closed forms for orders 1-3 on open uniform knots. They equal `end_derivative_coefficients` for every `(p, n, r)` with `r <= p` (verified for `p = 1..8`, `n = p+1..p+11`); an implementation may use 7.1.

---

## 8. VarPro core (`bspline_varpro`: `reparam`, `chain`, `problem`, `fitting`)

### 8.1 Chain data and chain parameters

Sections are arrays `(n_s, dim)`.

- `shared_junctions(sections)`: for each `s >= 1`, true when `max|last point of s-1 - first point of s| <= 1e-12 · max(E, 1e-300)`, where `E` is the largest per-axis range (ptp) of all section points stacked.
- `chain_data(sections)`: section 0, then each later section without its first point when shared. Result `D` with `N` rows.
- `section_ranges(sections)`: `(first, last)` row of each section in `D` (a shared junction row belongs to both neighbours).
- A chain parameter `u` lies in `[0, S]`. `locate(u, S)`: clip `u` to `[0, S]`; `section = min(floor(u), S-1)`; local parameter `u - section`. A point at an integer `u = s < S` is on section `s` at local 0; `u = S` is on section `S-1` at local 1.
- `groups(u, S)`: per section, the row indices and local parameters of the points located on it.
- `chain_parameters(t_sections, sections)`: for each section `s` with local parameters `t` (0..1): if it shares its first point with the previous section, drop `t[0]`; else if `s > 0` and `len(t) > 1`, set `t[0] = t[1]/2`; add `s`; concatenate.
- `initial_chain_parameters(None, sections) = chain_parameters([centripetal_parameters(sec) for each sec], sections)`; with a value, `validate_chain_parameters`.
- `validate_chain_parameters(params, sections, data=None)` (`data` is `chain_data(sections)`, computed when not passed; passing it changes nothing but the work): if `params` is non-empty and `params[0]` is a scalar, `params` must be 1-D, of length `N`, finite, nondecreasing, with `isclose(u[0], 0)` and `isclose(u[-1], S)` (NumPy `isclose` defaults), else `ValueError("Chain parameters need one value per point, increasing from 0 to the section count.")`. Otherwise it is one array per section (`S` of them), each 1-D, of the section's length, finite, nondecreasing, `isclose` to 0 and 1 at its ends, else `ValueError("Parameters must match each section and increase from 0 to 1.")`; converted with `chain_parameters`.
- `evaluate_chain(curves, u, nu=0)`: each point evaluated (or its `nu`-th derivative) on its located section.
- `chain_deviations(D, curves, samples_per_span=32) = distance_to_curves` (section 12.1).

### 8.2 Softplus reparameterization

- `softplus(x) = x` where `x > 20`, else `log1p(exp(clip(x, -700, 20)))`.
- `sigmoid(x) = 1/(1+e^{-|x|})` for `x >= 0`, `e^{-|x|}/(1+e^{-|x|})` otherwise.
- `inverse_softplus(y)`: `y' = max(y, 1e-15)`; result `y'` where `y' > 20`, else `log(expm1(clip(y', 1e-15, 20)))`.
- `t_to_eta(u)`: gaps `g = diff(u)`; `floor = max(1e-3 · mean(g), 1e-15)` (or `1e-15` with no gaps); `η = inverse_softplus(max(g, floor))`, length `N - 1`.
- `eta_to_t(η, S)`: replace NaN by 0, clip to `[-700, 700]`; `c = cumsum(softplus(η))`; `u_0 = 0`, `u_k = (S · c_{k-1}) / c_last` for `k = 1..N-1`, evaluated in that order (multiply first, then divide; with `S` an integer this still ends exactly at `S`). The ends are exactly 0 and `S` and `u` is strictly increasing.
- `eta_jacobian(η, u, S)`: `(N, N-1)` matrix `∂u_k/∂η_j = slope_j · ([j < k] - u_k/S)` with `slope_j = S·sigmoid(η_j)/Σ softplus(η)` where `|η_j| <= 700` (after NaN -> 0, clipped values in the sums), else 0.

### 8.3 KKT solver

`KKTSolver(A, C, reg)`: `K = [[A + reg·I, Cᵀ], [C, 0]]`, LU-factored with partial pivoting. Let `piv = |diag(U)|`. If any pivot is non-finite or `min(piv) <= size · eps · max(piv)` (`size` = order of `K`, `eps` = double machine epsilon; `min` and `max` of an empty set are 1), every later solve uses `pinv(K, rcond=1e-12)` instead (minimum-norm solution). `solve_full(rhs, d=None)` solves `K [x; μ] = [rhs; d]` (`d = None` means zeros) for any number of right-hand-side columns and returns `(x, μ)`; `solve` returns `x`.

### 8.4 `VarProProblem`

Signature: `VarProProblem(sections, knots, p, continuity="G1", fix_junctions=True, smoothing=1e-3, regularization=1e-12, start=End(), end=End())`.

Setup:

1. `D = chain_data(sections)`. Knots: one vector shared by all sections, or one per section (count mismatch: `ValueError("Need one knot vector per section.")`); each passes `validate_knots`.
2. `n_s = len(t_s) - p - 1`, offsets `o_s`, total `T`.
3. Penalty `Q = reg·I_T`; if `smoothing > 0`: `S_b = block_diag(bending_energy_matrix(t_s, p) for each s)`, `Q += smoothing·S_b`. `R = matrix_root(Q)`: the transposed lower Cholesky factor (so `RᵀR = Q`); if Cholesky fails, `R = diag(sqrt(max(λ, 0))) Vᵀ` from the symmetric eigen-decomposition `Q = V diag(λ) Vᵀ`.
4. Centroid `c = mean of the rows of D`.
5. Junctions from `parse_junctions(continuity, S-1, fix_junctions)`; `ChainConstraints(...)`; `θ0 = initial_params()`.
6. `spread = mean_i |D_i - c|²`. `alpha_pull = sqrt(smoothing · 3e-5 · spread)` if `smoothing > 0` else 0. Number of pull rows = number of `α` slots when `alpha_pull > 0`, else 0.
7. Speed rows: `chord_i = |D_{i+1} - D_i|` (`i = 0..N-2`); `moving_i = chord_i > 1e-12 · max(max chord, 1e-300)`. Reference parameters `r = initial_chain_parameters(None, sections)`. Gap section `sec_i = min(floor((r_i + r_{i+1})/2), S-1)`. Speed pairs: indices `i` (`0 <= i <= N-3`) with `moving_i`, `moving_{i+1}` and `sec_i == sec_{i+1}`. `log_chord_i = ln chord_i` where moving, else 0. For parameters `u`: `g = max(diff(u), 1e-300)`, `ls = ln g - log_chord`, `change_i = ls_{i+1} - ls_i` for each pair. `reference = change(r)`. `speed_pull = 1e-3 · sqrt(spread)`; number of speed rows = number of pairs when `speed_pull > 0`, else 0.

`solve(u, θ)` (cached for the most recent `(u, θ)` by exact value):

1. `N(u)`: `(N, T)`; the rows of points on section `s` hold `basis_matrix(local, t_s, p)` in columns `o_s .. o_{s+1}-1`, other entries 0.
2. `A = NᵀN + smoothing·S_b` (the KKT solver adds `reg·I`).
3. `(C, d) = build(θ)`.
4. Solve `KKT [O; μ] = [Nᵀ(D - 1cᵀ) ; d - C·1cᵀ]` where `1cᵀ` is the `T × dim` matrix whose rows are all `c`; `P = O + 1cᵀ`. (Since each basis row sums to 1, this is the minimizer of `|NP - D|² + (P - 1cᵀ)ᵀQ(P - 1cᵀ)` subject to `CP = d`, and the minimum-norm fallback does not depend on where the data lie.)

`residual(u, θ)`, concatenated in this order:

1. `N P - D`, point-major (`N · dim` values);
2. `R (P - 1cᵀ)`, point-major (`T · dim`);
3. pull rows `alpha_pull · ln(α_k / ratio_k)` (one per `α` slot);
4. speed rows: `excess = change(u) - reference`; row `= speed_pull · (excess - sign(excess)·1.0)` where `|excess| > 1.0`, else 0.

It also stores `last_control_points = P` and `last_rms_error = sqrt(Σ (NP - D)² / N)` (data residual only).

`jacobian(u, θ)` (exact, `len(residual) × N`):

1. `dN` like `N` with first-derivative basis functions; tangents `Tg = dN P`; misfit `M = D - N P`.
2. For each coordinate `k`: `dP_k = kkt.solve(B_k)` where column `j` of `B_k` is `dN_jᵀ M_{j,k} - N_jᵀ Tg_{j,k}` (`N_j`, `dN_j` row `j` as columns), with zero constraint values.
3. Data rows (row `i·dim + k`): `N dP_k`; penalty rows (row `N·dim + m·dim + k`): `R dP_k`; pull and speed rows zero. Then add `Tg_{i,k}` to column `i` of data row `i·dim + k`.
4. Speed rows of active pairs `i` (row index = first speed row + position of `i` among the pairs): column `i` `+= speed_pull/g_i`; column `i+1` `-= speed_pull/g_i + speed_pull/g_{i+1}`; column `i+2` `+= speed_pull/g_{i+1}`.

A point at an integer chain parameter takes the derivative of the section starting there (one-sided at a geometric junction).

`parameter_jacobian(u, θ)` (`len(residual) × len(θ)`; an empty-column matrix when `θ` is empty): with `μ` from the last solve, for each coordinate `k`, solve `KKT [dP; ·] = [-dC_iᵀ μ_k ; dd_{i,k} - dC_i P_k]` with one column per `θ_i`; data and penalty rows as above; pull rows = `alpha_pull ·` the `alpha_deviations` Jacobian; speed rows 0.

`split_control_points(P)`: per-section slices of the stacked rows.

### 8.5 Levenberg-Marquardt

`_lm_run(residual, jacobian, x0, max_nfev, tol, verbose)`:

1. `r = residual(x0)`, `nfev = 1`, `cost = ½ rᵀr`. `jacobian` is a function or `"2-point"` (forward differences: column `j` with step `h = sqrt(eps)·max(1, |x_j|)`, `len(x)` extra evaluations each time). `affordable(e) = (max_nfev is None) or (nfev + e <= max_nfev)`.
2. Finite differences and not `affordable(len(x))`: return status 0.
3. Linearize: `J`, `A = JᵀJ`, `g = Jᵀr`; `scale = max(diag A, 1e-300)`; `μ = 1e-3`, `ν = 2`; status 0.
4. While `affordable(1)`:
   1. `cost`, `A` or `g` not all finite: status -1, stop.
   2. `cos_j = |g_j| / (sqrt(max(A_jj, 0)) · sqrt(2·cost))` (0 where the column norm is 0). If `sqrt(2·cost) == 0` or `max cos <= tol` (the maximum of no values is 0): status 1, stop.
   3. Solve `(A + μ·diag(scale)) δ = -g` by Cholesky, as `δ = -cho_solve(cho_factor(A + μ·diag(scale)), g)` (upper factor). On failure or non-finite `δ`: `μ ← μν`, `ν ← 2ν`; if `μ > 1e20`, status 3, stop; else repeat step 4.
   4. If `‖sqrt(scale)⊙δ‖ <= tol·(tol + ‖sqrt(scale)⊙x‖)`: status 3, stop.
   5. `x' = x + δ`, `r' = residual(x')`, `nfev += 1`, `cost' = ½ r'ᵀr'`; `pred = -(g·δ + 0.5·(δ·(A δ)))`, evaluated in that order (the matrix-vector product `A δ` first); `ρ = (cost - cost')/pred` if `pred > 0`, else -1. With `verbose > 1` print `LM nfev={nfev} cost={cost':.6e} mu={μ:.2e} rho={ρ:.3f}`.
   6. If `ρ > 0` and `cost'` finite: `actual = (cost - cost')/cost` (0 if `cost <= 0`); accept (`x, r, cost ← x', r', cost'`); if `actual <= tol` and `pred / max(cost', 1e-300) <= tol`: status 2, stop. If finite differences and not `affordable(len(x))`: stop with status 0. Relinearize; `scale = max(scale, diag A)`; `μ ← μ·max(1/3, 1 - (2ρ - 1)³)`; `ν = 2`.
   7. Else: `μ ← μν`, `ν ← 2ν`; if `μ > 1e20`: status 3, stop.
5. Leaving the loop by budget keeps status 0.

Result: `x`, `cost`, `fun = r`, `nfev`, `status`, `success = status > 0`, message: -1 `"The residual or Jacobian is not finite."`, 0 `"The maximum number of function evaluations is exceeded."`, 1 ``"`gtol` termination condition is satisfied."``, 2 ``"`ftol` termination condition is satisfied."``, 3 ``"`xtol` termination condition is satisfied."`` (the backticks are part of the text).

`levenberg_marquardt(residual, jacobian, x0, *, max_nfev=None, tolerance=1e-10, verbose=0, max_restarts=8)`: `max_nfev < 1` -> `ValueError("max_nfev must be at least 1 or None.")`; `not tolerance > 0` -> `ValueError("tolerance must be positive.")`. Up to `1 + max_restarts` runs; each starts from the previous run's `x` with the remaining budget (`max_nfev - used`). After each run: `used += nfev`; `improved = run.cost < 0.99 · best` (`best` starts at +∞); `best = min(best, run.cost)`; continue only if the run ended with status 3, improved, and (`max_nfev` is `None` or `used < max_nfev`). Return the last run's result with `nfev = used`.

### 8.6 `solve_problem`

Signature: `solve_problem(problem, u0, θ0, *, max_nfev=None, tolerance=1e-10, exact_jacobian=True, verbose=0, callback=None)`.

1. Unknowns `x = [η, θ]`, `η = t_to_eta(u0)` (`N - 1` entries); `unpack(x) = (eta_to_t(x[:N-1], S), x[N-1:])`.
2. Residual function: `r = problem.residual(unpack(x))`, `nfev += 1`; keep `best_x` = the `x` with the smallest `rᵀr` so far; then `report(callback, split_control_points(last_control_points), last_rms_error, problem.knots)` (8.8); true -> stop.
3. Jacobian: `reduced_jacobian(problem, η, θ) = [J_t · eta_jacobian(η, u, S) | parameter_jacobian(u, θ)]` when `exact_jacobian`, else `"2-point"`.
4. `levenberg_marquardt(residual, jacobian, x0, max_nfev, tolerance, verbose)`. On a stop: `x = best_x`, `success = False`, message `STOPPED = "Stopped by callback."`. A `ValueError` or `LinAlgError` raised inside becomes `ValueError("VarPro fit failed: {message}")`.
5. Return `result_from_problem(problem, u, θ, nfev, success, message)`.

`result_from_problem(problem, u, θ, nfev, success, message)`: set `problem.theta = θ`; evaluate `residual(u, θ)`; return a library `FitResult` (8.7) with per-section copies of `P`, `parameters = u`, knot copies, `degree`, `rms_error = last_rms_error`, `max_residual = max_i |N_i P - D_i|`, `nfev`, `success`, `message`, `continuity` (junction labels), `shape_parameters = shape_values(θ)`, `theta = θ`.

### 8.7 Library `FitResult` and `ChainOptions`

`FitResult` fields: `control_points` (list of `(n_s, dim)` arrays), `parameters`, `knots` (list), `degree`, `rms_error`, `max_residual`, `nfev`, `success`, `message`, `continuity = []`, `shape_parameters = []`, `theta = []`, `max_deviation = None`, `history = []`, `options = None`. `num_control_points` = list of counts; `curves()` = one SciPy-style `BSpline(t_s, c_s, p)` per section; `deviations(sections) = chain_deviations(chain_data(sections), curves())`.

`ChainOptions(continuity="G1", fix_junctions=True, fix_ends=True, start_tangent=None, end_tangent=None, smoothing=1e-3, regularization=1e-12)` (immutable); `problem(sections, knots, p)` builds a `VarProProblem` (a tuple continuity is passed as a list; `start = End(fix_ends, start_tangent)`, `end = End(fix_ends, end_tangent)`).

### 8.8 Progress callbacks

`report(callback, control_points, rms, knots)`: `None` -> false. If the callback's signature has a parameter named `knots` or a `**kwargs` parameter, call `callback(control_points, rms, knots=knots)`, else `callback(control_points, rms)`; a callable whose signature cannot be inspected counts as not taking `knots`. Returns the truth value of the result; true requests a stop.

### 8.9 Continuation for geometric junctions

`solve_with_continuation(problem, options, u0, **kw)`:

1. No shape parameters: `solve_problem(problem, u0, problem.theta, **kw)`.
2. Else build the same problem with every junction replaced by `Junction(order, geometric=False, fixed, tangent)` and solve it from `u0` and its initial `θ` (stage 1).
3. `θ0 = theta_from_parametric(stage1.theta)`; `remaining = max_nfev - stage1.nfev` (`None` if unlimited).
4. If stage 1 stopped or `remaining < 1`: return `result_from_problem(problem, stage1.parameters, θ0, stage1.nfev, False, message)` with message `STOPPED` if stopped, else the status-0 message.
5. Else solve the geometric problem from `(stage1.parameters, θ0)` with `max_nfev = remaining`, then add `stage1.nfev` to its `nfev`.

### 8.10 `fit`

Signature: `fit(points, num_control_points=None, degree=3, *, knots=None, continuity="G1", fix_junctions=True, fix_ends=True, start_tangent=None, end_tangent=None, smoothing=1e-3, regularization=1e-12, initial_parameters=None, initial_theta=None, max_points_per_section=None, max_nfev=None, tolerance=1e-10, exact_jacobian=True, verbose=0, callback=None, knot_placement="uniform", grow=False)`.

1. `sections = as_sections(points)`: a 2-D array is one section; a sequence whose first item is 1-D is one section; otherwise each item is a section. Each must be 2-D with `>= 2` rows, all finite, else `ValueError("Each section needs at least two finite points shaped (n, dim).")`; all of one `dim`, else `ValueError("All sections must have the same dimension.")`.
2. Knots. `knots is None`: `num_control_points` required (`ValueError("Give num_control_points or knots.")`); scalar -> that count for every section; sequence -> one per section (`ValueError("Need one control-point count per section.")`); any count `< p + 1` -> `ValueError("Control-point count must be at least degree + 1.")`; knots `open_uniform_knots(p, n_s)`. Otherwise `knots` is one shared 1-D vector or one per section.
3. Validation (`ValueError`): smoothing finite and `>= 0` (`"Smoothing weight must be finite and nonnegative."`); regularization finite and `>= 0` (`"Regularization must be finite and nonnegative."`); `max_nfev` `None` or `>= 1` (`"max_nfev must be at least 1 or None."`); tolerance finite and `> 0` (`"tolerance must be finite and positive."`); placement (`"knot_placement must be 'uniform' or 'free'."`); `grow` with explicit knots (`"Growing the control-point count needs num_control_points, not knots."`).
4. `u0 = initial_chain_parameters(initial_parameters, sections)`; `options = ChainOptions(continuity, fix_junctions, fix_ends, start_tangent, end_tangent, smoothing, regularization)`.
5. `max_points_per_section` given and `< 2`: `ValueError("max_points_per_section must be at least 2.")`. `dense` = given and some section has more points.
6. Dispatch: `grow` -> `grow_to_counts(sections, counts, p, options, u0, initial_theta, max_nfev, max_points_per_section, callback, knot_placement)` (9.4); else `dense` -> `fit_subsampled(sections, knots, p, options, u0, initial_theta, max_nfev, max_points_per_section, callback)` (9.2); else `_fit_once`: with `initial_theta` (must have `num_params` entries, else `ValueError("initial_theta must have {k} entries.")`) `solve_problem`, otherwise `solve_with_continuation`.
7. `result.options = options`.
8. If `knot_placement == "free"`, not `grow`, not `dense`, and the result is not stopped: `polished = polish_knots(sections, result, options, callback=callback)`. A stopped `polished` is returned. Else if `polished.rms_error <= result.rms_error`: `polished.nfev += result.nfev`, `polished.success, polished.message = result's`, `polished.options = options`, return `polished`; otherwise return `result`.

---

## 9. VarPro procedures (`bspline_varpro.tolerance`, `joint`, `refinement`)

### 9.1 Refit helper

`_refit(sections, knots, p, options, u, θ, max_nfev, max_points=None, callback=None)`: if `max_points` is given and some section is longer: `fit_subsampled(..., max_points, callback)`. Else build the problem; if `θ is None` or `len(θ) != num_params`: `solve_with_continuation(problem, options, u, max_nfev=max_nfev, callback=callback)`; else `solve_problem(problem, u, θ, max_nfev=max_nfev, callback=callback)`.

### 9.2 Subsampling (`fit_subsampled`)

Signature: `fit_subsampled(sections, knots, p, options, u, θ, max_nfev, max_points, callback=None)`.

1. `subsample_indices(n, m)`: `arange(n)` if `n <= m`; else `unique(round(linspace(0, n-1, m)))` as integers (both ends kept).
2. With `D = chain_data(sections)` and `(first, last)` from `section_ranges`: indices of section `s` = `first + subsample_indices(last - first + 1, m)`; `chosen = unique(all indices)`; subsets `D[indices_s]` (shared junction rows stay shared).
3. `coarse = _refit(subsets, knots, p, options, u[chosen], θ, max_nfev, callback=callback)` (no `max_points`).
4. `coarse.parameters = project_to_chain(D, coarse.curves(), chosen, coarse.parameters)`; `coarse.options = options`. If stopped, return `coarse`.
5. `refined = refine(sections, coarse, options, callback=callback)`; `refined.nfev += coarse.nfev`; return `refined`.

`project_between(D, curves, lower, upper, samples=16, start=None)`: per point a grid `lower + (upper - lower)·linspace(0, 1, 17)` (after `upper = max(upper, lower)`), plus `start` as an extra candidate when given; take the candidate of least squared distance. Then up to 20 Newton steps for all points together: `g = (C(u) - q)·C'(u)`, `h = |C'|² + (C - q)·C''`; trial `clip(u - g/h, lower, upper)` where `h > 0`, else `u`; non-finite trials stay `u`; a point accepts its trial only if its squared distance decreases; stop when no point improves.

`project_to_chain(D, curves, chosen, u_chosen)`: for each row `i`: `before = clip(searchsorted(chosen, i, "left") - 1, 0, last)`, `after = clip(searchsorted(chosen, i, "right"), 0, last)` (`last = len(chosen) - 1`); `u = project_between(D, curves, u_chosen[before], u_chosen[after])`; `u = maximum.accumulate(clip(u, 0, S))`; `u[0] = 0`; `u[-1] = S`.

### 9.3 Knot insertion and removal helpers

`spans(t, u)`: `breaks = unique(t)`; span index of each `u` = `clip(searchsorted(breaks, u, "right") - 1, 0, len(breaks) - 2)`.

`new_knots(t, u, dev, tol, min_span)` (points of one section, local parameters `u`): for each span `i` with `[a, b] = [breaks_i, breaks_{i+1}]`: skip unless some point in it has `dev > tol`; skip if it holds fewer than 2 points or `b - a < 2·min_span`; `v = clip(u of its point with the largest dev, a + (b-a)/4, b - (b-a)/4)`; if not (some point `< v` and some point `> v`), `v = median(u of the span)`; keep `(max dev, v)` when `v - a >= min_span` and `b - v >= min_span`. Return the `v`s ordered by `(max dev, v)` descending.

`removal_order(t, p, u, dev, tol)`: for each distinct interior knot `k` (ascending): `i = searchsorted(breaks, k)` (left); score = (the largest `dev` of the points in spans `i-1` and `i`, or 0 when they hold no points) / `tol`. Return the knots ordered by `(score, k)` ascending.

### 9.4 Growing to counts (`grow_to_counts`)

Signature: `grow_to_counts(sections, counts, p, options, u, θ, max_nfev, max_points=None, callback=None, knot_placement="free")`.

1. `need` per section (library rule of 5.2 with the options' junctions and ends). Start knots `open_uniform_knots(p, min(need_s, count_s))`.
2. Loop:
   1. `result = _refit(sections, knots, p, options, u, θ, max_nfev, max_points, callback)`; stopped -> return it; `u, θ ← result.parameters, result.theta`.
   2. `missing_s = count_s - current count_s`; all `<= 0` -> leave the loop.
   3. `dev = chain_deviations(D, result.curves())`. For each section with `room = missing_s > 0`, over the points located on it: `new = new_knots(t_s, local, dev_s, 0.5·max(dev_s, initial 0), 1e-4)[:room]`; if empty, `new = new_knots(t_s, local, dev_s, -1.0, 1e-4)[:room]`. Uniform placement: `t_s = open_uniform_knots(p, current + max(len(new), 1))` (counts as inserted). Free placement: if `new` is non-empty, `t_s = clamped_knots(p, interior(t_s) ∪ new)` (inserted).
   4. Nothing inserted: leave the loop.
3. If free and (`max_points is None` or every section has `<= max_points` points): `polished = polish_knots(sections, result, options, callback=callback)`; stopped -> return it; if `polished.rms_error <= result.rms_error`: add `result.nfev`, copy `success` and `message`, use it.
4. Return the result.

### 9.5 Knot polishing (`polish_knots`)

Signature: `polish_knots(points, result, options=None, max_nfev=60, min_gap=1e-3, should_stop=None, callback=None)`.

1. `options = options or result.options or ChainOptions(continuity=result.continuity or "G1")`.
2. Per section, `r_s` = number of interior knots if they are strictly increasing, else 0 (that section keeps its knots). `Σ r_s == 0`: return `result`.
3. Floor of each polished section: `ts = unique([0, local parameters of its located points, 1])`; `floor_s = elementwise min(max(min(diff(ts)), min_gap), 0.999 · diff([0, interior, 1]))` (`r_s + 1` values); then `floor_s *= min(1, 0.5 / Σ floor_s)`.
4. Gap parameters: `ξ_s = inverse_softplus(max(diff([0, interior, 1]) - floor_s, 1e-12))`. Knots of `ξ`: gaps `softplus(ξ_s) + floor_s`, interior knots `cumsum(gaps)[:-1] / Σ gaps`, `clamped_knots`.
5. Unknowns `[η, θ, ξ]` (`θ` = `result.theta`). Residual: if `should_stop` is given and returns true, stop; the problem for the current `ξ` (cache of one entry); `problem.residual(eta_to_t(η, S), θ)`; then `report(callback, ...)` with the moving knots (8.8); true -> stop.
6. Jacobian: exact `reduced_jacobian` for `[η, θ]`; each knot column by forward difference with `h = 1e-7·(1 + |ξ_i|)` on a freshly built problem, at the same `u`, `θ`.
7. `levenberg_marquardt(residual, jacobian, x0, max_nfev=max_nfev)` (tolerance 1e-10). Stop -> copy of `result` with `success = False`, message `STOPPED`. `ValueError` or `LinAlgError` -> `result` unchanged. Else `result_from_problem` on the polished knots with the solution's `nfev`, `success`, `message`.

### 9.6 Tolerance search (`fit_to_tolerance`)

Signature: `fit_to_tolerance(points, tolerance, degree=5, *, continuity="G2", fix_junctions=True, fix_ends=True, start_tangent=None, end_tangent=None, smoothing=1e-7, regularization=1e-12, initial_knots=None, initial_control_points=None, initial_parameters=None, max_control_points=None, optimize_knots=True, remove_knots=True, max_points_per_section=200, max_nfev=None, max_iterations=40, callback=None, knot_placement="free", should_stop=None, step_callback=None)`.

Validation (`ValueError`): tolerance finite and `> 0` (`"Tolerance must be positive."`); placement (`"knot_placement must be 'free' or 'uniform'."`); `max_iterations >= 1` (`"max_iterations must be at least 1."`); sections as in 8.10.

1. `need_s` (library rule). Start knots: `initial_knots` if given; else, from `initial_control_points` (scalar or one per section; count mismatch `ValueError("Need one initial control-point count per section.")`), `open_uniform_knots(p, max(n_s, need_s))`; else `open_uniform_knots(p, need_s)`.
2. `limit_s = min(len(section_s) - 1, max_control_points or len(section_s))`; any `need_s > limit_s`: `ValueError("Sections have too few points for the requested end conditions.")`.
3. `u = initial_chain_parameters(initial_parameters, sections)`, `θ = None`, `history = []`.
4. `step` (defined only when `should_stop` or `step_callback` is given): called on every residual evaluation of every refit and polish; returns true, and records that a stop happened, when `should_stop()` is true or `report(step_callback, control_points, rms, knots)` is true.
5. `record(stage, result, dev)`: append `{"stage", "control_points": result.num_control_points, "max_deviation": dev, "rms_error": result.rms_error}`; return the truth of `callback(result, dev)` (false without callback).
6. `finish(result, dev, success, message)`: set `max_deviation = dev`, `history`, `options`, `success`, `message`; return it.
7. Insertion, at most `max_iterations` rounds:
   1. `result = _refit(sections, knots, p, options, u, θ, max_nfev, max_points_per_section, step)`; `u, θ ← result's`; `devs = chain_deviations(D, result.curves())`; `dev = max(devs)`.
   2. Stopped, or `record("insert", result, dev)` true: `finish(result, dev, False, STOPPED)`.
   3. `dev <= tolerance`: leave the loop.
   4. For each section whose located points have `max dev > tolerance`: `count` = its control points, `room = limit_s - count`; `new = new_knots(t_s, local, devs_s, tolerance, 1e-4)[:max(room, 0)]`; uniform with `room > 0`: `t_s = open_uniform_knots(p, count + max(len(new), 1))` (inserted); free with `new` non-empty: insert them.
   5. Nothing inserted: `finish(result, dev, False, "Tolerance not reached: no knot span can be refined further.")`.

   All rounds used without meeting the tolerance: `finish(result, dev, False, "Tolerance not reached within max_iterations.")`.
8. Uniform placement: for each section `s`, while its count `> need_s`: trial knots with `t_s = open_uniform_knots(p, count - 1)`; `trial = _refit(..., result.parameters, result.theta, ...)`; stopped -> `finish(result, dev, False, STOPPED)`; trial max deviation `> tolerance` -> stop for this section; else `result, dev = trial, its deviation` and `record("remove")` (true -> stopped finish). Then `finish(result, dev, True, "Within tolerance {tolerance:g}.")`.
9. Free placement: `dense` as in 8.10; `polish = optimize_knots and not dense`. `polished(result, dev)`: `candidate = polish_knots(sections, result, options, callback=step)`; stopped candidate -> keep `result`; accept when its max deviation `<= tolerance` and its RMS `<= result.rms_error`.
   1. If `polish`: apply `polished`; if accepted and `record("polish")` is true, or a stop was requested (a recorded stop, or `should_stop()` true): stopped finish.
   2. If `remove_knots`: for each section `s`, repeat while the last pass removed a knot and the count `> need_s`: compute deviations of the current result; for each `k` in `removal_order(t_s, p, local, devs_s, tolerance)`: trial knots = interior of `t_s` without the entry nearest `k` (one entry); refit from the current `u`, `θ` (stopped -> stopped finish); if its max deviation `> tolerance` and `polish`: polish the trial (stopped -> stopped finish) and measure again; if `<= tolerance`: accept, `record("remove")` (true -> stopped finish), and restart the scan for this section.
   3. If `polish`, at least one knot was removed and no stop was requested: apply `polished` once more; if accepted, `record("polish")` (its return value is ignored).
   4. `finish(result, dev, True, "Within tolerance {tolerance:g}.")`.

`{tolerance:g}` is Python's general format with 6 significant digits (`1e-05`, `0.0001`).

### 9.7 Joint refinement

`JointProblem(sections, knots, p, C, d, smoothing=0.0, regularization=1e-12)`:

- `D = chain_data(sections)`. If `C` has rows: SVD `C = U Σ Vᵀ`; `rank = #{σ > 1e-10 · max(σ_0, 1)}`; null space `Z = (rows rank.. of Vᵀ)ᵀ`; `P0 = lstsq(C, d)` (minimum norm). Else `Z = I_T`, `P0 = 0`.
- `Q = reg·I + smoothing·S_b` (bending term only if `smoothing > 0`); `R = matrix_root(Q)`; `c = mean(D)`; `RZ = R Z`; `RP0 = R (P0 - 1cᵀ)`; `evaluations = 0`.
- Unknowns: `z` (`num_z × dim`, flattened row-major) with `P = P0 + Z z`, and the chain parameters of rows `1..N-2` (the first and last stay at 0 and `S`).
- `evaluate(u, z, with_jacobian=False)`: `evaluations += 1`; residual `[N(u)P - D ; RP0 + RZ z]` point-major; with the Jacobian also `tangent = dN(u) P` and `Jz = [kron(N Z, I_dim) ; kron(RZ, I_dim)]`.
- `z_from(P) = Zᵀ(P - P0)` flattened. `control_points(z) = P0 + Z z` with `z` reshaped to `(num_z, dim)` (the stacked control points; used in 9.7 and section 10).

`solve_joint(problem, u, z, max_iterations=None, tolerance=1e-12, on_step=None)` returns `(u, z, iterations, converged, message)`:

1. `max_iterations` not `None` and `< 1`: `ValueError("max_iterations must be at least 1 or None.")`; `not tolerance > 0`: `ValueError("tolerance must be positive.")`.
2. `u = clip(maximum.accumulate(u), 0, S)`; evaluate with Jacobian; `cost = ½ rᵀr`; `μ` unset; `ν = 2`; message `"Maximum iterations reached."`, `converged = False`.
3. For `it = 1, 2, ...` (up to `max_iterations` when given):
   1. `Tf = tangent[free]`; `rd` = data residual rows of the free points; `g_t = Σ_k Tf·rd`; `U = Σ_k Tf²`; `W[i, :] = Σ_k Tf[i,k] · Jz[row of (free_i, k)]`; `V = JzᵀJz`; `g_z = Jzᵀr`.
   2. Gradient cosine `= max |[g_t, g_z]| / (sqrt(max([U, diag V], 0)) · sqrt(2 cost))` (0 where a norm is 0, and 0 when `cost == 0`). `<= tolerance`: `converged = True`, `"Gradient below tolerance."`, stop.
   3. `dV = max(diag V, 1e-12 · max(max(diag V, initial 0), 1e-300))`; `Uf = max(U, 1e-12 · max(max(U, initial 1), 1e-300))`; `μ = 1e-3` if unset.
   4. Inner loop: `Ud = U + μ·Uf`; `Vd = V + μ·diag(dV)`; `Sc = Vd - Wᵀ(W / Ud)`; `rhs = -g_z + Wᵀ(g_t / Ud)`; `dz = solve(Sc, rhs)`; on `LinAlgError`: `μ ← μν`, `ν ← 2ν`, if `μ > 1e16` return `(u, z, it, False, "Singular normal equations.")`, else retry. `dt = -(g_t + W dz)/Ud`; `u'[free] = clip(PAV(u[free] + dt), 0, S)`; `dt = u'[free] - u[free]`; `z' = z + dz`; `cost' = ½|r(u', z')|²`; `pred = -(g_t·dt + g_z·dz) - ½(dt·(U⊙dt) + 2 dt·(W dz) + dz·(V dz))`; `ρ = (cost - cost')/pred` if `pred > 0` else -1. If `ρ > 0`: `μ ← μ·max(1/3, 1 - (2ρ - 1)³)`, `ν = 2`, accept. Else `μ ← μν`, `ν ← 2ν`; if `μ > 1e16`: return `(u, z, it, cos <= sqrt(tolerance), "No further decrease possible.")` with `cos` the gradient cosine of 3.2.
   5. Accepted: `decrease = cost - cost'`; `u, z ← u', z'`; re-evaluate with Jacobian; `cost = cost'`; if `on_step` returns true: return `(u, z, it, False, "Stopped by callback.")`; if `decrease <= tolerance · cost`: `converged = True`, `"Relative decrease below tolerance."`, stop.
4. Return `(u, z, it, converged, message)` (`it` = last iteration number, 0 if none ran).

PAV (pool adjacent violators, least-squares nondecreasing projection): keep a stack of blocks `(mean, count)`; push each value as `(v, 1)`; while the top two blocks have `mean_below > mean_top`, merge them into `((m1 c1 + m2 c2)/(c1 + c2), c1 + c2)`; finally expand each block mean `count` times.

`refine(points, result, options=None, max_iterations=None, callback=None)`:

1. Options: the argument, else `result.options`, else `ChainOptions(continuity=result.continuity or "G1")`.
2. `(C, d) = options.problem(sections, result.knots, p).constraints.build(θ)` with `θ = result.theta` if its length equals `num_params`, else `initial_params()`; `JointProblem(sections, result.knots, p, C, d, smoothing=options.smoothing, regularization=options.regularization)`.
3. Start `u = result.parameters`, `z = z_from(vstack(result.control_points))`. `on_step` (only with a callback) reports each accepted step's control points, RMS and knots (8.8).
4. Result: per-section control points, `parameters = u`, knots and degree of `result`, `rms_error = sqrt(mean_i |N_i P - D_i|²)`, `max_residual = max_i |N_i P - D_i|`, `nfev = evaluations`, `success = converged`, `message`, `continuity`, `shape_parameters`, `theta` copied from `result`, `options`, and `max_deviation = max chain_deviations(D, curves)`.

### 9.8 Library constants

| Name | Value |
|---|---|
| `ALPHA_RANGE` | 10 |
| `ALPHA_PULL` | 3e-5 |
| `SPEED_PULL` | 1e-3 |
| `SPEED_SLACK` | 1.0 |
| `ETA_LIMIT` | 700 |
| `MIN_RELATIVE_GAP` | 1e-3 |
| `STOPPED` | `"Stopped by callback."` |
| LM damping start, limit | 1e-3, 1e20 (joint refinement: 1e-3, 1e16) |
| LM restarts | 8 |
| knot polish budget | 60 residual evaluations |
| knot polish finite-difference step | `1e-7·(1 + |ξ|)` |
| minimum knot span | 1e-4 |
| tolerance-search rounds | 40 |
| projection grid, Newton steps (`project_between`) | 16 intervals, 20 |

---

## 10. Direct LSQ (`optimization/optimizers/constrained_lsq.py`, `ConstrainedLSQOptimizer`, name `"Direct LSQ"`)

Always parametric continuity; reads `config.degree`, `config.num_control_points`, `config.continuity` (its order only), the airfoil sections, and `opt_config.custom_parameters`, `parameterization`, `optimize_le_position`. `geometric_continuity`, `junction_continuity`, `smoothing_weight`, `tolerance`, `knot_placement`, `grow_control_points` and `max_points_per_section` are ignored. `MAX_ITERATIONS = 100`.

`fit(config, opt_config, airfoil, callback=None)`:

1. `sections` = the airfoil sections; `p`, `n = num_control_points`; `S` = number of sections; shared knots `t = open_uniform_knots(p, n)`; `D = chain_data(sections)`.
2. Parameters `u`:
   - `custom_parameters` given: `validate_chain_parameters(custom_parameters, sections, D)`; fixed (`fixed = True`);
   - else `"centripetal"`: `chain_parameters([centripetal_parametrize(sec, 0.5)], sections)`;
   - else (`"arc_length"`): `chain_parameters([compute_arc_length_parameters(sec)], sections)`.
3. `free = optimize_le_position and S > 1`. `(C, d) = constraints(free)` (10.1).
4. Basis `N(u)`: `(N, S·n)`, block of section `s` in columns `s·n .. (s+1)·n - 1` filled with the design matrix of the local parameters of the points located on `s`.
5. `solve(N, D, C, d)`: `K = [[NᵀN + 1e-10·I, Cᵀ], [C, 0]]`, right-hand side `[NᵀD ; d]`, solved as a symmetric indefinite system (`scipy.linalg.solve`, LAPACK `dsysv`, see 23.3); `P` = first `S·n` rows.
6. If `fixed`: `P = solve(N(u), D, C, d)`; one evaluation counted (`increment_evaluation`, 11.4).
7. Else:
   1. stages = `[constraints(False)]` if `free` else `[]`; `(C0, d0)` = the first stage if `free` else `(C, d)`;
   2. `P = solve(N(u), D, C0, d0)`; count one evaluation;
   3. for each `(Cs, ds)` in `stages + [(C, d)]`: `JointProblem(sections, [t]*S, p, Cs, ds, regularization=0.0)` (smoothing 0); `(u, z, ...) = solve_joint(problem, u, z_from(P), 100)`; `P = control_points(z)`; count one evaluation.
8. `section_control_points[s]` = rows `s·n .. (s+1)·n - 1` of `P` as float tuples.
9. `final_error = sqrt(mean_i |evaluate_chain(curves, u)_i - D_i|²)` with `curves` = `BSpline(t, ctrl_s, p)`.
10. `x` = all control points flattened section by section, `[x0, y0, x1, y1, ...]`.
11. If a callback is given: if it accepts a `preview` keyword (11.3), call `callback(x, final_error, preview=([arrays of ctrl_s], [t]*S, p))`, else `callback(x, final_error)`. The return value is ignored.
12. Return `RawOptimResult(x, fun=final_error, nit=1, section_control_points, parameters=u)` (`success = True`, `message = ""`, `section_knots = None`, `rms_error = final_error`).

### 10.1 Constraint rows (`constraints(free)`)

Columns: `S·n` stacked control points; values `(x, y)` per row. In order:

1. `P_0 = first point of section 0`;
2. `P_{S·n - 1} = last point of the last section`;
3. for each junction `j = 0..S-2` (`e = j·n + n - 1`, `b = (j+1)·n`):
   1. `P_e - P_b = 0`;
   2. if not `free`: `P_e = last point of section j`;
   3. if continuity order `>= 1`: `[ce_1 on P_{e-1}, P_e] - [cs_1 on P_b, P_{b+1}] = 0`;
   4. if order `>= 2` and `n >= 3`: the same with order-2 coefficients on `P_{e-2..e}` and `P_{b..b+2}`;
   5. if order 3 and `n >= 4`: order-3 coefficients on `P_{e-3..e}` and `P_{b..b+3}`;

where `(ce_r, cs_r) = end_derivative_coefficients(t, p, r)` (7.1/7.5). Orders are cumulative: continuity level 2 (stored as `ContinuityMode.G2`, shown as `C2` for this optimizer, section 22 item 7) adds the rows of orders 1 and 2. The `n >= 3` and `n >= 4` guards of rows 4 and 5 never take effect: `SplineConfig` requires `degree >=` the order (5.4) and the knots require `n >= p + 1` (2.1).

---

## 11. Optimizer dispatch and adapters

### 11.1 `RawOptimResult`

`x` (flat control points), `fun` (error), `nit`, `section_control_points = None`, `parameters = None`, `success = True`, `message = ""`, `section_knots = None`, `rms_error` (defaults to `fun` when not given).

### 11.2 `run_optimization(config, opt_config, airfoil, callback=None, should_stop=None)`

`get_optimizer(type)` returns a new `ConstrainedLSQOptimizer` or `GNVarProOptimizer` (unknown: `ValueError("Unknown optimizer type: {type}")`). For VarPro: `optimizer.config.smoothing_weight = opt_config.smoothing_weight`, then `fit(config, opt_config, airfoil, callback=callback, should_stop=should_stop)`. For Direct LSQ: `fit(config, opt_config, airfoil, callback=callback)`.

### 11.3 `GNVarProOptimizer` (name `"VarPro LM"`)

`GNVarProConfig`: `regularization = 1e-12`, `smoothing_weight = 1e-3`, `use_exact_jacobian = True`, `verbose = 0`, `tolerance = 1e-10`, `tolerance_smoothing = 1e-7`.

Keyword detection: `accepts(cb, name)` is true when `cb` is not `None` and its signature has a parameter named `name` (inspection failure: false). `accepts_preview(cb) = accepts(cb, "preview")`.

`continuity_arg(config)`: `specs = junction_specs()`; `len(specs) > 1` -> the list; one spec -> that string; none (one section) -> `"G1"`.

`fit(...)`: sections as float arrays; tolerance given -> tolerance mode, else fixed mode.

**Fixed mode**: callback `on_evaluation(control_points, rms, knots=None)`: `evaluations += 1`; `increment_evaluation()`; no app callback -> false; with preview -> `callback(flat(control_points), rms, preview=(control_points, knots, p))`, else `callback(flat, rms)`; returns its truth value. Call `bspline_varpro.fit(sections, num_control_points=n, degree=p, continuity=continuity_arg, fix_junctions=not optimize_le_position, smoothing=config.smoothing_weight, regularization=1e-12, initial_parameters=custom_parameters, max_points_per_section, knot_placement, grow=grow_control_points, tolerance=1e-10, exact_jacobian=True, verbose=0, callback=on_evaluation)`. Result: `fun = rms_error`, `nit = max(evaluations, result.nfev)`.

**Tolerance mode**: `with_steps = accepts(callback, "step")`. `on_step(control_points, rms, knots)`: `increment_evaluation()`; return `callback(flat, rms, preview=(control_points, knots, p) if preview accepted else None, step=True)`. `on_refit(result, deviation)`: `refits += 1`; if not `with_steps`, `increment_evaluation()`; no callback -> false; with preview `callback(flat, deviation, preview=(result.control_points, result.knots, result.degree))`, else `callback(flat, deviation)`. Call `fit_to_tolerance(sections, tolerance, p, continuity=continuity_arg, fix_junctions=not optimize_le_position, smoothing=1e-7, regularization=1e-12, initial_parameters=custom_parameters, max_points_per_section, knot_placement, initial_control_points=None if grow_control_points else n, callback=on_refit, should_stop=should_stop (only when given), step_callback=on_step (only when with_steps))`. Result: `fun = max_deviation`, `nit = max(refits, 1)`.

Both modes return `RawOptimResult(x = flat control points, fun, nit, section_control_points as float tuples, parameters, success, message, section_knots as lists, rms_error)`.

Tolerance mode never reads `smoothing_weight` (fixed 1e-7) and passes `fit_to_tolerance` defaults for `max_iterations` (40), `optimize_knots`, `remove_knots`, `max_control_points`, `max_nfev`.

### 11.4 Counters (`optimization/tracking.py`)

Two process-global integers. `reset_iteration_tracking()` (alias `reset_evaluation_tracking`) sets both to 0. `increment_evaluation()` and `increment_iteration()` add 1 and return the new value; `get_evaluation_count()` reads the evaluation count. Evaluations are counted by the adapters (11.3, 10), iterations by the live preview (19.12).

---

## 12. Measuring a fit

### 12.1 Orthogonal projection (`diagnostics.project(points, curve, samples_per_span=32, candidates=3, chunk=4096)`)

1. Grid: `breaks = unique(t[p : len(t) - p])`; for each span `[a, b]`, `linspace(a, b, 33)` without its last value; then the final break.
2. For each point (in chunks of 4096): squared distances `d²` to all samples; a sample is a local minimum when `d²_j <= d²_{j-1}` and `d²_j <= d²_{j+1}` (missing neighbours count as +∞). Rank samples by `d²` with non-minima set to +∞ (stable argsort) and take the first 3; a taken sample that is not a local minimum is invalid.
3. For each valid candidate: bracket `lo = grid[j-1]`, `hi = grid[j+1]` (clamped indices), `u = grid[j]`; up to 40 iterations while active: `g = (C(u) - q)·C'(u)`, `h = |C'|² + (C - q)·C''`; `hi = u` where `g > 0`, `lo = u` where `g < 0`; Newton `u - g/h` is taken when `h > 0` and strictly inside `(lo, hi)`, else the bisection `(lo + hi)/2`; a candidate stops when `|u_new - u| <= 1e-15 (1 + |u|)`. Clip `u` to the domain.
4. Candidates also include both domain ends. The foot is the candidate of least distance (first on ties, in the order: refined candidates, start, end). Returns `(u, distance)`.

`distance_to_curves(points, curves) = elementwise min over curves of project(points, curve)[1]`. `section_deviations(sections, curves)` = that per section. `max_deviation(sections, curves)` = the max. `point_on_curves(point, curves)` = `(section, u, distance)` of the nearest curve (first on ties).

### 12.2 Fairness and junction measures

- `fairness(curves, samples_per_span=64, min_amplitude=1e-5)`: per curve the span grid of 12.1 with 64 samples per span; `κ` on it; `max_curvature = max|κ|` over all curves; `noise = 1000 · eps / size` where `size` = largest per-axis range of all control points (or 1 if 0); `curvature_extrema = Σ count_turns(κ of curve, max(1e-5 · max_curvature, noise))`; `roughness = Σ_curves Σ_spans Σ_nodes w (dκ/ds)² |C'|` (8-point Gauss per span between the unique knots of the domain); `max_curvature_slope = max |dκ/ds|` over the grids. Returns the four keys.
- `count_turns(values, threshold)`: `turns = 0`, `direction = 0`, `lo = hi = values[0]`; for each later `v`: if `direction == 0`: `v - lo > threshold` -> `direction = 1, hi = v`; elif `hi - v > threshold` -> `direction = -1, lo = v`; else `lo = min(lo, v)`, `hi = max(hi, v)`. If `direction == 1`: `v > hi` -> `hi = v`; elif `hi - v > threshold` -> `turns += 1`, `direction = -1`, `lo = v`. If `direction == -1`: `v < lo` -> `lo = v`; elif `v - lo > threshold` -> `turns += 1`, `direction = 1`, `hi = v`.
- `junctions(curves)`: per consecutive pair `(a, b)` at `ua = a.t[-1]`, `ub = b.t[0]`: unit tangents `ta`, `tb`; `gap = |a(ua) - b(ub)|`; `tangent_angle = 2·atan2(|ta - tb|, |ta + tb|)` (radians); `curvature_jump = |κ_a(ua) - κ_b(ub)|`; `curvature_slope_jump = |dκ/ds_a - dκ/ds_b|`.

### 12.3 Report (`analysis.fit_report(sections, result, precision=None, tolerance=None)`)

`curves = scipy_curves(result)`: `BSpline(result.knots(i), ctrl_i, p)` per section. `D = chain_data(sections)`. `dist[s, i] = project(D, curve_s)[1]`; `nearest_i = argmin_s` (first on ties); `dev_i = dist[nearest_i, i]`.

| Key | Value |
|---|---|
| `max_deviation` | `max dev` |
| `rms_deviation` | `sqrt(mean dev²)` |
| `section_max_deviation` | per `s`: max `dev` over points with `nearest == s` (0 if none) |
| `control_points` | `result.num_control_points` |
| `section_control_points` | counts per section |
| `section_interior_knots` | per section, number of distinct values in `knots(i)[p+1 : -p-1]` |
| `degree` | `p` |
| `tolerance` | argument |
| `precision_floor` | `0.5·sqrt(2)·10^-precision`, or `None` without precision |
| `curvature_extrema`, `roughness`, `max_curvature_slope`, `max_curvature` | 12.2, only when `p >= 2` |
| `junctions` | when `p >= 2`: list of `{gap, tangent_angle_deg (degrees), curvature_jump, curvature_slope_jump}` |
| `leading_edge_radius` | when `p >= 2`: 13.2, or `None` on any exception |
| `max_thickness`, `max_thickness_x`, `max_camber`, `max_camber_x` | 13.3 as floats, or all `None` when it raises |

`summary_line(report)`: `"max deviation {number(max,'.2e')}, RMS {number(rms,'.2e')}, {control_points} control points"` plus `" (tolerance {number(tol,'.2e')})"` when the report's tolerance is truthy.

### 12.4 Report HTML (`format_report_html(report, title="Fit report")`)

`<html><body><h3>{escaped title}</h3>{body}</body></html>`. The body is `<table>` of rows `<tr><td>{escaped label}</td><td class='num'>{value}</td></tr>`:

1. `Max orthogonal deviation`: `number(max, '.3e')`
2. `RMS orthogonal deviation`: `number(rms, '.3e')`
3. `Max deviation per section`: `number(d, '.2e')` joined by `", "`
4. if tolerance truthy: `Tolerance`: `"{number(tol,'.3e')} ({verdict})"`, verdict `<span class='ok'>met</span>` when `max <= tol·(1 + 1e-9)`, else `<b class='warn'>not met</b>`
5. if `precision_floor` truthy: `Rounding error of the data`: `number(floor,'.2e')`, plus ` <b class='warn'>(tolerance is below the data rounding)</b>` when a tolerance is set and below the floor
6. `Degree`, `Control points (chain)`, `Control points per section` (`", "`-joined), `Interior knots per section`
7. if `max_curvature` present: `Max curvature` (`.4g`), `Curvature extrema (wiggles)`, `Roughness ∫(dκ/ds)² ds` (`.4g`); and `Leading-edge radius` (`.6g`) when it is truthy (so not `None`, 0 or missing; infinity is shown as `inf`)
8. if `max_thickness` is not `None`: `Max thickness` and `Max camber` as `"{number(100v,'.3f')}% at {number(100x,'.1f')}% chord"`.

Then `</table>`. If `junctions` is a non-empty list: `<h4>Junctions</h4><table>` with header `<tr><th align='left'>Junction</th><th align='left'>Gap</th><th align='left'>Tangent angle</th><th align='left'>Δκ</th><th align='left'>Δ dκ/ds</th></tr>` and one row per junction (1-based index; gap `.1e`; angle `.2e` followed by `°`; jumps `.2e`), all cells `class='num'`, then `</table>`.

### 12.5 Signed deviations and foot points (`analysis.foot_points(points, result, start=None)`)

Returns `(signed distance (n,), foot (n,2), unit normal (n,2), u (n,S), section (n,))`:

1. `curves = scipy_curves(result)`. If `start` has shape `(n, S)`: `u = refine(start)`: for each section, clip to the domain, then `REFINE_STEPS = 3` Newton steps `u ← clip(u - g/h if h > 0 else u, domain)` with `g`, `h` as 12.1. Else `u[:, s] = project(points, curve_s)[0]`.
2. For each section in order: `at = curve(u_s)`, `d = |points - at|`; where `d < best`, take this section's foot and tangent.
3. Normal = `(t_y, -t_x)` normalized (denominator floored at 1e-300): right of the direction of travel (outside for Selig data).
4. Signed distance = `copysign(best, (point - foot)·normal)`.
5. Section: `tie = 1e-9 · max(largest per-axis range of the points, 1e-300)`; first section with `distance <= best + tie`; then running maximum along the points.

`signed_deviations(points, result)` = the first item.

### 12.6 Default tolerance

`analysis.default_tolerance(coords, precision)` = library `default_tolerance(points, decimals=precision if precision else None)` (precision 0 counts as unknown):

1. `chord = max(ptp x, ptp y)`; `tol = 1e-6 · chord`.
2. `σ = noise_level(points)`:
   1. fewer than 9 points -> 0;
   2. windows of 9 consecutive points; for each, centre = point 4, `e` = unit vector from window point 0 to point 8 (windows with zero length dropped), `n = (-e_y, e_x)`; `x_k = ((w_k - centre)·e) / length`, `y_k = (w_k - centre)·n`;
   3. keep windows whose `x` is strictly increasing; none -> 0;
   4. Vandermonde `V` with columns `x^0..x^6`; hat `H = V pinv(V)`; `r = (y_4 - H_4·y) / sqrt(max(1 - H_44, 1e-12))`;
   5. `σ = 1.4826 · median |r|`.
3. `d = decimals`, or `decimal_places(points)`: the smallest `d` in `0..10` with `|v - np.round(v, d)| <= 1e-12·max(1, max|v|)` for every coordinate (NumPy rounding: halves to even at the binary value), else `None`.
4. If `d` is known: `tol = max(tol, 10^-d)`; if `σ <= 3·10^-d/sqrt(12)`, return `tol`.
5. Return `max(tol, (sqrt(2 ln max(N, 2)) + 0.5)·σ)`.

`precision_floor(points) = 0.5·sqrt(2)·10^-d` for detected `d`, else 0.

---

## 13. Airfoil geometry of a fit (`bspline_varpro.airfoil`)

Curve inputs (`as_curves`): an object with a callable `curves()` (a library `FitResult`) -> its curves; one B-spline -> `[it]`; a sequence -> list. Curve start = `curve(t[p])`, end = `curve(t[-p-1])`.

### 13.1 NACA 4-digit (`NACA4(code="0012", closed_te=True)`)

`code` must be 4 characters, all digits, else `ValueError("NACA 4-digit code must have four digits.")`. `m = d1/100`, `p = d2/10`, `t = int(d3d4)/100`, `a4 = -0.1036` (closed) or `-0.1015` (open).

- Thickness at `σ = sqrt(x)`: `y_t = 5t(0.2969σ - 0.1260x - 0.3516x² + 0.2843x³ + a4·x⁴)`.
- Camber: if `m == 0` or `p == 0`: 0 and slope 0. Else `x < p`: `y_c = m/p²·(2px - x²)`, slope `2m/p²·(p - x)`; `x >= p`: `y_c = m/(1-p)²·((1-2p) + 2px - x²)`, slope `2m/(1-p)²·(p - x)`.
- `surface(σ, upper)`: `θ = atan(slope)`; `x_s = x - (±y_t) sin θ`, `y_s = y_c + (±y_t) cos θ` (`+` upper, `-` lower).
- `points(n_per_side=81, spacing="cosine")`: cosine `σ = sin(linspace(0, π, n)/2)`; `"uniform"` `σ = sqrt(linspace(0, 1, n))`; other: `ValueError("spacing must be 'cosine' or 'uniform'.")`. Result: upper surface at `σ` reversed, then lower at `σ[1:]` (`2n - 1` points, TE upper first).
- `leading_edge_curvature(upper=True)`: with `h = 1e-4`, points at `σ = -h, 0, h`; `d1 = (P₂ - P₀)/2h`, `d2 = (P₂ - 2P₁ + P₀)/h²`; `|d1 × d2| / |d1|³`.

### 13.2 Leading edge and radius

`leading_edge(curves)`: `te` = midpoint of the chain start and end. For each curve, 2001 samples `linspace(t[p], t[-p-1], 2001)`, distances to `te`; keep the curve and sample `j` of the overall largest distance (first curve on ties). Refine with bounded Brent minimization of `-|C(v) - te|` on `[u_{j-1}, u_{j+1}]` (clamped), `xatol = 1e-14`; use the minimizer if its distance `>=` that of `u_j`, else `u_j`. Return `(section, v, C(v))`.

`leading_edge_radius(curves)`: at that point `κ = |d1 × d2| / |d1|³`; `1/κ` if `κ > 0`, else `inf`.

### 13.3 Thickness and camber (`thickness_and_camber(curves, stations=201, samples=4000)`)

1. `(s, u_le, le) = leading_edge(curves)`; if (`s == 0` and `u_le <= start`) or (`s == last` and `u_le >= end`): `ValueError("Leading edge found at an end; expected an airfoil (TE -> LE -> TE).")`.
2. `te` = midpoint of chain start and end; `pts = resample(curves, samples)` (13.4, `2·samples - 1` points).
3. `front = argmin pts.x` (first); `chord = te_x - pts[front].x`; `chord <= 0`: `ValueError("Trailing edge not behind the leading edge along x.")`.
4. `local = (pts - (pts[front].x, te_y)) / chord`.
5. `upper = local[:front+1]`, `lower = local[front:]`; if `Σ (x_i y_{i+1} - y_i x_{i+1})` (cyclic) `< 0`, swap them.
6. `surface(side, x)`: sort the side by `x` (stable) and linearly interpolate `y` (NumPy `interp`: constant beyond the ends).
7. `distributions(x)`: `thickness = y_u - y_l`, `camber = (y_u + y_l)/2`.
8. `fine = 0.5(1 - cos(linspace(0, π, 4·samples)))`; peaks `peak(fine, values)`: `j = argmax|y|`; if `0 < j < last`: fit a degree-2 polynomial through `(x_{j-1..j+1}, y_{j-1..j+1})`; if the leading coefficient is not 0 and its vertex `xv` satisfies `x_{j-1} <= xv <= x_{j+1}`, return `(xv, poly(xv))`; else `(x_j, y_j)`.
9. Stations `x = 0.5(1 - cos(linspace(0, π, stations)))`.

Returns `x`, `thickness`, `camber` (at the stations), `chord`, `max_thickness`, `max_thickness_x`, `max_camber` (signed), `max_camber_x` (fractions of the chord).

`analysis.airfoil_shape(result)` = this on `scipy_curves(result)`, or `None` on any exception.

### 13.4 Resampling (`resample(curves, n_per_side=100, spacing="cosine")`)

1. `n_per_side < 2`: `ValueError("Need at least two points per side.")`; spacing other than `cosine`/`arc`: `ValueError("spacing must be 'cosine' or 'arc'.")`.
2. Chain arc table: per curve `arc_table(curve, 4096)`: `u = linspace(t[p], t[-p-1], 4096)`, cumulative trapezoid of `|C'(u)|`; concatenated with section ids and arc lengths offset by the previous totals (a section boundary appears twice).
3. `(i, u_le, p_le) = leading_edge(curves)`; `s_le = interp(u_le, table u of section i, table s of section i)`.
4. `f` = `0.5(1 - cos(linspace(0, π, n)))` (cosine) or `linspace(0, 1, n)` (arc); arc positions `[f·s_le, s_le + f[1:]·(s_total - s_le)]`.
5. Evaluate each position: `idx = clip(searchsorted(s, value, "right") - 1, 0, len - 2)`; `frac = clip((value - s_idx)/(s_{idx+1} - s_idx), 0, 1)` (0 where the step is 0); parameter `u_idx + frac(u_{idx+1} - u_idx)` on the section of `idx`.
6. Overwrite: index `n - 1` with `p_le`, the first with the chain start, the last with the chain end. Result `2n - 1` points, Selig order.

### 13.5 Trailing-edge segment (`trailing_edge_segment(curves)`)

`start` = chain end (lower TE), `end` = chain start (upper TE). If `|end - start| <= 1e-12·max(1, max|start|)`: `None`. Else a B-spline of the first curve's degree `k` with knots `[0]*(k+1) + [1]*(k+1)` and control points `start + (i/k)(end - start)`, `i = 0..k` (a straight line).

`trailing_edge_gap(points_or_curves)`: for an array, `|first - last|`; for curves, `|chain start - chain end|`.

### 13.6 `fit_airfoil` (library convenience, unused by the application)

Signature: `fit_airfoil(points, tolerance=None, degree=5, *, le_continuity="G2", fix_leading_edge=True, sections=None, **kw)`.

`tolerance` defaults to `default_tolerance(points)`; if `tolerance < precision_floor(points)`, issue a warning `"Tolerance {tol:g} is below the rounding error of the data ({floor:g})."`; sections default to `split_at_leading_edge`; returns `fit_to_tolerance(sections, tolerance, degree, continuity=le_continuity, fix_junctions=fix_leading_edge, **kw)`.

`write_dat(path, points, name="airfoil", decimals=6)`: `"{name}\n"` then lines `f"{x: .{d}f} {y: .{d}f}"` joined by `\n`, final `\n`, ASCII.

---

## 14. Output formats

### 14.1 Airfoil `.dat` (`io.export_airfoil_data(path, coords, name=None, precision=None)`)

Empty `coords`: `ValueError("No coordinates to export.")`. `precision` defaults to 8, then `max(0, min(int(precision), 16))`. Create missing parent directories. UTF-8. If `name` is truthy, first line `_name_line(name)`. Then one line per point `f"{x:.{d}f} {y:.{d}f}"`. Every line, the last included, ends with a line end (the platform's, section 1 "Text files").

### 14.2 Spline `.spline.dat` (`io.export_spline_data`)

Signature: `io.export_spline_data(path, degree, section_control_points, name=None, precision=None, section_knots=None, continuity=None)`.

1. Empty list or all sections empty: `ValueError("No control points to export.")`. Precision as 14.1.
2. `section_knots` default: open uniform per section. Count mismatch: `ValueError("Need one knot vector per section.")`. Any `len(knots) != len(ctrl) + degree + 1`: `ValueError("Knot vector length must be control points + degree + 1.")`.
3. Create parent directories; write UTF-8:

```
<_name_line(name)>                         (only if name is truthy)
# spline-fitter B-spline v2
# degree <int(degree)>
# sections <S>
# continuity <label> <label> ...           (only if continuity is non-empty)
# section 1 control_points <n_1>
# knots 1 <repr(float(k)) ...>             (space separated)
# section 2 control_points <n_2>
# knots 2 ...
<x y>                                      (section 1: all points)
<x y>                                      (each later section: all but its first point)
```

In this block, `<...>` is a value to fill in and the parenthesized text at the right of a line is a note, not file content; nothing follows the last value on a line. The lines of one section (`# section i ...`, then `# knots i ...`) are written for every section in order. Points use `f"{x:.{d}f} {y:.{d}f}"`. Every line, the last included, ends with a line end (the platform's, section 1 "Text files").

**Reading** (`parse_spline_data(path)`, text via 3.1):

1. For each line, `text = line.strip()`; skip empty lines; the first non-empty line is `first_line`.
2. Lines starting with `#`: `words = text[1:].split()`; `degree <d>` (>= 2 words) sets the degree; `section <i> control_points <n>` (>= 4 words, word 2 `control_points`) records count `n` for section `i`; `knots <i> <v...>` (>= 3 words) records floats; `continuity <labels...>` (>= 2 words) records the list. Other comments are ignored.
3. Other lines: `parts = text.split()`; if `float(parts[0])`, `float(parts[1])` parse, it is a point; else if it is the first non-empty line, `name = _name_of_line(text)`.
4. Degree missing: from the file's basename by regex `\.deg(\d+)\.`; else `None`.
5. No counts: return `{name, degree, section_control_points: [all points], section_knots: None, continuity}`.
6. `expected = Σ counts - (#counts - 1)`. If `name` is empty, `len(points) == expected + 1` and `first_line` does not start with `#`: `name = first_line`, drop the first point.
7. Sections in ascending section number: the first takes `n` points; each later starts with the previous section's last point and takes `n - 1` more.
8. Points left over or missing: `ValueError("{path}: expected {consumed} control points, found {len(points)}.")`.
9. `section_knots` = the knot lists in section order when every counted section has one, else `None`.

### 14.3 CAD common rules (`bspline_varpro.export`)

- `curve_data(curve, scale)`: knots `t` as is; `n = len(t) - p - 1`; control points `c[:n]·scale` padded to 3 coordinates with zeros (1-D coefficients become one column); more than 3 dims: `ValueError("Only curves in 1 to 3 dimensions can be exported.")`.
- `closed(ctrl)`: `|ctrl_last - ctrl_first| <= 1e-12·max(1, max|ctrl|)`.
- Any non-finite value written: `ValueError("Cannot export non-finite coordinates or knots.")`.
- Units: `"mm"`, `"m"`, `"inch"`; else `ValueError("units must be 'mm', 'm', or 'inch'.")`.
- `write(path, curves, **kw)`: suffix (lower case) `.step`/`.stp` -> STEP with `name = kw.pop("name", path stem)`; `.iges`/`.igs` -> IGES likewise; `.dxf` -> DXF (a `name` keyword is discarded); else `ValueError("Unsupported export format {suffix!r}; use .step, .iges, or .dxf.")`. Written ASCII with `\n`; returns the path.

### 14.4 STEP AP214 (`to_step(curves, name="curves", scale=1.0, units="mm")`)

`real(v)`: `s = repr(float(v)).upper()`; if `s` has `E`: split into mantissa and exponent, add `.` to the mantissa if missing, result `mantissa + "E" + str(int(exponent))` (`1e-07` -> `1.E-7`, `2.5e+16` -> `2.5E16`, `1e+16` -> `1.E16`; Python's `repr` switches to exponent form below `1e-4` and from `1e16`); else append `.` if `s` has none (never the case for a float's `repr`, so `3.0`, `0.5`, `123.25` are written as is).

`string(text)`: `'` -> `''`, `\` -> `\\`, printable ASCII (32..126) kept, other characters up to U+FFFF -> `\X2\hhhh\X0\` (4 upper-case hex digits), above -> `\X4\hhhhhhhh\X0\`; wrapped in `'...'`. Each such character gets its own escape (`éé` -> `'\X2\00E9\X0\\X2\00E9\X0\'`).

Entities, numbered from `#1` in this order:

1. `APPLICATION_CONTEXT('core data for automotive mechanical design processes')`
2. `APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#1)`
3. `PRODUCT_CONTEXT('',#1,'mechanical')`
4. `PRODUCT({name},{name},'',(#3))`
5. `PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#4))`
6. `PRODUCT_DEFINITION_FORMATION('','',#4)`
7. `PRODUCT_DEFINITION_CONTEXT('part definition',#1,'design')`
8. `PRODUCT_DEFINITION('design','',#6,#7)`
9. `PRODUCT_DEFINITION_SHAPE('','',#8)`
10. length unit: `mm`: `(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))`; `m`: `(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))`; `inch`: four entities `(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))`, `LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#mm)`, `DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)`, `(CONVERSION_BASED_UNIT('INCH',#factor)LENGTH_UNIT()NAMED_UNIT(#dims))` (the last is the length unit);
11. `(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))`
12. `(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())`
13. `UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#length,'distance_accuracy_value','confusion accuracy')`
14. `(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#unc))GLOBAL_UNIT_ASSIGNED_CONTEXT((#length,#angle,#solid))REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY'))`
15. `CARTESIAN_POINT('',(0.,0.,0.))`, `DIRECTION('',(0.,0.,1.))`, `DIRECTION('',(1.,0.,0.))`, `AXIS2_PLACEMENT_3D('',#origin,#z,#x)`
16. per curve `i` (1-based): one `CARTESIAN_POINT('',(x,y,z))` per control point (`real` values joined by `,`), then `B_SPLINE_CURVE_WITH_KNOTS({string(name + " " + i)},{p},(#pts...),.UNSPECIFIED.,{.T. if closed else .F.},.F.,({multiplicities}),({distinct knots as real}),.UNSPECIFIED.)` (run-length grouping of the knot vector by exact equality);
17. `GEOMETRIC_CURVE_SET('curves',(#curves...))`
18. `GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION({name},(#set,#placement),#context)`
19. `SHAPE_DEFINITION_REPRESENTATION(#9,#representation)`

(`{name}` is `string(name)`.) File lines: `ISO-10303-21;`, `HEADER;`, `FILE_DESCRIPTION(('B-spline curves'),'2;1');`, `FILE_NAME({string(name)},'{local time YYYY-MM-DDTHH:MM:SS}',(''),(''),'bspline-varpro','bspline-varpro','');`, `FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));`, `ENDSEC;`, `DATA;`, then `#k=<entity>;` for every entity, `ENDSEC;`, `END-ISO-10303-21;`, joined with `\n` and ending with `\n`.

### 14.5 IGES 5.3 (`to_iges(curves, name="curves", scale=1.0, units="mm")`)

- Unit flag and name: `mm` `(2, "MM")`, `m` `(6, "M")`, `inch` `(1, "IN")`.
- `ascii(text)`: NFKD-normalize, drop combining marks, replace characters outside 32..126 by `?`. `hollerith(text)`: `ascii(text)[:60]`; empty -> empty field, else `f"{len}H{text}"`. `real(v) = f"{v:.16E}"`.
- `extent = max over curves of max|ctrl|` (or 1.0 if 0). `stamp` = local time `%Y%m%d.%H%M%S`.
- **S section**: one record: `"B-spline curves written by bspline-varpro"` left-justified to 72 columns, `S`, sequence `1` right-justified in 7.
- **G section** fields, in order: `1H,`, `1H;`, `hollerith(name)`, `hollerith(name + ".igs")`, `hollerith("bspline-varpro")`, `hollerith("bspline-varpro")`, `32`, `38`, `6`, `308`, `15`, `hollerith(name)`, `1.0`, flag, `hollerith(unit name)`, `1`, `1.0`, `hollerith(stamp)`, `1.0E-07`, `real(extent)`, `` (empty), `` (empty), `11`, `0`, `hollerith(stamp)`. Every field but the last gets `,` appended; the last gets `;`.
- **Packing** (`lines(fields, letter, start, pointer="")`): limit 64 columns for `P`, 72 otherwise; append fields to the current text; when the current text is non-empty and adding the next field would exceed the limit, start a new record (fields are never split). `G` and `S` records: text left-justified to 72, letter, 7-digit sequence. `P` records: text left-justified to 64, the pointer right-justified in 8, `P`, 7-digit sequence.
- **Per curve `i`** (0-based): directory sequence `de = 2i + 1`; `planar = allclose(z, z_0)`; parameter fields `126`, `K = n - 1`, `p`, `planar (1/0)`, `closed (1/0)`, `1`, `0`, then each knot `real`, then `1.0` per control point, then each control point's `x, y, z` as `real`, then `real(t[p])`, `real(t[-p-1])`, then `0.0, 0.0, 1.0` if planar else `0.0, 0.0, 0.0`; commas and final `;` as for G. `P` sequence numbers continue across curves (first record of a curve: `p_start = (records so far) + 1`), pointer `str(de)`.
- **D records** per curve: `f"{126:>8}{p_start:>8}{0:>8}{0:>8}{0:>8}{0:>8}{0:>8}{0:>8}{'00000000':>8}D{de:7d}"` and `f"{126:>8}{0:>8}{0:>8}{#P records:>8}{0:>8}{'':>8}{'':>8}{('SPL' + str(i+1))[:8]:>8}{0:>8}D{de+1:7d}"`.
- **T record**: `f"S{1:7d}G{#G:7d}D{#D:7d}P{#P:7d}{'':<40}T{1:7d}"`.
- File: S, G, D, P records then T, joined by `\n`, final `\n`. Every record is exactly 80 characters.

### 14.6 DXF R2000 (`to_dxf(curves, scale=1.0, layer="0", units="mm")`)

- `$INSUNITS`: mm 4, m 6, inch 1.
- Layer name: forbidden characters are the 12 characters `<`, `>`, `/`, `\`, `"`, `:`, `;`, `?`, `*`, `|`, `=` and the backtick (U+0060). An empty name, a name with a forbidden character, or one with a code point below 32 raises `ValueError` with the text `Invalid DXF layer name {text!r}: it must not be empty or contain control characters or any of ` followed by those 12 characters in that order separated by single spaces, followed by `.`. Characters 127..U+FFFF become `\U+XXXX` (at least 4 upper-case hex digits); above U+FFFF `_`.
- A group is two lines: the code right-justified in 3 characters, then the value (`str`). Handles are upper-case hexadecimal from 1, in creation order.
- Body, in order:
  1. `0 SECTION`, `2 CLASSES`, `0 ENDSEC`.
  2. `0 SECTION`, `2 TABLES`, then tables. A table: `0 TABLE`, `2 <name>`, `5 <handle>`, `330 0`, `100 AcDbSymbolTable`, `70 <record count>`, (DIMSTYLE only: `100 AcDbDimStyleTable`, `71 0`), records, `0 ENDTAB`. A record: `0 <kind>`, `<handle code> <handle>` (5, or 105 for DIMSTYLE), `330 <table handle>`, `100 AcDbSymbolTableRecord`, `100 <subclass>`, `2 <name>`, `70 0`, extra groups. Tables in order: `VPORT` (none); `LTYPE` (`ByBlock`, `ByLayer` with `3 ""`, `Continuous` with `3 Solid line`; each `72 65`, `73 0`, `40 0.0`; subclass `AcDbLinetypeTableRecord`); `LAYER` (`0` and the layer if different; `62 7`, `6 Continuous`, `370 -3`; `AcDbLayerTableRecord`); `STYLE` (`Standard`: `40 0.0`, `41 1.0`, `50 0.0`, `71 0`, `42 2.5`, `3 txt`, `4 ""`; `AcDbTextStyleTableRecord`); `VIEW` (none); `UCS` (none); `APPID` (`ACAD`; `AcDbRegAppTableRecord`); `DIMSTYLE` (`Standard`; `AcDbDimStyleTableRecord`); `BLOCK_RECORD` (`*Model_Space`, `*Paper_Space`; `AcDbBlockTableRecord`). `0 ENDSEC`.
  3. `0 SECTION`, `2 BLOCKS`; for model space then paper space: `0 BLOCK`, `5 <h>`, `330 <record>`, `100 AcDbEntity`, (paper: `67 1`), `8 0`, `100 AcDbBlockBegin`, `2 <name>`, `70 0`, `10 0.0`, `20 0.0`, `30 0.0`, `3 <name>`, `1 ""`, `0 ENDBLK`, `5 <h>`, `330 <record>`, `100 AcDbEntity`, (paper: `67 1`), `8 0`, `100 AcDbBlockEnd`. `0 ENDSEC`.
  4. `0 SECTION`, `2 ENTITIES`; per curve: `0 SPLINE`, `5 <h>`, `330 <model space record>`, `100 AcDbEntity`, `8 <layer>`, `100 AcDbSpline`, if planar (`allclose(z, z_0)`) `210 0.0`, `220 0.0`, `230 1.0`; `70 <8 if planar else 0>`, `71 <p>`, `72 <#knots>`, `73 <#control points>`, `74 0`, `42 1e-10`, `43 1e-10`, `40 <repr(knot)>` per knot, then per control point `10`, `20`, `30` with `repr`. `0 ENDSEC`.
  5. `0 SECTION`, `2 OBJECTS`, `0 DICTIONARY`, `5 <root>`, `330 0`, `100 AcDbDictionary`, `281 1`, `3 ACAD_GROUP`, `350 <group>`, `0 DICTIONARY`, `5 <group>`, `330 <root>`, `100 AcDbDictionary`, `281 1`, `0 ENDSEC`, `0 EOF`.
- Header, written before the body: `0 SECTION`, `2 HEADER`, `9 $ACADVER`, `1 AC1015`, `9 $HANDSEED`, `5 <next unused handle>`, `9 $INSUNITS`, `70 <units>`, `0 ENDSEC`. The file is header + body lines joined with `\n`, final `\n`.

### 14.7 Merging, arc length and sampling

`arc_length(curve, a=None, b=None, quad_order=16)`: breaks = `unique([a, b] ∪ knots strictly inside)`; 16-point Gauss per break interval on `|C'|`.

`merge_curves(curves)`: one B-spline on `[0, 1]` reproducing a position-continuous chain.

1. All degrees equal, else `ValueError("All sections must have the same degree to be merged.")`.
2. `lengths_0 = arc_length(curve_0)`. For each junction `(a, b)`: `da = a'(end)·(a.t[-p-1] - a.t[p])`, `db = b'(start)·(b.t[-p-1] - b.t[p])`; `smooth = |da| > 0 and |db| > 0 and |da/|da| - db/|db|| < 1e-7`; `length_{s+1} = length_s · (|db|/|da| if smooth else arc(b)/arc(a))`. `bounds = [0, cumsum(lengths)/Σ lengths]`.
3. Per section: knots mapped affinely from `[t[p], t[-p-1]]` to `[bounds_s, bounds_{s+1}]`, first `p+1` set to `bounds_s`, last `p+1` to `bounds_{s+1}`; keep from index `0` (first section) or `p + 1` (others) up to the end (last section) or one before it (others); control points `c[:n]`; for `s > 0`, the previous section's last control point becomes the average of it and this section's first, and this section's first is dropped.
4. Concatenate; set the first `p+1` knots to 0 and last `p+1` to 1.
5. For each interior bound `u`, up to `p - 1` times: `remove_knot(merged, u)`; stop for this bound when it returns the curve unchanged.

`remove_knot(curve, u, rtol=1e-10)`: delete the first occurrence of `u`; sample `x` = unique union over the new spans of `a + (b-a)·0.5(1 - cos(linspace(0, π, 2p+3)))`; least-squares control points fitting `curve(x)` with the new knots; check at `linspace(t_new[0], t_new[-1], 20·len(t_new))`: if `max|candidate - curve| > 1e-10·max(1, max|curve(x)|)`, return the original; else set the candidate's first control point to the original first and its last to the original's control point at index `len(t) - p - 2`, and return it.

`sample(curve, n, spacing="cosine")`: `n < 2`: `ValueError("Need at least two samples.")`. `"parameter"`: `curve(linspace(lo, hi, n))`. Else fractions `cosine = 0.5(1 - cos(linspace(0, π, n)))`, `arc = linspace(0, 1, n)`, `half-cosine-start = 1 - cos(linspace(0, π/2, n))` (other: `ValueError("spacing must be 'cosine', 'arc', 'half-cosine-start', or 'parameter'.")`); parameters by interpolating the arc table (4096 samples) at `fraction·total`; first and last set to `lo`, `hi`.

---

## 15. Project files (`.asfproj`, `project_io`)

### 15.1 Format

UTF-8 JSON, `indent=2`, `ensure_ascii=False`; NumPy scalars are written as Python numbers and arrays as lists. Top level, in this key order:

```json
{
  "format_version": "1.0.0",
  "metadata": {"format_version": "1.0.0", "created": "<ISO>", "modified": "<ISO>", "app_version": "2.0.0", "platform": "<system> <release>"},
  "airfoil": {"coordinates": [[x, y], ...], "sections": [[[x, y], ...], ...], "junction_indices": [..], "num_sections": S, "filename": "...", "name": "...", "precision": d},
  "optimization_runs": [run, ...]
}
```

Run, in key order:

```json
{
  "run_id": "<uuid4>",
  "number": 1,
  "timestamp": "<ISO local, microseconds>",
  "spline_config": {"degree": 5, "num_control_points": 25, "num_sections": 2, "continuity": "G3", "geometric_continuity": true, "junction_continuity": []},
  "optimization_config": {"optimizer": "gn-varpro", "smoothing_weight": 0.001, "optimize_le_position": false, "parameterization": "centripetal", "tolerance": null, "max_points_per_section": null, "knot_placement": "uniform", "grow_control_points": false, "custom_parameters": null},
  "result": {"degree": 5, "final_error": 0.0, "num_iterations": 0, "num_evaluations": 0, "elapsed_seconds": 0.0, "converged": true, "cancelled": false, "section_knots": null, "report": null, "rms_error": null},
  "loss_history": [{"iter": 1, "error": 0.0, "time": 0.0}],
  "section_control_points": [[[x, y], ...], ...]
}
```

The two blocks above show the keys, their order and example values; they are not literal file content (the real file is laid out as Python's `json.dumps(..., indent=2)` writes it: 2 spaces per level, every object member and array element on its own line, `,` directly at the end of a line, `": "` between key and value, empty arrays as `[]`). The file does not end with a line end.

Non-finite numbers are written as the bare tokens `NaN`, `Infinity` and `-Infinity` (Python's `json` default, not valid in standard JSON) and read back as those values. They occur in practice: `report.leading_edge_radius` is infinite for a straight leading edge (13.2). A reader for this format must accept the three tokens (section 22 item 16).

`continuity` is the enum name. `custom_parameters` is a list of number lists or `null`. Timestamps use `datetime.now().isoformat()` (local time without zone, microseconds when non-zero). `platform` is `f"{platform.system()} {platform.release()}"`.

### 15.2 Creating

- `create_project_metadata(created=None)`: `format_version "1.0.0"`, `created = created or now`, `modified = now`, `app_version = __version__`, `platform`.
- `create_project(airfoil)`: new metadata, the airfoil, no runs.
- `create_optimization_run(spline_config, opt_config, result, loss_history)`: `run_id = str(uuid4())`, `timestamp = now ISO`, `section_control_points = result.section_control_points`, `number = 0`.
- `number_runs(runs, after=0)`: `last = max(after, max run numbers)`; each run with `number <= 0`, in list order, gets `last += 1`; returns `last`.

### 15.3 Saving (`save_project(project, path, indent=2)`)

1. Copy the metadata with `modified = now`.
2. Encode the whole document to text first (an exception leaves the file untouched).
3. If `path` exists and is not writable: `PermissionError("{path} is read-only.")`.
4. Create a temporary file in the target directory (`mkstemp`, prefix `.`, suffix `.tmp`); write, flush, `fsync`; if the target exists copy its permission bits to the temporary file; atomically replace the target.
5. On any exception: make the temporary file readable and writable, delete it (ignore OS errors), re-raise.

### 15.4 Loading (`load_project(path)`)

1. Read as `utf-8-sig`. Invalid JSON or UTF-8: `ProjectFormatError("The file is not a project file: {error}.")` (`ProjectFormatError` subclasses `ValueError`). A non-object: `ProjectFormatError("The file is not a project file: it holds no project.")`.
2. `version = str(data.get("format_version", "1.0.0"))`; `int(version.split(".")[0])` fails: `ValueError("Project file version {version!r} is not valid.")`; major `> 1`: `ValueError("Project file version {version} is not supported. Please update the application.")`.
3. `runs = list(data.get("optimization_runs", []))` (errors reported as "Run list"); decode each (where = `"Run {i}"`, 1-based); `number_runs(runs)`; decode `metadata` ("Metadata") and `airfoil` ("Airfoil").
4. Errors inside a part: a missing key -> `ProjectFormatError("{where}: the {key!r} field is missing.")`; `TypeError`, `ValueError`, `IndexError`, `AttributeError` -> `ProjectFormatError("{where}: {error}.")`; nested `ProjectFormatError` propagates unchanged. Parts of a run: `"Run i control points"`, `"Run i spline settings"`, `"Run i fit settings"`, `"Run i result"`, `"Run i loss history"`, then `"Run i"` for the run fields.

Decoding rules:

| Field | Rule |
|---|---|
| control points | `[[float(p[0]), float(p[1])] ...]` per section; shared by the run and its result |
| `degree`, `num_control_points`, `continuity` | required |
| `continuity` | `G0..G3` case-insensitive; anything else -> `G1` |
| `geometric_continuity` | if absent: `str(continuity_type).upper() == "GEOMETRIC"` (absent -> false) |
| `num_sections` | default 2 |
| `junction_continuity` | default `()` (then validated by `SplineConfig`) |
| `optimizer` | required; matches a value (lower-cased); retired names: `hierarchical-lsq` -> VarPro, `constrained-lsq` -> Direct LSQ, `orthogonal-lsq`, `param-opt`, `bfgs`, `slsqp`, `l-bfgs-b`, `cg`, `tnc`, `powell`, `nelder-mead`, `cobyla`, `de`, `da`, `bh` -> VarPro, logging warning `"Optimizer %r was retired; showing the run as %s."`; unknown -> VarPro with warning `"Unknown optimizer %r; showing the run as %s."` |
| `smoothing_weight` | default `1e-3` |
| `optimize_le_position` | default false |
| `parameterization` | default `"centripetal"` |
| `tolerance`, `max_points_per_section` | default `null` |
| `knot_placement` | default `"uniform"` when the run has no tolerance, else `"free"` |
| `grow_control_points` | default: the run has a tolerance |
| `custom_parameters` | `null` or list of float arrays |
| result `degree`, `final_error`, `num_iterations`, `num_evaluations`, `elapsed_seconds` | required |
| result `converged` / `cancelled` | default true / false |
| result `section_knots`, `report`, `rms_error` | default `null` |
| loss record | `iter`, `error` required; `time` default 0.0 |
| `run_id`, `timestamp` | required |
| `number` | `int(data.get("number", 0))` |
| metadata fields | defaults `"1.0.0"`, `""` ... |
| airfoil `coordinates`, `sections`, `junction_indices`, `num_sections` | required; `filename`, `name` default `""`, `precision` default 6 |

Unknown keys are ignored (older projects carry `fairing`, `metric`, `seed`, `smoothing_factor`, `early_stop_patience`, `relative_tolerance`, `max_iterations`, `early_stopped`, `continuity_type`, `eval`).

`get_run_summary(run)`: lines `Run ID: {run_id[:8]}...`, `Time: {timestamp}`, `Optimizer: {display_name}`, `Degree: {p}, Control Points: {n} per section, {fitted} fitted`, `Continuity: {", ".join(specs) or continuity name}`, `Tolerance: {tol:.3e}` (only when truthy), `Final Error: {final_error:.6e}`, `Iterations: {nit}, Evaluations: {nev}`, `Elapsed: {s:.2f}s`, `Converged: {True/False}`, joined by `\n`.

---

## 16. Batch output naming (`gui/batch.py`)

Reference settings: `DEFAULT_SPLINE = SplineConfig(continuity=G3, geometric_continuity=True)` (degree 5, 25 control points, 2 sections) and `DEFAULT_OPTIMIZATION = OptimizationConfig(knot_placement="uniform", grow_control_points=False)` (VarPro, centripetal, smoothing 1e-3, no tolerance, no subsampling, junctions pinned).

`num(v)`: if `v != 0` and `|v| < 1e-2`: `mantissa, exponent = f"{v:.2e}".split("e")`; result `mantissa.rstrip("0").rstrip(".") + "e" + str(int(exponent))`; else `f"{v:.3g}"`. Examples: `1e-4`, `2.5e-5`, `1e-3`, `3.33e-7`, `0.0123`, `0.5`.

`output_name(stem, config, opt, auto_tolerance=False)`: parts, joined by `.`, then `.spline.dat`:

1. `stem`, `deg{p}`.
2. `n{n}` when (`opt.tolerance is None` or not `opt.grow_control_points`) and `n != 25`.
3. `s{S}` when `S != 2`.
4. `specs = junction_specs()`; when non-empty and not all equal to `"G3"`: `"-".join(specs)` if they are not all equal, else the single spec.
5. When the optimizer differs from VarPro: `lsq` for Direct LSQ (else the value); then, still only in that case, the parameterization with `_` -> `-` when it is not `centripetal`.
6. When `tolerance` is not `None`: `tol-auto` if `auto_tolerance` else `tol{num(tolerance)}`.
7. `knots-{placement}` when not `uniform`.
8. `grow` / `no-grow` when it differs from false (so only `grow`).
9. `smooth{num(w)}` when `w != 1e-3`.
10. `sfactor{num}` when `smoothing_factor` is not `None`; `sub{m}` when `max_points_per_section` is not `None`; `free-le` when junctions are free; `patience{k}` when `!= 50`; `rtol{num}` when `!= 1e-6`.

Examples: section 21 item 8.

---

## 17. Optimizer logs (`optimization/run_log.py`)

`OptimizerErrorLogger(run_name="unnamed", log_dir=None)`: `start_time = now`; `log_dir = log_dir or OptimizerErrorLogger.DEFAULT_LOG_DIR (class attribute, None) or default_log_dir()`. `log(iteration, evaluation, error, elapsed=0.0)` appends to four lists.

`default_log_dir()`: `SPLINE_FITTER_LOG_DIR` if non-empty; else if `pyproject.toml` exists in the directory four levels above `run_log.py` (`src/spline_fitter/optimization/run_log.py` -> checkout root), `<root>/optimizer_logs`; else `~/.spline_fitter/optimizer_logs`.

`save(optimizer_key="unknown", extra_info="")`:

1. No records: return `None`.
2. Create the directory (`exist_ok`); an `OSError` prints `Failed to save optimizer log: {e}` and returns `None`.
3. Stem `"{start:%Y%m%d_%H%M%S}_{sanitize_filename(run_name)}_{optimizer_key}"`; path `<stem>.csv`, else `<stem>_2.csv`, `_3`, ... (first not existing); open with exclusive creation, UTF-8.
4. Content, each line ending with a line end (the platform's, section 1 "Text files"): `# Optimizer Error Log`, `# Run: {run_name}`, `# Optimizer: {key}`, `# Start Time: {start ISO}`, `# Info: {extra}` (only if non-empty), `# Total Iterations: {count}`, `# Total Evaluations: {last evaluation}`, `# Final Error: {last:.12e}`, `# Min Error: {min:.12e}`, `# Total Time: {last elapsed:.3f}s`, `#`, `iteration,evaluation,error,elapsed_s`, then rows `{it},{ev},{err:.12e},{t:.3f}`.
5. Any exception: print `Failed to save optimizer log: {e}`, return `None`. Success returns the path (also `file_path`).

---

## 18. Display formatting (`formatting.py`)

- A process-global display decimal point, default `.`; `set_decimal_point(c)` sets it (`c or "."`); `decimal_point()` reads it. The GUI sets it from the system locale at startup (19.19). Files never use it.
- `number(v, spec="g") = format(v, spec).replace(".", decimal_point())`.
- `format_error(v)`: not convertible to float -> `str(v)`; non-finite -> `str(float)` (`inf`, `nan`); `0 < |v| < 1e-3` or `|v| >= 1e4` -> `number(v, ".3e")`; else `number(v, ".6f")`.
- `format_duration(s)`: `s < 1e-3` -> `f"{s·1e6:.0f} µs"`; `< 1` -> `f"{s·1e3:.0f} ms"`; `< 60` -> `number(s, ".2f") + " s"`; else `m, sec = divmod(s, 60)`: `m < 60` -> `f"{int(m)}m {int(sec)}s"`; else `h, m = divmod(int(m), 60)` -> `f"{h}h {m}m {int(sec)}s"`. The µs and ms forms do not localize.
- `format_count(v, singular, plural=None)`: plural defaults to `singular + "s"`; noun singular iff `v == 1`; `f"{v:,} {noun}"` with `,` replaced by `.` when the decimal point is `,`.
- `format_percentage(v, precision=1) = number(100·v, f".{precision}f") + "%"`.
- `sanitize_filename(s)`: every character not in `A-Z a-z 0-9 - _ .` becomes `_`.
- `file_stem(s, default="airfoil")`: characters `< > : " / \ | ? *` and code points below 32 become `_`; then `strip()`, then strip trailing `.` and spaces; empty -> `default`.

---

## 19. Graphical user interface (`spline_fitter.gui`)

PyQt6 widgets and pyqtgraph plots. All colours, fonts and sizes come from the theme (19.19). `UNLIMITED = 2³¹ - 1` is the maximum of integer fields without a limit; floating fields without a limit use +∞.

### 19.1 Window construction (`SplineFitterWindow`)

In this order:

1. `apply_theme()` (19.19) and `install_smooth_scrolling()` (19.20), then the `QMainWindow` is created; title `Airfoil Spline Fitter`.
2. State: `airfoil = None`, `fit_result = None`, `is_fitting = False`, `cancel_requested = False`; `QSettings("MSFra", "AirfoilSplineFitter")`; empty sweep group.
3. Build the menu bar (19.3) and the layout (19.2); status bar text `Ready`.
4. Create the plot manager (19.14), the project manager (19.16), the fitting controller (19.12) and the batch controller (19.18); connect signals; take the Advanced baseline (19.9).
5. Create the settings history (19.8) over these widgets, in this order: optimizer, goal, Auto tolerance, tolerance, knot placement, parameterization, grow points, sections, degree, continuity level, continuity type, control points, per-junction text, subsample check, subsample count, smoothing weight, optimize junctions, then the sweep widgets in creation order (degree, control points, sections, smoothing). Undo/Redo actions follow its `changed` signal.
6. Accept drops (19.11).

On the first show, `size_sidebar()` runs (19.2). On every resize, while no splitter handle has been dragged, the sidebar and explorer are reset to their default widths. On close: if the project manager's `confirm_discard()` (19.16) is false the close is ignored; if a fit runs, the cancel flag is set; then the window closes.

### 19.2 Layout

Central widget, left to right: a horizontal splitter `[sidebar | centre | explorer]` (stretch factors 0, 1, 0), then the explorer rail.

- **Splitter**: initial sizes 330, 440, 570 px; sidebar minimum 300, explorer minimum 180; no pane collapsible; a handle is 1 px. Double-clicking handle 1 sets the sidebar to its default width, handle 2 the explorer to 570; the difference is taken from or given to the centre. Dragging a handle sets `panels_dragged` (resizes no longer reset widths).
- **Sidebar default width** (`size_sidebar`, on first show): expand Advanced; `w` = the sections' minimum width; for each sweep widget that is enabled and not sweeping, turn its sweep on, measure, turn it off and restore its state (inside the history's quiet scope, so no step results); restore Advanced; default width = `max(330, ceil((w + vertical scrollbar width)/10)·10)`; reset both panel widths.
- **Sidebar**: a vertical scroll area (no frame, horizontal bar never shown) holding, top to bottom: Data card, Method section, Curve section, Advanced section (collapsible, collapsed), stretch; below it the fixed run bar (19.5). Every combo box in the sidebar sizes to a minimum of 6 characters; every combo box and spin box expands horizontally with a minimum width of 70 px.
- **Sidebar forms**: label column 92 px (labels have object name `FormLabel`), horizontal spacing 8, vertical 6, fields expand; labels left/vertically centred.
- **Centre**: margins 8, spacing 6: the plot toolbar (19.14.1), the main plot (stretch 3), and a reflowing row (19.21, spacing 8) of the loss plot block and the comparison plot block (stretch 2). Each block is a toolbar above its plot (spacing 6).
- **Explorer panel** (`Explorer`): margins 12, 10, 12, 12, spacing 8: a header (caption `PROJECT`, stretch, an icon button with `chevron_right` and tooltip `Hide the project explorer (Ctrl+Shift+E)`), the project label (`Muted`, word wrap, initially `No project`), the run table (19.17, stretch 1), the run details (`ReadoutList`, initial message `Select a run to view details`), and a button row `Load settings` (disabled; tooltip `Set the sidebar to the selected run's settings (or double-click the run)`) and `Delete` (disabled; tooltip `Delete the selected runs from the project (Delete key).` / `Edit > Restore deleted runs brings them back.`), then stretch.
- **Explorer rail** (`ExplorerRail`): hidden at start; margins 4, 10, 4, 12; one icon button with `chevron_left`, tooltip `Show the project explorer (Ctrl+Shift+E)`, then stretch.
- Hiding the explorer (button, menu or shortcut) hides the panel and shows the rail; showing does the reverse. The hide button unchecks View > Project explorer; the rail button checks it.

### 19.3 Menus

| Menu | Item text | Shortcut | Initial state | Action |
|---|---|---|---|---|
| `&File` | `&New NACA airfoil...` | Ctrl+N | enabled | 19.11.5 |
| | `&Open airfoil...` | Ctrl+O | enabled | 19.11.1 |
| | `Open &recent` (submenu, tooltips shown) | | enabled | 19.11.3 |
| | separator | | | |
| | `Save fitted airfoil (.dat)...` | | disabled | 19.15.1 |
| | `Save spline (.spline.dat)...` | | disabled | 19.15.2 |
| | `Export CAD (STEP, IGES, DXF)...` | Ctrl+E | disabled | 19.15.3 |
| | separator | | | |
| | `Open &project...` | Ctrl+Shift+O | enabled | 19.16.2 |
| | `&Save project` | Ctrl+S | disabled | 19.16.3 |
| | `Save project &as...` | | disabled | 19.16.3 |
| | separator | | | |
| | `&Batch convert...` | | enabled | 19.18 |
| | separator | | | |
| | `E&xit` | Alt+F4 | enabled | close the window |
| `&Edit` | `&Undo` | platform Undo keys (Ctrl+Z on Windows) | disabled | 19.8 |
| | `&Redo` | platform Redo keys (Ctrl+Y, Ctrl+Shift+Z on Windows) | disabled | 19.8 |
| | `Restore &deleted runs` | | disabled | 19.16.6 |
| | separator | | | |
| | `&Paste airfoil` | platform Paste keys (Ctrl+V) | enabled | 19.11.4 |
| `&View` | `&Fit report...` | Ctrl+R | disabled | 19.15.4 |
| | separator | | | |
| | `Project &explorer` (checkable, checked) | Ctrl+Shift+E | enabled | show/hide explorer |
| | separator | | | |
| | `&Theme...` | | enabled | 19.19.4 |
| | `&Scrolling and zooming...` | | enabled | 19.20.3 |
| `&Help` | `&About` | | enabled | status bar `Airfoil Spline Fitter v1.0` (no dialog) |

The paste shortcut reaches the action only when the focused widget does not consume it (a focused text field pastes text instead).

### 19.4 Sidebar controls

**Data** (section caption `DATA`): a bold file name label (`FileName`, initially `No airfoil loaded`), below it a muted label (initially `Open a .dat file or a project`), and at the right a button `Open...` (tooltip `Open an airfoil coordinate file (Ctrl+O)`) that does File > Open airfoil. After loading, the name label shows `name or filename or "Airfoil"`, the muted label `"{N} points, {precision} decimals"`, and both have the file path as tooltip (empty for pasted or generated airfoils).

**Method** (caption `METHOD`), rows in order:

| Label | Widget | Items / range | Default | Tooltips |
|---|---|---|---|---|
| `Optimizer` | segmented | `VarPro LM` (data VarPro), `Direct LSQ` (data Direct LSQ) | VarPro LM | per item, 5.3 |
| `Goal` | segmented | `Fixed count` (false), `Tolerance` (true) | Fixed count | `Fit the given number of control points per section.`; `Find the fewest control points that keep` / `every data point within the tolerance.` |
| `Max deviation` | scientific spin box + check box `Auto` | spin `1e-9 .. +∞`, step `1e-6` | spin `1e-5`, Auto checked | spin: `Maximum orthogonal distance of any data point to the curve.`; Auto: `Derive the tolerance from the decimal precision of the file and` / `the estimated noise of the points.` |
| `Knots` | segmented | `Uniform` (`"uniform"`), `Free` (`"free"`) | Uniform | `Knots stay evenly spaced in the curve parameter.`; `Knots move to where they reduce the error most.` |
| `Parameterization` | segmented | `Centripetal` (`"centripetal"`), `Arc length` (`"arc_length"`) | Centripetal | `Space the data parameters by the square root of the point distances.` / `Follows tight curvature, such as the leading edge, more closely.`; `Space the data parameters by the distance along the data.` |
| `Grow points` | check box `Add where the fit is worst` | | unchecked | `On: start from the fewest control points and add them where the` / `fit is worst, refitting after each step.` / `Off: start from the Control points setting (Curve). With a tolerance` / `goal the count can still grow or shrink from there to meet it.` |

The tolerance spin box and Auto share one row (spin stretches, spacing 6).

**Curve** (caption `CURVE`):

| Label | Widget | Range | Default | Tooltip |
|---|---|---|---|---|
| `Degree` | spin box with sweep (19.21 item 3) | 1 .. UNLIMITED | 5 | `Polynomial degree of B-spline basis functions.` / `Typical: 3 (cubic), 4, or 5 (quintic).` |
| `Control points` (`Start count` in tolerance goal) | spin box with sweep | 2 .. UNLIMITED | 25 | `Control points per section.` / `More = more flexibility but more variables.` |
| `Sections` | spin box with sweep | 1 .. UNLIMITED, maximum reset to `max(1, N - 1)` for each airfoil | 2 | `Number of sections, each its own spline, joined at junctions.` / `2 (default) splits upper and lower surface at the leading edge.` |
| `Continuity` | segmented, 4 items with data G0..G3; texts `C0..C3` or `G0..G3` (19.6) | | index 3 (G3) | item `i`: `{Position, Tangent, Curvature, Curvature rate}[i] matches across junctions` |
| `Type` | segmented `Geometric` (true), `Parametric` (false) | | Geometric | `Geometric: tangent direction, curvature and curvature rate match` / `across the junction (what CAD calls G1/G2/G3).` / `Parametric: the derivative vectors themselves match (C1/C2/C3),` / `which is stricter and needs more control points.` / `Only VarPro LM supports geometric continuity; other optimizers` / `always use parametric.` |

**Advanced** (collapsible, header ` ADVANCED`, collapsed):

| Label | Widget | Range | Default | Tooltip |
|---|---|---|---|---|
| `Per junction` | line edit, placeholder `G2, C1  (blank = same everywhere)` (two spaces before the parenthesis) | | empty | `Continuity of each junction in order, overriding the Continuity` / `setting. One entry per junction (sections - 1), each G0-G3 or C0-C3.` / `VarPro LM only.` |
| `Subsample` | check box `to` + spin box with suffix ` points` | spin 20 .. UNLIMITED | unchecked; 200; the spin is enabled only while the check box is checked | check: `For dense data: fit an evenly spaced subsample of each section, then` / `refine the fit on all points. Much faster for thousands of points.` / `Off: every point is used throughout.`; spin: `Points per section. Sections with more points than this are subsampled.` |
| `Smoothing` | scientific spin box with sweep (19.21 item 4) | 0 .. +∞, step 0.001 | 0.001 | `Bending energy weight for VarPro LM (0 turns it off; default 0.001).` |
| `Junctions` | check box `Optimize positions` | | unchecked | `Optimize the position of every junction, instead of keeping the` / `section split points fixed. The split points are where the junctions` / `start; the data is fitted by the whole curve, so points move between` / `sections with them.` |

(` / ` in tooltips marks a line break.)

### 19.5 Run bar

Fixed below the sidebar (`RunBar`), margins and spacing 12:

- `Fit spline` (`PrimaryButton`, pointing-hand cursor) and `Cancel` (hidden and disabled until a fit runs; then the Fit button is hidden and disabled instead).
- A result box, hidden until there is a fit: a grid with captions (`Muted`) over readouts (`Readout`): column 1 caption `Max deviation` (or `Error`), column 2 `Control points`, column 3 `Time`; below it buttons `Save .dat` (tooltip `Save fitted airfoil coordinates (.dat)`), `Save spline` (`Save spline control points and knots (.spline.dat)`), `Export CAD` (`Export the exact B-spline curves as STEP, IGES or DXF`), all disabled until there is a fit.

`set_save_enabled(e)`: the three buttons, the three File actions and View > Fit report take `e`; the result box is shown when `e` and a fit exists. Its readouts: with a report, caption `Max deviation`, value `number(max_deviation, ".2e")`; else caption `Error`, value `format_error(final_error)`. Control points: one section -> its count; all sections equal -> `"{S} × {n}"`; else the sum; tooltip `"{c1}, {c2}, ... per section"` when there are several sections, else empty. Time: `format_duration(elapsed_seconds)`.

### 19.6 Visibility and enabling (`on_optimizer_changed`)

Runs when the optimizer, goal, Grow points, Auto, or section count changes, after undo/redo, after applying a run, and from `set_fitting_ui`. Let `varpro` = optimizer is VarPro, `tolerance_mode = varpro and goal is Tolerance`, `has_junctions = sections > 1`, `editable = not is_fitting`, `count_used = not tolerance_mode or not Grow points`.

1. Shown only for VarPro: Goal, Max deviation row, Knots, Grow points, Type, Per junction, Subsample row, Smoothing. Shown only for Direct LSQ: Parameterization.
2. `Max deviation` label enabled iff `tolerance_mode`; tolerance spin enabled iff `editable and tolerance_mode and not Auto`; Auto enabled iff `editable and tolerance_mode`.
3. Control points label text `Start count` if `tolerance_mode` else `Control points`; label enabled iff `count_used`. In tolerance mode, each of the control-point and smoothing sweeps that is on is turned off (`stop_sweep`, which emits its `sweep_changed` and so runs the curve rule with `"count"` for the control-point sweep); a sweep that is already off is left alone, with no signal. Only after that is the control-point sweep toggle's allowed state set (19.21 item 3: disallowing turns a sweep that is on off). Control-point sweep widget enabled iff `editable and count_used`; its sweep toggle is allowed only when not `tolerance_mode`.
4. Continuity and Type labels enabled iff `has_junctions`; their controls enabled iff `editable and has_junctions`. Continuity item texts are `G{i}` when VarPro and Type is Geometric, else `C{i}` (also updated when Type changes).
5. Per junction label and field enabled iff `has_junctions`; Smoothing label and widget enabled iff not `tolerance_mode`; Junctions label and check box enabled iff `has_junctions`. While a fit runs, the whole Advanced content widget is disabled (`set_fitting_ui`), so every widget inside it then reports disabled regardless of these rules.
6. Update the Advanced badge (19.9) and run the curve rule (19.7) with `changed = ""`.

`set_fitting_ui(fitting)`: `is_fitting = fitting`; Fit hidden/disabled and Cancel shown/enabled while fitting (reverse otherwise); disabled while fitting: the Open button, Open airfoil, Open project, New NACA, Open recent, Paste airfoil, Batch convert, Undo/Redo, the explorer's Load settings and Delete (otherwise Load settings needs exactly one selected run and Delete at least one), Restore deleted runs, the degree and section sweep widgets, the control-point spin box, Continuity, Type, Optimizer, Goal, the tolerance row, Knots, Parameterization, Grow points and the whole Advanced content; then `on_optimizer_changed`; finally `set_save_enabled(not fitting and fit exists)`.

### 19.7 Curve validity rule (`keep_curve_valid(changed)`)

Keeps degree, control points, sections and continuity fittable. Triggered with `changed = "degree"` (degree value or its sweep changed), `"count"` (control points or its sweep), `"sections"` (sections or its sweep), `"continuity"` (level changed), and `""` (Per junction editing finished, and from 19.6). It does nothing while it is already running (re-entrancy guard, also set until the signals are connected) or when the continuity has no current item.

Definitions: for a sweep widget, `ends()` = `[value box]`, plus `[sweep end box]` while sweeping. `open_sweep` = control points sweeping and (degree or sections sweeping). `low_end(w)` = all ends, or with `open_sweep` only the end of smallest value (the first in `ends()` order, the value box, on a tie); `high_end(w)` likewise with the largest (first on a tie). `junction_orders(S)`: if VarPro and the Per junction text (`,` and `;` replaced by spaces, split on whitespace) has at least one entry and exactly `S - 1` entries (so never with one section), each of 2 characters with first in `CGcg` and second in `0123`: `(their digits, True)`; else `([Continuity order] * (S - 1), False)`. `orders()` = `junction_orders(max value over low_end(sections))`. `need()` = `minimum_control_points(orders, max value over low_end(degree))`. `lower_continuity(k)`: set the Continuity index to `min(k, current order)`. `move(boxes, limit, up)`: set each box below (up) or above (down) `limit` to `limit`.

1. `lowest = min degree end value`. If `max(orders) > lowest` (`max` of no orders, one section, is 0): if `changed` is `"degree"` or `"count"` and the orders do not come from Per junction: `lower_continuity(lowest)`; else raise all degree ends to `max(orders)`.
2. If `count_used` (19.6): `have = min value over high_end(count)`; if `need() > have`:
   - if `changed == "count"`: lower `low_end(degree)` boxes to `have - 1`; then if still `need() > have` and sections are sweeping, lower the sections sweep's last end (the end box) to `max(Sections value, 2)` if it is above that (it is never raised here); then while `need() > have`, the orders are not from Per junction and the Continuity order is above 0: `lower_continuity(order - 1)`;
   - else raise `high_end(count)` boxes to `need()`.
3. Hard minimums: `(fixed_orders, fixed) = junction_orders(Sections value)`; degree minimum (value and sweep end) = `max(fixed_orders)` when `fixed` (so entries all of order 0 give a minimum of 0), else 1; control-point minimum = `minimum_control_points(fixed_orders, that degree minimum)` when `fixed`, else 2.
4. Sweep caps: `degree_cap` = degree spin maximum, `sections_cap` = sections spin maximum. If `count_used` and control points are not sweeping: `have` = control points value; `degree_cap = max(have - 1, degree minimum)`; `fitting` = the section counts `k` in `1..min(sections_cap, 3)` with `minimum_control_points([order]*(k-1), 1) <= have` (order = Continuity order); if `fitting[-1] < 3`, `sections_cap = fitting[-1]`; `sections_cap = max(sections_cap, Sections value)`. The degree sweep end maximum becomes `degree_cap`, the sections sweep end maximum `sections_cap` (the value boxes keep their own maxima).

### 19.8 Settings history (undo/redo)

Constants `MERGE_MS = 500`, `LIMIT = 200`. Widget values: line edit -> text; check box -> checked; combo or segmented -> current index; spin box -> value; sweep widget -> its sweep state (19.21). Each widget's change signal (text changed, toggled, index changed, value changed, sweep changed) feeds `on_change`.

State: list of states (tuples of all values) starting with the initial one, current index 0, a single-shot 500 ms timer, `pending_quiet`, `latest`.

- `on_change`: ignored while restoring. `quiet = (inside quiet()) or is_quiet()` (`is_quiet` = a fit is running). If the timer is active and `quiet != pending_quiet`: settle `latest` with `pending_quiet` first. Then `pending_quiet = quiet`, `latest = capture()`, restart the timer, emit `changed`.
- `commit()` (timer timeout, or called): stop the timer; settle `capture()` with `pending_quiet`; reset `pending_quiet = False`, `latest = None`; emit `changed`.
- `settle(state, quiet)`: quiet -> replace the current state; else if `state != current`: drop states after the current, append, keep only the last 200, index = last.
- `can_undo = index > 0 or (timer active and not pending_quiet)`; `can_redo = index < last and timer not active`.
- `undo`: commit; if `index > 0`: `index -= 1`, restore. `redo`: commit; if `index < last`: `index += 1`, restore.
- `restore`: with `restoring` set, set every widget in list order, call `after_restore` (`on_optimizer_changed`), then store the widgets' actual values as the current state (limits may clamp); emit `changed`.
- `checkpoint()`: commit, return a copy of `(states, index)`. `rollback(cp)`: stop the timer, clear pending, restore `(states, index)` (widgets untouched), emit `changed`.
- Keyboard: an application event filter consumes the shortcut-override event of the Undo or Redo key sequence when the focus widget is one of the settings widgets or inside one, so the Edit actions receive the key instead of the field.

Quiet changes: the automatic tolerance set when an airfoil loads, `size_sidebar` measuring, restoring after a failed project open, and every change while a fit runs.

### 19.9 Advanced badge

Tracked fields: `("Per junction", text)`, `("Subsample", check)`, `("Subsample points", spin)`, `("Smoothing", spin)`, `("Junctions", check)`. The baseline (value and display text of each) is taken after the section is built and again once the window is constructed. A field counts as changed when it is visible within the Advanced content (regardless of collapse) and its value (text stripped; checked; index; numeric value) differs from its baseline. Badge text `"{k} changed"` (empty when none); tooltip `Changed this session:` followed by one line per field `"{name}: {old} → {new}"` (texts: check boxes `on`/`off`; spin boxes their displayed text; empty text `(blank)`). Updated on every change of these fields and from 19.6.

### 19.10 Building configurations from the sidebar

`spline_config()`: `junctions` = for VarPro, the Per junction text with `,` and `;` replaced by spaces, split on whitespace (as given, upper-cased by `SplineConfig`); else `()`. `SplineConfig(degree, control points, sections, Continuity data, geometric = VarPro and Type is Geometric, junctions)`. Its `ValueError`s reach the user (19.12).

`optimization_config()`: tolerance = `None` unless VarPro and goal Tolerance; then, if Auto and an airfoil is loaded, the spin box is first set to `default_tolerance(coords, precision)` (any exception ignored), and the tolerance is the spin value. `OptimizationConfig(optimizer, optimize_le_position = checked and sections > 1, smoothing_weight = spin, tolerance, max_points_per_section = subsample count if VarPro and checked else None, knot_placement, parameterization, grow_control_points = checked)`.

`apply_run_settings(run)` (explorer double-click, Load settings, opening a project), each `choose` selects the item with that data if present:

1. optimizer; goal (`tolerance is not None`); if a tolerance: uncheck Auto and set the tolerance; knot placement; parameterization; Grow points;
2. sections value; Continuity; degree; Type (geometric flag); control points;
3. Per junction text `", ".join(junction_continuity)`; Subsample check (`max_points is not None`) and, when set, its count; smoothing weight; Optimize positions;
4. `on_optimizer_changed`.

All changes fall within 500 ms and form one undo step.

### 19.11 Opening and creating airfoils

#### 19.11.1 From a file

Open dialog: title `Open airfoil`, home directory, filter `Airfoil data (*.dat)`. `load_airfoil(path)`:

1. Extension (lower-cased) not `.dat`: status `Unsupported file type: {ext}`; stop.
2. Status `Loading: {basename}...`; process events.
3. `parse_airfoil_data`; an exception: status `Failed to load: {e}`, error chime; no coordinates: status `No coordinates found in the file`, error chime.
4. `set_airfoil(coords, filename = filename_name or basename, name = content_name or filename_name or "", precision = the int precision or 6, path)`; on success add the path to recent files.

`set_airfoil(coords, filename, name, precision, path="", verb="Loaded")`:

1. `confirm_discard()` false: status `Cancelled: the project stays open`; return false.
2. Set the Sections maximum to `max(1, len(coords) - 1)` (this may lower the Sections value, which re-splits the previous airfoil, 19.11.6).
3. `create_airfoil_data(coords, filename, name, precision, Sections value)`.
4. Clear the fit, `set_save_enabled(False)`, reset the project (19.16.1), show the file card, `plot_airfoil(coords)` (clears fit items and the loss trace, sets the data points, auto-ranges).
5. Quietly (19.8) update the automatic tolerance (if Auto is checked).
6. Status `"{verb}: {name or filename} ({len(coords)} points, {S} sections: {sizes})"`, where `sizes` is the Python list text of point counts per section (e.g. `[81, 81]`). Return true.
7. Any exception in 2-6: status `Failed to load: {e}`, error chime, return false.

`open_path(path)`: while fitting: status `Cancel the running fit to open a file`. Path not a file: status `File not found: {path}`, remove it from recent files, error chime. Extension `.asfproj` (case-insensitive): open the project (19.16.2). Else `load_airfoil`.

#### 19.11.2 Drag and drop

A drag is accepted when not fitting and its URLs include a local file with extension `.dat` or `.asfproj` (case-insensitive). On drop: activate the window and `open_path` the first such file.

#### 19.11.3 Recent files

QSettings key `recent_files` (a single string read back is treated as a one-item list). `add_recent(path)`: absolute path at the top, others with a different case-normalized path after it, at most 10. `remove_recent(path)` removes case-normalized matches. The submenu is rebuilt each time it opens: with no files, one disabled item `No recent files`; else for `i = 1..`: text `&{i} {basename}` for `i < 10`, else `{basename}` (any `&` in the name doubled), tooltip the full path, action `open_path`; then a separator and `Clear recent files` (sets the list empty).

#### 19.11.4 Paste airfoil

Ignored while fitting. If the clipboard has supported file URLs (as 19.11.2), `open_path` the first. Else take its text (empty if none), `parse_airfoil_text(text)` (an exception counts as no coordinates). Fewer than `MIN_PASTED_POINTS = 5` coordinates: status `No airfoil coordinates on the clipboard: copy x y pairs, one point per line`, error chime. Else `set_airfoil(coords, filename="", name=content_name or "Pasted airfoil", precision=precision or 6, verb="Pasted")`. Pasted airfoils are not added to recent files.

#### 19.11.5 New NACA airfoil

Dialog `New NACA airfoil` (form with 16 px padding):

- `Digits`: line edit, initial QSettings `naca_code` (default `2412`), validator: exactly four digits; tooltip `Four digits: the camber in percent of the chord, its position in tenths` / `of the chord, and the thickness in percent. 2412 has 2% camber at 40%` / `of the chord and is 12% thick; 0012 is symmetric.`
- `Points per side`: spin box 10 .. UNLIMITED, initial `naca_points` (default 81); tooltip `Spaced closer toward the leading and trailing edge`.
- `Trailing edge`: segmented `Closed` (true; tooltip `Modified thickness coefficient: a sharp trailing edge`), `Open` (false; tooltip `Original thickness coefficient: a thin blunt trailing edge`); initial `naca_closed_te` (default true).
- A button box (19.21 item 13) with `Create` (accept role, default) and `Cancel` (reject role). `Create` is enabled exactly while the digits are acceptable (updated on every edit).

On Create: save the three values to QSettings, then `new_naca_airfoil(code, n, closed)`: ignored while fitting; `NACA4(code, closed).points(n)` rounded to 6 decimals (a `ValueError`: status `Cannot create the airfoil: {e}`, error chime); `set_airfoil(points, filename="naca{code}", name="NACA {code}", precision=6, verb="Created")`.

#### 19.11.6 Section count changes

When the Sections value changes and an airfoil is loaded: re-split with `create_airfoil_data(coords, filename, name, precision, S)`; clear the fit; `set_save_enabled(False)`; status `"Re-split: {name or filename} ({N} points, {S} sections: {sizes})"`; `plot_airfoil(coords)`. An exception: status `Failed to re-split airfoil: {e}`, error chime (the previous split stays). The project is not reset.

### 19.12 Fitting

#### 19.12.1 Fit (`on_fit`, the Fit spline button)

1. No airfoil: status `Load an airfoil first.`.
2. Build both configurations (19.10); a `ValueError`: status = its message.
3. If any sweep widget is sweeping: run a sweep (19.13) instead.
4. Clear the sweep state. Status `"Preparing {size}, degree {p}, {mode} continuity..."` where `size` = `"tolerance {number(tol, '.2e')}"` if a tolerance, else `"{n} control points"`, and `mode = ", ".join(junction_specs()) or continuity name`; process events.
5. `cancel_requested = False`; clear the plot items; reset the loss plot (label `Iterations` or `Time (s)` by the X axis setting, log modes by the scale setting); `reset_iteration_tracking()`; `set_fitting_ui(True)`; start a timer; `logger = OptimizerErrorLogger(airfoil name or filename)`; status `Starting {display name}...`.
6. `live = LiveProgress(...)` (19.12.2). Work (worker thread): `result = run_optimization(config, opt, airfoil, callback=live.callback, should_stop=lambda: cancel_requested)`; `cancelled = cancel_requested` (read once here); `elapsed`; `fit = make_fit_result(...)` (19.12.4). Run it with `run_in_background(work, live.on_tick)` (19.12.3).
7. If `fit` exists: it becomes the current fit; `kept = not add_optimization_run(config, opt, fit, loss_history)` (19.16.4; loss history from the logger: `IterationRecord(iteration, error, elapsed)`); plot the fit; refresh the comb. Else redraw the previous fit (if any).
8. `live.show_loss()`.
9. Save the log: `logger.save(optimizer value, "continuity={c}, degree={p}")` with `c` = `"none"` for one section, the single spec when all are equal, else specs joined by `/`; print `Log saved: {path}` when saved.
10. Status: `effort` = tolerance mode: `"{nit} refit(s), {evaluations} evaluation(s)"`; Direct LSQ: `"{evaluations or nit} solve(s)"`; else `"{evaluations} evaluation(s)"` (singular when 1, plural noun `+s`; `evaluations = get_evaluation_count()`). Cancelled: `"Cancelled: {effort} in {format_duration(elapsed)} | Best error {format_error(result.fun)}"`; else `"Completed ({display name}): {effort} in {duration} | Error {format_error(result.fun)}"`. Then `" | " + summary_line(report)` when the fit has a report; `" | Solver did not converge: {message}"` when not successful and not cancelled; `" | The project keeps its finished run of these settings"` when `kept`.
11. Completion chime unless cancelled.
12. Any exception in 6-11: print the traceback, status `Fitting error: {e}`, error chime, redraw the previous fit.
13. Always `set_fitting_ui(False)`.

Cancel button: `cancel_requested = True`; status `Cancelling... please wait`.

#### 19.12.2 Live preview (`LiveProgress`)

Modes (combo `Live preview`: `Minimal`, `Moderate`, `Detailed`; default `Moderate`; tooltip `How often the plots update while fitting.` / `Minimal: only at the end. Moderate: once a second.` / `Detailed: every iteration (slower).`). The current index is copied into a plain integer on every change and may change during a fit. `MODERATE_INTERVAL = 1.0` s, `STATUS_INTERVAL = 0.5` s. Constructed with the window, configs, logger, start time, `status_prefix` (default `""`) and the airfoil (default the window's).

`callback(x, error, preview=None, step=False)` (worker thread), returns true to stop:

1. `cancel_requested`: return true.
2. If `step`: mark that steps arrive. Else if steps have arrived: remember `error` as the last refit's deviation; return false.
3. Record: `iteration = increment_iteration()`; `logger.log(iteration, get_evaluation_count(), error, now - start)`.
4. Status text: `"{prefix}Fitting... {format_duration(elapsed)} | Evaluations {count} | "` + (steps arrive: `"RMS error {format_error(error)}"` plus `" | Max deviation of last refit {format_error(dev)}"` when known; else `"Best error {format_error(min of logged errors)}"`).
5. Clear the "drawn" event; publish `{x copy, preview, new: True, status}`.
6. Detailed mode: request an immediate tick (19.12.3) and wait for the "drawn" event in 50 ms slices, returning true if a cancel is requested meanwhile.
7. Return false.

`on_tick()` (GUI thread): nothing new -> return. If Detailed, or Moderate and at least 1.0 s since the last drawing: draw (show the loss, `update_live_plot(x, config, preview, tolerance)` (19.14.4), status = the published text), mark drawn. Otherwise mark drawn (releasing a waiting worker) and, if at least 0.5 s since the last status, show the status text.

`show_loss()`: put the logger's iterations, times and errors on the loss plot and refresh the trace if there is data.

#### 19.12.3 Background work (`run_in_background(work, on_tick=None)`)

Runs `work` on a daemon thread named `spline-fitter-work` while a local Qt event loop keeps the GUI responsive. A 50 ms timer (`TICK_MS`) calls `on_tick` (exceptions printed, not raised) and quits the loop when the thread has ended; `request_tick()` (callable from any thread) makes every running background job tick at once via a queued signal. During the work the interpreter thread switch interval is 0.0005 s (restored after). After the thread joins, `on_tick` runs once more. An exception from `work` is re-raised on the caller; otherwise its value is returned. Jobs may nest (a report measured during a fit).

#### 19.12.4 From optimizer result to fit (`make_fit_result`, worker thread)

`result.x is None` -> no fit. Else `FitResult(section_control_points, degree, final_error = fun, num_iterations = nit, num_evaluations = get_evaluation_count(), elapsed_seconds, converged = success and not cancelled, cancelled, section_knots, rms_error)`, then `report = fit_report(airfoil.sections, fit, precision=airfoil.precision, tolerance=opt.tolerance)`; a report exception prints `Fit report failed: {e}` and leaves `report = None`.

### 19.13 Parameter sweeps

A sweep runs when Fit is pressed with at least one sweep widget sweeping. Axes, in sidebar order (degree, control points, sections, smoothing):

| Widget | Axis name | Values | Applied as |
|---|---|---|---|
| degree | `Degree` | integers from min to max of (value, end) | `degree` |
| control points | `Control points` | same | `num_control_points` |
| sections | `Sections` | same | `num_sections`; Per junction entries are dropped when their count does not fit |
| smoothing | `Smoothing weight` | `sweep_values()` (19.21 item 4) | `smoothing_weight` |

1. Combinations: Cartesian product in axis order; a combination whose configuration has fewer control points than its `minimum_control_points` is left out and counted.
2. Reset cancel, results; the comparison x axis is the axis with the most values (first on ties); show its name on the comparison plot (x label, y `Error`, log y, linear x) without redrawing; clear the plot items, the comparison trace and its marker.
3. `left_out` = `" | Left out {k} combinations with too few control points for their degree and junctions"` when `k > 0`. Status `"Sweep: fitting {count} splines ({names joined by ', '}){left_out}..."`.
4. `set_fitting_ui(True)`; for each combination `i` (1-based of `T`):
   1. cancel requested: mark cancelled, stop;
   2. `label` = `"{name} {value}"` pairs joined by `", "` (floats as `number(v, '.5g')`);
   3. airfoil: the current one if its section count matches, else re-split a copy;
   4. clear plot items, `reset_iteration_tracking()`, reset the loss plot, new logger, `LiveProgress` with prefix `"Sweep {i}/{T} | {label} | "`;
   5. run the fit in the background (as 19.12.1 step 6) and show the loss;
   6. a fit exists: `add_optimization_run(..., replot=False)`; if not cancelled, append `(values, fit)` to the sweep results; any exception: traceback printed, counted as failed;
   7. a cancelled fit marks the sweep cancelled; a finished fit becomes the current fit, is plotted with its comb, and the sweep comparison is redrawn (exceptions printed only);
   8. status `"Sweep: {i}/{T} ({number(100i/T, '.0f')}%) | {label}"` + `" | Failed: {e}"` or `" | Cancelled"` or `", error {format_error(comparison_error)}"`, then `" | {format_duration(elapsed)}"`; process events.
5. `set_fitting_ui(False)`.
6. With finished results: redraw the sweep comparison (marking the best: least `comparison_error`, first on ties); if not cancelled, apply the best values (19.13.1); the best fit becomes the current fit (plotted, comb refreshed, saving enabled); status `"Sweep {cancelled|complete}: {count} fits in {duration} | Best: {describe(best)}, error {format_error(best comparison error)}"`. Without results: redraw the previous fit; status `Sweep cancelled: no fit finished` or `Sweep failed: no fit succeeded`. Append `left_out`, then `" | {k} failed"` when any failed.
7. Error chime if any failed; else completion chime if not cancelled and results exist.

**Sweep comparison**: `sweep_series(config, opt, swept names, x axis)` over the project's runs: a run belongs when its optimizer equals, its junctions equal (with `Sections` swept: Continuity and Type equal; otherwise `junction_specs()` equal), its tolerance presence equals, knot placement and parameterization equal, and every non-swept setting equals the sweep's base configuration, compared as configured: `num_control_points` (the configured count, also for tolerance runs), `degree`, `num_sections`, `smoothing_weight`. Points keyed by the run's swept values (later runs replace earlier ones with the same values) with `(axis value of the run, comparison_error)`; the points are sorted as `(x, error)` tuples (by x, then by error). Axis value (used for the keys and the plotted x value, not for the membership test above): smoothing -> weight; degree; sections; control points -> the fitted chain's count for tolerance runs, else the configured count. Plotted connected iff the x values are unique; the marker at the best result's x value of this sweep.

#### 19.13.1 Applying the best values

Turn off every swept widget's sweep (sections first, then degree, then the others), then set their values in the same order. One undo step returns to the sweep settings.

### 19.14 Plots (`PlotManager`)

#### 19.14.1 Plot toolbar (above the main plot)

A reflowing row (spacing 6) of two groups: (a) `Knots` toggle (tool button, checkable; tooltip `Mark the curve points at the knots of every section.`), the `Curvature comb` split toggle with a drop-down arrow button, the `Deviation` split toggle with its arrow, stretch; (b) `Live preview` label and combo (19.12.2), stretch.

Curvature comb toggle tooltip: `Display curvature as perpendicular lines from the curve,` / `blue and pink on alternate spline sections.` / `Use the arrow for quantity, height and density.` Its drop-down (arrow tooltip `Curvature comb settings`) holds a form (12 px padding): `Quantity` combo `Curvature (κ)` (0), `Gradient (dκ/ds)` (1), `Acceleration (d²κ/ds²)` (2), `Jerk (d³κ/ds³)` (3); `Height` spin 0..1000, 4 decimals, step 0.01, default 0.05, tooltip `Length of a tooth, in chords, where the quantity is 1 on the airfoil` / `scaled to a unit chord. The comb looks the same at any size of the data.`; `Density` spin 5..2000, default 250.

Deviation toggle tooltip: `Display the distance of every data point from the curve, magnified` / `along the normal, outward for a point outside, in the color of its` / `nearest spline section (as the curvature comb). The dashed lines` / `are the tolerance. Use the arrow for the height.` Drop-down (arrow tooltip `Deviation settings`): `Height` spin 0.001..1.0, 3 decimals, step 0.01, default 0.05, tooltip `How far from the curve the tolerance is drawn, as a share of the chord` / `(without a tolerance, the largest deviation).`

Drop-downs are `DropDownPanel`s (19.21 item 7).

Changes: Knots toggled -> show/hide knot markers of the current fit. Comb toggled off -> comb hidden and its settings cleared; on, or any comb setting changed while on -> comb redrawn from `(height, density, quantity)` of the current fit. Deviation toggled or its height changed -> deviation height = the value when checked, else none; redrawn for the current fit.

#### 19.14.2 Main plot

Aspect ratio locked 1:1; axis labels `x` (bottom) and `y` (left); the view box is a `SmoothViewBox` (19.20). Items (each holds all sections, separated by NaN gaps):

| Item | Style | Z |
|---|---|---|
| data points | no line; circles, size 4, brush `Ink.DATA` alpha 220, no pen | 5 |
| fitted curve | pen `Ink.CURVE`, width 2 | 10 |
| control polygon | dashed pen `Ink.CONSTRUCTION` alpha 170; squares size 6, brush `Palette.CANVAS`, pen `Ink.CONSTRUCTION` | 0 |
| knots | no line; diamonds size 8, brush `Ink.KNOTS`, pen `Palette.CANVAS` | 30 |
| comb teeth (parity 0, 1) | pen `Ink.COMB_EVEN`/`COMB_ODD`, alpha 90 | -10 |
| comb envelope (parity 0, 1) | pen same colour, width 1.5 | -10 |
| deviation whiskers (parity 0, 1) | pen `Ink.DEVIATION_EVEN`/`ODD`, alpha 90 | -10 |
| deviation envelope (parity 0, 1) | pen same colour, width 1.5; circles size 4, brush same colour, no pen | -10 |
| tolerance band | dashed pen `Ink.CONSTRUCTION`, alpha 110 | 0 |

Pens are cosmetic (width in pixels). Lines connect only finite consecutive values. Items hold exactly the values computed here; pyqtgraph's per-item `dynamicRangeLimit` (default 1e6) may clip far off-screen y values when drawing, which is the library's rendering, not the data.

- `plot_airfoil(coords)`: clear curve, polygon, comb, knots, deviation (and the live foot points); clear the loss data and trace; remember `coords`; set the data points; auto-range.
- `draw_spline(curves)`: curve = per section `curve.sample(1000)`; polygon = per section its control points.
- `plot_fitted_spline(fit)`: draw the fit's curves (built once per fit and cached); on an exception clear curve and polygon; refresh knots; refresh deviation; settle the range.
- Knots: for each section, the curve points at the distinct values of its knot vector (exceptions ignored).
- `clear_plot_items()`: curve, polygon, comb, knots, deviation (not the data points).

#### 19.14.3 Curvature comb

`COMB_FINE_SAMPLES = 4000`, `COMB_TURNING_SHARE = 0.5`. For `density = max(5, density)`, `chord` = the x range of all curves evaluated at `linspace(0, 1, 200)` each (their domain is `[0, 1]`; 1 if the range is 0); for each section `i` (parity `i mod 2`; parity 0 = first, third, ... in `COMB_EVEN` blue, parity 1 in `COMB_ODD` pink):

1. `comb_parameters(curve, n)`: fine grid `linspace(0, 1, max(4000, 20n))` (the curve's whole domain, `[0, 1]` for the application's clamped knots); speed `|C'|`; `κ` (2.4 application guard); `s` = cumulative trapezoid of speed; `turn` = cumulative trapezoid of `|κ|·speed`, where NaN values of `κ` are replaced by 0 before the product is formed and integrated. If `s_last <= 0`: `linspace(0, 1, n)`. Else `measure = s/s_last`, and if `turn_last > 0`, `measure = 0.5·measure + 0.5·turn/turn_last`. Teeth at `interp(linspace(0, 1, n), measure, grid)`.
2. `comb_values(curve, t, q)`: `q <= 1`: `κ` or `dκ/ds` at `t` (comb guard of 2.4). `q >= 2`: on the fine grid of `len(t)`, `dκ/ds` and arc length `s`; keep the first grid point and every point whose `s` is strictly greater than that of the grid point immediately before it (compared with its neighbour on the full grid, not with the last kept point); apply `q - 1` times NumPy `gradient(values, s, edge_order=2)`; interpolate at `t`.
3. Tooth: base `C(t)`, unit normal `(y', -x')/|C'|` (zero where `|C'| <= 1e-12`), tip `base + value·height·chord^(q+2)·normal`.
4. Teeth item: per section, the x (and y) values are, for every tooth in order, `base, tip, NaN` (three entries per tooth, so a section's list ends with a NaN); the sections of one parity are concatenated with one further NaN between consecutive sections. Envelope item: per section the tips as a polyline (no NaN), sections of one parity joined with one NaN between them.

Any exception clears the comb.

#### 19.14.4 Live drawing (`update_live_plot(x, config, preview=None, tolerance=None)`)

Curves from `preview = (control points per section, knots per section or None, degree)` when given, else from `x` decoded as `S` sections of `n` points (`x[2k], x[2k+1]`) on open uniform knots. Draw the spline; the comb if enabled; knots if enabled; the deviation (live mode) if enabled, from a temporary fit with those control points and knots. All exceptions are swallowed.

#### 19.14.5 Deviation

Hidden when the height is none or fewer than 2 data points. With the data `coords` and a fit:

1. Foot points (12.5): live mode refines the previous live step's parameters (`start`) and stores the new ones; finished fits are measured once and cached per (fit, coords).
2. `reference = tolerance or max|signed distance|`; `chord` = x range of `coords`; if either `<= 0`, hide.
3. `scale = height·chord/reference`; tips `foot + scale·distance·normal`.
4. Split the points into runs of equal nearest section; each run after the first begins at the last point of the previous run; a run goes to the parity of its section. Whiskers foot→tip, laid out exactly like the comb teeth of 19.14.3 step 4 (`foot, tip, NaN` per point, one further NaN between runs); envelope through the tips, like the comb envelope (runs joined with one NaN).
5. Tolerance band (only with a tolerance): for each section, 400 parameters over its domain; offsets `±(scale·tolerance)` along the right normal `(y', -x')/|C'|` (denominator floored at 1e-300); the lines in the order section 1 `+`, section 1 `-`, section 2 `+`, ..., joined with one NaN between consecutive lines.
6. Any exception hides the deviation.

For a finished fit, the tolerance is `report["tolerance"]` (none without a report); live, the running fit's tolerance.

#### 19.14.6 Loss plot

Toolbar: caption `CONVERGENCE`, stretch, `X axis` combo `Iterations` ("iters", default), `Time` ("time"); `Log scale` combo `None`, `X only`, `Y only` (default), `Both`. Axis labels: bottom the X axis text at start, then `Iterations` or `Time (s)`; left `Error`. One trace: no line, circles size 4 in `Ink.TRACE`.

Data: iteration numbers, times, errors floored at `1e-12`. `set_loss_curve_data(mode)`: x = iterations or times; empty -> empty trace; else `n = min(lengths)`, indices `unique(round(linspace(0, n-1, min(n, 100000))))` (`MAX_PLOT_POINTS = 100000`). Changing the X axis relabels, auto-ranges and redraws; changing the scale sets log modes and auto-ranges. `reset_loss_plot` clears the plot and data and recreates the trace.

#### 19.14.7 Comparison plot

Toolbar: caption `RUNS`, stretch, `X axis` combo listing every explorer column name (default `Control points`; tooltip `The explorer column the runs are plotted along.` / `Only the runs that pass the explorer's filter are shown.`), `Y axis` combo listing the metric columns (default `Error`; tooltip `The fit metric plotted for each run`). Both combos are compact (minimum 4 characters) and their lists as wide as the longest name plus 24 px. Initial axes: bottom `Control points`, left `Error`, log y.

Items: a trace (line `Ink.TRACE` width 1.5 when connected, else no line; circles size 6 in `Ink.TRACE`) and a marker (circles size 13, brush `Ink.MARKER` (orange), pen `Palette.CANVAS` width 2), the marker excluded from auto-range while the trace reserves room for it.

`refresh_comparison_plot()` (on explorer filter changes, combo changes, run list changes, after adding runs with `replot`):

1. Rows = the explorer's visible rows in table order; points `(x, y, run)` where `x` is not `None` and `y` is a number (bools are not numbers).
2. If any `x` is not a number: labels = the distinct `x` texts sorted case-insensitively; `x` becomes the label's index; tick labels at those positions.
3. `log_x` = the column is a log column, no tick labels, and all `x > 0`; `log_y` = the column is a log column and all `y > 0`.
4. Sort points by `(x, y)`; set axes (labels = column names, log modes, ticks), the trace (connected iff `x` values are unique), auto-range; remember the points; mark the selected runs (marker at their points).

Refilling the explorer forgets the remembered points without redrawing. Marking the selection (on every selection change) redraws the whole plot first when no points are remembered; otherwise it only moves the marker. A sweep draws its own series (19.13) and leaves no points remembered, so the next selection change, filter change, axis change or `replot` run addition replaces the sweep series with the explorer's runs.

### 19.15 File actions

#### 19.15.1 Save fitted airfoil

No fit or no airfoil: status `No fitted spline to save.`. Integer dialog title `Save fitted airfoil`, label `Points per side, spaced closer toward the leading and trailing edge`, default `max(20, (N + 1)//2)`, range 3..100000; cancel -> status `Save cancelled.`. Points = `resample(curves, n, "cosine")` (13.4). Save dialog title `Save fitted airfoil`, default `<home>/{file_stem(name or filename)}_fitted.dat`, filter `Airfoil data (*.dat)`; cancel -> `Save cancelled.`. Write with `export_airfoil_data(path, points, name=name or filename, precision=max(6, precision))`. Status `Saved: {path}`; any exception: `Save failed: {e}`, error chime.

#### 19.15.2 Save spline

No fit: `No fitted spline to save.`. Default `<home>/{file_stem(filename or name)}.deg{p}.spline.dat`, dialog title `Save spline control points`, filter `Spline data (*.spline.dat)`; cancel -> `Save cancelled.`. Continuity = `junction_specs()` of the run whose result is the shown fit, else of the sidebar's spline configuration (a `ValueError` -> no continuity). `export_spline_data(path, p, control points, name=name or (filename or name), precision=12, section_knots=[knots(i) as lists], continuity)`. `Saved: {path}` / `Save failed: {e}` + error chime.

#### 19.15.3 Export CAD

No fit: `No fitted spline to export.`. Dialog `Export CAD` (form):

- `Format` combo: `STEP AP214 (*.step)` (`.step`), `IGES 5.3 (*.igs)` (`.igs`), `DXF R2000 (*.dxf)` (`.dxf`); initial index QSettings `cad_format` (default 0).
- `Units` segmented `mm`, `m`, `inch`; initial `cad_units` (default `mm`; unknown -> first).
- `Scale` scientific spin `1e-6 .. +∞`, initial `cad_scale` (default 1.0); tooltip `Coordinates are multiplied by this factor, for example the chord length.`
- `Sections` check box `Merge sections into one curve`, initial `cad_merge` (false); tooltip `Write the whole chain as a single B-spline (junctions become knots).` / `Otherwise every section is its own curve.`
- `Trailing edge` check box `Close a blunt trailing edge with a line`, initial `cad_close_te` (true).
- A button box (19.21 item 13) with `Export...` (accept role, default) and `Cancel` (reject role). Cancel -> status `Export cancelled.`.

On accept: store the five values; save dialog `Export CAD` with default `<home>/{file_stem(filename or name or "airfoil")}{suffix}` and the format's text as filter; cancel -> `Export cancelled.`. Export: `curves = scipy_curves(fit)`; merged into one (14.7) when asked; when closing is asked and `trailing_edge_segment(curves)` is not `None`, append it; `write(path, curves, name=name or filename or "airfoil", units, scale)`. Status `Exported: {path}`; exception: `Export failed: {e}`, error chime.

#### 19.15.4 Fit report

No fit: `No fitted spline.`. If the fit has no report: status `Measuring the fit...`, compute `fit_report(sections, fit, precision)` in the background and store it on the fit; else, if the report lacks `max_thickness`, add the shape keys (12.3). Title `name or filename or "Fit report"`. Shape = `airfoil_shape(fit)`; chart = the shape plot when available. Show a modal document (19.21 item 10) titled `Fit report`, 560 × 520 px (720 px high with the chart). Exception: `Report failed: {e}`, error chime.

Shape plot: axis labels `Chord (%)` and `Percent of the chord`; minimum height 200 px; legend at offset (-12, 12); `Thickness` line (`Ink.TRACE`, width 1.5, solid) and `Camber` line (`Ink.CONSTRUCTION`, width 1.5, dashed) over `100·x`, `100·value`; a dot (size 7, brush the line colour, pen `Palette.CANVAS` width 2) at each maximum (`100·max_x`, `100·max`); restyled on theme changes.

### 19.16 Projects in the window (`ProjectManager`)

State: project, path, `modified`, selected runs, a stack of deletions, `last_number` (highest number given), comparison points.

#### 19.16.1 Reset and unsaved changes

`reset()` (on a new airfoil): clear project, path, modified, deletions and `last_number`; label and title (19.16.5); disable Save project actions; empty the explorer; refresh the comparison plot; update run buttons.

`confirm_discard()`: true without a project or without unsaved changes. Else a message box: title `Unsaved runs`, text `Save the changes to {basename of path, or "the project"}?`, informative `Runs that are not saved are lost.`, buttons `Save` (default), `Don't save` (Discard), `Cancel` (escape). Save -> save (19.16.3), true iff no longer modified; Don't save -> true; Cancel -> false.

#### 19.16.2 Opening a project

Dialog title `Open project`, home directory, filter `Spline project (*.asfproj);;All files (*)`. `open_project(path)`:

1. Fitting: `Cancel the running fit to open a file`; false.
2. `load_project` and a pre-check (computing every explorer cell of every run, splitting the airfoil at its own and the last run's section counts, building the last run's curves); exception: `Failed to load project: {e}`, error chime; false. A broken file closes nothing.
3. `confirm_discard()` false: status `Open cancelled.`; false.
4. Capture everything opening changes (project state, airfoil, fit, sections maximum, all settings values, history checkpoint, file card texts, selection), then show the project:
   1. project, path, not modified, no deletions, `last_number` = highest run number;
   2. the airfoil becomes the project's; no fit; saving disabled; sections maximum `max(1, N - 1)`; file card with the path; Sections value = the airfoil's section count; label; Save project actions enabled; clear plot items; plot the airfoil;
   3. with runs: apply the last run's settings (19.10; each change of the Sections value re-splits the airfoil as in 19.11.6), load it into the plots (19.16.7), status `"Loaded project with {k} run{s}, showing {optimizer display name}"`; without runs: `"Loaded project: {basename}"`;
   4. fill the explorer, refresh the comparison plot, update run buttons.
5. An exception while showing: restore the capture (settings quietly and in order, then `on_optimizer_changed`, history rolled back so the failed open leaves no undo step, airfoil, file card, explorer and selection, plots and fit), status `Failed to load project: {e}`, error chime; false.
6. Success: add to recent files; true.

#### 19.16.3 Saving

`Save project`: without a project or path, does `Save project as`. Else `save_project`; success: not modified, label, `Saved: {path}`; failure: `Save failed: {e}`, error chime.

`Save project as`: no airfoil: `No airfoil loaded to save.`. Create a project when there is none. Dialog title `Save project as`, default `<home>/{file_stem(name or filename or "project")}.asfproj`, filter `Spline project (*.asfproj)`; cancel -> `Save cancelled.`. If the chosen path does not end with `.asfproj` (case-insensitive), append it; if that file exists, ask (title `Save project as`, text `{basename} already exists. Replace it?`, Yes/No, default No); No -> `Save cancelled.`. Save; success: path set, not modified, label, Save actions enabled, recent files, `Saved: {path}`; failure: `Save failed: {e}`, error chime.

#### 19.16.4 Adding runs (`add_optimization_run(config, opt, fit, loss_history, replot=True)`, returns whether added)

1. No airfoil: false.
2. `fit.cancelled` and a project run exists that is not cancelled and matches: false.
3. No project: create one from the airfoil; enable Save project actions.
4. `run = create_optimization_run(...)`.
5. The first matching run is replaced in place and `run.number` = its number; otherwise append and `last_number = number_runs(runs, last_number)`.
6. Modified; label; fill the explorer; refresh the comparison plot if `replot`. Return true.

Match: equal `degree`, `num_control_points`, `num_sections`, `continuity`, `geometric_continuity`, `junction_continuity`, `tolerance`, `max_points_per_section`, `knot_placement`, `parameterization`, `optimize_le_position`, `grow_control_points`, `optimizer`, and, only when the new run has no tolerance, `smoothing_weight`.

#### 19.16.5 Label and title

No project: label `No project`; title `Airfoil Spline Fitter`; window not modified. Else `name` = basename of the path or `Unsaved project`; label `"{name}{'*' if modified}\n{k} optimization run{s}"`; title `"{name}[*] - Airfoil Spline Fitter"` with Qt's window-modified flag = modified (Qt shows `*` in place of `[*]` while modified, else nothing).

#### 19.16.6 Selection, details, deletion

- Filling the explorer keeps selected runs that still exist; if the selection changed, re-read it.
- Selection change: buttons; details: none -> message `Select a run to view details`; several -> `"{k} runs selected"`; one -> title = optimizer display name over pairs `Degree`, `Control points` (fitted chain count for tolerance runs, else configured), `Continuity` (`", ".join(specs)` or the continuity name), `Tolerance` (`number(tol, '.3e')`, tolerance runs only), `Error` (`number(final_error, '.3e')`), with a report `Max deviation` and `RMS deviation` (`.3e`), `Iterations`, `Evaluations`, `Time` (`format_duration`), `Converged` (`yes`/`no`). With one run selected and no fit running, that run is loaded into the plots (19.16.7). Selected runs are marked on the comparison plot.
- Double-click a run, or `Load settings` with one selected: while fitting `Cancel the running fit to load a run`; else apply its settings (19.10), load it into the plots, status `"Applied the settings of run {number} ({optimizer}, degree {p}, {n} control points, {S} sections)"`.
- Delete (button or the Delete key while the table has focus; not while fitting): a message box titled `Delete runs`, text `"Delete {k} run{s} from the project?"`, informative `Edit > Restore deleted runs brings them back.`, buttons `Delete` (destructive, default) and `Cancel`. Confirmed: remove the selected runs (by identity), push `[(index, run)]` on the deletion stack, modified; if the shown fit belonged to a deleted run, clear it (fit none, plot items and loss trace cleared, saving disabled); fill the explorer, label, re-read the selection; status `"Deleted {k} run{s}; Edit > Restore deleted runs brings them back"`.
- Edit > Restore deleted runs (enabled while the stack is non-empty and no fit runs): pop the last deletion, insert each run at `min(index, len)` in the stored order, modified, fill the explorer, label, select the restored runs, status `"Restored {k} run{s}"`.

`{s}` is `s` unless the count is 1.

#### 19.16.7 Loading a run into the plots

The run's result becomes the current fit; the loss data become its history (iteration numbers, times, errors); plot the fit; refresh the loss trace and the comb; enable saving.

### 19.17 Project explorer table (`RunTable`)

#### 19.17.1 Columns

| # | Name | Value | Format | Log | Metric | Blank unless | Tooltip |
|---|---|---|---|---|---|---|---|
| 1 | `Run` | run number | | | | | `Order in which the runs were done; a number stays when other runs are deleted` |
| 2 | `Optimizer` | display name | | | | | |
| 3 | `Goal` | `Tolerance` / `Fixed count` | | | | VarPro | |
| 4 | `Tolerance` | tolerance | `.3g` | yes | | VarPro with tolerance | `Max deviation the tolerance goal had to meet` |
| 5 | `Knots` | placement capitalized (`Uniform`, `Free`) | | | | VarPro | |
| 6 | `Parameterization` | the stored value with `_` replaced by a space, then Python `str.capitalize()` (`Centripetal`, `Arc length`) | | | | Direct LSQ | |
| 7 | `Grow points` | bool | | | | VarPro | |
| 8 | `Degree` | degree | | | | | |
| 9 | `Control points` | configured count | | | | | `Control points per section as set (the start count of a tolerance goal)` |
| 10 | `Sections` | sections | | | | | |
| 11 | `Continuity` | `G0`..`G3` (enum name) | | | | sections > 1 | |
| 12 | `Type` | `Geometric` / `Parametric` | | | | VarPro and sections > 1 | |
| 13 | `Per junction` | entries joined by `", "` | | | | VarPro | |
| 14 | `Subsample` | max points or `None` | | | | VarPro | `Points per section of the subsample; blank when every point is used` |
| 15 | `Smoothing` | smoothing weight | `.3g` | yes | | VarPro without tolerance | |
| 16 | `Optimize junctions` | bool | | | | sections > 1 | |
| 17 | `Error` | `comparison_error` | `.3e` | yes | yes | | `RMS distance of the points to the curve, the same measure for every run` |
| 18 | `Max deviation` | report value | `.3e` | yes | yes | | |
| 19 | `RMS deviation` | report value | `.3e` | yes | yes | | |
| 20 | `Fitted control points` | `result.num_control_points` | | | yes | | `Distinct control points of the fitted chain, junction points counted once` |
| 21 | `Iterations` | `num_iterations` | | | yes | | |
| 22 | `Evaluations` | `num_evaluations` | | | yes | | |
| 23 | `Time` | elapsed seconds | duration | yes | yes | | |
| 24 | `Converged` | bool | | | | | |
| 25 | `Date` | `timestamp[:16]` with `T` replaced by a space | | | | | |

Cell text: `None` -> empty; bool -> `yes`/`no`; duration -> `format_duration`; float with a format -> `number(v, spec)`; else `str(v)`. Numbers (int or float, not bool) use the monospace font and align right; other cells align left (all centred when pivoted). Rows have no grid lines and no word wrap; columns resize to contents when runs are set.

#### 19.17.2 Filter syntax

`parse_filter(text)`: strip; remove whitespace after any operator (`<=`, `>=`, `!=`, `<`, `>`, `=`); split on whitespace. A word matching `^(<=|>=|!=|<|>|=)(.+)$` whose operand parses as a float (the display decimal point replaced by `.`, then Python `float`, so `inf` and `nan` count as numbers) is a numeric comparison; a word with `=`/`!=` and a non-numeric operand is a whole-cell match (`=text` / `!=text`, case-folded); any other word is a case-folded substring.

`term_matches(term, value, text)`: substring -> in `text.casefold()`; whole-cell -> `(text.casefold() == operand) == (op is "=")`; numeric -> `value` is a number and `value op operand`.

Row filter: with no global terms and no column exclusions every row passes. Else a row matches when every global term matches (a numeric term against the `Error` column only; other terms, substring and whole-cell (`=`/`!=`) alike, match when they match in at least one column, so `!=x` passes a row as soon as one cell differs from `x`), and for every filtered column its cell text is not excluded. `Invert` shows exactly the non-matching rows (when any filter is set).

#### 19.17.3 Table widget

Above the table: a line edit (placeholder `Filter all columns`, clear button; tooltip `Show the runs that match in any column, and pass the filter of every` / `column (the drop-down at the right of a column header).` / `Every word must match.` / `Text matches a part of a cell, for example varpro or tolerance.` / `A number with a comparison compares the error:` / `>1e-5 or <=3e-4. =free matches a whole cell.`), checkable `Invert` (tooltip `Show the runs the filters do not match`), checkable `Pivot` (tooltip `Swap rows and columns: one column per run, one row per setting`).

Table: extended selection of whole rows (whole columns when pivoted), not editable, horizontal per-pixel scrolling. Header sections are movable and clickable; clicking a section sorts by it (toggling order); the run header is hidden. Initial sort: `Run` ascending. Sorting: `None` cells last in both orders; numbers by value; others by case-folded text. Double-click emits the run.

Header drawing (`FilterHeader`): each section shows, right-aligned (glyph size 10, 6 px from the right edge), a `chevron_down` glyph (`TEXT_3`), or a `filter` funnel (`TEXT`) when the column is filtered; the sorted section also shows `sort_up`/`sort_down` to the left of it (10 + 4 px further left). A left click within the glyph's horizontal extent (widened by 4 px left and 2 px right, full height) opens the column filter instead of sorting. Selecting runs does not highlight header sections.

Header context menu: `Plot as x axis`; `Plot as y axis` (metric columns only); separator; `Filter this column...`; `Clear all filters` (empties global text and all column filters). Plot items set the comparison combo.

Pivot (`set_pivoted`): keeps the selected runs selected, the visual column order (applied to the row names), filters and sorting; resizes columns to contents; scrolls to the first selected.

#### 19.17.4 Column filter drop-down

Popup (closes on outside click, deleted on close), 12 px padding, width `max(260, natural)`, placed under the header section's left edge, or right-aligned to the section's right edge when it would pass the window's right edge, kept on screen (pivoted: right of the row name). Contents:

1. caption = column name upper-cased;
2. buttons `Sort ascending`, `Sort descending` (sort and close);
3. `Condition` line edit (placeholder `Condition`, clear button; tooltip `Unchecks the values it does not match. Enter keeps the result,` / `Shift+Enter checks the values it matches instead.` / `Every word must match.` / `Text matches a part of a cell, for example varpro or tolerance.` / `A number with a comparison compares the value:` / `>1e-5, <=12, =5 or !=3. =free matches the whole cell.`), focused on open;
4. buttons `Keep matching` (tooltip `Uncheck the values the condition does not match (Enter)`) and `Add matching` (`Check the values the condition matches (Shift+Enter)`), enabled while the condition has terms;
5. `Select all` tri-state check box;
6. a checkable list of the column's distinct values (empty shown as `Not used`), at most 10 rows high; values = cells of rows passing all other filters plus currently excluded ones, ordered: blanks last, numbers before text, numbers by value, text case-folded;
7. `Clear filter` (remove this column's filter and close).

Behaviour: `excluded` starts as the column's current exclusions. While a condition is typed, the values it rules out show unchecked and greyed (`TEXT_3`) and the table filters live with `excluded ∪ ruled out`. `keep(add=False)`: with terms, `excluded ∪= ruled out` (or with `add`, `excluded -= values it matches`); clear the condition; apply. Enter in the condition = keep (Shift+Enter = add); the clear button keeps what the condition showed, then empties the field; clicking a value first keeps, then includes the value if it is now checked or excludes it if unchecked; clicking Select all first keeps, then includes every value if any value is excluded, else excludes every value; closing the drop-down keeps; Enter pressed while the focus is not in the condition closes it. `Select all` shows checked when all values are checked, partially checked when some are, unchecked when none are. A column filter with no exclusions is removed. A filtered column shows the funnel.

### 19.18 Batch conversion (dialog and run)

Dialog `Batch convert` (minimum width 480, form): a muted note `Fit every .dat file in the source folder with the current settings` / `and save each as .spline.dat in the target folder.`; rows `Source` and `Target`: a muted, wrapping path label (the folder chosen earlier in this session, else `Not chosen`) and `Choose...` (directory dialog titled `Source folder` / `Target folder`, starting at home). A button box (19.21 item 13) with `Convert` (accept role, default) and `Close` (reject role). Convert closes the dialog and runs:

1. Missing or non-existent source: `Choose a source folder first.`; target: `Choose a target folder first.`.
2. Files: names in the source folder (not recursive), sorted by name (Python string order), ending in `.dat` (case-insensitive) and not containing `.spline.` (case-insensitive). None: `No .dat files in the source folder.`.
3. Configurations from the sidebar (a `ValueError`: its message).
4. `cancel_requested = False`; `set_fitting_ui(True)`; for each file `idx` of `T` (stop when cancel is requested):
   1. parse (`parse_airfoil_data`); no coordinates -> error `Failed to parse`; airfoil with the configuration's section count (`filename = filename_name or file name`, `name = content_name or filename_name or ""`, precision int or 6);
   2. output path `target/output_name(filename or name, config, opt, Auto checked)`; exists -> skipped, status `"[{idx}/{T}] Skip: {path}"`, next file;
   3. tolerance goal with Auto: the file's tolerance = `default_tolerance(coords, precision)`;
   4. `reset_iteration_tracking()`; status `"[{idx}/{T}] Fitting {file}..."`; plot the file's points; reset the loss plot; live preview with prefix `"[{idx}/{T}] {file} | "`;
   5. run in the background; show the loss; `x is None` -> error `Optimization failed`; cancel requested -> stop; not successful -> error `Optimizer did not converge: {message}`;
   6. draw the result (`update_live_plot` with its control points and knots); write `export_spline_data(path, p, control points, name=name or filename, precision=12, section_knots, continuity=junction_specs())`; done; status `"[{idx}/{T}] Done: {file} | {evaluations} evaluations, error {format_error(fun)} | {duration}"`;
   7. any error: failed, status `"[{idx}/{T}] Failed: {file} - {e}"`, error chime.
5. Final status: cancelled `"Batch cancelled after {duration}. Completed {done}/{T}, failed {failed}."`; else `"Batch complete: {done} files in {duration} ({failed} failed)"`; plus `" | Skipped {k} already in the target folder with these settings"` when `k > 0`.
6. Always: `set_fitting_ui(False)`; plot the window's own airfoil (or nothing) and its fit with its comb.

The batch adds no project runs and writes no logs.

### 19.19 Theme

#### 19.19.1 Tokens

Greys are generated in OKLab from a lightness and the temperature `T ∈ [-1, 1]` (setting / 100):

```
grey(L, T):
  hue    = radians(95 if T >= 0 else 255)
  chroma = |T| * 0.035 * L * min(1, (1 - L) * 20)
  a, b   = chroma cos(hue), chroma sin(hue)
  l_ = (L + 0.3963377774 a + 0.2158037573 b)^3
  m_ = (L - 0.1055613458 a - 0.0638541728 b)^3
  s_ = (L - 0.0894841775 a - 1.2914855480 b)^3
  R =  4.0767416621 l_ - 3.3077115913 m_ + 0.2309699292 s_
  G = -1.2684380046 l_ + 2.6097574011 m_ - 0.3413193965 s_
  B = -0.0041960863 l_ - 0.7034186147 m_ + 1.7076147010 s_
  each channel: clamp to [0,1]; sRGB encode (12.92c if c <= 0.0031308 else 1.055 c^(1/2.4) - 0.055);
  round(c * 255); format "#RRGGBB" upper-case hex
```

Lightness (dark, light) and the neutral (`T = 0`) results:

| Token | Dark L | Light L | Dark | Light |
|---|---|---|---|---|
| `CANVAS` | 0.213 | 0.995 | `#191919` | `#FDFDFD` |
| `WINDOW` | 0.236 | 0.970 | `#1E1E1E` | `#F5F5F5` |
| `PANEL` | 0.268 | 0.950 | `#262626` | `#EEEEEE` |
| `FIELD` | 0.309 | 1.000 | `#303030` | `#FFFFFF` |
| `FIELD_HOVER` | 0.344 | 0.965 | `#393939` | `#F3F3F3` |
| `RULE` | 0.360 | 0.870 | `#3D3D3D` | `#D4D4D4` |
| `RULE_STRONG` | 0.434 | 0.780 | `#515151` | `#B7B7B7` |
| `TEXT` | 0.982 | 0.250 | `#F9F9F9` | `#222222` |
| `TEXT_2` | 0.807 | 0.430 | `#C0C0C0` | `#505050` |
| `TEXT_3` | 0.686 | 0.580 | `#9A9A9A` | `#7A7A7A` |
| `SELECT` | 0.405 | 0.895 | `#494949` | `#DCDCDC` |
| `SELECT_LINE` | 0.609 | 0.600 | `#838383` | `#808080` |
| `CONSTRUCTION` (plot) | 0.639 | 0.620 | `#8C8C8C` | `#868686` |

Checks at `T = +1`: dark `CANVAS #1A1915`, light `WINDOW #F9F5E6`; at `T = -1`: dark `CANVAS #17191C`, light `PANEL #E0F0FF`.

Mode colours (dark, light): `OK #4FBF85/#1A7A4A`, `WARN #F2B84B/#8F5E00`, `ERROR #F0605D/#C02F2B`, `KNOTS #F2C94C/#C99A06`, `COMB_EVEN #49B6E8/#1E88C7`, `COMB_ODD #E5689A/#CC3F78`.

Fixed ink: `INK #FF6E10`, `INK_HOVER #FF8636`, `INK_PRESSED #E05E08`, `INK_DEEP #AD4200`, `ON_INK #1A0C02`.

Palette (UI) tokens: the twelve greys of the table other than `CONSTRUCTION` (`CANVAS` to `SELECT_LINE`), the four `INK*` colours, `ON_INK`, `OK`, `WARN` and `ERROR`. Plot (`Ink`) tokens: `CONSTRUCTION`, `KNOTS`, `COMB_EVEN`, `COMB_ODD` (RGB of the above), `DATA = TEXT`, `TRACE = GRID = TEXT_2`, `DEVIATION_EVEN/ODD = COMB_EVEN/ODD`, `CURVE = MARKER = INK`, `COMB_TEETH_ALPHA = 90`, `GRID_ALPHA = 0.10`.

`Z`: `COMB -10`, `CONSTRUCTION 0`, `DATA 5`, `CURVE 10`, `KNOTS 30`.

`Size`: `TIGHT 4`, `GAP 6`, `GROUP 8`, `PAD 12`, `PAD_LARGE 16`, `CONTROL_HEIGHT 26`, `GLYPH 10`, `GLYPH_LARGE 16`, `RADIUS 3`, `LABEL_WIDTH 92`, `FIELD_MIN_WIDTH 70`, `SIDEBAR_MIN_WIDTH 300`, `EXPLORER_MIN_WIDTH 180`, `SLIDER_WIDTH 220`, `DIALOG_MIN_WIDTH 480`, `POPUP_MIN_WIDTH 260`, `CURVE_WIDTH 2`, `LINE_WIDTH 1`, `DATA_SIZE 4`, `CONTROL_SIZE 6`, `KNOT_SIZE 8`, `TRACE_POINT 4`, `MARKER_SIZE 13`, `PLOT_MARGIN 24`, `SHAPE_PLOT_HEIGHT 200`.

`Type`: `MONO = "Cascadia Mono, Consolas, Menlo, monospace"`, families `Cascadia Mono`, `Consolas`, `Menlo`, `DejaVu Sans Mono`; `BASE_PT 9`; `CAPTION_PX 10`; `READOUT_PX 15`.

#### 19.19.2 Applying the theme

`apply_theme(app)` (once per application; later calls do nothing):

1. Read QSettings `theme/mode` (`system`, `light`, `dark`; anything else -> `system`), `theme/temperature` (integer, unreadable -> 0, clamped to -100..100), `motion/scroll_ms`, `motion/zoom_ms` (19.20).
2. `platform_dark` = the application's default palette window colour has lightness `< 128`.
3. Style `Fusion`. Locale: the system locale without group separators becomes the default; the display decimal point (18) = its decimal point.
4. Application font point size 9; window icon: one `QIcon` with both `assets/app_icon.ico` and `assets/app_icon.png` added (each only if the file exists). Both files are given byte for byte, with checksums, in Appendix B (`spec-appendix-icons.md`, part of this specification).
5. Restyle whenever the system colour scheme changes; restyle now.

`restyle`: decide dark: mode `light`/`dark` -> that (and tell Qt the colour scheme, so title bars and native dialogs match); `system` -> hand the decision back to Qt (scheme Unknown) and use Qt's reported scheme, or `platform_dark` when Qt reports none. Set the tokens; set the Qt palette (Window `WINDOW`, WindowText `TEXT`, Base `FIELD`, AlternateBase `PANEL`, Text `TEXT`, Button `FIELD`, ButtonText `TEXT`, Highlight `SELECT`, HighlightedText `TEXT`, ToolTipBase `FIELD`, ToolTipText `TEXT`, PlaceholderText `TEXT_3`, Link `TEXT_2`, Mid `RULE`, Midlight `RULE`, Dark `WINDOW`, Light `RULE_STRONG`; disabled WindowText, Text, ButtonText `TEXT_3`); set the stylesheet (Appendix A); pyqtgraph defaults background `CANVAS`, foreground `TEXT_2`, antialiasing on; emit `theme_changed`. Plots and items not covered by the stylesheet restyle on that signal (the window's three plots, every plot item, open reports and charts, readout titles, filter drop-downs).

`set_theme(mode=None, temperature=None)`: store given values in the scheme and QSettings, then restyle (if the theme was applied).

#### 19.19.3 Plot styling

`style_plot(plot)`: plot item margins (4, 24, 24, 4); left and bottom axes are decade axes (below); every axis pen `Ink.GRID` width 1, text `TEXT_2`, tick font 8 pt, no SI prefix scaling, label colour `TEXT_2`; grid on both axes with alpha 0.10; background `CANVAS`. `label_plot(plot, side, text)`: label in `TEXT_2`, 9 pt.

Decade axis: tick texts use the display decimal point; on a log axis, if at least one integer decade is in view, only ticks at integer decades are labelled; cached as a pixmap.

Framed plot: the plot item fills the viewport inside the 1 px frame, inset by 2 px on every side.

Rich text (reports): document margin 16; CSS: `body` colour `TEXT`, line height 1.45; headings `TEXT` weight 600; `td, th` padding `3px 12px 3px 0`; `th` colour `TEXT_3`, weight 600, left, bottom border 1 px `RULE`; `pre, code` monospace on `FIELD`; `.num` monospace; `a` `TEXT_2`; `b` `TEXT`; `.muted` `TEXT_3`; `.ok` `OK`; `.warn` `WARN`; `.error` `ERROR`. On a theme change the HTML is re-parsed keeping the scroll position. Readout titles are `<span style='color: {TEXT}'>{escaped title}</span>`.

#### 19.19.4 Theme dialog (View > Theme...)

Title `Theme`, deleted on close, form with 16 px padding:

- `Mode`: segmented `System`, `Light`, `Dark` (tooltip on System: `Follow the light or dark setting of the system`), initially the current mode; selecting applies at once (`set_theme(mode)`).
- `Temperature`: `Cool` label, slider -100..100 initially at the current temperature (page step 10, minimum width 220, object name `TemperatureSlider`, tooltip `Tint the greys cool (blue) or warm (sand); neutral in the middle`), `Warm` label. While dragging, a value with `0 < |v| <= 4` snaps to 0. Each change (re)starts a 60 ms single-shot timer; on timeout, if the value differs from the current temperature, `set_theme(temperature=v)`. Closing applies a pending value.
- A button box (19.21 item 13) with `Close` (reject role) and `Neutral` (reset role; tooltip `Untinted greys (the default)`; sets the slider to 0).

### 19.20 Motion

`Motion`: `DEFAULT_SCROLL_MS 350`, `DEFAULT_ZOOM_MS 650`, `MAX_MS 1000`; current values from QSettings clamped to `0..1000`. `set_motion(scroll_ms, zoom_ms)`: clamp, store in QSettings (`motion/scroll_ms`, `motion/zoom_ms`), used by the next wheel step.

#### 19.20.1 Smooth scrolling (application event filter)

- Every item view is switched to per-pixel scrolling when polished.
- A wheel event on a widget is taken over only when: no keyboard modifiers; no pixel delta (touchpads keep Qt's handling); the angle delta's larger axis (`|y| >= |x|` = vertical) gives `notches = delta / 120 ≠ 0`; the widget is a scroll bar, or the viewport of a scroll area that is not a graphics view (plots zoom instead); the bar is enabled with a non-empty range. Horizontal bars in right-to-left widgets invert the direction.
- Distance `= -notches · wheelScrollLines · singleStep`. The bar's animation (created on first use, out-cubic easing): `current` = the animated value if running, else the bar value; `start` = the running target, else `current`; `target = clamp(start + distance, min, max)`. If not running and `round(target) == round(current)`, the event is not taken (it passes on to an enclosing scroll area). Else stop, duration = `SCROLL_MS`; duration 0 sets the bar at once; otherwise animate `current → target`, setting `round(value)` on each frame. A slider press, or a bar value change not made by the animation while it runs, stops it.

#### 19.20.2 Smooth zoom (`SmoothViewBox`)

On a wheel event (optionally restricted to one axis, honouring mouse-enabled axes): `log = ln(1.02) · delta · wheelScaleFactor` (pyqtgraph's own factor); if a glide is running, add what remains of it (`total · (1 - done)`); centre = the cursor in view coordinates. Animate progress 0→1 over `ZOOM_MS` with out-cubic easing; each step scales by `exp(total · (progress - done))` about the centre and records the range; if the view range changed by anything else meanwhile (auto-range, panning), the glide stops. Duration 0 applies the whole zoom at once. The event is accepted and reported as a manual range change.

#### 19.20.3 Scrolling and zooming dialog

Title `Scrolling and zooming`, deleted on close, form with 16 px padding: `Scrolling` and `Zooming` spin boxes 0..1000 step 25, initially the current `SCROLL_MS` and `ZOOM_MS`, suffix ` milliseconds`, 0 shown as `Off`; tooltips `How long a wheel step glides in lists, tables, panels and documents; off jumps at once` and `How long a wheel step glides when zooming a plot; off jumps at once`; changes apply at once. A button box (19.21 item 13) with `Close` (reject role) and `Default` (reset role) (tooltip `350 milliseconds scrolling, 650 zooming`; sets both defaults).

### 19.21 Widgets

1. **SidebarSection**: caption (title upper-cased, `SectionCaption`) above content; margins 12, 10, 12, 12; spacing 8 (content spacing 6); bottom rule.
2. **CollapsibleSection**: header is a flat button with text `" " + TITLE` and a `chevron_down` (open) / `chevron_right` (closed) icon, pointing-hand cursor, toggling the content; a badge label (`SectionBadge`) at the right.
3. **SweepableSpinBox(spin, group)**: the spin box (stretch), a 26×26 checkable toggle with the `range` glyph (16 px; checked colour `TEXT`; tooltip `Sweep a range of values: one fit per value, or per combination with the other swept settings`), and a hidden end spin box (same range and step, initial `min(2·value, max)`). Toggling on shows the end box, sets its minimum to the value box's minimum and its value to `min(2·value, value maximum)`; off hides it; both emit `sweep_changed`, as do end value changes. `sweep_state() = (active, end)`; `set_sweep_state` toggles if needed then sets the end. `set_sweep_allowed(False)` turns a sweep off and disables the toggle. `set_minimum(m)` sets both boxes' minimum; `set_sweep_maximum(m)` the end box's maximum. `sweep_range = (value, end)`.
4. **SweepableDoubleSpinBox**: as 3 with an end box and a `Step` label and step box of the same spin type (step box range `singleStep/1000 .. max`, tooltip `Distance between the swept values`). Toggling on: `start = value or singleStep`; end minimum = value minimum; end = `min(10·start, max)`; step = `start`. State `(active, end, step)`. `sweep_values(limit=1000)`: `low, high = sorted(value, end)`; `count = min(floor((high - low)/step · (1 + 1e-9)) + 1, limit)`; values `float(f"{low + i·step:.12g}")`.
5. **SegmentedControl**: joined checkable buttons (`Segment`, expanding, `position` property `first`/`last`/empty), exclusive; combo-box API (`addItem(text, data)`, `count`, `itemData`, `itemText`, `setItemText`, `setItemToolTip`, `findData` (-1 if absent), `currentIndex`, `currentData`, `setCurrentIndex` (ignores out-of-range; emits `currentIndexChanged` only on change)). The first added item becomes current.
6. **ScientificSpinBox** (double spin box with 323 internal decimals): shows `f"{v:.6g}"` with the locale decimal point; parses text made only of digits, `eE+-.` and the locale decimal point, finite; invalid text keeps the old value. Validation: parseable and within range -> Acceptable; parseable outside range -> Intermediate; unparseable but only number characters and not overflowing -> Intermediate; else Invalid. Stepping by `k` steps applies `k` single ladder steps (each from the previous unclamped value) and then clamps the result to `[minimum, maximum]` once; the starting value is the box's value, which Qt always keeps within the range. One ladder step: from `v <= 0` up -> single step, down -> 0; else with `e = floor(log10 v)`, ladder of the nine values `m * 10.0 ** k` (`m` in `1.0, 2.0, 5.0`, `k` in `e-1, e, e+1`, Python floats); up -> smallest `> v(1 + 1e-9)`; down -> largest `< v(1 - 1e-9)` if it is `>=` the minimum, else the minimum; clamped to the range. `setDecimals` is ignored.
7. **DropDownPanel(button, content, window)**: a frameless tool window below the button's bottom-left, toggled by the (checkable) button, following the window when it moves or resizes; Escape unchecks the button. It does not grab the mouse.
8. **ReadoutList**: a muted title (rich text when pairs exist) above label/value pairs (`Muted` labels, `ReadoutValue` plain-text values with an 8 px left margin) in a grid; it uses the largest column count `k` (from the number of pairs down to 2, else 1) for which `Σ_columns (max label width + max value width) + (k-1)·16 <= width`; pairs fill rows left to right; an empty stretch column of at least 16 px between pairs. With no pairs, only the message.
9. **ReflowRow(spacing)**: side by side when the width is at least the sum of the children's minimum widths plus spacing, else stacked; its minimum width is the largest child minimum.
10. **show_document(parent, title, html, width, height, extra=None)**: modal dialog deleted on close; margins 0, 0, 0, 16; spacing 16; a themed text browser (opens external links) with the HTML (stretch 1); `extra` in a holder with 16 px side margins (stretch 1); a `Close` button box with 16 px side margins.
11. **ResettableSplitter**: reports double-clicks on a handle with the handle index.
12. **Glyphs**: stroked paths on a 12×12 grid, stroke 1.4, round caps and joins, painted in a palette token (disabled: `TEXT_3`; checked: the checked token): `chevron_down M2.5 4.5 L6 8 L9.5 4.5`, `chevron_up M2.5 7.5 L6 4 L9.5 7.5`, `chevron_right M4.5 2.5 L8 6 L4.5 9.5`, `chevron_left M7.5 2.5 L4 6 L7.5 9.5`, `check M2.5 6.2 L5 8.6 L9.5 3.6`, `dash M3 6 L9 6`, `sort_up M6 10 L6 2 M3 5 L6 2 L9 5`, `sort_down M6 2 L6 10 M3 7 L6 10 L9 7`, `filter M1.5 2.5 L10.5 2.5 L7 6.5 L7 10 L5 9 L5 6.5 L1.5 2.5`, `range M1 3 L1 9 M11 3 L11 9 M3 6 L9 6 M4.6 4.4 L3 6 L4.6 7.6 M7.4 4.4 L9 6 L7.4 7.6`. Default token `TEXT_2`. For the stylesheet, glyph SVG files are written (only if the file does not exist yet, creating the folder) to `<temp>/spline_fitter_glyphs/{name}_{colour}_{pad}.svg`, where `<temp>` is the system temporary directory, `{colour}` is the token's hex value without `#` in lower case, and the path is written with forward slashes. Their whole content, UTF-8, no line break: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-pad} {-pad} {12+2pad} {12+2pad}"><path d="{path data}" fill="none" stroke="{#RRGGBB as in the palette}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>` (numbers as integers, for example `viewBox="-6 -6 24 24"`).
13. **Button boxes**: dialog buttons sit in a `QDialogButtonBox`, which orders them by role following the style's button layout (the application's `Fusion` style on Windows uses Qt's Windows layout: reset-role buttons at the left, then a stretch, then accept-role, then reject-role buttons). Accepting closes the dialog with its accept code, rejecting or closing with its reject code.

### 19.22 Chimes

`completion` plays the `chime` library's `success()` and `error` its `error()`, theme `material`, synchronously on a daemon thread named `chime`; every failure (missing library, no audio) is ignored.

### 19.23 Persisted settings

`QSettings("MSFra", "AirfoilSplineFitter")` keys: `recent_files` (list), `naca_code` (`2412`), `naca_points` (81), `naca_closed_te` (true), `cad_format` (0), `cad_units` (`mm`), `cad_scale` (1.0), `cad_merge` (false), `cad_close_te` (true), `theme/mode` (`system`), `theme/temperature` (0), `motion/scroll_ms` (350), `motion/zoom_ms` (650). Nothing else is persisted: the sidebar, display toggles, live preview mode, loss and comparison axes, splitter sizes and batch folders start at their defaults in every session.

---

## 20. Error handling summary

| Situation | Behaviour |
|---|---|
| Open: unsupported extension | status `Unsupported file type: {ext}`; nothing changes |
| Open: missing file (recent, drop) | status `File not found: {path}`; removed from recent files; error chime |
| Open while fitting | status `Cancel the running fit to open a file` |
| Airfoil unparseable / empty | status `Failed to load: {e}` or `No coordinates found in the file`; error chime; nothing changes |
| Clipboard without an airfoil | status `No airfoil coordinates on the clipboard: copy x y pairs, one point per line`; error chime |
| Invalid sidebar combination | prevented by 19.7; a remaining `ValueError` from `SplineConfig`/`OptimizationConfig` is shown as the status text and no fit starts |
| Fit raises | traceback printed; status `Fitting error: {e}`; error chime; the previous fit is redrawn |
| Solver did not converge | the fit is kept; status gains ` | Solver did not converge: {message}` |
| Fit report fails during a fit | `Fit report failed: {e}` printed; the fit is kept without a report |
| Log cannot be written | `Failed to save optimizer log: {e}` printed; the run continues |
| Project file invalid | status `Failed to load project: {e}` (`ProjectFormatError` naming the part); error chime; the open project and settings are unchanged |
| Project of a newer major version | `ValueError`: `... is not supported. Please update the application.` shown as above |
| Project read-only on save | status `Save failed: {path} is read-only.`; error chime; the file is untouched |
| Save, export or report fails | status `Save failed: {e}` / `Export failed: {e}` / `Report failed: {e}`; error chime |
| Batch file fails | counted as failed; status `[{i}/{T}] Failed: {file} - {e}`; error chime; the batch continues |
| Drawing error during a live preview | swallowed (the plot keeps the last good state); background tick errors are printed |

---

## 21. Acceptance examples

Each item must hold exactly unless a tolerance is stated; for the items that run an iterative fit (11, 12, 17), "exactly" applies to the reference build and 0.3 says what another build or language must reproduce. Every item was checked by running the reference implementation (the reference build: Windows 11, Python 3.12, NumPy 2.2.5, SciPy 1.15.2 on OpenBLAS 0.3.28, PyQt6 with Qt 6.9.0, offscreen); see 23.2. Items marked "(run)" give values measured there.

1. `open_uniform_knots(3, 5) = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]`.
2. `format_error(0.0000123) = "1.230e-05"`, `format_error(0.00456) = "0.004560"`, `format_error(12345.6) = "1.235e+04"` (decimal point `.`).
3. `format_duration`: `0.000123 -> "123 µs"`, `0.456 -> "456 ms"`, `12.34 -> "12.34 s"`, `123.4 -> "2m 3s"`, `3723.0 -> "1h 2m 3s"`.
4. `get_precision`: `"0.0123" -> 4`, `"1.2346e-03" -> 7`, `"1.23e+01" -> 1`.
5. `_number_fields`: `"0.95000-0.01200" -> ["0.95000", "-0.01200"]`; `"0,5 0,01" -> ["0.5", "0.01"]`; `"1.0, 0.5" -> ["1.0", "0.5"]`; `"1.0D-03 2.5d+00" -> ["1.0e-03", "2.5e+00"]`; `'(1.0; "2.0")' -> ["1.0", "2.0"]` (run).
6. `parse_airfoil_text` with filename name `"f"` (run):
   - `"NACA test\n1.0000 0.0000\n0.5000 0.0500\n0.0000 0.0000\n0.5000 -0.0500\n1.0000 0.0000\n"` -> `("f", "NACA test", [(1,0), (0.5,0.05), (0,0), (0.5,-0.05), (1,0)], 4)` (already unit chord: kept exactly);
   - `"NACA test\n100 0\n50 5\n0 0\n50 -5\n100 0\n"` -> same points, precision 2 (0-100 rule: `0 - log10(0.01)`);
   - Lednicer `"T\n3 3\n0 0\n0.5 0.06\n1 0\n0 0\n0.5 -0.05\n1 0\n"` -> `("f", "T", [(1,0), (0.5,0.06), (0,0), (0.5,-0.05), (1,0)], 2)`;
   - numeric name `"0012 64\n1 0\n0.5 0.06\n0 0\n0.5 -0.06\n1 0\n"` -> content name `"0012 64"`, five points, precision 2;
   - escaped name: a comment line `#c`, then the line `\#name`, then the five points -> content name `"#name"`;
   - `"x\n2 0.2\n1 0.1\n0 0\n1 -0.1\n2 -0.2\n"` -> `[(1,0.1), (0.5,0.05), (0,0), (0.5,-0.05), (1,-0.1)]` (divided by chord 2), precision 1 (`1 - log10(0.5) = 1.30` floored).
7. `scaled_precision(4, 0.01) = 6`, `scaled_precision(5, 0.5) = 5`, `scaled_precision(6, 1/0.98) = 6` (run).
8. `output_name` (run):
   - `("naca0012", SplineConfig(degree=5, num_control_points=30, num_sections=3, junction_continuity=("G2","C1")), OptimizationConfig(tolerance=1e-4, knot_placement="uniform", grow_control_points=False))` -> `naca0012.deg5.n30.s3.G2-C1.tol1e-4.spline.dat`;
   - the reference settings -> `naca0012.deg5.spline.dat`;
   - reference spline, `OptimizationConfig(tolerance=1e-5)` (model defaults free + grow), auto -> `naca0012.deg5.tol-auto.knots-free.grow.spline.dat`;
   - `("x", SplineConfig(degree=3, continuity=G2), OptimizationConfig(optimizer=DIRECT_LSQ, parameterization="arc_length", knot_placement="uniform", grow_control_points=False, smoothing_weight=0.01, optimize_le_position=True, max_points_per_section=150))` -> `x.deg3.C2.lsq.arc-length.smooth0.01.sub150.free-le.spline.dat`;
   - one section, otherwise reference -> `x.deg5.s1.spline.dat`.
9. `num`: `1e-4 -> "1e-4"`, `2.5e-5 -> "2.5e-5"`, `0.001 -> "1e-3"`, `0.0123 -> "0.0123"`, `0.5 -> "0.5"`, `1e-6 -> "1e-6"`, `3.33333e-7 -> "3.33e-7"`.
10. `sanitize_filename("a b/c") = "a_b_c"`; `file_stem("NACA 2412 t/c 12%") = "NACA 2412 t_c 12%"`; `file_stem("") = "airfoil"`.
11. Reference fits on `NACA4("2412", closed_te=True).points(81)` rounded to 6 decimals (161 points), two sections (81 + 81), degree 5, 25 control points (run; regression targets, relative tolerance 1e-3 on errors, exact on counts):
    - `default_tolerance(points, 6) = 1e-6`;
    - VarPro LM, G3 geometric, uniform knots, no growth, smoothing 1e-3: `fun = rms_error ≈ 8.978e-7`, `nit = 40`, success, message ``"`ftol` termination condition is satisfied."``, counts `[25, 25]`; report `max_deviation ≈ 4.587e-6`, `rms_deviation ≈ 8.978e-7`, `max_thickness ≈ 0.12004` at `≈ 0.2988`, `max_camber ≈ 0.020000` at `≈ 0.4065`, `leading_edge_radius ≈ 0.015698`, `curvature_extrema = 5`;
    - Direct LSQ, C2: `fun ≈ 2.942e-7`, `nit = 1`, counts `[25, 25]`; report `max_deviation ≈ 1.270e-6`, `rms_deviation ≈ 2.933e-7`, `curvature_extrema = 12`;
    - VarPro LM, tolerance `1e-6`, G2 geometric, model defaults otherwise (free knots, growth): success, 17 refits (`nit = 17`), control points `[10, 11]`, `fun` (max deviation) `9.1121e-7` (relative tolerance 1e-3; about 2 minutes on the reference machine).
12. CAD export of the VarPro fit of item 11 (run): STEP contains 2 `B_SPLINE_CURVE_WITH_KNOTS`; every IGES record is 80 characters and the last is `S      1G      4D      4P    112` + 40 spaces + `T      1`; DXF contains 2 `SPLINE` entities; `merge_curves` gives one degree-5 curve with 54 knots.
13. STEP reals: `1e-7 -> "1.E-7"`, `3.0 -> "3.0"`, `0.5 -> "0.5"`, `2.5e16 -> "2.5E16"`, `-1e16 -> "-1.E16"` (run).
14. Spline file round trip: `export_spline_data` then `parse_spline_data` returns the same degree, counts, continuity, knots (exactly, `repr`) and control points (to the written decimals).
15. Project round trip: `save_project` then `load_project` keeps run numbers, settings, control points (exactly), result fields and loss history; a run with optimizer `"bfgs"` loads as VarPro LM.
16. Theme greys: table of 19.19.1 (run).
17. Window (run, offscreen Qt, decimal point `.`): at start status `Ready`, title `Airfoil Spline Fitter`; defaults VarPro LM, Fixed count, degree 5, 25 control points, 2 sections, G3, Geometric, Uniform, Grow off, Auto on, Live preview Moderate. New NACA 2412 (81 per side): status `Created: NACA 2412 (161 points, 2 sections: [81, 81])`, automatic tolerance `1e-06`. Fit: status `Completed (VarPro LM): 40 evaluations in {duration} | Error 8.978e-07 | max deviation 4.59e-06, RMS 8.98e-07, 49 control points`; project label `Unsaved project*` / `1 optimization run`; title `Unsaved project[*] - Airfoil Spline Fitter`; one log `{YYYYmmdd_HHMMSS}_NACA_2412_gn-varpro.csv`; readouts `Max deviation`, `4.59e-06`, `2 × 25`. Then Direct LSQ with C2: `Completed (Direct LSQ): 2 solves in {duration} | Error 2.942e-07 | max deviation 1.27e-06, RMS 2.93e-07, 49 control points`, two runs. Undo returns the optimizer and continuity to VarPro LM and G3 in one step. A control-point sweep 25..26 (VarPro, G3): `Sweep complete: 2 fits in {duration} | Best: Control points 25, error 8.978e-07`, after which the sweep is off and the count is 25.
18. Batch: two `.dat` files converted into a target folder produce two `.spline.dat` files named by item 8; running again skips both (`Skipped 2 already in the target folder with these settings`).

Test suites covering these rules: `tests/` (application) and `packages/bspline-varpro/tests/` (library). `tests/test_design_language.py` checks the theming rules (no colour literals or stylesheets outside the theme, control heights, label wording).

---

## 22. Known inconsistencies and their resolution

1. **README vs window defaults.** The README describes Tolerance goal and Auto tolerance as defaults. The window starts in the Fixed count goal (first segment) with Auto checked. This spec requires the window behaviour.
2. **Model vs window defaults.** `OptimizationConfig` defaults `knot_placement="free"`, `grow_control_points=True`; the window starts with Uniform and Grow off. Both are required: programmatic configs use the model defaults, the window its widget defaults.
3. **Smoothing in tolerance mode.** The sidebar Smoothing value is disabled and ignored in the tolerance goal; tolerance fits use `1e-7`.
4. **`fit_to_tolerance` subsampling default** is 200 in the library; the application passes its own value (`None` unless Subsample is checked).
5. **Two airfoil readers.** The application uses only section 3; the library's `read_dat` (3.9) differs and is not used.
6. **Direct LSQ junction type.** Always parametric; the Type control is hidden and the configuration's geometric flag is false for Direct LSQ.
7. **Continuity labels.** Shown as `C0..C3` or `G0..G3` by the type (19.6); the stored value is always the `ContinuityMode` name `G0..G3`.
8. **Retired fields** in projects are ignored (15.4); `smoothing_factor`, `early_stop_patience` and `relative_tolerance` survive only in the model and batch naming.
9. **Two knot checks.** `BSplineCurve2D` uses the weaker check (2.1); the library the strict one. Every knot vector the application produces passes both.
10. **Curve objects.** `FitResult.curves()` returns `BSplineCurve2D` (no SciPy attributes); everything passed to the library uses `scipy_curves(result)`.
11. **About.** Help > About only shows `Airfoil Spline Fitter v1.0` in the status bar although the package version is 2.0.0; this text is required as is.
12. **Comb chord vs deviation chord.** The comb scales by the x extent of the fitted curves (sampled), the deviation by the x extent of the data points (19.14.3, 19.14.5).
13. **Quiet undo merge.** The history keeps a quiet change and a user change that arrive within 500 ms apart: the quiet one merges into the state before the user's step (19.8).
14. **Exit (resolved).** The reference implementation at `2febd13` returned the exit code from `main()` and let interpreter shutdown destroy the window and the `QApplication`; the process then usually crashed (exit status 139: 5 of 6 offscreen runs, 4 of 6 native Windows runs). `main()` now ends the process itself as section 1.1 step 5 describes. This spec requires that behaviour.
15. **Line ends of written files.** The application's writers open their files in text mode without fixing the newline, so `.dat`, `.spline.dat`, `.asfproj` and CSV log files carry `\r\n` on Windows and `\n` elsewhere, while the library's CAD and `write_dat` files always carry `\n` (section 1, "Text files"). Nothing in the code or its tests states this as intended. This spec records the behaviour and leaves these line ends out of the byte comparison of 0.3: an implementation may write the platform line end, as the reference does, or always `\n`. Either way the files load identically on every platform, because every reader splits on any line end (3.2, 14.2, 15.4).
16. **Non-finite numbers in project files.** `save_project` writes `NaN`, `Infinity` and `-Infinity` as bare tokens (15.1), which standard JSON parsers reject. This spec requires writing and reading them as the reference does, so existing projects keep loading.

---

## 23. Review and verification record (informative)

### 23.1 External-developer review

The document was re-read as an engineer holding only this file, asking at each rule "what exactly happens, with which numbers and which text?". The previous draft of this file was compared line by line with the source; corrections made:

- Sidebar (19.4): row labels are `Max deviation`, `Knots`, `Per junction`, `Subsample`, `Smoothing`, `Junctions`; the Curve rows are ordered Degree, Control points, Sections; the optimizer is a segmented control; Live preview and the loss axes are plot toolbars, not sidebar rows.
- Help > About shows a status message, not a dialog.
- The comb's length scale uses the curves' x extent, not the data's; comb colours are by section parity, blue first.
- The settings history keeps quiet and user changes in separate steps (the earlier draft merged them).
- QSettings persist only the keys of 19.23 (the earlier draft listed sidebar values).
- VarPro speed rows, joint refinement damping floors, the polish floor formula, restart rule, `fit_to_tolerance` stop and record semantics, and LM ftol test (uses the new cost) are now stated exactly.
- The curve validity rule (19.7), visibility rules (19.6), Advanced badge (19.9), sweeps (19.13), explorer filters and pivot (19.17), project open rollback (19.16.2), batch statuses (19.18), theme maths (19.19) and motion (19.20) were added in full; they were summaries before.
- STEP real formatting examples and the DXF layer-name message were corrected.

Ambiguities found and resolved in the text:

| Question | Resolution |
|---|---|
| Which rows count for file precision? | Rows remaining after the name/Lednicer header is removed, before outlier filtering (3.6). |
| Is the BOM removed per line? | Only at the start of the text (3.2). |
| What does the Paste shortcut do in a text field? | The field pastes text; the action is not triggered (19.3). |
| Which knots does Save spline write for Direct LSQ? | The open uniform knots of each section (`FitResult.knots`). |
| Which continuity does Save spline write? | The shown run's, else the sidebar's (19.15.2). |
| Which smoothing does a sweep vary in tolerance mode? | None: smoothing and control-point sweeps are off in tolerance mode (19.6). |
| What happens to the project when the section count changes? | Nothing; only the airfoil is re-split and the fit cleared (19.11.6). |
| Does a cancelled fit replace a finished run? | No (19.16.4). |
| Is the evaluation count of Direct LSQ 1? | 1 with fixed parameters; otherwise 2 (one solve, one joint stage), or 3 with free junctions (one solve, two joint stages) (10). `nit` is always 1. |

### 23.2 Verification by execution

The verification results are recorded in `spec-verification.md`, beside this file, together with the SHA-256 of the exact `spec.md` they were run against. That hash is of revision 1 of this file as stored in git, with `\n` line ends (commit `000f500`; a Windows checkout with line-end conversion hashes differently). Revision 2, this text, was edited after those runs; 23.4 lists the edits and what was re-run. The checks, all run on the reference build (Python 3.12, NumPy 2.2.5, SciPy 1.15.2 on OpenBLAS 0.3.28, PyQt6 with Qt 6.9.0, offscreen platform) against the source at `2febd13` with the exit fix applied (the tree that is now commit `6db8ad3`):

1. **Acceptance:** every item of section 21 is asserted literally by one script. Item 16 parses the grey table out of this file. Items 17 and 18 drive a real `SplineFitterWindow`.
2. **Clean-room differential test:** components were reimplemented following the text of this document (sections 2.1, 2.3, 2.6, 3, 4.1, 4.2, 5.2, 7.1, 7.3, 7.4, 8.2, 8.4 residual, 8.5, 9.3, 9.7, 12.6, 13.1, 14.2, 14.4, 16, 18, 19.17.2, 19.19.1, 19.21 items 4 and 6). Each reimplementation was compared with the source on random inputs: exact equality for discrete results and text, and floating-point agreement within 1e-6 relative for the VarPro residual (its solver is a different LAPACK routine) and within 1e-9 to 1e-15 elsewhere. The airfoil parser was fuzzed with Selig and Lednicer files in many number formats, scales, separators and name forms.
3. **Blind reimplementation:** a separate agent that was given only this document (and instructed to read no other file of the repository, and to record every place where the text was unclear) implemented the same components as item 2 (`spec-verification/blind/blind_impl.py`) and the GUI rules of items 5 and 6 (19.6, 19.7, 19.8, 19.13, 19.16.4, 19.17.1, 19.17.2; `spec-verification/blind/blind_gui.py`); the same harnesses compare them with the source, the GUI harnesses keeping only Qt's value clamping and signal routing and taking every application rule from the blind module. The same agent also wrote, from this document alone, the expected values of items 7 to 10: the stylesheet, palette and glyph files (`blind_theme.py`), the comb and deviation geometry (`blind_plot.py`), and the status texts, dialog arguments and contents, file defaults and texts of the file flows, interactions and dialogs (`blind_expect.py`); items 7 to 10 are run with these in place of their own expectations as well. Its notes on unclear text were resolved in this document, after which the agent revised its implementation from the revised text alone.
4. **Bit-for-bit reproducibility:** three separate processes computed the item-11 fits and hashed every output.
5. **GUI rule model test:** sections 19.6 and 19.7, including the signal cascades they name, are implemented as a model following this document (`spec-verification/gui_model.py`). Random user actions (optimizer, goal, Grow points, Auto, degree, control points, sections, continuity, type, Per junction text, sweep toggles and sweep ends) are applied to the model and to a real `SplineFitterWindow`, and after every action 25 widget states must be equal: values and ranges of the degree, control-point and section boxes and their sweep ends, the continuity level, the sweep states, and the 19.6 visibility, enabling and label flags. Deliberately wrong models (the 5.2 rule off by one, the degree sweep cap without its `- 1`, a wrong label) must be detected.
6. **GUI logic model test:** sections 19.8 (settings history: steps, quiet merging, undo/redo availability, the 200-step limit, checkpoint and rollback, and the widget values each undo or redo restores), 19.17.1 (all 25 column values and texts), 19.17.2 (row filter with global terms, column exclusions and Invert), 19.17.3 (sort order), 19.16.4 (run matching) and 19.13 (sweep comparison series) are implemented following this document (`spec-verification/gui_logic.py`) and compared with the source on random runs and random action sequences.
7. **Theme check:** the stylesheet the running application installs is compared with Appendix A filled in from the token tables of 19.19.1 (whitespace and comments ignored), with every Qt palette role of 19.19.2 and with the content of every glyph file of 19.21 item 12, after real theme switches in light and dark mode at temperatures -100, -40, 0, 25 and 100 (`spec-verification/theme_check.py`).
8. **Plot geometry check:** the curvature comb (19.14.3, all four quantities) and the deviation whiskers, envelopes and tolerance band (19.14.5, with the projection of 12.1 and the foot points of 12.5) are computed following this document and compared with the data the real plot manager sets, for random fits of 1 to 4 sections (`spec-verification/plot_check.py`).
9. **File and project flow check:** a real window runs the flows of 19.11, 19.15 and 19.16 (saving without a fit, unsupported and missing files, a bad paste, save fitted airfoil, save spline, CAD export with each cancel, the fit report window, project save-as with the added suffix and the replace prompt, the recent-files menu, deleting and restoring runs, cancelling at the unsaved-changes prompt, a broken project file) with dialogs replaced by stubs, and its status texts, dialog titles, default paths, filters and written files are compared with the literals of this document (`spec-verification/file_flows_check.py`).
10. **Interaction and dialog check:** drag and drop (19.11.2), pivoting the explorer (19.17.3), the column filter drop-down (19.17.4: values and their order, condition, Keep matching, Add matching, Select all, single values, Clear filter), the easing curve and parameters of scrolling and zooming (19.20), the button-box order (19.21 item 13) and the rows, items, ranges and buttons of the New NACA airfoil, Export CAD, Batch convert, Theme and Scrolling and zooming dialogs, in a real window (`spec-verification/ui_extras_check.py`).
11. **Exit status:** `main()` is run with its window closed by a timer (or the event loop ended with `QApplication.exit(3)`), on the offscreen and on the native Windows platform. The exit status must equal the event loop's code, and a QSettings value written just before the window closes must be readable by a new process (`spec-verification/exit_probe.py`).
12. **Full test suites:** `python -m pytest tests packages/bspline-varpro/tests`.

### 23.3 Completeness statement

The aim is that every behaviour of the application is specified and nothing is left to the implementer's choice. No finite set of checks proves that for a document of this size: the checks of 23.2 show it for the behaviours they compare, and the review of revision 2 (23.4) still found behaviours that revision 1 stated wrongly (line ends) or not at all (non-finite numbers in project files) because no check compared them. A reader who finds a rule that admits two readings should treat it as a defect of this document, not as a freedom. The boundaries below are deliberate dependencies, specified as such:

- **Iterative fits.** Every iteration is specified exactly: unknowns, residual rows and their order, Jacobians, damping updates, acceptance and stopping tests, constants (sections 8-10). The dense linear algebra is pinned to these LAPACK routines (double precision), as called by the reference build (NumPy 2.2.5 and SciPy 1.15.2, both on OpenBLAS 0.3.28):

  | Operation | Routine |
  |---|---|
  | KKT LU factor and solve (8.3) | `dgetrf`, `dgetrs` |
  | Cholesky step of LM (8.5) | `dpotrf`, `dpotrs` |
  | `matrix_root` Cholesky / fallback (8.4) | `dpotrf` / `dsyevd` |
  | Direct LSQ KKT solve (10, `scipy.linalg.solve` with the default structure detection, which finds the KKT matrix symmetric) | `dsysv` (Bunch-Kaufman, lower triangle) with a `dsycon` condition estimate; a singular matrix raises, an ill-conditioned one only warns |
  | joint Schur solve (9.7, `numpy.linalg.solve`) | `dgesv` |
  | null space (9.7) and pseudo-inverses (8.3, 12.6) | `dgesdd` |
  | `P0` least squares (9.7), knot removal fit (14.7) | `dgelsd` |
  | bounded leading-edge search (13.2) | Brent's bounded minimization (`fminbound`, `xatol = 1e-14`) |
  | Gauss-Legendre nodes (2.5) | `numpy.polynomial.legendre.leggauss` (companion-matrix eigenvalues, one Newton correction, symmetrized) |

  On the reference build, repeated runs give bit-identical results (23.2: three processes, identical hashes of all outputs of the item-11 fits). With another correct LAPACK implementation, the rounding inside these routines differs in the last bits. Because the solvers stop on relative tests (`1e-10`, `1e-12`), results then agree to the tolerances of section 21 item 11, which is the conformance test for such builds.
- **Widget drawing.** The application sets Qt's `Fusion` style, the full palette (19.19.2), the complete stylesheet (Appendix A), all fonts, sizes, plot pens and brushes, and z-order. How Qt and pyqtgraph rasterize those (anti-aliasing, glyph shapes) is the toolkit's behaviour, a declared dependency (section 1).
- **Icons.** Appendix B gives both files byte for byte.

### 23.4 Revision 2 (2026-11-10)

Revision 2 is a review of revision 1 against the source at `6db8ad3`. It changes no algorithm, constant, label or message. What it changes, and how each change was checked:

| Change | Sections | Checked by |
|---|---|---|
| Line ends of the application's written files follow the platform (`\r\n` on the reference build); revision 1 said `\n` | 1, 14.1, 14.2, 17, 22 item 15 | writing a `.dat`, a `.spline.dat`, a log and a project with the source on Windows and reading the bytes |
| Project files carry `NaN` / `Infinity` tokens and no final line end; the JSON blocks are key lists, not literal content | 15.1, 22 item 16 | saving and reloading a project with a NaN error and an infinite leading-edge radius |
| Conformance: the fit regression item is 21 item 11 (revision 1 named item 10); byte equality is scoped to the writers given equal values; two levels for results of iterative fits, which removes the conflict between "bitwise identity is not required" and exact evaluation counts | 0.3, 21 | reading 0.3, 21 and 23.3 side by side |
| Baseline named as commit `6db8ad3`; revision line; statement that this file is not edited after the runs replaced | header, 23.2 | `git show --stat 6db8ad3`; the SHA-256 of the committed revision 1 equals the recorded one |
| Library `read_dat` stated as a procedure (it was a summary) | 3.9 | `bspline_varpro/airfoil.py` |
| Definitions that were used but missing: `L_node`, `JointProblem.control_points`, the `data` argument of `validate_chain_parameters` | 7.3, 8.1, 9.7 | `constraints.py`, `joint.py`, `chain.py` |
| Wording: Direct LSQ continuity level and its unreachable guards; the twelve palette greys; the slider gradient's out-of-range temperatures; command-line arguments and Qt; the `.spline.dat` layout block | 10.1, 19.19.1, Appendix A, 1.1, 14.2 | the source files named in those sections |
| Contents table, normative/informative statement, note on reused symbols; signatures moved out of the longest headings; completeness statement reworded as an aim | header, 8-9, 13.6, 14.2, 23.3 | no behaviour involved |

Of the checks of 23.2, only the theme check (item 7) and acceptance item 16 read this file; the theme check was re-run on revision 2 (30 of 30), and the grey table that item 16 parses is unchanged (13 rows). `spec-verification.md` itself was not regenerated: its hashes and results describe revision 1. The blind reimplementation (item 3) was not repeated for the changed passages: the line-end and non-finite-number rules have no blind confirmation.

---

## Appendix A. Application stylesheet

The stylesheet is this template with `{TOKEN}` replaced by the current `Palette` value (19.19.1), `{H}` by `24` (`CONTROL_HEIGHT - 2`), `{FIXED}` by `min-height: 24px; max-height: 24px;`, `{MONO}` by `Cascadia Mono, Consolas, Menlo, monospace`, and `{glyph:NAME:TOKEN:PAD}` by the path (forward slashes) of the glyph SVG file of 19.21 item 12 (`TOKEN` defaults to `TEXT_2`, `PAD` to 0). `{SLIDER_GRADIENT}` is `stop:0 #6288BA, stop:0.5 #868686, stop:1 #968645` (`grey(0.62, -4)`, `grey(0.62, 0)`, `grey(0.62, 4)`, independent of the theme; the temperatures `-4` and `4` are deliberately four times beyond the user's range `[-1, 1]` of 19.19.1 so the groove reads as cool to warm, and the formula of 19.19.1 is applied to them unchanged).

```css
QWidget { color: {TEXT}; }
QMainWindow, QDialog { background: {WINDOW}; }
QToolTip { background: {FIELD}; color: {TEXT}; border: 1px solid {RULE_STRONG}; padding: 4px 6px; }
QMenuBar { background: {WINDOW}; border-bottom: 1px solid {RULE}; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected { background: {FIELD_HOVER}; }
QMenu { background: {FIELD}; border: 1px solid {RULE_STRONG}; padding: 4px 0; }
QMenu::item { padding: 5px 24px 5px 20px; }
QMenu::item:selected { background: {SELECT}; }
QMenu::item:disabled { color: {TEXT_3}; }
QMenu::separator { height: 1px; background: {RULE}; margin: 4px 0; }
QStatusBar { background: {WINDOW}; color: {TEXT_2}; border-top: 1px solid {RULE}; }
QStatusBar::item { border: none; }
QSplitter::handle { background: {RULE}; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle { background: {RULE_STRONG}; border-radius: 3px; min-height: 24px; min-width: 24px; margin: 2px; }
QScrollBar::handle:hover { background: {TEXT_3}; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QPushButton, QToolButton { background: {FIELD}; border: 1px solid {RULE}; border-radius: 3px; padding: 0 10px; {FIXED} }
QPushButton:hover, QToolButton:hover { background: {FIELD_HOVER}; border-color: {RULE_STRONG}; }
QPushButton:pressed, QToolButton:pressed { background: {RULE}; }
QPushButton:checked, QToolButton:checked { background: {SELECT}; border-color: {SELECT_LINE}; }
QPushButton:disabled, QToolButton:disabled { color: {TEXT_3}; background: {PANEL}; border-color: {RULE}; }
QPushButton:focus, QToolButton:focus { outline: none; }
QPushButton:default { border-color: {SELECT_LINE}; }
QToolButton#ArrowButton { padding: 0 5px; border-top-left-radius: 0; border-bottom-left-radius: 0; margin-left: -1px; }
QToolButton#SplitToggle { border-top-right-radius: 0; border-bottom-right-radius: 0; }
QLineEdit QToolButton { background: transparent; border: none; padding: 0; min-height: 0; max-height: 16777215px; }
QPushButton#PrimaryButton { background: {INK}; color: {ON_INK}; border-color: {INK}; font-weight: 600; }
QPushButton#PrimaryButton:hover { background: {INK_HOVER}; border-color: {INK_HOVER}; }
QPushButton#PrimaryButton:pressed { background: {INK_PRESSED}; border-color: {INK_PRESSED}; }
QPushButton#PrimaryButton:disabled { background: {RULE}; border-color: {RULE}; color: {TEXT_3}; }
QPushButton#IconButton { background: transparent; border-color: transparent; padding: 0; min-width: {H}px; max-width: {H}px; }
QPushButton#IconButton:hover { background: {FIELD_HOVER}; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: {FIELD}; border: 1px solid {RULE}; border-radius: 3px; padding: 0 6px; {FIXED} selection-background-color: {SELECT}; }
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover { border-color: {RULE_STRONG}; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: {SELECT_LINE}; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { color: {TEXT_3}; background: {PANEL}; }
QSpinBox, QDoubleSpinBox { font-family: {MONO}; }
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button { width: 16px; border: none; background: transparent; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: {FIELD_HOVER}; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url({glyph:chevron_up}); width: 9px; height: 9px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url({glyph:chevron_down}); width: 9px; height: 9px; }
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled, QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled { image: none; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { image: url({glyph:chevron_down}); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: {FIELD}; border: 1px solid {RULE_STRONG}; selection-background-color: {SELECT}; outline: none; }
QCheckBox { spacing: 6px; }
QCheckBox:disabled { color: {TEXT_3}; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid {RULE_STRONG}; border-radius: 2px; background: {FIELD}; }
QCheckBox::indicator:hover { border-color: {SELECT_LINE}; }
QCheckBox::indicator:checked { background: {SELECT}; border-color: {SELECT_LINE}; image: url({glyph:check:TEXT}); }
QCheckBox::indicator:indeterminate { background: {SELECT}; border-color: {SELECT_LINE}; image: url({glyph:dash:TEXT}); }
QCheckBox::indicator:disabled { background: {PANEL}; border-color: {RULE}; }
QListView::indicator { width: 14px; height: 14px; border: 1px solid {RULE_STRONG}; border-radius: 2px; background: {FIELD}; }
QListView::indicator:checked { background: {SELECT}; border-color: {SELECT_LINE}; image: url({glyph:check:TEXT}); }
QListView::indicator:disabled { background: {PANEL}; border-color: {RULE}; }
QListView::item { padding: 2px 0; }
QListView::item:disabled { color: {TEXT_3}; }
QListView::item:hover { background: {FIELD_HOVER}; }
QSlider { {FIXED} }
QSlider::groove:horizontal { height: 4px; background: {RULE}; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 7px; background: {TEXT_2}; border: 1px solid {CANVAS}; }
QSlider::handle:horizontal:hover { background: {TEXT}; }
QSlider#TemperatureSlider::groove:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {SLIDER_GRADIENT}); }
QTreeView, QListView, QTableView, QTextBrowser { background: {PANEL}; border: 1px solid {RULE}; border-radius: 3px; outline: none; }
QTreeView::item { padding: 2px 0; }
QTableView::item { padding: 0 6px; }
QTreeView::item:hover { background: {FIELD}; }
QTreeView::item:selected, QTableView::item:selected { background: {SELECT}; color: {TEXT}; }
QTreeView::branch { background: transparent; }
QTreeView::branch:has-children:closed { image: url({glyph:chevron_right:TEXT_2:6}); }
QTreeView::branch:has-children:open { image: url({glyph:chevron_down:TEXT_2:6}); }
QHeaderView { background: {PANEL}; border: none; border-bottom: 1px solid {RULE}; }
QHeaderView::section { background: {PANEL}; color: {TEXT_3}; border: none; padding: 4px 6px; }
QHeaderView:vertical { border-bottom: none; border-right: 1px solid {RULE}; }
QHeaderView#FilterHeader::section { padding: 4px 36px 4px 6px; }
PlotWidget, QGraphicsView { border: 1px solid {RULE}; border-radius: 3px; }
#Sidebar, #Explorer { background: {PANEL}; }
#ExplorerRail { background: {PANEL}; border-left: 1px solid {RULE}; }
#SidebarSection { border-bottom: 1px solid {RULE}; }
#SectionCaption, #SectionHeader { color: {TEXT_3}; font-size: 10px; font-weight: 600; letter-spacing: 1px; }
QPushButton#SectionHeader { text-align: left; border: none; padding: 0; background: transparent; min-height: 0; max-height: 16777215px; }
QPushButton#SectionHeader:hover { color: {TEXT}; background: transparent; }
#SectionBadge { color: {TEXT_2}; font-size: 10px; }
#FileName { font-weight: 600; }
#Muted { color: {TEXT_3}; font-size: 11px; }
#Readout { font-family: {MONO}; font-size: 15px; color: {TEXT}; }
#ReadoutValue { font-family: {MONO}; color: {TEXT}; }
#RunBar { background: {WINDOW}; border-top: 1px solid {RULE}; }
#DropDownPanel { background: {FIELD}; border: 1px solid {RULE_STRONG}; border-radius: 3px; }
#FormLabel { color: {TEXT_2}; }
QPushButton#Segment { border-radius: 0; margin-left: -1px; padding: 0 6px; }
QPushButton#Segment[position="first"] { border-top-left-radius: 3px; border-bottom-left-radius: 3px; margin-left: 0; }
QPushButton#Segment[position="last"] { border-top-right-radius: 3px; border-bottom-right-radius: 3px; }
QPushButton#Segment:checked { color: {TEXT}; }
QPushButton#SweepToggle { padding: 0; }
```

(`36px` = `2·GLYPH + TIGHT + 2·GAP`.)

Object names used: `Sidebar`, `Explorer`, `ExplorerRail`, `SidebarSection`, `SectionCaption`, `SectionHeader`, `SectionBadge`, `FileName`, `Muted`, `Readout`, `ReadoutValue`, `RunBar`, `DropDownPanel`, `FormLabel`, `Segment`, `SweepToggle`, `PrimaryButton`, `IconButton`, `ArrowButton`, `SplitToggle`, `FilterHeader`, `TemperatureSlider`.
