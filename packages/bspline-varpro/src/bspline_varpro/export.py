"""Exact CAD export of B-spline chains (STEP, IGES, DXF) and resampling.

The writers are dependency free and store the curves exactly: degree, knot
vector, and control points (non-rational B-splines).

* STEP AP214: ``B_SPLINE_CURVE_WITH_KNOTS`` in a
  ``GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION`` of one product.
* IGES 5.3: entity 126 (rational B-spline curve with unit weights).
* DXF R2000: ``SPLINE`` entities in a minimal file (HEADER and ENTITIES).
"""

from __future__ import annotations

import datetime as _dt
import unicodedata
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
from scipy.interpolate import BSpline

from .basis import gauss_legendre

CurveInput = Union["FitResult", BSpline, Sequence[BSpline]]  # noqa: F821


def as_curves(curves: CurveInput) -> List[BSpline]:
    """B-spline list from a FitResult, a single BSpline, or a sequence of them."""
    if hasattr(curves, "curves") and callable(curves.curves):
        return list(curves.curves())
    if isinstance(curves, BSpline):
        return [curves]
    return list(curves)


def _knots_controls(curve: BSpline, scale: float):
    k = curve.k
    n = len(curve.t) - k - 1
    ctrl = np.asarray(curve.c, dtype=np.float64)[:n] * scale
    if ctrl.ndim == 1:
        ctrl = ctrl[:, None]
    if ctrl.shape[1] > 3:
        raise ValueError("Only curves in 1 to 3 dimensions can be exported.")
    ctrl3 = np.zeros((n, 3))
    ctrl3[:, :ctrl.shape[1]] = ctrl
    return np.asarray(curve.t, dtype=np.float64), ctrl3, k


def _closed(ctrl: np.ndarray) -> bool:
    """Whether a clamped curve ends where it starts (its end control points coincide)."""
    scale = max(1.0, float(np.max(np.abs(ctrl))))
    return bool(np.linalg.norm(ctrl[-1] - ctrl[0]) <= 1e-12 * scale)


def arc_length(curve: BSpline, a: float = None, b: float = None, quad_order: int = 16) -> float:
    """Arc length by Gauss-Legendre quadrature on every knot span."""
    k = curve.k
    lo = curve.t[k] if a is None else a
    hi = curve.t[-k - 1] if b is None else b
    breaks = np.unique(np.concatenate([[lo, hi], curve.t[(curve.t > lo) & (curve.t < hi)]]))
    nodes, weights = gauss_legendre(quad_order)
    total = 0.0
    for x0, x1 in zip(breaks[:-1], breaks[1:]):
        u = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * nodes
        total += 0.5 * (x1 - x0) * float(np.sum(weights * np.linalg.norm(curve(u, nu=1), axis=-1)))
    return total


def merge_curves(curves: CurveInput) -> BSpline:
    """Join a position-continuous chain into one B-spline on [0, 1].

    The geometry is reproduced exactly (including geometric G1-G3 junctions).
    Parameter ranges are scaled so the parametric speed is equal on both sides
    of every tangent-continuous junction, which makes the merged curve C1
    there; junction knots then keep only the multiplicity the actual
    parametric continuity needs (``degree`` at a corner), so CAD readers do not
    split the curve.
    """
    curves = as_curves(curves)
    degree = curves[0].k
    if any(c.k != degree for c in curves):
        raise ValueError("All sections must have the same degree to be merged.")
    lengths = [arc_length(curves[0])]
    for a, b in zip(curves[:-1], curves[1:]):
        # Derivatives w.r.t. each section's normalized parameter (its domain mapped to [0, 1])
        da = a(a.t[-degree - 1], nu=1) * (a.t[-degree - 1] - a.t[degree])
        db = b(b.t[degree], nu=1) * (b.t[-degree - 1] - b.t[degree])
        na, nb = np.linalg.norm(da), np.linalg.norm(db)
        smooth = na > 0 and nb > 0 and np.linalg.norm(da / na - db / nb) < 1e-7
        # Equal parametric speed on both sides makes a tangent-continuous junction C1.
        lengths.append(lengths[-1] * (nb / na if smooth else arc_length(b) / arc_length(a)))
    lengths = np.array(lengths)
    bounds = np.concatenate([[0.0], np.cumsum(lengths) / np.sum(lengths)])
    last = len(curves) - 1
    knots: List[np.ndarray] = []
    ctrl: List[np.ndarray] = []
    for s, curve in enumerate(curves):
        t = np.asarray(curve.t, dtype=np.float64)
        lo, hi = t[degree], t[-degree - 1]
        mapped = bounds[s] + (t - lo) / (hi - lo) * (bounds[s + 1] - bounds[s])
        mapped[:degree + 1] = bounds[s]
        mapped[-degree - 1:] = bounds[s + 1]
        c = np.array(curve.c, dtype=np.float64)[:len(t) - degree - 1]
        # Drop one end knot at every junction so it keeps multiplicity `degree`.
        head = 0 if s == 0 else degree + 1
        tail = len(mapped) if s == last else len(mapped) - 1
        knots.append(mapped[head:tail])
        if s > 0:
            ctrl[-1][-1] = 0.5 * (ctrl[-1][-1] + c[0])
            c = c[1:]
        ctrl.append(c)
    merged_knots = np.concatenate(knots)
    merged_knots[:degree + 1] = 0.0
    merged_knots[-degree - 1:] = 1.0
    merged = BSpline(merged_knots, np.vstack(ctrl), degree)
    for u in bounds[1:-1]:
        for _ in range(degree - 1):
            reduced = _remove_knot(merged, u)
            if reduced is merged:
                break
            merged = reduced
    return merged


def _remove_knot(curve: BSpline, u: float, rtol: float = 1e-10) -> BSpline:
    """Remove one occurrence of knot ``u`` if the curve stays the same within ``rtol``."""
    k = curve.k
    t = np.asarray(curve.t)
    i = int(np.flatnonzero(t == u)[0])
    new_t = np.delete(t, i)
    breaks = np.unique(new_t[k:len(new_t) - k])
    nodes = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 2 * k + 3)))
    x = np.unique(np.concatenate([a + (b - a) * nodes for a, b in zip(breaks[:-1], breaks[1:])]))
    A = BSpline.design_matrix(x, new_t, k).toarray()
    target = curve(x)
    ctrl = np.linalg.lstsq(A, target, rcond=None)[0]
    candidate = BSpline(new_t, ctrl, k)
    check = np.linspace(new_t[0], new_t[-1], 20 * len(new_t))
    scale = max(1.0, float(np.max(np.abs(target))))
    if np.max(np.abs(candidate(check) - curve(check))) > rtol * scale:
        return curve
    ctrl[0], ctrl[-1] = curve.c[0], curve.c[len(t) - k - 2]
    return BSpline(new_t, ctrl, k)


def _distinct(knots: np.ndarray):
    values, counts = [], []
    for k in knots:
        if values and k == values[-1]:
            counts[-1] += 1
        else:
            values.append(float(k))
            counts.append(1)
    return values, counts


def _finite(v: float) -> float:
    v = float(v)
    if not np.isfinite(v):
        raise ValueError("Cannot export non-finite coordinates or knots.")
    return v


def _step_real(v: float) -> str:
    # repr is the shortest text that reads back as the same double
    s = repr(_finite(v)).upper()
    if "E" in s:
        mantissa, exponent = s.split("E")
        if "." not in mantissa:
            mantissa += "."
        return f"{mantissa}E{int(exponent)}"
    return s if "." in s else s + "."


def _step_string(text: str) -> str:
    r"""STEP string literal (ISO 10303-21): ``'`` and ``\`` doubled, non-ASCII as ``\X2\...\X0\``."""
    out = []
    for ch in text:
        code = ord(ch)
        if ch == "'":
            out.append("''")
        elif ch == "\\":
            out.append("\\\\")
        elif 32 <= code < 127:
            out.append(ch)
        elif code <= 0xFFFF:
            out.append(r"\X2" + f"\\{code:04X}" + r"\X0" + "\\")
        else:
            out.append(r"\X4" + f"\\{code:08X}" + r"\X0" + "\\")
    return "'" + "".join(out) + "'"


def to_step(curves: CurveInput, name: str = "curves", scale: float = 1.0, units: str = "mm") -> str:
    """STEP AP214 text with one ``B_SPLINE_CURVE_WITH_KNOTS`` per curve.

    Args:
        curves: FitResult, BSpline, or sequence of BSplines (2D curves get z = 0).
        name: Product name.
        scale: Factor applied to all coordinates (e.g. chord length).
        units: ``"mm"``, ``"m"``, or ``"inch"``.
    """
    unit_entity = {
        "mm": "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))",
        "m": "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))",
        "inch": None,
    }
    if units not in unit_entity:
        raise ValueError("units must be 'mm', 'm', or 'inch'.")
    entities: List[str] = []

    def add(text: str) -> int:
        entities.append(text)
        return len(entities)

    ctx = add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{ctx})")
    pctx = add(f"PRODUCT_CONTEXT('',#{ctx},'mechanical')")
    product = add(f"PRODUCT({_step_string(name)},{_step_string(name)},'',(#{pctx}))")
    add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{product}))")
    pdf = add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
    pdctx = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{ctx},'design')")
    pd = add(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdctx})")
    pds = add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
    if units == "inch":
        mm = add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
        factor = add(f"LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#{mm})")
        dims = add("DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)")
        length = add(f"(CONVERSION_BASED_UNIT('INCH',#{factor})LENGTH_UNIT()NAMED_UNIT(#{dims}))")
    else:
        length = add(unit_entity[units])
    angle = add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    solid = add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    uncertainty = add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{length},"
                      "'distance_accuracy_value','confusion accuracy')")
    context = add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty}))"
                  f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid}))"
                  "REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY'))")
    origin = add("CARTESIAN_POINT('',(0.,0.,0.))")
    z_axis = add("DIRECTION('',(0.,0.,1.))")
    x_axis = add("DIRECTION('',(1.,0.,0.))")
    placement = add(f"AXIS2_PLACEMENT_3D('',#{origin},#{z_axis},#{x_axis})")

    curve_ids = []
    for i, curve in enumerate(as_curves(curves)):
        knots, ctrl, degree = _knots_controls(curve, scale)
        point_ids = [add("CARTESIAN_POINT('',(" + ",".join(_step_real(v) for v in p) + "))") for p in ctrl]
        values, counts = _distinct(knots)
        curve_ids.append(add(
            f"B_SPLINE_CURVE_WITH_KNOTS({_step_string(f'{name} {i + 1}')},{degree},("
            + ",".join(f"#{p}" for p in point_ids)
            + f"),.UNSPECIFIED.,{'.T.' if _closed(ctrl) else '.F.'},.F.,("
            + ",".join(str(c) for c in counts) + "),("
            + ",".join(_step_real(v) for v in values) + "),.UNSPECIFIED.)"))
    curve_set = add("GEOMETRIC_CURVE_SET('curves',(" + ",".join(f"#{c}" for c in curve_ids) + "))")
    representation = add(f"GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION({_step_string(name)},"
                         f"(#{curve_set},#{placement}),#{context})")
    add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{representation})")

    stamp = _dt.datetime.now().replace(microsecond=0).isoformat()
    lines = [
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('B-spline curves'),'2;1');",
        f"FILE_NAME({_step_string(name)},'{stamp}',(''),(''),'bspline-varpro','bspline-varpro','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));",
        "ENDSEC;",
        "DATA;",
    ]
    lines += [f"#{i}={text};" for i, text in enumerate(entities, start=1)]
    lines += ["ENDSEC;", "END-ISO-10303-21;", ""]
    return "\n".join(lines)


def _ascii(text: str) -> str:
    """Printable ASCII approximation of ``text`` (accents dropped, other characters as ``?``)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in decomposed if not unicodedata.combining(c))


def _hollerith(text: str, limit: int = 60) -> str:
    """IGES Hollerith string, ASCII and at most ``limit`` characters so it fits one record.

    An empty string is an empty field (the default value), not a zero-length ``0H``.
    """
    text = _ascii(text)[:limit]
    return f"{len(text)}H{text}" if text else ""


def _iges_real(v: float) -> str:
    # 17 significant digits read back as the same double
    return f"{_finite(v):.16E}"


def _iges_lines(fields: Sequence[str], letter: str, start: int, width: int = 72,
                pointer: str = "") -> List[str]:
    """Pack delimited fields into fixed-width records without splitting a field."""
    lines, current = [], ""
    limit = 64 if letter == "P" else width
    for field in fields:
        if current and len(current) + len(field) > limit:
            lines.append(current)
            current = ""
        current += field
    if current:
        lines.append(current)
    out = []
    for i, text in enumerate(lines):
        if letter == "P":
            out.append(f"{text:<64}{pointer:>8}{letter}{start + i:7d}")
        else:
            out.append(f"{text:<72}{letter}{start + i:7d}")
    return out


def to_iges(curves: CurveInput, name: str = "curves", scale: float = 1.0, units: str = "mm") -> str:
    """IGES 5.3 text with one entity 126 (B-spline curve) per curve."""
    unit_flags = {"mm": (2, "MM"), "m": (6, "M"), "inch": (1, "IN")}
    if units not in unit_flags:
        raise ValueError("units must be 'mm', 'm', or 'inch'.")
    flag, unit_name = unit_flags[units]
    curve_list = as_curves(curves)
    data = [_knots_controls(c, scale) for c in curve_list]
    extent = max(float(np.max(np.abs(ctrl))) for _, ctrl, _ in data) or 1.0
    stamp = _dt.datetime.now().strftime("%Y%m%d.%H%M%S")

    start = [f"{'B-spline curves written by bspline-varpro':<72}S{1:7d}"]
    global_fields = [
        "1H,", "1H;", _hollerith(name), _hollerith(name + ".igs"), _hollerith("bspline-varpro"),
        _hollerith("bspline-varpro"), "32", "38", "6", "308", "15", _hollerith(name), "1.0",
        str(flag), _hollerith(unit_name), "1", "1.0", _hollerith(stamp), "1.0E-07",
        _iges_real(extent), "", "", "11", "0", _hollerith(stamp),  # author, organization: default
    ]
    global_fields = [f + "," for f in global_fields[:-1]] + [global_fields[-1] + ";"]
    global_lines = _iges_lines(global_fields, "G", 1)

    directory, parameter = [], []
    for i, (knots, ctrl, degree) in enumerate(data):
        de_seq = 2 * i + 1
        k_upper = len(ctrl) - 1
        planar = int(np.allclose(ctrl[:, 2], ctrl[0, 2]))
        fields = ["126", str(k_upper), str(degree), str(planar), str(int(_closed(ctrl))), "1", "0"]
        fields += [_iges_real(v) for v in knots]
        fields += ["1.0"] * len(ctrl)
        fields += [_iges_real(v) for p in ctrl for v in p]
        fields += [_iges_real(knots[degree]), _iges_real(knots[-degree - 1])]
        fields += ["0.0", "0.0", "1.0"] if planar else ["0.0", "0.0", "0.0"]
        fields = [f + "," for f in fields[:-1]] + [fields[-1] + ";"]
        p_start = len(parameter) + 1
        lines = _iges_lines(fields, "P", p_start, pointer=str(de_seq))
        parameter += lines
        label = f"SPL{i + 1}"[:8]
        directory.append(f"{126:>8}{p_start:>8}{0:>8}{0:>8}{0:>8}{0:>8}{0:>8}{0:>8}{'00000000':>8}D{de_seq:7d}")
        directory.append(f"{126:>8}{0:>8}{0:>8}{len(lines):>8}{0:>8}{'':>8}{'':>8}{label:>8}{0:>8}D{de_seq + 1:7d}")
    terminate = (f"S{len(start):7d}G{len(global_lines):7d}D{len(directory):7d}P{len(parameter):7d}"
                 f"{'':<40}T{1:7d}")
    return "\n".join(start + global_lines + directory + parameter + [terminate]) + "\n"


_DXF_FORBIDDEN = set('<>/\\":;?*|=`')


def _dxf_name(text: str) -> str:
    r"""Symbol-table name for an ASCII DXF: non-ASCII characters as ``\U+XXXX``.

    Characters outside the Basic Multilingual Plane, which the escape cannot
    hold, become ``_``.
    """
    if not text or set(text) & _DXF_FORBIDDEN or any(ord(c) < 32 for c in text):
        raise ValueError(f"Invalid DXF layer name {text!r}: it must not be empty or contain "
                         "control characters or any of < > / \\ \" : ; ? * | = `.")
    return "".join(c if ord(c) < 127 else f"\\U+{ord(c):04X}" if ord(c) <= 0xFFFF else "_"
                   for c in text)


def to_dxf(curves: CurveInput, scale: float = 1.0, layer: str = "0", units: str = "mm") -> str:
    """DXF R2000 text with one ``SPLINE`` entity per curve.

    AutoCAD reads only minimal DXF R12 files; R2000 (the first version with
    ``SPLINE``) needs the symbol tables, the model and paper space blocks, the
    root dictionary and owner handles, so the file has all of them.

    Args:
        layer: Layer of the splines; non-ASCII characters are written as
            ``\\U+XXXX`` escapes, as AutoCAD does in R2000 files.
        units: ``"mm"``, ``"m"``, or ``"inch"``, written as ``$INSUNITS``.
    """
    insunits = {"mm": 4, "m": 6, "inch": 1}
    if units not in insunits:
        raise ValueError("units must be 'mm', 'm', or 'inch'.")
    layer = _dxf_name(layer)
    out: List[str] = []
    handles = iter(range(1, 1 << 30))

    def pair(code: int, value) -> None:
        out.append(f"{code:>3}")
        out.append(str(value))

    def new_handle() -> str:
        return f"{next(handles):X}"

    def record(kind: str, owner: str, subclass: str, name: str, *extra, handle_code: int = 5) -> str:
        h = new_handle()
        pair(0, kind); pair(handle_code, h); pair(330, owner)
        pair(100, "AcDbSymbolTableRecord"); pair(100, subclass)
        pair(2, name); pair(70, 0)
        for code, value in extra:
            pair(code, value)
        return h

    def table(name: str, entries, subclass: str = "") -> List[str]:
        """Symbol table whose records ``entries(table_handle)`` writes; their handles."""
        h = new_handle()
        pair(0, "TABLE"); pair(2, name); pair(5, h); pair(330, 0); pair(100, "AcDbSymbolTable")
        count_at = len(out) + 1
        pair(70, 0)
        if subclass:
            pair(100, subclass); pair(71, 0)
        written = entries(h)
        out[count_at] = str(len(written))
        pair(0, "ENDTAB")
        return written

    pair(0, "SECTION"); pair(2, "CLASSES"); pair(0, "ENDSEC")
    pair(0, "SECTION"); pair(2, "TABLES")
    table("VPORT", lambda t: [])
    table("LTYPE", lambda t: [
        record("LTYPE", t, "AcDbLinetypeTableRecord", name, (3, text), (72, 65), (73, 0), (40, 0.0))
        for name, text in (("ByBlock", ""), ("ByLayer", ""), ("Continuous", "Solid line"))])
    table("LAYER", lambda t: [
        record("LAYER", t, "AcDbLayerTableRecord", name, (62, 7), (6, "Continuous"), (370, -3))
        for name in dict.fromkeys(["0", layer])])
    table("STYLE", lambda t: [record("STYLE", t, "AcDbTextStyleTableRecord", "Standard", (40, 0.0),
                                     (41, 1.0), (50, 0.0), (71, 0), (42, 2.5), (3, "txt"), (4, ""))])
    table("VIEW", lambda t: [])
    table("UCS", lambda t: [])
    table("APPID", lambda t: [record("APPID", t, "AcDbRegAppTableRecord", "ACAD")])
    table("DIMSTYLE", lambda t: [record("DIMSTYLE", t, "AcDbDimStyleTableRecord", "Standard",
                                        handle_code=105)], subclass="AcDbDimStyleTable")
    model, paper = table("BLOCK_RECORD", lambda t: [
        record("BLOCK_RECORD", t, "AcDbBlockTableRecord", name) for name in ("*Model_Space", "*Paper_Space")])
    pair(0, "ENDSEC")

    pair(0, "SECTION"); pair(2, "BLOCKS")
    for owner, name, paper_space in ((model, "*Model_Space", False), (paper, "*Paper_Space", True)):
        pair(0, "BLOCK"); pair(5, new_handle()); pair(330, owner); pair(100, "AcDbEntity")
        if paper_space:
            pair(67, 1)
        pair(8, "0"); pair(100, "AcDbBlockBegin"); pair(2, name); pair(70, 0)
        pair(10, 0.0); pair(20, 0.0); pair(30, 0.0); pair(3, name); pair(1, "")
        pair(0, "ENDBLK"); pair(5, new_handle()); pair(330, owner); pair(100, "AcDbEntity")
        if paper_space:
            pair(67, 1)
        pair(8, "0"); pair(100, "AcDbBlockEnd")
    pair(0, "ENDSEC")

    pair(0, "SECTION"); pair(2, "ENTITIES")
    for curve in as_curves(curves):
        knots, ctrl, degree = _knots_controls(curve, scale)
        planar = bool(np.allclose(ctrl[:, 2], ctrl[0, 2]))
        pair(0, "SPLINE")
        pair(5, new_handle())
        pair(330, model)
        pair(100, "AcDbEntity")
        pair(8, layer)
        pair(100, "AcDbSpline")
        if planar:
            pair(210, 0.0); pair(220, 0.0); pair(230, 1.0)
        pair(70, 8 if planar else 0)
        pair(71, degree)
        pair(72, len(knots))
        pair(73, len(ctrl))
        pair(74, 0)
        pair(42, "1e-10")
        pair(43, "1e-10")
        for k in knots:
            pair(40, repr(_finite(k)))
        for p in ctrl:
            pair(10, repr(_finite(p[0]))); pair(20, repr(_finite(p[1]))); pair(30, repr(_finite(p[2])))
    pair(0, "ENDSEC")

    pair(0, "SECTION"); pair(2, "OBJECTS")
    root, group = new_handle(), new_handle()
    pair(0, "DICTIONARY"); pair(5, root); pair(330, 0); pair(100, "AcDbDictionary"); pair(281, 1)
    pair(3, "ACAD_GROUP"); pair(350, group)
    pair(0, "DICTIONARY"); pair(5, group); pair(330, root); pair(100, "AcDbDictionary"); pair(281, 1)
    pair(0, "ENDSEC")
    pair(0, "EOF")

    body = out
    out = []
    pair(0, "SECTION"); pair(2, "HEADER")
    pair(9, "$ACADVER"); pair(1, "AC1015")
    pair(9, "$HANDSEED"); pair(5, new_handle())
    pair(9, "$INSUNITS"); pair(70, insunits[units])
    pair(0, "ENDSEC")
    return "\n".join(out + body) + "\n"


def write(path: Union[str, Path], curves: CurveInput, **kwargs) -> Path:
    """Write curves to ``.step``/``.stp``, ``.iges``/``.igs``, or ``.dxf`` by suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".step", ".stp"):
        text = to_step(curves, name=kwargs.pop("name", path.stem), **kwargs)
    elif suffix in (".iges", ".igs"):
        text = to_iges(curves, name=kwargs.pop("name", path.stem), **kwargs)
    elif suffix == ".dxf":
        kwargs.pop("name", None)
        text = to_dxf(curves, **kwargs)
    else:
        raise ValueError(f"Unsupported export format {suffix!r}; use .step, .iges, or .dxf.")
    path.write_text(text, encoding="ascii", newline="\n")
    return path


def _arc_table(curve: BSpline, samples: int = 4096):
    k = curve.k
    u = np.linspace(curve.t[k], curve.t[-k - 1], samples)
    speed = np.linalg.norm(curve(u, nu=1), axis=-1)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u))])
    return u, s


def sample(curve: BSpline, n: int, spacing: str = "cosine") -> np.ndarray:
    """``n`` points exactly on a curve.

    Args:
        spacing: ``"cosine"`` (arc length, clustered at both ends), ``"arc"``
            (uniform arc length), ``"half-cosine-start"`` (clustered at the
            start only), or ``"parameter"`` (uniform parameter).
    """
    if n < 2:
        raise ValueError("Need at least two samples.")
    k = curve.k
    lo, hi = curve.t[k], curve.t[-k - 1]
    if spacing == "parameter":
        return curve(np.linspace(lo, hi, n))
    fractions = {
        "cosine": 0.5 * (1 - np.cos(np.linspace(0, np.pi, n))),
        "arc": np.linspace(0, 1, n),
        "half-cosine-start": 1 - np.cos(np.linspace(0, np.pi / 2, n)),
    }
    if spacing not in fractions:
        raise ValueError("spacing must be 'cosine', 'arc', 'half-cosine-start', or 'parameter'.")
    u_table, s_table = _arc_table(curve)
    u = np.interp(fractions[spacing] * s_table[-1], s_table, u_table)
    u[0], u[-1] = lo, hi
    return curve(u)
