import re

import numpy as np
import pytest
from scipy.interpolate import BSpline

from bspline_varpro import airfoil, diagnostics, export, fit_to_tolerance
from bspline_varpro.airfoil import NACA4


@pytest.fixture(scope="module")
def fitted():
    return airfoil.fit_airfoil(np.round(NACA4("4412", closed_te=False).points(61), 5), 1e-4)


def _step_curves(text):
    entities = dict(re.findall(r"^#(\d+)=(.*);$", text, flags=re.M))
    curves = []
    for body in entities.values():
        if not body.startswith("B_SPLINE_CURVE_WITH_KNOTS"):
            continue
        m = re.match(r"B_SPLINE_CURVE_WITH_KNOTS\('(?:[^']|'')*',(\d+),\(([^)]*)\),\.\w+\.,\.F\.,\.F\.,"
                     r"\(([^)]*)\),\(([^)]*)\),\.UNSPECIFIED\.\)", body)
        degree = int(m.group(1))
        refs = [r.strip("#") for r in m.group(2).split(",")]
        points = [[float(v) for v in re.search(r"\(([^()]*)\)\)$", entities[r]).group(1).split(",")]
                  for r in refs]
        mults = [int(v) for v in m.group(3).split(",")]
        values = [float(v) for v in m.group(4).split(",")]
        curves.append((degree, np.repeat(values, mults), np.array(points)))
    return curves


def _same_curve(curve, degree, knots, points):
    n = len(curve.t) - curve.k - 1
    assert degree == curve.k
    assert np.allclose(knots, curve.t, rtol=0, atol=1e-15)
    assert np.allclose(points[:, :2], np.asarray(curve.c)[:n], rtol=0, atol=1e-15)
    assert np.all(points[:, 2] == 0)


def test_merge_reproduces_every_section_exactly(fitted):
    curves = fitted.curves()
    merged = export.merge_curves(curves)
    assert merged.t[0] == 0 and merged.t[-1] == 1
    assert np.allclose(merged(0.0), curves[0](0.0)) and np.allclose(merged(1.0), curves[-1](1.0))
    for c in curves:
        _, dist = diagnostics.project(c(np.linspace(c.t[c.k], c.t[-c.k - 1], 301)), merged)
        assert np.max(dist) < 1e-11
    assert export.arc_length(merged) == pytest.approx(sum(export.arc_length(c) for c in curves), rel=1e-10)


def test_merged_airfoil_is_c1_at_a_g2_junction(fitted):
    merged = export.merge_curves(fitted.curves())
    values, counts = np.unique(merged.t[merged.k + 1:-merged.k - 1], return_counts=True)
    u = values[counts.argmax()]
    assert counts.max() <= merged.k - 1
    left, right = merged(u - 1e-13, nu=1), merged(u + 1e-13, nu=1)
    assert np.linalg.norm(left - right) < 1e-8 * np.linalg.norm(left)


def test_merge_multiplicities_follow_the_junction_continuity():
    pts = NACA4("2412").points(61)
    result = fit_to_tolerance([pts[:30], pts[29:60], pts[59:90], pts[89:]], 1e-4,
                              continuity=["G1", "C2", "G0"], fairing=False)
    merged = export.merge_curves(result)
    values, counts = np.unique(merged.t, return_counts=True)
    assert np.min(np.diff(values)) > 1e-9
    k = merged.k
    interior_high = sorted(counts[1:-1][counts[1:-1] > 1])
    assert interior_high == [k - 2, k - 1, k]
    for c in result.curves():
        _, dist = diagnostics.project(c(np.linspace(0, 1, 101)), merged)
        assert np.max(dist) < 1e-11


def test_merge_keeps_a_corner():
    a = BSpline(np.r_[[0.0] * 4, [1.0] * 4], np.array([[0, 0], [1, 0], [2, 0], [3, 0.0]]), 3)
    b = BSpline(np.r_[[0.0] * 4, [1.0] * 4], np.array([[3, 0], [3, 1], [3, 2], [3, 3.0]]), 3)
    merged = export.merge_curves([a, b])
    values, counts = np.unique(merged.t, return_counts=True)
    assert list(counts) == [4, 3, 4]
    assert np.allclose(merged(values[1]), [3, 0])
    assert np.allclose(merged(np.linspace(0, values[1], 5)), a(np.linspace(0, 1, 5)))


def test_merge_rejects_mixed_degrees():
    a = BSpline(np.r_[[0.0] * 4, [1.0] * 4], np.zeros((4, 2)), 3)
    b = BSpline(np.r_[[0.0] * 3, [1.0] * 3], np.zeros((3, 2)), 2)
    with pytest.raises(ValueError, match="same degree"):
        export.merge_curves([a, b])


def test_step_stores_the_exact_curves(fitted):
    text = export.to_step(fitted, name="naca'4412")
    assert text.startswith("ISO-10303-21;") and text.rstrip().endswith("END-ISO-10303-21;")
    assert "AUTOMOTIVE_DESIGN" in text and "SI_UNIT(.MILLI.,.METRE.)" in text
    assert "'naca''4412'" in text
    parsed = _step_curves(text)
    assert len(parsed) == 2
    for curve, (degree, knots, points) in zip(fitted.curves(), parsed):
        _same_curve(curve, degree, knots, points)
    for real in re.findall(r"CARTESIAN_POINT\('',\(([^)]*)\)\)", text):
        assert all("." in v for v in real.split(","))


def test_step_scale_and_units(fitted):
    text = export.to_step(fitted, scale=1000.0, units="inch")
    assert "CONVERSION_BASED_UNIT('INCH'" in text
    (_, _, points), *_ = _step_curves(text)
    assert np.allclose(points[:, :2], np.asarray(fitted.control_points[0]) * 1000.0)
    with pytest.raises(ValueError, match="units"):
        export.to_step(fitted, units="furlong")


def _iges_parameters(text):
    lines = text.splitlines()
    assert all(len(line) == 80 for line in lines)
    letters = [line[72] for line in lines]
    assert letters == sorted(letters, key="SGDPT".index)
    for letter in "SGDP":
        seq = [int(line[73:]) for line in lines if line[72] == letter]
        assert seq == list(range(1, len(seq) + 1))
    counts = dict(re.findall(r"([SGDP])\s*(\d+)", lines[-1][:32]))
    for letter in "SGDP":
        assert int(counts[letter]) == letters.count(letter)
    directory = [line for line in lines if line[72] == "D"]
    params = [line for line in lines if line[72] == "P"]
    entities = []
    for first in directory[::2]:
        start = int(first[8:16])
        de = int(first[73:])
        body = "".join(line[:64] for line in params if int(line[64:72]) == de)
        assert int(params[start - 1][64:72]) == de
        entities.append([v.strip() for v in body.strip().rstrip(";").split(",")])
    return entities


def test_iges_entity_126_matches(fitted):
    text = export.to_iges(fitted)
    entities = _iges_parameters(text)
    assert len(entities) == 2
    for curve, fields in zip(fitted.curves(), entities):
        assert fields[0] == "126"
        upper, degree = int(fields[1]), int(fields[2])
        n = upper + 1
        assert degree == curve.k and n == len(curve.t) - curve.k - 1
        assert fields[3:7] == ["1", "0", "1", "0"]
        values = np.array([float(v) for v in fields[7:]])
        knots = values[:n + degree + 1]
        weights = values[n + degree + 1:2 * n + degree + 1]
        points = values[2 * n + degree + 1:5 * n + degree + 1].reshape(n, 3)
        assert np.all(weights == 1)
        _same_curve(curve, degree, knots, points)
        assert list(values[-5:]) == [curve.t[curve.k], curve.t[-curve.k - 1], 0.0, 0.0, 1.0]


def _dxf_pairs(text):
    lines = text.splitlines()
    return [(int(lines[i]), lines[i + 1]) for i in range(0, len(lines) - 1, 2)]


def test_dxf_spline_matches(fitted):
    pairs = _dxf_pairs(export.to_dxf(fitted))
    assert ("1", "AC1015") in [(str(c), v) for c, v in pairs]
    assert pairs[-1] == (0, "EOF")
    starts = [i for i, p in enumerate(pairs) if p == (0, "SPLINE")]
    assert len(starts) == 2
    for curve, start, end in zip(fitted.curves(), starts, starts[1:] + [len(pairs)]):
        group = pairs[start:end]
        get = lambda code: [v for c, v in group if c == code]  # noqa: E731
        degree = int(get(71)[0])
        knots = np.array(get(40), dtype=float)
        points = np.column_stack([np.array(get(c), dtype=float) for c in (10, 20, 30)])
        assert int(get(72)[0]) == len(knots) and int(get(73)[0]) == len(points)
        assert int(get(70)[0]) == 8
        _same_curve(curve, degree, knots, points)


def test_write_by_suffix(tmp_path, fitted):
    for suffix, marker in [(".stp", "ISO-10303-21"), (".step", "ISO-10303-21"),
                           (".igs", "126"), (".iges", "126"), (".dxf", "SPLINE")]:
        path = export.write(tmp_path / f"part{suffix}", fitted, units="mm")
        assert marker in path.read_text()
    with pytest.raises(ValueError, match="Unsupported"):
        export.write(tmp_path / "part.obj", fitted)


def test_sample_spacings():
    curve = BSpline(np.r_[[0.0] * 4, [1.0] * 4], np.array([[0, 0], [1, 2], [2, 2], [3, 0.0]]), 3)
    for spacing in ("cosine", "arc", "half-cosine-start", "parameter"):
        pts = export.sample(curve, 41, spacing)
        assert pts.shape == (41, 2)
        assert np.allclose(pts[0], [0, 0]) and np.allclose(pts[-1], [3, 0])
    steps = np.linalg.norm(np.diff(export.sample(curve, 201, "arc"), axis=0), axis=1)
    assert np.ptp(steps) < 1e-3 * steps.mean()
    cos_steps = np.linalg.norm(np.diff(export.sample(curve, 41, "cosine"), axis=0), axis=1)
    assert cos_steps[0] < 0.1 * cos_steps[20] and cos_steps[-1] < 0.1 * cos_steps[20]
    with pytest.raises(ValueError, match="spacing"):
        export.sample(curve, 10, "random")


def test_arc_length_of_a_line():
    line = BSpline(np.r_[[0.0] * 6, 0.3, [1.0] * 6], np.linspace([0, 0], [3, 4], 7), 5)
    assert export.arc_length(line) == pytest.approx(5.0, rel=1e-12)


@pytest.mark.parametrize("units, code", [("mm", "4"), ("m", "6"), ("inch", "1")])
def test_dxf_records_the_requested_units(tmp_path, fitted, units, code):
    for text in (export.to_dxf(fitted, units=units),
                 export.write(tmp_path / "part.dxf", fitted, units=units).read_text()):
        pairs = _dxf_pairs(text)
        assert pairs[pairs.index((9, "$INSUNITS")) + 1] == (70, code)
    with pytest.raises(ValueError, match="units"):
        export.to_dxf(fitted, units="furlong")


def test_step_escapes_quotes_backslashes_and_non_ascii_names(tmp_path, fitted):
    name = "Göttingen 387 \\ it's \U0001d4d0"
    text = export.write(tmp_path / "part.step", fitted, name=name).read_text(encoding="ascii")
    escaped = "'G\\X2\\00F6\\X0\\ttingen 387 \\\\ it''s \\X4\\0001D4D0\\X0\\'"
    assert f"PRODUCT({escaped},{escaped}," in text
    assert len(_step_curves(text)) == 2


def test_iges_names_are_ascii_and_fit_one_record(tmp_path, fitted):
    name = "Göttingen " + "x" * 100
    text = export.write(tmp_path / "part.igs", fitted, name=name).read_text(encoding="ascii")
    assert len(_iges_parameters(text)) == 2
    global_section = "".join(line[:72] for line in text.splitlines() if line[72] == "G")
    assert "60HGottingen xxx" in global_section
