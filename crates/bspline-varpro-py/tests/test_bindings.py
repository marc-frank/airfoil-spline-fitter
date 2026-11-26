"""The Python module: `maturin develop` in crates/bspline-varpro-py, then pytest."""
import math

import pytest

import bspline_varpro as bv


@pytest.fixture(scope="module")
def naca2412():
    return [(round(x, 6), round(y, 6)) for x, y in bv.naca4("2412", 81)]


def test_naca_points_in_selig_order(naca2412):
    assert len(naca2412) == 161
    assert naca2412[0] == pytest.approx((1.0, 0.0), abs=1e-9)
    assert naca2412[80] == pytest.approx((0.0, 0.0), abs=1e-9)


def test_fit_airfoil_meets_the_tolerance_as_the_app_does(naca2412):
    tolerance = bv.default_tolerance(naca2412, 6)
    fit = bv.fit_airfoil(naca2412, tolerance, decimals=6)
    assert fit.success
    assert fit.max_deviation <= tolerance
    assert fit.measure(naca2412)["max_deviation"] == pytest.approx(fit.max_deviation)
    assert len(fit.control_points) == 2 and fit.degree == 5
    assert fit.num_control_points[0] == fit.num_control_points[1]


def test_measure_reports_thickness_camber_and_junctions(naca2412):
    fit = bv.fit_airfoil(naca2412, 1e-5, decimals=6)
    m = fit.measure()
    assert m["max_thickness"] == pytest.approx(0.12, abs=1e-4)
    assert m["max_camber"] == pytest.approx(0.02, abs=1e-4)
    assert 0.39 < m["max_camber_x"] < 0.42
    assert len(m["junctions"]) == 1 and m["junctions"][0]["gap"] < 1e-12
    assert m["leading_edge_radius"] > 0


def test_fixed_count_fit_with_continuity_per_junction(naca2412):
    sections = bv.split_sections(naca2412, 4)
    assert len(sections) == 4
    fit = bv.fit(sections, [10, 11, 11, 10], continuity=["G2", "C2", "G2"])
    assert fit.num_control_points == [10, 11, 11, 10]
    assert all(len(k) == n + 6 for k, n in zip(fit.knots, fit.num_control_points))
    assert fit.rms_error < 1e-4


def test_export_writes_every_format(naca2412, tmp_path):
    fit = bv.fit_airfoil(naca2412, 1e-5, decimals=6)
    for suffix, marker in ((".step", "B_SPLINE_CURVE_WITH_KNOTS"), (".igs", "126,"), (".dxf", "SPLINE")):
        path = fit.export(str(tmp_path / f"a{suffix}"), scale=100.0, merge=True)
        assert marker in open(path).read()


def test_read_airfoil_keeps_the_file_values(naca2412, tmp_path):
    path = tmp_path / "naca2412.dat"
    path.write_text("NACA 2412\n" + "".join(f"{x:.6f} {y:.6f}\n" for x, y in naca2412))
    name, points, precision = bv.read_airfoil(str(path))
    assert name == "NACA 2412" and precision == 6
    assert [tuple(p) for p in points] == pytest.approx(naca2412)


def wedge(kink=None, step=0.005):
    """Straight sides meeting at a point, with a kink on the lower side."""
    xs = [1.0 - i * step for i in range(round(1 / step) + 1)]
    half = lambda x: 0.1 * min(x, 1 - x) + 0.02 * x
    lower = lambda x: -half(x) - (0.5 * max(x - kink, 0.0) if kink else 0.0)
    return [(x, half(x)) for x in xs] + [(x, lower(x)) for x in reversed(xs[:-1])]


def test_corners_are_read_from_any_points(naca2412):
    assert bv.corners(naca2412) == []
    points = wedge(kink=0.75)
    found = bv.corners(points)
    # The nose and the kink, both clear.
    assert [(i, clear) for i, _, clear in found] == [(200, True), (350, True)]
    assert points[200][0] == pytest.approx(0.0, abs=1e-12) and points[350][0] == pytest.approx(0.75)
    assert found[0][1] > 150 and 20 < found[1][1] < 30
    sections = bv.split_at_corners(points)
    assert [len(s) for s in sections] == [201, 151, 51]
    fit = bv.fit(sections, 8, continuity="G0")
    assert fit.rms_error < 2e-4
    # The same corners with the points turned and scaled.
    turned = [(300 * (0.6 * x - 0.8 * y), 300 * (0.8 * x + 0.6 * y)) for x, y in points]
    assert [i for i, _, _ in bv.corners(turned)] == [200, 350]


def test_fit_airfoil_joins_corners_with_g0():
    points = wedge(kink=0.75)
    fit = bv.fit_airfoil(points, 1e-4)
    assert fit.max_deviation <= 1e-4 and len(fit.control_points) == 3
    # The curve turns at the nose and at the kink as the points do.
    turns = [math.degrees(abs(j["tangent_angle"])) for j in fit.measure()["junctions"]]
    assert turns[0] > 150 and 20 < turns[1] < 30
    # Cut the same way by hand, and by the section count alone.
    assert [len(s) for s in bv.split_sections(points, 2, corners=True)] == [201, 151, 51]
    assert [len(s) for s in bv.split_sections(points, 2)] == [201, 201]
    # Left alone, the kink is no junction.
    assert len(bv.split_sections(points, 2, corners=False)) == 2


def test_few_points_get_company_on_their_smoothest_curve():
    line = bv.smoothest_through([(0.0, 0.0), (1.0, 2.0)], 5)
    assert len(line) == 5
    for i, (x, y) in enumerate(line):
        assert (x, y) == pytest.approx((0.25 * i, 0.5 * i))
    assert len(bv.smoothest_through([(0, 0), (1, 1), (3, 0)], 7)) == 7


@pytest.mark.parametrize("call, message", [
    (lambda s: bv.fit(s, 3), r"degree \+ 1"),
    (lambda s: bv.fit(s, 10, continuity="X9"), "Invalid continuity"),
    (lambda s: bv.fit(s, 10, knots="random"), "uniform"),
])
def test_bad_input_raises_value_error(naca2412, call, message):
    with pytest.raises(ValueError, match=message):
        call(bv.split_sections(naca2412))
