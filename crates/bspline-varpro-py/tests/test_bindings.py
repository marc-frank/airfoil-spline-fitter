"""The Python module: `maturin develop` in crates/bspline-varpro-py, then pytest."""
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


@pytest.mark.parametrize("call, message", [
    (lambda s: bv.fit(s, 3), r"degree \+ 1"),
    (lambda s: bv.fit(s, 10, continuity="X9"), "Invalid continuity"),
    (lambda s: bv.fit(s, 10, knots="random"), "uniform"),
])
def test_bad_input_raises_value_error(naca2412, call, message):
    with pytest.raises(ValueError, match=message):
        call(bv.split_sections(naca2412))
