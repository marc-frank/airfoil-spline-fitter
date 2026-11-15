"""Regression tests for .dat parsing, project files, section splitting, formatting and scripts."""
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from bspline_varpro.airfoil import NACA4

from spline_fitter import formatting
from spline_fitter.analysis import default_tolerance
from spline_fitter.geometry import create_airfoil_data, split_into_sections
from spline_fitter.io import (
    export_airfoil_data,
    export_spline_data,
    get_precision,
    parse_airfoil_data,
    parse_airfoil_text,
    parse_spline_data,
)
from spline_fitter.optimization.models import (
    ContinuityMode,
    FitResult,
    OptimizationConfig,
    OptimizerType,
    SplineConfig,
)
from spline_fitter.project_io import (
    create_optimization_run,
    create_project,
    load_project,
    save_project,
)

ROOT = Path(__file__).resolve().parents[1]
PTS = np.round(NACA4("2412").points(31), 6)
ROWS = [f"{x:.6f} {y:.6f}" for x, y in PTS]


def _parse(text):
    _, name, coords, precision = parse_airfoil_text(text, "file")
    return name, None if coords is None else np.array(coords), precision


def _same_contour(coords):
    return coords is not None and coords.shape == PTS.shape and np.allclose(coords, PTS, atol=1e-6)


# --- .dat parsing ----------------------------------------------------------

@pytest.mark.parametrize("rows", [
    [r.replace(" ", ",") for r in ROWS],
    [r.replace(" ", ", ") for r in ROWS],
    [r.replace(" ", ";") for r in ROWS],
    [r.replace(" ", "\t") for r in ROWS],
    [r.replace(".", ",") for r in ROWS],
    [f"{x:.6E} {y:.6E}".replace("E", "D") for x, y in PTS],
    [f'"{x:.6f}","{y:.6f}"' for x, y in PTS],
    [f'"{x:.6f}";"{y:.6f}"'.replace(".", ",") for x, y in PTS],
    [f"{x:9.6f}{y:9.6f}" for x, y in PTS],
    [f"{x:9.6f}{y:9.6f}".replace(".", ",") for x, y in PTS],
    [f"{x:.6e}{y:+.6e}" for x, y in PTS],
], ids=["comma", "comma space", "semicolon", "tab", "decimal comma", "fortran exponent",
        "quoted csv", "quoted decimal comma", "fixed width", "fixed width decimal comma",
        "fixed width e-notation"])
def test_delimiter_variants_load(rows):
    name, coords, _ = _parse("NACA 2412\n" + "\n".join(rows))
    assert name == "NACA 2412" and _same_contour(coords)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-16-le", "cp1252"])
@pytest.mark.parametrize("with_name", [True, False])
def test_encodings_keep_every_point_and_the_name(tmp_path, encoding, with_name):
    path = tmp_path / "foil.dat"
    header = ["Profil été"] if with_name else []
    path.write_bytes("\n".join(header + ROWS).encode(encoding))
    _, name, coords, precision = parse_airfoil_data(str(path))
    assert name == ("Profil été" if with_name else "foil")
    assert _same_contour(np.array(coords)) and precision == 6


def test_precision_is_at_unit_chord():
    scaled = "\n".join(f"{100 * x:.4f} {100 * y:.4f}" for x, y in PTS)
    assert _parse("X\n" + scaled)[2] == 6
    mm = "\n".join(f"{200 * x:.3f} {200 * y:.3f}" for x, y in PTS)
    assert _parse("X\n" + mm)[2] == 5  # 0.001 mm at 200 mm chord is 5e-6
    assert default_tolerance(PTS, 6) < 2e-5  # what the 0-100 file now gets


def test_precision_of_e_notation():
    assert get_precision("1.234560E-03") == 9
    assert get_precision("1.23e+01") == 1
    assert get_precision("0.0123") == 4
    # The largest numbers are rounded coarsest
    sci = "\n".join(f"{x:.6E} {y:.6E}" for x, y in PTS)
    assert _parse("X\n" + sci)[2] == 6
    percent = "\n".join(f"{100 * x:.5E} {100 * y:.5E}" for x, y in PTS)
    assert _parse("X\n" + percent)[2] == 5  # 1.00000E+02: 3 decimals, 5 at unit chord


def test_selig_with_whole_number_first_point_is_not_lednicer():
    rows = [f"{200 * x:.3f} {200 * y:.3f}" for x, y in PTS]
    rows[0], rows[-1] = "200 2", "200 -2"  # blunt trailing edge in mm
    _, coords, _ = _parse("X\n" + "\n".join(rows))
    assert len(coords) == len(PTS)
    assert coords[:, 0].min() == pytest.approx(0) and coords[:, 0].max() == pytest.approx(1)


@pytest.mark.parametrize("text", [
    "inf 5\n1 0\n0.5 0.06\n0 0\n0.5 -0.06\n1 0\n",
    "inf 5\n1 0\n0.5 0.06\nnan nan\n0 0\n1e400 0\n0.5 -0.06\n1 0\n",
], ids=["inf first line", "nan and overflow rows"])
def test_non_finite_numbers_are_not_coordinates(text):
    name, coords, precision = _parse(text)
    assert name == "inf 5" and precision == 2
    assert coords.tolist() == [[1, 0], [0.5, 0.06], [0, 0], [0.5, -0.06], [1, 0]]


@pytest.mark.parametrize("name", ["2412 15", "0012 64"])
def test_numeric_name_is_a_name(tmp_path, name):
    assert _parse(name + "\n" + "\n".join(ROWS))[0] == name
    path = tmp_path / "x.dat"
    export_airfoil_data(str(path), [tuple(p) for p in PTS], name=name, precision=6)
    _, back, coords, _ = parse_airfoil_data(str(path))
    assert back == name and _same_contour(np.array(coords))


def test_a_name_with_a_hash_is_kept_whole(tmp_path):
    assert _parse("NACA 0012 #2\n" + "\n".join(ROWS))[0] == "NACA 0012 #2"
    assert _parse("# comment line\nNACA 0012\n" + "\n".join(ROWS))[0] == "NACA 0012"
    path = tmp_path / "x.dat"
    export_airfoil_data(str(path), [tuple(p) for p in PTS], name="E387 #2 (smoothed)", precision=6)
    _, back, coords, _ = parse_airfoil_data(str(path))
    assert back == "E387 #2 (smoothed)" and _same_contour(np.array(coords))


@pytest.mark.parametrize("name", ["#2 NACA 0012", "# 2412", "\\#odd", "\\name", "NACA \\#3"])
def test_a_name_that_starts_with_a_hash_round_trips(tmp_path, name):
    ctrl = [[(1.0, 0.0), (0.5, 0.1), (0.0, 0.0)], [(0.0, 0.0), (0.5, -0.1), (1.0, 0.0)]]
    path = tmp_path / "x.dat"
    export_airfoil_data(str(path), [tuple(p) for p in PTS], name=name, precision=6)
    _, back, coords, _ = parse_airfoil_data(str(path))
    assert back == name and _same_contour(np.array(coords))
    spline = tmp_path / "x.deg2.spline.dat"
    export_spline_data(spline, 2, ctrl, name=name)
    data = parse_spline_data(spline)
    assert data["name"] == name and data["section_control_points"] == ctrl


def test_a_hash_comment_first_line_of_another_tool_is_still_a_comment():
    assert _parse("# written by xfoil\nNACA 0012\n" + "\n".join(ROWS))[0] == "NACA 0012"
    assert _parse("# NACA 0012\n" + "\n".join(ROWS))[0] == "file"
    assert _parse("\\#3 NACA 0012\n" + "\n".join(ROWS))[0] == "#3 NACA 0012"


def test_spline_file_numeric_name_and_leading_blank_line(tmp_path):
    ctrl = [[(1.0, 0.0), (0.5, 0.1), (0.0, 0.0)], [(0.0, 0.0), (0.5, -0.1), (1.0, 0.0)]]
    path = tmp_path / "a.deg2.spline.dat"
    export_spline_data(path, 2, ctrl, name="0012 64")
    data = parse_spline_data(path)
    assert data["name"] == "0012 64" and data["section_control_points"] == ctrl
    path.write_text("\n" + path.read_text().replace("0012 64", "NACA 0012"))
    assert parse_spline_data(path)["name"] == "NACA 0012"


# --- project files ---------------------------------------------------------

def _project(report=None, **options):
    airfoil = create_airfoil_data([tuple(p) for p in PTS], name="N", precision=6)
    result = FitResult([[(1.0, 0.0), (0.0, 0.1), (0.0, 0.0)], [(0.0, 0.0), (0.0, -0.1), (1.0, 0.0)]],
                       2, 1e-5, 3, 4, 0.5, report=report)
    project = create_project(airfoil)
    project.optimization_runs.append(create_optimization_run(
        SplineConfig(degree=2, num_control_points=3, continuity=ContinuityMode.G1,
                     geometric_continuity=False), OptimizationConfig(**options), result, []))
    return project


def test_failed_save_keeps_the_previous_project(tmp_path):
    path = tmp_path / "p.asfproj"
    save_project(_project(), str(path))
    before = path.read_bytes()
    with pytest.raises(TypeError):
        save_project(_project(report={"bad": object()}), str(path))
    assert path.read_bytes() == before and load_project(str(path)).optimization_runs
    assert [p.name for p in tmp_path.iterdir()] == ["p.asfproj"]  # no temporary file left


def test_numpy_values_in_a_report_save(tmp_path):
    path = tmp_path / "p.asfproj"
    save_project(_project(report={"curvature_extrema": np.int64(3), "gap": np.float32(0.5),
                                  "list": np.arange(3)}), str(path))
    report = load_project(str(path)).optimization_runs[0].result.report
    assert report == {"curvature_extrema": 3, "gap": 0.5, "list": [0, 1, 2]}


def test_custom_parameters_round_trip(tmp_path):
    path = tmp_path / "p.asfproj"
    u = [np.linspace(0, 1, 5), np.linspace(0, 1, 4)]
    save_project(_project(custom_parameters=u), str(path))
    back = load_project(str(path)).optimization_runs[0].optimization_config.custom_parameters
    assert all(np.array_equal(a, b) for a, b in zip(back, u, strict=True))


def test_unknown_optimizer_warns_and_bad_version_is_a_value_error(tmp_path, caplog):
    path = tmp_path / "p.asfproj"
    save_project(_project(), str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["optimization_runs"][0]["optimization_config"]["optimizer"] = "future-solver"
    path.write_text(json.dumps(data), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="spline_fitter.project_io"):
        run = load_project(str(path)).optimization_runs[0]
    assert run.optimization_config.optimizer == OptimizerType.GN_VARPRO
    assert "future-solver" in caplog.text
    data["format_version"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    load_project(str(path))
    data["format_version"] = "abc"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="not valid"):
        load_project(str(path))


def test_non_finite_numbers_are_standard_json_and_round_trip(tmp_path):
    path = tmp_path / "p.asfproj"
    project = _project(report={"leading_edge_radius": np.inf, "nested": [{"v": -np.inf}]})
    project.optimization_runs[0].result.final_error = float("nan")
    save_project(project, str(path))
    text = path.read_text(encoding="utf-8")

    def reject(token):
        raise ValueError(f"non-standard token {token}")

    data = json.loads(text, parse_constant=reject)  # a strict JSON reader accepts it
    result = data["optimization_runs"][0]["result"]
    assert result["final_error"] == "NaN" and result["report"]["leading_edge_radius"] == "Infinity"
    assert result["report"]["nested"] == [{"v": "-Infinity"}]
    back = load_project(str(path)).optimization_runs[0].result
    assert np.isnan(back.final_error) and back.report["leading_edge_radius"] == np.inf
    assert back.report["nested"] == [{"v": -np.inf}]
    # A file of an earlier version, with bare tokens, loads to the same values
    path.write_text(json.dumps(data).replace('"NaN"', "NaN").replace('"Infinity"', "Infinity")
                    .replace('"-Infinity"', "-Infinity"), encoding="utf-8")
    old = load_project(str(path)).optimization_runs[0].result
    assert np.isnan(old.final_error) and old.report["leading_edge_radius"] == np.inf


def test_written_files_have_unix_line_ends(tmp_path):
    ctrl = [[(1.0, 0.0), (0.5, 0.1), (0.0, 0.0)], [(0.0, 0.0), (0.5, -0.1), (1.0, 0.0)]]
    export_airfoil_data(str(tmp_path / "a.dat"), [tuple(p) for p in PTS], name="N")
    export_spline_data(str(tmp_path / "a.deg2.spline.dat"), 2, ctrl, name="N")
    save_project(_project(), str(tmp_path / "p.asfproj"))
    for name in ("a.dat", "a.deg2.spline.dat", "p.asfproj"):
        content = (tmp_path / name).read_bytes()
        assert b"\n" in content and b"\r" not in content, name


def test_stored_sections_and_degree_are_rebuilt_not_read(tmp_path):
    path = tmp_path / "p.asfproj"
    save_project(_project(), str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    junctions = data["airfoil"]["junction_indices"]
    data["airfoil"]["sections"] = [[[9.0, 9.0]]]
    data["airfoil"]["num_sections"] = 7
    data["optimization_runs"][0]["result"]["degree"] = 9
    data["metadata"]["format_version"] = "0.1"
    path.write_text(json.dumps(data), encoding="utf-8")
    project = load_project(str(path))
    expected = create_airfoil_data([tuple(p) for p in PTS], name="N", precision=6)
    assert project.airfoil.sections == expected.sections and project.airfoil.num_sections == 2
    assert list(project.airfoil.junction_indices) == junctions
    assert project.optimization_runs[0].result.degree == 2
    assert project.metadata.format_version == "1.0.0"


@pytest.mark.parametrize("junctions", [[0], [20, 10], [len(PTS) - 1], [5, 5]])
def test_junction_indices_that_do_not_split_the_coordinates_are_an_error(tmp_path, junctions):
    path = tmp_path / "p.asfproj"
    save_project(_project(), str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["airfoil"]["junction_indices"] = junctions
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(
            "Airfoil: the junction indices do not split the coordinates.")):
        load_project(str(path))


# --- geometry, formatting, package ----------------------------------------

@pytest.mark.parametrize("n_sections", range(1, len(PTS)))
def test_every_section_has_a_segment(n_sections):
    points = [tuple(p) for p in PTS]
    sections, junctions = split_into_sections(points, n_sections)
    assert len(sections) == n_sections and min(len(s) for s in sections) >= 2
    assert list(junctions) == sorted(set(junctions))
    assert sum(len(s) for s in sections) - (n_sections - 1) == len(points)


def test_too_few_points_on_one_side_is_an_error():
    points = [(1.0, 0.0), (0.0, 0.0), (0.3, -0.05), (0.6, -0.05), (1.0, 0.0)]
    with pytest.raises(ValueError, match="each side"):
        split_into_sections(points, 4)


def test_a_curve_checks_its_knots_with_the_library():
    from spline_fitter.spline import BSplineCurve2D
    ctrl = [(0.0, 0.0), (0.3, 0.2), (0.7, 0.2), (1.0, 0.0)]
    with pytest.raises(ValueError, match=re.escape(
            "Knot vector length must be control points + degree + 1.")):
        BSplineCurve2D(2, ctrl, [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="Knot vector must be clamped on"):
        BSplineCurve2D(2, ctrl, [0, 0, 0.1, 0.5, 1, 1, 1])
    assert BSplineCurve2D(2, ctrl).knot_vector.tolist() == [0, 0, 0, 0.5, 1, 1, 1]


def test_count_groups_thousands_apart_from_the_decimal_point():
    # The window takes the system locale's decimal point: set both explicitly
    before = formatting.decimal_point()
    try:
        formatting.set_decimal_point(".")
        assert formatting.format_count(1500, "point") == "1,500 points"
        formatting.set_decimal_point(",")
        assert formatting.format_count(1500, "point") == "1.500 points"
    finally:
        formatting.set_decimal_point(before)


def test_core_modules_import_without_qt():
    code = ("import sys; import spline_fitter.io, spline_fitter.project_io, "
            "spline_fitter.geometry, spline_fitter.analysis; "
            "assert 'PyQt6' not in sys.modules, 'PyQt6 imported'")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "packages" / "bspline-varpro" / "src")]))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# --- scripts ---------------------------------------------------------------

def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_script_imports():
    assert _script("benchmark_new_optimizers").OPTIMIZERS


def test_analysis_scripts_read_current_projects(tmp_path, monkeypatch):
    path = tmp_path / "p.asfproj"
    save_project(_project(grow_control_points=True), str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    times = _script("analyze_fit_times").analyze_fit_times(data)
    assert times == {"gn-varpro": [(5, 0.5)]}  # fitted, not configured, control points
    _script("compare_search_strategies").analyze_project(data, 1e-5, [1, 2])
    monkeypatch.setattr(sys, "argv", ["compare_search_strategies.py"])
    with pytest.raises(SystemExit):
        _script("compare_search_strategies").main()


def test_remove_evals_script_writes_the_migrated_file(tmp_path, monkeypatch):
    path = tmp_path / "p.asfproj"
    save_project(_project(), str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["optimization_runs"][0]["loss_history"] = [
        {"eval": 1, "iter": 1, "error": 2.0, "time": 0.1},
        {"eval": 2, "iter": 1, "error": 1.0, "time": 0.2}]
    path.write_text(json.dumps(data), encoding="utf-8")
    script = _script("remove_evals_from_proj")
    migrated, before, after = script.migrate_project(json.loads(path.read_text(encoding="utf-8")))
    assert (before, after) == (2, 1)
    monkeypatch.setattr(sys, "argv", ["x", str(path), str(tmp_path / "out.asfproj")])
    script.main()
    history = load_project(str(tmp_path / "out.asfproj")).optimization_runs[0].loss_history
    assert [(r.iteration_number, r.error_value) for r in history] == [(1, 1.0)]


# --- leading edge and retired project fields -------------------------------

LE_INDEX = int(np.argmin(np.hypot(PTS[:, 0], PTS[:, 1])))


@pytest.mark.parametrize("transform", [
    lambda p: p,
    lambda p: p + [0.3, -0.2],                      # shifted, LE no longer at the origin
    lambda p: p * 200.0,                            # millimetres
    lambda p: (p - [1.0, 0.0]) * 100.0,             # TE at the origin, 0-100 scale
    lambda p: p @ np.array([[np.cos(0.2), np.sin(0.2)], [-np.sin(0.2), np.cos(0.2)]]),  # rotated
], ids=["normalized", "shifted", "mm", "te at origin", "rotated"])
def test_leading_edge_does_not_depend_on_where_the_data_sits(transform):
    from spline_fitter.geometry import find_leading_edge_index
    points = [tuple(p) for p in transform(PTS)]
    assert find_leading_edge_index(points) == LE_INDEX
    assert find_leading_edge_index(np.array(points)) == LE_INDEX  # as the deviation plot passes it
    for n_sections in (2, 4):
        sections, junctions = split_into_sections(points, n_sections)
        assert LE_INDEX in junctions
        assert sections[n_sections // 2 - 1][-1] == points[LE_INDEX]


def test_unused_fields_are_not_written_and_old_projects_load(tmp_path):
    path = tmp_path / "p.asfproj"
    save_project(_project(), str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    run = data["optimization_runs"][0]
    for key in ("smoothing_factor", "early_stop_patience", "relative_tolerance"):
        assert key not in run["optimization_config"]
    assert "early_stopped" not in run["result"]
    # A project saved before these fields were dropped still loads
    run["optimization_config"].update(smoothing_factor=0.5, early_stop_patience=20,
                                      relative_tolerance=1e-4)
    run["result"]["early_stopped"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_project(str(path)).optimization_runs[0]
    assert loaded.optimization_config == _project().optimization_runs[0].optimization_config
    assert loaded.result.final_error == 1e-5


def test_rms_error_round_trips_and_old_projects_load_without_it(tmp_path):
    path = tmp_path / "p.asfproj"
    project = _project(tolerance=1e-4)
    project.optimization_runs[0].result.rms_error = 2e-6  # final_error is the max deviation
    save_project(project, str(path))
    assert load_project(str(path)).optimization_runs[0].result.comparison_error == 2e-6
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["optimization_runs"][0]["result"]["rms_error"]
    path.write_text(json.dumps(data), encoding="utf-8")
    result = load_project(str(path)).optimization_runs[0].result
    assert result.rms_error is None and result.comparison_error == 1e-5


def test_a_project_with_a_byte_order_mark_loads(tmp_path):
    path = tmp_path / "p.asfproj"
    save_project(_project(), str(path))
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert len(load_project(str(path)).optimization_runs) == 1


@pytest.mark.parametrize("change, message", [
    (lambda d: d["optimization_runs"][0]["result"].pop("num_evaluations"),
     "Run 1 result: the 'num_evaluations' field is missing."),
    (lambda d: d["airfoil"].update(coordinates=None), "Airfoil: "),
    (lambda d: d["optimization_runs"][0].update(section_control_points=[[["a", 0]]]),
     "Run 1 control points: "),
    (lambda d: d.pop("metadata"), "Metadata: the 'metadata' field is missing."),
], ids=["missing field", "null airfoil", "bad control point", "no metadata"])
def test_a_malformed_project_says_what_is_wrong_and_where(tmp_path, change, message):
    path = tmp_path / "p.asfproj"
    save_project(_project(), str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        load_project(str(path))


@pytest.mark.parametrize("text", ["not json", "[1, 2]"])
def test_a_file_that_is_not_a_project_says_so(tmp_path, text):
    path = tmp_path / "p.asfproj"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="not a project file"):
        load_project(str(path))


def test_saving_over_a_read_only_project_names_it_and_leaves_no_temporary_file(tmp_path):
    import stat
    path = tmp_path / "p.asfproj"
    save_project(_project(), str(path))
    os.chmod(path, stat.S_IREAD)
    try:
        with pytest.raises(PermissionError, match="read-only"):
            save_project(_project(), str(path))
        assert [p.name for p in tmp_path.iterdir()] == ["p.asfproj"]
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.parametrize("name, stem", [
    ("NACA 2412", "NACA 2412"),
    ("E387 t/c=9%", "E387 t_c=9%"),
    ('FX 63-137: "smoothed"?', "FX 63-137_ _smoothed__"),
    ("a\\b. ", "a_b"),
    ("///", "___"),
    ("", "airfoil"),
])
def test_file_stem_makes_any_name_a_file_name(name, stem):
    assert formatting.file_stem(name) == stem
