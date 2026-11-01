"""The chain is fitted to all points as a whole: the sections only give the
starting split, and points move across junctions."""
import numpy as np
import pytest

import bspline_varpro as bv
from bspline_varpro.airfoil import NACA4
from bspline_varpro.chain import (
    chain_data, chain_parameters, locate, project_to_chain, section_ranges, shared_junctions,
)


def naca_sections(n=61, code="2412"):
    pts = NACA4(code).points(n)
    return [pts[:n], pts[n - 1:]]


def test_a_shared_junction_point_is_one_data_point():
    sections = naca_sections()
    assert shared_junctions(sections) == [True]
    assert len(chain_data(sections)) == 121
    assert section_ranges(sections) == [(0, 60), (60, 120)]
    u = chain_parameters([np.linspace(0, 1, 61)] * 2, sections)
    assert len(u) == 121 and u[60] == 1.0 and u[-1] == 2.0


def test_separate_sections_start_apart():
    sections = [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])]
    u = chain_parameters([np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0])], sections)
    assert np.all(np.diff(u) > 0)


def test_chain_parameters_place_points_on_sections():
    section, local = locate(np.array([0.0, 0.5, 1.0, 1.25, 2.0]), 2)
    np.testing.assert_array_equal(section, [0, 0, 1, 1, 1])
    np.testing.assert_allclose(local, [0.0, 0.5, 0.0, 0.25, 1.0])


def test_points_cross_a_free_junction():
    sections = naca_sections()
    result = bv.fit(sections, 12, 3, continuity="G1", max_nfev=400, fix_junctions=False)
    # Upper-surface points now lie on the second section, past the moved junction
    assert np.any(result.parameters[:61] > 1.0)
    assert result.control_points[0][-1][1] > 1e-3


def test_both_parameter_forms_are_accepted():
    sections = naca_sections(31)
    first = bv.fit(sections, 8, 3, max_nfev=100)
    per_section = [np.linspace(0, 1, 31)] * 2
    a = bv.fit(sections, 8, 3, max_nfev=100, initial_parameters=first.parameters)
    b = bv.fit(sections, 8, 3, max_nfev=100, initial_parameters=per_section)
    assert a.rms_error <= first.rms_error * (1 + 1e-9)
    assert np.isfinite(b.rms_error)
    with pytest.raises(ValueError, match="Chain parameters"):
        bv.fit(sections, 8, 3, initial_parameters=np.linspace(0, 1, 61))


def test_projection_stays_between_its_neighbours():
    # The two trailing-edge ends coincide: a point near one must not jump to the other
    sections = naca_sections(201, code="0012")
    data = chain_data(sections)
    fit = bv.fit([s[::10] for s in sections], 10, 5, continuity="G2")
    chosen = np.concatenate([np.arange(0, 201, 10), np.arange(200, 401, 10)])
    chosen = np.unique(chosen)
    u = project_to_chain(data, fit.curves(), chosen, fit.parameters)
    assert np.all(np.diff(u) >= 0)
    assert np.all(u[:200] <= 1.0 + 1e-9) and np.all(u[201:] >= 1.0 - 1e-9)
