"""
Fit Quality Report
==================

Deviation, curvature, and junction measurements of a fitted spline chain,
computed with the exact diagnostics of the ``bspline-varpro`` library. Works
for the result of any optimizer.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

import numpy as np
from bspline_varpro import airfoil as bv_airfoil
from bspline_varpro import diagnostics
from scipy.interpolate import BSpline

from .optimization.models import FitResult


def scipy_curves(result: FitResult) -> list[BSpline]:
    """The fitted sections as SciPy B-splines with their knot vectors."""
    return [BSpline(result.knots(i), np.asarray(ctrl, dtype=np.float64), result.degree)
            for i, ctrl in enumerate(result.section_control_points)]


def fit_report(
    sections: Sequence[Sequence[Sequence[float]]],
    result: FitResult,
    precision: int | None = None,
    tolerance: float | None = None,
) -> dict:
    """Quality measurements of a fit.

    Args:
        sections: Data points of every section.
        result: Fitted chain.
        precision: Decimal places of the source data, to flag tolerances and
            deviations below its rounding error.
        tolerance: Requested maximum deviation, if any.

    Returns:
        A JSON-serializable dict with the maximum and RMS orthogonal
        deviation, control-point and knot counts, curvature fairness, the
        leading-edge radius (for two or more sections), and one entry per
        junction.
    """
    curves = scipy_curves(result)
    data = [np.asarray(sec, dtype=np.float64) for sec in sections]
    deviations = diagnostics.section_deviations(data, curves)
    all_dev = np.concatenate(deviations)
    report = {
        "max_deviation": float(np.max(all_dev)),
        "rms_deviation": float(np.sqrt(np.mean(all_dev ** 2))),
        "section_max_deviation": [float(np.max(d)) for d in deviations],
        "control_points": result.num_control_points,
        "section_control_points": [len(c) for c in result.section_control_points],
        "section_interior_knots": [
            int(len(np.unique(result.knots(i)[result.degree + 1:-result.degree - 1])))
            for i in range(len(curves))
        ],
        "degree": result.degree,
        "tolerance": tolerance,
        "precision_floor": None if precision is None else 0.5 * math.sqrt(2.0) * 10.0 ** -precision,
    }
    if result.degree >= 2:
        report.update(diagnostics.fairness(curves))
        report["junctions"] = [
            {
                "gap": j["gap"],
                "tangent_angle_deg": math.degrees(j["tangent_angle"]),
                "curvature_jump": j["curvature_jump"],
                "curvature_slope_jump": j["curvature_slope_jump"],
            }
            for j in diagnostics.junctions(curves)
        ]
        try:
            report["leading_edge_radius"] = bv_airfoil.leading_edge_radius(curves)
        except Exception:
            report["leading_edge_radius"] = None
    return report


def default_tolerance(coordinates: Sequence[Sequence[float]], precision: int | None) -> float:
    """Tolerance from the decimal precision of the source file and the data noise."""
    return bv_airfoil.default_tolerance(np.asarray(coordinates, dtype=np.float64),
                                        decimals=precision if precision else None)


def summary_line(report: dict) -> str:
    """One-line summary for the status bar."""
    text = (f"max dev {report['max_deviation']:.2e}, rms {report['rms_deviation']:.2e}, "
            f"{report['control_points']} control points")
    if report.get("tolerance"):
        text += f" (tol {report['tolerance']:.2e})"
    return text


def format_report_html(report: dict, title: str = "Fit report") -> str:
    """The report as an HTML page for a text browser."""
    def row(label: str, value: str) -> str:
        return f"<tr><td style='padding:2px 12px 2px 0'>{html.escape(label)}</td><td>{value}</td></tr>"

    rows = [
        row("Max orthogonal deviation", f"{report['max_deviation']:.3e}"),
        row("RMS orthogonal deviation", f"{report['rms_deviation']:.3e}"),
        row("Max deviation per section",
            ", ".join(f"{d:.2e}" for d in report["section_max_deviation"])),
    ]
    if report.get("tolerance"):
        met = report["max_deviation"] <= report["tolerance"] * (1 + 1e-9)
        verdict = "<span class='ok'>met</span>" if met else "<b class='warn'>not met</b>"
        rows.append(row("Tolerance", f"{report['tolerance']:.3e} ({verdict})"))
    floor = report.get("precision_floor")
    if floor:
        note = f"{floor:.2e}"
        if report.get("tolerance") and report["tolerance"] < floor:
            note += " <b class='warn'>(tolerance is below the data rounding)</b>"
        rows.append(row("Rounding error of the data", note))
    rows += [
        row("Degree", str(report["degree"])),
        row("Control points (chain)", str(report["control_points"])),
        row("Control points per section", ", ".join(map(str, report["section_control_points"]))),
        row("Interior knots per section", ", ".join(map(str, report["section_interior_knots"]))),
    ]
    if "max_curvature" in report:
        rows += [
            row("Max curvature", f"{report['max_curvature']:.4g}"),
            row("Curvature extrema (wiggles)", str(report["curvature_extrema"])),
            row("Roughness \u222b(d\u03ba/ds)\u00b2 ds", f"{report['roughness']:.4g}"),
        ]
        if report.get("leading_edge_radius"):
            rows.append(row("Leading-edge radius", f"{report['leading_edge_radius']:.6g}"))
    body = "<table>" + "".join(rows) + "</table>"
    if report.get("junctions"):
        body += ("<h4>Junctions</h4><table><tr><th align='left'>#</th><th align='left'>gap</th>"
                 "<th align='left'>tangent angle</th><th align='left'>\u0394\u03ba</th>"
                 "<th align='left'>\u0394 d\u03ba/ds</th></tr>")
        for i, j in enumerate(report["junctions"], start=1):
            body += (f"<tr><td>{i}</td><td>{j['gap']:.1e}</td><td>{j['tangent_angle_deg']:.2e}\u00b0</td>"
                     f"<td>{j['curvature_jump']:.2e}</td><td>{j['curvature_slope_jump']:.2e}</td></tr>")
        body += "</table>"
    return f"<html><body><h3>{html.escape(title)}</h3>{body}</body></html>"
