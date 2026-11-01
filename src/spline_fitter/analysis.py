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

from .formatting import number
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


# Newton steps that refine the foot points of a fit from those of the fit
# before it (a live preview's previous step)
REFINE_STEPS = 3


def _domain(curve: BSpline) -> tuple[float, float]:
    return float(curve.t[curve.k]), float(curve.t[-curve.k - 1])


def _search(pts: np.ndarray, curves: list[BSpline]) -> np.ndarray:
    """Parameter of every point's nearest point on each whole curve, (n, sections)."""
    return np.column_stack([diagnostics.project(pts, curve)[0] for curve in curves])


def _refine(pts: np.ndarray, curves: list[BSpline], u: np.ndarray) -> np.ndarray:
    """Newton steps on each curve from the previous parameters, (n, sections).

    Every section keeps its own foot of every point, so a point whose
    nearest section changes (at a junction, or across the trailing edge,
    where a lower-surface point also has a foot on the upper surface) is
    still caught: the nearest of the feet is taken afterwards."""
    u = u.copy()
    for i, curve in enumerate(curves):
        lo, hi = _domain(curve)
        ui = np.clip(u[:, i], lo, hi)
        for _ in range(REFINE_STEPS):
            c0, c1, c2 = curve(ui), curve(ui, nu=1), curve(ui, nu=2)
            diff = c0 - pts
            g = np.sum(diff * c1, axis=1)
            h = np.sum(c1 * c1, axis=1) + np.sum(diff * c2, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ui = np.clip(ui - np.where(h > 0, g / h, 0.0), lo, hi)
        u[:, i] = ui
    return u


def foot_points(
    points: Sequence[Sequence[float]], result: FitResult, start: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Signed orthogonal distance of every point to the nearest section of a
    fit, with its foot point on the curve and the unit normal there.

    The normal points to the right of the curve's direction of travel, and
    the distance is positive on that side: outside the airfoil for
    coordinates in Selig order (counterclockwise), inside for the other way
    round. So ``foot + distance * normal`` is the point.

    Args:
        start: The parameters this returned for a fit close to this one, to
            refine with a few Newton steps instead of searching every curve
            (the live preview, step after step). A start for another number
            of points or sections is ignored.

    Returns:
        ``(distance, foot, normal, start)``, shaped (n,), (n, 2), (n, 2), and
        the parameters of every point on every section, (n, sections), to
        start from for the next fit.
    """
    pts = np.asarray(points, dtype=np.float64)
    curves = scipy_curves(result)
    if start is not None and start.shape == (len(pts), len(curves)):
        u = _refine(pts, curves, start)
    else:
        u = _search(pts, curves)
    best = np.full(len(pts), np.inf)
    foot, tangent = np.zeros_like(pts), np.zeros_like(pts)
    for i, curve in enumerate(curves):
        at = curve(u[:, i])
        d = np.hypot(*(pts - at).T)
        closer = d < best
        best = np.where(closer, d, best)
        foot[closer] = at[closer]
        tangent[closer] = curve(u[closer, i], nu=1)
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    normal /= np.maximum(np.hypot(normal[:, 0], normal[:, 1]), 1e-300)[:, None]
    signed = np.copysign(best, np.sum((pts - foot) * normal, axis=1))
    return signed, foot, normal, u


def signed_deviations(points: Sequence[Sequence[float]], result: FitResult) -> np.ndarray:
    """Signed orthogonal distance of every point to the nearest section of a
    fit, positive outside (see :func:`foot_points`)."""
    return foot_points(points, result)[0]


def default_tolerance(coordinates: Sequence[Sequence[float]], precision: int | None) -> float:
    """Tolerance from the decimal precision of the source file and the data noise."""
    return bv_airfoil.default_tolerance(np.asarray(coordinates, dtype=np.float64),
                                        decimals=precision if precision else None)


def summary_line(report: dict) -> str:
    """One-line summary for the status bar."""
    text = (f"max deviation {number(report['max_deviation'], '.2e')}, "
            f"RMS {number(report['rms_deviation'], '.2e')}, "
            f"{report['control_points']} control points")
    if report.get("tolerance"):
        text += f" (tolerance {number(report['tolerance'], '.2e')})"
    return text


def format_report_html(report: dict, title: str = "Fit report") -> str:
    """The report as an HTML page for a text browser."""
    def row(label: str, value: str) -> str:
        return f"<tr><td>{html.escape(label)}</td><td class='num'>{value}</td></tr>"

    rows = [
        row("Max orthogonal deviation", f"{number(report['max_deviation'], '.3e')}"),
        row("RMS orthogonal deviation", f"{number(report['rms_deviation'], '.3e')}"),
        row("Max deviation per section",
            ", ".join(f"{number(d, '.2e')}" for d in report["section_max_deviation"])),
    ]
    if report.get("tolerance"):
        met = report["max_deviation"] <= report["tolerance"] * (1 + 1e-9)
        verdict = "<span class='ok'>met</span>" if met else "<b class='warn'>not met</b>"
        rows.append(row("Tolerance", f"{number(report['tolerance'], '.3e')} ({verdict})"))
    floor = report.get("precision_floor")
    if floor:
        note = f"{number(floor, '.2e')}"
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
            row("Max curvature", f"{number(report['max_curvature'], '.4g')}"),
            row("Curvature extrema (wiggles)", str(report["curvature_extrema"])),
            row("Roughness \u222b(d\u03ba/ds)\u00b2 ds", f"{number(report['roughness'], '.4g')}"),
        ]
        if report.get("leading_edge_radius"):
            rows.append(row("Leading-edge radius", number(report["leading_edge_radius"], ".6g")))
    body = "<table>" + "".join(rows) + "</table>"
    if report.get("junctions"):
        body += ("<h4>Junctions</h4><table><tr><th align='left'>Junction</th>"
                 "<th align='left'>Gap</th>"
                 "<th align='left'>Tangent angle</th><th align='left'>\u0394\u03ba</th>"
                 "<th align='left'>\u0394 d\u03ba/ds</th></tr>")
        for i, j in enumerate(report["junctions"], start=1):
            body += (f"<tr><td class='num'>{i}</td><td class='num'>{number(j['gap'], '.1e')}</td>"
                     f"<td class='num'>{number(j['tangent_angle_deg'], '.2e')}\u00b0</td>"
                     f"<td class='num'>{number(j['curvature_jump'], '.2e')}</td>"
                     f"<td class='num'>{number(j['curvature_slope_jump'], '.2e')}</td></tr>")
        body += "</table>"
    return f"<html><body><h3>{html.escape(title)}</h3>{body}</body></html>"
