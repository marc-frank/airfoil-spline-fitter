"""Re-import exported STEP, IGES and DXF files with OpenCASCADE and ezdxf.

Needs ``pip install cadquery-ocp ezdxf`` (not dependencies of the library).
Fits a four-section NACA 2412 with mixed G1/C2/G3 junctions and a blunt
NACA 4412 with its trailing-edge segment, writes every format separately and
merged, and reports the largest point error after re-import.

    python benchmarks/verify_cad.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from bspline_varpro import airfoil, export, fit_to_tolerance
from bspline_varpro.airfoil import NACA4

UNIT_MM = {"mm": 1.0, "m": 1000.0, "inch": 25.4}


def ocp_edges(path: Path):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_BSplineCurve
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESControl import IGESControl_Reader
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    reader = STEPControl_Reader() if path.suffix == ".step" else IGESControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"OpenCASCADE could not read {path}")
    reader.TransferRoots()
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_EDGE)
    edges = []
    while explorer.More():
        cast = TopoDS.Edge if hasattr(TopoDS, "Edge") else TopoDS.Edge_s
        adaptor = BRepAdaptor_Curve(cast(explorer.Current()))
        if adaptor.GetType() != GeomAbs_BSplineCurve:
            raise RuntimeError(f"{path.name}: edge is not a B-spline")
        edges.append(adaptor)
        explorer.Next()
    return edges


def ocp_error(path: Path, curves, scale: float, unit_mm: float) -> float:
    edges = ocp_edges(path)
    if len(edges) != len(curves):
        raise RuntimeError(f"{path.name}: {len(edges)} edges, expected {len(curves)}")
    worst = 0.0
    for edge, curve in zip(edges, curves):
        spline = edge.BSpline()
        if spline.Degree() != curve.k or spline.NbPoles() != len(curve.t) - curve.k - 1:
            raise RuntimeError(f"{path.name}: degree or pole count changed")
        u0, u1 = edge.FirstParameter(), edge.LastParameter()
        lo, hi = curve.t[curve.k], curve.t[-curve.k - 1]
        for s in np.linspace(0.0, 1.0, 51):
            p = edge.Value(u0 + s * (u1 - u0))
            q = curve(lo + s * (hi - lo)) * scale * unit_mm
            worst = max(worst, float(np.hypot(p.X() - q[0], p.Y() - q[1])), abs(p.Z()))
    return worst / unit_mm


def dxf_error(path: Path, curves, scale: float) -> float:
    import ezdxf

    doc = ezdxf.readfile(str(path))
    if doc.audit().errors:
        raise RuntimeError(f"{path.name}: ezdxf audit errors")
    splines = list(doc.modelspace().query("SPLINE"))
    worst = 0.0
    for entity, curve in zip(splines, curves):
        tool = entity.construction_tool()
        lo, hi = curve.t[curve.k], curve.t[-curve.k - 1]
        for s in np.linspace(0.0, 1.0, 51):
            p = tool.point(s * tool.max_t)
            q = curve(lo + s * (hi - lo)) * scale
            worst = max(worst, float(np.hypot(p[0] - q[0], p[1] - q[1])))
    return worst


def main() -> None:
    naca2412 = NACA4("2412").points(81)
    chain = fit_to_tolerance([naca2412[:40], naca2412[39:80], naca2412[79:120], naca2412[119:]],
                             1e-5, continuity=["G1", "C2", "G3"]).curves()
    blunt = airfoil.fit_airfoil(np.round(NACA4("4412", closed_te=False).points(81), 6), 1e-5)
    blunt_curves = blunt.curves() + [airfoil.trailing_edge_segment(blunt)]
    cases = [
        ("naca2412 4 sections", chain, 250.0, "inch"),
        ("naca2412 merged", [export.merge_curves(chain)], 250.0, "inch"),
        ("naca4412 blunt + TE", blunt_curves, 1000.0, "mm"),
        ("naca4412 merged", [export.merge_curves(blunt.curves())], 1.0, "m"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for label, curves, scale, units in cases:
            errors = []
            for suffix in (".step", ".iges", ".dxf"):
                path = export.write(Path(tmp) / f"part{suffix}", curves, name="it's a test",
                                    scale=scale, units=units)
                if suffix == ".dxf":
                    errors.append(dxf_error(path, curves, scale))
                else:
                    errors.append(ocp_error(path, curves, scale, UNIT_MM[units]))
            print(f"{label:22s} STEP {errors[0]:.1e}  IGES {errors[1]:.1e}  DXF {errors[2]:.1e}  "
                  f"({units}, x{scale:g})")


if __name__ == "__main__":
    main()
