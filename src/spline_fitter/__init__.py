"""
Airfoil Spline Fitter
=====================

A professional-grade tool for fitting B-spline curves to airfoil coordinate data.

This package provides:
    - Interactive GUI for airfoil spline fitting with real-time visualization
    - VarPro Levenberg-Marquardt (via bspline-varpro) and Direct LSQ fitting
    - Tolerance mode: fewest control points (non-uniform knots) within a maximum deviation
    - Parametric (C0-C3) or geometric (G1-G3) continuity, selectable per junction
    - Fit reports, STEP/IGES/DXF export and batch conversion of airfoil libraries

Architecture
------------
The package is organized into focused modules:

    optimization    Data models (optimization.models), optimizer wrappers, error logging
    spline          B-spline curve evaluation and curvature helpers
    geometry        Coordinate processing and geometric utilities
    io              Airfoil and spline .dat files
    project_io      Project files (.asfproj)
    analysis        Deviation, curvature and junction reports of a fit
    gui             PyQt6-based graphical user interface (imported on first use,
                    so the rest of the package works without Qt)
    formatting      Display formatting utilities

Example
-------
To run the GUI application from a source checkout:

    python -m src.spline_fitter

Or programmatically:

    from spline_fitter.gui import SplineFitterWindow
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = SplineFitterWindow()
    window.show()
    app.exec()
"""

__version__ = "2.0.0"
__author__ = "Airfoil Library Project"

import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path

# Running from a source checkout without an installed bspline-varpro: use the
# checkout's own. An installed library is never shadowed.
_checkout_library = _Path(__file__).resolve().parents[2] / "packages" / "bspline-varpro" / "src"
if _importlib_util.find_spec("bspline_varpro") is None and _checkout_library.is_dir():
    _sys.path.append(str(_checkout_library))

# Core data models
# Fit analysis
from .analysis import fit_report, scipy_curves

# Geometry utilities
from .geometry import (
    compute_arc_length_parameters,
    compute_bounding_box,
    create_airfoil_data,
    find_leading_edge_index,
    interpolate_along_polyline,
    split_at_leading_edge,
    split_into_sections,
)

# Optimization engine
from .optimization import (
    OptimizerErrorLogger,
    run_optimization,
)
from .optimization.models import (
    AirfoilData,
    BoundingBox,
    ContinuityMode,
    FitResult,
    IterationRecord,
    OptimizationConfig,
    OptimizationRun,
    OptimizerType,
    ProjectFile,
    ProjectMetadata,
    SplineConfig,
)

# Project file I/O
from .project_io import (
    create_optimization_run,
    create_project,
    create_project_metadata,
    load_project,
    save_project,
)

# Spline mathematics
from .spline import (
    BSplineCurve2D,
    compute_arc_length,
    compute_curvature_derivative,
    compute_signed_curvature,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Data models
    "ContinuityMode",
    "OptimizerType",
    "AirfoilData",
    "SplineConfig",
    "OptimizationConfig",
    "FitResult",
    "BoundingBox",
    "IterationRecord",
    "OptimizationRun",
    "ProjectFile",
    "ProjectMetadata",
    # Spline math
    "BSplineCurve2D",
    "compute_signed_curvature",
    "compute_curvature_derivative",
    "compute_arc_length",
    # Geometry
    "find_leading_edge_index",
    "split_at_leading_edge",
    "split_into_sections",
    "create_airfoil_data",
    "compute_arc_length_parameters",
    "interpolate_along_polyline",
    "compute_bounding_box",
    # Optimization
    "run_optimization",
    "OptimizerErrorLogger",
    # Analysis
    "fit_report",
    "scipy_curves",
    # Project I/O
    "save_project",
    "load_project",
    "create_project",
    "create_project_metadata",
    "create_optimization_run",
    # UI
    "SplineFitterWindow",
]


def __getattr__(name):
    # The GUI needs PyQt6 and pyqtgraph: import it only when asked for
    if name == "SplineFitterWindow":
        from .gui import SplineFitterWindow
        return SplineFitterWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
