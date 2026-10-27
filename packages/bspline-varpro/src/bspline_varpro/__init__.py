"""CAD-grade B-spline curve fitting by Levenberg-Marquardt variable projection.

Tolerance-driven non-uniform knots, geometric G1-G3 continuity per junction,
curvature fairing, diagnostics, airfoil helpers and STEP/IGES/DXF export.
"""

from .basis import (
    averaged_knots,
    basis_derivative_matrix,
    basis_matrix,
    bending_energy_matrix,
    centripetal_parameters,
    clamped_knots,
    insert_knot,
    interior_knots,
    open_uniform_knots,
    validate_knots,
)
from . import airfoil, diagnostics, export
from .constraints import (
    ChainConstraints,
    End,
    Junction,
    build_constraints,
    end_derivative_coefficients,
    parse_junctions,
)
from .fairing import fair, refine
from .fitting import ChainOptions, FitResult, fit
from .joint import JointProblem
from .problem import KKTSolver, VarProProblem
from .tolerance import fit_to_tolerance, polish_knots

__version__ = "1.0.0"

__all__ = [
    "airfoil",
    "diagnostics",
    "export",
    "fit",
    "fit_to_tolerance",
    "polish_knots",
    "fair",
    "refine",
    "JointProblem",
    "ChainOptions",
    "FitResult",
    "VarProProblem",
    "KKTSolver",
    "Junction",
    "End",
    "ChainConstraints",
    "parse_junctions",
    "open_uniform_knots",
    "clamped_knots",
    "interior_knots",
    "validate_knots",
    "averaged_knots",
    "insert_knot",
    "centripetal_parameters",
    "basis_matrix",
    "basis_derivative_matrix",
    "bending_energy_matrix",
    "build_constraints",
    "end_derivative_coefficients",
    "__version__",
]
