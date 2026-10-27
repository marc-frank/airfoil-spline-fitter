"""
Data Models
===========

Immutable data structures for the spline fitting application.

This module defines all the core data types used throughout the application,
following the principle of making state explicit and easy to reason about.

All models are implemented as dataclasses or NamedTuples for immutability,
clear documentation, and automatic comparison/hashing support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

# =============================================================================
# Type Aliases
# =============================================================================

Point2D = tuple[float, float]
"""A 2D point represented as (x, y) coordinates."""

PointList = list[Point2D]
"""A list of 2D points forming a polyline or point cloud."""

Bounds = tuple[float, float]
"""A numeric range represented as (lower, upper) bounds."""


# =============================================================================
# Enumerations
# =============================================================================

class ContinuityMode(Enum):
    """
    Continuity order at curve junctions.
    
    Airfoils are fit with n separate splines that meet at junction points.
    This enum specifies how many derivatives are linked at each junction.
    Whether they are linked geometrically (G^n: matching tangent direction,
    curvature, curvature rate) or parametrically (C^n: equal derivative
    vectors) is chosen by ``SplineConfig.geometric_continuity``. Only the
    VarPro LM optimizer supports geometric continuity; the other optimizers
    always use the parametric form.
    
    Attributes:
        G0: Position continuity only (curves meet but may have a corner)
        G1: Tangent continuity (smooth tangent direction, cusp-free)
        G2: Curvature continuity (matching curvature at junction)
        G3: Curvature-rate continuity (smoothest possible connection)
    """
    G0 = auto()
    G1 = auto()
    G2 = auto()
    G3 = auto()

    @property
    def minimum_degree(self) -> int:
        """Return the minimum spline degree required for this continuity level."""
        return {
            ContinuityMode.G0: 1,
            ContinuityMode.G1: 1,
            ContinuityMode.G2: 2,
            ContinuityMode.G3: 3,
        }[self]

    @property
    def constrained_control_points(self) -> int:
        """Return the number of control points constrained at each junction."""
        return {
            ContinuityMode.G0: 1,  # Just the LE point
            ContinuityMode.G1: 2,  # LE + 1 tangent control point per side
            ContinuityMode.G2: 3,  # LE + 2 curvature control points per side
            ContinuityMode.G3: 4,  # LE + 3 curvature-rate control points per side
        }[self]

    def minimum_control_points(self, degree: int) -> int:
        """
        Calculate minimum control points needed for given degree and continuity.
        
        Args:
            degree: The B-spline degree
            
        Returns:
            Minimum number of control points per curve segment
        """
        # Need at least degree+1 for a valid B-spline
        base_minimum = degree + 1
        # Plus enough for the constrained LE points
        return max(base_minimum, self.constrained_control_points + 1)


class OptimizerType(Enum):
    """
    Available optimization algorithms.
    
    Ordered by recommended usage based on benchmark testing (41 airfoils):
    
    Tier 1 - Production (recommended):
        - BFGS: Best speed/accuracy tradeoff (~9s, 90% of best accuracy)
        - SLSQP: Highest accuracy but 26x slower (~220s)
    
    Tier 2 - Acceptable alternatives:
        - L_BFGS_B: Fastest optimizer (~3s), lower accuracy
        - CG: Conjugate Gradient, moderate performance
    
    Tier 3 - Not recommended:
        - TNC: Truncated Newton
        - POWELL: High timeout rate (68%)
        - NELDER_MEAD: Poor accuracy (74x worse)
        - COBYLA: Poor accuracy (41x worse)
    
    Global optimizers (slow but thorough):
        - DE: Differential Evolution
        - DA: Dual Annealing
        - BH: Basin-hopping
    """
    # Tier 1: Recommended (ordered by speed/accuracy tradeoff)
    BFGS = "bfgs"
    SLSQP = "slsqp"

    # Tier 2: Acceptable alternatives
    L_BFGS_B = "l-bfgs-b"
    CG = "cg"

    # Tier 3: Not recommended (poor performance in benchmarks)
    TNC = "tnc"
    POWELL = "powell"
    NELDER_MEAD = "nelder-mead"
    COBYLA = "cobyla"

    # Global optimizers (slow but thorough search)
    DE = "de"
    DA = "da"
    BH = "bh"

    # Algebraic (non-iterative) method
    CONSTRAINED_LSQ = "constrained-lsq"

    # Alternating LSQ for orthogonal distance
    ORTHOGONAL_LSQ = "orthogonal-lsq"

    # Two-stage parameterization optimizer
    PARAM_OPT = "param-opt"

    # Direct LSQ (single-step, no iteration)
    DIRECT_LSQ = "direct-lsq"

    # Levenberg-Marquardt variable projection (bspline-varpro library)
    GN_VARPRO = "gn-varpro"

    @property
    def is_global(self) -> bool:
        """Return True if this is a global optimizer."""
        return self in (OptimizerType.DE, OptimizerType.DA, OptimizerType.BH)

    @property
    def is_algebraic(self) -> bool:
        """Return True if this is an algebraic (non-iterative) method."""
        return self in (OptimizerType.CONSTRAINED_LSQ, OptimizerType.DIRECT_LSQ)

    @property
    def is_iterative_algebraic(self) -> bool:
        """Return True if this uses iterative algebraic (LSQ) steps."""
        return self in (
            OptimizerType.ORTHOGONAL_LSQ,
            OptimizerType.PARAM_OPT,
            OptimizerType.GN_VARPRO,
        )

    @property
    def is_param_optimizer(self) -> bool:
        """Return True if this is the two-stage parameterization optimizer."""
        return self == OptimizerType.PARAM_OPT

    @property
    def display_name(self) -> str:
        """Return a human-readable name for the optimizer."""
        names = {
            # Tier 1: Recommended
            OptimizerType.BFGS: "BFGS (recommended)",
            OptimizerType.SLSQP: "SLSQP (most accurate)",
            # Tier 2: Acceptable
            OptimizerType.L_BFGS_B: "L-BFGS-B (fastest)",
            OptimizerType.CG: "CG",
            # Tier 3: Not recommended
            OptimizerType.TNC: "TNC",
            OptimizerType.POWELL: "Powell",
            OptimizerType.NELDER_MEAD: "Nelder-Mead",
            OptimizerType.COBYLA: "COBYLA",
            # Global
            OptimizerType.DE: "Differential Evolution (global)",
            OptimizerType.DA: "Dual Annealing (global)",
            OptimizerType.BH: "Basin-hopping (global)",
            # Algebraic
            OptimizerType.CONSTRAINED_LSQ: "Constrained LSQ (algebraic)",
            OptimizerType.ORTHOGONAL_LSQ: "Orthogonal LSQ (alternating)",
            # Two-stage parameterization
            OptimizerType.PARAM_OPT: "Param Optimizer (two-stage)",
            # New high-performance optimizers
            OptimizerType.DIRECT_LSQ: "Direct LSQ (fastest, single-step)",
            OptimizerType.GN_VARPRO: "VarPro LM (CAD: tolerance, G1-G3)",
        }
        return names.get(self, self.value)

    @property
    def tooltip(self) -> str:
        """Return a tooltip description with benchmark performance info."""
        tooltips = {
            # Tier 1: Recommended
            OptimizerType.BFGS: (
                "Quasi-Newton method. Best speed/accuracy tradeoff.\n"
                "Benchmark: ~9s avg, 7.8×10⁻⁵ error, 100% reliable.\n"
                "Recommended for interactive use."
            ),
            OptimizerType.SLSQP: (
                "Sequential Least Squares Programming.\n"
                "Benchmark: ~220s avg, 7.0×10⁻⁵ error (best), 100% reliable.\n"
                "Use when maximum accuracy is needed."
            ),
            # Tier 2: Acceptable
            OptimizerType.L_BFGS_B: (
                "Limited-memory BFGS with bounds.\n"
                "Benchmark: ~3s avg, 1.7×10⁻⁴ error, 100% reliable.\n"
                "Fastest option, lower accuracy."
            ),
            OptimizerType.CG: (
                "Conjugate Gradient method.\n"
                "Benchmark: ~10s avg, 1.8×10⁻⁴ error, 100% reliable.\n"
                "No clear advantage over BFGS."
            ),
            # Tier 3: Not recommended
            OptimizerType.TNC: (
                "Truncated Newton method.\n"
                "Not benchmarked. Use BFGS instead."
            ),
            OptimizerType.POWELL: (
                "Powell's direction-set method.\n"
                "Benchmark: 68% timeout rate (>5 min).\n"
                "Not recommended for spline fitting."
            ),
            OptimizerType.NELDER_MEAD: (
                "Simplex method (derivative-free).\n"
                "Benchmark: 74× worse accuracy than SLSQP.\n"
                "Not recommended for spline fitting."
            ),
            OptimizerType.COBYLA: (
                "Constrained Optimization BY Linear Approximation.\n"
                "Benchmark: 41× worse accuracy than SLSQP.\n"
                "Not recommended for spline fitting."
            ),
            # Global optimizers
            OptimizerType.DE: (
                "Differential Evolution (global optimizer).\n"
                "Explores entire search space. Very slow but robust.\n"
                "Use only if local optimizers fail."
            ),
            OptimizerType.DA: (
                "Dual Annealing (global optimizer).\n"
                "Simulated annealing variant. Very slow.\n"
                "Use only if local optimizers fail."
            ),
            OptimizerType.BH: (
                "Basin-hopping (global optimizer).\n"
                "Local optimizer with random restarts. Slow.\n"
                "Use only if local optimizers fail."
            ),
            # Algebraic method
            OptimizerType.CONSTRAINED_LSQ: (
                "Constrained Least-Squares B-spline fitting.\n"
                "Non-iterative: algebraically optimal control points.\n"
                "Fixed TE endpoints, configurable continuity (G0-G2).\n"
                "Junction positions fixed or free via 'Optimize junctions' checkbox.\n"
                "Very fast with proper constraint enforcement."
            ),
            OptimizerType.ORTHOGONAL_LSQ: (
                "Alternating LSQ for orthogonal distance fitting.\n"
                "Iterates: LSQ solve → foot-point projection → repeat.\n"
                "Combines LSQ speed with orthogonal distance accuracy.\n"
                "Typically converges in 3-10 iterations."
            ),
            # Two-stage parameterization optimizer
            OptimizerType.PARAM_OPT: (
                "Two-stage parameterization optimization.\n"
                "Outer: Optimizes parameter (t) values for each data point.\n"
                "Inner: Constrained LSQ fits control points.\n"
                "Objective: Minimize orthogonal distance or bending energy.\n"
                "Use Param Optimizer settings panel to configure."
            ),
            # New high-performance optimizers
            OptimizerType.DIRECT_LSQ: (
                "Direct LSQ (single-step, no iteration).\n"
                "Uses centripetal parameterization + one LSQ solve.\n"
                "Fastest option (~1ms), moderate accuracy (~1e-4).\n"
                "Ideal for dense data or real-time preview."
            ),
            OptimizerType.GN_VARPRO: (
                "Levenberg-Marquardt Variable Projection (bspline-varpro).\n"
                "Tolerance mode: fewest control points within a maximum\n"
                "orthogonal deviation, non-uniform knots.\n"
                "Geometric (G) or parametric (C) continuity per junction.\n"
                "Recommended for CAD-quality results."
            ),
        }
        return tooltips.get(self, "")


class ErrorMetric(Enum):
    """
    Distance metric for measuring spline fit quality.
    
    Attributes:
        PARAMETRIC: Average distance at corresponding parameter values (fast)
        ORTHOGONAL: Average closest-point distance (accurate but slower)
        ORTHOGONAL_LOCAL: Orthogonal within local parameter neighborhood (balanced)
    """
    PARAMETRIC = "avg_point"
    ORTHOGONAL = "orthogonal"
    ORTHOGONAL_LOCAL = "orthogonal_local"

    @property
    def display_name(self) -> str:
        """Return a human-readable name for the metric."""
        names = {
            ErrorMetric.PARAMETRIC: "Avg pointwise",
            ErrorMetric.ORTHOGONAL: "Avg orthogonal",
            ErrorMetric.ORTHOGONAL_LOCAL: "Avg orthogonal (local)",
        }
        return names.get(self, self.value)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class AirfoilData:
    """
    Parsed airfoil coordinate data with metadata.
    
    This represents a complete airfoil as loaded from a .dat file,
    with both the raw coordinates and extracted metadata.
    
    Attributes:
        coordinates: All (x, y) points in Selig format (TE → LE → TE)
        sections: List of n curve sections (each as tuple of points)
        junction_indices: Indices in coordinates of junction points between sections
        num_sections: int
        filename: Original filename without path
        name: Airfoil name from file content (may differ from filename)
        precision: Number of decimal places in the original data
    """
    coordinates: tuple[Point2D, ...]
    sections: tuple[tuple[Point2D, ...], ...]
    junction_indices: tuple[int, ...]
    num_sections: int
    filename: str = ""
    name: str = ""
    precision: int = 6

    @property
    def num_points(self) -> int:
        """Return the total number of coordinate points."""
        return len(self.coordinates)

    def as_numpy(self) -> np.ndarray:
        """Return coordinates as a numpy array of shape (N, 2)."""
        return np.array(self.coordinates, dtype=np.float64)


@dataclass(frozen=True)
class SplineConfig:
    """
    Configuration for B-spline curve fitting.
    
    Defines the mathematical properties of the fitted spline curves.
    
    Attributes:
        degree: Polynomial degree of the B-spline (typically 3-5)
        num_control_points: Number of control points per curve segment
        num_sections: Number of curve sections (1 = single wrap, 2 = upper/lower split)
        continuity: Continuity order at junctions (G0/G1/G2/G3)
        bbox_scale: Scale factor for control point bounding box (1.0 = data bbox)
        geometric_continuity: Link junction derivatives geometrically (G^n)
            instead of parametrically (C^n). VarPro LM only.
        junction_continuity: Optional per-junction overrides such as
            ``("G2", "C1", "G3")``, one per junction (VarPro LM only).
    """
    degree: int = 5
    num_control_points: int = 25
    num_sections: int = 2
    continuity: ContinuityMode = ContinuityMode.G1
    bbox_scale: float = 3.0
    geometric_continuity: bool = False
    junction_continuity: tuple[str, ...] = ()

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_sections < 1:
            raise ValueError(
                f"num_sections must be >= 1, got {self.num_sections}"
            )
        if self.degree < self.continuity.minimum_degree:
            raise ValueError(
                f"{self.continuity.name} continuity requires degree >= "
                f"{self.continuity.minimum_degree}, got {self.degree}"
            )
        object.__setattr__(self, "junction_continuity",
                           tuple(s.strip().upper() for s in self.junction_continuity))
        if self.junction_continuity:
            if len(self.junction_continuity) != self.num_sections - 1:
                raise ValueError(
                    f"{self.num_sections} sections have {self.num_sections - 1} junction(s), "
                    f"but {len(self.junction_continuity)} continuity entries were given"
                )
            for spec in self.junction_continuity:
                if len(spec) != 2 or spec[0] not in "CG" or spec[1] not in "0123":
                    raise ValueError(f"Invalid junction continuity {spec!r}; use e.g. G2 or C1")
                if int(spec[1]) > self.degree:
                    raise ValueError(f"{spec} continuity requires degree >= {spec[1]}")

    def junction_specs(self) -> list[str]:
        """Continuity of every junction as ``"G2"``/``"C1"`` style strings."""
        if self.junction_continuity:
            return list(self.junction_continuity)
        kind = "G" if self.geometric_continuity else "C"
        return [f"{kind}{self.continuity.name[1]}"] * (self.num_sections - 1)

    @property
    def free_control_points_per_section(self) -> int:
        """
        Calculate the number of free (optimizable) control points per section.
        
        Each section has: [endpoint] + [free points] + [constrained junction points]
        """
        # Total minus endpoints (2) minus constrained junction points
        return self.num_control_points - 2 - (self.continuity.constrained_control_points - 1)


@dataclass(frozen=True)
class OptimizationConfig:
    """
    Configuration for the optimization process.

    Attributes:
        optimizer: Which optimization algorithm to use
        metric: Distance metric for error computation
        max_iterations: Maximum optimizer iterations
        seed: Random seed for reproducibility (-1 for random)
        early_stop_patience: Iterations without improvement before stopping
        relative_tolerance: Relative improvement threshold for early stopping
        optimize_le_position: If True, LE point (x, y) becomes optimization variables
        smoothing_factor: Smoothing parameter for algebraic methods (0 = interpolation)
        smoothing_weight: Bending energy smoothing weight for GN-VarPro (0 = disabled).
            Weight of the integrated squared parameter second derivative.
            The same penalty is included in the inner solve and outer objective.
        custom_parameters: Optional per-section parameter values for data points.
            If provided, must be a list with one array per section, each array
            having same length as data points in that section. Values should
            be in [0, 1] range. If None, arc-length parametrisation is used.
        tolerance: Maximum orthogonal deviation for VarPro LM tolerance mode,
            which picks the number of control points and the knots itself.
            None fits the configured number of control points.
        max_points_per_section: VarPro LM only. Sections with more points are
            fitted on an evenly spaced subsample of this many points, then
            refined jointly on all points. None (default) uses every point.
        knot_placement: VarPro LM only. "uniform" keeps knots evenly spaced;
            "free" (default) moves them to where they reduce the error most.
        grow_control_points: VarPro LM only. Start from the fewest control
            points and add them where the fit is worst (default). If False,
            start from the configured control-point count (in tolerance mode
            it can still grow or shrink from there).
    """
    optimizer: OptimizerType = OptimizerType.DE
    metric: ErrorMetric = ErrorMetric.PARAMETRIC
    max_iterations: int = 10000
    smoothing_factor: float | None = None
    seed: int = -1
    early_stop_patience: int = 50
    relative_tolerance: float = 1e-6
    optimize_le_position: bool = False
    smoothing_weight: float = 1e-3
    custom_parameters: list[np.ndarray] | None = None
    tolerance: float | None = None
    max_points_per_section: int | None = None
    knot_placement: str = "free"
    grow_control_points: bool = True

    def __post_init__(self):
        if self.tolerance is not None and not self.tolerance > 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
        if self.max_points_per_section is not None and self.max_points_per_section < 2:
            raise ValueError(
                f"max_points_per_section must be at least 2, got {self.max_points_per_section}")
        if self.knot_placement not in ("uniform", "free"):
            raise ValueError(f"knot_placement must be 'uniform' or 'free', got {self.knot_placement!r}")

    @property
    def random_seed(self) -> int | None:
        """Return the seed value, or None if random."""
        return None if self.seed < 0 else self.seed


@dataclass(frozen=True)
class FairnessWeights:
    """
    Weights for fairness regularization terms.
    
    Fairness regularization encourages aesthetically pleasing control
    point distributions and smooth curvature profiles. Each weight
    controls how strongly to penalize a particular aspect of unfairness.
    
    All weights should be in range [0, 1]. A value of 0 disables the term.
    
    Attributes:
        length_variance: Penalize uneven spacing between control points
        polygon_bend: Penalize sharp bends in the control polygon
        anti_bunching: Penalize control points that are too close together
        end_handles: Penalize unusual tangent handle lengths at endpoints
        curvature_squared: Penalize high curvature (bending energy)
        curvature_gradient: Penalize rapid curvature changes (wiggliness)
    """
    length_variance: float = 0.0
    polygon_bend: float = 0.0
    anti_bunching: float = 0.0
    end_handles: float = 0.0
    curvature_squared: float = 0.0
    curvature_gradient: float = 0.0

    @property
    def is_enabled(self) -> bool:
        """Return True if any fairness term has nonzero weight."""
        return any([
            self.length_variance > 0,
            self.polygon_bend > 0,
            self.anti_bunching > 0,
            self.end_handles > 0,
            self.curvature_squared > 0,
            self.curvature_gradient > 0,
        ])

    @property
    def total_weight(self) -> float:
        """Return the sum of all weights."""
        return (
            self.length_variance +
            self.polygon_bend +
            self.anti_bunching +
            self.end_handles +
            self.curvature_squared +
            self.curvature_gradient
        )

    def as_tuple(self) -> tuple[float, ...]:
        """Return weights as a tuple for passing to optimization."""
        return (
            self.length_variance,
            self.polygon_bend,
            self.anti_bunching,
            self.end_handles,
            self.curvature_squared,
            self.curvature_gradient,
        )


@dataclass
class FitResult:
    """
    Results from a spline fitting operation.
    
    Contains both the fitted control points and metadata about the
    optimization process.
    
    Attributes:
        section_control_points: List of control point lists, one per curve section
        degree: Spline degree used
        final_error: Final objective function value
        num_iterations: Number of optimizer iterations completed
        num_evaluations: Total objective function evaluations
        elapsed_seconds: Total optimization time
        converged: Whether optimization converged (vs. max iter or cancel)
        early_stopped: Whether optimization stopped due to plateau detection
        cancelled: Whether optimization was cancelled by user
        section_knots: Knot vector of every section; None means open uniform
            knots (the fixed-count optimizers).
        report: Quality report (deviation, curvature, junctions), see
            :func:`spline_fitter.analysis.fit_report`.
    """
    section_control_points: list[PointList]
    degree: int
    final_error: float
    num_iterations: int
    num_evaluations: int
    elapsed_seconds: float
    converged: bool = True
    early_stopped: bool = False
    cancelled: bool = False
    section_knots: list[list[float]] | None = None
    report: dict | None = None

    def __post_init__(self):
        """Ensure control points are stored as lists, not tuples."""
        converted_sections = []
        for section_points in self.section_control_points:
            if isinstance(section_points, tuple):
                converted_sections.append(list(section_points))
            else:
                converted_sections.append(section_points)
        object.__setattr__(self, 'section_control_points', converted_sections)
        if self.section_knots is not None:
            self.section_knots = [[float(k) for k in knots] for knots in self.section_knots]

    def knots(self, section: int) -> np.ndarray:
        """Knot vector of one section (open uniform when none was stored)."""
        from ..spline.spline import create_open_uniform_knots
        if self.section_knots is not None:
            return np.asarray(self.section_knots[section], dtype=np.float64)
        return create_open_uniform_knots(self.degree, len(self.section_control_points[section]))

    def curves(self) -> list:
        """One :class:`~spline_fitter.spline.BSplineCurve2D` per section."""
        from ..spline.spline import BSplineCurve2D
        return [BSplineCurve2D(self.degree, ctrl, self.knots(i))
                for i, ctrl in enumerate(self.section_control_points)]

    @property
    def num_control_points(self) -> int:
        """Distinct control points of the chain (junction points counted once)."""
        return sum(len(c) for c in self.section_control_points) - (len(self.section_control_points) - 1)


@dataclass(frozen=True)
class BoundingBox:
    """
    Axis-aligned bounding box.
    
    Attributes:
        x_min: Minimum x coordinate
        x_max: Maximum x coordinate
        y_min: Minimum y coordinate
        y_max: Maximum y coordinate
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def center(self) -> Point2D:
        """Return the center point of the bounding box."""
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )

    @property
    def width(self) -> float:
        """Return the width (x-extent) of the bounding box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Return the height (y-extent) of the bounding box."""
        return self.y_max - self.y_min

    @property
    def diagonal(self) -> float:
        """Return the diagonal length of the bounding box."""
        return np.hypot(self.width, self.height)

    def scaled(self, factor: float) -> BoundingBox:
        """
        Return a new bounding box scaled around its center.
        
        Args:
            factor: Scale factor (1.0 = no change, 2.0 = double size)
            
        Returns:
            New BoundingBox with scaled dimensions
        """
        cx, cy = self.center
        half_w = self.width / 2.0 * factor
        half_h = self.height / 2.0 * factor
        return BoundingBox(
            x_min=cx - half_w,
            x_max=cx + half_w,
            y_min=cy - half_h,
            y_max=cy + half_h,
        )

    def x_bounds(self) -> Bounds:
        """Return (x_min, x_max) as a tuple."""
        return (self.x_min, self.x_max)

    def y_bounds(self) -> Bounds:
        """Return (y_min, y_max) as a tuple."""
        return (self.y_min, self.y_max)


@dataclass
class IterationRecord:
    """
    Record of a single optimizer iteration.
    
    Used for tracking optimization progress and generating convergence plots.
    Records are created once per optimizer iteration (not per evaluation).
    
    Attributes:
        iteration_number: Optimizer iteration number (1-indexed)
        error_value: Best objective function value at this iteration
        elapsed_seconds: Time elapsed since optimization start
    """
    iteration_number: int
    error_value: float
    elapsed_seconds: float = 0.0


# Backwards compatibility alias
EvaluationRecord = IterationRecord


@dataclass(frozen=True)
class ProjectMetadata:
    """
    Metadata for a spline fitting project file.
    
    Attributes:
        format_version: File format version for compatibility checking
        created: ISO 8601 timestamp when project was created
        modified: ISO 8601 timestamp of last modification
        app_version: Application version that created the file
        platform: OS/platform information
    """
    format_version: str = "1.0.0"
    created: str = ""
    modified: str = ""
    app_version: str = ""
    platform: str = ""


@dataclass
class OptimizationRun:
    """
    Complete record of a single optimization run.
    
    Stores all configuration, results, and history for reproducibility
    and comparison between different optimization approaches.
    
    Attributes:
        run_id: Unique identifier (UUID) for this run
        timestamp: ISO 8601 timestamp when run was executed
        spline_config: Spline configuration used
        optimization_config: Optimizer configuration used
        fairness_weights: Fairness regularization weights used
        result: Optimization result (errors, timing, convergence)
        loss_history: Complete history of all evaluations
        section_control_points: Final fitted control points per section
    """
    run_id: str
    timestamp: str
    spline_config: SplineConfig
    optimization_config: OptimizationConfig
    fairness_weights: FairnessWeights
    result: FitResult
    loss_history: list[IterationRecord]
    section_control_points: list[PointList]


@dataclass
class ProjectFile:
    """
    Complete spline fitting project file.
    
    Contains the reference airfoil data and all optimization runs
    performed on it, enabling session restore and run comparison.
    
    Attributes:
        metadata: Project file metadata (version, timestamps)
        airfoil: Original reference airfoil coordinate data
        optimization_runs: List of all optimization runs in this project
    """
    metadata: ProjectMetadata
    airfoil: AirfoilData
    optimization_runs: list[OptimizationRun] = field(default_factory=list)
