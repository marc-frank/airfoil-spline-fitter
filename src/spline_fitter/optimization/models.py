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

from collections.abc import Sequence
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
    VarPro LM optimizer supports geometric continuity; Direct LSQ
    always uses the parametric form.
    
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
    def order(self) -> int:
        """The number of derivatives linked at a junction (0 to 3)."""
        return int(self.name[1])


def minimum_control_points(junction_orders: Sequence[int], degree: int) -> int:
    """Fewest control points per section for a chain of sections joined
    with the given continuity orders (one per junction): more than the
    degree, and enough that the conditions at both ends of a section do not
    overlap. A section between two junctions of order k needs 2k + 2, an end
    section k + 2. The rule of ``bspline_varpro.constraints``; the other
    optimizers lay the control points out the same way."""
    ends = [0, *junction_orders, 0]
    return max(degree + 1, *(a + b + 2 for a, b in zip(ends, ends[1:])))


class OptimizerType(Enum):
    """
    Available fitting algorithms.

    Attributes:
        DIRECT_LSQ: One constrained least-squares solve at fixed parameters
        GN_VARPRO: Levenberg-Marquardt variable projection (bspline-varpro)
    """
    # Direct LSQ (single-step, no iteration)
    DIRECT_LSQ = "direct-lsq"

    # Levenberg-Marquardt variable projection (bspline-varpro library)
    GN_VARPRO = "gn-varpro"

    @property
    def is_algebraic(self) -> bool:
        """Return True if this is an algebraic (non-iterative) method."""
        return self == OptimizerType.DIRECT_LSQ

    @property
    def is_iterative_algebraic(self) -> bool:
        """Return True if this uses iterative algebraic (LSQ) steps."""
        return self == OptimizerType.GN_VARPRO

    @property
    def display_name(self) -> str:
        """Return a human-readable name for the optimizer."""
        names = {
            OptimizerType.GN_VARPRO: "VarPro LM",
            OptimizerType.DIRECT_LSQ: "Direct LSQ",
        }
        return names.get(self, self.value)

    @property
    def tooltip(self) -> str:
        """Return a tooltip description of the method."""
        tooltips = {
            OptimizerType.DIRECT_LSQ: (
                "Direct LSQ (single-step, no iteration).\n"
                "One constrained least-squares solve, with fixed trailing-edge\n"
                "end points and junctions fixed or free (Advanced).\n"
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
        geometric_continuity: Link junction derivatives geometrically (G^n)
            instead of parametrically (C^n). VarPro LM only.
        junction_continuity: Optional per-junction overrides such as
            ``("G2", "C1", "G3")``, one per junction (VarPro LM only).
    """
    degree: int = 5
    num_control_points: int = 25
    num_sections: int = 2
    continuity: ContinuityMode = ContinuityMode.G1
    geometric_continuity: bool = False
    junction_continuity: tuple[str, ...] = ()

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_sections < 1:
            raise ValueError(
                f"num_sections must be >= 1, got {self.num_sections}"
            )
        object.__setattr__(self, "junction_continuity",
                           tuple(s.strip().upper() for s in self.junction_continuity))
        # The continuity applies to the junctions only when the per-junction
        # entries do not override it (one section has no junction at all)
        if (self.num_sections > 1 and not self.junction_continuity
                and self.degree < self.continuity.minimum_degree):
            raise ValueError(
                f"{self.continuity.name} continuity requires degree >= "
                f"{self.continuity.minimum_degree}, got {self.degree}"
            )
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
    def minimum_control_points(self) -> int:
        """Fewest control points per section the degree and the continuity
        of every junction allow."""
        return minimum_control_points([int(s[1]) for s in self.junction_specs()], self.degree)


@dataclass(frozen=True)
class OptimizationConfig:
    """
    Configuration for the optimization process.

    Attributes:
        optimizer: Which optimization algorithm to use
        max_iterations: Unused: no optimizer limits its evaluations or
            iterations, every fit runs until it converges. Kept so stored
            projects and settings that carry it still load; left out of
            comparisons, so a stored run's settings equal the same settings
            applied in the sidebar whatever limit the run recorded.
        early_stop_patience: Iterations without improvement before stopping
        relative_tolerance: Relative improvement threshold for early stopping
        optimize_le_position: If True, every junction's (x, y) is fitted too,
            anywhere (the split points are only where they start)
        smoothing_factor: Smoothing parameter for algebraic methods (0 = interpolation)
        smoothing_weight: Bending energy smoothing weight for VarPro LM (0 = disabled).
            Weight of the integrated squared parameter second derivative.
            The same penalty is included in the inner solve and outer objective.
        custom_parameters: Optional parameter values for the data points, kept
            fixed by Direct LSQ: chain parameters (one array from 0 to the
            section count, a shared junction point once; see
            bspline_varpro.chain) or one array from 0 to 1 per section. If
            None, `parameterization` is used.
        parameterization: Direct LSQ only. Where the data points' curve
            parameters start, within each section: "centripetal" (default) or
            "arc_length". They then move to the closest chain points.
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
    optimizer: OptimizerType = OptimizerType.GN_VARPRO
    # Limits nothing, so it does not tell two configurations apart
    max_iterations: int = field(default=10000, compare=False)
    smoothing_factor: float | None = None
    early_stop_patience: int = 50
    relative_tolerance: float = 1e-6
    optimize_le_position: bool = False
    smoothing_weight: float = 1e-3
    custom_parameters: list[np.ndarray] | None = None
    parameterization: str = "centripetal"
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
        if self.parameterization not in ("centripetal", "arc_length"):
            raise ValueError(
                "parameterization must be 'centripetal' or 'arc_length', "
                f"got {self.parameterization!r}")


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
        rms_error: RMS distance of the data points to their curve points.
            ``final_error`` is the max deviation for a tolerance goal, so runs
            compare by :attr:`comparison_error` instead.
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
    rms_error: float | None = None

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
    def comparison_error(self) -> float:
        """The error runs compare by, the same measure for every optimizer
        and goal: the report's RMS orthogonal deviation, else the RMS error
        (``final_error`` only for runs that recorded neither)."""
        if self.report and self.report.get("rms_deviation") is not None:
            return float(self.report["rms_deviation"])
        if self.rms_error is not None:
            return self.rms_error
        return self.final_error

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
        result: Optimization result (errors, timing, convergence)
        loss_history: Complete history of all evaluations
        section_control_points: Final fitted control points per section
        number: Number of the run in its project, in the order the runs
            were done; kept when other runs are deleted. 0 until numbered
            (see ``project_io.number_runs``).
    """
    run_id: str
    timestamp: str
    spline_config: SplineConfig
    optimization_config: OptimizationConfig
    result: FitResult
    loss_history: list[IterationRecord]
    section_control_points: list[PointList]
    number: int = 0


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
