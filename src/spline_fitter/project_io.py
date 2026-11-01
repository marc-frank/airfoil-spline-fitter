"""
Project File I/O
================

Functions for saving and loading complete spline fitting project files.

Project files (.asfproj) store:
    - Original reference airfoil coordinates
    - All optimization runs with full configuration
    - Loss history for every evaluation
    - Final control points for each fitted spline

File format: JSON with the following structure:
{
    "format_version": "1.0.0",
    "metadata": { ... },
    "airfoil": { ... },
    "optimization_runs": [ ... ]
}
"""

from __future__ import annotations

import json
import logging
import platform
import uuid
from datetime import datetime
from typing import Any

from .optimization.models import (
    AirfoilData,
    ContinuityMode,
    FitResult,
    IterationRecord,
    OptimizationConfig,
    OptimizationRun,
    OptimizerType,
    PointList,
    ProjectFile,
    ProjectMetadata,
    SplineConfig,
)

# Import version from package
try:
    from . import __version__
except ImportError:
    __version__ = "unknown"


# =============================================================================
# Serialization Helpers
# =============================================================================

def _enum_to_str(obj: Any) -> str:
    """Convert an enum to its value string."""
    if hasattr(obj, 'value'):
        return obj.value
    return str(obj)


def _continuity_mode_from_str(s: str) -> ContinuityMode:
    """Convert string to ContinuityMode enum."""
    mapping = {
        'G0': ContinuityMode.G0,
        'G1': ContinuityMode.G1,
        'G2': ContinuityMode.G2,
        'G3': ContinuityMode.G3,
    }
    return mapping.get(s.upper(), ContinuityMode.G1)


# Logger for migration warnings
_logger = logging.getLogger(__name__)


_RETIRED_OPTIMIZERS = {
    # Its coarse-to-fine role is taken over by VarPro LM tolerance mode.
    "hierarchical-lsq": OptimizerType.GN_VARPRO,
    # Direct LSQ with arc-length parameters, and two methods VarPro LM supersedes.
    "constrained-lsq": OptimizerType.DIRECT_LSQ,
    "orthogonal-lsq": OptimizerType.GN_VARPRO,
    "param-opt": OptimizerType.GN_VARPRO,
    # The general-purpose scipy optimizers, local and global.
    **dict.fromkeys(("bfgs", "slsqp", "l-bfgs-b", "cg", "tnc", "powell", "nelder-mead",
                     "cobyla", "de", "da", "bh"), OptimizerType.GN_VARPRO),
}


def _optimizer_type_from_str(s: str) -> OptimizerType:
    """Convert string to OptimizerType enum."""
    for opt in OptimizerType:
        if opt.value == s.lower():
            return opt
    if s.lower() in _RETIRED_OPTIMIZERS:
        _logger.warning("Optimizer %r was retired; showing the run as %s.", s,
                        _RETIRED_OPTIMIZERS[s.lower()].display_name)
        return _RETIRED_OPTIMIZERS[s.lower()]
    return OptimizerType.GN_VARPRO


# =============================================================================
# Encoding (Python -> JSON)
# =============================================================================

def _encode_airfoil(airfoil: AirfoilData) -> dict[str, Any]:
    """Encode AirfoilData to JSON-serializable dict."""
    return {
        "coordinates": list(airfoil.coordinates),
        "sections": [list(section) for section in airfoil.sections],
        "junction_indices": list(airfoil.junction_indices),
        "num_sections": airfoil.num_sections,
        "filename": airfoil.filename,
        "name": airfoil.name,
        "precision": airfoil.precision,
    }


def _encode_spline_config(config: SplineConfig) -> dict[str, Any]:
    """Encode SplineConfig to JSON-serializable dict."""
    return {
        "degree": config.degree,
        "num_control_points": config.num_control_points,
        "num_sections": config.num_sections,
        "continuity": config.continuity.name,
        "geometric_continuity": config.geometric_continuity,
        "junction_continuity": list(config.junction_continuity),
    }


def _encode_optimization_config(config: OptimizationConfig) -> dict[str, Any]:
    """Encode OptimizationConfig to JSON-serializable dict."""
    return {
        "optimizer": config.optimizer.value,
        "max_iterations": config.max_iterations,
        "smoothing_factor": config.smoothing_factor,
        "smoothing_weight": config.smoothing_weight,
        "early_stop_patience": config.early_stop_patience,
        "relative_tolerance": config.relative_tolerance,
        "optimize_le_position": config.optimize_le_position,
        "parameterization": config.parameterization,
        "tolerance": config.tolerance,
        "max_points_per_section": config.max_points_per_section,
        "knot_placement": config.knot_placement,
        "grow_control_points": config.grow_control_points,
    }


def _encode_fit_result(result: FitResult) -> dict[str, Any]:
    """Encode FitResult to JSON-serializable dict (without control points)."""
    return {
        "degree": result.degree,
        "final_error": result.final_error,
        "num_iterations": result.num_iterations,
        "num_evaluations": result.num_evaluations,
        "elapsed_seconds": result.elapsed_seconds,
        "converged": result.converged,
        "early_stopped": result.early_stopped,
        "cancelled": result.cancelled,
        "section_knots": result.section_knots,
        "report": result.report,
    }


def _encode_iteration_record(record: IterationRecord) -> dict[str, Any]:
    """Encode IterationRecord to JSON-serializable dict."""
    return {
        "iter": record.iteration_number,
        "error": record.error_value,
        "time": record.elapsed_seconds,
    }


def _encode_optimization_run(run: OptimizationRun) -> dict[str, Any]:
    """Encode OptimizationRun to JSON-serializable dict."""
    return {
        "run_id": run.run_id,
        "timestamp": run.timestamp,
        "spline_config": _encode_spline_config(run.spline_config),
        "optimization_config": _encode_optimization_config(run.optimization_config),
        "result": _encode_fit_result(run.result),
        "loss_history": [_encode_iteration_record(r) for r in run.loss_history],
        "section_control_points": [
            [[float(x), float(y)] for x, y in section]
            for section in run.section_control_points
        ],
    }


def _encode_metadata(metadata: ProjectMetadata) -> dict[str, Any]:
    """Encode ProjectMetadata to JSON-serializable dict."""
    return {
        "format_version": metadata.format_version,
        "created": metadata.created,
        "modified": metadata.modified,
        "app_version": metadata.app_version,
        "platform": metadata.platform,
    }


def _encode_project(project: ProjectFile) -> dict[str, Any]:
    """Encode ProjectFile to JSON-serializable dict."""
    return {
        "format_version": project.metadata.format_version,
        "metadata": _encode_metadata(project.metadata),
        "airfoil": _encode_airfoil(project.airfoil),
        "optimization_runs": [_encode_optimization_run(r) for r in project.optimization_runs],
    }


# =============================================================================
# Decoding (JSON -> Python)
# =============================================================================

def _decode_airfoil(data: dict[str, Any]) -> AirfoilData:
    """Decode AirfoilData from JSON dict."""
    return AirfoilData(
        coordinates=tuple(tuple(p) for p in data["coordinates"]),
        sections=tuple(tuple(tuple(p) for p in section) for section in data["sections"]),
        junction_indices=tuple(data["junction_indices"]),
        num_sections=data["num_sections"],
        filename=data.get("filename", ""),
        name=data.get("name", ""),
        precision=data.get("precision", 6),
    )


def _decode_spline_config(data: dict[str, Any]) -> SplineConfig:
    """Decode SplineConfig from JSON dict.
    
    Old projects stored a ``continuity_type`` field ("GEOMETRIC" or
    "PARAMETRIC"); it maps to ``geometric_continuity``.
    """
    geometric = data.get("geometric_continuity")
    if geometric is None:
        geometric = str(data.get("continuity_type", "")).upper() == "GEOMETRIC"
    return SplineConfig(
        degree=data["degree"],
        num_control_points=data["num_control_points"],
        num_sections=data.get("num_sections", 2),
        continuity=_continuity_mode_from_str(data["continuity"]),
        geometric_continuity=bool(geometric),
        junction_continuity=tuple(data.get("junction_continuity", ())),
    )


def _decode_optimization_config(data: dict[str, Any]) -> OptimizationConfig:
    """Decode OptimizationConfig from JSON dict."""
    return OptimizationConfig(
        optimizer=_optimizer_type_from_str(data["optimizer"]),
        max_iterations=data.get("max_iterations", 10000),
        smoothing_factor=data.get("smoothing_factor"),
        smoothing_weight=data.get("smoothing_weight", 1e-3),
        early_stop_patience=data.get("early_stop_patience", 50),
        relative_tolerance=data.get("relative_tolerance", 1e-6),
        optimize_le_position=data.get("optimize_le_position", False),
        parameterization=data.get("parameterization", "centripetal"),
        tolerance=data.get("tolerance"),
        max_points_per_section=data.get("max_points_per_section"),
        # Projects from before these settings: fixed counts used uniform knots,
        # tolerance mode grew the count with freely placed knots.
        knot_placement=data.get("knot_placement",
                                "uniform" if data.get("tolerance") is None else "free"),
        grow_control_points=data.get("grow_control_points", data.get("tolerance") is not None),
        # Older projects may also contain "fairing", "metric" and "seed"; they are ignored.
    )


def _decode_fit_result(data: dict[str, Any], section_control_points: list[PointList]) -> FitResult:
    """Decode FitResult from JSON dict."""
    return FitResult(
        section_control_points=section_control_points,
        degree=data["degree"],
        final_error=data["final_error"],
        num_iterations=data["num_iterations"],
        num_evaluations=data["num_evaluations"],
        elapsed_seconds=data["elapsed_seconds"],
        converged=data.get("converged", True),
        early_stopped=data.get("early_stopped", False),
        cancelled=data.get("cancelled", False),
        section_knots=data.get("section_knots"),
        report=data.get("report"),
    )


def _decode_iteration_record(data: dict[str, Any]) -> IterationRecord:
    """Decode IterationRecord from JSON dict.
    
    Handles both old format (with 'eval' field) and new format (without).
    """
    return IterationRecord(
        iteration_number=data["iter"],
        error_value=data["error"],
        elapsed_seconds=data.get("time", 0.0),
    )


def _decode_optimization_run(data: dict[str, Any]) -> OptimizationRun:
    """Decode OptimizationRun from JSON dict."""
    # Decode control points first (needed for FitResult)
    section_control_points = [
        [(float(p[0]), float(p[1])) for p in section]
        for section in data["section_control_points"]
    ]

    return OptimizationRun(
        run_id=data["run_id"],
        timestamp=data["timestamp"],
        spline_config=_decode_spline_config(data["spline_config"]),
        optimization_config=_decode_optimization_config(data["optimization_config"]),
        result=_decode_fit_result(data["result"], section_control_points),
        loss_history=[_decode_iteration_record(r) for r in data["loss_history"]],
        section_control_points=section_control_points,
    )


def _decode_metadata(data: dict[str, Any]) -> ProjectMetadata:
    """Decode ProjectMetadata from JSON dict."""
    return ProjectMetadata(
        format_version=data.get("format_version", "1.0.0"),
        created=data.get("created", ""),
        modified=data.get("modified", ""),
        app_version=data.get("app_version", ""),
        platform=data.get("platform", ""),
    )


def _decode_project(data: dict[str, Any]) -> ProjectFile:
    """Decode ProjectFile from JSON dict."""
    return ProjectFile(
        metadata=_decode_metadata(data["metadata"]),
        airfoil=_decode_airfoil(data["airfoil"]),
        optimization_runs=[_decode_optimization_run(r) for r in data.get("optimization_runs", [])],
    )


# =============================================================================
# Public API
# =============================================================================

def create_project_metadata(created: str | None = None) -> ProjectMetadata:
    """
    Create new project metadata with current timestamp and system info.
    
    Args:
        created: Optional ISO timestamp for creation time (uses now if None)
        
    Returns:
        ProjectMetadata with current system information
    """
    now = datetime.now().isoformat()
    return ProjectMetadata(
        format_version="1.0.0",
        created=created or now,
        modified=now,
        app_version=__version__,
        platform=f"{platform.system()} {platform.release()}",
    )


def create_optimization_run(
    spline_config: SplineConfig,
    optimization_config: OptimizationConfig,
    result: FitResult,
    loss_history: list[IterationRecord],
) -> OptimizationRun:
    """
    Create a new optimization run record.
    
    Args:
        spline_config: Spline configuration used
        optimization_config: Optimizer configuration used
        result: Optimization result
        loss_history: Iteration history (one record per optimizer iteration)
        
    Returns:
        OptimizationRun with unique ID and current timestamp
    """
    return OptimizationRun(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        spline_config=spline_config,
        optimization_config=optimization_config,
        result=result,
        loss_history=loss_history,
        section_control_points=result.section_control_points,
    )


def create_project(airfoil: AirfoilData) -> ProjectFile:
    """
    Create a new project file for an airfoil.
    
    Args:
        airfoil: The reference airfoil data
        
    Returns:
        Empty ProjectFile ready for optimization runs
    """
    return ProjectFile(
        metadata=create_project_metadata(),
        airfoil=airfoil,
        optimization_runs=[],
    )


def save_project(project: ProjectFile, filepath: str, indent: int = 2) -> None:
    """
    Save a project file to disk.
    
    Args:
        project: The project to save
        filepath: Destination path (should end with .splineproj)
        indent: JSON indentation level (default 2, use None for compact)
        
    Raises:
        IOError: If the file cannot be written
    """
    # Update modified timestamp
    updated_metadata = ProjectMetadata(
        format_version=project.metadata.format_version,
        created=project.metadata.created,
        modified=datetime.now().isoformat(),
        app_version=project.metadata.app_version,
        platform=project.metadata.platform,
    )
    project = ProjectFile(
        metadata=updated_metadata,
        airfoil=project.airfoil,
        optimization_runs=project.optimization_runs,
    )

    data = _encode_project(project)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_project(filepath: str) -> ProjectFile:
    """
    Load a project file from disk.
    
    Args:
        filepath: Path to the project file
        
    Returns:
        Loaded ProjectFile
        
    Raises:
        IOError: If the file cannot be read
        ValueError: If the file format is invalid
    """
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)

    # Version check
    version = data.get("format_version", "1.0.0")
    major_version = int(version.split(".")[0])
    if major_version > 1:
        raise ValueError(
            f"Project file version {version} is not supported. "
            f"Please update the application."
        )

    return _decode_project(data)


def get_run_summary(run: OptimizationRun) -> str:
    """
    Generate a human-readable summary of an optimization run.
    
    Args:
        run: The optimization run to summarize
        
    Returns:
        Multi-line summary string
    """
    cfg = run.spline_config
    opt = run.optimization_config
    res = run.result

    lines = [
        f"Run ID: {run.run_id[:8]}...",
        f"Time: {run.timestamp}",
        f"Optimizer: {opt.optimizer.display_name}",
        f"Degree: {cfg.degree}, Control Points: "
        f"{res.num_control_points if opt.tolerance else cfg.num_control_points}",
        f"Continuity: {', '.join(cfg.junction_specs()) or cfg.continuity.name}",
        *([f"Tolerance: {opt.tolerance:.3e}"] if opt.tolerance else []),
        f"Final Error: {res.final_error:.6e}",
        f"Iterations: {res.num_iterations}, Evaluations: {res.num_evaluations}",
        f"Time: {res.elapsed_seconds:.2f}s",
        f"Converged: {res.converged}, Early Stopped: {res.early_stopped}",
    ]
    return "\n".join(lines)
