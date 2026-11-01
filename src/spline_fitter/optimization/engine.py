"""
Optimization Engine
===================

Dispatch of a fit to VarPro LM or Direct LSQ (``run_optimization``).

Evaluation tracking and run logging live in their own modules and are
re-exported here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .models import (
    AirfoilData,
    OptimizationConfig,
    OptimizerType,
    SplineConfig,
)
from .optimizers import (
    RawOptimResult,
    get_optimizer,
)
from .run_log import OptimizerErrorLogger  # noqa: F401

# Re-exported so existing ``from .engine import ...`` imports keep working
from .tracking import (  # noqa: F401
    get_evaluation_count,
    increment_evaluation,
    increment_iteration,
    reset_evaluation_tracking,
    reset_iteration_tracking,
)

# =============================================================================
# Optimizer Dispatch
# =============================================================================

def run_optimization(
    config: SplineConfig,
    opt_config: OptimizationConfig,
    airfoil: AirfoilData,
    callback: Callable[[np.ndarray, float], bool] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> RawOptimResult:
    """
    Fit splines to airfoil data with the configured optimizer.

    Args:
        config: Spline configuration
        opt_config: Optimization configuration
        airfoil: Airfoil data to fit
        callback: Optional callback(x, error) returning True to stop
        should_stop: Optional cancel check for optimizers that can stop
            between callbacks (VarPro LM tolerance mode)

    Returns:
        RawOptimResult with the fitted control points
    """
    optimizer_type = opt_config.optimizer
    optimizer = get_optimizer(optimizer_type)

    if optimizer_type == OptimizerType.GN_VARPRO:
        optimizer.config.smoothing_weight = opt_config.smoothing_weight
        return optimizer.fit(
            config,
            opt_config,
            airfoil,
            callback=callback,
            should_stop=should_stop,
        )

    return optimizer.fit(config, opt_config, airfoil)
