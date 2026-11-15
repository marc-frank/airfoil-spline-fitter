"""
Evaluation Tracking
===================

Module-level evaluation and iteration counters shared by the objective
function and the optimizer adapters.
"""

from __future__ import annotations

# =============================================================================
# Module State for Evaluation Tracking
# =============================================================================

_evaluation_count: int = 0
_iteration_count: int = 0


def reset_iteration_tracking() -> None:
    """Reset the global evaluation counter and iteration counter."""
    global _evaluation_count, _iteration_count
    _evaluation_count = 0
    _iteration_count = 0


def get_evaluation_count() -> int:
    """Return the current evaluation count."""
    return _evaluation_count


def increment_evaluation() -> int:
    """Increment and return the evaluation counter (called by objective function)."""
    global _evaluation_count
    _evaluation_count += 1
    return _evaluation_count


def increment_iteration() -> int:
    """Increment and return the iteration counter."""
    global _iteration_count
    _iteration_count += 1
    return _iteration_count
