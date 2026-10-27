"""Monotone softplus reparameterization of per-section data parameters.

Each section's parameters t_0 = 0 < t_1 < ... < t_{n-1} = 1 are written as
normalized cumulative sums of softplus(eta), so any unconstrained eta vector
yields strictly increasing parameters with fixed end values. This lets an
unconstrained Levenberg-Marquardt solver optimize the parameters.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(1 + exp(x))``."""
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -500.0, 20.0))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable ``1 / (1 + exp(-x))``, the derivative of softplus."""
    exp_neg = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + exp_neg), exp_neg / (1.0 + exp_neg))


def inverse_softplus(y: np.ndarray) -> np.ndarray:
    """``log(exp(y) - 1)`` for y > 0 (clamped to 1e-15)."""
    y_safe = np.maximum(y, 1e-15)
    return np.where(y_safe > 20.0, y_safe, np.log(np.expm1(np.clip(y_safe, 1e-15, 20.0))))


def t_to_eta(t_sections: Sequence[np.ndarray]) -> np.ndarray:
    """Flat eta vector (length ``sum(n_s - 1)``) reproducing the given parameters."""
    return np.concatenate([
        inverse_softplus(np.maximum(np.diff(t_sec), 1e-15)) for t_sec in t_sections
    ])


def eta_to_t(
    eta: np.ndarray, section_sizes: Sequence[int],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Map a flat eta vector to ``(t_flat, t_sections)``."""
    t_sections = []
    offset = 0
    # Huge LM steps must not overflow the cumulative sum.
    eta = np.clip(np.nan_to_num(eta, nan=0.0), -700.0, 700.0)
    for n_s in section_sizes:
        cumsum = np.cumsum(softplus(eta[offset:offset + n_s - 1]))
        offset += n_s - 1
        t_sec = np.empty(n_s)
        t_sec[0] = 0.0
        t_sec[1:] = cumsum / cumsum[-1]
        t_sections.append(t_sec)
    return np.concatenate(t_sections), t_sections


def eta_jacobian_block(eta_sec: np.ndarray, t_sec: np.ndarray) -> np.ndarray:
    """``dt/deta`` for one section, shape (n_s, n_s - 1).

    With t_k = cumsum(softplus(eta))_k / S:
    ``dt_k/deta_j = sigmoid(eta_j) / S * ([j < k] - t_k)``.
    """
    total = np.sum(softplus(eta_sec))
    indicator = np.tri(len(t_sec), len(eta_sec), k=-1)
    return (sigmoid(eta_sec) / total)[np.newaxis, :] * (indicator - t_sec[:, np.newaxis])
