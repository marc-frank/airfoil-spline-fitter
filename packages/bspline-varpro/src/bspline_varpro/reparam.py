"""Monotone softplus reparameterization of the chain parameters.

The chain parameters u_0 = 0 < u_1 < ... < u_{n-1} = S of all data points are
written as cumulative sums of softplus(eta) scaled to end at S, so any
unconstrained eta vector yields strictly increasing parameters with fixed end
values. This lets an unconstrained Levenberg-Marquardt solver optimize the
parameters, and lets a point cross a junction (an integer u) as it slides
along the chain.
"""

from __future__ import annotations

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


def t_to_eta(u: np.ndarray) -> np.ndarray:
    """Eta vector (one shorter than ``u``) reproducing the given chain parameters."""
    return inverse_softplus(np.maximum(np.diff(np.asarray(u, dtype=np.float64)), 1e-15))


def eta_to_t(eta: np.ndarray, span: float) -> np.ndarray:
    """Chain parameters from 0 to ``span`` for an eta vector."""
    # Huge LM steps must not overflow the cumulative sum.
    eta = np.clip(np.nan_to_num(eta, nan=0.0), -700.0, 700.0)
    cumsum = np.cumsum(softplus(eta))
    u = np.empty(len(eta) + 1)
    u[0] = 0.0
    u[1:] = span * cumsum / cumsum[-1]
    return u


def eta_jacobian(eta: np.ndarray, u: np.ndarray, span: float) -> np.ndarray:
    """``du/deta``, shape (n, n - 1).

    With u_k = span * cumsum(softplus(eta))_k / T:
    ``du_k/deta_j = span * sigmoid(eta_j) / T * ([j < k] - u_k / span)``.
    """
    total = np.sum(softplus(eta))
    indicator = np.tri(len(u), len(eta), k=-1)
    return (span * sigmoid(eta) / total)[np.newaxis, :] * (indicator - u[:, np.newaxis] / span)
