"""
Constraint Coefficients for B-Spline Continuity
================================================

This module provides functions for computing the coefficients needed to
enforce derivative continuity constraints at B-spline junctions.

For B-splines with clamped (open uniform) knot vectors, the relationship
between control points and derivatives at the endpoints follows specific
formulas derived from the de Boor recurrence relations.
"""

from __future__ import annotations

import numpy as np


def compute_derivative_constraint_coeffs(
    knots: np.ndarray,
    n_ctrl: int,
    degree: int,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute constraint coefficients for matching k-th derivative at junction.
    
    For B-splines with clamped ends, the relationship between derivatives and
    control points depends on the knot vector. This function computes the correct
    coefficients for d^k C/du^k at u=0 and u=1.
    
    Args:
        knots: The knot vector
        n_ctrl: Number of control points
        degree: Spline degree
        order: Derivative order (1, 2, or 3)
        
    Returns:
        (coef_end, coef_start) where:
        - coef_end: coefficients for P[n-k-1], ..., P[n-1] at u=1
        - coef_start: coefficients for P[0], ..., P[k] at u=0
        
    Example:
        For order=1, returns arrays of length 2:
        - coef_end applies to [P[n-2], P[n-1]]
        - coef_start applies to [P[0], P[1]]
        
        For order=2, returns arrays of length 3:
        - coef_end applies to [P[n-3], P[n-2], P[n-1]]
        - coef_start applies to [P[0], P[1], P[2]]
        
        For order=3, returns arrays of length 4:
        - coef_end applies to [P[n-4], P[n-3], P[n-2], P[n-1]]
        - coef_start applies to [P[0], P[1], P[2], P[3]]
    """
    t = knots
    p = degree
    n = n_ctrl

    if order == 1:
        # d/du at u=0: coef * (P[1] - P[0])
        # d/du at u=1: coef * (P[n-1] - P[n-2])
        denom_start = t[p + 1] - t[1]
        denom_end = t[n + p - 1] - t[n - 1]

        coef_start = np.array([-p / denom_start, p / denom_start])
        coef_end = np.array([-p / denom_end, p / denom_end])

        return coef_end, coef_start

    elif order == 2:
        # For d²/du² we use the recurrence through Q coefficients
        # At u=0: R_0 = beta*P[0] + (-alpha-beta)*P[1] + alpha*P[2]
        # At u=1: R_{n-3} = delta*P[-3] + (-gamma-delta)*P[-2] + gamma*P[-1]

        # Start (u=0) coefficients
        denom_Q0 = t[p + 1] - t[1]
        denom_Q1 = t[p + 2] - t[2]
        denom_R0 = t[p + 1] - t[2]

        alpha = (p - 1) * p / (denom_R0 * denom_Q1)
        beta = (p - 1) * p / (denom_R0 * denom_Q0)

        coef_start = np.array([beta, -alpha - beta, alpha])

        # End (u=1) coefficients
        denom_Qnm2 = t[n + p - 1] - t[n - 1]
        denom_Qnm3 = t[n + p - 2] - t[n - 2]
        denom_Rnm3 = t[n + p - 1] - t[n - 1]

        gamma = (p - 1) * p / (denom_Rnm3 * denom_Qnm2)
        delta = (p - 1) * p / (denom_Rnm3 * denom_Qnm3)

        coef_end = np.array([delta, -gamma - delta, gamma])

        return coef_end, coef_start

    elif order == 3:
        # For d³/du³ we extend the recurrence one more level
        # This requires computing S coefficients from R coefficients
        # S_i = (p-2) * (R_{i+1} - R_i) / (t_{i+p+1} - t_{i+3})

        # For simplicity, we compute this numerically by evaluating
        # the constraint that makes d³ equal at the junction

        # At u=0, d³ depends on P[0], P[1], P[2], P[3]
        # At u=1, d³ depends on P[n-4], P[n-3], P[n-2], P[n-1]

        # We need to derive the coefficients through the recurrence:
        # R_i depends on Q_{i+1}, Q_i
        # S_i depends on R_{i+1}, R_i

        # Start (u=0):
        denom_Q0 = t[p + 1] - t[1]
        denom_Q1 = t[p + 2] - t[2]
        denom_Q2 = t[p + 3] - t[3]
        denom_R0 = t[p + 1] - t[2]
        denom_R1 = t[p + 2] - t[3]
        denom_S0 = t[p + 1] - t[3]

        # Q_i = p / denom_Qi * DP_i
        cQ0 = p / denom_Q0
        cQ1 = p / denom_Q1
        cQ2 = p / denom_Q2

        # R_0 = (p-1)/denom_R0 * (Q_1 - Q_0) = (p-1)/denom_R0 * (cQ1*DP_1 - cQ0*DP_0)
        # R_1 = (p-1)/denom_R1 * (Q_2 - Q_1) = (p-1)/denom_R1 * (cQ2*DP_2 - cQ1*DP_1)
        cR0_dp0 = -(p - 1) / denom_R0 * cQ0
        cR0_dp1 = (p - 1) / denom_R0 * cQ1
        cR1_dp1 = -(p - 1) / denom_R1 * cQ1
        cR1_dp2 = (p - 1) / denom_R1 * cQ2

        # S_0 = (p-2)/denom_S0 * (R_1 - R_0)
        # S_0 = (p-2)/denom_S0 * (cR1_dp1*DP_1 + cR1_dp2*DP_2 - cR0_dp0*DP_0 - cR0_dp1*DP_1)
        #     = (p-2)/denom_S0 * (-cR0_dp0*DP_0 + (cR1_dp1 - cR0_dp1)*DP_1 + cR1_dp2*DP_2)

        scale_S0 = (p - 2) / denom_S0
        cS0_dp0 = scale_S0 * (-cR0_dp0)
        cS0_dp1 = scale_S0 * (cR1_dp1 - cR0_dp1)
        cS0_dp2 = scale_S0 * cR1_dp2

        # Now convert from DP coefficients to P coefficients
        # DP_0 = P_1 - P_0, DP_1 = P_2 - P_1, DP_2 = P_3 - P_2
        # S_0 = cS0_dp0*(P_1-P_0) + cS0_dp1*(P_2-P_1) + cS0_dp2*(P_3-P_2)
        #     = -cS0_dp0*P_0 + (cS0_dp0 - cS0_dp1)*P_1 + (cS0_dp1 - cS0_dp2)*P_2 + cS0_dp2*P_3

        coef_start = np.array([
            -cS0_dp0,
            cS0_dp0 - cS0_dp1,
            cS0_dp1 - cS0_dp2,
            cS0_dp2
        ])

        # End (u=1):
        denom_Qnm2 = t[n + p - 1] - t[n - 1]
        denom_Qnm3 = t[n + p - 2] - t[n - 2]
        denom_Qnm4 = t[n + p - 3] - t[n - 3]
        denom_Rnm3 = t[n + p - 1] - t[n - 1]
        denom_Rnm4 = t[n + p - 2] - t[n - 2]
        denom_Snm4 = t[n + p - 1] - t[n - 1]

        cQnm2 = p / denom_Qnm2
        cQnm3 = p / denom_Qnm3
        cQnm4 = p / denom_Qnm4

        # R_{n-3} = (p-1)/denom_Rnm3 * (Q_{n-2} - Q_{n-3})
        # R_{n-4} = (p-1)/denom_Rnm4 * (Q_{n-3} - Q_{n-4})
        cRnm3_dpnm2 = (p - 1) / denom_Rnm3 * cQnm2
        cRnm3_dpnm3 = -(p - 1) / denom_Rnm3 * cQnm3
        cRnm4_dpnm3 = (p - 1) / denom_Rnm4 * cQnm3
        cRnm4_dpnm4 = -(p - 1) / denom_Rnm4 * cQnm4

        # S_{n-4} = (p-2)/denom_Snm4 * (R_{n-3} - R_{n-4})
        scale_Snm4 = (p - 2) / denom_Snm4
        cSnm4_dpnm4 = scale_Snm4 * (-cRnm4_dpnm4)
        cSnm4_dpnm3 = scale_Snm4 * (cRnm3_dpnm3 - cRnm4_dpnm3)
        cSnm4_dpnm2 = scale_Snm4 * cRnm3_dpnm2

        # Convert to P coefficients
        # DP_{n-4} = P[-3] - P[-4], DP_{n-3} = P[-2] - P[-3], DP_{n-2} = P[-1] - P[-2]
        # S_{n-4} = cSnm4_dpnm4*(P[-3]-P[-4]) + cSnm4_dpnm3*(P[-2]-P[-3]) + cSnm4_dpnm2*(P[-1]-P[-2])

        coef_end = np.array([
            -cSnm4_dpnm4,
            cSnm4_dpnm4 - cSnm4_dpnm3,
            cSnm4_dpnm3 - cSnm4_dpnm2,
            cSnm4_dpnm2
        ])

        return coef_end, coef_start

    else:
        raise ValueError(f"Derivative order {order} not supported (must be 1, 2, or 3)")
