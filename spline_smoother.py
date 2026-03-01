"""Cubic B-Spline path smoother for RRT planners.

Fits a cubic B-spline through the pruned/shortcut waypoints to produce
a trajectory with C² continuity (smooth position, velocity, *and*
acceleration).  This is pure math — **no physics-engine calls** — so it
runs in < 0.01 s regardless of obstacle count.

The spline implicitly minimises

    J = ∫ ‖q̈(t)‖² dt

because the natural cubic spline is the minimum-energy interpolant.

Usage::

    from panda_rrt.spline_smoother import SplineSmoother
    smooth_path = SplineSmoother.smooth(pruned_waypoints, n_points=60)
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.interpolate import CubicSpline

from panda_rrt.common_utils.robot_constants import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
)


class SplineSmoother:
    """Lightweight cubic B-spline smoother — no collision queries."""

    @staticmethod
    def smooth(
        waypoints: List[np.ndarray],
        n_points: int = 60,
        bc_type: str = "clamped",
    ) -> List[np.ndarray]:
        """Fit a cubic spline through *waypoints* and resample.

        Parameters
        ----------
        waypoints : list of np.ndarray
            Ordered joint-space waypoints (at least 2).
        n_points : int
            Number of output samples along the spline.
        bc_type : str
            Boundary condition for ``scipy.interpolate.CubicSpline``.
            ``"clamped"`` pins zero velocity at start/goal (smooth stop).
            ``"natural"`` gives zero second-derivative at boundaries.

        Returns
        -------
        list of np.ndarray
            *n_points* evenly-spaced configurations along the spline,
            clamped to joint limits.
        """
        if len(waypoints) <= 1:
            return list(waypoints)
        if len(waypoints) == 2:
            return SplineSmoother._lerp(waypoints[0], waypoints[1], n_points)

        # Stack waypoints → (N, dof) matrix
        Q = np.array(waypoints)  # shape (N, 7)

        # Parameterise by cumulative chord length
        dists = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        t = np.concatenate(([0.0], np.cumsum(dists)))
        t /= t[-1]                 # normalise to [0, 1]

        # Fit one cubic spline per joint (vectorised via CubicSpline)
        cs = CubicSpline(t, Q, bc_type=bc_type, axis=0)

        # Sample uniformly in normalised arc-length
        t_sample = np.linspace(0.0, 1.0, n_points)
        Q_smooth = cs(t_sample)    # shape (n_points, 7)

        # Clamp to joint limits
        Q_smooth = np.clip(Q_smooth, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)

        return [Q_smooth[i] for i in range(n_points)]

    @staticmethod
    def integrated_acceleration_cost(
        waypoints: List[np.ndarray],
        n_eval: int = 200,
        bc_type: str = "clamped",
    ) -> float:
        r"""Compute  J = ∫₀¹ ‖q̈(t)‖² dt  for the spline through *waypoints*.

        Useful for comparing trajectories: lower = smoother.
        """
        if len(waypoints) < 3:
            return 0.0

        Q = np.array(waypoints)
        dists = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        t = np.concatenate(([0.0], np.cumsum(dists)))
        t /= t[-1]

        cs = CubicSpline(t, Q, bc_type=bc_type, axis=0)
        t_eval = np.linspace(0.0, 1.0, n_eval)
        qddot = cs(t_eval, 2)             # second derivative, (n_eval, 7)
        integrand = np.sum(qddot ** 2, axis=1)
        # Trapezoidal integration
        dt = 1.0 / (n_eval - 1)
        return float(np.trapz(integrand, dx=dt))

    # ── helpers ─────────────────────────────────────

    @staticmethod
    def _lerp(
        q0: np.ndarray, q1: np.ndarray, n: int,
    ) -> List[np.ndarray]:
        """Linear interpolation between two configs."""
        return [
            np.clip(q0 + t * (q1 - q0), PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)
            for t in np.linspace(0.0, 1.0, n)
        ]
