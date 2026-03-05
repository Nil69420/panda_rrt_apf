r"""Cubic B-spline path smoother.

Fits a natural / clamped cubic spline through a sequence of joint-space
waypoints and resamples at uniform arc-length intervals, producing a
trajectory with :math:`C^2` continuity (smooth position, velocity, and
acceleration).  C^2 means the path and its first two derivatives are
continuous -- no sudden jumps in acceleration.

The natural cubic spline is the minimum-energy interpolant:

.. math::
    \min \int_0^1 \|\ddot{q}(t)\|^2 \, dt

In plain terms: among all smooth curves passing through the waypoints,
the cubic spline minimises the total squared acceleration (the
smoothest possible interpolation).

This module is pure maths -- **no physics-engine calls** -- so it runs in
sub-millisecond time regardless of obstacle count.

Usage::

    from panda_rrt.computations.spline_smoother import SplineSmoother
    smooth = SplineSmoother.smooth(waypoints, n_points=60)
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
    """Stateless cubic-spline smoother (all methods are ``@staticmethod``)."""

    @staticmethod
    def smooth(
        waypoints: List[np.ndarray],
        n_points: int = 60,
        bc_type: str = "clamped",
    ) -> List[np.ndarray]:
        """Fit a cubic spline and resample to *n_points* configurations.

        Parameters
        ----------
        waypoints : list[ndarray]
            Ordered joint-space waypoints (at least 2).
        n_points : int
            Number of output samples.
        bc_type : str
            ``"clamped"`` pins zero velocity at boundaries;
            ``"natural"`` sets zero second derivative.

        Returns
        -------
        list[ndarray]
            Evenly-spaced configurations along the spline, clamped to
            joint limits.
        """
        if len(waypoints) <= 1:
            return list(waypoints)
        if len(waypoints) == 2:
            return SplineSmoother._lerp(waypoints[0], waypoints[1], n_points)

        Q = np.array(waypoints)                          # (N, 7)
        dists = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        t = np.concatenate(([0.0], np.cumsum(dists)))
        t /= t[-1]                                       # normalise to [0, 1]

        cs = CubicSpline(t, Q, bc_type=bc_type, axis=0)
        t_sample = np.linspace(0.0, 1.0, n_points)
        Q_smooth = np.clip(cs(t_sample), PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)

        return [Q_smooth[i] for i in range(n_points)]

    @staticmethod
    def integrated_acceleration_cost(
        waypoints: List[np.ndarray],
        n_eval: int = 200,
        bc_type: str = "clamped",
    ) -> float:
        r"""Compute :math:`J = \int_0^1 \|\ddot{q}(t)\|^2\,dt`.

        In plain terms: integrate the squared acceleration along the
        spline from start to end.  Lower values mean a smoother
        trajectory.

        Useful for comparing trajectory smoothness: lower is better.

        Parameters
        ----------
        waypoints : list[ndarray]
        n_eval : int
            Quadrature samples.

        Returns
        -------
        float
        """
        if len(waypoints) < 3:
            return 0.0

        Q = np.array(waypoints)
        dists = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        t = np.concatenate(([0.0], np.cumsum(dists)))
        t /= t[-1]

        cs = CubicSpline(t, Q, bc_type=bc_type, axis=0)
        t_eval = np.linspace(0.0, 1.0, n_eval)
        qddot = cs(t_eval, 2)
        integrand = np.sum(qddot ** 2, axis=1)
        dt = 1.0 / (n_eval - 1)
        return float(np.trapz(integrand, dx=dt))

    # ── Helpers ─────────────────────────────────────

    @staticmethod
    def _lerp(
        q0: np.ndarray, q1: np.ndarray, n: int,
    ) -> List[np.ndarray]:
        """Linear interpolation between two configurations."""
        return [
            np.clip(q0 + t * (q1 - q0), PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)
            for t in np.linspace(0.0, 1.0, n)
        ]
