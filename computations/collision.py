"""Collision checking via PyBullet closest-point queries.

Provides configuration-space validity checks (joint limits + obstacle
clearance) and edge interpolation used by every planner variant.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from panda_rrt.common_utils.robot_constants import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PANDA_NUM_JOINTS,
)
from panda_rrt.config import get as cfg
from panda_rrt.computations.jit_kernels import in_limits


COLLISION_LINKS: List[int] = cfg("collision", "links")
SAFETY_MARGIN: float = cfg("collision", "safety_margin")


class CollisionChecker:
    """Thin wrapper around ``pybullet.getClosestPoints`` for link--obstacle
    distance queries and configuration-space validity checks.

    Parameters
    ----------
    physics_client : BulletClient
        Active PyBullet client.
    robot_id : int
        Robot body index.
    obstacle_ids : list[int]
        Body indices of spawned obstacles.
    """

    def __init__(
        self, physics_client, robot_id: int, obstacle_ids: List[int],
    ) -> None:
        self._p = physics_client
        self._robot_id = robot_id
        self._obstacle_ids = obstacle_ids

    # ── Joint-state helpers ─────────────────────────

    def set_config(self, q: np.ndarray) -> None:
        """Apply joint angles to the robot without running dynamics."""
        for i in range(PANDA_NUM_JOINTS):
            self._p.resetJointState(self._robot_id, i, q[i])

    # ── Distance queries ────────────────────────────

    def min_link_obstacle_distance(
        self, q: np.ndarray,
    ) -> Tuple[float, int, int]:
        """Minimum signed distance between collision links and obstacles.

        Parameters
        ----------
        q : ndarray, shape (7,)

        Returns
        -------
        (distance, link_index, obstacle_id)
            ``distance`` is negative when penetrating.
        """
        self.set_config(q)
        min_d = float("inf")
        min_link = -1
        min_obs = -1

        for obs_id in self._obstacle_ids:
            contacts = self._p.getClosestPoints(
                bodyA=self._robot_id, bodyB=obs_id, distance=1.0,
            )
            for cp in contacts:
                link_idx = cp[3]
                if link_idx not in COLLISION_LINKS:
                    continue
                d = cp[8]
                if d < min_d:
                    min_d = d
                    min_link = link_idx
                    min_obs = obs_id
        return min_d, min_link, min_obs

    def per_link_distances(self, q: np.ndarray) -> Dict[int, float]:
        """Return ``{link: min_distance}`` for every collision link.

        Parameters
        ----------
        q : ndarray, shape (7,)

        Returns
        -------
        dict[int, float]
        """
        self.set_config(q)
        dists: Dict[int, float] = {lk: float("inf") for lk in COLLISION_LINKS}

        for obs_id in self._obstacle_ids:
            contacts = self._p.getClosestPoints(
                bodyA=self._robot_id, bodyB=obs_id, distance=1.0,
            )
            for cp in contacts:
                link_idx = cp[3]
                if link_idx in dists:
                    d = cp[8]
                    if d < dists[link_idx]:
                        dists[link_idx] = d
        return dists

    # ── Validity checks ─────────────────────────────

    def is_config_valid(self, q: np.ndarray) -> bool:
        """Check joint limits and obstacle clearance at *q*.

        Returns ``True`` when *q* lies within joint limits **and** every
        collision link is at least ``SAFETY_MARGIN`` from all obstacles.
        """
        if not in_limits(q, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS):
            return False
        d_min, _, _ = self.min_link_obstacle_distance(q)
        return d_min > SAFETY_MARGIN

    def is_edge_valid(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        n_checks: int = None,
    ) -> bool:
        """Check collision along a linearly-interpolated edge.

        Parameters
        ----------
        q_from, q_to : ndarray, shape (7,)
            Start and end configurations.
        n_checks : int or None
            Number of interpolation samples.  Defaults to
            ``collision.edge_checks`` from config.
        """
        if n_checks is None:
            n_checks = cfg("collision", "edge_checks")
        for i in range(1, n_checks + 1):
            t = i / n_checks
            q_interp = q_from + t * (q_to - q_from)
            if not self.is_config_valid(q_interp):
                return False
        return True
