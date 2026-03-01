from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from panda_rrt.common_utils.robot_constants import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PANDA_NUM_JOINTS,
)
from panda_rrt.collision import CollisionChecker, COLLISION_LINKS
from panda_rrt.config import get as cfg


class CSpaceAPF:
    """Configuration-space Artificial Potential Field.

    Computes attractive + repulsive gradients using finite-difference
    approximation of link–obstacle distances from :class:`CollisionChecker`.
    """

    def __init__(
        self,
        collision_checker: CollisionChecker,
        k_att: float | None = None,
        k_rep: float | None = None,
        rho_0: float | None = None,
        delta_q: float | None = None,
    ) -> None:
        self.cc = collision_checker
        self.k_att = k_att if k_att is not None else cfg("apf", "k_att")
        self.k_rep = k_rep if k_rep is not None else cfg("apf", "k_rep")
        self.rho_0 = rho_0 if rho_0 is not None else cfg("apf", "rho_0")
        self.delta_q = delta_q if delta_q is not None else cfg("apf", "delta_q")

    # ── gradients ───────────────────────────────────

    def attractive_gradient(self, q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        return self.k_att * (q - q_goal)

    def _numerical_distance_gradient(
        self, q: np.ndarray,
    ) -> Tuple[Dict[int, float], Dict[int, np.ndarray]]:
        dists_center = self.cc.per_link_distances(q)

        # Early exit: skip expensive finite-difference gradient when no link
        # is within the repulsive influence range.
        active_links = [lk for lk in COLLISION_LINKS if dists_center[lk] < self.rho_0]
        if not active_links:
            return dists_center, {lk: np.zeros(PANDA_NUM_JOINTS) for lk in COLLISION_LINKS}

        grad_rho: Dict[int, np.ndarray] = {
            lk: np.zeros(PANDA_NUM_JOINTS) for lk in COLLISION_LINKS
        }
        for j in range(PANDA_NUM_JOINTS):
            q_plus = q.copy()
            q_plus[j] += self.delta_q
            q_plus = np.clip(q_plus, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)
            d_plus = self.cc.per_link_distances(q_plus)

            q_minus = q.copy()
            q_minus[j] -= self.delta_q
            q_minus = np.clip(q_minus, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)
            d_minus = self.cc.per_link_distances(q_minus)

            actual_delta = q_plus[j] - q_minus[j]
            if abs(actual_delta) > 1e-10:
                for link in active_links:
                    grad_rho[link][j] = (d_plus[link] - d_minus[link]) / actual_delta

        return dists_center, grad_rho

    def repulsive_gradient(self, q: np.ndarray) -> np.ndarray:
        dists, grad_rho = self._numerical_distance_gradient(q)
        grad = np.zeros(PANDA_NUM_JOINTS)

        for link in COLLISION_LINKS:
            rho_i = dists[link]
            if rho_i >= self.rho_0 or rho_i < 1e-6:
                continue
            coeff = self.k_rep * (1.0 / self.rho_0 - 1.0 / rho_i) / (rho_i * rho_i)
            grad += coeff * grad_rho[link]

        return grad

    def total_force(self, q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        return -self.attractive_gradient(q, q_goal) - self.repulsive_gradient(q)
