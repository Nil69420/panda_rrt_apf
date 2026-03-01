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

    Three upgrades over the original implementation:

    1. **Conic / Quadratic attractive gradient** — prevents the goal from
       overpowering repulsive fields at large distances.
    2. **Jacobian-transpose repulsive gradient** — maps 3-D contact normals
       to 7-D joint torques via ``J^T``, replacing the 14-call
       finite-difference sweep with one ``calculateJacobian`` per active
       link (~92 % fewer physics calls).
    3. **Force capping & normalization** — individual forces are capped at
       ``f_max`` before summation, and the resultant is unit-normalized so
       ``step_size`` alone governs displacement.
    """

    def __init__(
        self,
        collision_checker: CollisionChecker,
        k_att: float | None = None,
        k_rep: float | None = None,
        rho_0: float | None = None,
        delta_q: float | None = None,
        f_max: float | None = None,
        d_thresh: float | None = None,
    ) -> None:
        self.cc = collision_checker
        self.k_att = k_att if k_att is not None else cfg("apf", "k_att")
        self.k_rep = k_rep if k_rep is not None else cfg("apf", "k_rep")
        self.rho_0 = rho_0 if rho_0 is not None else cfg("apf", "rho_0")
        self.delta_q = delta_q if delta_q is not None else cfg("apf", "delta_q")
        self.f_max = f_max if f_max is not None else cfg("apf", "f_max")
        self.d_thresh = d_thresh if d_thresh is not None else cfg("apf", "d_thresh")

    # ── Conic / Quadratic attractive gradient ───────

    def attractive_gradient(self, q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        """Piecewise attractive potential gradient.

        * ``d ≤ d_thresh`` → quadratic:  ``∇U = k_att · v``
        * ``d > d_thresh`` → conic:      ``∇U = k_att · d_thresh · v / d``

        This caps the attractive pull at large distances so the goal
        cannot overpower the repulsive field.
        """
        v = q - q_goal
        d = np.linalg.norm(v)
        if d < 1e-12:
            return np.zeros(PANDA_NUM_JOINTS)
        if d <= self.d_thresh:
            return self.k_att * v
        # Conic: constant-magnitude pull = k_att * d_thresh
        inv_d = 1.0 / d          # single divide, broadcast multiply
        return (self.k_att * self.d_thresh * inv_d) * v

    # ── Jacobian-transpose repulsive gradient ───────

    def _per_link_contact_info(
        self, q: np.ndarray,
    ) -> Dict[int, Tuple[float, np.ndarray, np.ndarray]]:
        """Sweep ``getClosestPoints`` once per obstacle.

        Returns ``{link: (distance, contact_normal_3d, pos_on_robot_3d)}``
        keeping only the closest obstacle per link.
        """
        self.cc.set_config(q)
        info: Dict[int, Tuple[float, np.ndarray, np.ndarray]] = {}

        for obs_id in self.cc._obstacle_ids:
            contacts = self.cc._p.getClosestPoints(
                bodyA=self.cc._robot_id, bodyB=obs_id, distance=1.0,
            )
            for cp in contacts:
                link_idx = cp[3]
                if link_idx not in COLLISION_LINKS:
                    continue
                d = cp[8]
                if link_idx not in info or d < info[link_idx][0]:
                    # cp[7] = contactNormalOnB  → obstacle → robot
                    # cp[5] = positionOnA       → world-frame on robot
                    info[link_idx] = (d, np.array(cp[7]), np.array(cp[5]))
        return info

    def _world_to_link_local(
        self, link_idx: int, world_pos: np.ndarray,
    ) -> List[float]:
        """Convert a world-frame point to the link's local URDF frame."""
        state = self.cc._p.getLinkState(self.cc._robot_id, link_idx)
        inv_pos, inv_orn = self.cc._p.invertTransform(state[4], state[5])
        local, _ = self.cc._p.multiplyTransforms(
            inv_pos, inv_orn,
            world_pos.tolist(), [0, 0, 0, 1],
        )
        return list(local)

    def _link_jacobian(
        self, q: np.ndarray, link_idx: int, local_pos: List[float],
    ) -> np.ndarray:
        """Compute the 3×7 linear Jacobian at *link_idx* / *local_pos*."""
        q_list = q.tolist()
        zero_vec = [0.0] * PANDA_NUM_JOINTS
        lin_jac, _ = self.cc._p.calculateJacobian(
            bodyUniqueId=self.cc._robot_id,
            linkIndex=link_idx,
            localPosition=local_pos,
            objPositions=q_list + [0.0, 0.0],
            objVelocities=zero_vec + [0.0, 0.0],
            objAccelerations=zero_vec + [0.0, 0.0],
        )
        return np.array(lin_jac)[:, :PANDA_NUM_JOINTS]      # 3×7

    def repulsive_gradient(self, q: np.ndarray) -> np.ndarray:
        """Repulsive potential gradient via the Jacobian transpose.

        For each link within ``rho_0``:

        1. ``getClosestPoints`` sweep → contact normal n̂ + distance ρ.
        2. ``calculateJacobian`` at the contact point → J (3×7).
        3. ``∇_q U_rep = −k_rep (1/ρ − 1/ρ₀) / ρ²  ·  Jᵀ n̂``

        Physics calls: ``N_obs + N_active`` vs. ``N_obs × (1 + 14)`` before.
        """
        contact_info = self._per_link_contact_info(q)
        grad = np.zeros(PANDA_NUM_JOINTS)

        for link_idx, (rho_i, normal, pos_on_robot) in contact_info.items():
            if rho_i >= self.rho_0 or rho_i < 1e-6:
                continue
            # ∇_q U_rep = −k_rep (1/ρ − 1/ρ₀) / ρ² · Jᵀ · n̂
            scalar = -self.k_rep * (1.0 / rho_i - 1.0 / self.rho_0) / (rho_i ** 2)
            local_pos = self._world_to_link_local(link_idx, pos_on_robot)
            J = self._link_jacobian(q, link_idx, local_pos)
            grad += scalar * (J.T @ normal)

        return grad

    # ── Force capping & normalization ───────────────

    @staticmethod
    def _cap(f: np.ndarray, f_max: float) -> np.ndarray:
        """Cap vector magnitude at *f_max*, preserving direction."""
        n = np.linalg.norm(f)
        if n > f_max and n > 1e-12:
            return f * (f_max / n)
        return f

    def total_force(self, q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        """Normalized, capped resultant of attractive + repulsive forces.

        1. Compute raw forces ``F = −∇U``.
        2. Cap each at ``f_max`` so neither dominates.
        3. Normalize the sum so ``step_size`` governs displacement.
        """
        f_att = self._cap(-self.attractive_gradient(q, q_goal), self.f_max)
        f_rep = self._cap(-self.repulsive_gradient(q), self.f_max)

        f_total = f_att + f_rep
        norm = np.linalg.norm(f_total)
        if norm < 1e-12:
            return f_total
        return f_total / norm
