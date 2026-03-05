r"""Configuration-space Artificial Potential Field (C-space APF).

Computes attractive and repulsive forces in the robot's joint space
to bias RRT sampling toward the goal while avoiding obstacles.

Key design choices
------------------
1. **Conic / quadratic attractive gradient** -- prevents the goal from
   overpowering repulsive fields at large distances.  Within
   :math:`d_{\text{thresh}}` (the crossover distance) the gradient is
   quadratic; beyond it the magnitude is clamped (conic regime).
2. **Jacobian-transpose repulsive gradient** -- maps 3-D contact normals
   :math:`\hat{n}` (unit vector from obstacle to link) to 7-D joint
   torques via :math:`J^T` (the transpose of the Jacobian matrix),
   replacing the old 14-call finite-difference sweep with one
   ``calculateJacobian`` per active link.
3. **Per-component force capping** -- individual attractive/repulsive
   forces are capped at ``f_max`` before summation; the resultant is
   normalised so ``step_size`` alone governs displacement magnitude.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from panda_rrt.common_utils.robot_constants import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PANDA_NUM_JOINTS,
)
from panda_rrt.computations.collision import CollisionChecker, COLLISION_LINKS
from panda_rrt.computations.jit_kernels import attractive_gradient_jit
from panda_rrt.config import get as cfg


class CSpaceAPF:
    r"""APF force provider in 7-DOF configuration space.

    Parameters
    ----------
    collision_checker : CollisionChecker
    k_att : float
        Attractive gain.
    k_rep : float
        Repulsive gain.
    rho_0 : float
        Repulsion influence radius (metres).
    delta_q : float
        Finite-difference step (unused after Jacobian upgrade, kept for API).
    f_max : float
        Per-component force cap.
    d_thresh : float
        Conic/quadratic crossover distance (radians in C-space).
    """

    def __init__(
        self,
        collision_checker: CollisionChecker,
        k_att: float = None,
        k_rep: float = None,
        rho_0: float = None,
        delta_q: float = None,
        f_max: float = None,
        d_thresh: float = None,
    ) -> None:
        self.cc = collision_checker
        self.k_att = k_att if k_att is not None else cfg("apf", "k_att")
        self.k_rep = k_rep if k_rep is not None else cfg("apf", "k_rep")
        self.rho_0 = rho_0 if rho_0 is not None else cfg("apf", "rho_0")
        self.delta_q = delta_q if delta_q is not None else cfg("apf", "delta_q")
        self.f_max = f_max if f_max is not None else cfg("apf", "f_max")
        self.d_thresh = d_thresh if d_thresh is not None else cfg("apf", "d_thresh")

    # ── Attractive gradient ─────────────────────────

    def attractive_gradient(
        self, q: np.ndarray, q_goal: np.ndarray,
    ) -> np.ndarray:
        r"""Piecewise attractive potential gradient.

        Delegates to a JIT-compiled kernel for the inner arithmetic.

        .. math::
            \nabla U_{\text{att}} =
            \begin{cases}
            k_{\text{att}} \, (q - q_g) & d \le d_{\text{thresh}} \\[4pt]
            k_{\text{att}} \, d_{\text{thresh}} \, \dfrac{q - q_g}{d}
              & d > d_{\text{thresh}}
            \end{cases}

        In plain terms:

        * When close to the goal (distance <= threshold):
          gradient = k_att * (current - goal)  (quadratic pull).
        * When far from the goal (distance > threshold):
          gradient = k_att * threshold * (current - goal) / distance
          (constant-magnitude pull, so it doesn't overpower repulsion).

        The conic branch caps the attractive pull so it cannot overpower
        the repulsive field at large goal distances.
        """
        return attractive_gradient_jit(
            np.ascontiguousarray(q, dtype=np.float64),
            np.ascontiguousarray(q_goal, dtype=np.float64),
            self.k_att,
            self.d_thresh,
        )

    # ── Repulsive gradient (Jacobian-transpose) ─────

    def _per_link_contact_info(
        self, q: np.ndarray,
    ) -> Dict[int, Tuple[float, np.ndarray, np.ndarray]]:
        """Closest obstacle per collision link.

        Returns ``{link: (distance, contact_normal_3d, pos_on_robot_3d)}``.
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
                    info[link_idx] = (d, np.array(cp[7]), np.array(cp[5]))
        return info

    def _world_to_link_local(
        self, link_idx: int, world_pos: np.ndarray,
    ) -> List[float]:
        """Convert a world-frame point to the link's local URDF frame."""
        state = self.cc._p.getLinkState(self.cc._robot_id, link_idx)
        inv_pos, inv_orn = self.cc._p.invertTransform(state[4], state[5])
        local, _ = self.cc._p.multiplyTransforms(
            inv_pos, inv_orn, world_pos.tolist(), [0, 0, 0, 1],
        )
        return list(local)

    def _link_jacobian(
        self, q: np.ndarray, link_idx: int, local_pos: List[float],
    ) -> np.ndarray:
        r"""3x7 linear Jacobian at *link_idx* / *local_pos*."""
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
        return np.array(lin_jac)[:, :PANDA_NUM_JOINTS]

    def repulsive_gradient(self, q: np.ndarray) -> np.ndarray:
        r"""Repulsive gradient via the Jacobian transpose.

        For each link within :math:`\rho_0` (the repulsion influence
        radius):

        .. math::
            \nabla_q U_{\text{rep}}
            = -k_{\text{rep}} \left(\frac{1}{\rho} - \frac{1}{\rho_0}\right)
              \frac{1}{\rho^2} \, J^T \hat{n}

        In plain terms: for each robot link closer than rho_0 to an
        obstacle, the repulsive gradient is:
        -k_rep * (1/distance - 1/rho_0) / distance^2 * J^T * normal,
        where J^T maps the 3-D push direction into joint-space torques.
        """
        contact_info = self._per_link_contact_info(q)
        grad = np.zeros(PANDA_NUM_JOINTS)

        for link_idx, (rho_i, normal, pos_on_robot) in contact_info.items():
            if rho_i >= self.rho_0 or rho_i < 1e-6:
                continue
            scalar = -self.k_rep * (1.0 / rho_i - 1.0 / self.rho_0) / (rho_i ** 2)
            local_pos = self._world_to_link_local(link_idx, pos_on_robot)
            J = self._link_jacobian(q, link_idx, local_pos)
            grad += scalar * (J.T @ normal)

        return grad

    # ── Force aggregation ───────────────────────────

    @staticmethod
    def _cap(f: np.ndarray, f_max: float) -> np.ndarray:
        """Cap vector magnitude at *f_max*, preserving direction."""
        n = np.linalg.norm(f)
        if n > f_max and n > 1e-12:
            return f * (f_max / n)
        return f

    def total_force(
        self, q: np.ndarray, q_goal: np.ndarray,
    ) -> np.ndarray:
        r"""Normalised resultant of attractive + repulsive forces.

        .. math::
            F = \frac{\text{cap}(-\nabla U_{\text{att}})
                      + \text{cap}(-\nabla U_{\text{rep}})}
                     {\|\cdot\|}

        In plain terms: negate both gradients (so they become forces),
        cap each at f_max to prevent either from dominating, add them
        together, then normalise to a unit vector.
        """
        f_att = self._cap(-self.attractive_gradient(q, q_goal), self.f_max)
        f_rep = self._cap(-self.repulsive_gradient(q), self.f_max)

        f_total = f_att + f_rep
        norm = np.linalg.norm(f_total)
        if norm < 1e-12:
            return f_total
        return f_total / norm
