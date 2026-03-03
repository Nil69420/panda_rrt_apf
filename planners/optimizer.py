"""Gradient-descent path optimiser for RRT planners.

Minimises a composite objective:

.. math::

    J(P) = J_{\\text{smooth}}(P) + \\lambda \\, J_{\\text{obs}}(P)

where:

.. math::

    J_{\\text{smooth}} = \\sum_{i=1}^{n-2} \\|q_{i+1} - 2q_i + q_{i-1}\\|^2

.. math::

    J_{\\text{obs}} = \\sum_i \\max\\!\\left(0,\\;
        \\frac{1}{d_{\\min}(q_i)} - \\frac{1}{d_{\\text{safe}}}\\right)^{\\!2}

Start and goal waypoints are pinned; only interior waypoints move.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from panda_rrt.common_utils.robot_constants import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PANDA_NUM_JOINTS,
)
from panda_rrt.computations.collision import CollisionChecker, COLLISION_LINKS
from panda_rrt.computations.jit_kernels import (
    smoothness_cost as _jit_smoothness_cost,
    smooth_gradient_all as _jit_smooth_gradient_all,
)
from panda_rrt.config import get as cfg


class PathOptimizer:
    """Gradient-descent optimiser that smooths an RRT path in C-space.

    Parameters
    ----------
    collision_checker : CollisionChecker
    lr : float
        Learning rate :math:`\\eta`.
    lam : float
        Obstacle-cost weight :math:`\\lambda`.
    n_iters : int
        Maximum gradient-descent iterations.
    delta_q : float
        Finite-difference step (radians) -- unused with analytic gradients
        but retained for backward compatibility.
    min_clearance : float
        Reject waypoint updates that violate this clearance (metres).
    danger_threshold : float
        Hinge cutoff :math:`d_{\\text{safe}}` (metres). Waypoints further
        than this from any obstacle contribute zero obstacle cost.
    """

    # Max C-space movement per waypoint per iteration (prevents teleporting)
    _MAX_STEP = 0.05

    def __init__(
        self,
        collision_checker: CollisionChecker,
        lr: float = None,
        lam: float = None,
        n_iters: int = None,
        delta_q: float = None,
        min_clearance: float = None,
        danger_threshold: float = None,
    ) -> None:
        _c = cfg("optimizer")
        self.cc = collision_checker
        self.lr = lr if lr is not None else _c["lr"]
        self.lam = lam if lam is not None else _c["lambda"]
        self.n_iters = n_iters if n_iters is not None else _c["n_iters"]
        self.delta_q = delta_q if delta_q is not None else _c["delta_q"]
        self.min_clearance = min_clearance if min_clearance is not None else _c["min_clearance"]
        self.danger_threshold = (
            danger_threshold if danger_threshold is not None else _c["danger_threshold"]
        )
        self._inv_d_safe = 1.0 / self.danger_threshold

    # ── cost functions ──────────────────────────────

    @staticmethod
    def smoothness_cost(path: List[np.ndarray]) -> float:
        """Squared-acceleration smoothness cost.

        Delegates to a JIT-compiled kernel for the inner loop.

        .. math::

            J_{\\text{smooth}} = \\sum_{i=1}^{n-2}
                \\|q_{i+1} - 2q_i + q_{i-1}\\|^2
        """
        if len(path) < 3:
            return 0.0
        return float(_jit_smoothness_cost(np.array(path)))

    def obstacle_cost(self, path: List[np.ndarray]) -> float:
        """Hinge-loss obstacle proximity cost.

        Only penalises waypoints closer than ``danger_threshold``.
        """
        total = 0.0
        for i in range(1, len(path) - 1):
            d_min, _, _ = self.cc.min_link_obstacle_distance(path[i])
            d_min = max(d_min, 1e-6)
            if d_min < self.danger_threshold:
                h = 1.0 / d_min - self._inv_d_safe
                total += h * h
        return total

    def total_cost(self, path: List[np.ndarray]) -> float:
        """Combined smoothness + weighted obstacle cost."""
        return self.smoothness_cost(path) + self.lam * self.obstacle_cost(path)

    # ── gradient computation ────────────────────────

    @staticmethod
    def _smooth_gradient(path: List[np.ndarray], i: int) -> np.ndarray:
        """Analytic gradient of :math:`J_{\\text{smooth}}` w.r.t. waypoint *i*.

        Waypoint *q_i* appears in three acceleration terms
        (:math:`a_{i-1}, a_i, a_{i+1}`), giving:

        .. math::

            \\frac{\\partial J}{\\partial q_i} =
                2 a_{i-1} - 4 a_i + 2 a_{i+1}

        Boundary terms are omitted when :math:`i-2` or :math:`i+2` are
        outside the path.
        """
        n = len(path)
        grad = np.zeros(PANDA_NUM_JOINTS)
        if i >= 2:
            grad += 2.0 * (path[i] - 2.0 * path[i - 1] + path[i - 2])
        grad -= 4.0 * (path[i + 1] - 2.0 * path[i] + path[i - 1])
        if i <= n - 3:
            grad += 2.0 * (path[i + 2] - 2.0 * path[i + 1] + path[i])
        return grad

    def _obstacle_gradient(self, q: np.ndarray) -> np.ndarray:
        """Gradient of hinge-loss obstacle cost via Jacobian transpose.

        .. math::

            \\frac{\\partial h}{\\partial q} =
                2\\left(\\frac{1}{d} - \\frac{1}{d_{\\text{safe}}}\\right)
                \\cdot \\left(-\\frac{1}{d^2}\\right)
                \\cdot J^T \\hat{n}

        Uses ``getClosestPoints`` once to find contact normals, then
        ``calculateJacobian`` per active link.
        """
        contact_info = self._per_link_contact_info(q)
        d_min = min(
            (info[0] for info in contact_info.values()),
            default=float("inf"),
        )
        d_min = max(d_min, 1e-6)
        if d_min >= self.danger_threshold:
            return np.zeros(PANDA_NUM_JOINTS)

        inv_d = 1.0 / d_min
        prefactor = 2.0 * (inv_d - self._inv_d_safe) * (-inv_d * inv_d)

        grad = np.zeros(PANDA_NUM_JOINTS)
        for link_idx, (rho_i, normal, pos_on_robot) in contact_info.items():
            if rho_i >= self.danger_threshold or rho_i < 1e-6:
                continue
            local_pos = self._world_to_link_local(link_idx, pos_on_robot)
            J = self._link_jacobian(q, link_idx, local_pos)
            grad += J.T @ normal
        return prefactor * grad

    def _compute_gradients(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Compute combined gradients for all interior waypoints.

        The smoothness gradient is computed in a single parallel JIT
        call via :func:`smooth_gradient_all`; only the obstacle
        gradient (which requires PyBullet queries) runs in Python.

        Returns
        -------
        list of ndarray
            One gradient vector per interior waypoint (len = n - 2).
        """
        path_arr = np.array(path)
        smooth_grads = _jit_smooth_gradient_all(path_arr)
        grads = []
        for idx, i in enumerate(range(1, len(path) - 1)):
            g_obs = self._obstacle_gradient(path[i])
            grads.append(smooth_grads[idx] + self.lam * g_obs)
        return grads

    # ── gradient application ────────────────────────

    def _apply_updates(
        self, path: List[np.ndarray], grads: List[np.ndarray],
    ) -> bool:
        """Apply step-clipped gradient updates to interior waypoints.

        Parameters
        ----------
        path : list of ndarray
            Modified **in place**.
        grads : list of ndarray
            Gradient per interior waypoint.

        Returns
        -------
        bool
            ``True`` if at least one waypoint moved.
        """
        any_moved = False
        for idx, i in enumerate(range(1, len(path) - 1)):
            raw_step = -self.lr * grads[idx]
            step_norm = np.linalg.norm(raw_step)
            if step_norm > self._MAX_STEP:
                raw_step *= self._MAX_STEP / step_norm
            q_new = np.clip(
                path[i] + raw_step, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS,
            )
            d_min, _, _ = self.cc.min_link_obstacle_distance(q_new)
            if d_min > self.min_clearance:
                if step_norm > 1e-8:
                    any_moved = True
                path[i] = q_new
        return any_moved

    # ── convergence check ───────────────────────────

    @staticmethod
    def _check_convergence(prev_cost: float, cur_cost: float, tol: float = 1e-4) -> bool:
        """Return ``True`` when relative cost change falls below *tol*."""
        return abs(prev_cost - cur_cost) / (abs(prev_cost) + 1e-12) < tol

    # ── Jacobian-transpose helpers ──────────────────

    def _per_link_contact_info(self, q: np.ndarray):
        """Closest-point data for every collision link.

        Returns
        -------
        dict
            ``{link_idx: (distance, normal_3d, position_on_robot_3d)}``.
        """
        self.cc.set_config(q)
        info = {}
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

    def _world_to_link_local(self, link_idx: int, world_pos: np.ndarray):
        """Convert a world-frame point to the link's local URDF frame."""
        state = self.cc._p.getLinkState(self.cc._robot_id, link_idx)
        inv_pos, inv_orn = self.cc._p.invertTransform(state[4], state[5])
        local, _ = self.cc._p.multiplyTransforms(
            inv_pos, inv_orn, world_pos.tolist(), [0, 0, 0, 1],
        )
        return list(local)

    def _link_jacobian(self, q: np.ndarray, link_idx: int, local_pos):
        """3x7 linear Jacobian at the given link / local position."""
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

    # ── main optimiser loop ─────────────────────────

    def optimize(
        self, path: List[np.ndarray], verbose: bool = False,
    ) -> List[np.ndarray]:
        """Run gradient-descent smoothing on a collision-free path.

        The loop is decomposed into :meth:`_compute_gradients`,
        :meth:`_apply_updates`, and :meth:`_check_convergence`, each of
        which can be overridden independently.

        Parameters
        ----------
        path : list of ndarray
            Waypoints (start and goal are pinned).
        verbose : bool
            Print cost every 10 iterations and timing breakdown.

        Returns
        -------
        list of ndarray
            Optimised path.
        """
        if len(path) <= 2:
            return list(path)

        if verbose:
            print(f"  Optimizer: {len(path) - 2} interior waypoints, "
                  f"lr={self.lr}, lambda={self.lam}, max_iters={self.n_iters}")

        t_start = time.perf_counter()
        opt = [q.copy() for q in path]
        prev_cost = self.total_cost(opt)
        t_grad = t_coll = 0.0
        final_iter = 0

        for k in range(self.n_iters):
            final_iter = k

            if verbose and k % 10 == 0:
                j_s = self.smoothness_cost(opt)
                j_o = self.obstacle_cost(opt)
                print(f"    iter {k:3d}  J={prev_cost:.4f}  "
                      f"(smooth={j_s:.4f}, obs={j_o:.4f})")

            # 1. Compute gradients for all interior waypoints
            t0 = time.perf_counter()
            grads = self._compute_gradients(opt)
            t_grad += time.perf_counter() - t0

            # 2. Apply step-clipped updates
            t0 = time.perf_counter()
            any_moved = self._apply_updates(opt, grads)
            t_coll += time.perf_counter() - t0

            if not any_moved:
                if verbose:
                    print(f"    converged at iter {k} (no movement)")
                break

            # 3. Check cost convergence
            cur_cost = self.total_cost(opt)
            if self._check_convergence(prev_cost, cur_cost):
                if verbose:
                    rel = abs(prev_cost - cur_cost) / (abs(prev_cost) + 1e-12)
                    print(f"    converged at iter {k} "
                          f"(rel cost change {rel:.2e})")
                break
            prev_cost = cur_cost

        t_total = time.perf_counter() - t_start
        if verbose:
            print(f"    final J={self.total_cost(opt):.4f}  "
                  f"after {final_iter + 1} iters")
            print(f"    Timing: total={t_total:.3f}s  grads={t_grad:.3f}s  "
                  f"collision={t_coll:.3f}s")
            in_len = self.path_length(path)
            out_len = self.path_length(opt)
            arrow = "\u2193" if out_len < in_len else "\u2191"
            print(f"    Path length: {in_len:.3f} -> {out_len:.3f} rad "
                  f"({arrow} {abs(out_len - in_len):.3f})")

        # Final validation -- fall back to original on invalid edge
        for i in range(len(opt) - 1):
            if not self.cc.is_edge_valid(opt[i], opt[i + 1]):
                if verbose:
                    print("    WARNING: optimised path has invalid edge, "
                          "returning original")
                return list(path)

        return opt

    # ── utilities ───────────────────────────────────

    @staticmethod
    def path_length(path: List[np.ndarray]) -> float:
        """Cumulative Euclidean arc-length of *path*."""
        return sum(
            np.linalg.norm(path[i + 1] - path[i])
            for i in range(len(path) - 1)
        )

    @staticmethod
    def resample_path(
        path: List[np.ndarray], n_waypoints: int = 20,
    ) -> List[np.ndarray]:
        """Resample *path* to *n_waypoints* uniformly-spaced waypoints.

        Ensures the optimizer has enough interior points to work with
        even when shortcut smoothing has collapsed the path to very few
        waypoints.
        """
        if len(path) <= 2 or n_waypoints <= 2:
            return list(path)

        cum = [0.0]
        for i in range(1, len(path)):
            cum.append(cum[-1] + np.linalg.norm(path[i] - path[i - 1]))
        total_len = cum[-1]
        if total_len < 1e-12:
            return list(path)

        resampled: List[np.ndarray] = [path[0].copy()]
        seg = 0
        for k in range(1, n_waypoints - 1):
            target = total_len * k / (n_waypoints - 1)
            while seg < len(cum) - 2 and cum[seg + 1] < target:
                seg += 1
            seg_len = cum[seg + 1] - cum[seg]
            t = (target - cum[seg]) / seg_len if seg_len > 1e-12 else 0.0
            resampled.append(path[seg] + t * (path[seg + 1] - path[seg]))
        resampled.append(path[-1].copy())
        return resampled
