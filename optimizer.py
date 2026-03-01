"""Optimization-based path smoothing for RRT planners.

Minimises  J(P) = J_smooth(P) + lambda * J_obs(P)

where
    J_smooth = sum_i ||q_{i+1} - 2*q_i + q_{i-1}||^2   (acceleration penalty)
    J_obs    = hinge-loss obstacle proximity
             = sum_i max(0, 1/d_min(q_i) - 1/d_safe)^2

             Only penalises waypoints closer than ``danger_threshold`` to
             an obstacle.  Waypoints that are safely far away contribute
             **zero** obstacle cost, preventing the path from ballooning.

Start and goal waypoints are pinned; only interior waypoints are updated.
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
from panda_rrt.collision import CollisionChecker, COLLISION_LINKS
from panda_rrt.config import get as cfg


class PathOptimizer:
    """Gradient-descent optimiser that smooths an RRT path in C-space."""

    def __init__(
        self,
        collision_checker: CollisionChecker,
        lr: float | None = None,
        lam: float | None = None,
        n_iters: int | None = None,
        delta_q: float | None = None,
        min_clearance: float | None = None,
        danger_threshold: float | None = None,
    ) -> None:
        _c = cfg("optimizer")
        self.cc = collision_checker
        self.lr = lr if lr is not None else _c["lr"]
        self.lam = lam if lam is not None else _c["lambda"]
        self.n_iters = n_iters if n_iters is not None else _c["n_iters"]
        self.delta_q = delta_q if delta_q is not None else _c["delta_q"]
        self.min_clearance = min_clearance if min_clearance is not None else _c["min_clearance"]
        self.danger_threshold = danger_threshold if danger_threshold is not None else _c["danger_threshold"]
        # Pre-compute the hinge pivot: 1 / d_safe
        self._inv_d_safe = 1.0 / self.danger_threshold

    # ── cost terms ──────────────────────────────────

    @staticmethod
    def smoothness_cost(path: List[np.ndarray]) -> float:
        """Acceleration-based smoothness cost (sum of squared finite diffs)."""
        total = 0.0
        for i in range(1, len(path) - 1):
            accel = path[i + 1] - 2.0 * path[i] + path[i - 1]
            total += float(np.dot(accel, accel))
        return total

    def obstacle_cost(self, path: List[np.ndarray]) -> float:
        """Hinge-loss obstacle cost (only penalises within danger zone).

        For each interior waypoint q_i:
            h_i = max(0, 1/d_min(q_i) - 1/d_safe)^2
        Total = sum_i h_i

        Waypoints further than ``danger_threshold`` contribute **zero**.
        """
        total = 0.0
        for i in range(1, len(path) - 1):
            d_min, _, _ = self.cc.min_link_obstacle_distance(path[i])
            if d_min < 1e-6:
                d_min = 1e-6
            if d_min < self.danger_threshold:
                h = 1.0 / d_min - self._inv_d_safe
                total += h * h
        return total

    def total_cost(self, path: List[np.ndarray]) -> float:
        return self.smoothness_cost(path) + self.lam * self.obstacle_cost(path)

    # ── gradients ───────────────────────────────────

    @staticmethod
    def _smooth_gradient(path: List[np.ndarray], i: int) -> np.ndarray:
        """Gradient of J_smooth w.r.t. waypoint q_i.

        q_i appears in acceleration terms a_{i-1}, a_i, a_{i+1}.
        dJ/dq_i = 2*(a_{i-1} - 2*a_i + a_{i+1})  for interior i with
        boundary handling when i-2 or i+2 don't exist.
        """
        n = len(path)
        grad = np.zeros(PANDA_NUM_JOINTS)

        # a_{i-1} contribution (exists if i >= 2)
        if i >= 2:
            a_prev = path[i] - 2.0 * path[i - 1] + path[i - 2]
            grad += 2.0 * a_prev

        # a_i contribution (always exists for interior i)
        a_i = path[i + 1] - 2.0 * path[i] + path[i - 1]
        grad -= 4.0 * a_i

        # a_{i+1} contribution (exists if i <= n-3)
        if i <= n - 3:
            a_next = path[i + 2] - 2.0 * path[i + 1] + path[i]
            grad += 2.0 * a_next

        return grad

    def _obstacle_gradient(self, q: np.ndarray) -> np.ndarray:
        """Gradient of hinge-loss obstacle cost via the Jacobian transpose.

        h(q) = max(0, 1/d - 1/d_safe)^2
        dh/dq = 2*(1/d - 1/d_safe) * (-1/d^2) * J^T * n_hat   if d < d_safe
              = 0                                                otherwise

        Uses ``getClosestPoints`` once to find contact normals, then
        ``calculateJacobian`` per active link — replaces the old 14-call
        finite-difference sweep.
        """
        # Sweep all obstacles — one set_config + N_obs getClosestPoints
        contact_info = self._per_link_contact_info(q)

        # Find global min distance for the hinge test
        d_min = min(
            (info[0] for info in contact_info.values()),
            default=float("inf"),
        )
        if d_min < 1e-6:
            d_min = 1e-6

        # Outside danger zone → zero gradient (skip all Jacobian work)
        if d_min >= self.danger_threshold:
            return np.zeros(PANDA_NUM_JOINTS)

        inv_d = 1.0 / d_min
        h_val = inv_d - self._inv_d_safe
        prefactor = 2.0 * h_val * (-inv_d * inv_d)

        # Accumulate J^T · n for every link within danger_threshold
        grad = np.zeros(PANDA_NUM_JOINTS)
        for link_idx, (rho_i, normal, pos_on_robot) in contact_info.items():
            if rho_i >= self.danger_threshold or rho_i < 1e-6:
                continue
            local_pos = self._world_to_link_local(link_idx, pos_on_robot)
            J = self._link_jacobian(q, link_idx, local_pos)
            # Contact normal points obstacle → robot; we want dd/dq ≈ J^T · n
            grad += J.T @ normal

        return prefactor * grad

    # ── Jacobian-transpose helpers (shared with CSpaceAPF) ──

    def _per_link_contact_info(self, q: np.ndarray):
        """Sweep ``getClosestPoints`` for every obstacle.

        Returns ``{link: (distance, contact_normal_3d, pos_on_robot_3d)}``.
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
            inv_pos, inv_orn,
            world_pos.tolist(), [0, 0, 0, 1],
        )
        return list(local)

    def _link_jacobian(self, q: np.ndarray, link_idx: int, local_pos):
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
        return np.array(lin_jac)[:, :PANDA_NUM_JOINTS]

    # ── optimiser ───────────────────────────────────

    def optimize(
        self, path: List[np.ndarray], verbose: bool = False,
    ) -> List[np.ndarray]:
        """Run gradient-descent smoothing on a collision-free path.

        Parameters
        ----------
        path : list of np.ndarray
            Waypoints (start and goal are pinned).
        verbose : bool
            Print cost every 10 iterations and timing breakdown.

        Returns
        -------
        list of np.ndarray
            Optimised path (same start/goal, potentially fewer interior points).
        """
        if len(path) <= 2:
            return list(path)

        n_interior = len(path) - 2
        if verbose:
            print(f"  Optimizer: {n_interior} interior waypoints, "
                  f"lr={self.lr}, λ={self.lam}, max_iters={self.n_iters}")

        t_total_start = time.perf_counter()

        # Work on a mutable copy
        opt = [q.copy() for q in path]
        prev_cost = self.total_cost(opt)
        t_grad_total = 0.0
        t_collision_total = 0.0
        pybullet_calls = 0
        final_iter = 0

        for k in range(self.n_iters):
            final_iter = k
            if verbose and k % 10 == 0:
                j_s = self.smoothness_cost(opt)
                j_o = self.obstacle_cost(opt)
                print(f"    iter {k:3d}  J={prev_cost:.4f}  "
                      f"(smooth={j_s:.4f}, obs={j_o:.4f})")

            # Compute gradients for all interior waypoints
            t_g0 = time.perf_counter()
            grads = []
            for i in range(1, len(opt) - 1):
                g_smooth = self._smooth_gradient(opt, i)
                g_obs = self._obstacle_gradient(opt[i])
                # N_obs getClosestPoints + N_active calculateJacobian
                pybullet_calls += len(self.cc._obstacle_ids) + 1
                grads.append(g_smooth + self.lam * g_obs)
            t_grad_total += time.perf_counter() - t_g0

            # Apply updates – step-clipped to prevent teleporting
            max_step = 0.05  # max C-space movement per waypoint per iter
            any_moved = False
            t_c0 = time.perf_counter()
            for idx, i in enumerate(range(1, len(opt) - 1)):
                raw_step = -self.lr * grads[idx]
                # ── step clipping: cap the norm to prevent teleporting ──
                step_norm = np.linalg.norm(raw_step)
                if step_norm > max_step:
                    raw_step = raw_step * (max_step / step_norm)
                q_new = np.clip(
                    opt[i] + raw_step, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS,
                )
                # Only accept if the config is collision-free
                d_min, _, _ = self.cc.min_link_obstacle_distance(q_new)
                pybullet_calls += 1
                if d_min > self.min_clearance:
                    if step_norm > 1e-8:
                        any_moved = True
                    opt[i] = q_new
            t_collision_total += time.perf_counter() - t_c0

            if not any_moved:
                if verbose:
                    print(f"    converged at iter {k} (no movement)")
                break

            # Early stop on cost convergence
            cur_cost = self.total_cost(opt)
            rel_change = abs(prev_cost - cur_cost) / (abs(prev_cost) + 1e-12)
            if rel_change < 1e-4:
                if verbose:
                    print(f"    converged at iter {k} "
                          f"(rel cost change {rel_change:.2e})")
                break
            prev_cost = cur_cost

        t_total = time.perf_counter() - t_total_start

        if verbose:
            print(f"    final J={self.total_cost(opt):.4f}  "
                  f"after {final_iter + 1} iters")
            print(f"    Timing: total={t_total:.3f}s  "
                  f"gradients={t_grad_total:.3f}s  "
                  f"collision_checks={t_collision_total:.3f}s")
            print(f"    PyBullet getClosestPoints calls: ~{pybullet_calls}")
            in_len = self.path_length(path)
            out_len = self.path_length(opt)
            print(f"    Path length: {in_len:.3f} → {out_len:.3f} rad "
                  f"({'↓' if out_len < in_len else '↑'} "
                  f"{abs(out_len - in_len):.3f})")

        # Validate all edges; if any edge is invalid, fall back to original
        for i in range(len(opt) - 1):
            if not self.cc.is_edge_valid(opt[i], opt[i + 1]):
                if verbose:
                    print("    WARNING: optimised path has invalid edge, "
                          "returning original")
                return list(path)

        return opt

    @staticmethod
    def path_length(path: List[np.ndarray]) -> float:
        return sum(
            np.linalg.norm(path[i + 1] - path[i])
            for i in range(len(path) - 1)
        )

    @staticmethod
    def resample_path(
        path: List[np.ndarray], n_waypoints: int = 20,
    ) -> List[np.ndarray]:
        """Resample *path* to *n_waypoints* uniformly-spaced waypoints.

        Ensures the optimizer has enough interior points to work with,
        even when the shortcut smoother has collapsed the path to very
        few waypoints.
        """
        if len(path) <= 2 or n_waypoints <= 2:
            return list(path)

        # Cumulative arc-length along the path
        cum = [0.0]
        for i in range(1, len(path)):
            cum.append(cum[-1] + np.linalg.norm(path[i] - path[i - 1]))
        total_len = cum[-1]

        if total_len < 1e-12:
            return list(path)

        resampled: List[np.ndarray] = [path[0].copy()]
        seg = 0  # current segment index

        for k in range(1, n_waypoints - 1):
            target = total_len * k / (n_waypoints - 1)
            # Advance to the segment that contains target
            while seg < len(cum) - 2 and cum[seg + 1] < target:
                seg += 1
            seg_len = cum[seg + 1] - cum[seg]
            t = (target - cum[seg]) / seg_len if seg_len > 1e-12 else 0.0
            q = path[seg] + t * (path[seg + 1] - path[seg])
            resampled.append(q)

        resampled.append(path[-1].copy())
        return resampled
