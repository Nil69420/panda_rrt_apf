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
from panda_rrt.collision import CollisionChecker
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
        """Gradient of hinge-loss obstacle cost for waypoint *q*.

        h(q) = max(0, 1/d - 1/d_safe)^2
        dh/dq = 2*(1/d - 1/d_safe) * (-1/d^2) * dd/dq   if d < d_safe
              = 0                                          otherwise

        Uses central-difference for dd/dq.
        """
        d_center, _, _ = self.cc.min_link_obstacle_distance(q)
        if d_center < 1e-6:
            d_center = 1e-6

        # Outside danger zone → zero gradient (big speedup)
        if d_center >= self.danger_threshold:
            return np.zeros(PANDA_NUM_JOINTS)

        inv_d = 1.0 / d_center
        h_val = inv_d - self._inv_d_safe       # positive because d < d_safe
        prefactor = 2.0 * h_val * (-inv_d * inv_d)  # 2*(1/d - 1/d_safe)*(-1/d²)

        grad = np.zeros(PANDA_NUM_JOINTS)
        for j in range(PANDA_NUM_JOINTS):
            q_plus = q.copy()
            q_plus[j] += self.delta_q
            q_plus = np.clip(q_plus, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)

            q_minus = q.copy()
            q_minus[j] -= self.delta_q
            q_minus = np.clip(q_minus, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)

            actual_delta = q_plus[j] - q_minus[j]
            if abs(actual_delta) < 1e-10:
                continue

            d_plus, _, _ = self.cc.min_link_obstacle_distance(q_plus)
            d_minus, _, _ = self.cc.min_link_obstacle_distance(q_minus)
            if d_plus < 1e-6:
                d_plus = 1e-6
            if d_minus < 1e-6:
                d_minus = 1e-6

            dd_dq = (d_plus - d_minus) / actual_delta
            grad[j] = prefactor * dd_dq

        return grad

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
                # 2 * PANDA_NUM_JOINTS + 1 pybullet calls per interior point
                pybullet_calls += 2 * 7 + 1
                grads.append(g_smooth + self.lam * g_obs)
            t_grad_total += time.perf_counter() - t_g0

            # Apply updates
            any_moved = False
            t_c0 = time.perf_counter()
            for idx, i in enumerate(range(1, len(opt) - 1)):
                step = -self.lr * grads[idx]
                q_new = np.clip(
                    opt[i] + step, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS,
                )
                # Only accept if the new config is collision-free
                d_min, _, _ = self.cc.min_link_obstacle_distance(q_new)
                pybullet_calls += 1
                if d_min > self.min_clearance:
                    if np.linalg.norm(step) > 1e-8:
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
