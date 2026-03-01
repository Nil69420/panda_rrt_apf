"""RRT* (asymptotically optimal RRT) planner for the Panda arm.

Key differences from vanilla RRT:
  1. **Near-neighbour search** within a shrinking radius around ``q_new``.
  2. **Choose best parent** — pick the near neighbour that minimises
     cost-to-root through ``q_new``.
  3. **Rewire** — after inserting ``q_new``, check whether routing
     through ``q_new`` improves cost for other neighbours.

Returns the *first* feasible path found (like the other planners),
but the tree is higher-quality because of the rewiring.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional

import numpy as np

from panda_rrt.common_utils.robot_constants import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PANDA_JOINT_RANGES,
    PANDA_NUM_JOINTS,
)
from panda_rrt.collision import CollisionChecker
from panda_rrt.pure_rrt import Node, PlannerResult
from panda_rrt.config import get as cfg


class RRTStar:
    """RRT* planner with near-neighbour rewiring."""

    def __init__(
        self,
        env,
        goal_bias: float | None = None,
        step_size: float | None = None,
        max_iterations: int | None = None,
        goal_threshold: float | None = None,
        rewire_radius: float | None = None,
        seed: Optional[int] = None,
    ) -> None:
        _c = cfg("rrt_star")
        self.env = env
        self.goal_bias = goal_bias if goal_bias is not None else _c["goal_bias"]
        self.step_size = step_size if step_size is not None else _c["step_size"]
        self.max_iterations = max_iterations if max_iterations is not None else _c["max_iterations"]
        self.goal_threshold = goal_threshold if goal_threshold is not None else _c["goal_threshold"]
        self.rewire_radius = rewire_radius if rewire_radius is not None else _c["rewire_radius"]
        self.rng = np.random.default_rng(seed)
        self.nodes: List[Node] = []

        self.cc = CollisionChecker(
            env._p, env.robot_id, env.obstacle_body_ids(),
        )

    # ── sampling helpers ────────────────────────────

    def _random_config(self) -> np.ndarray:
        return PANDA_LOWER_LIMITS + self.rng.random(PANDA_NUM_JOINTS) * PANDA_JOINT_RANGES

    def _nearest(self, q: np.ndarray) -> int:
        dists = [np.linalg.norm(q - n.q) for n in self.nodes]
        return int(np.argmin(dists))

    def _near(self, q: np.ndarray, radius: float) -> List[int]:
        """Return indices of all nodes within *radius* of *q*."""
        return [
            i for i, n in enumerate(self.nodes)
            if np.linalg.norm(q - n.q) <= radius
        ]

    def _steer(self, q_near: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        diff = q_target - q_near
        dist = np.linalg.norm(diff)
        if dist < self.step_size:
            return q_target.copy()
        return q_near + (diff / dist) * self.step_size

    def _shrinking_radius(self) -> float:
        """RRT* ball radius: gamma * (log(n)/n)^(1/d)."""
        n = max(len(self.nodes), 2)
        d = PANDA_NUM_JOINTS
        # gamma chosen so initial radius ≈ rewire_radius
        gamma = self.rewire_radius * (n ** (1.0 / d)) / max((math.log(n) ** (1.0 / d)), 1e-6)
        radius = gamma * (math.log(n) / n) ** (1.0 / d)
        return min(radius, self.rewire_radius)

    # ── tree expansion ──────────────────────────────

    def _sample_and_extend(self, q_goal: np.ndarray) -> Optional[int]:
        """Sample, steer, choose best parent, insert, rewire.

        Returns new-node index on success, ``None`` if rejected.
        """
        # 1. Sample
        q_sample = (q_goal.copy()
                    if self.rng.random() < self.goal_bias
                    else self._random_config())

        # 2. Find nearest, steer
        idx_nearest = self._nearest(q_sample)
        q_nearest = self.nodes[idx_nearest].q
        q_new = self._steer(q_nearest, q_sample)

        # 3. Validate q_new
        if not self.cc.is_config_valid(q_new):
            return None

        # 4. Find near neighbours
        radius = self._shrinking_radius()
        near_idxs = self._near(q_new, radius)
        if not near_idxs:
            near_idxs = [idx_nearest]

        # 5. Choose best parent (minimum cost-to-root + edge cost)
        best_parent = None
        best_cost = float("inf")
        for ni in near_idxs:
            edge_cost = np.linalg.norm(q_new - self.nodes[ni].q)
            candidate_cost = self.nodes[ni].cost + edge_cost
            if candidate_cost < best_cost:
                if self.cc.is_edge_valid(self.nodes[ni].q, q_new):
                    best_cost = candidate_cost
                    best_parent = ni

        if best_parent is None:
            return None

        # 6. Insert
        new_idx = len(self.nodes)
        self.nodes.append(Node(q=q_new, parent=best_parent, cost=best_cost))

        # 7. Rewire — can we improve any neighbour by going through q_new?
        for ni in near_idxs:
            if ni == best_parent:
                continue
            edge_cost = np.linalg.norm(q_new - self.nodes[ni].q)
            candidate_cost = best_cost + edge_cost
            if candidate_cost < self.nodes[ni].cost:
                if self.cc.is_edge_valid(q_new, self.nodes[ni].q):
                    # Rewire: ni's parent becomes new_idx
                    old_cost = self.nodes[ni].cost
                    self.nodes[ni].parent = new_idx
                    self.nodes[ni].cost = candidate_cost
                    # Propagate cost improvement to subtree
                    self._propagate_cost(ni, candidate_cost - old_cost)

        return new_idx

    def _propagate_cost(self, idx: int, delta: float) -> None:
        """Propagate a cost change down the tree from *idx*."""
        for i, node in enumerate(self.nodes):
            if node.parent == idx:
                node.cost += delta
                self._propagate_cost(i, delta)

    def _try_connect_goal(self, new_idx: int, q_goal: np.ndarray) -> Optional[int]:
        q_new = self.nodes[new_idx].q
        d_to_goal = np.linalg.norm(q_new - q_goal)
        if d_to_goal >= self.goal_threshold:
            return None

        nc = max(5, int(d_to_goal / cfg("smoothing", "resolution")))
        if not self.cc.is_edge_valid(q_new, q_goal, n_checks=nc):
            return None

        goal_cost = self.nodes[new_idx].cost + d_to_goal
        self.nodes.append(Node(q=q_goal.copy(), parent=new_idx, cost=goal_cost))
        return len(self.nodes) - 1

    # ── path post-processing ────────────────────────

    def _extract_path(self, goal_idx: int) -> List[np.ndarray]:
        path: List[np.ndarray] = []
        idx: Optional[int] = goal_idx
        while idx is not None:
            path.append(self.nodes[idx].q.copy())
            idx = self.nodes[idx].parent
        path.reverse()
        return path

    def _smooth_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        max_attempts = cfg("smoothing", "max_attempts")
        resolution = cfg("smoothing", "resolution")
        if len(path) <= 2:
            return list(path)
        smoothed = list(path)
        for _ in range(max_attempts):
            if len(smoothed) <= 2:
                break
            i = self.rng.integers(0, len(smoothed) - 2)
            j = self.rng.integers(i + 2, len(smoothed))
            dist = np.linalg.norm(smoothed[j] - smoothed[i])
            n_checks = max(8, int(dist / resolution))
            if self.cc.is_edge_valid(smoothed[i], smoothed[j], n_checks=n_checks):
                smoothed = smoothed[: i + 1] + smoothed[j:]
        return smoothed

    @staticmethod
    def _path_length(path: List[np.ndarray]) -> float:
        return sum(
            np.linalg.norm(path[i + 1] - path[i])
            for i in range(len(path) - 1)
        )

    def _build_result(
        self, path: List[np.ndarray], iteration: int, t0: float,
    ) -> PlannerResult:
        smoothed = self._smooth_path(path)
        return PlannerResult(
            success=True,
            path=path,
            smoothed_path=smoothed,
            tree_size=len(self.nodes),
            iterations=iteration,
            planning_time=time.perf_counter() - t0,
            path_length=self._path_length(path),
            smoothed_length=self._path_length(smoothed),
        )

    # ── main entry point ────────────────────────────

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlannerResult:
        t0 = time.perf_counter()
        self.nodes = [Node(q=q_start.copy(), parent=None, cost=0.0)]

        if not self.cc.is_config_valid(q_start) or not self.cc.is_config_valid(q_goal):
            return PlannerResult(success=False, planning_time=time.perf_counter() - t0)

        for iteration in range(1, self.max_iterations + 1):
            new_idx = self._sample_and_extend(q_goal)
            if new_idx is None:
                continue

            goal_idx = self._try_connect_goal(new_idx, q_goal)
            if goal_idx is not None:
                path = self._extract_path(goal_idx)
                return self._build_result(path, iteration, t0)

        return PlannerResult(
            success=False,
            tree_size=len(self.nodes),
            iterations=self.max_iterations,
            planning_time=time.perf_counter() - t0,
        )
