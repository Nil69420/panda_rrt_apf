"""RRT* (asymptotically optimal RRT) planner for the Panda arm.

Key differences from vanilla RRT:

1. **Near-neighbour search** within a shrinking ball radius.
2. **Best parent** -- pick the near neighbour that minimises cost-to-root
   through the new node.
3. **Rewire** -- after insertion, check whether routing through the new
   node improves cost for other neighbours.

Ball radius follows the RRT* formula:

.. math::

    r_n = \\gamma \\left(\\frac{\\log n}{n}\\right)^{1/d}

In plain terms: the search radius shrinks as the tree grows --
radius = gamma * (log(n) / n)^(1/d), where n is the number of nodes
and d is the number of joints. This guarantees the planner converges
to the optimal path as the tree gets larger.

Uses JIT-accelerated nearest-neighbour search when Numba is available.
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
from panda_rrt.computations.collision import CollisionChecker
from panda_rrt.computations.jit_kernels import nearest_node_idx, steer, near_indices
from panda_rrt.planners.core import Node, PlannerResult
from panda_rrt.config import get as cfg

_BUFFER_CHUNK = 2048


class RRTStar:
    """RRT* planner with near-neighbour rewiring.

    Parameters
    ----------
    env : RRTEnvironment
    goal_bias : float
        Probability of sampling the goal directly.
    step_size : float
        Maximum extension distance (radians).
    max_iterations : int
    goal_threshold : float
    rewire_radius : float
        Maximum near-neighbour ball radius.
    seed : int or None
    """

    def __init__(
        self,
        env,
        goal_bias: float = None,
        step_size: float = None,
        max_iterations: int = None,
        goal_threshold: float = None,
        rewire_radius: float = None,
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
        self._nodes_q = np.empty((_BUFFER_CHUNK, PANDA_NUM_JOINTS))

        self.cc = CollisionChecker(
            env._p, env.robot_id, env.obstacle_body_ids(),
        )

    # ── buffer management ───────────────────────────

    def _ensure_capacity(self) -> None:
        n = len(self.nodes)
        if n >= self._nodes_q.shape[0]:
            self._nodes_q = np.resize(
                self._nodes_q, (n + _BUFFER_CHUNK, PANDA_NUM_JOINTS),
            )

    def _add_node(self, q: np.ndarray, parent: Optional[int], cost: float) -> int:
        idx = len(self.nodes)
        self._ensure_capacity()
        self._nodes_q[idx] = q
        self.nodes.append(Node(q=q, parent=parent, cost=cost))
        return idx

    # ── sampling helpers ────────────────────────────

    def _random_config(self) -> np.ndarray:
        return PANDA_LOWER_LIMITS + self.rng.random(PANDA_NUM_JOINTS) * PANDA_JOINT_RANGES

    def _nearest(self, q: np.ndarray) -> int:
        return nearest_node_idx(self._nodes_q, len(self.nodes), q)

    def _near(self, q: np.ndarray, radius: float) -> List[int]:
        """Return indices of all nodes within *radius* of *q*.

        Delegates to a parallel JIT kernel that computes squared
        distances without allocating a temporary (N, 7) difference
        array.
        """
        idxs = near_indices(self._nodes_q, len(self.nodes), q, radius)
        return list(idxs)

    def _shrinking_radius(self) -> float:
        """RRT* ball radius: :math:`\\gamma (\\log n / n)^{1/d}`.

        In plain terms: radius = gamma * (log(tree_size) / tree_size)^(1/num_joints).
        """
        n = max(len(self.nodes), 2)
        d = PANDA_NUM_JOINTS
        gamma = self.rewire_radius * (n ** (1.0 / d)) / max((math.log(n) ** (1.0 / d)), 1e-6)
        return min(gamma * (math.log(n) / n) ** (1.0 / d), self.rewire_radius)

    # ── tree expansion ──────────────────────────────

    def _sample_and_extend(self, q_goal: np.ndarray) -> Optional[int]:
        """Sample, steer, choose best parent, insert, rewire.

        Returns
        -------
        int or None
            Index of the newly inserted node, or ``None`` if rejected.
        """
        # 1. Sample
        q_sample = (
            q_goal.copy()
            if self.rng.random() < self.goal_bias
            else self._random_config()
        )

        # 2. Nearest + steer
        idx_nearest = self._nearest(q_sample)
        q_new = steer(self.nodes[idx_nearest].q, q_sample, self.step_size)

        # 3. Validate
        if not self.cc.is_config_valid(q_new):
            return None

        # 4. Near neighbours
        radius = self._shrinking_radius()
        near_idxs = self._near(q_new, radius)
        if not near_idxs:
            near_idxs = [idx_nearest]

        # 5. Best parent (minimum cost-to-root + edge)
        best_parent = None
        best_cost = float("inf")
        for ni in near_idxs:
            edge_cost = np.linalg.norm(q_new - self.nodes[ni].q)
            candidate = self.nodes[ni].cost + edge_cost
            if candidate < best_cost and self.cc.is_edge_valid(self.nodes[ni].q, q_new):
                best_cost = candidate
                best_parent = ni

        if best_parent is None:
            return None

        # 6. Insert
        new_idx = self._add_node(q_new, best_parent, best_cost)

        # 7. Rewire neighbours through q_new if cheaper
        for ni in near_idxs:
            if ni == best_parent:
                continue
            edge_cost = np.linalg.norm(q_new - self.nodes[ni].q)
            candidate = best_cost + edge_cost
            if candidate < self.nodes[ni].cost:
                if self.cc.is_edge_valid(q_new, self.nodes[ni].q):
                    delta = candidate - self.nodes[ni].cost
                    self.nodes[ni].parent = new_idx
                    self.nodes[ni].cost = candidate
                    self._propagate_cost(ni, delta)

        return new_idx

    def _propagate_cost(self, idx: int, delta: float) -> None:
        """Propagate a cost delta down the sub-tree rooted at *idx*."""
        for i, node in enumerate(self.nodes):
            if node.parent == idx:
                node.cost += delta
                self._propagate_cost(i, delta)

    def _try_connect_goal(self, new_idx: int, q_goal: np.ndarray) -> Optional[int]:
        q_new = self.nodes[new_idx].q
        d = np.linalg.norm(q_new - q_goal)
        if d >= self.goal_threshold:
            return None
        nc = max(5, int(d / cfg("smoothing", "resolution")))
        if not self.cc.is_edge_valid(q_new, q_goal, n_checks=nc):
            return None
        return self._add_node(q_goal.copy(), new_idx, self.nodes[new_idx].cost + d)

    # ── path extraction and smoothing ───────────────

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

    # ── public API ──────────────────────────────────

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlannerResult:
        """Run RRT* from *q_start* to *q_goal*.

        Parameters
        ----------
        q_start, q_goal : ndarray, shape (7,)

        Returns
        -------
        PlannerResult
        """
        t0 = time.perf_counter()
        self.nodes = []
        self._nodes_q = np.empty((_BUFFER_CHUNK, PANDA_NUM_JOINTS))
        self._add_node(q_start.copy(), parent=None, cost=0.0)

        if not self.cc.is_config_valid(q_start) or not self.cc.is_config_valid(q_goal):
            return PlannerResult(success=False, planning_time=time.perf_counter() - t0)

        for iteration in range(1, self.max_iterations + 1):
            new_idx = self._sample_and_extend(q_goal)
            if new_idx is None:
                continue
            goal_idx = self._try_connect_goal(new_idx, q_goal)
            if goal_idx is not None:
                return self._build_result(self._extract_path(goal_idx), iteration, t0)

        return PlannerResult(
            success=False,
            tree_size=len(self.nodes),
            iterations=self.max_iterations,
            planning_time=time.perf_counter() - t0,
        )
