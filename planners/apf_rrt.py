"""APF-Guided RRT planner: RRT with C-space potential-field bias.

Each iteration chooses one of three strategies stochastically:

1. **Goal bias** -- steer directly toward the goal.
2. **APF bias** -- follow the resultant APF force (attractive + repulsive).
3. **Explore** -- standard random-sample extension.

Uses JIT-accelerated nearest-neighbour search when Numba is available.
"""
from __future__ import annotations

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
from panda_rrt.computations.cspace_apf import CSpaceAPF
from panda_rrt.computations.jit_kernels import nearest_node_idx, steer
from panda_rrt.planners.core import Node, PlannerResult
from panda_rrt.config import get as cfg

_BUFFER_CHUNK = 2048


class APFGuidedRRT:
    """RRT planner augmented with a C-space Artificial Potential Field.

    Parameters
    ----------
    env : RRTEnvironment
    goal_bias, apf_bias : float
        Sampling strategy probabilities.
    step_size : float
        Maximum extension per iteration (radians).
    max_iterations : int
    goal_threshold : float
    k_att, k_rep, rho_0 : float
        APF parameters forwarded to :class:`CSpaceAPF`.
    seed : int or None
    """

    def __init__(
        self,
        env,
        goal_bias: float = None,
        apf_bias: float = None,
        step_size: float = None,
        max_iterations: int = None,
        goal_threshold: float = None,
        k_att: float = None,
        k_rep: float = None,
        rho_0: float = None,
        seed: Optional[int] = None,
    ) -> None:
        _c = cfg("apf_rrt")
        self.env = env
        self.goal_bias = goal_bias if goal_bias is not None else _c["goal_bias"]
        self.apf_bias = apf_bias if apf_bias is not None else _c["apf_bias"]
        self.step_size = step_size if step_size is not None else _c["step_size"]
        self.max_iterations = (
            max_iterations if max_iterations is not None else _c["max_iterations"]
        )
        self.goal_threshold = (
            goal_threshold if goal_threshold is not None else _c["goal_threshold"]
        )
        self.rng = np.random.default_rng(seed)
        self.nodes: List[Node] = []
        self._nodes_q = np.empty((_BUFFER_CHUNK, PANDA_NUM_JOINTS))

        self.cc = CollisionChecker(
            env._p, env.robot_id, env.obstacle_body_ids(),
        )
        self.apf = CSpaceAPF(self.cc, k_att=k_att, k_rep=k_rep, rho_0=rho_0)

    # ── Internal helpers ────────────────────────────

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

    def _random_config(self) -> np.ndarray:
        return PANDA_LOWER_LIMITS + self.rng.random(PANDA_NUM_JOINTS) * PANDA_JOINT_RANGES

    def _nearest(self, q: np.ndarray) -> int:
        return nearest_node_idx(self._nodes_q, len(self.nodes), q)

    def _apf_step(self, q_near: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        """Follow the APF force from *q_near*, clipped to joint limits."""
        f = self.apf.total_force(q_near, q_goal)
        if np.linalg.norm(f) < 1e-8:
            return self._random_config()
        q_new = q_near + self.step_size * (f / np.linalg.norm(f))
        return np.clip(q_new, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)

    # ── Tree expansion ──────────────────────────────

    def _sample_and_extend(self, q_goal: np.ndarray) -> Optional[int]:
        """Pick strategy, extend, validate, insert.

        Returns the new-node index on success, ``None`` on rejection.
        """
        r = self.rng.random()
        apf_threshold = self.goal_bias + self.apf_bias

        # Strategy dispatch via threshold comparison
        if r < self.goal_bias:
            idx_near = self._nearest(q_goal)
            q_near = self.nodes[idx_near].q
            q_new = steer(q_near, q_goal, self.step_size)
        elif r < apf_threshold:
            idx_near = self._nearest(self._random_config())
            q_near = self.nodes[idx_near].q
            q_new = self._apf_step(q_near, q_goal)
        else:
            q_rand = self._random_config()
            idx_near = self._nearest(q_rand)
            q_near = self.nodes[idx_near].q
            q_new = steer(q_near, q_rand, self.step_size)

        if not self.cc.is_config_valid(q_new):
            return None
        if not self.cc.is_edge_valid(q_near, q_new):
            return None

        cost = self.nodes[idx_near].cost + np.linalg.norm(q_new - q_near)
        return self._add_node(q_new, idx_near, cost)

    def _try_connect_goal(
        self, new_idx: int, q_goal: np.ndarray,
    ) -> Optional[int]:
        """If *new_idx* is within threshold, try a direct edge to goal."""
        q_new = self.nodes[new_idx].q
        d = np.linalg.norm(q_new - q_goal)
        if d >= self.goal_threshold:
            return None
        nc = max(5, int(d / cfg("smoothing", "resolution")))
        if not self.cc.is_edge_valid(q_new, q_goal, n_checks=nc):
            return None
        return self._add_node(
            q_goal.copy(), new_idx, self.nodes[new_idx].cost + d,
        )

    # ── Path extraction and smoothing ───────────────

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

    # ── Public API ──────────────────────────────────

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlannerResult:
        """Run APF-guided RRT from *q_start* to *q_goal*.

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
                return self._build_result(
                    self._extract_path(goal_idx), iteration, t0,
                )

        return PlannerResult(
            success=False,
            tree_size=len(self.nodes),
            iterations=self.max_iterations,
            planning_time=time.perf_counter() - t0,
        )
