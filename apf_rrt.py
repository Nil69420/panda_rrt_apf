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
from panda_rrt.collision import CollisionChecker
from panda_rrt.cspace_apf import CSpaceAPF
from panda_rrt.pure_rrt import Node, PlannerResult
from panda_rrt.config import get as cfg


class APFGuidedRRT:
    """RRT planner augmented with a C-space Artificial Potential Field.

    Each iteration either (a) steers toward the goal, (b) follows the
    APF force, or (c) takes a standard random-explore step — chosen
    stochastically via ``goal_bias`` and ``apf_bias``.
    """

    def __init__(
        self,
        env,
        goal_bias: float | None = None,
        apf_bias: float | None = None,
        step_size: float | None = None,
        max_iterations: int | None = None,
        goal_threshold: float | None = None,
        k_att: float | None = None,
        k_rep: float | None = None,
        rho_0: float | None = None,
        seed: Optional[int] = None,
    ) -> None:
        _c = cfg("apf_rrt")
        self.env = env
        self.goal_bias = goal_bias if goal_bias is not None else _c["goal_bias"]
        self.apf_bias = apf_bias if apf_bias is not None else _c["apf_bias"]
        self.step_size = step_size if step_size is not None else _c["step_size"]
        self.max_iterations = max_iterations if max_iterations is not None else _c["max_iterations"]
        self.goal_threshold = goal_threshold if goal_threshold is not None else _c["goal_threshold"]
        self.rng = np.random.default_rng(seed)
        self.nodes: List[Node] = []

        self.cc = CollisionChecker(
            env._p, env.robot_id, env.obstacle_body_ids(),
        )
        self.apf = CSpaceAPF(
            self.cc, k_att=k_att, k_rep=k_rep, rho_0=rho_0,
        )

    # ── sampling helpers ────────────────────────────

    def _random_config(self) -> np.ndarray:
        return PANDA_LOWER_LIMITS + self.rng.random(PANDA_NUM_JOINTS) * PANDA_JOINT_RANGES

    def _nearest(self, q: np.ndarray) -> int:
        dists = [np.linalg.norm(q - n.q) for n in self.nodes]
        return int(np.argmin(dists))

    def _steer(self, q_near: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        diff = q_target - q_near
        dist = np.linalg.norm(diff)
        if dist < self.step_size:
            return q_target.copy()
        return q_near + (diff / dist) * self.step_size

    def _apf_step(self, q_near: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        f_total = self.apf.total_force(q_near, q_goal)
        f_norm = np.linalg.norm(f_total)
        if f_norm < 1e-8:
            return self._random_config()
        q_new = q_near + self.step_size * (f_total / f_norm)
        return np.clip(q_new, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)

    # ── tree expansion ──────────────────────────────

    def _sample_and_extend(self, q_goal: np.ndarray) -> Optional[int]:
        """Pick a strategy, find nearest node, propose ``q_new``.

        Returns the new-node index on success, ``None`` if rejected.
        """
        r = self.rng.random()
        apf_threshold = self.goal_bias + self.apf_bias

        # Choose strategy
        if r < self.goal_bias:
            idx_near = self._nearest(q_goal)
            q_near = self.nodes[idx_near].q
            q_new = self._steer(q_near, q_goal)
        elif r < apf_threshold:
            idx_near = self._nearest(self._random_config())
            q_near = self.nodes[idx_near].q
            q_new = self._apf_step(q_near, q_goal)
        else:
            q_rand = self._random_config()
            idx_near = self._nearest(q_rand)
            q_near = self.nodes[idx_near].q
            q_new = self._steer(q_near, q_rand)

        # Validate
        if not self.cc.is_config_valid(q_new):
            return None
        if not self.cc.is_edge_valid(q_near, q_new):
            return None

        # Insert
        cost_new = self.nodes[idx_near].cost + np.linalg.norm(q_new - q_near)
        self.nodes.append(Node(q=q_new, parent=idx_near, cost=cost_new))
        return len(self.nodes) - 1

    def _try_connect_goal(self, new_idx: int, q_goal: np.ndarray) -> Optional[int]:
        """If ``q_new`` is close enough to the goal, try a direct edge."""
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
        return sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))

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

        # Fast-fail on invalid start / goal
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
