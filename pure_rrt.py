from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from common_utils.robot_constants import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PANDA_JOINT_RANGES,
    PANDA_NUM_JOINTS,
)


@dataclass
class Node:
    q: np.ndarray
    parent: Optional[int] = None
    cost: float = 0.0


@dataclass
class PlannerResult:
    success: bool
    path: List[np.ndarray] = field(default_factory=list)
    smoothed_path: List[np.ndarray] = field(default_factory=list)
    tree_size: int = 0
    iterations: int = 0
    planning_time: float = 0.0
    path_length: float = 0.0
    smoothed_length: float = 0.0


class PureRRT:
    def __init__(
        self,
        env,
        goal_bias: float = 0.10,
        step_size: float = 0.30,
        max_iterations: int = 10000,
        goal_threshold: float = 0.40,
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.goal_bias = goal_bias
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold
        self.rng = np.random.default_rng(seed)
        self.nodes: List[Node] = []

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

    def _extract_path(self, goal_idx: int) -> List[np.ndarray]:
        path = []
        idx = goal_idx
        while idx is not None:
            path.append(self.nodes[idx].q.copy())
            idx = self.nodes[idx].parent
        path.reverse()
        return path

    def _smooth_path(self, path: List[np.ndarray], max_attempts: int = 200) -> List[np.ndarray]:
        if len(path) <= 2:
            return list(path)
        smoothed = list(path)
        for _ in range(max_attempts):
            if len(smoothed) <= 2:
                break
            i = self.rng.integers(0, len(smoothed) - 2)
            j = self.rng.integers(i + 2, len(smoothed))
            dist = np.linalg.norm(smoothed[j] - smoothed[i])
            n_checks = max(8, int(dist / 0.05))
            if self.env.is_edge_valid(smoothed[i], smoothed[j], n_checks=n_checks):
                smoothed = smoothed[:i+1] + smoothed[j:]
        return smoothed

    def _path_length(self, path: List[np.ndarray]) -> float:
        return sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path) - 1))

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlannerResult:
        import time
        t0 = time.perf_counter()

        self.nodes = [Node(q=q_start.copy(), parent=None, cost=0.0)]

        if not self.env.is_config_valid(q_start):
            return PlannerResult(success=False, planning_time=time.perf_counter() - t0)
        if not self.env.is_config_valid(q_goal):
            return PlannerResult(success=False, planning_time=time.perf_counter() - t0)

        for iteration in range(1, self.max_iterations + 1):
            if self.rng.random() < self.goal_bias:
                q_sample = q_goal.copy()
            else:
                q_sample = self._random_config()

            idx_near = self._nearest(q_sample)
            q_near = self.nodes[idx_near].q
            q_new = self._steer(q_near, q_sample)

            if not self.env.is_config_valid(q_new):
                continue
            if not self.env.is_edge_valid(q_near, q_new):
                continue

            cost_new = self.nodes[idx_near].cost + np.linalg.norm(q_new - q_near)
            new_node = Node(q=q_new, parent=idx_near, cost=cost_new)
            self.nodes.append(new_node)
            new_idx = len(self.nodes) - 1

            d_to_goal = np.linalg.norm(q_new - q_goal)
            if d_to_goal < self.goal_threshold:
                nc = max(5, int(d_to_goal / 0.05))
                if self.env.is_edge_valid(q_new, q_goal, n_checks=nc):
                    goal_cost = cost_new + d_to_goal
                    goal_node = Node(q=q_goal.copy(), parent=new_idx, cost=goal_cost)
                    self.nodes.append(goal_node)
                    goal_idx = len(self.nodes) - 1

                    path = self._extract_path(goal_idx)
                    smoothed = self._smooth_path(path)
                    elapsed = time.perf_counter() - t0

                    return PlannerResult(
                        success=True,
                        path=path,
                        smoothed_path=smoothed,
                        tree_size=len(self.nodes),
                        iterations=iteration,
                        planning_time=elapsed,
                        path_length=self._path_length(path),
                        smoothed_length=self._path_length(smoothed),
                    )

        elapsed = time.perf_counter() - t0
        return PlannerResult(
            success=False,
            tree_size=len(self.nodes),
            iterations=self.max_iterations,
            planning_time=elapsed,
        )
