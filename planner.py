from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from common_utils.robot_constants import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PANDA_JOINT_RANGES,
    PANDA_NUM_JOINTS,
)

from panda_rrt.pure_rrt import Node, PlannerResult


COLLISION_LINKS = [3, 4, 5, 6, 7, 8, 9, 10, 11]
SAFETY_MARGIN = 0.01


class CollisionChecker:
    def __init__(self, physics_client, robot_id: int, obstacle_ids: List[int]) -> None:
        self._p = physics_client
        self._robot_id = robot_id
        self._obstacle_ids = obstacle_ids

    def set_config(self, q: np.ndarray) -> None:
        for i in range(PANDA_NUM_JOINTS):
            self._p.resetJointState(self._robot_id, i, q[i])

    def min_link_obstacle_distance(self, q: np.ndarray) -> Tuple[float, int, int]:
        self.set_config(q)
        min_d = float("inf")
        min_link = -1
        min_obs = -1

        for obs_id in self._obstacle_ids:
            contacts = self._p.getClosestPoints(
                bodyA=self._robot_id,
                bodyB=obs_id,
                distance=1.0,
            )
            for cp in contacts:
                link_idx = cp[3]
                if link_idx not in COLLISION_LINKS:
                    continue
                d = cp[8]
                if d < min_d:
                    min_d = d
                    min_link = link_idx
                    min_obs = obs_id
        return min_d, min_link, min_obs

    def per_link_distances(self, q: np.ndarray) -> Dict[int, float]:
        self.set_config(q)
        dists: Dict[int, float] = {lk: float("inf") for lk in COLLISION_LINKS}

        for obs_id in self._obstacle_ids:
            contacts = self._p.getClosestPoints(
                bodyA=self._robot_id,
                bodyB=obs_id,
                distance=1.0,
            )
            for cp in contacts:
                link_idx = cp[3]
                if link_idx in dists:
                    d = cp[8]
                    if d < dists[link_idx]:
                        dists[link_idx] = d
        return dists

    def is_config_valid(self, q: np.ndarray) -> bool:
        if np.any(q < PANDA_LOWER_LIMITS) or np.any(q > PANDA_UPPER_LIMITS):
            return False
        d_min, _, _ = self.min_link_obstacle_distance(q)
        return d_min > SAFETY_MARGIN

    def is_edge_valid(self, q_from: np.ndarray, q_to: np.ndarray, n_checks: int = 8) -> bool:
        for i in range(1, n_checks + 1):
            t = i / n_checks
            q_interp = q_from + t * (q_to - q_from)
            if not self.is_config_valid(q_interp):
                return False
        return True


class CSpaceAPF:
    def __init__(
        self,
        collision_checker: CollisionChecker,
        k_att: float = 1.0,
        k_rep: float = 0.01,
        rho_0: float = 0.15,
        delta_q: float = 0.01,
    ) -> None:
        self.cc = collision_checker
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
        self.delta_q = delta_q

    def attractive_gradient(self, q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        return self.k_att * (q - q_goal)

    def _numerical_distance_gradient(self, q: np.ndarray) -> Tuple[Dict[int, float], Dict[int, np.ndarray]]:
        dists_center = self.cc.per_link_distances(q)

        # Early exit: skip expensive finite-difference gradient when no link
        # is within the repulsive influence range.
        active_links = [lk for lk in COLLISION_LINKS if dists_center[lk] < self.rho_0]
        if not active_links:
            return dists_center, {lk: np.zeros(PANDA_NUM_JOINTS) for lk in COLLISION_LINKS}

        grad_rho: Dict[int, np.ndarray] = {lk: np.zeros(PANDA_NUM_JOINTS) for lk in COLLISION_LINKS}
        for j in range(PANDA_NUM_JOINTS):
            q_plus = q.copy()
            q_plus[j] += self.delta_q
            q_plus = np.clip(q_plus, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)
            d_plus = self.cc.per_link_distances(q_plus)

            q_minus = q.copy()
            q_minus[j] -= self.delta_q
            q_minus = np.clip(q_minus, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)
            d_minus = self.cc.per_link_distances(q_minus)

            actual_delta = (q_plus[j] - q_minus[j])
            if abs(actual_delta) > 1e-10:
                for link in active_links:
                    grad_rho[link][j] = (d_plus[link] - d_minus[link]) / actual_delta

        return dists_center, grad_rho

    def repulsive_gradient(self, q: np.ndarray) -> np.ndarray:
        dists, grad_rho = self._numerical_distance_gradient(q)
        grad = np.zeros(PANDA_NUM_JOINTS)

        for link in COLLISION_LINKS:
            rho_i = dists[link]
            if rho_i >= self.rho_0 or rho_i < 1e-6:
                continue
            coeff = self.k_rep * (1.0 / self.rho_0 - 1.0 / rho_i) / (rho_i * rho_i)
            grad += coeff * grad_rho[link]

        return grad

    def total_force(self, q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        grad_att = self.attractive_gradient(q, q_goal)
        grad_rep = self.repulsive_gradient(q)
        return -grad_att - grad_rep


class APFGuidedRRT:
    def __init__(
        self,
        env,
        goal_bias: float = 0.10,
        apf_bias: float = 0.30,
        step_size: float = 0.30,
        max_iterations: int = 10000,
        goal_threshold: float = 0.40,
        k_att: float = 1.0,
        k_rep: float = 0.01,
        rho_0: float = 0.15,
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.goal_bias = goal_bias
        self.apf_bias = apf_bias
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold
        self.rng = np.random.default_rng(seed)
        self.nodes: List[Node] = []

        self.cc = CollisionChecker(
            env._p, env.robot_id, env.obstacle_body_ids(),
        )
        self.apf = CSpaceAPF(
            self.cc, k_att=k_att, k_rep=k_rep, rho_0=rho_0,
        )

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
        q_new = np.clip(q_new, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)
        return q_new

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
            if self.cc.is_edge_valid(smoothed[i], smoothed[j], n_checks=n_checks):
                smoothed = smoothed[:i+1] + smoothed[j:]
        return smoothed

    def _path_length(self, path: List[np.ndarray]) -> float:
        return sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path) - 1))

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlannerResult:
        import time
        t0 = time.perf_counter()

        self.nodes = [Node(q=q_start.copy(), parent=None, cost=0.0)]

        if not self.cc.is_config_valid(q_start):
            return PlannerResult(success=False, planning_time=time.perf_counter() - t0)
        if not self.cc.is_config_valid(q_goal):
            return PlannerResult(success=False, planning_time=time.perf_counter() - t0)

        for iteration in range(1, self.max_iterations + 1):
            r = self.rng.random()

            if r < self.goal_bias:
                # Goal bias: steer toward goal
                idx_near = self._nearest(q_goal)
                q_near = self.nodes[idx_near].q
                q_new = self._steer(q_near, q_goal)
            elif r < self.goal_bias + self.apf_bias:
                # APF bias: use potential field to guide expansion
                q_rand = self._random_config()
                idx_near = self._nearest(q_rand)
                q_near = self.nodes[idx_near].q
                q_new = self._apf_step(q_near, q_goal)
            else:
                # Standard RRT exploration
                q_rand = self._random_config()
                idx_near = self._nearest(q_rand)
                q_near = self.nodes[idx_near].q
                q_new = self._steer(q_near, q_rand)

            if not self.cc.is_config_valid(q_new):
                continue
            if not self.cc.is_edge_valid(q_near, q_new):
                continue

            cost_new = self.nodes[idx_near].cost + np.linalg.norm(q_new - q_near)
            new_node = Node(q=q_new, parent=idx_near, cost=cost_new)
            self.nodes.append(new_node)
            new_idx = len(self.nodes) - 1

            d_to_goal = np.linalg.norm(q_new - q_goal)
            if d_to_goal < self.goal_threshold:
                nc = max(5, int(d_to_goal / 0.05))
                if self.cc.is_edge_valid(q_new, q_goal, n_checks=nc):
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
