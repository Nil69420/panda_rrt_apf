from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from panda_gym.pybullet import PyBullet

from common_utils.robot_constants import (
    PANDA_EE_LINK,
    PANDA_JOINT_INDICES,
    PANDA_JOINT_FORCES,
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PANDA_REST_POSES,
    PANDA_NUM_JOINTS,
)


@dataclass
class Obstacle:
    name: str
    shape: str
    position: np.ndarray
    radius: float
    height: float = 0.0
    body_id: int = -1


def _random_rgba(rng: np.random.Generator) -> np.ndarray:
    rgb = rng.uniform(0.15, 1.0, size=3)
    return np.append(rgb, 1.0)


class RRTEnvironment:
    TABLE_X_OFFSET = -0.3
    TABLE_LENGTH = 1.1
    TABLE_WIDTH = 0.7
    TABLE_HEIGHT = 0.4

    def __init__(
        self,
        render_mode: str = "human",
        n_substeps: int = 20,
    ) -> None:
        self.sim = PyBullet(render_mode=render_mode, n_substeps=n_substeps)
        self._p = self.sim.physics_client

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(
            length=self.TABLE_LENGTH,
            width=self.TABLE_WIDTH,
            height=self.TABLE_HEIGHT,
            x_offset=self.TABLE_X_OFFSET,
        )

        self.sim.loadURDF(
            body_name="panda",
            fileName="franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
        )
        self.robot_id: int = self.sim._bodies_idx["panda"]

        self.obstacles: List[Obstacle] = []
        self._set_joint_config(PANDA_REST_POSES)

    def _set_joint_config(self, q: np.ndarray) -> None:
        for i in range(PANDA_NUM_JOINTS):
            self._p.resetJointState(self.robot_id, i, q[i])

    def get_joint_config(self) -> np.ndarray:
        return np.array([
            self._p.getJointState(self.robot_id, i)[0]
            for i in range(PANDA_NUM_JOINTS)
        ])

    def get_ee_position(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        if q is not None:
            self._set_joint_config(q)
        state = self._p.getLinkState(self.robot_id, PANDA_EE_LINK)
        return np.array(state[0])

    def obstacle_body_ids(self) -> List[int]:
        return [o.body_id for o in self.obstacles]

    def obstacle_positions(self) -> List[np.ndarray]:
        return [o.position.copy() for o in self.obstacles]

    def spawn_obstacle_sphere(
        self,
        position: np.ndarray,
        radius: float = 0.05,
        name: Optional[str] = None,
        color: Optional[np.ndarray] = None,
    ) -> Obstacle:
        if name is None:
            name = f"obs_sphere_{len(self.obstacles)}"
        if color is None:
            color = np.array([0.8, 0.2, 0.2, 0.7])
        self.sim.create_sphere(
            body_name=name,
            radius=radius,
            mass=0.0,
            position=position,
            rgba_color=color,
        )
        bid = self.sim._bodies_idx[name]
        obs = Obstacle(name=name, shape="sphere", position=position.copy(),
                       radius=radius, body_id=bid)
        self.obstacles.append(obs)
        return obs

    def spawn_obstacle_cylinder(
        self,
        position: np.ndarray,
        radius: float = 0.03,
        height: float = 0.30,
        name: Optional[str] = None,
        color: Optional[np.ndarray] = None,
    ) -> Obstacle:
        if name is None:
            name = f"obs_cyl_{len(self.obstacles)}"
        if color is None:
            color = np.array([0.2, 0.6, 0.2, 0.7])
        self.sim.create_cylinder(
            body_name=name,
            radius=radius,
            height=height,
            mass=0.0,
            position=position,
            rgba_color=color,
        )
        bid = self.sim._bodies_idx[name]
        obs = Obstacle(
            name=name, shape="cylinder", position=position.copy(),
            radius=radius, height=height, body_id=bid,
        )
        self.obstacles.append(obs)
        return obs

    def generate_cluttered_canopy(
        self,
        n_obstacles: int = 10,
        rng: Optional[np.random.Generator] = None,
        x_range: Tuple[float, float] = (-0.15, 0.25),
        y_range: Tuple[float, float] = (-0.25, 0.25),
        z_range: Tuple[float, float] = (0.10, 0.50),
        sphere_ratio: float = 0.4,
    ) -> List[Obstacle]:
        if rng is None:
            rng = np.random.default_rng()

        placed: List[np.ndarray] = []
        created: List[Obstacle] = []
        min_sep = 0.08

        for i in range(n_obstacles):
            success = False
            for _ in range(200):
                pos = np.array([
                    rng.uniform(*x_range),
                    rng.uniform(*y_range),
                    rng.uniform(*z_range),
                ])
                if not all(np.linalg.norm(pos - p) > min_sep for p in placed):
                    continue

                if rng.random() < sphere_ratio:
                    r = rng.uniform(0.025, 0.05)
                    obs = self.spawn_obstacle_sphere(pos, radius=r,
                                                     color=_random_rgba(rng))
                else:
                    r = rng.uniform(0.015, 0.035)
                    h = rng.uniform(0.10, 0.30)
                    obs = self.spawn_obstacle_cylinder(pos, radius=r, height=h,
                                                       color=_random_rgba(rng))

                if self.is_config_collision_free(PANDA_REST_POSES):
                    placed.append(pos)
                    created.append(obs)
                    success = True
                    break
                else:
                    self.obstacles.remove(obs)
                    body_id = self.sim._bodies_idx.pop(obs.name)
                    self._p.removeBody(body_id)

            if not success:
                if rng.random() < sphere_ratio:
                    r = rng.uniform(0.025, 0.05)
                    obs = self.spawn_obstacle_sphere(pos, radius=r,
                                                     color=_random_rgba(rng))
                else:
                    r = rng.uniform(0.015, 0.035)
                    h = rng.uniform(0.10, 0.30)
                    obs = self.spawn_obstacle_cylinder(pos, radius=r, height=h,
                                                       color=_random_rgba(rng))
                placed.append(pos)
                created.append(obs)

        return created

    def is_config_collision_free(self, q: np.ndarray) -> bool:
        from panda_rrt.planner import CollisionChecker, SAFETY_MARGIN
        if np.any(q < PANDA_LOWER_LIMITS) or np.any(q > PANDA_UPPER_LIMITS):
            return False
        if not self.obstacles:
            return True
        cc = CollisionChecker(self._p, self.robot_id, self.obstacle_body_ids())
        return cc.is_config_valid(q)

    def is_config_valid(self, q: np.ndarray) -> bool:
        return self.is_config_collision_free(q)

    def is_edge_valid(self, q_from: np.ndarray, q_to: np.ndarray, n_checks: int = 8) -> bool:
        for i in range(1, n_checks + 1):
            t = i / n_checks
            q_interp = q_from + t * (q_to - q_from)
            if not self.is_config_valid(q_interp):
                return False
        return True

    def visualize_config(self, q: np.ndarray) -> None:
        self._set_joint_config(q)
        self.sim.step()

    def close(self) -> None:
        self.sim.close()
