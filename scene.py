"""Scene management: obstacle spawning and goal/start generation."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from panda_rrt.common_utils.robot_constants import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PANDA_NUM_JOINTS,
)
from panda_rrt.config import get as cfg
from panda_rrt.environment import RRTEnvironment


def _fmt_pos(p: np.ndarray) -> str:
    return f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"


def load_demo_obstacles() -> List[Dict]:
    """Return the obstacle list from config."""
    return cfg("demo", "obstacles")


def load_demo_goal() -> np.ndarray:
    """Return the demo goal configuration from config."""
    return np.array(cfg("demo", "goal"))


def spawn_obstacles(env: RRTEnvironment, obstacles: List[Dict] | None = None) -> None:
    """Spawn obstacles in *env* and log each one.

    If *obstacles* is ``None``, loads the demo set from config.
    """
    if obstacles is None:
        obstacles = load_demo_obstacles()

    print("--- Spawning obstacles ---")
    for obs_def in obstacles:
        pos = np.array(obs_def["pos"])
        r = obs_def["radius"]
        color = np.array(obs_def.get("color", [0.8, 0.2, 0.2, 0.7]))
        name = obs_def["name"]

        if obs_def["shape"] == "sphere":
            env.spawn_obstacle_sphere(pos, radius=r, name=name, color=color)
            print(f"  {name:12s}  sphere   pos={_fmt_pos(pos)}  r={r:.3f}")
        else:
            h = obs_def.get("height", 0.18)
            env.spawn_obstacle_cylinder(pos, radius=r, height=h, name=name, color=color)
            print(f"  {name:12s}  cylinder pos={_fmt_pos(pos)}  r={r:.3f}  h={h:.2f}")

    print(f"  Total: {len(env.obstacles)} obstacles\n")


def sample_random_config(
    env: RRTEnvironment,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 500,
) -> np.ndarray:
    """Sample a random collision-free joint configuration."""
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(max_attempts):
        q = rng.uniform(PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)
        if env.is_config_valid(q):
            return q
    raise RuntimeError(
        f"Could not find a collision-free config in {max_attempts} attempts"
    )


def sample_random_start_goal(
    env: RRTEnvironment,
    seed: Optional[int] = None,
    min_c_dist: float = 1.0,
    max_attempts: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a random (start, goal) pair that are collision-free and
    separated by at least *min_c_dist* in C-space."""
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        q_start = sample_random_config(env, rng)
        q_goal = sample_random_config(env, rng)
        if np.linalg.norm(q_goal - q_start) >= min_c_dist:
            return q_start, q_goal
    raise RuntimeError(
        f"Could not find a valid start/goal pair with d >= {min_c_dist} "
        f"in {max_attempts} attempts"
    )
