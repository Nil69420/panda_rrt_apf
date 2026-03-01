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


def load_demo_obstacles(scene: str = "wall") -> List[Dict]:
    """Return the obstacle list from config for the given scene.

    Parameters
    ----------
    scene : str
        ``"wall"`` (original pillars) or ``"canopy"`` (overhead cage).
    """
    valid = ("wall", "canopy")
    if scene not in valid:
        raise ValueError(f"Unknown scene {scene!r}, expected one of {valid}")
    return cfg("demo", scene)


def load_demo_goal(scene: str = "wall") -> np.ndarray:
    """Return the demo goal configuration from config for the given scene."""
    key = f"{scene}_goal"
    return np.array(cfg("demo", key))


def spawn_obstacles(
    env: RRTEnvironment,
    obstacles: List[Dict] | None = None,
    scene: str = "wall",
) -> None:
    """Spawn obstacles in *env* and log each one.

    If *obstacles* is ``None``, loads the named scene from config.
    """
    if obstacles is None:
        obstacles = load_demo_obstacles(scene)

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


def greedy_shortcut(
    path: List[np.ndarray],
    env: RRTEnvironment,
    resolution: float = 0.05,
) -> List[np.ndarray]:
    """Greedy forward shortcutter — deterministic, O(n²) worst case.

    Starting from waypoint 0, try to connect directly to the *last*
    waypoint.  If the straight-line edge is collision-free, delete
    everything in between and jump.  Otherwise try the second-to-last,
    and so on.  Repeat from the new cursor until the goal is reached.

    Returns a pruned path (typically 3–8 waypoints) that preserves
    only the essential corners.
    """
    if len(path) <= 2:
        return list(path)

    pruned: List[np.ndarray] = [path[0]]
    cursor = 0

    while cursor < len(path) - 1:
        # Try to jump as far forward as possible
        best = cursor + 1  # fallback: next waypoint
        for target in range(len(path) - 1, cursor, -1):
            d = np.linalg.norm(path[target] - path[cursor])
            n_checks = max(8, int(d / resolution))
            if env.is_edge_valid(path[cursor], path[target], n_checks=n_checks):
                best = target
                break
        pruned.append(path[best])
        cursor = best

    return pruned


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


def sample_random_goal(
    env: RRTEnvironment,
    q_start: np.ndarray,
    seed: Optional[int] = None,
    min_c_dist: float = 1.0,
    max_attempts: int = 500,
) -> np.ndarray:
    """Sample a random collision-free goal that is at least *min_c_dist*
    from *q_start* in C-space."""
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        q_goal = sample_random_config(env, rng)
        if np.linalg.norm(q_goal - q_start) >= min_c_dist:
            return q_goal
    raise RuntimeError(
        f"Could not find a valid goal with d >= {min_c_dist} "
        f"in {max_attempts} attempts"
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
