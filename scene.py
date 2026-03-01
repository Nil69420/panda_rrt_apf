"""Scene management: obstacle spawning and visual markers."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

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
