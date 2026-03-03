"""3-D visualisation of RRT trees and final paths using Matplotlib.

Renders scatter/line plots showing tree edges, the smoothed path,
obstacle positions, and start/goal markers.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from panda_rrt.common_utils.robot_constants import PANDA_REST_POSES
from panda_rrt.environment import RRTEnvironment
from panda_rrt.computations.forward_kinematics import PandaFK
from panda_rrt.computations.collision import CollisionChecker
from panda_rrt.planners.core import Node
from panda_rrt.planners.pure_rrt import PureRRT
from panda_rrt.planners.rrt_star import RRTStar
from panda_rrt.planners.apf_rrt import APFGuidedRRT
from panda_rrt.planners.optimizer import PathOptimizer
from panda_rrt.scene import load_demo_goal, spawn_obstacles


# ── core plotting ───────────────────────────────────

def plot_tree_3d(
    fk: PandaFK,
    nodes: List[Node],
    smoothed_path: List[np.ndarray],
    obstacles: list,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    title: str = "RRT Tree",
    optimized_path: Optional[List[np.ndarray]] = None,
    ax=None,
    save_path: Optional[str] = None,
) -> None:
    """Render a tree and path in a 3-D Matplotlib axes.

    Parameters
    ----------
    fk : PandaFK
    nodes : list of Node
    smoothed_path : list of ndarray
    obstacles : list of Obstacle
    q_start, q_goal : ndarray
    title : str
    optimized_path : list of ndarray or None
    ax : Axes3D or None
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Tree edges
    ee_pos = [fk.ee_position(n.q) for n in nodes]
    for i, node in enumerate(nodes):
        if node.parent is not None:
            c, p = ee_pos[i], ee_pos[node.parent]
            ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]],
                    color="lightgrey", linewidth=0.3, alpha=0.5)

    # Tree nodes
    xs, ys, zs = zip(*ee_pos) if ee_pos else ([], [], [])
    ax.scatter(xs, ys, zs, c="lightgrey", s=1, alpha=0.3, label="Tree nodes")

    # Obstacles
    for obs in obstacles:
        p = obs.position
        ax.scatter([p[0]], [p[1]], [p[2]], c="red", s=80, alpha=0.6,
                   marker="^", edgecolors="darkred", linewidths=0.5)

    # Smoothed path
    if smoothed_path:
        pe = [fk.ee_position(q) for q in smoothed_path]
        px, py, pz = zip(*pe)
        ax.plot(px, py, pz, color="dodgerblue", linewidth=2.5,
                label="Smoothed path", zorder=5)

    # Optimised path overlay
    if optimized_path:
        oe = [fk.ee_position(q) for q in optimized_path]
        ox, oy, oz = zip(*oe)
        ax.plot(ox, oy, oz, color="limegreen", linewidth=2.5,
                linestyle="--", label="Optimised path", zorder=6)

    # Start / goal markers
    ee_s = fk.ee_position(q_start)
    ee_g = fk.ee_position(q_goal)
    ax.scatter(*ee_s, c="cyan", s=120, marker="o",
               edgecolors="black", linewidths=1, zorder=10, label="Start")
    ax.scatter(*ee_g, c="lime", s=120, marker="*",
               edgecolors="black", linewidths=1, zorder=10, label="Goal")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved -> {save_path}")
        else:
            plt.show()


# ── planner factory (dict dispatch) ─────────────────

def _make_planner(key: str, env, seed: int):
    """Instantiate a planner by key string."""
    _FACTORY = {
        "pure_rrt":    lambda: PureRRT(env, seed=seed),
        "rrt_star":    lambda: RRTStar(env, seed=seed),
        "apf_rrt":     lambda: APFGuidedRRT(env, seed=seed),
        "apf_rrt_opt": lambda: APFGuidedRRT(env, seed=seed),
    }
    factory = _FACTORY.get(key)
    if factory is None:
        raise ValueError(f"Unknown planner key: {key!r}")
    return factory()


_LABELS = {
    "pure_rrt":    "Pure RRT",
    "rrt_star":    "RRT*",
    "apf_rrt":     "APF-RRT",
    "apf_rrt_opt": "APF-RRT + Optimisation",
}


# ── multi-planner comparison ───────────────────────

def plot_comparison(
    save_path: Optional[str] = None,
    scene: str = "wall",
) -> None:
    """Generate a 2x2 figure comparing four planners."""
    import matplotlib.pyplot as plt

    q_start = PANDA_REST_POSES.copy()
    q_goal = load_demo_goal(scene)
    seed = 42

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("RRT Tree & Path Comparison", fontsize=14, fontweight="bold")

    configs = ["pure_rrt", "rrt_star", "apf_rrt", "apf_rrt_opt"]

    for idx, key in enumerate(configs):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        env = RRTEnvironment(render_mode="rgb_array")
        spawn_obstacles(env, scene=scene)
        fk = PandaFK(env._p, env.robot_id)

        planner = _make_planner(key, env, seed)
        result = planner.plan(q_start, q_goal)
        optimized_path = None

        if result.success and key == "apf_rrt_opt":
            cc = CollisionChecker(env._p, env.robot_id, env.obstacle_body_ids())
            optimized_path = PathOptimizer(cc).optimize(result.path)

        label = _LABELS[key]
        status = "OK" if result.success else "FAIL"
        subtitle = (
            f"{label}\n{status} | {result.iterations} it | "
            f"{result.planning_time:.2f}s | {result.tree_size} nodes"
        )
        if result.success:
            subtitle += f" | path {result.smoothed_length:.2f} rad"

        plot_tree_3d(
            fk=fk, nodes=planner.nodes,
            smoothed_path=result.smoothed_path if result.success else [],
            obstacles=env.obstacles, q_start=q_start, q_goal=q_goal,
            title=subtitle, optimized_path=optimized_path, ax=ax,
        )
        env.close()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()


# ── single planner ──────────────────────────────────

def plot_single(
    planner_key: str = "apf_rrt",
    save_path: Optional[str] = None,
    scene: str = "wall",
) -> None:
    """Plot one planner's tree + path."""
    import matplotlib.pyplot as plt

    q_start = PANDA_REST_POSES.copy()
    q_goal = load_demo_goal(scene)
    seed = 42

    env = RRTEnvironment(render_mode="rgb_array")
    spawn_obstacles(env, scene=scene)
    fk = PandaFK(env._p, env.robot_id)

    planner = _make_planner(planner_key, env, seed)
    result = planner.plan(q_start, q_goal)
    optimized_path = None

    if result.success and planner_key == "apf_rrt_opt":
        cc = CollisionChecker(env._p, env.robot_id, env.obstacle_body_ids())
        optimized_path = PathOptimizer(cc).optimize(result.path, verbose=True)

    label = _LABELS.get(planner_key, planner_key)
    if result.success:
        title = (f"{label} -- {result.iterations} it, "
                 f"{result.planning_time:.2f}s, {result.tree_size} nodes")
    else:
        title = f"{label} -- FAILED ({result.iterations} it)"

    plot_tree_3d(
        fk=fk, nodes=planner.nodes,
        smoothed_path=result.smoothed_path if result.success else [],
        obstacles=env.obstacles, q_start=q_start, q_goal=q_goal,
        title=title, optimized_path=optimized_path, save_path=save_path,
    )
    env.close()
