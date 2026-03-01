"""3-D visualisation of RRT trees and final paths using Matplotlib.

Generates a 3-D scatter/line plot showing:
  - Tree edges (light grey) — every node's EE position connected to its parent.
  - Final smoothed path (coloured line).
  - Obstacle positions (red/blue spheres).
  - Start and goal EE markers.

Usage::

    python3 -m panda_rrt.tree_viz
    python3 -m panda_rrt.tree_viz --planner apf_rrt --save tree.png
    python3 -m panda_rrt.tree_viz --planner all
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np

from panda_rrt.common_utils.robot_constants import PANDA_REST_POSES
from panda_rrt.environment import RRTEnvironment
from panda_rrt.forward_kinematics import PandaFK
from panda_rrt.collision import CollisionChecker
from panda_rrt.pure_rrt import PureRRT, PlannerResult, Node
from panda_rrt.rrt_star import RRTStar
from panda_rrt.apf_rrt import APFGuidedRRT
from panda_rrt.optimizer import PathOptimizer
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
    """Render the tree and path in a 3-D Matplotlib axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        standalone = True
    else:
        standalone = False

    # ── tree edges ──────────────────────────────────
    ee_positions = [fk.ee_position(n.q) for n in nodes]
    for i, node in enumerate(nodes):
        if node.parent is not None:
            p_child = ee_positions[i]
            p_parent = ee_positions[node.parent]
            ax.plot(
                [p_parent[0], p_child[0]],
                [p_parent[1], p_child[1]],
                [p_parent[2], p_child[2]],
                color="lightgrey", linewidth=0.3, alpha=0.5,
            )

    # ── tree nodes ──────────────────────────────────
    xs = [p[0] for p in ee_positions]
    ys = [p[1] for p in ee_positions]
    zs = [p[2] for p in ee_positions]
    ax.scatter(xs, ys, zs, c="lightgrey", s=1, alpha=0.3, label="Tree nodes")

    # ── obstacles ───────────────────────────────────
    for obs in obstacles:
        pos = obs.position
        ax.scatter(
            [pos[0]], [pos[1]], [pos[2]],
            c="red", s=80, alpha=0.6, marker="^",
            edgecolors="darkred", linewidths=0.5,
        )

    # ── smoothed path ───────────────────────────────
    if smoothed_path:
        path_ee = [fk.ee_position(q) for q in smoothed_path]
        px = [p[0] for p in path_ee]
        py = [p[1] for p in path_ee]
        pz = [p[2] for p in path_ee]
        ax.plot(px, py, pz, color="dodgerblue", linewidth=2.5,
                label="Smoothed path", zorder=5)

    # ── optimised path (if provided) ────────────────
    if optimized_path:
        opt_ee = [fk.ee_position(q) for q in optimized_path]
        ox = [p[0] for p in opt_ee]
        oy = [p[1] for p in opt_ee]
        oz = [p[2] for p in opt_ee]
        ax.plot(ox, oy, oz, color="limegreen", linewidth=2.5,
                linestyle="--", label="Optimised path", zorder=6)

    # ── start / goal markers ────────────────────────
    ee_start = fk.ee_position(q_start)
    ee_goal = fk.ee_position(q_goal)
    ax.scatter(*ee_start, c="cyan", s=120, marker="o",
               edgecolors="black", linewidths=1, zorder=10, label="Start")
    ax.scatter(*ee_goal, c="lime", s=120, marker="*",
               edgecolors="black", linewidths=1, zorder=10, label="Goal")

    # ── formatting ──────────────────────────────────
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {save_path}")
        else:
            plt.show()


# ── multi-planner comparison figure ─────────────────

def plot_comparison(save_path: Optional[str] = None) -> None:
    """Generate a 2x2 figure comparing all four planners."""
    import matplotlib.pyplot as plt

    q_start = PANDA_REST_POSES.copy()
    q_goal = load_demo_goal()
    seed = 42

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("RRT Tree & Path Comparison", fontsize=14, fontweight="bold")

    configs = [
        ("Pure RRT", "pure_rrt"),
        ("RRT*", "rrt_star"),
        ("APF-RRT (heuristic smoothing)", "apf_rrt"),
        ("APF-RRT + Optimisation", "apf_rrt_opt"),
    ]

    for idx, (label, key) in enumerate(configs):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")

        env = RRTEnvironment(render_mode="rgb_array")
        spawn_obstacles(env)
        fk = PandaFK(env._p, env.robot_id)

        optimized_path = None

        if key == "pure_rrt":
            planner = PureRRT(env, seed=seed)
        elif key == "rrt_star":
            planner = RRTStar(env, seed=seed)
        elif key == "apf_rrt":
            planner = APFGuidedRRT(env, seed=seed)
        else:  # apf_rrt_opt
            planner = APFGuidedRRT(env, seed=seed)

        result = planner.plan(q_start, q_goal)

        if result.success and key == "apf_rrt_opt":
            cc = CollisionChecker(env._p, env.robot_id, env.obstacle_body_ids())
            opt = PathOptimizer(cc)
            optimized_path = opt.optimize(result.path)

        status = "OK" if result.success else "FAIL"
        subtitle = (
            f"{label}\n"
            f"{status} | {result.iterations} it | "
            f"{result.planning_time:.2f}s | "
            f"{result.tree_size} nodes"
        )
        if result.success:
            subtitle += f" | path {result.smoothed_length:.2f} rad"

        plot_tree_3d(
            fk=fk,
            nodes=planner.nodes,
            smoothed_path=result.smoothed_path if result.success else [],
            obstacles=env.obstacles,
            q_start=q_start,
            q_goal=q_goal,
            title=subtitle,
            optimized_path=optimized_path,
            ax=ax,
        )
        env.close()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ── single-planner plot ─────────────────────────────

def plot_single(
    planner_key: str = "apf_rrt",
    save_path: Optional[str] = None,
) -> None:
    """Plot one planner's tree + path."""
    import matplotlib.pyplot as plt

    q_start = PANDA_REST_POSES.copy()
    q_goal = load_demo_goal()
    seed = 42

    env = RRTEnvironment(render_mode="rgb_array")
    spawn_obstacles(env)
    fk = PandaFK(env._p, env.robot_id)

    optimized_path = None
    label = planner_key

    if planner_key == "pure_rrt":
        planner = PureRRT(env, seed=seed)
        label = "Pure RRT"
    elif planner_key == "rrt_star":
        planner = RRTStar(env, seed=seed)
        label = "RRT*"
    elif planner_key == "apf_rrt":
        planner = APFGuidedRRT(env, seed=seed)
        label = "APF-RRT"
    elif planner_key == "apf_rrt_opt":
        planner = APFGuidedRRT(env, seed=seed)
        label = "APF-RRT + Optimisation"
    else:
        print(f"Unknown planner: {planner_key}")
        env.close()
        return

    result = planner.plan(q_start, q_goal)

    if result.success and planner_key == "apf_rrt_opt":
        cc = CollisionChecker(env._p, env.robot_id, env.obstacle_body_ids())
        opt = PathOptimizer(cc)
        optimized_path = opt.optimize(result.path, verbose=True)

    if result.success:
        title = (f"{label} — {result.iterations} it, "
                 f"{result.planning_time:.2f}s, "
                 f"{result.tree_size} nodes")
    else:
        title = f"{label} — FAILED ({result.iterations} it)"

    plot_tree_3d(
        fk=fk,
        nodes=planner.nodes,
        smoothed_path=result.smoothed_path if result.success else [],
        obstacles=env.obstacles,
        q_start=q_start,
        q_goal=q_goal,
        title=title,
        optimized_path=optimized_path,
        save_path=save_path,
    )
    env.close()


# ── CLI ─────────────────────────────────────────────

parser = argparse.ArgumentParser(description="3D RRT tree visualiser")
parser.add_argument(
    "--planner", type=str, default="all",
    choices=["pure_rrt", "rrt_star", "apf_rrt", "apf_rrt_opt", "all"],
    help="Which planner to visualise",
)
parser.add_argument("--save", type=str, default=None,
                    help="Save figure to file instead of showing")
args = parser.parse_args()

if args.planner == "all":
    plot_comparison(save_path=args.save)
else:
    plot_single(planner_key=args.planner, save_path=args.save)
