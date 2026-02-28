from __future__ import annotations

import argparse
import time
import sys
from typing import Dict, List, Tuple

import numpy as np

from common_utils.robot_constants import (
    PANDA_REST_POSES,
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PANDA_JOINT_RANGES,
    PANDA_NUM_JOINTS,
)
from panda_rrt.environment import RRTEnvironment
from panda_rrt.forward_kinematics import PandaFK
from panda_rrt.planner import APFGuidedRRT, CollisionChecker
from panda_rrt.pure_rrt import PureRRT, PlannerResult


# ---------------------------------------------------------------------------
# Curated demo scene
# ---------------------------------------------------------------------------
# Ten obstacles (7 cylinders + 3 spheres) placed along the arm's swept
# volume between the rest pose and goal configuration.  They create narrow
# passages that force the planner to weave through tight gaps.
#
# Validated properties (SAFETY_MARGIN = 0.01 m):
#   Start clearance : 0.0214 m  (link 4)
#   Goal  clearance : 0.0214 m  (link 6)
#   Direct path     : 19/21 samples blocked
#
# Solvability (seed 42):
#   APF-RRT : OK  in   149 it,  1.0 s
#   Pure RRT: FAIL at 10 000 it, 132 s

DEMO_OBSTACLES: List[Dict] = [
    {   # Cylinder near the start, tight to link 4
        "shape": "cylinder",
        "name": "pillar_A",
        "pos": [0.44, 0.15, 0.52],
        "radius": 0.025,
        "height": 0.18,
        "color": [0.85, 0.25, 0.20, 0.85],
    },
    {   # Cylinder on the far side of the mid-sweep
        "shape": "cylinder",
        "name": "pillar_B",
        "pos": [0.52, 0.45, 0.52],
        "radius": 0.025,
        "height": 0.18,
        "color": [0.20, 0.55, 0.85, 0.85],
    },
    {   # Cylinder near the goal region
        "shape": "cylinder",
        "name": "pillar_C",
        "pos": [0.28, 0.55, 0.32],
        "radius": 0.025,
        "height": 0.18,
        "color": [0.90, 0.70, 0.10, 0.85],
    },
    {   # Cylinder blocking a wide shortcut near the goal
        "shape": "cylinder",
        "name": "pillar_D",
        "pos": [0.52, 0.55, 0.32],
        "radius": 0.025,
        "height": 0.18,
        "color": [0.20, 0.80, 0.35, 0.85],
    },
    {   # Cylinder on the near side, blocking the low path
        "shape": "cylinder",
        "name": "pillar_E",
        "pos": [0.44, -0.05, 0.32],
        "radius": 0.025,
        "height": 0.18,
        "color": [0.65, 0.20, 0.80, 0.85],
    },
    {   # Sphere close to the base sweep
        "shape": "sphere",
        "name": "ball_F",
        "pos": [0.36, 0.05, 0.42],
        "radius": 0.03,
        "color": [1.00, 0.45, 0.15, 0.85],
    },
    {   # Sphere low and forward, blocks the start-side shortcut
        "shape": "sphere",
        "name": "ball_G",
        "pos": [0.60, 0.15, 0.22],
        "radius": 0.03,
        "color": [0.15, 0.80, 0.80, 0.85],
    },
    {   # Cylinder through the mid-corridor
        "shape": "cylinder",
        "name": "pillar_H",
        "pos": [0.44, 0.15, 0.32],
        "radius": 0.025,
        "height": 0.18,
        "color": [0.80, 0.55, 0.20, 0.85],
    },
    {   # Sphere floating near the goal-side passage
        "shape": "sphere",
        "name": "ball_I",
        "pos": [0.44, 0.45, 0.42],
        "radius": 0.03,
        "color": [0.35, 0.35, 0.90, 0.85],
    },
    {   # Cylinder in the deep passage between mid and goal
        "shape": "cylinder",
        "name": "pillar_J",
        "pos": [0.44, 0.35, 0.52],
        "radius": 0.025,
        "height": 0.18,
        "color": [0.90, 0.30, 0.60, 0.85],
    },
]

# Goal: arm swings ~69° left, elbow adjusts, wrist re-orients.
# EE moves from (0.638, 0, 0.197) to roughly (0.41, 0.58, 0.41).
# C-space distance from rest ≈ 1.58 rad.
DEMO_Q_GOAL = np.array([1.2, 0.2, -0.3, -1.8, 0.5, 2.5, 0.0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pos(p: np.ndarray) -> str:
    return f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"


def _fmt_q(q: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:+.2f}" for v in q) + "]"


def generate_random_goal(env: RRTEnvironment, rng: np.random.Generator,
                         max_attempts: int = 500) -> np.ndarray:
    for _ in range(max_attempts):
        q = PANDA_LOWER_LIMITS + rng.random(PANDA_NUM_JOINTS) * PANDA_JOINT_RANGES
        if env.is_config_valid(q):
            return q
    return PANDA_REST_POSES.copy()


def _interpolate_path(path: List[np.ndarray],
                      max_step: float = 0.05) -> List[np.ndarray]:
    """Densely interpolate a path so animation is smooth."""
    dense: List[np.ndarray] = [path[0].copy()]
    for i in range(len(path) - 1):
        diff = path[i + 1] - path[i]
        dist = np.linalg.norm(diff)
        n_seg = max(1, int(np.ceil(dist / max_step)))
        for k in range(1, n_seg + 1):
            dense.append(path[i] + diff * (k / n_seg))
    return dense


def _draw_ee_trace(p, fk: PandaFK, path: List[np.ndarray],
                   color: List[float], width: float = 2.0) -> List[int]:
    """Draw lines through the EE positions along path in the viewer."""
    line_ids = []
    prev = fk.ee_position(path[0])
    for q in path[1:]:
        cur = fk.ee_position(q)
        lid = p.addUserDebugLine(prev.tolist(), cur.tolist(),
                                 lineColorRGB=color[:3],
                                 lineWidth=width,
                                 lifeTime=0)
        line_ids.append(lid)
        prev = cur
    return line_ids


def _place_marker(sim, name: str, position: np.ndarray,
                  color: np.ndarray, radius: float = 0.025) -> None:
    """Place a small translucent sphere as a visual marker."""
    sim.create_sphere(
        body_name=name,
        radius=radius,
        mass=0.0,
        position=position,
        rgba_color=color,
    )


def _spawn_demo_obstacles(env: RRTEnvironment) -> None:
    """Spawn the curated obstacle set and log each one."""
    print("--- Spawning obstacles ---")
    for obs_def in DEMO_OBSTACLES:
        pos = np.array(obs_def["pos"])
        r = obs_def["radius"]
        color = np.array(obs_def["color"])
        name = obs_def["name"]

        if obs_def["shape"] == "sphere":
            env.spawn_obstacle_sphere(pos, radius=r, name=name, color=color)
            print(f"  {name:12s}  sphere   pos={_fmt_pos(pos)}  "
                  f"r={r:.3f}")
        else:
            h = obs_def["height"]
            env.spawn_obstacle_cylinder(pos, radius=r, height=h,
                                        name=name, color=color)
            print(f"  {name:12s}  cylinder pos={_fmt_pos(pos)}  "
                  f"r={r:.3f}  h={h:.2f}")
    print(f"  Total: {len(env.obstacles)} obstacles\n")


# ---------------------------------------------------------------------------
# Visual simulation  (--rrt / --rrt_apf)
# ---------------------------------------------------------------------------

def run_visual(planner_type: str, delay: float = 1 / 60) -> None:
    """Plan and visualise the trajectory in a GUI window."""
    label = "APF-Guided RRT" if planner_type == "apf" else "Pure RRT"
    print(f"{'='*55}")
    print(f"  Visual Simulation — {label}")
    print(f"{'='*55}\n")

    # --- environment ---
    print("[1/5] Setting up environment …")
    env = RRTEnvironment(render_mode="human")
    fk = PandaFK(env._p, env.robot_id)

    # --- obstacles ---
    print("[2/5] Placing obstacles …")
    _spawn_demo_obstacles(env)

    # --- start / goal ---
    q_start = PANDA_REST_POSES.copy()
    q_goal = DEMO_Q_GOAL.copy()

    ee_start = fk.ee_position(q_start)
    ee_goal = fk.ee_position(q_goal)
    c_dist = np.linalg.norm(q_goal - q_start)

    print("--- Task description ---")
    print(f"  Start joints: {_fmt_q(q_start)}")
    print(f"  Goal  joints: {_fmt_q(q_goal)}")
    print(f"  Start EE pos: {_fmt_pos(ee_start)}")
    print(f"  Goal  EE pos: {_fmt_pos(ee_goal)}")
    print(f"  EE displacement: {np.linalg.norm(ee_goal - ee_start):.3f} m")
    print(f"  C-space dist:    {c_dist:.3f} rad")

    # clearance info
    cc = CollisionChecker(env._p, env.robot_id, env.obstacle_body_ids())
    d_s, lk_s, _ = cc.min_link_obstacle_distance(q_start)
    d_g, lk_g, _ = cc.min_link_obstacle_distance(q_goal)
    print(f"  Start clearance: {d_s:.4f} m (link {lk_s})")
    print(f"  Goal  clearance: {d_g:.4f} m (link {lk_g})")
    print()

    # visual markers
    _place_marker(env.sim, "marker_start", ee_start,
                  np.array([0.0, 0.6, 1.0, 0.6]))
    _place_marker(env.sim, "marker_goal", ee_goal,
                  np.array([0.0, 1.0, 0.3, 0.6]))
    env.sim.step()

    # --- plan ---
    if planner_type == "apf":
        note = " (APF gradient + sampling — typically 1-10 s)"
    else:
        note = " (sampling — may struggle with narrow passages)"
    print(f"[3/5] Planning with {label}{note}")

    t_plan_start = time.perf_counter()
    if planner_type == "apf":
        planner = APFGuidedRRT(env, seed=42)
    else:
        planner = PureRRT(env, seed=42)

    result = planner.plan(q_start, q_goal)
    t_plan_end = time.perf_counter()

    print()
    if not result.success:
        print(f"  FAILED after {result.iterations} iterations "
              f"({result.planning_time:.2f} s)")
        print("  The planner could not find a collision-free path.")
        input("\nPress Enter to close …")
        env.close()
        return

    print("--- Planning result ---")
    print(f"  Status:        SUCCESS")
    print(f"  Iterations:    {result.iterations}")
    print(f"  Wall time:     {result.planning_time:.2f} s")
    print(f"  Tree size:     {result.tree_size} nodes")
    print(f"  Raw path:      {len(result.path)} waypoints, "
          f"length {result.path_length:.3f} rad")
    print(f"  Smoothed path: {len(result.smoothed_path)} waypoints, "
          f"length {result.smoothed_length:.3f} rad")
    ratio = result.smoothed_length / c_dist if c_dist > 0 else 0
    print(f"  Path / straight-line ratio: {ratio:.2f}x")
    print()

    # --- EE trace ---
    print("[4/5] Drawing EE path trace …")
    _draw_ee_trace(env._p, fk, result.smoothed_path,
                   color=[1.0, 0.85, 0.0], width=2.5)

    # --- animate ---
    dense = _interpolate_path(result.smoothed_path, max_step=0.04)
    duration_est = len(dense) * delay
    print(f"[5/5] Animating {len(dense)} frames "
          f"(~{duration_est:.1f} s at {1/delay:.0f} fps) …\n")

    # move to start
    env.visualize_config(q_start)
    time.sleep(0.5)

    for i, q in enumerate(dense):
        env.visualize_config(q)
        time.sleep(delay)

    # hold at goal for a moment
    time.sleep(1.0)
    print("Animation complete.")
    print("Close the PyBullet window or press Enter to exit.")
    try:
        input()
    except EOFError:
        pass
    env.close()


# ---------------------------------------------------------------------------
# Headless demo / comparison  (existing functionality)
# ---------------------------------------------------------------------------

def demo() -> None:
    print("=== Single Plan Demo (headless) ===\n")
    env = RRTEnvironment(render_mode="rgb_array")
    fk = PandaFK(env._p, env.robot_id)

    print("Spawning obstacles …")
    _spawn_demo_obstacles(env)

    q_start = PANDA_REST_POSES.copy()
    q_goal = DEMO_Q_GOAL.copy()

    ee_start = fk.ee_position(q_start)
    ee_goal = fk.ee_position(q_goal)
    print(f"Start EE: {_fmt_pos(ee_start)}")
    print(f"Goal  EE: {_fmt_pos(ee_goal)}")
    print(f"C-space dist: {np.linalg.norm(q_goal - q_start):.3f}\n")

    planner = APFGuidedRRT(env, seed=42)
    result = planner.plan(q_start, q_goal)

    if result.success:
        print(f"SUCCESS in {result.iterations} iterations, "
              f"{result.planning_time:.3f}s")
        print(f"  Tree size:     {result.tree_size} nodes")
        print(f"  Raw path:      {len(result.path)} waypoints, "
              f"length={result.path_length:.3f}")
        print(f"  Smoothed path: {len(result.smoothed_path)} waypoints, "
              f"length={result.smoothed_length:.3f}")
    else:
        print(f"FAILED after {result.iterations} iterations, "
              f"{result.planning_time:.3f}s")

    env.close()


def compare(n_trials: int = 20) -> None:
    print(f"=== Comparative Analysis: APF-RRT vs Pure RRT ===")
    print(f"  Trials: {n_trials}, using curated obstacle scene\n")

    apf_results: List[PlannerResult] = []
    pure_results: List[PlannerResult] = []

    for trial in range(n_trials):
        seed = 1000 + trial
        env = RRTEnvironment(render_mode="rgb_array")

        _spawn_demo_obstacles(env)

        q_start = PANDA_REST_POSES.copy()
        q_goal = DEMO_Q_GOAL.copy()

        apf = APFGuidedRRT(env, seed=seed)
        apf_res = apf.plan(q_start, q_goal)
        apf_results.append(apf_res)

        pure = PureRRT(env, seed=seed)
        pure_res = pure.plan(q_start, q_goal)
        pure_results.append(pure_res)

        sa = "OK" if apf_res.success else "FAIL"
        sp = "OK" if pure_res.success else "FAIL"
        print(
            f"  Trial {trial+1:2d}/{n_trials}: "
            f"APF [{sa}] {apf_res.iterations:5d}it {apf_res.planning_time:.2f}s | "
            f"Pure [{sp}] {pure_res.iterations:5d}it {pure_res.planning_time:.2f}s"
        )
        env.close()

    for label, results in [("APF-Guided RRT", apf_results),
                           ("Pure RRT", pure_results)]:
        successes = [r for r in results if r.success]
        n = len(results)
        ns = len(successes)
        print(f"\n{'='*50}")
        print(f"  {label}: {ns}/{n} ({100*ns/n:.1f}%)")
        if successes:
            times = [r.planning_time for r in successes]
            iters = [r.iterations for r in successes]
            trees = [r.tree_size for r in successes]
            print(f"  Time:  {np.mean(times):.3f}s "
                  f"(std {np.std(times):.3f})")
            print(f"  Iters: {np.mean(iters):.0f} "
                  f"(std {np.std(iters):.0f})")
            print(f"  Tree:  {np.mean(trees):.0f} "
                  f"(std {np.std(trees):.0f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panda RRT motion-planning demo",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rrt", action="store_true",
                       help="Visualise Pure RRT in the GUI")
    group.add_argument("--rrt_apf", action="store_true",
                       help="Visualise APF-Guided RRT in the GUI")
    group.add_argument("--demo", action="store_true",
                       help="Headless single-plan demo (APF-RRT)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials for comparison mode")
    parser.add_argument("--delay", type=float, default=1 / 60,
                        help="Delay per animation frame (s)")
    args = parser.parse_args()

    if args.rrt:
        run_visual("pure", delay=args.delay)
    elif args.rrt_apf:
        run_visual("apf", delay=args.delay)
    elif args.demo:
        demo()
    else:
        compare(n_trials=args.trials)


if __name__ == "__main__":
    main()
