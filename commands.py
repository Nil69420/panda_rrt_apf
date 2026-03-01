"""High-level demo commands for the Panda RRT planner.

Functions
---------
run_visual  – plan and animate in a PyBullet GUI window
demo        – headless single-plan benchmark
compare     – multi-trial APF-RRT vs Pure-RRT comparison
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from panda_rrt.common_utils.robot_constants import PANDA_REST_POSES
from panda_rrt.environment import RRTEnvironment
from panda_rrt.forward_kinematics import PandaFK
from panda_rrt.collision import CollisionChecker
from panda_rrt.apf_rrt import APFGuidedRRT
from panda_rrt.pure_rrt import PureRRT, PlannerResult
from panda_rrt.optimizer import PathOptimizer
from panda_rrt.spline_smoother import SplineSmoother
from panda_rrt.scene import load_demo_goal, spawn_obstacles, sample_random_start_goal
from panda_rrt.visualisation import (
    draw_ee_trace,
    interpolate_path,
    place_marker,
)
from panda_rrt.config import get as cfg


# ── formatting helpers ──────────────────────────────

def fmt_pos(p: np.ndarray) -> str:
    """Format a 3-D position vector for display."""
    return f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"


def fmt_q(q: np.ndarray) -> str:
    """Format a joint-configuration vector for display."""
    return "[" + ", ".join(f"{v:+.2f}" for v in q) + "]"


# ── printing helpers ────────────────────────────────

def print_task_description(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    ee_start: np.ndarray,
    ee_goal: np.ndarray,
    c_dist: float,
    env: RRTEnvironment,
) -> None:
    """Print start/goal info and obstacle clearances."""
    print("--- Task description ---")
    print(f"  Start joints: {fmt_q(q_start)}")
    print(f"  Goal  joints: {fmt_q(q_goal)}")
    print(f"  Start EE pos: {fmt_pos(ee_start)}")
    print(f"  Goal  EE pos: {fmt_pos(ee_goal)}")
    print(f"  EE displacement: {np.linalg.norm(ee_goal - ee_start):.3f} m")
    print(f"  C-space dist:    {c_dist:.3f} rad")

    cc = CollisionChecker(env._p, env.robot_id, env.obstacle_body_ids())
    d_s, lk_s, _ = cc.min_link_obstacle_distance(q_start)
    d_g, lk_g, _ = cc.min_link_obstacle_distance(q_goal)
    print(f"  Start clearance: {d_s:.4f} m (link {lk_s})")
    print(f"  Goal  clearance: {d_g:.4f} m (link {lk_g})")
    print()


def print_result(result: PlannerResult, c_dist: float) -> None:
    """Print a successful planning result block."""
    print("\n--- Planning result ---")
    print(f"  Status:        SUCCESS")
    print(f"  Iterations:    {result.iterations}")
    print(f"  Wall time:     {result.planning_time:.2f} s")
    print(f"  Tree size:     {result.tree_size} nodes")
    print(f"  Raw path:      {len(result.path)} waypoints, "
          f"length {result.path_length:.3f} rad")
    print(f"  Smoothed path: {len(result.smoothed_path)} waypoints, "
          f"length {result.smoothed_length:.3f} rad")
    ratio = result.smoothed_length / c_dist if c_dist > 0 else 0
    print(f"  Path / straight-line ratio: {ratio:.2f}x\n")


def print_summary(label: str, results: List[PlannerResult]) -> None:
    """Print aggregate statistics for a list of planner results."""
    successes = [r for r in results if r.success]
    n, ns = len(results), len(successes)
    print(f"\n{'=' * 50}")
    print(f"  {label}: {ns}/{n} ({100 * ns / n:.1f}%)")
    if successes:
        times = [r.planning_time for r in successes]
        iters = [r.iterations for r in successes]
        trees = [r.tree_size for r in successes]
        print(f"  Time:  {np.mean(times):.3f}s (std {np.std(times):.3f})")
        print(f"  Iters: {np.mean(iters):.0f} (std {np.std(iters):.0f})")
        print(f"  Tree:  {np.mean(trees):.0f} (std {np.std(trees):.0f})")


def wait_and_close(env: RRTEnvironment) -> None:
    """Block until the user closes the window / presses Enter, then close."""
    print("Close the PyBullet window or press Enter to exit.")
    try:
        input()
    except EOFError:
        pass
    env.close()


# ── commands ────────────────────────────────────────

def run_visual(
    planner_type: str,
    delay: float | None = None,
    random_mode: bool = False,
) -> None:
    """Plan and visualise the trajectory in a GUI window."""
    if delay is None:
        delay = cfg("visualisation", "frame_delay")

    _labels = {
        "apf": "APF-Guided RRT",
        "apf_opt": "APF-RRT + Optimisation",
        "apf_spline": "APF-RRT + B-Spline",
        "pure": "Pure RRT",
    }
    label = _labels.get(planner_type, "Pure RRT")
    print(f"{'=' * 55}")
    print(f"  Visual Simulation — {label}")
    print(f"{'=' * 55}\n")

    # --- environment ---
    print("[1/5] Setting up environment …")
    env = RRTEnvironment(render_mode="human")
    fk = PandaFK(env._p, env.robot_id)

    # --- obstacles ---
    print("[2/5] Placing obstacles …")
    spawn_obstacles(env)

    # --- start / goal ---
    if random_mode:
        from panda_rrt.scene import sample_random_start_goal
        q_start, q_goal = sample_random_start_goal(env, seed=None)
        print("  Using RANDOM start / goal")
    else:
        q_start = PANDA_REST_POSES.copy()
        q_goal = load_demo_goal()

    ee_start = fk.ee_position(q_start)
    ee_goal = fk.ee_position(q_goal)
    c_dist = np.linalg.norm(q_goal - q_start)

    print_task_description(q_start, q_goal, ee_start, ee_goal, c_dist, env)

    # visual markers
    place_marker(env.sim, "marker_start", ee_start,
                 np.array(cfg("visualisation", "start_marker_color")))
    place_marker(env.sim, "marker_goal", ee_goal,
                 np.array(cfg("visualisation", "goal_marker_color")))
    env.sim.step()

    # --- plan ---
    _notes = {
        "apf": " (APF gradient + sampling — typically 1-10 s)",
        "apf_opt": " (APF-RRT then gradient-descent optimisation)",
        "apf_spline": " (APF-RRT then cubic B-spline — very fast)",
        "pure": " (sampling — may struggle with narrow passages)",
    }
    has_post = planner_type in ("apf_opt", "apf_spline")
    steps = "6" if has_post else "5"
    print(f"[3/{steps}] Planning with {label}{_notes.get(planner_type, '')}")

    planner = (APFGuidedRRT(env, seed=42)
               if planner_type in ("apf", "apf_opt", "apf_spline")
               else PureRRT(env, seed=42))
    result = planner.plan(q_start, q_goal)

    if not result.success:
        print(f"\n  FAILED after {result.iterations} iterations "
              f"({result.planning_time:.2f} s)")
        print("  The planner could not find a collision-free path.")
        wait_and_close(env)
        return

    print_result(result, c_dist)

    # --- post-processing (opt or spline) ---
    final_path = result.smoothed_path
    if planner_type == "apf_opt":
        cc = CollisionChecker(env._p, env.robot_id, env.obstacle_body_ids())
        opt = PathOptimizer(cc)
        resampled = PathOptimizer.resample_path(result.smoothed_path)
        print(f"[4/{steps}] Optimising path ({len(resampled)} waypoints, "
              f"{opt.n_iters} iters, lr={opt.lr}, \u03bb={opt.lam}) \u2026")
        final_path = opt.optimize(resampled, verbose=True)
        opt_len = PathOptimizer.path_length(final_path)
        print(f"  Optimised path: {len(final_path)} waypoints, "
              f"length {opt_len:.3f} rad\n")
    elif planner_type == "apf_spline":
        import time as _t
        t0 = _t.perf_counter()
        final_path = SplineSmoother.smooth(result.smoothed_path, n_points=60)
        t_spline = _t.perf_counter() - t0
        spline_len = PathOptimizer.path_length(final_path)
        j_accel = SplineSmoother.integrated_acceleration_cost(
            result.smoothed_path,
        )
        print(f"[4/{steps}] B-Spline smoothing ({len(final_path)} pts, "
              f"{t_spline * 1000:.1f} ms)")
        print(f"  Spline path length: {spline_len:.3f} rad")
        print(f"  \u222b\u2016q\u0308(t)\u2016\u00b2 dt = {j_accel:.2f}\n")

    # --- EE trace ---
    ee_step = int(steps) - 1
    print(f"[{ee_step}/{steps}] Drawing EE path trace …")
    draw_ee_trace(env._p, fk, final_path)

    # --- animate ---
    dense = interpolate_path(final_path)
    duration_est = len(dense) * delay
    anim_step = int(steps)
    print(f"[{anim_step}/{steps}] Animating {len(dense)} frames "
          f"(~{duration_est:.1f} s at {1 / delay:.0f} fps) …\n")

    env.visualize_config(q_start)
    time.sleep(0.5)
    for q in dense:
        env.visualize_config(q)
        time.sleep(delay)
    time.sleep(1.0)

    print("Animation complete.")
    wait_and_close(env)


def demo() -> None:
    """Run a single headless APF-RRT plan and print the result."""
    print("=== Single Plan Demo (headless) ===\n")
    env = RRTEnvironment(render_mode="rgb_array")
    fk = PandaFK(env._p, env.robot_id)

    print("Spawning obstacles …")
    spawn_obstacles(env)

    q_start = PANDA_REST_POSES.copy()
    q_goal = load_demo_goal()

    ee_start = fk.ee_position(q_start)
    ee_goal = fk.ee_position(q_goal)
    print(f"Start EE: {fmt_pos(ee_start)}")
    print(f"Goal  EE: {fmt_pos(ee_goal)}")
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
    """Run *n_trials* plans with each planner and print aggregate stats."""
    print(f"=== Comparative Analysis: APF-RRT vs Pure RRT ===")
    print(f"  Trials: {n_trials}, using curated obstacle scene\n")

    apf_results: List[PlannerResult] = []
    pure_results: List[PlannerResult] = []

    q_start = PANDA_REST_POSES.copy()
    q_goal = load_demo_goal()

    for trial in range(n_trials):
        seed = 1000 + trial
        env = RRTEnvironment(render_mode="rgb_array")
        spawn_obstacles(env)

        apf_res = APFGuidedRRT(env, seed=seed).plan(q_start, q_goal)
        apf_results.append(apf_res)

        pure_res = PureRRT(env, seed=seed).plan(q_start, q_goal)
        pure_results.append(pure_res)

        sa = "OK" if apf_res.success else "FAIL"
        sp = "OK" if pure_res.success else "FAIL"
        print(
            f"  Trial {trial + 1:2d}/{n_trials}: "
            f"APF [{sa}] {apf_res.iterations:5d}it {apf_res.planning_time:.2f}s | "
            f"Pure [{sp}] {pure_res.iterations:5d}it {pure_res.planning_time:.2f}s"
        )
        env.close()

    print_summary("APF-Guided RRT", apf_results)
    print_summary("Pure RRT", pure_results)
