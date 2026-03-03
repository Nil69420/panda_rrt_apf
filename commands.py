"""High-level demo commands and CLI helpers.

Functions
---------
run_visual  -- plan and animate in a PyBullet GUI window
demo        -- headless single-plan benchmark
do_benchmark -- launch benchmark runner
do_graph     -- run benchmark and generate graphs
do_viz       -- 3-D tree visualisation
interactive_menu -- interactive planner picker
build_parser -- construct the argparse parser
"""
from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np

from panda_rrt.common_utils.robot_constants import PANDA_REST_POSES
from panda_rrt.environment import RRTEnvironment
from panda_rrt.computations.forward_kinematics import PandaFK
from panda_rrt.computations.collision import CollisionChecker
from panda_rrt.planners.apf_rrt import APFGuidedRRT
from panda_rrt.planners.pure_rrt import PureRRT
from panda_rrt.planners.optimizer import PathOptimizer
from panda_rrt.computations.spline_smoother import SplineSmoother
from panda_rrt.planners.core import PlannerResult
from panda_rrt.scene import (
    load_demo_goal,
    spawn_obstacles,
    sample_random_goal,
    greedy_shortcut,
)
from panda_rrt.benchmark_utilities.visualisation import (
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

def print_task(q_start, q_goal, ee_start, ee_goal, c_dist, env):
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
    print(f"  Goal  clearance: {d_g:.4f} m (link {lk_g})\n")


def print_result(result: PlannerResult, c_dist: float):
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


def print_summary(label: str, results: List[PlannerResult]):
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


def wait_and_close(env: RRTEnvironment):
    """Block until the user closes the window / presses Enter, then close."""
    print("Close the PyBullet window or press Enter to exit.")
    try:
        input()
    except EOFError:
        pass
    env.close()


# ── planner factory (dict dispatch) ─────────────────

PLANNER_LABELS = {
    "pure":       "Pure RRT",
    "apf":        "APF-Guided RRT",
    "apf_opt":    "APF-RRT + Optimisation",
    "apf_spline": "APF-RRT + B-Spline",
}

PLANNER_NOTES = {
    "pure":       " (sampling -- may struggle with narrow passages)",
    "apf":        " (APF gradient + sampling -- typically 1-10 s)",
    "apf_opt":    " (APF-RRT then gradient-descent optimisation)",
    "apf_spline": " (APF-RRT then cubic B-spline -- very fast)",
}

FLAG_DISPATCH = {
    "rrt":            "pure",
    "rrt_apf":        "apf",
    "rrt_apf_opt":    "apf_opt",
    "rrt_apf_spline": "apf_spline",
}

INTERACTIVE_MAP = {
    "1": "pure",
    "2": "apf",
    "3": "apf_opt",
    "4": "apf_spline",
    "":  "apf",
}


def make_planner(key: str, env, seed: int = 42):
    """Instantiate a planner from a string key (dict dispatch)."""
    _FACTORY = {
        "pure":       lambda: PureRRT(env, seed=seed),
        "apf":        lambda: APFGuidedRRT(env, seed=seed),
        "apf_opt":    lambda: APFGuidedRRT(env, seed=seed),
        "apf_spline": lambda: APFGuidedRRT(env, seed=seed),
    }
    factory = _FACTORY.get(key)
    if factory is None:
        raise ValueError(f"Unknown planner type: {key!r}")
    return factory()


# ── commands ────────────────────────────────────────

def run_visual(
    planner_type: str,
    delay: float = None,
    random_mode: bool = False,
    scene: str = "wall",
) -> None:
    """Plan and visualise the trajectory in a PyBullet GUI window."""
    if delay is None:
        delay = cfg("visualisation", "frame_delay")

    label = PLANNER_LABELS.get(planner_type, "Pure RRT")
    print(f"{'=' * 55}\n  Visual Simulation -- {label}\n{'=' * 55}\n")

    # 1. Environment
    print("[1/5] Setting up environment ...")
    env = RRTEnvironment(render_mode="human")
    fk = PandaFK(env._p, env.robot_id)

    # 2. Obstacles
    print("[2/5] Placing obstacles ...")
    spawn_obstacles(env, scene=scene)

    # 3. Start / goal
    q_start = PANDA_REST_POSES.copy()
    if random_mode:
        q_goal = sample_random_goal(env, q_start, seed=None)
        print("  Using RANDOM goal")
    else:
        q_goal = load_demo_goal(scene)

    ee_start = fk.ee_position(q_start)
    ee_goal = fk.ee_position(q_goal)
    c_dist = np.linalg.norm(q_goal - q_start)
    print_task(q_start, q_goal, ee_start, ee_goal, c_dist, env)

    place_marker(env.sim, "marker_start", ee_start,
                 np.array(cfg("visualisation", "start_marker_color")))
    place_marker(env.sim, "marker_goal", ee_goal,
                 np.array(cfg("visualisation", "goal_marker_color")))

    # 4. Plan
    has_post = planner_type in ("apf_opt", "apf_spline")
    steps = "6" if has_post else "5"
    print(f"[3/{steps}] Planning with {label}"
          f"{PLANNER_NOTES.get(planner_type, '')}")

    planner = make_planner(planner_type, env, seed=42)
    result = planner.plan(q_start, q_goal)

    if not result.success:
        print(f"\n  FAILED after {result.iterations} iterations "
              f"({result.planning_time:.2f} s)")
        wait_and_close(env)
        return

    print_result(result, c_dist)

    # 5. Post-processing (dict dispatch)
    final_path = result.smoothed_path

    def _post_opt():
        nonlocal final_path
        cc = CollisionChecker(env._p, env.robot_id, env.obstacle_body_ids())
        opt = PathOptimizer(cc)
        resampled = PathOptimizer.resample_path(result.smoothed_path)
        print(f"[4/{steps}] Optimising path ({len(resampled)} wps, "
              f"{opt.n_iters} iters, lr={opt.lr}, "
              f"\u03bb={opt.lam}) \u2026")
        final_path = opt.optimize(resampled, verbose=True)
        opt_len = PathOptimizer.path_length(final_path)
        print(f"  Optimised path: {len(final_path)} waypoints, "
              f"length {opt_len:.3f} rad\n")

    def _post_spline():
        nonlocal final_path
        t0 = time.perf_counter()
        pruned = greedy_shortcut(result.path, env)
        final_path = SplineSmoother.smooth(pruned, n_points=60)
        t_spline = time.perf_counter() - t0
        spline_len = PathOptimizer.path_length(final_path)
        j_accel = SplineSmoother.integrated_acceleration_cost(pruned)
        print(f"[4/{steps}] Shortcut+Spline ({len(result.path)} -> "
              f"{len(pruned)} wps, {len(final_path)} spline pts, "
              f"{t_spline * 1000:.1f} ms)")
        print(f"  Spline path length: {spline_len:.3f} rad")
        print(f"  \u222b\u2016q\u0308(t)\u2016\u00b2 dt = {j_accel:.2f}\n")

    _POST_DISPATCH = {
        "apf_opt":    _post_opt,
        "apf_spline": _post_spline,
    }
    post = _POST_DISPATCH.get(planner_type)
    if post:
        post()

    # 6. EE trace + animate
    ee_step = int(steps) - 1
    print(f"[{ee_step}/{steps}] Drawing EE path trace ...")
    draw_ee_trace(env._p, fk, final_path)

    dense = interpolate_path(final_path)
    duration_est = len(dense) * delay
    print(f"[{steps}/{steps}] Animating {len(dense)} frames "
          f"(~{duration_est:.1f} s at {1 / delay:.0f} fps) ...\n")

    env.visualize_config(q_start)
    print("\n  Press Enter to start the animation ...")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

    for q in dense:
        env.visualize_config(q)
        time.sleep(delay)
    time.sleep(1.0)

    print("Animation complete.")
    wait_and_close(env)


def demo() -> None:
    """Single headless APF-RRT plan."""
    print("=== Single Plan Demo (headless) ===\n")
    env = RRTEnvironment(render_mode="rgb_array")
    fk = PandaFK(env._p, env.robot_id)
    spawn_obstacles(env)

    q_start = PANDA_REST_POSES.copy()
    q_goal = load_demo_goal()

    ee_start = fk.ee_position(q_start)
    ee_goal = fk.ee_position(q_goal)
    print(f"Start EE: {fmt_pos(ee_start)}")
    print(f"Goal  EE: {fmt_pos(ee_goal)}")
    print(f"C-space dist: {np.linalg.norm(q_goal - q_start):.3f}\n")

    result = APFGuidedRRT(env, seed=42).plan(q_start, q_goal)

    if result.success:
        print(f"SUCCESS in {result.iterations} iterations, "
              f"{result.planning_time:.3f}s")
        print(f"  Tree size:     {result.tree_size} nodes")
        print(f"  Raw path:      {len(result.path)} wps, "
              f"length={result.path_length:.3f}")
        print(f"  Smoothed path: {len(result.smoothed_path)} wps, "
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


# ── benchmark / graph / viz dispatch ────────────────

def do_benchmark(args):
    """Launch the benchmark runner."""
    from panda_rrt.benchmark_utilities.runner import run_benchmark
    run_benchmark(
        n_trials=args.trials, preset=args.preset,
        random_mode=args.random, scene=args.scene,
    )


def do_graph(args):
    """Run benchmark then generate graphs."""
    from panda_rrt.benchmark_utilities.runner import run_benchmark, PRESETS
    from panda_rrt.benchmark_utilities.graphs import generate_benchmark_graphs
    preset = args.preset or "apf"
    all_results = run_benchmark(
        n_trials=args.trials, preset=preset,
        random_mode=args.random, scene=args.scene,
    )
    planners = PRESETS.get(preset, PRESETS["apf"])
    generate_benchmark_graphs(all_results, planners, save_path=args.save)


def do_viz(args):
    """Launch 3-D tree visualisation."""
    from panda_rrt.benchmark_utilities.tree_viz import plot_comparison, plot_single
    if args.planner == "all":
        plot_comparison(save_path=args.save, scene=args.scene)
    else:
        plot_single(planner_key=args.planner, save_path=args.save,
                    scene=args.scene)


def interactive_menu(args):
    """Display the planner picker and dispatch."""
    print("\n  Select a planner to visualise:")
    print("    [1] Pure RRT")
    print("    [2] APF-Guided RRT")
    print("    [3] APF-RRT + Gradient-Descent Optimisation")
    print("    [4] APF-RRT + Cubic B-Spline Smoothing")
    print("    [5] Headless demo (APF-RRT)")
    print("    [6] Benchmark suite")
    print("    [7] Generate benchmark graphs")
    print("    [8] 3-D tree visualisation")
    print()
    try:
        choice = input("  Choice [2]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    _MENU_DISPATCH = {
        "5": lambda: demo(),
        "6": lambda: do_benchmark(args),
        "7": lambda: do_graph(args),
        "8": lambda: do_viz(args),
    }
    handler = _MENU_DISPATCH.get(choice)
    if handler:
        handler()
    else:
        run_visual(
            INTERACTIVE_MAP.get(choice, "apf"),
            delay=args.delay, random_mode=args.random, scene=args.scene,
        )


def build_parser() -> argparse.ArgumentParser:
    """Construct the unified CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Panda RRT motion-planning -- unified CLI",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rrt", action="store_true",
                      help="Visualise Pure RRT in the GUI")
    mode.add_argument("--rrt_apf", action="store_true",
                      help="Visualise APF-Guided RRT in the GUI")
    mode.add_argument("--rrt_apf_opt", action="store_true",
                      help="Visualise APF-RRT + gradient-descent optimisation")
    mode.add_argument("--rrt_apf_spline", action="store_true",
                      help="Visualise APF-RRT + cubic B-spline smoothing")
    mode.add_argument("--demo", action="store_true",
                      help="Headless single-plan demo (APF-RRT)")
    mode.add_argument("--benchmark", action="store_true",
                      help="Launch the benchmark suite")
    mode.add_argument("--graph", action="store_true",
                      help="Run benchmark and generate graphs")
    mode.add_argument("--viz", action="store_true",
                      help="3-D tree visualisation (Matplotlib)")

    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials for benchmark mode")
    parser.add_argument("--delay", type=float, default=None,
                        help="Delay per animation frame (s)")
    parser.add_argument("--random", action="store_true",
                        help="Use random start/goal positions")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["apf", "rrt", "star", "full"],
                        help="Benchmark preset (skip interactive menu)")
    parser.add_argument("--planner", type=str, default="all",
                        choices=["pure_rrt", "rrt_star", "apf_rrt",
                                 "apf_rrt_opt", "all"],
                        help="Planner for --viz mode")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure/graph to file")

    scene_grp = parser.add_mutually_exclusive_group()
    scene_grp.add_argument("--wall", action="store_const", dest="scene",
                           const="wall", help="Wall obstacle scene (default)")
    scene_grp.add_argument("--canopy", action="store_const", dest="scene",
                           const="canopy", help="Canopy obstacle scene")
    scene_grp.add_argument("--passage", action="store_const", dest="scene",
                           const="passage", help="Narrow passage scene")
    parser.set_defaults(scene="wall")

    return parser
