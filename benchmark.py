"""Benchmark: interactive planner comparison.

Runs *n_trials* randomised seeds on the demo obstacle scene and prints a
comparison table of success rate, computation time, path length, and node
count.

Usage::

    python3 -m panda_rrt.benchmark                 # interactive menu
    python3 -m panda_rrt.benchmark --preset apf    # APF-RRT vs APF-RRT+Opt vs APF-RRT+Spline
    python3 -m panda_rrt.benchmark --preset rrt    # Pure RRT vs RRT*
    python3 -m panda_rrt.benchmark --preset full   # all planners
    python3 -m panda_rrt.benchmark --trials 10 --verbose
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from panda_rrt.common_utils.robot_constants import PANDA_REST_POSES
from panda_rrt.environment import RRTEnvironment
from panda_rrt.collision import CollisionChecker
from panda_rrt.pure_rrt import PureRRT, PlannerResult
from panda_rrt.rrt_star import RRTStar
from panda_rrt.apf_rrt import APFGuidedRRT
from panda_rrt.optimizer import PathOptimizer
from panda_rrt.spline_smoother import SplineSmoother
from panda_rrt.scene import load_demo_goal, spawn_obstacles
from panda_rrt.config import get as cfg


# ── preset definitions ──────────────────────────────

PRESETS: Dict[str, List[str]] = {
    "apf":     ["APF-RRT", "APF-RRT+Opt", "APF-RRT+Spline"],
    "rrt":     ["Pure RRT", "RRT*"],
    "star":    ["RRT*", "APF-RRT+Opt"],
    "full":    ["Pure RRT", "RRT*", "APF-RRT", "APF-RRT+Opt", "APF-RRT+Spline"],
}


# ── data container ──────────────────────────────────

@dataclass
class TrialResult:
    planner_name: str
    success: bool
    iterations: int = 0
    tree_size: int = 0
    planning_time: float = 0.0
    smoothed_length: float = 0.0
    total_time: float = 0.0        # planning + optional post-processing


# ── single-planner runners ─────────────────────────

def _run_pure_rrt(env, q_start, q_goal, seed, **_kw) -> TrialResult:
    res = PureRRT(env, seed=seed).plan(q_start, q_goal)
    return TrialResult(
        planner_name="Pure RRT", success=res.success,
        iterations=res.iterations, tree_size=res.tree_size,
        planning_time=res.planning_time, smoothed_length=res.smoothed_length,
        total_time=res.planning_time,
    )


def _run_rrt_star(env, q_start, q_goal, seed, **_kw) -> TrialResult:
    res = RRTStar(env, seed=seed).plan(q_start, q_goal)
    return TrialResult(
        planner_name="RRT*", success=res.success,
        iterations=res.iterations, tree_size=res.tree_size,
        planning_time=res.planning_time, smoothed_length=res.smoothed_length,
        total_time=res.planning_time,
    )


def _run_apf_rrt(env, q_start, q_goal, seed, **_kw) -> TrialResult:
    res = APFGuidedRRT(env, seed=seed).plan(q_start, q_goal)
    return TrialResult(
        planner_name="APF-RRT", success=res.success,
        iterations=res.iterations, tree_size=res.tree_size,
        planning_time=res.planning_time, smoothed_length=res.smoothed_length,
        total_time=res.planning_time,
    )


def _run_apf_rrt_opt(env, q_start, q_goal, seed, verbose=False) -> TrialResult:
    """APF-RRT + gradient-descent optimizer on the smoothed path."""
    res = APFGuidedRRT(env, seed=seed).plan(q_start, q_goal)
    if not res.success:
        return TrialResult(
            planner_name="APF-RRT+Opt", success=False,
            iterations=res.iterations, tree_size=res.tree_size,
            planning_time=res.planning_time, total_time=res.planning_time,
        )

    cc = CollisionChecker(env._p, env.robot_id, env.obstacle_body_ids())
    opt = PathOptimizer(cc)
    t0 = time.perf_counter()
    resampled = PathOptimizer.resample_path(res.smoothed_path)
    opt_path = opt.optimize(resampled, verbose=verbose)
    t_opt = time.perf_counter() - t0

    return TrialResult(
        planner_name="APF-RRT+Opt", success=True,
        iterations=res.iterations, tree_size=res.tree_size,
        planning_time=res.planning_time,
        smoothed_length=PathOptimizer.path_length(opt_path),
        total_time=res.planning_time + t_opt,
    )


def _run_apf_rrt_spline(env, q_start, q_goal, seed, **_kw) -> TrialResult:
    """APF-RRT + cubic B-spline smoothing (no physics calls)."""
    res = APFGuidedRRT(env, seed=seed).plan(q_start, q_goal)
    if not res.success:
        return TrialResult(
            planner_name="APF-RRT+Spline", success=False,
            iterations=res.iterations, tree_size=res.tree_size,
            planning_time=res.planning_time, total_time=res.planning_time,
        )

    t0 = time.perf_counter()
    spline_path = SplineSmoother.smooth(res.smoothed_path, n_points=60)
    t_spline = time.perf_counter() - t0

    return TrialResult(
        planner_name="APF-RRT+Spline", success=True,
        iterations=res.iterations, tree_size=res.tree_size,
        planning_time=res.planning_time,
        smoothed_length=PathOptimizer.path_length(spline_path),
        total_time=res.planning_time + t_spline,
    )


# ── dispatcher ──────────────────────────────────────

_RUNNERS = {
    "Pure RRT":        _run_pure_rrt,
    "RRT*":            _run_rrt_star,
    "APF-RRT":         _run_apf_rrt,
    "APF-RRT+Opt":     _run_apf_rrt_opt,
    "APF-RRT+Spline":  _run_apf_rrt_spline,
}


def _run_trial(
    planners: List[str], seed: int, verbose: bool = False,
) -> Dict[str, TrialResult]:
    """Run selected planners on a single shared env."""
    q_start = PANDA_REST_POSES.copy()
    q_goal = load_demo_goal()
    results: Dict[str, TrialResult] = {}

    env = RRTEnvironment(render_mode="rgb_array")
    spawn_obstacles(env)

    for name in planners:
        results[name] = _RUNNERS[name](
            env, q_start, q_goal, seed, verbose=verbose,
        )

    env.close()
    return results


# ── aggregate & table ───────────────────────────────

def _print_table(
    planners: List[str], all_results: List[Dict[str, TrialResult]],
) -> None:
    n = len(all_results)

    header = (
        f"{'Planner':<18s} | {'Success':>8s} | {'Avg Time (s)':>12s} | "
        f"{'Avg Path Len':>12s} | {'Avg Nodes':>10s}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for name in planners:
        trials = [r[name] for r in all_results]
        n_success = sum(1 for t in trials if t.success)
        rate = f"{n_success}/{n}"

        successes = [t for t in trials if t.success]
        if successes:
            avg_time = np.mean([t.total_time for t in successes])
            avg_len = np.mean([t.smoothed_length for t in successes])
            avg_nodes = np.mean([t.tree_size for t in successes])
            print(
                f"{name:<18s} | {rate:>8s} | {avg_time:>12.3f} | "
                f"{avg_len:>12.3f} | {avg_nodes:>10.0f}"
            )
        else:
            print(
                f"{name:<18s} | {rate:>8s} | {'N/A':>12s} | "
                f"{'N/A':>12s} | {'N/A':>10s}"
            )

    print(sep)


def _print_detailed(
    planners: List[str], all_results: List[Dict[str, TrialResult]],
) -> None:
    """Print per-planner aggregate stats."""
    for name in planners:
        successes = [r[name] for r in all_results if r[name].success]
        failures = [r[name] for r in all_results if not r[name].success]
        ns, nf = len(successes), len(failures)

        print(f"\n  {name}:")
        print(f"    Successes: {ns}  Failures: {nf}")
        if successes:
            times = [t.total_time for t in successes]
            lengths = [t.smoothed_length for t in successes]
            nodes = [t.tree_size for t in successes]
            iters = [t.iterations for t in successes]
            print(f"    Time:   {np.mean(times):.3f}s  "
                  f"(std {np.std(times):.3f}, "
                  f"min {np.min(times):.3f}, max {np.max(times):.3f})")
            print(f"    Length: {np.mean(lengths):.3f}  "
                  f"(std {np.std(lengths):.3f})")
            print(f"    Nodes:  {np.mean(nodes):.0f}  "
                  f"(std {np.std(nodes):.0f})")
            print(f"    Iters:  {np.mean(iters):.0f}  "
                  f"(std {np.std(iters):.0f})")


# ── interactive menu ────────────────────────────────

def _choose_preset() -> str:
    """Display an interactive menu and return chosen preset key."""
    print("\n  Select benchmark preset:")
    print("    [1] apf   — APF-RRT  vs  APF-RRT+Opt  vs  APF-RRT+Spline  (default)")
    print("    [2] rrt   — Pure RRT  vs  RRT*")
    print("    [3] star  — RRT*  vs  APF-RRT+Opt")
    print("    [4] full  — all five planners (slow)")
    print()

    try:
        choice = input("  Choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    _map = {"1": "apf", "2": "rrt", "3": "star", "4": "full", "": "apf"}
    return _map.get(choice, "apf")


# ── entry point ─────────────────────────────────────

def run_benchmark(
    n_trials: int = 20,
    verbose: bool = False,
    preset: Optional[str] = None,
) -> None:
    # Resolve preset
    if preset is None:
        preset = _choose_preset()
    planners = PRESETS.get(preset, PRESETS["apf"])

    print("=" * 64)
    print(f"  Benchmark: {' vs '.join(planners)}")
    print(f"  Trials: {n_trials}   |   Demo obstacle scene")
    print("=" * 64)

    all_results: List[Dict[str, TrialResult]] = []

    for trial in range(n_trials):
        seed = 1000 + trial
        t0 = time.perf_counter()
        results = _run_trial(planners, seed, verbose=verbose)
        elapsed = time.perf_counter() - t0
        all_results.append(results)

        status = "  ".join(
            f"{name[:10]:>10s}={'OK' if results[name].success else 'FAIL'}"
            for name in planners
        )
        print(f"  Trial {trial + 1:2d}/{n_trials}  ({elapsed:5.1f}s)  {status}")

    _print_table(planners, all_results)
    _print_detailed(planners, all_results)


# ── CLI ─────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Panda RRT planner benchmark")
parser.add_argument("--trials", type=int,
                    default=cfg("benchmark", "n_trials"),
                    help="Number of randomised trials")
parser.add_argument("--verbose", action="store_true",
                    help="Print optimizer progress per trial")
parser.add_argument("--preset", type=str, default=None,
                    choices=list(PRESETS.keys()),
                    help="Planner preset (skip interactive menu)")
args = parser.parse_args()

run_benchmark(n_trials=args.trials, verbose=args.verbose, preset=args.preset)
