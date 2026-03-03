"""Benchmark runner: multi-trial planner comparison engine.

Provides the core benchmark logic that was previously in ``benchmark.py``.
CLI wrappers live in :mod:`panda_rrt.main` and :mod:`panda_rrt.benchmark`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from panda_rrt.common_utils.robot_constants import PANDA_REST_POSES
from panda_rrt.environment import RRTEnvironment
from panda_rrt.computations.collision import CollisionChecker
from panda_rrt.planners.pure_rrt import PureRRT
from panda_rrt.planners.rrt_star import RRTStar
from panda_rrt.planners.apf_rrt import APFGuidedRRT
from panda_rrt.planners.optimizer import PathOptimizer
from panda_rrt.computations.spline_smoother import SplineSmoother
from panda_rrt.planners.core import PlannerResult
from panda_rrt.scene import (
    load_demo_goal,
    spawn_obstacles,
    sample_random_goal,
    greedy_shortcut,
)
from panda_rrt.config import get as cfg


# ── preset definitions ──────────────────────────────

PRESETS: Dict[str, List[str]] = {
    "apf":  ["APF-RRT", "APF-RRT+Opt", "APF-RRT+Spline"],
    "rrt":  ["Pure RRT", "RRT*"],
    "star": ["RRT*", "APF-RRT+Opt"],
    "full": ["Pure RRT", "RRT*", "APF-RRT", "APF-RRT+Opt", "APF-RRT+Spline"],
}


# ── data container ──────────────────────────────────

@dataclass
class TrialResult:
    """Outcome of a single planner trial.

    Attributes
    ----------
    planner_name : str
    success : bool
    iterations, tree_size : int
    planning_time, smoothed_length, total_time : float
    """

    planner_name: str
    success: bool
    iterations: int = 0
    tree_size: int = 0
    planning_time: float = 0.0
    smoothed_length: float = 0.0
    total_time: float = 0.0


# ── single-planner runners (dict-dispatched) ───────

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
    """APF-RRT followed by gradient-descent optimisation."""
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
    """APF-RRT followed by cubic B-spline smoothing."""
    res = APFGuidedRRT(env, seed=seed).plan(q_start, q_goal)
    if not res.success:
        return TrialResult(
            planner_name="APF-RRT+Spline", success=False,
            iterations=res.iterations, tree_size=res.tree_size,
            planning_time=res.planning_time, total_time=res.planning_time,
        )
    t0 = time.perf_counter()
    pruned = greedy_shortcut(res.path, env)
    spline_path = SplineSmoother.smooth(pruned, n_points=60)
    t_spline = time.perf_counter() - t0
    return TrialResult(
        planner_name="APF-RRT+Spline", success=True,
        iterations=res.iterations, tree_size=res.tree_size,
        planning_time=res.planning_time,
        smoothed_length=PathOptimizer.path_length(spline_path),
        total_time=res.planning_time + t_spline,
    )


# Dict dispatch table replaces if/elif chains
_RUNNERS = {
    "Pure RRT":       _run_pure_rrt,
    "RRT*":           _run_rrt_star,
    "APF-RRT":        _run_apf_rrt,
    "APF-RRT+Opt":    _run_apf_rrt_opt,
    "APF-RRT+Spline": _run_apf_rrt_spline,
}


# ── trial execution ────────────────────────────────

def _run_trial(
    planners: List[str],
    seed: int,
    verbose: bool = False,
    random_mode: bool = False,
    scene: str = "wall",
) -> Dict[str, TrialResult]:
    """Execute all *planners* on a single shared environment."""
    env = RRTEnvironment(render_mode="rgb_array")
    spawn_obstacles(env, scene=scene)

    q_start = PANDA_REST_POSES.copy()
    if random_mode:
        q_goal = sample_random_goal(env, q_start, seed=seed)
    else:
        q_goal = load_demo_goal(scene)

    results = {
        name: _RUNNERS[name](env, q_start, q_goal, seed, verbose=verbose)
        for name in planners
    }
    env.close()
    return results


# ── aggregate / table printing ──────────────────────

def _print_table(
    planners: List[str], all_results: List[Dict[str, TrialResult]],
) -> None:
    n = len(all_results)
    header = (
        f"{'Planner':<18s} | {'Success':>8s} | {'Avg Time (s)':>12s} | "
        f"{'Avg Path Len':>12s} | {'Avg Nodes':>10s}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for name in planners:
        trials = [r[name] for r in all_results]
        n_success = sum(1 for t in trials if t.success)
        s = [t for t in trials if t.success]
        if s:
            print(
                f"{name:<18s} | {n_success}/{n:>5d} | "
                f"{np.mean([t.total_time for t in s]):>12.3f} | "
                f"{np.mean([t.smoothed_length for t in s]):>12.3f} | "
                f"{np.mean([t.tree_size for t in s]):>10.0f}"
            )
        else:
            print(
                f"{name:<18s} | {n_success}/{n:>5d} | "
                f"{'N/A':>12s} | {'N/A':>12s} | {'N/A':>10s}"
            )
    print(sep)


def _print_detailed(
    planners: List[str], all_results: List[Dict[str, TrialResult]],
) -> None:
    for name in planners:
        successes = [r[name] for r in all_results if r[name].success]
        failures = [r[name] for r in all_results if not r[name].success]
        print(f"\n  {name}:")
        print(f"    Successes: {len(successes)}  Failures: {len(failures)}")
        if successes:
            t = [s.total_time for s in successes]
            l_ = [s.smoothed_length for s in successes]
            nd = [s.tree_size for s in successes]
            it = [s.iterations for s in successes]
            print(f"    Time:   {np.mean(t):.3f}s (std {np.std(t):.3f}, "
                  f"min {np.min(t):.3f}, max {np.max(t):.3f})")
            print(f"    Length: {np.mean(l_):.3f} (std {np.std(l_):.3f})")
            print(f"    Nodes:  {np.mean(nd):.0f} (std {np.std(nd):.0f})")
            print(f"    Iters:  {np.mean(it):.0f} (std {np.std(it):.0f})")


# ── interactive menu ────────────────────────────────

def _choose_preset() -> str:
    print("\n  Select benchmark preset:")
    print("    [1] apf   -- APF-RRT  vs  APF-RRT+Opt  vs  APF-RRT+Spline  (default)")
    print("    [2] rrt   -- Pure RRT  vs  RRT*")
    print("    [3] star  -- RRT*  vs  APF-RRT+Opt")
    print("    [4] full  -- all five planners (slow)")
    print()
    try:
        choice = input("  Choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""
    _map = {"1": "apf", "2": "rrt", "3": "star", "4": "full", "": "apf"}
    return _map.get(choice, "apf")


# ── public entry point ──────────────────────────────

def run_benchmark(
    n_trials: int = 20,
    verbose: bool = False,
    preset: Optional[str] = None,
    random_mode: bool = False,
    scene: str = "wall",
) -> List[Dict[str, TrialResult]]:
    """Run a multi-trial planner comparison.

    Parameters
    ----------
    n_trials : int
    verbose : bool
    preset : str or None
        One of ``"apf"``, ``"rrt"``, ``"star"``, ``"full"``.
        If ``None``, an interactive menu is shown.
    random_mode : bool
    scene : str

    Returns
    -------
    list of dict
        One dict per trial mapping planner name to :class:`TrialResult`.
    """
    if preset is None:
        preset = _choose_preset()
    planners = PRESETS.get(preset, PRESETS["apf"])

    mode_label = "random start/goal" if random_mode else "demo obstacle scene"
    print("=" * 64)
    print(f"  Benchmark: {' vs '.join(planners)}")
    print(f"  Trials: {n_trials}   |   {mode_label}   |   scene: {scene}")
    print("=" * 64)

    all_results: List[Dict[str, TrialResult]] = []
    for trial in range(n_trials):
        seed = 1000 + trial
        t0 = time.perf_counter()
        results = _run_trial(
            planners, seed, verbose=verbose, random_mode=random_mode,
            scene=scene,
        )
        elapsed = time.perf_counter() - t0
        all_results.append(results)
        status = "  ".join(
            f"{name[:10]:>10s}={'OK' if results[name].success else 'FAIL'}"
            for name in planners
        )
        print(f"  Trial {trial + 1:2d}/{n_trials}  ({elapsed:5.1f}s)  {status}")

    _print_table(planners, all_results)
    _print_detailed(planners, all_results)
    return all_results
