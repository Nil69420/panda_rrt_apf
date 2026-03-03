"""Thin CLI wrapper around :mod:`benchmark_utilities.runner`.

Usage::

    python3 -m panda_rrt.benchmark
    python3 -m panda_rrt.benchmark --preset apf --trials 10
    python3 -m panda_rrt.benchmark --graph --save bench.png
    python3 -m panda_rrt.benchmark --viz --planner all
"""
from __future__ import annotations

import argparse

from panda_rrt.benchmark_utilities.runner import run_benchmark, PRESETS
from panda_rrt.config import get as cfg

parser = argparse.ArgumentParser(
    description="Panda RRT planner benchmark",
)
parser.add_argument("--trials", type=int,
                    default=cfg("benchmark", "n_trials"),
                    help="Number of randomised trials")
parser.add_argument("--verbose", action="store_true",
                    help="Print optimizer progress per trial")
parser.add_argument("--random", action="store_true",
                    help="Use random start/goal positions each trial")
parser.add_argument("--preset", type=str, default=None,
                    choices=list(PRESETS.keys()),
                    help="Planner preset (skip interactive menu)")
parser.add_argument("--graph", action="store_true",
                    help="Generate benchmark graphs after running")
parser.add_argument("--viz", action="store_true",
                    help="3-D tree visualisation (Matplotlib)")
parser.add_argument("--planner", type=str, default="all",
                    choices=["pure_rrt", "rrt_star", "apf_rrt",
                             "apf_rrt_opt", "all"],
                    help="Planner for --viz mode")
parser.add_argument("--save", type=str, default=None,
                    help="Save figure to file instead of displaying")

_scene_grp = parser.add_mutually_exclusive_group()
_scene_grp.add_argument("--wall", action="store_const", dest="scene",
                        const="wall", help="Wall obstacle scene (default)")
_scene_grp.add_argument("--canopy", action="store_const", dest="scene",
                        const="canopy", help="Canopy obstacle scene")
_scene_grp.add_argument("--passage", action="store_const", dest="scene",
                        const="passage", help="Narrow passage scene")
parser.set_defaults(scene="wall")

args = parser.parse_args()

if args.viz:
    from panda_rrt.benchmark_utilities.tree_viz import plot_comparison, plot_single
    if args.planner == "all":
        plot_comparison(save_path=args.save, scene=args.scene)
    else:
        plot_single(planner_key=args.planner, save_path=args.save,
                    scene=args.scene)
else:
    all_results = run_benchmark(
        n_trials=args.trials, verbose=args.verbose,
        preset=args.preset, random_mode=args.random,
        scene=args.scene,
    )
    if args.graph:
        from panda_rrt.benchmark_utilities.graphs import generate_benchmark_graphs
        preset = args.preset or "apf"
        planners = PRESETS.get(preset, PRESETS["apf"])
        generate_benchmark_graphs(all_results, planners, save_path=args.save)
