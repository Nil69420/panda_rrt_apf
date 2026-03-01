"""CLI entry-point for the Panda RRT motion-planning demo.

Usage::

    python -m panda_rrt.main                  # interactive planner menu
    python -m panda_rrt.main --rrt            # Pure RRT in GUI
    python -m panda_rrt.main --rrt_apf        # APF-Guided RRT in GUI
    python -m panda_rrt.main --rrt_apf_opt    # APF-RRT + optimisation in GUI
    python -m panda_rrt.main --rrt_apf_spline # APF-RRT + B-spline in GUI
    python -m panda_rrt.main --demo           # Headless single-plan
    python -m panda_rrt.main --benchmark      # Launch benchmark (interactive)
    python -m panda_rrt.main --rrt --delay 0.03
"""
import argparse

from panda_rrt.commands import compare, demo, run_visual

# ── CLI (no functions — just top-level script logic) ──

parser = argparse.ArgumentParser(
    description="Panda RRT motion-planning demo",
)
_group = parser.add_mutually_exclusive_group()
_group.add_argument("--rrt", action="store_true",
                    help="Visualise Pure RRT in the GUI")
_group.add_argument("--rrt_apf", action="store_true",
                    help="Visualise APF-Guided RRT in the GUI")
_group.add_argument("--rrt_apf_opt", action="store_true",
                    help="Visualise APF-RRT + gradient-descent optimisation")
_group.add_argument("--rrt_apf_spline", action="store_true",
                    help="Visualise APF-RRT + cubic B-spline smoothing")
_group.add_argument("--demo", action="store_true",
                    help="Headless single-plan demo (APF-RRT)")
_group.add_argument("--benchmark", action="store_true",
                    help="Launch the benchmark suite")
parser.add_argument("--trials", type=int, default=20,
                    help="Number of trials for comparison / benchmark mode")
parser.add_argument("--delay", type=float, default=None,
                    help="Delay per animation frame (s)")
parser.add_argument("--random", action="store_true",
                    help="Use random start/goal positions")
_scene_group = parser.add_mutually_exclusive_group()
_scene_group.add_argument("--wall", action="store_const", dest="scene",
                          const="wall", help="Wall obstacle scene (default)")
_scene_group.add_argument("--canopy", action="store_const", dest="scene",
                          const="canopy", help="Canopy obstacle scene")
parser.set_defaults(scene="wall")

args = parser.parse_args()

# ── dispatch ──

if args.benchmark:
    from panda_rrt.benchmark import run_benchmark
    run_benchmark(n_trials=args.trials, random_mode=args.random, scene=args.scene)
elif args.rrt:
    run_visual("pure", delay=args.delay, random_mode=args.random, scene=args.scene)
elif args.rrt_apf:
    run_visual("apf", delay=args.delay, random_mode=args.random, scene=args.scene)
elif args.rrt_apf_opt:
    run_visual("apf_opt", delay=args.delay, random_mode=args.random, scene=args.scene)
elif args.rrt_apf_spline:
    run_visual("apf_spline", delay=args.delay, random_mode=args.random, scene=args.scene)
elif args.demo:
    demo()
else:
    # No flag → interactive planner picker
    print("\n  Select a planner to visualise:")
    print("    [1] Pure RRT")
    print("    [2] APF-Guided RRT")
    print("    [3] APF-RRT + Gradient-Descent Optimisation")
    print("    [4] APF-RRT + Cubic B-Spline Smoothing")
    print("    [5] Headless demo (APF-RRT)")
    print("    [6] Benchmark suite")
    print()
    try:
        choice = input("  Choice [2]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    _map = {
        "1": "pure", "2": "apf", "3": "apf_opt",
        "4": "apf_spline", "": "apf",
    }
    if choice == "5":
        demo()
    elif choice == "6":
        from panda_rrt.benchmark import run_benchmark
        run_benchmark(n_trials=args.trials, random_mode=args.random, scene=args.scene)
    else:
        run_visual(_map.get(choice, "apf"), delay=args.delay,
                   random_mode=args.random, scene=args.scene)
