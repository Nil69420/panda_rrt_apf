"""Unified CLI entry-point for the Panda RRT motion-planning project.

This script contains **no function definitions**; all logic lives in
:mod:`panda_rrt.commands`.

Usage::

    python -m panda_rrt.main                    # interactive planner menu
    python -m panda_rrt.main --rrt              # Pure RRT in GUI
    python -m panda_rrt.main --rrt_apf          # APF-Guided RRT in GUI
    python -m panda_rrt.main --rrt_apf_opt      # APF-RRT + optimisation
    python -m panda_rrt.main --rrt_apf_spline   # APF-RRT + B-spline
    python -m panda_rrt.main --demo             # Headless single-plan
    python -m panda_rrt.main --benchmark        # Multi-trial benchmark
    python -m panda_rrt.main --graph            # Generate benchmark graphs
    python -m panda_rrt.main --viz              # 3-D tree plot
"""
from panda_rrt.commands import (
    build_parser,
    demo,
    do_benchmark,
    do_graph,
    do_viz,
    interactive_menu,
    run_visual,
    FLAG_DISPATCH,
)

args = build_parser().parse_args()

if args.benchmark:
    do_benchmark(args)
elif args.graph:
    do_graph(args)
elif args.viz:
    do_viz(args)
elif args.demo:
    demo()
else:
    for flag, key in FLAG_DISPATCH.items():
        if getattr(args, flag, False):
            run_visual(key, delay=args.delay, random_mode=args.random,
                       scene=args.scene)
            break
    else:
        interactive_menu(args)
