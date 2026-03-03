"""Benchmark utilities: runner, graph generation, and visualisation.

Public API
----------
.. autosummary::
    runner.run_benchmark
    runner.TrialResult
    runner.PRESETS
    graphs.generate_benchmark_graphs
    tree_viz.plot_tree_3d
    tree_viz.plot_comparison
    tree_viz.plot_single
    visualisation.interpolate_path
    visualisation.draw_ee_trace
    visualisation.place_marker
"""
from panda_rrt.benchmark_utilities.runner import (
    run_benchmark,
    TrialResult,
    PRESETS,
)
from panda_rrt.benchmark_utilities.graphs import generate_benchmark_graphs
from panda_rrt.benchmark_utilities.tree_viz import (
    plot_tree_3d,
    plot_comparison,
    plot_single,
)
from panda_rrt.benchmark_utilities.visualisation import (
    interpolate_path,
    draw_ee_trace,
    place_marker,
)

__all__ = [
    "run_benchmark", "TrialResult", "PRESETS",
    "generate_benchmark_graphs",
    "plot_tree_3d", "plot_comparison", "plot_single",
    "interpolate_path", "draw_ee_trace", "place_marker",
]
