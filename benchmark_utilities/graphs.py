"""Matplotlib benchmark graphs: bar + box plots of planner metrics.

Generates a 2x2 figure with:

1. Success Rate (bar)
2. Computation Time (box)
3. Path Length (box)
4. Node Count (box)

Usage::

    from panda_rrt.benchmark_utilities.graphs import generate_benchmark_graphs
    generate_benchmark_graphs(all_results, planners, save_path="bench.png")
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def generate_benchmark_graphs(
    all_results: list,
    planners: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Create a 2x2 figure summarising benchmark results.

    Parameters
    ----------
    all_results : list of dict
        As returned by :func:`runner.run_benchmark`.
    planners : list of str
        Planner names (determines bar ordering).
    save_path : str or None
        If given, save the figure instead of showing.
    """
    import matplotlib.pyplot as plt

    n_trials = len(all_results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Benchmark Results ({n_trials} trials)",
        fontsize=14, fontweight="bold",
    )

    # Gather per-planner data
    success_rates: List[float] = []
    time_data: List[List[float]] = []
    length_data: List[List[float]] = []
    node_data: List[List[float]] = []

    for name in planners:
        trials = [r[name] for r in all_results]
        n_succ = sum(1 for t in trials if t.success)
        success_rates.append(100.0 * n_succ / max(n_trials, 1))
        successes = [t for t in trials if t.success]
        time_data.append([t.total_time for t in successes] or [0.0])
        length_data.append([t.smoothed_length for t in successes] or [0.0])
        node_data.append([t.tree_size for t in successes] or [0])

    short_names = [_shorten(n) for n in planners]

    # 1. Success Rate (bar chart)
    ax = axes[0, 0]
    colours = plt.cm.Set2(np.linspace(0, 1, len(planners)))
    bars = ax.bar(short_names, success_rates, color=colours, edgecolor="black")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate")
    ax.set_ylim(0, 110)
    for bar, rate in zip(bars, success_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{rate:.0f}%", ha="center", va="bottom", fontsize=9,
        )

    # 2. Computation Time (box plot)
    ax = axes[0, 1]
    bp = ax.boxplot(time_data, labels=short_names, patch_artist=True)
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
    ax.set_ylabel("Time (s)")
    ax.set_title("Computation Time")

    # 3. Path Length (box plot)
    ax = axes[1, 0]
    bp = ax.boxplot(length_data, labels=short_names, patch_artist=True)
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
    ax.set_ylabel("Path Length (rad)")
    ax.set_title("Path Length (smoothed)")

    # 4. Node Count (box plot)
    ax = axes[1, 1]
    bp = ax.boxplot(node_data, labels=short_names, patch_artist=True)
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
    ax.set_ylabel("Nodes")
    ax.set_title("Tree Node Count")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Graph saved -> {save_path}")
    else:
        plt.show()


def _shorten(name: str) -> str:
    """Abbreviate planner names for tick labels."""
    _MAP = {
        "Pure RRT":       "RRT",
        "RRT*":           "RRT*",
        "APF-RRT":        "APF",
        "APF-RRT+Opt":    "APF+O",
        "APF-RRT+Spline": "APF+S",
    }
    return _MAP.get(name, name[:8])
