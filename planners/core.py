"""Shared data structures used by all planner variants.

Classes
-------
Node
    A single vertex in the RRT search tree.
PlannerResult
    Immutable container returned by every ``plan()`` call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Node:
    """Single RRT tree vertex.

    Attributes
    ----------
    q : ndarray, shape (7,)
        Joint configuration.
    parent : int or None
        Index of the parent node in the tree list, ``None`` for the root.
    cost : float
        Cumulative cost-to-root (arc-length in C-space).
    """

    q: np.ndarray
    parent: Optional[int] = None
    cost: float = 0.0


@dataclass
class PlannerResult:
    """Container for the output of a single planning query.

    Attributes
    ----------
    success : bool
        Whether a feasible path was found.
    path : list[ndarray]
        Raw waypoints from root to goal (empty on failure).
    smoothed_path : list[ndarray]
        Waypoints after shortcut smoothing.
    tree_size : int
        Number of nodes in the final tree.
    iterations : int
        Expansion iterations executed.
    planning_time : float
        Wall-clock seconds spent planning.
    path_length : float
        Arc-length of the raw path (radians in C-space).
    smoothed_length : float
        Arc-length of the smoothed path.
    """

    success: bool
    path: List[np.ndarray] = field(default_factory=list)
    smoothed_path: List[np.ndarray] = field(default_factory=list)
    tree_size: int = 0
    iterations: int = 0
    planning_time: float = 0.0
    path_length: float = 0.0
    smoothed_length: float = 0.0
