"""Artificial Potential Field (APF) force functions.

When Numba is installed the heavy arithmetic is JIT-compiled; otherwise
the functions fall back to equivalent pure-NumPy implementations
automatically.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from panda_rrt.computations.jit_kernels import HAS_NUMBA

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def _wrap(fn):
            return fn
        return _wrap


_EMPTY_OBS = np.empty((0, 3), dtype=np.float64)


def _pack_obstacles(obstacles: Sequence[np.ndarray]) -> np.ndarray:
    if len(obstacles) == 0:
        return _EMPTY_OBS
    return np.ascontiguousarray(obstacles, dtype=np.float64).reshape(-1, 3)


@njit(cache=True)
def _attractive_jit(position, goal, xi):
    out = np.empty(3)
    for i in range(3):
        out[i] = xi * (goal[i] - position[i])
    return out


@njit(cache=True)
def _repulsive_jit(position, obstacles, n_obs, eta, d_thresh):
    out = np.zeros(3)
    for k in range(n_obs):
        d_sq = 0.0
        for i in range(3):
            diff = position[i] - obstacles[k, i]
            d_sq += diff * diff
        d = np.sqrt(d_sq)
        if d < 1e-6:
            d = 1e-6
        if d < d_thresh:
            mag = eta * (1.0 / d - 1.0 / d_thresh) / (d * d)
            for i in range(3):
                out[i] += mag * (position[i] - obstacles[k, i]) / d
    return out


@njit(cache=True)
def _total_apf_jit(position, goal, obstacles, n_obs, xi, eta, d_thresh):
    out = np.empty(3)
    for i in range(3):
        out[i] = xi * (goal[i] - position[i])
    for k in range(n_obs):
        d_sq = 0.0
        for i in range(3):
            diff = position[i] - obstacles[k, i]
            d_sq += diff * diff
        d = np.sqrt(d_sq)
        if d < 1e-6:
            d = 1e-6
        if d < d_thresh:
            mag = eta * (1.0 / d - 1.0 / d_thresh) / (d * d)
            for i in range(3):
                out[i] += mag * (position[i] - obstacles[k, i]) / d
    return out


def attractive_force(
    position: np.ndarray,
    goal: np.ndarray,
    xi: float = 1.0,
) -> np.ndarray:
    return _attractive_jit(
        np.ascontiguousarray(position, dtype=np.float64),
        np.ascontiguousarray(goal, dtype=np.float64),
        xi,
    )


def repulsive_force(
    position: np.ndarray,
    obstacles: Sequence[np.ndarray],
    eta: float = 0.005,
    d_thresh: float = 0.12,
) -> np.ndarray:
    obs_2d = _pack_obstacles(obstacles)
    return _repulsive_jit(
        np.ascontiguousarray(position, dtype=np.float64),
        obs_2d,
        obs_2d.shape[0],
        eta,
        d_thresh,
    )


def total_field_force(
    position: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[np.ndarray],
    xi: float = 1.0,
    eta: float = 0.005,
    d_thresh: float = 0.12,
) -> np.ndarray:
    obs_2d = _pack_obstacles(obstacles)
    return _total_apf_jit(
        np.ascontiguousarray(position, dtype=np.float64),
        np.ascontiguousarray(goal, dtype=np.float64),
        obs_2d,
        obs_2d.shape[0],
        xi,
        eta,
        d_thresh,
    )
