"""PyBullet visualisation helpers: markers, EE traces, path interpolation."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from panda_rrt.config import get as cfg
from panda_rrt.computations.forward_kinematics import PandaFK


def interpolate_path(
    path: List[np.ndarray],
    max_step: Optional[float] = None,
) -> List[np.ndarray]:
    """Densely interpolate a C-space path for smooth animation.

    Parameters
    ----------
    path : list of ndarray
        Sparse waypoints.
    max_step : float or None
        Maximum joint-space step between consecutive frames.
        Defaults to ``visualisation.animation_step`` from config.

    Returns
    -------
    list of ndarray
    """
    if max_step is None:
        max_step = cfg("visualisation", "animation_step")
    dense: List[np.ndarray] = [path[0].copy()]
    for i in range(len(path) - 1):
        diff = path[i + 1] - path[i]
        dist = np.linalg.norm(diff)
        n_seg = max(1, int(np.ceil(dist / max_step)))
        for k in range(1, n_seg + 1):
            dense.append(path[i] + diff * (k / n_seg))
    return dense


def draw_ee_trace(
    physics_client,
    fk: PandaFK,
    path: List[np.ndarray],
    color: Optional[List[float]] = None,
    width: Optional[float] = None,
) -> List[int]:
    """Draw lines through the EE positions along *path* in the viewer.

    Parameters
    ----------
    physics_client
        PyBullet physics client handle.
    fk : PandaFK
    path : list of ndarray
    color : list of float or None
    width : float or None

    Returns
    -------
    list of int
        PyBullet debug-line IDs.
    """
    if color is None:
        color = cfg("visualisation", "trace_color")
    if width is None:
        width = cfg("visualisation", "trace_width")

    line_ids: List[int] = []
    prev = fk.ee_position(path[0])
    for q in path[1:]:
        cur = fk.ee_position(q)
        lid = physics_client.addUserDebugLine(
            prev.tolist(), cur.tolist(),
            lineColorRGB=color[:3], lineWidth=width, lifeTime=0,
        )
        line_ids.append(lid)
        prev = cur
    return line_ids


def place_marker(
    sim,
    name: str,
    position: np.ndarray,
    color: np.ndarray,
    radius: Optional[float] = None,
) -> None:
    """Place a small translucent sphere as a visual marker.

    Parameters
    ----------
    sim : PyBullet
    name : str
    position : ndarray, shape (3,)
    color : ndarray, shape (4,)
    radius : float or None
    """
    if radius is None:
        radius = cfg("visualisation", "marker_radius")
    sim.create_sphere(
        body_name=name, radius=radius, mass=0.0,
        position=position, rgba_color=color,
    )
