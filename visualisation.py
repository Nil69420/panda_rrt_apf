"""PyBullet visualisation helpers: markers, EE traces, path interpolation."""
from __future__ import annotations

from typing import List

import numpy as np

from panda_rrt.config import get as cfg
from panda_rrt.forward_kinematics import PandaFK


def interpolate_path(
    path: List[np.ndarray],
    max_step: float | None = None,
) -> List[np.ndarray]:
    """Densely interpolate a C-space path for smooth animation."""
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
    color: List[float] | None = None,
    width: float | None = None,
) -> List[int]:
    """Draw lines through the EE positions along *path* in the viewer."""
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
    radius: float | None = None,
) -> None:
    """Place a small translucent sphere as a visual marker."""
    if radius is None:
        radius = cfg("visualisation", "marker_radius")
    sim.create_sphere(
        body_name=name, radius=radius, mass=0.0,
        position=position, rgba_color=color,
    )
