from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from common_utils.robot_constants import (
    PANDA_EE_LINK,
    PANDA_NUM_JOINTS,
    PANDA_REST_POSES,
)


class PandaFK:
    def __init__(self, physics_client, robot_body_id: int) -> None:
        self._p = physics_client
        self._robot_id = robot_body_id
        self._n_joints = PANDA_NUM_JOINTS
        self._ee_link = PANDA_EE_LINK

    def _apply(self, q: np.ndarray) -> None:
        for i in range(self._n_joints):
            self._p.resetJointState(self._robot_id, i, q[i])

    def ee_position(self, q: np.ndarray) -> np.ndarray:
        self._apply(q)
        return np.array(self._p.getLinkState(self._robot_id, self._ee_link)[0])

    def link_positions(self, q: np.ndarray) -> Dict[int, np.ndarray]:
        self._apply(q)
        positions: Dict[int, np.ndarray] = {}
        for link in range(self._ee_link + 1):
            positions[link] = np.array(self._p.getLinkState(self._robot_id, link)[0])
        return positions

    def link_position(self, q: np.ndarray, link: int) -> np.ndarray:
        self._apply(q)
        return np.array(self._p.getLinkState(self._robot_id, link)[0])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        self._apply(q)
        zero_vec = [0.0] * self._n_joints
        q_list = q.tolist()
        lin_jac, _ = self._p.calculateJacobian(
            bodyUniqueId=self._robot_id,
            linkIndex=self._ee_link,
            localPosition=[0.0, 0.0, 0.0],
            objPositions=q_list + [0.0, 0.0],
            objVelocities=zero_vec + [0.0, 0.0],
            objAccelerations=zero_vec + [0.0, 0.0],
        )
        return np.array(lin_jac)[:, :self._n_joints]

    def verify(self, q: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if q is None:
            q = PANDA_REST_POSES
        ee = self.ee_position(q)
        links = self.link_positions(q)
        jac = self.jacobian(q)
        return {
            "config": q.copy(),
            "ee_position": ee,
            "link_positions": links,
            "jacobian_shape": np.array(jac.shape),
            "jacobian": jac,
        }
