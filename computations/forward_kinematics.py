"""Forward kinematics for the Franka Emika Panda via PyBullet.

Wraps :pep:`PyBullet's <getLinkState>` FK and Jacobian queries behind a
lightweight interface that accepts joint-configuration vectors.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from panda_rrt.common_utils.robot_constants import (
    PANDA_EE_LINK,
    PANDA_NUM_JOINTS,
    PANDA_REST_POSES,
)


class PandaFK:
    """PyBullet-backed forward kinematics for the 7-DOF Panda arm.

    Parameters
    ----------
    physics_client : BulletClient
        Active PyBullet physics client.
    robot_body_id : int
        Body index returned by ``loadURDF``.
    """

    def __init__(self, physics_client, robot_body_id: int) -> None:
        self._p = physics_client
        self._robot_id = robot_body_id
        self._n_joints = PANDA_NUM_JOINTS
        self._ee_link = PANDA_EE_LINK

    def _apply(self, q: np.ndarray) -> None:
        """Set the robot's joint state without simulating dynamics."""
        for i in range(self._n_joints):
            self._p.resetJointState(self._robot_id, i, q[i])

    def ee_position(self, q: np.ndarray) -> np.ndarray:
        """Return the 3-D world-frame position of the end-effector.

        Parameters
        ----------
        q : ndarray, shape (7,)
            Joint configuration.

        Returns
        -------
        ndarray, shape (3,)
        """
        self._apply(q)
        return np.array(self._p.getLinkState(self._robot_id, self._ee_link)[0])

    def link_positions(self, q: np.ndarray) -> Dict[int, np.ndarray]:
        """Map from link index to world-frame position for all links.

        Parameters
        ----------
        q : ndarray, shape (7,)

        Returns
        -------
        dict[int, ndarray]
            ``{link_idx: pos_3d}`` for links 0 .. ``PANDA_EE_LINK``.
        """
        self._apply(q)
        positions: Dict[int, np.ndarray] = {}
        for link in range(self._ee_link + 1):
            positions[link] = np.array(
                self._p.getLinkState(self._robot_id, link)[0]
            )
        return positions

    def link_position(self, q: np.ndarray, link: int) -> np.ndarray:
        """World-frame position of a single link.

        Parameters
        ----------
        q : ndarray, shape (7,)
        link : int
            Link index (0-based).

        Returns
        -------
        ndarray, shape (3,)
        """
        self._apply(q)
        return np.array(self._p.getLinkState(self._robot_id, link)[0])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        r"""Linear Jacobian of the end-effector: :math:`J \in \mathbb{R}^{3 \times 7}`.

        Relates joint velocities to EE linear velocity via
        :math:`\dot{x} = J\,\dot{q}`.

        Parameters
        ----------
        q : ndarray, shape (7,)

        Returns
        -------
        ndarray, shape (3, 7)
        """
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
        return np.array(lin_jac)[:, : self._n_joints]

    def verify(self, q: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Diagnostic: compute and return all FK quantities at *q*.

        Parameters
        ----------
        q : ndarray or None
            Defaults to ``PANDA_REST_POSES``.

        Returns
        -------
        dict
            Keys: ``config``, ``ee_position``, ``link_positions``,
            ``jacobian_shape``, ``jacobian``.
        """
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
