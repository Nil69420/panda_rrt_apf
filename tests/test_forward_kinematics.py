import pytest
import numpy as np

from panda_rrt.environment import RRTEnvironment
from panda_rrt.forward_kinematics import PandaFK
from panda_rrt.common_utils.robot_constants import PANDA_REST_POSES


@pytest.fixture
def fk():
    env = RRTEnvironment(render_mode="rgb_array")
    f = PandaFK(env._p, env.robot_id)
    yield f, env
    env.close()


class TestForwardKinematics:
    def test_ee_position_shape(self, fk):
        f, _ = fk
        ee = f.ee_position(PANDA_REST_POSES)
        assert ee.shape == (3,)

    def test_ee_position_deterministic(self, fk):
        f, _ = fk
        ee1 = f.ee_position(PANDA_REST_POSES)
        ee2 = f.ee_position(PANDA_REST_POSES)
        np.testing.assert_allclose(ee1, ee2, atol=1e-8)

    def test_different_configs_different_ee(self, fk):
        f, _ = fk
        ee1 = f.ee_position(PANDA_REST_POSES)
        q2 = PANDA_REST_POSES.copy()
        q2[0] += 0.5
        ee2 = f.ee_position(q2)
        assert not np.allclose(ee1, ee2)

    def test_link_positions_count(self, fk):
        f, _ = fk
        links = f.link_positions(PANDA_REST_POSES)
        assert len(links) == 12

    def test_link_positions_are_3d(self, fk):
        f, _ = fk
        links = f.link_positions(PANDA_REST_POSES)
        for link_id, pos in links.items():
            assert pos.shape == (3,)

    def test_jacobian_shape(self, fk):
        f, _ = fk
        J = f.jacobian(PANDA_REST_POSES)
        assert J.shape == (3, 7)

    def test_jacobian_not_zero(self, fk):
        f, _ = fk
        J = f.jacobian(PANDA_REST_POSES)
        assert np.linalg.norm(J) > 0

    def test_verify_keys(self, fk):
        f, _ = fk
        result = f.verify(PANDA_REST_POSES)
        assert "config" in result
        assert "ee_position" in result
        assert "link_positions" in result
        assert "jacobian_shape" in result
        assert "jacobian" in result
