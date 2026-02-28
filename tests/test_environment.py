import pytest
import numpy as np

from panda_rrt.environment import RRTEnvironment, Obstacle
from common_utils.robot_constants import PANDA_REST_POSES


@pytest.fixture
def env():
    e = RRTEnvironment(render_mode="rgb_array")
    yield e
    e.close()


@pytest.fixture
def env_with_obstacles():
    e = RRTEnvironment(render_mode="rgb_array")
    rng = np.random.default_rng(42)
    e.generate_cluttered_canopy(n_obstacles=8, rng=rng)
    yield e
    e.close()


class TestEnvironmentSetup:
    def test_robot_loaded(self, env):
        assert env.robot_id >= 0
        assert "panda" in env.sim._bodies_idx

    def test_table_constants(self, env):
        assert env.TABLE_X_OFFSET == -0.3
        assert env.TABLE_LENGTH == 1.1
        assert env.TABLE_WIDTH == 0.7
        assert env.TABLE_HEIGHT == 0.4

    def test_initial_config_is_rest(self, env):
        q = env.get_joint_config()
        np.testing.assert_allclose(q, PANDA_REST_POSES, atol=1e-4)

    def test_set_and_get_config(self, env):
        q = np.array([0.1, 0.2, 0.3, -1.0, 0.0, 2.0, 0.5])
        env._set_joint_config(q)
        q_read = env.get_joint_config()
        np.testing.assert_allclose(q_read, q, atol=1e-4)

    def test_ee_position_returns_3d(self, env):
        ee = env.get_ee_position()
        assert ee.shape == (3,)

    def test_ee_position_with_config(self, env):
        ee1 = env.get_ee_position(PANDA_REST_POSES)
        ee2 = env.get_ee_position(PANDA_REST_POSES)
        np.testing.assert_allclose(ee1, ee2, atol=1e-6)


class TestObstacles:
    def test_spawn_sphere(self, env):
        pos = np.array([0.2, 0.0, 0.3])
        obs = env.spawn_obstacle_sphere(pos, radius=0.04)
        assert obs.shape == "sphere"
        assert obs.radius == 0.04
        assert obs.body_id >= 0
        np.testing.assert_allclose(obs.position, pos)
        assert len(env.obstacles) == 1

    def test_spawn_cylinder(self, env):
        pos = np.array([-0.1, 0.1, 0.25])
        obs = env.spawn_obstacle_cylinder(pos, radius=0.03, height=0.2)
        assert obs.shape == "cylinder"
        assert obs.radius == 0.03
        assert obs.height == 0.2
        assert obs.body_id >= 0

    def test_obstacle_body_ids(self, env):
        env.spawn_obstacle_sphere(np.array([0.2, 0.0, 0.3]), radius=0.04)
        env.spawn_obstacle_cylinder(np.array([0.0, 0.1, 0.3]), radius=0.02, height=0.15)
        ids = env.obstacle_body_ids()
        assert len(ids) == 2
        assert all(isinstance(bid, int) for bid in ids)
        assert all(bid >= 0 for bid in ids)

    def test_obstacle_positions(self, env):
        p1 = np.array([0.1, 0.0, 0.3])
        p2 = np.array([0.0, 0.1, 0.4])
        env.spawn_obstacle_sphere(p1, radius=0.03)
        env.spawn_obstacle_cylinder(p2, radius=0.02, height=0.1)
        positions = env.obstacle_positions()
        assert len(positions) == 2
        np.testing.assert_allclose(positions[0], p1)

    def test_cluttered_canopy_count(self, env):
        rng = np.random.default_rng(42)
        obstacles = env.generate_cluttered_canopy(n_obstacles=8, rng=rng)
        assert len(obstacles) == 8
        assert len(env.obstacles) == 8

    def test_cluttered_canopy_rest_pose_valid(self, env):
        rng = np.random.default_rng(42)
        env.generate_cluttered_canopy(n_obstacles=10, rng=rng)
        assert env.is_config_valid(PANDA_REST_POSES)

    def test_cluttered_canopy_separation(self, env):
        rng = np.random.default_rng(99)
        obstacles = env.generate_cluttered_canopy(n_obstacles=6, rng=rng)
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                d = np.linalg.norm(obstacles[i].position - obstacles[j].position)
                assert d > 0.05


class TestCollisionChecking:
    def test_rest_pose_valid_without_obstacles(self, env):
        assert env.is_config_valid(PANDA_REST_POSES)

    def test_rest_pose_valid_with_obstacles(self, env_with_obstacles):
        assert env_with_obstacles.is_config_valid(PANDA_REST_POSES)

    def test_out_of_limits_invalid(self, env):
        from common_utils.robot_constants import PANDA_LOWER_LIMITS
        q = PANDA_LOWER_LIMITS - 0.1
        assert not env.is_config_valid(q)

    def test_edge_valid_small_step(self, env_with_obstacles):
        q_a = PANDA_REST_POSES.copy()
        q_b = q_a.copy()
        q_b[0] += 0.01
        assert env_with_obstacles.is_edge_valid(q_a, q_b)

    def test_edge_valid_same_config(self, env):
        q = PANDA_REST_POSES.copy()
        assert env.is_edge_valid(q, q)
