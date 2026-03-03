import pytest
import numpy as np

from panda_rrt.environment import RRTEnvironment
from panda_rrt.computations.collision import CollisionChecker, COLLISION_LINKS
from panda_rrt.computations.cspace_apf import CSpaceAPF
from panda_rrt.planners.apf_rrt import APFGuidedRRT
from panda_rrt.planners.core import Node, PlannerResult
from panda_rrt.common_utils.robot_constants import PANDA_REST_POSES, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS


@pytest.fixture
def env_obs():
    e = RRTEnvironment(render_mode="rgb_array")
    rng = np.random.default_rng(42)
    e.generate_cluttered_canopy(n_obstacles=8, rng=rng)
    yield e
    e.close()


@pytest.fixture
def cc(env_obs):
    return CollisionChecker(env_obs._p, env_obs.robot_id, env_obs.obstacle_body_ids())


@pytest.fixture
def apf(cc):
    return CSpaceAPF(cc, k_att=1.0, k_rep=0.01, rho_0=0.15)


class TestCollisionChecker:
    def test_rest_pose_valid(self, cc):
        assert cc.is_config_valid(PANDA_REST_POSES)

    def test_out_of_limits(self, cc):
        q = PANDA_LOWER_LIMITS - 0.5
        assert not cc.is_config_valid(q)

    def test_min_distance_positive_at_rest(self, cc):
        d, link, obs = cc.min_link_obstacle_distance(PANDA_REST_POSES)
        assert d > 0.0
        assert link in COLLISION_LINKS

    def test_per_link_distances_all_positive(self, cc):
        dists = cc.per_link_distances(PANDA_REST_POSES)
        assert set(dists.keys()) == set(COLLISION_LINKS)
        for link, d in dists.items():
            assert d > 0.0

    def test_edge_valid_identity(self, cc):
        q = PANDA_REST_POSES.copy()
        assert cc.is_edge_valid(q, q)

    def test_edge_valid_small_step(self, cc):
        q_a = PANDA_REST_POSES.copy()
        q_b = q_a.copy()
        q_b[0] += 0.01
        assert cc.is_edge_valid(q_a, q_b)

    def test_getClosestPoints_used(self, cc):
        cc.set_config(PANDA_REST_POSES)
        for obs_id in cc._obstacle_ids:
            contacts = cc._p.getClosestPoints(
                bodyA=cc._robot_id, bodyB=obs_id, distance=1.0,
            )
            assert isinstance(contacts, tuple)


class TestCSpaceAPF:
    def test_attractive_gradient_direction(self, apf):
        q = PANDA_REST_POSES.copy()
        q_goal = q + 0.5
        grad = apf.attractive_gradient(q, q_goal)
        assert grad.shape == (7,)
        for i in range(7):
            assert grad[i] < 0

    def test_attractive_gradient_zero_at_goal(self, apf):
        q = PANDA_REST_POSES.copy()
        grad = apf.attractive_gradient(q, q)
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)

    def test_attractive_gradient_quadratic(self, apf):
        """Within d_thresh the gradient is purely quadratic."""
        q = PANDA_REST_POSES.copy()
        q_goal = q + 0.3            # d ≈ 0.79 rad  (< d_thresh)
        grad = apf.attractive_gradient(q, q_goal)
        expected = apf.k_att * (q - q_goal)
        np.testing.assert_allclose(grad, expected, atol=1e-10)

    def test_attractive_gradient_conic(self, apf):
        """Beyond d_thresh the gradient magnitude is clamped."""
        q = PANDA_REST_POSES.copy()
        q_goal = q + 2.0            # d ≈ 5.29 rad  (> d_thresh)
        v = q - q_goal
        d = np.linalg.norm(v)
        grad = apf.attractive_gradient(q, q_goal)
        expected = apf.k_att * apf.d_thresh * v / d
        np.testing.assert_allclose(grad, expected, atol=1e-10)
        # Magnitude should equal k_att * d_thresh, not k_att * d
        assert np.linalg.norm(grad) < np.linalg.norm(apf.k_att * v)

    def test_repulsive_gradient_shape(self, apf):
        grad = apf.repulsive_gradient(PANDA_REST_POSES)
        assert grad.shape == (7,)

    def test_total_force_shape(self, apf):
        q_goal = PANDA_REST_POSES + 0.3
        q_goal = np.clip(q_goal, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)
        f = apf.total_force(PANDA_REST_POSES, q_goal)
        assert f.shape == (7,)

    def test_total_force_not_zero(self, apf):
        q_goal = PANDA_REST_POSES + 0.5
        q_goal = np.clip(q_goal, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)
        f = apf.total_force(PANDA_REST_POSES, q_goal)
        assert np.linalg.norm(f) > 0


class TestNode:
    def test_node_creation(self):
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        n = Node(q=q)
        np.testing.assert_array_equal(n.q, q)
        assert n.parent is None
        assert n.cost == 0.0

    def test_node_with_parent(self):
        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        n = Node(q=q, parent=3, cost=2.5)
        assert n.parent == 3
        assert n.cost == 2.5


class TestPlannerResult:
    def test_default_failure(self):
        r = PlannerResult(success=False)
        assert not r.success
        assert r.path == []
        assert r.tree_size == 0

    def test_success_fields(self):
        r = PlannerResult(
            success=True,
            path=[np.zeros(7)],
            tree_size=50,
            iterations=100,
            planning_time=0.5,
            path_length=2.0,
        )
        assert r.success
        assert len(r.path) == 1
        assert r.tree_size == 50
