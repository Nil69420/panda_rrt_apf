import pytest
import numpy as np

from panda_rrt.environment import RRTEnvironment
from panda_rrt.planner import APFGuidedRRT
from panda_rrt.pure_rrt import PureRRT
from panda_rrt.common_utils.robot_constants import (
    PANDA_REST_POSES, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS, PANDA_JOINT_RANGES,
)


def _make_env_and_goal(seed=42, n_obstacles=6):
    env = RRTEnvironment(render_mode="rgb_array")
    rng = np.random.default_rng(seed)
    env.generate_cluttered_canopy(n_obstacles=n_obstacles, rng=rng)

    rng_goal = np.random.default_rng(seed + 500)
    q_goal = None
    for _ in range(500):
        q = PANDA_LOWER_LIMITS + rng_goal.random(7) * PANDA_JOINT_RANGES
        if env.is_config_valid(q):
            q_goal = q
            break
    return env, q_goal


class TestAPFGuidedRRT:
    def test_plan_invalid_start(self):
        env = RRTEnvironment(render_mode="rgb_array")
        planner = APFGuidedRRT(env, seed=0)
        q_bad = PANDA_LOWER_LIMITS - 1.0
        result = planner.plan(q_bad, PANDA_REST_POSES)
        assert not result.success
        env.close()

    def test_plan_invalid_goal(self):
        env = RRTEnvironment(render_mode="rgb_array")
        planner = APFGuidedRRT(env, seed=0)
        q_bad = PANDA_UPPER_LIMITS + 1.0
        result = planner.plan(PANDA_REST_POSES, q_bad)
        assert not result.success
        env.close()

    def test_plan_finds_path(self):
        env, q_goal = _make_env_and_goal(seed=42, n_obstacles=5)
        assert q_goal is not None
        planner = APFGuidedRRT(env, seed=42, max_iterations=5000)
        result = planner.plan(PANDA_REST_POSES, q_goal)
        assert result.success
        assert len(result.path) >= 2
        assert result.tree_size > 0
        assert result.iterations > 0
        assert result.planning_time > 0
        env.close()

    def test_path_starts_and_ends_correctly(self):
        env, q_goal = _make_env_and_goal(seed=42, n_obstacles=5)
        planner = APFGuidedRRT(env, seed=42, max_iterations=5000)
        result = planner.plan(PANDA_REST_POSES, q_goal)
        if result.success:
            np.testing.assert_allclose(result.path[0], PANDA_REST_POSES, atol=1e-6)
            np.testing.assert_allclose(result.path[-1], q_goal, atol=1e-6)
        env.close()

    def test_smoothed_path_shorter_or_equal(self):
        env, q_goal = _make_env_and_goal(seed=42, n_obstacles=5)
        planner = APFGuidedRRT(env, seed=42, max_iterations=5000)
        result = planner.plan(PANDA_REST_POSES, q_goal)
        if result.success:
            assert result.smoothed_length <= result.path_length + 1e-6
        env.close()

    def test_nodes_have_cost(self):
        env, q_goal = _make_env_and_goal(seed=42, n_obstacles=5)
        planner = APFGuidedRRT(env, seed=42, max_iterations=5000)
        result = planner.plan(PANDA_REST_POSES, q_goal)
        if result.success:
            assert planner.nodes[0].cost == 0.0
            for n in planner.nodes[1:]:
                assert n.cost > 0.0
        env.close()

    def test_node_parent_chain(self):
        env, q_goal = _make_env_and_goal(seed=42, n_obstacles=5)
        planner = APFGuidedRRT(env, seed=42, max_iterations=5000)
        result = planner.plan(PANDA_REST_POSES, q_goal)
        if result.success:
            idx = len(planner.nodes) - 1
            visited = set()
            while idx is not None:
                assert idx not in visited
                visited.add(idx)
                idx = planner.nodes[idx].parent
        env.close()


class TestPureRRT:
    def test_plan_invalid_start(self):
        env = RRTEnvironment(render_mode="rgb_array")
        planner = PureRRT(env, seed=0)
        q_bad = PANDA_LOWER_LIMITS - 1.0
        result = planner.plan(q_bad, PANDA_REST_POSES)
        assert not result.success
        env.close()

    def test_plan_finds_path(self):
        env, q_goal = _make_env_and_goal(seed=42, n_obstacles=5)
        assert q_goal is not None
        planner = PureRRT(env, seed=42, max_iterations=5000)
        result = planner.plan(PANDA_REST_POSES, q_goal)
        assert result.success
        assert len(result.path) >= 2
        env.close()

    def test_path_starts_and_ends_correctly(self):
        env, q_goal = _make_env_and_goal(seed=42, n_obstacles=5)
        planner = PureRRT(env, seed=42, max_iterations=5000)
        result = planner.plan(PANDA_REST_POSES, q_goal)
        if result.success:
            np.testing.assert_allclose(result.path[0], PANDA_REST_POSES, atol=1e-6)
            np.testing.assert_allclose(result.path[-1], q_goal, atol=1e-6)
        env.close()

    def test_nodes_have_cost(self):
        env, q_goal = _make_env_and_goal(seed=42, n_obstacles=5)
        planner = PureRRT(env, seed=42, max_iterations=5000)
        result = planner.plan(PANDA_REST_POSES, q_goal)
        if result.success:
            assert planner.nodes[0].cost == 0.0
        env.close()
