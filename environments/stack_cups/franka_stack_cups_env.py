"""
Stack one cup on top of another.

This is the online-interactive-learning style of environment (paired with a
`robot` wrapper and a rewards/RewardSource, same shape as
environments/reach/franka_reach_env.py) - not the robosuite/teleoperation
style used by push, lift_nut, sandbox and reach_env.py. See
docs/adding_environments_tasks_learners.md for which style fits a new task.

Concretely this reuses panda-gym's PandaStack-v3 physics via
robot/franka_sim.py(env_id="PandaStack-v3"). panda-gym ships that task with
cubes, not cups - swap in a robot wrapper built on a cup-shaped MuJoCo asset
if the visual/contact geometry needs to be accurate. The reward/success
logic below only cares about object positions, not their shape.
"""

import gymnasium as gym
import numpy as np

from rewards.feedback_queue import feedback_queue


class FrankaStackCupsEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot,
        human_reward_weight=0.1,
        success_threshold=0.05,
    ):

        super().__init__()

        self.robot = robot
        self.human_reward_weight = human_reward_weight
        self.success_threshold = success_threshold

        self.robot.reset()
        state = self.robot.get_state()
        goal = self.robot.get_goal()

        self.action_space = self.robot.action_space

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state.shape[0] + goal.shape[0],),
            dtype=np.float32,
        )

    def _get_obs(self):

        return np.concatenate(
            [self.robot.get_state(), self.robot.get_goal()]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.robot.reset()

        while not feedback_queue.empty():
            feedback_queue.get()

        return self._get_obs(), {}

    def step(self, action):

        self.robot.apply_action(action)

        human_reward = 0.0

        while not feedback_queue.empty():

            item = feedback_queue.get()

            if "reward" in item:
                human_reward += float(
                    item["reward"]
                )

        achieved = self.robot.get_achieved_goal()
        goal = self.robot.get_goal()

        distance = np.linalg.norm(
            achieved - goal
        )

        env_reward = -distance

        reward = (
            env_reward
            + self.human_reward_weight
            * human_reward
        )

        terminated = distance < self.success_threshold

        info = {
            "distance": distance,
            "env_reward": env_reward,
            "human_reward": human_reward,
        }

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            info,
        )
