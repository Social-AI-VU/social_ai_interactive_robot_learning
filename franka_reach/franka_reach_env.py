import gymnasium as gym
import numpy as np

from feedback.feedback_queue import (
    feedback_queue,
)


class FrankaReachEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot,
        human_reward_weight=0.1,
    ):

        super().__init__()

        self.robot = robot

        self.goal = np.zeros(
            3,
            dtype=np.float32,
        )

        self.human_reward_weight = (
            human_reward_weight
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

    def _get_obs(self):

        ee = self.robot.get_ee_position()

        return np.concatenate(
            [ee, self.goal]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.robot.reset()

        while not feedback_queue.empty():
            feedback_queue.get()

        print("\nEnter goal position")

        x = float(input("x: "))
        y = float(input("y: "))
        z = float(input("z: "))

        self.goal = np.array(
            [x, y, z],
            dtype=np.float32,
        )

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

        ee = self.robot.get_ee_position()

        distance = np.linalg.norm(
            ee - self.goal
        )

        env_reward = -distance

        reward = (
            env_reward
            + self.human_reward_weight
            * human_reward
        )

        terminated = distance < 0.02

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