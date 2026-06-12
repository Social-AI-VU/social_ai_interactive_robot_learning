import gymnasium as gym
import numpy as np

from feedback.feedback_queue import feedback_queue


class FrankaReachEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, robot):

        super().__init__()

        self.robot = robot

        self.goal = np.array(
            [0.6, 0.2, 0.3],
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

    def _get_obs(self):

        ee = self.robot.get_ee_position()

        return np.concatenate(
            [ee, self.goal]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.robot.reset()

        return self._get_obs(), {}

    def step(self, action):

        self.robot.apply_action(action)

        human_reward = 0.0
        correction = None

        while not feedback_queue.empty():

            item = feedback_queue.get()

            if "reward" in item:
                human_reward += item["reward"]

            if "correction" in item:
                correction = np.array(
                    item["correction"],
                    dtype=np.float32
                )

        ee = self.robot.get_ee_position()

        distance = np.linalg.norm(
            ee - self.goal
        )

        env_reward = -distance

        terminated = distance < 0.02

        info = {
            "human_reward": human_reward,
            "correction": correction,
            "distance": distance
        }

        reward = env_reward

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            info
        )