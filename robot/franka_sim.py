import gymnasium as gym
import panda_gym.envs
import numpy as np


class FrankaSim:

    def __init__(self):
        self.env = gym.make(
            "PandaReach-v3",
            render_mode="human",
        )
        self.obs, _ = self.env.reset()

    def get_ee_position(self):
        return np.asarray(
            self.obs["achieved_goal"],
            dtype=np.float32,
        )

    def apply_action(self, action):
        self.obs, _, _, _, _ = (
            self.env.step(action)
        )
        self.env.render()

    def reset(self):
        self.obs, _ = self.env.reset()

    def get_goal(self):
        return np.asarray(
            self.obs["desired_goal"],
            dtype=np.float32,
        )