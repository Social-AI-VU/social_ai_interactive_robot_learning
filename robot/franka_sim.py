import gymnasium as gym
import panda_gym.envs
import numpy as np


class FrankaSim:
    """
    Thin wrapper around a panda-gym goal-conditioned env (obs is a dict with
    "observation", "achieved_goal", "desired_goal" - true of every panda-gym
    task, not just PandaReach-v3). Pass a different `env_id` to drive a
    different task; see environments/stack_cups/franka_stack_cups_env.py for
    an example that uses "PandaStack-v3" instead of the reach default.
    """

    def __init__(self, env_id="PandaReach-v3"):
        self.env = gym.make(
            env_id,
            render_mode="human",
        )
        self.obs, _ = self.env.reset()

    @property
    def action_space(self):
        return self.env.action_space

    def get_state(self):
        """Full underlying observation plus where the tracked point(s) currently are."""
        return np.concatenate(
            [self.obs["observation"], self.obs["achieved_goal"]]
        ).astype(np.float32)

    def get_achieved_goal(self):
        return np.asarray(
            self.obs["achieved_goal"],
            dtype=np.float32,
        )

    def get_ee_position(self):
        """For reach specifically, achieved_goal IS the end-effector position."""
        return self.get_achieved_goal()

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