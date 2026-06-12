import numpy as np


class FrankaInterface:
    """
    Mock Franka robot.
    """

    def __init__(self):

        self.ee_position = np.array(
            [0.4, 0.0, 0.3],
            dtype=np.float32
        )

    def reset(self):

        self.ee_position = np.array(
            [0.4, 0.0, 0.3],
            dtype=np.float32
        )

    def get_ee_position(self):

        return self.ee_position.copy()

    def apply_action(self, action):

        self.ee_position += 0.01 * action

    def close(self):
        pass