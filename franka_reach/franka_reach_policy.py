import numpy as np

class ReachPolicy:

    def predict(self, obs):
        ee = obs[:3]
        goal = obs[3:]
        direction = goal - ee
        norm = np.linalg.norm(direction)

        if norm > 1e-6:
            direction /= norm

        return direction