# rewards/spacemouse_reward.py

from rewards.feedback_queue import feedback_queue
from rewards.reward_source import RewardSource


class SpacemouseRewardSource(RewardSource):

    def __init__(self, desktop):

        self.desktop = desktop
        self.last_buttons = [0, 0]

    def on_click(self, states):

        if states.buttons[0] and not self.last_buttons[0]:

            feedback_queue.put(
                {"reward": 1.0}
            )

            print("GOOD")

        if states.buttons[1] and not self.last_buttons[1]:

            feedback_queue.put(
                {"reward": -1.0}
            )

            print("BAD")

        self.last_buttons = list(
            states.buttons
        )

    def start(self):

        self.desktop.spacemouse.register_callback(
            self.on_click
        )