from feedback.feedback_queue import (
    feedback_queue,
)


class RewardShapingHandler:

    def __init__(self):

        self.last_buttons = [0, 0]

    def on_click(self, states):

        if (
            states.buttons[0] == 1
            and self.last_buttons[0] == 0
        ):
            feedback_queue.put(
                {"reward": 1.0}
            )

            print("GOOD")

        if (
            states.buttons[1] == 1
            and self.last_buttons[1] == 0
        ):
            feedback_queue.put(
                {"reward": -1.0}
            )

            print("BAD")

        self.last_buttons = list(
            states.buttons
        )