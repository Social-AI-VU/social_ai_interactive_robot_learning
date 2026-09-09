from flask import Flask

from rewards.feedback_queue import feedback_queue
from rewards.reward_source import RewardSource


HTML = """
<html>
<head>
    <title>Human Feedback</title>
</head>
<body>

<h1>Reward Shaping</h1>

<form action="/good">
    <button
        style="width:300px;height:120px;font-size:32px;">
        GOOD
    </button>
</form>

<br>

<form action="/bad">
    <button
        style="width:300px;height:120px;font-size:32px;">
        BAD
    </button>
</form>

</body>
</html>
"""


class WebRewardSource(RewardSource):

    def __init__(self):

        self.app = Flask(__name__)

        self.app.add_url_rule(
            "/",
            "index",
            self.index,
        )

        self.app.add_url_rule(
            "/good",
            "good",
            self.good,
        )

        self.app.add_url_rule(
            "/bad",
            "bad",
            self.bad,
        )

    def index(self):

        return HTML

    def good(self):

        feedback_queue.put(
            {"reward": 1.0}
        )

        print("GOOD")

        return HTML

    def bad(self):

        feedback_queue.put(
            {"reward": -1.0}
        )

        print("BAD")

        return HTML

    def start(self):

        import threading

        threading.Thread(
            target=lambda: self.app.run(
                host="0.0.0.0",
                port=5000,
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
        ).start()