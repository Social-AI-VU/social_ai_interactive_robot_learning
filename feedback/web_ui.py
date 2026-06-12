from flask import Flask, redirect
from threading import Thread

from feedback.feedback_queue import feedback_queue

app = Flask(__name__)


@app.route("/")
def home():

    return """
    <html>
    <head>
        <title>Franka Feedback</title>
    </head>

    <body>
        <h1>Human Feedback</h1>

        <form action="/good" method="post">
            <button style="width:200px;height:80px;font-size:24px">
                GOOD (+1)
            </button>
        </form>

        <br>

        <form action="/bad" method="post">
            <button style="width:200px;height:80px;font-size:24px">
                BAD (-1)
            </button>
        </form>

        <br>

        <form action="/left" method="post">
            <button style="width:200px;height:80px">
                LEFT
            </button>
        </form>

        <form action="/right" method="post">
            <button style="width:200px;height:80px">
                RIGHT
            </button>
        </form>

        <form action="/forward" method="post">
            <button style="width:200px;height:80px">
                FORWARD
            </button>
        </form>

        <form action="/backward" method="post">
            <button style="width:200px;height:80px">
                BACKWARD
            </button>
        </form>
    </body>
    </html>
    """


@app.route("/good", methods=["POST"])
def good():

    feedback_queue.put({"reward": 1})
    return redirect("/")


@app.route("/bad", methods=["POST"])
def bad():

    feedback_queue.put({"reward": -1})
    return redirect("/")


@app.route("/left", methods=["POST"])
def left():

    feedback_queue.put(
        {"correction": [-0.5, 0.0, 0.0]}
    )

    return redirect("/")


@app.route("/right", methods=["POST"])
def right():

    feedback_queue.put(
        {"correction": [0.5, 0.0, 0.0]}
    )

    return redirect("/")


@app.route("/forward", methods=["POST"])
def forward():

    feedback_queue.put(
        {"correction": [0.0, 0.5, 0.0]}
    )

    return redirect("/")


@app.route("/backward", methods=["POST"])
def backward():

    feedback_queue.put(
        {"correction": [0.0, -0.5, 0.0]}
    )

    return redirect("/")


def launch_ui():

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )


def start_ui():

    thread = Thread(
        target=launch_ui,
        daemon=True
    )

    thread.start()

    return thread