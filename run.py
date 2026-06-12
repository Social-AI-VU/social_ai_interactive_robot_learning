import time

from robot.franka_mock import FrankaInterface
from franka_reach.franka_reach_env import FrankaReachEnv
from franka_reach.franka_reach_policy import ReachPolicy

from feedback.web_ui import start_ui


def main():

    start_ui()

    print(
        "\nOpen browser:\n"
        "http://localhost:5000\n"
    )

    robot = FrankaInterface()

    env = FrankaReachEnv(robot)

    policy = ReachPolicy()

    obs, _ = env.reset()

    done = False

    while not done:

        action = policy.predict(obs)

        obs, reward, done, _, info = env.step(
            action
        )

        if info["correction"] is not None:

            action += info["correction"]

            print(
                "Human correction:",
                info["correction"]
            )

        print(
            f"distance={info['distance']:.3f} "
            f"human_reward={info['human_reward']}"
        )

        time.sleep(0.2)

    print("Goal reached")


if __name__ == "__main__":
    main()