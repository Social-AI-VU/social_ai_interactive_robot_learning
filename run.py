from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from robot.franka_mock import FrankaInterface
from franka_reach.franka_reach_env import FrankaReachEnv
from feedback.web_ui import start_ui


def main():

    start_ui()

    print(
        "\nOpen browser:\n"
        "http://localhost:5000\n"
    )

    robot = FrankaInterface()

    env = Monitor(
        gym.wrappers.TimeLimit(
            FrankaReachEnv(robot),
            max_episode_steps=100
        )
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
    )

    model.learn(
        total_timesteps=500_000
    )

    model.save(
        "models/franka_reach_sac"
    )


if __name__ == "__main__":
    main()