from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from robot.franka_mock import FrankaInterface
from franka_reach.franka_reach_env import FrankaReachEnv
from feedback.web_ui import start_ui


MODEL_PATH = "models/franka_reach_sac"


def main():

    # Start the web UI for Good / Bad button presses
    start_ui()

    print(
        "\nOpen browser:\n"
        "http://localhost:5000\n"
    )

    # Create robot and environment
    robot = FrankaInterface()

    env = FrankaReachEnv(robot)

    # Prevent episodes from running forever
    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=100
    )

    # Add episode statistics logging
    env = Monitor(env)

    # Create a new SAC policy
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        device="auto",
    )

    try:

        print("\nStarting training...")
        print("Use the web UI to provide GOOD / BAD feedback.\n")

        model.learn(
            total_timesteps=500_000,
            progress_bar=True,
        )

    except KeyboardInterrupt:

        print("\nTraining interrupted by user.")

    finally:

        print(f"\nSaving model to {MODEL_PATH}")
        model.save(MODEL_PATH)

        env.close()


if __name__ == "__main__":
    main()