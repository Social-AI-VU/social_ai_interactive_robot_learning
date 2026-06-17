import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import (
    Monitor,
)

from sic_framework.devices.desktop import (
    Desktop,
)

from robot.franka_rl_interface import (
    FrankaRLInterface,
)

from feedback.spacemouse_reward import (
    RewardShapingHandler,
)

from franka_reach.franka_reach_env import (
    FrankaReachEnv,
)


MODEL_PATH = "franka_reach_sac"


def main():

    desktop = Desktop()

    reward_handler = (
        RewardShapingHandler()
    )

    desktop.spacemouse.register_callback(
        reward_handler.on_click
    )

    print(
        "\nSpaceMouse reward shaping active\n"
        "Left button = GOOD\n"
        "Right button = BAD\n"
    )

    robot = FrankaRLInterface()

    env = FrankaReachEnv(
        robot=robot,
        human_reward_weight=0.1,
    )

    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=100,
    )

    env = Monitor(env)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        device="auto",
    )

    try:

        model.learn(
            total_timesteps=500_000,
            progress_bar=True,
        )

    except KeyboardInterrupt:

        print("\nTraining interrupted.")

    finally:

        model.save(MODEL_PATH)

        print(
            f"\nModel saved to "
            f"{MODEL_PATH}"
        )


if __name__ == "__main__":
    main()