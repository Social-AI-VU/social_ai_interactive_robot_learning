import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from franka_reach.franka_reach_env import FrankaReachEnv

# some imports are in the code so that not everything is imported when the real robot, spacemouse etc. 
# is not connected, which would cause issues

MODEL_PATH = "franka_reach_sac"
USE_SIM = True
USE_WEB = True

def main():

    if USE_WEB:
        from feedback.web_reward import WebRewardSource
        reward_source = (
            WebRewardSource()
        )
    else:
        from sic_framework.devices.desktop import Desktop
        from feedback.spacemouse_reward import SpacemouseRewardSource
        desktop = Desktop()
        reward_source = (
            SpacemouseRewardSource(
                desktop
            )
        )

    reward_source.start()

    if USE_SIM:
        from robot.franka_sim import FrankaSim
        robot = FrankaSim()
    else:
        from robot.franka_real import FrankaReal # import here to avoid issues when the real robot is not connected
        robot = FrankaReal(goal=[0.6, 0.0, 0.4])
        

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