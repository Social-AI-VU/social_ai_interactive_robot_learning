import gymnasium as gym

from environments.stack_cups.franka_stack_cups_env import FrankaStackCupsEnv
from learners.tamer_learner import TamerLearner
from robot.franka_sim import FrankaSim
from rewards.web_reward import WebRewardSource

# TAMER learns entirely from live human feedback - there's no notion of
# "simulation vs. real" reward here the way run.py has for SAC, so this
# script is sim-only for now. See docs/adding_environments_tasks_learners.md
# for how to add a real-robot variant.

MODEL_PATH = "stack_cups_tamer.pt"
MAX_EPISODES = 200
MAX_STEPS_PER_EPISODE = 150


def main():

    reward_source = WebRewardSource()
    reward_source.start()
    print("Open http://localhost:5000 and click GOOD/BAD while the robot moves.")

    robot = FrankaSim(env_id="PandaStack-v3")

    env = FrankaStackCupsEnv(
        robot=robot,
        human_reward_weight=0.1,
    )

    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=MAX_STEPS_PER_EPISODE,
    )

    learner = TamerLearner(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )

    try:

        for episode in range(MAX_EPISODES):

            obs, _ = env.reset()
            learner.reset()
            episode_human_reward = 0.0

            for _ in range(MAX_STEPS_PER_EPISODE):

                action = learner.select_action(obs)
                next_obs, _, terminated, truncated, info = env.step(action)

                human_reward = info["human_reward"]
                learner.observe(obs, action, human_reward)
                episode_human_reward += human_reward

                obs = next_obs

                if terminated or truncated:
                    break

            print(
                f"Episode {episode}: "
                f"total human reward = {episode_human_reward:.1f}"
            )

    except KeyboardInterrupt:

        print("\nTraining interrupted.")

    finally:

        learner.save(MODEL_PATH)

        print(
            f"\nModel saved to "
            f"{MODEL_PATH}"
        )


if __name__ == "__main__":
    main()
