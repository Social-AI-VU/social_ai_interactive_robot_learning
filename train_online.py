"""
Generic online-learning entry point: pick any registered environment (see
environments/factory.py) and any registered learner (see RUNNERS below) and
run them together. Nothing here checks whether a combination makes sense -
e.g. TAMER on an environment nobody's tuned it for is allowed to run and
just do a bad job. That judgment is left to whoever runs it.

Run: python train_online.py --environment stack_cups --learner tamer --output out.pt
"""

import argparse

import gymnasium as gym

from environments.factory import make_env
from rewards.web_reward import WebRewardSource

MAX_EPISODES = 200
MAX_STEPS_PER_EPISODE = 150


def run_tamer(env, output_path):

    from learners.tamer_learner import TamerLearner

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

        learner.save(output_path)

        print(
            f"\nModel saved to "
            f"{output_path}"
        )


def run_sac(env, output_path):

    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=MAX_STEPS_PER_EPISODE,
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

        model.save(output_path)

        print(
            f"\nModel saved to "
            f"{output_path}"
        )


RUNNERS = {
    "tamer": run_tamer,
    "sac": run_sac,
}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        required=True,
        help="Key from environments/factory.py's ENVIRONMENT_BUILDERS.",
    )
    parser.add_argument(
        "--learner",
        required=True,
        choices=sorted(RUNNERS),
    )
    parser.add_argument("--output", default="model.pt")
    parser.add_argument("--human-reward-weight", type=float, default=0.1)
    args = parser.parse_args()

    reward_source = WebRewardSource()
    reward_source.start()
    print("Open http://localhost:5000 to give GOOD/BAD feedback while the robot moves.")

    env = make_env(args.environment, human_reward_weight=args.human_reward_weight)

    RUNNERS[args.learner](env, args.output)


if __name__ == "__main__":
    main()
