"""
Build any "online" environment by key as a ready gymnasium.Env, so it can be
driven by any learner (see train_online.py) through the same step()/reset()
interface - add an entry here once a new environments/<name>/franka_<name>_env.py
is ready to be driven online.

Every entry here is sim-only for now; see
docs/adding_environments_tasks_learners.md for how to add a real-robot variant.
"""

from environments.reach.franka_reach_env import FrankaReachEnv
from environments.stack_cups.franka_stack_cups_env import FrankaStackCupsEnv
from robot.franka_sim import FrankaSim

ENVIRONMENT_BUILDERS = {
    "reach_online": lambda human_reward_weight: FrankaReachEnv(
        robot=FrankaSim(env_id="PandaReach-v3"),
        human_reward_weight=human_reward_weight,
    ),
    "stack_cups": lambda human_reward_weight: FrankaStackCupsEnv(
        robot=FrankaSim(env_id="PandaStack-v3"),
        human_reward_weight=human_reward_weight,
    ),
}


def make_env(env_key, human_reward_weight=0.1):
    if env_key not in ENVIRONMENT_BUILDERS:
        raise ValueError(
            f"No online-track builder registered for '{env_key}'. "
            f"Available: {sorted(ENVIRONMENT_BUILDERS)}"
        )
    return ENVIRONMENT_BUILDERS[env_key](human_reward_weight)
