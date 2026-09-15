# Adding a new environment, task, or learner

This repo currently has two independent workflows for teaching a robot something new.
Before adding anything, figure out which one you're extending - they don't share code
paths, and forcing a fit between them will just cause confusion.

| | **Teleoperated demos → BC** | **Online interactive learning** |
|---|---|---|
| How it's driven | `streamlit run ui/ui.py` (Record → Train → Execute tabs) | A standalone script, e.g. `run.py`, `train_stack_cups_tamer.py` |
| Environment style | `robosuite.environments.manipulation.ManipulationEnv` subclass, `@register_env`-decorated | `gymnasium.Env` subclass wrapping a `robot/` object |
| Existing examples | `environments/push/push_env.py`, `environments/lift_nut/`, `environments/sandbox/`, `environments/reach/reach_env.py` | `environments/reach/franka_reach_env.py`, `environments/stack_cups/franka_stack_cups_env.py` |
| How a human teaches it | Teleoperates full demonstrations (SpaceMouse), which are merged and used to train a policy with robomimic | Rates the robot's behaviour live (a `rewards/RewardSource`), which a `learners/Learner` (or SAC) learns from step by step |

A word on terminology: in this codebase "environment" and "task" mostly mean the same
thing - one class under `environments/<name>/` defining an observation space, action
space, and reward/success condition. The one place "task" means something else is the
Streamlit UI's **task name**, which is just a free-text label for a data folder
(`data/<task>/`) to organise recording sessions under a given environment - creating one
needs no code at all, just typing a name into the UI.

---

## Adding a new environment / task

### If it belongs on the teleoperated-demos track

Follow the shape of `environments/push/push_env.py`:

1. Create `environments/<name>/<name>_env.py`.
2. Define a class subclassing `robosuite.environments.manipulation.manipulation_env.ManipulationEnv`,
   decorated with `@register_env` (from `robosuite.environments.base`).
3. Implement (see `push_env.py` for a complete reference): `__init__` (arena/table
   setup, goal parameters), `_load_model` (build the scene), `_setup_references`,
   `_reset_internal`, `reward`, `_check_success`, `_setup_observables`.
4. Register it for **sim demo collection**: add `from environments.<name> import <name>_env  # noqa: F401`
   to `environments/registry.py`. This is the one central place a new
   robosuite task needs to be imported so `robosuite.make("<YourTask>", ...)`
   can find it - see the note at the top of that file for why it isn't just
   `environments/__init__.py`.
5. Add an entry to `ENVIRONMENTS` in `ui/ui.py` so it shows up in the UI's Environment
   picker:
   ```python
   "<name>": {
       "label": "<Display Name>",
       "sim_env_name": "<YourRegisteredClassName>",
       "robot": "Franka Panda",
   },
   ```
   That's it - the Record/Train/Execute tabs are already generic over whatever
   environment is selected.

### If it belongs on the online-interactive-learning track

Follow the shape of `environments/reach/franka_reach_env.py` (and
`environments/stack_cups/franka_stack_cups_env.py` for a second worked example):

1. Create `environments/<name>/franka_<name>_env.py`.
2. Define a `gymnasium.Env` subclass that:
   - Takes a `robot` object in `__init__` (something with `reset()`, an
     `apply_action(action)` that steps the sim/hardware, and whatever
     state-reading methods the reward needs - `get_ee_position()`/`get_goal()`
     for reach, `get_state()`/`get_achieved_goal()`/`get_goal()` for stacking).
   - In `step()`, drains `rewards.feedback_queue.feedback_queue` for any human
     feedback that arrived since the last step, and returns it in
     `info["human_reward"]` (in addition to folding it into the blended `reward`
     if you want the env usable with RL too - see below).
   - Computes `terminated` from whatever counts as task success.
3. If a robot wrapper for this task doesn't exist yet, check whether
   `robot/franka_sim.py`'s `FrankaSim` already covers it first - it's a generic
   panda-gym wrapper parameterised by `env_id`
   (`FrankaSim(env_id="PandaStack-v3")`, for instance), so a new panda-gym task
   usually doesn't need a new robot file at all. Only write a new one under
   `robot/` if the task needs something panda-gym's dict-obs shape can't express.
4. This track has **no central registry** - a script drives the environment
   directly (`from environments.<name>.franka_<name>_env import ...`), same as
   `run.py` and `train_stack_cups_tamer.py` do. Nothing in `ui/ui.py` needs to
   change; the Streamlit UI doesn't cover this track at all right now.
5. A real-robot variant (paralleling `robot/franka_real.py`) is a natural next
   step once the sim version works, but isn't required to get started.

---

## Adding a new learner

`learners/learner.py` defines the interface:

```python
class Learner(ABC):
    def select_action(self, obs): ...      # required
    def observe(self, obs, action, reward, **kwargs): ...  # required
    def reset(self): ...                   # optional, called each episode
    def save(self, path): ...              # optional
    def load(self, path): ...              # optional
```

This is for algorithms that need their own step-by-step training loop, like TAMER
(see `learners/tamer_learner.py`). `run.py`'s SAC training doesn't go through this
interface - stable-baselines3's own `.learn()` already is that loop - so `Learner` is
currently only used by the TAMER example. Wrapping SAC in the same interface for
consistency is a reasonable follow-up, not something this example does.

To add a new learner:

1. Create `learners/<name>_learner.py`, subclassing `Learner`.
2. Implement `select_action(obs) -> action` and `observe(obs, action, reward)`.
   `reward` here means whatever *your* learner should learn from - it doesn't have
   to be the human feedback signal specifically (e.g. a corrective-feedback learner
   might instead take a corrected action as `**kwargs` and learn to imitate it).
3. Write a small training-loop script (see `train_stack_cups_tamer.py`) that:
   - builds the environment + robot + a `rewards/RewardSource` for capturing
     human input,
   - instantiates your learner,
   - loops `action = learner.select_action(obs)` → `env.step(action)` →
     `learner.observe(...)` → repeat.

---

## Worked example: Stack Cups + TAMER

Files added for this example:

| File | What it is |
|---|---|
| `learners/learner.py` | The `Learner` interface described above |
| `learners/tamer_learner.py` | `TamerLearner`: an online-trained reward model `H(s,a)` plus greedy (random-shooting) action selection. See the file's docstring for the two simplifications it makes vs. the published TAMER algorithm |
| `robot/franka_sim.py` | Generalised to take `env_id` (was hardcoded to `PandaReach-v3`); `run.py`'s existing usage is unaffected since `env_id` defaults to `PandaReach-v3` |
| `environments/stack_cups/franka_stack_cups_env.py` | `FrankaStackCupsEnv`, same shape as `FrankaReachEnv` but reward/success are based on distance between the two objects' achieved vs. desired positions |
| `train_stack_cups_tamer.py` | The runnable training loop, mirroring `run.py` |

**Known simplification:** panda-gym doesn't ship a cup asset, so this reuses its
`PandaStack-v3` task, which stacks two cubes. The reward/termination logic only looks
at object positions, so it's agnostic to the object's shape - if the visual/contact
geometry needs to actually be cups, that means writing a robot wrapper around a custom
MuJoCo or robosuite cup asset instead of `PandaStack-v3`; nothing else here would need
to change.

**To run it:**

```bash
pip install -r requirements.txt   # pulls in panda-gym, needed for PandaStack-v3
python train_stack_cups_tamer.py
```

A simulation window opens, and `http://localhost:5000` serves the GOOD/BAD feedback
buttons (`rewards/web_reward.py`, already used by `run.py`). Click GOOD when the robot
does something toward stacking the cubes and BAD when it doesn't - each click trains
`TamerLearner`'s reward model on the last few steps (`credit_window`, default 10), and
the robot immediately starts picking actions to maximise it. There's no separate
"training phase" - action selection and learning happen in the same loop, so behaviour
should visibly shift within the first few dozen clicks.
