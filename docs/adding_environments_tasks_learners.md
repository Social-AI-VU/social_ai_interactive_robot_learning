# Adding a new environment, task, or learner

This repo has two independent workflows for teaching a robot something new. Both are
reachable from the same UI (`streamlit run ui/ui.py`), via the **Environment** picker
on the configuration page, but they drive completely different code underneath. Before
adding anything, figure out which one you're extending - they don't share code paths,
and forcing a fit between them will just cause confusion.

| | **Teleoperated demos → BC** | **Online interactive learning** |
|---|---|---|
| Workspace page | Record → Train → Execute → Log tabs | Live Training → Log tabs |
| Environment style | `robosuite.environments.manipulation.ManipulationEnv` subclass, `@register_env`-decorated | `gymnasium.Env` subclass wrapping a `robot/` object |
| Existing examples | `environments/push/push_env.py`, `environments/lift_nut/`, `environments/sandbox/`, `environments/reach/reach_env.py` | `environments/reach/franka_reach_env.py`, `environments/stack_cups/franka_stack_cups_env.py` |
| Learner | Fixed: robomimic BC, trained on collected demos | Any of `LEARNERS` in `ui/ui.py` (TAMER, SAC, ...) - **every online-track environment is combinable with every learner**; nothing checks whether a given pairing is sensible, that's left to whoever runs it |
| Behind the "Launch"/"Start" button | The UI launches `input_devices/collect_human_demonstrations.py` in a new terminal and manages a multi-step pipeline itself | The UI launches `train_online.py --environment <env> --learner <learner>` in a new terminal and otherwise gets out of the way - the whole experiment runs in that one script |

A word on terminology: in this codebase "environment" and "task" mostly mean the same
thing - one class under `environments/<name>/` defining an observation space, action
space, and reward/success condition. The one place "task" means something else is the
Streamlit UI's **task name**, which is just a free-text label for a data folder
(`data/<task>/`, `models/<task>/`) to organise recording sessions under a given
environment - creating one needs no code at all, just typing a name into the UI.

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
       "track": "teleop",
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
     `info["human_reward"]` (in addition to folding it into the blended `reward`,
     which is what makes the environment usable by a bootstrapping learner like
     SAC, not just a purely-greedy one like TAMER).
   - Computes `terminated` from whatever counts as task success.
3. If a robot wrapper for this task doesn't exist yet, check whether
   `robot/franka_sim.py`'s `FrankaSim` already covers it first - it's a generic
   panda-gym wrapper parameterised by `env_id`
   (`FrankaSim(env_id="PandaStack-v3")`, for instance), so a new panda-gym task
   usually doesn't need a new robot file at all. Only write a new one under
   `robot/` if the task needs something panda-gym's dict-obs shape can't express.
4. Register a builder for it in `environments/factory.py`'s `ENVIRONMENT_BUILDERS`:
   ```python
   "<name>": lambda human_reward_weight: Franka<Name>Env(
       robot=FrankaSim(env_id="<PandaGymEnvId>"),
       human_reward_weight=human_reward_weight,
   ),
   ```
   This one dict entry is what makes the environment usable by **every** learner in
   `train_online.py`'s `RUNNERS` (see "Adding a new learner" below) - there's no
   separate per-learner wiring to do.
5. Add an entry to `ENVIRONMENTS` in `ui/ui.py` so it shows up in the UI's Environment
   picker, with `"track": "online"` (no learner list needed - every online-track
   environment is automatically combinable with every registered learner):
   ```python
   "<name>": {
       "label": "<Display Name>",
       "track": "online",
       "robot": "Franka Panda (sim only)",   # or whatever's accurate
   },
   ```
   Use the **same key** here as in `environments/factory.py`'s `ENVIRONMENT_BUILDERS`
   (this is how `train_online.py --environment <name>` finds it). If that key would
   collide with an existing teleop-track entry - e.g. reach has both a robosuite
   `ReachTask` (teleop) and a `FrankaReachEnv` (online), which are different classes -
   pick a distinguishing key like `reach_online`; see that entry for the pattern.
6. A real-robot variant (paralleling `robot/franka_real.py`) is a natural next
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
(see `learners/tamer_learner.py`). SAC (via stable-baselines3's own `.learn()`) doesn't
go through this interface - its training loop is entirely internal to `.learn()` - so
`Learner` is currently only used by TAMER. That's fine: `train_online.py` doesn't
require a learner to subclass `Learner`, only that it's registered in `RUNNERS` with a
function that knows how to drive it.

To add a new learner:

1. If it needs its own step-by-step loop (not an existing library's `.learn()`),
   create `learners/<name>_learner.py`, subclassing `Learner`. Implement
   `select_action(obs) -> action` and `observe(obs, action, reward)` - `reward` here
   means whatever *your* learner should learn from, not necessarily the human
   feedback signal specifically (e.g. a corrective-feedback learner might instead
   take a corrected action as `**kwargs` and learn to imitate it).
2. Add a `run_<name>(env, output_path)` function to `train_online.py` that drives it
   to completion (see `run_tamer` and `run_sac` there for the two different shapes
   this can take - a manual step loop, or handing the whole env to a library).
3. Register it in `train_online.py`'s `RUNNERS` dict: `"<name>": run_<name>`.
4. Add an entry to `LEARNERS` in `ui/ui.py` (`label`, `description`, and
   `feedback_url` if your learner starts a feedback server) - it becomes selectable
   for **every** online-track environment immediately, no further wiring needed.

---

## Running the UI

```bash
pip install -r requirements.txt
streamlit run ui/ui.py
```

This opens the configuration page. It works the same way regardless of which track
you're using:

1. **Environment** - pick from the dropdown (populated from `ENVIRONMENTS` in
   `ui/ui.py`). This alone decides the rest of the page: picking a `"track": "online"`
   environment (Stack Cups, Reach (Online)) swaps the Robot Mode / Input controls for
   a **Learner** dropdown, listing every learner in `LEARNERS` - any of them can be
   paired with any online-track environment. A `"track": "teleop"` environment (Reach)
   keeps the original Robot Mode / Input / Teaching Signal controls and its fixed BC
   pipeline.
2. Fill in whatever the page asks for (a Learner, for online-track environments;
   Robot Mode and - if Real Robot - a robot connection, for teleop-track ones).
3. **Task** - type a name and click **Create task** (or pick an existing one). This
   is just a folder name (`data/<task>/`, `models/<task>/`) to keep a session's
   demos/checkpoints separate from other sessions.
4. Click **Continue to Workspace →** (disabled until the required fields above are
   filled in).

**On the workspace page:**

- **Teleop-track environments** (Reach, Push, ...): use the Record tab to collect
  demonstrations, Train to run BC, Execute to try the trained policy - as before.
- **Online-track environments** (Stack Cups, Reach (Online), with any learner): the
  **Live Training** tab has one **Launch Training** button. Clicking it runs
  `train_online.py --environment <env> --learner <learner> --output models/<task>/<env>_<learner>.pt`
  in a new terminal window. Everything after that happens in the launched terminal,
  not the Streamlit page - the UI's job was just to get the right script running with
  the right arguments. If the learner's config has a `feedback_url` (TAMER's and
  SAC's both do, since both can use live feedback), the page links to it - that's
  where you actually give feedback while the robot moves. Close the terminal (or
  Ctrl+C in it) to stop; there's no stop button in the UI yet.

---

## Worked example: Stack Cups + TAMER

Files added for this example:

| File | What it is |
|---|---|
| `learners/learner.py` | The `Learner` interface described above |
| `learners/tamer_learner.py` | `TamerLearner`: an online-trained reward model `H(s,a)` plus greedy (random-shooting) action selection. See the file's docstring for the two simplifications it makes vs. the published TAMER algorithm |
| `robot/franka_sim.py` | Generalised to take `env_id` (was hardcoded to `PandaReach-v3`); `run.py`'s existing usage is unaffected since `env_id` defaults to `PandaReach-v3` |
| `environments/stack_cups/franka_stack_cups_env.py` | `FrankaStackCupsEnv`, same shape as `FrankaReachEnv` but reward/success are based on distance between the two objects' achieved vs. desired positions |
| `environments/factory.py` | `ENVIRONMENT_BUILDERS`: registers `FrankaReachEnv` (as `reach_online`) and `FrankaStackCupsEnv` (as `stack_cups`) as generic, learner-agnostic `gymnasium.Env`s |
| `train_online.py` | The generic training entry point (`--environment`, `--learner`, `--output`) - `run_tamer()` and `run_sac()` are the two learners wired up so far |
| `ui/ui.py`: `ENVIRONMENTS["stack_cups"]`, `ENVIRONMENTS["reach_online"]`, `LEARNERS` | Registers both environments and both learners with the UI - fully cross-combinable (see "Running the UI" above) |
| `ui/ui.py`: `launch_online_training()`, `_show_online_workspace()` | The generic UI-side machinery any `"track": "online"` environment/learner pair uses |

Because environments and learners are independently registered, this worked example
is really four combinations, not one: **Stack Cups + TAMER**, **Stack Cups + SAC**,
**Reach (Online) + TAMER**, and **Reach (Online) + SAC** are all launchable today,
from the UI or the command line. Nothing here has been tuned to make all four work
*well* - TAMER on Reach (Online) or SAC on Stack Cups are exactly the kind of
"maybe this makes sense, maybe it doesn't" combinations that are intentionally not
blocked.

**Known simplification:** panda-gym doesn't ship a cup asset, so Stack Cups reuses its
`PandaStack-v3` task, which stacks two cubes. The reward/termination logic only looks
at object positions, so it's agnostic to the object's shape - if the visual/contact
geometry needs to actually be cups, that means writing a robot wrapper around a custom
MuJoCo or robosuite cup asset instead of `PandaStack-v3`; nothing else here would need
to change.

**To run it standalone** (no UI):

```bash
pip install -r requirements.txt   # pulls in panda-gym, needed for PandaStack-v3
python train_online.py --environment stack_cups --learner tamer --output stack_cups_tamer.pt
```

Swap `--environment reach_online` and/or `--learner sac` for any of the other three
combinations.

**To run it through the UI:** follow "Running the UI" above, pick **Stack Cups** (or
**Reach (Online)**) as the Environment, **TAMER** (or **SAC (RL)**) as the Learner,
create/pick a task, Continue to Workspace, then **Launch Training**.

Either way: a simulation window opens, and `http://localhost:5000` serves the
GOOD/BAD feedback buttons (`rewards/web_reward.py`, already used by `run.py`). For
TAMER: click GOOD when the robot does something toward the goal and BAD when it
doesn't - each click trains `TamerLearner`'s reward model on the last few steps
(`credit_window`, default 10), and the robot immediately starts picking actions to
maximise it, with no separate "training phase". For SAC: feedback still reaches the
environment the same way, but SAC only updates its policy in batches as its replay
buffer fills up, so behaviour changes far more slowly and after far more steps.
