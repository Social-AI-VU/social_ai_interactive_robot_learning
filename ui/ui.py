"""
Robot LfD UI - Streamlit interface for demo recording, training, and policy execution.
Run with: streamlit run ui/ui_two_step.py (from the repo root)
"""

import streamlit as st
import subprocess
import sys
import threading
import queue
import time
import os
import glob
import json
from pathlib import Path

# Log buffer - background threads write here, main thread drains it
_log_queue: queue.Queue = queue.Queue()

# Config
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # ui/ scripts are launched with only ui/ on sys.path, so imports of our
    # own packages (environments, robot, rewards, scripts, learning) need
    # the repo root added explicitly.
    sys.path.insert(0, str(REPO_ROOT))

ROBOT_IP      = "172.16.0.2"
DEMOS_DIR     = REPO_ROOT / "data"
MODELS_DIR    = REPO_ROOT / "models"
CONFIG_DIR    = REPO_ROOT / "config"
ASSETS_DIR    = Path(__file__).parent / "assets"
RECORDER_PATH = REPO_ROOT / "scripts" / "demo_recorder.py"
EXECUTE_PATH  = REPO_ROOT / "learning" / "execute_policy.py"

# Environments wired up end-to-end for this UI. Add an entry here once a new
# task under environments/ is ready to be driven from this UI - it will then
# show up in the Environment picker below. See
# docs/adding_environments_tasks_learners.md for the full checklist.
#
# "track" selects which workspace page this environment gets:
#   "teleop"  - teleoperated demos -> BC (the Record/Train/Execute/Log tabs
#               below). Needs "sim_env_name" (the robosuite env registered
#               under environments/registry.py).
#   "online"  - any LEARNERS entry (below) picks actions live from human
#               feedback, run generically by train_online.py. Needs no
#               per-environment learner list - every online-track
#               environment is combinable with every registered learner
#               (see environments/factory.py); it's up to whoever runs it to
#               judge whether a given combination makes sense.
ENVIRONMENTS = {
    "reach": {
        "label": "Reach",
        "track": "teleop",
        "sim_env_name": "ReachTask",  # robosuite env registered in environments/registry.py
        "robot": "Franka Panda",
    },
    "reach_online": {
        # A separate entry from "reach" above since it's a different
        # environment class (FrankaReachEnv, not the robosuite ReachTask) -
        # see docs/adding_environments_tasks_learners.md for why the two
        # tracks don't share one. This key must match an entry in
        # environments/factory.py's ENVIRONMENT_BUILDERS.
        "label": "Reach (Online)",
        "track": "online",
        "robot": "Franka Panda (sim only)",
    },
    "stack_cups": {
        "label": "Stack Cups",
        "track": "online",
        "robot": "Franka Panda (sim only)",
    },
}

# Every learner here is combinable with every "online" track environment
# above, run via train_online.py --environment <env> --learner <learner>.
LEARNERS = {
    "tamer": {
        "label": "TAMER",
        "description": (
            "Learns online from live GOOD/BAD feedback only - no environment "
            "reward is used. Behaviour starts shifting after the first few clicks."
        ),
        "feedback_url": "http://localhost:5000",
    },
    "sac": {
        "label": "SAC (RL)",
        "description": (
            "Trains a Soft Actor-Critic policy (stable-baselines3) against the "
            "environment's own reward, optionally shaped by live human feedback. "
            "Bootstraps across steps like standard RL, unlike TAMER's greedy "
            "policy - expect it to need far more steps before behaviour changes."
        ),
        "feedback_url": "http://localhost:5000",
    },
}

DEMOS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

SIM_PREVIEWS = {
    "Reach":  ASSETS_DIR / "sim_reach.png",
    "Push":   ASSETS_DIR / "sim_push.png",
    "Lift":   ASSETS_DIR / "sim_lift.png",
    "Nut":   ASSETS_DIR / "sim_lift_nut.png",
    "Stack":  ASSETS_DIR / "sim_stack.png",
    "Sandbox":  ASSETS_DIR / "sim_sandbox.png",
}

# Session state

def init_state():
    defaults = {
        "robot_connected":     False,
        "recording":           False,
        "training":            False,
        "executing":           False,
        "current_task":        None,
        "environment":         next(iter(ENVIRONMENTS)),
        "learner":             None,
        "robot_ip":            ROBOT_IP,
        "mode":                "Simulation",   # Real Robot or Simulation
        "input_mode":          "Joint",
        "ui_page":              "configuration",
        "teaching_signal":     "evaluative",
        "evaluative_method":   "deep tamer",
        "log":                 [],
        "recorder":            None,
        "sim_collecting":      False,   # True while collection terminal is open
        "sim_last_demo_path":  None,    # path to most recent demo.hdf5
        "sim_processing":      False,   # True while post-processing runs
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# Logging

def log(msg: str):
    """Log a message. Call from background threads."""
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    try:
        st.session_state.log.append(entry)
        if len(st.session_state.log) > 100:
            st.session_state.log = st.session_state.log[-100:]
    except Exception:
        _log_queue.put(entry)

def drain_log_queue():
    """Drain queued log messages into session state. Call at the top of main()."""
    while not _log_queue.empty():
        try:
            entry = _log_queue.get_nowait()
            st.session_state.log.append(entry)
        except queue.Empty:
            break
    if len(st.session_state.log) > 100:
        st.session_state.log = st.session_state.log[-100:]

# Helpers
def get_tasks():
    if not DEMOS_DIR.exists():
        return []
    return sorted([d.name for d in DEMOS_DIR.iterdir() if d.is_dir()])

def get_demo_count(task: str) -> int:
    import h5py
    total = 0

    # Real robot demos - demos.hdf5 directly in task folder
    real_path = DEMOS_DIR / task / "demos.hdf5"
    if real_path.exists():
        try:
            with h5py.File(real_path, "r") as f:
                total += len(f["data"].keys())
        except Exception:
            pass

    # Sim demos - demo.hdf5 inside timestamped subdirectories
    sim_pattern = str(DEMOS_DIR / task / "**" / "demo.hdf5")
    for sim_demo in glob.glob(sim_pattern, recursive=True):
        try:
            with h5py.File(sim_demo, "r") as f:
                total += len(f["data"].keys())
        except Exception:
            pass

    return total

def get_models(task: str):
    pattern = str(MODELS_DIR / task / "**" / "*.pth")
    return sorted(glob.glob(pattern, recursive=True))


def find_latest_checkpoint(task: str) -> Path | None:
    """Return path to last.pth for the most recently modified training run, or None."""
    pattern = str(MODELS_DIR / task / "**" / "last.pth")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    return Path(max(matches, key=os.path.getmtime))


def get_epoch_from_checkpoint(ckpt_path: Path) -> int | None:
    """Read the last completed epoch number from a robomimic checkpoint."""
    try:
        import torch as _torch
        ckpt = _torch.load(str(ckpt_path), map_location="cpu")
        return int(ckpt["variable_state"]["epoch"])
    except Exception:
        return None

def run_in_thread(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()

def get_environment_config():
    """Return the registry entry for the environment chosen on the configuration page."""
    return ENVIRONMENTS[st.session_state.environment]


BASE_CONFIG_PATH = REPO_ROOT / "learning" / "train_bc_rnn.json"


def generate_train_config(task_name: str, dataset_path: Path,
                          n_epochs: int, batch_size: int,
                          resume_from_epoch: int | None = None) -> Path:
    """
    Read the base train_bc_rnn.json, patch the fields that change per run,
    and write the result to config/<task_name>/bc_rnn.json.

    If resume_from_epoch is provided, num_epochs is set to
    resume_from_epoch + n_epochs so --resume continues for exactly
    n_epochs more steps.

    Returns the path to the generated config.
    """
    import json as _json

    with open(BASE_CONFIG_PATH, "r") as f:
        cfg = _json.load(f)

    output_dir = str((MODELS_DIR / task_name).resolve())

    cfg["train"]["data"]       = [{"path": str(dataset_path.resolve())}]
    cfg["train"]["output_dir"] = output_dir
    cfg["train"]["batch_size"] = int(batch_size)

    if resume_from_epoch is not None:
        cfg["train"]["num_epochs"] = int(resume_from_epoch) + int(n_epochs)
    else:
        cfg["train"]["num_epochs"] = int(n_epochs)

    out_path = CONFIG_DIR / task_name / "bc_rnn.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        _json.dump(cfg, f, indent=4)

    return out_path


def find_latest_demo(task_name: str):
    """
    Return Path to the most recently modified demo.hdf5 inside data/{task_name}/.
    Excludes the merged folder.
    Returns None if nothing found.
    """
    pattern = str(DEMOS_DIR / task_name / "**" / "demo.hdf5")
    matches = [
        p for p in glob.glob(pattern, recursive=True)
        if "merged" not in Path(p).parts
    ]
    if not matches:
        return None
    return Path(max(matches, key=os.path.getmtime))


def find_all_hdf5(task_name: str) -> list[Path]:
    """
    Return all .hdf5 files for a task (obs.hdf5, merged files, etc).
    Sorted by modification time, newest first.
    """
    pattern = str(DEMOS_DIR / task_name / "**" / "*.hdf5")
    matches = glob.glob(pattern, recursive=True)
    return sorted([Path(p) for p in matches], key=os.path.getmtime, reverse=True)


def get_merged_path(task_name: str) -> Path:
    """Return the canonical path for the merged demo file."""
    return DEMOS_DIR / task_name / "merged" / "merged.hdf5"


def launch_sim_collection(task_name: str, env_name: str):
    """
    Open a new terminal window running our collect_human_demonstrations script.
    Run as `-m` from the repo root so `environments` is importable and the
    custom robosuite task (e.g. ReachTask) is registered before use.
    Returns the Popen object for the terminal.
    """
    output_dir = str((DEMOS_DIR / task_name).resolve())
    python = sys.executable
    cmd_inner = (
        f"cd {REPO_ROOT} && "
        f"{python} -m input_devices.collect_human_demonstrations "
        f"--environment {env_name} "
        f"--robots Panda "
        f"--device spacemouse "
        f"--directory {output_dir}; "
        f"echo 'Collection done - press Enter to close'; read"
    )
    # Try gnome-terminal first, fall back to xterm
    try:
        proc = subprocess.Popen(
            ["gnome-terminal", "--", "bash", "-c", cmd_inner],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        proc = subprocess.Popen(
            ["xterm", "-e", f"bash -c '{cmd_inner}'"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return proc


def online_output_path(task_name: str, env_key: str, learner_key: str) -> Path:
    """Where train_online.py saves the model for a given environment+learner+task."""
    return MODELS_DIR / task_name / f"{env_key}_{learner_key}.pt"


def launch_online_training(task_name: str, env_key: str, learner_key: str):
    """
    Launch train_online.py --environment <env_key> --learner <learner_key> in
    a new terminal - same pattern as launch_sim_collection() above, just for
    a different script. Any registered environment/learner pair can be
    launched this way; train_online.py doesn't check whether the combination
    makes sense. The UI doesn't manage the process after this: close the
    terminal (or Ctrl+C in it) to stop training. Returns the Popen object for
    the terminal.
    """
    script_path = REPO_ROOT / "train_online.py"
    output_path = online_output_path(task_name, env_key, learner_key).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    cmd_inner = (
        f"cd {REPO_ROOT} && "
        f"{python} {script_path} "
        f"--environment {env_key} --learner {learner_key} --output {output_path}; "
        f"echo 'Training stopped - press Enter to close'; read"
    )
    try:
        proc = subprocess.Popen(
            ["gnome-terminal", "--", "bash", "-c", cmd_inner],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        proc = subprocess.Popen(
            ["xterm", "-e", f"bash -c '{cmd_inner}'"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return proc


def process_sim_demos(demo_path: Path):
    """
    Run dataset_states_to_obs then split_train_val in a new terminal window.
    Both commands run sequentially in one terminal so the user can watch progress.
    """
    obs_path = demo_path.parent / "obs.hdf5"

    cmd1 = [
        "python", "-m", "robomimic.scripts.dataset_states_to_obs",
        "--dataset", str(demo_path),
        "--output_name", "obs.hdf5",
        "--done_mode", "0",
    ]
    cmd2 = [
        "python", "-m", "robomimic.scripts.split_train_val",
        "--dataset", str(obs_path),
        "--ratio", "0.1",
    ]

    cmd1_str = " ".join(cmd1)
    cmd2_str = " ".join(cmd2)
    bash_cmd = (
        f"{cmd1_str} && {cmd2_str} && "
        f"echo 'Press Enter to close.'; read"
    )

    subprocess.Popen(
        ["gnome-terminal", "--", "bash", "-c", bash_cmd],
        cwd=str(REPO_ROOT)
    )
    log("Processing started in a new terminal window.")
    log("When done, click Refresh to check if obs.hdf5 is ready.")
    st.session_state.sim_processing = False
    
def sim_preview(task_name: str):
    """Show the simulation screenshot for a task or Sandbox env."""
    # Match task name
    key = next((k for k in SIM_PREVIEWS if k.lower() in task_name.lower()), None)
    path = SIM_PREVIEWS.get(key) if key else None
    sandbox_path = SIM_PREVIEWS.get("Sandbox")

    if path and path.exists():
        st.image(str(path), caption=f"{key} - simulation view", width='content')
    elif sandbox_path.exists():
        sandbox_path = SIM_PREVIEWS.get("Sandbox")
        st.image(str(sandbox_path), caption="Sandbox - simulation view", width='content')
    else:
        st.markdown(
            """
            <div style="
                border: 2px dashed #555;
                border-radius: 8px;
                padding: 40px;
                text-align: center;
                color: #888;
                background: #1a1a1a;
            ">
                Simulation preview<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Main UI

def _show_configuration_page():
    """First page: collect all configuration choices before entering the workspace."""
    st.title("Robot Learning from Demonstration")
    st.subheader("1. Configuration")
    st.caption(
        "Choose the environment, input, teaching signal, and task first. "
        "These choices are then used by the workspace on the next page."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Environment")
        env_keys = list(ENVIRONMENTS.keys())
        environment = st.selectbox(
            "Environment",
            env_keys,
            index=env_keys.index(st.session_state.environment),
            format_func=lambda k: ENVIRONMENTS[k]["label"],
            key="config_environment",
            label_visibility="collapsed",
        )
        st.session_state.environment = environment
        env_cfg = ENVIRONMENTS[environment]
        track = env_cfg.get("track", "teleop")
        st.caption(f"Robot: **{env_cfg['robot']}**")

        if track == "online":
            st.markdown("### Learner")
            learner_keys = list(LEARNERS.keys())
            default_learner = (
                st.session_state.learner
                if st.session_state.learner in learner_keys
                else learner_keys[0]
            )
            learner = st.selectbox(
                "Learner",
                learner_keys,
                index=learner_keys.index(default_learner),
                format_func=lambda k: LEARNERS[k]["label"],
                key="config_learner",
                label_visibility="collapsed",
            )
            st.session_state.learner = learner
            st.info(LEARNERS[learner]["description"])
            st.caption("Any learner can be paired with any online environment here.")

        else:
            st.session_state.learner = None

            st.markdown("### Robot Mode")
            mode = st.radio(
                "Robot mode",
                ["Simulation", "Real Robot"],
                index=1 if st.session_state.mode == "Real Robot" else 0,
                horizontal=True,
                key="config_mode",
                label_visibility="collapsed",
            )
            st.session_state.mode = mode

            if mode == "Real Robot":
                st.info(f"Real Robot — kinesthetic teaching on {env_cfg['robot']}.")
            else:
                st.info("Simulation — robosuite SpaceMouse collection.")

            st.markdown("### Input")
            input_mode = st.radio(
                "Input mode",
                ["Joint", "Image"],
                index=0 if st.session_state.input_mode == "Joint" else 1,
                horizontal=True,
                key="config_input_mode",
            )
            st.session_state.input_mode = input_mode

    with col2:
        if track == "teleop":
            st.markdown("### Teaching Signal")
            teaching_signal = st.selectbox(
                "Teaching signal",
                ["evaluative", "demonstrations", "corrective", "rankings"],
                index=[
                    "evaluative",
                    "demonstrations",
                    "corrective",
                    "rankings",
                ].index(st.session_state.teaching_signal),
                key="config_teaching_signal",
            )
            st.session_state.teaching_signal = teaching_signal

            if teaching_signal == "evaluative":
                evaluative_method = st.selectbox(
                    "Evaluation method",
                    ["deep tamer", "tamer with shaping"],
                    index=[
                        "deep tamer",
                        "tamer with shaping",
                    ].index(st.session_state.evaluative_method),
                    key="config_evaluative_method",
                )
                st.session_state.evaluative_method = evaluative_method

        st.markdown("### Task")
        tasks = get_tasks()

        new_task = st.text_input(
            "New task name",
            key="config_new_task",
            placeholder="e.g. push",
        )
        if st.button("Create task", key="config_create_task"):
            task_name = new_task.strip().replace(" ", "_")
            if task_name:
                task_dir = DEMOS_DIR / task_name
                task_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.current_task = task_name
                log(f"Created task: {task_name}")
                st.rerun()
            else:
                st.warning("Enter a task name first.")

        tasks = get_tasks()
        if tasks:
            selected = st.selectbox(
                "Task",
                tasks,
                index=(
                    tasks.index(st.session_state.current_task)
                    if st.session_state.current_task in tasks
                    else 0
                ),
                key="config_task",
            )
            if selected != st.session_state.current_task:
                st.session_state.current_task = selected
                log(f"Switched to task: {selected}")

            if st.session_state.current_task:
                if track == "online":
                    st.caption(
                        f"Checkpoints will be saved under "
                        f"`models/{st.session_state.current_task}/`."
                    )
                else:
                    st.metric(
                        "Demonstrations recorded",
                        get_demo_count(st.session_state.current_task),
                    )
        else:
            st.info("No tasks yet. Create one above.")

    if track == "teleop" and st.session_state.mode == "Real Robot":
        st.divider()
        st.markdown("### Robot Connection")

        robot_ip = st.text_input(
            "Robot IP",
            value=st.session_state.robot_ip,
            disabled=st.session_state.robot_connected,
            key="config_robot_ip",
        )
        st.session_state.robot_ip = robot_ip

        robot_color = "🟢" if st.session_state.robot_connected else "🔴"
        status = "Connected" if st.session_state.robot_connected else "Disconnected"
        st.markdown(f"{robot_color} **{status}**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Connect",
                disabled=st.session_state.robot_connected,
                key="config_connect",
            ):
                with st.spinner(f"Connecting to robot at {robot_ip}..."):
                    try:
                        import panda_py
                        from panda_py import libfranka
                        st.session_state.panda = panda_py.Panda(robot_ip)
                        st.session_state.gripper = libfranka.Gripper(robot_ip)
                        st.session_state.robot_connected = True
                        log(f"Connected to Franka at {robot_ip}")
                    except Exception as e:
                        log(f"Connection failed: {e}")
                        st.error(f"Connection failed: {e}")
                st.rerun()

        with col2:
            if st.button(
                "Disconnect",
                disabled=not st.session_state.robot_connected,
                key="config_disconnect",
            ):
                st.session_state.robot_connected = False
                st.session_state.panda = None
                st.session_state.gripper = None
                log("Disconnected")
                st.rerun()

    st.divider()

    # Configuration summary before entering the workspace.
    st.markdown("### Configuration Summary")
    if track == "online":
        summary_cols = st.columns(3)
        summary_cols[0].metric("Environment", env_cfg["label"])
        summary_cols[1].metric(
            "Learner",
            LEARNERS[st.session_state.learner]["label"] if st.session_state.learner else "Not selected",
        )
        summary_cols[2].metric("Task", st.session_state.current_task or "Not selected")
    else:
        summary_cols = st.columns(5)
        summary_cols[0].metric("Environment", env_cfg["label"])
        summary_cols[1].metric("Robot Mode", st.session_state.mode)
        summary_cols[2].metric("Input", st.session_state.input_mode)
        summary_cols[3].metric("Teaching", st.session_state.teaching_signal)
        summary_cols[4].metric(
            "Task",
            st.session_state.current_task or "Not selected",
        )

        if st.session_state.teaching_signal == "evaluative":
            st.caption(
                f"Evaluation method: **{st.session_state.evaluative_method}**"
            )

    needs_robot = (
        track == "teleop"
        and st.session_state.mode == "Real Robot"
        and not st.session_state.robot_connected
    )
    needs_learner = track == "online" and not st.session_state.learner

    if not st.session_state.current_task:
        st.warning("Select or create a task before continuing.")
    if needs_robot:
        st.warning("Connect to the robot before continuing.")
    if needs_learner:
        st.warning("Select a learner before continuing.")

    if st.button(
        "Continue to Workspace →",
        type="primary",
        disabled=not st.session_state.current_task or needs_robot or needs_learner,
        use_container_width=True,
        key="config_continue",
    ):
        # Set up the experiment: make sure its data/model directories exist
        # so the workspace page (and any background process it launches)
        # has somewhere to write to right away.
        (DEMOS_DIR / st.session_state.current_task).mkdir(parents=True, exist_ok=True)
        (MODELS_DIR / st.session_state.current_task).mkdir(parents=True, exist_ok=True)

        log_parts = [f"environment={st.session_state.environment}"]
        if track == "online":
            log_parts.append(f"learner={st.session_state.learner}")
        else:
            log_parts.append(f"mode={st.session_state.mode}")
            log_parts.append(f"input={st.session_state.input_mode}")
            log_parts.append(f"teaching={st.session_state.teaching_signal}")
        log_parts.append(f"task={st.session_state.current_task}")

        log("Configuration confirmed: " + ", ".join(log_parts))
        st.session_state.ui_page = "workspace"
        st.rerun()


def _show_online_workspace(env_cfg):
    """
    Workspace page for "online" track environments (e.g. Stack Cups +
    TAMER, or Reach (Online) + SAC). There's no Record/Train/Execute
    pipeline here - the learner picks actions and updates itself in one
    live loop, so "running the experiment" just means launching
    train_online.py (see launch_online_training()) and giving feedback
    while it runs. Any registered learner can be picked for any
    online-track environment; nothing here checks whether that combination
    is sensible.
    """
    env_key = st.session_state.environment
    learner_key = st.session_state.learner
    learner_cfg = LEARNERS[learner_key]
    output_path = online_output_path(st.session_state.current_task, env_key, learner_key)

    tab_train, tab_log = st.tabs(["▶ Live Training", "📋 Log"])

    with tab_train:
        st.header(f"{env_cfg['label']} — {learner_cfg['label']}")
        st.info(learner_cfg["description"])

        st.markdown(
            "Launching this runs `train_online.py --environment "
            f"{env_key} --learner {learner_key}` in a new terminal window, "
            "which opens its own simulation window (and, for feedback-based "
            "learners, a feedback page). There's no separate training "
            "phase - the robot starts acting and learning from your "
            "feedback immediately."
        )

        if st.button("▶ Launch Training", type="primary", key="btn_launch_online"):
            try:
                launch_online_training(st.session_state.current_task, env_key, learner_key)
                log(
                    f"Launched train_online.py --environment {env_key} "
                    f"--learner {learner_key} for task '{st.session_state.current_task}'"
                )
            except Exception as e:
                log(f"Failed to launch training: {e}")
                st.error(f"Failed to launch training: {e}")

        if "feedback_url" in learner_cfg:
            st.markdown(f"Give feedback at: {learner_cfg['feedback_url']}")

        st.caption(
            f"The model is saved to `{output_path}` when the terminal is "
            "closed (or Ctrl+C there) - there's no programmatic stop from "
            "here yet."
        )

    with tab_log:
        st.header("Activity Log")
        if st.button("Clear log", key="btn_clear_log_online"):
            st.session_state.log = []
        log_text = (
            "\n".join(reversed(st.session_state.log))
            if st.session_state.log else "No activity yet."
        )
        st.text_area(
            "Log", value=log_text, height=400,
            label_visibility="collapsed", key="log_area_online",
        )


def main():
    init_state()
    drain_log_queue()

    st.set_page_config(
        page_title="Robot LfD",
        page_icon="🤖",
        layout="wide",
    )

    st.markdown("""
        <style>
            /* General text */
            .stApp * {
                font-size: 17px;
            }

            /* Tab labels */
            .stTabs [data-baseweb="tab"] {
                font-size: 18px !important;
                padding: 12px 24px !important;
            }

            /* Configuration metric values */
            [data-testid="stMetricValue"] {
                font-size: 24px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    if st.session_state.ui_page == "configuration":
        _show_configuration_page()
        return

    # ---------------------------------------------------------------
    # Workspace page
    # ---------------------------------------------------------------
    st.title("Robot Learning from Demonstration")

    env_cfg = ENVIRONMENTS[st.session_state.environment]
    track = env_cfg.get("track", "teleop")

    back_col, status_col = st.columns([1, 5])
    with back_col:
        if st.button("← Configuration", key="back_to_config"):
            st.session_state.ui_page = "configuration"
            st.rerun()

    with status_col:
        if track == "online":
            learner_label = (
                LEARNERS[st.session_state.learner]["label"]
                if st.session_state.learner else "?"
            )
            st.markdown(
                f"**{env_cfg['label']}** · "
                f"**{learner_label} (online learning)** · "
                f"**Task:** `{st.session_state.current_task}`"
            )
        else:
            st.markdown(
                f"**{env_cfg['label']}** · "
                f"**{st.session_state.mode}** · "
                f"**{st.session_state.input_mode} input** · "
                f"**{st.session_state.teaching_signal} teaching** · "
                f"**Task:** `{st.session_state.current_task}`"
            )

    if track == "online":
        _show_online_workspace(env_cfg)
        return

    # The workspace is deliberately based on the choices made on page 1.
    if st.session_state.mode == "Real Robot":
        mode_description = "Franka / real-robot workflow"
    else:
        mode_description = "robosuite / simulation workflow"

    teaching_description = {
        "evaluative": "evaluative feedback",
        "demonstrations": "demonstration-based teaching",
        "corrective": "corrective teaching",
        "rankings": "ranking-based teaching",
    }[st.session_state.teaching_signal]

    if st.session_state.teaching_signal == "evaluative":
        teaching_description += (
            f" ({st.session_state.evaluative_method})"
        )

    st.info(
        f"**Current configuration:** {mode_description}, "
        f"{st.session_state.input_mode.lower()} input, "
        f"{teaching_description}."
    )

# Main tabs 
    tab_record, tab_train, tab_execute, tab_log = st.tabs(
        ["⏺ Record", "⚙ Train", "▶ Execute", "📋 Log"]
    )

    # RECORD tab 
    with tab_record:
        col_header, col_refresh = st.columns([6, 1])
        with col_header:
            st.header("Record Demonstrations")
        with col_refresh:
            st.write("")
            if st.button("🔄 Refresh", key="btn_refresh_record"):
                st.rerun()

        if not st.session_state.current_task:
            st.warning("Select or create a task in the sidebar first.")
        else:
            st.markdown(f"**Task:** `{st.session_state.current_task}`  |  "
                        f"**Demos recorded:** {get_demo_count(st.session_state.current_task)}")

            if st.session_state.mode == "Real Robot":
                # Real robot recording
                st.info(
                    "1. Click **Start Recording**\n"
                    "2. Perform the demonstration\n"
                    "3. Click **Stop Recording**"
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "⏺ Start Recording",
                        disabled=st.session_state.recording or not st.session_state.robot_connected,
                        type="primary",
                    ):
                        try:
                            from scripts.demo_recorder import KinestheticDemoRecorder
                            rec = KinestheticDemoRecorder(robot_ip=st.session_state.robot_ip)
                            rec.panda = st.session_state.panda
                            rec.gripper = st.session_state.gripper
                            rec.enable_teaching_mode()
                            rec.start_recording()
                            st.session_state.recorder = rec
                            st.session_state.recording = True
                            log("Recording started")

                            def poll():
                                while st.session_state.recording:
                                    st.session_state.recorder.record_step()
                                    time.sleep(0.05)
                            run_in_thread(poll)
                        except Exception as e:
                            log(f"Recording error: {e}")

                with col2:
                    if st.button(
                        "■ Stop Recording",
                        disabled=not st.session_state.recording,
                        type="secondary",
                    ):
                        st.session_state.recording = False
                        time.sleep(0.1)
                        try:
                            rec  = st.session_state.recorder
                            demo = rec.stop_recording()
                            rec.disable_teaching_mode()
                            T = demo["actions"].shape[0]
                            save_path = DEMOS_DIR / st.session_state.current_task / "demos.hdf5"
                            rec.save(str(save_path), task_name=st.session_state.current_task)
                            log(f"Saved demo - {T} steps ({T/20:.1f}s)")
                            st.session_state.recorder = None
                        except Exception as e:
                            log(f"Stop recording error: {e}")
                        st.rerun()

                if st.session_state.recording:
                    st.error("🔴 Recording in progress.")

            else:
                # Simulation recording
                task_name = st.session_state.current_task
                env_name  = get_environment_config()["sim_env_name"]

                col_info, col_preview = st.columns([1, 1])

                with col_info:
                    st.subheader("Collect in Simulation")

                    # Step by step instruction
                    st.success("**Step by step instruction for collection**\n"
                        " 1. Read the how to use SpaceMouse\n"
                        " 2. Press 'Start Collecting Demonstrations' to open simulation\n"
                        " 3. After collection press 'Collection Done' button\n"
                        " 4. Process the demonstration file in the 'Process & Merge' section\n"
                        " - This edits data in the file so its suitable for training.\n"
                        " - If the system doesn't see your file refresh the tab with the button in top right corner.\n"
                    )

                    # Controls instruction
                    with st.expander("SpaceMouse controls", expanded=False):
                        st.markdown(
                            "- **Move / tilt** → move end-effector\n"
                            "- **Left button** → hold to close gripper\n"
                            "- **Right button** → reset the demo\n"
                            "- **Both buttons** → save the demo\n"
                            "- **CTRL + Q** in viewer → quit\n"
                            "- **CTRL + C** in terminal (after collecting all demos) → quit"
                        )

                    if st.button(
                        "▶ Start Collecting Demonstrations",
                        type="primary",
                        disabled=st.session_state.sim_collecting,
                        key="btn_launch_sim",
                    ):
                        (DEMOS_DIR / task_name).mkdir(parents=True, exist_ok=True)
                        try:
                            launch_sim_collection(task_name, env_name)
                            st.session_state.sim_collecting = True
                            log(f"Launched sim collection - env={env_name}")
                            log(f"Output will appear in data/{task_name}/TEMP/demo.hdf5")
                        except Exception as e:
                            log(f"Failed to launch terminal: {e}")

                    if st.session_state.sim_collecting:
                        st.info(
                            "🟢 Collection terminal and simulation window is open.\n\n"
                            "When you're done, come back here and click **Collection Done**."
                        )

                        done_clicked = st.button("Collection Done", key="btn_done_sim",help = "Press after completing demos", icon="✅", icon_position="left")
                        if done_clicked:
                            st.session_state.sim_collecting = False
                            # Auto-detect the latest demo.hdf5
                            latest = find_latest_demo(task_name)
                            if latest:
                                st.session_state.sim_last_demo_path = str(latest)
                                log(f"Detected demo file: {latest}")
                            else:
                                log("No demo.hdf5 found yet - check the data folder.")
                            st.rerun()

                    st.divider()

                    # Process & Merge
                    st.subheader("Process & Merge Demos")

                    latest_demo = find_latest_demo(task_name)
                    merged_path = get_merged_path(task_name)
                    all_obs = [p for p in find_all_hdf5(task_name) if p.name == "obs.hdf5"]

                    if latest_demo:
                        obs_ready = (latest_demo.parent / "obs.hdf5").exists()
                        if obs_ready:
                            st.success(f"Latest batch already processed - `obs.hdf5` exists.")
                        else:
                            st.markdown(f"**Latest batch:** `{latest_demo}`")
                    else:
                        st.warning("No demo.hdf5 found in this task's data folder yet.")

                    if merged_path.exists():
                        merged_mtime = merged_path.stat().st_mtime
                        new_obs = [p for p in all_obs if p.stat().st_mtime > merged_mtime]
                        st.info(f"Merged file exists - {len(new_obs)} unmerged batch(es) found.")
                    else:
                        new_obs = all_obs

                    if st.button(
                        "⚙ Process & Merge",
                        type="primary",
                        disabled=latest_demo is None,
                        key="btn_process_merge",
                    ):
                        merged_path.parent.mkdir(parents=True, exist_ok=True)
                        obs_path = latest_demo.parent / "obs.hdf5"
                        merge_script = str(REPO_ROOT / "scripts" / "merge_demos.py")

                        # Step 1: process latest demo.hdf5 → obs.hdf5
                        cmd1 = " ".join([
                            "python", "-m", "robomimic.scripts.dataset_states_to_obs",
                            "--dataset", str(latest_demo),
                            "--output_name", "obs.hdf5",
                            "--done_mode", "0",
                        ])
                        cmd2 = " ".join([
                            "python", "-m", "robomimic.scripts.split_train_val",
                            "--dataset", str(obs_path),
                            "--ratio", "0.1",
                        ])

                        # Step 2: merge new obs files into existing merged 
                        if merged_path.exists():
                            merged_mtime = merged_path.stat().st_mtime
                            sources = [p for p in find_all_hdf5(task_name)
                                       if p.name == "obs.hdf5" and p.stat().st_mtime > merged_mtime]
                            sources.append(merged_path)
                        else:
                            sources = [p for p in find_all_hdf5(task_name) if p.name == "obs.hdf5"]
                        # include the just-processed file even if not on disk yet
                        if obs_path not in sources:
                            sources.append(obs_path)
                        src_args = " ".join(f'"{str(s)}"' for s in sources)
                        cmd3 = f"python {merge_script} --inputs {src_args} --out \"{str(merged_path)}\""

                        bash_cmd = (
                            f"{cmd1} && {cmd2} && {cmd3} && "
                            f"echo 'Press Enter to close.'; read"
                        )
                        subprocess.Popen(
                            ["gnome-terminal", "--", "bash", "-c", bash_cmd],
                            cwd=str(REPO_ROOT)
                        )
                        log("Process & Merge started in a new terminal.")
                        st.rerun()

                    st.caption(
                        "**What this does:** converts raw simulation states to observations (obs.hdf5), "
                        "splits into train/val sets, then if multiple demo files exist merges all batches into a single file. "
                        f"The merged file is saved to `data/{task_name}/merged/merged.hdf5` - "
                        "use this file for training."
                    )

                with col_preview:
                    st.subheader("Environment Preview")
                    sim_preview(task_name)

    # TRAIN tab
    with tab_train:
        col_header, col_refresh = st.columns([6, 1])
        with col_header:
            st.header("Train Policy")
        with col_refresh:
            st.write("")
            if st.button("🔄 Refresh", key="btn_refresh_train"):
                st.rerun()

        st.success("**How to train a policy**\n"
                        " 1. If you already have recorded demonstrations choose which file you want to train on.\n"
                        " 2. Choose the hyperparameters.\n"
                        " - Epochs - how many times the model trains through your demos. More = longer training but better learning. Start with 500.\n"
                        " - Batch size - how many steps the model learns from at once. Keep at 16 unless you get memory issues errors.\n"
                        " 3. When ready press 'Train Localy'. Training will take at least 10 minutes.\n"
                        " 4. After training you can see the performance of policy in 'Execute' tab\n"
                    )

        if not st.session_state.current_task:
            st.warning("Select a task first.")
        else:
            n_demos = get_demo_count(st.session_state.current_task)
            st.markdown(f"**Task:** `{st.session_state.current_task}` - {n_demos} demos")

            if n_demos < 15:
                st.warning(f"Only {n_demos} demos. Recommend at least 20 before training.")

            st.subheader("Local training")

            # Dataset picker - any .hdf5 in the task folder
            available_hdf5 = find_all_hdf5(st.session_state.current_task)
            # Filter to files that look like processed obs files (not raw demo.hdf5)
            trainable = [p for p in available_hdf5 if p.name != "demo.hdf5"]
            if trainable:
                hdf5_labels = [str(p.relative_to(DEMOS_DIR)) for p in trainable]
                selected_label = st.selectbox("Dataset to train on", hdf5_labels, index=0)
                dataset_path = DEMOS_DIR / selected_label
            else:
                st.warning("No processed .hdf5 files found. Run Post-process first.")
                dataset_path = None

            # Check for existing checkpoint
            last_ckpt = find_latest_checkpoint(st.session_state.current_task)
            last_epoch = get_epoch_from_checkpoint(last_ckpt) if last_ckpt else None
            is_resume = last_epoch is not None

            if is_resume:
                st.info(
                    f"**Continued training** - an existing model was found "
                    f"(trained up to epoch {last_epoch}). "
                    f"Training will resume from epoch {last_epoch + 1} and run for the "
                    f"number of additional epochs you set below."
                )

            col1, col2 = st.columns(2)
            with col1:
                epoch_label = "Additional epochs" if is_resume else "Epochs"
                n_epochs = st.number_input(epoch_label, value=100 if is_resume else 500, min_value=10, step=50)
            with col2:
                batch_size = st.number_input("Batch size", value=16, min_value=8, step=4)

            if is_resume:
                st.caption(f"Total epochs after training: **{last_epoch + n_epochs}**")

            btn_label = "⚙ Continue Training" if is_resume else "⚙ Train locally"
            if st.button(
                btn_label,
                disabled=st.session_state.training or n_demos == 0 or dataset_path is None,
                type="primary",
            ):
                config_path = generate_train_config(
                    task_name=st.session_state.current_task,
                    dataset_path=dataset_path,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    resume_from_epoch=last_epoch,
                )
                log(f"Config written to {config_path}")

                def train():
                    cmd = [
                        "python", "-m", "robomimic.scripts.train",
                        "--config", str(config_path),
                    ]
                    if is_resume:
                        cmd.append("--resume")
                    cmd_str = " ".join(cmd)
                    bash_cmd = (
                        f"{cmd_str} &&"
                        f"echo 'Press Enter to close.'; read"
                    )

                    subprocess.Popen(
                        ["gnome-terminal", "--", "bash", "-c", bash_cmd],
                        cwd=str(REPO_ROOT)
                    )
                    extra = f"(continuing from epoch {last_epoch})" if is_resume else ""
                    log(f"Training started in a new terminal - {n_epochs} epochs {extra}")
                    st.session_state.training = False

                train()
                st.rerun()

            st.divider()

            models = get_models(st.session_state.current_task)
            if models:
                st.subheader("Saved checkpoints")
                for m in models[-5:]:
                    st.text(Path(m).name)

    # EXECUTE tab 
    with tab_execute:
        col_header, col_refresh = st.columns([6, 1])
        with col_header:
            st.header("Execute Policy")
        with col_refresh:
            st.write("")
            if st.button("🔄 Refresh", key="btn_refresh_execute"):
                st.rerun()

        st.success("**How to execute a policy**\n"
                        " 1. Choose the model you want to see. Use model with highest number\n"
                        " 2. Choose the hyperparameters.\n"
                        " - Horizon - maximum steps the robot gets per attempt. 200 steps = 7 seconds. Match it to how long your demos were.\n"
                        " - Number of demos executed - how many attempts the policy makes.\n"
                        " 3. When ready press 'Execute in Simulation'.\n"
                    )

        models = get_models(st.session_state.current_task) if st.session_state.current_task else []

        if not models:
            st.warning("No trained models found for this task.")
        else:
            model_path = st.selectbox(
                "Select checkpoint",
                models,
                format_func=lambda p: Path(p).name,
                index=len(models) - 1,
            )

            col1, col2 = st.columns(2)
            with col1:
                horizon = st.number_input("Horizon (steps)", value=400, min_value=50, step=50)
            with col2:
                num_demos_ex = st.number_input("Number of demos executed", value=5, min_value=1, step=1)

            if st.session_state.mode == "Real Robot":
                st.warning("⚠ Make sure the workspace is clear before running.")

                col_run, col_stop = st.columns(2)
                with col_run:
                    if st.button(
                        "▶ Run on Robot",
                        disabled=st.session_state.executing or not st.session_state.robot_connected,
                        type="primary",
                    ):
                        def execute():
                            cmd = [
                                "python", "-u", str(EXECUTE_PATH),
                                "--policy", model_path,
                                "--horizon", str(horizon),
                            ]
                            cmd_str = " ".join(cmd)
                            bash_cmd = (
                                f"{cmd_str} &&"
                                f"echo 'Press Enter to close.'; read"
                            )

                            subprocess.Popen(
                                ["gnome-terminal", "--", "bash", "-c", bash_cmd],
                                cwd=str(REPO_ROOT)
                            )
                            log(f"Policy execution started in a new terminal.")
                            st.session_state.executing = False

                        execute()
                        st.rerun()

                with col_stop:
                    if st.button("■ Stop", disabled=not st.session_state.executing, type="secondary"):
                        st.session_state.executing = False
                        log("Execution stopped by user")

                if st.session_state.executing:
                    st.warning("🟡 Policy running.")

            else:
                # Simulation execution
                if st.button("▶ Execute in Simulation"):
                    cmd = [
                        "python", "-m", "robomimic.scripts.run_trained_agent",
                        "--agent", model_path,
                        "--n_rollouts", str(num_demos_ex),
                        "--horizon", str(horizon),
#                        "--render",
                        "--results_path user_study_data/user2_push.csv",
                    ]
                    cmd_str = " ".join(cmd)
                    bash_cmd = (
                        f"{cmd_str} &&"
                        f"echo 'Press Enter to close.'; read"
                    )

                    subprocess.Popen(
                        ["gnome-terminal", "--", "bash", "-c", bash_cmd],
                        cwd=str(REPO_ROOT)
                    )
                    log("Sim execution started in a new terminal.")



    # LOG tab
    with tab_log:
        st.header("Activity Log")
        if st.button("Clear log"):
            st.session_state.log = []
        log_text = "\n".join(reversed(st.session_state.log)) if st.session_state.log else "No activity yet."
        st.text_area("Log", value=log_text, height=400, label_visibility="collapsed")




if __name__ == "__main__":
    main()
