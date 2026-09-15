"""
Import every robosuite-registered task here so `robosuite.make(...)` can
find it. Add a new task's import when you wire it up for sim demo
collection (see docs/adding_environments_tasks_learners.md).

This is deliberately not environments/__init__.py: Python always runs a
package's __init__.py before any of its submodules, so an eager import
here would force robosuite onto every consumer of anything under
environments/ - including environments.reach.franka_reach_env, which
run.py uses without robosuite at all. Only import this module from code
that already needs robosuite (currently just
input_devices/collect_human_demonstrations.py).
"""

from environments.reach import reach_env  # noqa: F401  registers "ReachTask"
