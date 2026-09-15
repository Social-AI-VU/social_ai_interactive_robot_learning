"""
Base interface for pluggable learning algorithms.

A Learner picks actions and improves from experience. It sits below
whatever training-loop script drives an environment - the loop is the only
thing that needs to change to swap algorithms; environments and reward
sources don't need to know which learner is in use.

run.py trains with stable-baselines3's SAC directly rather than through
this interface (SAC's own `.learn()` loop already does the job). This
interface is for algorithms - like TAMER - that need their own step-by-step
training loop instead. See learners/tamer_learner.py for a worked example,
driven by train_stack_cups_tamer.py.
"""

from abc import ABC, abstractmethod


class Learner(ABC):

    @abstractmethod
    def select_action(self, obs):
        """Return an action for the given observation."""

    @abstractmethod
    def observe(self, obs, action, reward, **kwargs):
        """Update the learner from one step of experience."""

    def reset(self):
        """Called at the start of each episode. Optional to override."""

    def save(self, path):
        """Optional: persist the learner's parameters."""

    def load(self, path):
        """Optional: restore the learner's parameters."""
