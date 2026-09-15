"""
A minimal TAMER learner (Knox & Stone, 2009): an online-trained model
H(s, a) that predicts the reward a human WOULD give for a state/action,
and a greedy policy that picks the action maximising H at every step.

Unlike RL, TAMER never uses an environment reward and never bootstraps
across steps - it purely regresses towards the human's signal. Because
human feedback usually lands a beat after the behaviour it's rating,
each new signal is credited back across the last `credit_window` steps
rather than only the one just taken.

Simplifications vs. the published algorithm (fine for a teaching example,
worth revisiting for real use):
  - Credit assignment uses a fixed step-count window with uniform weight,
    not the timestamped credit function from the paper.
  - Actions are chosen by "random shooting" (sample N candidates, act
    greedily w.r.t. H) rather than gradient-based action optimisation -
    simple and derivative-free, but scales poorly to high-dimensional
    action spaces.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learners.learner import Learner


class _RewardModel(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class TamerLearner(Learner):

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_low,
        action_high,
        credit_window=10,
        n_candidate_actions=64,
        lr=1e-3,
    ):
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.n_candidate_actions = n_candidate_actions

        self.model = _RewardModel(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self._history = deque(maxlen=credit_window)

    def reset(self):
        self._history.clear()

    def select_action(self, obs):
        candidates = np.random.uniform(
            self.action_low,
            self.action_high,
            size=(self.n_candidate_actions, len(self.action_low)),
        ).astype(np.float32)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).expand(len(candidates), -1)
            act_t = torch.as_tensor(candidates, dtype=torch.float32)
            values = self.model(obs_t, act_t).numpy()

        return candidates[int(np.argmax(values))]

    def observe(self, obs, action, reward, **kwargs):
        """
        `reward` here is the human's feedback for this step (e.g. from
        info["human_reward"] - see environments/stack_cups/franka_stack_cups_env.py),
        not an environment reward. A zero means "no feedback arrived this
        step" and only extends the history; a nonzero value also triggers
        a training step, crediting it across the recent window.
        """
        self._history.append((
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
        ))

        if reward == 0.0:
            return

        obs_batch = torch.as_tensor(np.stack([o for o, _ in self._history]), dtype=torch.float32)
        act_batch = torch.as_tensor(np.stack([a for _, a in self._history]), dtype=torch.float32)
        target = torch.full((len(self._history),), float(reward))

        pred = self.model(obs_batch, act_batch)
        loss = F.mse_loss(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
