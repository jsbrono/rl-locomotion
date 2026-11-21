"""Experience storage: a replay buffer for SAC and TD3, and GAE for PPO."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class Batch(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor


class ReplayBuffer:
    """A fixed-size ring buffer of transitions, sampled uniformly.

    Only ``terminated`` is stored, not ``truncated``: an episode cut off by a time limit did not
    reach a terminal state, so its last transition must still bootstrap from ``next_obs``.
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, seed: int | None = None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        i = self.position
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, terminated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> Batch:
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self.rng.integers(0, self.size, size=batch_size)
        arrays = (self.obs, self.actions, self.rewards, self.next_obs, self.terminated)
        return Batch(*(torch.as_tensor(a[idx], device=device) for a in arrays))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_end: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation (Schulman et al. 2016) over a rollout that may span
    several episodes. Returns ``(advantages, returns)`` with ``returns = advantages + values``.

    ``next_values[t]`` is ``V(s_{t+1})`` for the observation actually returned by step ``t``.
    The TD error bootstraps from it unless the episode *terminated*; after a time-limit
    truncation it still bootstraps. ``episode_end`` (terminated or truncated) stops the
    advantage from leaking into the next episode. The last step of the rollout bootstraps from
    ``next_values`` like any other.
    """
    not_terminal = 1.0 - terminated.astype(np.float64)
    continues = 1.0 - episode_end.astype(np.float64)
    deltas = rewards + gamma * not_terminal * next_values - values

    advantages = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = deltas[t] + gamma * gae_lambda * continues[t] * running
        advantages[t] = running
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)
