"""A tiny continuous-control task, so tests can check that agents learn in seconds."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

TOY_ENV = "ReachTarget-v0"


class ReachTarget(gym.Env):
    """Move a point on a line to a random target.

    The observation is ``(position, target)``. Each step moves the point by ``0.05 * action``
    and pays ``1 - |position - target|``. Leaving ``[-1, 1]`` terminates the episode, which
    forfeits the remaining reward; otherwise it is truncated after 30 steps (by the
    TimeLimit wrapper). The action bounds are ``[-2, 2]`` so the agents' ``[-1, 1]`` actions
    have to be rescaled.
    """

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-2.0, 2.0, (2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0.0
        self.target = float(self.np_random.uniform(-0.8, 0.8))
        return self._obs(), {}

    def step(self, action):
        self.position += 0.05 * float(np.clip(action[0], -2.0, 2.0))
        terminated = abs(self.position) > 1.0
        reward = 0.0 if terminated else 1.0 - abs(self.position - self.target)
        return self._obs(), reward, terminated, False, {}

    def _obs(self) -> np.ndarray:
        return np.array([self.position, self.target], dtype=np.float32)


if TOY_ENV not in gym.registry:
    gym.register(id=TOY_ENV, entry_point=ReachTarget, max_episode_steps=30)


@pytest.fixture
def toy_env_id() -> str:
    return TOY_ENV
