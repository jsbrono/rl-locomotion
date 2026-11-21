"""Environment construction."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


def make_env(env_id: str, seed: int | None = None, **kwargs: Any) -> gym.Env:
    """Creates ``env_id`` with actions rescaled to ``[-1, 1]`` and clipped to that range, and
    with episode returns recorded in ``info["episode"]`` when an episode ends."""
    env = gym.make(env_id, **kwargs)
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError(f"{env_id} does not have a continuous (Box) action space")
    shape = env.action_space.shape
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RescaleAction(
        env, np.full(shape, -1.0, dtype=np.float32), np.full(shape, 1.0, dtype=np.float32)
    )
    env = gym.wrappers.ClipAction(env)
    env.reset(seed=seed)
    return env


def space_dims(env: gym.Env) -> tuple[int, int]:
    """``(obs_dim, act_dim)`` for a flat Box observation space and Box action space."""
    (obs_dim,) = env.observation_space.shape
    (act_dim,) = env.action_space.shape
    return obs_dim, act_dim
