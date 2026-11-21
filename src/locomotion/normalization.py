"""Running statistics for observation normalisation and reward scaling (used by PPO)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RunningMeanStd(nn.Module):
    """Mean and variance of everything seen so far, merged batch by batch (Chan et al.). Stored
    as buffers so they are saved with the agent's ``state_dict``."""

    def __init__(self, shape: tuple[int, ...] = ()):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    def update(self, batch: np.ndarray) -> None:
        """``batch`` has shape ``(n, *shape)``."""
        batch = torch.as_tensor(batch, dtype=torch.float64, device=self.mean.device)
        n = batch.shape[0]
        batch_mean, batch_var = batch.mean(0), batch.var(0, unbiased=False)
        total = self.count + n
        delta = batch_mean - self.mean
        m2 = self.var * self.count + batch_var * n + delta**2 * self.count * n / total
        self.mean += delta * n / total
        self.var.copy_(m2 / total)
        self.count.copy_(total)


class ObservationNormalizer(nn.Module):
    """``clip((obs - mean) / std, -clip, clip)`` with running statistics."""

    def __init__(self, obs_dim: int, clip: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.stats = RunningMeanStd((obs_dim,))
        self.clip, self.eps = clip, eps

    def __call__(self, obs: np.ndarray, update: bool = False) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        if update:
            self.stats.update(obs.reshape(1, -1) if obs.ndim == 1 else obs)
        mean, var = self.stats.mean.cpu().numpy(), self.stats.var.cpu().numpy()
        return np.clip((obs - mean) / np.sqrt(var + self.eps), -self.clip, self.clip).astype(
            np.float32
        )


class RewardScaler(nn.Module):
    """Divides rewards by the running standard deviation of the discounted return, as OpenAI
    Baselines' PPO does (described by Engstrom et al. 2020). The mean is not subtracted, so the
    sign of rewards is kept."""

    def __init__(self, gamma: float, eps: float = 1e-8):
        super().__init__()
        self.stats = RunningMeanStd()
        self.gamma, self.eps = gamma, eps
        self.discounted_return = 0.0

    def __call__(self, reward: float, episode_end: bool) -> float:
        self.discounted_return = self.discounted_return * self.gamma + reward
        self.stats.update(np.array([self.discounted_return]))
        if episode_end:
            self.discounted_return = 0.0
        return float(reward / np.sqrt(self.stats.var.item() + self.eps))
