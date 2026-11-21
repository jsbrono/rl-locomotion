"""Proximal Policy Optimisation with the clipped objective (Schulman et al. 2017).

PPO alternates between collecting a fixed number of steps with the current policy and
running several epochs of minibatch updates on them. The clipped probability ratio keeps each
update close to the policy that collected the data.

Following the implementation details that matter for MuJoCo (Engstrom et al. 2020; Huang et
al. 2022): observations are normalised with running statistics, rewards are scaled by the
running standard deviation of the return, advantages are normalised per minibatch, gradients
are clipped and the learning rate decays linearly to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from locomotion.buffers import compute_gae
from locomotion.networks import GaussianPolicy, ValueNetwork
from locomotion.normalization import ObservationNormalizer, RewardScaler


@dataclass
class PPOConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rollout_steps: int = 2048
    epochs: int = 10
    minibatch_size: int = 64
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    init_log_std: float = 0.0
    normalize_obs: bool = True
    scale_rewards: bool = True
    normalize_advantages: bool = True


@dataclass
class Rollout:
    """``rollout_steps`` transitions, with observations normalised as the policy saw them and
    rewards as used for training (scaled if ``scale_rewards``)."""

    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    episode_end: np.ndarray


class PPO:
    Config = PPOConfig
    on_policy = True

    def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig, device: torch.device):
        self.config, self.device = config, device
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.policy = GaussianPolicy(obs_dim, act_dim, config.hidden_sizes, config.init_log_std).to(
            device
        )
        self.value = ValueNetwork(obs_dim, config.hidden_sizes).to(device)
        self.parameters = [*self.policy.parameters(), *self.value.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=config.lr, eps=1e-5)
        self.obs_normalizer = ObservationNormalizer(obs_dim) if config.normalize_obs else None
        self.reward_scaler = RewardScaler(config.gamma) if config.scale_rewards else None
        self._obs: np.ndarray | None = None  # normalised observation where collection resumes

    def normalize(self, obs: np.ndarray, update: bool = False) -> np.ndarray:
        if self.obs_normalizer is None:
            return np.asarray(obs, dtype=np.float32)
        return self.obs_normalizer(obs, update=update)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Action for a raw observation, using frozen normalisation statistics. The action may
        fall outside ``[-1, 1]``; the environment clips it."""
        obs_t = torch.as_tensor(self.normalize(obs), device=self.device).unsqueeze(0)
        dist = self.policy.distribution(obs_t)
        action = dist.mean if deterministic else dist.sample()
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def collect(self, env: gym.Env, num_steps: int) -> tuple[Rollout, list[dict]]:
        """Runs the current policy for ``num_steps`` steps, continuing the episode left
        unfinished by the previous call. Returns the rollout and the episodes that ended
        (``{"step", "return", "length"}``, ``step`` counted within this rollout, returns
        unscaled)."""
        obs = np.zeros((num_steps, self.obs_dim), dtype=np.float32)
        next_obs = np.zeros_like(obs)
        actions = np.zeros((num_steps, self.act_dim), dtype=np.float32)
        log_probs, rewards = np.zeros(num_steps, np.float32), np.zeros(num_steps, np.float32)
        terminated, episode_end = np.zeros(num_steps, bool), np.zeros(num_steps, bool)
        episodes = []

        if self._obs is None:
            self._obs = self.normalize(env.reset()[0], update=True)
        for t in range(num_steps):
            obs[t] = self._obs
            dist = self.policy.distribution(torch.as_tensor(obs[t], device=self.device))
            action = dist.sample()
            log_probs[t] = dist.log_prob(action).sum().item()
            actions[t] = action.cpu().numpy()

            raw_next, reward, terminated[t], truncated, info = env.step(actions[t])
            episode_end[t] = terminated[t] or truncated
            # The observation after a time limit is still a real state: keep it for
            # bootstrapping, then reset.
            next_obs[t] = self.normalize(raw_next, update=True)
            rewards[t] = (
                self.reward_scaler(float(reward), episode_end[t]) if self.reward_scaler else reward
            )
            if episode_end[t]:
                stats = info["episode"]
                episodes.append(
                    {"step": t + 1, "return": float(stats["r"]), "length": int(stats["l"])}
                )
                self._obs = self.normalize(env.reset()[0], update=True)
            else:
                self._obs = next_obs[t]

        rollout = Rollout(obs, actions, log_probs, rewards, next_obs, terminated, episode_end)
        return rollout, episodes

    def update(self, rollout: Rollout, progress: float = 0.0) -> dict[str, float]:
        """Runs ``epochs`` passes of minibatch updates. ``progress`` (fraction of training
        done) sets the learning rate when ``anneal_lr`` is on."""
        config = self.config
        lr = config.lr * (1.0 - progress) if config.anneal_lr else config.lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        def tensor(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(x, device=self.device)

        obs, actions, old_log_probs = (
            tensor(rollout.obs),
            tensor(rollout.actions),
            tensor(rollout.log_probs),
        )
        with torch.no_grad():
            values = self.value(obs).cpu().numpy()
            next_values = self.value(tensor(rollout.next_obs)).cpu().numpy()
        advantages, returns = compute_gae(
            rollout.rewards,
            values,
            next_values,
            rollout.terminated,
            rollout.episode_end,
            config.gamma,
            config.gae_lambda,
        )
        advantages, returns = tensor(advantages), tensor(returns)

        totals: dict[str, float] = {}
        num_minibatches = 0
        for _ in range(config.epochs):
            for idx in torch.randperm(len(obs), device=self.device).split(config.minibatch_size):
                dist = self.policy.distribution(obs[idx])
                log_prob = dist.log_prob(actions[idx]).sum(-1)
                log_ratio = log_prob - old_log_probs[idx]
                ratio = log_ratio.exp()

                adv = advantages[idx]
                if config.normalize_advantages and len(idx) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                clipped = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range)
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                value_loss = F.mse_loss(self.value(obs[idx]), returns[idx])
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters, config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    batch = {
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy": entropy.item(),
                        # Low-variance KL(old || new) estimator (Schulman, "Approximating KL").
                        "approx_kl": ((ratio - 1) - log_ratio).mean().item(),
                        "clip_fraction": ((ratio - 1).abs() > config.clip_range)
                        .float()
                        .mean()
                        .item(),
                    }
                for key, value in batch.items():
                    totals[key] = totals.get(key, 0.0) + value
                num_minibatches += 1

        metrics = {key: value / num_minibatches for key, value in totals.items()}
        metrics["lr"] = lr
        metrics["action_std"] = self.policy.log_std.exp().mean().item()
        return metrics

    def state_dict(self) -> dict:
        state = {"policy": self.policy.state_dict(), "value": self.value.state_dict()}
        if self.obs_normalizer is not None:
            state["obs_normalizer"] = self.obs_normalizer.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        if self.obs_normalizer is not None:
            self.obs_normalizer.load_state_dict(state["obs_normalizer"])
