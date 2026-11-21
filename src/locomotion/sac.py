"""Soft Actor-Critic (Haarnoja et al. 2018, with the learned temperature of the 2019 follow-up).

SAC maximises reward plus the policy's entropy, weighted by a temperature ``alpha``. The
temperature is tuned automatically so the policy's entropy stays near a target (``-act_dim``
by default) unless ``alpha`` is fixed in the config.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from locomotion.buffers import Batch
from locomotion.networks import SquashedGaussianActor, TwinQNetwork, soft_update


@dataclass
class SACConfig:
    hidden_sizes: tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    alpha: float | None = None  # None: learn the temperature
    target_entropy: float | None = None  # None: -act_dim


class SAC:
    Config = SACConfig
    on_policy = False

    def __init__(self, obs_dim: int, act_dim: int, config: SACConfig, device: torch.device):
        self.config, self.device = config, device
        self.actor = SquashedGaussianActor(obs_dim, act_dim, config.hidden_sizes).to(device)
        self.critic = TwinQNetwork(obs_dim, act_dim, config.hidden_sizes).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)

        self.learn_alpha = config.alpha is None
        initial_alpha = 1.0 if config.alpha is None else config.alpha
        self.log_alpha = torch.tensor(np.log(initial_alpha), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr)
        self.target_entropy = (
            -float(act_dim) if config.target_entropy is None else config.target_entropy
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.detach().exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Samples from the policy while training; ``deterministic`` takes its mean action."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: Batch) -> dict[str, float]:
        gamma, alpha = self.config.gamma, self.alpha

        # Critic: regress both Q-functions onto the soft Bellman target.
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(batch.next_obs)
            target_q = torch.min(*self.critic_target(batch.next_obs, next_actions))
            soft_value = target_q - alpha * next_log_prob
            target = batch.rewards + gamma * (1.0 - batch.terminated) * soft_value
        q1, q2 = self.critic(batch.obs, batch.actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor: maximise min Q minus alpha * log pi, through the reparameterised sample. The
        # critic is frozen so the actor loss does not leave gradients on it.
        actions, log_prob = self.actor(batch.obs)
        self.critic.requires_grad_(False)
        q = torch.min(*self.critic(batch.obs, actions))
        self.critic.requires_grad_(True)
        actor_loss = (alpha * log_prob - q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "entropy": -log_prob.mean().item(),
        }
        # Temperature: raise alpha when entropy is below target, lower it when above.
        if self.learn_alpha:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        metrics["alpha"] = self.alpha.item()

        soft_update(self.critic_target, self.critic, self.config.tau)
        return metrics

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
