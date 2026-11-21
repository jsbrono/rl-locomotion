"""Twin Delayed DDPG (Fujimoto et al. 2018).

TD3 is DDPG with three fixes for value overestimation: two critics with the minimum taken in
the target (clipped double Q-learning), clipped noise on the target action (target policy
smoothing), and actor and target updates only every ``policy_delay`` critic updates.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from locomotion.buffers import Batch
from locomotion.networks import DeterministicActor, TwinQNetwork, soft_update


@dataclass
class TD3Config:
    hidden_sizes: tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    exploration_noise: float = 0.1  # std of Gaussian noise added to actions while training
    target_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2


class TD3:
    Config = TD3Config
    on_policy = False

    def __init__(self, obs_dim: int, act_dim: int, config: TD3Config, device: torch.device):
        self.config, self.device = config, device
        self.actor = DeterministicActor(obs_dim, act_dim, config.hidden_sizes).to(device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic = TwinQNetwork(obs_dim, act_dim, config.hidden_sizes).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.num_updates = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """The actor's action, plus exploration noise unless ``deterministic``."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_t).squeeze(0)
        if not deterministic:
            noise = torch.randn_like(action) * self.config.exploration_noise
            action = (action + noise).clamp(-1.0, 1.0)
        return action.cpu().numpy()

    def update(self, batch: Batch) -> dict[str, float]:
        config = self.config
        self.num_updates += 1

        with torch.no_grad():
            noise = (torch.randn_like(batch.actions) * config.target_noise).clamp(
                -config.noise_clip, config.noise_clip
            )
            next_actions = (self.actor_target(batch.next_obs) + noise).clamp(-1.0, 1.0)
            target_q = torch.min(*self.critic_target(batch.next_obs, next_actions))
            target = batch.rewards + config.gamma * (1.0 - batch.terminated) * target_q
        q1, q2 = self.critic(batch.obs, batch.actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        metrics = {"critic_loss": critic_loss.item()}

        if self.num_updates % config.policy_delay == 0:
            # Deterministic policy gradient through the first critic, as in the paper.
            self.critic.requires_grad_(False)
            q1_pi, _ = self.critic(batch.obs, self.actor(batch.obs))
            self.critic.requires_grad_(True)
            actor_loss = -q1_pi.mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            metrics["actor_loss"] = actor_loss.item()

            soft_update(self.actor_target, self.actor, config.tau)
            soft_update(self.critic_target, self.critic, config.tau)
        return metrics

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.num_updates = state["num_updates"]
