"""Policy and value networks.

Every agent works with actions in ``[-1, 1]``; :func:`locomotion.envs.make_env` rescales them to
the environment's real bounds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def mlp(
    in_dim: int,
    out_dim: int,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for size in hidden_sizes:
        layers += [nn.Linear(in_dim, size), activation()]
        in_dim = size
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class TwinQNetwork(nn.Module):
    """Two independent Q-functions ``Q(s, a)``. Taking the minimum of the two counters the
    overestimation bias of a single critic (clipped double Q-learning, Fujimoto et al. 2018)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden_sizes)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden_sizes)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class DeterministicActor(nn.Module):
    """``a = tanh(f(s))``, used by TD3."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden_sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class SquashedGaussianActor(nn.Module):
    """A Gaussian squashed through ``tanh``, used by SAC (Haarnoja et al. 2018).

    The log-probability includes the change-of-variables correction for the ``tanh``:
    ``log pi(a|s) = log N(u; mu, sigma) - sum log(1 - tanh(u)^2)``, computed in a numerically
    stable form.
    """

    LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_sizes[-1], hidden_sizes[:-1])
        self.mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], act_dim)

    def gaussian(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log standard deviation of the Gaussian before squashing."""
        hidden = F.relu(self.trunk(obs))
        log_std = self.log_std(hidden).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return self.mean(hidden), log_std

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(action, log_prob)``; with ``deterministic`` the action is ``tanh(mean)``."""
        mean, log_std = self.gaussian(obs)
        std = log_std.exp()

        u = mean if deterministic else mean + std * torch.randn_like(mean)
        gaussian_log_prob = -0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        # log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))
        tanh_correction = 2 * (math.log(2) - u - F.softplus(-2 * u))
        log_prob = (gaussian_log_prob - tanh_correction).sum(-1)
        return torch.tanh(u), log_prob


class GaussianPolicy(nn.Module):
    """A diagonal Gaussian with a state-independent, learned log standard deviation, used by
    PPO. Actions are unbounded; the environment clips them to ``[-1, 1]``."""

    def __init__(
        self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int], init_log_std: float = 0.0
    ):
        super().__init__()
        self.mean = mlp(obs_dim, act_dim, hidden_sizes, activation=nn.Tanh)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        orthogonal_init(self.mean, output_gain=0.01)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp())


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = mlp(obs_dim, 1, hidden_sizes, activation=nn.Tanh)
        orthogonal_init(self.net, output_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def orthogonal_init(net: nn.Sequential, output_gain: float) -> None:
    """Orthogonal weights with gain sqrt(2) in hidden layers and zero biases, the initialisation
    commonly used for PPO (Andrychowicz et al. 2021). A small output gain for the policy starts
    it close to zero mean."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linears):
        nn.init.orthogonal_(layer.weight, output_gain if i == len(linears) - 1 else math.sqrt(2))
        nn.init.zeros_(layer.bias)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """``target <- (1 - tau) * target + tau * source`` (Polyak averaging)."""
    for t, s in zip(target.parameters(), source.parameters(), strict=True):
        t.lerp_(s, tau)
