"""Run configuration (YAML), agent construction and checkpoints.

A config file looks like::

    algo: sac                 # sac | td3 | ppo
    env:
      id: Ant-v5
      kwargs: {}              # passed to gymnasium.make
    train:
      total_steps: 1000000
      seed: 0
    agent:                    # fields of SACConfig, TD3Config or PPOConfig
      lr: 0.0003

Anything left out takes the dataclass default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import yaml

from locomotion.ppo import PPO, PPOConfig
from locomotion.sac import SAC, SACConfig
from locomotion.td3 import TD3, TD3Config

ALGORITHMS = {"sac": SAC, "td3": TD3, "ppo": PPO}
Agent = SAC | TD3 | PPO
AgentConfig = SACConfig | TD3Config | PPOConfig


@dataclass
class TrainConfig:
    total_steps: int = 1_000_000
    eval_every: int = 10_000
    eval_episodes: int = 10
    seed: int = 0
    # Off-policy only (SAC, TD3):
    learning_starts: int = 10_000  # uniform-random actions, and no updates, for this many steps
    buffer_size: int = 1_000_000


@dataclass
class Config:
    algo: str
    env_id: str
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    train: TrainConfig = field(default_factory=TrainConfig)
    agent: AgentConfig | None = None

    def __post_init__(self) -> None:
        if self.algo not in ALGORITHMS:
            raise ValueError(f"unknown algo {self.algo!r}; expected one of {list(ALGORITHMS)}")
        config_cls = ALGORITHMS[self.algo].Config
        if self.agent is None:
            self.agent = config_cls()
        elif not isinstance(self.agent, config_cls):
            raise TypeError(f"{self.algo} needs a {config_cls.__name__}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        unknown = set(data) - {"algo", "env", "train", "agent"}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        algo = data["algo"]
        if algo not in ALGORITHMS:
            raise ValueError(f"unknown algo {algo!r}; expected one of {list(ALGORITHMS)}")
        env = data["env"]
        return cls(
            algo=algo,
            env_id=env["id"],
            env_kwargs=dict(env.get("kwargs") or {}),
            train=_build(TrainConfig, data.get("train")),
            agent=_build(ALGORITHMS[algo].Config, data.get("agent")),
        )

    def to_dict(self) -> dict[str, Any]:
        agent = asdict(self.agent)
        agent["hidden_sizes"] = list(agent["hidden_sizes"])
        return {
            "algo": self.algo,
            "env": {"id": self.env_id, "kwargs": self.env_kwargs},
            "train": asdict(self.train),
            "agent": agent,
        }


def _build(cls: type, values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    if "hidden_sizes" in values:
        values["hidden_sizes"] = tuple(values["hidden_sizes"])
    return cls(**values)


def load_config(path: str | Path) -> Config:
    return Config.from_dict(yaml.safe_load(Path(path).read_text()))


def make_agent(config: Config, obs_dim: int, act_dim: int, device: torch.device) -> Agent:
    return ALGORITHMS[config.algo](obs_dim, act_dim, config.agent, device)


def save_checkpoint(agent: Agent, config: Config, obs_dim: int, act_dim: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "agent": agent.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[Agent, Config]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = Config.from_dict(checkpoint["config"])
    agent = make_agent(config, checkpoint["obs_dim"], checkpoint["act_dim"], torch.device(device))
    agent.load_state_dict(checkpoint["agent"])
    return agent, config
