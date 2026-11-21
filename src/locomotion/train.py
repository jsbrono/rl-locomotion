"""Train an agent.

Usage::

    python -m locomotion.train --config configs/sac-ant.yaml
    python -m locomotion.train --config configs/ppo-ant.yaml --seed 1 --total-steps 200000

Writes to ``runs/<config>-seed<seed>/`` (or ``--out``):

- ``config.yaml``, the resolved config;
- ``episodes.csv``, the return and length of every training episode;
- ``eval.csv``, the mean and standard deviation of the return over ``eval_episodes``
  deterministic episodes, every ``eval_every`` steps, on a separately seeded environment;
- ``model.pt``, the agent at the latest evaluation.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from locomotion.buffers import ReplayBuffer
from locomotion.config import Agent, Config, load_config, make_agent, save_checkpoint
from locomotion.envs import make_env, space_dims
from locomotion.evaluate import evaluate_policy


class RunLogger:
    """Writes the CSV logs, runs evaluations and saves checkpoints."""

    def __init__(self, out_dir: Path, agent: Agent, config: Config, eval_env: gym.Env):
        self.out_dir, self.agent, self.config, self.eval_env = out_dir, agent, config, eval_env
        self.dims = space_dims(eval_env)
        self.recent_returns: list[float] = []
        self.start = time.perf_counter()
        self.history: list[dict[str, float]] = []
        self._episodes = (out_dir / "episodes.csv").open("w", newline="")
        self._episodes_csv = csv.writer(self._episodes)
        self._episodes_csv.writerow(["step", "return", "length"])
        self._eval = (out_dir / "eval.csv").open("w", newline="")
        self._eval_csv = csv.writer(self._eval)
        self._eval_csv.writerow(["step", "return_mean", "return_std"])

    def episode(self, step: int, episode_return: float, length: int) -> None:
        self._episodes_csv.writerow([step, f"{episode_return:.3f}", length])
        self.recent_returns.append(episode_return)

    def evaluate(self, step: int, metrics: dict[str, float]) -> None:
        returns = evaluate_policy(self.agent, self.eval_env, self.config.train.eval_episodes)
        mean, std = float(returns.mean()), float(returns.std())
        self._eval_csv.writerow([step, f"{mean:.3f}", f"{std:.3f}"])
        self._eval.flush()
        self._episodes.flush()
        self.history.append({"step": step, "return_mean": mean, "return_std": std})
        save_checkpoint(self.agent, self.config, *self.dims, self.out_dir / "model.pt")

        train_return = np.mean(self.recent_returns) if self.recent_returns else float("nan")
        self.recent_returns.clear()
        elapsed = time.perf_counter() - self.start
        shown = "  ".join(f"{k} {v:.3g}" for k, v in metrics.items())
        print(
            f"step {step:>9,}  eval {mean:9.1f} +/- {std:7.1f}  train {train_return:9.1f}  "
            f"({step / elapsed:,.0f} steps/s)  {shown}"
        )

    def close(self) -> None:
        self._episodes.close()
        self._eval.close()


def train_off_policy(agent: Agent, env: gym.Env, config: Config, logger: RunLogger) -> None:
    """SAC and TD3: one gradient step per environment step, after ``learning_starts`` steps of
    uniformly random actions."""
    train = config.train
    obs_dim, act_dim = space_dims(env)
    buffer = ReplayBuffer(obs_dim, act_dim, train.buffer_size, seed=train.seed)
    rng = np.random.default_rng(train.seed)
    metrics: dict[str, float] = {}

    obs, _ = env.reset()
    for step in range(1, train.total_steps + 1):
        if step <= train.learning_starts:
            action = rng.uniform(-1.0, 1.0, act_dim).astype(np.float32)
        else:
            action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, float(reward), next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            logger.episode(step, float(info["episode"]["r"]), int(info["episode"]["l"]))
            obs, _ = env.reset()

        if step > train.learning_starts:
            metrics = agent.update(buffer.sample(agent.config.batch_size, agent.device))
        if step % train.eval_every == 0 or step == train.total_steps:
            logger.evaluate(step, metrics)


def train_on_policy(agent: Agent, env: gym.Env, config: Config, logger: RunLogger) -> None:
    """PPO: alternate between collecting ``rollout_steps`` steps and updating on them. Training
    stops at the first whole rollout reaching ``total_steps``."""
    train, rollout_steps = config.train, config.agent.rollout_steps
    step = 0
    while step < train.total_steps:
        rollout, episodes = agent.collect(env, rollout_steps)
        for episode in episodes:
            logger.episode(step + episode["step"], episode["return"], episode["length"])
        metrics = agent.update(rollout, progress=step / train.total_steps)

        previous, step = step, step + rollout_steps
        if step // train.eval_every > previous // train.eval_every or step >= train.total_steps:
            logger.evaluate(step, metrics)


def run(config: Config, out_dir: Path, device: torch.device) -> list[dict[str, float]]:
    """Trains with ``config``, writing logs and the model to ``out_dir``. Returns the
    evaluation history."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    seed = config.train.seed
    torch.manual_seed(seed)

    env = make_env(config.env_id, seed=seed, **config.env_kwargs)
    eval_env = make_env(config.env_id, seed=seed + 10_000, **config.env_kwargs)
    agent = make_agent(config, *space_dims(env), device)
    logger = RunLogger(out_dir, agent, config, eval_env)
    print(f"{config.algo.upper()} on {config.env_id}, seed {seed}; writing to {out_dir}")
    try:
        loop = train_on_policy if agent.on_policy else train_off_policy
        loop(agent, env, config, logger)
    finally:
        logger.close()
        env.close()
        eval_env.close()
    return logger.history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train SAC, TD3 or PPO.")
    parser.add_argument("--config", required=True, type=Path, help="YAML config file")
    parser.add_argument("--seed", type=int, help="overrides train.seed")
    parser.add_argument("--total-steps", type=int, help="overrides train.total_steps")
    parser.add_argument("--out", type=Path, help="output directory (default: runs/<name>)")
    parser.add_argument(
        "--device", default="cpu", help="torch device; the MLPs are small, so CPU is usually fine"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    overrides = {"seed": args.seed, "total_steps": args.total_steps}
    config.train = replace(config.train, **{k: v for k, v in overrides.items() if v is not None})
    out_dir = args.out or Path("runs") / f"{args.config.stem}-seed{config.train.seed}"
    run(config, out_dir, torch.device(args.device))


if __name__ == "__main__":
    main()
