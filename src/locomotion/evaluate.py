"""Evaluate a trained agent, optionally rendering it or recording videos.

Usage::

    python -m locomotion.evaluate runs/sac-ant-seed0/model.pt --episodes 20
    python -m locomotion.evaluate runs/sac-ant-seed0/model.pt --video videos/sac-ant
    python -m locomotion.evaluate runs/sac-ant-seed0/model.pt --render

Videos need ``pip install -e ".[video]"``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from locomotion.config import load_checkpoint
from locomotion.envs import make_env


def evaluate_policy(agent, env: gym.Env, episodes: int) -> np.ndarray:
    """Returns of ``episodes`` episodes with the agent's deterministic actions."""
    returns = np.zeros(episodes)
    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
            returns[i] += reward
            done = terminated or truncated
    return returns


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained agent.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--video", type=Path, help="record every episode to this directory")
    parser.add_argument("--render", action="store_true", help="show the episodes in a window")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    agent, config = load_checkpoint(args.checkpoint, args.device)
    env_kwargs = dict(config.env_kwargs)
    if args.render or args.video:
        env_kwargs["render_mode"] = "human" if args.render else "rgb_array"
    env = make_env(config.env_id, seed=args.seed, **env_kwargs)
    if args.video:
        env = gym.wrappers.RecordVideo(env, str(args.video), episode_trigger=lambda _: True)

    returns = evaluate_policy(agent, env, args.episodes)
    env.close()
    print(
        json.dumps(
            {
                "env": config.env_id,
                "algo": config.algo,
                "episodes": args.episodes,
                "return_mean": float(returns.mean()),
                "return_std": float(returns.std()),
                "returns": [round(float(r), 2) for r in returns],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
