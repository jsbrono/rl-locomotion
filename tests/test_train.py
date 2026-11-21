import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from locomotion import evaluate, plot
from locomotion.config import Config, load_checkpoint, load_config, make_agent
from locomotion.envs import make_env, space_dims
from locomotion.evaluate import evaluate_policy
from locomotion.train import main

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))

TINY_AGENTS = {
    "sac": {"hidden_sizes": [64, 64], "batch_size": 64},
    "td3": {"hidden_sizes": [64, 64], "batch_size": 64},
    "ppo": {"hidden_sizes": [64, 64], "rollout_steps": 512, "lr": 0.001},
}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_build_and_step(path):
    config = load_config(path)
    env = make_env(config.env_id, seed=0, **config.env_kwargs)
    agent = make_agent(config, *space_dims(env), torch.device("cpu"))
    obs, _ = env.reset()
    env.step(agent.act(obs))
    env.close()
    assert Config.from_dict(config.to_dict()) == config


def test_config_errors():
    with pytest.raises(ValueError, match="unknown algo"):
        Config.from_dict({"algo": "dqn", "env": {"id": "x"}})
    with pytest.raises(ValueError, match="SACConfig fields"):
        Config.from_dict({"algo": "sac", "env": {"id": "x"}, "agent": {"learning_rate": 1}})
    with pytest.raises(ValueError, match="sections"):
        Config.from_dict({"algo": "sac", "env": {"id": "x"}, "trian": {}})


def write_config(tmp_path: Path, env_id: str, algo: str, **train) -> Path:
    config = {
        "algo": algo,
        "env": {"id": env_id},
        "train": {
            "learning_starts": 300,
            "buffer_size": 5000,
            "eval_every": 500,
            "eval_episodes": 5,
            **train,
        },
        "agent": TINY_AGENTS[algo],
    }
    path = tmp_path / f"{algo}-toy.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("algo", ["sac", "td3", "ppo"])
def test_train_evaluate_and_plot_end_to_end(tmp_path, toy_env_id, algo, capsys):
    config_path = write_config(tmp_path, toy_env_id, algo)
    out = tmp_path / "runs" / f"{algo}-toy-seed3"
    main(["--config", str(config_path), "--out", str(out), "--total-steps", "1000", "--seed", "3"])

    assert yaml.safe_load((out / "config.yaml").read_text())["train"]["seed"] == 3
    episodes = read_csv(out / "episodes.csv")
    assert len(episodes) >= 1000 // 30 and all(int(e["length"]) <= 30 for e in episodes)
    evals = [int(row["step"]) for row in read_csv(out / "eval.csv")]
    assert evals == ([512, 1024] if algo == "ppo" else [500, 1000])

    capsys.readouterr()
    evaluate.main([str(out / "model.pt"), "--episodes", "3"])
    report = json.loads(capsys.readouterr().out)
    assert report["algo"] == algo and len(report["returns"]) == 3

    plot.main([str(out), "--out", str(tmp_path / "curves.png")])
    assert (tmp_path / "curves.png").stat().st_size > 0


def test_checkpoint_reproduces_the_final_evaluation(tmp_path, toy_env_id):
    config_path = write_config(tmp_path, toy_env_id, "td3")
    out = tmp_path / "run"
    main(["--config", str(config_path), "--out", str(out), "--total-steps", "500"])
    agent, config = load_checkpoint(out / "model.pt")
    assert config.algo == "td3"

    env = make_env(toy_env_id, seed=config.train.seed + 10_000)  # the training eval env
    returns = evaluate_policy(agent, env, config.train.eval_episodes)
    logged = float(read_csv(out / "eval.csv")[-1]["return_mean"])
    assert returns.mean() == pytest.approx(logged, abs=1e-3)


# Standing still scores about 17.7 on the toy task and random actions about 15; moving to
# the target scores about 29.
@pytest.mark.parametrize(("algo", "steps"), [("sac", 3000), ("td3", 3000), ("ppo", 8192)])
def test_agents_learn_the_toy_task(tmp_path, toy_env_id, algo, steps):
    config_path = write_config(
        tmp_path, toy_env_id, algo, eval_every=steps, eval_episodes=20, learning_starts=500
    )
    out = tmp_path / "run"
    main(["--config", str(config_path), "--out", str(out), "--total-steps", str(steps)])
    final = float(read_csv(out / "eval.csv")[-1]["return_mean"])
    assert final > 25.0


def test_plot_groups_seeds(tmp_path):
    for seed, offset in [(0, 0.0), (1, 2.0)]:
        run = tmp_path / f"sac-toy-seed{seed}"
        run.mkdir()
        rows = [f"{s},{s / 100 + offset},0" for s in (100, 200, 300)]
        (run / "eval.csv").write_text("\n".join(["step,return_mean,return_std", *rows]) + "\n")
    groups = plot.group_runs(list(tmp_path.iterdir()))
    assert list(groups) == ["sac-toy"]
    steps, mean, std = plot.seed_statistics(groups["sac-toy"])
    np.testing.assert_allclose(steps, [100, 200, 300])
    np.testing.assert_allclose(mean, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(std, [1.0, 1.0, 1.0])
