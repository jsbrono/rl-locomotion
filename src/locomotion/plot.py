"""Plot evaluation curves, averaged over seeds.

Usage::

    python -m locomotion.plot runs/*-ant-seed* --out docs/ant.png --title Ant-v5

Run directories named ``<name>-seed<k>`` are grouped by ``<name>``. Each line is the mean
evaluation return across seeds, with a band of one standard deviation across seeds.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def read_eval(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with (run_dir / "eval.csv").open() as f:
        rows = list(csv.DictReader(f))
    return (
        np.array([int(r["step"]) for r in rows]),
        np.array([float(r["return_mean"]) for r in rows]),
    )


def group_runs(run_dirs: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for run_dir in sorted(run_dirs):
        groups[re.sub(r"-seed\d+$", "", run_dir.name)].append(run_dir)
    return dict(groups)


def seed_statistics(runs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(steps, mean, std)`` over the evaluation steps every run has reached."""
    curves = [read_eval(run) for run in runs]
    length = min(len(steps) for steps, _ in curves)
    steps = curves[0][0][:length]
    for other, _ in curves[1:]:
        if not np.array_equal(other[:length], steps):
            raise ValueError(f"runs in {runs[0].parent} were evaluated at different steps")
    returns = np.stack([r[:length] for _, r in curves])
    return steps, returns.mean(0), returns.std(0)


def plot(run_dirs: list[Path], out: Path, title: str | None = None) -> None:
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    for name, runs in group_runs(run_dirs).items():
        steps, mean, std = seed_statistics(runs)
        (line,) = ax.plot(steps, mean, label=f"{name} ({len(runs)} seeds)")
        ax.fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=0.2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("evaluation return")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation curves over seeds.")
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--out", type=Path, default=Path("curves.png"))
    parser.add_argument("--title")
    args = parser.parse_args(argv)
    plot(args.runs, args.out, args.title)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
