# Locomotion: SAC, TD3 and PPO from scratch

Three deep reinforcement learning algorithms written from scratch in PyTorch, trained to make
MuJoCo robots walk (Gymnasium's `Ant-v5` and `Humanoid-v5`):

- **SAC**, Soft Actor-Critic (Haarnoja et al. 2018, 2019): off-policy, maximum-entropy, with a
  learned temperature.
- **TD3**, Twin Delayed DDPG (Fujimoto et al. 2018): off-policy, deterministic policy.
- **PPO**, Proximal Policy Optimisation (Schulman et al. 2017): on-policy, clipped objective.

This is my own rewrite of a six-person university project at Bath that I led; the original
team code is not included. Each algorithm is one short file that follows its paper, and the
tests check the parts that are easy to get wrong.

## How the algorithms differ

| | SAC | TD3 | PPO |
|---|---|---|---|
| Data | replay buffer (1M) | replay buffer (1M) | fresh 2048-step rollouts |
| Policy | tanh-squashed Gaussian | deterministic, plus Gaussian exploration noise | Gaussian, learned state-independent std |
| Critic | twin Q, soft Bellman target | twin Q, clipped noise on the target action | state value, GAE(λ) advantages |
| Exploration | entropy bonus, temperature tuned to a target entropy | fixed action noise | the policy's own std |
| Updates | 1 gradient step per env step | 1 critic step per env step, actor every 2 | 10 epochs over each rollout |

## Implementation details

These are the details that most often break these algorithms without any error:

- **Time limits are not terminal states.** A MuJoCo episode that hits its 1000-step limit is
  *truncated*: the robot could have kept going. SAC and TD3 store only `terminated` in the
  replay buffer. PPO's GAE bootstraps from the value of the last observation after a
  truncation, while still stopping the advantage from leaking into the next episode.
- **SAC's log-probability** includes the tanh change-of-variables term, in a numerically
  stable form. A test checks it against PyTorch's own `TanhTransform` distribution.
- **The actor loss leaves the critics alone.** The critic is frozen while the actor loss is
  computed; a test recomputes the critic gradient independently and compares.
- **TD3** updates the actor and all target networks only every `policy_delay` critic updates,
  and evaluates without exploration noise.
- **PPO** uses the details that matter on MuJoCo (Engstrom et al. 2020; Huang et al. 2022):
  running observation normalisation (saved in the checkpoint and used at evaluation), reward
  scaling by the running std of the discounted return, per-minibatch advantage
  normalisation, orthogonal initialisation, gradient clipping and linear learning-rate decay.
- **Evaluation is separate from training.** Every 10k steps the policy runs 10 episodes on its
  own seeded environment with deterministic actions (SAC's mean action, TD3 without noise,
  PPO's mean). Returns of the noisy training episodes are logged too, but not used to compare.

## Results

Not benchmarked yet. Each config trains with one command, and `locomotion.plot` averages runs
over seeds. Results will go here as:

| Algorithm | Ant-v5, return at 1M steps | Humanoid-v5, return at 1M steps |
|---|---|---|
| SAC | | |
| TD3 | | |
| PPO | | |

On `Ant-v5`, a robot that only stands still scores about 1,000, because the environment pays
1 per step for staying upright. Walking policies score several thousand.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch                                    # pick your build from pytorch.org
pip install -e ".[dev]"                              # ".[dev,video]" to record videos
```

**Train:**

```bash
python -m locomotion.train --config configs/sac-ant.yaml
python -m locomotion.train --config configs/ppo-ant.yaml --seed 1
python -m locomotion.train --config configs/td3-humanoid.yaml --total-steps 300000
```

Each run writes `config.yaml`, `episodes.csv` (every training episode), `eval.csv` (the
evaluation return every `eval_every` steps) and `model.pt` to `runs/<config>-seed<seed>/`.
The networks are small, so the default device is the CPU; pass `--device cuda` for a GPU.

**Evaluate, watch, record and plot:**

```bash
python -m locomotion.evaluate runs/sac-ant-seed0/model.pt --episodes 20
python -m locomotion.evaluate runs/sac-ant-seed0/model.pt --render
python -m locomotion.evaluate runs/sac-ant-seed0/model.pt --video videos/sac-ant
python -m locomotion.plot runs/sac-ant-seed* runs/td3-ant-seed* runs/ppo-ant-seed* --out ant.png

pytest
```

Configs are YAML. Anything left out takes the default from the agent's config dataclass
(`SACConfig`, `TD3Config`, `PPOConfig`). Any Gymnasium environment with continuous actions
works, for example `HalfCheetah-v5` or `Walker2d-v5`.

## Layout

```
configs/                    SAC, TD3 and PPO on Ant-v5 and Humanoid-v5
src/locomotion/
  sac.py, td3.py, ppo.py    the three agents
  networks.py               MLPs, actors, twin Q-network, soft target updates
  buffers.py                replay buffer and GAE
  normalization.py          running observation normalisation and reward scaling
  envs.py                   environment wrappers (actions rescaled to [-1, 1])
  config.py                 YAML config and checkpoints
  train.py                  training loops and CLI
  evaluate.py               deterministic evaluation, rendering and video CLI
  plot.py                   learning curves averaged over seeds
tests/                      unit tests, plus end-to-end runs on a tiny environment
```

The tests include a small "reach the target" task (`tests/conftest.py`) that every agent must
learn within seconds. A broken update then shows up as a failing test, not as a flat learning
curve hours into a MuJoCo run.

## References

- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
  with a Stochastic Actor*, 2018; and *Soft Actor-Critic Algorithms and Applications*, 2019.
- Fujimoto, van Hoof and Meger, *Addressing Function Approximation Error in Actor-Critic
  Methods*, 2018.
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017; and *High-Dimensional
  Continuous Control Using Generalized Advantage Estimation*, 2016.
- Engstrom et al., *Implementation Matters in Deep Policy Gradients*, 2020.
- Huang et al., *The 37 Implementation Details of Proximal Policy Optimization*, 2022.
