import copy

import numpy as np
import pytest
import torch

from locomotion.buffers import Batch
from locomotion.envs import make_env
from locomotion.ppo import PPO, PPOConfig, Rollout
from locomotion.sac import SAC
from locomotion.td3 import TD3

CPU = torch.device("cpu")
OBS_DIM, ACT_DIM = 3, 2


def random_batch(n: int = 32) -> Batch:
    return Batch(
        obs=torch.randn(n, OBS_DIM),
        actions=torch.rand(n, ACT_DIM) * 2 - 1,
        rewards=torch.randn(n),
        next_obs=torch.randn(n, OBS_DIM),
        terminated=(torch.rand(n) < 0.2).float(),
    )


def make(algo, **config):
    cls = {"sac": SAC, "td3": TD3, "ppo": PPO}[algo]
    return cls(OBS_DIM, ACT_DIM, cls.Config(hidden_sizes=(16, 16), **config), CPU)


def params(module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in module.parameters()]).clone()


@pytest.mark.parametrize("algo", ["sac", "td3", "ppo"])
def test_act_shapes_and_state_dict_round_trip(algo):
    torch.manual_seed(0)
    agent, fresh = make(algo), make(algo)
    obs = np.random.default_rng(0).normal(size=OBS_DIM).astype(np.float32)
    action = agent.act(obs)
    assert action.shape == (ACT_DIM,) and action.dtype == np.float32

    fresh.load_state_dict(copy.deepcopy(agent.state_dict()))
    np.testing.assert_allclose(
        fresh.act(obs, deterministic=True), agent.act(obs, deterministic=True)
    )


def test_sac_update_moves_target_by_tau():
    torch.manual_seed(0)
    agent = make("sac", tau=0.1)
    target_before = params(agent.critic_target)
    metrics = agent.update(random_batch())
    expected = 0.9 * target_before + 0.1 * params(agent.critic)
    torch.testing.assert_close(params(agent.critic_target), expected)
    assert all(np.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize(("target_entropy", "direction"), [(100.0, 1), (-100.0, -1)])
def test_sac_temperature_follows_entropy_target(target_entropy, direction):
    torch.manual_seed(0)
    agent = make("sac", target_entropy=target_entropy)
    for _ in range(5):
        agent.update(random_batch())
    # Entropy below target raises alpha (more exploration); above target lowers it.
    assert np.sign(agent.alpha.item() - 1.0) == direction


def test_sac_fixed_temperature():
    agent = make("sac", alpha=0.2)
    agent.update(random_batch())
    assert agent.alpha.item() == pytest.approx(0.2)


def test_sac_actor_loss_leaves_no_gradient_on_the_critic():
    torch.manual_seed(0)
    agent = make("sac")
    reference = copy.deepcopy(agent)
    batch = random_batch()

    torch.manual_seed(1)
    agent.update(batch)
    # The critic loss alone, recomputed with the same random draws.
    torch.manual_seed(1)
    with torch.no_grad():
        next_actions, next_log_prob = reference.actor(batch.next_obs)
        soft_value = torch.min(*reference.critic_target(batch.next_obs, next_actions))
        target = batch.rewards + 0.99 * (1 - batch.terminated) * (soft_value - next_log_prob)
    q1, q2 = reference.critic(batch.obs, batch.actions)
    (torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)).backward()

    for got, expected in zip(agent.critic.parameters(), reference.critic.parameters(), strict=True):
        torch.testing.assert_close(got.grad, expected.grad)


def test_td3_delays_actor_and_target_updates():
    torch.manual_seed(0)
    agent = make("td3", policy_delay=3)
    actor, actor_target, critic_target = (
        params(agent.actor),
        params(agent.actor_target),
        params(agent.critic_target),
    )
    for _ in range(2):
        metrics = agent.update(random_batch())
        assert "actor_loss" not in metrics
    torch.testing.assert_close(params(agent.actor), actor)
    torch.testing.assert_close(params(agent.actor_target), actor_target)
    torch.testing.assert_close(params(agent.critic_target), critic_target)

    assert "actor_loss" in agent.update(random_batch())
    assert not torch.equal(params(agent.actor), actor)
    assert not torch.equal(params(agent.critic_target), critic_target)


def test_td3_exploration_noise():
    torch.manual_seed(0)
    agent = make("td3", exploration_noise=0.5)
    obs = np.zeros(OBS_DIM, np.float32)
    clean = agent.act(obs, deterministic=True)
    noisy = np.stack([agent.act(obs) for _ in range(200)])
    assert np.abs(noisy).max() <= 1.0
    assert noisy.std(0).min() > 0.2
    np.testing.assert_allclose(agent.act(obs, deterministic=True), clean)


def test_ppo_collect_continues_episodes_across_calls(toy_env_id):
    torch.manual_seed(0)
    env = make_env(toy_env_id, seed=0)
    agent = PPO(2, 1, PPOConfig(hidden_sizes=(8,), normalize_obs=False, scale_rewards=False), CPU)
    first, episodes = agent.collect(env, 45)  # 30-step episodes, unless one terminates early
    second, _ = agent.collect(env, 45)

    # Within an episode, each next observation is the following step's observation.
    for t in range(44):
        if not first.episode_end[t]:
            np.testing.assert_array_equal(first.next_obs[t], first.obs[t + 1])
    if not first.episode_end[-1]:
        np.testing.assert_array_equal(first.next_obs[-1], second.obs[0])
    # Episode returns reported are the sums of the rewards in the rollout.
    ends = np.flatnonzero(first.episode_end)
    assert [e["step"] for e in episodes] == list(ends + 1)
    assert episodes[0]["return"] == pytest.approx(first.rewards[: ends[0] + 1].sum(), rel=1e-5)
    # Log-probabilities are those of the stored actions under the collecting policy.
    dist = agent.policy.distribution(torch.as_tensor(first.obs))
    stored = torch.as_tensor(first.log_probs)
    torch.testing.assert_close(dist.log_prob(torch.as_tensor(first.actions)).sum(-1), stored)


def test_ppo_update_anneals_learning_rate_and_stays_near_old_policy(toy_env_id):
    torch.manual_seed(0)
    env = make_env(toy_env_id, seed=0)
    agent = PPO(2, 1, PPOConfig(hidden_sizes=(16,), epochs=4, minibatch_size=32), CPU)
    rollout, _ = agent.collect(env, 256)
    metrics = agent.update(rollout, progress=0.75)
    assert metrics["lr"] == pytest.approx(agent.config.lr * 0.25)
    assert 0 <= metrics["clip_fraction"] <= 1
    assert metrics["approx_kl"] < 0.1


def test_ppo_clipped_objective_has_no_gradient_outside_the_trust_region():
    torch.manual_seed(0)
    agent = make("ppo", epochs=1, minibatch_size=4, normalize_advantages=False, value_coef=0.0)
    obs = np.zeros((4, OBS_DIM), np.float32)
    actions = np.full((4, ACT_DIM), 0.5, np.float32)
    with torch.no_grad():
        log_prob = agent.policy.distribution(torch.as_tensor(obs)).log_prob(
            torch.as_tensor(actions)
        )
    # Pretend the old policy made these actions far less likely: the ratio is about e^2,
    # beyond 1 + clip_range, so with positive advantages the clipped term is constant.
    rollout = Rollout(
        obs=obs,
        actions=actions,
        log_probs=(log_prob.sum(-1) - 2.0).numpy(),
        rewards=np.ones(4, np.float32),
        next_obs=obs,
        terminated=np.ones(4, bool),
        episode_end=np.ones(4, bool),
    )
    before = params(agent.policy)
    agent.update(rollout)
    torch.testing.assert_close(params(agent.policy), before)
