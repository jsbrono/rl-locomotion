import numpy as np
import pytest

from locomotion.buffers import ReplayBuffer, compute_gae
from locomotion.normalization import ObservationNormalizer, RewardScaler, RunningMeanStd


def fill(buffer: ReplayBuffer, n: int) -> None:
    for i in range(n):
        buffer.add(np.full(2, i), np.full(1, i), float(i), np.full(2, i + 1), i % 2 == 0)


def test_replay_buffer_samples_only_filled_entries():
    buffer = ReplayBuffer(2, 1, capacity=10, seed=0)
    with pytest.raises(ValueError):
        buffer.sample(4)
    fill(buffer, 3)
    batch = buffer.sample(500)
    assert len(buffer) == 3
    assert set(batch.rewards.tolist()) == {0.0, 1.0, 2.0}
    # Fields of one transition stay together.
    assert (batch.obs[:, 0] == batch.rewards).all()
    assert (batch.next_obs[:, 0] == batch.rewards + 1).all()
    assert (batch.terminated == (batch.rewards % 2 == 0).float()).all()


def test_replay_buffer_overwrites_oldest():
    buffer = ReplayBuffer(2, 1, capacity=3, seed=0)
    fill(buffer, 5)
    assert len(buffer) == 3
    assert sorted(buffer.rewards.tolist()) == [2.0, 3.0, 4.0]
    assert buffer.position == 2


def brute_force_gae(rewards, values, next_values, terminated, episode_end, gamma, lam):
    """sum_k (gamma * lam)^k * delta_{t+k}, up to the end of the episode or rollout."""
    deltas = rewards + gamma * (1 - terminated) * next_values - values
    advantages = []
    for t in range(len(rewards)):
        total, k = 0.0, t
        while True:
            total += (gamma * lam) ** (k - t) * deltas[k]
            if episode_end[k] or k == len(rewards) - 1:
                break
            k += 1
        advantages.append(total)
    return np.array(advantages)


@pytest.mark.parametrize("seed", range(5))
def test_gae_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 40
    terminated = rng.random(n) < 0.1
    episode_end = terminated | (rng.random(n) < 0.1)
    args = (rng.normal(size=n), rng.normal(size=n), rng.normal(size=n), terminated, episode_end)
    advantages, returns = compute_gae(*args, gamma=0.9, gae_lambda=0.8)
    np.testing.assert_allclose(advantages, brute_force_gae(*args, 0.9, 0.8), atol=1e-5)
    np.testing.assert_allclose(returns, advantages + args[1], atol=1e-5)


def test_gae_terminal_and_truncated_steps():
    rewards, values = np.array([1.0, 1.0, 1.0]), np.zeros(3)
    next_values = np.array([10.0, 10.0, 10.0])
    ends = np.array([False, True, False])
    # Step 1 terminated: no bootstrap, and step 0 does not see past it.
    adv, _ = compute_gae(rewards, values, next_values, ends, ends, gamma=0.5, gae_lambda=1.0)
    np.testing.assert_allclose(adv, [(1 + 0.5 * 10) + 0.5 * 1.0, 1.0, 1 + 0.5 * 10])
    # Step 1 truncated: it still bootstraps from V(s_2'), but nothing flows across.
    adv, _ = compute_gae(rewards, values, next_values, np.zeros(3, bool), ends, 0.5, 1.0)
    np.testing.assert_allclose(adv[:2], [(1 + 0.5 * 10) + 0.5 * (1 + 0.5 * 10), 1 + 0.5 * 10])


def test_gae_with_lambda_one_is_the_monte_carlo_return_minus_value():
    rewards = np.array([1.0, 2.0, 3.0])
    values = np.array([0.5, -1.0, 2.0])
    next_values = np.array([-1.0, 2.0, 4.0])  # V(s_1), V(s_2), and a bootstrap for s_3
    no = np.zeros(3, bool)
    _, returns = compute_gae(rewards, values, next_values, no, no, 0.9, 1.0)
    expected0 = 1 + 0.9 * 2 + 0.81 * 3 + 0.729 * 4
    np.testing.assert_allclose(returns[0], expected0, rtol=1e-6)


def test_running_mean_std_merges_batches():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(1000, 4))
    stats = RunningMeanStd((4,))
    for chunk in np.array_split(data, 7):
        stats.update(chunk)
    np.testing.assert_allclose(stats.mean.numpy(), data.mean(0))
    np.testing.assert_allclose(stats.var.numpy(), data.var(0))
    assert stats.count.item() == 1000


def test_observation_normalizer():
    rng = np.random.default_rng(0)
    normalizer = ObservationNormalizer(3, clip=5.0)
    data = rng.normal([10.0, -5.0, 0.0], [1.0, 3.0, 0.1], size=(2000, 3))
    for row in data:
        normalizer(row, update=True)
    normalized = normalizer(data)
    np.testing.assert_allclose(normalized.mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(0), 1, atol=1e-3)
    # Without update the statistics are frozen, and outliers are clipped.
    count = normalizer.stats.count.item()
    assert normalizer(np.array([1e6, 0, 0]))[0] == 5.0
    assert normalizer.stats.count.item() == count


def test_reward_scaler_tracks_discounted_return_per_episode():
    scaler = RewardScaler(gamma=0.5)
    scaler(1.0, episode_end=False)
    assert scaler.discounted_return == 1.0
    scaler(1.0, episode_end=True)
    assert scaler.discounted_return == 0.0  # reset for the next episode
    scaled = scaler(2.0, episode_end=False)
    returns = np.array([1.0, 1.5, 2.0])
    assert scaled == pytest.approx(2.0 / np.sqrt(returns.var() + 1e-8))
