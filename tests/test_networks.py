import pytest
import torch
from torch import nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from locomotion.networks import (
    DeterministicActor,
    GaussianPolicy,
    SquashedGaussianActor,
    TwinQNetwork,
    ValueNetwork,
    mlp,
    soft_update,
)


def test_mlp_layers():
    net = mlp(3, 2, (8, 8))
    assert [type(m) for m in net] == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    assert net(torch.zeros(5, 3)).shape == (5, 2)


def test_squashed_gaussian_log_prob_matches_torch_tanh_distribution():
    torch.manual_seed(0)
    actor = SquashedGaussianActor(4, 3, (16, 16))
    obs = torch.randn(256, 4)
    action, log_prob = actor(obs)

    mean, log_std = actor.gaussian(obs)
    reference = TransformedDistribution(Normal(mean, log_std.exp()), TanhTransform(cache_size=1))
    # Compare away from |a| = 1, where torch's own atanh round trip loses precision.
    ok = action.abs().max(-1).values < 0.999
    expected = reference.log_prob(action).sum(-1)
    torch.testing.assert_close(log_prob[ok], expected[ok], atol=1e-3, rtol=1e-4)


def test_squashed_gaussian_bounds_and_determinism():
    torch.manual_seed(0)
    actor = SquashedGaussianActor(4, 3, (16,))
    with torch.no_grad():
        actor.mean.bias.fill_(50.0)  # far into tanh saturation
    obs = torch.randn(64, 4)
    action, log_prob = actor(obs)
    assert action.abs().max() <= 1.0
    assert torch.isfinite(log_prob).all()
    a1, _ = actor(obs, deterministic=True)
    a2, _ = actor(obs, deterministic=True)
    torch.testing.assert_close(a1, a2)


def test_deterministic_actor_and_critic_shapes():
    actor, critic = DeterministicActor(5, 2, (8,)), TwinQNetwork(5, 2, (8,))
    obs = torch.randn(7, 5) * 100
    action = actor(obs)
    assert action.shape == (7, 2) and action.abs().max() <= 1.0
    q1, q2 = critic(obs, action)
    assert q1.shape == q2.shape == (7,)
    assert not torch.equal(q1, q2)  # independently initialised


def test_gaussian_policy_initialisation():
    policy = GaussianPolicy(5, 3, (64, 64), init_log_std=-0.5)
    dist = policy.distribution(torch.randn(10, 5))
    assert dist.mean.abs().max() < 0.1  # small output gain: starts near zero mean
    torch.testing.assert_close(dist.stddev, torch.full((10, 3), torch.tensor(-0.5).exp().item()))
    assert ValueNetwork(5, (8,))(torch.zeros(4, 5)).shape == (4,)


@pytest.mark.parametrize("tau", [0.0, 0.25, 1.0])
def test_soft_update(tau):
    torch.manual_seed(0)
    source, target = nn.Linear(3, 3), nn.Linear(3, 3)
    before = target.weight.detach().clone()
    soft_update(target, source, tau)
    torch.testing.assert_close(target.weight, (1 - tau) * before + tau * source.weight)
