import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_diagonal_normal_law_preserves_sample_batch_and_event_axes():
    law = phx.uq.DiagonalNormalLaw(
        jnp.asarray([[0.0, 1.0], [2.0, 3.0]]),
        jnp.asarray([0.5, 2.0]),
        event_shape=(2,),
    )
    samples = law.sample(jr.key(0), (4, 3))
    value = law.location
    score = law.score(value)

    assert law.batch_shape == (2,)
    assert law.event_shape == (2,)
    assert samples.shape == (4, 3, 2, 2)
    assert law.log_prob(value).shape == (2,)
    assert jnp.array_equal(score, jnp.zeros_like(value))
    assert jnp.all(law.contains(value))


def test_diagonal_normal_score_matches_log_density_gradient():
    law = phx.uq.DiagonalNormalLaw(
        jnp.asarray([0.3, -0.2]),
        jnp.asarray([0.7, 1.4]),
        event_shape=(2,),
    )
    value = jnp.asarray([0.8, -1.1])
    gradient = jax.grad(law.log_prob)(value)

    assert jnp.allclose(law.score(value), gradient, rtol=1e-12, atol=1e-12)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        phx.uq.DiagonalNormalLaw(
            jnp.zeros((2,)),
            jnp.asarray([1.0, 0.0]),
            event_shape=(2,),
        )


def test_variance_preserving_transition_and_score_identities():
    process = phx.stochastic.VariancePreservingDiffusion(
        2,
        beta_minimum=0.2,
        beta_maximum=2.0,
    )
    state = jnp.asarray([0.4, -0.3])
    t0 = 0.1
    tmid = 0.4
    t1 = 0.9
    first_mean = process.transition_mean_scale(t0, tmid)
    second_mean = process.transition_mean_scale(tmid, t1)
    complete_mean = process.transition_mean_scale(t0, t1)
    first_variance = process.transition_scale(t0, tmid) ** 2
    second_variance = process.transition_scale(tmid, t1) ** 2
    complete_variance = process.transition_scale(t0, t1) ** 2
    marginal = process.marginal_transition(state, t0=t0, t1=t1)
    value = marginal.mean + jnp.asarray([0.2, -0.1])

    assert jnp.allclose(first_mean * second_mean, complete_mean, rtol=1e-12)
    assert jnp.allclose(
        second_mean**2 * first_variance + second_variance,
        complete_variance,
        rtol=1e-12,
    )
    assert jnp.allclose(
        process.transition_mean_scale(0.0, t1) ** 2
        + process.transition_scale(0.0, t1) ** 2,
        1.0,
        rtol=1e-12,
    )
    assert jnp.allclose(
        process.conditional_score(value, state, t0=t0, t1=t1),
        marginal.score(value),
    )
    reference = process.asymptotic_terminal_reference()
    assert reference.relationship == "asymptotic"
    assert reference.residual_signal_scale > 0.0


def test_variance_exploding_variance_is_additive_and_has_exact_rate():
    process = phx.stochastic.VarianceExplodingDiffusion(
        3,
        initial_scale=0.02,
        terminal_scale=3.0,
    )
    t0 = 0.1
    tmid = 0.45
    t1 = 0.8
    complete = process.transition_variance(t0, t1)
    composed = process.transition_variance(t0, tmid) + process.transition_variance(
        tmid, t1
    )
    time = jnp.asarray(0.37)
    derivative = jax.grad(lambda current: process.reference_scale(current) ** 2)(time)

    assert jnp.allclose(complete, composed, rtol=1e-12)
    assert jnp.allclose(process.diffusion_scale(time) ** 2, derivative, rtol=1e-12)
    assert process.drift(time, jnp.ones((3,))).shape == (3,)
    reference = process.asymptotic_terminal_reference()
    assert reference.relationship == "asymptotic"
    assert reference.residual_signal_scale == 1.0


def test_gaussian_diffusion_rejects_invalid_intervals_and_parameters():
    process = phx.stochastic.VariancePreservingDiffusion(1)
    with pytest.raises((ValueError, RuntimeError), match="t1 > t0"):
        process.transition_scale(0.2, 0.2)
    with pytest.raises(ValueError, match="beta_maximum"):
        phx.stochastic.VariancePreservingDiffusion(
            1, beta_minimum=2.0, beta_maximum=1.0
        )
    with pytest.raises(ValueError, match="terminal_scale"):
        phx.stochastic.VarianceExplodingDiffusion(
            1, initial_scale=1.0, terminal_scale=1.0
        )
