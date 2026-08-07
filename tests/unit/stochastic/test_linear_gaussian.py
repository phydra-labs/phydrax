import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_exact_lti_ou_brownian_and_affine_offset():
    ou = phx.stochastic.LinearGaussianDynamics(
        jnp.asarray([[-0.7]]),
        jnp.asarray([[1.3]]),
        state_shape=(1,),
        offset=jnp.asarray([0.4]),
        dynamics_id="ou-dynamics",
    )
    parameters = ou.parameters(0.2, 1.1)
    duration = 0.9
    expected_transition = jnp.exp(-0.7 * duration)
    expected_offset = 0.4 * (1.0 - expected_transition) / 0.7
    expected_covariance = 1.3**2 * (1.0 - jnp.exp(-1.4 * duration)) / 1.4
    assert jnp.allclose(parameters.transition[0, 0], expected_transition)
    assert jnp.allclose(parameters.offset[0], expected_offset)
    assert jnp.allclose(parameters.covariance[0, 0], expected_covariance)
    assert jnp.allclose(ou(0.0, jnp.asarray([2.0]), None), jnp.asarray([-1.0]))

    brownian = phx.stochastic.LinearGaussianDynamics(
        jnp.zeros((2, 2)),
        jnp.asarray([[2.0, 0.0], [0.0, 0.5]]),
        state_shape=(2,),
        offset=jnp.asarray([1.0, -2.0]),
    )
    brownian_parameters = brownian.discretize(2.0, 2.25)
    assert jnp.allclose(brownian_parameters.transition, jnp.eye(2))
    assert jnp.allclose(brownian_parameters.offset, jnp.asarray([0.25, -0.5]))
    assert jnp.allclose(
        brownian_parameters.covariance, jnp.diag(jnp.asarray([1.0, 0.0625]))
    )


def test_exact_lti_nonnormal_semigroup_and_zero_duration():
    dynamics = phx.stochastic.LinearGaussianDynamics(
        jnp.asarray([[-1.0, 4.0], [0.0, -1.0]]),
        jnp.asarray([[1.0], [0.25]]),
        state_shape=(2,),
        offset=jnp.asarray([0.3, -0.2]),
    )
    duration = 0.6
    direct = dynamics.parameters(0.0, duration)
    expected = jnp.exp(-duration) * jnp.asarray(
        [[1.0, 4.0 * duration], [0.0, 1.0]]
    )
    assert jnp.allclose(direct.transition, expected, rtol=1e-11, atol=1e-11)

    first = dynamics.parameters(0.0, 0.2)
    second = dynamics.parameters(0.2, duration)
    assert jnp.allclose(
        direct.transition, second.transition @ first.transition, atol=1e-11
    )
    assert jnp.allclose(
        direct.offset,
        second.transition @ first.offset + second.offset,
        atol=1e-11,
    )
    assert jnp.allclose(
        direct.covariance,
        second.transition @ first.covariance @ second.transition.T
        + second.covariance,
        atol=1e-11,
    )

    zero = dynamics.parameters(3.0, 3.0)
    derivatives = jax.jacfwd(
        lambda end: dynamics.parameters(0.0, end)
    )(0.0)
    assert jnp.allclose(derivatives.transition, dynamics.drift_matrix)
    assert jnp.allclose(derivatives.offset, dynamics.offset)
    assert jnp.allclose(
        derivatives.covariance,
        dynamics.dispersion @ dynamics.dispersion.T,
    )
    assert jnp.array_equal(zero.transition, jnp.eye(2))
    assert jnp.array_equal(zero.offset, jnp.zeros(2))
    assert jnp.array_equal(zero.covariance, jnp.zeros((2, 2)))


def test_exact_lti_is_jittable_vmappable_and_differentiable():
    dynamics = phx.stochastic.LinearGaussianDynamics(
        jnp.asarray([[-0.4]]),
        jnp.asarray([[0.8]]),
        state_shape=(1,),
    )
    compiled = jax.jit(lambda start, end: dynamics.parameters(start, end))(0.0, 0.75)
    batched = jax.vmap(dynamics.parameters)(
        jnp.zeros(3), jnp.asarray([0.0, 0.5, 1.0])
    )
    derivative = jax.grad(
        lambda end: dynamics.parameters(0.0, end).covariance[0, 0]
    )(0.7)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(compiled))
    assert batched.transition.shape == (3, 1, 1)
    assert jnp.array_equal(batched.covariance[0], jnp.zeros((1, 1)))
    assert jnp.allclose(derivative, 0.8**2 * jnp.exp(-0.8 * 0.7))


def test_singular_transition_sampling_log_density_and_provenance():
    dynamics = phx.stochastic.LinearGaussianDynamics(
        jnp.zeros((2, 2)),
        jnp.asarray([[1.0], [0.0]]),
        state_shape=(2,),
        dynamics_id="rank-one",
        process_id="rank-one-process",
        approximation_id="exact-rank-one",
    )
    kernel = phx.stochastic.LinearGaussianTransitionKernel(dynamics)
    sample = kernel.sample(jr.key(2), jnp.asarray([0.0, 3.0]), 0.0, 0.5)
    assert sample.valid
    assert sample.values[1] == 3.0
    assert jnp.isfinite(kernel.log_prob(jnp.asarray([0.2, 3.0]), jnp.asarray([0.0, 3.0]), 0.0, 0.5))
    assert kernel.log_prob(
        jnp.asarray([0.2, 3.1]), jnp.asarray([0.0, 3.0]), 0.0, 0.5
    ) == -jnp.inf
    assert kernel.log_prob(
        jnp.asarray([0.0, 3.0]), jnp.asarray([0.0, 3.0]), 1.0, 1.0
    ) == 0.0
    assert kernel.process_id == "rank-one-process"
    assert kernel.approximation_id == "exact-rank-one"
    assert kernel.parameterization_id == "rank-one"
    assert kernel.resolved_method == dynamics.resolved_method

    with pytest.raises(TypeError, match="offset must be omitted"):
        phx.stochastic.LinearGaussianTransitionKernel(
            dynamics, offset=jnp.ones(2)
        )

def test_legacy_transition_constructor_uses_one_parameterization_object():
    kernel = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.eye(2, dtype=jnp.int32),
        jnp.diag(jnp.asarray([1, 2], dtype=jnp.int32)),
        state_shape=(2,),
        offset=jnp.asarray([1, -1], dtype=jnp.int32),
    )
    transition, offset, covariance = kernel.parameters(0.0, 1.0)
    assert isinstance(
        kernel.parameterization, phx.stochastic.LinearGaussianParameterization
    )
    assert jnp.array_equal(transition, kernel.transition)
    assert jnp.array_equal(offset, kernel.offset)
    assert jnp.array_equal(covariance, kernel.covariance)
    assert jnp.issubdtype(covariance.dtype, jnp.inexact)
    assert kernel.sample(jr.key(8), jnp.zeros(2), 0.0, 1.0).valid
    assert jnp.isfinite(kernel.log_prob(jnp.zeros(2), jnp.zeros(2), 0.0, 1.0))
