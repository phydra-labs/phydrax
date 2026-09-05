#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import expm

import phydrax as phx


def _augmented_reference(matrix, state, forcing, duration):
    size = matrix.shape[-1]
    augmented = jnp.zeros((size + 1, size + 1), dtype=matrix.dtype)
    augmented = augmented.at[:size, :size].set(matrix)
    augmented = augmented.at[:size, size].set(forcing)
    initial = jnp.concatenate((state, jnp.ones((1,), dtype=state.dtype)))
    return (expm(duration * augmented) @ initial)[:size]


def test_affine_exponential_step_handles_zero_and_singular_operator():
    matrix = jnp.asarray(((0.0, 1.0), (0.0, 0.0)), dtype=jnp.float64)
    state = jnp.asarray((2.0, 3.0), dtype=jnp.float64)
    forcing = jnp.asarray((1.0, 2.0), dtype=jnp.float64)
    operator = phx.linalg.DenseLinearOperator(matrix)

    result = phx.dynamics.affine_exponential_step(
        operator,
        state,
        forcing,
        jnp.asarray(0.25),
    )
    expected = _augmented_reference(matrix, state, forcing, 0.25)

    np.testing.assert_allclose(result.value, expected, rtol=1e-12, atol=1e-12)
    assert result.successful
    zero = phx.dynamics.affine_exponential_step(operator, state, forcing, 0.0)
    np.testing.assert_array_equal(zero.value, state)


def test_affine_exponential_step_supports_batched_dense_operators():
    matrices = jnp.asarray(
        (((-1.0, 0.0), (0.0, -2.0)), ((0.0, 1.0), (0.0, 0.0))),
        dtype=jnp.float64,
    )
    states = jnp.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=jnp.float64)
    forcing = jnp.asarray(((0.5, 0.25), (1.0, 0.0)), dtype=jnp.float64)
    duration = jnp.asarray((0.1, 0.2), dtype=jnp.float64)

    result = phx.dynamics.affine_exponential_step(
        phx.linalg.DenseLinearOperator(matrices),
        states,
        forcing,
        duration,
    )
    expected = jax.vmap(_augmented_reference)(matrices, states, forcing, duration)

    np.testing.assert_allclose(result.value, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(result.successful, (True, True))


def test_affine_exponential_step_rejects_invalid_duration():
    operator = phx.linalg.DenseLinearOperator(jnp.eye(2, dtype=jnp.float64))
    state = jnp.ones((2,), dtype=jnp.float64)

    with pytest.raises(Exception, match="duration must be non-negative"):
        phx.dynamics.affine_exponential_step(operator, state, state, -1.0)


def test_reverse_parameter_gradient_survives_exact_krylov_breakdown():
    # A one-dimensional affine flow closes its Krylov subspace exactly. The
    # discarded zero residual must not inject sqrt(0) NaNs into reverse AD.
    conductance, duration = 10.0, 500.0
    initial, ambient, heat = 300.0, 280.0, 50.0

    def temperature(log_capacity):
        capacity = 10000.0 * jnp.exp(log_capacity)
        operator = phx.linalg.DenseLinearOperator(
            jnp.reshape(-conductance / capacity, (1, 1))
        )
        forcing = jnp.reshape((conductance * ambient + heat) / capacity, (1,))
        return phx.dynamics.affine_exponential_step(
            operator, jnp.array([initial]), forcing, duration
        ).value[0]

    log_capacity = jnp.asarray(0.2, dtype=jnp.float64)
    exponent = conductance * duration / (10000.0 * np.exp(0.2))
    equilibrium = ambient + heat / conductance
    expected = (initial - equilibrium) * np.exp(-exponent) * exponent
    reverse = jax.grad(temperature)(log_capacity)
    forward = jax.jacfwd(temperature)(log_capacity)
    np.testing.assert_allclose(reverse, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(forward, expected, rtol=1e-10, atol=1e-10)
