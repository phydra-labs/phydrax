"""Adam history agrees with independent closed-form moments under strict promotion."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax import Array
from jax.typing import DTypeLike

from phydrax.optim import adam


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


def test_adam_history_matches_closed_form_moments_under_jit() -> None:
    rate, first_decay, second_decay, epsilon = 0.03, 0.7, 0.8, 0.05
    optimizer = adam(rate, b1=first_decay, b2=second_decay, eps=epsilon)
    parameters = jnp.asarray([0.5, -0.4], dtype=jnp.float64)
    state = optimizer.init(parameters)
    expected = np.asarray(parameters)
    first = np.zeros((2,), dtype=np.float64)
    second = np.zeros((2,), dtype=np.float64)
    gradients = np.asarray([[2.0, -1.0], [0.5, 3.0], [-2.0, 1.0]], dtype=np.float64)
    update = jax.jit(optimizer.update)
    for count, gradient in enumerate(gradients, start=1):
        first = first_decay * first + (1.0 - first_decay) * gradient
        second = second_decay * second + (1.0 - second_decay) * gradient**2
        corrected_first = first / (1.0 - first_decay**count)
        corrected_second = second / (1.0 - second_decay**count)
        expected = expected - rate * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )
        updates, state = update(
            jnp.asarray(gradient, dtype=parameters.dtype), state, parameters
        )
        if not isinstance(updates, Array):
            raise AssertionError(
                "An array gradient must retain array update coordinates."
            )
        parameters = eqx.apply_updates(parameters, updates)
        np.testing.assert_allclose(parameters, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [
        (jnp.float64, 1.0e-12),
        (jnp.float32, 2.0e-6),
        (jnp.bfloat16, 0.0),
        (jnp.float16, 0.0),
    ],
)
def test_adam_default_decays_preserve_precise_history(
    dtype: DTypeLike, tolerance: float
) -> None:
    optimizer = adam(0.03)
    parameter = jnp.asarray(0.5, dtype=dtype)
    state = optimizer.init(params=parameter)
    update = jax.jit(optimizer.update)
    first, second = 0.0, 0.0
    expected_parameter = 0.5
    compute_dtype = jnp.float64 if parameter.dtype == jnp.float64 else jnp.float32
    moment_tolerance = 1.0e-12 if compute_dtype == jnp.float64 else 2.0e-6
    for count, gradient in enumerate((2.0, 0.5, -1.0, 3.0), start=1):
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        expected_update = (
            -0.03
            * (first / (1.0 - 0.9**count))
            / (np.sqrt(second / (1.0 - 0.999**count)) + 1.0e-8)
        )
        rounded_update = float(
            np.asarray(jnp.asarray(expected_update, dtype=dtype), dtype=np.float64)
        )
        updates, state = update(
            updates=jnp.asarray(gradient, dtype=dtype), state=state, params=parameter
        )
        if not isinstance(updates, Array):
            raise AssertionError("A scalar array gradient must return an array update.")
        assert updates.dtype == parameter.dtype
        assert np.isfinite(np.asarray(updates, dtype=np.float64))
        assert float(np.asarray(updates, dtype=np.float64)) != 0.0
        np.testing.assert_allclose(
            np.asarray(updates, dtype=np.float64),
            rounded_update,
            rtol=tolerance,
            atol=0.0,
        )
        if not isinstance(state, tuple) or len(state) != 2:
            raise AssertionError("Adam must preserve its two-transform chain state.")
        moments, scale_state = state
        if not isinstance(moments, optax.ScaleByAdamState):
            raise AssertionError("Adam must retain the owning moment state.")
        assert isinstance(scale_state, optax.EmptyState)
        assert moments.count.dtype == jnp.int32
        assert moments.count.shape == ()
        assert int(moments.count) == count
        assert moments.mu.dtype == compute_dtype
        assert moments.nu.dtype == compute_dtype
        np.testing.assert_allclose(moments.mu, first, rtol=moment_tolerance, atol=0.0)
        np.testing.assert_allclose(moments.nu, second, rtol=moment_tolerance, atol=0.0)
        parameter = eqx.apply_updates(parameter, updates)
        expected_parameter = float(
            np.asarray(
                jnp.asarray(expected_parameter + rounded_update, dtype=dtype),
                dtype=np.float64,
            )
        )
        np.testing.assert_allclose(
            np.asarray(parameter, dtype=np.float64),
            expected_parameter,
            rtol=tolerance,
            atol=0.0,
        )


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32, jnp.bfloat16, jnp.float16])
def test_adam_zero_gradient_stays_finite_and_zero(dtype: DTypeLike) -> None:
    optimizer = adam(0.03)
    parameter = jnp.asarray([0.0, 1.0], dtype=dtype)
    state = optimizer.init(params=parameter)
    update = jax.jit(optimizer.update)
    for _ in range(3):
        updates, state = update(
            updates=jnp.zeros_like(parameter), state=state, params=parameter
        )
        np.testing.assert_array_equal(
            np.asarray(updates, dtype=np.float64), np.zeros((2,), dtype=np.float64)
        )
        parameter = eqx.apply_updates(parameter, updates)
    np.testing.assert_array_equal(
        np.asarray(parameter, dtype=np.float64), np.asarray([0.0, 1.0], dtype=np.float64)
    )


@pytest.mark.parametrize(
    ("dtype", "tolerance"), [(jnp.complex64, 2.0e-6), (jnp.complex128, 1.0e-12)]
)
def test_adam_complex_history_uses_real_squared_norm(
    dtype: DTypeLike, tolerance: float
) -> None:
    optimizer = adam(0.03)
    parameter = jnp.asarray(0.5 + 0.25j, dtype=dtype)
    state = optimizer.init(params=parameter)
    update = jax.jit(optimizer.update)
    first, second = 0.0j, 0.0
    for count, gradient in enumerate((2.0 + 1.0j, -0.5 + 3.0j, 1.0 - 2.0j), start=1):
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * abs(gradient) ** 2
        expected_update = (
            -0.03
            * (first / (1.0 - 0.9**count))
            / (np.sqrt(second / (1.0 - 0.999**count)) + 1.0e-8)
        )
        updates, state = update(
            updates=jnp.asarray(gradient, dtype=dtype), state=state, params=parameter
        )
        if not isinstance(updates, Array):
            raise AssertionError("A complex array gradient must return an array update.")
        assert updates.dtype == parameter.dtype
        np.testing.assert_allclose(updates, expected_update, rtol=tolerance, atol=0.0)
        if not isinstance(state, tuple) or not isinstance(
            state[0], optax.ScaleByAdamState
        ):
            raise AssertionError("Adam must retain its owning moment state.")
        moments = state[0]
        assert moments.mu.dtype == parameter.dtype
        assert moments.nu.dtype == jnp.real(parameter).dtype
        assert moments.count.dtype == jnp.int32
        assert int(moments.count) == count
        np.testing.assert_allclose(moments.nu, second, rtol=tolerance, atol=0.0)
        parameter = eqx.apply_updates(parameter, updates)


def test_adam_zero_decays_follow_current_gradient_under_jit() -> None:
    optimizer = adam(0.1, b1=0.0, b2=0.0, eps=0.25)
    parameter = jnp.asarray(0.5, dtype=jnp.float64)
    state = optimizer.init(params=parameter)
    update = jax.jit(optimizer.update)
    for gradient in (2.0, -3.0, 0.0):
        updates, state = update(
            updates=jnp.asarray(gradient, dtype=parameter.dtype),
            state=state,
            params=parameter,
        )
        expected = -0.1 * gradient / (abs(gradient) + 0.25)
        np.testing.assert_allclose(updates, expected, rtol=1.0e-12, atol=0.0)
        parameter = eqx.apply_updates(parameter, updates)
