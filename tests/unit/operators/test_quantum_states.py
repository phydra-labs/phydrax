#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


SIGMA_X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def test_pauli_expectations_norm_and_variance():
    time = phx.domain.TimeInterval(0.0, 1.0)
    state = time.Function()(jnp.asarray([1.0, 1.0], dtype=complex) / jnp.sqrt(2.0))
    sigma_x = time.Function()(SIGMA_X)
    sigma_y = time.Function()(SIGMA_Y)
    sigma_z = time.Function()(SIGMA_Z)

    assert jnp.allclose(phx.operators.state_norm_residual(state).func(), 0.0)
    assert jnp.allclose(phx.operators.state_expectation(state, sigma_x).func(), 1.0)
    assert jnp.allclose(phx.operators.state_expectation(state, sigma_y).func(), 0.0)
    assert jnp.allclose(phx.operators.state_expectation(state, sigma_z).func(), 0.0)
    assert jnp.allclose(phx.operators.observable_variance(state, sigma_z).func(), 1.0)


def test_state_and_density_expectations_agree_for_pure_state():
    time = phx.domain.TimeInterval(0.0, 1.0)

    @time.Function("t")
    def state(t):
        return jnp.asarray([jnp.exp(-0.5j * t), jnp.exp(0.5j * t)]) / jnp.sqrt(2.0)

    @time.Function("t")
    def factor(t):
        return state.func(t)[:, None]

    sigma_x = time.Function()(SIGMA_X)
    density = phx.operators.density_from_factor(factor)
    state_value = phx.operators.state_expectation(state, sigma_x)
    density_value = phx.operators.density_expectation(density, sigma_x)

    point = 0.37
    assert jnp.allclose(state_value.func(point), jnp.cos(point), atol=1e-12)
    assert jnp.allclose(density_value.func(point), state_value.func(point), atol=1e-12)


def test_rectangular_density_factor_is_physical():
    time = phx.domain.TimeInterval(0.0, 1.0)
    factor_value = jnp.asarray(
        [[1.0 + 1.0j, 0.2], [0.3j, 1.4], [0.5, -0.7j]],
        dtype=complex,
    )
    density = phx.operators.density_from_factor(time.Function()(factor_value))
    value = eqx.filter_jit(density.func)()

    assert jnp.allclose(value, jnp.conj(value.T), atol=1e-12)
    assert jnp.allclose(jnp.trace(value), 1.0, atol=1e-12)
    assert jnp.all(jnp.linalg.eigvalsh(value) >= -1e-12)
    assert jnp.allclose(phx.operators.hermiticity_residual(density).func(), 0.0)
    assert jnp.allclose(phx.operators.unit_trace_residual(density).func(), 0.0)


def test_density_factorization_is_jittable_and_parameter_differentiable():
    time = phx.domain.TimeInterval(0.0, 1.0)
    sigma_z = time.Function()(SIGMA_Z)

    def expectation(theta):
        factor = time.Function()(jnp.diag(jnp.asarray([theta, 1.0])))
        density = phx.operators.density_from_factor(factor)
        value = phx.operators.density_expectation(density, sigma_z).func()
        return jnp.real(value)

    theta = 0.7
    derivative = jax.jit(jax.grad(expectation))(theta)
    expected = 4.0 * theta / (theta**2 + 1.0) ** 2
    assert jnp.allclose(derivative, expected, atol=1e-12)


def test_density_from_factor_rejects_zero_and_invalid_shapes():
    time = phx.domain.TimeInterval(0.0, 1.0)
    zero = phx.operators.density_from_factor(time.Function()(jnp.zeros((2, 1))))
    invalid = phx.operators.density_from_factor(time.Function()(jnp.ones((2,))))

    with pytest.raises((eqx.EquinoxRuntimeError, ValueError), match="nonzero Frobenius norm"):
        zero.func()
    with pytest.raises(eqx.EquinoxRuntimeError, match="nonzero Frobenius norm"):
        eqx.filter_jit(zero.func)()
    with pytest.raises(ValueError, match=r"shape \(n, r\)"):
        invalid.func()


def test_quantum_state_operators_validate_value_dimensions():
    time = phx.domain.TimeInterval(0.0, 1.0)
    scalar = time.Function()(1.0)
    state = time.Function()(jnp.ones((2,)))
    larger_observable = time.Function()(jnp.eye(3))
    density = time.Function()(jnp.eye(2) / 2.0)

    with pytest.raises(ValueError, match="quantum state must be a vector"):
        phx.operators.state_norm_residual(scalar).func()
    with pytest.raises(ValueError, match="dimensions must match"):
        phx.operators.state_expectation(state, larger_observable).func()
    with pytest.raises(ValueError, match="dimensions must match"):
        phx.operators.density_expectation(density, larger_observable).func()
