#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


SIGMA_X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _bell_fields():
    time = phx.domain.TimeInterval(0.0, 1.0)
    zero = time.Function()(jnp.asarray([1.0, 0.0], dtype=complex))
    one = time.Function()(jnp.asarray([0.0, 1.0], dtype=complex))
    bell_state = (
        phx.operators.tensor_product(zero, zero)
        + phx.operators.tensor_product(one, one)
    ) / jnp.sqrt(2.0)

    @time.Function()
    def bell_factor():
        return bell_state.func()[:, None]

    density = phx.operators.density_from_factor(bell_factor)
    return time, bell_state, density


def test_bell_state_reduces_to_maximally_mixed_subsystems():
    _time, _state, density = _bell_fields()
    reduced_a = phx.operators.partial_trace(
        density,
        subsystem_dims=(2, 2),
        trace_out=1,
    )
    reduced_b = phx.operators.partial_trace(
        density,
        subsystem_dims=(2, 2),
        trace_out=0,
    )
    value_a = eqx.filter_jit(reduced_a.func)()
    value_b = reduced_b.func()
    expected = 0.5 * jnp.eye(2)

    assert jnp.allclose(value_a, expected, atol=1e-12)
    assert jnp.allclose(value_b, expected, atol=1e-12)
    assert jnp.allclose(jnp.trace(density.func()), 1.0, atol=1e-12)
    assert jnp.allclose(value_a, jnp.conj(value_a.T), atol=1e-12)
    assert jnp.all(jnp.linalg.eigvalsh(value_a) >= -1e-12)
    assert jnp.all(jnp.linalg.eigvalsh(value_b) >= -1e-12)


def test_bell_state_local_expectations_vanish_and_correlations_are_unit():
    time, _state, density = _bell_fields()
    sigma_x = time.Function()(SIGMA_X)
    sigma_y = time.Function()(SIGMA_Y)
    sigma_z = time.Function()(SIGMA_Z)

    for pauli in (sigma_x, sigma_y, sigma_z):
        first = phx.operators.embed_operator(
            pauli,
            subsystem=0,
            subsystem_dims=(2, 2),
        )
        second = phx.operators.embed_operator(
            pauli,
            subsystem=1,
            subsystem_dims=(2, 2),
        )
        assert jnp.allclose(phx.operators.density_expectation(density, first).func(), 0.0)
        assert jnp.allclose(phx.operators.density_expectation(density, second).func(), 0.0)

    zz = phx.operators.tensor_product(sigma_z, sigma_z)
    assert jnp.allclose(phx.operators.density_expectation(density, zz).func(), 1.0)


def test_partial_trace_is_jittable_and_parameter_differentiable():
    time = phx.domain.TimeInterval(0.0, 1.0)

    def reduced_ground_population(theta):
        state = jnp.asarray(
            [jnp.cos(theta), 0.0, 0.0, jnp.sin(theta)],
            dtype=complex,
        )
        density = phx.operators.density_from_factor(time.Function()(state[:, None]))
        reduced = phx.operators.partial_trace(
            density,
            subsystem_dims=(2, 2),
            trace_out=1,
        )
        return jnp.real(reduced.func()[0, 0])

    theta = 0.37
    derivative = jax.jit(jax.grad(reduced_ground_population))(theta)
    assert jnp.allclose(derivative, -jnp.sin(2.0 * theta), atol=1e-12)
