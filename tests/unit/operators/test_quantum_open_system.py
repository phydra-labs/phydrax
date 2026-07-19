#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


SIGMA_Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
LOWERING = jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def test_amplitude_damping_lindblad_residual_is_zero():
    time = phx.domain.TimeInterval(0.0, 2.0)
    rate = 0.8
    hamiltonian = time.Function()(jnp.zeros((2, 2), dtype=complex))
    collapse = time.Function()(jnp.sqrt(rate) * LOWERING)

    @time.Function("t")
    def density(t):
        excited = jnp.exp(-rate * t)
        return jnp.asarray(
            [[1.0 - excited, 0.0], [0.0, excited]],
            dtype=complex,
        )

    residual = phx.operators.lindblad_residual(density, hamiltonian, collapse)
    assert jnp.allclose(eqx.filter_jit(residual.func)(0.43), 0.0, atol=1e-11)
    assert density.func(1.2)[1, 1] < density.func(0.2)[1, 1]


def test_pure_dephasing_lindblad_residual_is_zero():
    time = phx.domain.TimeInterval(0.0, 2.0)
    rate = 0.6
    hamiltonian = time.Function()(jnp.zeros((2, 2), dtype=complex))
    collapse = time.Function()(jnp.sqrt(rate / 2.0) * SIGMA_Z)

    @time.Function("t")
    def density(t):
        coherence = jnp.exp(-rate * t)
        return 0.5 * jnp.asarray(
            [[1.0, coherence], [coherence, 1.0]],
            dtype=complex,
        )

    residual = phx.operators.lindblad_residual(density, hamiltonian, (collapse,))
    assert jnp.allclose(residual.func(0.61), 0.0, atol=1e-11)


def test_lindblad_dissipator_preserves_trace_and_hermiticity():
    time = phx.domain.TimeInterval(0.0, 1.0)
    factor = time.Function()(
        jnp.asarray([[1.0 + 0.2j, 0.3], [0.4j, 0.8 - 0.1j]], dtype=complex)
    )
    density = phx.operators.density_from_factor(factor)
    lowering = time.Function()(0.7 * LOWERING)
    dephasing = time.Function()(0.2 * SIGMA_Z)

    combined = phx.operators.lindblad_dissipator(
        density,
        (lowering, dephasing),
    )
    separate = phx.operators.lindblad_dissipator(
        density,
        lowering,
    ) + phx.operators.lindblad_dissipator(density, dephasing)
    value = combined.func()

    assert jnp.allclose(value, separate.func(), atol=1e-12)
    assert jnp.allclose(jnp.trace(value), 0.0, atol=1e-12)
    assert jnp.allclose(value, jnp.conj(value.T), atol=1e-12)


def test_empty_lindblad_collection_reduces_to_von_neumann_dynamics():
    time = phx.domain.TimeInterval(0.0, 1.0)
    hamiltonian = time.Function()(0.5 * SIGMA_Z)

    @time.Function("t")
    def density(t):
        return 0.5 * jnp.asarray(
            [
                [1.0, jnp.exp(-1.0j * t)],
                [jnp.exp(1.0j * t), 1.0],
            ]
        )

    dissipator = phx.operators.lindblad_dissipator(density, ())
    lindblad = phx.operators.lindblad_residual(density, hamiltonian, ())
    von_neumann = phx.operators.von_neumann_residual(density, hamiltonian)

    assert jnp.allclose(dissipator.func(0.3), jnp.zeros((2, 2)))
    assert jnp.allclose(lindblad.func(0.3), von_neumann.func(0.3), atol=1e-12)


def test_lindblad_dissipator_is_parameter_differentiable():
    time = phx.domain.TimeInterval(0.0, 1.0)
    excited_density = time.Function()(
        jnp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    )

    def ground_population_rate(rate):
        collapse = time.Function()(jnp.sqrt(rate) * LOWERING)
        value = phx.operators.lindblad_dissipator(
            excited_density,
            collapse,
        ).func()
        return jnp.real(value[0, 0])

    derivative = jax.jit(jax.grad(ground_population_rate))(0.7)
    assert jnp.allclose(derivative, 1.0, atol=1e-12)


def test_lindblad_operators_validate_collections_and_dimensions():
    time = phx.domain.TimeInterval(0.0, 1.0)
    density = time.Function()(jnp.eye(2) / 2.0)
    larger = time.Function()(jnp.eye(3))
    rectangular = time.Function()(jnp.ones((2, 3)))

    with pytest.raises(TypeError, match="contain only DomainFunctions"):
        phx.operators.lindblad_dissipator(density, [object()])
    with pytest.raises(TypeError, match="DomainFunction or a sequence"):
        phx.operators.lindblad_dissipator(density, object())
    with pytest.raises(ValueError, match="square matrix"):
        phx.operators.lindblad_dissipator(density, rectangular).func()
    with pytest.raises(ValueError, match="dimensions must match"):
        phx.operators.lindblad_dissipator(density, larger).func()
