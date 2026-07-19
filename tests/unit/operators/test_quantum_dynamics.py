#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


SIGMA_X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def test_two_level_schrodinger_residual_is_zero():
    time = phx.domain.TimeInterval(0.0, 1.0)
    omega = 1.7
    hamiltonian = time.Function()(0.5 * omega * SIGMA_Z)

    @time.Function("t")
    def state(t):
        return jnp.asarray([jnp.exp(-0.5j * omega * t), 0.0j])

    residual = phx.operators.schrodinger_residual(state, hamiltonian)
    value = eqx.filter_jit(residual.func)(0.37)
    assert jnp.allclose(value, jnp.zeros((2,), dtype=complex), atol=1e-11)


def test_heisenberg_spin_precession_residual_is_zero():
    time = phx.domain.TimeInterval(0.0, 1.0)
    omega = 1.3
    hamiltonian = time.Function()(0.5 * omega * SIGMA_Z)

    @time.Function("t")
    def observable(t):
        return jnp.cos(omega * t) * SIGMA_X - jnp.sin(omega * t) * SIGMA_Y

    residual = phx.operators.heisenberg_residual(observable, hamiltonian)
    assert jnp.allclose(residual.func(0.29), jnp.zeros((2, 2)), atol=1e-11)


def test_von_neumann_density_evolution_preserves_structure():
    time = phx.domain.TimeInterval(0.0, 1.0)
    omega = 0.9
    hamiltonian = time.Function()(0.5 * omega * SIGMA_Z)

    @time.Function("t")
    def density(t):
        return 0.5 * (
            jnp.eye(2, dtype=complex)
            + jnp.cos(omega * t) * SIGMA_X
            + jnp.sin(omega * t) * SIGMA_Y
        )

    residual = phx.operators.von_neumann_residual(density, hamiltonian)
    assert jnp.allclose(residual.func(0.41), jnp.zeros((2, 2)), atol=1e-11)
    assert jnp.allclose(phx.operators.unit_trace_residual(density).func(0.41), 0.0)
    assert jnp.allclose(phx.operators.hermiticity_residual(density).func(0.41), 0.0)


def test_schrodinger_residual_accepts_differential_hamiltonian_action():
    space = phx.domain.Interval1d(-1.0, 1.0)
    time = phx.domain.TimeInterval(0.0, 1.0)
    spacetime = space @ time
    wave_number = 1.3
    mass = 0.8
    hbar = 0.7
    frequency = hbar * wave_number**2 / (2.0 * mass)

    @spacetime.Function("x", "t")
    def wave(x, t):
        return jnp.exp(1j * (wave_number * x[0] - frequency * t))

    def free_particle_action(state):
        return -(hbar**2 / (2.0 * mass)) * phx.operators.laplacian(state, var="x")

    residual = phx.operators.schrodinger_residual(
        wave,
        free_particle_action,
        hbar=hbar,
    )
    assert jnp.allclose(residual.func(jnp.asarray([0.2]), 0.31), 0.0, atol=1e-10)


def test_quantum_dynamics_validates_state_and_action_contracts():
    time = phx.domain.TimeInterval(0.0, 1.0)
    hamiltonian = time.Function()(jnp.eye(2))
    static_state = time.Function()(jnp.ones((2,)))

    with pytest.raises(ValueError, match="must depend on time_var"):
        phx.operators.schrodinger_residual(static_state, hamiltonian)

    @time.Function("t")
    def state(t):
        return jnp.asarray([jnp.exp(-1j * t), 0.0j])

    bad_residual = phx.operators.schrodinger_residual(
        state,
        lambda _state: time.Function()(1.0),
    )
    with pytest.raises(ValueError, match="must match the state shape"):
        bad_residual.func(0.2)
