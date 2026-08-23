#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_stormer_verlet_preserves_harmonic_energy_and_is_differentiable():
    initial_position = jnp.asarray([1.0])
    initial_momentum = jnp.asarray([0.0])

    def terminal(step_size):
        result = phx.solver.integrate_stormer_verlet(
            initial_position,
            initial_momentum,
            lambda position: position,
            lambda momentum: momentum,
            step_size=step_size,
            steps=100,
        )
        energy = 0.5 * (jnp.sum(result.position**2) + jnp.sum(result.momentum**2))
        return energy, result

    energy, result = terminal(0.05)
    derivative = jax.grad(lambda step: terminal(step)[0])(0.05)

    assert jnp.abs(energy - 0.5) < 5e-4
    assert result.method_id == "stormer-verlet"
    assert jnp.isfinite(derivative)
