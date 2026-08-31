import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from examples.differentiable_cosmological_particle_mesh import build_workflow


def test_two_lpt_particle_mesh_workflow_is_differentiable_end_to_end():
    background, growth, provenance, lpt, rollout, gravity, white_noise = build_workflow()
    k = jnp.linspace(1.0, 30.0, 96)
    first_growth = growth.evaluate(0.1)[0]

    def objective(amplitude):
        base = amplitude / (1.0 + (k / 8.0) ** 2)
        power = phx.applications.cosmology.MatterPowerTable(
            [0.1, 1.0],
            k,
            jnp.stack((first_growth**2 * base, base)),
            background.scale,
            provenance,
        )
        initial = lpt.realize(background, growth, power, white_noise, 0.1)
        evolved = rollout.rollout(background, initial.state)
        density, _ = gravity.density(evolved.state.positions)
        contrast = density.density / jnp.mean(density.density) - 1.0
        return jnp.mean(contrast**2), (initial, evolved)

    amplitude = jnp.asarray(1.0e-7)
    (value, (initial, evolved)), derivative = jax.value_and_grad(objective, has_aux=True)(
        amplitude
    )
    epsilon = jnp.asarray(1.0e-9)
    finite_difference = (
        objective(amplitude + epsilon)[0] - objective(amplitude - epsilon)[0]
    ) / (2.0 * epsilon)

    assert bool(initial.successful)
    assert bool(evolved.successful)
    assert int(evolved.diagnostics.accepted_steps) == 2
    assert evolved.diagnostics.maximum_mass_balance_defect < 1e-10
    assert evolved.diagnostics.maximum_net_force_norm < 1e-8
    assert jnp.isfinite(value)
    assert jnp.isfinite(derivative)
    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=2e-2,
        atol=1e-6,
    )
