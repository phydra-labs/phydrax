"""Qualification evidence for native FLRW growth, 2LPT, and scale-factor PM."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

import phydrax as phx
from examples.differentiable_cosmological_particle_mesh import build_workflow


def main() -> None:
    background, growth, provenance, lpt, rollout, gravity, white_noise = build_workflow()
    eds = phx.applications.cosmology.FLRWBackground(1.0, 1.0, scale=background.scale)
    eds_nodes = jnp.geomspace(1.0e-2, 1.0, 64)
    eds_growth = phx.applications.cosmology.FLRWGrowthPlan(eds_nodes).solve(eds)
    k = jnp.linspace(1.0, 30.0, 96)
    first_growth = growth.evaluate(0.1)[0]

    def objective(amplitude):
        base = amplitude / (1.0 + (k / 8.0) ** 2)
        power = phx.applications.cosmology.MatterPowerTable(
            jnp.asarray([0.1, 1.0]),
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
    value, tangent, (initial, evolved) = jax.jvp(
        objective,
        (amplitude,),
        (jnp.asarray(1.0),),
        has_aux=True,
    )
    epsilon = jnp.asarray(1.0e-9)
    finite_difference = (
        objective(amplitude + epsilon)[0] - objective(amplitude - epsilon)[0]
    ) / (2.0 * epsilon)
    expected_second = (3.0 / 7.0) * eds_nodes**2
    report = {
        "eds_first_growth_max_error": float(
            jnp.max(jnp.abs(eds_growth.first_order_growth - eds_nodes))
        ),
        "eds_first_rate_max_error": float(
            jnp.max(jnp.abs(eds_growth.first_order_rate - 1.0))
        ),
        "eds_second_growth_max_error": float(
            jnp.max(jnp.abs(eds_growth.second_order_growth - expected_second))
        ),
        "initial_successful": bool(initial.successful),
        "rollout_completed": bool(evolved.successful),
        "accepted_steps": int(evolved.diagnostics.accepted_steps),
        "maximum_mass_balance_defect": float(
            evolved.diagnostics.maximum_mass_balance_defect
        ),
        "maximum_net_force_norm": float(evolved.diagnostics.maximum_net_force_norm),
        "density_variance": float(value),
        "directional_derivative": float(tangent),
        "finite_difference_derivative": float(finite_difference),
        "derivative_residual": float(jnp.abs(tangent - finite_difference)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
