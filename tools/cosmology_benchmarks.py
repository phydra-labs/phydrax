"""Compile and steady-state benchmarks for native cosmology workflows."""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx
from examples.differentiable_cosmological_particle_mesh import build_workflow


def _block(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _timed(function, *args):
    start = time.perf_counter()
    value = function(*args)
    _block(value)
    return value, time.perf_counter() - start


def main() -> None:
    background, growth, provenance, lpt, rollout, gravity, white_noise = build_workflow()
    k = jnp.linspace(1.0, 30.0, 96)
    first_growth = growth.evaluate(0.1)[0]
    base = 1.0e-7 / (1.0 + (k / 8.0) ** 2)
    power = phx.applications.cosmology.MatterPowerTable(
        jnp.asarray([0.1, 1.0]),
        k,
        jnp.stack((first_growth**2 * base, base)),
        background.scale,
        provenance,
    )

    initial_function = jax.jit(
        lambda noise: lpt.realize(background, growth, power, noise, 0.1)
    )
    initial, initial_compile = _timed(initial_function, white_noise)
    _, initial_steady = _timed(initial_function, white_noise)

    rollout_function = jax.jit(lambda state: rollout.rollout(background, state))
    evolved, rollout_compile = _timed(rollout_function, initial.state)
    _, rollout_steady = _timed(rollout_function, initial.state)

    def statistic(amplitude):
        values = power.power_values * (amplitude / 1.0e-7)
        scaled = phx.applications.cosmology.MatterPowerTable(
            power.scale_factors,
            power.wavenumbers,
            values,
            power.scale,
            power.provenance,
        )
        state = lpt.realize(background, growth, scaled, white_noise, 0.1).state
        result = rollout.rollout(background, state)
        density, _ = gravity.density(result.state.positions)
        contrast = density.density / jnp.mean(density.density) - 1.0
        return jnp.mean(contrast**2)

    gradient_function = jax.jit(jax.value_and_grad(statistic))
    _, gradient_compile = _timed(gradient_function, jnp.asarray(1.0e-7))
    gradient_result, gradient_steady = _timed(gradient_function, jnp.asarray(1.0e-7))

    report = {
        "shape": list(lpt.shape),
        "particles": lpt.particles.capacity,
        "steps": int(rollout.scale_factors.size - 1),
        "lpt_compile_seconds": initial_compile,
        "lpt_steady_seconds": initial_steady,
        "rollout_compile_seconds": rollout_compile,
        "rollout_steady_seconds": rollout_steady,
        "gradient_compile_seconds": gradient_compile,
        "gradient_steady_seconds": gradient_steady,
        "accepted_steps": int(evolved.diagnostics.accepted_steps),
        "maximum_mass_balance_defect": float(
            evolved.diagnostics.maximum_mass_balance_defect
        ),
        "maximum_net_force_norm": float(evolved.diagnostics.maximum_net_force_norm),
        "statistic": float(gradient_result[0]),
        "statistic_gradient": float(gradient_result[1]),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
