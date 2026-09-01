"""Qualification evidence for Fourier shells and inverse particle realization."""

from __future__ import annotations

import json

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    cosmology = phx.applications.cosmology
    count = 8
    x = (jnp.arange(count) + 0.5) / count
    field = jnp.cos(2.0 * jnp.pi * x)
    shifted = jnp.roll(field, 1)
    shells = phx.discretization.PeriodicFourierShellPlan(
        (count,),
        (1.0,),
        [0.0, jnp.pi, 3.0 * jnp.pi, 8.0 * jnp.pi],
    )
    auto = shells.auto_power(shells.transform(field))
    discrepancy = cosmology.SpectralFieldDiscrepancyPlan(shells).evaluate(
        field, shifted, "field", "shifted"
    )
    full_fft = shells.cell_volume * jnp.fft.fftn(field)
    full_power = jnp.abs(full_fft) ** 2
    rfft_total = shells.volume * auto.total_weighted_value
    full_total = jnp.sum(full_power)

    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), [0.5, 0.5], ambient_dimension=1
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    target_positions = jnp.asarray([[0.25], [0.75]])
    target = transfer.deposit_content(
        transfer.build(target_positions), particles.masses
    ).density
    layout = phx.observation.CoordinateLayout(
        tuple(f"density:{index}" for index in range(target.size))
    )
    observation = phx.solver.FieldObservationPlan(
        lambda value, args: value,
        target,
        phx.observation.CholeskyCovarianceAction(jnp.eye(target.size), layout),
        observation_id="qualification-target",
    )
    inverse = cosmology.ParticleFieldRealizationPlan(
        transfer, observation, plan_id="qualification-inverse"
    )
    initial = jnp.asarray([[0.2], [0.7]])
    value, gradient = inverse.value_and_gradient(initial)
    direction = jnp.asarray([[0.1], [-0.1]])
    sensitivity = inverse.sensitivity(initial, direction)

    report = {
        "single_mode_power_error": float(jnp.abs(auto.shell_values[1] - 0.25)),
        "weighted_mode_count": int(jnp.sum(auto.weighted_mode_count)),
        "rfft_full_fft_energy_error": float(jnp.abs(rfft_total - full_total)),
        "phase_discrepancy": float(discrepancy.total_discrepancy),
        "parseval_residual": float(discrepancy.parseval_residual),
        "inverse_initial_objective": float(value),
        "inverse_gradient_finite": bool(jnp.all(jnp.isfinite(gradient))),
        "inverse_jvp_residual": float(sensitivity.jvp_residual),
        "inverse_mass_defect": float(
            inverse.evaluate(target_positions).mass_balance_defect
        ),
        "inverse_support_complete": bool(
            inverse.evaluate(target_positions).support_complete
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
