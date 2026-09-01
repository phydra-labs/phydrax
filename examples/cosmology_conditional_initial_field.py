"""Condition a whitened periodic initial field through spectral discrepancy observations."""

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    shape = (4, 4)
    x, y = jnp.meshgrid(
        (jnp.arange(4) + 0.5) / 4.0,
        (jnp.arange(4) + 0.5) / 4.0,
        indexing="ij",
    )
    target = jnp.cos(2.0 * jnp.pi * x) + 0.5 * jnp.cos(2.0 * jnp.pi * y)
    shells = phx.discretization.PeriodicFourierShellPlan(
        shape,
        (1.0, 1.0),
        jnp.linspace(0.0, jnp.sqrt(2.0) * jnp.pi * 4.0, 8),
    )
    discrepancy = phx.applications.cosmology.SpectralFieldDiscrepancyPlan(shells)
    valid_count = int(jnp.sum(shells.valid_shells))
    layout = phx.observation.CoordinateLayout(
        tuple(f"shell:{index}" for index in range(valid_count))
    )

    def observe(field, args):
        result = discrepancy.evaluate(
            field.reshape(shape), target, "latent-field", "target-field"
        )
        return result.shell_discrepancy[result.valid_shells]

    observation = phx.solver.FieldObservationPlan(
        observe,
        jnp.zeros((valid_count,)),
        phx.observation.CholeskyCovarianceAction(0.1 * jnp.eye(valid_count), layout),
        observation_id="conditional-spectral-shells",
    )
    inference = phx.solver.WhitenedFieldInferencePlan(
        lambda field, args: field,
        observation,
        jnp.eye(target.size),
        plan_id="conditional-initial-field",
    )
    initial = jnp.zeros((target.size,))
    result = phx.optim.minimize(
        lambda latent, args: -inference.log_density(latent),
        initial,
        method=phx.optim.NonlinearConjugateGradient(),
        termination=phx.optim.OptimizationTermination(maximum_steps=24),
    )
    conditioned = inference.physical_field(result.parameters).reshape(shape)
    final = discrepancy.evaluate(conditioned, target, "conditioned-field", "target-field")
    print("final_discrepancy", float(final.total_discrepancy))
    print("parseval_residual", float(final.parseval_residual))


if __name__ == "__main__":
    main()
