"""Covariance-aware inverse particle realization using native splatting and optimization."""

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2),
        jnp.asarray([0.5, 0.5]),
        ambient_dimension=1,
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    target_positions = jnp.asarray([[0.25], [0.75]])
    target_routes = transfer.build(target_positions)
    target = transfer.deposit_content(target_routes, particles.masses).density
    layout = phx.observation.CoordinateLayout(
        tuple(f"density:{index}" for index in range(target.size))
    )
    observation = phx.solver.FieldObservationPlan(
        lambda field, args: field,
        target,
        phx.observation.CholeskyCovarianceAction(0.05 * jnp.eye(target.size), layout),
        observation_id="inverse-density-target",
    )
    plan = phx.applications.cosmology.ParticleFieldRealizationPlan(
        transfer,
        observation,
        target_kind="density",
        plan_id="two-particle-density-fit",
    )
    initial = jnp.asarray([[0.18], [0.68]])
    initial_objective = plan.objective(initial)
    result = phx.optim.minimize(
        lambda positions, args: plan.objective(positions),
        initial,
        method=phx.optim.NonlinearConjugateGradient(),
        termination=phx.optim.OptimizationTermination(maximum_steps=24),
    )
    final = plan.evaluate(result.parameters)
    print("initial_objective", float(initial_objective))
    print("final_objective", float(final.objective))
    print("positions", final.positions)
    print("mass_defect", float(final.mass_balance_defect))
    print("support_complete", bool(final.support_complete))


if __name__ == "__main__":
    main()
