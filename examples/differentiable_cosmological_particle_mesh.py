"""Differentiable 2LPT to periodic particle-mesh cosmology workflow."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import phydrax as phx


def _periodic_gravity(shape: tuple[int, ...], particles):
    dimension = len(shape)
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for count in shape
        ),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    system = phx.equations.EulerSystem(dimension)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "cosmology-example-gravity",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    gravity = phx.solver.NewtonianSelfGravityPlan(0.02).prepare(
        phx.solver.prepare_balance_law_transport(runtime)
    )
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    return phx.solver.ParticleMeshGravityPlan(gravity, transfer)


def build_workflow():
    shape = (4, 4, 4)
    count = 4**3
    scale = phx.applications.cosmology.CosmologyScaleContract(
        phx.applications.cosmology.CODE_COSMOLOGY_SCALE.length_unit,
        phx.applications.cosmology.CODE_COSMOLOGY_SCALE.mass_unit,
        phx.applications.cosmology.CODE_COSMOLOGY_SCALE.time_unit,
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.full((count,), 1.0 / count),
        ambient_dimension=3,
    ).prepare()
    background = phx.applications.cosmology.FLRWBackground(1.0, 0.3, scale=scale)
    growth_plan = phx.applications.cosmology.FLRWGrowthPlan(
        jnp.geomspace(1.0e-2, 1.0, 64)
    )
    growth = growth_plan.solve(background)
    gravity = _periodic_gravity(shape, particles)
    kdk = phx.applications.cosmology.CosmologicalKDKPlan(
        particles, (1.0, 1.0, 1.0), scale=scale
    )
    rollout = phx.applications.cosmology.CosmologicalParticleMeshPlan(
        kdk,
        gravity,
        jnp.linspace(0.1, 0.12, 3),
    )
    lpt = phx.applications.cosmology.LagrangianPerturbationInitialConditionPlan(
        particles,
        shape,
        (1.0, 1.0, 1.0),
        order=2,
        dealiasing="three_halves",
        scale=scale,
    )
    provenance = phx.applications.cosmology.CosmologyProductProvenance(
        producer="phydrax-example",
        producer_version="native",
        model_form_id=background.model_form_id,
        request_id="native-example-linear-power",
        numerical_policy_id="example-linear-power",
        physics_policy_id="linear-cold-baryon-power",
        scale_id=scale.scale_id,
        source_kind="native",
        differentiation="native-parameter",
    )
    coordinates = tuple(
        (jnp.arange(count_, dtype=float) + 0.5) / count_ for count_ in shape
    )
    mesh = jnp.meshgrid(*coordinates, indexing="ij")
    white_noise = sum(
        jnp.cos(2.0 * jnp.pi * (axis + 1) * coordinate)
        for axis, coordinate in enumerate(mesh)
    )
    return background, growth, provenance, lpt, rollout, gravity, white_noise


def main() -> None:
    background, growth, provenance, lpt, rollout, gravity, white_noise = build_workflow()
    wavenumbers = jnp.linspace(1.0, 30.0, 96)
    first_growth = growth.evaluate(0.1)[0]

    def objective(amplitude):
        base_power = amplitude / (1.0 + (wavenumbers / 8.0) ** 2)
        power = phx.applications.cosmology.MatterPowerTable(
            jnp.asarray([0.1, 1.0]),
            wavenumbers,
            jnp.stack((first_growth**2 * base_power, base_power)),
            phx.applications.cosmology.MatterPowerDescriptor(
                "cold_baryon", "cold_baryon"
            ),
            background.scale,
            provenance,
            background.realization,
        )
        initial = lpt.realize(background, growth, power, white_noise, 0.1)
        evolved = rollout.rollout(background, initial.state)
        density, _ = gravity.density(evolved.state.positions)
        contrast = density.density / jnp.mean(density.density) - 1.0
        return jnp.mean(contrast**2), (initial, evolved)

    (value, (initial, evolved)), gradient = jax.value_and_grad(objective, has_aux=True)(
        jnp.asarray(1.0e-7)
    )
    print("initial_successful", bool(initial.successful))
    print("completed", bool(evolved.successful))
    print("accepted_steps", int(evolved.diagnostics.accepted_steps))
    print("maximum_mass_defect", float(evolved.diagnostics.maximum_mass_balance_defect))
    print("maximum_net_force_norm", float(evolved.diagnostics.maximum_net_force_norm))
    print("density_variance", float(value))
    print("amplitude_gradient", float(gradient))


if __name__ == "__main__":
    main()
