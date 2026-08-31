#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic plane-strain elastic wave with explicit APIC material points."""

import jax.numpy as jnp

import phydrax as phx


def run():
    nx, ny = 24, 6
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(nx, periodic=True, endpoint=False),
            phx.discretization.UniformAxisSpec(ny, periodic=True, endpoint=False),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 0.25]]))

    x = (jnp.arange(nx) + 0.37) / nx
    y = (jnp.arange(ny) + 0.37) * (0.25 / ny)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    position = jnp.stack((xx, yy), axis=-1).reshape((-1, 2))
    reference_volume = jnp.full((position.shape[0],), (1.0 / nx) * (0.25 / ny))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0]),
        reference_volume,
        ambient_dimension=2,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(particles)

    material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
    problem = phx.equations.MaterialPointProblemIR("elastic-wave", material)
    compiled = phx.equations.compile_material_point_problem(
        problem,
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 0.25]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
    )
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
        2.0, 8.0
    )
    arguments = phx.equations.MaterialPointArguments(parameters)
    amplitude = 1.0e-3
    wave_number = 2.0 * jnp.pi
    velocity = jnp.stack(
        (
            amplitude * jnp.sin(wave_number * position[:, 0]),
            jnp.zeros(position.shape[0]),
        ),
        axis=-1,
    )
    gradient = amplitude * wave_number * jnp.cos(wave_number * position[:, 0])
    affine = jnp.zeros((position.shape[0], 2, 2)).at[:, 0, 0].set(gradient)
    initial = compiled.initialize_state(
        position,
        velocity,
        reference_volume,
        arguments,
        affine_velocity=affine,
    )
    temporal_mesh = phx.discretization.TemporalMesh.uniform(
        0.0, 0.004, 16, role="internal"
    )
    rollout = phx.solver.ScheduledMPMRolloutPlan(
        compiled.dynamics,
        temporal_mesh,
        retention="trajectory",
        replay=phx.solver.MPMReplayPolicy("block", block_size=4),
    )
    return rollout.rollout(initial, arguments)


if __name__ == "__main__":
    solution = run()
    print(
        {
            "successful": bool(jnp.all(solution.accepted)),
            "steps": int(solution.accepted.shape[0]),
            "maximum_mass_defect": float(jnp.max(solution.relative_mass_defects)),
            "minimum_jacobian": float(jnp.min(solution.minimum_jacobians)),
        }
    )
