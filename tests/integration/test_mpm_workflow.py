#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_public_material_point_compile_rollout_and_gradient_workflow():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(16, periodic=True, endpoint=False),
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 0.5]]))
    xx, yy = jnp.meshgrid(
        (jnp.arange(8) + 0.5) / 8.0,
        (jnp.arange(4) + 0.5) / 8.0,
        indexing="ij",
    )
    position = jnp.stack((xx, yy), axis=-1).reshape((-1, 2))
    volume = jnp.full((position.shape[0],), 1.0 / position.shape[0] * 0.5)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0]),
        volume,
        ambient_dimension=2,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(particles)
    problem = phx.equations.MaterialPointProblemIR(
        "public-mpm",
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
    )
    compiled = phx.equations.compile_material_point_problem(
        problem,
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 0.5]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
    )
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
        1.0, 4.0
    )
    arguments = phx.equations.MaterialPointArguments(parameters)
    velocity = jnp.stack(
        (1.0e-3 * jnp.sin(2.0 * jnp.pi * position[:, 0]), jnp.zeros(position.shape[0])),
        axis=-1,
    )
    initial = compiled.initialize_state(position, velocity, volume, arguments)
    mesh = phx.discretization.TemporalMesh.uniform(0.0, 0.002, 4, role="internal")
    rollout = phx.solver.ScheduledMPMRolloutPlan(
        compiled.dynamics,
        mesh,
        retention="trajectory",
        replay=phx.solver.MPMReplayPolicy("block", block_size=3),
    )
    result = jax.jit(lambda state: rollout.rollout(state, arguments))(initial)

    assert jnp.all(result.accepted)
    assert result.retained.times.shape == (4,)
    assert result.final_state.time == mesh.t1
    assert jnp.all(result.transfer_successful)
    assert jnp.all(jnp.isfinite(result.final_state.particles.position))

    def loss(scale):
        scaled_velocity = velocity * scale
        state = compiled.initialize_state(position, scaled_velocity, volume, arguments)
        final = (
            phx.solver.ScheduledMPMRolloutPlan(
                compiled.dynamics,
                mesh,
                replay=phx.solver.MPMReplayPolicy("step"),
            )
            .rollout(state, arguments)
            .final_state
        )
        return jnp.sum(final.particles.position[:, 0] ** 2)

    gradient = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(gradient)
    assert jnp.abs(gradient) > 0.0
    np.testing.assert_allclose(
        result.final_state.particles.reference_volume,
        volume,
        rtol=0.0,
        atol=0.0,
    )
