#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def test_active_block_mask_is_transactional_in_explicit_mpm():
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(16, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.2, 0.2], [0.25, 0.22]])
    volume = jnp.full((2,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    blocks = phx.discretization.MPMActiveBlockPlan((16, 16), (4, 4), 12, halo_blocks=1)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "active-block-integration",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
        active_blocks=blocks,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    state = compiled.initialize_state(
        position,
        jnp.broadcast_to(jnp.asarray((0.02, 0.0)), position.shape),
        volume,
        arguments,
    )
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)

    assert bool(detail.successful)
    assert state.storage_state is not None
    assert detail.accepted_state.storage_state is not None
    assert int(detail.accepted_state.storage_state.active_block_count) <= 12
    assert jnp.all(
        detail.grid.active[0] <= detail.accepted_state.storage_state.active_node_mask
    )
