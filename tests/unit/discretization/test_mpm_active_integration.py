#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def test_sparse_block_topology_is_transactional_in_explicit_mpm():
    grid_plan = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(16, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    )
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    index_space = grid_plan.prepare_index_space(bounds)
    position = jnp.asarray([[0.2, 0.2], [0.25, 0.22]])
    volume = jnp.full((2,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        index_space, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    topology = phx.discretization.SparseBlockTopologyPlan(
        index_space,
        (4, 4),
        12,
        layout=index_space.vertices(),
    )
    storage = phx.discretization.BlockSparseMPMNodalStoragePlan(topology)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "active-block-integration",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            bounds,
            periodic=(True, True),
            support_margin=0.0,
        ),
        nodal_storage=storage,
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
    assert dict(splat.preparation.resource_counts)["scalar_workspace_bytes"] == 0
    assert state.storage_state is not None
    assert detail.accepted_state.storage_state is not None
    assert int(detail.accepted_state.storage_state.evidence.required_blocks) <= 12
    assert jnp.all(
        detail.grid.active[0]
        <= detail.accepted_state.storage_state.node_valid.reshape((-1,))
    )
