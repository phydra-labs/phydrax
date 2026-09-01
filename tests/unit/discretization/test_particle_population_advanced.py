#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_population_allocation_deactivation_and_incarnation_are_transactional():
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(6),
        jnp.ones((6,)),
        ambient_dimension=2,
    ).prepare()
    plan = phx.discretization.ParticlePopulationPlan(particles)
    state = plan.initialize(
        active_mask=jnp.asarray([True, True, False, False, False, False]),
        masses=jnp.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    allocated = plan.allocate(
        state,
        phx.discretization.ParticleAllocationRequest(
            jnp.asarray([2, 1], dtype=jnp.int64),
            jnp.asarray([0.5, 0.25]),
            jnp.asarray([True, True]),
        ),
    )
    assert allocated.successful
    assert allocated.allocated_count == 2
    assert jnp.sum(allocated.accepted_state.active) == 4
    removed = plan.deactivate(
        allocated.accepted_state,
        allocated.accepted_state.active & ~state.active,
    )
    assert removed.successful
    np.testing.assert_allclose(removed.removed_mass, 0.75)
    reused = plan.allocate(
        removed.accepted_state,
        phx.discretization.ParticleAllocationRequest(
            jnp.asarray([3], dtype=jnp.int64),
            jnp.asarray([0.4]),
            jnp.asarray([True]),
        ),
    )
    assert reused.successful
    slot = reused.slots[0]
    assert reused.accepted_state.incarnation[slot] >= 2


def test_particle_splat_runtime_mask_excludes_inactive_payload():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), jnp.ones((3,)), ambient_dimension=1
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    position = jnp.asarray([[0.1], [0.4], [0.8]])
    state = splat.build(position, active_mask=jnp.asarray([True, False, True]))
    result = splat.deposit_content(state, jnp.asarray([1.0, 1000.0, 2.0]))
    assert result.successful
    np.testing.assert_allclose(jnp.sum(result.content), 3.0)
