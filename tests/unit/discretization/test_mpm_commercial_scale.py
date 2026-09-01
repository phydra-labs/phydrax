#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_three_field_contact_solves_simultaneously_with_essential_rows():
    mass = jnp.asarray([[1.0], [1.5], [2.0]])
    velocity = jnp.asarray([[[0.8, 0.3]], [[0.0, 0.0]], [[-0.6, -0.1]]])
    gradients = jnp.asarray([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, -1.0]]])
    plan = phx.discretization.KWayMPMContactPlan(
        3,
        friction=phx.discretization.SharpCoulombMPMFrictionPlan(0.2),
        maximum_steps=40,
        tolerance=1e-8,
    )
    graph = plan.build_graph(mass, gradients)
    essential = jnp.zeros_like(velocity, dtype=bool).at[1, 0, 1].set(True)
    values = jnp.zeros_like(velocity)
    result = plan.solve(
        mass,
        velocity,
        graph,
        0.01,
        essential_mask=essential,
        essential_values=values,
    )

    assert bool(result.successful)
    assert result.complementarity_residual < 1e-8
    assert result.cone_residual < 1e-8
    assert result.equality_residual < 1e-12
    assert result.action_reaction_defect < 1e-12
    assert result.dissipation >= 0.0


def test_shared_rigid_actor_reaction_is_global_across_contact_nodes():
    actors = phx.discretization.MPMRigidActorState(
        jnp.asarray((2.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray([[0.0, 0.0]]),
        jnp.asarray([[0.0, 0.0]]),
        jnp.asarray((0.0,)),
        jnp.asarray((True,)),
    )
    updated = phx.discretization.apply_rigid_actor_reactions(
        actors,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.0, 1.0], [-1.0, 0.0]]),
        jnp.asarray((0, 0)),
    )
    np.testing.assert_allclose(updated.linear_velocity[0], (-0.5, 0.5))
    assert updated.angular_velocity[0] == 2.0


def test_distributed_ownership_migration_reduction_and_global_no_commit():
    owner = jnp.asarray([[0, 0], [1, 1]], dtype=jnp.int32)
    plan = phx.discretization.MPMDistributedPlan(
        (8, 8),
        (4, 4),
        owner,
        device_count=2,
        particle_capacity_per_device=3,
    )
    position = jnp.asarray([[0.1, 0.1], [0.8, 0.2], [0.7, 0.8]])
    migration = phx.discretization.migrate_particles(
        plan,
        position,
        jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
        jnp.asarray((0, 0, 0)),
        jnp.asarray((True, True, True)),
    )
    assert bool(migration.successful)
    assert int(jnp.sum(migration.per_device_count)) == 3

    reduced, defect = phx.discretization.distributed_p2g_reduce(
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    )
    np.testing.assert_allclose(reduced, (4.0, 6.0))
    assert defect == 0.0

    transaction = phx.discretization.distributed_global_transaction(
        jnp.asarray((True, False)), 7
    )
    assert not bool(transaction.global_success)
    assert int(transaction.commit_generation) == 7


def _particle_state():
    return phx.discretization.MPMParticleState(
        jnp.asarray([[0.2, 0.2], [0.8, 0.8], [0.0, 0.0], [0.0, 0.0]]),
        jnp.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        jnp.broadcast_to(jnp.eye(2), (4, 2, 2)),
        jnp.zeros((4, 2, 2)),
        jnp.asarray((0.5, 0.5, 0.0, 0.0)),
        jnp.zeros((4, 2, 2)),
        jnp.zeros((4,)),
        jnp.ones((4,)),
        jnp.zeros((4, 1)),
    )


def test_particle_split_merge_and_capacity_bucket_are_conservative():
    plan = phx.discretization.MPMParticleLifecyclePlan(4)
    lifecycle, valid = plan.initialize(
        jnp.asarray((10, 11, -1, -1)),
        jnp.asarray((1.0, 1.0, 0.0, 0.0)),
        jnp.asarray((True, True, False, False)),
    )
    assert bool(valid)
    particles = _particle_state()
    split = plan.split(
        particles,
        lifecycle,
        0,
        jnp.asarray((2, 3)),
        jnp.asarray((20, 21)),
        jnp.asarray((0.4, 0.6)),
        jnp.asarray([[-0.01, 0.0], [0.01, 0.0]]),
    )
    assert bool(split.evidence.successful)
    merged = plan.merge(
        split.particles,
        split.lifecycle,
        jnp.asarray((2, 3)),
        0,
        30,
    )
    assert bool(merged.evidence.successful)
    assert phx.discretization.MPMCapacityBucketPlan((4, 8, 16)).select(7) == 8


def test_page_table_and_ratio_two_amr_are_deterministic():
    table_plan = phx.discretization.MPMPageTablePlan(16)
    table, inserted = table_plan.insert(
        table_plan.empty(),
        jnp.asarray((5, 21, 7), dtype=jnp.int64),
        jnp.asarray((50, 210, 70), dtype=jnp.int32),
    )
    assert jnp.all(inserted)
    assert not bool(table.overflow)
    assert int(table.count) == 3

    amr = phx.discretization.MPMAMRPlan(
        ((4, 4), (8, 8)),
        (4, 16),
    )
    fine = jnp.arange(64.0).reshape((8, 8))
    coarse = amr.restrict(fine)
    prolonged = amr.prolong(coarse)
    assert coarse.shape == (4, 4)
    assert prolonged.shape == fine.shape
    np.testing.assert_allclose(amr.restrict(prolonged), coarse)
