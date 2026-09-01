#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _physical_pairs(state):
    pairs = state.pair_relation
    valid = np.asarray(pairs.valid)
    left = np.asarray(pairs.left_particle_ids)[valid]
    right = np.asarray(pairs.right_particle_ids)[valid]
    return set(zip(left.tolist(), right.tolist(), strict=True))


def test_sparse_hierarchy_matches_dense_radius_authority_across_levels_and_seam():
    particle_ids = jnp.asarray([60, 10, 40, 20, 50, 30])
    interaction_radii = jnp.asarray([0.05, 0.05, 0.1, 0.2, 0.4, 0.4])
    positions = jnp.asarray(
        [
            [-0.98, -0.8],
            [0.98, -0.8],
            [-0.45, 0.0],
            [-0.2, 0.0],
            [0.35, 0.0],
            [0.9, 0.7],
        ]
    )
    particles = phx.discretization.ParticleSetPlan(
        particle_ids,
        jnp.ones((6,)),
        ambient_dimension=2,
    ).prepare()
    box = phx.discretization.ParticleBox(
        jnp.asarray([-1.0, -1.0]),
        jnp.asarray([1.0, 1.0]),
        periodic_axes=(True, True),
    )
    dense = (
        phx.discretization.DenseParticleNeighborhoodPlan(15, box=box)
        .prepare(particles)
        .build(positions)
    )
    dense_pairs = dense.pair_relation
    dense_displacement = box.minimum_image(
        positions[dense_pairs.left_indices] - positions[dense_pairs.right_indices]
    )
    dense_distance = jnp.sqrt(jnp.sum(dense_displacement**2, axis=-1))
    dense_reach = (
        interaction_radii[dense_pairs.left_indices]
        + interaction_radii[dense_pairs.right_indices]
    )
    expected = set(
        zip(
            np.asarray(dense_pairs.left_particle_ids)[
                np.asarray(dense_distance < dense_reach)
            ].tolist(),
            np.asarray(dense_pairs.right_particle_ids)[
                np.asarray(dense_distance < dense_reach)
            ].tolist(),
            strict=True,
        )
    )

    hierarchy = phx.discretization.HierarchicalRadiusParticleNeighborhoodPlan(
        interaction_radii,
        jnp.asarray([0.04, 0.08, 0.16, 0.32, 0.5]),
        4,
        15,
        box,
    ).prepare(particles)
    result = hierarchy.build(positions)

    assert result.successful
    assert _physical_pairs(result) == expected
    assert result.prepared_neighborhood_id == hierarchy.prepared_id
    resources = dict(hierarchy.preparation.resource_counts)
    assert resources["dense_cell_slots"] == 0
    assert resources["sorted_cell_slots"] == 24
    assert result.candidate_pair_count < 15

    key_space = phx.discretization.ParticlePairKeySpace(particles)
    original_keys = key_space.keys(result.pair_relation)
    moved = hierarchy.build(positions.at[3, 0].add(0.01))
    moved_keys = key_space.keys(moved.pair_relation)
    remap = phx.discretization.match_particle_pair_keys(
        original_keys.keys,
        original_keys.valid,
        moved_keys.keys,
        moved_keys.valid,
    )
    assert remap.successful
    assert jnp.sum(remap.continued) == moved.pair_count


def test_sparse_hierarchy_fails_closed_on_pair_and_cell_overflow():
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), jnp.ones((4,)), ambient_dimension=2
    ).prepare()
    box = phx.discretization.ParticleBox(
        jnp.asarray([-1.0, -1.0]),
        jnp.asarray([1.0, 1.0]),
        periodic_axes=(False, False),
    )
    positions = jnp.asarray([[-0.05, 0.0], [0.0, 0.0], [0.05, 0.0], [0.1, 0.0]])
    plan = phx.discretization.HierarchicalRadiusParticleNeighborhoodPlan(
        jnp.full((4,), 0.2),
        jnp.asarray([0.1, 0.3]),
        2,
        1,
        box,
    ).prepare(particles)
    state = plan.build(positions)
    assert state.cell_overflow
    assert not state.successful
    assert state.cell_overflow_count == 2

    pair_plan = phx.discretization.HierarchicalRadiusParticleNeighborhoodPlan(
        jnp.full((4,), 0.2),
        jnp.asarray([0.1, 0.3]),
        4,
        1,
        box,
    ).prepare(particles)
    pair_state = pair_plan.build(positions)
    assert not pair_state.cell_overflow
    assert pair_state.pair_overflow
    assert pair_state.pair_overflow_count == 5

    with pytest.raises(ValueError, match="candidate slots"):
        phx.discretization.HierarchicalRadiusParticleNeighborhoodPlan(
            jnp.full((4,), 0.2),
            jnp.asarray([0.1, 0.3]),
            2,
            6,
            box,
            maximum_candidate_slots=1,
        ).prepare(particles)
