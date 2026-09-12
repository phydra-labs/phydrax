#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _particles(ids, dimension=1, *, active_mask=None):
    count = len(ids)
    masses = np.ones((count,))
    if active_mask is not None:
        masses = np.where(np.asarray(active_mask), masses, np.nan)
    return phx.discretization.ParticleSetPlan(
        ids,
        masses,
        ambient_dimension=dimension,
        active_mask=active_mask,
    ).prepare()


def _stable_pairs(state):
    valid = np.asarray(state.pair_relation.valid, dtype=bool)
    left = np.asarray(state.pair_relation.left_particle_ids)[valid]
    right = np.asarray(state.pair_relation.right_particle_ids)[valid]
    return list(zip(left.tolist(), right.tolist(), strict=True))


def test_morton_neighborhood_prepares_native_resources_and_stable_pairs() -> None:
    particles = _particles([40, 10, 30, 20])
    box = phx.discretization.ParticleBox([0.0], [1.0])
    prepared = phx.discretization.MortonTreeParticleNeighborhoodPlan(
        0.3,
        6,
        box,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
    ).prepare(particles)
    state = prepared.build(jnp.asarray([[0.82], [0.08], [0.56], [0.31]]))

    assert prepared.backend == "morton_tree"
    assert prepared.resource_evidence_id == prepared.preparation.report_id
    assert dict(prepared.preparation.resource_counts)["leaf_capacity"] == 2
    assert bool(state.successful)
    assert int(jnp.sum(state.cell_counts)) == 4
    assert _stable_pairs(state) == [(10, 20), (10, 40), (20, 30), (30, 40)]
    np.testing.assert_array_equal(
        state.logical_to_storage[state.storage_to_logical], jnp.arange(4)
    )


def test_morton_neighborhood_matches_cell_list_with_periodic_masks() -> None:
    particles = _particles([0, 1, 2, 3], dimension=2)
    box = phx.discretization.ParticleBox(
        [0.0, 0.0],
        [1.0, 1.0],
        periodic_axes=(True, False),
    )
    positions = jnp.asarray([[0.05, 0.5], [0.95, 0.5], [0.5, 0.5], [0.55, 0.5]])
    active = jnp.asarray([True, True, True, False])
    morton = phx.discretization.MortonTreeParticleNeighborhoodPlan(
        0.2,
        6,
        box,
        maximum_leaf_occupancy=2,
        target_top_nodes=1,
    ).prepare(particles)
    cell = phx.discretization.CellListParticleNeighborhoodPlan(
        0.2,
        4,
        6,
        box,
    ).prepare(particles)

    morton_state = morton.build(positions, active_mask=active)
    cell_state = cell.build(positions, active_mask=active)
    assert bool(morton_state.successful)
    assert set(_stable_pairs(morton_state)) == set(_stable_pairs(cell_state))

    invalid = morton.build(positions.at[2, 1].set(1.0))
    assert bool(invalid.domain_violation)
    assert int(invalid.domain_violation_count) == 1
    assert not bool(invalid.successful)


def test_morton_neighborhood_pair_overflow_fails_closed() -> None:
    particles = _particles(range(4))
    box = phx.discretization.ParticleBox([0.0], [1.0])
    prepared = phx.discretization.MortonTreeParticleNeighborhoodPlan(
        0.5,
        2,
        box,
        maximum_leaf_occupancy=2,
        target_top_nodes=1,
    ).prepare(particles)
    state = prepared.build(jnp.asarray([[0.1], [0.11], [0.12], [0.13]]))
    assert bool(state.pair_overflow)
    assert int(state.pair_overflow_count) == 4
    assert int(state.candidate_pair_count) == 6
    assert int(state.pair_count) == 0
    assert not bool(state.successful)
    np.testing.assert_array_equal(state.pair_relation.valid, False)


def test_morton_neighborhood_build_is_filter_jittable() -> None:
    particles = _particles(range(8))
    box = phx.discretization.ParticleBox([0.0], [1.0])
    prepared = phx.discretization.MortonTreeParticleNeighborhoodPlan(
        0.3,
        24,
        box,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
    ).prepare(particles)
    position = (jnp.arange(8, dtype=float) + 0.5)[:, None] / 8.0
    eager = prepared.build(position)
    compiled = eqx.filter_jit(prepared.build)(position)
    assert bool(compiled.successful)
    np.testing.assert_array_equal(compiled.pair_relation.valid, eager.pair_relation.valid)
    np.testing.assert_array_equal(
        compiled.pair_relation.left_particle_ids,
        eager.pair_relation.left_particle_ids,
    )
    np.testing.assert_array_equal(
        compiled.pair_relation.right_particle_ids,
        eager.pair_relation.right_particle_ids,
    )
