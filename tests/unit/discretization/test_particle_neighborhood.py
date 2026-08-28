#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_dense_neighborhood_uses_canonical_stable_pair_ids():
    particles = phx.discretization.ParticleSetPlan(
        [20, 5, 13], [1.0, 1.0, 1.0], ambient_dimension=1
    ).prepare()
    prepared = phx.discretization.DenseParticleNeighborhoodPlan(3).prepare(particles)
    state = prepared.build(jnp.asarray([[0.0], [1.0], [2.0]]))
    pairs = state.pair_relation

    assert pairs.capacity == 3
    assert prepared.pair_capacity == 3
    assert int(state.pair_count) == 3
    assert state.successful
    assert jnp.array_equal(state.storage_to_logical, jnp.arange(3))
    assert jnp.array_equal(state.logical_to_storage, jnp.arange(3))
    assert np.all(
        np.asarray(pairs.left_particle_ids) < np.asarray(pairs.right_particle_ids)
    )
    assert set(
        zip(
            np.asarray(pairs.left_particle_ids).tolist(),
            np.asarray(pairs.right_particle_ids).tolist(),
            strict=True,
        )
    ) == {(5, 13), (5, 20), (13, 20)}
    assert np.all(np.asarray(pairs.left_indices) != np.asarray(pairs.right_indices))
    assert prepared.resource_evidence_id == prepared.preparation.report_id


def test_dense_neighborhood_rejects_allocation_over_budget():
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(5), jnp.ones((5,)), ambient_dimension=2
    ).prepare()
    with pytest.raises(ValueError, match="requires 10 pairs"):
        phx.discretization.DenseParticleNeighborhoodPlan(9).prepare(particles)


def test_periodic_pair_geometry_uses_minimum_image_and_zero_safe_direction():
    particles = phx.discretization.ParticleSetPlan(
        [0, 1], [1.0, 1.0], ambient_dimension=1
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    prepared = phx.discretization.DenseParticleNeighborhoodPlan(1, box=box).prepare(
        particles
    )

    position = jnp.asarray([[0.05], [0.95]])
    state = prepared.build(position)
    geometry = phx.discretization.particle_pair_geometry(
        position, state.pair_relation, box=box
    )
    assert jnp.allclose(geometry.displacement, jnp.asarray([[0.1]]))
    assert jnp.allclose(geometry.distance, jnp.asarray([0.1]))
    assert jnp.allclose(geometry.direction, jnp.asarray([[1.0]]))

    coincident_position = jnp.asarray([[0.2], [0.2]])
    coincident = phx.discretization.particle_pair_geometry(
        coincident_position,
        prepared.build(coincident_position).pair_relation,
        box=box,
    )
    assert jnp.array_equal(coincident.direction, jnp.zeros((1, 1)))
    assert jnp.all(jnp.isfinite(coincident.direction))


def test_pair_exchange_is_equal_opposite_for_every_accumulation_policy():
    particles = phx.discretization.ParticleSetPlan(
        [0, 1, 2], [1.0, 1.0, 1.0], ambient_dimension=2
    ).prepare()
    prepared = phx.discretization.DenseParticleNeighborhoodPlan(3).prepare(particles)
    pairs = prepared.build(jnp.zeros((3, 2))).pair_relation
    values = jnp.asarray([[1.0, -0.5], [0.25, 2.0], [-3.0, 1.0]])

    for accumulation in ("fast", "deterministic", "compensated"):
        result = phx.discretization.scatter_pair_exchange(
            pairs,
            values,
            size=3,
            accumulation=accumulation,
        )
        assert jnp.allclose(jnp.sum(result, axis=0), jnp.zeros((2,)), atol=1e-14)


def test_inactive_particles_never_form_valid_pairs():
    particles = phx.discretization.ParticleSetPlan(
        [0, 1, -1],
        [1.0, 1.0, np.nan],
        ambient_dimension=1,
        active_mask=[True, True, False],
    ).prepare()
    prepared = phx.discretization.DenseParticleNeighborhoodPlan(3).prepare(particles)
    state = prepared.build(jnp.zeros((3, 1)))
    relation = state.pair_relation

    assert relation.capacity == 3
    assert int(jnp.sum(relation.valid)) == 1
    assert int(state.pair_count) == 1
