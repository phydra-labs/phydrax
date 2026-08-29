#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_pair_key_space_is_stable_under_arbitrary_particle_ids():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([30, 2, 10]),
        jnp.ones((3,)),
        ambient_dimension=2,
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(3).prepare(particles)
    relation = neighborhood.build(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ).pair_relation

    keys = phx.discretization.ParticlePairKeySpace(particles).keys(relation)

    assert keys.successful
    assert jnp.array_equal(jnp.sort(keys.keys), jnp.asarray([0, 1, 2]))


def test_pair_history_remap_preserves_continued_values_and_zeros_births():
    remap = phx.discretization.match_particle_pair_keys(
        jnp.asarray([0, 1, 2]),
        jnp.asarray([True, True, True]),
        jnp.asarray([2, 0, 3]),
        jnp.asarray([True, True, True]),
        maximum_key=5,
    )
    values = {
        "scalar": jnp.asarray([10.0, 20.0, 30.0]),
        "vector": jnp.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
    }

    remapped = phx.discretization.remap_particle_pair_values(remap, values)

    assert remap.successful
    assert jnp.array_equal(remap.continued, jnp.asarray([True, True, False]))
    assert jnp.array_equal(remap.born, jnp.asarray([False, False, True]))
    assert remap.ended_count == 1
    assert jnp.allclose(remapped["scalar"], jnp.asarray([30.0, 10.0, 0.0]))
    assert jnp.allclose(
        remapped["vector"], jnp.asarray([[3.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    )

    tangent = jax.jvp(
        lambda value: phx.discretization.remap_particle_pair_values(remap, value)[
            "scalar"
        ],
        (values,),
        ({"scalar": jnp.ones((3,)), "vector": jnp.zeros((3, 2))},),
    )[1]
    assert jnp.array_equal(tangent, jnp.asarray([1.0, 1.0, 0.0]))


def test_pair_remap_reports_duplicates_and_accepts_empty_relations():
    duplicate = phx.discretization.match_particle_pair_keys(
        jnp.asarray([1, 1]),
        jnp.asarray([True, True]),
        jnp.asarray([1, 2]),
        jnp.asarray([True, True]),
        maximum_key=3,
    )
    empty = phx.discretization.match_particle_pair_keys(
        jnp.zeros((0,), dtype=jnp.int64),
        jnp.zeros((0,), dtype=bool),
        jnp.zeros((0,), dtype=jnp.int64),
        jnp.zeros((0,), dtype=bool),
        maximum_key=0,
    )

    assert not duplicate.successful
    assert empty.successful
    assert empty.source_indices.shape == (0,)
