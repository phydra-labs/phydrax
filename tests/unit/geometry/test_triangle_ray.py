#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.geometry._triangle_ray import (
    intersect_triangle_rays,
    prepare_triangle_ray_query,
    TriangleRayIntersectionStatus,
    TriangleRayQueryPlan,
)


def _cube_mesh():
    vertices = jnp.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    triangles = jnp.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=jnp.int32,
    )
    return vertices, triangles


def test_bvh_matches_exhaustive_for_ten_thousand_deterministic_rays():
    vertices, triangles = _cube_mesh()
    entity_ids = jnp.repeat(jnp.arange(6, dtype=jnp.int32), 2)
    common = dict(entity_ids=entity_ids, leaf_size=2, traversal_stack_capacity=16)
    bvh = prepare_triangle_ray_query(
        TriangleRayQueryPlan(vertices, triangles, acceleration="bvh", **common)
    )
    exhaustive = prepare_triangle_ray_query(
        TriangleRayQueryPlan(vertices, triangles, acceleration="exhaustive", **common)
    )
    keys = jax.random.split(jax.random.PRNGKey(3419), 2)
    origins = 4.0 * jax.random.normal(keys[0], (10_000, 3))
    targets = 0.5 * jax.random.normal(keys[1], (10_000, 3))
    directions = targets - origins
    accelerated = intersect_triangle_rays(bvh, origins, directions)
    reference = intersect_triangle_rays(exhaustive, origins, directions)

    np.testing.assert_array_equal(accelerated.status, reference.status)
    np.testing.assert_array_equal(
        accelerated.entity_ids[accelerated.successful],
        reference.entity_ids[reference.successful],
    )
    np.testing.assert_allclose(
        accelerated.intersection.distances[accelerated.successful],
        reference.intersection.distances[reference.successful],
        rtol=2e-6,
        atol=2e-6,
    )
    assert bool(jnp.all(accelerated.triangle_tests <= triangles.shape[0]))


def test_shared_edge_ties_require_one_physical_entity():
    vertices = jnp.asarray(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    triangles = jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    origin = jnp.asarray([[0.0, 0.0, 1.0]])
    direction = jnp.asarray([[0.0, 0.0, -1.0]])
    shared = prepare_triangle_ray_query(
        TriangleRayQueryPlan(vertices, triangles, entity_ids=jnp.asarray([7, 7]))
    )
    distinct = prepare_triangle_ray_query(
        TriangleRayQueryPlan(vertices, triangles, entity_ids=jnp.asarray([7, 8]))
    )

    shared_hit = intersect_triangle_rays(shared, origin, direction)
    distinct_hit = intersect_triangle_rays(distinct, origin, direction)
    assert int(shared_hit.status[0]) == int(TriangleRayIntersectionStatus.SUCCESS)
    assert int(shared_hit.tie_count[0]) == 2
    assert int(shared_hit.entity_ids[0]) == 7
    assert int(distinct_hit.status[0]) == int(TriangleRayIntersectionStatus.AMBIGUOUS_HIT)
    assert not bool(distinct_hit.intersection.valid[0])


def test_vertex_hit_orientation_and_rigid_transform_covariance():
    vertices = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    origin = jnp.asarray([[0.0, 0.0, 2.0]])
    direction = jnp.asarray([[0.0, 0.0, -2.0]])
    base = prepare_triangle_ray_query(TriangleRayQueryPlan(vertices, triangles))
    base_hit = intersect_triangle_rays(base, origin, direction)

    angle = 0.37
    rotation = jnp.asarray(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = jnp.asarray([2.0, -3.0, 0.7])
    transformed_vertices = vertices @ rotation.T + translation
    transformed_origin = origin @ rotation.T + translation
    transformed_direction = direction @ rotation.T
    transformed = prepare_triangle_ray_query(
        TriangleRayQueryPlan(transformed_vertices, triangles)
    )
    transformed_hit = intersect_triangle_rays(
        transformed, transformed_origin, transformed_direction
    )

    assert bool(base_hit.successful[0])
    assert bool(base_hit.front_facing[0])
    np.testing.assert_allclose(
        transformed_hit.intersection.points[0],
        base_hit.intersection.points[0] @ rotation.T + translation,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        transformed_hit.oriented_normals[0],
        base_hit.oriented_normals[0] @ rotation.T,
        atol=2e-6,
    )


def test_origin_on_surface_is_not_positionally_nudged_and_capacity_is_explicit():
    vertices, triangles = _cube_mesh()
    exact = prepare_triangle_ray_query(
        TriangleRayQueryPlan(vertices, triangles, entity_ids=jnp.repeat(jnp.arange(6), 2))
    )
    self_query = intersect_triangle_rays(
        exact,
        jnp.asarray([[0.0, 0.0, 1.0]]),
        jnp.asarray([[0.0, 0.0, 1.0]]),
    )
    assert int(self_query.status[0]) == int(TriangleRayIntersectionStatus.MISS)

    exhausted = prepare_triangle_ray_query(
        TriangleRayQueryPlan(
            vertices,
            triangles,
            leaf_size=1,
            traversal_stack_capacity=1,
        )
    )
    exhausted_hit = intersect_triangle_rays(
        exhausted,
        jnp.asarray([[0.0, 0.0, 3.0]]),
        jnp.asarray([[0.0, 0.0, -1.0]]),
    )
    assert exhausted.required_stack_capacity > exhausted.traversal_stack_capacity
    assert int(exhausted_hit.status[0]) == int(
        TriangleRayIntersectionStatus.TRAVERSAL_CAPACITY_EXHAUSTED
    )
    assert not bool(exhausted_hit.successful[0])
