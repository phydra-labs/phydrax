#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.geometry import (
    intersect_ray_plane,
    RayIntersectionStatus,
)


def test_batched_scaled_rays_return_physical_distance_and_normal_sign_independence():
    result = intersect_ray_plane(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, -2.0, 1.0))),
        jnp.asarray(((0.0, 0.0, 4.0), (0.0, 0.0, 2.0))),
        jnp.asarray(((0.0, 0.0, 3.0), (4.0, 5.0, 3.0))),
        jnp.asarray(((0.0, 0.0, 2.0), (0.0, 0.0, -7.0))),
    )

    np.testing.assert_allclose(result.distances, (3.0, 2.0))
    np.testing.assert_allclose(result.points, ((0.0, 0.0, 3.0), (1.0, -2.0, 3.0)))
    assert bool(jnp.all(result.valid))
    assert bool(jnp.all(result.status == int(RayIntersectionStatus.SUCCESS)))


def test_degenerate_coplanar_parallel_behind_and_nonfinite_status_precedence():
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (jnp.nan, 0.0, 0.0),
        )
    )
    directions = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
        )
    )
    normals = jnp.asarray(
        (
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
        )
    )
    result = intersect_ray_plane(
        origins,
        directions,
        jnp.zeros_like(origins),
        normals,
    )

    expected = jnp.asarray(
        (
            RayIntersectionStatus.DEGENERATE_DIRECTION,
            RayIntersectionStatus.DEGENERATE_NORMAL,
            RayIntersectionStatus.PARALLEL,
            RayIntersectionStatus.BEHIND_RAY,
            RayIntersectionStatus.NONFINITE_INPUT,
        ),
        dtype=jnp.int32,
    )
    np.testing.assert_array_equal(result.status, expected)
    np.testing.assert_allclose(result.distances, jnp.zeros((5,)))
    np.testing.assert_allclose(result.points[:4], origins[:4])

    coplanar = intersect_ray_plane(
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0)),
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
    )
    assert int(coplanar.status) == int(RayIntersectionStatus.COPLANAR)


def test_forward_tolerance_accepts_a_small_negative_signed_distance():
    result = intersect_ray_plane(
        jnp.asarray((0.0, 0.0, 5e-10)),
        jnp.asarray((0.0, 0.0, 3.0)),
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
        forward_tolerance=1e-9,
    )

    assert bool(result.valid)
    np.testing.assert_allclose(result.distances, -5e-10, rtol=1e-6)


def test_intersection_is_jittable_vmappable_and_differentiable_on_valid_branch():
    def distance(origin_z):
        return intersect_ray_plane(
            jnp.stack((0.2 * origin_z, jnp.zeros_like(origin_z), origin_z)),
            jnp.asarray((0.0, 0.0, 5.0)),
            jnp.asarray((0.0, 0.0, 3.0)),
            jnp.asarray((0.0, 0.0, 2.0)),
        ).distances

    values = jax.jit(jax.vmap(distance))(jnp.asarray((0.0, 0.5, 1.0)))
    gradient = jax.grad(distance)(jnp.asarray(0.5))

    np.testing.assert_allclose(values, (3.0, 2.5, 2.0))
    np.testing.assert_allclose(gradient, -1.0)
