#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer-visible posed signed-field enclosures, exact values, and refusals."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import (
    ConvexShapePlan,
    ConvexShapeSetPlan,
    ParticleDiscretization,
    ParticleSetPlan,
    PreparedRigidSphereSet,
    RigidBodyKinematics,
    RigidSphereClumpSetPlan,
    RigidSphereSetPlan,
    SphereClumpTemplatePlan,
    SuperquadricSetPlan,
)


pytestmark = pytest.mark.strict_jax


def _particles(count: int, dimension: int) -> ParticleDiscretization:
    return ParticleSetPlan(
        np.arange(count, dtype=np.int64),
        np.ones(count, dtype=np.float64),
        ambient_dimension=dimension,
    ).prepare()


def _pose(position: Array, orientation: Array) -> RigidBodyKinematics:
    dimension = position.shape[1]
    return RigidBodyKinematics(
        position,
        jnp.zeros_like(position),
        orientation,
        jnp.zeros((position.shape[0], 1 if dimension == 2 else 3), dtype=position.dtype),
    )


def _sphere(dimension: int) -> PreparedRigidSphereSet:
    return RigidSphereSetPlan(
        np.asarray([0.7], dtype=np.float64),
        np.asarray([0], dtype=np.int32),
    ).prepare(_particles(1, dimension))


@pytest.mark.parametrize("dimension", [2, 3], ids=["disk", "sphere"])
def test_sphere_signed_distance_is_exact_inside_outside_and_at_center(
    dimension: int,
) -> None:
    shape = _sphere(dimension)
    center = np.arange(dimension, dtype=np.float64) + 2.0
    offsets = np.zeros((4, dimension), dtype=np.float64)
    offsets[:, 0] = np.asarray([0.0, 0.3, 1.4, 2.0], dtype=np.float64)
    orientation = np.zeros((1, 1 if dimension == 2 else 4), dtype=np.float64)
    if dimension == 3:
        orientation[0, 0] = 1.0
    pose = _pose(jnp.asarray(center[None, :]), jnp.asarray(orientation))
    query = shape.evaluate_signed_field_query(
        pose,
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.asarray(center[None, :] + offsets),
        shape_scale=jnp.asarray([2.0], dtype=jnp.float64),
    )
    assert query.successful
    np.testing.assert_array_equal(query.supported, True)
    np.testing.assert_allclose(
        query.value, np.linalg.norm(offsets, axis=-1) - 1.4, atol=1e-14
    )
    np.testing.assert_array_equal(query.lipschitz_bound, np.ones(4))
    np.testing.assert_array_equal(query.evaluation_error, np.zeros(4))
    assert query.shape_id == shape.shape_id


def test_sphere_spatial_certificate_contains_independent_nearby_distances() -> None:
    shape = _sphere(3)
    pose = _pose(
        jnp.asarray([[2.0, -1.0, 3.0]], dtype=jnp.float64),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float64),
    )
    points = np.asarray(
        [[2.0, -1.0, 3.0], [2.2, -1.0, 3.0], [4.0, 0.0, 2.0]], dtype=np.float64
    )
    displacements = np.asarray(
        [[0.1, 0.2, 0.0], [-0.2, 0.1, 0.1], [0.1, -0.1, 0.2]], dtype=np.float64
    )
    result = eqx.filter_jit(shape.evaluate_signed_field_query)(
        pose, jnp.zeros((3,), dtype=jnp.int32), jnp.asarray(points)
    )
    truth = (
        np.linalg.norm(points + displacements - np.asarray(pose.position), axis=-1) - 0.7
    )
    enclosure = np.asarray(result.lipschitz_bound) * np.linalg.norm(
        displacements, axis=-1
    ) + np.asarray(result.evaluation_error)
    assert result.successful
    assert np.all(truth >= np.asarray(result.value) - enclosure - 1e-14)
    assert np.all(truth <= np.asarray(result.value) + enclosure + 1e-14)


def test_clump_minimum_field_respects_pose_scale_and_component_padding() -> None:
    one = SphereClumpTemplatePlan(
        np.zeros((1, 3), dtype=np.float64),
        np.asarray([0.2], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        np.zeros(1, dtype=np.int32),
    )
    two = SphereClumpTemplatePlan(
        np.asarray([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=np.float64),
        np.full(2, 0.4, dtype=np.float64),
        np.full(2, 0.5, dtype=np.float64),
        np.zeros(2, dtype=np.int32),
    )
    shape = RigidSphereClumpSetPlan(
        (one, two),
        np.asarray([0, 1], dtype=np.int32),
        np.zeros(2, dtype=np.int32),
    ).prepare(_particles(2, 3))
    center = np.asarray([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]], dtype=np.float64)
    quaternion = np.asarray(
        [[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]] * 2, dtype=np.float64
    )
    pose = _pose(jnp.asarray(center), jnp.asarray(quaternion))
    local = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [1.6, 0.2, 0.0]],
        dtype=np.float64,
    )
    owners = np.asarray([0, 1, 1, 1], dtype=np.int32)
    scale = 1.7
    # Independent 90-degree rotation about z.
    world = center[owners] + scale * np.stack(
        (-local[:, 1], local[:, 0], local[:, 2]), axis=-1
    )
    result = shape.evaluate_signed_field_query(
        pose,
        jnp.asarray(owners),
        jnp.asarray(world),
        shape_scale=jnp.full((2,), scale, dtype=jnp.float64),
    )
    offsets = np.asarray([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=np.float64)
    truth = (
        np.concatenate(
            (
                [-0.2],
                np.min(
                    np.linalg.norm(local[1:, None, :] - offsets[None, :, :], axis=-1)
                    - 0.4,
                    axis=-1,
                ),
            )
        )
        * scale
    )
    assert result.successful
    np.testing.assert_allclose(result.value, truth, atol=1e-14)
    np.testing.assert_array_equal(result.lipschitz_bound, np.ones(4))
    np.testing.assert_array_equal(result.evaluation_error, np.zeros(4))


def _box_template() -> ConvexShapePlan:
    vertices = np.asarray(
        [
            [-1.0, -0.5, -0.25],
            [1.0, -0.5, -0.25],
            [1.0, 0.5, -0.25],
            [-1.0, 0.5, -0.25],
            [-1.0, -0.5, 0.25],
            [1.0, -0.5, 0.25],
            [1.0, 0.5, 0.25],
            [-1.0, 0.5, 0.25],
        ],
        dtype=np.float64,
    ) + np.asarray([0.4, 1.2, -0.8], dtype=np.float64)
    triangles = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return ConvexShapePlan(vertices, triangles, 0)


def _box_distance(points: np.ndarray) -> np.ndarray:
    excess = np.abs(points) - np.asarray([1.0, 0.5, 0.25], dtype=np.float64)
    return np.linalg.norm(np.maximum(excess, 0.0), axis=-1) + np.minimum(
        np.max(excess, axis=-1), 0.0
    )


def test_convex_exact_distance_covers_face_edge_vertex_and_interior_after_pose() -> None:
    shape = ConvexShapeSetPlan(
        (_box_template(),),
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.int32),
    ).prepare(_particles(1, 3))
    center = np.asarray([3.0, -2.0, 5.0], dtype=np.float64)
    pose = _pose(
        jnp.asarray(center[None, :]),
        jnp.asarray([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]], dtype=jnp.float64),
    )
    local = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.2, 0.1],
            [2.0, 0.0, 0.0],
            [2.0, 1.5, 0.0],
            [2.0, 1.5, 1.25],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    world = center + 2.0 * np.stack((-local[:, 1], local[:, 0], local[:, 2]), axis=-1)
    result = eqx.filter_jit(shape.evaluate_signed_field_query)(
        pose,
        jnp.zeros((6,), dtype=jnp.int32),
        jnp.asarray(world),
        shape_scale=jnp.asarray([2.0], dtype=jnp.float64),
    )
    assert result.successful
    np.testing.assert_allclose(result.value, 2.0 * _box_distance(local), atol=1e-13)
    np.testing.assert_array_equal(result.lipschitz_bound, np.ones(6))
    np.testing.assert_array_equal(result.evaluation_error, np.zeros(6))


def test_convex_spatial_certificate_contains_exact_inside_and_outside_box_distances() -> (
    None
):
    shape = ConvexShapeSetPlan(
        (_box_template(),),
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.int32),
    ).prepare(_particles(1, 3))
    pose = _pose(
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float64),
    )
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.9, 0.4, 0.2], [2.0, 1.5, 1.25]], dtype=np.float64
    )
    delta = np.asarray(
        [[0.1, -0.1, 0.1], [0.2, 0.1, 0.1], [-0.2, 0.2, -0.3]], dtype=np.float64
    )
    result = shape.evaluate_signed_field_query(
        pose, jnp.zeros((3,), dtype=jnp.int32), jnp.asarray(points)
    )
    truth = _box_distance(points + delta)
    radius = np.linalg.norm(delta, axis=-1)
    assert result.successful
    assert np.all(truth >= np.asarray(result.value) - radius - 1e-13)
    assert np.all(truth <= np.asarray(result.value) + radius + 1e-13)


@pytest.mark.parametrize(
    "bad_case",
    ["index", "scale", "point", "rotation"],
    ids=[
        "out-of-range-owner",
        "nonpositive-scale",
        "nonfinite-point",
        "nonunit-quaternion",
    ],
)
def test_signed_field_invalid_routes_refuse_enclosure(bad_case: str) -> None:
    shape = _sphere(3)
    pose = _pose(
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float64),
    )
    owner = jnp.asarray([2 if bad_case == "index" else 0], dtype=jnp.int32)
    scale = jnp.asarray([0.0 if bad_case == "scale" else 1.0], dtype=jnp.float64)
    point = jnp.asarray(
        [[jnp.nan if bad_case == "point" else 0.5, 0.0, 0.0]], dtype=jnp.float64
    )
    if bad_case == "rotation":
        pose = eqx.tree_at(lambda item: item.orientation, pose, 2.0 * pose.orientation)
    result = shape.evaluate_signed_field_query(pose, owner, point, shape_scale=scale)
    assert not result.successful
    assert not result.supported[0]
    assert np.isnan(np.asarray(result.value)[0])
    assert np.isinf(np.asarray(result.lipschitz_bound)[0])
    assert np.isinf(np.asarray(result.evaluation_error)[0])


def test_superquadric_explicitly_refuses_unavailable_signed_field_bound() -> None:
    shape = SuperquadricSetPlan(
        np.ones((1, 3), dtype=np.float64),
        np.full(1, 2.0, dtype=np.float64),
        np.full(1, 2.0, dtype=np.float64),
        np.zeros(1, dtype=np.int32),
    ).prepare(_particles(1, 3))
    pose = _pose(
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float64),
    )
    result = shape.evaluate_signed_field_query(
        pose,
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64),
    )
    assert not result.successful
    np.testing.assert_array_equal(result.supported, False)
    assert np.all(np.isnan(np.asarray(result.value)))
    assert np.all(np.isinf(np.asarray(result.lipschitz_bound)))
    assert np.all(np.isinf(np.asarray(result.evaluation_error)))
