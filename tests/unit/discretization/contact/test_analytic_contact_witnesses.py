#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.contact._distance import (
    ContactDistanceEvaluation,
    ContactWitnessSlot,
    offset_surface_witnesses,
    point_plane_distance,
    segment_segment_support_distance,
)


def _rows(value: tuple[float, ...]) -> jax.Array:
    return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), (2, 3))


def _support(
    a0: tuple[float, ...],
    a1: tuple[float, ...],
    b0: tuple[float, ...],
    b1: tuple[float, ...],
) -> tuple[ContactDistanceEvaluation, np.ndarray]:
    slots = jnp.asarray(
        (int(ContactWitnessSlot.FIRST), int(ContactWitnessSlot.SECOND)), dtype=jnp.int32
    )
    evaluation, active = segment_segment_support_distance(
        _rows(a0), _rows(a1), _rows(b0), _rows(b1), slots
    )
    return evaluation, np.asarray(active)


def test_capsule_plane_two_endpoint_support_balances_centered_load() -> None:
    radius = 0.1
    normal = jnp.asarray((0.0, 0.0, 1.0))
    endpoints = jnp.asarray(((-1.0, 0.0, 0.1), (1.0, 0.0, 0.1)))
    evaluation = point_plane_distance(endpoints, normal, 0.0)
    gap, capsule_points, plane_points = offset_surface_witnesses(
        evaluation, jnp.full((2,), radius), jnp.zeros((2,))
    )
    np.testing.assert_allclose(gap, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(capsule_points[:, 2], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(plane_points, capsule_points, atol=1.0e-15)
    assert bool(jnp.all(evaluation.nondegenerate))

    # A centered weight W is balanced by equal endpoint impulses W/2 with zero
    # resultant torque about the capsule center; a single selected endpoint
    # cannot balance it.
    weight = 2.0
    center = jnp.mean(endpoints, axis=0)
    impulses = jnp.full((2,), 0.5 * weight)
    forces = impulses[:, None] * evaluation.normal
    resultant = jnp.sum(forces, axis=0) + jnp.asarray((0.0, 0.0, -weight))
    torque = jnp.sum(jnp.cross(capsule_points - center, forces), axis=0)
    np.testing.assert_allclose(resultant, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(torque, 0.0, atol=1.0e-14)
    single = jnp.cross(capsule_points[0] - center, weight * evaluation.normal[0])
    assert float(jnp.linalg.norm(single)) > 1.0


def test_capsule_plane_pitch_and_reversed_endpoint_order() -> None:
    normal = jnp.asarray((0.0, 0.0, 1.0))
    pitched = jnp.asarray(((-1.0, 0.0, 0.1), (1.0, 0.0, 0.3)))
    evaluation = point_plane_distance(pitched, normal, 0.0)
    gap, _, _ = offset_surface_witnesses(evaluation, jnp.full((2,), 0.1), jnp.zeros(2))
    np.testing.assert_allclose(gap, (0.0, 0.2), atol=1.0e-15)
    reversed_gap, _, _ = offset_surface_witnesses(
        point_plane_distance(pitched[::-1], normal, 0.0),
        jnp.full((2,), 0.1),
        jnp.zeros(2),
    )
    # Slot k is endpoint k in body-attached order; reversing the order permutes
    # the slots rather than changing the support set.
    np.testing.assert_allclose(reversed_gap, gap[::-1], atol=1.0e-15)


def test_point_plane_keeps_signed_gap_and_jacobian_through_penetration() -> None:
    normal = jnp.asarray((0.0, 0.0, 1.0))
    points = jnp.asarray(((0.0, 0.0, -0.01), (0.0, 0.0, 0.0), (0.0, 0.0, 0.5)))
    evaluation = point_plane_distance(points, normal, 0.0)
    # Penetrating trial points keep the plane normal: validity is geometric,
    # admissibility of a negative gap belongs to the path certificate.
    np.testing.assert_array_equal(np.asarray(evaluation.nondegenerate), (True,) * 3)
    np.testing.assert_allclose(evaluation.normal, jnp.broadcast_to(normal, (3, 3)))
    gap, _, _ = offset_surface_witnesses(evaluation, jnp.zeros(3), jnp.zeros(3))
    np.testing.assert_allclose(gap, (-0.01, 0.0, 0.5), atol=1.0e-15)

    def signed_gap(point: jax.Array) -> jax.Array:
        value = point_plane_distance(point, normal, 0.0)
        return jnp.sum(value.distance_vector * value.normal)

    np.testing.assert_allclose(jax.grad(signed_gap)(points[0]), normal, atol=1.0e-15)
    skewed = point_plane_distance(points, jnp.asarray((0.0, 0.0, 2.0)), 0.0)
    np.testing.assert_array_equal(np.asarray(skewed.nondegenerate), (False,) * 3)


def test_parallel_capsule_support_emits_two_extremal_slots() -> None:
    evaluation, active = _support((0.0, 0, 0), (2.0, 0, 0), (1.0, 0.3, 0), (3.0, 0.3, 0))
    np.testing.assert_array_equal(active, (True, True))
    np.testing.assert_allclose(
        evaluation.left_witness, ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)), atol=1.0e-15
    )
    np.testing.assert_allclose(
        evaluation.right_witness, ((1.0, 0.3, 0.0), (2.0, 0.3, 0.0)), atol=1.0e-15
    )
    np.testing.assert_allclose(jnp.sqrt(evaluation.squared_distance), 0.3, atol=1e-15)
    np.testing.assert_allclose(evaluation.normal, ((0.0, -1.0, 0.0),) * 2, atol=1e-15)
    # Nonunique support: primal rows are valid, derivatives are not admitted.
    np.testing.assert_array_equal(np.asarray(evaluation.nondegenerate), (True, True))
    np.testing.assert_allclose(evaluation.feature_margin, 0.0)

    # Equal support impulses carry a load centered on the overlap without torque.
    forces = 0.5 * evaluation.normal
    center = jnp.asarray((1.5, 0.0, 0.0))
    torque = jnp.sum(jnp.cross(evaluation.left_witness - center, forces), axis=0)
    np.testing.assert_allclose(torque, 0.0, atol=1.0e-15)


def test_parallel_support_slots_follow_left_axis_order() -> None:
    forward, _ = _support((0.0, 0, 0), (2.0, 0, 0), (1.0, 0.3, 0), (3.0, 0.3, 0))
    right_reversed, _ = _support((0.0, 0, 0), (2.0, 0, 0), (3.0, 0.3, 0), (1.0, 0.3, 0))
    left_reversed, active = _support(
        (2.0, 0, 0), (0.0, 0, 0), (1.0, 0.3, 0), (3.0, 0.3, 0)
    )
    np.testing.assert_allclose(
        right_reversed.left_witness, forward.left_witness, atol=1.0e-15
    )
    np.testing.assert_array_equal(active, (True, True))
    np.testing.assert_allclose(
        left_reversed.left_witness, forward.left_witness[::-1], atol=1.0e-15
    )


def test_disjoint_collapsed_and_nonparallel_capsules_use_unique_witness() -> None:
    for b0, b1 in (
        ((3.0, 0.3, 0.0), (4.0, 0.3, 0.0)),
        ((2.0, 0.3, 0.0), (3.0, 0.3, 0.0)),
        ((1.0, -1.0, 0.3), (1.0, 1.0, 0.3)),
    ):
        evaluation, active = _support((0.0, 0, 0), (2.0, 0, 0), b0, b1)
        np.testing.assert_array_equal(active, (True, False))
        assert bool(evaluation.nondegenerate[0])
    evaluation, _ = _support((0.0, 0, 0), (2.0, 0, 0), (1.0, -1.0, 0.3), (1.0, 1.0, 0.3))
    np.testing.assert_allclose(jnp.sqrt(evaluation.squared_distance[0]), 0.3)
    assert float(evaluation.feature_margin[0]) > 0.0


def test_coincident_capsule_axes_refuse_normal() -> None:
    evaluation, _ = _support((0.0, 0, 0), (2.0, 0, 0), (1.0, 0.0, 0), (3.0, 0.0, 0))
    np.testing.assert_array_equal(np.asarray(evaluation.nondegenerate), (False, False))


def test_support_slots_are_dynamic_under_jit_and_vmap() -> None:
    @jax.jit
    def rows(slot: jax.Array) -> tuple[jax.Array, jax.Array]:
        evaluation, active = segment_segment_support_distance(
            jnp.zeros(3),
            jnp.asarray((2.0, 0.0, 0.0)),
            jnp.asarray((1.0, 0.3, 0.0)),
            jnp.asarray((3.0, 0.3, 0.0)),
            slot,
        )
        return evaluation.left_witness, active

    witnesses, active = jax.vmap(rows)(jnp.asarray((0, 1), dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(active), (True, True))
    np.testing.assert_allclose(witnesses[:, 0], (1.0, 2.0), atol=1.0e-15)
