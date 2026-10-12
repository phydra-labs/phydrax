#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded packing admission, physical-ID reproducibility, and geometry contracts."""

import jax
import numpy as np
import pytest

from phydrax.discretization import (
    ParticleSetPlan,
    PreparedRigidSphereSet,
    RandomSequentialPackingPlan,
    RigidSphereClumpSetPlan,
    RigidSphereSetPlan,
    SphereClumpTemplatePlan,
)


pytestmark = pytest.mark.strict_jax


def _spheres(
    ids: np.ndarray,
    radii: np.ndarray,
    dimension: int,
    *,
    active: np.ndarray | None = None,
) -> PreparedRigidSphereSet:
    particles = ParticleSetPlan(
        ids,
        np.ones(ids.shape[0], dtype=np.float64),
        ambient_dimension=dimension,
        active_mask=active,
    ).prepare()
    return RigidSphereSetPlan(radii, np.zeros(ids.shape[0], dtype=np.int32)).prepare(
        particles
    )


@pytest.mark.parametrize("dimension", [2, 3], ids=["disk-packing", "sphere-packing"])
def test_complete_packing_is_repeatable_contained_and_pairwise_nonoverlapping(
    dimension: int,
) -> None:
    radii = np.asarray([0.2, 0.3, 0.25], dtype=np.float64)
    shape = _spheres(np.asarray([51, 7, 29], dtype=np.int64), radii, dimension)
    plan = RandomSequentialPackingPlan(
        np.full(dimension, -2.0, dtype=np.float64),
        np.full(dimension, 2.0, dtype=np.float64),
        16,
        maximum_candidate_evaluations=144,
    )
    first = plan.initialize(shape, jax.random.key(21))
    second = plan.initialize(shape, jax.random.key(21))
    assert first.complete and first.successful
    assert first.requested_count == 3 and first.placed_count == 3
    np.testing.assert_array_equal(first.accepted_mask, np.ones(3, dtype=np.bool_))
    np.testing.assert_array_equal(
        first.candidate_kinematics.position, second.candidate_kinematics.position
    )
    np.testing.assert_array_equal(
        first.candidate_kinematics.orientation, second.candidate_kinematics.orientation
    )
    np.testing.assert_array_equal(first.attempt_count, second.attempt_count)
    np.testing.assert_array_equal(
        first.accepted_kinematics.position, first.candidate_kinematics.position
    )
    assert np.all(np.asarray(first.attempt_count) >= 1)
    assert np.all(np.asarray(first.attempt_count) <= 16)
    assert np.all(np.asarray(first.bounds_margin) >= 0.0)
    assert first.candidate_evaluation_count <= first.required_evaluation_budget
    assert first.required_evaluation_budget == 144
    position = np.asarray(first.accepted_kinematics.position)
    assert np.all(position - radii[:, None] >= -2.0)
    assert np.all(position + radii[:, None] <= 2.0)
    left, right = np.triu_indices(3, 1)
    distance = np.linalg.norm(position[left] - position[right], axis=-1)
    assert np.all(distance >= radii[left] + radii[right])
    if dimension == 3:
        np.testing.assert_allclose(
            np.linalg.norm(first.accepted_kinematics.orientation, axis=-1),
            1.0,
            atol=1e-14,
        )


def test_packing_is_permutation_stable_for_full_64_bit_physical_ids() -> None:
    ids = np.asarray([2**40 + 7, 7, 93], dtype=np.int64)
    radii = np.asarray([0.2, 0.3, 0.25], dtype=np.float64)
    permutation = np.asarray([2, 0, 1], dtype=np.int32)
    original = _spheres(ids, radii, 3)
    permuted = _spheres(ids[permutation], radii[permutation], 3)
    plan = RandomSequentialPackingPlan(
        np.full(3, -3.0, dtype=np.float64),
        np.full(3, 3.0, dtype=np.float64),
        12,
        maximum_candidate_evaluations=108,
    )
    result = plan.initialize(original, jax.random.key(31))
    reordered = plan.initialize(permuted, jax.random.key(31))
    assert result.successful and reordered.successful
    np.testing.assert_array_equal(
        result.accepted_kinematics.position[permutation],
        reordered.accepted_kinematics.position,
    )
    np.testing.assert_array_equal(
        result.accepted_kinematics.orientation[permutation],
        reordered.accepted_kinematics.orientation,
    )
    np.testing.assert_array_equal(
        result.attempt_count[permutation], reordered.attempt_count
    )
    np.testing.assert_array_equal(
        result.bounds_margin[permutation], reordered.bounds_margin
    )
    assert result.candidate_evaluation_count == reordered.candidate_evaluation_count


def test_distinct_high_id_words_receive_distinct_pose_streams() -> None:
    # Separate one-body runs remove insertion/overlap effects from the RNG contract.
    lower_id = _spheres(
        np.asarray([7], dtype=np.int64), np.asarray([0.2], dtype=np.float64), 3
    )
    higher_id = _spheres(
        np.asarray([2**40 + 7], dtype=np.int64), np.asarray([0.2], dtype=np.float64), 3
    )
    plan = RandomSequentialPackingPlan(
        np.full(3, -3.0, dtype=np.float64),
        np.full(3, 3.0, dtype=np.float64),
        1,
        maximum_candidate_evaluations=0,
    )
    low = plan.initialize(lower_id, jax.random.key(31))
    high = plan.initialize(higher_id, jax.random.key(31))
    assert low.successful and high.successful
    assert not np.array_equal(
        low.accepted_kinematics.position, high.accepted_kinematics.position
    )
    assert not np.array_equal(
        low.accepted_kinematics.orientation, high.accepted_kinematics.orientation
    )
    assert low.candidate_evaluation_count == 0 and high.candidate_evaluation_count == 0


def test_incomplete_packing_exhausts_bounded_tries_and_admits_no_partial_state() -> None:
    shape = _spheres(
        np.asarray([17, 41], dtype=np.int64), np.full(2, 0.5, dtype=np.float64), 3
    )
    plan = RandomSequentialPackingPlan(
        np.full(3, -0.51, dtype=np.float64),
        np.full(3, 0.51, dtype=np.float64),
        5,
        maximum_candidate_evaluations=10,
    )
    result = plan.initialize(shape, jax.random.key(1))
    assert not result.complete and not result.successful
    assert result.requested_count == 2 and result.placed_count == 1
    np.testing.assert_array_equal(result.attempt_count, [1, 5])
    np.testing.assert_array_equal(result.placed_mask, [True, False])
    np.testing.assert_array_equal(result.accepted_mask, [False, False])
    assert np.all(np.asarray(result.geometry_valid))
    assert np.all(np.isnan(np.asarray(result.accepted_kinematics.position)))
    assert np.all(np.isnan(np.asarray(result.accepted_kinematics.orientation)))
    assert result.candidate_evaluation_count == 5
    assert result.candidate_evaluation_count <= 10


def test_degenerate_narrow_phase_refuses_instead_of_inventing_contact_normal() -> None:
    shape = _spheres(
        np.asarray([17, 41], dtype=np.int64), np.full(2, 0.5, dtype=np.float64), 3
    )
    plan = RandomSequentialPackingPlan(
        np.full(3, -0.5, dtype=np.float64),
        np.full(3, 0.5, dtype=np.float64),
        3,
        maximum_candidate_evaluations=6,
    )
    result = plan.initialize(shape, jax.random.key(1))
    assert not result.successful and not result.complete
    np.testing.assert_array_equal(result.geometry_valid, [True, False])
    np.testing.assert_array_equal(result.attempt_count, [1, 3])
    np.testing.assert_array_equal(result.accepted_mask, [False, False])
    assert np.all(np.isnan(np.asarray(result.accepted_kinematics.position)))


def test_body_too_large_for_domain_remains_unplaced_with_negative_bounds_margin() -> None:
    shape = _spheres(
        np.asarray([17], dtype=np.int64), np.asarray([1.0], dtype=np.float64), 2
    )
    plan = RandomSequentialPackingPlan(
        np.zeros(2, dtype=np.float64),
        np.ones(2, dtype=np.float64),
        4,
        maximum_candidate_evaluations=0,
    )
    result = plan.initialize(shape, jax.random.key(5))
    assert not result.successful and result.placed_count == 0
    assert result.attempt_count[0] == 4
    assert result.bounds_margin[0] < 0.0
    assert not result.accepted_mask[0]
    assert result.candidate_evaluation_count == 0


def test_capacity_budget_is_refused_before_candidate_geometry_allocation() -> None:
    shape = _spheres(
        np.asarray([17, 41, 63], dtype=np.int64), np.full(3, 0.2, dtype=np.float64), 3
    )
    plan = RandomSequentialPackingPlan(
        np.zeros(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        7,
        maximum_candidate_evaluations=62,
    )
    with pytest.raises(ValueError, match="requires 63, exceeding 62"):
        plan.initialize(shape, jax.random.key(0))


def test_legacy_untyped_prng_keys_are_refused_at_public_boundary() -> None:
    shape = _spheres(
        np.asarray([17], dtype=np.int64), np.asarray([0.2], dtype=np.float64), 3
    )
    plan = RandomSequentialPackingPlan(
        np.zeros(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        1,
        maximum_candidate_evaluations=0,
    )
    with pytest.raises(TypeError):
        plan.initialize(shape, jax.random.PRNGKey(0))


def test_inactive_capacity_does_not_consume_tries_or_prevent_complete_admission() -> None:
    shape = _spheres(
        np.asarray([17, 41], dtype=np.int64),
        np.asarray([0.2, 100.0], dtype=np.float64),
        2,
        active=np.asarray([True, False], dtype=np.bool_),
    )
    plan = RandomSequentialPackingPlan(
        np.zeros(2, dtype=np.float64),
        np.ones(2, dtype=np.float64),
        2,
        maximum_candidate_evaluations=4,
    )
    result = plan.initialize(shape, jax.random.key(5))
    assert result.successful and result.complete
    assert result.requested_count == 1 and result.placed_count == 1
    np.testing.assert_array_equal(result.attempt_count, [1, 0])
    np.testing.assert_array_equal(result.accepted_mask, [True, False])


def test_clump_packing_uses_component_narrow_phase_not_overlapping_bounding_balls() -> (
    None
):
    template = SphereClumpTemplatePlan(
        np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        np.full(2, 0.15, dtype=np.float64),
        np.full(2, 0.5, dtype=np.float64),
        np.zeros(2, dtype=np.int32),
    )
    particles = ParticleSetPlan(
        np.asarray([17, 41], dtype=np.int64),
        np.ones(2, dtype=np.float64),
        ambient_dimension=3,
    ).prepare()
    shape = RigidSphereClumpSetPlan(
        (template,),
        np.zeros(2, dtype=np.int32),
        np.zeros(2, dtype=np.int32),
    ).prepare(particles)
    plan = RandomSequentialPackingPlan(
        np.full(3, -1.16, dtype=np.float64),
        np.full(3, 1.16, dtype=np.float64),
        16,
        maximum_candidate_evaluations=32,
    )
    result = plan.initialize(shape, jax.random.key(13))
    assert result.complete and result.successful
    centers = np.asarray(result.accepted_kinematics.position)
    assert np.linalg.norm(centers[0] - centers[1]) < 2.3
    components = shape.component_kinematics(result.accepted_kinematics)
    points = np.asarray(components.position)
    distance = np.linalg.norm(points[0, :, None, :] - points[1, None, :, :], axis=-1)
    assert np.all(distance >= 0.3)
    assert result.candidate_evaluation_count > 0
    np.testing.assert_array_equal(result.accepted_mask, [True, True])


def test_try_budget_that_cannot_be_represented_by_attempt_evidence_is_refused() -> None:
    with pytest.raises(ValueError, match="int32 attempt capacity"):
        RandomSequentialPackingPlan(
            np.zeros(3, dtype=np.float64),
            np.ones(3, dtype=np.float64),
            2**31,
            maximum_candidate_evaluations=0,
        )
