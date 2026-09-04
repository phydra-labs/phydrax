#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.applications.contact._cone import (
    ContactConeSolverPlan,
    project_signorini_coulomb_product,
)
from phydrax.applications.contact._rod_contact_lifecycle import (
    CompositeContactParticipantBlock,
    CompositeContactResponse,
    RodContactCCDPlan,
    RodContactManifoldState,
    RodContactSearchFailure,
    RodContactSearchPlan,
    RodContactWitnessBatch,
)
from phydrax.discretization.contact._implicit_geometry import PlaneContactGeometry
from phydrax.discretization.contact._stencils import ContactStencilKind
from phydrax.discretization.contact._surface import (
    CollisionFeatureKind,
    CollisionFeaturePolicy,
    CollisionSurfacePlan,
    ContactPairPolicy,
)
from phydrax.linalg import ArraySpace, DenseLinearOperator, DualSpace


def _rod_surface(
    vertex_count: int,
    edges: np.ndarray,
    /,
    *,
    radius: float = 0.1,
    pair_policy: ContactPairPolicy | None = None,
) -> CollisionSurfacePlan:
    return CollisionSurfacePlan(
        np.arange(vertex_count, dtype=np.int64),
        ambient_dimension=3,
        edges=edges,
        edge_ids=np.arange(100, 100 + edges.shape[0], dtype=np.int64),
        participant_ids=0,
        body_ids=0,
        material_ids=0,
        patch_ids=0,
        physical_radius=radius,
        pair_policy=pair_policy,
    )


def _plane(feature_id: int = 1000) -> PlaneContactGeometry:
    policy = CollisionFeaturePolicy(
        np.asarray((feature_id,), dtype=np.int64),
        np.asarray((int(CollisionFeatureKind.ANALYTIC),), dtype=np.int32),
        participant_ids=1,
        body_ids=1,
        material_ids=1,
        patch_ids=0,
        static_mask=True,
        provenance_id=f"plane-{feature_id}",
    )
    return PlaneContactGeometry((0.0, 0.0, 1.0), 0.0, feature_policy=policy)


def _active_keys(result) -> np.ndarray:
    valid = np.asarray(result.witnesses.valid)
    return np.asarray(result.witnesses.route_keys)[valid]


def _history_witness(
    keys: tuple[int, ...],
    normals: np.ndarray,
    tangent_basis: np.ndarray,
) -> RodContactWitnessBatch:
    count = len(keys)
    indices = np.tile(np.asarray((0, 1, 2, 3), dtype=np.int32), (count, 1))
    zeros = np.zeros((count,), dtype=np.float64)
    left_center = np.zeros((count, 3), dtype=np.float64)
    right_center = left_center - normals
    left_surface = left_center - 0.1 * normals
    right_surface = right_center + 0.1 * normals
    left_axis = tangent_basis[:, :, 0]
    right_axis = tangent_basis[:, :, 1]
    coefficients = np.tile(
        np.asarray((1.0, 0.0, -1.0, 0.0), dtype=np.float64), (count, 1)
    )
    return RodContactWitnessBatch(
        indices,
        np.arange(10, 10 + count, dtype=np.int64),
        np.arange(20, 20 + count, dtype=np.int64),
        np.asarray(keys, dtype=np.int64),
        np.full((count,), int(ContactStencilKind.EDGE_EDGE), dtype=np.int32),
        np.arange(count, dtype=np.int32),
        np.arange(count, dtype=np.int32),
        zeros,
        zeros,
        coefficients,
        left_center,
        right_center,
        left_surface,
        right_surface,
        left_axis,
        right_axis,
        normals,
        tangent_basis,
        np.ones((count,), dtype=np.float64),
        np.full((count,), 0.8, dtype=np.float64),
        np.full((count,), 0.8, dtype=np.float64),
        np.full((count,), 0.1, dtype=np.float64),
        np.full((count,), 0.1, dtype=np.float64),
        np.ones((count,), dtype=bool),
        capacity=count,
    )


def test_signorini_coulomb_projection_preserves_normal_complementarity() -> None:
    projected = project_signorini_coulomb_product(
        jnp.asarray((1.0, 2.0)),
        jnp.asarray(0.5),
    )

    np.testing.assert_allclose(np.asarray(projected), np.asarray((1.0, 0.5)))


def test_dense_and_lbvh_capsule_search_have_identical_stable_routes() -> None:
    edges = np.asarray(((0, 1), (1, 2), (2, 3)), dtype=np.int32)
    surface = _rod_surface(4, edges)
    positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.15, 0.0),
        ),
        dtype=np.float64,
    )
    dense = RodContactSearchPlan(
        capacity=4, activation_distance=0.05, route="dense"
    ).prepare(surface)
    lbvh = RodContactSearchPlan(
        capacity=4,
        activation_distance=0.05,
        route="lbvh",
        maximum_traversal_visits=256,
    ).prepare(surface)

    dense_result = dense.search(positions)
    lbvh_result = lbvh.search(positions)

    assert bool(dense_result.successful)
    assert bool(lbvh_result.successful)
    np.testing.assert_array_equal(_active_keys(dense_result), _active_keys(lbvh_result))
    np.testing.assert_array_equal(
        np.asarray(dense_result.epoch.edge_edge.vertex_indices),
        np.asarray(lbvh_result.epoch.edge_edge.vertex_indices),
    )
    assert int(dense_result.evidence.adjacency_filtered_count) == 2


def test_self_adjacency_and_explicit_vertex_pair_exclusions_are_authoritative() -> None:
    edges = np.asarray(((0, 1), (1, 2), (2, 3)), dtype=np.int32)
    pair_policy = ContactPairPolicy(
        4, excluded_vertex_pairs=np.asarray(((0, 3),), dtype=np.int64)
    )
    surface = _rod_surface(4, edges, pair_policy=pair_policy)
    positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.1, 0.0),
        )
    )
    result = (
        RodContactSearchPlan(capacity=4, activation_distance=0.05)
        .prepare(surface)
        .search(positions)
    )

    assert bool(result.successful)
    assert int(result.evidence.candidate_count) == 0
    assert int(result.evidence.adjacency_filtered_count) == 3


def test_search_capacity_overflow_is_fail_closed_with_required_evidence() -> None:
    edges = np.asarray(((0, 1), (2, 3), (4, 5)), dtype=np.int32)
    surface = _rod_surface(6, edges, radius=0.2)
    positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.1, 0.0),
            (1.0, 0.1, 0.0),
            (0.0, 0.2, 0.0),
            (1.0, 0.2, 0.0),
        )
    )
    result = (
        RodContactSearchPlan(capacity=1, activation_distance=0.01)
        .prepare(surface)
        .search(positions)
    )

    assert not bool(result.successful)
    assert int(result.evidence.failure) == int(RodContactSearchFailure.CAPACITY_OVERFLOW)
    assert int(result.evidence.required_capacity) == 3
    assert int(result.evidence.overflow_count) == 2
    assert not np.any(np.asarray(result.epoch.edge_edge.valid))


def test_static_plane_candidates_share_canonical_epoch_and_witness_order() -> None:
    edges = np.asarray(((0, 1), (2, 3)), dtype=np.int32)
    surface = _rod_surface(4, edges)
    positions = np.asarray(
        ((0.0, 0.0, 0.15), (1.0, 0.0, 0.2), (0.0, 2.0, 1.0), (1.0, 2.0, 1.0))
    )
    prepared = RodContactSearchPlan(
        capacity=2, plane_capacity=2, activation_distance=0.1
    ).prepare(surface, planes=(_plane(),))

    result = prepared.search(positions)

    assert bool(result.successful)
    assert result.plane_witnesses is not None
    valid = np.asarray(result.witnesses.valid)
    assert int(np.count_nonzero(valid)) == 1
    assert np.asarray(result.witnesses.stencil_kinds)[valid].item() == int(
        ContactStencilKind.EDGE_VERTEX
    )
    np.testing.assert_allclose(
        np.asarray(result.witnesses.physical_gap)[valid], np.asarray((0.05,))
    )
    np.testing.assert_array_equal(
        np.asarray(result.epoch.edge_vertex.route_keys)[
            np.asarray(result.epoch.edge_vertex.valid)
        ],
        _active_keys(result),
    )


def test_manifold_birth_death_reorder_material_reset_and_objective_transport() -> None:
    normal = np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)))
    frame = np.asarray(
        (
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
        )
    )
    first = _history_witness((20, 10), normal, frame)
    state = RodContactManifoldState.empty(3)
    born = state.update(first, material_revision=7, retention_steps=2)
    assert int(born.born_count) == 2
    stored = born.state.record_response(
        first.route_keys,
        np.asarray(((2.0, 0.4, -0.2), (1.0, -0.3, 0.1))),
        np.asarray((False, True)),
        np.asarray(((0.2, 0.0), (0.0, 0.0))),
        step_size=0.5,
    )

    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    next_normal = normal @ rotation.T
    next_frame = np.einsum("ij,cjk->cik", rotation, frame)
    second = _history_witness((10, 30), next_normal, next_frame)
    transition = stored.update(second, material_revision=7, retention_steps=2)

    assert int(transition.born_count) == 1
    np.testing.assert_array_equal(
        np.asarray(transition.state.route_keys)[np.asarray(transition.state.occupied)],
        np.asarray((10, 20, 30)),
    )
    np.testing.assert_allclose(
        np.asarray(transition.warm_start)[0], np.asarray((1.0, -0.3, 0.1))
    )
    old_world_tangent = frame[1] @ np.asarray((-0.3, 0.1))
    new_world_tangent = (
        np.asarray(transition.witnesses.tangent_basis)[0]
        @ np.asarray(transition.warm_start)[0, 1:]
    )
    np.testing.assert_allclose(new_world_tangent, rotation @ old_world_tangent)

    changed = transition.state.update(second, material_revision=8, retention_steps=2)
    np.testing.assert_allclose(np.asarray(changed.warm_start), 0.0)
    assert int(changed.material_changed_count) == 2
    assert 20 in np.asarray(changed.died_keys).tolist()
    once_missing = changed.state.update(
        _history_witness((10,), next_normal[:1], next_frame[:1]),
        material_revision=8,
        retention_steps=2,
    )
    twice_missing = once_missing.state.update(
        _history_witness((10,), next_normal[:1], next_frame[:1]),
        material_revision=8,
        retention_steps=2,
    )
    assert 30 in np.asarray(twice_missing.died_keys).tolist()


def test_ccd_detects_analytic_capsule_plane_impact_without_shortening_silently() -> None:
    surface = _rod_surface(2, np.asarray(((0, 1),), dtype=np.int32))
    start = np.asarray(((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)))
    end = np.asarray(((0.0, 0.0, -1.0), (1.0, 0.0, -1.0)))
    search = RodContactSearchPlan(
        capacity=1, plane_capacity=1, activation_distance=0.0
    ).prepare(surface, planes=(_plane(),))

    result = RodContactCCDPlan(
        maximum_iterations=64,
        distance_tolerance=1.0e-10,
        safety_fraction=0.9,
    ).evaluate(search, start, end)

    assert bool(result.successful)
    assert bool(result.evidence.impact_detected)
    assert not bool(result.evidence.full_step_safe)
    assert not bool(result.evidence.certified_safe_prefix)
    assert float(result.safe_step_fraction) <= float(result.impact_fraction)

    np.testing.assert_allclose(float(result.impact_fraction), 0.45, atol=1.0e-7)


def test_supported_initial_plane_contact_cannot_mask_later_penetration() -> None:
    surface = _rod_surface(2, np.asarray(((0, 1),), dtype=np.int32))
    start = np.asarray(((0.0, 0.0, 0.1), (1.0, 0.0, 1.1)))
    separated_end = np.asarray(((0.0, 0.0, 1.1), (1.0, 0.0, 0.1)))
    penetrating_end = np.asarray(((0.0, 0.0, 1.1), (1.0, 0.0, -0.1)))
    search = RodContactSearchPlan(
        capacity=1,
        plane_capacity=1,
        activation_distance=0.0,
    ).prepare(surface, planes=(_plane(),))
    initial = search.search(start)
    supported_keys = initial.witnesses.route_keys[initial.witnesses.valid]
    ccd = RodContactCCDPlan(
        maximum_iterations=64,
        distance_tolerance=1.0e-10,
        safety_fraction=0.9,
    )

    separated = ccd.evaluate(
        search,
        start,
        separated_end,
        supported_initial_plane_route_keys=supported_keys,
    )
    penetrating = ccd.evaluate(
        search,
        start,
        penetrating_end,
        supported_initial_plane_route_keys=supported_keys,
    )

    assert bool(separated.evidence.full_step_safe)
    assert not bool(penetrating.evidence.full_step_safe)
    assert bool(penetrating.evidence.impact_detected)


def test_ccd_detects_transient_capsule_crossing_with_separated_endpoint() -> None:
    surface = _rod_surface(
        4,
        np.asarray(((0, 1), (2, 3)), dtype=np.int32),
    )
    start = np.asarray(
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
    )
    end = start.copy()
    end[:2, 1] = 1.0
    search = RodContactSearchPlan(
        capacity=1,
        activation_distance=0.0,
    ).prepare(surface)

    result = RodContactCCDPlan(
        maximum_iterations=64,
        distance_tolerance=1.0e-10,
        safety_fraction=0.9,
    ).evaluate(search, start, end)

    assert bool(result.successful)
    assert bool(result.evidence.impact_detected)
    assert not bool(result.evidence.full_step_safe)
    assert np.min(end[:2, 1]) > np.max(end[2:, 1])
    np.testing.assert_allclose(float(result.impact_fraction), 0.4, atol=1.0e-7)


def test_composite_response_is_true_dual_action_reaction_and_dense_parity() -> None:
    normal = np.asarray(((1.0, 0.0, 0.0),))
    frame = np.asarray((((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),))
    witnesses = _history_witness((42,), normal, frame)
    dtype = witnesses.normal.dtype
    tangent = ArraySpace((3,), dtype=dtype)
    contact = ArraySpace((1, 3), dtype=dtype)
    identity = jnp.eye(3, dtype=dtype)
    first_g = DenseLinearOperator(identity, source=tangent, target=contact)
    second_g = DenseLinearOperator(-identity, source=tangent, target=contact)
    first_inverse_mass = DenseLinearOperator(
        identity, source=DualSpace(tangent), target=tangent
    )
    second_inverse_mass = DenseLinearOperator(
        identity, source=DualSpace(tangent), target=tangent
    )
    blocks = (
        CompositeContactParticipantBlock(
            first_g,
            first_inverse_mass,
            jnp.asarray((-0.5, 0.2, 0.0), dtype=dtype),
        ),
        CompositeContactParticipantBlock(
            second_g,
            second_inverse_mass,
            jnp.asarray((0.5, 0.0, 0.0), dtype=dtype),
        ),
    )
    response = CompositeContactResponse(
        blocks,
        witnesses,
        dynamic_friction=0.5,
        static_friction=0.6,
        solver=ContactConeSolverPlan(
            maximum_iterations=400,
            absolute_tolerance=1.0e-9,
            relative_tolerance=1.0e-7,
        ),
    )

    matrix_free = response.solve()
    dense = response.solve_dense_authority()

    assert bool(matrix_free.successful)
    assert bool(dense.successful)
    np.testing.assert_allclose(
        np.asarray(matrix_free.impulse),
        np.asarray(dense.impulse),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(matrix_free.generalized_impulses[0])
        + np.asarray(matrix_free.generalized_impulses[1]),
        0.0,
        atol=1.0e-10,
    )
    assert bool(matrix_free.evidence.duality_valid)
    assert float(matrix_free.post_contact_velocity[0, 0]) >= -1.0e-7
