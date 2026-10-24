#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Canonical device contact routes: catalogue, keys, epochs and failure closure."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization.contact._cache import CachedContactSearchPlan
from phydrax.discretization.contact._compiled_search import (
    CompiledContactSearchPlan,
    LBVHContactSearchPlan,
    LBVHContactSearchResult,
)
from phydrax.discretization.contact._distance import ContactWitnessSlot
from phydrax.discretization.contact._implicit_geometry import PlaneContactGeometry
from phydrax.discretization.contact._kinematics import evaluate_contact_kinematics
from phydrax.discretization.contact._participant import (
    ContactParticipantScene,
    LinearContactParticipant,
)
from phydrax.discretization.contact._search import (
    ContactCandidateEpoch,
    ContactSearchStatus,
    DenseContactSearchPlan,
    SweepAndPruneContactSearchPlan,
)
from phydrax.discretization.contact._stencils import (
    ContactPrimitiveProfile,
    ContactRouteKeySchema,
    ContactStencilBatch,
    ContactStencilKind,
    pack_contact_stencil_batch,
    prepare_contact_stencil_batch,
)
from phydrax.discretization.contact._surface import (
    CollisionFeatureKind,
    CollisionFeaturePolicy,
    CollisionSurfacePlan,
    ContactFeatureCatalogue,
    ContactPairPolicy,
    PreparedCollisionScene,
    PreparedCollisionSurface,
    selection_collision_operator,
)
from phydrax.discretization.contact._trajectory import ContactPathEnclosure
from phydrax.linalg import ArraySpace


_ACTIVATION = 0.1
_LARGE_ID = 2**62 + 5


def _segment(
    space: ArraySpace,
    vertex_ids: tuple[int, int],
    edge_id: int,
    rows: tuple[int, int],
    rest: tuple[tuple[float, float], tuple[float, float]],
    /,
    *,
    participant: int,
    static: bool = False,
) -> PreparedCollisionSurface:
    plan = CollisionSurfacePlan(
        np.asarray(vertex_ids, dtype=np.int64),
        ambient_dimension=2,
        edges=np.asarray(((0, 1),), dtype=np.int32),
        edge_ids=np.asarray((edge_id,), dtype=np.int64),
        participant_ids=participant,
        body_ids=participant,
        static_mask=static,
    )
    return PreparedCollisionSurface(
        plan,
        jnp.asarray(rest, dtype=jnp.float64),
        selection_collision_operator(space, np.asarray(rows, dtype=np.int32)),
    )


def _crossing_scene(*, swapped: bool = False) -> tuple[PreparedCollisionScene, Array]:
    """Two segments, 0.05 apart; vertex ID 2 and edge ID 2 share a raw value."""
    space = ArraySpace((4, 2), dtype=np.float64)
    upper = _segment(
        space,
        (0, _LARGE_ID),
        2,
        (0, 1),
        ((-0.5, 0.05), (0.5, 0.05)),
        participant=0,
    )
    lower = _segment(
        space,
        (2, 3),
        7,
        (2, 3),
        ((-1.0, 0.0), (1.0, 0.0)),
        participant=1,
        static=True,
    )
    surfaces = (lower, upper) if swapped else (upper, lower)
    scene = PreparedCollisionScene(surfaces)
    positions = jnp.concatenate(tuple(value.rest_positions for value in surfaces))
    return scene, positions


def _dense(
    *,
    edge_vertex: int = 8,
    envelope: float = 0.0,
    sweep: bool = False,
) -> DenseContactSearchPlan | SweepAndPruneContactSearchPlan:
    plan = SweepAndPruneContactSearchPlan if sweep else DenseContactSearchPlan
    return plan(
        edge_vertex_capacity=edge_vertex,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=_ACTIVATION,
        envelope_radius=envelope,
    )


type _Identity = tuple[int, int, int]
type _Route = tuple[int, _Identity, _Identity, int]


def _identities(
    epoch: ContactCandidateEpoch, scene: PreparedCollisionScene
) -> set[_Route]:
    catalogue = scene.feature_catalogue
    rows: set[_Route] = set()
    for batch in epoch.active_batches:
        valid = np.asarray(batch.valid)
        left = np.asarray(batch.left_feature_indices)[valid]
        right = np.asarray(batch.right_feature_indices)[valid]
        kinds = np.asarray(catalogue.feature_kinds)
        participants = np.asarray(catalogue.participant_ids)
        ids = np.asarray(catalogue.feature_ids)
        slots = np.asarray(batch.witness_slots)[valid]
        for first, second, slot in zip(left, right, slots, strict=True):
            rows.add(
                (
                    int(batch.kind),
                    (int(participants[first]), int(kinds[first]), int(ids[first])),
                    (int(participants[second]), int(kinds[second]), int(ids[second])),
                    int(slot),
                )
            )
    return rows


def _valid_keys(batch: ContactStencilBatch) -> list[int]:
    return np.asarray(batch.route_keys)[np.asarray(batch.valid)].tolist()


def test_catalogue_is_typed_permutation_invariant_and_preserves_raw_ids() -> None:
    scene, positions = _crossing_scene()
    swapped, swapped_positions = _crossing_scene(swapped=True)
    epoch = _dense().build(scene, positions)
    swapped_epoch = _dense().build(swapped, swapped_positions)

    assert scene.scene_id != swapped.scene_id
    assert scene.feature_catalogue.catalogue_id == swapped.feature_catalogue.catalogue_id
    assert bool(epoch.successful) and bool(swapped_epoch.successful)
    assert _identities(epoch, scene) == _identities(swapped_epoch, swapped)
    assert sorted(_valid_keys(epoch.edge_vertex)) == sorted(
        _valid_keys(swapped_epoch.edge_vertex)
    )
    # Vertex ID 2 and edge ID 2 coexist; typed identities keep them distinct.
    catalogue = scene.feature_catalogue
    kinds = np.asarray(catalogue.feature_kinds)
    ids = np.asarray(catalogue.feature_ids)
    assert {int(kind) for kind in kinds[ids == 2]} == {
        int(CollisionFeatureKind.VERTEX),
        int(CollisionFeatureKind.EDGE),
    }
    # The 2**62 scale raw ID is carried unchanged, never narrowed or hashed.
    left_ids = np.asarray(epoch.edge_vertex.left_feature_ids)
    assert _LARGE_ID in left_ids[np.asarray(epoch.edge_vertex.valid)].tolist()
    assert np.asarray(epoch.edge_vertex.left_feature_ids).dtype == np.int64


def test_vertex_source_order_does_not_change_route_identity() -> None:
    space = ArraySpace((4, 2), dtype=np.float64)
    static = _segment(
        space, (20, 21), 30, (2, 3), ((-1.0, 0.0), (1.0, 0.0)), participant=1
    )
    forward = _segment(
        space, (10, 11), 31, (0, 1), ((-0.5, 0.05), (0.5, 0.05)), participant=0
    )
    reverse = _segment(
        space, (11, 10), 31, (0, 1), ((0.5, 0.05), (-0.5, 0.05)), participant=0
    )
    first = PreparedCollisionScene((forward, static))
    second = PreparedCollisionScene((reverse, static))
    first_epoch = _dense().build(
        first, jnp.concatenate((forward.rest_positions, static.rest_positions))
    )
    second_epoch = _dense().build(
        second, jnp.concatenate((reverse.rest_positions, static.rest_positions))
    )

    assert first.feature_catalogue.catalogue_id == second.feature_catalogue.catalogue_id
    assert _identities(first_epoch, first) == _identities(second_epoch, second)
    assert _valid_keys(first_epoch.edge_vertex) == _valid_keys(second_epoch.edge_vertex)


def test_catalogue_refuses_ambiguous_or_negative_identities() -> None:
    with pytest.raises(ValueError, match="unique"):
        ContactFeatureCatalogue(
            np.asarray((0, 0)), np.asarray((0, 0)), np.asarray((5, 5))
        )
    with pytest.raises(ValueError, match="nonnegative"):
        ContactFeatureCatalogue(np.asarray((0,)), np.asarray((0,)), np.asarray((-1,)))
    with pytest.raises(ValueError, match="CollisionFeatureKind"):
        ContactFeatureCatalogue(np.asarray((0,)), np.asarray((9,)), np.asarray((1,)))


def test_route_keys_are_injective_bounded_and_decodable() -> None:
    catalogue = ContactFeatureCatalogue(
        np.asarray((0, 0, 1)), np.asarray((0, 1, 3)), np.asarray((4, 4, 4))
    )
    schema = ContactRouteKeySchema(catalogue)
    grid = np.stack(
        np.meshgrid(
            np.arange(len(ContactStencilKind)),
            np.arange(3),
            np.arange(3),
            np.arange(len(ContactWitnessSlot)),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, 4))
    keys = np.asarray(schema.encode(grid[:, 0], grid[:, 1], grid[:, 2], grid[:, 3]))

    assert np.unique(keys).size == keys.size
    assert int(keys.max()) == schema.maximum_key < schema.sentinel
    assert int(keys.min()) == 0
    decoded = np.stack([np.asarray(value) for value in schema.decode(keys)], axis=1)
    np.testing.assert_array_equal(decoded, grid)
    assert schema.catalogue_id == catalogue.catalogue_id


def test_backends_agree_on_canonical_keys_under_jit_and_vmap() -> None:
    scene, positions = _crossing_scene()
    host = _dense().build(scene, positions)
    sweep = _dense(sweep=True).build(scene, positions)
    compiled_plan = CompiledContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=_ACTIVATION,
    )
    lbvh_plan = LBVHContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=_ACTIVATION,
        maximum_traversal_visits=256,
    )
    compiled = eqx.filter_jit(compiled_plan.evaluate)(scene, positions)
    lbvh = eqx.filter_jit(lbvh_plan.evaluate)(scene, positions)

    for epoch in (host, sweep, compiled.epoch, lbvh.epoch):
        assert bool(epoch.successful)
        assert _valid_keys(epoch.edge_vertex) == _valid_keys(host.edge_vertex)
        assert epoch.catalogue_id == scene.feature_catalogue.catalogue_id
        assert epoch.topology_id == scene.scene_id
        keys = _valid_keys(epoch.edge_vertex)
        assert keys == sorted(keys)
    assert int(compiled.evidence.duplicate_route_count) == 0
    assert int(lbvh.evidence.duplicate_route_count) == 0

    lifted = jnp.stack(
        (positions, positions.at[:2, 1].add(1.0), positions.at[:2, 1].add(-0.02))
    )

    def count(values: Array) -> Array:
        return lbvh_plan.evaluate(scene, values).epoch.candidate_count

    counts = jax.vmap(count)(lifted)
    expected = [int(_dense().build(scene, value).candidate_count) for value in lifted]
    np.testing.assert_array_equal(np.asarray(counts), expected)


def test_dynamic_updates_keep_static_identity_and_tree_structure() -> None:
    scene, positions = _crossing_scene()
    plan = CompiledContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=_ACTIVATION,
    )
    near = plan.evaluate(scene, positions, revision=3).epoch
    far = plan.evaluate(scene, positions.at[:2, 1].add(5.0), revision=4).epoch

    assert int(near.candidate_count) > 0
    assert int(far.candidate_count) == 0
    assert jax.tree.structure(near) == jax.tree.structure(far)
    assert near.epoch_id == far.epoch_id
    assert near.edge_vertex.batch_id == far.edge_vertex.batch_id
    assert (int(near.revision), int(far.revision)) == (3, 4)
    host = _dense().build(scene, positions)
    assert host.edge_vertex.batch_id == near.edge_vertex.batch_id


@pytest.mark.parametrize("backend", ["dense", "compiled", "lbvh"])
def test_exact_capacity_succeeds_and_every_overflow_fails_closed(backend: str) -> None:
    scene, positions = _crossing_scene()
    exact = int(_dense().build(scene, positions).edge_vertex.actual_count)
    assert exact >= 2

    def search(capacity: int) -> ContactCandidateEpoch:
        if backend == "dense":
            return _dense(edge_vertex=capacity).build(scene, positions)
        if backend == "compiled":
            return CompiledContactSearchPlan(
                scene,
                edge_vertex_capacity=capacity,
                edge_edge_capacity=0,
                face_vertex_capacity=0,
                activation_distance=_ACTIVATION,
            ).build(scene, positions)
        return LBVHContactSearchPlan(
            scene,
            edge_vertex_capacity=capacity,
            edge_edge_capacity=0,
            face_vertex_capacity=0,
            activation_distance=_ACTIVATION,
            maximum_traversal_visits=256,
        ).build(scene, positions)

    complete = search(exact)
    truncated = search(exact - 1)

    assert bool(complete.successful)
    assert int(complete.edge_vertex.actual_count) == exact
    assert int(jnp.sum(complete.edge_vertex.valid)) == exact
    assert not bool(truncated.successful)
    assert int(truncated.status) == int(ContactSearchStatus.CANDIDATE_OVERFLOW)
    assert int(truncated.edge_vertex.actual_count) == exact
    assert int(truncated.edge_vertex.overflow_count) == 1
    assert bool(truncated.edge_vertex.overflow)
    assert not bool(jnp.any(truncated.edge_vertex.valid))
    kinematics = evaluate_contact_kinematics(
        scene, truncated, positions, jnp.zeros_like(positions), 0.01
    )
    assert not bool(kinematics.evidence.successful)
    assert int(kinematics.evidence.active_contacts) == 0


def test_lbvh_traversal_and_stack_exhaustion_are_incomplete() -> None:
    scene, positions = _crossing_scene()
    visits = LBVHContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=_ACTIVATION,
        maximum_traversal_visits=1,
    ).evaluate(scene, positions)
    depth = LBVHContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=_ACTIVATION,
        maximum_traversal_visits=256,
        maximum_tree_depth=1,
    ).evaluate(scene, positions)

    assert bool(visits.evidence.visit_overflow)
    assert int(visits.evidence.traversal_visits) == 1
    assert int(visits.epoch.status) == int(ContactSearchStatus.TRAVERSAL_EXHAUSTED)
    assert not bool(visits.epoch.successful)
    assert bool(depth.evidence.stack_overflow)
    assert int(depth.epoch.status) == int(ContactSearchStatus.TRAVERSAL_EXHAUSTED)
    assert not bool(depth.epoch.successful)


def test_empty_scene_epochs_are_complete_for_every_backend() -> None:
    scene, positions = _crossing_scene()
    separated = positions.at[:2, 1].add(10.0)
    plans = (
        _dense(),
        _dense(sweep=True),
        CompiledContactSearchPlan(
            scene,
            edge_vertex_capacity=8,
            edge_edge_capacity=0,
            face_vertex_capacity=0,
            activation_distance=_ACTIVATION,
        ),
        LBVHContactSearchPlan(
            scene,
            edge_vertex_capacity=8,
            edge_edge_capacity=0,
            face_vertex_capacity=0,
            activation_distance=_ACTIVATION,
            maximum_traversal_visits=256,
        ),
    )
    for plan in plans:
        epoch = plan.build(scene, separated)
        assert bool(epoch.successful)
        assert int(epoch.candidate_count) == 0
        assert not bool(jnp.any(epoch.edge_vertex.valid))


def test_active_degenerate_witness_fails_kinematics_instead_of_masking() -> None:
    scene, positions = _crossing_scene()
    touching = positions.at[:2, 1].set(0.0)
    epoch = _dense().build(scene, touching)
    kinematics = evaluate_contact_kinematics(
        scene, epoch, touching, jnp.zeros_like(touching), 0.01
    )
    batch = kinematics.batches[0]

    assert bool(epoch.successful)
    assert int(kinematics.evidence.failed_contacts) >= 1
    assert not bool(kinematics.evidence.successful)
    assert not bool(jnp.any(batch.failed & batch.valid))
    separated = _dense().build(scene, positions)
    healthy = evaluate_contact_kinematics(
        scene, separated, positions, jnp.zeros_like(positions), 0.01
    )
    assert bool(healthy.evidence.successful)
    assert int(healthy.evidence.failed_contacts) == 0


def test_routes_and_kinematics_refuse_foreign_catalogues() -> None:
    scene, positions = _crossing_scene()
    space = ArraySpace((4, 2), dtype=np.float64)
    other = PreparedCollisionScene(
        (
            _segment(space, (40, 41), 42, (0, 1), ((0.0, 0.0), (1.0, 0.0)), participant=0),
            _segment(space, (43, 44), 45, (2, 3), ((0.0, 0.05), (1.0, 0.05)), participant=1),
        )
    )
    epoch = _dense().build(scene, positions)
    with pytest.raises(ValueError, match="another scene, catalogue or routing"):
        evaluate_contact_kinematics(
            other, epoch, positions, jnp.zeros_like(positions), 0.01
        )
    compiled = CompiledContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=_ACTIVATION,
    )
    with pytest.raises(ValueError, match="another scene"):
        compiled.evaluate(other, positions)


def _enclosure(lower: Array, upper: Array, *, revision: int) -> ContactPathEnclosure:
    return ContactPathEnclosure(
        lower,
        upper,
        jnp.asarray(0.0, dtype=lower.dtype),
        jnp.asarray(1.0, dtype=lower.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(revision, dtype=jnp.int64),
        jnp.asarray(True),
        jnp.asarray(True),
        "probe-trajectory",
    )


def test_full_path_enclosure_finds_interior_contact_with_clear_endpoints() -> None:
    scene, positions = _crossing_scene()
    lifted = positions.at[:2, 1].set(0.5)
    excursion_lower = lifted.at[:2, 1].set(-0.1)
    enclosure = _enclosure(excursion_lower, lifted, revision=7)
    endpoint_only = _dense().build(scene, lifted, end_positions=lifted)
    host = _dense().build(scene, lifted, path_enclosure=enclosure)
    device = CompiledContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=_ACTIVATION,
    ).build(scene, lifted, path_enclosure=enclosure)

    assert int(endpoint_only.candidate_count) == 0
    assert int(host.candidate_count) > 0
    assert _valid_keys(device.edge_vertex) == _valid_keys(host.edge_vertex)
    assert host.trajectory_id == device.trajectory_id == "probe-trajectory"
    assert int(device.revision) == 7
    np.testing.assert_allclose(device.coverage_lower, excursion_lower)
    assert bool(device.covers(excursion_lower, lifted))


def test_enclosure_that_misses_an_endpoint_is_invalid_input() -> None:
    scene, positions = _crossing_scene()
    wrong = _enclosure(positions + 1.0, positions + 2.0, revision=0)
    device = CompiledContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=_ACTIVATION,
    ).build(scene, positions, path_enclosure=wrong)

    assert int(device.status) == int(ContactSearchStatus.INVALID_INPUT)
    assert not bool(device.successful)
    with pytest.raises(ValueError, match="path enclosure"):
        _dense().build(scene, positions, path_enclosure=wrong)


def test_envelope_covers_both_moving_primitives_of_a_pair() -> None:
    scene, positions = _crossing_scene()
    # Segments 0.27 apart: within the pair radius 0.1 + 2 * 0.1 of the search.
    lifted = positions.at[:2, 1].set(0.27)
    epoch = _dense(envelope=0.1).build(scene, lifted)
    # Both primitives move 0.09 toward each other (each within its envelope).
    approached = lifted.at[:2, 1].add(-0.09).at[2:, 1].add(0.09)

    assert int(epoch.candidate_count) > 0
    assert bool(epoch.covers(lifted, lifted))
    assert bool(epoch.covers(approached, approached))
    assert not bool(epoch.covers(lifted.at[:2, 1].add(-0.11), lifted))
    assert not bool(epoch.covers(lifted, lifted - 1.0))
    exact = _dense().build(scene, approached)
    assert _valid_keys(exact.edge_vertex)
    assert set(_valid_keys(exact.edge_vertex)) <= set(_valid_keys(epoch.edge_vertex))


def test_device_cache_reuses_with_bounded_branch_and_rebinds_revision() -> None:
    scene, positions = _crossing_scene()
    lifted = positions.at[:2, 1].set(0.3)
    search = CompiledContactSearchPlan(
        scene,
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=_ACTIVATION,
        envelope_radius=0.1,
    )
    cache = CachedContactSearchPlan(search, skin=0.1, rebuild_fraction=1.0)
    state = cache.initialize(scene, lifted)
    update = eqx.filter_jit(cache.update)
    reused = update(scene, state, lifted.at[:2, 1].add(-0.05))
    rebuilt = update(scene, reused.candidate, lifted.at[:2, 1].add(-0.25))

    assert bool(reused.reused)
    assert int(reused.candidate.reuse_count) == 1
    assert not bool(rebuilt.reused)
    assert int(rebuilt.candidate.rebuild_count) == 2
    assert bool(rebuilt.successful)
    np.testing.assert_allclose(
        rebuilt.candidate.epoch.reference_positions, lifted.at[:2, 1].add(-0.25)
    )
    enclosure = _enclosure(lifted, lifted, revision=1)
    with pytest.raises(ValueError, match="trajectory identity"):
        cache.update(scene, state, lifted, path_enclosure=enclosure)


def _analytic_scene(
    vertices: list[tuple[float, float, float]],
    radii: list[float],
    edges: list[tuple[int, int]],
) -> tuple[ContactParticipantScene, Array]:
    count = len(vertices)
    space = ArraySpace((count, 3), dtype=np.float64)
    plan = CollisionSurfacePlan(
        np.arange(count, dtype=np.int64),
        ambient_dimension=3,
        edges=np.asarray(edges, dtype=np.int32).reshape((-1, 2)),
        participant_ids=0,
        body_ids=0,
        physical_radius=np.asarray(radii),
        allow_isolated_vertices=True,
    )
    positions = jnp.asarray(vertices, dtype=jnp.float64)
    surface = PreparedCollisionSurface(
        plan,
        positions,
        selection_collision_operator(space, np.arange(count, dtype=np.int32)),
    )
    ground = PlaneContactGeometry(
        jnp.asarray((0.0, 0.0, 1.0)),
        0.0,
        feature_policy=CollisionFeaturePolicy(
            np.asarray((1000,), dtype=np.int64),
            np.asarray((int(CollisionFeatureKind.ANALYTIC),), dtype=np.int32),
            participant_ids=1,
            body_ids=1,
            material_ids=0,
            patch_ids=0,
            static_mask=True,
            provenance_id="ground-plane",
        ),
    )
    scene = ContactParticipantScene((LinearContactParticipant(surface),), planes=(ground,))
    return scene, positions


def _capsule_search(
    scene: ContactParticipantScene, backend: str
) -> DenseContactSearchPlan | CompiledContactSearchPlan | LBVHContactSearchPlan:
    capacities = dict(
        edge_vertex_capacity=4,
        edge_edge_capacity=4,
        face_vertex_capacity=0,
        vertex_vertex_capacity=4,
        vertex_plane_capacity=4,
        edge_plane_capacity=4,
        activation_distance=_ACTIVATION,
        primitive_profile=ContactPrimitiveProfile.SPHERE_CAPSULE,
    )
    if backend == "dense":
        # ty: ignore[invalid-argument-type]
        return DenseContactSearchPlan(**capacities)
    if backend == "compiled":
        # ty: ignore[invalid-argument-type]
        return CompiledContactSearchPlan(scene, **capacities)
    return LBVHContactSearchPlan(
        scene,
        maximum_traversal_visits=256,
        # ty: ignore[invalid-argument-type]
        **capacities,
    )


@pytest.mark.parametrize("backend", ["dense", "compiled", "lbvh"])
def test_capsule_plane_emits_two_endpoint_witnesses_and_sphere_plane_one(
    backend: str,
) -> None:
    scene, positions = _analytic_scene(
        [(2.0, 0.0, 0.15), (-0.5, 0.0, 0.06), (0.5, 0.0, 0.06)],
        [0.1, 0.05, 0.05],
        [(1, 2)],
    )
    epoch = _capsule_search(scene, backend).build(scene, positions)
    kinematics = evaluate_contact_kinematics(
        scene,
        epoch,
        positions,
        jnp.zeros_like(positions),
        0.01,
        rest_positions=positions,
    )
    by_kind = {batch.kind: batch for batch in kinematics.batches}
    capsule = by_kind[ContactStencilKind.EDGE_PLANE]
    sphere = by_kind[ContactStencilKind.VERTEX_PLANE]
    capsule_valid = np.asarray(capsule.valid)

    assert bool(epoch.successful) and bool(kinematics.evidence.successful)
    # Capsule endpoints are not spheres: one sphere row, no vertex-vertex rows.
    assert int(epoch.vertex_plane.actual_count) == 1
    assert int(epoch.vertex_vertex.actual_count) == 0
    assert int(epoch.edge_plane.actual_count) == 2
    np.testing.assert_array_equal(
        np.asarray(capsule.witness_slots)[capsule_valid],
        (int(ContactWitnessSlot.FIRST), int(ContactWitnessSlot.SECOND)),
    )
    np.testing.assert_allclose(np.asarray(capsule.gap)[capsule_valid], (0.01, 0.01))
    np.testing.assert_allclose(
        np.asarray(capsule.left_witness)[capsule_valid], positions[1:3], atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(capsule.normal)[capsule_valid], ((0.0, 0.0, 1.0),) * 2
    )
    np.testing.assert_allclose(np.asarray(sphere.gap)[np.asarray(sphere.valid)], 0.05)
    keys = _valid_keys(epoch.edge_plane)
    assert len(set(keys)) == 2


def test_parallel_capsules_emit_two_ordered_support_witnesses() -> None:
    scene, positions = _analytic_scene(
        [(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0), (-0.5, 0.0, 1.12), (0.5, 0.0, 1.12)],
        [0.05, 0.05, 0.05, 0.05],
        [(0, 1), (2, 3)],
    )
    keys = []
    for backend in ("dense", "compiled", "lbvh"):
        epoch = _capsule_search(scene, backend).build(scene, positions)
        keys.append(_valid_keys(epoch.edge_edge))
        kinematics = evaluate_contact_kinematics(
            scene,
            epoch,
            positions,
            jnp.zeros_like(positions),
            0.01,
            rest_positions=positions,
        )
        batch = {value.kind: value for value in kinematics.batches}[
            ContactStencilKind.EDGE_EDGE
        ]
        valid = np.asarray(batch.valid)

        assert bool(epoch.successful) and bool(kinematics.evidence.successful)
        assert int(epoch.edge_edge.actual_count) == 2
        np.testing.assert_array_equal(np.asarray(batch.witness_slots)[valid], (0, 1))
        np.testing.assert_allclose(np.asarray(batch.gap)[valid], (0.02, 0.02))
        np.testing.assert_allclose(
            np.asarray(batch.left_witness)[valid][:, 0], (-0.5, 0.5), atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(batch.normal)[valid], ((0.0, 0.0, -1.0),) * 2, atol=1e-12
        )
    assert keys[0] == keys[1] == keys[2]


def test_plane_routes_require_declared_capacity() -> None:
    scene, positions = _analytic_scene(
        [(2.0, 0.0, 0.15), (-0.5, 0.0, 0.06), (0.5, 0.0, 0.06)],
        [0.1, 0.05, 0.05],
        [(1, 2)],
    )
    epoch = DenseContactSearchPlan(
        edge_vertex_capacity=4,
        edge_edge_capacity=4,
        face_vertex_capacity=0,
        vertex_plane_capacity=4,
        activation_distance=_ACTIVATION,
        primitive_profile=ContactPrimitiveProfile.SPHERE_CAPSULE,
    ).build(scene, positions)

    assert int(epoch.edge_plane.actual_count) == 2
    assert int(epoch.edge_plane.overflow_count) == 2
    assert int(epoch.status) == int(ContactSearchStatus.CANDIDATE_OVERFLOW)
    assert not bool(epoch.successful)


def _plane_case() -> tuple[ContactParticipantScene, Array]:
    return _analytic_scene(
        [(2.0, 0.0, 0.15), (-0.5, 0.0, 0.06), (0.5, 0.0, 0.06)],
        [0.1, 0.05, 0.05],
        [(1, 2)],
    )


def test_epoch_binding_refuses_numeric_plane_and_radius_changes() -> None:
    scene, positions = _plane_case()
    plan = _capsule_search(scene, "compiled")
    epoch = plan.build(scene, positions)
    raised = eqx.tree_at(
        lambda value: value.planes[0].offset, scene, jnp.asarray(0.5)
    )
    thicker = eqx.tree_at(
        lambda value: value.participants[0].surface.plan.feature_policy.physical_radius,
        scene,
        # Three vertex features and one capsule edge feature.
        jnp.asarray((0.1, 0.2, 0.2, 0.2)),
    )

    assert bool(epoch.successful) and bool(epoch.binds(scene))
    assert raised.scene_id == scene.scene_id
    for changed in (raised, thicker):
        assert not bool(epoch.binds(changed))
        kinematics = evaluate_contact_kinematics(
            changed,
            epoch,
            positions,
            jnp.zeros_like(positions),
            0.01,
            rest_positions=positions,
        )
        assert not bool(kinematics.evidence.successful)
    cached_search = CompiledContactSearchPlan(
        scene,
        edge_vertex_capacity=4,
        edge_edge_capacity=4,
        face_vertex_capacity=0,
        vertex_vertex_capacity=4,
        vertex_plane_capacity=4,
        edge_plane_capacity=4,
        activation_distance=_ACTIVATION,
        envelope_radius=0.05,
        primitive_profile=ContactPrimitiveProfile.SPHERE_CAPSULE,
    )
    cache = CachedContactSearchPlan(cached_search, skin=0.05, rebuild_fraction=1.0)
    state = cache.initialize(scene, positions)
    assert bool(cache.update(scene, state, positions).reused)
    moved_plane = cache.update(raised, state, positions)
    assert not bool(moved_plane.reused)
    assert bool(moved_plane.candidate.epoch.binds(raised))


def test_changed_pair_policy_of_same_size_cannot_reuse_routes() -> None:
    scene, positions = _crossing_scene()
    space = ArraySpace((4, 2), dtype=np.float64)
    excluded: list[PreparedCollisionSurface] = []
    for index, surface in enumerate(scene.surfaces):
        plan = surface.plan
        excluded.append(
            PreparedCollisionSurface(
                CollisionSurfacePlan(
                    np.asarray(plan.vertex_ids),
                    ambient_dimension=2,
                    edges=np.asarray(plan.edges),
                    edge_ids=np.asarray(plan.edge_ids),
                    participant_ids=int(np.asarray(plan.feature_policy.participant_ids)[0]),
                    body_ids=int(np.asarray(plan.feature_policy.body_ids)[0]),
                    static_mask=bool(np.asarray(plan.feature_policy.static_mask)[0]),
                    pair_policy=ContactPairPolicy(
                        2, allowed_participant_pairs=np.asarray(((0, 0), (1, 1)))
                    ),
                ),
                surface.rest_positions,
                selection_collision_operator(
                    space,
                    np.arange(2 * index, 2 * index + 2, dtype=np.int32),
                ),
            )
        )
    restricted = PreparedCollisionScene(tuple(excluded))
    epoch = _dense().build(restricted, positions)

    assert restricted.feature_catalogue.catalogue_id == scene.feature_catalogue.catalogue_id
    assert restricted.routing_id != scene.routing_id
    assert int(epoch.candidate_count) == 0
    with pytest.raises(ValueError, match="another scene, catalogue or routing"):
        evaluate_contact_kinematics(
            scene, epoch, positions, jnp.zeros_like(positions), 0.01
        )


def test_non_unit_plane_normal_fails_search_before_pruning() -> None:
    scene, positions = _plane_case()
    scaled = eqx.tree_at(
        lambda value: value.planes[0].unit_normal, scene, jnp.asarray((0.0, 0.0, 2.0))
    )
    device = _capsule_search(scaled, "compiled").build(scaled, positions)
    lbvh = _capsule_search(scaled, "lbvh").build(scaled, positions)

    assert int(device.status) == int(ContactSearchStatus.INVALID_INPUT)
    assert int(lbvh.status) == int(ContactSearchStatus.INVALID_INPUT)
    assert not bool(device.successful) and not bool(lbvh.successful)
    with pytest.raises(ValueError, match="unit plane normals"):
        _capsule_search(scaled, "dense").build(scaled, positions)


@pytest.mark.parametrize("backend", ["dense", "compiled", "lbvh"])
def test_zero_capacity_is_legal_and_overflows_only_on_real_candidates(
    backend: str,
) -> None:
    scene, positions = _crossing_scene()
    capacities = dict(
        edge_vertex_capacity=0,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=_ACTIVATION,
    )

    def search(values: Array) -> ContactCandidateEpoch:
        if backend == "dense":
            # ty: ignore[invalid-argument-type]
            return DenseContactSearchPlan(**capacities).build(scene, values)
        if backend == "compiled":
            # ty: ignore[invalid-argument-type]
            plan = CompiledContactSearchPlan(scene, **capacities)
            return plan.build(scene, values)
        return LBVHContactSearchPlan(
            scene,
            maximum_traversal_visits=256,
            # ty: ignore[invalid-argument-type]
            **capacities,
        ).build(scene, values)

    separated = search(positions.at[:2, 1].add(10.0))
    touching = search(positions)

    assert bool(separated.successful)
    assert int(separated.candidate_count) == 0
    assert separated.active_batches == ()
    assert not bool(touching.successful)
    assert int(touching.status) == int(ContactSearchStatus.CANDIDATE_OVERFLOW)
    assert int(touching.edge_vertex.overflow_count) == int(
        touching.edge_vertex.actual_count
    )
    assert int(touching.edge_vertex.actual_count) > 0


@pytest.mark.parametrize("backend", ["compiled", "lbvh"])
def test_plane_only_capacities_trace_with_zero_capacity_tree_families(
    backend: str,
) -> None:
    scene, positions = _analytic_scene(
        [(-0.5, 0.0, 0.06), (0.5, 0.0, 0.06)], [0.05, 0.05], [(0, 1)]
    )
    capacities = dict(
        vertex_vertex_capacity=0,
        edge_vertex_capacity=0,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        vertex_plane_capacity=0,
        edge_plane_capacity=2,
        activation_distance=_ACTIVATION,
        primitive_profile=ContactPrimitiveProfile.SPHERE_CAPSULE,
    )
    if backend == "compiled":
        # ty: ignore[invalid-argument-type]
        plan = CompiledContactSearchPlan(scene, **capacities)
    else:
        plan = LBVHContactSearchPlan(
            scene,
            maximum_traversal_visits=64,
            # ty: ignore[invalid-argument-type]
            **capacities,
        )
    result = eqx.filter_jit(plan.evaluate)(scene, positions)

    assert bool(result.epoch.successful)
    assert int(result.epoch.edge_plane.actual_count) == 2
    assert int(result.epoch.candidate_count) == 2
    assert [batch.kind for batch in result.epoch.active_batches] == [
        ContactStencilKind.EDGE_PLANE
    ]
    if isinstance(result, LBVHContactSearchResult):
        assert int(result.evidence.plane_pair_work) == 1
    else:
        assert backend == "compiled"


def test_catalogue_lookups_refuse_out_of_range_addresses_explicitly() -> None:
    catalogue = ContactFeatureCatalogue(
        np.asarray((0, 0, 1)), np.asarray((0, 1, 0)), np.asarray((7, 7, 9))
    )
    features = jnp.asarray((0, 2, 3, -1, 1), dtype=jnp.int32)
    claimed = jnp.asarray((True, True, True, True, False))
    ranks, resolved = catalogue.ranks(features, valid=claimed)
    participant, kind, identifier, known = catalogue.identities(
        jnp.asarray((0, 2, 3, -1)), valid=jnp.asarray((True, True, True, False))
    )

    np.testing.assert_array_equal(resolved, (True, True, False, False, False))
    np.testing.assert_array_equal(np.asarray(ranks)[2:], (-1, -1, -1))
    assert int(ranks[1]) == 2
    np.testing.assert_array_equal(known, (True, True, False, False))
    np.testing.assert_array_equal(identifier, (7, 9, -1, -1))
    np.testing.assert_array_equal(participant, (0, 1, -1, -1))
    np.testing.assert_array_equal(kind, (0, 0, -1, -1))


def test_device_packer_reports_claimed_out_of_range_features() -> None:
    scene, _ = _crossing_scene()
    layout = _dense().layouts(scene).edge_vertex
    upper_vertex, lower_edge = 0, scene.vertex_count + 1
    packing = pack_contact_stencil_batch(
        layout,
        scene.feature_catalogue,
        jnp.asarray(((0, 2, 3, -1), (1, 2, 3, -1)), dtype=jnp.int32),
        jnp.asarray((upper_vertex, scene.feature_count + 5), dtype=jnp.int32),
        jnp.asarray((lower_edge, lower_edge), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.float64),
        jnp.asarray((True, True)),
    )

    assert int(packing.invalid_address_count) == 1
    assert int(jnp.sum(packing.batch.valid)) == 1
    assert int(packing.batch.left_feature_indices[0]) == upper_vertex


def test_host_builder_overflow_makes_every_row_unusable_like_device() -> None:
    scene, positions = _crossing_scene()
    layout = _dense(edge_vertex=2).layouts(scene).edge_vertex
    host = _dense(edge_vertex=2).build(scene, positions).edge_vertex
    rebuilt = prepare_contact_stencil_batch(
        scene,
        layout,
        np.asarray(host.left_feature_indices),
        np.asarray(host.right_feature_indices),
        valid=np.asarray(host.valid),
        actual_count=3,
        overflow_count=1,
    )

    assert bool(jnp.all(host.valid))
    assert bool(rebuilt.overflow)
    assert not bool(jnp.any(rebuilt.valid))
    np.testing.assert_array_equal(rebuilt.route_keys, host.route_keys)
