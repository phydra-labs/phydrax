#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..spatial import morton_encode_integer
from ._participant import ContactParticipantScene
from ._search import (
    contact_route_capacity_bounds,
    contact_search_capacities,
    box_distance_squared,
    contact_epoch_identity,
    contact_route_families,
    ContactCandidateEpoch,
    ContactRouteFamily,
    ContactRouteLayouts,
    ContactSearchScene,
    ContactSearchStatus,
    ROUTE_KIND_ORDER,
    route_batch_tuple,
    RouteBatches,
    scene_route_geometry,
    DenseContactSearchPlan,
    enclosure_admits,
    enclosure_binding,
    feature_boxes,
    legal_route_pairs,
    plane_box_clearance,
    route_pair_radius,
    scene_excluded_vertex_pairs,
    SweepAndPruneContactSearchPlan,
    vertex_path_boxes,
)
from ._stencils import (
    contact_stencil_symmetric,
    ContactPrimitiveProfile,
    ContactStencilBatch,
    ContactStencilKind,
    ContactStencilLayout,
    pack_contact_stencil_batch,
    route_endpoints,
    scene_feature_endpoints,
)
from ._surface import CollisionFeatureKind, PreparedCollisionScene
from ._trajectory import ContactPathEnclosure


_PLANE_KINDS = (ContactStencilKind.VERTEX_PLANE, ContactStencilKind.EDGE_PLANE)


class PreparedRouteCandidates(StrictModule, NonTrainableState):
    """Every legal oriented witness row of one route family (topology only).

    Prepared once on the host from the scene topology, pair policy and
    exclusions; runtime only evaluates geometric activation on these rows.
    """

    endpoints: Array
    left_features: Array
    right_features: Array
    witness_slots: Array
    kind: ContactStencilKind = eqx.field(static=True)


def prepare_route_candidates(
    scene: ContactSearchScene,
    family: ContactRouteFamily,
    endpoint_table: np.ndarray,
    excluded: np.ndarray,
    /,
) -> PreparedRouteCandidates:
    """Enumerate legal oriented candidate rows of a route family."""
    kind, left_set, right_set, slots = family
    left = np.repeat(left_set, right_set.size)
    right = np.tile(right_set, left_set.size)
    legal = legal_route_pairs(
        scene, kind, left, right, endpoints=endpoint_table, excluded=excluded
    )
    left = np.repeat(left[legal], slots)
    right = np.repeat(right[legal], slots)
    witness = np.tile(np.arange(slots, dtype=np.int64), left.size // slots)
    table = jnp.asarray(endpoint_table, dtype=jnp.int32)
    endpoints = route_endpoints(
        kind,
        table,
        jnp.asarray(left, dtype=jnp.int32),
        jnp.asarray(right, dtype=jnp.int32),
    ).reshape((-1, 4))
    return PreparedRouteCandidates(
        endpoints.astype(jnp.int32),
        jnp.asarray(left, dtype=jnp.int32),
        jnp.asarray(right, dtype=jnp.int32),
        jnp.asarray(witness, dtype=jnp.int32),
        kind,
    )


def empty_route_candidates(kind: ContactStencilKind, /) -> PreparedRouteCandidates:
    return PreparedRouteCandidates(
        jnp.zeros((0, 4), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int32),
        kind,
    )


def route_activation_mask(
    scene: ContactSearchScene,
    candidates: PreparedRouteCandidates,
    feature_lower: Array,
    feature_upper: Array,
    extents: Array,
    plane_normals: Array,
    plane_offsets: Array,
    activation_distance: float,
    envelope_radius: float,
    /,
) -> Array:
    """Device activation test of prepared rows against full-path feature boxes.

    Plane rows use the exact box support bound and therefore require validated
    unit normals (see :func:`scene_route_geometry`).
    """
    left = candidates.left_features
    right = candidates.right_features
    radius = route_pair_radius(
        activation_distance,
        envelope_radius,
        extents[left],
        extents[right],
        candidates.kind in _PLANE_KINDS,
    )
    if candidates.kind in _PLANE_KINDS:
        plane = right - (scene.feature_count - scene.plane_count)
        clearance = plane_box_clearance(
            feature_lower[left],
            feature_upper[left],
            plane_normals[plane],
            plane_offsets[plane],
        )
        return clearance <= radius
    distance = box_distance_squared(
        feature_lower[left], feature_upper[left], feature_lower[right], feature_upper[right]
    )
    return distance <= radius * radius


def _epoch_status(
    admitted: Array,
    traversal_exhausted: Array,
    overflow: Array,
    duplicates: Array,
    invalid_addresses: Array,
    /,
) -> Array:
    return jnp.where(
        ~admitted,
        int(ContactSearchStatus.INVALID_INPUT),
        jnp.where(
            invalid_addresses > 0,
            int(ContactSearchStatus.INVALID_ROUTE_ADDRESS),
            jnp.where(
                traversal_exhausted,
                int(ContactSearchStatus.TRAVERSAL_EXHAUSTED),
                jnp.where(
                    overflow,
                    int(ContactSearchStatus.CANDIDATE_OVERFLOW),
                    jnp.where(
                        duplicates > 0,
                        int(ContactSearchStatus.DUPLICATE_ROUTE),
                        int(ContactSearchStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)


def _assemble_epoch(
    scene: ContactSearchScene,
    search_id: str,
    epoch_id: str,
    batches: RouteBatches,
    start: Array,
    lower: Array,
    upper: Array,
    envelope_radius: float,
    path_enclosure: ContactPathEnclosure | None,
    revision: ArrayLike | None,
    geometry: tuple[Array, Array, Array],
    status: Array,
    /,
) -> ContactCandidateEpoch:
    lower_time, upper_time, revision_, trajectory_id = enclosure_binding(
        path_enclosure, revision, start.dtype
    )
    candidate_count = sum(
        (batch.actual_count for batch in batches), jnp.asarray(0, dtype=jnp.int32)
    )
    vertex_vertex, edge_vertex, edge_edge, face_vertex, vertex_plane, edge_plane = (
        batches
    )
    contact_extents, plane_normals, plane_offsets = geometry
    return ContactCandidateEpoch(
        vertex_vertex=vertex_vertex,
        edge_vertex=edge_vertex,
        edge_edge=edge_edge,
        face_vertex=face_vertex,
        vertex_plane=vertex_plane,
        edge_plane=edge_plane,
        reference_positions=start,
        coverage_lower=lower,
        coverage_upper=upper,
        envelope_radius=jnp.asarray(envelope_radius, dtype=start.dtype),
        lower_time=lower_time,
        upper_time=upper_time,
        revision=revision_,
        contact_extents=contact_extents,
        plane_normals=plane_normals,
        plane_offsets=plane_offsets,
        candidate_count=candidate_count,
        status=status,
        complete=status == int(ContactSearchStatus.SUCCESS),
        search_id=search_id,
        catalogue_id=scene.feature_catalogue.catalogue_id,
        topology_id=scene.scene_id,
        routing_id=scene.routing_id,
        trajectory_id=trajectory_id,
        epoch_id=epoch_id,
    )


def _check_device_plan_inputs(
    scene: ContactSearchScene,
    scene_id: str,
    /,
) -> None:
    if not isinstance(scene, (PreparedCollisionScene, ContactParticipantScene)):
        raise TypeError(
            "scene must be PreparedCollisionScene or ContactParticipantScene."
        )
    if scene.scene_id != scene_id:
        raise ValueError("Device contact search plan belongs to another scene.")


def _validated_search_scalars(
    activation_distance: float,
    envelope_radius: float,
    primitive_profile: ContactPrimitiveProfile,
    /,
) -> tuple[float, float]:
    activation = float(activation_distance)
    envelope = float(envelope_radius)
    if not isfinite(activation) or activation <= 0.0:
        raise ValueError("activation_distance must be finite and positive.")
    if not isfinite(envelope) or envelope < 0.0:
        raise ValueError("envelope_radius must be finite and nonnegative.")
    if not isinstance(primitive_profile, ContactPrimitiveProfile):
        raise TypeError("primitive_profile must be ContactPrimitiveProfile.")
    return activation, envelope


class CompiledContactSearchEvidence(StrictModule):
    candidate_count: Array
    overflow_count: Array
    duplicate_route_count: Array
    finite: Array
    complete: Array
    plan_id: str = eqx.field(static=True)


class CompiledContactSearchResult(StrictModule):
    epoch: ContactCandidateEpoch
    evidence: CompiledContactSearchEvidence


class CompiledContactSearchPlan(StrictModule, NonTrainableState):
    """Fixed-shape device search over predeclared legal oriented route rows.

    Preparation enumerates every legal witness row of each route family once.
    Runtime evaluates full-path activation, then canonicalizes, counts and
    orders accepted rows on device into the shared :class:`ContactCandidateEpoch`
    schema. This is the deterministic compiled authority for the LBVH backend.
    """

    layouts: ContactRouteLayouts
    candidates: tuple[PreparedRouteCandidates, ...]
    feature_endpoints: Array
    activation_distance: float = eqx.field(static=True)
    envelope_radius: float = eqx.field(static=True)
    primitive_profile: ContactPrimitiveProfile = eqx.field(static=True)
    scene_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        scene: ContactSearchScene,
        /,
        *,
        edge_vertex_capacity: int,
        edge_edge_capacity: int,
        face_vertex_capacity: int,
        activation_distance: float,
        envelope_radius: float = 0.0,
        vertex_vertex_capacity: int = 0,
        vertex_plane_capacity: int = 0,
        edge_plane_capacity: int = 0,
        primitive_profile: ContactPrimitiveProfile = ContactPrimitiveProfile.MESH,
    ) -> None:
        _check_device_plan_inputs(scene, scene.scene_id)
        capacities = contact_search_capacities(
            vertex_vertex_capacity,
            edge_vertex_capacity,
            edge_edge_capacity,
            face_vertex_capacity,
            vertex_plane_capacity,
            edge_plane_capacity,
        )
        activation, envelope = _validated_search_scalars(
            activation_distance, envelope_radius, primitive_profile
        )
        layouts = ContactRouteLayouts(
            scene, capacities=capacities, profile=primitive_profile
        )
        families = {
            family[0]: family
            for family in contact_route_families(scene, primitive_profile)
        }
        table = scene_feature_endpoints(scene)
        excluded = scene_excluded_vertex_pairs(scene)
        candidates = tuple(
            prepare_route_candidates(scene, families[layout.kind], table, excluded)
            if layout.kind in families
            else empty_route_candidates(layout.kind)
            for layout in layouts.layouts
        )
        self.layouts = layouts
        self.candidates = candidates
        self.feature_endpoints = jnp.asarray(table, dtype=jnp.int32)
        self.activation_distance = activation
        self.envelope_radius = envelope
        self.primitive_profile = primitive_profile
        self.scene_id = scene.scene_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compiled-contact-search-plan",
                "scene": scene.scene_id,
                "layouts": layouts.layouts_id,
                "candidates": [
                    array_tree_fingerprint(
                        (
                            np.asarray(value.left_features),
                            np.asarray(value.right_features),
                            np.asarray(value.witness_slots),
                        )
                    )
                    for value in candidates
                ],
                "activation_distance": float(activation).hex(),
                "envelope_radius": float(envelope).hex(),
            }
        )
        self.epoch_id = contact_epoch_identity(self.plan_id, scene, layouts)

    def evaluate(
        self,
        scene: ContactSearchScene,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
        path_enclosure: ContactPathEnclosure | None = None,
        revision: ArrayLike | None = None,
    ) -> CompiledContactSearchResult:
        _check_device_plan_inputs(scene, self.scene_id)
        start, lower, upper = vertex_path_boxes(
            scene, positions, end_positions, path_enclosure
        )
        end = (
            start
            if end_positions is None
            else jnp.asarray(end_positions, dtype=start.dtype)
        )
        extents, normals, offsets, geometry_valid = scene_route_geometry(
            scene, start.dtype
        )
        admitted = geometry_valid & enclosure_admits(
            path_enclosure, start, end, lower, upper
        )
        feature_lower, feature_upper = feature_boxes(
            self.feature_endpoints, lower, upper
        )
        geometry = (extents, normals, offsets)
        catalogue = scene.feature_catalogue
        batches: list[ContactStencilBatch] = []
        duplicates = jnp.asarray(0, dtype=jnp.int32)
        addresses = jnp.asarray(0, dtype=jnp.int32)
        for layout, candidates in zip(
            self.layouts.layouts, self.candidates, strict=True
        ):
            mask = admitted & route_activation_mask(
                scene,
                candidates,
                feature_lower,
                feature_upper,
                extents,
                normals,
                offsets,
                self.activation_distance,
                self.envelope_radius,
            )
            packing = pack_contact_stencil_batch(
                layout,
                catalogue,
                candidates.endpoints,
                candidates.left_features,
                candidates.right_features,
                candidates.witness_slots,
                extents[candidates.left_features] + extents[candidates.right_features],
                mask,
            )
            batches.append(packing.batch)
            duplicates = duplicates + packing.duplicate_count
            addresses = addresses + packing.invalid_address_count
        overflow_count = sum(
            (batch.overflow_count for batch in batches), jnp.asarray(0, dtype=jnp.int32)
        )
        status = _epoch_status(
            admitted, jnp.asarray(False), overflow_count > 0, duplicates, addresses
        )
        epoch = _assemble_epoch(
            scene,
            self.plan_id,
            self.epoch_id,
            route_batch_tuple(batches),
            start,
            lower,
            upper,
            self.envelope_radius,
            path_enclosure,
            revision,
            geometry,
            status,
        )
        evidence = CompiledContactSearchEvidence(
            epoch.candidate_count,
            overflow_count,
            duplicates,
            admitted,
            epoch.complete,
            self.plan_id,
        )
        return CompiledContactSearchResult(epoch, evidence)

    def build(
        self,
        scene: ContactSearchScene,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
        path_enclosure: ContactPathEnclosure | None = None,
        revision: ArrayLike | None = None,
    ) -> ContactCandidateEpoch:
        return self.evaluate(
            scene,
            positions,
            end_positions=end_positions,
            path_enclosure=path_enclosure,
            revision=revision,
        ).epoch


class LBVHContactSearchEvidence(StrictModule):
    """LBVH backend evidence.

    ``traversal_visits`` and per-kind discovered counts are work actually
    discovered; when traversal is exhausted they are not the total required
    capacity.
    """

    candidate_count: Array
    node_count: Array
    tree_depth: Array
    traversal_visits: Array
    duplicate_code_count: Array
    stack_overflow: Array
    visit_overflow: Array
    output_overflow: Array
    duplicate_route_count: Array
    plane_pair_work: Array
    finite_bounds: Array
    complete: Array
    plan_id: str = eqx.field(static=True)


class LBVHContactSearchResult(StrictModule):
    epoch: ContactCandidateEpoch
    evidence: LBVHContactSearchEvidence


type _FamilyBuffer = tuple[Array, Array, Array, Array, Array]
type _TraversalState = tuple[Array, Array, Array, Array, Array, tuple[_FamilyBuffer, ...]]


class _TreeFamily(StrictModule, NonTrainableState):
    """One tree-traversed route family: feature classes and witness slots."""

    layout_index: int = eqx.field(static=True)
    kind: ContactStencilKind = eqx.field(static=True)
    left_class: int = eqx.field(static=True)
    right_class: int = eqx.field(static=True)
    witness_slot_count: int = eqx.field(static=True)


_FEATURE_CLASS = {
    CollisionFeatureKind.VERTEX: 0,
    CollisionFeatureKind.EDGE: 1,
    CollisionFeatureKind.FACE: 2,
}

_TREE_FEATURE_KINDS = {
    ContactStencilKind.VERTEX_VERTEX: (0, 0),
    ContactStencilKind.EDGE_VERTEX: (0, 1),
    ContactStencilKind.EDGE_EDGE: (1, 1),
    ContactStencilKind.FACE_VERTEX: (0, 2),
}


class LBVHContactSearchPlan(StrictModule, NonTrainableState):
    """Deterministic fixed-budget device broad phase over Morton-ordered AABBs.

    Finite primitive full-path boxes (inflated per axis by contact extent and
    envelope) are ordered by Morton code and assembled into a balanced binary
    hierarchy. A depth-first node-pair traversal applies the exact pair
    activation and legality at leaves and emits oriented witness rows into
    fixed buffers, which are then canonicalized on device into the shared
    epoch. Static planes use the bounded analytic plane route, never a box:
    planes are streamed one at a time against the prepared primitive set into
    the fixed output capacity, so no primitive x plane table is stored.
    Search selection is stop-gradient topology.
    """

    layouts: ContactRouteLayouts
    plane_left_sets: tuple[Array, Array]
    tree_left: Array
    tree_right: Array
    node_leaf_count: Array
    feature_endpoints: Array
    feature_class: Array
    vertex_admitted: Array
    feature_participant_ids: Array
    feature_static_mask: Array
    allowed_participant_pairs: Array
    excluded_vertex_pairs: Array
    tree_families: tuple[_TreeFamily, ...]
    unrestricted_pairs: bool = eqx.field(static=True)
    primitive_count: int = eqx.field(static=True)
    hierarchy_root: int = eqx.field(static=True)
    hierarchy_depth: int = eqx.field(static=True)
    activation_distance: float = eqx.field(static=True)
    envelope_radius: float = eqx.field(static=True)
    primitive_profile: ContactPrimitiveProfile = eqx.field(static=True)
    morton_bits: int = eqx.field(static=True)
    maximum_tree_depth: int = eqx.field(static=True)
    maximum_traversal_visits: int = eqx.field(static=True)
    plane_pair_work: int = eqx.field(static=True)
    scene_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        scene: ContactSearchScene,
        /,
        *,
        edge_vertex_capacity: int,
        edge_edge_capacity: int,
        face_vertex_capacity: int,
        activation_distance: float,
        maximum_traversal_visits: int,
        envelope_radius: float = 0.0,
        vertex_vertex_capacity: int = 0,
        vertex_plane_capacity: int = 0,
        edge_plane_capacity: int = 0,
        primitive_profile: ContactPrimitiveProfile = ContactPrimitiveProfile.MESH,
        morton_bits: int = 10,
        maximum_tree_depth: int = 64,
    ) -> None:
        _check_device_plan_inputs(scene, scene.scene_id)
        capacities = contact_search_capacities(
            vertex_vertex_capacity,
            edge_vertex_capacity,
            edge_edge_capacity,
            face_vertex_capacity,
            vertex_plane_capacity,
            edge_plane_capacity,
        )
        activation, envelope = _validated_search_scalars(
            activation_distance, envelope_radius, primitive_profile
        )
        bits = int(morton_bits)
        depth = int(maximum_tree_depth)
        visits = int(maximum_traversal_visits)
        if bits <= 0 or bits > 10:
            raise ValueError("morton_bits must lie in [1, 10] for uint32 codes.")
        if depth <= 0 or visits <= 0:
            raise ValueError("LBVH depth and traversal visit budgets must be positive.")
        if visits > np.iinfo(np.int32).max:
            raise ValueError(
                "LBVH traversal visit budget exceeds int32 evidence capacity."
            )
        layouts = ContactRouteLayouts(
            scene, capacities=capacities, profile=primitive_profile
        )
        families = {
            family[0]: family
            for family in contact_route_families(scene, primitive_profile)
        }
        # Exact host proof that discovered-row counters (int32 evidence) and
        # the streamed plane work counter cannot wrap: tree families discover
        # at most one pair per traversal visit, plane families at most every
        # primitive x plane row.
        bounds = contact_route_capacity_bounds(scene, primitive_profile)
        count_limit = int(np.iinfo(np.int32).max)
        total = 0
        for kind, bound in zip(ROUTE_KIND_ORDER, bounds, strict=True):
            slots = families[kind][3] if kind in families else 1
            discovered = bound if kind in _PLANE_KINDS else min(bound, slots * visits)
            total += discovered
            if discovered > count_limit or total > count_limit:
                raise ValueError(
                    f"LBVH {kind.name} discovery counts may exceed int32 evidence; "
                    "reduce the traversal budget or partition the scene."
                )
        plane_work = sum(
            families[kind][1].size * scene.plane_count
            for kind in _PLANE_KINDS
            if kind in families
        )
        work_dtype = np.dtype(jax.dtypes.canonicalize_dtype(jnp.int64))
        if plane_work > int(np.iinfo(work_dtype).max):
            raise ValueError("LBVH plane-route work exceeds the device integer range.")
        table = scene_feature_endpoints(scene)
        excluded = scene_excluded_vertex_pairs(scene)
        primitive_count = scene.feature_count - scene.plane_count
        kinds = np.asarray(scene.feature_catalogue.feature_kinds)[:primitive_count]
        feature_class = np.asarray(
            [_FEATURE_CLASS[CollisionFeatureKind(int(kind))] for kind in kinds],
            dtype=np.int32,
        )
        admitted = np.zeros((scene.vertex_count,), dtype=np.bool_)
        tree_families: list[_TreeFamily] = []
        for index, layout in enumerate(layouts.layouts):
            family = families.get(layout.kind)
            if family is None or layout.kind in _PLANE_KINDS:
                continue
            left_class, right_class = _TREE_FEATURE_KINDS[layout.kind]
            for feature_set, feature_class_ in (
                (family[1], left_class),
                (family[2], right_class),
            ):
                if feature_class_ == 0:
                    admitted[feature_set] = True
            tree_families.append(
                _TreeFamily(
                    index, layout.kind, left_class, right_class, family[3]
                )
            )
        left, right, leaf_count, root, hierarchy_depth = _balanced_morton_hierarchy(
            primitive_count
        )
        policy = scene.pair_policy
        self.layouts = layouts
        vertex_planes, edge_planes = (
            jnp.asarray(
                families[kind][1] if kind in families else np.zeros((0,)),
                dtype=jnp.int32,
            )
            for kind in _PLANE_KINDS
        )
        self.plane_left_sets = (vertex_planes, edge_planes)
        self.tree_left = jnp.asarray(left, dtype=jnp.int32)
        self.tree_right = jnp.asarray(right, dtype=jnp.int32)
        self.node_leaf_count = jnp.asarray(leaf_count, dtype=jnp.int32)
        self.feature_endpoints = jnp.asarray(table, dtype=jnp.int32)
        self.feature_class = jnp.asarray(feature_class, dtype=jnp.int32)
        self.vertex_admitted = jnp.asarray(admitted)
        self.feature_participant_ids = jnp.asarray(
            scene.feature_participant_ids, dtype=jnp.int64
        )
        self.feature_static_mask = jnp.asarray(scene.feature_static_mask, dtype=jnp.bool_)
        self.allowed_participant_pairs = jnp.asarray(
            policy.allowed_participant_pairs, dtype=jnp.int64
        ).reshape((-1, 2))
        self.excluded_vertex_pairs = jnp.asarray(excluded, dtype=jnp.int32).reshape(
            (-1, 2)
        )
        self.tree_families = tuple(tree_families)
        self.unrestricted_pairs = policy.unrestricted
        self.primitive_count = primitive_count
        self.hierarchy_root = root
        self.hierarchy_depth = hierarchy_depth
        self.activation_distance = activation
        self.envelope_radius = envelope
        self.primitive_profile = primitive_profile
        self.morton_bits = bits
        self.maximum_tree_depth = depth
        self.maximum_traversal_visits = visits
        self.plane_pair_work = plane_work
        self.scene_id = scene.scene_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lbvh-contact-search-plan",
                "scene": scene.scene_id,
                "layouts": layouts.layouts_id,
                "activation_distance": float(activation).hex(),
                "envelope_radius": float(envelope).hex(),
                "morton_bits": bits,
                "maximum_tree_depth": depth,
                "maximum_traversal_visits": visits,
            }
        )
        self.epoch_id = contact_epoch_identity(self.plan_id, scene, layouts)

    def _pair_legal(self, left: Array, right: Array, /) -> Array:
        nonstatic = ~(self.feature_static_mask[left] & self.feature_static_mask[right])
        participant_pair = jnp.sort(
            jnp.stack(
                (
                    self.feature_participant_ids[left],
                    self.feature_participant_ids[right],
                )
            )
        )
        participant_allowed = self.unrestricted_pairs | jnp.any(
            jnp.all(self.allowed_participant_pairs == participant_pair[None, :], axis=1)
        )
        left_vertices = self.feature_endpoints[left][:, None]
        right_vertices = self.feature_endpoints[right][None, :]
        present = (left_vertices >= 0) & (right_vertices >= 0)
        shared = jnp.any(present & (left_vertices == right_vertices))
        low = jnp.minimum(left_vertices, right_vertices)
        high = jnp.maximum(left_vertices, right_vertices)
        pairs = jnp.stack((low, high), axis=-1)
        excluded = jnp.any(
            present[..., None]
            & jnp.all(
                pairs[..., None, :] == self.excluded_vertex_pairs[None, None, :, :],
                axis=-1,
            )
        )
        return nonstatic & participant_allowed & ~shared & ~excluded

    def evaluate(
        self,
        scene: ContactSearchScene,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
        path_enclosure: ContactPathEnclosure | None = None,
        revision: ArrayLike | None = None,
    ) -> LBVHContactSearchResult:
        _check_device_plan_inputs(scene, self.scene_id)
        start, lower, upper = vertex_path_boxes(
            scene, positions, end_positions, path_enclosure
        )
        end = (
            start
            if end_positions is None
            else jnp.asarray(end_positions, dtype=start.dtype)
        )
        dtype = start.dtype
        extents, normals, offsets, geometry_valid = scene_route_geometry(scene, dtype)
        admitted = geometry_valid & enclosure_admits(
            path_enclosure, start, end, lower, upper
        )
        feature_lower, feature_upper = feature_boxes(
            self.feature_endpoints, lower, upper
        )
        geometry = (extents, normals, offsets)
        catalogue = scene.feature_catalogue
        ranks = catalogue.canonical_ranks
        count = self.primitive_count
        raw_lower = feature_lower[:count]
        raw_upper = feature_upper[:count]
        inflation = (extents[:count] + self.envelope_radius)[:, None]
        primitive_min = raw_lower - inflation
        primitive_max = raw_upper + inflation
        finite_bounds = admitted & jnp.all(jnp.isfinite(primitive_min)) & jnp.all(
            jnp.isfinite(primitive_max)
        )
        centroid = 0.5 * primitive_min + 0.5 * primitive_max
        global_min = jnp.min(primitive_min, axis=0)
        global_max = jnp.max(primitive_max, axis=0)
        extent = global_max - global_min
        safe_extent = jnp.where(jnp.isfinite(extent) & (extent > 0.0), extent, 1.0)
        maximum_code_coordinate = (1 << self.morton_bits) - 1
        normalized = (centroid - global_min) / safe_extent
        normalized = jnp.where(jnp.isfinite(normalized), normalized, 0.0)
        quantized = jnp.floor(
            jnp.clip(normalized, 0.0, 1.0) * maximum_code_coordinate
        ).astype(jnp.uint32)
        codes = morton_encode_integer(quantized, self.morton_bits)
        stable_id = jnp.arange(count, dtype=jnp.int32)
        order = jnp.lexsort((stable_id, self.feature_class, codes)).astype(jnp.int32)
        sorted_codes = codes[order]
        duplicate_codes = jnp.sum(sorted_codes[1:] == sorted_codes[:-1], dtype=jnp.int32)
        node_count = 2 * count - 1
        node_min = jnp.zeros((node_count, scene.ambient_dimension), dtype=dtype)
        node_max = jnp.zeros((node_count, scene.ambient_dimension), dtype=dtype)
        node_min = node_min.at[:count].set(primitive_min[order])
        node_max = node_max.at[:count].set(primitive_max[order])
        buffers = tuple(
            (
                jnp.full((self.layouts.layouts[family.layout_index].capacity, 4), -1,
                         dtype=jnp.int32),
                jnp.zeros((self.layouts.layouts[family.layout_index].capacity,),
                          dtype=jnp.int32),
                jnp.zeros((self.layouts.layouts[family.layout_index].capacity,),
                          dtype=jnp.int32),
                jnp.zeros((self.layouts.layouts[family.layout_index].capacity,),
                          dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            )
            for family in self.tree_families
        )
        exhausted_depth = self.hierarchy_depth > self.maximum_tree_depth
        stack_overflow = jnp.asarray(exhausted_depth)
        visit_overflow = jnp.asarray(False)
        traversal_visits = jnp.asarray(0, dtype=jnp.int32)
        if count > 1 and not exhausted_depth:

            def combine_bounds(
                index: Array, bounds: tuple[Array, Array]
            ) -> tuple[Array, Array]:
                low, high = bounds
                left_child = self.tree_left[index]
                right_child = self.tree_right[index]
                parent = count + index
                low = low.at[parent].set(jnp.minimum(low[left_child], low[right_child]))
                high = high.at[parent].set(
                    jnp.maximum(high[left_child], high[right_child])
                )
                return low, high

            node_min, node_max = jax.lax.fori_loop(
                0, count - 1, combine_bounds, (node_min, node_max)
            )
            buffers, traversal_visits, stack_overflow, visit_overflow = self._traverse(
                node_min,
                node_max,
                order,
                raw_lower,
                raw_upper,
                extents,
                ranks,
                buffers,
            )
        tree_batches: dict[int, ContactStencilBatch] = {}
        duplicates = jnp.asarray(0, dtype=jnp.int32)
        addresses = jnp.asarray(0, dtype=jnp.int32)
        for family, buffer in zip(self.tree_families, buffers, strict=True):
            layout = self.layouts.layouts[family.layout_index]
            endpoints, left, right, slots, discovered = buffer
            accepted = jnp.arange(layout.capacity, dtype=jnp.int32) < jnp.minimum(
                discovered, layout.capacity
            )
            packing = pack_contact_stencil_batch(
                layout,
                catalogue,
                endpoints,
                left,
                right,
                slots,
                extents[left] + extents[right],
                accepted,
                discovered_count=discovered,
            )
            tree_batches[family.layout_index] = packing.batch
            duplicates = duplicates + packing.duplicate_count
            addresses = addresses + packing.invalid_address_count
        plane_work = jnp.asarray(self.plane_pair_work, dtype=jnp.int64)
        for kind, left_set in zip(_PLANE_KINDS, self.plane_left_sets, strict=True):
            index = ROUTE_KIND_ORDER.index(kind)
            layout = self.layouts.layouts[index]
            endpoints, left, right, slots, discovered = self._stream_plane_rows(
                scene,
                layout,
                left_set,
                feature_lower,
                feature_upper,
                extents,
                normals,
                offsets,
                admitted,
            )
            accepted = jnp.arange(layout.capacity, dtype=jnp.int32) < jnp.minimum(
                discovered, layout.capacity
            )
            packing = pack_contact_stencil_batch(
                layout,
                catalogue,
                endpoints,
                left,
                right,
                slots,
                extents[left] + extents[right],
                accepted,
                discovered_count=discovered,
            )
            tree_batches[index] = packing.batch
            duplicates = duplicates + packing.duplicate_count
            addresses = addresses + packing.invalid_address_count
        batches = tuple(
            tree_batches[index]
            if index in tree_batches
            else pack_contact_stencil_batch(
                layout,
                catalogue,
                jnp.zeros((0, 4), dtype=jnp.int32),
                jnp.zeros((0,), dtype=jnp.int32),
                jnp.zeros((0,), dtype=jnp.int32),
                jnp.zeros((0,), dtype=jnp.int32),
                jnp.zeros((0,), dtype=dtype),
                jnp.zeros((0,), dtype=jnp.bool_),
            ).batch
            for index, layout in enumerate(self.layouts.layouts)
        )
        output_overflow = sum(
            (batch.overflow_count for batch in batches), jnp.asarray(0, dtype=jnp.int32)
        )
        status = _epoch_status(
            finite_bounds,
            stack_overflow | visit_overflow,
            output_overflow > 0,
            duplicates,
            addresses,
        )
        epoch = _assemble_epoch(
            scene,
            self.plan_id,
            self.epoch_id,
            route_batch_tuple(batches),
            start,
            lower,
            upper,
            self.envelope_radius,
            path_enclosure,
            revision,
            geometry,
            status,
        )
        evidence = LBVHContactSearchEvidence(
            epoch.candidate_count,
            jnp.asarray(node_count, dtype=jnp.int32),
            jnp.asarray(self.hierarchy_depth, dtype=jnp.int32),
            traversal_visits,
            duplicate_codes,
            stack_overflow,
            visit_overflow,
            output_overflow,
            duplicates,
            plane_work,
            finite_bounds,
            epoch.complete,
            self.plan_id,
        )
        return LBVHContactSearchResult(epoch, evidence)

    def build(
        self,
        scene: ContactSearchScene,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
        path_enclosure: ContactPathEnclosure | None = None,
        revision: ArrayLike | None = None,
    ) -> ContactCandidateEpoch:
        return self.evaluate(
            scene,
            positions,
            end_positions=end_positions,
            path_enclosure=path_enclosure,
            revision=revision,
        ).epoch

    def _stream_plane_rows(
        self,
        scene: ContactSearchScene,
        layout: ContactStencilLayout,
        left_set: Array,
        feature_lower: Array,
        feature_upper: Array,
        extents: Array,
        normals: Array,
        offsets: Array,
        admitted: Array,
        /,
    ) -> _FamilyBuffer:
        """Stream planes one at a time into fixed-capacity plane-route rows.

        Working storage is O(primitives + capacity); the total legality and
        activation work, primitives x planes, is reported as evidence.
        """
        capacity = layout.capacity
        slots = layout.witness_slot_count
        empty: _FamilyBuffer = (
            jnp.full((capacity, 4), -1, dtype=jnp.int32),
            jnp.zeros((capacity,), dtype=jnp.int32),
            jnp.zeros((capacity,), dtype=jnp.int32),
            jnp.zeros((capacity,), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        plane_count = scene.plane_count
        if left_set.shape[0] == 0 or plane_count == 0:
            return empty
        first_plane = scene.feature_count - plane_count
        left_lower = feature_lower[left_set]
        left_upper = feature_upper[left_set]
        left_extent = extents[left_set]
        left_participant = self.feature_participant_ids[left_set]
        left_static = self.feature_static_mask[left_set]
        slot_values = jnp.arange(slots, dtype=jnp.int32)

        def stream(plane: Array, buffer: _FamilyBuffer, /) -> _FamilyBuffer:
            endpoints, lefts, rights, witness, discovered = buffer
            feature = first_plane + plane
            radius = route_pair_radius(
                self.activation_distance,
                self.envelope_radius,
                left_extent,
                extents[feature],
                True,
            )
            near = (
                plane_box_clearance(left_lower, left_upper, normals[plane], offsets[plane])
                <= radius
            )
            pair = jnp.stack(
                (
                    jnp.minimum(left_participant, self.feature_participant_ids[feature]),
                    jnp.maximum(left_participant, self.feature_participant_ids[feature]),
                ),
                axis=-1,
            )
            allowed = self.unrestricted_pairs | jnp.any(
                jnp.all(
                    pair[:, None, :] == self.allowed_participant_pairs[None, :, :],
                    axis=-1,
                ),
                axis=-1,
            )
            accepted = (
                admitted
                & near
                & allowed
                & ~(left_static & self.feature_static_mask[feature])
            )
            order = jnp.cumsum(accepted.astype(jnp.int32)) - 1
            plane_feature = jnp.full(left_set.shape, feature, dtype=jnp.int32)
            rows = route_endpoints(
                layout.kind, self.feature_endpoints, left_set, plane_feature
            )
            for slot in range(slots):
                target = discovered + order * slots + slot
                target = jnp.where(accepted & (target < capacity), target, capacity)
                endpoints, lefts, rights, witness = _append_rows(
                    (endpoints, lefts, rights, witness),
                    target,
                    rows,
                    left_set,
                    plane_feature,
                    jnp.broadcast_to(slot_values[slot], left_set.shape),
                )
            discovered = discovered + slots * jnp.sum(accepted, dtype=jnp.int32)
            return endpoints, lefts, rights, witness, discovered

        return jax.lax.fori_loop(0, plane_count, stream, empty)

    def _record_leaf_pair(
        self,
        first: Array,
        second: Array,
        raw_lower: Array,
        raw_upper: Array,
        extents: Array,
        ranks: Array,
        buffers: tuple[_FamilyBuffer, ...],
        /,
    ) -> tuple[_FamilyBuffer, ...]:
        first_class = self.feature_class[first]
        second_class = self.feature_class[second]
        distinct = first != second
        updated: list[_FamilyBuffer] = []
        for family, buffer in zip(self.tree_families, buffers, strict=True):
            endpoints, lefts, rights, slots, discovered = buffer
            capacity = lefts.shape[0]
            if contact_stencil_symmetric(family.kind):
                match = (
                    (first_class == family.left_class)
                    & (second_class == family.left_class)
                    & distinct
                )
                first_lower = ranks[first] < ranks[second]
                left = jnp.where(first_lower, first, second)
                right = jnp.where(first_lower, second, first)
            else:
                forward = (first_class == family.left_class) & (
                    second_class == family.right_class
                )
                reverse = (first_class == family.right_class) & (
                    second_class == family.left_class
                )
                match = forward | reverse
                left = jnp.where(forward, first, second)
                right = jnp.where(forward, second, first)
            vertex_limit = self.vertex_admitted.shape[0] - 1
            if family.left_class == 0:
                match = match & self.vertex_admitted[jnp.clip(left, 0, vertex_limit)]
            if family.right_class == 0:
                match = match & self.vertex_admitted[jnp.clip(right, 0, vertex_limit)]
            radius = route_pair_radius(
                self.activation_distance,
                self.envelope_radius,
                extents[left],
                extents[right],
                False,
            )
            near = box_distance_squared(
                raw_lower[left], raw_upper[left], raw_lower[right], raw_upper[right]
            ) <= radius * radius
            accepted = match & near & self._pair_legal(left, right)
            row = route_endpoints(
                family.kind, self.feature_endpoints, left[None], right[None]
            )[0]
            for slot in range(family.witness_slot_count):
                target = jnp.where(
                    accepted & (discovered < capacity), discovered, capacity
                )
                endpoints, lefts, rights, slots = _append_rows(
                    (endpoints, lefts, rights, slots),
                    target,
                    row,
                    left,
                    right,
                    jnp.asarray(slot, dtype=jnp.int32),
                )
                discovered = discovered + accepted.astype(jnp.int32)
            updated.append((endpoints, lefts, rights, slots, discovered))
        return tuple(updated)

    def _traverse(
        self,
        node_min: Array,
        node_max: Array,
        order: Array,
        raw_lower: Array,
        raw_upper: Array,
        extents: Array,
        ranks: Array,
        buffers: tuple[_FamilyBuffer, ...],
        /,
    ) -> tuple[tuple[_FamilyBuffer, ...], Array, Array, Array]:
        count = self.primitive_count
        stack_capacity = 4 * self.hierarchy_depth
        stack_first = jnp.full((stack_capacity,), -1, dtype=jnp.int32)
        stack_second = jnp.full((stack_capacity,), -1, dtype=jnp.int32)
        stack_first = stack_first.at[0].set(self.hierarchy_root)
        stack_second = stack_second.at[0].set(self.hierarchy_root)
        initial: _TraversalState = (
            stack_first,
            stack_second,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            buffers,
        )

        def continue_traversal(state: _TraversalState) -> Array:
            _, _, stack_size, visits, overflow, _ = state
            return (
                (stack_size > 0)
                & (visits < self.maximum_traversal_visits)
                & ~overflow
            )

        def traverse_one(state: _TraversalState) -> _TraversalState:
            first_stack, second_stack, stack_size, visits, overflow, outputs = state
            stack_size = stack_size - 1
            first = first_stack[stack_size]
            second = second_stack[stack_size]
            delta = jnp.maximum(
                0.0,
                jnp.maximum(
                    node_min[first] - node_max[second], node_min[second] - node_max[first]
                ),
            )
            overlap = jnp.sum(delta * delta) <= self.activation_distance**2
            first_leaf = first < count
            second_leaf = second < count
            leaves = overlap & first_leaf & second_leaf
            outputs = jax.lax.cond(
                leaves,
                lambda values: self._record_leaf_pair(
                    order[jnp.clip(first, 0, count - 1)],
                    order[jnp.clip(second, 0, count - 1)],
                    raw_lower,
                    raw_upper,
                    extents,
                    ranks,
                    values,
                ),
                lambda values: values,
                outputs,
            )
            internal_first = jnp.clip(first - count, 0, count - 2)
            internal_second = jnp.clip(second - count, 0, count - 2)
            first_left = self.tree_left[internal_first]
            first_right = self.tree_right[internal_first]
            second_left = self.tree_left[internal_second]
            second_right = self.tree_right[internal_second]
            same = first == second
            split_first = (~first_leaf) & (
                second_leaf
                | (self.node_leaf_count[first] >= self.node_leaf_count[second])
            )
            new_first = jnp.where(
                same,
                jnp.stack((first_right, first_left, first_left)),
                jnp.where(
                    split_first,
                    jnp.stack((first_right, first_left, first_left)),
                    jnp.stack((first, first, first)),
                ),
            )
            new_second = jnp.where(
                same,
                jnp.stack((first_right, first_right, first_left)),
                jnp.where(
                    split_first,
                    jnp.stack((second, second, second)),
                    jnp.stack((second_right, second_left, second_left)),
                ),
            )
            pushes = jnp.where(
                overlap & ~(first_leaf & second_leaf), jnp.where(same, 3, 2), 0
            ).astype(jnp.int32)
            fits = stack_size + pushes <= stack_capacity
            slots = stack_size + jnp.arange(3, dtype=jnp.int32)
            target = jnp.where(
                (jnp.arange(3, dtype=jnp.int32) < pushes) & fits,
                slots,
                stack_capacity,
            )
            first_stack = first_stack.at[target].set(new_first, mode="drop")
            second_stack = second_stack.at[target].set(new_second, mode="drop")
            return (
                first_stack,
                second_stack,
                jnp.where(fits, stack_size + pushes, stack_size),
                visits + 1,
                overflow | ~fits,
                outputs,
            )

        _, _, remaining, visits, overflow, outputs = jax.lax.while_loop(
            continue_traversal, traverse_one, initial
        )
        return outputs, visits, overflow, (remaining > 0) & ~overflow


type ContactSearchPlan = (
    DenseContactSearchPlan
    | SweepAndPruneContactSearchPlan
    | CompiledContactSearchPlan
    | LBVHContactSearchPlan
)


type _RowBuffers = tuple[Array, Array, Array, Array]


def _append_rows(
    buffers: _RowBuffers,
    target: Array,
    endpoints: Array,
    left: Array,
    right: Array,
    slot: Array,
    /,
) -> _RowBuffers:
    """Scatter accepted rows to ``target`` (``capacity`` means dropped).

    Zero-capacity buffers are returned unchanged without indexing their empty
    axis; callers still count discovered rows, so overflow stays observable.
    """
    row_endpoints, lefts, rights, slots = buffers
    if lefts.shape[0] == 0:
        return buffers
    return (
        row_endpoints.at[target].set(endpoints, mode="drop"),
        lefts.at[target].set(left, mode="drop"),
        rights.at[target].set(right, mode="drop"),
        slots.at[target].set(slot, mode="drop"),
    )


def _balanced_morton_hierarchy(
    leaf_count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    count = int(leaf_count)
    if count <= 0:
        raise ValueError("LBVH construction requires at least one primitive.")
    left_children: list[int] = []
    right_children: list[int] = []
    node_leaf_count = [1] * count

    def build(first: int, last: int) -> tuple[int, int]:
        if last - first == 1:
            return first, 1
        middle = first + (last - first) // 2
        left, left_depth = build(first, middle)
        right, right_depth = build(middle, last)
        parent = count + len(left_children)
        left_children.append(left)
        right_children.append(right)
        node_leaf_count.append(node_leaf_count[left] + node_leaf_count[right])
        return parent, max(left_depth, right_depth) + 1

    root, depth = build(0, count)
    return (
        np.asarray(left_children, dtype=np.int32),
        np.asarray(right_children, dtype=np.int32),
        np.asarray(node_leaf_count, dtype=np.int32),
        root,
        depth,
    )


__all__ = [
    "CompiledContactSearchEvidence",
    "CompiledContactSearchPlan",
    "CompiledContactSearchResult",
    "ContactSearchPlan",
    "LBVHContactSearchEvidence",
    "LBVHContactSearchPlan",
    "LBVHContactSearchResult",
    "PreparedRouteCandidates",
    "prepare_route_candidates",
    "route_activation_mask",
]
