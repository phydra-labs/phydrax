#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
import time
from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._participant import ContactParticipantScene
from ._stencils import (
    contact_stencil_symmetric,
    ContactPrimitiveProfile,
    ContactRouteKeySchema,
    ContactStencilBatch,
    ContactStencilKind,
    ContactStencilLayout,
    prepare_contact_stencil_batch,
    scene_feature_endpoints,
)
from ._surface import PreparedCollisionScene
from ._trajectory import ContactPathEnclosure


type ContactSearchScene = PreparedCollisionScene | ContactParticipantScene

# Host route family: (kind, left scene features, right scene features, slots).
type ContactRouteFamily = tuple[ContactStencilKind, np.ndarray, np.ndarray, int]

ROUTE_KIND_ORDER = (
    ContactStencilKind.VERTEX_VERTEX,
    ContactStencilKind.EDGE_VERTEX,
    ContactStencilKind.EDGE_EDGE,
    ContactStencilKind.FACE_VERTEX,
    ContactStencilKind.VERTEX_PLANE,
    ContactStencilKind.EDGE_PLANE,
)


class ContactSearchStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    CANDIDATE_OVERFLOW = 2
    MEMORY_LIMIT = 3
    TIME_LIMIT = 4
    TRAVERSAL_EXHAUSTED = 5
    DUPLICATE_ROUTE = 6
    INVALID_ROUTE_ADDRESS = 7


class ContactSearchLimits(StrictModule, NonTrainableState):
    maximum_candidates: int | None = eqx.field(static=True)
    maximum_memory_bytes: int | None = eqx.field(static=True)
    maximum_time_seconds: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_candidates: int | None = None,
        maximum_memory_bytes: int | None = None,
        maximum_time_seconds: float | None = None,
    ) -> None:
        candidates = None if maximum_candidates is None else int(maximum_candidates)
        memory = None if maximum_memory_bytes is None else int(maximum_memory_bytes)
        seconds = None if maximum_time_seconds is None else float(maximum_time_seconds)
        if candidates is not None and candidates < 0:
            raise ValueError("maximum_candidates must be nonnegative or None.")
        if memory is not None and memory < 0:
            raise ValueError("maximum_memory_bytes must be nonnegative or None.")
        if seconds is not None and (not isfinite(seconds) or seconds < 0.0):
            raise ValueError("maximum_time_seconds must be finite/nonnegative or None.")
        self.maximum_candidates = candidates
        self.maximum_memory_bytes = memory
        self.maximum_time_seconds = seconds


class ContactRouteLayouts(StrictModule, NonTrainableState):
    """Prepared static layouts of every route kind of one search over one catalogue."""

    vertex_vertex: ContactStencilLayout
    edge_vertex: ContactStencilLayout
    edge_edge: ContactStencilLayout
    face_vertex: ContactStencilLayout
    vertex_plane: ContactStencilLayout
    edge_plane: ContactStencilLayout
    profile: ContactPrimitiveProfile = eqx.field(static=True)
    layouts_id: str = eqx.field(static=True)

    def __init__(
        self,
        scene: ContactSearchScene,
        /,
        *,
        capacities: tuple[int, int, int, int, int, int],
        profile: ContactPrimitiveProfile,
    ) -> None:
        if not isinstance(profile, ContactPrimitiveProfile):
            raise TypeError("profile must be ContactPrimitiveProfile.")
        schema = ContactRouteKeySchema(scene.feature_catalogue)
        capsule = profile == ContactPrimitiveProfile.SPHERE_CAPSULE
        layouts = tuple(
            ContactStencilLayout(
                schema,
                kind,
                capacity,
                witness_slot_count=(
                    2
                    if kind == ContactStencilKind.EDGE_PLANE
                    or (capsule and kind == ContactStencilKind.EDGE_EDGE)
                    else 1
                ),
            )
            for kind, capacity in zip(ROUTE_KIND_ORDER, capacities, strict=True)
        )
        (
            self.vertex_vertex,
            self.edge_vertex,
            self.edge_edge,
            self.face_vertex,
            self.vertex_plane,
            self.edge_plane,
        ) = layouts
        self.profile = profile
        self.layouts_id = canonical_fingerprint(
            {
                "kind": "contact-route-layouts",
                "profile": int(profile),
                "batches": [layout.batch_id for layout in layouts],
            }
        )

    @property
    def layouts(self) -> tuple[ContactStencilLayout, ...]:
        return (
            self.vertex_vertex,
            self.edge_vertex,
            self.edge_edge,
            self.face_vertex,
            self.vertex_plane,
            self.edge_plane,
        )

    @property
    def schema(self) -> ContactRouteKeySchema:
        return self.vertex_vertex.schema


class ContactCandidateEpoch(StrictModule, NonTrainableState):
    """Canonical fixed-capacity route epoch with fail-closed coverage evidence.

    Every backend (dense, sweep-and-prune, compiled, LBVH) returns this one
    schema. Rows are canonical oriented witnesses keyed over the scene's typed
    feature catalogue. Coverage: each vertex path box searched is
    ``[coverage_lower, coverage_upper]``; the epoch is complete for any motion
    in which every vertex stays within ``envelope_radius`` (Euclidean) of its
    searched box, since pair activation used ``2 * envelope_radius`` for two
    moving primitives. ``trajectory_id``/``revision``/``lower_time``/
    ``upper_time`` bind the coverage to the declared path enclosure. The
    numeric search geometry (per-feature contact extents and plane normals and
    offsets) is retained so ``binds(scene)`` refuses reuse after any radius,
    clearance or plane change; ``routing_id`` binds pair policy and static
    masks. Static identity is content-free; numeric revisions are dynamic.
    """

    vertex_vertex: ContactStencilBatch
    edge_vertex: ContactStencilBatch
    edge_edge: ContactStencilBatch
    face_vertex: ContactStencilBatch
    vertex_plane: ContactStencilBatch
    edge_plane: ContactStencilBatch
    reference_positions: Array
    coverage_lower: Array
    coverage_upper: Array
    envelope_radius: Array
    lower_time: Array
    upper_time: Array
    revision: Array
    contact_extents: Array
    plane_normals: Array
    plane_offsets: Array
    candidate_count: Array
    status: Array
    complete: Array
    search_id: str = eqx.field(static=True)
    catalogue_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    routing_id: str = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.complete & (self.status == int(ContactSearchStatus.SUCCESS))

    @property
    def batches(self) -> tuple[ContactStencilBatch, ...]:
        return (
            self.vertex_vertex,
            self.edge_vertex,
            self.edge_edge,
            self.face_vertex,
            self.vertex_plane,
            self.edge_plane,
        )

    @property
    def active_batches(self) -> tuple[ContactStencilBatch, ...]:
        return tuple(batch for batch in self.batches if batch.capacity)

    @property
    def overflow_count(self) -> Array:
        return sum(
            (batch.overflow_count for batch in self.batches),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def covers(self, lower: ArrayLike, upper: ArrayLike, /) -> Array:
        """Whether per-vertex path boxes lie in the searched box ⊕ envelope ball.

        Uses the exact farthest-corner distance of each query box from the
        searched box. Nonfinite or reversed queries are not covered.
        """
        lower_ = jnp.asarray(lower, dtype=self.coverage_lower.dtype)
        upper_ = jnp.asarray(upper, dtype=self.coverage_lower.dtype)
        if lower_.shape != self.coverage_lower.shape or upper_.shape != lower_.shape:
            raise ValueError("Coverage queries must have the epoch vertex box shape.")
        excess = jnp.maximum(
            0.0,
            jnp.maximum(self.coverage_lower - lower_, upper_ - self.coverage_upper),
        )
        squared = jnp.sum(excess * excess, axis=-1)
        radius = self.envelope_radius.astype(lower_.dtype)
        return (
            jnp.all(jnp.isfinite(lower_))
            & jnp.all(jnp.isfinite(upper_))
            & jnp.all(lower_ <= upper_)
            & jnp.all(squared <= radius * radius)
        )

    def binds(self, scene: ContactSearchScene, /) -> Array:
        """Whether this epoch was searched for ``scene``'s current geometry.

        Foreign catalogue, topology or routing policy raises (static refusal).
        The returned device predicate is false when the scene's contact
        extents or plane geometry differ numerically from those searched, or
        when the scene geometry is invalid (nonfinite, negative extents,
        non-unit plane normals).
        """
        if (
            scene.feature_catalogue.catalogue_id != self.catalogue_id
            or scene.scene_id != self.topology_id
            or scene.routing_id != self.routing_id
        ):
            raise ValueError(
                "Candidate epoch belongs to another scene, catalogue or routing policy."
            )
        extents, normals, offsets, valid = scene_route_geometry(
            scene, self.contact_extents.dtype
        )
        if (
            extents.shape != self.contact_extents.shape
            or normals.shape != self.plane_normals.shape
        ):
            raise ValueError("Candidate epoch geometry binding changed shape.")
        return (
            valid
            & jnp.all(extents == self.contact_extents)
            & jnp.all(normals == self.plane_normals)
            & jnp.all(offsets == self.plane_offsets)
        )


def scene_route_geometry(
    scene: ContactSearchScene, dtype: jnp.dtype, /
) -> tuple[Array, Array, Array, Array]:
    """Return ``(extents, plane normals, plane offsets, valid)`` used by search.

    ``valid`` requires finite nonnegative contact extents and finite unit plane
    normals and offsets; route activation assumes unit normals, so invalid
    geometry must refuse search rather than prune pairs with a scaled bound.
    """
    extents = scene.feature_contact_extent.astype(dtype)
    dimension = scene.ambient_dimension
    if isinstance(scene, ContactParticipantScene) and scene.plane_count:
        normals = scene.plane_normals.astype(dtype)
        offsets = scene.plane_offsets.astype(dtype)
    else:
        normals = jnp.zeros((0, dimension), dtype=dtype)
        offsets = jnp.zeros((0,), dtype=dtype)
    tolerance = 64.0 * jnp.finfo(dtype).eps
    unit = jnp.abs(jnp.sum(normals * normals, axis=-1) - 1.0) <= tolerance
    valid = (
        jnp.all(jnp.isfinite(extents))
        & jnp.all(extents >= 0.0)
        & jnp.all(jnp.isfinite(normals))
        & jnp.all(jnp.isfinite(offsets))
        & jnp.all(unit)
    )
    return extents, normals, offsets, valid


type RouteBatches = tuple[
    ContactStencilBatch,
    ContactStencilBatch,
    ContactStencilBatch,
    ContactStencilBatch,
    ContactStencilBatch,
    ContactStencilBatch,
]


def route_batch_tuple(values: Sequence[ContactStencilBatch], /) -> RouteBatches:
    """Return batches in ``ROUTE_KIND_ORDER`` as the fixed epoch tuple."""
    batches = tuple(values)
    if len(batches) != len(ROUTE_KIND_ORDER) or any(
        batch.kind != kind for batch, kind in zip(batches, ROUTE_KIND_ORDER, strict=True)
    ):
        raise ValueError("Epoch batches must follow the canonical route kind order.")
    vertex_vertex, edge_vertex, edge_edge, face_vertex, vertex_plane, edge_plane = (
        batches
    )
    return (
        vertex_vertex,
        edge_vertex,
        edge_edge,
        face_vertex,
        vertex_plane,
        edge_plane,
    )


def contact_epoch_identity(
    search_id: str,
    scene: ContactSearchScene,
    layouts: ContactRouteLayouts,
    /,
) -> str:
    """Content-free static identity of epochs of one search over one scene.

    The trajectory binding is the separate static ``trajectory_id`` field and
    the dynamic enclosure revision/interval; numeric content never enters it.
    """
    return canonical_fingerprint(
        {
            "kind": "contact-candidate-epoch",
            "search": search_id,
            "catalogue": scene.feature_catalogue.catalogue_id,
            "topology": scene.scene_id,
            "routing": scene.routing_id,
            "layouts": layouts.layouts_id,
        }
    )


class HostContactSearchEvidence(StrictModule):
    """Host-authority resource evidence, kept outside the scientific epoch."""

    elapsed_seconds: Array
    estimated_memory_bytes: Array
    plan_id: str = eqx.field(static=True)


class HostContactSearchResult(StrictModule):
    epoch: ContactCandidateEpoch
    evidence: HostContactSearchEvidence


def _isolated_vertices(scene: ContactSearchScene, /) -> np.ndarray:
    used = np.zeros((scene.vertex_count,), dtype=np.bool_)
    edges = np.asarray(scene.edges, dtype=np.int64).reshape((-1, 2))
    faces = np.asarray(scene.faces, dtype=np.int64).reshape((-1, 3))
    used[edges.reshape(-1)] = True
    used[faces.reshape(-1)] = True
    return np.flatnonzero(~used).astype(np.int64)


def contact_route_families(
    scene: ContactSearchScene,
    profile: ContactPrimitiveProfile,
    /,
) -> tuple[ContactRouteFamily, ...]:
    """Host declaration of which feature sets each route kind may pair.

    Scene feature indices: vertices ``[0, V)``, edges ``[V, V+E)``, faces
    ``[V+E, V+E+F)``, planes after them. Families with an empty side are
    omitted. The same declaration drives host, compiled, and LBVH search.
    """
    vertices = scene.vertex_count
    edge_count = scene.edge_count
    face_count = scene.face_count
    plane_count = scene.plane_count
    all_vertices = np.arange(vertices, dtype=np.int64)
    edges = np.arange(vertices, vertices + edge_count, dtype=np.int64)
    faces = np.arange(
        vertices + edge_count, vertices + edge_count + face_count, dtype=np.int64
    )
    planes = np.arange(
        vertices + edge_count + face_count,
        vertices + edge_count + face_count + plane_count,
        dtype=np.int64,
    )
    families: list[ContactRouteFamily] = []
    if profile == ContactPrimitiveProfile.SPHERE_CAPSULE:
        if face_count:
            raise ValueError("The sphere-capsule primitive profile is face-free.")
        spheres = _isolated_vertices(scene)
        families.extend(
            (
                (ContactStencilKind.VERTEX_VERTEX, spheres, spheres, 1),
                (ContactStencilKind.EDGE_VERTEX, spheres, edges, 1),
                (ContactStencilKind.EDGE_EDGE, edges, edges, 2),
                (ContactStencilKind.VERTEX_PLANE, spheres, planes, 1),
                (ContactStencilKind.EDGE_PLANE, edges, planes, 2),
            )
        )
    elif profile == ContactPrimitiveProfile.MESH:
        if scene.ambient_dimension == 2 or face_count == 0:
            families.append((ContactStencilKind.EDGE_VERTEX, all_vertices, edges, 1))
        else:
            families.append((ContactStencilKind.FACE_VERTEX, all_vertices, faces, 1))
            families.append((ContactStencilKind.EDGE_EDGE, edges, edges, 1))
        families.append((ContactStencilKind.VERTEX_PLANE, all_vertices, planes, 1))
    else:
        raise TypeError("Unsupported contact primitive profile.")
    return tuple(
        family for family in families if family[1].size and family[2].size
    )


def contact_route_capacity_bounds(
    scene: ContactSearchScene,
    profile: ContactPrimitiveProfile,
    /,
) -> tuple[int, int, int, int, int, int]:
    """Exact upper bounds on witness rows per route kind, without pair tables.

    Returned in ``ContactRouteLayouts`` order (vertex-vertex, edge-vertex,
    edge-edge, face-vertex, vertex-plane, edge-plane). Each bound counts every
    oriented feature pair a family could ever emit (unordered pairs for
    symmetric kinds) times its witness slots, before geometry or legality, in
    O(features) host work. Capacities at these bounds can never overflow; any
    smaller fixed budget is a declared resource choice whose overflow fails
    closed at runtime.
    """
    bounds = dict.fromkeys(ROUTE_KIND_ORDER, 0)
    for kind, left, right, slots in contact_route_families(scene, profile):
        if contact_stencil_symmetric(kind):
            pairs = left.size * (left.size - 1) // 2
        else:
            pairs = left.size * right.size
        bounds[kind] = pairs * slots
    return (
        bounds[ContactStencilKind.VERTEX_VERTEX],
        bounds[ContactStencilKind.EDGE_VERTEX],
        bounds[ContactStencilKind.EDGE_EDGE],
        bounds[ContactStencilKind.FACE_VERTEX],
        bounds[ContactStencilKind.VERTEX_PLANE],
        bounds[ContactStencilKind.EDGE_PLANE],
    )


def scene_excluded_vertex_pairs(scene: ContactSearchScene, /) -> np.ndarray:
    """Sorted unique scene-vertex exclusion pairs ``(pairs, 2)``."""
    if isinstance(scene, PreparedCollisionScene):
        plans = tuple(surface.plan for surface in scene.surfaces)
    elif isinstance(scene, ContactParticipantScene):
        plans = tuple(participant.surface_plan for participant in scene.participants)
    else:
        raise TypeError("scene must be a prepared collision or participant scene.")
    pairs: list[np.ndarray] = [np.empty((0, 2), dtype=np.int64)]
    for surface_index, plan in enumerate(plans):
        offset = scene.vertex_offsets[surface_index]
        local = np.asarray(plan.pair_policy.excluded_vertex_pairs, dtype=np.int64)
        pairs.append(np.sort(local.reshape((-1, 2)) + offset, axis=1))
    return np.unique(np.concatenate(pairs, axis=0), axis=0).reshape((-1, 2))


def legal_route_pairs(
    scene: ContactSearchScene,
    kind: ContactStencilKind,
    left: np.ndarray,
    right: np.ndarray,
    /,
    *,
    endpoints: np.ndarray,
    excluded: np.ndarray,
) -> np.ndarray:
    """Vectorized host legality of oriented candidate feature pairs.

    Refuses shared vertices, static-static pairs, disallowed participant pairs,
    excluded vertex pairs and (for symmetric kinds) non-canonical orientation.
    """
    if left.size == 0:
        return np.zeros((0,), dtype=np.bool_)
    catalogue = scene.feature_catalogue
    static = np.asarray(scene.feature_static_mask, dtype=np.bool_)
    participants = np.asarray(scene.feature_participant_ids, dtype=np.int64)
    legal = ~(static[left] & static[right])
    if contact_stencil_symmetric(kind):
        ranks = np.asarray(catalogue.canonical_ranks, dtype=np.int64)
        legal &= ranks[left] < ranks[right]
    policy = scene.pair_policy
    if not policy.unrestricted:
        allowed = np.asarray(policy.allowed_participant_pairs, dtype=np.int64)
        labels, inverse = np.unique(
            np.concatenate((participants, allowed.reshape(-1))), return_inverse=True
        )
        dense = inverse[: participants.size]
        allowed_dense = inverse[participants.size :].reshape((-1, 2))
        base = labels.size
        allowed_codes = allowed_dense[:, 0] * base + allowed_dense[:, 1]
        low = np.minimum(dense[left], dense[right])
        high = np.maximum(dense[left], dense[right])
        legal &= np.isin(low * base + high, allowed_codes)
    left_vertices = endpoints[left]
    right_vertices = endpoints[right]
    present = (left_vertices[:, :, None] >= 0) & (right_vertices[:, None, :] >= 0)
    shared = present & (left_vertices[:, :, None] == right_vertices[:, None, :])
    legal &= ~np.any(shared, axis=(1, 2))
    if excluded.size:
        count = scene.vertex_count
        if count > 3_037_000_499:
            raise ValueError("Scene vertex count is too large for exclusion codes.")
        excluded_codes = excluded[:, 0] * count + excluded[:, 1]
        low = np.minimum(left_vertices[:, :, None], right_vertices[:, None, :])
        high = np.maximum(left_vertices[:, :, None], right_vertices[:, None, :])
        hit = present & np.isin(low * count + high, excluded_codes)
        legal &= ~np.any(hit, axis=(1, 2))
    return legal


def vertex_path_boxes(
    scene: ContactSearchScene,
    positions: ArrayLike,
    end_positions: ArrayLike | None,
    path_enclosure: ContactPathEnclosure | None,
    /,
) -> tuple[Array, Array, Array]:
    """Return ``(start, lower, upper)`` per-vertex path boxes.

    With a path enclosure its boxes are used (and must contain both endpoint
    configurations); otherwise the endpoint hull of linear motion is used.
    """
    start = jnp.asarray(positions)
    if not jnp.issubdtype(start.dtype, jnp.floating):
        start = start.astype(jnp.float64)
    end = start if end_positions is None else jnp.asarray(end_positions, dtype=start.dtype)
    expected = (scene.vertex_count, scene.ambient_dimension)
    if start.shape != expected or end.shape != expected:
        raise ValueError(f"Contact search positions must have shape {expected}.")
    lower = jnp.minimum(start, end)
    upper = jnp.maximum(start, end)
    if path_enclosure is not None:
        if not isinstance(path_enclosure, ContactPathEnclosure):
            raise TypeError("path_enclosure must be ContactPathEnclosure or None.")
        if path_enclosure.lower.shape != expected:
            raise ValueError("Path enclosure boxes must match the scene vertices.")
        lower = jnp.asarray(path_enclosure.lower, dtype=start.dtype)
        upper = jnp.asarray(path_enclosure.upper, dtype=start.dtype)
    return start, lower, upper


def enclosure_admits(
    path_enclosure: ContactPathEnclosure | None,
    start: Array,
    end: Array,
    lower: Array,
    upper: Array,
    /,
) -> Array:
    """Device predicate: finite boxes containing endpoints, enclosure successful."""
    admitted = (
        jnp.all(jnp.isfinite(start))
        & jnp.all(jnp.isfinite(end))
        & jnp.all(jnp.isfinite(lower))
        & jnp.all(jnp.isfinite(upper))
        & jnp.all(lower <= start)
        & jnp.all(start <= upper)
        & jnp.all(lower <= end)
        & jnp.all(end <= upper)
    )
    if path_enclosure is not None:
        admitted = admitted & path_enclosure.successful & path_enclosure.finite
    return admitted


def enclosure_binding(
    path_enclosure: ContactPathEnclosure | None,
    revision: ArrayLike | None,
    dtype: jnp.dtype,
    /,
) -> tuple[Array, Array, Array, str]:
    """Return ``(lower_time, upper_time, revision, trajectory_id)``."""
    if path_enclosure is None:
        value = jnp.asarray(0 if revision is None else revision, dtype=jnp.int64)
        zero = jnp.asarray(0.0, dtype=dtype)
        return zero, zero, value, ""
    if revision is not None:
        raise ValueError("revision is supplied by the path enclosure.")
    return (
        jnp.asarray(path_enclosure.lower_time, dtype=dtype),
        jnp.asarray(path_enclosure.upper_time, dtype=dtype),
        jnp.asarray(path_enclosure.revision, dtype=jnp.int64),
        path_enclosure.trajectory_id,
    )


def contact_search_capacities(
    vertex_vertex_capacity: int,
    edge_vertex_capacity: int,
    edge_edge_capacity: int,
    face_vertex_capacity: int,
    vertex_plane_capacity: int,
    edge_plane_capacity: int,
    /,
) -> tuple[int, int, int, int, int, int]:
    capacities = (
        int(vertex_vertex_capacity),
        int(edge_vertex_capacity),
        int(edge_edge_capacity),
        int(face_vertex_capacity),
        int(vertex_plane_capacity),
        int(edge_plane_capacity),
    )
    # Zero capacity is a legal declaration for scenes with no possible routes;
    # any discovered candidate under it overflows and fails closed at runtime.
    if any(value < 0 for value in capacities):
        raise ValueError("Contact search capacities must be nonnegative.")
    return capacities


class _AbstractContactSearchPlan(StrictModule, NonTrainableState):
    vertex_vertex_capacity: int = eqx.field(static=True)
    edge_vertex_capacity: int = eqx.field(static=True)
    edge_edge_capacity: int = eqx.field(static=True)
    face_vertex_capacity: int = eqx.field(static=True)
    vertex_plane_capacity: int = eqx.field(static=True)
    edge_plane_capacity: int = eqx.field(static=True)
    activation_distance: float = eqx.field(static=True)
    envelope_radius: float = eqx.field(static=True)
    primitive_profile: ContactPrimitiveProfile = eqx.field(static=True)
    limits: ContactSearchLimits
    method: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def _search_kind(self) -> str:
        raise NotImplementedError

    def __init__(
        self,
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
        limits: ContactSearchLimits | None = None,
        method: str,
    ) -> None:
        capacities = contact_search_capacities(
            vertex_vertex_capacity,
            edge_vertex_capacity,
            edge_edge_capacity,
            face_vertex_capacity,
            vertex_plane_capacity,
            edge_plane_capacity,
        )
        activation = float(activation_distance)
        envelope = float(envelope_radius)
        if not isfinite(activation) or activation <= 0.0:
            raise ValueError("activation_distance must be finite and positive.")
        if not isfinite(envelope) or envelope < 0.0:
            raise ValueError("envelope_radius must be finite and nonnegative.")
        if not isinstance(primitive_profile, ContactPrimitiveProfile):
            raise TypeError("primitive_profile must be ContactPrimitiveProfile.")
        limits_ = ContactSearchLimits() if limits is None else limits
        if not isinstance(limits_, ContactSearchLimits):
            raise TypeError("limits must be ContactSearchLimits or None.")
        method_ = str(method)
        if method_ not in ("dense", "sweep-and-prune"):
            raise ValueError("Unknown contact search method.")
        (
            self.vertex_vertex_capacity,
            self.edge_vertex_capacity,
            self.edge_edge_capacity,
            self.face_vertex_capacity,
            self.vertex_plane_capacity,
            self.edge_plane_capacity,
        ) = capacities
        self.activation_distance = activation
        self.envelope_radius = envelope
        self.primitive_profile = primitive_profile
        self.limits = limits_
        self.method = method_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "contact-search-plan",
                "method": method_,
                "capacities": capacities,
                "activation_distance": activation.hex(),
                "envelope_radius": envelope.hex(),
                "primitive_profile": int(primitive_profile),
                "limits": (
                    limits_.maximum_candidates,
                    limits_.maximum_memory_bytes,
                    limits_.maximum_time_seconds,
                ),
            }
        )

    @property
    def capacities(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.vertex_vertex_capacity,
            self.edge_vertex_capacity,
            self.edge_edge_capacity,
            self.face_vertex_capacity,
            self.vertex_plane_capacity,
            self.edge_plane_capacity,
        )

    def layouts(self, scene: ContactSearchScene, /) -> ContactRouteLayouts:
        return ContactRouteLayouts(
            scene, capacities=self.capacities, profile=self.primitive_profile
        )

    def search(
        self,
        scene: ContactSearchScene,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
        path_enclosure: ContactPathEnclosure | None = None,
        revision: ArrayLike | None = None,
    ) -> HostContactSearchResult:
        """Host search returning the epoch and separate host resource evidence."""
        return _build_epoch(
            self, scene, positions, end_positions, path_enclosure, revision
        )

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
        return self.search(
            scene,
            positions,
            end_positions=end_positions,
            path_enclosure=path_enclosure,
            revision=revision,
        ).epoch


class DenseContactSearchPlan(_AbstractContactSearchPlan):
    """Exhaustive correctness-authority contact candidate search."""

    def _search_kind(self) -> str:
        return "dense"

    def __init__(
        self,
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
        limits: ContactSearchLimits | None = None,
    ) -> None:
        super().__init__(
            edge_vertex_capacity=edge_vertex_capacity,
            edge_edge_capacity=edge_edge_capacity,
            face_vertex_capacity=face_vertex_capacity,
            activation_distance=activation_distance,
            envelope_radius=envelope_radius,
            vertex_vertex_capacity=vertex_vertex_capacity,
            vertex_plane_capacity=vertex_plane_capacity,
            edge_plane_capacity=edge_plane_capacity,
            primitive_profile=primitive_profile,
            limits=limits,
            method="dense",
        )


class SweepAndPruneContactSearchPlan(_AbstractContactSearchPlan):
    """Deterministic exhaustive host sweep-and-prune over primitive AABBs."""

    def _search_kind(self) -> str:
        return "sweep-and-prune"

    def __init__(
        self,
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
        limits: ContactSearchLimits | None = None,
    ) -> None:
        super().__init__(
            edge_vertex_capacity=edge_vertex_capacity,
            edge_edge_capacity=edge_edge_capacity,
            face_vertex_capacity=face_vertex_capacity,
            activation_distance=activation_distance,
            envelope_radius=envelope_radius,
            vertex_vertex_capacity=vertex_vertex_capacity,
            vertex_plane_capacity=vertex_plane_capacity,
            edge_plane_capacity=edge_plane_capacity,
            primitive_profile=primitive_profile,
            limits=limits,
            method="sweep-and-prune",
        )


def feature_boxes(
    endpoints: np.ndarray | Array,
    vertex_lower: Array,
    vertex_upper: Array,
    /,
) -> tuple[Array, Array]:
    """Primitive boxes as hulls of their vertex path boxes (planes: empty rows).

    Every point of a primitive is a convex combination of its vertices, so the
    hull of vertex boxes contains the primitive throughout the path.
    """
    table = jnp.asarray(endpoints, dtype=jnp.int32)
    present = table >= 0
    safe = jnp.clip(table, 0, vertex_lower.shape[0] - 1)
    lower = jnp.min(
        jnp.where(present[..., None], vertex_lower[safe], jnp.inf), axis=1
    )
    upper = jnp.max(
        jnp.where(present[..., None], vertex_upper[safe], -jnp.inf), axis=1
    )
    return lower, upper


def box_distance_squared(
    first_lower: Array,
    first_upper: Array,
    second_lower: Array,
    second_upper: Array,
    /,
) -> Array:
    delta = jnp.maximum(
        0.0,
        jnp.maximum(first_lower - second_upper, second_lower - first_upper),
    )
    return jnp.sum(delta * delta, axis=-1)


def plane_box_clearance(
    lower: Array,
    upper: Array,
    normal: Array,
    offset: Array,
    /,
) -> Array:
    """Exact minimum of ``n·x - d`` over a box (support function bound)."""
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    return (
        jnp.sum(center * normal, axis=-1)
        - jnp.sum(jnp.abs(normal) * half, axis=-1)
        - offset
    )


def route_pair_radius(
    activation_distance: float,
    envelope_radius: float,
    left_extent: Array,
    right_extent: Array,
    plane: bool,
    /,
) -> Array:
    """Pair activation radius: activation, both contact extents, and envelope.

    Finite pairs add ``2 * envelope`` because both primitives may move by the
    per-object envelope; plane pairs add one envelope (planes are static).
    """
    moving = envelope_radius if plane else 2.0 * envelope_radius
    return activation_distance + moving + left_extent + right_extent


def _bipartite_pairs(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
    radius: float,
    method: str,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if left_min.shape[0] == 0 or right_min.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty
    if method == "dense":
        delta = np.maximum(
            0.0,
            np.maximum(
                left_min[:, None, :] - right_max[None, :, :],
                right_min[None, :, :] - left_max[:, None, :],
            ),
        )
        hit = np.sum(delta * delta, axis=-1) <= radius * radius
        left, right = np.nonzero(hit)
        return left.astype(np.int64), right.astype(np.int64)
    all_min = np.minimum(left_min.min(axis=0), right_min.min(axis=0))
    all_max = np.maximum(left_max.max(axis=0), right_max.max(axis=0))
    axis = int(np.argmax(all_max - all_min))
    order = np.argsort(right_min[:, axis], kind="stable")
    sorted_lower = right_min[order, axis]
    lefts: list[int] = []
    rights: list[int] = []
    for left in range(left_min.shape[0]):
        stop = int(
            np.searchsorted(sorted_lower, left_max[left, axis] + radius, side="right")
        )
        candidates = order[:stop]
        candidates = candidates[
            right_max[candidates, axis] + radius >= left_min[left, axis]
        ]
        delta = np.maximum(
            0.0,
            np.maximum(
                left_min[left] - right_max[candidates],
                right_min[candidates] - left_max[left],
            ),
        )
        hit = candidates[np.sum(delta * delta, axis=-1) <= radius * radius]
        lefts.extend([left] * hit.size)
        rights.extend(int(value) for value in hit)
    return np.asarray(lefts, dtype=np.int64), np.asarray(rights, dtype=np.int64)


def _host_route_records(
    plan: _AbstractContactSearchPlan,
    scene: ContactSearchScene,
    family: ContactRouteFamily,
    lower: np.ndarray,
    upper: np.ndarray,
    endpoints: np.ndarray,
    excluded: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return oriented ``(left, right, slots, minimum_separation)`` records."""
    kind, left_set, right_set, slots = family
    extent = np.asarray(scene.feature_contact_extent, dtype=np.float64)
    plane = kind in (ContactStencilKind.VERTEX_PLANE, ContactStencilKind.EDGE_PLANE)
    if plane:
        left = np.repeat(left_set, right_set.size)
        right = np.tile(right_set, left_set.size)
        first_plane = scene.feature_count - scene.plane_count
        _, plane_normals, plane_offsets, _ = scene_route_geometry(
            scene, jnp.dtype(jnp.float64)
        )
        normals = np.asarray(plane_normals)
        offsets = np.asarray(plane_offsets)
        clearance = np.asarray(
            plane_box_clearance(
                jnp.asarray(lower[left]),
                jnp.asarray(upper[left]),
                jnp.asarray(normals[right - first_plane]),
                jnp.asarray(offsets[right - first_plane]),
            )
        )
        radius = np.asarray(
            route_pair_radius(
                plan.activation_distance,
                plan.envelope_radius,
                jnp.asarray(extent[left]),
                jnp.asarray(extent[right]),
                True,
            )
        )
        near = clearance <= radius
    else:
        broad = (
            plan.activation_distance
            + 2.0 * plan.envelope_radius
            + 2.0 * float(np.max(extent, initial=0.0))
        )
        local_left, local_right = _bipartite_pairs(
            lower[left_set],
            upper[left_set],
            lower[right_set],
            upper[right_set],
            broad,
            plan.method,
        )
        first = left_set[local_left]
        second = right_set[local_right]
        if contact_stencil_symmetric(kind):
            ranks = np.asarray(scene.feature_catalogue.canonical_ranks, dtype=np.int64)
            swap = ranks[first] > ranks[second]
            first, second = np.where(swap, second, first), np.where(swap, first, second)
            unique = np.unique(np.stack((first, second), axis=1), axis=0)
            first, second = unique[:, 0], unique[:, 1]
        left, right = first, second
        distance = np.asarray(
            box_distance_squared(
                jnp.asarray(lower[left]),
                jnp.asarray(upper[left]),
                jnp.asarray(lower[right]),
                jnp.asarray(upper[right]),
            )
        )
        radius = np.asarray(
            route_pair_radius(
                plan.activation_distance,
                plan.envelope_radius,
                jnp.asarray(extent[left]),
                jnp.asarray(extent[right]),
                False,
            )
        )
        near = distance <= radius * radius
    left, right = left[near], right[near]
    legal = legal_route_pairs(
        scene, kind, left, right, endpoints=endpoints, excluded=excluded
    )
    left, right = left[legal], right[legal]
    left = np.repeat(left, slots)
    right = np.repeat(right, slots)
    slot = np.tile(np.arange(slots, dtype=np.int64), left.size // max(slots, 1))
    separation = extent[left] + extent[right]
    return left, right, slot, separation


def _limit_status(
    plan: _AbstractContactSearchPlan,
    count: int,
    elapsed: float,
    /,
) -> ContactSearchStatus:
    limits = plan.limits
    if limits.maximum_candidates is not None and count > limits.maximum_candidates:
        return ContactSearchStatus.CANDIDATE_OVERFLOW
    if (
        limits.maximum_memory_bytes is not None
        and count * _ROUTE_BYTES > limits.maximum_memory_bytes
    ):
        return ContactSearchStatus.MEMORY_LIMIT
    if limits.maximum_time_seconds is not None and elapsed > limits.maximum_time_seconds:
        return ContactSearchStatus.TIME_LIMIT
    return ContactSearchStatus.SUCCESS


# Host estimate of one route row: endpoints, IDs, indices, key, slot, scalars.
_ROUTE_BYTES = 96


def _build_epoch(
    plan: _AbstractContactSearchPlan,
    scene: ContactSearchScene,
    positions: ArrayLike,
    end_positions: ArrayLike | None,
    path_enclosure: ContactPathEnclosure | None,
    revision: ArrayLike | None,
    /,
) -> HostContactSearchResult:
    if not isinstance(scene, (PreparedCollisionScene, ContactParticipantScene)):
        raise TypeError(
            "scene must be PreparedCollisionScene or ContactParticipantScene."
        )
    start_time = time.perf_counter()
    start, lower, upper = vertex_path_boxes(
        scene, positions, end_positions, path_enclosure
    )
    end = start if end_positions is None else jnp.asarray(end_positions, dtype=start.dtype)
    if not bool(enclosure_admits(path_enclosure, start, end, lower, upper)):
        raise ValueError(
            "Contact search requires finite positions inside a successful path "
            "enclosure."
        )
    extents, normals, offsets, geometry_valid = scene_route_geometry(
        scene, start.dtype
    )
    if not bool(geometry_valid):
        raise ValueError(
            "Contact search requires finite nonnegative contact extents and finite "
            "unit plane normals."
        )
    lower_time, upper_time, revision_, trajectory_id = enclosure_binding(
        path_enclosure, revision, start.dtype
    )
    layouts = plan.layouts(scene)
    endpoints = scene_feature_endpoints(scene)
    excluded = scene_excluded_vertex_pairs(scene)
    vertex_lower = np.asarray(lower)
    vertex_upper = np.asarray(upper)
    primitive_lower, primitive_upper = feature_boxes(
        endpoints, jnp.asarray(vertex_lower), jnp.asarray(vertex_upper)
    )
    primitive_lower = np.asarray(primitive_lower)
    primitive_upper = np.asarray(primitive_upper)
    families = {
        family[0]: family
        for family in contact_route_families(scene, plan.primitive_profile)
    }
    batches: list[ContactStencilBatch] = []
    candidate_count = 0
    overflow = False
    for layout in layouts.layouts:
        family = families.get(layout.kind)
        if family is None:
            left = right = slots = np.zeros((0,), dtype=np.int64)
            separation = np.zeros((0,), dtype=np.float64)
        else:
            left, right, slots, separation = _host_route_records(
                plan,
                scene,
                family,
                primitive_lower,
                primitive_upper,
                endpoints,
                excluded,
            )
        count = left.size
        candidate_count += count
        capacity = layout.capacity
        overflow_count = max(count - capacity, 0)
        overflow |= overflow_count > 0
        schema = layout.schema
        ranks = np.asarray(scene.feature_catalogue.canonical_ranks, dtype=np.int64)
        keys = np.asarray(
            schema.encode(int(layout.kind), ranks[left], ranks[right], slots)
        )
        order = np.argsort(keys, kind="stable")[:capacity]
        kept = order.size
        padded = capacity - kept

        def pad(values: np.ndarray, fill: int | float, /) -> np.ndarray:
            return np.concatenate(
                (values[order], np.full((padded,), fill, dtype=values.dtype))
            )

        batches.append(
            prepare_contact_stencil_batch(
                scene,
                layout,
                pad(left, 0),
                pad(right, 0),
                witness_slots=pad(slots, 0),
                minimum_separation=pad(separation.astype(np.float64), 0.0).astype(
                    start.dtype
                ),
                valid=np.arange(capacity) < (kept if overflow_count == 0 else 0),
                actual_count=count,
                overflow_count=overflow_count,
                dtype=start.dtype,
            )
        )
    if candidate_count > int(np.iinfo(np.int32).max):
        raise ValueError("Contact candidate count exceeds int32 epoch evidence.")
    elapsed = time.perf_counter() - start_time
    status = (
        ContactSearchStatus.CANDIDATE_OVERFLOW
        if overflow
        else _limit_status(plan, candidate_count, elapsed)
    )
    complete = status == ContactSearchStatus.SUCCESS
    vertex_vertex, edge_vertex, edge_edge, face_vertex, vertex_plane, edge_plane = (
        route_batch_tuple(batches)
    )
    epoch = ContactCandidateEpoch(
        vertex_vertex=vertex_vertex,
        edge_vertex=edge_vertex,
        edge_edge=edge_edge,
        face_vertex=face_vertex,
        vertex_plane=vertex_plane,
        edge_plane=edge_plane,
        reference_positions=start,
        coverage_lower=lower,
        coverage_upper=upper,
        envelope_radius=jnp.asarray(plan.envelope_radius, dtype=start.dtype),
        lower_time=lower_time,
        upper_time=upper_time,
        revision=revision_,
        contact_extents=extents,
        plane_normals=normals,
        plane_offsets=offsets,
        candidate_count=jnp.asarray(candidate_count, dtype=jnp.int32),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        complete=jnp.asarray(complete),
        search_id=plan.plan_id,
        catalogue_id=scene.feature_catalogue.catalogue_id,
        topology_id=scene.scene_id,
        routing_id=scene.routing_id,
        trajectory_id=trajectory_id,
        epoch_id=contact_epoch_identity(plan.plan_id, scene, layouts),
    )
    evidence = HostContactSearchEvidence(
        jnp.asarray(elapsed),
        jnp.asarray(candidate_count * _ROUTE_BYTES, dtype=jnp.int64),
        plan.plan_id,
    )
    return HostContactSearchResult(epoch, evidence)


__all__ = [
    "ContactCandidateEpoch",
    "ContactRouteLayouts",
    "ContactSearchLimits",
    "ContactSearchStatus",
    "DenseContactSearchPlan",
    "HostContactSearchEvidence",
    "HostContactSearchResult",
    "SweepAndPruneContactSearchPlan",
    "contact_epoch_identity",
    "scene_route_geometry",
    "contact_route_capacity_bounds",
    "contact_route_families",
    "route_batch_tuple",
    "RouteBatches",
]
