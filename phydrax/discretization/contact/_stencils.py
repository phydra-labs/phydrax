#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._distance import (
    ContactDistanceEvaluation,
    ContactWitnessSlot,
    edge_edge_distance,
    edge_edge_mollifier,
    edge_edge_mollifier_threshold,
    point_edge_distance,
    point_plane_distance,
    point_point_distance,
    point_triangle_distance,
    segment_segment_support_distance,
)
from ._participant import ContactParticipantScene
from ._surface import (
    CollisionFeatureKind,
    ContactFeatureCatalogue,
    PreparedCollisionScene,
)


type ContactStencilScene = PreparedCollisionScene | ContactParticipantScene


class ContactStencilKind(IntEnum):
    """Canonical route kinds and their oriented ``(left, right)`` features.

    ``VERTEX_VERTEX``: two vertices (sphere-sphere), lower canonical rank left.
    ``EDGE_VERTEX``: vertex left, edge right (sphere-capsule, point-segment).
    ``EDGE_EDGE``: two edges, lower canonical rank left.
    ``FACE_VERTEX``: vertex left, face right.
    ``VERTEX_PLANE``: vertex left, static analytic plane right.
    ``EDGE_PLANE``: edge left, static analytic plane right; witness slot ``k``
    is edge endpoint ``k``.
    """

    VERTEX_VERTEX = 0
    EDGE_VERTEX = 1
    EDGE_EDGE = 2
    FACE_VERTEX = 3
    VERTEX_PLANE = 4
    EDGE_PLANE = 5


class ContactPrimitiveProfile(IntEnum):
    """Declared primitive interpretation of a scene for route emission.

    ``MESH``: polygonal surfaces. Two-dimensional or face-free scenes emit
    edge-vertex routes; three-dimensional surfaces emit face-vertex and
    edge-edge routes; vertices of any surface may meet static planes.
    ``SPHERE_CAPSULE``: a face-free scene whose isolated vertices are sphere
    centres and whose edges are capsule axes (radius = physical radius).
    Sphere rows use only isolated vertices, so capsule endpoints never
    duplicate capsule rows; capsule-capsule routes carry two support slots and
    capsule-plane routes one slot per capsule endpoint.
    """

    MESH = 0
    SPHERE_CAPSULE = 1



_ARITY = {
    ContactStencilKind.VERTEX_VERTEX: 2,
    ContactStencilKind.EDGE_VERTEX: 3,
    ContactStencilKind.EDGE_EDGE: 4,
    ContactStencilKind.FACE_VERTEX: 4,
    ContactStencilKind.VERTEX_PLANE: 1,
    ContactStencilKind.EDGE_PLANE: 2,
}

_FEATURE_KINDS = {
    ContactStencilKind.VERTEX_VERTEX: (
        CollisionFeatureKind.VERTEX,
        CollisionFeatureKind.VERTEX,
    ),
    ContactStencilKind.EDGE_VERTEX: (
        CollisionFeatureKind.VERTEX,
        CollisionFeatureKind.EDGE,
    ),
    ContactStencilKind.EDGE_EDGE: (CollisionFeatureKind.EDGE, CollisionFeatureKind.EDGE),
    ContactStencilKind.FACE_VERTEX: (
        CollisionFeatureKind.VERTEX,
        CollisionFeatureKind.FACE,
    ),
    ContactStencilKind.VERTEX_PLANE: (
        CollisionFeatureKind.VERTEX,
        CollisionFeatureKind.ANALYTIC,
    ),
    ContactStencilKind.EDGE_PLANE: (
        CollisionFeatureKind.EDGE,
        CollisionFeatureKind.ANALYTIC,
    ),
}


def contact_stencil_arity(kind: ContactStencilKind, /) -> int:
    """Number of scene vertex endpoints carried by one route row."""
    return _ARITY[kind]


def contact_stencil_feature_kinds(
    kind: ContactStencilKind, /
) -> tuple[CollisionFeatureKind, CollisionFeatureKind]:
    """Oriented ``(left, right)`` feature kinds of a route kind."""
    return _FEATURE_KINDS[kind]


def contact_stencil_symmetric(kind: ContactStencilKind, /) -> bool:
    """Whether left/right share a feature kind and are ordered by canonical rank."""
    left, right = _FEATURE_KINDS[kind]
    return left == right


def _admitted_witness_slots(kind: ContactStencilKind, /) -> tuple[int, ...]:
    if kind == ContactStencilKind.EDGE_PLANE:
        return (len(ContactWitnessSlot),)
    if kind == ContactStencilKind.EDGE_EDGE:
        return (1, len(ContactWitnessSlot))
    return (1,)


class ContactRouteKeySchema(StrictModule, NonTrainableState):
    """Bounded injective scalar route keys over one typed feature catalogue.

    ``key = ((kind * F + left_rank) * F + right_rank) * W + slot`` with ``F``
    the catalogue size, kinds from :class:`ContactStencilKind` and ``W`` from
    :class:`ContactWitnessSlot`. Preparation proves with exact host integers
    that every intermediate and final value is strictly below the maximum of
    the canonical device integer type, so the maximum is reserved as the empty
    sentinel. Keys are meaningful only together with ``catalogue_id``.
    """

    feature_count: int = eqx.field(static=True)
    kind_count: int = eqx.field(static=True)
    witness_slot_count: int = eqx.field(static=True)
    key_dtype: str = eqx.field(static=True)
    maximum_key: int = eqx.field(static=True)
    catalogue_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(self, catalogue: ContactFeatureCatalogue, /) -> None:
        if not isinstance(catalogue, ContactFeatureCatalogue):
            raise TypeError("catalogue must be ContactFeatureCatalogue.")
        features = catalogue.feature_count
        kinds = len(ContactStencilKind)
        slots = len(ContactWitnessSlot)
        dtype = np.dtype(jax.dtypes.canonicalize_dtype(jnp.int64))
        # Monotone in every digit: the final bound dominates every intermediate.
        maximum = ((kinds - 1) * features + (features - 1)) * features
        maximum = (maximum + (features - 1)) * slots + (slots - 1)
        if maximum >= int(np.iinfo(dtype).max):
            raise ValueError(
                "The feature catalogue is too large for injective route keys in "
                f"{dtype.name}; enable 64-bit JAX integers or partition the scene."
            )
        self.feature_count = features
        self.kind_count = kinds
        self.witness_slot_count = slots
        self.key_dtype = dtype.name
        self.maximum_key = maximum
        self.catalogue_id = catalogue.catalogue_id
        self.schema_id = canonical_fingerprint(
            {
                "kind": "contact-route-key-schema",
                "catalogue": catalogue.catalogue_id,
                "route_kinds": kinds,
                "witness_slots": slots,
                "dtype": dtype.name,
            }
        )

    @property
    def sentinel(self) -> int:
        """Key value reserved for empty slots; strictly above every route key."""
        return int(np.iinfo(np.dtype(self.key_dtype)).max)

    def encode(
        self,
        kind: ArrayLike,
        left_rank: ArrayLike,
        right_rank: ArrayLike,
        slot: ArrayLike,
        /,
    ) -> Array:
        dtype = jnp.dtype(self.key_dtype)
        features = jnp.asarray(self.feature_count, dtype=dtype)
        slots = jnp.asarray(self.witness_slot_count, dtype=dtype)
        value = jnp.asarray(kind, dtype=dtype) * features + jnp.asarray(
            left_rank, dtype=dtype
        )
        value = value * features + jnp.asarray(right_rank, dtype=dtype)
        return value * slots + jnp.asarray(slot, dtype=dtype)

    def decode(self, keys: ArrayLike, /) -> tuple[Array, Array, Array, Array]:
        """Return ``(kind, left_rank, right_rank, slot)`` of encoded keys."""
        dtype = jnp.dtype(self.key_dtype)
        value = jnp.asarray(keys, dtype=dtype)
        features = jnp.asarray(self.feature_count, dtype=dtype)
        slots = jnp.asarray(self.witness_slot_count, dtype=dtype)
        slot = value % slots
        value = value // slots
        right = value % features
        value = value // features
        left = value % features
        kind = value // features
        return (
            kind.astype(jnp.int32),
            left.astype(jnp.int32),
            right.astype(jnp.int32),
            slot.astype(jnp.int32),
        )


class ContactStencilLayout(StrictModule, NonTrainableState):
    """Static prepared layout of one homogeneous route batch."""

    schema: ContactRouteKeySchema
    kind: ContactStencilKind = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    witness_slot_count: int = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: ContactRouteKeySchema,
        kind: ContactStencilKind,
        capacity: int,
        /,
        *,
        witness_slot_count: int = 1,
    ) -> None:
        if not isinstance(schema, ContactRouteKeySchema):
            raise TypeError("schema must be ContactRouteKeySchema.")
        if not isinstance(kind, ContactStencilKind):
            raise TypeError("kind must be ContactStencilKind.")
        count = int(capacity)
        slots = int(witness_slot_count)
        if count < 0:
            raise ValueError("Contact stencil capacity must be nonnegative.")
        if slots not in _admitted_witness_slots(kind):
            raise ValueError(
                f"{kind.name} routes admit witness slot counts "
                f"{_admitted_witness_slots(kind)}."
            )
        self.schema = schema
        self.kind = kind
        self.capacity = count
        self.witness_slot_count = slots
        self.batch_id = canonical_fingerprint(
            {
                "kind": "contact-stencil-layout",
                "schema": schema.schema_id,
                "stencil_kind": int(kind),
                "capacity": count,
                "witness_slots": slots,
            }
        )

    @property
    def catalogue_id(self) -> str:
        return self.schema.catalogue_id


class ContactStencilBatch(StrictModule, NonTrainableState):
    """One fixed-capacity homogeneous batch of canonical contact routes.

    Rows are ordered by ascending route key; valid rows occupy a prefix. Each
    row is one witness of one oriented primitive pair: ``left``/``right``
    feature indices address the scene feature arrays, ``*_feature_ids`` are the
    original scientific IDs, and ``witness_slots`` names the stable witness of
    that pair (also folded into ``route_keys``). Overflowing batches keep their
    discovered diagnostic rows but mark every row invalid; truncation is never
    a successful smaller contact set. Static identity is content-free.
    """

    vertex_indices: Array
    left_feature_ids: Array
    right_feature_ids: Array
    left_feature_indices: Array
    right_feature_indices: Array
    route_keys: Array
    witness_slots: Array
    weights: Array
    minimum_separation: Array
    valid: Array
    actual_count: Array
    overflow_count: Array
    overflow: Array
    kind: ContactStencilKind = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    witness_slot_count: int = eqx.field(static=True)
    catalogue_id: str = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)


def scene_feature_endpoints(scene: ContactStencilScene, /) -> np.ndarray:
    """Host ``(features, 3)`` vertex table per scene feature; ``-1`` pads."""
    rows = np.full((scene.feature_count, 3), -1, dtype=np.int64)
    vertices = scene.vertex_count
    edges = np.asarray(scene.edges, dtype=np.int64).reshape((-1, 2))
    faces = np.asarray(scene.faces, dtype=np.int64).reshape((-1, 3))
    rows[:vertices, 0] = np.arange(vertices, dtype=np.int64)
    rows[vertices : vertices + edges.shape[0], :2] = edges
    start = vertices + edges.shape[0]
    rows[start : start + faces.shape[0], :] = faces
    return rows


def route_endpoints(
    kind: ContactStencilKind,
    endpoints: Array,
    left: Array,
    right: Array,
    /,
) -> Array:
    """Oriented ``(rows, 4)`` vertex endpoints of routes from the feature table.

    Shared by host preparation and device packing: left-feature endpoints come
    first, then right-feature endpoints, padded with ``-1``. ``endpoints`` is
    the ``(features, 3)`` table of :func:`scene_feature_endpoints`.
    """
    left_rows = endpoints[left]
    right_rows = endpoints[right]
    pad = jnp.full(left_rows.shape[:-1] + (1,), -1, dtype=endpoints.dtype)
    if kind == ContactStencilKind.VERTEX_VERTEX:
        parts = (left_rows[..., :1], right_rows[..., :1], pad, pad)
    elif kind == ContactStencilKind.EDGE_VERTEX:
        parts = (left_rows[..., :1], right_rows[..., :2], pad)
    elif kind == ContactStencilKind.EDGE_EDGE:
        parts = (left_rows[..., :2], right_rows[..., :2])
    elif kind == ContactStencilKind.FACE_VERTEX:
        parts = (left_rows[..., :1], right_rows)
    elif kind == ContactStencilKind.VERTEX_PLANE:
        parts = (left_rows[..., :1], pad, pad, pad)
    elif kind == ContactStencilKind.EDGE_PLANE:
        parts = (left_rows[..., :2], pad, pad)
    else:
        raise TypeError("Unsupported contact stencil kind.")
    return jnp.concatenate(parts, axis=-1)


def prepare_contact_stencil_batch(
    scene: ContactStencilScene,
    layout: ContactStencilLayout,
    left_features: ArrayLike,
    right_features: ArrayLike,
    /,
    *,
    witness_slots: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    minimum_separation: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    actual_count: ArrayLike | None = None,
    overflow_count: ArrayLike = 0,
    dtype: DTypeLike | None = None,
) -> ContactStencilBatch:
    """Validate a host-authored route batch and return its canonical form.

    This is the preparation boundary for host search authorities and explicit
    fixtures. Rows are given by oriented scene feature indices; endpoints and
    original IDs are derived from the scene topology and catalogue, keys use
    the shared :class:`ContactRouteKeySchema`, and valid rows are reordered by
    ascending key. Orientation, witness-slot ranges and key uniqueness are
    checked exactly. Device search uses :func:`pack_contact_stencil_batch`.
    """
    if not isinstance(scene, (PreparedCollisionScene, ContactParticipantScene)):
        raise TypeError(
            "scene must be PreparedCollisionScene or ContactParticipantScene."
        )
    if not isinstance(layout, ContactStencilLayout):
        raise TypeError("layout must be ContactStencilLayout.")
    catalogue = scene.feature_catalogue
    if layout.catalogue_id != catalogue.catalogue_id:
        raise ValueError("Stencil layout belongs to another feature catalogue.")
    capacity = layout.capacity
    kind = layout.kind
    left = np.asarray(left_features)
    right = np.asarray(right_features)
    if left.shape != (capacity,) or right.shape != (capacity,):
        raise ValueError("Route feature indices must have capacity shape.")
    if not np.issubdtype(left.dtype, np.integer) or not np.issubdtype(
        right.dtype, np.integer
    ):
        raise TypeError("Route feature indices must contain integers.")
    left = left.astype(np.int64, copy=False)
    right = right.astype(np.int64, copy=False)
    active = (
        np.ones((capacity,), dtype=np.bool_)
        if valid is None
        else np.asarray(valid, dtype=np.bool_)
    )
    if active.shape != (capacity,):
        raise ValueError("Contact stencil valid mask must have capacity shape.")
    slots = (
        np.zeros((capacity,), dtype=np.int64)
        if witness_slots is None
        else np.asarray(witness_slots)
    )
    if slots.shape != (capacity,) or not np.issubdtype(slots.dtype, np.integer):
        raise TypeError("witness_slots must be one integer capacity vector.")
    slots = slots.astype(np.int64, copy=False)
    count = scene.feature_count
    if np.any(active & ((left < 0) | (left >= count) | (right < 0) | (right >= count))):
        raise ValueError("Active route feature indices are outside the scene.")
    if np.any(active & ((slots < 0) | (slots >= layout.witness_slot_count))):
        raise ValueError("Active witness slots exceed the layout slot count.")
    safe_left = np.where(active, left, 0)
    safe_right = np.where(active, right, 0)
    kinds = np.asarray(catalogue.feature_kinds, dtype=np.int64)
    ranks = np.asarray(catalogue.canonical_ranks, dtype=np.int64)
    expected_left, expected_right = _FEATURE_KINDS[kind]
    if np.any(
        active
        & (
            (kinds[safe_left] != int(expected_left))
            | (kinds[safe_right] != int(expected_right))
        )
    ):
        raise ValueError(
            f"{kind.name} rows must be oriented as "
            f"({expected_left.name}, {expected_right.name})."
        )
    if contact_stencil_symmetric(kind) and np.any(
        active & (ranks[safe_left] >= ranks[safe_right])
    ):
        raise ValueError(
            f"{kind.name} rows must place the lower canonical rank on the left."
        )
    expected = np.asarray(
        route_endpoints(
            kind,
            jnp.asarray(scene_feature_endpoints(scene), dtype=jnp.int32),
            jnp.asarray(safe_left, dtype=jnp.int32),
            jnp.asarray(safe_right, dtype=jnp.int32),
        )
    )
    indices = np.where(active[:, None], expected, -1).astype(np.int32)
    if np.any(active[:, None] & (indices[:, : _ARITY[kind]] < 0)):
        raise ValueError("Active route features do not define complete endpoints.")
    distinct = ~np.eye(4, dtype=np.bool_)[None]
    repeated = (
        (indices[:, :, None] == indices[:, None, :])
        & distinct
        & (indices[:, :, None] >= 0)
    )
    if np.any(active[:, None, None] & repeated):
        raise ValueError("Active route endpoints must be distinct vertices.")
    floating = np.dtype(np.float64 if dtype is None else dtype)
    weight = (
        np.ones((capacity,), dtype=floating)
        if weights is None
        else np.asarray(weights, dtype=floating)
    )
    separation = (
        np.zeros((capacity,), dtype=floating)
        if minimum_separation is None
        else np.asarray(minimum_separation, dtype=floating)
    )
    if weight.shape != (capacity,) or separation.shape != (capacity,):
        raise ValueError(
            "Contact weights and minimum separation must have capacity shape."
        )
    if (
        np.any(~np.isfinite(weight))
        or np.any(~np.isfinite(separation))
        or np.any(separation < 0.0)
    ):
        raise ValueError(
            "Contact weights/separations must be finite and separations nonnegative."
        )
    true_count = (
        int(np.count_nonzero(active))
        if actual_count is None
        else int(np.asarray(actual_count))
    )
    overflow_count_ = int(np.asarray(overflow_count))
    if true_count < 0 or overflow_count_ < 0 or true_count > capacity + overflow_count_:
        raise ValueError("Contact stencil counts are inconsistent.")
    schema = layout.schema
    keys = np.asarray(
        schema.encode(int(kind), ranks[safe_left], ranks[safe_right], slots)
    ).astype(np.int64)
    active_keys = keys[active]
    if np.unique(active_keys).size != active_keys.size:
        raise ValueError("Active contact route keys must be unique.")
    sentinel = schema.sentinel
    order = np.argsort(np.where(active, keys, sentinel), kind="stable")
    feature_ids = np.asarray(catalogue.feature_ids, dtype=np.int64)
    ordered_active = active[order]
    key_dtype = np.dtype(schema.key_dtype)
    return ContactStencilBatch(
        jnp.asarray(indices[order], dtype=jnp.int32),
        jnp.asarray(np.where(ordered_active, feature_ids[safe_left[order]], 0),
                    dtype=jnp.int64),
        jnp.asarray(np.where(ordered_active, feature_ids[safe_right[order]], 0),
                    dtype=jnp.int64),
        jnp.asarray(np.where(ordered_active, left[order], -1), dtype=jnp.int32),
        jnp.asarray(np.where(ordered_active, right[order], -1), dtype=jnp.int32),
        jnp.asarray(np.where(ordered_active, keys[order], sentinel).astype(key_dtype)),
        jnp.asarray(np.where(ordered_active, slots[order], 0), dtype=jnp.int32),
        jnp.asarray(weight[order]),
        jnp.asarray(separation[order]),
        # Parity with device packing: an overflowing batch is globally
        # incomplete, so no row is usable even if the caller marked it valid.
        jnp.asarray(ordered_active & (overflow_count_ == 0)),
        jnp.asarray(true_count, dtype=jnp.int32),
        jnp.asarray(overflow_count_, dtype=jnp.int32),
        jnp.asarray(overflow_count_ > 0),
        kind,
        capacity,
        layout.witness_slot_count,
        layout.catalogue_id,
        layout.batch_id,
    )


def empty_contact_stencil_batch(
    layout: ContactStencilLayout,
    /,
    *,
    dtype: DTypeLike = jnp.float64,
) -> ContactStencilBatch:
    """Return an empty, complete batch with the given layout (device-safe)."""
    if not isinstance(layout, ContactStencilLayout):
        raise TypeError("layout must be ContactStencilLayout.")
    capacity = layout.capacity
    return ContactStencilBatch(
        jnp.full((capacity, 4), -1, dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=jnp.int64),
        jnp.zeros((capacity,), dtype=jnp.int64),
        jnp.full((capacity,), -1, dtype=jnp.int32),
        jnp.full((capacity,), -1, dtype=jnp.int32),
        jnp.full((capacity,), layout.schema.sentinel, dtype=layout.schema.key_dtype),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.ones((capacity,), dtype=dtype),
        jnp.zeros((capacity,), dtype=dtype),
        jnp.zeros((capacity,), dtype=jnp.bool_),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        layout.kind,
        capacity,
        layout.witness_slot_count,
        layout.catalogue_id,
        layout.batch_id,
    )


class ContactStencilPacking(StrictModule):
    """Device packing result: canonical batch plus identity failure evidence.

    ``duplicate_count`` counts adjacent equal keys among packed rows and
    ``invalid_address_count`` counts accepted rows whose claimed scene feature
    index is outside the catalogue; both make the epoch unsuccessful.
    """

    batch: ContactStencilBatch
    duplicate_count: Array
    invalid_address_count: Array


def pack_contact_stencil_batch(
    layout: ContactStencilLayout,
    catalogue: ContactFeatureCatalogue,
    vertex_indices: Array,
    left_features: Array,
    right_features: Array,
    witness_slots: Array,
    minimum_separation: Array,
    accepted: Array,
    /,
    *,
    discovered_count: Array | None = None,
) -> ContactStencilPacking:
    """Canonicalize fixed-shape device candidates into a route batch.

    Inputs are ``P`` candidate rows already oriented by the caller (endpoints
    from scene topology, left/right scene feature indices, witness slots) with
    an acceptance mask. Keys are encoded with the prepared schema, accepted
    rows are ordered by key, and the first ``capacity`` are kept. The count is
    ``discovered_count`` when the caller's buffer itself may have saturated.
    Any overflow invalidates every row. Adjacent equal keys among accepted rows
    are reported as duplicate identities, which callers treat as failure.
    No host conversion, fingerprint, or NumPy operation runs here.
    """
    if catalogue.catalogue_id != layout.catalogue_id:
        raise ValueError("Stencil layout belongs to another feature catalogue.")
    capacity = layout.capacity
    schema = layout.schema
    dtype = minimum_separation.dtype
    accepted = jnp.asarray(accepted, dtype=jnp.bool_)
    count = (
        jnp.sum(accepted, dtype=jnp.int32)
        if discovered_count is None
        else jnp.asarray(discovered_count, dtype=jnp.int32)
    )
    overflow_count = jnp.maximum(count - capacity, 0)
    if accepted.shape[0] == 0 or capacity == 0:
        empty = empty_contact_stencil_batch(layout, dtype=dtype)
        batch = eqx.tree_at(
            lambda value: (value.actual_count, value.overflow_count, value.overflow),
            empty,
            (count, overflow_count, overflow_count > 0),
        )
        return ContactStencilPacking(
            batch, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)
        )
    left_rank, left_resolved = catalogue.ranks(left_features, valid=accepted)
    right_rank, right_resolved = catalogue.ranks(right_features, valid=accepted)
    resolved = left_resolved & right_resolved
    invalid_address_count = jnp.sum(accepted & ~resolved, dtype=jnp.int32)
    accepted = accepted & resolved
    keys = schema.encode(int(layout.kind), left_rank, right_rank, witness_slots)
    sentinel = jnp.asarray(schema.sentinel, dtype=keys.dtype)
    sort_keys = jnp.where(accepted, keys, sentinel)
    order = jnp.argsort(sort_keys, stable=True).astype(jnp.int32)
    candidates = order.shape[0]
    if candidates < capacity:
        order = jnp.concatenate(
            (order, jnp.zeros((capacity - candidates,), dtype=jnp.int32))
        )
    order = order[:capacity]
    slot = jnp.arange(capacity, dtype=jnp.int32)
    present = (slot < jnp.minimum(count, min(candidates, capacity))) & accepted[order]
    valid = present & (overflow_count == 0)
    ordered_keys = sort_keys[order]
    adjacent = (ordered_keys[1:] == ordered_keys[:-1]) & present[1:] & present[:-1]
    duplicate_count = jnp.sum(adjacent, dtype=jnp.int32)
    left = left_features[order].astype(jnp.int32)
    right = right_features[order].astype(jnp.int32)
    limit = catalogue.feature_count - 1
    batch = ContactStencilBatch(
        jnp.where(present[:, None], vertex_indices[order], -1).astype(jnp.int32),
        jnp.where(present, catalogue.feature_ids[jnp.clip(left, 0, limit)], 0),
        jnp.where(present, catalogue.feature_ids[jnp.clip(right, 0, limit)], 0),
        jnp.where(present, left, -1),
        jnp.where(present, right, -1),
        jnp.where(present, ordered_keys, sentinel),
        jnp.where(present, witness_slots[order], 0).astype(jnp.int32),
        jnp.ones((capacity,), dtype=dtype),
        jnp.where(present, minimum_separation[order], 0.0).astype(dtype),
        valid,
        count,
        overflow_count,
        overflow_count > 0,
        layout.kind,
        capacity,
        layout.witness_slot_count,
        layout.catalogue_id,
        layout.batch_id,
    )
    return ContactStencilPacking(batch, duplicate_count, invalid_address_count)


class ContactStencilEvaluation(StrictModule):
    """Narrow-phase evaluation of one route batch.

    ``active`` marks rows whose witness exists (an inactive ``SECOND`` support
    slot is not a failure). ``failed`` marks valid active rows whose geometry
    is nonfinite, degenerate, or contained; ``valid`` excludes both.
    ``successful`` is false on overflow or any failed row.
    """

    distance: ContactDistanceEvaluation
    mollifier: Array
    mollifier_margin: Array
    active: Array
    failed: Array
    valid: Array
    finite: Array
    successful: Array
    route_keys: Array
    witness_slots: Array
    weights: Array
    minimum_separation: Array
    batch_id: str = eqx.field(static=True)


def _plane_rows(
    scene: ContactStencilScene | None,
    batch: ContactStencilBatch,
    dtype: DTypeLike,
    /,
) -> tuple[Array, Array]:
    if not isinstance(scene, ContactParticipantScene) or scene.plane_count == 0:
        raise TypeError("Plane routes require the ContactParticipantScene with planes.")
    first_plane = scene.feature_count - scene.plane_count
    plane = jnp.clip(batch.right_feature_indices - first_plane, 0, scene.plane_count - 1)
    normals = scene.plane_normals.astype(dtype)
    offsets = scene.plane_offsets.astype(dtype)
    return normals[plane], offsets[plane]


def evaluate_contact_stencils(
    batch: ContactStencilBatch,
    positions: ArrayLike,
    rest_positions: ArrayLike,
    /,
    *,
    scene: ContactStencilScene | None = None,
    tolerance: float = 1.0e-12,
) -> ContactStencilEvaluation:
    """Evaluate canonical routes with the shared distance kernels.

    Plane routes read the plane geometry of ``scene``. Capsule support
    manifolds (``EDGE_EDGE`` with two witness slots) evaluate each slot with
    the rigid support kernel; mesh edge-edge routes keep the mollified unique
    closest-point law.
    """
    if not isinstance(batch, ContactStencilBatch):
        raise TypeError("batch must be ContactStencilBatch.")
    current = jnp.asarray(positions)
    rest = jnp.asarray(rest_positions, dtype=current.dtype)
    if current.shape != rest.shape or current.ndim != 2 or current.shape[1] not in (2, 3):
        raise ValueError(
            "positions and rest_positions must have matching (vertices, dimension) shape."
        )
    safe = jnp.clip(batch.vertex_indices, 0, current.shape[0] - 1)
    values = current[safe]
    rest_values = rest[safe]
    ones = jnp.ones((batch.capacity,), dtype=current.dtype)
    unbounded = jnp.full((batch.capacity,), jnp.inf, dtype=current.dtype)
    active = jnp.ones((batch.capacity,), dtype=jnp.bool_)
    mollifier = ones
    mollifier_margin = unbounded
    kind = batch.kind
    if kind == ContactStencilKind.VERTEX_VERTEX:
        distance = point_point_distance(values[:, 0], values[:, 1], tolerance=tolerance)
    elif kind == ContactStencilKind.EDGE_VERTEX:
        distance = point_edge_distance(
            values[:, 0], values[:, 1], values[:, 2], tolerance=tolerance
        )
    elif kind == ContactStencilKind.EDGE_EDGE and batch.witness_slot_count > 1:
        distance, active = segment_segment_support_distance(
            values[:, 0],
            values[:, 1],
            values[:, 2],
            values[:, 3],
            batch.witness_slots,
            tolerance=tolerance,
        )
    elif kind == ContactStencilKind.EDGE_EDGE:
        distance = edge_edge_distance(
            values[:, 0], values[:, 1], values[:, 2], values[:, 3], tolerance=tolerance
        )
        threshold = edge_edge_mollifier_threshold(
            rest_values[:, 0], rest_values[:, 1], rest_values[:, 2], rest_values[:, 3]
        )
        mollifier, mollifier_margin = edge_edge_mollifier(
            values[:, 0], values[:, 1], values[:, 2], values[:, 3], threshold
        )
    elif kind == ContactStencilKind.FACE_VERTEX:
        distance = point_triangle_distance(
            values[:, 0], values[:, 1], values[:, 2], values[:, 3], tolerance=tolerance
        )
    elif kind == ContactStencilKind.VERTEX_PLANE:
        normal, offset = _plane_rows(scene, batch, current.dtype)
        distance = point_plane_distance(values[:, 0], normal, offset, tolerance=tolerance)
    elif kind == ContactStencilKind.EDGE_PLANE:
        normal, offset = _plane_rows(scene, batch, current.dtype)
        second = batch.witness_slots == int(ContactWitnessSlot.SECOND)
        endpoint = jnp.where(second[:, None], values[:, 1], values[:, 0])
        distance = point_plane_distance(endpoint, normal, offset, tolerance=tolerance)
        unit = distance.coefficients[:, 0]
        coefficients = jnp.zeros_like(distance.coefficients)
        coefficients = coefficients.at[:, 0].set(jnp.where(second, 0.0, unit))
        coefficients = coefficients.at[:, 1].set(jnp.where(second, unit, 0.0))
        distance = eqx.tree_at(lambda value: value.coefficients, distance, coefficients)
    else:
        raise TypeError("Unsupported contact stencil kind.")
    finite = distance.finite & jnp.isfinite(mollifier)
    considered = batch.valid & active
    failed = considered & ~(finite & distance.nondegenerate)
    valid = considered & ~failed
    successful = ~batch.overflow & ~jnp.any(failed)
    return ContactStencilEvaluation(
        distance,
        jnp.where(valid, mollifier, 0.0),
        jnp.where(valid, mollifier_margin, 0.0),
        active,
        failed,
        valid,
        jnp.all((~considered) | finite),
        successful,
        batch.route_keys,
        batch.witness_slots,
        batch.weights.astype(current.dtype),
        batch.minimum_separation.astype(current.dtype),
        batch.batch_id,
    )


__all__ = [
    "ContactPrimitiveProfile",
    "ContactRouteKeySchema",
    "ContactStencilBatch",
    "ContactStencilEvaluation",
    "ContactStencilKind",
    "ContactStencilLayout",
    "ContactStencilPacking",
    "contact_stencil_arity",
    "contact_stencil_feature_kinds",
    "contact_stencil_symmetric",
    "empty_contact_stencil_batch",
    "evaluate_contact_stencils",
    "pack_contact_stencil_batch",
    "prepare_contact_stencil_batch",
    "route_endpoints",
    "scene_feature_endpoints",
]
