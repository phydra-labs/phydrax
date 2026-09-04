#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._distance import (
    ContactDistanceEvaluation,
    edge_edge_distance,
    edge_edge_mollifier,
    edge_edge_mollifier_threshold,
    point_edge_distance,
    point_point_distance,
    point_triangle_distance,
)


class ContactStencilKind(IntEnum):
    VERTEX_VERTEX = 0
    EDGE_VERTEX = 1
    EDGE_EDGE = 2
    FACE_VERTEX = 3


_ARITY = {
    ContactStencilKind.VERTEX_VERTEX: 2,
    ContactStencilKind.EDGE_VERTEX: 3,
    ContactStencilKind.EDGE_EDGE: 4,
    ContactStencilKind.FACE_VERTEX: 4,
}


def canonical_contact_route_keys(
    kind: ContactStencilKind,
    left_feature_ids: ArrayLike,
    right_feature_ids: ArrayLike,
    /,
) -> np.ndarray:
    """Return collision-free Cantor keys for canonical nonnegative feature pairs."""
    if not isinstance(kind, ContactStencilKind):
        raise TypeError("kind must be ContactStencilKind.")
    left = np.asarray(left_feature_ids)
    right = np.asarray(right_feature_ids)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Contact feature ID arrays must be matching vectors.")
    if not np.issubdtype(left.dtype, np.integer) or not np.issubdtype(
        right.dtype, np.integer
    ):
        raise TypeError("Contact feature IDs must contain integers.")
    left = left.astype(np.int64, copy=False)
    right = right.astype(np.int64, copy=False)
    if np.any(left < 0) or np.any(right < 0):
        raise ValueError("Contact feature IDs must be nonnegative.")
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    maximum = int(max(np.max(high, initial=0), np.max(low, initial=0)))
    if maximum > 1_500_000_000:
        raise ValueError(
            "Contact feature IDs are too large for collision-free int64 keys."
        )
    total = low + high
    pair = total * (total + 1) // 2 + high
    key = 4 * pair + int(kind)
    if np.any(key < 0) or np.unique(key).size != key.size:
        raise ValueError(
            "Contact route-key construction overflowed or produced duplicates."
        )
    return key.astype(np.int64, copy=False)


class ContactStencilBatch(StrictModule, NonTrainableState):
    """One fixed-capacity homogeneous collision-stencil batch."""

    vertex_indices: Array
    left_feature_ids: Array
    right_feature_ids: Array
    left_feature_indices: Array
    right_feature_indices: Array
    route_keys: Array
    weights: Array
    minimum_separation: Array
    valid: Array
    actual_count: Array
    overflow_count: Array
    overflow: Array
    kind: ContactStencilKind = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ContactStencilKind,
        vertex_indices: ArrayLike,
        left_feature_ids: ArrayLike,
        right_feature_ids: ArrayLike,
        /,
        *,
        capacity: int,
        feature_indices: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        minimum_separation: ArrayLike | None = None,
        valid: ArrayLike | None = None,
        actual_count: ArrayLike | None = None,
        overflow_count: ArrayLike = 0,
        route_keys: ArrayLike | None = None,
        batch_id: str | None = None,
    ):
        if not isinstance(kind, ContactStencilKind):
            raise TypeError("kind must be ContactStencilKind.")
        count = int(capacity)
        if count < 0:
            raise ValueError("Contact stencil capacity must be nonnegative.")
        indices = np.asarray(vertex_indices)
        if indices.shape != (count, 4) or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("vertex_indices must be an integer (capacity, 4) array.")
        indices = indices.astype(np.int32, copy=False)
        arity = _ARITY[kind]
        active = (
            np.ones((count,), dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if active.shape != (count,):
            raise ValueError("Contact stencil valid mask must have capacity shape.")
        if np.any(active[:, None] & (indices < 0) & (np.arange(4)[None, :] < arity)):
            raise ValueError("Active contact stencil endpoints must be nonnegative.")
        if np.any(active[:, None] & (indices >= 0) & (np.arange(4)[None, :] >= arity)):
            raise ValueError(
                "Contact stencil padding must use negative endpoint indices."
            )
        left = np.asarray(left_feature_ids)
        right = np.asarray(right_feature_ids)
        if left.shape != (count,) or right.shape != (count,):
            raise ValueError("Contact feature IDs must have capacity shape.")
        if not np.issubdtype(left.dtype, np.integer) or not np.issubdtype(
            right.dtype, np.integer
        ):
            raise TypeError("Contact feature IDs must contain integers.")
        left = left.astype(np.int64, copy=False)
        right = right.astype(np.int64, copy=False)
        if np.any(active & ((left < 0) | (right < 0))):
            raise ValueError("Active contact feature IDs must be nonnegative.")
        if feature_indices is None:
            local_features = np.stack((left, right), axis=1)
        else:
            local_features = np.asarray(feature_indices)
            if local_features.shape != (count, 2) or not np.issubdtype(
                local_features.dtype, np.integer
            ):
                raise TypeError("feature_indices must be an integer (capacity, 2) array.")
            local_features = local_features.astype(np.int32, copy=False)
        if np.any(active[:, None] & (local_features < 0)):
            raise ValueError("Active local feature indices must be nonnegative.")
        weight = (
            np.ones((count,), dtype=float) if weights is None else np.asarray(weights)
        )
        separation = (
            np.zeros((count,), dtype=float)
            if minimum_separation is None
            else np.asarray(minimum_separation)
        )
        if weight.shape != (count,) or separation.shape != (count,):
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
        if true_count < 0 or overflow_count_ < 0 or true_count > count + overflow_count_:
            raise ValueError("Contact stencil counts are inconsistent.")
        if route_keys is None:
            keys = np.zeros((count,), dtype=np.int64)
            if np.any(active):
                keys[active] = canonical_contact_route_keys(
                    kind, left[active], right[active]
                )
        else:
            keys = np.asarray(route_keys)
            if keys.shape != (count,) or not np.issubdtype(keys.dtype, np.integer):
                raise TypeError("route_keys must be one integer capacity vector.")
            keys = keys.astype(np.int64, copy=False)
        if np.any(active & (keys < 0)) or np.unique(keys[active]).size != int(
            np.count_nonzero(active)
        ):
            raise ValueError("Active contact route keys must be unique and nonnegative.")
        generated = canonical_fingerprint(
            {
                "kind": "contact-stencil-batch",
                "stencil_kind": int(kind),
                "capacity": count,
                "indices": array_tree_fingerprint(indices),
                "features": array_tree_fingerprint((left, right)),
                "feature_indices": array_tree_fingerprint(local_features),
                "keys": array_tree_fingerprint(keys),
                "valid": array_tree_fingerprint(active),
            }
        )
        identifier = generated if batch_id is None else str(batch_id)
        if not identifier:
            raise ValueError("batch_id must be nonempty or None.")
        dtype = jnp.result_type(weight, separation, float)
        self.vertex_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.left_feature_ids = jnp.asarray(left, dtype=jnp.int64)
        self.right_feature_ids = jnp.asarray(right, dtype=jnp.int64)
        self.left_feature_indices = jnp.asarray(local_features[:, 0], dtype=jnp.int32)
        self.right_feature_indices = jnp.asarray(local_features[:, 1], dtype=jnp.int32)
        self.route_keys = jnp.asarray(keys, dtype=jnp.int64)
        self.weights = jnp.asarray(weight, dtype=dtype)
        self.minimum_separation = jnp.asarray(separation, dtype=dtype)
        self.valid = jnp.asarray(active)
        self.actual_count = jnp.asarray(true_count, dtype=jnp.int32)
        self.overflow_count = jnp.asarray(overflow_count_, dtype=jnp.int32)
        self.overflow = jnp.asarray(overflow_count_ > 0)
        self.kind = kind
        self.capacity = count
        self.batch_id = identifier

    @classmethod
    def empty(
        cls,
        kind: ContactStencilKind,
        capacity: int,
        /,
        *,
        dtype=jnp.float64,
    ) -> ContactStencilBatch:
        count = int(capacity)
        return cls(
            kind,
            np.full((count, 4), -1, dtype=np.int32),
            np.zeros((count,), dtype=np.int64),
            np.zeros((count,), dtype=np.int64),
            capacity=count,
            weights=np.zeros((count,), dtype=np.dtype(dtype)),
            minimum_separation=np.zeros((count,), dtype=np.dtype(dtype)),
            feature_indices=np.zeros((count, 2), dtype=np.int32),
            valid=np.zeros((count,), dtype=bool),
        )


class ContactStencilEvaluation(StrictModule):
    distance: ContactDistanceEvaluation
    mollifier: Array
    mollifier_margin: Array
    valid: Array
    finite: Array
    successful: Array
    route_keys: Array
    weights: Array
    minimum_separation: Array
    batch_id: str = eqx.field(static=True)


def evaluate_contact_stencils(
    batch: ContactStencilBatch,
    positions: ArrayLike,
    rest_positions: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-12,
) -> ContactStencilEvaluation:
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
    if batch.kind == ContactStencilKind.VERTEX_VERTEX:
        distance = point_point_distance(values[:, 0], values[:, 1], tolerance=tolerance)
        mollifier = jnp.ones((batch.capacity,), dtype=current.dtype)
        mollifier_margin = jnp.full((batch.capacity,), jnp.inf, dtype=current.dtype)
    elif batch.kind == ContactStencilKind.EDGE_VERTEX:
        distance = point_edge_distance(
            values[:, 0], values[:, 1], values[:, 2], tolerance=tolerance
        )
        mollifier = jnp.ones((batch.capacity,), dtype=current.dtype)
        mollifier_margin = jnp.full((batch.capacity,), jnp.inf, dtype=current.dtype)
    elif batch.kind == ContactStencilKind.EDGE_EDGE:
        distance = edge_edge_distance(
            values[:, 0], values[:, 1], values[:, 2], values[:, 3], tolerance=tolerance
        )
        threshold = edge_edge_mollifier_threshold(
            rest_values[:, 0], rest_values[:, 1], rest_values[:, 2], rest_values[:, 3]
        )
        mollifier, mollifier_margin = edge_edge_mollifier(
            values[:, 0], values[:, 1], values[:, 2], values[:, 3], threshold
        )
    elif batch.kind == ContactStencilKind.FACE_VERTEX:
        distance = point_triangle_distance(
            values[:, 0], values[:, 1], values[:, 2], values[:, 3], tolerance=tolerance
        )
        mollifier = jnp.ones((batch.capacity,), dtype=current.dtype)
        mollifier_margin = jnp.full((batch.capacity,), jnp.inf, dtype=current.dtype)
    else:
        raise TypeError("Unsupported contact stencil kind.")
    finite = distance.finite & jnp.isfinite(mollifier)
    valid = batch.valid & distance.nondegenerate & finite
    successful = (
        ~batch.overflow
        & jnp.all((~batch.valid) | finite)
        & jnp.all((~batch.valid) | distance.nondegenerate)
    )
    return ContactStencilEvaluation(
        distance,
        jnp.where(valid, mollifier, 0.0),
        jnp.where(valid, mollifier_margin, 0.0),
        valid,
        jnp.all((~batch.valid) | finite),
        successful,
        batch.route_keys,
        batch.weights.astype(current.dtype),
        batch.minimum_separation.astype(current.dtype),
        batch.batch_id,
    )


__all__ = [
    "ContactStencilBatch",
    "ContactStencilEvaluation",
    "ContactStencilKind",
    "canonical_contact_route_keys",
    "evaluate_contact_stencils",
]
