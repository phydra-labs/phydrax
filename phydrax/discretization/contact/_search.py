#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import time
from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._participant import ContactParticipantScene
from ._stencils import (
    canonical_contact_route_keys,
    ContactStencilBatch,
    ContactStencilKind,
)
from ._surface import PreparedCollisionScene


ContactSearchScene = PreparedCollisionScene | ContactParticipantScene


class ContactSearchStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    CANDIDATE_OVERFLOW = 2
    MEMORY_LIMIT = 3
    TIME_LIMIT = 4


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
    ):
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


class ContactCandidateEpoch(StrictModule, NonTrainableState):
    """Fixed-capacity candidates and fail-closed search evidence for one geometry epoch."""

    edge_vertex: ContactStencilBatch
    edge_edge: ContactStencilBatch
    face_vertex: ContactStencilBatch
    reference_positions: Array
    envelope_radius: Array
    candidate_count: Array
    estimated_memory_bytes: Array
    elapsed_seconds: Array
    status: Array
    complete: Array
    search_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.complete & (self.status == int(ContactSearchStatus.SUCCESS))

    @property
    def active_batches(self) -> tuple[ContactStencilBatch, ...]:
        return tuple(
            batch
            for batch in (self.edge_vertex, self.edge_edge, self.face_vertex)
            if batch.capacity
        )


class _AbstractContactSearchPlan(StrictModule, NonTrainableState):
    edge_vertex_capacity: int = eqx.field(static=True)
    edge_edge_capacity: int = eqx.field(static=True)
    face_vertex_capacity: int = eqx.field(static=True)
    activation_distance: float = eqx.field(static=True)
    envelope_radius: float = eqx.field(static=True)
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
        limits: ContactSearchLimits | None = None,
        method: str,
    ):
        capacities = tuple(
            int(value)
            for value in (
                edge_vertex_capacity,
                edge_edge_capacity,
                face_vertex_capacity,
            )
        )
        if any(value < 0 for value in capacities) or sum(capacities) <= 0:
            raise ValueError(
                "Contact search capacities must be nonnegative with positive total."
            )
        activation = float(activation_distance)
        envelope = float(envelope_radius)
        if not isfinite(activation) or activation <= 0.0:
            raise ValueError("activation_distance must be finite and positive.")
        if not isfinite(envelope) or envelope < 0.0:
            raise ValueError("envelope_radius must be finite and nonnegative.")
        limits_ = ContactSearchLimits() if limits is None else limits
        if not isinstance(limits_, ContactSearchLimits):
            raise TypeError("limits must be ContactSearchLimits or None.")
        method_ = str(method)
        if method_ not in ("dense", "sweep-and-prune"):
            raise ValueError("Unknown contact search method.")
        self.edge_vertex_capacity, self.edge_edge_capacity, self.face_vertex_capacity = (
            capacities
        )
        self.activation_distance = activation
        self.envelope_radius = envelope
        self.limits = limits_
        self.method = method_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "contact-search-plan",
                "method": method_,
                "capacities": capacities,
                "activation_distance": activation.hex(),
                "envelope_radius": envelope.hex(),
                "limits": (
                    limits_.maximum_candidates,
                    limits_.maximum_memory_bytes,
                    limits_.maximum_time_seconds,
                ),
            }
        )

    def build(
        self,
        scene: ContactSearchScene,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
    ) -> ContactCandidateEpoch:
        return _build_epoch(self, scene, positions, end_positions)


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
        limits: ContactSearchLimits | None = None,
    ):
        super().__init__(
            edge_vertex_capacity=edge_vertex_capacity,
            edge_edge_capacity=edge_edge_capacity,
            face_vertex_capacity=face_vertex_capacity,
            activation_distance=activation_distance,
            envelope_radius=envelope_radius,
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
        limits: ContactSearchLimits | None = None,
    ):
        super().__init__(
            edge_vertex_capacity=edge_vertex_capacity,
            edge_edge_capacity=edge_edge_capacity,
            face_vertex_capacity=face_vertex_capacity,
            activation_distance=activation_distance,
            envelope_radius=envelope_radius,
            limits=limits,
            method="sweep-and-prune",
        )


def _swept_bounds(
    positions: np.ndarray,
    topology: np.ndarray,
    end_positions: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    start = positions[topology]
    end = end_positions[topology]
    return np.minimum(start.min(axis=1), end.min(axis=1)), np.maximum(
        start.max(axis=1), end.max(axis=1)
    )


def _point_bounds(
    positions: np.ndarray,
    end_positions: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    return np.minimum(positions, end_positions), np.maximum(positions, end_positions)


def _aabb_distance_squared(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
    /,
) -> float:
    delta = np.maximum(0.0, np.maximum(left_min - right_max, right_min - left_max))
    return float(np.dot(delta, delta))


def _bipartite_pairs(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
    radius: float,
    method: str,
    /,
) -> list[tuple[int, int]]:
    if method == "dense":
        return [
            (left, right)
            for left in range(left_min.shape[0])
            for right in range(right_min.shape[0])
            if _aabb_distance_squared(
                left_min[left], left_max[left], right_min[right], right_max[right]
            )
            <= radius * radius
        ]
    all_min = np.minimum(left_min.min(axis=0), right_min.min(axis=0))
    all_max = np.maximum(left_max.max(axis=0), right_max.max(axis=0))
    axis = int(np.argmax(all_max - all_min))
    order = np.argsort(right_min[:, axis], kind="stable")
    sorted_lower = right_min[order, axis]
    result: list[tuple[int, int]] = []
    for left in range(left_min.shape[0]):
        stop = int(
            np.searchsorted(sorted_lower, left_max[left, axis] + radius, side="right")
        )
        candidates = order[:stop]
        candidates = candidates[
            right_max[candidates, axis] + radius >= left_min[left, axis]
        ]
        for right in candidates.tolist():
            if (
                _aabb_distance_squared(
                    left_min[left], left_max[left], right_min[right], right_max[right]
                )
                <= radius * radius
            ):
                result.append((left, int(right)))
    return result


def _same_set_pairs(
    lower: np.ndarray,
    upper: np.ndarray,
    radius: float,
    method: str,
    /,
) -> list[tuple[int, int]]:
    if method == "dense":
        return [
            (left, right)
            for left in range(lower.shape[0])
            for right in range(left + 1, lower.shape[0])
            if _aabb_distance_squared(
                lower[left], upper[left], lower[right], upper[right]
            )
            <= radius * radius
        ]
    axis = int(np.argmax(upper.max(axis=0) - lower.min(axis=0)))
    order = np.argsort(lower[:, axis], kind="stable")
    result: list[tuple[int, int]] = []
    for ordered_left in range(order.size):
        left = int(order[ordered_left])
        for ordered_right in range(ordered_left + 1, order.size):
            right = int(order[ordered_right])
            if lower[right, axis] > upper[left, axis] + radius:
                break
            if upper[right, axis] + radius < lower[left, axis]:
                continue
            if (
                _aabb_distance_squared(
                    lower[left], upper[left], lower[right], upper[right]
                )
                <= radius * radius
            ):
                result.append((min(left, right), max(left, right)))
    return result


def _scene_exclusions(scene: ContactSearchScene, /) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    if isinstance(scene, PreparedCollisionScene):
        plans = tuple(surface.plan for surface in scene.surfaces)
    elif isinstance(scene, ContactParticipantScene):
        plans = tuple(participant.surface_plan for participant in scene.participants)
    else:
        raise TypeError("scene must be a prepared collision or participant scene.")
    for surface_index, plan in enumerate(plans):
        offset = scene.vertex_offsets[surface_index]
        for left, right in np.asarray(plan.pair_policy.excluded_vertex_pairs).tolist():
            pairs.add(
                (
                    min(offset + int(left), offset + int(right)),
                    max(offset + int(left), offset + int(right)),
                )
            )
    return pairs


def _primitive_allowed(
    left_vertices: tuple[int, ...],
    right_vertices: tuple[int, ...],
    left_feature: int,
    right_feature: int,
    static: np.ndarray,
    participants: np.ndarray,
    pair_policy,
    exclusions: set[tuple[int, int]],
    /,
) -> bool:
    if set(left_vertices) & set(right_vertices):
        return False
    if bool(static[left_feature]) and bool(static[right_feature]):
        return False
    if not pair_policy.allows(
        int(participants[left_feature]), int(participants[right_feature])
    ):
        return False
    for left in left_vertices:
        for right in right_vertices:
            if (min(left, right), max(left, right)) in exclusions:
                return False
    return True


def _pair_minimum_separation(
    left_feature: int,
    right_feature: int,
    contact_extent: np.ndarray,
    /,
) -> float:
    return float(contact_extent[left_feature] + contact_extent[right_feature])


def _pack_batch(
    kind: ContactStencilKind,
    records: list[tuple[tuple[int, int], tuple[int, int], tuple[int, ...], float]],
    capacity: int,
    dtype: np.dtype,
    /,
) -> ContactStencilBatch:
    actual = len(records)
    overflow_count = max(actual - capacity, 0)
    indices = np.full((capacity, 4), -1, dtype=np.int32)
    left = np.zeros((capacity,), dtype=np.int64)
    right = np.zeros((capacity,), dtype=np.int64)
    separation = np.zeros((capacity,), dtype=dtype)
    feature_indices = np.zeros((capacity, 2), dtype=np.int32)
    valid = np.zeros((capacity,), dtype=bool)
    if records:
        feature_left = np.asarray([record[0][0] for record in records], dtype=np.int64)
        feature_right = np.asarray([record[0][1] for record in records], dtype=np.int64)
        keys = canonical_contact_route_keys(kind, feature_left, feature_right)
        order = np.argsort(keys, kind="stable")
        ordered = [records[index] for index in order[:capacity]]
        for slot, (features, local_features, endpoints, dmin) in enumerate(ordered):
            indices[slot, : len(endpoints)] = endpoints
            left[slot], right[slot] = features
            feature_indices[slot] = local_features
            separation[slot] = dmin
        if overflow_count == 0:
            valid[: len(ordered)] = True
    return ContactStencilBatch(
        kind,
        indices,
        left,
        right,
        capacity=capacity,
        feature_indices=feature_indices,
        minimum_separation=separation,
        valid=valid,
        actual_count=actual,
        overflow_count=overflow_count,
    )


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
        and count * 96 > limits.maximum_memory_bytes
    ):
        return ContactSearchStatus.MEMORY_LIMIT
    if limits.maximum_time_seconds is not None and elapsed > limits.maximum_time_seconds:
        return ContactSearchStatus.TIME_LIMIT
    return ContactSearchStatus.SUCCESS


def _build_epoch(
    plan: _AbstractContactSearchPlan,
    scene: ContactSearchScene,
    positions: ArrayLike,
    end_positions: ArrayLike | None,
    /,
) -> ContactCandidateEpoch:
    if not isinstance(scene, (PreparedCollisionScene, ContactParticipantScene)):
        raise TypeError(
            "scene must be PreparedCollisionScene or ContactParticipantScene."
        )
    start_time = time.perf_counter()
    start = np.asarray(positions)
    if not np.issubdtype(start.dtype, np.floating):
        start = start.astype(np.float64)
    end = start if end_positions is None else np.asarray(end_positions, dtype=start.dtype)
    expected = (scene.vertex_count, scene.ambient_dimension)
    if (
        start.shape != expected
        or end.shape != expected
        or np.any(~np.isfinite(start))
        or np.any(~np.isfinite(end))
    ):
        raise ValueError(
            f"Contact search positions must be finite arrays of shape {expected}."
        )
    edges = np.asarray(scene.edges, dtype=np.int32)
    faces = np.asarray(scene.faces, dtype=np.int32)
    point_min, point_max = _point_bounds(start, end)
    edge_min, edge_max = _swept_bounds(start, edges, end)
    face_min, face_max = (
        _swept_bounds(start, faces, end)
        if faces.shape[0]
        else (
            np.empty((0, scene.ambient_dimension)),
            np.empty((0, scene.ambient_dimension)),
        )
    )
    feature_ids = np.asarray(scene.feature_ids, dtype=np.int64)
    feature_participants = np.asarray(scene.feature_participant_ids, dtype=np.int64)
    feature_static = np.asarray(scene.feature_static_mask, dtype=bool)
    contact_extent = np.asarray(scene.feature_contact_extent)
    maximum_pair_separation = 2.0 * float(np.max(contact_extent, initial=0.0))
    broad_radius = (
        plan.activation_distance + maximum_pair_separation + plan.envelope_radius
    )
    exclusions = _scene_exclusions(scene)
    pair_policy = scene.pair_policy
    vertex_count = scene.vertex_count
    edge_feature_offset = vertex_count
    face_feature_offset = vertex_count + scene.edge_count

    edge_vertex_records: list[
        tuple[tuple[int, int], tuple[int, int], tuple[int, ...], float]
    ] = []
    edge_edge_records: list[
        tuple[tuple[int, int], tuple[int, int], tuple[int, ...], float]
    ] = []
    face_vertex_records: list[
        tuple[tuple[int, int], tuple[int, int], tuple[int, ...], float]
    ] = []

    if scene.ambient_dimension == 2 or faces.shape[0] == 0:
        for vertex, edge_index in _bipartite_pairs(
            point_min, point_max, edge_min, edge_max, broad_radius, plan.method
        ):
            edge = tuple(int(value) for value in edges[edge_index])
            point = (int(vertex),)
            edge_feature = edge_feature_offset + edge_index
            if not _primitive_allowed(
                point,
                edge,
                int(vertex),
                edge_feature,
                feature_static,
                feature_participants,
                pair_policy,
                exclusions,
            ):
                continue
            dmin = _pair_minimum_separation(int(vertex), edge_feature, contact_extent)
            if (
                _aabb_distance_squared(
                    point_min[vertex],
                    point_max[vertex],
                    edge_min[edge_index],
                    edge_max[edge_index],
                )
                > (plan.activation_distance + dmin + plan.envelope_radius) ** 2
            ):
                continue
            edge_vertex_records.append(
                (
                    (int(feature_ids[vertex]), int(feature_ids[edge_feature])),
                    (int(vertex), edge_feature),
                    point + edge,
                    dmin,
                )
            )
    else:
        for vertex, face_index in _bipartite_pairs(
            point_min, point_max, face_min, face_max, broad_radius, plan.method
        ):
            face = tuple(int(value) for value in faces[face_index])
            point = (int(vertex),)
            face_feature = face_feature_offset + face_index
            if not _primitive_allowed(
                point,
                face,
                int(vertex),
                face_feature,
                feature_static,
                feature_participants,
                pair_policy,
                exclusions,
            ):
                continue
            dmin = _pair_minimum_separation(int(vertex), face_feature, contact_extent)
            if (
                _aabb_distance_squared(
                    point_min[vertex],
                    point_max[vertex],
                    face_min[face_index],
                    face_max[face_index],
                )
                > (plan.activation_distance + dmin + plan.envelope_radius) ** 2
            ):
                continue
            face_vertex_records.append(
                (
                    (int(feature_ids[vertex]), int(feature_ids[face_feature])),
                    (int(vertex), face_feature),
                    point + face,
                    dmin,
                )
            )
        for first_edge, second_edge in _same_set_pairs(
            edge_min, edge_max, broad_radius, plan.method
        ):
            first = tuple(int(value) for value in edges[first_edge])
            second = tuple(int(value) for value in edges[second_edge])
            left_feature = edge_feature_offset + first_edge
            right_feature = edge_feature_offset + second_edge
            if not _primitive_allowed(
                first,
                second,
                left_feature,
                right_feature,
                feature_static,
                feature_participants,
                pair_policy,
                exclusions,
            ):
                continue
            dmin = _pair_minimum_separation(left_feature, right_feature, contact_extent)
            if (
                _aabb_distance_squared(
                    edge_min[first_edge],
                    edge_max[first_edge],
                    edge_min[second_edge],
                    edge_max[second_edge],
                )
                > (plan.activation_distance + dmin + plan.envelope_radius) ** 2
            ):
                continue
            edge_edge_records.append(
                (
                    (
                        int(feature_ids[left_feature]),
                        int(feature_ids[right_feature]),
                    ),
                    (left_feature, right_feature),
                    first + second,
                    dmin,
                )
            )

    edge_vertex = _pack_batch(
        ContactStencilKind.EDGE_VERTEX,
        edge_vertex_records,
        plan.edge_vertex_capacity,
        start.dtype,
    )
    edge_edge = _pack_batch(
        ContactStencilKind.EDGE_EDGE,
        edge_edge_records,
        plan.edge_edge_capacity,
        start.dtype,
    )
    face_vertex = _pack_batch(
        ContactStencilKind.FACE_VERTEX,
        face_vertex_records,
        plan.face_vertex_capacity,
        start.dtype,
    )
    candidate_count = (
        len(edge_vertex_records) + len(edge_edge_records) + len(face_vertex_records)
    )
    elapsed = time.perf_counter() - start_time
    overflow = bool(edge_vertex.overflow or edge_edge.overflow or face_vertex.overflow)
    status = (
        ContactSearchStatus.CANDIDATE_OVERFLOW
        if overflow
        else _limit_status(plan, candidate_count, elapsed)
    )
    complete = status == ContactSearchStatus.SUCCESS
    epoch_id = canonical_fingerprint(
        {
            "kind": "contact-candidate-epoch",
            "scene": scene.scene_id,
            "search": plan.plan_id,
            "batches": (edge_vertex.batch_id, edge_edge.batch_id, face_vertex.batch_id),
            "complete": complete,
        }
    )
    return ContactCandidateEpoch(
        edge_vertex,
        edge_edge,
        face_vertex,
        jnp.asarray(start),
        jnp.asarray(plan.envelope_radius, dtype=start.dtype),
        jnp.asarray(candidate_count, dtype=jnp.int32),
        jnp.asarray(candidate_count * 96, dtype=jnp.int64),
        jnp.asarray(elapsed),
        jnp.asarray(int(status), dtype=jnp.int32),
        jnp.asarray(complete),
        plan.plan_id,
        epoch_id,
    )


__all__ = [
    "ContactCandidateEpoch",
    "ContactSearchLimits",
    "ContactSearchStatus",
    "DenseContactSearchPlan",
    "SweepAndPruneContactSearchPlan",
]
