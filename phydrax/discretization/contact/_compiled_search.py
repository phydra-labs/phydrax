#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._stencils import ContactStencilKind
from ._surface import PreparedCollisionScene


class CompiledCandidateBatch(StrictModule):
    vertex_indices: Array
    valid: Array
    actual_count: Array
    overflow_count: Array
    kind: ContactStencilKind = eqx.field(static=True)
    capacity: int = eqx.field(static=True)


class CompiledContactSearchEvidence(StrictModule):
    candidate_count: Array
    overflow_count: Array
    finite: Array
    complete: Array
    plan_id: str = eqx.field(static=True)


class CompiledContactSearchResult(StrictModule):
    edge_vertex: CompiledCandidateBatch
    edge_edge: CompiledCandidateBatch
    face_vertex: CompiledCandidateBatch
    evidence: CompiledContactSearchEvidence


class CompiledContactSearchPlan(StrictModule, NonTrainableState):
    """Fixed-shape device search over predeclared legal primitive pairs.

    The runtime filter is fully JAX compilable. Preparation enumerates legal
    pairs once; runtime evaluates swept/static AABB separation and packs each
    kind under fixed capacity. This is the deterministic compiled authority
    before a future asymptotically faster device LBVH backend.
    """

    edge_vertex_pairs: Array
    edge_edge_pairs: Array
    face_vertex_pairs: Array
    edges: Array
    faces: Array
    edge_vertex_capacity: int = eqx.field(static=True)
    edge_edge_capacity: int = eqx.field(static=True)
    face_vertex_capacity: int = eqx.field(static=True)
    activation_distance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scene: PreparedCollisionScene,
        /,
        *,
        edge_vertex_capacity: int,
        edge_edge_capacity: int,
        face_vertex_capacity: int,
        activation_distance: float,
    ):
        if not isinstance(scene, PreparedCollisionScene):
            raise TypeError("scene must be PreparedCollisionScene.")
        capacities = (
            int(edge_vertex_capacity),
            int(edge_edge_capacity),
            int(face_vertex_capacity),
        )
        if any(value < 0 for value in capacities) or sum(capacities) <= 0:
            raise ValueError("Compiled search capacities are invalid.")
        activation = float(activation_distance)
        if not np.isfinite(activation) or activation <= 0.0:
            raise ValueError("activation_distance must be finite and positive.")
        edges = np.asarray(scene.edges, dtype=np.int32)
        faces = np.asarray(scene.faces, dtype=np.int32)
        edge_vertex = []
        edge_edge = []
        face_vertex = []
        if scene.ambient_dimension == 2 or faces.size == 0:
            for vertex in range(scene.vertex_count):
                for edge_index, edge in enumerate(edges):
                    if vertex not in edge:
                        edge_vertex.append((vertex, edge_index))
        else:
            for vertex in range(scene.vertex_count):
                for face_index, face in enumerate(faces):
                    if vertex not in face:
                        face_vertex.append((vertex, face_index))
            for first in range(edges.shape[0]):
                for second in range(first + 1, edges.shape[0]):
                    if not set(edges[first]).intersection(edges[second]):
                        edge_edge.append((first, second))
        self.edge_vertex_pairs = jnp.asarray(
            np.asarray(edge_vertex, dtype=np.int32).reshape((-1, 2))
        )
        self.edge_edge_pairs = jnp.asarray(
            np.asarray(edge_edge, dtype=np.int32).reshape((-1, 2))
        )
        self.face_vertex_pairs = jnp.asarray(
            np.asarray(face_vertex, dtype=np.int32).reshape((-1, 2))
        )
        self.edges = jnp.asarray(edges)
        self.faces = jnp.asarray(faces)
        (
            self.edge_vertex_capacity,
            self.edge_edge_capacity,
            self.face_vertex_capacity,
        ) = capacities
        self.activation_distance = activation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compiled-contact-search-plan",
                "scene": scene.scene_id,
                "edge_vertex": array_tree_fingerprint(edge_vertex),
                "edge_edge": array_tree_fingerprint(edge_edge),
                "face_vertex": array_tree_fingerprint(face_vertex),
                "capacities": capacities,
                "activation_distance": activation.hex(),
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
    ) -> CompiledContactSearchResult:
        start = jnp.asarray(positions)
        end = (
            start
            if end_positions is None
            else jnp.asarray(end_positions, dtype=start.dtype)
        )
        if start.shape != end.shape or start.ndim != 2:
            raise ValueError("Compiled search positions have invalid shape.")
        point_min = jnp.minimum(start, end)
        point_max = jnp.maximum(start, end)
        edge_start = start[self.edges]
        edge_end = end[self.edges]
        edge_min = jnp.minimum(edge_start, edge_end).min(axis=1)
        edge_max = jnp.maximum(edge_start, edge_end).max(axis=1)
        if self.faces.size:
            face_start = start[self.faces]
            face_end = end[self.faces]
            face_min = jnp.minimum(face_start, face_end).min(axis=1)
            face_max = jnp.maximum(face_start, face_end).max(axis=1)
        else:
            face_min = jnp.empty((0, start.shape[1]), dtype=start.dtype)
            face_max = jnp.empty((0, start.shape[1]), dtype=start.dtype)

        edge_vertex = _pack_compiled_pairs(
            ContactStencilKind.EDGE_VERTEX,
            self.edge_vertex_pairs,
            self.edge_vertex_capacity,
            point_min,
            point_max,
            edge_min,
            edge_max,
            self.edges,
            self.activation_distance,
        )
        edge_edge = _pack_compiled_same_pairs(
            ContactStencilKind.EDGE_EDGE,
            self.edge_edge_pairs,
            self.edge_edge_capacity,
            edge_min,
            edge_max,
            self.edges,
            self.activation_distance,
        )
        face_vertex = _pack_compiled_pairs(
            ContactStencilKind.FACE_VERTEX,
            self.face_vertex_pairs,
            self.face_vertex_capacity,
            point_min,
            point_max,
            face_min,
            face_max,
            self.faces,
            self.activation_distance,
        )
        count = (
            edge_vertex.actual_count + edge_edge.actual_count + face_vertex.actual_count
        )
        overflow = (
            edge_vertex.overflow_count
            + edge_edge.overflow_count
            + face_vertex.overflow_count
        )
        finite = jnp.all(jnp.isfinite(start)) & jnp.all(jnp.isfinite(end))
        evidence = CompiledContactSearchEvidence(
            count,
            overflow,
            finite,
            finite & (overflow == 0),
            self.plan_id,
        )
        return CompiledContactSearchResult(edge_vertex, edge_edge, face_vertex, evidence)


def _aabb_mask(first_min, first_max, second_min, second_max, radius):
    delta = jnp.maximum(
        0.0,
        jnp.maximum(first_min - second_max, second_min - first_max),
    )
    return jnp.sum(delta * delta, axis=-1) <= radius * radius


def _pack_indices(mask, capacity):
    count = jnp.sum(mask, dtype=jnp.int32)
    selected = jnp.nonzero(mask, size=capacity, fill_value=0)[0]
    valid = jnp.arange(capacity) < jnp.minimum(count, capacity)
    overflow = jnp.maximum(count - capacity, 0)
    return selected, valid, count, overflow


def _pack_compiled_pairs(
    kind,
    pairs,
    capacity,
    point_min,
    point_max,
    primitive_min,
    primitive_max,
    primitive_topology,
    radius,
):
    if pairs.shape[0] == 0:
        return CompiledCandidateBatch(
            jnp.full((capacity, 4), -1, dtype=jnp.int32),
            jnp.zeros((capacity,), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            kind,
            capacity,
        )
    point = pairs[:, 0]
    primitive = pairs[:, 1]
    mask = _aabb_mask(
        point_min[point],
        point_max[point],
        primitive_min[primitive],
        primitive_max[primitive],
        radius,
    )
    selected, valid, count, overflow = _pack_indices(mask, capacity)
    selected_pairs = pairs[selected]
    endpoints = primitive_topology[selected_pairs[:, 1]]
    padding = 4 - (1 + endpoints.shape[1])
    indices = jnp.concatenate(
        (
            selected_pairs[:, :1],
            endpoints,
            jnp.full((capacity, padding), -1, dtype=jnp.int32),
        ),
        axis=1,
    )
    return CompiledCandidateBatch(indices, valid, count, overflow, kind, capacity)


def _pack_compiled_same_pairs(
    kind,
    pairs,
    capacity,
    lower,
    upper,
    topology,
    radius,
):
    if pairs.shape[0] == 0:
        return CompiledCandidateBatch(
            jnp.full((capacity, 4), -1, dtype=jnp.int32),
            jnp.zeros((capacity,), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            kind,
            capacity,
        )
    first = pairs[:, 0]
    second = pairs[:, 1]
    mask = _aabb_mask(lower[first], upper[first], lower[second], upper[second], radius)
    selected, valid, count, overflow = _pack_indices(mask, capacity)
    selected_pairs = pairs[selected]
    indices = jnp.concatenate(
        (
            topology[selected_pairs[:, 0]],
            topology[selected_pairs[:, 1]],
        ),
        axis=1,
    )
    return CompiledCandidateBatch(indices, valid, count, overflow, kind, capacity)


__all__ = [
    "CompiledCandidateBatch",
    "CompiledContactSearchEvidence",
    "CompiledContactSearchPlan",
    "CompiledContactSearchResult",
]
