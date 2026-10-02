#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity combinatorial topology of a labeled multiregion surface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import nonnegative_integer, positive_integer, unique_identifiers
from ...typing import Bool, checked, Dim, Identifier, Int64, Integer, parse, Size
from ._contracts import (
    MultiRegionDomain,
    MultiRegionKind,
    MultiRegionSurfaceCapacityError,
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceCounts,
)


class _VertexDim(Dim, minimum=1):
    """Vertex slots of one multiregion surface."""


class _EdgeDim(Dim, minimum=1):
    """Edge slots of one multiregion surface."""


class _FaceDim(Dim, minimum=1):
    """Face slots of one multiregion surface."""


class _RegionDim(Dim, minimum=2):
    """Region (label) slots of one multiregion surface."""


class _PairDim(Dim, minimum=1):
    """Region-pair (sheet) slots of one multiregion surface."""


class _ValenceDim(Dim, minimum=1):
    """Incident-face slots of one edge."""


class _SlotDim(Dim, minimum=1):
    """Region-pair slots of one vertex."""


@dataclass(frozen=True, slots=True)
class _HostIncidence:
    """Canonical host incidence of the active entities (no padding)."""

    edges: np.ndarray
    face_edges: np.ndarray
    face_edge_signs: np.ndarray
    valence: np.ndarray
    edge_faces: np.ndarray
    edge_face_signs: np.ndarray
    region_pairs: np.ndarray
    face_pairs: np.ndarray
    face_pair_signs: np.ndarray
    vertex_slot_count: np.ndarray
    vertex_pair_slots: np.ndarray
    face_corner_slots: np.ndarray


def _edge_incidence(
    inverse: np.ndarray, face_edge_signs: np.ndarray, edge_count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valence and face-ordered ``(edge, incident face)`` rows with traversal signs."""
    valence = np.bincount(inverse, minlength=edge_count)
    width = int(np.max(valence))
    half_face = np.repeat(np.arange(inverse.size // 3), 3)
    order = np.lexsort((half_face, inverse))
    sorted_edges = inverse[order]
    starts = np.concatenate(([0], np.cumsum(valence)[:-1]))
    rank = np.arange(order.size) - starts[sorted_edges]
    edge_faces = np.full((edge_count, width), -1, dtype=np.int64)
    edge_face_signs = np.zeros((edge_count, width), dtype=np.int64)
    edge_faces[sorted_edges, rank] = half_face[order]
    edge_face_signs[sorted_edges, rank] = face_edge_signs.reshape(-1)[order]
    return valence, edge_faces, edge_face_signs


def _vertex_slots(
    origin: np.ndarray, face_pairs: np.ndarray, vertex_count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-vertex ``(vertex, region-pair)`` slot table sorted by pair."""
    corner_keys = np.stack((origin, np.repeat(face_pairs, 3)), axis=1)
    slots, slot_inverse = np.unique(corner_keys, axis=0, return_inverse=True)
    slot_inverse = slot_inverse.reshape(-1)
    slot_vertices = slots[:, 0]
    counts = np.bincount(slot_vertices, minlength=vertex_count)
    slot_starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    slot_rank = np.arange(slots.shape[0]) - slot_starts[slot_vertices]
    table = np.full((vertex_count, int(np.max(counts))), -1, dtype=np.int64)
    table[slot_vertices, slot_rank] = slots[:, 1]
    return counts, table, slot_rank[slot_inverse].reshape((-1, 3))


def _host_incidence(
    faces: np.ndarray, labels: np.ndarray, vertex_count: int, /
) -> _HostIncidence:
    origin = faces.reshape(-1)
    destination = np.roll(faces, -1, axis=1).reshape(-1)
    keys = np.stack(
        (np.minimum(origin, destination), np.maximum(origin, destination)), axis=1
    )
    edges, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    face_edge_signs = np.where(origin < destination, 1, -1).reshape((-1, 3))
    valence, edge_faces, edge_face_signs = _edge_incidence(
        inverse, face_edge_signs, edges.shape[0]
    )
    canonical = np.sort(labels, axis=1)
    region_pairs, pair_inverse = np.unique(canonical, axis=0, return_inverse=True)
    face_pairs = pair_inverse.reshape(-1)
    slot_count, slot_table, corner_slots = _vertex_slots(origin, face_pairs, vertex_count)
    return _HostIncidence(
        edges=edges,
        face_edges=inverse.reshape((-1, 3)),
        face_edge_signs=face_edge_signs,
        valence=valence,
        edge_faces=edge_faces,
        edge_face_signs=edge_face_signs,
        region_pairs=region_pairs,
        face_pairs=face_pairs,
        face_pair_signs=np.where(labels[:, 0] == canonical[:, 0], 1, -1),
        vertex_slot_count=slot_count,
        vertex_pair_slots=slot_table,
        face_corner_slots=corner_slots,
    )


def _host_faces(faces: ArrayLike, vertex_count: int, /) -> np.ndarray:
    rows = np.asarray(faces)
    if rows.ndim != 2 or rows.shape[1] != 3 or rows.shape[0] < 1:
        raise ValueError("faces must have shape (face_count >= 1, 3).")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("faces must be an integer array.")
    rows = rows.astype(np.int64)
    if np.any(rows < 0) or np.any(rows >= vertex_count):
        raise ValueError("faces reference vertices outside [0, vertex_count).")
    if (
        np.any(rows[:, 0] == rows[:, 1])
        or np.any(rows[:, 1] == rows[:, 2])
        or np.any(rows[:, 0] == rows[:, 2])
    ):
        raise ValueError("Every face must reference three distinct vertices.")
    if np.unique(np.sort(rows, axis=1), axis=0).shape[0] != rows.shape[0]:
        raise ValueError("faces contains duplicate triangles.")
    if np.unique(rows).size != vertex_count:
        raise ValueError("Every active vertex must belong to at least one face.")
    return rows


def _host_labels(labels: ArrayLike, face_count: int, region_count: int, /) -> np.ndarray:
    rows = np.asarray(labels)
    if rows.shape != (face_count, 2):
        raise ValueError("face_labels must have shape (face_count, 2).")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("face_labels must be an integer array.")
    rows = rows.astype(np.int64)
    if np.any(rows < 0) or np.any(rows >= region_count):
        raise ValueError("face_labels reference regions outside the region table.")
    if np.any(rows[:, 0] == rows[:, 1]):
        raise ValueError("Every face must separate two distinct regions.")
    if np.unique(rows).size != region_count:
        raise ValueError("Every declared region must bound at least one face.")
    return rows


def _host_ids(values: ArrayLike | None, count: int, name: str, /) -> np.ndarray:
    if values is None:
        return np.arange(count, dtype=np.int64)
    ids = np.asarray(values)
    if ids.shape != (count,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array of shape ({count},).")
    ids = ids.astype(np.int64)
    if np.any(ids < 0) or np.unique(ids).size != count:
        raise ValueError(f"{name} must be unique and nonnegative.")
    return ids


def _padded(values: np.ndarray, rows: int, fill: int, dtype: DTypeLike, /) -> np.ndarray:
    shape = (rows,) + values.shape[1:]
    result = np.full(shape, fill, dtype=dtype)
    result[: values.shape[0]] = values
    return result


def _padded_width(values: np.ndarray, width: int, fill: int, /) -> np.ndarray:
    result = np.full((values.shape[0], width), fill, dtype=values.dtype)
    result[:, : values.shape[1]] = values
    return result


def _mask(count: int, capacity: int, /) -> np.ndarray:
    return np.arange(capacity) < count


@final
class MultiRegionSurfaceTopology(StrictModule, NonTrainableState):
    """Padded combinatorial topology of one labeled multiregion surface.

    Active entities occupy the leading slots; padding rows hold ``-1`` indices
    and ``False`` masks. ``face_labels[f] = (left, right)`` index the region
    table ``region_ids``; the face normal points out of ``left``. Edges are the
    lexicographically sorted undirected vertex pairs ``(v0 < v1)``;
    ``edge_faces`` lists incident faces in face order with ``edge_face_signs``
    ``+1`` when the face traverses ``v0 -> v1``. ``region_pairs`` are the
    canonical ``(min, max)`` label pairs; ``face_pair_signs`` is ``+1`` when the
    face normal points out of the pair's first label. ``vertex_pair_slots[v, s]``
    is the region pair of sheet slot ``s`` at vertex ``v`` (sorted by pair) and
    ``face_corner_slots[f, i]`` is the slot of corner ``i`` of face ``f``; these
    ``(vertex, region-pair)`` slots carry sheet fields. Stable host identities
    are ``vertex_global_ids``/``face_global_ids`` and the region identifiers.
    """

    __strict_contract__ = True

    plan: MultiRegionSurfaceCapacityPlan
    vertex_capacity: Size[_VertexDim] = eqx.field(static=True)
    edge_capacity: Size[_EdgeDim] = eqx.field(static=True)
    face_capacity: Size[_FaceDim] = eqx.field(static=True)
    region_capacity: Size[_RegionDim] = eqx.field(static=True)
    region_pair_capacity: Size[_PairDim] = eqx.field(static=True)
    valence_width: Size[_ValenceDim] = eqx.field(static=True)
    slot_width: Size[_SlotDim] = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    region_count: int = eqx.field(static=True)
    region_pair_count: int = eqx.field(static=True)
    faces: Integer[_FaceDim, Literal[3]]
    face_labels: Integer[_FaceDim, Literal[2]]
    face_edges: Integer[_FaceDim, Literal[3]]
    face_edge_signs: Integer[_FaceDim, Literal[3]]
    face_pairs: Integer[_FaceDim]
    face_pair_signs: Integer[_FaceDim]
    face_corner_slots: Integer[_FaceDim, Literal[3]]
    edges: Integer[_EdgeDim, Literal[2]]
    edge_faces: Integer[_EdgeDim, _ValenceDim]
    edge_face_signs: Integer[_EdgeDim, _ValenceDim]
    region_pairs: Integer[_PairDim, Literal[2]]
    vertex_pair_slots: Integer[_VertexDim, _SlotDim]
    vertex_active: Bool[_VertexDim]
    edge_active: Bool[_EdgeDim]
    face_active: Bool[_FaceDim]
    region_active: Bool[_RegionDim]
    region_finite: Bool[_RegionDim]
    pair_active: Bool[_PairDim]
    slot_active: Bool[_VertexDim, _SlotDim]
    vertex_global_ids: Int64[_VertexDim]
    face_global_ids: Int64[_FaceDim]
    region_ids: tuple[str, ...] = eqx.field(static=True)
    region_kinds: tuple[MultiRegionKind, ...] = eqx.field(static=True)
    domain: MultiRegionDomain = eqx.field(static=True)
    epoch: int = eqx.field(static=True)
    topology_id: Identifier = eqx.field(static=True)
    lineage_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: MultiRegionSurfaceCapacityPlan,
        faces: ArrayLike,
        face_labels: ArrayLike,
        region_ids: Sequence[str],
        region_kinds: Sequence[MultiRegionKind],
        /,
        *,
        vertex_count: int,
        vertex_global_ids: ArrayLike | None = None,
        face_global_ids: ArrayLike | None = None,
        epoch: int = 0,
        domain: MultiRegionDomain = "free_space",
    ) -> None:
        domain_ = parse(domain, MultiRegionDomain, "domain")
        if domain_ == "periodic":
            raise ValueError(
                "Periodic multiregion surfaces are refused: signed region volumes "
                "require unwrapped periodic coordinates, which are not represented."
            )
        vertices = positive_integer(vertex_count, "vertex_count")
        epoch_ = nonnegative_integer(epoch, "epoch")
        ids = unique_identifiers(region_ids, "region_ids")
        if isinstance(region_kinds, str) or len(region_kinds) != len(ids):
            raise ValueError("region_kinds must give one kind per region id.")
        kinds = tuple(
            parse(kind, MultiRegionKind, f"region_kinds[{index}]")
            for index, kind in enumerate(region_kinds)
        )
        if len(ids) < 2:
            raise ValueError("A multiregion surface needs at least two regions.")
        rows = _host_faces(faces, vertices)
        labels = _host_labels(face_labels, rows.shape[0], len(ids))
        vertex_ids = _host_ids(vertex_global_ids, vertices, "vertex_global_ids")
        face_ids = _host_ids(face_global_ids, rows.shape[0], "face_global_ids")
        host = _host_incidence(rows, labels, vertices)
        counts = MultiRegionSurfaceCounts(
            vertex=vertices,
            edge=host.edges.shape[0],
            face=rows.shape[0],
            region=len(ids),
            region_pair=host.region_pairs.shape[0],
            edge_valence=int(np.max(host.valence)),
            vertex_region_pairs=int(np.max(host.vertex_slot_count)),
        )
        capacity = plan.capacity_evidence(counts)
        if not capacity.admitted:
            raise MultiRegionSurfaceCapacityError(capacity)
        index = np.dtype(plan.index_dtype)
        vcap, ecap, fcap = plan.vertex_capacity, plan.edge_capacity, plan.face_capacity
        rcap, pcap = plan.region_capacity, plan.region_pair_capacity
        width, slot_width = plan.maximum_edge_valence, plan.maximum_vertex_region_pairs
        edge_faces = _padded_width(host.edge_faces, width, -1)
        edge_face_signs = _padded_width(host.edge_face_signs, width, 0)
        vertex_slots = _padded_width(host.vertex_pair_slots, slot_width, -1)
        finite = np.zeros((rcap,), dtype=np.bool_)
        finite[: len(ids)] = np.asarray([kind == "finite" for kind in kinds])
        payload = {
            "faces": _padded(rows, fcap, -1, index),
            "face_labels": _padded(labels, fcap, -1, index),
            "face_edges": _padded(host.face_edges, fcap, -1, index),
            "face_edge_signs": _padded(host.face_edge_signs, fcap, 0, index),
            "face_pairs": _padded(host.face_pairs, fcap, -1, index),
            "face_pair_signs": _padded(host.face_pair_signs, fcap, 0, index),
            "face_corner_slots": _padded(host.face_corner_slots, fcap, -1, index),
            "edges": _padded(host.edges, ecap, -1, index),
            "edge_faces": _padded(edge_faces, ecap, -1, index),
            "edge_face_signs": _padded(edge_face_signs, ecap, 0, index),
            "region_pairs": _padded(host.region_pairs, pcap, -1, index),
            "vertex_pair_slots": _padded(vertex_slots, vcap, -1, index),
        }
        self.plan = plan
        self.vertex_capacity = vcap
        self.edge_capacity = ecap
        self.face_capacity = fcap
        self.region_capacity = rcap
        self.region_pair_capacity = pcap
        self.valence_width = width
        self.slot_width = slot_width
        self.vertex_count = vertices
        self.edge_count = counts.edge
        self.face_count = counts.face
        self.region_count = counts.region
        self.region_pair_count = counts.region_pair
        self.faces = jnp.asarray(payload["faces"])
        self.face_labels = jnp.asarray(payload["face_labels"])
        self.face_edges = jnp.asarray(payload["face_edges"])
        self.face_edge_signs = jnp.asarray(payload["face_edge_signs"])
        self.face_pairs = jnp.asarray(payload["face_pairs"])
        self.face_pair_signs = jnp.asarray(payload["face_pair_signs"])
        self.face_corner_slots = jnp.asarray(payload["face_corner_slots"])
        self.edges = jnp.asarray(payload["edges"])
        self.edge_faces = jnp.asarray(payload["edge_faces"])
        self.edge_face_signs = jnp.asarray(payload["edge_face_signs"])
        self.region_pairs = jnp.asarray(payload["region_pairs"])
        self.vertex_pair_slots = jnp.asarray(payload["vertex_pair_slots"])
        self.vertex_active = jnp.asarray(_mask(vertices, vcap))
        self.edge_active = jnp.asarray(_mask(counts.edge, ecap))
        self.face_active = jnp.asarray(_mask(counts.face, fcap))
        self.region_active = jnp.asarray(_mask(counts.region, rcap))
        self.region_finite = jnp.asarray(finite)
        self.pair_active = jnp.asarray(_mask(counts.region_pair, pcap))
        self.slot_active = jnp.asarray(payload["vertex_pair_slots"] >= 0)
        self.vertex_global_ids = jnp.asarray(_padded(vertex_ids, vcap, -1, np.int64))
        self.face_global_ids = jnp.asarray(_padded(face_ids, fcap, -1, np.int64))
        self.region_ids = ids
        self.region_kinds = kinds
        self.domain = domain_
        self.epoch = epoch_
        self.topology_id = canonical_fingerprint(
            {
                "kind": "multiregion-surface-topology",
                "plan": plan.plan_id,
                "faces": array_tree_fingerprint(rows),
                "face_labels": array_tree_fingerprint(labels),
                "region_ids": list(ids),
                "region_kinds": list(kinds),
                "vertex_count": vertices,
                "domain": domain_,
            }
        )
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "multiregion-surface-lineage",
                "topology": self.topology_id,
                "epoch": epoch_,
                "vertex_global_ids": array_tree_fingerprint(vertex_ids),
                "face_global_ids": array_tree_fingerprint(face_ids),
            }
        )

    @property
    def counts(self) -> MultiRegionSurfaceCounts:
        """Exact active counts and valence maxima of this topology."""
        valence = np.asarray(self.edge_faces[: self.edge_count]) >= 0
        slots = np.asarray(self.slot_active[: self.vertex_count])
        return MultiRegionSurfaceCounts(
            vertex=self.vertex_count,
            edge=self.edge_count,
            face=self.face_count,
            region=self.region_count,
            region_pair=self.region_pair_count,
            edge_valence=int(np.max(np.sum(valence, axis=1))),
            vertex_region_pairs=int(np.max(np.sum(slots, axis=1))),
        )

    @property
    def finite_region_indices(self) -> tuple[int, ...]:
        """Region-table indices of the finite regions (3-cells), in table order."""
        return tuple(
            index for index, kind in enumerate(self.region_kinds) if kind == "finite"
        )

    @property
    def boundary_region_indices(self) -> tuple[int, ...]:
        """Region-table indices of boundary labels, in table order."""
        return tuple(
            index for index, kind in enumerate(self.region_kinds) if kind == "boundary"
        )

    @property
    def finite_region_ids(self) -> tuple[str, ...]:
        return tuple(self.region_ids[index] for index in self.finite_region_indices)

    def region_index(self, region_id: str, /) -> int:
        """Region-table index of one declared region identifier."""
        if region_id not in self.region_ids:
            raise ValueError(f"Unknown region id {region_id!r}.")
        return self.region_ids.index(region_id)

    def host_faces(self) -> np.ndarray:
        """Active face vertex triples as host int64 (host preparation boundary)."""
        return np.asarray(self.faces[: self.face_count], dtype=np.int64)

    def host_face_labels(self) -> np.ndarray:
        """Active ``(left, right)`` labels as host int64."""
        return np.asarray(self.face_labels[: self.face_count], dtype=np.int64)


__all__ = ["MultiRegionSurfaceTopology"]
