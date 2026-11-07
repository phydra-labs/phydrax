#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import mmap

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array
from scipy.sparse.csgraph import connected_components

from ..._strict import StrictModule
from ...discretization._topology import (
    CellComplexTopology,
    EntitySet,
    EntitySubset,
    OrientedIncidence,
)
from ...sparse import EdgeRelation
from ...typing import ConvertibleToArray


def _transfer_private_array(
    values: np.ndarray, *, device: jax.Device | None = None
) -> Array:
    """Transfer a sealed internal buffer; CPU backends may safely retain it."""
    values.setflags(write=False)
    return jax.device_put(values, device=device, may_alias=True)


def _has_duplicate_faces(faces: np.ndarray) -> bool:
    """Detect unoriented duplicates using one in-place sortable face buffer."""
    with mmap.mmap(-1, faces.size * np.dtype(np.int32).itemsize) as scratch:
        canonical = np.ndarray(faces.shape, dtype=np.int32, buffer=scratch)
        records = canonical.view(
            np.dtype([("a", np.int32), ("b", np.int32), ("c", np.int32)])
        ).reshape((-1,))
        try:
            canonical[:] = faces
            canonical.sort(axis=1)
            records.sort(kind="quicksort")
            # Bound the comparison scratch independently of the number of faces.
            for start in range(1, faces.shape[0], 65536):
                stop = min(start + 65536, faces.shape[0])
                if np.any(
                    np.all(
                        canonical[start:stop] == canonical[start - 1 : stop - 1], axis=1
                    )
                ):
                    return True
            return False
        finally:
            del records, canonical


def _csr(
    owners: np.ndarray, values: np.ndarray | None, count: int
) -> tuple[np.ndarray, np.ndarray]:
    if (
        len(owners) <= 65536
        or len(owners) > np.iinfo(np.int32).max
        or count * len(owners) > np.iinfo(np.int64).max
    ):
        order = np.argsort(owners, kind="stable")
        values_sorted = order.astype(np.int32) if values is None else values[order]
        counts = np.bincount(owners, minlength=count)
        offsets = np.empty((count + 1,), dtype=np.int32)
        offsets[0] = 0
        np.cumsum(counts, dtype=np.int32, out=offsets[1:])
        return offsets, values_sorted.astype(np.int32, copy=False)

    # A mixed-radix integer key orders each owner by original position. Numeric
    # sorting avoids structured-record comparison and preserves stable incidence.
    size = len(owners)
    with mmap.mmap(-1, size * np.dtype(np.int64).itemsize) as scratch:
        records = np.ndarray((size,), dtype=np.int64, buffer=scratch)
        block = None
        try:
            for start in range(0, size, 65536):
                stop = min(start + 65536, size)
                records[start:stop] = owners[start:stop].astype(
                    np.int64
                ) * size + np.arange(start, stop, dtype=np.int64)
            records.sort(kind="quicksort")
            offsets = np.full((count + 1,), len(owners), dtype=np.int32)
            values_sorted = np.empty(len(owners), dtype=np.int32)
            for start in range(0, len(owners), 65536):
                stop = min(start + 65536, len(owners))
                block = records[start:stop]
                block_owners = block // size
                starts = np.empty(stop - start, dtype=np.bool_)
                starts[0] = start == 0 or block_owners[0] != records[start - 1] // size
                starts[1:] = block_owners[1:] != block_owners[:-1]
                rows = np.flatnonzero(starts)
                offsets[block_owners[rows]] = rows + start
                indices = block % size
                values_sorted[start:stop] = indices if values is None else values[indices]
            np.minimum.accumulate(offsets[::-1], out=offsets[::-1])
            return offsets, values_sorted
        finally:
            del block, records


def _boundary_loops(
    starts: np.ndarray, ends: np.ndarray, vertex_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Trace boundary half-edges, given in increasing half-edge order, into loops.

    Loops are ordered by their smallest half-edge and list origins starting
    there, exactly as a traversal from the smallest remaining half-edge would.
    """

    count = starts.size
    if count == 0:
        return np.empty(0, dtype=np.int32), np.zeros(1, dtype=np.int32)
    outgoing = np.bincount(starts, minlength=vertex_count)
    incoming = np.bincount(ends, minlength=vertex_count)
    if np.any(outgoing > 1) or np.any(incoming > 1):
        raise ValueError("Boundary is non-manifold at a vertex.")
    if np.any(outgoing != incoming):
        raise ValueError("Boundary half-edges do not form closed loops.")
    # Every boundary vertex has one outgoing and one incoming half-edge, so the
    # successor map is a permutation whose cycles are the loops.
    position = np.empty((vertex_count,), dtype=np.int64)
    position[starts] = np.arange(count, dtype=np.int64)
    successor = position[ends]
    indices = np.arange(count, dtype=np.int64)
    # Pointer doubling: after k rounds `root` is the minimum over 2**k
    # successors, so bit_length(count) rounds reach each loop's minimum.
    root = indices.copy()
    jump = successor.copy()
    for _ in range(count.bit_length()):
        root = np.minimum(root, root[jump])
        jump = jump[jump]
    # List ranking toward each root yields the forward distance from the root.
    is_root = root == indices
    steps = np.where(is_root, 0, 1)
    jump = np.where(is_root, indices, successor)
    for _ in range(count.bit_length()):
        steps = steps + steps[jump]
        jump = jump[jump]
    lengths = np.bincount(root, minlength=count)
    order = np.lexsort(((lengths[root] - steps) % lengths[root], root))
    offsets = np.zeros((np.count_nonzero(is_root) + 1,), dtype=np.int32)
    np.cumsum(lengths[is_root], out=offsets[1:])
    return starts[order].astype(np.int32), offsets


def _face_components(
    first_faces: np.ndarray, second_faces: np.ndarray, face_count: int
) -> tuple[np.ndarray, int]:
    """Label edge-connected faces with components ordered by smallest face."""

    adjacency = sp.csr_matrix(
        (np.ones((first_faces.size,), dtype=np.int8), (first_faces, second_faces)),
        shape=(face_count, face_count),
    )
    component_count, labels = connected_components(adjacency, directed=False)
    if component_count == 1:
        # A connected graph already has canonical label zero at every face.
        return labels.astype(np.int32, copy=False), component_count
    _, smallest = np.unique(labels, return_index=True)
    relabel = np.empty((component_count,), dtype=np.int32)
    relabel[np.argsort(smallest)] = np.arange(component_count, dtype=np.int32)
    return relabel[labels], component_count


class SegmentTopology(StrictModule):
    """Validated unoriented segment connectivity with vertex incidence CSR."""

    edges: Array
    vertex_edge_offsets: Array
    vertex_edges: Array
    num_vertices: int = eqx.field(static=True)

    def __init__(
        self, edges: ConvertibleToArray, *, num_vertices: int | None = None
    ) -> None:
        edges_host = np.asarray(edges, dtype=np.int32)
        if edges_host.ndim != 2 or edges_host.shape[1] != 2 or edges_host.shape[0] == 0:
            raise ValueError("edges must have shape (num_edges > 0, 2).")
        if np.any(edges_host < 0) or np.any(edges_host[:, 0] == edges_host[:, 1]):
            raise ValueError("Segment edges require distinct non-negative vertices.")
        inferred = int(np.max(edges_host)) + 1
        count = inferred if num_vertices is None else int(num_vertices)
        if count < inferred:
            raise ValueError("num_vertices does not cover every edge index.")
        canonical = np.sort(edges_host, axis=1)
        if np.unique(canonical, axis=0).shape[0] != edges_host.shape[0]:
            raise ValueError("SegmentTopology contains duplicate edges.")
        owners = edges_host.reshape((-1,))
        values = np.repeat(np.arange(edges_host.shape[0], dtype=np.int32), 2)
        offsets, vertex_edges = _csr(owners, values, count)
        self.edges = jnp.asarray(edges_host, dtype=jnp.int32)
        self.vertex_edge_offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.vertex_edges = jnp.asarray(vertex_edges, dtype=jnp.int32)
        self.num_vertices = count

    @property
    def num_edges(self) -> int:
        return self.edges.shape[0]

    @property
    def vertex_degree(self) -> Array:
        return self.vertex_edge_offsets[1:] - self.vertex_edge_offsets[:-1]

    def cell_complex_topology(self, /) -> CellComplexTopology:
        """Return the canonical oriented one-complex view."""
        edges = np.asarray(self.edges, dtype=np.int32)
        boundary_vertices = np.zeros((self.num_vertices,), dtype=np.bool_)
        degree = np.bincount(edges.reshape((-1,)), minlength=self.num_vertices)
        boundary_vertices[degree == 1] = True
        vertices = EntitySet(
            "vertices",
            0,
            np.arange(self.num_vertices, dtype=np.int32),
            subsets=(EntitySubset("boundary", boundary_vertices),),
        )
        edge_entities = EntitySet(
            "edges",
            1,
            np.arange(self.num_edges, dtype=np.int32),
            subsets=(
                EntitySubset(
                    "boundary",
                    np.zeros((self.num_edges,), dtype=np.bool_),
                ),
            ),
        )
        relation = EdgeRelation(
            edges.reshape((-1,)),
            np.repeat(np.arange(self.num_edges, dtype=np.int32), 2),
            source_size=self.num_vertices,
            target_size=self.num_edges,
        )
        signs = np.tile(np.asarray([-1.0, 1.0]), self.num_edges)
        return CellComplexTopology(
            (vertices, edge_entities),
            (OrientedIncidence(1, vertices, edge_entities, relation, signs),),
        )


class TriangleTopology(StrictModule):
    """Canonical oriented half-edge topology for a triangular two-complex."""

    faces: Array
    edges: Array
    halfedge_origin: Array
    halfedge_destination: Array
    halfedge_face: Array
    halfedge_next: Array
    halfedge_previous: Array
    halfedge_twin: Array
    halfedge_edge: Array
    edge_halfedges: Array
    boundary_halfedges: Array
    boundary_loop_vertices: Array
    boundary_loop_offsets: Array
    vertex_face_offsets: Array
    vertex_faces: Array
    vertex_halfedge_offsets: Array
    vertex_halfedges: Array
    face_component_ids: Array
    num_face_components: int = eqx.field(static=True)
    num_vertices: int = eqx.field(static=True)
    watertight: bool = eqx.field(static=True)

    def __init__(
        self, faces: Array | np.ndarray, *, num_vertices: int | None = None
    ) -> None:
        faces_host = np.asarray(faces, dtype=np.int32)
        if faces_host.ndim != 2 or faces_host.shape[1] != 3 or faces_host.shape[0] == 0:
            raise ValueError("faces must have shape (num_faces > 0, 3).")
        if np.any(faces_host < 0):
            raise ValueError("faces must contain non-negative indices.")
        if np.any(
            (faces_host[:, 0] == faces_host[:, 1])
            | (faces_host[:, 1] == faces_host[:, 2])
            | (faces_host[:, 2] == faces_host[:, 0])
        ):
            raise ValueError("Every face must reference three distinct vertices.")
        inferred = int(np.max(faces_host)) + 1
        vertex_count = inferred if num_vertices is None else int(num_vertices)
        if vertex_count < inferred:
            raise ValueError("num_vertices does not cover every face index.")
        if _has_duplicate_faces(faces_host):
            raise ValueError("TriangleTopology contains duplicate faces.")

        face_count = faces_host.shape[0]
        halfedge_count = 3 * face_count
        origin = faces_host.reshape((-1,))
        destination = faces_host[:, [1, 2, 0]].reshape((-1,))
        halfedge_face = np.repeat(np.arange(face_count, dtype=np.int32), 3)

        # Sort (undirected key, half-edge ID), preserving the stable ordering.
        # Anonymous mapped scratch is released as soon as incidence is built,
        # rather than remaining in the host allocator during source replay.
        index_dtype = np.int32 if halfedge_count <= np.iinfo(np.int32).max else np.int64
        record_dtype = np.dtype([("key", np.int64), ("halfedge", index_dtype)])
        scratch = mmap.mmap(-1, halfedge_count * record_dtype.itemsize)
        records = np.ndarray((halfedge_count,), dtype=record_dtype, buffer=scratch)
        for start in range(0, halfedge_count, 65536):
            stop = min(start + 65536, halfedge_count)
            a, b = origin[start:stop], destination[start:stop]
            records["key"][start:stop] = np.minimum(a, b).astype(
                np.int64
            ) * vertex_count + np.maximum(a, b)
            records["halfedge"][start:stop] = np.arange(start, stop, dtype=index_dtype)
        records.sort(order=("key", "halfedge"), kind="quicksort")
        keys = records["key"]
        order = records["halfedge"]
        starts = np.ones((halfedge_count,), dtype=np.bool_)
        for start in range(1, halfedge_count, 65536):
            stop = min(start + 65536, halfedge_count)
            current = order[start:stop]
            previous_order = order[start - 1 : stop - 1]
            current_a, current_b = origin[current], destination[current]
            previous_a, previous_b = origin[previous_order], destination[previous_order]
            starts[start:stop] = (
                np.minimum(current_a, current_b) != np.minimum(previous_a, previous_b)
            ) | (np.maximum(current_a, current_b) != np.maximum(previous_a, previous_b))
        del current, previous_order, current_a, current_b, previous_a, previous_b
        first = np.empty(np.count_nonzero(starts), dtype=order.dtype)
        position = 0
        for start in range(0, halfedge_count, 65536):
            stop = min(start + 65536, halfedge_count)
            rows = np.flatnonzero(starts[start:stop])
            first[position : position + len(rows)] = rows + start
            position += len(rows)
        paired = np.empty(len(first), dtype=np.bool_)
        for start in range(0, len(first), 65536):
            stop = min(start + 65536, len(first))
            next_rows = np.empty(stop - start, dtype=first.dtype)
            if stop < len(first):
                next_rows[:] = first[start + 1 : stop + 1]
            else:
                next_rows[:-1] = first[start + 1 : stop]
                next_rows[-1] = halfedge_count
            sizes = next_rows - first[start:stop]
            if np.any(sizes > 2):
                raise ValueError(
                    "TriangleTopology is non-manifold: an edge has more than two incident faces."
                )
            paired[start:stop] = sizes == 2
        edges = np.empty((first.size, 2), dtype=np.int32)
        for start in range(0, len(first), 65536):
            stop = min(start + 65536, len(first))
            selected = order[first[start:stop]]
            first_vertices = origin[selected]
            second_vertices = destination[selected]
            edges[start:stop, 0] = np.minimum(first_vertices, second_vertices)
            edges[start:stop, 1] = np.maximum(first_vertices, second_vertices)
        del keys
        halfedge_edge = np.empty((halfedge_count,), dtype=np.int32)
        previous = 0
        for start in range(0, halfedge_count, 65536):
            stop = min(start + 65536, halfedge_count)
            cumulative = np.cumsum(starts[start:stop], dtype=np.int32)
            cumulative += previous
            previous = int(cumulative[-1])
            halfedge_edge[order[start:stop]] = cumulative - 1
        del starts
        del cumulative, next_rows, sizes, rows, selected, first_vertices, second_vertices
        first_halfedges = order[first[paired]]
        second_halfedges = order[first[paired] + 1]
        for start in range(0, len(first_halfedges), 65536):
            stop = min(start + 65536, len(first_halfedges))
            a, b = first_halfedges[start:stop], second_halfedges[start:stop]
            if np.any((origin[a] == origin[b]) | (destination[a] == destination[b])):
                raise ValueError(
                    "Adjacent faces have inconsistent orientation across an edge."
                )
        edge_halfedges = np.full((first.size, 2), -1, dtype=np.int32)
        for start in range(0, len(first), 65536):
            stop = min(start + 65536, len(first))
            edge_halfedges[start:stop, 0] = order[first[start:stop]]
        edge_halfedges[paired, 1] = second_halfedges
        del order, records, first, paired
        scratch.close()
        halfedge_twin = np.full((halfedge_count,), -1, dtype=np.int32)
        halfedge_twin[first_halfedges] = second_halfedges
        halfedge_twin[second_halfedges] = first_halfedges

        boundary_halfedges = np.flatnonzero(halfedge_twin < 0).astype(np.int32)
        loop_vertices, loop_offsets = _boundary_loops(
            origin[boundary_halfedges], destination[boundary_halfedges], vertex_count
        )
        # Twin construction consumed these private half-edge IDs. Reuse their
        # storage for face IDs rather than gathering two more incidence banks.
        first_faces, second_faces = first_halfedges, second_halfedges
        del first_halfedges, second_halfedges
        np.floor_divide(first_faces, 3, out=first_faces)
        np.floor_divide(second_faces, 3, out=second_faces)
        face_component_ids, face_component_count = _face_components(
            first_faces, second_faces, face_count
        )
        del first_faces, second_faces
        vertex_halfedge_offsets, vertex_halfedges = _csr(origin, None, vertex_count)
        # Vertex-face incidence is the face of each vertex-origin half-edge.
        vertex_face_offsets = vertex_halfedge_offsets
        vertex_faces = halfedge_face[vertex_halfedges]
        halfedge_next = np.empty(halfedge_count, dtype=np.int32).reshape(-1, 3)
        halfedge_previous = np.empty_like(halfedge_next)
        for start in range(0, face_count, 65536):
            stop = min(start + 65536, face_count)
            base = np.arange(start, stop, dtype=np.int32) * 3
            halfedge_next[start:stop, 0] = base + 1
            halfedge_next[start:stop, 1] = base + 2
            halfedge_next[start:stop, 2] = base
            halfedge_previous[start:stop, 0] = base + 2
            halfedge_previous[start:stop, 1] = base
            halfedge_previous[start:stop, 2] = base + 1
        halfedge_next = halfedge_next.reshape(-1)
        halfedge_previous = halfedge_previous.reshape(-1)

        self.faces = jnp.asarray(faces, dtype=jnp.int32)
        self.edges = _transfer_private_array(edges)
        del edges
        device = (
            next(iter(self.faces.sharding.device_set))
            if isinstance(self.faces.sharding, jax.sharding.SingleDeviceSharding)
            else None
        )
        if device is not None and device.platform == "cpu":
            # Borrow the immutable JAX-owned buffer, never a mutable caller array.
            self.halfedge_origin = _transfer_private_array(
                np.asarray(self.faces).reshape(-1),
                device=device if self.faces.committed else None,
            )
        else:
            self.halfedge_origin = self.faces.reshape(-1)
        del origin
        self.halfedge_destination = _transfer_private_array(destination)
        del destination
        self.halfedge_face = _transfer_private_array(halfedge_face)
        del halfedge_face
        self.halfedge_next = _transfer_private_array(halfedge_next)
        del halfedge_next
        self.halfedge_previous = _transfer_private_array(halfedge_previous)
        del halfedge_previous
        self.halfedge_twin = _transfer_private_array(halfedge_twin)
        del halfedge_twin
        self.halfedge_edge = _transfer_private_array(halfedge_edge)
        del halfedge_edge
        self.edge_halfedges = _transfer_private_array(edge_halfedges)
        del edge_halfedges
        self.boundary_halfedges = _transfer_private_array(boundary_halfedges)
        self.boundary_loop_vertices = _transfer_private_array(loop_vertices)
        self.boundary_loop_offsets = _transfer_private_array(loop_offsets)
        self.vertex_halfedge_offsets = _transfer_private_array(vertex_halfedge_offsets)
        self.vertex_face_offsets = self.vertex_halfedge_offsets
        del vertex_face_offsets, vertex_halfedge_offsets
        self.vertex_faces = _transfer_private_array(vertex_faces)
        del vertex_faces
        self.vertex_halfedges = _transfer_private_array(vertex_halfedges)
        del vertex_halfedges
        self.face_component_ids = _transfer_private_array(face_component_ids)
        self.num_face_components = face_component_count
        self.num_vertices = vertex_count
        self.watertight = boundary_halfedges.size == 0

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edges.shape[0]

    @property
    def num_halfedges(self) -> int:
        return self.halfedge_origin.shape[0]

    @property
    def num_boundary_loops(self) -> int:
        return self.boundary_loop_offsets.shape[0] - 1

    def cell_complex_topology(self, /) -> CellComplexTopology:
        """Return the canonical oriented two-complex view."""
        edges = np.asarray(self.edges, dtype=np.int32)
        faces = np.asarray(self.faces, dtype=np.int32)
        halfedge_edges = np.asarray(self.halfedge_edge, dtype=np.int32).reshape((-1, 3))
        origin = faces.reshape((-1,))
        destination = faces[:, [1, 2, 0]].reshape((-1,))
        selected_edges = edges[halfedge_edges.reshape((-1,))]
        face_signs = np.where(
            (selected_edges[:, 0] == origin) & (selected_edges[:, 1] == destination),
            1.0,
            -1.0,
        )
        boundary_edges = np.asarray(self.edge_halfedges)[:, 1] < 0
        boundary_vertices = np.zeros((self.num_vertices,), dtype=np.bool_)
        boundary_vertices[np.unique(edges[boundary_edges].reshape((-1,)))] = True
        vertices = EntitySet(
            "vertices",
            0,
            np.arange(self.num_vertices, dtype=np.int32),
            subsets=(EntitySubset("boundary", boundary_vertices),),
        )
        edge_entities = EntitySet(
            "edges",
            1,
            np.arange(self.num_edges, dtype=np.int32),
            subsets=(EntitySubset("boundary", boundary_edges),),
        )
        face_entities = EntitySet(
            "faces",
            2,
            np.arange(self.num_faces, dtype=np.int32),
            subsets=(
                EntitySubset(
                    "boundary",
                    np.zeros((self.num_faces,), dtype=np.bool_),
                ),
            ),
        )
        vertex_edge_relation = EdgeRelation(
            edges.reshape((-1,)),
            np.repeat(np.arange(self.num_edges, dtype=np.int32), 2),
            source_size=self.num_vertices,
            target_size=self.num_edges,
        )
        edge_face_relation = EdgeRelation(
            halfedge_edges.reshape((-1,)),
            np.repeat(np.arange(self.num_faces, dtype=np.int32), 3),
            source_size=self.num_edges,
            target_size=self.num_faces,
        )
        return CellComplexTopology(
            (vertices, edge_entities, face_entities),
            (
                OrientedIncidence(
                    1,
                    vertices,
                    edge_entities,
                    vertex_edge_relation,
                    np.tile(np.asarray([-1.0, 1.0]), self.num_edges),
                ),
                OrientedIncidence(
                    2,
                    edge_entities,
                    face_entities,
                    edge_face_relation,
                    face_signs,
                ),
            ),
        )

    @property
    def euler_characteristic(self) -> int:
        return self.num_vertices - self.num_edges + self.num_faces


__all__ = ["SegmentTopology", "TriangleTopology"]
