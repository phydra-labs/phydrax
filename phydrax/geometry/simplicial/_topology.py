#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
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


def _csr(
    owners: np.ndarray, values: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(owners, kind="stable")
    owners_sorted = owners[order]
    values_sorted = values[order]
    counts = np.bincount(owners_sorted, minlength=count)
    offsets = np.concatenate(
        (np.asarray([0], dtype=np.int32), np.cumsum(counts, dtype=np.int32))
    )
    return offsets.astype(np.int32), values_sorted.astype(np.int32)


def _boundary_loops(
    starts: np.ndarray, ends: np.ndarray, vertex_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Trace boundary half-edges, given in increasing half-edge order, into loops.

    Loops are ordered by their smallest half-edge and list origins starting
    there, exactly as a traversal from the smallest remaining half-edge would.
    """

    count = starts.size
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

    def __init__(self, edges: Array, *, num_vertices: int | None = None) -> None:
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

    def __init__(self, faces: Array, *, num_vertices: int | None = None) -> None:
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
        if np.unique(np.sort(faces_host, axis=1), axis=0).shape[0] != faces_host.shape[0]:
            raise ValueError("TriangleTopology contains duplicate faces.")

        face_count = faces_host.shape[0]
        halfedge_count = 3 * face_count
        origin = faces_host.reshape((-1,))
        destination = faces_host[:, [1, 2, 0]].reshape((-1,))
        halfedge_face = np.repeat(np.arange(face_count, dtype=np.int32), 3)
        local = np.arange(halfedge_count, dtype=np.int32).reshape((-1, 3))
        halfedge_next = local[:, [1, 2, 0]].reshape((-1,))
        halfedge_previous = local[:, [2, 0, 1]].reshape((-1,))

        # Stable sort by undirected key: edges are lexicographic and each pair
        # keeps its half-edges in increasing order.
        low = np.minimum(origin, destination).astype(np.int64)
        high = np.maximum(origin, destination).astype(np.int64)
        order = np.argsort(low * vertex_count + high, kind="stable")
        starts = np.ones((halfedge_count,), dtype=np.bool_)
        starts[1:] = (low[order[1:]] != low[order[:-1]]) | (
            high[order[1:]] != high[order[:-1]]
        )
        first = np.flatnonzero(starts)
        group_sizes = np.diff(np.append(first, halfedge_count))
        if np.any(group_sizes > 2):
            raise ValueError(
                "TriangleTopology is non-manifold: an edge has more than two incident faces."
            )
        edges = np.stack((low[order[first]], high[order[first]]), axis=1).astype(np.int32)
        halfedge_edge = np.empty((halfedge_count,), dtype=np.int32)
        halfedge_edge[order] = np.cumsum(starts) - 1
        paired = group_sizes == 2
        first_halfedges = order[first[paired]]
        second_halfedges = order[first[paired] + 1]
        if np.any(
            (origin[first_halfedges] == origin[second_halfedges])
            | (destination[first_halfedges] == destination[second_halfedges])
        ):
            raise ValueError(
                "Adjacent faces have inconsistent orientation across an edge."
            )
        edge_halfedges = np.full((first.size, 2), -1, dtype=np.int32)
        edge_halfedges[:, 0] = order[first]
        edge_halfedges[paired, 1] = second_halfedges
        halfedge_twin = np.full((halfedge_count,), -1, dtype=np.int32)
        halfedge_twin[first_halfedges] = second_halfedges
        halfedge_twin[second_halfedges] = first_halfedges

        boundary_halfedges = np.flatnonzero(halfedge_twin < 0).astype(np.int32)
        loop_vertices, loop_offsets = _boundary_loops(
            origin[boundary_halfedges], destination[boundary_halfedges], vertex_count
        )
        vertex_halfedge_offsets, vertex_halfedges = _csr(
            origin, np.arange(halfedge_count, dtype=np.int32), vertex_count
        )
        # Vertex-face incidence is the face of each vertex-origin half-edge.
        vertex_face_offsets = vertex_halfedge_offsets
        vertex_faces = halfedge_face[vertex_halfedges]
        face_component_ids, face_component_count = _face_components(
            halfedge_face[first_halfedges], halfedge_face[second_halfedges], face_count
        )

        self.faces = jnp.asarray(faces_host, dtype=jnp.int32)
        self.edges = jnp.asarray(edges, dtype=jnp.int32)
        self.halfedge_origin = jnp.asarray(origin, dtype=jnp.int32)
        self.halfedge_destination = jnp.asarray(destination, dtype=jnp.int32)
        self.halfedge_face = jnp.asarray(halfedge_face, dtype=jnp.int32)
        self.halfedge_next = jnp.asarray(halfedge_next, dtype=jnp.int32)
        self.halfedge_previous = jnp.asarray(halfedge_previous, dtype=jnp.int32)
        self.halfedge_twin = jnp.asarray(halfedge_twin, dtype=jnp.int32)
        self.halfedge_edge = jnp.asarray(halfedge_edge, dtype=jnp.int32)
        self.edge_halfedges = jnp.asarray(edge_halfedges, dtype=jnp.int32)
        self.boundary_halfedges = jnp.asarray(boundary_halfedges, dtype=jnp.int32)
        self.boundary_loop_vertices = jnp.asarray(loop_vertices, dtype=jnp.int32)
        self.boundary_loop_offsets = jnp.asarray(loop_offsets, dtype=jnp.int32)
        self.vertex_face_offsets = jnp.asarray(vertex_face_offsets, dtype=jnp.int32)
        self.vertex_faces = jnp.asarray(vertex_faces, dtype=jnp.int32)
        self.vertex_halfedge_offsets = jnp.asarray(
            vertex_halfedge_offsets, dtype=jnp.int32
        )
        self.vertex_halfedges = jnp.asarray(vertex_halfedges, dtype=jnp.int32)
        self.face_component_ids = jnp.asarray(face_component_ids, dtype=jnp.int32)
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
