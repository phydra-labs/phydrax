#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...discretization import (
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


class SegmentTopology(StrictModule):
    """Validated unoriented segment connectivity with vertex incidence CSR."""

    edges: Array
    vertex_edge_offsets: Array
    vertex_edges: Array
    num_vertices: int = eqx.field(static=True)

    def __init__(self, edges: Array, *, num_vertices: int | None = None):
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
        boundary_vertices = np.zeros((self.num_vertices,), dtype=bool)
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
                    np.zeros((self.num_edges,), dtype=bool),
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
    num_vertices: int = eqx.field(static=True)
    watertight: bool = eqx.field(static=True)

    def __init__(self, faces: Array, *, num_vertices: int | None = None):
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

        edge_groups: dict[tuple[int, int], list[int]] = {}
        for halfedge, (start, end) in enumerate(zip(origin, destination, strict=True)):
            key = (min(int(start), int(end)), max(int(start), int(end)))
            edge_groups.setdefault(key, []).append(halfedge)
        if any(len(group) > 2 for group in edge_groups.values()):
            raise ValueError(
                "TriangleTopology is non-manifold: an edge has more than two incident faces."
            )

        halfedge_twin = np.full((halfedge_count,), -1, dtype=np.int32)
        halfedge_edge = np.empty((halfedge_count,), dtype=np.int32)
        edges = np.empty((len(edge_groups), 2), dtype=np.int32)
        edge_halfedges = np.full((len(edge_groups), 2), -1, dtype=np.int32)
        for edge_index, (key, group) in enumerate(sorted(edge_groups.items())):
            edges[edge_index] = key
            edge_halfedges[edge_index, : len(group)] = group
            halfedge_edge[group] = edge_index
            if len(group) == 2:
                first, second = group
                if (
                    origin[first] == origin[second]
                    or destination[first] == destination[second]
                ):
                    raise ValueError(
                        "Adjacent faces have inconsistent orientation across an edge."
                    )
                halfedge_twin[first] = second
                halfedge_twin[second] = first

        boundary_halfedges = np.flatnonzero(halfedge_twin < 0).astype(np.int32)
        boundary_loops: list[np.ndarray] = []
        if boundary_halfedges.size:
            outgoing: dict[int, int] = {}
            incoming: dict[int, int] = {}
            for halfedge in boundary_halfedges:
                start = int(origin[halfedge])
                end = int(destination[halfedge])
                if start in outgoing or end in incoming:
                    raise ValueError("Boundary is non-manifold at a vertex.")
                outgoing[start] = int(halfedge)
                incoming[end] = int(halfedge)
            if set(outgoing) != set(incoming):
                raise ValueError("Boundary half-edges do not form closed loops.")
            remaining = set(map(int, boundary_halfedges))
            while remaining:
                first_halfedge = min(remaining)
                first_vertex = int(origin[first_halfedge])
                vertices: list[int] = []
                current = first_vertex
                while True:
                    halfedge = outgoing[current]
                    if halfedge not in remaining:
                        if current != first_vertex:
                            raise ValueError(
                                "Boundary traversal encountered a repeated half-edge."
                            )
                        break
                    remaining.remove(halfedge)
                    vertices.append(current)
                    current = int(destination[halfedge])
                    if current == first_vertex:
                        break
                boundary_loops.append(np.asarray(vertices, dtype=np.int32))
        loop_offsets = np.zeros((len(boundary_loops) + 1,), dtype=np.int32)
        if boundary_loops:
            loop_offsets[1:] = np.cumsum(
                [loop.size for loop in boundary_loops], dtype=np.int32
            )
            loop_vertices = np.concatenate(boundary_loops)
        else:
            loop_vertices = np.zeros((0,), dtype=np.int32)

        vertex_face_owners = faces_host.reshape((-1,))
        vertex_face_values = np.repeat(np.arange(face_count, dtype=np.int32), 3)
        vertex_face_offsets, vertex_faces = _csr(
            vertex_face_owners, vertex_face_values, vertex_count
        )
        vertex_halfedge_offsets, vertex_halfedges = _csr(
            origin, np.arange(halfedge_count, dtype=np.int32), vertex_count
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
        boundary_vertices = np.zeros((self.num_vertices,), dtype=bool)
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
                    np.zeros((self.num_faces,), dtype=bool),
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
