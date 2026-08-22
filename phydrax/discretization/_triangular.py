#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation
from ._topology import (
    CellComplexTopology,
    EntitySet,
    EntitySubset,
    OrientedIncidence,
)


class TriangleConnectivity(StrictModule, NonTrainableState):
    """Canonical edge incidence for an oriented triangular cell block."""

    edges: Array
    cell_edges: Array
    cell_edge_signs: Array
    edge_cell_counts: Array
    boundary_edges: Array
    boundary_vertices: Array
    vertex_count: int = eqx.field(static=True)


def triangle_connectivity(
    faces: ArrayLike,
    vertex_count: int,
    /,
) -> TriangleConnectivity:
    cells = np.asarray(faces, dtype=np.int32)
    vertices = int(vertex_count)
    if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] != 3:
        raise ValueError("faces must have shape (num_faces > 0, 3).")
    if vertices <= 0 or np.any(cells < 0) or np.any(cells >= vertices):
        raise ValueError("faces must index a positive covered vertex set.")
    edge_keys: dict[tuple[int, int], int] = {}
    cell_edges = np.empty(cells.shape, dtype=np.int32)
    cell_signs = np.empty(cells.shape, dtype=float)
    for cell_index, face in enumerate(cells):
        oriented = ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        for local_index, (start, stop) in enumerate(oriented):
            key = (min(int(start), int(stop)), max(int(start), int(stop)))
            if key not in edge_keys:
                edge_keys[key] = len(edge_keys)
            edge_index = edge_keys[key]
            cell_edges[cell_index, local_index] = edge_index
            cell_signs[cell_index, local_index] = 1.0 if (start, stop) == key else -1.0
    edges = np.asarray(tuple(edge_keys), dtype=np.int32)
    counts = np.bincount(cell_edges.reshape((-1,)), minlength=edges.shape[0])
    if np.any(counts > 2):
        raise ValueError("Triangular cell blocks must be edge-manifold.")
    boundary_edges = counts == 1
    boundary_vertices = np.zeros((vertices,), dtype=bool)
    if np.any(boundary_edges):
        boundary_vertices[np.unique(edges[boundary_edges].reshape((-1,)))] = True
    return TriangleConnectivity(
        edges=jnp.asarray(edges),
        cell_edges=jnp.asarray(cell_edges),
        cell_edge_signs=jnp.asarray(cell_signs),
        edge_cell_counts=jnp.asarray(counts),
        boundary_edges=jnp.asarray(boundary_edges),
        boundary_vertices=jnp.asarray(boundary_vertices),
        vertex_count=vertices,
    )


def triangle_cell_complex(
    faces: ArrayLike,
    vertex_count: int,
    /,
) -> CellComplexTopology:
    cells = np.asarray(faces, dtype=np.int32)
    connectivity = triangle_connectivity(cells, vertex_count)
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
    vertices = EntitySet(
        "vertices",
        0,
        np.arange(vertex_count, dtype=np.int32),
        subsets=(EntitySubset("boundary", connectivity.boundary_vertices),),
    )
    edge_entities = EntitySet(
        "edges",
        1,
        np.arange(edges.shape[0], dtype=np.int32),
        subsets=(EntitySubset("boundary", connectivity.boundary_edges),),
    )
    cell_entities = EntitySet(
        "faces",
        2,
        np.arange(cells.shape[0], dtype=np.int32),
        subsets=(EntitySubset("boundary", np.zeros((cells.shape[0],), dtype=bool)),),
    )
    vertex_edge_relation = EdgeRelation(
        edges.reshape((-1,)),
        np.repeat(np.arange(edges.shape[0], dtype=np.int32), 2),
        source_size=vertex_count,
        target_size=edges.shape[0],
    )
    edge_cell_relation = EdgeRelation(
        cell_edges.reshape((-1,)),
        np.repeat(np.arange(cells.shape[0], dtype=np.int32), 3),
        source_size=edges.shape[0],
        target_size=cells.shape[0],
    )
    return CellComplexTopology(
        (vertices, edge_entities, cell_entities),
        (
            OrientedIncidence(
                1,
                vertices,
                edge_entities,
                vertex_edge_relation,
                np.tile(np.asarray([-1.0, 1.0]), edges.shape[0]),
            ),
            OrientedIncidence(
                2,
                edge_entities,
                cell_entities,
                edge_cell_relation,
                connectivity.cell_edge_signs.reshape((-1,)),
            ),
        ),
    )


__all__ = [
    "TriangleConnectivity",
    "triangle_cell_complex",
    "triangle_connectivity",
]
