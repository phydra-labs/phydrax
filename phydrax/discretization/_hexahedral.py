from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation
from ._topology import CellComplexTopology, EntitySet, EntitySubset, OrientedIncidence


_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
_FACES = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


class HexahedralConnectivity(StrictModule, NonTrainableState):
    edges: Array
    faces: Array
    face_edges: Array
    face_edge_signs: Array
    cell_edges: Array
    cell_faces: Array
    cell_face_signs: Array
    face_cell_counts: Array
    boundary_vertices: Array
    boundary_edges: Array
    boundary_faces: Array
    vertex_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)


def _cycle_sign(cycle, canonical):
    n = len(cycle)
    for shift in range(n):
        if tuple(cycle[shift:] + cycle[:shift]) == canonical:
            return 1.0
    reversed_cycle = list(reversed(cycle))
    for shift in range(n):
        if tuple(reversed_cycle[shift:] + reversed_cycle[:shift]) == canonical:
            return -1.0
    raise ValueError("Hex face orientation is inconsistent.")


def hexahedral_connectivity(
    hexahedra: ArrayLike, vertex_count: int, /
) -> HexahedralConnectivity:
    cells = np.asarray(hexahedra, dtype=np.int32)
    if (
        cells.ndim != 2
        or cells.shape[1] != 8
        or np.any(cells < 0)
        or np.any(cells >= vertex_count)
    ):
        raise ValueError("hexahedra must have shape (n,8) with valid vertices.")
    edge_map = {}
    edges = []
    cell_edges = np.empty((len(cells), 12), np.int32)
    for ci, cell in enumerate(cells):
        for li, (a, b) in enumerate(_EDGES):
            pair = (int(cell[a]), int(cell[b]))
            key = tuple(sorted(pair))
            if key not in edge_map:
                edge_map[key] = len(edges)
                edges.append(key)
            cell_edges[ci, li] = edge_map[key]
    face_map = {}
    faces = []
    face_counts = []
    cell_faces = np.empty((len(cells), 6), np.int32)
    cell_signs = np.empty((len(cells), 6), float)
    for ci, cell in enumerate(cells):
        for li, template in enumerate(_FACES):
            cycle = [int(cell[i]) for i in template]
            key = tuple(sorted(cycle))
            if key not in face_map:
                face_map[key] = len(faces)
                faces.append(tuple(cycle))
                face_counts.append(0)
            fi = face_map[key]
            cell_faces[ci, li] = fi
            cell_signs[ci, li] = _cycle_sign(cycle, faces[fi])
            face_counts[fi] += 1
            if face_counts[fi] > 2:
                raise ValueError("Non-manifold hexahedral face.")
    edges = np.asarray(edges, np.int32)
    faces = np.asarray(faces, np.int32)
    counts = np.asarray(face_counts, np.int32)
    face_edges = np.empty((len(faces), 4), np.int32)
    face_edge_signs = np.empty((len(faces), 4), float)
    for fi, face in enumerate(faces):
        for j in range(4):
            pair = (int(face[j]), int(face[(j + 1) % 4]))
            key = tuple(sorted(pair))
            face_edges[fi, j] = edge_map[key]
            face_edge_signs[fi, j] = 1.0 if pair == key else -1.0
    boundary_faces = counts == 1
    boundary_edges = np.zeros(len(edges), bool)
    boundary_vertices = np.zeros(vertex_count, bool)
    if np.any(boundary_faces):
        boundary_edges[np.unique(face_edges[boundary_faces])] = True
        boundary_vertices[np.unique(faces[boundary_faces])] = True
    return HexahedralConnectivity(
        edges=jnp.asarray(edges),
        faces=jnp.asarray(faces),
        face_edges=jnp.asarray(face_edges),
        face_edge_signs=jnp.asarray(face_edge_signs),
        cell_edges=jnp.asarray(cell_edges),
        cell_faces=jnp.asarray(cell_faces),
        cell_face_signs=jnp.asarray(cell_signs),
        face_cell_counts=jnp.asarray(counts),
        boundary_vertices=jnp.asarray(boundary_vertices),
        boundary_edges=jnp.asarray(boundary_edges),
        boundary_faces=jnp.asarray(boundary_faces),
        vertex_count=vertex_count,
        cell_count=len(cells),
    )


def hexahedral_cell_complex(
    hexahedra: ArrayLike,
    vertex_count: int,
    /,
    *,
    vertex_global_ids=None,
    cell_global_ids=None,
) -> CellComplexTopology:
    c = hexahedral_connectivity(hexahedra, vertex_count)
    edges = np.asarray(c.edges)
    faces = np.asarray(c.faces)
    vids = (
        np.arange(vertex_count, dtype=np.int64)
        if vertex_global_ids is None
        else np.asarray(vertex_global_ids, dtype=np.int64)
    )
    cids = (
        np.arange(c.cell_count, dtype=np.int64)
        if cell_global_ids is None
        else np.asarray(cell_global_ids, dtype=np.int64)
    )
    v = EntitySet(
        "vertices", 0, vids, subsets=(EntitySubset("boundary", c.boundary_vertices),)
    )
    e = EntitySet(
        "edges",
        1,
        np.arange(len(edges)),
        subsets=(EntitySubset("boundary", c.boundary_edges),),
    )
    f = EntitySet(
        "faces",
        2,
        np.arange(len(faces)),
        subsets=(EntitySubset("boundary", c.boundary_faces),),
    )
    cells = EntitySet(
        "cells",
        3,
        cids,
        subsets=(EntitySubset("boundary", np.zeros(c.cell_count, bool)),),
    )
    ve = OrientedIncidence(
        1,
        v,
        e,
        EdgeRelation(
            edges.reshape(-1),
            np.repeat(np.arange(len(edges)), 2),
            source_size=vertex_count,
            target_size=len(edges),
        ),
        np.tile(np.asarray([-1.0, 1.0]), len(edges)),
    )
    ef = OrientedIncidence(
        2,
        e,
        f,
        EdgeRelation(
            np.asarray(c.face_edges).reshape(-1),
            np.repeat(np.arange(len(faces)), 4),
            source_size=len(edges),
            target_size=len(faces),
        ),
        np.asarray(c.face_edge_signs).reshape(-1),
    )
    fc = OrientedIncidence(
        3,
        f,
        cells,
        EdgeRelation(
            np.asarray(c.cell_faces).reshape(-1),
            np.repeat(np.arange(c.cell_count), 6),
            source_size=len(faces),
            target_size=c.cell_count,
        ),
        np.asarray(c.cell_face_signs).reshape(-1),
    )
    return CellComplexTopology((v, e, f, cells), (ve, ef, fc))


__all__ = ["HexahedralConnectivity", "hexahedral_connectivity", "hexahedral_cell_complex"]
