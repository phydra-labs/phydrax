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


def _permutation_sign(values: tuple[int, ...], canonical: tuple[int, ...], /) -> float:
    positions = [canonical.index(value) for value in values]
    inversions = sum(
        positions[left] > positions[right]
        for left in range(len(positions))
        for right in range(left + 1, len(positions))
    )
    return -1.0 if inversions % 2 else 1.0


def _validated_cells(
    name: str,
    value: ArrayLike | None,
    arity: int,
    vertex_count: int,
    /,
) -> np.ndarray:
    if value is None:
        return np.empty((0, arity), dtype=np.int32)
    cells = np.asarray(value, dtype=np.int32)
    if cells.ndim != 2 or cells.shape[1] != arity:
        raise ValueError(f"{name} must have shape (n, {arity}).")
    if np.any(cells < 0) or np.any(cells >= vertex_count):
        raise ValueError(f"{name} index vertices outside the declared vertex set.")
    if cells.size and np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        raise ValueError(f"{name} must contain distinct vertices per cell.")
    if (
        cells.shape[0]
        and np.unique(np.sort(cells, axis=1), axis=0).shape[0] != cells.shape[0]
    ):
        raise ValueError(f"{name} contains duplicate cells.")
    return cells


def _resolved_entity_ids(
    name: str,
    value: ArrayLike | None,
    count: int,
    /,
) -> np.ndarray:
    identifiers = (
        np.arange(count, dtype=np.int64)
        if value is None
        else np.asarray(value, dtype=np.int64)
    )
    if identifiers.shape != (count,):
        raise ValueError(f"{name} must have shape {(count,)}.")
    if np.any(identifiers < 0) or np.unique(identifiers).size != count:
        raise ValueError(f"{name} must contain unique non-negative IDs.")
    return identifiers


def _canonical_entity_ids(keys: np.ndarray, /) -> np.ndarray:
    if keys.ndim != 2:
        raise ValueError("Canonical entity keys must be rank-2.")
    order = np.lexsort(tuple(keys[:, axis] for axis in range(keys.shape[1] - 1, -1, -1)))
    identifiers = np.empty((keys.shape[0],), dtype=np.int64)
    identifiers[order] = np.arange(keys.shape[0], dtype=np.int64)
    return identifiers


class PolygonalConnectivity(StrictModule, NonTrainableState):
    """Canonical edge incidence for mixed triangular/quadrilateral cells."""

    edges: Array
    cell_vertices: Array
    cell_vertex_valid: Array
    cell_kinds: Array
    cell_edges: Array
    cell_edge_signs: Array
    cell_edge_valid: Array
    edge_cell_counts: Array
    boundary_edges: Array
    boundary_vertices: Array
    vertex_count: int = eqx.field(static=True)
    triangle_count: int = eqx.field(static=True)
    quadrilateral_count: int = eqx.field(static=True)

    @property
    def cell_count(self) -> int:
        return self.triangle_count + self.quadrilateral_count


class TetrahedralConnectivity(StrictModule, NonTrainableState):
    """Canonical vertex/edge/face incidence for tetrahedral cells."""

    edges: Array
    faces: Array
    face_edges: Array
    face_edge_signs: Array
    cell_faces: Array
    cell_face_signs: Array
    face_cell_counts: Array
    boundary_vertices: Array
    boundary_edges: Array
    boundary_faces: Array
    vertex_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)


def polygonal_connectivity(
    triangles: ArrayLike | None,
    quadrilaterals: ArrayLike | None,
    vertex_count: int,
    /,
) -> PolygonalConnectivity:
    """Build one edge-manifold connectivity record for mixed 2-D cells."""

    vertices = int(vertex_count)
    if vertices <= 0:
        raise ValueError("vertex_count must be positive.")
    triangle_cells = _validated_cells("triangles", triangles, 3, vertices)
    quadrilateral_cells = _validated_cells("quadrilaterals", quadrilaterals, 4, vertices)
    if triangle_cells.shape[0] + quadrilateral_cells.shape[0] == 0:
        raise ValueError("At least one polygonal cell is required.")

    cell_count = triangle_cells.shape[0] + quadrilateral_cells.shape[0]
    cell_vertices = np.zeros((cell_count, 4), dtype=np.int32)
    cell_valid = np.zeros((cell_count, 4), dtype=bool)
    cell_kinds = np.empty((cell_count,), dtype=np.int8)
    triangle_count = triangle_cells.shape[0]
    cell_vertices[:triangle_count, :3] = triangle_cells
    cell_valid[:triangle_count, :3] = True
    cell_kinds[:triangle_count] = 3
    cell_vertices[triangle_count:] = quadrilateral_cells
    cell_valid[triangle_count:] = True
    cell_kinds[triangle_count:] = 4

    edge_keys: dict[tuple[int, int], int] = {}
    cell_edges = np.zeros((cell_count, 4), dtype=np.int32)
    cell_signs = np.zeros((cell_count, 4), dtype=float)
    incidents: list[list[float]] = []
    for cell in range(cell_count):
        arity = int(cell_kinds[cell])
        local_vertices = cell_vertices[cell, :arity]
        for local in range(arity):
            start = int(local_vertices[local])
            stop = int(local_vertices[(local + 1) % arity])
            key = (min(start, stop), max(start, stop))
            if key not in edge_keys:
                edge_keys[key] = len(edge_keys)
                incidents.append([])
            edge = edge_keys[key]
            sign = 1.0 if (start, stop) == key else -1.0
            cell_edges[cell, local] = edge
            cell_signs[cell, local] = sign
            incidents[edge].append(sign)
    if any(len(values) > 2 for values in incidents):
        raise ValueError("Polygonal cells must be edge-manifold.")
    if any(len(values) == 2 and values[0] == values[1] for values in incidents):
        raise ValueError("Shared polygon edges must have opposite orientation.")

    edges = np.asarray(tuple(edge_keys), dtype=np.int32)
    counts = np.asarray([len(values) for values in incidents], dtype=np.int32)
    boundary_edges = counts == 1
    boundary_vertices = np.zeros((vertices,), dtype=bool)
    boundary_vertices[np.unique(edges[boundary_edges].reshape((-1,)))] = True
    return PolygonalConnectivity(
        edges=jnp.asarray(edges),
        cell_vertices=jnp.asarray(cell_vertices),
        cell_vertex_valid=jnp.asarray(cell_valid),
        cell_kinds=jnp.asarray(cell_kinds),
        cell_edges=jnp.asarray(cell_edges),
        cell_edge_signs=jnp.asarray(cell_signs),
        cell_edge_valid=jnp.asarray(cell_valid),
        edge_cell_counts=jnp.asarray(counts),
        boundary_edges=jnp.asarray(boundary_edges),
        boundary_vertices=jnp.asarray(boundary_vertices),
        vertex_count=vertices,
        triangle_count=triangle_count,
        quadrilateral_count=quadrilateral_cells.shape[0],
    )


def polygonal_cell_complex(
    triangles: ArrayLike | None,
    quadrilaterals: ArrayLike | None,
    vertex_count: int,
    /,
    *,
    vertex_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
) -> CellComplexTopology:

    connectivity = polygonal_connectivity(triangles, quadrilaterals, vertex_count)
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
    cell_valid = np.asarray(connectivity.cell_edge_valid, dtype=bool)
    cell_signs = np.asarray(connectivity.cell_edge_signs)
    vertex_ids = _resolved_entity_ids(
        "vertex_global_ids", vertex_global_ids, vertex_count
    )
    cell_ids_global = _resolved_entity_ids(
        "cell_global_ids", cell_global_ids, connectivity.cell_count
    )
    edge_keys = np.sort(vertex_ids[edges], axis=1)
    edge_ids_global = _canonical_entity_ids(edge_keys)
    vertices = EntitySet(
        "vertices",
        0,
        vertex_ids,
        subsets=(EntitySubset("boundary", connectivity.boundary_vertices),),
    )
    edge_entities = EntitySet(
        "edges",
        1,
        edge_ids_global,
        subsets=(EntitySubset("boundary", connectivity.boundary_edges),),
    )
    cell_entities = EntitySet(
        "cells",
        2,
        cell_ids_global,
        subsets=(
            EntitySubset("boundary", np.zeros((connectivity.cell_count,), dtype=bool)),
        ),
    )
    vertex_edge_relation = EdgeRelation(
        edges.reshape((-1,)),
        np.repeat(np.arange(edges.shape[0], dtype=np.int32), 2),
        source_size=vertex_count,
        target_size=edges.shape[0],
    )
    cell_ids = np.broadcast_to(
        np.arange(connectivity.cell_count, dtype=np.int32)[:, None], cell_edges.shape
    )
    edge_cell_relation = EdgeRelation(
        cell_edges[cell_valid],
        cell_ids[cell_valid],
        source_size=edges.shape[0],
        target_size=connectivity.cell_count,
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
                cell_signs[cell_valid],
            ),
        ),
    )


def tetrahedral_connectivity(
    tetrahedra: ArrayLike,
    vertex_count: int,
    /,
) -> TetrahedralConnectivity:
    """Build canonical oriented connectivity for an affine tetrahedral mesh."""

    vertices = int(vertex_count)
    if vertices <= 0:
        raise ValueError("vertex_count must be positive.")
    cells = _validated_cells("tetrahedra", tetrahedra, 4, vertices)
    if cells.shape[0] == 0:
        raise ValueError("At least one tetrahedral cell is required.")

    edge_keys: dict[tuple[int, int], int] = {}
    face_keys: dict[tuple[int, int, int], int] = {}
    face_incidents: list[list[float]] = []
    cell_faces = np.empty((cells.shape[0], 4), dtype=np.int32)
    cell_face_signs = np.empty((cells.shape[0], 4), dtype=float)
    local_face_routes = (
        (1, 2, 3),
        (0, 3, 2),
        (0, 1, 3),
        (0, 2, 1),
    )
    for cell, tetrahedron in enumerate(cells):
        for local, route in enumerate(local_face_routes):
            oriented = (
                int(tetrahedron[route[0]]),
                int(tetrahedron[route[1]]),
                int(tetrahedron[route[2]]),
            )
            ordered = sorted(oriented)
            canonical = (ordered[0], ordered[1], ordered[2])
            if canonical not in face_keys:
                face_keys[canonical] = len(face_keys)
                face_incidents.append([])
            face = face_keys[canonical]
            sign = _permutation_sign(oriented, canonical)
            cell_faces[cell, local] = face
            cell_face_signs[cell, local] = sign
            face_incidents[face].append(sign)
            for start, stop in (
                (canonical[0], canonical[1]),
                (canonical[0], canonical[2]),
                (canonical[1], canonical[2]),
            ):
                edge_keys.setdefault((start, stop), len(edge_keys))
    if any(len(values) > 2 for values in face_incidents):
        raise ValueError("Tetrahedral cells must be face-manifold.")
    if any(len(values) == 2 and values[0] == values[1] for values in face_incidents):
        raise ValueError("Shared tetrahedral faces must have opposite orientation.")

    edges = np.asarray(tuple(edge_keys), dtype=np.int32)
    faces = np.asarray(tuple(face_keys), dtype=np.int32)
    face_edges = np.empty((faces.shape[0], 3), dtype=np.int32)
    face_edge_signs = np.empty((faces.shape[0], 3), dtype=float)
    for face, (first, second, third) in enumerate(faces):
        oriented_edges = ((second, third), (third, first), (first, second))
        for local, (start, stop) in enumerate(oriented_edges):
            canonical_edge = (min(int(start), int(stop)), max(int(start), int(stop)))
            face_edges[face, local] = edge_keys[canonical_edge]
            face_edge_signs[face, local] = (
                1.0 if (int(start), int(stop)) == canonical_edge else -1.0
            )
    counts = np.asarray([len(values) for values in face_incidents], dtype=np.int32)
    boundary_faces = counts == 1
    boundary_edges = np.zeros((edges.shape[0],), dtype=bool)
    boundary_edges[np.unique(face_edges[boundary_faces].reshape((-1,)))] = True
    boundary_vertices = np.zeros((vertices,), dtype=bool)
    boundary_vertices[np.unique(faces[boundary_faces].reshape((-1,)))] = True
    return TetrahedralConnectivity(
        edges=jnp.asarray(edges),
        faces=jnp.asarray(faces),
        face_edges=jnp.asarray(face_edges),
        face_edge_signs=jnp.asarray(face_edge_signs),
        cell_faces=jnp.asarray(cell_faces),
        cell_face_signs=jnp.asarray(cell_face_signs),
        face_cell_counts=jnp.asarray(counts),
        boundary_vertices=jnp.asarray(boundary_vertices),
        boundary_edges=jnp.asarray(boundary_edges),
        boundary_faces=jnp.asarray(boundary_faces),
        vertex_count=vertices,
        cell_count=cells.shape[0],
    )


def tetrahedral_cell_complex(
    tetrahedra: ArrayLike,
    vertex_count: int,
    /,
    *,
    vertex_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
) -> CellComplexTopology:
    """Build the validated 0→1→2→3 complex for tetrahedral cells."""

    connectivity = tetrahedral_connectivity(tetrahedra, vertex_count)
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    faces = np.asarray(connectivity.faces, dtype=np.int32)
    face_edges = np.asarray(connectivity.face_edges, dtype=np.int32)
    cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int32)
    cells = np.asarray(tetrahedra, dtype=np.int32)
    vertex_ids = _resolved_entity_ids(
        "vertex_global_ids", vertex_global_ids, vertex_count
    )
    cell_ids_global = _resolved_entity_ids(
        "cell_global_ids", cell_global_ids, cells.shape[0]
    )
    edge_ids_global = _canonical_entity_ids(np.sort(vertex_ids[edges], axis=1))
    face_ids_global = _canonical_entity_ids(np.sort(vertex_ids[faces], axis=1))
    vertex_entities = EntitySet(
        "vertices",
        0,
        vertex_ids,
        subsets=(EntitySubset("boundary", connectivity.boundary_vertices),),
    )
    edge_entities = EntitySet(
        "edges",
        1,
        edge_ids_global,
        subsets=(EntitySubset("boundary", connectivity.boundary_edges),),
    )
    face_entities = EntitySet(
        "faces",
        2,
        face_ids_global,
        subsets=(EntitySubset("boundary", connectivity.boundary_faces),),
    )
    cell_entities = EntitySet(
        "cells",
        3,
        cell_ids_global,
        subsets=(EntitySubset("boundary", np.zeros((cells.shape[0],), dtype=bool)),),
    )
    vertex_edge_relation = EdgeRelation(
        edges.reshape((-1,)),
        np.repeat(np.arange(edges.shape[0], dtype=np.int32), 2),
        source_size=vertex_count,
        target_size=edges.shape[0],
    )
    edge_face_relation = EdgeRelation(
        face_edges.reshape((-1,)),
        np.repeat(np.arange(faces.shape[0], dtype=np.int32), 3),
        source_size=edges.shape[0],
        target_size=faces.shape[0],
    )
    face_cell_relation = EdgeRelation(
        cell_faces.reshape((-1,)),
        np.repeat(np.arange(cells.shape[0], dtype=np.int32), 4),
        source_size=faces.shape[0],
        target_size=cells.shape[0],
    )
    return CellComplexTopology(
        (vertex_entities, edge_entities, face_entities, cell_entities),
        (
            OrientedIncidence(
                1,
                vertex_entities,
                edge_entities,
                vertex_edge_relation,
                np.tile(np.asarray([-1.0, 1.0]), edges.shape[0]),
            ),
            OrientedIncidence(
                2,
                edge_entities,
                face_entities,
                edge_face_relation,
                np.asarray(connectivity.face_edge_signs).reshape((-1,)),
            ),
            OrientedIncidence(
                3,
                face_entities,
                cell_entities,
                face_cell_relation,
                np.asarray(connectivity.cell_face_signs).reshape((-1,)),
            ),
        ),
    )


__all__ = [
    "PolygonalConnectivity",
    "TetrahedralConnectivity",
    "polygonal_cell_complex",
    "polygonal_connectivity",
    "tetrahedral_cell_complex",
    "tetrahedral_connectivity",
]
