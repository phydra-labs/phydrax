#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

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


class IntervalConnectivity(StrictModule, NonTrainableState):
    """Canonical vertex incidence for one-dimensional interval cells."""

    cell_vertices: Array
    vertex_cell_counts: Array
    boundary_vertices: Array
    vertex_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)


def interval_connectivity(
    intervals: ArrayLike,
    vertex_count: int,
    /,
) -> IntervalConnectivity:
    cells = _validated_cells("intervals", intervals, 2, int(vertex_count))
    counts = np.bincount(cells.reshape((-1,)), minlength=int(vertex_count))
    if np.any(counts > 2):
        raise ValueError("Interval cells must form a vertex-manifold mesh.")
    return IntervalConnectivity(
        jnp.asarray(cells),
        jnp.asarray(counts, dtype=jnp.int32),
        jnp.asarray(counts == 1),
        int(vertex_count),
        int(cells.shape[0]),
    )


def interval_cell_complex(
    intervals: ArrayLike,
    vertex_count: int,
    /,
    *,
    vertex_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
) -> CellComplexTopology:
    connectivity = interval_connectivity(intervals, vertex_count)
    cells = np.asarray(connectivity.cell_vertices, dtype=np.int32)
    vertex_ids = _resolved_entity_ids(
        "vertex_global_ids", vertex_global_ids, int(vertex_count)
    )
    cell_ids = _resolved_entity_ids(
        "cell_global_ids", cell_global_ids, int(cells.shape[0])
    )
    vertex_entities = EntitySet(
        "vertices",
        0,
        vertex_ids,
        subsets=(EntitySubset("boundary", connectivity.boundary_vertices),),
    )
    cell_entities = EntitySet(
        "cells",
        1,
        cell_ids,
        subsets=(EntitySubset("boundary", np.zeros((cells.shape[0],), dtype=bool)),),
    )
    relation = EdgeRelation(
        cells.reshape((-1,)),
        np.repeat(np.arange(cells.shape[0], dtype=np.int32), 2),
        source_size=int(vertex_count),
        target_size=int(cells.shape[0]),
    )
    incidence = OrientedIncidence(
        1,
        vertex_entities,
        cell_entities,
        relation,
        np.tile(np.asarray((-1.0, 1.0)), cells.shape[0]),
    )
    return CellComplexTopology((vertex_entities, cell_entities), (incidence,))


class PolygonalConnectivity(StrictModule, NonTrainableState):
    """Canonical edge incidence for mixed two-dimensional polygonal cells."""

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
    polygon_count: int = eqx.field(static=True)

    @property
    def cell_count(self) -> int:
        return self.triangle_count + self.quadrilateral_count + self.polygon_count


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


class PolyhedralConnectivity(StrictModule, NonTrainableState):
    """Canonical oriented incidence for mixed or face-defined polyhedra."""

    edges: Array
    faces: Array
    face_arities: Array
    face_vertex_valid: Array
    face_edges: Array
    face_edge_signs: Array
    face_edge_valid: Array
    cell_faces: Array
    cell_face_signs: Array
    cell_face_valid: Array
    cell_vertices: Array
    cell_vertex_valid: Array
    face_owner: Array
    face_neighbour: Array
    face_owner_local: Array
    face_neighbour_local: Array
    face_cell_counts: Array
    boundary_vertices: Array
    boundary_edges: Array
    boundary_faces: Array
    vertex_global_ids: Array
    edge_global_ids: Array
    face_global_ids: Array
    cell_global_ids: Array
    vertex_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    maximum_face_arity: int = eqx.field(static=True)
    maximum_cell_faces: int = eqx.field(static=True)
    maximum_cell_vertices: int = eqx.field(static=True)

    @property
    def face_vertices(self) -> Array:
        return jnp.where(self.face_vertex_valid, self.faces, 0)


def polygonal_connectivity(
    triangles: ArrayLike | None,
    quadrilaterals: ArrayLike | None,
    vertex_count: int,
    /,
    *,
    polygons: Sequence[ArrayLike] = (),
) -> PolygonalConnectivity:
    """Build one edge-manifold record for mixed 2-D polygonal cells."""

    vertices = int(vertex_count)
    if vertices <= 0:
        raise ValueError("vertex_count must be positive.")
    triangle_cells = _validated_cells("triangles", triangles, 3, vertices)
    quadrilateral_cells = _validated_cells("quadrilaterals", quadrilaterals, 4, vertices)
    polygon_cells = []
    for index, values in enumerate(polygons):
        array = np.asarray(values, dtype=np.int32)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(f"polygon block {index} must have shape (n, arity >= 3).")
        polygon_cells.append(
            _validated_cells(
                f"polygon block {index}",
                array,
                int(array.shape[1]),
                vertices,
            )
        )
    blocks = (triangle_cells, quadrilateral_cells, *polygon_cells)
    if sum(block.shape[0] for block in blocks) == 0:
        raise ValueError("At least one polygonal cell is required.")
    canonical_cells = [
        tuple(sorted(int(value) for value in cell)) for block in blocks for cell in block
    ]
    if len(set(canonical_cells)) != len(canonical_cells):
        raise ValueError("Polygonal connectivity contains duplicate cells.")

    capacity = max(4, *(block.shape[1] for block in blocks))
    cell_count = sum(block.shape[0] for block in blocks)
    cell_vertices = np.zeros((cell_count, capacity), dtype=np.int32)
    cell_valid = np.zeros((cell_count, capacity), dtype=bool)
    cell_kinds = np.empty((cell_count,), dtype=np.int32)
    offset = 0
    for block in blocks:
        count, arity = block.shape
        cell_vertices[offset : offset + count, :arity] = block
        cell_valid[offset : offset + count, :arity] = True
        cell_kinds[offset : offset + count] = arity
        offset += count

    edge_keys: dict[tuple[int, int], int] = {}
    cell_edges = np.zeros((cell_count, capacity), dtype=np.int32)
    cell_signs = np.zeros((cell_count, capacity), dtype=float)
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
        triangle_count=triangle_cells.shape[0],
        quadrilateral_count=quadrilateral_cells.shape[0],
        polygon_count=sum(block.shape[0] for block in polygon_cells),
    )


def polygonal_cell_complex(
    triangles: ArrayLike | None,
    quadrilaterals: ArrayLike | None,
    vertex_count: int,
    /,
    *,
    polygons: Sequence[ArrayLike] = (),
    vertex_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
) -> CellComplexTopology:

    connectivity = polygonal_connectivity(
        triangles,
        quadrilaterals,
        vertex_count,
        polygons=polygons,
    )
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


def _rotations(values: tuple[int, ...], /) -> tuple[tuple[int, ...], ...]:
    return tuple(values[index:] + values[:index] for index in range(len(values)))


def _canonical_face_loop(values: Sequence[int], /) -> tuple[tuple[int, ...], float]:
    loop = tuple(int(value) for value in values)
    forward = _rotations(loop)
    reverse = _rotations(tuple(reversed(loop)))
    canonical = min(*forward, *reverse)
    return canonical, 1.0 if canonical in forward else -1.0


def _canonical_variable_ids(keys: Sequence[tuple[int, ...]], /) -> np.ndarray:
    order = sorted(range(len(keys)), key=lambda index: keys[index])
    identifiers = np.empty((len(keys),), dtype=np.int64)
    identifiers[np.asarray(order, dtype=np.int32)] = np.arange(len(keys), dtype=np.int64)
    return identifiers


_POLYHEDRAL_FACE_ROUTES = {
    "tetrahedron": ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
    "hexahedron": (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
    "prism": (
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (2, 0, 3, 5),
    ),
    "pyramid": (
        (0, 3, 2, 1),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ),
}

_POLYHEDRAL_CELL_ARITIES = {
    "tetrahedron": 4,
    "hexahedron": 8,
    "prism": 6,
    "pyramid": 5,
}


def _polyhedral_face_cells(
    cells: Sequence[Sequence[ArrayLike]] | Sequence[tuple[str, ArrayLike]],
    vertex_count: int,
    /,
) -> tuple[tuple[ArrayLike, ...], ...]:
    entries = tuple(cells)
    descriptors = tuple(
        isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str)
        for entry in entries
    )
    if not any(descriptors):
        return tuple(tuple(faces) for faces in entries)
    if not all(descriptors):
        raise ValueError(
            "Polyhedral connectivity cannot mix cell blocks and explicit face loops."
        )
    normalized: list[tuple[ArrayLike, ...]] = []
    for kind_value, values in entries:
        kind = str(kind_value)
        if kind not in _POLYHEDRAL_FACE_ROUTES:
            raise ValueError(f"Unsupported polyhedral cell kind {kind!r}.")
        block = _validated_cells(
            kind,
            values,
            _POLYHEDRAL_CELL_ARITIES[kind],
            vertex_count,
        )
        routes = _POLYHEDRAL_FACE_ROUTES[kind]
        normalized.extend(
            tuple(
                np.asarray(
                    tuple(int(cell_vertices[index]) for index in route),
                    dtype=np.int32,
                )
                for route in routes
            )
            for cell_vertices in block
        )
    return tuple(normalized)


def polyhedral_connectivity(
    cells: Sequence[Sequence[ArrayLike]] | Sequence[tuple[str, ArrayLike]],
    vertex_count: int,
    /,
    *,
    vertex_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
) -> PolyhedralConnectivity:
    """Build exact oriented incidence from standard cells or explicit face loops."""

    vertices = int(vertex_count)
    if vertices <= 0:
        raise ValueError("vertex_count must be positive.")
    normalized_cells = _polyhedral_face_cells(cells, vertices)
    if not normalized_cells:
        raise ValueError("At least one polyhedral cell is required.")
    vertex_ids = _resolved_entity_ids("vertex_global_ids", vertex_global_ids, vertices)
    cell_ids = _resolved_entity_ids(
        "cell_global_ids", cell_global_ids, len(normalized_cells)
    )

    face_keys: dict[tuple[int, ...], int] = {}
    canonical_faces: list[tuple[int, ...]] = []
    face_incidents: list[list[tuple[int, int, float]]] = []
    cell_face_rows: list[list[int]] = []
    cell_sign_rows: list[list[float]] = []
    cell_vertex_rows: list[tuple[int, ...]] = []
    for cell_index, faces in enumerate(normalized_cells):
        if len(faces) < 4:
            raise ValueError("A polyhedral cell requires at least four faces.")
        local_faces: list[int] = []
        local_signs: list[float] = []
        local_vertices: set[int] = set()
        shell_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for local_index, face_values in enumerate(faces):
            face = np.asarray(face_values, dtype=np.int32)
            if face.ndim != 1 or face.size < 3:
                raise ValueError("Polyhedral faces must be one-dimensional loops.")
            if np.any(face < 0) or np.any(face >= vertices):
                raise ValueError("Polyhedral faces index undeclared vertices.")
            if np.unique(face).size != face.size:
                raise ValueError("Each polyhedral face loop must be simple.")
            loop = tuple(int(value) for value in face)
            canonical, sign = _canonical_face_loop(loop)
            if canonical not in face_keys:
                face_keys[canonical] = len(canonical_faces)
                canonical_faces.append(canonical)
                face_incidents.append([])
            face_index = face_keys[canonical]
            if face_index in local_faces:
                raise ValueError("A polyhedral cell cannot repeat a face.")
            local_faces.append(face_index)
            local_signs.append(sign)
            face_incidents[face_index].append((cell_index, local_index, sign))
            local_vertices.update(loop)
            for index, start in enumerate(loop):
                stop = loop[(index + 1) % len(loop)]
                key = (min(start, stop), max(start, stop))
                shell_edges.setdefault(key, []).append((start, stop))
        if len(local_vertices) < 4:
            raise ValueError("A polyhedral cell requires at least four vertices.")
        for directions in shell_edges.values():
            if len(directions) != 2 or directions[0] != tuple(reversed(directions[1])):
                raise ValueError(
                    "Polyhedral face loops must form one closed oriented two-manifold."
                )
        cell_face_rows.append(local_faces)
        cell_sign_rows.append(local_signs)
        cell_vertex_rows.append(tuple(sorted(local_vertices)))

    if any(len(incidents) > 2 for incidents in face_incidents):
        raise ValueError("Polyhedral cells must be face-manifold.")
    if any(
        len(incidents) == 2 and incidents[0][2] == incidents[1][2]
        for incidents in face_incidents
    ):
        raise ValueError("Shared polyhedral faces must have opposite orientation.")

    maximum_face_arity = max(len(face) for face in canonical_faces)
    maximum_cell_faces = max(len(faces) for faces in cell_face_rows)
    maximum_cell_vertices = max(len(row) for row in cell_vertex_rows)
    face_count = len(canonical_faces)
    cell_count = len(normalized_cells)
    face_vertices = np.zeros((face_count, maximum_face_arity), dtype=np.int32)
    face_vertex_valid = np.zeros_like(face_vertices, dtype=bool)
    for face_index, face in enumerate(canonical_faces):
        face_vertices[face_index, : len(face)] = face
        face_vertex_valid[face_index, : len(face)] = True

    cell_faces = np.zeros((cell_count, maximum_cell_faces), dtype=np.int32)
    cell_face_signs = np.zeros((cell_count, maximum_cell_faces), dtype=float)
    cell_face_valid = np.zeros_like(cell_faces, dtype=bool)
    for cell_index, (faces, signs) in enumerate(
        zip(cell_face_rows, cell_sign_rows, strict=True)
    ):
        cell_faces[cell_index, : len(faces)] = faces
        cell_face_signs[cell_index, : len(signs)] = signs
        cell_face_valid[cell_index, : len(faces)] = True
    cell_vertices = np.zeros((cell_count, maximum_cell_vertices), dtype=np.int32)
    cell_vertex_valid = np.zeros_like(cell_vertices, dtype=bool)
    for cell_index, row in enumerate(cell_vertex_rows):
        cell_vertices[cell_index, : len(row)] = row
        cell_vertex_valid[cell_index, : len(row)] = True

    edge_keys: dict[tuple[int, int], int] = {}
    for face in canonical_faces:
        for index, start in enumerate(face):
            stop = face[(index + 1) % len(face)]
            key = (min(start, stop), max(start, stop))
            edge_keys.setdefault(key, len(edge_keys))
    edges = np.asarray(tuple(edge_keys), dtype=np.int32)
    face_edges = np.zeros((face_count, maximum_face_arity), dtype=np.int32)
    face_edge_signs = np.zeros((face_count, maximum_face_arity), dtype=float)
    face_edge_valid = np.zeros_like(face_edges, dtype=bool)
    for face_index, face in enumerate(canonical_faces):
        for local, start in enumerate(face):
            stop = face[(local + 1) % len(face)]
            key = (min(start, stop), max(start, stop))
            face_edges[face_index, local] = edge_keys[key]
            face_edge_signs[face_index, local] = 1.0 if (start, stop) == key else -1.0
            face_edge_valid[face_index, local] = True

    owner = np.full((face_count,), -1, dtype=np.int32)
    neighbour = np.full((face_count,), -1, dtype=np.int32)
    owner_local = np.full((face_count,), -1, dtype=np.int32)
    neighbour_local = np.full((face_count,), -1, dtype=np.int32)
    counts = np.asarray([len(value) for value in face_incidents], dtype=np.int32)
    for face_index, incidents in enumerate(face_incidents):
        owner[face_index], owner_local[face_index], _ = incidents[0]
        if len(incidents) == 2:
            neighbour[face_index], neighbour_local[face_index], _ = incidents[1]
    boundary_faces = counts == 1
    boundary_edges = np.zeros((edges.shape[0],), dtype=bool)
    boundary_face_edges = face_edges[boundary_faces]
    boundary_face_edge_valid = face_edge_valid[boundary_faces]
    boundary_edges[np.unique(boundary_face_edges[boundary_face_edge_valid])] = True
    boundary_vertices = np.zeros((vertices,), dtype=bool)
    boundary_face_vertices = face_vertices[boundary_faces]
    boundary_face_vertex_valid = face_vertex_valid[boundary_faces]
    boundary_vertices[np.unique(boundary_face_vertices[boundary_face_vertex_valid])] = (
        True
    )

    edge_ids = _canonical_entity_ids(np.sort(vertex_ids[edges], axis=1))
    global_face_keys = tuple(
        tuple(sorted(int(vertex_ids[value]) for value in face))
        for face in canonical_faces
    )
    face_ids = _canonical_variable_ids(global_face_keys)
    return PolyhedralConnectivity(
        edges=jnp.asarray(edges),
        faces=jnp.asarray(np.where(face_vertex_valid, face_vertices, -1)),
        face_arities=jnp.asarray(np.sum(face_vertex_valid, axis=1), dtype=jnp.int32),
        face_vertex_valid=jnp.asarray(face_vertex_valid),
        face_edges=jnp.asarray(face_edges),
        face_edge_signs=jnp.asarray(face_edge_signs),
        face_edge_valid=jnp.asarray(face_edge_valid),
        cell_faces=jnp.asarray(cell_faces),
        cell_face_signs=jnp.asarray(cell_face_signs),
        cell_face_valid=jnp.asarray(cell_face_valid),
        cell_vertices=jnp.asarray(cell_vertices),
        cell_vertex_valid=jnp.asarray(cell_vertex_valid),
        face_owner=jnp.asarray(owner),
        face_neighbour=jnp.asarray(neighbour),
        face_owner_local=jnp.asarray(owner_local),
        face_neighbour_local=jnp.asarray(neighbour_local),
        face_cell_counts=jnp.asarray(counts),
        boundary_vertices=jnp.asarray(boundary_vertices),
        boundary_edges=jnp.asarray(boundary_edges),
        boundary_faces=jnp.asarray(boundary_faces),
        vertex_global_ids=jnp.asarray(vertex_ids),
        edge_global_ids=jnp.asarray(edge_ids),
        face_global_ids=jnp.asarray(face_ids),
        cell_global_ids=jnp.asarray(cell_ids),
        vertex_count=vertices,
        edge_count=int(edges.shape[0]),
        face_count=face_count,
        cell_count=cell_count,
        maximum_face_arity=maximum_face_arity,
        maximum_cell_faces=maximum_cell_faces,
        maximum_cell_vertices=maximum_cell_vertices,
    )


def polyhedral_cell_complex(
    value: (
        PolyhedralConnectivity
        | Sequence[Sequence[ArrayLike]]
        | Sequence[tuple[str, ArrayLike]]
    ),
    vertex_count: int | None = None,
    /,
    *,
    vertex_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
) -> CellComplexTopology:
    """Build the validated 0→1→2→3 complex for polyhedral cells."""

    if isinstance(value, PolyhedralConnectivity):
        if (
            vertex_count is not None
            or vertex_global_ids is not None
            or cell_global_ids is not None
        ):
            raise ValueError(
                "Connectivity-backed polyhedral topology already owns its entity IDs."
            )
        connectivity = value
    else:
        if vertex_count is None:
            raise TypeError("vertex_count is required for polyhedral cell definitions.")
        connectivity = polyhedral_connectivity(
            value,
            vertex_count,
            vertex_global_ids=vertex_global_ids,
            cell_global_ids=cell_global_ids,
        )
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    face_edges = np.asarray(connectivity.face_edges, dtype=np.int32)
    face_edge_valid = np.asarray(connectivity.face_edge_valid, dtype=bool)
    cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int32)
    cell_face_valid = np.asarray(connectivity.cell_face_valid, dtype=bool)
    vertex_entities = EntitySet(
        "vertices",
        0,
        connectivity.vertex_global_ids,
        subsets=(EntitySubset("boundary", connectivity.boundary_vertices),),
    )
    edge_entities = EntitySet(
        "edges",
        1,
        connectivity.edge_global_ids,
        subsets=(EntitySubset("boundary", connectivity.boundary_edges),),
    )
    face_entities = EntitySet(
        "faces",
        2,
        connectivity.face_global_ids,
        subsets=(EntitySubset("boundary", connectivity.boundary_faces),),
    )
    cell_entities = EntitySet(
        "cells",
        3,
        connectivity.cell_global_ids,
        subsets=(
            EntitySubset("boundary", np.zeros((connectivity.cell_count,), dtype=bool)),
        ),
    )
    vertex_edge_relation = EdgeRelation(
        edges.reshape((-1,)),
        np.repeat(np.arange(connectivity.edge_count, dtype=np.int32), 2),
        source_size=connectivity.vertex_count,
        target_size=connectivity.edge_count,
    )
    edge_face_relation = EdgeRelation(
        face_edges[face_edge_valid],
        np.repeat(
            np.arange(connectivity.face_count, dtype=np.int32),
            np.sum(face_edge_valid, axis=1),
        ),
        source_size=connectivity.edge_count,
        target_size=connectivity.face_count,
    )
    face_cell_relation = EdgeRelation(
        cell_faces[cell_face_valid],
        np.repeat(
            np.arange(connectivity.cell_count, dtype=np.int32),
            np.sum(cell_face_valid, axis=1),
        ),
        source_size=connectivity.face_count,
        target_size=connectivity.cell_count,
    )
    return CellComplexTopology(
        (vertex_entities, edge_entities, face_entities, cell_entities),
        (
            OrientedIncidence(
                1,
                vertex_entities,
                edge_entities,
                vertex_edge_relation,
                np.tile(np.asarray([-1.0, 1.0]), connectivity.edge_count),
            ),
            OrientedIncidence(
                2,
                edge_entities,
                face_entities,
                edge_face_relation,
                np.asarray(connectivity.face_edge_signs)[face_edge_valid],
            ),
            OrientedIncidence(
                3,
                face_entities,
                cell_entities,
                face_cell_relation,
                np.asarray(connectivity.cell_face_signs)[cell_face_valid],
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
    "IntervalConnectivity",
    "interval_cell_complex",
    "interval_connectivity",
    "PolyhedralConnectivity",
    "TetrahedralConnectivity",
    "polygonal_cell_complex",
    "polygonal_connectivity",
    "polyhedral_cell_complex",
    "polyhedral_connectivity",
    "tetrahedral_cell_complex",
    "tetrahedral_connectivity",
]
