#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation
from ._topology import (
    _has_duplicate_rows,
    _row_order,
    _row_run_starts,
    CellComplexTopology,
    EntitySet,
    EntitySubset,
    OrientedIncidence,
)


def _first_appearance_groups(rows: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Group equal integer rows and number the groups by first appearance.

    Returns the group of every row and, per group, the increasing index of the
    row where it first appears. The stable sort makes this numbering identical
    to a sequential dictionary scan over the rows.
    """

    order = _row_order(rows)
    starts = _row_run_starts(rows[order])
    first = order[starts]
    ranking = np.argsort(first)
    run_groups = np.empty((first.size,), dtype=np.int64)
    run_groups[ranking] = np.arange(first.size, dtype=np.int64)
    groups = np.empty((rows.shape[0],), dtype=np.int64)
    groups[order] = run_groups[np.cumsum(starts) - 1]
    return groups, first[ranking]


def _variable_row_ranks(offsets: np.ndarray, values: np.ndarray, /) -> np.ndarray:
    """Rank CSR integer rows in lexicographic tuple order.

    Equal rows share one rank and a row precedes every longer row it prefixes,
    exactly as Python tuples compare. A rank is the first sorted position of its
    row group. Each refinement pass sorts only rows still tied with another row
    on the columns seen so far, so the passes are bounded by the longest tied
    prefix and the total work by the entry count.
    """

    lengths = np.diff(offsets)
    ranks = np.zeros((lengths.size,), dtype=np.int64)
    active = np.arange(lengths.size, dtype=np.int64)
    # One below the smallest value marks an exhausted row, which sorts first.
    exhausted = int(np.min(values, initial=0)) - 1
    depth = 0
    while active.size > 1:
        present = lengths[active] > depth
        column = np.full((active.size,), exhausted, dtype=np.int64)
        column[present] = values[offsets[active[present]] + depth]
        order = _row_order(np.stack((ranks[active], column), axis=1))
        active = active[order]
        column = column[order]
        tied = ranks[active]
        position = np.arange(active.size, dtype=np.int64)
        group = np.ones((active.size,), dtype=np.bool_)
        group[1:] = tied[1:] != tied[:-1]
        split = group.copy()
        split[1:] |= column[1:] != column[:-1]
        ranks[active] = tied + (
            np.maximum.accumulate(np.where(split, position, 0))
            - np.maximum.accumulate(np.where(group, position, 0))
        )
        split_ids = np.cumsum(split) - 1
        still_tied = np.bincount(split_ids)[split_ids] > 1
        active = active[still_tied & (column != exhausted)]
        depth += 1
    return ranks


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
    ordered = np.sort(cells, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError(f"{name} must contain distinct vertices per cell.")
    if _has_duplicate_rows(ordered):
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
        cells.shape[0],
    )


def interval_cell_complex(
    intervals: ArrayLike,
    vertex_count: int,
    /,
    *,
    vertex_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
) -> CellComplexTopology:
    return _interval_complex(
        interval_connectivity(intervals, vertex_count),
        vertex_global_ids=vertex_global_ids,
        cell_global_ids=cell_global_ids,
    )


def _interval_complex(
    connectivity: IntervalConnectivity,
    /,
    *,
    vertex_global_ids: ArrayLike | None,
    cell_global_ids: ArrayLike | None,
) -> CellComplexTopology:
    cells = np.asarray(connectivity.cell_vertices, dtype=np.int32)
    vertex_ids = _resolved_entity_ids(
        "vertex_global_ids", vertex_global_ids, connectivity.vertex_count
    )
    cell_ids = _resolved_entity_ids("cell_global_ids", cell_global_ids, cells.shape[0])
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
        subsets=(EntitySubset("boundary", np.zeros((cells.shape[0],), dtype=np.bool_)),),
    )
    relation = EdgeRelation(
        cells.reshape((-1,)),
        np.repeat(np.arange(cells.shape[0], dtype=np.int32), 2),
        source_size=connectivity.vertex_count,
        target_size=cells.shape[0],
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
    """Canonical packed oriented incidence for mixed or face-defined polyhedra."""

    edges: Array
    face_vertex_offsets: Array
    face_vertex_values: Array
    face_edge_offsets: Array
    face_edge_values: Array
    face_edge_sign_values: Array
    cell_face_offsets: Array
    cell_face_values: Array
    cell_face_sign_values: Array
    cell_vertex_offsets: Array
    cell_vertex_values: Array
    face_owner: Array
    face_neighbor: Array
    face_owner_local: Array
    face_neighbor_local: Array
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


class PolyhedralWorksetLimitError(ValueError):
    """Raised before a dense polyhedral workset exceeds its declared capacity."""


class PolyhedralWorksets(StrictModule, NonTrainableState):
    """Bounded dense worksets prepared from packed polyhedral incidence."""

    face_vertices: Array
    face_vertex_valid: Array
    face_edges: Array
    face_edge_signs: Array
    face_edge_valid: Array
    cell_faces: Array
    cell_face_signs: Array
    cell_face_valid: Array
    cell_vertices: Array
    cell_vertex_valid: Array
    allocated_entries: int = eqx.field(static=True)


def _dense_rows(
    offsets: np.ndarray,
    values: np.ndarray,
    width: int,
    /,
) -> np.ndarray:
    widths = np.diff(offsets)
    dense = np.zeros((widths.size, width), dtype=values.dtype)
    dense[np.arange(width)[None, :] < widths[:, None]] = values[offsets[0] : offsets[-1]]
    return dense


def prepare_polyhedral_worksets(
    connectivity: PolyhedralConnectivity,
    /,
    *,
    maximum_entries: int,
) -> PolyhedralWorksets:
    """Materialize dense incidence only within an aggregate entry budget.

    The budget includes every index, sign, and validity entry in the returned
    worksets. Capacity is checked before allocating any dense row arrays.
    """

    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("connectivity must be PolyhedralConnectivity.")
    limit = int(maximum_entries)
    if limit <= 0:
        raise ValueError("maximum_entries must be positive.")
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int32)
    face_edge_offsets = np.asarray(connectivity.face_edge_offsets, dtype=np.int32)
    cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int32)
    vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int32)
    face_widths = np.diff(face_offsets)
    edge_widths = np.diff(face_edge_offsets)
    cell_widths = np.diff(cell_offsets)
    vertex_widths = np.diff(vertex_offsets)
    face_width = int(np.max(face_widths, initial=0))
    edge_width = int(np.max(edge_widths, initial=0))
    cell_width = int(np.max(cell_widths, initial=0))
    vertex_width = int(np.max(vertex_widths, initial=0))
    entries = (
        2 * face_widths.size * face_width
        + 3 * edge_widths.size * edge_width
        + 3 * cell_widths.size * cell_width
        + 2 * vertex_widths.size * vertex_width
    )
    if entries > limit:
        raise PolyhedralWorksetLimitError(
            f"Dense polyhedral worksets require {entries} entries; limit is {limit}."
        )
    return PolyhedralWorksets(
        face_vertices=jnp.asarray(
            _dense_rows(
                face_offsets,
                np.asarray(connectivity.face_vertex_values, dtype=np.int32),
                face_width,
            )
        ),
        face_vertex_valid=jnp.asarray(
            np.arange(face_width)[None, :] < face_widths[:, None]
        ),
        face_edges=jnp.asarray(
            _dense_rows(
                face_edge_offsets,
                np.asarray(connectivity.face_edge_values, dtype=np.int32),
                edge_width,
            )
        ),
        face_edge_signs=jnp.asarray(
            _dense_rows(
                face_edge_offsets,
                np.asarray(connectivity.face_edge_sign_values),
                edge_width,
            )
        ),
        face_edge_valid=jnp.asarray(
            np.arange(edge_width)[None, :] < edge_widths[:, None]
        ),
        cell_faces=jnp.asarray(
            _dense_rows(
                cell_offsets,
                np.asarray(connectivity.cell_face_values, dtype=np.int32),
                cell_width,
            )
        ),
        cell_face_signs=jnp.asarray(
            _dense_rows(
                cell_offsets,
                np.asarray(connectivity.cell_face_sign_values),
                cell_width,
            )
        ),
        cell_face_valid=jnp.asarray(
            np.arange(cell_width)[None, :] < cell_widths[:, None]
        ),
        cell_vertices=jnp.asarray(
            _dense_rows(
                vertex_offsets,
                np.asarray(connectivity.cell_vertex_values, dtype=np.int32),
                vertex_width,
            )
        ),
        cell_vertex_valid=jnp.asarray(
            np.arange(vertex_width)[None, :] < vertex_widths[:, None]
        ),
        allocated_entries=entries,
    )


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
                array.shape[1],
                vertices,
            )
        )
    blocks = (triangle_cells, quadrilateral_cells, *polygon_cells)
    if sum(block.shape[0] for block in blocks) == 0:
        raise ValueError("At least one polygonal cell is required.")
    for arity in sorted({block.shape[1] for block in blocks}):
        same_arity = tuple(block for block in blocks if block.shape[1] == arity)
        # `_validated_cells` already rejects duplicates inside one block.
        if len(same_arity) > 1 and _has_duplicate_rows(
            np.sort(np.concatenate(same_arity, axis=0), axis=1)
        ):
            raise ValueError("Polygonal connectivity contains duplicate cells.")

    capacity = max(4, *(block.shape[1] for block in blocks))
    cell_count = sum(block.shape[0] for block in blocks)
    cell_vertices = np.zeros((cell_count, capacity), dtype=np.int32)
    cell_valid = np.zeros((cell_count, capacity), dtype=np.bool_)
    cell_kinds = np.empty((cell_count,), dtype=np.int32)
    offset = 0
    for block in blocks:
        count, arity = block.shape
        cell_vertices[offset : offset + count, :arity] = block
        cell_valid[offset : offset + count, :arity] = True
        cell_kinds[offset : offset + count] = arity
        offset += count

    # Row-major valid sides enumerate (cell, local side) in scan order, so the
    # first-appearance edge numbering matches a sequential cell traversal.
    following = np.arange(1, capacity + 1, dtype=np.int32)[None, :]
    following = np.where(following < cell_kinds[:, None], following, 0)
    starts = cell_vertices[cell_valid]
    stops = np.take_along_axis(cell_vertices, following, axis=1)[cell_valid]
    side_keys = np.stack((np.minimum(starts, stops), np.maximum(starts, stops)), axis=1)
    side_edges, first_sides = _first_appearance_groups(side_keys)
    side_signs = np.where(starts < stops, 1.0, -1.0)
    counts = np.bincount(side_edges, minlength=first_sides.size).astype(np.int32)
    if np.any(counts > 2):
        raise ValueError("Polygonal cells must be edge-manifold.")
    orientation = np.bincount(side_edges, weights=side_signs, minlength=first_sides.size)
    if np.any((counts == 2) & (np.abs(orientation) == 2.0)):
        raise ValueError("Shared polygon edges must have opposite orientation.")

    edges = side_keys[first_sides]
    cell_edges = np.zeros((cell_count, capacity), dtype=np.int32)
    cell_edges[cell_valid] = side_edges
    cell_signs = np.zeros((cell_count, capacity), dtype=np.float64)
    cell_signs[cell_valid] = side_signs
    boundary_edges = counts == 1
    boundary_vertices = np.zeros((vertices,), dtype=np.bool_)
    boundary_vertices[edges[boundary_edges].reshape((-1,))] = True
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
    edge_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
) -> CellComplexTopology:
    return _polygonal_complex(
        polygonal_connectivity(
            triangles,
            quadrilaterals,
            vertex_count,
            polygons=polygons,
        ),
        vertex_global_ids=vertex_global_ids,
        edge_global_ids=edge_global_ids,
        cell_global_ids=cell_global_ids,
    )


def _polygonal_complex(
    connectivity: PolygonalConnectivity,
    /,
    *,
    vertex_global_ids: ArrayLike | None,
    edge_global_ids: ArrayLike | None,
    cell_global_ids: ArrayLike | None,
) -> CellComplexTopology:
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
    cell_valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
    cell_signs = np.asarray(connectivity.cell_edge_signs)
    vertex_ids = _resolved_entity_ids(
        "vertex_global_ids", vertex_global_ids, connectivity.vertex_count
    )
    cell_ids_global = _resolved_entity_ids(
        "cell_global_ids", cell_global_ids, connectivity.cell_count
    )
    edge_keys = np.sort(vertex_ids[edges], axis=1)
    edge_ids_global = (
        _canonical_entity_ids(edge_keys)
        if edge_global_ids is None
        else _resolved_entity_ids("edge_global_ids", edge_global_ids, edges.shape[0])
    )
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
            EntitySubset(
                "boundary", np.zeros((connectivity.cell_count,), dtype=np.bool_)
            ),
        ),
    )
    vertex_edge_relation = EdgeRelation(
        edges.reshape((-1,)),
        np.repeat(np.arange(edges.shape[0], dtype=np.int32), 2),
        source_size=connectivity.vertex_count,
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


def _csr_offsets(counts: np.ndarray, /) -> np.ndarray:
    offsets = np.zeros((counts.size + 1,), dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    return offsets


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

# Face-loop defects in the order one sequential scan checks a loop.
_POLYHEDRAL_LOOP_DEFECTS = (
    "Polyhedral faces must be one-dimensional loops.",
    "Polyhedral faces index undeclared vertices.",
    "Each polyhedral face loop must be simple.",
    "A polyhedral cell cannot repeat a face.",
)


def _polyhedral_face_loops(
    cells: Sequence[Sequence[ArrayLike] | tuple[str, ArrayLike]],
    vertex_count: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten standard cells and explicit face loops into ordered loop occurrences.

    Returns per-cell face counts, per-loop lengths, the explicit-loop mask, and
    the concatenated loop vertices. Length ``-1`` marks an explicit loop that is
    not a rank-1 array of at least three vertices; it contributes no vertices.
    Every standard block is validated before any explicit loop is converted.
    """

    entries = tuple(cells)
    standard: dict[int, tuple[str, np.ndarray]] = {}
    explicit: dict[int, tuple[ArrayLike, ...]] = {}
    for index, entry in enumerate(entries):
        descriptor = (
            isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str)
        )
        if not descriptor:
            # ty: ignore[invalid-assignment]
            explicit[index] = tuple(entry)
            continue
        kind_value, values = entry
        kind = str(kind_value)
        if kind not in _POLYHEDRAL_FACE_ROUTES:
            raise ValueError(f"Unsupported polyhedral cell kind {kind!r}.")
        standard[index] = (
            kind,
            _validated_cells(kind, values, _POLYHEDRAL_CELL_ARITIES[kind], vertex_count),
        )
    face_counts: list[np.ndarray] = []
    loop_lengths: list[np.ndarray] = []
    explicit_loops: list[np.ndarray] = []
    loop_values: list[np.ndarray] = []
    for index in range(len(entries)):
        if index in standard:
            kind, block = standard[index]
            routes = _POLYHEDRAL_FACE_ROUTES[kind]
            face_counts.append(np.full((block.shape[0],), len(routes), dtype=np.int64))
            loop_lengths.append(
                np.tile(
                    np.asarray([len(route) for route in routes], dtype=np.int64),
                    block.shape[0],
                )
            )
            explicit_loops.append(
                np.zeros((block.shape[0] * len(routes),), dtype=np.bool_)
            )
            loop_values.append(block[:, np.concatenate(routes)].reshape((-1,)))
            continue
        # Explicit loops are host input adaptation: one conversion per loop.
        loops = tuple(np.asarray(face, dtype=np.int32) for face in explicit[index])
        lengths = np.asarray(
            [loop.size if loop.ndim == 1 and loop.size >= 3 else -1 for loop in loops],
            dtype=np.int64,
        )
        face_counts.append(np.asarray((len(loops),), dtype=np.int64))
        loop_lengths.append(lengths)
        explicit_loops.append(np.ones((len(loops),), dtype=np.bool_))
        loop_values.extend(
            loop for loop, length in zip(loops, lengths, strict=True) if length > 0
        )
    return (
        np.concatenate(face_counts) if face_counts else np.empty((0,), np.int64),
        np.concatenate(loop_lengths) if loop_lengths else np.empty((0,), np.int64),
        np.concatenate(explicit_loops) if explicit_loops else np.empty((0,), np.bool_),
        np.concatenate(loop_values) if loop_values else np.empty((0,), np.int32),
    )


def _canonical_face_loops(
    offsets: np.ndarray,
    values: np.ndarray,
    entry_loops: np.ndarray,
    positions: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical loop vertices and loop orientation signs.

    Each loop starts at its first smallest vertex and runs toward the smaller of
    that vertex's two neighbors, which is the lexicographically smaller of its
    two traversals from there. A loop and its reverse share one canonical loop
    with opposite signs.
    """

    lengths = np.diff(offsets)
    formed = np.flatnonzero(lengths > 0)
    starts = offsets[formed]
    first = np.zeros((lengths.size,), dtype=np.int64)
    signs = np.ones((lengths.size,), dtype=np.int64)
    if formed.size:
        minimum = np.repeat(np.minimum.reduceat(values, starts), lengths[formed])
        candidates = np.where(values == minimum, positions, np.iinfo(np.int64).max)
        first[formed] = np.minimum.reduceat(candidates, starts)
        sizes = lengths[formed]
        successor = values[starts + (first[formed] + 1) % sizes]
        predecessor = values[starts + (first[formed] - 1) % sizes]
        signs[formed] = np.where(successor < predecessor, 1, -1)
    canonical = values[
        offsets[entry_loops]
        + (first[entry_loops] + signs[entry_loops] * positions) % lengths[entry_loops]
    ]
    return canonical, signs


def _canonical_loop_faces(
    offsets: np.ndarray, canonical: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Group equal canonical loops into faces numbered by first appearance.

    Loops of different lengths never coincide, so every distinct length groups
    as one dense block; returns each loop's face and each face's first loop.
    """

    lengths = np.diff(offsets)
    representatives = np.empty((lengths.size,), dtype=np.int64)
    for length in np.unique(lengths):
        loops = np.flatnonzero(lengths == length)
        rows = canonical[offsets[loops][:, None] + np.arange(length)]
        groups, first = _first_appearance_groups(rows)
        representatives[loops] = loops[first][groups]
    return _first_appearance_groups(representatives[:, None])


def _cell_vertex_rows(
    entry_cells: np.ndarray, values: np.ndarray, cell_count: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-cell distinct vertex counts and their sorted packed values."""

    pairs = np.stack((entry_cells, values), axis=1)
    ordered = pairs[_row_order(pairs)]
    distinct = ordered[_row_run_starts(ordered)]
    return (
        np.bincount(distinct[:, 0], minlength=cell_count),
        distinct[:, 1].astype(np.int32),
    )


def _polyhedral_loop_defects(
    lengths: np.ndarray,
    explicit: np.ndarray,
    loop_cells: np.ndarray,
    loop_faces: np.ndarray,
    values: np.ndarray,
    entry_loops: np.ndarray,
    vertex_count: int,
    /,
) -> np.ndarray:
    """Return per-loop defect codes: zero or one plus the defect's scan index."""

    count = lengths.size
    outside = (values < 0) | (values >= vertex_count)
    outside_loops = np.bincount(entry_loops[outside], minlength=count) > 0
    # Validated standard blocks already reference distinct vertices.
    checked = explicit[entry_loops]
    pairs = np.stack((entry_loops[checked], values[checked]), axis=1)
    ordered = pairs[_row_order(pairs)]
    repeated_vertices = np.all(ordered[1:] == ordered[:-1], axis=1)
    non_simple = np.bincount(ordered[1:, 0][repeated_vertices], minlength=count) > 0
    faces = np.stack((loop_cells, loop_faces), axis=1)
    order = _row_order(faces)
    ordered_faces = faces[order]
    repeated_faces = np.zeros((count,), dtype=np.bool_)
    repeated_faces[order[1:][np.all(ordered_faces[1:] == ordered_faces[:-1], axis=1)]] = (
        True
    )
    return np.select(
        (lengths < 0, outside_loops, non_simple, repeated_faces),
        (1, 2, 3, 4),
        default=0,
    )


def _open_shell_cells(
    loop_cells: np.ndarray,
    offsets: np.ndarray,
    values: np.ndarray,
    entry_loops: np.ndarray,
    positions: np.ndarray,
    cell_count: int,
    /,
) -> np.ndarray:
    """Mark cells whose loops do not traverse each edge once in each direction."""

    sizes = np.diff(offsets)
    stops = values[offsets[entry_loops] + (positions + 1) % sizes[entry_loops]]
    sides = np.stack(
        (loop_cells[entry_loops], np.minimum(values, stops), np.maximum(values, stops)),
        axis=1,
    )
    order = _row_order(sides)
    first = np.flatnonzero(_row_run_starts(sides[order]))
    run_lengths = np.diff(np.append(first, order.size))
    second = np.minimum(first + 1, order.size - 1)
    opposed = (run_lengths == 2) & (values[order[first]] != values[order[second]])
    return np.bincount(sides[order[first[~opposed]], 0], minlength=cell_count) > 0


def _first_polyhedral_cell_defect(
    cell_face_counts: np.ndarray,
    cell_vertex_counts: np.ndarray,
    loop_cells: np.ndarray,
    loop_codes: np.ndarray,
    open_shells: np.ndarray,
    /,
) -> str | None:
    """Return the defect a sequential scan over the cells reports first.

    The scan checks each cell in order: its face count, every face loop in local
    order, its distinct vertex count, and finally its closed oriented shell.
    """

    defective_loops = loop_codes > 0
    face_defects = (
        np.bincount(loop_cells[defective_loops], minlength=cell_face_counts.size) > 0
    )
    defective = (
        (cell_face_counts < 4) | face_defects | (cell_vertex_counts < 4) | open_shells
    )
    if not np.any(defective):
        return None
    cell = np.argmax(defective)
    if cell_face_counts[cell] < 4:
        return "A polyhedral cell requires at least four faces."
    if face_defects[cell]:
        loop = np.argmax(defective_loops & (loop_cells == cell))
        return _POLYHEDRAL_LOOP_DEFECTS[loop_codes[loop] - 1]
    if cell_vertex_counts[cell] < 4:
        return "A polyhedral cell requires at least four vertices."
    return "Polyhedral face loops must form one closed oriented two-manifold."


def polyhedral_connectivity(
    cells: Sequence[Sequence[ArrayLike] | tuple[str, ArrayLike]],
    vertex_count: int,
    /,
    *,
    vertex_global_ids: ArrayLike | None = None,
    edge_global_ids: ArrayLike | None = None,
    face_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
) -> PolyhedralConnectivity:
    """Build exact oriented incidence from standard cells or explicit face loops."""

    vertices = int(vertex_count)
    if vertices <= 0:
        raise ValueError("vertex_count must be positive.")
    cell_face_counts, loop_lengths, explicit_loops, loop_values = _polyhedral_face_loops(
        cells, vertices
    )
    cell_count = cell_face_counts.size
    if cell_count == 0:
        raise ValueError("At least one polyhedral cell is required.")
    vertex_ids = _resolved_entity_ids("vertex_global_ids", vertex_global_ids, vertices)
    cell_ids = _resolved_entity_ids("cell_global_ids", cell_global_ids, cell_count)

    loop_sizes = np.maximum(loop_lengths, 0)
    loop_offsets = np.zeros((loop_sizes.size + 1,), dtype=np.int64)
    np.cumsum(loop_sizes, out=loop_offsets[1:])
    loop_cells = np.repeat(np.arange(cell_count, dtype=np.int64), cell_face_counts)
    entry_loops = np.repeat(np.arange(loop_sizes.size, dtype=np.int64), loop_sizes)
    entry_positions = (
        np.arange(loop_values.size, dtype=np.int64) - loop_offsets[entry_loops]
    )
    canonical, loop_signs = _canonical_face_loops(
        loop_offsets, loop_values, entry_loops, entry_positions
    )
    loop_faces, face_loops = _canonical_loop_faces(loop_offsets, canonical)
    cell_vertex_counts, cell_vertex_values = _cell_vertex_rows(
        loop_cells[entry_loops], loop_values, cell_count
    )
    defect = _first_polyhedral_cell_defect(
        cell_face_counts,
        cell_vertex_counts,
        loop_cells,
        _polyhedral_loop_defects(
            loop_lengths,
            explicit_loops,
            loop_cells,
            loop_faces,
            loop_values,
            entry_loops,
            vertices,
        ),
        _open_shell_cells(
            loop_cells,
            loop_offsets,
            loop_values,
            entry_loops,
            entry_positions,
            cell_count,
        ),
    )
    if defect is not None:
        raise ValueError(defect)

    face_count = face_loops.size
    counts = np.bincount(loop_faces, minlength=face_count).astype(np.int32)
    if np.any(counts > 2):
        raise ValueError("Polyhedral cells must be face-manifold.")
    cell_face_sign_values = loop_signs.astype(np.float64)
    orientation = np.bincount(
        loop_faces, weights=cell_face_sign_values, minlength=face_count
    )
    if np.any((counts == 2) & (np.abs(orientation) == 2.0)):
        raise ValueError("Shared polyhedral faces must have opposite orientation.")

    face_sizes = loop_sizes[face_loops]
    face_vertex_offsets = _csr_offsets(face_sizes)
    face_entries = np.repeat(np.arange(face_count, dtype=np.int64), face_sizes)
    face_positions = (
        np.arange(face_entries.size, dtype=np.int64) - face_vertex_offsets[face_entries]
    )
    face_vertex_values = canonical[
        loop_offsets[face_loops][face_entries] + face_positions
    ]
    following = face_vertex_values[
        face_vertex_offsets[face_entries]
        + (face_positions + 1) % face_sizes[face_entries]
    ]
    side_keys = np.stack(
        (
            np.minimum(face_vertex_values, following),
            np.maximum(face_vertex_values, following),
        ),
        axis=1,
    )
    face_edge_groups, edge_sides = _first_appearance_groups(side_keys)
    edges = side_keys[edge_sides]
    face_edge_values = face_edge_groups.astype(np.int32)
    face_edge_sign_values = np.where(face_vertex_values < following, 1.0, -1.0)

    cell_face_offsets = _csr_offsets(cell_face_counts)
    loop_locals = (
        np.arange(loop_faces.size, dtype=np.int64) - cell_face_offsets[loop_cells]
    )
    face_occurrence_starts = _csr_offsets(counts)[:-1]
    face_occurrences = np.argsort(loop_faces, kind="stable")
    owner_loops = face_occurrences[face_occurrence_starts]
    shared = counts == 2
    neighbor_loops = face_occurrences[face_occurrence_starts[shared] + 1]
    owner = loop_cells[owner_loops].astype(np.int32)
    owner_local = loop_locals[owner_loops].astype(np.int32)
    neighbor = np.full((face_count,), -1, dtype=np.int32)
    neighbor[shared] = loop_cells[neighbor_loops]
    neighbor_local = np.full((face_count,), -1, dtype=np.int32)
    neighbor_local[shared] = loop_locals[neighbor_loops]
    boundary_faces = counts == 1
    boundary_sides = np.repeat(boundary_faces, face_sizes)
    boundary_edges = np.zeros((edges.shape[0],), dtype=np.bool_)
    boundary_edges[face_edge_values[boundary_sides]] = True
    boundary_vertices = np.zeros((vertices,), dtype=np.bool_)
    boundary_vertices[face_vertex_values[boundary_sides]] = True

    edge_ids = (
        _canonical_entity_ids(np.sort(vertex_ids[edges], axis=1))
        if edge_global_ids is None
        else _resolved_entity_ids("edge_global_ids", edge_global_ids, edges.shape[0])
    )
    global_vertices = vertex_ids[face_vertex_values]
    face_keys = global_vertices[
        _row_order(np.stack((face_entries, global_vertices), axis=1))
    ]
    face_ids = (
        _canonical_entity_ids(
            _variable_row_ranks(face_vertex_offsets, face_keys)[:, None]
        )
        if face_global_ids is None
        else _resolved_entity_ids("face_global_ids", face_global_ids, face_count)
    )
    return PolyhedralConnectivity(
        edges=jnp.asarray(edges),
        face_vertex_offsets=jnp.asarray(face_vertex_offsets),
        face_vertex_values=jnp.asarray(face_vertex_values),
        face_edge_offsets=jnp.asarray(face_vertex_offsets),
        face_edge_values=jnp.asarray(face_edge_values),
        face_edge_sign_values=jnp.asarray(face_edge_sign_values),
        cell_face_offsets=jnp.asarray(cell_face_offsets),
        cell_face_values=jnp.asarray(loop_faces.astype(np.int32)),
        cell_face_sign_values=jnp.asarray(cell_face_sign_values),
        cell_vertex_offsets=jnp.asarray(_csr_offsets(cell_vertex_counts)),
        cell_vertex_values=jnp.asarray(cell_vertex_values),
        face_owner=jnp.asarray(owner),
        face_neighbor=jnp.asarray(neighbor),
        face_owner_local=jnp.asarray(owner_local),
        face_neighbor_local=jnp.asarray(neighbor_local),
        face_cell_counts=jnp.asarray(counts),
        boundary_vertices=jnp.asarray(boundary_vertices),
        boundary_edges=jnp.asarray(boundary_edges),
        boundary_faces=jnp.asarray(boundary_faces),
        vertex_global_ids=jnp.asarray(vertex_ids),
        edge_global_ids=jnp.asarray(edge_ids),
        face_global_ids=jnp.asarray(face_ids),
        cell_global_ids=jnp.asarray(cell_ids),
        vertex_count=vertices,
        edge_count=edges.shape[0],
        face_count=face_count,
        cell_count=cell_count,
        maximum_face_arity=int(np.max(face_sizes)),
        maximum_cell_faces=int(np.max(cell_face_counts)),
        maximum_cell_vertices=int(np.max(cell_vertex_counts)),
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
    edge_global_ids: ArrayLike | None = None,
    face_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
) -> CellComplexTopology:
    """Build the validated 0→1→2→3 complex for polyhedral cells."""

    if isinstance(value, PolyhedralConnectivity):
        if (
            vertex_count is not None
            or vertex_global_ids is not None
            or edge_global_ids is not None
            or face_global_ids is not None
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
            edge_global_ids=edge_global_ids,
            face_global_ids=face_global_ids,
            cell_global_ids=cell_global_ids,
        )
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    face_edges = np.asarray(connectivity.face_edge_values, dtype=np.int32)
    face_edge_counts = np.diff(np.asarray(connectivity.face_edge_offsets, dtype=np.int32))
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int32)
    cell_face_counts = np.diff(np.asarray(connectivity.cell_face_offsets, dtype=np.int32))
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
            EntitySubset(
                "boundary", np.zeros((connectivity.cell_count,), dtype=np.bool_)
            ),
        ),
    )
    vertex_edge_relation = EdgeRelation(
        edges.reshape((-1,)),
        np.repeat(np.arange(connectivity.edge_count, dtype=np.int32), 2),
        source_size=connectivity.vertex_count,
        target_size=connectivity.edge_count,
    )
    edge_face_relation = EdgeRelation(
        face_edges,
        np.repeat(
            np.arange(connectivity.face_count, dtype=np.int32),
            face_edge_counts,
        ),
        source_size=connectivity.edge_count,
        target_size=connectivity.face_count,
    )
    face_cell_relation = EdgeRelation(
        cell_faces,
        np.repeat(
            np.arange(connectivity.cell_count, dtype=np.int32),
            cell_face_counts,
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
                np.asarray(connectivity.face_edge_sign_values),
            ),
            OrientedIncidence(
                3,
                face_entities,
                cell_entities,
                face_cell_relation,
                np.asarray(connectivity.cell_face_sign_values),
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

    # Local faces enumerate (cell, local face) in scan order, so first-appearance
    # numbering matches a sequential traversal of the cells.
    oriented = cells[:, np.asarray(_POLYHEDRAL_FACE_ROUTES["tetrahedron"])].reshape(
        (-1, 3)
    )
    first, second, third = oriented.T
    inversions = (first > second).astype(np.int8) + (first > third) + (second > third)
    side_signs = np.where(inversions % 2 == 1, -1.0, 1.0)
    side_keys = np.sort(oriented, axis=1)
    side_faces, face_sides = _first_appearance_groups(side_keys)
    counts = np.bincount(side_faces, minlength=face_sides.size).astype(np.int32)
    if np.any(counts > 2):
        raise ValueError("Tetrahedral cells must be face-manifold.")
    orientation = np.bincount(side_faces, weights=side_signs, minlength=counts.size)
    if np.any((counts == 2) & (np.abs(orientation) == 2.0)):
        raise ValueError("Shared tetrahedral faces must have opposite orientation.")

    faces = side_keys[face_sides]
    # A face contributes its edges (0, 1), (0, 2), (1, 2) when first seen.
    face_edge_keys = faces[:, np.asarray(((0, 1), (0, 2), (1, 2)))].reshape((-1, 2))
    edge_groups, edge_keys = _first_appearance_groups(face_edge_keys)
    edges = face_edge_keys[edge_keys]
    # The boundary (1, 2), (2, 0), (0, 1) of a sorted face has signs (+, -, +).
    face_edges = edge_groups.reshape((-1, 3))[:, ::-1].astype(np.int32)
    face_edge_signs = np.tile(np.asarray((1.0, -1.0, 1.0)), (faces.shape[0], 1))
    cell_faces = side_faces.reshape((-1, 4)).astype(np.int32)
    cell_face_signs = side_signs.reshape((-1, 4))
    boundary_faces = counts == 1
    boundary_edges = np.zeros((edges.shape[0],), dtype=np.bool_)
    boundary_edges[face_edges[boundary_faces].reshape((-1,))] = True
    boundary_vertices = np.zeros((vertices,), dtype=np.bool_)
    boundary_vertices[faces[boundary_faces].reshape((-1,))] = True
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
    edge_global_ids: ArrayLike | None = None,
    face_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
) -> CellComplexTopology:
    """Build the validated 0→1→2→3 complex for tetrahedral cells."""

    return _tetrahedral_complex(
        tetrahedral_connectivity(tetrahedra, vertex_count),
        vertex_global_ids=vertex_global_ids,
        edge_global_ids=edge_global_ids,
        face_global_ids=face_global_ids,
        cell_global_ids=cell_global_ids,
    )


def _tetrahedral_complex(
    connectivity: TetrahedralConnectivity,
    /,
    *,
    vertex_global_ids: ArrayLike | None,
    edge_global_ids: ArrayLike | None,
    face_global_ids: ArrayLike | None,
    cell_global_ids: ArrayLike | None,
) -> CellComplexTopology:
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    faces = np.asarray(connectivity.faces, dtype=np.int32)
    face_edges = np.asarray(connectivity.face_edges, dtype=np.int32)
    cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int32)
    cell_count = connectivity.cell_count
    vertex_ids = _resolved_entity_ids(
        "vertex_global_ids", vertex_global_ids, connectivity.vertex_count
    )
    cell_ids_global = _resolved_entity_ids("cell_global_ids", cell_global_ids, cell_count)
    edge_ids_global = (
        _canonical_entity_ids(np.sort(vertex_ids[edges], axis=1))
        if edge_global_ids is None
        else _resolved_entity_ids("edge_global_ids", edge_global_ids, edges.shape[0])
    )
    face_ids_global = (
        _canonical_entity_ids(np.sort(vertex_ids[faces], axis=1))
        if face_global_ids is None
        else _resolved_entity_ids("face_global_ids", face_global_ids, faces.shape[0])
    )
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
        subsets=(EntitySubset("boundary", np.zeros((cell_count,), dtype=np.bool_)),),
    )
    vertex_edge_relation = EdgeRelation(
        edges.reshape((-1,)),
        np.repeat(np.arange(edges.shape[0], dtype=np.int32), 2),
        source_size=connectivity.vertex_count,
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
        np.repeat(np.arange(cell_count, dtype=np.int32), 4),
        source_size=faces.shape[0],
        target_size=cell_count,
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
    "PolyhedralWorksetLimitError",
    "PolyhedralWorksets",
    "prepare_polyhedral_worksets",
    "TetrahedralConnectivity",
    "polygonal_cell_complex",
    "polygonal_connectivity",
    "polyhedral_cell_complex",
    "polyhedral_connectivity",
    "tetrahedral_cell_complex",
    "tetrahedral_connectivity",
]
