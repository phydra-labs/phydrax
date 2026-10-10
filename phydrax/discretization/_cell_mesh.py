#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._array_archive import array_collection_digest
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation
from ..typing import checked, Dim, Int32
from ._cell_complex import (
    _canonical_entity_ids,
    _interval_complex,
    _polygonal_complex,
    _tetrahedral_complex,
    interval_connectivity,
    IntervalConnectivity,
    polygonal_connectivity,
    PolygonalConnectivity,
    polyhedral_cell_complex,
    polyhedral_connectivity as _build_polyhedral_connectivity,
    PolyhedralConnectivity,
    simplicial_cell_complex,
    tetrahedral_connectivity,
    TetrahedralConnectivity,
)
from ._hexahedral import (
    _hexahedral_complex,
    hexahedral_connectivity,
    HexahedralConnectivity,
)
from ._periodic_topology import PeriodicMeshTopology
from ._reference_cell import (
    reference_cell_topology,
    REFERENCE_TOPOLOGIES,
    ReferenceCellTopology,
)
from ._support import DiscreteSupport
from ._topology import (
    _has_duplicate_rows,
    CellComplexTopology,
    EntitySet,
    EntitySubset,
    OrientedIncidence,
)


if TYPE_CHECKING:
    from ._cell_geometry import CellGeometrySpec, CellGeometryStorageProjection


def _admit_cell_kind(kind: str, /) -> ReferenceCellTopology | None:
    if (
        kind not in REFERENCE_TOPOLOGIES
        and kind not in ("polygon", "polyhedron")
        and not kind.startswith(("simplex:", "tensor:"))
    ):
        raise ValueError(
            "cell_kind must name a reference cell, simplex:N, tensor:N, polygon, or polyhedron."
        )
    return None if kind in ("polygon", "polyhedron") else reference_cell_topology(kind)


def _admit_cell_rows(
    kind: str, reference: ReferenceCellTopology | None, vertices: ArrayLike, /
) -> tuple[np.ndarray, int]:
    """Own reference-cell arity and nonempty rank-two connectivity admission."""
    cells = np.asarray(vertices, dtype=np.int32)
    if reference is None:
        arity = cells.shape[1] if cells.ndim == 2 else -1
        minimum_arity = 4 if kind == "polyhedron" else 3
    else:
        arity = len(reference.vertices)
        minimum_arity = 2 if reference.dimension == 1 else 3
    if (
        cells.ndim != 2
        or cells.shape[0] == 0
        or cells.shape[1] != arity
        or arity < minimum_arity
        or (kind == "polygon" and arity < 5)
    ):
        raise ValueError(f"{kind} cell vertices have incompatible arity {arity}.")
    return cells, minimum_arity


def _admit_active_cell_vertices(
    kind: str, cells: np.ndarray, vertex_valid: ArrayLike | None, minimum_arity: int, /
) -> np.ndarray:
    """Validate fixed-capacity padding and active vertex-set uniqueness."""
    valid = (
        np.ones_like(cells, dtype=np.bool_)
        if vertex_valid is None
        else np.asarray(vertex_valid, dtype=np.bool_)
    )
    if valid.shape != cells.shape:
        raise ValueError("vertex_valid must match cell vertex storage.")
    if kind != "polyhedron" and not np.all(valid):
        raise ValueError("Only polyhedron blocks may contain padded vertices.")
    if np.any(np.sum(valid, axis=1) < minimum_arity):
        raise ValueError(f"Each {kind} cell requires at least {minimum_arity} vertices.")
    if np.any(cells[valid] < 0):
        raise ValueError("Cell vertex indices must be non-negative.")
    # Padding sorts first as -1, so equal sorted rows are equal active sets.
    active_rows = np.sort(np.where(valid, cells, -1), axis=1)
    if np.any((active_rows[:, 1:] == active_rows[:, :-1]) & (active_rows[:, 1:] >= 0)):
        raise ValueError("Each cell must reference distinct active vertices.")
    if _has_duplicate_rows(active_rows):
        raise ValueError("Cell blocks cannot contain duplicate cells.")
    return valid


def _admit_cell_global_ids(
    cell_count: int, global_ids: ArrayLike | None, /
) -> np.ndarray:
    """Admit one unique stable identity per cell without changing numeric representation."""
    ids = (
        np.arange(cell_count, dtype=np.int64)
        if global_ids is None
        else np.asarray(global_ids, dtype=np.int64)
    )
    if ids.shape != (cell_count,):
        raise ValueError("Cell global_ids must have shape (cell_count,).")
    if np.any(ids < 0) or np.unique(ids).size != ids.size:
        raise ValueError("Cell global_ids must be unique non-negative integers.")
    return ids


class CellBlock(StrictModule, NonTrainableState):
    """One homogeneous, ordered block of top-dimensional cells."""

    name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    vertices: Array
    vertex_valid: Array
    global_ids: Array
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        cell_kind: str,
        vertices: ArrayLike,
        /,
        *,
        vertex_valid: ArrayLike | None = None,
        global_ids: ArrayLike | None = None,
    ) -> None:
        block_name = str(name)
        kind = str(cell_kind)
        if not block_name:
            raise ValueError("Cell block name must be non-empty.")
        reference = _admit_cell_kind(kind)
        cells, minimum_arity = _admit_cell_rows(kind, reference, vertices)
        valid = _admit_active_cell_vertices(kind, cells, vertex_valid, minimum_arity)
        ids = _admit_cell_global_ids(cells.shape[0], global_ids)
        self.name = block_name
        self.cell_kind = kind
        self.vertices = jnp.asarray(cells)
        self.vertex_valid = jnp.asarray(valid)
        self.global_ids = jnp.asarray(ids)
        self.block_id = canonical_fingerprint(
            {
                "kind": "cell-block",
                "name": block_name,
                "cell_kind": kind,
                "vertices": array_tree_fingerprint(cells),
                "vertex_valid": array_tree_fingerprint(valid),
                "global_ids": array_tree_fingerprint(ids),
            }
        )

    @property
    def cell_count(self) -> int:
        return self.vertices.shape[0]

    @property
    def arity(self) -> int:
        return (
            self.vertices.shape[1]
            if self.cell_kind in ("polygon", "polyhedron")
            else len(reference_cell_topology(self.cell_kind).vertices)
        )

    @property
    def topological_dimension(self) -> int:
        if self.cell_kind == "polygon":
            return 2
        if self.cell_kind == "polyhedron":
            return 3
        return reference_cell_topology(self.cell_kind).dimension


class PolyhedralBlock(StrictModule, NonTrainableState):
    """One exact-width block of arbitrary polyhedra grouped by vertex count."""

    name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    vertices: Array
    vertex_valid: Array
    global_ids: Array
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        vertices: ArrayLike,
        /,
        *,
        global_ids: ArrayLike | None = None,
    ) -> None:
        block_name = str(name)
        cells = np.asarray(vertices, dtype=np.int32)
        if not block_name:
            raise ValueError("Polyhedral block name must be non-empty.")
        if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] < 4:
            raise ValueError(
                "Polyhedral block vertices must have shape (cells > 0, arity >= 4)."
            )
        if np.any(cells < 0):
            raise ValueError("Polyhedral cell vertex indices must be non-negative.")
        ordered = np.sort(cells, axis=1)
        if np.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError("Each polyhedral cell must reference distinct vertices.")
        if _has_duplicate_rows(ordered):
            raise ValueError("Polyhedral blocks cannot contain duplicate cells.")
        ids = (
            np.arange(cells.shape[0], dtype=np.int64)
            if global_ids is None
            else np.asarray(global_ids, dtype=np.int64)
        )
        if ids.shape != (cells.shape[0],):
            raise ValueError("Polyhedral global_ids must match the cell count.")
        if np.any(ids < 0) or np.unique(ids).size != ids.size:
            raise ValueError(
                "Polyhedral global_ids must be unique non-negative integers."
            )
        valid = np.ones_like(cells, dtype=np.bool_)
        self.name = block_name
        self.cell_kind = "polyhedron"
        self.vertices = jnp.asarray(cells)
        self.vertex_valid = jnp.asarray(valid)
        self.global_ids = jnp.asarray(ids)
        self.block_id = canonical_fingerprint(
            {
                "kind": "polyhedral-block",
                "name": block_name,
                "vertices": array_tree_fingerprint(cells),
                "global_ids": array_tree_fingerprint(ids),
            }
        )

    @property
    def cell_count(self) -> int:
        return self.vertices.shape[0]

    @property
    def arity(self) -> int:
        return self.vertices.shape[1]

    @property
    def topological_dimension(self) -> int:
        return 3


class SimplexCellDim(Dim):
    """Top-dimensional cells in one simplicial mesh."""


class SimplexVertexDim(Dim):
    """Reference vertices per simplex."""


@final
class SimplicialConnectivity(StrictModule, NonTrainableState):
    """Explicit n-D simplex incidence routes in mesh-cell order."""

    __strict_contract__ = True
    dimension: int = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    cells: Int32[SimplexCellDim, SimplexVertexDim]
    entities: tuple[Array, ...]
    cell_entities: tuple[Array, ...]
    cell_entity_signs: tuple[Array, ...]
    boundary_masks: tuple[Array, ...]
    connectivity_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        topology: CellComplexTopology,
        cells: ArrayLike,
        /,
        *,
        entities: Sequence[ArrayLike],
        cell_entities: Sequence[ArrayLike],
        cell_entity_signs: Sequence[ArrayLike],
        boundary_masks: Sequence[ArrayLike],
    ) -> None:
        dimension = topology.dimension
        cell_rows = np.asarray(cells, dtype=np.int32)
        vertex_rows = tuple(np.asarray(value, dtype=np.int32) for value in entities)
        routes = tuple(np.asarray(value, dtype=np.int32) for value in cell_entities)
        signs = tuple(np.asarray(value, dtype=np.float64) for value in cell_entity_signs)
        masks = tuple(np.asarray(value, dtype=np.bool_) for value in boundary_masks)
        if cell_rows.shape != (topology.entity_sets[-1].count, dimension + 1):
            raise ValueError(
                "Simplex cells must have one reference-vertex row per top cell."
            )
        if any(
            len(values) != dimension + 1 for values in (vertex_rows, routes, signs, masks)
        ):
            raise ValueError("Simplex connectivity must contain every entity degree.")
        for degree, entity_set in enumerate(topology.entity_sets):
            expected = len(tuple(combinations(range(dimension + 1), degree + 1)))
            if vertex_rows[degree].shape != (entity_set.count, degree + 1):
                raise ValueError("Simplex entity vertices have incompatible shape.")
            if routes[degree].shape != (cell_rows.shape[0], expected):
                raise ValueError("Simplex cell-to-entity routes have incompatible shape.")
            if signs[degree].shape != routes[degree].shape or np.any(
                np.abs(signs[degree]) != 1.0
            ):
                raise ValueError(
                    "Simplex route signs must be unit orientation coefficients."
                )
            if masks[degree].shape != (entity_set.count,):
                raise ValueError("Simplex boundary masks must match entity counts.")
            if np.any(routes[degree] < 0) or np.any(routes[degree] >= entity_set.count):
                raise ValueError("Simplex routes index undeclared entities.")
        self.dimension = dimension
        self.vertex_count = topology.entity_sets[0].count
        self.cells = jnp.asarray(cell_rows, dtype=jnp.int32)
        self.entities = tuple(
            jnp.asarray(value, dtype=jnp.int32) for value in vertex_rows
        )
        self.cell_entities = tuple(
            jnp.asarray(value, dtype=jnp.int32) for value in routes
        )
        self.cell_entity_signs = tuple(
            jnp.asarray(value, dtype=jnp.float64) for value in signs
        )
        self.boundary_masks = tuple(
            jnp.asarray(value, dtype=jnp.bool_) for value in masks
        )
        self.connectivity_id = canonical_fingerprint(
            {
                "kind": "simplicial-connectivity",
                "topology": topology.topology_id,
                "cells": array_tree_fingerprint(cell_rows),
            }
        )

    @property
    def cell_count(self) -> int:
        return self.cells.shape[0]

    @property
    def boundary_vertices(self) -> Array:
        return self.boundary_masks[0]

    @property
    def boundary_edges(self) -> Array:
        return self.boundary_masks[1]

    @property
    def boundary_faces(self) -> Array:
        if self.dimension < 2:
            raise ValueError("This simplex complex has no degree-two faces.")
        return self.boundary_masks[2]

    @property
    def boundary_facets(self) -> Array:
        return self.boundary_masks[self.dimension - 1]


_CellMeshConnectivity = (
    IntervalConnectivity
    | PolygonalConnectivity
    | TetrahedralConnectivity
    | HexahedralConnectivity
    | PolyhedralConnectivity
    | SimplicialConnectivity
)


class _PreparedCellMeshTopology(NamedTuple):
    """Validated coordinate-independent state shared by coordinate refreshes."""

    blocks: tuple[CellBlock | PolyhedralBlock, ...]
    vertex_global_ids: Array | np.ndarray
    connectivity: _CellMeshConnectivity
    topology: CellComplexTopology
    topology_id: str
    topological_dimension: int
    periodic_topology: PeriodicMeshTopology | None


def _interval_mesh_topology(
    blocks: tuple[CellBlock | PolyhedralBlock, ...],
    coordinate_count: int,
    vertex_ids: np.ndarray,
    cell_ids: np.ndarray,
    /,
) -> tuple[IntervalConnectivity, CellComplexTopology]:
    if any(block.cell_kind != "interval" for block in blocks):
        raise ValueError("One-dimensional meshes support interval blocks only.")
    intervals = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int32) for block in blocks),
        axis=0,
    )
    connectivity = interval_connectivity(intervals, coordinate_count)
    return connectivity, _interval_complex(
        connectivity, vertex_global_ids=vertex_ids, cell_global_ids=cell_ids
    )


def _polygonal_mesh_topology(
    blocks: tuple[CellBlock | PolyhedralBlock, ...],
    coordinate_count: int,
    vertex_ids: np.ndarray,
    edge_ids: np.ndarray | None,
    cell_ids: np.ndarray,
    /,
) -> tuple[PolygonalConnectivity, CellComplexTopology]:
    if any(
        block.cell_kind not in ("triangle", "quadrilateral", "polygon")
        for block in blocks
    ):
        raise ValueError("Two-dimensional meshes support polygonal blocks only.")
    arities = tuple(block.arity for block in blocks)
    if arities != tuple(sorted(arities)):
        raise ValueError("Polygonal CellMesh blocks must be ordered by increasing arity.")
    triangles = [
        np.asarray(block.vertices, dtype=np.int32)
        for block in blocks
        if block.cell_kind == "triangle"
    ]
    quadrilaterals = [
        np.asarray(block.vertices, dtype=np.int32)
        for block in blocks
        if block.cell_kind == "quadrilateral"
    ]
    connectivity = polygonal_connectivity(
        np.concatenate(triangles, axis=0) if triangles else None,
        np.concatenate(quadrilaterals, axis=0) if quadrilaterals else None,
        coordinate_count,
        polygons=tuple(
            np.asarray(block.vertices, dtype=np.int32)
            for block in blocks
            if block.cell_kind == "polygon"
        ),
    )
    return connectivity, _polygonal_complex(
        connectivity,
        vertex_global_ids=vertex_ids,
        edge_global_ids=edge_ids,
        cell_global_ids=cell_ids,
    )


def _validate_supplied_polyhedral_connectivity(
    connectivity: PolyhedralConnectivity,
    blocks: tuple[CellBlock | PolyhedralBlock, ...],
    coordinate_count: int,
    vertex_ids: np.ndarray,
    entity_ids: Mapping[int, np.ndarray],
    cell_ids: np.ndarray,
    /,
) -> None:
    if (
        connectivity.vertex_count != coordinate_count
        or connectivity.cell_count != sum(block.cell_count for block in blocks)
        or not np.array_equal(np.asarray(connectivity.vertex_global_ids), vertex_ids)
        or (
            entity_ids.get(1) is not None
            and not np.array_equal(
                entity_ids[1], np.asarray(connectivity.edge_global_ids)
            )
        )
        or (
            entity_ids.get(2) is not None
            and not np.array_equal(
                entity_ids[2], np.asarray(connectivity.face_global_ids)
            )
        )
        or not np.array_equal(np.asarray(connectivity.cell_global_ids), cell_ids)
    ):
        raise ValueError("PolyhedralConnectivity does not match the CellMesh IDs.")
    offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int32)
    values = np.asarray(connectivity.cell_vertex_values, dtype=np.int32)
    cell_offset = 0
    for block in blocks:
        widths = np.diff(offsets[cell_offset : cell_offset + block.cell_count + 1])
        if np.any(widths != block.arity):
            raise ValueError("PolyhedralConnectivity cell widths do not match blocks.")
        start = int(offsets[cell_offset])
        stop = int(offsets[cell_offset + block.cell_count])
        expected = np.asarray(block.vertices, dtype=np.int32)
        if not np.array_equal(
            values[start:stop].reshape(expected.shape), np.sort(expected, axis=1)
        ):
            raise ValueError("PolyhedralConnectivity cell vertices do not match blocks.")
        cell_offset += block.cell_count


def _volume_mesh_topology(
    blocks: tuple[CellBlock | PolyhedralBlock, ...],
    coordinate_count: int,
    vertex_ids: np.ndarray,
    entity_ids: Mapping[int, np.ndarray],
    cell_ids: np.ndarray,
    supplied: PolyhedralConnectivity | None,
    /,
) -> tuple[_CellMeshConnectivity, CellComplexTopology]:
    if supplied is not None:
        _validate_supplied_polyhedral_connectivity(
            supplied, blocks, coordinate_count, vertex_ids, entity_ids, cell_ids
        )
        return supplied, polyhedral_cell_complex(supplied)
    if len(blocks) == 1 and blocks[0].cell_kind == "tetrahedron":
        connectivity = tetrahedral_connectivity(
            np.asarray(blocks[0].vertices, dtype=np.int32), coordinate_count
        )
        return connectivity, _tetrahedral_complex(
            connectivity,
            vertex_global_ids=vertex_ids,
            edge_global_ids=entity_ids.get(1),
            face_global_ids=entity_ids.get(2),
            cell_global_ids=cell_ids,
        )
    if len(blocks) == 1 and blocks[0].cell_kind == "hexahedron":
        connectivity = hexahedral_connectivity(
            np.asarray(blocks[0].vertices, dtype=np.int32), coordinate_count
        )
        return connectivity, _hexahedral_complex(
            connectivity,
            vertex_global_ids=vertex_ids,
            edge_global_ids=entity_ids.get(1),
            face_global_ids=entity_ids.get(2),
            cell_global_ids=cell_ids,
        )
    if any(block.cell_kind == "polyhedron" for block in blocks):
        raise ValueError("Polyhedron blocks require matching PolyhedralConnectivity.")
    connectivity = _build_polyhedral_connectivity(
        tuple(
            (block.cell_kind, np.asarray(block.vertices, dtype=np.int32))
            for block in blocks
        ),
        coordinate_count,
        vertex_global_ids=vertex_ids,
        edge_global_ids=entity_ids.get(1),
        face_global_ids=entity_ids.get(2),
        cell_global_ids=cell_ids,
    )
    return connectivity, polyhedral_cell_complex(connectivity)


def _validated_mesh_blocks(
    point_shape: tuple[int, ...],
    blocks: Sequence[CellBlock | PolyhedralBlock],
    /,
) -> tuple[tuple[CellBlock | PolyhedralBlock, ...], int]:
    coordinate_count, ambient_dimension = point_shape
    normalized_blocks = tuple(blocks)
    if not normalized_blocks:
        raise ValueError("Cell mesh requires at least one cell block.")
    if not all(
        isinstance(block, (CellBlock, PolyhedralBlock)) for block in normalized_blocks
    ):
        raise TypeError(
            "blocks must contain only CellBlock or PolyhedralBlock instances."
        )
    names = tuple(block.name for block in normalized_blocks)
    if len(set(names)) != len(names):
        raise ValueError("Cell block names must be unique.")
    dimensions = {block.topological_dimension for block in normalized_blocks}
    if len(dimensions) != 1:
        raise ValueError("All cell blocks must share one topological dimension.")
    topological_dimension = dimensions.pop()
    if ambient_dimension < topological_dimension:
        raise ValueError(
            "Cell mesh ambient dimension cannot be smaller than its topological dimension."
        )
    for block in normalized_blocks:
        vertices_ = np.asarray(block.vertices)
        valid_ = np.asarray(block.vertex_valid, dtype=np.bool_)
        if np.any(vertices_[valid_] >= coordinate_count):
            raise ValueError(f"Cell block {block.name!r} indexes undeclared vertices.")
    return normalized_blocks, topological_dimension


def _resolved_mesh_identities(
    coordinate_count: int,
    blocks: tuple[CellBlock | PolyhedralBlock, ...],
    topological_dimension: int,
    vertex_global_ids: ArrayLike | None,
    entity_global_ids: Mapping[int, ArrayLike] | None,
    /,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """Return consistent vertex, supplied entity, and cell global IDs."""

    entity_ids = (
        {}
        if entity_global_ids is None
        else {
            int(dimension): np.asarray(values, dtype=np.int64)
            for dimension, values in entity_global_ids.items()
        }
    )
    if any(
        dimension < 0 or dimension > topological_dimension for dimension in entity_ids
    ):
        raise ValueError("entity_global_ids contains an undeclared dimension.")
    supplied_vertices = entity_ids.get(0)
    if supplied_vertices is not None and vertex_global_ids is not None:
        if not np.array_equal(
            supplied_vertices, np.asarray(vertex_global_ids, dtype=np.int64)
        ):
            raise ValueError(
                "vertex_global_ids contradicts entity_global_ids dimension zero."
            )
    global_ids = (
        supplied_vertices
        if supplied_vertices is not None
        else (
            np.arange(coordinate_count, dtype=np.int64)
            if vertex_global_ids is None
            else np.asarray(vertex_global_ids, dtype=np.int64)
        )
    )
    if global_ids.shape != (coordinate_count,):
        raise ValueError("vertex_global_ids must have shape (coordinate_count,).")
    if np.any(global_ids < 0) or np.unique(global_ids).size != global_ids.size:
        raise ValueError("vertex_global_ids must be unique non-negative integers.")
    cell_global_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in blocks)
    )
    if np.unique(cell_global_ids).size != cell_global_ids.size:
        raise ValueError("Cell global IDs must be unique across mesh blocks.")
    supplied_cells = entity_ids.get(topological_dimension)
    if supplied_cells is not None and not np.array_equal(supplied_cells, cell_global_ids):
        raise ValueError(
            "Top-dimensional entity_global_ids contradict cell block global IDs."
        )
    return global_ids, entity_ids, cell_global_ids


def _simplex_logical_topology_arrays(
    block: CellBlock,
    vertex_ids: np.ndarray,
    connectivity: IntervalConnectivity
    | PolygonalConnectivity
    | TetrahedralConnectivity
    | SimplicialConnectivity,
    topology: CellComplexTopology,
    /,
) -> dict[str, np.ndarray]:
    """Canonical logical simplex content, independent of local slots/placement."""
    cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    cell_order = np.argsort(cell_ids, kind="stable")
    arrays = {
        "vertex_global_ids": np.sort(vertex_ids),
        "cell_global_ids": cell_ids[cell_order],
        "cell_vertices": vertex_ids[np.asarray(block.vertices, dtype=np.int32)][
            cell_order
        ],
    }
    entities: tuple[np.ndarray, ...]
    if isinstance(connectivity, SimplicialConnectivity):
        entities = tuple(
            np.asarray(value, dtype=np.int32) for value in connectivity.entities[1:-1]
        )
    elif isinstance(connectivity, IntervalConnectivity):
        entities = ()
    elif isinstance(connectivity, PolygonalConnectivity):
        entities = (np.asarray(connectivity.edges, dtype=np.int32),)
    else:
        entities = (
            np.asarray(connectivity.edges, dtype=np.int32),
            np.asarray(connectivity.faces, dtype=np.int32),
        )
    for degree, vertices in enumerate(entities, start=1):
        keys = np.sort(vertex_ids[vertices], axis=1)
        order = np.lexsort(
            tuple(keys[:, column] for column in range(keys.shape[1] - 1, -1, -1))
        )
        arrays[f"entity_vertices_{degree}"] = keys[order]
        arrays[f"entity_global_ids_{degree}"] = np.asarray(
            topology.entities(degree).entity_ids, dtype=np.int64
        )[order]
    return arrays


def _simplex_orientation(vertices: np.ndarray, /) -> np.ndarray:
    """Orientation of each ordered simplex relative to its ascending vertex row."""
    inversions = np.zeros(vertices.shape[0], dtype=np.int64)
    for first, second in combinations(range(vertices.shape[1]), 2):
        inversions += vertices[:, first] > vertices[:, second]
    return np.where(inversions % 2, -1.0, 1.0).astype(np.float64)


def _simplex_mesh_levels(
    cells: np.ndarray, dimension: int, vertex_count: int, /
) -> tuple[np.ndarray, ...]:
    top = np.sort(cells, axis=1)
    if _has_duplicate_rows(top):
        raise ValueError("Simplex mesh blocks cannot contain duplicate cells.")
    levels: list[np.ndarray] = [np.arange(vertex_count, dtype=np.int32)[:, None]]
    for degree in range(1, dimension):
        faces = {
            tuple(sorted(int(cell[index]) for index in indices))
            for cell in cells
            for indices in combinations(range(dimension + 1), degree + 1)
        }
        levels.append(np.asarray(sorted(faces), dtype=np.int32))
    levels.append(top)
    return tuple(levels)


def _simplex_mesh_boundary(
    levels: tuple[np.ndarray, ...], topology: CellComplexTopology, /
) -> tuple[np.ndarray, ...]:
    dimension = topology.dimension
    relation = topology.incidences[-1].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    source = np.asarray(relation.source_indices)[valid]
    counts = np.bincount(source, minlength=levels[-2].shape[0])
    if np.any(counts > 2):
        raise ValueError("Simplex mesh facets may have at most two incident cells.")
    facets = levels[-2][counts == 1]
    masks: list[np.ndarray] = []
    for degree in range(dimension):
        boundary = {
            tuple(int(vertex) for vertex in face)
            for facet in facets
            for face in combinations(facet, degree + 1)
        }
        masks.append(
            np.asarray(
                [
                    tuple(int(vertex) for vertex in row) in boundary
                    for row in levels[degree]
                ],
                dtype=np.bool_,
            )
        )
    masks.append(np.zeros(levels[-1].shape[0], dtype=np.bool_))
    return tuple(masks)


def _simplex_cell_routes(
    cells: np.ndarray, levels: tuple[np.ndarray, ...], /
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    dimension = len(levels) - 1
    routes: list[np.ndarray] = []
    signs: list[np.ndarray] = []
    for degree, entities in enumerate(levels):
        local = tuple(combinations(range(dimension + 1), degree + 1))
        lookup = {
            tuple(int(vertex) for vertex in row): index
            for index, row in enumerate(entities)
        }
        selected = cells[:, np.asarray(local, dtype=np.int32)]
        ordered = np.sort(selected, axis=-1)
        route = np.asarray(
            [
                lookup[tuple(int(vertex) for vertex in row)]
                for row in ordered.reshape((-1, degree + 1))
            ],
            dtype=np.int32,
        ).reshape((cells.shape[0], len(local)))
        coefficient = _simplex_orientation(selected.reshape((-1, degree + 1))).reshape(
            route.shape
        )
        if degree == dimension:
            coefficient = np.ones_like(coefficient, dtype=np.float64)
        routes.append(route)
        signs.append(coefficient)
    return tuple(routes), tuple(signs)


def _simplex_mesh_topology(
    blocks: tuple[CellBlock | PolyhedralBlock, ...],
    vertex_count: int,
    vertex_ids: np.ndarray,
    entity_ids: dict[int, np.ndarray],
    cell_ids: np.ndarray,
    /,
) -> tuple[SimplicialConnectivity, CellComplexTopology]:
    dimension = blocks[0].topological_dimension
    if any(
        block.cell_kind not in ("interval", "triangle", "tetrahedron")
        and not block.cell_kind.startswith("simplex:")
        for block in blocks
    ):
        raise ValueError("General simplicial meshes require simplex blocks.")
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int32) for block in blocks)
    )
    levels = _simplex_mesh_levels(cells, dimension, vertex_count)
    canonical = simplicial_cell_complex(levels)
    masks = _simplex_mesh_boundary(levels, canonical)
    sets: list[EntitySet] = []
    for degree, (rows, source) in enumerate(
        zip(levels, canonical.entity_sets, strict=True)
    ):
        ids = (
            vertex_ids
            if degree == 0
            else cell_ids
            if degree == dimension
            else entity_ids.get(degree, _canonical_entity_ids(vertex_ids[rows]))
        )
        if ids.shape != (source.count,):
            raise ValueError("Simplex entity global IDs must match entity counts.")
        sets.append(
            EntitySet(
                source.name,
                degree,
                ids,
                subsets=(EntitySubset("boundary", masks[degree]),),
            )
        )
    orientation = _simplex_orientation(cells)
    incidences: list[OrientedIncidence] = []
    for degree, incidence in enumerate(canonical.incidences, start=1):
        coefficients = np.asarray(incidence.signs)
        if degree == dimension:
            coefficients = (
                coefficients * orientation[np.asarray(incidence.relation.target_indices)]
            )
        incidences.append(
            OrientedIncidence(
                degree, sets[degree - 1], sets[degree], incidence.relation, coefficients
            )
        )
    topology = CellComplexTopology(sets, incidences)
    routes, signs = _simplex_cell_routes(cells, levels)
    return SimplicialConnectivity(
        topology,
        cells,
        entities=levels,
        cell_entities=routes,
        cell_entity_signs=signs,
        boundary_masks=masks,
    ), topology


def _tensor_mesh_route_blocks(
    blocks: tuple[CellBlock | PolyhedralBlock, ...], /
) -> tuple[CellBlock | PolyhedralBlock, ...]:
    """Bind dimension-qualified cubes to the native 1-D/2-D/3-D vertex routes."""
    names = {1: "interval", 2: "quadrilateral", 3: "hexahedron"}
    result: list[CellBlock | PolyhedralBlock] = []
    for block in blocks:
        if not block.cell_kind.startswith("tensor:"):
            result.append(block)
            continue
        dimension = block.topological_dimension
        if dimension not in names:
            raise ValueError(
                "Unstructured tensor CellMesh supports dimensions one through three."
            )
        result.append(
            CellBlock(
                block.name, names[dimension], block.vertices, global_ids=block.global_ids
            )
        )
    return tuple(result)


def _prepare_cell_mesh_topology(
    point_shape: tuple[int, ...],
    blocks: Sequence[CellBlock | PolyhedralBlock],
    /,
    *,
    vertex_global_ids: ArrayLike | None,
    entity_global_ids: Mapping[int, ArrayLike] | None,
    polyhedral_connectivity: PolyhedralConnectivity | None,
    periodic_topology: PeriodicMeshTopology | None,
) -> _PreparedCellMeshTopology:
    """Validate blocks and identities, then build connectivity and topology once."""

    coordinate_count = point_shape[0]
    normalized_blocks, topological_dimension = _validated_mesh_blocks(point_shape, blocks)
    global_ids, entity_ids, cell_global_ids = _resolved_mesh_identities(
        coordinate_count,
        normalized_blocks,
        topological_dimension,
        vertex_global_ids,
        entity_global_ids,
    )
    if polyhedral_connectivity is not None and topological_dimension != 3:
        raise ValueError(
            "polyhedral_connectivity is valid only for three-dimensional meshes."
        )
    route_blocks = _tensor_mesh_route_blocks(normalized_blocks)
    if any(block.cell_kind.startswith("simplex:") for block in normalized_blocks):
        if polyhedral_connectivity is not None:
            raise ValueError("Simplicial meshes cannot use polyhedral connectivity.")
        connectivity, topology = _simplex_mesh_topology(
            normalized_blocks, coordinate_count, global_ids, entity_ids, cell_global_ids
        )
    elif topological_dimension == 1:
        connectivity, topology = _interval_mesh_topology(
            route_blocks, coordinate_count, global_ids, cell_global_ids
        )
    elif topological_dimension == 2:
        connectivity, topology = _polygonal_mesh_topology(
            route_blocks,
            coordinate_count,
            global_ids,
            entity_ids.get(1),
            cell_global_ids,
        )
    else:
        connectivity, topology = _volume_mesh_topology(
            route_blocks,
            coordinate_count,
            global_ids,
            entity_ids,
            cell_global_ids,
            polyhedral_connectivity,
        )

    if (
        len(normalized_blocks) == 1
        and isinstance(normalized_blocks[0], CellBlock)
        and (
            normalized_blocks[0].cell_kind in ("interval", "triangle", "tetrahedron")
            or normalized_blocks[0].cell_kind.startswith("simplex:")
        )
        and isinstance(
            connectivity,
            (
                IntervalConnectivity,
                PolygonalConnectivity,
                TetrahedralConnectivity,
                SimplicialConnectivity,
            ),
        )
    ):
        topology_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-topology",
                "dimension": topological_dimension,
                "blocks": [(normalized_blocks[0].name, normalized_blocks[0].cell_kind)],
                "arrays": array_collection_digest(
                    _simplex_logical_topology_arrays(
                        normalized_blocks[0], global_ids, connectivity, topology
                    )
                ),
            }
        )
    else:
        canonical_blocks = []
        for block in normalized_blocks:
            block_ids = np.asarray(block.global_ids, dtype=np.int64)
            order = np.argsort(block_ids, kind="stable")
            global_vertices = global_ids[np.asarray(block.vertices, dtype=np.int32)]
            canonical_blocks.append(
                {
                    "name": block.name,
                    "cell_kind": block.cell_kind,
                    "global_ids": array_tree_fingerprint(block_ids[order]),
                    "global_vertices": array_tree_fingerprint(global_vertices[order]),
                    "vertex_valid": array_tree_fingerprint(
                        np.asarray(block.vertex_valid)[order]
                    ),
                }
            )
        topology_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-topology",
                "topological_dimension": topological_dimension,
                "vertex_global_ids": array_tree_fingerprint(global_ids),
                "blocks": canonical_blocks,
                "cell_complex": topology.topology_id,
            }
        )
    if periodic_topology is not None:
        if not isinstance(periodic_topology, PeriodicMeshTopology):
            raise TypeError("periodic_topology must be PeriodicMeshTopology or None.")
        if periodic_topology.lifted_topology_id != topology_id:
            raise ValueError(
                "periodic_topology was not prepared for this lifted mesh topology."
            )
        # Quotient identity is part of topology identity; plain meshes keep theirs.
        topology_id = canonical_fingerprint(
            {
                "kind": "periodic-cell-mesh-topology",
                "lifted": topology_id,
                "periodic": periodic_topology.periodic_topology_id,
            }
        )
    return _PreparedCellMeshTopology(
        normalized_blocks,
        global_ids,
        connectivity,
        topology,
        topology_id,
        topological_dimension,
        periodic_topology,
    )


def _reused_cell_mesh_topology(
    prepared: _PreparedCellMeshTopology,
    point_shape: tuple[int, ...],
    blocks: Sequence[CellBlock | PolyhedralBlock],
    /,
    *,
    identities_supplied: bool,
) -> _PreparedCellMeshTopology:
    """Check that a coordinate refresh keeps the prepared topology intact."""

    if not isinstance(prepared, _PreparedCellMeshTopology):
        raise TypeError("A prepared cell-mesh topology must come from a CellMesh.")
    if identities_supplied:
        raise ValueError("A prepared cell-mesh topology already owns its identities.")
    normalized_blocks = tuple(blocks)
    if len(normalized_blocks) != len(prepared.blocks) or any(
        block is not prepared_block
        for block, prepared_block in zip(normalized_blocks, prepared.blocks, strict=True)
    ):
        raise ValueError("blocks must be the prepared topology's blocks.")
    if point_shape[0] != prepared.vertex_global_ids.shape[0]:
        raise ValueError("vertex_global_ids must have shape (coordinate_count,).")
    if point_shape[1] < prepared.topological_dimension:
        raise ValueError(
            "Cell mesh ambient dimension cannot be smaller than its topological dimension."
        )
    return prepared


class CellMeshStorage(StrictModule, NonTrainableState):
    """Globally logical storage with a process-local, stable-ID closure view.

    ``CellMesh`` numerical fields are the locally addressable lowered view, not
    complete global arrays. ``logical_arrays`` are the accepted globally shaped
    arrays used by addressable checkpoint publication. Neither ownership nor
    positive local geometry is a global certificate; ``evidence_id`` binds the
    separately established collective coverage.

    ``local_coordinates`` are sampled topological corners; mapped coordinates
    retain original source coefficients in ``local_geometry``. Their scientific
    DOF IDs/count/owners are independent of topological vertex identities.
    ``logical_geometry_id`` binds the sampled carrier, whereas
    ``logical_coordinate_geometry_id`` binds the authoritative source map and
    exact reference ancestry. ``restore_geometry`` validates and restores that
    map from the logical coefficient, basis, route, and ancestry banks.
    Non-addressable banks require a ``CellGeometryStorageProjection`` computed
    collectively from fixed-capacity closure queries before owner-local lowering;
    geometry validation subsequently reads only its addressable receipts.
    """

    entity_global_ids: tuple[Array, ...]
    entity_owned: tuple[Array, ...]
    entity_owner: tuple[Array, ...]
    logical_arrays: tuple[tuple[str, Array], ...]
    local_coordinates: Array
    local_blocks: tuple[CellBlock | PolyhedralBlock, ...]
    coordinate_global_ids: Array
    coordinate_owner: Array
    local_geometry: CellGeometrySpec | None
    geometry_projection: CellGeometryStorageProjection | None
    global_coordinate_count: int = eqx.field(static=True)
    local_geometry_source_id: str = eqx.field(static=True)
    geometry_source_blocks: tuple[tuple[str, str], ...] = eqx.field(static=True)
    local_physical_boundary_facets: Array | None
    local_neighborhood_complete: Array | None
    neighborhood_depth: int = eqx.field(static=True)
    local_coordinate_id: str = eqx.field(static=True)
    global_entity_counts: tuple[int, ...] = eqx.field(static=True)
    partition_index: int = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    logical_topology_id: str = eqx.field(static=True)
    logical_geometry_id: str = eqx.field(static=True)
    logical_coordinate_geometry_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    maximum_materialization_bytes: int = eqx.field(static=True)
    storage_id: str = eqx.field(static=True)

    def __init__(
        self,
        global_entity_counts: Sequence[int],
        entity_global_ids: Sequence[ArrayLike],
        entity_owner: Sequence[ArrayLike],
        /,
        *,
        partition_index: int,
        partition_count: int,
        logical_topology_id: str,
        logical_geometry_id: str,
        evidence_id: str,
        logical_arrays: Sequence[tuple[str, Array]],
        local_coordinates: ArrayLike,
        local_blocks: Sequence[CellBlock | PolyhedralBlock],
        maximum_materialization_bytes: int = 0,
        local_physical_boundary_facets: ArrayLike | None = None,
        local_neighborhood_complete: ArrayLike | None = None,
        neighborhood_depth: int = 0,
        global_coordinate_count: int | None = None,
        coordinate_global_ids: ArrayLike | None = None,
        coordinate_owner: ArrayLike | None = None,
        local_geometry: CellGeometrySpec | None = None,
        geometry_source_blocks: Mapping[str, str] | None = None,
        logical_coordinate_geometry_id: str | None = None,
        geometry_projection: CellGeometryStorageProjection | None = None,
    ) -> None:
        if not jax.config.jax_enable_x64:
            raise ValueError(
                "Owner-local scientific IDs and coordinate witnesses require JAX x64 execution."
            )
        counts = tuple(operator.index(value) for value in global_entity_counts)
        rank = operator.index(partition_index)
        parts = operator.index(partition_count)
        bound = operator.index(maximum_materialization_bytes)
        if (
            len(counts) < 2
            or any(
                isinstance(value, bool) or value <= 0 for value in global_entity_counts
            )
            or isinstance(partition_index, bool)
            or isinstance(partition_count, bool)
            or not 0 <= rank < parts
            or isinstance(maximum_materialization_bytes, bool)
            or bound < 0
        ):
            raise ValueError(
                "Owner-local storage requires positive counts and a valid placement."
            )
        if len(entity_global_ids) != len(counts) or len(entity_owner) != len(counts):
            raise ValueError("Each declared entity degree requires IDs and ownership.")
        ids = []
        owners = []
        for count, values, placement in zip(
            counts, entity_global_ids, entity_owner, strict=True
        ):
            if (isinstance(values, Array) and not values.is_fully_addressable) or (
                isinstance(placement, Array) and not placement.is_fully_addressable
            ):
                raise ValueError("Local ID and ownership maps must be fully addressable.")
            identifiers = np.asarray(values)
            routing = np.asarray(placement)
            if (
                identifiers.ndim != 1
                or identifiers.dtype.kind not in "iu"
                or identifiers.size > count
                or routing.shape != identifiers.shape
                or routing.dtype.kind not in "iu"
                or np.any(identifiers < 0)
                or np.any(identifiers > np.iinfo(np.int64).max)
                or np.unique(identifiers).size != identifiers.size
                or np.any(routing < 0)
                or np.any(routing >= parts)
            ):
                raise ValueError(
                    "Local entity IDs and ownership violate the logical storage contract."
                )
            ids.append(jnp.asarray(identifiers, dtype=jnp.int64))
            owners.append(jnp.asarray(routing, dtype=jnp.int32))
        if (
            isinstance(local_coordinates, Array)
            and not local_coordinates.is_fully_addressable
        ):
            raise ValueError("Local coordinate evidence must be fully addressable.")
        coordinates = np.asarray(local_coordinates, dtype=np.float64)
        if (
            coordinates.ndim != 2
            or coordinates.shape[0] != ids[0].shape[0]
            or coordinates.shape[1] <= 0
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError(
                "Local coordinate evidence must match finite locally routed vertices."
            )
        local_coordinate_id = canonical_fingerprint(array_tree_fingerprint(coordinates))
        if (
            local_geometry is not None or geometry_projection is not None
        ) and logical_coordinate_geometry_id is None:
            raise ValueError(
                "Mapped storage requires a distinct scientific coordinate-map identity."
            )
        coordinate_geometry_id = (
            logical_geometry_id
            if logical_coordinate_geometry_id is None
            else logical_coordinate_geometry_id
        )
        identities = (
            logical_topology_id,
            logical_geometry_id,
            coordinate_geometry_id,
            evidence_id,
        )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in identities
        ):
            raise ValueError(
                "Logical storage identities must be canonical SHA-256 digests."
            )
        arrays = tuple(logical_arrays)
        names = tuple(name for name, _ in arrays)
        if (
            not arrays
            or names != tuple(sorted(names))
            or len(set(names)) != len(names)
            or any(not name or not isinstance(value, Array) for name, value in arrays)
        ):
            raise ValueError(
                "Logical arrays require uniquely named canonical ordered JAX arrays."
            )
        blocks = tuple(local_blocks)
        if blocks:
            _validated_mesh_blocks(coordinates.shape, blocks)
            if not np.array_equal(
                np.concatenate(tuple(np.asarray(block.global_ids) for block in blocks)),
                np.asarray(ids[-1]),
            ):
                raise ValueError(
                    "Local incidence witnesses must match the routed cell identities."
                )
        elif (
            coordinates.shape[0] != 0
            or any(value.shape[0] != 0 for value in ids)
            or coordinates.shape[1] < len(counts) - 1
            or np.asarray(local_coordinates).dtype != np.float64
            or any(np.asarray(value).dtype != np.int64 for value in entity_global_ids)
            or any(np.asarray(value).dtype != np.int32 for value in entity_owner)
        ):
            raise ValueError(
                "Zero-resident storage requires an actual empty, exactly typed local closure."
            )
        source_blocks = (
            {block.name: block.name for block in blocks}
            if geometry_source_blocks is None
            else dict(geometry_source_blocks)
        )
        if set(source_blocks) != {block.name for block in blocks} or any(
            not isinstance(value, str) or not value for value in source_blocks.values()
        ):
            raise ValueError(
                "Every local coordinate block requires one explicit scientific source bank."
            )
        from ._cell_geometry import (
            _require_logical_geometry_lowering,
            _restore_authored_storage_geometry,
            CellGeometrySpec,
            CellGeometryStorageProjection,
        )
        from ._cell_geometry_validity import cell_geometry_id

        if geometry_projection is not None and not isinstance(
            geometry_projection, CellGeometryStorageProjection
        ):
            raise TypeError(
                "geometry_projection must be an actual CellGeometryStorageProjection."
            )
        packets = None
        if geometry_projection is not None:
            geometry_projection.require_source(arrays, coordinate_geometry_id)
            if (
                geometry_projection.partition_count != parts
                or geometry_projection.global_cell_count != counts[-1]
            ):
                raise ValueError(
                    "Geometry projection differs from the actual globally nonempty mesh placement."
                )
            packets = dict(geometry_projection.addressable_arrays(rank))
        if not blocks:
            if (
                geometry_projection is None
                or packets is None
                or not geometry_projection.source_elements
                or "geometry/owned_cell_count" not in packets
                or int(np.asarray(packets["geometry/owned_cell_count"])) != 0
                or packets["geometry/coordinate_ids"].shape != (0,)
                or packets["geometry/coordinates"].shape != coordinates.shape
                or local_geometry is not None
                or packets["geometry/coordinate_owners"].shape != (0,)
            ):
                raise ValueError(
                    "Zero-resident storage requires its actual passive-owner scientific source projection."
                )
        if local_geometry is None and blocks and geometry_projection is not None:
            if coordinate_global_ids is None:
                raise ValueError(
                    "Authored source geometry requires explicit scientific coordinate IDs."
                )
            local_geometry = _restore_authored_storage_geometry(
                blocks,
                source_blocks,
                geometry_projection,
                rank,
                coordinate_global_ids,
            )

        if local_geometry is None and geometry_projection is None:
            if any(
                value is not None
                for value in (
                    global_coordinate_count,
                    coordinate_global_ids,
                    coordinate_owner,
                    geometry_projection,
                )
            ):
                raise ValueError(
                    "Explicit coordinate identity requires its scientific geometry layout."
                )
            coordinate_count = counts[0]
            coordinate_ids = ids[0]
            coordinate_owners = owners[0]
            geometry_source_id = local_coordinate_id
        else:
            if local_geometry is not None and (
                not isinstance(local_geometry, CellGeometrySpec)
                or local_geometry.storage is not None
            ):
                raise TypeError("local_geometry must be an unbound CellGeometrySpec.")
            if (
                global_coordinate_count is None
                or coordinate_global_ids is None
                or coordinate_owner is None
            ):
                raise ValueError(
                    "Scientific storage requires explicit coordinate DOF count, IDs, and owners."
                )
            coordinate_count = operator.index(global_coordinate_count)
            if isinstance(global_coordinate_count, bool) or coordinate_count <= 0:
                raise ValueError("Global coordinate DOF count must be positive.")
            if any(
                isinstance(value, Array) and not value.is_fully_addressable
                for value in (coordinate_global_ids, coordinate_owner)
            ):
                raise ValueError(
                    "Coordinate DOF identity lowering must be locally addressable."
                )
            coordinate_ids_ = np.asarray(coordinate_global_ids)
            coordinate_owners_ = np.asarray(coordinate_owner)
            if (
                coordinate_ids_.ndim != 1
                or coordinate_ids_.dtype.kind not in "iu"
                or coordinate_ids_.size > coordinate_count
                or coordinate_ids_.size
                != (0 if local_geometry is None else local_geometry.coordinates.shape[0])
                or np.unique(coordinate_ids_).size != coordinate_ids_.size
                or np.any(coordinate_ids_ < 0)
                or np.any(coordinate_ids_ > np.iinfo(np.int64).max)
                or coordinate_owners_.shape != coordinate_ids_.shape
                or coordinate_owners_.dtype.kind not in "iu"
                or np.any(coordinate_owners_ < 0)
                or np.any(coordinate_owners_ >= parts)
            ):
                raise ValueError(
                    "Coordinate DOF identities and ownership violate their scientific layout."
                )
            coordinate_ids = jnp.asarray(coordinate_ids_, dtype=jnp.int64)
            coordinate_owners = jnp.asarray(coordinate_owners_, dtype=jnp.int32)
            witness_arrays = arrays
            witness_count = coordinate_count
            if geometry_projection is None and any(
                name.startswith("geometry/") and not value.is_fully_addressable
                for name, value in arrays
            ):
                raise ValueError(
                    "Non-addressable scientific banks require their collective geometry projection."
                )
            if geometry_projection is not None:
                if (
                    geometry_projection.global_coordinate_count != coordinate_count
                    or geometry_projection.partition_count != parts
                ):
                    raise ValueError(
                        "Collective coordinate projection differs from scientific storage placement."
                    )
                geometry_projection.require_source(arrays, coordinate_geometry_id)
                witness_arrays = geometry_projection.addressable_arrays(rank)
                witness_count = dict(witness_arrays)["geometry/coordinate_ids"].shape[0]
            if local_geometry is not None:
                _require_logical_geometry_lowering(
                    local_geometry,
                    blocks,
                    witness_arrays,
                    witness_count,
                    coordinate_ids,
                    coordinate_owners,
                    source_blocks,
                )
                geometry_source_id = cell_geometry_id(local_geometry)
            else:
                if (
                    coordinate_ids_.dtype != np.int64
                    or coordinate_owners_.dtype != np.int32
                ):
                    raise ValueError(
                        "Zero-resident scientific coordinate routes require exact ID and owner dtypes."
                    )
                geometry_source_id = canonical_fingerprint(
                    {
                        "kind": "zero-resident-coordinate-source",
                        "coordinate_geometry": coordinate_geometry_id,
                        "coordinates": array_tree_fingerprint(coordinates),
                    }
                )
        depth = operator.index(neighborhood_depth)
        if isinstance(neighborhood_depth, bool) or depth < 0:
            raise ValueError("neighborhood_depth must be a non-negative integer.")

        def local_mask(value: ArrayLike | None, count: int, name: str) -> Array | None:
            if value is None:
                return None
            if isinstance(value, Array) and not value.is_fully_addressable:
                raise ValueError(f"{name} must be a locally addressable witness.")
            mask = np.asarray(value)
            if mask.shape != (count,) or mask.dtype != np.bool_:
                raise ValueError(
                    f"{name} must match the local entity count with boolean dtype."
                )
            return jnp.asarray(mask, dtype=jnp.bool_)

        physical = local_mask(
            local_physical_boundary_facets,
            ids[-2].shape[0],
            "local_physical_boundary_facets",
        )
        complete = local_mask(
            local_neighborhood_complete, ids[-1].shape[0], "local_neighborhood_complete"
        )
        self.entity_global_ids = tuple(ids)
        self.entity_owner = tuple(owners)
        self.entity_owned = tuple(value == rank for value in owners)
        self.logical_arrays = arrays
        self.local_coordinates = jnp.asarray(coordinates)
        self.local_blocks = blocks
        self.coordinate_global_ids = coordinate_ids
        self.coordinate_owner = coordinate_owners
        self.global_coordinate_count = coordinate_count
        self.local_geometry = local_geometry
        self.geometry_projection = geometry_projection
        self.local_geometry_source_id = geometry_source_id
        self.geometry_source_blocks = tuple(sorted(source_blocks.items()))
        self.local_physical_boundary_facets = physical
        self.local_neighborhood_complete = complete
        self.neighborhood_depth = depth
        self.local_coordinate_id = local_coordinate_id
        self.global_entity_counts = counts
        self.partition_index = rank
        self.partition_count = parts
        self.logical_topology_id = logical_topology_id
        self.logical_geometry_id = logical_geometry_id
        self.logical_coordinate_geometry_id = coordinate_geometry_id
        self.evidence_id = evidence_id
        self.maximum_materialization_bytes = bound
        self.storage_id = canonical_fingerprint(
            {
                "kind": "owner-local-cell-mesh-storage",
                "topology": logical_topology_id,
                "geometry": logical_geometry_id,
                "coordinate_geometry": coordinate_geometry_id,
                "coverage": evidence_id,
                "local_coordinate_id": local_coordinate_id,
                "local_geometry_source_id": geometry_source_id,
                "geometry_source_blocks": tuple(sorted(source_blocks.items())),
                "global_coordinate_count": coordinate_count,
                "coordinate_ids": array_tree_fingerprint(coordinate_ids),
                "coordinate_owners": array_tree_fingerprint(coordinate_owners),
                "local_blocks": [block.block_id for block in blocks],
                "local_physical_boundary_facets": (
                    None if physical is None else array_tree_fingerprint(physical)
                ),
                "local_neighborhood_complete": (
                    None if complete is None else array_tree_fingerprint(complete)
                ),
                "neighborhood_depth": depth,
                "global_entity_counts": counts,
                "partition_index": rank,
                "partition_count": parts,
                "local_ids": array_tree_fingerprint(tuple(ids)),
                "local_owners": array_tree_fingerprint(tuple(owners)),
                "logical_arrays": [
                    (name, value.shape, value.dtype.str) for name, value in arrays
                ],
                "maximum_materialization_bytes": bound,
            }
        )

    def require_local_topology(self, topology: CellComplexTopology, /) -> None:
        if topology.dimension + 1 != len(self.global_entity_counts):
            raise ValueError(
                "Storage entity degrees differ from the local closure topology."
            )
        for degree, identifiers in enumerate(self.entity_global_ids):
            if not np.array_equal(
                np.asarray(identifiers), np.asarray(topology.entities(degree).entity_ids)
            ):
                raise ValueError("Storage IDs differ from the local topology lowering.")

    def restore_geometry(self) -> CellGeometrySpec:
        """Restore the actual coefficient basis, never the sampled corner carrier."""
        from ._cell_geometry import (
            CellGeometrySpec,
            CellVertexGeometryElement,
            coordinate_lagrange_element,
        )

        geometry = self.local_geometry
        if geometry is None:
            return CellGeometrySpec(
                {
                    block.name: (
                        CellVertexGeometryElement(block.cell_kind, block.arity)
                        if block.cell_kind in ("polygon", "polyhedron")
                        else coordinate_lagrange_element(block.cell_kind, 1)
                    )
                    for block in self.local_blocks
                },
                {block.name: block.vertices for block in self.local_blocks},
                self.local_coordinates,
                storage=self,
            )
        return CellGeometrySpec(
            dict(zip(geometry.block_names, geometry.elements, strict=True)),
            dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
            geometry.coordinates,
            storage=self,
            restriction_source=geometry.restriction_source,
            periodic_source=geometry.periodic_source,
            exact_source=geometry.exact_source,
        )


def _empty_storage_topology(storage: CellMeshStorage, /) -> _PreparedCellMeshTopology:
    """Construct the actual zero subcomplex of a proven nonempty logical carrier."""
    if storage.local_blocks or any(value.shape[0] for value in storage.entity_global_ids):
        raise ValueError("Zero topology requires exactly zero resident entities.")
    projection = storage.geometry_projection
    if projection is None or not projection.source_elements:
        raise ValueError(
            "Zero topology requires its actual global scientific source layout."
        )
    dimension = len(storage.global_entity_counts) - 1
    source = dict(projection.source_arrays)
    for bank, element in projection.source_elements:
        matrix = source.get(f"geometry/matrix/{bank}")
        target_dimension = (
            (
                2
                if element.cell_kind == "polygon"
                else 3
                if element.cell_kind == "polyhedron"
                else reference_cell_topology(element.cell_kind).dimension
            )
            if matrix is None
            else matrix.shape[-1]
        )
        if target_dimension != dimension:
            raise ValueError(
                "Global coordinate charts differ from the declared mesh topological degree."
            )
    integer = jnp.empty((0,), dtype=jnp.int32)
    boolean = jnp.empty((0,), dtype=jnp.bool_)
    signs = jnp.empty((0,), dtype=jnp.float64)
    connectivity: _CellMeshConnectivity
    if dimension == 1:
        connectivity = IntervalConnectivity(
            jnp.empty((0, 2), dtype=jnp.int32), integer, boolean, 0, 0
        )
    elif dimension == 2:
        connectivity = PolygonalConnectivity(
            edges=jnp.empty((0, 2), dtype=jnp.int32),
            cell_vertices=jnp.empty((0, 0), dtype=jnp.int32),
            cell_vertex_valid=jnp.empty((0, 0), dtype=jnp.bool_),
            cell_kinds=integer,
            cell_edges=jnp.empty((0, 0), dtype=jnp.int32),
            cell_edge_signs=jnp.empty((0, 0), dtype=jnp.float64),
            cell_edge_valid=jnp.empty((0, 0), dtype=jnp.bool_),
            edge_cell_counts=integer,
            boundary_edges=boolean,
            boundary_vertices=boolean,
            vertex_count=0,
            triangle_count=0,
            quadrilateral_count=0,
            polygon_count=0,
        )
    elif dimension == 3:
        offsets = jnp.zeros((1,), dtype=jnp.int32)
        connectivity = PolyhedralConnectivity(
            edges=jnp.empty((0, 2), dtype=jnp.int32),
            face_vertex_offsets=offsets,
            face_vertex_values=integer,
            face_edge_offsets=offsets,
            face_edge_values=integer,
            face_edge_sign_values=signs,
            cell_face_offsets=offsets,
            cell_face_values=integer,
            cell_face_sign_values=signs,
            cell_vertex_offsets=offsets,
            cell_vertex_values=integer,
            face_owner=integer,
            face_neighbor=integer,
            face_owner_local=integer,
            face_neighbor_local=integer,
            face_cell_counts=integer,
            boundary_vertices=boolean,
            boundary_edges=boolean,
            boundary_faces=boolean,
            vertex_global_ids=storage.entity_global_ids[0],
            edge_global_ids=storage.entity_global_ids[1],
            face_global_ids=storage.entity_global_ids[2],
            cell_global_ids=storage.entity_global_ids[3],
            vertex_count=0,
            edge_count=0,
            face_count=0,
            cell_count=0,
            maximum_face_arity=0,
            maximum_cell_faces=0,
            maximum_cell_vertices=0,
        )
    names = tuple(
        "vertices"
        if degree == 0
        else "cells"
        if degree == dimension
        else "edges"
        if degree == 1
        else "faces"
        if degree == 2
        else f"entities-{degree}"
        for degree in range(dimension + 1)
    )
    entities = tuple(
        EntitySet(
            name,
            degree,
            storage.entity_global_ids[degree],
            subsets=(EntitySubset("boundary", boolean),),
        )
        for degree, name in enumerate(names)
    )
    incidences = tuple(
        OrientedIncidence(
            degree,
            entities[degree - 1],
            entities[degree],
            EdgeRelation(integer, integer, source_size=0, target_size=0),
            signs,
        )
        for degree in range(1, dimension + 1)
    )
    topology = CellComplexTopology(entities, incidences)
    if dimension > 3:
        reference = reference_cell_topology(f"simplex:{dimension}")
        routes = tuple(
            jnp.empty((0, len(level)), dtype=jnp.int32) for level in reference.entities
        )
        connectivity = SimplicialConnectivity(
            topology,
            jnp.empty((0, dimension + 1), dtype=jnp.int32),
            entities=tuple(
                jnp.empty((0, degree + 1), dtype=jnp.int32)
                for degree in range(dimension + 1)
            ),
            cell_entities=routes,
            cell_entity_signs=tuple(
                jnp.empty(value.shape, dtype=jnp.float64) for value in routes
            ),
            boundary_masks=(boolean,) * (dimension + 1),
        )
    return _PreparedCellMeshTopology(
        (),
        storage.entity_global_ids[0],
        connectivity,
        topology,
        storage.logical_topology_id,
        dimension,
        None,
    )


class CellMesh(StrictModule, NonTrainableState):
    """Canonical computational mesh shared by unstructured discretizations.

    A periodic mesh is its finite lifted carrier plus ``periodic_topology``, the
    quotient descriptor built from the plain lifted mesh; the descriptor is part
    of topology identity. Boundary-paired meshes carry no descriptor.
    """

    coordinates: Array
    blocks: tuple[CellBlock | PolyhedralBlock, ...]
    vertex_global_ids: Array
    connectivity: (
        IntervalConnectivity
        | PolygonalConnectivity
        | TetrahedralConnectivity
        | HexahedralConnectivity
        | PolyhedralConnectivity
        | SimplicialConnectivity
    )
    topology: CellComplexTopology
    support: DiscreteSupport
    periodic_topology: PeriodicMeshTopology | None
    storage: CellMeshStorage | None
    topological_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        blocks: Sequence[CellBlock | PolyhedralBlock],
        /,
        *,
        vertex_global_ids: ArrayLike | None = None,
        entity_global_ids: Mapping[int, ArrayLike] | None = None,
        polyhedral_connectivity: PolyhedralConnectivity | None = None,
        periodic_topology: PeriodicMeshTopology | None = None,
        numeric_version: str = "0",
        storage: CellMeshStorage | None = None,
        _prepared_topology: _PreparedCellMeshTopology | None = None,
    ) -> None:
        if storage is not None and not isinstance(storage, CellMeshStorage):
            raise TypeError("storage must be CellMeshStorage or None.")
        if isinstance(coordinates, Array) and not coordinates.is_fully_addressable:
            raise ValueError(
                "CellMesh coordinates require a local view; bind global arrays in storage."
            )
        if storage is not None and _prepared_topology is None:
            if vertex_global_ids is not None or entity_global_ids is not None:
                raise ValueError(
                    "Owner-local entity identities are owned by the storage descriptor."
                )
            entity_global_ids = dict(enumerate(storage.entity_global_ids))
        points = np.asarray(coordinates, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] == 0
            or (points.shape[0] == 0 and storage is None)
        ):
            raise ValueError(
                "Cell mesh coordinates require a nonempty serial mesh or bound zero-resident storage."
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("Cell mesh coordinates must be finite.")
        if (
            storage is not None
            and canonical_fingerprint(array_tree_fingerprint(points))
            != storage.local_coordinate_id
        ):
            raise ValueError(
                "Local mesh coordinates differ from their logical storage witness."
            )
        if points.shape[0] == 0 and storage is not None:
            if (
                blocks
                or polyhedral_connectivity is not None
                or periodic_topology is not None
            ):
                raise ValueError(
                    "Zero-resident meshes require their exact empty storage closure."
                )
            prepared = _empty_storage_topology(storage)
        elif _prepared_topology is None:
            prepared = _prepare_cell_mesh_topology(
                points.shape,
                blocks,
                vertex_global_ids=vertex_global_ids,
                entity_global_ids=entity_global_ids,
                polyhedral_connectivity=polyhedral_connectivity,
                periodic_topology=periodic_topology,
            )
        else:
            prepared = _reused_cell_mesh_topology(
                _prepared_topology,
                points.shape,
                blocks,
                identities_supplied=(
                    vertex_global_ids is not None
                    or entity_global_ids is not None
                    or polyhedral_connectivity is not None
                    or periodic_topology is not None
                ),
            )
        if prepared.periodic_topology is not None:
            prepared.periodic_topology.require_lift(points)
        if storage is not None:
            storage.require_local_topology(prepared.topology)
            if tuple(block.block_id for block in prepared.blocks) != tuple(
                block.block_id for block in storage.local_blocks
            ):
                raise ValueError("Local mesh incidence differs from its storage witness.")
        topology_id = (
            prepared.topology_id if storage is None else storage.logical_topology_id
        )
        geometry_layout_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-geometry-layout",
                "topology": topology_id,
                "ambient_dimension": points.shape[1],
                "coordinate_count": (
                    points.shape[0]
                    if storage is None
                    else storage.global_entity_counts[0]
                ),
                "coordinate_dtype": str(points.dtype),
            }
        )
        if storage is not None:
            geometry_id = storage.logical_geometry_id
        elif (
            len(prepared.blocks) == 1
            and isinstance(prepared.blocks[0], CellBlock)
            and prepared.blocks[0].cell_kind in ("interval", "triangle", "tetrahedron")
        ):
            order = np.argsort(prepared.vertex_global_ids, kind="stable")
            geometry_id = canonical_fingerprint(
                {
                    "kind": "cell-mesh-geometry",
                    "topology": topology_id,
                    "ambient_dimension": points.shape[1],
                    "coordinates": array_collection_digest(
                        {"coordinates": points[order]}
                    ),
                }
            )
        else:
            geometry_id = canonical_fingerprint(
                {
                    "kind": "cell-mesh-geometry",
                    "layout": geometry_layout_id,
                    "coordinates": array_tree_fingerprint(points),
                    "numeric_version": str(numeric_version),
                }
            )
        support = DiscreteSupport(prepared.topology, points.shape[1], geometry_layout_id)
        self.coordinates = (
            jnp.asarray(points) if storage is None else storage.local_coordinates
        )
        self.blocks = prepared.blocks
        self.vertex_global_ids = jnp.asarray(prepared.vertex_global_ids)
        self.topology = prepared.topology
        self.connectivity = prepared.connectivity
        self.support = support
        self.periodic_topology = prepared.periodic_topology
        self.storage = storage
        self.topological_dimension = prepared.topological_dimension
        self.ambient_dimension = points.shape[1]
        self.topology_id = topology_id
        self.geometry_layout_id = geometry_layout_id
        self.geometry_id = geometry_id
        self.mesh_id = canonical_fingerprint(
            {
                "kind": "cell-mesh",
                "topology": topology_id,
                "geometry": geometry_id,
            }
        )
        self.numeric_version = str(numeric_version)

    def require_dense(self, operation: str, /) -> None:
        """Guard a serial-only consumer before any implicit global conversion."""
        if self.storage is not None:
            raise ValueError(
                f"{operation} requires explicit bounded materialization of owner-local CellMesh storage."
            )

    @classmethod
    def from_simplices(
        cls,
        coordinates: ArrayLike,
        simplices: ArrayLike,
        /,
        *,
        dimension: int,
        block_name: str = "simplices",
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
    ) -> CellMesh:
        """Build simplicial mesh geometry with an explicitly declared intrinsic dimension."""
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("Simplex dimension must be an integer.")
        if dimension < 1:
            raise ValueError("Simplex dimension must be positive.")
        return cls(
            coordinates,
            (
                CellBlock(
                    block_name,
                    f"simplex:{dimension}",
                    simplices,
                    global_ids=cell_global_ids,
                ),
            ),
            vertex_global_ids=vertex_global_ids,
            numeric_version=numeric_version,
        )

    @classmethod
    def from_triangles(
        cls,
        coordinates: ArrayLike,
        triangles: ArrayLike,
        /,
        *,
        block_name: str = "triangles",
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
    ) -> CellMesh:
        return cls(
            coordinates,
            (
                CellBlock(
                    block_name,
                    "triangle",
                    triangles,
                    global_ids=cell_global_ids,
                ),
            ),
            vertex_global_ids=vertex_global_ids,
            numeric_version=numeric_version,
        )

    @classmethod
    def from_polygons(
        cls,
        coordinates: ArrayLike,
        polygons: Sequence[ArrayLike],
        /,
        *,
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
    ) -> CellMesh:
        """Build a canonical mixed-arity polygon mesh from cyclic vertex loops."""

        points = np.asarray(coordinates, dtype=np.float64)
        loops = tuple(np.asarray(loop, dtype=np.int32) for loop in polygons)
        if not loops:
            raise ValueError("from_polygons requires at least one cell.")
        if any(loop.ndim != 1 or loop.size < 3 for loop in loops):
            raise ValueError(
                "Every polygon must be one rank-1 loop with at least 3 vertices."
            )
        identifiers = (
            np.arange(len(loops), dtype=np.int64)
            if cell_global_ids is None
            else np.asarray(cell_global_ids, dtype=np.int64)
        )
        if identifiers.shape != (len(loops),):
            raise ValueError("cell_global_ids must have one ID per polygon.")
        grouped: dict[int, list[tuple[np.ndarray, int]]] = {}
        for loop, identifier in zip(loops, identifiers, strict=True):
            if np.any(loop < 0) or np.any(loop >= points.shape[0]):
                raise ValueError("Polygon loop indexes an undeclared vertex.")
            cell_points = points[loop]
            area2 = float(
                np.sum(
                    cell_points[:, 0] * np.roll(cell_points[:, 1], -1)
                    - np.roll(cell_points[:, 0], -1) * cell_points[:, 1]
                )
            )
            if not np.isfinite(area2) or area2 == 0.0:
                raise ValueError("Polygon loops require finite nonzero signed area.")
            oriented = loop[::-1] if area2 < 0.0 else loop
            grouped.setdefault(loop.size, []).append((oriented, int(identifier)))
        blocks = []
        for arity in sorted(grouped):
            entries = grouped[arity]
            kind = (
                "triangle" if arity == 3 else "quadrilateral" if arity == 4 else "polygon"
            )
            blocks.append(
                CellBlock(
                    f"polygons-{arity}",
                    kind,
                    np.stack(tuple(entry[0] for entry in entries)),
                    global_ids=np.asarray(
                        tuple(entry[1] for entry in entries), dtype=np.int64
                    ),
                )
            )
        return cls(
            points,
            tuple(blocks),
            vertex_global_ids=vertex_global_ids,
            numeric_version=numeric_version,
        )

    @classmethod
    def from_tetrahedra(
        cls,
        coordinates: ArrayLike,
        tetrahedra: ArrayLike,
        /,
        *,
        block_name: str = "tetrahedra",
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
    ) -> CellMesh:
        return cls(
            coordinates,
            (
                CellBlock(
                    block_name,
                    "tetrahedron",
                    tetrahedra,
                    global_ids=cell_global_ids,
                ),
            ),
            vertex_global_ids=vertex_global_ids,
            numeric_version=numeric_version,
        )

    @classmethod
    def from_polyhedra(
        cls,
        coordinates: ArrayLike,
        cells: Sequence[Sequence[ArrayLike]],
        /,
        *,
        block_name: str = "polyhedra",
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
    ) -> CellMesh:
        """Build a canonical mesh from closed outward-oriented face loops."""

        points = np.asarray(coordinates, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                "Polyhedral coordinates must have shape (vertex_count, d >= 3)."
            )
        normalized_cells = tuple(tuple(cell) for cell in cells)
        if not normalized_cells:
            raise ValueError("At least one polyhedral cell is required.")
        source_ids = (
            np.arange(len(normalized_cells), dtype=np.int64)
            if cell_global_ids is None
            else np.asarray(cell_global_ids, dtype=np.int64)
        )
        if source_ids.shape != (len(normalized_cells),):
            raise ValueError("cell_global_ids must match the polyhedral cell count.")
        vertex_counts = np.asarray(
            [
                len(
                    {
                        int(vertex)
                        for face in cell
                        for vertex in np.asarray(face, dtype=np.int32)
                    }
                )
                for cell in normalized_cells
            ],
            dtype=np.int32,
        )
        order = np.argsort(vertex_counts, kind="stable")
        ordered_cells = tuple(normalized_cells[int(index)] for index in order)
        ordered_ids = source_ids[order]
        ordered_counts = vertex_counts[order]
        connectivity = _build_polyhedral_connectivity(
            ordered_cells,
            points.shape[0],
            vertex_global_ids=vertex_global_ids,
            cell_global_ids=ordered_ids,
        )
        offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int32)
        values = np.asarray(connectivity.cell_vertex_values, dtype=np.int32)
        blocks = []
        start_cell = 0
        unique_counts = tuple(np.unique(ordered_counts))
        for vertex_count in unique_counts:
            stop_cell = start_cell + int(np.sum(ordered_counts == vertex_count))
            start = int(offsets[start_cell])
            stop = int(offsets[stop_cell])
            name = (
                str(block_name)
                if len(unique_counts) == 1
                else f"{block_name}-{vertex_count}"
            )
            blocks.append(
                PolyhedralBlock(
                    name,
                    values[start:stop].reshape((-1, vertex_count)),
                    global_ids=ordered_ids[start_cell:stop_cell],
                )
            )
            start_cell = stop_cell
        return cls(
            points,
            tuple(blocks),
            vertex_global_ids=connectivity.vertex_global_ids,
            polyhedral_connectivity=connectivity,
            numeric_version=numeric_version,
        )

    @classmethod
    def from_mixed_3d(
        cls,
        coordinates: ArrayLike,
        blocks: Sequence[CellBlock],
        /,
        *,
        polyhedra: Mapping[str, Sequence[Sequence[ArrayLike]]],
        vertex_global_ids: ArrayLike | None = None,
        polyhedral_cell_global_ids: Mapping[str, ArrayLike] | None = None,
        numeric_version: str = "0",
    ) -> CellMesh:
        """Build one mixed standard/polyhedral three-dimensional mesh."""

        points = np.asarray(coordinates, dtype=np.float64)
        standard_blocks = tuple(blocks)
        if any(
            block.cell_kind not in ("tetrahedron", "hexahedron", "prism", "pyramid")
            for block in standard_blocks
        ):
            raise ValueError("Mixed 3-D dense blocks use standard volume cell kinds.")
        named_cells = tuple(
            (str(name), tuple(tuple(cell) for cell in cells))
            for name, cells in sorted(polyhedra.items())
        )
        if any(not name or not cells for name, cells in named_cells):
            raise ValueError(
                "Polyhedral block names and cell collections must be non-empty."
            )
        id_mapping = (
            {}
            if polyhedral_cell_global_ids is None
            else {
                str(name): np.asarray(values, dtype=np.int64)
                for name, values in polyhedral_cell_global_ids.items()
            }
        )
        if set(id_mapping) not in (set(), {name for name, _ in named_cells}):
            raise ValueError(
                "polyhedral_cell_global_ids must cover every polyhedral block."
            )
        used_ids = [
            int(value)
            for block in standard_blocks
            for value in np.asarray(block.global_ids, dtype=np.int64)
        ]
        next_id = max(used_ids, default=-1) + 1
        poly_blocks: list[PolyhedralBlock] = []
        explicit_cells: list[tuple[ArrayLike, ...]] = []
        explicit_ids: list[int] = []
        for name, cells in named_cells:
            ids = id_mapping.get(name)
            if ids is None:
                ids = np.arange(next_id, next_id + len(cells), dtype=np.int64)
            if ids.shape != (len(cells),):
                raise ValueError(
                    f"Polyhedral global IDs for {name!r} must match its cell count."
                )
            counts = np.asarray(
                [
                    len(
                        {
                            int(vertex)
                            for face in cell
                            for vertex in np.asarray(face, dtype=np.int32)
                        }
                    )
                    for cell in cells
                ],
                dtype=np.int32,
            )
            order = np.argsort(counts, kind="stable")
            ordered_cells = tuple(cells[int(index)] for index in order)
            ordered_ids = ids[order]
            ordered_counts = counts[order]
            for vertex_count in tuple(np.unique(ordered_counts)):
                selected = np.flatnonzero(ordered_counts == vertex_count)
                selected_cells = tuple(ordered_cells[int(index)] for index in selected)
                selected_ids = ordered_ids[selected]
                vertices = np.asarray(
                    [
                        sorted(
                            {
                                int(vertex)
                                for face in cell
                                for vertex in np.asarray(face, dtype=np.int32)
                            }
                        )
                        for cell in selected_cells
                    ],
                    dtype=np.int32,
                )
                block_name = (
                    name
                    if len(np.unique(ordered_counts)) == 1
                    else f"{name}-{vertex_count}"
                )
                poly_blocks.append(
                    PolyhedralBlock(
                        block_name,
                        vertices,
                        global_ids=selected_ids,
                    )
                )
                explicit_cells.extend(selected_cells)
                explicit_ids.extend(int(value) for value in selected_ids)
            next_id = max(next_id, int(np.max(ids, initial=next_id - 1)) + 1)
        combined_blocks = (*standard_blocks, *poly_blocks)
        if not combined_blocks:
            raise ValueError("Mixed 3-D mesh requires at least one cell block.")
        entries: list[Sequence[ArrayLike] | tuple[str, ArrayLike]] = [
            (block.cell_kind, np.asarray(block.vertices, dtype=np.int32))
            for block in standard_blocks
        ]
        entries.extend(explicit_cells)
        all_ids = np.concatenate(
            (
                *(
                    np.asarray(block.global_ids, dtype=np.int64)
                    for block in standard_blocks
                ),
                np.asarray(explicit_ids, dtype=np.int64),
            )
        )
        connectivity = _build_polyhedral_connectivity(
            entries,
            points.shape[0],
            vertex_global_ids=vertex_global_ids,
            cell_global_ids=all_ids,
        )
        return cls(
            points,
            combined_blocks,
            vertex_global_ids=connectivity.vertex_global_ids,
            polyhedral_connectivity=connectivity,
            numeric_version=numeric_version,
        )

    def block(self, name: str, /) -> CellBlock | PolyhedralBlock:
        requested = str(name)
        for block in self.blocks:
            if block.name == requested:
                return block
        raise KeyError(f"Unknown cell block {requested!r}.")

    def entity_set(self, dimension: int, /) -> EntitySet:
        target = int(dimension)
        for entities in self.topology.entity_sets:
            if entities.intrinsic_dimension == target:
                return entities
        raise KeyError(f"Cell mesh has no entity set of dimension {target}.")

    @property
    def boundary_masks(self) -> tuple[Array, ...]:
        """Boundary subcomplex membership in each explicitly declared entity degree."""
        return tuple(
            entities.subset("boundary").mask for entities in self.topology.entity_sets
        )

    def with_coordinates(
        self,
        coordinates: ArrayLike,
        /,
        *,
        numeric_version: str,
        storage: CellMeshStorage | None = None,
    ) -> CellMesh:
        if self.storage is not None and storage is None:
            raise ValueError(
                "Owner-local coordinate refresh requires renewed logical geometry evidence."
            )
        if self.storage is None and storage is not None:
            raise ValueError("Coordinate refresh cannot change mesh storage placement.")
        if (
            storage is not None
            and self.storage is not None
            and (
                storage.logical_topology_id != self.topology_id
                or storage.partition_index != self.storage.partition_index
                or storage.partition_count != self.storage.partition_count
            )
        ):
            raise ValueError(
                "Coordinate refresh must preserve logical topology and local placement."
            )
        points = jnp.asarray(coordinates)
        if points.shape != self.coordinates.shape:
            raise ValueError(
                "Fixed-topology coordinate refresh must preserve coordinate shape."
            )
        # The blocks, identities, connectivity, and topology are coordinate
        # independent, so the refresh reuses them instead of rebuilding them.
        return CellMesh(
            points,
            self.blocks,
            numeric_version=numeric_version,
            storage=storage,
            _prepared_topology=_PreparedCellMeshTopology(
                self.blocks,
                self.vertex_global_ids,
                self.connectivity,
                self.topology,
                self.topology_id,
                self.topological_dimension,
                self.periodic_topology,
            ),
        )


__all__ = [
    "CellBlock",
    "CellMesh",
    "CellMeshStorage",
    "PolyhedralBlock",
    "SimplicialConnectivity",
]
