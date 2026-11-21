#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation
from ..typing import checked, ConvertibleToArray, Dim, Int32
from ._cell_complex import (
    _canonical_entity_ids,
    _csr_offsets,
    _interval_complex,
    _polygonal_complex,
    _polyhedral_face_loops,
    _polyhedral_loop_connectivity,
    _simplicial_incidence_routes,
    _tetrahedral_complex,
    interval_connectivity,
    IntervalConnectivity,
    polygonal_connectivity,
    PolygonalConnectivity,
    polyhedral_cell_complex,
    polyhedral_connectivity as _build_polyhedral_connectivity,
    PolyhedralConnectivity,
    tetrahedral_connectivity,
    TetrahedralConnectivity,
)
from ._hexahedral import (
    _hexahedral_complex,
    hexahedral_connectivity,
    HexahedralConnectivity,
)
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
    has_repeated_values,
    OrientedIncidence,
    row_positions,
    rows_increase,
    sorted_unique_rows,
)


if TYPE_CHECKING:
    from ._periodic_topology import PeriodicMeshTopology


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
    kind: str, reference: ReferenceCellTopology | None, vertices: ConvertibleToArray, /
) -> tuple[np.ndarray, int]:
    """Admit rank-two connectivity, including an explicit zero-cell local block."""
    cells = np.asarray(vertices, dtype=np.int32)
    if reference is None:
        arity = cells.shape[1] if cells.ndim == 2 else -1
        minimum_arity = 4 if kind == "polyhedron" else 3
    else:
        arity = len(reference.vertices)
        minimum_arity = 2 if reference.dimension == 1 else 3
    if (
        cells.ndim != 2
        or cells.shape[1] != arity
        or arity < minimum_arity
        or (kind == "polygon" and arity < 5)
    ):
        raise ValueError(f"{kind} cell vertices have incompatible arity {arity}.")
    return cells, minimum_arity


def _admit_active_cell_vertices(
    kind: str,
    cells: np.ndarray,
    vertex_valid: ConvertibleToArray | None,
    minimum_arity: int,
    /,
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
    cell_count: int, global_ids: ConvertibleToArray | None, /
) -> np.ndarray:
    """Admit one unique stable identity per cell without changing numeric representation."""
    ids = (
        np.arange(cell_count, dtype=np.int64)
        if global_ids is None
        else np.asarray(global_ids, dtype=np.int64)
    )
    if ids.shape != (cell_count,):
        raise ValueError("Cell global_ids must have shape (cell_count,).")
    if np.any(ids < 0) or has_repeated_values(ids):
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
        vertices: ConvertibleToArray,
        /,
        *,
        vertex_valid: ConvertibleToArray | None = None,
        global_ids: ConvertibleToArray | None = None,
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


@final
class EmptyLocalCellMeshAuthority(StrictModule, NonTrainableState):
    """Partition-bound declaration for an exactly zero-resident cell carrier."""

    source_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    distribution_id: str = eqx.field(static=True)
    group_id: str = eqx.field(static=True)
    local_rank: int = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    block_name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    authority_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_topology_id: str,
        source_geometry_id: str,
        distribution_id: str,
        group_id: str,
        local_rank: int,
        part_count: int,
        block_name: str,
        cell_kind: str,
        topological_dimension: int,
        ambient_dimension: int,
        /,
    ) -> None:
        identifiers = (
            source_topology_id,
            source_geometry_id,
            distribution_id,
            group_id,
            block_name,
            cell_kind,
        )
        reference = _admit_cell_kind(str(cell_kind))
        if (
            any(not isinstance(value, str) or not value.strip() for value in identifiers)
            or reference is None
            or isinstance(local_rank, (bool, np.bool_))
            or not isinstance(local_rank, (int, np.integer))
            or isinstance(part_count, (bool, np.bool_))
            or not isinstance(part_count, (int, np.integer))
            or int(part_count) <= 0
            or not 0 <= int(local_rank) < int(part_count)
            or isinstance(topological_dimension, (bool, np.bool_))
            or not isinstance(topological_dimension, (int, np.integer))
            or int(topological_dimension) != reference.dimension
            or isinstance(ambient_dimension, (bool, np.bool_))
            or not isinstance(ambient_dimension, (int, np.integer))
            or int(ambient_dimension) < reference.dimension
        ):
            raise ValueError("Empty-local cell mesh authority is malformed.")
        rank = int(local_rank)
        parts = int(part_count)
        dimension = int(topological_dimension)
        ambient = int(ambient_dimension)
        self.source_topology_id = source_topology_id
        self.source_geometry_id = source_geometry_id
        self.distribution_id = distribution_id
        self.group_id = group_id
        self.local_rank = rank
        self.part_count = parts
        self.block_name = block_name
        self.cell_kind = cell_kind
        self.topological_dimension = dimension
        self.ambient_dimension = ambient
        self.authority_id = canonical_fingerprint(
            {
                "kind": "empty-local-cell-mesh-authority",
                "source_topology": source_topology_id,
                "source_geometry": source_geometry_id,
                "distribution": distribution_id,
                "group": group_id,
                "local_rank": rank,
                "part_count": parts,
                "block_name": block_name,
                "cell_kind": cell_kind,
                "topological_dimension": dimension,
                "ambient_dimension": ambient,
            }
        )


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
    multiregion_surface: MultiRegionSurfaceTopology | None = None,
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
        multiregion_surface=multiregion_surface,
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
    if np.any(global_ids < 0) or has_repeated_values(global_ids):
        raise ValueError("vertex_global_ids must be unique non-negative integers.")
    cell_global_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in blocks)
    )
    if has_repeated_values(cell_global_ids):
        raise ValueError("Cell global IDs must be unique across mesh blocks.")
    supplied_cells = entity_ids.get(topological_dimension)
    if supplied_cells is not None and not np.array_equal(supplied_cells, cell_global_ids):
        raise ValueError(
            "Top-dimensional entity_global_ids contradict cell block global IDs."
        )
    return global_ids, entity_ids, cell_global_ids


def _simplex_orientation(vertices: np.ndarray, /) -> np.ndarray:
    """Orientation of each ordered simplex relative to its ascending vertex row."""
    inversions = np.zeros(vertices.shape[0], dtype=np.int64)
    for first, second in combinations(range(vertices.shape[1]), 2):
        inversions += vertices[:, first] > vertices[:, second]
    return np.where(inversions % 2, -1.0, 1.0).astype(np.float64)


def _simplex_mesh_levels(
    cells: np.ndarray, dimension: int, vertex_count: int, block_count: int, /
) -> tuple[np.ndarray, ...]:
    top = np.sort(cells, axis=1)
    # Every cell block refused its own duplicate cells; only several blocks can
    # still share a cell.
    if block_count > 1 and _has_duplicate_rows(top):
        raise ValueError("Simplex mesh blocks cannot contain duplicate cells.")
    levels: list[np.ndarray] = [np.arange(vertex_count, dtype=np.int32)[:, None]]
    for degree in range(1, dimension):
        if dimension == 3 and cells.shape[0] and cells.dtype == np.dtype(np.int32):
            indices = np.asarray(
                tuple(combinations(range(dimension + 1), degree + 1)), dtype=np.int32
            )
            faces = np.sort(cells[:, indices].reshape((-1, degree + 1)), axis=1)
            # Both the sorted unique rows and the general set/sorted route use
            # lexicographic integer order and discard equal rows.
            levels.append(sorted_unique_rows(faces))
        else:
            faces = {
                tuple(sorted(int(cell[index]) for index in indices))
                for cell in cells
                for indices in combinations(range(dimension + 1), degree + 1)
            }
            levels.append(np.asarray(sorted(faces), dtype=np.int32))
    levels.append(top)
    return tuple(levels)


def _simplex_mesh_boundary(
    levels: tuple[np.ndarray, ...], relation: EdgeRelation, /
) -> tuple[np.ndarray, ...]:
    dimension = len(levels) - 1
    valid = np.asarray(relation.valid, dtype=np.bool_)
    source = np.asarray(relation.source_indices)[valid]
    counts = np.bincount(source, minlength=levels[-2].shape[0])
    if np.any(counts > 2):
        raise ValueError("Simplex mesh facets may have at most two incident cells.")
    facets = levels[-2][counts == 1]
    masks: list[np.ndarray] = []
    for degree in range(dimension):
        columns = np.asarray(
            tuple(combinations(range(dimension), degree + 1)), dtype=np.int64
        )
        faces = facets[:, columns].reshape((-1, degree + 1))
        rows = row_positions(levels[degree], faces)
        mask = np.zeros((levels[degree].shape[0],), dtype=np.bool_)
        mask[rows[rows >= 0]] = True
        masks.append(mask)
    masks.append(np.zeros(levels[-1].shape[0], dtype=np.bool_))
    return tuple(masks)


_TETRAHEDRON_PACK_WIDTHS = tuple(64 // components for components in (2, 3))


def _pack_tetrahedron_entities(rows: np.ndarray, /) -> np.ndarray:
    """Pack prevalidated nonnegative rows with the first component most significant."""
    values = rows.astype(np.uint64)
    width = np.uint64(_TETRAHEDRON_PACK_WIDTHS[rows.shape[-1] - 2])
    packed = values[..., 0]
    for column in range(1, rows.shape[-1]):
        packed = (packed << width) | values[..., column]
    return packed


def _tetrahedron_route_signs(cells: np.ndarray, /) -> tuple[np.ndarray, ...]:
    """Orientation of every cell's local faces of each degree, combination order."""
    signs: list[np.ndarray] = []
    for degree in range(4):
        local = tuple(combinations(range(4), degree + 1))
        selected = cells[:, np.asarray(local, dtype=np.int32)]
        if degree == 3:
            signs.append(np.ones((cells.shape[0], 1), dtype=np.float64))
        else:
            signs.append(
                _simplex_orientation(selected.reshape((-1, degree + 1))).reshape(
                    selected.shape[:2]
                )
            )
    return tuple(signs)


def _packed_tetrahedron_routes(
    cells: np.ndarray, levels: tuple[np.ndarray, ...], /
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Route validated tetrahedra using lexicographically ordered integer fields."""
    routes: list[np.ndarray] = []
    for degree, entities in enumerate(levels):
        local = tuple(combinations(range(4), degree + 1))
        selected = cells[:, np.asarray(local, dtype=np.int32)]
        if degree == 0:
            route = selected.reshape(cells.shape).copy()
        elif degree == 3:
            route = np.arange(cells.shape[0], dtype=np.int32)[:, None]
        else:
            keys = _pack_tetrahedron_entities(entities)
            queries = _pack_tetrahedron_entities(np.sort(selected, axis=-1))
            route = np.searchsorted(keys, queries).astype(np.int32)
        routes.append(route)
    return tuple(routes), _tetrahedron_route_signs(cells)


class TetrahedronEntityTables(NamedTuple):
    """Precomputed tetrahedral entity tables, checked before use.

    ``edges`` and ``faces`` are the distinct increasing vertex rows of the
    cells' edges and faces in lexicographic order; ``cell_edges`` ``(m, 6)``
    and ``cell_faces`` ``(m, 4)`` route every cell's local edges and faces in
    combination order; ``face_edges`` routes each face's edges opposite its
    vertices in turn; ``incidence_orders`` sort each degree's flat face routes
    by face row (checked where entity IDs follow row order). The native
    ``tet_mesh_entities`` produces them.
    """

    edges: np.ndarray
    faces: np.ndarray
    cell_edges: np.ndarray
    cell_faces: np.ndarray
    face_edges: np.ndarray
    incidence_orders: tuple[np.ndarray, np.ndarray, np.ndarray]


def sorted_tetrahedron_faces(cells: np.ndarray, degree: int, /) -> tuple[np.ndarray, ...]:
    """Columns of the increasing vertex rows of cells' local edges (1) or faces (2).

    Each column is ``(m, 6)`` or ``(m, 4)`` in combination order, equal to
    sorting each row: column minima and maxima, the middle of three being
    their exclusive or.
    """
    columns = tuple(
        cells[:, np.asarray(column, dtype=np.int32)]
        for column in zip(*combinations(range(4), degree + 1), strict=True)
    )
    if degree == 1:
        first, second = columns
        return np.minimum(first, second), np.maximum(first, second)
    if degree != 2:
        raise ValueError("Sorted tetrahedron faces have degree one or two.")
    first, second, third = columns
    low = np.minimum(np.minimum(first, second), third)
    high = np.maximum(np.maximum(first, second), third)
    return low, first ^ second ^ third ^ low ^ high, high


def _routed_rows(
    entities: np.ndarray, routes: np.ndarray, expected: tuple[np.ndarray, ...], /
) -> bool:
    """Whether in-range ``routes`` select exactly the ``expected`` column values."""
    flat = routes.reshape((-1,))
    if flat.size and (flat.min() < 0 or flat.max() >= entities.shape[0]):
        return False
    return all(
        np.array_equal(
            np.ascontiguousarray(entities[:, column])[flat], values.reshape((-1,))
        )
        for column, values in enumerate(expected)
    )


def _supplied_tetrahedron_entities(
    cells: np.ndarray,
    vertex_count: int,
    block_count: int,
    tables: TetrahedronEntityTables,
    /,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray]:
    """Levels and cell routes from supplied tables, equal to the generic ones.

    Strictly increasing entity rows that every cell route hits, each route
    selecting exactly its cell's sorted face, are the sorted distinct faces and
    their searched rows; the face-edge routes are checked the same way.
    """
    top = np.sort(cells, axis=1)
    if block_count > 1 and _has_duplicate_rows(top):
        raise ValueError("Simplex mesh blocks cannot contain duplicate cells.")
    edges, faces, cell_edges, cell_faces, face_edges = (
        np.asarray(value) for value in tables[:5]
    )
    if (
        not all(
            value.dtype == np.dtype(np.int32)
            and value.ndim == 2
            and value.shape[1] == width
            for value, width in (
                (edges, 2),
                (faces, 3),
                (cell_edges, 6),
                (cell_faces, 4),
                (face_edges, 3),
            )
        )
        or cell_edges.shape[0] != cells.shape[0]
        or cell_faces.shape[0] != cells.shape[0]
    ):
        raise ValueError("Tetrahedron entity tables have inconsistent shapes.")
    sorted_edges = sorted_tetrahedron_faces(cells, 1)
    sorted_faces = sorted_tetrahedron_faces(cells, 2)
    for entities, routes, expected in (
        (edges, cell_edges, sorted_edges),
        (faces, cell_faces, sorted_faces),
    ):
        if not rows_increase(entities, strict=True) or not _routed_rows(
            entities, routes, expected
        ):
            raise ValueError("Tetrahedron entity tables differ from the cells' faces.")
        covered = np.zeros((entities.shape[0],), dtype=np.bool_)
        covered[routes.reshape((-1,))] = True
        if not np.all(covered):
            raise ValueError("Tetrahedron entity tables list faces of no cell.")
    if face_edges.shape[0] != faces.shape[0] or not _routed_rows(
        edges,
        face_edges,
        tuple(
            faces[:, np.asarray(column, dtype=np.int32)]
            for column in ((1, 0, 0), (2, 2, 1))
        ),
    ):
        raise ValueError("Tetrahedron face-edge routes differ from the faces' edges.")
    levels = (np.arange(vertex_count, dtype=np.int32)[:, None], edges, faces, top)
    routes = (
        cells.copy(),
        cell_edges,
        cell_faces,
        np.arange(cells.shape[0], dtype=np.int32)[:, None],
    )
    return levels, routes, face_edges


def _simplex_cell_routes(
    cells: np.ndarray, levels: tuple[np.ndarray, ...], /
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    dimension = len(levels) - 1
    # Levels are generated immediately above this route: vertices enumerate
    # rows, intermediate levels are sorted unique, and top rows follow cells.
    # Two- and three-component fields use 32 and 21 bits respectively.
    # Validate every operand before conversion so negatives and sentinels cannot wrap.
    if (
        dimension == 3
        and cells.dtype == np.dtype(np.int32)
        and cells.ndim == 2
        and cells.shape[1] == 4
        and all(
            level.dtype == np.dtype(np.int32)
            and level.ndim == 2
            and level.shape[1] == degree + 1
            for degree, level in enumerate(levels)
        )
        and all(
            np.min(values, initial=0) >= 0
            and np.max(values, initial=0) <= (1 << _TETRAHEDRON_PACK_WIDTHS[1]) - 1
            for values in (cells, *levels)
        )
    ):
        return _packed_tetrahedron_routes(cells, levels)
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
    *,
    tetrahedron_entities: TetrahedronEntityTables | None = None,
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
    face_routes: dict[int, np.ndarray] = {}
    if tetrahedron_entities is None:
        levels = _simplex_mesh_levels(cells, dimension, vertex_count, len(blocks))
        routes, route_signs = _simplex_cell_routes(cells, levels)
    elif dimension == 3:
        levels, routes, face_routes[2] = _supplied_tetrahedron_entities(
            cells, vertex_count, len(blocks), tetrahedron_entities
        )
        route_signs = _tetrahedron_route_signs(cells)
    else:
        raise ValueError("Tetrahedron entity tables require a three-dimensional mesh.")
    # The face opposite sorted vertex r of a cell omits its local vertex
    # order[r]; in combination order that face is number dimension - order[r].
    order = np.argsort(cells, axis=1, kind="stable")
    face_routes[dimension] = np.take_along_axis(
        routes[dimension - 1], dimension - order, axis=1
    )
    # The levels are increasing, distinct and face closed by construction, so
    # the topology is assembled once from their incidence routes.
    relations = tuple(
        (
            EdgeRelation(
                source,
                target,
                source_size=levels[degree - 1].shape[0],
                target_size=levels[degree].shape[0],
            ),
            signs,
        )
        for degree, (source, target, signs) in enumerate(
            _simplicial_incidence_routes(levels, face_routes=face_routes), start=1
        )
    )
    masks = _simplex_mesh_boundary(levels, relations[-1][0])
    sets: list[EntitySet] = []
    for degree, rows in enumerate(levels):
        if degree == 0:
            ids = vertex_ids
        elif degree == dimension:
            ids = cell_ids
        else:
            supplied = entity_ids.get(degree)
            ids = (
                _canonical_entity_ids(vertex_ids[rows]) if supplied is None else supplied
            )
        if ids.shape != (rows.shape[0],):
            raise ValueError("Simplex entity global IDs must match entity counts.")
        sets.append(
            EntitySet(
                f"simplicial_{degree}_cells",
                degree,
                ids,
                subsets=(EntitySubset("boundary", masks[degree]),),
            )
        )
    orientation = _simplex_orientation(cells)
    # Supplied incidence orders sort by face row, then simplex row; that is
    # the canonical ID order where both entity ID banks increase with rows.
    increasing = tuple(
        bool(np.all(ids[1:] > ids[:-1]))
        for ids in (np.asarray(entity_set.entity_ids) for entity_set in sets)
    )
    incidences: list[OrientedIncidence] = []
    for degree, (relation, signs) in enumerate(relations, start=1):
        coefficients = signs
        if degree == dimension:
            coefficients = coefficients * orientation[np.asarray(relation.target_indices)]
        incidences.append(
            OrientedIncidence(
                degree,
                sets[degree - 1],
                sets[degree],
                relation,
                coefficients,
                canonical_order=(
                    tetrahedron_entities.incidence_orders[degree - 1]
                    if tetrahedron_entities is not None
                    and increasing[degree - 1]
                    and increasing[degree]
                    else None
                ),
            )
        )
    # Checked tables are exactly the cells' faces, and alternating signs on
    # the faces opposite each vertex make every boundary of a boundary vanish.
    topology = CellComplexTopology(
        sets, incidences, validate=tetrahedron_entities is None
    )
    return SimplicialConnectivity(
        topology,
        cells,
        entities=levels,
        cell_entities=routes,
        cell_entity_signs=route_signs,
        boundary_masks=masks,
    ), topology


def _named_mesh_route_blocks(
    blocks: tuple[CellBlock | PolyhedralBlock, ...], /
) -> tuple[CellBlock | PolyhedralBlock, ...]:
    """Prepare native incidence routes without relabeling stored mesh blocks."""
    result: list[CellBlock | PolyhedralBlock] = []
    for block in blocks:
        if not block.cell_kind.startswith(("simplex:", "tensor:")):
            result.append(block)
            continue
        family = reference_cell_topology(block.cell_kind).family
        if family == block.cell_kind:
            if block.cell_kind.startswith("tensor:"):
                raise ValueError(
                    "Unstructured tensor CellMesh supports dimensions one through three."
                )
            raise ValueError(
                "Mixed reference-cell routing requires a native cell family."
            )
        result.append(
            CellBlock(block.name, family, block.vertices, global_ids=block.global_ids)
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
    multiregion_surface: MultiRegionSurfaceTopology | None = None,
    tetrahedron_entities: TetrahedronEntityTables | None = None,
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
    if multiregion_surface is not None and (
        topological_dimension != 2
        or any(block.cell_kind != "triangle" for block in normalized_blocks)
    ):
        raise ValueError(
            "A multiregion surface topology binds only two-dimensional triangle meshes."
        )
    general_simplex = any(
        block.cell_kind.startswith("simplex:") for block in normalized_blocks
    ) and all(
        block.cell_kind in ("interval", "triangle", "tetrahedron")
        or block.cell_kind.startswith("simplex:")
        for block in normalized_blocks
    )
    route_blocks = (
        normalized_blocks
        if general_simplex
        else _named_mesh_route_blocks(normalized_blocks)
    )
    if tetrahedron_entities is not None and not general_simplex:
        raise ValueError("Tetrahedron entity tables bind only simplex:3 meshes.")
    if general_simplex:
        if polyhedral_connectivity is not None:
            raise ValueError("Simplicial meshes cannot use polyhedral connectivity.")
        connectivity, topology = _simplex_mesh_topology(
            normalized_blocks,
            coordinate_count,
            global_ids,
            entity_ids,
            cell_global_ids,
            tetrahedron_entities=tetrahedron_entities,
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
            multiregion_surface,
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
    return _PreparedCellMeshTopology(
        normalized_blocks,
        global_ids,
        connectivity,
        topology,
        topology_id,
        topological_dimension,
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


def _rebuild_polyhedral_connectivity_from_stored(
    value: PolyhedralConnectivity,
    blocks: tuple[CellBlock | PolyhedralBlock, ...],
    vertex_global_ids: ArrayLike,
    /,
) -> PolyhedralConnectivity:
    if type(value) is not PolyhedralConnectivity:
        raise TypeError("Polyhedral mesh lost its canonical connectivity bank.")
    cell_ids = np.asarray(value.cell_global_ids, dtype=np.int64)
    cell_rows = {int(identifier): row for row, identifier in enumerate(cell_ids)}
    if len(cell_rows) != cell_ids.size:
        raise ValueError("Polyhedral connectivity cell IDs must be unique.")
    face_offsets = np.asarray(value.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(value.face_vertex_values, dtype=np.int64)
    cell_face_offsets = np.asarray(value.cell_face_offsets, dtype=np.int64)
    cell_face_values = np.asarray(value.cell_face_values, dtype=np.int64)
    cell_face_signs = np.asarray(value.cell_face_sign_values)
    if (
        face_offsets.shape != (int(value.face_count) + 1,)
        or face_offsets.size == 0
        or face_offsets[0] != 0
        or face_offsets[-1] != face_vertices.size
        or cell_face_offsets.shape != (cell_ids.size + 1,)
        or cell_face_offsets.size == 0
        or cell_face_offsets[0] != 0
        or cell_face_offsets[-1] != cell_face_values.size
        or cell_face_signs.shape != cell_face_values.shape
    ):
        raise ValueError("Polyhedral connectivity row offsets are malformed.")
    entries: list[tuple[str, np.ndarray] | tuple[np.ndarray, ...]] = []
    rebuilt_cell_ids: list[int] = []
    for block in blocks:
        if type(block) is CellBlock:
            for row, identifier in enumerate(
                np.asarray(block.global_ids, dtype=np.int64)
            ):
                cell_id = int(identifier)
                if cell_id not in cell_rows:
                    raise ValueError("Polyhedral connectivity lost a standard cell row.")
                # Dimension-qualified kinds (simplex:3) share their family's routes.
                entries.append(
                    (
                        reference_cell_topology(block.cell_kind).family,
                        np.asarray(block.vertices[row : row + 1], dtype=np.int32),
                    )
                )
                rebuilt_cell_ids.append(cell_id)
        elif type(block) is PolyhedralBlock:
            for identifier in np.asarray(block.global_ids, dtype=np.int64):
                cell_id = int(identifier)
                try:
                    cell_row = cell_rows[cell_id]
                except KeyError as error:
                    raise ValueError(
                        "Polyhedral connectivity lost a polyhedron row."
                    ) from error
                loops: list[np.ndarray] = []
                for occurrence in range(
                    int(cell_face_offsets[cell_row]),
                    int(cell_face_offsets[cell_row + 1]),
                ):
                    face_row = int(cell_face_values[occurrence])
                    if not 0 <= face_row < face_offsets.size - 1:
                        raise ValueError("Polyhedral cell references an absent face row.")
                    loop = face_vertices[
                        int(face_offsets[face_row]) : int(face_offsets[face_row + 1])
                    ].astype(np.int32, copy=True)
                    orientation = cell_face_signs[occurrence]
                    if orientation == -1:
                        loop = loop[::-1]
                    elif orientation != 1:
                        raise ValueError(
                            "Polyhedral face-use orientation is not signed unit data."
                        )
                    loops.append(loop)
                if not loops:
                    raise ValueError(
                        "Polyhedral cell must retain its exact face-loop bank."
                    )
                entries.append(tuple(loops))
                rebuilt_cell_ids.append(cell_id)
        else:
            raise TypeError("CellMesh contains an unsupported block type.")
    if not np.array_equal(np.asarray(rebuilt_cell_ids, dtype=np.int64), cell_ids):
        raise ValueError(
            "Polyhedral cell blocks do not match the stored connectivity row order."
        )
    rebuilt = _build_polyhedral_connectivity(
        tuple(entries),
        int(value.vertex_count),
        vertex_global_ids=vertex_global_ids,
        edge_global_ids=value.edge_global_ids,
        face_global_ids=value.face_global_ids,
        cell_global_ids=np.asarray(rebuilt_cell_ids, dtype=np.int64),
    )
    arrays = (
        "edges",
        "face_vertex_offsets",
        "face_vertex_values",
        "face_edge_offsets",
        "face_edge_values",
        "face_edge_sign_values",
        "cell_face_offsets",
        "cell_face_values",
        "cell_face_sign_values",
        "cell_vertex_offsets",
        "cell_vertex_values",
        "face_owner",
        "face_neighbor",
        "face_owner_local",
        "face_neighbor_local",
        "face_cell_counts",
        "boundary_vertices",
        "boundary_edges",
        "boundary_faces",
        "vertex_global_ids",
        "edge_global_ids",
        "face_global_ids",
        "cell_global_ids",
    )
    if any(
        not np.array_equal(
            np.asarray(getattr(rebuilt, name)), np.asarray(getattr(value, name))
        )
        for name in arrays
    ) or (
        rebuilt.vertex_count,
        rebuilt.edge_count,
        rebuilt.face_count,
        rebuilt.cell_count,
        rebuilt.maximum_face_arity,
        rebuilt.maximum_cell_faces,
        rebuilt.maximum_cell_vertices,
    ) != (
        value.vertex_count,
        value.edge_count,
        value.face_count,
        value.cell_count,
        value.maximum_face_arity,
        value.maximum_cell_faces,
        value.maximum_cell_vertices,
    ):
        raise ValueError(
            "Stored polyhedral connectivity differs from its exact face loops."
        )
    return rebuilt


def _packed_ranges(starts: np.ndarray, counts: np.ndarray, /) -> np.ndarray:
    """The ranges ``starts[i] : starts[i] + counts[i]``, concatenated."""
    counts = counts.astype(np.int64)
    ends = np.cumsum(counts)
    return np.arange(int(ends[-1]) if ends.size else 0, dtype=np.int64) + np.repeat(
        starts.astype(np.int64) - (ends - counts), counts
    )


class CellMesh(StrictModule, NonTrainableState):
    """Canonical computational mesh shared by unstructured discretizations."""

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
    empty_local_authority: EmptyLocalCellMeshAuthority | None
    lifted_topology_id: str = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ConvertibleToArray,
        blocks: Sequence[CellBlock | PolyhedralBlock],
        /,
        *,
        vertex_global_ids: ArrayLike | None = None,
        entity_global_ids: Mapping[int, ArrayLike] | None = None,
        polyhedral_connectivity: PolyhedralConnectivity | None = None,
        numeric_version: str = "0",
        periodic_topology: PeriodicMeshTopology | None = None,
        empty_local_authority: EmptyLocalCellMeshAuthority | None = None,
        multiregion_surface: MultiRegionSurfaceTopology | None = None,
        tetrahedron_entities: TetrahedronEntityTables | None = None,
        _prepared_topology: _PreparedCellMeshTopology | None = None,
    ) -> None:
        points = np.asarray(coordinates, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] == 0:
            raise ValueError("Cell mesh coordinates must have shape (n >= 0, d > 0).")
        if not np.all(np.isfinite(points)):
            raise ValueError("Cell mesh coordinates must be finite.")
        normalized_blocks = tuple(blocks)
        if not all(
            isinstance(block, (CellBlock, PolyhedralBlock)) for block in normalized_blocks
        ):
            raise TypeError(
                "blocks must contain only CellBlock or PolyhedralBlock instances."
            )
        cell_count = sum(block.cell_count for block in normalized_blocks)
        if cell_count == 0:
            if not isinstance(empty_local_authority, EmptyLocalCellMeshAuthority):
                raise ValueError(
                    "Zero-resident CellMesh requires explicit partition authority."
                )
            if (
                points.shape[0] != 0
                or len(normalized_blocks) != 1
                or type(normalized_blocks[0]) is not CellBlock
                or normalized_blocks[0].name != empty_local_authority.block_name
                or normalized_blocks[0].cell_kind != empty_local_authority.cell_kind
                or points.shape[1] != empty_local_authority.ambient_dimension
                or normalized_blocks[0].topological_dimension
                != empty_local_authority.topological_dimension
                or periodic_topology is not None
                or _prepared_topology is not None
            ):
                raise ValueError(
                    "Zero-resident CellMesh differs from its typed partition authority."
                )
        elif empty_local_authority is not None:
            raise ValueError(
                "Empty-local partition authority cannot bind resident cell rows."
            )
        if _prepared_topology is None:
            prepared = _prepare_cell_mesh_topology(
                points.shape,
                normalized_blocks,
                vertex_global_ids=vertex_global_ids,
                entity_global_ids=entity_global_ids,
                polyhedral_connectivity=polyhedral_connectivity,
                multiregion_surface=multiregion_surface,
                tetrahedron_entities=tetrahedron_entities,
            )
        else:
            prepared = _reused_cell_mesh_topology(
                _prepared_topology,
                points.shape,
                normalized_blocks,
                identities_supplied=(
                    vertex_global_ids is not None
                    or entity_global_ids is not None
                    or polyhedral_connectivity is not None
                    or multiregion_surface is not None
                    or tetrahedron_entities is not None
                ),
            )
        topology_identity = prepared.topology_id
        if periodic_topology is not None:
            from ._periodic_topology import PeriodicMeshTopology

            if not isinstance(periodic_topology, PeriodicMeshTopology):
                raise TypeError("periodic_topology must be PeriodicMeshTopology or None.")
            periodic_topology.require_lifted(
                prepared.topology, prepared.topology_id, prepared.blocks
            )
            periodic_topology.require_coordinates(points)
            topology_identity = canonical_fingerprint(
                {
                    "kind": "periodic-cell-mesh-topology",
                    "lifted_topology": prepared.topology_id,
                    "periodic_topology": periodic_topology.periodic_topology_id,
                }
            )
        if empty_local_authority is not None:
            topology_identity = canonical_fingerprint(
                {
                    "kind": "empty-local-cell-mesh-topology",
                    "local_topology": prepared.topology_id,
                    "authority": empty_local_authority.authority_id,
                }
            )
        geometry_layout_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-geometry-layout",
                "topology": topology_identity,
                "ambient_dimension": points.shape[1],
                "coordinate_count": points.shape[0],
                "coordinate_dtype": str(points.dtype),
                "empty_local_authority": (
                    None
                    if empty_local_authority is None
                    else empty_local_authority.authority_id
                ),
            }
        )
        geometry_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-geometry",
                "layout": geometry_layout_id,
                "coordinates": array_tree_fingerprint(points),
                "numeric_version": str(numeric_version),
            }
        )
        support = DiscreteSupport(prepared.topology, points.shape[1], geometry_layout_id)
        self.coordinates = jnp.asarray(points)
        self.blocks = prepared.blocks
        self.vertex_global_ids = jnp.asarray(prepared.vertex_global_ids)
        self.topology = prepared.topology
        self.connectivity = prepared.connectivity
        self.support = support
        self.periodic_topology = periodic_topology
        self.empty_local_authority = empty_local_authority
        self.lifted_topology_id = prepared.topology_id
        self.topological_dimension = prepared.topological_dimension
        self.ambient_dimension = points.shape[1]
        self.topology_id = topology_identity
        self.geometry_layout_id = geometry_layout_id
        self.geometry_id = geometry_id
        self.mesh_id = canonical_fingerprint(
            {
                "kind": "cell-mesh",
                "topology": topology_identity,
                "geometry": geometry_id,
            }
        )
        self.numeric_version = str(numeric_version)

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[CellMesh],
        value: CellMesh,
        /,
        *,
        actual_geometry: object | None = None,
        multiregion_surface: MultiRegionSurfaceTopology | None = None,
    ) -> CellMesh:
        if type(value) is not cls:
            raise TypeError("value must be CellMesh.")
        if not isinstance(value.topology, CellComplexTopology):
            raise TypeError("Stored CellMesh must retain CellComplexTopology.")
        blocks: list[CellBlock | PolyhedralBlock] = []
        for block in value.blocks:
            if type(block) is CellBlock:
                rebuilt_block = CellBlock(
                    block.name,
                    block.cell_kind,
                    block.vertices,
                    vertex_valid=block.vertex_valid,
                    global_ids=block.global_ids,
                )
            elif type(block) is PolyhedralBlock:
                rebuilt_block = PolyhedralBlock(
                    block.name,
                    block.vertices,
                    global_ids=block.global_ids,
                )
            else:
                raise TypeError("CellMesh contains an unsupported block type.")
            if (
                rebuilt_block.block_id != block.block_id
                or array_tree_fingerprint(rebuilt_block.vertices)
                != array_tree_fingerprint(block.vertices)
                or array_tree_fingerprint(rebuilt_block.vertex_valid)
                != array_tree_fingerprint(block.vertex_valid)
                or array_tree_fingerprint(rebuilt_block.global_ids)
                != array_tree_fingerprint(block.global_ids)
            ):
                raise ValueError("Stored CellMesh block differs from its exact rows.")
            blocks.append(rebuilt_block)
        rebuilt_blocks = tuple(blocks)
        entity_global_ids = {
            entity.intrinsic_dimension: np.asarray(entity.entity_ids, dtype=np.int64)
            for entity in value.topology.entity_sets
            if entity.intrinsic_dimension > 0
        }
        polyhedral = (
            _rebuild_polyhedral_connectivity_from_stored(
                value.connectivity,
                rebuilt_blocks,
                value.vertex_global_ids,
            )
            if type(value.connectivity) is PolyhedralConnectivity
            else None
        )
        if any(block.cell_kind == "polyhedron" for block in rebuilt_blocks) and (
            polyhedral is None
        ):
            raise TypeError("Polyhedral CellMesh lost its exact connectivity owner.")
        if value.periodic_topology is None:
            if (
                multiregion_surface is not None
                and value.empty_local_authority is not None
            ):
                raise ValueError(
                    "Empty-local meshes cannot bind multiregion surface topology."
                )
            if actual_geometry is not None:
                raise ValueError(
                    "A nonperiodic CellMesh cannot bind periodic actual_geometry."
                )
            rebuilt = cls(
                value.coordinates,
                rebuilt_blocks,
                vertex_global_ids=value.vertex_global_ids,
                entity_global_ids=entity_global_ids,
                numeric_version=value.numeric_version,
                empty_local_authority=value.empty_local_authority,
                polyhedral_connectivity=polyhedral,
                multiregion_surface=multiregion_surface,
            )
        else:
            if multiregion_surface is not None:
                raise ValueError(
                    "Periodic mesh restoration cannot bind a multiregion surface premise."
                )
            from ._periodic_topology import PeriodicMeshTopology

            if type(value.periodic_topology) is not PeriodicMeshTopology:
                raise TypeError("CellMesh periodic topology has an unexpected type.")
            if actual_geometry is not None:
                from ._cell_geometry import CellGeometrySpec

                if type(actual_geometry) is not CellGeometrySpec:
                    raise TypeError("actual_geometry must be a CellGeometrySpec.")
            lifted = cls(
                value.coordinates,
                rebuilt_blocks,
                vertex_global_ids=value.vertex_global_ids,
                entity_global_ids=entity_global_ids,
                numeric_version=value.numeric_version,
                empty_local_authority=value.empty_local_authority,
                polyhedral_connectivity=polyhedral,
            )
            periodic = PeriodicMeshTopology.from_stored_fields(
                value.periodic_topology,
                lifted,
                actual_geometry=actual_geometry,
                actual_mesh=value if actual_geometry is not None else None,
            )
            rebuilt = cls(
                value.coordinates,
                rebuilt_blocks,
                vertex_global_ids=value.vertex_global_ids,
                entity_global_ids=entity_global_ids,
                numeric_version=value.numeric_version,
                periodic_topology=periodic,
                empty_local_authority=value.empty_local_authority,
                polyhedral_connectivity=polyhedral,
            )
        if (
            rebuilt.mesh_id != value.mesh_id
            or rebuilt.topology_id != value.topology_id
            or rebuilt.lifted_topology_id != value.lifted_topology_id
            or rebuilt.geometry_layout_id != value.geometry_layout_id
            or rebuilt.geometry_id != value.geometry_id
            or rebuilt.topological_dimension != value.topological_dimension
            or rebuilt.ambient_dimension != value.ambient_dimension
            or rebuilt.numeric_version != value.numeric_version
            or type(rebuilt.connectivity) is not type(value.connectivity)
            or array_tree_fingerprint(rebuilt.coordinates)
            != array_tree_fingerprint(value.coordinates)
            or (
                None
                if rebuilt.empty_local_authority is None
                else rebuilt.empty_local_authority.authority_id
            )
            != (
                None
                if value.empty_local_authority is None
                else value.empty_local_authority.authority_id
            )
            or array_tree_fingerprint(rebuilt.vertex_global_ids)
            != array_tree_fingerprint(value.vertex_global_ids)
            or array_tree_fingerprint(rebuilt.connectivity)
            != array_tree_fingerprint(value.connectivity)
            or array_tree_fingerprint(rebuilt.topology)
            != array_tree_fingerprint(value.topology)
            or array_tree_fingerprint(rebuilt.support)
            != array_tree_fingerprint(value.support)
            or rebuilt.support.support_id != value.support.support_id
            or (
                None
                if rebuilt.periodic_topology is None
                else rebuilt.periodic_topology.periodic_topology_id
            )
            != (
                None
                if value.periodic_topology is None
                else value.periodic_topology.periodic_topology_id
            )
        ):
            raise ValueError(
                "Stored CellMesh differs from its canonical topology and geometry."
            )
        return rebuilt

    @classmethod
    @checked
    def empty_local(
        cls: type[CellMesh],
        authority: EmptyLocalCellMeshAuthority,
        /,
        *,
        numeric_version: str = "0",
    ) -> CellMesh:
        """Construct an explicitly authorized zero-resident local cell carrier."""
        if type(authority) is not EmptyLocalCellMeshAuthority:
            raise TypeError("authority must be EmptyLocalCellMeshAuthority.")
        reference = reference_cell_topology(authority.cell_kind)
        empty_ids = np.empty((0,), dtype=np.int64)
        return cls(
            np.empty((0, authority.ambient_dimension), dtype=np.float64),
            (
                CellBlock(
                    authority.block_name,
                    authority.cell_kind,
                    np.empty((0, len(reference.vertices)), dtype=np.int32),
                    global_ids=empty_ids,
                ),
            ),
            numeric_version=numeric_version,
            empty_local_authority=authority,
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
        tetrahedron_entities: TetrahedronEntityTables | None = None,
    ) -> CellMesh:
        """Build simplicial mesh geometry with an explicitly declared intrinsic dimension.

        ``tetrahedron_entities`` may supply a ``simplex:3`` mesh's entity
        tables (checked against the cells, so the topology is the same).
        """
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
            tetrahedron_entities=tetrahedron_entities,
        )

    @classmethod
    def from_triangles(
        cls,
        coordinates: ConvertibleToArray,
        triangles: ConvertibleToArray,
        /,
        *,
        block_name: str = "triangles",
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ConvertibleToArray | None = None,
        numeric_version: str = "0",
        multiregion_surface: MultiRegionSurfaceTopology | None = None,
    ) -> CellMesh:
        """Build a triangle mesh; a labeled multiregion surface admits junction edges."""
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
            multiregion_surface=multiregion_surface,
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
        cells: Sequence[Sequence[ConvertibleToArray]],
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
        return cls._from_packed_polyhedra(
            points,
            *_polyhedral_face_loops(normalized_cells, points.shape[0]),
            block_name=block_name,
            vertex_global_ids=vertex_global_ids,
            cell_global_ids=cell_global_ids,
            numeric_version=numeric_version,
        )

    @classmethod
    def _from_packed_polyhedra(
        cls,
        points: np.ndarray,
        cell_face_counts: np.ndarray,
        loop_lengths: np.ndarray,
        explicit_loops: np.ndarray,
        loop_values: np.ndarray,
        /,
        *,
        block_name: str,
        vertex_global_ids: ArrayLike | None,
        cell_global_ids: ArrayLike | None,
        numeric_version: str,
    ) -> CellMesh:
        """``from_polyhedra`` over packed face loops: cell ``c`` owns the next
        ``cell_face_counts[c]`` loops, loop ``i`` the next ``loop_lengths[i]``
        entries of ``loop_values`` (``-1``: a malformed loop without entries),
        ``explicit_loops`` as ``_polyhedral_face_loops`` returns them."""

        cell_count = cell_face_counts.size
        source_ids = (
            np.arange(cell_count, dtype=np.int64)
            if cell_global_ids is None
            else np.asarray(cell_global_ids, dtype=np.int64)
        )
        if source_ids.shape != (cell_count,):
            raise ValueError("cell_global_ids must match the polyhedral cell count.")
        sizes = np.maximum(loop_lengths, 0)
        loop_cells = np.repeat(np.arange(cell_count, dtype=np.int64), cell_face_counts)
        # Distinct (cell, vertex) keys; malformed vertex rows are refused by
        # the connectivity below.
        span = max(points.shape[0], int(np.max(loop_values, initial=0)) + 1)
        keys = np.unique(np.repeat(loop_cells, sizes) * span + loop_values)
        vertex_counts = np.bincount(keys // span, minlength=cell_count)
        # Cells in stable order of their distinct vertex counts, their loops
        # and loop entries gathered along.
        order = np.argsort(vertex_counts, kind="stable")
        loop_rows = _packed_ranges(
            _csr_offsets(cell_face_counts)[:-1][order], cell_face_counts[order]
        )
        entry_rows = _packed_ranges(_csr_offsets(sizes)[:-1][loop_rows], sizes[loop_rows])
        ordered_ids = source_ids[order]
        ordered_counts = vertex_counts[order]
        connectivity = _polyhedral_loop_connectivity(
            cell_face_counts[order],
            loop_lengths[loop_rows],
            explicit_loops[loop_rows],
            loop_values[entry_rows],
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
    ) -> CellMesh:
        points = jnp.asarray(coordinates)
        if points.shape != self.coordinates.shape:
            raise ValueError(
                "Fixed-topology coordinate refresh must preserve coordinate shape."
            )
        if self.empty_local_authority is not None:
            return CellMesh(
                points,
                self.blocks,
                numeric_version=numeric_version,
                empty_local_authority=self.empty_local_authority,
            )
        # The blocks, identities, connectivity, and topology are coordinate
        # independent, so the refresh reuses them instead of rebuilding them.
        return CellMesh(
            points,
            self.blocks,
            numeric_version=numeric_version,
            periodic_topology=self.periodic_topology,
            _prepared_topology=_PreparedCellMeshTopology(
                self.blocks,
                self.vertex_global_ids,
                self.connectivity,
                self.topology,
                self.lifted_topology_id,
                self.topological_dimension,
            ),
        )


from ..geometry.multiregion_surface._topology import MultiRegionSurfaceTopology


__all__ = [
    "CellBlock",
    "CellMesh",
    "EmptyLocalCellMeshAuthority",
    "PolyhedralBlock",
    "SimplicialConnectivity",
]
