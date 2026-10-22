#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from itertools import combinations
from math import comb
from typing import assert_never, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from phydrax.discretization._topology_rows import has_duplicate_rows

from .._array_archive import array_collection_digest
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._jax_config import x64_enabled
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation
from ..typing import checked, ConvertibleToArray, Dim, Int32
from . import (
    _cell_geometry as cell_geometry_owner,
    _cell_geometry_validity as cell_geometry_validity_owner,
    _periodic_topology as periodic_topology_owner,
)
from ._cell_complex import (
    _OPEN_SHELL_CHUNK_CELLS,
    canonical_entity_ids,
    interval_complex,
    interval_connectivity,
    IntervalConnectivity,
    polygonal_complex,
    polygonal_connectivity,
    PolygonalConnectivity,
    polyhedral_cell_complex,
    polyhedral_connectivity as _build_polyhedral_connectivity,
    PolyhedralConnectivity,
    simplicial_cell_complex,
    tetrahedral_complex,
    tetrahedral_connectivity,
    TetrahedralConnectivity,
)
from ._coordinate_budget import CoordinateEnclosureBudget
from ._hexahedral import (
    hexahedral_complex,
    hexahedral_connectivity,
    HexahedralConnectivity,
)
from ._reference_cell import (
    reference_cell_topology,
)
from ._reference_cell_types import CellBlock, PolyhedralBlock
from ._support import DiscreteSupport
from ._topology import CellComplexTopology, EntitySet, EntitySubset, OrientedIncidence


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
                f"SimplicialConnectivity.cells must have one reference-vertex row per top cell; got {cell_rows.shape}."
            )
        if any(
            len(values) != dimension + 1 for values in (vertex_rows, routes, signs, masks)
        ):
            raise ValueError(
                f"SimplicialConnectivity.entities must contain every entity degree in vertices, routes, signs, and masks; got {tuple(len(values) for values in (vertex_rows, routes, signs, masks))}."
            )
        for degree, entity_set in enumerate(topology.entity_sets):
            expected = len(tuple(combinations(range(dimension + 1), degree + 1)))
            if vertex_rows[degree].shape != (entity_set.count, degree + 1):
                raise ValueError(
                    f"SimplicialConnectivity.entities must match the entity count and vertex arity at degree {degree}; got {vertex_rows[degree].shape}."
                )
            if routes[degree].shape != (cell_rows.shape[0], expected):
                raise ValueError(
                    f"SimplicialConnectivity.cell_entities must match cell count and reference entity count at degree {degree}; got {routes[degree].shape}."
                )
            if signs[degree].shape != routes[degree].shape or np.any(
                np.abs(signs[degree]) != 1.0
            ):
                raise ValueError(
                    f"SimplicialConnectivity.cell_entity_signs must match routes with unit orientation coefficients; got {signs[degree].shape}."
                )
            if masks[degree].shape != (entity_set.count,):
                raise ValueError(
                    f"SimplicialConnectivity.boundary_masks must match entity counts at degree {degree}; got {masks[degree].shape}."
                )
            if np.any(routes[degree] < 0) or np.any(routes[degree] >= entity_set.count):
                raise ValueError(
                    f"SimplicialConnectivity.cell_entities must index declared entities at degree {degree}; got {routes[degree].shape}."
                )
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
            raise ValueError(
                f"SimplicialConnectivity.dimension must be at least two to expose degree-two faces; got {self.dimension}."
            )
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
    periodic_topology: periodic_topology_owner.PeriodicMeshTopology | None


def _interval_mesh_topology(
    blocks: tuple[CellBlock | PolyhedralBlock, ...],
    coordinate_count: int,
    vertex_ids: np.ndarray,
    cell_ids: np.ndarray,
    /,
) -> tuple[IntervalConnectivity, CellComplexTopology]:
    if any(block.cell_kind != "interval" for block in blocks):
        raise ValueError(
            f"CellMesh.blocks must use interval blocks in one dimension; got {tuple(block.cell_kind for block in blocks)}."
        )
    intervals = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int32) for block in blocks),
        axis=0,
    )
    connectivity = interval_connectivity(intervals, coordinate_count)
    return connectivity, interval_complex(
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
        raise ValueError(
            f"CellMesh.blocks must use polygonal blocks in two dimensions; got {tuple(block.cell_kind for block in blocks)}."
        )
    arities = tuple(block.arity for block in blocks)
    if arities != tuple(sorted(arities)):
        raise ValueError(
            f"CellMesh.blocks must be ordered by increasing arity; got {arities}."
        )
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
    return connectivity, polygonal_complex(
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
        raise ValueError(
            f"CellMesh.polyhedral_connectivity must match mesh vertex, entity, and cell IDs; got {(connectivity.vertex_count, connectivity.cell_count)}."
        )
    offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int32)
    values = np.asarray(connectivity.cell_vertex_values, dtype=np.int32)
    cell_offset = 0
    for block in blocks:
        widths = np.diff(offsets[cell_offset : cell_offset + block.cell_count + 1])
        if np.any(widths != block.arity):
            raise ValueError(
                f"CellMesh.polyhedral_connectivity must match block cell widths; got {widths}."
            )
        start = int(offsets[cell_offset])
        stop = int(offsets[cell_offset + block.cell_count])
        expected = np.asarray(block.vertices, dtype=np.int32)
        if not np.array_equal(
            values[start:stop].reshape(expected.shape), np.sort(expected, axis=1)
        ):
            raise ValueError(
                f"CellMesh.polyhedral_connectivity must match sorted block cell vertices; got {block.name!r}."
            )
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
        return connectivity, tetrahedral_complex(
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
        return connectivity, hexahedral_complex(
            connectivity,
            vertex_global_ids=vertex_ids,
            edge_global_ids=entity_ids.get(1),
            face_global_ids=entity_ids.get(2),
            cell_global_ids=cell_ids,
        )
    if any(block.cell_kind == "polyhedron" for block in blocks):
        raise ValueError(
            f"CellMesh.polyhedral_connectivity must be supplied for polyhedron blocks; got {supplied}."
        )
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
        raise ValueError(
            f"CellMesh.blocks must contain at least one cell block; got {len(normalized_blocks)}."
        )
    if not all(
        isinstance(block, (CellBlock, PolyhedralBlock)) for block in normalized_blocks
    ):
        raise TypeError(
            f"CellMesh.blocks must contain only CellBlock or PolyhedralBlock instances; got {tuple(type(block).__name__ for block in normalized_blocks)}."
        )
    names = tuple(block.name for block in normalized_blocks)
    if len(set(names)) != len(names):
        raise ValueError(f"CellMesh.blocks must have unique block names; got {names}.")
    dimensions = {block.topological_dimension for block in normalized_blocks}
    if len(dimensions) != 1:
        raise ValueError(
            f"CellMesh.blocks must share one topological dimension; got {dimensions}."
        )
    topological_dimension = dimensions.pop()
    if ambient_dimension < topological_dimension:
        raise ValueError(
            f"CellMesh.coordinates must have ambient dimension at least the topological dimension; got {point_shape[1]}."
        )
    for block in normalized_blocks:
        vertices_ = np.asarray(block.vertices)
        valid_ = np.asarray(block.vertex_valid, dtype=np.bool_)
        if np.any(vertices_[valid_] >= coordinate_count):
            raise ValueError(
                f"CellMesh.blocks must index declared vertices; got {block.name!r}."
            )
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
        raise ValueError(
            f"CellMesh.entity_global_ids must index declared entity dimensions; got {tuple(entity_ids)}."
        )
    supplied_vertices = entity_ids.get(0)
    if supplied_vertices is not None and vertex_global_ids is not None:
        if not np.array_equal(
            supplied_vertices, np.asarray(vertex_global_ids, dtype=np.int64)
        ):
            raise ValueError(
                f"CellMesh.vertex_global_ids must agree with entity_global_ids at degree zero; got {np.asarray(vertex_global_ids).shape}."
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
        raise ValueError(
            f"CellMesh.vertex_global_ids must have one entry per coordinate; got {global_ids.shape}."
        )
    if np.any(global_ids < 0) or np.unique(global_ids).size != global_ids.size:
        raise ValueError(
            f"CellMesh.vertex_global_ids must be unique nonnegative integers; got {global_ids.shape}."
        )
    cell_global_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in blocks)
    )
    if np.unique(cell_global_ids).size != cell_global_ids.size:
        raise ValueError(
            f"CellMesh.blocks must have unique global IDs across blocks; got {cell_global_ids.shape}."
        )
    supplied_cells = entity_ids.get(topological_dimension)
    if supplied_cells is not None and not np.array_equal(supplied_cells, cell_global_ids):
        raise ValueError(
            f"CellMesh.entity_global_ids must agree with top-dimensional block global IDs; got {supplied_cells.shape}."
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
    """Return logical simplex content independent of local slots and placement."""
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
    match connectivity:
        case SimplicialConnectivity():
            entities = tuple(
                np.asarray(value, dtype=np.int32) for value in connectivity.entities[1:-1]
            )
        case IntervalConnectivity():
            entities = ()
        case PolygonalConnectivity():
            entities = (np.asarray(connectivity.edges, dtype=np.int32),)
        case TetrahedralConnectivity():
            entities = (
                np.asarray(connectivity.edges, dtype=np.int32),
                np.asarray(connectivity.faces, dtype=np.int32),
            )
        case _ as invalid:
            assert_never(invalid)
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
    if has_duplicate_rows(top):
        raise ValueError(
            f"CellMesh.blocks must contain no duplicate simplices; got {cells.shape}."
        )
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
        raise ValueError(
            f"CellMesh.blocks must give each facet at most two incident cells; got {counts}."
        )
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
        raise ValueError(
            f"CellMesh.blocks must use simplex cell kinds for simplicial construction; got {tuple(block.cell_kind for block in blocks)}."
        )
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
            else entity_ids.get(degree, canonical_entity_ids(vertex_ids[rows]))
        )
        if ids.shape != (source.count,):
            raise ValueError(
                f"CellMesh.entity_global_ids must match the entity count at degree {degree}; got {ids.shape}."
            )
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
                f"CellMesh.blocks must use tensor cell dimensions one through three; got {dimension}."
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
    periodic_topology: periodic_topology_owner.PeriodicMeshTopology | None,
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
            f"CellMesh.polyhedral_connectivity must be supplied only for three-dimensional meshes; got {topological_dimension}."
        )
    route_blocks = _tensor_mesh_route_blocks(normalized_blocks)
    if any(block.cell_kind.startswith("simplex:") for block in normalized_blocks):
        if polyhedral_connectivity is not None:
            raise ValueError(
                f"CellMesh.polyhedral_connectivity must be omitted for simplicial meshes; got {tuple(block.cell_kind for block in normalized_blocks)}."
            )
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
        if periodic_topology.lifted_topology_id != topology_id:
            raise ValueError(
                f"CellMesh.periodic_topology must match the lifted mesh topology identity; got {periodic_topology.lifted_topology_id}."
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

    if identities_supplied:
        raise ValueError(
            f"CellMesh.prepared_topology must be reused without separately supplied identities; got {identities_supplied}."
        )
    normalized_blocks = tuple(blocks)
    if len(normalized_blocks) != len(prepared.blocks) or any(
        block is not prepared_block
        for block, prepared_block in zip(normalized_blocks, prepared.blocks, strict=True)
    ):
        raise ValueError(
            f"CellMesh.blocks must be the prepared topology blocks; got {tuple(type(block).__name__ for block in normalized_blocks)}."
        )
    if point_shape[0] != prepared.vertex_global_ids.shape[0]:
        raise ValueError(
            f"CellMesh.coordinates must match the prepared vertex identity count; got {point_shape[0]}."
        )
    if point_shape[1] < prepared.topological_dimension:
        raise ValueError(
            f"CellMesh.coordinates must have ambient dimension at least the topological dimension; got {point_shape[1]}."
        )
    return prepared


class _StoragePlacement(StrictModule):
    """Validated global degree counts and owner-local placement metadata."""

    counts: tuple[int, ...] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    parts: int = eqx.field(static=True)
    bound: int = eqx.field(static=True)


def _prepare_storage_placement(
    global_entity_counts: Sequence[int],
    partition_index: int,
    partition_count: int,
    maximum_materialization_bytes: int,
) -> _StoragePlacement:
    if not x64_enabled():
        raise ValueError(
            "CellMeshStorage.entity_global_ids must retain int64 identity with JAX x64 execution; got x64 disabled."
        )
    counts = tuple(operator.index(value) for value in global_entity_counts)
    rank = operator.index(partition_index)
    parts = operator.index(partition_count)
    bound = operator.index(maximum_materialization_bytes)
    if (
        len(counts) < 2
        or any(isinstance(value, bool) or value <= 0 for value in global_entity_counts)
        or isinstance(partition_index, bool)
        or isinstance(partition_count, bool)
        or not 0 <= rank < parts
        or isinstance(maximum_materialization_bytes, bool)
        or bound < 0
    ):
        raise ValueError(
            "CellMeshStorage.global_entity_counts must be positive with a valid partition and materialization bound; got incompatible placement metadata."
        )
    return _StoragePlacement(counts, rank, parts, bound)


class _StorageRoutes(StrictModule):
    """Validated placement and locally addressable entity closure."""

    counts: tuple[int, ...] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    parts: int = eqx.field(static=True)
    bound: int = eqx.field(static=True)
    ids: tuple[Array, ...]
    owners: tuple[Array, ...]
    owned: tuple[Array, ...]
    coordinates: np.ndarray
    local_coordinate_id: str = eqx.field(static=True)


def _prepare_storage_routes(
    global_entity_counts: Sequence[int],
    entity_global_ids: Sequence[ArrayLike],
    entity_owner: Sequence[ArrayLike],
    *,
    partition_index: int,
    partition_count: int,
    maximum_materialization_bytes: int,
    local_coordinates: ArrayLike,
) -> _StorageRoutes:
    placement = _prepare_storage_placement(
        global_entity_counts,
        partition_index,
        partition_count,
        maximum_materialization_bytes,
    )
    counts, rank, parts, bound = (
        placement.counts,
        placement.rank,
        placement.parts,
        placement.bound,
    )
    if len(entity_global_ids) != len(counts) or len(entity_owner) != len(counts):
        raise ValueError(
            f"CellMeshStorage.entity_global_ids must cover every declared degree with entity_owner; got {(len(entity_global_ids), len(entity_owner), len(counts))}."
        )
    ids = []
    owners = []
    owned = []
    for count, values, placement in zip(
        counts, entity_global_ids, entity_owner, strict=True
    ):
        if (isinstance(values, Array) and not values.is_fully_addressable) or (
            isinstance(placement, Array) and not placement.is_fully_addressable
        ):
            raise ValueError(
                f"CellMeshStorage.entity_global_ids must be fully addressable with local owner maps; got {'non-addressable identity or owner data'}."
            )
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
                "CellMeshStorage.entity_global_ids must be unique with compatible entity_owner; got invalid identity or owner routes."
            )
        ids.append(jnp.asarray(identifiers, dtype=jnp.int64))
        routing = np.asarray(routing, dtype=np.int32)
        owners.append(jnp.asarray(routing, dtype=jnp.int32))
        owned.append(jnp.asarray(routing == rank, dtype=jnp.bool_))
    if (
        isinstance(local_coordinates, Array)
        and not local_coordinates.is_fully_addressable
    ):
        raise ValueError(
            f"CellMeshStorage.local_coordinates must be fully addressable; got {'non-addressable coordinate data'}."
        )
    coordinates = np.asarray(local_coordinates, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] != ids[0].shape[0]
        or coordinates.shape[1] <= 0
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError(
            f"CellMeshStorage.local_coordinates must match finite locally routed vertices; got {coordinates.shape}."
        )
    local_coordinate_id = canonical_fingerprint(array_tree_fingerprint(coordinates))
    return _StorageRoutes(
        counts,
        rank,
        parts,
        bound,
        tuple(ids),
        tuple(owners),
        tuple(owned),
        coordinates,
        local_coordinate_id,
    )


class _StorageClosure(StrictModule):
    """Logical source identity and validated local block-to-source routes."""

    coordinate_geometry_id: str = eqx.field(static=True)
    arrays: tuple[tuple[str, Array], ...]
    blocks: tuple[CellBlock | PolyhedralBlock, ...]
    source_blocks: tuple[tuple[str, str], ...] = eqx.field(static=True)


def _prepare_storage_closure(
    routes: _StorageRoutes,
    logical_topology_id: str,
    logical_geometry_id: str,
    evidence_id: str,
    logical_arrays: Sequence[tuple[str, Array]],
    local_blocks: Sequence[CellBlock | PolyhedralBlock],
    local_coordinates: ArrayLike,
    entity_global_ids: Sequence[ArrayLike],
    entity_owner: Sequence[ArrayLike],
    local_geometry: cell_geometry_owner.CellGeometrySpec | None,
    geometry_projection: cell_geometry_owner.CellGeometryStorageProjection | None,
    geometry_source_blocks: Mapping[str, str] | None,
    logical_coordinate_geometry_id: str | None,
) -> _StorageClosure:
    coordinates, counts, ids = routes.coordinates, routes.counts, routes.ids
    if (
        local_geometry is not None or geometry_projection is not None
    ) and logical_coordinate_geometry_id is None:
        raise ValueError(
            "CellMeshStorage.logical_coordinate_geometry_id must identify the source map for mapped storage; got None."
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
            f"CellMeshStorage.logical_ids must be lowercase SHA-256 digests; got {identities}."
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
            f"CellMeshStorage.logical_arrays must contain uniquely named JAX arrays in sorted name order; got {names}."
        )
    blocks = tuple(local_blocks)
    if blocks:
        _validated_mesh_blocks(coordinates.shape, blocks)
        if not np.array_equal(
            np.concatenate(tuple(np.asarray(block.global_ids) for block in blocks)),
            np.asarray(ids[-1]),
        ):
            raise ValueError(
                "CellMeshStorage.local_blocks must match routed cell identities; got different global IDs."
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
            "CellMeshStorage.local_coordinates must describe an empty, exactly typed local closure; got incompatible closure banks."
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
            f"CellMeshStorage.geometry_source_blocks must name one scientific source bank per local block; got {source_blocks}."
        )
    return _StorageClosure(
        coordinate_geometry_id, arrays, blocks, tuple(source_blocks.items())
    )


class _StorageGeometryBinding(StrictModule):
    """Restored local coefficient geometry and its admitted projection packets."""

    geometry: cell_geometry_owner.CellGeometrySpec | None


def _prepare_storage_geometry_binding(
    routes: _StorageRoutes,
    closure: _StorageClosure,
    local_geometry: cell_geometry_owner.CellGeometrySpec | None,
    geometry_projection: cell_geometry_owner.CellGeometryStorageProjection | None,
    coordinate_global_ids: ArrayLike | None,
) -> _StorageGeometryBinding:
    arrays, blocks = closure.arrays, closure.blocks
    source_blocks = dict(closure.source_blocks)
    coordinate_geometry_id = closure.coordinate_geometry_id
    coordinates, counts, rank, parts = (
        routes.coordinates,
        routes.counts,
        routes.rank,
        routes.parts,
    )
    packets = None
    if geometry_projection is not None:
        geometry_projection.require_source(arrays, coordinate_geometry_id)
        if (
            geometry_projection.partition_count != parts
            or geometry_projection.global_cell_count != counts[-1]
        ):
            raise ValueError(
                "CellMeshStorage.geometry_projection must match global mesh placement; got different partition or cell counts."
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
                "CellMeshStorage.geometry_projection must provide the passive-owner source projection; got incompatible empty-owner banks."
            )
    if local_geometry is None and blocks and geometry_projection is not None:
        if coordinate_global_ids is None:
            raise ValueError(
                f"CellMeshStorage.coordinate_global_ids must be supplied for restored source geometry; got {coordinate_global_ids}."
            )
        local_geometry = cell_geometry_owner.restore_authored_storage_geometry(
            blocks,
            source_blocks,
            geometry_projection,
            rank,
            coordinate_global_ids,
        )
    return _StorageGeometryBinding(local_geometry)


class _StorageCoordinateRoutes(StrictModule):
    """Scientific coordinate identity lowered independently of corner IDs."""

    count: int = eqx.field(static=True)
    ids: Array
    owners: Array
    source_id: str = eqx.field(static=True)


class _ScientificCoordinateIdentity(StrictModule):
    """Validated coordinate IDs and owners before scientific source binding."""

    count: int = eqx.field(static=True)
    ids: Array
    owners: Array
    exact_route_dtypes: bool = eqx.field(static=True)


def _prepare_scientific_coordinate_identity(
    local_geometry: cell_geometry_owner.CellGeometrySpec | None,
    global_coordinate_count: int | None,
    coordinate_global_ids: ArrayLike | None,
    coordinate_owner: ArrayLike | None,
    parts: int,
) -> _ScientificCoordinateIdentity:
    if local_geometry is not None and local_geometry.storage is not None:
        raise TypeError(
            f"CellMeshStorage.local_geometry must be an unbound CellGeometrySpec; got storage type {type(local_geometry.storage).__name__}."
        )
    if (
        global_coordinate_count is None
        or coordinate_global_ids is None
        or coordinate_owner is None
    ):
        raise ValueError(
            f"CellMeshStorage.coordinate_global_ids must be supplied with coordinate DOF count and owners; got {(global_coordinate_count, coordinate_global_ids is not None, coordinate_owner is not None)}."
        )
    coordinate_count = operator.index(global_coordinate_count)
    if isinstance(global_coordinate_count, bool) or coordinate_count <= 0:
        raise ValueError(
            f"CellMeshStorage.global_coordinate_count must be positive; got {coordinate_count}."
        )
    if any(
        isinstance(value, Array) and not value.is_fully_addressable
        for value in (coordinate_global_ids, coordinate_owner)
    ):
        raise ValueError(
            f"CellMeshStorage.coordinate_global_ids must be locally addressable with coordinate_owner; got {'non-addressable scientific coordinate routes'}."
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
            "CellMeshStorage.coordinate_global_ids must match coordinate count and coordinate_owner; got incompatible scientific routes."
        )
    coordinate_ids = jnp.asarray(coordinate_ids_, dtype=jnp.int64)
    coordinate_owners = jnp.asarray(coordinate_owners_, dtype=jnp.int32)
    return _ScientificCoordinateIdentity(
        coordinate_count,
        coordinate_ids,
        coordinate_owners,
        coordinate_ids_.dtype == np.int64 and coordinate_owners_.dtype == np.int32,
    )


def _prepare_storage_coordinate_routes(
    routes: _StorageRoutes,
    closure: _StorageClosure,
    local_geometry: cell_geometry_owner.CellGeometrySpec | None,
    geometry_projection: cell_geometry_owner.CellGeometryStorageProjection | None,
    global_coordinate_count: int | None,
    coordinate_global_ids: ArrayLike | None,
    coordinate_owner: ArrayLike | None,
) -> _StorageCoordinateRoutes:
    counts, ids, owners = routes.counts, routes.ids, routes.owners
    coordinates, rank, parts = routes.coordinates, routes.rank, routes.parts
    local_coordinate_id = routes.local_coordinate_id
    arrays, blocks = closure.arrays, closure.blocks
    source_blocks = dict(closure.source_blocks)
    coordinate_geometry_id = closure.coordinate_geometry_id
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
                "CellMeshStorage.local_geometry must define explicit scientific coordinate routes; got no geometry layout."
            )
        coordinate_count = counts[0]
        coordinate_ids = ids[0]
        coordinate_owners = owners[0]
        geometry_source_id = local_coordinate_id
    else:
        coordinate_identity = _prepare_scientific_coordinate_identity(
            local_geometry,
            global_coordinate_count,
            coordinate_global_ids,
            coordinate_owner,
            parts,
        )
        coordinate_count = coordinate_identity.count
        coordinate_ids = coordinate_identity.ids
        coordinate_owners = coordinate_identity.owners
        witness_arrays = arrays
        witness_count = coordinate_count
        if geometry_projection is None and any(
            name.startswith("geometry/") and not value.is_fully_addressable
            for name, value in arrays
        ):
            raise ValueError(
                f"CellMeshStorage.geometry_projection must be supplied for non-addressable scientific banks; got {geometry_projection}."
            )
        if geometry_projection is not None:
            if (
                geometry_projection.global_coordinate_count != coordinate_count
                or geometry_projection.partition_count != parts
            ):
                raise ValueError(
                    "CellMeshStorage.geometry_projection must match scientific coordinate placement; got different coordinate or partition counts."
                )
            geometry_projection.require_source(arrays, coordinate_geometry_id)
            witness_arrays = geometry_projection.addressable_arrays(rank)
            witness_count = dict(witness_arrays)["geometry/coordinate_ids"].shape[0]
        if local_geometry is not None:
            cell_geometry_owner.require_logical_geometry_lowering(
                local_geometry,
                blocks,
                witness_arrays,
                witness_count,
                coordinate_ids,
                coordinate_owners,
                source_blocks,
            )
            geometry_source_id = cell_geometry_validity_owner.cell_geometry_id(
                local_geometry
            )
        else:
            if not coordinate_identity.exact_route_dtypes:
                raise ValueError(
                    f"CellMeshStorage.coordinate_global_ids must use int64 IDs and int32 owners for zero-resident routes; got {coordinate_identity.exact_route_dtypes}."
                )
            geometry_source_id = canonical_fingerprint(
                {
                    "kind": "zero-resident-coordinate-source",
                    "coordinate_geometry": coordinate_geometry_id,
                    "coordinates": array_tree_fingerprint(coordinates),
                }
            )
    return _StorageCoordinateRoutes(
        coordinate_count, coordinate_ids, coordinate_owners, geometry_source_id
    )


def _storage_local_mask(value: ArrayLike | None, count: int, name: str) -> Array | None:
    if value is None:
        return None
    if isinstance(value, Array) and not value.is_fully_addressable:
        raise ValueError(
            f"CellMeshStorage.{name} must be locally addressable; got a non-addressable array."
        )
    mask = np.asarray(value)
    if mask.shape != (count,) or mask.dtype != np.bool_:
        raise ValueError(
            f"CellMeshStorage.{name} must match the local entity count with boolean dtype; got {(mask.shape, mask.dtype)}."
        )
    return jnp.asarray(mask, dtype=jnp.bool_)


class CellMeshStorage(StrictModule, NonTrainableState):
    """Globally logical storage with a process-local, stable-ID closure view.

    ``CellMesh`` numerical fields are the locally addressable lowered view, not
    global array contents. ``logical_arrays`` are the accepted globally shaped
    arrays used by addressable checkpoint publication. Neither ownership nor
    positive local geometry is a global certificate; ``evidence_id`` binds the
    separately established collective coverage.

    ``local_coordinates`` are sampled topological corners; mapped coordinates
    retain source coefficients in ``local_geometry``. Their scientific
    DOF IDs/count/owners are independent of topological vertex identities.
    ``logical_geometry_id`` binds the sampled carrier, whereas
    ``logical_coordinate_geometry_id`` binds the source map and
    reference ancestry. ``restore_geometry`` validates and restores that
    map from the logical coefficient, basis, route, and ancestry banks.
    Non-addressable banks require a ``CellGeometryStorageProjection`` computed
    collectively from fixed-capacity closure queries before owner-local lowering;
    geometry validation subsequently reads only its addressable banks.
    ``local_physical_boundary_entities`` projects the scientific global boundary
    by stable entity ID at every degree. A resident vertex can belong to that
    boundary even when none of its incident physical facets is resident.
    """

    entity_global_ids: tuple[Array, ...]
    entity_owned: tuple[Array, ...]
    entity_owner: tuple[Array, ...]
    logical_arrays: tuple[tuple[str, Array], ...]
    local_coordinates: Array
    local_blocks: tuple[CellBlock | PolyhedralBlock, ...]
    coordinate_global_ids: Array
    coordinate_owner: Array
    local_geometry: cell_geometry_owner.CellGeometrySpec | None
    geometry_projection: cell_geometry_owner.CellGeometryStorageProjection | None
    global_coordinate_count: int = eqx.field(static=True)
    local_geometry_source_id: str = eqx.field(static=True)
    geometry_source_blocks: tuple[tuple[str, str], ...] = eqx.field(static=True)
    local_physical_boundary_facets: Array | None
    local_physical_boundary_entities: tuple[Array, ...] | None
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

    @checked
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
        local_physical_boundary_entities: Sequence[ArrayLike] | None = None,
        local_neighborhood_complete: ArrayLike | None = None,
        neighborhood_depth: int = 0,
        global_coordinate_count: int | None = None,
        coordinate_global_ids: ArrayLike | None = None,
        coordinate_owner: ArrayLike | None = None,
        local_geometry: cell_geometry_owner.CellGeometrySpec | None = None,
        geometry_source_blocks: Mapping[str, str] | None = None,
        logical_coordinate_geometry_id: str | None = None,
        geometry_projection: cell_geometry_owner.CellGeometryStorageProjection
        | None = None,
    ) -> None:
        routes = _prepare_storage_routes(
            global_entity_counts,
            entity_global_ids,
            entity_owner,
            partition_index=partition_index,
            partition_count=partition_count,
            maximum_materialization_bytes=maximum_materialization_bytes,
            local_coordinates=local_coordinates,
        )
        closure = _prepare_storage_closure(
            routes,
            logical_topology_id,
            logical_geometry_id,
            evidence_id,
            logical_arrays,
            local_blocks,
            local_coordinates,
            entity_global_ids,
            entity_owner,
            local_geometry,
            geometry_projection,
            geometry_source_blocks,
            logical_coordinate_geometry_id,
        )
        local_geometry = _prepare_storage_geometry_binding(
            routes,
            closure,
            local_geometry,
            geometry_projection,
            coordinate_global_ids,
        ).geometry
        coordinate_routes = _prepare_storage_coordinate_routes(
            routes,
            closure,
            local_geometry,
            geometry_projection,
            global_coordinate_count,
            coordinate_global_ids,
            coordinate_owner,
        )
        counts, rank, parts, bound = (
            routes.counts,
            routes.rank,
            routes.parts,
            routes.bound,
        )
        ids, owners, coordinates = routes.ids, routes.owners, routes.coordinates
        local_coordinate_id = routes.local_coordinate_id
        arrays, blocks = closure.arrays, closure.blocks
        source_blocks = dict(closure.source_blocks)
        coordinate_geometry_id = closure.coordinate_geometry_id
        coordinate_count = coordinate_routes.count
        coordinate_ids, coordinate_owners = (
            coordinate_routes.ids,
            coordinate_routes.owners,
        )
        geometry_source_id = coordinate_routes.source_id
        depth = operator.index(neighborhood_depth)
        if isinstance(neighborhood_depth, bool) or depth < 0:
            raise ValueError(
                f"CellMeshStorage.neighborhood_depth must be a nonnegative integer; got {neighborhood_depth!r}."
            )

        physical = _storage_local_mask(
            local_physical_boundary_facets,
            ids[-2].shape[0],
            "local_physical_boundary_facets",
        )
        boundary_entities = None
        if local_physical_boundary_entities is not None:
            if len(local_physical_boundary_entities) != len(ids):
                raise ValueError(
                    "CellMeshStorage.local_physical_boundary_entities must provide every entity degree; got an incomplete receipt."
                )
            boundary = []
            for degree, (value, identifiers) in enumerate(
                zip(local_physical_boundary_entities, ids, strict=True)
            ):
                mask = _storage_local_mask(
                    value,
                    identifiers.shape[0],
                    f"local_physical_boundary_entities[{degree}]",
                )
                if mask is None:
                    raise ValueError(
                        "CellMeshStorage.local_physical_boundary_entities must retain boolean receipts; got a missing degree."
                    )
                boundary.append(mask)
            boundary_entities = tuple(boundary)
            if np.any(np.asarray(boundary_entities[-1])):
                raise ValueError(
                    "CellMeshStorage.local_physical_boundary_entities must exclude top-dimensional cells; got boundary cells."
                )
            if physical is not None and not np.array_equal(
                np.asarray(physical), np.asarray(boundary_entities[-2])
            ):
                raise ValueError(
                    "CellMeshStorage.local_physical_boundary_entities must agree with physical facet receipts; got different facets."
                )
            physical = boundary_entities[-2]
        complete = _storage_local_mask(
            local_neighborhood_complete, ids[-1].shape[0], "local_neighborhood_complete"
        )
        self.entity_global_ids = tuple(ids)
        self.entity_owner = tuple(owners)
        self.entity_owned = routes.owned
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
        self.local_physical_boundary_entities = boundary_entities
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
                "local_physical_boundary_entities": (
                    None
                    if boundary_entities is None
                    else array_tree_fingerprint(boundary_entities)
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

    @classmethod
    @checked
    def from_stored_fields(cls, value: CellMeshStorage, /) -> CellMeshStorage:
        """Revalidate stored logical banks and rebuild their local closure."""
        scientific = (
            value.local_geometry is not None or value.geometry_projection is not None
        )
        return cls(
            value.global_entity_counts,
            value.entity_global_ids,
            value.entity_owner,
            partition_index=value.partition_index,
            partition_count=value.partition_count,
            logical_topology_id=value.logical_topology_id,
            logical_geometry_id=value.logical_geometry_id,
            evidence_id=value.evidence_id,
            logical_arrays=value.logical_arrays,
            local_coordinates=value.local_coordinates,
            local_blocks=value.local_blocks,
            maximum_materialization_bytes=value.maximum_materialization_bytes,
            local_physical_boundary_facets=value.local_physical_boundary_facets,
            local_physical_boundary_entities=value.local_physical_boundary_entities,
            local_neighborhood_complete=value.local_neighborhood_complete,
            neighborhood_depth=value.neighborhood_depth,
            global_coordinate_count=value.global_coordinate_count if scientific else None,
            coordinate_global_ids=value.coordinate_global_ids if scientific else None,
            coordinate_owner=value.coordinate_owner if scientific else None,
            local_geometry=value.local_geometry,
            geometry_source_blocks=dict(value.geometry_source_blocks),
            logical_coordinate_geometry_id=value.logical_coordinate_geometry_id,
            geometry_projection=value.geometry_projection,
        )

    @checked
    def require_local_topology(self, topology: CellComplexTopology, /) -> None:
        if topology.dimension + 1 != len(self.global_entity_counts):
            raise ValueError(
                f"CellMeshStorage.global_entity_counts must match local closure topology degrees; got {(len(self.global_entity_counts), topology.dimension)}."
            )
        for degree, identifiers in enumerate(self.entity_global_ids):
            if not np.array_equal(
                np.asarray(identifiers), np.asarray(topology.entities(degree).entity_ids)
            ):
                raise ValueError(
                    f"CellMeshStorage.entity_global_ids must match local topology IDs at degree {degree}; got {identifiers.shape}."
                )

    @checked
    def restore_geometry(
        self, *, budget: CoordinateEnclosureBudget | None = None
    ) -> cell_geometry_owner.CellGeometrySpec:
        """Restore the stored coefficient basis and bind its coordinate routes."""

        geometry = self.local_geometry
        if geometry is None:
            return cell_geometry_owner.CellGeometrySpec(
                {
                    block.name: (
                        cell_geometry_owner.CellVertexGeometryElement(
                            block.cell_kind, block.arity
                        )
                        if block.cell_kind in ("polygon", "polyhedron")
                        else cell_geometry_owner.coordinate_lagrange_element(
                            block.cell_kind, 1
                        )
                    )
                    for block in self.local_blocks
                },
                {block.name: block.vertices for block in self.local_blocks},
                self.local_coordinates,
                storage=self,
                budget=budget,
            )
        return cell_geometry_owner.CellGeometrySpec(
            dict(zip(geometry.block_names, geometry.elements, strict=True)),
            dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
            geometry.coordinates,
            storage=self,
            restriction_source=geometry.restriction_source,
            periodic_source=geometry.periodic_source,
            exact_source=geometry.exact_source,
            budget=budget,
        )


def _empty_storage_topology(storage: CellMeshStorage, /) -> _PreparedCellMeshTopology:
    """Construct the zero subcomplex of the nonempty logical carrier."""
    if storage.local_blocks or any(value.shape[0] for value in storage.entity_global_ids):
        raise ValueError(
            f"CellMeshStorage.local_blocks must describe zero resident entities for empty topology construction; got {(len(storage.local_blocks), tuple(value.shape for value in storage.entity_global_ids))}."
        )
    projection = storage.geometry_projection
    if projection is None or not projection.source_elements:
        raise ValueError(
            "CellMeshStorage.geometry_projection must declare the global source layout; got no source elements."
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
                f"CellMeshStorage.geometry_projection must use source charts matching the mesh topological degree; got {target_dimension}."
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


class _CellMeshIdentity(StrictModule):
    """Geometry-layout and numerical IDs bound to the accepted topology."""

    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


def _prepare_cell_mesh_identity(
    prepared: _PreparedCellMeshTopology,
    points: np.ndarray,
    storage: CellMeshStorage | None,
    numeric_version: str,
) -> _CellMeshIdentity:
    topology_id = prepared.topology_id if storage is None else storage.logical_topology_id
    geometry_layout_id = canonical_fingerprint(
        {
            "kind": "cell-mesh-geometry-layout",
            "topology": topology_id,
            "ambient_dimension": points.shape[1],
            "coordinate_count": (
                points.shape[0] if storage is None else storage.global_entity_counts[0]
            ),
            "coordinate_dtype": str(points.dtype),
        }
    )
    match storage:
        case CellMeshStorage() as placement:
            geometry_id = placement.logical_geometry_id
        case None:
            match prepared.blocks:
                case (CellBlock(cell_kind="interval" | "triangle" | "tetrahedron"),):
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
                case _:
                    geometry_id = canonical_fingerprint(
                        {
                            "kind": "cell-mesh-geometry",
                            "layout": geometry_layout_id,
                            "coordinates": array_tree_fingerprint(points),
                            "numeric_version": str(numeric_version),
                        }
                    )
        case _ as invalid:
            assert_never(invalid)
    return _CellMeshIdentity(topology_id, geometry_layout_id, geometry_id)


class _MixedPolyhedralBlocks(StrictModule):
    """Arity-grouped blocks with face loops and stable source cell identities."""

    standard_blocks: tuple[CellBlock, ...]
    poly_blocks: tuple[PolyhedralBlock, ...]
    explicit_cells: tuple[tuple[ArrayLike, ...], ...]
    explicit_ids: tuple[int, ...] = eqx.field(static=True)


def _polyhedral_cell_vertex_set(cell: tuple[ArrayLike, ...]) -> set[int]:
    """Identify distinct cell vertices from authored face loops on the host."""
    return {int(vertex) for face in cell for vertex in np.asarray(face, dtype=np.int32)}


def _append_polyhedral_arity_blocks(
    name: str,
    cells: tuple[tuple[ArrayLike, ...], ...],
    ids: NDArray[np.int64],
    poly_blocks: list[PolyhedralBlock],
    explicit_cells: list[tuple[ArrayLike, ...]],
    explicit_ids: list[int],
) -> None:
    """Keep stable source identities aligned with each vertex-arity block."""
    counts = np.asarray(
        [len(_polyhedral_cell_vertex_set(cell)) for cell in cells],
        dtype=np.int32,
    )
    order = np.argsort(counts, kind="stable")
    ordered_cells = tuple(cells[int(index)] for index in order)
    ordered_ids = ids[order]
    ordered_counts = counts[order]
    unique_counts = tuple(np.unique(ordered_counts))
    for vertex_count in unique_counts:
        selected = np.flatnonzero(ordered_counts == vertex_count)
        selected_cells = tuple(ordered_cells[int(index)] for index in selected)
        selected_ids = ordered_ids[selected]
        vertices = np.asarray(
            [sorted(_polyhedral_cell_vertex_set(cell)) for cell in selected_cells],
            dtype=np.int32,
        )
        block_name = name if len(unique_counts) == 1 else f"{name}-{vertex_count}"
        poly_blocks.append(
            PolyhedralBlock(
                block_name,
                vertices,
                global_ids=selected_ids,
            )
        )
        explicit_cells.extend(selected_cells)
        explicit_ids.extend(int(value) for value in selected_ids)


def _prepare_mixed_polyhedral_blocks(
    blocks: Sequence[CellBlock],
    polyhedra: Mapping[str, Sequence[Sequence[ArrayLike]]],
    polyhedral_cell_global_ids: Mapping[str, ArrayLike] | None,
) -> _MixedPolyhedralBlocks:
    standard_blocks = tuple(blocks)
    if any(
        block.cell_kind not in ("tetrahedron", "hexahedron", "prism", "pyramid")
        for block in standard_blocks
    ):
        raise ValueError(
            f"CellMesh.from_mixed_3d.blocks must use standard volume cell kinds; got {tuple(block.cell_kind for block in standard_blocks)}."
        )
    named_cells = tuple(
        (str(name), tuple(tuple(cell) for cell in cells))
        for name, cells in sorted(polyhedra.items())
    )
    if any(not name or not cells for name, cells in named_cells):
        raise ValueError(
            f"CellMesh.from_mixed_3d.polyhedra must have nonempty block names and cell collections; got {tuple((name, len(cells)) for name, cells in named_cells)}."
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
            f"CellMesh.from_mixed_3d.polyhedral_cell_global_ids must cover every polyhedral block when supplied; got {tuple(id_mapping)}."
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
                f"CellMesh.from_mixed_3d.polyhedral_cell_global_ids must match the cell count for source block {name!r}; got {ids.shape}."
            )
        _append_polyhedral_arity_blocks(
            name, cells, ids, poly_blocks, explicit_cells, explicit_ids
        )
        next_id = max(next_id, int(np.max(ids, initial=next_id - 1)) + 1)
    return _MixedPolyhedralBlocks(
        standard_blocks,
        tuple(poly_blocks),
        tuple(explicit_cells),
        tuple(explicit_ids),
    )


class CellMesh(StrictModule, NonTrainableState):
    """Computational mesh shared by unstructured discretizations.

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
    periodic_topology: periodic_topology_owner.PeriodicMeshTopology | None
    storage: CellMeshStorage | None
    topological_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        coordinates: ArrayLike,
        blocks: Sequence[CellBlock | PolyhedralBlock],
        /,
        *,
        vertex_global_ids: ArrayLike | None = None,
        entity_global_ids: Mapping[int, ArrayLike] | None = None,
        polyhedral_connectivity: PolyhedralConnectivity | None = None,
        periodic_topology: periodic_topology_owner.PeriodicMeshTopology | None = None,
        numeric_version: str = "0",
        storage: CellMeshStorage | None = None,
        _prepared_topology: _PreparedCellMeshTopology | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        if isinstance(coordinates, Array) and not coordinates.is_fully_addressable:
            raise ValueError(
                f"CellMesh.coordinates must be locally addressable with global arrays bound in storage; got {'non-addressable coordinates'}."
            )
        if storage is not None and _prepared_topology is None:
            if vertex_global_ids is not None or entity_global_ids is not None:
                raise ValueError(
                    f"CellMesh.entity_global_ids must be omitted with storage-backed identity routing; got {(vertex_global_ids is not None, entity_global_ids is not None)}."
                )
            entity_global_ids = dict(enumerate(storage.entity_global_ids))
        points = np.asarray(coordinates, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] == 0
            or (points.shape[0] == 0 and storage is None)
        ):
            raise ValueError(
                f"CellMesh.coordinates must be rank two with nonempty serial data or bound zero-resident storage; got {points.shape}."
            )
        if not np.all(np.isfinite(points)):
            raise ValueError(f"CellMesh.coordinates must be finite; got {points.shape}.")
        if (
            storage is not None
            and canonical_fingerprint(array_tree_fingerprint(points))
            != storage.local_coordinate_id
        ):
            raise ValueError(
                "CellMesh.coordinates must match storage.local_coordinate_id; got a different local coordinate digest."
            )
        if points.shape[0] == 0 and storage is not None:
            if (
                blocks
                or polyhedral_connectivity is not None
                or periodic_topology is not None
            ):
                raise ValueError(
                    f"CellMesh.blocks must be empty without separate connectivity or periodic descriptors for zero-resident storage; got {(len(blocks), polyhedral_connectivity is not None, periodic_topology is not None)}."
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
            # A transient request ledger owns the exact lift proof; it is not
            # part of the mesh identity.
            prepared.periodic_topology.require_lift(points, budget=budget)
        if storage is not None:
            storage.require_local_topology(prepared.topology)
            if tuple(block.block_id for block in prepared.blocks) != tuple(
                block.block_id for block in storage.local_blocks
            ):
                raise ValueError(
                    "CellMesh.blocks must match storage.local_blocks; got different block identities."
                )
        identity = _prepare_cell_mesh_identity(prepared, points, storage, numeric_version)
        topology_id = identity.topology_id
        geometry_layout_id = identity.geometry_layout_id
        geometry_id = identity.geometry_id
        support = DiscreteSupport(prepared.topology, points.shape[1], geometry_layout_id)
        self.coordinates = (
            jax.device_put(points) if storage is None else storage.local_coordinates
        )
        self.blocks = prepared.blocks
        self.vertex_global_ids = jax.device_put(prepared.vertex_global_ids)
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

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: CellMesh, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> CellMesh:
        """Revalidate stored incidence and rebuild any quotient on its lifted carrier."""
        if value.storage is not None:
            return cls(
                value.coordinates,
                value.blocks,
                storage=value.storage,
                periodic_topology=value.periodic_topology,
                numeric_version=value.numeric_version,
                budget=budget,
            )
        lifted = cls.lifted_from_stored_fields(value, budget=budget)
        periodic = value.periodic_topology
        if periodic is None:
            return lifted
        return cls._from_stored_carrier(
            value, periodic.rebuilt(lifted, budget=budget), budget=budget
        )

    @classmethod
    @checked
    def lifted_from_stored_fields(
        cls: type[CellMesh],
        value: CellMesh,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> CellMesh:
        """Revalidate stored lifted incidence without binding a quotient descriptor."""
        value.require_dense("CellMesh.lifted_from_stored_fields")
        return cls._from_stored_carrier(value, None, budget=budget)

    @classmethod
    def _from_stored_carrier(
        cls: type[CellMesh],
        value: CellMesh,
        periodic: periodic_topology_owner.PeriodicMeshTopology | None,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> CellMesh:
        """Rebuild a carrier from stored topological IDs and authored face incidence."""
        entity_ids = {
            entities.intrinsic_dimension: entities.entity_ids
            for entities in value.topology.entity_sets
        }
        match value.connectivity:
            case PolyhedralConnectivity() as connectivity:
                polyhedral = connectivity
            case (
                IntervalConnectivity()
                | PolygonalConnectivity()
                | TetrahedralConnectivity()
                | HexahedralConnectivity()
                | SimplicialConnectivity()
            ):
                polyhedral = None
            case _ as invalid:
                assert_never(invalid)

        return cls(
            value.coordinates,
            value.blocks,
            vertex_global_ids=value.vertex_global_ids,
            entity_global_ids=entity_ids,
            polyhedral_connectivity=polyhedral,
            periodic_topology=periodic,
            numeric_version=value.numeric_version,
            budget=budget,
        )

    def require_dense(self, operation: str, /) -> None:
        """Guard a serial-only consumer before any implicit global conversion."""
        if self.storage is not None:
            raise ValueError(
                f"CellMesh.storage must be explicitly materialized within bounds for {operation}; got {type(self.storage).__name__}."
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
            raise TypeError(
                f"CellMesh.from_simplices.dimension must be a non-boolean integer; got {type(dimension).__name__}."
            )
        if dimension < 1:
            raise ValueError(
                f"CellMesh.from_simplices.dimension must be positive; got {dimension}."
            )
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
        """Build a mixed-arity polygon mesh from cyclic vertex loops."""

        points = np.asarray(coordinates, dtype=np.float64)
        loops = tuple(np.asarray(loop, dtype=np.int32) for loop in polygons)
        if not loops:
            raise ValueError(
                f"CellMesh.from_polygons.polygons must contain at least one cell; got {len(loops)}."
            )
        if any(loop.ndim != 1 or loop.size < 3 for loop in loops):
            raise ValueError(
                f"CellMesh.from_polygons.polygons must contain rank-one loops with at least three vertices; got {tuple(loop.shape for loop in loops)}."
            )
        identifiers = (
            np.arange(len(loops), dtype=np.int64)
            if cell_global_ids is None
            else np.asarray(cell_global_ids, dtype=np.int64)
        )
        if identifiers.shape != (len(loops),):
            raise ValueError(
                f"CellMesh.from_polygons.cell_global_ids must have one ID per polygon; got {identifiers.shape}."
            )
        grouped: dict[int, list[tuple[np.ndarray, int]]] = {}
        for loop, identifier in zip(loops, identifiers, strict=True):
            if np.any(loop < 0) or np.any(loop >= points.shape[0]):
                raise ValueError(
                    f"CellMesh.from_polygons.polygons must index declared vertices; got {loop.shape}."
                )
            cell_points = points[loop]
            area2 = float(
                np.sum(
                    cell_points[:, 0] * np.roll(cell_points[:, 1], -1)
                    - np.roll(cell_points[:, 0], -1) * cell_points[:, 1]
                )
            )
            if not np.isfinite(area2) or area2 == 0.0:
                raise ValueError(
                    f"CellMesh.from_polygons.polygons must have finite nonzero signed area; got {area2}."
                )
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
        """Build a mesh from closed outward-oriented face loops."""

        points = np.asarray(coordinates, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                f"CellMesh.from_polyhedra.coordinates must have shape (vertex_count, d >= 3); got {points.shape}."
            )
        normalized_cells = tuple(tuple(cell) for cell in cells)
        if not normalized_cells:
            raise ValueError(
                f"CellMesh.from_polyhedra.cells must contain at least one cell; got {len(normalized_cells)}."
            )
        source_ids = (
            np.arange(len(normalized_cells), dtype=np.int64)
            if cell_global_ids is None
            else np.asarray(cell_global_ids, dtype=np.int64)
        )
        if source_ids.shape != (len(normalized_cells),):
            raise ValueError(
                f"CellMesh.from_polyhedra.cell_global_ids must match the polyhedral cell count; got {source_ids.shape}."
            )
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
        prepared_blocks = _prepare_mixed_polyhedral_blocks(
            blocks,
            polyhedra,
            polyhedral_cell_global_ids,
        )
        standard_blocks = prepared_blocks.standard_blocks
        poly_blocks = prepared_blocks.poly_blocks
        explicit_cells = prepared_blocks.explicit_cells
        explicit_ids = prepared_blocks.explicit_ids
        combined_blocks = (*standard_blocks, *poly_blocks)
        if not combined_blocks:
            raise ValueError(
                f"CellMesh.from_mixed_3d.blocks must provide at least one dense or polyhedral block; got {len(combined_blocks)}."
            )
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
        raise KeyError(f"CellMesh.block.name must name a mesh block; got {requested!r}.")

    def entity_set(self, dimension: int, /) -> EntitySet:
        target = int(dimension)
        for entities in self.topology.entity_sets:
            if entities.intrinsic_dimension == target:
                return entities
        raise KeyError(
            f"CellMesh.entity_set.dimension must name a declared entity degree; got {target}."
        )

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
                f"CellMesh.with_coordinates.storage must supply renewed logical geometry evidence for owner-local refresh; got {storage}."
            )
        if self.storage is None and storage is not None:
            raise ValueError(
                f"CellMesh.with_coordinates.storage must preserve mesh storage placement; got {'storage supplied for a serial mesh'}."
            )
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
                f"CellMesh.with_coordinates.storage must preserve logical topology and local placement; got {(storage.logical_topology_id, storage.partition_index, storage.partition_count)}."
            )
        points = jnp.asarray(coordinates)
        if points.shape != self.coordinates.shape:
            raise ValueError(
                f"CellMesh.with_coordinates.coordinates must preserve coordinate shape; got {points.shape}."
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


def cell_block_construction_bounds(cell_count: int, arity: int, /) -> int:
    """Return the temporary bytes admitting one ``CellBlock``.

    ``K = cell_count * arity`` corners and ``C = cell_count``. Bytes bound the
    live host temporaries: the rows (4K) and mask (K) stay live while the row
    sums (8C), active gather and sign mask (5K), padded ``where`` and its sorted
    copy (8K, then 4K of sorted rows), adjacent-equality masks (3K) and
    duplicate test (row-order key, column, order and sort buffer 32C; gathered
    rows 4K; run masks K + C) run, followed by the identity banks (26C): peak
    ``14K + 33C``. Device copies of retained arrays are not counted.
    """
    cells, width = operator.index(cell_count), operator.index(arity)
    if cells < 1 or width < 1:
        raise ValueError(
            f"cell_block_construction_bounds.shape must have positive cell count and arity; got {(cells, width)}."
        )
    corners = cells * width
    return 16 * corners + 40 * cells + 65536


def cell_mesh_construction_bounds(
    vertex_count: int,
    ambient_dimension: int,
    blocks: Sequence[tuple[str, int, int]],
    /,
    *,
    realized: CellMesh | None = None,
) -> int:
    """Return the temporary bytes admitting one ``CellMesh``.

    ``blocks`` holds each block's ``(cell_kind, cell_count, arity)``. The bound
    covers ``CellMesh(coordinates, blocks)`` with optional ``vertex_global_ids``
    and ``numeric_version``; supplied entity IDs, polyhedral connectivity,
    periodic topology, storage and prepared topologies are outside it.
    ``polyhedron`` blocks raise ``ValueError``: their authored face loops have
    sizes the cell counts do not bound. Given the mesh built from these inputs
    as ``realized``, entity counts are its actual ones instead of their
    occurrence bounds.

    Bytes bound live host temporaries: a routine's peak is every named array it
    may hold at once plus its largest expression temporary or callee peak.
    Python containers cost 256 bytes per entry. Hashing streams 64 KiB chunks;
    one MiB covers chunk buffers, JSON metadata and small objects. Device
    copies of retained arrays are not counted.

    With ``V`` vertices, ``d`` axes, ``N`` cells and ``K`` corners, distinct
    entities are bounded by their occurrences: polygon edges by sides
    (``E <= K``); tetrahedral faces ``4N`` and edges ``6N``; hexahedral faces
    ``6N`` and edges ``12N``; standard-cell face loops ``Fl`` (4/6/5/5 per
    tetrahedron/hexahedron/prism/pyramid) bound faces, and loop entries ``L``
    (12/24/18/16) bound face entries and edges; simplex ``k``-faces
    ``N * C(D + 1, k + 1)``. Each count enters the bound monotonically, so a
    bound from upper values admits any smaller construction.
    """
    vertices = operator.index(vertex_count)
    ambient = operator.index(ambient_dimension)
    shapes = tuple(
        (str(kind), operator.index(count), operator.index(arity))
        for kind, count, arity in blocks
    )
    if (
        vertices < 1
        or ambient < 1
        or not shapes
        or any(count < 1 or arity < 1 for _, count, arity in shapes)
    ):
        raise ValueError(
            f"cell_mesh_construction_bounds.counts must be positive with at least one block; got {(vertices, ambient, shapes)}."
        )
    dimensions: set[int] = set()
    for kind, _, arity in shapes:
        if kind == "polyhedron":
            raise ValueError(
                "cell_mesh_construction_bounds.blocks must not contain polyhedron blocks; got authored face loops whose sizes the cell counts do not bound."
            )
        if kind == "polygon":
            dimensions.add(2)
            continue
        reference = reference_cell_topology(kind)
        if len(reference.vertices) != arity:
            raise ValueError(
                f"cell_mesh_construction_bounds.blocks must use the reference arity of {kind!r}; got {arity}."
            )
        dimensions.add(reference.dimension)
    if len(dimensions) != 1:
        raise ValueError(
            f"cell_mesh_construction_bounds.blocks must share one topological dimension; got {sorted(dimensions)}."
        )
    dimension = dimensions.pop()
    kinds = tuple(kind for kind, _, _ in shapes)
    cells = sum(count for _, count, _ in shapes)
    corners = sum(count * arity for _, count, arity in shapes)
    points = vertices * ambient
    entry = 256

    def entity_count(degree: int, upper: int, /) -> int:
        # Occurrence bound a priori; the built count for a realization.
        return upper if realized is None else realized.topology.entities(degree).count

    # Shared peak bytes beyond their caller-held inputs.
    def rows(n: int, /) -> int:
        # row_order: one int64 key and its stable argsort, or per-column sorts.
        return 32 * n

    def duplicates(n: int, w: int, itemsize: int, /) -> int:
        # has_duplicate_rows: row order, gathered rows and run masks.
        return rows(n) + (1 + (itemsize + 1) * w) * n

    def groups(n: int, w: int, itemsize: int, /) -> int:
        # first_appearance_groups: row order, gathered rows and run masks, then
        # firsts, their argsort, run groups, outputs and cumulative temporaries.
        return rows(n) + (40 + (itemsize + 1) * w) * n

    def identifiers(n: int, /) -> int:
        # _resolved_entity_ids: int64 copy, sign mask and np.unique banks.
        return 26 * n

    def canonical(n: int, /) -> int:
        # canonical_entity_ids: lexsort with its key buffer, IDs and range.
        return 40 * n

    def entity_set(n: int, /) -> int:
        # EntitySubset and EntitySet: host ID and mask copies, active IDs,
        # their sorted copy and comparison masks.
        return 40 * n

    def relation(routes: int, targets: int, /) -> int:
        # EdgeRelation with its arange/repeat arguments: host route copies,
        # int32 casts, validity mask and two host bound checks.
        return 40 * routes + 8 * targets

    def incidence(routes: int, lower: int, upper: int, /) -> int:
        # OrientedIncidence with its sign argument: masks, active signs,
        # endpoint and ID gathers, the stacked (lower ID, upper ID, sign) rows,
        # their order and ordered copy.
        return 112 * routes + 8 * (lower + upper)

    def chain(
        lower: int, upper: int, sources: int, middle: int, targets: int, width: int, /
    ) -> int:
        # Boundary-of-boundary check: COO/CSR boundaries of both incidences and
        # their product, which has at most `width` (the boundary size of a
        # middle entity) nonzeros per upper route.
        return (
            56 * (lower + upper) + 16 * width * upper + 32 * (sources + middle + targets)
        )

    def validated(n: int, w: int, /) -> int:
        # _validated_cells on vertex-index rows: int32 rows, range masks,
        # row-sorted copy, distinct-vertex mask and the duplicate-row test.
        return 8 * n * w + duplicates(n, w, 4)

    # Held throughout: the float64 coordinate copy and the vertex and cell
    # global-ID banks. Before topology: the finiteness mask, per-block index
    # checks (host rows, mask, active gather, comparison) and the vertex- and
    # concatenated cell-ID uniqueness tests.
    held = 8 * points + 8 * vertices + 8 * cells
    phases = [points, 10 * corners, 18 * vertices, 25 * cells]
    # Dimension-qualified cubes are rebuilt as native blocks.
    phases.extend(
        cell_block_construction_bounds(count, arity)
        for kind, count, arity in shapes
        if kind.startswith("tensor:")
    )
    entity_counts: tuple[int, ...] = ()
    if any(kind.startswith("simplex:") for kind in kinds):
        if any(
            kind not in ("interval", "triangle", "tetrahedron")
            and not kind.startswith("simplex:")
            for kind in kinds
        ):
            raise ValueError(
                f"cell_mesh_construction_bounds.blocks must use simplex kinds for simplicial construction; got {kinds}."
            )
        width = dimension + 1
        local = tuple(comb(width, degree + 1) for degree in range(width))
        level = (
            vertices,
            *(
                entity_count(degree, cells * local[degree])
                for degree in range(1, dimension)
            ),
            cells,
        )
        incident = tuple(
            (degree + 1) * level[degree] if degree else 0 for degree in range(width)
        )
        # Held: concatenated and sorted cells, every vertex level, boundary
        # masks, cell orientations, and per-degree cell routes and signs.
        held_route = (
            8 * corners
            + 4 * vertices
            + sum(4 * (degree + 1) * level[degree] for degree in range(1, dimension))
            + sum(level[:dimension])
            + 9 * cells
            + 12 * cells * sum(local)
        )
        steps = [4 * corners, duplicates(cells, width, 4)]
        # _simplex_mesh_levels: one Python set and sorted list per degree.
        for degree in range(1, dimension):
            steps.append(2 * entry * level[degree])
        # simplicial_cell_complex: level checks, entity sets, then one
        # np.unique(axis=0) facet lookup per degree, whose iterations may
        # overlap and are summed.
        lookups = 0
        for degree in range(width):
            span = degree + 1
            steps.append(2 * span * level[degree] + duplicates(level[degree], span, 4))
            steps.append(
                8 * level[degree]
                + max(
                    entity_set(level[degree]),
                    canonical(level[degree]) if degree else 0,
                )
            )
        for degree in range(1, width):
            stacked = level[degree - 1] + incident[degree]
            lookups += (
                4 * degree * (incident[degree] + stacked)
                + (12 * degree + 45) * stacked
                + 16 * incident[degree]
                + max(
                    relation(incident[degree], level[degree]),
                    incidence(incident[degree], level[degree - 1], level[degree]),
                )
            )
        steps.append(lookups)
        # Both the canonical and the published complex check every chain pair.
        steps.extend(
            chain(
                incident[degree],
                incident[degree + 1],
                level[degree - 1],
                level[degree],
                level[degree + 1],
                degree + 1,
            )
            for degree in range(1, dimension)
        )
        # _simplex_mesh_boundary: facet counts, then per degree one Python set of
        # boundary faces and one membership list over the level.
        facets = level[dimension - 1]
        steps.append(
            9 * incident[dimension]
            + (9 + 4 * dimension) * facets
            + entry
            * max(
                facets * comb(dimension, degree + 1) + level[degree]
                for degree in range(dimension)
            )
        )
        # Published entity sets, with canonical IDs for intermediate degrees.
        for degree in range(width):
            span = degree + 1
            interior = 0 < degree < dimension
            steps.append(
                (
                    (8 * span + 8) * level[degree] + canonical(level[degree])
                    if interior
                    else 0
                )
                + entity_set(level[degree])
            )
        # Cell orientation and the oriented incidences.
        steps.append(25 * cells)
        for degree in range(1, width):
            steps.append(
                (28 if degree == dimension else 8) * incident[degree]
                + incidence(incident[degree], level[degree - 1], level[degree])
            )
        # _simplex_cell_routes: per degree a Python lookup over the level, the
        # selected and sorted local rows, a Python route list and orientations.
        for degree in range(width):
            span = degree + 1
            occurrences = cells * local[degree]
            steps.append(entry * level[degree] + (8 * span + entry + 25) * occurrences)
        # SimplicialConnectivity: sign and range masks per degree.
        steps.append(10 * cells * max(local))
        route = held_route + max(steps)
        entity_counts = level[1:dimension]
    elif dimension == 1:
        if any(kind not in ("interval", "tensor:1") for kind in kinds):
            raise ValueError(
                f"cell_mesh_construction_bounds.blocks must use interval blocks in one dimension; got {kinds}."
            )
        # Host block rows and their concatenation (held through the complex);
        # connectivity validates them and counts vertex valences.
        connectivity = 4 * corners + max(4 * corners, validated(cells, 2), 9 * vertices)
        complex_ = (
            8 * corners
            + 8 * vertices
            + 8 * cells
            + max(
                identifiers(vertices),
                identifiers(cells),
                entity_set(vertices),
                entity_set(cells) + cells,
                relation(corners, cells),
                incidence(corners, vertices, cells),
            )
        )
        route = max(connectivity, complex_)
    elif dimension == 2:
        if any(
            kind not in ("triangle", "quadrilateral", "polygon", "tensor:2")
            for kind in kinds
        ):
            raise ValueError(
                f"cell_mesh_construction_bounds.blocks must use polygonal blocks in two dimensions; got {kinds}."
            )
        triangles = sum(count for kind, count, _ in shapes if kind == "triangle")
        quadrilaterals = sum(
            count for kind, count, _ in shapes if kind in ("quadrilateral", "tensor:2")
        )
        polygons = tuple(
            (count, arity) for kind, count, arity in shapes if kind == "polygon"
        )
        checked_rows = [
            *(
                (count, arity)
                for count, arity in ((triangles, 3), (quadrilaterals, 4))
                if count
            ),
            *polygons,
        ]
        validated_peak = max(validated(count, arity) for count, arity in checked_rows)
        # Same-arity polygon blocks: concatenation, sorted copy, duplicate test.
        duplicate_peak = 0
        for arity in {arity for _, arity in polygons}:
            same = [count for count, other in polygons if other == arity]
            if len(same) > 1:
                count = sum(same)
                duplicate_peak = max(
                    duplicate_peak,
                    8 * count * arity + duplicates(count, arity, 4),
                )
        padded = cells * max(4, *(arity for _, _, arity in shapes))
        edges = entity_count(1, corners)
        # Host rows and concatenations, padded cell banks, side rows and keys,
        # signs, edge counts and orientation sums, cell-edge banks and boundary
        # masks; the complex then copies its host edge and cell-edge banks,
        # row-sorts edge keys and gathers the active cell-edge routes and signs.
        connectivity = 8 * corners + max(
            validated_peak,
            duplicate_peak,
            21 * padded
            + 4 * cells
            + 32 * corners
            + 29 * edges
            + vertices
            + max(
                4 * padded,
                8 * corners,
                groups(corners, 2, 4),
                20 * edges,
            ),
        )
        complex_ = (
            4 * corners
            + 32 * edges
            + 13 * padded
            + 8 * vertices
            + 12 * cells
            + max(
                identifiers(vertices),
                identifiers(cells),
                16 * edges + canonical(edges),
                entity_set(vertices),
                entity_set(edges),
                entity_set(cells) + cells,
                relation(2 * edges, edges),
                8 * corners + relation(corners, cells),
                incidence(2 * edges, vertices, edges),
                8 * corners + incidence(corners, edges, cells),
                chain(2 * edges, corners, vertices, edges, cells, 2),
            )
        )
        route = max(connectivity, complex_)
        entity_counts = (edges,)
    elif dimension == 3:
        if any(
            kind not in ("tetrahedron", "hexahedron", "prism", "pyramid", "tensor:3")
            for kind in kinds
        ):
            raise ValueError(
                f"cell_mesh_construction_bounds.blocks must use standard volume blocks in three dimensions; got {kinds}."
            )
        if len(shapes) == 1 and kinds[0] == "tetrahedron":
            faces, edges = entity_count(2, 4 * cells), entity_count(1, 6 * cells)
            # Host rows, local faces, inversions and signs, row-sorted face
            # keys, face counts and orientation, face and edge rows, face-edge
            # routes and boundary masks; the complex copies its host banks and
            # gathers and row-sorts the global edge and face keys.
            connectivity = 16 * cells + max(
                validated(cells, 4),
                180 * cells
                + 117 * faces
                + 17 * edges
                + vertices
                + max(
                    16 * cells,
                    groups(4 * cells, 3, 4),
                    20 * faces,
                    groups(3 * faces, 2, 4),
                ),
            )
            complex_ = (
                16 * edges
                + 32 * faces
                + 24 * cells
                + 8 * vertices
                + max(
                    identifiers(vertices),
                    identifiers(cells),
                    32 * edges + canonical(edges),
                    48 * faces + canonical(faces),
                    entity_set(vertices),
                    entity_set(edges),
                    entity_set(faces),
                    entity_set(cells) + cells,
                    relation(2 * edges, edges),
                    relation(3 * faces, faces),
                    relation(4 * cells, cells),
                    incidence(2 * edges, vertices, edges),
                    incidence(3 * faces, edges, faces),
                    incidence(4 * cells, faces, cells),
                    chain(2 * edges, 3 * faces, vertices, edges, faces, 2),
                    chain(3 * faces, 4 * cells, edges, faces, cells, 3),
                )
            )
            entity_counts = (edges, faces)
        elif len(shapes) == 1 and kinds[0] in ("hexahedron", "tensor:3"):
            faces, edges = entity_count(2, 6 * cells), entity_count(1, 12 * cells)
            # hexahedral_connectivity holds host rows, range masks, row-sorted
            # cells, local edges and their row-sorted keys, edge runs and
            # routes, face cycles, rotations, permutations, row-sorted face
            # keys, face runs and occurrences and final casts per cell, and
            # face rows, counts, sides and their sorted keys, edge queries,
            # signs and masks per face.
            connectivity = 32 * cells + max(
                55 * cells,
                32 * cells + duplicates(cells, 8, 4),
                1310 * cells
                + 181 * faces
                + 17 * edges
                + vertices
                + max(
                    480 * cells,
                    96 * faces,
                    16 * edges + 32 * faces,
                    rows(12 * cells),
                    rows(6 * cells),
                ),
            )
            # The cell entity set identity lists the sorted cell IDs in Python.
            complex_ = (
                16 * edges
                + 24 * faces
                + 8 * vertices
                + 8 * cells
                + max(
                    entity_set(vertices),
                    entity_set(edges),
                    entity_set(faces),
                    entity_set(cells) + (9 + entry) * cells,
                    relation(2 * edges, edges),
                    relation(4 * faces, faces),
                    relation(6 * cells, cells),
                    incidence(2 * edges, vertices, edges),
                    incidence(4 * faces, edges, faces),
                    incidence(6 * cells, faces, cells),
                    chain(2 * edges, 4 * faces, vertices, edges, faces, 2),
                    chain(4 * faces, 6 * cells, edges, faces, cells, 4),
                )
            )
        else:
            loop_shapes = {
                "tetrahedron": (4, 12),
                "hexahedron": (6, 24),
                "tensor:3": (6, 24),
                "prism": (5, 18),
                "pyramid": (5, 16),
            }
            loops = sum(count * loop_shapes[kind][0] for kind, count, _ in shapes)
            entries = sum(count * loop_shapes[kind][1] for kind, count, _ in shapes)
            faces, edges = entity_count(2, loops), entity_count(1, entries)
            # Every face side is one edge-to-face route.
            face_entries = (
                entries
                if realized is None
                else realized.topology.incidences[1].relation.capacity
            )
            face_width = 3 if all(kind == "tetrahedron" for kind in kinds) else 4
            # variable_row_ranks: each face row stays active for at most its
            # length plus one refinement pass.
            ranked = face_entries + faces
            # Cell-local shell sorts hold only one bounded interval; include
            # rebased loop/entry arguments and the owned full-cell output.
            chunk_cells = min(cells, _OPEN_SHELL_CHUNK_CELLS)
            chunk_loops = min(
                loops, chunk_cells * max(loop_shapes[kind][0] for kind in kinds)
            )
            chunk_entries = min(
                entries, chunk_cells * max(loop_shapes[kind][1] for kind in kinds)
            )
            # Conservative union of named banks, split at actual last uses.
            # Early validation has no face-side/owner/edge output banks.
            # After defects, raw loops and entry maps (9Fl + 20L) are dead.
            # After face extraction and side grouping, canonical loops and
            # side keys/maps (16Fl + 4L + 8F + 28S + 8E) are dead as well.
            banks = (
                4 * corners
                + 28 * cells
                + 81 * loops
                + 28 * entries
                + 85 * faces
                + 69 * face_entries
                + 25 * edges
                + 9 * vertices
            )
            early = banks - 77 * faces - 69 * face_entries - 25 * edges
            middle = banks - 9 * loops - 20 * entries
            late = (
                middle
                - 16 * loops
                - 4 * entries
                - 8 * faces
                - 28 * face_entries
                - 8 * edges
            )
            connectivity = max(
                early
                + max(
                    max(validated(count, arity) for _, count, arity in shapes),
                    16 * cells + 18 * loops + 8 * entries,
                    identifiers(vertices),
                    identifiers(cells),
                    16 * entries,
                    64 * loops + 48 * entries,
                    32 * loops
                    + 12 * entries
                    + max(groups(loops, face_width, 4), groups(loops, 1, 8)),
                    64 * entries + 8 * cells + rows(entries),
                    max(
                        56 * entries + 48 * loops + rows(loops),
                        8 * loops
                        + cells
                        + 32 * chunk_loops
                        + 72 * chunk_entries
                        + 16 * chunk_cells
                        + rows(chunk_entries),
                    ),
                    24 * loops + 24 * cells,
                ),
                middle
                + max(
                    28 * faces,
                    8 * faces + 32 * face_entries,
                    groups(face_entries, 2, 4),
                ),
                late
                + max(
                    16 * loops + 8 * faces + 8 * face_entries,
                    32 * edges + canonical(edges),
                    16 * face_entries + rows(face_entries),
                    180 * faces + rows(ranked),
                    canonical(faces),
                ),
            )
            complex_ = (
                8 * edges
                + 4 * face_entries
                + 8 * faces
                + 4 * loops
                + 8 * cells
                + max(
                    entity_set(vertices),
                    entity_set(edges),
                    entity_set(faces),
                    entity_set(cells) + cells,
                    relation(2 * edges, edges),
                    relation(face_entries, faces),
                    relation(loops, cells),
                    incidence(2 * edges, vertices, edges),
                    incidence(face_entries, edges, faces),
                    incidence(loops, faces, cells),
                    chain(2 * edges, face_entries, vertices, edges, faces, 2),
                    chain(face_entries, loops, edges, faces, cells, face_width),
                )
            )
        route = max(connectivity, complex_)
    else:
        raise ValueError(
            f"cell_mesh_construction_bounds.blocks must have a native topology route; got {kinds} in dimension {dimension}."
        )
    if len(shapes) == 1 and (
        kinds[0] in ("interval", "triangle", "tetrahedron")
        or kinds[0].startswith("simplex:")
    ):
        # Logical simplex arrays: cell-ID order, sorted vertex IDs, global cell
        # rows, and per degree the host entity rows, their row-sorted global
        # keys, lexsort and ordered copies and the ordered entity IDs.
        terms = sum(
            (degree + 1) * count for degree, count in enumerate(entity_counts, start=1)
        )
        identity = (
            28 * cells
            + 8 * vertices
            + 20 * corners
            + 12 * terms
            + 8 * sum(entity_counts)
            + max(
                (
                    (16 * (degree + 1) + 32) * count
                    for degree, count in enumerate(entity_counts, start=1)
                ),
                default=0,
            )
        )
    else:
        # Per block: stable cell-ID order, global vertex rows and their ordered
        # copies with the reordered activity mask.
        identity = 28 * cells + 22 * corners
    if len(shapes) == 1 and kinds[0] in ("interval", "triangle", "tetrahedron"):
        # The geometry identity hashes coordinates in stable vertex-ID order.
        identity = max(identity, 12 * vertices + 8 * points)
    return held + max(*phases, route, identity) + (1 << 20) + 4096 * len(shapes)


def cell_mesh_reuse_bounds(
    vertex_count: int,
    ambient_dimension: int,
    blocks: Sequence[tuple[str, int, int]],
    /,
) -> int:
    """Return the temporary bytes of ``mesh.with_coordinates``.

    The coordinate-only rebuild reuses the prepared blocks, identities,
    connectivity and topology of a plain mesh without storage or periodic
    topology. It holds the host float64 coordinate copy with its finiteness
    mask, or, for one interval, triangle or tetrahedron block whose geometry
    identity hashes coordinates in vertex-ID order, the ID copy, order, sort
    buffer and reordered coordinates.
    """
    vertices = operator.index(vertex_count)
    points = vertices * operator.index(ambient_dimension)
    kinds = tuple(str(kind) for kind, _, _ in blocks)
    if len(kinds) == 1 and kinds[0] in ("interval", "triangle", "tetrahedron"):
        return 8 * points + max(points, 20 * vertices + 8 * points) + (1 << 20)
    return 9 * points + (1 << 20)


__all__ = [
    "CellMesh",
    "CellMeshStorage",
    "SimplicialConnectivity",
]
