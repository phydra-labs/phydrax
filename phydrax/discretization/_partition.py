#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import operator

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._cell_complex import (
    IntervalConnectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ._cell_mesh import CellMesh, CellMeshStorage
from ._hexahedral import HexahedralConnectivity


class CellPartition(StrictModule, NonTrainableState):
    """Solver-neutral exactly-once cell ownership in canonical local cell order."""

    cell_owner: Array
    part_count: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    storage_id: str | None = eqx.field(static=True)
    global_cell_count: int = eqx.field(static=True)

    def __init__(
        self,
        cell_owner: ArrayLike,
        part_count: int,
        /,
        *,
        storage: CellMeshStorage | None = None,
    ) -> None:
        owner = np.asarray(cell_owner)
        count = operator.index(part_count)
        if owner.ndim != 1 or not np.issubdtype(owner.dtype, np.integer):
            raise TypeError("Cell ownership must be an integer vector.")
        if (
            isinstance(part_count, bool)
            or count <= 0
            or np.any(owner < 0)
            or np.any(owner >= count)
        ):
            raise ValueError("Cell ownership or part_count is invalid.")
        if storage is None:
            if np.unique(owner).size != count:
                raise ValueError("Every serial partition must own at least one cell.")
        elif (
            not isinstance(storage, CellMeshStorage)
            or storage.partition_count != count
            or not np.array_equal(owner, np.asarray(storage.entity_owner[-1]))
        ):
            raise ValueError(
                "Owner-local partition ownership must match its canonical storage route."
            )
        owner = owner.astype(np.int32, copy=False)
        self.cell_owner = jnp.asarray(owner)
        self.part_count = count
        self.storage_id = None if storage is None else storage.storage_id
        self.global_cell_count = (
            owner.size if storage is None else storage.global_entity_counts[-1]
        )
        self.partition_id = canonical_fingerprint(
            {
                "kind": "cell-partition",
                "cell_owner": array_tree_fingerprint(owner),
                "part_count": count,
            }
        )
        if storage is not None:
            self.partition_id = canonical_fingerprint(
                {
                    "kind": "owner-local-cell-partition",
                    "cell_owner": array_tree_fingerprint(owner),
                    "part_count": count,
                    "storage": storage.storage_id,
                    "global_cell_count": storage.global_entity_counts[-1],
                }
            )


def partition_cells_contiguous(mesh: CellMesh, part_count: int, /) -> CellPartition:
    """Partition the canonical concatenated block ordering without splitting cells."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    count = sum(block.cell_count for block in mesh.blocks)
    parts = operator.index(part_count)
    if isinstance(part_count, bool) or parts <= 0 or parts > count:
        raise ValueError("part_count must lie between one and the cell count.")
    owner = np.arange(count, dtype=np.int64) * parts // count
    return CellPartition(owner, parts)


class CellAdjacency(StrictModule, NonTrainableState):
    """Symmetric cell graph in canonical CSR order (row, then neighbor index).

    Duplicate and reversed pairs collapse to one undirected edge; self pairs are
    rejected because a cell is never its own face neighbor.
    """

    offsets: Array
    neighbors: Array
    cell_count: int = eqx.field(static=True)
    adjacency_id: str = eqx.field(static=True)

    def __init__(self, pairs: ArrayLike, cell_count: int, /) -> None:
        values = np.asarray(pairs)
        count = operator.index(cell_count)
        if isinstance(cell_count, bool) or count <= 0:
            raise ValueError("Cell adjacency requires a positive cell count.")
        if values.size == 0:
            values = np.zeros((0, 2), dtype=np.int64)
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("Cell adjacency pairs must be integer cell indices.")
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("Cell adjacency pairs must have shape (pairs, 2).")
        if (
            np.any(values < 0)
            or np.any(values >= count)
            or np.any(values[:, 0] == values[:, 1])
        ):
            raise ValueError(
                "Cell adjacency pairs must join two distinct in-range cells."
            )
        directed = np.concatenate((values, values[:, ::-1])).astype(np.int64)
        keys = np.unique(directed[:, 0] * count + directed[:, 1])
        offsets = np.zeros((count + 1,), dtype=np.int64)
        np.cumsum(np.bincount(keys // count, minlength=count), out=offsets[1:])
        neighbors = (keys % count).astype(np.int32)
        self.offsets = jnp.asarray(offsets)
        self.neighbors = jnp.asarray(neighbors)
        self.cell_count = count
        self.adjacency_id = canonical_fingerprint(
            {
                "kind": "cell-adjacency",
                "cell_count": count,
                "offsets": array_tree_fingerprint(offsets),
                "neighbors": array_tree_fingerprint(neighbors),
            }
        )

    def undirected_pairs(self) -> np.ndarray:
        """Return each edge once as ``(lower, upper)`` cell rows in CSR order."""
        offsets = np.asarray(self.offsets)
        rows = np.repeat(np.arange(self.cell_count, dtype=np.int32), np.diff(offsets))
        neighbors = np.asarray(self.neighbors)
        upper = rows < neighbors
        return np.stack((rows[upper], neighbors[upper]), axis=1)


def _expand_adjacency(
    offsets: np.ndarray, neighbors: np.ndarray, cells: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return (query position, neighbor cell) for every CSR entry of ``cells``."""
    starts = offsets[cells]
    degree = offsets[cells + 1] - starts
    ends = np.cumsum(degree)
    entries = np.arange(ends[-1] if ends.size else 0, dtype=np.int64)
    shift = np.repeat(starts - (ends - degree), degree)
    return np.repeat(np.arange(cells.size), degree), neighbors[entries + shift]


def mesh_cell_adjacency(mesh: CellMesh, /) -> CellAdjacency:
    """Facet-sharing adjacency in canonical concatenated block cell order."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    cell_count = sum(block.cell_count for block in mesh.blocks)
    connectivity = mesh.connectivity
    match connectivity:
        case PolyhedralConnectivity():
            owner = np.asarray(connectivity.face_owner, dtype=np.int64)
            neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)
            interior = neighbor >= 0
            return CellAdjacency(
                np.stack((owner[interior], neighbor[interior]), axis=1), cell_count
            )
        case IntervalConnectivity():
            incidence = np.asarray(connectivity.cell_vertices, dtype=np.int64)
            valid = np.ones(incidence.shape, dtype=np.bool_)
        case PolygonalConnectivity():
            incidence = np.asarray(connectivity.cell_edges, dtype=np.int64)
            valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
        case TetrahedralConnectivity() | HexahedralConnectivity():
            incidence = np.asarray(connectivity.cell_faces, dtype=np.int64)
            valid = np.ones(incidence.shape, dtype=np.bool_)
        case _:
            raise TypeError("Unsupported cell connectivity for adjacency.")
    cells = np.broadcast_to(
        np.arange(incidence.shape[0], dtype=np.int64)[:, None], incidence.shape
    )[valid]
    facets = incidence[valid]
    order = np.lexsort((cells, facets))
    facets, cells = facets[order], cells[order]
    shared = facets[1:] == facets[:-1]
    if np.any(shared[1:] & shared[:-1]):
        raise ValueError("A cell facet is shared by more than two cells.")
    return CellAdjacency(
        np.stack((cells[:-1][shared], cells[1:][shared]), axis=1), cell_count
    )


def _validated_cell_ids(value: ArrayLike | None, count: int, /) -> np.ndarray:
    identifiers = np.arange(count, dtype=np.int64) if value is None else np.asarray(value)
    if not np.issubdtype(identifiers.dtype, np.integer):
        raise TypeError("Cell global IDs must be integers.")
    if (
        identifiers.shape != (count,)
        or np.any(identifiers < 0)
        or np.unique(identifiers).size != count
    ):
        raise ValueError("Cell global IDs must be unique, non-negative, one per cell.")
    return identifiers.astype(np.int64, copy=False)


def _grouped_offsets(groups: np.ndarray, group_count: int, /) -> np.ndarray:
    offsets = np.zeros((group_count + 1,), dtype=np.int64)
    np.cumsum(np.bincount(groups, minlength=group_count), out=offsets[1:])
    return offsets


def padded_part_table(offsets: ArrayLike, values: ArrayLike, /) -> np.ndarray:
    """Lay CSR part groups into a ``(parts, largest group)`` table padded with -1.

    The capacity is the largest group, never the global cell count, so balanced
    partitions stay near ``cells / parts`` columns.
    """
    starts = np.asarray(offsets, dtype=np.int64)
    entries = np.asarray(values, dtype=np.int32)
    counts = np.diff(starts)
    if starts.ndim != 1 or entries.shape != (starts[-1],) or np.any(counts < 0):
        raise ValueError("Part groups must be CSR offsets over their values.")
    table = np.full((counts.size, np.max(counts, initial=0)), -1, dtype=np.int32)
    columns = np.arange(entries.size) - np.repeat(starts[:-1], counts)
    table[np.repeat(np.arange(counts.size), counts), columns] = entries
    return table


class CellPartitionHalo(StrictModule, NonTrainableState):
    """Owned cells and ``layers`` face-adjacency ghost layers for every part.

    Owned and ghost rows are grouped by part in CSR form (never a parts x cells
    table) and ordered by stable cell global ID within each part, so replica
    order is deterministic and independent of storage order. ``halo_layers``
    records the graph distance (1..layers) of every ghost from its part.
    ``dependencies[p, q]`` is true when part ``p`` reads a ghost owned by ``q``.
    """

    owned_offsets: Array
    owned_cells: Array
    halo_offsets: Array
    halo_cells: Array
    halo_layers: Array
    dependencies: Array
    part_count: int = eqx.field(static=True)
    layers: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    adjacency_id: str = eqx.field(static=True)
    halo_id: str = eqx.field(static=True)

    def __init__(
        self,
        partition: CellPartition,
        adjacency: CellAdjacency,
        /,
        *,
        layers: int,
        cell_global_ids: ArrayLike | None = None,
    ) -> None:
        if not isinstance(partition, CellPartition) or not isinstance(
            adjacency, CellAdjacency
        ):
            raise TypeError("Partition halos require CellPartition and CellAdjacency.")
        depth = operator.index(layers)
        if isinstance(layers, bool) or depth < 0:
            raise ValueError("Ghost layer count must be a non-negative integer.")
        owner = np.asarray(partition.cell_owner, dtype=np.int64)
        count = owner.size
        if adjacency.cell_count != count:
            raise ValueError("Cell adjacency and partition cover different cells.")
        identifiers = _validated_cell_ids(cell_global_ids, count)
        offsets = np.asarray(adjacency.offsets)
        neighbors = np.asarray(adjacency.neighbors, dtype=np.int64)
        # Breadth-first search over (part, cell) keys for all parts at once; each
        # layer expands the previous frontier through the CSR rows in one pass.
        visited = np.sort(owner * count + np.arange(count, dtype=np.int64))
        frontier = visited
        found: list[np.ndarray] = []
        found_layers: list[np.ndarray] = []
        for layer in range(1, depth + 1):
            position, reached = _expand_adjacency(offsets, neighbors, frontier % count)
            candidates = np.unique((frontier // count)[position] * count + reached)
            frontier = candidates[
                ~np.isin(candidates, visited, assume_unique=True, kind="sort")
            ]
            if frontier.size == 0:
                break
            visited = np.union1d(visited, frontier)
            found.append(frontier)
            found_layers.append(np.full(frontier.shape, layer, dtype=np.int32))
        halo_keys = np.concatenate(found) if found else np.zeros((0,), np.int64)
        halo_depth = np.concatenate(found_layers) if found else np.zeros((0,), np.int32)
        halo_parts, halo_cells = halo_keys // count, halo_keys % count
        order = np.lexsort((identifiers[halo_cells], halo_parts))
        halo_parts, halo_cells = halo_parts[order], halo_cells[order]
        halo_depth = halo_depth[order]
        owned_cells = np.lexsort((identifiers, owner))
        parts = partition.part_count
        dependencies = np.zeros((parts, parts), dtype=np.bool_)
        dependencies[halo_parts, owner[halo_cells]] = True
        owned_offsets = _grouped_offsets(owner, parts)
        halo_offsets = _grouped_offsets(halo_parts, parts)
        self.owned_offsets = jnp.asarray(owned_offsets)
        self.owned_cells = jnp.asarray(owned_cells.astype(np.int32))
        self.halo_offsets = jnp.asarray(halo_offsets)
        self.halo_cells = jnp.asarray(halo_cells.astype(np.int32))
        self.halo_layers = jnp.asarray(halo_depth)
        self.dependencies = jnp.asarray(dependencies)
        self.part_count = parts
        self.layers = depth
        self.partition_id = partition.partition_id
        self.adjacency_id = adjacency.adjacency_id
        self.halo_id = canonical_fingerprint(
            {
                "kind": "cell-partition-halo",
                "partition": partition.partition_id,
                "adjacency": adjacency.adjacency_id,
                "cell_ids": array_tree_fingerprint(identifiers),
                "layers": depth,
                "halo_cells": array_tree_fingerprint(halo_cells),
                "halo_offsets": array_tree_fingerprint(halo_offsets),
            }
        )

    def _part(self, part: int, /) -> int:
        index = operator.index(part)
        if not 0 <= index < self.part_count:
            raise ValueError("Partition index is out of range.")
        return index

    def owned(self, part: int, /) -> Array:
        index = self._part(part)
        offsets = np.asarray(self.owned_offsets)
        return self.owned_cells[offsets[index] : offsets[index + 1]]

    def halo(self, part: int, /) -> Array:
        index = self._part(part)
        offsets = np.asarray(self.halo_offsets)
        return self.halo_cells[offsets[index] : offsets[index + 1]]


def inherit_cell_owners(
    source_owner: ArrayLike,
    source_rows: ArrayLike,
    target_rows: ArrayLike,
    target_count: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Inherit ownership through lineage routes ``source_rows -> target_rows``.

    Returns the majority inherited owner per target (ties resolve to the lowest
    part, -1 without routes) and the number of distinct inherited owners, so
    callers decide whether merged cells may straddle parts.
    """
    owners_by_source = np.asarray(source_owner, dtype=np.int64)
    sources = np.asarray(source_rows, dtype=np.int64)
    targets = np.asarray(target_rows, dtype=np.int64)
    count = operator.index(target_count)
    if (
        owners_by_source.ndim != 1
        or sources.ndim != 1
        or targets.shape != sources.shape
        or count < 0
        or np.any(sources < 0)
        or np.any(sources >= owners_by_source.size)
        or np.any(targets < 0)
        or np.any(targets >= count)
    ):
        raise ValueError("Ownership inheritance routes are out of range.")
    inherited = owners_by_source[sources]
    if np.any(inherited < 0):
        raise ValueError("Ownership inheritance routes read an unowned source cell.")
    owner = np.full((count,), -1, dtype=np.int32)
    if targets.size == 0:
        return owner, np.zeros((count,), dtype=np.int32)
    pairs, multiplicity = np.unique(
        np.stack((targets, inherited), axis=1), axis=0, return_counts=True
    )
    distinct = np.bincount(pairs[:, 0], minlength=count).astype(np.int32)
    order = np.lexsort((pairs[:, 1], -multiplicity, pairs[:, 0]))
    ranked = pairs[order]
    first = np.concatenate(([True], ranked[1:, 0] != ranked[:-1, 0]))
    owner[ranked[first, 0]] = ranked[first, 1]
    return owner, distinct


__all__ = [
    "CellAdjacency",
    "CellPartition",
    "CellPartitionHalo",
    "inherit_cell_owners",
    "mesh_cell_adjacency",
    "padded_part_table",
    "partition_cells_contiguous",
]
