#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Topology-neutral host banks retained for explicit migration replay."""

from __future__ import annotations

from typing import final, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np

import phydrax.meshing._bisection as bisection_records

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_mesh import CellMesh
from ..discretization._reference_cell_types import CellBlock
from ..typing import checked, Dim, HostBool, HostFloat64, HostInt64, validate


if TYPE_CHECKING:
    from . import (
        _bisection_kernel_host as bisection_host,
        _device_adaptation as adaptation_owner,
    )


class MigrationCellDim(Dim):
    """Rows of one active cell bank."""


class MigrationWidthDim(Dim):
    """Corners of one simplex bank."""


class MigrationRecordDim(Dim):
    """Rows of one binary ancestry bank."""


class MigrationGrowthDim(Dim):
    """Issued compact vertices of one growth bank."""


class MigrationSplitDim(Dim):
    """Known split-edge keys and their compact vertex indices."""


class MigrationVertexDim(Dim):
    """Vertices of the sorted source bank."""


class MigrationAmbientDim(Dim):
    """Coordinate components of the source bank."""


class MigrationMarkDim(Dim):
    """Selected front-cell indices."""


class MigrationRetiredDim(Dim):
    """Rows of one retired intermediate-entity bank."""


class MigrationEntityWidthDim(Dim):
    """Corners of one intermediate entity."""


class MigrationSlotDim(Dim):
    """Slots of the source ancestry forest."""


class MigrationAnchorSlotDim(Dim):
    """Forest slots belonging to the epoch anchor."""


class MigrationBlockDim(Dim):
    """Raw presentation blocks mapped to resident blocks."""


def _nonnegative(values: np.ndarray, name: str, /) -> None:
    if np.any(values < 0):
        raise ValueError(
            f"{name} must contain nonnegative values; got minimum {values.min(initial=0)!r}"
        )


def _unique_ids(values: np.ndarray, name: str, /) -> None:
    _nonnegative(values, name)
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} must contain unique IDs; got {values!r}")


def _simplex_values(
    rows: np.ndarray, tuples: np.ndarray, tags: np.ndarray, name: str, /
) -> None:
    width = rows.shape[1]
    if width not in (3, 4):
        raise ValueError(f"{name}.rows must have three or four corners; got {width!r}")
    _nonnegative(rows, f"{name}.rows")
    ordered = np.sort(rows, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]) or not np.array_equal(
        ordered, np.sort(tuples, axis=1)
    ):
        raise ValueError(
            f"{name}.tuples must retain distinct row vertices; got {(rows, tuples)!r}"
        )
    # Unlabeled scientific source cells carry zero tags.
    if np.any((tags < 0) | (tags >= width)):
        raise ValueError(
            f"{name}.tags must lie in zero through the simplex dimension; got {tags!r}"
        )


def _cell_values(
    ids: np.ndarray,
    rows: np.ndarray,
    tuples: np.ndarray,
    tags: np.ndarray,
    blocks: np.ndarray,
    origins: np.ndarray,
    generations: np.ndarray,
    /,
) -> None:
    _unique_ids(ids, "MigrationCells.ids")
    _simplex_values(rows, tuples, tags, "MigrationCells")
    _nonnegative(blocks, "MigrationCells.blocks")
    _nonnegative(origins, "MigrationCells.origins")
    _nonnegative(generations, "MigrationCells.generations")


@final
class MigrationCells(StrictModule, NonTrainableState):
    """Explicit active-cell labels and source lineage."""

    __strict_contract__ = True

    ids: HostInt64[MigrationCellDim]
    rows: HostInt64[MigrationCellDim, MigrationWidthDim]
    tuples: HostInt64[MigrationCellDim, MigrationWidthDim]
    tags: HostInt64[MigrationCellDim]
    blocks: HostInt64[MigrationCellDim]
    origins: HostInt64[MigrationCellDim]
    generations: HostInt64[MigrationCellDim]

    @checked
    def __init__(
        self,
        ids: HostInt64[MigrationCellDim],
        rows: HostInt64[MigrationCellDim, MigrationWidthDim],
        tuples: HostInt64[MigrationCellDim, MigrationWidthDim],
        tags: HostInt64[MigrationCellDim],
        blocks: HostInt64[MigrationCellDim],
        origins: HostInt64[MigrationCellDim],
        generations: HostInt64[MigrationCellDim],
        /,
    ) -> None:
        _cell_values(ids, rows, tuples, tags, blocks, origins, generations)
        self.ids = ids
        self.rows = rows
        self.tuples = tuples
        self.tags = tags
        self.blocks = blocks
        self.origins = origins
        self.generations = generations

    @classmethod
    def from_live(cls, value: bisection_host.Cells, /) -> MigrationCells:
        return cls(*value)

    def to_live(self) -> bisection_host.Cells:
        from ._bisection_kernel_host import Cells

        return Cells(
            self.ids,
            self.rows,
            self.tuples,
            self.tags,
            self.blocks,
            self.origins,
            self.generations,
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: MigrationCells, /) -> MigrationCells:
        return cls(
            value.ids,
            value.rows,
            value.tuples,
            value.tags,
            value.blocks,
            value.origins,
            value.generations,
        )

    def _validate_values(self) -> None:
        _cell_values(
            self.ids,
            self.rows,
            self.tuples,
            self.tags,
            self.blocks,
            self.origins,
            self.generations,
        )

    def validate_restored(self) -> None:
        validate(self)
        self._validate_values()


def _record_values(
    parent_ids: np.ndarray,
    blocks: np.ndarray,
    rows: np.ndarray,
    tuples: np.ndarray,
    tags: np.ndarray,
    child_ids: np.ndarray,
    vertices: np.ndarray,
    /,
) -> None:
    _unique_ids(parent_ids, "MigrationRecords.parent_ids")
    _simplex_values(rows, tuples, tags, "MigrationRecords")
    _nonnegative(blocks, "MigrationRecords.blocks")
    _nonnegative(vertices, "MigrationRecords.vertices")
    _unique_ids(child_ids, "MigrationRecords.child_ids")
    if np.any(tags == 0) or np.any(child_ids == parent_ids[:, None]):
        raise ValueError(
            "MigrationRecords.child_ids must descend from positively tagged parents; "
            f"got {(parent_ids, tags, child_ids)!r}"
        )


@final
class MigrationRecords(StrictModule, NonTrainableState):
    """Explicit binary bisection ancestry, including empty forests."""

    __strict_contract__ = True

    parent_ids: HostInt64[MigrationRecordDim]
    blocks: HostInt64[MigrationRecordDim]
    rows: HostInt64[MigrationRecordDim, MigrationWidthDim]
    tuples: HostInt64[MigrationRecordDim, MigrationWidthDim]
    tags: HostInt64[MigrationRecordDim]
    child_ids: HostInt64[MigrationRecordDim, Literal[2]]
    vertices: HostInt64[MigrationRecordDim]

    @checked
    def __init__(
        self,
        parent_ids: HostInt64[MigrationRecordDim],
        blocks: HostInt64[MigrationRecordDim],
        rows: HostInt64[MigrationRecordDim, MigrationWidthDim],
        tuples: HostInt64[MigrationRecordDim, MigrationWidthDim],
        tags: HostInt64[MigrationRecordDim],
        child_ids: HostInt64[MigrationRecordDim, Literal[2]],
        vertices: HostInt64[MigrationRecordDim],
        /,
    ) -> None:
        _record_values(parent_ids, blocks, rows, tuples, tags, child_ids, vertices)
        self.parent_ids = parent_ids
        self.blocks = blocks
        self.rows = rows
        self.tuples = tuples
        self.tags = tags
        self.child_ids = child_ids
        self.vertices = vertices

    @classmethod
    def from_live(cls, value: bisection_host.Records, /) -> MigrationRecords:
        return cls(*value)

    def to_live(self) -> bisection_host.Records:
        from ._bisection_kernel_host import Records

        return Records(
            self.parent_ids,
            self.blocks,
            self.rows,
            self.tuples,
            self.tags,
            self.child_ids,
            self.vertices,
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: MigrationRecords, /) -> MigrationRecords:
        return cls(
            value.parent_ids,
            value.blocks,
            value.rows,
            value.tuples,
            value.tags,
            value.child_ids,
            value.vertices,
        )

    def _validate_values(self) -> None:
        _record_values(
            self.parent_ids,
            self.blocks,
            self.rows,
            self.tuples,
            self.tags,
            self.child_ids,
            self.vertices,
        )

    def validate_restored(self) -> None:
        validate(self)
        self._validate_values()


def _growth_values(
    parents: np.ndarray, weights: np.ndarray, levels: np.ndarray, /
) -> None:
    if parents.shape[1] not in (3, 4) or np.any(parents < -1):
        raise ValueError(
            "MigrationGrowth.parents must hold three or four source corners padded by -1; "
            f"got {(parents.shape, parents.min(initial=0))!r}"
        )
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(weights[parents < 0] != 0.0)
        or np.any(weights[parents >= 0] <= 0.0)
        or not np.all(np.sum(weights, axis=1) == 1.0)
        or np.any(np.sum(parents >= 0, axis=1) < 2)
    ):
        raise ValueError(
            "MigrationGrowth.weights must contain finite normalized construction weights "
            f"on at least two parent corners; got {(parents, weights)!r}"
        )
    ordered = np.sort(parents, axis=1)
    if np.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)):
        raise ValueError(
            f"MigrationGrowth.parents must contain distinct supported corners; got {parents!r}"
        )
    _nonnegative(levels, "MigrationGrowth.levels")


@final
class MigrationGrowth(StrictModule, NonTrainableState):
    """Padded parent corners and numerical construction weights."""

    __strict_contract__ = True

    parents: HostInt64[MigrationGrowthDim, MigrationWidthDim]
    weights: HostFloat64[MigrationGrowthDim, MigrationWidthDim]
    levels: HostInt64[MigrationGrowthDim]

    @checked
    def __init__(
        self,
        parents: HostInt64[MigrationGrowthDim, MigrationWidthDim],
        weights: HostFloat64[MigrationGrowthDim, MigrationWidthDim],
        levels: HostInt64[MigrationGrowthDim],
        /,
    ) -> None:
        _growth_values(parents, weights, levels)
        self.parents = parents
        self.weights = weights
        self.levels = levels

    @classmethod
    def from_live(cls, value: bisection_host.Growth, /) -> MigrationGrowth:
        return cls(*value)

    def to_live(self) -> bisection_host.Growth:
        from ._bisection_kernel_host import Growth

        return Growth(self.parents, self.weights, self.levels)

    @classmethod
    @checked
    def from_stored_fields(cls, value: MigrationGrowth, /) -> MigrationGrowth:
        return cls(value.parents, value.weights, value.levels)

    def _validate_values(self) -> None:
        _growth_values(self.parents, self.weights, self.levels)

    def validate_restored(self) -> None:
        validate(self)
        self._validate_values()


def _counter(value: int, name: str, /) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python integer; got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative; got {value!r}")


@checked
def _front_values(
    cells: MigrationCells,
    growth: MigrationGrowth,
    split_keys: HostInt64[MigrationSplitDim] | HostInt64[Literal[0], Literal[2]],
    split_vertices: HostInt64[MigrationSplitDim],
    vertex_base: int,
    next_cell: int,
    /,
) -> None:
    cells._validate_values()
    growth._validate_values()
    _counter(vertex_base, "MigrationFront.vertex_base")
    _counter(next_cell, "MigrationFront.next_cell")
    if cells.rows.shape[1] != growth.parents.shape[1]:
        raise ValueError(
            "MigrationFront.growth must have the cell simplex width; "
            f"got {(cells.rows.shape, growth.parents.shape)!r}"
        )
    vertex_count = vertex_base + growth.levels.size
    if np.any(cells.rows >= vertex_count) or np.any(cells.ids >= next_cell):
        raise ValueError(
            "MigrationFront.cells must lie below the vertex and cell high-water marks; "
            f"got {(vertex_count, next_cell)!r}"
        )
    # A construction can reference only vertices issued before its own row.
    if np.any(
        growth.parents
        >= vertex_base + np.arange(growth.levels.size, dtype=np.int64)[:, None]
    ):
        raise ValueError(
            "MigrationFront.growth must reference preceding compact vertices; "
            f"got {(vertex_base, growth.parents)!r}"
        )
    if split_keys.shape[0] != split_vertices.size:
        raise ValueError(
            "MigrationFront.split_vertices must align with split keys; "
            f"got {(split_keys.shape, split_vertices.shape)!r}"
        )
    _nonnegative(split_keys, "MigrationFront.split_keys")
    if split_keys.ndim == 1 and np.any(split_keys[1:] <= split_keys[:-1]):
        raise ValueError(
            f"MigrationFront.split_keys must be strictly increasing; got {split_keys!r}"
        )
    if np.any((split_vertices < vertex_base) | (split_vertices >= vertex_count)):
        raise ValueError(
            "MigrationFront.split_vertices must identify issued growth vertices; "
            f"got {(vertex_base, vertex_count, split_vertices)!r}"
        )
    if split_keys.ndim == 1 and split_keys.size:
        first, second = split_keys // np.int64(1 << 31), split_keys % np.int64(1 << 31)
        if np.any(first >= second) or np.any(second >= vertex_count):
            raise ValueError(
                "MigrationFront.split_keys must encode distinct issued compact endpoints; "
                f"got {(split_keys, vertex_count)!r}"
            )


@final
class MigrationFront(StrictModule, NonTrainableState):
    """Cell and vertex-growth banks with their issuance high-water marks."""

    __strict_contract__ = True

    cells: MigrationCells
    growth: MigrationGrowth
    split_keys: HostInt64[MigrationSplitDim] | HostInt64[Literal[0], Literal[2]]
    split_vertices: HostInt64[MigrationSplitDim]
    vertex_base: int = eqx.field(static=True)
    next_cell: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        cells: MigrationCells,
        growth: MigrationGrowth,
        split_keys: HostInt64[MigrationSplitDim] | HostInt64[Literal[0], Literal[2]],
        split_vertices: HostInt64[MigrationSplitDim],
        vertex_base: int,
        next_cell: int,
        /,
    ) -> None:
        _front_values(cells, growth, split_keys, split_vertices, vertex_base, next_cell)
        self.cells = cells
        self.growth = growth
        self.split_keys = split_keys
        self.split_vertices = split_vertices
        self.vertex_base = vertex_base
        self.next_cell = next_cell

    @classmethod
    def from_live(cls, value: bisection_host.Front, /) -> MigrationFront:
        return cls(
            MigrationCells.from_live(value.cells),
            MigrationGrowth.from_live(value.growth),
            value.split_keys,
            value.split_vertices,
            value.vertex_base,
            value.next_cell,
        )

    def to_live(self) -> bisection_host.Front:
        from ._bisection_kernel_host import Front

        return Front(
            self.cells.to_live(),
            self.growth.to_live(),
            self.split_keys,
            self.split_vertices,
            self.vertex_base,
            self.next_cell,
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: MigrationFront, /) -> MigrationFront:
        return cls(
            MigrationCells.from_stored_fields(value.cells),
            MigrationGrowth.from_stored_fields(value.growth),
            value.split_keys,
            value.split_vertices,
            value.vertex_base,
            value.next_cell,
        )

    def _validate_values(self) -> None:
        _front_values(
            self.cells,
            self.growth,
            self.split_keys,
            self.split_vertices,
            self.vertex_base,
            self.next_cell,
        )

    def validate_restored(self) -> None:
        validate(self)
        self._validate_values()


def _retired_values(keys: np.ndarray, ids: np.ndarray, /) -> None:
    if (
        keys.shape[1] not in (2, 3)
        or np.any(keys < 0)
        or np.any(keys[:, 1:] <= keys[:, :-1])
    ):
        raise ValueError(
            "MigrationRetiredEntities.keys must contain ascending edge or face corners; "
            f"got {keys!r}"
        )
    _unique_ids(ids, "MigrationRetiredEntities.ids")
    if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
        raise ValueError(f"MigrationRetiredEntities.keys must be unique; got {keys!r}")


@final
class MigrationRetiredEntities(StrictModule, NonTrainableState):
    """One intermediate-dimensional retired key and identity bank."""

    __strict_contract__ = True

    keys: HostInt64[MigrationRetiredDim, MigrationEntityWidthDim]
    ids: HostInt64[MigrationRetiredDim]

    @checked
    def __init__(
        self,
        keys: HostInt64[MigrationRetiredDim, MigrationEntityWidthDim],
        ids: HostInt64[MigrationRetiredDim],
        /,
    ) -> None:
        _retired_values(keys, ids)
        self.keys = keys
        self.ids = ids

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: MigrationRetiredEntities, /
    ) -> MigrationRetiredEntities:
        return cls(value.keys, value.ids)

    def _validate_values(self) -> None:
        _retired_values(self.keys, self.ids)

    def validate_restored(self) -> None:
        validate(self)
        self._validate_values()


@checked
def _source_values(
    mesh: CellMesh,
    dimension: int,
    vertex_ids: HostInt64[MigrationVertexDim],
    original: HostInt64[MigrationVertexDim],
    coordinates: HostFloat64[MigrationVertexDim, MigrationAmbientDim],
    cells: MigrationCells,
    /,
) -> None:
    cells._validate_values()
    if type(dimension) is not int or dimension not in (2, 3):
        raise ValueError(
            f"MigrationSource.dimension must be two or three; got {dimension!r}"
        )
    if mesh.topological_dimension != dimension or cells.rows.shape[1] != dimension + 1:
        raise ValueError(
            "MigrationSource.dimension must agree with its mesh and simplex banks; "
            f"got {(dimension, mesh.topological_dimension, cells.rows.shape)!r}"
        )
    _unique_ids(vertex_ids, "MigrationSource.vertex_ids")
    if np.any(vertex_ids[1:] <= vertex_ids[:-1]):
        raise ValueError(
            f"MigrationSource.vertex_ids must be strictly increasing; got {vertex_ids!r}"
        )
    if not np.array_equal(np.sort(original), np.arange(vertex_ids.size, dtype=np.int64)):
        raise ValueError(
            f"MigrationSource.original must permute source vertices; got {original!r}"
        )
    mesh_ids, mesh_coordinates, mesh_blocks = jax.device_get(
        (
            mesh.vertex_global_ids,
            mesh.coordinates,
            tuple((block.global_ids, block.vertices) for block in mesh.blocks),
        )
    )
    if (
        np.asarray(mesh_ids).shape != vertex_ids.shape
        or np.asarray(mesh_coordinates).shape != coordinates.shape
    ):
        raise ValueError(
            "MigrationSource.mesh must agree with source vertex and coordinate extents; "
            f"got {(np.asarray(mesh_ids).shape, np.asarray(mesh_coordinates).shape)!r}"
        )
    if not np.array_equal(
        np.asarray(mesh_ids)[original], vertex_ids
    ) or not np.array_equal(np.asarray(mesh_coordinates)[original], coordinates):
        raise ValueError(
            "MigrationSource.coordinates must retain the mesh's sorted vertex bank; "
            f"got {(vertex_ids.shape, coordinates.shape)!r}"
        )
    if not np.all(np.isfinite(coordinates)) or coordinates.shape[1] < dimension:
        raise ValueError(
            "MigrationSource.coordinates must contain finite points of at least the simplex dimension; "
            f"got {(coordinates.shape, dimension)!r}"
        )
    if np.any(cells.rows >= vertex_ids.size) or np.any(cells.blocks >= len(mesh.blocks)):
        raise ValueError(
            "MigrationSource.cells must reference source vertices and blocks; "
            f"got {(vertex_ids.size, len(mesh.blocks))!r}"
        )
    compact = np.empty_like(original)
    compact[original] = np.arange(original.size, dtype=np.int64)
    for block_index, (identifiers, vertices) in enumerate(mesh_blocks):
        selected = np.flatnonzero(cells.blocks == block_index)
        if selected.size == 0:
            continue
        block_ids = np.asarray(identifiers)
        order = np.argsort(block_ids, kind="stable")
        positions = np.searchsorted(block_ids[order], cells.ids[selected])
        if np.any(positions >= block_ids.size):
            raise ValueError(
                "MigrationSource.cells.ids must belong to their declared source blocks; "
                f"got {cells.ids[selected]!r}"
            )
        rows = order[positions]
        if not np.array_equal(block_ids[rows], cells.ids[selected]) or not np.array_equal(
            compact[np.asarray(vertices)[rows]], cells.rows[selected]
        ):
            raise ValueError(
                "MigrationSource.cells.rows must retain source block incidence and IDs; "
                f"got {(block_index, cells.ids[selected])!r}"
            )


@final
class MigrationSource(StrictModule, NonTrainableState):
    """Scientific source mesh and its sorted compact host presentation."""

    __strict_contract__ = True

    mesh: CellMesh
    dimension: int = eqx.field(static=True)
    vertex_ids: HostInt64[MigrationVertexDim]
    original: HostInt64[MigrationVertexDim]
    coordinates: HostFloat64[MigrationVertexDim, MigrationAmbientDim]
    cells: MigrationCells

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        dimension: int,
        vertex_ids: HostInt64[MigrationVertexDim],
        original: HostInt64[MigrationVertexDim],
        coordinates: HostFloat64[MigrationVertexDim, MigrationAmbientDim],
        cells: MigrationCells,
        /,
    ) -> None:
        _source_values(mesh, dimension, vertex_ids, original, coordinates, cells)
        self.mesh = mesh
        self.dimension = dimension
        self.vertex_ids = vertex_ids
        self.original = original
        self.coordinates = coordinates
        self.cells = cells

    @classmethod
    def from_live(cls, value: bisection_host.Source, /) -> MigrationSource:
        return cls(
            value.mesh,
            value.dimension,
            value.vertex_ids,
            value.original,
            value.coordinates,
            MigrationCells.from_live(value.cells),
        )

    def to_live(self) -> bisection_host.Source:
        from ._bisection_kernel_host import Source

        return Source(
            self.mesh,
            self.dimension,
            self.vertex_ids,
            self.original,
            self.coordinates,
            self.cells.to_live(),
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: MigrationSource, /) -> MigrationSource:
        return cls(
            value.mesh,
            value.dimension,
            value.vertex_ids,
            value.original,
            value.coordinates,
            MigrationCells.from_stored_fields(value.cells),
        )

    def _validate_values(self) -> None:
        _source_values(
            self.mesh,
            self.dimension,
            self.vertex_ids,
            self.original,
            self.coordinates,
            self.cells,
        )

    def validate_restored(self) -> None:
        validate(self)
        self._validate_values()


@checked
def _start_values(
    front: MigrationFront,
    records: MigrationRecords,
    marks: HostInt64[MigrationMarkDim] | HostBool[MigrationCellDim],
    next_vertex: int,
    compatible: bool | None,
    incompatible: int,
    uniform: bool,
    retired: tuple[MigrationRetiredEntities, ...],
    uniform_refinement: bisection_records.BisectionUniformRefinement | None,
    /,
) -> None:
    front._validate_values()
    records._validate_values()
    _counter(next_vertex, "MigrationStart.next_vertex")
    _counter(incompatible, "MigrationStart.incompatible")
    if type(uniform) is not bool or (
        compatible is not None and type(compatible) is not bool
    ):
        raise TypeError(
            "MigrationStart.compatible must be bool or None and uniform must be bool; "
            f"got {(compatible, uniform)!r}"
        )
    width = front.cells.rows.shape[1]
    if records.rows.shape[1] != width:
        raise ValueError(
            "MigrationStart.records must have the front simplex width; "
            f"got {(records.rows.shape, width)!r}"
        )
    if marks.dtype == np.dtype(np.bool_):
        if marks.shape != front.cells.ids.shape:
            raise ValueError(
                "MigrationStart.marks must align boolean marks with front cells; "
                f"got {(marks.shape, front.cells.ids.shape)!r}"
            )
    elif (
        np.any((marks < 0) | (marks >= front.cells.ids.size))
        or np.unique(marks).size != marks.size
    ):
        raise ValueError(
            f"MigrationStart.marks must select distinct front-cell indices; got {marks!r}"
        )
    if len(retired) != width - 2:
        raise ValueError(
            "MigrationStart.retired must contain one bank per intermediate dimension; "
            f"got {(len(retired), width - 2)!r}"
        )
    for degree, bank in enumerate(retired, start=1):
        _retired_binding(bank, degree + 1, front.vertex_base)
    if uniform and uniform_refinement is None:
        raise ValueError(
            "MigrationStart.uniform_refinement must retain uniform preparation metadata; got None"
        )
    if uniform_refinement is not None:
        _uniform_values(uniform_refinement, width - 1)


@checked
def _retired_binding(
    bank: MigrationRetiredEntities, width: int, vertex_base: int, /
) -> None:
    bank._validate_values()
    if bank.keys.shape[1] != width or np.any(bank.keys >= vertex_base):
        raise ValueError(
            "MigrationStart.retired must bind ordered source-vertex entity banks; "
            f"got {(width, bank.keys.shape, vertex_base)!r}"
        )


@checked
def _uniform_values(
    value: bisection_records.BisectionUniformRefinement, dimension: int, /
) -> None:
    from ._bisection import _prepare_uniform_refinement

    if value.dimension != dimension:
        raise ValueError(
            "MigrationStart.uniform_refinement must have the front simplex dimension; "
            f"got {(value.dimension, dimension)!r}"
        )
    _prepare_uniform_refinement(
        value.dimension, *value.host_arrays().values(), source=value.source
    )


@final
class MigrationStart(StrictModule, NonTrainableState):
    """Prepared front, binary ancestry, marks, and compatibility metadata."""

    __strict_contract__ = True

    front: MigrationFront
    records: MigrationRecords
    marks: HostInt64[MigrationMarkDim] | HostBool[MigrationCellDim]
    next_vertex: int = eqx.field(static=True)
    compatible: bool | None = eqx.field(static=True)
    incompatible: int = eqx.field(static=True)
    uniform: bool = eqx.field(static=True)
    retired: tuple[MigrationRetiredEntities, ...]
    uniform_refinement: bisection_records.BisectionUniformRefinement | None

    @checked
    def __init__(
        self,
        front: MigrationFront,
        records: MigrationRecords,
        marks: HostInt64[MigrationMarkDim] | HostBool[MigrationCellDim],
        next_vertex: int,
        compatible: bool | None,
        incompatible: int,
        uniform: bool,
        retired: tuple[MigrationRetiredEntities, ...],
        uniform_refinement: bisection_records.BisectionUniformRefinement | None,
        /,
    ) -> None:
        _start_values(
            front,
            records,
            marks,
            next_vertex,
            compatible,
            incompatible,
            uniform,
            retired,
            uniform_refinement,
        )
        self.front = front
        self.records = records
        self.marks = marks
        self.next_vertex = next_vertex
        self.compatible = compatible
        self.incompatible = incompatible
        self.uniform = uniform
        self.retired = retired
        self.uniform_refinement = uniform_refinement

    @classmethod
    def from_live(cls, value: bisection_host.Start, /) -> MigrationStart:
        return cls(
            MigrationFront.from_live(value.front),
            MigrationRecords.from_live(value.records),
            value.marks,
            value.next_vertex,
            value.compatible,
            value.incompatible,
            value.uniform,
            tuple(MigrationRetiredEntities(*bank) for bank in value.retired),
            value.uniform_refinement,
        )

    def to_live(self) -> bisection_host.Start:
        from ._bisection_kernel_host import Start

        return Start(
            self.front.to_live(),
            self.records.to_live(),
            self.marks,
            self.next_vertex,
            self.compatible,
            self.incompatible,
            self.uniform,
            tuple((bank.keys, bank.ids) for bank in self.retired),
            self.uniform_refinement,
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: MigrationStart, /) -> MigrationStart:
        return cls(
            MigrationFront.from_stored_fields(value.front),
            MigrationRecords.from_stored_fields(value.records),
            value.marks,
            value.next_vertex,
            value.compatible,
            value.incompatible,
            value.uniform,
            tuple(
                MigrationRetiredEntities.from_stored_fields(bank)
                for bank in value.retired
            ),
            value.uniform_refinement,
        )

    def _validate_values(self) -> None:
        _start_values(
            self.front,
            self.records,
            self.marks,
            self.next_vertex,
            self.compatible,
            self.incompatible,
            self.uniform,
            self.retired,
            self.uniform_refinement,
        )

    def validate_restored(self) -> None:
        validate(self)
        self._validate_values()


@checked
def _anchor_values(
    source: MigrationSource,
    start: MigrationStart,
    cell_ids: HostInt64[MigrationSlotDim],
    slot_origins: HostInt64[MigrationSlotDim],
    root_origins: HostInt64[MigrationSlotDim],
    anchor_slots: HostInt64[MigrationAnchorSlotDim],
    block_indices: HostInt64[MigrationBlockDim] | None,
    source_blocks: tuple[CellBlock, ...] | None,
    /,
) -> None:
    source._validate_values()
    start._validate_values()
    _unique_ids(cell_ids, "MigrationAnchor.cell_ids")
    if np.any(slot_origins < -1) or np.any(root_origins < -1):
        raise ValueError(
            "MigrationAnchor.slot_origins must use nonnegative IDs or -1 sentinels; "
            f"got {(slot_origins.min(initial=0), root_origins.min(initial=0))!r}"
        )
    _unique_ids(anchor_slots, "MigrationAnchor.anchor_slots")
    if np.any(anchor_slots >= cell_ids.size):
        raise ValueError(
            "MigrationAnchor.anchor_slots must index the source forest; "
            f"got {(anchor_slots, cell_ids.size)!r}"
        )
    if (
        start.front.vertex_base != source.vertex_ids.size
        or start.front.cells.rows.shape[1] != source.dimension + 1
    ):
        raise ValueError(
            "MigrationAnchor.start must bind the source vertex bank and simplex dimension; "
            f"got {(start.front.vertex_base, source.vertex_ids.size, source.dimension)!r}"
        )
    if np.any(start.records.rows >= start.front.vertex_base) or np.any(
        start.records.vertices >= start.front.vertex_base
    ):
        raise ValueError(
            "MigrationAnchor.start.records must reference retained source vertices; "
            f"got {start.front.vertex_base!r}"
        )
    if np.any(start.records.blocks >= len(source.mesh.blocks)):
        raise ValueError(
            "MigrationAnchor.start.records.blocks must reference resident source blocks; "
            f"got {len(source.mesh.blocks)!r}"
        )
    if source.vertex_ids.size and start.next_vertex <= source.vertex_ids[-1]:
        raise ValueError(
            "MigrationAnchor.start.next_vertex must exceed source vertex IDs; "
            f"got {(start.next_vertex, source.vertex_ids[-1])!r}"
        )
    if block_indices is not None and np.any(
        (block_indices < -1) | (block_indices >= len(source.mesh.blocks))
    ):
        raise ValueError(
            "MigrationAnchor.block_indices must map raw blocks to resident blocks or -1; "
            f"got {block_indices!r}"
        )
    if source_blocks is not None:
        for block in source_blocks:
            _source_block_values(block, source.dimension)


@checked
def _source_block_values(block: CellBlock, dimension: int, /) -> None:
    if block.topological_dimension != dimension:
        raise ValueError(
            "MigrationAnchor.source_blocks must retain the source simplex dimension; "
            f"got {(block.cell_kind, dimension)!r}"
        )


@final
class MigrationAnchor(StrictModule, NonTrainableState):
    """Portable source and start banks for replay into a chosen topology."""

    __strict_contract__ = True

    source: MigrationSource
    start: MigrationStart
    cell_ids: HostInt64[MigrationSlotDim]
    slot_origins: HostInt64[MigrationSlotDim]
    root_origins: HostInt64[MigrationSlotDim]
    anchor_slots: HostInt64[MigrationAnchorSlotDim]
    block_indices: HostInt64[MigrationBlockDim] | None
    source_blocks: tuple[CellBlock, ...] | None

    @checked
    def __init__(
        self,
        source: MigrationSource,
        start: MigrationStart,
        cell_ids: HostInt64[MigrationSlotDim],
        slot_origins: HostInt64[MigrationSlotDim],
        root_origins: HostInt64[MigrationSlotDim],
        anchor_slots: HostInt64[MigrationAnchorSlotDim],
        block_indices: HostInt64[MigrationBlockDim] | None = None,
        source_blocks: tuple[CellBlock, ...] | None = None,
    ) -> None:
        _anchor_values(
            source,
            start,
            cell_ids,
            slot_origins,
            root_origins,
            anchor_slots,
            block_indices,
            source_blocks,
        )
        self.source = source
        self.start = start
        self.cell_ids = cell_ids
        self.slot_origins = slot_origins
        self.root_origins = root_origins
        self.anchor_slots = anchor_slots
        self.block_indices = block_indices
        self.source_blocks = source_blocks

    @classmethod
    def from_live(cls, value: adaptation_owner._Anchor, /) -> MigrationAnchor:
        """Retain every host bank without executing generation or replay."""
        return cls(
            MigrationSource.from_live(value.source),
            MigrationStart.from_live(value.start),
            value.cell_ids,
            value.slot_origins,
            value.root_origins,
            value.anchor_slots,
            value.block_indices,
            value.source_blocks,
        )

    def to_live(self) -> adaptation_owner._Anchor:
        """Reconstruct live NamedTuples only at explicit replay."""
        from ._device_adaptation import _Anchor

        self.validate_restored()
        return _Anchor(
            self.source.to_live(),
            self.start.to_live(),
            self.cell_ids,
            self.slot_origins,
            self.root_origins,
            self.anchor_slots,
            self.block_indices,
            self.source_blocks,
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: MigrationAnchor, /) -> MigrationAnchor:
        """Reconstruct retained fields through their numerical bank constructors."""
        return cls(
            MigrationSource.from_stored_fields(value.source),
            MigrationStart.from_stored_fields(value.start),
            value.cell_ids,
            value.slot_origins,
            value.root_origins,
            value.anchor_slots,
            value.block_indices,
            value.source_blocks,
        )

    def _validate_values(self) -> None:
        _anchor_values(
            self.source,
            self.start,
            self.cell_ids,
            self.slot_origins,
            self.root_origins,
            self.anchor_slots,
            self.block_indices,
            self.source_blocks,
        )

    def validate_restored(self) -> None:
        """Validate restored structure once, then scientific values."""
        validate(self)
        self._validate_values()


MIGRATION_ANCHOR_ARTIFACT_TYPES: tuple[type, ...] = (
    MigrationCells,
    MigrationRecords,
    MigrationGrowth,
    MigrationFront,
    MigrationRetiredEntities,
    MigrationSource,
    MigrationStart,
    MigrationAnchor,
)
