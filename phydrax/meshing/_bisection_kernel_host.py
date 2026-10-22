#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host preparation, compatible closure, and inverse bisection kernels.

The public bisection records and this kernel use module references so neither
imports a partially initialized record class during fresh-module loading.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from fractions import Fraction
from itertools import combinations
from typing import final, NamedTuple

import jax
import numpy as np
from numpy.typing import ArrayLike

import phydrax.discretization._cell_geometry as _owner_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_transition_types as transition_types_owner
import phydrax.discretization._cell_geometry_validity as _owner_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._coordinate_enclosure as _owner_phydrax_discretization__coordinate_enclosure
import phydrax.meshing._bisection as bisection_records
import phydrax.meshing._contracts as meshing_contracts
import phydrax.meshing._entity_keys as entity_keys_owner
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._lineage_records as _dependency_phydrax_meshing__lineage
import phydrax.meshing._result as _dependency_phydrax_meshing__result
import phydrax.meshing._topology_edit as _dependency_phydrax_meshing__topology_edit

from .._meshcore import NativeExecutionLimits, NativeExecutionScope
from .._strict import StrictModule
from .._validation import positive_integer
from ..discretization import CellMesh
from ..discretization._adaptive_simplex import maubach_bisection_tables
from ..typing import checked


# Vertices are addressed by compact indices during one adaptation: source
# vertices by the rank of their global ID, new vertices after them in issue
# order. New IDs exceed every existing ID, so compact order is global-ID order
# and every canonical ordering below equals the global-ID ordering.
SHIFT = np.int64(1 << 31)
_VERTEX_LIMIT = 1 << 31


# The Maubach templates are owned by the device layout so that host and device
# bisection share one table set.
MAUBACH_TABLES = {dimension: maubach_bisection_tables(dimension) for dimension in (2, 3)}


def identifiers(values: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1 or np.any(array < 0):
        raise ValueError(
            f"Bisection.{name} must be a rank-1 array of nonnegative IDs; got {(array.shape, array.min(initial=0))!r}"
        )
    return array


def tag_array(values: ArrayLike, count: int, dimension: int, name: str, /) -> np.ndarray:
    tags = np.asarray(values, dtype=np.int32)
    if tags.shape != (count,) or np.any(tags < 1) or np.any(tags > dimension):
        raise ValueError(
            f"Bisection.{name} must hold one tag in the declared simplex tag range; got {(tags.shape, tags, count, dimension)!r}"
        )
    return tags


def simplex_rows(
    values: ArrayLike, count: int, dimension: int, name: str, /
) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64)
    if rows.shape != (count, dimension + 1) or np.any(rows < 0):
        raise ValueError(
            f"Bisection.{name} must have the declared simplex row shape and nonnegative vertex IDs; got {(rows.shape, rows.min(initial=0), count, dimension)!r}"
        )
    ordered = np.sort(rows, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError(
            f"Bisection.{name} must reference distinct vertices in each row; got {rows!r}"
        )
    return rows


def retired_tables(
    keys: tuple[ArrayLike, ...], ids: tuple[ArrayLike, ...], dimension: int, /
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    key_tables = tuple(np.asarray(value, dtype=np.int64) for value in keys)
    id_tables = tuple(np.asarray(value, dtype=np.int64) for value in ids)
    if len(key_tables) != dimension - 1 or len(id_tables) != dimension - 1:
        raise ValueError(
            f"BisectionHierarchy.retired_entity_keys must have one key and ID table per intermediate dimension; got {(len(key_tables), len(id_tables), dimension)!r}"
        )
    for degree, (table, identifiers) in enumerate(
        zip(key_tables, id_tables, strict=True), start=1
    ):
        if table.ndim != 2 or table.shape[1] != degree + 1:
            raise ValueError(
                f"BisectionHierarchy.retired_entity_keys must have the declared entity width; got {(degree, table.shape)!r}"
            )
        if identifiers.shape != (table.shape[0],) or np.any(identifiers < 0):
            raise ValueError(
                f"BisectionHierarchy.retired_entity_ids must align nonnegative IDs with keys; got {(identifiers.shape, identifiers.min(initial=0), table.shape)!r}"
            )
        if np.any(table[:, 1:] <= table[:, :-1]):
            raise ValueError(
                f"BisectionHierarchy.retired_entity_keys must contain ascending vertex IDs per key; got {(degree, table)!r}"
            )
    return key_tables, id_tables


def validated_forest(
    dimension: int,
    cell_global_ids: ArrayLike,
    ordered_vertices: ArrayLike,
    tags: ArrayLike,
    generations: ArrayLike,
    records: dict[str, ArrayLike],
    /,
) -> dict[str, np.ndarray]:
    ids = identifiers(cell_global_ids, "cell_global_ids")
    if ids.size == 0 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError(
            f"BisectionHierarchy.cell_global_ids must be nonempty and strictly increasing; got {ids!r}"
        )
    count = ids.size
    parents = identifiers(records["parent_ids"], "record_parent_ids")
    if np.any(parents[1:] <= parents[:-1]) or np.intersect1d(parents, ids).size:
        raise ValueError(
            f"BisectionHierarchy.record_parent_ids must be strictly increasing and disjoint from active cell IDs; got {(parents, ids)!r}"
        )
    total = parents.size
    children = np.asarray(records["child_ids"], dtype=np.int64)
    blocks = np.asarray(records["parent_blocks"], dtype=np.int32)
    vertices = identifiers(records["vertex_ids"], "record_vertex_ids")
    if children.shape != (total, 2) or np.any(children < 0):
        raise ValueError(
            f"BisectionHierarchy.record_child_ids must contain two nonnegative child IDs per record; got {(children.shape, children.min(initial=0), total)!r}"
        )
    if np.unique(children).size != children.size:
        raise ValueError(
            f"BisectionHierarchy.record_child_ids must assign distinct child IDs to every bisection; got {children!r}"
        )
    if blocks.shape != (total,) or np.any(blocks < 0) or vertices.shape != (total,):
        raise ValueError(
            f"BisectionHierarchy.record_parent_blocks must align nonnegative block and vertex IDs with records; got {(blocks.shape, blocks.min(initial=0), vertices.shape, total)!r}"
        )
    generation = np.asarray(generations, dtype=np.int32)
    if generation.shape != (count,) or np.any(generation < 0):
        raise ValueError(
            f"BisectionHierarchy.generations must hold one nonnegative value per active cell; got {(generation.shape, generation.min(initial=0), count)!r}"
        )
    return {
        "cell_global_ids": ids,
        "ordered_vertices": simplex_rows(
            ordered_vertices, count, dimension, "ordered_vertices"
        ),
        "tags": tag_array(tags, count, dimension, "tags"),
        "generations": generation,
        "record_parent_ids": parents,
        "record_parent_blocks": blocks,
        "record_parent_rows": simplex_rows(
            records["parent_rows"], total, dimension, "record_parent_rows"
        ),
        "record_parent_vertices": simplex_rows(
            records["parent_vertices"], total, dimension, "record_parent_vertices"
        ),
        "record_parent_tags": tag_array(
            records["parent_tags"], total, dimension, "record_parent_tags"
        ),
        "record_child_ids": children,
        "record_vertex_ids": vertices,
    }


def require_uniform_action_family(barycentric: np.ndarray, dimension: int, /) -> None:
    width = dimension + 1
    arrangements = MAUBACH_TABLES[dimension][4].shape[0]
    expected = {
        tuple(
            sorted(
                tuple(
                    1.0 / size if axis in order[:size] else 0.0 for axis in range(width)
                )
                for size in range(1, width + 1)
            )
        )
        for order in MAUBACH_TABLES[dimension][4]
    }
    for sibling_actions in barycentric:
        actual = [
            tuple(sorted(tuple(point.tolist()) for point in action))
            for action in sibling_actions
        ]
        if len(set(actual)) != arrangements or set(actual) != expected:
            raise ValueError(
                f"BisectionUniformRefinement.barycentric_weights must contain the declared subdivision actions; got {(sibling_actions, arrangements)!r}"
            )


class Cells(NamedTuple):
    ids: np.ndarray
    rows: np.ndarray
    tuples: np.ndarray
    tags: np.ndarray
    blocks: np.ndarray
    origins: np.ndarray
    generations: np.ndarray


class Records(NamedTuple):
    parent_ids: np.ndarray
    blocks: np.ndarray
    rows: np.ndarray
    tuples: np.ndarray
    tags: np.ndarray
    child_ids: np.ndarray
    vertices: np.ndarray


class Growth(NamedTuple):
    parents: np.ndarray
    weights: np.ndarray
    levels: np.ndarray


class Front(NamedTuple):
    cells: Cells
    growth: Growth
    split_keys: np.ndarray
    split_vertices: np.ndarray
    vertex_base: int
    next_cell: int


class Flags(NamedTuple):
    marked: np.ndarray
    blocked: np.ndarray
    classes: np.ndarray


class Source(NamedTuple):
    mesh: CellMesh
    dimension: int
    vertex_ids: np.ndarray
    original: np.ndarray
    coordinates: np.ndarray
    cells: Cells


class Request(NamedTuple):
    refine_ids: np.ndarray
    coarsen_ids: np.ndarray
    protected_codes: np.ndarray
    protected_vertices: np.ndarray
    cell_classes: np.ndarray
    facet_keys: np.ndarray
    facet_classes: np.ndarray
    limit: int


class UniformAllowance(NamedTuple):
    work: int
    geometry_queries: int
    cells: int
    vertices: int
    scratch_bytes: int
    wall_seconds: float
    cavity_cells: int

    @classmethod
    @checked
    def from_limits(
        cls,
        limits: meshing_contracts.MeshingLimits,
        /,
        *,
        work: int | None = None,
        geometry_queries: int | None = None,
        seconds: float | None = None,
    ) -> UniformAllowance:
        """Bind published capacities and the remaining execution allowance."""
        return cls(
            limits.maximum_work_units if work is None else work,
            limits.maximum_geometry_queries
            if geometry_queries is None
            else geometry_queries,
            limits.maximum_cells,
            limits.maximum_vertices,
            limits.maximum_scratch_bytes,
            limits.maximum_wall_seconds if seconds is None else seconds,
            limits.maximum_cavity_cells,
        )

    def native_limits(self) -> NativeExecutionLimits:
        """Lower this uniform request's mechanical allowance once."""
        return NativeExecutionLimits(
            self.work,
            self.geometry_queries,
            self.cavity_cells,
            self.scratch_bytes,
            self.wall_seconds,
        )


@contextmanager
def uniform_execution(allowance: UniformAllowance, /) -> Iterator[NativeExecutionScope]:
    """Open one explicit native owner for the uniform request."""
    with NativeExecutionScope(allowance.native_limits()) as scope:
        yield scope


@contextmanager
@checked
def uniform_budget(
    allowance: UniformAllowance,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None = None,
) -> Iterator[
    _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
]:
    """Reserve the request's host stage and settle its measured local counts."""
    remaining = scope.remaining()
    domain_work = (
        allowance.work
        if budget is None
        else min(allowance.work, budget.maximum_work_units - budget.work_units)
    )
    ticket = scope.batch(
        min(domain_work, remaining.remaining_work_units),
        queries_upper=min(
            allowance.geometry_queries, remaining.remaining_geometry_queries
        ),
        bytes_upper=min(allowance.scratch_bytes, remaining.remaining_managed_bytes),
        cavity_upper=min(allowance.cavity_cells, remaining.remaining_cavity_cells),
    )
    if budget is not None:
        with budget.bind_ticket(ticket):
            yield budget
    else:
        budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=allowance.work,
            maximum_memory_bytes=allowance.scratch_bytes,
        )
        with ticket, budget.activate():
            yield budget


@checked
def uniform_count(
    work: int,
    storage_bytes: int = 0,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Count measured host visits and declared storage on the stage ticket."""

    ticket = budget.ticket
    retained = budget.storage_ticket
    baseline = 0 if retained is None else retained.bytes_upper
    proposed_bytes = max(
        0,
        budget.retained_basis_bytes
        + budget.temporary_bytes_upper
        + storage_bytes
        - baseline,
    )
    if work > ticket.remaining_work_units or proposed_bytes > ticket.bytes_upper:
        ticket.refuse(work, bytes_upper=proposed_bytes)
    budget.reserve(work, storage_bytes)


def _uniform_cavity(
    count: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Count one host batch's cavity capacity under its admitted request."""
    if count > budget.ticket.cavity_upper:
        budget.ticket.refuse(0, cavity_upper=count)
    budget.ticket.consume(cavity_cells=count)


@checked
def uniform_retain_source(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Retain the scientific source's explicitly declared host storage."""
    budget.retain_owner(source)


class _Capacity(NamedTuple):
    """Declared cell, vertex, and construction-step budgets of one closure."""

    cells: int
    vertices: int
    work_units: int


class Start(NamedTuple):
    front: Front
    records: Records
    marks: np.ndarray
    next_vertex: int
    compatible: bool | None
    incompatible: int
    uniform: bool
    retired: tuple[tuple[np.ndarray, np.ndarray], ...]
    uniform_refinement: bisection_records.BisectionUniformRefinement | None


class Refinement(NamedTuple):
    front: Front
    records: Records
    rejected_ids: np.ndarray
    tests: int
    iterations: int


class _Family(NamedTuple):
    removed: np.ndarray
    undo: np.ndarray
    facet_keys: np.ndarray
    facet_classes: np.ndarray


class Coarsening(NamedTuple):
    restored: Cells
    removed_ids: np.ndarray
    link_targets: np.ndarray
    undone: np.ndarray
    removed_vertices: np.ndarray
    supports: np.ndarray
    support_weights: np.ndarray
    passes: int
    rejected_ids: np.ndarray


def select[T: (Cells, Records, Growth, Flags)](table: T, index: np.ndarray, /) -> T:
    return type(table)(*(value[index] for value in table))


def joined[T: (Cells, Records, Growth, Flags)](first: T, second: T, /) -> T:
    return type(first)(
        *(np.concatenate((a, b), axis=0) for a, b in zip(first, second, strict=True))
    )


def _empty_records(dimension: int, /) -> Records:
    width = dimension + 1
    empty = np.zeros((0,), dtype=np.int64)
    rows = np.zeros((0, width), dtype=np.int64)
    return Records(
        empty, empty, rows, rows, empty, np.zeros((0, 2), dtype=np.int64), empty
    )


def _edge_codes(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    return np.minimum(first, second) * SHIFT + np.maximum(first, second)


def _members(table: np.ndarray, queries: np.ndarray, /) -> np.ndarray:
    """Membership of queries in one ascending unique table."""

    if table.size == 0:
        return np.zeros(queries.shape, dtype=np.bool_)
    position = np.minimum(np.searchsorted(table, queries), table.size - 1)
    return table[position] == queries


def simplex_keys(rows: np.ndarray, size: int, /) -> np.ndarray:
    columns = np.asarray(tuple(combinations(range(rows.shape[1]), size)))
    faces = np.sort(rows[:, columns], axis=2).reshape((-1, size))
    return np.unique(faces, axis=0)


def _distinct_rows(values: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Distinct non-negative entries of each row, ascending, padded with -1."""

    sentinel = np.iinfo(np.int64).max
    ordered = np.sort(np.where(values < 0, sentinel, values), axis=1)
    fresh = np.ones(ordered.shape, dtype=np.bool_)
    fresh[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    fresh &= ordered != sentinel
    counts = np.sum(fresh, axis=1)
    packed = np.take_along_axis(
        ordered, np.argsort(~fresh, axis=1, kind="stable"), axis=1
    )
    columns = np.arange(values.shape[1])[None, :]
    return np.where(columns < counts[:, None], packed, -1), counts


def _merged_rows(
    sources: np.ndarray, weights: np.ndarray, width: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Sum duplicate sources of each sparse row and pack them into ``width``."""

    count = sources.shape[0]
    owner = np.repeat(np.arange(count, dtype=np.int64), sources.shape[1])
    flat_sources = sources.reshape((-1,))
    flat_weights = weights.reshape((-1,))
    valid = flat_sources >= 0
    owner, flat_sources, flat_weights = (
        owner[valid],
        flat_sources[valid],
        flat_weights[valid],
    )
    order = np.lexsort((flat_sources, owner))
    owner, flat_sources, flat_weights = (
        owner[order],
        flat_sources[order],
        flat_weights[order],
    )
    start = np.ones(owner.shape, dtype=np.bool_)
    start[1:] = (owner[1:] != owner[:-1]) | (flat_sources[1:] != flat_sources[:-1])
    totals = np.bincount(np.cumsum(start) - 1, weights=flat_weights)
    group_owner = owner[start]
    column = np.arange(group_owner.size) - np.searchsorted(group_owner, group_owner)
    if np.any(column >= width):
        raise ValueError(
            f"Bisection.vertex_stencil must fit within one source simplex; got {(column.max(initial=-1), width)!r}"
        )
    merged_sources = np.full((count, width), -1, dtype=np.int64)
    merged_weights = np.zeros((count, width), dtype=np.float64)
    merged_sources[group_owner, column] = flat_sources[start]
    merged_weights[group_owner, column] = totals
    return merged_sources, merged_weights


def _odd_relative(rows: np.ndarray, tuples: np.ndarray, dimension: int, /) -> np.ndarray:
    """Whether each tuple is an odd permutation of its oriented row."""

    pairs = MAUBACH_TABLES[dimension][3]
    position = np.argmax(tuples[:, :, None] == rows[:, None, :], axis=2)
    inversions = np.sum(position[:, pairs[:, 0]] > position[:, pairs[:, 1]], axis=1)
    return inversions % 2 == 1


def _reversed(tuples: np.ndarray, tags: np.ndarray, dimension: int, /) -> np.ndarray:
    return np.take_along_axis(tuples, MAUBACH_TABLES[dimension][2][tags], axis=1)


def _canonical(
    tuples: np.ndarray, tags: np.ndarray, odd: np.ndarray, dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Lexicographic representative of each tagged simplex and its orientation parity."""

    reversed_ = _reversed(tuples, tags, dimension)
    differs = reversed_ != tuples
    first = np.argmax(differs, axis=1)
    rows = np.arange(tuples.shape[0])
    smaller = np.any(differs, axis=1) & (reversed_[rows, first] < tuples[rows, first])
    flips = (tags * (tags + 1) // 2) % 2 == 1
    return (
        np.where(smaller[:, None], reversed_, tuples),
        odd ^ (smaller & flips),
    )


def _oriented_rows(tuples: np.ndarray, odd: np.ndarray, /) -> np.ndarray:
    rows = tuples.copy()
    rows[odd, -2] = tuples[odd, -1]
    rows[odd, -1] = tuples[odd, -2]
    return rows


def _children(
    tuples: np.ndarray, tags: np.ndarray, midpoints: np.ndarray, dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    first_table, second_table = MAUBACH_TABLES[dimension][:2]
    extended = np.concatenate((tuples, midpoints[:, None]), axis=1)
    return (
        np.take_along_axis(extended, first_table[tags], axis=1),
        np.take_along_axis(extended, second_table[tags], axis=1),
    )


@checked
def prepared_source(mesh: CellMesh, /) -> Source:
    dimension = mesh.topological_dimension
    kind = {2: "triangle", 3: "tetrahedron"}.get(dimension)
    if kind is None or any(block.cell_kind != kind for block in mesh.blocks):
        raise ValueError(
            f"Bisection.mesh must contain only triangle or tetrahedral simplex blocks; got {(dimension, tuple(block.cell_kind for block in mesh.blocks))!r}"
        )
    vertex_ids = np.asarray(jax.device_get(mesh.vertex_global_ids), dtype=np.int64)
    if vertex_ids.size >= _VERTEX_LIMIT:
        raise ValueError(
            f"Bisection.mesh.vertex_global_ids must contain fewer than 2**31 vertices; got {vertex_ids.size!r}"
        )
    original = np.argsort(vertex_ids, kind="stable")
    compact = np.empty_like(original)
    compact[original] = np.arange(original.size)
    host_blocks = jax.device_get(
        tuple((block.vertices, block.global_ids) for block in mesh.blocks)
    )
    rows = np.concatenate(
        tuple(
            compact[np.asarray(vertices, dtype=np.int64)] for vertices, _ in host_blocks
        )
    )
    ids = np.concatenate(
        tuple(np.asarray(identifiers, dtype=np.int64) for _, identifiers in host_blocks)
    )
    blocks = np.concatenate(
        tuple(
            np.full((block.cell_count,), index, dtype=np.int64)
            for index, block in enumerate(mesh.blocks)
        )
    )
    order = np.argsort(ids, kind="stable")
    zeros = np.zeros(ids.shape, dtype=np.int64)
    cells = Cells(
        ids[order], rows[order], rows[order], zeros, blocks[order], ids[order], zeros
    )
    coordinates = np.asarray(jax.device_get(mesh.coordinates), dtype=np.float64)[original]
    return Source(mesh, dimension, vertex_ids[original], original, coordinates, cells)


def _compact_vertices(
    source: Source, identifiers: np.ndarray, name: str, /
) -> np.ndarray:
    ids = source.vertex_ids
    position = np.minimum(np.searchsorted(ids, identifiers), ids.size - 1)
    if np.any(ids[position] != identifiers):
        raise ValueError(
            f"Bisection.{name} must reference source mesh vertices; got {identifiers[ids[position] != identifiers]!r}"
        )
    return position


def _marked_ids(values: np.ndarray, source: Source, name: str, /) -> np.ndarray:
    ids = np.asarray(values, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError(
            f"Bisection.{name} must be a strictly increasing rank-1 ID array; got {ids!r}"
        )
    if not np.all(_members(source.cells.ids, ids)):
        raise ValueError(
            f"Bisection.{name} must reference cells present in the source mesh; got {ids!r}"
        )
    return ids


def _aligned_classes(values: np.ndarray, count: int, name: str, /) -> np.ndarray:
    classes = np.asarray(values, dtype=np.int64)
    if classes.shape != (count,):
        raise ValueError(
            f"Bisection.{name} must hold one class per entity; got {(classes.shape, count)!r}"
        )
    return classes


@checked
def prepared_request(
    source: Source,
    refine_cell_ids: np.ndarray,
    coarsen_cell_ids: np.ndarray,
    protected_edges: np.ndarray,
    protected_vertices: np.ndarray,
    cell_classes: np.ndarray,
    facet_classes: np.ndarray,
    maximum_closure_iterations: int,
    /,
    *,
    source_cell_classes: np.ndarray | None = None,
) -> Request:
    refine = _marked_ids(refine_cell_ids, source, "refine_cell_ids")
    coarsen = _marked_ids(coarsen_cell_ids, source, "coarsen_cell_ids")
    if np.intersect1d(refine, coarsen).size:
        raise ValueError(
            f"Bisection.refine_cell_ids must be disjoint from coarsen_cell_ids; got {np.intersect1d(refine, coarsen)!r}"
        )
    edges = np.asarray(protected_edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or np.any(edges[:, 0] >= edges[:, 1]):
        raise ValueError(
            f"Bisection.protected_edges must contain ascending vertex-ID pairs; got {(edges.shape, edges)!r}"
        )
    ends = _compact_vertices(source, edges, "protected_edges")
    vertices = np.asarray(protected_vertices, dtype=np.int64)
    if vertices.ndim != 1:
        raise ValueError(
            f"Bisection.protected_vertices must be a rank-1 ID array; got {vertices.shape!r}"
        )
    protected = np.zeros(source.vertex_ids.shape, dtype=np.bool_)
    protected[_compact_vertices(source, vertices, "protected_vertices")] = True
    maximum_closure_iterations = positive_integer(
        maximum_closure_iterations, "maximum_closure_iterations"
    )
    mesh, dimension = source.mesh, source.dimension
    if source_cell_classes is None:
        cell_order = np.searchsorted(
            source.cells.ids,
            entity_keys_owner.entity_keys(mesh, dimension)[:, 0],
        )
        classes = np.empty(source.cells.ids.shape, dtype=np.int64)
        classes[cell_order] = _aligned_classes(
            cell_classes, cell_order.size, "cell_classes"
        )
    else:
        # An authenticated root projection retains the whole source mesh and
        # coefficient bank; only its selected SCI cell axis is classified.
        classes = _aligned_classes(
            source_cell_classes, source.cells.ids.size, "source_cell_classes"
        )
    facet_keys = np.searchsorted(
        source.vertex_ids,
        entity_keys_owner.entity_keys(mesh, dimension - 1),
    )
    return Request(
        refine,
        coarsen,
        np.unique(_edge_codes(ends[:, 0], ends[:, 1])),
        protected,
        classes,
        facet_keys,
        _aligned_classes(facet_classes, facet_keys.shape[0], "facet_classes"),
        int(maximum_closure_iterations),
    )


def _longest_edge_labels(
    cells: Cells, coordinates: np.ndarray, dimension: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Maubach labels and their shared strict edge-rank/incidence preparation."""

    pairs = MAUBACH_TABLES[dimension][3]
    low = np.minimum(cells.rows[:, pairs[:, 0]], cells.rows[:, pairs[:, 1]])
    high = np.maximum(cells.rows[:, pairs[:, 0]], cells.rows[:, pairs[:, 1]])
    codes = low * SHIFT + high
    unique, first, inverse = np.unique(
        codes.reshape((-1,)), return_index=True, return_inverse=True
    )
    lengths = np.sum(
        (coordinates[high.reshape((-1,))[first]] - coordinates[low.reshape((-1,))[first]])
        ** 2,
        axis=1,
    )
    rank = np.empty(unique.shape, dtype=np.int64)
    rank[np.lexsort((unique, -lengths))] = np.arange(unique.size)
    count = cells.ids.size
    best = np.argmin(rank[inverse.reshape(codes.shape)], axis=1)
    x0 = low[np.arange(count), best]
    xk = high[np.arange(count), best]
    others = cells.rows[(cells.rows != x0[:, None]) & (cells.rows != xk[:, None])]
    others = others.reshape((count, dimension - 1))
    if dimension == 2:
        tuples = np.stack((x0, others[:, 0], xk), axis=1)
    else:
        first_rank = rank[np.searchsorted(unique, _edge_codes(x0, others[:, 0]))]
        second_rank = rank[np.searchsorted(unique, _edge_codes(x0, others[:, 1]))]
        greater = first_rank < second_rank
        x2 = np.where(greater, others[:, 0], others[:, 1])
        x1 = np.where(greater, others[:, 1], others[:, 0])
        tuples = np.stack((x0, x1, x2, xk), axis=1)
    tags = np.full((count,), dimension, dtype=np.int64)
    tuples, _ = _canonical(tuples, tags, np.zeros((count,), dtype=np.bool_), dimension)
    return tuples, tags, unique, rank, inverse.reshape(codes.shape)


def _reflected(
    first: np.ndarray, second: np.ndarray, tags: np.ndarray, dimension: int, /
) -> np.ndarray:
    mirrored = _reversed(second, tags, dimension)
    return (np.sum(first != second, axis=1) == 1) | (
        np.sum(first != mirrored, axis=1) == 1
    )


def _facet_children(
    cells: Cells, owners: np.ndarray, opposite: np.ndarray, dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Child of each owner containing the facet opposite ``opposite``."""

    tuples, tags = cells.tuples[owners], cells.tags[owners]
    first, second = _children(tuples, tags, -1 - owners, dimension)
    at_start = opposite == tuples[:, 0]
    at_end = opposite == tuples[np.arange(owners.size), tags]
    child = np.where(at_start[:, None], second, first)
    return child, at_start | at_end


def _incompatible_facets(cells: Cells, dimension: int, /) -> int:
    """Interior facets violating the matching condition of the current labels."""

    width = dimension + 1
    count = cells.ids.size
    keys = np.sort(
        np.stack(
            tuple(np.delete(cells.rows, column, axis=1) for column in range(width)),
            axis=1,
        ),
        axis=2,
    ).reshape((-1, dimension))
    owners = np.repeat(np.arange(count, dtype=np.int64), width)
    opposite = cells.rows.reshape((-1,))
    order = np.lexsort(keys.T[::-1])
    keys, owners, opposite = keys[order], owners[order], opposite[order]
    shared = np.all(keys[1:] == keys[:-1], axis=1)
    if np.any(shared[1:] & shared[:-1]):
        raise ValueError(
            f"Bisection.mesh.facets must bound at most two cells; got {int(np.count_nonzero(shared[1:] & shared[:-1]))!r}"
        )
    pair = np.flatnonzero(shared)
    left, right = owners[pair], owners[pair + 1]
    same_tag = cells.tags[left] == cells.tags[right]
    tags = cells.tags[left]
    matched = same_tag & _reflected(
        cells.tuples[left], cells.tuples[right], tags, dimension
    )
    left_child, left_open = _facet_children(cells, left, opposite[pair], dimension)
    right_child, right_open = _facet_children(cells, right, opposite[pair + 1], dimension)
    child_tags = np.where(tags > 1, tags - 1, dimension)
    matched |= (
        same_tag
        & left_open
        & right_open
        & _reflected(left_child, right_child, child_tags, dimension)
    )
    return int(np.sum(~matched))


def _subdivided(front: Front, dimension: int, maximum_vertices: int, /) -> Front:
    """Barycentric subdivision: each simplex into (d+1)! simplices (v, m, [f,] c)_d."""

    cells, width = front.cells, dimension + 1
    orders, odd_orders = MAUBACH_TABLES[dimension][4:]
    count, arrangements = cells.ids.size, orders.shape[0]
    tables = [simplex_keys(cells.rows, size) for size in range(2, width)]
    tables.append(cells.rows)
    offsets = np.cumsum([0] + [table.shape[0] for table in tables])
    if front.vertex_base + offsets[-1] >= _VERTEX_LIMIT:
        raise ValueError(
            f"Bisection.uniform_vertices must contain fewer than 2**31 compact vertices; got {front.vertex_base + offsets[-1]!r}"
        )
    if front.vertex_base + offsets[-1] > maximum_vertices:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"Bisection.uniform_vertices must fit within the source vertex allowance; got {(front.vertex_base + offsets[-1], maximum_vertices)!r}",
            stage="uniform-source-preparation",
        )
    columns = [cells.rows[:, orders[:, 0]]]
    for size in range(2, width):
        prefix = np.sort(cells.rows[:, orders[:, :size]], axis=2).reshape((-1, size))
        rows = entity_keys_owner.key_rows(tables[size - 2], prefix).reshape(
            (count, arrangements)
        )
        columns.append(front.vertex_base + offsets[size - 2] + rows)
    centers = front.vertex_base + offsets[-2] + np.arange(count, dtype=np.int64)
    columns.append(np.repeat(centers[:, None], arrangements, axis=1))
    tuples = np.stack(columns, axis=2).reshape((-1, width))
    tags = np.full((tuples.shape[0],), dimension, dtype=np.int64)
    tuples, odd = _canonical(tuples, tags, np.tile(odd_orders, count), dimension)
    children = Cells(
        front.next_cell + np.arange(tuples.shape[0], dtype=np.int64),
        _oriented_rows(tuples, odd),
        tuples,
        tags,
        np.repeat(cells.blocks, arrangements),
        np.repeat(cells.origins, arrangements),
        np.zeros(tags.shape, dtype=np.int64),
    )
    parents = np.concatenate(
        tuple(
            np.pad(table, ((0, 0), (0, width - table.shape[1])), constant_values=-1)
            for table in tables
        )
    )
    weights = np.concatenate(
        tuple(
            np.pad(
                np.full(table.shape, 1.0 / table.shape[1], dtype=np.float64),
                ((0, 0), (0, width - table.shape[1])),
            )
            for table in tables
        )
    )
    growth = Growth(parents, weights, np.zeros((parents.shape[0],), dtype=np.int64))
    return front._replace(
        cells=children, growth=growth, next_cell=front.next_cell + tuples.shape[0]
    )


def _split_edges(
    front: Front, codes: np.ndarray, level: int, dimension: int, /
) -> tuple[Front, np.ndarray, np.ndarray]:
    """Midpoint of each refinement edge, issuing new vertices in sorted key order."""

    known = _members(front.split_keys, codes)
    created = np.unique(codes[~known])
    first = front.vertex_base + front.growth.levels.size
    if first + created.size >= _VERTEX_LIMIT:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"Bisection.vertices must contain fewer than 2**31 compact vertices per adaptation; got {first + created.size!r}",
            stage="bisection-closure",
        )
    vertices = first + np.arange(created.size, dtype=np.int64)
    parents = np.full((created.size, dimension + 1), -1, dtype=np.int64)
    parents[:, 0] = created // SHIFT
    parents[:, 1] = created % SHIFT
    weights = np.zeros(parents.shape, dtype=np.float64)
    weights[:, :2] = 0.5
    growth = joined(
        front.growth,
        Growth(parents, weights, np.full(created.shape, level, dtype=np.int64)),
    )
    keys = np.concatenate((front.split_keys, created))
    owners = np.concatenate((front.split_vertices, vertices))
    order = np.argsort(keys, kind="stable")
    keys, owners = keys[order], owners[order]
    midpoints = owners[np.searchsorted(keys, codes)]
    return (
        front._replace(growth=growth, split_keys=keys, split_vertices=owners),
        midpoints,
        created,
    )


def _bisected(
    front: Front,
    selected: np.ndarray,
    level: int,
    dimension: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[Front, Records, np.ndarray]:
    """Bisect the selected cells once; children follow in (parent ID, 1, 2) order."""

    cells = front.cells
    chosen = select(cells, selected)
    count = selected.size
    uniform_count(count, budget=budget)
    codes = _edge_codes(chosen.tuples[:, 0], chosen.tuples[np.arange(count), chosen.tags])
    # Each local edit bisects the batch cells sharing one refinement edge: it
    # removes those cells and proposes two children each. Independent edges are
    # separate cavities, so the batch is admitted at its largest edge cavity.
    _uniform_cavity(
        3 * int(np.max(np.unique(codes, return_counts=True)[1], initial=0)),
        budget=budget,
    )
    front, midpoints, created = _split_edges(front, codes, level, dimension)
    first, second = _children(chosen.tuples, chosen.tags, midpoints, dimension)
    odd = _odd_relative(chosen.rows, chosen.tuples, dimension)
    tuples = np.stack((first, second), axis=1).reshape((2 * count, dimension + 1))
    odd = np.stack((odd, odd ^ (chosen.tags % 2 == 1)), axis=1).reshape((-1,))
    tags = np.repeat(np.where(chosen.tags > 1, chosen.tags - 1, dimension), 2)
    tuples, odd = _canonical(tuples, tags, odd, dimension)
    ids = front.next_cell + np.arange(2 * count, dtype=np.int64)
    children = Cells(
        ids,
        _oriented_rows(tuples, odd),
        tuples,
        tags,
        np.repeat(chosen.blocks, 2),
        np.repeat(chosen.origins, 2),
        np.repeat(chosen.generations + 1, 2),
    )
    keep = np.ones(cells.ids.shape, dtype=np.bool_)
    keep[selected] = False
    records = Records(
        chosen.ids,
        chosen.blocks,
        chosen.rows,
        chosen.tuples,
        chosen.tags,
        ids.reshape((count, 2)),
        midpoints,
    )
    front = front._replace(
        cells=joined(select(cells, keep), children),
        next_cell=front.next_cell + 2 * count,
    )
    return front, records, created


def _nonconforming(cells: Cells, split_keys: np.ndarray, dimension: int, /) -> np.ndarray:
    pairs = MAUBACH_TABLES[dimension][3]
    codes = _edge_codes(cells.tuples[:, pairs[:, 0]], cells.tuples[:, pairs[:, 1]])
    return np.any(_members(split_keys, codes), axis=1)


def _admit_closure_batch(
    front: Front, records: Records, selected: np.ndarray, capacity: _Capacity, /
) -> None:
    """Refuse one closure batch before it allocates beyond the declared budgets.

    A bisection is one construction step: it retires one cell, creates two, and
    issues the midpoint of its refinement edge unless this adaptation already
    split that edge.
    """

    chosen = select(front.cells, selected)
    ends = chosen.tuples[np.arange(selected.size), chosen.tags]
    codes = np.unique(_edge_codes(chosen.tuples[:, 0], ends))
    cells = front.cells.ids.size + selected.size
    vertices = (
        front.vertex_base
        + front.growth.levels.size
        + int(np.count_nonzero(~_members(front.split_keys, codes)))
    )
    steps = records.parent_ids.size + selected.size
    if (
        cells > capacity.cells
        or vertices > capacity.vertices
        or steps > capacity.work_units
    ):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"Bisection.closure_capacity must admit the next cell, vertex and work batch; got {(cells, vertices, steps, capacity.cells, capacity.vertices, capacity.work_units)!r}",
            stage="bisection-closure",
            entity_ids=tuple(int(value) for value in chosen.ids),
            requested=(
                ("maximum_cells", float(capacity.cells)),
                ("maximum_vertices", float(capacity.vertices)),
                ("maximum_work_units", float(capacity.work_units)),
            ),
            achieved=(
                ("active_cells", float(cells)),
                ("vertices", float(vertices)),
                ("bisection_steps", float(steps)),
            ),
        )


def _closure(
    front: Front,
    selected: np.ndarray,
    protected_codes: np.ndarray,
    limit: int,
    capacity: _Capacity,
    dimension: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[Front, Records, int] | None:
    """Bisect the selected cells, then every cell with a split edge, until conforming.

    Returns ``None`` as soon as the closure would split a protected edge. Every
    batch is admitted against the declared capacity before it is bisected.
    """

    records = _empty_records(dimension)
    iterations = 0
    while selected.size:
        if iterations == limit:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"Bisection.closure_iterations must conform within the declared iteration limit; got {(iterations, limit, front.cells.ids.size, records.parent_ids.size, selected.size)!r}",
                stage="bisection-closure",
                entity_ids=tuple(int(value) for value in front.cells.ids[selected]),
            )
        _admit_closure_batch(front, records, selected, capacity)
        budget.ticket.checkpoint()
        iterations += 1
        front, record, created = _bisected(
            front, selected, iterations, dimension, scope=scope, budget=budget
        )
        if np.any(_members(protected_codes, created)):
            return None
        records = joined(records, record)
        selected = np.flatnonzero(
            _nonconforming(front.cells, front.split_keys, dimension)
        )
    return front, records, iterations


def _admissible_marks(
    front: Front,
    marks: np.ndarray,
    request: Request,
    capacity: _Capacity,
    dimension: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split marks into those whose own closure keeps every protected edge whole.

    Closures of a union are unions of closures, so a clean group is admissible
    wholesale; dirty groups are halved until single offending marks remain.
    """

    empty = np.zeros((0,), dtype=np.int64)
    protected = request.protected_codes
    if protected.size == 0 or marks.size == 0:
        return marks, empty, 0
    tuples, tags = front.cells.tuples[marks], front.cells.tags[marks]
    direct = _members(
        protected, _edge_codes(tuples[:, 0], tuples[np.arange(marks.size), tags])
    )
    accepted, rejected = [empty], [marks[direct]]
    pending = [marks[~direct]] if np.any(~direct) else []
    tests = 0
    while pending:
        group = pending.pop()
        tests += 1
        closure = _closure(
            front,
            group,
            protected,
            request.limit,
            capacity,
            dimension,
            scope=scope,
            budget=budget,
        )
        if closure is not None:
            accepted.append(group)
        elif group.size == 1:
            rejected.append(group)
        else:
            half = group.size // 2
            pending.extend((group[half:], group[:half]))
    return np.sort(np.concatenate(accepted)), np.sort(np.concatenate(rejected)), tests


def _edge_star_template(
    mask: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> tuple[Cells, Records]:
    """One of 64 source-edge subsets, in global lexicographic edge order.

    Each selected edge explicitly relabels its incident parents with tag three;
    records retain that ordered edge and children inherit tag two. The relabeling
    is not a continuation of the parent's previous Maubach similarity class.
    """
    root = np.arange(4, dtype=np.int64)[None, :]
    zero = np.zeros((1,), dtype=np.int64)
    cells = Cells(zero, root, root, zero + 3, zero, zero, zero)
    front = _initial_front(cells, 4, 1, 3)
    records = _empty_records(3)
    for column, (first, last) in enumerate(combinations(range(4), 2)):
        if not mask & (1 << column):
            continue
        rows = front.cells.rows
        selected = np.flatnonzero(
            np.any(rows == first, axis=1) & np.any(rows == last, axis=1)
        )
        middle = rows[selected][
            (rows[selected] != first) & (rows[selected] != last)
        ].reshape((-1, 2))
        tuples = front.cells.tuples.copy()
        tags = front.cells.tags.copy()
        tuples[selected] = np.column_stack((
            np.full(selected.size, first), middle, np.full(selected.size, last),
        ))
        tags[selected] = 3
        tuples[selected], _ = _canonical(
            tuples[selected], tags[selected], np.zeros(selected.size, dtype=np.bool_), 3
        )
        front = front._replace(cells=front.cells._replace(tuples=tuples, tags=tags))
        front, one, _ = _bisected(
            front, selected, column + 1, 3, scope=scope, budget=budget
        )
        records = joined(records, one)
    return front.cells, records


class _StarCoverPriority:
    """A live heap key billing each actually inspected ordering field."""

    __slots__ = ("negative_coverage", "rank", "slot", "budget")

    def __init__(
        self, negative_coverage: int, rank: int, slot: int, *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> None:
        self.negative_coverage = negative_coverage
        self.rank = rank
        self.slot = slot
        self.budget = budget

    def __lt__(self, other: _StarCoverPriority) -> bool:
        # Admission precedes the comparison. A refusal escapes heapq and the
        # request terminates; a partially modified heap is never resumed.
        self.budget.reserve(1)
        if self.negative_coverage != other.negative_coverage:
            return self.negative_coverage < other.negative_coverage
        self.budget.reserve(1)
        if self.rank != other.rank:
            return self.rank < other.rank
        self.budget.reserve(1)
        return self.slot < other.slot


def _whole_edge_star_generation(
    start: Start,
    request: Request,
    capacity: _Capacity,
    /,
    *,
    coordinates: np.ndarray,
    templates: dict[int, tuple[Cells, Records]],
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> Refinement:
    """Prepare complete source-edge stars through a shared finite template bank.

    Every current mark announces its edge using the existing binary64 squared-
    length priority and stable global edge rank. A deterministic uncovered-
    marked-star cover chooses enough announced edges to cut every mark.
    Each source cell applies its selected edges in the same global edge order. This is the
    complete-star closure, factorized over cells, not independent facet splits:
    every incident cell sees every selected edge in the identical order. No
    intermediate mesh is published. One generation has at most eight children
    per source tetrahedron; subsequent generations have their own actual edges.
    """
    front, cells = start.front, start.front.cells
    count = cells.ids.size
    budget.reserve(16 * count, 512 * count)
    tuples, tags, edge_keys, edge_rank, incidence = _longest_edge_labels(cells, coordinates, 3)
    del tuples, tags
    marked_edges = incidence[start.marks]
    ranks = edge_rank[marked_edges]
    if request.protected_codes.size:
        protected = _members(request.protected_codes, edge_keys)
        ranks = np.where(protected[marked_edges], edge_rank.size, ranks)
    first = np.argmin(ranks, axis=1)
    blocked = ranks[np.arange(start.marks.size), first] == edge_rank.size
    rejected = cells.origins[start.marks[blocked]]
    candidates = np.unique(marked_edges[~blocked, first[~blocked]])
    candidate_mask = np.zeros(edge_keys.size, dtype=np.bool_)
    candidate_mask[candidates] = True
    occurrences = np.flatnonzero(candidate_mask[incidence.reshape(-1)])
    occurrence_edges = incidence.reshape(-1)[occurrences]
    order = np.argsort(occurrence_edges, kind="stable")
    occurrence_edges = occurrence_edges[order]
    owners = (occurrences // 6)[order]
    uncovered = np.zeros(count, dtype=np.bool_)
    uncovered[start.marks[~blocked]] = True
    # Count actual uncovered marked owners, not unmarked neighbours. Coverage
    # only decreases: an old heap key is an upper bound until refreshed, so a
    # live top entry is the true maximum with the original global rank tie.
    depth = max(1, (int(owners.size) - 1).bit_length())
    budget.reserve(
        3 * owners.size + edge_keys.size + 7 * candidates.size
        + 2 * candidates.size * depth + start.marks.size,
        512 + 16 * owners.size + 16 * edge_keys.size + 256 * candidates.size,
    )
    coverage = np.bincount(
        occurrence_edges[uncovered[owners]], minlength=edge_keys.size
    )
    starts = np.searchsorted(occurrence_edges, candidates, side="left")
    stops = np.searchsorted(occurrence_edges, candidates, side="right")
    queue = [
        _StarCoverPriority(
            -int(coverage[edge]), int(edge_rank[edge]), slot, budget=budget
        )
        for slot, edge in enumerate(candidates)
    ]
    heapq.heapify(queue)
    remaining = int(start.marks.size - np.count_nonzero(blocked))
    selected_edges = np.empty(candidates.size, dtype=np.int64)
    selected_count = 0
    while queue and remaining:
        # Heap comparisons admit their actual inspected fields through the
        # private key; the unchanged fixed pop/bookkeeping tariff remains.
        budget.reserve(3)
        item = heapq.heappop(queue)
        slot = item.slot
        edge = candidates[slot]
        live = int(coverage[edge])
        if -item.negative_coverage != live:
            if live:
                budget.reserve(3)
                heapq.heappush(
                    queue, _StarCoverPriority(-live, item.rank, slot, budget=budget)
                )
            continue
        star = owners[starts[slot]:stops[slot]]
        budget.reserve(2 * star.size, 256 + 16 * star.size)
        newly_covered = star[uncovered[star]]
        selected_edges[selected_count] = edge_keys[edge]
        selected_count += 1
        uncovered[newly_covered] = False
        remaining -= newly_covered.size
        # Each original marked owner reaches this update once, so the total
        # incidence scan is at most six entries per mark, never a star retry.
        entries = 6 * newly_covered.size
        budget.reserve(3 * entries, 512 + 24 * entries)
        incident = incidence[newly_covered].reshape(-1)
        active = incident[candidate_mask[incident]]
        budget.reserve(active.size)
        np.add.at(coverage, active, -1)
    if np.any(uncovered):
        raise ValueError("Bisection.whole_edge_star must cover every admissible mark")
    edges = np.sort(selected_edges[:selected_count])
    if not edges.size:
        return Refinement(front, _empty_records(3), rejected, 0, 0)
    del marked_edges, ranks, incidence, edge_keys, edge_rank, candidate_mask
    del coverage, queue, item, star, newly_covered, incident, active
    del occurrences, occurrence_edges, owners, candidates, starts, stops, order, uncovered
    bases = np.sort(cells.rows, axis=1)
    pairs = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int64)
    codes = _edge_codes(bases[:, pairs[:, 0]], bases[:, pairs[:, 1]])
    masks = np.sum(
        _members(edges, codes) * (1 << np.arange(6, dtype=np.int64)), axis=1
    )
    patterns, inverse, multiplicities = np.unique(
        masks, return_inverse=True, return_counts=True
    )
    steps = 0
    star_sizes = np.zeros(edges.size, dtype=np.int64)
    groups: list[tuple[np.ndarray, Cells, Records, np.ndarray]] = []
    for group, (mask, multiplicity) in enumerate(zip(patterns, multiplicities, strict=True)):
        key = int(mask)
        template = templates.get(key)
        if template is None:
            template = _edge_star_template(key, scope=scope, budget=budget)
            templates[key] = template
        leaves, records = template
        chosen = np.flatnonzero(inverse == group)
        local_edges = np.flatnonzero(key & (1 << np.arange(6)))
        if local_edges.size:
            edge_rows = np.searchsorted(edges, codes[chosen][:, local_edges])
            local_star = np.bincount(
                records.vertices - 4, minlength=local_edges.size
            )
            np.add.at(star_sizes, edge_rows, np.broadcast_to(local_star, edge_rows.shape))
        steps += int(multiplicity) * records.parent_ids.size
        groups.append((chosen, leaves, records, local_edges))
    vertices = front.vertex_base + front.growth.levels.size + edges.size
    if count + steps > capacity.cells or vertices > capacity.vertices or steps > capacity.work_units:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"Bisection.whole_edge_star must fit actual cells, vertices and binary steps; got {(count + steps, vertices, steps, capacity)!r}",
            stage="bisection-closure",
        )
    if request.limit < 1:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Bisection.whole_edge_star must fit the declared generation limit",
            stage="bisection-closure",
        )
    _uniform_cavity(3 * int(np.max(star_sizes, initial=0)), budget=budget)
    uniform_count(steps, 512 * (count + steps + edges.size), budget=budget)
    level = int(np.max(front.growth.levels, initial=0)) + 1
    front, midpoints, _ = _split_edges(front, edges, level, 3)
    children_parts: list[Cells] = []
    record_parts: list[Records] = []
    next_cell = front.next_cell
    for chosen, leaves, records, local_edges in groups:
        size, splits = chosen.size, records.parent_ids.size
        vertex_map = np.concatenate((
            bases[chosen],
            midpoints[np.searchsorted(edges, codes[chosen][:, local_edges])],
        ), axis=1)
        node_ids = next_cell + np.arange(size * 2 * splits, dtype=np.int64).reshape(size, 2 * splits)
        node_ids = np.column_stack((cells.ids[chosen], node_ids))
        next_cell += size * 2 * splits
        odd = _odd_relative(cells.rows[chosen], bases[chosen], 3)
        rows = _oriented_rows(
            vertex_map[:, leaves.rows].reshape((-1, 4)),
            np.repeat(odd, leaves.ids.size),
        )
        if not splits:
            rows = cells.rows[chosen]
        tuples = (
            vertex_map[:, leaves.tuples].reshape((-1, 4))
            if splits else cells.tuples[chosen]
        )
        tags = np.tile(leaves.tags, size) if splits else cells.tags[chosen]
        children_parts.append(Cells(
            node_ids[:, leaves.ids].reshape(-1),
            rows,
            tuples,
            tags,
            np.repeat(cells.blocks[chosen], leaves.ids.size),
            np.repeat(cells.origins[chosen], leaves.ids.size),
            (cells.generations[chosen, None] + leaves.generations).reshape(-1),
        ))
        if splits:
            parent_rows = _oriented_rows(
                vertex_map[:, records.rows].reshape((-1, 4)),
                np.repeat(odd, splits),
            ).reshape((size, splits, 4))
            parent_rows[:, records.parent_ids == 0] = cells.rows[chosen, None, :]
            record_parts.append(Records(
                node_ids[:, records.parent_ids].reshape(-1),
                np.repeat(cells.blocks[chosen], splits),
                parent_rows.reshape((-1, 4)),
                vertex_map[:, records.tuples].reshape((-1, 4)),
                np.tile(records.tags, size),
                node_ids[:, records.child_ids].reshape((-1, 2)),
                vertex_map[:, records.vertices].reshape(-1),
            ))
    children = Cells(*(np.concatenate([part[index] for part in children_parts]) for index in range(len(Cells._fields))))
    records = (
        Records(*(np.concatenate([part[index] for part in record_parts]) for index in range(len(Records._fields))))
        if record_parts else _empty_records(3)
    )
    front = front._replace(cells=children, next_cell=next_cell)
    return Refinement(front, records, np.asarray(rejected, dtype=np.int64), 0, 1)


def _whole_edge_star_refinement(
    start: Start,
    request: Request,
    capacity: _Capacity,
    /,
    *,
    generations: int,
    target_cells: int | None,
    coordinates: np.ndarray,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> Refinement:
    """Execute the declared descendant generations under one request allowance."""
    marked_origins = start.front.cells.origins[start.marks]
    front = start.front
    records = _empty_records(3)
    rejected = np.empty((0,), dtype=np.int64)
    iterations = 0
    points = coordinates
    templates: dict[int, tuple[Cells, Records]] = {}
    for generation in range(generations):
        marks = np.flatnonzero(np.isin(front.cells.origins, marked_origins))
        if not marks.size:
            break
        prior_growth = front.growth.levels.size
        one = _whole_edge_star_generation(
            start._replace(front=front, marks=marks),
            request._replace(limit=request.limit - iterations),
            capacity._replace(work_units=capacity.work_units - records.parent_ids.size),
            coordinates=points, templates=templates, scope=scope, budget=budget,
        )
        front = one.front
        records = joined(records, one.records)
        rejected = np.union1d(rejected, one.rejected_ids)
        iterations += one.iterations
        if target_cells is not None and front.cells.ids.size >= target_cells:
            break
        if generation + 1 < generations:
            # Only append this generation's new midpoints to the unpublished
            # edge-ordering view. Exact root stencils/maps are reconstructed
            # once by build_bisection_edit, never replaced by this priority view.
            parents = front.growth.parents[prior_growth:, :2]
            created = parents.shape[0]
            budget.reserve(
                16 * created + 3 * points.shape[0], points.nbytes + 96 * created
            )
            points = np.concatenate((
                points,
                0.5 * points[parents[:, 0]] + 0.5 * points[parents[:, 1]],
            ))
    return Refinement(front, records, rejected, 0, iterations)


@checked
def refinement(
    start: Start,
    request: Request,
    capacity: _Capacity,
    dimension: int,
    /,
    *,
    scope: NativeExecutionScope,
    compatibility: bisection_records.BisectionCompatibility,
    refinement_generations: int,
    refinement_target_cells: int | None,
    coordinates: np.ndarray,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Refinement:
    if compatibility is bisection_records.BisectionCompatibility.WHOLE_EDGE_STAR:
        if dimension != 3:
            raise ValueError("Bisection.WHOLE_EDGE_STAR requires tetrahedral cells")
        return _whole_edge_star_refinement(
            start, request, capacity, generations=refinement_generations,
            target_cells=refinement_target_cells,
            coordinates=coordinates, budget=budget, scope=scope
        )
    accepted, rejected, tests = _admissible_marks(
        start.front, start.marks, request, capacity, dimension, scope=scope, budget=budget
    )
    closure = _closure(
        start.front,
        accepted,
        request.protected_codes,
        request.limit,
        capacity,
        dimension,
        scope=scope,
        budget=budget,
    )
    if closure is None:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.INVALID_SPECIFICATION,
            f"Bisection.refinement_closure must retain protected edges when admissible closures are combined; got {(request.protected_codes.size, accepted.size)!r}",
            stage="bisection-closure",
        )
    front, records, iterations = closure
    return Refinement(
        front,
        records,
        np.unique(start.front.cells.origins[rejected]),
        tests,
        iterations,
    )


def _initial_front(
    cells: Cells, vertex_base: int, next_cell: int, dimension: int
) -> Front:
    width = dimension + 1
    empty = np.zeros((0,), dtype=np.int64)
    growth = Growth(
        np.zeros((0, width), dtype=np.int64),
        np.zeros((0, width), dtype=np.float64),
        empty,
    )
    return Front(cells, growth, empty, empty, vertex_base, next_cell)


def _uniform_lineage(
    source: Source,
    request: Request,
    parents: Cells,
    front: Front,
    next_vertex: int,
    scientific_source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bisection_records.BisectionUniformRefinement:
    uniform_retain_source(scientific_source, budget=budget)
    dimension, width = source.dimension, source.dimension + 1
    count = source.vertex_ids.size
    arrangements = MAUBACH_TABLES[dimension][4].shape[0]
    one_hot = np.full((count, width), -1, dtype=np.int64)
    one_hot[:, 0] = source.vertex_ids
    growth = front.growth
    sources = np.concatenate(
        (
            one_hot,
            np.where(
                growth.parents >= 0, source.vertex_ids[np.maximum(growth.parents, 0)], -1
            ),
        )
    )
    weights = np.concatenate(
        (
            np.eye(1, width, dtype=np.float64).repeat(count, axis=0),
            growth.weights,
        )
    )
    parent_rows = source.vertex_ids[np.repeat(parents.rows, arrangements, axis=0)]
    references = _dependency_phydrax_meshing__topology_edit.nested_reference_vertices(
        sources[front.cells.rows],
        weights[front.cells.rows],
        parent_rows,
    ).reshape((parents.ids.size, arrangements, width, dimension))
    matches = sources[front.cells.rows][..., :, None] == parent_rows[:, None, None, :]
    barycentric = np.sum(
        np.where(matches, weights[front.cells.rows][..., None], 0.0),
        axis=2,
    ).reshape((parents.ids.size, arrangements, width, width))
    original_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in source.mesh.blocks)
    )
    original_rows = np.concatenate(
        tuple(np.arange(block.cell_count, dtype=np.int64) for block in source.mesh.blocks)
    )
    original_order = np.argsort(original_ids, kind="stable")
    positions = np.searchsorted(original_ids[original_order], parents.ids)
    if np.any(positions >= original_ids.size):
        raise ValueError(
            f"BisectionUniformRefinement.parent_ids must belong to the declared source parent cohort; got {parents.ids!r}"
        )
    original_positions = original_order[positions]
    if not np.array_equal(original_ids[original_positions], parents.ids):
        raise ValueError(
            f"BisectionUniformRefinement.parent_ids must belong to the declared source parent cohort; got {parents.ids!r}"
        )
    row_order = original_rows[original_positions]
    facet_columns = np.asarray(
        tuple(combinations(range(width), dimension)), dtype=np.int64
    )
    facet_keys = np.sort(parents.rows[:, facet_columns], axis=2).reshape((-1, dimension))
    facets = entity_keys_owner.key_rows(request.facet_keys, facet_keys)
    if np.any(facets < 0):
        raise ValueError(
            f"BisectionUniformRefinement.parent_facet_classes must bind every source facet; got {facets!r}"
        )
    return bisection_records.BisectionUniformRefinement(
        dimension,
        parents.ids,
        parents.blocks,
        row_order,
        source.vertex_ids[parents.rows],
        source.vertex_ids[parents.tuples],
        parents.tags,
        parents.generations,
        request.cell_classes,
        request.facet_classes[facets].reshape((-1, width)),
        front.cells.ids.reshape((-1, arrangements)),
        global_vertices(front.cells.rows, source, next_vertex).reshape(
            (-1, arrangements, width)
        ),
        references,
        barycentric,
        source=scientific_source,
    )


@final
class _UniformReissuanceAdmission(StrictModule):
    original: dict[str, np.ndarray]
    positions: np.ndarray
    retired: tuple[tuple[np.ndarray, np.ndarray], ...]


@final
class _UniformReissuedActions(StrictModule):
    front: Front
    uniform_refinement: bisection_records.BisectionUniformRefinement


def _prepare_uniform_reissuance_admission(
    source: Source,
    request: Request,
    admitted: bisection_records.BisectionUniformRefinement,
    next_vertex: int,
    next_cell: int,
    retained_retired: tuple[tuple[np.ndarray, np.ndarray], ...],
    /,
) -> _UniformReissuanceAdmission:
    vertices = np.asarray(admitted.source.mesh.vertex_global_ids, dtype=np.int64)
    coordinates = np.asarray(admitted.source.mesh.coordinates, dtype=np.float64)
    if (
        source.original.shape != vertices.shape
        or source.coordinates.shape != coordinates.shape
        or np.any((source.original < 0) | (source.original >= vertices.size))
        or not np.array_equal(source.vertex_ids, vertices[source.original])
        or not np.array_equal(
            source.coordinates.view(np.uint64),
            coordinates[source.original].view(np.uint64),
        )
    ):
        raise ValueError(
            f"BisectionUniformRefinement.source must retain the admitted coordinate bank and vertex routing; got {(source.mesh.mesh_id, source.original.shape, source.coordinates.shape, vertices.shape, coordinates.shape)!r}"
        )
    original = admitted.host_arrays()
    positions = np.searchsorted(original["parent_ids"], source.cells.ids)
    if np.any(positions >= original["parent_ids"].size):
        raise ValueError(
            f"BisectionUniformRefinement.parent_ids must belong to the admitted source parent cohort; got {source.cells.ids!r}"
        )
    if not np.array_equal(original["parent_ids"][positions], source.cells.ids):
        raise ValueError(
            f"BisectionUniformRefinement.parent_ids must belong to the admitted source parent cohort; got {source.cells.ids!r}"
        )
    if next_vertex <= max(
        int(source.vertex_ids[-1]), int(np.max(original["child_vertices"]))
    ) or next_cell <= max(int(source.cells.ids[-1]), int(np.max(original["child_ids"]))):
        raise ValueError(
            f"BisectionUniformRefinement.issuers must exceed current and historical scientific IDs; got {(next_vertex, next_cell, source.vertex_ids[-1], source.cells.ids[-1], original['child_vertices'].max(), original['child_ids'].max())!r}"
        )
    for name, current in (
        ("parent_rows", source.vertex_ids[source.cells.rows]),
        ("parent_vertices", source.vertex_ids[source.cells.tuples]),
        ("parent_blocks", source.cells.blocks),
        ("parent_tags", source.cells.tags),
        ("parent_levels", source.cells.generations),
        ("parent_classes", request.cell_classes),
    ):
        if not np.array_equal(original[name][positions], current):
            raise ValueError(
                f"BisectionUniformRefinement.{name} must retain the admitted parent metadata; got {(original[name][positions], current)!r}"
            )
    keys, identifiers = retired_tables(
        tuple(value[0] for value in retained_retired),
        tuple(value[1] for value in retained_retired),
        source.dimension,
    )
    if any(np.any((table < 0) | (table >= source.vertex_ids.size)) for table in keys):
        raise ValueError(
            f"BisectionUniformRefinement.retired_entity_keys must bind the source vertex axis; got {(source.vertex_ids.size, keys)!r}"
        )
    pairs = MAUBACH_TABLES[source.dimension][3]
    edges = _edge_codes(
        source.cells.rows[:, pairs[:, 0]], source.cells.rows[:, pairs[:, 1]]
    )
    if np.any(_members(request.protected_codes, edges)):
        raise ValueError(
            f"BisectionUniformRefinement.protected_edges must remain unsplit during admitted reissuance; got {request.protected_codes!r}"
        )
    return _UniformReissuanceAdmission(
        original, positions, tuple(zip(keys, identifiers, strict=True))
    )


def _reissue_uniform_actions(
    admission: _UniformReissuanceAdmission,
    source: Source,
    request: Request,
    admitted: bisection_records.BisectionUniformRefinement,
    allowance: UniformAllowance,
    next_vertex: int,
    next_cell: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _UniformReissuedActions:
    original, positions = admission.original, admission.positions
    count = source.cells.ids.size
    arrangements = MAUBACH_TABLES[source.dimension][4].shape[0]
    front = _subdivided(
        _initial_front(source.cells, source.vertex_ids.size, next_cell, source.dimension),
        source.dimension,
        allowance.vertices,
    )
    generated = _uniform_lineage(
        source,
        request,
        source.cells,
        front,
        next_vertex,
        admitted.source,
        scope=scope,
        budget=budget,
    )
    incoming = generated.host_arrays()
    order = []
    for parent, old in enumerate(positions):
        authored = {
            matrix.tobytes(): child
            for child, matrix in enumerate(incoming["barycentric_weights"][parent])
        }
        for matrix in original["barycentric_weights"][old]:
            child = authored.get(matrix.tobytes())
            if child is None:
                raise ValueError(
                    f"BisectionUniformRefinement.barycentric_weights must retain the authored raw coefficient actions; got {(parent, old, matrix)!r}"
                )
            order.append(parent * arrangements + child)
    order = np.asarray(order, dtype=np.int64)
    actual = incoming
    regenerated = generated
    if not np.array_equal(order, np.arange(count * arrangements, dtype=np.int64)):
        front = front._replace(
            cells=select(front.cells, order)._replace(
                ids=next_cell + np.arange(count * arrangements, dtype=np.int64),
            )
        )
        actual = dict(incoming)
        actual["child_ids"] = front.cells.ids.reshape((count, arrangements))
        for name in ("child_vertices", "reference_vertices", "barycentric_weights"):
            values = incoming[name]
            actual[name] = values.reshape((-1, *values.shape[2:]))[order].reshape(
                values.shape
            )
    for name in (
        "parent_row_order",
        "parent_facet_classes",
        "reference_vertices",
        "barycentric_weights",
    ):
        expected = original[name][positions]
        if expected.dtype == np.float64:
            equal = np.array_equal(actual[name].view(np.uint64), expected.view(np.uint64))
        else:
            equal = np.array_equal(actual[name], expected)
        if not equal:
            raise ValueError(
                f"BisectionUniformRefinement.{name} must retain the admitted source actions bitwise; got {(actual[name], expected)!r}"
            )
    if actual is not incoming:
        regenerated = bisection_records.BisectionUniformRefinement(
            source.dimension,
            *actual.values(),
            source=admitted.source,
        )
    return _UniformReissuedActions(front, regenerated)


@checked
def reprepare_admitted_uniform(
    source: Source,
    request: Request,
    admitted: bisection_records.BisectionUniformRefinement,
    allowance: UniformAllowance,
    retained_issuers: tuple[int, int],
    retained_retired: tuple[tuple[np.ndarray, np.ndarray], ...],
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Start:
    """Reissue an authenticated existing uniform law, not a policy fallback.

    The collective owner authenticates prior admission and the restored cohort.
    Current parent metadata must still be its source scientific metadata.
    Fresh identifiers retain the source action rows and arrangement order;
    discarded historical children and construction identifiers are never reused.
    """
    if (
        source.dimension != admitted.dimension
        or source.mesh.mesh_id != admitted.source.mesh.mesh_id
    ):
        raise ValueError(
            f"BisectionUniformRefinement.source must bind the admitted source mesh and dimension; got {(source.mesh.mesh_id, admitted.source.mesh.mesh_id, source.dimension, admitted.dimension)!r}"
        )
    if request.coarsen_ids.size:
        raise ValueError(
            f"BisectionUniformRefinement.coarsen_cell_ids must be empty during admitted source preparation; got {request.coarsen_ids.size!r}"
        )
    width = source.dimension + 1
    count = source.cells.ids.size
    arrangements = MAUBACH_TABLES[source.dimension][4].shape[0]
    if count == 0 or count * arrangements > allowance.cells:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"BisectionUniformRefinement.child_ids must fit a nonempty parent cohort within the cell allowance; got {(count, arrangements, allowance.cells)!r}",
            stage="uniform-source-preparation",
        )
    if len(retained_issuers) != 2 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in retained_issuers
    ):
        raise TypeError(
            f"BisectionUniformRefinement.retained_issuers must contain two integer high-water marks; got {retained_issuers!r}"
        )
    next_vertex, next_cell = (int(value) for value in retained_issuers)
    with budget.temporary_scope():
        # Each parent's subdivision is one local edit: it removes the parent
        # and proposes its arrangement children.
        _uniform_cavity(1 + arrangements, budget=budget)
        slots = count * arrangements * width
        uniform_count(
            slots + admitted.child_vertices.size + source.coordinates.size,
            4 * slots * (64 + 16 * width)
            + source.vertex_ids.nbytes
            + source.coordinates.nbytes,
            budget=budget,
        )
        admission = _prepare_uniform_reissuance_admission(
            source, request, admitted, next_vertex, next_cell, retained_retired
        )
        actions = _reissue_uniform_actions(
            admission,
            source,
            request,
            admitted,
            allowance,
            next_vertex,
            next_cell,
            scope=scope,
            budget=budget,
        )
        front, regenerated = actions.front, actions.uniform_refinement
        incompatible = _incompatible_facets(source.cells, source.dimension)
        if _incompatible_facets(front.cells, source.dimension):
            raise ValueError(
                f"BisectionUniformRefinement.child_vertices must match the admitted scientific neighbors; got {(source.dimension, front.cells.ids.size)!r}"
            )
        return Start(
            front,
            _empty_records(source.dimension),
            np.flatnonzero(np.isin(front.cells.origins, request.refine_ids)),
            next_vertex,
            incompatible == 0,
            incompatible,
            True,
            admission.retired,
            regenerated,
        )


def _labeled_start(
    source: Source,
    request: Request,
    compatibility: bisection_records.BisectionCompatibility,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    uniform_allowance: UniformAllowance,
    source_hierarchy: bisection_records.BisectionHierarchy | None,
    scientific_source: _dependency_phydrax_meshing__result.CellMeshingResult,
    retained_issuers: tuple[int, int] | None = None,
    retained_retired: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
) -> Start:
    dimension = source.dimension
    if request.coarsen_ids.size:
        raise ValueError(
            f"Bisection.coarsen_cell_ids must use a BisectionHierarchy from a previous bisection; got {request.coarsen_ids.size!r}"
        )
    tuples, tags, edge_keys, edge_rank, incidence = _longest_edge_labels(
        source.cells, source.coordinates, dimension
    )
    del edge_keys, edge_rank, incidence
    cells = source.cells._replace(tuples=tuples, tags=tags)
    next_vertex = (
        int(source.vertex_ids[-1]) + 1
        if source_hierarchy is None
        else source_hierarchy.next_vertex_id
    )
    next_cell = (
        int(cells.ids[-1]) + 1
        if source_hierarchy is None
        else source_hierarchy.next_cell_id
    )
    if retained_issuers is not None:
        if source_hierarchy is not None or len(retained_issuers) != 2:
            raise ValueError(
                f"Bisection.retained_issuers must bind one predecessor without a second hierarchy; got {(retained_issuers, source_hierarchy is not None)!r}"
            )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in retained_issuers
        ):
            raise TypeError(
                f"Bisection.retained_issuers must contain integer high-water marks; got {retained_issuers!r}"
            )
        next_vertex, next_cell = (int(value) for value in retained_issuers)
        if next_vertex <= source.vertex_ids[-1] or next_cell <= cells.ids[-1]:
            raise ValueError(
                f"Bisection.retained_issuers must exceed current source vertex and cell IDs; got {(next_vertex, next_cell, source.vertex_ids[-1], cells.ids[-1])!r}"
            )
    front = _initial_front(cells, source.vertex_ids.size, next_cell, dimension)
    incompatible = _incompatible_facets(cells, dimension)
    uniform_refinement = None
    uniform = False
    if incompatible:
        match compatibility:
            case bisection_records.BisectionCompatibility.REJECT:
                raise ValueError(
                    f"Bisection.compatibility must admit matching labels using UNIFORM_REFINEMENT or triangle CONFORMING_CLOSURE; got {(compatibility, incompatible, dimension)!r}"
                )
            case bisection_records.BisectionCompatibility.UNIFORM_REFINEMENT:
                pairs = MAUBACH_TABLES[dimension][3]
                edges = _edge_codes(
                    cells.rows[:, pairs[:, 0]], cells.rows[:, pairs[:, 1]]
                )
                if np.any(_members(request.protected_codes, edges)):
                    raise ValueError(
                        f"Bisection.protected_edges must remain unsplit during uniform barycentric refinement; got {request.protected_codes!r}"
                    )
                arrangements = MAUBACH_TABLES[dimension][4].shape[0]
                if cells.ids.size * arrangements > uniform_allowance.cells:
                    raise _leaf_phydrax_meshing__failures.MeshingFailure(
                        _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                        f"Bisection.uniform_cells must fit within the source cell allowance; got {(cells.ids.size * arrangements, uniform_allowance.cells)!r}",
                        stage="uniform-source-preparation",
                    )
                with budget.temporary_scope():
                    # One parent's subdivision is one local edit.
                    _uniform_cavity(1 + arrangements, budget=budget)
                    slots = cells.ids.size * arrangements * (dimension + 1)
                    uniform_count(
                        slots,
                        2 * slots * (64 + 16 * (dimension + 1))
                        + source.vertex_ids.nbytes,
                        budget=budget,
                    )
                    front = _subdivided(front, dimension, uniform_allowance.vertices)
                    uniform_refinement = _uniform_lineage(
                        source,
                        request,
                        cells,
                        front,
                        next_vertex,
                        scientific_source,
                        scope=scope,
                        budget=budget,
                    )
                uniform = True
            case bisection_records.BisectionCompatibility.CONFORMING_CLOSURE:
                if dimension != 2:
                    raise ValueError(
                        f"Bisection.compatibility must apply CONFORMING_CLOSURE to triangle meshes only; tetrahedra require UNIFORM_REFINEMENT; got {(compatibility, dimension, incompatible)!r}"
                    )
            case bisection_records.BisectionCompatibility.WHOLE_EDGE_STAR:
                if dimension != 3:
                    raise ValueError("Bisection.WHOLE_EDGE_STAR requires tetrahedral cells")
            case _:
                raise ValueError(
                    f"Bisection.compatibility must select a supported bisection compatibility mode; got {compatibility!r}"
                )
    marks = np.flatnonzero(np.isin(front.cells.origins, request.refine_ids))
    retired = tuple(
        (np.zeros((0, degree + 1), dtype=np.int64), np.zeros((0,), dtype=np.int64))
        for degree in range(1, dimension)
    )
    if source_hierarchy is not None:
        retired = _bound_start(source, request, source_hierarchy).retired
    if retained_retired is not None:
        if source_hierarchy is not None:
            raise ValueError(
                f"Bisection.retained_retired must bind one predecessor without a second source hierarchy; got {source_hierarchy.hierarchy_id!r}"
            )
        keys, identifiers = retired_tables(
            tuple(value[0] for value in retained_retired),
            tuple(value[1] for value in retained_retired),
            dimension,
        )
        if any(np.any((table < 0) | (table >= source.vertex_ids.size)) for table in keys):
            raise ValueError(
                f"Bisection.retained_retired must reference the source vertex axis; got {(source.vertex_ids.size, keys)!r}"
            )
        retired = tuple(zip(keys, identifiers, strict=True))
    return Start(
        front,
        _empty_records(dimension),
        marks,
        next_vertex,
        incompatible == 0,
        incompatible,
        uniform,
        retired,
        uniform_refinement,
    )


@checked
def prepared_start(
    source: Source,
    request: Request,
    compatibility: bisection_records.BisectionCompatibility,
    hierarchy: bisection_records.BisectionHierarchy | None,
    allowance: UniformAllowance,
    scientific_source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    future_refinement: bool = False,
    retained_issuers: tuple[int, int] | None = None,
    retained_retired: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
) -> Start:
    # Device preparation admits marks later. A completely inverted hierarchy
    # retains the source (possibly incompatible) root tags, not the uniform
    # children's matching condition; relabel those roots before future splits.
    # These inputs are authenticated by the accepted collective source
    # replay before entry and retain its issuer and retirement law
    # without constructing a hierarchy for a different presentation mesh.
    if (retained_issuers is None) != (retained_retired is None):
        raise ValueError(
            f"Bisection.retained_issuers must share the predecessor with retained retirement; got {(retained_issuers is not None, retained_retired is not None)!r}"
        )
    if hierarchy is not None and retained_issuers is not None:
        raise ValueError(
            f"Bisection.retained_issuers must have one predecessor without a second hierarchy; got {(hierarchy.hierarchy_id, retained_issuers)!r}"
        )
    if hierarchy is not None:
        bound = _bound_start(source, request, hierarchy)
        if bound.uniform_refinement is not None:
            _require_uniform_source_geometry(
                scientific_source, bound.uniform_refinement, scope=scope, budget=budget
            )
        if (
            bound.records.parent_ids.size
            or bound.uniform_refinement is not None
            or request.coarsen_ids.size
            or (not request.refine_ids.size and not future_refinement)
        ):
            return bound
    return _labeled_start(
        source,
        request,
        compatibility,
        uniform_allowance=allowance,
        source_hierarchy=hierarchy,
        scientific_source=scientific_source,
        retained_issuers=retained_issuers,
        retained_retired=retained_retired,
        scope=scope,
        budget=budget,
    )


def _require_uniform_source_geometry(
    current: _dependency_phydrax_meshing__result.CellMeshingResult,
    lineage: bisection_records.BisectionUniformRefinement,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Bind action groups to source SCI roots, columns, and coefficient banks."""

    original = lineage.source
    uniform_retain_source(original, budget=budget)
    uniform_count(
        lineage.barycentric_weights.size,
        lineage.barycentric_weights.size * 32,
        budget=budget,
    )
    arrays = lineage.host_arrays()
    require_uniform_action_family(arrays["barycentric_weights"], lineage.dimension)
    for identifier, block, row, vertices in zip(
        arrays["parent_ids"],
        arrays["parent_blocks"],
        arrays["parent_row_order"],
        arrays["parent_rows"],
        strict=True,
    ):
        if block < 0 or block >= len(original.mesh.blocks):
            raise ValueError(
                f"BisectionUniformRefinement.parent_blocks must bind declared source blocks; got {block!r}"
            )
        definition = original.mesh.blocks[int(block)]
        if (
            row < 0
            or row >= definition.cell_count
            or np.asarray(definition.global_ids)[row] != identifier
        ):
            raise ValueError(
                f"BisectionUniformRefinement.parent_blocks must bind source block and row IDs; got block={block!r}, row={row!r}, parent={identifier!r}"
            )
        if not np.array_equal(
            np.asarray(original.mesh.vertex_global_ids)[
                np.asarray(definition.vertices)[row]
            ],
            vertices,
        ):
            raise ValueError(
                f"BisectionUniformRefinement.parent_rows must retain the source scientific vertex columns; got {(identifier, block, row, vertices)!r}"
            )
    origin = current.geometry.restriction_source
    if origin is None or (
        origin.source_geometry_id
        != _owner_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
            original.geometry
        )
        or origin.source_topology_id != original.mesh.topology_id
    ):
        raise ValueError(
            f"BisectionUniformRefinement.source must retain the restriction source geometry and topology; got {(origin is None, original.mesh.topology_id, original.geometry.geometry_layout_id)!r}"
        )
    original_elements, original_routes, _ = original.geometry.resolve(
        original.mesh, budget=budget
    )
    elements, routes, _ = current.geometry.resolve(current.mesh, budget=budget)
    uniform_count(
        original.geometry.coordinates.size + current.geometry.coordinates.size,
        (original.geometry.coordinates.size + current.geometry.coordinates.size) * 512,
        budget=budget,
    )

    original_bank = _owner_phydrax_discretization__coordinate_enclosure.prepared_coordinate_source_bank(
        original.geometry, budget=budget
    )
    current_bank = (
        original_bank
        if current.geometry.coordinates is original.geometry.coordinates
        and current.geometry.exact_source is original.geometry.exact_source
        else _owner_phydrax_discretization__coordinate_enclosure.prepared_coordinate_source_bank(
            current.geometry, budget=budget
        )
    )
    original_vertex_ids = np.asarray(jax.device_get(original.mesh.vertex_global_ids))
    roots: dict[
        int,
        tuple[
            _owner_phydrax_discretization__cell_geometry.CellGeometryElement,
            np.ndarray,
            np.ndarray,
        ],
    ] = {}
    for block, element, route in zip(
        original.mesh.blocks, original_elements, original_routes, strict=True
    ):
        identifiers, rows, source_route = jax.device_get(
            (block.global_ids, block.vertices, route)
        )
        identifiers, rows, source_route = (
            np.asarray(identifiers),
            np.asarray(rows),
            np.asarray(source_route),
        )
        for row, identifier in enumerate(identifiers):
            roots[int(identifier)] = (
                element,
                source_route[row],
                original_vertex_ids[rows[row]],
            )
    coefficient_ids = (
        np.arange(current.geometry.coordinates.shape[0], dtype=np.int64)
        if current.mesh.storage is None
        else np.asarray(jax.device_get(current.mesh.storage.coordinate_global_ids))
    )
    for block, element, route in zip(current.mesh.blocks, elements, routes, strict=True):
        root = element
        while isinstance(
            root,
            (
                _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement,
                _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
            ),
        ):
            root = root.source_element
        _owner_phydrax_discretization__cell_geometry.require_p1_cardinal_source(root)
        parents, corners = jax.device_get(
            (
                origin.block_parent_cell_ids[block.name],
                origin.block_parent_vertex_ids[block.name],
            )
        )
        parents, corners = np.asarray(parents), np.asarray(corners)
        uniform_count(block.vertices.size, block.vertices.size * 32, budget=budget)
        host_route = np.asarray(jax.device_get(route))
        for parent, vertices, coefficients in zip(
            parents, corners, host_route, strict=True
        ):
            if int(parent) not in roots:
                raise ValueError(
                    f"BisectionUniformRefinement.parent_ids must reference a declared scientific source cell; got {parent!r}"
                )
            original_element, original_route, original_corners = roots[int(parent)]
            ancestor = element
            while (
                isinstance(
                    ancestor,
                    (
                        _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement,
                        _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
                    ),
                )
                and ancestor.element_id != original_element.element_id
            ):
                ancestor = ancestor.source_element
            if ancestor.element_id != original_element.element_id or not np.array_equal(
                vertices, original_corners
            ):
                raise ValueError(
                    f"BisectionUniformRefinement.source_element must retain the source basis and corner columns; got {(ancestor.element_id, original_element.element_id, vertices, original_corners)!r}"
                )
            if not np.array_equal(coefficient_ids[coefficients], original_route):
                raise ValueError(
                    f"BisectionUniformRefinement.coefficient_ids must retain the source coefficient SCI columns; got {(coefficient_ids[coefficients], original_route)!r}"
                )
            if tuple(current_bank[int(index)] for index in coefficients) != tuple(
                original_bank[int(index)] for index in original_route
            ):
                raise ValueError(
                    f"BisectionUniformRefinement.coefficient_bank must retain the source coefficient values; got {(parent, coefficients, original_route)!r}"
                )


@checked
def uniform_binary_chart(
    identifier: int,
    hierarchy: bisection_records.BisectionHierarchy,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[int, int | None, tuple[np.ndarray, ...]]:
    """Reconstruct authored binary half actions, never compress source weights."""
    lineage = hierarchy.uniform_refinement
    if lineage is None:
        raise ValueError(
            f"BisectionHierarchy.uniform_refinement must retain a source for packed chart reconstruction; got {hierarchy.hierarchy_id!r}"
        )
    uniform_count(
        hierarchy.record_parent_ids.size + lineage.child_ids.size,
        256 * (hierarchy.record_parent_ids.size + lineage.child_ids.size),
        budget=budget,
    )
    arrays = lineage.host_arrays()
    roots = {int(parent): (row, None) for row, parent in enumerate(arrays["parent_ids"])}
    roots.update(
        {
            int(child): (row, column)
            for row, children in enumerate(arrays["child_ids"])
            for column, child in enumerate(children)
        }
    )
    parents = np.asarray(hierarchy.record_parent_ids)
    children = np.asarray(hierarchy.record_child_ids)
    rows = np.asarray(hierarchy.record_parent_rows)
    tuples = np.asarray(hierarchy.record_parent_vertices)
    tags = np.asarray(hierarchy.record_parent_tags)
    vertices = np.asarray(hierarchy.record_vertex_ids)
    records = {int(parent): row for row, parent in enumerate(parents)}
    predecessors: dict[int, int] = {}
    for parent, descendants in zip(parents, children, strict=True):
        for child in descendants:
            known = predecessors.setdefault(int(child), int(parent))
            if known != int(parent):
                raise ValueError(
                    f"BisectionHierarchy.record_child_ids must assign one scientific predecessor per packed chart cell; got {(child, known, parent)!r}"
                )
    path = []
    current = identifier
    visited: set[int] = set()
    while current not in roots:
        if current in visited or current not in predecessors or current not in records:
            raise ValueError(
                f"BisectionHierarchy.record_parent_ids must provide an acyclic chart path to a source root; got {(identifier, current, tuple(path))!r}"
            )
        visited.add(current)
        path.append(current)
        current = predecessors[current]
    root, child = roots[current]
    steps = []
    width = lineage.dimension + 1
    for descendant in reversed(path):
        parent = predecessors[descendant]
        parent_row = records[parent]
        parent_vertices = rows[parent_row]
        corners = rows[records[descendant]]
        action = np.zeros((width, width), dtype=np.float64)
        endpoints = (tuples[parent_row, 0], tuples[parent_row, tags[parent_row]])
        for corner, vertex in enumerate(corners):
            if vertex == vertices[parent_row]:
                for endpoint in endpoints:
                    matches = np.flatnonzero(parent_vertices == endpoint)
                    if matches.size != 1:
                        raise ValueError(
                            f"BisectionHierarchy.record_parent_vertices must retain the source columns for each split edge; got {(parent, endpoint, matches.size)!r}"
                        )
                    action[corner, matches[0]] = 0.5
            else:
                matches = np.flatnonzero(parent_vertices == vertex)
                if matches.size != 1:
                    raise ValueError(
                        f"BisectionHierarchy.record_parent_rows must retain source support for every chart corner; got {(parent, vertex, matches.size)!r}"
                    )
                action[corner, matches[0]] = 1.0
        steps.append(action)
    return root, child, tuple(steps)


@contextmanager
def _bisection_presentation_phase(
    work: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Iterator[None]:
    """Keep presentation traversal and its scratch inside a live phase."""
    if not budget.ticket.settled:
        with budget.temporary_scope():
            yield
        return
    retained = budget.storage_ticket
    baseline = 0 if retained is None else retained.bytes_upper
    storage = max(
        0,
        budget.retained_basis_bytes
        + budget.temporary_bytes_upper
        + 256 * work
        - baseline,
    )
    ticket = scope.batch(work, bytes_upper=storage)
    with budget.bind_ticket(ticket), budget.temporary_scope():
        yield


@checked
def rebind_bisection_presentation_blocks(
    hierarchy: bisection_records.BisectionHierarchy,
    target: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bisection_records.BisectionHierarchy:
    """Lower binary records onto descendant presentation groups."""
    if (
        hierarchy.record_parent_ids.size == 0
        or target.geometry.restriction_source is None
    ):
        return hierarchy
    work = hierarchy.record_child_ids.size + hierarchy.cell_global_ids.size
    with _bisection_presentation_phase(work, scope=scope, budget=budget):
        uniform_count(work, 256 * work, budget=budget)
        origin = target.geometry.restriction_source
        owners = {}
        roots = {}
        for block_index, block in enumerate(target.mesh.blocks):
            ancestors = np.asarray(
                jax.device_get(origin.block_parent_cell_ids[block.name])
            )
            identifiers = np.asarray(jax.device_get(block.global_ids))
            for identifier, ancestor in zip(identifiers, ancestors, strict=True):
                owners[int(identifier)] = block_index
                roots[int(identifier)] = int(ancestor)
        parents = np.asarray(hierarchy.record_parent_ids)
        children = np.asarray(hierarchy.record_child_ids)
        if np.any(children <= parents[:, None]):
            raise ValueError(
                f"BisectionHierarchy.record_child_ids must exceed parent birth IDs monotonically; got {(parents, children)!r}"
            )
        descendants = {
            int(parent): tuple(int(child) for child in row)
            for parent, row in zip(parents, children, strict=True)
        }
        representatives: dict[int, tuple[int, int]] = {}
        pending: list[tuple[int, bool]] = [
            (int(parent), False) for parent in parents[::-1]
        ]
        while pending:
            identifier, expanded = pending.pop()
            if identifier in representatives:
                continue
            if identifier in owners:
                representatives[identifier] = (identifier, roots[identifier])
                continue
            if identifier not in descendants:
                raise ValueError(
                    f"BisectionHierarchy.record_child_ids must provide an active descendant for each presentation record; got {identifier!r}"
                )
            row = descendants[identifier]
            if not expanded:
                pending.append((identifier, True))
                pending.extend((child, False) for child in reversed(row))
                continue
            values = [representatives[child] for child in row]
            if len({root for _, root in values}) != 1:
                raise ValueError(
                    f"BisectionHierarchy.record_child_ids must retain one source root per descendant cohort; got {(identifier, values)!r}"
                )
            representatives[identifier] = min(values)
        blocks = np.asarray(
            [owners[representatives[int(parent)][0]] for parent in parents],
            dtype=np.int32,
        )
        if np.array_equal(blocks, np.asarray(hierarchy.record_parent_blocks)):
            return hierarchy
        return bisection_records.BisectionHierarchy(
            hierarchy.dimension,
            hierarchy.cell_global_ids,
            hierarchy.ordered_vertices,
            hierarchy.tags,
            hierarchy.generations,
            scientific_cell_ids=hierarchy.scientific_cell_ids,
            scientific_block_ids=hierarchy.scientific_block_ids,
            record_parent_ids=hierarchy.record_parent_ids,
            record_parent_blocks=blocks,
            record_parent_rows=hierarchy.record_parent_rows,
            record_parent_vertices=hierarchy.record_parent_vertices,
            record_parent_tags=hierarchy.record_parent_tags,
            record_child_ids=hierarchy.record_child_ids,
            record_vertex_ids=hierarchy.record_vertex_ids,
            retired_entity_keys=hierarchy.retired_entity_keys,
            retired_entity_ids=hierarchy.retired_entity_ids,
            next_vertex_id=hierarchy.next_vertex_id,
            next_cell_id=hierarchy.next_cell_id,
            uniform_refinement=hierarchy.uniform_refinement,
            refinement_rule=hierarchy.refinement_rule,
        )


def _bound_records(
    source: Source, hierarchy: bisection_records.BisectionHierarchy, /
) -> Records:
    def vertices(values: ArrayLike) -> np.ndarray:
        return _compact_vertices(
            source, np.asarray(values, dtype=np.int64), "BisectionHierarchy"
        )

    blocks = np.asarray(hierarchy.record_parent_blocks, dtype=np.int64)
    if np.any(blocks >= len(source.mesh.blocks)):
        raise ValueError(
            f"BisectionHierarchy.record_parent_blocks must reference declared mesh blocks; got {(blocks, len(source.mesh.blocks))!r}"
        )
    return Records(
        np.asarray(hierarchy.record_parent_ids, dtype=np.int64),
        blocks,
        vertices(hierarchy.record_parent_rows),
        vertices(hierarchy.record_parent_vertices),
        np.asarray(hierarchy.record_parent_tags, dtype=np.int64),
        np.asarray(hierarchy.record_child_ids, dtype=np.int64),
        vertices(hierarchy.record_vertex_ids),
    )


def _bound_start(
    source: Source, request: Request, hierarchy: bisection_records.BisectionHierarchy, /
) -> Start:
    dimension = source.dimension
    if hierarchy.dimension != dimension:
        raise ValueError(
            f"BisectionHierarchy.dimension must match the source mesh dimension; got {(hierarchy.dimension, dimension)!r}"
        )
    if not np.array_equal(np.asarray(hierarchy.cell_global_ids), source.cells.ids):
        raise ValueError(
            f"BisectionHierarchy.cell_global_ids must match the active source mesh IDs; got {(np.asarray(hierarchy.cell_global_ids), source.cells.ids)!r}"
        )
    tuples = _compact_vertices(
        source,
        np.asarray(hierarchy.ordered_vertices, dtype=np.int64),
        "BisectionHierarchy",
    )
    if not np.array_equal(np.sort(tuples, axis=1), np.sort(source.cells.rows, axis=1)):
        raise ValueError(
            f"BisectionHierarchy.ordered_vertices must retain the source mesh vertex sets; got {(tuples, source.cells.rows)!r}"
        )
    if (
        hierarchy.next_vertex_id <= source.vertex_ids[-1]
        or hierarchy.next_cell_id <= source.cells.ids[-1]
    ):
        raise ValueError(
            f"BisectionHierarchy.counters must exceed source mesh IDs; got {(hierarchy.next_vertex_id, hierarchy.next_cell_id, source.vertex_ids[-1], source.cells.ids[-1])!r}"
        )
    cells = source.cells._replace(
        tuples=tuples,
        tags=np.asarray(hierarchy.tags, dtype=np.int64),
        generations=np.asarray(hierarchy.generations, dtype=np.int64),
    )
    front = _initial_front(
        cells, source.vertex_ids.size, hierarchy.next_cell_id, dimension
    )
    retired = tuple(
        (
            _compact_vertices(
                source, np.asarray(keys, dtype=np.int64), "BisectionHierarchy"
            ),
            np.asarray(ids, dtype=np.int64),
        )
        for keys, ids in zip(
            hierarchy.retired_entity_keys, hierarchy.retired_entity_ids, strict=True
        )
    )
    return Start(
        front,
        _bound_records(source, hierarchy),
        np.searchsorted(cells.ids, request.refine_ids),
        hierarchy.next_vertex_id,
        None,
        0,
        False,
        retired,
        hierarchy.uniform_refinement,
    )


def _record_links(
    ids: np.ndarray, child_ids: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Record of each active cell as a child and the row of its active sibling."""

    flat = child_ids.reshape((-1,))
    order = np.argsort(flat, kind="stable")
    ordered = flat[order]
    position = np.minimum(np.searchsorted(ordered, ids), ordered.size - 1)
    hit = ordered[position] == ids
    slot = order[position]
    partner_ids = flat[slot ^ 1]
    partner = np.minimum(np.searchsorted(ids, partner_ids), ids.size - 1)
    sibling = np.where(hit & (ids[partner] == partner_ids), partner, -1)
    return np.where(hit, slot // 2, -1), sibling


def _facet_merges(
    records: Records,
    family: np.ndarray,
    facets: tuple[np.ndarray, np.ndarray],
    dimension: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Class agreement of the facet halves each undone bisection merges."""

    keys, classes = facets
    tuples, tags = records.tuples[family], records.tags[family]
    count = family.size
    first = tuples.copy()
    first[np.arange(count), tags] = records.vertices[family]
    second = tuples.copy()
    second[:, 0] = records.vertices[family]
    agree = np.ones((count,), dtype=np.bool_)
    restored, restored_classes, owners = [], [], []
    for column in range(1, dimension + 1):
        active = np.flatnonzero(tags != column)
        halves = [
            np.sort(np.delete(value[active], column, axis=1), axis=1)
            for value in (first, second, tuples)
        ]
        first_rows = entity_keys_owner.key_rows(keys, halves[0])
        second_rows = entity_keys_owner.key_rows(keys, halves[1])
        if np.any(first_rows < 0) or np.any(second_rows < 0):
            raise ValueError(
                f"BisectionHierarchy.record_parent_vertices must agree with current mesh facets; got {(first_rows, second_rows)!r}"
            )
        agree[active] &= classes[first_rows] == classes[second_rows]
        restored.append(halves[2])
        restored_classes.append(classes[first_rows])
        owners.append(active)
    return (
        agree,
        np.concatenate(restored),
        np.concatenate(restored_classes),
        np.concatenate(owners),
    )


def _removable_family(
    cells: Cells,
    flags: Flags,
    records: Records,
    protected: np.ndarray,
    facets: tuple[np.ndarray, np.ndarray],
    dimension: int,
    /,
) -> _Family | None:
    """Good vertices: unprotected, star = marked sibling pairs of their bisections."""

    if records.parent_ids.size == 0:
        return None
    vertex_count = protected.size
    record, sibling = _record_links(cells.ids, records.child_ids)
    leaf = sibling >= 0
    creator = np.where(leaf, records.vertices[np.maximum(record, 0)], -1)
    star = np.bincount(cells.rows.reshape((-1,)), minlength=vertex_count)
    created = np.bincount(creator[leaf], minlength=vertex_count)
    candidate = (star > 0) & (star == created) & ~protected
    member = leaf & candidate[np.maximum(creator, 0)]
    if not np.any(member):
        return None
    family = np.unique(record[member])
    agree, facet_keys, facet_classes, owners = _facet_merges(
        records, family, facets, dimension
    )
    record_agrees = np.zeros(records.parent_ids.shape, dtype=np.bool_)
    record_agrees[family] = agree
    eligible = (
        flags.marked
        & ~flags.blocked
        & (flags.classes == flags.classes[np.maximum(sibling, 0)])
        & record_agrees[np.maximum(record, 0)]
    )
    spoiled = np.bincount(creator[member & ~eligible], minlength=vertex_count) > 0
    good = candidate & ~spoiled
    removed = member & good[np.maximum(creator, 0)]
    if not np.any(removed):
        return None
    # Neighboring records restore the same facet; keep one row per facet key.
    kept = np.flatnonzero(good[records.vertices[family[owners]]])
    restored_keys, first = np.unique(facet_keys[kept], axis=0, return_index=True)
    return _Family(
        removed,
        family[good[records.vertices[family]]],
        restored_keys,
        facet_classes[kept[first]],
    )


def _undone(
    cells: Cells, flags: Flags, records: Records, family: _Family, /
) -> tuple[Cells, Flags, Records]:
    """Restore the parents of one coarsening pass in ascending cell-ID order."""

    undo = family.undo
    first_child = np.searchsorted(cells.ids, records.child_ids[undo, 0])
    restored = Cells(
        records.parent_ids[undo],
        records.rows[undo],
        records.tuples[undo],
        records.tags[undo],
        records.blocks[undo],
        records.parent_ids[undo],
        cells.generations[first_child] - 1,
    )
    restored_flags = Flags(
        np.ones(undo.shape, dtype=np.bool_),
        np.zeros(undo.shape, dtype=np.bool_),
        flags.classes[first_child],
    )
    merged = joined(select(cells, ~family.removed), restored)
    merged_flags = joined(select(flags, ~family.removed), restored_flags)
    order = np.argsort(merged.ids, kind="stable")
    keep = np.ones(records.parent_ids.shape, dtype=np.bool_)
    keep[undo] = False
    return select(merged, order), select(merged_flags, order), select(records, keep)


def reverse_supports(
    steps: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    vertex_count: int,
    dimension: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Surviving vertices spanning each source vertex and its reference weights.

    A removed vertex is the reference midpoint of its bisection edge, so its
    weights compose half of each endpoint's weights, level by level.
    """

    width = dimension + 1
    supports = np.full((vertex_count, width), -1, dtype=np.int64)
    supports[:, 0] = np.arange(vertex_count)
    weights = np.zeros((vertex_count, width), dtype=np.float64)
    weights[:, 0] = 1.0
    for vertices, first, second in reversed(steps):
        unique, index = np.unique(vertices, return_index=True)
        spans, counts = _distinct_rows(
            np.concatenate((supports[first[index]], supports[second[index]]), axis=1)
        )
        if np.any(counts > width):
            raise ValueError(
                f"Bisection.coarsening_supports must fit within one target simplex; got {(counts.max(initial=0), width)!r}"
            )
        merged, merged_weights = _merged_rows(
            np.concatenate((supports[first[index]], supports[second[index]]), axis=1),
            0.5 * np.concatenate((weights[first[index]], weights[second[index]]), axis=1),
            width,
        )
        if not np.array_equal(merged, spans[:, :width]):
            raise RuntimeError(
                f"Bisection.coarsening_weights must agree with the reconstructed vertex supports; got {(merged, spans[:, :width])!r}"
            )
        supports[unique] = merged
        weights[unique] = merged_weights
    return supports, weights


def _uniform_vertex_positions(
    source: Source, start: Start, values: np.ndarray, /
) -> np.ndarray:
    positions = np.searchsorted(source.vertex_ids, values)
    original = (positions < source.vertex_ids.size) & (
        source.vertex_ids[np.minimum(positions, source.vertex_ids.size - 1)] == values
    )
    issued = values - start.next_vertex
    if np.any(~original & ((issued < 0) | (issued >= start.front.growth.levels.size))):
        raise ValueError(
            f"BisectionUniformRefinement.child_vertices must reference admitted scientific vertices; got {(values, start.next_vertex, start.front.growth.levels.size)!r}"
        )
    return np.where(original, positions, source.vertex_ids.size + issued)


def _uniform_facet_agreement(
    source: Source,
    start: Start,
    arrays: dict[str, np.ndarray],
    parent: int,
    facets: tuple[np.ndarray, np.ndarray],
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    dimension, width = source.dimension, source.dimension + 1
    columns = tuple(combinations(range(width), dimension))
    vertices = arrays["child_vertices"][parent]
    barycentric = arrays["barycentric_weights"][parent]
    for child, row in enumerate(vertices):
        for local in columns:
            uniform_count(width * len(local), 0, budget=budget)
            key = np.sort(_uniform_vertex_positions(source, start, row[list(local)]))[
                None
            ]
            slot = entity_keys_owner.key_rows(facets[0], key)[0]
            if slot < 0:
                raise ValueError(
                    f"BisectionUniformRefinement.child_vertices must retain every source facet occurrence; got {(parent, child, local, slot)!r}"
                )
            original = [
                facet
                for facet, corner_indices in enumerate(columns)
                if np.all(
                    barycentric[
                        child,
                        list(local),
                        next(
                            index for index in range(width) if index not in corner_indices
                        ),
                    ]
                    == 0.0
                )
            ]
            if len(original) > 1:
                raise ValueError(
                    f"BisectionUniformRefinement.reference_vertices must have nondegenerate source facet incidence; got {(parent, child, local, original)!r}"
                )
            if original:
                if facets[1][slot] != arrays["parent_facet_classes"][parent, original[0]]:
                    return False
            elif facets[1][slot] not in (-1, 0):
                # A newly organized interior facet is scientific source data,
                # not an unlabelled subdivision seam that may disappear.
                return False
    return True


def _uniform_reverse_supports(
    source: Source,
    start: Start,
    arrays: dict[str, np.ndarray],
    selected: np.ndarray,
    coarsening: Coarsening,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, width = source.vertex_ids.size, source.dimension + 1
    uniform_count(count * width, 2 * count * width * 16, budget=budget)
    replacements = np.full((count, width), -1, dtype=np.int64)
    replacements[:, 0] = np.arange(count, dtype=np.int64)
    replacement_weights = np.zeros((count, width), dtype=np.float64)
    replacement_weights[:, 0] = 1.0
    discarded = []
    assigned: set[int] = set()
    for parent in selected:
        original = _uniform_vertex_positions(source, start, arrays["parent_rows"][parent])
        for vertices, barycentric in zip(
            arrays["child_vertices"][parent],
            arrays["barycentric_weights"][parent],
            strict=True,
        ):
            positions = _uniform_vertex_positions(source, start, vertices)
            for position, weights in zip(positions, barycentric, strict=True):
                uniform_count(width, 0, budget=budget)
                if position >= count:
                    discarded.append(int(position))
                    continue
                valid = weights != 0.0
                ids = original[valid]
                values = weights[valid]
                order = np.argsort(ids, kind="stable")
                packed_ids = np.full((width,), -1, dtype=np.int64)
                packed_weights = np.zeros((width,), dtype=np.float64)
                packed_ids[: ids.size], packed_weights[: ids.size] = (
                    ids[order],
                    values[order],
                )
                if int(position) in assigned:
                    if not np.array_equal(
                        replacements[position], packed_ids
                    ) or not np.array_equal(
                        replacement_weights[position], packed_weights
                    ):
                        raise ValueError(
                            f"BisectionUniformRefinement.barycentric_weights must agree on source-vertex actions at incident roots; got {(position, replacements[position], packed_ids, replacement_weights[position], packed_weights)!r}"
                        )
                replacements[position], replacement_weights[position] = (
                    packed_ids,
                    packed_weights,
                )
                assigned.add(int(position))
                if int(position) not in original:
                    discarded.append(int(position))
    support = coarsening.supports
    uniform_count(count * width * width, 4 * count * width * width * 16, budget=budget)
    expanded = replacements[np.maximum(support, 0)]
    coefficients = (
        coarsening.support_weights[..., None]
        * replacement_weights[np.maximum(support, 0)]
    )
    expanded = np.where((support >= 0)[..., None], expanded, -1)
    coefficients = np.where((support >= 0)[..., None], coefficients, 0.0)
    sources, weights = _merged_rows(
        expanded.reshape((count, width * width)),
        coefficients.reshape((count, width * width)),
        width,
    )
    return sources, weights, np.unique(np.asarray(discarded, dtype=np.int64))


@checked
def uniform_inverse(
    source: Source,
    start: Start,
    request: Request,
    cells: Cells,
    flags: Flags,
    facets: tuple[np.ndarray, np.ndarray],
    coarsening: Coarsening,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Coarsening:

    # This one execution-only import avoids mixed -> periodic -> BisectionEvidence
    # eagerly loading a partially initialized bisection record class.
    import phydrax.meshing._mixed_adaptation as _owner_phydrax_meshing__mixed_adaptation

    lineage = start.uniform_refinement
    if lineage is None or request.coarsen_ids.size == 0:
        return coarsening
    uniform_retain_source(lineage.source, budget=budget)
    arrays = lineage.host_arrays()
    kind = "triangle" if source.dimension == 2 else "tetrahedron"
    uniform_count(cells.ids.size, 512 * cells.ids.size, budget=budget)
    _uniform_cavity(int(np.count_nonzero(flags.marked)), budget=budget)
    child_blocks = {
        int(child): int(block)
        for children, block in zip(
            arrays["child_ids"], arrays["parent_blocks"], strict=True
        )
        for child in children
    }
    active = {int(identifier): index for index, identifier in enumerate(cells.ids)}
    records = []
    # Packing rows into a MixedSiblingBank allocates, per row, at most 1 KiB of
    # row/column objects, 128 bytes per integer slot and 96 bytes per reference
    # coordinate (exact-bits key, transient binary64 table, dictionary float).
    bank_bytes = 0
    for parent, identifier in enumerate(arrays["parent_ids"]):
        children = arrays["child_ids"][parent]
        uniform_count(children.size, 128 * children.size, budget=budget)
        if not all(int(child) in active for child in children):
            continue
        slots = np.asarray([active[int(child)] for child in children], dtype=np.int64)
        if not np.all(flags.marked[slots]) or np.any(flags.blocked[slots]):
            continue
        if not _uniform_facet_agreement(
            source, start, arrays, parent, facets, scope=scope, budget=budget
        ):
            continue
        references = []
        for child, vertices, reference in zip(
            children,
            arrays["child_vertices"][parent],
            arrays["reference_vertices"][parent],
            strict=True,
        ):
            uniform_count(
                source.dimension + 1, 256 * (source.dimension + 1), budget=budget
            )
            original_block = lineage.source.mesh.blocks[
                int(arrays["parent_blocks"][parent])
            ]
            if (
                source.mesh.blocks[int(cells.blocks[active[int(child)]])].cell_kind
                != original_block.cell_kind
            ):
                raise ValueError(
                    f"BisectionUniformRefinement.child_vertices must retain the recorded scientific child cell kind; got {(child, source.mesh.blocks[int(cells.blocks[active[int(child)]])].cell_kind, original_block.cell_kind)!r}"
                )
            current = global_vertices(
                cells.rows[active[int(child)]], source, start.next_vertex
            )
            if not np.array_equal(np.sort(current), np.sort(vertices)):
                raise ValueError(
                    f"BisectionUniformRefinement.child_vertices must retain the recorded scientific child incidence; got {(child, current, vertices)!r}"
                )
            positions = {int(vertex): index for index, vertex in enumerate(vertices)}
            references.append(
                tuple(
                    tuple(float(value) for value in reference[positions[int(vertex)]])
                    for vertex in current
                )
            )
        records.append(
            _owner_phydrax_meshing__mixed_adaptation.Sibling(
                int(identifier),
                lineage.source.mesh.blocks[int(arrays["parent_blocks"][parent])].name,
                kind,
                tuple(arrays["parent_rows"][parent].tolist()),
                tuple(children.tolist()),
                tuple(references),
                int(arrays["parent_classes"][parent]),
                None,
                None,
                (),
                (),
            )
        )
        bank_bytes += (
            1024
            + 128 * (source.dimension + 1 + children.size)
            + 96 * children.size * (source.dimension + 1) * source.dimension
        )
    uniform_count(cells.rows.size, 128 * cells.rows.size, budget=budget)
    current = [
        _owner_phydrax_meshing__mixed_adaptation.Cell(
            int(identifier),
            lineage.source.mesh.blocks[child_blocks[int(identifier)]].name
            if int(identifier) in child_blocks
            else source.mesh.blocks[int(cells.blocks[index])].name,
            kind,
            global_vertices(cells.rows[index], source, start.next_vertex),
            int(flags.classes[index]),
        )
        for index, identifier in enumerate(cells.ids)
    ]
    protected = set(
        global_vertices(
            np.flatnonzero(request.protected_vertices), source, start.next_vertex
        ).tolist()
    )
    protected.update(
        global_vertices(
            request.protected_codes // SHIFT, source, start.next_vertex
        ).tolist()
    )
    protected.update(
        global_vertices(
            request.protected_codes % SHIFT, source, start.next_vertex
        ).tolist()
    )
    uniform_count(0, bank_bytes, budget=budget)
    # Uniform simplex ancestry has no layer-column cohorts and does not author
    # the mixed hierarchy's unchanged-cell scientific signatures.
    patch = _owner_phydrax_meshing__mixed_adaptation.coarsen_patch(
        current,
        _owner_phydrax_meshing__mixed_adaptation.MixedSiblingBank(tuple(records)),
        set(cells.ids[flags.marked].tolist()),
        protected,
        set(cells.ids[flags.blocked].tolist()),
        source.vertex_ids,
        column_members={},
        unchanged=set(),
        scope=scope,
        budget=budget,
    )
    if not patch.restored:
        return coarsening
    restored_ids = np.asarray(
        [cell.identifier for cell in patch.restored], dtype=np.int64
    )
    selected = np.searchsorted(arrays["parent_ids"], restored_ids)
    restored = Cells(
        restored_ids,
        _uniform_vertex_positions(source, start, arrays["parent_rows"][selected]),
        _uniform_vertex_positions(source, start, arrays["parent_vertices"][selected]),
        arrays["parent_tags"][selected],
        arrays["parent_blocks"][selected],
        restored_ids,
        arrays["parent_levels"][selected],
    )
    retained = select(
        coarsening.restored,
        ~np.isin(
            coarsening.restored.ids, np.asarray(tuple(patch.removed), dtype=np.int64)
        ),
    )
    links = {
        child: parent for child, parent in zip(patch.fine, patch.parents, strict=True)
    }
    extra = source.cells.ids[
        np.isin(source.cells.ids, np.asarray(tuple(patch.removed), dtype=np.int64))
    ]
    removed = np.concatenate((coarsening.removed_ids, extra))
    targets = np.asarray(
        [links.get(int(target), int(target)) for target in coarsening.link_targets]
        + [links[int(child)] for child in extra],
        dtype=np.int64,
    )
    order = np.argsort(removed, kind="stable")
    supports, weights, discarded = _uniform_reverse_supports(
        source, start, arrays, selected, coarsening, scope=scope, budget=budget
    )
    return Coarsening(
        joined(retained, restored),
        removed[order],
        targets[order],
        np.concatenate((coarsening.undone, restored_ids)),
        np.union1d(coarsening.removed_vertices, discarded),
        supports,
        weights,
        coarsening.passes + 1,
        np.setdiff1d(
            coarsening.rejected_ids,
            np.asarray(tuple(patch.removed), dtype=np.int64),
        ),
    )


@checked
def coarsening(
    source: Source,
    start: Start,
    request: Request,
    blocked: np.ndarray,
    dimension: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Coarsening:
    cells, records = start.front.cells, start.records
    flags = Flags(np.isin(cells.ids, request.coarsen_ids), blocked, request.cell_classes)
    facets = (request.facet_keys, request.facet_classes)
    steps, children, parents, undone = [], [], [], []
    while request.coarsen_ids.size:
        family = _removable_family(
            cells, flags, records, request.protected_vertices, facets, dimension
        )
        if family is None:
            break
        undo = family.undo
        tuples, tags = records.tuples[undo], records.tags[undo]
        steps.append(
            (records.vertices[undo], tuples[:, 0], tuples[np.arange(undo.size), tags])
        )
        children.append(records.child_ids[undo].reshape((-1,)))
        parents.append(np.repeat(records.parent_ids[undo], 2))
        undone.append(records.parent_ids[undo])
        facets = (
            np.concatenate((facets[0], family.facet_keys)),
            np.concatenate((facets[1], family.facet_classes)),
        )
        cells, flags, records = _undone(cells, flags, records, family)
    source_ids = start.front.cells.ids
    removed_ids = source_ids[~_members(cells.ids, source_ids)]
    link_children = np.concatenate([np.zeros((0,), dtype=np.int64), *children])
    link_parents = np.concatenate([np.zeros((0,), dtype=np.int64), *parents])
    order = np.argsort(link_children, kind="stable")
    link_children, link_parents = link_children[order], link_parents[order]
    targets = removed_ids.copy()
    for _ in steps:
        linked = _members(link_children, targets)
        targets[linked] = link_parents[np.searchsorted(link_children, targets[linked])]
    removed_vertices = np.unique(
        np.concatenate([np.zeros((0,), dtype=np.int64), *(step[0] for step in steps)])
    )
    coarsening = Coarsening(
        select(cells, ~_members(source_ids, cells.ids)),
        removed_ids,
        targets,
        np.concatenate([np.zeros((0,), dtype=np.int64), *undone]),
        removed_vertices,
        *reverse_supports(steps, start.front.vertex_base, dimension),
        len(steps),
        request.coarsen_ids[_members(cells.ids, request.coarsen_ids)],
    )
    return uniform_inverse(
        source,
        start,
        request,
        cells,
        flags,
        facets,
        coarsening,
        scope=scope,
        budget=budget,
    )


def resolved_stencil(
    growth: Growth, vertex_base: int, dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Source-vertex stencil of every new vertex, composed level by level."""

    width = dimension + 1
    count = growth.levels.size
    sources = np.full((count, width), -1, dtype=np.int64)
    weights = np.zeros((count, width), dtype=np.float64)
    for level in np.unique(growth.levels):
        chosen = np.flatnonzero(growth.levels == level)
        parents = growth.parents[chosen]
        coefficients = growth.weights[chosen]
        derived = parents >= vertex_base
        index = np.maximum(parents - vertex_base, 0)
        direct_sources = np.full(parents.shape + (width,), -1, dtype=np.int64)
        direct_sources[..., 0] = parents
        direct_weights = np.zeros(parents.shape + (width,), dtype=np.float64)
        direct_weights[..., 0] = 1.0
        parent_sources = np.where(derived[..., None], sources[index], direct_sources)
        parent_weights = np.where(derived[..., None], weights[index], direct_weights)
        parent_sources = np.where((parents >= 0)[..., None], parent_sources, -1)
        sources[chosen], weights[chosen] = _merged_rows(
            parent_sources.reshape((chosen.size, -1)),
            (parent_weights * coefficients[..., None]).reshape((chosen.size, -1)),
            width,
        )
    return sources, weights


def _nondyadic_growth(growth: Growth, /) -> bool:
    """Whether any authored construction weight is not a power of two."""
    weights = growth.weights[growth.parents >= 0]
    return bool(np.any(np.frexp(weights)[0] != 0.5))


def _exact_stencil(
    growth: Growth,
    vertex_base: int,
    sources: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Integer homogeneous rows and correctly rounded weights of every new vertex.

    Each construction averages its ``m`` parents with the authored weight
    ``1/m`` (midpoints, face and cell centroids), composed exactly level by
    level over the source columns of ``resolved_stencil``. Rows ``h`` satisfy
    ``h / sum(h)`` equal to the exact barycentric stencil; the returned weights
    are its binary64 rounding.
    """

    count, width = sources.shape
    uniform_count(count * width * width, 192 * count * width, budget=budget)
    rows: list[dict[int, Fraction]] = [{} for _ in range(count)]
    for level in np.unique(growth.levels).tolist():
        for vertex in np.flatnonzero(growth.levels == level).tolist():
            parents = growth.parents[vertex]
            valid = parents >= 0
            arity = int(np.count_nonzero(valid))
            row: dict[int, Fraction] = {}
            for parent, weight in zip(
                parents[valid].tolist(), growth.weights[vertex][valid].tolist(), strict=True
            ):
                exact = Fraction(1, arity) if weight == 1.0 / arity else Fraction(weight)
                if parent >= vertex_base:
                    for column, value in rows[parent - vertex_base].items():
                        row[column] = row.get(column, Fraction(0)) + exact * value
                else:
                    row[parent] = row.get(parent, Fraction(0)) + exact
            rows[vertex] = row
    homogeneous = np.zeros((count, width), dtype=np.float64)
    weights = np.zeros((count, width), dtype=np.float64)
    for vertex, row in enumerate(rows):
        columns = sources[vertex]
        if sum(row.values(), Fraction(0)) != 1 or sorted(row) != sorted(
            columns[columns >= 0].tolist()
        ):
            raise ValueError(
                f"Bisection.vertex_stencil must be an exact barycentric source row; got vertex {vertex!r}"
            )
        scale = math.lcm(*(value.denominator for value in row.values()))
        if scale > 2**53:
            raise ValueError(
                f"Bisection.vertex_stencil must have a binary64 homogeneous row; got denominator {scale!r}"
            )
        for slot, column in enumerate(columns.tolist()):
            if column >= 0:
                homogeneous[vertex, slot] = float(row[column] * scale)
                weights[vertex, slot] = float(row[column])
    return homogeneous, weights


def global_vertices(
    compact: np.ndarray, source: Source, next_vertex: int, /
) -> np.ndarray:
    count = source.vertex_ids.size
    return np.where(
        compact < count,
        source.vertex_ids[np.clip(compact, 0, count - 1)],
        next_vertex + compact - count,
    )


def _contained(
    keys_from: np.ndarray, supports: np.ndarray, keys_to: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Entities of one mesh lying inside one same-dimension entity of the other."""

    width = keys_from.shape[1]
    missing = keys_from[entity_keys_owner.key_rows(keys_to, keys_from) < 0]
    spans, counts = _distinct_rows(
        supports[missing].reshape((missing.shape[0], width * supports.shape[1]))
    )
    candidates = spans[:, :width]
    found = np.zeros(counts.shape, dtype=np.bool_)
    single = np.flatnonzero(counts == width)
    found[single] = entity_keys_owner.key_rows(keys_to, candidates[single]) >= 0
    return missing[found], candidates[found]


def _grouped_relations(
    degree: int,
    groups: tuple[
        tuple[
            np.ndarray, np.ndarray, _dependency_phydrax_meshing__lineage.EntityLineageKind
        ],
        ...,
    ],
    /,
) -> _dependency_phydrax_meshing__topology_edit.EntityRelations:
    """One relation record from (source keys, target keys, kind) groups."""

    return _dependency_phydrax_meshing__topology_edit.EntityRelations(
        degree,
        np.concatenate(tuple(group[0] for group in groups)).astype(np.int64),
        np.concatenate(tuple(group[1] for group in groups)).astype(np.int64),
        np.concatenate(
            tuple(
                np.full((group[0].shape[0],), int(group[2]), dtype=np.int32)
                for group in groups
            )
        ),
    )


def relations(
    source: Source,
    start: Start,
    cells: Cells,
    stencil: tuple[np.ndarray, np.ndarray],
    coarsening: Coarsening,
    entity_tables: tuple[tuple[np.ndarray, np.ndarray], ...],
    /,
) -> tuple[_dependency_phydrax_meshing__topology_edit.EntityRelations, ...]:
    """Non-identity lineage of every dimension; identities stay implicit."""

    dimension, count = source.dimension, source.vertex_ids.size

    def ids(values: np.ndarray) -> np.ndarray:
        return global_vertices(values, source, start.next_vertex)

    split_sources, split_weights = stencil
    owner, column = np.nonzero((split_sources >= 0) & (split_weights > 0.0))
    records = [
        _grouped_relations(
            0,
            (
                (
                    source.vertex_ids[split_sources[owner, column]][:, None],
                    ids(count + owner)[:, None],
                    _dependency_phydrax_meshing__lineage.EntityLineageKind.SPLIT_FROM,
                ),
            ),
        )
    ]
    forward = np.full((count + split_sources.shape[0], dimension + 1), -1, np.int64)
    forward[:count, 0] = np.arange(count)
    forward[count:] = split_sources
    for degree, (source_keys, target_keys) in enumerate(entity_tables, start=1):
        refined, within = _contained(target_keys, forward, source_keys)
        merged, into = _contained(source_keys, coarsening.supports, target_keys)
        records.append(
            _grouped_relations(
                degree,
                (
                    (
                        ids(within),
                        ids(refined),
                        _dependency_phydrax_meshing__lineage.EntityLineageKind.REFINED_FROM,
                    ),
                    (
                        ids(merged),
                        ids(into),
                        _dependency_phydrax_meshing__lineage.EntityLineageKind.COARSENED_INTO,
                    ),
                ),
            )
        )
    new_cells = ~_members(source.cells.ids, cells.ids) & (cells.origins != cells.ids)
    records.append(
        _grouped_relations(
            dimension,
            (
                (
                    cells.origins[new_cells][:, None],
                    cells.ids[new_cells][:, None],
                    _dependency_phydrax_meshing__lineage.EntityLineageKind.REFINED_FROM,
                ),
                (
                    coarsening.removed_ids[:, None],
                    coarsening.link_targets[:, None],
                    _dependency_phydrax_meshing__lineage.EntityLineageKind.COARSENED_INTO,
                ),
            ),
        )
    )
    return tuple(records)


def _entity_identities(
    source: Source,
    start: Start,
    entity_tables: tuple[tuple[np.ndarray, np.ndarray], ...],
    alive: np.ndarray,
    /,
) -> tuple[
    tuple[_dependency_phydrax_meshing__topology_edit.PrescribedEntityIds, ...],
    tuple[tuple[np.ndarray, np.ndarray], ...],
]:
    """Restored entities regain retired IDs; newly retired entities are recorded.

    Only entities whose vertices all survive can reappear, so retired entries
    touching a removed vertex are dropped.
    """

    prescribed, retired = [], []
    for degree, (keys, targets) in enumerate(entity_tables, start=1):
        old_keys, old_ids = start.retired[degree - 1]
        created = targets[entity_keys_owner.key_rows(keys, targets) < 0]
        restore = entity_keys_owner.key_rows(old_keys, created)
        restored = restore >= 0
        prescribed.append(
            _dependency_phydrax_meshing__topology_edit.PrescribedEntityIds(
                degree,
                global_vertices(created[restored], source, start.next_vertex),
                old_ids[restore[restored]],
            )
        )
        keep = np.all(alive[old_keys], axis=1)
        keep[restore[restored]] = False
        vanished = np.flatnonzero(entity_keys_owner.key_rows(targets, keys) < 0)
        vanished = vanished[np.all(alive[keys[vanished]], axis=1)]
        entity_ids = np.asarray(source.mesh.entity_set(degree).entity_ids, dtype=np.int64)
        table_keys = np.concatenate((old_keys[keep], keys[vanished]))
        table_ids = np.concatenate((old_ids[keep], entity_ids[vanished]))
        order = np.lexsort(table_keys.T[::-1])
        retired.append((source.vertex_ids[table_keys[order]], table_ids[order]))
    return tuple(prescribed), tuple(retired)


@checked
def merged_forest(
    start: Start, refinement: Refinement, coarsening: Coarsening
) -> tuple[Cells, Records]:
    cells = refinement.front.cells
    cells = joined(
        select(cells, ~_members(coarsening.removed_ids, cells.ids)),
        coarsening.restored,
    )
    records = start.records
    records = joined(
        select(records, ~np.isin(records.parent_ids, coarsening.undone)),
        refinement.records,
    )
    return (
        select(cells, np.argsort(cells.ids, kind="stable")),
        select(records, np.argsort(records.parent_ids, kind="stable")),
    )


@checked
def target_hierarchy(
    source: Source,
    start: Start,
    cells: Cells,
    records: Records,
    retired: tuple[tuple[np.ndarray, np.ndarray], ...],
    front: Front,
    /,
    *,
    refinement_rule: bisection_records.BisectionRefinementRule,
    prior: bisection_records.BisectionHierarchy | None = None,
    source_authority: tuple[np.ndarray, np.ndarray] | None = None,
) -> bisection_records.BisectionHierarchy:
    def ids(values: np.ndarray) -> np.ndarray:
        return global_vertices(values, source, start.next_vertex)

    uniform = start.uniform_refinement
    if uniform is not None and np.all(
        np.isin(
            np.asarray(jax.device_get(uniform.parent_ids)),
            cells.ids,
        )
    ):
        uniform = None
    if prior is not None:
        source_ids, source_blocks = jax.device_get(
            (prior.scientific_cell_ids, prior.scientific_block_ids)
        )
        authority = {
            int(identifier): int(block)
            for identifier, block in zip(source_ids, source_blocks, strict=True)
        }
    else:
        source_ids, source_blocks = (
            (source.cells.ids, source.cells.blocks)
            if source_authority is None
            else source_authority
        )
        authority = {
            int(identifier): int(block)
            for identifier, block in zip(source_ids, source_blocks, strict=True)
        }
    if start.uniform_refinement is not None:
        lineage = start.uniform_refinement
        parent_ids, child_ids = jax.device_get((lineage.parent_ids, lineage.child_ids))
        for parent, children in zip(parent_ids, child_ids, strict=True):
            parent_id = int(parent)
            if parent_id not in authority:
                raise ValueError(
                    f"BisectionHierarchy.scientific_block_ids must retain source block ownership for uniform parents; got {parent_id!r}"
                )
            for child in children:
                authority[int(child)] = authority[parent_id]
    for parent, children in zip(records.parent_ids, records.child_ids, strict=True):
        parent_id = int(parent)
        if parent_id not in authority:
            raise ValueError(
                f"BisectionHierarchy.scientific_block_ids must retain block ownership for bisection parents; got {parent_id!r}"
            )
        for child in children:
            authority[int(child)] = authority[parent_id]
    scientific_ids = np.asarray(sorted(authority), dtype=np.int64)
    scientific_blocks = np.asarray(
        [authority[int(identifier)] for identifier in scientific_ids], dtype=np.int32
    )

    return bisection_records.BisectionHierarchy(
        source.dimension,
        cells.ids,
        ids(cells.tuples),
        cells.tags,
        cells.generations,
        scientific_cell_ids=scientific_ids,
        scientific_block_ids=scientific_blocks,
        record_parent_ids=records.parent_ids,
        record_parent_blocks=records.blocks,
        record_parent_rows=ids(records.rows),
        record_parent_vertices=ids(records.tuples),
        record_parent_tags=records.tags,
        record_child_ids=records.child_ids,
        record_vertex_ids=ids(records.vertices),
        retired_entity_keys=tuple(value[0] for value in retired),
        retired_entity_ids=tuple(value[1] for value in retired),
        next_vertex_id=start.next_vertex + front.growth.levels.size,
        next_cell_id=front.next_cell,
        uniform_refinement=uniform,
        refinement_rule=refinement_rule,
    )


def _target_vertices(
    source: Source, removed: np.ndarray, new_count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Surviving source vertices in source row order, then surviving new vertices.

    ``removed`` may hold new vertices (created and coarsened away inside one
    device epoch); they are absent from the target like removed source vertices.
    """

    count = source.vertex_ids.size
    alive = np.ones((count + new_count,), dtype=np.bool_)
    alive[removed] = False
    compact = np.empty((count,), dtype=np.int64)
    compact[source.original] = np.arange(count)
    ordered = np.concatenate(
        (compact[alive[compact]], count + np.flatnonzero(alive[count:]))
    )
    position = np.full(alive.shape, -1, dtype=np.int64)
    position[ordered] = np.arange(ordered.size)
    return ordered, position, alive


def _target_stencil(
    source: Source, stencil: tuple[np.ndarray, np.ndarray], ordered: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates and source stencil rows of the ordered target vertices."""

    count, width = source.vertex_ids.size, source.dimension + 1
    new_sources, new_weights = stencil
    new_coordinates = np.sum(
        np.where(
            (new_sources >= 0)[..., None],
            source.coordinates[np.maximum(new_sources, 0)] * new_weights[..., None],
            0.0,
        ),
        axis=1,
    )
    one_hot = np.full((count, width), -1, dtype=np.int64)
    one_hot[:, 0] = np.arange(count)
    sources = np.concatenate((one_hot, new_sources))[ordered]
    weights = np.concatenate(
        (np.eye(1, width, dtype=np.float64).repeat(count, axis=0), new_weights)
    )[ordered]
    valid = sources >= 0
    return (
        np.concatenate((source.coordinates, new_coordinates))[ordered],
        np.where(valid, source.vertex_ids[np.maximum(sources, 0)], -1),
        np.where(valid, weights, 0.0),
        valid,
    )


@checked
def build_bisection_edit(
    source: Source,
    start: Start,
    refinement: Refinement,
    coarsening: Coarsening,
    cells: Cells,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    tuple[tuple[np.ndarray, np.ndarray], ...],
]:
    dimension, count = source.dimension, source.vertex_ids.size
    growth = refinement.front.growth
    if start.uniform_refinement is not None:
        uniform_count(
            cells.rows.size + source.cells.rows.size + growth.parents.size,
            cells.rows.nbytes * 8
            + source.cells.rows.nbytes * 8
            + growth.parents.nbytes * 8,
            budget=budget,
        )
    stencil = resolved_stencil(growth, count, dimension)
    exact = None
    if _nondyadic_growth(growth):
        # Centroid weights are not binary64: retain exact homogeneous rows so
        # nested witnesses carry the rational reference bank, not roundings.
        exact, rounded = _exact_stencil(growth, count, stencil[0], budget=budget)
        stencil = (stencil[0], rounded)
    ordered, position, alive = _target_vertices(
        source, coarsening.removed_vertices, growth.levels.size
    )
    # Removed new vertices keep their stencil for composition but relate nothing.
    stencil = (np.where(alive[count:, None], stencil[0], -1), stencil[1])
    coordinates, sources, weights, valid = _target_stencil(source, stencil, ordered)
    stencil_homogeneous = None
    if exact is not None:
        identity = np.zeros((count, dimension + 1), dtype=np.float64)
        identity[:, 0] = 1.0
        stencil_homogeneous = np.where(
            valid, np.concatenate((identity, exact))[ordered], 0.0
        )
    entity_tables = tuple(
        (
            np.searchsorted(
                source.vertex_ids,
                entity_keys_owner.entity_keys(source.mesh, degree),
            ),
            simplex_keys(cells.rows, degree + 1),
        )
        for degree in range(1, dimension)
    )
    prescribed, retired = _entity_identities(source, start, entity_tables, alive)
    block_definitions = tuple(
        (block.name, block.cell_kind) for block in source.mesh.blocks
    )
    block_rows = tuple(
        np.flatnonzero(cells.blocks == index) for index in range(len(source.mesh.blocks))
    )
    uniform = start.uniform_refinement
    if uniform is not None:
        original_ids = np.asarray(uniform.parent_ids)
        restored = np.isin(cells.ids, original_ids)
        if np.any(restored):
            original_blocks = np.asarray(uniform.parent_blocks)[
                np.searchsorted(original_ids, cells.ids[restored])
            ]
            definitions = list(block_definitions)
            rows = [
                np.flatnonzero((cells.blocks == index) & ~restored)
                for index in range(len(definitions))
            ]
            restored_rows = np.flatnonzero(restored)
            for index, block in enumerate(uniform.source.mesh.blocks):
                group = restored_rows[original_blocks == index]
                if group.size == 0:
                    continue
                parent_positions = np.searchsorted(original_ids, cells.ids[group])
                group = group[
                    np.argsort(
                        np.asarray(uniform.parent_row_order)[parent_positions],
                        kind="stable",
                    )
                ]
                definition = (block.name, block.cell_kind)
                if definition in definitions:
                    slot = definitions.index(definition)
                    rows[slot] = np.concatenate((rows[slot], group))
                else:
                    definitions.append(definition)
                    rows.append(group)
            block_definitions, block_rows = tuple(definitions), tuple(rows)
        if cells.ids.size == original_ids.size and np.array_equal(
            cells.ids, original_ids
        ):
            original_order = np.asarray(uniform.parent_row_order)
            block_definitions = tuple(
                (block.name, block.cell_kind) for block in uniform.source.mesh.blocks
            )
            block_rows = tuple(
                np.flatnonzero(
                    np.asarray(uniform.parent_blocks)[
                        np.searchsorted(original_ids, cells.ids)
                    ]
                    == index
                )
                for index in range(len(block_definitions))
            )
            block_rows = tuple(
                rows[
                    np.argsort(
                        original_order[np.searchsorted(original_ids, cells.ids[rows])],
                        kind="stable",
                    )
                ]
                for rows in block_rows
            )
    refined = bool(refinement.records.parent_ids.size) or start.uniform
    coarsened = bool(coarsening.removed_ids.size)
    refinement_witnesses, coarsening_witnesses = _nested_witnesses(
        source,
        cells,
        coarsening,
        (sources, weights),
        position,
        stencil_homogeneous,
        budget=budget,
    )
    edit = _dependency_phydrax_meshing__topology_edit.CellTopologyEdit(
        "nested_adaptation"
        if refined and coarsened
        else "nested_coarsening"
        if coarsened
        else "nested_refinement",
        coordinates,
        global_vertices(ordered, source, start.next_vertex),
        _dependency_phydrax_meshing__topology_edit.source_family_blocks(
            block_definitions,
            tuple(position[cells.rows[rows]].astype(np.int32) for rows in block_rows),
            tuple(cells.ids[rows] for rows in block_rows),
        ),
        sources,
        weights,
        valid,
        relations(source, start, cells, stencil, coarsening, entity_tables),
        prescribed,
        refinement=refinement_witnesses,
        coarsening=coarsening_witnesses,
        stencil_homogeneous=stencil_homogeneous,
    )
    return edit, retired


def _nested_witnesses(
    source: Source,
    cells: Cells,
    coarsening: Coarsening,
    stencil: tuple[np.ndarray, np.ndarray],
    position: np.ndarray,
    homogeneous: np.ndarray | None,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    transition_types_owner.NestedReferenceWitnesses,
    transition_types_owner.NestedReferenceWitnesses,
]:
    """Reference witnesses of every target cell of one bisection edit.

    A target cell restored by coarsening contains its removed source cells, whose
    vertices are reference-midpoint combinations of the restored corners; every
    other target cell lies in its origin source cell (itself when preserved), with
    corners given by their source-vertex stencils.
    """

    sources, weights = stencil
    restored = _members(np.unique(coarsening.link_targets), cells.ids)
    origins = cells.origins[~restored]
    parent_rows = source.cells.rows[np.searchsorted(source.cells.ids, origins)]
    fine = position[cells.rows[~restored]]
    coarse_rows = source.vertex_ids[parent_rows]
    if homogeneous is None:
        reference = _dependency_phydrax_meshing__topology_edit.nested_reference_vertices(
            sources[fine], weights[fine], coarse_rows
        )
        bank = None
    else:
        reference, bank = (
            _dependency_phydrax_meshing__topology_edit.exact_nested_reference_vertices(
                sources[fine], homogeneous[fine], coarse_rows, budget=budget
            )
        )
    refined = transition_types_owner.NestedReferenceWitnesses(
        cells.ids[~restored], origins, reference, bank
    )
    children = source.cells.rows[
        np.searchsorted(source.cells.ids, coarsening.removed_ids)
    ]
    coarse = cells.rows[np.searchsorted(cells.ids, coarsening.link_targets)]
    coarsened = transition_types_owner.NestedReferenceWitnesses(
        coarsening.removed_ids,
        coarsening.link_targets,
        _dependency_phydrax_meshing__topology_edit.nested_reference_vertices(
            coarsening.supports[children],
            coarsening.support_weights[children],
            coarse,
        ),
    )
    return refined, coarsened


def prepared_capacity(cells: int, vertices: int, work_units: int, /) -> _Capacity:
    return _Capacity(
        positive_integer(cells, "maximum_cells"),
        positive_integer(vertices, "maximum_vertices"),
        positive_integer(work_units, "maximum_work_units"),
    )
