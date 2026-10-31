#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic compatible bisection and coarsening of simplex meshes.

Triangle meshes (ambient dimension 2 or 3) and tetrahedral meshes are refined by
Maubach's tagged-simplex bisection under Stevenson's matching condition and
coarsened by removing bisection vertices whose star consists exactly of the
children of their bisections (Chen-Zhang vertex coarsening, generalized to
tetrahedral patches). All work is host NumPy preparation: every closure
iteration and every coarsening pass is one vectorized batch over cells.

A tagged simplex ``T = (x0, ..., xd)_k`` with ``k`` in ``1..d`` bisects its
refinement edge ``(x0, xk)`` at the midpoint ``z`` into
``T1 = (x0, ..., x_{k-1}, z, x_{k+1}, ..., xd)`` and
``T2 = (x1, ..., xk, z, x_{k+1}, ..., xd)``, both tagged ``k - 1`` (``d`` when
``k = 1``). ``T`` is the same tagged simplex as the tuple with its first
``k + 1`` entries reversed; the canonical form is the lexicographically smaller
tuple by vertex global ID. In two dimensions this is newest-vertex bisection.

Shape regularity: Maubach bisection generates only finitely many similarity
classes from each initial simplex (at most four per triangle in two dimensions,
a dimension-dependent bound for tetrahedra; Maubach 1995, Traxler 1997), so every
shape measure of every descendant is bounded below by its minimum over the
classes of the initial mesh. The matching condition (Stevenson 2008) makes the
conformity closure terminate with the optimal closure-size bound; initial
labellings that violate it are rejected or first made compatible by one
barycentric subdivision, whose simplices are pairwise reflected neighbors. That
subdivision is a recorded refinement level of the forest: coarsening removes it
once every created vertex's star is undone, restoring the source cells exactly.

Tetrahedral meshes whose longest-edge labels violate the matching condition are
otherwise labelled as the marked tetrahedra of Arnold, Mukherjee and Pouly
(2000): each face marks its longest edge under the global strict edge order, so
neighbors agree on every shared face and the closure terminates on any
conforming mesh. Planar, mixed and flagged-planar marked tetrahedra are Maubach
tags 2, 3 and 1; adjacent and opposite ones carry tags 4 and 5 (refinement edge
``(x0, x3)``), whose children are tag 2, so the descendants of every marked
tetrahedron are Maubach simplices with finitely many similarity classes.
"""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from itertools import combinations, permutations
from typing import Any, final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._adaptive_simplex import bisection_label_tables
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._lineage import EntityLineageKind
from ._periodic import PeriodicRefinementEdgeLock
from ._topology_edit import (
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    key_rows,
    PrescribedEntityIds,
    source_family_blocks,
)


class BisectionCompatibility(StrEnum):
    """Handling of an initial labelling that violates the matching condition.

    ``REJECT`` refuses an incompatible triangle labelling and relabels an
    incompatible tetrahedral one as marked tetrahedra, whose closure needs no
    matching condition. ``UNIFORM_REFINEMENT`` first subdivides every source cell
    barycentrically into a compatible mesh in both dimensions.
    """

    REJECT = "reject"
    UNIFORM_REFINEMENT = "uniform_refinement"


def _identifiers(values: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1 or np.any(array < 0):
        raise ValueError(f"{name} must be a rank-1 array of non-negative IDs.")
    return array


def _tag_array(
    values: ArrayLike, count: int, dimension: int, name: str, /, *, lowest: int = 1
) -> np.ndarray:
    tags = np.asarray(values, dtype=np.int32)
    highest = _TABLES[dimension].child.size - 1
    if tags.shape != (count,) or np.any(tags < lowest) or np.any(tags > highest):
        raise ValueError(f"{name} must hold one tag in {lowest}..{highest} per simplex.")
    return tags


def _simplex_rows(values: ArrayLike, count: int, dimension: int, name: str, /) -> Any:
    rows = np.asarray(values, dtype=np.int64)
    if rows.shape != (count, dimension + 1) or np.any(rows < 0):
        raise ValueError(f"{name} must have shape ({count}, {dimension + 1}).")
    ordered = np.sort(rows, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError(f"{name} rows must reference distinct vertices.")
    return rows


def _retired_tables(keys: Any, ids: Any, dimension: int, /) -> Any:
    key_tables = tuple(np.asarray(value, dtype=np.int64) for value in keys)
    id_tables = tuple(np.asarray(value, dtype=np.int64) for value in ids)
    if len(key_tables) != dimension - 1 or len(id_tables) != dimension - 1:
        raise ValueError("Retired entity tables need one entry per dimension 1..d-1.")
    for degree, (table, identifiers) in enumerate(
        zip(key_tables, id_tables, strict=True), start=1
    ):
        if table.ndim != 2 or table.shape[1] != degree + 1:
            raise ValueError(f"Retired dimension-{degree} keys have the wrong width.")
        if identifiers.shape != (table.shape[0],) or np.any(identifiers < 0):
            raise ValueError("Retired entity IDs must align with their keys.")
        if np.any(table[:, 1:] <= table[:, :-1]):
            raise ValueError("Retired entity keys must be ascending vertex IDs.")
    return key_tables, id_tables


def _subdivision_width(dimension: int, /) -> tuple[int, int]:
    """Children and created vertices of one barycentric subdivision record."""

    return (6, 4) if dimension == 2 else (24, 11)


def _validated_forest(
    dimension: int,
    cell_global_ids: ArrayLike,
    ordered_vertices: ArrayLike,
    tags: ArrayLike,
    generations: ArrayLike,
    records: dict[str, ArrayLike],
    subdivisions: dict[str, ArrayLike],
    /,
) -> dict[str, np.ndarray]:
    ids = _identifiers(cell_global_ids, "cell_global_ids")
    if ids.size == 0 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("cell_global_ids must be non-empty and strictly increasing.")
    count = ids.size
    parents = _identifiers(records["parent_ids"], "record_parent_ids")
    if np.any(parents[1:] <= parents[:-1]) or np.intersect1d(parents, ids).size:
        raise ValueError(
            "Record parent IDs must be strictly increasing and inactive cells."
        )
    total = parents.size
    children = np.asarray(records["child_ids"], dtype=np.int64)
    blocks = np.asarray(records["parent_blocks"], dtype=np.int32)
    vertices = _identifiers(records["vertex_ids"], "record_vertex_ids")
    if children.shape != (total, 2) or np.any(children < 0):
        raise ValueError("record_child_ids must have shape (records, 2).")
    roots = _identifiers(subdivisions["parent_ids"], "subdivision_parent_ids")
    if (
        np.any(roots[1:] <= roots[:-1])
        or np.intersect1d(roots, ids).size
        or np.intersect1d(roots, parents).size
    ):
        raise ValueError(
            "Subdivision parent IDs must be strictly increasing inactive cells "
            "without bisection records."
        )
    arrangements, created = _subdivision_width(dimension)
    root_count = roots.size
    root_children = np.asarray(subdivisions["child_ids"], dtype=np.int64)
    root_blocks = np.asarray(subdivisions["parent_blocks"], dtype=np.int32)
    root_vertices = np.asarray(subdivisions["vertex_ids"], dtype=np.int64)
    if root_children.shape != (root_count, arrangements) or np.any(root_children < 0):
        raise ValueError(
            f"subdivision_child_ids must have shape (subdivisions, {arrangements})."
        )
    if root_vertices.shape != (root_count, created) or np.any(root_vertices < 0):
        raise ValueError(
            f"subdivision_vertex_ids must have shape (subdivisions, {created})."
        )
    if root_blocks.shape != (root_count,) or np.any(root_blocks < 0):
        raise ValueError("subdivision_parent_blocks must align with the subdivisions.")
    issued = np.concatenate((children.reshape((-1,)), root_children.reshape((-1,))))
    if np.unique(issued).size != issued.size:
        raise ValueError("Every refinement record owns distinct child IDs.")
    if blocks.shape != (total,) or np.any(blocks < 0) or vertices.shape != (total,):
        raise ValueError("Record blocks and vertices must align with the records.")
    generation = np.asarray(generations, dtype=np.int32)
    if generation.shape != (count,) or np.any(generation < 0):
        raise ValueError("generations must hold one non-negative value per cell.")
    return {
        "cell_global_ids": ids,
        "ordered_vertices": _simplex_rows(
            ordered_vertices, count, dimension, "ordered_vertices"
        ),
        "tags": _tag_array(tags, count, dimension, "tags", lowest=0),
        "generations": generation,
        "record_parent_ids": parents,
        "record_parent_blocks": blocks,
        "record_parent_rows": _simplex_rows(
            records["parent_rows"], total, dimension, "record_parent_rows"
        ),
        "record_parent_vertices": _simplex_rows(
            records["parent_vertices"], total, dimension, "record_parent_vertices"
        ),
        "record_parent_tags": _tag_array(
            records["parent_tags"], total, dimension, "record_parent_tags"
        ),
        "record_child_ids": children,
        "record_vertex_ids": vertices,
        "subdivision_parent_ids": roots,
        "subdivision_parent_blocks": root_blocks,
        "subdivision_parent_rows": _simplex_rows(
            subdivisions["parent_rows"],
            root_count,
            dimension,
            "subdivision_parent_rows",
        ),
        "subdivision_child_ids": root_children,
        "subdivision_vertex_ids": root_vertices,
    }


@final
class BisectionHierarchy(StrictModule, NonTrainableState):
    """Persistent bisection labels and refinement forest of one bisection mesh.

    Active cells are listed by ascending global ID with their canonical tagged
    tuple (vertex global IDs), tag (Maubach ``1..d``; marked-tetrahedron ``4``
    and ``5``), and generation. Each forest record is one
    bisection whose children are leaves or have descendants: the parent cell ID,
    source block, exact block row (global IDs), tagged tuple and tag, the two
    child cell IDs (first, second), and the bisection vertex. Retired entity
    tables keep the global IDs of edges/faces removed by refinement so that
    coarsening restores them under their original IDs. Vertex and cell IDs are
    issued from the high-water counters and never reused.

    Subdivision records are the barycentric pre-subdivision of a source cell whose
    labels violated the matching condition: the parent cell ID, block, exact block
    row, its ``(d + 1)!`` child IDs in arrangement order, and the created vertices
    of the subsets of the ascending parent vertex IDs (edges, then faces of a
    tetrahedron, then the cell center, each lexicographic). Coarsening removes a
    subdivision once every child and every created vertex's star is undone, which
    restores the source cell exactly. A restored cell carries tag ``0`` (its
    labels are incompatible) and its block row as ordered vertices; refining a
    hierarchy with such cells subdivides all of them again.

    A hierarchy applies to a mesh iff its active cell IDs equal the mesh cell IDs
    and every cell has the same vertex-ID set in both.
    """

    dimension: int = eqx.field(static=True)
    cell_global_ids: Array
    ordered_vertices: Array
    tags: Array
    generations: Array
    record_parent_ids: Array
    record_parent_blocks: Array
    record_parent_rows: Array
    record_parent_vertices: Array
    record_parent_tags: Array
    record_child_ids: Array
    record_vertex_ids: Array
    subdivision_parent_ids: Array
    subdivision_parent_blocks: Array
    subdivision_parent_rows: Array
    subdivision_child_ids: Array
    subdivision_vertex_ids: Array
    retired_entity_keys: tuple[Array, ...]
    retired_entity_ids: tuple[Array, ...]
    next_vertex_id: int = eqx.field(static=True)
    next_cell_id: int = eqx.field(static=True)
    hierarchy_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        cell_global_ids: ArrayLike,
        ordered_vertices: ArrayLike,
        tags: ArrayLike,
        generations: ArrayLike,
        /,
        *,
        record_parent_ids: ArrayLike,
        record_parent_blocks: ArrayLike,
        record_parent_rows: ArrayLike,
        record_parent_vertices: ArrayLike,
        record_parent_tags: ArrayLike,
        record_child_ids: ArrayLike,
        record_vertex_ids: ArrayLike,
        subdivision_parent_ids: ArrayLike,
        subdivision_parent_blocks: ArrayLike,
        subdivision_parent_rows: ArrayLike,
        subdivision_child_ids: ArrayLike,
        subdivision_vertex_ids: ArrayLike,
        retired_entity_keys: tuple[ArrayLike, ...],
        retired_entity_ids: tuple[ArrayLike, ...],
        next_vertex_id: int,
        next_cell_id: int,
    ) -> None:
        if isinstance(dimension, bool) or dimension not in (2, 3):
            raise ValueError("Bisection hierarchies are two- or three-dimensional.")
        arrays = _validated_forest(
            dimension,
            cell_global_ids,
            ordered_vertices,
            tags,
            generations,
            {
                "parent_ids": record_parent_ids,
                "parent_blocks": record_parent_blocks,
                "parent_rows": record_parent_rows,
                "parent_vertices": record_parent_vertices,
                "parent_tags": record_parent_tags,
                "child_ids": record_child_ids,
                "vertex_ids": record_vertex_ids,
            },
            {
                "parent_ids": subdivision_parent_ids,
                "parent_blocks": subdivision_parent_blocks,
                "parent_rows": subdivision_parent_rows,
                "child_ids": subdivision_child_ids,
                "vertex_ids": subdivision_vertex_ids,
            },
        )
        retired_keys, retired_ids = _retired_tables(
            retired_entity_keys, retired_entity_ids, dimension
        )
        vertex_counter, cell_counter = int(next_vertex_id), int(next_cell_id)
        vertex_high = max(
            int(np.max(arrays["ordered_vertices"])),
            int(np.max(arrays["record_parent_vertices"], initial=-1)),
            int(np.max(arrays["record_vertex_ids"], initial=-1)),
            int(np.max(arrays["subdivision_parent_rows"], initial=-1)),
            int(np.max(arrays["subdivision_vertex_ids"], initial=-1)),
        )
        cell_high = max(
            int(arrays["cell_global_ids"][-1]),
            int(np.max(arrays["record_parent_ids"], initial=-1)),
            int(np.max(arrays["record_child_ids"], initial=-1)),
            int(np.max(arrays["subdivision_parent_ids"], initial=-1)),
            int(np.max(arrays["subdivision_child_ids"], initial=-1)),
        )
        if vertex_counter <= vertex_high or cell_counter <= cell_high:
            raise ValueError("Hierarchy counters must exceed every issued ID.")
        self.dimension = dimension
        self.cell_global_ids = jnp.asarray(arrays["cell_global_ids"])
        self.ordered_vertices = jnp.asarray(arrays["ordered_vertices"])
        self.tags = jnp.asarray(arrays["tags"])
        self.generations = jnp.asarray(arrays["generations"])
        self.record_parent_ids = jnp.asarray(arrays["record_parent_ids"])
        self.record_parent_blocks = jnp.asarray(arrays["record_parent_blocks"])
        self.record_parent_rows = jnp.asarray(arrays["record_parent_rows"])
        self.record_parent_vertices = jnp.asarray(arrays["record_parent_vertices"])
        self.record_parent_tags = jnp.asarray(arrays["record_parent_tags"])
        self.record_child_ids = jnp.asarray(arrays["record_child_ids"])
        self.record_vertex_ids = jnp.asarray(arrays["record_vertex_ids"])
        self.subdivision_parent_ids = jnp.asarray(arrays["subdivision_parent_ids"])
        self.subdivision_parent_blocks = jnp.asarray(arrays["subdivision_parent_blocks"])
        self.subdivision_parent_rows = jnp.asarray(arrays["subdivision_parent_rows"])
        self.subdivision_child_ids = jnp.asarray(arrays["subdivision_child_ids"])
        self.subdivision_vertex_ids = jnp.asarray(arrays["subdivision_vertex_ids"])
        self.retired_entity_keys = tuple(jnp.asarray(value) for value in retired_keys)
        self.retired_entity_ids = tuple(jnp.asarray(value) for value in retired_ids)
        self.next_vertex_id = vertex_counter
        self.next_cell_id = cell_counter
        self.hierarchy_id = canonical_fingerprint(
            {
                "kind": "bisection-hierarchy",
                "dimension": dimension,
                "arrays": array_tree_fingerprint(arrays),
                "retired_entity_keys": array_tree_fingerprint(retired_keys),
                "retired_entity_ids": array_tree_fingerprint(retired_ids),
                "next_vertex_id": vertex_counter,
                "next_cell_id": cell_counter,
            }
        )


@final
class BisectionEvidence(StrictModule, NonTrainableState):
    """What one bisection adaptation requested, did, and refused.

    ``rejected_refinement_ids`` are marked source cells whose own conformity
    closure would split a protected edge; ``admissibility_tests`` counts the
    closure simulations of the group test that isolated them.
    ``initially_compatible`` is ``None`` when a supplied hierarchy carried the
    labels. ``rejected_coarsening_ids`` are marked source cells that stay active
    (their family is incomplete, unmarked, protected, class-mixed, or touches the
    refinement closure).
    """

    requested_refinements: int = eqx.field(static=True)
    accepted_refinements: int = eqx.field(static=True)
    rejected_refinement_ids: Array
    admissibility_tests: int = eqx.field(static=True)
    bisections: int = eqx.field(static=True)
    closure_iterations: int = eqx.field(static=True)
    created_vertices: int = eqx.field(static=True)
    maximum_generation: int = eqx.field(static=True)
    initially_compatible: bool | None = eqx.field(static=True)
    incompatible_facets: int = eqx.field(static=True)
    uniform_refinement_applied: bool = eqx.field(static=True)
    requested_coarsenings: int = eqx.field(static=True)
    coarsened_vertices: int = eqx.field(static=True)
    coarsening_passes: int = eqx.field(static=True)
    restored_cells: int = eqx.field(static=True)
    rejected_coarsening_ids: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        requested_refinements: int,
        accepted_refinements: int,
        rejected_refinement_ids: ArrayLike,
        admissibility_tests: int,
        bisections: int,
        closure_iterations: int,
        created_vertices: int,
        maximum_generation: int,
        initially_compatible: bool | None,
        incompatible_facets: int,
        uniform_refinement_applied: bool,
        requested_coarsenings: int,
        coarsened_vertices: int,
        coarsening_passes: int,
        restored_cells: int,
        rejected_coarsening_ids: ArrayLike,
    ) -> None:
        counts = {
            "requested_refinements": int(requested_refinements),
            "accepted_refinements": int(accepted_refinements),
            "admissibility_tests": int(admissibility_tests),
            "bisections": int(bisections),
            "closure_iterations": int(closure_iterations),
            "created_vertices": int(created_vertices),
            "maximum_generation": int(maximum_generation),
            "incompatible_facets": int(incompatible_facets),
            "requested_coarsenings": int(requested_coarsenings),
            "coarsened_vertices": int(coarsened_vertices),
            "coarsening_passes": int(coarsening_passes),
            "restored_cells": int(restored_cells),
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError("Bisection evidence counts must be non-negative.")
        if counts["accepted_refinements"] > counts["requested_refinements"]:
            raise ValueError("Accepted refinements cannot exceed the request.")
        if initially_compatible is not None and not isinstance(
            initially_compatible, bool
        ):
            raise TypeError("initially_compatible must be bool or None.")
        if not isinstance(uniform_refinement_applied, bool):
            raise TypeError("uniform_refinement_applied must be bool.")
        rejected_refinements = _identifiers(
            rejected_refinement_ids, "rejected_refinement_ids"
        )
        rejected_coarsenings = _identifiers(
            rejected_coarsening_ids, "rejected_coarsening_ids"
        )
        self.requested_refinements = counts["requested_refinements"]
        self.accepted_refinements = counts["accepted_refinements"]
        self.rejected_refinement_ids = jnp.asarray(rejected_refinements)
        self.admissibility_tests = counts["admissibility_tests"]
        self.bisections = counts["bisections"]
        self.closure_iterations = counts["closure_iterations"]
        self.created_vertices = counts["created_vertices"]
        self.maximum_generation = counts["maximum_generation"]
        self.initially_compatible = initially_compatible
        self.incompatible_facets = counts["incompatible_facets"]
        self.uniform_refinement_applied = uniform_refinement_applied
        self.requested_coarsenings = counts["requested_coarsenings"]
        self.coarsened_vertices = counts["coarsened_vertices"]
        self.coarsening_passes = counts["coarsening_passes"]
        self.restored_cells = counts["restored_cells"]
        self.rejected_coarsening_ids = jnp.asarray(rejected_coarsenings)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "bisection-evidence",
                **counts,
                "initially_compatible": initially_compatible,
                "uniform_refinement_applied": uniform_refinement_applied,
                "rejected_refinement_ids": array_tree_fingerprint(rejected_refinements),
                "rejected_coarsening_ids": array_tree_fingerprint(rejected_coarsenings),
            }
        )


class BisectionOutcome(NamedTuple):
    """Topology edit, target hierarchy, and evidence of one bisection adaptation."""

    edit: CellTopologyEdit
    hierarchy: BisectionHierarchy
    evidence: BisectionEvidence
    refinement_parent_cell_global_ids: np.ndarray
    refinement_source_edge_global_ids: np.ndarray
    refinement_edge_vertex_ids: np.ndarray


# Vertices are addressed by compact indices during one adaptation: source
# vertices by the rank of their global ID, new vertices after them in issue
# order. New IDs exceed every existing ID, so compact order is global-ID order
# and every canonical ordering below equals the global-ID ordering.
_SHIFT = np.int64(1 << 31)
_VERTEX_LIMIT = 1 << 31


# The bisection label templates are owned by the device layout so that host and
# device bisection share one table set.
_TABLES = {dimension: bisection_label_tables(dimension) for dimension in (2, 3)}


def _subdivision_tables(dimension: int, /) -> tuple[tuple[tuple[int, ...], ...], Any]:
    """Created-vertex subsets and the barycentric pieces of every parent facet.

    Subsets index positions of the ascending parent row: edges, then faces of a
    tetrahedron, then the center. Pieces address the extended row (parent row,
    then created vertices) with shape ``(d + 1, d!, d)``: facet ``j`` omits
    position ``j`` and splits into the chains ``(p0, m(p0 p1)[, c(p0 p1 p2)])``.
    """

    width = dimension + 1
    subsets = tuple(
        subset
        for size in range(2, width + 1)
        for subset in combinations(range(width), size)
    )
    column = {subset: width + index for index, subset in enumerate(subsets)}
    pieces = np.asarray(
        tuple(
            tuple(
                (order[0],)
                + tuple(
                    column[tuple(sorted(order[:size]))]
                    for size in range(2, dimension + 1)
                )
                for order in permutations(
                    tuple(position for position in range(width) if position != omitted)
                )
            )
            for omitted in range(width)
        ),
        dtype=np.int64,
    )
    return subsets, pieces


_SUBDIVISION = {dimension: _subdivision_tables(dimension) for dimension in (2, 3)}


class _Cells(NamedTuple):
    ids: np.ndarray
    rows: np.ndarray
    tuples: np.ndarray
    tags: np.ndarray
    blocks: np.ndarray
    origins: np.ndarray
    generations: np.ndarray


class _Records(NamedTuple):
    parent_ids: np.ndarray
    blocks: np.ndarray
    rows: np.ndarray
    tuples: np.ndarray
    tags: np.ndarray
    child_ids: np.ndarray
    vertices: np.ndarray


class _Subdivisions(NamedTuple):
    parent_ids: np.ndarray
    blocks: np.ndarray
    rows: np.ndarray
    child_ids: np.ndarray
    vertices: np.ndarray


class _Growth(NamedTuple):
    parents: np.ndarray
    weights: np.ndarray
    levels: np.ndarray


class _Front(NamedTuple):
    cells: _Cells
    growth: _Growth
    split_keys: np.ndarray
    split_vertices: np.ndarray
    vertex_base: int
    next_cell: int


class _Flags(NamedTuple):
    marked: np.ndarray
    blocked: np.ndarray
    classes: np.ndarray


class _Source(NamedTuple):
    mesh: CellMesh
    dimension: int
    vertex_ids: np.ndarray
    original: np.ndarray
    coordinates: np.ndarray
    cells: _Cells


class _Request(NamedTuple):
    refine_ids: np.ndarray
    coarsen_ids: np.ndarray
    protected_codes: np.ndarray
    protected_vertices: np.ndarray
    cell_classes: np.ndarray
    facet_keys: np.ndarray
    facet_classes: np.ndarray


class _Start(NamedTuple):
    front: _Front
    records: _Records
    marks: np.ndarray
    next_vertex: int
    compatible: bool | None
    incompatible: int
    uniform: bool
    retired: tuple[tuple[np.ndarray, np.ndarray], ...]
    subdivisions: _Subdivisions


class _Refinement(NamedTuple):
    front: _Front
    records: _Records
    rejected_ids: np.ndarray
    tests: int
    iterations: int


class _Family(NamedTuple):
    removed: np.ndarray
    undo: np.ndarray
    facet_keys: np.ndarray
    facet_classes: np.ndarray


class _Coarsening(NamedTuple):
    restored: _Cells
    removed_ids: np.ndarray
    link_targets: np.ndarray
    undone: np.ndarray
    removed_vertices: np.ndarray
    supports: np.ndarray
    passes: int
    rejected_ids: np.ndarray


def _select(table: Any, index: Any, /) -> Any:
    return type(table)(*(value[index] for value in table))


def _joined(first: Any, second: Any, /) -> Any:
    return type(first)(
        *(np.concatenate((a, b), axis=0) for a, b in zip(first, second, strict=True))
    )


def _empty_records(dimension: int, /) -> _Records:
    width = dimension + 1
    empty = np.zeros((0,), dtype=np.int64)
    rows = np.zeros((0, width), dtype=np.int64)
    return _Records(
        empty, empty, rows, rows, empty, np.zeros((0, 2), dtype=np.int64), empty
    )


def _edge_codes(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    return np.minimum(first, second) * _SHIFT + np.maximum(first, second)


def _members(table: np.ndarray, queries: np.ndarray, /) -> np.ndarray:
    """Membership of queries in one ascending unique table."""

    if table.size == 0:
        return np.zeros(queries.shape, dtype=np.bool_)
    position = np.minimum(np.searchsorted(table, queries), table.size - 1)
    return table[position] == queries


def _simplex_keys(rows: np.ndarray, size: int, /) -> np.ndarray:
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


def _merged_rows(sources: np.ndarray, weights: np.ndarray, width: int, /) -> Any:
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
        raise ValueError("A bisection vertex stencil exceeds one source simplex.")
    merged_sources = np.full((count, width), -1, dtype=np.int64)
    merged_weights = np.zeros((count, width), dtype=np.float64)
    merged_sources[group_owner, column] = flat_sources[start]
    merged_weights[group_owner, column] = totals
    return merged_sources, merged_weights


def _odd_relative(rows: np.ndarray, tuples: np.ndarray, dimension: int, /) -> Any:
    """Whether each tuple is an odd permutation of its oriented row."""

    pairs = _TABLES[dimension].pairs
    position = np.argmax(tuples[:, :, None] == rows[:, None, :], axis=2)
    inversions = np.sum(position[:, pairs[:, 0]] > position[:, pairs[:, 1]], axis=1)
    return inversions % 2 == 1


def _reversed(tuples: np.ndarray, tags: np.ndarray, dimension: int, /) -> Any:
    return np.take_along_axis(tuples, _TABLES[dimension].equivalent[tags], axis=1)


def _refinement_ends(tuples: np.ndarray, tags: np.ndarray, dimension: int, /) -> Any:
    """Second refinement-edge endpoint ``x_position(tag)`` of each tagged tuple."""

    position = _TABLES[dimension].position[tags]
    return tuples[np.arange(tuples.shape[0]), position]


def _canonical(tuples: Any, tags: Any, odd: Any, dimension: int, /) -> Any:
    """Canonical representative of each tagged simplex and its orientation parity."""

    reversed_ = _reversed(tuples, tags, dimension)
    differs = reversed_ != tuples
    first = np.argmax(differs, axis=1)
    rows = np.arange(tuples.shape[0])
    smaller = np.any(differs, axis=1) & (reversed_[rows, first] < tuples[rows, first])
    flips = _TABLES[dimension].equivalent_odd[tags]
    return (
        np.where(smaller[:, None], reversed_, tuples),
        odd ^ (smaller & flips),
    )


def _periodic_locked_front(
    source: _Source,
    front: _Front,
    request: _Request,
    edge_lock: PeriodicRefinementEdgeLock,
    /,
) -> _Front:
    if source.dimension not in (2, 3):
        raise ValueError(
            "Periodic edge-locked bisection requires triangle or tetrahedron cells."
        )
    cell_ids, selected_edges = edge_lock.require_source(source.mesh)
    if not np.array_equal(cell_ids, request.refine_ids):
        raise ValueError(
            "Periodic edge lock cells must equal the explicit refinement marks."
        )
    edge_ids = np.asarray(source.mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_keys = entity_keys(source.mesh, 1)
    endpoints_by_edge = {
        int(identifier): tuple(int(value) for value in key if int(value) >= 0)
        for key, identifier in zip(edge_keys, edge_ids, strict=True)
    }
    row_by_cell = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(front.cells.ids, dtype=np.int64))
    }
    tuples = np.array(front.cells.tuples, dtype=np.int64, copy=True)
    tags = np.array(front.cells.tags, dtype=np.int64, copy=True)
    for cell_id, edge_id in zip(cell_ids, selected_edges, strict=True):
        row = row_by_cell.get(int(cell_id))
        edge_vertices = endpoints_by_edge.get(int(edge_id))
        if row is None or edge_vertices is None or len(edge_vertices) != 2:
            raise ValueError("Periodic lock names a missing cell or source edge.")
        compact = np.searchsorted(
            source.vertex_ids, np.asarray(edge_vertices, dtype=np.int64)
        )
        if np.any(compact >= source.vertex_ids.size) or not np.array_equal(
            source.vertex_ids[compact], np.asarray(edge_vertices, dtype=np.int64)
        ):
            raise ValueError(
                "Periodic locked edge is absent from the source vertex bank."
            )
        cell_vertices = tuples[row]
        if not np.all(np.isin(compact, cell_vertices)):
            raise ValueError(
                "Periodic locked edge is not a corner edge of its marked cell."
            )
        first, last = (int(value) for value in compact)
        remaining = [
            int(value) for value in cell_vertices if int(value) not in (first, last)
        ]
        desired = np.asarray((first, *remaining, last), dtype=np.int64)
        tags[row] = source.dimension
        canonical, _ = _canonical(
            desired[None, :],
            tags[row : row + 1],
            np.zeros((1,), dtype=np.bool_),
            source.dimension,
        )
        tuples[row] = canonical[0]
    cells = front.cells._replace(tuples=tuples, tags=tags)
    return front._replace(cells=cells)


def _oriented_rows(tuples: np.ndarray, odd: np.ndarray, /) -> np.ndarray:
    rows = tuples.copy()
    rows[odd, -2] = tuples[odd, -1]
    rows[odd, -1] = tuples[odd, -2]
    return rows


def _children(tuples: Any, tags: Any, midpoints: Any, dimension: int, /) -> Any:
    tables = _TABLES[dimension]
    first_table, second_table = tables.first, tables.second
    extended = np.concatenate((tuples, midpoints[:, None]), axis=1)
    return (
        np.take_along_axis(extended, first_table[tags], axis=1),
        np.take_along_axis(extended, second_table[tags], axis=1),
    )


def _prepared_source(mesh: CellMesh, /) -> _Source:
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    dimension = mesh.topological_dimension
    match dimension:
        case 2:
            kinds = ("triangle",)
        case 3:
            kinds = ("tetrahedron", "simplex:3")
        case _:
            raise ValueError("Bisection requires a triangle mesh or a tetrahedral mesh.")
    if any(block.cell_kind not in kinds for block in mesh.blocks):
        raise ValueError("Bisection requires a triangle mesh or a tetrahedral mesh.")
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if vertex_ids.size >= _VERTEX_LIMIT:
        raise ValueError("Bisection supports fewer than 2**31 vertices.")
    original = np.argsort(vertex_ids, kind="stable")
    compact = np.empty_like(original)
    compact[original] = np.arange(original.size)
    rows = np.concatenate(
        tuple(
            compact[np.asarray(block.vertices, dtype=np.int64)] for block in mesh.blocks
        )
    )
    ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    blocks = np.concatenate(
        tuple(
            np.full((block.cell_count,), index, dtype=np.int64)
            for index, block in enumerate(mesh.blocks)
        )
    )
    order = np.argsort(ids, kind="stable")
    zeros = np.zeros(ids.shape, dtype=np.int64)
    cells = _Cells(
        ids[order], rows[order], rows[order], zeros, blocks[order], ids[order], zeros
    )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)[original]
    return _Source(mesh, dimension, vertex_ids[original], original, coordinates, cells)


def _compact_vertices(source: _Source, identifiers: np.ndarray, name: str, /) -> Any:
    ids = source.vertex_ids
    position = np.minimum(np.searchsorted(ids, identifiers), ids.size - 1)
    if np.any(ids[position] != identifiers):
        raise ValueError(f"{name} references vertices absent from the mesh.")
    return position


def _marked_ids(values: np.ndarray, source: _Source, name: str, /) -> np.ndarray:
    ids = np.asarray(values, dtype=np.int64)
    if ids.ndim != 1 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError(f"{name} must be a strictly increasing rank-1 ID array.")
    if not np.all(_members(source.cells.ids, ids)):
        raise ValueError(f"{name} references cells absent from the mesh.")
    return ids


def _aligned_classes(values: np.ndarray, count: int, name: str, /) -> np.ndarray:
    classes = np.asarray(values, dtype=np.int64)
    if classes.shape != (count,):
        raise ValueError(f"{name} must hold one class per entity ({count}).")
    return classes


def _prepared_request(
    source: _Source,
    refine_cell_ids: np.ndarray,
    coarsen_cell_ids: np.ndarray,
    protected_edges: np.ndarray,
    protected_vertices: np.ndarray,
    cell_classes: np.ndarray,
    facet_classes: np.ndarray,
    /,
) -> _Request:
    refine = _marked_ids(refine_cell_ids, source, "refine_cell_ids")
    coarsen = _marked_ids(coarsen_cell_ids, source, "coarsen_cell_ids")
    if np.intersect1d(refine, coarsen).size:
        raise ValueError("Refinement and coarsening marks must be disjoint.")
    edges = np.asarray(protected_edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or np.any(edges[:, 0] >= edges[:, 1]):
        raise ValueError("protected_edges must be ascending (p, 2) vertex-ID keys.")
    ends = _compact_vertices(source, edges, "protected_edges")
    vertices = np.asarray(protected_vertices, dtype=np.int64)
    if vertices.ndim != 1:
        raise ValueError("protected_vertices must be a rank-1 ID array.")
    protected = np.zeros(source.vertex_ids.shape, dtype=np.bool_)
    protected[_compact_vertices(source, vertices, "protected_vertices")] = True
    mesh, dimension = source.mesh, source.dimension
    cell_order = np.searchsorted(source.cells.ids, entity_keys(mesh, dimension)[:, 0])
    classes = np.empty(source.cells.ids.shape, dtype=np.int64)
    classes[cell_order] = _aligned_classes(cell_classes, cell_order.size, "cell_classes")
    facet_keys = np.searchsorted(source.vertex_ids, entity_keys(mesh, dimension - 1))
    return _Request(
        refine,
        coarsen,
        np.unique(_edge_codes(ends[:, 0], ends[:, 1])),
        protected,
        classes,
        facet_keys,
        _aligned_classes(facet_classes, facet_keys.shape[0], "facet_classes"),
    )


class _EdgeOrder(NamedTuple):
    """Global strict edge order: longest first, ties by ascending edge code."""

    codes: np.ndarray
    ranks: np.ndarray


def _edge_order(cells: _Cells, coordinates: np.ndarray, dimension: int, /) -> Any:
    pairs = _TABLES[dimension].pairs
    low = np.minimum(cells.rows[:, pairs[:, 0]], cells.rows[:, pairs[:, 1]])
    high = np.maximum(cells.rows[:, pairs[:, 0]], cells.rows[:, pairs[:, 1]])
    unique, first = np.unique((low * _SHIFT + high).reshape((-1,)), return_index=True)
    lengths = np.sum(
        (coordinates[high.reshape((-1,))[first]] - coordinates[low.reshape((-1,))[first]])
        ** 2,
        axis=1,
    )
    ranks = np.empty(unique.shape, dtype=np.int64)
    ranks[np.lexsort((unique, -lengths))] = np.arange(unique.size)
    return _EdgeOrder(unique, ranks)


def _edge_ranks(order: _EdgeOrder, first: np.ndarray, second: np.ndarray, /) -> Any:
    return order.ranks[np.searchsorted(order.codes, _edge_codes(first, second))]


def _longest_edges(cells: _Cells, order: _EdgeOrder, dimension: int, /) -> Any:
    """Longest edge ``(low, high)`` of each cell and its other vertices."""

    pairs = _TABLES[dimension].pairs
    rows = cells.rows
    count = cells.ids.size
    best = np.argmin(
        _edge_ranks(order, rows[:, pairs[:, 0]], rows[:, pairs[:, 1]]), axis=1
    )
    ends = rows[np.arange(count)[:, None], pairs[best]]
    low, high = np.min(ends, axis=1), np.max(ends, axis=1)
    others = rows[(rows != low[:, None]) & (rows != high[:, None])]
    return low, high, others.reshape((count, dimension - 1))


def _longest_edge_labels(
    cells: _Cells, coordinates: np.ndarray, dimension: int, /
) -> Any:
    """Maubach labels from the global strict edge order (longest first, then key)."""

    order = _edge_order(cells, coordinates, dimension)
    x0, xk, others = _longest_edges(cells, order, dimension)
    count = cells.ids.size
    if dimension == 2:
        tuples = np.stack((x0, others[:, 0], xk), axis=1)
    else:
        first_rank = _edge_ranks(order, x0, others[:, 0])
        second_rank = _edge_ranks(order, x0, others[:, 1])
        greater = first_rank < second_rank
        x2 = np.where(greater, others[:, 0], others[:, 1])
        x1 = np.where(greater, others[:, 1], others[:, 0])
        tuples = np.stack((x0, x1, x2, xk), axis=1)
    tags = np.full((count,), dimension, dtype=np.int64)
    tuples, _ = _canonical(tuples, tags, np.zeros((count,), dtype=np.bool_), dimension)
    return tuples, tags


def _marked_tetrahedron_labels(cells: _Cells, coordinates: np.ndarray, /) -> Any:
    """Arnold-Mukherjee-Pouly initial marking under the global strict edge order.

    Every face is marked by its longest edge and every tetrahedron bisects its
    longest edge ``(a, b)``. With ``e_a``/``e_b`` the marks of the faces opposite
    ``b``/``a`` and ``{c, d}`` the other vertices, the tetrahedron is planar
    (``e_a = (a, k)``, ``e_b = (b, k)``: Maubach ``(a, k, b, j)_2``), mixed
    (``e_a = (a, k_a)``, ``e_b = (b, k_b)``, ``k_a != k_b``: ``(a, k_b, k_a, b)_3``),
    adjacent (``e_a = (c, d)``, ``e_b = (b, k)``: ``(a, j, k, b)_4``, and
    symmetrically) or opposite (``e_a = e_b = (c, d)``:
    ``(min(a, b), min(c, d), max(c, d), max(a, b))_5``). Flagged planar children
    are Maubach tag 1. Neighbors see one mark per shared face, so the conformity
    closure terminates on any conforming tetrahedral mesh without the matching
    condition (Arnold, Mukherjee and Pouly 2000).
    """

    order = _edge_order(cells, coordinates, 3)
    a, b, others = _longest_edges(cells, order, 3)
    c, d = others[:, 0], others[:, 1]

    def face_mark(apex: np.ndarray) -> np.ndarray:
        """Longest edge of face ``(apex, c, d)``: 0 = (apex, c), 1 = (apex, d), 2 = (c, d)."""

        return np.argmin(
            np.stack(
                (
                    _edge_ranks(order, apex, c),
                    _edge_ranks(order, apex, d),
                    _edge_ranks(order, c, d),
                ),
                axis=1,
            ),
            axis=1,
        )

    mark_a, mark_b = face_mark(a), face_mark(b)
    k_a, j_a = np.where(mark_a == 0, c, d), np.where(mark_a == 0, d, c)
    k_b, j_b = np.where(mark_b == 0, c, d), np.where(mark_b == 0, d, c)
    opposite = (mark_a == 2) & (mark_b == 2)
    adjacent_a = (mark_a == 2) & (mark_b < 2)
    adjacent_b = (mark_b == 2) & (mark_a < 2)
    planar = (mark_a < 2) & (mark_b < 2) & (k_a == k_b)
    conditions = (opposite, adjacent_a, adjacent_b, planar)
    tuples = np.select(
        tuple(condition[:, None] for condition in conditions),
        (
            np.stack(
                (
                    np.minimum(a, b),
                    np.minimum(c, d),
                    np.maximum(c, d),
                    np.maximum(a, b),
                ),
                axis=1,
            ),
            np.stack((a, j_b, k_b, b), axis=1),
            np.stack((b, j_a, k_a, a), axis=1),
            np.stack((a, k_a, b, j_a), axis=1),
        ),
        np.stack((a, k_b, k_a, b), axis=1),
    )
    tags = np.select(conditions, (5, 4, 4, 2), 3).astype(np.int64)
    count = cells.ids.size
    tuples, _ = _canonical(tuples, tags, np.zeros((count,), dtype=np.bool_), 3)
    return tuples, tags


def _reflected(first: Any, second: Any, tags: Any, dimension: int, /) -> np.ndarray:
    mirrored = _reversed(second, tags, dimension)
    return (np.sum(first != second, axis=1) == 1) | (
        np.sum(first != mirrored, axis=1) == 1
    )


def _facet_children(cells: _Cells, owners: Any, opposite: Any, dimension: int, /) -> Any:
    """Child of each owner containing the facet opposite ``opposite``."""

    tuples, tags = cells.tuples[owners], cells.tags[owners]
    first, second = _children(tuples, tags, -1 - owners, dimension)
    at_start = opposite == tuples[:, 0]
    at_end = opposite == _refinement_ends(tuples, tags, dimension)
    child = np.where(at_start[:, None], second, first)
    return child, at_start | at_end


def _incompatible_facets(cells: _Cells, dimension: int, /) -> int:
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
        raise ValueError("Bisection requires every facet to bound at most two cells.")
    pair = np.flatnonzero(shared)
    left, right = owners[pair], owners[pair + 1]
    same_tag = cells.tags[left] == cells.tags[right]
    tags = cells.tags[left]
    matched = same_tag & _reflected(
        cells.tuples[left], cells.tuples[right], tags, dimension
    )
    left_child, left_open = _facet_children(cells, left, opposite[pair], dimension)
    right_child, right_open = _facet_children(cells, right, opposite[pair + 1], dimension)
    child_tags = _TABLES[dimension].child[tags]
    matched |= (
        same_tag
        & left_open
        & right_open
        & _reflected(left_child, right_child, child_tags, dimension)
    )
    return int(np.sum(~matched))


def _empty_subdivisions(dimension: int, /) -> _Subdivisions:
    arrangements, created = _subdivision_width(dimension)
    empty = np.zeros((0,), dtype=np.int64)
    return _Subdivisions(
        empty,
        empty,
        np.zeros((0, dimension + 1), dtype=np.int64),
        np.zeros((0, arrangements), dtype=np.int64),
        np.zeros((0, created), dtype=np.int64),
    )


def _subdivided(
    front: _Front, selected: np.ndarray, dimension: int, /
) -> tuple[_Front, _Subdivisions]:
    """Barycentric subdivision of the selected cells into (v, m, [f,] c)_d simplices.

    Each selected simplex becomes (d+1)! children; the rest stay. The selection
    must share no edge with the kept cells, and no vertex may have been issued
    yet, so the created vertices are the first growth rows.
    """

    if front.growth.levels.size:
        raise ValueError("Barycentric subdivision precedes every bisection vertex.")
    cells, width = _select(front.cells, selected), dimension + 1
    orders, odd_orders = _TABLES[dimension].orders, _TABLES[dimension].odd_orders
    count, arrangements = cells.ids.size, orders.shape[0]
    tables = [_simplex_keys(cells.rows, size) for size in range(2, width)]
    tables.append(cells.rows)
    offsets = np.cumsum([0] + [table.shape[0] for table in tables])
    if front.vertex_base + offsets[-1] >= _VERTEX_LIMIT:
        raise ValueError("Barycentric subdivision would exceed 2**31 vertices.")
    columns = [cells.rows[:, orders[:, 0]]]
    for size in range(2, width):
        prefix = np.sort(cells.rows[:, orders[:, :size]], axis=2).reshape((-1, size))
        rows = key_rows(tables[size - 2], prefix).reshape((count, arrangements))
        columns.append(front.vertex_base + offsets[size - 2] + rows)
    centers = front.vertex_base + offsets[-2] + np.arange(count, dtype=np.int64)
    columns.append(np.repeat(centers[:, None], arrangements, axis=1))
    tuples = np.stack(columns, axis=2).reshape((-1, width))
    tags = np.full((tuples.shape[0],), dimension, dtype=np.int64)
    tuples, odd = _canonical(tuples, tags, np.tile(odd_orders, count), dimension)
    child_ids = front.next_cell + np.arange(tuples.shape[0], dtype=np.int64)
    children = _Cells(
        child_ids,
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
    growth = _Growth(parents, weights, np.zeros((parents.shape[0],), dtype=np.int64))
    ascending = np.sort(cells.rows, axis=1)
    created = [
        front.vertex_base
        + offsets[len(subset) - 2]
        + key_rows(tables[len(subset) - 2], ascending[:, list(subset)])
        for subset in _SUBDIVISION[dimension][0][:-1]
    ]
    created.append(centers)
    records = _Subdivisions(
        cells.ids,
        cells.blocks,
        cells.rows,
        child_ids.reshape((count, arrangements)),
        np.stack(created, axis=1),
    )
    front = front._replace(
        cells=_joined(_select(front.cells, ~selected), children),
        growth=growth,
        next_cell=front.next_cell + tuples.shape[0],
    )
    return front, records


def _split_edges(front: _Front, codes: np.ndarray, level: int, dimension: int, /) -> Any:
    """Midpoint of each refinement edge, issuing new vertices in sorted key order."""

    known = _members(front.split_keys, codes)
    created = np.unique(codes[~known])
    first = front.vertex_base + front.growth.levels.size
    if first + created.size >= _VERTEX_LIMIT:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Bisection would exceed 2**31 vertices in one adaptation.",
            stage="bisection-closure",
        )
    vertices = first + np.arange(created.size, dtype=np.int64)
    parents = np.full((created.size, dimension + 1), -1, dtype=np.int64)
    parents[:, 0] = created // _SHIFT
    parents[:, 1] = created % _SHIFT
    weights = np.zeros(parents.shape, dtype=np.float64)
    weights[:, :2] = 0.5
    growth = _joined(
        front.growth,
        _Growth(parents, weights, np.full(created.shape, level, dtype=np.int64)),
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


def _bisected(front: _Front, selected: np.ndarray, level: int, dimension: int, /) -> Any:
    """Bisect the selected cells once; children follow in (parent ID, 1, 2) order."""

    cells = front.cells
    chosen = _select(cells, selected)
    count = selected.size
    tables = _TABLES[dimension]
    codes = _edge_codes(
        chosen.tuples[:, 0], _refinement_ends(chosen.tuples, chosen.tags, dimension)
    )
    front, midpoints, created = _split_edges(front, codes, level, dimension)
    first, second = _children(chosen.tuples, chosen.tags, midpoints, dimension)
    odd = _odd_relative(chosen.rows, chosen.tuples, dimension)
    tuples = np.stack((first, second), axis=1).reshape((2 * count, dimension + 1))
    odd = np.stack(
        (odd ^ tables.first_odd[chosen.tags], odd ^ tables.second_odd[chosen.tags]),
        axis=1,
    ).reshape((-1,))
    tags = np.repeat(tables.child[chosen.tags], 2)
    tuples, odd = _canonical(tuples, tags, odd, dimension)
    ids = front.next_cell + np.arange(2 * count, dtype=np.int64)
    children = _Cells(
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
    records = _Records(
        chosen.ids,
        chosen.blocks,
        chosen.rows,
        chosen.tuples,
        chosen.tags,
        ids.reshape((count, 2)),
        midpoints,
    )
    front = front._replace(
        cells=_joined(_select(cells, keep), children),
        next_cell=front.next_cell + 2 * count,
    )
    return front, records, created


def _nonconforming(cells: _Cells, split_keys: np.ndarray, dimension: int, /) -> Any:
    pairs = _TABLES[dimension].pairs
    codes = _edge_codes(cells.tuples[:, pairs[:, 0]], cells.tuples[:, pairs[:, 1]])
    return np.any(_members(split_keys, codes), axis=1)


def _closure(
    front: _Front,
    selected: np.ndarray,
    protected_codes: np.ndarray,
    dimension: int,
    /,
) -> tuple[_Front, _Records, int] | None:
    """Bisect selected cells until the refinement closure is conforming.

    Returns ``None`` as soon as the closure would split a protected edge. The
    Maubach compatibility invariant provides finite closure; a real storage or
    index overflow is reported by the allocation/identifier owner instead of a
    policy iteration quota.
    """

    records = _empty_records(dimension)
    iterations = 0
    while selected.size:
        iterations += 1
        front, record, created = _bisected(front, selected, iterations, dimension)
        if np.any(_members(protected_codes, created)):
            return None
        records = _joined(records, record)
        selected = np.flatnonzero(
            _nonconforming(front.cells, front.split_keys, dimension)
        )
    return front, records, iterations


def _admissible_marks(
    front: _Front, marks: np.ndarray, request: _Request, dimension: int, /
) -> Any:
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
        protected,
        _edge_codes(tuples[:, 0], _refinement_ends(tuples, tags, dimension)),
    )
    accepted, rejected = [empty], [marks[direct]]
    pending = [marks[~direct]] if np.any(~direct) else []
    tests = 0
    while pending:
        group = pending.pop()
        tests += 1
        if _closure(front, group, protected, dimension) is not None:
            accepted.append(group)
        elif group.size == 1:
            rejected.append(group)
        else:
            half = group.size // 2
            pending.extend((group[half:], group[:half]))
    return np.sort(np.concatenate(accepted)), np.sort(np.concatenate(rejected)), tests


def _refinement(start: _Start, request: _Request, dimension: int, /) -> _Refinement:
    accepted, rejected, tests = _admissible_marks(
        start.front, start.marks, request, dimension
    )
    closure = _closure(start.front, accepted, request.protected_codes, dimension)
    if closure is None:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "The union of admissible bisection closures split a protected edge.",
            stage="bisection-closure",
        )
    front, records, iterations = closure
    return _Refinement(
        front,
        records,
        np.unique(start.front.cells.origins[rejected]),
        tests,
        iterations,
    )


def _initial_front(
    cells: _Cells, vertex_base: int, next_cell: int, dimension: int
) -> Any:
    width = dimension + 1
    empty = np.zeros((0,), dtype=np.int64)
    growth = _Growth(
        np.zeros((0, width), dtype=np.int64),
        np.zeros((0, width), dtype=np.float64),
        empty,
    )
    return _Front(cells, growth, empty, empty, vertex_base, next_cell)


def _labelled_start(
    source: _Source,
    request: _Request,
    compatibility: BisectionCompatibility,
    /,
    *,
    periodic_edge_lock: PeriodicRefinementEdgeLock | None = None,
) -> _Start:
    dimension = source.dimension
    if request.coarsen_ids.size:
        raise ValueError(
            "Coarsening requires the BisectionHierarchy of a previous bisection."
        )
    tuples, tags = _longest_edge_labels(source.cells, source.coordinates, dimension)
    cells = source.cells._replace(tuples=tuples, tags=tags)
    front = _initial_front(
        cells, source.vertex_ids.size, int(cells.ids[-1]) + 1, dimension
    )
    subdivisions = _empty_subdivisions(dimension)
    if periodic_edge_lock is not None:
        front = _periodic_locked_front(source, front, request, periodic_edge_lock)
    incompatible = _incompatible_facets(front.cells, dimension)
    uniform = False
    if incompatible:
        if periodic_edge_lock is not None:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SPECIFICATION,
                "Locked periodic Maubach edges violate the local matching condition.",
                stage="bisection-closure",
            )
        if dimension == 3 and compatibility is BisectionCompatibility.REJECT:
            # Marked tetrahedra need no matching condition: the closure of any
            # conforming tetrahedral mesh terminates under their labels.
            tuples, tags = _marked_tetrahedron_labels(source.cells, source.coordinates)
            front = front._replace(cells=cells._replace(tuples=tuples, tags=tags))
        else:
            front, subdivisions = _barycentric_front(
                front,
                np.ones(front.cells.ids.shape, dtype=np.bool_),
                request,
                compatibility,
                f"{incompatible} interior facets violate the bisection matching "
                "condition of the longest-edge labelling",
                dimension,
            )
            uniform = True
    marks = np.flatnonzero(np.isin(front.cells.origins, request.refine_ids))
    retired = tuple(
        (np.zeros((0, degree + 1), dtype=np.int64), np.zeros((0,), dtype=np.int64))
        for degree in range(1, dimension)
    )
    return _Start(
        front,
        _empty_records(dimension),
        marks,
        int(source.vertex_ids[-1]) + 1,
        incompatible == 0,
        incompatible,
        uniform,
        retired,
        subdivisions,
    )


def _barycentric_front(
    front: _Front,
    selected: np.ndarray,
    request: _Request,
    compatibility: BisectionCompatibility,
    reason: str,
    dimension: int,
    /,
) -> tuple[_Front, _Subdivisions]:
    """Subdivide incompatible cells barycentrically when the policy admits it."""

    match compatibility:
        case BisectionCompatibility.REJECT:
            raise ValueError(
                f"{reason}; use BisectionCompatibility.UNIFORM_REFINEMENT to "
                "subdivide barycentrically into a compatible mesh first."
            )
        case BisectionCompatibility.UNIFORM_REFINEMENT:
            pairs = _TABLES[dimension].pairs
            rows = front.cells.rows[selected]
            edges = _edge_codes(rows[:, pairs[:, 0]], rows[:, pairs[:, 1]])
            if np.any(_members(request.protected_codes, edges)):
                raise ValueError(
                    "Uniform barycentric refinement would split protected edges."
                )
            return _subdivided(front, selected, dimension)
        case _:
            raise ValueError(f"Unsupported bisection compatibility {compatibility!r}.")


def _resubdivided(
    start: _Start,
    request: _Request,
    compatibility: BisectionCompatibility,
    dimension: int,
    /,
) -> _Start:
    """Subdivide every restored (tag 0) source cell again before bisection.

    Restored cells form whole edge-connected components (coarsening removes a
    subdivision only with every cell around its created vertices), so their
    subdivision never splits an edge of a kept cell.
    """

    pending = start.front.cells.tags == 0
    if not np.any(pending):
        return start
    front, subdivisions = _barycentric_front(
        start.front,
        pending,
        request,
        compatibility,
        f"{int(np.count_nonzero(pending))} restored source cells keep labels that "
        "violate the bisection matching condition",
        dimension,
    )
    return start._replace(
        front=front,
        marks=np.flatnonzero(np.isin(front.cells.origins, request.refine_ids)),
        uniform=True,
        subdivisions=_joined(start.subdivisions, subdivisions),
    )


def _bound_records(
    source: _Source, hierarchy: BisectionHierarchy, /
) -> tuple[_Records, _Subdivisions]:
    def vertices(values: Array) -> np.ndarray:
        return _compact_vertices(
            source, np.asarray(values, dtype=np.int64), "BisectionHierarchy"
        )

    blocks = np.asarray(hierarchy.record_parent_blocks, dtype=np.int64)
    roots = np.asarray(hierarchy.subdivision_parent_blocks, dtype=np.int64)
    if np.any(blocks >= len(source.mesh.blocks)) or np.any(
        roots >= len(source.mesh.blocks)
    ):
        raise ValueError("BisectionHierarchy records reference undeclared blocks.")
    return _Records(
        np.asarray(hierarchy.record_parent_ids, dtype=np.int64),
        blocks,
        vertices(hierarchy.record_parent_rows),
        vertices(hierarchy.record_parent_vertices),
        np.asarray(hierarchy.record_parent_tags, dtype=np.int64),
        np.asarray(hierarchy.record_child_ids, dtype=np.int64),
        vertices(hierarchy.record_vertex_ids),
    ), _Subdivisions(
        np.asarray(hierarchy.subdivision_parent_ids, dtype=np.int64),
        roots,
        vertices(hierarchy.subdivision_parent_rows),
        np.asarray(hierarchy.subdivision_child_ids, dtype=np.int64),
        vertices(hierarchy.subdivision_vertex_ids),
    )


def _bound_start(
    source: _Source, request: _Request, hierarchy: BisectionHierarchy, /
) -> _Start:
    if not isinstance(hierarchy, BisectionHierarchy):
        raise TypeError("hierarchy must be BisectionHierarchy or None.")
    dimension = source.dimension
    if hierarchy.dimension != dimension:
        raise ValueError("BisectionHierarchy dimension does not match the mesh.")
    if not np.array_equal(np.asarray(hierarchy.cell_global_ids), source.cells.ids):
        raise ValueError("BisectionHierarchy active cell IDs differ from the mesh.")
    tuples = _compact_vertices(
        source,
        np.asarray(hierarchy.ordered_vertices, dtype=np.int64),
        "BisectionHierarchy",
    )
    if not np.array_equal(np.sort(tuples, axis=1), np.sort(source.cells.rows, axis=1)):
        raise ValueError("BisectionHierarchy cell vertex sets differ from the mesh.")
    if (
        hierarchy.next_vertex_id <= source.vertex_ids[-1]
        or hierarchy.next_cell_id <= source.cells.ids[-1]
    ):
        raise ValueError("BisectionHierarchy counters do not exceed the mesh IDs.")
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
    records, subdivisions = _bound_records(source, hierarchy)
    return _Start(
        front,
        records,
        np.searchsorted(cells.ids, request.refine_ids),
        hierarchy.next_vertex_id,
        None,
        0,
        False,
        retired,
        subdivisions,
    )


def _record_links(ids: np.ndarray, child_ids: np.ndarray, /) -> Any:
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
    records: _Records, family: np.ndarray, facets: Any, dimension: int, /
) -> Any:
    """Class agreement of the facet halves each undone bisection merges."""

    keys, classes = facets
    tuples = records.tuples[family]
    position = _TABLES[dimension].position[records.tags[family]]
    count = family.size
    first = tuples.copy()
    first[np.arange(count), position] = records.vertices[family]
    second = tuples.copy()
    second[:, 0] = records.vertices[family]
    agree = np.ones((count,), dtype=np.bool_)
    restored, restored_classes, owners = [], [], []
    for column in range(1, dimension + 1):
        active = np.flatnonzero(position != column)
        halves = [
            np.sort(np.delete(value[active], column, axis=1), axis=1)
            for value in (first, second, tuples)
        ]
        first_rows = key_rows(keys, halves[0])
        second_rows = key_rows(keys, halves[1])
        if np.any(first_rows < 0) or np.any(second_rows < 0):
            raise ValueError("BisectionHierarchy records disagree with the mesh facets.")
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
    cells: _Cells,
    flags: _Flags,
    records: _Records,
    protected: np.ndarray,
    facets: Any,
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


def _undone(cells: _Cells, flags: _Flags, records: _Records, family: _Family, /) -> Any:
    """Restore the parents of one coarsening pass in ascending cell-ID order."""

    undo = family.undo
    first_child = np.searchsorted(cells.ids, records.child_ids[undo, 0])
    restored = _Cells(
        records.parent_ids[undo],
        records.rows[undo],
        records.tuples[undo],
        records.tags[undo],
        records.blocks[undo],
        records.parent_ids[undo],
        cells.generations[first_child] - 1,
    )
    restored_flags = _Flags(
        np.ones(undo.shape, dtype=np.bool_),
        np.zeros(undo.shape, dtype=np.bool_),
        flags.classes[first_child],
    )
    merged = _joined(_select(cells, ~family.removed), restored)
    merged_flags = _joined(_select(flags, ~family.removed), restored_flags)
    order = np.argsort(merged.ids, kind="stable")
    keep = np.ones(records.parent_ids.shape, dtype=np.bool_)
    keep[undo] = False
    return _select(merged, order), _select(merged_flags, order), _select(records, keep)


def _subdivision_facets(
    subdivisions: _Subdivisions, rows: np.ndarray, facets: Any, dimension: int, /
) -> Any:
    """Class agreement of the barycentric pieces of each parent facet."""

    keys, classes = facets
    pieces = _SUBDIVISION[dimension][1]
    ascending = np.sort(subdivisions.rows[rows], axis=1)
    extended = np.concatenate((ascending, subdivisions.vertices[rows]), axis=1)
    found = key_rows(keys, np.sort(extended[:, pieces], axis=3).reshape((-1, dimension)))
    if np.any(found < 0):
        raise ValueError("BisectionHierarchy subdivisions disagree with the mesh facets.")
    piece_classes = classes[found].reshape((rows.size, dimension + 1, -1))
    agree = np.all(piece_classes == piece_classes[..., :1], axis=(1, 2))
    restored = np.stack(
        tuple(np.delete(ascending, column, axis=1) for column in range(dimension + 1)),
        axis=1,
    )
    return agree, restored, piece_classes[..., 0]


def _undone_subdivisions(
    cells: _Cells,
    flags: _Flags,
    subdivisions: _Subdivisions,
    protected: np.ndarray,
    facets: Any,
    dimension: int,
    /,
) -> Any:
    """Restore every subdivided source cell whose children and created stars undo.

    A subdivision is removable when all its children are active, marked,
    unblocked, and of one class, its facet pieces agree in class, and the star of
    each created vertex consists only of children of removable subdivisions; the
    last condition is iterated to its fixed point. Returns ``None`` when nothing
    is removable.
    """

    if subdivisions.parent_ids.size == 0:
        return None
    flat = subdivisions.child_ids.reshape((-1,))
    position = np.minimum(np.searchsorted(cells.ids, flat), cells.ids.size - 1)
    present = (cells.ids[position] == flat).reshape(subdivisions.child_ids.shape)
    position = position.reshape(subdivisions.child_ids.shape)
    classes = flags.classes[position]
    candidate = (
        np.all(present, axis=1)
        & np.all(flags.marked[position], axis=1)
        & ~np.any(flags.blocked[position], axis=1)
        & np.all(classes == classes[:, :1], axis=1)
    )
    rows = np.flatnonzero(candidate)
    if rows.size == 0:
        return None
    agree, restored_keys, restored_classes = _subdivision_facets(
        subdivisions, rows, facets, dimension
    )
    candidate[rows[~agree]] = False
    vertex_count = protected.size
    star = np.bincount(cells.rows.reshape((-1,)), minlength=vertex_count)
    while np.any(candidate):
        covered = np.bincount(
            cells.rows[position[candidate].reshape((-1,))].reshape((-1,)),
            minlength=vertex_count,
        )
        good = (star == covered) & ~protected
        kept = candidate & np.all(good[subdivisions.vertices], axis=1)
        if np.array_equal(kept, candidate):
            break
        candidate = kept
    if not np.any(candidate):
        return None
    undo = np.flatnonzero(candidate)
    undone_rows = candidate[rows]
    keys, first = np.unique(
        restored_keys[undone_rows].reshape((-1, dimension)), axis=0, return_index=True
    )
    facet_classes = restored_classes[undone_rows].reshape((-1,))[first]
    count = undo.size
    zeros = np.zeros((count,), dtype=np.int64)
    restored = _Cells(
        subdivisions.parent_ids[undo],
        subdivisions.rows[undo],
        subdivisions.rows[undo],
        zeros,
        subdivisions.blocks[undo],
        subdivisions.parent_ids[undo],
        zeros,
    )
    restored_flags = _Flags(
        np.ones((count,), dtype=np.bool_),
        np.zeros((count,), dtype=np.bool_),
        classes[undo, 0],
    )
    removed = np.zeros(cells.ids.shape, dtype=np.bool_)
    removed[position[undo].reshape((-1,))] = True
    merged = _joined(_select(cells, ~removed), restored)
    merged_flags = _joined(_select(flags, ~removed), restored_flags)
    order = np.argsort(merged.ids, kind="stable")
    width = dimension + 1
    ascending = np.sort(subdivisions.rows[undo], axis=1)
    spans = np.stack(
        tuple(
            np.pad(
                ascending[:, list(subset)],
                ((0, 0), (0, width - len(subset))),
                constant_values=-1,
            )
            for subset in _SUBDIVISION[dimension][0]
        ),
        axis=1,
    )
    arrangements = subdivisions.child_ids.shape[1]
    return (
        _select(merged, order),
        _select(merged_flags, order),
        _select(subdivisions, ~candidate),
        (subdivisions.vertices[undo].reshape((-1,)), spans.reshape((-1, width))),
        subdivisions.child_ids[undo].reshape((-1,)),
        np.repeat(subdivisions.parent_ids[undo], arrangements),
        subdivisions.parent_ids[undo],
        (keys, facet_classes),
    )


def _reverse_supports(steps: Any, vertex_count: int, dimension: int, /) -> np.ndarray:
    """Surviving vertices spanning each source vertex (removed ones by their parents).

    Each step pairs removed vertices with their parent vertices, padded by -1.
    """

    width = dimension + 1
    supports = np.full((vertex_count, width), -1, dtype=np.int64)
    supports[:, 0] = np.arange(vertex_count)
    for vertices, spans in reversed(steps):
        unique, index = np.unique(vertices, return_index=True)
        chosen = spans[index]
        gathered = np.where((chosen >= 0)[..., None], supports[np.maximum(chosen, 0)], -1)
        combined, counts = _distinct_rows(gathered.reshape((unique.size, -1)))
        if np.any(counts > width):
            raise ValueError("A coarsened vertex spans more than one target simplex.")
        supports[unique] = combined[:, :width]
    return supports


def _coarsening(
    source: _Source,
    start: _Start,
    request: _Request,
    blocked: np.ndarray,
    dimension: int,
    /,
) -> _Coarsening:
    cells, records = start.front.cells, start.records
    subdivisions = start.subdivisions
    flags = _Flags(
        np.isin(cells.ids, request.coarsen_ids),
        blocked,
        request.cell_classes[np.searchsorted(source.cells.ids, cells.origins)],
    )
    facets = (request.facet_keys, request.facet_classes)
    # Vertices issued by this adaptation (a repeated subdivision) stay.
    protected = np.concatenate(
        (
            request.protected_vertices,
            np.ones((start.front.growth.levels.size,), dtype=np.bool_),
        )
    )
    width = dimension + 1
    steps, children, parents, undone = [], [], [], []
    while request.coarsen_ids.size:
        family = _removable_family(cells, flags, records, protected, facets, dimension)
        if family is None:
            roots = _undone_subdivisions(
                cells, flags, subdivisions, protected, facets, dimension
            )
            if roots is None:
                break
            cells, flags, subdivisions, step, child_ids, parent_ids, ids, restored = roots
            steps.append(step)
            children.append(child_ids)
            parents.append(parent_ids)
            undone.append(ids)
            facets = (
                np.concatenate((facets[0], restored[0])),
                np.concatenate((facets[1], restored[1])),
            )
            continue
        undo = family.undo
        tuples, tags = records.tuples[undo], records.tags[undo]
        spans = np.full((undo.size, width), -1, dtype=np.int64)
        spans[:, 0] = tuples[:, 0]
        spans[:, 1] = _refinement_ends(tuples, tags, dimension)
        steps.append((records.vertices[undo], spans))
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
    return _Coarsening(
        _select(cells, ~_members(source_ids, cells.ids)),
        removed_ids,
        targets,
        np.concatenate([np.zeros((0,), dtype=np.int64), *undone]),
        removed_vertices,
        _reverse_supports(steps, start.front.vertex_base, dimension),
        len(steps),
        request.coarsen_ids[_members(cells.ids, request.coarsen_ids)],
    )


class _EpochSubdivisions(NamedTuple):
    """Subdivisions one finished device epoch undoes, with its composed lineage."""

    cells: _Cells
    subdivisions: _Subdivisions
    steps: tuple[tuple[np.ndarray, np.ndarray], ...]
    removed_ids: np.ndarray
    link_targets: np.ndarray
    undone_children: np.ndarray
    restored_ids: np.ndarray


def _undone_epoch_subdivisions(
    source: _Source,
    request: _Request,
    subdivisions: _Subdivisions,
    cells: _Cells,
    flags: _Flags,
    facets: tuple[np.ndarray, np.ndarray],
    vertex_count: int,
    removed_ids: np.ndarray,
    link_targets: np.ndarray,
    /,
) -> _EpochSubdivisions | None:
    """Undo the subdivisions exposed once an epoch's bisection coarsening is done.

    Host coarsening removes a subdivision when no bisection family is removable
    any more; a device epoch reaches that state after its compiled passes, so the
    host rule restores the parents of its active cells. ``cells`` are the active
    cells in ascending ID order, ``facets`` their exact facet keys and classes,
    and every vertex issued after the source stays protected. Source cells that
    were already removed keep their target, composed through restored parents.
    """

    protected = np.ones((vertex_count,), dtype=np.bool_)
    protected[: source.vertex_ids.size] = request.protected_vertices
    steps, children, parents, undone = [], [], [], []
    while True:
        roots = _undone_subdivisions(
            cells, flags, subdivisions, protected, facets, source.dimension
        )
        if roots is None:
            break
        cells, flags, subdivisions, step, child_ids, parent_ids, ids, restored = roots
        steps.append(step)
        children.append(child_ids)
        parents.append(parent_ids)
        undone.append(ids)
        facets = (
            np.concatenate((facets[0], restored[0])),
            np.concatenate((facets[1], restored[1])),
        )
    if not steps:
        return None
    link_children = np.concatenate(children)
    order = np.argsort(link_children, kind="stable")
    link_children = link_children[order]
    link_parents = np.concatenate(parents)[order]

    def resolved(values: np.ndarray, /) -> np.ndarray:
        result = values.copy()
        for _ in steps:
            linked = _members(link_children, result)
            result[linked] = link_parents[np.searchsorted(link_children, result[linked])]
        return result

    newly = source.cells.ids[_members(link_children, source.cells.ids)]
    removed = np.concatenate((removed_ids, newly))
    targets = np.concatenate((resolved(link_targets), resolved(newly)))
    order = np.argsort(removed, kind="stable")
    return _EpochSubdivisions(
        cells,
        subdivisions,
        tuple(steps),
        removed[order],
        targets[order],
        link_children,
        np.concatenate(undone),
    )


def _resolved_stencil(growth: _Growth, vertex_base: int, dimension: int, /) -> Any:
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


def _exact_resolved_weights(
    growth: _Growth, vertex_base: int, sources: np.ndarray, /
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Exact rational stencil rows aligned with ``_resolved_stencil`` sources.

    Every growth row is the average of its parents (edge midpoints, face or cell
    centroids). Binary64 weights are exact only for power-of-two parent counts, so
    exact rows are composed only when a barycentric centroid of three parents
    occurs; otherwise ``None`` keeps the binary64 weights as the exact record.
    """
    parent_counts = np.sum(growth.parents >= 0, axis=1)
    if np.all((parent_counts & (parent_counts - 1)) == 0):
        return None
    resolved: list[dict[int, Fraction]] = [{} for _ in range(growth.levels.size)]
    for row in np.argsort(growth.levels, kind="stable").tolist():
        parents = tuple(int(value) for value in growth.parents[row] if value >= 0)
        share = Fraction(1, len(parents))
        if any(
            float(growth.weights[row, column]) != float(share)
            for column in range(len(parents))
        ):
            raise ValueError("A bisection growth vertex is not its parents' average.")
        combined: dict[int, Fraction] = {}
        for parent in parents:
            ancestry = (
                resolved[parent - vertex_base]
                if parent >= vertex_base
                else {parent: Fraction(1)}
            )
            for source_vertex, weight in ancestry.items():
                combined[source_vertex] = (
                    combined.get(source_vertex, Fraction(0)) + share * weight
                )
        resolved[row] = combined
    return tuple(
        tuple(
            Fraction(0) if int(source_vertex) < 0 else ancestry[int(source_vertex)]
            for source_vertex in row_sources
        )
        for ancestry, row_sources in zip(resolved, sources, strict=True)
    )


def _global_vertices(compact: np.ndarray, source: _Source, next_vertex: int, /) -> Any:
    count = source.vertex_ids.size
    return np.where(
        compact < count,
        source.vertex_ids[np.clip(compact, 0, count - 1)],
        next_vertex + compact - count,
    )


def _contained(keys_from: Any, supports: Any, keys_to: Any, /) -> Any:
    """Entities of one mesh lying inside one same-dimension entity of the other."""

    width = keys_from.shape[1]
    missing = keys_from[key_rows(keys_to, keys_from) < 0]
    spans, counts = _distinct_rows(
        supports[missing].reshape((missing.shape[0], width * supports.shape[1]))
    )
    candidates = spans[:, :width]
    found = np.zeros(counts.shape, dtype=np.bool_)
    single = np.flatnonzero(counts == width)
    found[single] = key_rows(keys_to, candidates[single]) >= 0
    return missing[found], candidates[found]


def _grouped_relations(degree: int, groups: Any, /) -> EntityRelations:
    """One relation record from (source keys, target keys, kind) groups."""

    return EntityRelations(
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


def _relations(
    source: _Source,
    start: _Start,
    cells: _Cells,
    stencil: tuple[np.ndarray, np.ndarray],
    coarsening: _Coarsening,
    entity_tables: tuple[tuple[np.ndarray, np.ndarray], ...],
    /,
) -> tuple[EntityRelations, ...]:
    """Non-identity lineage of every dimension; identities stay implicit."""

    dimension, count = source.dimension, source.vertex_ids.size

    def ids(values: np.ndarray) -> np.ndarray:
        return _global_vertices(values, source, start.next_vertex)

    split_sources, _ = stencil
    records = [
        EntityRelations(
            0,
            np.empty((0, 1), dtype=np.int64),
            np.empty((0, 1), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
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
                    (ids(within), ids(refined), EntityLineageKind.REFINED_FROM),
                    (ids(merged), ids(into), EntityLineageKind.COARSENED_INTO),
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
                    EntityLineageKind.REFINED_FROM,
                ),
                (
                    coarsening.removed_ids[:, None],
                    coarsening.link_targets[:, None],
                    EntityLineageKind.COARSENED_INTO,
                ),
            ),
        )
    )
    return tuple(records)


def _entity_identities(
    source: _Source,
    start: _Start,
    entity_tables: tuple[tuple[np.ndarray, np.ndarray], ...],
    alive: np.ndarray,
    /,
) -> Any:
    """Restored entities regain retired IDs; newly retired entities are recorded.

    Only entities whose vertices all survive can reappear, so retired entries
    touching a removed vertex are dropped.
    """

    prescribed, retired = [], []
    for degree, (keys, targets) in enumerate(entity_tables, start=1):
        old_keys, old_ids = start.retired[degree - 1]
        created = targets[key_rows(keys, targets) < 0]
        restore = key_rows(old_keys, created)
        restored = restore >= 0
        prescribed.append(
            PrescribedEntityIds(
                degree,
                _global_vertices(created[restored], source, start.next_vertex),
                old_ids[restore[restored]],
            )
        )
        keep = np.all(alive[old_keys], axis=1)
        keep[restore[restored]] = False
        vanished = np.flatnonzero(key_rows(targets, keys) < 0)
        vanished = vanished[np.all(alive[keys[vanished]], axis=1)]
        entity_ids = np.asarray(source.mesh.entity_set(degree).entity_ids, dtype=np.int64)
        table_keys = np.concatenate((old_keys[keep], keys[vanished]))
        table_ids = np.concatenate((old_ids[keep], entity_ids[vanished]))
        order = np.lexsort(table_keys.T[::-1])
        retired.append((source.vertex_ids[table_keys[order]], table_ids[order]))
    return tuple(prescribed), tuple(retired)


def _merged_forest(
    start: _Start, refinement: _Refinement, coarsening: _Coarsening
) -> Any:
    cells = refinement.front.cells
    cells = _joined(
        _select(cells, ~_members(coarsening.removed_ids, cells.ids)),
        coarsening.restored,
    )
    records = start.records
    records = _joined(
        _select(records, ~np.isin(records.parent_ids, coarsening.undone)),
        refinement.records,
    )
    subdivisions = start.subdivisions
    subdivisions = _select(
        subdivisions, ~np.isin(subdivisions.parent_ids, coarsening.undone)
    )
    return (
        _select(cells, np.argsort(cells.ids, kind="stable")),
        _select(records, np.argsort(records.parent_ids, kind="stable")),
        _select(subdivisions, np.argsort(subdivisions.parent_ids, kind="stable")),
    )


def _target_hierarchy(
    source: _Source,
    start: _Start,
    cells: _Cells,
    records: _Records,
    subdivisions: _Subdivisions,
    retired: Any,
    front: _Front,
    /,
) -> BisectionHierarchy:
    def ids(values: np.ndarray) -> np.ndarray:
        return _global_vertices(values, source, start.next_vertex)

    return BisectionHierarchy(
        source.dimension,
        cells.ids,
        ids(cells.tuples),
        cells.tags,
        cells.generations,
        record_parent_ids=records.parent_ids,
        record_parent_blocks=records.blocks,
        record_parent_rows=ids(records.rows),
        record_parent_vertices=ids(records.tuples),
        record_parent_tags=records.tags,
        record_child_ids=records.child_ids,
        record_vertex_ids=ids(records.vertices),
        subdivision_parent_ids=subdivisions.parent_ids,
        subdivision_parent_blocks=subdivisions.blocks,
        subdivision_parent_rows=ids(subdivisions.rows),
        subdivision_child_ids=subdivisions.child_ids,
        subdivision_vertex_ids=ids(subdivisions.vertices),
        retired_entity_keys=tuple(value[0] for value in retired),
        retired_entity_ids=tuple(value[1] for value in retired),
        next_vertex_id=start.next_vertex + front.growth.levels.size,
        next_cell_id=front.next_cell,
    )


def _target_vertices(source: _Source, removed: np.ndarray, new_count: int, /) -> Any:
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


def _target_stencil(source: _Source, stencil: Any, ordered: np.ndarray, /) -> Any:
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


def _edit(
    source: _Source,
    start: _Start,
    refinement: _Refinement,
    coarsening: _Coarsening,
    cells: _Cells,
    /,
) -> Any:
    dimension, count = source.dimension, source.vertex_ids.size
    growth = refinement.front.growth
    stencil = _resolved_stencil(growth, count, dimension)
    ordered, position, alive = _target_vertices(
        source, coarsening.removed_vertices, growth.levels.size
    )
    # Removed new vertices keep their stencil for composition but relate nothing.
    stencil = (np.where(alive[count:, None], stencil[0], -1), stencil[1])
    coordinates, sources, weights, valid = _target_stencil(source, stencil, ordered)
    exact_new = _exact_resolved_weights(growth, count, stencil[0])
    source_row = (Fraction(1),) + (Fraction(0),) * dimension
    exact_weights = (
        None
        if exact_new is None
        else tuple(
            source_row if row < count else exact_new[row - count]
            for row in ordered.tolist()
        )
    )
    entity_tables = tuple(
        (
            np.searchsorted(source.vertex_ids, entity_keys(source.mesh, degree)),
            _simplex_keys(cells.rows, degree + 1),
        )
        for degree in range(1, dimension)
    )
    prescribed, retired = _entity_identities(source, start, entity_tables, alive)
    block_rows = tuple(
        np.flatnonzero(cells.blocks == index) for index in range(len(source.mesh.blocks))
    )
    edit = CellTopologyEdit(
        coordinates,
        _global_vertices(ordered, source, start.next_vertex),
        source_family_blocks(
            tuple((block.name, block.cell_kind) for block in source.mesh.blocks),
            tuple(position[cells.rows[rows]].astype(np.int32) for rows in block_rows),
            tuple(cells.ids[rows] for rows in block_rows),
        ),
        sources,
        weights,
        valid,
        _relations(source, start, cells, stencil, coarsening, entity_tables),
        prescribed,
        stencil_exact_weights=exact_weights,
    )
    return edit, retired


def _refinement_operation_rows(
    source: _Source,
    start: _Start,
    refinement: _Refinement,
    records: _Records,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expose each retained split's source-cell orbit and exact endpoint ancestry."""
    old_records = set(
        int(value) for value in np.asarray(start.records.parent_ids, dtype=np.int64)
    )
    parent_ids = np.asarray(records.parent_ids, dtype=np.int64)
    child_ids = np.asarray(records.child_ids, dtype=np.int64)
    tags = np.asarray(records.tags, dtype=np.int64)
    tuples = np.asarray(records.tuples, dtype=np.int64)
    new_rows = [
        row
        for row, identifier in enumerate(parent_ids)
        if int(identifier) not in old_records
    ]
    origin_by_cell = {
        int(identifier): int(origin)
        for identifier, origin in zip(
            np.asarray(start.front.cells.ids, dtype=np.int64),
            np.asarray(start.front.cells.origins, dtype=np.int64),
            strict=True,
        )
    }
    generated: list[tuple[int, int]] = []
    pending = list(new_rows)
    while pending:
        remaining = []
        for row in pending:
            parent = int(parent_ids[row])
            origin = origin_by_cell.get(parent)
            if origin is None:
                remaining.append(row)
                continue
            generated.append((row, origin))
            for child in child_ids[row]:
                origin_by_cell[int(child)] = origin
        if len(remaining) == len(pending):
            raise ValueError("A bisection split has no exact source cell-orbit ancestor.")
        pending = remaining

    source_count = source.vertex_ids.size
    next_vertex = start.next_vertex
    parents = np.asarray(refinement.front.growth.parents, dtype=np.int64)
    weights = np.asarray(refinement.front.growth.weights, dtype=np.float64)
    supports: dict[int, frozenset[int]] = {
        row: frozenset((int(identifier),))
        for row, identifier in enumerate(np.asarray(source.vertex_ids, dtype=np.int64))
    }
    for row in range(parents.shape[0]):
        compact_id = source_count + row
        support: set[int] = set()
        for parent, weight in zip(parents[row], weights[row], strict=True):
            if parent < 0 or weight == 0.0:
                continue
            inherited = supports.get(int(parent))
            if inherited is None:
                raise ValueError(
                    "A bisection split vertex has unknown exact source stencil ancestry."
                )
            support.update(inherited)
        supports[compact_id] = frozenset(support)

    edge_lookup: dict[tuple[int, ...], int] = {}
    source_edge_ids = np.asarray(source.mesh.entity_set(1).entity_ids, dtype=np.int64)
    source_edge_keys = entity_keys(source.mesh, 1)
    for key, identifier in zip(source_edge_keys, source_edge_ids, strict=True):
        vertices = tuple(int(value) for value in key if int(value) >= 0)
        edge_lookup[vertices] = int(identifier)

    def global_vertex(compact: int, /) -> int:
        if compact < source_count:
            return int(source.vertex_ids[compact])
        return int(next_vertex + compact - source_count)

    split_parents = np.empty((len(generated),), dtype=np.int64)
    split_source_edges = np.full((len(generated),), -1, dtype=np.int64)
    split_vertices = np.empty((len(generated), 2), dtype=np.int64)
    for output_row, (record_row, origin) in enumerate(generated):
        position = int(_TABLES[source.dimension].position[tags[record_row]])
        first = int(tuples[record_row, 0])
        second = int(tuples[record_row, position])
        endpoints = tuple(sorted((global_vertex(first), global_vertex(second))))
        split_parents[output_row] = origin
        split_vertices[output_row] = endpoints
        support = supports.get(first, frozenset()) | supports.get(second, frozenset())
        if len(support) == 2:
            split_source_edges[output_row] = edge_lookup.get(tuple(sorted(support)), -1)
    return split_parents, split_source_edges, split_vertices


def execute_bisection(
    mesh: CellMesh,
    refine_cell_ids: np.ndarray,
    coarsen_cell_ids: np.ndarray,
    /,
    *,
    hierarchy: BisectionHierarchy | None,
    compatibility: BisectionCompatibility,
    protected_edges: np.ndarray,
    protected_vertices: np.ndarray,
    cell_classes: np.ndarray,
    facet_classes: np.ndarray,
    periodic_edge_lock: PeriodicRefinementEdgeLock | None = None,
) -> BisectionOutcome:
    """Refine and coarsen one simplex mesh by tagged-simplex bisection.

    Refinement marks whose own conformity closure would split a protected edge are
    rejected (reported in the evidence); the rest are bisected once and closed to
    a conforming mesh. Coarsening removes, pass by pass, every unprotected
    bisection vertex whose star is the marked, class-uniform children of its
    bisections and does not touch the refinement closure. Children stay in their
    parent's block with the parent's orientation.
    """

    if not isinstance(compatibility, BisectionCompatibility):
        raise TypeError("compatibility must be BisectionCompatibility.")
    source = _prepared_source(mesh)
    dimension = source.dimension
    request = _prepared_request(
        source,
        refine_cell_ids,
        coarsen_cell_ids,
        protected_edges,
        protected_vertices,
        cell_classes,
        facet_classes,
    )
    if periodic_edge_lock is not None and not isinstance(
        periodic_edge_lock, PeriodicRefinementEdgeLock
    ):
        raise TypeError("periodic_edge_lock must be PeriodicRefinementEdgeLock or None.")
    if (
        source.mesh.periodic_topology is not None
        and request.refine_ids.size
        and periodic_edge_lock is None
    ):
        raise ValueError(
            "Periodic bisection requires a source-bound selected-edge orbit lock."
        )
    if hierarchy is None:
        start = _labelled_start(
            source,
            request,
            compatibility,
            periodic_edge_lock=periodic_edge_lock,
        )
    else:
        start = _bound_start(source, request, hierarchy)
        if request.refine_ids.size:
            start = _resubdivided(start, request, compatibility, dimension)
        if periodic_edge_lock is not None:
            front = _periodic_locked_front(
                source, start.front, request, periodic_edge_lock
            )
            incompatible = _incompatible_facets(front.cells, dimension)
            if incompatible:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SPECIFICATION,
                    "Locked periodic Maubach edges violate the local matching condition.",
                    stage="bisection-closure",
                )
            start = start._replace(front=front, compatible=True, incompatible=0)
    refinement = _refinement(start, request, dimension)
    blocked = np.isin(start.front.cells.ids, refinement.records.parent_ids)
    coarsening = _coarsening(source, start, request, blocked, dimension)
    cells, records, subdivisions = _merged_forest(start, refinement, coarsening)
    edit, retired = _edit(source, start, refinement, coarsening, cells)
    evidence = BisectionEvidence(
        requested_refinements=request.refine_ids.size,
        accepted_refinements=request.refine_ids.size - refinement.rejected_ids.size,
        rejected_refinement_ids=refinement.rejected_ids,
        admissibility_tests=refinement.tests,
        bisections=refinement.records.parent_ids.size,
        closure_iterations=refinement.iterations,
        created_vertices=refinement.front.growth.levels.size,
        maximum_generation=int(np.max(cells.generations)),
        initially_compatible=start.compatible,
        incompatible_facets=start.incompatible,
        uniform_refinement_applied=start.uniform,
        requested_coarsenings=request.coarsen_ids.size,
        coarsened_vertices=coarsening.removed_vertices.size,
        coarsening_passes=coarsening.passes,
        restored_cells=coarsening.restored.ids.size,
        rejected_coarsening_ids=coarsening.rejected_ids,
    )
    refinement_parents, refinement_edges, refinement_edge_vertices = (
        _refinement_operation_rows(source, start, refinement, records)
    )
    if periodic_edge_lock is not None:
        locked_cells, locked_edges = periodic_edge_lock.require_source(mesh)
        present = {
            (int(parent), int(edge))
            for parent, edge in zip(refinement_parents, refinement_edges, strict=True)
        }
        if any(
            (int(parent), int(edge)) not in present
            for parent, edge in zip(locked_cells, locked_edges, strict=True)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SPECIFICATION,
                "Periodic bisection did not execute every locked parent-edge pair.",
                stage="bisection-closure",
            )
    return BisectionOutcome(
        edit,
        _target_hierarchy(
            source, start, cells, records, subdivisions, retired, refinement.front
        ),
        evidence,
        refinement_parents,
        refinement_edges,
        refinement_edge_vertices,
    )


__all__ = [
    "BisectionCompatibility",
    "BisectionEvidence",
    "BisectionHierarchy",
    "BisectionOutcome",
    "execute_bisection",
]
