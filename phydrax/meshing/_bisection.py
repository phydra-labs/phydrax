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
labellings that violate it are rejected, first made compatible by one
barycentric subdivision, whose simplices are pairwise reflected neighbors, or,
for triangle meshes, closed locally under their original labels.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from itertools import combinations
from typing import Any, final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    current_native_execution_budget,
    current_native_host_workspace,
    NativeExecutionBudget,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._adaptive_simplex import maubach_bisection_tables
from ..discretization._cell_geometry_transfer import NestedReferenceWitnesses
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._lineage import EntityLineageKind
from ._result import CellMeshingResult
from ._topology_edit import (
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    key_rows,
    nested_reference_vertices,
    PrescribedEntityIds,
    source_family_blocks,
)


class BisectionCompatibility(StrEnum):
    """Handling of an initial labelling that violates the matching condition.

    ``REJECT`` refuses it. ``UNIFORM_REFINEMENT`` first subdivides every simplex
    barycentrically into pairwise reflected neighbors. ``CONFORMING_CLOSURE``
    keeps the longest-edge labels of a triangle mesh and lets the conformity
    closure bisect every neighbor across a split edge whose refinement edge
    differs. Two-dimensional newest-vertex bisection needs no matching condition:
    bisecting a triangle twice splits all three of its edges, so every even
    uniform generation is conforming and bounds the closure, whose size estimate
    holds for arbitrary initial refinement edges (Karkulik, Pavlicek, Praetorius
    2013). From a labelled start the closure splits only source edges, so each
    source triangle yields at most four children. Tetrahedral closure relies on
    the matching condition; incompatible tetrahedral labels stay refused.
    """

    REJECT = "reject"
    UNIFORM_REFINEMENT = "uniform_refinement"
    CONFORMING_CLOSURE = "conforming_closure"


def _identifiers(values: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1 or np.any(array < 0):
        raise ValueError(f"{name} must be a rank-1 array of non-negative IDs.")
    return array


def _tag_array(values: ArrayLike, count: int, dimension: int, name: str, /) -> np.ndarray:
    tags = np.asarray(values, dtype=np.int32)
    if tags.shape != (count,) or np.any(tags < 1) or np.any(tags > dimension):
        raise ValueError(f"{name} must hold one tag in 1..{dimension} per simplex.")
    return tags


def _simplex_rows(
    values: ArrayLike, count: int, dimension: int, name: str, /
) -> np.ndarray:
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


def _validated_forest(
    dimension: int,
    cell_global_ids: ArrayLike,
    ordered_vertices: ArrayLike,
    tags: ArrayLike,
    generations: ArrayLike,
    records: dict[str, ArrayLike],
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
    if np.unique(children).size != children.size:
        raise ValueError("Every bisection record owns two distinct child IDs.")
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
        "tags": _tag_array(tags, count, dimension, "tags"),
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
    }


def _require_uniform_action_family(barycentric: np.ndarray, dimension: int, /) -> None:
    width = dimension + 1
    arrangements = _TABLES[dimension][4].shape[0]
    expected = {
        tuple(
            sorted(
                tuple(
                    1.0 / size if axis in order[:size] else 0.0 for axis in range(width)
                )
                for size in range(1, width + 1)
            )
        )
        for order in _TABLES[dimension][4]
    }
    for sibling_actions in barycentric:
        actual = [
            tuple(sorted(tuple(point.tolist()) for point in action))
            for action in sibling_actions
        ]
        if len(set(actual)) != arrangements or set(actual) != expected:
            raise ValueError(
                "Uniform actions must be the complete canonical barycentric source subdivision."
            )


@final
class BisectionUniformRefinement(StrictModule, NonTrainableState):
    """Actual compatibility siblings and their original source reference actions."""

    dimension: int = eqx.field(static=True)
    parent_ids: Array
    parent_blocks: Array
    parent_row_order: Array
    parent_rows: Array
    parent_vertices: Array
    parent_tags: Array
    parent_levels: Array
    parent_classes: Array
    parent_facet_classes: Array
    child_ids: Array
    child_vertices: Array
    reference_vertices: Array
    barycentric_weights: Array
    source: CellMeshingResult
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        parent_ids: ArrayLike,
        parent_blocks: ArrayLike,
        parent_row_order: ArrayLike,
        parent_rows: ArrayLike,
        parent_vertices: ArrayLike,
        parent_tags: ArrayLike,
        parent_levels: ArrayLike,
        parent_classes: ArrayLike,
        parent_facet_classes: ArrayLike,
        child_ids: ArrayLike,
        child_vertices: ArrayLike,
        reference_vertices: ArrayLike,
        barycentric_weights: ArrayLike,
        /,
        *,
        source: CellMeshingResult,
    ) -> None:
        if isinstance(dimension, bool) or dimension not in (2, 3):
            raise ValueError("Uniform simplex lineage requires dimension two or three.")
        ids = _identifiers(parent_ids, "uniform_parent_ids")
        count, width = ids.size, dimension + 1
        if count == 0 or np.any(ids[1:] <= ids[:-1]):
            raise ValueError(
                "Uniform parent IDs must be nonempty and strictly increasing."
            )
        children = np.asarray(child_ids, dtype=np.int64)
        arrangements = _TABLES[dimension][4].shape[0]
        if (
            children.shape != (count, arrangements)
            or np.any(children < 0)
            or np.unique(children).size != children.size
        ):
            raise ValueError(
                "Uniform siblings require all six or twenty-four distinct child IDs."
            )
        if np.intersect1d(ids, children).size:
            raise ValueError(
                "Uniform parents and their children must have distinct identities."
            )
        rows = _simplex_rows(parent_rows, count, dimension, "uniform_parent_rows")
        vertices = _simplex_rows(
            parent_vertices, count, dimension, "uniform_parent_vertices"
        )
        if not np.array_equal(np.sort(rows, axis=1), np.sort(vertices, axis=1)):
            raise ValueError(
                "Uniform source rows and tagged tuples must own the same vertices."
            )
        child_rows = np.asarray(child_vertices, dtype=np.int64)
        if child_rows.shape != (count, arrangements, width) or np.any(child_rows < 0):
            raise ValueError(
                "Uniform child vertex identities must align with all actual siblings."
            )
        if np.any(np.diff(np.sort(child_rows, axis=2), axis=2) == 0):
            raise ValueError("Every uniform child must have distinct vertex identities.")
        references = np.asarray(reference_vertices, dtype=np.float64)
        if references.shape != (count, arrangements, width, dimension) or not np.all(
            np.isfinite(references)
        ):
            raise ValueError(
                "Uniform reference actions must be finite complete simplex corner maps."
            )
        if np.any(references < 0.0) or np.any(np.sum(references, axis=3) > 1.0):
            raise ValueError(
                "Uniform reference corners must lie in their actual source simplex."
            )
        barycentric = np.asarray(barycentric_weights, dtype=np.float64)
        if (
            barycentric.shape != (count, arrangements, width, width)
            or not np.all(np.isfinite(barycentric))
            or np.any(barycentric < 0.0)
            or not np.all(np.sum(barycentric, axis=3) == 1.0)
            or not np.array_equal(barycentric[..., 1:], references)
        ):
            raise ValueError(
                "Uniform actions must retain their complete original barycentric construction weights."
            )
        _require_uniform_action_family(barycentric, dimension)
        construction: dict[int, tuple[tuple[int, float], ...]] = {}
        construction_vertices: dict[tuple[tuple[int, float], ...], int] = {}
        for parent_index in range(count):
            for child_index in range(arrangements):
                for vertex_index in range(width):
                    vertex = int(child_rows[parent_index, child_index, vertex_index])
                    key = tuple(
                        sorted(
                            (
                                int(rows[parent_index, source_index]),
                                float(
                                    barycentric[
                                        parent_index,
                                        child_index,
                                        vertex_index,
                                        source_index,
                                    ]
                                ),
                            )
                            for source_index in range(width)
                            if barycentric[
                                parent_index, child_index, vertex_index, source_index
                            ]
                        )
                    )
                    known = construction.setdefault(vertex, key)
                    if known != key:
                        raise ValueError(
                            "Incident uniform siblings must retain the same original barycentric vertex action."
                        )
                    owner = construction_vertices.setdefault(key, vertex)
                    if owner != vertex:
                        raise ValueError(
                            "A shared original barycentric construction must have one scientific vertex identity."
                        )
        original_vertices = set(rows.reshape(-1).tolist())
        if not original_vertices.issubset(construction):
            raise ValueError(
                "Uniform subdivision must contain every original scientific source vertex."
            )
        for vertex, key in construction.items():
            if vertex in original_vertices and key != ((vertex, 1.0),):
                raise ValueError(
                    "Uniform subdivision must retain every original vertex under its scientific identity."
                )
        blocks = np.asarray(parent_blocks, dtype=np.int32)
        row_order = np.asarray(parent_row_order, dtype=np.int64)
        levels = np.asarray(parent_levels, dtype=np.int32)
        classes = np.asarray(parent_classes, dtype=np.int64)
        facet_classes = np.asarray(parent_facet_classes, dtype=np.int64)
        if any(value.shape != (count,) for value in (blocks, row_order, levels, classes)):
            raise ValueError(
                "Uniform source blocks, row order, levels and classes must align with parents."
            )
        if np.any(blocks < 0) or np.any(row_order < 0) or np.any(levels < 0):
            raise ValueError(
                "Uniform source blocks, row order and levels must be nonnegative."
            )
        if facet_classes.shape != (count, width):
            raise ValueError(
                "Uniform source facet classes must align with all parent facets."
            )
        if np.unique(np.stack((blocks, row_order), axis=1), axis=0).shape[0] != count:
            raise ValueError(
                "Uniform source rows must have distinct explicit block positions."
            )
        source_facets: dict[tuple[int, ...], int] = {}
        for parent_index in range(count):
            for facet_index, local in enumerate(combinations(range(width), dimension)):
                key = tuple(sorted(int(rows[parent_index, index]) for index in local))
                cell_class = int(facet_classes[parent_index, facet_index])
                known = source_facets.setdefault(key, cell_class)
                if known != cell_class:
                    raise ValueError(
                        "Incident uniform parents must agree on their original source facet class."
                    )
        if (
            not isinstance(source, CellMeshingResult)
            or source.mesh.topological_dimension != dimension
        ):
            raise TypeError(
                "Uniform source lineage requires its actual accepted scientific source carrier."
            )
        source_vertices = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
        for identifier, block, row, parent in zip(
            ids, blocks, row_order, rows, strict=True
        ):
            if block >= len(source.mesh.blocks):
                raise ValueError(
                    "Uniform parent blocks must belong to the retained original scientific source."
                )
            original = source.mesh.blocks[int(block)]
            if (
                row >= original.cell_count
                or np.asarray(original.global_ids)[row] != identifier
            ):
                raise ValueError(
                    "Uniform parent IDs and source row order must bind the retained scientific source."
                )
            if not np.array_equal(
                source_vertices[np.asarray(original.vertices)[row]], parent
            ):
                raise ValueError(
                    "Uniform parent vertex rows must bind the retained original scientific source."
                )
        arrays = {
            "parent_ids": ids,
            "parent_blocks": blocks,
            "parent_row_order": row_order,
            "parent_rows": rows,
            "parent_vertices": vertices,
            "parent_tags": _tag_array(
                parent_tags, count, dimension, "uniform_parent_tags"
            ),
            "parent_levels": levels,
            "parent_classes": classes,
            "parent_facet_classes": facet_classes,
            "child_ids": children,
            "child_vertices": child_rows,
            "reference_vertices": references,
            "barycentric_weights": barycentric,
        }
        self.dimension = dimension
        self.source = source
        self.parent_ids = jnp.asarray(arrays["parent_ids"])
        self.parent_blocks = jnp.asarray(arrays["parent_blocks"])
        self.parent_row_order = jnp.asarray(arrays["parent_row_order"])
        self.parent_rows = jnp.asarray(arrays["parent_rows"])
        self.parent_vertices = jnp.asarray(arrays["parent_vertices"])
        self.parent_tags = jnp.asarray(arrays["parent_tags"])
        self.parent_levels = jnp.asarray(arrays["parent_levels"])
        self.parent_classes = jnp.asarray(arrays["parent_classes"])
        self.parent_facet_classes = jnp.asarray(arrays["parent_facet_classes"])
        self.child_ids = jnp.asarray(arrays["child_ids"])
        self.child_vertices = jnp.asarray(arrays["child_vertices"])
        self.reference_vertices = jnp.asarray(arrays["reference_vertices"])
        self.barycentric_weights = jnp.asarray(arrays["barycentric_weights"])
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "bisection-uniform-refinement",
                "dimension": dimension,
                "source": source.result_id,
                "arrays": array_tree_fingerprint(arrays),
            }
        )

    def host_arrays(self) -> dict[str, np.ndarray]:
        return {
            "parent_ids": np.asarray(self.parent_ids),
            "parent_blocks": np.asarray(self.parent_blocks),
            "parent_row_order": np.asarray(self.parent_row_order),
            "parent_rows": np.asarray(self.parent_rows),
            "parent_vertices": np.asarray(self.parent_vertices),
            "parent_tags": np.asarray(self.parent_tags),
            "parent_levels": np.asarray(self.parent_levels),
            "parent_classes": np.asarray(self.parent_classes),
            "parent_facet_classes": np.asarray(self.parent_facet_classes),
            "child_ids": np.asarray(self.child_ids),
            "child_vertices": np.asarray(self.child_vertices),
            "reference_vertices": np.asarray(self.reference_vertices),
            "barycentric_weights": np.asarray(self.barycentric_weights),
        }

    def select(self, rows: np.ndarray, /) -> BisectionUniformRefinement | None:
        if rows.size == 0:
            return None
        if rows.size == self.parent_ids.size and all(
            row == index for index, row in enumerate(rows)
        ):
            return self
        arrays = self.host_arrays()
        return BisectionUniformRefinement(
            self.dimension,
            *(value[rows] for value in arrays.values()),
            source=self.source,
        )


def _reconcile_uniform_refinement(
    regenerated: BisectionUniformRefinement,
    retained: BisectionUniformRefinement,
    restored_parent_ids: ArrayLike,
    /,
) -> tuple[BisectionUniformRefinement, dict[int, int]]:
    """Replace actual restored roots while preserving live construction SCIs.

    The owning collective preparation authenticates which roots were restored.
    Only unaffected, still-live uniform roots supply reusable vertices; retired
    construction identities never become a source for a new issuance.
    """
    if (
        regenerated.dimension != retained.dimension
        or regenerated.source.result_id != retained.source.result_id
    ):
        raise ValueError(
            "Uniform reconciliation requires the same genuine original scientific source."
        )
    width = retained.dimension + 1
    entries = regenerated.child_vertices.size + retained.child_vertices.size
    _uniform_charge(entries * width, entries * (512 + 128 * width))
    original = retained.host_arrays()
    incoming = regenerated.host_arrays()
    restored = _identifiers(restored_parent_ids, "restored_uniform_parent_ids")
    if restored.size == 0 or np.any(restored[1:] <= restored[:-1]):
        raise ValueError(
            "Uniform reconciliation requires actual distinct restored parents in SCI order."
        )
    if not np.array_equal(incoming["parent_ids"], restored):
        raise ValueError(
            "Regenerated uniform roots differ from the actual restored parent cohort."
        )
    positions = np.searchsorted(original["parent_ids"], restored)
    if np.any(positions >= original["parent_ids"].size):
        raise ValueError(
            "Regenerated uniform roots contain an undeclared original parent."
        )
    if not np.array_equal(original["parent_ids"][positions], restored):
        raise ValueError(
            "Regenerated uniform roots contain an undeclared original parent."
        )
    for name in (
        "parent_blocks",
        "parent_row_order",
        "parent_rows",
        "parent_vertices",
        "parent_tags",
        "parent_levels",
        "parent_classes",
        "parent_facet_classes",
    ):
        if not np.array_equal(original[name][positions], incoming[name]):
            raise ValueError(
                f"Uniform reconciliation changed the original scientific {name}."
            )

    type Construction = tuple[tuple[int, float], ...]
    live: dict[Construction, int] = {}
    live_vertices: dict[int, Construction] = {}
    kept = np.ones(original["parent_ids"].shape, dtype=np.bool_)
    kept[positions] = False

    def construction(corners: np.ndarray, weights: np.ndarray) -> Construction:
        return tuple(
            sorted(
                (int(identifier), float(weight))
                for identifier, weight in zip(corners, weights, strict=True)
                if weight != 0.0
            )
        )

    for parent in np.flatnonzero(kept):
        corners = original["parent_rows"][parent]
        for vertices, weights in zip(
            original["child_vertices"][parent],
            original["barycentric_weights"][parent],
            strict=True,
        ):
            for identifier, coefficients in zip(vertices, weights, strict=True):
                key = construction(corners, coefficients)
                identifier = int(identifier)
                if key in live and live[key] != identifier:
                    raise ValueError(
                        "Live uniform roots disagree on an actual shared construction SCI."
                    )
                if identifier in live_vertices and live_vertices[identifier] != key:
                    raise ValueError(
                        "A live uniform construction SCI has contradictory original source support."
                    )
                live[key], live_vertices[identifier] = identifier, key

    remap: dict[int, int] = {}
    generated_vertices: dict[int, Construction] = {}
    child_vertices = np.array(incoming["child_vertices"], copy=True)
    for parent, corners in enumerate(incoming["parent_rows"]):
        for child, (vertices, weights) in enumerate(
            zip(
                incoming["child_vertices"][parent],
                incoming["barycentric_weights"][parent],
                strict=True,
            )
        ):
            for corner, (identifier, coefficients) in enumerate(
                zip(vertices, weights, strict=True)
            ):
                identifier = int(identifier)
                key = construction(corners, coefficients)
                if (
                    identifier in generated_vertices
                    and generated_vertices[identifier] != key
                ):
                    raise ValueError(
                        "A regenerated uniform vertex has contradictory original source support."
                    )
                generated_vertices[identifier] = key
                if key in live:
                    replacement = live[key]
                    if len(key) == 1 and key[0][1] == 1.0 and replacement != identifier:
                        raise ValueError(
                            "Uniform reconciliation cannot rename an original scientific source corner."
                        )
                    if replacement != identifier:
                        remap[identifier] = replacement
                        child_vertices[parent, child, corner] = replacement
                elif identifier in live_vertices:
                    raise ValueError(
                        "A regenerated uniform vertex collides with a different live scientific construction."
                    )
    combined = dict(original)
    for name in (
        "child_ids",
        "child_vertices",
        "reference_vertices",
        "barycentric_weights",
    ):
        combined[name] = np.array(original[name], copy=True)
        combined[name][positions] = (
            child_vertices if name == "child_vertices" else incoming[name]
        )
    return BisectionUniformRefinement(
        retained.dimension,
        *combined.values(),
        source=retained.source,
    ), remap


@final
class BisectionHierarchy(StrictModule, NonTrainableState):
    """Persistent Maubach labels and bisection forest of one bisection mesh.

    Active cells are listed by ascending global ID with their canonical Maubach
    tuple (vertex global IDs), tag, and generation. Each forest record is one
    bisection whose children are leaves or have descendants: the parent cell ID,
    source block, exact block row (global IDs), Maubach tuple and tag, the two
    child cell IDs (first, second), and the bisection vertex. Retired entity
    tables keep the global IDs of edges/faces removed by refinement so that
    coarsening restores them under their original IDs. Vertex and cell IDs are
    issued from the high-water counters and never reused.
    ``scientific_cell_ids`` and ``scientific_block_ids`` retain the original
    authored block authority of every issued cell, including inactive parents.
    Descendants inherit it at birth; coordinate presentation regrouping never
    rewrites it. The forest's source-block rows remain presentation routing.

    A hierarchy applies to a mesh iff its active cell IDs equal the mesh cell IDs
    and every cell has the same vertex-ID set in both.
    """

    dimension: int = eqx.field(static=True)
    cell_global_ids: Array
    ordered_vertices: Array
    tags: Array
    generations: Array
    scientific_cell_ids: Array
    scientific_block_ids: Array
    record_parent_ids: Array
    record_parent_blocks: Array
    record_parent_rows: Array
    record_parent_vertices: Array
    record_parent_tags: Array
    record_child_ids: Array
    record_vertex_ids: Array
    retired_entity_keys: tuple[Array, ...]
    retired_entity_ids: tuple[Array, ...]
    uniform_refinement: BisectionUniformRefinement | None
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
        retired_entity_keys: tuple[ArrayLike, ...],
        retired_entity_ids: tuple[ArrayLike, ...],
        next_vertex_id: int,
        next_cell_id: int,
        uniform_refinement: BisectionUniformRefinement | None,
        scientific_cell_ids: ArrayLike,
        scientific_block_ids: ArrayLike,
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
        )
        scientific_ids = np.asarray(scientific_cell_ids, dtype=np.int64)
        scientific_blocks = np.asarray(scientific_block_ids, dtype=np.int32)
        if (
            scientific_ids.ndim != 1
            or scientific_blocks.shape != scientific_ids.shape
            or np.any(scientific_blocks < 0)
            or np.any(scientific_ids < 0)
            or np.any(np.diff(scientific_ids) <= 0)
            or not np.all(np.isin(arrays["cell_global_ids"], scientific_ids))
            or not np.all(np.isin(arrays["record_parent_ids"], scientific_ids))
            or not np.all(np.isin(arrays["record_child_ids"], scientific_ids))
        ):
            raise ValueError(
                "Scientific block authority must cover active and forest cells in canonical ID order."
            )
        authority_parts = [
            arrays["cell_global_ids"],
            arrays["record_parent_ids"],
            arrays["record_child_ids"].reshape(-1),
        ]
        if uniform_refinement is not None:
            if not isinstance(uniform_refinement, BisectionUniformRefinement):
                raise TypeError(
                    "uniform_refinement must be the canonical uniform source lineage or None."
                )
            authority_parts.extend(
                (
                    np.asarray(uniform_refinement.parent_ids),
                    np.asarray(uniform_refinement.child_ids).reshape(-1),
                )
            )
        required_authority = np.unique(np.concatenate(authority_parts))
        authority_rows = np.searchsorted(scientific_ids, required_authority)
        canonical_scientific_ids = scientific_ids[authority_rows]
        canonical_scientific_blocks = scientific_blocks[authority_rows]
        parent_authority = scientific_blocks[
            np.searchsorted(scientific_ids, arrays["record_parent_ids"])
        ]
        child_authority = scientific_blocks[
            np.searchsorted(scientific_ids, arrays["record_child_ids"])
        ]
        if np.any(child_authority != parent_authority[:, None]):
            raise ValueError(
                "Bisection children must retain their parent's scientific block authority."
            )
        if uniform_refinement is not None:
            if uniform_refinement.dimension != dimension:
                raise ValueError(
                    "Uniform source lineage must have the hierarchy's simplex dimension."
                )
            roots = np.concatenate(
                (arrays["cell_global_ids"], arrays["record_parent_ids"])
            )
            restored = np.isin(np.asarray(uniform_refinement.parent_ids), roots)
            children = np.asarray(uniform_refinement.child_ids)
            if not np.all(np.isin(children[~restored], roots)) or np.any(
                np.isin(children[restored], roots)
            ):
                raise ValueError(
                    "Uniform source roots must retain exactly their actual child or restored-parent ancestry."
                )
            node_vertices = np.concatenate(
                (arrays["ordered_vertices"], arrays["record_parent_rows"])
            )
            node_order = np.argsort(roots, kind="stable")
            node_ids = roots[node_order]
            lineage = uniform_refinement.host_arrays()
            for row, is_restored in enumerate(restored):
                identifiers = (
                    lineage["parent_ids"][row : row + 1]
                    if is_restored
                    else lineage["child_ids"][row]
                )
                expected = (
                    lineage["parent_rows"][row : row + 1]
                    if is_restored
                    else lineage["child_vertices"][row]
                )
                positions = node_order[np.searchsorted(node_ids, identifiers)]
                if not np.array_equal(
                    np.sort(node_vertices[positions], axis=1), np.sort(expected, axis=1)
                ):
                    raise ValueError(
                        "Packed uniform source ancestry changed its actual scientific vertex incidence."
                    )
        retired_keys, retired_ids = _retired_tables(
            retired_entity_keys, retired_entity_ids, dimension
        )
        vertex_counter, cell_counter = int(next_vertex_id), int(next_cell_id)
        vertex_high = max(
            int(np.max(arrays["ordered_vertices"])),
            int(np.max(arrays["record_parent_vertices"], initial=-1)),
            int(np.max(arrays["record_vertex_ids"], initial=-1)),
        )
        cell_high = max(
            int(arrays["cell_global_ids"][-1]),
            int(np.max(arrays["record_parent_ids"], initial=-1)),
            int(np.max(arrays["record_child_ids"], initial=-1)),
        )
        if uniform_refinement is not None:
            vertex_high = max(
                vertex_high, int(np.max(np.asarray(uniform_refinement.child_vertices)))
            )
            cell_high = max(
                cell_high, int(np.max(np.asarray(uniform_refinement.child_ids)))
            )
        if vertex_counter <= vertex_high or cell_counter <= cell_high:
            raise ValueError("Hierarchy counters must exceed every issued ID.")
        self.dimension = dimension
        self.cell_global_ids = jnp.asarray(arrays["cell_global_ids"])
        self.ordered_vertices = jnp.asarray(arrays["ordered_vertices"])
        self.tags = jnp.asarray(arrays["tags"])
        self.generations = jnp.asarray(arrays["generations"])
        self.scientific_cell_ids = jnp.asarray(scientific_ids)
        self.scientific_block_ids = jnp.asarray(scientific_blocks)
        self.record_parent_ids = jnp.asarray(arrays["record_parent_ids"])
        self.record_parent_blocks = jnp.asarray(arrays["record_parent_blocks"])
        self.record_parent_rows = jnp.asarray(arrays["record_parent_rows"])
        self.record_parent_vertices = jnp.asarray(arrays["record_parent_vertices"])
        self.record_parent_tags = jnp.asarray(arrays["record_parent_tags"])
        self.record_child_ids = jnp.asarray(arrays["record_child_ids"])
        self.record_vertex_ids = jnp.asarray(arrays["record_vertex_ids"])
        self.retired_entity_keys = tuple(jnp.asarray(value) for value in retired_keys)
        self.retired_entity_ids = tuple(jnp.asarray(value) for value in retired_ids)
        self.uniform_refinement = uniform_refinement
        self.next_vertex_id = vertex_counter
        self.next_cell_id = cell_counter
        self.hierarchy_id = canonical_fingerprint(
            {
                "kind": "bisection-hierarchy",
                "dimension": dimension,
                "arrays": array_tree_fingerprint(arrays),
                "scientific_blocks": array_tree_fingerprint(
                    (canonical_scientific_ids, canonical_scientific_blocks)
                ),
                "retired_entity_keys": array_tree_fingerprint(retired_keys),
                "retired_entity_ids": array_tree_fingerprint(retired_ids),
                "uniform_refinement": None
                if uniform_refinement is None
                else uniform_refinement.lineage_id,
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
    labels; ``incompatible_facets`` counts the source facets violating the
    matching condition. Incompatible labels with ``uniform_refinement_applied``
    false were closed under ``CONFORMING_CLOSURE``: ``bisections`` and
    ``closure_iterations`` then include every closure bisection beyond the
    accepted marks. ``rejected_coarsening_ids`` are marked source cells that stay
    active (their family is incomplete, unmarked, protected, class-mixed, or
    touches the refinement closure).
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


# Vertices are addressed by compact indices during one adaptation: source
# vertices by the rank of their global ID, new vertices after them in issue
# order. New IDs exceed every existing ID, so compact order is global-ID order
# and every canonical ordering below equals the global-ID ordering.
_SHIFT = np.int64(1 << 31)
_VERTEX_LIMIT = 1 << 31


# The Maubach templates are owned by the device layout so that host and device
# bisection share one table set.
_TABLES = {dimension: maubach_bisection_tables(dimension) for dimension in (2, 3)}


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
    limit: int


class _UniformAllowance(NamedTuple):
    work: int
    geometry_queries: int
    cells: int
    vertices: int
    scratch_bytes: int
    wall_seconds: float
    cavity_cells: int


@contextmanager
def _uniform_execution(
    allowance: _UniformAllowance, /
) -> Iterator[NativeExecutionBudget]:
    active = current_native_execution_budget()
    if active is not None:
        if current_native_host_workspace() is None:
            with active.host_workspace():
                yield active
        else:
            yield active
        return
    with NativeExecutionBudget(
        max_work=allowance.work,
        max_geometry_queries=allowance.geometry_queries,
        max_cavity_cells=allowance.cavity_cells,
        max_scratch_bytes=allowance.scratch_bytes,
        max_wall_seconds=allowance.wall_seconds,
    ) as budget:
        with budget.host_workspace():
            yield budget


def _uniform_charge(work: int, storage_bytes: int = 0, /) -> None:
    budget = current_native_execution_budget()
    if budget is None:
        raise RuntimeError(
            "Uniform source preparation and inverse require their original native allowance."
        )
    budget.charge(work=work)
    workspace = current_native_host_workspace()
    if workspace is None:
        raise RuntimeError(
            "Uniform source storage requires its owning native host workspace."
        )
    if storage_bytes:
        workspace.set_bound(workspace.bound + storage_bytes)


def _uniform_retain_source(source: CellMeshingResult, /) -> None:
    workspace = current_native_host_workspace()
    if workspace is None:
        raise RuntimeError(
            "Packed scientific source retention requires its owning native host workspace."
        )
    workspace.retain_owner(source)


class _Capacity(NamedTuple):
    """Declared cell, vertex, and construction-step budgets of one closure."""

    cells: int
    vertices: int
    work_units: int


class _Start(NamedTuple):
    front: _Front
    records: _Records
    marks: np.ndarray
    next_vertex: int
    compatible: bool | None
    incompatible: int
    uniform: bool
    retired: tuple[tuple[np.ndarray, np.ndarray], ...]
    uniform_refinement: BisectionUniformRefinement | None


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
    support_weights: np.ndarray
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

    pairs = _TABLES[dimension][3]
    position = np.argmax(tuples[:, :, None] == rows[:, None, :], axis=2)
    inversions = np.sum(position[:, pairs[:, 0]] > position[:, pairs[:, 1]], axis=1)
    return inversions % 2 == 1


def _reversed(tuples: np.ndarray, tags: np.ndarray, dimension: int, /) -> Any:
    return np.take_along_axis(tuples, _TABLES[dimension][2][tags], axis=1)


def _canonical(tuples: Any, tags: Any, odd: Any, dimension: int, /) -> Any:
    """Canonical representative of each tagged simplex and its orientation parity."""

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


def _children(tuples: Any, tags: Any, midpoints: Any, dimension: int, /) -> Any:
    first_table, second_table = _TABLES[dimension][:2]
    extended = np.concatenate((tuples, midpoints[:, None]), axis=1)
    return (
        np.take_along_axis(extended, first_table[tags], axis=1),
        np.take_along_axis(extended, second_table[tags], axis=1),
    )


def _prepared_source(mesh: CellMesh, /) -> _Source:
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    dimension = mesh.topological_dimension
    kind = {2: "triangle", 3: "tetrahedron"}.get(dimension)
    if kind is None or any(block.cell_kind != kind for block in mesh.blocks):
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
    maximum_closure_iterations: int,
    /,
    *,
    source_cell_classes: np.ndarray | None = None,
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
    if isinstance(maximum_closure_iterations, bool) or not isinstance(
        maximum_closure_iterations, (int, np.integer)
    ):
        raise TypeError("maximum_closure_iterations must be an integer.")
    if maximum_closure_iterations < 1:
        raise ValueError("maximum_closure_iterations must be positive.")
    mesh, dimension = source.mesh, source.dimension
    if source_cell_classes is None:
        cell_order = np.searchsorted(source.cells.ids, entity_keys(mesh, dimension)[:, 0])
        classes = np.empty(source.cells.ids.shape, dtype=np.int64)
        classes[cell_order] = _aligned_classes(
            cell_classes, cell_order.size, "cell_classes"
        )
    else:
        # An authenticated root projection retains the whole original mesh and
        # coefficient bank; only its actual selected SCI cell axis is classified.
        classes = _aligned_classes(
            source_cell_classes, source.cells.ids.size, "source_cell_classes"
        )
    facet_keys = np.searchsorted(source.vertex_ids, entity_keys(mesh, dimension - 1))
    return _Request(
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
    cells: _Cells, coordinates: np.ndarray, dimension: int, /
) -> Any:
    """Maubach labels from the global strict edge order (longest first, then key)."""

    pairs = _TABLES[dimension][3]
    low = np.minimum(cells.rows[:, pairs[:, 0]], cells.rows[:, pairs[:, 1]])
    high = np.maximum(cells.rows[:, pairs[:, 0]], cells.rows[:, pairs[:, 1]])
    codes = low * _SHIFT + high
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
    at_end = opposite == tuples[np.arange(owners.size), tags]
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
    child_tags = np.where(tags > 1, tags - 1, dimension)
    matched |= (
        same_tag
        & left_open
        & right_open
        & _reflected(left_child, right_child, child_tags, dimension)
    )
    return int(np.sum(~matched))


def _subdivided(front: _Front, dimension: int, maximum_vertices: int, /) -> _Front:
    """Barycentric subdivision: each simplex into (d+1)! simplices (v, m, [f,] c)_d."""

    cells, width = front.cells, dimension + 1
    orders, odd_orders = _TABLES[dimension][4:]
    count, arrangements = cells.ids.size, orders.shape[0]
    tables = [_simplex_keys(cells.rows, size) for size in range(2, width)]
    tables.append(cells.rows)
    offsets = np.cumsum([0] + [table.shape[0] for table in tables])
    if front.vertex_base + offsets[-1] >= _VERTEX_LIMIT:
        raise ValueError("Barycentric subdivision would exceed 2**31 vertices.")
    if front.vertex_base + offsets[-1] > maximum_vertices:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Uniform source preparation exceeds its original vertex allowance.",
            stage="uniform-source-preparation",
        )
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
    children = _Cells(
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
    growth = _Growth(parents, weights, np.zeros((parents.shape[0],), dtype=np.int64))
    return front._replace(
        cells=children, growth=growth, next_cell=front.next_cell + tuples.shape[0]
    )


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
    budget = current_native_execution_budget()
    if budget is not None:
        budget.admit_cavity(count)
        budget.charge(work=count)
    codes = _edge_codes(chosen.tuples[:, 0], chosen.tuples[np.arange(count), chosen.tags])
    front, midpoints, created = _split_edges(front, codes, level, dimension)
    first, second = _children(chosen.tuples, chosen.tags, midpoints, dimension)
    odd = _odd_relative(chosen.rows, chosen.tuples, dimension)
    tuples = np.stack((first, second), axis=1).reshape((2 * count, dimension + 1))
    odd = np.stack((odd, odd ^ (chosen.tags % 2 == 1)), axis=1).reshape((-1,))
    tags = np.repeat(np.where(chosen.tags > 1, chosen.tags - 1, dimension), 2)
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
    pairs = _TABLES[dimension][3]
    codes = _edge_codes(cells.tuples[:, pairs[:, 0]], cells.tuples[:, pairs[:, 1]])
    return np.any(_members(split_keys, codes), axis=1)


def _admit_closure_batch(
    front: _Front, records: _Records, selected: np.ndarray, capacity: _Capacity, /
) -> None:
    """Refuse one closure batch before it allocates beyond the declared budgets.

    A bisection is one construction step: it retires one cell, creates two, and
    issues the midpoint of its refinement edge unless this adaptation already
    split that edge.
    """

    chosen = _select(front.cells, selected)
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
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"The next bisection closure batch would reach {cells} active cells, "
            f"{vertices} vertices and {steps} bisection steps, beyond the declared "
            f"{capacity.cells} cells, {capacity.vertices} vertices and "
            f"{capacity.work_units} work units.",
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
    front: _Front,
    selected: np.ndarray,
    protected_codes: np.ndarray,
    limit: int,
    capacity: _Capacity,
    dimension: int,
    /,
) -> tuple[_Front, _Records, int] | None:
    """Bisect the selected cells, then every cell with a split edge, until conforming.

    Returns ``None`` as soon as the closure would split a protected edge. Every
    batch is admitted against the declared capacity before it is bisected.
    """

    records = _empty_records(dimension)
    iterations = 0
    while selected.size:
        if iterations == limit:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"Bisection closure did not conform within {limit} iterations "
                f"({front.cells.ids.size} active cells, {records.parent_ids.size} "
                f"bisections, {selected.size} cells still non-conforming).",
                stage="bisection-closure",
                entity_ids=tuple(int(value) for value in front.cells.ids[selected]),
            )
        _admit_closure_batch(front, records, selected, capacity)
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
    front: _Front,
    marks: np.ndarray,
    request: _Request,
    capacity: _Capacity,
    dimension: int,
    /,
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
        protected, _edge_codes(tuples[:, 0], tuples[np.arange(marks.size), tags])
    )
    accepted, rejected = [empty], [marks[direct]]
    pending = [marks[~direct]] if np.any(~direct) else []
    tests = 0
    while pending:
        group = pending.pop()
        tests += 1
        closure = _closure(front, group, protected, request.limit, capacity, dimension)
        if closure is not None:
            accepted.append(group)
        elif group.size == 1:
            rejected.append(group)
        else:
            half = group.size // 2
            pending.extend((group[half:], group[:half]))
    return np.sort(np.concatenate(accepted)), np.sort(np.concatenate(rejected)), tests


def _refinement(
    start: _Start, request: _Request, capacity: _Capacity, dimension: int, /
) -> _Refinement:
    accepted, rejected, tests = _admissible_marks(
        start.front, start.marks, request, capacity, dimension
    )
    closure = _closure(
        start.front,
        accepted,
        request.protected_codes,
        request.limit,
        capacity,
        dimension,
    )
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


def _uniform_lineage(
    source: _Source,
    request: _Request,
    parents: _Cells,
    front: _Front,
    next_vertex: int,
    scientific_source: CellMeshingResult,
    /,
) -> BisectionUniformRefinement:
    _uniform_retain_source(scientific_source)
    dimension, width = source.dimension, source.dimension + 1
    count = source.vertex_ids.size
    arrangements = _TABLES[dimension][4].shape[0]
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
    references = nested_reference_vertices(
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
            "Uniform source lineage contains an undeclared original parent identity."
        )
    original_positions = original_order[positions]
    if not np.array_equal(original_ids[original_positions], parents.ids):
        raise ValueError(
            "Uniform source lineage contains an undeclared original parent identity."
        )
    row_order = original_rows[original_positions]
    facet_columns = np.asarray(
        tuple(combinations(range(width), dimension)), dtype=np.int64
    )
    facet_keys = np.sort(parents.rows[:, facet_columns], axis=2).reshape((-1, dimension))
    facets = key_rows(request.facet_keys, facet_keys)
    if np.any(facets < 0):
        raise ValueError("Uniform source lineage lost an original source facet.")
    return BisectionUniformRefinement(
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
        _global_vertices(front.cells.rows, source, next_vertex).reshape(
            (-1, arrangements, width)
        ),
        references,
        barycentric,
        source=scientific_source,
    )


def _reprepare_admitted_uniform(
    source: _Source,
    request: _Request,
    admitted: BisectionUniformRefinement,
    allowance: _UniformAllowance,
    retained_issuers: tuple[int, int],
    retained_retired: tuple[tuple[np.ndarray, np.ndarray], ...],
    /,
) -> _Start:
    """Reissue an authenticated existing uniform law, not a policy fallback.

    The collective owner authenticates prior admission and the restored cohort.
    Current parent metadata must still be its original scientific metadata.
    Fresh identifiers retain the original action rows and arrangement order;
    discarded historical children and construction identifiers are never reused.
    """
    if (
        source.dimension != admitted.dimension
        or source.mesh.mesh_id != admitted.source.mesh.mesh_id
    ):
        raise ValueError(
            "Admitted uniform preparation requires its actual original source mesh."
        )
    if request.coarsen_ids.size:
        raise ValueError(
            "Admitted uniform source preparation cannot replace a coarsening request."
        )
    width = source.dimension + 1
    count = source.cells.ids.size
    arrangements = _TABLES[source.dimension][4].shape[0]
    if count == 0 or count * arrangements > allowance.cells:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Admitted uniform source preparation exceeds its original cell allowance.",
            stage="uniform-source-preparation",
        )
    if len(retained_issuers) != 2 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in retained_issuers
    ):
        raise TypeError(
            "Admitted uniform preparation requires exact retained integer issuers."
        )
    next_vertex, next_cell = (int(value) for value in retained_issuers)
    with _uniform_execution(allowance) as budget:
        budget.admit_cavity(count)
        slots = count * arrangements * width
        _uniform_charge(
            slots + admitted.child_vertices.size + source.coordinates.size,
            4 * slots * (64 + 16 * width)
            + source.vertex_ids.nbytes
            + source.coordinates.nbytes,
        )
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
                "Admitted uniform preparation changed the actual original scientific coordinate bank."
            )
        original = admitted.host_arrays()
        positions = np.searchsorted(original["parent_ids"], source.cells.ids)
        if np.any(positions >= original["parent_ids"].size):
            raise ValueError(
                "Admitted uniform cohort contains an undeclared original parent."
            )
        if not np.array_equal(original["parent_ids"][positions], source.cells.ids):
            raise ValueError(
                "Admitted uniform cohort contains an undeclared original parent."
            )
        if next_vertex <= max(
            int(source.vertex_ids[-1]), int(np.max(original["child_vertices"]))
        ) or next_cell <= max(
            int(source.cells.ids[-1]), int(np.max(original["child_ids"]))
        ):
            raise ValueError(
                "Admitted uniform reissuance cannot revive an existing or historical scientific identity."
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
                    f"Admitted uniform preparation changed the original scientific {name}."
                )
        keys, identifiers = _retired_tables(
            tuple(value[0] for value in retained_retired),
            tuple(value[1] for value in retained_retired),
            source.dimension,
        )
        if any(np.any((table < 0) | (table >= source.vertex_ids.size)) for table in keys):
            raise ValueError(
                "Admitted uniform retirement must bind the actual original source vertex axis."
            )
        pairs = _TABLES[source.dimension][3]
        edges = _edge_codes(
            source.cells.rows[:, pairs[:, 0]], source.cells.rows[:, pairs[:, 1]]
        )
        if np.any(_members(request.protected_codes, edges)):
            raise ValueError(
                "Admitted uniform reissuance would split an actual protected edge."
            )
        front = _subdivided(
            _initial_front(
                source.cells, source.vertex_ids.size, next_cell, source.dimension
            ),
            source.dimension,
            allowance.vertices,
        )
        generated = _uniform_lineage(
            source, request, source.cells, front, next_vertex, admitted.source
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
                        "Admitted uniform reissuance changed an actual original raw coefficient action."
                    )
                order.append(parent * arrangements + child)
        order = np.asarray(order, dtype=np.int64)
        actual = incoming
        regenerated = generated
        if not np.array_equal(order, np.arange(count * arrangements, dtype=np.int64)):
            front = front._replace(
                cells=_select(front.cells, order)._replace(
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
                equal = np.array_equal(
                    actual[name].view(np.uint64), expected.view(np.uint64)
                )
            else:
                equal = np.array_equal(actual[name], expected)
            if not equal:
                raise ValueError(
                    f"Admitted uniform reissuance changed the original scientific {name}."
                )
        if actual is not incoming:
            regenerated = BisectionUniformRefinement(
                source.dimension,
                *actual.values(),
                source=admitted.source,
            )
        incompatible = _incompatible_facets(source.cells, source.dimension)
        if _incompatible_facets(front.cells, source.dimension):
            raise ValueError(
                "The retained admitted uniform arrangement does not match its actual scientific neighbors."
            )
        return _Start(
            front,
            _empty_records(source.dimension),
            np.flatnonzero(np.isin(front.cells.origins, request.refine_ids)),
            next_vertex,
            incompatible == 0,
            incompatible,
            True,
            tuple(zip(keys, identifiers, strict=True)),
            regenerated,
        )


def _labelled_start(
    source: _Source,
    request: _Request,
    compatibility: BisectionCompatibility,
    /,
    *,
    uniform_allowance: _UniformAllowance,
    source_hierarchy: BisectionHierarchy | None,
    scientific_source: CellMeshingResult,
    retained_issuers: tuple[int, int] | None = None,
    retained_retired: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
) -> _Start:
    dimension = source.dimension
    if request.coarsen_ids.size:
        raise ValueError(
            "Coarsening requires the BisectionHierarchy of a previous bisection."
        )
    tuples, tags = _longest_edge_labels(source.cells, source.coordinates, dimension)
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
                "Retained relabel issuers require one independently bound predecessor, not a second hierarchy."
            )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in retained_issuers
        ):
            raise TypeError(
                "Retained relabel issuers must be exact integer high-water marks."
            )
        next_vertex, next_cell = (int(value) for value in retained_issuers)
        if next_vertex <= source.vertex_ids[-1] or next_cell <= cells.ids[-1]:
            raise ValueError(
                "Retained relabel issuers must exceed every current scientific source identity."
            )
    front = _initial_front(cells, source.vertex_ids.size, next_cell, dimension)
    incompatible = _incompatible_facets(cells, dimension)
    uniform_refinement = None
    uniform = False
    if incompatible:
        match compatibility:
            case BisectionCompatibility.REJECT:
                raise ValueError(
                    f"{incompatible} interior facets violate the bisection matching "
                    "condition of the longest-edge labelling; use "
                    "BisectionCompatibility.UNIFORM_REFINEMENT to subdivide "
                    "barycentrically into a compatible mesh first, or, for triangle "
                    "meshes, BisectionCompatibility.CONFORMING_CLOSURE to close the "
                    "original labels locally."
                )
            case BisectionCompatibility.UNIFORM_REFINEMENT:
                pairs = _TABLES[dimension][3]
                edges = _edge_codes(
                    cells.rows[:, pairs[:, 0]], cells.rows[:, pairs[:, 1]]
                )
                if np.any(_members(request.protected_codes, edges)):
                    raise ValueError(
                        "Uniform barycentric refinement would split protected edges."
                    )
                arrangements = _TABLES[dimension][4].shape[0]
                if cells.ids.size * arrangements > uniform_allowance.cells:
                    raise MeshingFailure(
                        MeshingFailureCategory.RESOURCE_EXHAUSTED,
                        "Uniform source preparation exceeds its original cell allowance.",
                        stage="uniform-source-preparation",
                    )
                with _uniform_execution(uniform_allowance) as budget:
                    budget.admit_cavity(cells.ids.size)
                    slots = cells.ids.size * arrangements * (dimension + 1)
                    _uniform_charge(
                        slots,
                        2 * slots * (64 + 16 * (dimension + 1))
                        + source.vertex_ids.nbytes,
                    )
                    front = _subdivided(front, dimension, uniform_allowance.vertices)
                    uniform_refinement = _uniform_lineage(
                        source,
                        request,
                        cells,
                        front,
                        next_vertex,
                        scientific_source,
                    )
                uniform = True
            case BisectionCompatibility.CONFORMING_CLOSURE:
                if dimension != 2:
                    raise ValueError(
                        f"{incompatible} interior facets violate the bisection "
                        "matching condition; CONFORMING_CLOSURE closes incompatible "
                        "labels of triangle meshes only, and tetrahedral bisection "
                        "requires BisectionCompatibility.UNIFORM_REFINEMENT."
                    )
            case _:
                raise ValueError(
                    f"Unsupported bisection compatibility {compatibility!r}."
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
                "Retained relabel retirement must have exactly one predecessor owner."
            )
        keys, identifiers = _retired_tables(
            tuple(value[0] for value in retained_retired),
            tuple(value[1] for value in retained_retired),
            dimension,
        )
        if any(np.any((table < 0) | (table >= source.vertex_ids.size)) for table in keys):
            raise ValueError(
                "Retained relabel retirement keys must use the actual source vertex axis."
            )
        retired = tuple(zip(keys, identifiers, strict=True))
    return _Start(
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


def _prepared_start(
    source: _Source,
    request: _Request,
    compatibility: BisectionCompatibility,
    hierarchy: BisectionHierarchy | None,
    allowance: _UniformAllowance,
    scientific_source: CellMeshingResult,
    /,
    *,
    future_refinement: bool = False,
    retained_issuers: tuple[int, int] | None = None,
    retained_retired: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
) -> _Start:
    # Device preparation admits marks later. A completely inverted hierarchy
    # retains the original (possibly incompatible) root tags, not the uniform
    # children's matching theorem; relabel those roots before future splits.
    # These private inputs are authenticated by the accepted collective source
    # replay before entry; they retain its genuine issuer and retirement law
    # without constructing a hierarchy for a different presentation mesh.
    if (retained_issuers is None) != (retained_retired is None):
        raise ValueError(
            "Retained relabel issuers and retirement require the same complete predecessor."
        )
    if hierarchy is not None and retained_issuers is not None:
        raise ValueError(
            "Retained relabel preparation cannot have two predecessor authorities."
        )
    if hierarchy is not None:
        bound = _bound_start(source, request, hierarchy)
        if bound.uniform_refinement is not None:
            _require_uniform_source_geometry(scientific_source, bound.uniform_refinement)
        if (
            bound.records.parent_ids.size
            or bound.uniform_refinement is not None
            or request.coarsen_ids.size
            or (not request.refine_ids.size and not future_refinement)
        ):
            return bound
    return _labelled_start(
        source,
        request,
        compatibility,
        uniform_allowance=allowance,
        source_hierarchy=hierarchy,
        scientific_source=scientific_source,
        retained_issuers=retained_issuers,
        retained_retired=retained_retired,
    )


def _require_uniform_source_geometry(
    current: CellMeshingResult,
    lineage: BisectionUniformRefinement,
    /,
) -> None:
    """Bind action groups to original SCI roots, columns, and coefficient banks."""
    from ..discretization._cell_geometry import (
        _require_p1_cardinal_source,
        BarycentricCellGeometryElement,
        PolynomialComposedCellGeometryElement,
    )
    from ..discretization._cell_geometry_validity import cell_geometry_id

    original = lineage.source
    _uniform_retain_source(original)
    _uniform_charge(
        lineage.barycentric_weights.size, lineage.barycentric_weights.size * 32
    )
    arrays = lineage.host_arrays()
    _require_uniform_action_family(arrays["barycentric_weights"], lineage.dimension)
    for identifier, block, row, vertices in zip(
        arrays["parent_ids"],
        arrays["parent_blocks"],
        arrays["parent_row_order"],
        arrays["parent_rows"],
        strict=True,
    ):
        if block < 0 or block >= len(original.mesh.blocks):
            raise ValueError(
                "Uniform lineage references an undeclared original scientific block."
            )
        definition = original.mesh.blocks[int(block)]
        if (
            row < 0
            or row >= definition.cell_count
            or np.asarray(definition.global_ids)[row] != identifier
        ):
            raise ValueError(
                "Uniform lineage changed its original scientific block or row order."
            )
        if not np.array_equal(
            np.asarray(original.mesh.vertex_global_ids)[
                np.asarray(definition.vertices)[row]
            ],
            vertices,
        ):
            raise ValueError(
                "Uniform lineage changed its original scientific vertex columns."
            )
    origin = current.geometry.restriction_source
    if origin is None or (
        origin.source_geometry_id != cell_geometry_id(original.geometry)
        or origin.source_topology_id != original.mesh.topology_id
    ):
        raise ValueError(
            "Uniform source actions lost their original scientific geometry owner."
        )
    original_elements, original_routes, _ = original.geometry.resolve(original.mesh)
    elements, routes, _ = current.geometry.resolve(current.mesh)
    _uniform_charge(
        original.geometry.coordinates.size + current.geometry.coordinates.size,
        (original.geometry.coordinates.size + current.geometry.coordinates.size) * 512,
    )
    from ..discretization._coordinate_enclosure import prepared_coordinate_source_bank

    original_bank = prepared_coordinate_source_bank(original.geometry)
    current_bank = (
        original_bank
        if current.geometry.coordinates is original.geometry.coordinates
        and current.geometry.exact_source is original.geometry.exact_source
        else prepared_coordinate_source_bank(current.geometry)
    )
    roots = {
        int(identifier): (
            element,
            np.asarray(route)[row],
            np.asarray(original.mesh.vertex_global_ids)[np.asarray(block.vertices)[row]],
        )
        for block, element, route in zip(
            original.mesh.blocks, original_elements, original_routes, strict=True
        )
        for row, identifier in enumerate(block.global_ids)
    }
    coefficient_ids = (
        np.arange(current.geometry.coordinates.shape[0], dtype=np.int64)
        if current.mesh.storage is None
        else np.asarray(current.mesh.storage.coordinate_global_ids)
    )
    for block, element, route in zip(current.mesh.blocks, elements, routes, strict=True):
        root = element
        while isinstance(
            root,
            (
                BarycentricCellGeometryElement,
                PolynomialComposedCellGeometryElement,
            ),
        ):
            root = root.source_element
        _require_p1_cardinal_source(root)
        parents = np.asarray(origin.block_parent_cell_ids[block.name])
        corners = np.asarray(origin.block_parent_vertex_ids[block.name])
        _uniform_charge(block.vertices.size, block.vertices.size * 32)
        for parent, vertices, coefficients in zip(
            parents, corners, np.asarray(route), strict=True
        ):
            if int(parent) not in roots:
                raise ValueError(
                    "Uniform source action references an undeclared original scientific cell."
                )
            original_element, original_route, original_corners = roots[int(parent)]
            ancestor = element
            while (
                isinstance(
                    ancestor,
                    (
                        BarycentricCellGeometryElement,
                        PolynomialComposedCellGeometryElement,
                    ),
                )
                and ancestor.element_id != original_element.element_id
            ):
                ancestor = ancestor.source_element
            if ancestor.element_id != original_element.element_id or not np.array_equal(
                vertices, original_corners
            ):
                raise ValueError(
                    "Uniform source action changed its original scientific basis or corner columns."
                )
            if not np.array_equal(coefficient_ids[coefficients], original_route):
                raise ValueError(
                    "Uniform source action changed its original scientific coefficient SCI columns."
                )
            if tuple(current_bank[int(index)] for index in coefficients) != tuple(
                original_bank[int(index)] for index in original_route
            ):
                raise ValueError(
                    "Uniform source action changed its original scientific coefficient bank."
                )


def _uniform_binary_chart(
    identifier: int,
    hierarchy: BisectionHierarchy,
    /,
) -> tuple[int, int | None, tuple[np.ndarray, ...]]:
    """Reconstruct authored binary half actions, never compress source weights."""
    lineage = hierarchy.uniform_refinement
    if lineage is None:
        raise ValueError(
            "Exact packed chart reconstruction requires its original source owner."
        )
    _uniform_charge(
        hierarchy.record_parent_ids.size + lineage.child_ids.size,
        256 * (hierarchy.record_parent_ids.size + lineage.child_ids.size),
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
                    "A packed binary chart has contradictory scientific parents."
                )
    path = []
    current = identifier
    visited: set[int] = set()
    while current not in roots:
        if current in visited or current not in predecessors or current not in records:
            raise ValueError(
                "A restored packed cell has no complete original scientific chart path."
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
                            "A binary chart split edge lost its original parent columns."
                        )
                    action[corner, matches[0]] = 0.5
            else:
                matches = np.flatnonzero(parent_vertices == vertex)
                if matches.size != 1:
                    raise ValueError(
                        "A binary chart corner has foreign original scientific support."
                    )
                action[corner, matches[0]] = 1.0
        steps.append(action)
    return root, child, tuple(steps)


def _rebind_bisection_presentation_blocks(
    hierarchy: BisectionHierarchy,
    target: CellMeshingResult,
    /,
) -> BisectionHierarchy:
    """Lower binary records onto actual descendant presentation groups."""
    if (
        hierarchy.record_parent_ids.size == 0
        or target.geometry.restriction_source is None
    ):
        return hierarchy
    _uniform_charge(
        hierarchy.record_child_ids.size + hierarchy.cell_global_ids.size,
        256 * (hierarchy.record_child_ids.size + hierarchy.cell_global_ids.size),
    )
    origin = target.geometry.restriction_source
    owners = {}
    roots = {}
    for block_index, block in enumerate(target.mesh.blocks):
        ancestors = np.asarray(origin.block_parent_cell_ids[block.name])
        for identifier, ancestor in zip(block.global_ids, ancestors, strict=True):
            owners[int(identifier)] = block_index
            roots[int(identifier)] = int(ancestor)
    parents = np.asarray(hierarchy.record_parent_ids)
    children = np.asarray(hierarchy.record_child_ids)
    if np.any(children <= parents[:, None]):
        raise ValueError(
            "Binary presentation records require their actual monotone scientific birth IDs."
        )
    descendants = {
        int(parent): tuple(int(child) for child in row)
        for parent, row in zip(parents, children, strict=True)
    }
    representatives: dict[int, tuple[int, int]] = {}
    pending: list[tuple[int, bool]] = [(int(parent), False) for parent in parents[::-1]]
    while pending:
        identifier, expanded = pending.pop()
        if identifier in representatives:
            continue
        if identifier in owners:
            representatives[identifier] = (identifier, roots[identifier])
            continue
        if identifier not in descendants:
            raise ValueError(
                "A binary presentation record has no active scientific descendant."
            )
        row = descendants[identifier]
        if not expanded:
            pending.append((identifier, True))
            pending.extend((child, False) for child in reversed(row))
            continue
        values = [representatives[child] for child in row]
        if len({root for _, root in values}) != 1:
            raise ValueError(
                "Binary descendants disagree on their explicit original scientific root."
            )
        representatives[identifier] = min(values)
    blocks = np.asarray(
        [owners[representatives[int(parent)][0]] for parent in parents], dtype=np.int32
    )
    if np.array_equal(blocks, np.asarray(hierarchy.record_parent_blocks)):
        return hierarchy
    return BisectionHierarchy(
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
    )


def _bound_records(source: _Source, hierarchy: BisectionHierarchy, /) -> _Records:
    def vertices(values: Array) -> np.ndarray:
        return _compact_vertices(
            source, np.asarray(values, dtype=np.int64), "BisectionHierarchy"
        )

    blocks = np.asarray(hierarchy.record_parent_blocks, dtype=np.int64)
    if np.any(blocks >= len(source.mesh.blocks)):
        raise ValueError("BisectionHierarchy records reference undeclared blocks.")
    return _Records(
        np.asarray(hierarchy.record_parent_ids, dtype=np.int64),
        blocks,
        vertices(hierarchy.record_parent_rows),
        vertices(hierarchy.record_parent_vertices),
        np.asarray(hierarchy.record_parent_tags, dtype=np.int64),
        np.asarray(hierarchy.record_child_ids, dtype=np.int64),
        vertices(hierarchy.record_vertex_ids),
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
    return _Start(
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


def _reverse_supports(
    steps: Any, vertex_count: int, dimension: int, /
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
            raise ValueError("A coarsened vertex spans more than one target simplex.")
        merged, merged_weights = _merged_rows(
            np.concatenate((supports[first[index]], supports[second[index]]), axis=1),
            0.5 * np.concatenate((weights[first[index]], weights[second[index]]), axis=1),
            width,
        )
        if not np.array_equal(merged, spans[:, :width]):
            raise RuntimeError("Coarsening supports disagree with their weights.")
        supports[unique] = merged
        weights[unique] = merged_weights
    return supports, weights


def _uniform_vertex_positions(
    source: _Source, start: _Start, values: np.ndarray, /
) -> np.ndarray:
    positions = np.searchsorted(source.vertex_ids, values)
    original = (positions < source.vertex_ids.size) & (
        source.vertex_ids[np.minimum(positions, source.vertex_ids.size - 1)] == values
    )
    issued = values - start.next_vertex
    if np.any(~original & ((issued < 0) | (issued >= start.front.growth.levels.size))):
        raise ValueError(
            "Uniform source lineage references an unowned scientific vertex."
        )
    return np.where(original, positions, source.vertex_ids.size + issued)


def _uniform_facet_agreement(
    source: _Source,
    start: _Start,
    arrays: dict[str, np.ndarray],
    parent: int,
    facets: tuple[np.ndarray, np.ndarray],
    /,
) -> bool:
    dimension, width = source.dimension, source.dimension + 1
    columns = tuple(combinations(range(width), dimension))
    vertices = arrays["child_vertices"][parent]
    barycentric = arrays["barycentric_weights"][parent]
    for child, row in enumerate(vertices):
        for local in columns:
            _uniform_charge(width * len(local), 0)
            key = np.sort(_uniform_vertex_positions(source, start, row[list(local)]))[
                None
            ]
            slot = key_rows(facets[0], key)[0]
            if slot < 0:
                raise ValueError(
                    "Uniform inverse lost an actual source facet occurrence."
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
                    "Uniform inverse contains a degenerate original source facet incidence."
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
    source: _Source,
    start: _Start,
    arrays: dict[str, np.ndarray],
    selected: np.ndarray,
    coarsening: _Coarsening,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, width = source.vertex_ids.size, source.dimension + 1
    _uniform_charge(count * width, 2 * count * width * 16)
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
                _uniform_charge(width, 0)
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
                            "Incident uniform roots disagree on their exact source-vertex reference action."
                        )
                replacements[position], replacement_weights[position] = (
                    packed_ids,
                    packed_weights,
                )
                assigned.add(int(position))
                if int(position) not in original:
                    discarded.append(int(position))
    support = coarsening.supports
    _uniform_charge(count * width * width, 4 * count * width * width * 16)
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


def _uniform_inverse(
    source: _Source,
    start: _Start,
    request: _Request,
    cells: _Cells,
    flags: _Flags,
    facets: tuple[np.ndarray, np.ndarray],
    coarsening: _Coarsening,
    /,
) -> _Coarsening:
    from ._mixed_adaptation import _Cell, _coarsen_patch, _Sibling

    lineage = start.uniform_refinement
    if lineage is None or request.coarsen_ids.size == 0:
        return coarsening
    _uniform_retain_source(lineage.source)
    arrays = lineage.host_arrays()
    kind = "triangle" if source.dimension == 2 else "tetrahedron"
    _uniform_charge(cells.ids.size, 512 * cells.ids.size)
    budget = current_native_execution_budget()
    if budget is None:
        raise RuntimeError("Uniform inverse lost its original native admission owner.")
    budget.admit_cavity(int(np.count_nonzero(flags.marked)))
    child_blocks = {
        int(child): int(block)
        for children, block in zip(
            arrays["child_ids"], arrays["parent_blocks"], strict=True
        )
        for child in children
    }
    active = {int(identifier): index for index, identifier in enumerate(cells.ids)}
    records = []
    for parent, identifier in enumerate(arrays["parent_ids"]):
        children = arrays["child_ids"][parent]
        _uniform_charge(children.size, 128 * children.size)
        if not all(int(child) in active for child in children):
            continue
        slots = np.asarray([active[int(child)] for child in children], dtype=np.int64)
        if not np.all(flags.marked[slots]) or np.any(flags.blocked[slots]):
            continue
        if not _uniform_facet_agreement(source, start, arrays, parent, facets):
            continue
        references = []
        for child, vertices, reference in zip(
            children,
            arrays["child_vertices"][parent],
            arrays["reference_vertices"][parent],
            strict=True,
        ):
            _uniform_charge(source.dimension + 1, 256 * (source.dimension + 1))
            original_block = lineage.source.mesh.blocks[
                int(arrays["parent_blocks"][parent])
            ]
            if (
                source.mesh.blocks[int(cells.blocks[active[int(child)]])].cell_kind
                != original_block.cell_kind
            ):
                raise ValueError(
                    "Uniform inverse changed the recorded scientific child cell kind."
                )
            current = _global_vertices(
                cells.rows[active[int(child)]], source, start.next_vertex
            )
            if not np.array_equal(np.sort(current), np.sort(vertices)):
                raise ValueError(
                    "Uniform inverse changed the recorded scientific child incidence."
                )
            positions = {int(vertex): index for index, vertex in enumerate(vertices)}
            references.append(
                tuple(
                    tuple(float(value) for value in reference[positions[int(vertex)]])
                    for vertex in current
                )
            )
        records.append(
            _Sibling(
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
    _uniform_charge(cells.rows.size, 128 * cells.rows.size)
    current = [
        _Cell(
            int(identifier),
            lineage.source.mesh.blocks[child_blocks[int(identifier)]].name
            if int(identifier) in child_blocks
            else source.mesh.blocks[int(cells.blocks[index])].name,
            kind,
            _global_vertices(cells.rows[index], source, start.next_vertex),
            int(flags.classes[index]),
        )
        for index, identifier in enumerate(cells.ids)
    ]
    protected = set(
        _global_vertices(
            np.flatnonzero(request.protected_vertices), source, start.next_vertex
        ).tolist()
    )
    protected.update(
        _global_vertices(
            request.protected_codes // _SHIFT, source, start.next_vertex
        ).tolist()
    )
    protected.update(
        _global_vertices(
            request.protected_codes % _SHIFT, source, start.next_vertex
        ).tolist()
    )
    # Uniform simplex ancestry has no layer-column cohorts and does not author
    # the mixed hierarchy's unchanged-cell scientific signatures.
    patch = _coarsen_patch(
        current,
        tuple(records),
        set(cells.ids[flags.marked].tolist()),
        protected,
        set(cells.ids[flags.blocked].tolist()),
        source.vertex_ids,
        column_members={},
        unchanged=set(),
    )
    if not patch.restored:
        return coarsening
    restored_ids = np.asarray(
        [cell.identifier for cell in patch.restored], dtype=np.int64
    )
    selected = np.searchsorted(arrays["parent_ids"], restored_ids)
    restored = _Cells(
        restored_ids,
        _uniform_vertex_positions(source, start, arrays["parent_rows"][selected]),
        _uniform_vertex_positions(source, start, arrays["parent_vertices"][selected]),
        arrays["parent_tags"][selected],
        arrays["parent_blocks"][selected],
        restored_ids,
        arrays["parent_levels"][selected],
    )
    retained = _select(
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
        source, start, arrays, selected, coarsening
    )
    return _Coarsening(
        _joined(retained, restored),
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


def _coarsening(
    source: _Source,
    start: _Start,
    request: _Request,
    blocked: np.ndarray,
    dimension: int,
    /,
) -> _Coarsening:
    cells, records = start.front.cells, start.records
    flags = _Flags(np.isin(cells.ids, request.coarsen_ids), blocked, request.cell_classes)
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
    coarsening = _Coarsening(
        _select(cells, ~_members(source_ids, cells.ids)),
        removed_ids,
        targets,
        np.concatenate([np.zeros((0,), dtype=np.int64), *undone]),
        removed_vertices,
        *_reverse_supports(steps, start.front.vertex_base, dimension),
        len(steps),
        request.coarsen_ids[_members(cells.ids, request.coarsen_ids)],
    )
    return _uniform_inverse(source, start, request, cells, flags, facets, coarsening)


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

    split_sources, split_weights = stencil
    owner, column = np.nonzero((split_sources >= 0) & (split_weights > 0.0))
    records = [
        _grouped_relations(
            0,
            (
                (
                    source.vertex_ids[split_sources[owner, column]][:, None],
                    ids(count + owner)[:, None],
                    EntityLineageKind.SPLIT_FROM,
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
    return (
        _select(cells, np.argsort(cells.ids, kind="stable")),
        _select(records, np.argsort(records.parent_ids, kind="stable")),
    )


def _target_hierarchy(
    source: _Source,
    start: _Start,
    cells: _Cells,
    records: _Records,
    retired: Any,
    front: _Front,
    /,
    *,
    prior: BisectionHierarchy | None = None,
) -> BisectionHierarchy:
    def ids(values: np.ndarray) -> np.ndarray:
        return _global_vertices(values, source, start.next_vertex)

    uniform = start.uniform_refinement
    if uniform is not None and np.all(
        np.isin(
            np.asarray(uniform.parent_ids),
            cells.ids,
        )
    ):
        uniform = None
    authority = (
        {
            int(identifier): int(block)
            for identifier, block in zip(
                prior.scientific_cell_ids, prior.scientific_block_ids, strict=True
            )
        }
        if prior is not None
        else {
            int(identifier): int(block)
            for identifier, block in zip(
                source.cells.ids, source.cells.blocks, strict=True
            )
        }
    )
    if start.uniform_refinement is not None:
        lineage = start.uniform_refinement
        for parent, children in zip(lineage.parent_ids, lineage.child_ids, strict=True):
            parent_id = int(parent)
            if parent_id not in authority:
                raise ValueError(
                    "Uniform refinement lost its original scientific block authority."
                )
            for child in children:
                authority[int(child)] = authority[parent_id]
    for parent, children in zip(records.parent_ids, records.child_ids, strict=True):
        parent_id = int(parent)
        if parent_id not in authority:
            raise ValueError("Bisection lost its parent scientific block authority.")
        for child in children:
            authority[int(child)] = authority[parent_id]
    scientific_ids = np.asarray(sorted(authority), dtype=np.int64)
    scientific_blocks = np.asarray(
        [authority[int(identifier)] for identifier in scientific_ids], dtype=np.int32
    )

    return BisectionHierarchy(
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
    if start.uniform_refinement is not None:
        _uniform_charge(
            cells.rows.size + source.cells.rows.size + growth.parents.size,
            cells.rows.nbytes * 8
            + source.cells.rows.nbytes * 8
            + growth.parents.nbytes * 8,
        )
    stencil = _resolved_stencil(growth, count, dimension)
    ordered, position, alive = _target_vertices(
        source, coarsening.removed_vertices, growth.levels.size
    )
    # Removed new vertices keep their stencil for composition but relate nothing.
    stencil = (np.where(alive[count:, None], stencil[0], -1), stencil[1])
    coordinates, sources, weights, valid = _target_stencil(source, stencil, ordered)
    entity_tables = tuple(
        (
            np.searchsorted(source.vertex_ids, entity_keys(source.mesh, degree)),
            _simplex_keys(cells.rows, degree + 1),
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
        source, cells, coarsening, (sources, weights), position
    )
    edit = CellTopologyEdit(
        "nested_adaptation"
        if refined and coarsened
        else "nested_coarsening"
        if coarsened
        else "nested_refinement",
        coordinates,
        _global_vertices(ordered, source, start.next_vertex),
        source_family_blocks(
            block_definitions,
            tuple(position[cells.rows[rows]].astype(np.int32) for rows in block_rows),
            tuple(cells.ids[rows] for rows in block_rows),
        ),
        sources,
        weights,
        valid,
        _relations(source, start, cells, stencil, coarsening, entity_tables),
        prescribed,
        refinement=refinement_witnesses,
        coarsening=coarsening_witnesses,
    )
    return edit, retired


def _nested_witnesses(
    source: _Source,
    cells: _Cells,
    coarsening: _Coarsening,
    stencil: tuple[np.ndarray, np.ndarray],
    position: np.ndarray,
    /,
) -> tuple[NestedReferenceWitnesses, NestedReferenceWitnesses]:
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
    refined = NestedReferenceWitnesses(
        cells.ids[~restored],
        origins,
        nested_reference_vertices(
            sources[fine], weights[fine], source.vertex_ids[parent_rows]
        ),
    )
    children = source.cells.rows[
        np.searchsorted(source.cells.ids, coarsening.removed_ids)
    ]
    coarse = cells.rows[np.searchsorted(cells.ids, coarsening.link_targets)]
    coarsened = NestedReferenceWitnesses(
        coarsening.removed_ids,
        coarsening.link_targets,
        nested_reference_vertices(
            coarsening.supports[children],
            coarsening.support_weights[children],
            coarse,
        ),
    )
    return refined, coarsened


def _prepared_capacity(cells: int, vertices: int, work_units: int, /) -> _Capacity:
    for value, name in (
        (cells, "maximum_cells"),
        (vertices, "maximum_vertices"),
        (work_units, "maximum_work_units"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
        if value < 1:
            raise ValueError(f"{name} must be positive.")
    return _Capacity(int(cells), int(vertices), int(work_units))


def execute_bisection(
    mesh: CellMesh,
    refine_cell_ids: np.ndarray,
    coarsen_cell_ids: np.ndarray,
    /,
    *,
    source_result: CellMeshingResult,
    hierarchy: BisectionHierarchy | None,
    compatibility: BisectionCompatibility,
    protected_edges: np.ndarray,
    protected_vertices: np.ndarray,
    cell_classes: np.ndarray,
    facet_classes: np.ndarray,
    maximum_closure_iterations: int,
    maximum_cavity_cells: int,
    maximum_cells: int,
    maximum_vertices: int,
    maximum_work_units: int,
    maximum_scratch_bytes: int,
    maximum_wall_seconds: float,
    maximum_geometry_queries: int,
) -> BisectionOutcome:
    """Refine and coarsen one simplex mesh by compatible Maubach bisection.

    Refinement marks whose own conformity closure would split a protected edge are
    rejected (reported in the evidence); the rest are bisected once and closed to
    a conforming mesh. Every closure batch, including the admissibility closures,
    is admitted before it is bisected: active cells, vertices, and bisection steps
    (one work unit each) stay within the declared budgets or the adaptation is
    refused. Coarsening removes, pass by pass, every unprotected bisection vertex
    whose star is the marked, class-uniform children of its bisections and does
    not touch the refinement closure. Children stay in their parent's block with
    the parent's orientation.
    """

    if not isinstance(compatibility, BisectionCompatibility):
        raise TypeError("compatibility must be BisectionCompatibility.")
    if (
        not isinstance(source_result, CellMeshingResult)
        or source_result.mesh.mesh_id != mesh.mesh_id
    ):
        raise ValueError(
            "Native bisection requires its actual accepted scientific source result."
        )
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
        maximum_closure_iterations,
    )
    capacity = _prepared_capacity(maximum_cells, maximum_vertices, maximum_work_units)
    allowance = _UniformAllowance(
        maximum_work_units,
        maximum_geometry_queries,
        maximum_cells,
        maximum_vertices,
        maximum_scratch_bytes,
        maximum_wall_seconds,
        maximum_cavity_cells,
    )
    with _uniform_execution(allowance):
        start = _prepared_start(
            source, request, compatibility, hierarchy, allowance, source_result
        )
        refinement = _refinement(start, request, capacity, dimension)
        blocked = np.isin(start.front.cells.ids, refinement.records.parent_ids)
        coarsening = _coarsening(source, start, request, blocked, dimension)
        cells, records = _merged_forest(start, refinement, coarsening)
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
        return BisectionOutcome(
            edit,
            _target_hierarchy(
                source, start, cells, records, retired, refinement.front, prior=hierarchy
            ),
            evidence,
        )


__all__ = [
    "BisectionCompatibility",
    "BisectionEvidence",
    "BisectionHierarchy",
    "BisectionUniformRefinement",
    "BisectionOutcome",
    "execute_bisection",
]
