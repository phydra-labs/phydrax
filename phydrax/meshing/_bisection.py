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
``k + 1`` entries reversed; the lexicographic form is the lexicographically smaller
tuple by vertex global ID. In two dimensions this is newest-vertex bisection.

Shape regularity: Maubach bisection generates only finitely many similarity
classes from each initial simplex (at most four per triangle in two dimensions,
a dimension-dependent bound for tetrahedra; Maubach 1995, Traxler 1997), so every
shape measure of every descendant is bounded below by its minimum over the
classes of the initial mesh. The matching condition (Stevenson 2008) makes the
conformity closure terminate with the optimal closure-size bound; initial
labelings that violate it are rejected, first made compatible by one
barycentric subdivision, whose simplices are pairwise reflected neighbors, or,
for triangle meshes, closed locally under their source labels.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import combinations
from typing import final, Literal, NamedTuple

import equinox as eqx
import jax
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.meshing._bisection_kernel_host as host_kernel
import phydrax.meshing._result as _dependency_phydrax_meshing__result
import phydrax.meshing._topology_edit as _dependency_phydrax_meshing__topology_edit

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import NativeExecutionScope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..discretization import CellMesh
from ..typing import checked, parse


class BisectionCompatibility(StrEnum):
    """Handle an initial labeling that violates the matching condition.

    ``REJECT`` refuses it. ``UNIFORM_REFINEMENT`` first subdivides every simplex
    barycentrically into pairwise reflected neighbors. ``CONFORMING_CLOSURE``
    keeps the longest-edge labels of a triangle mesh and lets the conformity
    closure bisect every neighbor across a split edge whose refinement edge
    differs. Two-dimensional newest-vertex bisection needs no matching condition:
    bisecting a triangle twice splits all three of its edges, so every even
    uniform generation is conforming and bounds the closure, whose size estimate
    holds for arbitrary initial refinement edges (Karkulik, Pavlicek, Praetorius
    2013). From a labeled start the closure splits only source edges, so each
    source triangle yields at most four children. ``WHOLE_EDGE_STAR`` instead
    splits complete tetrahedral stars of source edges. It needs no matching
    labels and makes no newest-vertex shape-class claim for the entire forest.
    Its records retain each actual ordered subdivision edge with tag three;
    that split's children inherit tag two. A later announced edge starts a new
    tagged reference closure, not the original source's similarity class.
    """

    REJECT = "reject"
    UNIFORM_REFINEMENT = "uniform_refinement"
    CONFORMING_CLOSURE = "conforming_closure"
    WHOLE_EDGE_STAR = "whole_edge_star"


type BisectionRefinementRule = Literal["maubach", "whole_edge_star"]


@final
class _UniformGeometry(StrictModule):
    parent_ids: np.ndarray
    parent_rows: np.ndarray
    parent_vertices: np.ndarray
    child_ids: np.ndarray
    child_vertices: np.ndarray
    reference_vertices: np.ndarray
    barycentric_weights: np.ndarray


@final
class _UniformMetadata(StrictModule):
    parent_blocks: np.ndarray
    parent_row_order: np.ndarray
    parent_levels: np.ndarray
    parent_classes: np.ndarray
    parent_facet_classes: np.ndarray


@final
class _UniformPreparation(StrictModule):
    arrays: dict[str, np.ndarray]


def _prepare_uniform_geometry(
    dimension: int,
    parent_ids: ArrayLike,
    parent_rows: ArrayLike,
    parent_vertices: ArrayLike,
    child_ids: ArrayLike,
    child_vertices: ArrayLike,
    reference_vertices: ArrayLike,
    barycentric_weights: ArrayLike,
    /,
) -> _UniformGeometry:
    if isinstance(dimension, bool) or dimension not in (2, 3):
        raise ValueError(
            f"BisectionUniformRefinement.dimension must be two or three; got {dimension!r}"
        )
    ids = host_kernel.identifiers(parent_ids, "uniform_parent_ids")
    count, width = ids.size, dimension + 1
    if count == 0 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError(
            f"BisectionUniformRefinement.parent_ids must be nonempty and strictly increasing; got {ids!r}"
        )
    children = np.asarray(child_ids, dtype=np.int64)
    arrangements = host_kernel.MAUBACH_TABLES[dimension][4].shape[0]
    if (
        children.shape != (count, arrangements)
        or np.any(children < 0)
        or np.unique(children).size != children.size
    ):
        raise ValueError(
            f"BisectionUniformRefinement.child_ids must hold six or twenty-four distinct siblings per parent; got {(children.shape, count, arrangements, children.min(initial=0), np.unique(children).size)!r}"
        )
    if np.intersect1d(ids, children).size:
        raise ValueError(
            f"BisectionUniformRefinement.child_ids must be disjoint from parent_ids; got {np.intersect1d(ids, children)!r}"
        )
    rows = host_kernel.simplex_rows(parent_rows, count, dimension, "uniform_parent_rows")
    vertices = host_kernel.simplex_rows(
        parent_vertices, count, dimension, "uniform_parent_vertices"
    )
    if not np.array_equal(np.sort(rows, axis=1), np.sort(vertices, axis=1)):
        raise ValueError(
            f"BisectionUniformRefinement.parent_vertices must match parent_rows incidence; got {(rows, vertices)!r}"
        )
    child_rows = np.asarray(child_vertices, dtype=np.int64)
    if child_rows.shape != (count, arrangements, width) or np.any(child_rows < 0):
        raise ValueError(
            f"BisectionUniformRefinement.child_vertices must align with every sibling; got {(child_rows.shape, count, arrangements, width, child_rows.min(initial=0))!r}"
        )
    if np.any(np.diff(np.sort(child_rows, axis=2), axis=2) == 0):
        raise ValueError(
            f"BisectionUniformRefinement.child_vertices must hold distinct vertices per child; got {child_rows!r}"
        )
    references = np.asarray(reference_vertices, dtype=np.float64)
    if references.shape != (count, arrangements, width, dimension) or not np.all(
        np.isfinite(references)
    ):
        raise ValueError(
            f"BisectionUniformRefinement.reference_vertices must hold finite simplex corner maps; got {(references.shape, count, arrangements, width, dimension, bool(np.all(np.isfinite(references))))!r}"
        )
    if np.any(references < 0.0) or np.any(np.sum(references, axis=3) > 1.0):
        raise ValueError(
            f"BisectionUniformRefinement.reference_vertices must lie in the source simplex; got {(references.min(initial=0), np.sum(references, axis=3).max(initial=0))!r}"
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
            f"BisectionUniformRefinement.barycentric_weights must retain normalized source construction weights and reference corners; got {(barycentric.shape, bool(np.all(np.isfinite(barycentric))), barycentric.min(initial=0), barycentric)!r}"
        )
    host_kernel.require_uniform_action_family(barycentric, dimension)
    return _UniformGeometry(
        ids, rows, vertices, children, child_rows, references, barycentric
    )


def _validate_uniform_constructions(geometry: _UniformGeometry, /) -> None:
    rows, child_rows, barycentric = (
        geometry.parent_rows,
        geometry.child_vertices,
        geometry.barycentric_weights,
    )
    count, arrangements, width = child_rows.shape
    construction: dict[int, tuple[tuple[int, float], ...]] = {}
    construction_vertices: dict[tuple[tuple[int, float], ...], int] = {}
    for parent_index in range(count):
        for child_index in range(arrangements):
            for vertex_index in range(width):
                vertex = int(child_rows[parent_index, child_index, vertex_index])
                key = _uniform_construction(
                    rows[parent_index],
                    barycentric[parent_index, child_index, vertex_index],
                )
                known = construction.setdefault(vertex, key)
                if known != key:
                    raise ValueError(
                        f"BisectionUniformRefinement.child_vertices must retain the same barycentric action at incident siblings; got {(vertex, known, key)!r}"
                    )
                owner = construction_vertices.setdefault(key, vertex)
                if owner != vertex:
                    raise ValueError(
                        f"BisectionUniformRefinement.child_vertices must assign one scientific vertex ID to each shared construction; got {(owner, vertex, key)!r}"
                    )
    original_vertices = set(rows.reshape(-1).tolist())
    if not original_vertices.issubset(construction):
        raise ValueError(
            f"BisectionUniformRefinement.parent_rows must retain every source vertex in the subdivision; got {sorted(original_vertices.difference(construction))!r}"
        )
    for vertex, key in construction.items():
        if vertex in original_vertices and key != ((vertex, 1.0),):
            raise ValueError(
                f"BisectionUniformRefinement.child_vertices must retain source corners under their scientific identities; got {(vertex, key)!r}"
            )


def _prepare_uniform_metadata(
    geometry: _UniformGeometry,
    dimension: int,
    parent_blocks: ArrayLike,
    parent_row_order: ArrayLike,
    parent_levels: ArrayLike,
    parent_classes: ArrayLike,
    parent_facet_classes: ArrayLike,
    /,
) -> _UniformMetadata:
    count, width = geometry.parent_rows.shape
    rows = geometry.parent_rows
    blocks = np.asarray(parent_blocks, dtype=np.int32)
    row_order = np.asarray(parent_row_order, dtype=np.int64)
    levels = np.asarray(parent_levels, dtype=np.int32)
    classes = np.asarray(parent_classes, dtype=np.int64)
    facet_classes = np.asarray(parent_facet_classes, dtype=np.int64)
    if any(value.shape != (count,) for value in (blocks, row_order, levels, classes)):
        raise ValueError(
            f"BisectionUniformRefinement.parent_metadata must align blocks, row order, levels and classes with parent IDs; got {(count, blocks.shape, row_order.shape, levels.shape, classes.shape)!r}"
        )
    if np.any(blocks < 0) or np.any(row_order < 0) or np.any(levels < 0):
        raise ValueError(
            f"BisectionUniformRefinement.parent_metadata must have nonnegative blocks, row order and levels; got {(blocks.min(initial=0), row_order.min(initial=0), levels.min(initial=0))!r}"
        )
    if facet_classes.shape != (count, width):
        raise ValueError(
            f"BisectionUniformRefinement.parent_facet_classes must align with every parent facet; got {(facet_classes.shape, count, width)!r}"
        )
    if np.unique(np.stack((blocks, row_order), axis=1), axis=0).shape[0] != count:
        raise ValueError(
            f"BisectionUniformRefinement.parent_row_order must identify distinct block-row positions; got {(count, np.unique(np.stack((blocks, row_order), axis=1), axis=0).shape[0])!r}"
        )
    source_facets: dict[tuple[int, ...], int] = {}
    for parent_index in range(count):
        for facet_index, local in enumerate(combinations(range(width), dimension)):
            key = tuple(sorted(int(rows[parent_index, index]) for index in local))
            cell_class = int(facet_classes[parent_index, facet_index])
            known = source_facets.setdefault(key, cell_class)
            if known != cell_class:
                raise ValueError(
                    f"BisectionUniformRefinement.parent_facet_classes must agree at incident parent facets; got {(key, known, cell_class)!r}"
                )
    return _UniformMetadata(blocks, row_order, levels, classes, facet_classes)


def _validate_uniform_source(
    geometry: _UniformGeometry,
    metadata: _UniformMetadata,
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    dimension: int,
    /,
) -> None:
    if source.mesh.topological_dimension != dimension:
        raise TypeError(
            f"BisectionUniformRefinement.source must have the declared simplex dimension; got {(source.mesh.topological_dimension, dimension)!r}"
        )
    ids, rows = geometry.parent_ids, geometry.parent_rows
    blocks, row_order = metadata.parent_blocks, metadata.parent_row_order
    vertex_values, host_blocks = jax.device_get(
        (
            source.mesh.vertex_global_ids,
            tuple((block.global_ids, block.vertices) for block in source.mesh.blocks),
        )
    )
    source_vertices = np.asarray(vertex_values, dtype=np.int64)
    for identifier, block, row, parent in zip(ids, blocks, row_order, rows, strict=True):
        if block >= len(source.mesh.blocks):
            raise ValueError(
                f"BisectionUniformRefinement.parent_blocks must bind retained scientific blocks; got {(block, len(source.mesh.blocks))!r}"
            )
        original = source.mesh.blocks[int(block)]
        original_ids, original_vertices = host_blocks[int(block)]
        if row >= original.cell_count or np.asarray(original_ids)[row] != identifier:
            raise ValueError(
                f"BisectionUniformRefinement.parent_blocks must bind parent row order and parent IDs in the source; got {(block, row, identifier, original.cell_count)!r}"
            )
        if not np.array_equal(
            source_vertices[np.asarray(original_vertices)[row]], parent
        ):
            raise ValueError(
                f"BisectionUniformRefinement.parent_rows must bind the retained scientific source vertices; got {(parent, source_vertices[np.asarray(original_vertices)[row]])!r}"
            )


def _prepare_uniform_refinement(
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
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
) -> _UniformPreparation:
    geometry = _prepare_uniform_geometry(
        dimension,
        parent_ids,
        parent_rows,
        parent_vertices,
        child_ids,
        child_vertices,
        reference_vertices,
        barycentric_weights,
    )
    _validate_uniform_constructions(geometry)
    metadata = _prepare_uniform_metadata(
        geometry,
        dimension,
        parent_blocks,
        parent_row_order,
        parent_levels,
        parent_classes,
        parent_facet_classes,
    )
    _validate_uniform_source(geometry, metadata, source, dimension)
    return _UniformPreparation(
        {
            "parent_ids": geometry.parent_ids,
            "parent_blocks": metadata.parent_blocks,
            "parent_row_order": metadata.parent_row_order,
            "parent_rows": geometry.parent_rows,
            "parent_vertices": geometry.parent_vertices,
            "parent_tags": host_kernel.tag_array(
                parent_tags, geometry.parent_ids.size, dimension, "uniform_parent_tags"
            ),
            "parent_levels": metadata.parent_levels,
            "parent_classes": metadata.parent_classes,
            "parent_facet_classes": metadata.parent_facet_classes,
            "child_ids": geometry.child_ids,
            "child_vertices": geometry.child_vertices,
            "reference_vertices": geometry.reference_vertices,
            "barycentric_weights": geometry.barycentric_weights,
        }
    )


@final
class BisectionUniformRefinement(StrictModule, NonTrainableState):
    """Uniform compatibility siblings and their source reference actions."""

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
    source: _dependency_phydrax_meshing__result.CellMeshingResult
    lineage_id: str = eqx.field(static=True)

    @checked
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
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
    ) -> None:
        prepared = _prepare_uniform_refinement(
            dimension,
            parent_ids,
            parent_blocks,
            parent_row_order,
            parent_rows,
            parent_vertices,
            parent_tags,
            parent_levels,
            parent_classes,
            parent_facet_classes,
            child_ids,
            child_vertices,
            reference_vertices,
            barycentric_weights,
            source=source,
        )
        arrays = prepared.arrays
        self.dimension = dimension
        self.source = source
        (
            self.parent_ids,
            self.parent_blocks,
            self.parent_row_order,
            self.parent_rows,
            self.parent_vertices,
            self.parent_tags,
            self.parent_levels,
            self.parent_classes,
            self.parent_facet_classes,
            self.child_ids,
            self.child_vertices,
            self.reference_vertices,
            self.barycentric_weights,
        ) = jax.device_put(tuple(arrays.values()))
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "bisection-uniform-refinement",
                "dimension": dimension,
                "source": source.result_id,
                "arrays": array_tree_fingerprint(arrays),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: BisectionUniformRefinement, /
    ) -> BisectionUniformRefinement:
        """Rebuild stored uniform actions through their source-binding validation."""
        if type(value) is not cls:
            raise TypeError(
                f"BisectionUniformRefinement.stored_fields must have type {cls.__name__}; got {type(value).__name__}"
            )
        return cls(value.dimension, *value.host_arrays().values(), source=value.source)

    def host_arrays(self) -> dict[str, np.ndarray]:
        """Snapshot the action leaves once, retaining constructor field order."""
        arrays = {
            "parent_ids": self.parent_ids,
            "parent_blocks": self.parent_blocks,
            "parent_row_order": self.parent_row_order,
            "parent_rows": self.parent_rows,
            "parent_vertices": self.parent_vertices,
            "parent_tags": self.parent_tags,
            "parent_levels": self.parent_levels,
            "parent_classes": self.parent_classes,
            "parent_facet_classes": self.parent_facet_classes,
            "child_ids": self.child_ids,
            "child_vertices": self.child_vertices,
            "reference_vertices": self.reference_vertices,
            "barycentric_weights": self.barycentric_weights,
        }
        snapshot = jax.device_get(arrays)
        # JAX traverses mapping keys lexicographically; constructor order is explicit.
        return {name: np.asarray(snapshot[name]) for name in arrays}

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


type _UniformConstruction = tuple[tuple[int, float], ...]


@final
class _UniformCohort(StrictModule):
    original: dict[str, np.ndarray]
    incoming: dict[str, np.ndarray]
    positions: np.ndarray


@final
class _LiveUniformConstructions(StrictModule):
    live: dict[_UniformConstruction, int]
    vertices: dict[int, _UniformConstruction]


@final
class _UniformRemapping(StrictModule):
    child_vertices: np.ndarray
    remap: dict[int, int]


def _uniform_construction(
    corners: np.ndarray, weights: np.ndarray, /
) -> _UniformConstruction:
    return tuple(
        sorted(
            (int(identifier), float(weight))
            for identifier, weight in zip(corners, weights, strict=True)
            if weight != 0.0
        )
    )


def _prepare_uniform_cohort(
    regenerated: BisectionUniformRefinement,
    retained: BisectionUniformRefinement,
    restored_parent_ids: ArrayLike,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _UniformCohort:
    if (
        regenerated.dimension != retained.dimension
        or regenerated.source.result_id != retained.source.result_id
    ):
        raise ValueError(
            f"BisectionUniformRefinement.source must match during reconciliation; got {(regenerated.source.result_id, retained.source.result_id, regenerated.dimension, retained.dimension)!r}"
        )
    width = retained.dimension + 1
    entries = regenerated.child_vertices.size + retained.child_vertices.size
    host_kernel.uniform_count(
        entries * width, entries * (512 + 128 * width), budget=budget
    )
    original = retained.host_arrays()
    incoming = regenerated.host_arrays()
    restored = host_kernel.identifiers(restored_parent_ids, "restored_uniform_parent_ids")
    if restored.size == 0 or np.any(restored[1:] <= restored[:-1]):
        raise ValueError(
            f"BisectionUniformRefinement.restored_parent_ids must be nonempty and strictly increasing; got {restored!r}"
        )
    if not np.array_equal(incoming["parent_ids"], restored):
        raise ValueError(
            f"BisectionUniformRefinement.restored_parent_ids must match the regenerated parent cohort; got {(incoming['parent_ids'], restored)!r}"
        )
    positions = np.searchsorted(original["parent_ids"], restored)
    if np.any(positions >= original["parent_ids"].size):
        raise ValueError(
            f"BisectionUniformRefinement.restored_parent_ids must belong to the retained parent cohort; got {(restored, original['parent_ids'])!r}"
        )
    if not np.array_equal(original["parent_ids"][positions], restored):
        raise ValueError(
            f"BisectionUniformRefinement.restored_parent_ids must belong to the retained parent cohort; got {(restored, original['parent_ids'])!r}"
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
                f"BisectionUniformRefinement.{name} must retain the stored parent metadata; got {(original[name][positions], incoming[name])!r}"
            )
    return _UniformCohort(original, incoming, positions)


def _prepare_live_uniform_constructions(
    cohort: _UniformCohort, /
) -> _LiveUniformConstructions:
    original, positions = cohort.original, cohort.positions
    live: dict[_UniformConstruction, int] = {}
    live_vertices: dict[int, _UniformConstruction] = {}
    kept = np.ones(original["parent_ids"].shape, dtype=np.bool_)
    kept[positions] = False

    for parent in np.flatnonzero(kept):
        corners = original["parent_rows"][parent]
        for vertices, weights in zip(
            original["child_vertices"][parent],
            original["barycentric_weights"][parent],
            strict=True,
        ):
            for identifier, coefficients in zip(vertices, weights, strict=True):
                key = _uniform_construction(corners, coefficients)
                identifier = int(identifier)
                if key in live and live[key] != identifier:
                    raise ValueError(
                        f"BisectionUniformRefinement.child_vertices must assign one live SCI to each shared construction; got {(key, live[key], identifier)!r}"
                    )
                if identifier in live_vertices and live_vertices[identifier] != key:
                    raise ValueError(
                        f"BisectionUniformRefinement.child_vertices must retain one source support per live SCI; got {(identifier, live_vertices[identifier], key)!r}"
                    )
                live[key], live_vertices[identifier] = identifier, key
    return _LiveUniformConstructions(live, live_vertices)


def _reconciled_uniform_identity(
    identifier: int,
    key: _UniformConstruction,
    generated_vertices: dict[int, _UniformConstruction],
    constructions: _LiveUniformConstructions,
    /,
) -> int:
    if identifier in generated_vertices and generated_vertices[identifier] != key:
        raise ValueError(
            f"BisectionUniformRefinement.child_vertices must retain one source support per regenerated SCI; got {(identifier, generated_vertices[identifier], key)!r}"
        )
    generated_vertices[identifier] = key
    if key in constructions.live:
        replacement = constructions.live[key]
        if len(key) == 1 and key[0][1] == 1.0 and replacement != identifier:
            raise ValueError(
                f"BisectionUniformRefinement.child_vertices must preserve scientific source corner IDs; got {(identifier, replacement, key)!r}"
            )
        return replacement
    if identifier in constructions.vertices:
        raise ValueError(
            f"BisectionUniformRefinement.child_vertices must avoid collisions with different live constructions; got {(identifier, constructions.vertices[identifier], key)!r}"
        )
    return identifier


def _remap_uniform_vertices(
    cohort: _UniformCohort, constructions: _LiveUniformConstructions, /
) -> _UniformRemapping:
    incoming = cohort.incoming
    remap: dict[int, int] = {}
    generated_vertices: dict[int, _UniformConstruction] = {}
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
                key = _uniform_construction(corners, coefficients)
                replacement = _reconciled_uniform_identity(
                    identifier, key, generated_vertices, constructions
                )
                if replacement != identifier:
                    remap[identifier] = replacement
                    child_vertices[parent, child, corner] = replacement
    return _UniformRemapping(child_vertices, remap)


@checked
def reconcile_uniform_refinement(
    regenerated: BisectionUniformRefinement,
    retained: BisectionUniformRefinement,
    restored_parent_ids: ArrayLike,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[BisectionUniformRefinement, dict[int, int]]:
    """Replace restored roots while preserving live construction SCIs.

    The collective preparation authenticates which roots were restored.
    Only unaffected, still-live uniform roots supply reusable vertices; retired
    construction identities never become a source for a new issuance.
    """
    cohort = _prepare_uniform_cohort(
        regenerated, retained, restored_parent_ids, budget=budget
    )
    constructions = _prepare_live_uniform_constructions(cohort)
    remapping = _remap_uniform_vertices(cohort, constructions)
    original, incoming, positions = cohort.original, cohort.incoming, cohort.positions
    child_vertices, remap = remapping.child_vertices, remapping.remap
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
class _HierarchyAuthority(StrictModule):
    scientific_ids: np.ndarray
    scientific_blocks: np.ndarray
    canonical_scientific_ids: np.ndarray
    canonical_scientific_blocks: np.ndarray


@final
class _HierarchyIssuers(StrictModule):
    vertex_counter: int = eqx.field(static=True)
    cell_counter: int = eqx.field(static=True)


def _prepare_hierarchy_authority(
    arrays: dict[str, np.ndarray],
    scientific_cell_ids: ArrayLike,
    scientific_block_ids: ArrayLike,
    uniform_refinement: BisectionUniformRefinement | None,
    /,
) -> _HierarchyAuthority:
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
            f"BisectionHierarchy.scientific_block_ids must cover active and forest cells with nonnegative blocks in ascending SCI order; got {(scientific_ids, scientific_blocks)!r}"
        )
    authority_parts = [
        arrays["cell_global_ids"],
        arrays["record_parent_ids"],
        arrays["record_child_ids"].reshape(-1),
    ]
    if uniform_refinement is not None:
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
            f"BisectionHierarchy.scientific_block_ids must retain parent block ownership at child cells; got {(parent_authority, child_authority)!r}"
        )
    return _HierarchyAuthority(
        scientific_ids,
        scientific_blocks,
        canonical_scientific_ids,
        canonical_scientific_blocks,
    )


def _validate_uniform_ancestry(
    arrays: dict[str, np.ndarray],
    dimension: int,
    uniform_refinement: BisectionUniformRefinement,
    /,
) -> None:
    if uniform_refinement.dimension != dimension:
        raise ValueError(
            f"BisectionHierarchy.uniform_refinement must have the declared simplex dimension; got {(dimension, uniform_refinement.dimension)!r}"
        )
    roots = np.concatenate((arrays["cell_global_ids"], arrays["record_parent_ids"]))
    restored = np.isin(np.asarray(uniform_refinement.parent_ids), roots)
    children = np.asarray(uniform_refinement.child_ids)
    if not np.all(np.isin(children[~restored], roots)) or np.any(
        np.isin(children[restored], roots)
    ):
        raise ValueError(
            f"BisectionHierarchy.uniform_refinement must retain child or restored-parent ancestry without mixing both; got {(roots, children, restored)!r}"
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
                f"BisectionHierarchy.ordered_vertices must retain the recorded uniform source incidence; got {(node_vertices[positions], expected)!r}"
            )


def _prepare_hierarchy_issuers(
    arrays: dict[str, np.ndarray],
    uniform_refinement: BisectionUniformRefinement | None,
    next_vertex_id: int,
    next_cell_id: int,
    /,
) -> _HierarchyIssuers:
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
        cell_high = max(cell_high, int(np.max(np.asarray(uniform_refinement.child_ids))))
    if vertex_counter <= vertex_high or cell_counter <= cell_high:
        raise ValueError(
            f"BisectionHierarchy.counters must exceed every issued ID; got {(vertex_counter, vertex_high, cell_counter, cell_high)!r}"
        )
    return _HierarchyIssuers(vertex_counter, cell_counter)


@final
class BisectionHierarchy(StrictModule, NonTrainableState):
    """Persistent tagged labels and exact binary forest of one bisection mesh.

    Active cells are listed by ascending global ID with their lexicographic Maubach
    tuple (vertex global IDs), tag, and generation. Each forest record is one
    bisection whose children are leaves or have descendants: the parent cell ID,
    source block, block row (global IDs), Maubach tuple and tag, the two
    child cell IDs (first, second), and the bisection vertex. Retired entity
    tables keep the global IDs of edges/faces removed by refinement so that
    coarsening restores them under their source IDs. Vertex and cell IDs are
    issued from the high-water counters and never reused.
    ``scientific_cell_ids`` and ``scientific_block_ids`` retain the source
    authored block ownership of every issued cell, including inactive parents.
    Descendants inherit it at birth; coordinate presentation regrouping never
    rewrites it. The forest's source-block rows remain presentation routing.
    For ``whole_edge_star``, a record's ordered tuple is the actual parent
    relabeling: its first and tag-three vertices are the announced edge, and
    its two children inherit the corresponding tag-two tuples. Unsplit cells
    retain their labels. Each relabeling requires its own exact geometric
    reference proof; no finite similarity closure over arbitrary edge sequences
    is asserted.

    A hierarchy applies to a mesh iff its active cell IDs equal the mesh cell IDs
    and every cell has the same vertex-ID set in both.
    """

    dimension: int = eqx.field(static=True)
    refinement_rule: BisectionRefinementRule = eqx.field(static=True)
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

    @checked
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
        refinement_rule: BisectionRefinementRule,
    ) -> None:
        if isinstance(dimension, bool) or dimension not in (2, 3):
            raise ValueError(
                f"BisectionHierarchy.dimension must be two or three; got {dimension!r}"
            )
        refinement_rule = parse(
            refinement_rule, BisectionRefinementRule, "refinement_rule"
        )
        if refinement_rule == "whole_edge_star" and dimension != 3:
            raise ValueError(
                f"BisectionHierarchy.refinement_rule must select a supported dimensional subdivision law; got {(refinement_rule, dimension)!r}"
            )
        arrays = host_kernel.validated_forest(
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
        authority = _prepare_hierarchy_authority(
            arrays, scientific_cell_ids, scientific_block_ids, uniform_refinement
        )
        scientific_ids, scientific_blocks = (
            authority.scientific_ids,
            authority.scientific_blocks,
        )
        canonical_scientific_ids = authority.canonical_scientific_ids
        canonical_scientific_blocks = authority.canonical_scientific_blocks
        if uniform_refinement is not None:
            _validate_uniform_ancestry(arrays, dimension, uniform_refinement)
        retired_keys, retired_ids = host_kernel.retired_tables(
            retired_entity_keys, retired_entity_ids, dimension
        )
        issuers = _prepare_hierarchy_issuers(
            arrays, uniform_refinement, next_vertex_id, next_cell_id
        )
        vertex_counter, cell_counter = issuers.vertex_counter, issuers.cell_counter
        self.dimension = dimension
        self.refinement_rule = refinement_rule
        _host_cell_global_ids = np.asarray(arrays["cell_global_ids"])
        _host_ordered_vertices = np.asarray(arrays["ordered_vertices"])
        _host_tags = np.asarray(arrays["tags"])
        _host_generations = np.asarray(arrays["generations"])
        _host_scientific_cell_ids = np.asarray(scientific_ids)
        _host_scientific_block_ids = np.asarray(scientific_blocks)
        _host_record_parent_ids = np.asarray(arrays["record_parent_ids"])
        _host_record_parent_blocks = np.asarray(arrays["record_parent_blocks"])
        _host_record_parent_rows = np.asarray(arrays["record_parent_rows"])
        _host_record_parent_vertices = np.asarray(arrays["record_parent_vertices"])
        _host_record_parent_tags = np.asarray(arrays["record_parent_tags"])
        _host_record_child_ids = np.asarray(arrays["record_child_ids"])
        _host_record_vertex_ids = np.asarray(arrays["record_vertex_ids"])
        _host_retired_entity_keys = tuple(np.asarray(value) for value in retired_keys)
        _host_retired_entity_ids = tuple(np.asarray(value) for value in retired_ids)
        (
            self.cell_global_ids,
            self.ordered_vertices,
            self.tags,
            self.generations,
            self.scientific_cell_ids,
            self.scientific_block_ids,
            self.record_parent_ids,
            self.record_parent_blocks,
            self.record_parent_rows,
            self.record_parent_vertices,
            self.record_parent_tags,
            self.record_child_ids,
            self.record_vertex_ids,
            self.retired_entity_keys,
            self.retired_entity_ids,
        ) = jax.device_put(
            (
                _host_cell_global_ids,
                _host_ordered_vertices,
                _host_tags,
                _host_generations,
                _host_scientific_cell_ids,
                _host_scientific_block_ids,
                _host_record_parent_ids,
                _host_record_parent_blocks,
                _host_record_parent_rows,
                _host_record_parent_vertices,
                _host_record_parent_tags,
                _host_record_child_ids,
                _host_record_vertex_ids,
                _host_retired_entity_keys,
                _host_retired_entity_ids,
            )
        )
        self.uniform_refinement = uniform_refinement
        self.next_vertex_id = vertex_counter
        self.next_cell_id = cell_counter
        self.hierarchy_id = canonical_fingerprint(
            {
                "kind": "bisection-hierarchy",
                "dimension": dimension,
                "refinement_rule": refinement_rule,
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

    @classmethod
    @checked
    def from_stored_fields(cls, value: BisectionHierarchy, /) -> BisectionHierarchy:
        """Rebuild a stored forest through ancestry and issuance validation."""
        if type(value) is not cls:
            raise TypeError(
                f"BisectionHierarchy.stored_fields must have type {cls.__name__}; got {type(value).__name__}"
            )
        return cls(
            value.dimension,
            value.cell_global_ids,
            value.ordered_vertices,
            value.tags,
            value.generations,
            record_parent_ids=value.record_parent_ids,
            record_parent_blocks=value.record_parent_blocks,
            record_parent_rows=value.record_parent_rows,
            record_parent_vertices=value.record_parent_vertices,
            record_parent_tags=value.record_parent_tags,
            record_child_ids=value.record_child_ids,
            record_vertex_ids=value.record_vertex_ids,
            retired_entity_keys=value.retired_entity_keys,
            retired_entity_ids=value.retired_entity_ids,
            next_vertex_id=value.next_vertex_id,
            next_cell_id=value.next_cell_id,
            uniform_refinement=value.uniform_refinement,
            scientific_cell_ids=value.scientific_cell_ids,
            scientific_block_ids=value.scientific_block_ids,
            refinement_rule=value.refinement_rule,
        )


@final
class _BisectionEvidencePreparation(StrictModule):
    counts: dict[str, int]
    rejected_refinements: np.ndarray
    rejected_coarsenings: np.ndarray


def _prepare_bisection_evidence(
    counts: dict[str, int],
    initially_compatible: bool | None,
    uniform_refinement_applied: bool,
    rejected_refinement_ids: ArrayLike,
    rejected_coarsening_ids: ArrayLike,
    /,
) -> _BisectionEvidencePreparation:
    if any(value < 0 for value in counts.values()):
        raise ValueError(f"BisectionEvidence.counts must be nonnegative; got {counts!r}")
    if counts["accepted_refinements"] > counts["requested_refinements"]:
        raise ValueError(
            f"BisectionEvidence.accepted_refinements must not exceed requested refinements; got {(counts['accepted_refinements'], counts['requested_refinements'])!r}"
        )
    if initially_compatible is not None and not isinstance(initially_compatible, bool):
        raise TypeError(
            f"BisectionEvidence.initially_compatible must be bool or None; got {type(initially_compatible).__name__!r}"
        )
    if not isinstance(uniform_refinement_applied, bool):
        raise TypeError(
            f"BisectionEvidence.uniform_refinement_applied must be bool; got {type(uniform_refinement_applied).__name__!r}"
        )
    rejected_refinements = host_kernel.identifiers(
        rejected_refinement_ids, "rejected_refinement_ids"
    )
    rejected_coarsenings = host_kernel.identifiers(
        rejected_coarsening_ids, "rejected_coarsening_ids"
    )
    return _BisectionEvidencePreparation(
        counts, rejected_refinements, rejected_coarsenings
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
        prepared = _prepare_bisection_evidence(
            counts,
            initially_compatible,
            uniform_refinement_applied,
            rejected_refinement_ids,
            rejected_coarsening_ids,
        )
        counts = prepared.counts
        rejected_refinements = prepared.rejected_refinements
        rejected_coarsenings = prepared.rejected_coarsenings
        self.requested_refinements = counts["requested_refinements"]
        self.accepted_refinements = counts["accepted_refinements"]
        _host_rejected_refinement_ids = np.asarray(rejected_refinements)
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
        _host_rejected_coarsening_ids = np.asarray(rejected_coarsenings)
        (
            self.rejected_refinement_ids,
            self.rejected_coarsening_ids,
        ) = jax.device_put(
            (
                _host_rejected_refinement_ids,
                _host_rejected_coarsening_ids,
            )
        )
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

    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    hierarchy: BisectionHierarchy
    evidence: BisectionEvidence


@checked
def execute_bisection(
    mesh: CellMesh,
    refine_cell_ids: np.ndarray,
    coarsen_cell_ids: np.ndarray,
    /,
    *,
    source_result: _dependency_phydrax_meshing__result.CellMeshingResult,
    hierarchy: BisectionHierarchy | None,
    compatibility: BisectionCompatibility,
    protected_edges: np.ndarray,
    protected_vertices: np.ndarray,
    cell_classes: np.ndarray,
    facet_classes: np.ndarray,
    maximum_closure_iterations: int,
    refinement_generations: int,
    refinement_target_cells: int | None,
    maximum_cavity_cells: int,
    maximum_cells: int,
    maximum_vertices: int,
    maximum_work_units: int,
    maximum_scratch_bytes: int,
    maximum_wall_seconds: float,
    maximum_geometry_queries: int,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
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

    compatibility = parse(compatibility, BisectionCompatibility, "compatibility")
    refinement_generations = positive_integer(refinement_generations, "refinement_generations")
    if refinement_target_cells is not None:
        refinement_target_cells = positive_integer(
            refinement_target_cells, "refinement_target_cells"
        )
    if (
        refinement_generations != 1 or refinement_target_cells is not None
    ) and compatibility is not BisectionCompatibility.WHOLE_EDGE_STAR:
        raise ValueError("Bisection.refinement_generations/target_cells require WHOLE_EDGE_STAR")
    if compatibility is BisectionCompatibility.WHOLE_EDGE_STAR and mesh.topological_dimension != 3:
        raise ValueError("Bisection.WHOLE_EDGE_STAR requires tetrahedral cells")
    if (
        hierarchy is not None
        and hierarchy.refinement_rule == "whole_edge_star"
        and refine_cell_ids.size
        and compatibility is not BisectionCompatibility.WHOLE_EDGE_STAR
    ):
        raise ValueError(
            "Bisection.compatibility must retain WHOLE_EDGE_STAR for further refinement of its forest"
        )
    if source_result.mesh.mesh_id != mesh.mesh_id:
        raise ValueError(
            f"Bisection.source_result must bind the accepted source mesh; got {(source_result.mesh.mesh_id, mesh.mesh_id)!r}"
        )
    source = host_kernel.prepared_source(mesh)
    dimension = source.dimension
    request = host_kernel.prepared_request(
        source,
        refine_cell_ids,
        coarsen_cell_ids,
        protected_edges,
        protected_vertices,
        cell_classes,
        facet_classes,
        maximum_closure_iterations,
    )
    capacity = host_kernel.prepared_capacity(
        maximum_cells, maximum_vertices, maximum_work_units
    )
    allowance = host_kernel.UniformAllowance(
        maximum_work_units,
        maximum_geometry_queries,
        maximum_cells,
        maximum_vertices,
        maximum_scratch_bytes,
        maximum_wall_seconds,
        maximum_cavity_cells,
    )
    with (
        host_kernel.uniform_budget(allowance, scope=scope, budget=budget),
        budget.temporary_scope(),
    ):
        start = host_kernel.prepared_start(
            source,
            request,
            compatibility,
            hierarchy,
            allowance,
            source_result,
            scope=scope,
            budget=budget,
        )
        refinement = host_kernel.refinement(
            start, request, capacity, dimension, compatibility=compatibility,
            refinement_generations=refinement_generations,
            refinement_target_cells=refinement_target_cells,
            coordinates=source.coordinates, scope=scope, budget=budget
        )
        blocked = np.isin(start.front.cells.ids, refinement.records.parent_ids)
        coarsening = host_kernel.coarsening(
            source, start, request, blocked, dimension, scope=scope, budget=budget
        )
        cells, records = host_kernel.merged_forest(start, refinement, coarsening)
        edit, retired = host_kernel.build_bisection_edit(
            source, start, refinement, coarsening, cells, scope=scope, budget=budget
        )
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
        outcome = BisectionOutcome(
            edit,
            host_kernel.target_hierarchy(
                source, start, cells, records, retired, refinement.front, prior=hierarchy,
                refinement_rule=(
                    "whole_edge_star"
                    if (
                        compatibility is BisectionCompatibility.WHOLE_EDGE_STAR
                        or (hierarchy is not None and hierarchy.refinement_rule == "whole_edge_star")
                    )
                    else "maubach"
                ),
            ),
            evidence,
        )
        budget.retain_owner(outcome)
        return outcome


__all__ = [
    "BisectionCompatibility",
    "BisectionEvidence",
    "BisectionHierarchy",
    "BisectionUniformRefinement",
    "BisectionOutcome",
    "execute_bisection",
]
