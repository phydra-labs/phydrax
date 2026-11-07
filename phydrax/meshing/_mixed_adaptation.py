#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conforming marked adaptation for tetrahedral and hybrid volume meshes."""

from __future__ import annotations

from collections import defaultdict, deque
from fractions import Fraction
from functools import cache
from heapq import heappop, heappush
from itertools import combinations
from typing import final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike, NDArray

from .._fingerprint import canonical_fingerprint
from .._meshcore import mixed_template, mixed_template_counts, MixedElementKind
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    coordinate_reference_chain,
    PerCellRestrictedCellGeometryElement,
    restriction_chart_arguments,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization._coordinate_enclosure import (
    axes,
    compose_expression,
    constant,
    CoordinateSourceBank,
    ExactScalar,
    Expression,
    expression_equal,
    scale,
    sum_polynomials,
)
from ..discretization._exact_algebraic_geometry import RadicalElement
from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from ..discretization._nested_reference import (
    _reference_shape_weights,
    coarsen_reference_cell_maps,
    compose_reference_cell_maps,
    nested_cell_reference_maps,
)
from ..discretization._nested_reference_types import (
    evaluate_reference_chart_arguments,
    exact_reference_cell_vertices,
    ExactReferenceChartExpression,
    freeze_reference_chart_arguments,
    NestedReferenceWitnesses,
    reference_chart_arguments_identity,
    ReferenceCellMapWitnesses,
    ReferenceCellPartition,
    validate_reference_cell_map_witnesses,
    validate_reference_cell_partition,
)
from ..discretization._periodic_exact import identification_id, identification_orders
from ..discretization._periodic_topology import (
    _entity_vertex_rows,
    _interleaved_keys,
    _orientation_orders,
)
from ..discretization._reference_cell import (
    facet_orientation_between,
    reference_cell_topology,
)
from ..linalg._exact import (
    ExactInconsistentSystemError,
    ExactRankDeficientSystemError,
    solve_exact,
    solve_exact_sparse,
)
from ._lineage import EntityLineageKind
from ._periodic import (
    periodic_entity_orbit_map,
    periodic_next_entity_ids,
    periodic_orbit_closure,
    PeriodicQuotientHistory,
    PeriodicRefinementEdgeLock,
    update_periodic_quotient_history,
)
from ._topology_edit import (
    assemble_topology_edit,
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    key_rows,
    PeriodicTopologyEditWitness,
    PrescribedEntityIds,
    SharedFaceWitnesses,
    TopologyEditBlock,
)


_PointKey = tuple[tuple[int, Fraction], ...]
_FaceSignature = tuple[tuple[_PointKey, ...], ...]
_LocalPointCache = dict[tuple[str, tuple[float, ...], int], _PointKey]


class MixedSibling(NamedTuple):
    """One family-preserving refinement and its exact reference partition."""

    parent_id: int
    block_name: str
    cell_kind: str
    parent_vertices: tuple[int, ...]
    child_ids: tuple[int, ...]
    child_reference_vertices: tuple[tuple[tuple[Fraction, ...], ...], ...]
    cell_class: int
    column_id: int | None
    interval_index: int | None


class MixedCoarseningMap(NamedTuple):
    """Exact child-reference corners in one restored mixed parent reference cell."""

    parent_cell_id: int
    parent_block_name: str
    parent_cell_kind: str
    child_cell_ids: NDArray[np.int64]
    child_reference_vertices: tuple[tuple[tuple[Fraction, ...], ...], ...]


type _MixedReferenceAffineMap = tuple[
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
]


@final
class _RetainedCoordinateBank(StrictModule, NonTrainableState):
    """Exact coefficient data and observational source provenance, not authority."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)
    exact_coordinates: CoordinateSourceBank | None = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        source_map_id: str,
        exact_coordinates: CoordinateSourceBank | None,
        /,
    ) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (source_id, source_revision, source_map_id)
        ):
            raise ValueError(
                "Retained coordinate provenance requires full source identities."
            )
        if exact_coordinates is not None:
            exact_coordinates = tuple(tuple(row) for row in exact_coordinates)
            if (
                not exact_coordinates
                or not exact_coordinates[0]
                or any(len(row) != len(exact_coordinates[0]) for row in exact_coordinates)
                or any(
                    not isinstance(value, (Fraction, RadicalElement))
                    for row in exact_coordinates
                    for value in row
                )
            ):
                raise ValueError(
                    "Retained coordinate coefficients must form an exact scalar bank."
                )
            try:
                rounded = np.asarray(exact_coordinates, dtype=np.float64)
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    "Retained coordinate coefficients must have finite shadows."
                ) from error
            if not np.all(np.isfinite(rounded)):
                raise ValueError(
                    "Retained coordinate coefficients must have finite shadows."
                )
        self.source_id = source_id
        self.source_revision = source_revision
        self.source_map_id = source_map_id
        self.exact_coordinates = exact_coordinates

    @classmethod
    def from_stored_fields(
        cls, value: _RetainedCoordinateBank, /, *, coordinates: ArrayLike
    ) -> _RetainedCoordinateBank:
        """Revalidate retained data and its coefficient shadows, not source authority."""
        if type(value) is not cls:
            raise TypeError("value must be a retained coordinate bank.")
        rebuilt = cls(
            value.source_id,
            value.source_revision,
            value.source_map_id,
            value.exact_coordinates,
        )
        if rebuilt.exact_coordinates is not None and not np.array_equal(
            np.asarray(rebuilt.exact_coordinates, dtype=np.float64),
            np.asarray(coordinates, dtype=np.float64),
        ):
            raise ValueError(
                "Retained coordinate coefficients change their execution shadows."
            )
        return rebuilt

    def source_coordinates(self) -> CoordinateSourceBank | None:
        return self.exact_coordinates


def _require_retained_bank_origin(
    source_geometry: CellGeometrySpec, target_geometry: CellGeometrySpec, /
) -> None:
    """Compare retained provenance against its actual preceding coordinate owner."""
    retained = target_geometry.source_map
    if type(retained) is not _RetainedCoordinateBank:
        return
    origin = source_geometry.source_map
    if (
        origin is None
        or retained.source_id != origin.source_id
        or retained.source_revision != origin.source_revision
        or retained.source_map_id != origin.source_map_id
        or retained.exact_coordinates != source_geometry.source_coordinates()
    ):
        raise ValueError(
            "Retained coordinate bank differs from its actual source geometry."
        )


class MixedActiveCell(NamedTuple):
    """Identity and class state needed to bind a hierarchy to its active mesh."""

    cell_id: int
    block_name: str
    cell_kind: str
    vertices: tuple[int, ...]
    cell_class: int
    column_id: int | None
    interval_index: int | None


class MixedForestRoot(NamedTuple):
    """One initial cell and its exact chart into the independently bound root."""

    cell_id: int
    block_name: str
    cell_kind: str
    vertices: tuple[int, ...]
    cell_class: int
    column_id: int | None
    interval_index: int | None
    root_cell_id: int
    root_cell_kind: str
    root_vertices: tuple[int, ...]
    root_reference_vertices: tuple[tuple[Fraction, ...], ...]
    root_reference_chart_arguments: tuple[ExactReferenceChartExpression, ...]


@final
class MixedLayerColumns(StrictModule, NonTrainableState):
    """Explicit layer-cell ancestry and the permitted axial schedule changes."""

    cell_ids: Array
    column_ids: Array
    interval_indices: Array
    hard_first_thickness: bool = eqx.field(static=True)
    axial_refinement: bool = eqx.field(static=True)
    allow_schedule_change: bool = eqx.field(static=True)
    columns_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_ids: ArrayLike,
        column_ids: ArrayLike,
        interval_indices: ArrayLike,
        /,
        *,
        hard_first_thickness: bool = True,
        axial_refinement: bool = False,
        allow_schedule_change: bool = False,
    ) -> None:
        raw_cells = np.asarray(cell_ids)
        if raw_cells.ndim != 1 or raw_cells.dtype.kind not in "iu":
            raise TypeError("Layer cell IDs must be one integer vector.")
        cells = raw_cells.astype(np.int64, copy=False)
        columns = np.asarray(column_ids)
        intervals = np.asarray(interval_indices)
        if (
            columns.ndim != 1
            or columns.dtype.kind not in "iu"
            or intervals.ndim != 1
            or intervals.dtype.kind not in "iu"
        ):
            raise TypeError("Layer columns and intervals must be integer vectors.")
        columns = columns.astype(np.int64, copy=False)
        intervals = intervals.astype(np.int64, copy=False)
        if (
            cells.size == 0
            or columns.shape != cells.shape
            or intervals.shape != cells.shape
            or np.any(cells < 0)
            or np.any(columns < 0)
            or np.any(intervals < 0)
            or np.unique(cells).size != cells.size
        ):
            raise ValueError(
                "Layer columns need aligned unique cells and nonnegative ancestry."
            )
        flags = (hard_first_thickness, axial_refinement, allow_schedule_change)
        if any(not isinstance(value, (bool, np.bool_)) for value in flags):
            raise TypeError("Layer-column permissions must be bool values.")
        order = np.argsort(cells, kind="stable")
        cells, columns, intervals = cells[order], columns[order], intervals[order]
        hard_first, axial, schedule = (bool(value) for value in flags)
        self.cell_ids = jnp.asarray(cells, dtype=jnp.int64)
        self.column_ids = jnp.asarray(columns, dtype=jnp.int64)
        self.interval_indices = jnp.asarray(intervals, dtype=jnp.int64)
        self.hard_first_thickness = hard_first
        self.axial_refinement = axial
        self.allow_schedule_change = schedule
        self.columns_id = canonical_fingerprint(
            {
                "kind": "mixed-layer-columns",
                "cells": cells,
                "columns": columns,
                "intervals": intervals,
                "hard_first_thickness": hard_first,
                "axial_refinement": axial,
                "allow_schedule_change": schedule,
            }
        )


@final
class MixedAdaptationHierarchy(StrictModule, NonTrainableState):
    """Stable cell/vertex allocation and exact sibling ancestry for coarsening."""

    siblings: tuple[MixedSibling, ...] = eqx.field(static=True)
    active_cells: tuple[MixedActiveCell, ...] = eqx.field(static=True)
    forest_roots: tuple[MixedForestRoot, ...] = eqx.field(static=True)
    vertex_points: tuple[tuple[int, _PointKey], ...] = eqx.field(static=True)
    retired_entities: tuple[tuple[int, tuple[int, ...], int], ...] = eqx.field(
        static=True
    )
    periodic_quotient_history: PeriodicQuotientHistory | None
    active_lifted_topology_id: str = eqx.field(static=True)
    layer_policy: tuple[bool, bool, bool] | None = eqx.field(static=True)
    next_cell_id: int = eqx.field(static=True)
    next_vertex_id: int = eqx.field(static=True)
    hierarchy_id: str = eqx.field(static=True)

    def __init__(
        self,
        siblings: tuple[MixedSibling, ...],
        active_cells: tuple[MixedActiveCell, ...],
        forest_roots: tuple[MixedForestRoot, ...],
        vertex_points: tuple[tuple[int, _PointKey], ...],
        retired_entities: tuple[tuple[int, tuple[int, ...], int], ...],
        layer_policy: tuple[bool, bool, bool] | None,
        next_cell_id: int,
        next_vertex_id: int,
        periodic_quotient_history: PeriodicQuotientHistory | None,
        active_lifted_topology_id: str,
        /,
    ) -> None:
        ordered_cells = tuple(sorted(active_cells, key=lambda value: value.cell_id))
        ordered_forest = tuple(
            sorted(
                (
                    value._replace(
                        root_reference_chart_arguments=freeze_reference_chart_arguments(
                            (value.root_reference_chart_arguments,)
                        )[0]
                    )
                    for value in forest_roots
                ),
                key=lambda value: value.cell_id,
            )
        )
        ordered_siblings = tuple(sorted(siblings, key=lambda value: value.parent_id))
        ordered_points = tuple(sorted(vertex_points, key=lambda value: value[0]))
        ordered_retired = tuple(sorted(retired_entities))
        if (
            len({value.cell_id for value in ordered_cells}) != len(ordered_cells)
            or len({value.cell_id for value in ordered_forest}) != len(ordered_forest)
            or len({value.parent_id for value in ordered_siblings})
            != len(ordered_siblings)
            or len({value[0] for value in ordered_points}) != len(ordered_points)
            or len({(value[0], value[1]) for value in ordered_retired})
            != len(ordered_retired)
            or next_cell_id < 0
            or next_vertex_id < 0
            or not isinstance(active_lifted_topology_id, str)
            or not active_lifted_topology_id
            or (
                periodic_quotient_history is not None
                and not isinstance(periodic_quotient_history, PeriodicQuotientHistory)
            )
        ):
            raise ValueError("Mixed-adaptation hierarchy identities are malformed.")
        for root in ordered_forest:
            if (
                root.cell_id < 0
                or root.root_cell_id < 0
                or not root.block_name
                or root.cell_class < 0
                or len(root.vertices)
                != len(reference_cell_topology(root.cell_kind).vertices)
                or len(root.root_vertices)
                != len(reference_cell_topology(root.root_cell_kind).vertices)
                or any(value < 0 for value in (*root.vertices, *root.root_vertices))
                or len(set(root.vertices)) != len(root.vertices)
                or len(set(root.root_vertices)) != len(root.root_vertices)
            ):
                raise ValueError("Mixed hierarchy forest root identities are malformed.")
            validate_reference_cell_map_witnesses(
                ReferenceCellMapWitnesses(
                    np.asarray((0,), dtype=np.int64),
                    np.asarray((root.root_cell_id,), dtype=np.int64),
                    root.cell_kind,
                    root.root_cell_kind,
                    np.asarray((root.root_reference_vertices,), dtype=np.float64),
                    (root.root_reference_vertices,),
                    (root.root_reference_chart_arguments,),
                )
            )
        high_cell = max(
            (value.cell_id for value in ordered_cells),
            default=-1,
        )
        high_cell = max(
            high_cell,
            max((value.cell_id for value in ordered_forest), default=-1),
            max((value.parent_id for value in ordered_siblings), default=-1),
            max(
                (child for value in ordered_siblings for child in value.child_ids),
                default=-1,
            ),
        )
        high_vertex = max(
            max(
                (vertex for value in ordered_cells for vertex in value.vertices),
                default=-1,
            ),
            max(
                (
                    vertex
                    for value in ordered_siblings
                    for vertex in value.parent_vertices
                ),
                default=-1,
            ),
            max(
                (vertex for value in ordered_forest for vertex in value.vertices),
                default=-1,
            ),
            max((value[0] for value in ordered_points), default=-1),
        )
        if next_cell_id <= high_cell or next_vertex_id <= high_vertex:
            raise ValueError(
                "Mixed-adaptation hierarchy allocators must exceed issued IDs."
            )
        if layer_policy is not None and (
            len(layer_policy) != 3
            or any(type(value) is not bool for value in layer_policy)
        ):
            raise ValueError(
                "Mixed-adaptation layer policy must contain three bool values."
            )
        self.siblings = ordered_siblings
        self.active_cells = ordered_cells
        self.forest_roots = ordered_forest
        self.vertex_points = ordered_points
        self.retired_entities = ordered_retired
        self.periodic_quotient_history = periodic_quotient_history
        self.active_lifted_topology_id = active_lifted_topology_id
        self.layer_policy = layer_policy
        self.next_cell_id = next_cell_id
        self.next_vertex_id = next_vertex_id
        self.hierarchy_id = canonical_fingerprint(
            {
                "kind": "mixed-adaptation-hierarchy",
                "siblings": _sibling_payload(ordered_siblings),
                "active_cells": ordered_cells,
                "forest_roots": _forest_payload(ordered_forest),
                "vertex_points": _point_bank_payload(ordered_points),
                "retired_entities": ordered_retired,
                "periodic_quotient_history": (
                    None
                    if periodic_quotient_history is None
                    else periodic_quotient_history.history_id
                ),
                "active_lifted_topology": active_lifted_topology_id,
                "layer_policy": layer_policy,
                "next_cell_id": next_cell_id,
                "next_vertex_id": next_vertex_id,
            }
        )


class MixedAdaptationEvidence(NamedTuple):
    requested_refine_cell_ids: NDArray[np.int64]
    refined_cell_ids: NDArray[np.int64]
    coarsened_cell_ids: NDArray[np.int64]
    rejected_coarsening_ids: NDArray[np.int64]
    closure_cell_ids: NDArray[np.int64]
    schedule_changed_cell_ids: NDArray[np.int64]
    closure_steps: int
    interface_count: int
    work_units: int
    evidence_id: str


class MixedCellContentTransfer(NamedTuple):
    """Exact extensive-cell-content map for nested reference partitions."""

    source_cell_ids: NDArray[np.int64]
    target_cell_ids: NDArray[np.int64]
    row_source_ids: NDArray[np.int64]
    row_target_ids: NDArray[np.int64]
    exact_weights: tuple[Fraction, ...]
    weights: NDArray[np.float64]
    transfer_id: str

    def apply(self, source_cell_ids: ArrayLike, values: ArrayLike, /) -> NDArray:
        """Conservatively push per-cell extensive contents to target cell IDs."""
        identifiers = np.asarray(source_cell_ids)
        if identifiers.ndim != 1 or identifiers.dtype.kind not in "iu":
            raise TypeError("Source content cell IDs must be one integer vector.")
        identifiers = identifiers.astype(np.int64, copy=False)
        if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
            raise ValueError("Source content cell IDs must be unique and nonnegative.")
        data = np.asarray(values)
        if data.ndim == 0 or data.shape[0] != identifiers.size:
            raise ValueError("Cell contents must have one leading row per source ID.")
        order = np.argsort(identifiers, kind="stable")
        sorted_ids = identifiers[order]
        if np.unique(sorted_ids).size != sorted_ids.size:
            raise ValueError("Source content cell IDs must be unique.")
        source_rows = np.searchsorted(sorted_ids, self.row_source_ids)
        if np.any(source_rows >= sorted_ids.size) or not np.array_equal(
            sorted_ids[source_rows], self.row_source_ids
        ):
            raise ValueError("Cell-content transfer references an absent source cell.")
        target_rows = np.searchsorted(self.target_cell_ids, self.row_target_ids)
        output = np.zeros(
            (self.target_cell_ids.size, *data.shape[1:]),
            dtype=np.result_type(data.dtype, np.dtype(np.float64)),
        )
        coefficients = self.weights.reshape(
            (self.weights.size, *((1,) * (data.ndim - 1)))
        )
        np.add.at(output, target_rows, data[order[source_rows]] * coefficients)
        return output


class MixedAdaptationOutcome(NamedTuple):
    edit: CellTopologyEdit
    hierarchy: MixedAdaptationHierarchy
    evidence: MixedAdaptationEvidence
    cell_content_transfer: MixedCellContentTransfer
    coarsening_maps: tuple[MixedCoarseningMap, ...]
    reference_maps: tuple[ReferenceCellMapWitnesses, ...]
    target_geometry: CellGeometrySpec | None
    exact_vertex_coordinates: CoordinateSourceBank | None
    coarsening_partition: ReferenceCellPartition | None


class MixedTemplateClosureError(RuntimeError):
    """No family-preserving template assignment closes every shared face."""


class _Cell(NamedTuple):
    identifier: int
    name: str
    kind: str
    vertices: NDArray[np.int64]
    cell_class: int
    column_id: int | None
    interval_index: int | None


class _Template(NamedTuple):
    index: int
    children: NDArray[np.float64]


class _TemplateFacts(NamedTuple):
    points: tuple[tuple[_PointKey, ...], ...]
    facets: tuple[_FaceSignature, ...]
    split_edges: frozenset[tuple[int, int]]


class _TargetCell(NamedTuple):
    identifier: int
    name: str
    kind: str
    vertices: tuple[int, ...]
    cell_class: int
    column_id: int | None
    interval_index: int | None
    source_ids: tuple[int, ...]
    content_weights: tuple[Fraction, ...]
    reference_vertices: tuple[tuple[Fraction, ...], ...]


def _integer_vector(values: ArrayLike, name: str, /) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.size == 0:
        return np.empty((0,), dtype=np.int64)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise TypeError(f"{name} must be one integer vector.")
    result = array.astype(np.int64, copy=False)
    if np.any(result < 0) or np.unique(result).size != result.size:
        raise ValueError(f"{name} must contain unique nonnegative integers.")
    return np.sort(result)


def _immutable(values: NDArray, /) -> NDArray:
    result = np.ascontiguousarray(values)
    result.setflags(write=False)
    return result


def _fraction_payload(value: Fraction, /) -> tuple[int, int]:
    return value.numerator, value.denominator


def _point_bank_payload(
    points: tuple[tuple[int, _PointKey], ...], /
) -> tuple[tuple[int, tuple[tuple[int, int, int], ...]], ...]:
    return tuple(
        (
            identifier,
            tuple((vertex, *_fraction_payload(weight)) for vertex, weight in point),
        )
        for identifier, point in points
    )


def _sibling_payload(
    siblings: tuple[MixedSibling, ...], /
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            value.parent_id,
            value.block_name,
            value.cell_kind,
            value.parent_vertices,
            value.child_ids,
            tuple(
                tuple(
                    tuple(_fraction_payload(coordinate) for coordinate in point)
                    for point in child
                )
                for child in value.child_reference_vertices
            ),
            value.cell_class,
            value.column_id,
            value.interval_index,
        )
        for value in siblings
    )


def _forest_payload(
    roots: tuple[MixedForestRoot, ...], /
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            root.cell_id,
            root.block_name,
            root.cell_kind,
            root.vertices,
            root.cell_class,
            root.column_id,
            root.interval_index,
            root.root_cell_id,
            root.root_cell_kind,
            root.root_vertices,
            tuple(
                tuple(_fraction_payload(value) for value in point)
                for point in root.root_reference_vertices
            ),
            reference_chart_arguments_identity((root.root_reference_chart_arguments,)),
        )
        for root in roots
    )


def _point_key(
    kind: str, vertices: NDArray[np.int64], point: NDArray[np.float64], /
) -> _PointKey:
    coordinates = tuple(Fraction(float(value)) for value in point)
    x, y = coordinates[:2]
    if kind in ("tetrahedron", "simplex:3"):
        z = coordinates[2]
        weights = (1 - x - y - z, x, y, z)
    elif kind == "prism":
        z = coordinates[2]
        triangle = (1 - x - y, x, y)
        weights = tuple(value * (1 - z) for value in triangle) + tuple(
            value * z for value in triangle
        )
    elif kind == "hexahedron":
        z = coordinates[2]
        reference = reference_cell_topology(kind).vertices
        weights = tuple(
            (x if corner[0] else 1 - x)
            * (y if corner[1] else 1 - y)
            * (z if corner[2] else 1 - z)
            for corner in reference
        )
    elif kind == "pyramid":
        z = coordinates[2]
        height = 1 - z
        u, v = (
            ((x - z / 2) / height, (y - z / 2) / height)
            if height
            else (Fraction(1, 2), Fraction(1, 2))
        )
        weights = (
            height * (1 - u) * (1 - v),
            height * u * (1 - v),
            height * u * v,
            height * (1 - u) * v,
            z,
        )
    else:
        raise ValueError(f"Unsupported mixed cell family {kind!r}.")
    if len(weights) != vertices.size or sum(weights, Fraction(0)) != 1:
        raise ValueError("Reference shape functions do not form a partition of unity.")
    return tuple(
        sorted(
            (int(identifier), weight)
            for identifier, weight in zip(vertices, weights, strict=True)
            if weight
        )
    )


def _template_facts(
    cell: _Cell, template: _Template, /, *, point_cache: _LocalPointCache | None = None
) -> _TemplateFacts:
    topology = reference_cell_topology(cell.kind)

    def point_key(point: NDArray[np.float64], /) -> _PointKey:
        if point_cache is None:
            return _point_key(cell.kind, cell.vertices, point)
        key = (cell.kind, tuple(float(value) for value in point), cell.vertices.size)
        local = point_cache.get(key)
        if local is None:
            local = _point_key(
                cell.kind, np.arange(cell.vertices.size, dtype=np.int64), point
            )
            point_cache[key] = local
        return tuple(
            sorted((int(cell.vertices[index]), weight) for index, weight in local)
        )

    child_points = tuple(
        tuple(point_key(point) for point in child) for child in template.children
    )
    parent_facets = tuple(
        frozenset(int(cell.vertices[index]) for index in facet)
        for facet in topology.entities[2]
    )
    facets: list[list[tuple[_PointKey, ...]]] = [[] for _ in parent_facets]
    for points in child_points:
        for local_face in topology.entities[2]:
            face = tuple(sorted(points[index] for index in local_face))
            support = frozenset(vertex for point in face for vertex, _ in point)
            for facet, vertices in enumerate(parent_facets):
                if support.issubset(vertices):
                    facets[facet].append(face)
    local_edges = {
        tuple(sorted(int(cell.vertices[index]) for index in edge))
        for edge in topology.entities[1]
    }
    split_edges = frozenset(
        (min(first, second), max(first, second))
        for points in child_points
        for point in points
        if len(point) == 2
        for first, second in ((int(point[0][0]), int(point[1][0])),)
        if (min(first, second), max(first, second)) in local_edges
    )
    return _TemplateFacts(
        child_points,
        tuple(tuple(sorted(values)) for values in facets),
        split_edges,
    )


def _native_kind(kind: str, /) -> MixedElementKind:
    match kind:
        case "tetrahedron" | "simplex:3":
            return MixedElementKind.TETRAHEDRON
        case "hexahedron":
            return MixedElementKind.HEXAHEDRON
        case "prism":
            return MixedElementKind.PRISM
        case "pyramid":
            return MixedElementKind.PYRAMID
        case _:
            raise ValueError(f"Unsupported mixed cell family {kind!r}.")


def _templates(kind: str, axial: bool, /) -> tuple[_Template, ...]:
    native_kind = _native_kind(kind)
    count = mixed_template_counts(native_kind, axial, 0).template_count
    result = tuple(
        _Template(
            index,
            np.asarray(mixed_template(native_kind, axial, index), dtype=np.float64),
        )
        for index in range(count)
    )
    order = list(range(count))
    if kind in ("tetrahedron", "simplex:3"):
        order = [0, 8, 9, 10, 7, 1, 2, 3, 4, 5, 6]
    elif axial and kind in ("prism", "hexahedron"):
        order = [
            0,
            *sorted(
                range(1, count),
                key=lambda item: (
                    bool(np.any(np.ptp(result[item].children[..., 2], axis=1) < 1.0)),
                    item,
                ),
            ),
        ]
    return tuple(result[item] for item in order)


def _source_cells(
    mesh: CellMesh,
    classes: NDArray[np.int64],
    layer_by_id: dict[int, tuple[int, int]],
    /,
) -> list[_Cell]:
    result: list[_Cell] = []
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    row = 0
    for block in mesh.blocks:
        if not isinstance(block, CellBlock) or block.cell_kind not in (
            "tetrahedron",
            "simplex:3",
            "prism",
            "pyramid",
            "hexahedron",
        ):
            raise ValueError(
                "Mixed adaptation requires fixed tetrahedron, prism, pyramid, and hexahedron blocks."
            )
        vertices = vertex_ids[np.asarray(block.vertices, dtype=np.int64)]
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        for local_row in range(identifiers.size):
            cell_id = int(identifiers[local_row])
            layer = layer_by_id.get(cell_id)
            result.append(
                _Cell(
                    cell_id,
                    block.name,
                    block.cell_kind,
                    vertices[local_row, :].copy(),
                    int(classes[row]),
                    None if layer is None else layer[0],
                    None if layer is None else layer[1],
                )
            )
            row += 1
    if not result:
        raise ValueError("Mixed adaptation requires at least one volume cell.")
    if len({cell.identifier for cell in result}) != len(result):
        raise ValueError("Mixed cell global IDs must be unique across blocks.")
    return sorted(result, key=lambda value: value.identifier)


def _mesh_facet_owners(
    cells: list[_Cell], /
) -> dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]]:
    owners: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = defaultdict(
        list
    )
    for row, cell in enumerate(cells):
        facets = reference_cell_topology(cell.kind).entities[2]
        for facet, local in enumerate(facets):
            loop = tuple(int(cell.vertices[index]) for index in local)
            owners[tuple(sorted(loop))].append((row, facet, loop))
    if any(len(values) > 2 for values in owners.values()):
        raise MixedTemplateClosureError(
            "Mixed adaptation requires manifold shared faces."
        )
    return owners


def _periodic_seam_faces(
    mesh: CellMesh, cells: list[_Cell], /
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Return actual paired exterior faces from the source quotient authority."""
    if mesh.periodic_topology is None:
        return ()
    owners = _mesh_facet_owners(cells)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    face_rows = _entity_vertex_rows(mesh)[2]
    orbit_map = periodic_entity_orbit_map(mesh, 2)
    orbits = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, vertices in enumerate(face_rows):
        key = tuple(sorted(int(value) for value in vertex_ids[vertices]))
        face_owners = owners[key]
        if len(face_owners) == 1:
            cell, facet, _ = face_owners[0]
            groups[int(orbits[row])].append((cells[cell].identifier, facet))
    result = []
    for group in groups.values():
        if len(group) == 1:
            continue
        if len(group) != 2:
            raise MixedTemplateClosureError(
                "A periodic facet orbit must contain exactly two exterior faces."
            )
        ordered = sorted(group)
        result.append((ordered[0], ordered[1]))
    return tuple(sorted(result))


def _periodic_face_correspondence(
    mesh: CellMesh, left: tuple[int, ...], right: tuple[int, ...], /
) -> dict[int, int]:
    """Map right corners to left using only the source quotient and winding."""
    topology = mesh.periodic_topology
    if topology is None or len(left) != len(right):
        raise MixedTemplateClosureError(
            "Periodic faces have incompatible corner anatomy."
        )
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    rows = {int(identifier): row for row, identifier in enumerate(ids)}
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    roots = ids[representatives]
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    fixed = np.asarray(topology.vertex_fixed_generators, dtype=np.bool_)
    orders = identification_orders(topology.cell)

    def canonical(loop: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        indices = np.asarray([rows[vertex] for vertex in loop], dtype=np.int64)
        variants = []
        for order, _ in _orientation_orders(len(loop), simplex=False):
            key, _ = _interleaved_keys(
                roots[indices][None, :],
                shifts[indices][None, :, :],
                fixed[indices][None, :, :],
                order,
                orders,
            )
            variants.append((tuple(int(value) for value in key[0]), order))
        key, permutation = min(variants)
        if sum(candidate == key for candidate, _ in variants) != 1:
            raise MixedTemplateClosureError(
                "Periodic seam corners do not determine a unique quotient permutation."
            )
        return key, permutation

    left_key, left_order = canonical(left)
    right_key, right_order = canonical(right)
    if left_key != right_key:
        raise MixedTemplateClosureError(
            "Restored seam faces do not share one exact quotient face."
        )
    return {
        right[right_index]: left[left_index]
        for left_index, right_index in zip(left_order, right_order, strict=True)
    }


def _face_signature(
    cell: _Cell,
    template: _Template,
    facet: int,
    cache: dict[tuple[int, int], tuple[_FaceSignature, ...]],
    index: int,
    point_cache: _LocalPointCache | None = None,
    /,
) -> _FaceSignature:
    if template.index == 0:
        vertices = reference_cell_topology(cell.kind).entities[2][facet]
        return (
            tuple(
                sorted(((int(cell.vertices[index]), Fraction(1)),) for index in vertices)
            ),
        )
    key = (index, template.index)
    facets = cache.get(key)
    if facets is None:
        facets = _template_facts(cell, template, point_cache=point_cache).facets
        cache[key] = facets
    return facets[facet]


def _close_templates(
    cells: list[_Cell],
    domains: list[list[_Template]],
    interfaces: list[tuple[int, int, int, int, dict[int, int] | None]],
) -> tuple[list[_Template], int]:
    cache: dict[tuple[int, int], tuple[_FaceSignature, ...]] = {}
    point_cache: _LocalPointCache = {}
    incident: list[list[int]] = [[] for _ in cells]
    for index, (left, _, right, _, _) in enumerate(interfaces):
        incident[left].append(index)
        if right != left:
            incident[right].append(index)
    choices = [values.copy() for values in domains]
    # Keep the original branch priority while avoiding a full-domain scan on
    # every decision. Versions invalidate entries after narrowing or rollback.
    initially_has_identity = tuple(
        any(value.index == 0 for value in values) for values in domains
    )
    versions = [0] * len(choices)
    pending: list[tuple[bool, int, int, int, int]] = []
    # An identity completion is impossible while any unresolved domain lacks it.
    without_identity: set[int] = set()

    def install(index: int, values: list[_Template], /) -> None:
        choices[index] = values
        versions[index] += 1
        if len(values) > 1:
            heappush(
                pending,
                (
                    initially_has_identity[index],
                    len(values),
                    cells[index].identifier,
                    index,
                    versions[index],
                ),
            )
            if not any(value.index == 0 for value in values):
                without_identity.add(index)
            else:
                without_identity.discard(index)
        else:
            without_identity.discard(index)

    for index, values in enumerate(choices):
        install(index, values)
    trail: list[tuple[int, list[_Template]]] = []
    decisions: list[tuple[int, tuple[_Template, ...], int, int]] = []
    dirty = tuple(range(len(interfaces)))

    def next_branch() -> tuple[int, ...] | None:
        while decisions:
            selected, candidates, candidate_index, mark = decisions[-1]
            while len(trail) > mark:
                cell, previous = trail.pop()
                install(cell, previous)
            if candidate_index < len(candidates):
                decisions[-1] = (selected, candidates, candidate_index + 1, mark)
                trail.append((selected, choices[selected]))
                install(selected, [candidates[candidate_index]])
                return tuple(incident[selected])
            decisions.pop()
        return None

    steps = 0
    while True:
        steps += 1
        feasible = True
        queue = deque(dirty)
        queued = set(dirty)
        while queue and feasible:
            interface_index = queue.popleft()
            queued.remove(interface_index)
            changed_cells: list[int] = []
            left, left_facet, right, right_facet, vertex_map = interfaces[interface_index]

            def right_signature(value: _Template) -> _FaceSignature:
                signature = _face_signature(
                    cells[right], value, right_facet, cache, right, point_cache
                )
                if vertex_map is None:
                    return signature
                return tuple(
                    sorted(
                        tuple(
                            sorted(
                                tuple(
                                    sorted(
                                        (vertex_map[vertex], weight)
                                        for vertex, weight in point
                                    )
                                )
                                for point in face
                            )
                        )
                        for face in signature
                    )
                )

            if left == right:
                retained = [
                    value
                    for value in choices[left]
                    if _face_signature(
                        cells[left], value, left_facet, cache, left, point_cache
                    )
                    == right_signature(value)
                ]
                if not retained:
                    feasible = False
                    break
                if len(retained) != len(choices[left]):
                    trail.append((left, choices[left]))
                    install(left, retained)
                    changed_cells.append(left)
            else:
                right_signatures = {right_signature(value) for value in choices[right]}
                retained_left = [
                    value
                    for value in choices[left]
                    if _face_signature(
                        cells[left], value, left_facet, cache, left, point_cache
                    )
                    in right_signatures
                ]
                if not retained_left:
                    feasible = False
                    break
                if len(retained_left) != len(choices[left]):
                    trail.append((left, choices[left]))
                    install(left, retained_left)
                    changed_cells.append(left)
                left_signatures = {
                    _face_signature(
                        cells[left], value, left_facet, cache, left, point_cache
                    )
                    for value in choices[left]
                }
                retained_right = [
                    value
                    for value in choices[right]
                    if right_signature(value) in left_signatures
                ]
                if not retained_right:
                    feasible = False
                    break
                if len(retained_right) != len(choices[right]):
                    trail.append((right, choices[right]))
                    install(right, retained_right)
                    changed_cells.append(right)
            for changed_cell in changed_cells:
                for item in incident[changed_cell]:
                    if item not in queued:
                        queue.append(item)
                        queued.add(item)
        if not feasible:
            next_dirty = next_branch()
            if next_dirty is None:
                break
            dirty = next_dirty
            continue
        while pending and pending[0][4] != versions[pending[0][3]]:
            heappop(pending)
        if not pending:
            return [values[0] for values in choices], steps
        # Prove all remaining first-choice identities simultaneously against
        # every ordinary and periodic interface. This yields the same first
        # solution without one whole-mesh branch per untouched cell.
        identity_choice = (
            [
                values[0]
                if len(values) == 1
                else next((value for value in values if value.index == 0), None)
                for values in choices
            ]
            if not without_identity
            else []
        )
        if identity_choice and all(value is not None for value in identity_choice):
            identity_complete = True
            for left, left_facet, right, right_facet, vertex_map in interfaces:
                left_value = identity_choice[left]
                right_value = identity_choice[right]
                assert left_value is not None and right_value is not None
                left_trace = _face_signature(
                    cells[left], left_value, left_facet, cache, left, point_cache
                )
                right_trace = _face_signature(
                    cells[right], right_value, right_facet, cache, right, point_cache
                )
                if vertex_map is not None:
                    right_trace = tuple(
                        sorted(
                            tuple(
                                sorted(
                                    tuple(
                                        sorted(
                                            (vertex_map[vertex], weight)
                                            for vertex, weight in point
                                        )
                                    )
                                    for point in face
                                )
                            )
                            for face in right_trace
                        )
                    )
                if left_trace != right_trace:
                    identity_complete = False
                    break
            if identity_complete:
                return [value for value in identity_choice if value is not None], steps
        # Marked cells (identity excluded) are decided first in their preferred
        # template order; closure cells then follow their constrained faces.
        selected = pending[0][3]
        candidates = tuple(choices[selected])
        decisions.append((selected, candidates, 1, len(trail)))
        trail.append((selected, choices[selected]))
        install(selected, [candidates[0]])
        dirty = tuple(incident[selected])
    raise MixedTemplateClosureError(
        "No family-preserving reference templates close the marked shared-face signatures."
    )


class _RootCellChart(NamedTuple):
    root_cell_id: int
    root_cell_kind: str
    root_vertices: tuple[int, ...]
    arguments: tuple[ExactReferenceChartExpression, ...]


def _root_point_from_chart(
    chart: _RootCellChart, target_kind: str, point: tuple[Fraction, ...], /
) -> _PointKey:
    coordinates = evaluate_reference_chart_arguments(target_kind, chart.arguments, point)
    if any(not isinstance(value, Fraction) for value in coordinates):
        raise ValueError("Mixed root reference coordinates must be exact rationals.")
    reference = tuple(value for value in coordinates if isinstance(value, Fraction))
    weights = _reference_shape_weights(chart.root_cell_kind, reference)
    if len(weights) != len(chart.root_vertices):
        raise ValueError("Mixed root chart and root-cell corners do not align.")
    return tuple(
        sorted(
            (int(identifier), weight)
            for identifier, weight in zip(chart.root_vertices, weights, strict=True)
            if weight
        )
    )


def _source_root_charts(
    mesh: CellMesh,
    geometry: CellGeometrySpec | None,
    hierarchy: MixedAdaptationHierarchy | None,
    root_mesh: CellMesh | None,
    /,
) -> dict[int, _RootCellChart]:
    ancestry = (
        None
        if geometry is None
        or (
            hierarchy is None
            and isinstance(root_mesh, CellMesh)
            and root_mesh.mesh_id == mesh.mesh_id
        )
        else geometry.restriction_source
    )
    if ancestry is not None and hierarchy is None and root_mesh is None:
        raise ValueError(
            "Starting mixed adaptation of a restricted source requires its exact root mesh."
        )
    if root_mesh is not None:
        if not isinstance(root_mesh, CellMesh):
            raise TypeError("Mixed root_mesh must be CellMesh or None.")
        expected = (
            mesh.topology_id
            if ancestry is None and hierarchy is None
            else (None if ancestry is None else ancestry.source_topology_id)
        )
        if expected is not None and root_mesh.topology_id != expected:
            raise ValueError("Mixed root mesh does not match the retained root topology.")
    source_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if ancestry is None and hierarchy is None:
        return {
            int(identifier): _RootCellChart(
                int(identifier),
                block.cell_kind,
                tuple(
                    int(source_vertex_ids[index])
                    for index in np.asarray(block.vertices, dtype=np.int64)[row, :]
                ),
                freeze_reference_chart_arguments(
                    (axes(reference_cell_topology(block.cell_kind).dimension),)
                )[0],
            )
            for block in mesh.blocks
            for row, (identifier, _) in enumerate(
                zip(
                    np.asarray(block.global_ids, dtype=np.int64),
                    np.asarray(block.vertices, dtype=np.int64),
                    strict=True,
                )
            )
        }
    if ancestry is None:
        if geometry is not None:
            raise ValueError("Mixed geometry lost its retained root restriction.")
        if hierarchy is None:
            raise RuntimeError("Mixed root chart hierarchy was not initialized.")
        charts = {
            root.cell_id: _RootCellChart(
                root.root_cell_id,
                root.root_cell_kind,
                root.root_vertices,
                root.root_reference_chart_arguments,
            )
            for root in hierarchy.forest_roots
        }
        pending = list(hierarchy.siblings)
        while pending:
            remaining = []
            for sibling in pending:
                parent = charts.get(sibling.parent_id)
                if parent is None:
                    remaining.append(sibling)
                    continue
                for child, corners in zip(
                    sibling.child_ids, sibling.child_reference_vertices, strict=True
                ):
                    child_chart = _affine_chart(
                        *_exact_child_to_parent_affine_map(sibling.cell_kind, corners)
                    )
                    arguments = freeze_reference_chart_arguments(
                        (
                            tuple(
                                compose_expression(value, child_chart)
                                for value in parent.arguments
                            ),
                        )
                    )[0]
                    charts[child] = _RootCellChart(
                        parent.root_cell_id,
                        parent.root_cell_kind,
                        parent.root_vertices,
                        arguments,
                    )
            if len(remaining) == len(pending):
                raise ValueError("Mixed hierarchy forest has unresolved parent ancestry.")
            pending = remaining
        return charts
    if set(ancestry.block_names) != {block.name for block in mesh.blocks}:
        raise ValueError("Mixed root ancestry must cover every source block.")
    root_cells: dict[int, tuple[str, tuple[int, ...]]] = {}
    if root_mesh is not None:
        root_ids = np.asarray(root_mesh.vertex_global_ids, dtype=np.int64)
        for block in root_mesh.blocks:
            for row, (identifier, _) in enumerate(
                zip(
                    np.asarray(block.global_ids, dtype=np.int64),
                    np.asarray(block.vertices, dtype=np.int64),
                    strict=True,
                )
            ):
                root_cells[int(identifier)] = (
                    block.cell_kind,
                    tuple(
                        int(root_ids[index])
                        for index in np.asarray(block.vertices, dtype=np.int64)[row, :]
                    ),
                )
    elif hierarchy is not None:
        for root in hierarchy.forest_roots:
            value = (root.root_cell_kind, root.root_vertices)
            previous = root_cells.setdefault(root.root_cell_id, value)
            if previous != value:
                raise ValueError("Mixed forest has conflicting root-cell corners.")
    maps = dict(zip(ancestry.block_names, ancestry.parent_reference_maps, strict=True))
    supports = dict(zip(ancestry.block_names, ancestry.parent_vertex_ids, strict=True))
    result: dict[int, _RootCellChart] = {}
    for block in mesh.blocks:
        witness = maps[block.name]
        if witness is None:
            raise ValueError("Mixed root ancestry requires exact reference maps.")
        rows = nested_cell_reference_maps(witness)
        if len(rows) != block.cell_count:
            raise ValueError("Mixed root chart rows do not cover the source block.")
        parent_vertices = np.asarray(supports[block.name], dtype=np.int64)
        for index, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            reference_map = rows[index]
            if reference_map.arguments is None or reference_map.target_cell_id != index:
                raise ValueError("Mixed root chart rows do not align with source cells.")
            declared = tuple(
                int(value) for value in parent_vertices[index, :] if value >= 0
            )
            actual = root_cells.get(reference_map.source_cell_id)
            if actual is not None:
                if actual[0] != reference_map.source_cell_kind:
                    raise ValueError("Mixed root chart names a mismatched root cell.")
                root_vertices = actual[1]
                if declared and not set(declared).issubset(root_vertices):
                    raise ValueError("Mixed root support disagrees with the root mesh.")
            else:
                if root_mesh is not None:
                    raise ValueError("Mixed root chart names an absent root cell.")
                root_vertices = declared
                if len(root_vertices) != len(
                    reference_cell_topology(reference_map.source_cell_kind).vertices
                ):
                    raise ValueError(
                        "Mixed root mesh is required when root corners are not retained."
                    )
            result[int(identifier)] = _RootCellChart(
                reference_map.source_cell_id,
                reference_map.source_cell_kind,
                root_vertices,
                reference_map.arguments,
            )
    return result


def _chart_vertex_points(
    mesh: CellMesh, charts: dict[int, _RootCellChart], /
) -> dict[int, _PointKey]:
    points: dict[int, _PointKey] = {}
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    for block in mesh.blocks:
        corners = tuple(
            tuple(Fraction(value) for value in point)
            for point in reference_cell_topology(block.cell_kind).vertices
        )
        for row, (identifier, _) in enumerate(
            zip(
                np.asarray(block.global_ids, dtype=np.int64),
                np.asarray(block.vertices, dtype=np.int64),
                strict=True,
            )
        ):
            vertices = np.asarray(block.vertices, dtype=np.int64)[row, :]
            chart = charts.get(int(identifier))
            if chart is None:
                raise ValueError("Mixed root charts omit a source cell.")
            for vertex, corner in zip(vertices, corners, strict=True):
                vertex_id = int(ids[vertex])
                point = _root_point_from_chart(chart, block.cell_kind, corner)
                previous = points.get(vertex_id)
                if previous is not None and previous != point:
                    raise ValueError(
                        "Mixed incident root charts disagree at a source vertex."
                    )
                points[vertex_id] = point
    if set(points) != set(int(value) for value in ids):
        raise ValueError("Mixed root charts do not cover every source vertex.")
    return points


def _mapping_points(
    hierarchy: MixedAdaptationHierarchy | None,
    mesh: CellMesh,
    charts: dict[int, _RootCellChart],
    root_mesh: CellMesh | None,
    /,
) -> dict[int, _PointKey]:
    current = _chart_vertex_points(mesh, charts)
    points = {} if hierarchy is None else dict(hierarchy.vertex_points)
    root_vertices = {
        vertex for chart in charts.values() for vertex in chart.root_vertices
    }
    if root_mesh is not None:
        root_vertices.update(
            int(value) for value in np.asarray(root_mesh.vertex_global_ids)
        )
    for vertex in root_vertices:
        identity = ((vertex, Fraction(1)),)
        previous = points.setdefault(vertex, identity)
        if previous != identity:
            raise ValueError("A root vertex ID has conflicting mixed ancestry.")
    for identifier, point in current.items():
        previous = points.setdefault(identifier, point)
        if previous != point:
            raise ValueError(
                "Mixed hierarchy vertex roots disagree with exact source charts."
            )
    return points


def _forest_roots(
    cells: list[_Cell], charts: dict[int, _RootCellChart], /
) -> tuple[MixedForestRoot, ...]:
    result = []
    for cell in cells:
        chart = charts[cell.identifier]
        corners = tuple(
            tuple(Fraction(value) for value in point)
            for point in reference_cell_topology(cell.kind).vertices
        )
        exact = tuple(
            evaluate_reference_chart_arguments(cell.kind, chart.arguments, point)
            for point in corners
        )
        if any(not isinstance(value, Fraction) for point in exact for value in point):
            raise ValueError("Mixed forest corners must be exact rationals.")
        result.append(
            MixedForestRoot(
                cell.identifier,
                cell.name,
                cell.kind,
                tuple(int(value) for value in cell.vertices),
                cell.cell_class,
                cell.column_id,
                cell.interval_index,
                chart.root_cell_id,
                chart.root_cell_kind,
                chart.root_vertices,
                tuple(
                    tuple(value for value in point if isinstance(value, Fraction))
                    for point in exact
                ),
                chart.arguments,
            )
        )
    return tuple(result)


def _target_root_points(
    targets: list[_TargetCell],
    charts: dict[int, _RootCellChart],
    coarsened_parent_ids: set[int],
    source_points: dict[int, _PointKey],
    /,
) -> dict[int, _PointKey]:
    result: dict[int, _PointKey] = {}
    for target in targets:
        if target.identifier in coarsened_parent_ids:
            values = tuple(source_points[vertex] for vertex in target.vertices)
        else:
            if len(target.source_ids) != 1 or len(target.vertices) != len(
                target.reference_vertices
            ):
                raise ValueError("Mixed target has incomplete exact reference ancestry.")
            chart = charts[target.source_ids[0]]
            values = tuple(
                _root_point_from_chart(chart, target.kind, point)
                for point in target.reference_vertices
            )
        for vertex, point in zip(target.vertices, values, strict=True):
            previous = result.setdefault(vertex, point)
            if previous != point:
                raise ValueError(
                    "Mixed incident root charts disagree at a target vertex."
                )
    return result


def _coarsening_candidates(
    cells: list[_Cell],
    marks: set[int],
    hierarchy: MixedAdaptationHierarchy | None,
    protected_vertices: set[int],
    protected_cells: set[int],
    root_points: dict[int, _PointKey],
    /,
) -> tuple[list[MixedSibling], set[int], set[int]]:
    if not marks:
        return [], set(), set()
    if hierarchy is None:
        raise ValueError("Coarsening mixed cells requires their prior hierarchy.")
    active = {cell.identifier: cell for cell in cells}
    candidates: list[MixedSibling] = []
    rejected: set[int] = set()
    known_children: set[int] = set()
    for record in hierarchy.siblings:
        members = set(record.child_ids)
        known_children.update(members)
        marked = members & marks
        if not marked:
            continue
        if (
            not members.issubset(marks)
            or not members.issubset(active)
            or any(active[child].cell_class != record.cell_class for child in members)
            or members & protected_cells
            or (
                set().union(*(set(active[child].vertices) for child in members))
                - set(record.parent_vertices)
            )
            & protected_vertices
        ):
            rejected.update(marked)
            continue
        candidates.append(record)
    rejected.update(marks - known_children)
    accepted = {record.parent_id: record for record in candidates}
    changed = True
    while changed:
        changed = False
        removed_children = {
            child for record in accepted.values() for child in record.child_ids
        }
        faces = _mesh_facet_owners(cells)
        reject_parents: set[int] = set()
        for record in accepted.values():
            parent_points = tuple(root_points[value] for value in record.parent_vertices)
            parent_facets = reference_cell_topology(record.cell_kind).entities[2]
            coarse_faces = tuple(
                tuple(sorted(parent_points[index] for index in local))
                for local in parent_facets
            )
            coarse_supports = tuple(
                frozenset(root for point in face for root, _ in point)
                for face in coarse_faces
            )
            for owners in faces.values():
                if len(owners) != 2:
                    continue
                (first, _, _), (second, _, _) = owners
                first_id, second_id = cells[first].identifier, cells[second].identifier
                if (first_id in record.child_ids) == (second_id in record.child_ids):
                    continue
                neighbor_id = second_id if first_id in record.child_ids else first_id
                if neighbor_id in removed_children:
                    continue
                loop = owners[0][2] if first_id in record.child_ids else owners[1][2]
                fine_face = tuple(sorted(root_points[value] for value in loop))
                fine_support = frozenset(root for point in fine_face for root, _ in point)
                if any(
                    fine_support.issubset(coarse_support) and fine_face != coarse_face
                    for coarse_face, coarse_support in zip(
                        coarse_faces, coarse_supports, strict=True
                    )
                ):
                    reject_parents.add(record.parent_id)
                    break
        if reject_parents:
            for parent in reject_parents:
                record = accepted.pop(parent)
                rejected.update(set(record.child_ids) & marks)
            changed = True
    accepted_children = {
        child for record in accepted.values() for child in record.child_ids
    }
    rejected.update(marks - accepted_children)
    return list(accepted.values()), rejected, accepted_children


def _periodic_coarsening_candidates(
    cells: list[_Cell],
    marks: set[int],
    hierarchy: MixedAdaptationHierarchy | None,
    protected_vertices: set[int],
    protected_cells: set[int],
    root_points: dict[int, _PointKey],
    seams: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    /,
) -> tuple[list[MixedSibling], set[int], set[int]]:
    if not marks or hierarchy is None:
        return _coarsening_candidates(
            cells, marks, hierarchy, protected_vertices, protected_cells, root_points
        )
    active = {cell.identifier for cell in cells}
    sibling_by_child = {
        child: record for record in hierarchy.siblings for child in record.child_ids
    }
    closed = set(marks)
    while True:
        before = len(closed)
        for child in tuple(closed):
            record = sibling_by_child.get(child)
            if record is not None and set(record.child_ids).issubset(active):
                closed.update(record.child_ids)
        for (first, _), (second, _) in seams:
            if first in closed or second in closed:
                closed.update((first, second))
        if len(closed) == before:
            break
    permitted = set(closed)
    rejected: set[int] = set()
    while True:
        candidates, local_rejected, removed = _coarsening_candidates(
            cells,
            permitted,
            hierarchy,
            protected_vertices,
            protected_cells,
            root_points,
        )
        rejected.update(local_rejected)
        by_child = {child: record for record in candidates for child in record.child_ids}
        reject_parents = {
            by_child[side].parent_id
            for (first, _), (second, _) in seams
            if (first in removed) != (second in removed)
            for side in (first, second)
            if side in by_child
        }
        if not reject_parents:
            return candidates, rejected, removed
        discarded = {
            child
            for record in candidates
            if record.parent_id in reject_parents
            for child in record.child_ids
        }
        permitted.difference_update(discarded)
        rejected.update(discarded & closed)


def _periodic_working_interfaces(
    mesh: CellMesh,
    source_cells: list[_Cell],
    working: list[_Cell],
    candidates: list[MixedSibling],
    seams: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    /,
) -> list[tuple[int, int, int, int, dict[int, int] | None]]:
    source_by_id = {cell.identifier: cell for cell in source_cells}
    working_rows = {cell.identifier: row for row, cell in enumerate(working)}
    children = {
        child: (record, index)
        for record in candidates
        for index, child in enumerate(record.child_ids)
    }

    def working_face(face: tuple[int, int]) -> tuple[int, int]:
        cell_id, facet = face
        child = children.get(cell_id)
        if child is None:
            return working_rows[cell_id], facet
        record, child_index = child
        source = source_by_id[cell_id]
        local_face = reference_cell_topology(source.kind).entities[2][facet]
        parent_vertices = np.asarray(record.parent_vertices, dtype=np.int64)
        reference = record.child_reference_vertices[child_index]
        support = {
            vertex
            for local in local_face
            for vertex, _ in _point_key(
                record.cell_kind,
                parent_vertices,
                np.asarray(reference[local], dtype=np.float64),
            )
        }
        parent_faces = reference_cell_topology(record.cell_kind).entities[2]
        matches = [
            index
            for index, local in enumerate(parent_faces)
            if support.issubset({int(parent_vertices[value]) for value in local})
        ]
        if len(matches) != 1:
            raise MixedTemplateClosureError(
                "A periodic child seam does not descend from one parent face."
            )
        return working_rows[record.parent_id], matches[0]

    interfaces: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[int, int, int, int, dict[int, int] | None],
    ] = {}
    partners: dict[tuple[int, int], tuple[int, int]] = {}
    for first, second in seams:
        left = working_face(first)
        right = working_face(second)
        if left == right:
            raise MixedTemplateClosureError(
                "A periodic seam cannot identify one face to itself."
            )
        for face, partner in ((left, right), (right, left)):
            if face in partners and partners[face] != partner:
                raise MixedTemplateClosureError(
                    "Periodic coarsening must restore complete matching seam faces."
                )
            partners[face] = partner
        if right < left:
            left, right = right, left
        pair = (left, right)
        if pair in interfaces:
            continue
        left_cell, right_cell = working[left[0]], working[right[0]]
        left_loop = tuple(
            int(left_cell.vertices[index])
            for index in reference_cell_topology(left_cell.kind).entities[2][left[1]]
        )
        right_loop = tuple(
            int(right_cell.vertices[index])
            for index in reference_cell_topology(right_cell.kind).entities[2][right[1]]
        )
        correspondence = _periodic_face_correspondence(mesh, left_loop, right_loop)
        interfaces[pair] = (left[0], left[1], right[0], right[1], correspondence)
    return [interfaces[key] for key in sorted(interfaces)]


def _point_in_root_hull(point: _PointKey, vertices: tuple[_PointKey, ...], /) -> bool:
    """Decide exact convex containment in retained root-barycentric coordinates."""
    if point in vertices:
        return True
    point_values = dict(point)
    roots = sorted({key for vertex in (*vertices, point) for key, _ in vertex})
    vertex_values = tuple(dict(vertex) for vertex in vertices)
    for count in range(1, len(vertices) + 1):
        for selected in combinations(range(len(vertices)), count):
            matrix = [
                {
                    column: vertex_values[index][root]
                    for column, index in enumerate(selected)
                    if root in vertex_values[index]
                }
                for root in roots
            ]
            matrix.append({column: Fraction(1) for column in range(count)})
            right = [[point_values.get(root, Fraction(0))] for root in roots]
            right.append([Fraction(1)])
            try:
                coefficients, _ = solve_exact_sparse(matrix, right, count)
            except (ExactInconsistentSystemError, ExactRankDeficientSystemError):
                continue
            if all(row[0] >= 0 for row in coefficients):
                return True
    return False


def _entity_relations(
    source: CellMesh,
    probe: CellMesh,
    targets: list[_TargetCell],
    coarsen_sources: dict[int, tuple[int, ...]],
    coarsening_maps: tuple[MixedCoarseningMap, ...],
    /,
) -> tuple[EntityRelations, ...]:
    dimension = source.topological_dimension
    relations: list[EntityRelations] = []
    contained = cache(_point_in_root_hull)
    source_vertices_by_id = {}
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    for block in source.blocks:
        for row, (identifier, _) in enumerate(
            zip(
                np.asarray(block.global_ids, dtype=np.int64),
                np.asarray(block.vertices, dtype=np.int64),
                strict=True,
            )
        ):
            vertices = np.asarray(block.vertices, dtype=np.int64)[row, :]
            source_vertices_by_id[int(identifier)] = (
                block.cell_kind,
                tuple(int(source_vertex_ids[index]) for index in vertices),
            )
    coarsening_by_parent = {record.parent_cell_id: record for record in coarsening_maps}

    def coordinate_key(point: tuple[Fraction, ...], /) -> _PointKey:
        return tuple((axis, value) for axis, value in enumerate(point) if value)

    for degree in range(dimension + 1):
        if degree == 0:
            relations.append(
                EntityRelations(
                    degree,
                    np.empty((0, 1), dtype=np.int64),
                    np.empty((0, 1), dtype=np.int64),
                    np.empty((0,), dtype=np.int32),
                )
            )
            continue
        if degree == dimension:
            source_ids: list[int] = []
            target_ids: list[int] = []
            kinds: list[int] = []
            for target in targets:
                if target.source_ids == (target.identifier,):
                    source_ids.append(target.identifier)
                    target_ids.append(target.identifier)
                    kinds.append(int(EntityLineageKind.PRESERVED))
                elif target.identifier in coarsen_sources:
                    for source_id in coarsen_sources[target.identifier]:
                        source_ids.append(source_id)
                        target_ids.append(target.identifier)
                        kinds.append(int(EntityLineageKind.COARSENED_INTO))
                else:
                    source_ids.append(target.source_ids[0])
                    target_ids.append(target.identifier)
                    kinds.append(int(EntityLineageKind.REFINED_FROM))
            relations.append(
                EntityRelations(
                    degree,
                    np.asarray(source_ids, dtype=np.int64)[:, None],
                    np.asarray(target_ids, dtype=np.int64)[:, None],
                    np.asarray(kinds, dtype=np.int32),
                )
            )
            continue
        source_keys = entity_keys(source, degree)
        target_keys = entity_keys(probe, degree)
        entries_set: set[tuple[tuple[int, ...], tuple[int, ...], int]] = set()
        for target in targets:
            target_topology = reference_cell_topology(target.kind)
            target_entities = target_topology.entities[degree]
            target_corners = tuple(
                tuple(Fraction(value) for value in point)
                for point in target_topology.vertices
            )
            target_keys_by_local = tuple(
                tuple(sorted(target.vertices[index] for index in entity))
                for entity in target_entities
            )
            record = coarsening_by_parent.get(target.identifier)
            if record is None:
                if len(target.source_ids) != 1:
                    raise ValueError("Mixed target facet lacks one source parent.")
                source_record = source_vertices_by_id.get(target.source_ids[0])
                if source_record is None or source_record[0] != target.kind:
                    raise ValueError("Mixed target facet has a mismatched source family.")
                source_topology = reference_cell_topology(source_record[0])
                source_corners = tuple(
                    tuple(Fraction(value) for value in point)
                    for point in source_topology.vertices
                )
                if len(target.reference_vertices) != len(target.vertices):
                    raise ValueError(
                        "Mixed target facet has incomplete reference corners."
                    )
                # An unchanged cell with the identity reference map has the
                # same entity keys. Distinct equal-dimensional reference
                # entities cannot contain one another, so it contributes no
                # refined relation rows.
                if (
                    target.source_ids == (target.identifier,)
                    and target.vertices == source_record[1]
                    and target.reference_vertices == source_corners
                ):
                    continue
                for local_target, target_key in zip(
                    target_entities, target_keys_by_local, strict=True
                ):
                    mapped = tuple(
                        coordinate_key(target.reference_vertices[index])
                        for index in local_target
                    )
                    for local_source in source_topology.entities[degree]:
                        source_key = tuple(
                            sorted(source_record[1][index] for index in local_source)
                        )
                        hull = tuple(
                            coordinate_key(source_corners[index])
                            for index in local_source
                        )
                        if source_key != target_key and all(
                            contained(point, hull) for point in mapped
                        ):
                            entries_set.add(
                                (
                                    source_key,
                                    target_key,
                                    int(EntityLineageKind.REFINED_FROM),
                                )
                            )
            else:
                children = tuple(int(value) for value in record.child_cell_ids)
                if (
                    record.parent_cell_kind != target.kind
                    or children != coarsen_sources.get(target.identifier)
                    or len(children) != len(record.child_reference_vertices)
                ):
                    raise ValueError("Mixed coarsening facet partition is inconsistent.")
                for child_id, child_corners in zip(
                    children, record.child_reference_vertices, strict=True
                ):
                    source_record = source_vertices_by_id.get(child_id)
                    if source_record is None or source_record[0] != target.kind:
                        raise ValueError("Mixed coarsening facet has a mismatched child.")
                    source_topology = reference_cell_topology(source_record[0])
                    for local_source in source_topology.entities[degree]:
                        source_key = tuple(
                            sorted(source_record[1][index] for index in local_source)
                        )
                        mapped = tuple(
                            coordinate_key(child_corners[index]) for index in local_source
                        )
                        for local_target, target_key in zip(
                            target_entities, target_keys_by_local, strict=True
                        ):
                            hull = tuple(
                                coordinate_key(target_corners[index])
                                for index in local_target
                            )
                            if source_key != target_key and all(
                                contained(point, hull) for point in mapped
                            ):
                                entries_set.add(
                                    (
                                        source_key,
                                        target_key,
                                        int(EntityLineageKind.COARSENED_INTO),
                                    )
                                )
        entries = sorted(entries_set)
        width = max(source_keys.shape[1], target_keys.shape[1], degree + 1)
        source_rows = np.full((len(entries), width), -1, dtype=np.int64)
        target_rows = np.full((len(entries), width), -1, dtype=np.int64)
        for index, (source_key, target_key, _) in enumerate(entries):
            source_rows[index, : len(source_key)] = source_key
            target_rows[index, : len(target_key)] = target_key
        relations.append(
            EntityRelations(
                degree,
                source_rows,
                target_rows,
                np.asarray([value[2] for value in entries], dtype=np.int32),
            )
        )
    return tuple(relations)


def _shared_faces(
    blocks: tuple[TopologyEditBlock, ...], vertex_ids: NDArray[np.int64], /
) -> SharedFaceWitnesses:
    owners: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = defaultdict(
        list
    )
    for block in blocks:
        topology = reference_cell_topology(block.cell_kind)
        for identifier, row in zip(block.cell_ids, block.cells, strict=True):
            for facet, local in enumerate(topology.entities[2]):
                loop = tuple(int(vertex_ids[row[index]]) for index in local)
                owners[tuple(sorted(loop))].append((int(identifier), facet, loop))
    pairs = [values for values in owners.values() if len(values) == 2]
    if any(len(values) > 2 for values in owners.values()):
        raise MixedTemplateClosureError("The target contains a nonmanifold shared face.")
    permutations = np.full((len(pairs), 4), -1, dtype=np.int32)
    for row, (left, right) in enumerate(pairs):
        action = facet_orientation_between(left[2], right[2])
        permutations[row, : len(action.permutation)] = action.permutation
    return SharedFaceWitnesses(
        np.asarray([value[0][0] for value in pairs], dtype=np.int64),
        np.asarray([value[0][1] for value in pairs], dtype=np.int32),
        np.asarray([value[1][0] for value in pairs], dtype=np.int64),
        np.asarray([value[1][1] for value in pairs], dtype=np.int32),
        permutations,
    )


def _protected_keys(
    mesh: CellMesh, values: ArrayLike | None, degree: int, name: str, /
) -> set[tuple[int, ...]]:
    if values is None:
        return set()
    raw = np.asarray(values)
    if raw.size == 0:
        return set()
    if raw.ndim != 2 or raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer key table.")
    keys = {tuple(sorted(int(value) for value in row if value >= 0)) for row in raw}
    known = {
        tuple(int(value) for value in row if value >= 0)
        for row in entity_keys(mesh, degree)
    }
    if any(not key for key in keys) or not keys.issubset(known):
        raise ValueError(f"{name} must name current mesh entities.")
    return keys


def _protected_id_set(values: ArrayLike | None, name: str, /) -> set[int]:
    if values is None:
        return set()
    return {int(value) for value in _integer_vector(values, name)}


def _validate_hierarchy(
    hierarchy: MixedAdaptationHierarchy,
    mesh: CellMesh,
    cells: list[_Cell],
    vertex_points: dict[int, _PointKey],
    /,
) -> None:
    if mesh.lifted_topology_id != hierarchy.active_lifted_topology_id:
        raise ValueError("Mixed-adaptation hierarchy belongs to another lifted topology.")
    history = hierarchy.periodic_quotient_history
    if (history is not None) != (mesh.periodic_topology is not None):
        raise ValueError("Mixed-adaptation hierarchy periodic ancestry is inconsistent.")
    if (
        mesh.periodic_topology is not None
        and history is not None
        and history.identification_id != identification_id(mesh.periodic_topology.cell)
    ):
        raise ValueError("Mixed-adaptation history uses another periodic identification.")
    active = tuple(
        MixedActiveCell(
            cell.identifier,
            cell.name,
            cell.kind,
            tuple(int(value) for value in cell.vertices),
            cell.cell_class,
            cell.column_id,
            cell.interval_index,
        )
        for cell in cells
    )
    if active != hierarchy.active_cells:
        raise ValueError(
            "Mixed-adaptation hierarchy is not bound to this exact active mesh."
        )
    if not set(vertex_points).issuperset(
        int(value)
        for value in np.asarray([vertex for cell in cells for vertex in cell.vertices])
    ):
        raise ValueError("Mixed-adaptation hierarchy lacks source-vertex ancestry.")


def _template_domain(
    cell: _Cell,
    *,
    axial: bool,
    refine: bool,
    blocked: bool,
    protected_edges: set[tuple[int, ...]],
    protected_faces: set[tuple[int, ...]],
) -> list[_Template]:
    templates = _templates(cell.kind, axial)
    identity = templates[0]
    if blocked:
        return [identity]
    candidates = templates[1:] if refine else list(templates)
    if not candidates:
        return []
    topology = reference_cell_topology(cell.kind)
    protected_facets = {
        facet
        for facet, local in enumerate(topology.entities[2])
        if tuple(sorted(int(cell.vertices[index]) for index in local)) in protected_faces
    }
    identity_facts = _template_facts(cell, identity)
    accepted: list[_Template] = []
    for template in candidates:
        facts = _template_facts(cell, template)
        if any(tuple(sorted(edge)) in protected_edges for edge in facts.split_edges):
            continue
        if any(
            facts.facets[facet] != identity_facts.facets[facet]
            for facet in protected_facets
        ):
            continue
        accepted.append(template)
    return accepted


def _target_edit(
    mesh: CellMesh,
    target_cells: list[_TargetCell],
    target_stencils: dict[int, _PointKey],
    coordinates_by_id: dict[int, NDArray[np.float64]],
    prescribed: tuple[PrescribedEntityIds, ...],
    /,
) -> CellTopologyEdit:
    vertex_ids = np.asarray(
        sorted({value for cell in target_cells for value in cell.vertices}),
        dtype=np.int64,
    )
    vertex_rows = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    coordinates = np.asarray(
        [coordinates_by_id[int(identifier)] for identifier in vertex_ids],
        dtype=np.float64,
    )
    width = max(
        (len(target_stencils[int(identifier)]) for identifier in vertex_ids), default=1
    )
    sources = np.full((vertex_ids.size, width), -1, dtype=np.int64)
    weights = np.zeros((vertex_ids.size, width), dtype=np.float64)
    valid = np.zeros((vertex_ids.size, width), dtype=np.bool_)
    exact_rows: list[tuple[Fraction, ...]] = []
    for row, identifier in enumerate(vertex_ids):
        point = target_stencils[int(identifier)]
        exact = [Fraction(0)] * width
        for column, (source_id, coefficient) in enumerate(point):
            sources[row, column] = source_id
            weights[row, column] = float(coefficient)
            valid[row, column] = True
            exact[column] = coefficient
        exact_rows.append(tuple(exact))
    families = tuple((block.name, block.cell_kind) for block in mesh.blocks)
    blocks: list[TopologyEditBlock] = []
    for name, kind in families:
        selected = [
            cell for cell in target_cells if cell.name == name and cell.kind == kind
        ]
        if selected:
            selected.sort(key=lambda value: value.identifier)
            blocks.append(
                TopologyEditBlock(
                    name,
                    kind,
                    kind,
                    np.asarray(
                        [
                            [vertex_rows[value] for value in cell.vertices]
                            for cell in selected
                        ],
                        dtype=np.int32,
                    ),
                    np.asarray([cell.identifier for cell in selected], dtype=np.int64),
                )
            )
    if not blocks:
        raise ValueError("Mixed adaptation cannot remove every source cell.")
    empty_relations = tuple(
        EntityRelations(
            degree,
            np.empty((0, 1 if degree in (0, 3) else degree + 1), dtype=np.int64),
            np.empty((0, 1 if degree in (0, 3) else degree + 1), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
        )
        for degree in range(4)
    )
    shell = CellTopologyEdit(
        coordinates,
        vertex_ids,
        tuple(blocks),
        sources,
        weights,
        valid,
        empty_relations,
        prescribed_entity_ids=prescribed,
        shared_faces=_shared_faces(tuple(blocks), vertex_ids),
        stencil_exact_weights=tuple(exact_rows),
    )
    return shell


def _retired_entity_records(
    source: CellMesh,
    target: CellMesh,
    prior: tuple[tuple[int, tuple[int, ...], int], ...],
    /,
) -> tuple[tuple[int, tuple[int, ...], int], ...]:
    restored: set[tuple[int, tuple[int, ...]]] = set()
    retained: list[tuple[int, tuple[int, ...], int]] = []
    for degree, key, identifier in prior:
        target_keys = entity_keys(target, degree)
        if key_rows(target_keys, np.asarray([key], dtype=np.int64))[0] >= 0:
            restored.add((degree, key))
        else:
            retained.append((degree, key, identifier))
    for degree in (1, 2):
        source_keys = entity_keys(source, degree)
        target_keys = entity_keys(target, degree)
        ids = np.asarray(source.entity_set(degree).entity_ids, dtype=np.int64)
        rows = key_rows(target_keys, source_keys)
        retained.extend(
            (degree, tuple(int(value) for value in key if value >= 0), int(identifier))
            for key, identifier, row in zip(source_keys, ids, rows, strict=True)
            if row < 0
        )
    unique = {
        (degree, key): identifier
        for degree, key, identifier in retained
        if (degree, key) not in restored
    }
    return tuple(
        sorted((degree, key, identifier) for (degree, key), identifier in unique.items())
    )


def _initial_root_retired_entities(
    source: CellMesh, root: CellMesh | None, /
) -> tuple[tuple[int, tuple[int, ...], int], ...]:
    if root is None:
        return ()
    result = []
    for degree in (1, 2):
        source_keys = entity_keys(source, degree)
        source_ids = np.asarray(source.entity_set(degree).entity_ids, dtype=np.int64)
        current = {
            tuple(int(value) for value in key if value >= 0): int(identifier)
            for key, identifier in zip(source_keys, source_ids, strict=True)
        }
        current_by_id = {identifier: key for key, identifier in current.items()}
        root_keys = entity_keys(root, degree)
        root_ids = np.asarray(root.entity_set(degree).entity_ids, dtype=np.int64)
        for key_row, raw_identifier in zip(root_keys, root_ids, strict=True):
            key = tuple(int(value) for value in key_row if value >= 0)
            identifier = int(raw_identifier)
            existing = current.get(key)
            if existing is not None:
                if existing != identifier:
                    raise ValueError("Mixed source and root disagree on an entity ID.")
            elif identifier in current_by_id:
                raise ValueError("Mixed source reuses an original root entity ID.")
            else:
                result.append((degree, key, identifier))
    return tuple(sorted(result))


def _prescribed_entities(
    source: CellMesh,
    retired_entities: tuple[tuple[int, tuple[int, ...], int], ...],
    /,
) -> tuple[PrescribedEntityIds, ...]:
    records = []
    for degree in (1, 2):
        current = {
            tuple(int(value) for value in row if value >= 0)
            for row in entity_keys(source, degree)
        }
        values = [
            (key, identifier)
            for item_degree, key, identifier in retired_entities
            if item_degree == degree and key not in current
        ]
        if values:
            width = max(len(key) for key, _ in values)
            keys = np.full((len(values), width), -1, dtype=np.int64)
            ids = np.empty((len(values),), dtype=np.int64)
            for row, (key, identifier) in enumerate(values):
                keys[row, : len(key)] = key
                ids[row] = identifier
            records.append(PrescribedEntityIds(degree, keys, ids))
    return tuple(records)


def _cell_transfer(
    source_ids: NDArray[np.int64],
    targets: list[_TargetCell],
    /,
) -> MixedCellContentTransfer:
    rows = sorted(
        (
            (source_id, target.identifier, weight)
            for target in targets
            for source_id, weight in zip(
                target.source_ids, target.content_weights, strict=True
            )
        ),
        key=lambda value: (value[1], value[0]),
    )
    row_sources = np.asarray([value[0] for value in rows], dtype=np.int64)
    row_targets = np.asarray([value[1] for value in rows], dtype=np.int64)
    exact = tuple(value[2] for value in rows)
    weights = np.asarray([float(value) for value in exact], dtype=np.float64)
    target_ids = np.asarray(
        sorted({target.identifier for target in targets}), dtype=np.int64
    )
    source_sorted = np.sort(source_ids)
    transfer_id = canonical_fingerprint(
        {
            "kind": "mixed-cell-content-transfer",
            "source_cell_ids": source_sorted,
            "target_cell_ids": target_ids,
            "row_source_ids": row_sources,
            "row_target_ids": row_targets,
            "exact_weights": tuple(_fraction_payload(value) for value in exact),
        }
    )
    return MixedCellContentTransfer(
        _immutable(source_sorted),
        _immutable(target_ids),
        _immutable(row_sources),
        _immutable(row_targets),
        exact,
        _immutable(weights),
        transfer_id,
    )


def _mixed_reference_maps(
    targets: list[_TargetCell],
    coarsened_parent_ids: set[int],
    /,
) -> tuple[ReferenceCellMapWitnesses, ...]:
    grouped: dict[str, list[_TargetCell]] = defaultdict(list)
    for target in targets:
        if target.identifier not in coarsened_parent_ids:
            if len(target.source_ids) != 1:
                raise ValueError(
                    "Direct mixed reference maps require one source-cell parent."
                )
            grouped[target.kind].append(target)
    witnesses = []

    @cache
    def local_affine_chart(
        kind: str, vertices: tuple[tuple[Fraction, ...], ...], /
    ) -> tuple[Expression, ...]:
        return _affine_chart(*_exact_child_to_parent_affine_map(kind, vertices))

    for kind, values in sorted(grouped.items()):
        values.sort(key=lambda value: value.identifier)
        exact = tuple(value.reference_vertices for value in values)
        points = np.asarray(
            [
                [[float(coordinate) for coordinate in point] for point in cell]
                for cell in exact
            ],
            dtype=np.float64,
        )
        witness = ReferenceCellMapWitnesses(
            np.asarray([value.identifier for value in values], dtype=np.int64),
            np.asarray([value.source_ids[0] for value in values], dtype=np.int64),
            kind,
            kind,
            points,
            exact,
            freeze_reference_chart_arguments(
                tuple(local_affine_chart(kind, vertices) for vertices in exact)
            ),
        )
        validate_reference_cell_map_witnesses(witness)
        witnesses.append(witness)
    return tuple(witnesses)


def _mixed_coarsening_partition(
    records: tuple[MixedCoarseningMap, ...], /
) -> ReferenceCellPartition | None:
    if not records:
        return None
    grouped: dict[str, list[tuple[int, int, tuple[tuple[Fraction, ...], ...]]]] = (
        defaultdict(list)
    )
    for record in records:
        for child, vertices in zip(
            record.child_cell_ids, record.child_reference_vertices, strict=True
        ):
            grouped[record.parent_cell_kind].append(
                (int(child), record.parent_cell_id, vertices)
            )
    groups = []
    for kind, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row[0])
        exact = tuple(row[2] for row in rows)
        groups.append(
            ReferenceCellMapWitnesses(
                np.asarray([row[0] for row in rows], dtype=np.int64),
                np.asarray([row[1] for row in rows], dtype=np.int64),
                kind,
                kind,
                np.asarray(
                    [
                        [[float(value) for value in point] for point in cell]
                        for cell in exact
                    ],
                    dtype=np.float64,
                ),
                exact,
                freeze_reference_chart_arguments(
                    tuple(
                        _affine_chart(*_exact_child_to_parent_affine_map(kind, vertices))
                        for vertices in exact
                    )
                ),
            )
        )
    partition = ReferenceCellPartition(tuple(groups))
    validate_reference_cell_partition(partition)
    return partition


def _reference_map_sources(
    witnesses: tuple[ReferenceCellMapWitnesses, ...], /
) -> dict[int, int]:
    result: dict[int, int] = {}
    for witness in witnesses:
        for mapping in nested_cell_reference_maps(witness):
            if mapping.arguments is None:
                raise ValueError("Mixed reference-map corners lack an exact chart.")
            result[mapping.target_cell_id] = mapping.source_cell_id
    return result


def _exact_child_to_parent_affine_map(
    cell_kind: str,
    child_parent_vertices: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    topology = reference_cell_topology(cell_kind)
    dimension = topology.dimension
    local = tuple(
        tuple(Fraction(float(value)) for value in point) for point in topology.vertices
    )
    if cell_kind in ("tetrahedron", "simplex:3", "prism"):
        basis = (0, 1, 2, 3)
    elif cell_kind in ("hexahedron", "pyramid"):
        basis = (0, 1, 3, 4)
    else:
        raise ValueError(f"Unsupported mixed coarsening family {cell_kind!r}.")
    if len(child_parent_vertices) != len(local) or any(
        len(point) != dimension for point in child_parent_vertices
    ):
        raise ValueError(
            "Mixed sibling reference corners do not match their cell family."
        )
    local_matrix = tuple(
        tuple(
            local[basis[column + 1]][row] - local[basis[0]][row]
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    parent_matrix = tuple(
        tuple(
            child_parent_vertices[basis[column + 1]][row]
            - child_parent_vertices[basis[0]][row]
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    transposed = solve_exact(
        tuple(zip(*local_matrix, strict=True)),
        tuple(zip(*parent_matrix, strict=True)),
    )
    matrix = tuple(
        tuple(transposed[column][row] for column in range(dimension))
        for row in range(dimension)
    )
    offset = tuple(
        child_parent_vertices[0][row]
        - sum(
            (matrix[row][column] * local[0][column] for column in range(dimension)),
            Fraction(0),
        )
        for row in range(dimension)
    )
    if any(
        tuple(
            offset[row]
            + sum(
                (matrix[row][column] * point[column] for column in range(dimension)),
                Fraction(0),
            )
            for row in range(dimension)
        )
        != parent_point
        for point, parent_point in zip(local, child_parent_vertices, strict=True)
    ):
        raise MixedTemplateClosureError(
            "The authored mixed sibling map is not an affine reference restriction."
        )
    return matrix, offset


def _mixed_coarsening_reference_transforms(
    maps: tuple[MixedCoarseningMap, ...],
    /,
) -> dict[tuple[int, int], _MixedReferenceAffineMap]:
    result: dict[tuple[int, int], _MixedReferenceAffineMap] = {}
    seen_children: set[int] = set()
    for record in maps:
        child_ids = np.asarray(record.child_cell_ids, dtype=np.int64)
        if child_ids.ndim != 1 or child_ids.size != len(record.child_reference_vertices):
            raise ValueError(
                "Mixed coarsening reference rows do not align with child IDs."
            )
        for child_id_raw, exact_vertices in zip(
            child_ids, record.child_reference_vertices, strict=True
        ):
            child_id = int(child_id_raw)
            if child_id in seen_children:
                raise ValueError("Mixed coarsening maps repeat a source child cell.")
            seen_children.add(child_id)
            result[(record.parent_cell_id, child_id)] = _exact_child_to_parent_affine_map(
                record.parent_cell_kind, exact_vertices
            )
    return result


def _affine_chart(
    matrix: tuple[tuple[Fraction, ...], ...],
    offset: tuple[Fraction, ...],
    /,
) -> tuple[Expression, ...]:
    dimension = len(offset)
    variables = axes(dimension)
    return tuple(
        sum_polynomials(
            (
                constant(offset[row], dimension),
                *(
                    scale(variables[column], matrix[row][column])
                    for column in range(dimension)
                ),
            ),
            dimension,
        )
        for row in range(dimension)
    )


def _parent_to_child_chart(
    matrix: tuple[tuple[Fraction, ...], ...],
    offset: tuple[Fraction, ...],
    /,
) -> tuple[Expression, ...]:
    dimension = len(offset)
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )
    inverse = tuple(tuple(row) for row in solve_exact(matrix, identity))
    return _affine_chart(
        inverse,
        tuple(
            -sum(
                (inverse[row][column] * offset[column] for column in range(dimension)),
                Fraction(0),
            )
            for row in range(dimension)
        ),
    )


def _mixed_restriction_source(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    targets: list[_TargetCell],
    coarsening_maps: tuple[MixedCoarseningMap, ...],
    coarsening_partition: ReferenceCellPartition | None,
    bind_current_as_root: bool,
    /,
) -> CellGeometryRestrictionSource:
    """Compose current cells into one retained root, proving complete restorations."""
    ancestry = None if bind_current_as_root else source_geometry.restriction_source
    source_blocks = {block.name: block for block in source_mesh.blocks}
    if ancestry is None:
        vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
        parent_cells = {
            block.name: np.asarray(block.global_ids, dtype=np.int64)
            for block in source_mesh.blocks
        }
        parent_vertices = {
            block.name: vertex_ids[np.asarray(block.vertices, dtype=np.int64)]
            for block in source_mesh.blocks
        }
        root_maps: dict[str, ReferenceCellMapWitnesses | None] = {}
        for block in source_mesh.blocks:
            topology = reference_cell_topology(block.cell_kind)
            corners = tuple(
                tuple(Fraction(value) for value in point) for point in topology.vertices
            )
            exact = tuple(corners for _ in range(block.cell_count))
            root_maps[block.name] = ReferenceCellMapWitnesses(
                np.arange(block.cell_count, dtype=np.int64),
                parent_cells[block.name],
                block.cell_kind,
                block.cell_kind,
                np.asarray(exact, dtype=np.float64),
                exact,
                freeze_reference_chart_arguments(
                    tuple(axes(topology.dimension) for _ in range(block.cell_count))
                ),
            )
        ancestry = CellGeometryRestrictionSource(
            cell_geometry_id(source_geometry),
            source_mesh.topology_id,
            parent_cells,
            parent_vertices,
            block_source_blocks={block.name: block.name for block in source_mesh.blocks},
            block_parent_reference_maps=root_maps,
        )
    if set(ancestry.block_names) != set(source_blocks):
        raise ValueError("Mixed root ancestry must cover every source block.")
    root_maps = dict(
        zip(ancestry.block_names, ancestry.parent_reference_maps, strict=True)
    )
    root_vertices = dict(
        zip(ancestry.block_names, ancestry.parent_vertex_ids, strict=True)
    )
    root_blocks = (
        None
        if ancestry.source_block_names is None
        else dict(zip(ancestry.block_names, ancestry.source_block_names, strict=True))
    )
    if coarsening_maps:
        if coarsening_partition is None:
            raise ValueError(
                "Mixed root restoration requires a complete child partition."
            )
        validate_reference_cell_partition(coarsening_partition)
    elif coarsening_partition is not None:
        raise ValueError("Mixed root restoration has an unexpected child partition.")
    target_by_id = {value.identifier: value for value in targets}
    coarsened = {value.parent_cell_id: value for value in coarsening_maps}
    simplex_records = tuple(
        record
        for record in coarsening_maps
        if record.parent_cell_kind in ("tetrahedron", "simplex:3")
    )
    simplex_restored = {}
    if simplex_records:
        available_maps = {
            name: witness for name, witness in root_maps.items() if witness is not None
        }
        if set(available_maps) != set(source_blocks):
            raise ValueError("Mixed root coarsening requires all source reference maps.")
        child_ids = tuple(
            int(child)
            for record in simplex_records
            for child in np.asarray(record.child_cell_ids, dtype=np.int64)
        )
        parent_ids = tuple(
            record.parent_cell_id
            for record in simplex_records
            for _ in record.child_reference_vertices
        )
        corners = tuple(
            vertices
            for record in simplex_records
            for vertices in record.child_reference_vertices
        )
        restored_maps = coarsen_reference_cell_maps(
            NestedReferenceWitnesses(
                np.asarray(child_ids, dtype=np.int64),
                np.asarray(parent_ids, dtype=np.int64),
                np.asarray(corners, dtype=np.float64),
                fine_reference_exact_vertices=corners,
            ),
            available_maps,
            source_mesh,
            target_mesh,
        )
        target_blocks = {block.name: block for block in target_mesh.blocks}
        for name, witness in restored_maps.items():
            ids = np.asarray(target_blocks[name].global_ids, dtype=np.int64)
            for row in nested_cell_reference_maps(witness):
                simplex_restored[int(ids[row.target_cell_id])] = row
    output_cells: dict[str, NDArray[np.int64]] = {}
    output_vertices: dict[str, NDArray[np.int64]] = {}
    output_maps: dict[str, ReferenceCellMapWitnesses] = {}
    for block in target_mesh.blocks:
        source_block = source_blocks[block.name]
        root_map = root_maps[block.name]
        if root_map is None:
            raise ValueError("Mixed root ancestry requires exact reference maps.")
        root_rows = nested_cell_reference_maps(root_map)
        if len(root_rows) != source_block.cell_count:
            raise ValueError("Mixed root reference rows do not cover the source block.")
        source_ids = np.asarray(source_block.global_ids, dtype=np.int64)
        source_rows = {int(identifier): row for row, identifier in enumerate(source_ids)}
        parent_vertex_rows = np.asarray(root_vertices[block.name], dtype=np.int64)
        target_ids = np.asarray(block.global_ids, dtype=np.int64)
        selected = [target_by_id[int(identifier)] for identifier in target_ids]
        direct = _mixed_reference_maps(selected, set(coarsened))
        by_target = {}
        if direct:
            if len(direct) != 1:
                raise ValueError("Mixed block reference maps have inconsistent families.")
            composed = compose_reference_cell_maps(
                direct[0], root_map, intermediate_cell_global_ids=source_ids
            )
            composed_exact = exact_reference_cell_vertices(composed)
            arguments = composed.source_reference_chart_arguments
            if arguments is None:
                raise ValueError("Mixed root composition lost exact chart arguments.")
            for index, identifier in enumerate(
                np.asarray(composed.target_cell_ids, dtype=np.int64)
            ):
                by_target[int(identifier)] = (
                    int(composed.source_cell_ids[index]),
                    composed_exact[index],
                    arguments[index],
                )
        source_supports = []
        for target in selected:
            record = coarsened.get(target.identifier)
            if record is None:
                source_row = source_rows[target.source_ids[0]]
                source_supports.append(parent_vertex_rows[source_row])
                continue
            restored = simplex_restored.get(target.identifier)
            if restored is not None:
                supports = tuple(
                    parent_vertex_rows[source_rows[int(child)]]
                    for child in np.asarray(record.child_cell_ids, dtype=np.int64)
                )
                if (
                    not supports
                    or restored.arguments is None
                    or any(not np.array_equal(supports[0], row) for row in supports[1:])
                ):
                    raise ValueError(
                        "Mixed simplex siblings have inconsistent root support."
                    )
                by_target[target.identifier] = (
                    restored.source_cell_id,
                    restored.source_reference_vertices,
                    restored.arguments,
                )
                source_supports.append(supports[0])
                continue
            candidates = []
            supports = []
            for child_id, vertices in zip(
                np.asarray(record.child_cell_ids, dtype=np.int64),
                record.child_reference_vertices,
                strict=True,
            ):
                source_row = source_rows[int(child_id)]
                root = root_rows[source_row]
                if root.arguments is None:
                    raise ValueError("Mixed root child lacks an exact reference chart.")
                inverse = _parent_to_child_chart(
                    *_exact_child_to_parent_affine_map(record.parent_cell_kind, vertices)
                )
                chart = tuple(
                    compose_expression(value, inverse) for value in root.arguments
                )
                candidates.append((root.source_cell_id, chart))
                supports.append(parent_vertex_rows[source_row])
            if not candidates:
                raise ValueError("Mixed root restoration requires all sibling charts.")
            root_id, chart = candidates[0]
            if any(
                other_id != root_id
                or len(other_chart) != len(chart)
                or any(
                    not expression_equal(first, second)
                    for first, second in zip(chart, other_chart, strict=True)
                )
                for other_id, other_chart in candidates[1:]
            ) or any(not np.array_equal(supports[0], row) for row in supports[1:]):
                raise ValueError("Mixed siblings do not restore one exact root chart.")
            frozen_chart = freeze_reference_chart_arguments((chart,))[0]
            corners = tuple(
                tuple(Fraction(value) for value in point)
                for point in reference_cell_topology(block.cell_kind).vertices
            )
            evaluated = tuple(
                evaluate_reference_chart_arguments(block.cell_kind, frozen_chart, point)
                for point in corners
            )
            if any(
                not isinstance(value, Fraction) for point in evaluated for value in point
            ):
                raise ValueError("Mixed restored root corners must be exact rationals.")
            exact = tuple(
                tuple(value for value in point if isinstance(value, Fraction))
                for point in evaluated
            )
            by_target[target.identifier] = (root_id, exact, frozen_chart)
            source_supports.append(supports[0])
        if set(by_target) != set(int(value) for value in target_ids):
            raise ValueError("Mixed root charts do not cover every target cell.")
        ordered = tuple(by_target[int(identifier)] for identifier in target_ids)
        exact = tuple(row[1] for row in ordered)
        output_cells[block.name] = np.asarray([row[0] for row in ordered], dtype=np.int64)
        output_vertices[block.name] = np.stack(source_supports)
        output_maps[block.name] = ReferenceCellMapWitnesses(
            np.arange(block.cell_count, dtype=np.int64),
            output_cells[block.name],
            block.cell_kind,
            root_map.source_cell_kind,
            np.asarray(exact, dtype=np.float64),
            exact,
            freeze_reference_chart_arguments(tuple(row[2] for row in ordered)),
        )
    return CellGeometryRestrictionSource(
        ancestry.source_geometry_id,
        ancestry.source_topology_id,
        output_cells,
        output_vertices,
        block_source_blocks=(
            None
            if root_blocks is None
            else {block.name: root_blocks[block.name] for block in target_mesh.blocks}
        ),
        block_parent_reference_maps=output_maps,
        carrier_publication=ancestry.carrier_publication,
        carrier_geometry=ancestry.carrier_geometry,
    )


def _mixed_target_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    targets: list[_TargetCell],
    reference_maps: tuple[ReferenceCellMapWitnesses, ...],
    coarsening_maps: tuple[MixedCoarseningMap, ...],
    coarsening_partition: ReferenceCellPartition | None,
    bind_current_as_root: bool,
    /,
) -> CellGeometrySpec:
    source_elements, source_routes, _ = source_geometry.resolve(source_mesh)
    source_locations = {
        int(identifier): (block_index, row)
        for block_index, block in enumerate(source_mesh.blocks)
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    target_by_id = {target.identifier: target for target in targets}
    direct_sources = _reference_map_sources(reference_maps)
    coarsened = {value.parent_cell_id: value for value in coarsening_maps}

    def source_chart(
        cell_id: int,
    ) -> tuple[ScalarCoordinateElement, tuple[Expression, ...], NDArray[np.int32]]:
        location = source_locations.get(cell_id)
        if location is None:
            raise ValueError("Mixed target geometry references an absent source cell.")
        block_index, row = location
        element = source_elements[block_index]
        if not isinstance(element, ScalarCoordinateElement):
            raise ValueError("Mixed target geometry requires scalar H1 source maps.")
        chain = coordinate_reference_chain(element)
        chart = restriction_chart_arguments(element, cell_row=row)
        if chart is None:
            chart = axes(
                reference_cell_topology(
                    source_mesh.blocks[block_index].cell_kind
                ).dimension
            )
        return (
            chain[0],
            tuple(chart),
            np.asarray(source_routes[block_index], dtype=np.int32)[row],
        )

    elements: dict[str, PerCellRestrictedCellGeometryElement] = {}
    routes: dict[str, NDArray[np.int32]] = {}
    for block in target_mesh.blocks:
        target_ids = np.asarray(block.global_ids, dtype=np.int64)
        roots: list[ScalarCoordinateElement] = []
        source_ids: list[int] = []
        chart_rows: list[tuple[Expression, ...]] = []
        route_rows: list[NDArray[np.int32]] = []
        for target_id in target_ids.tolist():
            target = target_by_id[int(target_id)]
            coarsening = coarsened.get(int(target_id))
            if coarsening is None:
                source_id = direct_sources.get(int(target_id), -1)
                if source_id < 0 or source_id != target.source_ids[0]:
                    raise ValueError(
                        "Mixed target geometry lacks exact source-cell ancestry."
                    )
                # Family-preserving templates are affine sub-cells; the exact affine
                # chart avoids the removable rational form of collapsed families.
                target_chart = _affine_chart(
                    *_exact_child_to_parent_affine_map(
                        target.kind, target.reference_vertices
                    )
                )
                root, source_chart_arguments, route = source_chart(source_id)
                chart = tuple(
                    compose_expression(value, target_chart)
                    for value in source_chart_arguments
                )
            else:
                candidates: list[
                    tuple[
                        int,
                        ScalarCoordinateElement,
                        tuple[Expression, ...],
                        NDArray[np.int32],
                    ]
                ] = []
                for child_id, child_vertices in zip(
                    np.asarray(coarsening.child_cell_ids, dtype=np.int64).tolist(),
                    coarsening.child_reference_vertices,
                    strict=True,
                ):
                    child_root, child_chart, child_route = source_chart(child_id)
                    child_matrix, child_offset = _exact_child_to_parent_affine_map(
                        coarsening.parent_cell_kind, child_vertices
                    )
                    parent_to_child = _parent_to_child_chart(child_matrix, child_offset)
                    candidates.append(
                        (
                            child_id,
                            child_root,
                            tuple(
                                compose_expression(value, parent_to_child)
                                for value in child_chart
                            ),
                            child_route,
                        )
                    )
                if not candidates:
                    raise ValueError("A mixed coarsening map has no source children.")
                source_id, root, chart, route = candidates[0]
                if any(
                    other_root.element_id != root.element_id
                    or not np.array_equal(other_route, route)
                    or any(
                        not expression_equal(first, second)
                        for first, second in zip(chart, other_chart, strict=True)
                    )
                    for _, other_root, other_chart, other_route in candidates[1:]
                ):
                    raise ValueError(
                        "Mixed sibling source maps do not reconstruct one exact parent map."
                    )
            if route.shape != (root.local_dof_count,):
                raise ValueError(
                    "Mixed source geometry roots do not match their DOF routes."
                )
            roots.append(root)
            source_ids.append(source_id)
            chart_rows.append(chart)
            route_rows.append(route)
        if not roots or any(value.element_id != roots[0].element_id for value in roots):
            raise ValueError(
                "A mixed target block cannot combine unrelated geometry roots."
            )
        elements[block.name] = PerCellRestrictedCellGeometryElement(
            roots[0],
            block.cell_kind,
            target_ids,
            source_ids,
            chart_rows,
        )
        routes[block.name] = np.stack(route_rows).astype(np.int32, copy=False)
    restriction = _mixed_restriction_source(
        source_mesh,
        source_geometry,
        target_mesh,
        targets,
        coarsening_maps,
        coarsening_partition,
        bind_current_as_root,
    )
    return CellGeometrySpec(
        elements, routes, source_geometry.coordinates, restriction_source=restriction
    )


def _exact_target_vertex_coordinates(
    geometry: CellGeometrySpec,
    mesh: CellMesh,
    edit_vertex_ids: NDArray[np.int64],
    /,
) -> CoordinateSourceBank:
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    exact_coordinates: dict[int, tuple[ExactScalar, ...]] = {}
    for block in mesh.blocks:
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        corners = tuple(
            tuple(Fraction(float(value)) for value in point)
            for point in reference_cell_topology(block.cell_kind).vertices
        )
        evaluated = geometry.evaluate_exact(
            mesh, identifiers, tuple(corners for _ in identifiers)
        )
        cells = np.asarray(block.vertices, dtype=np.int64)
        for cell, points in zip(cells, evaluated, strict=True):
            local_ids = vertex_ids[cell]
            for vertex_id, point in zip(local_ids, points, strict=True):
                key = int(vertex_id)
                previous = exact_coordinates.get(key)
                value = tuple(point)
                if previous is not None and previous != value:
                    raise ValueError(
                        "Mixed source restrictions disagree at a shared target vertex."
                    )
                exact_coordinates[key] = value
    if set(exact_coordinates) != set(int(value) for value in edit_vertex_ids):
        raise ValueError("Exact mixed geometry does not cover the target vertex bank.")
    return tuple(exact_coordinates[int(identifier)] for identifier in edit_vertex_ids)


def adapt_mixed_mesh(
    mesh: CellMesh,
    /,
    *,
    refine_cell_ids: ArrayLike = (),
    coarsen_cell_ids: ArrayLike = (),
    hierarchy: MixedAdaptationHierarchy | None = None,
    cell_classes: ArrayLike | None = None,
    protected_vertex_ids: ArrayLike | None = None,
    protected_cell_ids: ArrayLike | None = None,
    protected_edge_keys: ArrayLike | None = None,
    protected_face_keys: ArrayLike | None = None,
    layer_columns: MixedLayerColumns | None = None,
    periodic_edge_lock: PeriodicRefinementEdgeLock | None = None,
    source_geometry: CellGeometrySpec | None = None,
    root_mesh: CellMesh | None = None,
) -> MixedAdaptationOutcome:
    """Build a face-conforming family-preserving refinement/coarsening edit.

    ``cell_classes`` is aligned with source block order and separates material,
    region, layer, and other organization classes without freezing the interface.
    Coarsening marks identify all active children of each restored sibling family.
    Starting from an already restricted geometry requires ``root_mesh`` bound
    to its retained source topology. Exact charts and root vertex identities
    seed the bounded forest and vertex ancestry, including inactive root IDs.
    """
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError("Mixed volume adaptation requires a 3D mesh in 3D coordinates.")
    if hierarchy is not None and not isinstance(hierarchy, MixedAdaptationHierarchy):
        raise TypeError("hierarchy must be MixedAdaptationHierarchy or None.")
    if layer_columns is not None and not isinstance(layer_columns, MixedLayerColumns):
        raise TypeError("layer_columns must be MixedLayerColumns or None.")
    if source_geometry is not None:
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("source_geometry must be CellGeometrySpec or None.")
        source_geometry.resolve(mesh)
    elif mesh.periodic_topology is not None:
        raise ValueError("Periodic mixed adaptation requires exact source geometry.")
    block_count = sum(block.cell_count for block in mesh.blocks)
    classes = (
        np.zeros((block_count,), dtype=np.int64)
        if cell_classes is None
        else np.asarray(cell_classes)
    )
    if (
        classes.ndim != 1
        or classes.shape != (block_count,)
        or classes.dtype.kind not in "iu"
    ):
        raise TypeError(
            "cell_classes must be an integer vector aligned with source block order."
        )
    classes = classes.astype(np.int64, copy=False)
    if np.any(classes < 0):
        raise ValueError("cell_classes must be nonnegative.")
    layer_values = (
        {}
        if layer_columns is None
        else {
            int(cell): (int(column), int(interval))
            for cell, column, interval in zip(
                np.asarray(layer_columns.cell_ids),
                np.asarray(layer_columns.column_ids),
                np.asarray(layer_columns.interval_indices),
                strict=True,
            )
        }
    )
    if hierarchy is not None and cell_classes is None:
        class_by_id = {
            value.cell_id: value.cell_class for value in hierarchy.active_cells
        }
        row = 0
        for block in mesh.blocks:
            for identifier in np.asarray(block.global_ids, dtype=np.int64):
                classes[row] = class_by_id.get(int(identifier), -1)
                row += 1
        if np.any(classes < 0):
            raise ValueError("Mixed hierarchy lacks class ancestry for the source mesh.")
    if hierarchy is not None and layer_columns is None:
        layer_values = {
            value.cell_id: (value.column_id, value.interval_index)
            for value in hierarchy.active_cells
            if value.column_id is not None and value.interval_index is not None
        }
    cells = _source_cells(mesh, classes, layer_values)
    source_ids = np.asarray([cell.identifier for cell in cells], dtype=np.int64)
    active_ids = set(source_ids.tolist())
    if not set(layer_values).issubset(active_ids):
        raise ValueError("Layer columns name cells absent from the source mesh.")
    source_charts = _source_root_charts(mesh, source_geometry, hierarchy, root_mesh)
    root_points = _mapping_points(hierarchy, mesh, source_charts, root_mesh)
    forest = (
        _forest_roots(cells, source_charts)
        if hierarchy is None
        else hierarchy.forest_roots
    )
    if hierarchy is not None:
        _validate_hierarchy(hierarchy, mesh, cells, root_points)
        if layer_columns is not None:
            old_layers = {
                value.cell_id: (value.column_id, value.interval_index)
                for value in hierarchy.active_cells
                if value.column_id is not None
            }
            if old_layers != layer_values:
                raise ValueError(
                    "Layer-column ancestry differs from the bound hierarchy."
                )
            if hierarchy.layer_policy != (
                layer_columns.hard_first_thickness,
                layer_columns.axial_refinement,
                layer_columns.allow_schedule_change,
            ):
                raise ValueError(
                    "Layer-column permissions differ from the bound hierarchy."
                )
    refine = set(_integer_vector(refine_cell_ids, "refine_cell_ids").tolist())
    coarsen_marks = set(_integer_vector(coarsen_cell_ids, "coarsen_cell_ids").tolist())
    if refine & coarsen_marks:
        raise ValueError(
            "A mixed cell cannot be marked for both refinement and coarsening."
        )
    if not refine.issubset(active_ids) or not coarsen_marks.issubset(active_ids):
        raise ValueError("Mixed adaptation marks must name current source cell IDs.")
    requested_refine = set(refine)
    locked_edges: dict[int, tuple[int, int]] = {}
    if periodic_edge_lock is not None:
        if not isinstance(periodic_edge_lock, PeriodicRefinementEdgeLock):
            raise TypeError("periodic_edge_lock must be PeriodicRefinementEdgeLock.")
        locked_cells, edge_ids = periodic_edge_lock.require_source(mesh)
        if set(locked_cells.tolist()) != requested_refine:
            raise ValueError(
                "Periodic edge-lock cells must equal the requested refinement marks."
            )
        edge_keys_by_id = {
            int(identifier): (int(key[0]), int(key[1]))
            for identifier, key in zip(
                np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64),
                entity_keys(mesh, 1),
                strict=True,
            )
        }
        locked_edges = {
            int(cell): edge_keys_by_id[int(edge)]
            for cell, edge in zip(locked_cells, edge_ids, strict=True)
        }
    seams = _periodic_seam_faces(mesh, cells)
    if mesh.periodic_topology is not None and refine:
        refine = set(
            periodic_orbit_closure(mesh, 3, np.asarray(sorted(refine), dtype=np.int64))
            .require_source(mesh, 3)
            .tolist()
        )
        if refine & coarsen_marks:
            raise ValueError(
                "Periodic refinement closure conflicts with coarsening marks."
            )
    protected_vertices = _protected_id_set(protected_vertex_ids, "protected_vertex_ids")
    known_vertices = set(np.asarray(mesh.vertex_global_ids, dtype=np.int64).tolist())
    if not protected_vertices.issubset(known_vertices):
        raise ValueError("protected_vertex_ids must name source vertices.")
    protected_cells = _protected_id_set(protected_cell_ids, "protected_cell_ids")
    if not protected_cells.issubset(active_ids):
        raise ValueError("protected_cell_ids must name source cells.")
    protected_edges = _protected_keys(mesh, protected_edge_keys, 1, "protected_edge_keys")
    protected_faces = _protected_keys(mesh, protected_face_keys, 2, "protected_face_keys")
    if refine & protected_cells:
        raise MixedTemplateClosureError("A protected cell cannot be refined.")

    if mesh.periodic_topology is None:
        candidates, rejected, removed = _coarsening_candidates(
            cells,
            coarsen_marks,
            hierarchy,
            protected_vertices,
            protected_cells,
            root_points,
        )
    else:
        candidates, rejected, removed = _periodic_coarsening_candidates(
            cells,
            coarsen_marks,
            hierarchy,
            protected_vertices,
            protected_cells,
            root_points,
            seams,
        )
    if refine & removed:
        raise ValueError("Periodic coarsening closure conflicts with refinement marks.")
    coordinate_rows = {
        int(identifier): np.asarray(point, dtype=np.float64)
        for identifier, point in zip(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64),
            np.asarray(mesh.coordinates, dtype=np.float64),
            strict=True,
        )
    }
    working = [cell for cell in cells if cell.identifier not in removed]
    coarsen_sources: dict[int, tuple[int, ...]] = {}
    for record in candidates:
        parent_vertices = np.asarray(record.parent_vertices, dtype=np.int64)
        parent = _Cell(
            record.parent_id,
            record.block_name,
            record.cell_kind,
            parent_vertices,
            record.cell_class,
            record.column_id,
            record.interval_index,
        )
        working.append(parent)
        coarsen_sources[record.parent_id] = record.child_ids
    working.sort(key=lambda value: value.identifier)
    if len(working) == 0:
        raise MixedTemplateClosureError(
            "Mixed adaptation cannot remove every source cell."
        )
    if any(
        any(int(vertex) not in coordinate_rows for vertex in cell.vertices)
        for cell in working
    ):
        raise ValueError(
            "Restored mixed parent corners are absent from source coordinates."
        )

    owners = _mesh_facet_owners(working)
    interfaces: list[tuple[int, int, int, int, dict[int, int] | None]] = [
        (left[0], left[1], right[0], right[1], None)
        for values in owners.values()
        if len(values) == 2
        for left, right in (values,)
    ]
    interfaces.extend(
        _periodic_working_interfaces(mesh, cells, working, candidates, seams)
    )
    policy = (
        hierarchy.layer_policy
        if hierarchy is not None and layer_columns is None
        else None
        if layer_columns is None
        else (
            layer_columns.hard_first_thickness,
            layer_columns.axial_refinement,
            layer_columns.allow_schedule_change,
        )
    )
    domains: list[list[_Template]] = []
    for cell in working:
        column = cell.column_id
        axial = column is None or (policy is not None and policy[1])
        if (
            column is not None
            and cell.interval_index == 0
            and policy is not None
            and policy[0]
            and not policy[2]
        ):
            axial = False
        blocked = cell.identifier in protected_cells or cell.identifier in coarsen_sources
        domain = _template_domain(
            cell,
            axial=axial,
            refine=cell.identifier in refine,
            blocked=blocked,
            protected_edges=protected_edges,
            protected_faces=protected_faces,
        )
        locked_edge = locked_edges.get(cell.identifier)
        if locked_edge is not None:
            domain = [
                value
                for value in domain
                if locked_edge in _template_facts(cell, value).split_edges
            ]
        if not domain:
            raise MixedTemplateClosureError(
                f"No permitted family template refines cell {cell.identifier}."
            )
        domains.append(domain)
    chosen, closure_steps = _close_templates(working, domains, interfaces)
    source_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    next_cell_id = (
        max(int(np.max(source_ids, initial=-1)) + 1, 0)
        if hierarchy is None
        else hierarchy.next_cell_id
    )
    next_vertex_id = (
        max(int(np.max(source_vertex_ids, initial=-1)) + 1, 0)
        if hierarchy is None
        else hierarchy.next_vertex_id
    )
    next_cell_id = max(
        next_cell_id,
        max((chart.root_cell_id for chart in source_charts.values()), default=-1) + 1,
    )
    next_vertex_id = max(next_vertex_id, max(root_points, default=-1) + 1)
    if root_mesh is not None:
        root_cell_ids = (
            int(value)
            for block in root_mesh.blocks
            for value in np.asarray(block.global_ids, dtype=np.int64)
        )
        next_cell_id = max(next_cell_id, max(root_cell_ids, default=-1) + 1)
    if mesh.periodic_topology is not None:
        next_ids = periodic_next_entity_ids(mesh)
        next_cell_id = max(next_cell_id, next_ids[3])
        next_vertex_id = max(next_vertex_id, next_ids[0])
    point_to_vertex: dict[_PointKey, int] = {}
    target_stencils: dict[int, _PointKey] = {
        int(identifier): ((int(identifier), Fraction(1)),)
        for identifier in source_vertex_ids
    }
    output: list[_TargetCell] = []
    new_siblings: list[MixedSibling] = []
    schedule_changed: list[int] = []
    for cell, template in zip(working, chosen, strict=True):
        if template.index == 0:
            output_sources = coarsen_sources.get(cell.identifier, (cell.identifier,))
            content_weights = tuple(Fraction(1) for _ in output_sources)
            reference = tuple(
                tuple(Fraction(float(value)) for value in point)
                for point in reference_cell_topology(cell.kind).vertices
            )
            output.append(
                _TargetCell(
                    cell.identifier,
                    cell.name,
                    cell.kind,
                    tuple(int(value) for value in cell.vertices),
                    cell.cell_class,
                    cell.column_id,
                    cell.interval_index,
                    tuple(output_sources),
                    content_weights,
                    reference,
                )
            )
            continue
        facts = _template_facts(cell, template)
        child_ids: list[int] = []
        reference_rows: list[tuple[tuple[Fraction, ...], ...]] = []
        children = tuple(
            template.children[child_row, :, :]
            for child_row in range(template.children.shape[0])
        )
        split_axial = cell.kind in ("prism", "hexahedron") and any(
            np.ptp(child[:, 2]) < 1.0 for child in children
        )
        for child, point_keys in zip(children, facts.points, strict=True):
            target_vertices: list[int] = []
            for key in point_keys:
                if len(key) == 1 and key[0][1] == 1:
                    identifier = key[0][0]
                else:
                    identifier = point_to_vertex.get(key, -1)
                    if identifier < 0:
                        identifier = next_vertex_id
                        next_vertex_id += 1
                        point_to_vertex[key] = identifier
                        target_stencils[identifier] = key
                        coordinate_rows[identifier] = sum(
                            (
                                float(weight) * coordinate_rows[source_id]
                                for source_id, weight in key
                            ),
                            start=np.zeros((mesh.ambient_dimension,), dtype=np.float64),
                        )
                target_vertices.append(identifier)
            identifier = next_cell_id
            next_cell_id += 1
            child_ids.append(identifier)
            reference = tuple(
                tuple(Fraction(float(value)) for value in point) for point in child
            )
            reference_rows.append(reference)
            interval = cell.interval_index
            if cell.column_id is not None and split_axial and interval is not None:
                midpoint = float(np.mean(child[:, 2]))
                interval = 2 * interval + int(midpoint >= 0.5)
                schedule_changed.append(identifier)
            output.append(
                _TargetCell(
                    identifier,
                    cell.name,
                    cell.kind,
                    tuple(target_vertices),
                    cell.cell_class,
                    cell.column_id,
                    interval,
                    (cell.identifier,),
                    (Fraction(1, len(template.children)),),
                    reference,
                )
            )
        new_siblings.append(
            MixedSibling(
                cell.identifier,
                cell.name,
                cell.kind,
                tuple(int(value) for value in cell.vertices),
                tuple(child_ids),
                tuple(reference_rows),
                cell.cell_class,
                cell.column_id,
                cell.interval_index,
            )
        )
    target_used = {vertex for cell in output for vertex in cell.vertices}
    for identifier in target_used:
        if identifier not in target_stencils:
            target_stencils[identifier] = ((identifier, Fraction(1)),)
    # Keep only active output vertices; fresh IDs remain above the hierarchy high-water mark.
    target_stencils = {
        identifier: target_stencils[identifier] for identifier in target_used
    }
    target_coordinates = {
        identifier: coordinate_rows[identifier] for identifier in target_used
    }
    prior_retired = (
        _initial_root_retired_entities(mesh, root_mesh)
        if hierarchy is None
        else hierarchy.retired_entities
    )
    prescribed = _prescribed_entities(mesh, prior_retired)
    provisional = _target_edit(
        mesh,
        output,
        target_stencils,
        target_coordinates,
        prescribed,
    )
    probe, _, _ = assemble_topology_edit(
        mesh, provisional, numeric_version="mixed-adaptation-probe", bind_periodic=False
    )
    retired = _retired_entity_records(
        mesh,
        probe,
        prior_retired,
    )
    all_prescribed: dict[tuple[int, tuple[int, ...]], int] = {}
    for degree, key, identifier in (*prior_retired, *retired):
        previous = all_prescribed.setdefault((degree, key), identifier)
        if previous != identifier:
            raise ValueError("Mixed retired entity has conflicting source identities.")
    final_prescribed = _prescribed_entities(
        mesh,
        tuple(
            (degree, key, identifier)
            for (degree, key), identifier in sorted(all_prescribed.items())
        ),
    )
    provisional = _target_edit(
        mesh,
        output,
        target_stencils,
        target_coordinates,
        tuple(final_prescribed),
    )
    probe, _, _ = assemble_topology_edit(
        mesh,
        provisional,
        numeric_version="mixed-adaptation-final-probe",
        bind_periodic=False,
    )
    coarsening_maps = tuple(
        MixedCoarseningMap(
            record.parent_id,
            record.block_name,
            record.cell_kind,
            _immutable(np.asarray(record.child_ids, dtype=np.int64)),
            record.child_reference_vertices,
        )
        for record in sorted(candidates, key=lambda value: value.parent_id)
    )
    coarsening_partition = _mixed_coarsening_partition(coarsening_maps)
    exact_target_points = _target_root_points(
        output, source_charts, {record.parent_id for record in candidates}, root_points
    )
    relations = _entity_relations(
        mesh,
        probe,
        output,
        coarsen_sources,
        coarsening_maps,
    )
    edit = provisional._replace(relations=relations)
    target_mesh, _, _ = assemble_topology_edit(
        mesh, edit, numeric_version="mixed-adaptation-identity-check", bind_periodic=False
    )
    if target_mesh.topology_id != probe.topology_id:
        raise RuntimeError("Mixed topology edit changed while binding lineage relations.")
    reference_maps = _mixed_reference_maps(
        output, {value.parent_cell_id for value in coarsening_maps}
    )
    target_geometry = None
    exact_vertex_coordinates = None
    if source_geometry is not None:
        target_geometry = _mixed_target_geometry(
            mesh,
            source_geometry,
            target_mesh,
            output,
            reference_maps,
            coarsening_maps,
            coarsening_partition,
            hierarchy is None
            and root_mesh is not None
            and root_mesh.mesh_id == mesh.mesh_id,
        )
        retained_map = (
            None
            if source_geometry.source_map is None
            else _RetainedCoordinateBank(
                source_geometry.source_map.source_id,
                source_geometry.source_map.source_revision,
                source_geometry.source_map.source_map_id,
                source_geometry.source_coordinates(),
            )
        )
        if retained_map is not None:
            target_geometry = CellGeometrySpec(
                dict(
                    zip(
                        target_geometry.block_names, target_geometry.elements, strict=True
                    )
                ),
                dict(
                    zip(
                        target_geometry.block_names,
                        target_geometry.geometry_dofs,
                        strict=True,
                    )
                ),
                target_geometry.coordinates,
                restriction_source=target_geometry.restriction_source,
                source_map=retained_map,
                source_mesh=target_mesh,
            )
        exact_vertex_coordinates = _exact_target_vertex_coordinates(
            target_geometry,
            target_mesh,
            np.asarray(edit.vertex_global_ids, dtype=np.int64),
        )
        edit = edit._replace(
            coordinates=np.asarray(
                [[float(value) for value in row] for row in exact_vertex_coordinates],
                dtype=np.float64,
            )
        )
        target_mesh, _, _ = assemble_topology_edit(
            mesh,
            edit,
            numeric_version="mixed-adaptation-mapped-geometry",
            bind_periodic=False,
        )
        if retained_map is not None:
            target_geometry = CellGeometrySpec(
                dict(
                    zip(
                        target_geometry.block_names, target_geometry.elements, strict=True
                    )
                ),
                dict(
                    zip(
                        target_geometry.block_names,
                        target_geometry.geometry_dofs,
                        strict=True,
                    )
                ),
                target_geometry.coordinates,
                restriction_source=target_geometry.restriction_source,
                source_map=retained_map,
                source_mesh=target_mesh,
            )

    all_vertex_points = dict(root_points)
    for identifier in sorted(target_used):
        point = exact_target_points[identifier]
        previous = all_vertex_points.get(identifier)
        if previous is not None and previous != point:
            raise ValueError("A mixed vertex identity has conflicting root ancestry.")
        all_vertex_points[identifier] = point
    vertex_points = tuple(sorted(all_vertex_points.items()))
    removed_parent_ids = {record.parent_id for record in candidates}
    old_siblings = tuple(
        record
        for record in (() if hierarchy is None else hierarchy.siblings)
        if record.parent_id not in removed_parent_ids
    )
    successor_siblings = tuple((*old_siblings, *new_siblings))
    active_cells = tuple(
        MixedActiveCell(
            cell.identifier,
            cell.name,
            cell.kind,
            cell.vertices,
            cell.cell_class,
            cell.column_id,
            cell.interval_index,
        )
        for cell in sorted(output, key=lambda value: value.identifier)
    )
    if layer_columns is None:
        layer_policy = None if hierarchy is None else hierarchy.layer_policy
    else:
        layer_policy = (
            layer_columns.hard_first_thickness,
            layer_columns.axial_refinement,
            layer_columns.allow_schedule_change,
        )
    successor = MixedAdaptationHierarchy(
        successor_siblings,
        active_cells,
        forest,
        vertex_points,
        retired,
        layer_policy,
        next_cell_id,
        next_vertex_id,
        (
            None
            if mesh.periodic_topology is None
            else update_periodic_quotient_history(
                mesh,
                None if hierarchy is None else hierarchy.periodic_quotient_history,
                root=(
                    root_mesh
                    if root_mesh is not None and root_mesh.periodic_topology is not None
                    else None
                ),
            )
        ),
        target_mesh.lifted_topology_id,
    )
    if mesh.periodic_topology is not None:
        if exact_vertex_coordinates is None or target_geometry is None:
            raise ValueError("Periodic mixed adaptation requires exact source geometry.")
        exact_source = None
        if any(
            Fraction(float(value)) != value
            for row in exact_vertex_coordinates
            for value in row
        ):
            identifiers = np.asarray(edit.vertex_global_ids, dtype=np.int64)
            exact_source = ExactPlcCellGeometrySource(
                "periodic-quotient-chart",
                mesh.mesh_id,
                identifiers,
                exact_vertex_coordinates,
                vertex_source_entity_ids=(None,) * identifiers.size,
                vertex_occurrence_paths=((),) * identifiers.size,
            )
        edit = edit._replace(
            periodic_witness=PeriodicTopologyEditWitness(
                mesh.mesh_id,
                target_mesh.lifted_topology_id,
                successor,
                exact_source,
                target_geometry,
            )
        )
    transfer = _cell_transfer(source_ids, output)
    all_refined = {record.parent_id for record in new_siblings}
    requested_done = np.asarray(sorted(requested_refine & all_refined), dtype=np.int64)
    closure_ids = np.asarray(sorted(all_refined - requested_refine), dtype=np.int64)
    coarsened_ids = np.asarray(sorted(removed), dtype=np.int64)
    rejected_ids = np.asarray(sorted(rejected), dtype=np.int64)
    changed_ids = np.asarray(sorted(schedule_changed), dtype=np.int64)
    work = (
        sum(template.children.size for template in chosen)
        + len(interfaces)
        + closure_steps
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "mixed-adaptation-evidence",
            "requested_refine": np.asarray(sorted(requested_refine), dtype=np.int64),
            "refined": requested_done,
            "coarsened": coarsened_ids,
            "rejected_coarsening": rejected_ids,
            "closure": closure_ids,
            "schedule_changed": changed_ids,
            "closure_steps": closure_steps,
            "interfaces": len(interfaces),
            "work_units": work,
            "hierarchy": successor.hierarchy_id,
        }
    )
    evidence = MixedAdaptationEvidence(
        _immutable(np.asarray(sorted(requested_refine), dtype=np.int64)),
        _immutable(requested_done),
        _immutable(coarsened_ids),
        _immutable(rejected_ids),
        _immutable(closure_ids),
        _immutable(changed_ids),
        closure_steps,
        len(interfaces),
        work,
        evidence_id,
    )

    return MixedAdaptationOutcome(
        edit,
        successor,
        evidence,
        transfer,
        coarsening_maps,
        reference_maps,
        target_geometry,
        exact_vertex_coordinates,
        coarsening_partition,
    )


__all__ = [
    "MixedAdaptationEvidence",
    "MixedAdaptationHierarchy",
    "MixedAdaptationOutcome",
    "MixedCoarseningMap",
    "MixedForestRoot",
    "MixedLayerColumns",
    "MixedSibling",
    "adapt_mixed_mesh",
]
