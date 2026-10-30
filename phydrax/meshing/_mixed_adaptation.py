#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conforming marked adaptation for tetrahedral and hybrid volume meshes."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import NamedTuple, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .._fingerprint import canonical_fingerprint
from .._meshcore import MixedElementKind, mixed_template, mixed_template_counts
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import (
    CellGeometryCoordinateMap,
    CellGeometrySpec,
    coordinate_reference_chain,
    PerCellRestrictedCellGeometryElement,
    restriction_chart_arguments,
    ScalarCoordinateElement,
)
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization._coordinate_enclosure import (
    axes,
    compose_expression,
    constant,
    CoordinateSourceBank,
    Expression,
    expression_equal,
    scale,
    sum_polynomials,
)
from ..discretization._nested_reference import nested_cell_reference_maps
from ..discretization._nested_reference_types import (
    ReferenceCellMapWitnesses,
    validate_reference_cell_map_witnesses,
)
from ..discretization._reference_cell import (
    facet_orientation_between,
    reference_cell_topology,
)
from ..linalg._exact import solve_exact
from ._lineage import EntityLineageKind
from ._topology_edit import (
    CellTopologyEdit,
    EntityRelations,
    PrescribedEntityIds,
    SharedFaceWitnesses,
    TopologyEditBlock,
    assemble_topology_edit,
    entity_keys,
    key_rows,
)


_PointKey = tuple[tuple[int, Fraction], ...]
_FaceSignature = tuple[tuple[_PointKey, ...], ...]


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
    """Retain exact coefficient values and source identity under a cell restriction."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)
    exact_coordinates: CoordinateSourceBank | None = eqx.field(static=True)

    def __init__(
        self,
        source_map: CellGeometryCoordinateMap,
        exact_coordinates: CoordinateSourceBank | None,
        /,
    ) -> None:
        if not isinstance(source_map, CellGeometryCoordinateMap):
            raise TypeError("source_map must satisfy CellGeometryCoordinateMap.")
        self.source_id = source_map.source_id
        self.source_revision = source_map.source_revision
        self.source_map_id = source_map.source_map_id
        self.exact_coordinates = exact_coordinates

    def source_coordinates(self) -> CoordinateSourceBank | None:
        return self.exact_coordinates


class MixedActiveCell(NamedTuple):
    """Identity and class state needed to bind a hierarchy to its active mesh."""

    cell_id: int
    block_name: str
    cell_kind: str
    vertices: tuple[int, ...]
    cell_class: int
    column_id: int | None
    interval_index: int | None


@final
class MixedLayerColumns(StrictModule, NonTrainableState):
    """Explicit layer-cell ancestry and the permitted axial schedule changes."""

    __strict_contract__ = True
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
            raise ValueError("Layer columns need aligned unique cells and nonnegative ancestry.")
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

    __strict_contract__ = True
    siblings: tuple[MixedSibling, ...] = eqx.field(static=True)
    active_cells: tuple[MixedActiveCell, ...] = eqx.field(static=True)
    vertex_points: tuple[tuple[int, _PointKey], ...] = eqx.field(static=True)
    retired_entities: tuple[tuple[int, tuple[int, ...], int], ...] = eqx.field(static=True)
    layer_policy: tuple[bool, bool, bool] | None = eqx.field(static=True)
    next_cell_id: int = eqx.field(static=True)
    next_vertex_id: int = eqx.field(static=True)
    hierarchy_id: str = eqx.field(static=True)

    def __init__(
        self,
        siblings: tuple[MixedSibling, ...],
        active_cells: tuple[MixedActiveCell, ...],
        vertex_points: tuple[tuple[int, _PointKey], ...],
        retired_entities: tuple[tuple[int, tuple[int, ...], int], ...],
        layer_policy: tuple[bool, bool, bool] | None,
        next_cell_id: int,
        next_vertex_id: int,
        /,
    ) -> None:
        ordered_cells = tuple(sorted(active_cells, key=lambda value: value.cell_id))
        ordered_siblings = tuple(sorted(siblings, key=lambda value: value.parent_id))
        ordered_points = tuple(sorted(vertex_points, key=lambda value: value[0]))
        ordered_retired = tuple(sorted(retired_entities))
        if (
            len({value.cell_id for value in ordered_cells}) != len(ordered_cells)
            or len({value.parent_id for value in ordered_siblings}) != len(ordered_siblings)
            or len({value[0] for value in ordered_points}) != len(ordered_points)
            or len({(value[0], value[1]) for value in ordered_retired})
            != len(ordered_retired)
            or next_cell_id < 0
            or next_vertex_id < 0
        ):
            raise ValueError("Mixed-adaptation hierarchy identities are malformed.")
        high_cell = max(
            (value.cell_id for value in ordered_cells),
            default=-1,
        )
        high_cell = max(
            high_cell,
            max((value.parent_id for value in ordered_siblings), default=-1),
            max((child for value in ordered_siblings for child in value.child_ids), default=-1),
        )
        high_vertex = max(
            max((vertex for value in ordered_cells for vertex in value.vertices), default=-1),
            max((vertex for value in ordered_siblings for vertex in value.parent_vertices), default=-1),
            max((value[0] for value in ordered_points), default=-1),
        )
        if next_cell_id <= high_cell or next_vertex_id <= high_vertex:
            raise ValueError("Mixed-adaptation hierarchy allocators must exceed issued IDs.")
        if layer_policy is not None and (
            len(layer_policy) != 3 or any(type(value) is not bool for value in layer_policy)
        ):
            raise ValueError("Mixed-adaptation layer policy must contain three bool values.")
        self.siblings = ordered_siblings
        self.active_cells = ordered_cells
        self.vertex_points = ordered_points
        self.retired_entities = ordered_retired
        self.layer_policy = layer_policy
        self.next_cell_id = next_cell_id
        self.next_vertex_id = next_vertex_id
        self.hierarchy_id = canonical_fingerprint(
            {
                "kind": "mixed-adaptation-hierarchy",
                "siblings": _sibling_payload(ordered_siblings),
                "active_cells": ordered_cells,
                "vertex_points": _point_bank_payload(ordered_points),
                "retired_entities": ordered_retired,
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
                tuple(tuple(_fraction_payload(coordinate) for coordinate in point) for point in child)
                for child in value.child_reference_vertices
            ),
            value.cell_class,
            value.column_id,
            value.interval_index,
        )
        for value in siblings
    )


def _point_key(kind: str, vertices: NDArray[np.int64], point: NDArray[np.float64], /) -> _PointKey:
    coordinates = tuple(Fraction(float(value)) for value in point)
    x, y = coordinates[:2]
    if kind == "tetrahedron":
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

def _template_facts(cell: _Cell, template: _Template, /) -> _TemplateFacts:
    topology = reference_cell_topology(cell.kind)
    child_points = tuple(
        tuple(_point_key(cell.kind, cell.vertices, point) for point in child)
        for child in template.children
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
        tuple(sorted((int(point[0][0]), int(point[1][0]))))
        for points in child_points
        for point in points
        if len(point) == 2
        and tuple(sorted((int(point[0][0]), int(point[1][0])))) in local_edges
    )
    return _TemplateFacts(
        child_points,
        tuple(tuple(sorted(values)) for values in facets),
        split_edges,
    )


def _native_kind(kind: str, /) -> MixedElementKind:
    match kind:
        case "tetrahedron":
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
    if kind == "tetrahedron":
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
    row = 0
    for block in mesh.blocks:
        if not isinstance(block, CellBlock) or block.cell_kind not in (
            "tetrahedron",
            "prism",
            "pyramid",
            "hexahedron",
        ):
            raise ValueError(
                "Mixed adaptation requires fixed tetrahedron, prism, pyramid, and hexahedron blocks."
            )
        vertices = np.asarray(block.vertices, dtype=np.int64)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        for cell_id, cell_vertices in zip(identifiers, vertices, strict=True):
            layer = layer_by_id.get(int(cell_id))
            result.append(
                _Cell(
                    int(cell_id),
                    block.name,
                    block.cell_kind,
                    cell_vertices.copy(),
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
    owners: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = defaultdict(list)
    for row, cell in enumerate(cells):
        facets = reference_cell_topology(cell.kind).entities[2]
        for facet, local in enumerate(facets):
            loop = tuple(int(cell.vertices[index]) for index in local)
            owners[tuple(sorted(loop))].append((row, facet, loop))
    if any(len(values) > 2 for values in owners.values()):
        raise MixedTemplateClosureError("Mixed adaptation requires manifold shared faces.")
    return owners


def _face_signature(
    cell: _Cell, template: _Template, facet: int, cache: dict[tuple[int, int], _TemplateFacts], index: int, /
) -> _FaceSignature:
    key = (index, template.index)
    facts = cache.get(key)
    if facts is None:
        facts = _template_facts(cell, template)
        cache[key] = facts
    return facts.facets[facet]


def _close_templates(
    cells: list[_Cell],
    domains: list[list[_Template]],
    interfaces: list[tuple[int, int, int, int]],
) -> tuple[list[_Template], int]:
    cache: dict[tuple[int, int], _TemplateFacts] = {}
    pending: list[list[list[_Template]]] = [[values.copy() for values in domains]]
    steps = 0
    while pending:
        choices = pending.pop()
        steps += 1
        feasible = True
        changed = True
        while changed and feasible:
            changed = False
            for left, left_facet, right, right_facet in interfaces:
                right_signatures = {
                    _face_signature(cells[right], value, right_facet, cache, right)
                    for value in choices[right]
                }
                retained_left = [
                    value
                    for value in choices[left]
                    if _face_signature(cells[left], value, left_facet, cache, left)
                    in right_signatures
                ]
                if not retained_left:
                    feasible = False
                    break
                if len(retained_left) != len(choices[left]):
                    choices[left] = retained_left
                    changed = True
                left_signatures = {
                    _face_signature(cells[left], value, left_facet, cache, left)
                    for value in choices[left]
                }
                retained_right = [
                    value
                    for value in choices[right]
                    if _face_signature(cells[right], value, right_facet, cache, right)
                    in left_signatures
                ]
                if not retained_right:
                    feasible = False
                    break
                if len(retained_right) != len(choices[right]):
                    choices[right] = retained_right
                    changed = True
        if not feasible:
            continue
        unresolved = [index for index, values in enumerate(choices) if len(values) > 1]
        if not unresolved:
            return [values[0] for values in choices], steps
        selected = min(
            unresolved,
            key=lambda index: (len(choices[index]), cells[index].identifier),
        )
        for candidate in reversed(choices[selected]):
            branch = [values.copy() for values in choices]
            branch[selected] = [candidate]
            pending.append(branch)
    raise MixedTemplateClosureError(
        "No family-preserving reference templates close the marked shared-face signatures."
    )


def _mapping_points(hierarchy: MixedAdaptationHierarchy | None, mesh: CellMesh, /) -> dict[int, _PointKey]:
    if hierarchy is None:
        return {
            int(identifier): ((int(identifier), Fraction(1)),)
            for identifier in np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        }
    points = dict(hierarchy.vertex_points)
    current = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    missing = [int(value) for value in current if int(value) not in points]
    if missing:
        raise ValueError("Mixed hierarchy does not retain ancestry for every source vertex.")
    return points


def _root_key(point: _PointKey, root_points: dict[int, _PointKey], /) -> _PointKey:
    accumulated: dict[int, Fraction] = defaultdict(Fraction)
    for identifier, coefficient in point:
        for root, weight in root_points[identifier]:
            accumulated[root] += coefficient * weight
    return tuple(sorted((identifier, value) for identifier, value in accumulated.items() if value))


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
        removed_children = {child for record in accepted.values() for child in record.child_ids}
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
    accepted_children = {child for record in accepted.values() for child in record.child_ids}
    rejected.update(marks - accepted_children)
    return list(accepted.values()), rejected, accepted_children


def _entity_relations(
    source: CellMesh,
    probe: CellMesh,
    targets: list[_TargetCell],
    vertex_stencils: dict[int, _PointKey],
    root_points: dict[int, _PointKey],
    coarsen_sources: dict[int, tuple[int, ...]],
    /,
) -> tuple[EntityRelations, ...]:
    dimension = source.topological_dimension
    relations: list[EntityRelations] = []
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
        source_vertices = [tuple(int(value) for value in row if value >= 0) for row in source_keys]
        target_vertices = [tuple(int(value) for value in row if value >= 0) for row in target_keys]
        source_root_supports = [
            frozenset(root for vertex in row for root, _ in root_points[vertex])
            for row in source_vertices
        ]
        target_root_supports = [
            frozenset(
                root
                for vertex in row
                for root, _ in _root_key(vertex_stencils[vertex], root_points)
            )
            for row in target_vertices
        ]
        refined: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
        coarsened: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
        source_containing: dict[int, list[int]] = defaultdict(list)
        source_supported: dict[int, list[int]] = defaultdict(list)
        for row, (vertices, support) in enumerate(
            zip(source_vertices, source_root_supports, strict=True)
        ):
            for vertex in support:
                source_containing[vertex].append(row)
                source_supported[vertex].append(row)
        for target_key, target_support in zip(
            target_vertices, target_root_supports, strict=True
        ):
            candidate_rows = (
                source_containing.get(min(target_support), ()) if target_support else ()
            )
            for row in candidate_rows:
                if (
                    target_support.issubset(source_root_supports[row])
                    and tuple(target_key) != source_vertices[row]
                ):
                    refined.append(
                        (source_keys[row], target_key, int(EntityLineageKind.REFINED_FROM))
                    )
            coarsen_rows = sorted(
                {
                    row
                    for root in target_support
                    for row in source_supported.get(root, ())
                    if source_root_supports[row].issubset(target_support)
                    and source_vertices[row] != tuple(target_key)
                }
            )
            for row in coarsen_rows:
                coarsened.append(
                    (source_keys[row], target_key, int(EntityLineageKind.COARSENED_INTO))
                )
        entries = refined + coarsened
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


def _shared_faces(blocks: tuple[TopologyEditBlock, ...], vertex_ids: NDArray[np.int64], /) -> SharedFaceWitnesses:
    owners: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = defaultdict(list)
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
    cells: list[_Cell],
    vertex_points: dict[int, _PointKey],
    /,
) -> None:
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
        raise ValueError("Mixed-adaptation hierarchy is not bound to this exact active mesh.")
    if not set(vertex_points).issuperset(
        int(value) for value in np.asarray([vertex for cell in cells for vertex in cell.vertices])
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
        if any(facts.facets[facet] != identity_facts.facets[facet] for facet in protected_facets):
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
    vertex_ids = np.asarray(sorted({value for cell in target_cells for value in cell.vertices}), dtype=np.int64)
    vertex_rows = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    coordinates = np.asarray([coordinates_by_id[int(identifier)] for identifier in vertex_ids], dtype=np.float64)
    width = max((len(target_stencils[int(identifier)]) for identifier in vertex_ids), default=1)
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
        selected = [cell for cell in target_cells if cell.name == name and cell.kind == kind]
        if selected:
            selected.sort(key=lambda value: value.identifier)
            blocks.append(
                TopologyEditBlock(
                    name,
                    kind,
                    kind,
                    np.asarray(
                        [[vertex_rows[value] for value in cell.vertices] for cell in selected],
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
    unique = {(degree, key): identifier for degree, key, identifier in retained if (degree, key) not in restored}
    return tuple(sorted((degree, key, identifier) for (degree, key), identifier in unique.items()))


def _prescribed_entities(
    source: CellMesh,
    hierarchy: MixedAdaptationHierarchy | None,
    /,
) -> tuple[PrescribedEntityIds, ...]:
    if hierarchy is None:
        return ()
    records = []
    for degree in (1, 2):
        current = {
            tuple(int(value) for value in row if value >= 0)
            for row in entity_keys(source, degree)
        }
        values = [
            (key, identifier)
            for item_degree, key, identifier in hierarchy.retired_entities
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
            for source_id, weight in zip(target.source_ids, target.content_weights, strict=True)
        ),
        key=lambda value: (value[1], value[0]),
    )
    row_sources = np.asarray([value[0] for value in rows], dtype=np.int64)
    row_targets = np.asarray([value[1] for value in rows], dtype=np.int64)
    exact = tuple(value[2] for value in rows)
    weights = np.asarray([float(value) for value in exact], dtype=np.float64)
    target_ids = np.asarray(sorted({target.identifier for target in targets}), dtype=np.int64)
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
                raise ValueError("Direct mixed reference maps require one source-cell parent.")
            grouped[target.kind].append(target)
    witnesses = []
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
        )
        validate_reference_cell_map_witnesses(witness)
        witnesses.append(witness)
    return tuple(witnesses)


def _reference_map_arguments(
    witnesses: tuple[ReferenceCellMapWitnesses, ...], /
) -> dict[int, tuple[int, tuple[Expression, ...]]]:
    result: dict[int, tuple[int, tuple[Expression, ...]]] = {}
    for witness in witnesses:
        for mapping in nested_cell_reference_maps(witness):
            if mapping.arguments is None:
                raise ValueError("Mixed reference-map corners lack an exact chart.")
            result[mapping.target_cell_id] = (
                mapping.source_cell_id,
                mapping.arguments,
            )
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
    if cell_kind in ("tetrahedron", "prism"):
        basis = (0, 1, 2, 3)
    elif cell_kind in ("hexahedron", "pyramid"):
        basis = (0, 1, 3, 4)
    else:
        raise ValueError(f"Unsupported mixed coarsening family {cell_kind!r}.")
    if len(child_parent_vertices) != len(local) or any(
        len(point) != dimension for point in child_parent_vertices
    ):
        raise ValueError("Mixed sibling reference corners do not match their cell family.")
    local_matrix = tuple(
        tuple(local[basis[column]][row] - local[0][row] for column in range(dimension))
        for row in range(dimension)
    )
    parent_matrix = tuple(
        tuple(
            child_parent_vertices[basis[column]][row] - child_parent_vertices[0][row]
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
        if child_ids.ndim != 1 or child_ids.size != len(
            record.child_reference_vertices
        ):
            raise ValueError("Mixed coarsening reference rows do not align with child IDs.")
        for child_id_raw, exact_vertices in zip(
            child_ids, record.child_reference_vertices, strict=True
        ):
            child_id = int(child_id_raw)
            if child_id in seen_children:
                raise ValueError("Mixed coarsening maps repeat a source child cell.")
            seen_children.add(child_id)
            result[(record.parent_cell_id, child_id)] = (
                _exact_child_to_parent_affine_map(
                    record.parent_cell_kind, exact_vertices
                )
            )
    return result


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
    inverse = solve_exact(matrix, identity)
    variables = axes(dimension)
    return tuple(
        sum_polynomials(
            (
                constant(
                    -sum(
                        (inverse[row][column] * offset[column] for column in range(dimension)),
                        Fraction(0),
                    ),
                    dimension,
                ),
                *(
                    scale(variables[column], inverse[row][column])
                    for column in range(dimension)
                ),
            ),
            dimension,
        )
        for row in range(dimension)
    )


def _mixed_target_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    targets: list[_TargetCell],
    reference_maps: tuple[ReferenceCellMapWitnesses, ...],
    coarsening_maps: tuple[MixedCoarseningMap, ...],
    /,
) -> CellGeometrySpec:
    source_elements, source_routes, _ = source_geometry.resolve(source_mesh)
    source_locations = {
        int(identifier): (block_index, row)
        for block_index, block in enumerate(source_mesh.blocks)
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    target_by_id = {target.identifier: target for target in targets}
    direct_maps = _reference_map_arguments(reference_maps)
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
            chart = axes(reference_cell_topology(source_mesh.blocks[block_index].cell_kind).dimension)
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
                direct = direct_maps.get(int(target_id))
                if direct is None:
                    raise ValueError("Mixed target geometry lacks exact source-cell ancestry.")
                source_id, target_chart = direct
                root, source_chart_arguments, route = source_chart(source_id)
                chart = tuple(
                    compose_expression(value, target_chart)
                    for value in source_chart_arguments
                )
            else:
                root = None
                route = None
                chart = None
                for child_id, child_vertices in zip(
                    np.asarray(coarsening.child_cell_ids, dtype=np.int64),
                    coarsening.child_reference_vertices,
                    strict=True,
                ):
                    child_root, child_chart, child_route = source_chart(int(child_id))
                    child_matrix, child_offset = _exact_child_to_parent_affine_map(
                        coarsening.parent_cell_kind, child_vertices
                    )
                    parent_to_child = _parent_to_child_chart(child_matrix, child_offset)
                    candidate_chart = tuple(
                        compose_expression(value, parent_to_child)
                        for value in child_chart
                    )
                    if root is None:
                        root = child_root
                        route = child_route
                        chart = candidate_chart
                        source_id = int(child_id)
                    elif (
                        root.element_id != child_root.element_id
                        or not np.array_equal(route, child_route)
                        or any(
                            not expression_equal(first, second)
                            for first, second in zip(chart, candidate_chart, strict=True)
                        )
                    ):
                        raise ValueError(
                            "Mixed sibling source maps do not reconstruct one exact parent map."
                        )
                if root is None or route is None or chart is None:
                    raise ValueError("A mixed coarsening map has no source children.")
            if route.shape != (root.local_dof_count,):
                raise ValueError("Mixed source geometry roots do not match their DOF routes.")
            roots.append(root)
            source_ids.append(source_id)
            chart_rows.append(chart)
            route_rows.append(route)
        if not roots or any(value.element_id != roots[0].element_id for value in roots):
            raise ValueError("A mixed target block cannot combine unrelated geometry roots.")
        elements[block.name] = PerCellRestrictedCellGeometryElement(
            roots[0],
            block.cell_kind,
            target_ids,
            source_ids,
            chart_rows,
        )
        routes[block.name] = np.stack(route_rows).astype(np.int32, copy=False)
    return CellGeometrySpec(elements, routes, source_geometry.coordinates)


def _exact_target_vertex_coordinates(
    geometry: CellGeometrySpec,
    mesh: CellMesh,
    targets: list[_TargetCell],
    edit_vertex_ids: NDArray[np.int64],
    /,
) -> NDArray[np.float64]:
    target_by_id = {target.identifier: target for target in targets}
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    exact_coordinates: dict[int, tuple[object, ...]] = {}
    for block in mesh.blocks:
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        references = tuple(
            (target_by_id[int(identifier)].reference_vertices,)
            for identifier in identifiers
        )
        evaluated = geometry.evaluate_exact(mesh, identifiers, references)
        cells = np.asarray(block.vertices, dtype=np.int64)
        for row, (cell, points) in enumerate(zip(cells, evaluated, strict=True)):
            local_ids = vertex_ids[cell]
            for vertex_id, point in zip(local_ids, points[0], strict=True):
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
    return np.asarray(
        [
            [float(value) for value in exact_coordinates[int(identifier)]]
            for identifier in edit_vertex_ids
        ],
        dtype=np.float64,
    )

@final
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
    source_geometry: CellGeometrySpec | None = None,
) -> MixedAdaptationOutcome:
    """Build a face-conforming family-preserving refinement/coarsening edit.

    ``cell_classes`` is aligned with source block order and separates material,
    region, layer, and other organization classes without freezing the interface.
    Coarsening marks identify all active children of each restored sibling family.
    """
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError("Mixed volume adaptation requires a 3D mesh in 3D coordinates.")
    if mesh.periodic_topology is not None:
        raise ValueError("Mixed-family edits require an explicit periodic-orbit closure route.")
    if hierarchy is not None and not isinstance(hierarchy, MixedAdaptationHierarchy):
        raise TypeError("hierarchy must be MixedAdaptationHierarchy or None.")
    if layer_columns is not None and not isinstance(layer_columns, MixedLayerColumns):
        raise TypeError("layer_columns must be MixedLayerColumns or None.")
    if source_geometry is not None:
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("source_geometry must be CellGeometrySpec or None.")
        source_geometry.resolve(mesh)
    block_count = sum(block.cell_count for block in mesh.blocks)
    classes = (
        np.zeros((block_count,), dtype=np.int64)
        if cell_classes is None
        else np.asarray(cell_classes)
    )
    if classes.ndim != 1 or classes.shape != (block_count,) or classes.dtype.kind not in "iu":
        raise TypeError("cell_classes must be an integer vector aligned with source block order.")
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
        class_by_id = {value.cell_id: value.cell_class for value in hierarchy.active_cells}
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
    root_points = _mapping_points(hierarchy, mesh)
    if hierarchy is not None:
        _validate_hierarchy(hierarchy, cells, root_points)
        if layer_columns is not None:
            old_layers = {
                value.cell_id: (value.column_id, value.interval_index)
                for value in hierarchy.active_cells
                if value.column_id is not None
            }
            if old_layers != layer_values:
                raise ValueError("Layer-column ancestry differs from the bound hierarchy.")
            if hierarchy.layer_policy != (
                layer_columns.hard_first_thickness,
                layer_columns.axial_refinement,
                layer_columns.allow_schedule_change,
            ):
                raise ValueError("Layer-column permissions differ from the bound hierarchy.")
    refine = set(_integer_vector(refine_cell_ids, "refine_cell_ids").tolist())
    coarsen_marks = set(_integer_vector(coarsen_cell_ids, "coarsen_cell_ids").tolist())
    if refine & coarsen_marks:
        raise ValueError("A mixed cell cannot be marked for both refinement and coarsening.")
    if not refine.issubset(active_ids) or not coarsen_marks.issubset(active_ids):
        raise ValueError("Mixed adaptation marks must name current source cell IDs.")
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

    candidates, rejected, removed = _coarsening_candidates(
        cells,
        coarsen_marks,
        hierarchy,
        protected_vertices,
        protected_cells,
        root_points,
    )
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
        raise MixedTemplateClosureError("Mixed adaptation cannot remove every source cell.")
    if any(not set(cell.vertices.tolist()).issubset(coordinate_rows) for cell in working):
        raise ValueError("Restored mixed parent corners are absent from source coordinates.")

    owners = _mesh_facet_owners(working)
    interfaces = [
        (left[0], left[1], right[0], right[1])
        for values in owners.values()
        if len(values) == 2
        for left, right in (values,)
    ]
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
        split_axial = cell.kind in ("prism", "hexahedron") and any(
            np.ptp(child[:, 2]) < 1.0 for child in template.children
        )
        for child_index, (child, point_keys) in enumerate(
            zip(template.children, facts.points, strict=True)
        ):
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
                tuple(Fraction(float(value)) for value in point)
                for point in child
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
    target_stencils = {identifier: target_stencils[identifier] for identifier in target_used}
    target_coordinates = {identifier: coordinate_rows[identifier] for identifier in target_used}
    provisional = _target_edit(
        mesh,
        output,
        target_stencils,
        target_coordinates,
        prescribed,
    )
    probe, _, _ = assemble_topology_edit(
        mesh, provisional, numeric_version="mixed-adaptation-probe"
    )
    retired = _retired_entity_records(
        mesh,
        probe,
        () if hierarchy is None else hierarchy.retired_entities,
    )
    final_prescribed: list[PrescribedEntityIds] = []
    for degree in (1, 2):
        current = {
            tuple(int(value) for value in row if value >= 0)
            for row in entity_keys(mesh, degree)
        }
        values = [
            (key, identifier)
            for record_degree, key, identifier in retired
            if record_degree == degree and key not in current
        ]
        if values:
            width = max(len(key) for key, _ in values)
            keys = np.full((len(values), width), -1, dtype=np.int64)
            ids = np.empty((len(values),), dtype=np.int64)
            for row, (key, identifier) in enumerate(values):
                keys[row, : len(key)] = key
                ids[row] = identifier
            final_prescribed.append(PrescribedEntityIds(degree, keys, ids))
    provisional = _target_edit(
        mesh,
        output,
        target_stencils,
        target_coordinates,
        tuple(final_prescribed),
    )
    probe, _, _ = assemble_topology_edit(
        mesh, provisional, numeric_version="mixed-adaptation-final-probe"
    )
    relations = _entity_relations(
        mesh,
        probe,
        output,
        target_stencils,
        root_points,
        coarsen_sources,
    )
    edit = provisional._replace(relations=relations)
    target_mesh, _, _ = assemble_topology_edit(
        mesh, edit, numeric_version="mixed-adaptation-identity-check"
    )
    if target_mesh.topology_id != probe.topology_id:
        raise RuntimeError("Mixed topology edit changed while binding lineage relations.")
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
    reference_maps = _mixed_reference_maps(
        output, {value.parent_cell_id for value in coarsening_maps}
    )
    target_geometry = None
    if source_geometry is not None:
        target_geometry = _mixed_target_geometry(
            mesh,
            source_geometry,
            target_mesh,
            output,
            reference_maps,
            coarsening_maps,
        )
        retained_map = (
            None
            if source_geometry.source_map is None
            else _RetainedCoordinateBank(
                source_geometry.source_map,
                source_geometry.source_coordinates(),
            )
        )
        if retained_map is not None:
            target_geometry = CellGeometrySpec(
                dict(zip(target_geometry.block_names, target_geometry.elements, strict=True)),
                dict(zip(target_geometry.block_names, target_geometry.geometry_dofs, strict=True)),
                target_geometry.coordinates,
                source_map=retained_map,
                source_mesh=target_mesh,
            )
        exact_coordinates = _exact_target_vertex_coordinates(
            target_geometry,
            target_mesh,
            output,
            np.asarray(edit.vertex_global_ids, dtype=np.int64),
        )
        edit = edit._replace(coordinates=exact_coordinates)
        target_mesh, _, _ = assemble_topology_edit(
            mesh, edit, numeric_version="mixed-adaptation-mapped-geometry"
        )
        if retained_map is not None:
            target_geometry = CellGeometrySpec(
                dict(zip(target_geometry.block_names, target_geometry.elements, strict=True)),
                dict(zip(target_geometry.block_names, target_geometry.geometry_dofs, strict=True)),
                target_geometry.coordinates,
                source_map=retained_map,
                source_mesh=target_mesh,
            )

    vertex_points = tuple(
        (identifier, _root_key(target_stencils[identifier], root_points))
        for identifier in sorted(target_used)
    )
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
        vertex_points,
        retired,
        layer_policy,
        next_cell_id,
        next_vertex_id,
    )
    transfer = _cell_transfer(source_ids, output)
    all_refined = {record.parent_id for record in new_siblings}
    requested_done = np.asarray(sorted(refine & all_refined), dtype=np.int64)
    closure_ids = np.asarray(sorted(all_refined - refine), dtype=np.int64)
    coarsened_ids = np.asarray(sorted(removed), dtype=np.int64)
    rejected_ids = np.asarray(sorted(rejected), dtype=np.int64)
    changed_ids = np.asarray(sorted(schedule_changed), dtype=np.int64)
    work = sum(template.children.size for template in chosen) + len(interfaces) + closure_steps
    evidence_id = canonical_fingerprint(
        {
            "kind": "mixed-adaptation-evidence",
            "requested_refine": np.asarray(sorted(refine), dtype=np.int64),
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
        _immutable(np.asarray(sorted(refine), dtype=np.int64)),
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
    )
__all__ = [
    "MixedAdaptationEvidence",
    "MixedAdaptationHierarchy",
    "MixedAdaptationOutcome",
    "MixedCoarseningMap",
    "MixedLayerColumns",
    "MixedSibling",
    "adapt_mixed_mesh",
]
