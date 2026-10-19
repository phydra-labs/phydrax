#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conforming family-preserving reference templates and sibling restoration.

Closure compares oriented *subfaces*, not just split-edge masks. A finite-domain
constraint solve selects compatible templates across every affected interface.
The candidate and hierarchy are immutable until the adaptation transaction accepts them.
Column inversion uses current physical-column membership, not a census inferred
from possibly incomplete sibling records. Only authenticated unchanged SCI cells
are excluded from that cohort; unknown or incompatible intervals retain the fine mesh.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from typing import NamedTuple, Self

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

import phydrax._meshcore as _owner_phydrax__meshcore
import phydrax.discretization._cell_geometry_transition_types as transition_types_owner
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.meshing._entity_keys as entity_keys_owner
import phydrax.meshing._lineage_records as _dependency_phydrax_meshing__lineage
import phydrax.meshing._periodic as _dependency_phydrax_meshing__periodic
import phydrax.meshing._topology_edit as _dependency_phydrax_meshing__topology_edit

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import NativeBatchTicket, NativeExecutionScope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..discretization import CellBlock, CellMesh
from ..discretization._reference_cell import (
    facet_orientation_between,
    reference_cell_topology,
)
from ..geometry._rational import rational_vector
from ..typing import checked, ConvertibleToArray
from ._contracts import MeshingLimits
from ._measurements import NativeExecutionRecord


class MixedLayerColumns(StrictModule, NonTrainableState):
    """Explicit ancestry and permitted axial changes, aligned with source IDs.

    Axial refinement is never inferred from wall distance. Hard first intervals
    cannot split unless ``allow_schedule_change`` explicitly selects a new schedule.
    """

    cell_ids: tuple[int, ...] = eqx.field(static=True)
    column_ids: tuple[int, ...] = eqx.field(static=True)
    interval_indices: tuple[int, ...] = eqx.field(static=True)
    hard_first_thickness: bool = eqx.field(static=True)
    axial_refinement: bool = eqx.field(static=True)
    allow_schedule_change: bool = eqx.field(static=True)
    columns_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_ids: ConvertibleToArray,
        column_ids: ConvertibleToArray,
        interval_indices: ConvertibleToArray,
        *,
        hard_first_thickness: bool = True,
        axial_refinement: bool = False,
        allow_schedule_change: bool = False,
    ) -> None:
        ids = np.asarray(cell_ids, dtype=np.int64)
        columns = np.asarray(column_ids, dtype=np.int64)
        intervals = np.asarray(interval_indices, dtype=np.int64)
        if (
            ids.ndim != 1
            or columns.shape != ids.shape
            or intervals.shape != ids.shape
            or np.any(ids < 0)
            or np.any(columns < 0)
            or np.any(intervals < 0)
            or np.unique(ids).size != ids.size
        ):
            raise ValueError(
                "MixedLayerColumns.ancestry must contain unique cell IDs and aligned nonnegative columns and intervals; got invalid arrays"
            )
        if any(
            not isinstance(flag, (bool, np.bool_))
            for flag in (hard_first_thickness, axial_refinement, allow_schedule_change)
        ):
            raise TypeError(
                "MixedLayerColumns.permissions must be bool values; got an invalid flag"
            )
        order = np.argsort(ids, kind="stable")
        ids, columns, intervals = ids[order], columns[order], intervals[order]
        self.cell_ids = tuple(int(value) for value in ids)
        self.column_ids = tuple(int(value) for value in columns)
        self.interval_indices = tuple(int(value) for value in intervals)
        self.hard_first_thickness = bool(hard_first_thickness)
        self.axial_refinement = bool(axial_refinement)
        self.allow_schedule_change = bool(allow_schedule_change)
        self.columns_id = canonical_fingerprint(
            {
                "kind": "mixed-layer-columns",
                "cells": self.cell_ids,
                "columns": self.column_ids,
                "intervals": self.interval_indices,
                "hard_first_thickness": self.hard_first_thickness,
                "axial_refinement": self.axial_refinement,
                "allow_schedule_change": self.allow_schedule_change,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: MixedLayerColumns, /) -> Self:
        """Rebuild layer ancestry through its input validation and fingerprint."""
        if type(value) is not cls:
            raise TypeError(
                f"MixedLayerColumns.value must have type {cls.__name__} without subclass substitution; got {type(value).__name__}"
            )
        rebuilt = cls(
            value.cell_ids,
            value.column_ids,
            value.interval_indices,
            hard_first_thickness=value.hard_first_thickness,
            axial_refinement=value.axial_refinement,
            allow_schedule_change=value.allow_schedule_change,
        )
        if rebuilt.columns_id != value.columns_id:
            raise ValueError(
                f"MixedLayerColumns.columns_id must match rebuilt ancestry; got {value.columns_id}"
            )
        return rebuilt


class Sibling(NamedTuple):
    parent_id: int
    block_name: str
    cell_kind: str
    parent_vertices: tuple[int, ...]
    child_ids: tuple[int, ...]
    references: tuple[tuple[tuple[float, ...], ...], ...]
    cell_class: int
    layer_column: int | None
    layer_interval: int | None
    periodic_roots: tuple[int, ...]
    periodic_shifts: tuple[tuple[int, ...], ...]


class MixedAdaptationHierarchy(StrictModule, NonTrainableState):
    """Persistent template siblings, parent scientific identities and classes."""

    records: tuple[Sibling, ...] = eqx.field(static=True)
    next_cell_id: int = eqx.field(static=True)
    next_vertex_id: int = eqx.field(static=True)
    hierarchy_id: str = eqx.field(static=True)
    retired_entities: tuple[tuple[int, tuple[int, ...], int], ...] = eqx.field(
        static=True
    )
    quotient_entities: tuple[
        _dependency_phydrax_meshing__topology_edit.PeriodicEntityIdentityBank, ...
    ] = eqx.field(static=True)
    identity_cells: tuple[tuple[int, str, str, tuple[int, ...], int], ...] = eqx.field(
        static=True
    )
    layer_columns: MixedLayerColumns | None

    @checked
    def __init__(
        self,
        records: tuple[Sibling, ...] = (),
        *,
        next_cell_id: int = 0,
        next_vertex_id: int = 0,
        retired_entities: tuple[tuple[int, tuple[int, ...], int], ...] = (),
        quotient_entities: tuple[
            _dependency_phydrax_meshing__topology_edit.PeriodicEntityIdentityBank, ...
        ] = (),
        layer_columns: MixedLayerColumns | None = None,
        identity_cells: tuple[tuple[int, str, str, tuple[int, ...], int], ...] = (),
    ) -> None:
        if next_cell_id < 0 or next_vertex_id < 0:
            raise ValueError(
                "MixedAdaptationHierarchy.next_ids must be nonnegative; got a negative allocator identity"
            )
        if len({record.parent_id for record in records}) != len(records):
            raise ValueError(
                "MixedAdaptationHierarchy.records must have unique parent IDs; got duplicate parents"
            )
        ordered = tuple(sorted(records, key=lambda record: record.parent_id))
        retired = tuple(sorted(retired_entities))
        if len({row[0] for row in identity_cells}) != len(identity_cells):
            raise ValueError(
                "MixedAdaptationHierarchy.identity_cells must have unique cell IDs; got duplicate IDs"
            )
        if {row[0] for row in identity_cells} & {
            child for record in ordered for child in record.child_ids
        }:
            raise ValueError(
                "MixedAdaptationHierarchy.identity_cells must exclude refinement siblings; got overlapping IDs"
            )
        self.records = ordered
        self.next_cell_id = next_cell_id
        self.next_vertex_id = next_vertex_id
        self.retired_entities = retired
        self.quotient_entities = quotient_entities
        self.layer_columns = layer_columns
        self.identity_cells = tuple(sorted(identity_cells))
        self.hierarchy_id = canonical_fingerprint(
            {
                "kind": "mixed-template-hierarchy",
                "records": ordered,
                "next_cell_id": next_cell_id,
                "next_vertex_id": next_vertex_id,
                "retired_entities": self.retired_entities,
                "quotient_entities": quotient_entities,
                "layer_columns": None
                if layer_columns is None
                else layer_columns.columns_id,
                "identity_cells": self.identity_cells,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: MixedAdaptationHierarchy, /) -> Self:
        """Rebuild hierarchy state through its constructor validation."""
        if type(value) is not cls:
            raise TypeError(
                f"MixedAdaptationHierarchy.value must have type {cls.__name__} without subclass substitution; got {type(value).__name__}"
            )
        rebuilt = cls(
            value.records,
            next_cell_id=value.next_cell_id,
            next_vertex_id=value.next_vertex_id,
            retired_entities=value.retired_entities,
            quotient_entities=value.quotient_entities,
            layer_columns=value.layer_columns,
            identity_cells=value.identity_cells,
        )
        if rebuilt.hierarchy_id != value.hierarchy_id:
            raise ValueError(
                f"MixedAdaptationHierarchy.hierarchy_id must match rebuilt state; got {value.hierarchy_id}"
            )
        return rebuilt


class MixedAdaptationEvidence(NamedTuple):
    refined_cell_ids: np.ndarray
    coarsened_cell_ids: np.ndarray
    rejected_coarsening_ids: np.ndarray
    closure_cell_ids: np.ndarray
    closure_steps: int
    schedule_changed_cell_ids: np.ndarray
    work_units: int
    execution_evidence: NativeExecutionRecord | None

    @property
    def evidence_id(self) -> str:
        record = self.execution_evidence
        execution = (
            None
            if record is None
            else (
                record.work_counters,
                record.memory_counters,
                record.status,
                record.host_storage_live_bytes_upper,
                record.host_storage_peak_bytes_upper,
                record.externally_charged_work,
                record.externally_charged_geometry_queries,
                record.native_primitive_queries,
            )
        )
        return canonical_fingerprint(
            {
                "kind": "mixed-adaptation-evidence",
                "refined": array_tree_fingerprint(self.refined_cell_ids),
                "coarsened": array_tree_fingerprint(self.coarsened_cell_ids),
                "rejected_coarsening": array_tree_fingerprint(
                    self.rejected_coarsening_ids
                ),
                "closure": array_tree_fingerprint(self.closure_cell_ids),
                "closure_steps": self.closure_steps,
                "schedule_changes": array_tree_fingerprint(
                    self.schedule_changed_cell_ids
                ),
                "work_units": self.work_units,
                "execution": array_tree_fingerprint(execution),
            }
        )


class MixedAdaptationOutcome(NamedTuple):
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    hierarchy: MixedAdaptationHierarchy
    evidence: MixedAdaptationEvidence


class MixedTemplateClosureError(RuntimeError):
    """No permitted finite template closes the marked interface signatures."""


class _Template(NamedTuple):
    name: str
    children: np.ndarray
    interior_diagonal: tuple[tuple[int, int], tuple[int, int]] | None


def _reference(kind: str) -> np.ndarray:
    return np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)


def _templates(
    kind: str,
    axial: bool,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ticket: NativeBatchTicket,
) -> tuple[_Template, ...]:

    with ticket.native_target():
        references = _owner_phydrax__meshcore.mixed_refinement_templates(
            kind, axial=axial
        )
    order = list(range(len(references)))
    diagonals: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    if kind == "tetrahedron":
        # Red templates share all boundary subdivisions; only their interior
        # midpoint diagonal differs. The scientific cell IDs choose that action.
        order = [0, 8, 9, 10, 7, *range(1, 7)]
        diagonals = {8: ((0, 1), (2, 3)), 9: ((0, 2), (1, 3)), 10: ((0, 3), (1, 2))}
    elif axial and reference_cell_topology(kind).dimension == 3:
        order = [
            0,
            *sorted(
                range(1, len(references)),
                key=lambda index: (
                    not np.any(np.ptp(references[index][..., 2], axis=1) < 1.0)
                ),
            ),
        ]
    return tuple(
        _Template(
            "identity" if index == 0 else f"template-{index}",
            references[native_index],
            diagonals.get(native_index),
        )
        for index, native_index in enumerate(order)
    )


def _interior_diagonal_key(
    diagonal: tuple[tuple[int, int], tuple[int, int]], vertices: np.ndarray
) -> tuple[tuple[int, ...], ...]:
    """Return lexicographically ordered opposite midpoint-edge identities."""
    return tuple(
        sorted(
            tuple(sorted((int(vertices[first]), int(vertices[second]))))
            for first, second in diagonal
        )
    )


type _FaceSignature = tuple[
    tuple[
        _dependency_phydrax_meshing__periodic.ConstructionPointKey
        | _dependency_phydrax_meshing__periodic.PeriodicConstructionPointKey,
        ...,
    ],
    ...,
]


def _point_keys(
    kind: str,
    ids: np.ndarray,
    points: np.ndarray,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_dependency_phydrax_meshing__periodic.ConstructionPointKey, ...]:
    """Build construction identities from rational reference fractions."""
    if kind == "triangle":
        topology = reference_cell_topology(kind)
        if (
            ids.ndim != 1
            or ids.shape != (len(topology.vertices),)
            or points.ndim != 2
            or points.shape[1] != topology.dimension
            or not np.all(np.isfinite(points))
            or np.any(points < 0.0)
            or np.any(np.sum(points, axis=1) > 1.0)
        ):
            raise ValueError(
                "mixed_adaptation.reference_corners must contain every simplex corner; got an incomplete corner set"
            )
    # Admit rational keys using input-derived Fraction bit/storage bounds.
    bits = max(
        (
            abs(first).bit_length() + second.bit_length()
            for point in points
            for value in point
            for first, second in (float(value).as_integer_ratio(),)
        ),
        default=1,
    )
    # The cube/simplex products have at most 2d+1 input occurrences; the
    # collapsed pyramid's three rational factors have at most 32.
    occurrences = 32 if kind == "pyramid" else 2 * points.shape[1] + 1
    digits = (
        occurrences * bits + sys.int_info.bits_per_digit - 1
    ) // sys.int_info.bits_per_digit
    fraction = 96 + 2 * (32 + digits * sys.int_info.sizeof_digit)
    integer = max(
        ((32 + (abs(int(value)).bit_length() + 7) // 8) for value in ids), default=32
    )
    key_bytes = 64 + ids.size * (3 * np.dtype(np.intp).itemsize + 64 + integer + fraction)
    _reserve_host_storage(
        points.shape[0] * (key_bytes + fraction * points.shape[1]),
        scope=scope,
        budget=budget,
    )
    result = []
    corners = reference_cell_topology(kind).vertices
    for point in points:
        values = rational_vector(point)
        x, y = values[:2]
        if kind == "quadrilateral":
            weights = ((1 - x) * (1 - y), x * (1 - y), x * y, (1 - x) * y)
        elif kind == "triangle":
            weights = (1 - x - y, x, y)
        else:
            z = values[2]
            if kind == "tetrahedron":
                weights = (1 - x - y - z, x, y, z)
            elif kind == "prism":
                weights = tuple(
                    value * height for height in (1 - z, z) for value in (1 - x - y, x, y)
                )
            elif kind == "hexahedron":
                weights = tuple(
                    (x if corner[0] else 1 - x)
                    * (y if corner[1] else 1 - y)
                    * (z if corner[2] else 1 - z)
                    for corner in corners
                )
            elif kind == "pyramid":
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
                raise ValueError(
                    "mixed_adaptation.cell_kind must have a supported corner map; got an unsupported reference basis"
                )
        result.append(
            tuple(
                sorted(
                    (int(identifier), weight)
                    for identifier, weight in zip(ids, weights, strict=True)
                    if weight
                )
            )
        )
    return tuple(result)


def _periodic_facet_signatures(
    orbits: _dependency_phydrax_meshing__periodic.PeriodicConstructionOrbits,
    cell: Cell,
    template: _Template,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_FaceSignature, ...]:
    topology = reference_cell_topology(cell.kind)
    facets = topology.entities[topology.dimension - 1]
    supports = [set(int(cell.vertices[index]) for index in local) for local in facets]
    anchors = [
        orbits.face(cell.vertices[np.asarray(local, dtype=np.int32)])[1]
        for local in facets
    ]
    faces: list[
        list[
            tuple[_dependency_phydrax_meshing__periodic.PeriodicConstructionPointKey, ...]
        ]
    ] = [[] for _ in facets]
    for child in template.children:
        points = _point_keys(cell.kind, cell.vertices, child, scope=scope, budget=budget)
        for face in facets:
            keys = tuple(points[index] for index in face)
            vertices = {identifier for key in keys for identifier, _ in key}
            for facet, support in enumerate(supports):
                if vertices.issubset(support):
                    faces[facet].append(
                        tuple(
                            sorted(orbits.point(key, anchors[facet])[0] for key in keys)
                        )
                    )
    return tuple(tuple(sorted(values)) for values in faces)


def _facet_signatures(
    kind: str,
    ids: np.ndarray,
    template: _Template,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_FaceSignature, ...]:
    """Return sorted child subfaces on each parent facet, in facet order."""
    topology = reference_cell_topology(kind)
    facets = topology.entities[topology.dimension - 1]
    supports = [set(int(ids[index]) for index in facet) for facet in facets]
    faces: list[
        list[tuple[_dependency_phydrax_meshing__periodic.ConstructionPointKey, ...]]
    ] = [[] for _ in facets]
    for child in template.children:
        points = _point_keys(kind, ids, child, scope=scope, budget=budget)
        for face in facets:
            keys = tuple(points[index] for index in face)
            vertices = {identifier for key in keys for identifier, _ in key}
            for facet, support in enumerate(supports):
                if vertices.issubset(support):
                    faces[facet].append(tuple(sorted(keys)))
    return tuple(tuple(sorted(values)) for values in faces)


class Cell(NamedTuple):
    identifier: int
    name: str
    kind: str
    vertices: np.ndarray
    cell_class: int


class _MixedClosure(StrictModule):
    chosen: list[_Template]
    steps: int
    work: int


def _closure(
    cells: list[Cell],
    domains: list[list[_Template]],
    maximum_steps: int,
    maximum_work_units: int,
    orbits: _dependency_phydrax_meshing__periodic.PeriodicConstructionOrbits
    | None = None,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedClosure:
    """Solve one host closure stage on the supplied native ticket."""
    # The chosen table is retained by the caller; only closure indices,
    # lazy signatures, and live backtracking copies occupy this workspace.
    with budget.temporary_scope():
        return _solve_closure(
            cells,
            domains,
            maximum_steps,
            maximum_work_units,
            orbits,
            scope=scope,
            budget=budget,
        )


class _ClosureInterfaces(StrictModule):
    interfaces: list[list[tuple[int, int]]]


def _prepare_closure_interfaces(
    cells: list[Cell],
    orbits: _dependency_phydrax_meshing__periodic.PeriodicConstructionOrbits | None,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _ClosureInterfaces:
    faces: dict[
        tuple[int, ...]
        | _dependency_phydrax_meshing__periodic.PeriodicConstructionPointKey,
        list[tuple[int, int]],
    ] = {}
    for index, cell in enumerate(cells):
        topology = reference_cell_topology(cell.kind)
        for facet, local in enumerate(topology.entities[topology.dimension - 1]):
            _reserve_host_storage(256 + 64 * len(local), scope=scope, budget=budget)
            corners = cell.vertices[np.asarray(local, dtype=np.int32)]
            key = (
                tuple(sorted(int(value) for value in corners))
                if orbits is None
                else orbits.face(corners)[0]
            )
            faces.setdefault(key, []).append((index, facet))
    if any(len(owners) > 2 for owners in faces.values()):
        raise MixedTemplateClosureError(
            "mixed_adaptation.mesh must have manifold shared faces; got a face with more than two owners"
        )
    _reserve_host_storage(
        64 + 2 * len(faces) * np.dtype(np.intp).itemsize, scope=scope, budget=budget
    )
    interfaces = [owners for owners in faces.values() if len(owners) == 2]
    return _ClosureInterfaces(interfaces)


class _ClosureSignature(StrictModule):
    value: tuple[_FaceSignature, ...]
    work: int


def _prepare_closure_signature(
    cell: Cell,
    template: _Template,
    work: int,
    maximum_work_units: int,
    orbits: _dependency_phydrax_meshing__periodic.PeriodicConstructionOrbits | None,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _ClosureSignature:
    work += template.children.shape[0] * len(
        reference_cell_topology(cell.kind).entities[
            reference_cell_topology(cell.kind).dimension - 1
        ]
    )
    if work > maximum_work_units:
        raise MixedTemplateClosureError(
            "mixed_adaptation.maximum_work_units must admit signature preparation; got a work budget refusal"
        )
    budget.reserve(
        template.children.shape[0]
        * len(
            reference_cell_topology(cell.kind).entities[
                reference_cell_topology(cell.kind).dimension - 1
            ]
        )
    )
    _reserve_host_storage(
        256
        + 64
        * template.children.shape[0]
        * len(
            reference_cell_topology(cell.kind).entities[
                reference_cell_topology(cell.kind).dimension - 1
            ]
        ),
        scope=scope,
        budget=budget,
    )
    value = (
        _facet_signatures(cell.kind, cell.vertices, template, scope=scope, budget=budget)
        if orbits is None
        else _periodic_facet_signatures(
            orbits, cell, template, scope=scope, budget=budget
        )
    )
    return _ClosureSignature(value, work)


# Recursive closure shares lazy signatures, monotone work/step charges, and ordered branch rollback as one invariant.
def _solve_closure(
    cells: list[Cell],
    domains: list[list[_Template]],
    maximum_steps: int,
    maximum_work_units: int,
    orbits: _dependency_phydrax_meshing__periodic.PeriodicConstructionOrbits | None,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedClosure:
    interfaces = _prepare_closure_interfaces(
        cells, orbits, scope=scope, budget=budget
    ).interfaces
    signatures: dict[tuple[int, int], tuple[_FaceSignature, ...]] = {}
    steps, work = 0, 0

    def signature(index: int, candidate: int, facet: int, /) -> _FaceSignature:
        nonlocal work
        if (index, candidate) not in signatures:
            prepared = _prepare_closure_signature(
                cells[index],
                domains[index][candidate],
                work,
                maximum_work_units,
                orbits,
                scope=scope,
                budget=budget,
            )
            work = prepared.work
            signatures[index, candidate] = prepared.value
        return signatures[index, candidate][facet]

    def preferred_assignment(choices: list[list[int]]) -> list[int] | None:
        nonlocal work
        _reserve_host_storage(
            64 + 2 * len(choices) * np.dtype(np.intp).itemsize, scope=scope, budget=budget
        )
        preferred = [values[0] for values in choices]
        for (left, lf), (right, rf) in interfaces:
            work += 2
            if work > maximum_work_units:
                raise MixedTemplateClosureError(
                    "mixed_adaptation.maximum_work_units must admit face closure; got a work budget refusal"
                )
            budget.reserve(2)
            if signature(left, preferred[left], lf) != signature(
                right, preferred[right], rf
            ):
                return None
        return preferred

    def solve(choices: list[list[int]]) -> list[int] | None:
        nonlocal steps, work
        preferred = preferred_assignment(choices)
        if preferred is not None:
            return preferred
        changed = True
        while changed:
            steps += 1
            if steps > maximum_steps:
                raise MixedTemplateClosureError(
                    "mixed_adaptation.maximum_closure_steps must admit face closure; got an iteration budget refusal"
                )
            budget.ticket.checkpoint()
            changed = False
            for (left, lf), (right, rf) in interfaces:
                work += len(choices[left]) + len(choices[right])
                if work > maximum_work_units:
                    raise MixedTemplateClosureError(
                        "mixed_adaptation.maximum_work_units must admit face closure; got a work budget refusal"
                    )
                budget.reserve(len(choices[left]) + len(choices[right]))
                for first, ff, second, sf in (
                    (left, lf, right, rf),
                    (right, rf, left, lf),
                ):
                    _reserve_host_storage(
                        256 + 64 * (len(choices[first]) + len(choices[second])),
                        scope=scope,
                        budget=budget,
                    )
                    allowed = {signature(second, value, sf) for value in choices[second]}
                    retained = [
                        value
                        for value in choices[first]
                        if signature(first, value, ff) in allowed
                    ]
                    if not retained:
                        return None
                    if retained != choices[first]:
                        choices[first] = retained
                        changed = True
        preferred = preferred_assignment(choices)
        if preferred is not None:
            return preferred
        unresolved = next(
            (index for index, values in enumerate(choices) if len(values) > 1), None
        )
        if unresolved is None:
            return [values[0] for values in choices]
        for candidate in choices[unresolved]:
            branch_bytes = (
                64
                + 2 * len(choices) * np.dtype(np.intp).itemsize
                + sum((64 + 16 * len(values)) for values in choices)
            )
            _reserve_host_storage(branch_bytes, scope=scope, budget=budget)
            branch = [values.copy() for values in choices]
            branch[unresolved] = [candidate]
            try:
                result = solve(branch)
            finally:
                del branch
                budget.release_temporary_storage(branch_bytes)
            if result is not None:
                return result
        return None

    _reserve_host_storage(
        64
        + 2 * len(domains) * np.dtype(np.intp).itemsize
        + sum(
            64 + 2 * len(values) * np.dtype(np.intp).itemsize + len(values) * 32
            for values in domains
        ),
        scope=scope,
        budget=budget,
    )
    result = solve([list(range(len(values))) for values in domains])
    if result is None:
        raise MixedTemplateClosureError(
            "mixed_adaptation.templates must match oriented shared-face subdivisions; got no compatible family-preserving assignment"
        )
    return _MixedClosure(
        [domains[index][value] for index, value in enumerate(result)], steps, work
    )


def _shared_faces(
    blocks: tuple[_dependency_phydrax_meshing__topology_edit.TopologyEditBlock, ...],
    vertex_ids: np.ndarray,
) -> _dependency_phydrax_meshing__topology_edit.SharedFaceWitnesses:
    owners: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = {}
    for block in blocks:
        for identifier, row in zip(block.cell_ids, block.cells, strict=True):
            topology = reference_cell_topology(block.cell_kind)
            for facet, local in enumerate(topology.entities[topology.dimension - 1]):
                loop = tuple(int(vertex_ids[row[index]]) for index in local)
                owners.setdefault(tuple(sorted(loop)), []).append(
                    (int(identifier), facet, loop)
                )
    paired = [value for value in owners.values() if len(value) == 2]
    permutations = np.full((len(paired), 4), -1, dtype=np.int32)
    for index, (left, right) in enumerate(paired):
        action = facet_orientation_between(left[2], right[2])
        permutations[index, : len(action.permutation)] = action.permutation
    return _dependency_phydrax_meshing__topology_edit.SharedFaceWitnesses(
        np.asarray([value[0][0] for value in paired], dtype=np.int64),
        np.asarray([value[0][1] for value in paired], dtype=np.int32),
        np.asarray([value[1][0] for value in paired], dtype=np.int64),
        np.asarray([value[1][1] for value in paired], dtype=np.int32),
        permutations,
    )


class CoarsenPatch(StrictModule):
    restored: list[Cell]
    removed: set[int]
    fine: list[int]
    parents: list[int]
    references: list[np.ndarray]
    source_support: dict[int, set[int]]


class _CoarsenCandidates(StrictModule):
    active: dict[int, Cell]
    candidates: list[tuple[Sibling, set[int]]]


def _admit_coarsen_candidates(
    cells: list[Cell],
    records: tuple[Sibling, ...],
    marks: set[int],
    protected: set[int],
    blocked: set[int],
    orbits: _dependency_phydrax_meshing__periodic.PeriodicConstructionOrbits | None,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CoarsenCandidates:
    active = {cell.identifier: cell for cell in cells}
    candidates = []
    for record in records:
        _reserve_host_storage(
            256 + 64 * len(record.child_ids), scope=scope, budget=budget
        )
        # A physical column is one refinement/coarsening cohort, even when its
        # intervals do not share a face (for example, separate material blocks).
        # Every live column interval participates unless its unchanged SCI
        # signature is authenticated; missing ancestry cannot hide an interval.
        siblings = set(record.child_ids)
        if (
            not siblings.issubset(marks)
            or not siblings.issubset(active)
            or siblings & blocked
        ):
            continue
        if any(active[child].cell_class != record.cell_class for child in siblings):
            continue
        if orbits is not None:
            if len(record.periodic_roots) != len(record.parent_vertices) or len(
                record.periodic_shifts
            ) != len(record.parent_vertices):
                raise MixedTemplateClosureError(
                    "mixed_adaptation.hierarchy.periodic_roots must align with parent corners; got missing orbit data"
                )
            for vertex, root, shift in zip(
                record.parent_vertices,
                record.periodic_roots,
                record.periodic_shifts,
                strict=True,
            ):
                if (
                    vertex not in orbits.vertices
                    or orbits.vertices[vertex][0] != root
                    or not np.array_equal(orbits.vertices[vertex][1], shift)
                ):
                    raise MixedTemplateClosureError(
                        "mixed_adaptation.hierarchy.periodic_roots must retain source-corner orbits; got changed orbit data"
                    )
        discarded = set(
            int(value) for child in siblings for value in active[child].vertices
        ) - set(record.parent_vertices)
        if not discarded & protected:
            candidates.append((record, discarded))
    return _CoarsenCandidates(active, candidates)


def _restore_coarsen_candidates(
    active: dict[int, Cell],
    candidates: list[tuple[Sibling, set[int]]],
    vertex_ids: np.ndarray,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> CoarsenPatch:
    restored, removed = [], set()
    fine, parents, references = [], [], []
    source_support = {int(identifier): {int(identifier)} for identifier in vertex_ids}
    for record, _ in candidates:
        restored.append(
            Cell(
                record.parent_id,
                record.block_name,
                record.cell_kind,
                np.asarray(record.parent_vertices, dtype=np.int64),
                record.cell_class,
            )
        )
        removed.update(record.child_ids)
        for child, reference in zip(record.child_ids, record.references, strict=True):
            corners = np.asarray(reference, dtype=np.float64)
            for identifier, key in zip(
                active[child].vertices,
                _point_keys(
                    record.cell_kind,
                    np.asarray(record.parent_vertices, dtype=np.int64),
                    corners,
                    scope=scope,
                    budget=budget,
                ),
                strict=True,
            ):
                source_support[int(identifier)] = {vertex for vertex, _ in key}
            fine.append(child)
            parents.append(record.parent_id)
            references.append(corners)
    return CoarsenPatch(restored, removed, fine, parents, references, source_support)


@checked
def coarsen_patch(
    cells: list[Cell],
    records: tuple[Sibling, ...],
    marks: set[int],
    protected: set[int],
    blocked: set[int],
    vertex_ids: np.ndarray,
    orbits: _dependency_phydrax_meshing__periodic.PeriodicConstructionOrbits
    | None = None,
    *,
    column_members: dict[int, set[int]],
    unchanged: set[int],
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> CoarsenPatch:
    """Restore sibling patches jointly across shared faces and physical columns."""
    admitted = _admit_coarsen_candidates(
        cells,
        records,
        marks,
        protected,
        blocked,
        orbits,
        scope=scope,
        budget=budget,
    )
    candidates = admitted.candidates
    changed = True
    column_children = {
        column: members - unchanged for column, members in column_members.items()
    }
    while changed:
        _reserve_host_storage(
            256
            + 64
            * (
                sum(len(record.child_ids) for record, _ in candidates)
                + sum(cell.vertices.size for cell in cells)
            ),
            scope=scope,
            budget=budget,
        )
        removed_candidates = {
            identifier for record, _ in candidates for identifier in record.child_ids
        }
        # A sibling patch survives only when no retained cell uses a discarded vertex.
        outside = {
            int(vertex)
            for cell in cells
            if cell.identifier not in removed_candidates
            for vertex in cell.vertices
        }
        retained = [
            (record, discarded)
            for record, discarded in candidates
            if not discarded & outside
        ]
        if orbits is not None:
            outside_roots = {orbits.vertices[vertex][0] for vertex in outside}
            retained = [
                (record, discarded)
                for record, discarded in retained
                if not {orbits.vertices[vertex][0] for vertex in discarded}
                & outside_roots
            ]
        retained_children = {
            identifier for record, _ in retained for identifier in record.child_ids
        }
        retained = [
            (record, discarded)
            for record, discarded in retained
            if record.layer_column is None
            or (
                record.layer_column in column_children
                and column_children[record.layer_column].issubset(retained_children)
            )
        ]
        changed = len(retained) != len(candidates)
        candidates = retained
    return _restore_coarsen_candidates(
        admitted.active,
        candidates,
        vertex_ids,
        scope=scope,
        budget=budget,
    )


def _protected_entity_keys(
    mesh: CellMesh, values: ArrayLike, dimension: int
) -> set[tuple[int, ...]]:
    array = np.asarray(values)
    if array.size == 0:
        return set()
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            "mixed_adaptation.protected_entity_keys must contain integer vertex-ID rows; got an incompatible array"
        )
    keys = {tuple(sorted(int(value) for value in row if value >= 0)) for row in array}
    available = {
        tuple(int(value) for value in row if value >= 0)
        for row in entity_keys_owner.entity_keys(mesh, dimension)
    }
    if not keys.issubset(available):
        raise ValueError(
            "mixed_adaptation.protected_entity_keys must name current source entities; got unknown keys"
        )
    return keys


def _permitted_subdivision(
    cell: Cell,
    template: _Template,
    edges: set[tuple[int, ...]],
    faces: set[tuple[int, ...]],
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    if not edges and not faces:
        return True
    for child in template.children:
        for key in _point_keys(
            cell.kind, cell.vertices, child, scope=scope, budget=budget
        ):
            if len(key) == 2 and tuple(identifier for identifier, _ in key) in edges:
                return False
    if faces:
        topology = reference_cell_topology(cell.kind)
        protected = [
            facet
            for facet, local in enumerate(topology.entities[topology.dimension - 1])
            if tuple(sorted(int(cell.vertices[index]) for index in local)) in faces
        ]
        if protected:
            candidate = _facet_signatures(
                cell.kind, cell.vertices, template, scope=scope, budget=budget
            )
            identity = _facet_signatures(
                cell.kind,
                cell.vertices,
                _Template("identity", _reference(cell.kind)[None], None),
                scope=scope,
                budget=budget,
            )
            if any(candidate[facet] != identity[facet] for facet in protected):
                return False
    return True


def _reserve_host_storage(
    amount: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Count declared live host storage locally before allocation."""

    retained = budget.storage_ticket
    baseline = 0 if retained is None else retained.bytes_upper
    proposed = max(
        0, budget.retained_basis_bytes + budget.temporary_bytes_upper + amount - baseline
    )
    if proposed > budget.ticket.bytes_upper:
        budget.ticket.refuse(0, bytes_upper=proposed)
    budget.reserve(0, amount)


@checked
def adapt_mixed_mesh(
    mesh: CellMesh,
    *,
    refine_cell_ids: ArrayLike,
    coarsen_cell_ids: ArrayLike | None = None,
    hierarchy: MixedAdaptationHierarchy | None = None,
    cell_classes: ArrayLike | None = None,
    protected_vertex_ids: ArrayLike | None = None,
    protected_cell_ids: ArrayLike | None = None,
    protected_edge_keys: ArrayLike | None = None,
    protected_face_keys: ArrayLike | None = None,
    layer_columns: MixedLayerColumns | None = None,
    maximum_cells: int = 1 << 24,
    maximum_closure_steps: int = 1 << 20,
    maximum_work_units: int = 1 << 26,
    maximum_scratch_bytes: int = MeshingLimits().maximum_scratch_bytes,
) -> MixedAdaptationOutcome:
    """Build a mixed edit within its explicit native request allowance."""
    if (
        isinstance(maximum_scratch_bytes, bool)
        or not isinstance(maximum_scratch_bytes, (int, np.integer))
        or maximum_scratch_bytes < 0
    ):
        raise ValueError(
            "mixed_adaptation.maximum_scratch_bytes must be a nonnegative integer; got an invalid bound"
        )
    limits = MeshingLimits(
        maximum_work_units=maximum_work_units,
        maximum_scratch_bytes=maximum_scratch_bytes,
    )
    with NativeExecutionScope(limits.native_limits()) as scope:
        remaining = scope.remaining()
        ticket = scope.batch(
            min(maximum_work_units, remaining.remaining_work_units),
            queries_upper=min(
                limits.maximum_geometry_queries, remaining.remaining_geometry_queries
            ),
            cavity_upper=min(
                limits.maximum_cavity_cells, remaining.remaining_cavity_cells
            ),
            bytes_upper=min(maximum_scratch_bytes, remaining.remaining_managed_bytes),
        )
        budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=maximum_work_units,
            maximum_memory_bytes=maximum_scratch_bytes,
        )
        with ticket, budget.activate(), budget.temporary_scope():
            outcome = _adapt_mixed_mesh(
                mesh,
                refine_cell_ids=refine_cell_ids,
                coarsen_cell_ids=coarsen_cell_ids,
                hierarchy=hierarchy,
                cell_classes=cell_classes,
                protected_vertex_ids=protected_vertex_ids,
                protected_cell_ids=protected_cell_ids,
                protected_edge_keys=protected_edge_keys,
                protected_face_keys=protected_face_keys,
                layer_columns=layer_columns,
                maximum_cells=maximum_cells,
                maximum_closure_steps=maximum_closure_steps,
                maximum_work_units=maximum_work_units,
                scope=scope,
                budget=budget,
                ticket=ticket,
            )
    return outcome._replace(
        evidence=outcome.evidence._replace(
            execution_evidence=NativeExecutionRecord(scope.completed_evidence())
        )
    )


class _MixedRequest(StrictModule):
    mesh: CellMesh
    refine_cell_ids: ArrayLike
    coarsen_cell_ids: ArrayLike | None
    hierarchy: MixedAdaptationHierarchy | None
    cell_classes: ArrayLike | None
    protected_vertex_ids: ArrayLike | None
    protected_cell_ids: ArrayLike | None
    protected_edge_keys: ArrayLike | None
    protected_face_keys: ArrayLike | None
    layer_columns: MixedLayerColumns | None
    maximum_cells: int
    maximum_closure_steps: int
    maximum_work_units: int


class _MixedControls(StrictModule):
    dimension: int
    layer_columns: MixedLayerColumns | None
    maximum_cells: int
    maximum_closure_steps: int
    maximum_work_units: int


class _MixedSource(StrictModule):
    vertex_ids: np.ndarray
    orbits: _dependency_phydrax_meshing__periodic.PeriodicConstructionOrbits | None
    edges: set[tuple[int, ...]]
    faces: set[tuple[int, ...]]
    protected_boundary_vertices: set[int]
    coarsen_cell_ids: ArrayLike
    protected_vertex_ids: ArrayLike
    protected_cell_ids: ArrayLike


class _MixedMarks(StrictModule):
    ids: np.ndarray
    refine: set[int]
    requested_refine: set[int]
    coarsen: set[int]
    protected: set[int]
    blocked: set[int]
    cells: list[Cell]
    records: tuple[Sibling, ...]


class _MixedAncestry(StrictModule):
    sibling_ids: set[int]
    identity_bank: set[tuple[int, str, str, tuple[int, ...], int]]
    unchanged_marks: set[int]
    intervals: dict[int, int]
    columns: dict[int, int]
    identity_work: int
    column_members: dict[int, set[int]]
    unchanged: set[int]


class _MixedCoarsening(StrictModule):
    removed: set[int]
    restored_ids: set[int]
    coarsening_fine: list[int]
    coarsening_parent: list[int]
    coarsening_reference: list[np.ndarray]
    source_support: dict[int, set[int]]
    working: list[Cell]


class _MixedDomains(StrictModule):
    domains: list[list[_Template]]


class _MixedChildren(StrictModule):
    next_cell: int
    next_vertex: int
    points: dict[_dependency_phydrax_meshing__periodic.ConstructionPointKey, int]
    stencils: dict[int, _dependency_phydrax_meshing__periodic.ConstructionPointKey]
    coordinates: dict[int, np.ndarray]
    target: dict[str, list[tuple[int, np.ndarray]]]
    target_kinds: dict[str, str]
    refinements: list[int]
    parents: list[int]
    references: list[np.ndarray]
    new_records: list[Sibling]
    refined: list[int]
    relation_sources: list[int]
    relation_targets: list[int]
    relation_kinds: list[int]
    next_columns: dict[int, int]
    next_intervals: dict[int, int]
    successor_identities: list[tuple[int, str, str, tuple[int, ...], int]]


class _MixedAllocation(StrictModule):
    next_cell: int
    next_vertex: int


class _MixedTopology(StrictModule):
    used: np.ndarray
    blocks: tuple[_dependency_phydrax_meshing__topology_edit.TopologyEditBlock, ...]
    sources: np.ndarray
    weights: np.ndarray
    target_coordinates: np.ndarray
    probe: CellMesh
    periodic_witness: (
        _dependency_phydrax_meshing__topology_edit.PeriodicVertexOrbitWitness | None
    )
    prescribed: list[_dependency_phydrax_meshing__topology_edit.PrescribedEntityIds]
    retired: dict[tuple[int, tuple[int, ...]], int]


class _MixedRelations(StrictModule):
    relations: list[_dependency_phydrax_meshing__topology_edit.EntityRelations]


class _MixedPublication(StrictModule):
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    successor: MixedAdaptationHierarchy
    evidence: MixedAdaptationEvidence


def _validate_mixed_controls(
    request: _MixedRequest,
) -> _MixedControls:
    mesh = request.mesh
    hierarchy = request.hierarchy
    layer_columns = request.layer_columns
    maximum_cells = request.maximum_cells
    maximum_closure_steps = request.maximum_closure_steps
    maximum_work_units = request.maximum_work_units
    dimension = mesh.topological_dimension
    if dimension == 2:
        if any(
            block.cell_kind != "quadrilateral" for block in mesh.blocks
        ) or mesh.ambient_dimension not in (2, 3):
            raise ValueError(
                "mixed_adaptation.mesh must contain quadrilaterals with ambient dimension 2 or 3; got an incompatible surface mesh"
            )
        if layer_columns is not None:
            raise ValueError(
                "mixed_adaptation.layer_columns must accompany three-dimensional cells; got a surface mesh"
            )
    elif dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError(
            "mixed_adaptation.mesh must have topological and ambient dimension 3; got an incompatible volume mesh"
        )
    if layer_columns is None and hierarchy is not None:
        layer_columns = hierarchy.layer_columns
    maximum_cells = positive_integer(maximum_cells, "maximum_cells")
    maximum_closure_steps = positive_integer(
        maximum_closure_steps, "maximum_closure_steps"
    )
    maximum_work_units = positive_integer(maximum_work_units, "maximum_work_units")
    return _MixedControls(
        dimension=dimension,
        layer_columns=layer_columns,
        maximum_cells=maximum_cells,
        maximum_closure_steps=maximum_closure_steps,
        maximum_work_units=maximum_work_units,
    )


def _prepare_mixed_source(
    request: _MixedRequest,
    controls: _MixedControls,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedSource:
    mesh = request.mesh
    coarsen_cell_ids = request.coarsen_cell_ids
    protected_vertex_ids = request.protected_vertex_ids
    protected_cell_ids = request.protected_cell_ids
    protected_edge_keys = request.protected_edge_keys
    protected_face_keys = request.protected_face_keys
    dimension = controls.dimension
    _reserve_host_storage(
        256 * 12
        + 64 * 20
        + sum(
            block.vertices.size * 8
            + block.cell_count
            * (64 + len(Cell._fields) * np.dtype(np.intp).itemsize + 256)
            for block in mesh.blocks
        )
        + mesh.coordinates.shape[0]
        * (
            256
            + 5 * 32
            + 3 * 64
            + 3 * 128
            + mesh.ambient_dimension * np.dtype(np.float64).itemsize
        ),
        scope=scope,
        budget=budget,
    )
    empty_ids = np.zeros((0,), dtype=np.int64)
    coarsen_cell_ids = empty_ids if coarsen_cell_ids is None else coarsen_cell_ids
    protected_vertex_ids = (
        empty_ids if protected_vertex_ids is None else protected_vertex_ids
    )
    protected_cell_ids = empty_ids if protected_cell_ids is None else protected_cell_ids
    protected_edge_keys = (
        empty_ids if protected_edge_keys is None else protected_edge_keys
    )
    protected_face_keys = (
        empty_ids if protected_face_keys is None else protected_face_keys
    )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    orbits = (
        None
        if mesh.periodic_topology is None
        else _dependency_phydrax_meshing__periodic.PeriodicConstructionOrbits(mesh)
    )
    edges = _protected_entity_keys(mesh, protected_edge_keys, 1)
    if dimension != 3 and np.asarray(protected_face_keys).size:
        raise ValueError(
            "mixed_adaptation.protected_face_keys must accompany volume cells; got face keys on a surface mesh"
        )
    faces = (
        _protected_entity_keys(mesh, protected_face_keys, 2) if dimension == 3 else set()
    )
    if orbits is not None:
        for degree, protected_keys in ((1, edges), (2, faces)):
            if protected_keys:
                keys = [
                    tuple(int(value) for value in row if value >= 0)
                    for row in entity_keys_owner.entity_keys(mesh, degree)
                ]
                selected = {
                    orbits.face(np.asarray(key, dtype=np.int64))[0]
                    for key in protected_keys
                }
                protected_keys.update(
                    key
                    for key in keys
                    if orbits.face(np.asarray(key, dtype=np.int64))[0] in selected
                )
    protected_boundary_vertices = {
        identifier for key in edges | faces for identifier in key
    }
    return _MixedSource(
        vertex_ids=vertex_ids,
        orbits=orbits,
        edges=edges,
        faces=faces,
        protected_boundary_vertices=protected_boundary_vertices,
        coarsen_cell_ids=coarsen_cell_ids,
        protected_vertex_ids=protected_vertex_ids,
        protected_cell_ids=protected_cell_ids,
    )


def _admit_mixed_marks(
    request: _MixedRequest,
    source: _MixedSource,
) -> _MixedMarks:
    mesh = request.mesh
    refine_cell_ids = request.refine_cell_ids
    coarsen_cell_ids = source.coarsen_cell_ids
    hierarchy = request.hierarchy
    cell_classes = request.cell_classes
    protected_vertex_ids = source.protected_vertex_ids
    protected_cell_ids = source.protected_cell_ids
    vertex_ids = source.vertex_ids
    orbits = source.orbits
    protected_boundary_vertices = source.protected_boundary_vertices
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    classes = (
        np.zeros(ids.size, dtype=np.int64)
        if cell_classes is None
        else np.asarray(cell_classes, dtype=np.int64)
    )
    if classes.shape != ids.shape:
        raise ValueError(
            "mixed_adaptation.cell_classes must align with concatenated source cells; got incompatible rows"
        )
    refine = set(
        int(value) for value in np.asarray(refine_cell_ids, dtype=np.int64).reshape(-1)
    )
    requested_refine = set(refine)
    coarsen = set(
        int(value) for value in np.asarray(coarsen_cell_ids, dtype=np.int64).reshape(-1)
    )
    if refine & coarsen or not (refine | coarsen).issubset(set(ids)):
        raise ValueError(
            "mixed_adaptation.cell_ids must contain disjoint current refinement and coarsening marks; got overlapping or unknown IDs"
        )
    protected = set(
        int(value)
        for value in np.asarray(protected_vertex_ids, dtype=np.int64).reshape(-1)
    )
    blocked = set(
        int(value) for value in np.asarray(protected_cell_ids, dtype=np.int64).reshape(-1)
    )
    protected.update(protected_boundary_vertices)
    if orbits is not None:
        protected_roots = {orbits.vertices[identifier][0] for identifier in protected}
        protected.update(
            identifier
            for identifier, (root, _, _) in orbits.vertices.items()
            if root in protected_roots
        )
    if not blocked.issubset(set(ids)):
        raise ValueError(
            "mixed_adaptation.protected_cell_ids must name current cells; got unknown IDs"
        )
    if refine & blocked:
        raise MixedTemplateClosureError(
            "mixed_adaptation.protected_cell_ids must avoid marked refinement; got a protected marked cell"
        )
    cells = []
    offset = 0
    for block in mesh.blocks:
        for row, identifier in zip(
            np.asarray(block.vertices), np.asarray(block.global_ids), strict=True
        ):
            cells.append(
                Cell(
                    int(identifier),
                    block.name,
                    block.cell_kind,
                    vertex_ids[row],
                    int(classes[offset]),
                )
            )
            offset += 1
    cells.sort(key=lambda cell: cell.identifier)
    records = () if hierarchy is None else hierarchy.records
    return _MixedMarks(
        ids=ids,
        refine=refine,
        requested_refine=requested_refine,
        coarsen=coarsen,
        protected=protected,
        blocked=blocked,
        cells=cells,
        records=records,
    )


def _admit_mixed_ancestry(
    request: _MixedRequest,
    controls: _MixedControls,
    marks: _MixedMarks,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedAncestry:
    hierarchy = request.hierarchy
    layer_columns = controls.layer_columns
    maximum_work_units = controls.maximum_work_units
    ids = marks.ids
    coarsen = marks.coarsen
    cells = marks.cells
    records = marks.records
    _reserve_host_storage(
        256
        + 128
        * (
            sum(len(record.child_ids) for record in records)
            + (0 if hierarchy is None else len(hierarchy.identity_cells))
            + len(cells)
        ),
        scope=scope,
        budget=budget,
    )
    sibling_ids = {child for record in records for child in record.child_ids}
    identity_bank = set(() if hierarchy is None else hierarchy.identity_cells)
    unchanged = {
        cell.identifier
        for cell in cells
        if identity_bank
        and (
            cell.identifier,
            cell.name,
            cell.kind,
            tuple(int(value) for value in cell.vertices),
            cell.cell_class,
        )
        in identity_bank
    }
    unchanged_marks = unchanged & coarsen
    _reserve_host_storage(
        3 * 256
        + (0 if layer_columns is None else len(layer_columns.cell_ids))
        * (2 * 256 + 6 * 32 + 192),
        scope=scope,
        budget=budget,
    )
    intervals = (
        {}
        if layer_columns is None
        else dict(
            zip(layer_columns.cell_ids, layer_columns.interval_indices, strict=True)
        )
    )
    columns = (
        {}
        if layer_columns is None
        else dict(zip(layer_columns.cell_ids, layer_columns.column_ids, strict=True))
    )
    if not set(columns).issubset(set(ids)):
        raise ValueError(
            "mixed_adaptation.layer_columns.cell_ids must name current source cells; got unknown IDs"
        )
    column_members: dict[int, set[int]] = {}
    for identifier, column in columns.items():
        column_members.setdefault(int(column), set()).add(int(identifier))
    identity_work = (
        sum(len(record.child_ids) for record in records)
        + len(identity_bank)
        + len(cells)
        + (sum(cell.vertices.size for cell in cells) if identity_bank else 0)
        + len(columns)
    )
    if identity_work > maximum_work_units:
        raise MixedTemplateClosureError(
            "mixed_adaptation.maximum_work_units must admit source identities; got a work budget refusal"
        )
    budget.reserve(identity_work)
    return _MixedAncestry(
        sibling_ids=sibling_ids,
        identity_bank=identity_bank,
        unchanged_marks=unchanged_marks,
        intervals=intervals,
        columns=columns,
        identity_work=identity_work,
        column_members=column_members,
        unchanged=unchanged,
    )


def _prepare_mixed_coarsening(
    source: _MixedSource,
    marks: _MixedMarks,
    ancestry: _MixedAncestry,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedCoarsening:
    vertex_ids = source.vertex_ids
    orbits = source.orbits
    refine = marks.refine
    coarsen = marks.coarsen
    protected = marks.protected
    blocked = marks.blocked
    cells = marks.cells
    records = marks.records
    unchanged_marks = ancestry.unchanged_marks
    intervals = ancestry.intervals
    columns = ancestry.columns
    column_members = ancestry.column_members
    unchanged = ancestry.unchanged
    patch = coarsen_patch(
        cells,
        records,
        coarsen,
        protected,
        blocked,
        vertex_ids,
        orbits,
        column_members=column_members,
        unchanged=unchanged,
        scope=scope,
        budget=budget,
    )
    restored, removed = patch.restored, patch.removed
    if orbits is not None and coarsen - removed - unchanged_marks:
        raise MixedTemplateClosureError(
            "mixed_adaptation.coarsen_cell_ids must cover compatible sibling and facet orbits; got an incomplete or protected orbit"
        )
    restored_ids = {cell.identifier for cell in restored}
    coarsening_fine, coarsening_parent, coarsening_reference = (
        patch.fine,
        patch.parents,
        patch.references,
    )
    source_support = patch.source_support
    working = [cell for cell in cells if cell.identifier not in removed] + restored
    working.sort(key=lambda cell: cell.identifier)
    for record in records:
        if (
            record.parent_id in restored_ids
            and record.layer_column is not None
            and record.layer_interval is not None
        ):
            columns[record.parent_id] = record.layer_column
            intervals[record.parent_id] = record.layer_interval
    selected_columns = {
        columns[identifier] for identifier in refine if identifier in columns
    }
    refine.update(
        identifier for identifier, column in columns.items() if column in selected_columns
    )
    if refine & removed:
        raise MixedTemplateClosureError(
            "mixed_adaptation.layer_columns must separate refinement and coarsening cohorts; got a conflicting column"
        )
    if refine & blocked:
        raise MixedTemplateClosureError(
            "mixed_adaptation.protected_cell_ids must avoid layer-column closure; got a protected closure cell"
        )
    return _MixedCoarsening(
        removed=removed,
        restored_ids=restored_ids,
        coarsening_fine=coarsening_fine,
        coarsening_parent=coarsening_parent,
        coarsening_reference=coarsening_reference,
        source_support=source_support,
        working=working,
    )


def _prepare_mixed_domains(
    controls: _MixedControls,
    source: _MixedSource,
    marks: _MixedMarks,
    ancestry: _MixedAncestry,
    coarsening: _MixedCoarsening,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ticket: NativeBatchTicket,
) -> _MixedDomains:
    layer_columns = controls.layer_columns
    dimension = controls.dimension
    edges = source.edges
    faces = source.faces
    refine = marks.refine
    blocked = marks.blocked
    intervals = ancestry.intervals
    restored_ids = coarsening.restored_ids
    working = coarsening.working
    domains = []
    tables: dict[tuple[str, bool], tuple[_Template, ...]] = {}
    for cell in working:
        _reserve_host_storage(256, scope=scope, budget=budget)
        axial = dimension == 3 and (
            cell.identifier not in intervals
            or (layer_columns is not None and layer_columns.axial_refinement)
        )
        if (
            cell.identifier in intervals
            and intervals[cell.identifier] == 0
            and layer_columns is not None
            and layer_columns.hard_first_thickness
            and not layer_columns.allow_schedule_change
        ):
            axial = False
        if (cell.kind, axial) not in tables:
            tables[cell.kind, axial] = _templates(
                cell.kind, axial, scope=scope, budget=budget, ticket=ticket
            )
            _reserve_host_storage(
                sum(
                    256 + template.children.nbytes
                    for template in tables[cell.kind, axial]
                ),
                scope=scope,
                budget=budget,
            )
        templates = list(tables[cell.kind, axial])
        red: list[tuple[_Template, tuple[tuple[int, int], tuple[int, int]]]] = []
        for template in templates:
            diagonal = template.interior_diagonal
            if diagonal is not None:
                red.append((template, diagonal))
        if red:
            selected = red[
                min(
                    (_interior_diagonal_key(diagonal, cell.vertices), index)
                    for index, (_, diagonal) in enumerate(red)
                )[1]
            ][0]
            templates = [
                template
                for template in templates
                if template.interior_diagonal is None or template is selected
            ]
        if cell.identifier in refine:
            templates = templates[1:]
        if cell.identifier in restored_ids:
            templates = templates[:1]
        if cell.identifier in blocked:
            templates = templates[:1]
        templates = [
            template
            for template in templates
            if _permitted_subdivision(
                cell, template, edges, faces, scope=scope, budget=budget
            )
        ]
        if not templates:
            raise MixedTemplateClosureError(
                "mixed_adaptation.refine_cell_ids must preserve protected edge and face signatures; got no permitted template"
            )
        domains.append(templates)
    _reserve_host_storage(
        64 + 2 * len(working) * np.dtype(np.intp).itemsize, scope=scope, budget=budget
    )
    return _MixedDomains(
        domains=domains,
    )


def _execute_mixed_closure(
    controls: _MixedControls,
    source: _MixedSource,
    ancestry: _MixedAncestry,
    coarsening: _MixedCoarsening,
    prepared_domains: _MixedDomains,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedClosure:
    maximum_cells = controls.maximum_cells
    maximum_closure_steps = controls.maximum_closure_steps
    maximum_work_units = controls.maximum_work_units
    orbits = source.orbits
    identity_work = ancestry.identity_work
    working = coarsening.working
    domains = prepared_domains.domains
    selection = _closure(
        working,
        domains,
        maximum_closure_steps,
        maximum_work_units - identity_work,
        orbits,
        scope=scope,
        budget=budget,
    )
    chosen, steps, work = selection.chosen, selection.steps, selection.work
    work += identity_work
    work += sum(template.children.size for template in chosen)
    if work > maximum_work_units:
        raise MixedTemplateClosureError(
            "mixed_adaptation.maximum_work_units must admit child construction; got a work budget refusal"
        )
    budget.reserve(sum(template.children.size for template in chosen))
    count = sum(template.children.shape[0] for template in chosen)
    if count > maximum_cells:
        raise MixedTemplateClosureError(
            "mixed_adaptation.maximum_cells must admit every child cell; got a target cell budget refusal"
        )
    return _MixedClosure(
        chosen=chosen,
        steps=steps,
        work=work,
    )


def _prepare_mixed_children(
    request: _MixedRequest,
    source: _MixedSource,
    marks: _MixedMarks,
    coarsening: _MixedCoarsening,
    closure: _MixedClosure,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedChildren:
    mesh = request.mesh
    hierarchy = request.hierarchy
    vertex_ids = source.vertex_ids
    ids = marks.ids
    working = coarsening.working
    chosen = closure.chosen
    next_cell = max(
        int(np.max(ids, initial=-1)) + 1,
        0 if hierarchy is None else hierarchy.next_cell_id,
    )
    next_vertex = max(
        int(np.max(vertex_ids, initial=-1)) + 1,
        0 if hierarchy is None else hierarchy.next_vertex_id,
    )
    _reserve_host_storage(
        sum(
            template.children.shape[0]
            * (
                template.children.shape[1] * (np.dtype(np.int64).itemsize + 3 * 32)
                + 64
                + len(Sibling._fields) * np.dtype(np.intp).itemsize
                + 512
                + (64 + 4 * len(cell.name))
                + (64 + 4 * len(template.name))
            )
            for cell, template in zip(working, chosen, strict=True)
        ),
        scope=scope,
        budget=budget,
    )
    points: dict[_dependency_phydrax_meshing__periodic.ConstructionPointKey, int] = {
        ((int(identifier), Fraction(1)),): int(identifier) for identifier in vertex_ids
    }
    stencils: dict[int, _dependency_phydrax_meshing__periodic.ConstructionPointKey] = {
        int(identifier): ((int(identifier), Fraction(1)),) for identifier in vertex_ids
    }
    coordinates = {
        int(identifier): row
        for identifier, row in zip(vertex_ids, np.asarray(mesh.coordinates), strict=True)
    }
    target: dict[str, list[tuple[int, np.ndarray]]] = {}
    target_kinds: dict[str, str] = {}
    refinements, parents, references, new_records, refined = [], [], [], [], []
    relation_sources, relation_targets, relation_kinds = [], [], []
    next_columns, next_intervals = {}, {}
    successor_identities = []
    return _MixedChildren(
        next_cell=next_cell,
        next_vertex=next_vertex,
        points=points,
        stencils=stencils,
        coordinates=coordinates,
        target=target,
        target_kinds=target_kinds,
        refinements=refinements,
        parents=parents,
        references=references,
        new_records=new_records,
        refined=refined,
        relation_sources=relation_sources,
        relation_targets=relation_targets,
        relation_kinds=relation_kinds,
        next_columns=next_columns,
        next_intervals=next_intervals,
        successor_identities=successor_identities,
    )


def _construct_mixed_vertices(
    mesh: CellMesh,
    children: _MixedChildren,
    cell: Cell,
    reference: np.ndarray,
    next_vertex: int,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[list[int], int]:
    """Issue construction-point identities and their ordered coordinate stencils."""
    child = []
    for key in _point_keys(
        cell.kind, cell.vertices, reference, scope=scope, budget=budget
    ):
        if key not in children.points:
            _reserve_host_storage(
                512 + len(key) * 64 + mesh.ambient_dimension * 8,
                scope=scope,
                budget=budget,
            )
            children.points[key] = next_vertex
            children.stencils[next_vertex] = key
            children.coordinates[next_vertex] = sum(
                float(weight) * children.coordinates[identifier]
                for identifier, weight in key
            )
            next_vertex += 1
        child.append(children.points[key])
    return child, next_vertex


# Vertex issuance owns stencil reduction order; cell publication retains sibling and scientific identity order.
def _construct_mixed_cell(
    request: _MixedRequest,
    source: _MixedSource,
    ancestry: _MixedAncestry,
    coarsening: _MixedCoarsening,
    children: _MixedChildren,
    allocation: _MixedAllocation,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    cell: Cell,
    template: _Template,
) -> _MixedAllocation:
    mesh = request.mesh
    hierarchy = request.hierarchy
    orbits = source.orbits
    sibling_ids = ancestry.sibling_ids
    identity_bank = ancestry.identity_bank
    intervals = ancestry.intervals
    columns = ancestry.columns
    restored_ids = coarsening.restored_ids
    next_cell = allocation.next_cell
    next_vertex = allocation.next_vertex
    target = children.target
    target_kinds = children.target_kinds
    refinements = children.refinements
    parents = children.parents
    references = children.references
    new_records = children.new_records
    refined = children.refined
    relation_sources = children.relation_sources
    relation_targets = children.relation_targets
    relation_kinds = children.relation_kinds
    next_columns = children.next_columns
    next_intervals = children.next_intervals
    successor_identities = children.successor_identities
    signature = (
        cell.identifier,
        cell.name,
        cell.kind,
        tuple(int(value) for value in cell.vertices),
        cell.cell_class,
    )
    if (
        template.name == "identity"
        and cell.identifier not in sibling_ids
        and (
            hierarchy is None
            or cell.identifier in restored_ids
            or signature in identity_bank
        )
    ):
        successor_identities.append(signature)
    child_ids = []
    for child_slot, reference in enumerate(template.children):
        child, next_vertex = _construct_mixed_vertices(
            mesh,
            children,
            cell,
            reference,
            next_vertex,
            scope=scope,
            budget=budget,
        )
        identifier = cell.identifier if template.name == "identity" else next_cell
        if template.name != "identity":
            next_cell += 1
        child_ids.append(identifier)
        if cell.identifier in columns:
            next_columns[identifier] = columns[cell.identifier]
            next_intervals[identifier] = intervals[cell.identifier]
        name = (
            cell.name
            if template.name == "identity"
            else f"{cell.name}:{template.name}:{child_slot}"
        )
        target.setdefault(name, []).append(
            (identifier, np.asarray(child, dtype=np.int64))
        )
        target_kinds[name] = cell.kind
        if cell.identifier not in restored_ids:
            refinements.append(identifier)
            parents.append(cell.identifier)
            references.append(reference)
            relation_sources.append(cell.identifier)
            relation_targets.append(identifier)
            relation_kinds.append(
                int(
                    _dependency_phydrax_meshing__lineage.EntityLineageKind.PRESERVED
                    if template.name == "identity"
                    else _dependency_phydrax_meshing__lineage.EntityLineageKind.REFINED_FROM
                )
            )
    if template.name != "identity":
        refined.append(cell.identifier)
        parent_roots = (
            ()
            if orbits is None
            else tuple(orbits.vertices[int(vertex)][0] for vertex in cell.vertices)
        )
        parent_shifts = (
            ()
            if orbits is None
            else tuple(
                tuple(int(value) for value in orbits.vertices[int(vertex)][1])
                for vertex in cell.vertices
            )
        )
        new_records.append(
            Sibling(
                cell.identifier,
                cell.name,
                cell.kind,
                tuple(int(value) for value in cell.vertices),
                tuple(child_ids),
                tuple(
                    tuple(tuple(float(value) for value in row) for row in child)
                    for child in template.children
                ),
                cell.cell_class,
                columns.get(cell.identifier),
                intervals.get(cell.identifier),
                parent_roots,
                parent_shifts,
            )
        )
    return _MixedAllocation(
        next_cell=next_cell,
        next_vertex=next_vertex,
    )


class _MixedEntityIdentities(StrictModule):
    prescribed: list[_dependency_phydrax_meshing__topology_edit.PrescribedEntityIds]
    retired: dict[tuple[int, tuple[int, ...]], int]


def _recover_mixed_entity_identities(
    request: _MixedRequest,
    controls: _MixedControls,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedEntityIdentities:
    mesh, hierarchy, dimension = request.mesh, request.hierarchy, controls.dimension
    prescribed = []
    retired = (
        {}
        if hierarchy is None
        else {
            (dimension, key): identifier
            for dimension, key, identifier in hierarchy.retired_entities
        }
    )
    for degree in range(1, dimension):
        current_keys = entity_keys_owner.entity_keys(mesh, degree)
        current_ids = np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
        _reserve_host_storage(
            current_keys.size * (64 + 3 * np.dtype(np.int64).itemsize),
            scope=scope,
            budget=budget,
        )
        for key, identifier in zip(current_keys, current_ids, strict=True):
            retired[(degree, tuple(int(value) for value in key))] = int(identifier)
        current_set = {tuple(int(value) for value in key) for key in current_keys}
        restored_keys = [
            (key, identifier)
            for (level, key), identifier in retired.items()
            if level == degree and key not in current_set
        ]
        if restored_keys:
            prescribed.append(
                _dependency_phydrax_meshing__topology_edit.PrescribedEntityIds(
                    degree,
                    np.asarray([key for key, _ in restored_keys], dtype=np.int64),
                    np.asarray(
                        [identifier for _, identifier in restored_keys], dtype=np.int64
                    ),
                )
            )
    return _MixedEntityIdentities(prescribed, retired)


def _prepare_mixed_topology(
    request: _MixedRequest,
    controls: _MixedControls,
    source: _MixedSource,
    coarsening: _MixedCoarsening,
    children: _MixedChildren,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ticket: NativeBatchTicket,
) -> _MixedTopology:
    mesh = request.mesh
    hierarchy = request.hierarchy
    vertex_ids = source.vertex_ids
    orbits = source.orbits
    coarsening_fine = coarsening.coarsening_fine
    coarsening_parent = coarsening.coarsening_parent
    stencils = children.stencils
    coordinates = children.coordinates
    target = children.target
    target_kinds = children.target_kinds
    relation_sources = children.relation_sources
    relation_targets = children.relation_targets
    relation_kinds = children.relation_kinds
    for child, parent in zip(coarsening_fine, coarsening_parent, strict=True):
        relation_sources.append(child)
        relation_targets.append(parent)
        relation_kinds.append(
            int(_dependency_phydrax_meshing__lineage.EntityLineageKind.COARSENED_INTO)
        )
    corner_slots = sum(row.size for values in target.values() for _, row in values)
    _reserve_host_storage(
        corner_slots * (3 * np.dtype(np.int64).itemsize + 128)
        + sum(len(values) for values in target.values()) * 128,
        scope=scope,
        budget=budget,
    )
    used_vertices = {
        int(value) for block in target.values() for _, row in block for value in row
    }
    original_vertices = set(vertex_ids.tolist())
    # Dense scientific vertex order is an authored control. Retain every
    # surviving source row in that order; append only newly allocated identities
    # in allocator order. All same-family templates retain their parent corners,
    # so nested/partial inverse epochs recover the source rows inductively.
    used = np.asarray(
        [int(identifier) for identifier in vertex_ids if int(identifier) in used_vertices]
        + sorted(used_vertices - original_vertices),
        dtype=np.int64,
    )
    local = {int(identifier): index for index, identifier in enumerate(used)}
    source_kinds = {block.name: block.cell_kind for block in mesh.blocks}
    blocks = tuple(
        _dependency_phydrax_meshing__topology_edit.TopologyEditBlock(
            name,
            target_kinds[name],
            source_kinds.get(name),
            np.stack(
                [[local[int(value)] for value in row] for _, row in target[name]]
            ).astype(np.int32),
            np.asarray([identifier for identifier, _ in target[name]], dtype=np.int64),
        )
        for name in sorted(target)
    )
    width = max(len(stencils[int(identifier)]) for identifier in used)
    _reserve_host_storage(
        used.size * width * (np.dtype(np.int64).itemsize + np.dtype(np.float64).itemsize)
        + used.size * mesh.ambient_dimension * np.dtype(np.float64).itemsize,
        scope=scope,
        budget=budget,
    )
    sources = np.full((used.size, width), -1, dtype=np.int64)
    weights = np.zeros((used.size, width), dtype=np.float64)
    for index, identifier in enumerate(used):
        for slot, (source_id, weight) in enumerate(stencils[int(identifier)]):
            sources[index, slot], weights[index, slot] = source_id, weight
    target_coordinates = np.stack([coordinates[int(identifier)] for identifier in used])
    # Connectivity preparation uses bounded edge/face routes and sorting
    # workspaces from child occurrences rather than all entity pairs.
    entity_occurrences = sum(
        block.cells.shape[0]
        * sum(len(row) for row in reference_cell_topology(block.cell_kind).entities[1:-1])
        for block in blocks
    )
    _reserve_host_storage(
        entity_occurrences * (12 * np.dtype(np.int64).itemsize + 2 * 64),
        scope=scope,
        budget=budget,
    )
    with ticket.native_target():
        probe = CellMesh(
            target_coordinates,
            tuple(
                CellBlock(
                    block.name, block.cell_kind, block.cells, global_ids=block.cell_ids
                )
                for block in blocks
            ),
            vertex_global_ids=used,
        )
    periodic_witness = (
        None
        if orbits is None
        else orbits.witness(
            mesh,
            probe,
            stencils,
            () if hierarchy is None else hierarchy.quotient_entities,
            budget=budget,
        )
    )
    identities = _recover_mixed_entity_identities(
        request, controls, scope=scope, budget=budget
    )
    prescribed, retired = identities.prescribed, identities.retired
    return _MixedTopology(
        used=used,
        blocks=blocks,
        sources=sources,
        weights=weights,
        target_coordinates=target_coordinates,
        probe=probe,
        periodic_witness=periodic_witness,
        prescribed=prescribed,
        retired=retired,
    )


# Paired indexed emission preserves source-row order, duplicate support matches, and interleaved resource charges.
def _prepare_mixed_relations(
    request: _MixedRequest,
    controls: _MixedControls,
    coarsening: _MixedCoarsening,
    children: _MixedChildren,
    topology: _MixedTopology,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedRelations:
    mesh = request.mesh
    dimension = controls.dimension
    source_support = coarsening.source_support
    stencils = children.stencils
    relation_sources = children.relation_sources
    relation_targets = children.relation_targets
    relation_kinds = children.relation_kinds
    probe = topology.probe
    relations = []
    for degree in range(dimension + 1):
        keys = entity_keys_owner.entity_keys(mesh, degree)
        if degree == dimension:
            relations.append(
                _dependency_phydrax_meshing__topology_edit.EntityRelations(
                    degree,
                    np.asarray(relation_sources, dtype=np.int64)[:, None],
                    np.asarray(relation_targets, dtype=np.int64)[:, None],
                    np.asarray(relation_kinds, dtype=np.int32),
                )
            )
        else:
            target_keys = entity_keys_owner.entity_keys(probe, degree)
            inherited_source, inherited_target, inherited_kinds = [], [], []
            if 0 < degree < dimension:
                # Index source entities by a required support vertex, preserving
                # source-row order within each target entity.
                _reserve_host_storage(
                    keys.shape[0] * (2 * 256 + 128) + keys.size * (32 + 64),
                    scope=scope,
                    budget=budget,
                )
                source_sets = [
                    {int(value) for value in key if value >= 0} for key in keys
                ]
                source_supports = [
                    set().union(*(source_support[vertex] for vertex in vertices))
                    for vertices in source_sets
                ]
                containing: dict[int, list[int]] = {}
                supported: dict[int, list[int]] = {}
                for row, (vertices, support) in enumerate(
                    zip(source_sets, source_supports, strict=True)
                ):
                    for vertex in vertices:
                        containing.setdefault(vertex, []).append(row)
                    supported.setdefault(min(support), []).append(row)
                for target_key in target_keys:
                    active_target = {int(value) for value in target_key if value >= 0}
                    support = {
                        source
                        for vertex in active_target
                        for source, _ in stencils[vertex]
                    }
                    for row in containing.get(min(support), ()):
                        if (
                            support.issubset(source_sets[row])
                            and active_target != source_sets[row]
                        ):
                            _reserve_host_storage(
                                128 + 2 * keys.shape[1] * np.dtype(np.int64).itemsize,
                                scope=scope,
                                budget=budget,
                            )
                            inherited_source.append(keys[row])
                            inherited_target.append(target_key)
                            inherited_kinds.append(
                                int(
                                    _dependency_phydrax_meshing__lineage.EntityLineageKind.REFINED_FROM
                                )
                            )
                    for row in sorted(
                        row
                        for vertex in active_target
                        for row in supported.get(vertex, ())
                    ):
                        if (
                            source_supports[row].issubset(active_target)
                            and active_target != source_sets[row]
                        ):
                            _reserve_host_storage(
                                128 + 2 * keys.shape[1] * np.dtype(np.int64).itemsize,
                                scope=scope,
                                budget=budget,
                            )
                            inherited_source.append(keys[row])
                            inherited_target.append(target_key)
                            inherited_kinds.append(
                                int(
                                    _dependency_phydrax_meshing__lineage.EntityLineageKind.COARSENED_INTO
                                )
                            )
            relations.append(
                _dependency_phydrax_meshing__topology_edit.EntityRelations(
                    degree,
                    np.asarray(inherited_source, dtype=np.int64).reshape(
                        (-1, keys.shape[1])
                    ),
                    np.asarray(inherited_target, dtype=np.int64).reshape(
                        (-1, keys.shape[1])
                    ),
                    np.asarray(inherited_kinds, dtype=np.int32),
                )
            )
    return _MixedRelations(
        relations=relations,
    )


def _publish_mixed_edit(
    controls: _MixedControls,
    marks: _MixedMarks,
    ancestry: _MixedAncestry,
    coarsening: _MixedCoarsening,
    closure: _MixedClosure,
    children: _MixedChildren,
    allocation: _MixedAllocation,
    topology: _MixedTopology,
    lineage: _MixedRelations,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedPublication:
    layer_columns = controls.layer_columns
    dimension = controls.dimension
    requested_refine = marks.requested_refine
    coarsen = marks.coarsen
    records = marks.records
    unchanged_marks = ancestry.unchanged_marks
    intervals = ancestry.intervals
    removed = coarsening.removed
    restored_ids = coarsening.restored_ids
    coarsening_fine = coarsening.coarsening_fine
    coarsening_parent = coarsening.coarsening_parent
    coarsening_reference = coarsening.coarsening_reference
    working = coarsening.working
    chosen = closure.chosen
    steps = closure.steps
    work = closure.work
    next_cell = allocation.next_cell
    next_vertex = allocation.next_vertex
    refinements = children.refinements
    parents = children.parents
    references = children.references
    new_records = children.new_records
    refined = children.refined
    next_columns = children.next_columns
    next_intervals = children.next_intervals
    successor_identities = children.successor_identities
    used = topology.used
    blocks = topology.blocks
    sources = topology.sources
    weights = topology.weights
    target_coordinates = topology.target_coordinates
    periodic_witness = topology.periodic_witness
    prescribed = topology.prescribed
    retired = topology.retired
    relations = lineage.relations

    def witnesses(
        fine: list[int], coarse: list[int], vertices: list[np.ndarray]
    ) -> transition_types_owner.NestedReferenceWitnesses | None:
        if not fine:
            return None
        corner_count = max(row.shape[0] for row in vertices)
        _reserve_host_storage(
            len(fine)
            * (
                corner_count * dimension * np.dtype(np.float64).itemsize
                + 2 * np.dtype(np.int64).itemsize
            ),
            scope=scope,
            budget=budget,
        )
        padded = np.zeros((len(fine), corner_count, dimension), dtype=np.float64)
        for index, row in enumerate(vertices):
            padded[index, : row.shape[0]] = row
        return transition_types_owner.NestedReferenceWitnesses(
            np.asarray(fine, dtype=np.int64), np.asarray(coarse, dtype=np.int64), padded
        )

    refinement = witnesses(refinements, parents, references)
    coarsening_witness = witnesses(
        coarsening_fine, coarsening_parent, coarsening_reference
    )
    edit = _dependency_phydrax_meshing__topology_edit.CellTopologyEdit(
        "nested_adaptation"
        if coarsening_witness is not None and refinement is not None
        else "nested_coarsening"
        if coarsening_witness is not None
        else "nested_refinement",
        target_coordinates,
        used,
        blocks,
        sources,
        weights,
        sources >= 0,
        tuple(relations),
        prescribed_entity_ids=tuple(prescribed),
        shared_faces=_shared_faces(blocks, used),
        refinement=refinement,
        coarsening=coarsening_witness,
        periodic_orbits=periodic_witness,
    )
    retained = tuple(record for record in records if record.parent_id not in restored_ids)
    successor_columns = (
        None
        if layer_columns is None
        else MixedLayerColumns(
            np.asarray(sorted(next_columns), dtype=np.int64),
            np.asarray(
                [next_columns[identifier] for identifier in sorted(next_columns)],
                dtype=np.int64,
            ),
            np.asarray(
                [next_intervals[identifier] for identifier in sorted(next_columns)],
                dtype=np.int64,
            ),
            hard_first_thickness=layer_columns.hard_first_thickness,
            axial_refinement=layer_columns.axial_refinement,
            allow_schedule_change=layer_columns.allow_schedule_change,
        )
    )
    successor = MixedAdaptationHierarchy(
        (*retained, *new_records),
        next_cell_id=next_cell,
        next_vertex_id=next_vertex,
        retired_entities=tuple(
            (dimension, key, identifier)
            for (dimension, key), identifier in retired.items()
        ),
        quotient_entities=()
        if periodic_witness is None
        else periodic_witness.quotient_entities,
        layer_columns=successor_columns,
        identity_cells=tuple(successor_identities),
    )
    changed = np.asarray(
        [
            cell.identifier
            for cell, template in zip(working, chosen, strict=True)
            if cell.identifier in intervals
            and intervals[cell.identifier] == 0
            and np.any(np.ptp(template.children[..., 2], axis=1) < 1.0)
        ],
        dtype=np.int64,
    )
    evidence = MixedAdaptationEvidence(
        np.asarray(refined, dtype=np.int64),
        np.asarray(sorted(removed), dtype=np.int64),
        np.asarray(sorted(coarsen - removed - unchanged_marks), dtype=np.int64),
        np.asarray(sorted(set(refined) - requested_refine), dtype=np.int64),
        steps,
        changed,
        work,
        None,
    )
    return _MixedPublication(
        edit=edit,
        successor=successor,
        evidence=evidence,
    )


def _adapt_mixed_mesh(
    mesh: CellMesh,
    *,
    refine_cell_ids: ArrayLike,
    coarsen_cell_ids: ArrayLike | None,
    hierarchy: MixedAdaptationHierarchy | None,
    cell_classes: ArrayLike | None,
    protected_vertex_ids: ArrayLike | None,
    protected_cell_ids: ArrayLike | None,
    protected_edge_keys: ArrayLike | None,
    protected_face_keys: ArrayLike | None,
    layer_columns: MixedLayerColumns | None,
    maximum_cells: int,
    maximum_closure_steps: int,
    maximum_work_units: int,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ticket: NativeBatchTicket,
) -> MixedAdaptationOutcome:
    """Build a mixed candidate through admission, execution, and publication phases."""
    request = _MixedRequest(
        mesh=mesh,
        refine_cell_ids=refine_cell_ids,
        coarsen_cell_ids=coarsen_cell_ids,
        hierarchy=hierarchy,
        cell_classes=cell_classes,
        protected_vertex_ids=protected_vertex_ids,
        protected_cell_ids=protected_cell_ids,
        protected_edge_keys=protected_edge_keys,
        protected_face_keys=protected_face_keys,
        layer_columns=layer_columns,
        maximum_cells=maximum_cells,
        maximum_closure_steps=maximum_closure_steps,
        maximum_work_units=maximum_work_units,
    )
    controls = _validate_mixed_controls(request)
    source = _prepare_mixed_source(request, controls, scope, budget)
    marks = _admit_mixed_marks(request, source)
    ancestry = _admit_mixed_ancestry(request, controls, marks, scope, budget)
    coarsening = _prepare_mixed_coarsening(source, marks, ancestry, scope, budget)
    domains = _prepare_mixed_domains(
        controls, source, marks, ancestry, coarsening, scope, budget, ticket
    )
    closure = _execute_mixed_closure(
        controls, source, ancestry, coarsening, domains, scope, budget
    )
    children = _prepare_mixed_children(
        request, source, marks, coarsening, closure, scope, budget
    )
    allocation = _MixedAllocation(children.next_cell, children.next_vertex)
    for cell, template in zip(coarsening.working, closure.chosen, strict=True):
        allocation = _construct_mixed_cell(
            request,
            source,
            ancestry,
            coarsening,
            children,
            allocation,
            scope,
            budget,
            cell,
            template,
        )
    topology = _prepare_mixed_topology(
        request, controls, source, coarsening, children, scope, budget, ticket
    )
    relations = _prepare_mixed_relations(
        request, controls, coarsening, children, topology, scope, budget
    )
    publication = _publish_mixed_edit(
        controls,
        marks,
        ancestry,
        coarsening,
        closure,
        children,
        allocation,
        topology,
        relations,
        scope,
        budget,
    )
    return MixedAdaptationOutcome(
        publication.edit, publication.successor, publication.evidence
    )
