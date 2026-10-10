#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conforming family-preserving reference templates and sibling restoration.

Closure compares oriented *subfaces*, not just split-edge masks. A finite-domain
constraint solve selects compatible templates across every affected interface.
The candidate and hierarchy are immutable until canonical adaptation accepts them.
Column inversion uses current physical-column membership, not a census inferred
from possibly incomplete sibling records. Only authenticated unchanged SCI cells
are excluded from that cohort; unknown or incompatible intervals retain the fine mesh.
"""

from __future__ import annotations

import sys
from contextlib import AbstractContextManager, nullcontext
from fractions import Fraction
from typing import NamedTuple

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    current_native_execution_budget,
    current_native_host_workspace,
    NativeExecutionBudget,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellMesh
from ..discretization._cell_geometry_transfer import NestedReferenceWitnesses
from ..discretization._reference_cell import (
    facet_orientation_between,
    reference_cell_topology,
)
from ._contracts import MeshingLimits
from ._lineage import EntityLineageKind
from ._measurements import NativeExecutionRecord
from ._periodic import (
    ConstructionPointKey,
    PeriodicConstructionOrbits,
    PeriodicConstructionPointKey,
)
from ._topology_edit import (
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    PeriodicEntityIdentityBank,
    PrescribedEntityIds,
    SharedFaceWitnesses,
    TopologyEditBlock,
)


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
        cell_ids: ArrayLike,
        column_ids: ArrayLike,
        interval_indices: ArrayLike,
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
                "Layer ancestry requires unique IDs and nonnegative aligned column/interval arrays."
            )
        if any(
            not isinstance(flag, (bool, np.bool_))
            for flag in (hard_first_thickness, axial_refinement, allow_schedule_change)
        ):
            raise TypeError("Layer refinement permissions must be bools.")
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


class _Sibling(NamedTuple):
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

    records: tuple[_Sibling, ...] = eqx.field(static=True)
    next_cell_id: int = eqx.field(static=True)
    next_vertex_id: int = eqx.field(static=True)
    hierarchy_id: str = eqx.field(static=True)
    retired_entities: tuple[tuple[int, tuple[int, ...], int], ...] = eqx.field(
        static=True
    )
    quotient_entities: tuple[PeriodicEntityIdentityBank, ...] = eqx.field(static=True)
    identity_cells: tuple[tuple[int, str, str, tuple[int, ...], int], ...] = eqx.field(
        static=True
    )
    layer_columns: MixedLayerColumns | None

    def __init__(
        self,
        records: tuple[_Sibling, ...] = (),
        *,
        next_cell_id: int = 0,
        next_vertex_id: int = 0,
        retired_entities: tuple[tuple[int, tuple[int, ...], int], ...] = (),
        quotient_entities: tuple[PeriodicEntityIdentityBank, ...] = (),
        layer_columns: MixedLayerColumns | None = None,
        identity_cells: tuple[tuple[int, str, str, tuple[int, ...], int], ...] = (),
    ) -> None:
        if next_cell_id < 0 or next_vertex_id < 0:
            raise ValueError("Next scientific identities must be nonnegative.")
        if len({record.parent_id for record in records}) != len(records):
            raise ValueError("Template parent IDs must be unique.")
        ordered = tuple(sorted(records, key=lambda record: record.parent_id))
        self.records = ordered
        self.next_cell_id = next_cell_id
        self.next_vertex_id = next_vertex_id
        self.retired_entities = tuple(sorted(retired_entities))
        self.quotient_entities = quotient_entities
        self.layer_columns = layer_columns
        if len({row[0] for row in identity_cells}) != len(identity_cells):
            raise ValueError("Identity-template scientific cell IDs must be unique.")
        if {row[0] for row in identity_cells} & {
            child for record in ordered for child in record.child_ids
        }:
            raise ValueError(
                "Identity-template cells cannot also be recorded refinement siblings."
            )
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
                record.work,
                record.memory,
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
    edit: CellTopologyEdit
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


def _templates(kind: str, axial: bool) -> tuple[_Template, ...]:
    from .._meshcore import mixed_refinement_templates

    references = mixed_refinement_templates(kind, axial=axial)
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
    template: _Template, vertices: np.ndarray
) -> tuple[tuple[int, ...], ...]:
    """Canonical opposite midpoint-edge identities, independent of local numbering."""
    diagonal = template.interior_diagonal
    if diagonal is None:
        raise RuntimeError("A red tetrahedral template lost its interior diagonal.")
    return tuple(
        sorted(
            tuple(sorted((int(vertices[first]), int(vertices[second]))))
            for first, second in diagonal
        )
    )


type _FaceSignature = tuple[
    tuple[ConstructionPointKey | PeriodicConstructionPointKey, ...], ...
]


def _point_keys(
    kind: str, ids: np.ndarray, points: np.ndarray
) -> tuple[ConstructionPointKey, ...]:
    """Construction identities use exact reference fractions, not rounded weights."""
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
                "Triangle construction keys require complete canonical simplex reference corners."
            )
    # Admit exact keys using input-derived Fraction bit/storage bounds.
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
    fraction = sys.getsizeof(Fraction(0)) + 2 * (
        sys.getsizeof(0) + digits * sys.int_info.sizeof_digit
    )
    integer = max((sys.getsizeof(int(value)) for value in ids), default=sys.getsizeof(0))
    key_bytes = sys.getsizeof(()) + ids.size * (
        3 * np.dtype(np.intp).itemsize + sys.getsizeof(()) + integer + fraction
    )
    _reserve_host_storage(points.shape[0] * (key_bytes + fraction * points.shape[1]))
    result = []
    corners = reference_cell_topology(kind).vertices
    for point in points:
        values = tuple(Fraction(float(value)) for value in point)
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
                raise ValueError("Unsupported mixed corner map.")
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
    orbits: PeriodicConstructionOrbits,
    cell: _Cell,
    template: _Template,
) -> tuple[_FaceSignature, ...]:
    topology = reference_cell_topology(cell.kind)
    facets = topology.entities[topology.dimension - 1]
    supports = [set(int(cell.vertices[index]) for index in local) for local in facets]
    anchors = [
        orbits.face(cell.vertices[np.asarray(local, dtype=np.int32)])[1]
        for local in facets
    ]
    faces: list[list[tuple[PeriodicConstructionPointKey, ...]]] = [[] for _ in facets]
    for child in template.children:
        points = _point_keys(cell.kind, cell.vertices, child)
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
    kind: str, ids: np.ndarray, template: _Template
) -> tuple[_FaceSignature, ...]:
    """Canonical child subfaces lying on each parent facet, in facet order."""
    topology = reference_cell_topology(kind)
    facets = topology.entities[topology.dimension - 1]
    supports = [set(int(ids[index]) for index in facet) for facet in facets]
    faces: list[list[tuple[ConstructionPointKey, ...]]] = [[] for _ in facets]
    for child in template.children:
        points = _point_keys(kind, ids, child)
        for face in facets:
            keys = tuple(points[index] for index in face)
            vertices = {identifier for key in keys for identifier, _ in key}
            for facet, support in enumerate(supports):
                if vertices.issubset(support):
                    faces[facet].append(tuple(sorted(keys)))
    return tuple(tuple(sorted(values)) for values in faces)


class _Cell(NamedTuple):
    identifier: int
    name: str
    kind: str
    vertices: np.ndarray
    cell_class: int


def _closure(
    cells: list[_Cell],
    domains: list[list[_Template]],
    maximum_steps: int,
    maximum_work_units: int,
    orbits: PeriodicConstructionOrbits | None = None,
) -> tuple[list[_Template], int, int]:
    # The chosen table is retained by the caller; only actual closure indices,
    # lazy signatures and live backtracking copies live in this workspace.
    native = current_native_execution_budget()
    workspace = nullcontext() if native is None else native.host_workspace()
    with workspace:
        return _solve_closure(cells, domains, maximum_steps, maximum_work_units, orbits)


def _solve_closure(
    cells: list[_Cell],
    domains: list[list[_Template]],
    maximum_steps: int,
    maximum_work_units: int,
    orbits: PeriodicConstructionOrbits | None,
) -> tuple[list[_Template], int, int]:
    faces: dict[
        tuple[int, ...] | PeriodicConstructionPointKey, list[tuple[int, int]]
    ] = {}
    for index, cell in enumerate(cells):
        topology = reference_cell_topology(cell.kind)
        for facet, local in enumerate(topology.entities[topology.dimension - 1]):
            _reserve_host_storage(256 + 64 * len(local))
            corners = cell.vertices[np.asarray(local, dtype=np.int32)]
            key = (
                tuple(sorted(int(value) for value in corners))
                if orbits is None
                else orbits.face(corners)[0]
            )
            faces.setdefault(key, []).append((index, facet))
    if any(len(owners) > 2 for owners in faces.values()):
        raise MixedTemplateClosureError("Source has a nonmanifold shared face.")
    _reserve_host_storage(sys.getsizeof([]) + 2 * len(faces) * np.dtype(np.intp).itemsize)
    interfaces = [owners for owners in faces.values() if len(owners) == 2]
    signatures: dict[tuple[int, int], tuple[_FaceSignature, ...]] = {}
    steps, work = 0, 0

    def signature(index: int, candidate: int, facet: int, /) -> _FaceSignature:
        # Signatures are evaluated only for candidates the solve actually compares;
        # each evaluation is charged children x facets before it runs.
        nonlocal work
        if (index, candidate) not in signatures:
            cell, template = cells[index], domains[index][candidate]
            work += template.children.shape[0] * len(
                reference_cell_topology(cell.kind).entities[
                    reference_cell_topology(cell.kind).dimension - 1
                ]
            )
            if work > maximum_work_units:
                raise MixedTemplateClosureError(
                    "Mixed signature preparation exhausted its work budget."
                )
            _charge_host_work(
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
                )
            )
            signatures[index, candidate] = (
                _facet_signatures(cell.kind, cell.vertices, template)
                if orbits is None
                else _periodic_facet_signatures(orbits, cell, template)
            )
        return signatures[index, candidate][facet]

    def preferred_assignment(choices: list[list[int]]) -> list[int] | None:
        nonlocal work
        _reserve_host_storage(
            sys.getsizeof([]) + 2 * len(choices) * np.dtype(np.intp).itemsize
        )
        preferred = [values[0] for values in choices]
        for (left, lf), (right, rf) in interfaces:
            work += 2
            if work > maximum_work_units:
                raise MixedTemplateClosureError(
                    "Mixed face closure exhausted its work budget."
                )
            _charge_host_work(2)
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
                    "Mixed face closure exhausted its iteration budget."
                )
            changed = False
            for (left, lf), (right, rf) in interfaces:
                work += len(choices[left]) + len(choices[right])
                if work > maximum_work_units:
                    raise MixedTemplateClosureError(
                        "Mixed face closure exhausted its work budget."
                    )
                _charge_host_work(len(choices[left]) + len(choices[right]))
                for first, ff, second, sf in (
                    (left, lf, right, rf),
                    (right, rf, left, lf),
                ):
                    _reserve_host_storage(
                        256 + 64 * (len(choices[first]) + len(choices[second]))
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
                sys.getsizeof([])
                + 2 * len(choices) * np.dtype(np.intp).itemsize
                + sum(sys.getsizeof(values) for values in choices)
            )
            _reserve_host_storage(branch_bytes)
            branch = [values.copy() for values in choices]
            branch[unresolved] = [candidate]
            try:
                result = solve(branch)
            finally:
                del branch
                _release_host_storage(branch_bytes)
            if result is not None:
                return result
        return None

    _reserve_host_storage(
        sys.getsizeof([])
        + 2 * len(domains) * np.dtype(np.intp).itemsize
        + sum(
            sys.getsizeof([])
            + 2 * len(values) * np.dtype(np.intp).itemsize
            + len(values) * sys.getsizeof(0)
            for values in domains
        )
    )
    result = solve([list(range(len(values))) for values in domains])
    if result is None:
        raise MixedTemplateClosureError(
            "No family-preserving template matches all oriented shared-face subdivisions."
        )
    return [domains[index][value] for index, value in enumerate(result)], steps, work


def _shared_faces(
    blocks: tuple[TopologyEditBlock, ...], vertex_ids: np.ndarray
) -> SharedFaceWitnesses:
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
    return SharedFaceWitnesses(
        np.asarray([value[0][0] for value in paired], dtype=np.int64),
        np.asarray([value[0][1] for value in paired], dtype=np.int32),
        np.asarray([value[1][0] for value in paired], dtype=np.int64),
        np.asarray([value[1][1] for value in paired], dtype=np.int32),
        permutations,
    )


class _CoarsenPatch(NamedTuple):
    restored: list[_Cell]
    removed: set[int]
    fine: list[int]
    parents: list[int]
    references: list[np.ndarray]
    source_support: dict[int, set[int]]


def _coarsen_patch(
    cells: list[_Cell],
    records: tuple[_Sibling, ...],
    marks: set[int],
    protected: set[int],
    blocked: set[int],
    vertex_ids: np.ndarray,
    orbits: PeriodicConstructionOrbits | None = None,
    *,
    column_members: dict[int, set[int]],
    unchanged: set[int],
) -> _CoarsenPatch:
    """Close complete sibling patches jointly, including shared face siblings."""
    active = {cell.identifier: cell for cell in cells}
    candidates = []
    for record in records:
        _reserve_host_storage(sys.getsizeof(set()) + 64 * len(record.child_ids))
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
                    "Periodic siblings lack their recorded source-corner orbits."
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
                        "Periodic sibling source-corner orbits changed since refinement."
                    )
        discarded = set(
            int(value) for child in siblings for value in active[child].vertices
        ) - set(record.parent_vertices)
        if not discarded & protected:
            candidates.append((record, discarded))
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
            )
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
    restored, removed = [], set()
    fine, parents, references = [], [], []
    source_support = {int(identifier): {int(identifier)} for identifier in vertex_ids}
    for record, _ in candidates:
        restored.append(
            _Cell(
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
                ),
                strict=True,
            ):
                source_support[int(identifier)] = {vertex for vertex, _ in key}
            fine.append(child)
            parents.append(record.parent_id)
            references.append(corners)
    return _CoarsenPatch(restored, removed, fine, parents, references, source_support)


def _protected_entity_keys(
    mesh: CellMesh, values: ArrayLike, dimension: int
) -> set[tuple[int, ...]]:
    array = np.asarray(values)
    if array.size == 0:
        return set()
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Protected subdivision keys must be integer vertex-ID rows.")
    keys = {tuple(sorted(int(value) for value in row if value >= 0)) for row in array}
    available = {
        tuple(int(value) for value in row if value >= 0)
        for row in entity_keys(mesh, dimension)
    }
    if not keys.issubset(available):
        raise ValueError("Protected subdivision keys must name current source entities.")
    return keys


def _permitted_subdivision(
    cell: _Cell,
    template: _Template,
    edges: set[tuple[int, ...]],
    faces: set[tuple[int, ...]],
) -> bool:
    if not edges and not faces:
        return True
    for child in template.children:
        for key in _point_keys(cell.kind, cell.vertices, child):
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
            candidate = _facet_signatures(cell.kind, cell.vertices, template)
            identity = _facet_signatures(
                cell.kind,
                cell.vertices,
                _Template("identity", _reference(cell.kind)[None], None),
            )
            if any(candidate[facet] != identity[facet] for facet in protected):
                return False
    return True


def _reserve_host_storage(amount: int, /) -> None:
    """Admit this owner's next bounded Python allocation before it grows."""
    workspace = current_native_host_workspace()
    if workspace is not None:
        workspace.set_bound(workspace.bound + amount)


def _release_host_storage(amount: int, /) -> None:
    workspace = current_native_host_workspace()
    if workspace is not None:
        workspace.set_bound(workspace.bound - amount)


def _charge_host_work(amount: int, /) -> None:
    native = current_native_execution_budget()
    if native is not None:
        native.charge(work=amount)


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
    """Build an edit under its one original native work/wall/host-storage owner."""
    defaults = MeshingLimits()
    native = current_native_execution_budget()
    scope: AbstractContextManager[NativeExecutionBudget]
    if (
        isinstance(maximum_scratch_bytes, bool)
        or not isinstance(maximum_scratch_bytes, (int, np.integer))
        or maximum_scratch_bytes < 0
    ):
        raise ValueError(
            "Mixed scratch storage requires a nonnegative explicit integer bound."
        )
    owned_scope = native is None or maximum_scratch_bytes < native._limits[3]
    if native is None:
        scope = NativeExecutionBudget(
            max_work=maximum_work_units,
            max_geometry_queries=defaults.maximum_geometry_queries,
            max_cavity_cells=defaults.maximum_cavity_cells,
            max_scratch_bytes=maximum_scratch_bytes,
            max_wall_seconds=defaults.maximum_wall_seconds,
        )
    elif owned_scope:
        remaining = native.remaining()
        scope = NativeExecutionBudget(
            max_work=min(maximum_work_units, remaining.remaining_work_units),
            max_geometry_queries=remaining.remaining_geometry_queries,
            max_cavity_cells=remaining.maximum_cavity_cells,
            max_scratch_bytes=min(
                maximum_scratch_bytes, remaining.remaining_scratch_bytes
            ),
            max_wall_seconds=remaining.remaining_wall_seconds,
        )
    else:
        scope = nullcontext(native)
    with scope as budget:
        if budget is None:
            raise RuntimeError("Mixed adaptation lost its native resource owner.")
        workspace = (
            nullcontext(current_native_host_workspace())
            if current_native_host_workspace() is not None
            else budget.host_workspace()
        )
        with workspace:
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
            )
    if owned_scope:
        if budget.evidence is None:
            raise RuntimeError(
                "Standalone mixed adaptation lacks its ended original resource evidence."
            )
        outcome = outcome._replace(
            evidence=outcome.evidence._replace(
                execution_evidence=NativeExecutionRecord(budget.evidence)
            )
        )
    return outcome


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
) -> MixedAdaptationOutcome:
    """Build one complete mixed edit; canonical adaptation owns acceptance/rollback."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("Mixed adaptation requires a CellMesh.")
    dimension = mesh.topological_dimension
    if dimension == 2:
        if any(
            block.cell_kind != "quadrilateral" for block in mesh.blocks
        ) or mesh.ambient_dimension not in (2, 3):
            raise ValueError(
                "Two-dimensional mixed adaptation requires quadrilateral cells in 2D or 3D."
            )
        if layer_columns is not None:
            raise ValueError(
                "Layer-column axial semantics are defined only for three-dimensional mixed cells."
            )
    elif dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError(
            "Mixed volume adaptation requires three-dimensional coordinates."
        )
    if hierarchy is not None and not isinstance(hierarchy, MixedAdaptationHierarchy):
        raise TypeError("hierarchy must be MixedAdaptationHierarchy or None.")
    if layer_columns is None and hierarchy is not None:
        layer_columns = hierarchy.layer_columns
    if layer_columns is not None and not isinstance(layer_columns, MixedLayerColumns):
        raise TypeError("layer_columns must be MixedLayerColumns or None.")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in (maximum_cells, maximum_closure_steps, maximum_work_units)
    ):
        raise TypeError("Mixed adaptation budgets must be integers.")
    if maximum_cells < 1 or maximum_closure_steps < 1 or maximum_work_units < 1:
        raise ValueError("Mixed adaptation budgets must be positive.")
    _reserve_host_storage(
        sys.getsizeof({}) * 12
        + sys.getsizeof([]) * 20
        + sum(
            block.vertices.size * 8
            + block.cell_count
            * (sys.getsizeof(()) + len(_Cell._fields) * np.dtype(np.intp).itemsize + 256)
            for block in mesh.blocks
        )
        + mesh.coordinates.shape[0]
        * (
            sys.getsizeof(set())
            + 5 * sys.getsizeof(0)
            + 3 * sys.getsizeof(())
            + 3 * 128
            + mesh.ambient_dimension * np.dtype(np.float64).itemsize
        ),
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
    orbits = None if mesh.periodic_topology is None else PeriodicConstructionOrbits(mesh)
    edges = _protected_entity_keys(mesh, protected_edge_keys, 1)
    if dimension != 3 and np.asarray(protected_face_keys).size:
        raise ValueError("Protected face signatures are defined only for volume cells.")
    faces = (
        _protected_entity_keys(mesh, protected_face_keys, 2) if dimension == 3 else set()
    )
    if orbits is not None:
        for degree, protected_keys in ((1, edges), (2, faces)):
            if protected_keys:
                keys = [
                    tuple(int(value) for value in row if value >= 0)
                    for row in entity_keys(mesh, degree)
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
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    classes = (
        np.zeros(ids.size, dtype=np.int64)
        if cell_classes is None
        else np.asarray(cell_classes, dtype=np.int64)
    )
    if classes.shape != ids.shape:
        raise ValueError("cell_classes must align with concatenated source cells.")
    refine = set(
        int(value) for value in np.asarray(refine_cell_ids, dtype=np.int64).reshape(-1)
    )
    requested_refine = set(refine)
    coarsen = set(
        int(value) for value in np.asarray(coarsen_cell_ids, dtype=np.int64).reshape(-1)
    )
    if refine & coarsen or not (refine | coarsen).issubset(set(ids)):
        raise ValueError(
            "Refinement/coarsening marks must be disjoint current cell identities."
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
        raise ValueError("Protected mixed cell IDs must name current cells.")
    if refine & blocked:
        raise MixedTemplateClosureError("Marked refinement intersects a protected cell.")
    cells = []
    offset = 0
    for block in mesh.blocks:
        for row, identifier in zip(
            np.asarray(block.vertices), np.asarray(block.global_ids), strict=True
        ):
            cells.append(
                _Cell(
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
    _reserve_host_storage(
        256
        + 128
        * (
            sum(len(record.child_ids) for record in records)
            + (0 if hierarchy is None else len(hierarchy.identity_cells))
            + len(cells)
        )
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
        3 * sys.getsizeof({})
        + (0 if layer_columns is None else len(layer_columns.cell_ids))
        * (2 * sys.getsizeof(set()) + 6 * sys.getsizeof(0) + 192)
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
        raise ValueError("Layer-column identities must name current source cells.")
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
            "Mixed source identity admission exhausted its work budget."
        )
    _charge_host_work(identity_work)
    patch = _coarsen_patch(
        cells,
        records,
        coarsen,
        protected,
        blocked,
        vertex_ids,
        orbits,
        column_members=column_members,
        unchanged=unchanged,
    )
    restored, removed = patch.restored, patch.removed
    if orbits is not None and coarsen - removed - unchanged_marks:
        raise MixedTemplateClosureError(
            "Periodic coarsening requires complete compatible sibling and facet orbits."
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
            "Layer-column refinement conflicts with a marked sibling coarsening."
        )
    if refine & blocked:
        raise MixedTemplateClosureError(
            "Layer-column closure intersects a protected cell."
        )
    domains = []
    tables: dict[tuple[str, bool], tuple[_Template, ...]] = {}
    for cell in working:
        _reserve_host_storage(256)
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
            tables[cell.kind, axial] = _templates(cell.kind, axial)
            _reserve_host_storage(
                sum(
                    sys.getsizeof(template) + sys.getsizeof(template.children)
                    for template in tables[cell.kind, axial]
                )
            )
        templates = list(tables[cell.kind, axial])
        red = [
            template for template in templates if template.interior_diagonal is not None
        ]
        if red:
            selected = red[
                min(
                    (_interior_diagonal_key(template, cell.vertices), index)
                    for index, template in enumerate(red)
                )[1]
            ]
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
            if _permitted_subdivision(cell, template, edges, faces)
        ]
        if not templates:
            raise MixedTemplateClosureError(
                "Marked refinement cannot preserve protected edge/face subdivision signatures."
            )
        domains.append(templates)
    _reserve_host_storage(
        sys.getsizeof([]) + 2 * len(working) * np.dtype(np.intp).itemsize
    )
    chosen, steps, work = _closure(
        working,
        domains,
        maximum_closure_steps,
        maximum_work_units - identity_work,
        orbits,
    )
    work += identity_work
    work += sum(template.children.size for template in chosen)
    if work > maximum_work_units:
        raise MixedTemplateClosureError(
            "Mixed child construction exhausted its work budget."
        )
    _charge_host_work(sum(template.children.size for template in chosen))
    count = sum(template.children.shape[0] for template in chosen)
    if count > maximum_cells:
        raise MixedTemplateClosureError(
            "Mixed refinement exceeds its target cell budget."
        )
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
                template.children.shape[1]
                * (np.dtype(np.int64).itemsize + 3 * sys.getsizeof(0.0))
                + sys.getsizeof(())
                + len(_Sibling._fields) * np.dtype(np.intp).itemsize
                + 512
                + sys.getsizeof(cell.name)
                + sys.getsizeof(template.name)
            )
            for cell, template in zip(working, chosen, strict=True)
        ),
    )
    points: dict[ConstructionPointKey, int] = {
        ((int(identifier), Fraction(1)),): int(identifier) for identifier in vertex_ids
    }
    stencils: dict[int, ConstructionPointKey] = {
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
    for cell, template in zip(working, chosen, strict=True):
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
            child = []
            for key in _point_keys(cell.kind, cell.vertices, reference):
                if key not in points:
                    _reserve_host_storage(
                        512 + len(key) * 64 + mesh.ambient_dimension * 8
                    )
                    points[key] = next_vertex
                    stencils[next_vertex] = key
                    coordinates[next_vertex] = sum(
                        float(weight) * coordinates[identifier]
                        for identifier, weight in key
                    )
                    next_vertex += 1
                child.append(points[key])
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
                        EntityLineageKind.PRESERVED
                        if template.name == "identity"
                        else EntityLineageKind.REFINED_FROM
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
                _Sibling(
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
    for child, parent in zip(coarsening_fine, coarsening_parent, strict=True):
        relation_sources.append(child)
        relation_targets.append(parent)
        relation_kinds.append(int(EntityLineageKind.COARSENED_INTO))
    corner_slots = sum(row.size for values in target.values() for _, row in values)
    _reserve_host_storage(
        corner_slots * (3 * np.dtype(np.int64).itemsize + 128)
        + sum(len(values) for values in target.values()) * 128,
    )
    used_vertices = {
        int(value) for block in target.values() for _, row in block for value in row
    }
    original_vertices = set(vertex_ids.tolist())
    # Dense scientific vertex order is an authored control. Retain every
    # surviving source row in that order; append only newly allocated identities
    # in allocator order. All same-family templates retain their parent corners,
    # so nested/partial inverse epochs recover the original rows inductively.
    used = np.asarray(
        [int(identifier) for identifier in vertex_ids if int(identifier) in used_vertices]
        + sorted(used_vertices - original_vertices),
        dtype=np.int64,
    )
    local = {int(identifier): index for index, identifier in enumerate(used)}
    source_kinds = {block.name: block.cell_kind for block in mesh.blocks}
    blocks = tuple(
        TopologyEditBlock(
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
        + used.size * mesh.ambient_dimension * np.dtype(np.float64).itemsize
    )
    sources = np.full((used.size, width), -1, dtype=np.int64)
    weights = np.zeros((used.size, width), dtype=np.float64)
    for index, identifier in enumerate(used):
        for slot, (source, weight) in enumerate(stencils[int(identifier)]):
            sources[index, slot], weights[index, slot] = source, weight
    target_coordinates = np.stack([coordinates[int(identifier)] for identifier in used])
    # The canonical connectivity owner materializes bounded edge/face routes and
    # sorting workspaces from these actual child occurrences, not all pairs.
    entity_occurrences = sum(
        block.cells.shape[0]
        * sum(len(row) for row in reference_cell_topology(block.cell_kind).entities[1:-1])
        for block in blocks
    )
    _reserve_host_storage(
        entity_occurrences * (12 * np.dtype(np.int64).itemsize + 2 * sys.getsizeof(()))
    )
    probe = CellMesh(
        target_coordinates,
        tuple(
            CellBlock(block.name, block.cell_kind, block.cells, global_ids=block.cell_ids)
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
        )
    )
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
        current_keys = entity_keys(mesh, degree)
        current_ids = np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
        _reserve_host_storage(current_keys.size * (64 + 3 * np.dtype(np.int64).itemsize))
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
                PrescribedEntityIds(
                    degree,
                    np.asarray([key for key, _ in restored_keys], dtype=np.int64),
                    np.asarray(
                        [identifier for _, identifier in restored_keys], dtype=np.int64
                    ),
                )
            )
    relations = []
    for degree in range(dimension + 1):
        keys = entity_keys(mesh, degree)
        if degree == dimension:
            relations.append(
                EntityRelations(
                    degree,
                    np.asarray(relation_sources, dtype=np.int64)[:, None],
                    np.asarray(relation_targets, dtype=np.int64)[:, None],
                    np.asarray(relation_kinds, dtype=np.int32),
                )
            )
        else:
            target_keys = entity_keys(probe, degree)
            inherited_source, inherited_target, inherited_kinds = [], [], []
            if 0 < degree < dimension:
                # Index source entities by one vertex every admissible witness must
                # contain, preserving source-row order within each target entity.
                _reserve_host_storage(
                    keys.shape[0] * (2 * sys.getsizeof(set()) + 128)
                    + keys.size * (sys.getsizeof(0) + 64)
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
                                128 + 2 * keys.shape[1] * np.dtype(np.int64).itemsize
                            )
                            inherited_source.append(keys[row])
                            inherited_target.append(target_key)
                            inherited_kinds.append(int(EntityLineageKind.REFINED_FROM))
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
                                128 + 2 * keys.shape[1] * np.dtype(np.int64).itemsize
                            )
                            inherited_source.append(keys[row])
                            inherited_target.append(target_key)
                            inherited_kinds.append(int(EntityLineageKind.COARSENED_INTO))
            relations.append(
                EntityRelations(
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

    def witnesses(
        fine: list[int], coarse: list[int], vertices: list[np.ndarray]
    ) -> NestedReferenceWitnesses | None:
        if not fine:
            return None
        corner_count = max(row.shape[0] for row in vertices)
        _reserve_host_storage(
            len(fine)
            * (
                corner_count * dimension * np.dtype(np.float64).itemsize
                + 2 * np.dtype(np.int64).itemsize
            )
        )
        padded = np.zeros((len(fine), corner_count, dimension), dtype=np.float64)
        for index, row in enumerate(vertices):
            padded[index, : row.shape[0]] = row
        return NestedReferenceWitnesses(
            np.asarray(fine, dtype=np.int64), np.asarray(coarse, dtype=np.int64), padded
        )

    refinement = witnesses(refinements, parents, references)
    coarsening = witnesses(coarsening_fine, coarsening_parent, coarsening_reference)
    edit = CellTopologyEdit(
        "nested_adaptation"
        if coarsening is not None and refinement is not None
        else "nested_coarsening"
        if coarsening is not None
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
        coarsening=coarsening,
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
    return MixedAdaptationOutcome(edit, successor, evidence)
