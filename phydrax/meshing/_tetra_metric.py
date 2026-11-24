#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-aware SPD metric adaptation kernels for tetrahedral cell meshes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from fractions import Fraction
from itertools import combinations
from math import lcm, sqrt
from typing import Any, assert_never, cast, final, NamedTuple, NoReturn

import equinox as eqx
import numpy as np
from numpy.typing import ArrayLike

from .._exact_kernels import exact_kernel_parallelism
from .._fingerprint import canonical_fingerprint
from .._geometry_predicates import orient3d, PredicateMode, PredicateSign
from .._meshcore import (
    _raise_for_call,
    interval_program_execute,
    IntervalBoxProgram,
    IntervalBranchProgram,
    metric_band_quanta,
    metric_collapse_positions,
    metric_congruence_defects,
    metric_hull_frame,
    metric_interpolations,
    metric_lengths,
    metric_logarithms,
    metric_midpoints,
    metric_relative_determinants,
    metric_relocation_pulls,
    metric_tensor_properties,
    metric_tetra_quality,
    metric_tetra_quality_min,
    MetricCarrierOrder,
    MetricCollapseStar,
    MetricPositionVerdict,
    MetricRelocationCandidate,
    MetricRelocationEnd,
    MetricRelocationStar,
    MetricSearchEnd,
    MetricSource,
    parallel_for,
    PlcBoundaryPolicy,
    relocated_tet_mesh_arrays,
    TetMesh3D,
    TetMeshArrays,
    TetMeshEdit,
    TetMeshEditKind,
    TetMeshEditStatus,
    TetMeshImplicitSource,
    TetMeshMetricAdaptation,
    TetMeshPeriodicState,
    TetMeshSourceComplex,
    tetrahedra_positive_orientation,
)
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_complex import TetrahedralConnectivity
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellBlock
from ..linalg._exact import solve_exact
from ..typing import ConvertibleToArray
from ._lineage import EntityLineageKind
from ._metric import _tensor_array
from ._periodic import periodic_entity_orbit_map, stage_periodic_topology_edit
from ._result import CellMeshingResult
from ._topology_edit import (
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    source_family_blocks,
)
from .providers._native_periodic_tet import (
    tet_mesh_periodic_edit,
    tet_mesh_periodic_state,
)


_LOWER = 1.0 / np.sqrt(2.0)
_UPPER = np.sqrt(2.0)
_IMPROVEMENT = 1.0e-6
# A metric split point is the metric midpoint rounded to the dyadic grid
# 2**-bits of its edge: 5 bits place it within 1/64 of the edge length.
_SPLIT_PARAMETER_BITS = 5
# Band violations |log(l / band edge)| are compared in quanta of 2**-10: far
# below the band's log half-width log(sqrt(2)) ~ 0.35, so an edge entering
# the band is always progress, while moves changing no violation by a quantum
# are not progress (they would otherwise smooth without bound).
_VIOLATION_QUANTUM = 2.0**-10
# kMetricDeterminantFloor of native quality improvement (improve3d.hpp): below
# this relative determinant |det E| / max_v prod |e| a cell's binary64 metric
# determinant det(E^T E) is not certainly positive. No edit creates such a cell.
_DETERMINANT_FLOOR = 2.0**-22
# Coordinates lie in the exact domain, of magnitude at most 2^120, so an edge
# vector has components of at most 2^121 and a form d^T M d sums nine products
# of at most 2^242 times the largest entry. Entries up to this bound cannot
# overflow it: a larger tensor has no finite length to measure.
_METRIC_ENTRY_BOUND = 2.0**778
# An interior vertex moves along the coordinate axes.
_AXES = np.eye(3, dtype=np.float64)
_AXES.setflags(write=False)
_TETRA_EDGE_CORNERS = tuple(combinations(range(4), 2))
_TETRA_FACE_CORNERS = ((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2))
_TETRA_EDGES = np.asarray(_TETRA_EDGE_CORNERS, dtype=np.int32)
_TETRA_FACES = np.asarray(_TETRA_FACE_CORNERS, dtype=np.int32)
_TETRA_EDGES.setflags(write=False)
_TETRA_FACES.setflags(write=False)


def _edge_key(vertices: Iterable[int], /) -> tuple[int, int]:
    try:
        first, second = vertices
    except ValueError as error:
        raise ValueError(
            "A tetra edge key must contain exactly two vertex IDs."
        ) from error
    first, second = int(first), int(second)
    return (first, second) if first <= second else (second, first)


def _face_key(vertices: Iterable[int], /) -> tuple[int, int, int]:
    values = sorted(int(value) for value in vertices)
    if len(values) != 3:
        raise ValueError("A tetra face key must contain exactly three vertex IDs.")
    return values[0], values[1], values[2]


def _tetra_key(vertices: Iterable[int], /) -> tuple[int, int, int, int]:
    values = tuple(int(value) for value in vertices)
    if len(values) != 4:
        raise ValueError("A tetra cell must contain exactly four vertex IDs.")
    return values[0], values[1], values[2], values[3]


class TetraMetricStatus(StrEnum):
    COMPLETE = "complete"
    STALLED = "stalled"
    NUMERICAL_STOP = "numerical_stop"


class TetraMetricUnmetKind(StrEnum):
    """Why an out-of-band edge remains when adaptation stops.

    ``SOURCE_BOUNDARY``: a short edge whose collapse the source boundary forbids
    in both directions (each endpoint fixed, protected, pinned at a source
    corner, or carried by a source triangle/edge that does not hold the other
    endpoint). ``UNREPRESENTABLE``: a long edge refused a split because no
    short dyadic point of it has a consistent binary64 realization.
    ``UNIMPROVED``: an admissible operation exists, but none strictly decreased
    the termination objective within the determinant floor and exact checks.
    """

    SOURCE_BOUNDARY = "source_boundary"
    UNREPRESENTABLE = "unrepresentable"
    UNIMPROVED = "unimproved"


class TetraMetricUnmetEdge(NamedTuple):
    """Witness of one out-of-band edge: its target vertex IDs and metric length."""

    kind: TetraMetricUnmetKind
    vertex_ids: tuple[int, int]
    metric_length: float


class TetraMetricEvidence(StrictModule, NonTrainableState):
    """Actual native operation counts and final SPD-metric measurements.

    The metric lengths, the metric quality and the region volume error are
    binary64 values of the deterministic native kernels (``metric_lengths``,
    ``metric_tetra_quality``, ``metric_relative_determinants``): the same
    mesh gives the same values on every platform, within the error bounds
    stated at the kernels (``exp`` 4 ulp, ``log`` 6 ulp, ``x^(2/3)`` relative
    ``2^-43``). They are measurements of the adapted mesh. During adaptation
    every acceptance that uses such a value is an exact comparison of kernel
    values, and the validity of an edit rests on exact predicates alone.
    """

    status: TetraMetricStatus = eqx.field(static=True)
    passes: int = eqx.field(static=True)
    splits: int = eqx.field(static=True)
    collapses: int = eqx.field(static=True)
    flips23: int = eqx.field(static=True)
    flips32: int = eqx.field(static=True)
    ring_reconnections: int = eqx.field(static=True)
    relocations: int = eqx.field(static=True)
    rejected_topology: int = eqx.field(static=True)
    rejected_protected: int = eqx.field(static=True)
    rejected_uncertain: int = eqx.field(static=True)
    rejected_invalid: int = eqx.field(static=True)
    measured_edges: int = eqx.field(static=True)
    out_of_range_edges: int = eqx.field(static=True)
    minimum_metric_length: float = eqx.field(static=True)
    maximum_metric_length: float = eqx.field(static=True)
    minimum_metric_quality: float = eqx.field(static=True)
    maximum_region_volume_error: float = eqx.field(static=True)
    source_boundary_edges: int = eqx.field(static=True)
    unrepresentable_edges: int = eqx.field(static=True)
    unimproved_edges: int = eqx.field(static=True)
    unmet_edges: tuple[TetraMetricUnmetEdge, ...] = eqx.field(static=True)
    implicit_refusals: tuple[int, int] | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        status: TetraMetricStatus,
        passes: int,
        splits: int,
        collapses: int,
        flips23: int,
        flips32: int,
        ring_reconnections: int,
        relocations: int,
        rejected_topology: int,
        rejected_protected: int,
        rejected_uncertain: int,
        rejected_invalid: int,
        measured_edges: int,
        out_of_range_edges: int,
        minimum_metric_length: float,
        maximum_metric_length: float,
        minimum_metric_quality: float,
        maximum_region_volume_error: float,
        unmet_edges: tuple[TetraMetricUnmetEdge, ...],
        implicit_refusals: tuple[int, int] | None = None,
    ) -> None:
        if not isinstance(status, TetraMetricStatus):
            raise TypeError("status must be TetraMetricStatus.")
        counts = (
            passes,
            splits,
            collapses,
            flips23,
            flips32,
            ring_reconnections,
            relocations,
            rejected_topology,
            rejected_protected,
            rejected_uncertain,
            rejected_invalid,
            measured_edges,
            out_of_range_edges,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("Tetra metric counts must be nonnegative integers.")
        if implicit_refusals is not None and (
            len(implicit_refusals) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in implicit_refusals
            )
        ):
            raise ValueError(
                "Implicit source refusal counts must be nonnegative integers."
            )
        values = (
            minimum_metric_length,
            maximum_metric_length,
            minimum_metric_quality,
            maximum_region_volume_error,
        )
        if any(
            not isinstance(value, float) or not np.isfinite(value) or value < 0.0
            for value in values
        ):
            raise ValueError("Tetra metric measurements must be finite and nonnegative.")
        if out_of_range_edges > measured_edges:
            raise ValueError("out_of_range_edges cannot exceed measured_edges.")
        unmet = tuple(unmet_edges)
        if any(
            not isinstance(edge, TetraMetricUnmetEdge)
            or not isinstance(edge.kind, TetraMetricUnmetKind)
            for edge in unmet
        ):
            raise TypeError("unmet_edges must hold TetraMetricUnmetEdge records.")
        if len(unmet) != out_of_range_edges:
            raise ValueError("Every out-of-range edge needs exactly one unmet record.")
        kinds = [edge.kind for edge in unmet]
        self.status = status
        self.passes = passes
        self.splits = splits
        self.collapses = collapses
        self.flips23 = flips23
        self.flips32 = flips32
        self.ring_reconnections = ring_reconnections
        self.relocations = relocations
        self.rejected_topology = rejected_topology
        self.rejected_protected = rejected_protected
        self.rejected_uncertain = rejected_uncertain
        self.rejected_invalid = rejected_invalid
        self.measured_edges = measured_edges
        self.out_of_range_edges = out_of_range_edges
        self.minimum_metric_length = minimum_metric_length
        self.maximum_metric_length = maximum_metric_length
        self.minimum_metric_quality = minimum_metric_quality
        self.maximum_region_volume_error = maximum_region_volume_error
        self.source_boundary_edges = kinds.count(TetraMetricUnmetKind.SOURCE_BOUNDARY)
        self.unrepresentable_edges = kinds.count(TetraMetricUnmetKind.UNREPRESENTABLE)
        self.unimproved_edges = kinds.count(TetraMetricUnmetKind.UNIMPROVED)
        self.unmet_edges = unmet
        self.implicit_refusals = implicit_refusals
        identity = {
            "kind": "tetra-metric-evidence",
            "status": status.value,
            "counts": counts,
            "measurements": values,
            "unmet": tuple(
                (edge.kind.value, edge.vertex_ids, edge.metric_length) for edge in unmet
            ),
        }
        if implicit_refusals is not None:
            identity["implicit_refusals"] = implicit_refusals
        self.evidence_id = canonical_fingerprint(identity)

    @property
    def converged(self) -> bool:
        return self.status is TetraMetricStatus.COMPLETE


class TetraMetricOutcome(NamedTuple):
    """Numerical topology edit, vertex metric, and truthful termination evidence."""

    edit: CellTopologyEdit
    metric: np.ndarray
    evidence: TetraMetricEvidence
    exact_source_vertices: tuple[tuple[Fraction, Fraction, Fraction], ...]


class _ImplicitMetricSource(NamedTuple):
    value: IntervalBranchProgram
    gradient: IntervalBoxProgram
    tolerance: float
    source_id: str
    source_revision: str
    binding_id: str
    frame_id: str


class _TetTopology(NamedTuple):
    edge_cells: dict[tuple[int, int], tuple[int, ...]]
    face_cells: dict[tuple[int, int, int], tuple[int, ...]]
    vertex_cells: dict[int, tuple[int, ...]]
    cell_vertex_ids: np.ndarray
    vertex_row: dict[int, int]
    cell_row: dict[int, int]
    facet_row: dict[tuple[int, int, int], int]
    segment_row: dict[tuple[int, int], int]


type _SourceTetRecord = tuple[
    int,
    tuple[int, ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[tuple[int, ...], ...],
    int,
]


@final
@dataclass(frozen=True, slots=True, eq=False)
class _SourceRecords:
    """Source cell records in ascending cell ID with their binary64 boxes.

    ``lower`` and ``upper`` repeat each record's box corners as arrays, one row
    per record, so a point is compared against every box in one operation.
    """

    records: tuple[_SourceTetRecord, ...]
    lower: np.ndarray
    upper: np.ndarray

    def __iter__(self) -> Iterator[_SourceTetRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


class _EditCounters(NamedTuple):
    splits: int = 0
    collapses: int = 0
    flips23: int = 0
    flips32: int = 0
    ring_reconnections: int = 0
    relocations: int = 0
    rejected_topology: int = 0
    rejected_protected: int = 0
    rejected_uncertain: int = 0
    rejected_invalid: int = 0


@final
@dataclass(slots=True)
class _RelocationSearch:
    """One pending proposal and its finite continuation on an unchanged star."""

    candidates: Iterator[dict[str, Any]]
    proposal: dict[str, Any] | None = None


type _RelocationCaches = tuple[dict[int, _RelocationSearch | None], ...]


@dataclass(slots=True)
class _NativeState:
    native: TetMesh3D
    snapshot: TetMeshArrays
    source_mesh: CellMesh
    source_points: np.ndarray
    source_exact_points: tuple[tuple[Fraction, Fraction, Fraction], ...]
    source_has_authored_rounding: bool
    implicit_groups: frozenset[int]
    target_mesh: CellMesh
    source_metric: np.ndarray
    source_metric_log: np.ndarray
    periodic_vertex_windings: dict[int, np.ndarray]
    # ``tet_mesh_periodic_state`` of ``target_mesh`` with its cell regions in
    # block order, replaced with every committed periodic edit (``None`` off a
    # periodic lift); the next edit's source quotient state.
    periodic_state: TetMeshPeriodicState | None
    source_cell_ids: np.ndarray
    source_vertex_ids: np.ndarray
    source_vertex_rows: dict[int, int]
    source_cell_vertices: np.ndarray
    source_cell_regions: np.ndarray
    source_cell_blocks: np.ndarray
    source_region_volumes: dict[int, float]
    source_identity: tuple[str, str, str, str]
    vertex_target_ids: dict[int, int]
    cell_target_ids: dict[int, int]
    cell_blocks: dict[int, int]
    cell_ancestry: dict[int, dict[int, EntityLineageKind]]
    vertex_stencils: dict[int, dict[int, Fraction]]
    metric_by_vertex: dict[int, np.ndarray]
    fixed_vertices: set[int]
    protected_edge_ids: set[tuple[int, int]]
    source_edge_ids: dict[tuple[int, int], int]
    edge_classes: dict[tuple[int, int], int]
    source_face_group_rows: dict[int, int]
    source_segment_group_rows: dict[int, int]
    # Source vertex IDs carrying each facet group row and segment group row.
    source_face_vertices: dict[int, frozenset[int]]
    source_segment_vertices: dict[int, frozenset[int]]
    # Source-vertex sets of source triangles on a declared exact plane.
    declared_source_faces: frozenset[frozenset[int]]
    # Source triangles (source vertex IDs) of each planar facet group row.
    source_group_triangles: dict[int, tuple[tuple[int, int, int], ...]]
    next_native_vertex: int
    next_native_cell: int
    next_native_facet: int
    next_native_segment: int
    next_target_vertex: int
    next_target_cell: int
    block_families: tuple[tuple[str, str], ...]
    history_vertex: list[tuple[int, int, EntityLineageKind]]
    history_cells: list[tuple[int, int, EntityLineageKind]]
    changed_source_cell_ids: set[int]
    # Edges refused for having no binary64-consistent split point.
    unrepresentable_edges: set[tuple[int, int]]
    # Host tables derived from exactly ``snapshot`` (see ``_snapshot_tables``).
    tables: _SnapshotTables | None = None
    # Exact affine frames of carrier simplices, keyed by their ordered source
    # vertex IDs (``_source_carrier_frame``): a pure function of the immutable
    # source, built once per adaptation.
    source_frames: dict[tuple[int, ...], _EntityFrame] = field(default_factory=dict)
    # Corners, projector and orthonormal basis of a carrier's affine hull:
    # kernel values that depend only on the same immutable source.
    source_projections: dict[
        tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = field(default_factory=dict)
    # The native exact source (``_source_bank``), created on first use.
    source_bank: _SourceBank | None = None
    _host_source_records: _SourceRecords | None = field(default=None, init=False)

    @property
    def source_records(self) -> _SourceRecords:
        """Exact inverse records for the host proposal engine, not the native loop."""
        records = self._host_source_records
        if records is None:
            records = _source_tetra_inverse(
                self.source_points,
                self.source_cell_vertices,
                self.source_vertex_ids,
                self.source_cell_ids,
                exact_points=self.source_exact_points,
            )
            self._host_source_records = records
        return records


@dataclass(slots=True)
class _SnapshotTables:
    """Host tables of one native snapshot, built lazily and shared by every
    phase that reads that snapshot.

    ``snapshot`` is the identity key: each native commit replaces
    ``_NativeState.snapshot`` together with every state map these tables read
    (``vertex_target_ids``, ``metric_by_vertex``, ``target_mesh``), so a
    refused edit, a retry and the next candidate scan reuse the tables.
    """

    snapshot: TetMeshArrays
    topology: _TetTopology | None = None
    edge_rows: np.ndarray | None = None
    metric_table: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    edge_controls: (
        tuple[dict[tuple[int, int], int], dict[tuple[int, int], bool]] | None
    ) = None
    boundary_records: _BoundaryRecords | None = None
    facet_groups: dict[tuple[int, int, int], int] | None = None
    orbit_groups: dict[int, tuple[tuple[tuple[int, ...], ...], ...]] | None = None
    edge_orbits: tuple[tuple[tuple[int, int], ...], ...] | None = None
    face_orbits: tuple[tuple[tuple[int, int, int], ...], ...] | None = None
    vertex_orbits: tuple[tuple[int, ...], ...] | None = None
    vertex_orbit_members: dict[int, tuple[int, ...]] | None = None
    replacement_keys: (
        dict[str, tuple[tuple[tuple[int, ...], ...], frozenset[tuple[int, ...]]]] | None
    ) = None


class _CandidateTetraSnapshot(NamedTuple):
    points: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_regions: np.ndarray
    vertex_ids: np.ndarray
    cell_ids: np.ndarray


def _band_violations(lengths: np.ndarray, /) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Max-first quantized long and short distances ``|log(l / band edge)|``.

    Each violation is counted in quanta ``_VIOLATION_QUANTUM`` rounded up, so
    any violation is at least one quantum and an in-band edge is none. A
    length that is zero, infinite or not a number has no finite count: it was
    not measured, which is neither inside the band nor a violation of it.
    """
    above, below = metric_band_quanta(
        np.asarray(lengths, dtype=np.float64).reshape((-1,))
    )
    if not (np.all(np.isfinite(above)) and np.all(np.isfinite(below))):
        raise FloatingPointError("A metric edge length is not a positive finite number.")

    def quanta(counts: np.ndarray) -> tuple[int, ...]:
        return tuple(
            sorted((int(value) for value in counts[counts != 0.0].tolist()), reverse=True)
        )

    return quanta(above), quanta(below)


def _objective_decreases(
    removed: np.ndarray, added: np.ndarray, quality_gain: bool, /
) -> bool:
    """Whether a local edit strictly decreases the termination objective.

    The objective is lexicographic: the multiset of quantized long violations,
    then of short violations (``_band_violations``), each in max-first order
    (compare violations sorted descending; a proper prefix is smaller), then
    the cell quality. ``removed`` and ``added`` are the metric lengths of the
    measured edges the edit deletes or creates (or whose length it changes),
    and ``quality_gain`` states that the edit replaces cells whose minimum
    quality ``q`` only by cells better than ``q + _IMPROVEMENT``. Each
    component is a multiset order on a total order, compatible with multiset
    union, so a local decrease is a global one; the quantized violations take
    finitely many values and a mesh has finitely many binary64 states, so
    strictly decreasing edits terminate.
    """
    removed_long, removed_short = _band_violations(removed)
    added_long, added_short = _band_violations(added)
    if added_long != removed_long:
        return added_long < removed_long
    if added_short != removed_short:
        return added_short < removed_short
    return quality_gain


def _metric_logarithms(tensors: np.ndarray, /) -> np.ndarray:
    logarithms, valid = metric_logarithms(tensors)
    if not np.all(valid):
        raise FloatingPointError("Log-Euclidean interpolation requires SPD metrics.")
    return logarithms


def _clears_determinant_floor(corners: np.ndarray, /) -> bool:
    """Whether every created cell clears the improvement determinant floor."""
    _, relative = metric_relative_determinants(corners.reshape((-1, 4, 3)))
    return bool(np.all(relative > _DETERMINANT_FLOOR))


def _floor_oriented(corners: np.ndarray, /) -> bool:
    """Whether cells are positively oriented above the determinant floor.

    The evaluated determinant is within ``50 * 2^-53`` times the largest vertex
    edge product of the exact one, and a relative determinant above
    ``_DETERMINANT_FLOOR`` (2^-22) exceeds that bound by orders of magnitude,
    so the float sign is the exact orientation.
    """
    determinants, relative = metric_relative_determinants(corners.reshape((-1, 4, 3)))
    return bool(np.all(determinants > 0.0)) and bool(
        np.all(relative > _DETERMINANT_FLOOR)
    )


def _metric_midpoint(
    first: np.ndarray,
    second: np.ndarray,
    first_metric: np.ndarray,
    second_metric: np.ndarray,
    /,
) -> tuple[float, np.ndarray]:
    """Parameter and point of an edge at half its metric length."""
    parameters, points = metric_midpoints(
        first[None], second[None], first_metric[None], second_metric[None]
    )
    return float(parameters[0]), points[0]


def _tetra_orientation(
    points: np.ndarray, cells: np.ndarray, mode: PredicateMode, /
) -> np.ndarray:
    """Exact orientation signs; a filtered-mode uncertainty is resolved exactly."""
    if cells.shape[0] == 0:
        return np.zeros((0,), dtype=np.int8)
    corners = points[cells]
    result = orient3d(
        corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3], mode=mode
    )
    signs = np.asarray(result.signs).copy()
    certain = np.asarray(result.certain)
    if mode is not PredicateMode.EXACT and not np.all(certain):
        uncertain = np.flatnonzero(~certain)
        resolved = orient3d(
            *(corners[uncertain, corner] for corner in range(4)), mode=PredicateMode.EXACT
        )
        signs[uncertain] = np.asarray(resolved.signs)
        certain = certain.copy()
        certain[uncertain] = np.asarray(resolved.certain)
    oriented = np.select(
        (signs == PredicateSign.POSITIVE, signs == PredicateSign.NEGATIVE), (1, -1), 0
    )
    return np.where(~certain, 2, oriented).astype(np.int8)


def _grouped_rows(
    keys: np.ndarray, rows: np.ndarray, /
) -> tuple[list[list[int]], list[tuple[int, ...]]]:
    """Distinct key rows in ascending lexicographic order, each with its
    ascending incidence rows (repeats kept), from ``(n, k)`` keys."""
    if keys.shape[0] == 0:
        return [], []
    order = np.lexsort(
        (rows, *(keys[:, column] for column in range(keys.shape[1] - 1, -1, -1)))
    )
    ordered_keys = keys[order]
    ordered_rows = rows[order].tolist()
    starts = np.concatenate(
        ([0], np.flatnonzero(np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)) + 1)
    )
    bounds = np.append(starts, keys.shape[0]).tolist()
    return ordered_keys[starts].tolist(), [
        tuple(ordered_rows[begin:end]) for begin, end in zip(bounds[:-1], bounds[1:])
    ]


def _topology(snapshot: TetMeshArrays, /) -> _TetTopology:
    """Edge, face and vertex cell incidence of a snapshot.

    Every dict iterates its keys in ascending order and lists incident cell
    rows in ascending order (``_edge_key``/``_face_key`` keys, Python ints).
    """
    vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    cell_ids = np.asarray(snapshot.cell_ids, dtype=np.int64)
    local_cells = np.asarray(snapshot.tetrahedra, dtype=np.int64).reshape((-1, 4))
    cells = np.asarray(vertex_ids[local_cells], dtype=np.int64).reshape((-1, 4))
    rows = np.arange(cells.shape[0], dtype=np.int64)
    edge_keys, edge_incidence = _grouped_rows(
        np.sort(cells[:, _TETRA_EDGES], axis=2).reshape((-1, 2)), np.repeat(rows, 6)
    )
    face_keys, face_incidence = _grouped_rows(
        np.sort(cells[:, _TETRA_FACES], axis=2).reshape((-1, 3)), np.repeat(rows, 4)
    )
    vertex_keys, vertex_incidence = _grouped_rows(
        cells.reshape((-1, 1)), np.repeat(rows, 4)
    )
    facet_keys = np.sort(vertex_ids[np.asarray(snapshot.faces, dtype=np.int64)], axis=1)
    segment_keys = np.sort(
        vertex_ids[np.asarray(snapshot.segments, dtype=np.int64)], axis=1
    )
    return _TetTopology(
        dict(zip(map(tuple, edge_keys), edge_incidence, strict=True)),
        dict(zip(map(tuple, face_keys), face_incidence, strict=True)),
        {key[0]: value for key, value in zip(vertex_keys, vertex_incidence, strict=True)},
        cells,
        {int(identifier): row for row, identifier in enumerate(vertex_ids.tolist())},
        {int(identifier): row for row, identifier in enumerate(cell_ids.tolist())},
        {
            (int(row[0]), int(row[1]), int(row[2])): index
            for index, row in enumerate(facet_keys.tolist())
        },
        {
            (int(row[0]), int(row[1])): index
            for index, row in enumerate(segment_keys.tolist())
        },
    )


def _snapshot_tables(state: _NativeState, /) -> _SnapshotTables:
    tables = state.tables
    if tables is None or tables.snapshot is not state.snapshot:
        tables = _SnapshotTables(state.snapshot)
        state.tables = tables
    return tables


def _state_topology(state: _NativeState, /) -> _TetTopology:
    """Topology of the current snapshot, built once per snapshot."""
    tables = _snapshot_tables(state)
    if tables.topology is None:
        tables.topology = _topology(tables.snapshot)
    return tables.topology


def _state_edge_rows(state: _NativeState, /) -> np.ndarray:
    """``(E, 2)`` snapshot vertex rows of the topology's edges, in key order."""
    tables = _snapshot_tables(state)
    if tables.edge_rows is None:
        topology = _state_topology(state)
        vertex_ids = np.asarray(tables.snapshot.vertex_ids, dtype=np.int64)
        keys = np.asarray(list(topology.edge_cells), dtype=np.int64).reshape((-1, 2))
        order = np.argsort(vertex_ids, kind="stable")
        rows = order[np.searchsorted(vertex_ids[order], keys)]
        rows.setflags(write=False)
        tables.edge_rows = rows
    return tables.edge_rows


def _disjoint_candidates(
    candidates: list[tuple[tuple[Any, ...], Any, frozenset[int]]], /
) -> list[Any]:
    """Deterministic greedy maximal set over complete finite cavity records."""
    selected: list[Any] = []
    occupied: set[int] = set()
    for _, value, cavity in sorted(candidates, key=lambda row: row[0]):
        if occupied.isdisjoint(cavity):
            selected.append(value)
            occupied.update(cavity)
    return selected


def _oriented_tetrahedra(
    points: np.ndarray,
    topology: _TetTopology,
    cells: np.ndarray,
    mode: PredicateMode,
    /,
    *,
    reversible: bool = False,
) -> np.ndarray | None:
    """Local rows of cells that are all positively oriented in the given order.

    A cell is never reoriented on its own: a negatively oriented cell in an
    otherwise positive set is inverted. ``reversible`` admits a set whose cells
    are all negative, i.e. given in the uniformly mirrored corner order, and
    returns it uniformly reversed.
    """
    local = np.asarray(
        [[topology.vertex_row[int(vertex)] for vertex in cell] for cell in cells],
        dtype=np.int64,
    ).reshape((-1, 4))
    signs = _tetra_orientation(points, local, mode)
    if np.all(signs == 1):
        return local
    if reversible and np.all(signs == -1):
        local[:, [2, 3]] = local[:, [3, 2]]
        return local
    return None


def _edge_ring(
    topology: _TetTopology, edge: tuple[int, int], cell_rows: tuple[int, ...], /
) -> tuple[int, ...] | None:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for cell_row in cell_rows:
        other = [
            int(vertex)
            for vertex in topology.cell_vertex_ids[cell_row]
            if int(vertex) not in edge
        ]
        if len(other) != 2:
            return None
        a, b = other
        adjacency[a].add(b)
        adjacency[b].add(a)
    if len(adjacency) < 3 or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None
    start = min(adjacency)
    previous = start
    current = min(adjacency[start])
    ring = [start, current]
    while current != start:
        following = next(
            (value for value in sorted(adjacency[current]) if value != previous), None
        )
        if following is None or len(ring) > len(adjacency):
            return None
        previous, current = current, following
        if current != start:
            if current in ring:
                return None
            ring.append(current)
    if len(ring) != len(adjacency):
        return None
    return tuple(ring)


def _best_ring_triangulation(
    points: np.ndarray,
    metrics: np.ndarray,
    topology: _TetTopology,
    edge: tuple[int, int],
    ring: tuple[int, ...],
    mode: PredicateMode,
    /,
) -> tuple[tuple[tuple[int, int, int, int], ...], float] | None:
    """Band-first, then max-min quality triangulation of a finite edge ring.

    A ring triangle ``(a, b, c)`` in ring order yields ``(e0, a, b, c)`` and
    ``(e1, a, c, b)``; one orientation of the ring orders every such cell
    positively. Both ring orientations are tried. Minimizing new-edge band
    violations before quality avoids discarding a feasible triangulation just
    because a higher-quality one would create an inadmissibly long diagonal.
    """
    count = len(ring)
    if count < 3:
        return None
    best_result: tuple[tuple[tuple[int, int, int, int], ...], float] | None = None
    best_key: tuple[tuple[int, ...], tuple[int, ...], float] | None = None
    for oriented_ring in (ring, ring[::-1]):
        result = _ring_triangulation(points, metrics, topology, edge, oriented_ring, mode)
        if result is None:
            continue
        introduced = sorted(
            {
                _edge_key((cell[first], cell[second]))
                for cell in result[0]
                for first, second in _TETRA_EDGE_CORNERS
                if _edge_key((cell[first], cell[second])) not in topology.edge_cells
            }
        )
        violation = _band_violations(
            _edge_set_lengths(points, metrics, topology, introduced)
        )
        key = (*violation, -result[1])
        if best_key is None or key < best_key:
            best_result, best_key = result, key
    return best_result


def _ring_triangulation(
    points: np.ndarray,
    metrics: np.ndarray,
    topology: _TetTopology,
    edge: tuple[int, int],
    ring: tuple[int, ...],
    mode: PredicateMode,
    /,
) -> tuple[tuple[tuple[int, int, int, int], ...], float] | None:
    count = len(ring)
    best = np.full((count, count), -np.inf, dtype=np.float64)
    split = np.full((count, count), -1, dtype=np.int32)
    violations: dict[tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]] = {}
    pairs = list(combinations(range(count), 2))
    lengths = _edge_set_lengths(
        points, metrics, topology, [(ring[first], ring[last]) for first, last in pairs]
    )
    diagonals = {
        pair: (
            ((), ())
            if _edge_key((ring[pair[0]], ring[pair[1]])) in topology.edge_cells
            else _band_violations(np.asarray((length,)))
        )
        for pair, length in zip(pairs, lengths, strict=True)
    }
    for index in range(count - 1):
        best[index, index + 1] = np.inf
        violations[index, index + 1] = ((), ())
    # Evaluate both cells of every ring triangle ``(first, pivot, last)`` in
    # one batch; every kernel is row-independent, so each triangle's values
    # equal its separate evaluation.
    triangles = [
        (first, pivot, last)
        for first in range(count)
        for last in range(first + 2, count)
        for pivot in range(first + 1, last)
    ]
    ring_rows = np.asarray(
        [topology.vertex_row[vertex] for vertex in ring], dtype=np.int64
    )
    first_rows, pivot_rows, last_rows = (
        ring_rows[np.asarray([triangle[axis] for triangle in triangles], dtype=np.int64)]
        for axis in range(3)
    )
    apex = np.full(first_rows.shape, topology.vertex_row[edge[0]], dtype=np.int64)
    opposite = np.full(first_rows.shape, topology.vertex_row[edge[1]], dtype=np.int64)
    created = np.stack(
        (
            np.stack((apex, first_rows, pivot_rows, last_rows), axis=1),
            np.stack((opposite, first_rows, last_rows, pivot_rows), axis=1),
        ),
        axis=1,
    ).reshape((-1, 4))
    determinants, relative = metric_relative_determinants(points[created])
    floor_cleared = (determinants > 0.0) & (relative > _DETERMINANT_FLOOR)
    cell_quality = metric_tetra_quality(points, metrics, created)
    admitted = (
        (floor_cleared & np.isfinite(cell_quality) & (cell_quality > 0.0))
        .reshape((-1, 2))
        .all(axis=1)
    )
    pair_quality = cell_quality.reshape((-1, 2)).min(axis=1)
    triangle_quality = {
        triangle: float(quality)
        for triangle, quality, valid in zip(
            triangles, pair_quality.tolist(), admitted.tolist(), strict=True
        )
        if valid
    }
    for span in range(2, count):
        for first in range(count - span):
            last = first + span
            best_score = -np.inf
            best_split = -1
            best_key: tuple[tuple[int, ...], tuple[int, ...], float] | None = None
            for pivot in range(first + 1, last):
                left_score = best[first, pivot]
                right_score = best[pivot, last]
                if left_score == -np.inf or right_score == -np.inf:
                    continue
                # Created cells must clear the determinant floor, which also
                # certifies their binary64 orientation (``_floor_oriented``).
                quality = triangle_quality.get((first, pivot, last))
                if quality is None:
                    continue
                score = min(left_score, right_score, quality)
                left = violations[first, pivot]
                right = violations[pivot, last]
                closing = diagonals[first, last]
                # A subproblem owns its closing diagonal, so every introduced
                # edge contributes once, including an open boundary ring's
                # new closing edge.
                combined = tuple(
                    tuple(sorted(left[axis] + right[axis] + closing[axis], reverse=True))
                    for axis in range(2)
                )
                key = (combined[0], combined[1], -score)
                if best_key is None or key < best_key:
                    best_score, best_split, best_key = score, pivot, key
            best[first, last] = best_score
            split[first, last] = best_split
            if best_key is not None:
                violations[first, last] = (best_key[0], best_key[1])
    if split[0, count - 1] < 0:
        return None
    cells: list[tuple[int, int, int, int]] = []
    intervals = [(0, count - 1)]
    while intervals:
        first, last = intervals.pop()
        if last - first < 2:
            continue
        pivot = int(split[first, last])
        if pivot < 0:
            return None
        a, b, c = ring[first], ring[pivot], ring[last]
        cells.extend(((edge[0], a, b, c), (edge[1], a, c, b)))
        intervals.append((pivot, last))
        intervals.append((first, pivot))
    return tuple(cells), float(best[0, count - 1])


def _fraction_stencil(
    sources: tuple[int, ...], weights: tuple[Fraction, ...], /
) -> dict[int, Fraction]:
    merged: dict[int, Fraction] = {}
    for source, weight in zip(sources, weights, strict=True):
        if weight:
            identifier = int(source)
            merged[identifier] = (
                merged[identifier] + weight if identifier in merged else weight
            )
    return {source: weight for source, weight in sorted(merged.items()) if weight}


def _binary64_ancestry(stencil: dict[int, Fraction], /) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(stencil)
    identifiers = np.asarray(ordered, dtype=np.uint64)
    exact = [stencil[identifier] for identifier in ordered]
    converted = [float(weight) for weight in exact]
    weights = np.asarray(converted, dtype=np.float64)
    # Integer forms of ``Fraction(float(w)) == w`` and ``sum(w) == 1``.
    for rounded, weight in zip(converted, exact, strict=True):
        if rounded.as_integer_ratio() != (weight.numerator, weight.denominator):
            raise FloatingPointError(
                "Exact source ancestry is not representable in binary64."
            )
    denominator = lcm(*(weight.denominator for weight in exact))
    if (
        sum(weight.numerator * (denominator // weight.denominator) for weight in exact)
        != denominator
    ):
        raise FloatingPointError(
            "Represented source ancestry weights do not sum exactly to one."
        )
    return identifiers, weights


def _ancestry_csr(
    stencils: list[dict[int, Fraction]], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.zeros((len(stencils) + 1,), dtype=np.int64)
    identifiers: list[int] = []
    weights: list[float] = []
    for row, stencil in enumerate(stencils):
        source_ids, source_weights = _binary64_ancestry(stencil)
        identifiers.extend(int(value) for value in source_ids.tolist())
        weights.extend(float(value) for value in source_weights.tolist())
        offsets[row + 1] = len(identifiers)
    return (
        offsets,
        np.asarray(identifiers, dtype=np.uint64),
        np.asarray(weights, dtype=np.float64),
    )


def _source_stencil_point(
    source_points: np.ndarray,
    source_vertex_rows: dict[int, int],
    stencil: dict[int, Fraction],
    /,
) -> np.ndarray:
    exact = [
        sum(
            (
                stencil[identifier]
                * Fraction(float(source_points[source_vertex_rows[identifier], axis]))
                for identifier in stencil
            ),
            Fraction(0),
        )
        for axis in range(3)
    ]
    point = np.asarray([float(value) for value in exact], dtype=np.float64)
    if not np.all(np.isfinite(point)):
        raise FloatingPointError("Exact source stencil realization is not finite.")
    return point


def _current_edge_stencil(
    state: _NativeState, proposal: _SplitProposal, /
) -> dict[int, Fraction]:
    parameter = Fraction(float(proposal.parameter))
    stencil = _fraction_stencil(proposal.edge, (Fraction(1) - parameter, parameter))
    _binary64_ancestry(stencil)
    vertex_ids = np.asarray(state.snapshot.vertex_ids, dtype=np.int64)
    rows = {int(identifier): row for row, identifier in enumerate(vertex_ids.tolist())}
    point = _source_stencil_point(
        np.asarray(state.snapshot.points, dtype=np.float64), rows, stencil
    )
    if not np.array_equal(point, np.asarray(proposal.point, dtype=np.float64)):
        raise FloatingPointError(
            "Current lifted split ancestry does not reproduce its exact carrier."
        )
    return stencil


def _native_ancestor_windings(
    state: _NativeState, ancestor_ids: ArrayLike, /
) -> np.ndarray:
    identifiers = np.asarray(ancestor_ids, dtype=np.int64)
    if state.target_mesh.periodic_topology is None:
        return np.zeros((identifiers.size, 3), dtype=np.int64)
    try:
        return np.asarray(
            [
                state.periodic_vertex_windings[int(identifier)]
                for identifier in identifiers
            ],
            dtype=np.int64,
        ).reshape((-1, 3))
    except KeyError as error:
        raise FloatingPointError(
            "Periodic native ancestry references a vertex outside the current quotient map."
        ) from error


def _source_metric_from_stencil(
    source_metric: np.ndarray,
    source_vertex_rows: dict[int, int],
    stencil: dict[int, Fraction],
    /,
    *,
    source_metric_log: np.ndarray | None = None,
) -> np.ndarray:
    identifiers, weights = _binary64_ancestry(stencil)
    rows = np.asarray(
        [source_vertex_rows[int(identifier)] for identifier in identifiers],
        dtype=np.int64,
    )
    # One row over the stencil's own support in ascending source identifier
    # order; a support of equal tensors returns its tensor unchanged.
    values, valid = metric_interpolations(
        np.asarray((0, rows.size), dtype=np.int64),
        rows,
        weights,
        source_metric,
        source_metric_log,
    )
    if not valid[0]:
        raise FloatingPointError(
            "Source metric interpolation is not a finite SPD tensor."
        )
    return values[0]


def _source_tetra_inverse(
    points: np.ndarray,
    cells: np.ndarray,
    global_vertex_ids: np.ndarray,
    cell_ids: np.ndarray,
    /,
    *,
    exact_points: tuple[tuple[Fraction, Fraction, Fraction], ...] | None = None,
) -> _SourceRecords:
    records: list[_SourceTetRecord] = []
    for row, vertices in enumerate(cells):
        coordinates = [
            exact_points[int(vertex)]
            if exact_points is not None
            else tuple(Fraction(float(value)) for value in points[int(vertex)])
            for vertex in vertices
        ]
        origin = coordinates[0]
        matrix = [
            [coordinates[column + 1][axis] - origin[axis] for column in range(3)]
            for axis in range(3)
        ]
        identity = [
            [Fraction(int(row == column)) for column in range(3)] for row in range(3)
        ]
        inverse = tuple(tuple(row) for row in solve_exact(matrix, identity))
        affine = tuple(
            (
                *inverse_row,
                -sum(
                    (inverse_row[axis] * origin[axis] for axis in range(3)), Fraction(0)
                ),
            )
            for inverse_row in inverse
        )
        denominator = lcm(
            *(value.denominator for affine_row in affine for value in affine_row)
        )
        integer_affine = tuple(
            tuple(
                value.numerator * (denominator // value.denominator)
                for value in affine_row
            )
            for affine_row in affine
        )
        records.append(
            (
                int(cell_ids[row]),
                tuple(int(global_vertex_ids[int(vertex)]) for vertex in vertices),
                inverse,
                origin,
                tuple(
                    float(min(corner[axis] for corner in coordinates))
                    for axis in range(3)
                ),
                tuple(
                    float(max(corner[axis] for corner in coordinates))
                    for axis in range(3)
                ),
                integer_affine,
                denominator,
            )
        )
    return _source_records(tuple(sorted(records, key=lambda row: row[0])))


def _source_records(records: tuple[_SourceTetRecord, ...], /) -> _SourceRecords:
    """Bind records already in ascending cell ID to their box arrays."""
    lower = np.asarray([record[4] for record in records], dtype=np.float64)
    upper = np.asarray([record[5] for record in records], dtype=np.float64)
    lower = lower.reshape((len(records), 3))
    upper = upper.reshape((len(records), 3))
    lower.setflags(write=False)
    upper.setflags(write=False)
    return _SourceRecords(records, lower, upper)


def _integer_point(
    point: tuple[Fraction, Fraction, Fraction], /
) -> tuple[tuple[int, int, int], int]:
    """An exact point as integer coordinates over one common denominator."""
    denominator = lcm(*(value.denominator for value in point))
    return (
        (
            point[0].numerator * (denominator // point[0].denominator),
            point[1].numerator * (denominator // point[1].denominator),
            point[2].numerator * (denominator // point[2].denominator),
        ),
        denominator,
    )


def _record_weights(
    scaled_point: tuple[int, int, int],
    point_denominator: int,
    record: _SourceTetRecord,
    /,
) -> tuple[int, tuple[int, ...], tuple[Fraction, ...]] | None:
    """Exact barycentric weights of an integer point in one source cell."""
    common = record[7] * point_denominator
    remaining = common
    tail = []
    for row in record[6]:
        weight = (
            row[0] * scaled_point[0]
            + row[1] * scaled_point[1]
            + row[2] * scaled_point[2]
            + row[3] * point_denominator
        )
        if weight < 0 or weight > remaining:
            return None
        tail.append(weight)
        remaining -= weight
    return (
        record[0],
        record[1],
        tuple(Fraction(value, common) for value in (remaining, *tail)),
    )


def _source_cell_weights(
    point: tuple[Fraction, Fraction, Fraction],
    record: _SourceTetRecord,
    /,
) -> tuple[int, tuple[int, ...], tuple[Fraction, ...]] | None:
    """Exact barycentric weights of a point in one source cell, or ``None``."""
    scaled_point, point_denominator = _integer_point(point)
    return _record_weights(scaled_point, point_denominator, record)


def _locate_source_weights(
    point: tuple[Fraction, Fraction, Fraction],
    source_records: _SourceRecords,
    /,
) -> tuple[int, tuple[int, ...], tuple[Fraction, ...]] | None:
    """First source cell (by ID) containing an exact point, with its exact
    barycentric weights in the cell's vertex order."""
    try:
        rounded = (float(point[0]), float(point[1]), float(point[2]))
    except OverflowError:
        # Source vertices are finite binary64, so an overflowed coordinate
        # lies outside every source box.
        return None
    # Correct rounding is monotone and the bounds are binary64 source
    # vertices: rounding a point within a closed box cannot leave it. The
    # float comparison therefore never drops a containing cell; the exact
    # integer test below decides among the boxes that pass.
    inside = np.flatnonzero(
        np.all(
            (source_records.lower <= rounded) & (rounded <= source_records.upper),
            axis=1,
        )
    )
    if inside.size == 0:
        return None
    scaled_point, point_denominator = _integer_point(point)
    records = source_records.records
    for row in inside.tolist():
        located = _record_weights(scaled_point, point_denominator, records[row])
        if located is not None:
            return located
    return None


def _locate_source_stencil(
    point: np.ndarray,
    source_records: _SourceRecords,
    /,
) -> tuple[int, dict[int, Fraction]] | None:
    query = tuple(Fraction(float(value)) for value in point)
    located = _locate_source_weights((query[0], query[1], query[2]), source_records)
    if located is None:
        return None
    cell_id, vertices, weights = located
    return cell_id, _fraction_stencil(vertices, weights)


type _EntityFrame = tuple[
    tuple[Fraction, ...],
    tuple[tuple[Fraction, ...], ...] | None,
]


def _entity_frame(corners: tuple[tuple[Fraction, ...], ...], /) -> _EntityFrame:
    origin = corners[0]
    axes = tuple(
        tuple(corner[axis] - origin[axis] for axis in range(3)) for corner in corners[1:]
    )
    gram = tuple(
        tuple(
            sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))
            for second in axes
        )
        for first in axes
    )
    if len(axes) == 1:
        if gram[0][0] == 0:
            return origin, None
        gradients = (tuple(value / gram[0][0] for value in axes[0]),)
    else:
        determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
        if determinant == 0:
            return origin, None
        gradients = (
            tuple(
                (axes[0][k] * gram[1][1] - axes[1][k] * gram[0][1]) / determinant
                for k in range(3)
            ),
            tuple(
                (axes[1][k] * gram[0][0] - axes[0][k] * gram[1][0]) / determinant
                for k in range(3)
            ),
        )
    return origin, gradients


def _fresh_ids(first: int, count: int, name: str, /) -> np.ndarray:
    if count < 0 or first < 0 or first + count > np.iinfo(np.int64).max:
        raise OverflowError(f"{name} exceeds the stable int64 identifier range.")
    return np.arange(first, first + count, dtype=np.int64)


def _relation_records(
    dimension: int,
    source_keys: list[tuple[int, ...]],
    target_keys: list[tuple[int, ...]],
    kinds: list[EntityLineageKind],
    /,
) -> EntityRelations:
    width = 1 if dimension in (0, 3) else dimension + 1
    if not source_keys:
        return EntityRelations(
            dimension,
            np.zeros((0, width), dtype=np.int64),
            np.zeros((0, width), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )
    records = sorted(
        set(zip(source_keys, target_keys, kinds, strict=True)),
        key=lambda row: (row[0], row[1], int(row[2])),
    )
    return EntityRelations(
        dimension,
        np.asarray([row[0] for row in records], dtype=np.int64).reshape((-1, width)),
        np.asarray([row[1] for row in records], dtype=np.int64).reshape((-1, width)),
        np.asarray([int(row[2]) for row in records], dtype=np.int32),
    )


type _FacetSides = tuple[int, tuple[int, ...], int]
type _SourceFacet = tuple[int, tuple[int, int, int], np.ndarray, _FacetSides]


def _constrained_source_facets(
    mesh: CellMesh,
    cells: np.ndarray,
    cell_ids: np.ndarray,
    regions: np.ndarray,
    face_ids: np.ndarray,
    face_keys: np.ndarray,
    facet_codes: np.ndarray,
    /,
) -> list[_SourceFacet]:
    """Reuse the source's facet-to-cell incidence; orient only constrained rows."""
    relation = mesh.topology.incidences[2].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    facets = np.asarray(relation.source_indices)[valid]
    top_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    incident = np.searchsorted(
        cell_ids, top_ids[np.asarray(relation.target_indices)[valid]]
    )
    counts = np.bincount(facets, minlength=face_ids.size)
    lowest = np.full(face_ids.size, np.iinfo(np.int32).max, dtype=np.int32)
    highest = np.full(face_ids.size, -1, dtype=np.int32)
    incident_regions = regions[incident]
    np.minimum.at(lowest, facets, incident_regions)
    np.maximum.at(highest, facets, incident_regions)
    select = ((counts == 1) | (lowest != highest) | (facet_codes > 0)) & (counts > 0)
    eligible = select[facets]
    facets, incident = facets[eligible], incident[eligible]
    order = np.lexsort((incident, facets))
    facet_rows, starts, sizes = np.unique(
        facets[order], return_index=True, return_counts=True
    )
    incident = incident[order]
    owner_cells = cells[incident[starts]]
    owner_points = np.asarray(mesh.vertex_global_ids)[owner_cells]
    on_face = np.any(owner_points[:, :, None] == face_keys[facet_rows, None], axis=-1)
    if np.any(np.count_nonzero(on_face, axis=1) != 3):
        raise ValueError("A constrained tetra facet has no source identity.")
    oriented = np.take_along_axis(
        owner_cells, _TETRA_FACES[np.argmin(on_face, axis=1)], axis=1
    ).astype(np.int32)
    ordered = np.argsort(face_ids[facet_rows])
    result: list[_SourceFacet] = []
    for index in ordered.tolist():
        row, start, count = facet_rows[index], starts[index], sizes[index]
        if lowest[row] == highest[row]:
            active_regions = (int(lowest[row]),)
        elif count == 2:
            active_regions = (int(lowest[row]), int(highest[row]))
        else:
            active_regions = tuple(
                np.unique(regions[incident[start : start + count]]).tolist()
            )
        result.append(
            (
                int(face_ids[row]),
                _face_key(face_keys[row]),
                oriented[index],
                (int(count), active_regions, int(facet_codes[row])),
            )
        )
    return result


def _coplanar_face_groups(
    points: np.ndarray,
    faces: np.ndarray,
    sides: list[_FacetSides],
    segments: set[tuple[int, int]],
    /,
) -> np.ndarray:
    """Planar source facet label (a representative row) of each source triangle.

    Two triangles sharing an edge join when the edge is no source segment and
    is used exactly by them in opposite directions (a consistently oriented
    manifold), their sides agree (incident cell count, regions, facet class),
    and the fourth vertex is exactly coplanar (exact rational orientation).
    """
    parent = list(range(faces.shape[0]))

    def find(row: int) -> int:
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
        return row

    uses: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    rows = np.asarray(faces, dtype=np.int64).tolist()
    for row, face in enumerate(rows):
        for corner in range(3):
            first, second = int(face[corner]), int(face[(corner + 1) % 3])
            uses[_edge_key((first, second))].append((row, (first, second)))

    candidates: list[tuple[int, int]] = []
    tests: list[list[int]] = []
    for key, incident in uses.items():
        if len(incident) != 2 or key in segments:
            continue
        (first_row, first_use), (second_row, second_use) = incident
        if first_use != second_use[::-1] or sides[first_row] != sides[second_row]:
            continue
        apex = next(vertex for vertex in rows[second_row] if vertex not in key)
        tests.append([*rows[first_row], apex])
        candidates.append((first_row, second_row))
    signs = _tetra_orientation(
        points, np.asarray(tests, dtype=np.int64).reshape((-1, 4)), PredicateMode.EXACT
    )
    if np.any(signs == 2):
        raise ValueError("Exact source facet coplanarity is unresolved.")
    for (first_row, second_row), sign in zip(candidates, signs, strict=True):
        if sign == 0:
            parent[find(first_row)] = find(second_row)
    return np.asarray([find(row) for row in range(faces.shape[0])], dtype=np.int64)


def _declared_face_planes(points: np.ndarray, faces: np.ndarray, /) -> np.ndarray:
    """``(F, 3, 3)`` exact planes (origin, two axes) of source triangles.

    A triangle ``abc`` declares ``(a, b - a, c - a)`` when both axes are exact
    binary64 differences; otherwise its row is NaN (undeclared). The plane is
    then exactly the triangle's plane and a point ``a + s (b - a) + t (c - a)``
    has source weights ``(1 - s - t, s, t)``.
    """
    planes = np.full((faces.shape[0], 3, 3), np.nan, dtype=np.float64)
    for row, face in enumerate(np.asarray(faces, dtype=np.int64).tolist()):
        origin, first, second = (points[int(vertex)] for vertex in face)
        axes = (first - origin, second - origin)
        if all(
            Fraction(float(axis[k]))
            == Fraction(float(end[k])) - Fraction(float(origin[k]))
            for axis, end in zip(axes, (first, second), strict=True)
            for k in range(3)
        ):
            planes[row] = (origin, axes[0], axes[1])
    return planes


def _cartesian_source_seams(
    points: np.ndarray,
    vertex_ids: np.ndarray,
    vertex_row_by_id: Mapping[int, int],
    constrained_faces: list[_SourceFacet],
    face_groups: np.ndarray,
    declared_planes: np.ndarray,
    edge_id_by_key: Mapping[tuple[int, int], int],
    live_segment_keys: list[tuple[int, int]],
    fixed: np.ndarray,
    /,
) -> tuple[list[tuple[int, int]], dict[int, tuple[tuple[int, int], int]]]:
    """Find immutable ridge edges and canonical source endpoint witnesses.

    A ridge uses exactly two oppositely oriented exterior triangles from
    different Cartesian planes. Scan the constrained source incidence once;
    neither live facets nor the entire tetrahedral edge bank define the chart.
    """
    constant_axes = np.all(declared_planes[:, 1:] == 0.0, axis=1)
    if np.count_nonzero(np.any(constant_axes, axis=0)) < 2:
        return [], {}
    group_axes = np.where(
        np.count_nonzero(constant_axes, axis=1) == 1,
        np.argmax(constant_axes, axis=1),
        -1,
    )
    authored = set(live_segment_keys)
    authored_vertices = {vertex for key in authored for vertex in key}
    edge_uses: dict[tuple[int, int], list[tuple[int, bool]]] = defaultdict(list)
    vertex_groups: dict[int, set[int]] = defaultdict(set)
    for face_row, (_, _, corners, _) in enumerate(constrained_faces):
        a, b, c = (int(vertex_ids[corner]) for corner in corners)
        group = int(face_groups[face_row])
        for first, second in ((a, b), (b, c), (c, a)):
            vertex_groups[first].add(group)
            forward = first < second
            key = (first, second) if forward else (second, first)
            edge_uses[key].append((face_row, forward))

    seams: list[tuple[int, int]] = []
    incident_seams: dict[int, list[tuple[tuple[int, int], int]]] = defaultdict(list)
    for key, uses in edge_uses.items():
        if len(uses) != 2 or key in authored:
            continue
        (first, forward), (second, reverse) = uses
        if (
            forward == reverse
            or constrained_faces[first][3][0] != 1
            or constrained_faces[second][3][0] != 1
        ):
            continue
        first_axis = int(group_axes[face_groups[first]])
        second_axis = int(group_axes[face_groups[second]])
        if first_axis < 0 or second_axis < 0 or first_axis == second_axis:
            continue
        # Two different fixed axes leave one direction along the intersection.
        axis = 3 - first_axis - second_axis
        seams.append(key)
        for vertex in key:
            incident_seams[vertex].append((key, axis))

    witnesses: dict[int, tuple[tuple[int, int], int]] = {}
    for vertex, incident in incident_seams.items():
        row = vertex_row_by_id[vertex]
        if (
            len(vertex_groups[vertex]) != 2
            or len(incident) != 2
            or vertex in authored_vertices
            or fixed[row]
        ):
            continue
        (first_key, axis), (second_key, other_axis) = incident
        if axis != other_axis:
            continue
        point = points[row]
        first_end = first_key[0] if vertex == first_key[1] else first_key[1]
        second_end = second_key[0] if vertex == second_key[1] else second_key[1]
        first = points[vertex_row_by_id[first_end], axis]
        second = points[vertex_row_by_id[second_end], axis]
        if not (first < point[axis] < second or second < point[axis] < first):
            continue
        # Expansion realization canonicalizes zero; only install a witness
        # when the unchanged native carrier also has that sign bit.
        if np.any((point == 0.0) & np.signbit(point)):
            continue
        key = (
            first_key
            if edge_id_by_key[first_key] < edge_id_by_key[second_key]
            else second_key
        )
        witnesses[row] = (key, 0 if vertex == key[0] else 1)
    return seams, witnesses


def _cell_volume(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    determinants, _ = metric_relative_determinants(points[cells])
    return np.abs(determinants) / 6.0


def _region_volumes(
    points: np.ndarray, cells: np.ndarray, regions: np.ndarray, /
) -> dict[int, float]:
    measures = _cell_volume(points, cells)
    totals: dict[int, float] = defaultdict(float)
    for region, measure in zip(regions.tolist(), measures.tolist(), strict=True):
        totals[int(region)] += float(measure)
    return dict(sorted(totals.items()))


def _relative_region_error(before: dict[int, float], after: dict[int, float], /) -> float:
    if before.keys() != after.keys():
        return float("inf")
    return max(
        (
            abs(after[region] - value) / max(abs(value), np.finfo(np.float64).tiny)
            for region, value in before.items()
        ),
        default=0.0,
    )


def _native_periodic_state(
    mesh: CellMesh, cell_regions: np.ndarray | None, /
) -> TetMeshPeriodicState | None:
    if mesh.periodic_topology is None:
        return None
    return tet_mesh_periodic_state(mesh, cell_regions)


def _refresh_native_periodic_windings(state: _NativeState, /) -> None:
    mesh = state.target_mesh
    if mesh.periodic_topology is None:
        state.periodic_vertex_windings = {}
        return
    periodic_state = state.periodic_state
    if (
        periodic_state is None
        or periodic_state.identity_map.source_mesh_id != mesh.mesh_id
    ):
        raise ValueError("Periodic native windings require a bound periodic target.")
    native_by_target = {
        int(target): int(native) for native, target in state.vertex_target_ids.items()
    }
    windings: dict[int, np.ndarray] = {}
    for orbit in periodic_state.orbits:
        for member in orbit.members:
            native_id = native_by_target.get(int(member.vertex_id))
            if native_id is None:
                raise ValueError("Periodic native winding names an absent native vertex.")
            windings[native_id] = np.asarray(member.winding, dtype=np.int64).copy()
    state.periodic_vertex_windings = windings


def _validate_periodic_metric_values(
    mesh: CellMesh,
    metric_by_vertex_id: dict[int, np.ndarray],
    periodic_state: TetMeshPeriodicState | None = None,
    /,
) -> None:
    """``periodic_state``: the mesh's native quotient state when already built."""
    if mesh.periodic_topology is None:
        return
    if periodic_state is None:
        periodic_state = _native_periodic_state(mesh, None)
    if periodic_state is None:
        raise ValueError("Periodic metric validation requires a native quotient state.")
    linear: list[np.ndarray] = []
    metrics: list[np.ndarray] = []
    references: list[np.ndarray] = []
    missing: str | None = None
    for orbit in periodic_state.orbits:
        representative = metric_by_vertex_id.get(int(orbit.representative_vertex_id))
        if representative is None:
            missing = "Periodic metric evidence is missing an orbit representative."
            break
        for member in orbit.members:
            metric = metric_by_vertex_id.get(int(member.vertex_id))
            if metric is None:
                missing = "Periodic metric evidence is missing a lifted vertex."
                break
            linear.append(
                np.asarray(member.representative_to_member, dtype=np.float64)[:, :3]
            )
            metrics.append(metric)
            references.append(representative)
        if missing is not None:
            break
    # A metric M at the image of the isometry x -> A x + t agrees with the
    # representative's metric R when M = A R A^T: lengths of mapped segments
    # are equal, and the transpose of an isometry is its inverse. The defect
    # is measured at the member, M - A R A^T, the quantity the allowance is
    # stated for: its norm does not change with the frame of the
    # representative. Every member before a missing one is judged first.
    isometries = np.asarray(linear, dtype=np.float64).reshape((-1, 3, 3))
    errors, scales = metric_congruence_defects(
        np.ascontiguousarray(np.swapaxes(isometries, 1, 2)),
        np.asarray(references, dtype=np.float64).reshape((-1, 3, 3)),
        np.asarray(metrics, dtype=np.float64).reshape((-1, 3, 3)),
    )
    tolerance = (256.0 * np.finfo(np.float64).eps) * np.maximum(
        scales, np.finfo(np.float64).tiny
    )
    if not np.all(errors <= tolerance):
        raise ValueError(
            "SPD metric values must transform equivariantly across periodic vertex orbits."
        )
    if missing is not None:
        raise ValueError(missing)


def _implicit_metric_source(source: CellMeshingResult, /) -> _ImplicitMetricSource | None:
    """Lower the actual accepted implicit authority when its source tube applies."""
    from ..geometry._source_boundary_records import SourceBoundaryChartWitness
    from .providers._native_implicit import _implicit_field_programs
    from .providers._native_sources import NativeImplicitSource

    report = source.certification
    if report is None or source.mesh.periodic_topology is not None:
        return None
    inputs = report.inputs
    authority = inputs.feature_source
    witness = inputs.fidelity_witness
    if (
        not isinstance(authority, NativeImplicitSource)
        or authority.discovery is None
        or authority.authoring_profile is None
        or not isinstance(witness, SourceBoundaryChartWitness)
        or not witness.implicit_cell_keys.size
        or inputs.source is None
        or inputs.fidelity_tolerance is None
        or (inputs.source.source_id, inputs.source.source_revision)
        != (authority.source_id, authority.source_revision)
    ):
        return None
    report.require_current(source.mesh, source.geometry)
    report.require_passed()
    profile = authority.authoring_profile
    profile.require_bound(authority.geometry, source.coordinate_contract)
    tolerance = float(inputs.fidelity_tolerance)
    for query, scoped_witness, scoped_tolerance in inputs.scoped_fidelity:
        if query is inputs.source and scoped_witness is witness:
            tolerance = min(tolerance, float(scoped_tolerance))
    if tolerance <= 0.0 or tolerance >= profile.tube_radius:
        return None
    value, gradient = _implicit_field_programs(authority.geometry)
    return _ImplicitMetricSource(
        value,
        gradient,
        tolerance,
        authority.source_id,
        authority.source_revision,
        authority.binding_id,
        source.coordinate_contract.reference_frame,
    )


def _canonical_lagrange_layout(
    mesh: CellMesh, geometry: CellGeometrySpec, points: np.ndarray, /
) -> bool:
    """Recognize the exact binary64 nodal map without expanding its cells."""
    from ..discretization.fem import lagrange_element

    if (
        mesh.periodic_topology is not None
        or geometry.source_map is not None
        or geometry.source_mesh_id is not None
        or geometry.exact_coordinate_bank is not None
        or geometry.restriction_source is not None
        or geometry.block_names != tuple(block.name for block in mesh.blocks)
        or not np.array_equal(np.asarray(geometry.coordinates), points)
    ):
        return False
    linear = lagrange_element("tetrahedron", 1)
    covered = np.zeros(points.shape[0], dtype=np.bool_)
    for block, element, route in zip(
        mesh.blocks, geometry.elements, geometry.geometry_dofs, strict=True
    ):
        if block.cell_kind != "tetrahedron" or element is not linear:
            return False
        vertices = np.asarray(block.vertices)
        if not np.array_equal(np.asarray(route), vertices):
            return False
        covered[vertices] = True
    if not np.all(covered):
        raise ValueError("Exact source geometry does not cover every metric vertex.")
    return True


def _canonical_lagrange_vertices(
    mesh: CellMesh, geometry: CellGeometrySpec, points: np.ndarray, /
) -> tuple[tuple[Fraction, Fraction, Fraction], ...] | None:
    if not _canonical_lagrange_layout(mesh, geometry, points):
        return None
    return tuple(
        (Fraction(float(x)), Fraction(float(y)), Fraction(float(z))) for x, y, z in points
    )


def _admit_metric_bank(values: np.ndarray, vertex_count: int, /) -> np.ndarray:
    metric = _tensor_array(values, vertex_count)
    if metric.shape[1:] != (3, 3):
        raise ValueError(
            "Tetra adaptation requires a true 3x3 SPD metric at every vertex."
        )
    symmetric, positive_definite = metric_tensor_properties(metric)
    if not np.all(symmetric) or not np.all(positive_definite):
        raise ValueError(
            "Tetra adaptation requires finite symmetric positive-definite metrics."
        )
    if np.any(np.abs(metric) > _METRIC_ENTRY_BOUND):
        raise ValueError(
            "Tetra adaptation requires metric entries of magnitude at most 2**778: "
            "larger ones cannot be measured over the exact coordinate domain."
        )
    return metric


def _plain_tetra_source_is_current(source: CellMeshingResult, /) -> bool:
    mesh = source.mesh
    audit = source.audit
    if (
        source.certification is not None
        or not audit.passed
        or audit.recorded
        or audit.unresolved
        or audit.manifold_scope is not None
        or not audit.validity.all_certified
        or audit.mesh_id != mesh.mesh_id
        or audit.geometry_id != cell_geometry_id(source.geometry)
        or not source.coordinate_contract.reference_frame
    ):
        return False
    checks = dict(audit.check_counts)
    for name in (
        "coincident_vertices",
        "collapsed_cells",
        "duplicate_cells",
        "nonmanifold_facets",
        "nonmanifold_edges",
        "nonmanifold_vertices",
        "inconsistent_orientation",
        "self_intersection",
        "invalid_geometry",
    ):
        if name not in audit.evaluated_checks or checks.get(name) != 0:
            return False
    audit.validity.require_bound(source.geometry, mesh=mesh)
    return True


def _noop_metric_length_extrema(
    points: np.ndarray, metric: np.ndarray, edges: np.ndarray, workers: int, /
) -> tuple[float, float]:
    """Measure the original ordered edges without retaining the length bank."""
    count = edges.shape[0]
    block = 65_536
    points = np.ascontiguousarray(points)
    metric = np.ascontiguousarray(metric)

    def extrema(rows: np.ndarray, /) -> tuple[float, float]:
        values = metric_lengths(points, metric, rows)
        return float(np.min(values)), float(np.max(values))

    if count <= block:
        return extrema(edges)
    bounds = np.empty(((count + block - 1) // block, 2), dtype=np.float64)

    def measure(begin: int, end: int, chunk: int, /) -> None:
        bounds[chunk] = extrema(edges[begin:end])

    status = parallel_for(count, measure, chunk_size=block, parallelism=workers)
    _raise_for_call(status, "tetrahedral metric length extrema")
    return float(np.min(bounds[:, 0])), float(np.max(bounds[:, 1]))


def strict_plain_tetra_metric_noop(
    source: CellMeshingResult, metric_values: np.ndarray, /
) -> TetraMetricEvidence | None:
    """Measure strict-band completion of an unprotected, single-region P1 mesh."""
    mesh = source.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity) or len(mesh.blocks) != 1:
        return None
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices)
    if (
        not points.size
        or not cells.size
        or max(points.shape[0], cells.shape[0]) >= np.iinfo(np.int32).max // 4
        or not _canonical_lagrange_layout(mesh, source.geometry, points)
        or not _plain_tetra_source_is_current(source)
    ):
        return None
    workers = exact_kernel_parallelism()
    if not tetrahedra_positive_orientation(points, cells, parallelism=workers):
        return None
    metric = _admit_metric_bank(metric_values, points.shape[0])
    edges = np.asarray(connectivity.edges)
    vertex_ids = np.asarray(mesh.vertex_global_ids)
    if not np.all(vertex_ids[:-1] < vertex_ids[1:]) or not np.all(
        edges[:, 0] < edges[:, 1]
    ):
        reverse = vertex_ids[edges[:, 0]] > vertex_ids[edges[:, 1]]
        if np.any(reverse):
            edges = edges.copy()
            edges[reverse] = edges[reverse, ::-1]
    minimum_length, maximum_length = _noop_metric_length_extrema(
        points, metric, edges, workers
    )
    if not _LOWER <= minimum_length <= maximum_length <= _UPPER:
        return None
    minimum_quality = metric_tetra_quality_min(points, metric, cells, parallelism=workers)
    if not minimum_quality > 0.0:
        return None
    # Admission measured every edge; there is no protected or unmet subset.
    return TetraMetricEvidence(
        status=TetraMetricStatus.COMPLETE,
        passes=1,
        splits=0,
        collapses=0,
        flips23=0,
        flips32=0,
        ring_reconnections=0,
        relocations=0,
        rejected_topology=0,
        rejected_protected=0,
        rejected_uncertain=0,
        rejected_invalid=0,
        measured_edges=edges.shape[0],
        out_of_range_edges=0,
        minimum_metric_length=minimum_length,
        maximum_metric_length=maximum_length,
        minimum_metric_quality=minimum_quality,
        maximum_region_volume_error=0.0,
        unmet_edges=(),
    )


def _prepare_native_state(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    coordinate_contract: SpatialCoordinateContract,
    metric_values: np.ndarray,
    /,
    *,
    cell_classes: ArrayLike,
    facet_classes: ArrayLike,
    edge_classes: ArrayLike,
    protected_edges: ArrayLike,
    fixed_vertices: ArrayLike,
    implicit_source: _ImplicitMetricSource | None = None,
    derive_cartesian_seams: bool = False,
) -> _NativeState:
    """Prepare a source-bound P1 tetrahedral complex and persistent native owner."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or not mesh.blocks
        or any(
            block.cell_kind not in ("tetrahedron", "simplex:3") for block in mesh.blocks
        )
    ):
        raise ValueError("Native tetra metric adaptation requires 3D tetrahedron blocks.")
    from ._adaptation import _metric_exact_source_vertices, _piecewise_linear_geometry
    from ._local_metric import _boolean_array, _integer_array

    points = np.asarray(mesh.coordinates, dtype=np.float64)
    exact_source_points = _canonical_lagrange_vertices(mesh, geometry, points)
    has_authored_rounding = False
    if exact_source_points is None:
        if not _piecewise_linear_geometry(mesh, geometry):
            raise ValueError(
                "Native tetra metric adaptation requires an exact P1 source map."
            )
        exact_rows = _metric_exact_source_vertices(mesh, geometry)
        exact_source_points = tuple((row[0], row[1], row[2]) for row in exact_rows)
        if not np.array_equal(
            np.asarray([[float(value) for value in row] for row in exact_source_points]),
            points,
        ):
            raise ValueError(
                "Tetra source vertices are not canonical exact-source roundings."
            )
        has_authored_rounding = any(
            value != float(value) for point in exact_source_points for value in point
        )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if vertex_ids.shape != (points.shape[0],):
        raise ValueError("Source vertex identities do not align with coordinates.")
    metric = _admit_metric_bank(metric_values, points.shape[0])

    block_cells = []
    block_cell_ids = []
    block_numbers = []
    for block_index, block in enumerate(mesh.blocks):
        cells = np.asarray(block.vertices, dtype=np.int64)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] != 4:
            raise ValueError("Every native tetrahedron block must have arity four.")
        block_cells.append(cells)
        block_cell_ids.append(identifiers)
        block_numbers.append(np.full((identifiers.size,), block_index, dtype=np.int32))
    unsorted_cells = np.concatenate(block_cells, axis=0)
    unsorted_ids = np.concatenate(block_cell_ids, axis=0)
    unsorted_blocks = np.concatenate(block_numbers, axis=0)
    order = np.argsort(unsorted_ids, kind="stable")
    source_cells = unsorted_cells[order].copy()
    source_cell_ids = unsorted_ids[order]
    source_cell_blocks = unsorted_blocks[order]
    orientation = _tetra_orientation(points, source_cells, PredicateMode.EXACT)
    if np.any(orientation == 2):
        raise ValueError("Exact source tetrahedron orientation is unresolved.")
    negative = orientation < 0
    if mesh.periodic_topology is not None and np.any(negative):
        raise ValueError(
            "Periodic native tetra source cells must preserve lifted corner order."
        )
    source_cells[negative, 1], source_cells[negative, 2] = (
        source_cells[negative, 2].copy(),
        source_cells[negative, 1].copy(),
    )
    if np.any(_tetra_orientation(points, source_cells, PredicateMode.EXACT) != 1):
        raise ValueError("Source tetrahedra must have nonzero exact orientation.")

    cell_entity_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    cell_values = _integer_array(cell_classes, cell_entity_ids.size, "cell_classes")
    if np.any(cell_values < 0) or np.any(cell_values > np.iinfo(np.int32).max):
        raise ValueError("Tetra region identities must fit nonnegative int32 labels.")
    region_by_id = {
        int(identifier): int(value)
        for identifier, value in zip(
            cell_entity_ids.tolist(), cell_values.tolist(), strict=True
        )
    }
    regions = np.asarray(
        [region_by_id[int(identifier)] for identifier in source_cell_ids.tolist()],
        dtype=np.int32,
    )
    periodic_regions = np.asarray(
        [
            region_by_id[int(identifier)]
            for block in mesh.blocks
            for identifier in np.asarray(block.global_ids, dtype=np.int64).tolist()
        ],
        dtype=np.int32,
    )
    periodic_state = _native_periodic_state(mesh, periodic_regions)
    if periodic_state is not None:
        _validate_periodic_metric_values(
            mesh,
            {
                int(identifier): metric[row]
                for row, identifier in enumerate(vertex_ids.tolist())
            },
            periodic_state,
        )

    edge_keys = np.asarray(entity_keys(mesh, 1), dtype=np.int64)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    face_keys = np.asarray(entity_keys(mesh, 2), dtype=np.int64)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    if edge_keys.shape != (edge_ids.size, 2) or face_keys.shape != (face_ids.size, 3):
        raise ValueError("Tetra edge and facet identity tables are malformed.")
    edge_codes = _integer_array(edge_classes, edge_ids.size, "edge_classes")
    facet_codes = _integer_array(facet_classes, face_ids.size, "facet_classes")
    if (
        np.any(edge_codes < 0)
        or np.any(edge_codes > np.iinfo(np.int32).max)
        or np.any(facet_codes < 0)
        or np.any(facet_codes > np.iinfo(np.int32).max)
    ):
        raise ValueError("Tetra feature classes must fit nonnegative int32 labels.")
    protected = _boolean_array(protected_edges, edge_ids.size, "protected_edges")
    fixed = _boolean_array(fixed_vertices, vertex_ids.size, "fixed_vertices")
    edge_id_by_key = {
        _edge_key(row): int(identifier)
        for row, identifier in zip(edge_keys.tolist(), edge_ids.tolist(), strict=True)
    }
    edge_class_by_key = {
        _edge_key(row): int(value)
        for row, value in zip(edge_keys.tolist(), edge_codes.tolist(), strict=True)
    }
    protected_edge_ids = {
        _edge_key(edge_keys[row].tolist()) for row in np.flatnonzero(protected).tolist()
    }

    constrained_faces = _constrained_source_facets(
        mesh, source_cells, source_cell_ids, regions, face_ids, face_keys, facet_codes
    )
    source_faces = np.asarray(
        [row for _, _, row, _ in constrained_faces], dtype=np.int32
    ).reshape((-1, 3))
    source_face_row_ids = np.asarray(
        [identifier for identifier, *_ in constrained_faces], dtype=np.int64
    )

    vertex_row_by_id = {
        int(identifier): row for row, identifier in enumerate(vertex_ids.tolist())
    }
    live_segment_keys = [
        _edge_key(edge_keys[row].tolist())
        for row in range(edge_ids.size)
        if edge_codes[row] > 0 or protected[row]
    ]
    live_segment_keys.sort(key=lambda key: edge_id_by_key[key])
    live_segment_ids = np.asarray(
        [edge_id_by_key[key] for key in live_segment_keys], dtype=np.int64
    )
    live_segments = np.asarray(
        [
            [vertex_row_by_id[identifier] for identifier in key]
            for key in live_segment_keys
        ],
        dtype=np.int32,
    ).reshape((-1, 2))
    # Exactly coplanar source triangles joined across non-segment edges form
    # one planar source facet (one group): boundary vertices then move and
    # collapse across the facet's internal diagonals. On a periodic lift each
    # paired cell-face cap is such a facet, so seam vertices collapse within
    # their cap plane (or along the cell edge two caps share).
    row_groups = _coplanar_face_groups(
        points,
        source_faces,
        [sides for *_, sides in constrained_faces],
        {_edge_key(row) for row in live_segments.tolist()},
    )
    implicit_rows: list[int] = []
    if implicit_source is not None:
        if periodic_state is not None:
            raise ValueError(
                "Implicit tetra source admission does not support periodic state."
            )
        group_by_sides: dict[_FacetSides, int] = {}
        next_group = int(np.max(row_groups, initial=-1)) + 1
        for row, (_, _, _, sides) in enumerate(constrained_faces):
            if sides[0] != 1:
                continue
            group = group_by_sides.get(sides)
            if group is None:
                group = next_group
                next_group += 1
                group_by_sides[sides] = group
            row_groups[row] = group
            implicit_rows.append(row)
        if not implicit_rows:
            raise ValueError("Implicit tetra source requires actual exterior facets.")
    _, representative_rows, face_groups = np.unique(
        row_groups, return_index=True, return_inverse=True
    )
    # Native group order follows the first face's global identity.
    group_order = np.argsort(representative_rows)
    representative_rows = representative_rows[group_order]
    source_face_ids = source_face_row_ids[representative_rows]
    ranks = np.empty(group_order.shape, dtype=np.int32)
    ranks[group_order] = np.arange(group_order.size, dtype=np.int32)
    face_groups = ranks[face_groups]
    implicit_groups = frozenset(int(face_groups[row]) for row in implicit_rows)
    source_face_group_rows = {
        int(identifier): row for row, identifier in enumerate(source_face_ids.tolist())
    }
    grouped_triangles: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for row, (_, key, _, _) in enumerate(constrained_faces):
        grouped_triangles[int(face_groups[row])].append(key)
    source_group_triangles = {
        group: tuple(triangles) for group, triangles in grouped_triangles.items()
    }
    source_face_vertices = {
        group: frozenset(vertex for triangle in triangles for vertex in triangle)
        for group, triangles in source_group_triangles.items()
    }
    seam_keys: list[tuple[int, int]] = []
    endpoint_witnesses: dict[int, tuple[tuple[int, int], int]] = {}
    # Off a periodic lift each facet declares the exact plane of its first
    # triangle (origin a, axes b - a and c - a) when those axes are binary64,
    # so the native admits a carrier rounding a point of the plane
    # (declared-plane rounding-box admission).
    if periodic_state is None:
        representatives = _declared_face_planes(points, source_faces[representative_rows])
        face_planes = representatives[face_groups]
        if implicit_groups:
            face_planes[implicit_rows] = np.nan
        elif derive_cartesian_seams and not has_authored_rounding:
            seam_keys, endpoint_witnesses = _cartesian_source_seams(
                points,
                vertex_ids,
                vertex_row_by_id,
                constrained_faces,
                face_groups,
                representatives,
                edge_id_by_key,
                live_segment_keys,
                fixed,
            )
    else:
        face_planes = np.full((source_faces.shape[0], 3, 3), np.nan, dtype=np.float64)
    declared_source_faces = frozenset(
        frozenset(constrained_faces[row][1])
        for row in np.flatnonzero(~np.isnan(face_planes[:, 0, 0])).tolist()
    )
    witness_strata = np.zeros((vertex_ids.size,), dtype=np.int8)
    witness_entities = np.full((vertex_ids.size,), -1, dtype=np.int32)
    witness_parameters = np.zeros((vertex_ids.size, 2), dtype=np.float64)
    source_segment_keys = live_segment_keys
    source_segment_ids = live_segment_ids
    source_segments = live_segments
    if seam_keys:
        # Live constraints keep their original IDs. The source bank
        # interleaves ridge primitives in the same stable physical ID order.
        source_segment_keys = live_segment_keys + seam_keys
        source_segment_keys.sort(key=lambda key: edge_id_by_key[key])
        source_segment_ids = np.asarray(
            [edge_id_by_key[key] for key in source_segment_keys], dtype=np.int64
        )
        source_segments = np.asarray(
            [(vertex_row_by_id[a], vertex_row_by_id[b]) for a, b in source_segment_keys],
            dtype=np.int32,
        )
    source_segment_vertices = {
        row: frozenset(key) for row, key in enumerate(source_segment_keys)
    }
    source_segment_group_rows = {
        int(identifier): row for row, identifier in enumerate(source_segment_ids.tolist())
    }
    for row, (key, endpoint) in endpoint_witnesses.items():
        # One primitive per source group; witness entities index primitive rows.
        witness_strata[row] = 1  # SourceStratum::kSegment
        witness_entities[row] = source_segment_group_rows[edge_id_by_key[key]]
        witness_parameters[row, 0] = endpoint
    source_complex = TetMeshSourceComplex(
        points=points,
        faces=source_faces,
        face_group_rows=face_groups,
        face_tolerances=np.zeros((source_faces.shape[0],), dtype=np.float64),
        segments=source_segments,
        segment_group_rows=np.arange(source_segment_ids.size, dtype=np.int32),
        segment_tolerances=np.zeros((source_segment_ids.size,), dtype=np.float64),
        witness_strata=witness_strata,
        witness_entities=witness_entities,
        witness_parameters=witness_parameters,
        face_ids=source_face_ids,
        segment_ids=source_segment_ids,
        face_planes=face_planes,
        face_plane_tolerances=np.zeros((source_faces.shape[0],), dtype=np.float64),
    )

    source_map = geometry.source_map
    frame_id = coordinate_contract.reference_frame
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError(
            "Tetra adaptation requires the literal coordinate reference frame."
        )
    if implicit_source is not None:
        source_identity = (
            implicit_source.source_id,
            implicit_source.source_revision,
            implicit_source.binding_id,
            frame_id,
        )
    elif source_map is None:
        source_identity = (
            mesh.mesh_id,
            mesh.numeric_version,
            "",
            frame_id,
        )
    else:
        source_identity = (
            source_map.source_id,
            source_map.source_revision,
            source_map.source_map_id,
            frame_id,
        )
    if any(not isinstance(value, str) for value in source_identity):
        raise TypeError("Tetra source identity fields must be exact strings.")
    native_implicit = (
        None
        if implicit_source is None
        else TetMeshImplicitSource(
            implicit_source.value,
            implicit_source.gradient,
            np.asarray(sorted(implicit_groups), dtype=np.int32),
            implicit_source.tolerance,
            0,
            implicit_source.source_id,
            implicit_source.source_revision,
            implicit_source.binding_id,
            implicit_source.frame_id,
        )
    )

    native = TetMesh3D(
        points,
        source_cells.astype(np.int32),
        regions,
        source_faces,
        source_face_ids[face_groups],
        live_segments,
        live_segment_ids,
        boundary_policy=PlcBoundaryPolicy.CONFORMING,
        protection_radii=np.zeros((vertex_ids.size,), dtype=np.float64),
        protected_vertices=np.asarray(fixed, dtype=np.int8),
        vertex_ids=vertex_ids,
        cell_ids=source_cell_ids,
        facet_ids=source_face_row_ids,
        segment_ids=live_segment_ids,
        source=source_complex,
        implicit_source=native_implicit,
        source_id=source_identity[0],
        source_revision=source_identity[1],
        source_map_id=source_identity[2],
        frame_id=source_identity[3],
        periodic_state=periodic_state,
    )
    snapshot = native.arrays()
    native_vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    native_cell_ids = np.asarray(snapshot.cell_ids, dtype=np.int64)
    if set(native_vertex_ids.tolist()) != set(vertex_ids.tolist()) or set(
        native_cell_ids.tolist()
    ) != set(source_cell_ids.tolist()):
        raise ValueError("Native tetra source IDs do not preserve exact lifted ancestry.")
    source_vertex_row_by_id = {
        int(identifier): row for row, identifier in enumerate(vertex_ids.tolist())
    }
    source_block_by_cell_id = {
        int(identifier): int(block)
        for identifier, block in zip(
            source_cell_ids.tolist(), source_cell_blocks.tolist(), strict=True
        )
    }
    vertex_target_ids = {
        int(identifier): int(identifier) for identifier in native_vertex_ids.tolist()
    }
    cell_target_ids = {
        int(identifier): int(identifier) for identifier in native_cell_ids.tolist()
    }
    cell_blocks = {
        int(identifier): source_block_by_cell_id[int(identifier)]
        for identifier in native_cell_ids.tolist()
    }
    cell_ancestry = {
        int(identifier): {int(identifier): EntityLineageKind.PRESERVED}
        for identifier in native_cell_ids.tolist()
    }
    vertex_stencils = {
        int(identifier): {int(identifier): Fraction(1)}
        for identifier in native_vertex_ids.tolist()
    }
    metric_by_vertex = {
        int(identifier): metric[source_vertex_row_by_id[int(identifier)]].copy()
        for identifier in native_vertex_ids.tolist()
    }
    fixed_vertices_native = {
        int(vertex_ids[row]) for row in np.flatnonzero(fixed).tolist()
    }
    families = tuple((block.name, block.cell_kind) for block in mesh.blocks)
    periodic_vertex_windings = (
        {
            int(member.vertex_id): np.asarray(member.winding, dtype=np.int64).copy()
            for orbit in periodic_state.orbits
            for member in orbit.members
        }
        if periodic_state is not None
        else {}
    )
    return _NativeState(
        native=native,
        snapshot=snapshot,
        source_mesh=mesh,
        source_points=points,
        source_exact_points=exact_source_points,
        source_has_authored_rounding=has_authored_rounding,
        implicit_groups=implicit_groups,
        target_mesh=mesh,
        periodic_vertex_windings=periodic_vertex_windings,
        periodic_state=periodic_state,
        source_metric=metric,
        source_metric_log=_metric_logarithms(metric),
        source_cell_ids=source_cell_ids,
        source_vertex_ids=vertex_ids,
        source_vertex_rows=source_vertex_row_by_id,
        source_cell_vertices=source_cells,
        source_cell_regions=regions,
        source_cell_blocks=source_cell_blocks,
        source_region_volumes=_region_volumes(points, source_cells, regions),
        source_identity=source_identity,
        vertex_target_ids=vertex_target_ids,
        cell_target_ids=cell_target_ids,
        cell_blocks=cell_blocks,
        cell_ancestry=cell_ancestry,
        vertex_stencils=vertex_stencils,
        metric_by_vertex=metric_by_vertex,
        fixed_vertices=fixed_vertices_native,
        protected_edge_ids=protected_edge_ids,
        source_edge_ids=edge_id_by_key,
        edge_classes=edge_class_by_key,
        source_face_group_rows=source_face_group_rows,
        source_segment_group_rows=source_segment_group_rows,
        source_face_vertices=source_face_vertices,
        source_segment_vertices=source_segment_vertices,
        declared_source_faces=declared_source_faces,
        source_group_triangles=dict(source_group_triangles),
        next_native_vertex=int(np.max(native_vertex_ids)) + 1,
        next_native_cell=int(np.max(native_cell_ids)) + 1,
        next_native_facet=(
            int(np.max(snapshot.facet_ids)) + 1 if snapshot.facet_ids.size else 0
        ),
        next_native_segment=(
            int(np.max(snapshot.segment_ids)) + 1 if snapshot.segment_ids.size else 0
        ),
        next_target_vertex=int(np.max(vertex_ids)) + 1,
        next_target_cell=int(np.max(source_cell_ids)) + 1,
        block_families=families,
        history_vertex=[],
        history_cells=[],
        changed_source_cell_ids=set(),
        unrepresentable_edges=set(),
    )


class _SplitProposal(NamedTuple):
    edge: tuple[int, int]
    parameter: float
    point: np.ndarray
    metric: np.ndarray
    stencil: dict[int, Fraction]
    native_vertex_id: int
    target_vertex_id: int


def _split_stencil(
    first_stencil: dict[int, Fraction],
    second_stencil: dict[int, Fraction],
    parameter: Fraction,
    /,
) -> dict[int, Fraction]:
    combined: dict[int, Fraction] = defaultdict(Fraction)
    for identifier, weight in first_stencil.items():
        combined[identifier] += (1 - parameter) * weight
    for identifier, weight in second_stencil.items():
        combined[identifier] += parameter * weight
    return {identifier: value for identifier, value in sorted(combined.items()) if value}


def _stencil_image(
    state: _NativeState, stencil: dict[int, Fraction], /
) -> tuple[Fraction, Fraction, Fraction]:
    """Exact P1 source image ``sum_i w_i X_i`` of a source-vertex stencil."""
    coordinates = state.source_exact_points
    rows = state.source_vertex_rows
    image = [Fraction(0), Fraction(0), Fraction(0)]
    for identifier, weight in stencil.items():
        point = coordinates[rows[identifier]]
        for axis in range(3):
            image[axis] += weight * point[axis]
    return image[0], image[1], image[2]


def _split_member_admissible(
    state: _NativeState,
    first: np.ndarray,
    second: np.ndarray,
    first_stencil: dict[int, Fraction],
    second_stencil: dict[int, Fraction],
    parameter: Fraction,
    boundary: bool,
    /,
) -> bool:
    """Whether ``parameter`` realizes one split point consistently in binary64.

    The carrier is the rounded exact source image ``I`` of the split stencil;
    the native edge witness rounds the current-edge point
    ``E = a + t (b - a)``, so both roundings must agree, and the stencil
    weights must be binary64. PWL facets and protected segments use exact
    carrier predicates, so there ``E`` must be representable and equal to
    ``I``. An implicit-only face is independently admitted in the source band.
    """
    stencil = _split_stencil(first_stencil, second_stencil, parameter)
    if any(Fraction(float(weight)) != weight for weight in stencil.values()):
        return False
    image = _stencil_image(state, stencil)
    for axis, (a, b) in enumerate(zip(first.tolist(), second.tolist(), strict=True)):
        start = Fraction(float(a))
        edge_point = start + parameter * (Fraction(float(b)) - start)
        if boundary:
            if edge_point != image[axis] or Fraction(float(edge_point)) != edge_point:
                return False
        elif float(edge_point) != float(image[axis]):
            return False
    return True


def _constrained_edges(
    state: _NativeState, topology: _TetTopology, /
) -> set[tuple[int, int]]:
    """Edges whose incident PWL support requires an exactly representable split."""
    edges = set(topology.segment_row)
    records = _boundary_records(state) if state.implicit_groups else None
    for face in topology.facet_row:
        if records is not None and records.facets[face][1] in state.implicit_groups:
            continue
        edges.update(((face[0], face[1]), (face[0], face[2]), (face[1], face[2])))
    return edges


def _exact_split_parameters(
    state: _NativeState,
    topology: _TetTopology,
    points: np.ndarray,
    metrics: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    constrained: set[tuple[int, int]],
    /,
) -> tuple[Fraction, ...] | None:
    """Short dyadic split parameters of current edges (``_dyadic_split_parameters``)."""
    members = []
    for edge in edges:
        first_row, second_row = topology.vertex_row[edge[0]], topology.vertex_row[edge[1]]
        members.append(
            (
                points[first_row],
                points[second_row],
                metrics[first_row],
                metrics[second_row],
                state.vertex_stencils[edge[0]],
                state.vertex_stencils[edge[1]],
                edge in constrained,
            )
        )
    return _dyadic_split_parameters(state, members)


def _dyadic_split_parameters(
    state: _NativeState,
    members: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[int, Fraction],
            dict[int, Fraction],
            bool,
        ]
    ],
    /,
) -> tuple[Fraction, ...] | None:
    """Short dyadic split parameters with a consistent binary64 realization.

    ``members`` hold each orbit edge's oriented endpoint points, metrics and
    stencils and whether its PWL support requires a representable split. Each
    metric midpoint is rounded to the dyadic grid ``2**-bits`` for
    ``bits = _SPLIT_PARAMETER_BITS, ..., 1``; the finest level at which every
    orbit member is admissible (``_split_member_admissible``) wins. Short
    parameters keep nested ancestry weights binary64. ``None`` means no dyadic
    point of the edge has a consistent realization.
    """
    parameters = metric_midpoints(
        np.asarray([member[0] for member in members], dtype=np.float64).reshape((-1, 3)),
        np.asarray([member[1] for member in members], dtype=np.float64).reshape((-1, 3)),
        np.asarray([member[2] for member in members], dtype=np.float64).reshape(
            (-1, 3, 3)
        ),
        np.asarray([member[3] for member in members], dtype=np.float64).reshape(
            (-1, 3, 3)
        ),
    )[0].tolist()
    tried: set[tuple[Fraction, ...]] = set()
    for bits in range(_SPLIT_PARAMETER_BITS, 0, -1):
        scale = 1 << bits
        candidate = tuple(
            Fraction(min(max(round(parameter * scale), 1), scale - 1), scale)
            for parameter in parameters
        )
        if candidate in tried:
            continue
        tried.add(candidate)
        if all(
            _split_member_admissible(
                state, first, second, first_stencil, second_stencil, value, boundary
            )
            for (
                first,
                second,
                _,
                _,
                first_stencil,
                second_stencil,
                boundary,
            ), value in zip(members, candidate, strict=True)
        ):
            return candidate
    return None


def _split_edit(
    state: _NativeState, split_parameters: dict[tuple[int, int], Fraction], /
) -> tuple[TetMeshEdit, dict[str, Any]]:
    """Build one complete disjoint metric edge-split transaction."""
    if not split_parameters:
        raise ValueError("A split edit requires at least one finite candidate.")
    snapshot = state.snapshot
    topology = _state_topology(state)
    points = np.asarray(snapshot.points, dtype=np.float64)
    native_vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)

    source_vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(state.source_vertex_ids.tolist())
    }
    canonical_edges: list[tuple[int, int]] = list(split_parameters)
    if any(edge != _edge_key(edge) for edge in canonical_edges):
        raise ValueError("A split transaction names canonical edge keys.")

    new_native_vertex_ids = _fresh_ids(
        state.next_native_vertex, len(canonical_edges), "native vertex IDs"
    )
    new_target_vertex_ids = _fresh_ids(
        state.next_target_vertex, len(canonical_edges), "target vertex IDs"
    )
    proposals: list[_SplitProposal] = []
    for row, edge in enumerate(canonical_edges):
        if edge not in topology.edge_cells:
            raise ValueError(
                "A split candidate is not an edge of the current native mesh."
            )
        first, second = edge
        fraction = split_parameters[edge]
        if not 0 < fraction < 1:
            raise ValueError("A tetra split parameter must lie strictly inside its edge.")
        stencil = _split_stencil(
            state.vertex_stencils[first], state.vertex_stencils[second], fraction
        )
        point = np.asarray(
            [float(value) for value in _stencil_image(state, stencil)], dtype=np.float64
        )
        if not np.all(np.isfinite(point)):
            raise FloatingPointError("The metric split point is not finite.")
        _binary64_ancestry(stencil)
        tensor = _source_metric_from_stencil(
            state.source_metric,
            source_vertex_rows,
            stencil,
            source_metric_log=state.source_metric_log,
        )
        proposals.append(
            _SplitProposal(
                edge,
                float(fraction),
                point,
                tensor,
                stencil,
                int(new_native_vertex_ids[row]),
                int(new_target_vertex_ids[row]),
            )
        )

    proposal_by_edge = {proposal.edge: proposal for proposal in proposals}
    facet_vertices = np.asarray(
        native_vertex_ids[np.asarray(snapshot.faces, dtype=np.int64)],
        dtype=np.int64,
    ).reshape((-1, 3))
    facet_ids = np.asarray(snapshot.facet_ids, dtype=np.int64)
    facet_sources = np.asarray(snapshot.face_sources, dtype=np.int64)
    facet_orientations = np.asarray(snapshot.facet_orientations, dtype=np.int8)
    protected_facets = np.asarray(snapshot.protected_facets, dtype=np.bool_)
    facet_group_by_key: dict[tuple[int, int, int], int] = {}
    facet_rows_by_key: dict[
        tuple[int, int, int], tuple[int, int, tuple[int, int, int], int, bool]
    ] = {}
    for row in range(facet_vertices.shape[0]):
        vertices = (
            int(facet_vertices[row, 0]),
            int(facet_vertices[row, 1]),
            int(facet_vertices[row, 2]),
        )
        key = _face_key(vertices)
        source_id = int(facet_sources[row])
        group = state.source_face_group_rows.get(source_id)
        if group is None:
            raise ValueError("A constrained native facet lost its source group.")
        facet_group_by_key[key] = group
        facet_rows_by_key[key] = (
            int(facet_ids[row]),
            source_id,
            vertices,
            int(facet_orientations[row]),
            bool(protected_facets[row]),
        )

    segment_vertices = np.asarray(
        native_vertex_ids[np.asarray(snapshot.segments, dtype=np.int64)],
        dtype=np.int64,
    ).reshape((-1, 2))
    segment_ids = np.asarray(snapshot.segment_ids, dtype=np.int64)
    segment_sources = np.asarray(snapshot.segment_sources, dtype=np.int64)
    protected_segments = np.asarray(snapshot.protected_segments, dtype=np.bool_)
    segment_rows_by_key: dict[
        tuple[int, int], tuple[int, int, tuple[int, int], bool]
    ] = {}
    for row in range(segment_vertices.shape[0]):
        vertices = (
            int(segment_vertices[row, 0]),
            int(segment_vertices[row, 1]),
        )
        key = _edge_key(vertices)
        source_id = int(segment_sources[row])
        group = state.source_segment_group_rows.get(source_id)
        if group is None:
            raise ValueError("A constrained native segment lost its source group.")
        segment_rows_by_key[key] = (
            int(segment_ids[row]),
            source_id,
            vertices,
            bool(protected_segments[row]),
        )

    removed_facet_ids: list[int] = []
    replacement_facet_ids: list[int] = []
    replacement_facet_vertices: list[tuple[int, int, int]] = []
    replacement_facet_sources: list[int] = []
    replacement_facet_orientations: list[int] = []
    replacement_protected_facets: list[int] = []
    next_facet = state.next_native_facet
    for edge in canonical_edges:
        first, second = edge
        proposal = proposal_by_edge[edge]
        for key, (facet_id, source_id, ordered, orientation, protected) in sorted(
            facet_rows_by_key.items()
        ):
            if first not in key or second not in key:
                continue
            directed = None
            for local in range(3):
                a = int(ordered[local])
                b = int(ordered[(local + 1) % 3])
                if {a, b} == {first, second}:
                    directed = (a, b, int(ordered[(local + 2) % 3]))
                    break
            if directed is None:
                raise ValueError("A constrained source facet has malformed edge order.")
            a, b, c = directed
            middle = proposal.native_vertex_id
            group = state.source_face_group_rows[source_id]
            removed_facet_ids.append(facet_id)
            for face in ((a, middle, c), (middle, b, c)):
                replacement_facet_ids.append(next_facet)
                replacement_facet_vertices.append(face)
                replacement_facet_sources.append(group)
                replacement_facet_orientations.append(orientation)
                replacement_protected_facets.append(int(protected))
                facet_group_by_key[_face_key(face)] = group
                next_facet += 1

    removed_segment_ids: list[int] = []
    replacement_segment_ids: list[int] = []
    replacement_segment_vertices: list[tuple[int, int]] = []
    replacement_segment_sources: list[int] = []
    replacement_protected_segments: list[int] = []
    next_segment = state.next_native_segment
    for edge in canonical_edges:
        record = segment_rows_by_key.get(edge)
        if record is None:
            continue
        segment_id, source_id, ordered, protected = record
        first, second = (int(value) for value in ordered)
        middle = proposal_by_edge[edge].native_vertex_id
        group = state.source_segment_group_rows[source_id]
        removed_segment_ids.append(segment_id)
        for segment in ((first, middle), (middle, second)):
            replacement_segment_ids.append(next_segment)
            replacement_segment_vertices.append(segment)
            replacement_segment_sources.append(group)
            replacement_protected_segments.append(int(protected))
            next_segment += 1

    removed_cells = sorted(
        {
            int(np.asarray(snapshot.cell_ids, dtype=np.int64)[row])
            for edge in canonical_edges
            for row in topology.edge_cells[edge]
        }
    )
    replacement_records: list[
        tuple[int, int, tuple[int, int], tuple[int, int, int, int], int, int]
    ] = []
    new_points = np.asarray([proposal.point for proposal in proposals], dtype=np.float64)
    extended_points = np.concatenate((points, new_points), axis=0)
    extended_vertex_ids = np.concatenate(
        (native_vertex_ids, new_native_vertex_ids), axis=0
    )
    extended_vertex_rows = dict(topology.vertex_row)
    extended_vertex_rows.update(
        {
            int(identifier): points.shape[0] + row
            for row, identifier in enumerate(new_native_vertex_ids.tolist())
        }
    )
    extended_topology = topology._replace(vertex_row=extended_vertex_rows)
    replacement_ids: list[int] = []
    replacement_regions: list[int] = []
    replacement_targets: list[int] = []
    replacement_blocks: list[int] = []
    replacement_ancestry: list[tuple[int, ...]] = []
    next_cell = state.next_native_cell
    next_target_cell = state.next_target_cell
    cell_regions = np.asarray(snapshot.tetrahedron_regions, dtype=np.int32)
    cell_row_by_id = topology.cell_row
    for parent_id in removed_cells:
        parent_row = cell_row_by_id[parent_id]
        vertices = topology.cell_vertex_ids[parent_row]
        matching = [
            edge for edge in canonical_edges if set(edge).issubset(set(vertices.tolist()))
        ]
        if len(matching) != 1:
            raise ValueError(
                "A disjoint split batch must touch one selected edge per cell."
            )
        first, second = matching[0]
        other = [
            int(value) for value in vertices.tolist() if int(value) not in matching[0]
        ]
        if len(other) != 2:
            raise ValueError("A tetra split cavity cell has invalid edge incidence.")
        middle = proposal_by_edge[matching[0]].native_vertex_id
        children = (
            (first, middle, other[0], other[1]),
            (middle, second, other[0], other[1]),
        )
        # Both children share the sign of (first, second, other...): reverse together.
        child_ids = _oriented_tetrahedra(
            extended_points,
            extended_topology,
            np.asarray(children, dtype=np.int64),
            PredicateMode.EXACT,
            reversible=True,
        )
        if child_ids is None:
            raise FloatingPointError(
                "A tetra split child has unresolved exact orientation."
            )
        child_vertices = extended_vertex_ids[child_ids]
        for side, row in enumerate(child_vertices.tolist()):
            replacement_records.append(
                (
                    parent_id,
                    side,
                    matching[0],
                    _tetra_key(row),
                    int(cell_regions[parent_row]),
                    int(state.cell_blocks[parent_id]),
                )
            )
    replacement_records.sort(key=lambda record: (record[0], record[1], record[2]))
    for row, (parent_id, _, _, vertices, region, block) in enumerate(replacement_records):
        replacement_ids.append(next_cell + row)
        replacement_targets.append(next_target_cell + row)
        replacement_regions.append(region)
        replacement_blocks.append(block)
        replacement_ancestry.append(tuple(sorted(state.cell_ancestry[parent_id])))

    cell_facet_sources = np.full((len(replacement_records), 4), -1, dtype=np.int32)
    for row, record in enumerate(replacement_records):
        vertices = record[3]
        for local_face, face in enumerate(_TETRA_FACES):
            key = _face_key(vertices[index] for index in face)
            cell_facet_sources[row, local_face] = facet_group_by_key.get(key, -1)

    changed_ids = np.asarray(new_native_vertex_ids, dtype=np.uint64)
    changed_points = np.asarray(
        [proposal.point for proposal in proposals], dtype=np.float64
    )
    changed_tensors = [proposal.metric for proposal in proposals]
    changed_stencils = [proposal.stencil for proposal in proposals]
    native_changed_stencils = [
        _current_edge_stencil(state, proposal) for proposal in proposals
    ]
    changed_ancestry = _ancestry_csr(native_changed_stencils)
    cell_offsets = np.arange(len(replacement_records) + 1, dtype=np.int64)
    cell_ancestor_ids = np.asarray(
        [record[0] for record in replacement_records], dtype=np.uint64
    )
    changed_ancestor_winding = _native_ancestor_windings(state, changed_ancestry[1])
    edit = TetMeshEdit(
        kind=TetMeshEditKind.SPLIT,
        expected_generation=int(snapshot.generation),
        source_id=state.source_identity[0],
        source_revision=state.source_identity[1],
        source_map_id=state.source_identity[2],
        frame_id=state.source_identity[3],
        split_edges=np.asarray(canonical_edges, dtype=np.uint64),
        removed_cell_ids=np.asarray(removed_cells, dtype=np.uint64),
        changed_vertex_ids=changed_ids,
        changed_vertex_points=changed_points,
        changed_witness_strata=np.zeros((len(proposals),), dtype=np.int8),
        changed_witness_entities=np.full((len(proposals),), -1, dtype=np.int32),
        changed_witness_parameters=np.zeros((len(proposals), 2), dtype=np.float64),
        changed_witness_deviations=np.zeros((len(proposals),), dtype=np.float64),
        changed_protection_radii=np.zeros((len(proposals),), dtype=np.float64),
        changed_protected_vertices=np.zeros((len(proposals),), dtype=np.int8),
        changed_edge_witness_present=np.ones((len(proposals),), dtype=np.int8),
        changed_edge_witness_endpoints=np.asarray(canonical_edges, dtype=np.uint64),
        changed_edge_parameters=np.asarray(
            [proposal.parameter for proposal in proposals], dtype=np.float64
        ),
        changed_vertex_ancestry_offsets=changed_ancestry[0],
        changed_vertex_ancestor_ids=changed_ancestry[1],
        changed_vertex_ancestor_winding=changed_ancestor_winding,
        changed_vertex_ancestry_weights=changed_ancestry[2],
        replacement_cell_ancestry_offsets=cell_offsets,
        replacement_cell_ancestor_ids=cell_ancestor_ids,
        replacement_cell_ids=np.asarray(replacement_ids, dtype=np.uint64),
        replacement_cell_vertices=np.asarray(
            [record[3] for record in replacement_records], dtype=np.uint64
        ),
        replacement_cell_facet_sources=cell_facet_sources,
        replacement_regions=np.asarray(replacement_regions, dtype=np.int32),
        removed_facet_ids=(
            np.asarray(sorted(removed_facet_ids), dtype=np.uint64)
            if removed_facet_ids
            else None
        ),
        replacement_facet_ids=(
            np.asarray(replacement_facet_ids, dtype=np.uint64)
            if replacement_facet_ids
            else None
        ),
        replacement_facet_vertices=(
            np.asarray(replacement_facet_vertices, dtype=np.uint64).reshape((-1, 3))
            if replacement_facet_vertices
            else None
        ),
        replacement_facet_sources=(
            np.asarray(replacement_facet_sources, dtype=np.int32)
            if replacement_facet_sources
            else None
        ),
        replacement_facet_orientations=(
            np.asarray(replacement_facet_orientations, dtype=np.int8)
            if replacement_facet_orientations
            else None
        ),
        replacement_protected_facets=(
            np.asarray(replacement_protected_facets, dtype=np.int8)
            if replacement_protected_facets
            else None
        ),
        removed_segment_ids=(
            np.asarray(sorted(removed_segment_ids), dtype=np.uint64)
            if removed_segment_ids
            else None
        ),
        replacement_segment_ids=(
            np.asarray(replacement_segment_ids, dtype=np.uint64)
            if replacement_segment_ids
            else None
        ),
        replacement_segment_vertices=(
            np.asarray(replacement_segment_vertices, dtype=np.uint64).reshape((-1, 2))
            if replacement_segment_vertices
            else None
        ),
        replacement_segment_sources=(
            np.asarray(replacement_segment_sources, dtype=np.int32)
            if replacement_segment_sources
            else None
        ),
        replacement_protected_segments=(
            np.asarray(replacement_protected_segments, dtype=np.int8)
            if replacement_protected_segments
            else None
        ),
    )
    metadata = {
        "proposals": proposals,
        "removed_cells": removed_cells,
        "replacement_native_cell_ids": np.asarray(replacement_ids, dtype=np.int64),
        "replacement_target_cell_ids": np.asarray(replacement_targets, dtype=np.int64),
        "replacement_parent_ids": np.asarray(
            [record[0] for record in replacement_records], dtype=np.int64
        ),
        "replacement_blocks": np.asarray(replacement_blocks, dtype=np.int32),
        "replacement_ancestry": replacement_ancestry,
        "next_native_cell": next_cell + len(replacement_ids),
        "next_target_cell": next_target_cell + len(replacement_targets),
        "next_native_facet": next_facet + len(replacement_facet_ids),
        "next_native_segment": next_segment + len(replacement_segment_ids),
        "next_native_vertex": state.next_native_vertex + len(proposals),
        "next_target_vertex": state.next_target_vertex + len(proposals),
        "changed_metrics": changed_tensors,
        "changed_stencils": changed_stencils,
        "replacement_facet_groups": tuple(replacement_facet_sources),
        "replacement_segment_groups": tuple(replacement_segment_sources),
    }
    return edit, metadata


class _PeriodicCandidateRejected(Exception):
    """The complete lifted operation does not preserve a periodic quotient."""


def _candidate_tetra_snapshot(
    state: _NativeState, edit: TetMeshEdit, /
) -> _CandidateTetraSnapshot:
    current = state.snapshot
    current_vertex_ids = np.asarray(current.vertex_ids, dtype=np.int64)
    removed_vertices = set(
        map(
            int,
            np.asarray(
                edit.removed_vertex_ids if edit.removed_vertex_ids is not None else (),
                dtype=np.int64,
            ).tolist(),
        )
    )
    points_by_id = {
        int(identifier): np.asarray(point, dtype=np.float64)
        for identifier, point in zip(
            current_vertex_ids.tolist(),
            np.asarray(current.points, dtype=np.float64),
            strict=True,
        )
        if int(identifier) not in removed_vertices
    }
    changed_ids = np.asarray(
        edit.changed_vertex_ids if edit.changed_vertex_ids is not None else (),
        dtype=np.int64,
    )
    changed_points = np.asarray(
        edit.changed_vertex_points
        if edit.changed_vertex_points is not None
        else np.zeros((0, 3), dtype=np.float64),
        dtype=np.float64,
    ).reshape((-1, 3))
    if changed_ids.shape != (changed_points.shape[0],):
        raise ValueError("Native candidate changed vertex records are malformed.")
    for identifier, point in zip(changed_ids.tolist(), changed_points, strict=True):
        if int(identifier) in removed_vertices:
            raise ValueError("A native candidate cannot both remove and change a vertex.")
        points_by_id[int(identifier)] = point
    vertex_ids = np.asarray(sorted(points_by_id), dtype=np.int64)
    points = np.asarray([points_by_id[int(identifier)] for identifier in vertex_ids])
    vertex_row = {
        int(identifier): row for row, identifier in enumerate(vertex_ids.tolist())
    }

    current_cell_ids = np.asarray(current.cell_ids, dtype=np.int64)
    current_cells = np.asarray(current.tetrahedra, dtype=np.int64).reshape((-1, 4))
    current_regions = np.asarray(current.tetrahedron_regions, dtype=np.int32)
    # Relocation carries its complete checked star in removed_cell_ids, but
    # retains those cells. Only topology-changing edits retire the listed cells.
    removed_cells = set(
        map(
            int,
            np.asarray(
                edit.removed_cell_ids
                if edit.kind is not TetMeshEditKind.RELOCATE
                and edit.removed_cell_ids is not None
                else (),
                dtype=np.int64,
            ).tolist(),
        )
    )
    cell_records: dict[int, tuple[tuple[int, int, int, int], int]] = {}
    for row in range(current_cells.shape[0]):
        identifier = int(current_cell_ids[row])
        if identifier not in removed_cells:
            local = (
                int(current_cells[row, 0]),
                int(current_cells[row, 1]),
                int(current_cells[row, 2]),
                int(current_cells[row, 3]),
            )
            vertices = (
                int(current_vertex_ids[local[0]]),
                int(current_vertex_ids[local[1]]),
                int(current_vertex_ids[local[2]]),
                int(current_vertex_ids[local[3]]),
            )
            cell_records[identifier] = (vertices, int(current_regions[row]))
    replacement_ids = np.asarray(
        edit.replacement_cell_ids if edit.replacement_cell_ids is not None else (),
        dtype=np.int64,
    )
    replacement_vertices = np.asarray(
        edit.replacement_cell_vertices
        if edit.replacement_cell_vertices is not None
        else np.zeros((0, 4), dtype=np.int64),
        dtype=np.int64,
    ).reshape((-1, 4))
    replacement_regions = np.asarray(
        edit.replacement_regions
        if edit.replacement_regions is not None
        else np.zeros((replacement_ids.size,), dtype=np.int32),
        dtype=np.int32,
    )
    if (
        replacement_ids.shape != (replacement_vertices.shape[0],)
        or replacement_regions.shape != replacement_ids.shape
    ):
        raise ValueError("Native candidate replacement cell records are malformed.")
    for identifier, vertices, region in zip(
        replacement_ids.tolist(),
        replacement_vertices,
        replacement_regions.tolist(),
        strict=True,
    ):
        if int(identifier) in cell_records:
            raise ValueError("Native candidate replacement cell identity is not fresh.")
        cell_records[int(identifier)] = (
            _tetra_key(vertices),
            int(region),
        )
    cell_ids = np.asarray(sorted(cell_records), dtype=np.int64)
    tetrahedra = np.asarray(
        [
            [vertex_row[int(vertex)] for vertex in cell_records[int(identifier)][0]]
            for identifier in cell_ids.tolist()
        ],
        dtype=np.int32,
    ).reshape((-1, 4))
    regions = np.asarray(
        [cell_records[int(identifier)][1] for identifier in cell_ids.tolist()],
        dtype=np.int32,
    )
    return _CandidateTetraSnapshot(points, tetrahedra, regions, vertex_ids, cell_ids)


def _prepare_periodic_native_edit(
    state: _NativeState,
    native_edit: TetMeshEdit,
    metadata: dict[str, Any],
    /,
) -> tuple[TetMeshEdit, tuple[CellMesh, TetMeshPeriodicState] | None]:
    """Stage one native edit as a complete periodic quotient edit.

    Returns the edit with its periodic payload and, on a periodic lift, the
    staged target mesh with its native quotient state (committed together by
    ``_commit_periodic_target`` once the native edit applies).
    """
    source = state.target_mesh
    if source.periodic_topology is None:
        return native_edit, None
    candidate = _candidate_tetra_snapshot(state, native_edit)
    candidate_vertex_ids = set(map(int, candidate.vertex_ids.tolist()))
    candidate_cell_ids = set(map(int, candidate.cell_ids.tolist()))
    proposals = metadata.get("proposals", ())
    split_by_native_id = {
        int(proposal.native_vertex_id): proposal
        for proposal in proposals
        if isinstance(proposal, _SplitProposal)
    }
    relocation_by_native_id = {
        int(proposal["vertex"]): proposal
        for proposal in proposals
        if isinstance(proposal, Mapping) and "current_stencil" in proposal
    }
    vertex_target_by_native = {
        int(native_id): int(target_id)
        for native_id, target_id in state.vertex_target_ids.items()
        if int(native_id) in candidate_vertex_ids
    }
    for native_id, proposal in split_by_native_id.items():
        vertex_target_by_native[native_id] = int(proposal.target_vertex_id)
    if set(vertex_target_by_native) != set(map(int, candidate.vertex_ids.tolist())):
        raise _PeriodicCandidateRejected(
            "Periodic candidate vertex IDs do not close over the native edit."
        )

    replacement_native = metadata.get(
        "replacement_native_cell_ids", metadata.get("replacement_native_ids")
    )
    replacement_target = metadata.get(
        "replacement_target_cell_ids", metadata.get("replacement_target_ids")
    )
    replacement_blocks = metadata.get("replacement_blocks")
    cell_target_by_native = {
        int(native_id): int(target_id)
        for native_id, target_id in state.cell_target_ids.items()
        if int(native_id) in candidate_cell_ids
    }
    block_by_native = {
        int(native_id): int(block)
        for native_id, block in state.cell_blocks.items()
        if int(native_id) in candidate_cell_ids
    }
    if replacement_native is not None and replacement_target is not None:
        replacement_native = np.asarray(replacement_native, dtype=np.int64)
        replacement_target = np.asarray(replacement_target, dtype=np.int64)
        replacement_blocks = np.asarray(replacement_blocks, dtype=np.int32)
        if not (
            replacement_native.shape
            == replacement_target.shape
            == replacement_blocks.shape
        ):
            raise ValueError("Native candidate target cell maps are malformed.")
        cell_target_by_native.update(
            zip(replacement_native.tolist(), replacement_target.tolist(), strict=True)
        )
        block_by_native.update(
            zip(replacement_native.tolist(), replacement_blocks.tolist(), strict=True)
        )
    if set(cell_target_by_native) != set(map(int, candidate.cell_ids.tolist())):
        raise _PeriodicCandidateRejected(
            "Periodic candidate cell IDs do not close over the native edit."
        )

    target_order = np.argsort(
        np.asarray(
            [
                vertex_target_by_native[int(identifier)]
                for identifier in candidate.vertex_ids
            ],
            dtype=np.int64,
        ),
        kind="stable",
    )
    target_vertex_ids = np.asarray(
        [
            vertex_target_by_native[int(candidate.vertex_ids[row])]
            for row in target_order.tolist()
        ],
        dtype=np.int64,
    )
    target_points = np.asarray(candidate.points, dtype=np.float64)[target_order]
    target_row_by_native = {
        int(candidate.vertex_ids[row]): target_row
        for target_row, row in enumerate(target_order.tolist())
    }
    target_stencils: dict[int, dict[int, float]] = {}
    relational_stencils: dict[int, dict[int, Fraction]] = {}
    for native_id in candidate.vertex_ids.tolist():
        target_id = vertex_target_by_native[int(native_id)]
        if int(native_id) in split_by_native_id:
            proposal = split_by_native_id[int(native_id)]
            current_stencil = _current_edge_stencil(state, proposal)
            stencil = {
                int(state.vertex_target_ids[int(source_id)]): float(weight)
                for source_id, weight in current_stencil.items()
            }
            relational_stencils[target_id] = {
                source_id: Fraction(1) for source_id in stencil
            }
        elif int(native_id) in state.vertex_target_ids:
            relocation = relocation_by_native_id.get(int(native_id))
            if relocation is None:
                stencil = {int(state.vertex_target_ids[int(native_id)]): 1.0}
            else:
                stencil = {
                    int(state.vertex_target_ids[int(source_id)]): float(weight)
                    for source_id, weight in relocation["current_stencil"].items()
                }
                relational_stencils[target_id] = {
                    source_id: Fraction(1) for source_id in stencil
                }
        else:
            raise _PeriodicCandidateRejected(
                "Periodic candidate has a new vertex without an exact edge operation."
            )
        target_stencils[target_id] = stencil
        relational_stencils.setdefault(
            target_id, {source_id: Fraction(1) for source_id in stencil}
        )

    target_cell_ids = np.asarray(
        [cell_target_by_native[int(identifier)] for identifier in candidate.cell_ids],
        dtype=np.int64,
    )
    target_cells = np.asarray(
        [
            [
                target_row_by_native[int(candidate.vertex_ids[int(vertex)])]
                for vertex in cell
            ]
            for cell in candidate.tetrahedra
        ],
        dtype=np.int32,
    ).reshape((-1, 4))
    target_block_by_native = block_by_native
    block_cells: list[np.ndarray] = []
    block_ids: list[np.ndarray] = []
    for block in range(len(state.block_families)):
        rows = np.flatnonzero(
            np.asarray(
                [
                    target_block_by_native[int(identifier)] == block
                    for identifier in candidate.cell_ids
                ],
                dtype=np.bool_,
            )
        )
        order = rows[np.argsort(target_cell_ids[rows], kind="stable")]
        block_cells.append(target_cells[order])
        block_ids.append(target_cell_ids[order])
    edit_blocks = source_family_blocks(
        state.block_families, tuple(block_cells), tuple(block_ids)
    )
    mesh_blocks = tuple(
        CellBlock(
            block.name,
            block.cell_kind,
            block.cells,
            global_ids=block.cell_ids,
        )
        for block in edit_blocks
    )
    provisional = CellMesh(
        target_points, mesh_blocks, vertex_global_ids=target_vertex_ids
    )

    width = max((len(stencil) for stencil in target_stencils.values()), default=1)
    stencil_sources = np.full((target_vertex_ids.size, width), -1, dtype=np.int64)
    stencil_weights = np.zeros((target_vertex_ids.size, width), dtype=np.float64)
    stencil_valid = np.zeros((target_vertex_ids.size, width), dtype=np.bool_)
    for row, identifier in enumerate(target_vertex_ids.tolist()):
        stencil = target_stencils[int(identifier)]
        ordered_sources = sorted(stencil)
        weights = np.asarray([stencil[key] for key in ordered_sources], dtype=np.float64)
        if (
            not np.isfinite(weights).all()
            or sum(
                (Fraction.from_float(float(weight)) for weight in weights.tolist()),
                Fraction(0),
            )
            != 1
        ):
            raise _PeriodicCandidateRejected(
                "Periodic candidate interpolation weights do not sum to one."
            )
        stencil_sources[row, : len(ordered_sources)] = ordered_sources
        stencil_weights[row, : len(ordered_sources)] = weights
        stencil_valid[row, : len(ordered_sources)] = True

    source_ids = set(map(int, np.asarray(source.vertex_global_ids).tolist()))
    target_ids = set(map(int, target_vertex_ids.tolist()))
    vertex_relations: set[tuple[tuple[int, ...], tuple[int, ...], EntityLineageKind]] = (
        set()
    )
    source_vertex_row = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(source.vertex_global_ids).tolist())
    }
    target_vertex_row = {
        int(identifier): row for row, identifier in enumerate(target_vertex_ids.tolist())
    }
    changed_native_ids = set(
        map(
            int,
            np.asarray(
                native_edit.changed_vertex_ids
                if native_edit.changed_vertex_ids is not None
                else (),
                dtype=np.int64,
            ).tolist(),
        )
    )
    current_native_by_target = {
        int(target): int(native) for native, target in state.vertex_target_ids.items()
    }
    for identifier in source_ids & target_ids:
        native_id = current_native_by_target.get(identifier)
        if native_id is None:
            raise _PeriodicCandidateRejected(
                "Periodic current vertex identity is absent from the native map."
            )
        changed = native_id in changed_native_ids and not np.array_equal(
            source.coordinates[source_vertex_row[identifier]],
            target_points[target_vertex_row[identifier]],
        )
        vertex_relations.add(
            (
                (identifier,),
                (identifier,),
                EntityLineageKind.RELOCATED if changed else EntityLineageKind.PRESERVED,
            )
        )
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            continue
        removed = proposal.get("removed")
        kept = proposal.get("kept")
        if removed is not None and kept is not None:
            vertex_relations.add(
                (
                    (int(state.vertex_target_ids[int(removed)]),),
                    (int(state.vertex_target_ids[int(kept)]),),
                    EntityLineageKind.COLLAPSED_INTO,
                )
            )
    ordered_vertex_relations = sorted(
        vertex_relations,
        key=lambda row: (row[0], row[1], int(row[2])),
    )

    target_edge_keys = entity_keys(provisional, 1)
    target_face_keys = entity_keys(provisional, 2)
    current_cell_ids = set(
        map(int, np.asarray(source.entity_set(3).entity_ids, dtype=np.int64).tolist())
    )
    target_cell_id_set = set(map(int, target_cell_ids.tolist()))
    replacement_parent_native = metadata.get("replacement_parent_ids")
    replacement_parent_groups = metadata.get("replacement_parent_native_ids")
    replacement_parent_rows: tuple[tuple[int, ...], ...] = ()
    if replacement_parent_groups is not None:
        replacement_parent_rows = tuple(
            tuple(int(parent) for parent in row) for row in replacement_parent_groups
        )
    elif replacement_parent_native is not None:
        replacement_parent_rows = tuple(
            (int(parent),)
            for parent in np.asarray(replacement_parent_native, dtype=np.int64).tolist()
        )
    replacement_kinds = metadata.get("replacement_kinds")
    if native_edit.kind in (TetMeshEditKind.SPLIT, TetMeshEditKind.REFINE_CELL):
        replacement_kind_default: EntityLineageKind | None = (
            EntityLineageKind.REFINED_FROM
        )
    elif native_edit.kind is TetMeshEditKind.COLLAPSE:
        replacement_kind_default = EntityLineageKind.COARSENED_INTO
    elif native_edit.kind in (
        TetMeshEditKind.FLIP23,
        TetMeshEditKind.FLIP32,
        TetMeshEditKind.RECONNECT,
    ):
        replacement_kind_default = EntityLineageKind.SWAPPED_FROM
    elif native_edit.kind in (
        TetMeshEditKind.RELOCATE,
        TetMeshEditKind.RETRIANGULATE,
    ):
        replacement_kind_default = None
    else:
        raise ValueError(f"Unsupported native tetra edit kind {native_edit.kind!r}.")
    replacement_target = metadata.get(
        "replacement_target_cell_ids", metadata.get("replacement_target_ids")
    )
    replacement_target_ids = (
        np.asarray(replacement_target, dtype=np.int64)
        if replacement_target is not None
        else np.zeros((0,), dtype=np.int64)
    )
    target_parent_native_by_id = {
        int(target_id): parents
        for target_id, parents in zip(
            replacement_target_ids.tolist(),
            replacement_parent_rows,
            strict=True,
        )
    }
    changed_target_vertices = {
        int(state.vertex_target_ids[native_id])
        for native_id in changed_native_ids
        if native_id in state.vertex_target_ids
    }
    relocated_cells: set[int] = set()
    if changed_target_vertices:
        for block in source.blocks:
            vertices = np.asarray(block.vertices, dtype=np.int64)
            source_block_ids = np.asarray(block.global_ids, dtype=np.int64)
            for cell_id, cell in zip(source_block_ids.tolist(), vertices, strict=True):
                cell_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)[
                    cell
                ]
                if changed_target_vertices.intersection(
                    map(int, cell_vertex_ids.tolist())
                ):
                    relocated_cells.add(int(cell_id))
    cell_relations: set[tuple[tuple[int, ...], tuple[int, ...], EntityLineageKind]] = (
        set()
    )
    replacement_target_set = set(map(int, replacement_target_ids.tolist()))
    for native_id, target_id in cell_target_by_native.items():
        target_id = int(target_id)
        if target_id not in target_cell_id_set:
            continue
        if target_id in current_cell_ids:
            kind = (
                EntityLineageKind.RELOCATED
                if target_id in relocated_cells
                else EntityLineageKind.PRESERVED
            )
            cell_relations.add(((target_id,), (target_id,), kind))
            continue
        parents = target_parent_native_by_id.get(target_id)
        if native_edit.kind is TetMeshEditKind.RETRIANGULATE and not parents:
            continue
        if parents is None:
            parents = tuple(
                int(value)
                for value in np.asarray(
                    native_edit.removed_cell_ids
                    if native_edit.removed_cell_ids is not None
                    else (),
                    dtype=np.int64,
                ).tolist()
            )
        if not parents:
            raise _PeriodicCandidateRejected(
                "Periodic replacement cell has no exact current source ancestry."
            )
        row = (
            int(np.flatnonzero(replacement_target_ids == target_id)[0])
            if target_id in replacement_target_set
            else 0
        )
        kind_value = (
            replacement_kinds[row]
            if replacement_kinds is not None
            else replacement_kind_default
        )
        if kind_value is None:
            if native_edit.kind is TetMeshEditKind.RETRIANGULATE:
                continue
            raise ValueError(
                "Periodic replacement cells require an explicit lineage kind."
            )
        kind = EntityLineageKind(int(kind_value))
        for parent in parents:
            source_id = state.cell_target_ids.get(int(parent))
            if source_id is None:
                raise _PeriodicCandidateRejected(
                    "Periodic replacement cell ancestry is not bound to the current target."
                )
            cell_relations.add(((int(source_id),), (target_id,), kind))
    relations = (
        _relation_records(
            0,
            [row[0] for row in ordered_vertex_relations],
            [row[1] for row in ordered_vertex_relations],
            [row[2] for row in ordered_vertex_relations],
        ),
        _intermediate_relations(
            source,
            1,
            [tuple(int(v) for v in row if int(v) >= 0) for row in target_edge_keys],
            relational_stencils,
        ),
        _intermediate_relations(
            source,
            2,
            [tuple(int(v) for v in row if int(v) >= 0) for row in target_face_keys],
            relational_stencils,
        ),
        _relation_records(
            3,
            [
                row[0]
                for row in sorted(
                    cell_relations, key=lambda value: (value[0], value[1], int(value[2]))
                )
            ],
            [
                row[1]
                for row in sorted(
                    cell_relations, key=lambda value: (value[0], value[1], int(value[2]))
                )
            ],
            [
                row[2]
                for row in sorted(
                    cell_relations, key=lambda value: (value[0], value[1], int(value[2]))
                )
            ],
        ),
    )
    transition = CellTopologyEdit(
        coordinates=target_points,
        vertex_global_ids=target_vertex_ids,
        blocks=edit_blocks,
        stencil_sources=stencil_sources,
        stencil_weights=stencil_weights,
        stencil_valid=stencil_valid,
        relations=relations,
    )
    try:
        target = stage_periodic_topology_edit(source, transition)
    except (TypeError, ValueError) as error:
        raise _PeriodicCandidateRejected(str(error)) from error
    if target is None:
        raise _PeriodicCandidateRejected(
            "Periodic source edit did not produce a target quotient."
        )

    source_regions_by_native = {
        int(identifier): int(region)
        for identifier, region in zip(
            np.asarray(state.snapshot.cell_ids, dtype=np.int64).tolist(),
            np.asarray(state.snapshot.tetrahedron_regions, dtype=np.int32).tolist(),
            strict=True,
        )
    }
    source_native_by_target = {
        int(target_id): int(native_id)
        for native_id, target_id in state.cell_target_ids.items()
    }
    source_regions = np.asarray(
        [
            source_regions_by_native[source_native_by_target[int(identifier)]]
            for block in source.blocks
            for identifier in np.asarray(block.global_ids, dtype=np.int64).tolist()
        ],
        dtype=np.int32,
    )
    candidate_regions_by_native = {
        int(identifier): int(region)
        for identifier, region in zip(
            candidate.cell_ids.tolist(),
            candidate.tetrahedron_regions.tolist(),
            strict=True,
        )
    }
    target_native_by_target = {
        int(target_id): int(native_id)
        for native_id, target_id in cell_target_by_native.items()
    }
    target_regions = np.asarray(
        [
            candidate_regions_by_native[target_native_by_target[int(identifier)]]
            for block in target.blocks
            for identifier in np.asarray(block.global_ids, dtype=np.int64).tolist()
        ],
        dtype=np.int32,
    )
    proposal_metric_by_native: dict[int, np.ndarray] = {}
    for proposal in proposals:
        if isinstance(proposal, _SplitProposal):
            proposal_metric_by_native[int(proposal.native_vertex_id)] = np.asarray(
                proposal.metric, dtype=np.float64
            )
        elif (
            isinstance(proposal, Mapping)
            and "vertex" in proposal
            and "metric" in proposal
        ):
            proposal_metric_by_native[int(proposal["vertex"])] = np.asarray(
                proposal["metric"], dtype=np.float64
            )
        elif isinstance(proposal, Mapping) and proposal.get("survivor") is not None:
            # A moved collapse survivor carries the metric of its new carrier.
            proposal_metric_by_native[int(proposal["kept"])] = np.asarray(
                proposal["survivor"]["metric"], dtype=np.float64
            )
    periodic_metrics = {
        int(vertex_target_by_native[int(native_id)]): (
            proposal_metric_by_native[int(native_id)]
            if int(native_id) in proposal_metric_by_native
            else state.metric_by_vertex[int(native_id)]
        )
        for native_id in candidate.vertex_ids.tolist()
    }
    try:
        target_state = tet_mesh_periodic_state(target, target_regions)
        _validate_periodic_metric_values(target, periodic_metrics, target_state)
        periodic = tet_mesh_periodic_edit(
            source,
            target,
            transition,
            source_regions=source_regions,
            target_regions=target_regions,
            source_state=state.periodic_state,
            target_state=target_state,
        )
    except (TypeError, ValueError) as error:
        raise _PeriodicCandidateRejected(str(error)) from error
    return replace(native_edit, periodic=periodic), (target, target_state)


def _commit_periodic_target(
    state: _NativeState, staged: tuple[CellMesh, TetMeshPeriodicState] | None, /
) -> None:
    """Bind the staged periodic target and its quotient state after an applied edit."""
    if staged is not None:
        state.target_mesh, state.periodic_state = staged


def _apply_split_batch(
    state: _NativeState, split_parameters: dict[tuple[int, int], Fraction], /
) -> TetMeshEditStatus:
    edit, metadata = _split_edit(state, split_parameters)
    try:
        edit, staged = _prepare_periodic_native_edit(state, edit, metadata)
    except _PeriodicCandidateRejected:
        return TetMeshEditStatus.REJECTED_TOPOLOGY
    status = state.native.apply_edit(edit)
    if status is not TetMeshEditStatus.APPLIED:
        return status
    _commit_periodic_target(state, staged)
    state.snapshot = state.native.arrays()
    proposals: list[_SplitProposal] = metadata["proposals"]
    for proposal in proposals:
        native_id = proposal.native_vertex_id
        target_id = proposal.target_vertex_id
        state.vertex_target_ids[native_id] = target_id
        state.vertex_stencils[native_id] = proposal.stencil
        state.metric_by_vertex[native_id] = proposal.metric
    removed = set(metadata["removed_cells"])
    for native_cell_id in removed:
        state.cell_target_ids.pop(native_cell_id)
        state.cell_blocks.pop(native_cell_id)
        ancestry = state.cell_ancestry.pop(native_cell_id)
        state.changed_source_cell_ids.update(ancestry)
    replacement_ids = metadata["replacement_native_cell_ids"]
    target_ids = metadata["replacement_target_cell_ids"]
    parent_ids = metadata["replacement_parent_ids"]
    blocks = metadata["replacement_blocks"]
    ancestries = metadata["replacement_ancestry"]
    for native_id, target_id, parent_id, block, ancestry in zip(
        replacement_ids.tolist(),
        target_ids.tolist(),
        parent_ids.tolist(),
        blocks.tolist(),
        ancestries,
        strict=True,
    ):
        relation = {
            int(source_id): EntityLineageKind.REFINED_FROM for source_id in ancestry
        }
        state.cell_target_ids[int(native_id)] = int(target_id)
        state.cell_blocks[int(native_id)] = int(block)
        state.cell_ancestry[int(native_id)] = relation
        state.history_cells.extend(
            (source_id, int(target_id), EntityLineageKind.REFINED_FROM)
            for source_id in relation
        )
        state.changed_source_cell_ids.update(relation)
    state.next_native_vertex = int(metadata["next_native_vertex"])
    state.next_native_cell = int(metadata["next_native_cell"])
    state.next_native_facet = int(metadata["next_native_facet"])
    state.next_native_segment = int(metadata["next_native_segment"])
    state.next_target_vertex = int(metadata["next_target_vertex"])
    state.next_target_cell = int(metadata["next_target_cell"])
    _refresh_native_periodic_windings(state)
    return status


def _current_edge_controls(
    state: _NativeState, topology: _TetTopology, /
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], bool]]:
    tables = _snapshot_tables(state)
    shared = topology is tables.topology
    if shared and tables.edge_controls is not None:
        return tables.edge_controls
    snapshot = state.snapshot
    vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    segments = vertex_ids[np.asarray(snapshot.segments, dtype=np.int64)]
    segment_sources = np.asarray(snapshot.segment_sources, dtype=np.int64)
    protected_segments = np.asarray(snapshot.protected_segments, dtype=np.bool_)
    segment_data: dict[tuple[int, int], tuple[int, bool]] = {}
    for vertices, source_id, protected in zip(
        segments.tolist(),
        segment_sources.tolist(),
        protected_segments.tolist(),
        strict=True,
    ):
        key = _edge_key(vertices)
        segment_data[key] = (int(source_id), bool(protected))
    source_edge_key_by_id = {
        int(identifier): key for key, identifier in state.source_edge_ids.items()
    }
    target_ids = state.vertex_target_ids
    edge_classes = state.edge_classes
    protected_edge_ids = state.protected_edge_ids
    classes: dict[tuple[int, int], int] = {}
    protected: dict[tuple[int, int], bool] = {}
    for edge in topology.edge_cells:
        first, second = target_ids[edge[0]], target_ids[edge[1]]
        target_key = (first, second) if first <= second else (second, first)
        classes[edge] = int(edge_classes.get(target_key, 0))
        is_protected = target_key in protected_edge_ids
        segment = segment_data.get(edge)
        if segment is not None:
            source_id, segment_protected = segment
            is_protected |= segment_protected
            source_key = source_edge_key_by_id.get(source_id)
            if source_key is not None:
                classes[edge] = int(edge_classes.get(source_key, classes[edge]))
        protected[edge] = is_protected
    if shared:
        tables.edge_controls = (classes, protected)
    return classes, protected


def _metric_edge_table(
    state: _NativeState, topology: _TetTopology, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Snapshot points, vertex metrics and the metric lengths of the
    topology's edges in key order (shared, read-only, per snapshot)."""
    tables = _snapshot_tables(state)
    shared = topology is tables.topology
    if shared and tables.metric_table is not None:
        return tables.metric_table
    vertex_ids = np.asarray(state.snapshot.vertex_ids, dtype=np.int64)
    points = np.asarray(state.snapshot.points, dtype=np.float64)
    metrics = np.asarray(
        [state.metric_by_vertex[identifier] for identifier in vertex_ids.tolist()],
        dtype=np.float64,
    ).reshape((-1, 3, 3))
    if shared:
        edge_rows = _state_edge_rows(state)
    else:
        edge_rows = np.asarray(
            [
                [topology.vertex_row[first], topology.vertex_row[second]]
                for first, second in topology.edge_cells
            ],
            dtype=np.int64,
        ).reshape((-1, 2))
    lengths = metric_lengths(points, metrics, edge_rows)
    if not shared:
        return points, metrics, lengths
    metrics.setflags(write=False)
    lengths.setflags(write=False)
    tables.metric_table = (points, metrics, lengths)
    return tables.metric_table


def _periodic_entity_orbit_groups(
    state: _NativeState, dimension: int, /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    tables = _snapshot_tables(state)
    if tables.orbit_groups is None:
        tables.orbit_groups = {}
    groups = tables.orbit_groups.get(dimension)
    if groups is None:
        groups = _entity_orbit_groups(state, dimension)
        tables.orbit_groups[dimension] = groups
    return groups


def _entity_orbit_groups(
    state: _NativeState, dimension: int, /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    mesh = state.target_mesh
    if dimension < 0 or dimension > min(2, mesh.topological_dimension):
        raise ValueError("Metric entity dimension must identify a vertex, edge, or face.")
    if mesh.periodic_topology is None:
        if dimension == 0:
            native_keys = (
                (identifier,) for identifier in sorted(state.vertex_target_ids)
            )
        else:
            topology = _state_topology(state)
            if dimension == 1:
                native_keys = topology.edge_cells
            else:
                native_keys = topology.face_cells
        return tuple((key,) for key in sorted(native_keys))

    target_to_native = {
        int(target): int(native) for native, target in state.vertex_target_ids.items()
    }
    keys = np.asarray(entity_keys(mesh, dimension), dtype=np.int64)
    orbit_map = periodic_entity_orbit_map(mesh, dimension)
    quotient_rows = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
    if quotient_rows.shape != (keys.shape[0],):
        raise ValueError("Periodic quotient rows do not align with target entities.")
    groups: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for row in range(keys.shape[0]):
        key = keys[row]
        target_vertices = [
            int(key[column]) for column in range(key.shape[0]) if int(key[column]) >= 0
        ]
        try:
            ordered_native = sorted(
                target_to_native[identifier] for identifier in target_vertices
            )
        except KeyError as error:
            raise ValueError(
                "Periodic entity references an absent native vertex."
            ) from error
        if dimension == 0 and len(ordered_native) == 1:
            native_key = (ordered_native[0],)
        elif dimension == 1 and len(ordered_native) == 2:
            native_key = (ordered_native[0], ordered_native[1])
        elif dimension == 2 and len(ordered_native) == 3:
            native_key = (ordered_native[0], ordered_native[1], ordered_native[2])
        else:
            raise ValueError("Periodic native entity key has the wrong arity.")
        groups[int(quotient_rows[row])].append(native_key)
    result: list[tuple[tuple[int, ...], ...]] = []
    for quotient_row in sorted(groups):
        members = tuple(sorted(groups[quotient_row]))
        if len(set(members)) != len(members):
            raise ValueError("A periodic entity orbit repeats a lifted native entity.")
        result.append(members)
    return tuple(result)


def _periodic_edge_orbit_groups(
    state: _NativeState, /
) -> tuple[tuple[tuple[int, int], ...], ...]:
    tables = _snapshot_tables(state)
    if tables.edge_orbits is None:
        tables.edge_orbits = tuple(
            tuple(_edge_key(member) for member in orbit)
            for orbit in _periodic_entity_orbit_groups(state, 1)
        )
    return tables.edge_orbits


def _periodic_face_orbit_groups(
    state: _NativeState, /
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    tables = _snapshot_tables(state)
    if tables.face_orbits is None:
        tables.face_orbits = tuple(
            tuple(_face_key(member) for member in orbit)
            for orbit in _periodic_entity_orbit_groups(state, 2)
        )
    return tables.face_orbits


def _periodic_vertex_orbit_groups(state: _NativeState, /) -> tuple[tuple[int, ...], ...]:
    tables = _snapshot_tables(state)
    if tables.vertex_orbits is not None:
        return tables.vertex_orbits
    groups = []
    for orbit in _periodic_entity_orbit_groups(state, 0):
        if any(len(member) != 1 for member in orbit):
            raise ValueError("Periodic vertex orbit rows must contain one vertex each.")
        groups.append(tuple(member[0] for member in orbit))
    tables.vertex_orbits = tuple(groups)
    return tables.vertex_orbits


def _periodic_vertex_orbit_members(state: _NativeState, /) -> dict[int, tuple[int, ...]]:
    """Native vertex -> every lifted native member of its vertex orbit."""
    tables = _snapshot_tables(state)
    if tables.vertex_orbit_members is None:
        tables.vertex_orbit_members = {
            vertex: orbit
            for orbit in _periodic_vertex_orbit_groups(state)
            for vertex in orbit
        }
    return tables.vertex_orbit_members


def _periodic_vertex_quotient_ids(state: _NativeState, /) -> dict[int, int]:
    mesh = state.target_mesh
    if mesh.periodic_topology is None:
        return {int(target): int(target) for target in state.vertex_target_ids.values()}
    orbit_map = periodic_entity_orbit_map(mesh, 0)
    lifted = np.asarray(orbit_map.lifted_entity_ids, dtype=np.int64)
    quotient = np.asarray(orbit_map.quotient_entity_ids, dtype=np.int64)
    rows = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
    return {
        int(identifier): int(quotient[rows[row]])
        for row, identifier in enumerate(lifted.tolist())
    }


def _split_metric_candidates(
    state: _NativeState, blocked: set[tuple[int, int]], /
) -> list[tuple[tuple[int, int], ...]]:
    topology = _state_topology(state)
    _, _, lengths = _metric_edge_table(state, topology)
    _, protected = _current_edge_controls(state, topology)
    cell_ids = np.asarray(state.snapshot.cell_ids, dtype=np.int64)
    eligible: dict[tuple[int, int], tuple[float, frozenset[int]]] = {}
    for edge, length in zip(topology.edge_cells, lengths.tolist(), strict=True):
        if edge in blocked or protected[edge] or length <= _UPPER:
            continue
        eligible[edge] = (
            float(length),
            frozenset(int(cell_ids[row]) for row in topology.edge_cells[edge]),
        )
    candidates = []
    for orbit in _periodic_edge_orbit_groups(state):
        if not orbit or any(edge not in eligible for edge in orbit):
            continue
        cavities = [eligible[edge][1] for edge in orbit]
        cavity = frozenset().union(*cavities)
        if sum(map(len, cavities)) != len(cavity):
            continue
        representative_edge = min(
            tuple(sorted(state.vertex_target_ids[int(vertex)] for vertex in edge))
            for edge in orbit
        )
        priority = (-min(eligible[edge][0] for edge in orbit), representative_edge)
        candidates.append((priority, orbit, cavity))
    return _disjoint_candidates(candidates)


def _raise_fatal_edit_status(status: TetMeshEditStatus, operation: str, /) -> None:
    if status is TetMeshEditStatus.STALE_GENERATION:
        raise RuntimeError(f"Native tetra {operation} used a stale topology generation.")
    if status is TetMeshEditStatus.ALLOCATION_FAILURE:
        raise MemoryError(f"Native tetra {operation} allocation failed.")
    if status is TetMeshEditStatus.INDEX_OVERFLOW:
        raise OverflowError(
            f"Native tetra {operation} exhausted its exact identifier range."
        )


def _count_rejection(
    status: TetMeshEditStatus | None, counters: dict[str, int], count: int, /
) -> None:
    """Record one refused candidate; a refusal blocks its candidate, never the run.

    ``None`` is a host-side refusal: the candidate has no exact binary64
    realization of its source ancestry.
    """
    if status is TetMeshEditStatus.UNCERTAIN_PREDICATE:
        counters["rejected_uncertain"] += count
    elif status is TetMeshEditStatus.REJECTED_PROTECTED:
        counters["rejected_protected"] += count
    elif status is TetMeshEditStatus.REJECTED_TOPOLOGY:
        counters["rejected_topology"] += count
    else:
        counters["rejected_invalid"] += count


def _try_split_batch(
    state: _NativeState,
    edge_groups: list[tuple[tuple[int, int], ...]],
    counters: dict[str, int],
    blocked: set[tuple[int, int]],
    /,
) -> int:
    """Apply disjoint split orbits; refused orbits are blocked and counted.

    A split replaces its long edge by halves and medians to the edge ring; it
    is proposed only when that strictly decreases the violation multiset
    (``_objective_decreases``) and its cells clear the determinant floor.
    """
    topology = _state_topology(state)
    points, metrics, lengths = _metric_edge_table(state, topology)
    length_by_edge = dict(zip(topology.edge_cells, lengths.tolist(), strict=True))
    constrained = _constrained_edges(state, topology)
    parameters: dict[tuple[int, int], Fraction] = {}
    representable: list[tuple[tuple[int, int], ...]] = []
    for group in edge_groups:
        values = _exact_split_parameters(
            state, topology, points, metrics, group, constrained
        )
        if values is None:
            blocked.update(group)
            state.unrepresentable_edges.update(group)
            _count_rejection(None, counters, 1)
            continue
        removed = np.asarray([length_by_edge[edge] for edge in group])
        effects = [
            _split_effects(state, topology, points, metrics, edge, value)
            for edge, value in zip(group, values, strict=True)
        ]
        added = np.concatenate([lengths for lengths, _ in effects])
        if not all(clears for _, clears in effects) or not _objective_decreases(
            removed, added, False
        ):
            blocked.update(group)
            continue
        parameters.update(zip(group, values, strict=True))
        representable.append(group)
    return _apply_split_groups(state, representable, parameters, counters, blocked)


def _split_effects(
    state: _NativeState,
    topology: _TetTopology,
    points: np.ndarray,
    metrics: np.ndarray,
    edge: tuple[int, int],
    parameter: Fraction,
    /,
) -> tuple[np.ndarray, bool]:
    """Metric lengths of the halves and ring medians a split would create, and
    whether its cells clear the determinant floor."""
    stencil = _split_stencil(
        state.vertex_stencils[edge[0]], state.vertex_stencils[edge[1]], parameter
    )
    point = np.asarray([float(value) for value in _stencil_image(state, stencil)])
    metric = _source_metric_from_stencil(
        state.source_metric,
        state.source_vertex_rows,
        stencil,
        source_metric_log=state.source_metric_log,
    )
    others = sorted(
        {
            int(vertex)
            for row in topology.edge_cells[edge]
            for vertex in topology.cell_vertex_ids[row]
        }
    )
    rows = np.asarray([topology.vertex_row[vertex] for vertex in others], dtype=np.int64)
    extended_points = np.concatenate((points[rows], point[None]), axis=0)
    extended_metrics = np.concatenate((metrics[rows], metric[None]), axis=0)
    middle = len(others)
    pairs = np.asarray([[index, middle] for index in range(middle)], dtype=np.int64)
    local = {vertex: index for index, vertex in enumerate(others)}
    created = np.asarray(
        [
            [
                middle if int(vertex) == endpoint else local[int(vertex)]
                for vertex in topology.cell_vertex_ids[row]
            ]
            for row in topology.edge_cells[edge]
            for endpoint in edge
        ],
        dtype=np.int64,
    ).reshape((-1, 4))
    return (
        metric_lengths(extended_points, extended_metrics, pairs),
        _clears_determinant_floor(extended_points[created]),
    )


def _apply_split_groups(
    state: _NativeState,
    edge_groups: list[tuple[tuple[int, int], ...]],
    parameters: dict[tuple[int, int], Fraction],
    counters: dict[str, int],
    blocked: set[tuple[int, int]],
    /,
) -> int:
    if not edge_groups:
        return 0
    try:
        status: TetMeshEditStatus | None = _apply_split_batch(
            state,
            {edge: parameters[edge] for group in edge_groups for edge in group},
        )
    except FloatingPointError:
        status = None
    if status is TetMeshEditStatus.APPLIED:
        count = sum(len(group) for group in edge_groups)
        counters["splits"] += count
        return count
    if status is not None:
        _raise_fatal_edit_status(status, "split")
    if len(edge_groups) > 1:
        middle = len(edge_groups) // 2
        return _apply_split_groups(
            state, edge_groups[:middle], parameters, counters, blocked
        ) + _apply_split_groups(
            state, edge_groups[middle:], parameters, counters, blocked
        )
    blocked.update(edge_groups[0])
    _count_rejection(status, counters, 1)
    return 0


def _split_phase(
    state: _NativeState, counters: dict[str, int], /, *, one_batch: bool = False
) -> int:
    applied = 0
    blocked: set[tuple[int, int]] = set()
    while True:
        candidates = _split_metric_candidates(state, blocked)
        if not candidates:
            return applied
        changed = _try_split_batch(state, candidates, counters, blocked)
        applied += changed
        if changed:
            if one_batch:
                return applied
            blocked.clear()


class _BoundaryRecords(NamedTuple):
    """Current constrained facets and segments keyed by native vertex IDs."""

    # key -> (facet ID, group row, ordered vertices, orientation, protected)
    facets: dict[tuple[int, int, int], tuple[int, int, tuple[int, int, int], int, bool]]
    # key -> (segment ID, group row, ordered vertices, protected)
    segments: dict[tuple[int, int], tuple[int, int, tuple[int, int], bool]]
    vertex_facets: dict[int, tuple[tuple[int, int, int], ...]]
    vertex_segments: dict[int, tuple[tuple[int, int], ...]]


def _boundary_records(state: _NativeState, /) -> _BoundaryRecords:
    tables = _snapshot_tables(state)
    if tables.boundary_records is None:
        tables.boundary_records = _snapshot_boundary_records(
            state.snapshot,
            state.source_face_group_rows,
            state.source_segment_group_rows,
        )
    return tables.boundary_records


def _snapshot_boundary_records(
    snapshot: TetMeshArrays,
    face_group_rows: Mapping[int, int],
    segment_group_rows: Mapping[int, int],
    /,
) -> _BoundaryRecords:
    vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    facets: dict[
        tuple[int, int, int], tuple[int, int, tuple[int, int, int], int, bool]
    ] = {}
    vertex_facets: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for row, identifier, source_id, orientation, protected in zip(
        vertex_ids[np.asarray(snapshot.faces, dtype=np.int64)].reshape((-1, 3)).tolist(),
        np.asarray(snapshot.facet_ids, dtype=np.int64).tolist(),
        np.asarray(snapshot.face_sources, dtype=np.int64).tolist(),
        np.asarray(snapshot.facet_orientations, dtype=np.int8).tolist(),
        np.asarray(snapshot.protected_facets, dtype=np.bool_).tolist(),
        strict=True,
    ):
        group = face_group_rows.get(int(source_id))
        if group is None:
            raise ValueError("A constrained native facet lost its source group.")
        ordered = (int(row[0]), int(row[1]), int(row[2]))
        key = _face_key(ordered)
        facets[key] = (int(identifier), group, ordered, int(orientation), bool(protected))
        for vertex in ordered:
            vertex_facets[vertex].append(key)
    segments: dict[tuple[int, int], tuple[int, int, tuple[int, int], bool]] = {}
    vertex_segments: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, identifier, source_id, protected in zip(
        vertex_ids[np.asarray(snapshot.segments, dtype=np.int64)]
        .reshape((-1, 2))
        .tolist(),
        np.asarray(snapshot.segment_ids, dtype=np.int64).tolist(),
        np.asarray(snapshot.segment_sources, dtype=np.int64).tolist(),
        np.asarray(snapshot.protected_segments, dtype=np.bool_).tolist(),
        strict=True,
    ):
        group = segment_group_rows.get(int(source_id))
        if group is None:
            raise ValueError("A constrained native segment lost its source group.")
        ordered_segment = (int(row[0]), int(row[1]))
        key = _edge_key(ordered_segment)
        segments[key] = (int(identifier), group, ordered_segment, bool(protected))
        for vertex in ordered_segment:
            vertex_segments[vertex].append(key)
    return _BoundaryRecords(
        facets,
        segments,
        {vertex: tuple(keys) for vertex, keys in vertex_facets.items()},
        {vertex: tuple(keys) for vertex, keys in vertex_segments.items()},
    )


def _implicit_only_vertex(
    state: _NativeState, records: _BoundaryRecords, vertex: int, /
) -> bool:
    """Whether only interval-admitted implicit facets carry this unprotected vertex."""
    facets = records.vertex_facets.get(vertex, ())
    return (
        bool(facets)
        and not records.vertex_segments.get(vertex)
        and all(
            records.facets[key][1] in state.implicit_groups and not records.facets[key][4]
            for key in facets
        )
    )


def _carrier_entity(
    state: _NativeState, records: _BoundaryRecords, vertex: int, /
) -> frozenset[int] | None:
    """Source vertices spanning the smallest source entity under a boundary vertex.

    ``None`` marks an interior vertex. A boundary vertex lies on the planar
    source facet (its triangles' vertices), on the curve shared by two facets
    or on a source segment (their common vertices) carrying every incident
    constrained facet and segment; with fewer than two it is pinned.
    """
    entity: frozenset[int] | None = None
    for key in records.vertex_facets.get(vertex, ()):
        carried = state.source_face_vertices[records.facets[key][1]]
        entity = carried if entity is None else entity & carried
    for key in records.vertex_segments.get(vertex, ()):
        carried = state.source_segment_vertices[records.segments[key][1]]
        entity = carried if entity is None else entity & carried
    return entity


def _carrier_simplices(
    state: _NativeState, records: _BoundaryRecords, vertex: int, /
) -> tuple[tuple[int, ...], ...] | None:
    """Source simplices a boundary vertex may move in (``None``: interior).

    A vertex on one source segment moves along it; on one planar facet, within
    its source triangles; on the curve shared by two facets, along the source
    edges common to both; otherwise (a corner) nowhere (empty).
    """
    facets = {records.facets[key][1] for key in records.vertex_facets.get(vertex, ())}
    segments = {
        records.segments[key][1] for key in records.vertex_segments.get(vertex, ())
    }
    if not facets and not segments:
        return None
    if segments:
        if len(segments) != 1:
            return ()
        return (tuple(sorted(state.source_segment_vertices[next(iter(segments))])),)
    triangles = [state.source_group_triangles[group] for group in sorted(facets)]
    if len(triangles) == 1:
        return triangles[0]
    if len(triangles) != 2:
        return ()

    def edges(group: tuple[tuple[int, int, int], ...]) -> set[tuple[int, int]]:
        return {
            _edge_key((triangle[corner], triangle[(corner + 1) % 3]))
            for triangle in group
            for corner in range(3)
        }

    return tuple(sorted(edges(triangles[0]) & edges(triangles[1])))


def _collapse_proposal(
    state: _NativeState,
    topology: _TetTopology,
    records: _BoundaryRecords,
    protected: dict[tuple[int, int], bool],
    edge: tuple[int, int],
    /,
) -> dict[str, Any] | None:
    """Best admissible direction to collapse one unclassified short edge.

    Interior and boundary vertices collapse alike: a boundary vertex collapses
    only into a vertex on the source facet or edge carrying it (its stencil
    support lies in that entity), retiring its constrained facets and segments
    for their images under the collapse. When collapsing in place would create
    a long edge, the survivor may also move toward the edge's metric midpoint
    (``_survivor_moves``), replacing both stars. The collapse must strictly
    decrease the termination objective (``_objective_decreases``) over the
    spokes of both endpoints.
    """
    proposals = [
        proposal
        for removed, kept in (edge, edge[::-1])
        if (
            proposal := _directed_collapse(
                state, topology, records, protected, removed, kept
            )
        )
        is not None
    ]
    if not proposals:
        return None
    return min(
        proposals,
        key=lambda row: (
            row["added_violations"],
            -int(state.vertex_target_ids[int(row["removed"])]),
        ),
    )


def _source_permits_collapse(
    state: _NativeState,
    topology: _TetTopology,
    records: _BoundaryRecords,
    removed: int,
    kept: int,
    /,
) -> bool:
    """Whether the source boundary lets ``removed`` collapse into ``kept``.

    The removed vertex is neither fixed nor protected and carries no protected
    facet or segment. A PWL boundary vertex must collapse into a vertex on its
    source facet or edge (and a constrained segment only along itself). Two
    implicit-only endpoints both belong solely to unprotected facets of the
    same bound implicit source, with no source segment: the source stratum is
    the zero set, not the old PWL facet's interpolation support. The native
    owner still certifies every replacement facet against that source's tube
    and embedding before committing.
    On a periodic lift a seam vertex is carried by its planar cell-face cap,
    so it collapses only within the cap (or along the edge two caps share);
    the orbit-locked candidate applies the same collapse to every
    lifted copy (``_collapse_metric_candidates``). The native mesh refuses a
    protected survivor; a periodic lift, whose fixed lattice contour borders
    many short interior edges, rules that out here rather than paying for a
    refused staged quotient edit (off the lift the proposal set is unchanged).
    """
    protected_vertices = np.asarray(state.snapshot.protected_vertices)
    return _collapse_admitted(
        state,
        records,
        removed,
        kept,
        removed_protected=bool(protected_vertices[topology.vertex_row[removed]]),
        kept_protected=bool(protected_vertices[topology.vertex_row[kept]]),
        kept_support=frozenset(state.vertex_stencils[kept]),
    )


def _collapse_admitted(
    state: _NativeState,
    records: _BoundaryRecords,
    removed: int,
    kept: int,
    /,
    *,
    removed_protected: bool,
    kept_protected: bool,
    kept_support: frozenset[int],
) -> bool:
    """``_source_permits_collapse`` given the endpoints' protection flags and
    the source vertex IDs of the survivor's stencil."""
    if removed in state.fixed_vertices or removed_protected:
        return False
    if state.target_mesh.periodic_topology is not None and kept_protected:
        return False
    implicit_edge = (
        bool(state.implicit_groups)
        and _implicit_only_vertex(state, records, removed)
        and _implicit_only_vertex(state, records, kept)
    )
    entity = None if implicit_edge else _carrier_entity(state, records, removed)
    if entity is not None and (len(entity) < 2 or not kept_support <= entity):
        return False
    if records.vertex_segments.get(removed) and (
        _edge_key((removed, kept)) not in records.segments
    ):
        return False
    return not any(
        records.facets[key][4] for key in records.vertex_facets.get(removed, ())
    ) and not any(
        records.segments[key][3] for key in records.vertex_segments.get(removed, ())
    )


def _moved_spokes_splittable(
    state: _NativeState,
    records: _BoundaryRecords,
    removed: int,
    kept: int,
    local_vertices: tuple[int, ...],
    move: dict[str, Any],
    local_points: np.ndarray,
    local_metrics: np.ndarray,
    spokes: np.ndarray,
    lengths: np.ndarray,
    new_facets: list[tuple[tuple[int, int, int], int, int]],
    new_segments: list[tuple[tuple[int, int], int]],
    /,
) -> bool:
    """Whether every long spoke of a moved collapse survivor keeps an exact split.

    Moving the survivor changes its point and source stencil, and with them the
    binary64 realization of every dyadic point on its spokes. A move that
    leaves a long spoke without a representable split
    (``_dyadic_split_parameters``) trades a long edge the next split batch can
    remove for one no split reaches, so the collapse must stay in place.
    ``local_points``/``local_metrics`` are the star-local arrays (row 0 the
    survivor before the move); ``spokes`` and ``lengths`` are the measured
    survivor spokes and their lengths after the moved collapse. The constrained
    support is the survivor's after the collapse: its surviving segments and
    non-implicit facets plus the images of the removed vertex's.
    """
    long_rows = np.flatnonzero(lengths > _UPPER)
    if long_rows.size == 0:
        return True
    points = local_points.copy()
    points[0] = move["point"]
    metrics = local_metrics.copy()
    metrics[0] = move["metric"]
    implicit = state.implicit_groups
    constrained = {
        key for key in records.vertex_segments.get(kept, ()) if removed not in key
    }
    constrained.update(_edge_key(segment) for segment, _ in new_segments)
    faces = [
        key
        for key in records.vertex_facets.get(kept, ())
        if removed not in key and records.facets[key][1] not in implicit
    ]
    faces.extend(
        _face_key(face) for face, group, _ in new_facets if group not in implicit
    )
    for first, second, third in faces:
        constrained.update(((first, second), (first, third), (second, third)))
    for row in long_rows.tolist():
        other_index = int(spokes[row, 1])
        other = local_vertices[other_index]
        ends = (0, other_index) if kept < other else (other_index, 0)
        stencils = [
            move["stencil"] if index == 0 else state.vertex_stencils[other]
            for index in ends
        ]
        member = (
            points[ends[0]],
            points[ends[1]],
            metrics[ends[0]],
            metrics[ends[1]],
            stencils[0],
            stencils[1],
            _edge_key((kept, other)) in constrained,
        )
        if _dyadic_split_parameters(state, [member]) is None:
            return False
    return True


def _directed_collapse(
    state: _NativeState,
    topology: _TetTopology,
    records: _BoundaryRecords,
    protected: dict[tuple[int, int], bool],
    removed: int,
    kept: int,
    /,
) -> dict[str, Any] | None:
    snapshot = state.snapshot
    if not _source_permits_collapse(state, topology, records, removed, kept):
        return None
    edge = _edge_key((removed, kept))
    removed_rows = topology.vertex_cells[removed]
    kept_rows = topology.vertex_cells[kept]
    cavity_rows = tuple(sorted(set(removed_rows) | set(kept_rows)))
    cell_ids = np.asarray(snapshot.cell_ids, dtype=np.int64)
    regions = np.asarray(snapshot.tetrahedron_regions, dtype=np.int64)
    cavity_regions = {int(regions[row]) for row in cavity_rows}
    cavity_blocks = {state.cell_blocks[int(cell_ids[row])] for row in cavity_rows}
    if len(cavity_regions) != 1 or len(cavity_blocks) != 1:
        return None
    edge_link = {
        int(vertex)
        for row in topology.edge_cells[edge]
        for vertex in topology.cell_vertex_ids[row]
        if int(vertex) not in edge
    }
    removed_link = {
        int(vertex)
        for row in removed_rows
        for vertex in topology.cell_vertex_ids[row]
        if int(vertex) != removed
    }
    kept_link = {
        int(vertex)
        for row in kept_rows
        for vertex in topology.cell_vertex_ids[row]
        if int(vertex) != kept
    }
    if removed_link & kept_link != edge_link:
        return None

    def image(vertex: int) -> int:
        return kept if vertex == removed else vertex

    # Constrained facets and segments of the removed vertex map to their images.
    removed_facets: list[int] = []
    new_facets: list[tuple[tuple[int, int, int], int, int]] = []
    for key in records.vertex_facets.get(removed, ()):
        facet_id, group, ordered, orientation, is_protected = records.facets[key]
        if is_protected:
            return None
        removed_facets.append(facet_id)
        if kept in key:
            continue
        mapped = (image(ordered[0]), image(ordered[1]), image(ordered[2]))
        if _face_key(mapped) in records.facets:
            return None
        new_facets.append((mapped, group, orientation))
    removed_segments: list[int] = []
    new_segments: list[tuple[tuple[int, int], int]] = []
    for key in records.vertex_segments.get(removed, ()):
        segment_id, group, ordered_segment, is_protected = records.segments[key]
        if is_protected:
            return None
        removed_segments.append(segment_id)
        if kept in key:
            continue
        mapped_segment = (image(ordered_segment[0]), image(ordered_segment[1]))
        if _edge_key(mapped_segment) in records.segments:
            return None
        new_segments.append((mapped_segment, group))

    vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    # Star-local arrays over both stars: row 0 is the survivor, row 1 the
    # removed vertex.
    local_vertices = (kept, removed, *sorted((removed_link | kept_link) - set(edge)))
    local_index = {vertex: index for index, vertex in enumerate(local_vertices)}
    global_rows = np.asarray(
        [topology.vertex_row[vertex] for vertex in local_vertices], dtype=np.int64
    )
    local_points = np.asarray(snapshot.points, dtype=np.float64)[global_rows]
    local_metrics = np.asarray(
        [state.metric_by_vertex[int(vertex_ids[row])] for row in global_rows.tolist()],
        dtype=np.float64,
    )
    removed_cells: list[tuple[int, int, int, int]] = []
    removed_parents: list[int] = []
    for row in removed_rows:
        old = topology.cell_vertex_ids[row]
        if kept in old:
            continue
        removed_cells.append(_tetra_key(local_index[image(int(value))] for value in old))
        removed_parents.append(int(cell_ids[row]))
    if not removed_cells:
        return None
    kept_only_rows = [
        row for row in kept_rows if removed not in topology.cell_vertex_ids[row]
    ]
    kept_cells = [
        _tetra_key(local_index[int(value)] for value in topology.cell_vertex_ids[row])
        for row in kept_only_rows
    ]
    # Termination objective over every measured edge of both stars: before,
    # the spokes of the two endpoints; after, the survivor's spokes. The set is
    # the same for both directions and every survivor position, so proposals
    # compare by their after-violations.
    before_edges = sorted(
        {
            _edge_key((center, other))
            for center, link in ((removed, removed_link), (kept, kept_link))
            for other in link
            if not protected[_edge_key((center, other))]
        }
    )
    before_lengths = metric_lengths(
        local_points,
        local_metrics,
        np.asarray(
            [[local_index[a], local_index[b]] for a, b in before_edges], dtype=np.int64
        ).reshape((-1, 2)),
    )
    after_spokes = np.asarray(
        [
            [0, local_index[other]]
            for other in local_vertices[2:]
            if not protected.get(_edge_key((kept, other)), False)
        ],
        dtype=np.int64,
    ).reshape((-1, 2))
    cavity_cells = np.asarray(
        [
            [local_index[int(value)] for value in topology.cell_vertex_ids[row]]
            for row in cavity_rows
        ],
        dtype=np.int64,
    ).reshape((-1, 4))
    star = MetricCollapseStar(
        local_points,
        local_metrics,
        after_spokes[:, 1],
        before_lengths,
        np.asarray(removed_cells, dtype=np.int64).reshape((-1, 4)),
        _DETERMINANT_FLOOR,
    )
    best = _collapse_position(star, local_points[0], local_metrics[0])
    survivor: dict[str, Any] | None = None
    if best is None or best["violations"][0]:
        # Collapsing in place creates a long edge (or is refused): move the
        # survivor to a quantized source point on its carrier that shortens
        # its longest spoke, replacing both stars. A move must keep every long
        # spoke of the moved survivor exactly splittable; otherwise the
        # collapse stays in place.
        moved_star = star._replace(
            cells=np.asarray(removed_cells + kept_cells, dtype=np.int64).reshape((-1, 4))
        )
        for move, moved in _survivor_moves(state, records, topology, kept, moved_star):
            if (
                best is None
                or (moved["violations"], -moved["quality"])
                < (best["violations"], -best["quality"])
            ) and _moved_spokes_splittable(
                state,
                records,
                removed,
                kept,
                local_vertices,
                move,
                local_points,
                local_metrics,
                after_spokes,
                moved["lengths"],
                new_facets,
                new_segments,
            ):
                best, survivor = moved, move
                if not best["violations"][0]:
                    break
    if best is None:
        return None
    before_quality = metric_tetra_quality(local_points, local_metrics, cavity_cells)
    parents = removed_parents + (
        [int(cell_ids[row]) for row in kept_only_rows] if survivor is not None else []
    )
    replaced_rows = cavity_rows if survivor is not None else tuple(removed_rows)
    oriented_native = np.asarray(local_vertices, dtype=np.int64)[best["rows"]]
    return {
        "edge": edge,
        "removed": removed,
        "kept": kept,
        "survivor": survivor,
        "removed_cell_ids": tuple(sorted(int(cell_ids[row]) for row in replaced_rows)),
        "replacement_cell_vertices": oriented_native,
        "replacement_parent_cell_ids": tuple(parents),
        "replacement_regions": np.full(
            (oriented_native.shape[0],), next(iter(cavity_regions)), dtype=np.int32
        ),
        "block": next(iter(cavity_blocks)),
        "cavity": frozenset(int(cell_ids[row]) for row in cavity_rows),
        "quality_gain": best["quality"] - float(np.min(before_quality)),
        "added_violations": best["violations"],
        "removed_facet_ids": tuple(removed_facets),
        "replacement_facets": tuple(new_facets),
        "removed_segment_ids": tuple(removed_segments),
        "replacement_segments": tuple(new_segments),
    }


class _SourceBank(NamedTuple):
    """The native exact source of one adaptation.

    Its points are in ascending source vertex ID, so the support of a native
    carrier (ascending rows) is a stencil in ascending ID; its cells are in
    ascending cell ID, the order in which a point is located.
    """

    source: MetricSource
    identifiers: tuple[int, ...]
    """Source vertex ID of each bank row."""
    rows: dict[int, int]
    """Bank row of each source vertex ID."""
    cells: dict[int, int]
    """Bank cell row of each source cell ID."""


def _source_bank(state: _NativeState, /) -> _SourceBank:
    bank = state.source_bank
    if bank is None:
        identifiers = tuple(sorted(state.source_vertex_rows))
        rows = {identifier: row for row, identifier in enumerate(identifiers)}
        order = [state.source_vertex_rows[identifier] for identifier in identifiers]
        bank = _SourceBank(
            MetricSource(
                state.source_points[order],
                [state.source_exact_points[row] for row in order],
                np.asarray(
                    [
                        [rows[int(vertex)] for vertex in record[1]]
                        for record in state.source_records.records
                    ],
                    dtype=np.int64,
                ).reshape((-1, 4)),
                state.source_metric[order],
                state.source_metric_log[order],
            ),
            identifiers,
            rows,
            {
                int(record[0]): row
                for row, record in enumerate(state.source_records.records)
            },
        )
        state.source_bank = bank
    return bank


def _position_evidence(
    star: MetricCollapseStar, quality: float, lengths: np.ndarray, /
) -> dict[str, Any]:
    return {
        "rows": star.cells,
        "quality": quality,
        "violations": _band_violations(lengths),
        "lengths": lengths,
    }


def _report_position(
    star: MetricCollapseStar, point: np.ndarray, lengths: np.ndarray, /
) -> NoReturn:
    """Raise the error of a survivor position the kernels give no verdict for.

    A length before the collapse or a spoke at the position was not measured,
    or a cell coordinate lies outside the exact predicate domain: the owners
    of those contracts report them, in the order the position is evaluated.
    """
    _band_violations(star.before_lengths)
    _band_violations(lengths)
    moved = np.array(star.points, dtype=np.float64)
    moved[0] = point
    _tetra_orientation(moved, star.cells, PredicateMode.EXACT)
    raise RuntimeError("A survivor position without a verdict raised no error.")


def _collapse_position(
    star: MetricCollapseStar, point: np.ndarray, metric: np.ndarray, /
) -> dict[str, Any] | None:
    """Evidence of the survivor of ``star`` at ``point``; ``None`` if not admitted.

    A position is admitted when it has no neighbor at it, strictly decreases
    the band violations of the measured edges, keeps every cell exactly
    positive above the determinant floor and gives every cell a positive
    quality (``metric_collapse_positions``).
    """
    verdicts, qualities, lengths = metric_collapse_positions(
        star, point[None, :], metric[None, :, :]
    )
    match MetricPositionVerdict(int(verdicts[0])):
        case MetricPositionVerdict.ADMITTED:
            return _position_evidence(star, float(qualities[0]), lengths[0])
        case (
            MetricPositionVerdict.COINCIDENT
            | MetricPositionVerdict.NOT_DECREASING
            | MetricPositionVerdict.NOT_ORIENTED
            | MetricPositionVerdict.NO_QUALITY
        ):
            return None
        case MetricPositionVerdict.UNMEASURED | MetricPositionVerdict.OUTSIDE_DOMAIN:
            _report_position(star, point, lengths[0])
        case _ as unreachable:
            assert_never(unreachable)


def _survivor_moves(
    state: _NativeState,
    records: _BoundaryRecords,
    topology: _TetTopology,
    kept: int,
    star: MetricCollapseStar,
    /,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Quantized source points a collapse survivor may move to, with their evidence.

    ``star`` holds the survivor, the removed vertex, then the merged link, and
    the cells of both stars after the collapse. Goals are the center of the
    smallest ball (in the endpoints' mean metric) holding the merged link,
    which minimizes the longest survivor spoke, the metric midpoint of the
    collapsed edge and the point halfway to it; each is projected onto the
    survivor's carrier and quantized to exact source carriers: an interior
    survivor moves in its source cell, a boundary survivor within its source
    facet or curve (on a declared plane when its carrier rounds). Each move
    is admitted as a position of the star (``_collapse_position``) by one
    native search, which ends at the first move that leaves no long spoke.
    Survivors that are fixed, protected or pinned at a source corner do not
    move. On a periodic lift only a survivor whose vertex orbit has a single
    lifted member moves: its star is the whole quotient star, so no deck
    partner has to follow it; a seam survivor stays in place.
    """
    row = topology.vertex_row[kept]
    snapshot = state.snapshot
    if (
        kept in state.fixed_vertices
        or bool(np.asarray(snapshot.protected_vertices)[row])
        or (
            state.target_mesh.periodic_topology is not None
            and len(_periodic_vertex_orbit_members(state)[kept]) != 1
        )
    ):
        return
    implicit_only = _implicit_only_vertex(state, records, kept)
    simplices = None if implicit_only else _carrier_simplices(state, records, kept)
    if (
        simplices is None
        and not implicit_only
        and int(np.asarray(snapshot.vertex_dimensions)[row]) != 3
    ):
        return
    if simplices is not None and not simplices:
        return
    bank = _source_bank(state)
    # The carrier simplices are coplanar (one facet) or collinear (one curve
    # or segment): the first spans their affine hull.
    entries, end = bank.source.survivor_moves(
        star, _bank_simplices(state, bank, simplices), implicit_only
    )
    referred = end in (MetricSearchEnd.UNMEASURED, MetricSearchEnd.OUTSIDE_DOMAIN)
    for index, entry in enumerate(entries):
        if referred and index == len(entries) - 1:
            _report_position(star, entry.point, entry.lengths)
        yield (
            {
                "point": entry.point,
                "metric": entry.metric,
                "stencil": {
                    bank.identifiers[source_row]: Fraction(weight, 1 << entry.bits)
                    for source_row, weight in zip(entry.rows, entry.weights, strict=True)
                },
            },
            _position_evidence(star, entry.quality, entry.lengths),
        )
    if end is MetricSearchEnd.GOAL_NOT_FINITE:
        raise FloatingPointError("A collapse survivor goal is not a finite point.")


def _edge_set_lengths(
    points: np.ndarray,
    metrics: np.ndarray,
    topology: _TetTopology,
    edges: list[tuple[int, int]],
    /,
) -> np.ndarray:
    rows = np.asarray(
        [[topology.vertex_row[a], topology.vertex_row[b]] for a, b in edges],
        dtype=np.int64,
    ).reshape((-1, 2))
    return metric_lengths(points, metrics, rows)


def _facet_group_map(
    state: _NativeState, snapshot: TetMeshArrays, /
) -> dict[tuple[int, int, int], int]:
    tables = _snapshot_tables(state)
    shared = snapshot is tables.snapshot
    if shared and tables.facet_groups is not None:
        return tables.facet_groups
    vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    rows = vertex_ids[np.asarray(snapshot.faces, dtype=np.int64)]
    sources = np.asarray(snapshot.face_sources, dtype=np.int64)
    result: dict[tuple[int, int, int], int] = {}
    for vertices, source_id in zip(rows.tolist(), sources.tolist(), strict=True):
        group = state.source_face_group_rows.get(int(source_id))
        if group is None:
            raise ValueError("A constrained native facet lost its source group.")
        result[_face_key(vertices)] = group
    if shared:
        tables.facet_groups = result
    return result


def _cell_ancestry_banks(
    parent_native_ids: list[int], /
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(len(parent_native_ids) + 1, dtype=np.int64)
    return offsets, np.asarray(parent_native_ids, dtype=np.uint64)


def _collapse_edit(
    state: _NativeState, proposals: list[dict[str, Any]], /
) -> tuple[TetMeshEdit, dict[str, Any]]:
    if not proposals:
        raise ValueError("A collapse edit requires at least one complete cavity.")
    removed_vertices = sorted(int(row["removed"]) for row in proposals)
    removed_cells = sorted(
        {int(identifier) for row in proposals for identifier in row["removed_cell_ids"]}
    )
    replacement_records: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    for proposal in proposals:
        for parent, vertices, region in zip(
            proposal["replacement_parent_cell_ids"],
            proposal["replacement_cell_vertices"],
            proposal["replacement_regions"],
            strict=True,
        ):
            replacement_records.append(
                (
                    int(parent),
                    _tetra_key(vertices),
                    int(region),
                    int(proposal["block"]),
                )
            )
    replacement_records.sort(key=lambda row: (row[0], row[1]))
    replacement_count = len(replacement_records)
    replacement_native_ids = _fresh_ids(
        state.next_native_cell, replacement_count, "native cell IDs"
    )
    replacement_target_ids = _fresh_ids(
        state.next_target_cell, replacement_count, "target cell IDs"
    )
    parent_native_ids = [row[0] for row in replacement_records]
    ancestry_offsets, ancestry_ids = _cell_ancestry_banks(parent_native_ids)
    boundary, new_facet_groups, next_facet, next_segment = _boundary_edit_fields(
        state,
        [facet for row in proposals for facet in row["removed_facet_ids"]],
        [facet for row in proposals for facet in row["replacement_facets"]],
        [segment for row in proposals for segment in row["removed_segment_ids"]],
        [segment for row in proposals for segment in row["replacement_segments"]],
    )
    facet_group_by_key = _facet_group_map(state, state.snapshot) | new_facet_groups
    replacement_facet_sources = np.full((replacement_count, 4), -1, dtype=np.int32)
    for row, record in enumerate(replacement_records):
        vertices = record[1]
        for local_face, face in enumerate(_TETRA_FACES):
            replacement_facet_sources[row, local_face] = facet_group_by_key.get(
                _face_key(vertices[index] for index in face), -1
            )
    regions = np.asarray([row[2] for row in replacement_records], dtype=np.int32)
    blocks = np.asarray([row[3] for row in replacement_records], dtype=np.int32)
    vertices = np.asarray(
        [row[1] for row in replacement_records], dtype=np.uint64
    ).reshape((-1, 4))
    # Rows sorted by removed vertex: the native request requires the pair
    # rows aligned with ``removed_vertex_ids``.
    collapse_pairs = np.asarray(
        sorted((int(row["removed"]), int(row["kept"])) for row in proposals),
        dtype=np.uint64,
    ).reshape((-1, 2))
    moved = sorted(
        (row for row in proposals if row["survivor"] is not None),
        key=lambda row: int(row["kept"]),
    )
    survivors: dict[str, Any] = {}
    if moved:
        # Moved survivors carry no source witness or ancestry (the stencil is
        # kept host-side, as for relocations) and clear any split-edge witness.
        count = len(moved)
        survivors = {
            "changed_vertex_ids": np.asarray(
                [row["kept"] for row in moved], dtype=np.uint64
            ),
            "changed_vertex_points": np.asarray(
                [row["survivor"]["point"] for row in moved], dtype=np.float64
            ).reshape((-1, 3)),
            "changed_witness_strata": np.zeros((count,), dtype=np.int8),
            "changed_witness_entities": np.full((count,), -1, dtype=np.int32),
            "changed_witness_parameters": np.zeros((count, 2), dtype=np.float64),
            "changed_witness_deviations": np.zeros((count,), dtype=np.float64),
            "changed_protection_radii": np.zeros((count,), dtype=np.float64),
            "changed_protected_vertices": np.zeros((count,), dtype=np.int8),
            "changed_edge_witness_present": np.zeros((count,), dtype=np.int8),
            "changed_edge_witness_endpoints": np.zeros((count, 2), dtype=np.uint64),
            "changed_edge_parameters": np.zeros((count,), dtype=np.float64),
        }
    edit = TetMeshEdit(
        kind=TetMeshEditKind.COLLAPSE,
        expected_generation=int(state.snapshot.generation),
        source_id=state.source_identity[0],
        source_revision=state.source_identity[1],
        source_map_id=state.source_identity[2],
        frame_id=state.source_identity[3],
        collapse_pairs=collapse_pairs,
        removed_vertex_ids=np.asarray(removed_vertices, dtype=np.uint64),
        removed_cell_ids=np.asarray(removed_cells, dtype=np.uint64),
        replacement_cell_ancestry_offsets=ancestry_offsets,
        replacement_cell_ancestor_ids=ancestry_ids,
        replacement_cell_ids=replacement_native_ids,
        replacement_cell_vertices=vertices,
        replacement_cell_facet_sources=replacement_facet_sources,
        replacement_regions=regions,
        **boundary,
        **survivors,
    )
    return edit, {
        "proposals": proposals,
        "removed_vertices": removed_vertices,
        "removed_cells": removed_cells,
        "replacement_native_ids": replacement_native_ids,
        "replacement_target_ids": replacement_target_ids,
        "replacement_parent_ids": np.asarray(parent_native_ids, dtype=np.int64),
        "replacement_blocks": blocks,
        "replacement_regions": regions,
        "replacement_parent_ancestry": [
            dict(state.cell_ancestry[parent]) for parent in parent_native_ids
        ],
        "next_native_cell": state.next_native_cell + replacement_count,
        "next_target_cell": state.next_target_cell + replacement_count,
        "next_native_facet": next_facet,
        "next_native_segment": next_segment,
    }


def _boundary_edit_fields(
    state: _NativeState,
    removed_facet_ids: list[int],
    new_facets: list[tuple[tuple[int, int, int], int, int]],
    removed_segment_ids: list[int],
    new_segments: list[tuple[tuple[int, int], int]],
    /,
) -> tuple[dict[str, Any], dict[tuple[int, int, int], int], int, int]:
    """Native edit fields retiring constrained facets/segments for replacements.

    Replacement facets keep the vertex order, group row and orientation of the
    facet they continue; fresh native IDs follow the current allocators.
    """
    fields: dict[str, Any] = {}
    next_facet = state.next_native_facet + len(new_facets)
    next_segment = state.next_native_segment + len(new_segments)
    if removed_facet_ids:
        fields["removed_facet_ids"] = np.asarray(
            sorted(removed_facet_ids), dtype=np.uint64
        )
    if new_facets:
        fields |= {
            "replacement_facet_ids": _fresh_ids(
                state.next_native_facet, len(new_facets), "native facet IDs"
            ).astype(np.uint64),
            "replacement_facet_vertices": np.asarray(
                [vertices for vertices, _, _ in new_facets], dtype=np.uint64
            ).reshape((-1, 3)),
            "replacement_facet_sources": np.asarray(
                [group for _, group, _ in new_facets], dtype=np.int32
            ),
            "replacement_facet_orientations": np.asarray(
                [orientation for _, _, orientation in new_facets], dtype=np.int8
            ),
            "replacement_protected_facets": np.zeros((len(new_facets),), dtype=np.int8),
        }
    if removed_segment_ids:
        fields["removed_segment_ids"] = np.asarray(
            sorted(removed_segment_ids), dtype=np.uint64
        )
    if new_segments:
        fields |= {
            "replacement_segment_ids": _fresh_ids(
                state.next_native_segment, len(new_segments), "native segment IDs"
            ).astype(np.uint64),
            "replacement_segment_vertices": np.asarray(
                [vertices for vertices, _ in new_segments], dtype=np.uint64
            ).reshape((-1, 2)),
            "replacement_segment_sources": np.asarray(
                [group for _, group in new_segments], dtype=np.int32
            ),
            "replacement_protected_segments": np.zeros(
                (len(new_segments),), dtype=np.int8
            ),
        }
    groups = {_face_key(vertices): group for vertices, group, _ in new_facets}
    return fields, groups, next_facet, next_segment


def _apply_collapse_batch(
    state: _NativeState, proposals: list[dict[str, Any]], /
) -> TetMeshEditStatus:
    edit, metadata = _collapse_edit(state, proposals)
    try:
        edit, staged = _prepare_periodic_native_edit(state, edit, metadata)
    except _PeriodicCandidateRejected:
        return TetMeshEditStatus.REJECTED_TOPOLOGY
    status = state.native.apply_edit(edit)
    if status is not TetMeshEditStatus.APPLIED:
        return status
    _commit_periodic_target(state, staged)
    state.snapshot = state.native.arrays()
    source_vertex_ids = {
        int(identifier) for identifier in state.source_vertex_ids.tolist()
    }
    for proposal in metadata["proposals"]:
        removed = int(proposal["removed"])
        kept = int(proposal["kept"])
        removed_target = state.vertex_target_ids[removed]
        kept_target = state.vertex_target_ids[kept]
        # Only a removed source vertex collapses; the stencil sources of a
        # generated vertex survive it. Earlier collapses into the removed vertex
        # now continue into the survivor.
        state.history_vertex[:] = [
            (source_id, kept_target if target_id == removed_target else target_id, kind)
            for source_id, target_id, kind in state.history_vertex
        ]
        if removed_target in source_vertex_ids:
            state.history_vertex.append(
                (removed_target, kept_target, EntityLineageKind.COLLAPSED_INTO)
            )
        state.vertex_target_ids.pop(removed)
        state.vertex_stencils.pop(removed)
        state.metric_by_vertex.pop(removed)
        state.fixed_vertices.discard(removed)
        survivor = proposal["survivor"]
        if survivor is not None:
            # A moved survivor is recorded as a relocation after the collapse.
            state.vertex_stencils[kept] = survivor["stencil"]
            state.metric_by_vertex[kept] = survivor["metric"]
            state.history_vertex[:] = [
                relation
                for relation in state.history_vertex
                if relation[1] != kept_target
                or relation[2] is EntityLineageKind.COLLAPSED_INTO
            ]
            if kept_target in source_vertex_ids:
                state.history_vertex.append(
                    (kept_target, kept_target, EntityLineageKind.RELOCATED)
                )
    for native_cell_id in metadata["removed_cells"]:
        state.cell_target_ids.pop(native_cell_id)
        state.cell_blocks.pop(native_cell_id)
        state.cell_ancestry.pop(native_cell_id)
    for native_id, target_id, block, parent_ancestry in zip(
        metadata["replacement_native_ids"].tolist(),
        metadata["replacement_target_ids"].tolist(),
        metadata["replacement_blocks"].tolist(),
        metadata["replacement_parent_ancestry"],
        strict=True,
    ):
        relation = {
            int(source_id): EntityLineageKind.COARSENED_INTO
            for source_id in parent_ancestry
        }
        state.cell_target_ids[int(native_id)] = int(target_id)
        state.cell_blocks[int(native_id)] = int(block)
        state.cell_ancestry[int(native_id)] = relation
        state.history_cells.extend(
            (source_id, int(target_id), EntityLineageKind.COARSENED_INTO)
            for source_id in relation
        )
        state.changed_source_cell_ids.update(relation)
    state.next_native_cell = int(metadata["next_native_cell"])
    state.next_target_cell = int(metadata["next_target_cell"])
    state.next_native_facet = int(metadata["next_native_facet"])
    state.next_native_segment = int(metadata["next_native_segment"])
    _refresh_native_periodic_windings(state)
    return status


def _orbit_directed_collapses(
    state: _NativeState,
    topology: _TetTopology,
    records: _BoundaryRecords,
    protected: dict[tuple[int, int], bool],
    orbit: tuple[tuple[int, int], ...],
    vertex_quotient: dict[int, int],
    /,
) -> tuple[dict[str, Any], ...] | None:
    """Members of one periodic edge orbit collapsing toward one quotient vertex.

    Each member ranks its own direction over its partial star, so lifted copies
    of one seam edge may disagree. Each quotient direction is then evaluated on
    every member; among the directions every member admits, the one with the
    least worst member violation wins (ties: the larger least removed target
    ID, as ``_collapse_proposal`` breaks ties). ``None``: no common direction.
    """

    def quotient(vertex: int, /) -> int:
        return vertex_quotient[int(state.vertex_target_ids[int(vertex)])]

    best: tuple[dict[str, Any], ...] | None = None
    best_key: tuple[Any, ...] | None = None
    for removed_orbit in sorted({quotient(vertex) for vertex in orbit[0]}):
        members: list[dict[str, Any]] = []
        for edge in orbit:
            removed, kept = (
                (edge[0], edge[1])
                if quotient(edge[0]) == removed_orbit
                else (edge[1], edge[0])
            )
            proposal = _directed_collapse(
                state, topology, records, protected, removed, kept
            )
            if proposal is None:
                break
            members.append(proposal)
        else:
            key = (
                max(member["added_violations"] for member in members),
                -min(
                    int(state.vertex_target_ids[int(member["removed"])])
                    for member in members
                ),
            )
            if best_key is None or key < best_key:
                best, best_key = tuple(members), key
    return best


def _complete_collapse_orbit(
    state: _NativeState, proposals: tuple[dict[str, Any], ...], /
) -> bool:
    """Whether one orbit's collapses remove a whole quotient vertex.

    Every lifted member of the removed vertex orbit is removed (no copy of a
    removed quotient vertex survives) and every member keeps a distinct lifted
    survivor.
    """
    removed = {int(proposal["removed"]) for proposal in proposals}
    kept = {int(proposal["kept"]) for proposal in proposals}
    members = _periodic_vertex_orbit_members(state)
    return (
        len(removed) == len(kept) == len(proposals)
        and set(members[next(iter(removed))]) == removed
    )


def _collapse_metric_candidates(
    state: _NativeState,
    cache: dict[tuple[int, int], tuple[dict[str, Any] | None, frozenset[int]]],
    /,
) -> list[tuple[dict[str, Any], ...]]:
    """Disjoint collapse orbits of the current short edges.

    ``cache`` maps an edge to its proposal (``None``: none, or refused by the
    native mesh) and the cell IDs of its endpoints' stars. A proposal depends
    only on those stars (their vertices, metrics and constrained facets).
    Topology edits retire changed cells; relocation explicitly evicts entries
    intersecting a moved star even when the cell IDs themselves survive.

    On a periodic lift one candidate is one quotient edge orbit: every lifted
    member collapses toward the same quotient vertex
    (``_orbit_directed_collapses``), the removed members are the whole removed
    vertex orbit and the survivors are distinct (``_complete_collapse_orbit``),
    and member cavities are disjoint.
    """
    topology = _state_topology(state)
    for old_edge in cache.keys() - topology.edge_cells.keys():
        del cache[old_edge]
    _, _, lengths = _metric_edge_table(state, topology)
    classes, protected = _current_edge_controls(state, topology)
    records: _BoundaryRecords | None = None
    cell_ids = np.asarray(state.snapshot.cell_ids, dtype=np.int64)
    live = set(cell_ids.tolist())
    eligible: dict[tuple[int, int], tuple[dict[str, Any], float]] = {}
    # Evaluate only one overlapping cavity per batch on a nonperiodic mesh.
    # Skipped edges remain eligible on the next pass, rather than paying for
    # exact proposals that cannot be applied together with an earlier one.
    occupied: set[int] = set()
    nonperiodic = state.target_mesh.periodic_topology is None
    edge_lengths = sorted(
        zip(topology.edge_cells, lengths.tolist(), strict=True),
        key=lambda item: (item[1], item[0]),
    )
    for edge, length in edge_lengths:
        if protected[edge] or classes[edge] != 0 or length >= _LOWER:
            continue
        stars = set(topology.vertex_cells[edge[0]]) | set(topology.vertex_cells[edge[1]])
        star_ids = frozenset(int(cell_ids[row]) for row in stars)
        if nonperiodic and not occupied.isdisjoint(star_ids):
            continue
        cached = cache.get(edge)
        if cached is None or not cached[1] <= live:
            if records is None:
                records = _boundary_records(state)
            cached = (
                _collapse_proposal(state, topology, records, protected, edge),
                star_ids,
            )
            cache[edge] = cached
        if cached[0] is not None:
            eligible[edge] = (cached[0], float(length))
            if nonperiodic:
                occupied.update(star_ids)
    vertex_quotient = _periodic_vertex_quotient_ids(state)
    candidates = []
    for orbit in _periodic_edge_orbit_groups(state):
        if not orbit or any(edge not in eligible for edge in orbit):
            continue
        proposals = tuple(eligible[edge][0] for edge in orbit)
        quotient_edges = {
            frozenset(
                vertex_quotient[int(state.vertex_target_ids[int(vertex)])]
                for vertex in proposal["edge"]
            )
            for proposal in proposals
        }
        kept_orbits = {
            vertex_quotient[int(state.vertex_target_ids[int(proposal["kept"])])]
            for proposal in proposals
        }
        if len(quotient_edges) != 1:
            continue
        if nonperiodic:
            if len(kept_orbits) != 1:
                continue
        else:
            if len(next(iter(quotient_edges))) != 2:
                # Both endpoints are lifted copies of one quotient vertex.
                continue
            if len(kept_orbits) != 1:
                if records is None:
                    records = _boundary_records(state)
                directed = _orbit_directed_collapses(
                    state, topology, records, protected, orbit, vertex_quotient
                )
                if directed is None:
                    continue
                proposals = directed
            if not _complete_collapse_orbit(state, proposals):
                continue
        cavities = [proposal["cavity"] for proposal in proposals]
        cavity = frozenset().union(*cavities)
        if sum(map(len, cavities)) != len(cavity):
            continue
        representative_edge = min(
            tuple(sorted(state.vertex_target_ids[int(vertex)] for vertex in edge))
            for edge in orbit
        )
        priority = (
            min(eligible[edge][1] for edge in orbit),
            -min(proposal["quality_gain"] for proposal in proposals),
            representative_edge,
        )
        candidates.append((priority, proposals, cavity))
    return _disjoint_candidates(candidates)


def _try_collapse_batch(
    state: _NativeState,
    proposal_groups: list[tuple[dict[str, Any], ...]],
    counters: dict[str, int],
    cache: dict[tuple[int, int], tuple[dict[str, Any] | None, frozenset[int]]],
    /,
) -> int:
    proposals = [proposal for group in proposal_groups for proposal in group]
    count = len(proposals)
    status = _apply_collapse_batch(state, proposals)
    if status is TetMeshEditStatus.APPLIED:
        counters["collapses"] += count
        return count
    _raise_fatal_edit_status(status, "collapse")
    if len(proposal_groups) > 1:
        middle = len(proposal_groups) // 2
        return _try_collapse_batch(
            state, proposal_groups[:middle], counters, cache
        ) + _try_collapse_batch(state, proposal_groups[middle:], counters, cache)
    for proposal in proposal_groups[0]:
        # Refused until a star of the edge changes.
        cache[proposal["edge"]] = (None, cache[proposal["edge"]][1])
    _count_rejection(status, counters, 1)
    return 0


def _collapse_phase(
    state: _NativeState,
    counters: dict[str, int],
    /,
    *,
    cache: dict[tuple[int, int], tuple[dict[str, Any] | None, frozenset[int]]]
    | None = None,
) -> int:
    """Collapse until no short edge has an applicable objective-decreasing
    collapse; each pass applies at least one collapse (a strict objective
    decrease) or caches a refusal, so the phase terminates."""
    applied = 0
    if cache is None:
        cache = {}
    while True:
        proposals = _collapse_metric_candidates(state, cache)
        if not proposals:
            return applied
        applied += _try_collapse_batch(state, proposals, counters, cache)


def _replacement_edit(
    state: _NativeState,
    kind: TetMeshEditKind,
    groups: list[dict[str, Any]],
    /,
) -> tuple[TetMeshEdit, dict[str, Any]]:
    if kind not in (
        TetMeshEditKind.FLIP23,
        TetMeshEditKind.FLIP32,
        TetMeshEditKind.RECONNECT,
        TetMeshEditKind.RETRIANGULATE,
    ):
        raise ValueError(f"Unsupported tetra replacement kind {kind!r}.")
    retriangulate = kind is TetMeshEditKind.RETRIANGULATE
    if not groups:
        raise ValueError("A tetra replacement edit requires a complete cavity.")
    records: list[
        tuple[
            int,
            int,
            int,
            tuple[int, int, int, int],
            int,
            int,
            tuple[int, ...],
            EntityLineageKind | None,
            tuple[int, ...],
        ]
    ] = []
    removed: set[int] = set()
    for group_index, group in enumerate(groups):
        removed.update(int(value) for value in group["removed_cell_ids"])
        if retriangulate:
            ancestors: tuple[int, ...] = ()
            native_ancestors: tuple[int, ...] = ()
            lineage_kind = None
        else:
            ancestors = tuple(sorted(int(value) for value in group["source_ancestors"]))
            native_ancestors = tuple(
                sorted(int(value) for value in group["removed_cell_ids"])
            )
            lineage_kind = group["lineage_kind"]
            if not isinstance(lineage_kind, EntityLineageKind):
                raise TypeError("Replacement lineage kind must be EntityLineageKind.")
        for local, (vertices, region) in enumerate(
            zip(
                group["replacement_cell_vertices"],
                group["replacement_regions"],
                strict=True,
            )
        ):
            records.append(
                (
                    min(
                        state.cell_target_ids[int(value)]
                        for value in group["removed_cell_ids"]
                    ),
                    group_index,
                    local,
                    _tetra_key(vertices),
                    int(region),
                    int(group["block"]),
                    ancestors,
                    lineage_kind,
                    native_ancestors,
                )
            )
    records.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    replacement_count = len(records)
    native_ids = _fresh_ids(state.next_native_cell, replacement_count, "native cell IDs")
    target_ids = _fresh_ids(state.next_target_cell, replacement_count, "target cell IDs")
    boundary, new_facet_groups, next_facet, next_segment = _boundary_edit_fields(
        state,
        [facet for group in groups for facet in group.get("removed_facet_ids", ())],
        [facet for group in groups for facet in group.get("replacement_facets", ())],
        [],
        [],
    )
    facet_groups = _facet_group_map(state, state.snapshot) | new_facet_groups
    facet_sources = np.full((replacement_count, 4), -1, dtype=np.int32)
    for row, record in enumerate(records):
        for local_face, face in enumerate(_TETRA_FACES):
            key = _face_key(record[3][index] for index in face)
            facet_sources[row, local_face] = facet_groups.get(key, -1)
    regions = np.asarray([record[4] for record in records], dtype=np.int32)
    blocks = np.asarray([record[5] for record in records], dtype=np.int32)
    ancestry_rows = [record[6] for record in records]
    native_ancestry_rows = [record[8] for record in records]
    ancestry_offsets = np.zeros((replacement_count + 1,), dtype=np.int64)
    ancestry_ids: list[int] = []
    for row, ancestors in enumerate(native_ancestry_rows):
        ancestry_ids.extend(ancestors)
        ancestry_offsets[row + 1] = len(ancestry_ids)
    edit = TetMeshEdit(
        kind=kind,
        expected_generation=int(state.snapshot.generation),
        source_id=state.source_identity[0],
        source_revision=state.source_identity[1],
        source_map_id=state.source_identity[2],
        frame_id=state.source_identity[3],
        removed_cell_ids=np.asarray(sorted(removed), dtype=np.uint64),
        replacement_cell_ancestry_offsets=ancestry_offsets,
        replacement_cell_ancestor_ids=np.asarray(ancestry_ids, dtype=np.uint64),
        replacement_cell_ids=native_ids,
        replacement_cell_vertices=np.asarray(
            [record[3] for record in records], dtype=np.uint64
        ).reshape((-1, 4)),
        replacement_cell_facet_sources=facet_sources,
        replacement_regions=regions,
        **boundary,
    )
    return edit, {
        "removed_cells": tuple(sorted(removed)),
        "replacement_native_ids": native_ids,
        "replacement_target_ids": target_ids,
        "replacement_blocks": blocks,
        "replacement_ancestry": ancestry_rows,
        "replacement_parent_native_ids": [record[8] for record in records],
        "replacement_kinds": [record[7] for record in records],
        "next_native_cell": state.next_native_cell + replacement_count,
        "next_target_cell": state.next_target_cell + replacement_count,
        "next_native_facet": next_facet,
        "next_native_segment": next_segment,
    }


def _apply_replacement_batch(
    state: _NativeState,
    kind: TetMeshEditKind,
    groups: list[dict[str, Any]],
    /,
) -> TetMeshEditStatus:
    edit, metadata = _replacement_edit(state, kind, groups)
    try:
        edit, staged = _prepare_periodic_native_edit(state, edit, metadata)
    except _PeriodicCandidateRejected:
        return TetMeshEditStatus.REJECTED_TOPOLOGY
    status = state.native.apply_edit(edit)
    if status is not TetMeshEditStatus.APPLIED:
        return status
    _commit_periodic_target(state, staged)
    state.snapshot = state.native.arrays()
    for native_cell_id in metadata["removed_cells"]:
        state.cell_target_ids.pop(native_cell_id)
        state.cell_blocks.pop(native_cell_id)
        state.cell_ancestry.pop(native_cell_id)
    for native_id, target_id, block, ancestors, lineage_kind in zip(
        metadata["replacement_native_ids"].tolist(),
        metadata["replacement_target_ids"].tolist(),
        metadata["replacement_blocks"].tolist(),
        metadata["replacement_ancestry"],
        metadata["replacement_kinds"],
        strict=True,
    ):
        if lineage_kind is None:
            relation: dict[int, EntityLineageKind] = {}
        elif isinstance(lineage_kind, EntityLineageKind):
            relation = {int(source): lineage_kind for source in ancestors}
        else:
            raise TypeError("Replacement lineage kind must be EntityLineageKind or None.")
        state.cell_target_ids[int(native_id)] = int(target_id)
        state.cell_blocks[int(native_id)] = int(block)
        state.cell_ancestry[int(native_id)] = relation
        if lineage_kind is not None:
            state.history_cells.extend(
                (source, int(target_id), lineage_kind) for source in relation
            )
            state.changed_source_cell_ids.update(relation)
    state.next_native_cell = int(metadata["next_native_cell"])
    state.next_target_cell = int(metadata["next_target_cell"])
    state.next_native_facet = int(metadata["next_native_facet"])
    state.next_native_segment = int(metadata["next_native_segment"])
    _refresh_native_periodic_windings(state)
    return status


@dataclass(slots=True)
class _ReplacementMemo:
    """Replacement proposals (``None``: none) of one edit kind on a
    nonperiodic mesh, kept across batches, phases and passes.

    A proposal for a face or edge reads only the cells around it, their
    vertices' points and metrics, the existence of edges and constrained
    facets among those vertices, and edge controls fixed by target IDs and
    constrained segments. ``_refresh_replacement_memo`` drops every key in
    the closure of the cells incident to a vertex that a commit touched: a
    vertex of a removed or created cell, or one whose point or metric
    changed. Constrained facets and segments change only with the cells of
    their vertices.
    """

    snapshot: TetMeshArrays | None = None
    metrics: np.ndarray | None = None
    interior: dict[tuple[int, ...], dict[str, Any] | None] = field(default_factory=dict)
    boundary: dict[tuple[int, ...], dict[str, Any] | None] = field(default_factory=dict)
    # The proposals (non-``None`` entries) of ``interior`` and ``boundary``.
    interior_found: dict[tuple[int, ...], dict[str, Any]] = field(default_factory=dict)
    boundary_found: dict[tuple[int, ...], dict[str, Any]] = field(default_factory=dict)


def _refresh_replacement_memo(
    state: _NativeState,
    memo: _ReplacementMemo,
    topology: _TetTopology,
    metrics: np.ndarray,
    /,
) -> None:
    previous, previous_metrics = memo.snapshot, memo.metrics
    current = state.snapshot
    memo.snapshot, memo.metrics = current, metrics
    if previous is None or previous_metrics is None or previous is current:
        return
    if not (memo.interior or memo.boundary):
        return
    previous_vertices = np.asarray(previous.vertex_ids, dtype=np.int64)
    current_vertices = np.asarray(current.vertex_ids, dtype=np.int64)
    _, previous_rows, current_rows = np.intersect1d(
        previous_vertices, current_vertices, assume_unique=True, return_indices=True
    )

    def bits(values: object, width: int, /) -> np.ndarray:
        array = np.ascontiguousarray(values, dtype=np.float64)
        return array.reshape((-1, width)).view(np.uint64)

    # Bitwise comparison: a signed zero is a different binary64 input.
    moved = np.any(
        bits(previous.points, 3)[previous_rows] != bits(current.points, 3)[current_rows],
        axis=1,
    ) | np.any(
        bits(previous_metrics, 9)[previous_rows] != bits(metrics, 9)[current_rows],
        axis=1,
    )
    affected = set(current_vertices[current_rows[moved]].tolist())
    previous_ids = np.asarray(previous.cell_ids, dtype=np.int64)
    current_ids = np.asarray(current.cell_ids, dtype=np.int64)
    removed = ~np.isin(previous_ids, current_ids)
    added = ~np.isin(current_ids, previous_ids)
    affected.update(
        previous_vertices[np.asarray(previous.tetrahedra, dtype=np.int64)[removed]]
        .ravel()
        .tolist()
    )
    affected.update(topology.cell_vertex_ids[added].ravel().tolist())
    rows = {row for vertex in affected for row in topology.vertex_cells.get(vertex, ())}
    stale: set[tuple[int, ...]] = set()
    for row in rows:
        cell = topology.cell_vertex_ids[row].tolist()
        stale.update(
            _edge_key((cell[first], cell[second]))
            for first, second in _TETRA_EDGE_CORNERS
        )
        stale.update(
            _face_key((cell[first], cell[second], cell[third]))
            for first, second, third in _TETRA_FACE_CORNERS
        )
    for key in stale:
        memo.interior.pop(key, None)
        memo.interior_found.pop(key, None)
        memo.boundary.pop(key, None)
        memo.boundary_found.pop(key, None)


def _replacement_keys(
    state: _NativeState, topology: _TetTopology, name: str, /
) -> tuple[tuple[tuple[int, ...], ...], frozenset[tuple[int, ...]]]:
    """Faces or edges a replacement kind proposes on, in key order, with their
    set; built once per snapshot. ``flip23``: interior faces of two cells;
    ``flip32`` / ``reconnect``: unconstrained, unprotected, unclassified
    edges of three / at least four cells; ``boundary``: every edge."""
    tables = _snapshot_tables(state)
    if tables.replacement_keys is None:
        tables.replacement_keys = {}
    cached = tables.replacement_keys.get(name)
    if cached is not None:
        return cached
    if name == "flip23":
        ordered: tuple[tuple[int, ...], ...] = tuple(
            face
            for face, rows in topology.face_cells.items()
            if len(rows) == 2 and face not in topology.facet_row
        )
    elif name == "boundary":
        ordered = tuple(topology.edge_cells)
    elif name in ("flip32", "reconnect"):
        classes, protected = _current_edge_controls(state, topology)
        constrained_edges = {
            _edge_key((face[index], face[(index + 1) % 3]))
            for face in topology.facet_row
            for index in range(3)
        }
        ordered = tuple(
            edge
            for edge, rows in topology.edge_cells.items()
            if edge not in constrained_edges
            and not protected[edge]
            and classes[edge] == 0
            and (len(rows) == 3 if name == "flip32" else len(rows) >= 4)
        )
    else:
        raise ValueError(f"Unknown replacement key set {name!r}.")
    cached = (ordered, frozenset(ordered))
    tables.replacement_keys[name] = cached
    return cached


def _replacement_candidates(
    state: _NativeState,
    kind: TetMeshEditKind,
    blocked: set[tuple[int, ...]],
    mode: PredicateMode,
    /,
    *,
    negative_rings: dict[tuple[int, int], tuple[Any, ...]] | None = None,
    memo: _ReplacementMemo | None = None,
) -> list[tuple[dict[str, Any], ...]]:
    """Objective-decreasing replacement proposals of one kind, grouped by
    periodic orbit. ``memo`` (nonperiodic meshes, one phase call) reuses the
    proposal of every face or edge whose closure no commit has touched."""
    topology = _state_topology(state)
    points, metrics, _ = _metric_edge_table(state, topology)
    regions = np.asarray(state.snapshot.tetrahedron_regions, dtype=np.int64)
    classes, protected = _current_edge_controls(state, topology)
    candidates: list[dict[str, Any]] = []
    cells = np.asarray(state.snapshot.tetrahedra, dtype=np.int64)
    if negative_rings is not None:
        for old_edge in negative_rings.keys() - topology.edge_cells.keys():
            del negative_rings[old_edge]
    cell_ids = np.asarray(state.snapshot.cell_ids, dtype=np.int64)

    vertex_ids = np.asarray(state.snapshot.vertex_ids, dtype=np.int64)

    def cell_edges(rows: np.ndarray) -> set[tuple[int, int]]:
        corners = vertex_ids[rows].tolist()
        return {
            (cell[first], cell[second])
            if cell[first] <= cell[second]
            else (cell[second], cell[first])
            for cell in corners
            for first, second in _TETRA_EDGE_CORNERS
        }

    # Proposals are queued and evaluated together by ``flush``: every kernel
    # (exact orientation, quality, determinant floor, metric length) is
    # row-independent, so each proposal's values are those of its own call.
    pending: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            int,
            int,
            np.ndarray,
            tuple[tuple[int, ...], tuple[tuple[tuple[int, int, int], int, int], ...]]
            | None,
        ]
    ] = []

    def propose(
        key: tuple[int, ...],
        cavity_rows: tuple[int, ...],
        raw_cells: list[tuple[int, int, int, int]],
        boundary: tuple[
            tuple[int, ...], tuple[tuple[tuple[int, int, int], int, int], ...]
        ]
        | None = None,
    ) -> None:
        if key in blocked:
            return
        cavity_ids = tuple(sorted(int(cell_ids[row]) for row in cavity_rows))
        cavity_regions = {int(regions[row]) for row in cavity_rows}
        cavity_blocks = {state.cell_blocks[int(cell_ids[row])] for row in cavity_rows}
        if len(cavity_regions) != 1 or len(cavity_blocks) != 1:
            return
        local = np.asarray(
            [[topology.vertex_row[int(vertex)] for vertex in cell] for cell in raw_cells],
            dtype=np.int64,
        ).reshape((-1, 4))
        pending.append(
            (
                key,
                cavity_rows,
                cavity_ids,
                next(iter(cavity_regions)),
                next(iter(cavity_blocks)),
                local,
                boundary,
            )
        )

    def split_rows(values: np.ndarray, counts: list[int]) -> list[np.ndarray]:
        return np.split(values, np.cumsum(counts)[:-1]) if counts else []

    def flush() -> dict[tuple[int, ...], dict[str, Any]]:
        """Evaluate the queued proposals; append and return those admitted."""
        queued = list(pending)
        pending.clear()
        admitted: dict[tuple[int, ...], dict[str, Any]] = {}
        if not queued:
            return admitted
        # Flip cells are listed in one consistent corner order: all positive,
        # or all mirrored (then reversed together); a mixed set is not a flip.
        signs = split_rows(
            _tetra_orientation(
                points, np.concatenate([record[5] for record in queued]), mode
            ),
            [record[5].shape[0] for record in queued],
        )
        oriented_records = []
        for record, sign in zip(queued, signs, strict=True):
            local = record[5]
            if np.all(sign == 1):
                oriented_records.append((record, local))
            elif np.all(sign == -1):
                local = local.copy()
                local[:, [2, 3]] = local[:, [3, 2]]
                oriented_records.append((record, local))
        if not oriented_records:
            return admitted
        before_rows = [
            cells[np.asarray(record[1], dtype=np.int64)] for record, _ in oriented_records
        ]
        after_rows = [oriented for _, oriented in oriented_records]
        before_counts = [rows.shape[0] for rows in before_rows]
        after_counts = [rows.shape[0] for rows in after_rows]
        qualities = metric_tetra_quality(
            points, metrics, np.concatenate([*before_rows, *after_rows])
        )
        before_qualities = split_rows(qualities[: sum(before_counts)], before_counts)
        after_qualities = split_rows(qualities[sum(before_counts) :], after_counts)
        _, relative = metric_relative_determinants(points[np.concatenate(after_rows)])
        floors = split_rows(relative > _DETERMINANT_FLOOR, after_counts)
        survivors = []
        for index, (record, oriented) in enumerate(oriented_records):
            before, after = before_qualities[index], after_qualities[index]
            if (
                before.size == 0
                or after.size == 0
                or not np.all(np.isfinite(after))
                or np.any(after <= 0.0)
                or not bool(np.all(floors[index]))
            ):
                continue
            old_edges = cell_edges(before_rows[index])
            new_edges = cell_edges(oriented)
            removed_edges = sorted(
                edge for edge in old_edges - new_edges if not protected[edge]
            )
            added_edges = sorted(new_edges - old_edges)
            survivors.append(
                (record, oriented, before, after, removed_edges, added_edges)
            )
        if not survivors:
            return admitted
        edge_counts = [
            count
            for *_, removed_edges, added_edges in survivors
            for count in (len(removed_edges), len(added_edges))
        ]
        lengths = split_rows(
            _edge_set_lengths(
                points,
                metrics,
                topology,
                [
                    edge
                    for *_, removed_edges, added_edges in survivors
                    for edge in (*removed_edges, *added_edges)
                ],
            ),
            edge_counts,
        )
        for index, (record, oriented, before, after, _, _) in enumerate(survivors):
            key, _, cavity_ids, region, block, _, boundary = record
            if not _objective_decreases(
                lengths[2 * index],
                lengths[2 * index + 1],
                float(np.min(after)) > float(np.min(before)) + _IMPROVEMENT,
            ):
                continue
            vertices = vertex_ids[oriented]
            ancestors = sorted(
                {
                    int(source_id)
                    for cell_id in cavity_ids
                    for source_id in state.cell_ancestry[cell_id]
                }
            )
            proposal = {
                "candidate_key": key,
                "removed_cell_ids": cavity_ids,
                "replacement_cell_vertices": vertices,
                "replacement_regions": np.full(
                    (vertices.shape[0],), region, dtype=np.int32
                ),
                "block": block,
                "source_ancestors": tuple(ancestors),
                "lineage_kind": EntityLineageKind.SWAPPED_FROM,
                "cavity": frozenset(cavity_ids),
                "gain": float(np.min(after) - np.min(before)),
                "removed_facet_ids": () if boundary is None else boundary[0],
                "replacement_facets": () if boundary is None else boundary[1],
            }
            candidates.append(proposal)
            admitted[key] = proposal
        return admitted

    # Edges whose ring proposal awaits ``flush``, with their ring signature.
    ring_signatures: dict[tuple[int, int], tuple[Any, ...]] = {}

    def scan(
        name: str,
        table: dict[tuple[int, ...], dict[str, Any] | None] | None,
        found: dict[tuple[int, ...], dict[str, Any]] | None,
        evaluate: Callable[[tuple[int, ...]], None],
    ) -> None:
        """Propose every unblocked key of ``name`` (``_replacement_keys``).

        With a memo, a key keeps its proposal while its closure is untouched
        (``_refresh_replacement_memo``) and only keys without one are
        evaluated; the proposals and hence the sorted groups are unchanged."""
        ordered, members = _replacement_keys(state, topology, name)
        if table is None or found is None:
            missing: list[tuple[int, ...]] = [
                key for key in ordered if key not in blocked
            ]
        else:
            candidates.extend(
                proposal
                for key, proposal in found.items()
                if key in members and key not in blocked
            )
            missing = sorted(members - blocked - table.keys())
        for key in missing:
            evaluate(key)
        admitted = flush()
        if negative_rings is not None:
            for edge, signature in ring_signatures.items():
                if edge in admitted:
                    negative_rings.pop(edge, None)
                else:
                    negative_rings[edge] = signature
        ring_signatures.clear()
        if table is not None and found is not None:
            for key in missing:
                proposal = admitted.get(key)
                table[key] = proposal
                if proposal is not None:
                    found[key] = proposal

    if memo is not None:
        _refresh_replacement_memo(state, memo, topology, metrics)
    interior = None if memo is None else memo.interior
    interior_found = None if memo is None else memo.interior_found

    if kind is TetMeshEditKind.FLIP23:

        def flip23(face: tuple[int, ...], rows: tuple[int, ...]) -> None:
            first, second = rows
            first_vertices = topology.cell_vertex_ids[first]
            second_vertices = topology.cell_vertex_ids[second]
            opposite_first = [
                int(value) for value in first_vertices if int(value) not in face
            ]
            opposite_second = [
                int(value) for value in second_vertices if int(value) not in face
            ]
            if len(opposite_first) != 1 or len(opposite_second) != 1:
                return
            u, v = opposite_first[0], opposite_second[0]
            new_edge = (min(u, v), max(u, v))
            target_edge = tuple(
                sorted(state.vertex_target_ids[vertex] for vertex in new_edge)
            )
            if (
                new_edge in topology.edge_cells
                or new_edge in topology.segment_row
                or target_edge in state.protected_edge_ids
                or state.edge_classes.get(target_edge, 0) > 0
            ):
                return
            a, b, c = face
            propose(
                face,
                rows,
                [
                    (u, v, a, b),
                    (u, v, b, c),
                    (u, v, c, a),
                ],
            )

        scan(
            "flip23",
            interior,
            interior_found,
            lambda face: flip23(
                face, topology.face_cells[cast("tuple[int, int, int]", face)]
            ),
        )
        return _group_periodic_replacement_candidates(state, candidates, 2)

    if kind not in (TetMeshEditKind.FLIP32, TetMeshEditKind.RECONNECT):
        raise ValueError(
            "Replacement candidate kind must be FLIP23, FLIP32, or RECONNECT."
        )

    def ring_edit(edge: tuple[int, int], rows: tuple[int, ...]) -> None:
        ring = _edge_ring(topology, edge, rows)
        if ring is None:
            return
        signature = None
        if negative_rings is not None and kind is TetMeshEditKind.RECONNECT:
            vertices = tuple(
                sorted({int(v) for row in rows for v in topology.cell_vertex_ids[row]})
            )
            vertex_rows = np.asarray(
                [topology.vertex_row[v] for v in vertices], dtype=np.int64
            )
            candidate_edges = tuple(
                (
                    _edge_key(pair),
                    _edge_key(pair) in topology.edge_cells,
                    protected.get(_edge_key(pair)),
                    classes.get(_edge_key(pair)),
                )
                for pair in combinations(vertices, 2)
            )
            signature = (
                tuple(sorted(int(cell_ids[row]) for row in rows)),
                tuple(
                    (int(regions[row]), state.cell_blocks[int(cell_ids[row])])
                    for row in rows
                ),
                tuple(
                    tuple(int(v) for v in topology.cell_vertex_ids[row]) for row in rows
                ),
                ring,
                vertices,
                points[vertex_rows].tobytes(),
                metrics[vertex_rows].tobytes(),
                candidate_edges,
            )
            if negative_rings.get(edge) == signature:
                return
        if kind is TetMeshEditKind.FLIP32:
            if len(ring) != 3:
                return
            a, b, c = ring
            raw = [
                (a, b, c, edge[0]),
                (a, c, b, edge[1]),
            ]
        else:
            triangulation = _best_ring_triangulation(
                points, metrics, topology, edge, ring, mode
            )
            if triangulation is None:
                if signature is not None and negative_rings is not None:
                    negative_rings[edge] = signature
                return
            raw = list(triangulation[0])
        if signature is not None:
            ring_signatures[edge] = signature
        propose(edge, rows, raw)

    scan(
        "flip32" if kind is TetMeshEditKind.FLIP32 else "reconnect",
        interior,
        interior_found,
        lambda edge: ring_edit(
            cast("tuple[int, int]", edge),
            topology.edge_cells[cast("tuple[int, int]", edge)],
        ),
    )
    if kind is TetMeshEditKind.RECONNECT and state.target_mesh.periodic_topology is None:
        records = _boundary_records(state)

        def boundary_edit(edge: tuple[int, int], rows: tuple[int, ...]) -> None:
            swap = _boundary_swap(
                state, topology, points, records, protected, classes, edge, rows
            )
            if swap is not None:
                ring, boundary = swap
                triangulation = _best_ring_triangulation(
                    points, metrics, topology, edge, ring, mode
                )
                if triangulation is not None:
                    propose(edge, rows, list(triangulation[0]), boundary)

        # ``_boundary_swap`` only reads; a blocked edge is never proposed.
        scan(
            "boundary",
            None if memo is None else memo.boundary,
            None if memo is None else memo.boundary_found,
            lambda edge: boundary_edit(
                cast("tuple[int, int]", edge),
                topology.edge_cells[cast("tuple[int, int]", edge)],
            ),
        )
    return _group_periodic_replacement_candidates(state, candidates, 1)


def _boundary_swap(
    state: _NativeState,
    topology: _TetTopology,
    points: np.ndarray,
    records: _BoundaryRecords,
    protected: dict[tuple[int, int], bool],
    classes: dict[tuple[int, int], int],
    edge: tuple[int, int],
    rows: tuple[int, ...],
    /,
) -> (
    tuple[
        tuple[int, ...],
        tuple[tuple[int, ...], tuple[tuple[tuple[int, int, int], int, int], ...]],
    ]
    | None
):
    """Open edge ring and facet changes of a coplanar boundary edge swap.

    A boundary edge ``ab`` shared by the facets ``abc`` and ``abd`` of one
    source group is replaced by ``cd``: the cells around ``ab`` (an open fan
    from ``c`` to ``d``) are retriangulated by the ring rule and the facets
    become ``acd`` and ``bcd``, each oriented like the facets it replaces.
    """
    if edge in topology.segment_row or protected[edge] or classes[edge] != 0:
        return None
    facets = [
        records.facets[key]
        for key in records.vertex_facets.get(edge[0], ())
        if edge[1] in key
    ]
    if len(facets) != 2 or facets[0][1] != facets[1][1]:
        return None
    if facets[0][4] or facets[1][4] or facets[0][3] != facets[1][3]:
        return None
    first, second = edge
    opposite = [
        next(int(vertex) for vertex in facet[2] if vertex not in edge) for facet in facets
    ]
    c, d = opposite
    if _edge_key((c, d)) in topology.edge_cells or len(rows) < 2:
        return None
    adjacency: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        other = [
            int(vertex)
            for vertex in topology.cell_vertex_ids[row]
            if int(vertex) not in edge
        ]
        if len(other) != 2:
            return None
        adjacency[other[0]].add(other[1])
        adjacency[other[1]].add(other[0])
    if len(adjacency[c]) != 1 or len(adjacency[d]) != 1:
        return None
    ring = [c]
    previous = None
    while ring[-1] != d:
        following = [vertex for vertex in adjacency[ring[-1]] if vertex != previous]
        if len(following) != 1 or following[0] in ring:
            return None
        previous = ring[-1]
        ring.append(following[0])
    if len(ring) != len(adjacency) or len(ring) < 3:
        return None
    # Each new facet keeps the oriented normal of the old facets (coplanar).
    exact = {
        vertex: [Fraction(float(value)) for value in points[topology.vertex_row[vertex]]]
        for vertex in (first, second, c, d)
    }

    def normal(triangle: tuple[int, int, int]) -> list[Fraction]:
        p, q, r = (exact[vertex] for vertex in triangle)
        u = [q[axis] - p[axis] for axis in range(3)]
        v = [r[axis] - p[axis] for axis in range(3)]
        return [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]

    reference = normal(facets[0][2])
    new_facets: list[tuple[tuple[int, int, int], int, int]] = []
    for apex in edge:
        triangle = (apex, c, d)
        alignment = sum(
            (
                left * right
                for left, right in zip(normal(triangle), reference, strict=True)
            ),
            Fraction(0),
        )
        if alignment == 0:
            return None
        if alignment < 0:
            triangle = (apex, d, c)
        new_facets.append((triangle, facets[0][1], facets[0][3]))
    return tuple(ring), (
        (facets[0][0], facets[1][0]),
        (new_facets[0], new_facets[1]),
    )


def _group_periodic_replacement_candidates(
    state: _NativeState,
    candidates: list[dict[str, Any]],
    dimension: int,
    /,
) -> list[tuple[dict[str, Any], ...]]:
    by_key: dict[tuple[int, ...], dict[str, Any]] = {}
    for candidate in candidates:
        candidate_key = candidate["candidate_key"]
        key = _edge_key(candidate_key) if dimension == 1 else _face_key(candidate_key)
        by_key[key] = candidate
    if dimension not in (1, 2):
        raise ValueError("Replacement candidates require edge or face orbits.")
    grouped: list[tuple[dict[str, Any], ...]] = []
    if state.target_mesh.periodic_topology is None:
        # Nonperiodic orbits are the single faces or edges of the topology.
        grouped = [(candidate,) for candidate in by_key.values()]
    else:
        orbit_groups = (
            _periodic_edge_orbit_groups(state)
            if dimension == 1
            else _periodic_face_orbit_groups(state)
        )
        for orbit in orbit_groups:
            keys = tuple(orbit)
            if not keys or any(key not in by_key for key in keys):
                continue
            members = tuple(by_key[key] for key in keys)
            cavities = [member["cavity"] for member in members]
            cavity = frozenset().union(*cavities)
            if sum(map(len, cavities)) != len(cavity):
                continue
            grouped.append(members)
    return sorted(
        grouped,
        key=lambda members: (
            -min(float(member["gain"]) for member in members),
            tuple(
                tuple(sorted(int(value) for value in member["candidate_key"]))
                for member in members
            ),
        ),
    )


def _try_replacement_batch(
    state: _NativeState,
    kind: TetMeshEditKind,
    candidate_groups: list[tuple[dict[str, Any], ...]],
    counter_name: str,
    counters: dict[str, int],
    blocked: set[tuple[int, ...]],
    /,
) -> int:
    groups = [candidate for orbit in candidate_groups for candidate in orbit]
    count = len(groups)
    status = _apply_replacement_batch(state, kind, groups)
    if status is TetMeshEditStatus.APPLIED:
        counters[counter_name] += count
        return count
    _raise_fatal_edit_status(status, "replacement")
    if len(candidate_groups) > 1:
        middle = len(candidate_groups) // 2
        return _try_replacement_batch(
            state, kind, candidate_groups[:middle], counter_name, counters, blocked
        ) + _try_replacement_batch(
            state, kind, candidate_groups[middle:], counter_name, counters, blocked
        )
    blocked.update(tuple(candidate["candidate_key"]) for candidate in candidate_groups[0])
    _count_rejection(status, counters, 1)
    return 0


def _replacement_phase(
    state: _NativeState,
    kind: TetMeshEditKind,
    counter_name: str,
    counters: dict[str, int],
    mode: PredicateMode,
    /,
    *,
    negative_rings: dict[tuple[int, int], tuple[Any, ...]] | None = None,
    memo: _ReplacementMemo | None = None,
) -> int:
    """Apply disjoint batches of one replacement kind until none applies.

    ``memo`` (nonperiodic meshes) carries this kind's proposals across calls;
    it is a pure cache and never changes the proposals or their order."""
    applied = 0
    blocked: set[tuple[int, ...]] = set()
    if state.target_mesh.periodic_topology is not None:
        memo = None
    while True:
        candidates = _replacement_candidates(
            state, kind, blocked, mode, negative_rings=negative_rings, memo=memo
        )
        if not candidates:
            return applied
        independent = _disjoint_candidates(
            [
                (
                    (
                        -min(float(candidate["gain"]) for candidate in orbit),
                        tuple(tuple(candidate["candidate_key"]) for candidate in orbit),
                    ),
                    orbit,
                    frozenset().union(*(candidate["cavity"] for candidate in orbit)),
                )
                for orbit in candidates
            ]
        )
        if not independent:
            return applied
        changed = _try_replacement_batch(
            state, kind, independent, counter_name, counters, blocked
        )
        applied += changed
        if changed:
            blocked.clear()


def _source_carrier_frame(
    state: _NativeState, vertices: tuple[int, ...], /
) -> _EntityFrame:
    """Exact affine frame of an edge or face in the immutable source."""
    frame = state.source_frames.get(vertices)
    if frame is None:
        corners = tuple(
            state.source_exact_points[state.source_vertex_rows[vertex]]
            for vertex in vertices
        )
        # Only an edge or a triangle spans a frame.
        frame = _entity_frame(corners) if len(corners) in (2, 3) else (corners[0], None)
        state.source_frames[vertices] = frame
    return frame


def _source_cell_carrier(
    state: _NativeState,
    record: _SourceTetRecord,
    simplices: tuple[tuple[int, ...], ...] | None,
    /,
) -> tuple[int, ...] | None:
    """Find the selected source tetrahedron's full-dimensional carrier face."""
    if simplices is None:
        return record[1]
    if not simplices:
        return None
    simplex = simplices[0]
    origin, gradients = _source_carrier_frame(state, simplex)
    if gradients is None:
        return None
    axes = tuple(
        tuple(
            state.source_exact_points[state.source_vertex_rows[v]][k] - origin[k]
            for k in range(3)
        )
        for v in simplex[1:]
    )
    members = []
    for vertex in record[1]:
        point = state.source_exact_points[state.source_vertex_rows[vertex]]
        offset = tuple(point[k] - origin[k] for k in range(3))
        weights = tuple(
            sum((g[k] * offset[k] for k in range(3)), Fraction(0)) for g in gradients
        )
        if all(
            offset[k]
            == sum((weights[j] * axes[j][k] for j in range(len(axes))), Fraction(0))
            for k in range(3)
        ):
            members.append(vertex)
    return tuple(sorted(members)) if len(members) == len(simplex) else None


# The 53 resolutions of a carrier and the coarsest one repeated by a
# coarse-and-fine search.
_ALL_CARRIERS = 54


def _bank_simplices(
    state: _NativeState,
    bank: _SourceBank,
    simplices: tuple[tuple[int, ...], ...] | None,
    /,
) -> tuple[tuple[tuple[int, ...], bool], ...] | None:
    """Carrier simplices as bank rows, each with whether it is a declared face."""
    if simplices is None:
        return None
    return tuple(
        (
            tuple(bank.rows[vertex] for vertex in simplex),
            len(simplex) != 2 and frozenset(simplex) in state.declared_source_faces,
        )
        for simplex in simplices
    )


def _source_carriers(
    state: _NativeState,
    simplices: tuple[tuple[int, ...], ...] | None,
    target: np.ndarray,
    current: np.ndarray,
    order: MetricCarrierOrder,
    limit: int,
    cell: int | None,
    /,
) -> tuple[tuple[dict[int, Fraction], np.ndarray], ...]:
    bank = _source_bank(state)
    return tuple(
        (
            {
                bank.identifiers[row]: Fraction(weight, 1 << carrier.bits)
                for row, weight in zip(carrier.rows, carrier.weights, strict=True)
            },
            carrier.point,
        )
        for carrier in bank.source.carriers(
            target,
            current,
            _bank_simplices(state, bank, simplices),
            order,
            limit,
            cell=None if cell is None else bank.cells[cell],
        )
    )


def _quantized_relocation_candidates(
    state: _NativeState,
    simplices: tuple[tuple[int, ...], ...] | None,
    target: np.ndarray,
    current: np.ndarray,
    /,
    *,
    fine_first: bool = False,
    coarse_and_fine: bool = False,
    limit: int = _ALL_CARRIERS,
    cell: int | None = None,
) -> tuple[tuple[dict[int, Fraction], np.ndarray], ...]:
    """Exact dyadic source carriers for one relocation target, at most ``limit``.

    Each carrier is closer to the target than the current point. Boundary
    images retain the exact or declared-plane admission required by the
    native source complex. The target is rounded at each dyadic resolution
    through 53 bits, in fine-to-coarse or coarse-to-fine order.
    With ``coarse_and_fine``, the first two results are the coarsest and
    finest admissible carriers, sharing the same exact source location.
    An interior target is located in the first source cell that holds it, or
    in source cell ``cell`` alone.

    Each support weight is rounded to the grid ``2**-bits`` through its
    partial sums (half to even, monotone, the last exactly one), so the
    quantized weights are nonnegative, sum to one and are binary64 values;
    the point stays in the entity spanned by the support. The rounding and
    the image ``sum_i w_i X_i`` are evaluated in integers over the exact
    source points by the native source (``MetricSource.carriers``), so
    stencils, carriers and admissions are exact.
    """
    if coarse_and_fine:
        order = MetricCarrierOrder.COARSE_THEN_FINE
    elif fine_first:
        order = MetricCarrierOrder.FINE_FIRST
    else:
        order = MetricCarrierOrder.COARSE_FIRST
    return _source_carriers(state, simplices, target, current, order, limit, cell)


def _quantized_relocation(
    state: _NativeState,
    simplices: tuple[tuple[int, ...], ...] | None,
    target: np.ndarray,
    current: np.ndarray,
    /,
) -> tuple[dict[int, Fraction], np.ndarray] | None:
    """Return the coarsest admissible exact source carrier, if one exists."""
    found = _source_carriers(
        state, simplices, target, current, MetricCarrierOrder.COARSE_FIRST, 1, None
    )
    return found[0] if found else None


def _implicit_tangent_pulls(
    state: _NativeState, point: np.ndarray, pulls: np.ndarray, /
) -> np.ndarray:
    """Propose current-level tangents; native admission still owns fidelity."""
    implicit = state.snapshot.implicit_source
    if implicit is None:
        raise RuntimeError("Implicit relocation requires its bound source program.")
    program = implicit.source.gradient
    lower = program.workspace_lower.copy()
    upper = program.workspace_upper.copy()
    lower[:3] = point
    upper[:3] = point
    interval_program_execute(
        program.instructions, program.indices, program.parameters, lower, upper
    )
    if not np.all(lower[program.guards] >= 0.0):
        return pulls[:0]
    gradient_lower = lower[program.outputs]
    gradient_upper = upper[program.outputs]
    if not (
        np.all(np.isfinite(gradient_lower))
        and np.all(np.isfinite(gradient_upper))
        and np.all(gradient_lower <= gradient_upper)
    ):
        return pulls[:0]
    scale = max(
        float(np.max(np.abs(gradient_lower))),
        float(np.max(np.abs(gradient_upper))),
    )
    if scale == 0.0:
        return pulls[:0]
    # Scaling before the midpoint avoids overflow and preserves tiny gradients.
    normal = 0.5 * (gradient_lower / scale) + 0.5 * (gradient_upper / scale)
    # Three-term sums in coordinate order: each a binary64 value of its own.
    norm_squared = float(
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
    )
    if norm_squared == 0.0:
        return pulls[:0]
    along = pulls[:, 0] * normal[0] + pulls[:, 1] * normal[1] + pulls[:, 2] * normal[2]
    return pulls - (along / norm_squared)[:, None] * normal


class _RelocationStar(NamedTuple):
    rows: np.ndarray
    cell_ids: tuple[int, ...]
    neighbors: tuple[int, ...]
    local_rows: np.ndarray
    local_cells: np.ndarray
    spokes: np.ndarray


class _RelocationPreparation(NamedTuple):
    topology: _TetTopology
    points: np.ndarray
    metrics: np.ndarray
    lengths: np.ndarray
    qualities: np.ndarray
    classes: dict[tuple[int, int], int]
    protected: dict[tuple[int, int], bool]
    records: _BoundaryRecords
    edge_rows: np.ndarray
    incident_edges: dict[int, tuple[int, ...]]
    stars: dict[int, _RelocationStar]
    # Filled by ``_relocation_proposals`` and kept while this geometry stands:
    # a round whose candidates are all refused changes neither. ``orders``
    # holds the searched vertices by target ID for ``active_only`` false and
    # true; ``star_cells`` the star cell IDs of a vertex (incidence only).
    orders: dict[bool, tuple[int, ...]]
    star_cells: dict[int, frozenset[int]]


def _prepare_relocation_geometry(
    state: _NativeState,
    /,
    *,
    previous: tuple[TetMeshArrays, _RelocationPreparation] | None = None,
    moved_vertices: frozenset[int] | None = None,
) -> _RelocationPreparation:
    snapshot = state.snapshot
    # Pure relocation normally preserves all incidence rows. Check the actual
    # snapshot banks before reusing them; a changed native representation takes
    # the ordinary complete rebuild, including for periodic edits.
    same_incidence = previous is not None and all(
        np.array_equal(getattr(snapshot, name), getattr(previous[0], name))
        for name in ("vertex_ids", "cell_ids", "tetrahedra", "faces", "segments")
    )
    stars: dict[int, _RelocationStar]
    star_cells: dict[int, frozenset[int]]
    if same_incidence and previous is not None:
        topology = previous[1].topology
        edge_rows = previous[1].edge_rows
        incident_edges = previous[1].incident_edges
        stars = previous[1].stars
        star_cells = previous[1].star_cells
        # Equal incidence banks give an equal topology: share it with every
        # phase that reads this snapshot.
        tables = _snapshot_tables(state)
        if tables.topology is None:
            tables.topology = topology
            tables.edge_rows = edge_rows
    else:
        topology = _state_topology(state)
        edge_rows = _state_edge_rows(state)
        stars = {}
        star_cells = {}
        incident_lists: dict[int, list[int]] = defaultdict(list)
        for index, (first, second) in enumerate(topology.edge_cells):
            incident_lists[first].append(index)
            incident_lists[second].append(index)
        incident_edges = {
            vertex: tuple(indices) for vertex, indices in incident_lists.items()
        }
    points = np.asarray(snapshot.points, dtype=np.float64)
    can_update = (
        same_incidence
        and previous is not None
        and moved_vertices is not None
        and state.target_mesh.periodic_topology is None
    )
    if can_update and previous is not None and moved_vertices is not None:
        moved_rows = np.asarray(
            [topology.vertex_row[vertex] for vertex in moved_vertices], dtype=np.int64
        )
        unchanged_rows = np.ones(len(points), dtype=np.bool_)
        unchanged_rows[moved_rows] = False
        can_update = np.array_equal(
            points[unchanged_rows], previous[1].points[unchanged_rows]
        )
    if can_update and previous is not None and moved_vertices is not None:
        metrics = previous[1].metrics.copy()
        for vertex in moved_vertices:
            metrics[topology.vertex_row[vertex]] = state.metric_by_vertex[vertex]
        changed_edges = np.asarray(
            sorted(
                {index for vertex in moved_vertices for index in incident_edges[vertex]}
            ),
            dtype=np.int64,
        )
        lengths = previous[1].lengths.copy()
        lengths[changed_edges] = metric_lengths(points, metrics, edge_rows[changed_edges])
        metrics.setflags(write=False)
        lengths.setflags(write=False)
        tables = _snapshot_tables(state)
        if topology is tables.topology and tables.metric_table is None:
            # Row-independent lengths: the updated table equals a rebuild.
            tables.metric_table = (points, metrics, lengths)
    else:
        points, metrics, lengths = _metric_edge_table(state, topology)
    cells = np.asarray(snapshot.tetrahedra, dtype=np.int64)
    if can_update and previous is not None and moved_vertices is not None:
        changed_cells = np.asarray(
            sorted(
                {
                    cell
                    for vertex in moved_vertices
                    for cell in topology.vertex_cells[vertex]
                }
            ),
            dtype=np.int64,
        )
        qualities = previous[1].qualities.copy()
        qualities[changed_cells] = metric_tetra_quality(
            points, metrics, cells[changed_cells]
        )
    else:
        qualities = metric_tetra_quality(points, metrics, cells)
    qualities.setflags(write=False)
    if (
        same_incidence
        and previous is not None
        and all(
            np.array_equal(getattr(snapshot, name), getattr(previous[0], name))
            for name in ("segment_sources", "protected_segments")
        )
    ):
        classes, protected = previous[1].classes, previous[1].protected
    else:
        classes, protected = _current_edge_controls(state, topology)
    if (
        same_incidence
        and previous is not None
        and all(
            np.array_equal(getattr(snapshot, name), getattr(previous[0], name))
            for name in (
                "facet_ids",
                "face_sources",
                "facet_orientations",
                "protected_facets",
                "segment_ids",
                "segment_sources",
                "protected_segments",
            )
        )
    ):
        records = previous[1].records
    else:
        records = _boundary_records(state)
    return _RelocationPreparation(
        topology,
        points,
        metrics,
        lengths,
        qualities,
        classes,
        protected,
        records,
        edge_rows,
        incident_edges,
        stars,
        {},
        star_cells,
    )


def _band_violation_rows(lengths: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """``_band_violations`` of each row of ``(g, m)`` lengths.

    Each row's long (short) quanta are the same elementwise values, sorted
    descending and padded with zeros to ``m``. Violations are at least one
    quantum, so the lexicographic order of padded rows is the max-first tuple
    order (a proper prefix is smaller). A zero or infinite length has an
    infinite count.
    """
    above, below = metric_band_quanta(lengths)
    return -np.sort(-above, axis=1), -np.sort(-below, axis=1)


def _report_relocation_candidate(
    state: _NativeState,
    bank: _SourceBank,
    candidate: MetricRelocationCandidate,
    measured: np.ndarray,
    end: MetricRelocationEnd,
    /,
) -> NoReturn:
    """Raise the error of a goal or carrier the native search could not decide.

    The owners of the two conditions raise: the band violations for a spoke
    without a finite length, the source tensor for a stencil without one.
    """
    if end is MetricRelocationEnd.NO_TENSOR:
        _source_metric_from_stencil(
            state.source_metric,
            state.source_vertex_rows,
            {
                bank.identifiers[row]: Fraction(weight, 1 << candidate.bits)
                for row, weight in zip(candidate.rows, candidate.weights, strict=True)
            },
            source_metric_log=state.source_metric_log,
        )
    else:
        _band_violations(candidate.lengths[measured])
    raise RuntimeError("A relocation candidate without a verdict raised no error.")


def _relocation_proposals(
    state: _NativeState,
    cache: dict[int, _RelocationSearch | None],
    /,
    *,
    deep: bool = False,
    refine: bool = False,
    active_only: bool = False,
    band_only: bool = False,
    prepared: _RelocationPreparation | None = None,
) -> list[tuple[dict[str, Any], ...]]:
    """Objective-decreasing vertex moves to quantized source points.

    A vertex is pulled toward unit metric length along its edges, both by the
    mean spring force and by each out-of-band spoke. Dyadic backtracking
    continues until the binary64 goal no longer differs from the current
    carrier; each goal is quantized by ``_quantized_relocation``. Interior
    vertices move in their source cell,
    boundary vertices within the source triangle or edge carrying them (pinned
    at source corners); a periodic lift moves interior vertices only. A move is
    proposed when its star stays positively oriented above the determinant
    floor (``_floor_oriented``) and strictly decreases the termination
    objective. ``cache`` retains the finite search on each unchanged star:
    native refusal consumes a proposal, not all remaining source candidates.
    The caller evicts searches on the closed stars of moved vertices.
    Preparation remains grouped to preserve its floating-point operation and
    goal order; resumable quantization is isolated in a star-local continuation.
    """
    snapshot = state.snapshot
    if prepared is None:
        prepared = _prepare_relocation_geometry(state)
    topology = prepared.topology
    points, metrics, edge_lengths = prepared.points, prepared.metrics, prepared.lengths
    protected_edges, records = prepared.protected, prepared.records
    order = prepared.orders.get(active_only)
    if order is None:
        violating_vertices = (
            {
                vertex
                for edge, length in zip(topology.edge_cells, edge_lengths, strict=True)
                if not protected_edges[edge] and (length < _LOWER or length > _UPPER)
                for vertex in edge
            }
            if active_only
            else None
        )
        order = tuple(
            sorted(
                topology.vertex_cells
                if violating_vertices is None
                else violating_vertices,
                key=lambda value: state.vertex_target_ids[value],
            )
        )
        prepared.orders[active_only] = order
    periodic = state.target_mesh.periodic_topology is not None
    dimensions = snapshot.vertex_dimensions
    protected_vertices = snapshot.protected_vertices
    cells = snapshot.tetrahedra
    regions = snapshot.tetrahedron_regions
    current_cell_ids = np.asarray(snapshot.cell_ids, dtype=np.int64)
    source_coordinates = state.source_points
    constant_metric = bool(np.all(state.source_metric == state.source_metric[0]))

    def propose(vertex: int) -> _RelocationSearch | None:
        row = topology.vertex_row[vertex]
        if vertex in state.fixed_vertices or protected_vertices[row]:
            return None
        implicit_only = _implicit_only_vertex(state, records, vertex)
        simplices = None if implicit_only else _carrier_simplices(state, records, vertex)
        if (periodic or (simplices is None and not implicit_only)) and dimensions[
            row
        ] != 3:
            return None
        if simplices is not None and (periodic or not simplices):
            return None
        star = topology.vertex_cells[vertex]
        if not star:
            return None
        star_shape = prepared.stars.get(vertex)
        if star_shape is None:
            star_rows = np.asarray(star, dtype=np.int64)
            star_ids = tuple(int(current_cell_ids[cell]) for cell in star)
        else:
            star_rows, star_ids = star_shape.rows, star_shape.cell_ids
        star_blocks = {state.cell_blocks[cell_id] for cell_id in star_ids}
        star_regions = {int(regions[cell]) for cell in star}
        if len(star_blocks) != 1 or len(star_regions) != 1:
            return None
        if star_shape is None:
            neighbors = tuple(
                sorted(
                    {
                        int(other)
                        for cell in star
                        for other in topology.cell_vertex_ids[cell]
                        if int(other) != vertex
                    }
                )
            )
            # Star-local arrays: row 0 is the moving vertex.
            local_rows = np.asarray(
                [row, *(topology.vertex_row[neighbor] for neighbor in neighbors)],
                dtype=np.int64,
            )
            local_index = {
                int(global_row): index for index, global_row in enumerate(local_rows)
            }
            local_cells = np.asarray(
                [
                    [local_index[int(value)] for value in cell]
                    for cell in cells[star_rows]
                ],
                dtype=np.int64,
            ).reshape((-1, 4))
            spokes = np.column_stack(
                (
                    np.zeros((len(neighbors),), dtype=np.int64),
                    np.arange(1, len(neighbors) + 1, dtype=np.int64),
                )
            )
            star_shape = _RelocationStar(
                star_rows, star_ids, neighbors, local_rows, local_cells, spokes
            )
            prepared.stars[vertex] = star_shape
        else:
            neighbors = star_shape.neighbors
            local_rows, local_cells, spokes = (
                star_shape.local_rows,
                star_shape.local_cells,
                star_shape.spokes,
            )
        local_points = points[local_rows]
        local_metrics = metrics[local_rows]
        local_constant = constant_metric and bool(
            np.all(local_metrics == local_metrics[0])
        )
        periodic_star = (
            (
                np.asarray((vertex, *neighbors), dtype=np.int64),
                np.asarray(star_ids, dtype=np.int64),
                {
                    identifier: index
                    for index, identifier in enumerate((vertex, *neighbors))
                },
            )
            if periodic
            else None
        )
        measured = np.asarray(
            [not protected_edges[_edge_key((vertex, other))] for other in neighbors],
            dtype=np.bool_,
        )
        # Incidence follows lexicographic edge order, so its opposite endpoints
        # have the same order as ``neighbors``. Equal endpoint tensors make
        # the metric length independent of the edge's orientation.
        lengths = (
            np.take(edge_lengths, prepared.incident_edges[vertex])
            if local_constant
            else metric_lengths(local_points, local_metrics, spokes)
        )
        # Raises for a measured spoke without a finite length, before a search
        # is made of the star.
        _band_violations(lengths[measured])
        outside = measured & ((lengths < _LOWER) | (lengths > _UPPER))
        before = prepared.qualities[star_rows]
        # The carrier simplices are coplanar (one facet) or collinear (one
        # curve or segment): the first spans their affine hull.
        corners = projection = None
        basis = _AXES
        if simplices is not None:
            simplex = simplices[0]
            frame = state.source_projections.get(simplex)
            if frame is None:
                corners = source_coordinates[
                    [state.source_vertex_rows[v] for v in simplex]
                ]
                basis, projection = metric_hull_frame(corners)
                for array in (corners, projection, basis):
                    array.setflags(write=False)
                frame = corners, projection, basis
                state.source_projections[simplex] = frame
            corners, projection, basis = frame
        jacobian = None
        if deep and np.any(outside) and not constant_metric:
            # A varying tensor changes with the vertex. Finite differences
            # along the carrier's basis propose directions only; exact ancestry
            # and the full objective below still decide whether a move is
            # admissible.
            scale = max(float(np.max(np.abs(local_points))), 1.0)
            step = sqrt(np.finfo(np.float64).eps) * scale
            derivatives = []
            for direction in basis:
                derivative = None
                for sign in (1.0, -1.0):
                    trial = points[row] + sign * step * direction
                    located = _quantized_relocation(state, simplices, trial, points[row])
                    if located is None:
                        continue
                    trial_stencil, carrier = located
                    moved = carrier - points[row]
                    displacement = float(
                        moved[0] * direction[0]
                        + moved[1] * direction[1]
                        + moved[2] * direction[2]
                    )
                    if displacement == 0.0:
                        continue
                    trial_metric = _source_metric_from_stencil(
                        state.source_metric,
                        state.source_vertex_rows,
                        trial_stencil,
                        source_metric_log=state.source_metric_log,
                    )
                    trial_points = local_points.copy()
                    trial_points[0] = carrier
                    trial_metrics = local_metrics.copy()
                    trial_metrics[0] = trial_metric
                    derivative = (
                        metric_lengths(trial_points, trial_metrics, spokes) - lengths
                    ) / displacement
                    break
                derivatives.append(derivative)
            if derivatives and all(derivative is not None for derivative in derivatives):
                jacobian = np.stack(
                    [value for value in derivatives if value is not None], axis=0
                )
        # Spring forces of the offending spokes and their mean (a mean spring
        # can cancel at a non-unit equilibrium; the individual spokes still
        # give descent directions), the coherent step and gradients of the
        # measured lengths when the tensor varies, and for a deep search the
        # tangents along the band constraints in the carrier's hull. They
        # propose only: acceptance checks the entire star.
        pulls = metric_relocation_pulls(
            local_points,
            local_metrics[0],
            lengths,
            measured,
            deep=deep,
            interior=corners is None,
            basis=basis,
            projector=projection,
            jacobian=jacobian,
        )
        if implicit_only and not periodic:
            tangent_pulls = _implicit_tangent_pulls(state, local_points[0], pulls)
            if tangent_pulls.shape[0]:
                # Retain every existing direction before this additional family.
                # Quantization may leave the tangent plane; source-hull and
                # native zero-set checks remain unchanged.
                pulls = np.concatenate((pulls, tangent_pulls), axis=0)

        # The candidate chain (goals, their screen, exact carriers, tensors
        # and evaluation) is one native sequence on the star-local arrays
        # above; the star, source carrier and metric field are fixed until
        # the caller evicts this search.
        before_quality = float(np.min(before))
        bank = _source_bank(state)
        search = bank.source.relocation_search(
            MetricRelocationStar(
                local_points,
                local_metrics,
                measured,
                lengths,
                before_quality,
                local_cells,
                _DETERMINANT_FLOOR,
                _IMPROVEMENT,
            ),
            pulls,
            None if corners is None or projection is None else (corners[0], projection),
            _bank_simplices(state, bank, simplices),
            deep=deep,
            refine=refine,
            band_only=band_only,
            constant_field=constant_metric,
        )

        def candidate_search() -> Iterator[dict[str, Any]]:
            # The current periodic star is fixed for this search. Build its
            # exact locator only if a carrier reaches this admission, then
            # retain it across subsequent carriers and generator resumptions.
            star_records: _SourceRecords | None = None
            while True:
                group, end = search.next()
                if end is MetricRelocationEnd.EXHAUSTED:
                    return
                if end is not MetricRelocationEnd.GROUP:
                    _report_relocation_candidate(state, bank, group[0], measured, end)
                admitted: list[tuple[Any, dict[str, Any]]] = []
                for candidate in group:
                    carrier = candidate.point
                    stencil = {
                        bank.identifiers[row]: Fraction(weight, 1 << candidate.bits)
                        for row, weight in zip(
                            candidate.rows, candidate.weights, strict=True
                        )
                    }
                    after_band = _band_violations(candidate.lengths[measured])
                    after_quality = candidate.quality
                    current_stencil = None
                    if periodic_star is not None:
                        # A host step of the sequence: the native candidates
                        # of a periodic star are filtered here.
                        if star_records is None:
                            star_records = _source_tetra_inverse(
                                local_points,
                                local_cells,
                                periodic_star[0],
                                periodic_star[1],
                            )
                        current_stencil = _current_star_stencil(
                            local_points,
                            local_cells,
                            *periodic_star,
                            carrier,
                            star_records=star_records,
                        )
                        if current_stencil is None:
                            continue
                    proposal = {
                        "vertex": int(vertex),
                        "point": carrier,
                        "metric": candidate.metric,
                        "stencil": stencil,
                        "current_stencil": current_stencil,
                        "star_cell_ids": star_ids,
                        "star_vertices": frozenset((vertex, *neighbors)),
                        "cavity": frozenset(star_ids),
                        "gain": after_quality - before_quality,
                        "badness": float(np.max(np.maximum(lengths, 1.0 / lengths))),
                        "block": next(iter(star_blocks)),
                        "region": next(iter(star_regions)),
                    }
                    admitted.append(((after_band, -after_quality), proposal))
                # At most the finite binary64 dyadic resolutions for one goal.
                # Stable sorting preserves the previous first-best proposal.
                for _, proposal in sorted(admitted, key=lambda item: item[0]):
                    yield proposal

        return _RelocationSearch(candidate_search())

    candidates: list[dict[str, Any]] = []
    occupied_cells: set[int] = set()
    for vertex in order:
        # A deep search can be costly; don't construct another proposal whose
        # star cannot be batched with one already found. Deferred stars are
        # searched again after the accepted batch, not treated as refusals.
        if deep and not periodic:
            star_ids = prepared.star_cells.get(vertex)
            if star_ids is None:
                star_ids = frozenset(
                    int(current_cell_ids[row]) for row in topology.vertex_cells[vertex]
                )
                prepared.star_cells[vertex] = star_ids
            if not occupied_cells.isdisjoint(star_ids):
                continue
        if vertex not in cache:
            cache[vertex] = propose(vertex)
        search = cache[vertex]
        if search is None:
            continue
        if search.proposal is None:
            search.proposal = next(search.candidates, None)
            if search.proposal is None:
                cache[vertex] = None
                continue
        proposal = search.proposal
        if proposal is not None:
            candidates.append(proposal)
            if deep and not periodic:
                occupied_cells.update(star_ids)
    if not periodic:
        grouped = [(candidate,) for candidate in candidates]
    else:
        by_vertex = {int(candidate["vertex"]): candidate for candidate in candidates}
        grouped = []
        for vertices in _periodic_vertex_orbit_groups(state):
            if not vertices or any(vertex not in by_vertex for vertex in vertices):
                continue
            members = tuple(by_vertex[vertex] for vertex in vertices)
            cavities = [proposal["cavity"] for proposal in members]
            cavity = frozenset().union(*cavities)
            if sum(map(len, cavities)) != len(cavity):
                continue
            grouped.append(members)
    return sorted(
        grouped,
        key=lambda members: (
            -min(float(proposal["badness"]) for proposal in members),
            -min(float(proposal["gain"]) for proposal in members),
            tuple(
                int(state.vertex_target_ids[proposal["vertex"]]) for proposal in members
            ),
        ),
    )


def _current_star_stencil(
    points: np.ndarray,
    star_cells: np.ndarray,
    vertex_ids: np.ndarray,
    star_cell_ids: np.ndarray,
    vertex_rows: dict[int, int],
    carrier: np.ndarray,
    /,
    *,
    star_records: _SourceRecords | None = None,
) -> dict[int, Fraction] | None:
    """Binary64 barycentric stencil of a carrier in the current star that
    reproduces it exactly, else ``None``.

    Precomputed records must describe these same points, cells, and identities.
    """
    if star_records is None:
        star_records = _source_tetra_inverse(
            points, star_cells, vertex_ids, star_cell_ids
        )
    located = _locate_source_stencil(carrier, star_records)
    if located is None:
        return None
    _, stencil = located
    try:
        _binary64_ancestry(stencil)
        realized = _source_stencil_point(points, vertex_rows, stencil)
    except FloatingPointError:
        return None
    return stencil if np.array_equal(realized, carrier) else None


def _relocation_edit(
    state: _NativeState, proposals: Sequence[Mapping[str, Any]], /
) -> TetMeshEdit:
    ordered = sorted(proposals, key=lambda row: state.vertex_target_ids[row["vertex"]])
    star_ids = sorted(
        {int(cell_id) for proposal in ordered for cell_id in proposal["star_cell_ids"]}
    )
    return TetMeshEdit(
        kind=TetMeshEditKind.RELOCATE,
        expected_generation=int(state.snapshot.generation),
        source_id=state.source_identity[0],
        source_revision=state.source_identity[1],
        source_map_id=state.source_identity[2],
        frame_id=state.source_identity[3],
        removed_cell_ids=np.asarray(star_ids, dtype=np.uint64),
        changed_vertex_ids=np.asarray(
            [proposal["vertex"] for proposal in ordered], dtype=np.uint64
        ),
        changed_vertex_points=np.asarray(
            [proposal["point"] for proposal in ordered], dtype=np.float64
        ),
        changed_witness_strata=np.zeros((len(ordered),), dtype=np.int8),
        changed_witness_entities=np.full((len(ordered),), -1, dtype=np.int32),
        changed_witness_parameters=np.zeros((len(ordered), 2), dtype=np.float64),
        changed_witness_deviations=np.zeros((len(ordered),), dtype=np.float64),
        changed_protection_radii=np.zeros((len(ordered),), dtype=np.float64),
        changed_protected_vertices=np.zeros((len(ordered),), dtype=np.int8),
        # A moved vertex no longer lies at its split-edge parameter: clear it
        # explicitly (the omitted witness would be inherited and refused).
        changed_edge_witness_present=np.zeros((len(ordered),), dtype=np.int8),
        changed_edge_witness_endpoints=np.zeros((len(ordered), 2), dtype=np.uint64),
        changed_edge_parameters=np.zeros((len(ordered),), dtype=np.float64),
    )


def _apply_relocation_in_place(
    state: _NativeState, proposals: Sequence[Mapping[str, Any]], /
) -> TetMeshEditStatus | None:
    """Apply the edit of ``_relocation_edit`` in place and update the snapshot.

    ``None`` means the native mesh does not decide the relocation in place;
    nothing changed and the staged edit decides it.
    """
    ordered = sorted(proposals, key=lambda row: state.vertex_target_ids[row["vertex"]])
    vertex_ids = np.asarray([proposal["vertex"] for proposal in ordered], dtype=np.uint64)
    points = np.asarray([proposal["point"] for proposal in ordered], dtype=np.float64)
    star_ids = sorted(
        {int(cell_id) for proposal in ordered for cell_id in proposal["star_cell_ids"]}
    )
    status = state.native.apply_relocation(
        int(state.snapshot.generation),
        vertex_ids,
        points,
        np.asarray(star_ids, dtype=np.uint64),
        source_id=state.source_identity[0],
        source_revision=state.source_identity[1],
        source_map_id=state.source_identity[2],
        frame_id=state.source_identity[3],
    )
    if status is TetMeshEditStatus.APPLIED:
        vertex_row = _state_topology(state).vertex_row
        state.snapshot = relocated_tet_mesh_arrays(
            state.snapshot,
            vertex_ids,
            np.asarray(
                [vertex_row[int(proposal["vertex"])] for proposal in ordered],
                dtype=np.int64,
            ),
            points,
            state.native.implicit_refusals,
        )
    return status


def _apply_relocation_batch(
    state: _NativeState, proposals: Sequence[Mapping[str, Any]], /
) -> TetMeshEditStatus:
    status = (
        _apply_relocation_in_place(state, proposals)
        if state.target_mesh.periodic_topology is None
        else None
    )
    if status is None:
        edit = _relocation_edit(state, proposals)
        try:
            edit, staged = _prepare_periodic_native_edit(
                state, edit, {"proposals": proposals}
            )
        except _PeriodicCandidateRejected:
            return TetMeshEditStatus.REJECTED_TOPOLOGY
        status = state.native.apply_edit(edit)
        if status is not TetMeshEditStatus.APPLIED:
            return status
        _commit_periodic_target(state, staged)
        state.snapshot = state.native.arrays()
    elif status is not TetMeshEditStatus.APPLIED:
        return status
    source_vertex_ids = {
        int(identifier) for identifier in state.source_vertex_ids.tolist()
    }
    for proposal in proposals:
        vertex = int(proposal["vertex"])
        target_id = state.vertex_target_ids[vertex]
        state.vertex_stencils[vertex] = proposal["stencil"]
        state.metric_by_vertex[vertex] = proposal["metric"]
        state.history_vertex[:] = [
            relation
            for relation in state.history_vertex
            if relation[1] != target_id or relation[2] is EntityLineageKind.COLLAPSED_INTO
        ]
        if target_id in source_vertex_ids:
            state.history_vertex.append(
                (target_id, target_id, EntityLineageKind.RELOCATED)
            )
    # A coupled native transaction can move several vertices of the same cell.
    # Its ancestry is recorded once, after every vertex state has been updated.
    affected_cells = {
        int(cell_id) for proposal in proposals for cell_id in proposal["star_cell_ids"]
    }
    for cell_id in sorted(affected_cells, key=state.cell_target_ids.__getitem__):
        ancestry = state.cell_ancestry[cell_id]
        state.history_cells.extend(
            (source_id, state.cell_target_ids[cell_id], EntityLineageKind.RELOCATED)
            for source_id in sorted(ancestry)
        )
        state.changed_source_cell_ids.update(ancestry)
    _refresh_native_periodic_windings(state)
    return status


def _probe_relocation_groups(
    state: _NativeState,
    candidate_groups: list[tuple[dict[str, Any], ...]],
    /,
) -> tuple[TetMeshEditStatus | None, ...]:
    """Native refusal of each orbit alone, or ``None`` where it is not refused.

    Every orbit is probed against the unchanged mesh as the relocation edit
    ``_relocation_edit`` would build for it alone.
    """
    vertex_offsets = [0]
    cell_offsets = [0]
    vertex_ids: list[int] = []
    points: list[Any] = []
    cell_ids: list[int] = []
    for group in candidate_groups:
        ordered = sorted(group, key=lambda row: state.vertex_target_ids[row["vertex"]])
        vertex_ids.extend(int(proposal["vertex"]) for proposal in ordered)
        points.extend(proposal["point"] for proposal in ordered)
        cell_ids.extend(
            sorted(
                {
                    int(cell_id)
                    for proposal in ordered
                    for cell_id in proposal["star_cell_ids"]
                }
            )
        )
        vertex_offsets.append(len(vertex_ids))
        cell_offsets.append(len(cell_ids))
    return state.native.probe_relocations(
        int(state.snapshot.generation),
        np.asarray(vertex_offsets, dtype=np.int64),
        np.asarray(vertex_ids, dtype=np.uint64),
        np.asarray(points, dtype=np.float64).reshape((-1, 3)),
        np.asarray(cell_offsets, dtype=np.int64),
        np.asarray(cell_ids, dtype=np.uint64),
        source_id=state.source_identity[0],
        source_revision=state.source_identity[1],
        source_map_id=state.source_identity[2],
        frame_id=state.source_identity[3],
    )


def _refuse_relocation_group(
    state: _NativeState,
    group: tuple[dict[str, Any], ...],
    status: TetMeshEditStatus | None,
    counters: dict[str, int],
    cache: dict[int, _RelocationSearch | None],
    /,
) -> None:
    """Consume a refused orbit's proposals and count the refusal."""
    for proposal in group:
        vertex = int(proposal["vertex"])
        search = cache[vertex]
        if search is None:
            raise RuntimeError("A refused relocation has no retained candidate search.")
        if state.target_mesh.periodic_topology is not None:
            # Periodic members are an indivisible orbit, not independent searches.
            cache[vertex] = None
        else:
            search.proposal = None
    _count_rejection(status, counters, len(group))


def _apply_relocation_groups(
    state: _NativeState,
    candidate_groups: list[tuple[dict[str, Any], ...]],
    counters: dict[str, int],
    /,
) -> tuple[list[dict[str, Any]], TetMeshEditStatus | None]:
    """One native relocation edit of the given orbits: the moves, or its refusal."""
    proposals = [proposal for group in candidate_groups for proposal in group]
    try:
        status: TetMeshEditStatus | None = _apply_relocation_batch(state, proposals)
    except FloatingPointError:
        status = None
    if status is TetMeshEditStatus.APPLIED:
        counters["relocations"] += len(proposals)
        return proposals, status
    if status is not None:
        _raise_fatal_edit_status(status, "relocation")
    return [], status


def _apply_relocation_partition(
    state: _NativeState,
    candidate_groups: list[tuple[dict[str, Any], ...]],
    counters: dict[str, int],
    cache: dict[int, _RelocationSearch | None],
    /,
) -> list[dict[str, Any]]:
    """Apply disjoint orbits, retaining nonperiodic searches after refusal.

    Off the periodic lift one native call first probes every orbit alone
    against the unchanged mesh; an orbit it refuses is consumed without an
    edit. The remaining orbits are applied in one edit. If that edit is
    refused, they are retried one orbit at a time in the given order: their
    cavities are disjoint, so halving the batch would only spend more native
    attempts to isolate the same orbits.
    """
    if state.target_mesh.periodic_topology is None:
        remaining = []
        for group, refusal in zip(
            candidate_groups,
            _probe_relocation_groups(state, candidate_groups),
            strict=True,
        ):
            if refusal is None:
                remaining.append(group)
            else:
                _raise_fatal_edit_status(refusal, "relocation")
                _refuse_relocation_group(state, group, refusal, counters, cache)
        candidate_groups = remaining
        if not candidate_groups:
            return []
    moved, status = _apply_relocation_groups(state, candidate_groups, counters)
    if status is TetMeshEditStatus.APPLIED:
        return moved
    if len(candidate_groups) == 1:
        _refuse_relocation_group(state, candidate_groups[0], status, counters, cache)
        return []
    for group in candidate_groups:
        applied, status = _apply_relocation_groups(state, [group], counters)
        if status is TetMeshEditStatus.APPLIED:
            moved.extend(applied)
        else:
            _refuse_relocation_group(state, group, status, counters, cache)
    return moved


def _invalidate_relocation_searches(
    caches: _RelocationCaches,
    moved: list[dict[str, Any]] | None = None,
    /,
) -> None:
    """Evict moved closed stars, or every search after topology/periodic edits."""
    if moved is None:
        for cached in caches:
            cached.clear()
    else:
        for proposal in moved:
            for vertex in proposal["star_vertices"]:
                for cached in caches:
                    cached.pop(int(vertex), None)


def _relocation_phase(
    state: _NativeState,
    counters: dict[str, int],
    /,
    *,
    topology_operations: bool = True,
    collapse_cache: dict[tuple[int, int], tuple[dict[str, Any] | None, frozenset[int]]]
    | None = None,
    caches: _RelocationCaches | None = None,
) -> int:
    """Relocate until exhausted or an admitted collapse requires fresh topology.

    Every pass applies a strictly improving move or consumes a distinct member
    of a finite candidate search. A native refusal does not prove exhaustion.
    A moved vertex evicts the searches of its closed star.
    When topology is enabled, the existing collapse phase owns admission;
    a host proposal or a native refusal alone never preempts this workset.
    """
    applied = 0
    # Only actual geometry/topology changes invalidate these worksets. A host
    # collapse proposal is not native admission and must not preempt a search.
    periodic = state.target_mesh.periodic_topology is not None
    if topology_operations and collapse_cache is None:
        collapse_cache = {}
    if caches is None:
        caches = ({}, {}, {}, {})
    elif periodic:
        _invalidate_relocation_searches(caches)
    stage = 0
    prepared: _RelocationPreparation | None = None
    while True:
        cache = caches[stage]
        if prepared is None:
            prepared = _prepare_relocation_geometry(state)
        candidates = _relocation_proposals(
            state,
            cache,
            deep=stage in (1, 2),
            refine=stage == 2,
            active_only=stage < 3,
            band_only=stage < 3,
            prepared=prepared,
        )
        if not candidates:
            if stage == 3:
                return applied
            stage += 1
            if periodic:
                caches[stage].clear()
            continue
        selected = _disjoint_candidates(
            [
                (
                    (
                        -min(float(proposal["badness"]) for proposal in orbit),
                        -min(float(proposal["gain"]) for proposal in orbit),
                        tuple(
                            int(state.vertex_target_ids[proposal["vertex"]])
                            for proposal in orbit
                        ),
                    ),
                    orbit,
                    frozenset().union(*(proposal["cavity"] for proposal in orbit)),
                )
                for orbit in candidates
            ]
        )
        previous_snapshot = state.snapshot
        moved = _apply_relocation_partition(state, selected, counters, cache)
        applied += len(moved)
        if moved and collapse_cache is not None:
            changed_cells = set().union(
                *(proposal["star_cell_ids"] for proposal in moved)
            )
            for edge in tuple(collapse_cache):
                if not collapse_cache[edge][1].isdisjoint(changed_cells):
                    del collapse_cache[edge]
        if moved:
            prepared = _prepare_relocation_geometry(
                state,
                previous=(previous_snapshot, prepared),
                moved_vertices=frozenset(int(proposal["vertex"]) for proposal in moved),
            )
            _invalidate_relocation_searches(caches, None if periodic else moved)
            if topology_operations and not periodic:
                if _collapse_phase(state, counters, cache=collapse_cache):
                    _invalidate_relocation_searches(caches)
                    return applied
        if moved and stage:
            stage = 0


def _native_metric_measurements(
    state: _NativeState, /
) -> tuple[
    tuple[tuple[int, int], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, float],
    float,
]:
    topology = _state_topology(state)
    points, metrics, lengths = _metric_edge_table(state, topology)
    edge_keys = tuple(topology.edge_cells)
    _, protected_by_key = _current_edge_controls(state, topology)
    protected = np.asarray([protected_by_key[edge] for edge in edge_keys], dtype=np.bool_)
    cells = np.asarray(state.snapshot.tetrahedra, dtype=np.int64)
    qualities = metric_tetra_quality(points, metrics, cells)
    regions = np.asarray(state.snapshot.tetrahedron_regions, dtype=np.int64)
    volumes = _region_volumes(points, cells, regions)
    volume_error = _relative_region_error(state.source_region_volumes, volumes)
    return edge_keys, lengths, protected, qualities, volumes, volume_error


def _unit_metric_status(
    state: _NativeState, /, *, lower: float = _LOWER
) -> tuple[bool, int, float, float, float]:
    _, lengths, protected, qualities, _, _ = _native_metric_measurements(state)
    measured = ~protected
    if not np.any(measured):
        measured = np.ones(protected.shape, dtype=np.bool_)
    values = lengths[measured]
    within = (values >= lower) & (values <= _UPPER)
    return (
        bool(np.all(within)),
        int(np.count_nonzero(~within)),
        float(np.min(values, initial=np.inf)),
        float(np.max(values, initial=0.0)),
        float(np.min(qualities, initial=np.inf)),
    )


def _unmet_edges(
    state: _NativeState,
    edge_keys: tuple[tuple[int, int], ...],
    lengths: np.ndarray,
    /,
) -> tuple[TetraMetricUnmetEdge, ...]:
    """Typed witnesses of the given out-of-band edges (``TetraMetricUnmetKind``)."""
    topology = _state_topology(state)
    records = _boundary_records(state)
    unmet = []
    for edge, length in zip(edge_keys, lengths.tolist(), strict=True):
        if length > _UPPER:
            kind = (
                TetraMetricUnmetKind.UNREPRESENTABLE
                if edge in state.unrepresentable_edges
                else TetraMetricUnmetKind.UNIMPROVED
            )
        elif any(
            _source_permits_collapse(state, topology, records, removed, kept)
            for removed, kept in (edge, edge[::-1])
        ):
            kind = TetraMetricUnmetKind.UNIMPROVED
        else:
            kind = TetraMetricUnmetKind.SOURCE_BOUNDARY
        first, second = (state.vertex_target_ids[vertex] for vertex in edge)
        unmet.append(
            TetraMetricUnmetEdge(kind, (min(first, second), max(first, second)), length)
        )
    return tuple(sorted(unmet, key=lambda row: (row.kind.value, row.vertex_ids)))


def _tetra_evidence(
    state: _NativeState,
    counters: Mapping[str, int],
    passes: int,
    status: TetraMetricStatus,
    /,
    *,
    lower: float = _LOWER,
) -> TetraMetricEvidence:
    edge_keys, lengths, protected, qualities, _, volume_error = (
        _native_metric_measurements(state)
    )
    return _metric_evidence(
        status,
        passes,
        counters,
        lengths,
        protected,
        qualities,
        volume_error,
        lambda rows: _unmet_edges(
            state, tuple(edge_keys[row] for row in rows.tolist()), lengths[rows]
        ),
        state.native.implicit_refusals,
        lower=lower,
    )


def _metric_evidence(
    status: TetraMetricStatus,
    passes: int,
    counters: Mapping[str, int],
    lengths: np.ndarray,
    protected: np.ndarray,
    qualities: np.ndarray,
    volume_error: float,
    unmet: Callable[[np.ndarray], tuple[TetraMetricUnmetEdge, ...]],
    implicit_refusals: tuple[int, int] | None,
    /,
    *,
    lower: float,
) -> TetraMetricEvidence:
    """Evidence of a final mesh: ``unmet`` types the out-of-band edge rows.

    Protected edges are not measured unless every edge is protected; a
    nominally complete run whose measured edges leave ``[lower, sqrt(2)]`` is
    stalled, and an infinite region volume error is a numerical stop.
    """
    measured = ~protected
    if not np.any(measured):
        measured = np.ones(protected.shape, dtype=np.bool_)
    values = lengths[measured]
    inside = (values >= lower) & (values <= _UPPER)
    if status is TetraMetricStatus.COMPLETE and not np.all(inside):
        status = TetraMetricStatus.STALLED
    if not np.isfinite(volume_error):
        status = TetraMetricStatus.NUMERICAL_STOP
        volume_error = np.finfo(np.float64).max
    return TetraMetricEvidence(
        status=status,
        passes=int(passes),
        splits=int(counters["splits"]),
        collapses=int(counters["collapses"]),
        flips23=int(counters["flips23"]),
        flips32=int(counters["flips32"]),
        ring_reconnections=int(counters["ring_reconnections"]),
        relocations=int(counters["relocations"]),
        rejected_topology=int(counters["rejected_topology"]),
        rejected_protected=int(counters["rejected_protected"]),
        rejected_uncertain=int(counters["rejected_uncertain"]),
        rejected_invalid=int(counters["rejected_invalid"]),
        measured_edges=int(values.size),
        out_of_range_edges=int(np.count_nonzero(~inside)),
        minimum_metric_length=float(np.min(values)),
        maximum_metric_length=float(np.max(values)),
        minimum_metric_quality=float(np.min(qualities)),
        maximum_region_volume_error=float(volume_error),
        unmet_edges=unmet(np.flatnonzero(measured)[~inside]),
        implicit_refusals=implicit_refusals,
    )


def execute_tetra_metric_adaptation(
    source: CellMeshingResult,
    metric_values: ConvertibleToArray,
    /,
    *,
    cell_classes: ArrayLike,
    facet_classes: ArrayLike,
    edge_classes: ArrayLike,
    protected_edges: ArrayLike,
    fixed_vertices: ArrayLike,
    predicate_mode: PredicateMode,
    topology_operations: bool,
    relocation: bool,
    accepted_minimum_length: float | None = None,
) -> TetraMetricOutcome:
    """Adapt a tetrahedral source toward a unit SPD-metric mesh.

    A nonperiodic source runs the native split/collapse/flip/ring/relocation
    loop (``TetMesh3D.metric_adapt``) in one call; a periodic lift runs the
    host phase schedule. The route accepts an exact P1 source coordinate map,
    keeps source identities
    in the native handle, and returns one generalized topology edit for the
    canonical transfer/certification owner. It never infers source ancestry from
    output samples. ``accepted_minimum_length`` is the request's declared
    completion bound (``MetricMeshAdaptation``); ``None`` is the unit band.
    """
    if not isinstance(predicate_mode, PredicateMode):
        raise TypeError("predicate_mode must be PredicateMode.")
    if not isinstance(topology_operations, bool) or not isinstance(relocation, bool):
        raise TypeError("topology_operations and relocation must be bool.")
    metric_array = np.asarray(metric_values)
    if not isinstance(source, CellMeshingResult):
        raise TypeError("source must be CellMeshingResult.")
    from .providers._native_sources import NativeImplicitSource

    # An implicit authority remains implicit even when its metric source
    # tube is not admitted for this request.
    report = source.certification
    ordinary_source = report is None or not isinstance(
        report.inputs.feature_source, NativeImplicitSource
    )
    state = _prepare_native_state(
        source.mesh,
        source.geometry,
        source.coordinate_contract,
        metric_array,
        cell_classes=cell_classes,
        facet_classes=facet_classes,
        edge_classes=edge_classes,
        protected_edges=protected_edges,
        fixed_vertices=fixed_vertices,
        implicit_source=_implicit_metric_source(source),
        derive_cartesian_seams=ordinary_source and relocation,
    )
    lower = _LOWER if accepted_minimum_length is None else accepted_minimum_length
    if state.periodic_state is not None:
        # periodic lift: pending S4-3D Q6
        return _execute_native_state(
            state,
            predicate_mode,
            topology_operations=topology_operations,
            relocation=relocation,
            lower=lower,
        )
    adaptation = state.native.metric_adapt(
        int(state.snapshot.generation),
        **_metric_adapt_inputs(state),
        lower=lower,
        topology_operations=topology_operations,
        relocation=relocation,
        source_id=state.source_identity[0],
        source_revision=state.source_identity[1],
        source_map_id=state.source_identity[2],
        frame_id=state.source_identity[3],
    )
    snapshot = state.native.arrays()
    # The native edit owns a private working mesh and uses unsigned IDs.
    # Refuse overflow before any narrowing or publication of its result.
    limit = np.iinfo(np.int64).max
    if any(
        int(np.max(bank, initial=0)) > limit
        for bank in (
            snapshot.vertex_ids,
            snapshot.cell_ids,
            adaptation.vertex_ids,
            adaptation.vertex_target_ids,
            adaptation.cell_ids,
            adaptation.cell_target_ids,
            adaptation.ancestry_ids,
            adaptation.vertex_relations,
            adaptation.unrepresentable_edges,
        )
    ):
        raise OverflowError(
            "Native adaptation IDs exceed the stable int64 identifier range."
        )
    edit, target_metric = _adapted_edit(state, adaptation, snapshot)
    evidence = _adapted_evidence(state, adaptation, snapshot, lower)
    return TetraMetricOutcome(edit, target_metric, evidence, state.source_exact_points)


def _metric_adapt_inputs(state: _NativeState, /) -> dict[str, Any]:
    """``TetMesh3D.metric_adapt`` banks of a freshly prepared source mesh.

    Every live vertex is its own source vertex (stencil ``{itself: 1}``) and
    every live cell its own preserved source cell; native, source and target
    IDs coincide.
    """
    source_ids = state.source_vertex_ids
    vertex_order = np.argsort(source_ids, kind="stable")
    vertex_ids = source_ids[vertex_order]
    cell_ids = state.source_cell_ids
    return {
        "source_points": state.source_points,
        "source_metrics": state.source_metric,
        "source_logarithms": state.source_metric_log,
        "source_vertex_ids": source_ids,
        "source_cells": state.source_cell_vertices,
        "exact_source_points": (
            state.source_exact_points if state.source_has_authored_rounding else None
        ),
        "vertex_ids": vertex_ids,
        "stencil_offsets": np.arange(vertex_ids.size + 1, dtype=np.int64),
        "stencil_rows": vertex_order.astype(np.int32),
        "stencil_weights": np.ones(vertex_ids.size, dtype=np.float64),
        "vertex_target_ids": vertex_ids,
        "cell_ids": cell_ids,
        "cell_blocks": state.source_cell_blocks,
        "cell_target_ids": cell_ids,
        "ancestry_offsets": np.arange(cell_ids.size + 1, dtype=np.int64),
        "ancestry_ids": cell_ids,
        "ancestry_kinds": np.full(
            cell_ids.size, int(EntityLineageKind.PRESERVED), dtype=np.int32
        ),
        "protected_edges": np.asarray(
            sorted(state.protected_edge_ids), dtype=np.int64
        ).reshape((-1, 2)),
        "next_ids": (
            state.next_native_vertex,
            state.next_native_cell,
            state.next_target_vertex,
            state.next_target_cell,
        ),
    }


def _id_relations(
    dimension: int,
    sources: np.ndarray,
    targets: np.ndarray,
    kinds: np.ndarray,
    /,
) -> EntityRelations:
    """Distinct single-ID relations ordered by (source, target, kind)."""
    records = np.unique(
        np.stack(
            (
                sources.astype(np.int64),
                targets.astype(np.int64),
                kinds.astype(np.int64),
            ),
            axis=1,
        ).reshape((-1, 3)),
        axis=0,
    )
    return EntityRelations(
        dimension,
        records[:, :1].copy(),
        records[:, 1:2].copy(),
        records[:, 2].astype(np.int32),
    )


def _adapted_edit(
    state: _NativeState,
    adaptation: TetMeshMetricAdaptation,
    snapshot: TetMeshArrays,
    /,
) -> tuple[CellTopologyEdit, np.ndarray]:
    """Topology edit and target metric of a native ``metric_adapt`` result.

    Vertices are ordered by target ID; each block lists its cells by target
    cell ID; stencils name source vertex IDs in ascending order.
    """
    native_ids = adaptation.vertex_ids.astype(np.int64)
    target_ids = adaptation.vertex_target_ids.astype(np.int64)
    rows = np.searchsorted(native_ids, np.asarray(snapshot.vertex_ids, dtype=np.int64))
    points = np.empty((native_ids.size, 3), dtype=np.float64)
    points[rows] = np.asarray(snapshot.points, dtype=np.float64)
    order = np.argsort(target_ids, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    target_vertex_ids = target_ids[order]

    counts = np.diff(adaptation.stencil_offsets)
    term_rows = rank[np.repeat(np.arange(counts.size), counts)]
    term_sources = state.source_vertex_ids[adaptation.stencil_rows]
    term_order = np.lexsort((term_sources, term_rows))
    term_rows = term_rows[term_order]
    term_sources = term_sources[term_order]
    starts = np.zeros(order.size + 1, dtype=np.int64)
    starts[1:] = np.cumsum(np.bincount(term_rows, minlength=order.size))
    columns = np.arange(term_rows.size) - starts[term_rows]
    width = max(int(np.max(counts, initial=0)), 1)
    stencil_sources = np.full((order.size, width), -1, dtype=np.int64)
    stencil_weights = np.zeros((order.size, width), dtype=np.float64)
    stencil_valid = np.zeros((order.size, width), dtype=np.bool_)
    stencil_sources[term_rows, columns] = term_sources
    stencil_weights[term_rows, columns] = adaptation.stencil_weights[term_order]
    stencil_valid[term_rows, columns] = True
    supports = dict(
        zip(
            target_vertex_ids.tolist(),
            (frozenset(part.tolist()) for part in np.split(term_sources, starts[1:-1])),
            strict=True,
        )
    )

    cell_target_ids = adaptation.cell_target_ids.astype(np.int64)
    cell_rows = np.searchsorted(
        adaptation.cell_ids.astype(np.int64),
        np.asarray(snapshot.cell_ids, dtype=np.int64),
    )
    cells = np.empty((cell_target_ids.size, 4), dtype=np.int32)
    cells[cell_rows] = rank[rows[np.asarray(snapshot.tetrahedra, dtype=np.int64)]]
    block_ids: list[np.ndarray] = []
    block_cells: list[np.ndarray] = []
    for block_index in range(len(state.block_families)):
        members = np.flatnonzero(adaptation.cell_blocks == block_index)
        members = members[np.argsort(cell_target_ids[members], kind="stable")]
        block_cells.append(cells[members])
        block_ids.append(cell_target_ids[members])
    corners = target_vertex_ids[cells]
    edge_keys = np.unique(
        np.sort(corners[:, _TETRA_EDGES], axis=2).reshape((-1, 2)), axis=0
    )
    face_keys = np.unique(
        np.sort(corners[:, _TETRA_FACES], axis=2).reshape((-1, 3)), axis=0
    )

    relation_sources = adaptation.vertex_relations[:, 0].astype(np.int64)
    preserved = np.setdiff1d(
        np.intersect1d(state.source_vertex_ids, target_vertex_ids), relation_sources
    )
    owners = np.repeat(
        np.arange(cell_target_ids.size), np.diff(adaptation.ancestry_offsets)
    )
    ancestors = adaptation.ancestry_ids.astype(np.int64)
    relocated = adaptation.cell_relocated[owners] != 0
    relations = (
        _id_relations(
            0,
            np.concatenate((relation_sources, preserved)),
            np.concatenate(
                (adaptation.vertex_relations[:, 1].astype(np.int64), preserved)
            ),
            np.concatenate(
                (
                    adaptation.vertex_relation_kinds,
                    np.full(preserved.size, int(EntityLineageKind.PRESERVED)),
                )
            ),
        ),
        _intermediate_relations(
            state.source_mesh, 1, [tuple(key) for key in edge_keys.tolist()], supports
        ),
        _intermediate_relations(
            state.source_mesh, 2, [tuple(key) for key in face_keys.tolist()], supports
        ),
        _id_relations(
            3,
            np.concatenate((ancestors, ancestors[relocated])),
            np.concatenate((cell_target_ids[owners], cell_target_ids[owners[relocated]])),
            np.concatenate(
                (
                    adaptation.ancestry_kinds[owners],
                    np.full(
                        int(np.count_nonzero(relocated)),
                        int(EntityLineageKind.RELOCATED),
                    ),
                )
            ),
        ),
    )
    edit = CellTopologyEdit(
        coordinates=points[order],
        vertex_global_ids=target_vertex_ids,
        blocks=source_family_blocks(
            state.block_families, tuple(block_cells), tuple(block_ids)
        ),
        stencil_sources=stencil_sources,
        stencil_weights=stencil_weights,
        stencil_valid=stencil_valid,
        relations=relations,
    )
    return edit, adaptation.metrics[order]


def _row_member(rows: np.ndarray, table: np.ndarray, /) -> np.ndarray:
    """Whether each ``(n, 2)`` integer row occurs in the ``(k, 2)`` table."""

    def keys(values: np.ndarray) -> np.ndarray:
        pairs = np.ascontiguousarray(values, dtype=np.int64).reshape((-1, 2))
        return pairs.view(np.dtype((np.void, 16))).reshape((-1,))

    return np.isin(keys(rows), keys(table))


def _adapted_evidence(
    state: _NativeState,
    adaptation: TetMeshMetricAdaptation,
    snapshot: TetMeshArrays,
    lower: float,
    /,
) -> TetraMetricEvidence:
    """Measured evidence of the final native mesh (``_tetra_evidence`` semantics).

    Out-of-band long edges are unrepresentable when the native reports no
    exact split point for them; out-of-band short edges are source-boundary
    edges when the source boundary forbids their collapse both ways.
    """
    native_ids = adaptation.vertex_ids.astype(np.int64)
    snapshot_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    rows = np.searchsorted(native_ids, snapshot_ids)
    targets = adaptation.vertex_target_ids.astype(np.int64)[rows]
    points = np.asarray(snapshot.points, dtype=np.float64)
    metrics = adaptation.metrics[rows]
    cells = np.asarray(snapshot.tetrahedra, dtype=np.int64).reshape((-1, 4))
    edges = np.unique(np.sort(cells[:, _TETRA_EDGES], axis=2).reshape((-1, 2)), axis=0)
    reverse = snapshot_ids[edges[:, 0]] > snapshot_ids[edges[:, 1]]
    measurement_edges = edges
    if np.any(reverse):
        measurement_edges = edges.copy()
        measurement_edges[reverse] = edges[reverse, ::-1]
    lengths = metric_lengths(points, metrics, measurement_edges)
    segments = np.sort(np.asarray(snapshot.segments, dtype=np.int64), axis=1)
    protected = _row_member(
        np.sort(targets[edges], axis=1),
        np.asarray(sorted(state.protected_edge_ids), dtype=np.int64),
    ) | _row_member(
        edges, segments[np.asarray(snapshot.protected_segments, dtype=np.bool_)]
    )
    regions = np.asarray(snapshot.tetrahedron_regions, dtype=np.int64)
    volume_error = _relative_region_error(
        state.source_region_volumes, _region_volumes(points, cells, regions)
    )
    unrepresentable = np.sort(adaptation.unrepresentable_edges.astype(np.int64), axis=1)

    def unmet(outside: np.ndarray) -> tuple[TetraMetricUnmetEdge, ...]:
        records = _snapshot_boundary_records(
            snapshot, state.source_face_group_rows, state.source_segment_group_rows
        )
        protected_vertices = np.asarray(snapshot.protected_vertices, dtype=np.bool_)
        row_by_id = {
            identifier: row for row, identifier in enumerate(snapshot_ids.tolist())
        }
        offsets = adaptation.stencil_offsets

        def admitted(removed: int, kept: int) -> bool:
            kept_row = int(rows[row_by_id[kept]])
            support = state.source_vertex_ids[
                adaptation.stencil_rows[offsets[kept_row] : offsets[kept_row + 1]]
            ]
            return _collapse_admitted(
                state,
                records,
                removed,
                kept,
                removed_protected=bool(protected_vertices[row_by_id[removed]]),
                kept_protected=bool(protected_vertices[row_by_id[kept]]),
                kept_support=frozenset(support.tolist()),
            )

        pairs = np.sort(targets[edges[outside]], axis=1)
        split_refused = _row_member(pairs, unrepresentable)
        witnesses = []
        for edge, pair, length, refused in zip(
            snapshot_ids[edges[outside]].tolist(),
            pairs.tolist(),
            lengths[outside].tolist(),
            split_refused.tolist(),
            strict=True,
        ):
            if length > _UPPER:
                kind = (
                    TetraMetricUnmetKind.UNREPRESENTABLE
                    if refused
                    else TetraMetricUnmetKind.UNIMPROVED
                )
            elif admitted(edge[0], edge[1]) or admitted(edge[1], edge[0]):
                kind = TetraMetricUnmetKind.UNIMPROVED
            else:
                kind = TetraMetricUnmetKind.SOURCE_BOUNDARY
            witnesses.append(TetraMetricUnmetEdge(kind, (pair[0], pair[1]), length))
        return tuple(sorted(witnesses, key=lambda row: (row.kind.value, row.vertex_ids)))

    match adaptation.status:
        case 0:
            status = TetraMetricStatus.COMPLETE
        case 1:
            status = TetraMetricStatus.STALLED
        case 2:
            status = TetraMetricStatus.NUMERICAL_STOP
        case other:
            raise ValueError(f"The native metric adaptation reported status {other}.")
    return _metric_evidence(
        status,
        adaptation.passes,
        dict(zip(_EditCounters._fields, adaptation.counters.tolist(), strict=True)),
        lengths,
        protected,
        metric_tetra_quality(points, metrics, cells),
        volume_error,
        unmet,
        state.native.implicit_refusals,
        lower=lower,
    )


def _execute_native_state(
    state: _NativeState,
    predicate_mode: PredicateMode,
    /,
    *,
    topology_operations: bool,
    relocation: bool,
    lower: float,
) -> TetraMetricOutcome:
    """Run the host phase schedule on one native owner (periodic lift only).

    ``lower`` widens only the completion criterion: every phase still targets
    the unit band, and the evidence counts unmet edges against ``lower``.
    """
    counters = dict(zip(_EditCounters._fields, _EditCounters(), strict=True))
    passes = 0
    status = TetraMetricStatus.COMPLETE
    negative_rings: dict[tuple[int, int], tuple[Any, ...]] = {}
    replacement_memos = {
        kind: _ReplacementMemo()
        for kind in (
            TetMeshEditKind.FLIP23,
            TetMeshEditKind.FLIP32,
            TetMeshEditKind.RECONNECT,
        )
    }
    collapse_cache: dict[
        tuple[int, int], tuple[dict[str, Any] | None, frozenset[int]]
    ] = {}
    # Exhausted searches survive unchanged outer passes; actual topology and
    # closed-star changes invalidate them at their owning commit boundaries.
    relocation_caches: _RelocationCaches = ({}, {}, {}, {}) if relocation else ()
    while True:
        converged, _, _, _, _ = _unit_metric_status(state, lower=lower)
        if converged:
            status = TetraMetricStatus.COMPLETE
            break
        passes += 1
        applied = 0
        if topology_operations:
            split = _split_phase(state, counters, one_batch=True)
            applied += split
            applied += _collapse_phase(state, counters, cache=collapse_cache)
            if split:
                # Revisit the remaining long edges after coarsening the short
                # spokes created by this disjoint split batch. Both edits
                # strictly decrease the same objective; there is no pass cap.
                _invalidate_relocation_searches(relocation_caches)
                continue
            for kind, name in (
                (TetMeshEditKind.FLIP23, "flips23"),
                (TetMeshEditKind.FLIP32, "flips32"),
                (TetMeshEditKind.RECONNECT, "ring_reconnections"),
            ):
                applied += _replacement_phase(
                    state,
                    kind,
                    name,
                    counters,
                    predicate_mode,
                    negative_rings=negative_rings
                    if state.target_mesh.periodic_topology is None
                    else None,
                    memo=replacement_memos[kind],
                )
            if applied:
                _invalidate_relocation_searches(relocation_caches)
        if relocation:
            applied += _relocation_phase(
                state,
                counters,
                topology_operations=topology_operations,
                collapse_cache=collapse_cache,
                caches=relocation_caches,
            )
        if relocation and applied == 0:
            from ._tetra_coupled import _coupled_relocation_phase

            applied = _coupled_relocation_phase(state, counters)
            if applied:
                _invalidate_relocation_searches(relocation_caches)
                collapse_cache.clear()
        if applied == 0:
            status = TetraMetricStatus.STALLED
            break
    edit, target_metric = _assemble_native_edit(state)
    evidence = _tetra_evidence(state, counters, passes, status, lower=lower)
    return TetraMetricOutcome(edit, target_metric, evidence, state.source_exact_points)


def _intermediate_relations(
    source: CellMesh,
    dimension: int,
    target_keys: list[tuple[int, ...]],
    stencils: Mapping[int, Iterable[int]],
    /,
) -> EntityRelations:
    source_rows = entity_keys(source, dimension)
    source_keys = [
        tuple(int(value) for value in row if int(value) >= 0)
        for row in source_rows.tolist()
    ]
    source_set = set(source_keys)
    width = dimension + 1
    rows_by_vertex: dict[int, set[int]] | None = None
    relations: set[tuple[tuple[int, ...], tuple[int, ...], EntityLineageKind]] = set()
    for target_key in target_keys:
        if target_key in source_set:
            relations.add((target_key, target_key, EntityLineageKind.PRESERVED))
        support: set[int] = set()
        for vertex in target_key:
            support.update(stencils[int(vertex)])
        if not support or len(support) > width:
            continue
        # Full simplex support fits a source row of this width only by
        # equality. Smaller supports still need every incident source row.
        if source_rows.shape[1] <= width and len(support) == width:
            key = tuple(sorted(support))
            if key != target_key and key in source_set:
                relations.add((key, target_key, EntityLineageKind.REFINED_FROM))
            continue
        if rows_by_vertex is None:
            rows_by_vertex = defaultdict(set)
            for row, key in enumerate(source_keys):
                for vertex in key:
                    rows_by_vertex[vertex].add(row)
        candidates: set[int] | None = None
        for vertex in support:
            rows = rows_by_vertex.get(vertex, set())
            candidates = set(rows) if candidates is None else candidates & rows
            if not candidates:
                break
        for row in candidates or ():
            source_key = source_keys[row]
            if source_key != target_key and support <= set(source_key):
                relations.add((source_key, target_key, EntityLineageKind.REFINED_FROM))
    ordered = sorted(relations, key=lambda row: (row[0], row[1], int(row[2])))
    return _relation_records(
        dimension,
        [row[0] for row in ordered],
        [row[1] for row in ordered],
        [row[2] for row in ordered],
    )


def _assemble_native_edit(state: _NativeState, /) -> tuple[CellTopologyEdit, np.ndarray]:
    snapshot = state.snapshot
    native_vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    vertex_order = np.argsort(
        np.asarray(
            [
                state.vertex_target_ids[int(identifier)]
                for identifier in native_vertex_ids
            ],
            dtype=np.int64,
        ),
        kind="stable",
    )
    target_vertex_ids = np.asarray(
        [state.vertex_target_ids[int(native_vertex_ids[row])] for row in vertex_order],
        dtype=np.int64,
    )
    target_coordinates = np.asarray(snapshot.points, dtype=np.float64)[vertex_order]
    target_row_by_native = {
        int(native_vertex_ids[row]): target_row
        for target_row, row in enumerate(vertex_order.tolist())
    }
    target_stencils = [
        state.vertex_stencils[int(native_vertex_ids[row])]
        for row in vertex_order.tolist()
    ]
    target_stencils_by_id = {
        int(target_id): state.vertex_stencils[int(native_vertex_ids[native_row])]
        for target_id, native_row in zip(
            target_vertex_ids.tolist(), vertex_order.tolist(), strict=True
        )
    }
    width = max((len(stencil) for stencil in target_stencils), default=1)
    stencil_sources = np.full((len(target_stencils), width), -1, dtype=np.int64)
    stencil_weights = np.zeros((len(target_stencils), width), dtype=np.float64)
    stencil_valid = np.zeros((len(target_stencils), width), dtype=np.bool_)
    for row, stencil in enumerate(target_stencils):
        if not stencil:
            continue
        identifiers, weights = _binary64_ancestry(stencil)
        stencil_sources[row, : identifiers.size] = identifiers.astype(np.int64)
        stencil_weights[row, : weights.size] = weights
        stencil_valid[row, : weights.size] = True

    cell_ids = np.asarray(snapshot.cell_ids, dtype=np.int64)
    local_cells = np.asarray(snapshot.tetrahedra, dtype=np.int64)
    cells_native = native_vertex_ids[local_cells]
    cells_target = np.asarray(
        [[target_row_by_native[int(vertex)] for vertex in cell] for cell in cells_native],
        dtype=np.int32,
    )
    target_cell_ids = np.asarray(
        [state.cell_target_ids[int(identifier)] for identifier in cell_ids],
        dtype=np.int64,
    )
    block_ids: list[np.ndarray] = []
    block_cells: list[np.ndarray] = []
    for block_index in range(len(state.block_families)):
        members = np.flatnonzero(
            np.asarray(
                [
                    state.cell_blocks[int(identifier)] == block_index
                    for identifier in cell_ids
                ],
                dtype=np.bool_,
            )
        )
        order = members[np.argsort(target_cell_ids[members], kind="stable")]
        block_cells.append(cells_target[order])
        block_ids.append(target_cell_ids[order])
    target_edge_keys = sorted(
        {
            tuple(sorted(int(target_vertex_ids[vertex]) for vertex in cell[list(edge)]))
            for cell in cells_target
            for edge in _TETRA_EDGES.tolist()
        }
    )
    target_face_keys = sorted(
        {
            tuple(sorted(int(target_vertex_ids[vertex]) for vertex in cell[face]))
            for cell in cells_target
            for face in _TETRA_FACES.tolist()
        }
    )
    vertex_relations: set[tuple[tuple[int, ...], tuple[int, ...], EntityLineageKind]] = {
        ((int(source_id),), (int(target_id),), kind)
        for source_id, target_id, kind in state.history_vertex
    }
    source_vertex_ids = set(int(value) for value in state.source_vertex_ids.tolist())
    target_vertex_id_set = set(int(value) for value in target_vertex_ids.tolist())
    for identifier in source_vertex_ids & target_vertex_id_set:
        if not any(source_key == (identifier,) for source_key, _, _ in vertex_relations):
            vertex_relations.add(
                ((identifier,), (identifier,), EntityLineageKind.PRESERVED)
            )
    vertex_records = sorted(
        vertex_relations, key=lambda row: (row[0], row[1], int(row[2]))
    )
    live_target_cell_ids = set(int(value) for value in target_cell_ids.tolist())
    relations = (
        _relation_records(
            0,
            [row[0] for row in vertex_records],
            [row[1] for row in vertex_records],
            [row[2] for row in vertex_records],
        ),
        _intermediate_relations(
            state.source_mesh, 1, target_edge_keys, target_stencils_by_id
        ),
        _intermediate_relations(
            state.source_mesh, 2, target_face_keys, target_stencils_by_id
        ),
        _relation_records(
            3,
            [
                (source_id,)
                for native_id in cell_ids.tolist()
                for source_id in state.cell_ancestry[int(native_id)]
            ]
            + [
                (source_id,)
                for source_id, target_id, _ in state.history_cells
                if target_id in live_target_cell_ids
            ],
            [
                (state.cell_target_ids[int(native_id)],)
                for native_id in cell_ids.tolist()
                for _ in state.cell_ancestry[int(native_id)]
            ]
            + [
                (target_id,)
                for _, target_id, _ in state.history_cells
                if target_id in live_target_cell_ids
            ],
            [
                state.cell_ancestry[int(native_id)][source_id]
                for native_id in cell_ids.tolist()
                for source_id in state.cell_ancestry[int(native_id)]
            ]
            + [
                kind
                for _, target_id, kind in state.history_cells
                if target_id in live_target_cell_ids
            ],
        ),
    )
    edit = CellTopologyEdit(
        coordinates=target_coordinates,
        vertex_global_ids=target_vertex_ids,
        blocks=source_family_blocks(
            state.block_families, tuple(block_cells), tuple(block_ids)
        ),
        stencil_sources=stencil_sources,
        stencil_weights=stencil_weights,
        stencil_valid=stencil_valid,
        relations=relations,
    )
    target_metric = np.asarray(
        [state.metric_by_vertex[int(native_vertex_ids[row])] for row in vertex_order],
        dtype=np.float64,
    )
    return edit, target_metric


__all__ = [
    "TetraMetricEvidence",
    "TetraMetricOutcome",
    "TetraMetricStatus",
    "TetraMetricUnmetEdge",
    "TetraMetricUnmetKind",
    "execute_tetra_metric_adaptation",
]
