#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-aware SPD metric adaptation kernels for tetrahedral cell meshes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
from itertools import combinations
from typing import Any, NamedTuple

import equinox as eqx
import numpy as np
from numpy.typing import ArrayLike

from .._geometry_predicates import orient3d, PredicateMode, PredicateSign
from .._meshcore import (
    PlcBoundaryPolicy,
    TetMesh3D,
    TetMeshArrays,
    TetMeshEdit,
    TetMeshEditKind,
    TetMeshEditStatus,
    TetMeshPeriodicState,
    TetMeshSourceComplex,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._cell_mesh import CellBlock
from ..linalg._exact import solve_exact
from ..typing import ConvertibleToArray
from ._lineage import EntityLineageKind
from ._metric import _tensor_array, _tensor_properties
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
# A relocation tries steps 2**-k of its pull for k = 0, ..., bits: the same
# positional resolution as a split point.
_RELOCATION_STEP_BITS = _SPLIT_PARAMETER_BITS
# kMetricDeterminantFloor of native quality improvement (improve3d.hpp): below
# this relative determinant |det E| / max_v prod |e| a cell's binary64 metric
# determinant det(E^T E) is not certainly positive. No edit creates such a cell.
_DETERMINANT_FLOOR = 2.0**-22
_TETRA_EDGE_CORNERS = tuple(combinations(range(4), 2))
_TETRA_FACE_CORNERS = ((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2))
_TETRA_EDGES = np.asarray(_TETRA_EDGE_CORNERS, dtype=np.int32)
_TETRA_FACES = np.asarray(_TETRA_FACE_CORNERS, dtype=np.int32)
_TETRA_EDGES.setflags(write=False)
_TETRA_FACES.setflags(write=False)


def _edge_key(vertices: Iterable[int], /) -> tuple[int, int]:
    values = tuple(int(value) for value in vertices)
    if len(values) != 2:
        raise ValueError("A tetra edge key must contain exactly two vertex IDs.")
    first, second = values
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
    """Actual native operation counts and final SPD-metric measurements."""

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
        from .._fingerprint import canonical_fingerprint

        self.evidence_id = canonical_fingerprint(
            {
                "kind": "tetra-metric-evidence",
                "status": status.value,
                "counts": counts,
                "measurements": values,
                "unmet": tuple(
                    (edge.kind.value, edge.vertex_ids, edge.metric_length)
                    for edge in unmet
                ),
            }
        )

    @property
    def converged(self) -> bool:
        return self.status is TetraMetricStatus.COMPLETE


class TetraMetricOutcome(NamedTuple):
    """Numerical topology edit, vertex metric, and truthful termination evidence."""

    edit: CellTopologyEdit
    metric: np.ndarray
    evidence: TetraMetricEvidence


class _TetTopology(NamedTuple):
    edge_cells: dict[tuple[int, int], tuple[int, ...]]
    face_cells: dict[tuple[int, int, int], tuple[int, ...]]
    vertex_cells: dict[int, tuple[int, ...]]
    cell_vertex_ids: np.ndarray
    vertex_row: dict[int, int]
    cell_row: dict[int, int]
    facet_row: dict[tuple[int, int, int], int]
    segment_row: dict[tuple[int, int], int]


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


@dataclass(slots=True)
class _NativeState:
    native: TetMesh3D
    snapshot: TetMeshArrays
    source_mesh: CellMesh
    target_mesh: CellMesh
    source_metric: np.ndarray
    periodic_vertex_windings: dict[int, np.ndarray]
    source_cell_ids: np.ndarray
    source_vertex_ids: np.ndarray
    source_vertex_rows: dict[int, int]
    source_cell_vertices: np.ndarray
    source_cell_regions: np.ndarray
    source_cell_blocks: np.ndarray
    source_records: tuple[
        tuple[
            int,
            tuple[int, ...],
            tuple[tuple[Fraction, ...], ...],
            tuple[Fraction, ...],
        ],
        ...,
    ]
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
    face_classes: dict[tuple[int, int, int], int]
    source_face_ids: dict[tuple[int, int, int], int]
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


class _CandidateTetraSnapshot(NamedTuple):
    points: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_regions: np.ndarray
    vertex_ids: np.ndarray
    cell_ids: np.ndarray


def _metric_lengths(
    points: np.ndarray, metrics: np.ndarray, edges: np.ndarray, /
) -> np.ndarray:
    """Host Riemannian edge lengths, the formula of ``metric_edge_lengths``.

    The log-mean ``(l_p - l_q) / log(l_p / l_q)`` of the endpoint lengths
    (Alauzet 2010) is evaluated in NumPy: the host route measures small,
    changing edge sets whose shapes would otherwise each compile a kernel.
    """
    if edges.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    delta = points[edges[:, 1]] - points[edges[:, 0]]
    first = np.sqrt(np.einsum("ei,eij,ej->e", delta, metrics[edges[:, 0]], delta))
    second = np.sqrt(np.einsum("ei,eij,ej->e", delta, metrics[edges[:, 1]], delta))
    ratio = second / np.where(first > 0.0, first, 1.0)
    nearly_equal = np.abs(ratio - 1.0) <= 1.0e-6
    safe_ratio = np.where(nearly_equal, 2.0, ratio)
    logarithmic = (first - second) / np.log(1.0 / safe_ratio)
    return np.where(nearly_equal, 0.5 * (first + second), logarithmic)


def _band_violations(lengths: np.ndarray, /) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Max-first quantized long and short distances ``|log(l / band edge)|``.

    Each violation is counted in quanta ``_VIOLATION_QUANTUM`` rounded up, so
    any violation is at least one quantum and an in-band edge is none.
    """
    values = np.asarray(lengths, dtype=np.float64).reshape((-1,))

    def quanta(violation: np.ndarray) -> tuple[int, ...]:
        counts = np.maximum(np.ceil(violation / _VIOLATION_QUANTUM), 1.0)
        return tuple(sorted((int(value) for value in counts.tolist()), reverse=True))

    return (
        quanta(np.log(values[values > _UPPER] / _UPPER)),
        quanta(np.log(_LOWER / values[values < _LOWER])),
    )


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


def _log_euclidean_mean(tensors: np.ndarray, weights: np.ndarray, /) -> np.ndarray:
    """Host ``exp(sum_k w_k log M_k)`` of ``interpolate_mesh_metric``."""
    values, vectors = np.linalg.eigh(tensors)
    if not np.all(values > 0.0):
        raise FloatingPointError("Log-Euclidean interpolation requires SPD metrics.")
    logarithm = np.einsum("...ij,...j,...kj->...ik", vectors, np.log(values), vectors)
    mean = np.einsum("...k,...kij->...ij", weights, logarithm)
    mean_values, mean_vectors = np.linalg.eigh(0.5 * (mean + np.swapaxes(mean, -1, -2)))
    return np.einsum(
        "...ij,...j,...kj->...ik", mean_vectors, np.exp(mean_values), mean_vectors
    )


def _tetra_quality(
    points: np.ndarray, metrics: np.ndarray, cells: np.ndarray, /
) -> np.ndarray:
    """Metric mean-ratio quality in the log-Euclidean barycentric cell metric.

    Only ``det`` of the cell metric enters, and the log-Euclidean mean has
    ``log det = mean_k log det M_k``.
    """
    if cells.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    log_det = np.mean(np.log(np.linalg.det(metrics[cells])), axis=1)
    edges = cells[:, _TETRA_EDGES].reshape((-1, 2))
    lengths = _metric_lengths(points, metrics, edges).reshape((cells.shape[0], 6))
    corners = points[cells]
    jacobian = np.swapaxes(corners[:, 1:] - corners[:, :1], -1, -2)
    volume = np.abs(np.linalg.det(jacobian)) / 6.0
    metric_volume = volume * np.exp(0.5 * log_det)
    denominator = np.sum(lengths * lengths, axis=1)
    return np.where(
        (metric_volume > 0.0) & (denominator > 0.0),
        12.0 * np.power(3.0 * metric_volume, 2.0 / 3.0) / denominator,
        0.0,
    )


def _relative_determinants(corners: np.ndarray, /) -> np.ndarray:
    """Native ``relative_determinant``: ``|det E|`` over the largest product of
    the three edge lengths at a vertex, for ``(n, 4, 3)`` cell corners."""
    values = np.asarray(corners, dtype=np.float64).reshape((-1, 4, 3))
    determinants = np.abs(np.linalg.det(values[:, 1:] - values[:, :1]))
    lengths = np.linalg.norm(values[:, :, None, :] - values[:, None, :, :], axis=-1)
    lengths[:, np.arange(4), np.arange(4)] = 1.0
    products = np.max(np.prod(lengths, axis=2), axis=1)
    return np.where(
        np.isfinite(determinants) & np.isfinite(products) & (products > 0.0),
        determinants / np.where(products > 0.0, products, 1.0),
        0.0,
    )


def _clears_determinant_floor(corners: np.ndarray, /) -> bool:
    """Whether every created cell clears the improvement determinant floor."""
    return bool(np.all(_relative_determinants(corners) > _DETERMINANT_FLOOR))


def _floor_oriented(corners: np.ndarray, /) -> bool:
    """Whether cells are positively oriented above the determinant floor.

    A relative determinant above ``_DETERMINANT_FLOOR`` (2^-22) exceeds the
    binary64 error bound of the evaluated determinant (a small multiple of
    ``2^-53`` times the Hadamard bound, itself at most the largest vertex edge
    product) by orders of magnitude, so the float sign is the exact orientation.
    """
    values = np.asarray(corners, dtype=np.float64).reshape((-1, 4, 3))
    return bool(
        np.all(np.linalg.det(values[:, 1:] - values[:, :1]) > 0.0)
    ) and _clears_determinant_floor(values)


def _metric_midpoint(
    first: np.ndarray,
    second: np.ndarray,
    first_metric: np.ndarray,
    second_metric: np.ndarray,
    /,
) -> tuple[float, np.ndarray]:
    delta = second - first
    first_squared = float(delta @ first_metric @ delta)
    second_squared = float(delta @ second_metric @ delta)
    if (
        not np.isfinite(first_squared + second_squared)
        or min(first_squared, second_squared) <= 0.0
    ):
        return 0.5, 0.5 * (first + second)
    ratio = float(np.sqrt(second_squared / first_squared))
    delta_ratio = ratio - 1.0
    parameter = (
        0.5
        if ratio == 1.0
        else float(np.log1p(0.5 * delta_ratio) / np.log1p(delta_ratio))
    )
    return parameter, first + parameter * delta


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


def _topology(snapshot: TetMeshArrays, /) -> _TetTopology:
    vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    cell_ids = np.asarray(snapshot.cell_ids, dtype=np.int64)
    local_cells = np.asarray(snapshot.tetrahedra, dtype=np.int64).reshape((-1, 4))
    cells = np.asarray(vertex_ids[local_cells], dtype=np.int64).reshape((-1, 4))
    edge_cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    face_cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    vertex_cells: dict[int, list[int]] = defaultdict(list)
    for cell_row in range(cells.shape[0]):
        vertices = tuple(int(cells[cell_row, corner]) for corner in range(4))
        for vertex in vertices:
            vertex_cells[vertex].append(cell_row)
        for first, second in _TETRA_EDGE_CORNERS:
            edge_cells[_edge_key((vertices[first], vertices[second]))].append(cell_row)
        for first, second, third in _TETRA_FACE_CORNERS:
            face_cells[
                _face_key((vertices[first], vertices[second], vertices[third]))
            ].append(cell_row)
    facet_keys = np.sort(vertex_ids[np.asarray(snapshot.faces, dtype=np.int64)], axis=1)
    segment_keys = np.sort(
        vertex_ids[np.asarray(snapshot.segments, dtype=np.int64)], axis=1
    )
    return _TetTopology(
        {key: tuple(rows) for key, rows in sorted(edge_cells.items())},
        {key: tuple(rows) for key, rows in sorted(face_cells.items())},
        {key: tuple(rows) for key, rows in sorted(vertex_cells.items())},
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
    """Max-min metric-quality triangulation of a complete finite edge ring.

    A ring triangle ``(a, b, c)`` in ring order yields ``(e0, a, b, c)`` and
    ``(e1, a, c, b)``; one orientation of the ring orders every such cell
    positively. Both ring orientations are tried and the cells are returned
    positively ordered.
    """
    count = len(ring)
    if count < 3:
        return None
    best_result: tuple[tuple[tuple[int, int, int, int], ...], float] | None = None
    for oriented_ring in (ring, ring[::-1]):
        result = _ring_triangulation(points, metrics, topology, edge, oriented_ring, mode)
        if result is not None and (best_result is None or result[1] > best_result[1]):
            best_result = result
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
    for index in range(count - 1):
        best[index, index + 1] = np.inf
    for span in range(2, count):
        for first in range(count - span):
            last = first + span
            best_score = -np.inf
            best_split = -1
            for pivot in range(first + 1, last):
                left_score = best[first, pivot]
                right_score = best[pivot, last]
                if left_score == -np.inf or right_score == -np.inf:
                    continue
                a, b, c = ring[first], ring[pivot], ring[last]
                # Created cells must clear the determinant floor, which also
                # certifies their binary64 orientation (``_floor_oriented``).
                oriented = np.asarray(
                    [
                        [topology.vertex_row[vertex] for vertex in cell]
                        for cell in ((edge[0], a, b, c), (edge[1], a, c, b))
                    ],
                    dtype=np.int64,
                )
                if not _floor_oriented(points[oriented]):
                    continue
                quality = _tetra_quality(points, metrics, oriented)
                if not np.all(np.isfinite(quality)) or np.any(quality <= 0.0):
                    continue
                score = min(left_score, right_score, float(np.min(quality)))
                if score > best_score:
                    best_score = score
                    best_split = pivot
            best[first, last] = best_score
            split[first, last] = best_split
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
    merged: dict[int, Fraction] = defaultdict(Fraction)
    for source, weight in zip(sources, weights, strict=True):
        if weight:
            merged[int(source)] += weight
    return {source: weight for source, weight in sorted(merged.items()) if weight}


def _binary64_ancestry(stencil: dict[int, Fraction], /) -> tuple[np.ndarray, np.ndarray]:
    identifiers = np.asarray(sorted(stencil), dtype=np.uint64)
    weights = np.asarray(
        [float(stencil[int(identifier)]) for identifier in identifiers],
        dtype=np.float64,
    )
    if any(
        Fraction(float(weight)) != stencil[int(identifier)]
        for identifier, weight in zip(identifiers.tolist(), weights.tolist(), strict=True)
    ):
        raise FloatingPointError(
            "Exact source ancestry is not representable in binary64."
        )
    if sum((Fraction(float(weight)) for weight in weights.tolist()), Fraction(0)) != 1:
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
) -> np.ndarray:
    identifiers, weights = _binary64_ancestry(stencil)
    rows = np.asarray(
        [source_vertex_rows[int(identifier)] for identifier in identifiers],
        dtype=np.int64,
    )
    values = _log_euclidean_mean(source_metric[rows], weights)
    if values.shape != (3, 3) or np.any(~np.isfinite(values)):
        raise FloatingPointError(
            "Source metric interpolation is not a finite SPD tensor."
        )
    return values


def _source_tetra_inverse(
    points: np.ndarray,
    cells: np.ndarray,
    global_vertex_ids: np.ndarray,
    cell_ids: np.ndarray,
    /,
) -> tuple[
    tuple[
        int,
        tuple[int, ...],
        tuple[tuple[Fraction, ...], ...],
        tuple[Fraction, ...],
    ],
    ...,
]:
    records = []
    for row, vertices in enumerate(cells):
        coordinates = [
            tuple(Fraction(float(value)) for value in points[int(vertex)])
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
        records.append(
            (
                int(cell_ids[row]),
                tuple(int(global_vertex_ids[int(vertex)]) for vertex in vertices),
                inverse,
                origin,
            )
        )
    return tuple(sorted(records, key=lambda row: row[0]))


def _locate_source_weights(
    point: tuple[Fraction, Fraction, Fraction],
    source_records: tuple[
        tuple[
            int, tuple[int, ...], tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]
        ],
        ...,
    ],
    /,
) -> tuple[int, tuple[int, ...], tuple[Fraction, ...]] | None:
    """First source cell (by ID) containing an exact point, with its exact
    barycentric weights in the cell's vertex order."""
    for cell_id, vertices, inverse, origin in source_records:
        delta = tuple(point[axis] - origin[axis] for axis in range(3))
        tail = tuple(
            sum((inverse[row][axis] * delta[axis] for axis in range(3)), Fraction(0))
            for row in range(3)
        )
        weights = (Fraction(1) - sum(tail, Fraction(0)), *tail)
        if all(weight >= 0 for weight in weights):
            return cell_id, vertices, weights
    return None


def _locate_source_stencil(
    point: np.ndarray,
    source_records: tuple[
        tuple[
            int, tuple[int, ...], tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]
        ],
        ...,
    ],
    /,
) -> tuple[int, dict[int, Fraction]] | None:
    query = tuple(Fraction(float(value)) for value in point)
    located = _locate_source_weights((query[0], query[1], query[2]), source_records)
    if located is None:
        return None
    cell_id, vertices, weights = located
    return cell_id, _fraction_stencil(vertices, weights)


def _quantized_weights(
    weights: tuple[Fraction, ...], bits: int, /
) -> tuple[Fraction, ...]:
    """Nonnegative weights rounded to the dyadic grid ``2**-bits``.

    The partial sums are rounded (monotone, the last one exactly 1), so the
    quantized weights are nonnegative, sum exactly to one, and each is a binary64
    value for ``bits <= 53``: the point stays in the entity spanned by the support.
    """
    scale = 1 << bits
    result: list[Fraction] = []
    previous = 0
    running = Fraction(0)
    for weight in weights[:-1]:
        running += weight
        current = min(max(round(running * scale), previous), scale)
        result.append(Fraction(current - previous, scale))
        previous = current
    result.append(Fraction(scale - previous, scale))
    return tuple(result)


def _entity_weights(
    corners: tuple[tuple[Fraction, ...], ...],
    point: tuple[Fraction, Fraction, Fraction],
    /,
) -> tuple[Fraction, ...] | None:
    """Exact barycentric weights of the orthogonal projection of ``point`` onto
    a source edge (two corners) or triangle (three); ``None`` outside it."""
    origin = corners[0]
    axes = [[corner[axis] - origin[axis] for axis in range(3)] for corner in corners[1:]]
    offset = [point[axis] - origin[axis] for axis in range(3)]

    def dot(first: list[Fraction], second: list[Fraction]) -> Fraction:
        return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))

    gram = [[dot(first, second) for second in axes] for first in axes]
    right = [dot(axis, offset) for axis in axes]
    if len(axes) == 1:
        if gram[0][0] == 0:
            return None
        tail = (right[0] / gram[0][0],)
    else:
        determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
        if determinant == 0:
            return None
        tail = (
            (right[0] * gram[1][1] - right[1] * gram[0][1]) / determinant,
            (right[1] * gram[0][0] - right[0] * gram[1][0]) / determinant,
        )
    weights = (Fraction(1) - sum(tail, Fraction(0)), *tail)
    return weights if all(weight >= 0 for weight in weights) else None


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


def _coplanar_face_groups(
    points: np.ndarray,
    faces: np.ndarray,
    sides: list[tuple[Any, ...]],
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

    def exact(vertex: int) -> list[Fraction]:
        return [Fraction(float(value)) for value in points[vertex]]

    for key, incident in uses.items():
        if len(incident) != 2 or key in segments:
            continue
        (first_row, first_use), (second_row, second_use) = incident
        if first_use != second_use[::-1] or sides[first_row] != sides[second_row]:
            continue
        origin, first, second = (exact(vertex) for vertex in rows[first_row])
        apex = exact(next(v for v in rows[second_row] if v not in key))
        u = [first[axis] - origin[axis] for axis in range(3)]
        v = [second[axis] - origin[axis] for axis in range(3)]
        w = [apex[axis] - origin[axis] for axis in range(3)]
        volume = (
            u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
        )
        if volume == 0:
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


def _cell_volume(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    corners = points[cells]
    edges = np.swapaxes(corners[:, 1:] - corners[:, :1], -1, -2)
    return np.abs(np.linalg.det(edges)) / 6.0


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
    periodic_state = _native_periodic_state(mesh, None)
    if periodic_state is None:
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
    /,
) -> None:
    if mesh.periodic_topology is None:
        return
    periodic_state = _native_periodic_state(mesh, None)
    if periodic_state is None:
        raise ValueError("Periodic metric validation requires a native quotient state.")
    epsilon = np.finfo(np.float64).eps
    for orbit in periodic_state.orbits:
        representative = metric_by_vertex_id.get(int(orbit.representative_vertex_id))
        if representative is None:
            raise ValueError(
                "Periodic metric evidence is missing an orbit representative."
            )
        for member in orbit.members:
            metric = metric_by_vertex_id.get(int(member.vertex_id))
            if metric is None:
                raise ValueError("Periodic metric evidence is missing a lifted vertex.")
            linear = np.asarray(member.representative_to_member, dtype=np.float64)[:, :3]
            try:
                inverse = np.linalg.inv(linear)
            except np.linalg.LinAlgError as error:
                raise ValueError("Periodic metric isometry is singular.") from error
            expected = inverse.T @ representative @ inverse
            scale = max(
                float(np.linalg.norm(expected, ord=np.inf)),
                float(np.linalg.norm(metric, ord=np.inf)),
                np.finfo(np.float64).tiny,
            )
            error = float(np.linalg.norm(metric - expected, ord=np.inf))
            if error > 256.0 * epsilon * scale:
                raise ValueError(
                    "SPD metric values must transform equivariantly across periodic vertex orbits."
                )


def _prepare_native_state(
    source: CellMeshingResult,
    metric_values: np.ndarray,
    /,
    *,
    cell_classes: ArrayLike,
    facet_classes: ArrayLike,
    edge_classes: ArrayLike,
    protected_edges: ArrayLike,
    fixed_vertices: ArrayLike,
) -> _NativeState:
    """Prepare a source-bound P1 tetrahedral complex and persistent native owner."""
    if not isinstance(source, CellMeshingResult):
        raise TypeError("source must be CellMeshingResult.")
    mesh = source.mesh
    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or not mesh.blocks
        or any(
            block.cell_kind not in ("tetrahedron", "simplex:3") for block in mesh.blocks
        )
    ):
        raise ValueError("Native tetra metric adaptation requires 3D tetrahedron blocks.")
    from ._adaptation import _piecewise_linear_geometry

    if not _piecewise_linear_geometry(mesh, source.geometry):
        raise ValueError(
            "Native tetra metric adaptation requires an exact P1 source map."
        )
    from ._local_metric import _boolean_array, _integer_array

    points = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if vertex_ids.shape != (points.shape[0],):
        raise ValueError("Source vertex identities do not align with coordinates.")
    metric = _tensor_array(metric_values, points.shape[0])
    if metric.shape[1:] != (3, 3):
        raise ValueError(
            "Tetra adaptation requires a true 3x3 SPD metric at every vertex."
        )
    properties = _tensor_properties(metric)
    if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
        raise ValueError(
            "Tetra adaptation requires finite symmetric positive-definite metrics."
        )

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
    face_id_by_key = {
        _face_key(row): int(identifier)
        for row, identifier in zip(face_keys.tolist(), face_ids.tolist(), strict=True)
    }
    face_class_by_key = {
        _face_key(row): int(value)
        for row, value in zip(face_keys.tolist(), facet_codes.tolist(), strict=True)
    }

    incidence: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    global_cells = vertex_ids[source_cells]
    for cell_row, vertices in enumerate(global_cells):
        for local_face, face in enumerate(_TETRA_FACES):
            key = _face_key(int(vertices[index]) for index in face)
            incidence[key].append((cell_row, local_face))
    constrained_faces: list[
        tuple[int, tuple[int, int, int], np.ndarray, tuple[Any, ...]]
    ] = []
    for key, incident in incidence.items():
        incident_regions = {int(regions[row]) for row, _ in incident}
        if (
            len(incident) == 1
            or len(incident_regions) > 1
            or face_class_by_key.get(key, 0) > 0
        ):
            face_id = face_id_by_key.get(key)
            if face_id is None:
                raise ValueError("A constrained tetra facet has no source identity.")
            row, local_face = min(
                incident, key=lambda item: int(source_cell_ids[item[0]])
            )
            face_vertices = source_cells[row, _TETRA_FACES[local_face]].astype(
                np.int32, copy=True
            )
            sides = (
                len(incident),
                tuple(sorted(incident_regions)),
                face_class_by_key.get(key, 0),
            )
            constrained_faces.append((face_id, key, face_vertices, sides))
    constrained_faces.sort(key=lambda item: item[0])
    source_faces = np.asarray(
        [row for _, _, row, _ in constrained_faces], dtype=np.int32
    ).reshape((-1, 3))
    source_face_row_ids = np.asarray(
        [identifier for identifier, *_ in constrained_faces], dtype=np.int64
    )

    vertex_row_by_id = {
        int(identifier): row for row, identifier in enumerate(vertex_ids.tolist())
    }
    source_segment_keys = [
        _edge_key(edge_keys[row].tolist())
        for row in range(edge_ids.size)
        if edge_codes[row] > 0 or protected[row]
    ]
    source_segment_keys.sort(key=lambda key: edge_id_by_key[key])
    source_segment_ids = np.asarray(
        [edge_id_by_key[key] for key in source_segment_keys], dtype=np.int64
    )
    source_segments = np.asarray(
        [
            [vertex_row_by_id[identifier] for identifier in key]
            for key in source_segment_keys
        ],
        dtype=np.int32,
    ).reshape((-1, 2))
    segment_groups = np.arange(source_segment_ids.size, dtype=np.int32)
    # Off a periodic lift, exactly coplanar source triangles joined across
    # non-segment edges form one planar source facet (one group): boundary
    # vertices then move and collapse across the facet's internal diagonals.
    # Lifted periodic seams keep one group per triangle.
    row_groups = (
        _coplanar_face_groups(
            points,
            source_faces,
            [sides for *_, sides in constrained_faces],
            {_edge_key(row) for row in source_segments.tolist()},
        )
        if periodic_state is None
        else np.arange(source_face_row_ids.size, dtype=np.int64)
    )
    group_first_rows = {
        int(group): int(np.flatnonzero(row_groups == group)[0])
        for group in np.unique(row_groups).tolist()
    }
    source_face_ids = np.asarray(
        sorted(source_face_row_ids[row] for row in group_first_rows.values()),
        dtype=np.int64,
    )
    face_groups = np.searchsorted(
        source_face_ids,
        [source_face_row_ids[group_first_rows[int(group)]] for group in row_groups],
    ).astype(np.int32)
    source_face_group_rows = {
        int(identifier): row for row, identifier in enumerate(source_face_ids.tolist())
    }
    source_group_triangles: dict[int, tuple[tuple[int, int, int], ...]] = defaultdict(
        tuple
    )
    for row, (_, key, _, _) in enumerate(constrained_faces):
        source_group_triangles[int(face_groups[row])] += (key,)
    source_face_vertices = {
        group: frozenset(vertex for triangle in triangles for vertex in triangle)
        for group, triangles in source_group_triangles.items()
    }
    source_segment_vertices = {
        row: frozenset(key) for row, key in enumerate(source_segment_keys)
    }
    source_segment_group_rows = {
        int(identifier): row for row, identifier in enumerate(source_segment_ids.tolist())
    }
    # Off a periodic lift each facet declares the exact plane of its first
    # triangle (origin a, axes b - a and c - a) when those axes are binary64,
    # so the native admits a carrier rounding a point of the plane
    # (declared-plane rounding-box admission).
    row_planes = _declared_face_planes(points, source_faces)
    face_planes = np.full((source_faces.shape[0], 3, 3), np.nan, dtype=np.float64)
    if periodic_state is None:
        for row in range(source_faces.shape[0]):
            face_planes[row] = row_planes[group_first_rows[int(row_groups[row])]]
    declared_source_faces = frozenset(
        frozenset(constrained_faces[row][1])
        for row in np.flatnonzero(~np.isnan(face_planes[:, 0, 0])).tolist()
    )
    source_complex = TetMeshSourceComplex(
        points=points,
        faces=source_faces,
        face_group_rows=face_groups,
        face_tolerances=np.zeros((source_faces.shape[0],), dtype=np.float64),
        segments=source_segments,
        segment_group_rows=segment_groups,
        segment_tolerances=np.zeros((source_segment_ids.size,), dtype=np.float64),
        witness_strata=np.zeros((vertex_ids.size,), dtype=np.int8),
        witness_entities=np.full((vertex_ids.size,), -1, dtype=np.int32),
        witness_parameters=np.zeros((vertex_ids.size, 2), dtype=np.float64),
        face_ids=source_face_ids,
        segment_ids=source_segment_ids,
        face_planes=face_planes,
        face_plane_tolerances=np.zeros((source_faces.shape[0],), dtype=np.float64),
    )

    source_map = source.geometry.source_map
    frame_id = source.coordinate_contract.reference_frame
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError(
            "Tetra adaptation requires the literal coordinate reference frame."
        )
    if source_map is None:
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

    native = TetMesh3D(
        points,
        source_cells.astype(np.int32),
        regions,
        source_faces,
        source_face_ids[face_groups],
        source_segments,
        source_segment_ids,
        boundary_policy=PlcBoundaryPolicy.CONFORMING,
        protection_radii=np.zeros((vertex_ids.size,), dtype=np.float64),
        protected_vertices=np.asarray(fixed, dtype=np.int8),
        vertex_ids=vertex_ids,
        cell_ids=source_cell_ids,
        facet_ids=source_face_row_ids,
        segment_ids=source_segment_ids,
        source=source_complex,
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
        target_mesh=mesh,
        periodic_vertex_windings=periodic_vertex_windings,
        source_metric=metric,
        source_cell_ids=source_cell_ids,
        source_vertex_ids=vertex_ids,
        source_vertex_rows=source_vertex_row_by_id,
        source_cell_vertices=source_cells,
        source_cell_regions=regions,
        source_cell_blocks=source_cell_blocks,
        source_records=_source_tetra_inverse(
            points, source_cells, vertex_ids, source_cell_ids
        ),
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
        face_classes=face_class_by_key,
        source_face_ids=face_id_by_key,
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
    coordinates = np.asarray(state.source_mesh.coordinates, dtype=np.float64)
    rows = state.source_vertex_rows
    image = [Fraction(0), Fraction(0), Fraction(0)]
    for identifier, weight in stencil.items():
        point = coordinates[rows[identifier]]
        for axis in range(3):
            image[axis] += weight * Fraction(float(point[axis]))
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
    weights must be binary64. A boundary edge lies on a constrained facet or
    segment whose exact predicates read the carrier itself, so there ``E`` must
    be exactly representable and equal to ``I``.
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


def _constrained_edges(topology: _TetTopology, /) -> set[tuple[int, int]]:
    edges = set(topology.segment_row)
    for face in topology.facet_row:
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
    """Short dyadic split parameters with a consistent binary64 realization.

    Each metric midpoint is rounded to the dyadic grid ``2**-bits`` for
    ``bits = _SPLIT_PARAMETER_BITS, ..., 1``; the finest level at which every
    orbit member is admissible (``_split_member_admissible``) wins. Short
    parameters keep nested ancestry weights binary64. ``None`` means no dyadic
    point of the edge has a consistent realization.
    """
    members = []
    for edge in edges:
        first, second = edge
        first_row, second_row = topology.vertex_row[first], topology.vertex_row[second]
        parameter, _ = _metric_midpoint(
            points[first_row], points[second_row], metrics[first_row], metrics[second_row]
        )
        members.append((first_row, second_row, edge, parameter))
    tried: set[tuple[Fraction, ...]] = set()
    for bits in range(_SPLIT_PARAMETER_BITS, 0, -1):
        scale = 1 << bits
        candidate = tuple(
            Fraction(min(max(round(parameter * scale), 1), scale - 1), scale)
            for *_, parameter in members
        )
        if candidate in tried:
            continue
        tried.add(candidate)
        if all(
            _split_member_admissible(
                state,
                points[first_row],
                points[second_row],
                state.vertex_stencils[edge[0]],
                state.vertex_stencils[edge[1]],
                value,
                edge in constrained,
            )
            for (first_row, second_row, edge, _), value in zip(
                members, candidate, strict=True
            )
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
    topology = _topology(snapshot)
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
            state.source_metric, source_vertex_rows, stencil
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
    removed_cells = set(
        map(
            int,
            np.asarray(
                edit.removed_cell_ids if edit.removed_cell_ids is not None else (),
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
) -> tuple[TetMeshEdit, CellMesh | None]:
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
        if isinstance(proposal, dict) and "current_stencil" in proposal
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
        if not isinstance(proposal, dict):
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
        elif isinstance(proposal, dict) and "vertex" in proposal and "metric" in proposal:
            proposal_metric_by_native[int(proposal["vertex"])] = np.asarray(
                proposal["metric"], dtype=np.float64
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
        _validate_periodic_metric_values(target, periodic_metrics)
    except ValueError as error:
        raise _PeriodicCandidateRejected(str(error)) from error

    try:
        periodic = tet_mesh_periodic_edit(
            source,
            target,
            transition,
            source_regions=source_regions,
            target_regions=target_regions,
        )
    except (TypeError, ValueError) as error:
        raise _PeriodicCandidateRejected(str(error)) from error
    return replace(native_edit, periodic=periodic), target


def _apply_split_batch(
    state: _NativeState, split_parameters: dict[tuple[int, int], Fraction], /
) -> TetMeshEditStatus:
    edit, metadata = _split_edit(state, split_parameters)
    try:
        edit, target_mesh = _prepare_periodic_native_edit(state, edit, metadata)
    except _PeriodicCandidateRejected:
        return TetMeshEditStatus.REJECTED_TOPOLOGY
    status = state.native.apply_edit(edit)
    if status is not TetMeshEditStatus.APPLIED:
        return status
    if target_mesh is not None:
        state.target_mesh = target_mesh
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
    classes: dict[tuple[int, int], int] = {}
    protected: dict[tuple[int, int], bool] = {}
    for edge in topology.edge_cells:
        target_key = _edge_key(state.vertex_target_ids[int(vertex)] for vertex in edge)
        classes[edge] = int(state.edge_classes.get(target_key, 0))
        is_protected = target_key in state.protected_edge_ids
        segment = segment_data.get(edge)
        if segment is not None:
            source_id, segment_protected = segment
            is_protected |= segment_protected
            source_key = source_edge_key_by_id.get(source_id)
            if source_key is not None:
                classes[edge] = int(state.edge_classes.get(source_key, classes[edge]))
        protected[edge] = is_protected
    return classes, protected


def _metric_edge_table(
    state: _NativeState, topology: _TetTopology, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex_ids = np.asarray(state.snapshot.vertex_ids, dtype=np.int64)
    points = np.asarray(state.snapshot.points, dtype=np.float64)
    metrics = np.asarray(
        [state.metric_by_vertex[int(identifier)] for identifier in vertex_ids],
        dtype=np.float64,
    )
    edges = tuple(topology.edge_cells)
    edge_rows = np.asarray(
        [
            [topology.vertex_row[first], topology.vertex_row[second]]
            for first, second in edges
        ],
        dtype=np.int64,
    ).reshape((-1, 2))
    return points, metrics, _metric_lengths(points, metrics, edge_rows)


def _periodic_entity_orbit_groups(
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
            topology = _topology(state.snapshot)
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
    return tuple(
        tuple(_edge_key(member) for member in orbit)
        for orbit in _periodic_entity_orbit_groups(state, 1)
    )


def _periodic_face_orbit_groups(
    state: _NativeState, /
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    return tuple(
        tuple(_face_key(member) for member in orbit)
        for orbit in _periodic_entity_orbit_groups(state, 2)
    )


def _periodic_vertex_orbit_groups(state: _NativeState, /) -> tuple[tuple[int, ...], ...]:
    groups = []
    for orbit in _periodic_entity_orbit_groups(state, 0):
        if any(len(member) != 1 for member in orbit):
            raise ValueError("Periodic vertex orbit rows must contain one vertex each.")
        groups.append(tuple(member[0] for member in orbit))
    return tuple(groups)


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
    topology = _topology(state.snapshot)
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
    topology = _topology(state.snapshot)
    points, metrics, lengths = _metric_edge_table(state, topology)
    length_by_edge = dict(zip(topology.edge_cells, lengths.tolist(), strict=True))
    constrained = _constrained_edges(topology)
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
        state.source_metric, state.source_vertex_rows, stencil
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
        _metric_lengths(extended_points, extended_metrics, pairs),
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


def _split_phase(state: _NativeState, counters: dict[str, int], /) -> int:
    applied = 0
    blocked: set[tuple[int, int]] = set()
    while True:
        candidates = _split_metric_candidates(state, blocked)
        if not candidates:
            return applied
        changed = _try_split_batch(state, candidates, counters, blocked)
        applied += changed
        if changed:
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
    snapshot = state.snapshot
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
        group = state.source_face_group_rows.get(int(source_id))
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
        group = state.source_segment_group_rows.get(int(source_id))
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
    for their images under the collapse. The collapse must strictly decrease
    the termination objective (``_objective_decreases``).
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

    The removed vertex is neither fixed nor protected, carries no protected
    facet or segment, and is interior or carried by a source triangle or edge
    holding the kept vertex (a constrained segment collapses only along
    itself); periodic lifts collapse only interior vertices.
    """
    removed_row = topology.vertex_row[removed]
    if removed in state.fixed_vertices or bool(
        np.asarray(state.snapshot.protected_vertices)[removed_row]
    ):
        return False
    entity = _carrier_entity(state, records, removed)
    if state.target_mesh.periodic_topology is not None and (
        entity is not None or _carrier_entity(state, records, kept) is not None
    ):
        # Lifted periodic seams are constrained facets: their orbit copies
        # collapse only through interior vertices.
        return False
    if entity is not None and (
        len(entity) < 2 or not set(state.vertex_stencils[kept]) <= entity
    ):
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

    points = np.asarray(snapshot.points, dtype=np.float64)
    vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    metrics = np.asarray(
        [state.metric_by_vertex[int(identifier)] for identifier in vertex_ids],
        dtype=np.float64,
    )
    proposed_native_cells: list[tuple[int, int, int, int]] = []
    parent_cell_ids: list[int] = []
    for row in removed_rows:
        old = topology.cell_vertex_ids[row]
        if kept in old:
            continue
        proposed_native_cells.append(_tetra_key(image(int(value)) for value in old))
        parent_cell_ids.append(int(cell_ids[row]))
    if not proposed_native_cells:
        return None
    oriented_rows = _oriented_tetrahedra(
        points,
        topology,
        np.asarray(proposed_native_cells, dtype=np.int64),
        PredicateMode.EXACT,
    )
    if oriented_rows is None:
        return None
    before_quality = _tetra_quality(
        points,
        metrics,
        np.asarray(snapshot.tetrahedra, dtype=np.int64)[
            np.asarray(removed_rows, dtype=np.int64)
        ],
    )
    after_quality = _tetra_quality(points, metrics, oriented_rows)
    if (
        not np.all(np.isfinite(after_quality))
        or np.any(after_quality <= 0.0)
        or not _clears_determinant_floor(points[oriented_rows])
    ):
        return None

    removed_edges = [
        _edge_key((removed, other))
        for other in sorted(removed_link)
        if not protected[_edge_key((removed, other))]
    ]
    added_edges = [(kept, other) for other in sorted(removed_link - kept_link - {kept})]
    removed_lengths = _edge_set_lengths(points, metrics, topology, removed_edges)
    added_lengths = _edge_set_lengths(points, metrics, topology, added_edges)
    if not _objective_decreases(removed_lengths, added_lengths, False):
        return None
    oriented_native = vertex_ids[oriented_rows]
    return {
        "edge": edge,
        "removed": removed,
        "kept": kept,
        "removed_cell_ids": tuple(sorted(int(cell_ids[row]) for row in removed_rows)),
        "replacement_cell_vertices": oriented_native,
        "replacement_parent_cell_ids": tuple(parent_cell_ids),
        "replacement_regions": np.full(
            (oriented_native.shape[0],), next(iter(cavity_regions)), dtype=np.int32
        ),
        "block": next(iter(cavity_blocks)),
        "cavity": frozenset(int(cell_ids[row]) for row in cavity_rows),
        "quality_gain": float(np.min(after_quality) - np.min(before_quality)),
        "added_violations": _band_violations(added_lengths),
        "removed_facet_ids": tuple(removed_facets),
        "replacement_facets": tuple(new_facets),
        "removed_segment_ids": tuple(removed_segments),
        "replacement_segments": tuple(new_segments),
    }


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
    return _metric_lengths(points, metrics, rows)


def _facet_group_map(
    state: _NativeState, snapshot: TetMeshArrays, /
) -> dict[tuple[int, int, int], int]:
    vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    rows = vertex_ids[np.asarray(snapshot.faces, dtype=np.int64)]
    sources = np.asarray(snapshot.face_sources, dtype=np.int64)
    result: dict[tuple[int, int, int], int] = {}
    for vertices, source_id in zip(rows.tolist(), sources.tolist(), strict=True):
        group = state.source_face_group_rows.get(int(source_id))
        if group is None:
            raise ValueError("A constrained native facet lost its source group.")
        result[_face_key(vertices)] = group
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
    collapse_pairs = np.asarray(
        [(row["removed"], row["kept"]) for row in proposals], dtype=np.uint64
    ).reshape((-1, 2))
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
        edit, target_mesh = _prepare_periodic_native_edit(state, edit, metadata)
    except _PeriodicCandidateRejected:
        return TetMeshEditStatus.REJECTED_TOPOLOGY
    status = state.native.apply_edit(edit)
    if status is not TetMeshEditStatus.APPLIED:
        return status
    if target_mesh is not None:
        state.target_mesh = target_mesh
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


def _collapse_metric_candidates(
    state: _NativeState, blocked: set[tuple[int, int]], /
) -> list[tuple[dict[str, Any], ...]]:
    topology = _topology(state.snapshot)
    _, _, lengths = _metric_edge_table(state, topology)
    classes, protected = _current_edge_controls(state, topology)
    records = _boundary_records(state)
    eligible: dict[tuple[int, int], tuple[dict[str, Any], float]] = {}
    for edge, length in zip(topology.edge_cells, lengths.tolist(), strict=True):
        if edge in blocked or protected[edge] or classes[edge] != 0 or length >= _LOWER:
            continue
        proposal = _collapse_proposal(state, topology, records, protected, edge)
        if proposal is not None:
            eligible[edge] = (proposal, float(length))
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
        if len(quotient_edges) != 1 or len(kept_orbits) != 1:
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
    blocked: set[tuple[int, int]],
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
            state, proposal_groups[:middle], counters, blocked
        ) + _try_collapse_batch(state, proposal_groups[middle:], counters, blocked)
    blocked.update(proposal["edge"] for proposal in proposal_groups[0])
    _count_rejection(status, counters, 1)
    return 0


def _collapse_phase(state: _NativeState, counters: dict[str, int], /) -> int:
    applied = 0
    blocked: set[tuple[int, int]] = set()
    while True:
        proposals = _collapse_metric_candidates(state, blocked)
        if not proposals:
            return applied
        changed = _try_collapse_batch(state, proposals, counters, blocked)
        applied += changed
        if changed:
            blocked.clear()


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
        edit, target_mesh = _prepare_periodic_native_edit(state, edit, metadata)
    except _PeriodicCandidateRejected:
        return TetMeshEditStatus.REJECTED_TOPOLOGY
    status = state.native.apply_edit(edit)
    if status is not TetMeshEditStatus.APPLIED:
        return status
    if target_mesh is not None:
        state.target_mesh = target_mesh
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


def _replacement_candidates(
    state: _NativeState,
    kind: TetMeshEditKind,
    blocked: set[tuple[int, ...]],
    mode: PredicateMode,
    /,
) -> list[tuple[dict[str, Any], ...]]:
    topology = _topology(state.snapshot)
    points, metrics, _ = _metric_edge_table(state, topology)
    regions = np.asarray(state.snapshot.tetrahedron_regions, dtype=np.int64)
    classes, protected = _current_edge_controls(state, topology)
    candidates: list[dict[str, Any]] = []
    cells = np.asarray(state.snapshot.tetrahedra, dtype=np.int64)

    def cell_edges(rows: np.ndarray) -> set[tuple[int, int]]:
        vertex_ids = np.asarray(state.snapshot.vertex_ids, dtype=np.int64)
        return {
            _edge_key((int(vertex_ids[cell[first]]), int(vertex_ids[cell[second]])))
            for cell in rows
            for first, second in _TETRA_EDGE_CORNERS
        }

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
        cell_ids = np.asarray(state.snapshot.cell_ids, dtype=np.int64)
        cavity_ids = tuple(sorted(int(cell_ids[row]) for row in cavity_rows))
        cavity_regions = {int(regions[row]) for row in cavity_rows}
        cavity_blocks = {state.cell_blocks[int(cell_ids[row])] for row in cavity_rows}
        if len(cavity_regions) != 1 or len(cavity_blocks) != 1:
            return
        # Flip cells are listed in one consistent corner order: all positive,
        # or all mirrored (then reversed together); a mixed set is not a flip.
        oriented = _oriented_tetrahedra(
            points,
            topology,
            np.asarray(raw_cells, dtype=np.int64),
            mode,
            reversible=True,
        )
        if oriented is None:
            return
        before_rows = cells[np.asarray(cavity_rows, dtype=np.int64)]
        before = _tetra_quality(points, metrics, before_rows)
        after = _tetra_quality(points, metrics, oriented)
        if (
            before.size == 0
            or after.size == 0
            or not np.all(np.isfinite(after))
            or np.any(after <= 0.0)
            or not _clears_determinant_floor(points[oriented])
        ):
            return
        old_edges = cell_edges(before_rows)
        new_edges = cell_edges(oriented)
        removed_edges = sorted(
            edge for edge in old_edges - new_edges if not protected[edge]
        )
        added_edges = sorted(new_edges - old_edges)
        if not _objective_decreases(
            _edge_set_lengths(points, metrics, topology, removed_edges),
            _edge_set_lengths(points, metrics, topology, added_edges),
            float(np.min(after)) > float(np.min(before)) + _IMPROVEMENT,
        ):
            return
        vertices = np.asarray(state.snapshot.vertex_ids, dtype=np.int64)[oriented]
        ancestors = sorted(
            {
                int(source_id)
                for cell_id in cavity_ids
                for source_id in state.cell_ancestry[cell_id]
            }
        )
        candidates.append(
            {
                "candidate_key": key,
                "removed_cell_ids": cavity_ids,
                "replacement_cell_vertices": vertices,
                "replacement_regions": np.full(
                    (vertices.shape[0],),
                    next(iter(cavity_regions)),
                    dtype=np.int32,
                ),
                "block": next(iter(cavity_blocks)),
                "source_ancestors": tuple(ancestors),
                "lineage_kind": EntityLineageKind.SWAPPED_FROM,
                "cavity": frozenset(cavity_ids),
                "gain": float(np.min(after) - np.min(before)),
                "removed_facet_ids": () if boundary is None else boundary[0],
                "replacement_facets": () if boundary is None else boundary[1],
            }
        )

    if kind is TetMeshEditKind.FLIP23:
        face_sources = topology.facet_row
        for face, rows in topology.face_cells.items():
            if len(rows) != 2 or face in face_sources or face in blocked:
                continue
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
                continue
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
                continue
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
        return _group_periodic_replacement_candidates(state, candidates, 2)

    if kind not in (TetMeshEditKind.FLIP32, TetMeshEditKind.RECONNECT):
        raise ValueError(
            "Replacement candidate kind must be FLIP23, FLIP32, or RECONNECT."
        )
    constrained_edges = {
        _edge_key((face[index], face[(index + 1) % 3]))
        for face in topology.facet_row
        for index in range(3)
    }
    for edge, rows in topology.edge_cells.items():
        if (
            edge in blocked
            or edge in constrained_edges
            or protected[edge]
            or classes[edge] != 0
        ):
            continue
        if kind is TetMeshEditKind.FLIP32 and len(rows) != 3:
            continue
        if kind is TetMeshEditKind.RECONNECT and len(rows) < 4:
            continue
        ring = _edge_ring(topology, edge, rows)
        if ring is None:
            continue
        if kind is TetMeshEditKind.FLIP32:
            if len(ring) != 3:
                continue
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
                continue
            raw = list(triangulation[0])
        propose(edge, rows, raw)
    if kind is TetMeshEditKind.RECONNECT and state.target_mesh.periodic_topology is None:
        records = _boundary_records(state)
        for edge, rows in topology.edge_cells.items():
            swap = _boundary_swap(
                state, topology, points, records, protected, classes, edge, rows
            )
            if swap is not None and edge not in blocked:
                ring, boundary = swap
                triangulation = _best_ring_triangulation(
                    points, metrics, topology, edge, ring, mode
                )
                if triangulation is not None:
                    propose(edge, rows, list(triangulation[0]), boundary)
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
    grouped = []
    orbit_groups = (
        _periodic_edge_orbit_groups(state)
        if dimension == 1
        else _periodic_face_orbit_groups(state)
        if dimension == 2
        else None
    )
    if orbit_groups is None:
        raise ValueError("Replacement candidates require edge or face orbits.")
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
) -> int:
    applied = 0
    blocked: set[tuple[int, ...]] = set()
    while True:
        candidates = _replacement_candidates(state, kind, blocked, mode)
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


def _quantized_relocation(
    state: _NativeState,
    simplices: tuple[tuple[int, ...], ...] | None,
    target: np.ndarray,
    current: np.ndarray,
    /,
) -> tuple[dict[int, Fraction], np.ndarray] | None:
    """Dyadic source stencil and binary64 carrier of one relocation target.

    The target's exact weights (in the source cell containing it, or of its
    orthogonal projection into the first of the vertex's carrier simplices,
    source triangles or edges, containing it) are rounded to the coarsest
    dyadic grid ``2**-bits`` that resolves the move and whose point is closer
    to the exact goal than the current point is; the stencil weights are then
    binary64 and the carrier is the rounded exact image. An interior carrier
    may round; a carrier in a source triangle must be exact or lie on a
    declared exact plane (the native rounding-box admission); a carrier on a
    source edge must be exact. ``None``: the target leaves the carrier, or no
    representable point makes progress.
    """
    coordinates = np.asarray(state.source_mesh.coordinates, dtype=np.float64)
    rows = state.source_vertex_rows
    query = tuple(Fraction(float(value)) for value in target)
    exact_query = (query[0], query[1], query[2])
    located: tuple[tuple[int, ...], tuple[Fraction, ...]] | None = None
    if simplices is None:
        cell = _locate_source_weights(exact_query, state.source_records)
        if cell is not None:
            located = cell[1], cell[2]
    else:
        for simplex in simplices:
            simplex_weights = _entity_weights(
                tuple(
                    tuple(Fraction(float(value)) for value in coordinates[rows[vertex]])
                    for vertex in simplex
                ),
                exact_query,
            )
            if simplex_weights is not None:
                located = simplex, simplex_weights
                break
    if located is None:
        return None
    vertices, weights = located
    goal = np.asarray(
        [
            float(value)
            for value in _stencil_image(state, _fraction_stencil(vertices, weights))
        ]
    )
    distance = float(np.sum((goal - current) ** 2))
    corners = coordinates[[rows[vertex] for vertex in vertices]]
    span = float(np.max(np.ptp(corners, axis=0)))
    if not distance > 0.0 or not span > 0.0:
        return None
    # A grid coarser than the move cannot resolve it; start where it can.
    first = min(max(int(np.floor(np.log2(span / np.sqrt(distance)))), 1), 53)
    for bits in range(first, 54):
        stencil = _fraction_stencil(vertices, _quantized_weights(weights, bits))
        image = _stencil_image(state, stencil)
        carrier = np.asarray([float(value) for value in image], dtype=np.float64)
        if not float(np.sum((carrier - goal) ** 2)) < distance:
            continue
        if simplices is not None and any(Fraction(float(v)) != v for v in image):
            if (
                len(vertices) == 2
                or frozenset(vertices) not in state.declared_source_faces
            ):
                return None
        return stencil, carrier
    return None


def _relocation_proposals(
    state: _NativeState,
    cache: dict[int, dict[str, Any] | None],
    /,
) -> list[tuple[dict[str, Any], ...]]:
    """Objective-decreasing vertex moves to quantized source points.

    A vertex is pulled toward unit metric length along its edges; steps
    ``2**-k`` of the pull (``k <= _RELOCATION_STEP_BITS``) are quantized by
    ``_quantized_relocation``. Interior vertices move in their source cell,
    boundary vertices within the source triangle or edge carrying them (pinned
    at source corners); a periodic lift moves interior vertices only. A move is
    proposed when its star stays positively oriented above the determinant
    floor (``_floor_oriented``) and strictly decreases the termination
    objective. ``cache`` holds each vertex's proposal (or ``None``, also after a
    native refusal) while its closed star is unchanged; the caller evicts the
    closed stars of moved vertices.
    """
    snapshot = state.snapshot
    topology = _topology(snapshot)
    points, metrics, _ = _metric_edge_table(state, topology)
    _, protected_edges = _current_edge_controls(state, topology)
    records = _boundary_records(state)
    periodic = state.target_mesh.periodic_topology is not None
    vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    dimensions = np.asarray(snapshot.vertex_dimensions, dtype=np.int8)
    protected_vertices = np.asarray(snapshot.protected_vertices, dtype=np.bool_)
    cells = np.asarray(snapshot.tetrahedra, dtype=np.int64)
    regions = np.asarray(snapshot.tetrahedron_regions, dtype=np.int64)
    current_cell_ids = np.asarray(snapshot.cell_ids, dtype=np.int64)
    current_vertex_rows = {
        int(identifier): row for row, identifier in enumerate(vertex_ids.tolist())
    }
    source_coordinates = np.asarray(state.source_mesh.coordinates, dtype=np.float64)

    def propose(vertex: int) -> dict[str, Any] | None:
        row = topology.vertex_row[vertex]
        if vertex in state.fixed_vertices or protected_vertices[row]:
            return None
        simplices = _carrier_simplices(state, records, vertex)
        if (periodic or simplices is None) and dimensions[row] != 3:
            return None
        if simplices is not None and (periodic or not simplices):
            return None
        star = topology.vertex_cells[vertex]
        if not star:
            return None
        star_rows = np.asarray(star, dtype=np.int64)
        star_ids = tuple(int(current_cell_ids[cell]) for cell in star)
        star_blocks = {state.cell_blocks[cell_id] for cell_id in star_ids}
        star_regions = {int(regions[cell]) for cell in star}
        if len(star_blocks) != 1 or len(star_regions) != 1:
            return None
        neighbors = sorted(
            {
                int(other)
                for cell in star
                for other in topology.cell_vertex_ids[cell]
                if int(other) != vertex
            }
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
            [[local_index[int(value)] for value in cell] for cell in cells[star_rows]],
            dtype=np.int64,
        ).reshape((-1, 4))
        local_points = points[local_rows]
        local_metrics = metrics[local_rows]
        spokes = np.column_stack(
            (
                np.zeros((len(neighbors),), dtype=np.int64),
                np.arange(1, len(neighbors) + 1, dtype=np.int64),
            )
        )
        measured = np.asarray(
            [not protected_edges[_edge_key((vertex, other))] for other in neighbors],
            dtype=np.bool_,
        )
        lengths = _metric_lengths(local_points, local_metrics, spokes)
        pull = np.mean(
            (1.0 - 1.0 / lengths)[:, None] * (local_points[1:] - local_points[0]),
            axis=0,
        )
        if not np.any(pull != 0.0):
            return None
        before = _tetra_quality(local_points, local_metrics, local_cells)
        # The carrier simplices are coplanar (one facet) or collinear (one
        # curve or segment): the first spans their affine hull.
        corners = (
            None
            if simplices is None
            else source_coordinates[[state.source_vertex_rows[v] for v in simplices[0]]]
        )

        def evaluate(
            point: np.ndarray, metric: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray] | None:
            # Star quality and spoke lengths of a move that keeps the star
            # oriented above the determinant floor, never lowers its minimum
            # quality (a relocation smooths), and strictly decreases the
            # objective.
            moved_points = local_points.copy()
            moved_points[0] = point
            if not _floor_oriented(moved_points[local_cells]):
                return None
            moved_metrics = local_metrics.copy()
            moved_metrics[0] = metric
            after = _tetra_quality(moved_points, moved_metrics, local_cells)
            moved_lengths = _metric_lengths(moved_points, moved_metrics, spokes)
            if float(np.min(after)) < float(np.min(before)) or not _objective_decreases(
                lengths[measured],
                moved_lengths[measured],
                float(np.min(after)) > float(np.min(before)) + _IMPROVEMENT,
            ):
                return None
            return after, moved_lengths

        for step in range(_RELOCATION_STEP_BITS + 1):
            goal = points[row] + np.ldexp(pull, -step)
            if corners is not None:
                span = (corners[1:] - corners[0]).T
                coefficients = np.linalg.lstsq(span, goal - corners[0], rcond=None)[0]
                goal = corners[0] + span @ coefficients
            # Float screen with the current vertex metric; the quantized
            # carrier and its interpolated metric are then checked in full.
            if evaluate(goal, local_metrics[0]) is None:
                continue
            quantized = _quantized_relocation(state, simplices, goal, points[row])
            if quantized is None:
                continue
            stencil, carrier = quantized
            if np.array_equal(carrier, points[row]) or any(
                np.array_equal(carrier, other) for other in local_points[1:]
            ):
                continue
            moved_metric = _source_metric_from_stencil(
                state.source_metric, state.source_vertex_rows, stencil
            )
            evaluated = evaluate(carrier, moved_metric)
            if evaluated is None:
                continue
            after, _ = evaluated
            current_stencil = None
            if periodic:
                # A periodic lift also needs the binary64 stencil of the carrier
                # in the current star, reproducing it exactly.
                current_stencil = _current_star_stencil(
                    points,
                    cells[star_rows],
                    vertex_ids,
                    current_cell_ids[star_rows],
                    current_vertex_rows,
                    carrier,
                )
                if current_stencil is None:
                    continue
            return {
                "vertex": int(vertex),
                "point": carrier,
                "metric": moved_metric,
                "stencil": stencil,
                "current_stencil": current_stencil,
                "star_cell_ids": star_ids,
                "star_vertices": frozenset((vertex, *neighbors)),
                "cavity": frozenset(star_ids),
                "gain": float(np.min(after) - np.min(before)),
                "badness": float(np.max(np.maximum(lengths, 1.0 / lengths))),
                "block": next(iter(star_blocks)),
                "region": next(iter(star_regions)),
            }
        return None

    candidates: list[dict[str, Any]] = []
    for vertex in sorted(
        topology.vertex_cells, key=lambda value: state.vertex_target_ids[value]
    ):
        if vertex not in cache:
            cache[vertex] = propose(vertex)
        proposal = cache[vertex]
        if proposal is not None:
            candidates.append(proposal)
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
            -min(float(proposal["gain"]) for proposal in members),
            -min(float(proposal["badness"]) for proposal in members),
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
) -> dict[int, Fraction] | None:
    """Binary64 barycentric stencil of a carrier in the current star that
    reproduces it exactly, else ``None``."""
    located = _locate_source_stencil(
        carrier, _source_tetra_inverse(points, star_cells, vertex_ids, star_cell_ids)
    )
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
    state: _NativeState, proposals: list[dict[str, Any]], /
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


def _apply_relocation_batch(
    state: _NativeState, proposals: list[dict[str, Any]], /
) -> TetMeshEditStatus:
    edit = _relocation_edit(state, proposals)
    try:
        edit, target_mesh = _prepare_periodic_native_edit(
            state, edit, {"proposals": proposals}
        )
    except _PeriodicCandidateRejected:
        return TetMeshEditStatus.REJECTED_TOPOLOGY
    status = state.native.apply_edit(edit)
    if status is not TetMeshEditStatus.APPLIED:
        return status
    if target_mesh is not None:
        state.target_mesh = target_mesh
    state.snapshot = state.native.arrays()
    source_vertex_ids = {
        int(identifier) for identifier in state.source_vertex_ids.tolist()
    }
    for proposal in proposals:
        vertex = int(proposal["vertex"])
        target_id = state.vertex_target_ids[vertex]
        state.vertex_stencils[vertex] = proposal["stencil"]
        state.metric_by_vertex[vertex] = proposal["metric"]
        state.history_vertex[:] = [
            relation for relation in state.history_vertex if relation[1] != target_id
        ]
        if target_id in source_vertex_ids:
            state.history_vertex.append((target_id, target_id, EntityLineageKind.UNKNOWN))
        for cell_id in proposal["star_cell_ids"]:
            ancestry = state.cell_ancestry[int(cell_id)]
            for source_id in ancestry:
                state.history_cells.append(
                    (
                        source_id,
                        state.cell_target_ids[int(cell_id)],
                        EntityLineageKind.RELOCATED,
                    )
                )
        state.changed_source_cell_ids.update(
            source_id
            for cell_id in proposal["star_cell_ids"]
            for source_id in state.cell_ancestry[int(cell_id)]
        )
    _refresh_native_periodic_windings(state)
    return status


def _apply_relocation_partition(
    state: _NativeState,
    candidate_groups: list[tuple[dict[str, Any], ...]],
    counters: dict[str, int],
    cache: dict[int, dict[str, Any] | None],
    /,
) -> list[dict[str, Any]]:
    """Apply disjoint relocation orbits; refused orbits are counted and their
    vertices cached as having no proposal. Returns the applied proposals."""
    proposals = [proposal for group in candidate_groups for proposal in group]
    try:
        status: TetMeshEditStatus | None = _apply_relocation_batch(state, proposals)
    except FloatingPointError:
        status = None
    if status is TetMeshEditStatus.APPLIED:
        counters["relocations"] += len(proposals)
        return proposals
    if status is not None:
        _raise_fatal_edit_status(status, "relocation")
    if len(candidate_groups) > 1:
        middle = len(candidate_groups) // 2
        return _apply_relocation_partition(
            state, candidate_groups[:middle], counters, cache
        ) + _apply_relocation_partition(state, candidate_groups[middle:], counters, cache)
    for proposal in candidate_groups[0]:
        cache[int(proposal["vertex"])] = None
    _count_rejection(status, counters, len(proposals))
    return []


def _relocation_phase(
    state: _NativeState,
    counters: dict[str, int],
    /,
) -> int:
    """Relocate until no vertex has an applicable objective-decreasing move.

    Every pass either applies a move (a strict objective decrease) or caches a
    refused proposal as none, so the phase terminates; a moved vertex evicts the
    cached proposals of its closed star.
    """
    applied = 0
    cache: dict[int, dict[str, Any] | None] = {}
    while True:
        candidates = _relocation_proposals(state, cache)
        if not candidates:
            return applied
        selected = _disjoint_candidates(
            [
                (
                    (
                        -min(float(proposal["gain"]) for proposal in orbit),
                        -min(float(proposal["badness"]) for proposal in orbit),
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
        moved = _apply_relocation_partition(state, selected, counters, cache)
        applied += len(moved)
        for proposal in moved:
            for vertex in proposal["star_vertices"]:
                cache.pop(int(vertex), None)


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
    topology = _topology(state.snapshot)
    points, metrics, lengths = _metric_edge_table(state, topology)
    edge_keys = tuple(topology.edge_cells)
    _, protected_by_key = _current_edge_controls(state, topology)
    protected = np.asarray([protected_by_key[edge] for edge in edge_keys], dtype=np.bool_)
    cells = np.asarray(state.snapshot.tetrahedra, dtype=np.int64)
    qualities = _tetra_quality(points, metrics, cells)
    regions = np.asarray(state.snapshot.tetrahedron_regions, dtype=np.int64)
    volumes = _region_volumes(points, cells, regions)
    volume_error = _relative_region_error(state.source_region_volumes, volumes)
    return edge_keys, lengths, protected, qualities, volumes, volume_error


def _unit_metric_status(state: _NativeState, /) -> tuple[bool, int, float, float, float]:
    _, lengths, protected, qualities, _, _ = _native_metric_measurements(state)
    measured = ~protected
    if not np.any(measured):
        measured = np.ones(protected.shape, dtype=np.bool_)
    values = lengths[measured]
    within = (values >= _LOWER) & (values <= _UPPER)
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
    topology = _topology(state.snapshot)
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
    counters: dict[str, int],
    passes: int,
    status: TetraMetricStatus,
    /,
) -> TetraMetricEvidence:
    edge_keys, lengths, protected, qualities, _, volume_error = (
        _native_metric_measurements(state)
    )
    measured = ~protected
    if not np.any(measured):
        measured = np.ones(protected.shape, dtype=np.bool_)
    values = lengths[measured]
    inside = (values >= _LOWER) & (values <= _UPPER)
    if status is TetraMetricStatus.COMPLETE and not np.all(inside):
        status = TetraMetricStatus.STALLED
    if not np.isfinite(volume_error):
        status = TetraMetricStatus.NUMERICAL_STOP
        volume_error = np.finfo(np.float64).max
    outside = np.flatnonzero(measured)[~inside]
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
        unmet_edges=_unmet_edges(
            state, tuple(edge_keys[row] for row in outside.tolist()), lengths[outside]
        ),
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
) -> TetraMetricOutcome:
    """Execute quota-free native SPD split/collapse/flip/ring/relocation phases.

    The route accepts an exact P1 source coordinate map, keeps source identities
    in the native handle, and returns one generalized topology edit for the
    canonical transfer/certification owner. It never infers source ancestry from
    output samples.
    """
    if not isinstance(predicate_mode, PredicateMode):
        raise TypeError("predicate_mode must be PredicateMode.")
    if not isinstance(topology_operations, bool) or not isinstance(relocation, bool):
        raise TypeError("topology_operations and relocation must be bool.")
    metric_array = np.asarray(metric_values)
    state = _prepare_native_state(
        source,
        metric_array,
        cell_classes=cell_classes,
        facet_classes=facet_classes,
        edge_classes=edge_classes,
        protected_edges=protected_edges,
        fixed_vertices=fixed_vertices,
    )
    counters = {
        "splits": 0,
        "collapses": 0,
        "flips23": 0,
        "flips32": 0,
        "ring_reconnections": 0,
        "relocations": 0,
        "rejected_topology": 0,
        "rejected_protected": 0,
        "rejected_uncertain": 0,
        "rejected_invalid": 0,
    }
    passes = 0
    status = TetraMetricStatus.COMPLETE
    while True:
        converged, _, _, _, _ = _unit_metric_status(state)
        if converged:
            status = TetraMetricStatus.COMPLETE
            break
        passes += 1
        applied = 0
        if topology_operations:
            applied += _split_phase(state, counters)
            applied += _collapse_phase(state, counters)
            for kind, name in (
                (TetMeshEditKind.FLIP23, "flips23"),
                (TetMeshEditKind.FLIP32, "flips32"),
                (TetMeshEditKind.RECONNECT, "ring_reconnections"),
            ):
                applied += _replacement_phase(state, kind, name, counters, predicate_mode)
        if relocation:
            applied += _relocation_phase(state, counters)
        if applied == 0:
            status = TetraMetricStatus.STALLED
            break
    edit, target_metric = _assemble_native_edit(state)
    evidence = _tetra_evidence(state, counters, passes, status)
    return TetraMetricOutcome(edit, target_metric, evidence)


def _intermediate_relations(
    source: CellMesh,
    dimension: int,
    target_keys: list[tuple[int, ...]],
    stencils: dict[int, dict[int, Fraction]],
    /,
) -> EntityRelations:
    source_rows = entity_keys(source, dimension)
    source_keys = [
        tuple(int(value) for value in row if int(value) >= 0)
        for row in source_rows.tolist()
    ]
    rows_by_vertex: dict[int, set[int]] = defaultdict(set)
    for row, key in enumerate(source_keys):
        for vertex in key:
            rows_by_vertex[vertex].add(row)
    relations: set[tuple[tuple[int, ...], tuple[int, ...], EntityLineageKind]] = set()
    for target_key in target_keys:
        support: set[int] = set()
        for vertex in target_key:
            support.update(stencils[int(vertex)])
        if not support or len(support) > dimension + 1:
            continue
        candidates: set[int] | None = None
        for vertex in support:
            rows = rows_by_vertex.get(vertex, set())
            candidates = set(rows) if candidates is None else candidates & rows
            if not candidates:
                break
        for row in candidates or ():
            source_key = source_keys[row]
            if support <= set(source_key):
                kind = (
                    EntityLineageKind.PRESERVED
                    if source_key == target_key
                    else EntityLineageKind.REFINED_FROM
                )
                relations.add((source_key, target_key, kind))
    source_set = set(source_keys)
    for key in target_keys:
        if key in source_set:
            relations.add((key, key, EntityLineageKind.PRESERVED))
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
