#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Remesh embedded triangle surfaces against a source chart metric.

Per-cell charts preserve seams and pole occurrences. Batched source evaluation
and projection choose candidate coordinates. Publication reconstructs and
certifies the successor coordinate map through the geometry transition.
"""

from __future__ import annotations

import copy as _owner_copy
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
from math import isfinite, lcm, sqrt
from types import MappingProxyType
from typing import cast, NamedTuple, Protocol

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import phydrax._geometry_predicates as _dependency_phydrax__geometry_predicates
import phydrax.discretization._cell_geometry_transfer as _dependency_phydrax_discretization__cell_geometry_transfer
import phydrax.discretization._cell_geometry_transition_types as transition_types_owner
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._chart_reconstruction as chart_reconstruction_owner
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._coordinate_enclosure as _dependency_phydrax_discretization__coordinate_enclosure
import phydrax.discretization._coordinate_enclosure as _owner_phydrax_discretization__coordinate_enclosure
import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
import phydrax.discretization._sphere_chart_deformation as _dependency_phydrax_discretization__sphere_chart_deformation
import phydrax.discretization._sphere_chart_records as sphere_chart_records_owner
import phydrax.discretization._surface_chart_deformation as _dependency_phydrax_discretization__surface_chart_deformation
import phydrax.discretization._surface_chart_records as surface_chart_records_owner
import phydrax.geometry._domain_boundary as _dependency_phydrax_geometry__domain_boundary
import phydrax.geometry._domain_numerics as _owner_phydrax_geometry__domain_numerics
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.geometry._global_embedding as _dependency_phydrax_geometry__mesh_certificates
import phydrax.geometry._meshing_domain as _dependency_phydrax_geometry__meshing_domain
import phydrax.geometry._meshing_domain as _owner_phydrax_geometry__meshing_domain
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.geometry._sphere_material_atlas as _dependency_phydrax_geometry__sphere_material_atlas
import phydrax.geometry._surface_source_support as _owner_phydrax_geometry__surface_source_support
import phydrax.geometry.brep._patches as _dependency_phydrax_geometry_brep__patches
import phydrax.linalg._hermitian_spectral as _dependency_phydrax_linalg__hermitian_spectral
import phydrax.linalg._small_batched as _dependency_phydrax_linalg__small_batched
import phydrax.meshing._adaptation as _owner_phydrax_meshing__adaptation
import phydrax.meshing._association as _dependency_phydrax_meshing__association
import phydrax.meshing._curving as _dependency_phydrax_meshing__curving
import phydrax.meshing._entity_keys as entity_keys_owner
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._host_banks as _leaf_phydrax_meshing__host_banks
import phydrax.meshing._layer_core as _owner_phydrax_meshing__layer_core
import phydrax.meshing._lineage_records as _dependency_phydrax_meshing__lineage
import phydrax.meshing._metric as _dependency_phydrax_meshing__metric
import phydrax.meshing._periodic as _dependency_phydrax_meshing__periodic
import phydrax.meshing._periodic as _owner_phydrax_meshing__periodic
import phydrax.meshing._result as _dependency_phydrax_meshing__result
import phydrax.meshing._surface_association_transfer as _owner_phydrax_meshing__surface_association_transfer
import phydrax.meshing._surface_boundary_sources as surface_boundary_sources_owner
import phydrax.meshing._tetra_metric as _dependency_phydrax_meshing__tetra_metric
import phydrax.meshing._topology_edit as _dependency_phydrax_meshing__topology_edit
import phydrax.meshing._trace as _dependency_phydrax_meshing__trace
from phydrax.geometry._rational import fraction_endpoint

from .._bvh import bvh_overlap_pair_blocks, PackedBVH, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import exact_orient3d, NativeExecutionScope
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ..geometry._rational import rational_vector
from ..linalg import (
    hermitian_exp_enclosure,
    hermitian_log_enclosure,
    HermitianFunctionEnclosure,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ..typing import checked
from ._measurements import measure_phase, NativeMeshingPhase, NativeMeshingPhaseRecorder
from ._metric_execution import (
    metric_execution_passes,
    MetricExecutionSchedule,
    MetricOperationCounts,
    validate_metric_operation_controls,
    validate_metric_phase_recorder,
    validate_metric_quality_floor,
)


_LOWER = 1.0 / np.sqrt(2.0)
_UPPER = np.sqrt(2.0)
_LOCAL_EDGES = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int32)
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(8)
_PARAMETERS = (_GAUSS_NODES + 1.0) / 2.0
_WEIGHTS = _GAUSS_WEIGHTS / 2.0


class SurfaceMetricOutcome(NamedTuple):
    geometry: SurfaceMetricGeometryWitness
    metric: np.ndarray
    evidence: _dependency_phydrax_meshing__tetra_metric.MetricRemeshingEvidence
    periodic_stage: (
        _dependency_phydrax_meshing__periodic.PeriodicMetricOrbitOutcome | None
    ) = None


class SurfaceMetricGeometryWitness(NamedTuple):
    """Bind staged edits, material charts and scientific source identities."""

    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    cell_patches: np.ndarray
    cell_charts: np.ndarray
    cell_material_charts: np.ndarray
    source_cell_ids: np.ndarray
    source_material_charts: np.ndarray
    vertex_source_dimensions: np.ndarray | None
    vertex_source_indices: np.ndarray | None
    vertex_source_parameters: np.ndarray | None
    source_id: str
    source_revision: str
    domain_id: str
    cell_geometry_entity_ids: tuple[str, ...]
    cell_occurrence_paths: tuple[tuple[str, ...], ...]


class SurfaceSource(NamedTuple):
    cells: np.ndarray
    charts: np.ndarray
    patches: np.ndarray
    cell_ids: np.ndarray
    vertex_ids: np.ndarray
    metric: np.ndarray
    physical_charts: np.ndarray


@dataclass(slots=True)
class SurfaceState:
    points: np.ndarray
    metric: np.ndarray
    cells: np.ndarray
    patches: np.ndarray
    charts: np.ndarray
    classes: np.ndarray
    cell_ids: np.ndarray
    vertex_ids: np.ndarray
    parents: list[set[int]]
    vertex_sources: list[set[int]]
    next_cell_id: int
    next_vertex_id: int
    vertex_successor: dict[int, int] | None = None
    cell_kinds: dict[int, int] | None = None
    relocated_vertices: set[int] | None = None
    curve_witness: (
        _owner_phydrax_meshing__surface_association_transfer.PreparedSurfaceCurveWitness
        | None
    ) = None
    vertex_strata: list[tuple[int, int, np.ndarray]] | None = None
    curve_workspace_vertices: int = 0
    material_charts: np.ndarray | None = None
    material_workspace_bytes_upper: int = 0
    material_fraction_bits: int = 0


class SurfaceMetricControllerProtocol(Protocol):
    """Private orbit staging interface; scientific arrays remain owned by the state."""

    source: _dependency_phydrax_meshing__result.CellMeshingResult
    witness: surface_chart_records_owner.SurfaceChartWitness
    background: SurfaceSource | None
    prior_stage: _dependency_phydrax_meshing__periodic.PeriodicMetricOrbitOutcome | None
    minimum_metric_quality: float

    @property
    def operation_count(self) -> int: ...

    def stage(
        self,
        state: SurfaceState,
        operation: _dependency_phydrax_meshing__periodic.PeriodicMetricOperation,
        vertices: tuple[int, ...],
        fixed: np.ndarray,
        features: set[tuple[int, int]],
        feature_codes: dict[tuple[int, int], int],
        blocked: set[tuple[int, int]],
        /,
    ) -> SurfaceState | None: ...


def _admit_work_bound(
    work_upper: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Refuse an insufficient explicit batch before executing its host work."""

    if work_upper > budget.ticket.remaining_work_units:
        budget.ticket.refuse(work_upper)


def _consume_native_work(
    work: int = 0,
    /,
    *,
    geometry_queries: int = 0,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Admit and charge host work against the shared coordinate ticket."""

    ticket = budget.ticket
    if (
        work > ticket.remaining_work_units
        or geometry_queries > ticket.remaining_geometry_queries
    ):
        ticket.refuse(work, queries_upper=geometry_queries)
    ticket.consume(work, geometry_queries=geometry_queries)


def _admit_cavity(
    cavity_cells: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Admit the live cavity before carrier allocation or mutation."""

    if cavity_cells > budget.ticket.cavity_upper:
        budget.ticket.refuse(0, cavity_upper=cavity_cells)
    budget.ticket.consume(cavity_cells=cavity_cells)


def _edge_table(state: SurfaceState, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = np.sort(state.cells[:, _LOCAL_EDGES].reshape((-1, 2)), axis=1)
    edges, first, inverse = np.unique(
        records, axis=0, return_index=True, return_inverse=True
    )
    return edges, first, inverse.reshape((-1, 3))


def _mapped_quality(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    metric: np.ndarray,
    cells: np.ndarray,
    patches: np.ndarray,
    charts: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Mean ratio of the physical map differential, never a chord/UV surrogate."""
    _consume_native_work(geometry_queries=cells.shape[0], budget=budget)
    tangent = np.zeros((cells.shape[0], 3, 3), dtype=np.float64)
    for patch in np.unique(patches).tolist():
        selected = patches == patch
        differential = np.asarray(
            _dependency_phydrax_geometry_brep__patches.surface_differential(
                domain.patches[patch].surface,
                jnp.asarray(np.mean(charts[selected], axis=1), dtype=jnp.float64),
            ),
            dtype=np.float64,
        )
        tangent[selected, 1:] = (
            charts[selected, 1:] - charts[selected, :1]
        ) @ np.swapaxes(differential, -1, -2)
    mean = np.asarray(
        _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
            metric[cells], np.full(cells.shape, 1.0 / 3.0, dtype=np.float64)
        ),
        dtype=np.float64,
    )
    return np.asarray(
        _dependency_phydrax_meshing__metric.metric_shape_quality(
            jnp.asarray(mean), jnp.asarray(tangent), dimension=2
        ),
        dtype=np.float64,
    )


def _exact_uv(values: np.ndarray, /) -> tuple[Fraction, Fraction]:
    first, second = values
    return (
        first if isinstance(first, Fraction) else Fraction(float(first)),
        second if isinstance(second, Fraction) else Fraction(float(second)),
    )


class _MetricArcPiece(NamedTuple):
    edge: int
    patch: int
    first: tuple[Fraction, Fraction]
    velocity: tuple[Fraction, Fraction]
    material_first: tuple[Fraction, Fraction]
    material_velocity: tuple[Fraction, Fraction]
    lower: Fraction
    upper: Fraction
    triangle: tuple[tuple[Fraction, ...], ...]
    vertices: tuple[int, ...]


def _metric_arc_pieces(
    state: SurfaceState,
    source: SurfaceSource,
    maximum_pairs: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[list[_MetricArcPiece], int]:
    """Clip edge intervals against the source chart background and retain their cell ownership."""
    edges, first, _ = _edge_table(state)
    owner, local = first // 3, first % 3
    charts = state.charts if state.material_charts is None else state.material_charts
    endpoints = charts[owner[:, None], _LOCAL_EDGES[local]]
    actual_endpoints = state.charts[owner[:, None], _LOCAL_EDGES[local]]
    pieces, work = [], 0
    for edge, uv in enumerate(endpoints):
        origin = _exact_uv(uv[0])

        matching = np.flatnonzero(
            source.patches == int(state.patches[owner[edge]])
        ).tolist()
        _admit_work_bound(len(matching), budget=budget)
        last = _exact_uv(uv[1])
        direction = (last[0] - origin[0], last[1] - origin[1])
        actual_origin, actual_last = (
            _exact_uv(actual_endpoints[edge, 0]),
            _exact_uv(actual_endpoints[edge, 1]),
        )
        actual_direction = (
            actual_last[0] - actual_origin[0],
            actual_last[1] - actual_origin[1],
        )
        candidates = []
        patch = int(state.patches[owner[edge]])
        previous_work = work
        for cell in matching:
            work += 1
            if work > maximum_pairs:
                _consume_native_work(
                    work - previous_work, geometry_queries=0, budget=budget
                )
                raise _dependency_phydrax_meshing__tetra_metric.source_location_budget(
                    maximum_pairs, work
                )
            triangle = tuple(_exact_uv(point) for point in source.charts[cell])
            determinant = (triangle[1][0] - triangle[0][0]) * (
                triangle[2][1] - triangle[0][1]
            ) - (triangle[1][1] - triangle[0][1]) * (triangle[2][0] - triangle[0][0])
            sign = 1 if determinant > 0 else -1
            lower, upper = Fraction(0), Fraction(1)
            for slot in range(3):
                a, b = triangle[slot], triangle[(slot + 1) % 3]
                x, y = b[0] - a[0], b[1] - a[1]
                constant = sign * (x * (origin[1] - a[1]) - y * (origin[0] - a[0]))
                slope = sign * (x * direction[1] - y * direction[0])
                if slope > 0:
                    lower = max(lower, -constant / slope)
                elif slope < 0:
                    upper = min(upper, -constant / slope)
                elif constant < 0:
                    upper = lower
                    break
            if lower < upper:
                candidates.append((lower, upper, cell, triangle))
        _consume_native_work(len(matching), geometry_queries=0, budget=budget)
        breaks = sorted(
            {
                Fraction(0),
                Fraction(1),
                *(value for candidate in candidates for value in candidate[:2]),
            }
        )
        for lower, upper in zip(breaks[:-1], breaks[1:], strict=True):
            midpoint = (lower + upper) / 2
            selected = [
                candidate
                for candidate in candidates
                if candidate[0] <= midpoint <= candidate[1]
            ]
            if not selected:
                raise _leaf_phydrax_meshing__failures.MeshingFailure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
                    "surface_metric.metric_arc.material_interval must remain in the chart-field background; got an uncovered interval",
                    stage=_dependency_phydrax_meshing__trace.MeshingStageKind.LINEAGE_CONSTRUCTION.value,
                )
            chosen = min(selected, key=lambda candidate: source.cell_ids[candidate[2]])
            vertices = tuple(int(value) for value in source.cells[chosen[2]])
            pieces.append(
                _MetricArcPiece(
                    edge,
                    patch,
                    actual_origin,
                    actual_direction,
                    origin,
                    direction,
                    lower,
                    upper,
                    chosen[3],
                    vertices,
                )
            )
    return pieces, work


def _arc_metric_enclosure(
    piece: _MetricArcPiece,
    source: SurfaceSource,
    logarithms: dict[int, HermitianFunctionEnclosure],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[Fraction, ...], ...], Fraction]:
    tensors = source.metric[np.asarray(piece.vertices, dtype=np.int64)]
    if np.array_equal(tensors, np.broadcast_to(tensors[0], tensors.shape)):
        return tuple(rational_vector(row) for row in tensors[0]), Fraction(0)
    a, b, c = piece.triangle
    matrix = ((b[0] - a[0], c[0] - a[0]), (b[1] - a[1], c[1] - a[1]))
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    weights = []
    for parameter in (piece.lower, piece.upper):
        point = tuple(
            piece.material_first[axis]
            + parameter * piece.material_velocity[axis]
            - a[axis]
            for axis in range(2)
        )
        u = (point[0] * matrix[1][1] - point[1] * matrix[0][1]) / determinant
        v = (matrix[0][0] * point[1] - matrix[1][0] * point[0]) / determinant
        weights.append((1 - u - v, u, v))
    centers = tuple((first + last) / 2 for first, last in zip(*weights, strict=True))
    radii = tuple(abs(first - last) / 2 for first, last in zip(*weights, strict=True))
    logs = tuple(logarithms[vertex] for vertex in piece.vertices)
    nominal = tuple(
        tuple(
            sum(
                (
                    weight * log.matrix[i][j]
                    for weight, log in zip(centers, logs, strict=True)
                ),
                Fraction(0),
            )
            for j in range(3)
        )
        for i in range(3)
    )
    slope = tuple(
        tuple(
            sum(
                (
                    (last - first) * log.matrix[i][j] / 2
                    for first, last, log in zip(weights[0], weights[1], logs, strict=True)
                ),
                Fraction(0),
            )
            for j in range(3)
        )
        for i in range(3)
    )

    error = _dependency_phydrax_linalg__hermitian_spectral.matrix_norm_enclosure(
        slope, budget=budget
    ) + sum(
        (
            (abs(center) + radius) * log.error
            for center, radius, log in zip(centers, radii, logs, strict=True)
        ),
        Fraction(0),
    )
    return hermitian_exp_enclosure(
        HermitianFunctionEnclosure(nominal, error), budget=budget
    )


def _arc_piece_bounds(
    piece: _MetricArcPiece,
    lower: np.ndarray,
    upper: np.ndarray,
    metric: tuple[tuple[Fraction, ...], ...],
    metric_error: Fraction,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[Fraction, Fraction]:
    tangent = []
    for component in range(3):
        low = high = Fraction(0)
        for axis, velocity in enumerate(piece.velocity):
            a, b = (
                Fraction(float(lower[component, axis])) * velocity,
                Fraction(float(upper[component, axis])) * velocity,
            )
            low, high = low + min(a, b), high + max(a, b)
        tangent.append((low, high))
    gram_low = gram_high = Fraction(0)
    for i in range(3):
        for j in range(3):
            if i == j:
                a, b = tangent[i]
                products = [
                    Fraction(0) if a <= 0 <= b else min(a * a, b * b),
                    max(a * a, b * b),
                ]
            else:
                products = [a * b for a in tangent[i] for b in tangent[j]]
            values = [metric[i][j] * value for value in products]
            gram_low += min(values)
            gram_high += max(values)
    magnitude = sum((max(abs(a), abs(b)) ** 2 for a, b in tangent), Fraction(0))
    radius = metric_error * magnitude

    low = _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
        max(Fraction(0), gram_low - radius), budget=budget
    )[0]
    high = _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
        max(Fraction(0), gram_high + radius), budget=budget
    )[1]
    width = piece.upper - piece.lower
    return width * low, width * high


def _fraction_sqrt_lower(
    value: Fraction,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Fraction:
    """One-sided rational radical proof without constructing an unused upper bound."""
    if value <= 0:
        return Fraction(0)
    _dependency_phydrax_linalg__hermitian_spectral.reserve_fraction_work(
        budget,
        1,
        8,
        max(abs(value.numerator).bit_length(), value.denominator.bit_length(), 1075),
    )
    lower = sqrt(float(value))
    if not isfinite(lower) or lower == 0:
        raise ValueError(
            "surface_metric.embedded_measure must remain in the certified binary64 range; got unrepresentable radical"
        )
    while Fraction(lower) ** 2 > value:
        budget.reserve(1)
        lower = float(np.nextafter(lower, -np.inf))
    return Fraction(lower)


def _fraction_sqrt_upper(
    value: Fraction,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Fraction:
    """One-sided rational radical proof without constructing an unused lower bound."""
    if value <= 0:
        return Fraction(0)
    _dependency_phydrax_linalg__hermitian_spectral.reserve_fraction_work(
        budget,
        1,
        8,
        max(abs(value.numerator).bit_length(), value.denominator.bit_length(), 1075),
    )
    upper = sqrt(float(value))
    if not isfinite(upper) or upper == 0:
        raise ValueError(
            "surface_metric.embedded_measure must remain in the certified binary64 range; got unrepresentable radical"
        )
    while Fraction(upper) ** 2 < value:
        budget.reserve(1)
        upper = float(np.nextafter(upper, np.inf))
    return Fraction(upper)


def _constant_metric_tangent_upper(
    direction: tuple[Fraction, Fraction],
    lower: np.ndarray,
    upper: np.ndarray,
    matrix: tuple[tuple[Fraction, ...], ...],
    ceiling: Fraction | None,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Fraction:
    tangent = []
    for component in range(3):
        interval = Fraction(0), Fraction(0)
        for axis in range(2):
            values = (
                Fraction(float(lower[component, axis])) * direction[axis],
                Fraction(float(upper[component, axis])) * direction[axis],
            )
            interval = (
                interval[0] + min(values),
                interval[1] + max(values),
            )
        tangent.append(interval)
    gram_upper = Fraction(0)
    for i in range(3):
        for j in range(3):
            products = tuple(a * b for a in tangent[i] for b in tangent[j])
            gram_upper += max(matrix[i][j] * value for value in products)
    gram_upper = max(Fraction(0), gram_upper)
    if ceiling is not None and gram_upper <= ceiling * ceiling:
        return ceiling
    return _fraction_sqrt_upper(gram_upper, budget=budget)


def _constant_metric_subdivided_arc_upper(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patch: int,
    charts: np.ndarray,
    direction: tuple[Fraction, Fraction],
    matrix: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[Fraction, int]:
    """Integrate derivative bounds in the source patch's fixed subdivision order."""
    divisions = (
        64
        if isinstance(
            domain.patches[patch].surface,
            _dependency_phydrax_geometry_brep__patches.SpherePatch,
        )
        else 3
    )
    exact_charts = tuple(rational_vector(point) for point in charts)
    subboxes = []
    for section in range(divisions):
        interval = Fraction(section, divisions), Fraction(section + 1, divisions)
        endpoints = tuple(
            tuple(
                exact_charts[0][axis] + parameter * direction[axis] for axis in range(2)
            )
            for parameter in interval
        )
        subboxes.append(
            tuple(
                (
                    fraction_endpoint(
                        min(point[axis] for point in endpoints),
                        upper=False,
                        saturate=True,
                    ),
                    fraction_endpoint(
                        max(point[axis] for point in endpoints),
                        upper=True,
                        saturate=True,
                    ),
                )
                for axis in range(2)
            )
        )
    subboxes_ = np.asarray(subboxes, dtype=np.float64).transpose(0, 2, 1)
    sub_lower, sub_upper = domain.patches[patch].surface.derivative_bounds_batch(
        subboxes_, order=1, budget=budget
    )
    segment_direction = (
        direction[0] / divisions,
        direction[1] / divisions,
    )
    high = sum(
        (
            _constant_metric_tangent_upper(
                segment_direction, low, high_, matrix, None, budget=budget
            )
            for low, high_ in zip(sub_lower, sub_upper, strict=True)
        ),
        Fraction(0),
    )
    return high, divisions


# Fixed subdivision execution is separate from edge preparation and work settlement.
def _constant_surface_metric_arc_bounds(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: SurfaceState,
    metric: np.ndarray,
    lower_required: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, int]:
    """Bound constant-metric chords and source derivative integrals."""
    # Rational radical and subdivision order determine charged arithmetic visits.
    edges, first, _ = _edge_table(state)
    owner, local = first // 3, first % 3
    edge_charts = state.charts[owner[:, None], _LOCAL_EDGES[local]]
    patches = state.patches[owner]
    physical_delta = state.points[edges[:, 1]] - state.points[edges[:, 0]]
    chord_estimate = np.sqrt(
        np.einsum("ni,ij,nj->n", physical_delta, metric, physical_delta)
    )
    direct_subdivision = chord_estimate > 1.25
    derivative_lower = np.empty((edges.shape[0], 3, 2), dtype=np.float64)
    derivative_upper = np.empty_like(derivative_lower)
    boxes = np.stack(
        (
            np.min(edge_charts, axis=1),
            np.max(edge_charts, axis=1),
        ),
        axis=1,
    )
    base_rows = np.flatnonzero(~direct_subdivision)
    for patch in np.unique(patches[base_rows]).tolist():
        selected = base_rows[patches[base_rows] == patch]
        low, high = domain.patches[patch].surface.derivative_bounds_batch(
            boxes[selected], order=1, budget=budget
        )
        derivative_lower[selected] = low
        derivative_upper[selected] = high
    box_count = base_rows.size
    matrix = tuple(rational_vector(row) for row in metric)

    results = []
    arc_ceiling = Fraction(float(np.nextafter(_UPPER, -np.inf)))
    for edge_index, (edge, charts) in enumerate(zip(edges, edge_charts, strict=True)):
        direction = (
            Fraction(float(charts[1, 0])) - Fraction(float(charts[0, 0])),
            Fraction(float(charts[1, 1])) - Fraction(float(charts[0, 1])),
        )
        if direct_subdivision[edge_index]:
            high = Fraction(2)
        else:
            lower = derivative_lower[edge_index]
            upper = derivative_upper[edge_index]
            high = _constant_metric_tangent_upper(
                direction, lower, upper, matrix, arc_ceiling, budget=budget
            )
        if high > arc_ceiling:
            high, divisions = _constant_metric_subdivided_arc_upper(
                domain,
                int(patches[edge_index]),
                charts,
                direction,
                matrix,
                budget=budget,
            )
            box_count += divisions
        delta = tuple(
            Fraction(float(state.points[int(edge[1]), axis]))
            - Fraction(float(state.points[int(edge[0]), axis]))
            for axis in range(3)
        )
        chord_squared = sum(
            (matrix[i][j] * delta[i] * delta[j] for i in range(3) for j in range(3)),
            Fraction(0),
        )
        if lower_required[edge_index]:
            low = _fraction_sqrt_lower(chord_squared, budget=budget)
        else:
            low = Fraction(0)
        results.append(
            (
                fraction_endpoint(low, upper=False, saturate=True),
                fraction_endpoint(high, upper=True, saturate=True),
            )
        )
    work = box_count
    _consume_native_work(work, geometry_queries=box_count, budget=budget)
    return np.asarray(results, dtype=np.float64), work


def _certify_surface_metric_arcs(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: SurfaceState,
    source: SurfaceSource,
    /,
    *,
    maximum_work: int,
    maximum_pairs: int,
    features: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    generated_from: int,
    maximum_subdivisions: int = 4096,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, int]:
    """Enclose source-derivative metric integrals outward."""
    # Rational interval subdivision is one numerical proof kernel. Its queue and
    # branch order determine enclosure tightness and native work settlement.
    constant_metric = np.array_equal(
        source.metric, np.broadcast_to(source.metric[0], source.metric.shape)
    )
    if constant_metric:
        edges, _, _ = _edge_table(state)
        lower_required = np.asarray(
            [
                tuple(map(int, edge)) not in features
                and tuple(map(int, edge)) not in blocked
                and not np.any(edge >= generated_from)
                for edge in edges
            ],
            dtype=np.bool_,
        )
        return _constant_surface_metric_arc_bounds(
            domain, state, source.metric[0], lower_required, budget=budget
        )
    pieces, work = _metric_arc_pieces(state, source, maximum_pairs, budget=budget)
    edges, _, _ = _edge_table(state)
    bounds = [[Fraction(0), Fraction(0)] for _ in edges]
    logarithms: dict[int, HermitianFunctionEnclosure] = {}

    for vertex, tensor in enumerate(source.metric):
        with budget.temporary_scope():
            logarithms[vertex] = hermitian_log_enclosure(tensor, budget=budget)
        budget.retain_basis((logarithms[vertex].matrix, logarithms[vertex].error))
    pending, visited = pieces, 0
    while pending:
        boxes = []
        for piece in pending:
            endpoints = tuple(
                tuple(
                    piece.first[axis] + parameter * piece.velocity[axis]
                    for axis in range(2)
                )
                for parameter in (piece.lower, piece.upper)
            )
            boxes.append(
                [
                    [
                        fraction_endpoint(
                            min(point[axis] for point in endpoints),
                            upper=False,
                            saturate=True,
                        )
                        for axis in range(2)
                    ],
                    [
                        fraction_endpoint(
                            max(point[axis] for point in endpoints),
                            upper=True,
                            saturate=True,
                        )
                        for axis in range(2)
                    ],
                ]
            )
        visits = len(pending)
        visited += visits
        work += visits
        if work > maximum_work or visited > maximum_subdivisions * edges.shape[0]:
            raise _dependency_phydrax_meshing__tetra_metric.source_location_budget(
                maximum_work, work
            )
        _consume_native_work(visits, geometry_queries=visits, budget=budget)
        lower = np.empty((visits, 3, 2), dtype=np.float64)
        upper = np.empty_like(lower)
        patches = np.asarray([piece.patch for piece in pending], dtype=np.int32)
        for patch in np.unique(patches).tolist():
            selected = np.flatnonzero(patches == patch)
            low, high = domain.patches[patch].surface.derivative_bounds_batch(
                np.asarray(boxes, dtype=np.float64)[selected], budget=budget
            )
            lower[selected], upper[selected] = low, high
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError(
                "surface_metric.certify_surface_metric_arcs must retain finite representable bounds; got source metric arcs require finite derivative enclosures"
            )
        remaining = []
        for row, piece in enumerate(pending):
            with budget.temporary_scope():
                metric, error = _arc_metric_enclosure(
                    piece, source, logarithms, budget=budget
                )
                low, high = _arc_piece_bounds(
                    piece, lower[row], upper[row], metric, error, budget=budget
                )
            if high - low <= (piece.upper - piece.lower) / 1024:
                bounds[piece.edge][0] += low
                bounds[piece.edge][1] += high
            else:
                midpoint = (piece.lower + piece.upper) / 2
                remaining.extend(
                    (piece._replace(upper=midpoint), piece._replace(lower=midpoint))
                )
        pending = remaining
    return np.asarray(
        [
            [
                fraction_endpoint(low, upper=False, saturate=True),
                fraction_endpoint(high, upper=True, saturate=True),
            ]
            for low, high in bounds
        ],
        dtype=np.float64,
    ), work


def _metric_inner_bounds(
    left: tuple[tuple[Fraction, Fraction], ...],
    right: tuple[tuple[Fraction, Fraction], ...],
    matrix: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[Fraction, Fraction]:
    lower = upper = Fraction(0)
    for first in range(3):
        for second in range(3):
            if left is right and first == second:
                a, b = left[first]
                products = (
                    Fraction(0) if a <= 0 <= b else min(a * a, b * b),
                    max(a * a, b * b),
                )
            else:
                products = tuple(a * b for a in left[first] for b in right[second])
            values = tuple(matrix[first][second] * value for value in products)
            lower, upper = lower + min(values), upper + max(values)
    return lower, upper


def _surface_quality_box_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    frame: tuple[tuple[Fraction, Fraction], ...],
    matrix: tuple[tuple[Fraction, ...], ...],
    condition_factor: Fraction,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[float, float]:

    budget.reserve(
        256,
        128
        * (
            64 + 2 * _dependency_phydrax__geometry_predicates.bigint_bytes(16 * 1075 + 64)
        ),
    )
    tangents = []
    for direction in frame:
        components = []
        for component in range(3):
            low = high = Fraction(0)
            for axis, velocity in enumerate(direction):
                values = (
                    Fraction(float(lower[component, axis])) * velocity,
                    Fraction(float(upper[component, axis])) * velocity,
                )
                low, high = low + min(values), high + max(values)
            components.append((low, high))
        tangents.append(tuple(components))
    first = _metric_inner_bounds(tangents[0], tangents[0], matrix)
    last = _metric_inner_bounds(tangents[1], tangents[1], matrix)
    mixed = _metric_inner_bounds(tangents[0], tangents[1], matrix)
    determinant_low = (
        max(Fraction(0), first[0]) * max(Fraction(0), last[0])
        - max(abs(value) for value in mixed) ** 2
    )
    mixed_square_low = (
        Fraction(0)
        if mixed[0] <= 0 <= mixed[1]
        else min(value * value for value in mixed)
    )
    determinant_high = first[1] * last[1] - mixed_square_low
    denominator_low, denominator_high = (
        first[0] + last[0] - mixed[1],
        first[1] + last[1] - mixed[0],
    )
    minimum, maximum = 0.0, 1.0
    if determinant_low > 0 and denominator_high > 0:
        bound = (
            _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
                3 * determinant_low, budget=budget
            )[0]
            / denominator_high
        )
        minimum = max(
            0.0, fraction_endpoint(bound * condition_factor, upper=False, saturate=True)
        )
    if determinant_high >= 0 and denominator_low > 0:
        bound = (
            _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
                3 * determinant_high, budget=budget
            )[1]
            / denominator_low
        )
        maximum = min(
            1.0, fraction_endpoint(bound / condition_factor, upper=True, saturate=True)
        )
    return minimum, maximum


def _certify_surface_metric_quality(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: SurfaceState,
    source: SurfaceSource,
    /,
    *,
    minimum_quality: float,
    maximum_work: int,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, int]:
    """Enclose the curved differential over each surface cell."""
    # The rational Gram proof retains its bounded subdivision/reduction order;
    # splitting formulas or interval decisions changes charged arithmetic visits.

    constant = np.array_equal(
        source.metric, np.broadcast_to(source.metric[0], source.metric.shape)
    )
    factor = Fraction(1)
    if not constant:
        spectra = tuple(
            _owner_phydrax_geometry__domain_numerics.gram_spectrum_bounds(value)
            for value in source.metric
        )
        low, high = min(value[0] for value in spectra), max(value[1] for value in spectra)
        if low <= 0:
            raise ValueError(
                "surface_metric.certify_surface_metric_quality must retain a nonsingular positive-definite source frame; got surface SPD supports lack a certified positive spectral lower bound"
            )
        factor = (
            _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
                Fraction(low), budget=budget
            )[0]
            / _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
                Fraction(high), budget=budget
            )[1]
        )
    tensor = source.metric[0] if constant else np.eye(3, dtype=np.float64)
    matrix = tuple(rational_vector(row) for row in tensor)
    result = np.zeros(state.cells.shape[0], dtype=np.float64)
    work = 0
    queue_upper = 0
    queue_bits = 1075
    for cell, (patch, chart) in enumerate(zip(state.patches, state.charts, strict=True)):
        triangle = cast(
            tuple[
                tuple[Fraction, Fraction],
                tuple[Fraction, Fraction],
                tuple[Fraction, Fraction],
            ],
            tuple(_exact_uv(point) for point in chart),
        )
        frame = (
            (
                triangle[1][0] - triangle[0][0],
                triangle[1][1] - triangle[0][1],
            ),
            (
                triangle[2][0] - triangle[0][0],
                triangle[2][1] - triangle[0][1],
            ),
        )
        pending = [triangle]
        accepted = []
        refused = False
        while pending:
            upper = (
                256
                + (len(pending) + 1)
                * 6
                * (
                    56
                    + 2
                    * _dependency_phydrax__geometry_predicates.bigint_bytes(queue_bits)
                )
                + 40 * (len(accepted) + 1)
            )
            budget.reserve(0, max(0, upper - queue_upper))
            queue_upper = max(queue_upper, upper)
            panel = pending.pop()
            work += 1
            if work > maximum_work:
                raise _dependency_phydrax_meshing__tetra_metric.source_location_budget(
                    maximum_work, work
                )
            box = np.asarray(
                [
                    [
                        fraction_endpoint(
                            min(point[axis] for point in panel),
                            upper=False,
                            saturate=True,
                        )
                        for axis in range(2)
                    ],
                    [
                        fraction_endpoint(
                            max(point[axis] for point in panel), upper=True, saturate=True
                        )
                        for axis in range(2)
                    ],
                ]
            )
            _consume_native_work(1, geometry_queries=1, budget=budget)
            lower, upper = domain.patches[int(patch)].surface.derivative_bounds_batch(
                box[None], budget=budget
            )
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                raise ValueError(
                    "surface_metric.certify_surface_metric_quality must retain finite representable bounds; got whole-cell surface quality requires finite derivative enclosures"
                )
            with budget.temporary_scope():
                minimum, maximum = _surface_quality_box_bounds(
                    lower[0], upper[0], frame, matrix, factor, budget=budget
                )
            if maximum < minimum_quality:
                refused = True
                break
            if minimum >= minimum_quality:
                accepted.append(minimum)
                continue
            a, b, c = panel
            with _dependency_phydrax__geometry_predicates.exact_charge(6, budget=budget):
                queue_bits = max(
                    queue_bits,
                    max(
                        max(value.numerator.bit_length(), value.denominator.bit_length())
                        for point in panel
                        for value in point
                    )
                    + 2,
                )
            # UV inputs are dyadic binary64 values; dyadic midpoint
            # denominators grow by at most one bit. Admit the live frontier
            # before constructing its children, not a rounded carrier surrogate.
            upper = (
                256
                + (len(pending) + 5)
                * 6
                * (
                    56
                    + 2
                    * _dependency_phydrax__geometry_predicates.bigint_bytes(queue_bits)
                )
                + 40 * (len(accepted) + 1)
            )
            budget.reserve(0, max(0, upper - queue_upper))
            queue_upper = max(queue_upper, upper)
            with _dependency_phydrax__geometry_predicates.exact_charge(18, budget=budget):
                ab = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                bc = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
                ca = ((c[0] + a[0]) / 2, (c[1] + a[1]) / 2)
            pending.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        if not refused:
            result[cell] = min(accepted)
    return result, work


def _measure(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: SurfaceState,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges, first, _ = _edge_table(state)
    owner, local = first // 3, first % 3
    uv = state.charts[owner[:, None], _LOCAL_EDGES[local]]
    reversed_edges = state.cells[owner, _LOCAL_EDGES[local, 0]] != edges[:, 0]
    uv[reversed_edges] = uv[reversed_edges, ::-1]
    velocity = uv[:, 1] - uv[:, 0]
    charts = uv[:, :1] + _PARAMETERS[None, :, None] * velocity[:, None]
    patches = state.patches[owner]
    samples = np.asarray(
        _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
            np.broadcast_to(state.metric[edges][:, None], (edges.shape[0], 8, 2, 3, 3)),
            np.broadcast_to(
                np.stack((1.0 - _PARAMETERS, _PARAMETERS), axis=1), (edges.shape[0], 8, 2)
            ),
        ),
        dtype=np.float64,
    )
    tangent = np.zeros((edges.shape[0], 8, 3), dtype=np.float64)
    _consume_native_work(geometry_queries=edges.shape[0] * 8, budget=budget)
    for patch in np.unique(patches).tolist():
        selected = patches == patch
        differential = np.asarray(
            _dependency_phydrax_geometry_brep__patches.surface_differential(
                domain.patches[patch].surface,
                jnp.asarray(charts[selected].reshape((-1, 2)), dtype=jnp.float64),
            ),
            dtype=np.float64,
        ).reshape((-1, 8, 3, 2))
        tangent[selected] = (differential @ velocity[selected, None, :, None])[..., 0]
    speed = np.sqrt(np.sum(tangent * (samples @ tangent[..., None])[..., 0], axis=-1))
    lengths = speed @ _WEIGHTS
    quality = _mapped_quality(
        domain, state.metric, state.cells, state.patches, state.charts, budget=budget
    )
    fidelity = np.zeros((state.cells.shape[0],), dtype=np.float64)
    for patch in np.unique(state.patches).tolist():
        selected = state.patches == patch
        fidelity[selected] = domain.interpolation_bounds(
            patch,
            state.charts[selected],
            budget=budget,
            reserve_queries=lambda count: _consume_native_work(
                geometry_queries=count, budget=budget
            ),
        )
    return lengths, quality, fidelity


def _legal(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    cells: np.ndarray,
    patches: np.ndarray,
    charts: np.ndarray,
    points: np.ndarray,
    maximum_fidelity: float,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    predicate = _dependency_phydrax__geometry_predicates.orient2d(
        charts[:, 0],
        charts[:, 1],
        charts[:, 2],
        mode=_dependency_phydrax__geometry_predicates.PredicateMode.EXACT,
        budget=budget,
    )
    orientation = np.asarray(
        [domain.patches[patch].orientation for patch in patches.tolist()], dtype=np.int8
    )
    if not np.all(
        np.asarray(predicate.certain) & (np.asarray(predicate.signs) == orientation)
    ):
        return False
    normals, regular = domain.oriented_normals(patches, np.mean(charts, axis=1))
    physical = points[cells]
    signs = _dependency_phydrax__geometry_predicates.orient3d(
        physical[:, 0],
        physical[:, 1],
        physical[:, 2],
        physical[:, 0] + normals,
        mode=_dependency_phydrax__geometry_predicates.PredicateMode.EXACT,
        budget=budget,
    )
    if not np.all(
        regular
        & np.asarray(signs.certain)
        & (
            np.asarray(signs.signs)
            == _dependency_phydrax__geometry_predicates.PredicateSign.POSITIVE
        )
    ):
        return False
    for patch in np.unique(patches).tolist():
        if np.any(
            domain.interpolation_bounds(
                patch,
                charts[patches == patch],
                budget=budget,
                reserve_queries=lambda count: _consume_native_work(
                    geometry_queries=count, budget=budget
                ),
            )
            > maximum_fidelity
        ):
            return False
    return True


def _admit_material_carrier(
    state: SurfaceState,
    charts: np.ndarray,
    cells: int,
    /,
    *,
    midpoint: bool = False,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Admit the live material bank and its rebuild workspace on the source ledger."""
    input_bits = _dependency_phydrax__geometry_predicates.exact_bits(
        charts, budget=budget
    )
    bits = max(
        state.material_fraction_bits, 2 * input_bits + 4 if midpoint else input_bits
    )
    upper = 256 + 12 * cells * (
        56 + 2 * _dependency_phydrax__geometry_predicates.bigint_bytes(bits)
    )

    budget.reserve(0, max(0, upper - state.material_workspace_bytes_upper))
    state.material_fraction_bits = bits
    state.material_workspace_bytes_upper = max(
        state.material_workspace_bytes_upper, upper
    )


def _material_legal(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patches: np.ndarray,
    charts: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    """Check material orientation with the integer-bank predicate."""
    bits = _dependency_phydrax__geometry_predicates.exact_bits(charts, budget=budget)
    with _dependency_phydrax__geometry_predicates.exact_charge(
        12 * charts.shape[0],
        18
        * charts.shape[0]
        * _dependency_phydrax__geometry_predicates.bigint_bytes(7 * bits + 16),
        budget=budget,
    ):
        integers = np.empty(charts.shape, dtype=object)
        for row, triangle in enumerate(charts):
            denominator = lcm(*(value.denominator for value in triangle.flat))
            for slot, point in enumerate(triangle):
                for axis, value in enumerate(point):
                    integers[row, slot, axis] = value.numerator * (
                        denominator // value.denominator
                    )
        predicate = _dependency_phydrax__geometry_predicates.orient2d(
            integers[:, 0],
            integers[:, 1],
            integers[:, 2],
            mode=_dependency_phydrax__geometry_predicates.PredicateMode.EXACT,
            budget=budget,
        )
    return bool(
        np.all(np.asarray(predicate.certain))
        and np.all(
            np.asarray(predicate.signs)
            == np.asarray([domain.patches[int(patch)].orientation for patch in patches])
        )
    )


def _replace(
    state: SurfaceState,
    removed: np.ndarray,
    cells: np.ndarray,
    patches: np.ndarray,
    charts: np.ndarray,
    classes: np.ndarray,
    parents: list[set[int]],
    kind: _dependency_phydrax_meshing__lineage.EntityLineageKind,
    /,
    *,
    material_charts: np.ndarray | None = None,
    cell_ids: np.ndarray | None = None,
) -> None:
    keep = np.ones((state.cells.shape[0],), dtype=np.bool_)
    keep[removed] = False
    if state.material_charts is not None and material_charts is None:
        raise ValueError(
            "surface_metric.replace must retain source geometry invariants; got a material topology edit must retain its chart carrier"
        )
    ids = (
        _dependency_phydrax_meshing__tetra_metric.fresh_identifiers(
            state.next_cell_id, cells.shape[0]
        )
        if cell_ids is None
        else cell_ids
    )
    if state.cell_kinds is None:
        state.cell_kinds = {}
    kind = _dependency_phydrax_meshing__tetra_metric.lineage_operation_kind(
        kind,
        tuple(
            state.cell_kinds.get(
                int(identifier),
                int(_dependency_phydrax_meshing__lineage.EntityLineageKind.PRESERVED),
            )
            for identifier in state.cell_ids[removed]
        ),
    )
    for identifier in state.cell_ids[removed].tolist():
        state.cell_kinds.pop(identifier, None)
    state.cell_kinds.update((int(identifier), int(kind)) for identifier in ids)
    state.next_cell_id += cells.shape[0]
    state.cells = np.concatenate((state.cells[keep], cells))
    state.patches = np.concatenate((state.patches[keep], patches))
    state.charts = np.concatenate((state.charts[keep], charts))
    if state.material_charts is not None:
        assert material_charts is not None
        state.material_charts = np.concatenate(
            (state.material_charts[keep], material_charts)
        )
    state.classes = np.concatenate((state.classes[keep], classes))
    state.cell_ids = np.concatenate((state.cell_ids[keep], ids))
    state.parents = [
        parent for row, parent in enumerate(state.parents) if keep[row]
    ] + parents


def _curve_occurrence_chart(
    patch: int,
    anchors: tuple[tuple[np.ndarray, tuple[tuple[int, int, np.ndarray], ...]], ...],
    target_uses: tuple[tuple[int, int, np.ndarray], ...],
    tolerance: float,
    /,
) -> np.ndarray | None:
    """Select the coedge use, never a nearest periodic representative."""
    candidates = []
    for target_patch, occurrence, chart in target_uses:
        if target_patch != patch:
            continue
        if all(
            any(
                old_patch == patch
                and old_occurrence == occurrence
                and np.linalg.norm(old_chart - retained) <= tolerance
                for old_patch, old_occurrence, old_chart in old_uses
            )
            for retained, old_uses in anchors
        ):
            candidates.append(chart)
    return candidates[0] if len(candidates) == 1 else None


@checked
def split(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: SurfaceState,
    edge: np.ndarray,
    new_metric: np.ndarray,
    /,
    *,
    source_curve: bool = False,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    rows = np.flatnonzero(np.sum(np.isin(state.cells, edge), axis=1) == 2)
    if rows.size == 0:
        return False

    _admit_cavity(int(rows.size), budget=budget)
    token = None
    chart_midpoints = np.asarray(
        [
            np.mean(state.charts[row, np.isin(state.cells[row], edge)], axis=0)
            for row in rows
        ],
        dtype=np.float64,
    )
    witness, strata = state.curve_witness, state.vertex_strata

    if strata is not None:
        # A live token and its in-flight replacement each carry a conservative
        # 512-byte CPython upper bound, not native measured storage. Rejected
        # attempts reuse the admitted slot rather than accumulating dead tokens.
        required = int(state.points.shape[0]) + 1
        budget.reserve(1, 1024 * max(0, required - state.curve_workspace_vertices))
        state.curve_workspace_vertices = max(state.curve_workspace_vertices, required)
    if source_curve and witness is not None:
        if strata is None:
            raise ValueError(
                "surface_metric.split must retain source-stratum parameters and incidence; got an curve edit lacks its retained vertex strata"
            )
        tokens = [strata[int(vertex)] for vertex in edge]
        interval = witness.admissible_edge_interval(
            np.asarray([value[0] for value in tokens]),
            np.asarray([value[1] for value in tokens]),
            np.asarray([value[2] for value in tokens]),
        )
        if interval is None:
            return False
        curve, left, right = interval
        parameter = (left + right) / 2
        point, target_uses = witness.evaluate_curve(curve, parameter, budget=budget)
        _, left_uses = witness.evaluate_curve(curve, left, budget=budget)
        _, right_uses = witness.evaluate_curve(curve, right, budget=budget)
        for position, row in enumerate(rows):
            anchors = tuple(
                (state.charts[row, np.flatnonzero(state.cells[row] == vertex)[0]], uses)
                for vertex, uses in zip(edge, (left_uses, right_uses), strict=True)
            )
            chart = _curve_occurrence_chart(
                int(state.patches[row]), anchors, target_uses, domain.tolerance
            )
            if chart is None:
                return False
            chart_midpoints[position] = chart
        physical = np.broadcast_to(point, (rows.size, 3))
        represented = domain.evaluate(state.patches[rows], chart_midpoints)
        if (
            np.max(np.linalg.norm(represented - physical, axis=1), initial=0.0)
            > domain.tolerance * domain.scale
        ):
            return False
        token = (1, curve, np.asarray((parameter, 0.0), dtype=np.float64))
    else:
        physical = domain.evaluate(state.patches[rows], chart_midpoints)
    if (
        np.max(np.linalg.norm(physical - physical[:1], axis=1), initial=0.0)
        > domain.tolerance * domain.scale
    ):
        return False
    if state.material_charts is not None:
        _admit_material_carrier(
            state,
            state.material_charts[rows],
            int(state.cells.shape[0] + rows.size),
            midpoint=True,
            budget=budget,
        )
    vertex = state.points.shape[0]
    points = np.concatenate((state.points, physical[:1]))
    cells, charts, owners = [], [], []
    material_children = []
    for position, row in enumerate(rows.tolist()):
        for removed in edge.tolist():
            cell = state.cells[row].copy()
            chart = state.charts[row].copy()
            slot = np.flatnonzero(cell == removed)[0]
            cell[slot] = vertex
            chart[slot] = chart_midpoints[position]
            cells.append(cell)
            charts.append(chart)
            owners.append(row)
            if state.material_charts is not None:
                material = state.material_charts[row].copy()
                endpoints = state.material_charts[row, np.isin(state.cells[row], edge)]
                with _dependency_phydrax__geometry_predicates.exact_charge(
                    8, budget=budget
                ):
                    material[slot] = (endpoints[0] + endpoints[1]) / 2
                material_children.append(material)
    children = np.asarray(cells, dtype=np.int32)
    child_charts = np.asarray(charts, dtype=np.float64)
    owner = np.asarray(owners, dtype=np.int64)
    material_charts = (
        np.asarray(material_children, dtype=object)
        if state.material_charts is not None
        else None
    )
    if material_charts is not None and not _material_legal(
        domain, state.patches[owner], material_charts, budget=budget
    ):
        return False
    if not _legal(
        domain,
        children,
        state.patches[owner],
        child_charts,
        points,
        np.inf,
        budget=budget,
    ):
        return False
    vertex_id = _dependency_phydrax_meshing__tetra_metric.fresh_identifiers(
        state.next_vertex_id, 1
    )[0]
    child_ids = _dependency_phydrax_meshing__tetra_metric.fresh_identifiers(
        state.next_cell_id, children.shape[0]
    )
    state.points = points
    state.metric = np.concatenate((state.metric, new_metric[None]))
    state.vertex_ids = np.append(state.vertex_ids, vertex_id)
    state.next_vertex_id += 1
    state.vertex_sources.append(
        state.vertex_sources[edge[0]] | state.vertex_sources[edge[1]]
    )
    if strata is not None:
        strata.append(
            token
            if token is not None
            else (2, int(state.patches[rows[0]]), chart_midpoints[0].copy())
        )
    _replace(
        state,
        rows,
        children,
        state.patches[owner],
        child_charts,
        state.classes[owner],
        [state.parents[row].copy() for row in owners],
        _dependency_phydrax_meshing__lineage.EntityLineageKind.REFINED_FROM,
        material_charts=material_charts,
        cell_ids=child_ids,
    )
    return True


def _link(cells: np.ndarray, vertex: tuple[int, ...], /) -> set[tuple[int, ...]]:
    result: set[tuple[int, ...]] = {()}
    for cell in cells.tolist():
        if set(vertex).issubset(cell):
            remaining = sorted(set(cell) - set(vertex))
            for degree in range(1, len(remaining) + 1):
                result.update(combinations(remaining, degree))
    return result


@checked
def collapse(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: SurfaceState,
    removed: int,
    kept: int,
    fixed: np.ndarray,
    features: set[tuple[int, int]],
    minimum_quality: float,
    maximum_fidelity: float,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    if removed < fixed.size and fixed[removed]:
        return False
    if any(removed in edge for edge in features):
        return False
    rows = np.flatnonzero(np.any(state.cells == removed, axis=1))
    if (
        rows.size == 0
        or np.unique(state.classes[rows]).size != 1
        or np.unique(state.patches[rows]).size != 1
    ):
        return False
    if _link(state.cells, (removed,)) & _link(state.cells, (kept,)) != _link(
        state.cells, (removed, kept)
    ):
        return False
    keep_rows = np.flatnonzero(
        np.any(state.cells == kept, axis=1) & (state.patches == state.patches[rows[0]])
    )
    if keep_rows.size == 0:
        return False

    _admit_cavity(int(np.union1d(rows, keep_rows).size), budget=budget)
    owner = keep_rows[0]
    chart_kept = state.charts[owner, np.flatnonzero(state.cells[owner] == kept)[0]]
    proposed = state.cells[rows].copy()
    charts = state.charts[rows].copy()
    mask = proposed == removed
    proposed[mask] = kept
    charts[mask] = chart_kept
    live = np.all(np.diff(np.sort(proposed, axis=1), axis=1) != 0, axis=1)
    proposed, charts = proposed[live], charts[live]
    owners = rows[live]
    material_charts = None
    if state.material_charts is not None:
        material_charts = state.material_charts[rows].copy()
        material_charts[mask] = state.material_charts[
            owner, np.flatnonzero(state.cells[owner] == kept)[0]
        ]
        material_charts = material_charts[live]
        _admit_material_carrier(
            state,
            material_charts,
            int(state.cells.shape[0] - rows.size + proposed.shape[0]),
            budget=budget,
        )
        if not _material_legal(
            domain, state.patches[owners], material_charts, budget=budget
        ):
            return False
    if proposed.size == 0 or not _legal(
        domain,
        proposed,
        state.patches[owners],
        charts,
        state.points,
        maximum_fidelity,
        budget=budget,
    ):
        return False
    if (
        np.min(
            _mapped_quality(
                domain,
                state.metric,
                proposed,
                state.patches[owners],
                charts,
                budget=budget,
            )
        )
        < minimum_quality
    ):
        return False
    # All new metric edges obey the explicit collapse ceiling; next pass refines.
    trial = SurfaceState(
        state.points,
        state.metric,
        proposed,
        state.patches[owners],
        charts,
        state.classes[owners],
        state.cell_ids[owners],
        state.vertex_ids,
        [state.parents[row] for row in owners],
        state.vertex_sources,
        state.next_cell_id,
        state.next_vertex_id,
    )
    if np.max(_measure(domain, trial, budget=budget)[0], initial=0.0) > 2.0:
        return False
    parents = set().union(*(state.parents[row] for row in rows))
    _replace(
        state,
        rows,
        proposed,
        state.patches[owners],
        charts,
        state.classes[owners],
        [parents.copy() for _ in owners],
        _dependency_phydrax_meshing__lineage.EntityLineageKind.COLLAPSED_INTO,
        material_charts=material_charts,
    )
    if state.vertex_successor is None:
        state.vertex_successor = {}
    state.vertex_successor[removed] = kept
    return True


@checked
def flip(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: SurfaceState,
    edge: np.ndarray,
    features: set[tuple[int, int]],
    maximum_fidelity: float,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    if tuple(edge.tolist()) in features:
        return False
    rows = np.flatnonzero(np.sum(np.isin(state.cells, edge), axis=1) == 2)
    if (
        rows.size != 2
        or state.classes[rows[0]] != state.classes[rows[1]]
        or state.patches[rows[0]] != state.patches[rows[1]]
    ):
        return False

    _admit_cavity(int(rows.size), budget=budget)
    old = state.cells[rows]
    apex = old[~np.isin(old, edge)]
    if np.any(np.sum(np.isin(state.cells, apex), axis=1) == 2):
        return False
    chart_of = {
        int(vertex): chart
        for cell, chart in zip(old, state.charts[rows], strict=True)
        for vertex, chart in zip(cell, chart, strict=True)
    }
    proposed = np.asarray(
        ((apex[0], apex[1], edge[0]), (apex[1], apex[0], edge[1])), dtype=np.int32
    )
    charts = np.asarray(
        [[chart_of[vertex] for vertex in cell] for cell in proposed], dtype=np.float64
    )
    patch = state.patches[rows]
    predicate = _dependency_phydrax__geometry_predicates.orient2d(
        charts[:, 0],
        charts[:, 1],
        charts[:, 2],
        mode=_dependency_phydrax__geometry_predicates.PredicateMode.EXACT,
        budget=budget,
    )
    signs = np.asarray(predicate.signs)
    if (
        not np.all(np.asarray(predicate.certain))
        or signs[0] == _dependency_phydrax__geometry_predicates.PredicateSign.ZERO
        or signs[0] != signs[1]
    ):
        # A concave chart cavity cannot be repaired by reversing its two
        # triangles independently: that changes its oriented outer chain.
        return False
    reverse = signs != domain.patches[patch[0]].orientation
    proposed[reverse] = proposed[reverse, ::-1]
    charts[reverse] = charts[reverse, ::-1]
    material_charts = None
    if state.material_charts is not None:
        material_of = {
            int(vertex): chart
            for cell, chart in zip(old, state.material_charts[rows], strict=True)
            for vertex, chart in zip(cell, chart, strict=True)
        }
        material_charts = np.asarray(
            [[material_of[vertex] for vertex in cell] for cell in proposed], dtype=object
        )
        _admit_material_carrier(
            state, material_charts, int(state.cells.shape[0]), budget=budget
        )
        if not _material_legal(domain, patch, material_charts, budget=budget):
            return False
    if not _legal(
        domain, proposed, patch, charts, state.points, maximum_fidelity, budget=budget
    ):
        return False
    before = np.min(
        _mapped_quality(
            domain, state.metric, old, patch, state.charts[rows], budget=budget
        )
    )
    after = np.min(
        _mapped_quality(domain, state.metric, proposed, patch, charts, budget=budget)
    )
    if after <= before + 1.0e-6:
        return False
    parents = state.parents[rows[0]] | state.parents[rows[1]]
    _replace(
        state,
        rows,
        proposed,
        patch,
        charts,
        state.classes[rows],
        [parents.copy(), parents.copy()],
        _dependency_phydrax_meshing__lineage.EntityLineageKind.SWAPPED_FROM,
        material_charts=material_charts,
    )
    return True


def _source_curve_slide_charts(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: SurfaceState,
    vertices: np.ndarray,
    patches: np.ndarray,
    seeds: np.ndarray,
    features: set[tuple[int, int]],
    feature_codes: dict[tuple[int, int], int],
    blocked: set[tuple[int, int]],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int, np.ndarray]]]:
    witness, strata = state.curve_witness, state.vertex_strata
    if witness is None or strata is None:
        raise ValueError(
            "surface_metric.source_curve_slide_charts must retain source-stratum parameters and incidence; got an CAD curve slide lacks its source-bound strata"
        )
    proposals = seeds.copy()
    valid = np.zeros(vertices.size, dtype=np.bool_)
    tokens = {}
    for index, vertex in enumerate(vertices.tolist()):
        dimension, curve, current = strata[vertex]
        incident = [edge for edge in features if vertex in edge]
        if (
            dimension != 1
            or len(incident) != 2
            or any(edge in blocked for edge in incident)
        ):
            continue
        if len({feature_codes[edge] for edge in incident}) != 1:
            continue
        neighbors = [edge[0] if edge[1] == vertex else edge[1] for edge in incident]
        neighbor_tokens = [strata[neighbor] for neighbor in neighbors]
        interval = witness.admissible_edge_interval(
            np.asarray([value[0] for value in neighbor_tokens]),
            np.asarray([value[1] for value in neighbor_tokens]),
            np.asarray([value[2] for value in neighbor_tokens]),
        )
        if interval is None:
            continue
        owner, left, right = interval
        if owner != curve or not min(left, right) <= current[0] <= max(left, right):
            continue
        parameter = (left + right) / 2
        point, target_uses = witness.evaluate_curve(curve, parameter, budget=budget)
        _, original_uses = witness.evaluate_curve(curve, float(current[0]), budget=budget)
        rows = np.flatnonzero(np.any(state.cells == vertex, axis=1))
        charts = [
            _curve_occurrence_chart(
                int(state.patches[row]),
                (
                    (
                        state.charts[row, np.flatnonzero(state.cells[row] == vertex)[0]],
                        original_uses,
                    ),
                ),
                target_uses,
                domain.tolerance,
            )
            for row in rows
        ]
        resolved_charts = [value for value in charts if value is not None]
        if len(resolved_charts) != len(charts):
            continue
        chart = resolved_charts[0]
        if any(not np.array_equal(chart, other) for other in resolved_charts[1:]):
            continue
        represented = domain.evaluate(patches[index : index + 1], chart[None])[0]
        if np.linalg.norm(represented - point) > domain.tolerance * domain.scale:
            continue
        proposals[index], valid[index] = chart, True
        tokens[vertex] = (1, curve, np.asarray((parameter, 0.0), dtype=np.float64))
    return proposals, valid, tokens


def _feature_slide_charts(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: SurfaceState,
    source: SurfaceSource,
    vertices: np.ndarray,
    patches: np.ndarray,
    seeds: np.ndarray,
    displacement: np.ndarray,
    features: set[tuple[int, int]],
    feature_codes: dict[tuple[int, int], int],
    blocked: set[tuple[int, int]],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int, np.ndarray]]]:
    """Use CAD strata when supplied; otherwise retain mesh-chart features."""
    if state.curve_witness is not None:
        return _source_curve_slide_charts(
            domain,
            state,
            vertices,
            patches,
            seeds,
            features,
            feature_codes,
            blocked,
            budget=budget,
        )
    directions = np.zeros((vertices.size, 2), dtype=np.float64)
    endpoints = np.zeros((vertices.size, 2, 2), dtype=np.float64)
    permitted = np.zeros((vertices.size,), dtype=np.bool_)
    for index, vertex in enumerate(vertices.tolist()):
        support = sorted(state.vertex_sources[vertex])
        neighbors = {
            other
            for edge in features
            if vertex in edge
            for other in edge
            if other != vertex
        }
        incident_edges = [edge for edge in features if vertex in edge]
        if len(neighbors) != 2 or any(vertex in edge for edge in blocked):
            continue
        codes = {feature_codes[edge] for edge in incident_edges}
        if len(codes) != 1:
            continue
        if len(support) == 2:
            global_cells = source.vertex_ids[source.cells]
            rows = np.flatnonzero(
                (source.patches == patches[index])
                & (np.sum(np.isin(global_cells, support), axis=1) == 2)
            )
            if rows.size == 0:
                continue
            row = rows[np.argmin(source.cell_ids[rows])]
            selected = np.flatnonzero(np.isin(global_cells[row], support))
            uv = source.charts[row, selected]
        elif len(support) == 1 and next(iter(codes)) > 0:
            uv = np.zeros((2, 2), dtype=np.float64)
            for slot, neighbor in enumerate(sorted(neighbors)):
                rows = np.flatnonzero(
                    np.any(state.cells == neighbor, axis=1)
                    & (state.patches == patches[index])
                )
                row = rows[np.argmin(state.cell_ids[rows])]
                uv[slot] = state.charts[
                    row, np.flatnonzero(state.cells[row] == neighbor)[0]
                ]
        else:
            continue
        endpoints[index] = uv
        directions[index] = uv[1] - uv[0]
        incident = np.flatnonzero(np.any(state.cells == vertex, axis=1))
        uses = state.charts[incident][state.cells[incident] == vertex]
        if (
            np.max(np.linalg.norm(uses - seeds[index], axis=1), initial=0.0)
            > domain.tolerance
        ):
            continue
        signs = _dependency_phydrax__geometry_predicates.orient2d(
            uv[0],
            uv[1],
            seeds[index],
            mode=_dependency_phydrax__geometry_predicates.PredicateMode.EXACT,
            budget=budget,
        )
        permitted[index] = (
            bool(np.asarray(signs.certain))
            and np.asarray(signs.signs)
            == _dependency_phydrax__geometry_predicates.PredicateSign.ZERO
        )
    tangent = np.zeros((vertices.size, 3), dtype=np.float64)
    _consume_native_work(geometry_queries=vertices.size, budget=budget)
    for patch in np.unique(patches).tolist():
        selected = patches == patch
        differential = np.asarray(
            _dependency_phydrax_geometry_brep__patches.surface_differential(
                domain.patches[patch].surface,
                jnp.asarray(seeds[selected], dtype=jnp.float64),
            ),
            dtype=np.float64,
        )
        tangent[selected] = (differential @ directions[selected, :, None])[..., 0]
    metric_tangent = (state.metric[vertices] @ tangent[..., None])[..., 0]
    system = np.sum(tangent * metric_tangent, axis=1)[:, None, None]
    rhs = np.sum(displacement * metric_tangent, axis=1)[:, None]
    solve = solve_small_linear(SmallLinearSolvePlan(1), system, rhs)
    steps = np.where(
        np.asarray(solve.successful),
        np.asarray(solve.value, dtype=np.float64)[:, 0],
        0.0,
    )
    proposals = seeds + steps[:, None] * directions
    permitted &= np.asarray(solve.successful)
    permitted &= np.all(np.isfinite(proposals), axis=1)
    signs = _dependency_phydrax__geometry_predicates.orient2d(
        endpoints[:, 0],
        endpoints[:, 1],
        proposals,
        mode=_dependency_phydrax__geometry_predicates.PredicateMode.EXACT,
        budget=budget,
    )
    permitted &= np.asarray(signs.certain) & (
        np.asarray(signs.signs)
        == _dependency_phydrax__geometry_predicates.PredicateSign.ZERO
    )
    return proposals, permitted, {}


def _unit_defect(lengths: np.ndarray, /) -> float:
    return float(
        np.sum(
            np.maximum(_LOWER - lengths, 0.0) ** 2
            + np.maximum(lengths - _UPPER, 0.0) ** 2
        )
    )


def _surface_size_complete(
    lower: np.ndarray,
    upper: np.ndarray,
    edges: np.ndarray,
    features: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    /,
    *,
    generated_from: int | None = None,
) -> bool:
    """Require the lower band only where coarsening is scientifically admissible."""
    protected = np.asarray(
        [
            tuple(map(int, edge)) in features
            or tuple(map(int, edge)) in blocked
            or (generated_from is not None and np.any(edge >= generated_from))
            for edge in edges
        ],
        dtype=np.bool_,
    )
    return bool(np.all(upper <= _UPPER) and np.all((lower >= _LOWER) | protected))


class _SurfaceRelocationRequest(StrictModule):
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain
    state: SurfaceState
    fixed: np.ndarray
    features: set[tuple[int, int]]
    maximum_fidelity: float
    maximum_relocations: int
    maximum_pairs: int
    source: SurfaceSource
    feature_codes: dict[tuple[int, int], int]
    blocked: set[tuple[int, int]]
    minimum_quality: float
    record_phase: NativeMeshingPhaseRecorder | None
    selected_vertices: frozenset[int] | None
    budget: (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    )


class _SurfaceRelocationPreparation(StrictModule):
    selected: np.ndarray
    proposal_points: np.ndarray
    query_charts: np.ndarray
    proposal_valid: np.ndarray
    proposal_metrics: np.ndarray
    sampling: tuple[np.ndarray, np.ndarray, np.ndarray]
    proposal_tokens: dict[int, tuple[int, int, np.ndarray]]


class _SurfaceRelocationExecution(StrictModule):
    applied: int
    attempts: int


def _prepare_surface_relocation(
    request: _SurfaceRelocationRequest,
) -> _SurfaceRelocationPreparation | None:
    domain = request.domain
    state = request.state
    fixed = request.fixed
    features = request.features
    maximum_relocations = request.maximum_relocations
    maximum_pairs = request.maximum_pairs
    source = request.source
    feature_codes = request.feature_codes
    blocked = request.blocked
    record_phase = request.record_phase
    selected_vertices = request.selected_vertices
    budget = request.budget
    edges, _, _ = _edge_table(state)
    lengths, _, _ = _measure(domain, state, budget=budget)
    force = (1.0 - 1.0 / np.maximum(lengths, 1.0e-30))[:, None] * (
        state.points[edges[:, 1]] - state.points[edges[:, 0]]
    )
    total = np.zeros_like(state.points)
    degree = np.zeros((state.points.shape[0],), dtype=np.float64)
    np.add.at(total, edges[:, 0], force)
    np.add.at(total, edges[:, 1], -force)
    np.add.at(degree, edges.reshape((-1,)), 1.0)
    owners, vertices, seeds, sliding = [], [], [], []
    feature_vertices = {vertex for edge in features for vertex in edge}
    for vertex in np.unique(state.cells):
        if selected_vertices is not None and int(vertex) not in selected_vertices:
            continue
        rows = np.flatnonzero(np.any(state.cells == vertex, axis=1))
        if (
            (vertex < fixed.size and fixed[vertex])
            or np.unique(state.patches[rows]).size != 1
            or np.unique(state.classes[rows]).size != 1
        ):
            continue
        owners.append(rows[0])
        vertices.append(vertex)
        seeds.append(
            state.charts[rows[0], np.flatnonzero(state.cells[rows[0]] == vertex)[0]]
        )
        sliding.append(vertex in feature_vertices)
    if not vertices or maximum_relocations == 0:
        return None
    selected = np.asarray(vertices[:maximum_relocations], dtype=np.int32)
    _consume_native_work(int(selected.size), geometry_queries=0, budget=budget)
    owner = np.asarray(owners[:maximum_relocations], dtype=np.int64)
    displacement = total[selected] / np.maximum(degree[selected, None], 1.0)
    seed_charts = np.asarray(seeds[:maximum_relocations], dtype=np.float64)
    slide = np.asarray(sliding[:maximum_relocations], dtype=np.bool_)
    proposal_points = state.points[selected].copy()
    query_charts = seed_charts.copy()
    proposal_valid = np.zeros((selected.size,), dtype=np.bool_)
    free = np.flatnonzero(~slide)
    if free.size:
        with measure_phase(record_phase, "geometry_projection"):
            projection = domain.project(
                state.points[selected[free]] + displacement[free],
                state.patches[owner[free]],
                seed_charts[free],
            )
        proposal_valid[free] = projection.converged
        query_charts[free] = np.where(
            projection.converged[:, None], projection.parameters, seed_charts[free]
        )
        proposal_points[free] = np.where(
            projection.converged[:, None], projection.points, proposal_points[free]
        )
    constrained = np.flatnonzero(slide)
    proposal_tokens: dict[int, tuple[int, int, np.ndarray]] = {}
    if constrained.size:
        proposals, valid, proposal_tokens = _feature_slide_charts(
            domain,
            state,
            source,
            selected[constrained],
            state.patches[owner[constrained]],
            seed_charts[constrained],
            displacement[constrained],
            features,
            feature_codes,
            blocked,
            budget=budget,
        )
        query_charts[constrained] = np.where(
            valid[:, None], proposals, seed_charts[constrained]
        )
        proposal_points[constrained] = domain.evaluate(
            state.patches[owner[constrained]], query_charts[constrained]
        )
        proposal_valid[constrained] = valid
    query_material = query_charts
    if state.material_charts is not None:
        query_material = np.asarray(
            [
                state.material_charts[row, np.flatnonzero(state.cells[row] == vertex)[0]]
                for row, vertex in zip(owner, selected, strict=True)
            ],
            dtype=object,
        )
    sampling = chart_stencil(
        source.cells,
        source.charts,
        source.patches,
        source.cell_ids,
        source.vertex_ids,
        state.patches[owner],
        query_material,
        maximum_pairs,
        budget=budget,
    )
    source_rows = entity_keys_owner.key_rows(
        source.vertex_ids[:, None],
        sampling[0].reshape((-1, 1)),
    ).reshape(sampling[0].shape)
    proposal_metrics = np.asarray(
        _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
            source.metric[source_rows], sampling[1]
        ),
        dtype=np.float64,
    )
    return _SurfaceRelocationPreparation(
        selected,
        proposal_points,
        query_charts,
        proposal_valid,
        proposal_metrics,
        sampling,
        proposal_tokens,
    )


def _execute_surface_relocation(
    request: _SurfaceRelocationRequest, prepared: _SurfaceRelocationPreparation
) -> _SurfaceRelocationExecution:
    domain = request.domain
    state = request.state
    maximum_fidelity = request.maximum_fidelity
    minimum_quality = request.minimum_quality
    budget = request.budget
    selected = prepared.selected
    proposal_points = prepared.proposal_points
    query_charts = prepared.query_charts
    proposal_valid = prepared.proposal_valid
    proposal_metrics = prepared.proposal_metrics
    sampling = prepared.sampling
    proposal_tokens = prepared.proposal_tokens
    applied = 0
    for index, vertex in enumerate(selected.tolist()):
        if not proposal_valid[index]:
            continue
        rows = np.flatnonzero(np.any(state.cells == vertex, axis=1))

        _admit_cavity(int(rows.size), budget=budget)
        points = state.points.copy()
        points[vertex] = proposal_points[index]
        charts = state.charts[rows].copy()
        charts[state.cells[rows] == vertex] = query_charts[index]
        if not _legal(
            domain,
            state.cells[rows],
            state.patches[rows],
            charts,
            points,
            maximum_fidelity,
            budget=budget,
        ):
            continue
        before = np.min(
            _mapped_quality(
                domain,
                state.metric,
                state.cells[rows],
                state.patches[rows],
                state.charts[rows],
                budget=budget,
            )
        )
        metrics = state.metric.copy()
        metrics[vertex] = proposal_metrics[index]
        after = np.min(
            _mapped_quality(
                domain,
                metrics,
                state.cells[rows],
                state.patches[rows],
                charts,
                budget=budget,
            )
        )
        local = SurfaceState(
            state.points,
            state.metric,
            state.cells[rows],
            state.patches[rows],
            state.charts[rows],
            state.classes[rows],
            state.cell_ids[rows],
            state.vertex_ids,
            [state.parents[row] for row in rows],
            state.vertex_sources,
            state.next_cell_id,
            state.next_vertex_id,
        )
        trial = SurfaceState(
            points,
            metrics,
            state.cells[rows],
            state.patches[rows],
            charts,
            state.classes[rows],
            state.cell_ids[rows],
            state.vertex_ids,
            local.parents,
            state.vertex_sources,
            state.next_cell_id,
            state.next_vertex_id,
        )
        unit_improved = (
            _unit_defect(_measure(domain, trial, budget=budget)[0])
            < _unit_defect(_measure(domain, local, budget=budget)[0]) - 1.0e-12
        )
        if after <= before + 1.0e-6 and not (unit_improved and after >= minimum_quality):
            continue
        state.points = points
        state.charts[rows] = charts
        state.metric = metrics
        state.vertex_sources[vertex] = set(
            sampling[0][index, sampling[1][index] > 0.0].tolist()
        )
        if state.vertex_strata is not None:
            state.vertex_strata[vertex] = proposal_tokens.get(
                vertex, (2, int(state.patches[rows[0]]), query_charts[index].copy())
            )
        if state.relocated_vertices is None:
            state.relocated_vertices = set()
        state.relocated_vertices.add(vertex)
        if state.cell_kinds is None:
            state.cell_kinds = {}
        for identifier in state.cell_ids[rows].tolist():
            state.cell_kinds[identifier] = int(
                _dependency_phydrax_meshing__tetra_metric.lineage_operation_kind(
                    _dependency_phydrax_meshing__lineage.EntityLineageKind.RELOCATED,
                    (
                        state.cell_kinds.get(
                            identifier,
                            int(
                                _dependency_phydrax_meshing__lineage.EntityLineageKind.PRESERVED
                            ),
                        ),
                    ),
                )
            )
        applied += 1
    return _SurfaceRelocationExecution(applied, selected.size)


@checked
def relocate(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: SurfaceState,
    fixed: np.ndarray,
    features: set[tuple[int, int]],
    maximum_fidelity: float,
    maximum_relocations: int,
    maximum_pairs: int,
    source: SurfaceSource,
    feature_codes: dict[tuple[int, int], int],
    blocked: set[tuple[int, int]],
    minimum_quality: float,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    selected_vertices: frozenset[int] | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[int, int]:
    request = _SurfaceRelocationRequest(
        domain,
        state,
        fixed,
        features,
        maximum_fidelity,
        maximum_relocations,
        maximum_pairs,
        source,
        feature_codes,
        blocked,
        minimum_quality,
        record_phase,
        selected_vertices,
        budget,
    )
    prepared = _prepare_surface_relocation(request)
    if prepared is None:
        return 0, 0
    execution = _execute_surface_relocation(request, prepared)
    return execution.applied, execution.attempts


class _MaterialChartRequest(StrictModule):
    source_cells: np.ndarray
    source_charts: np.ndarray
    source_patches: np.ndarray
    source_cell_ids: np.ndarray
    source_vertex_ids: np.ndarray
    query_patches: np.ndarray
    query_charts: np.ndarray
    maximum_pairs: int
    cache: (
        dict[tuple[int, tuple[tuple[int, int], ...]], tuple[np.ndarray, np.ndarray, int]]
        | None
    )
    budget: (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    )


class _MaterialChartRouting(StrictModule):
    sources: np.ndarray
    weights: np.ndarray
    selected_ids: np.ndarray
    work: int
    query_keys: list[tuple[int, tuple[tuple[int, int], ...]]]


class _MaterialChartPatch(StrictModule):
    candidates: np.ndarray
    queries: np.ndarray
    tree: PackedBVH
    query_tree: PackedBVH


def _prepare_material_chart_cache(
    request: _MaterialChartRequest,
) -> _MaterialChartRouting:
    query_patches = request.query_patches
    query_charts = request.query_charts
    cache = request.cache
    sources = np.zeros((query_patches.size, 3), dtype=np.int64)
    weights = np.zeros((query_patches.size, 3), dtype=np.float64)
    selected_ids = np.full(query_patches.size, np.iinfo(np.int64).max, dtype=np.int64)
    work = 0

    query_keys: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for row, patch in enumerate(query_patches):
        point = _exact_uv(query_charts[row])
        key = (
            int(patch),
            tuple((value.numerator, value.denominator) for value in point),
        )
        query_keys.append(key)
        cached = None if cache is None else cache.get(key)
        if cached is not None:
            sources[row], weights[row], selected_ids[row] = cached
    return _MaterialChartRouting(sources, weights, selected_ids, work, query_keys)


def _prepare_material_chart_patch(
    request: _MaterialChartRequest,
    routing: _MaterialChartRouting,
    patch: np.integer | int,
) -> _MaterialChartPatch:
    source_charts = request.source_charts
    source_patches = request.source_patches
    query_patches = request.query_patches
    query_charts = request.query_charts
    selected_ids = routing.selected_ids
    candidates = np.flatnonzero(source_patches == patch)
    queries = np.flatnonzero(
        (query_patches == patch) & (selected_ids == np.iinfo(np.int64).max)
    )
    corners = source_charts[candidates]
    source_lower = np.asarray(
        [
            [
                fraction_endpoint(min(triangle[:, axis]), upper=False, saturate=True)
                for axis in range(2)
            ]
            for triangle in corners
        ]
    )
    source_upper = np.asarray(
        [
            [
                fraction_endpoint(max(triangle[:, axis]), upper=True, saturate=True)
                for axis in range(2)
            ]
            for triangle in corners
        ]
    )
    tree = prepare_bvh(source_lower, source_upper, dtype=jnp.float64)
    lower = np.asarray(
        [
            [
                fraction_endpoint(value, upper=False, saturate=True)
                for value in query_charts[row]
            ]
            for row in queries
        ]
    )
    upper = np.asarray(
        [
            [
                fraction_endpoint(value, upper=True, saturate=True)
                for value in query_charts[row]
            ]
            for row in queries
        ]
    )
    query_tree = prepare_bvh(lower, upper, dtype=jnp.float64)
    return _MaterialChartPatch(candidates, queries, tree, query_tree)


def _execute_material_chart_patch(
    request: _MaterialChartRequest,
    routing: _MaterialChartRouting,
    patch_prepared: _MaterialChartPatch,
) -> _MaterialChartRouting:
    source_cells = request.source_cells
    source_charts = request.source_charts
    source_cell_ids = request.source_cell_ids
    source_vertex_ids = request.source_vertex_ids
    query_charts = request.query_charts
    maximum_pairs = request.maximum_pairs
    cache = request.cache
    budget = request.budget
    sources = routing.sources
    weights = routing.weights
    selected_ids = routing.selected_ids
    work = routing.work
    query_keys = routing.query_keys
    candidates = patch_prepared.candidates
    queries = patch_prepared.queries
    tree = patch_prepared.tree
    query_tree = patch_prepared.query_tree
    for cells, rows in bvh_overlap_pair_blocks(tree, query_tree, include_touching=True):
        work += cells.size
        if work > maximum_pairs:
            raise _dependency_phydrax_meshing__tetra_metric.source_location_budget(
                maximum_pairs, work
            )
        with budget.temporary_scope():
            inverses: dict[int, tuple[tuple[Fraction, ...], ...]] = {}
            for local_cell, local_query in zip(cells, rows, strict=True):
                cell = int(candidates[local_cell])
                query = int(queries[local_query])
                identifier = int(source_cell_ids[cell])
                if identifier >= selected_ids[query]:
                    continue
                a, b, c = (_exact_uv(uv) for uv in source_charts[cell])
                inverse = inverses.get(cell)
                if inverse is None:
                    prepared = _dependency_phydrax_linalg__small_batched.prepare_exact_small_linear_actions(
                        ((b[0] - a[0], c[0] - a[0]), (b[1] - a[1], c[1] - a[1])),
                        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
                        budget=budget,
                    )
                    if prepared.actions is None:
                        raise ValueError(
                            "surface_metric.source_charts must contain nonsingular material cells; got singular chart frame"
                        )
                    inverse = prepared.actions
                    inverses[cell] = inverse
                point = _exact_uv(query_charts[query])
                delta = point[0] - a[0], point[1] - a[1]
                budget.reserve(12)
                u, v = (
                    sum(
                        (
                            coefficient * value
                            for coefficient, value in zip(row, delta, strict=True)
                        ),
                        Fraction(0),
                    )
                    for row in inverse
                )
                values = (1 - u - v, u, v)
                if min(values) < 0:
                    continue
                sources[query] = source_vertex_ids[source_cells[cell]]
                weights[query] = tuple(float(value) for value in values)
                selected_ids[query] = identifier
                if cache is not None:
                    cached_sources = sources[query].copy()
                    cached_weights = weights[query].copy()
                    budget.retain_basis(
                        (
                            query_keys[query],
                            cached_sources,
                            cached_weights,
                            identifier,
                        )
                    )
                    cache[query_keys[query]] = (
                        cached_sources,
                        cached_weights,
                        identifier,
                    )
    return _MaterialChartRouting(sources, weights, selected_ids, work, query_keys)


def _material_chart_stencil(
    source_cells: np.ndarray,
    source_charts: np.ndarray,
    source_patches: np.ndarray,
    source_cell_ids: np.ndarray,
    source_vertex_ids: np.ndarray,
    query_patches: np.ndarray,
    query_charts: np.ndarray,
    maximum_pairs: int,
    /,
    *,
    cache: dict[
        tuple[int, tuple[tuple[int, int], ...]], tuple[np.ndarray, np.ndarray, int]
    ]
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route material coordinates through rational barycentric containment."""
    request = _MaterialChartRequest(
        source_cells,
        source_charts,
        source_patches,
        source_cell_ids,
        source_vertex_ids,
        query_patches,
        query_charts,
        maximum_pairs,
        cache,
        budget,
    )
    routing = _prepare_material_chart_cache(request)
    for patch in np.unique(query_patches):
        prepared = _prepare_material_chart_patch(request, routing, int(patch))
        routing = _execute_material_chart_patch(request, routing, prepared)
    sources, weights, selected_ids = (
        routing.sources,
        routing.weights,
        routing.selected_ids,
    )
    if np.any(selected_ids == np.iinfo(np.int64).max):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
            "surface_metric.material_functional must lie in the source chart cover; got uncovered target coordinates",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.LINEAGE_CONSTRUCTION.value,
        )
    return sources, weights, np.ones(weights.shape, dtype=np.bool_)


@checked
def chart_stencil(
    source_cells: np.ndarray,
    source_charts: np.ndarray,
    source_patches: np.ndarray,
    source_cell_ids: np.ndarray,
    source_vertex_ids: np.ndarray,
    query_patches: np.ndarray,
    query_charts: np.ndarray,
    maximum_pairs: int,
    /,
    *,
    material_cache: dict[
        tuple[int, tuple[tuple[int, int], ...]], tuple[np.ndarray, np.ndarray, int]
    ]
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if query_charts.dtype == object:
        return _material_chart_stencil(
            source_cells,
            source_charts,
            source_patches,
            source_cell_ids,
            source_vertex_ids,
            query_patches,
            query_charts,
            maximum_pairs,
            cache=material_cache,
            budget=budget,
        )
    sources = np.zeros((query_patches.size, 3), dtype=np.int64)
    weights = np.zeros((query_patches.size, 3), dtype=np.float64)
    selected_ids = np.full((query_patches.size,), np.iinfo(np.int64).max, dtype=np.int64)
    work = 0
    for patch in np.unique(query_patches).tolist():
        candidates = np.flatnonzero(source_patches == patch)
        query_rows = np.flatnonzero(query_patches == patch)
        corners = source_charts[candidates]
        tree = prepare_bvh(
            np.min(corners, axis=1), np.max(corners, axis=1), dtype=jnp.float64
        )
        query_tree = prepare_bvh(
            query_charts[query_rows], query_charts[query_rows], dtype=jnp.float64
        )
        for cell, query in bvh_overlap_pair_blocks(
            tree, query_tree, include_touching=True
        ):
            work += cell.size
            if work > maximum_pairs:
                raise _dependency_phydrax_meshing__tetra_metric.source_location_budget(
                    maximum_pairs, work
                )
            triangle = corners[cell]
            target = query_charts[query_rows[query]]
            matrix = np.swapaxes(triangle[:, 1:] - triangle[:, :1], 1, 2)
            result = solve_small_linear(
                SmallLinearSolvePlan(2), matrix, target - triangle[:, 0]
            )
            reference = np.asarray(result.value, dtype=np.float64)
            valid = np.array(result.successful, dtype=np.bool_, copy=True)
            coefficients = np.concatenate(
                (1.0 - np.sum(reference, axis=1, keepdims=True), reference), axis=1
            )
            original = _dependency_phydrax__geometry_predicates.orient2d(
                triangle[:, 0],
                triangle[:, 1],
                triangle[:, 2],
                mode=_dependency_phydrax__geometry_predicates.PredicateMode.EXACT,
                budget=budget,
            )
            for slot in range(3):
                vertices = [triangle[:, index] for index in range(3)]
                vertices[slot] = target
                predicate = _dependency_phydrax__geometry_predicates.orient2d(
                    *vertices,
                    mode=_dependency_phydrax__geometry_predicates.PredicateMode.EXACT,
                    budget=budget,
                )
                signs = np.asarray(predicate.signs)
                valid &= np.asarray(predicate.certain) & (
                    (signs == np.asarray(original.signs))
                    | (
                        signs
                        == _dependency_phydrax__geometry_predicates.PredicateSign.ZERO
                    )
                )
                coefficients[
                    signs == _dependency_phydrax__geometry_predicates.PredicateSign.ZERO,
                    slot,
                ] = 0.0
            identifiers = source_cell_ids[candidates[cell]]
            accepted = np.flatnonzero(
                valid & (identifiers < selected_ids[query_rows[query]])
            )
            order = accepted[np.lexsort((identifiers[accepted], query[accepted]))]
            first = (
                np.concatenate(
                    (np.asarray((True,), dtype=np.bool_), np.diff(query[order]) != 0)
                )
                if order.size
                else np.empty((0,), dtype=np.bool_)
            )
            chosen = order[first]
            owners = query_rows[query[chosen]]
            sources[owners] = source_vertex_ids[source_cells[candidates[cell[chosen]]]]
            weights[owners] = coefficients[chosen]
            selected_ids[owners] = identifiers[chosen]
    if np.any(selected_ids == np.iinfo(np.int64).max):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
            "surface_metric.target_chart must retain source chart coverage; got uncovered target chart",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.LINEAGE_CONSTRUCTION.value,
        )
    weights = np.maximum(weights, 0.0)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return sources, weights, np.ones(weights.shape, dtype=np.bool_)


@checked
def chart_exact_weights(
    source_cells: np.ndarray,
    source_charts: np.ndarray,
    source_vertex_ids: np.ndarray,
    stencil_sources: np.ndarray,
    query_charts: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_dependency_phydrax_meshing__periodic.ConstructionPointKey, ...]:
    """Return a source-chart functional for native construction and field transport."""
    selected = entity_keys_owner.key_rows(
        np.sort(source_vertex_ids[source_cells], axis=1), np.sort(stencil_sources, axis=1)
    )
    if np.any(selected < 0):
        raise ValueError(
            "surface_metric.chart_exact_weights must retain source geometry invariants; got an UV construction lacks its source triangle"
        )
    keys: dict[int, _dependency_phydrax_meshing__periodic.ConstructionPointKey] = {}

    for row in np.unique(selected):
        queries = np.flatnonzero(selected == row)
        budget.reserve(6 + 2 * queries.size, 2048 * queries.size)
        a, b, c = tuple(_exact_uv(uv) for uv in source_charts[row])
        matrix = ((b[0] - a[0], c[0] - a[0]), (b[1] - a[1], c[1] - a[1]))
        right = tuple(
            tuple(
                (
                    query_charts[index, axis]
                    if isinstance(query_charts[index, axis], Fraction)
                    else Fraction(float(query_charts[index, axis]))
                )
                - a[axis]
                for index in queries
            )
            for axis in range(2)
        )
        solved = (
            _dependency_phydrax_linalg__small_batched.prepare_exact_small_linear_actions(
                matrix, right, budget=budget
            )
        )
        if not solved.successful or solved.actions is None:
            raise ValueError(
                f"surface_metric.source_triangle.chart_frame must have rank 2; got status={solved.status}, rank={solved.rank}"
            )
        for index, u, v in zip(queries, *solved.actions, strict=True):
            weights = (1 - u - v, u, v)
            if min(weights) < 0:
                raise ValueError(
                    "surface_metric.chart_exact_weights must retain source geometry invariants; got uV construction leaves its source triangle"
                )
            keys[int(index)] = tuple(
                sorted(
                    (int(parent), weight)
                    for parent, weight in zip(
                        source_vertex_ids[source_cells[row]], weights, strict=True
                    )
                    if weight
                )
            )
    return tuple(keys[index] for index in range(query_charts.shape[0]))


def _stencil(
    source_cells: np.ndarray,
    source_charts: np.ndarray,
    source_patches: np.ndarray,
    source_cell_ids: np.ndarray,
    source_vertex_ids: np.ndarray,
    state: SurfaceState,
    live: np.ndarray,
    maximum_pairs: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patches = np.zeros((live.size,), dtype=np.int32)
    charts = np.empty(
        (live.size, 2), dtype=object if state.material_charts is not None else np.float64
    )
    for position, vertex in enumerate(live.tolist()):
        rows = np.flatnonzero(np.any(state.cells == vertex, axis=1))
        owner = rows[np.argmin(state.cell_ids[rows])]
        patches[position] = state.patches[owner]
        bank = state.charts if state.material_charts is None else state.material_charts
        charts[position] = bank[owner, np.flatnonzero(state.cells[owner] == vertex)[0]]
    return chart_stencil(
        source_cells,
        source_charts,
        source_patches,
        source_cell_ids,
        source_vertex_ids,
        patches,
        charts,
        maximum_pairs,
        budget=budget,
    )


@checked
def surface_metric_source_cover(
    source: _dependency_phydrax_meshing__result.CellMeshingResult, /
) -> _dependency_phydrax_geometry__domain_boundary.MeshingDomainBoundarySource:
    """Recover source trim and span restrictions across accepted chart epochs."""

    if source.certification is None:
        raise ValueError(
            "surface_metric.surface_metric_source_cover must retain accepted source certificates; got uV metric adaptation requires its source-chart certificate"
        )
    retained = source.certification.request.source
    root = (
        retained.root
        if isinstance(
            retained,
            surface_boundary_sources_owner.SurfaceChartBoundarySource,
        )
        else retained
    )
    boundary = (
        root.root
        if isinstance(
            root,
            surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource,
        )
        else root
    )
    if isinstance(
        boundary,
        _owner_phydrax_geometry__surface_source_support.SurfaceNativeRestrictionBoundarySource,
    ):
        boundary = boundary.root
    if not isinstance(
        boundary,
        _dependency_phydrax_geometry__domain_boundary.MeshingDomainBoundarySource,
    ):
        raise ValueError(
            "surface_metric.surface_metric_source_cover must retain source geometry invariants; got uV metric adaptation lacks its continuous trim/chart source owner"
        )
    return boundary


@checked
def prepare_surface_metric_source(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> surface_chart_records_owner.SurfaceChartWitness:
    """Recover source-bound chart uses without inferring seams from coordinates."""

    transfer.source_associations(source)
    transfer.support.require_current()
    retained = (
        None if source.certification is None else source.certification.request.source
    )
    if isinstance(
        retained,
        surface_boundary_sources_owner.SurfaceChartBoundarySource,
    ):
        retained.require_current()
        if not isinstance(
            retained.deformation,
            surface_chart_records_owner.PreparedSurfaceChartDeformation,
        ):
            raise TypeError(
                "surface_metric.prepare_surface_metric_source must retain source geometry invariants; got a radial sphere source must use its SphereMaterialCellAtlas owner"
            )
        witness = retained.deformation.target_witness
        if (
            witness.geometry_id
            != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                source.geometry
            )
            or witness.topology_id != source.mesh.topology_id
        ):
            raise ValueError(
                "surface_metric.source_chart_binding must match the source mesh and geometry; got a stale chart binding"
            )
        return witness
    proof = _owner_phydrax_meshing__surface_association_transfer.source_proof(
        transfer.support, source, transfer.receipts, budget=budget
    )
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in source.mesh.blocks]
    )
    rows = entity_keys_owner.key_rows(proof.cell_ids[:, None], ids[:, None])
    if np.any(rows < 0):
        raise ValueError(
            "surface_metric.prepare_surface_metric_source must retain source geometry invariants; got source support omits current surface cells"
        )
    patches = proof.patches[rows]
    return surface_chart_records_owner.SurfaceChartWitness(
        geometry_id=_dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
            source.geometry
        ),
        topology_id=source.mesh.topology_id,
        domain_id=transfer.support.domain.domain_id,
        cell_global_ids=ids,
        patches=patches,
        charts=np.asarray(proof.source_corners, dtype=np.float64)[rows],
        geometry_entity_ids=tuple(
            transfer.support.domain.entity_id(2, int(patch)) for patch in patches
        ),
        occurrence_paths=tuple(
            transfer.support.domain.source_occurrences[2][int(patch)] for patch in patches
        ),
    )


class _SurfaceMetricExecutionRequest(StrictModule):
    mesh: CellMesh
    metric: ArrayLike
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain
    source_id: str
    source_revision: str
    cell_patches: ArrayLike
    cell_charts: ArrayLike
    cell_classes: ArrayLike | None
    edge_classes: ArrayLike | None
    protected_edges: ArrayLike | None
    fixed_vertices: ArrayLike | None
    maximum_fidelity: float
    minimum_metric_quality: float
    maximum_passes: int
    topology_operations: bool
    relocation: bool
    maximum_vertices: int
    maximum_cells: int
    maximum_operations: int
    maximum_location_pairs: int
    maximum_arc_work_units: int
    record_phase: NativeMeshingPhaseRecorder | None
    controller: SurfaceMetricControllerProtocol | None
    curve_witness: (
        _owner_phydrax_meshing__surface_association_transfer.PreparedSurfaceCurveWitness
        | None
    )
    source_material_charts: np.ndarray | None
    budget: (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    )
    scope: NativeExecutionScope


class _SurfaceMetricPreparedInputs(StrictModule):
    points: np.ndarray
    cells: np.ndarray
    patches: np.ndarray
    charts: np.ndarray
    values: np.ndarray
    classes: np.ndarray
    protected: np.ndarray
    fixed: np.ndarray
    cell_ids: np.ndarray
    schedule: MetricExecutionSchedule


class _SurfaceMetricCarrierPhase(StrictModule):
    source: SurfaceSource
    state: SurfaceState
    vertex_ids: np.ndarray
    material_source: np.ndarray
    original_vertex_count: int
    features: set[tuple[int, int]]
    feature_identity: dict[tuple[int, int], int]
    blocked: set[tuple[int, int]]


class _SurfaceMetricExecutionPhase(StrictModule):
    state: SurfaceState
    counts: MetricOperationCounts
    rejected: int
    work: int
    passes: int
    status: _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus


class _SurfaceMetricCertificatePhase(StrictModule):
    lengths: np.ndarray
    quality: np.ndarray
    fidelity: np.ndarray
    edges: np.ndarray
    arc_bounds: np.ndarray | None
    arc_edge_ids: np.ndarray | None
    arc_binding: str | None
    work: int
    status: _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus


def _prepare_surface_metric_inputs(
    request: _SurfaceMetricExecutionRequest,
) -> _SurfaceMetricPreparedInputs:
    mesh = request.mesh
    metric = request.metric
    domain = request.domain
    source_id = request.source_id
    source_revision = request.source_revision
    cell_patches = request.cell_patches
    cell_charts = request.cell_charts
    cell_classes = request.cell_classes
    protected_edges = request.protected_edges
    fixed_vertices = request.fixed_vertices
    maximum_fidelity = request.maximum_fidelity
    minimum_metric_quality = request.minimum_metric_quality
    maximum_passes = request.maximum_passes
    topology_operations = request.topology_operations
    relocation = request.relocation
    maximum_vertices = request.maximum_vertices
    maximum_cells = request.maximum_cells
    maximum_operations = request.maximum_operations
    maximum_location_pairs = request.maximum_location_pairs
    maximum_arc_work_units = request.maximum_arc_work_units
    record_phase = request.record_phase
    domain.require_current(source_id, source_revision)
    if mesh.ambient_dimension != 3 or any(
        block.cell_kind != "triangle" for block in mesh.blocks
    ):
        raise ValueError(
            "surface_metric.mesh must contain embedded three-dimensional triangles; got incompatible cell family"
        )
    if not np.isfinite(maximum_fidelity) or maximum_fidelity < 0.0:
        raise ValueError(
            "surface_metric.bounds must specify finite nonnegative fidelity and quality in [0, 1]; got invalid bounds"
        )
    validate_metric_quality_floor(
        minimum_metric_quality,
        message="surface_metric.bounds must specify finite nonnegative fidelity and quality in [0, 1]; got invalid bounds",
    )
    schedule = MetricExecutionSchedule(
        maximum_passes=maximum_passes,
        maximum_vertices=maximum_vertices,
        maximum_cells=maximum_cells,
        maximum_operations=maximum_operations,
        maximum_location_pairs=maximum_location_pairs,
        maximum_arc_work_units=maximum_arc_work_units,
    )
    maximum_vertices, maximum_cells = schedule.maximum_vertices, schedule.maximum_cells
    _dependency_phydrax_meshing__tetra_metric.stage_source_limits(
        mesh, maximum_vertices, maximum_cells
    )
    validate_metric_operation_controls(topology_operations, relocation)
    validate_metric_phase_recorder(record_phase)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks]
    )
    patches = np.asarray(cell_patches)
    charts = np.asarray(cell_charts, dtype=np.float64)
    if (
        patches.shape != (cells.shape[0],)
        or not np.issubdtype(patches.dtype, np.integer)
        or np.any((patches < 0) | (patches >= len(domain.patches)))
        or charts.shape != (cells.shape[0], 3, 2)
    ):
        raise ValueError(
            "surface_metric.prepare_surface_metric_inputs must retain the declared source and target bindings; got explicit patch identities and per-cell charts must align with cells"
        )
    patches = patches.astype(np.int32)
    represented = domain.evaluate(np.repeat(patches, 3), charts.reshape((-1, 2))).reshape(
        (-1, 3, 3)
    )
    if (
        np.max(np.linalg.norm(represented - points[cells], axis=-1), initial=0.0)
        > domain.tolerance * domain.scale
    ):
        raise ValueError(
            "surface_metric.prepare_surface_metric_inputs must retain source geometry invariants; got source mesh vertices must lie on their explicitly associated patches"
        )
    values = np.asarray(metric, dtype=np.float64)
    if values.shape != (points.shape[0], 3, 3):
        raise ValueError(
            "surface_metric.metric must carry one 3D SPD tensor per vertex; got incompatible tensor shape"
        )
    with measure_phase(record_phase, "metric_preparation"):
        properties = _dependency_phydrax_meshing__metric.tensor_properties(values)
        if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
            raise ValueError(
                "surface_metric.metric must contain finite Hermitian positive-definite physical tensors; got invalid spectrum"
            )
        values = properties.symmetric
    classes = (
        np.zeros((cells.shape[0],), dtype=np.int64)
        if cell_classes is None
        else np.asarray(cell_classes)
    )
    protected = (
        np.zeros((mesh.entity_set(1).count,), dtype=np.bool_)
        if protected_edges is None
        else np.asarray(protected_edges, dtype=np.bool_)
    )
    fixed = (
        np.zeros((points.shape[0],), dtype=np.bool_)
        if fixed_vertices is None
        else np.asarray(fixed_vertices, dtype=np.bool_)
    )
    if (
        classes.shape != (cells.shape[0],)
        or not np.issubdtype(classes.dtype, np.integer)
        or protected.shape != (mesh.entity_set(1).count,)
        or fixed.shape != (points.shape[0],)
    ):
        raise ValueError(
            "surface_metric.prepare_surface_metric_inputs must align with the declared triangle source entities; got cell classifications and constraint masks must align with source entities"
        )
    if np.any(classes < 0):
        raise ValueError(
            "surface_metric.cell_classes must be nonnegative; got negative classification"
        )
    _, classes = np.unique(classes, return_inverse=True)
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    return _SurfaceMetricPreparedInputs(
        points,
        cells,
        patches,
        charts,
        values,
        classes,
        protected,
        fixed,
        cell_ids,
        schedule,
    )


class _SurfaceMetricBackgroundPhase(StrictModule):
    source: SurfaceSource
    vertex_ids: np.ndarray
    material_source: np.ndarray


def _prepare_surface_metric_background(
    request: _SurfaceMetricExecutionRequest, prepared: _SurfaceMetricPreparedInputs
) -> _SurfaceMetricBackgroundPhase:
    mesh = request.mesh
    source_material_charts = request.source_material_charts
    budget = request.budget
    cells = prepared.cells
    patches = prepared.patches
    charts = prepared.charts
    values = prepared.values
    cell_ids = prepared.cell_ids
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    with budget.temporary_scope():
        if source_material_charts is None:
            budget.reserve(
                int(charts.size),
                int(charts.size)
                * (56 + 2 * _dependency_phydrax__geometry_predicates.bigint_bytes(1075))
                + int(charts.nbytes),
            )
            material_source = np.asarray(
                [Fraction(float(value)) for value in charts.flat], dtype=object
            ).reshape(charts.shape)
        else:
            supplied = np.asarray(source_material_charts)
            if (
                supplied.shape != charts.shape
                or supplied.dtype != object
                or any(not isinstance(value, Fraction) for value in supplied.flat)
            ):
                raise ValueError(
                    "surface_metric.prepare_surface_metric_background must fraction coordinates; got source material charts require Fraction coordinates"
                )
            budget.reserve(0, int(supplied.nbytes) + 128)
            material_source = supplied.copy()
        budget.retain_basis((material_source, tuple(material_source.flat)))
    source = SurfaceSource(
        cells, material_source, patches, cell_ids, vertex_ids, values, charts
    )
    return _SurfaceMetricBackgroundPhase(source, vertex_ids, material_source)


class _SurfaceMetricConstraintPhase(StrictModule):
    features: set[tuple[int, int]]
    feature_identity: dict[tuple[int, int], int]
    blocked: set[tuple[int, int]]


def _prepare_surface_metric_constraints(
    request: _SurfaceMetricExecutionRequest,
    prepared: _SurfaceMetricPreparedInputs,
    state: SurfaceState,
    vertex_ids: np.ndarray,
) -> _SurfaceMetricConstraintPhase:
    mesh = request.mesh
    edge_classes = request.edge_classes
    controller = request.controller
    patches = prepared.patches
    classes = prepared.classes
    protected = prepared.protected
    edges, _, inverse = _edge_table(state)
    rows = entity_keys_owner.key_rows(
        entity_keys_owner.entity_keys(mesh, 1),
        np.sort(vertex_ids[edges], axis=1),
    )
    blocked = {tuple(edge) for edge in edges[protected[rows]].tolist()}
    features: set[tuple[int, int]] = set(blocked)
    feature_codes = (
        np.zeros((mesh.entity_set(1).count,), dtype=np.int64)
        if edge_classes is None
        else np.asarray(edge_classes)
    )
    if (
        feature_codes.shape != (mesh.entity_set(1).count,)
        or not np.issubdtype(feature_codes.dtype, np.integer)
        or np.any(feature_codes < 0)
    ):
        raise ValueError(
            "surface_metric.edge_classes must be nonnegative integer source-entity labels; got invalid classification"
        )
    features.update(tuple(edge) for edge in edges[feature_codes[rows] != 0].tolist())
    for index, edge in enumerate(edges.tolist()):
        incident = np.flatnonzero(np.any(inverse == index, axis=1))
        if (
            incident.size != 2
            or np.unique(patches[incident]).size != 1
            or np.unique(classes[incident]).size != 1
        ):
            features.add(tuple(edge))
        else:
            first, second = incident.tolist()
            first_slots = [
                np.flatnonzero(state.cells[first] == vertex)[0] for vertex in edge
            ]
            second_slots = [
                np.flatnonzero(state.cells[second] == vertex)[0] for vertex in edge
            ]
            if not np.array_equal(
                state.charts[first, first_slots], state.charts[second, second_slots]
            ):
                # Distinct reference witnesses of the same mesh edge form a
                # chart transition; physical coincidence never erases that seam.
                features.add(tuple(edge))
    if mesh.periodic_topology is not None and controller is None:
        raise ValueError(
            "surface_metric.prepare_surface_metric_constraints must its source-owned orbit controller; got periodic UV adaptation requires its source-owned orbit controller"
        )
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    feature_identity = {
        tuple(edge): int(feature_codes[rows[index]])
        if feature_codes[rows[index]] > 0
        else -int(edge_ids[rows[index]]) - 1
        for index, edge in enumerate(edges.tolist())
        if tuple(edge) in features
    }
    return _SurfaceMetricConstraintPhase(features, feature_identity, blocked)


def _prepare_surface_metric_carriers(
    request: _SurfaceMetricExecutionRequest, prepared: _SurfaceMetricPreparedInputs
) -> _SurfaceMetricCarrierPhase:
    mesh = request.mesh
    domain = request.domain
    controller = request.controller
    curve_witness = request.curve_witness
    budget = request.budget
    points = prepared.points
    cells = prepared.cells
    patches = prepared.patches
    charts = prepared.charts
    values = prepared.values
    classes = prepared.classes
    cell_ids = prepared.cell_ids
    background = _prepare_surface_metric_background(request, prepared)
    source, vertex_ids, material_source = (
        background.source,
        background.vertex_ids,
        background.material_source,
    )
    if controller is not None:
        if (
            controller.source.mesh.mesh_id != mesh.mesh_id
            or controller.witness.domain_id != domain.domain_id
        ):
            raise ValueError(
                "surface_metric.prepare_surface_metric_carriers must retain the declared source and target bindings; got the UV controller is stale for the source mesh/domain"
            )
        controller.background = source
        controller.minimum_metric_quality = request.minimum_metric_quality
    state = SurfaceState(
        points.copy(),
        values,
        cells.copy(),
        patches.copy(),
        charts.copy(),
        classes,
        cell_ids.copy(),
        vertex_ids.copy(),
        [{int(value)} for value in cell_ids],
        [{int(value)} for value in vertex_ids],
        int(np.max(cell_ids)) + 1,
        int(np.max(vertex_ids)) + 1,
    )
    original_vertex_count = points.shape[0]
    budget.reserve(0, int(material_source.nbytes) + 128)
    state.material_charts = material_source.copy()
    _admit_material_carrier(
        state, state.material_charts, int(cells.shape[0]), budget=budget
    )
    if not _material_legal(domain, patches, state.material_charts, budget=budget):
        raise ValueError(
            "surface_metric.prepare_surface_metric_carriers must retain source geometry invariants; got the material charts must retain their source orientation"
        )
    if curve_witness is not None:
        curve_witness.require_bound(mesh)
        if curve_witness.support.domain.domain_id != domain.domain_id:
            raise ValueError(
                "surface_metric.curve_binding must match the metric source domain ID; got a different domain binding"
            )

        budget.reserve(int(vertex_ids.size), 1024 * int(vertex_ids.size))
        state.curve_witness = curve_witness
        state.curve_workspace_vertices = int(vertex_ids.size)
        state.vertex_strata = [
            (
                int(dimension),
                int(index),
                np.asarray(parameters, dtype=np.float64).copy(),
            )
            for dimension, index, parameters in zip(
                np.asarray(curve_witness.source_dimensions),
                np.asarray(curve_witness.source_indices),
                np.asarray(curve_witness.source_parameters),
                strict=True,
            )
        ]
    if not _legal(domain, cells, patches, charts, points, np.inf, budget=budget):
        raise ValueError(
            "surface_metric.prepare_surface_metric_carriers must retain source geometry invariants; got the source charts must be exactly oriented, regular surface cells"
        )
    constraints = _prepare_surface_metric_constraints(
        request, prepared, state, vertex_ids
    )
    features, feature_identity, blocked = (
        constraints.features,
        constraints.feature_identity,
        constraints.blocked,
    )
    return _SurfaceMetricCarrierPhase(
        source,
        state,
        vertex_ids,
        material_source,
        original_vertex_count,
        features,
        feature_identity,
        blocked,
    )


class _SurfaceMetricPassProgress(StrictModule):
    state: SurfaceState
    counts: MetricOperationCounts
    rejected: int
    work: int
    passes: int
    status: _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus
    applied: int


class _SurfaceMetricSplitPreparation(StrictModule):
    edges: np.ndarray
    lengths: np.ndarray
    forced: set[int]
    midpoint_metrics: np.ndarray
    split_admissible: np.ndarray
    split_order: np.ndarray


class _SurfaceMetricSplitEligibility(StrictModule):
    admissible: np.ndarray


def _prepare_surface_split_eligibility(
    request: _SurfaceMetricExecutionRequest,
    state: SurfaceState,
    edges: np.ndarray,
    features: set[tuple[int, int]],
) -> _SurfaceMetricSplitEligibility:
    domain, budget = request.domain, request.budget
    split_admissible = np.ones((edges.shape[0],), dtype=np.bool_)
    if state.curve_witness is not None and state.vertex_strata is not None:
        for edge_index, candidate_edge in enumerate(edges):
            key = tuple(map(int, candidate_edge))
            if key not in features:
                continue
            tokens = [state.vertex_strata[int(vertex)] for vertex in candidate_edge]
            interval = state.curve_witness.admissible_edge_interval(
                np.asarray([value[0] for value in tokens]),
                np.asarray([value[1] for value in tokens]),
                np.asarray([value[2] for value in tokens]),
            )
            if interval is None:
                split_admissible[edge_index] = False
                continue
            curve, left, right = interval
            point, target_uses = state.curve_witness.evaluate_curve(
                curve, (left + right) / 2, budget=budget
            )
            _, left_uses = state.curve_witness.evaluate_curve(curve, left, budget=budget)
            _, right_uses = state.curve_witness.evaluate_curve(
                curve, right, budget=budget
            )
            candidate_rows = np.flatnonzero(
                np.sum(np.isin(state.cells, candidate_edge), axis=1) == 2
            )
            candidate_charts = []
            for candidate_row in candidate_rows:
                anchors = tuple(
                    (
                        state.charts[
                            candidate_row,
                            np.flatnonzero(state.cells[candidate_row] == vertex)[0],
                        ],
                        uses,
                    )
                    for vertex, uses in zip(
                        candidate_edge,
                        (left_uses, right_uses),
                        strict=True,
                    )
                )
                chart = _curve_occurrence_chart(
                    int(state.patches[candidate_row]),
                    anchors,
                    target_uses,
                    domain.tolerance,
                )
                if chart is None:
                    split_admissible[edge_index] = False
                    break
                candidate_charts.append(chart)
            if split_admissible[edge_index] and candidate_rows.size:
                represented = domain.evaluate(
                    state.patches[candidate_rows],
                    np.asarray(candidate_charts, dtype=np.float64),
                )
                if (
                    np.max(
                        np.linalg.norm(represented - point, axis=1),
                        initial=0.0,
                    )
                    > domain.tolerance * domain.scale
                ):
                    split_admissible[edge_index] = False
    return _SurfaceMetricSplitEligibility(split_admissible)


class _SurfaceMetricOperationResult(StrictModule):
    state: SurfaceState
    changed: bool
    copies: int


def _stage_surface_metric_operation(
    request: _SurfaceMetricExecutionRequest,
    prepared: _SurfaceMetricPreparedInputs,
    carrier: _SurfaceMetricCarrierPhase,
    state: SurfaceState,
    operation: _dependency_phydrax_meshing__periodic.PeriodicMetricOperation,
    vertices: tuple[int, ...],
    split_metric: np.ndarray | None = None,
) -> _SurfaceMetricOperationResult:
    controller = request.controller
    if controller is not None:
        staged = controller.stage(
            state,
            operation,
            vertices,
            prepared.fixed,
            carrier.features,
            carrier.feature_identity,
            carrier.blocked,
        )
        if staged is None:
            return _SurfaceMetricOperationResult(state, False, 0)
        return _SurfaceMetricOperationResult(staged, True, controller.operation_count)
    domain, budget = request.domain, request.budget
    match operation:
        case "split":
            if split_metric is None:
                raise ValueError(
                    "surface_metric.split_metric must provide a midpoint tensor; got None"
                )
            changed = split(
                domain,
                state,
                np.asarray(vertices, dtype=np.int32),
                split_metric,
                source_curve=vertices in carrier.features,
                budget=budget,
            )
        case "collapse":
            first, second = vertices
            changed = collapse(
                domain,
                state,
                first,
                second,
                prepared.fixed,
                carrier.features,
                request.minimum_metric_quality,
                request.maximum_fidelity,
                budget=budget,
            ) or collapse(
                domain,
                state,
                second,
                first,
                prepared.fixed,
                carrier.features,
                request.minimum_metric_quality,
                request.maximum_fidelity,
                budget=budget,
            )
        case "flip":
            changed = flip(
                domain,
                state,
                np.asarray(vertices, dtype=np.int32),
                carrier.features,
                request.maximum_fidelity,
                budget=budget,
            )
        case "relocate":
            changes, _ = relocate(
                domain,
                state,
                prepared.fixed,
                carrier.features,
                request.maximum_fidelity,
                1,
                prepared.schedule.maximum_location_pairs,
                carrier.source,
                carrier.feature_identity,
                carrier.blocked,
                request.minimum_metric_quality,
                selected_vertices=frozenset(vertices),
                budget=budget,
            )
            changed = changes == 1
        case _:
            raise ValueError(
                "surface_metric.operation must select a supported edit; got unknown operation"
            )
    return _SurfaceMetricOperationResult(state, changed, int(changed))


def _prepare_surface_metric_splits(
    request: _SurfaceMetricExecutionRequest,
    prepared: _SurfaceMetricPreparedInputs,
    carrier: _SurfaceMetricCarrierPhase,
    progress: _SurfaceMetricPassProgress,
    edges: np.ndarray,
    first: np.ndarray,
    lengths: np.ndarray,
    forced: set[int],
    material_stencil_cache: dict[
        tuple[int, tuple[tuple[int, int], ...]], tuple[np.ndarray, np.ndarray, int]
    ]
    | None,
) -> _SurfaceMetricSplitPreparation:
    maximum_location_pairs = prepared.schedule.maximum_location_pairs
    budget = request.budget
    cells = prepared.cells
    patches = prepared.patches
    values = prepared.values
    cell_ids = prepared.cell_ids
    source = carrier.source
    state = progress.state
    vertex_ids = carrier.vertex_ids
    features = carrier.features
    blocked = carrier.blocked
    owner, local = first // 3, first % 3
    bank = state.charts if state.material_charts is None else state.material_charts
    midpoint_charts = np.mean(bank[owner[:, None], _LOCAL_EDGES[local]], axis=1)
    sampling = chart_stencil(
        cells,
        source.charts,
        patches,
        cell_ids,
        vertex_ids,
        state.patches[owner],
        midpoint_charts,
        maximum_location_pairs,
        material_cache=material_stencil_cache,
        budget=budget,
    )
    sample_rows = entity_keys_owner.key_rows(
        vertex_ids[:, None], sampling[0].reshape((-1, 1))
    ).reshape(sampling[0].shape)
    midpoint_metrics = np.asarray(
        _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
            values[sample_rows], sampling[1]
        ),
        dtype=np.float64,
    )
    split_admissible = _prepare_surface_split_eligibility(
        request, state, edges, features
    ).admissible
    split_order = np.argsort(-lengths, kind="stable")
    split_free = np.asarray(
        [
            tuple(map(int, edges[index])) not in features
            and tuple(map(int, edges[index])) not in blocked
            for index in split_order
        ],
        dtype=np.bool_,
    )
    free_order = split_order[split_free]
    free_indices = set(map(int, free_order))
    feature_order = np.asarray(
        [
            index
            for index in np.argsort(lengths, kind="stable")
            if int(index) not in free_indices
        ],
        dtype=np.int64,
    )
    split_order = np.concatenate((free_order, feature_order))
    return _SurfaceMetricSplitPreparation(
        edges, lengths, forced, midpoint_metrics, split_admissible, split_order
    )


def _run_surface_split_pass(
    request: _SurfaceMetricExecutionRequest,
    prepared: _SurfaceMetricPreparedInputs,
    carrier: _SurfaceMetricCarrierPhase,
    progress: _SurfaceMetricPassProgress,
    split_phase: _SurfaceMetricSplitPreparation,
) -> _SurfaceMetricPassProgress:
    maximum_vertices = prepared.schedule.maximum_vertices
    maximum_cells = prepared.schedule.maximum_cells
    maximum_operations = prepared.schedule.maximum_operations
    record_phase = request.record_phase
    controller = request.controller
    budget = request.budget
    state = progress.state
    features = carrier.features
    feature_identity = carrier.feature_identity
    blocked = carrier.blocked
    counts = progress.counts
    rejected = progress.rejected
    work = progress.work
    passes = progress.passes
    status = progress.status
    applied = progress.applied
    edges = split_phase.edges
    lengths = split_phase.lengths
    forced = split_phase.forced
    midpoint_metrics = split_phase.midpoint_metrics
    split_admissible = split_phase.split_admissible
    split_order = split_phase.split_order
    split_vertices: set[int] = set()
    for index in split_order:
        if not split_admissible[index]:
            rejected += 1
            continue
        if lengths[index] <= _UPPER and index not in forced:
            continue
        if (
            work >= maximum_operations
            or state.points.shape[0] >= maximum_vertices
            or state.cells.shape[0] + 2 > maximum_cells
        ):
            status = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
            break
        edge = edges[index]
        if any(int(vertex) in split_vertices for vertex in edge):
            continue
        if tuple(edge.tolist()) in blocked:
            rejected += 1
            continue
        work += 1
        _consume_native_work(1, geometry_queries=0, budget=budget)
        vertex = state.points.shape[0]
        with measure_phase(record_phase, "metric_split"):
            operation_result = _stage_surface_metric_operation(
                request,
                prepared,
                carrier,
                state,
                "split",
                tuple(map(int, edge)),
                midpoint_metrics[index],
            )
            state, changed = operation_result.state, operation_result.changed
        if changed:
            if controller is None and tuple(edge.tolist()) in features:
                key = tuple(edge.tolist())
                identity = feature_identity.pop(key)
                features.remove(key)
                first, second = int(edge[0]), int(edge[1])
                children = (
                    (min(first, vertex), max(first, vertex)),
                    (min(vertex, second), max(vertex, second)),
                )
                features.update(children)
                feature_identity.update((child, identity) for child in children)
            copies = operation_result.copies
            counts = replace(counts, splits=counts.splits + copies)
            applied += copies
            split_vertices.update(map(int, edge))
        else:
            rejected += 1
    return _SurfaceMetricPassProgress(
        state, counts, rejected, work, passes, status, applied
    )


def _run_surface_collapse_pass(
    request: _SurfaceMetricExecutionRequest,
    prepared: _SurfaceMetricPreparedInputs,
    carrier: _SurfaceMetricCarrierPhase,
    progress: _SurfaceMetricPassProgress,
) -> _SurfaceMetricPassProgress:
    domain = request.domain
    maximum_operations = prepared.schedule.maximum_operations
    record_phase = request.record_phase
    budget = request.budget
    state = progress.state
    original_vertex_count = carrier.original_vertex_count
    counts = progress.counts
    rejected = progress.rejected
    work = progress.work
    passes = progress.passes
    status = progress.status
    applied = progress.applied
    edges, _, _ = _edge_table(state)
    lengths, _, _ = _measure(domain, state, budget=budget)
    for index in np.argsort(lengths, kind="stable"):
        if applied:
            break
        if lengths[index] >= _LOWER or work >= maximum_operations:
            break
        if np.any(edges[index] >= original_vertex_count):
            continue
        work += 1
        _consume_native_work(1, geometry_queries=0, budget=budget)
        first, second = edges[index].tolist()
        with measure_phase(record_phase, "metric_collapse"):
            operation_result = _stage_surface_metric_operation(
                request, prepared, carrier, state, "collapse", (first, second)
            )
            state, changed = operation_result.state, operation_result.changed
        if changed:
            copies = operation_result.copies
            counts = replace(counts, collapses=counts.collapses + copies)
            applied += copies
        else:
            rejected += 1
    return _SurfaceMetricPassProgress(
        state, counts, rejected, work, passes, status, applied
    )


def _run_surface_flip_pass(
    request: _SurfaceMetricExecutionRequest,
    prepared: _SurfaceMetricPreparedInputs,
    carrier: _SurfaceMetricCarrierPhase,
    progress: _SurfaceMetricPassProgress,
) -> _SurfaceMetricPassProgress:
    maximum_operations = prepared.schedule.maximum_operations
    record_phase = request.record_phase
    budget = request.budget
    state = progress.state
    counts = progress.counts
    rejected = progress.rejected
    work = progress.work
    passes = progress.passes
    status = progress.status
    applied = progress.applied
    edges, _, _ = _edge_table(state)
    for edge in edges:
        if applied:
            break
        if work >= maximum_operations:
            break
        work += 1
        _consume_native_work(1, geometry_queries=0, budget=budget)
        with measure_phase(record_phase, "metric_reconnection"):
            operation_result = _stage_surface_metric_operation(
                request, prepared, carrier, state, "flip", tuple(map(int, edge))
            )
            state, changed = operation_result.state, operation_result.changed
        if changed:
            copies = operation_result.copies
            counts = replace(counts, reconnections=counts.reconnections + copies)
            applied += copies
    return _SurfaceMetricPassProgress(
        state, counts, rejected, work, passes, status, applied
    )


def _run_surface_relocation_pass(
    request: _SurfaceMetricExecutionRequest,
    prepared: _SurfaceMetricPreparedInputs,
    carrier: _SurfaceMetricCarrierPhase,
    progress: _SurfaceMetricPassProgress,
) -> _SurfaceMetricPassProgress:
    domain = request.domain
    maximum_fidelity = request.maximum_fidelity
    minimum_metric_quality = request.minimum_metric_quality
    maximum_operations = prepared.schedule.maximum_operations
    maximum_location_pairs = prepared.schedule.maximum_location_pairs
    record_phase = request.record_phase
    controller = request.controller
    budget = request.budget
    fixed = prepared.fixed
    source = carrier.source
    state = progress.state
    features = carrier.features
    feature_identity = carrier.feature_identity
    blocked = carrier.blocked
    counts = progress.counts
    rejected = progress.rejected
    work = progress.work
    passes = progress.passes
    status = progress.status
    applied = progress.applied
    with measure_phase(record_phase, "metric_relocation"):
        if controller is None:
            changes, attempts = relocate(
                domain,
                state,
                fixed,
                features,
                maximum_fidelity,
                maximum_operations - work,
                maximum_location_pairs,
                source,
                feature_identity,
                blocked,
                minimum_metric_quality,
                record_phase=record_phase,
                budget=budget,
            )
        else:
            changes, attempts = 0, 0
            for vertex in np.unique(state.cells).tolist():
                if attempts >= maximum_operations - work:
                    break
                attempts += 1
                operation_result = _stage_surface_metric_operation(
                    request, prepared, carrier, state, "relocate", (vertex,)
                )
                state = operation_result.state
                changes += operation_result.copies
    counts = replace(counts, relocations=counts.relocations + changes)
    work += attempts
    applied += changes
    return _SurfaceMetricPassProgress(
        state, counts, rejected, work, passes, status, applied
    )


def _run_surface_metric_execution(
    request: _SurfaceMetricExecutionRequest,
    prepared: _SurfaceMetricPreparedInputs,
    carrier: _SurfaceMetricCarrierPhase,
) -> _SurfaceMetricExecutionPhase:
    domain = request.domain
    maximum_fidelity = request.maximum_fidelity
    minimum_metric_quality = request.minimum_metric_quality
    topology_operations = request.topology_operations
    relocation = request.relocation
    maximum_operations = prepared.schedule.maximum_operations
    budget = request.budget
    state = carrier.state
    original_vertex_count = carrier.original_vertex_count
    features = carrier.features
    blocked = carrier.blocked
    counts = MetricOperationCounts()
    rejected, work, passes = 0, 0, 0
    status = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.PASS_LIMIT
    material_stencil_cache: (
        dict[
            tuple[int, tuple[tuple[int, int], ...]],
            tuple[np.ndarray, np.ndarray, int],
        ]
        | None
    ) = {} if state.material_charts is not None else None

    def checkpoint() -> None:

        budget.ticket.checkpoint()

    for passes in metric_execution_passes(prepared.schedule, checkpoint):
        lengths, quality, fidelity = _measure(domain, state, budget=budget)
        edges, first, cell_edges = _edge_table(state)
        if (
            topology_operations
            and _surface_size_complete(
                lengths,
                lengths,
                edges,
                features,
                blocked,
                generated_from=original_vertex_count,
            )
            and np.all(quality >= minimum_metric_quality)
            and np.all(fidelity <= maximum_fidelity)
        ):
            status = (
                _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.COMPLETE
            )
            break
        applied = 0
        bad_cells = np.flatnonzero(fidelity > maximum_fidelity)
        forced = set(
            cell_edges[
                bad_cells, np.argmax(lengths[cell_edges[bad_cells]], axis=1)
            ].tolist()
        )
        if topology_operations:
            if work >= maximum_operations:
                status = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
                break
            progress = _SurfaceMetricPassProgress(
                state, counts, rejected, work, passes, status, applied
            )
            split_phase = _prepare_surface_metric_splits(
                request,
                prepared,
                carrier,
                progress,
                edges,
                first,
                lengths,
                forced,
                material_stencil_cache,
            )
            progress = _run_surface_split_pass(
                request, prepared, carrier, progress, split_phase
            )
            state, counts, rejected, work, status, applied = (
                progress.state,
                progress.counts,
                progress.rejected,
                progress.work,
                progress.status,
                progress.applied,
            )
            if (
                status
                is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
            ):
                break
            progress = _run_surface_collapse_pass(request, prepared, carrier, progress)
            progress = _run_surface_flip_pass(request, prepared, carrier, progress)
            state, counts, rejected, work, status, applied = (
                progress.state,
                progress.counts,
                progress.rejected,
                progress.work,
                progress.status,
                progress.applied,
            )
        if relocation and work < maximum_operations:
            progress = _SurfaceMetricPassProgress(
                state, counts, rejected, work, passes, status, applied
            )
            progress = _run_surface_relocation_pass(request, prepared, carrier, progress)
            state, counts, rejected, work, status, applied = (
                progress.state,
                progress.counts,
                progress.rejected,
                progress.work,
                progress.status,
                progress.applied,
            )
        if applied == 0:
            status = (
                _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
                if work >= maximum_operations
                else _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.STALLED
            )
            if (
                not topology_operations
                and work < maximum_operations
                and np.all(quality >= minimum_metric_quality)
                and np.all(fidelity <= maximum_fidelity)
            ):
                status = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.COMPLETE
            break
    return _SurfaceMetricExecutionPhase(state, counts, rejected, work, passes, status)


def _certify_surface_metric_execution(
    request: _SurfaceMetricExecutionRequest,
    prepared: _SurfaceMetricPreparedInputs,
    carrier: _SurfaceMetricCarrierPhase,
    execution: _SurfaceMetricExecutionPhase,
) -> _SurfaceMetricCertificatePhase:
    domain = request.domain
    maximum_fidelity = request.maximum_fidelity
    minimum_metric_quality = request.minimum_metric_quality
    topology_operations = request.topology_operations
    maximum_location_pairs = prepared.schedule.maximum_location_pairs
    maximum_arc_work_units = prepared.schedule.maximum_arc_work_units
    record_phase = request.record_phase
    budget = request.budget
    source = carrier.source
    state = execution.state
    original_vertex_count = carrier.original_vertex_count
    features = carrier.features
    blocked = carrier.blocked
    work = execution.work
    status = execution.status
    lengths, quality, fidelity = _measure(domain, state, budget=budget)
    edges, _, _ = _edge_table(state)
    if not np.all(np.isfinite(fidelity)):
        raise ValueError(
            "surface_metric.fidelity must carry finite continuous interpolation bounds; got nonfinite enclosure"
        )
    if (
        status
        is not _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
        and topology_operations
        and _surface_size_complete(
            lengths,
            lengths,
            edges,
            features,
            blocked,
            generated_from=original_vertex_count,
        )
        and np.all(quality >= minimum_metric_quality)
        and np.all(fidelity <= maximum_fidelity)
    ):
        status = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.COMPLETE
    arc_bounds, arc_edge_ids, arc_binding = None, None, None
    if status is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.COMPLETE:
        with measure_phase(record_phase, "metric_preparation"):
            arc_bounds, arc_work = _certify_surface_metric_arcs(
                domain,
                state,
                source,
                maximum_work=maximum_arc_work_units,
                maximum_pairs=maximum_location_pairs,
                features=features,
                blocked=blocked,
                generated_from=original_vertex_count,
                budget=budget,
            )
        work += arc_work

        with measure_phase(record_phase, "metric_preparation"):
            with budget.temporary_scope():
                quality, quality_work = _certify_surface_metric_quality(
                    domain,
                    state,
                    source,
                    minimum_quality=minimum_metric_quality,
                    maximum_work=maximum_arc_work_units - arc_work,
                    budget=budget,
                )
        work += quality_work
        edges, _, _ = _edge_table(state)
        arc_edge_ids = np.sort(state.vertex_ids[edges], axis=1)
        arc_binding = canonical_fingerprint(
            {
                "kind": "source-chart-metric-arc-enclosure",
                "domain": domain.domain_id,
                "source": domain.source_id,
                "revision": domain.source_revision,
                "background": array_tree_fingerprint(
                    (
                        source.vertex_ids,
                        source.cell_ids,
                        source.cells,
                        source.physical_charts,
                        source.patches,
                        source.metric,
                    )
                ),
                "background_material_charts": tuple(
                    tuple((value.numerator, value.denominator) for value in triangle.flat)
                    for triangle in source.charts
                ),
                "target_edges": array_tree_fingerprint(arc_edge_ids),
                "target_charts": array_tree_fingerprint(
                    (state.cells, state.charts, state.patches)
                ),
                "target_material_charts": None
                if state.material_charts is None
                else tuple(
                    tuple((value.numerator, value.denominator) for value in triangle.flat)
                    for triangle in state.material_charts
                ),
                "quality_cells": array_tree_fingerprint(state.cell_ids),
                "quality_lower_bounds": array_tree_fingerprint(quality),
            }
        )
        if topology_operations and not _surface_size_complete(
            arc_bounds[:, 0],
            arc_bounds[:, 1],
            edges,
            features,
            blocked,
            generated_from=original_vertex_count,
        ):
            status = (
                _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.STALLED
            )
        if not np.all(quality >= minimum_metric_quality):
            status = (
                _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.STALLED
            )
    return _SurfaceMetricCertificatePhase(
        lengths,
        quality,
        fidelity,
        edges,
        arc_bounds,
        arc_edge_ids,
        arc_binding,
        work,
        status,
    )


def _assemble_surface_metric_outcome(
    request: _SurfaceMetricExecutionRequest,
    prepared: _SurfaceMetricPreparedInputs,
    carrier: _SurfaceMetricCarrierPhase,
    execution: _SurfaceMetricExecutionPhase,
    certificate: _SurfaceMetricCertificatePhase,
) -> SurfaceMetricOutcome:
    mesh = request.mesh
    domain = request.domain
    minimum_metric_quality = request.minimum_metric_quality
    topology_operations = request.topology_operations
    maximum_vertices = prepared.schedule.maximum_vertices
    maximum_cells = prepared.schedule.maximum_cells
    maximum_operations = prepared.schedule.maximum_operations
    maximum_location_pairs = prepared.schedule.maximum_location_pairs
    record_phase = request.record_phase
    controller = request.controller
    budget = request.budget
    cells = prepared.cells
    patches = prepared.patches
    cell_ids = prepared.cell_ids
    source = carrier.source
    state = execution.state
    material_source = carrier.material_source
    counts = execution.counts
    rejected = execution.rejected
    work = certificate.work
    passes = execution.passes
    status = certificate.status
    lengths = certificate.lengths
    quality = certificate.quality
    fidelity = certificate.fidelity
    arc_bounds = certificate.arc_bounds
    arc_edge_ids = certificate.arc_edge_ids
    arc_binding = certificate.arc_binding
    with measure_phase(record_phase, "lineage_construction"):
        edit, target_metric, ordered_patches, ordered_charts, ordered_material = (
            assemble_surface(
                mesh,
                state,
                cells,
                patches,
                source.charts,
                maximum_location_pairs,
                budget=budget,
            )
        )
        if controller is not None:
            if controller.prior_stage is None:
                edit = edit._replace(
                    periodic_orbits=_unchanged_periodic_witness(
                        controller.source, budget=budget
                    )
                )
            else:
                edit = controller.prior_stage.edit
    evidence = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingEvidence(
        status,
        passes=passes,
        counts=counts.evidence_counts(),
        rejected_operations=rejected,
        work_units=work,
        lengths=lengths,
        quality=quality,
        maximum_fidelity_bound=float(np.max(fidelity, initial=0.0)),
        metric_quality_floor=minimum_metric_quality,
        criterion=_dependency_phydrax_meshing__tetra_metric.MetricRemeshingCriterion.UNIT_MESH
        if topology_operations
        else _dependency_phydrax_meshing__tetra_metric.MetricRemeshingCriterion.RELOCATION_FIXED_POINT,
        native_work_units=0,
        metric_arc_bounds=arc_bounds,
        metric_arc_edge_ids=arc_edge_ids,
        metric_arc_binding=arc_binding,
        resource_message=(
            "surface_metric.capacity must admit the requested edit; got exhausted operation or entity capacity"
            if status
            is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
            else None
        ),
        resource_requested=(
            ("maximum_operations", float(maximum_operations)),
            ("maximum_vertices", float(maximum_vertices)),
            ("maximum_cells", float(maximum_cells)),
        )
        if status
        is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
        else (),
        resource_achieved=(
            ("operation_attempts", float(work)),
            ("vertices", float(state.points.shape[0])),
            ("cells", float(state.cells.shape[0])),
        )
        if status
        is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
        else (),
    )
    live = np.unique(state.cells)
    vertex_strata = surface_vertex_strata(state, live)
    geometry = SurfaceMetricGeometryWitness(
        edit,
        ordered_patches,
        ordered_charts,
        ordered_material,
        cell_ids,
        material_source,
        *vertex_strata,
        domain.source_id,
        domain.source_revision,
        domain.domain_id,
        tuple(domain.entity_id(2, patch) for patch in ordered_patches.tolist()),
        tuple(domain.source_occurrences[2][patch] for patch in ordered_patches.tolist()),
    )
    return SurfaceMetricOutcome(
        geometry,
        target_metric,
        evidence,
        None if controller is None else controller.prior_stage,
    )


@checked
def execute_surface_metric_adaptation(
    mesh: CellMesh,
    metric: ArrayLike,
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    /,
    *,
    source_id: str,
    source_revision: str,
    cell_patches: ArrayLike,
    cell_charts: ArrayLike,
    cell_classes: ArrayLike | None = None,
    edge_classes: ArrayLike | None = None,
    protected_edges: ArrayLike | None = None,
    fixed_vertices: ArrayLike | None = None,
    maximum_fidelity: float,
    minimum_metric_quality: float = 0.05,
    maximum_passes: int = 16,
    topology_operations: bool = True,
    relocation: bool = True,
    maximum_vertices: int = 1000000,
    maximum_cells: int = 2000000,
    maximum_operations: int = 100000,
    maximum_location_pairs: int = 10000000,
    maximum_arc_work_units: int = 100000000,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    controller: SurfaceMetricControllerProtocol | None = None,
    curve_witness: _owner_phydrax_meshing__surface_association_transfer.PreparedSurfaceCurveWitness
    | None = None,
    source_material_charts: np.ndarray | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> SurfaceMetricOutcome:
    """Remesh source-bound parametric patches with certified metric bounds.

    Gauss lengths select proposals. Completion requires outward derivative
    integration in the covered log-Euclidean source-chart background. Publication
    separately verifies interpolation fidelity, coordinate-map reconstruction and
    global embedding.
    """
    request = _SurfaceMetricExecutionRequest(
        mesh=mesh,
        metric=metric,
        domain=domain,
        source_id=source_id,
        source_revision=source_revision,
        cell_patches=cell_patches,
        cell_charts=cell_charts,
        cell_classes=cell_classes,
        edge_classes=edge_classes,
        protected_edges=protected_edges,
        fixed_vertices=fixed_vertices,
        maximum_fidelity=maximum_fidelity,
        minimum_metric_quality=minimum_metric_quality,
        maximum_passes=maximum_passes,
        topology_operations=topology_operations,
        relocation=relocation,
        maximum_vertices=maximum_vertices,
        maximum_cells=maximum_cells,
        maximum_operations=maximum_operations,
        maximum_location_pairs=maximum_location_pairs,
        maximum_arc_work_units=maximum_arc_work_units,
        record_phase=record_phase,
        controller=controller,
        curve_witness=curve_witness,
        source_material_charts=source_material_charts,
        budget=budget,
        scope=scope,
    )
    prepared = _prepare_surface_metric_inputs(request)
    with budget.activate(), budget.ticket.native_target():
        carrier = _prepare_surface_metric_carriers(request, prepared)
        execution = _run_surface_metric_execution(request, prepared, carrier)
        certificate = _certify_surface_metric_execution(
            request, prepared, carrier, execution
        )
        return _assemble_surface_metric_outcome(
            request, prepared, carrier, execution, certificate
        )


@checked
def surface_vertex_strata(
    state: SurfaceState,
    live: np.ndarray,
    /,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if state.vertex_strata is None:
        return None, None, None
    tokens = [state.vertex_strata[int(vertex)] for vertex in live]
    return (
        np.asarray([token[0] for token in tokens], dtype=np.int32),
        np.asarray([token[1] for token in tokens], dtype=np.int32),
        np.asarray([token[2] for token in tokens], dtype=np.float64),
    )


@checked
def assemble_surface(
    mesh: CellMesh,
    state: SurfaceState,
    source_cells: np.ndarray,
    source_patches: np.ndarray,
    source_charts: np.ndarray,
    maximum_pairs: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    live = np.unique(state.cells)
    row_of = np.full((state.points.shape[0],), -1, dtype=np.int32)
    row_of[live] = np.arange(live.size, dtype=np.int32)
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    source_cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    stencil = _stencil(
        source_cells,
        source_charts,
        source_patches,
        source_cell_ids,
        ids,
        state,
        live,
        maximum_pairs,
        budget=budget,
    )
    owner_of = {
        int(identifier): index
        for index, block in enumerate(mesh.blocks)
        for identifier in np.asarray(block.global_ids).tolist()
    }
    owners = np.asarray(
        [owner_of[min(parent)] for parent in state.parents], dtype=np.int32
    )
    block_rows = tuple(
        np.flatnonzero(owners == index)[
            np.argsort(state.cell_ids[owners == index], kind="stable")
        ]
        for index in range(len(mesh.blocks))
    )
    blocks = _dependency_phydrax_meshing__topology_edit.source_family_blocks(
        tuple((block.name, block.cell_kind) for block in mesh.blocks),
        tuple(row_of[state.cells[rows]] for rows in block_rows),
        tuple(state.cell_ids[rows] for rows in block_rows),
    )
    collapsed = _dependency_phydrax_meshing__tetra_metric.resolved_collapses(
        state.vertex_ids, ids.size, state.vertex_successor
    )
    relocated = set() if state.relocated_vertices is None else state.relocated_vertices
    vertex_pairs = [
        (
            parent,
            int(state.vertex_ids[vertex]),
            int(
                _dependency_phydrax_meshing__lineage.EntityLineageKind.RELOCATED
                if vertex in relocated
                else _dependency_phydrax_meshing__lineage.EntityLineageKind.REFINED_FROM
            ),
        )
        for vertex in live.tolist()
        if vertex >= ids.size
        for parent in sorted(state.vertex_sources[vertex])
    ]
    vertex_pairs.extend(
        (
            int(state.vertex_ids[vertex]),
            int(state.vertex_ids[vertex]),
            int(_dependency_phydrax_meshing__lineage.EntityLineageKind.RELOCATED),
        )
        for vertex in live.tolist()
        if vertex < ids.size and vertex in relocated
    )
    vertex_pairs.extend(
        (
            source,
            target,
            int(_dependency_phydrax_meshing__lineage.EntityLineageKind.COLLAPSED_INTO),
        )
        for source, target in sorted(collapsed.items())
    )
    cell_kinds = {} if state.cell_kinds is None else state.cell_kinds
    cell_pairs = [
        (
            parent,
            int(identifier),
            cell_kinds.get(
                int(identifier),
                int(_dependency_phydrax_meshing__lineage.EntityLineageKind.PRESERVED),
            ),
        )
        for parents, identifier in zip(state.parents, state.cell_ids, strict=True)
        for parent in sorted(parents)
        if parent != identifier or int(identifier) in cell_kinds
    ]
    relations = []
    for dimension, pairs in ((0, vertex_pairs), (1, []), (2, cell_pairs)):
        if dimension == 1:
            relations.append(
                _dependency_phydrax_meshing__tetra_metric.subentity_relations(
                    mesh,
                    1,
                    state.cells,
                    state.vertex_ids,
                    state.vertex_sources,
                    collapsed_vertices=collapsed,
                )
            )
            continue
        width = 2 if dimension == 1 else 1
        values = np.asarray(pairs, dtype=np.int64).reshape((-1, 3))
        if pairs:
            relations.append(
                _dependency_phydrax_meshing__topology_edit.EntityRelations(
                    dimension, values[:, :1], values[:, 1:2], values[:, 2].astype(np.int8)
                )
            )
        else:
            relations.append(
                _dependency_phydrax_meshing__topology_edit.EntityRelations(
                    dimension,
                    np.empty((0, width), dtype=np.int64),
                    np.empty((0, width), dtype=np.int64),
                    np.empty((0,), dtype=np.int8),
                )
            )
    ordered = np.concatenate(block_rows)
    edit = _dependency_phydrax_meshing__topology_edit.CellTopologyEdit(
        "local_reconnection",
        state.points[live],
        state.vertex_ids[live],
        blocks,
        *stencil,
        tuple(relations),
    )
    material = state.charts if state.material_charts is None else state.material_charts
    return (
        edit,
        state.metric[live],
        state.patches[ordered],
        state.charts[ordered],
        material[ordered],
    )


@checked
def reconstruct_surface_metric_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    outcome: SurfaceMetricOutcome | SurfaceMetricGeometryWitness,
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    /,
    *,
    maximum_fidelity: float,
    policy: transition_types_owner.CellGeometryTransitionPolicy | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_atlas: _owner_phydrax_geometry__surface_source_support.SurfaceSourceRootAtlas
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> chart_reconstruction_owner.SurfaceGeometryReconstruction:
    """Stage degree-preserving target geometry; old inverse coverage stays explicit."""
    match outcome:
        case SurfaceMetricOutcome(geometry=geometry):
            pass
        case SurfaceMetricGeometryWitness():
            geometry = outcome
        case _:
            raise TypeError(
                "surface_metric.outcome must contain source-bound edit and chart data; got an unsupported outcome type"
            )
    domain.require_current(geometry.source_id, geometry.source_revision)
    if geometry.domain_id != domain.domain_id:
        raise ValueError(
            "surface_metric.reconstruct_surface_metric_geometry must retain the declared source and target bindings; got the reconstruction domain must match the accepted chart witnesses"
        )
    identifiers = np.concatenate([block.cell_ids for block in geometry.edit.blocks])
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    if not np.array_equal(identifiers, target_ids):
        raise ValueError(
            "surface_metric.reconstruct_surface_metric_geometry must retain the declared source and target bindings; got the reconstruction target must preserve the ordered edit cell identities"
        )
    source_vertices = entity_keys_owner.key_rows(
        np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)[:, None],
        geometry.edit.vertex_global_ids[:, None],
    )
    if np.any(source_vertices < 0) or not np.array_equal(
        np.asarray(target_mesh.coordinates)[source_vertices],
        geometry.edit.coordinates,
    ):
        raise ValueError(
            "surface_metric.reconstruct_surface_metric_geometry must retain source geometry invariants; got the target must be the staged edit, not an unbound mesh"
        )
    if source_atlas is None:
        degree = _dependency_phydrax_discretization__cell_geometry_transfer.nested_geometry_degree(
            source_mesh, source_geometry
        )
        layout = (
            CellGeometrySpec.affine(target_mesh)
            if degree == 1
            else _dependency_phydrax_meshing__curving.straight_geometry(
                target_mesh, degree
            )
        )
    else:
        layout = None
    with measure_phase(record_phase, "geometry_transition"):
        return _dependency_phydrax_discretization__cell_geometry_transfer.reconstruct_parametric_surface_cell_geometry(
            source_mesh,
            source_geometry,
            target_mesh,
            layout,
            domain,
            domain_id=geometry.domain_id,
            cell_ids=identifiers,
            cell_patches=geometry.cell_patches,
            cell_charts=geometry.cell_charts,
            cell_geometry_entity_ids=geometry.cell_geometry_entity_ids,
            cell_occurrence_paths=geometry.cell_occurrence_paths,
            maximum_fidelity=maximum_fidelity,
            policy=policy,
            source_atlas=source_atlas,
            budget=budget,
        )


class SurfaceMetricGeometryTransition(NamedTuple):
    transition: transition_types_owner.CellGeometryTransition
    deformation: surface_chart_records_owner.PreparedSurfaceChartDeformation
    target_mesh: CellMesh


@checked
def prepare_surface_metric_geometry_transition(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    outcome: SurfaceMetricOutcome | SurfaceMetricGeometryWitness,
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    source_witness: surface_chart_records_owner.SurfaceChartWitness,
    /,
    *,
    maximum_fidelity: float,
    policy: transition_types_owner.CellGeometryTransitionPolicy,
    maximum_candidate_pairs: int = 1000000,
    maximum_memory_bytes: int = 268435456,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy
    | None = None,
    source_chart_cover: _dependency_phydrax_geometry__domain_boundary.MeshingDomainBoundarySource
    | None = None,
    source_atlas: _owner_phydrax_geometry__surface_source_support.SurfaceSourceRootAtlas
    | None = None,
    curve_witness: _owner_phydrax_meshing__surface_association_transfer.PreparedSurfaceCurveWitness
    | None = None,
    material_root_witness: surface_chart_records_owner.SurfaceChartWitness | None = None,
    target_chart_cover: _dependency_phydrax_geometry__domain_boundary.MeshingDomainBoundarySource
    | None = None,
    prepared_source_fidelity: source_fidelity_owner.SourceFidelityCertificate
    | None = None,
    prepared_source_certificates: tuple[
        _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate,
        _dependency_phydrax_geometry__mesh_certificates.GlobalEmbeddingCertificate,
    ]
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> SurfaceMetricGeometryTransition:
    """Accept only explicit old/target chart deformation with certificates."""
    match outcome:
        case SurfaceMetricOutcome(geometry=geometry):
            pass
        case SurfaceMetricGeometryWitness():
            geometry = outcome
        case _:
            raise TypeError(
                "surface_metric.outcome must contain source-bound edit and chart data; got an unsupported outcome type"
            )
    if (
        source_witness.geometry_id
        != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
            source_geometry
        )
        or source_witness.topology_id != source_mesh.topology_id
        or source_witness.domain_id != domain.domain_id
    ):
        raise ValueError(
            "surface_metric.source_chart_binding must match source geometry, topology and domain IDs; got a different source binding"
        )
    if policy.reconstruction != "bounded_chart_deformation":
        raise ValueError(
            "surface_metric.policy.reconstruction must select bounded_chart_deformation; got incompatible reconstruction"
        )
    reconstruction = reconstruct_surface_metric_geometry(
        source_mesh,
        source_geometry,
        target_mesh,
        outcome,
        domain,
        maximum_fidelity=maximum_fidelity,
        policy=policy,
        record_phase=record_phase,
        source_atlas=source_atlas,
        budget=budget,
    )
    target_mesh = reconstruction.target_mesh
    target_witness = surface_chart_records_owner.SurfaceChartWitness(
        geometry_id=_dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
            reconstruction.geometry
        ),
        topology_id=target_mesh.topology_id,
        domain_id=domain.domain_id,
        cell_global_ids=reconstruction.cell_global_ids,
        patches=reconstruction.cell_patches,
        charts=reconstruction.cell_charts,
        geometry_entity_ids=reconstruction.cell_geometry_entity_ids,
        occurrence_paths=reconstruction.cell_occurrence_paths,
    )
    source_rows = entity_keys_owner.key_rows(
        geometry.source_cell_ids[:, None],
        np.asarray(source_witness.cell_global_ids)[:, None],
    )
    edit_ids = np.concatenate([block.cell_ids for block in geometry.edit.blocks])
    target_rows = entity_keys_owner.key_rows(
        edit_ids[:, None], np.asarray(target_witness.cell_global_ids)[:, None]
    )
    if np.any(source_rows < 0) or np.any(target_rows < 0):
        raise ValueError(
            "surface_metric.prepare_surface_metric_geometry_transition must retain source geometry invariants; got material witnesses must retain every source/target scientific cell"
        )
    if material_root_witness is None:
        raise ValueError(
            "surface_metric.material_root_binding must retain the admitted root charts; got a missing root binding"
        )
    if curve_witness is not None and target_chart_cover is None:
        dimensions, indices, parameters = (
            geometry.vertex_source_dimensions,
            geometry.vertex_source_indices,
            geometry.vertex_source_parameters,
        )
        if (
            dimensions is None
            or indices is None
            or parameters is None
            or source_chart_cover is None
        ):
            raise ValueError(
                "surface_metric.prepare_surface_metric_geometry_transition must retain source-stratum parameters and incidence; got target curve provenance requires its retained strata and cover"
            )
        vertex_rows = entity_keys_owner.key_rows(
            geometry.edit.vertex_global_ids[:, None],
            np.asarray(target_mesh.vertex_global_ids)[:, None],
        )
        if np.any(vertex_rows < 0):
            raise ValueError(
                "surface_metric.prepare_surface_metric_geometry_transition must retain source-stratum parameters and incidence; got the retained target curve strata omit scientific vertices"
            )
        target_chart_cover = _owner_phydrax_meshing__surface_association_transfer.prepare_surface_curve_chart_cover(
            curve_witness,
            target_mesh,
            target_witness,
            dimensions[vertex_rows],
            indices[vertex_rows],
            parameters[vertex_rows],
            source_chart_cover=source_chart_cover,
            budget=budget,
        )
    with measure_phase(record_phase, "common_refinement"):
        deformation = _dependency_phydrax_discretization__surface_chart_deformation.prepare_surface_chart_deformation(
            source_mesh,
            source_geometry,
            target_mesh,
            reconstruction,
            domain,
            source_witness=source_witness,
            target_witness=target_witness,
            maximum_fidelity=maximum_fidelity,
            maximum_displacement=policy.reconstruction_tolerance,
            maximum_evaluations=policy.maximum_evaluations,
            maximum_candidate_pairs=maximum_candidate_pairs,
            maximum_memory_bytes=maximum_memory_bytes,
            maximum_measure_work=policy.maximum_evaluations,
            certificate_limits=certificate_limits,
            validity_policy=validity_policy,
            source_chart_cover=source_chart_cover,
            source_material_charts=geometry.source_material_charts[source_rows],
            target_material_charts=geometry.cell_material_charts[target_rows],
            material_root_witness=material_root_witness,
            target_chart_cover=target_chart_cover,
            prepared_source_fidelity=prepared_source_fidelity,
            prepared_source_restriction_id=None
            if prepared_source_fidelity is None
            or source_geometry.restriction_source is None
            else source_geometry.restriction_source.restriction_source_id,
            prepared_source_certificates=prepared_source_certificates,
            budget=budget,
            scope=scope,
        )
    with measure_phase(record_phase, "geometry_transition"):
        transition = _dependency_phydrax_discretization__cell_geometry_transfer.transition_chart_deformed_cell_geometry(
            source_mesh,
            source_geometry,
            target_mesh,
            reconstruction,
            deformation,
            policy=policy,
        )
    return SurfaceMetricGeometryTransition(transition, deformation, target_mesh)


@checked
def prepare_policy_surface_metric_transition(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    target_mesh: CellMesh,
    outcome: SurfaceMetricOutcome | SurfaceMetricGeometryWitness,
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    policy: _owner_phydrax_meshing__adaptation.MeshAdaptationPolicy,
    witness: surface_chart_records_owner.SurfaceChartWitness,
    /,
    *,
    maximum_fidelity: float,
    curve_witness: _owner_phydrax_meshing__surface_association_transfer.PreparedSurfaceCurveWitness
    | None = None,
    material_root_witness: surface_chart_records_owner.SurfaceChartWitness | None = None,
    prepared_source_fidelity: source_fidelity_owner.SourceFidelityCertificate
    | None = None,
    prepared_source_certificates: tuple[
        _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate,
        _dependency_phydrax_geometry__mesh_certificates.GlobalEmbeddingCertificate,
    ]
    | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> SurfaceMetricGeometryTransition:
    """Bind geometry continuation to the source policy and its certificate limits."""
    certification = source.certification
    if certification is None:
        raise ValueError(
            "surface_metric.source.certification must retain source limits; got None"
        )
    original = transfer.support.original
    match original:
        case _owner_phydrax_geometry__surface_source_support.SurfaceSourceRootAtlas():
            atlas = original
        case _:
            atlas = None
    return prepare_surface_metric_geometry_transition(
        source.mesh,
        source.geometry,
        target_mesh,
        outcome,
        transfer.support.domain,
        witness,
        maximum_fidelity=maximum_fidelity,
        policy=policy.geometry_transition,
        maximum_candidate_pairs=policy.limits.maximum_geometry_queries,
        maximum_memory_bytes=policy.limits.maximum_scratch_bytes,
        record_phase=record_phase,
        certificate_limits=certification.request.limits,
        validity_policy=policy.audit_policy.validity_policy,
        source_chart_cover=surface_metric_source_cover(source),
        source_atlas=atlas,
        curve_witness=curve_witness,
        material_root_witness=material_root_witness,
        prepared_source_fidelity=prepared_source_fidelity,
        prepared_source_certificates=prepared_source_certificates,
        budget=budget,
        scope=scope,
    )


__all__ = [
    "SurfaceMetricGeometryTransition",
    "SurfaceMetricOutcome",
    "execute_surface_metric_adaptation",
    "prepare_surface_metric_geometry_transition",
    "reconstruct_surface_metric_geometry",
    "SphereMetricOutcome",
    "execute_sphere_metric_adaptation",
    "build_sphere_periodic_metric_candidate",
]


# Physical sphere cells have their own nonsingular material references. They do
# not inherit the incomplete longitude/latitude triangles used by ordinary patches.


class SphereMetricOutcome(NamedTuple):
    """A private, fully certified physical-cell sphere successor."""

    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    target_mesh: CellMesh
    reconstruction: sphere_chart_records_owner.SphereGeometryReconstruction
    source_atlas: (
        _dependency_phydrax_geometry__sphere_material_atlas.SphereMaterialCellAtlas
    )
    target_atlas: (
        _dependency_phydrax_geometry__sphere_material_atlas.SphereMaterialCellAtlas
    )
    deformation: sphere_chart_records_owner.PreparedSphereChartDeformation
    metric: np.ndarray
    exact_stencils: Mapping[
        int, _dependency_phydrax_meshing__periodic.ConstructionPointKey
    ]
    evidence: _dependency_phydrax_meshing__tetra_metric.MetricRemeshingEvidence
    source_id: str
    source_revision: str
    domain_id: str


@dataclass(slots=True)
class _SphereMetricState:
    points: np.ndarray
    directions: np.ndarray
    metric: np.ndarray
    cells: np.ndarray
    patches: np.ndarray
    classes: np.ndarray
    cell_ids: np.ndarray
    vertex_ids: np.ndarray
    dimensions: list[int]
    indices: list[int]
    parameters: list[np.ndarray]
    parents: list[set[int]]
    vertex_sources: list[set[int]]
    stencils: list[_dependency_phydrax_meshing__periodic.ConstructionPointKey]
    next_cell_id: int
    next_vertex_id: int
    features: dict[tuple[int, int], int]
    blocked: set[tuple[int, int]]
    fixed: set[int]
    successors: dict[int, int]
    kinds: dict[int, int]
    relocated: set[int]
    maximum_fidelity: float
    minimum_metric_quality: float = 0.05


type _SpherePointValue = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    _dependency_phydrax_meshing__periodic.ConstructionPointKey,
]


class _SphereMetricBackground(NamedTuple):
    atlas: _dependency_phydrax_geometry__sphere_material_atlas.SphereMaterialCellAtlas
    vertex_ids: np.ndarray
    metric: np.ndarray
    corner_rows: np.ndarray
    matrices: tuple[tuple[tuple[Fraction, ...], ...], ...]
    normalization_norms: Mapping[int, float]
    vertex_memberships: Mapping[int, int]
    vertex_closure_memberships: Mapping[int, int]
    radial_bounds: dict[tuple[float, ...], tuple[Fraction, float, float, Fraction] | None]
    cone_orientations: np.ndarray
    cone_directions: np.ndarray
    cone_cofactors: dict[int, tuple[tuple[Fraction, ...], ...]]
    point_values: dict[tuple[int, int, int, bytes], _SpherePointValue]


def _sphere_edges(state: _SphereMetricState, /) -> tuple[np.ndarray, np.ndarray]:
    records = np.sort(state.cells[:, _LOCAL_EDGES].reshape((-1, 2)), axis=1)
    edges, first = np.unique(records, axis=0, return_index=True)
    return edges, first


def _sphere_charge(
    work: int,
    storage: int = 0,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    budget.reserve(work, storage)


def _sphere_location(
    background: _SphereMetricBackground,
    patch: int,
    direction: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__periodic.ConstructionPointKey:
    """Resolve positive homogeneous actions in the source physical material cells."""
    atlas = background.atlas
    budget.reserve(3, 4096)
    lifted = rational_vector(direction)
    point = (lifted[0], lifted[1], lifted[2])
    rows = np.flatnonzero(np.asarray(atlas.patches) == patch)
    for row in rows:
        # Cramer's numerator signs reject exterior cones exactly, before the
        # rational action constructs coefficients. Zero signs stay admitted:
        # shared boundaries retain the first source row's point key.
        with budget.temporary_scope():
            budget.reserve(0, 1024)
            corners = background.cone_directions[row]
            first = np.stack((direction, corners[0], corners[0]))
            second = np.stack((corners[1], direction, corners[1]))
            third = np.stack((corners[2], corners[2], direction))
            signs = exact_orient3d(np.zeros((3,), dtype=np.float64), first, second, third)

            if np.any(signs * background.cone_orientations[row] < 0):
                continue
        with budget.temporary_scope():
            if atlas.exact_determinants[row] == 0:
                raise ValueError(
                    "surface_metric.sphere_location must retain a nonsingular positive-definite source frame; got an admitted sphere cone became singular"
                )
            # The common Cramer determinant cancels in the point key. The source
            # cofactors are immutable; prepare them once per admitted source row.
            source_row = int(row)
            if source_row not in background.cone_cofactors:
                cofactors = _dependency_phydrax_geometry__sphere_material_atlas.sphere_cone_cofactors(
                    background.matrices[row], budget=budget
                )
                budget.retain_basis((source_row, cofactors))
                background.cone_cofactors[source_row] = cofactors
            cofactors = background.cone_cofactors[source_row]
            # Each nonzero action has three products and two sums. Exact zero
            # signs have already earned the zero numerator. Normalization adds
            # two sums and one division per nonzero key.
            _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
                2 + 6 * int(np.count_nonzero(signs)),
                49,
                0,
                72 * (1024 + 1075) + 6,
                budget=budget,
            )
            homogeneous = tuple(
                coefficients[0] * point[0]
                + coefficients[1] * point[1]
                + coefficients[2] * point[2]
                if sign
                else Fraction(0)
                for coefficients, sign in zip(cofactors, signs, strict=True)
            )
            denominator = homogeneous[0] + homogeneous[1] + homogeneous[2]
            orientation = 1 if atlas.exact_determinants[row] > 0 else -1
            if (orientation > 0 and denominator <= 0) or (
                orientation < 0 and denominator >= 0
            ):
                continue
            key = tuple(
                sorted(
                    (int(background.vertex_ids[vertex]), value / denominator)
                    for vertex, value in zip(
                        background.corner_rows[row], homogeneous, strict=True
                    )
                    if value
                )
            )
        budget.reserve(0, 65536)
        return key
    raise _leaf_phydrax_meshing__failures.MeshingFailure(
        _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
        "sphere_metric.construction must lie in the source physical material partition; got uncovered construction coordinates",
        stage=_dependency_phydrax_meshing__trace.MeshingStageKind.LINEAGE_CONSTRUCTION.value,
    )


def _sphere_background_metric(
    background: _SphereMetricBackground,
    key: _dependency_phydrax_meshing__periodic.ConstructionPointKey,
    /,
) -> np.ndarray:
    identifiers = np.asarray([identifier for identifier, _ in key], dtype=np.int64)
    rows = entity_keys_owner.key_rows(
        background.vertex_ids[:, None], identifiers[:, None]
    )
    if np.any(rows < 0):
        raise ValueError(
            "surface_metric.sphere_background_metric must retain the declared source and target bindings; got sphere metric support must name immutable vertex identities"
        )
    weights = np.asarray([float(weight) for _, weight in key], dtype=np.float64)
    result = np.asarray(
        _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
            background.metric[rows], weights
        ),
        dtype=np.float64,
    )
    properties = _dependency_phydrax_meshing__metric.tensor_properties(result[None])
    if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
        raise ValueError(
            "surface_metric.sphere_background_metric must retain a nonsingular positive-definite source frame; got sphere metric interpolation lost positive definiteness"
        )
    return result


def _sphere_point(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    background: _SphereMetricBackground,
    patch: int,
    dimension: int,
    index: int,
    parameters: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _SpherePointValue:
    budget.reserve(1 + dimension, 128 + 8 * dimension)
    # The request background binds source occurrences and the immutable metric.
    # Binary parameter bytes preserve signed zeros; unused padding is not a key.
    token = (patch, dimension, index, parameters[:dimension].tobytes())
    cached = background.point_values.get(token)
    if cached is not None:
        return cached
    _sphere_charge(64, budget=budget)
    _consume_native_work(geometry_queries=1, budget=budget)
    direction, _ = _dependency_phydrax_geometry__sphere_material_atlas.corner_direction(
        domain, patch, dimension, index, parameters
    )
    slot = int(
        np.asarray(background.atlas.source_slots)[
            np.flatnonzero(np.asarray(background.atlas.patches) == patch)[0]
        ]
    )
    atlas = background.atlas
    unit = np.asarray(atlas.axes)[slot] @ (direction / np.linalg.norm(direction))
    local = np.asarray(atlas.centers)[slot].copy()
    for radius in np.asarray(atlas.radius_terms)[slot]:
        local += radius * unit
    point = (
        np.asarray(atlas.rotations)[slot] @ local + np.asarray(atlas.translations)[slot]
    )
    key = _sphere_location(background, patch, direction, budget=budget)
    values = point, direction, _sphere_background_metric(background, key), key
    for value in values[:3]:
        value.setflags(write=False)
    budget.retain_basis((token, values))
    background.point_values[token] = values
    return values


def _sphere_radial_parameters(
    direction: np.ndarray,
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patch: int,
    /,
) -> np.ndarray:
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            "surface_metric.sphere_radial_parameters must retain finite representable bounds; got a sphere proposal has no finite positive radial direction"
        )
    unit = direction / norm
    uv = np.asarray((np.arctan2(unit[1], unit[0]), np.arcsin(unit[2])), dtype=np.float64)

    endpoints = np.concatenate(
        [
            _owner_phydrax_geometry__meshing_domain.use_endpoints(use)
            for loop in domain.patches[patch].loops
            for use in loop
        ]
    )
    box = np.stack((np.min(endpoints, axis=0), np.max(endpoints, axis=0)))
    period = 2 * np.pi
    if uv[0] < box[0, 0]:
        uv[0] += period
    elif uv[0] > box[1, 0]:
        uv[0] -= period
    domain.patches[patch].surface.validate_parameter_box(np.stack((uv, uv)))
    return uv


def _sphere_legal(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    cells: np.ndarray,
    patches: np.ndarray,
    directions: np.ndarray,
    /,
    *,
    points: np.ndarray,
    background: _SphereMetricBackground,
    maximum_fidelity: float,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        return False
    if np.unique(np.sort(cells, axis=1), axis=0).shape[0] != cells.shape[0]:
        return False
    for row in range(cells.shape[0]):
        cell, patch = cells[row, :], int(patches[row])
        with budget.temporary_scope():
            rays = directions[cell]
            key = tuple(
                float(rays[corner, axis]) for corner in range(3) for axis in range(3)
            )
            budget.reserve(9)
            if key not in background.radial_bounds:
                # The memo holds immutable cone enclosures, not rounded
                # source-cell identities or a renewed workspace allowance.
                budget.reserve(0, 4096)
                radial = _dependency_phydrax_geometry__sphere_material_atlas.sphere_radial_triangle_bounds(
                    rays, budget=budget
                )
                budget.retain_basis((key, radial))
                background.radial_bounds[key] = radial
            radial = background.radial_bounds[key]
            if radial is None:
                return False
            determinant, floor, ceiling, unit_error = radial
            sign = -1 if domain.patches[int(patch)].reversed else 1
            if sign * determinant <= 0:
                return False
            normalization = Fraction(background.normalization_norms[int(patch)]) * (
                max(1 - Fraction(floor), Fraction(ceiling) - 1, Fraction(0)) + unit_error
            )
            low, high = (
                _dependency_phydrax_geometry__sphere_material_atlas.source_image_enclosure(
                    domain.patches[int(patch)].surface, directions[cell], budget=budget
                )
            )
            error = max(
                sum(
                    (
                        max(
                            abs(
                                Fraction(float(points[cell[corner], axis]))
                                - Fraction(float(low[corner, axis]))
                            ),
                            abs(
                                Fraction(float(points[cell[corner], axis]))
                                - Fraction(float(high[corner, axis]))
                            ),
                        )
                        for axis in range(3)
                    ),
                    Fraction(0),
                )
                for corner in range(3)
            )
            # The final atlas adds only nonnegative polynomial-to-chord error.
            # This necessary gate never claims the full-map certificate;
            # reconstruction, partition and embedding still decide it.
            if (
                fraction_endpoint(normalization + error, upper=True, saturate=True)
                > maximum_fidelity
            ):
                return False
    return True


def _sphere_quality(
    state: _SphereMetricState,
    cells: np.ndarray,
    patches: np.ndarray,
    background: _SphereMetricBackground,
    /,
) -> np.ndarray:
    """Proposal quality uses the real source radial map differential."""
    tangents = np.zeros((cells.shape[0], 3, 3), dtype=np.float64)
    for row, (cell, patch) in enumerate(zip(cells, patches, strict=True)):
        atlas = background.atlas
        slot = int(
            np.asarray(atlas.source_slots)[
                np.flatnonzero(np.asarray(atlas.patches) == patch)[0]
            ]
        )
        rays = state.directions[cell]
        ray = np.mean(rays, axis=0)
        norm = np.linalg.norm(ray)
        velocity = (rays[1:] - rays[:1]).T
        derivative = velocity / norm - ray[:, None] * (ray @ velocity)[None, :] / norm**3
        physical = (
            np.asarray(atlas.rotations)[slot] @ np.asarray(atlas.axes)[slot] @ derivative
        )
        physical *= float(np.sum(np.asarray(atlas.radius_terms)[slot]))
        tangents[row, 1:] = physical.T
    mean = np.asarray(
        _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
            state.metric[cells], np.full(cells.shape, 1 / 3, dtype=np.float64)
        ),
        dtype=np.float64,
    )
    return np.asarray(
        _dependency_phydrax_meshing__metric.metric_shape_quality(
            jnp.asarray(mean), jnp.asarray(tangents), dimension=2
        ),
        dtype=np.float64,
    )


def _sphere_lengths(
    state: _SphereMetricState, /, *, edges: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if edges is None:
        edges, _ = _sphere_edges(state)
    vector = state.points[edges[:, 1]] - state.points[edges[:, 0]]
    mean = np.asarray(
        _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
            state.metric[edges], np.full(edges.shape, 0.5, dtype=np.float64)
        ),
        dtype=np.float64,
    )
    return edges, np.sqrt(
        np.sum(vector * np.sum(mean * vector[:, None, :], axis=-1), axis=-1)
    )


def _sphere_replace(
    state: _SphereMetricState,
    removed: np.ndarray,
    cells: np.ndarray,
    patches: np.ndarray,
    classes: np.ndarray,
    parents: list[set[int]],
    kind: _dependency_phydrax_meshing__lineage.EntityLineageKind,
    /,
    *,
    cell_ids: np.ndarray | None = None,
) -> None:
    keep = np.ones(state.cell_ids.size, dtype=np.bool_)
    keep[removed] = False
    identifiers = (
        _dependency_phydrax_meshing__tetra_metric.fresh_identifiers(
            state.next_cell_id, cells.shape[0]
        )
        if cell_ids is None
        else cell_ids
    )
    state.next_cell_id += cells.shape[0]
    for identifier in identifiers:
        state.kinds[int(identifier)] = int(kind)
    state.cells = np.concatenate((state.cells[keep], cells))
    state.patches = np.concatenate((state.patches[keep], patches))
    state.classes = np.concatenate((state.classes[keep], classes))
    state.cell_ids = np.concatenate((state.cell_ids[keep], identifiers))
    state.parents = [
        value for row, value in enumerate(state.parents) if keep[row]
    ] + parents


def _sphere_split(
    state: _SphereMetricState,
    edge: tuple[int, int],
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    background: _SphereMetricBackground,
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:

    rows = np.flatnonzero(np.sum(np.isin(state.cells, edge), axis=1) == 2)
    if (
        edge in state.blocked
        or rows.size != 2
        or np.unique(state.patches[rows]).size != 1
    ):
        return False
    membership = {
        background.vertex_memberships[parent]
        for vertex in edge
        for parent in state.vertex_sources[vertex]
    }
    if len(membership) != 1:
        return False
    patch = int(state.patches[rows[0]])
    curve = state.features.get(edge, -1)
    if curve >= 0:
        values = [
            _owner_phydrax_meshing__surface_association_transfer.curve_parameter(
                transfer.support,
                curve,
                state.dimensions[vertex],
                state.indices[vertex],
                state.parameters[vertex],
                None,
            )
            for vertex in edge
        ]
        dimension, index = 1, curve
        parameters = np.asarray((sum(values) / 2, np.nan), dtype=np.float64)
    elif curve == -2:
        return False
    else:
        dimension, index = 2, patch
        parameters = _sphere_radial_parameters(
            np.sum(state.directions[np.asarray(edge)], axis=0), domain, patch
        )
    point, direction, metric, stencil = _sphere_point(
        domain, background, patch, dimension, index, parameters, budget=budget
    )
    vertex = state.points.shape[0]
    children = []
    for row in rows:
        cell = state.cells[row]
        position = next(
            slot
            for slot in range(3)
            if {int(cell[slot]), int(cell[(slot + 1) % 3])} == set(edge)
        )
        a, b, c = (int(cell[(position + offset) % 3]) for offset in range(3))
        children.extend(((a, vertex, c), (vertex, b, c)))
    cells = np.asarray(children, dtype=np.int32)
    directions = np.concatenate((state.directions, direction[None]))
    candidate_points = np.concatenate((state.points, point[None]))
    if not _sphere_legal(
        domain,
        cells,
        np.repeat(state.patches[rows], 2),
        directions,
        points=candidate_points,
        background=background,
        maximum_fidelity=state.maximum_fidelity,
        budget=budget,
    ):
        return False
    _dependency_phydrax_meshing__tetra_metric.fresh_identifiers(state.next_vertex_id, 1)
    child_ids = _dependency_phydrax_meshing__tetra_metric.fresh_identifiers(
        state.next_cell_id, cells.shape[0]
    )
    state.points = candidate_points
    state.directions = directions
    state.metric = np.concatenate((state.metric, metric[None]))
    state.vertex_ids = np.concatenate(
        (state.vertex_ids, np.asarray((state.next_vertex_id,), dtype=np.int64))
    )
    state.next_vertex_id += 1
    state.dimensions.append(dimension)
    state.indices.append(index)
    state.parameters.append(parameters)
    state.vertex_sources.append(
        set().union(*(state.vertex_sources[value] for value in edge))
    )
    state.stencils.append(stencil)
    _sphere_replace(
        state,
        rows,
        cells,
        np.repeat(state.patches[rows], 2),
        np.repeat(state.classes[rows], 2),
        [state.parents[row].copy() for row in rows for _ in range(2)],
        _dependency_phydrax_meshing__lineage.EntityLineageKind.REFINED_FROM,
        cell_ids=child_ids,
    )
    if edge in state.features:
        identity = state.features.pop(edge)
        state.features[(min(edge[0], vertex), max(edge[0], vertex))] = identity
        state.features[(min(vertex, edge[1]), max(vertex, edge[1]))] = identity
    return True


def _sphere_collapse(
    state: _SphereMetricState,
    removed: int,
    kept: int,
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    background: _SphereMetricBackground,
    quality_floor: float,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    edge = (min(removed, kept), max(removed, kept))
    if removed in state.fixed or edge in state.blocked:
        return False
    removed_classes = {
        background.vertex_closure_memberships[parent]
        for parent in state.vertex_sources[removed]
    }
    kept_classes = {
        background.vertex_closure_memberships[parent]
        for parent in state.vertex_sources[kept]
    }
    if len(removed_classes) != 1 or removed_classes != kept_classes:
        return False
    if np.count_nonzero(np.sum(np.isin(state.cells, edge), axis=1) == 2) != 2:
        return False
    feature_edges = [value for value in state.features if removed in value]
    if feature_edges:
        curve = state.features.get(edge, -2)
        if (
            curve < 0
            or state.dimensions[removed] != 1
            or any(state.features[value] != curve for value in feature_edges)
        ):
            return False
        if state.dimensions[kept] == 1 and state.indices[kept] != curve:
            return False
    if _link(state.cells, (removed,)) & _link(state.cells, (kept,)) != _link(
        state.cells, edge
    ):
        return False
    rows = np.flatnonzero(np.any(state.cells == removed, axis=1))
    if (
        not rows.size
        or np.unique(state.patches[rows]).size != 1
        or np.unique(state.classes[rows]).size != 1
    ):
        return False
    candidates = state.cells[rows].copy()
    candidates[candidates == removed] = kept
    live = np.all(np.diff(np.sort(candidates, axis=1), axis=1) != 0, axis=1)
    cells, patches = candidates[live], state.patches[rows][live]
    untouched = np.delete(state.cells, rows, axis=0)
    # Untouched orientations were already admitted; only the new cavity can
    # introduce a duplicate against them.
    combined = np.concatenate((untouched, cells))
    if np.unique(np.sort(combined, axis=1), axis=0).shape[0] != combined.shape[0]:
        return False
    quality = _sphere_quality(state, cells, patches, background)
    if np.any(quality < quality_floor):
        return False
    local_edges = np.unique(
        np.sort(cells[:, _LOCAL_EDGES].reshape((-1, 2)), axis=1), axis=0
    )
    vectors = state.points[local_edges[:, 1]] - state.points[local_edges[:, 0]]
    means = np.asarray(
        _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
            state.metric[local_edges], np.full(local_edges.shape, 0.5, dtype=np.float64)
        ),
        dtype=np.float64,
    )
    if np.any(
        np.sqrt(np.sum(vectors * np.sum(means * vectors[:, None, :], axis=-1), axis=-1))
        > _UPPER
    ):
        return False
    if not _sphere_legal(
        domain,
        cells,
        patches,
        state.directions,
        points=state.points,
        background=background,
        maximum_fidelity=state.maximum_fidelity,
        budget=budget,
    ):
        return False
    cavity_parents = set().union(*(state.parents[row] for row in rows))
    parents = [cavity_parents.copy() for _ in range(cells.shape[0])]
    _sphere_replace(
        state,
        rows,
        cells,
        patches,
        state.classes[rows][live],
        parents,
        _dependency_phydrax_meshing__lineage.EntityLineageKind.COLLAPSED_INTO,
    )
    state.successors[removed] = kept
    state.vertex_sources[kept].update(state.vertex_sources[removed])
    for feature in feature_edges:
        identity = state.features.pop(feature)
        other = feature[0] if feature[1] == removed else feature[1]
        if other != kept:
            state.features[(min(kept, other), max(kept, other))] = identity
    return True


def _sphere_flip(
    state: _SphereMetricState,
    edge: tuple[int, int],
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    background: _SphereMetricBackground,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    if edge in state.features or edge in state.blocked:
        return False
    rows = np.flatnonzero(np.sum(np.isin(state.cells, edge), axis=1) == 2)
    if (
        rows.size != 2
        or np.unique(state.patches[rows]).size != 1
        or np.unique(state.classes[rows]).size != 1
    ):
        return False
    first = state.cells[rows[0]]
    position = next(
        slot
        for slot in range(3)
        if {int(first[slot]), int(first[(slot + 1) % 3])} == set(edge)
    )
    a, b, c = (int(first[(position + offset) % 3]) for offset in range(3))
    d = next(int(value) for value in state.cells[rows[1]] if value not in edge)
    if np.any(np.sum(np.isin(state.cells, (c, d)), axis=1) == 2):
        return False
    cells = np.asarray(((c, d, b), (d, c, a)), dtype=np.int32)
    patches = state.patches[rows]
    old = _sphere_quality(state, state.cells[rows], patches, background)
    new = _sphere_quality(state, cells, patches, background)
    if np.min(new) <= np.min(old):
        return False
    if not _sphere_legal(
        domain,
        cells,
        patches,
        state.directions,
        points=state.points,
        background=background,
        maximum_fidelity=state.maximum_fidelity,
        budget=budget,
    ):
        return False
    parents = set().union(*(state.parents[row] for row in rows))
    _sphere_replace(
        state,
        rows,
        cells,
        patches,
        state.classes[rows],
        [parents.copy(), parents.copy()],
        _dependency_phydrax_meshing__lineage.EntityLineageKind.SWAPPED_FROM,
    )
    return True


def _sphere_relocate(
    state: _SphereMetricState,
    vertex: int,
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    background: _SphereMetricBackground,
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    if vertex in state.fixed or state.dimensions[vertex] < 1:
        return False
    rows = np.flatnonzero(np.any(state.cells == vertex, axis=1))
    if (
        not rows.size
        or np.unique(state.patches[rows]).size != 1
        or np.unique(state.classes[rows]).size != 1
    ):
        return False
    patch = int(state.patches[rows[0]])
    incident = [edge for edge in state.features if vertex in edge]
    dimension, index = state.dimensions[vertex], state.indices[vertex]
    if dimension == 1:
        if len(incident) != 2 or any(
            edge in state.blocked or state.features[edge] != index for edge in incident
        ):
            return False
        neighbors = [edge[0] if edge[1] == vertex else edge[1] for edge in incident]
        values = [
            _owner_phydrax_meshing__surface_association_transfer.curve_parameter(
                transfer.support,
                index,
                state.dimensions[neighbor],
                state.indices[neighbor],
                state.parameters[neighbor],
                None,
            )
            for neighbor in neighbors
        ]
        parameters = np.asarray((sum(values) / 2, np.nan), dtype=np.float64)
    else:
        if incident:
            return False
        neighbors = np.unique(state.cells[rows])
        neighbors = neighbors[neighbors != vertex]
        parameters = _sphere_radial_parameters(
            np.mean(state.directions[neighbors], axis=0), domain, patch
        )
    point, direction, metric, stencil = _sphere_point(
        domain, background, patch, dimension, index, parameters, budget=budget
    )
    old_point, old_direction, old_metric = (
        state.points[vertex].copy(),
        state.directions[vertex].copy(),
        state.metric[vertex].copy(),
    )
    old_quality = _sphere_quality(
        state, state.cells[rows], state.patches[rows], background
    )
    neighbors = np.unique(state.cells[rows])
    neighbors = neighbors[neighbors != vertex]
    local_edges = np.unique(
        np.sort(
            np.column_stack(
                (np.full(neighbors.shape, vertex, dtype=np.int32), neighbors)
            ),
            axis=1,
        ),
        axis=0,
    )
    _, old_lengths = _sphere_lengths(state, edges=local_edges)
    state.points[vertex], state.directions[vertex], state.metric[vertex] = (
        point,
        direction,
        metric,
    )
    accepted = False
    try:
        new_quality = _sphere_quality(
            state, state.cells[rows], state.patches[rows], background
        )
        _, new_lengths = _sphere_lengths(state, edges=local_edges)
        # A proposal rejected by the existing numerical merit cannot become
        # admissible through a legality proof. Earn that proof only for a
        # nondegrading shape with a strictly improved edge-size defect.
        if np.min(new_quality) >= np.min(old_quality) and _unit_defect(
            new_lengths
        ) < _unit_defect(old_lengths):
            accepted = _sphere_legal(
                domain,
                state.cells[rows],
                state.patches[rows],
                state.directions,
                points=state.points,
                background=background,
                maximum_fidelity=state.maximum_fidelity,
                budget=budget,
            )
    finally:
        # Certification can refuse by raising after the trial coordinates have
        # been installed. Restore the numeric carrier on every noncommit exit;
        # the source ledgers still retain all attempted work and queries.
        if not accepted:
            state.points[vertex], state.directions[vertex], state.metric[vertex] = (
                old_point,
                old_direction,
                old_metric,
            )
    if not accepted:
        return False
    state.parameters[vertex] = parameters
    state.stencils[vertex] = stencil
    state.vertex_sources[vertex] = {parent for parent, _ in stencil}
    for identifier in state.cell_ids[rows]:
        state.kinds[int(identifier)] = int(
            _dependency_phydrax_meshing__lineage.EntityLineageKind.RELOCATED
        )
    state.relocated.add(vertex)
    return True


def _sphere_assemble(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    state: _SphereMetricState,
    /,
) -> tuple[
    _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Mapping[int, _dependency_phydrax_meshing__periodic.ConstructionPointKey],
]:
    live = np.unique(state.cells)
    row_of = np.full(state.points.shape[0], -1, dtype=np.int32)
    row_of[live] = np.arange(live.size, dtype=np.int32)
    owner_of = {
        int(identifier): row
        for row, block in enumerate(source.mesh.blocks)
        for identifier in np.asarray(block.global_ids)
    }
    owners = np.asarray(
        [owner_of[min(parents)] for parents in state.parents], dtype=np.int32
    )
    block_rows = tuple(
        np.flatnonzero(owners == row)[
            np.argsort(state.cell_ids[owners == row], kind="stable")
        ]
        for row in range(len(source.mesh.blocks))
    )
    blocks = _dependency_phydrax_meshing__topology_edit.source_family_blocks(
        tuple((block.name, block.cell_kind) for block in source.mesh.blocks),
        tuple(row_of[state.cells[rows]] for rows in block_rows),
        tuple(state.cell_ids[rows] for rows in block_rows),
    )
    stencils = {int(state.vertex_ids[row]): state.stencils[row] for row in live}
    width = max(map(len, stencils.values()))
    stencil_sources = np.zeros((live.size, width), dtype=np.int64)
    stencil_weights = np.zeros((live.size, width), dtype=np.float64)
    stencil_valid = np.zeros((live.size, width), dtype=np.bool_)
    for row, vertex in enumerate(live):
        for column, (identifier, weight) in enumerate(state.stencils[vertex]):
            (
                stencil_sources[row, column],
                stencil_weights[row, column],
                stencil_valid[row, column],
            ) = identifier, float(weight), True
    source_count = source.mesh.coordinates.shape[0]
    collapsed = _dependency_phydrax_meshing__tetra_metric.resolved_collapses(
        state.vertex_ids, source_count, state.successors
    )
    vertex_pairs = [
        (
            parent,
            int(state.vertex_ids[vertex]),
            int(
                _dependency_phydrax_meshing__lineage.EntityLineageKind.RELOCATED
                if vertex in state.relocated
                else _dependency_phydrax_meshing__lineage.EntityLineageKind.REFINED_FROM
            ),
        )
        for vertex in live
        if vertex >= source_count
        for parent in sorted(state.vertex_sources[vertex])
    ]
    vertex_pairs.extend(
        (
            int(state.vertex_ids[vertex]),
            int(state.vertex_ids[vertex]),
            int(_dependency_phydrax_meshing__lineage.EntityLineageKind.RELOCATED),
        )
        for vertex in live
        if vertex < source_count and vertex in state.relocated
    )
    vertex_pairs.extend(
        (
            parent,
            target,
            int(_dependency_phydrax_meshing__lineage.EntityLineageKind.COLLAPSED_INTO),
        )
        for parent, target in sorted(collapsed.items())
    )
    cell_pairs = [
        (
            parent,
            int(identifier),
            state.kinds.get(
                int(identifier),
                int(_dependency_phydrax_meshing__lineage.EntityLineageKind.PRESERVED),
            ),
        )
        for parents, identifier in zip(state.parents, state.cell_ids, strict=True)
        for parent in sorted(parents)
        if parent != identifier or int(identifier) in state.kinds
    ]
    relations = []
    for dimension, pairs in ((0, vertex_pairs), (2, cell_pairs)):
        values = np.asarray(pairs, dtype=np.int64).reshape((-1, 3))
        relations.append(
            _dependency_phydrax_meshing__topology_edit.EntityRelations(
                dimension, values[:, :1], values[:, 1:2], values[:, 2].astype(np.int8)
            )
        )
    edges = _dependency_phydrax_meshing__tetra_metric.subentity_relations(
        source.mesh,
        1,
        state.cells,
        state.vertex_ids,
        state.vertex_sources,
        collapsed_vertices=collapsed,
    )
    edit = _dependency_phydrax_meshing__topology_edit.CellTopologyEdit(
        "local_reconnection",
        state.points[live],
        state.vertex_ids[live],
        blocks,
        stencil_sources,
        stencil_weights,
        stencil_valid,
        (relations[0], edges, relations[1]),
    )
    return (
        edit,
        state.metric[live],
        state.patches[np.concatenate(block_rows)],
        live,
        MappingProxyType(stencils),
    )


def _sphere_expression_action(
    matrix: tuple[tuple[Fraction, ...], ...],
    values: tuple[
        _dependency_phydrax_discretization__coordinate_enclosure.Expression, ...
    ],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_dependency_phydrax_discretization__coordinate_enclosure.Expression, ...]:
    """Solve source coefficient columns, retaining nonremovable quotients."""

    parts = tuple(
        _owner_phydrax_discretization__coordinate_enclosure.expression_parts(
            value, 1, budget=budget
        )
        for value in values
    )
    denominator = _owner_phydrax_discretization__coordinate_enclosure.constant(
        1, 1, budget=budget
    )
    for _, divisor in parts:
        denominator = _owner_phydrax_discretization__coordinate_enclosure.multiply(
            denominator, divisor, budget=budget
        )
    numerators = []
    for row, (numerator, _) in enumerate(parts):
        for other, (_, divisor) in enumerate(parts):
            if row != other:
                numerator = _owner_phydrax_discretization__coordinate_enclosure.multiply(
                    numerator, divisor, budget=budget
                )
        numerators.append(numerator)
    indices = sorted(set().union(*(value.keys() for value in numerators)))
    if not indices:
        return ({}, {}, {})
    right = tuple(
        tuple(
            rational_coefficient(
                value.get(index, Fraction(0)),
                role="surface_metric.sphere_expression_action",
            )
            for index in indices
        )
        for value in numerators
    )
    prepared = (
        _dependency_phydrax_linalg__small_batched.prepare_exact_small_linear_actions(
            matrix, right, budget=budget
        )
    )
    if prepared.actions is None:
        raise ValueError(
            "surface_metric.sphere_expression_action must retain a nonsingular positive-definite source frame; got sphere physical/source coefficient frame is singular"
        )
    return tuple(
        _owner_phydrax_discretization__coordinate_enclosure.rational_expression(
            {
                index: coefficient
                for index, coefficient in zip(indices, row, strict=True)
                if coefficient
            },
            denominator,
            budget=budget,
        )
        for row in prepared.actions
    )


def _sphere_original_cones(
    background: _SphereMetricBackground,
    patch: int,
    edge: tuple[_dependency_phydrax_discretization__coordinate_enclosure.Expression, ...],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    tuple[
        int,
        tuple[_dependency_phydrax_discretization__coordinate_enclosure.Expression, ...],
    ],
    ...,
]:

    atlas = background.atlas
    rows = np.flatnonzero(np.asarray(atlas.patches) == patch)
    slot = int(np.asarray(atlas.source_slots)[rows[0]])
    rotation = tuple(rational_vector(row) for row in np.asarray(atlas.rotations)[slot])
    axes = tuple(rational_vector(row) for row in np.asarray(atlas.axes)[slot])
    center = rational_vector(np.asarray(atlas.centers)[slot])
    translation = rational_vector(np.asarray(atlas.translations)[slot])
    shift = tuple(
        sum((rotation[i][j] * center[j] for j in range(3)), translation[i])
        for i in range(3)
    )
    physical = tuple(
        _owner_phydrax_discretization__coordinate_enclosure.expression_add(
            value,
            _owner_phydrax_discretization__coordinate_enclosure.constant(
                -offset, 1, budget=budget
            ),
            budget=budget,
        )
        for value, offset in zip(edge, shift, strict=True)
    )
    results = []
    for row in rows:
        directions = background.matrices[row]
        matrix = tuple(
            tuple(
                sum(
                    (
                        rotation[i][k] * axes[k][l] * directions[l][j]
                        for k in range(3)
                        for l in range(3)
                    ),
                    Fraction(0),
                )
                for j in range(3)
            )
            for i in range(3)
        )
        action = _sphere_expression_action(matrix, physical, budget=budget)
        results.append((int(row), action))
    return tuple(results)


def _sphere_interval_controls(
    value: _dependency_phydrax_discretization__coordinate_enclosure.Expression,
    lower: Fraction,
    upper: Fraction,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[Fraction, Fraction]:

    argument = _owner_phydrax_discretization__coordinate_enclosure.add(
        _owner_phydrax_discretization__coordinate_enclosure.constant(
            lower, 1, budget=budget
        ),
        _owner_phydrax_discretization__coordinate_enclosure.scale(
            _owner_phydrax_discretization__coordinate_enclosure.axes(1, budget=budget)[0],
            upper - lower,
            budget=budget,
        ),
        budget=budget,
    )
    controls = _owner_phydrax_discretization__coordinate_enclosure.expression_bernstein_coefficients(
        _owner_phydrax_discretization__coordinate_enclosure.expression_compose(
            value, (argument,), budget=budget
        ),
        "box",
        1,
        budget=budget,
    )
    rational_controls = tuple(
        rational_coefficient(control, role="surface_metric.sphere_interval_controls")
        for control in controls
    )
    return min(rational_controls), max(rational_controls)


def _sphere_interval_metric(
    background: _SphereMetricBackground,
    row: int,
    ranges: tuple[tuple[Fraction, Fraction], ...],
    logarithms: tuple[HermitianFunctionEnclosure, ...],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[Fraction, ...], ...], Fraction]:
    vertices = background.corner_rows[row]
    logs = tuple(logarithms[vertex] for vertex in vertices)
    centers = tuple((low + high) / 2 for low, high in ranges)
    radii = tuple((high - low) / 2 for low, high in ranges)
    nominal = tuple(
        tuple(
            sum(
                (
                    weight * log.matrix[i][j]
                    for weight, log in zip(centers, logs, strict=True)
                ),
                Fraction(0),
            )
            for j in range(3)
        )
        for i in range(3)
    )
    error = sum(
        (
            radius
            * _dependency_phydrax_linalg__hermitian_spectral.matrix_norm_enclosure(
                log.matrix, budget=budget
            )
            + (abs(center) + radius) * log.error
            for center, radius, log in zip(centers, radii, logs, strict=True)
        ),
        Fraction(0),
    )
    return hermitian_exp_enclosure(
        HermitianFunctionEnclosure(nominal, error), budget=budget
    )


def _sphere_quadratic_interval(
    tangents: tuple[tuple[Fraction, Fraction], ...],
    metric: tuple[tuple[Fraction, ...], ...],
    error: Fraction,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[Fraction, Fraction]:
    low = high = Fraction(0)
    for i in range(3):
        for j in range(3):
            if i == j:
                a, b = tangents[i]
                products = (
                    Fraction(0) if a <= 0 <= b else min(a * a, b * b),
                    max(a * a, b * b),
                )
            else:
                products = tuple(a * b for a in tangents[i] for b in tangents[j])
            values = tuple(metric[i][j] * value for value in products)
            low += min(values)
            high += max(values)
    magnitude = sum((max(abs(a), abs(b)) ** 2 for a, b in tangents), Fraction(0))
    return (
        _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
            max(Fraction(0), low - error * magnitude), budget=budget
        )[0],
        _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
            max(Fraction(0), high + error * magnitude), budget=budget
        )[1],
    )


def _sphere_affine_constant_metric_arcs(
    points: np.ndarray,
    edges: np.ndarray,
    metric: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Certify straight P1 carrier edges without rebuilding expression programs."""
    bounds = np.empty((edges.shape[0], 2), dtype=np.float64)
    with budget.temporary_scope():
        budget.reserve(9, 4096)
        matrix = tuple(rational_vector(row) for row in metric)
        entries = tuple(
            (i, j, matrix[i][j]) for i in range(3) for j in range(3) if matrix[i][j]
        )
        for row, (first, second) in enumerate(edges):
            with budget.temporary_scope():
                _sphere_charge(3 + 3 * len(entries), 16384, budget=budget)
                delta = tuple(
                    Fraction(float(points[int(second), axis]))
                    - Fraction(float(points[int(first), axis]))
                    for axis in range(3)
                )
                squared = sum(
                    (value * delta[i] * delta[j] for i, j, value in entries),
                    Fraction(0),
                )
                if squared <= 0:
                    raise ValueError(
                        "surface_metric.sphere_affine_constant_metric_arcs must retain source geometry invariants; got an affine sphere carrier edge has no positive constant metric length"
                    )
                lower, upper = (
                    _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
                        squared, budget=budget
                    )
                )
                bounds[row] = (
                    fraction_endpoint(lower, upper=False, saturate=True),
                    fraction_endpoint(upper, upper=True, saturate=True),
                )
    return bounds


def _sphere_affine_constant_metric_quality(
    mesh: CellMesh,
    reconstruction: sphere_chart_records_owner.SphereGeometryReconstruction,
    metric: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Certify constant-Gram quality of affine successor cells."""
    points = np.asarray(reconstruction.vertex_coordinates, dtype=np.float64)
    if points.shape != mesh.coordinates.shape or not np.array_equal(
        points, np.asarray(mesh.coordinates)
    ):
        raise ValueError(
            "surface_metric.sphere_affine_constant_metric_quality must the bound scientific vertex coordinates; got affine sphere quality requires the bound scientific vertex coordinates"
        )
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    result = np.empty((cells.shape[0],), dtype=np.float64)
    with budget.temporary_scope():
        budget.reserve(9, 4096)
        matrix = tuple(rational_vector(row) for row in metric)
        entries = tuple(
            (i, j, matrix[i][j]) for i in range(3) for j in range(3) if matrix[i][j]
        )
        for row, cell in enumerate(cells):
            with budget.temporary_scope():
                _sphere_charge(11 + 12 * len(entries), 32768, budget=budget)
                origin = tuple(
                    Fraction(float(points[int(cell[0]), axis])) for axis in range(3)
                )
                tangents = tuple(
                    tuple(
                        Fraction(float(points[int(cell[corner]), axis])) - origin[axis]
                        for axis in range(3)
                    )
                    for corner in (1, 2)
                )
                gram = tuple(
                    tuple(
                        sum(
                            (
                                value * tangents[a][i] * tangents[b][j]
                                for i, j, value in entries
                            ),
                            Fraction(0),
                        )
                        for b in range(2)
                    )
                    for a in range(2)
                )
                determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
                denominator = gram[0][0] + gram[1][1] - gram[0][1]
                if determinant <= 0 or denominator <= 0:
                    result[row] = 0.0
                    continue
                lower, _ = (
                    _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
                        3 * determinant, budget=budget
                    )
                )
                result[row] = max(
                    0.0,
                    fraction_endpoint(lower / denominator, upper=False, saturate=True),
                )
    return result


type _SphereCoordinateExpressions = tuple[
    _dependency_phydrax_discretization__coordinate_enclosure.Expression, ...
]


def _sphere_constant_metric_arc_bounds(
    velocity: _SphereCoordinateExpressions,
    tensor: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[float, float]:
    """Integrate a full-map constant-metric Gram expression in its reduction order."""
    metric = tuple(rational_vector(line) for line in tensor)
    gram = _owner_phydrax_discretization__coordinate_enclosure.expression_sum(
        tuple(
            (
                _owner_phydrax_discretization__coordinate_enclosure.expression_scale(
                    _owner_phydrax_discretization__coordinate_enclosure.expression_multiply(
                        velocity[i], velocity[j], budget=budget
                    ),
                    metric[i][j],
                    budget=budget,
                )
                for i in range(3)
                for j in range(3)
            )
        ),
        budget=budget,
    )
    enclosure = mapped_measure_owner.PreparedMappedEdgeArcLength(gram).integrate(
        maximum_work=budget.maximum_work_units - budget.work_units,
        maximum_subcells=10000,
        maximum_binomial_terms=32,
        budget=budget,
    )
    return enclosure.lower, enclosure.upper


def _sphere_metric_edge_expressions(
    coordinates: _SphereCoordinateExpressions,
    start: tuple[Fraction, Fraction],
    end: tuple[Fraction, Fraction],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_SphereCoordinateExpressions, _SphereCoordinateExpressions]:
    """Restrict a full coordinate map and its velocity to one reference edge."""
    variable = _owner_phydrax_discretization__coordinate_enclosure.axes(1, budget=budget)[
        0
    ]
    arguments = tuple(
        _owner_phydrax_discretization__coordinate_enclosure.add(
            _owner_phydrax_discretization__coordinate_enclosure.constant(
                a, 1, budget=budget
            ),
            _owner_phydrax_discretization__coordinate_enclosure.scale(
                variable, b - a, budget=budget
            ),
            budget=budget,
        )
        for a, b in zip(start, end, strict=True)
    )
    edge = tuple(
        _owner_phydrax_discretization__coordinate_enclosure.expression_compose(
            value, arguments, budget=budget
        )
        for value in coordinates
    )
    velocity = tuple(
        _owner_phydrax_discretization__coordinate_enclosure.expression_derivative(
            value, 0, budget=budget
        )
        for value in edge
    )
    return edge, velocity


def _sphere_metric_cone_ranges(
    homogeneous: tuple[
        _dependency_phydrax_discretization__coordinate_enclosure.Expression, ...
    ],
    lower: Fraction,
    upper: Fraction,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[tuple[Fraction, Fraction], ...] | None:
    """Validate the source quotient before enclosing a cone's barycentric support."""
    numerator_parts = tuple(
        _owner_phydrax_discretization__coordinate_enclosure.expression_parts(
            value, 1, budget=budget
        )
        for value in homogeneous
    )
    # All rational actions share one denominator; retain its sign for cone decisions.
    divisor_low, divisor_high = _sphere_interval_controls(
        numerator_parts[0][1], lower, upper, budget=budget
    )
    if divisor_low <= 0 <= divisor_high:
        raise ValueError(
            "surface_metric.sphere_certify_metric_arcs must retain certified source support and denominator bounds; got an sphere edge quotient has an unresolved source denominator"
        )
    coefficients = tuple(
        _sphere_interval_controls(value, lower, upper, budget=budget)
        for value in homogeneous
    )
    if any(high < 0 for _, high in coefficients):
        return None
    denominator = _owner_phydrax_discretization__coordinate_enclosure.expression_sum(
        homogeneous, budget=budget
    )
    den_low, den_high = _sphere_interval_controls(
        denominator, lower, upper, budget=budget
    )
    if den_high <= 0:
        return None
    ranges = []
    for value, (a, b) in zip(homogeneous, coefficients, strict=True):
        if den_low > 0:
            numerator, divisor = (
                _owner_phydrax_discretization__coordinate_enclosure.expression_parts(
                    value, 1, budget=budget
                )
            )
            total_numerator, total_divisor = (
                _owner_phydrax_discretization__coordinate_enclosure.expression_parts(
                    denominator, 1, budget=budget
                )
            )
            quotient = (
                _owner_phydrax_discretization__coordinate_enclosure.rational_expression(
                    _owner_phydrax_discretization__coordinate_enclosure.multiply(
                        numerator, total_divisor, budget=budget
                    ),
                    _owner_phydrax_discretization__coordinate_enclosure.multiply(
                        divisor, total_numerator, budget=budget
                    ),
                    budget=budget,
                )
            )
            a, b = _sphere_interval_controls(quotient, lower, upper, budget=budget)
            ranges.append((max(Fraction(0), a), min(Fraction(1), b)))
        else:
            ranges.append((Fraction(0), Fraction(1)))
    if any(a > b for a, b in ranges):
        return None
    return tuple(ranges)


def _sphere_variable_metric_arc_bounds(
    background: _SphereMetricBackground,
    patch: int,
    edge: tuple[_dependency_phydrax_discretization__coordinate_enclosure.Expression, ...],
    velocity: tuple[
        _dependency_phydrax_discretization__coordinate_enclosure.Expression, ...
    ],
    logarithms: tuple[HermitianFunctionEnclosure, ...],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[float, float]:
    """Integrate all intersecting source cones with the fixed LIFO subdivision proof."""
    cones = _sphere_original_cones(background, patch, edge, budget=budget)
    pending = [(Fraction(0), Fraction(1))]
    integral_low = integral_high = Fraction(0)
    while pending:
        lower, upper = pending.pop()
        with budget.temporary_scope():
            _sphere_charge(64, budget=budget)
            tangent_bounds = tuple(
                _sphere_interval_controls(value, lower, upper, budget=budget)
                for value in velocity
            )
            candidates = []
            for source_row, homogeneous in cones:
                ranges = _sphere_metric_cone_ranges(
                    homogeneous, lower, upper, budget=budget
                )
                if ranges is None:
                    continue
                metric, error = _sphere_interval_metric(
                    background, source_row, ranges, logarithms, budget=budget
                )
                candidates.append(
                    _sphere_quadratic_interval(
                        tangent_bounds, metric, error, budget=budget
                    )
                )
            if not candidates:
                raise ValueError(
                    "surface_metric.sphere_certify_metric_arcs must retain certified source support and denominator bounds; got sphere metric arc has no radial background coverage"
                )
            low, high = (
                min(value[0] for value in candidates),
                max(value[1] for value in candidates),
            )
            if high - low <= Fraction(1, 1024):
                integral_low += (upper - lower) * low
                integral_high += (upper - lower) * high
            else:
                midpoint = (lower + upper) / 2
                pending.extend(((lower, midpoint), (midpoint, upper)))
    return (
        fraction_endpoint(integral_low, upper=False, saturate=True),
        fraction_endpoint(integral_high, upper=True, saturate=True),
    )


# Edge preparation, source-support proof and integration retain their ledger scopes.
def _sphere_certify_metric_arcs(
    mesh: CellMesh,
    reconstruction: sphere_chart_records_owner.SphereGeometryReconstruction,
    background: _SphereMetricBackground,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Enclose successor full-map arcs in the source SPD background.

    Intervals crossing source cone boundaries use all intersecting source cells.
    """
    # Cone, subdivision, reduction and charge order remain fixed within each proof.
    geometry = reconstruction.geometry
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    records = np.sort(cells[:, _LOCAL_EDGES].reshape((-1, 2)), axis=1)
    edges, first = np.unique(records, axis=0, return_index=True)
    owner, local_edge = first // 3, first % 3
    edge_ids = np.sort(np.asarray(mesh.vertex_global_ids, dtype=np.int64)[edges], axis=1)
    bounds = np.empty((edges.shape[0], 2), dtype=np.float64)
    constant_metric = np.array_equal(
        background.metric, np.broadcast_to(background.metric[0], background.metric.shape)
    )
    if (
        constant_metric
        and _dependency_phydrax_discretization__cell_geometry_transfer.nested_geometry_degree(
            mesh, geometry
        )
        == 1
    ):
        points = np.asarray(reconstruction.vertex_coordinates, dtype=np.float64)
        if points.shape != mesh.coordinates.shape or not np.array_equal(
            points, np.asarray(mesh.coordinates)
        ):
            raise ValueError(
                "surface_metric.sphere_certify_metric_arcs must retain certified source support and denominator bounds; got affine sphere arcs require the bound scientific vertex coordinates"
            )
        bounds = _sphere_affine_constant_metric_arcs(
            points, edges, background.metric[0], budget=budget
        )
        binding = canonical_fingerprint(
            {
                "kind": "sphere-original-background-full-coordinate-metric-arcs",
                "source_atlas": background.atlas.atlas_id,
                "target_atlas": reconstruction.target_atlas.atlas_id,
                "target_geometry": reconstruction.target_geometry_id,
                "background": array_tree_fingerprint(
                    (background.vertex_ids, background.metric)
                ),
                "edges": array_tree_fingerprint(edge_ids),
                "bounds": array_tree_fingerprint(bounds),
            }
        )
        return bounds, edge_ids, binding
    cached_logs: list[HermitianFunctionEnclosure] = []
    if not constant_metric:
        for value in background.metric:
            with budget.temporary_scope():
                logarithm = hermitian_log_enclosure(value, budget=budget)
            budget.retain_basis((logarithm.matrix, logarithm.error))
            cached_logs.append(logarithm)
    logarithms = tuple(cached_logs)
    with budget.activate():
        bank = _dependency_phydrax_discretization__coordinate_enclosure.prepared_coordinate_source_bank(
            geometry, budget=budget
        )
        cell_expressions = []
        for element, route in zip(elements, routes, strict=True):
            for nodes in np.asarray(route, dtype=np.int64):
                with budget.temporary_scope():
                    expressions = _owner_phydrax_discretization__coordinate_enclosure.coordinate_expressions(
                        element, tuple((bank[int(node)] for node in nodes)), budget=budget
                    )
                    if expressions is None:
                        raise ValueError(
                            "surface_metric.sphere_certify_metric_arcs must retain certified source support and denominator bounds; got sphere metric arcs require full coordinate expressions"
                        )
                cell_expressions.append(expressions)
        reference = (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        )
        for row, (cell, local) in enumerate(zip(owner, local_edge, strict=True)):
            with budget.temporary_scope():
                start, end = (reference[index] for index in _LOCAL_EDGES[local])
                edge, velocity = _sphere_metric_edge_expressions(
                    cell_expressions[cell], start, end, budget=budget
                )
                if constant_metric:
                    bounds[row] = _sphere_constant_metric_arc_bounds(
                        velocity, background.metric[0], budget=budget
                    )
                    continue
                bounds[row] = _sphere_variable_metric_arc_bounds(
                    background,
                    int(np.asarray(reconstruction.cell_patches)[cell]),
                    edge,
                    velocity,
                    logarithms,
                    budget=budget,
                )
    binding = canonical_fingerprint(
        {
            "kind": "sphere-original-background-full-coordinate-metric-arcs",
            "source_atlas": background.atlas.atlas_id,
            "target_atlas": reconstruction.target_atlas.atlas_id,
            "target_geometry": reconstruction.target_geometry_id,
            "background": array_tree_fingerprint(
                (background.vertex_ids, background.metric)
            ),
            "edges": array_tree_fingerprint(edge_ids),
            "bounds": array_tree_fingerprint(bounds),
        }
    )
    return bounds, edge_ids, binding


def _sphere_initial_state(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    metric: np.ndarray,
    atlas: _dependency_phydrax_geometry__sphere_material_atlas.SphereMaterialCellAtlas,
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    policy: _owner_phydrax_meshing__adaptation.MeshAdaptationPolicy,
    maximum_fidelity: float,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_SphereMetricState, _SphereMetricBackground]:

    mesh, domain = source.mesh, atlas.domain
    vertex, _ = transfer.source_associations(source)
    classes = transfer.classes(source)
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    vertex_rows = vertex.target_rows(ids)
    dimensions = np.asarray(vertex.source_dimensions)[vertex_rows].astype(np.int64)
    indices = np.asarray(classes[0].indices, dtype=np.int64)
    parameters = np.asarray(vertex.parameters, dtype=np.float64)[vertex_rows]
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks]
    )
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    physical = np.asarray(atlas.physical_rows, dtype=np.int64)
    patches = np.empty(cell_ids.size, dtype=np.int32)
    patches[physical] = np.asarray(atlas.patches)
    directions = np.empty((ids.size, 3), dtype=np.float64)
    physical_directions = np.empty((cell_ids.size, 3, 3), dtype=np.float64)
    physical_directions[physical] = np.asarray(atlas.directions)
    for cell, rays in zip(cells, physical_directions, strict=True):
        directions[cell] = rays
    organization = _leaf_phydrax_meshing__host_banks.organization_scopes(source)
    vertex_memberships = _leaf_phydrax_meshing__host_banks.row_classes(
        _leaf_phydrax_meshing__host_banks.membership(mesh, organization, 0)
    )
    closures = np.zeros((ids.size, len(organization)), dtype=np.bool_)
    for column, scope in enumerate(organization):
        rows = _leaf_phydrax_meshing__host_banks.selected_rows(mesh, scope)
        closures[
            _leaf_phydrax_meshing__host_banks.closure_rows(
                mesh, scope.entity_dimension, rows, 0
            ),
            column,
        ] = True
    vertex_closures = _leaf_phydrax_meshing__host_banks.row_classes(closures)
    cell_classes = _leaf_phydrax_meshing__host_banks.mesh_cell_classes(mesh, organization)
    cell_rows = entity_keys_owner.key_rows(
        entity_keys_owner.entity_keys(mesh, 2), cell_ids[:, None]
    )
    cell_classes = _leaf_phydrax_meshing__host_banks.row_classes(
        np.column_stack((cell_classes[cell_rows], classes[2].codes[cell_rows]))
    )
    if source.attributes:
        cell_classes = _leaf_phydrax_meshing__host_banks.row_classes(
            np.column_stack(
                (
                    cell_classes,
                    _owner_phydrax_meshing__layer_core.layer_interval_classes(source),
                )
            )
        )
    entity_cell_rows = entity_keys_owner.key_rows(
        cell_ids[:, None], entity_keys_owner.entity_keys(mesh, 2)
    )
    edge_classes = _leaf_phydrax_meshing__host_banks.edge_classes(
        mesh, organization, cell_classes[entity_cell_rows]
    )
    fixed = (dimensions == 0) | np.any(
        _leaf_phydrax_meshing__host_banks.membership(mesh, organization, 0), axis=1
    )
    protected = (
        transfer.protected_edges(source, midpoint_required=False) | ~classes[1].resolved
    )
    for scope in policy.protected_scopes:
        rows = _leaf_phydrax_meshing__host_banks.selected_rows(mesh, scope)
        fixed[
            _leaf_phydrax_meshing__host_banks.closure_rows(
                mesh, scope.entity_dimension, rows, 0
            )
        ] = True
        if scope.entity_dimension >= 1:
            protected[
                _leaf_phydrax_meshing__host_banks.closure_rows(
                    mesh, scope.entity_dimension, rows, 1
                )
            ] = True
    edge_vertices = entity_keys_owner.key_rows(
        ids[:, None],
        entity_keys_owner.entity_keys(mesh, 1).reshape((-1, 1)),
    ).reshape((-1, 2))
    blocked = {
        (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1])))
        for edge in edge_vertices[protected]
    }
    # No operation may rewrite an incident protected edge, even when the
    # selected operation is another edge of the same local cavity.
    fixed[np.unique(edge_vertices[protected])] = True
    features = {
        (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1]))): int(index)
        if dimension == 1
        else -2
        for edge, dimension, index in zip(
            edge_vertices, classes[1].dimensions, classes[1].indices, strict=True
        )
        if dimension < 2
    }
    for edge, classification in zip(edge_vertices, edge_classes, strict=True):
        if classification:
            features.setdefault(
                (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1]))), -2
            )
    state = _SphereMetricState(
        np.asarray(mesh.coordinates, dtype=np.float64).copy(),
        directions,
        metric.copy(),
        cells.copy(),
        patches,
        cell_classes,
        cell_ids.copy(),
        ids.copy(),
        dimensions.tolist(),
        indices.tolist(),
        [value.copy() for value in parameters],
        [{int(value)} for value in cell_ids],
        [{int(value)} for value in ids],
        [((int(value), Fraction(1)),) for value in ids],
        int(np.max(cell_ids)) + 1,
        int(np.max(ids)) + 1,
        features,
        blocked,
        set(map(int, np.flatnonzero(fixed))),
        {},
        {},
        set(),
        maximum_fidelity,
    )
    corners = entity_keys_owner.key_rows(
        ids[:, None], np.asarray(atlas.physical_corner_global_ids).reshape((-1, 1))
    ).reshape((-1, 3))
    matrices = tuple(
        tuple(rational_vector(line) for line in rays.T)
        for rays in np.asarray(atlas.directions)
    )
    immutable = metric.copy()
    immutable.setflags(write=False)
    norms = {
        int(
            patch
        ): _dependency_phydrax_geometry__sphere_material_atlas.sphere_radial_source_norm(
            domain.patches[int(patch)].surface, budget=budget
        )
        for patch in np.unique(patches)
    }
    rays = np.asarray(atlas.directions)
    cone_orientations = exact_orient3d(
        np.zeros((3,), dtype=np.float64), rays[:, 0], rays[:, 1], rays[:, 2]
    )

    return state, _SphereMetricBackground(
        atlas,
        ids.copy(),
        immutable,
        corners,
        matrices,
        MappingProxyType(norms),
        MappingProxyType(
            dict(zip(map(int, ids), map(int, vertex_memberships), strict=True))
        ),
        MappingProxyType(
            dict(zip(map(int, ids), map(int, vertex_closures), strict=True))
        ),
        {},
        cone_orientations,
        rays,
        {},
        {},
    )


def _sphere_strata(
    state: _SphereMetricState,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[tuple[tuple[str, ...], ...], ...],
    tuple[tuple[str, ...], ...],
]:
    rows = entity_keys_owner.key_rows(
        state.vertex_ids[:, None], edit.vertex_global_ids[:, None]
    )
    if np.any(rows < 0):
        raise ValueError(
            "surface_metric.sphere_strata must retain source-stratum parameters and incidence; got sphere successor strata omit generated vertices"
        )
    connectivity = []
    for block in edit.blocks:
        if (
            not isinstance(
                block, _dependency_phydrax_meshing__topology_edit.TopologyEditBlock
            )
            or block.cell_kind != "triangle"
        ):
            raise ValueError(
                "surface_metric.sphere_strata must retain source-stratum parameters and incidence; got sphere successor strata require typed triangle edit blocks"
            )
        connectivity.append(block.cells)
    cells = np.concatenate(connectivity)
    corners = rows[cells]
    dimensions = np.asarray(state.dimensions, dtype=np.int64)[corners]
    indices = np.asarray(state.indices, dtype=np.int64)[corners]
    original_indices = np.asarray(
        [
            [
                domain.source_indices[int(dim)][int(index)]
                for dim, index in zip(dims, values, strict=True)
            ]
            for dims, values in zip(dimensions, indices, strict=True)
        ],
        dtype=np.int64,
    )
    paths = tuple(
        tuple(
            domain.source_occurrences[int(dim)][int(index)]
            for dim, index in zip(dims, values, strict=True)
        )
        for dims, values in zip(dimensions, indices, strict=True)
    )
    entities = tuple(
        tuple(
            domain.entity_id(int(dim), int(index))
            for dim, index in zip(dims, values, strict=True)
        )
        for dims, values in zip(dimensions, indices, strict=True)
    )
    return (
        dimensions,
        original_indices,
        np.asarray(state.parameters, dtype=np.float64)[corners],
        paths,
        entities,
    )


def _sphere_reconstruct_stage(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    state: _SphereMetricState,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    target: CellMesh,
    patches: np.ndarray,
    atlas: _dependency_phydrax_geometry__sphere_material_atlas.SphereMaterialCellAtlas,
    policy: _owner_phydrax_meshing__adaptation.MeshAdaptationPolicy,
    maximum_fidelity: float,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> tuple[
    sphere_chart_records_owner.SphereGeometryReconstruction,
    sphere_chart_records_owner.PreparedSphereChartDeformation,
]:
    limits = policy.limits
    if source.certification is None:
        raise ValueError(
            "surface_metric.sphere_reconstruct_stage must retain accepted source certificates; got sphere reconstruction requires the accepted certificate limits"
        )
    certificate_limits = source.certification.request.limits
    degree = (
        _dependency_phydrax_discretization__cell_geometry_transfer.nested_geometry_degree(
            source.mesh, source.geometry
        )
    )
    maximum_nodes = min(
        policy.geometry_transition.maximum_evaluations,
        limits.maximum_data_bytes // (3 * np.dtype(np.float64).itemsize),
        budget.maximum_memory_bytes // (3 * np.dtype(np.float64).itemsize),
    )
    if degree == 1:
        layout = CellGeometrySpec.affine(target)
    else:
        layout = _dependency_phydrax_meshing__curving.straight_geometry(
            target,
            degree,
            maximum_nodes=maximum_nodes,
            maximum_entries=limits.maximum_connectivity_entries,
            maximum_bernstein_nodes=min(
                budget.maximum_work_units - budget.work_units,
                policy.audit_policy.validity_policy.maximum_bernstein_nodes,
            ),
        )
    dimensions, indices, parameters, paths, entities = _sphere_strata(
        state, edit, atlas.domain
    )
    with budget.activate(), budget.temporary_scope():
        reconstruction = _dependency_phydrax_discretization__sphere_chart_deformation.reconstruct_sphere_material_cell_geometry(
            atlas,
            target,
            layout,
            cell_patches=patches,
            corner_source_dimensions=dimensions,
            corner_source_indices=indices,
            corner_source_parameters=parameters,
            corner_source_occurrence_paths=paths,
            corner_source_entity_ids=entities,
            maximum_fidelity=maximum_fidelity,
            policy=policy.geometry_transition,
            budget=budget,
            certificate_limits=certificate_limits,
            validity_policy=policy.audit_policy.validity_policy,
            scope=scope,
        )
        original_embedding = source.certification.embedding
        original_certificates = (
            (source.audit.validity, original_embedding)
            if original_embedding is not None
            and source.audit.validity.policy_id
            == policy.audit_policy.validity_policy.policy_id
            and original_embedding.binding.limits_id == certificate_limits.limits_id
            else None
        )
        deformation = _dependency_phydrax_discretization__sphere_chart_deformation.prepare_sphere_chart_deformation(
            atlas,
            reconstruction.target_atlas,
            maximum_fidelity=maximum_fidelity,
            maximum_displacement=policy.geometry_transition.reconstruction_tolerance,
            maximum_candidate_pairs=limits.maximum_geometry_queries,
            maximum_pieces=limits.maximum_connectivity_entries,
            maximum_work_units=budget.maximum_work_units - budget.work_units,
            maximum_memory_bytes=budget.maximum_memory_bytes
            - budget.retained_basis_bytes
            - budget.temporary_bytes_upper,
            maximum_measure_work=budget.maximum_work_units - budget.work_units,
            budget=budget,
            certificate_limits=certificate_limits,
            validity_policy=policy.audit_policy.validity_policy,
            prepared_source_certificates=original_certificates,
            prepared_target_certificates=(
                reconstruction.target_validity,
                reconstruction.target_embedding,
            ),
            scope=scope,
        )
    reconstruction.require_bound(atlas, target)
    deformation.require_bound(
        source.mesh, source.geometry, target, reconstruction.geometry, budget=budget
    )
    return reconstruction, deformation


@checked
def build_sphere_periodic_metric_candidate(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    state: _SphereMetricState,
    background: _SphereMetricBackground,
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    policy: _owner_phydrax_meshing__adaptation.MeshAdaptationPolicy,
    candidate: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    exact_stencils: Mapping[
        int, _dependency_phydrax_meshing__periodic.ConstructionPointKey
    ],
    selection: _dependency_phydrax_meshing__periodic.PeriodicMetricOrbitSelection,
    prior_stage: _dependency_phydrax_meshing__periodic.PeriodicMetricOrbitOutcome | None,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> _dependency_phydrax_meshing__periodic.PeriodicMetricCandidateStage:
    """Apply radial operations to each carrier selected by source incidence."""
    current = source.mesh if prior_stage is None else prior_stage.target_mesh
    geometry = (
        source.geometry
        if prior_stage is None
        else prior_stage.geometry_stage.target_geometry
    )
    if (
        selection.original_result_id,
        selection.source_topology_id,
        selection.source_numeric_version,
        selection.source_frame_id,
        selection.source_geometry_id,
        selection.coordinate_contract_id,
    ) != (
        source.result_id,
        current.topology_id,
        current.numeric_version,
        canonical_fingerprint(array_tree_fingerprint(current.coordinates)),
        _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
            geometry
        ),
        source.coordinate_contract.spatial_id,
    ):
        raise ValueError(
            "surface_metric.build_sphere_periodic_metric_candidate must retain the declared source and target bindings; got sphere orbit selection is stale for the current private scientific frame"
        )
    if not exact_stencils or candidate.coordinates.shape[1] != 3:
        raise ValueError(
            "surface_metric.build_sphere_periodic_metric_candidate must retain source geometry invariants; got sphere orbit candidate lacks source support"
        )
    executed: list[int] = []
    with budget.activate():
        for carrier in selection.carriers:
            if carrier.protected_entity or carrier.protected_vertex_global_ids:
                raise ValueError(
                    "surface_metric.build_sphere_periodic_metric_candidate must retain source-stratum parameters and incidence; got a sphere quotient orbit intersects protected scientific strata"
                )
            ordered = tuple(
                carrier.vertex_global_ids[index] for index in carrier.vertex_permutation
            )
            rows = entity_keys_owner.key_rows(
                state.vertex_ids[:, None], np.asarray(ordered, dtype=np.int64)[:, None]
            )
            if np.any(rows < 0) or any(int(row) in state.successors for row in rows):
                raise ValueError(
                    "surface_metric.build_sphere_periodic_metric_candidate must retain source geometry invariants; got sphere orbit carriers have incompatible overlapping cavities"
                )
            changed = _sphere_apply_operation(
                state,
                selection.operation,
                tuple(map(int, rows)),
                background.atlas.domain,
                background,
                transfer,
                budget=budget,
                scope=scope,
            )
            if not changed:
                raise ValueError(
                    "surface_metric.build_sphere_periodic_metric_candidate must d sphere orbit operation fails its radial/source constraints; got a required sphere orbit operation fails its radial/source constraints"
                )
            executed.append(carrier.entity_global_id)
    edit, _, _, _, stencils = _sphere_assemble(source, state)
    return _dependency_phydrax_meshing__periodic.PeriodicMetricCandidateStage(
        edit, stencils, tuple(executed)
    )


def _sphere_apply_operation(
    state: _SphereMetricState,
    operation: _dependency_phydrax_meshing__periodic.PeriodicMetricOperation,
    vertices: tuple[int, ...],
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    background: _SphereMetricBackground,
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    /,
    *,
    reserved_workspace_upper: int = 0,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> bool:
    with budget.temporary_scope():
        # Orbit copies enter here without the producer's seed-cavity gate.
        # Admit each carrier before its proposal allocates or mutates.
        _admit_cavity(
            int(np.count_nonzero(np.any(np.isin(state.cells, vertices), axis=1))),
            budget=budget,
        )
        budget.reserve(
            0, max(0, _sphere_workspace_upper(state) - reserved_workspace_upper)
        )
        match operation:
            case "split":
                if len(vertices) != 2:
                    raise ValueError(
                        "surface_metric.sphere_apply_operation must its two edge vertices; got a sphere split requires its two edge vertices"
                    )
                edge = (min(vertices[0], vertices[1]), max(vertices[0], vertices[1]))
                return _sphere_split(
                    state, edge, domain, background, transfer, budget=budget
                )
            case "collapse":
                if len(vertices) != 2:
                    raise ValueError(
                        "surface_metric.sphere_apply_operation must its removed and kept vertices; got a sphere collapse requires its removed and kept vertices"
                    )
                return _sphere_collapse(
                    state,
                    vertices[0],
                    vertices[1],
                    domain,
                    background,
                    state.minimum_metric_quality,
                    budget=budget,
                ) or _sphere_collapse(
                    state,
                    vertices[1],
                    vertices[0],
                    domain,
                    background,
                    state.minimum_metric_quality,
                    budget=budget,
                )
            case "flip":
                if len(vertices) != 2:
                    raise ValueError(
                        "surface_metric.sphere_apply_operation must its two edge vertices; got a sphere flip requires its two edge vertices"
                    )
                edge = (min(vertices[0], vertices[1]), max(vertices[0], vertices[1]))
                return _sphere_flip(state, edge, domain, background, budget=budget)
            case "relocate":
                if len(vertices) != 1:
                    raise ValueError(
                        "surface_metric.sphere_apply_operation must one source vertex; got a sphere relocation requires one source vertex"
                    )
                return _sphere_relocate(
                    state, vertices[0], domain, background, transfer, budget=budget
                )
            case _:
                raise ValueError(
                    "surface_metric.sphere_apply_operation must select a supported metric operation; got unknown sphere metric operation"
                )


class _SphereMetricExecutionRequest(StrictModule):
    source: _dependency_phydrax_meshing__result.CellMeshingResult
    metric: ArrayLike
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain
    source_atlas: (
        _dependency_phydrax_geometry__sphere_material_atlas.SphereMaterialCellAtlas
    )
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer
    policy: _owner_phydrax_meshing__adaptation.MeshAdaptationPolicy
    coordinate_contract: SpatialCoordinateContract
    maximum_fidelity: float
    topology_operations: bool
    relocation: bool
    budget: (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    )
    record_phase: NativeMeshingPhaseRecorder | None
    scope: NativeExecutionScope
    minimum_metric_quality: float


class _SphereMetricPreparedInputs(StrictModule):
    values: np.ndarray
    initial_work: int
    schedule: MetricExecutionSchedule


class _SphereMetricCarrierPhase(StrictModule):
    state: _SphereMetricState
    background: _SphereMetricBackground
    workspace_upper: int


class _SphereMetricExecutionPhase(StrictModule):
    state: _SphereMetricState
    workspace_upper: int
    periodic_stage: (
        _dependency_phydrax_meshing__periodic.PeriodicMetricOrbitOutcome | None
    )
    periodic_geometry: (
        tuple[
            sphere_chart_records_owner.SphereGeometryReconstruction,
            sphere_chart_records_owner.PreparedSphereChartDeformation,
        ]
        | None
    )
    counts: MetricOperationCounts
    attempts: int
    rejected: int
    passes: int
    status: _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus
    applied: int


class _SphereMetricGeometryPhase(StrictModule):
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    target: CellMesh
    target_metric: np.ndarray
    stencils: Mapping[int, _dependency_phydrax_meshing__periodic.ConstructionPointKey]
    reconstruction: sphere_chart_records_owner.SphereGeometryReconstruction
    deformation: sphere_chart_records_owner.PreparedSphereChartDeformation


class _SphereMetricCertificatePhase(StrictModule):
    arcs: np.ndarray
    edge_ids: np.ndarray
    binding: str
    quality: np.ndarray
    status: _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus


def _prepare_sphere_metric_inputs(
    request: _SphereMetricExecutionRequest,
) -> _SphereMetricPreparedInputs:
    source = request.source
    metric = request.metric
    domain = request.domain
    source_atlas = request.source_atlas
    transfer = request.transfer
    policy = request.policy
    coordinate_contract = request.coordinate_contract
    maximum_fidelity = request.maximum_fidelity
    topology_operations = request.topology_operations
    relocation = request.relocation
    budget = request.budget
    record_phase = request.record_phase
    minimum_metric_quality = request.minimum_metric_quality
    source_atlas.require_bound(domain, source.mesh, source.geometry, coordinate_contract)
    domain.require_current(source_atlas.source_id, source_atlas.source_revision)
    if (
        not source.audit.passed
        or source.certification is None
        or not source.certification.passed
    ):
        raise ValueError(
            "surface_metric.prepare_sphere_metric_inputs must retain accepted source certificates; got sphere metric adaptation requires an accepted scientific predecessor"
        )

    budget.ticket.checkpoint()
    if transfer.support.domain.domain_id != domain.domain_id:
        raise ValueError(
            "surface_metric.prepare_sphere_metric_inputs must retain source geometry invariants; got sphere metric transfer changes the source domain"
        )
    if not np.isfinite(maximum_fidelity) or maximum_fidelity < 0:
        raise ValueError(
            "sphere_metric.maximum_fidelity must be finite and nonnegative; got invalid bound"
        )
    validate_metric_operation_controls(
        topology_operations,
        relocation,
        message="sphere_metric.operation_controls must use bool selectors; got nonboolean control",
    )
    validate_metric_phase_recorder(record_phase)
    validate_metric_quality_floor(
        minimum_metric_quality,
        message="sphere_metric.minimum_metric_quality must lie in [0, 1]; got invalid bound",
    )
    values = np.asarray(metric, dtype=np.float64)
    if values.shape != (source.mesh.coordinates.shape[0], 3, 3):
        raise ValueError(
            "sphere_metric.metric must carry one 3D tensor per source vertex; got incompatible tensor shape"
        )
    properties = _dependency_phydrax_meshing__metric.tensor_properties(values)
    if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
        raise ValueError(
            "sphere_metric.metric must contain finite Hermitian positive-definite tensors; got invalid spectrum"
        )
    limits = policy.limits
    schedule = MetricExecutionSchedule(
        maximum_passes=policy.maximum_passes,
        maximum_vertices=limits.maximum_vertices,
        maximum_cells=limits.maximum_cells,
        maximum_operations=limits.maximum_work_units,
        maximum_location_pairs=limits.maximum_geometry_queries,
        maximum_arc_work_units=limits.maximum_work_units,
    )
    _dependency_phydrax_meshing__tetra_metric.stage_source_limits(
        source.mesh, limits.maximum_vertices, limits.maximum_cells
    )
    initial_work = budget.work_units
    if (
        budget.maximum_work_units > limits.maximum_work_units
        or budget.maximum_memory_bytes > limits.maximum_scratch_bytes
    ):
        raise ValueError(
            "surface_metric.prepare_sphere_metric_inputs must stay within the authored work and workspace capacities; got sphere coordinate ledger cannot enlarge the authored work or workspace allowance"
        )
    return _SphereMetricPreparedInputs(values, initial_work, schedule)


def _prepare_sphere_metric_carriers(
    request: _SphereMetricExecutionRequest, prepared: _SphereMetricPreparedInputs
) -> _SphereMetricCarrierPhase:
    source = request.source
    source_atlas = request.source_atlas
    transfer = request.transfer
    policy = request.policy
    maximum_fidelity = request.maximum_fidelity
    budget = request.budget
    minimum_metric_quality = request.minimum_metric_quality
    values = prepared.values
    source_cells = sum(block.cell_count for block in source.mesh.blocks)
    workspace_upper, source_workspace_upper = _sphere_initial_workspace_upper(
        source.mesh, values
    )
    persistent_upper = workspace_upper + source_workspace_upper
    budget.reserve(9 * source_cells, persistent_upper)
    with budget.temporary_scope():
        construction_upper = source_cells * 65536 + values.nbytes * 16
        budget.reserve(0, max(0, construction_upper - persistent_upper))
        state, background = _sphere_initial_state(
            source,
            values,
            source_atlas,
            transfer,
            policy,
            maximum_fidelity,
            budget=budget,
        )
    state.minimum_metric_quality = minimum_metric_quality
    return _SphereMetricCarrierPhase(state, background, workspace_upper)


def _execute_sphere_metric_proposals(
    request: _SphereMetricExecutionRequest,
    prepared: _SphereMetricPreparedInputs,
    carrier: _SphereMetricCarrierPhase,
    execution: _SphereMetricExecutionPhase,
    proposals: list[
        tuple[
            _dependency_phydrax_meshing__periodic.PeriodicMetricOperation, tuple[int, ...]
        ]
    ],
) -> _SphereMetricExecutionPhase:
    source = request.source
    domain = request.domain
    transfer = request.transfer
    policy = request.policy
    maximum_fidelity = request.maximum_fidelity
    budget = request.budget
    record_phase = request.record_phase
    scope = request.scope
    state = execution.state
    background = carrier.background
    workspace_upper = execution.workspace_upper
    periodic_stage = execution.periodic_stage
    periodic_geometry = execution.periodic_geometry
    counts = execution.counts
    attempts = execution.attempts
    rejected = execution.rejected
    passes = execution.passes
    status = execution.status
    applied = execution.applied
    limits = request.policy.limits
    for operation, vertices in proposals:
        required_workspace = _sphere_workspace_upper(state)
        budget.reserve(0, max(0, required_workspace - workspace_upper))
        workspace_upper = max(workspace_upper, required_workspace)
        if budget.work_units >= budget.maximum_work_units:
            status = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
            break
        if operation == "split" and (
            state.points.shape[0] >= limits.maximum_vertices
            or state.cells.shape[0] + 2 > min(limits.maximum_cells, limits.maximum_faces)
            or 3 * (state.cells.shape[0] + 2) > limits.maximum_connectivity_entries
            or _sphere_edges(state)[0].shape[0] + 3 > limits.maximum_edges
        ):
            status = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
            break
        cavity = np.flatnonzero(np.any(np.isin(state.cells, vertices), axis=1))
        if cavity.size > limits.maximum_cavity_cells:
            rejected += 1
            continue

        _admit_cavity(cavity.size, budget=budget)
        attempts += 1
        budget.reserve(1)
        phase: NativeMeshingPhase
        match operation:
            case "split":
                phase = "metric_split"
            case "collapse":
                phase = "metric_collapse"
            case "flip":
                phase = "metric_reconnection"
            case "relocate":
                phase = "metric_relocation"
            case _:
                raise ValueError(
                    "surface_metric.execute_sphere_metric_proposals must select a supported metric operation; got unknown sphere metric phase"
                )
        with measure_phase(record_phase, phase):
            if source.mesh.periodic_topology is None:
                changed = _sphere_apply_operation(
                    state,
                    operation,
                    vertices,
                    domain,
                    background,
                    transfer,
                    reserved_workspace_upper=workspace_upper,
                    budget=budget,
                    scope=scope,
                )
            else:
                synchronized = _sphere_stage_periodic_operation(
                    source,
                    state,
                    background,
                    transfer,
                    policy,
                    maximum_fidelity,
                    operation,
                    vertices,
                    periodic_stage,
                    budget=budget,
                    scope=scope,
                )
                changed = synchronized is not None
                if synchronized is not None:
                    state, periodic_stage, reconstruction, deformation = synchronized
                    periodic_geometry = reconstruction, deformation
        if changed:
            match operation:
                case "split":
                    counts = replace(counts, splits=counts.splits + 1)
                case "collapse":
                    counts = replace(counts, collapses=counts.collapses + 1)
                case "flip":
                    counts = replace(counts, reconnections=counts.reconnections + 1)
                case "relocate":
                    counts = replace(counts, relocations=counts.relocations + 1)
                case _:
                    raise ValueError(
                        "sphere_metric.operation must select a supported edit; got unknown operation"
                    )
            applied += 1
        else:
            rejected += 1
    return _SphereMetricExecutionPhase(
        state,
        workspace_upper,
        periodic_stage,
        periodic_geometry,
        counts,
        attempts,
        rejected,
        passes,
        status,
        applied,
    )


def _run_sphere_metric_execution(
    request: _SphereMetricExecutionRequest,
    prepared: _SphereMetricPreparedInputs,
    carrier: _SphereMetricCarrierPhase,
) -> _SphereMetricExecutionPhase:
    topology_operations = request.topology_operations
    relocation = request.relocation
    budget = request.budget
    minimum_metric_quality = request.minimum_metric_quality
    state = carrier.state
    background = carrier.background
    workspace_upper = carrier.workspace_upper
    applied = 0
    counts = MetricOperationCounts()
    attempts = rejected = passes = 0
    status = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.PASS_LIMIT
    periodic_stage: (
        _dependency_phydrax_meshing__periodic.PeriodicMetricOrbitOutcome | None
    ) = None
    periodic_geometry: (
        tuple[
            sphere_chart_records_owner.SphereGeometryReconstruction,
            sphere_chart_records_owner.PreparedSphereChartDeformation,
        ]
        | None
    ) = None

    # Pass measurement and direct operations share these same four live
    # carriers. Keep their high-water reservation until the phase
    # ends; unchanged state must not allocate another carrier bank.
    def checkpoint() -> None:

        budget.ticket.checkpoint()

    for passes in metric_execution_passes(prepared.schedule, checkpoint):
        required_workspace = _sphere_workspace_upper(state)
        budget.reserve(0, max(0, required_workspace - workspace_upper))
        workspace_upper = max(workspace_upper, required_workspace)
        edges, lengths = _sphere_lengths(state)
        quality = _sphere_quality(state, state.cells, state.patches, background)
        if np.all((lengths >= _LOWER) & (lengths <= _UPPER)) and np.all(
            quality >= minimum_metric_quality
        ):
            status = (
                _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.COMPLETE
            )
            break
        proposals: list[
            tuple[
                _dependency_phydrax_meshing__periodic.PeriodicMetricOperation,
                tuple[int, ...],
            ]
        ] = []
        if topology_operations:
            proposals.extend(
                ("split", tuple(map(int, edges[row])))
                for row in np.argsort(-lengths, kind="stable")
                if lengths[row] > _UPPER
            )
            proposals.extend(
                ("collapse", tuple(map(int, edges[row])))
                for row in np.argsort(lengths, kind="stable")
                if lengths[row] < _LOWER
            )
            proposals.extend(("flip", tuple(map(int, edge))) for edge in edges)
        if relocation:
            proposals.extend(
                ("relocate", (int(vertex),)) for vertex in np.unique(state.cells)
            )
        applied = 0
        execution = _SphereMetricExecutionPhase(
            state,
            workspace_upper,
            periodic_stage,
            periodic_geometry,
            counts,
            attempts,
            rejected,
            passes,
            status,
            applied,
        )
        execution = _execute_sphere_metric_proposals(
            request, prepared, carrier, execution, proposals
        )
        state = execution.state
        workspace_upper = execution.workspace_upper
        periodic_stage = execution.periodic_stage
        periodic_geometry = execution.periodic_geometry
        counts = execution.counts
        attempts = execution.attempts
        rejected = execution.rejected
        passes = execution.passes
        status = execution.status
        applied = execution.applied
        if (
            status
            is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
        ):
            break
        if not applied:
            status = (
                _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.STALLED
            )
            break
    return _SphereMetricExecutionPhase(
        state,
        workspace_upper,
        periodic_stage,
        periodic_geometry,
        counts,
        attempts,
        rejected,
        passes,
        status,
        applied,
    )


def _assemble_sphere_metric_geometry(
    request: _SphereMetricExecutionRequest,
    prepared: _SphereMetricPreparedInputs,
    carrier: _SphereMetricCarrierPhase,
    execution: _SphereMetricExecutionPhase,
) -> _SphereMetricGeometryPhase:
    source = request.source
    source_atlas = request.source_atlas
    policy = request.policy
    maximum_fidelity = request.maximum_fidelity
    budget = request.budget
    scope = request.scope
    state = execution.state
    workspace_upper = execution.workspace_upper
    periodic_stage = execution.periodic_stage
    periodic_geometry = execution.periodic_geometry
    required_workspace = _sphere_workspace_upper(state)
    budget.reserve(0, max(0, required_workspace - workspace_upper))
    edit, target_metric, patches, _, stencils = _sphere_assemble(source, state)
    if periodic_stage is None:
        if source.mesh.periodic_topology is not None:
            # An unchanged quotient source has no born orbit identities.
            edit = edit._replace(
                periodic_orbits=_unchanged_periodic_witness(source, budget=budget)
            )
        version = canonical_fingerprint(
            {
                "kind": "sphere-metric-numeric-frame",
                "source": source.mesh.numeric_version,
                "coordinates": array_tree_fingerprint(edit.coordinates),
            }
        )
        target, _, _ = _dependency_phydrax_meshing__topology_edit.assemble_topology_edit(
            source.mesh, edit, numeric_version=version, budget=budget
        )
        reconstruction, deformation = _sphere_reconstruct_stage(
            source,
            state,
            edit,
            target,
            patches,
            source_atlas,
            policy,
            maximum_fidelity,
            budget=budget,
            scope=scope,
        )
    else:
        if periodic_geometry is None:
            raise ValueError(
                "surface_metric.assemble_sphere_metric_geometry must retain source geometry invariants; got sphere periodic edit lacks its full-map geometry stage"
            )
        edit, target = periodic_stage.edit, periodic_stage.target_mesh
        stencils = periodic_stage.exact_stencils
        reconstruction, deformation = periodic_geometry
    return _SphereMetricGeometryPhase(
        edit, target, target_metric, stencils, reconstruction, deformation
    )


def _certify_sphere_metric_execution(
    request: _SphereMetricExecutionRequest,
    prepared: _SphereMetricPreparedInputs,
    carrier: _SphereMetricCarrierPhase,
    execution: _SphereMetricExecutionPhase,
    geometry: _SphereMetricGeometryPhase,
) -> _SphereMetricCertificatePhase:
    budget = request.budget
    minimum_metric_quality = request.minimum_metric_quality
    background = carrier.background
    status = execution.status
    target = geometry.target
    reconstruction = geometry.reconstruction
    arcs, edge_ids, binding = _sphere_certify_metric_arcs(
        target, reconstruction, background, budget=budget
    )
    quality = _sphere_certify_metric_quality(
        target, reconstruction, background, budget=budget
    )
    accepted = np.all((arcs[:, 0] >= _LOWER) & (arcs[:, 1] <= _UPPER)) and np.all(
        quality >= minimum_metric_quality
    )
    if (
        accepted
        and status
        is not _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
    ):
        status = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.COMPLETE
    elif (
        status is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.COMPLETE
    ):
        status = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.STALLED
    return _SphereMetricCertificatePhase(arcs, edge_ids, binding, quality, status)


def _assemble_sphere_metric_outcome(
    request: _SphereMetricExecutionRequest,
    prepared: _SphereMetricPreparedInputs,
    carrier: _SphereMetricCarrierPhase,
    execution: _SphereMetricExecutionPhase,
    geometry: _SphereMetricGeometryPhase,
    certificate: _SphereMetricCertificatePhase,
) -> SphereMetricOutcome:
    domain = request.domain
    source_atlas = request.source_atlas
    topology_operations = request.topology_operations
    budget = request.budget
    minimum_metric_quality = request.minimum_metric_quality
    initial_work = prepared.initial_work
    state = execution.state
    counts = execution.counts
    attempts = execution.attempts
    rejected = execution.rejected
    passes = execution.passes
    status = certificate.status
    edit = geometry.edit
    target = geometry.target
    target_metric = geometry.target_metric
    stencils = geometry.stencils
    reconstruction = geometry.reconstruction
    deformation = geometry.deformation
    arcs = certificate.arcs
    edge_ids = certificate.edge_ids
    binding = certificate.binding
    quality = certificate.quality
    limits = request.policy.limits
    work = budget.work_units - initial_work
    evidence = _dependency_phydrax_meshing__tetra_metric.MetricRemeshingEvidence(
        status,
        passes=passes,
        counts=counts.evidence_counts(),
        rejected_operations=rejected,
        work_units=work,
        lengths=np.mean(arcs, axis=1),
        quality=quality,
        metric_quality_floor=minimum_metric_quality,
        maximum_fidelity_bound=float(np.max(np.asarray(reconstruction.fidelity_bounds))),
        criterion=_dependency_phydrax_meshing__tetra_metric.MetricRemeshingCriterion.UNIT_MESH
        if topology_operations
        else _dependency_phydrax_meshing__tetra_metric.MetricRemeshingCriterion.RELOCATION_FIXED_POINT,
        operation_attempts=attempts,
        native_work_units=0,
        metric_arc_bounds=arcs,
        metric_arc_edge_ids=edge_ids,
        metric_arc_binding=binding,
        resource_message=(
            "sphere_metric.capacity must admit the requested edit; got exhausted native work or entity capacity"
            if status
            is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
            else None
        ),
        resource_requested=(
            ("maximum_work_units", float(limits.maximum_work_units)),
            ("maximum_vertices", float(limits.maximum_vertices)),
            ("maximum_edges", float(limits.maximum_edges)),
            ("maximum_cells", float(limits.maximum_cells)),
            ("maximum_faces", float(limits.maximum_faces)),
            (
                "maximum_connectivity_entries",
                float(limits.maximum_connectivity_entries),
            ),
        )
        if status
        is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
        else (),
        resource_achieved=(
            ("coordinate_work_units", float(budget.work_units)),
            ("vertices", float(state.points.shape[0])),
            ("edges", float(_sphere_edges(state)[0].shape[0])),
            ("cells", float(state.cells.shape[0])),
            ("connectivity_entries", float(3 * state.cells.shape[0])),
        )
        if status
        is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
        else (),
    )
    return SphereMetricOutcome(
        edit,
        target,
        reconstruction,
        source_atlas,
        reconstruction.target_atlas,
        deformation,
        target_metric,
        stencils,
        evidence,
        domain.source_id,
        domain.source_revision,
        domain.domain_id,
    )


@checked
def execute_sphere_metric_adaptation(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    metric: ArrayLike,
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    source_atlas: _dependency_phydrax_geometry__sphere_material_atlas.SphereMaterialCellAtlas,
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    /,
    *,
    policy: _owner_phydrax_meshing__adaptation.MeshAdaptationPolicy,
    coordinate_contract: SpatialCoordinateContract,
    maximum_fidelity: float,
    minimum_metric_quality: float = 0.05,
    topology_operations: bool,
    relocation: bool,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
) -> SphereMetricOutcome:
    """Stage sphere edits under the source ledger and requested quality floor."""

    request = _SphereMetricExecutionRequest(
        source=source,
        metric=metric,
        domain=domain,
        source_atlas=source_atlas,
        transfer=transfer,
        policy=policy,
        coordinate_contract=coordinate_contract,
        maximum_fidelity=maximum_fidelity,
        topology_operations=topology_operations,
        relocation=relocation,
        budget=budget,
        record_phase=record_phase,
        scope=scope,
        minimum_metric_quality=minimum_metric_quality,
    )
    prepared = _prepare_sphere_metric_inputs(request)
    with _sphere_coordinate_phase(budget=budget):
        carrier = _prepare_sphere_metric_carriers(request, prepared)
        execution = _run_sphere_metric_execution(request, prepared, carrier)
        geometry = _assemble_sphere_metric_geometry(request, prepared, carrier, execution)
        certificate = _certify_sphere_metric_execution(
            request, prepared, carrier, execution, geometry
        )
        return _assemble_sphere_metric_outcome(
            request, prepared, carrier, execution, geometry, certificate
        )


def _sphere_certify_metric_quality(
    mesh: CellMesh,
    reconstruction: sphere_chart_records_owner.SphereGeometryReconstruction,
    background: _SphereMetricBackground,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Bound whole-cell quality from the successor differential."""
    # Rational Gram expressions and spectral interval arithmetic form one bounded
    # proof kernel; retain expression construction and charge/reduction order.

    constant_metric = np.array_equal(
        background.metric, np.broadcast_to(background.metric[0], background.metric.shape)
    )
    if (
        constant_metric
        and _dependency_phydrax_discretization__cell_geometry_transfer.nested_geometry_degree(
            mesh, reconstruction.geometry
        )
        == 1
    ):
        return _sphere_affine_constant_metric_quality(
            mesh, reconstruction, background.metric[0], budget=budget
        )
    ratio = Fraction(1)
    if not constant_metric:
        spectra = tuple(
            _owner_phydrax_geometry__domain_numerics.gram_spectrum_bounds(value)
            for value in background.metric
        )
        low = min(value[0] for value in spectra)
        high = max(value[1] for value in spectra)
        if low <= 0:
            raise ValueError(
                "surface_metric.sphere_certify_metric_quality must retain a nonsingular positive-definite source frame; got sPD background lacks a certified positive spectral lower bound"
            )
        ratio = (
            _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
                Fraction(low), budget=budget
            )[0]
            / _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
                Fraction(high), budget=budget
            )[1]
        )
    tensor = background.metric[0] if constant_metric else np.eye(3, dtype=np.float64)
    matrix = tuple(rational_vector(row) for row in tensor)
    elements, routes, _ = reconstruction.geometry.resolve(mesh, budget=budget)
    bank = _dependency_phydrax_discretization__coordinate_enclosure.prepared_coordinate_source_bank(
        reconstruction.geometry, budget=budget
    )
    result: list[float] = []
    for element, route in zip(elements, routes, strict=True):
        for nodes in np.asarray(route, dtype=np.int64):
            with budget.temporary_scope():
                coordinates = _owner_phydrax_discretization__coordinate_enclosure.coordinate_expressions(
                    element, tuple((bank[int(node)] for node in nodes)), budget=budget
                )
                if coordinates is None:
                    raise ValueError(
                        "surface_metric.sphere_certify_metric_quality must the coordinate map; got sphere quality requires the coordinate map"
                    )
                derivatives = tuple(
                    tuple(
                        _owner_phydrax_discretization__coordinate_enclosure.expression_derivative(
                            value, axis, budget=budget
                        )
                        for value in coordinates
                    )
                    for axis in range(2)
                )
                gram = tuple(
                    tuple(
                        _owner_phydrax_discretization__coordinate_enclosure.expression_sum(
                            tuple(
                                (
                                    _owner_phydrax_discretization__coordinate_enclosure.expression_scale(
                                        _owner_phydrax_discretization__coordinate_enclosure.expression_multiply(
                                            derivatives[a][i],
                                            derivatives[b][j],
                                            budget=budget,
                                        ),
                                        matrix[i][j],
                                        budget=budget,
                                    )
                                    for i in range(3)
                                    for j in range(3)
                                )
                            ),
                            budget=budget,
                        )
                        for b in range(2)
                    )
                    for a in range(2)
                )
                determinant = _owner_phydrax_discretization__coordinate_enclosure.expression_determinant(
                    gram, variable_dimension=2, budget=budget
                )
                denominator = _owner_phydrax_discretization__coordinate_enclosure.expression_sum(
                    (
                        gram[0][0],
                        gram[1][1],
                        _owner_phydrax_discretization__coordinate_enclosure.expression_scale(
                            gram[0][1], -1, budget=budget
                        ),
                    ),
                    budget=budget,
                )
                floor = min(
                    rational_coefficient(
                        control, role="surface_metric.sphere_quality.determinant_controls"
                    )
                    for control in _owner_phydrax_discretization__coordinate_enclosure.expression_bernstein_coefficients(
                        determinant, "simplex", 2, budget=budget
                    )
                )
                ceiling = max(
                    rational_coefficient(
                        control, role="surface_metric.sphere_quality.denominator_controls"
                    )
                    for control in _owner_phydrax_discretization__coordinate_enclosure.expression_bernstein_coefficients(
                        denominator, "simplex", 2, budget=budget
                    )
                )
                if floor <= 0 or ceiling <= 0:
                    result.append(0.0)
                    continue
                lower = (
                    _dependency_phydrax_linalg__hermitian_spectral.fraction_sqrt_interval(
                        3 * floor, budget=budget
                    )[0]
                    / ceiling
                ) * ratio
                result.append(
                    max(0.0, fraction_endpoint(lower, upper=False, saturate=True))
                )
    return np.asarray(result, dtype=np.float64)


@contextmanager
def _sphere_coordinate_phase(
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Iterator[None]:
    """Measure rational-arithmetic and native work on the phase ticket."""
    with budget.activate(), budget.ticket.native_target(), budget.temporary_scope():
        yield


def _sphere_stage_periodic_operation(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    state: _SphereMetricState,
    background: _SphereMetricBackground,
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer,
    policy: _owner_phydrax_meshing__adaptation.MeshAdaptationPolicy,
    maximum_fidelity: float,
    operation: _dependency_phydrax_meshing__periodic.PeriodicMetricOperation,
    vertices: tuple[int, ...],
    prior_stage: _dependency_phydrax_meshing__periodic.PeriodicMetricOrbitOutcome | None,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> (
    tuple[
        _SphereMetricState,
        _dependency_phydrax_meshing__periodic.PeriodicMetricOrbitOutcome,
        sphere_chart_records_owner.SphereGeometryReconstruction,
        sphere_chart_records_owner.PreparedSphereChartDeformation,
    ]
    | None
):

    if source.certification is None:
        raise ValueError(
            "surface_metric.sphere_stage_periodic_operation must retain accepted source certificates; got periodic sphere operations require source certificates"
        )
    # Separate seed and aggregate states are necessary rollback workspaces:
    # the seed carrier is executed first by the callback to retain born IDs.
    with budget.temporary_scope():
        budget.reserve(
            0,
            state.points.nbytes * 16
            + state.cells.nbytes * 32
            + len(state.stencils) * 65536,
        )
        seed = _owner_copy.deepcopy(state)
        if not _sphere_apply_operation(
            seed,
            operation,
            vertices,
            background.atlas.domain,
            background,
            transfer,
            budget=budget,
            scope=scope,
        ):
            return None
        candidate, _, _, _, exact = _sphere_assemble(source, seed)
        aggregate = _owner_copy.deepcopy(state)
        geometry_records: list[
            tuple[
                sphere_chart_records_owner.SphereGeometryReconstruction,
                sphere_chart_records_owner.PreparedSphereChartDeformation,
            ]
        ] = []

        def build_candidate(
            edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
            stencils: Mapping[
                int, _dependency_phydrax_meshing__periodic.ConstructionPointKey
            ],
            selection: _dependency_phydrax_meshing__periodic.PeriodicMetricOrbitSelection,
        ) -> _dependency_phydrax_meshing__periodic.PeriodicMetricCandidateStage:
            return build_sphere_periodic_metric_candidate(
                source,
                aggregate,
                background,
                transfer,
                policy,
                edit,
                stencils,
                selection,
                prior_stage,
                budget=budget,
                scope=scope,
            )

        def build_geometry(
            edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
            target: CellMesh,
        ) -> _owner_phydrax_meshing__periodic.PeriodicMetricGeometryStage:
            _, _, patches, _, _ = _sphere_assemble(source, aggregate)
            reconstruction, deformation = _sphere_reconstruct_stage(
                source,
                aggregate,
                edit,
                target,
                patches,
                background.atlas,
                policy,
                maximum_fidelity,
                budget=budget,
                scope=scope,
            )
            transition = _dependency_phydrax_discretization__cell_geometry_transfer.transition_chart_deformed_cell_geometry(
                source.mesh,
                source.geometry,
                target,
                reconstruction,
                deformation,
                policy=policy.geometry_transition,
            )
            lineage = _owner_phydrax_meshing__periodic.metric_target_lineage(
                source.mesh, edit, target, scope=scope, budget=budget
            )
            associations = transfer.propagate(
                source,
                lineage,
                target,
                geometry=reconstruction.geometry,
                embedding=reconstruction.target_embedding,
                deformation=deformation,
            )
            geometry_records.append((reconstruction, deformation))
            return _owner_phydrax_meshing__periodic.PeriodicMetricGeometryStage(
                target, reconstruction.geometry, associations, deformation, transition
            )

        retained = (
            () if prior_stage is None else prior_stage.periodic_witness.quotient_entities
        )
        result = _owner_phydrax_meshing__periodic.synchronize_periodic_metric_edit(
            _owner_phydrax_meshing__periodic.PeriodicMetricEditRequest(
                source,
                candidate,
                exact,
                operation=operation,
                limits=policy.limits,
                policy=policy,
                budget=budget,
                scope=scope,
                certificate_limits=source.certification.request.limits,
                prior_stage=prior_stage,
                build_candidate=build_candidate,
                build_geometry=build_geometry,
                retained_quotient_entities=retained,
            ),
        )
        if len(geometry_records) != 1:
            raise ValueError(
                "surface_metric.sphere_stage_periodic_operation must retain source geometry invariants; got sphere orbit controller did not build exactly one successor map"
            )
        reconstruction, deformation = geometry_records[0]
        return aggregate, result, reconstruction, deformation


def _unchanged_periodic_witness(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__topology_edit.PeriodicVertexOrbitWitness:

    state = _owner_phydrax_meshing__periodic.PeriodicConstructionOrbits(source.mesh)
    return state.witness(
        source.mesh,
        source.mesh,
        {
            int(identifier): ((int(identifier), Fraction(1)),)
            for identifier in np.asarray(source.mesh.vertex_global_ids)
        },
        (),
        budget=budget,
    )


def _sphere_workspace_upper(state: _SphereMetricState, /) -> int:
    """Host carrier/cavity-copy upper bound, not native memory or process RSS."""
    arrays = sum(
        value.nbytes
        for value in (
            state.points,
            state.directions,
            state.metric,
            state.cells,
            state.patches,
            state.classes,
            state.cell_ids,
            state.vertex_ids,
        )
    )
    memberships = sum(map(len, state.parents)) + sum(map(len, state.vertex_sources))
    support = sum(len(key) for key in state.stencils)
    # Four live carriers cover untouched state, candidate copies, NumPy edge
    # tables and lineage assembly; Python sets/dicts have separate slot bounds.
    return 4096 + 12 * arrays + 2048 * (memberships + support + len(state.features))


def _sphere_initial_workspace_upper(
    mesh: CellMesh, metric: np.ndarray, /
) -> tuple[int, int]:
    """Admit persistent carrier/source payload before its constructor grows it."""
    cells = sum(block.cell_count for block in mesh.blocks)
    vertices = mesh.coordinates.shape[0]
    arrays = (
        2 * mesh.coordinates.size * np.dtype(np.float64).itemsize
        + metric.nbytes
        + 3 * cells * np.dtype(np.int32).itemsize
        + cells * (np.dtype(np.int32).itemsize + 2 * np.dtype(np.int64).itemsize)
        + vertices * np.dtype(np.int64).itemsize
    )
    # Initial parent/support sets contain one scientific ID each; every edge
    # is an upper bound on feature memberships, including protected ones.
    carriers = (
        4096 + 12 * arrays + 2048 * (cells + 2 * vertices + mesh.entity_set(1).count)
    )
    # Nine dyadic direction fractions and their row tuples fit in 8 KiB
    # per cone, even at binary64's maximum denominator height. Immutable source
    # array copies and both ID/membership maps have separate vertex slot bounds.
    source = 4096 + 8192 * cells + 1024 * vertices + 2 * metric.nbytes
    return carriers, source
