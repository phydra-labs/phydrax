#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Source-chart constrained anisotropic remeshing of embedded triangle surfaces.

Per-cell charts retain seams and pole uses; coordinates are evaluated/projected
by the authoritative MeshingDomain in batches. The returned chart witnesses are
not an affine coordinate-map transfer. Publication must reconstruct and certify
the requested successor coordinate map through its geometry transition owner.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import isfinite, lcm, sqrt
from typing import cast, NamedTuple, Protocol, TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import (
    bigint_bytes,
    exact_bits,
    exact_charge,
    orient2d,
    orient3d,
    PredicateMode,
    PredicateSign,
)
from .._meshcore import charge_native_geometry_queries, current_native_execution_budget
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_transfer import (
    CellGeometryTransition,
    CellGeometryTransitionPolicy,
    nested_geometry_degree,
    reconstruct_parametric_surface_cell_geometry,
    SurfaceGeometryReconstruction,
    transition_chart_deformed_cell_geometry,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
)
from ..discretization._coordinate_enclosure import (
    _COORDINATE_BUDGET,
    outward,
    prepared_coordinate_source_bank,
)
from ..discretization._surface_chart_deformation import (
    prepare_surface_chart_deformation,
    PreparedSurfaceChartDeformation,
    SurfaceChartWitness,
)
from ..geometry._mesh_certificates import (
    GlobalEmbeddingCertificate,
    MeshCertificateLimits,
    SourceFidelityCertificate,
)
from ..geometry._meshing_domain import MeshingDomain, MeshingDomainBoundarySource
from ..geometry.brep._patches import SpherePatch, surface_differential
from ..linalg import (
    hermitian_exp_enclosure,
    hermitian_log_enclosure,
    HermitianFunctionEnclosure,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ..linalg._hermitian_spectral import (
    _fraction_sqrt_interval,
    _matrix_norm_enclosure,
    _reserve_fraction_work,
)
from ._association import SurfaceAssociationTransfer
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._curving import _straight_geometry
from ._lineage import EntityLineageKind
from ._measurements import measure_phase, NativeMeshingPhaseRecorder
from ._metric import _metric_shape_quality, _tensor_properties, interpolate_mesh_metric
from ._result import CellMeshingResult
from ._tetra_metric import (
    _fresh_identifiers,
    _lineage_operation_kind,
    _resolved_collapses,
    _source_location_budget,
    _stage_source_limits,
    _subentity_relations,
    MetricRemeshingCriterion,
    MetricRemeshingEvidence,
    MetricRemeshingStatus,
)
from ._topology_edit import (
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    key_rows,
    source_family_blocks,
)
from ._trace import MeshingStageKind


if TYPE_CHECKING:
    from ..geometry._surface_source_support import SurfaceSourceRootAtlas
    from ._surface_association_transfer import PreparedSurfaceCurveWitness

_LOWER = 1.0 / np.sqrt(2.0)
_UPPER = np.sqrt(2.0)
_LOCAL_EDGES = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int32)
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(8)
_PARAMETERS = (_GAUSS_NODES + 1.0) / 2.0
_WEIGHTS = _GAUSS_WEIGHTS / 2.0


class SurfaceMetricOutcome(NamedTuple):
    edit: CellTopologyEdit
    metric: np.ndarray
    evidence: MetricRemeshingEvidence
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
    periodic_stage: PeriodicMetricOrbitOutcome | None = None


class _SurfaceMetricGeometryWitness(NamedTuple):
    """Actual private edit/chart uses, without unearned completion evidence."""

    edit: CellTopologyEdit
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


class _SurfaceSource(NamedTuple):
    cells: np.ndarray
    charts: np.ndarray
    patches: np.ndarray
    cell_ids: np.ndarray
    vertex_ids: np.ndarray
    metric: np.ndarray
    physical_charts: np.ndarray


@dataclass(slots=True)
class _SurfaceState:
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
    curve_witness: PreparedSurfaceCurveWitness | None = None
    vertex_strata: list[tuple[int, int, np.ndarray]] | None = None
    curve_workspace_vertices: int = 0
    material_charts: np.ndarray | None = None
    material_workspace_bytes_upper: int = 0
    material_fraction_bits: int = 0


class _SurfaceMetricControllerProtocol(Protocol):
    """Private orbit staging interface; scientific arrays remain owned by the state."""

    source: CellMeshingResult
    witness: SurfaceChartWitness
    background: _SurfaceSource | None
    prior_stage: PeriodicMetricOrbitOutcome | None

    @property
    def operation_count(self) -> int: ...

    def stage(
        self,
        state: _SurfaceState,
        operation: PeriodicMetricOperation,
        vertices: tuple[int, ...],
        fixed: np.ndarray,
        features: set[tuple[int, int]],
        feature_codes: dict[tuple[int, int], int],
        blocked: set[tuple[int, int]],
        /,
    ) -> _SurfaceState | None: ...


def _edge_table(state: _SurfaceState, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = np.sort(state.cells[:, _LOCAL_EDGES].reshape((-1, 2)), axis=1)
    edges, first, inverse = np.unique(
        records, axis=0, return_index=True, return_inverse=True
    )
    return edges, first, inverse.reshape((-1, 3))


def _mapped_quality(
    domain: MeshingDomain,
    metric: np.ndarray,
    cells: np.ndarray,
    patches: np.ndarray,
    charts: np.ndarray,
    /,
) -> np.ndarray:
    """Mean ratio of the physical map differential, never a chord/UV surrogate."""
    charge_native_geometry_queries(cells.shape[0])
    tangent = np.zeros((cells.shape[0], 3, 3), dtype=np.float64)
    for patch in np.unique(patches).tolist():
        selected = patches == patch
        differential = np.asarray(
            surface_differential(
                domain.patches[patch].surface,
                jnp.asarray(np.mean(charts[selected], axis=1), dtype=jnp.float64),
            ),
            dtype=np.float64,
        )
        tangent[selected, 1:] = (
            charts[selected, 1:] - charts[selected, :1]
        ) @ np.swapaxes(differential, -1, -2)
    mean = np.asarray(
        interpolate_mesh_metric(
            metric[cells], np.full(cells.shape, 1.0 / 3.0, dtype=np.float64)
        ),
        dtype=np.float64,
    )
    return np.asarray(
        _metric_shape_quality(jnp.asarray(mean), jnp.asarray(tangent), dimension=2),
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
    state: _SurfaceState,
    source: _SurfaceSource,
    maximum_pairs: int,
    /,
) -> tuple[list[_MetricArcPiece], int]:
    """Exact source-background clipping; every edge interval has one original owner."""
    edges, first, _ = _edge_table(state)
    owner, local = first // 3, first % 3
    charts = state.charts if state.material_charts is None else state.material_charts
    endpoints = charts[owner[:, None], _LOCAL_EDGES[local]]
    actual_endpoints = state.charts[owner[:, None], _LOCAL_EDGES[local]]
    pieces, work = [], 0
    for edge, uv in enumerate(endpoints):
        origin = _exact_uv(uv[0])
        budget = current_native_execution_budget()
        matching = np.flatnonzero(
            source.patches == int(state.patches[owner[edge]])
        ).tolist()
        if budget is not None:
            budget.admit_work_bound(len(matching))
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
                charge_native_geometry_queries(0, work_units=work - previous_work)
                raise _source_location_budget(maximum_pairs, work)
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
        charge_native_geometry_queries(0, work_units=len(matching))
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
                raise MeshingFailure(
                    MeshingFailureCategory.LINEAGE_FAILED,
                    "A metric arc leaves its original chart-field background.",
                    stage=MeshingStageKind.LINEAGE_CONSTRUCTION.value,
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
    source: _SurfaceSource,
    logarithms: dict[int, HermitianFunctionEnclosure],
    /,
) -> tuple[tuple[tuple[Fraction, ...], ...], Fraction]:
    tensors = source.metric[np.asarray(piece.vertices, dtype=np.int64)]
    if np.array_equal(tensors, np.broadcast_to(tensors[0], tensors.shape)):
        return tuple(
            tuple(Fraction(float(value)) for value in row) for row in tensors[0]
        ), Fraction(0)
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
    budget = _COORDINATE_BUDGET.get()
    error = _matrix_norm_enclosure(slope, coordinate_budget=budget) + sum(
        (
            (abs(center) + radius) * log.error
            for center, radius, log in zip(centers, radii, logs, strict=True)
        ),
        Fraction(0),
    )
    return hermitian_exp_enclosure(
        HermitianFunctionEnclosure(nominal, error), coordinate_budget=budget
    )


def _arc_piece_bounds(
    piece: _MetricArcPiece,
    lower: np.ndarray,
    upper: np.ndarray,
    metric: tuple[tuple[Fraction, ...], ...],
    metric_error: Fraction,
    /,
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
    budget = _COORDINATE_BUDGET.get()
    low = _fraction_sqrt_interval(
        max(Fraction(0), gram_low - radius), coordinate_budget=budget
    )[0]
    high = _fraction_sqrt_interval(
        max(Fraction(0), gram_high + radius), coordinate_budget=budget
    )[1]
    width = piece.upper - piece.lower
    return width * low, width * high


def _fraction_sqrt_lower(
    value: Fraction, budget: CoordinateEnclosureBudget | None, /
) -> Fraction:
    """One-sided rational radical proof without constructing an unused upper bound."""
    if value <= 0:
        return Fraction(0)
    _reserve_fraction_work(
        budget,
        1,
        8,
        max(abs(value.numerator).bit_length(), value.denominator.bit_length(), 1075),
    )
    lower = sqrt(float(value))
    if not isfinite(lower) or lower == 0:
        raise ValueError("Embedded measure exceeds certified binary64 arithmetic range.")
    while Fraction(lower) ** 2 > value:
        if budget is not None:
            budget.reserve(1)
        lower = float(np.nextafter(lower, -np.inf))
    return Fraction(lower)


def _fraction_sqrt_upper(
    value: Fraction, budget: CoordinateEnclosureBudget | None, /
) -> Fraction:
    """One-sided rational radical proof without constructing an unused lower bound."""
    if value <= 0:
        return Fraction(0)
    _reserve_fraction_work(
        budget,
        1,
        8,
        max(abs(value.numerator).bit_length(), value.denominator.bit_length(), 1075),
    )
    upper = sqrt(float(value))
    if not isfinite(upper) or upper == 0:
        raise ValueError("Embedded measure exceeds certified binary64 arithmetic range.")
    while Fraction(upper) ** 2 < value:
        if budget is not None:
            budget.reserve(1)
        upper = float(np.nextafter(upper, np.inf))
    return Fraction(upper)


def _constant_metric_tangent_upper(
    direction: tuple[Fraction, Fraction],
    lower: np.ndarray,
    upper: np.ndarray,
    matrix: tuple[tuple[Fraction, ...], ...],
    budget: CoordinateEnclosureBudget | None,
    ceiling: Fraction | None,
    /,
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
    return _fraction_sqrt_upper(gram_upper, budget)


def _constant_surface_metric_arc_bounds(
    domain: MeshingDomain,
    state: _SurfaceState,
    metric: np.ndarray,
    lower_required: np.ndarray,
    /,
) -> tuple[np.ndarray, int]:
    """Whole-patch derivative upper bounds and exact constant-metric chords."""
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
            boxes[selected], order=1
        )
        derivative_lower[selected] = low
        derivative_upper[selected] = high
    box_count = base_rows.size
    matrix = tuple(tuple(Fraction(float(value)) for value in row) for row in metric)
    budget = _COORDINATE_BUDGET.get()
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
                direction, lower, upper, matrix, budget, arc_ceiling
            )
        if high > arc_ceiling:
            divisions = (
                64
                if isinstance(
                    domain.patches[int(patches[edge_index])].surface,
                    SpherePatch,
                )
                else 3
            )
            exact_charts = tuple(
                tuple(Fraction(float(value)) for value in point) for point in charts
            )
            subboxes = []
            for section in range(divisions):
                interval = Fraction(section, divisions), Fraction(section + 1, divisions)
                endpoints = tuple(
                    tuple(
                        exact_charts[0][axis] + parameter * direction[axis]
                        for axis in range(2)
                    )
                    for parameter in interval
                )
                subboxes.append(
                    tuple(
                        (
                            outward(min(point[axis] for point in endpoints), -np.inf),
                            outward(max(point[axis] for point in endpoints), np.inf),
                        )
                        for axis in range(2)
                    )
                )
            subboxes_ = np.asarray(subboxes, dtype=np.float64).transpose(0, 2, 1)
            sub_lower, sub_upper = domain.patches[
                int(patches[edge_index])
            ].surface.derivative_bounds_batch(subboxes_, order=1)
            segment_direction = (
                direction[0] / divisions,
                direction[1] / divisions,
            )
            high = sum(
                (
                    _constant_metric_tangent_upper(
                        segment_direction, low, high_, matrix, budget, None
                    )
                    for low, high_ in zip(sub_lower, sub_upper, strict=True)
                ),
                Fraction(0),
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
            low = _fraction_sqrt_lower(chord_squared, budget)
        else:
            low = Fraction(0)
        results.append((outward(low, -np.inf), outward(high, np.inf)))
    work = box_count
    charge_native_geometry_queries(box_count, work_units=work)
    return np.asarray(results, dtype=np.float64), work


def _certify_surface_metric_arcs(
    domain: MeshingDomain,
    state: _SurfaceState,
    source: _SurfaceSource,
    /,
    *,
    maximum_work: int,
    maximum_pairs: int,
    features: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    generated_from: int,
    maximum_subdivisions: int = 4096,
) -> tuple[np.ndarray, int]:
    """Outward metric integrals of full source derivatives, never Gauss error estimates."""
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
            domain, state, source.metric[0], lower_required
        )
    pieces, work = _metric_arc_pieces(state, source, maximum_pairs)
    edges, _, _ = _edge_table(state)
    bounds = [[Fraction(0), Fraction(0)] for _ in edges]
    logarithms: dict[int, HermitianFunctionEnclosure] = {}
    budget = _COORDINATE_BUDGET.get()
    for vertex, tensor in enumerate(source.metric):
        if budget is None:
            logarithms[vertex] = hermitian_log_enclosure(tensor)
        else:
            with budget.temporary_scope():
                logarithms[vertex] = hermitian_log_enclosure(
                    tensor, coordinate_budget=budget
                )
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
                        outward(min(point[axis] for point in endpoints), -np.inf)
                        for axis in range(2)
                    ],
                    [
                        outward(max(point[axis] for point in endpoints), np.inf)
                        for axis in range(2)
                    ],
                ]
            )
        visits = len(pending)
        visited += visits
        work += visits
        if work > maximum_work or visited > maximum_subdivisions * edges.shape[0]:
            raise _source_location_budget(maximum_work, work)
        charge_native_geometry_queries(visits, work_units=visits)
        lower = np.empty((visits, 3, 2), dtype=np.float64)
        upper = np.empty_like(lower)
        patches = np.asarray([piece.patch for piece in pending], dtype=np.int32)
        for patch in np.unique(patches).tolist():
            selected = np.flatnonzero(patches == patch)
            low, high = domain.patches[patch].surface.derivative_bounds_batch(
                np.asarray(boxes, dtype=np.float64)[selected]
            )
            lower[selected], upper[selected] = low, high
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError(
                "Source metric arcs require finite authoritative derivative enclosures."
            )
        remaining = []
        for row, piece in enumerate(pending):
            if budget is None:
                metric, error = _arc_metric_enclosure(piece, source, logarithms)
                low, high = _arc_piece_bounds(
                    piece, lower[row], upper[row], metric, error
                )
            else:
                with budget.temporary_scope():
                    metric, error = _arc_metric_enclosure(piece, source, logarithms)
                    low, high = _arc_piece_bounds(
                        piece, lower[row], upper[row], metric, error
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
        [[outward(low, -np.inf), outward(high, np.inf)] for low, high in bounds],
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
) -> tuple[float, float]:
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(256, 128 * (64 + 2 * bigint_bytes(16 * 1075 + 64)))
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
            _fraction_sqrt_interval(3 * determinant_low, coordinate_budget=budget)[0]
            / denominator_high
        )
        minimum = max(0.0, outward(bound * condition_factor, -np.inf))
    if determinant_high >= 0 and denominator_low > 0:
        bound = (
            _fraction_sqrt_interval(3 * determinant_high, coordinate_budget=budget)[1]
            / denominator_low
        )
        maximum = min(1.0, outward(bound / condition_factor, np.inf))
    return minimum, maximum


def _certify_surface_metric_quality(
    domain: MeshingDomain,
    state: _SurfaceState,
    source: _SurfaceSource,
    /,
    *,
    minimum_quality: float,
    maximum_work: int,
) -> tuple[np.ndarray, int]:
    """Enclose the complete curved differential, never a frozen centroid."""
    from ..geometry._meshing_domain import _gram_spectrum_bounds

    budget = _COORDINATE_BUDGET.get()
    constant = np.array_equal(
        source.metric, np.broadcast_to(source.metric[0], source.metric.shape)
    )
    factor = Fraction(1)
    if not constant:
        spectra = tuple(_gram_spectrum_bounds(value) for value in source.metric)
        low, high = min(value[0] for value in spectra), max(value[1] for value in spectra)
        if low <= 0:
            raise ValueError(
                "Original surface SPD supports lack a certified positive spectral lower bound."
            )
        factor = (
            _fraction_sqrt_interval(Fraction(low), coordinate_budget=budget)[0]
            / _fraction_sqrt_interval(Fraction(high), coordinate_budget=budget)[1]
        )
    tensor = source.metric[0] if constant else np.eye(3, dtype=np.float64)
    matrix = tuple(tuple(Fraction(float(value)) for value in row) for row in tensor)
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
                + (len(pending) + 1) * 6 * (56 + 2 * bigint_bytes(queue_bits))
                + 40 * (len(accepted) + 1)
            )
            if budget is not None:
                budget.reserve(0, max(0, upper - queue_upper))
            queue_upper = max(queue_upper, upper)
            panel = pending.pop()
            work += 1
            if work > maximum_work:
                raise _source_location_budget(maximum_work, work)
            box = np.asarray(
                [
                    [
                        outward(min(point[axis] for point in panel), -np.inf)
                        for axis in range(2)
                    ],
                    [
                        outward(max(point[axis] for point in panel), np.inf)
                        for axis in range(2)
                    ],
                ]
            )
            charge_native_geometry_queries(1, work_units=1)
            lower, upper = domain.patches[int(patch)].surface.derivative_bounds_batch(
                box[None]
            )
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                raise ValueError(
                    "Whole-cell surface quality requires finite original derivative enclosures."
                )
            with budget.temporary_scope() if budget is not None else nullcontext():
                minimum, maximum = _surface_quality_box_bounds(
                    lower[0], upper[0], frame, matrix, factor
                )
            if maximum < minimum_quality:
                refused = True
                break
            if minimum >= minimum_quality:
                accepted.append(minimum)
                continue
            a, b, c = panel
            with exact_charge(6):
                queue_bits = max(
                    queue_bits,
                    max(
                        max(value.numerator.bit_length(), value.denominator.bit_length())
                        for point in panel
                        for value in point
                    )
                    + 2,
                )
            # Actual UV inputs are dyadic binary64 values; exact midpoint
            # denominators grow by at most one bit. Admit the live frontier
            # before constructing its children, not a rounded carrier surrogate.
            upper = (
                256
                + (len(pending) + 5) * 6 * (56 + 2 * bigint_bytes(queue_bits))
                + 40 * (len(accepted) + 1)
            )
            if budget is not None:
                budget.reserve(0, max(0, upper - queue_upper))
            queue_upper = max(queue_upper, upper)
            with exact_charge(18):
                ab = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                bc = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
                ca = ((c[0] + a[0]) / 2, (c[1] + a[1]) / 2)
            pending.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        if not refused:
            result[cell] = min(accepted)
    return result, work


def _measure(
    domain: MeshingDomain, state: _SurfaceState, /
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
        interpolate_mesh_metric(
            np.broadcast_to(state.metric[edges][:, None], (edges.shape[0], 8, 2, 3, 3)),
            np.broadcast_to(
                np.stack((1.0 - _PARAMETERS, _PARAMETERS), axis=1), (edges.shape[0], 8, 2)
            ),
        ),
        dtype=np.float64,
    )
    tangent = np.zeros((edges.shape[0], 8, 3), dtype=np.float64)
    charge_native_geometry_queries(edges.shape[0] * 8)
    for patch in np.unique(patches).tolist():
        selected = patches == patch
        differential = np.asarray(
            surface_differential(
                domain.patches[patch].surface,
                jnp.asarray(charts[selected].reshape((-1, 2)), dtype=jnp.float64),
            ),
            dtype=np.float64,
        ).reshape((-1, 8, 3, 2))
        tangent[selected] = (differential @ velocity[selected, None, :, None])[..., 0]
    speed = np.sqrt(np.sum(tangent * (samples @ tangent[..., None])[..., 0], axis=-1))
    lengths = speed @ _WEIGHTS
    quality = _mapped_quality(
        domain, state.metric, state.cells, state.patches, state.charts
    )
    fidelity = np.zeros((state.cells.shape[0],), dtype=np.float64)
    for patch in np.unique(state.patches).tolist():
        selected = state.patches == patch
        fidelity[selected] = domain.interpolation_bounds(patch, state.charts[selected])
    return lengths, quality, fidelity


def _legal(
    domain: MeshingDomain,
    cells: np.ndarray,
    patches: np.ndarray,
    charts: np.ndarray,
    points: np.ndarray,
    maximum_fidelity: float,
    /,
) -> bool:
    predicate = orient2d(
        charts[:, 0], charts[:, 1], charts[:, 2], mode=PredicateMode.EXACT
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
    signs = orient3d(
        physical[:, 0],
        physical[:, 1],
        physical[:, 2],
        physical[:, 0] + normals,
        mode=PredicateMode.EXACT,
    )
    if not np.all(
        regular
        & np.asarray(signs.certain)
        & (np.asarray(signs.signs) == PredicateSign.POSITIVE)
    ):
        return False
    for patch in np.unique(patches).tolist():
        if np.any(
            domain.interpolation_bounds(patch, charts[patches == patch])
            > maximum_fidelity
        ):
            return False
    return True


def _admit_material_carrier(
    state: _SurfaceState,
    charts: np.ndarray,
    cells: int,
    /,
    *,
    midpoint: bool = False,
) -> None:
    """Keep one growing live/rebuild material bank under the original ledger."""
    input_bits = exact_bits(charts)
    bits = max(
        state.material_fraction_bits, 2 * input_bits + 4 if midpoint else input_bits
    )
    upper = 256 + 12 * cells * (56 + 2 * bigint_bytes(bits))
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(0, max(0, upper - state.material_workspace_bytes_upper))
    state.material_fraction_bits = bits
    state.material_workspace_bytes_upper = max(
        state.material_workspace_bytes_upper, upper
    )


def _material_legal(
    domain: MeshingDomain, patches: np.ndarray, charts: np.ndarray, /
) -> bool:
    """Exact material orientation through the owning integer-bank predicate."""
    bits = exact_bits(charts)
    with exact_charge(
        12 * charts.shape[0], 18 * charts.shape[0] * bigint_bytes(7 * bits + 16)
    ):
        integers = np.empty(charts.shape, dtype=object)
        for row, triangle in enumerate(charts):
            denominator = lcm(*(value.denominator for value in triangle.flat))
            for slot, point in enumerate(triangle):
                for axis, value in enumerate(point):
                    integers[row, slot, axis] = value.numerator * (
                        denominator // value.denominator
                    )
        predicate = orient2d(
            integers[:, 0], integers[:, 1], integers[:, 2], mode=PredicateMode.EXACT
        )
    return bool(
        np.all(np.asarray(predicate.certain))
        and np.all(
            np.asarray(predicate.signs)
            == np.asarray([domain.patches[int(patch)].orientation for patch in patches])
        )
    )


def _replace(
    state: _SurfaceState,
    removed: np.ndarray,
    cells: np.ndarray,
    patches: np.ndarray,
    charts: np.ndarray,
    classes: np.ndarray,
    parents: list[set[int]],
    kind: EntityLineageKind,
    /,
    *,
    material_charts: np.ndarray | None = None,
    cell_ids: np.ndarray | None = None,
) -> None:
    keep = np.ones((state.cells.shape[0],), dtype=np.bool_)
    keep[removed] = False
    if state.material_charts is not None and material_charts is None:
        raise ValueError(
            "A material topology edit must retain its complete exact chart carrier."
        )
    ids = (
        _fresh_identifiers(state.next_cell_id, cells.shape[0])
        if cell_ids is None
        else cell_ids
    )
    if state.cell_kinds is None:
        state.cell_kinds = {}
    kind = _lineage_operation_kind(
        kind,
        tuple(
            state.cell_kinds.get(int(identifier), int(EntityLineageKind.PRESERVED))
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
    """Select the actual coedge use, never a nearest periodic representative."""
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


def _split(
    domain: MeshingDomain,
    state: _SurfaceState,
    edge: np.ndarray,
    new_metric: np.ndarray,
    /,
    *,
    source_curve: bool = False,
) -> bool:
    rows = np.flatnonzero(np.sum(np.isin(state.cells, edge), axis=1) == 2)
    if rows.size == 0:
        return False
    budget = current_native_execution_budget()
    if budget is not None:
        budget.admit_cavity(int(rows.size))
    token = None
    chart_midpoints = np.asarray(
        [
            np.mean(state.charts[row, np.isin(state.cells[row], edge)], axis=0)
            for row in rows
        ],
        dtype=np.float64,
    )
    witness, strata = state.curve_witness, state.vertex_strata
    coordinate_budget = _COORDINATE_BUDGET.get()
    if strata is not None and coordinate_budget is not None:
        # A live token and its in-flight replacement each carry a conservative
        # 512-byte CPython upper bound, not native measured storage. Rejected
        # attempts reuse the admitted slot rather than accumulating dead tokens.
        required = int(state.points.shape[0]) + 1
        coordinate_budget.reserve(
            1, 1024 * max(0, required - state.curve_workspace_vertices)
        )
        state.curve_workspace_vertices = max(state.curve_workspace_vertices, required)
    if source_curve and witness is not None:
        if strata is None:
            raise ValueError("An actual curve edit lacks its retained vertex strata.")
        tokens = [strata[int(vertex)] for vertex in edge]
        try:
            curve, left, right = witness.edge_interval(
                np.asarray([value[0] for value in tokens]),
                np.asarray([value[1] for value in tokens]),
                np.asarray([value[2] for value in tokens]),
            )
        except ValueError:
            return False
        parameter = (left + right) / 2
        point, target_uses = witness.evaluate_curve(curve, parameter)
        _, left_uses = witness.evaluate_curve(curve, left)
        _, right_uses = witness.evaluate_curve(curve, right)
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
                with exact_charge(8):
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
        domain, state.patches[owner], material_charts
    ):
        return False
    if not _legal(domain, children, state.patches[owner], child_charts, points, np.inf):
        return False
    vertex_id = _fresh_identifiers(state.next_vertex_id, 1)[0]
    child_ids = _fresh_identifiers(state.next_cell_id, children.shape[0])
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
        EntityLineageKind.REFINED_FROM,
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


def _collapse(
    domain: MeshingDomain,
    state: _SurfaceState,
    removed: int,
    kept: int,
    fixed: np.ndarray,
    features: set[tuple[int, int]],
    minimum_quality: float,
    maximum_fidelity: float,
    /,
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
    budget = current_native_execution_budget()
    if budget is not None:
        budget.admit_cavity(int(np.union1d(rows, keep_rows).size))
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
        )
        if not _material_legal(domain, state.patches[owners], material_charts):
            return False
    if proposed.size == 0 or not _legal(
        domain, proposed, state.patches[owners], charts, state.points, maximum_fidelity
    ):
        return False
    if (
        np.min(
            _mapped_quality(domain, state.metric, proposed, state.patches[owners], charts)
        )
        < minimum_quality
    ):
        return False
    # All new metric edges obey the explicit collapse ceiling; next pass refines.
    trial = _SurfaceState(
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
    if np.max(_measure(domain, trial)[0], initial=0.0) > 2.0:
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
        EntityLineageKind.COLLAPSED_INTO,
        material_charts=material_charts,
    )
    if state.vertex_successor is None:
        state.vertex_successor = {}
    state.vertex_successor[removed] = kept
    return True


def _flip(
    domain: MeshingDomain,
    state: _SurfaceState,
    edge: np.ndarray,
    features: set[tuple[int, int]],
    maximum_fidelity: float,
    /,
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
    budget = current_native_execution_budget()
    if budget is not None:
        budget.admit_cavity(int(rows.size))
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
    predicate = orient2d(
        charts[:, 0], charts[:, 1], charts[:, 2], mode=PredicateMode.EXACT
    )
    signs = np.asarray(predicate.signs)
    if (
        not np.all(np.asarray(predicate.certain))
        or signs[0] == PredicateSign.ZERO
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
        _admit_material_carrier(state, material_charts, int(state.cells.shape[0]))
        if not _material_legal(domain, patch, material_charts):
            return False
    if not _legal(domain, proposed, patch, charts, state.points, maximum_fidelity):
        return False
    before = np.min(_mapped_quality(domain, state.metric, old, patch, state.charts[rows]))
    after = np.min(_mapped_quality(domain, state.metric, proposed, patch, charts))
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
        EntityLineageKind.SWAPPED_FROM,
        material_charts=material_charts,
    )
    return True


def _source_curve_slide_charts(
    domain: MeshingDomain,
    state: _SurfaceState,
    vertices: np.ndarray,
    patches: np.ndarray,
    seeds: np.ndarray,
    features: set[tuple[int, int]],
    feature_codes: dict[tuple[int, int], int],
    blocked: set[tuple[int, int]],
    /,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int, np.ndarray]]]:
    witness, strata = state.curve_witness, state.vertex_strata
    if witness is None or strata is None:
        raise ValueError("An actual CAD curve slide lacks its source-bound strata.")
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
        try:
            owner, left, right = witness.edge_interval(
                np.asarray([value[0] for value in neighbor_tokens]),
                np.asarray([value[1] for value in neighbor_tokens]),
                np.asarray([value[2] for value in neighbor_tokens]),
            )
        except ValueError:
            continue
        if owner != curve or not min(left, right) <= current[0] <= max(left, right):
            continue
        parameter = (left + right) / 2
        point, target_uses = witness.evaluate_curve(curve, parameter)
        _, original_uses = witness.evaluate_curve(curve, float(current[0]))
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
    domain: MeshingDomain,
    state: _SurfaceState,
    source: _SurfaceSource,
    vertices: np.ndarray,
    patches: np.ndarray,
    seeds: np.ndarray,
    displacement: np.ndarray,
    features: set[tuple[int, int]],
    feature_codes: dict[tuple[int, int], int],
    blocked: set[tuple[int, int]],
    /,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int, np.ndarray]]]:
    """Use actual CAD strata when supplied; otherwise retain mesh-chart features."""
    if state.curve_witness is not None:
        return _source_curve_slide_charts(
            domain, state, vertices, patches, seeds, features, feature_codes, blocked
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
        signs = orient2d(uv[0], uv[1], seeds[index], mode=PredicateMode.EXACT)
        permitted[index] = (
            bool(np.asarray(signs.certain))
            and np.asarray(signs.signs) == PredicateSign.ZERO
        )
    tangent = np.zeros((vertices.size, 3), dtype=np.float64)
    charge_native_geometry_queries(vertices.size)
    for patch in np.unique(patches).tolist():
        selected = patches == patch
        differential = np.asarray(
            surface_differential(
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
    signs = orient2d(
        endpoints[:, 0], endpoints[:, 1], proposals, mode=PredicateMode.EXACT
    )
    permitted &= np.asarray(signs.certain) & (
        np.asarray(signs.signs) == PredicateSign.ZERO
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


def _relocate(
    domain: MeshingDomain,
    state: _SurfaceState,
    fixed: np.ndarray,
    features: set[tuple[int, int]],
    maximum_fidelity: float,
    budget: int,
    maximum_pairs: int,
    source: _SurfaceSource,
    feature_codes: dict[tuple[int, int], int],
    blocked: set[tuple[int, int]],
    minimum_quality: float,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    selected_vertices: frozenset[int] | None = None,
) -> tuple[int, int]:
    edges, _, _ = _edge_table(state)
    lengths, _, _ = _measure(domain, state)
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
    if not vertices or budget == 0:
        return 0, 0
    selected = np.asarray(vertices[:budget], dtype=np.int32)
    charge_native_geometry_queries(0, work_units=int(selected.size))
    owner = np.asarray(owners[:budget], dtype=np.int64)
    displacement = total[selected] / np.maximum(degree[selected, None], 1.0)
    seed_charts = np.asarray(seeds[:budget], dtype=np.float64)
    slide = np.asarray(sliding[:budget], dtype=np.bool_)
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
    sampling = _chart_stencil(
        source.cells,
        source.charts,
        source.patches,
        source.cell_ids,
        source.vertex_ids,
        state.patches[owner],
        query_material,
        maximum_pairs,
    )
    source_rows = key_rows(
        source.vertex_ids[:, None],
        sampling[0].reshape((-1, 1)),
    ).reshape(sampling[0].shape)
    proposal_metrics = np.asarray(
        interpolate_mesh_metric(source.metric[source_rows], sampling[1]),
        dtype=np.float64,
    )
    applied = 0
    for index, vertex in enumerate(selected.tolist()):
        if not proposal_valid[index]:
            continue
        rows = np.flatnonzero(np.any(state.cells == vertex, axis=1))
        native_budget = current_native_execution_budget()
        if native_budget is not None:
            native_budget.admit_cavity(int(rows.size))
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
        ):
            continue
        before = np.min(
            _mapped_quality(
                domain,
                state.metric,
                state.cells[rows],
                state.patches[rows],
                state.charts[rows],
            )
        )
        metrics = state.metric.copy()
        metrics[vertex] = proposal_metrics[index]
        after = np.min(
            _mapped_quality(
                domain, metrics, state.cells[rows], state.patches[rows], charts
            )
        )
        local = _SurfaceState(
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
        trial = _SurfaceState(
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
            _unit_defect(_measure(domain, trial)[0])
            < _unit_defect(_measure(domain, local)[0]) - 1.0e-12
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
                _lineage_operation_kind(
                    EntityLineageKind.RELOCATED,
                    (state.cell_kinds.get(identifier, int(EntityLineageKind.PRESERVED)),),
                )
            )
        applied += 1
    return applied, selected.size


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
        tuple[int, tuple[tuple[int, int], ...]],
        tuple[np.ndarray, np.ndarray, int],
    ]
    | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native routing with exact material support, without rounded containment."""
    sources = np.zeros((query_patches.size, 3), dtype=np.int64)
    weights = np.zeros((query_patches.size, 3), dtype=np.float64)
    selected_ids = np.full(query_patches.size, np.iinfo(np.int64).max, dtype=np.int64)
    work = 0
    budget = _COORDINATE_BUDGET.get()
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
    for patch in np.unique(query_patches):
        candidates = np.flatnonzero(source_patches == patch)
        queries = np.flatnonzero(
            (query_patches == patch) & (selected_ids == np.iinfo(np.int64).max)
        )
        corners = source_charts[candidates]
        source_lower = np.asarray(
            [
                [outward(min(triangle[:, axis]), -np.inf) for axis in range(2)]
                for triangle in corners
            ]
        )
        source_upper = np.asarray(
            [
                [outward(max(triangle[:, axis]), np.inf) for axis in range(2)]
                for triangle in corners
            ]
        )
        tree = prepare_bvh(source_lower, source_upper, dtype=jnp.float64)
        lower = np.asarray(
            [[outward(value, -np.inf) for value in query_charts[row]] for row in queries]
        )
        upper = np.asarray(
            [[outward(value, np.inf) for value in query_charts[row]] for row in queries]
        )
        query_tree = prepare_bvh(lower, upper, dtype=jnp.float64)
        for cells, rows in bvh_overlap_pair_blocks(
            tree, query_tree, include_touching=True
        ):
            work += cells.size
            if work > maximum_pairs:
                raise _source_location_budget(maximum_pairs, work)
            with budget.temporary_scope() if budget is not None else nullcontext():
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
                        prepared = prepare_exact_small_linear_actions(
                            ((b[0] - a[0], c[0] - a[0]), (b[1] - a[1], c[1] - a[1])),
                            ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
                            coordinate_budget=budget,
                        )
                        if prepared.actions is None:
                            raise ValueError(
                                "The original material background contains a singular chart cell."
                            )
                        inverse = prepared.actions
                        inverses[cell] = inverse
                    point = _exact_uv(query_charts[query])
                    delta = point[0] - a[0], point[1] - a[1]
                    if budget is not None:
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
                        if budget is not None:
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
    if np.any(selected_ids == np.iinfo(np.int64).max):
        raise MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            "A target material functional lacks exact original chart coverage.",
            stage=MeshingStageKind.LINEAGE_CONSTRUCTION.value,
        )
    return sources, weights, np.ones(weights.shape, dtype=np.bool_)


def _chart_stencil(
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
        tuple[int, tuple[tuple[int, int], ...]],
        tuple[np.ndarray, np.ndarray, int],
    ]
    | None = None,
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
                raise _source_location_budget(maximum_pairs, work)
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
            original = orient2d(
                triangle[:, 0], triangle[:, 1], triangle[:, 2], mode=PredicateMode.EXACT
            )
            for slot in range(3):
                vertices = [triangle[:, index] for index in range(3)]
                vertices[slot] = target
                predicate = orient2d(*vertices, mode=PredicateMode.EXACT)
                signs = np.asarray(predicate.signs)
                valid &= np.asarray(predicate.certain) & (
                    (signs == np.asarray(original.signs)) | (signs == PredicateSign.ZERO)
                )
                coefficients[signs == PredicateSign.ZERO, slot] = 0.0
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
        raise MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            "A surface target chart lacks exact source-chart coverage.",
            stage=MeshingStageKind.LINEAGE_CONSTRUCTION.value,
        )
    weights = np.maximum(weights, 0.0)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return sources, weights, np.ones(weights.shape, dtype=np.bool_)


def _chart_exact_weights(
    source_cells: np.ndarray,
    source_charts: np.ndarray,
    source_vertex_ids: np.ndarray,
    stencil_sources: np.ndarray,
    query_charts: np.ndarray,
    /,
) -> tuple[ConstructionPointKey, ...]:
    """One canonical original-chart functional for native and source transport."""
    selected = key_rows(
        np.sort(source_vertex_ids[source_cells], axis=1), np.sort(stencil_sources, axis=1)
    )
    if np.any(selected < 0):
        raise ValueError("An actual UV construction lacks its original source triangle.")
    keys: dict[int, ConstructionPointKey] = {}
    budget = _COORDINATE_BUDGET.get()
    for row in np.unique(selected):
        queries = np.flatnonzero(selected == row)
        if budget is not None:
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
        solved = prepare_exact_small_linear_actions(
            matrix, right, coordinate_budget=budget
        )
        if not solved.successful or solved.actions is None:
            raise ValueError(
                f"An original UV triangle has a {solved.status} chart frame of rank {solved.rank}."
            )
        for index, u, v in zip(queries, *solved.actions, strict=True):
            weights = (1 - u - v, u, v)
            if min(weights) < 0:
                raise ValueError(
                    "Exact UV construction leaves its original source triangle."
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
    state: _SurfaceState,
    live: np.ndarray,
    maximum_pairs: int,
    /,
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
    return _chart_stencil(
        source_cells,
        source_charts,
        source_patches,
        source_cell_ids,
        source_vertex_ids,
        patches,
        charts,
        maximum_pairs,
    )


def _surface_metric_source_cover(
    source: CellMeshingResult, /
) -> MeshingDomainBoundarySource:
    """Recover the unchanged trim/span authority across accepted chart epochs."""
    from ..geometry._surface_source_support import (
        SurfaceNativeRestrictionBoundarySource,
    )
    from ._surface_association_transfer import (
        SurfaceChartBoundarySource,
        SurfaceSubdivisionBoundarySource,
    )

    if source.certification is None:
        raise ValueError(
            "UV metric adaptation requires its original source-chart certificate."
        )
    retained = source.certification.request.source
    root = retained.root if isinstance(retained, SurfaceChartBoundarySource) else retained
    boundary = root.root if isinstance(root, SurfaceSubdivisionBoundarySource) else root
    if isinstance(boundary, SurfaceNativeRestrictionBoundarySource):
        boundary = boundary.root
    if not isinstance(boundary, MeshingDomainBoundarySource):
        raise ValueError(
            "UV metric adaptation lacks its original continuous trim/chart source owner."
        )
    return boundary


def prepare_surface_metric_source(
    source: CellMeshingResult,
    transfer: SurfaceAssociationTransfer,
    /,
) -> SurfaceChartWitness:
    """Recover source-bound chart uses without inferring seams from coordinates."""
    from ._surface_association_transfer import _source_proof, SurfaceChartBoundarySource

    transfer.source_associations(source)
    transfer.support.require_current()
    retained = (
        None if source.certification is None else source.certification.request.source
    )
    if isinstance(retained, SurfaceChartBoundarySource):
        retained.require_current()
        if not isinstance(retained.deformation, PreparedSurfaceChartDeformation):
            raise TypeError(
                "A radial sphere source must use its actual SphereMaterialCellAtlas owner."
            )
        witness = retained.deformation.target_witness
        if (
            witness.geometry_id != cell_geometry_id(source.geometry)
            or witness.topology_id != source.mesh.topology_id
        ):
            raise ValueError("The retained source-chart witness is stale.")
        return witness
    proof = _source_proof(transfer.support, source, transfer.receipts)
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in source.mesh.blocks]
    )
    rows = key_rows(proof.cell_ids[:, None], ids[:, None])
    if np.any(rows < 0):
        raise ValueError("Original source support omits current surface cells.")
    patches = proof.patches[rows]
    return SurfaceChartWitness(
        geometry_id=cell_geometry_id(source.geometry),
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


def execute_surface_metric_adaptation(
    mesh: CellMesh,
    metric: ArrayLike,
    domain: MeshingDomain,
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
    controller: _SurfaceMetricControllerProtocol | None = None,
    curve_witness: PreparedSurfaceCurveWitness | None = None,
    source_material_charts: np.ndarray | None = None,
) -> SurfaceMetricOutcome:
    """Remesh exact parametric patches with chart-link and source-fidelity bounds.

    Gauss lengths propose edits only. COMPLETE requires outward integration of
    authoritative source interval jets in the original, exactly covered
    log-Euclidean chart metric background. Continuous fidelity, complete target
    coordinate-map reconstruction and global embedding remain separate owning
    certificates required at publication.
    """
    if not isinstance(mesh, CellMesh) or not isinstance(domain, MeshingDomain):
        raise TypeError("mesh and domain must be CellMesh and MeshingDomain.")
    domain.require_current(source_id, source_revision)
    if mesh.ambient_dimension != 3 or any(
        block.cell_kind != "triangle" for block in mesh.blocks
    ):
        raise ValueError("Surface metric adaptation requires embedded triangles.")
    if (
        not np.isfinite(maximum_fidelity)
        or maximum_fidelity < 0.0
        or not 0.0 <= minimum_metric_quality <= 1.0
    ):
        raise ValueError("Fidelity and metric-quality bounds are invalid.")
    budgets = (
        maximum_passes,
        maximum_vertices,
        maximum_cells,
        maximum_operations,
        maximum_location_pairs,
        maximum_arc_work_units,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in budgets
    ):
        raise ValueError("Surface work budgets must be nonnegative integers.")
    _stage_source_limits(mesh, maximum_vertices, maximum_cells)
    if not isinstance(topology_operations, bool) or not isinstance(relocation, bool):
        raise TypeError("Operation controls must be bool.")
    if record_phase is not None and not callable(record_phase):
        raise TypeError("record_phase must be a phase recorder callable.")
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
            "Explicit patch identities and per-cell charts must align with cells."
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
            "Source mesh vertices must lie on their explicitly associated patches."
        )
    values = np.asarray(metric, dtype=np.float64)
    if values.shape != (points.shape[0], 3, 3):
        raise ValueError("metric must carry one 3D SPD tensor per vertex.")
    with measure_phase(record_phase, "metric_preparation"):
        properties = _tensor_properties(values)
        if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
            raise ValueError(
                "Surface metrics require finite Hermitian positive-definite physical tensors."
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
            "Cell classifications and constraint masks must align with source entities."
        )
    if np.any(classes < 0):
        raise ValueError("Surface cell classifications must be nonnegative.")
    _, classes = np.unique(classes, return_inverse=True)
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    coordinate_budget = _COORDINATE_BUDGET.get()
    with (
        coordinate_budget.temporary_scope()
        if coordinate_budget is not None
        else nullcontext()
    ):
        if source_material_charts is None:
            if coordinate_budget is not None:
                coordinate_budget.reserve(
                    int(charts.size),
                    int(charts.size) * (56 + 2 * bigint_bytes(1075)) + int(charts.nbytes),
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
                    "Source material charts require complete original exact Fraction coordinates."
                )
            if coordinate_budget is not None:
                coordinate_budget.reserve(0, int(supplied.nbytes) + 128)
            material_source = supplied.copy()
        if coordinate_budget is not None:
            coordinate_budget.retain_basis((material_source, tuple(material_source.flat)))
    source = _SurfaceSource(
        cells, material_source, patches, cell_ids, vertex_ids, values, charts
    )
    if controller is not None:
        if (
            controller.source.mesh.mesh_id != mesh.mesh_id
            or controller.witness.domain_id != domain.domain_id
        ):
            raise ValueError(
                "The UV controller is stale for the actual source mesh/domain."
            )
        controller.background = source
    state = _SurfaceState(
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
    if coordinate_budget is not None:
        coordinate_budget.reserve(0, int(material_source.nbytes) + 128)
    state.material_charts = material_source.copy()
    _admit_material_carrier(state, state.material_charts, int(cells.shape[0]))
    if not _material_legal(domain, patches, state.material_charts):
        raise ValueError(
            "The original material charts must retain their exact source orientation."
        )
    if curve_witness is not None:
        curve_witness.require_bound(mesh)
        if curve_witness.support.domain.domain_id != domain.domain_id:
            raise ValueError(
                "The actual curve witness must bind the metric source domain."
            )
        coordinate_budget = _COORDINATE_BUDGET.get()
        if coordinate_budget is not None:
            coordinate_budget.reserve(int(vertex_ids.size), 1024 * int(vertex_ids.size))
        state.curve_witness = curve_witness
        state.curve_workspace_vertices = int(vertex_ids.size)
        state.vertex_strata = [
            (int(dimension), int(index), np.asarray(parameters, dtype=np.float64).copy())
            for dimension, index, parameters in zip(
                np.asarray(curve_witness.source_dimensions),
                np.asarray(curve_witness.source_indices),
                np.asarray(curve_witness.source_parameters),
                strict=True,
            )
        ]
    if not _legal(domain, cells, patches, charts, points, np.inf):
        raise ValueError(
            "The source charts must be exactly oriented, regular surface cells."
        )
    edges, _, inverse = _edge_table(state)
    rows = key_rows(entity_keys(mesh, 1), np.sort(vertex_ids[edges], axis=1))
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
            "Surface edge classes must be nonnegative integer source-entity labels."
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
            "Periodic UV adaptation requires its actual source-owned orbit controller."
        )
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    feature_identity = {
        tuple(edge): int(feature_codes[rows[index]])
        if feature_codes[rows[index]] > 0
        else -int(edge_ids[rows[index]]) - 1
        for index, edge in enumerate(edges.tolist())
        if tuple(edge) in features
    }
    counts = [0, 0, 0, 0]
    rejected, work, passes = 0, 0, 0
    status = MetricRemeshingStatus.PASS_LIMIT
    material_stencil_cache: (
        dict[
            tuple[int, tuple[tuple[int, int], ...]],
            tuple[np.ndarray, np.ndarray, int],
        ]
        | None
    ) = {} if state.material_charts is not None else None
    for passes in range(1, maximum_passes + 1):
        lengths, quality, fidelity = _measure(domain, state)
        edges, _, cell_edges = _edge_table(state)
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
            status = MetricRemeshingStatus.COMPLETE
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
                status = MetricRemeshingStatus.RESOURCE_LIMIT
                break
            _, first, _ = _edge_table(state)
            owner, local = first // 3, first % 3
            bank = (
                state.charts if state.material_charts is None else state.material_charts
            )
            midpoint_charts = np.mean(bank[owner[:, None], _LOCAL_EDGES[local]], axis=1)
            sampling = _chart_stencil(
                cells,
                source.charts,
                patches,
                cell_ids,
                vertex_ids,
                state.patches[owner],
                midpoint_charts,
                maximum_location_pairs,
                material_cache=material_stencil_cache,
            )
            sample_rows = key_rows(
                vertex_ids[:, None], sampling[0].reshape((-1, 1))
            ).reshape(sampling[0].shape)
            midpoint_metrics = np.asarray(
                interpolate_mesh_metric(values[sample_rows], sampling[1]),
                dtype=np.float64,
            )
            split_admissible = np.ones((edges.shape[0],), dtype=np.bool_)
            if state.curve_witness is not None and state.vertex_strata is not None:
                for edge_index, candidate_edge in enumerate(edges):
                    key = tuple(map(int, candidate_edge))
                    if key not in features:
                        continue
                    tokens = [
                        state.vertex_strata[int(vertex)] for vertex in candidate_edge
                    ]
                    try:
                        curve, left, right = state.curve_witness.edge_interval(
                            np.asarray([value[0] for value in tokens]),
                            np.asarray([value[1] for value in tokens]),
                            np.asarray([value[2] for value in tokens]),
                        )
                        point, target_uses = state.curve_witness.evaluate_curve(
                            curve, (left + right) / 2
                        )
                        _, left_uses = state.curve_witness.evaluate_curve(curve, left)
                        _, right_uses = state.curve_witness.evaluate_curve(curve, right)
                        candidate_rows = np.flatnonzero(
                            np.sum(np.isin(state.cells, candidate_edge), axis=1) == 2
                        )
                        candidate_charts = []
                        for candidate_row in candidate_rows:
                            anchors = tuple(
                                (
                                    state.charts[
                                        candidate_row,
                                        np.flatnonzero(
                                            state.cells[candidate_row] == vertex
                                        )[0],
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
                    except ValueError:
                        split_admissible[edge_index] = False
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
                    status = MetricRemeshingStatus.RESOURCE_LIMIT
                    break
                edge = edges[index]
                if any(int(vertex) in split_vertices for vertex in edge):
                    continue
                if tuple(edge.tolist()) in blocked:
                    rejected += 1
                    continue
                work += 1
                charge_native_geometry_queries(0, work_units=1)
                vertex = state.points.shape[0]
                with measure_phase(record_phase, "metric_split"):
                    if controller is None:
                        changed = _split(
                            domain,
                            state,
                            edge,
                            midpoint_metrics[index],
                            source_curve=tuple(edge.tolist()) in features,
                        )
                    else:
                        staged = controller.stage(
                            state,
                            "split",
                            tuple(map(int, edge)),
                            fixed,
                            features,
                            feature_identity,
                            blocked,
                        )
                        changed = staged is not None
                        if staged is not None:
                            state = staged
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
                    copies = 1 if controller is None else controller.operation_count
                    counts[0] += copies
                    applied += copies
                    split_vertices.update(map(int, edge))
                else:
                    rejected += 1
            if status is MetricRemeshingStatus.RESOURCE_LIMIT:
                break
            edges, _, _ = _edge_table(state)
            lengths, _, _ = _measure(domain, state)
            for index in np.argsort(lengths, kind="stable"):
                if applied:
                    break
                if lengths[index] >= _LOWER or work >= maximum_operations:
                    break
                if np.any(edges[index] >= original_vertex_count):
                    continue
                work += 1
                charge_native_geometry_queries(0, work_units=1)
                first, second = edges[index].tolist()
                with measure_phase(record_phase, "metric_collapse"):
                    if controller is None:
                        changed = _collapse(
                            domain,
                            state,
                            first,
                            second,
                            fixed,
                            features,
                            minimum_metric_quality,
                            maximum_fidelity,
                        ) or _collapse(
                            domain,
                            state,
                            second,
                            first,
                            fixed,
                            features,
                            minimum_metric_quality,
                            maximum_fidelity,
                        )
                    else:
                        staged = controller.stage(
                            state,
                            "collapse",
                            (first, second),
                            fixed,
                            features,
                            feature_identity,
                            blocked,
                        )
                        changed = staged is not None
                        if staged is not None:
                            state = staged
                if changed:
                    copies = 1 if controller is None else controller.operation_count
                    counts[1] += copies
                    applied += copies
                else:
                    rejected += 1
            edges, _, _ = _edge_table(state)
            for edge in edges:
                if applied:
                    break
                if work >= maximum_operations:
                    break
                work += 1
                charge_native_geometry_queries(0, work_units=1)
                with measure_phase(record_phase, "metric_reconnection"):
                    if controller is None:
                        changed = _flip(domain, state, edge, features, maximum_fidelity)
                    else:
                        staged = controller.stage(
                            state,
                            "flip",
                            tuple(map(int, edge)),
                            fixed,
                            features,
                            feature_identity,
                            blocked,
                        )
                        changed = staged is not None
                        if staged is not None:
                            state = staged
                if changed:
                    copies = 1 if controller is None else controller.operation_count
                    counts[2] += copies
                    applied += copies
        if relocation and work < maximum_operations:
            with measure_phase(record_phase, "metric_relocation"):
                if controller is None:
                    changes, attempts = _relocate(
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
                    )
                else:
                    changes, attempts = 0, 0
                    for vertex in np.unique(state.cells).tolist():
                        if attempts >= maximum_operations - work:
                            break
                        attempts += 1
                        staged = controller.stage(
                            state,
                            "relocate",
                            (vertex,),
                            fixed,
                            features,
                            feature_identity,
                            blocked,
                        )
                        if staged is not None:
                            state = staged
                            changes += controller.operation_count
            counts[3] += changes
            work += attempts
            applied += changes
        if applied == 0:
            status = (
                MetricRemeshingStatus.RESOURCE_LIMIT
                if work >= maximum_operations
                else MetricRemeshingStatus.STALLED
            )
            if (
                not topology_operations
                and work < maximum_operations
                and np.all(quality >= minimum_metric_quality)
                and np.all(fidelity <= maximum_fidelity)
            ):
                status = MetricRemeshingStatus.COMPLETE
            break
    lengths, quality, fidelity = _measure(domain, state)
    edges, _, _ = _edge_table(state)
    if not np.all(np.isfinite(fidelity)):
        raise ValueError("The source lacks a continuous interpolation fidelity bound.")
    if (
        status is not MetricRemeshingStatus.RESOURCE_LIMIT
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
        status = MetricRemeshingStatus.COMPLETE
    arc_bounds, arc_edge_ids, arc_binding = None, None, None
    if status is MetricRemeshingStatus.COMPLETE:
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
            )
        work += arc_work
        coordinate_budget = _COORDINATE_BUDGET.get()
        with measure_phase(record_phase, "metric_preparation"):
            with (
                coordinate_budget.temporary_scope()
                if coordinate_budget is not None
                else nullcontext()
            ):
                quality, quality_work = _certify_surface_metric_quality(
                    domain,
                    state,
                    source,
                    minimum_quality=minimum_metric_quality,
                    maximum_work=maximum_arc_work_units - arc_work,
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
            status = MetricRemeshingStatus.STALLED
        if not np.all(quality >= minimum_metric_quality):
            status = MetricRemeshingStatus.STALLED
    with measure_phase(record_phase, "lineage_construction"):
        edit, target_metric, ordered_patches, ordered_charts, ordered_material = (
            _assemble_surface(
                mesh, state, cells, patches, source.charts, maximum_location_pairs
            )
        )
        if controller is not None:
            if controller.prior_stage is None:
                edit = edit._replace(
                    periodic_orbits=_unchanged_periodic_witness(controller.source)
                )
            else:
                edit = controller.prior_stage.edit
    evidence = MetricRemeshingEvidence(
        status,
        passes=passes,
        counts=(counts[0], counts[1], counts[2], counts[3]),
        rejected_operations=rejected,
        work_units=work,
        lengths=lengths,
        quality=quality,
        maximum_fidelity_bound=float(np.max(fidelity, initial=0.0)),
        metric_quality_floor=minimum_metric_quality,
        criterion=MetricRemeshingCriterion.UNIT_MESH
        if topology_operations
        else MetricRemeshingCriterion.RELOCATION_FIXED_POINT,
        native_work_units=0,
        metric_arc_bounds=arc_bounds,
        metric_arc_edge_ids=arc_edge_ids,
        metric_arc_binding=arc_binding,
        resource_message=(
            "The actual UV metric operation capacity was exhausted."
            if status is MetricRemeshingStatus.RESOURCE_LIMIT
            else None
        ),
        resource_requested=(
            ("maximum_operations", float(maximum_operations)),
            ("maximum_vertices", float(maximum_vertices)),
            ("maximum_cells", float(maximum_cells)),
        )
        if status is MetricRemeshingStatus.RESOURCE_LIMIT
        else (),
        resource_achieved=(
            ("operation_attempts", float(work)),
            ("vertices", float(state.points.shape[0])),
            ("cells", float(state.cells.shape[0])),
        )
        if status is MetricRemeshingStatus.RESOURCE_LIMIT
        else (),
    )
    live = np.unique(state.cells)
    vertex_strata = _surface_vertex_strata(state, live)
    return SurfaceMetricOutcome(
        edit,
        target_metric,
        evidence,
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
        None if controller is None else controller.prior_stage,
    )


def _surface_vertex_strata(
    state: _SurfaceState,
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


def _assemble_surface(
    mesh: CellMesh,
    state: _SurfaceState,
    source_cells: np.ndarray,
    source_patches: np.ndarray,
    source_charts: np.ndarray,
    maximum_pairs: int,
    /,
) -> tuple[CellTopologyEdit, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    blocks = source_family_blocks(
        tuple((block.name, block.cell_kind) for block in mesh.blocks),
        tuple(row_of[state.cells[rows]] for rows in block_rows),
        tuple(state.cell_ids[rows] for rows in block_rows),
    )
    collapsed = _resolved_collapses(state.vertex_ids, ids.size, state.vertex_successor)
    relocated = set() if state.relocated_vertices is None else state.relocated_vertices
    vertex_pairs = [
        (
            parent,
            int(state.vertex_ids[vertex]),
            int(
                EntityLineageKind.RELOCATED
                if vertex in relocated
                else EntityLineageKind.REFINED_FROM
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
            int(EntityLineageKind.RELOCATED),
        )
        for vertex in live.tolist()
        if vertex < ids.size and vertex in relocated
    )
    vertex_pairs.extend(
        (source, target, int(EntityLineageKind.COLLAPSED_INTO))
        for source, target in sorted(collapsed.items())
    )
    cell_kinds = {} if state.cell_kinds is None else state.cell_kinds
    cell_pairs = [
        (
            parent,
            int(identifier),
            cell_kinds.get(int(identifier), int(EntityLineageKind.PRESERVED)),
        )
        for parents, identifier in zip(state.parents, state.cell_ids, strict=True)
        for parent in sorted(parents)
        if parent != identifier or int(identifier) in cell_kinds
    ]
    relations = []
    for dimension, pairs in ((0, vertex_pairs), (1, []), (2, cell_pairs)):
        if dimension == 1:
            relations.append(
                _subentity_relations(
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
                EntityRelations(
                    dimension, values[:, :1], values[:, 1:2], values[:, 2].astype(np.int8)
                )
            )
        else:
            relations.append(
                EntityRelations(
                    dimension,
                    np.empty((0, width), dtype=np.int64),
                    np.empty((0, width), dtype=np.int64),
                    np.empty((0,), dtype=np.int8),
                )
            )
    ordered = np.concatenate(block_rows)
    edit = CellTopologyEdit(
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


def reconstruct_surface_metric_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    outcome: SurfaceMetricOutcome | _SurfaceMetricGeometryWitness,
    domain: MeshingDomain,
    /,
    *,
    maximum_fidelity: float,
    policy: CellGeometryTransitionPolicy | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_atlas: SurfaceSourceRootAtlas | None = None,
) -> SurfaceGeometryReconstruction:
    """Stage degree-preserving target geometry; old inverse coverage stays explicit."""
    if not isinstance(outcome, (SurfaceMetricOutcome, _SurfaceMetricGeometryWitness)):
        raise TypeError(
            "outcome must contain its actual source-bound surface edit/chart witness."
        )
    domain.require_current(outcome.source_id, outcome.source_revision)
    if outcome.domain_id != domain.domain_id:
        raise ValueError(
            "The reconstruction domain must match the accepted chart witnesses."
        )
    identifiers = np.concatenate([block.cell_ids for block in outcome.edit.blocks])
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    if not np.array_equal(identifiers, target_ids):
        raise ValueError(
            "The reconstruction target must preserve the ordered edit cell identities."
        )
    source_vertices = key_rows(
        np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)[:, None],
        outcome.edit.vertex_global_ids[:, None],
    )
    if np.any(source_vertices < 0) or not np.array_equal(
        np.asarray(target_mesh.coordinates)[source_vertices],
        outcome.edit.coordinates,
    ):
        raise ValueError("The target must be the exact staged edit, not an unbound mesh.")
    if source_atlas is None:
        degree = nested_geometry_degree(source_mesh, source_geometry)
        layout = (
            CellGeometrySpec.affine(target_mesh)
            if degree == 1
            else _straight_geometry(target_mesh, degree)
        )
    else:
        layout = None
    with measure_phase(record_phase, "geometry_transition"):
        return reconstruct_parametric_surface_cell_geometry(
            source_mesh,
            source_geometry,
            target_mesh,
            layout,
            domain,
            domain_id=outcome.domain_id,
            cell_ids=identifiers,
            cell_patches=outcome.cell_patches,
            cell_charts=outcome.cell_charts,
            cell_geometry_entity_ids=outcome.cell_geometry_entity_ids,
            cell_occurrence_paths=outcome.cell_occurrence_paths,
            maximum_fidelity=maximum_fidelity,
            policy=policy,
            source_atlas=source_atlas,
        )


class SurfaceMetricGeometryTransition(NamedTuple):
    transition: CellGeometryTransition
    deformation: PreparedSurfaceChartDeformation
    target_mesh: CellMesh


def prepare_surface_metric_geometry_transition(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    outcome: SurfaceMetricOutcome | _SurfaceMetricGeometryWitness,
    domain: MeshingDomain,
    source_witness: SurfaceChartWitness,
    /,
    *,
    maximum_fidelity: float,
    policy: CellGeometryTransitionPolicy,
    maximum_candidate_pairs: int = 1000000,
    maximum_memory_bytes: int = 268435456,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
    validity_policy: CellValidityPolicy | None = None,
    source_chart_cover: MeshingDomainBoundarySource | None = None,
    source_atlas: SurfaceSourceRootAtlas | None = None,
    curve_witness: PreparedSurfaceCurveWitness | None = None,
    material_root_witness: SurfaceChartWitness | None = None,
    target_chart_cover: MeshingDomainBoundarySource | None = None,
    prepared_source_fidelity: SourceFidelityCertificate | None = None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
) -> SurfaceMetricGeometryTransition:
    """Accept only explicit old/target chart deformation with owning certificates."""
    if not isinstance(source_witness, SurfaceChartWitness):
        raise TypeError(
            "source_witness must be scientifically bound SurfaceChartWitness."
        )
    if (
        source_witness.geometry_id != cell_geometry_id(source_geometry)
        or source_witness.topology_id != source_mesh.topology_id
        or source_witness.domain_id != domain.domain_id
    ):
        raise ValueError(
            "The old chart witness must bind this exact source geometry/topology/domain."
        )
    if (
        not isinstance(policy, CellGeometryTransitionPolicy)
        or policy.reconstruction != "bounded_chart_deformation"
    ):
        raise ValueError("Select the explicit bounded_chart_deformation geometry policy.")
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
    )
    target_mesh = reconstruction.target_mesh
    target_witness = SurfaceChartWitness(
        geometry_id=cell_geometry_id(reconstruction.geometry),
        topology_id=target_mesh.topology_id,
        domain_id=domain.domain_id,
        cell_global_ids=reconstruction.cell_global_ids,
        patches=reconstruction.cell_patches,
        charts=reconstruction.cell_charts,
        geometry_entity_ids=reconstruction.cell_geometry_entity_ids,
        occurrence_paths=reconstruction.cell_occurrence_paths,
    )
    source_rows = key_rows(
        outcome.source_cell_ids[:, None],
        np.asarray(source_witness.cell_global_ids)[:, None],
    )
    edit_ids = np.concatenate([block.cell_ids for block in outcome.edit.blocks])
    target_rows = key_rows(
        edit_ids[:, None], np.asarray(target_witness.cell_global_ids)[:, None]
    )
    if np.any(source_rows < 0) or np.any(target_rows < 0):
        raise ValueError(
            "Material witnesses must retain every original source/target scientific cell."
        )
    if material_root_witness is None:
        raise ValueError(
            "Material deformation requires its original admitted root chart witness."
        )
    if curve_witness is not None and target_chart_cover is None:
        from ._surface_association_transfer import prepare_surface_curve_chart_cover

        dimensions, indices, parameters = (
            outcome.vertex_source_dimensions,
            outcome.vertex_source_indices,
            outcome.vertex_source_parameters,
        )
        if (
            dimensions is None
            or indices is None
            or parameters is None
            or source_chart_cover is None
        ):
            raise ValueError(
                "Actual target curve provenance requires its complete retained strata and original cover."
            )
        vertex_rows = key_rows(
            outcome.edit.vertex_global_ids[:, None],
            np.asarray(target_mesh.vertex_global_ids)[:, None],
        )
        if np.any(vertex_rows < 0):
            raise ValueError("The retained target curve strata omit scientific vertices.")
        target_chart_cover = prepare_surface_curve_chart_cover(
            curve_witness,
            target_mesh,
            target_witness,
            dimensions[vertex_rows],
            indices[vertex_rows],
            parameters[vertex_rows],
            source_chart_cover=source_chart_cover,
        )
    with measure_phase(record_phase, "common_refinement"):
        deformation = prepare_surface_chart_deformation(
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
            source_material_charts=outcome.source_material_charts[source_rows],
            target_material_charts=outcome.cell_material_charts[target_rows],
            material_root_witness=material_root_witness,
            target_chart_cover=target_chart_cover,
            prepared_source_fidelity=prepared_source_fidelity,
            prepared_source_restriction_id=(
                None
                if prepared_source_fidelity is None
                or source_geometry.restriction_source is None
                else source_geometry.restriction_source.restriction_source_id
            ),
            prepared_source_certificates=prepared_source_certificates,
        )
    with measure_phase(record_phase, "geometry_transition"):
        transition = transition_chart_deformed_cell_geometry(
            source_mesh,
            source_geometry,
            target_mesh,
            reconstruction,
            deformation,
            policy=policy,
        )
    return SurfaceMetricGeometryTransition(transition, deformation, target_mesh)


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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType

from .._physical import SpatialCoordinateContract
from ..discretization._coordinate_enclosure import CoordinateEnclosureBudget, Expression
from ..discretization._sphere_chart_deformation import (
    prepare_sphere_chart_deformation,
    PreparedSphereChartDeformation,
    reconstruct_sphere_material_cell_geometry,
    SphereGeometryReconstruction,
)
from ..geometry._sphere_material_atlas import (
    _corner_direction,
    _source_image_enclosure,
    _sphere_radial_source_norm,
    _sphere_radial_triangle_bounds,
    SphereMaterialCellAtlas,
)
from ..linalg._small_batched import prepare_exact_small_linear_actions
from ._measurements import NativeMeshingPhase
from ._periodic import (
    ConstructionPointKey,
    PeriodicMetricCandidateStage,
    PeriodicMetricOperation,
    PeriodicMetricOrbitOutcome,
    PeriodicMetricOrbitSelection,
)
from ._topology_edit import (
    assemble_topology_edit,
    PeriodicVertexOrbitWitness,
    TopologyEditBlock,
)


if TYPE_CHECKING:
    from ._adaptation import MeshAdaptationPolicy


class SphereMetricOutcome(NamedTuple):
    """A private, fully certified physical-cell sphere successor."""

    edit: CellTopologyEdit
    target_mesh: CellMesh
    reconstruction: SphereGeometryReconstruction
    source_atlas: SphereMaterialCellAtlas
    target_atlas: SphereMaterialCellAtlas
    deformation: PreparedSphereChartDeformation
    metric: np.ndarray
    exact_stencils: Mapping[int, ConstructionPointKey]
    evidence: MetricRemeshingEvidence
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
    stencils: list[ConstructionPointKey]
    next_cell_id: int
    next_vertex_id: int
    features: dict[tuple[int, int], int]
    blocked: set[tuple[int, int]]
    fixed: set[int]
    successors: dict[int, int]
    kinds: dict[int, int]
    relocated: set[int]
    maximum_fidelity: float


class _SphereMetricBackground(NamedTuple):
    atlas: SphereMaterialCellAtlas
    vertex_ids: np.ndarray
    metric: np.ndarray
    corner_rows: np.ndarray
    matrices: tuple[tuple[tuple[Fraction, ...], ...], ...]
    normalization_norms: Mapping[int, float]
    vertex_memberships: Mapping[int, int]
    vertex_closure_memberships: Mapping[int, int]
    radial_bounds: dict[tuple[float, ...], tuple[Fraction, float, float, Fraction] | None]


def _sphere_edges(state: _SphereMetricState, /) -> tuple[np.ndarray, np.ndarray]:
    records = np.sort(state.cells[:, _LOCAL_EDGES].reshape((-1, 2)), axis=1)
    edges, first = np.unique(records, axis=0, return_index=True)
    return edges, first


def _sphere_charge(
    budget: CoordinateEnclosureBudget, work: int, storage: int = 0, /
) -> None:
    budget.reserve(work, storage)


def _sphere_location(
    background: _SphereMetricBackground,
    patch: int,
    direction: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> ConstructionPointKey:
    """Positive homogeneous actions in immutable original physical material cells."""
    atlas = background.atlas
    right = tuple((Fraction(float(value)),) for value in direction)
    for row in np.flatnonzero(np.asarray(atlas.patches) == patch):
        with budget.temporary_scope():
            action = prepare_exact_small_linear_actions(
                background.matrices[row], right, coordinate_budget=budget
            )
            if action.actions is None:
                raise ValueError("An admitted original sphere cone became singular.")
            homogeneous = tuple(value[0] for value in action.actions)
            denominator = sum(homogeneous, Fraction(0))
            if denominator <= 0 or any(value < 0 for value in homogeneous):
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
    raise MeshingFailure(
        MeshingFailureCategory.LINEAGE_FAILED,
        "A sphere construction leaves the COMPLETE original physical material partition.",
        stage=MeshingStageKind.LINEAGE_CONSTRUCTION.value,
    )


def _sphere_background_metric(
    background: _SphereMetricBackground,
    key: ConstructionPointKey,
    /,
) -> np.ndarray:
    identifiers = np.asarray([identifier for identifier, _ in key], dtype=np.int64)
    rows = key_rows(background.vertex_ids[:, None], identifiers[:, None])
    if np.any(rows < 0):
        raise ValueError(
            "Sphere metric support must name original immutable vertex identities."
        )
    weights = np.asarray([float(weight) for _, weight in key], dtype=np.float64)
    result = np.asarray(
        interpolate_mesh_metric(background.metric[rows], weights), dtype=np.float64
    )
    properties = _tensor_properties(result[None])
    if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
        raise ValueError(
            "Original sphere metric interpolation lost positive definiteness."
        )
    return result


def _sphere_point(
    domain: MeshingDomain,
    background: _SphereMetricBackground,
    patch: int,
    dimension: int,
    index: int,
    parameters: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ConstructionPointKey]:
    _sphere_charge(budget, 64)
    charge_native_geometry_queries(1)
    direction, _ = _corner_direction(domain, patch, dimension, index, parameters)
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
    key = _sphere_location(background, patch, direction, budget)
    return point, direction, _sphere_background_metric(background, key), key


def _sphere_radial_parameters(
    direction: np.ndarray, domain: MeshingDomain, patch: int, /
) -> np.ndarray:
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("A sphere proposal has no finite positive radial direction.")
    unit = direction / norm
    uv = np.asarray((np.arctan2(unit[1], unit[0]), np.arcsin(unit[2])), dtype=np.float64)
    from ..geometry._meshing_domain import _use_endpoints

    endpoints = np.concatenate(
        [_use_endpoints(use) for loop in domain.patches[patch].loops for use in loop]
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
    domain: MeshingDomain,
    cells: np.ndarray,
    patches: np.ndarray,
    directions: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    points: np.ndarray,
    background: _SphereMetricBackground,
    maximum_fidelity: float,
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
                # The memo holds actual immutable cone enclosures, not rounded
                # source-cell identities or a renewed workspace allowance.
                budget.reserve(0, 4096)
                radial = _sphere_radial_triangle_bounds(rays, budget)
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
            low, high = _source_image_enclosure(
                domain.patches[int(patch)].surface, directions[cell]
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
            # actual reconstruction, partition and embedding still decide it.
            if outward(normalization + error, np.inf) > maximum_fidelity:
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
        interpolate_mesh_metric(
            state.metric[cells], np.full(cells.shape, 1 / 3, dtype=np.float64)
        ),
        dtype=np.float64,
    )
    return np.asarray(
        _metric_shape_quality(jnp.asarray(mean), jnp.asarray(tangents), dimension=2),
        dtype=np.float64,
    )


def _sphere_lengths(state: _SphereMetricState, /) -> tuple[np.ndarray, np.ndarray]:
    edges, _ = _sphere_edges(state)
    vector = state.points[edges[:, 1]] - state.points[edges[:, 0]]
    mean = np.asarray(
        interpolate_mesh_metric(
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
    kind: EntityLineageKind,
    /,
    *,
    cell_ids: np.ndarray | None = None,
) -> None:
    keep = np.ones(state.cell_ids.size, dtype=np.bool_)
    keep[removed] = False
    identifiers = (
        _fresh_identifiers(state.next_cell_id, cells.shape[0])
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
    domain: MeshingDomain,
    background: _SphereMetricBackground,
    transfer: SurfaceAssociationTransfer,
    budget: CoordinateEnclosureBudget,
    /,
) -> bool:
    from ._surface_association_transfer import _curve_parameter

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
            _curve_parameter(
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
        domain, background, patch, dimension, index, parameters, budget
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
        budget,
        points=candidate_points,
        background=background,
        maximum_fidelity=state.maximum_fidelity,
    ):
        return False
    _fresh_identifiers(state.next_vertex_id, 1)
    child_ids = _fresh_identifiers(state.next_cell_id, cells.shape[0])
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
        EntityLineageKind.REFINED_FROM,
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
    domain: MeshingDomain,
    background: _SphereMetricBackground,
    quality_floor: float,
    budget: CoordinateEnclosureBudget,
    /,
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
        interpolate_mesh_metric(
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
        budget,
        points=state.points,
        background=background,
        maximum_fidelity=state.maximum_fidelity,
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
        EntityLineageKind.COLLAPSED_INTO,
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
    domain: MeshingDomain,
    background: _SphereMetricBackground,
    budget: CoordinateEnclosureBudget,
    /,
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
        budget,
        points=state.points,
        background=background,
        maximum_fidelity=state.maximum_fidelity,
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
        EntityLineageKind.SWAPPED_FROM,
    )
    return True


def _sphere_relocate(
    state: _SphereMetricState,
    vertex: int,
    domain: MeshingDomain,
    background: _SphereMetricBackground,
    budget: CoordinateEnclosureBudget,
    transfer: SurfaceAssociationTransfer,
    /,
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
        from ._surface_association_transfer import _curve_parameter

        if len(incident) != 2 or any(
            edge in state.blocked or state.features[edge] != index for edge in incident
        ):
            return False
        neighbors = [edge[0] if edge[1] == vertex else edge[1] for edge in incident]
        values = [
            _curve_parameter(
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
        domain, background, patch, dimension, index, parameters, budget
    )
    old_point, old_direction, old_metric = (
        state.points[vertex].copy(),
        state.directions[vertex].copy(),
        state.metric[vertex].copy(),
    )
    old_quality = _sphere_quality(
        state, state.cells[rows], state.patches[rows], background
    )
    old_edges, old_lengths = _sphere_lengths(state)
    local = np.any(old_edges == vertex, axis=1)
    state.points[vertex], state.directions[vertex], state.metric[vertex] = (
        point,
        direction,
        metric,
    )
    accepted = False
    try:
        valid = _sphere_legal(
            domain,
            state.cells[rows],
            state.patches[rows],
            state.directions,
            budget,
            points=state.points,
            background=background,
            maximum_fidelity=state.maximum_fidelity,
        )
        new_quality = _sphere_quality(
            state, state.cells[rows], state.patches[rows], background
        )
        _, new_lengths = _sphere_lengths(state)
        accepted = bool(
            valid
            and np.min(new_quality) >= np.min(old_quality)
            and _unit_defect(new_lengths[local]) < _unit_defect(old_lengths[local])
        )
    finally:
        # Certification can refuse by raising after the trial coordinates have
        # been installed. Restore the numeric carrier on every noncommit exit;
        # the original ledgers still retain all attempted work and queries.
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
        state.kinds[int(identifier)] = int(EntityLineageKind.RELOCATED)
    state.relocated.add(vertex)
    return True


def _sphere_assemble(
    source: CellMeshingResult,
    state: _SphereMetricState,
    /,
) -> tuple[
    CellTopologyEdit,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Mapping[int, ConstructionPointKey],
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
    blocks = source_family_blocks(
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
    collapsed = _resolved_collapses(state.vertex_ids, source_count, state.successors)
    vertex_pairs = [
        (
            parent,
            int(state.vertex_ids[vertex]),
            int(
                EntityLineageKind.RELOCATED
                if vertex in state.relocated
                else EntityLineageKind.REFINED_FROM
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
            int(EntityLineageKind.RELOCATED),
        )
        for vertex in live
        if vertex < source_count and vertex in state.relocated
    )
    vertex_pairs.extend(
        (parent, target, int(EntityLineageKind.COLLAPSED_INTO))
        for parent, target in sorted(collapsed.items())
    )
    cell_pairs = [
        (
            parent,
            int(identifier),
            state.kinds.get(int(identifier), int(EntityLineageKind.PRESERVED)),
        )
        for parents, identifier in zip(state.parents, state.cell_ids, strict=True)
        for parent in sorted(parents)
        if parent != identifier or int(identifier) in state.kinds
    ]
    relations = []
    for dimension, pairs in ((0, vertex_pairs), (2, cell_pairs)):
        values = np.asarray(pairs, dtype=np.int64).reshape((-1, 3))
        relations.append(
            EntityRelations(
                dimension, values[:, :1], values[:, 1:2], values[:, 2].astype(np.int8)
            )
        )
    edges = _subentity_relations(
        source.mesh,
        1,
        state.cells,
        state.vertex_ids,
        state.vertex_sources,
        collapsed_vertices=collapsed,
    )
    edit = CellTopologyEdit(
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
    values: tuple[Expression, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[Expression, ...]:
    """Solve actual source coefficient columns, retaining nonremovable quotients."""
    from ..discretization._coordinate_enclosure import (
        constant,
        expression_parts,
        multiply,
        rational_expression,
    )

    parts = tuple(expression_parts(value, 1) for value in values)
    denominator = constant(1, 1)
    for _, divisor in parts:
        denominator = multiply(denominator, divisor)
    numerators = []
    for row, (numerator, _) in enumerate(parts):
        for other, (_, divisor) in enumerate(parts):
            if row != other:
                numerator = multiply(numerator, divisor)
        numerators.append(numerator)
    indices = sorted(set().union(*(value.keys() for value in numerators)))
    if not indices:
        return ({}, {}, {})
    right = tuple(
        tuple(value.get(index, Fraction(0)) for index in indices) for value in numerators
    )
    prepared = prepare_exact_small_linear_actions(matrix, right, coordinate_budget=budget)
    if prepared.actions is None:
        raise ValueError("Original sphere physical/source coefficient frame is singular.")
    return tuple(
        rational_expression(
            {
                index: coefficient
                for index, coefficient in zip(indices, row, strict=True)
                if coefficient
            },
            denominator,
        )
        for row in prepared.actions
    )


def _sphere_original_cones(
    background: _SphereMetricBackground,
    patch: int,
    edge: tuple[Expression, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[tuple[int, tuple[Expression, ...]], ...]:
    from ..discretization._coordinate_enclosure import constant, expression_add

    atlas = background.atlas
    rows = np.flatnonzero(np.asarray(atlas.patches) == patch)
    slot = int(np.asarray(atlas.source_slots)[rows[0]])
    rotation = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(atlas.rotations)[slot]
    )
    axes = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(atlas.axes)[slot]
    )
    center = tuple(Fraction(float(value)) for value in np.asarray(atlas.centers)[slot])
    translation = tuple(
        Fraction(float(value)) for value in np.asarray(atlas.translations)[slot]
    )
    shift = tuple(
        sum((rotation[i][j] * center[j] for j in range(3)), translation[i])
        for i in range(3)
    )
    physical = tuple(
        expression_add(value, constant(-offset, 1))
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
        action = _sphere_expression_action(matrix, physical, budget)
        results.append((int(row), action))
    return tuple(results)


def _sphere_interval_controls(
    value: Expression,
    lower: Fraction,
    upper: Fraction,
    /,
) -> tuple[Fraction, Fraction]:
    from ..discretization._coordinate_enclosure import (
        add,
        axes,
        constant,
        expression_bernstein_coefficients,
        expression_compose,
        scale,
    )

    argument = add(constant(lower, 1), scale(axes(1)[0], upper - lower))
    controls = expression_bernstein_coefficients(
        expression_compose(value, (argument,)), "box", 1
    )
    return min(controls), max(controls)


def _sphere_interval_metric(
    background: _SphereMetricBackground,
    row: int,
    ranges: tuple[tuple[Fraction, Fraction], ...],
    logarithms: tuple[HermitianFunctionEnclosure, ...],
    budget: CoordinateEnclosureBudget,
    /,
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
            radius * _matrix_norm_enclosure(log.matrix, coordinate_budget=budget)
            + (abs(center) + radius) * log.error
            for center, radius, log in zip(centers, radii, logs, strict=True)
        ),
        Fraction(0),
    )
    return hermitian_exp_enclosure(
        HermitianFunctionEnclosure(nominal, error), coordinate_budget=budget
    )


def _sphere_quadratic_interval(
    tangents: tuple[tuple[Fraction, Fraction], ...],
    metric: tuple[tuple[Fraction, ...], ...],
    error: Fraction,
    budget: CoordinateEnclosureBudget,
    /,
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
        _fraction_sqrt_interval(
            max(Fraction(0), low - error * magnitude), coordinate_budget=budget
        )[0],
        _fraction_sqrt_interval(
            max(Fraction(0), high + error * magnitude), coordinate_budget=budget
        )[1],
    )


def _sphere_affine_constant_metric_arcs(
    points: np.ndarray,
    edges: np.ndarray,
    metric: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> np.ndarray:
    """Certify straight P1 carrier edges without rebuilding expression programs."""
    bounds = np.empty((edges.shape[0], 2), dtype=np.float64)
    with budget.temporary_scope():
        budget.reserve(9, 4096)
        matrix = tuple(tuple(Fraction(float(value)) for value in row) for row in metric)
        entries = tuple(
            (i, j, matrix[i][j]) for i in range(3) for j in range(3) if matrix[i][j]
        )
        for row, (first, second) in enumerate(edges):
            with budget.temporary_scope():
                _sphere_charge(budget, 3 + 3 * len(entries), 16384)
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
                        "An affine sphere carrier edge has no positive constant metric length."
                    )
                lower, upper = _fraction_sqrt_interval(squared, coordinate_budget=budget)
                bounds[row] = outward(lower, -np.inf), outward(upper, np.inf)
    return bounds


def _sphere_affine_constant_metric_quality(
    mesh: CellMesh,
    reconstruction: SphereGeometryReconstruction,
    metric: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> np.ndarray:
    """Exact constant-Gram quality of the actual affine successor cells."""
    points = np.asarray(reconstruction.vertex_coordinates, dtype=np.float64)
    if points.shape != mesh.coordinates.shape or not np.array_equal(
        points, np.asarray(mesh.coordinates)
    ):
        raise ValueError(
            "Affine sphere quality requires the bound scientific vertex coordinates."
        )
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    result = np.empty((cells.shape[0],), dtype=np.float64)
    with budget.temporary_scope():
        budget.reserve(9, 4096)
        matrix = tuple(tuple(Fraction(float(value)) for value in row) for row in metric)
        entries = tuple(
            (i, j, matrix[i][j]) for i in range(3) for j in range(3) if matrix[i][j]
        )
        for row, cell in enumerate(cells):
            with budget.temporary_scope():
                _sphere_charge(budget, 11 + 12 * len(entries), 32768)
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
                lower, _ = _fraction_sqrt_interval(
                    3 * determinant, coordinate_budget=budget
                )
                result[row] = max(0.0, outward(lower / denominator, -np.inf))
    return result


def _sphere_certify_metric_arcs(
    mesh: CellMesh,
    reconstruction: SphereGeometryReconstruction,
    background: _SphereMetricBackground,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Enclose actual successor full-map arcs in the immutable original SPD field.

    Intervals crossing source cone boundaries use the union of every possible
    original cell. No rounded crossing or fake rational polynomial root is used.
    """
    from ..discretization._cell_geometry_transfer import PreparedMappedEdgeArcLength
    from ..discretization._coordinate_enclosure import (
        add,
        axes,
        constant,
        coordinate_expressions,
        expression_compose,
        expression_derivative,
        expression_parts,
        expression_sum,
        rational_expression,
        scale,
    )

    geometry = reconstruction.geometry
    elements, routes, _ = geometry.resolve(mesh)
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
    if constant_metric and nested_geometry_degree(mesh, geometry) == 1:
        points = np.asarray(reconstruction.vertex_coordinates, dtype=np.float64)
        if points.shape != mesh.coordinates.shape or not np.array_equal(
            points, np.asarray(mesh.coordinates)
        ):
            raise ValueError(
                "Affine sphere arcs require the bound scientific vertex coordinates."
            )
        bounds = _sphere_affine_constant_metric_arcs(
            points, edges, background.metric[0], budget
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
                logarithm = hermitian_log_enclosure(value, coordinate_budget=budget)
            budget.retain_basis((logarithm.matrix, logarithm.error))
            cached_logs.append(logarithm)
    logarithms = tuple(cached_logs)
    with budget.activate():
        bank = prepared_coordinate_source_bank(geometry)
        cell_expressions = []
        for element, route in zip(elements, routes, strict=True):
            for nodes in np.asarray(route, dtype=np.int64):
                with budget.temporary_scope():
                    expressions = coordinate_expressions(
                        element, tuple(bank[int(node)] for node in nodes)
                    )
                    if expressions is None:
                        raise ValueError(
                            "Sphere metric arcs require actual full coordinate expressions."
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
                variable = axes(1)[0]
                arguments = tuple(
                    add(constant(a, 1), scale(variable, b - a))
                    for a, b in zip(start, end, strict=True)
                )
                edge = tuple(
                    expression_compose(value, arguments)
                    for value in cell_expressions[cell]
                )
                velocity = tuple(expression_derivative(value, 0) for value in edge)
                if constant_metric:
                    from ..discretization._coordinate_enclosure import (
                        expression_multiply,
                        expression_scale,
                    )

                    metric = tuple(
                        tuple(Fraction(float(value)) for value in line)
                        for line in background.metric[0]
                    )
                    gram = expression_sum(
                        tuple(
                            expression_scale(
                                expression_multiply(velocity[i], velocity[j]),
                                metric[i][j],
                            )
                            for i in range(3)
                            for j in range(3)
                        )
                    )
                    enclosure = PreparedMappedEdgeArcLength(gram).integrate(
                        maximum_work=budget.maximum_work_units - budget.work_units,
                        maximum_subcells=10000,
                        maximum_binomial_terms=32,
                    )
                    bounds[row] = enclosure.lower, enclosure.upper
                    continue
                cones = _sphere_original_cones(
                    background,
                    int(np.asarray(reconstruction.cell_patches)[cell]),
                    edge,
                    budget,
                )
                pending = [(Fraction(0), Fraction(1))]
                integral_low = integral_high = Fraction(0)
                while pending:
                    lower, upper = pending.pop()
                    with budget.temporary_scope():
                        _sphere_charge(budget, 64)
                        tangent_bounds = tuple(
                            _sphere_interval_controls(value, lower, upper)
                            for value in velocity
                        )
                        candidates = []
                        for source_row, homogeneous in cones:
                            numerator_parts = tuple(
                                expression_parts(value, 1) for value in homogeneous
                            )
                            # All actions share one exact denominator; multiplying it
                            # away must retain its sign before original cone decisions.
                            divisor_low, divisor_high = _sphere_interval_controls(
                                numerator_parts[0][1], lower, upper
                            )
                            if divisor_low <= 0 <= divisor_high:
                                raise ValueError(
                                    "An actual sphere edge quotient has an unresolved source denominator."
                                )
                            coefficients = tuple(
                                _sphere_interval_controls(value, lower, upper)
                                for value in homogeneous
                            )
                            if any(high < 0 for _, high in coefficients):
                                continue
                            denominator = expression_sum(homogeneous)
                            den_low, den_high = _sphere_interval_controls(
                                denominator, lower, upper
                            )
                            if den_high <= 0:
                                continue
                            ranges = []
                            for value, (a, b) in zip(
                                homogeneous, coefficients, strict=True
                            ):
                                if den_low > 0:
                                    numerator, divisor = expression_parts(value, 1)
                                    total_numerator, total_divisor = expression_parts(
                                        denominator, 1
                                    )
                                    from ..discretization._coordinate_enclosure import (
                                        multiply,
                                    )

                                    quotient = rational_expression(
                                        multiply(numerator, total_divisor),
                                        multiply(divisor, total_numerator),
                                    )
                                    a, b = _sphere_interval_controls(
                                        quotient, lower, upper
                                    )
                                    ranges.append(
                                        (max(Fraction(0), a), min(Fraction(1), b))
                                    )
                                else:
                                    ranges.append((Fraction(0), Fraction(1)))
                            if any(a > b for a, b in ranges):
                                continue
                            metric, error = _sphere_interval_metric(
                                background, source_row, tuple(ranges), logarithms, budget
                            )
                            candidates.append(
                                _sphere_quadratic_interval(
                                    tangent_bounds, metric, error, budget
                                )
                            )
                        if not candidates:
                            raise ValueError(
                                "Actual sphere metric arc has no original radial background coverage."
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
                bounds[row] = (
                    outward(integral_low, -np.inf),
                    outward(integral_high, np.inf),
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
    source: CellMeshingResult,
    metric: np.ndarray,
    atlas: SphereMaterialCellAtlas,
    transfer: SurfaceAssociationTransfer,
    policy: MeshAdaptationPolicy,
    maximum_fidelity: float,
    /,
) -> tuple[_SphereMetricState, _SphereMetricBackground]:
    from ._adaptation import (
        _cell_classes,
        _closure_rows,
        _edge_classes,
        _membership,
        _organization_scopes,
        _row_classes,
        _selected_rows,
    )

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
    organization = _organization_scopes(source)
    vertex_memberships = _row_classes(_membership(mesh, organization, 0))
    closures = np.zeros((ids.size, len(organization)), dtype=np.bool_)
    for column, scope in enumerate(organization):
        rows = _selected_rows(mesh, scope)
        closures[_closure_rows(mesh, scope.entity_dimension, rows, 0), column] = True
    vertex_closures = _row_classes(closures)
    cell_classes = _cell_classes(mesh, organization)
    cell_rows = key_rows(entity_keys(mesh, 2), cell_ids[:, None])
    cell_classes = _row_classes(
        np.column_stack((cell_classes[cell_rows], classes[2].codes[cell_rows]))
    )
    if source.attributes:
        from ._layer_core import layer_interval_classes

        cell_classes = _row_classes(
            np.column_stack((cell_classes, layer_interval_classes(source)))
        )
    entity_cell_rows = key_rows(cell_ids[:, None], entity_keys(mesh, 2))
    edge_classes = _edge_classes(mesh, organization, cell_classes[entity_cell_rows])
    fixed = (dimensions == 0) | np.any(_membership(mesh, organization, 0), axis=1)
    protected = (
        transfer.protected_edges(source, midpoint_required=False) | ~classes[1].resolved
    )
    for scope in policy.protected_scopes:
        rows = _selected_rows(mesh, scope)
        fixed[_closure_rows(mesh, scope.entity_dimension, rows, 0)] = True
        if scope.entity_dimension >= 1:
            protected[_closure_rows(mesh, scope.entity_dimension, rows, 1)] = True
    edge_vertices = key_rows(ids[:, None], entity_keys(mesh, 1).reshape((-1, 1))).reshape(
        (-1, 2)
    )
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
    corners = key_rows(
        ids[:, None], np.asarray(atlas.physical_corner_global_ids).reshape((-1, 1))
    ).reshape((-1, 3))
    matrices = tuple(
        tuple(tuple(Fraction(float(value)) for value in line) for line in rays.T)
        for rays in np.asarray(atlas.directions)
    )
    immutable = metric.copy()
    immutable.setflags(write=False)
    norms = {
        int(patch): _sphere_radial_source_norm(domain.patches[int(patch)].surface)
        for patch in np.unique(patches)
    }
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
    )


def _sphere_strata(
    state: _SphereMetricState,
    edit: CellTopologyEdit,
    domain: MeshingDomain,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[tuple[tuple[str, ...], ...], ...],
    tuple[tuple[str, ...], ...],
]:
    rows = key_rows(state.vertex_ids[:, None], edit.vertex_global_ids[:, None])
    if np.any(rows < 0):
        raise ValueError("Sphere successor strata omit actual generated vertices.")
    connectivity = []
    for block in edit.blocks:
        if not isinstance(block, TopologyEditBlock) or block.cell_kind != "triangle":
            raise ValueError(
                "Sphere successor strata require actual typed triangle edit blocks."
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
    source: CellMeshingResult,
    state: _SphereMetricState,
    edit: CellTopologyEdit,
    target: CellMesh,
    patches: np.ndarray,
    atlas: SphereMaterialCellAtlas,
    policy: MeshAdaptationPolicy,
    maximum_fidelity: float,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[SphereGeometryReconstruction, PreparedSphereChartDeformation]:
    limits = policy.limits
    if source.certification is None:
        raise ValueError(
            "Sphere reconstruction requires the original accepted certificate limits."
        )
    certificate_limits = source.certification.request.limits
    degree = nested_geometry_degree(source.mesh, source.geometry)
    maximum_nodes = min(
        policy.geometry_transition.maximum_evaluations,
        limits.maximum_data_bytes // (3 * np.dtype(np.float64).itemsize),
        budget.maximum_memory_bytes // (3 * np.dtype(np.float64).itemsize),
    )
    if degree == 1:
        layout = CellGeometrySpec.affine(target)
    else:
        layout = _straight_geometry(
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
        reconstruction = reconstruct_sphere_material_cell_geometry(
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
            coordinate_budget=budget,
            certificate_limits=certificate_limits,
            validity_policy=policy.audit_policy.validity_policy,
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
        deformation = prepare_sphere_chart_deformation(
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
            coordinate_budget=budget,
            certificate_limits=certificate_limits,
            validity_policy=policy.audit_policy.validity_policy,
            prepared_source_certificates=original_certificates,
            prepared_target_certificates=(
                reconstruction.target_validity,
                reconstruction.target_embedding,
            ),
        )
    reconstruction.require_bound(atlas, target)
    deformation.require_bound(
        source.mesh, source.geometry, target, reconstruction.geometry
    )
    return reconstruction, deformation


def build_sphere_periodic_metric_candidate(
    source: CellMeshingResult,
    state: _SphereMetricState,
    background: _SphereMetricBackground,
    transfer: SurfaceAssociationTransfer,
    policy: MeshAdaptationPolicy,
    budget: CoordinateEnclosureBudget,
    candidate: CellTopologyEdit,
    exact_stencils: Mapping[int, ConstructionPointKey],
    selection: PeriodicMetricOrbitSelection,
    prior_stage: PeriodicMetricOrbitOutcome | None,
    /,
) -> PeriodicMetricCandidateStage:
    """Invoke the real radial owner on every exact source-incidence carrier."""
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
        cell_geometry_id(geometry),
        source.coordinate_contract.spatial_id,
    ):
        raise ValueError(
            "Sphere orbit selection is stale for the actual current private scientific frame."
        )
    if not exact_stencils or candidate.coordinates.shape[1] != 3:
        raise ValueError("Sphere orbit candidate lacks real source support.")
    executed: list[int] = []
    with budget.activate():
        for carrier in selection.carriers:
            if carrier.protected_entity or carrier.protected_vertex_global_ids:
                raise ValueError(
                    "A sphere quotient orbit intersects protected scientific strata."
                )
            ordered = tuple(
                carrier.vertex_global_ids[index] for index in carrier.vertex_permutation
            )
            rows = key_rows(
                state.vertex_ids[:, None], np.asarray(ordered, dtype=np.int64)[:, None]
            )
            if np.any(rows < 0) or any(int(row) in state.successors for row in rows):
                raise ValueError(
                    "Sphere orbit carriers have incompatible overlapping cavities."
                )
            changed = _sphere_apply_operation(
                state,
                selection.operation,
                tuple(map(int, rows)),
                background.atlas.domain,
                background,
                transfer,
                budget,
            )
            if not changed:
                raise ValueError(
                    "A required sphere orbit operation fails its actual radial/source constraints."
                )
            executed.append(carrier.entity_global_id)
    edit, _, _, _, stencils = _sphere_assemble(source, state)
    return PeriodicMetricCandidateStage(edit, stencils, tuple(executed))


def _sphere_apply_operation(
    state: _SphereMetricState,
    operation: PeriodicMetricOperation,
    vertices: tuple[int, ...],
    domain: MeshingDomain,
    background: _SphereMetricBackground,
    transfer: SurfaceAssociationTransfer,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    reserved_workspace_upper: int = 0,
) -> bool:
    with budget.temporary_scope():
        execution = current_native_execution_budget()
        if execution is not None:
            # Orbit copies enter here without the producer's seed-cavity gate.
            # Admit each actual carrier before its proposal allocates or mutates.
            execution.admit_cavity(
                int(np.count_nonzero(np.any(np.isin(state.cells, vertices), axis=1)))
            )
        budget.reserve(
            0, max(0, _sphere_workspace_upper(state) - reserved_workspace_upper)
        )
        match operation:
            case "split":
                if len(vertices) != 2:
                    raise ValueError(
                        "A sphere split requires its two actual edge vertices."
                    )
                edge = (min(vertices[0], vertices[1]), max(vertices[0], vertices[1]))
                return _sphere_split(state, edge, domain, background, transfer, budget)
            case "collapse":
                if len(vertices) != 2:
                    raise ValueError(
                        "A sphere collapse requires its actual removed and kept vertices."
                    )
                return _sphere_collapse(
                    state, vertices[0], vertices[1], domain, background, 0.05, budget
                ) or _sphere_collapse(
                    state, vertices[1], vertices[0], domain, background, 0.05, budget
                )
            case "flip":
                if len(vertices) != 2:
                    raise ValueError(
                        "A sphere flip requires its two actual edge vertices."
                    )
                edge = (min(vertices[0], vertices[1]), max(vertices[0], vertices[1]))
                return _sphere_flip(state, edge, domain, background, budget)
            case "relocate":
                if len(vertices) != 1:
                    raise ValueError(
                        "A sphere relocation requires one actual source vertex."
                    )
                return _sphere_relocate(
                    state, vertices[0], domain, background, budget, transfer
                )
            case _:
                raise ValueError("Unknown sphere metric operation.")


def execute_sphere_metric_adaptation(
    source: CellMeshingResult,
    metric: ArrayLike,
    domain: MeshingDomain,
    source_atlas: SphereMaterialCellAtlas,
    transfer: SurfaceAssociationTransfer,
    /,
    *,
    policy: MeshAdaptationPolicy,
    coordinate_contract: SpatialCoordinateContract,
    maximum_fidelity: float,
    topology_operations: bool,
    relocation: bool,
    coordinate_budget: CoordinateEnclosureBudget,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> SphereMetricOutcome:
    """Stage actual source-realized sphere operations under the original ledger."""
    from ._adaptation import MeshAdaptationPolicy

    if not isinstance(policy, MeshAdaptationPolicy) or not isinstance(
        coordinate_budget, CoordinateEnclosureBudget
    ):
        raise TypeError(
            "Sphere metric adaptation requires original policy and coordinate ledger owners."
        )
    source_atlas.require_bound(domain, source.mesh, source.geometry, coordinate_contract)
    domain.require_current(source_atlas.source_id, source_atlas.source_revision)
    if (
        not source.audit.passed
        or source.certification is None
        or not source.certification.passed
    ):
        raise ValueError(
            "Sphere metric adaptation requires an actually accepted scientific predecessor."
        )
    execution = current_native_execution_budget()
    if execution is None:
        raise ValueError(
            "Sphere metric adaptation requires the original active native execution scope."
        )
    execution.charge(work=0)
    if transfer.support.domain.domain_id != domain.domain_id:
        raise ValueError(
            "Sphere metric transfer changes the authoritative source domain."
        )
    if not np.isfinite(maximum_fidelity) or maximum_fidelity < 0:
        raise ValueError("Sphere metric fidelity must be finite and nonnegative.")
    if not isinstance(topology_operations, bool) or not isinstance(relocation, bool):
        raise TypeError("Sphere operation controls must be bool.")
    values = np.asarray(metric, dtype=np.float64)
    if values.shape != (source.mesh.coordinates.shape[0], 3, 3):
        raise ValueError("Sphere metrics must follow actual original vertex identities.")
    properties = _tensor_properties(values)
    if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
        raise ValueError(
            "Sphere metric background requires original finite Hermitian SPD tensors."
        )
    limits = policy.limits
    _stage_source_limits(source.mesh, limits.maximum_vertices, limits.maximum_cells)
    initial_work = coordinate_budget.work_units
    if (
        coordinate_budget.maximum_work_units > limits.maximum_work_units
        or coordinate_budget.maximum_memory_bytes > limits.maximum_scratch_bytes
    ):
        raise ValueError(
            "Sphere coordinate ledger cannot enlarge the original authored work or workspace allowance."
        )
    counts = [0, 0, 0, 0]
    attempts = rejected = passes = 0
    status = MetricRemeshingStatus.PASS_LIMIT
    with _sphere_coordinate_phase(coordinate_budget):
        source_cells = sum(block.cell_count for block in source.mesh.blocks)
        workspace_upper, source_workspace_upper = _sphere_initial_workspace_upper(
            source.mesh, values
        )
        persistent_upper = workspace_upper + source_workspace_upper
        coordinate_budget.reserve(9 * source_cells, persistent_upper)
        with coordinate_budget.temporary_scope():
            construction_upper = source_cells * 65536 + values.nbytes * 16
            coordinate_budget.reserve(0, max(0, construction_upper - persistent_upper))
            state, background = _sphere_initial_state(
                source, values, source_atlas, transfer, policy, maximum_fidelity
            )
        periodic_stage: PeriodicMetricOrbitOutcome | None = None
        periodic_geometry: (
            tuple[SphereGeometryReconstruction, PreparedSphereChartDeformation] | None
        ) = None
        # Pass measurement and direct operations share these same four live
        # carriers. Keep their high-water reservation until the owning phase
        # ends; unchanged state must not allocate another carrier bank.
        for passes in range(1, policy.maximum_passes + 1):
            required_workspace = _sphere_workspace_upper(state)
            coordinate_budget.reserve(0, max(0, required_workspace - workspace_upper))
            workspace_upper = max(workspace_upper, required_workspace)
            edges, lengths = _sphere_lengths(state)
            quality = _sphere_quality(state, state.cells, state.patches, background)
            if np.all((lengths >= _LOWER) & (lengths <= _UPPER)) and np.all(
                quality >= 0.05
            ):
                status = MetricRemeshingStatus.COMPLETE
                break
            proposals: list[tuple[PeriodicMetricOperation, tuple[int, ...]]] = []
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
            for operation, vertices in proposals:
                required_workspace = _sphere_workspace_upper(state)
                coordinate_budget.reserve(0, max(0, required_workspace - workspace_upper))
                workspace_upper = max(workspace_upper, required_workspace)
                if coordinate_budget.work_units >= coordinate_budget.maximum_work_units:
                    status = MetricRemeshingStatus.RESOURCE_LIMIT
                    break
                if operation == "split" and (
                    state.points.shape[0] >= limits.maximum_vertices
                    or state.cells.shape[0] + 2
                    > min(limits.maximum_cells, limits.maximum_faces)
                    or 3 * (state.cells.shape[0] + 2)
                    > limits.maximum_connectivity_entries
                    or _sphere_edges(state)[0].shape[0] + 3 > limits.maximum_edges
                ):
                    status = MetricRemeshingStatus.RESOURCE_LIMIT
                    break
                cavity = np.flatnonzero(np.any(np.isin(state.cells, vertices), axis=1))
                if cavity.size > limits.maximum_cavity_cells:
                    rejected += 1
                    continue
                execution = current_native_execution_budget()
                if execution is not None:
                    execution.admit_cavity(cavity.size)
                attempts += 1
                coordinate_budget.reserve(1)
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
                        raise ValueError("Unknown sphere metric phase.")
                with measure_phase(record_phase, phase):
                    if source.mesh.periodic_topology is None:
                        changed = _sphere_apply_operation(
                            state,
                            operation,
                            vertices,
                            domain,
                            background,
                            transfer,
                            coordinate_budget,
                            reserved_workspace_upper=workspace_upper,
                        )
                    else:
                        synchronized = _sphere_stage_periodic_operation(
                            source,
                            state,
                            background,
                            transfer,
                            policy,
                            maximum_fidelity,
                            coordinate_budget,
                            operation,
                            vertices,
                            periodic_stage,
                        )
                        changed = synchronized is not None
                        if synchronized is not None:
                            state, periodic_stage, reconstruction, deformation = (
                                synchronized
                            )
                            periodic_geometry = reconstruction, deformation
                if changed:
                    counts[
                        ("split", "collapse", "flip", "relocate").index(operation)
                    ] += 1
                    applied += 1
                else:
                    rejected += 1
            if status is MetricRemeshingStatus.RESOURCE_LIMIT:
                break
            if not applied:
                status = MetricRemeshingStatus.STALLED
                break
        required_workspace = _sphere_workspace_upper(state)
        coordinate_budget.reserve(0, max(0, required_workspace - workspace_upper))
        edit, target_metric, patches, _, stencils = _sphere_assemble(source, state)
        if periodic_stage is None:
            if source.mesh.periodic_topology is not None:
                # An unchanged quotient source has no born orbit identities.
                edit = edit._replace(periodic_orbits=_unchanged_periodic_witness(source))
            version = canonical_fingerprint(
                {
                    "kind": "sphere-metric-numeric-frame",
                    "source": source.mesh.numeric_version,
                    "coordinates": array_tree_fingerprint(edit.coordinates),
                }
            )
            target, _, _ = assemble_topology_edit(
                source.mesh, edit, numeric_version=version
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
                coordinate_budget,
            )
        else:
            if periodic_geometry is None:
                raise ValueError(
                    "Sphere periodic edit lacks its real full-map geometry stage."
                )
            edit, target = periodic_stage.edit, periodic_stage.target_mesh
            stencils = periodic_stage.exact_stencils
            reconstruction, deformation = periodic_geometry
        arcs, edge_ids, binding = _sphere_certify_metric_arcs(
            target, reconstruction, background, coordinate_budget
        )
        quality = _sphere_certify_metric_quality(
            target, reconstruction, background, coordinate_budget
        )
        accepted = np.all((arcs[:, 0] >= _LOWER) & (arcs[:, 1] <= _UPPER)) and np.all(
            quality >= 0.05
        )
        if accepted and status is not MetricRemeshingStatus.RESOURCE_LIMIT:
            status = MetricRemeshingStatus.COMPLETE
        elif status is MetricRemeshingStatus.COMPLETE:
            status = MetricRemeshingStatus.STALLED
        work = coordinate_budget.work_units - initial_work
        evidence = MetricRemeshingEvidence(
            status,
            passes=passes,
            counts=(counts[0], counts[1], counts[2], counts[3]),
            rejected_operations=rejected,
            work_units=work,
            lengths=np.mean(arcs, axis=1),
            quality=quality,
            maximum_fidelity_bound=float(
                np.max(np.asarray(reconstruction.fidelity_bounds))
            ),
            criterion=MetricRemeshingCriterion.UNIT_MESH
            if topology_operations
            else MetricRemeshingCriterion.RELOCATION_FIXED_POINT,
            operation_attempts=attempts,
            native_work_units=0,
            metric_arc_bounds=arcs,
            metric_arc_edge_ids=edge_ids,
            metric_arc_binding=binding,
            resource_message=(
                "The actual sphere metric operation capacity was exhausted."
                if status is MetricRemeshingStatus.RESOURCE_LIMIT
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
            if status is MetricRemeshingStatus.RESOURCE_LIMIT
            else (),
            resource_achieved=(
                ("coordinate_work_units", float(coordinate_budget.work_units)),
                ("vertices", float(state.points.shape[0])),
                ("edges", float(_sphere_edges(state)[0].shape[0])),
                ("cells", float(state.cells.shape[0])),
                ("connectivity_entries", float(3 * state.cells.shape[0])),
            )
            if status is MetricRemeshingStatus.RESOURCE_LIMIT
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


def _sphere_certify_metric_quality(
    mesh: CellMesh,
    reconstruction: SphereGeometryReconstruction,
    background: _SphereMetricBackground,
    budget: CoordinateEnclosureBudget,
    /,
) -> np.ndarray:
    """Whole-cell lower bounds from the complete successor differential."""
    from ..discretization._coordinate_enclosure import (
        coordinate_expressions,
        expression_bernstein_coefficients,
        expression_derivative,
        expression_determinant,
        expression_multiply,
        expression_scale,
        expression_sum,
    )
    from ..geometry._meshing_domain import _gram_spectrum_bounds

    constant_metric = np.array_equal(
        background.metric, np.broadcast_to(background.metric[0], background.metric.shape)
    )
    if constant_metric and nested_geometry_degree(mesh, reconstruction.geometry) == 1:
        return _sphere_affine_constant_metric_quality(
            mesh, reconstruction, background.metric[0], budget
        )
    ratio = Fraction(1)
    if not constant_metric:
        spectra = tuple(_gram_spectrum_bounds(value) for value in background.metric)
        low = min(value[0] for value in spectra)
        high = max(value[1] for value in spectra)
        if low <= 0:
            raise ValueError(
                "Original SPD background lacks a certified positive spectral lower bound."
            )
        ratio = (
            _fraction_sqrt_interval(Fraction(low), coordinate_budget=budget)[0]
            / _fraction_sqrt_interval(Fraction(high), coordinate_budget=budget)[1]
        )
    tensor = background.metric[0] if constant_metric else np.eye(3, dtype=np.float64)
    matrix = tuple(tuple(Fraction(float(value)) for value in row) for row in tensor)
    elements, routes, _ = reconstruction.geometry.resolve(mesh)
    bank = prepared_coordinate_source_bank(reconstruction.geometry)
    result: list[float] = []
    for element, route in zip(elements, routes, strict=True):
        for nodes in np.asarray(route, dtype=np.int64):
            with budget.temporary_scope():
                coordinates = coordinate_expressions(
                    element, tuple(bank[int(node)] for node in nodes)
                )
                if coordinates is None:
                    raise ValueError(
                        "Sphere quality requires the actual complete coordinate map."
                    )
                derivatives = tuple(
                    tuple(expression_derivative(value, axis) for value in coordinates)
                    for axis in range(2)
                )
                gram = tuple(
                    tuple(
                        expression_sum(
                            tuple(
                                expression_scale(
                                    expression_multiply(
                                        derivatives[a][i], derivatives[b][j]
                                    ),
                                    matrix[i][j],
                                )
                                for i in range(3)
                                for j in range(3)
                            )
                        )
                        for b in range(2)
                    )
                    for a in range(2)
                )
                determinant = expression_determinant(gram, variable_dimension=2)
                denominator = expression_sum(
                    (gram[0][0], gram[1][1], expression_scale(gram[0][1], -1))
                )
                floor = min(expression_bernstein_coefficients(determinant, "simplex", 2))
                ceiling = max(
                    expression_bernstein_coefficients(denominator, "simplex", 2)
                )
                if floor <= 0 or ceiling <= 0:
                    result.append(0.0)
                    continue
                lower = (
                    _fraction_sqrt_interval(3 * floor, coordinate_budget=budget)[0]
                    / ceiling
                ) * ratio
                result.append(max(0.0, outward(lower, -np.inf)))
    return np.asarray(result, dtype=np.float64)


@contextmanager
def _sphere_coordinate_phase(budget: CoordinateEnclosureBudget, /) -> Iterator[None]:
    """Charge actual exact host work once, including an atomic refused phase."""
    from ._adaptation import _charge_remaining_after_failure

    before = budget.work_units
    charged_before = budget.native_charged_work_units
    with budget.activate(), budget.temporary_scope():
        try:
            yield
        except Exception as error:
            _charge_remaining_after_failure(budget, error)
            raise
        else:
            already_charged = budget.native_charged_work_units - charged_before
            budget.charge_native_work(budget.work_units - before - already_charged)


def _sphere_stage_periodic_operation(
    source: CellMeshingResult,
    state: _SphereMetricState,
    background: _SphereMetricBackground,
    transfer: SurfaceAssociationTransfer,
    policy: MeshAdaptationPolicy,
    maximum_fidelity: float,
    budget: CoordinateEnclosureBudget,
    operation: PeriodicMetricOperation,
    vertices: tuple[int, ...],
    prior_stage: PeriodicMetricOrbitOutcome | None,
    /,
) -> (
    tuple[
        _SphereMetricState,
        PeriodicMetricOrbitOutcome,
        SphereGeometryReconstruction,
        PreparedSphereChartDeformation,
    ]
    | None
):
    from copy import deepcopy

    from ._periodic import (
        metric_target_lineage,
        PeriodicMetricGeometryStage,
        synchronize_periodic_metric_edit,
    )

    if source.certification is None:
        raise ValueError(
            "Periodic sphere operations require original source certificates."
        )
    # Separate seed and aggregate states are necessary rollback workspaces:
    # the original carrier is executed first by the callback to retain born IDs.
    with budget.temporary_scope():
        budget.reserve(
            0,
            state.points.nbytes * 16
            + state.cells.nbytes * 32
            + len(state.stencils) * 65536,
        )
        seed = deepcopy(state)
        if not _sphere_apply_operation(
            seed,
            operation,
            vertices,
            background.atlas.domain,
            background,
            transfer,
            budget,
        ):
            return None
        candidate, _, _, _, exact = _sphere_assemble(source, seed)
        aggregate = deepcopy(state)
        geometry_records: list[
            tuple[SphereGeometryReconstruction, PreparedSphereChartDeformation]
        ] = []

        def build_candidate(
            edit: CellTopologyEdit,
            stencils: Mapping[int, ConstructionPointKey],
            selection: PeriodicMetricOrbitSelection,
        ) -> PeriodicMetricCandidateStage:
            return build_sphere_periodic_metric_candidate(
                source,
                aggregate,
                background,
                transfer,
                policy,
                budget,
                edit,
                stencils,
                selection,
                prior_stage,
            )

        def build_geometry(
            edit: CellTopologyEdit, target: CellMesh
        ) -> PeriodicMetricGeometryStage:
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
                budget,
            )
            transition = transition_chart_deformed_cell_geometry(
                source.mesh,
                source.geometry,
                target,
                reconstruction,
                deformation,
                policy=policy.geometry_transition,
            )
            lineage = metric_target_lineage(source.mesh, edit, target)
            associations = transfer.propagate(
                source,
                lineage,
                target,
                geometry=reconstruction.geometry,
                embedding=reconstruction.target_embedding,
                deformation=deformation,
            )
            geometry_records.append((reconstruction, deformation))
            return PeriodicMetricGeometryStage(
                target, reconstruction.geometry, associations, deformation, transition
            )

        retained = (
            () if prior_stage is None else prior_stage.periodic_witness.quotient_entities
        )
        result = synchronize_periodic_metric_edit(
            source,
            candidate,
            exact,
            operation=operation,
            limits=policy.limits,
            policy=policy,
            coordinate_budget=budget,
            certificate_limits=source.certification.request.limits,
            prior_stage=prior_stage,
            build_candidate=build_candidate,
            build_geometry=build_geometry,
            retained_quotient_entities=retained,
        )
        if len(geometry_records) != 1:
            raise ValueError(
                "Sphere orbit controller did not build exactly one actual successor map."
            )
        reconstruction, deformation = geometry_records[0]
        return aggregate, result, reconstruction, deformation


def _unchanged_periodic_witness(
    source: CellMeshingResult, /
) -> PeriodicVertexOrbitWitness:
    from ._periodic import PeriodicConstructionOrbits

    state = PeriodicConstructionOrbits(source.mesh)
    return state.witness(
        source.mesh,
        source.mesh,
        {
            int(identifier): ((int(identifier), Fraction(1)),)
            for identifier in np.asarray(source.mesh.vertex_global_ids)
        },
        (),
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
    # is an upper bound on actual feature memberships, including protected ones.
    carriers = (
        4096 + 12 * arrays + 2048 * (cells + 2 * vertices + mesh.entity_set(1).count)
    )
    # Nine exact binary64 direction fractions and their row tuples fit in 8 KiB
    # per cone, even at binary64's maximum denominator height. Immutable source
    # array copies and both ID/membership maps have separate vertex slot bounds.
    source = 4096 + 8192 * cells + 1024 * vertices + 2 * metric.nbytes
    return carriers, source
