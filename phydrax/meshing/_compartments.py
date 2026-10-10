#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact occupied-image material complex and native constrained volume fill.

All material faces and junctions share one vertex table before native recovery.
Whole-cell ownership is obtained by protected-facet flooding; point sampling is
not used to classify cells or establish interface conformity.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np
from jax.typing import DTypeLike

from .._meshcore import (
    current_native_execution_budget,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionBudget,
    NativeExecutionEvidence,
    NativeHostStorageWorkspace,
)
from ..discretization import CellGeometrySpec, CellMesh, TetrahedralConnectivity
from ..discretization._cell_geometry_validity import (
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ..geometry._compartments import CompartmentMeshingSource
from ..geometry._mesh_certificates import (
    certify_domain_coverage,
    certify_global_embedding,
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
    MeshCertificateLimits,
    PiecewiseLinearDomain,
)
from ..geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding
from ..typing import checked
from ._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    VolumeMeshingSpec,
)
from ._measurements import (
    measure_phase,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from ._organization import (
    material_interface_facets,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    RegionMeshingEvidence,
)
from ._scope import MeshingEntityKind, MeshingScope
from ._trace import MeshingStageKind
from ._volume_generation import (
    _bind_volume_execution,
    _native_volume_operation_started,
    generate_plc_volume,
    native_volume_checkpoint,
    native_volume_execution_budget,
    NativeVolumeSchedule,
    PiecewiseLinearComplex,
    prepare_plc_source,
    PreparedPlcSource,
    VolumeConstruction,
)


if TYPE_CHECKING:
    from ..geometry._supermesh import PreparedCommonRefinement
    from ._lineage import MeshLineage
    from ._result import CellMeshingResult


@dataclass(frozen=True, slots=True)
class CompartmentConstruction:
    """Internal construction transaction; publication returns CellMeshingResult."""

    volume: VolumeConstruction
    region_evidence: RegionMeshingEvidence


_IMAGE_FACE_AXES = ((1, 2), (2, 0), (0, 1))


@dataclass(frozen=True, slots=True)
class _ImageFaceRectangle:
    """One disjoint lattice rectangle with a single oriented material incidence."""

    stratum: tuple[int, int, int, int, int]
    bounds: tuple[int, int, int, int]

    def corners(self) -> tuple[tuple[int, int, int], ...]:
        axis, plane, _, _, _ = self.stratum
        first_axis, second_axis = _IMAGE_FACE_AXES[axis]
        first_lower, second_lower, first_upper, second_upper = self.bounds
        keys = []
        for first, second in (
            (first_lower, second_lower),
            (first_upper, second_lower),
            (first_upper, second_upper),
            (first_lower, second_upper),
        ):
            key = [0, 0, 0]
            key[axis] = plane
            key[first_axis] = 2 * first - 1
            key[second_axis] = 2 * second - 1
            keys.append((key[0], key[1], key[2]))
        return tuple(keys)


def _check_image_preparation_limits(
    limits: MeshingLimits | None,
    requirements: tuple[tuple[str, int], ...],
    /,
) -> None:
    """Admit a complete preparation allocation before creating its storage."""
    if limits is None:
        return
    for name, required in requirements:
        maximum = getattr(limits, name)
        if required > maximum:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"The occupied-image preparation exceeds {name}.",
                stage="source_inspection",
                requested=((name, maximum),),
                achieved=((name.removeprefix("maximum_"), required),),
            )


@dataclass(slots=True)
class _ImagePreparationBudget:
    """Count executed source visits, exact tests and affine point evaluations."""

    limits: MeshingLimits | None
    started: float | None = field(default_factory=monotonic)
    work_units: int = 0
    geometry_queries: int = 0
    execution_budget: NativeExecutionBudget | None = None
    storage: NativeHostStorageWorkspace | None = None

    def admit_storage(self, bytes_upper: int, /) -> None:
        """Reserve the complete live preparation upper bound before its growth."""
        _check_image_preparation_limits(
            self.limits, (("maximum_scratch_bytes", bytes_upper),)
        )
        if self.storage is None:
            return
        execution = self.execution_budget
        if execution is None:
            raise RuntimeError("Image storage admission lost its actual execution owner.")
        execution.charge()
        available = execution.remaining().remaining_scratch_bytes
        completed = self.storage.bound
        try:
            self.storage.set_bound(bytes_upper)
        except MeshcoreError as error:
            if error.status not in (
                MeshcoreStatus.CAPACITY_EXCEEDED,
                MeshcoreStatus.TIMEOUT,
            ):
                raise
            requested = (("remaining_scratch_bytes", available),)
            if self.limits is not None:
                requested += (
                    ("maximum_scratch_bytes", self.limits.maximum_scratch_bytes),
                )
            raise MeshingFailure(
                MeshingFailureCategory.TIMED_OUT
                if error.status is MeshcoreStatus.TIMEOUT
                else MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Occupied-image host preparation exhausted its original storage allowance.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
                provider_code=error.status.name.lower(),
                requested=requested,
                achieved=(
                    ("source_storage_bytes_upper_requested", bytes_upper),
                    ("source_storage_bytes_upper_completed", completed),
                ),
            ) from error

    def host_array(self, shape: tuple[int, ...], dtype: DTypeLike, /) -> np.ndarray:
        """Own retained host payloads in the same allocator as native construction."""
        if self.execution_budget is None:
            return np.empty(shape, dtype=dtype)
        return self.execution_budget.allocate_host_array(shape, dtype)

    def consume(self, work: int = 1, *, queries: int = 0) -> None:
        if self.execution_budget is not None:
            self.execution_budget.charge(work=work, geometry_queries=queries)
        self.work_units += work
        self.geometry_queries += queries
        _check_image_preparation_limits(
            self.limits,
            (
                ("maximum_work_units", self.work_units),
                ("maximum_geometry_queries", self.geometry_queries),
            ),
        )
        if self.limits is not None and self.started is not None:
            elapsed = monotonic() - self.started
            if elapsed > self.limits.maximum_wall_seconds:
                raise MeshingFailure(
                    MeshingFailureCategory.TIMED_OUT,
                    "Occupied-image preparation exhausted its wall allowance.",
                    stage="source_inspection",
                    requested=(
                        ("maximum_wall_seconds", self.limits.maximum_wall_seconds),
                    ),
                    achieved=(("wall_seconds", elapsed), ("work_units", self.work_units)),
                )

    def record_native(self, work: int, /) -> None:
        """Carry an already charged child native phase without charging it twice."""
        self.work_units += work
        self.consume(0)


def _occupied_image_faces(
    code: np.ndarray,
    budget: _ImagePreparationBudget,
    /,
) -> Iterator[tuple[tuple[int, int, int, int, int], tuple[int, int]]]:
    """Yield each unequal voxel face once, with doubled-index plane ownership."""
    shape = code.shape
    for index in np.ndindex(shape):
        budget.consume()
        own = int(code[index])
        if own < 0:
            continue
        for axis, (first_axis, second_axis) in enumerate(_IMAGE_FACE_AXES):
            for direction in (-1, 1):
                budget.consume()
                neighbor = list(index)
                neighbor[axis] += direction
                other = (
                    int(code[tuple(neighbor)])
                    if 0 <= neighbor[axis] < shape[axis]
                    else -1
                )
                if other == own or (other >= 0 and direction < 0):
                    continue
                yield (
                    (axis, 2 * index[axis] + direction, direction, other, own),
                    (index[first_axis], index[second_axis]),
                )


def _partition_image_faces(
    faces: dict[tuple[int, int, int, int, int], set[tuple[int, int]]],
    budget: _ImagePreparationBudget,
    /,
) -> tuple[_ImageFaceRectangle, ...]:
    """Consume each stratum into deterministic, inclusion-maximal rectangles."""
    rectangles = []
    for stratum, remaining in sorted(faces.items()):
        for first, second in sorted(remaining):
            budget.consume()
            if (first, second) not in remaining:
                continue
            first_upper = first + 1
            while True:
                budget.consume()
                if (first_upper, second) not in remaining:
                    break
                first_upper += 1
            second_upper = second + 1
            while True:
                for value in range(first, first_upper):
                    budget.consume()
                    if (value, second_upper) not in remaining:
                        break
                else:
                    second_upper += 1
                    continue
                break
            for value in range(first, first_upper):
                for row in range(second, second_upper):
                    budget.consume()
                    remaining.remove((value, row))
            rectangles.append(
                _ImageFaceRectangle(
                    stratum,
                    (first, second, first_upper, second_upper),
                )
            )
    return tuple(rectangles)


def _positive_image_quads(
    exact: Mapping[tuple[int, int], tuple[Fraction, ...]],
    bounds: tuple[int, int, int, int],
    budget: _ImagePreparationBudget,
    projection: int,
    sign: int,
    /,
) -> bool:
    """Certify every original tile is a strictly oriented convex planar disk."""
    lower_first, lower_second, upper_first, upper_second = bounds
    first_projection, second_projection = _IMAGE_FACE_AXES[projection]
    for first in range(lower_first, upper_first):
        for second in range(lower_second, upper_second):
            quad = (
                exact[first, second],
                exact[first + 1, second],
                exact[first + 1, second + 1],
                exact[first, second + 1],
            )
            for index in range(4):
                budget.consume()
                a, b, c = (quad[(index + offset) % 4] for offset in range(3))
                orientation = (b[first_projection] - a[first_projection]) * (
                    c[second_projection] - a[second_projection]
                ) - (b[second_projection] - a[second_projection]) * (
                    c[first_projection] - a[first_projection]
                )
                if sign * orientation <= 0:
                    return False
    return True


def _straight_image_rectangle_boundary(
    exact: Mapping[tuple[int, int], tuple[Fraction, ...]],
    bounds: tuple[int, int, int, int],
    budget: _ImagePreparationBudget,
    /,
) -> bool:
    """Certify each source boundary chain equals its replacement segment."""
    lower_first, lower_second, upper_first, upper_second = bounds
    boundary = (
        tuple(
            exact[first, lower_second] for first in range(lower_first, upper_first + 1)
        ),
        tuple(
            exact[upper_first, second] for second in range(lower_second, upper_second + 1)
        ),
        tuple(
            exact[first, upper_second]
            for first in range(upper_first, lower_first - 1, -1)
        ),
        tuple(
            exact[lower_first, second]
            for second in range(upper_second, lower_second - 1, -1)
        ),
    )
    for edge in boundary:
        start, end = edge[0], edge[-1]
        varying = next((index for index in range(3) if start[index] != end[index]), None)
        if varying is None:
            return False
        previous = Fraction(-1)
        for point in edge:
            budget.consume()
            parameter = (point[varying] - start[varying]) / (
                end[varying] - start[varying]
            )
            if not previous < parameter <= 1:
                return False
            if any(
                point[index] != start[index] + parameter * (end[index] - start[index])
                for index in range(3)
            ):
                return False
            previous = parameter
    return True


def _exact_image_rectangle(
    rectangle: _ImageFaceRectangle,
    points: dict[tuple[int, int, int], np.ndarray],
    budget: _ImagePreparationBudget,
    /,
) -> bool:
    """Prove equality to the original triangle union in represented coordinates.

    No affine or tolerance-based planarity assumption is made. Exact coplanarity,
    positive elementary quads and a straight, ordered boundary make the original
    oriented lattice disk cover precisely this rectangle, for either diagonal.
    """
    axis = rectangle.stratum[0]
    first_axis, second_axis = _IMAGE_FACE_AXES[axis]
    lower_first, lower_second, upper_first, upper_second = rectangle.bounds
    exact = {}
    for first in range(lower_first, upper_first + 1):
        for second in range(lower_second, upper_second + 1):
            budget.consume(3)
            key = [0, 0, 0]
            key[axis] = rectangle.stratum[1]
            key[first_axis] = 2 * first - 1
            key[second_axis] = 2 * second - 1
            exact[first, second] = tuple(
                Fraction(float(value)) for value in points[key[0], key[1], key[2]]
            )
    corners = (
        exact[lower_first, lower_second],
        exact[upper_first, lower_second],
        exact[upper_first, upper_second],
        exact[lower_first, upper_second],
    )
    origin, first_corner, second_corner, _ = corners
    first_vector = tuple(
        value - start for value, start in zip(first_corner, origin, strict=True)
    )
    second_vector = tuple(
        value - start for value, start in zip(second_corner, origin, strict=True)
    )
    normal = tuple(
        first_vector[a] * second_vector[b] - first_vector[b] * second_vector[a]
        for a, b in _IMAGE_FACE_AXES
    )
    projection = next((index for index, value in enumerate(normal) if value), None)
    if projection is None:
        return False
    for point in exact.values():
        budget.consume()
        if sum(
            value * (coordinate - start)
            for value, coordinate, start in zip(normal, point, origin, strict=True)
        ):
            return False
    sign = 1 if normal[projection] > 0 else -1
    return _positive_image_quads(
        exact, rectangle.bounds, budget, projection, sign
    ) and _straight_image_rectangle_boundary(exact, rectangle.bounds, budget)


def _conform_image_polygon_edges(
    loops: tuple[tuple[tuple[int, int, int], ...], ...],
    budget: _ImagePreparationBudget,
    /,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Split partial lattice edges at all retained polygon/junction endpoints."""
    endpoints = {key for loop in loops for key in loop}
    lines: dict[tuple[int, int, int], list[int]] = {}
    for key in endpoints:
        for axis, (first, second) in enumerate(_IMAGE_FACE_AXES):
            budget.consume()
            lines.setdefault((axis, key[first], key[second]), []).append(key[axis])
    for positions in lines.values():
        positions.sort()
    result = []
    for loop in loops:
        split = []
        for start, end in zip(loop, loop[1:] + loop[:1], strict=True):
            budget.consume()
            split.append(start)
            varying = tuple(axis for axis in range(3) if start[axis] != end[axis])
            if len(varying) != 1:
                # Retained source-triangle diagonals have no lattice endpoint
                # strictly inside their unit-square segment.
                continue
            axis = varying[0]
            first, second = _IMAGE_FACE_AXES[axis]
            positions = lines[axis, start[first], start[second]]
            lower, upper = sorted((start[axis], end[axis]))
            interior = positions[
                bisect_right(positions, lower) : bisect_left(positions, upper)
            ]
            if start[axis] > end[axis]:
                interior.reverse()
            for position in interior:
                budget.consume()
                key = list(start)
                key[axis] = position
                split.append((key[0], key[1], key[2]))
        result.append(tuple(split))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _ImagePolygonSchedule:
    """Canonical exact source polygons and their authoritative facet incidence."""

    loops: tuple[tuple[tuple[int, int, int], ...], ...]
    polygon_facets: tuple[int, ...]
    facet_regions: tuple[tuple[int, int], ...]


def _image_polygon_schedule(
    rectangles: tuple[_ImageFaceRectangle, ...],
    represented_points: dict[tuple[int, int, int], np.ndarray],
    reflected: bool,
    region_ids: tuple[str, ...],
    first_region_by_pair: Mapping[tuple[str, ...], str],
    budget: _ImagePreparationBudget,
    /,
) -> _ImagePolygonSchedule:
    """Freeze source-equivalent loops, preserving failed-proof source diagonals."""
    strata = tuple(sorted({rectangle.stratum for rectangle in rectangles}))
    facet_by_stratum = {stratum: index for index, stratum in enumerate(strata)}
    loops = []
    polygon_facets = []
    for rectangle in rectangles:
        _, _, direction, other, own = rectangle.stratum
        lower_first, lower_second, upper_first, upper_second = rectangle.bounds
        exact_rectangle = _exact_image_rectangle(rectangle, represented_points, budget)
        pieces = (
            (rectangle,)
            if exact_rectangle
            else tuple(
                _ImageFaceRectangle(
                    rectangle.stratum, (first, second, first + 1, second + 1)
                )
                for first in range(lower_first, upper_first)
                for second in range(lower_second, upper_second)
            )
        )
        rotate = (
            other >= 0
            and region_ids[own]
            != first_region_by_pair[tuple(sorted((region_ids[own], region_ids[other])))]
        )
        for piece in pieces:
            corners = piece.corners()
            if (direction < 0) ^ reflected:
                corners = corners[::-1]
            if rotate:
                # Reverse-oriented extracted squares use the other diagonal;
                # rotation preserves the authoritative source normal.
                corners = corners[1:] + corners[:1]
            polygons = (
                (corners,)
                if exact_rectangle
                else (
                    (corners[0], corners[1], corners[2]),
                    (corners[0], corners[2], corners[3]),
                )
            )
            for polygon in polygons:
                budget.consume()
                loops.append(polygon)
                polygon_facets.append(facet_by_stratum[rectangle.stratum])
    return _ImagePolygonSchedule(
        tuple(loops),
        tuple(polygon_facets),
        tuple((stratum[3], stratum[4]) for stratum in strata),
    )


def prepare_compartment_complex(
    source: CompartmentMeshingSource,
    /,
    *,
    limits: MeshingLimits | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    budget: _ImagePreparationBudget | None = None,
) -> PiecewiseLinearComplex:
    """Compile exact image cells, removing only proved redundant source strata.

    Disjoint maximal rectangles replace voxel triangles only after exact proof
    in the represented world coordinates. Failed proofs retain the original
    triangles, including their interface-authoritative diagonal. Native PLC
    preparation alone owns polygon triangulation and coplanar facet recovery.
    """
    if not isinstance(source, CompartmentMeshingSource):
        raise TypeError("source must be CompartmentMeshingSource.")
    if budget is None:
        execution = current_native_execution_budget()
        budget = _ImagePreparationBudget(
            limits,
            monotonic() if execution is None else _native_volume_operation_started(),
            execution_budget=execution,
        )
    if budget.execution_budget is None:
        return _prepare_compartment_complex_bound(source, limits, record_phase, budget)
    with budget.execution_budget.host_workspace() as storage:
        budget.storage = storage
        try:
            return _prepare_compartment_complex_bound(
                source, limits, record_phase, budget
            )
        finally:
            budget.storage = None


def _prepare_compartment_complex_bound(
    source: CompartmentMeshingSource,
    limits: MeshingLimits | None,
    record_phase: NativeMeshingPhaseRecorder | None,
    budget: _ImagePreparationBudget,
    /,
) -> PiecewiseLinearComplex:
    started = phase_started(record_phase)
    labels = source.labels
    voxel_count = int(labels.asset.values.size)
    code_bytes = 32 * voxel_count + 512 * len(labels.ontology.labels)
    _check_image_preparation_limits(
        limits,
        (
            ("maximum_scratch_bytes", code_bytes),
            ("maximum_work_units", voxel_count),
        ),
    )
    budget.admit_storage(code_bytes)
    region_ids = tuple(value.compartment_id for value in source.compartments.compartments)
    ontology = {value.label_id: value.value for value in labels.ontology.labels}
    code = budget.host_array(labels.asset.values.shape, np.int32)
    code.fill(-1)
    values = np.asarray(labels.asset.values)
    valid = np.asarray(labels.asset.valid_mask, dtype=np.bool_)
    for region, definition in enumerate(source.compartments.compartments):
        budget.consume(voxel_count)
        selected = np.asarray(
            [ontology[name] for name in definition.label_ids], dtype=np.int64
        )
        code[valid & np.isin(values, selected)] = region
    face_count = sum(1 for _ in _occupied_image_faces(code, budget))
    # Covers host dictionaries, sorted copies, exact binary-rational coordinates
    # (including subnormals), conforming loops and simultaneous PLC array copies.
    scratch_bytes = code_bytes + 8192 * face_count
    budget.admit_storage(scratch_bytes)
    faces: dict[tuple[int, int, int, int, int], set[tuple[int, int]]] = {}
    original_keys: set[tuple[int, int, int]] = set()
    for stratum, (first, second) in _occupied_image_faces(code, budget):
        faces.setdefault(stratum, set()).add((first, second))
        original_keys.update(
            _ImageFaceRectangle(
                stratum,
                (first, second, first + 1, second + 1),
            ).corners()
        )
    ordered_keys = tuple(sorted(original_keys))
    budget.consume(len(ordered_keys), queries=len(ordered_keys))
    original_points = labels.asset.spatial_affine.index_to_world(
        np.asarray(ordered_keys, dtype=np.float64) * 0.5,
    )
    represented_points = dict(zip(ordered_keys, original_points, strict=True))
    reflected = np.linalg.det(labels.asset.spatial_affine.matrix[:3, :3]) < 0.0
    first_region_by_pair = {
        tuple(sorted((first, second))): first
        for _, first, second, _ in source.interface_definitions
    }
    schedule = _image_polygon_schedule(
        _partition_image_faces(faces, budget),
        represented_points,
        reflected,
        region_ids,
        first_region_by_pair,
        budget,
    )
    conforming_loops = _conform_image_polygon_edges(schedule.loops, budget)
    retained_keys = tuple(sorted({key for loop in conforming_loops for key in loop}))
    vertices = {key: index for index, key in enumerate(retained_keys)}
    triangles = sum(len(loop) - 2 for loop in conforming_loops)
    edges = {
        tuple(sorted((start, end)))
        for loop in conforming_loops
        for start, end in zip(loop, loop[1:] + loop[:1], strict=True)
    }
    edge_count = len(edges) + sum(len(loop) - 3 for loop in conforming_loops)
    entries = sum(len(loop) for loop in conforming_loops)
    data_bytes = 24 * len(vertices) + 8 * (
        entries + 2 * len(conforming_loops) + 1 + 2 * len(schedule.facet_regions)
    )
    _check_image_preparation_limits(
        limits,
        (
            ("maximum_vertices", len(vertices)),
            ("maximum_edges", edge_count),
            ("maximum_faces", triangles),
            ("maximum_connectivity_entries", 3 * triangles),
            ("maximum_data_bytes", data_bytes),
        ),
    )
    points = budget.host_array((len(retained_keys), 3), np.float64)
    for index, key in enumerate(retained_keys):
        points[index] = represented_points[key]
    polygon_facets = budget.host_array((len(schedule.polygon_facets),), np.int64)
    polygon_facets[:] = schedule.polygon_facets
    facet_regions = budget.host_array((len(schedule.facet_regions), 2), np.int64)
    facet_regions[:] = schedule.facet_regions
    complex_ = PiecewiseLinearComplex(
        points,
        tuple(
            np.asarray([vertices[key] for key in loop], dtype=np.int64)
            for loop in conforming_loops
        ),
        polygon_facets,
        facet_regions,
        region_ids,
        boundary="conforming",
    )
    budget.consume(0)
    record_elapsed(record_phase, "image_interpretation", started)
    return complex_


def _scope(mesh: CellMesh, dimension: int, identifiers: np.ndarray, /) -> MeshingScope:
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        mesh.entity_set(dimension).entity_set_id,
        np.sort(identifiers),
    )


def _material_patches(
    mesh: CellMesh,
    zones: tuple[MeshZone, ...],
    assignments: tuple[str, ...],
    definitions: tuple[tuple[str, str, str, bool], ...],
    /,
    *,
    patch_names: Mapping[str, str] | None = None,
) -> tuple[
    tuple[MeshPatch, ...],
    tuple[tuple[str, int, int, str, str], ...],
    tuple[tuple[str, str], ...],
]:
    facets = material_interface_facets(mesh, assignments, definitions)
    zone_ids = {
        region: zone.zone_id
        for region, zone in zip(sorted(set(assignments)), zones, strict=True)
    }
    patches = tuple(
        MeshPatch(
            name if patch_names is None else patch_names.get(name, name),
            _scope(
                mesh,
                2,
                np.asarray(
                    [facet for interface, facet, _, _, _ in facets if interface == name],
                    dtype=np.int64,
                ),
            ),
            connected=False,
            adjacent_zone_ids=(zone_ids[first], zone_ids[second]),
        )
        for name, first, second, _ in definitions
        if any(interface == name for interface, _, _, _, _ in facets)
    )
    observed = tuple(
        name
        for name, _, _, _ in definitions
        if any(interface == name for interface, _, _, _, _ in facets)
    )
    return (
        patches,
        facets,
        tuple(
            (name, patch.patch_id) for name, patch in zip(observed, patches, strict=True)
        ),
    )


def _coverage_failure(
    kind: str,
    certificate: DomainCoverageCertificate | GlobalEmbeddingCertificate,
    /,
) -> MeshingFailure:
    """Carry the original certificate refusal, including its owning resource."""
    findings = certificate.findings
    resources = tuple(value for value in findings if value.resource is not None)
    requested = tuple(
        sorted(
            (
                f"certificate:{certificate.certificate_id}:{value.finding_id}:{value.check}:{value.resource}:{key}",
                count,
            )
            for value in resources
            for key, count in value.requested
        )
    )
    achieved = tuple(
        sorted(
            (
                f"certificate:{certificate.certificate_id}:{value.finding_id}:{value.check}:{value.resource}:{key}",
                count,
            )
            for value in resources
            for key, count in value.achieved
        )
    )
    return MeshingFailure(
        MeshingFailureCategory.RESOURCE_EXHAUSTED
        if resources
        else MeshingFailureCategory.COMPLIANCE_FAILED,
        f"Compartment {kind} is not certified: {', '.join(value.check for value in findings)}.",
        stage="certification",
        entity_ids=tuple(
            sorted(
                {
                    identifier
                    for value in findings
                    if value.entity_kind in ("cell", "facet")
                    for identifier in value.entity_ids
                }
            )
        ),
        requested=requested,
        achieved=achieved,
        checkpoint_id=certificate.certificate_id,
    )


def _compartment_outer_domain(
    source: CompartmentMeshingSource,
    budget: _ImagePreparationBudget,
    /,
) -> PiecewiseLinearDomain:
    """Bind and bound the independently supplied outer-source triangle carrier."""
    outer = source.outer_surface.mesh
    if any(block.cell_kind != "triangle" for block in outer.blocks):
        raise TypeError("Compartment outer surfaces must contain triangles.")
    face_count = sum(block.vertices.shape[0] for block in outer.blocks)
    budget.consume(outer.coordinates.shape[0] + face_count)
    _check_image_preparation_limits(
        budget.limits,
        (
            ("maximum_faces", face_count),
            ("maximum_connectivity_entries", 3 * face_count),
            ("maximum_scratch_bytes", outer.coordinates.size * 8 + 64 * face_count),
        ),
    )
    outer_faces = budget.host_array((face_count, 3), np.int64)
    offset = 0
    for block in outer.blocks:
        count = block.vertices.shape[0]
        outer_faces[offset : offset + count] = np.asarray(block.vertices, dtype=np.int64)
        offset += count
    outer_points = budget.host_array(outer.coordinates.shape, np.float64)
    outer_points[:] = np.asarray(outer.coordinates, dtype=np.float64)
    regions = budget.host_array((face_count, 2), np.int64)
    regions[:, 0], regions[:, 1] = 0, -1
    return PiecewiseLinearDomain(
        outer_points,
        outer_faces,
        regions,
        ("occupied-domain",),
        source_id=source.source_id,
    )


@checked
def generate_compartment_volume(
    source: CompartmentMeshingSource,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    input_id: str | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    operation_started: float | None = None,
    execution_budget: NativeExecutionBudget | None = None,
) -> CompartmentConstruction:
    """Recover one material PLC and independently certify image/outer coverage."""
    active = current_native_execution_budget()
    if execution_budget is not None and execution_budget is not active:
        raise RuntimeError("Compartment generation must borrow the actual active budget.")
    started = _native_volume_operation_started(operation_started)
    if active is None and started is None:
        started = monotonic()
    with native_volume_execution_budget(
        specification.limits,
        operation_started=started,
    ) as execution:
        construction = _generate_compartment_volume_bound(
            source,
            specification,
            schedule,
            validity_policy=validity_policy,
            input_id=input_id,
            certificate_limits=certificate_limits,
            record_phase=record_phase,
            operation_started=started,
            execution_budget=execution,
        )
    if active is None:
        construction = replace(
            construction,
            volume=_bind_volume_execution(construction.volume, execution, 0, 0),
        )
    return construction


def _generate_compartment_volume_bound(
    source: CompartmentMeshingSource,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    input_id: str | None,
    certificate_limits: MeshCertificateLimits | None,
    record_phase: NativeMeshingPhaseRecorder | None,
    operation_started: float | None,
    execution_budget: NativeExecutionBudget,
) -> CompartmentConstruction:
    budget = _ImagePreparationBudget(
        specification.limits,
        operation_started,
        execution_budget=execution_budget,
    )
    complex_ = prepare_compartment_complex(
        source,
        limits=specification.limits,
        record_phase=record_phase,
        budget=budget,
    )
    outer_domain = _compartment_outer_domain(source, budget)
    volume = generate_plc_volume(
        complex_,
        specification,
        schedule,
        validity_policy=validity_policy,
        source_id=source.source_id,
        source_revision=source.source_revision,
        input_id=source.source_id if input_id is None else input_id,
        record_phase=record_phase,
        source_work_units=budget.work_units,
        operation_started=budget.started,
        source_geometry_queries=budget.geometry_queries,
        execution_budget=execution_budget,
    )
    mesh = volume.mesh
    with measure_phase(record_phase, "construction"):
        geometry = CellGeometrySpec.affine(mesh)
    with measure_phase(record_phase, "certification"):
        native_volume_checkpoint(specification.limits, MeshingStageKind.CERTIFICATION)
        validity = certify_cell_geometry_validity(
            geometry, mesh=mesh, policy=validity_policy
        )
    with measure_phase(record_phase, "certification"):
        native_volume_checkpoint(specification.limits, MeshingStageKind.CERTIFICATION)
        embedding = certify_global_embedding(
            mesh, geometry, validity, limits=certificate_limits
        )
    if embedding.status != "certified":
        raise _coverage_failure("global embedding", embedding)
    with measure_phase(record_phase, "certification"):
        native_volume_checkpoint(specification.limits, MeshingStageKind.CERTIFICATION)
        coverage = certify_domain_coverage(
            mesh,
            geometry,
            volume.domain,
            volume.cell_regions,
            embedding=embedding,
            limits=certificate_limits,
        )
    if coverage.status != "certified":
        raise _coverage_failure("material coverage", coverage)
    # The declared outer surface is checked independently with all materials
    # merged. This admits a coarse outer triangulation but cannot silently
    # replace an incompatible envelope with the image-cell boundary.
    budget.consume(0)
    with measure_phase(record_phase, "certification"):
        native_volume_checkpoint(specification.limits, MeshingStageKind.CERTIFICATION)
        outer_coverage = certify_domain_coverage(
            mesh,
            geometry,
            outer_domain,
            np.zeros(volume.cell_regions.shape, dtype=np.int64),
            embedding=embedding,
            limits=certificate_limits,
        )
    if outer_coverage.status != "certified":
        raise _coverage_failure("outer-source coverage", outer_coverage)
    started = phase_started(record_phase)
    assignments = tuple(
        complex_.region_ids[value] for value in volume.cell_regions.tolist()
    )
    zones_by_region = {
        region: zone
        for region, zone in zip(complex_.region_ids, volume.zones, strict=True)
    }
    for control in specification.region_controls:
        zone = zones_by_region[control.region_name]
        zones_by_region[control.region_name] = MeshZone(
            zone.name,
            MeshZoneRole.REGION,
            zone.scope,
            material_id=control.material_id,
            region_role=control.role,
        )
    zones = tuple(zones_by_region[region] for region in sorted(zones_by_region))
    definitions = source.interface_definitions
    patch_names = {
        source.interface_definitions[np.asarray(control.scope.entity_ids).item()][
            0
        ]: control.name
        for control in specification.patch_controls
        if control.scope.entity_set_id == f"{source.source_id}:interfaces"
    }
    patches, facets, patch_bindings = _material_patches(
        mesh, zones, assignments, definitions, patch_names=patch_names
    )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise TypeError("The native compartment carrier must be tetrahedral.")
    boundary_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)[
        np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    ]
    boundary_name = next(
        (
            control.name
            for control in specification.patch_controls
            if control.scope.entity_set_id == f"{source.source_id}:boundary"
        ),
        "outer_boundary",
    )
    patches = patches + (
        MeshPatch(boundary_name, _scope(mesh, 2, boundary_ids), connected=False),
    )
    record_elapsed(record_phase, "organization", started)
    started = phase_started(record_phase)
    evidence = RegionMeshingEvidence(
        mesh,
        geometry,
        source.source_revision,
        source.compartments.complex_id,
        assignments,
        tuple((region, zone.zone_id) for region, zone in sorted(zones_by_region.items())),
        patch_bindings,
        definitions,
        facets,
        volume.domain,
        coverage,
    )
    record_elapsed(record_phase, "construction", started)
    started = phase_started(record_phase)
    evidence.require_source(source.compartments)
    evidence.require_current(mesh, zones, patches, geometry=geometry)
    record_elapsed(record_phase, "certification", started)
    native_volume_checkpoint(specification.limits, MeshingStageKind.CERTIFICATION)
    return CompartmentConstruction(
        replace(volume, zones=zones, patches=patches), evidence
    )


def generate_reconstructed_image_volume(
    source: LabelFieldVolumeBinding,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    input_id: str | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    operation_started: float | None = None,
    execution_budget: NativeExecutionBudget | None = None,
) -> CompartmentConstruction:
    """Fill the declared shared reconstruction, retaining sampled-source error."""
    if not isinstance(source, LabelFieldVolumeBinding):
        raise TypeError("source must be LabelFieldVolumeBinding.")
    active = current_native_execution_budget()
    if execution_budget is not None and execution_budget is not active:
        raise RuntimeError("Image generation must borrow the actual active budget.")
    started = _native_volume_operation_started(operation_started)
    if active is None and started is None:
        started = monotonic()
    with native_volume_execution_budget(
        specification.limits,
        operation_started=started,
    ) as execution:
        construction = _generate_reconstructed_image_volume_bound(
            source,
            specification,
            schedule,
            validity_policy=validity_policy,
            input_id=input_id,
            certificate_limits=certificate_limits,
            record_phase=record_phase,
            operation_started=started,
            execution_budget=execution,
        )
    if active is None:
        construction = replace(
            construction,
            volume=_bind_volume_execution(construction.volume, execution, 0, 0),
        )
    return construction


def _generate_reconstructed_image_volume_bound(
    source: LabelFieldVolumeBinding,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    input_id: str | None,
    certificate_limits: MeshCertificateLimits | None,
    record_phase: NativeMeshingPhaseRecorder | None,
    operation_started: float | None,
    execution_budget: NativeExecutionBudget,
) -> CompartmentConstruction:
    started = phase_started(record_phase)
    domain = source.domain
    budget = _ImagePreparationBudget(
        specification.limits,
        operation_started,
        execution_budget=execution_budget,
    )
    vertex_count, face_count = domain.vertices.shape[0], domain.facets.shape[0]
    budget.consume(vertex_count + face_count)
    _check_image_preparation_limits(
        specification.limits,
        (
            ("maximum_vertices", vertex_count),
            ("maximum_faces", face_count),
            ("maximum_connectivity_entries", domain.facets.size),
            ("maximum_data_bytes", domain.vertices.nbytes + 56 * face_count + 8),
            ("maximum_scratch_bytes", domain.vertices.nbytes + 192 * face_count),
        ),
    )
    edges = np.unique(
        np.sort(domain.facets[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1),
        axis=0,
    )
    _check_image_preparation_limits(
        specification.limits, (("maximum_edges", edges.shape[0]),)
    )
    complex_ = PiecewiseLinearComplex(
        domain.vertices,
        tuple(domain.facets),
        np.arange(domain.facets.shape[0], dtype=np.int64),
        domain.facet_regions[:, ::-1],
        domain.region_ids,
        boundary="conforming",
    )
    record_elapsed(record_phase, "image_interpretation", started)
    volume = generate_plc_volume(
        complex_,
        specification,
        schedule,
        validity_policy=validity_policy,
        source_id=source.source_id,
        source_revision=source.source_revision,
        input_id=source.binding_id if input_id is None else input_id,
        record_phase=record_phase,
        source_work_units=budget.work_units,
        operation_started=budget.started,
        source_geometry_queries=budget.geometry_queries,
        execution_budget=execution_budget,
    )
    mesh = volume.mesh
    with measure_phase(record_phase, "construction"):
        geometry = CellGeometrySpec.affine(mesh)
    with measure_phase(record_phase, "certification"):
        native_volume_checkpoint(specification.limits, MeshingStageKind.CERTIFICATION)
        validity = certify_cell_geometry_validity(
            geometry, mesh=mesh, policy=validity_policy
        )
    with measure_phase(record_phase, "certification"):
        native_volume_checkpoint(specification.limits, MeshingStageKind.CERTIFICATION)
        embedding = certify_global_embedding(
            mesh, geometry, validity, limits=certificate_limits
        )
    if embedding.status != "certified":
        raise _coverage_failure("global embedding", embedding)
    with measure_phase(record_phase, "certification"):
        native_volume_checkpoint(specification.limits, MeshingStageKind.CERTIFICATION)
        coverage = certify_domain_coverage(
            mesh,
            geometry,
            domain,
            volume.cell_regions,
            embedding=embedding,
            limits=certificate_limits,
        )
    if coverage.status != "certified":
        raise _coverage_failure(
            "reconstructed-interface coverage",
            coverage,
        )
    started = phase_started(record_phase)
    assignments = tuple(
        domain.region_ids[value] for value in volume.cell_regions.tolist()
    )
    zones_by_region = {
        region: zone for region, zone in zip(domain.region_ids, volume.zones, strict=True)
    }
    for control in specification.region_controls:
        zone = zones_by_region[control.region_name]
        zones_by_region[control.region_name] = MeshZone(
            zone.name,
            MeshZoneRole.REGION,
            zone.scope,
            material_id=control.material_id,
            region_role=control.role,
        )
    zones = tuple(zones_by_region[region] for region in sorted(zones_by_region))
    patch_names = {
        source.interface_definitions[np.asarray(control.scope.entity_ids).item()][
            0
        ]: control.name
        for control in specification.patch_controls
        if control.scope.entity_set_id == f"{source.source_id}:interfaces"
    }
    patches, facets, patch_bindings = _material_patches(
        mesh, zones, assignments, source.interface_definitions, patch_names=patch_names
    )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise TypeError("The reconstructed-image carrier must be tetrahedral.")
    boundary_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)[
        np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    ]
    boundary_name = next(
        (
            control.name
            for control in specification.patch_controls
            if control.scope.entity_set_id == f"{source.source_id}:boundary"
        ),
        "outer_boundary",
    )
    patches += (MeshPatch(boundary_name, _scope(mesh, 2, boundary_ids), connected=False),)
    record_elapsed(record_phase, "organization", started)
    started = phase_started(record_phase)
    evidence = RegionMeshingEvidence(
        mesh,
        geometry,
        source.source_revision,
        source.binding_id,
        assignments,
        tuple((region, zone.zone_id) for region, zone in sorted(zones_by_region.items())),
        patch_bindings,
        source.interface_definitions,
        facets,
        domain,
        coverage,
    )
    record_elapsed(record_phase, "construction", started)
    started = phase_started(record_phase)
    evidence.require_current(mesh, zones, patches, geometry=geometry)
    record_elapsed(record_phase, "certification", started)
    native_volume_checkpoint(specification.limits, MeshingStageKind.CERTIFICATION)
    return CompartmentConstruction(
        replace(volume, domain=domain, zones=zones, patches=patches), evidence
    )


@dataclass(frozen=True, slots=True)
class RegionMeshingRenewal:
    zones: tuple[MeshZone, ...]
    patches: tuple[MeshPatch, ...]
    region_evidence: RegionMeshingEvidence
    source_preparation: PreparedPlcSource | None = None
    native_execution_evidence: NativeExecutionEvidence | None = None


def revalidate_region_evidence(
    source: CellMeshingResult,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    zones: tuple[MeshZone, ...],
    patches: tuple[MeshPatch, ...],
    /,
    *,
    lineage: MeshLineage | None = None,
    common_refinement: PreparedCommonRefinement | None = None,
    compartment_source: CompartmentMeshingSource | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
    limits: MeshingLimits | None = None,
) -> RegionMeshingRenewal:
    """Renew source, material and outer certificates under one original allowance."""
    renewal_limits = MeshingLimits() if limits is None else limits
    started = _native_volume_operation_started()
    if current_native_execution_budget() is None and started is None:
        started = monotonic()
    with native_volume_execution_budget(
        renewal_limits,
        operation_started=started,
        stage=MeshingStageKind.SOURCE_INSPECTION,
    ) as budget:
        renewal = _renew_region_evidence(
            source,
            mesh,
            geometry,
            zones,
            patches,
            lineage=lineage,
            common_refinement=common_refinement,
            compartment_source=compartment_source,
            certificate_limits=certificate_limits,
            limits=renewal_limits,
            execution_budget=budget,
            operation_started=started,
        )
    return replace(renewal, native_execution_evidence=budget.evidence)


def _renew_region_evidence(
    source: CellMeshingResult,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    zones: tuple[MeshZone, ...],
    patches: tuple[MeshPatch, ...],
    /,
    *,
    lineage: MeshLineage | None = None,
    common_refinement: PreparedCommonRefinement | None = None,
    compartment_source: CompartmentMeshingSource | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
    limits: MeshingLimits | None = None,
    execution_budget: NativeExecutionBudget,
    operation_started: float | None,
) -> RegionMeshingRenewal:
    """Remap exact material ancestry, rebuild organization and refresh coverage.

    Coarsening/merging across materials, unknown ancestry without a complete
    geometric overlap witness, moved source interfaces and stale source
    revisions fail before a result is published.
    Fixed topology retains assignments, never a geometry-bound certificate.
    An explicitly changed compartment source supplies a new authoritative
    domain; it is accepted only when the remapped assignments cover it exactly.
    """
    from ..geometry._supermesh import (
        CommonRefinementCoverage,
        CommonRefinementStatus,
        PreparedCommonRefinement,
    )
    from ._lineage import EntityLineageKind

    evidence = source.region_evidence
    if evidence is None:
        raise ValueError("Region renewal requires source material evidence.")
    evidence.require_current(
        source.mesh, source.zones, source.patches, geometry=source.geometry
    )
    renewal_limits = MeshingLimits() if limits is None else limits
    budget = _ImagePreparationBudget(
        renewal_limits,
        operation_started,
        execution_budget=execution_budget,
    )
    _check_image_preparation_limits(
        renewal_limits,
        (
            ("maximum_vertices", mesh.coordinates.shape[0]),
            ("maximum_edges", mesh.entity_set(1).entity_ids.size),
            ("maximum_faces", mesh.entity_set(2).entity_ids.size),
            ("maximum_cells", mesh.entity_set(3).entity_ids.size),
            (
                "maximum_connectivity_entries",
                sum(block.vertices.size for block in mesh.blocks),
            ),
            (
                "maximum_data_bytes",
                mesh.coordinates.nbytes
                + sum(block.vertices.nbytes for block in mesh.blocks),
            ),
        ),
    )
    target_ids = tuple(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64).tolist())
    budget.consume(len(target_ids))
    old = dict(zip(evidence.cell_global_ids, evidence.cell_region_ids, strict=True))
    if common_refinement is not None:
        if not isinstance(common_refinement, PreparedCommonRefinement):
            raise TypeError("common_refinement must be PreparedCommonRefinement or None.")
        overlap_source_ids = tuple(
            np.asarray(common_refinement.source_cell_global_ids, dtype=np.int64).tolist()
        )
        overlap_target_ids = tuple(
            np.asarray(common_refinement.target_cell_global_ids, dtype=np.int64).tolist()
        )
        if (
            common_refinement.status is not CommonRefinementStatus.SUCCESS
            or common_refinement.policy.coverage is not CommonRefinementCoverage.COMPLETE
            or common_refinement.source_mesh_id != source.mesh.mesh_id
            or common_refinement.target_mesh_id != mesh.mesh_id
            or common_refinement.source_topology_id != source.mesh.topology_id
            or common_refinement.target_topology_id != mesh.topology_id
            or set(overlap_source_ids) != set(evidence.cell_global_ids)
            or len(overlap_source_ids) != len(evidence.cell_global_ids)
            or set(overlap_target_ids) != set(target_ids)
            or len(overlap_target_ids) != len(target_ids)
        ):
            raise ValueError(
                "Material reclassification requires complete bound overlaps."
            )
        offsets = np.asarray(common_refinement.target_offsets, dtype=np.int64)
        source_rows = np.asarray(common_refinement.source_cells, dtype=np.int64)
        volumes = np.asarray(common_refinement.volumes, dtype=np.float64)
        if (
            offsets.shape != (len(target_ids) + 1,)
            or source_rows.shape != volumes.shape
            or np.any(volumes <= 0.0)
            or np.any(source_rows < 0)
            or np.any(source_rows >= len(evidence.cell_region_ids))
        ):
            raise ValueError("Material overlap rows are incomplete or nonpositive.")
        assigned_regions: dict[int, str] = {}
        for row in range(len(target_ids)):
            regions = {
                old[overlap_source_ids[value]]
                for value in source_rows[offsets[row] : offsets[row + 1]].tolist()
            }
            if len(regions) != 1:
                raise ValueError(
                    "A target cell is uncovered or overlaps distinct materials."
                )
            assigned_regions[overlap_target_ids[row]] = next(iter(regions))
        # Overlaps propose ownership; the independent exact material-domain
        # certificate below, not overlap tolerances, establishes conformity.
        assignments = tuple(assigned_regions[value] for value in target_ids)
    elif mesh.topology_id == source.mesh.topology_id:
        if set(target_ids) != set(old):
            raise ValueError("Fixed topology changed material cell identities.")
        assignments = tuple(old[value] for value in target_ids)
    else:
        if (
            lineage is None
            or lineage.source_topology_id != source.mesh.topology_id
            or lineage.target_topology_id != mesh.topology_id
        ):
            raise ValueError(
                "Material renewal requires exact source/target cell lineage."
            )
        relation = lineage.entity_lineage(3)
        if (
            relation.source_entity_set_id != source.mesh.entity_set(3).entity_set_id
            or relation.target_entity_set_id != mesh.entity_set(3).entity_set_id
        ):
            raise ValueError("Material lineage identifies different cell entity sets.")
        if np.asarray(relation.created_target_ids).size:
            raise ValueError(
                "Created material cells require certified geometric reclassification."
            )
        by_target: dict[int, set[str]] = {}
        for first, second, kind in zip(
            np.asarray(relation.source_global_ids).tolist(),
            np.asarray(relation.target_global_ids).tolist(),
            np.asarray(relation.relation_kinds).tolist(),
            strict=True,
        ):
            if (
                kind
                in (
                    int(EntityLineageKind.UNKNOWN),
                    int(EntityLineageKind.GENERATED_ON_GEOMETRY),
                )
                or first not in old
            ):
                raise ValueError("Material lineage contains unknown scientific ancestry.")
            by_target.setdefault(second, set()).add(old[first])
        if set(by_target) != set(target_ids) or any(
            len(value) != 1 for value in by_target.values()
        ):
            raise ValueError(
                "Material ancestry is incomplete or merges distinct regions."
            )
        assignments = tuple(next(iter(by_target[value])) for value in target_ids)
    domain = evidence.domain
    revision = evidence.source_revision
    complex_id = evidence.source_complex_id
    definitions = evidence.interface_definitions
    source_preparation = None
    outer_domain = None
    if compartment_source is not None:
        if (
            compartment_source.coordinate_contract.spatial_id
            != source.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Renewed image sources require the accepted spatial contract."
            )
        if tuple(
            value.compartment_id for value in compartment_source.compartments.compartments
        ) != tuple(sorted(evidence.domain.region_ids)):
            raise ValueError(
                "Source renewal cannot change authoritative material identities."
            )
        complex_ = prepare_compartment_complex(
            compartment_source,
            limits=renewal_limits,
            budget=budget,
        )
        outer_domain = _compartment_outer_domain(compartment_source, budget)
        source_preparation = prepare_plc_source(
            complex_,
            compartment_source.source_id,
            compartment_source.source_revision,
            source.coordinate_contract,
            limits=renewal_limits,
            source_work_units=budget.work_units,
            source_geometry_queries=budget.geometry_queries,
            operation_started=budget.started,
        )
        budget.record_native(dict(source_preparation.preparation_counters)["work_units"])
        domain = source_preparation.association_transfer.domain
        revision = compartment_source.source_revision
        complex_id = compartment_source.compartments.complex_id
        definitions = compartment_source.interface_definitions
    region_ids = tuple(sorted(domain.region_ids))
    old_zones = {value.zone_id: value for value in source.zones}
    source_region_zones = {
        region: old_zones[zone_id] for region, zone_id in evidence.region_zone_ids
    }
    region_zones = tuple(
        MeshZone(
            source_region_zones[region].name,
            MeshZoneRole.REGION,
            _scope(
                mesh,
                3,
                np.asarray(
                    [
                        value
                        for value, assigned in zip(target_ids, assignments, strict=True)
                        if assigned == region
                    ],
                    dtype=np.int64,
                ),
            ),
            material_id=source_region_zones[region].material_id,
            region_role=source_region_zones[region].region_role,
        )
        for region in region_ids
    )
    old_patches = {value.patch_id: value for value in source.patches}
    patch_names = {
        interface: old_patches[patch_id].name
        for interface, patch_id in evidence.interface_patch_ids
    }
    material_patches, facets, patch_bindings = _material_patches(
        mesh, region_zones, assignments, definitions, patch_names=patch_names
    )
    old_zone_ids = {value[1] for value in evidence.region_zone_ids}
    # Inherited zone IDs necessarily change with their scope. Select the source
    # material zone identity through its exact assignment, never by its name.
    material_selections = {
        frozenset(value.scope.entity_ids.tolist()) for value in region_zones
    }
    remaining_zones = tuple(
        value
        for value in zones
        if value.zone_id not in old_zone_ids
        and not (
            value.role is MeshZoneRole.REGION
            and frozenset(np.asarray(value.scope.entity_ids).tolist())
            in material_selections
        )
    )
    interface_ids = {facet for _, facet, _, _, _ in facets}
    remaining_patches = tuple(
        value
        for value in patches
        if not (
            value.scope.entity_dimension == 2
            and set(np.asarray(value.scope.entity_ids).tolist()) <= interface_ids
        )
    )
    zones_ = remaining_zones + region_zones
    patches_ = remaining_patches + material_patches
    native_volume_checkpoint(renewal_limits, MeshingStageKind.CERTIFICATION)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = certify_global_embedding(
        mesh, geometry, validity, limits=certificate_limits
    )
    native_volume_checkpoint(renewal_limits, MeshingStageKind.CERTIFICATION)
    if embedding.status != "certified":
        raise _coverage_failure("renewed global embedding", embedding)
    assignment_by_id = dict(zip(target_ids, assignments, strict=True))
    coverage_regions = np.asarray(
        [
            domain.region_ids.index(assignment_by_id[value])
            for block in mesh.blocks
            for value in np.asarray(block.global_ids, dtype=np.int64).tolist()
        ],
        dtype=np.int64,
    )
    native_volume_checkpoint(renewal_limits, MeshingStageKind.CERTIFICATION)
    coverage = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        coverage_regions,
        embedding=embedding,
        limits=certificate_limits,
    )
    if coverage.status != "certified":
        raise _coverage_failure("renewed material coverage", coverage)
    if outer_domain is not None:
        native_volume_checkpoint(renewal_limits, MeshingStageKind.CERTIFICATION)
        outer_coverage = certify_domain_coverage(
            mesh,
            geometry,
            outer_domain,
            np.zeros(len(target_ids), dtype=np.int64),
            embedding=embedding,
            limits=certificate_limits,
        )
        if outer_coverage.status != "certified":
            raise _coverage_failure(
                "renewed outer-source coverage",
                outer_coverage,
            )
    budget.consume(0)
    renewed = RegionMeshingEvidence(
        mesh,
        geometry,
        revision,
        complex_id,
        assignments,
        tuple(
            (region, value.zone_id)
            for region, value in zip(region_ids, region_zones, strict=True)
        ),
        patch_bindings,
        definitions,
        facets,
        domain,
        coverage,
    )
    renewed.require_current(mesh, zones_, patches_, geometry=geometry)
    if compartment_source is not None:
        renewed.require_source(compartment_source.compartments)
    return RegionMeshingRenewal(zones_, patches_, renewed, source_preparation)


__all__ = [
    "CompartmentConstruction",
    "generate_compartment_volume",
    "generate_reconstructed_image_volume",
    "prepare_compartment_complex",
    "RegionMeshingRenewal",
    "revalidate_region_evidence",
]
