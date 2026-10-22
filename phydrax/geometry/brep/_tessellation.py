#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Derive shared-edge CAD triangles through the native surface scheduler.

Carrier topology and model identity are unchanged by tessellation. The adapter
prepares a bounded request, executes the scheduler, certifies its source cover,
and expands authored occurrences without welding their vertices.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

from ..._meshcore import NativeExecutionScope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingLimits,
    SurfaceMeshingSpec,
)
from ...meshing._controls import FeatureKind, ProtectedFeature
from ...meshing._failures import MeshingFailure, MeshingFailureCategory
from ...meshing._scope import MeshingEntityKind, MeshingScope
from ...meshing._sizing import SizeControlStrength, UniformSizeControl
from ...meshing._surface_construction import SurfaceConstruction
from ...meshing.providers._native_schedules import NativeSurfaceSchedule
from ...typing import checked
from .._domain_boundary import MeshingDomainBoundarySource
from .._interval_enclosure import interval_add, interval_multiply
from .._meshing_domain import MeshingDomain
from .._rational import fraction_up, rational_dot, rational_vector
from .._source_boundary_records import SourceBoundaryChartCover
from ._constructors import BRepTessellationPolicy
from ._model import BRepEntityId, BRepGeometry
from ._patches import AbstractSurfacePatch


@dataclass(frozen=True, slots=True)
class BRepTessellation:
    vertices: np.ndarray
    triangles: np.ndarray
    face_ids: np.ndarray
    parameters: np.ndarray
    deviation_bounds: np.ndarray
    normal_bounds: np.ndarray
    vertex_source_dimensions: np.ndarray
    vertex_source_indices: np.ndarray
    vertex_parameters: np.ndarray
    chart_restriction_vertices: np.ndarray
    chart_restriction_edges: np.ndarray
    chart_restriction_endpoint_parameters: np.ndarray
    chart_restriction_parameters: np.ndarray
    triangle_occurrence_ids: np.ndarray
    vertex_occurrence_ids: np.ndarray

    @classmethod
    def empty(cls) -> BRepTessellation:
        """Return an unrealized artifact with the published host shapes and dtypes."""
        return cls(
            vertices=np.empty((0, 3), dtype=np.float64),
            triangles=np.empty((0, 3), dtype=np.int32),
            face_ids=np.empty((0,), dtype=np.int32),
            parameters=np.empty((0, 3, 2), dtype=np.float64),
            deviation_bounds=np.empty((0,), dtype=np.float64),
            normal_bounds=np.empty((0,), dtype=np.float64),
            vertex_source_dimensions=np.empty((0,), dtype=np.int32),
            vertex_source_indices=np.empty((0,), dtype=np.int64),
            vertex_parameters=np.empty((0, 2), dtype=np.float64),
            chart_restriction_vertices=np.empty((0,), dtype=np.int64),
            chart_restriction_edges=np.empty((0, 2), dtype=np.int64),
            chart_restriction_endpoint_parameters=np.empty((0, 2, 2), dtype=np.float64),
            chart_restriction_parameters=np.empty((0, 2), dtype=np.int64),
            triangle_occurrence_ids=np.empty((0,), dtype=np.int32),
            vertex_occurrence_ids=np.empty((0,), dtype=np.int32),
        )

    def select(self, rows: np.ndarray, /) -> BRepTessellation:
        """Select triangles, compact vertices, and preserve chart-root endpoints."""
        used, inverse = np.unique(self.triangles[rows], return_inverse=True)
        restriction_rows = np.flatnonzero(np.isin(self.chart_restriction_vertices, used))
        restricted_vertices = self.chart_restriction_vertices[restriction_rows]
        restricted_edges = self.chart_restriction_edges[restriction_rows]
        if restricted_edges.size and not np.all(np.isin(restricted_edges, used)):
            raise ValueError(
                "BRepTessellation.chart_restriction_edges must retain both endpoints; "
                "got an endpoint outside the selected vertices"
            )
        return BRepTessellation(
            vertices=self.vertices[used],
            triangles=inverse.reshape((-1, 3)).astype(np.int32),
            face_ids=self.face_ids[rows],
            parameters=self.parameters[rows],
            deviation_bounds=self.deviation_bounds[rows],
            normal_bounds=self.normal_bounds[rows],
            vertex_source_dimensions=self.vertex_source_dimensions[used],
            vertex_source_indices=self.vertex_source_indices[used],
            vertex_parameters=self.vertex_parameters[used],
            chart_restriction_vertices=np.searchsorted(used, restricted_vertices).astype(
                np.int64
            ),
            chart_restriction_edges=np.searchsorted(used, restricted_edges).astype(
                np.int64
            ),
            chart_restriction_endpoint_parameters=(
                self.chart_restriction_endpoint_parameters[restriction_rows]
            ),
            chart_restriction_parameters=self.chart_restriction_parameters[
                restriction_rows
            ],
            triangle_occurrence_ids=self.triangle_occurrence_ids[rows],
            vertex_occurrence_ids=self.vertex_occurrence_ids[used],
        )

    @classmethod
    def concatenate(cls, blocks: list[BRepTessellation], /) -> BRepTessellation:
        """Join disjoint blocks whose connectivity already uses global offsets."""
        return cls(
            vertices=np.concatenate([block.vertices for block in blocks]),
            triangles=np.concatenate([block.triangles for block in blocks]),
            face_ids=np.concatenate([block.face_ids for block in blocks]),
            parameters=np.concatenate([block.parameters for block in blocks]),
            deviation_bounds=np.concatenate([block.deviation_bounds for block in blocks]),
            normal_bounds=np.concatenate([block.normal_bounds for block in blocks]),
            vertex_source_dimensions=np.concatenate(
                [block.vertex_source_dimensions for block in blocks]
            ),
            vertex_source_indices=np.concatenate(
                [block.vertex_source_indices for block in blocks]
            ),
            vertex_parameters=np.concatenate(
                [block.vertex_parameters for block in blocks]
            ),
            chart_restriction_vertices=np.concatenate(
                [block.chart_restriction_vertices for block in blocks]
            ),
            chart_restriction_edges=np.concatenate(
                [block.chart_restriction_edges for block in blocks], axis=0
            ),
            chart_restriction_endpoint_parameters=np.concatenate(
                [block.chart_restriction_endpoint_parameters for block in blocks], axis=0
            ),
            chart_restriction_parameters=np.concatenate(
                [block.chart_restriction_parameters for block in blocks], axis=0
            ),
            triangle_occurrence_ids=np.concatenate(
                [block.triangle_occurrence_ids for block in blocks]
            ),
            vertex_occurrence_ids=np.concatenate(
                [block.vertex_occurrence_ids for block in blocks]
            ),
        )


@final
class BRepTessellationFailure(MeshingFailure):
    """Refuse a tessellation bound with its measured value and qualified entity.

    ``entity=None`` denotes a model-wide budget or source-cover obligation.
    Bounds use source length units for deviation and radians for normal turn.
    Native scheduler failures retain their existing ``MeshingFailure`` evidence.
    """

    @checked
    def __init__(
        self,
        category: MeshingFailureCategory,
        field: str,
        measured_bound: float,
        limit: float,
        entity: BRepEntityId | None,
        /,
    ) -> None:
        self.measured_bound = measured_bound
        self.limit = limit
        self.entity = entity
        super().__init__(
            category,
            f"BRepTessellation.{field} must be at most {limit:.17g}; "
            f"got {measured_bound:.17g}",
            stage="brep-tessellation",
            entity_ids=() if entity is None else (entity.index,),
            requested=((field, limit),),
            achieved=((field, measured_bound),),
        )


@final
class _TessellationRequest(StrictModule, NonTrainableState):
    domain: MeshingDomain
    specification: SurfaceMeshingSpec
    limits: MeshingLimits
    source_revision: str = eqx.field(static=True)


@final
class _TessellationExecution(StrictModule, NonTrainableState):
    construction: SurfaceConstruction
    cover: SourceBoundaryChartCover


@final
class _TessellationCertification(StrictModule, NonTrainableState):
    """Construction whose topology, source cover, and fidelity bounds passed."""

    construction: SurfaceConstruction


@dataclass(slots=True)
class _CoverQueries:
    """Track cover queries after construction against the same request allowance."""

    count: int
    maximum: int

    def reserve(self, count: int, /) -> None:
        if self.count + count > self.maximum:
            raise BRepTessellationFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "geometry_queries",
                float(self.count + count),
                float(self.maximum),
                None,
            )
        self.count += count


def _pose_fidelity(
    points: np.ndarray,
    world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    cells: np.ndarray,
    deviation: np.ndarray,
    normal: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose affine pose stretch and floating vertex evaluation error."""

    columns = tuple(rational_vector(rotation[:, axis]) for axis in range(3))
    gram = [[rational_dot(first, second) for second in columns] for first in columns]
    upper = max(
        gram[i][i] + sum((abs(gram[i][j]) for j in range(3) if j != i), Fraction(0))
        for i in range(3)
    )
    lower = min(
        gram[i][i] - sum((abs(gram[i][j]) for j in range(3) if j != i), Fraction(0))
        for i in range(3)
    )
    if lower <= 0:
        raise ValueError(
            "BRepOccurrence.rotation must have a positive certified Gram lower bound; "
            f"got {lower}"
        )

    def square_root_upper(value: Fraction, /) -> float:
        operand = fraction_up(value)
        result = float(np.sqrt(operand))
        return (
            float(np.nextafter(result, np.inf))
            if Fraction(result) ** 2 < value
            else result
        )

    stretch, condition = square_root_upper(upper), square_root_upper(upper / lower)
    lo = np.broadcast_to(translation, world.shape).copy()
    hi = lo.copy()
    for axis in range(3):
        product = interval_multiply(
            (points[:, axis : axis + 1], points[:, axis : axis + 1]),
            (rotation[:, axis], rotation[:, axis]),
        )
        lo, hi = interval_add((lo, hi), product)
    error = np.nextafter(np.maximum(np.abs(lo - world), np.abs(hi - world)), np.inf)
    squared = np.zeros((world.shape[0],), dtype=np.float64)
    for axis in range(3):
        squared = np.nextafter(
            squared + np.nextafter(error[:, axis] ** 2, np.inf), np.inf
        )
    vertex_error = np.nextafter(np.sqrt(squared), np.inf)
    positional = np.nextafter(
        np.nextafter(deviation * stretch, np.inf) + np.max(vertex_error[cells], axis=1),
        np.inf,
    )
    angular = np.nextafter(normal * condition, np.inf)
    return positional, angular


def _place_occurrences(
    template: BRepTessellation, geometry: BRepGeometry, maximum_triangles: int, /
) -> BRepTessellation:
    """Expand definition triangles by authored incidence without welding vertices."""
    if not geometry.occurrences:
        return template
    topology = geometry.topology()
    selections = [
        (index, topology.solid_faces[occurrence.solid])
        for index, occurrence in enumerate(geometry.occurrences)
    ]
    unshelled = tuple(
        face for face, owners in enumerate(topology.face_solids) if not owners
    )
    if unshelled:
        selections.append((-1, unshelled))
    face_counts = np.bincount(template.face_ids, minlength=topology.num_faces)
    count = sum(int(np.sum(face_counts[list(faces)])) for _, faces in selections)
    if count > maximum_triangles:
        raise BRepTessellationFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "maximum_triangles",
            float(count),
            float(maximum_triangles),
            None,
        )
    blocks: list[BRepTessellation] = []
    offset = 0
    for occurrence_index, faces in selections:
        rows = np.flatnonzero(np.isin(template.face_ids, faces))
        if not rows.size:
            continue
        block = template.select(rows)
        points, triangles, parameters = block.vertices, block.triangles, block.parameters
        deviation, normal = block.deviation_bounds, block.normal_bounds
        if occurrence_index >= 0:
            occurrence = geometry.occurrences[occurrence_index]
            local = points
            points = occurrence.place(local)
            deviation, normal = _pose_fidelity(
                local,
                points,
                np.asarray(occurrence.rotation, dtype=np.float64),
                np.asarray(occurrence.translation, dtype=np.float64),
                triangles,
                deviation,
                normal,
            )
            sides = dict(
                zip(
                    topology.solid_faces[occurrence.solid],
                    topology.solid_face_orientations[occurrence.solid],
                    strict=True,
                )
            )
            reverse = np.asarray(
                [sides[face] < 0 for face in block.face_ids], dtype=np.bool_
            )
            triangles[reverse] = triangles[reverse][:, (0, 2, 1)]
            parameters[reverse] = parameters[reverse][:, (0, 2, 1)]
        blocks.append(
            replace(
                block,
                vertices=points,
                triangles=triangles + offset,
                parameters=parameters,
                deviation_bounds=deviation,
                normal_bounds=normal,
                chart_restriction_vertices=block.chart_restriction_vertices + offset,
                chart_restriction_edges=block.chart_restriction_edges + offset,
                triangle_occurrence_ids=np.full(
                    (rows.size,), occurrence_index, dtype=np.int32
                ),
                vertex_occurrence_ids=np.full(
                    (points.shape[0],), occurrence_index, dtype=np.int32
                ),
            )
        )
        offset += points.shape[0]
    return BRepTessellation.concatenate(blocks)


def _prepare_request(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    orientation: np.ndarray,
    policy: BRepTessellationPolicy,
    /,
    *,
    source_id: str,
    source_revision: str,
) -> _TessellationRequest:
    domain = MeshingDomain.from_brep_geometry(
        geometry,
        patches,
        orientation,
        source_id=source_id,
        source_revision=source_revision,
        authority_id=geometry.geometry_id,
    )
    scope = MeshingScope(
        source_id,
        source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.source_indices[2], dtype=np.int64),
    )
    maximum = policy.maximum_triangles
    limits = MeshingLimits(
        maximum_vertices=max(1, 3 * maximum),
        maximum_edges=max(1, 3 * maximum),
        maximum_faces=maximum,
        maximum_cells=maximum,
        maximum_connectivity_entries=max(1, 12 * maximum),
        maximum_data_bytes=max(1, 256 * maximum),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        # Fidelity is a hard source obligation; density is only a soft request.
        size_controls=(
            UniformSizeControl(
                scope, domain.physical_extent, strength=SizeControlStrength.SOFT
            ),
        ),
        protected_features=(
            ProtectedFeature(
                scope, FeatureKind.SURFACE, maximum_deviation=policy.linear_deflection
            ),
        ),
        limits=limits,
    )
    return _TessellationRequest(
        domain=domain,
        specification=specification,
        limits=limits,
        source_revision=source_revision,
    )


def _retained_bytes(construction: SurfaceConstruction, /) -> int:
    """Count the retained scheduler buffers charged before source-cover work."""
    return sum(
        array.nbytes
        for array in (
            construction.vertices,
            construction.vertex_source_dimensions,
            construction.vertex_source_indices,
            construction.vertex_parameters,
            construction.chart_restriction_vertices,
            construction.chart_restriction_edges,
            construction.chart_restriction_endpoint_parameters,
            construction.chart_restriction_parameters,
            construction.triangles,
            construction.triangle_patches,
            construction.triangle_charts,
            construction.triangle_parameters,
            construction.triangle_deviations,
            construction.triangle_deviation_bounds,
            construction.triangle_normal_bounds,
            construction.curve_edges,
            construction.curve_edge_curves,
            construction.curve_edge_deviations,
            construction.curve_edge_deviation_bounds,
            construction.unresolved,
            construction.boundary_occurrences,
            *(
                array
                for (
                    _,
                    charts,
                    points,
                    cells,
                    boundary,
                    provenance,
                    restriction_required,
                    restriction_vertices,
                    restriction_edges,
                    restriction_parameters,
                ) in construction.chart_triangulations
                for array in (
                    charts,
                    points,
                    cells,
                    boundary,
                    provenance,
                    restriction_required,
                    restriction_vertices,
                    restriction_edges,
                    restriction_parameters,
                )
            ),
            *(array for _, array in construction.curve_parameters),
        )
    )


def _execute_request(
    request: _TessellationRequest, policy: BRepTessellationPolicy, /
) -> _TessellationExecution:
    # Domain compilation and surface execution consume B-Rep query owners.
    # Import these execution modules only after the unrealized return path.
    from ...meshing import (
        _domain as domain_owner,
        _surface_generation as surface_generation_owner,
    )

    domain, limits = request.domain, request.limits
    with NativeExecutionScope(limits.native_limits()) as execution_scope:
        allowance = execution_scope.remaining()
        with execution_scope.batch(
            min(limits.maximum_work_units, allowance.remaining_work_units),
            bytes_upper=min(
                limits.maximum_scratch_bytes, allowance.remaining_managed_bytes
            ),
            queries_upper=min(
                limits.maximum_geometry_queries, allowance.remaining_geometry_queries
            ),
        ) as preparation_ticket:
            owner = CoordinateEnclosureBudget(
                preparation_ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with preparation_ticket.native_target(), owner.activate():
                compiled = domain_owner.compile_surface_domain(
                    domain, request.specification, budget=owner
                )
        with owner.retained_storage(
            lambda size: execution_scope.batch(0, bytes_upper=size)
        ):
            construction = surface_generation_owner.generate_surface(
                compiled,
                NativeSurfaceSchedule(quality_angle_degrees=0.0),
                limits,
                0.0,
                maximum_normal_angle=policy.angular_deflection,
                scope=execution_scope,
                budget=owner,
            )
            source = MeshingDomainBoundarySource(
                domain,
                tuple(compiled.patches.tolist()),
                chart_triangulations=construction.chart_triangulations,
            )
            remaining_work = limits.maximum_work_units - construction.work_units
            remaining_memory = max(
                0, limits.maximum_scratch_bytes - _retained_bytes(construction)
            )
            queries = _CoverQueries(
                construction.geometry_queries, limits.maximum_geometry_queries
            )
            allowance = execution_scope.remaining()
            with owner.bind_ticket(
                execution_scope.batch(
                    min(remaining_work, allowance.remaining_work_units),
                    bytes_upper=min(remaining_memory, allowance.remaining_managed_bytes),
                    queries_upper=min(
                        max(0, limits.maximum_geometry_queries - queries.count),
                        allowance.remaining_geometry_queries,
                    ),
                )
            ):
                with (
                    owner.ticket.native_target(),
                    owner.bound_stage(remaining_work, remaining_memory),
                ):
                    cover = source.boundary_chart_cover(
                        remaining_work, budget=owner, reserve_queries=queries.reserve
                    )
    return _TessellationExecution(construction=construction, cover=cover)


def _certify_execution(
    request: _TessellationRequest,
    execution: _TessellationExecution,
    policy: BRepTessellationPolicy,
    /,
) -> _TessellationCertification:
    construction, cover = execution.construction, execution.cover
    # Preserve source-obligation order: topology, cover, surface, edge, normal.
    if np.any(construction.unresolved):
        row = int(np.flatnonzero(construction.unresolved)[0])
        raise BRepTessellationFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "unresolved",
            float(np.sum(construction.unresolved)),
            0.0,
            BRepEntityId(
                request.source_revision,
                "face",
                int(request.domain.source_indices[2][construction.triangle_patches[row]]),
            ),
        )
    if not cover.complete or cover.semantics != "certified":
        raise BRepTessellationFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "uncertified_cover",
            1.0,
            0.0,
            None,
        )
    covering_bound = float(np.max(cover.deviation_bounds, initial=0.0))
    if not np.isfinite(covering_bound) or covering_bound > policy.linear_deflection:
        raise BRepTessellationFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "cover_deviation",
            covering_bound if np.isfinite(covering_bound) else np.inf,
            policy.linear_deflection,
            None,
        )
    for bounds, indices, dimension, kind, field, limit in (
        (
            construction.triangle_deviation_bounds,
            construction.triangle_patches,
            2,
            "face",
            "triangle_deviation",
            policy.linear_deflection,
        ),
        (
            construction.curve_edge_deviation_bounds,
            construction.curve_edge_curves,
            1,
            "edge",
            "curve_deviation",
            policy.linear_deflection,
        ),
        (
            construction.triangle_normal_bounds,
            construction.triangle_patches,
            2,
            "face",
            "normal_angle",
            policy.angular_deflection,
        ),
    ):
        if np.any(bounds > limit):
            row = int(np.argmax(bounds))
            raise BRepTessellationFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                field,
                float(bounds[row]),
                limit,
                BRepEntityId(
                    request.source_revision,
                    kind,
                    int(request.domain.source_indices[dimension][indices[row]]),
                ),
            )
    return _TessellationCertification(construction=construction)


def _assemble_template(
    request: _TessellationRequest, certified: _TessellationCertification, /
) -> BRepTessellation:
    construction = certified.construction
    source_indices = construction.vertex_source_indices.copy()
    for dimension in range(3):
        selected = construction.vertex_source_dimensions == dimension
        source_indices[selected] = np.asarray(
            request.domain.source_indices[dimension], dtype=np.int64
        )[source_indices[selected]]
    return BRepTessellation(
        vertices=construction.vertices,
        triangles=construction.triangles.astype(np.int32),
        face_ids=construction.triangle_patches.astype(np.int32),
        parameters=construction.triangle_parameters,
        deviation_bounds=construction.triangle_deviation_bounds,
        normal_bounds=construction.triangle_normal_bounds,
        vertex_source_dimensions=construction.vertex_source_dimensions,
        vertex_source_indices=source_indices,
        vertex_parameters=construction.vertex_parameters,
        chart_restriction_vertices=construction.chart_restriction_vertices,
        chart_restriction_edges=construction.chart_restriction_edges,
        chart_restriction_endpoint_parameters=construction.chart_restriction_endpoint_parameters,
        chart_restriction_parameters=construction.chart_restriction_parameters,
        triangle_occurrence_ids=np.full(
            (construction.triangles.shape[0],), -1, dtype=np.int32
        ),
        vertex_occurrence_ids=np.full(
            (construction.vertices.shape[0],), -1, dtype=np.int32
        ),
    )


def _certify_placements(
    placed: BRepTessellation,
    geometry: BRepGeometry,
    policy: BRepTessellationPolicy,
    source_revision: str,
    /,
) -> BRepTessellation:
    for bounds, field, limit in (
        (placed.deviation_bounds, "placed_deviation", policy.linear_deflection),
        (placed.normal_bounds, "placed_normal_angle", policy.angular_deflection),
    ):
        if np.any(bounds > limit):
            row = int(np.argmax(bounds))
            occurrence_index = int(placed.triangle_occurrence_ids[row])
            path = (
                geometry.occurrences[occurrence_index].path
                if occurrence_index >= 0
                else ()
            )
            raise BRepTessellationFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                field,
                float(bounds[row]),
                limit,
                BRepEntityId(source_revision, "face", int(placed.face_ids[row]), path),
            )
    return placed


@checked
def tessellate_brep(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    orientation: np.ndarray,
    policy: BRepTessellationPolicy,
    /,
    *,
    source_id: str,
    source_revision: str,
) -> BRepTessellation:
    """Generate shared-edge, trim-conforming triangles with hard fidelity bounds.

    Refusals carry ``MeshingFailure`` categories and requested/achieved quantities.
    Post-generation bound refusals include the qualified entity when local.
    """
    if not patches or not policy.realize:
        return BRepTessellation.empty()
    request = _prepare_request(
        geometry,
        patches,
        orientation,
        policy,
        source_id=source_id,
        source_revision=source_revision,
    )
    execution = _execute_request(request, policy)
    certified = _certify_execution(request, execution, policy)
    template = _assemble_template(request, certified)
    placed = _place_occurrences(template, geometry, policy.maximum_triangles)
    return _certify_placements(placed, geometry, policy, source_revision)


__all__ = ["BRepTessellation", "BRepTessellationFailure", "tessellate_brep"]
