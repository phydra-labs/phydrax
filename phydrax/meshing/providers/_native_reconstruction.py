#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native conforming triangle meshing of reconstructed MeshRegion surfaces."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final

import numpy as np

import phydrax.discretization as cell_mesh_owner

from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellGeometrySpec
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...geometry._domain_boundary import MeshingDomainBoundarySource
from ...geometry._source_boundary_records import SourceBoundaryChartWitness
from ...geometry.reconstruction._boundary import (
    _sqrt_fraction_bounds,
    ReconstructedTriangleBoundarySource,
)
from ...geometry.reconstruction._core import ReconstructedGeometrySource
from ...geometry.simplicial._regions import MeshRegion
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from ...typing import checked
from .._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from .._certification_schedule import MeshCertificationSchedule
from .._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    SurfaceMeshingSpec,
)
from .._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from .._organization import MeshLabel, MeshPatch
from .._result import MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import (
    SizeCombinationPolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from ._native_options import NativeMeshingOptions


def _failure(
    category: MeshingFailureCategory,
    message: str,
    /,
    *,
    stage: str,
) -> MeshingFailure:
    return MeshingFailure(
        category,
        message,
        provider_code="native_reconstruction",
        stage=stage,
    )


def _mesh_region(source: ReconstructedGeometrySource, /) -> MeshRegion:
    region = source.source
    if not isinstance(region, MeshRegion):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The reconstructed native route requires a MeshRegion source.",
            stage="source_inspection",
        )
    return region


def _scope_ids(source: ReconstructedGeometrySource, /) -> np.ndarray:
    region = _mesh_region(source)
    identifiers = np.asarray(source.scope_entity_ids(2), dtype=np.int64)
    face_count = np.asarray(region.faces).shape[0]
    if identifiers.shape != (face_count,) or identifiers.size == 0:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "A reconstructed MeshRegion must expose one stable source identity per triangle.",
            stage="source_inspection",
        )
    return identifiers


def _scope_matches(
    scope: MeshingScope,
    source: ReconstructedGeometrySource,
    identifiers: np.ndarray,
    /,
) -> bool:
    return (
        scope.source_id == source.source_id
        and scope.source_revision == source.source_revision
        and scope.entity_kind is MeshingEntityKind.GEOMETRY
        and scope.entity_dimension == 2
        and scope.entity_set_id == source.scope_entity_set_id(2)
        and np.array_equal(np.asarray(scope.entity_ids, dtype=np.int64), identifiers)
    )


def _edge_key(first: int, last: int, /) -> tuple[int, int]:
    return (first, last) if first <= last else (last, first)


def _triangle_key(first: int, second: int, third: int, /) -> tuple[int, int, int]:
    if first > second:
        first, second = second, first
    if second > third:
        second, third = third, second
    if first > second:
        first, second = second, first
    return first, second, third


def reconstructed_surface_support_issues(
    source: ReconstructedGeometrySource,
    specification: SurfaceMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> list[str]:
    """Describe requests outside exact reconstructed-triangle surface meshing."""
    unsupported: list[str] = []
    if not isinstance(source.source, MeshRegion):
        unsupported.append("a reconstructed MeshRegion triangle surface")
        return unsupported
    target = specification.target
    family = target.cell_families
    if (
        target.topological_dimension != 2
        or target.ambient_dimension != 3
        or target.geometry_order != 1
        or set((*family.required, *family.preferred)) != {"triangle"}
        or family.allow_mixed
        or family.allowed_transitions
    ):
        unsupported.append("affine triangular cells in ambient dimension three")
    if options.route not in (None, "reconstructed_surface"):
        unsupported.append("the selected reconstructed_surface native route")
    if options.route is None and any(
        value is not None
        for value in (
            options.implicit_policy,
            options.curve_schedule,
            options.surface_schedule,
            options.volume_schedule,
            options.polyhedral_schedule,
            options.structured_schedule,
            options.hex_core_fraction,
            options.grid_schedule,
        )
    ):
        unsupported.append("route-specific schedules without an explicit native route")
    if (
        specification.size_controls
        and specification.size_combination is SizeCombinationPolicy.EXPLICIT_PRIORITY
    ):
        unsupported.append("explicit-priority size-control resolution")
    if specification.planar_embedding is not None:
        unsupported.append("planar embeddings")
    if specification.background_metric is not None:
        unsupported.append("background metrics")
    if specification.maximum_normal_deviation is not None:
        unsupported.append("maximum-normal-deviation controls")
    if specification.protected_features:
        unsupported.append("protected features")
    if specification.region_controls:
        unsupported.append("region controls")
    if specification.patch_controls:
        unsupported.append("patch controls")
    if specification.periodic_constraints:
        unsupported.append("periodic constraints")
    if specification.layer_controls:
        unsupported.append("boundary-layer controls")

    identifiers = _scope_ids(source)
    if not _scope_matches(specification.scope, source, identifiers):
        unsupported.append("the source-owned reconstructed-triangle scope")
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            unsupported.append(
                f"size-control family {type(control).__name__} outside uniform isotropic sizing"
            )
        elif not _scope_matches(control.scope, source, identifiers):
            unsupported.append("local or foreign-entity size-control scopes")
    return list(dict.fromkeys(unsupported))


def _unique_edges(faces: np.ndarray, /) -> np.ndarray:
    sides = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0)
    return np.unique(np.sort(sides, axis=1), axis=0)


def _edge_length_upper(first: np.ndarray, last: np.ndarray, /) -> float:
    squared = sum(
        (
            (Fraction.from_float(float(a)) - Fraction.from_float(float(b))) ** 2
            for a, b in zip(first, last, strict=True)
        ),
        Fraction(0),
    )
    return _sqrt_fraction_bounds(squared)[1]


def _edge_lengths(
    vertices: np.ndarray, faces: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    edges = _unique_edges(faces)
    lengths = np.empty((edges.shape[0],), dtype=np.float64)
    for row, edge in enumerate(edges.tolist()):
        first, last = int(edge[0]), int(edge[1])
        lengths[row] = _edge_length_upper(vertices[first], vertices[last])
    return edges, lengths


def _minimum_angle(vertices: np.ndarray, faces: np.ndarray, /) -> float:
    triangles = vertices[faces]
    angles: list[float] = []
    for triangle in triangles:
        lengths = (
            _edge_length_upper(triangle[0], triangle[1]),
            _edge_length_upper(triangle[1], triangle[2]),
            _edge_length_upper(triangle[2], triangle[0]),
        )
        for first, second, opposite in (
            (lengths[0], lengths[2], lengths[1]),
            (lengths[0], lengths[1], lengths[2]),
            (lengths[1], lengths[2], lengths[0]),
        ):
            cosine = (first * first + second * second - opposite * opposite) / (
                2.0 * first * second
            )
            angles.append(math.acos(min(1.0, max(-1.0, cosine))))
    return min(angles)


def _compliance(
    specification: SurfaceMeshingSpec,
    vertices: np.ndarray,
    faces: np.ndarray,
    /,
) -> MeshingComplianceReport:
    _, lengths = _edge_lengths(vertices, faces)
    if not lengths.size or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Reconstructed surface triangles must have finite positive edge lengths.",
            stage="specification_compliance",
        )
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = [("source_fidelity_bound", 0.0)]
    issues: list[str] = []
    policy = specification.size_compliance
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            continue
        key = f"size:{control.control_id}"
        requested.append((f"{key}:target_size", control.target_size))
        if control.minimum_size is not None:
            requested.append((f"{key}:minimum_size", control.minimum_size))
        if control.maximum_size is not None:
            requested.append((f"{key}:maximum_size", control.maximum_size))
        if control.maximum_growth_rate is not None:
            requested.append((f"{key}:maximum_growth_rate", control.maximum_growth_rate))
        statistics = {
            name: float(np.quantile(lengths, 0.5 if name == "p50" else 0.95))
            for name in policy.target_statistics
        }
        achieved.extend(
            (
                (f"{key}:minimum_edge", float(np.min(lengths))),
                (f"{key}:maximum_edge", float(np.max(lengths))),
                (f"{key}:size_field_growth_rate", 1.0),
                (
                    f"{key}:maximum_edge_to_target_size",
                    float(np.max(lengths) / control.target_size),
                ),
            )
        )
        achieved.extend(
            (f"{key}:{name}_edge", value) for name, value in statistics.items()
        )
        tolerance = (
            policy.absolute_tolerance + policy.relative_tolerance * control.target_size
        )
        if control.strength is SizeControlStrength.HARD:
            issues.extend(
                f"target_size_{name}:{control.control_id}"
                for name, value in statistics.items()
                if abs(value - control.target_size) > tolerance
            )
            if (
                control.minimum_size is not None
                and float(np.min(lengths)) < control.minimum_size - tolerance
            ):
                issues.append(f"minimum_size:{control.control_id}")
            if (
                control.maximum_size is not None
                and float(np.max(lengths)) > control.maximum_size + tolerance
            ):
                issues.append(f"maximum_size:{control.control_id}")
            if np.any(lengths > control.target_size + tolerance):
                issues.append(f"local_size:{control.control_id}")
            if (
                control.maximum_growth_rate is not None
                and 1.0 > control.maximum_growth_rate + tolerance
            ):
                issues.append(f"maximum_growth_rate:{control.control_id}")
    if specification.maximum_deviation is not None:
        requested.append(("maximum_deviation", specification.maximum_deviation))
        if specification.maximum_deviation < 0.0:
            issues.append("maximum_deviation")
    if specification.quality_target is not None:
        target = specification.quality_target
        angle = _minimum_angle(vertices, faces)
        requested.append(("minimum_angle", target.minimum_angle))
        achieved.append(("minimum_angle", angle))
        if target.hard and angle < target.minimum_angle:
            issues.append(f"minimum_angle:{target.target_id}")
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _can_refine(
    specification: SurfaceMeshingSpec,
    vertices: np.ndarray,
    faces: np.ndarray,
    /,
) -> bool:
    hard_controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
        and control.strength is SizeControlStrength.HARD
    )
    if not hard_controls or (
        specification.quality_target is not None and specification.quality_target.hard
    ):
        return False
    _, lengths = _edge_lengths(vertices, faces)
    policy = specification.size_compliance
    can_improve = False
    for control in hard_controls:
        tolerance = (
            policy.absolute_tolerance + policy.relative_tolerance * control.target_size
        )
        if (
            control.minimum_size is not None
            and float(np.min(lengths)) < control.minimum_size - tolerance
        ):
            return False
        if (
            control.maximum_growth_rate is not None
            and 1.0 > control.maximum_growth_rate + tolerance
        ):
            return False
        statistics = (
            float(np.quantile(lengths, 0.5))
            if "p50" in policy.target_statistics
            else None,
            float(np.quantile(lengths, 0.95))
            if "p95" in policy.target_statistics
            else None,
        )
        if any(
            value is not None and value < control.target_size - tolerance
            for value in statistics
        ):
            return False
        can_improve = can_improve or any(
            value is not None and value > control.target_size + tolerance
            for value in statistics
        )
        can_improve = can_improve or bool(
            np.any(lengths > control.target_size + tolerance)
        )
        if control.maximum_size is not None:
            can_improve = can_improve or bool(
                np.max(lengths) > control.maximum_size + tolerance
            )
    return can_improve


class _MidpointRepresentabilityError(ValueError):
    pass


def _red_refine(
    vertices: np.ndarray,
    faces: np.ndarray,
    parents: np.ndarray,
    source_parameters: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = _unique_edges(faces)
    if vertices.shape[0] + edges.shape[0] - 1 > np.iinfo(np.int32).max:
        raise _MidpointRepresentabilityError(
            "The refined vertex indices exceed the affine triangle connectivity representation."
        )
    midpoint_indices: dict[tuple[int, int], int] = {}
    midpoint_coordinates: list[tuple[float, float, float]] = []
    for edge in edges.tolist():
        first, last = int(edge[0]), int(edge[1])
        coordinates: list[float] = []
        for a, b in zip(vertices[first], vertices[last], strict=True):
            exact = (Fraction.from_float(float(a)) + Fraction.from_float(float(b))) / 2
            midpoint = float(exact)
            if Fraction.from_float(midpoint) != exact:
                raise _MidpointRepresentabilityError(
                    "A source-edge midpoint is not exactly representable in binary64."
                )
            coordinates.append(midpoint)
        midpoint = (coordinates[0], coordinates[1], coordinates[2])
        if midpoint in (tuple(vertices[first]), tuple(vertices[last])):
            raise _MidpointRepresentabilityError(
                "A source-edge midpoint coincides with an endpoint in binary64."
            )
        midpoint_indices[_edge_key(first, last)] = vertices.shape[0] + len(
            midpoint_coordinates
        )
        midpoint_coordinates.append(midpoint)
    refined_vertices = np.concatenate(
        (vertices, np.asarray(midpoint_coordinates, dtype=np.float64)), axis=0
    )
    children: list[tuple[int, int, int]] = []
    child_parents: list[int] = []
    for face, parent in zip(faces.tolist(), parents.tolist(), strict=True):
        first, second, third = int(face[0]), int(face[1]), int(face[2])
        first_second = midpoint_indices[_edge_key(first, second)]
        second_third = midpoint_indices[_edge_key(second, third)]
        third_first = midpoint_indices[_edge_key(third, first)]
        children.extend(
            (
                (first, first_second, third_first),
                (first_second, second, second_third),
                (third_first, second_third, third),
                (first_second, second_third, third_first),
            )
        )
        child_parents.extend((int(parent),) * 4)
    first_second_parameters = 0.5 * (source_parameters[:, 0] + source_parameters[:, 1])
    second_third_parameters = 0.5 * (source_parameters[:, 1] + source_parameters[:, 2])
    third_first_parameters = 0.5 * (source_parameters[:, 2] + source_parameters[:, 0])
    child_parameters = np.stack(
        (
            np.stack(
                (
                    source_parameters[:, 0],
                    first_second_parameters,
                    third_first_parameters,
                ),
                axis=1,
            ),
            np.stack(
                (
                    first_second_parameters,
                    source_parameters[:, 1],
                    second_third_parameters,
                ),
                axis=1,
            ),
            np.stack(
                (
                    third_first_parameters,
                    second_third_parameters,
                    source_parameters[:, 2],
                ),
                axis=1,
            ),
            np.stack(
                (
                    first_second_parameters,
                    second_third_parameters,
                    third_first_parameters,
                ),
                axis=1,
            ),
        ),
        axis=1,
    ).reshape((-1, 3, 2))
    return (
        refined_vertices,
        np.asarray(children, dtype=np.int32),
        np.asarray(child_parents, dtype=np.int64),
        child_parameters,
    )


def _refine_to_size(
    region: MeshRegion,
    specification: SurfaceMeshingSpec,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MeshingComplianceReport]:
    vertices = np.asarray(region.vertices, dtype=np.float64)
    faces = np.asarray(region.faces, dtype=np.int64)
    parents = np.arange(faces.shape[0], dtype=np.int64)
    source_chart = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    source_parameters = np.broadcast_to(source_chart, (faces.shape[0], 3, 2)).copy()
    while True:
        compliance = _compliance(specification, vertices, faces)
        if compliance.passed:
            return vertices, faces, parents, source_parameters, compliance
        if not _can_refine(specification, vertices, faces):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The exact reconstructed triangle surface cannot meet the requested physical controls by conforming red-refinement: "
                + "; ".join(compliance.issues),
                stage="specification_compliance",
            )
        try:
            vertices, faces, parents, source_parameters = _red_refine(
                vertices, faces, parents, source_parameters
            )
        except _MidpointRepresentabilityError as error:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Further conforming refinement would change or lose the exact reconstructed surface: {error}",
                stage="specification_compliance",
            ) from error


def _mesh_scope(
    mesh: cell_mesh_owner.CellMesh, dimension: int, identifiers: np.ndarray, /
) -> MeshingScope:
    entity_set = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entity_set.entity_set_id,
        identifiers,
    )


def _triangle_chart_orientation(points: np.ndarray, /) -> int:
    exact = tuple(
        tuple(Fraction.from_float(float(value)) for value in point) for point in points
    )
    area = (exact[1][0] - exact[0][0]) * (exact[2][1] - exact[0][1]) - (
        exact[1][1] - exact[0][1]
    ) * (exact[2][0] - exact[0][0])
    if area == 0:
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "A reconstructed source-chart witness contains a degenerate triangle.",
            stage="source_chart_witness",
        )
    return 1 if area > 0 else -1


def _source_triangle_chart_witness(
    source: ReconstructedGeometrySource,
    region: MeshRegion,
    domain_query: MeshingDomainBoundarySource,
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    faces: np.ndarray,
    parents: np.ndarray,
    source_parameters: np.ndarray,
    /,
) -> SourceBoundaryChartWitness:
    domain_query.require_current()
    domain = domain_query.domain
    source_faces = np.asarray(region.faces, dtype=np.int64)
    if (
        domain.source_id != source.source_id
        or domain.source_revision != source.source_revision
        or len(domain.patches) != source_faces.shape[0]
        or not np.array_equal(
            np.asarray(domain.source_indices[2], dtype=np.int64),
            np.asarray(source.persistent_facet_ids, dtype=np.int64),
        )
        or not np.array_equal(
            np.asarray(domain.corner_points),
            np.asarray(region.vertices),
        )
        or faces.ndim != 2
        or faces.shape[1] != 3
        or parents.shape != (faces.shape[0],)
        or source_parameters.shape != (faces.shape[0], 3, 2)
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The reconstructed chart query must retain the current persistent triangle domain.",
            stage="source_chart_witness",
        )
    block = mesh.blocks[0]
    target_faces = np.asarray(block.vertices, dtype=np.int64)
    target_cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    if (
        target_faces.shape != faces.shape
        or target_cell_ids.shape != (faces.shape[0],)
        or np.unique(target_cell_ids).size != target_cell_ids.size
    ):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Output triangle identities do not align with reconstructed source ancestry.",
            stage="source_chart_witness",
        )

    source_triangles: dict[
        tuple[int, int, int], tuple[tuple[int, int, int], int, np.ndarray]
    ] = {}
    for face, parent, parameters in zip(
        faces.tolist(), parents.tolist(), source_parameters, strict=True
    ):
        face_vertices = (int(face[0]), int(face[1]), int(face[2]))
        parent_row = int(parent)
        key = _triangle_key(*face_vertices)
        if (
            len(set(face_vertices)) != 3
            or parent_row < 0
            or parent_row >= source_faces.shape[0]
            or key in source_triangles
        ):
            raise _failure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Refined triangles must retain one unique, valid source-face ancestry.",
                stage="source_chart_witness",
            )
        source_triangles[key] = (face_vertices, parent_row, parameters)

    target_reference_triangle = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64
    )
    target_reference_vertices = np.broadcast_to(
        target_reference_triangle, (target_faces.shape[0], 3, 2)
    )
    mapped_source_parameters = np.empty((target_faces.shape[0], 3, 2), dtype=np.float64)
    chart_ids: list[str] = []
    entity_ids: list[str] = []
    occurrence_paths: list[tuple[str, ...]] = []
    orientations = np.empty((target_faces.shape[0],), dtype=np.int8)
    for row, target_face in enumerate(target_faces.tolist()):
        target_vertices = (
            int(target_face[0]),
            int(target_face[1]),
            int(target_face[2]),
        )
        ancestry = source_triangles.get(_triangle_key(*target_vertices))
        if ancestry is None:
            raise _failure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "A target triangle has no exact reconstructed source-face ancestry.",
                stage="source_chart_witness",
            )
        source_vertices, parent_row, source_uv = ancestry
        source_uv_by_vertex = {
            vertex: source_uv[index] for index, vertex in enumerate(source_vertices)
        }
        source_parameters_for_target = np.asarray(
            [source_uv_by_vertex[vertex] for vertex in target_vertices],
            dtype=np.float64,
        )
        mapped_source_parameters[row] = source_parameters_for_target
        scope_id = source.scope_entity_id_for_mesh_row(2, parent_row)
        entity_id, occurrence = domain.source_entity_key_for_scope(2, scope_id)
        if (
            source.source_entity_key_for_scope(2, scope_id) != (entity_id, occurrence)
            or domain.resolve_entity_id(2, entity_id, occurrence_path=occurrence)
            != parent_row
        ):
            raise _failure(
                MeshingFailureCategory.INVALID_SOURCE,
                "A reconstructed source-chart key does not resolve through its persistent facet identity.",
                stage="source_chart_witness",
            )
        chart_ids.append(entity_id)
        entity_ids.append(entity_id)
        occurrence_paths.append(occurrence)
        orientations[row] = _triangle_chart_orientation(
            source_parameters_for_target
        ) * _triangle_chart_orientation(target_reference_vertices[row])

    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=domain_query.source_scope_id,
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=target_cell_ids,
        entity_global_ids=target_cell_ids,
        local_entity_indices=np.full(target_cell_ids.shape, -1, dtype=np.int32),
        target_reference_vertices=target_reference_vertices,
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(entity_ids),
        occurrence_paths=tuple(occurrence_paths),
        source_parameters=mapped_source_parameters,
        orientations=orientations,
    )


def _organization(
    source: ReconstructedGeometrySource,
    region: MeshRegion,
    query: ReconstructedTriangleBoundarySource,
    coordinate_contract: SpatialCoordinateContract,
    mesh: cell_mesh_owner.CellMesh,
    faces: np.ndarray,
    parents: np.ndarray,
    /,
) -> NativeMeshOrganization:
    cell_set = mesh.entity_set(2)
    edge_set = mesh.entity_set(1)
    vertex_set = mesh.entity_set(0)
    block = mesh.blocks[0]
    cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    block_faces = np.asarray(block.vertices, dtype=np.int64)
    parent_by_triangle = {
        _triangle_key(int(triangle[0]), int(triangle[1]), int(triangle[2])): int(parent)
        for triangle, parent in zip(faces.tolist(), parents.tolist(), strict=True)
    }
    try:
        cell_parents = np.asarray(
            [
                parent_by_triangle[
                    _triangle_key(int(face[0]), int(face[1]), int(face[2]))
                ]
                for face in block_faces.tolist()
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "CellMesh triangle rows do not preserve the refined source ancestry.",
            stage="geometry_association",
        ) from error
    if cell_ids.shape != cell_parents.shape:
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "CellMesh triangle IDs and refined source ancestry do not align.",
            stage="geometry_association",
        )
    face_keys = tuple(
        source.source_entity_key_for_mesh_row(2, row)
        for row in range(source.persistent_facet_ids.shape[0])
    )
    cell_source_keys = tuple(face_keys[int(parent)] for parent in cell_parents)
    region_ids = np.asarray(source.scope_entity_ids(3), dtype=np.int64)
    if region_ids.shape != (1,):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "A reconstructed MeshRegion must retain one source region identity.",
            stage="geometry_association",
        )
    region_side = source.source_entity_key_for_scope(3, int(region_ids[0]))[0]
    cell_association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        cell_set.entity_set_id,
        cell_ids,
        tuple(key[0] for key in cell_source_keys),
        np.zeros((cell_ids.size,), dtype=np.float64),
        exact=True,
        source_roles=(GeometrySourceEntityRole.FACET,) * cell_ids.size,
        source_occurrence_paths=tuple(key[1] for key in cell_source_keys),
        source_region_sides=((region_side, None),) * cell_ids.size,
    )

    connectivity = mesh.connectivity
    if not isinstance(connectivity, cell_mesh_owner.PolygonalConnectivity) or (
        connectivity.triangle_count != cell_ids.size
        or connectivity.quadrilateral_count
        or connectivity.polygon_count
    ):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "The reconstructed surface mesh must retain canonical triangular incidence.",
            stage="geometry_association",
        )
    cell_vertex_rows = np.asarray(connectivity.cell_vertices, dtype=np.int64)
    cell_vertex_valid = np.asarray(connectivity.cell_vertex_valid, dtype=np.bool_)
    cell_edge_routes = np.asarray(connectivity.cell_edges, dtype=np.int64)
    cell_edge_valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
    if (
        cell_vertex_rows.ndim != 2
        or cell_vertex_rows.shape[0] != cell_ids.size
        or cell_vertex_valid.shape != cell_vertex_rows.shape
        or not np.all(cell_vertex_valid[:, :3])
        or np.any(cell_vertex_valid[:, 3:])
        or cell_edge_routes.shape != cell_vertex_rows.shape
        or cell_edge_valid.shape != cell_edge_routes.shape
        or not np.all(cell_edge_valid[:, :3])
        or np.any(cell_edge_valid[:, 3:])
        or np.asarray(connectivity.edges).shape != (edge_ids.size, 2)
    ):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Canonical triangle incidence does not align with reconstructed mesh entities.",
            stage="geometry_association",
        )
    connectivity_cells = cell_vertex_rows[:, :3]
    try:
        connectivity_parents = np.asarray(
            [
                parent_by_triangle[
                    _triangle_key(int(face[0]), int(face[1]), int(face[2]))
                ]
                for face in connectivity_cells.tolist()
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Canonical triangle cells do not preserve the refined source ancestry.",
            stage="geometry_association",
        ) from error

    vertex_parents = np.full((vertex_ids.shape[0],), -1, dtype=np.int64)
    edge_parents = np.full((edge_ids.shape[0],), -1, dtype=np.int64)
    for parent_value, edge_route, edge_valid, vertex_route, vertex_valid in zip(
        connectivity_parents.tolist(),
        cell_edge_routes.tolist(),
        cell_edge_valid.tolist(),
        cell_vertex_rows.tolist(),
        cell_vertex_valid.tolist(),
        strict=True,
    ):
        parent = int(parent_value)
        for edge_value, active in zip(edge_route, edge_valid, strict=True):
            if active:
                edge_index = int(edge_value)
                if edge_parents[edge_index] < 0:
                    edge_parents[edge_index] = parent
        for vertex_value, active in zip(vertex_route, vertex_valid, strict=True):
            if active:
                vertex_index = int(vertex_value)
                if vertex_parents[vertex_index] < 0:
                    vertex_parents[vertex_index] = parent
    if np.any(vertex_parents < 0) or np.any(edge_parents < 0):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Every output vertex and edge must retain an incident "
            "source-triangle association.",
            stage="geometry_association",
        )
    vertex_source_keys = tuple(
        face_keys[int(parent)] for parent in vertex_parents.tolist()
    )
    vertex_association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        vertex_set.entity_set_id,
        vertex_ids,
        tuple(key[0] for key in vertex_source_keys),
        np.zeros((vertex_ids.size,), dtype=np.float64),
        exact=True,
        source_roles=(GeometrySourceEntityRole.FACET,) * vertex_ids.size,
        source_occurrence_paths=tuple(key[1] for key in vertex_source_keys),
        source_region_sides=((region_side, None),) * vertex_ids.size,
    )

    edge_source_keys = tuple(face_keys[int(parent)] for parent in edge_parents.tolist())
    edge_association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        edge_set.entity_set_id,
        edge_ids,
        tuple(key[0] for key in edge_source_keys),
        np.zeros((edge_ids.size,), dtype=np.float64),
        exact=True,
        source_roles=(GeometrySourceEntityRole.FACET,) * edge_ids.size,
        source_occurrence_paths=tuple(key[1] for key in edge_source_keys),
        source_region_sides=((region_side, None),) * edge_ids.size,
    )

    cell_scope = _mesh_scope(mesh, 2, cell_ids)
    connected = region.triangle_mesh.topology.num_face_components == 1
    boundary = SurfaceModel(
        mesh,
        SurfaceMetadata(
            source_id=source.source_id,
            source_revision=source.source_revision,
            coordinate_contract=coordinate_contract,
            provenance=(
                "native-reconstructed-triangle-surface",
                source.binding_id,
                query.source_revision,
                source.report.algorithm,
                source.report.repair_interpretation
                or "unspecified-reconstruction-interpretation",
            ),
            cell_tags=("reconstructed-surface",) * cell_ids.size,
        ),
    )
    return NativeMeshOrganization(
        boundary=boundary,
        patches=(MeshPatch("reconstructed-surface", cell_scope, connected=connected),),
        labels=(MeshLabel("reconstructed-surface", cell_scope),),
        associations=(cell_association, edge_association, vertex_association),
    )


@final
class _PreparedReconstructionRoute(StrictModule, NonTrainableState):
    source: ReconstructedGeometrySource
    region: MeshRegion
    specification: SurfaceMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    query: ReconstructedTriangleBoundarySource
    domain_query: MeshingDomainBoundarySource

    def __init__(
        self,
        source: ReconstructedGeometrySource,
        region: MeshRegion,
        specification: SurfaceMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        query: ReconstructedTriangleBoundarySource,
        domain_query: MeshingDomainBoundarySource,
        /,
    ) -> None:
        self.source = source
        self.region = region
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.query = query
        self.domain_query = domain_query

    def execute(self, /) -> NativeMeshConstruction:
        source, region, specification = self.source, self.region, self.specification
        vertices, faces, parents, source_parameters, compliance = _refine_to_size(
            region, specification
        )
        mesh = cell_mesh_owner.CellMesh.from_triangles(
            vertices,
            faces.astype(np.int32, copy=False),
            numeric_version=source.source_revision,
        )
        geometry = CellGeometrySpec.affine(mesh)
        fidelity_witness = _source_triangle_chart_witness(
            source,
            region,
            self.domain_query,
            mesh,
            geometry,
            faces,
            parents,
            source_parameters,
        )
        organization = _organization(
            source,
            region,
            self.query,
            self.coordinate_contract,
            mesh,
            faces,
            parents,
        )
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("surface"),
            source.source_id,
            source.source_revision,
            fidelity_source=self.query,
            fidelity_tolerance=0.0,
            protected_features=(),
            fidelity_witness=fidelity_witness,
        )
        return NativeMeshConstruction(
            mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            organization=organization,
        )


@checked
def prepare_native_reconstruction(
    source: ReconstructedGeometrySource,
    specification: SurfaceMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedReconstructionRoute:
    """Prepare exact reuse or conforming refinement of reconstructed facets."""
    unsupported = reconstructed_surface_support_issues(source, specification, options)
    if unsupported:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "; ".join(unsupported),
            stage="source_inspection",
        )
    region = _mesh_region(source)
    report = source.report
    source_vertices = np.asarray(region.vertices)
    source_faces = np.asarray(region.faces)
    if (
        report.output_vertices != source_vertices.shape[0]
        or report.output_cells != source_faces.shape[0]
        or not report.watertight
        or not report.winding_consistent
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The reconstruction report must describe the current watertight MeshRegion triangle realization.",
            stage="source_inspection",
        )
    domain = source.meshing_domain
    if (
        domain is None
        or domain.source_id != source.source_id
        or domain.source_revision != source.source_revision
        or len(domain.patches) != source_faces.shape[0]
        or not np.array_equal(
            np.asarray(domain.source_indices[2], dtype=np.int64),
            np.asarray(source.persistent_facet_ids, dtype=np.int64),
        )
        or not np.array_equal(np.asarray(domain.corner_points), source_vertices)
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "A reconstructed MeshRegion requires its current persistent MeshingDomain.",
            stage="source_inspection",
        )
    query = ReconstructedTriangleBoundarySource(source)
    domain_query = MeshingDomainBoundarySource(domain, tuple(range(len(domain.patches))))
    return _PreparedReconstructionRoute(
        source,
        region,
        specification,
        coordinate_contract,
        query,
        domain_query,
    )


__all__ = [
    "prepare_native_reconstruction",
    "reconstructed_surface_support_issues",
]
