#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Canonical native PLC-restricted polyhedral publication.

Construction ancestry and numerical optimization diagnostics accompany the
existing CellMeshingResult. Independent affine validity, global embedding and
material-domain coverage remain mandatory; unresolved checks never publish.
VEM/FV are explicit downstream consumers, not hidden allocations in generation.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from time import monotonic

import equinox as eqx
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    PlcRecoveryFailure,
    RestrictedPowerFailure,
)
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_enclosure import CoordinateEnclosureResourceError
from ...geometry._triangulation import (
    PeriodicPowerImageCapacityRefusal,
    PeriodicPowerSourceRefusal,
)
from .._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from .._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from .._certification import MeshCertificationSchedule
from .._contracts import (
    MeshingDerivativeMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingProviderInfo,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._measurements import (
    measure_phase,
    NativeMeshingPhase,
    NativeMeshingPhaseMeasurement,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from .._organization import MeshLabel, MeshPatch, MeshZone, MeshZoneRole
from .._polyhedral_generation import (
    _polyhedral_connectivity,
    _polyhedral_periodic_group,
    _sqrt_upper,
    generate_polyhedral_volume,
    NativePolyhedralSchedule,
    PolyhedralConstruction,
    PolyhedralGenerationError,
)
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import UniformSizeControl
from .._trace import (
    MeshingDiagnostic,
    MeshingDiagnosticSeverity,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
)
from .._volume_generation import (
    _entity,
    _recovery_failure,
    _seeds,
    native_volume_checkpoint,
    native_volume_execution_budget,
)
from ._native_publication import (
    check_deadline,
    edge_size_evidence,
    NativeCertificationRequest,
    publish_native_result,
    uniform_size_compliance,
)
from ._native_sources import NativePlcSource, NativePolyhedralSource
from ._native_volume import volume_support_issues


def polyhedral_support_issues(
    source: NativePlcSource | NativePolyhedralSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    issues = [
        issue
        for issue in volume_support_issues(source, specification)
        if issue
        not in (
            "a tetrahedral volume target",
            "affine tetrahedra in ambient dimension three",
            "the simplex fill strategy",
            "periodic constraints",
        )
    ]
    target = specification.target
    if set((*target.cell_families.required, *target.cell_families.preferred)) != {
        "polyhedron"
    }:
        issues.append("a polyhedral volume target")
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
    ):
        issues.append("affine planar-face polyhedra in ambient dimension three")
    if specification.fill_strategy is not VolumeFillStrategy.POLYHEDRAL:
        issues.append("the polyhedral fill strategy")
    for constraint in specification.periodic_constraints:
        for scope in (constraint.source_scope, constraint.target_scope):
            identifiers = np.asarray(scope.entity_ids)
            if (
                scope.entity_dimension != 2
                or scope.entity_kind is not MeshingEntityKind.GEOMETRY
                or scope.source_id != source.source_id
                or scope.source_revision != source.source_revision
                or np.any(identifiers < 0)
                or np.any(identifiers >= source.complex.facet_count)
            ):
                issues.append("periodic scopes bound to original PLC facets")
    try:
        _polyhedral_periodic_group(tuple(specification.periodic_constraints))
    except (TypeError, ValueError) as error:
        issues.append(f"proper commuting periodic source actions: {error}")
    return issues


class PreparedPolyhedralVolume(StrictModule, NonTrainableState):
    """Bound source, specification, sites and the numerical refinement policy."""

    schedule: NativePolyhedralSchedule = eqx.field(static=True)
    sites: np.ndarray | None
    weights: np.ndarray | None
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: NativePlcSource | NativePolyhedralSource,
        specification: VolumeMeshingSpec,
        schedule: NativePolyhedralSchedule | None = None,
        /,
    ) -> None:
        if not isinstance(
            source, (NativePlcSource, NativePolyhedralSource)
        ) or not isinstance(specification, VolumeMeshingSpec):
            raise TypeError(
                "Native polyhedra require a PLC/polyhedral source and VolumeMeshingSpec."
            )
        issues = polyhedral_support_issues(source, specification)
        if issues:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "; ".join(issues),
                stage="preparation",
            )
        policy = NativePolyhedralSchedule() if schedule is None else schedule
        if not isinstance(policy, NativePolyhedralSchedule):
            raise TypeError("schedule must be NativePolyhedralSchedule.")
        limits = specification.limits
        control = specification.size_controls[0]
        if not isinstance(control, UniformSizeControl):
            raise TypeError(
                "Native polyhedral generation requires a uniform whole-domain size control."
            )
        maximum = (
            control.maximum_size
            if control.maximum_size is not None
            else control.target_size
        )
        # A hard upper edge-size request is enforced by the conservative whole
        # cell diameter bound. No request is weakened to site spacing.
        policy = replace(
            policy,
            maximum_vertices=min(policy.maximum_vertices, limits.maximum_vertices),
            maximum_cells=min(policy.maximum_cells, limits.maximum_cells),
            maximum_work_units=min(policy.maximum_work_units, limits.maximum_work_units),
            maximum_scratch_bytes=min(
                policy.maximum_scratch_bytes, limits.maximum_scratch_bytes
            ),
            maximum_cell_diameter=maximum
            if policy.maximum_cell_diameter is None
            else min(maximum, policy.maximum_cell_diameter),
        )
        self.schedule = policy
        self.sites = source.sites if isinstance(source, NativePolyhedralSource) else None
        self.weights = (
            source.weights if isinstance(source, NativePolyhedralSource) else None
        )
        self.source_binding_id = source.binding_id
        self.specification_id = specification.specification_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "native-polyhedral-prepared",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": asdict(policy),
                "sites": array_tree_fingerprint((self.sites, self.weights)),
            }
        )


def _protected_vertices(specification: VolumeMeshingSpec) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(vertex)
                for feature in specification.protected_features
                if feature.scope.entity_dimension == 0
                for vertex in np.asarray(feature.scope.entity_ids)
            }
        )
    )


def _vertex_association(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec | None,
    construction: PolyhedralConstruction,
    selected_faces: np.ndarray,
    face_residuals: np.ndarray,
) -> GeometryAssociation:
    from fractions import Fraction

    mesh, c = construction.mesh, _polyhedral_connectivity(construction.mesh)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    carriers = construction.diagram.construction.vertex_carriers
    original_count = source.complex.vertices.shape[0]
    dimensions = np.full(points.shape[0], 3, dtype=np.int64)
    sources = np.full(points.shape[0], -1, dtype=np.int64)
    residuals = np.zeros(points.shape[0])
    ambiguous = np.zeros(points.shape[0], dtype=bool)
    original_rows: dict[int, int] = {}
    for vertex, carrier in enumerate(carriers):
        carrier = np.asarray(carrier, dtype=np.int32)
        indices = carrier[carrier >= 0]
        if indices.size == 1 and indices[0] < original_count:
            index = int(indices[0])
            dimensions[vertex], sources[vertex] = 0, index
            original_rows[index] = vertex
            squared = sum(
                (
                    (Fraction(float(a)) - Fraction(float(b))) ** 2
                    for a, b in zip(
                        points[vertex], source.complex.vertices[index], strict=True
                    )
                ),
                Fraction(0),
            )
            residuals[vertex] = _sqrt_upper(squared)

    def assign(
        rows: np.ndarray, dimension: int, source_index: int, residual: float
    ) -> None:
        same = (dimensions[rows] == dimension) & (sources[rows] >= 0)
        ambiguous[rows[same & (sources[rows] != source_index)]] = True
        matches = rows[same & (sources[rows] == source_index)]
        residuals[matches] = np.maximum(residuals[matches], residual)
        replace = rows[(dimensions[rows] > dimension) | (sources[rows] < 0)]
        dimensions[replace], sources[replace] = dimension, source_index
        residuals[replace], ambiguous[replace] = residual, False

    offsets, vertices = (
        np.asarray(c.cell_vertex_offsets),
        np.asarray(c.cell_vertex_values),
    )
    for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        assign(vertices[a:b], 3, int(construction.cell_regions[cell]), 0.0)
    offsets, vertices = (
        np.asarray(c.face_vertex_offsets),
        np.asarray(c.face_vertex_values),
    )
    for face, residual in zip(selected_faces, face_residuals, strict=True):
        a, b = offsets[face : face + 2]
        assign(vertices[a:b], 2, int(construction.face_facets[face]), float(residual))
    edges = np.asarray(c.edges, dtype=np.int64)
    for edge in np.flatnonzero(construction.feature_edge_sources >= 0):
        source_edge = int(construction.feature_edge_sources[edge])
        assign(
            edges[edge],
            1,
            source_edge,
            float(construction.feature_maximum_distance[source_edge]),
        )
    missing = tuple(
        vertex
        for vertex in (
            () if specification is None else _protected_vertices(specification)
        )
        if vertex not in original_rows
    )
    if missing:
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "Protected PLC vertices disappeared from the restricted carrier.",
            stage="geometry_association",
            entity_ids=missing,
        )
    for feature in () if specification is None else specification.protected_features:
        if feature.scope.entity_dimension == 0:
            bad = tuple(
                int(vertex)
                for vertex in np.asarray(feature.scope.entity_ids)
                if residuals[original_rows[int(vertex)]] > feature.maximum_deviation
            )
            if bad:
                raise MeshingFailure(
                    MeshingFailureCategory.AUDIT_FAILED,
                    "Protected PLC point displacement exceeds its request.",
                    stage="geometry_association",
                    entity_ids=bad,
                )
    unresolved = ambiguous | (sources < 0)
    if np.any(unresolved):
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "Restricted vertex source ancestry is ambiguous or missing.",
            stage="geometry_association",
            entity_ids=tuple(map(int, np.asarray(c.vertex_global_ids)[unresolved])),
        )
    # These dimensions were assigned from the validated volume PLC carriers:
    # source points, curve edges, surface facets, and material regions.
    roles = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.FACET,
        GeometrySourceEntityRole.REGION,
    )
    return GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        mesh.entity_set(0).entity_set_id,
        c.vertex_global_ids,
        tuple(
            _entity(source.source_revision, roles[int(dimension)].value, int(index))
            for dimension, index in zip(dimensions, sources, strict=True)
        ),
        residuals,
        exact=False,
        source_dimensions=dimensions,
        source_indices=sources,
        source_entity_roles=tuple(roles[int(dimension)] for dimension in dimensions),
    )


def _organization(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec | None,
    construction: PolyhedralConstruction,
    tolerance: float,
) -> tuple[
    tuple[MeshZone, ...],
    tuple[MeshPatch, ...],
    tuple[MeshLabel, ...],
    tuple[GeometryAssociation, ...],
]:
    mesh, complex_ = construction.mesh, source.complex
    connectivity = _polyhedral_connectivity(mesh)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    edge_ids = np.asarray(connectivity.edge_global_ids, dtype=np.int64)

    def scope(dimension: int, rows: np.ndarray) -> MeshingScope:
        entity_set = mesh.entity_set(dimension)
        return MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            dimension,
            entity_set.entity_set_id,
            np.sort(rows),
        )

    seeds = (
        {}
        if specification is None
        else {seed.region_name: seed for seed in specification.region_seeds}
    )
    controls = (
        {}
        if specification is None
        else {control.region_name: control for control in specification.region_controls}
    )
    zones = tuple(
        MeshZone(
            name,
            MeshZoneRole.REGION,
            scope(3, cell_ids[construction.cell_regions == region]),
            material_id=(
                controls[name].material_id
                if name in controls
                else seeds[name].material_id
                if name in seeds
                else None
            ),
            region_role=(
                controls[name].role
                if name in controls
                else seeds[name].role
                if name in seeds
                else None
            ),
        )
        for region, name in enumerate(complex_.region_ids)
    )
    facets = construction.face_facets
    selected = np.flatnonzero(facets >= 0)
    patches = tuple(
        MeshPatch(f"facet:{facet}", scope(2, face_ids[facets == facet]))
        for facet in np.unique(facets[selected])
    )
    zone_ids = {zone.name: zone.zone_id for zone in zones}
    for control in () if specification is None else specification.patch_controls:
        rows = face_ids[np.isin(facets, np.asarray(control.scope.entity_ids))]
        if control.required and not rows.size:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A required source interface/patch has no published face fragments.",
                stage="geometry_association",
                entity_ids=tuple(map(int, np.asarray(control.scope.entity_ids))),
            )
        if rows.size:
            patches += (
                MeshPatch(
                    control.name,
                    scope(2, rows),
                    adjacent_zone_ids=tuple(
                        zone_ids[name] for name in control.adjacent_region_names
                    ),
                ),
            )
    labels = tuple(
        MeshLabel(name, scope(dimension, rows))
        for name, dimension, rows in (
            ("interface", 2, face_ids[construction.interface_faces]),
            ("internal_sheet", 2, face_ids[construction.sheet_faces]),
            (
                "internal_curve",
                1,
                edge_ids[
                    (construction.feature_edge_sources >= 0)
                    & (construction.feature_edge_sources < complex_.segments.shape[0])
                ],
            ),
        )
        if rows.size
    )
    if construction.seam_face_pairs is not None:
        paired = np.unique(construction.seam_face_pairs[:, :2])
        labels += (MeshLabel("periodic", scope(2, face_ids[paired])),)
    offsets, vertices = (
        np.asarray(connectivity.face_vertex_offsets),
        np.asarray(connectivity.face_vertex_values),
    )
    edge_rows = {
        tuple(sorted(map(int, edge))): row
        for row, edge in enumerate(np.asarray(connectivity.edges))
    }
    edge_facets = np.full(edge_ids.size, -1, dtype=np.int64)
    edge_residuals = np.zeros(edge_ids.size)
    face_residuals = np.zeros(selected.size)
    orientations = np.zeros(selected.size, dtype=np.int8)
    for row, face in enumerate(selected):
        facet = int(facets[face])
        triangle = int(construction.face_recovery_triangles[face])
        source_points = construction.recovery.points[
            construction.recovery.faces[triangle]
        ]
        normal = np.cross(
            source_points[1] - source_points[0], source_points[2] - source_points[0]
        )
        length = np.linalg.norm(normal)
        loop = vertices[offsets[face] : offsets[face + 1]]
        face_residuals[row] = float(
            np.max(np.abs((points[loop] - source_points[0]) @ normal)) / length
        )
        normal_mesh = np.sum(
            np.cross(
                points[loop] - points[loop[0]],
                np.roll(points[loop], -1, axis=0) - points[loop[0]],
            ),
            axis=0,
        )
        orientations[row] = 1 if np.dot(normal_mesh, normal) > 0 else -1
        for a, b in zip(loop, np.roll(loop, -1), strict=True):
            edge_facets[edge_rows[tuple(sorted((int(a), int(b))))]] = facet
            edge = edge_rows[tuple(sorted((int(a), int(b))))]
            edge_residuals[edge] = max(edge_residuals[edge], face_residuals[row])
    if np.any(face_residuals > tolerance):
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "Constrained power faces exceed planar source residual tolerance.",
            stage="geometry_association",
            entity_ids=tuple(map(int, face_ids[selected[face_residuals > tolerance]])),
        )
    feature_sources = construction.feature_edge_sources
    combined = construction.feature_maximum_distance + construction.feature_coverage_gap
    combined = np.where(combined == 0, 0.0, np.nextafter(combined, np.inf))
    for feature in () if specification is None else specification.protected_features:
        identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
        if feature.scope.entity_dimension == 1:
            bad = identifiers[combined[identifiers] > feature.maximum_deviation]
        elif feature.scope.entity_dimension == 2:
            bad = np.asarray(
                [
                    int(facet)
                    for facet in identifiers
                    if np.max(face_residuals[facets[selected] == facet], initial=0.0)
                    > feature.maximum_deviation
                ],
                dtype=np.int64,
            )
        else:
            continue
        if bad.size:
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "Protected PLC feature fidelity exceeds its request.",
                stage="geometry_association",
                entity_ids=tuple(map(int, bad)),
            )
    boundary_edges = np.flatnonzero((edge_facets >= 0) | (feature_sources >= 0))
    revision = source.source_revision
    edge_dimensions = np.where(feature_sources[boundary_edges] >= 0, 1, 2)
    edge_sources = np.where(
        feature_sources[boundary_edges] >= 0,
        feature_sources[boundary_edges],
        edge_facets[boundary_edges],
    )
    edge_orientations = np.zeros(boundary_edges.size, dtype=np.int8)
    for row, edge in enumerate(boundary_edges):
        if feature_sources[edge] >= 0:
            pair = construction.feature_source_edges[feature_sources[edge]]
            direction = (
                construction.recovery.points[pair[1]]
                - construction.recovery.points[pair[0]]
            )
            mesh_edge = np.asarray(connectivity.edges)[edge]
            edge_orientations[row] = (
                1
                if np.dot(points[mesh_edge[1]] - points[mesh_edge[0]], direction) > 0
                else -1
            )
    associations = (
        _vertex_association(
            source, specification, construction, selected, face_residuals
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            mesh.entity_set(3).entity_set_id,
            cell_ids,
            tuple(_entity(revision, "region", int(x)) for x in construction.cell_regions),
            np.zeros(cell_ids.size),
            exact=True,
            source_dimensions=np.full(cell_ids.size, 3),
            source_indices=construction.cell_regions,
            source_entity_roles=(GeometrySourceEntityRole.REGION,) * cell_ids.size,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            mesh.entity_set(2).entity_set_id,
            face_ids[selected],
            tuple(_entity(revision, "facet", int(facets[x])) for x in selected),
            face_residuals,
            exact=False,
            orientations=orientations,
            source_dimensions=np.full(selected.size, 2),
            source_indices=facets[selected],
            source_entity_roles=(GeometrySourceEntityRole.FACET,) * selected.size,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            mesh.entity_set(1).entity_set_id,
            edge_ids[boundary_edges],
            tuple(
                _entity(
                    revision,
                    "edge" if feature_sources[x] >= 0 else "facet",
                    int(
                        feature_sources[x] if feature_sources[x] >= 0 else edge_facets[x]
                    ),
                )
                for x in boundary_edges
            ),
            np.asarray(
                [
                    construction.feature_maximum_distance[feature_sources[x]]
                    if feature_sources[x] >= 0
                    else edge_residuals[x]
                    for x in boundary_edges
                ]
            ),
            exact=False,
            source_dimensions=edge_dimensions,
            source_indices=edge_sources,
            orientations=edge_orientations,
            source_entity_roles=tuple(
                GeometrySourceEntityRole.EDGE
                if feature_sources[edge] >= 0
                else GeometrySourceEntityRole.FACET
                for edge in boundary_edges
            ),
        ),
    )
    return zones, patches, labels, associations


def _native_resource_quantities(error: MeshcoreError, /) -> tuple[tuple[str, float], ...]:
    quantities = [("native_status", float(error.status))]
    for name, values in (
        ("native_execution", error.work_evidence),
        ("native_memory", error.memory_evidence),
    ):
        if values is not None:
            quantities.extend(
                (f"{name}:{index}", float(value)) for index, value in enumerate(values)
            )
    return tuple(quantities)


def _periodic_source_failure(
    error: PeriodicPowerSourceRefusal, stage: str, /
) -> MeshingFailure:
    native = _construction_failure(error.native_refusal, stage).evidence
    quantities = list(native.achieved)
    if error.requested_work is not None:
        quantities.append(("periodic_requested_work_units", float(error.requested_work)))
    if error.completed_work is not None:
        quantities.append(("periodic_completed_work_units", float(error.completed_work)))
    return MeshingFailure(
        native.category,
        f"{native.message}; periodic source={error.preparation.source_id}",
        provider_code=native.provider_code,
        stage=native.stage,
        entity_ids=native.entity_ids,
        locations=native.locations,
        requested=(
            *native.requested,
            ("periodic_call_work_units", float(error.maximum_work)),
        ),
        achieved=tuple(quantities),
        checkpoint_id=native.checkpoint_id,
        logical_findings=native.logical_findings,
    )


def _construction_failure(
    error: MeshcoreError | PolyhedralGenerationError | CoordinateEnclosureResourceError,
    stage: str,
    /,
) -> MeshingFailure:
    """Preserve native resource/scientific status at generation and adaptation."""
    if isinstance(error, CoordinateEnclosureResourceError):
        return MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            str(error),
            stage=stage,
            requested=((error.resource, error.limit),),
            achieved=(("requested", error.requested), ("completed", error.completed)),
        )
    if isinstance(error, PeriodicPowerImageCapacityRefusal):
        return MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            str(error),
            stage=stage,
            requested=(("site_images", float(error.maximum_images)),),
            achieved=(
                *_native_resource_quantities(error),
                ("requested_site_images", float(error.requested_images)),
                ("completed_site_images", float(error.completed_images)),
            ),
        )
    if isinstance(error, PeriodicPowerSourceRefusal):
        return _periodic_source_failure(error, stage)
    if isinstance(error, MeshcoreError) and error.status is MeshcoreStatus.TIMEOUT:
        return MeshingFailure(
            MeshingFailureCategory.TIMED_OUT,
            str(error),
            stage=stage,
            achieved=_native_resource_quantities(error),
        )
    if isinstance(error, PlcRecoveryFailure):
        return _recovery_failure(error)
    if isinstance(error, RestrictedPowerFailure):
        return MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED
            if error.status == MeshcoreStatus.CAPACITY_EXCEEDED
            else MeshingFailureCategory.AUDIT_FAILED,
            str(error),
            stage=stage,
            entity_ids=tuple(int(value) for value in error.failure[1:3] if value >= 0),
            achieved=(
                *_native_resource_quantities(error),
                *tuple(
                    (f"power_failure_{index}", float(value))
                    for index, value in enumerate(error.failure)
                ),
                *tuple(
                    (f"power_counter_{index}", float(value))
                    for index, value in enumerate(error.counters)
                ),
            ),
        )
    if isinstance(error, PolyhedralGenerationError):
        if isinstance(error.achieved, dict):
            quantities = tuple(
                (str(name), float(value)) for name, value in error.achieved.items()
            )
        elif isinstance(error.achieved, np.ndarray):
            quantities = tuple(
                (f"unmet_entity_{index}", float(value))
                for index, value in enumerate(error.achieved.flat)
            )
        elif isinstance(error.achieved, (int, float)):
            quantities = (("consumed_work", float(error.achieved)),)
        else:
            quantities = ()
        return MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED
            if "budget" in error.reason
            else MeshingFailureCategory.COMPLIANCE_FAILED,
            str(error),
            stage=stage,
            entity_ids=error.entities,
            achieved=quantities,
        )
    return MeshingFailure(
        MeshingFailureCategory.RESOURCE_EXHAUSTED
        if error.status == MeshcoreStatus.CAPACITY_EXCEEDED
        else MeshingFailureCategory.AUDIT_FAILED,
        str(error),
        stage=stage,
        achieved=_native_resource_quantities(error),
    )


def _execute_polyhedral_route(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedPolyhedralVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Construct, independently certify, and publish the canonical result."""
    if (
        prepared.source_binding_id != source.binding_id
        or prepared.specification_id != specification.specification_id
    ):
        raise ValueError(
            "Prepared polyhedral route binds a different source or specification."
        )
    started = monotonic()
    phase_names: dict[str, NativeMeshingPhase] = {
        "regular_triangulation": "regular_triangulation",
        "power_adjacency": "power_adjacency",
        "power_clipping": "power_clipping",
        "power_components": "power_components",
        "power_publication": "power_publication",
    }

    def native_phase(
        phase: str, elapsed: float, work_units: int | None, invocations: int
    ) -> None:
        if record_phase is not None:
            record_phase(
                NativeMeshingPhaseMeasurement(
                    phase_names[phase], elapsed, work_units, invocations
                )
            )

    construction_started = phase_started(record_phase)
    try:
        seeds, seed_regions = _seeds(source.complex, specification)
        construction = generate_polyhedral_volume(
            source.complex,
            sites=prepared.sites,
            weights=prepared.weights,
            schedule=prepared.schedule,
            source_id=source.source_id,
            seeds=seeds,
            seed_regions=seed_regions,
            protected_vertices=_protected_vertices(specification),
            periodic_constraints=specification.periodic_constraints,
            record_native_phase=None if record_phase is None else native_phase,
        )
    except (
        MeshcoreError,
        PolyhedralGenerationError,
        CoordinateEnclosureResourceError,
    ) as error:
        raise _construction_failure(error, "volume_fill") from error
    record_elapsed(
        record_phase,
        "construction",
        construction_started,
        work_units=construction.work_units,
    )
    check_deadline(started, specification.limits, MeshingStageKind.VOLUME_FILL)
    mesh = construction.mesh
    c = _polyhedral_connectivity(mesh)
    sizes = {
        "vertices": mesh.coordinates.shape[0],
        "edges": c.edge_count,
        "faces": c.face_count,
        "cells": c.cell_count,
        "connectivity_entries": sum(
            np.asarray(value).size
            for value in (c.cell_vertex_values, c.face_vertex_values, c.cell_face_values)
        ),
    }
    for name, achieved in sizes.items():
        maximum = getattr(specification.limits, f"maximum_{name}")
        if achieved > maximum:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"Polyhedral {name} capacity exceeded.",
                stage="volume_fill",
                requested=((name, maximum),),
                achieved=((name, achieved),),
            )
    compliance_started = phase_started(record_phase)
    lengths, growth = edge_size_evidence(
        np.asarray(mesh.coordinates), np.asarray(c.edges)
    )
    size_control = specification.size_controls[0]
    if not isinstance(size_control, UniformSizeControl):
        raise TypeError(
            "Polyhedral size compliance requires the admitted uniform control."
        )
    requested, achieved, issues = uniform_size_compliance(
        size_control, specification.size_compliance, lengths, growth
    )
    achieved.extend(
        (
            ("cell_diameter_upper", construction.maximum_cell_diameter),
            (
                "volume_optimization_relative_residual",
                construction.optimization_relative_residual,
            ),
        )
    )
    compliance = MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )
    record_elapsed(record_phase, "compliance", compliance_started)
    with measure_phase(record_phase, "geometry_association"):
        zones, patches, labels, associations = _organization(
            source, specification, construction, prepared.schedule.feature_tolerance
        )
    data = construction.diagram.construction
    diagnostics = (
        MeshingDiagnostic(
            MeshingDiagnosticSeverity.INFO,
            "Reciprocal restricted components; non-star-visible sites decomposed into convex pieces. Site ancestry is not a field interpolation stencil.",
            quantities=tuple(
                (f"power_counter_{i}", float(x), float(x))
                for i, x in enumerate(data.counters)
            ),
        ),
    )
    stages = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(source.binding_id,),
            output_ids=(prepared.prepared_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.VOLUME_FILL,
            MeshingStageStatus.PASSED,
            input_ids=(prepared.prepared_id,),
            output_ids=(construction.construction_id, mesh.mesh_id),
            created_count=c.cell_count,
            diagnostics=diagnostics,
        ),
    )
    check_deadline(started, specification.limits, MeshingStageKind.GEOMETRY_AUDIT)
    return publish_native_result(
        mesh,
        coordinate_contract,
        compliance,
        stages,
        provider,
        {
            "kind": "native-polyhedral-volume-mesh",
            "route": "plc_restricted_power",
            "source": source.binding_id,
            "plan": plan_id,
            "construction": construction.construction_id,
            "site_ancestry": construction.site_parents.tolist(),
            "cell_sites": construction.cell_sites.tolist(),
            "cell_components": construction.component_ids.tolist(),
            "cell_regions": construction.cell_regions.tolist(),
            "face_facets": construction.face_facets.tolist(),
            "face_recovery_triangles": construction.face_recovery_triangles.tolist(),
            "feature_edge_sources": construction.feature_edge_sources.tolist(),
            "feature_source_edges": construction.feature_source_edges.tolist(),
            "carrier_points": construction.recovery.points.tolist(),
            "carrier_tetrahedra": construction.recovery.tetrahedra.tolist(),
            "vertex_carriers": construction.diagram.construction.vertex_carriers.tolist(),
            "feature_distance_upper": construction.feature_maximum_distance.tolist(),
            "feature_coverage_gap_upper": construction.feature_coverage_gap.tolist(),
            "feature_tolerance": prepared.schedule.feature_tolerance,
            "requested_site_volumes": construction.requested_site_volumes.tolist(),
            "achieved_site_volumes": construction.achieved_site_volumes.tolist(),
            "weight_relative_residual": construction.optimization_relative_residual,
            "refinement_steps": construction.refinement_steps,
            "lloyd_steps": construction.lloyd_steps,
            "work_units": construction.work_units,
            "construction_cell_volumes": data.cell_volumes[
                np.asarray(c.cell_global_ids)
            ].tolist(),
            "construction_cell_first_moments": data.cell_moments[
                np.asarray(c.cell_global_ids)
            ].tolist(),
            "construction_cell_site_second_moments": data.cell_second_moments[
                np.asarray(c.cell_global_ids)
            ].tolist(),
            "piece_sites": construction.diagram.piece_original_sites.tolist(),
            "piece_image_sites": data.piece_sites.tolist(),
            "piece_tetrahedra": data.piece_tets.tolist(),
            "piece_cells": data.piece_cells.tolist(),
            "piece_volumes": data.piece_volumes.tolist(),
            "cell_image_sites": data.cell_sites[np.asarray(c.cell_global_ids)].tolist(),
            "cell_image_exponents": construction.diagram.cell_image_exponents[
                np.asarray(c.cell_global_ids)
            ].tolist(),
            "piece_image_exponents": construction.diagram.piece_image_exponents.tolist(),
            "periodic_constraints": [
                value.constraint_id for value in construction.periodic_constraints
            ],
            "periodic_seam_faces": None
            if construction.seam_face_pairs is None
            else construction.seam_face_pairs.tolist(),
            "periodic_seam_vertices": None
            if construction.seam_vertex_pairs is None
            else construction.seam_vertex_pairs.tolist(),
            "source_work_units": data.source_work_units,
            "preparation_work_units": construction.diagram.preparation_work_units,
            "adjacency_work_units": construction.diagram.adjacency_work_units,
            "site_points": construction.diagram.points.tolist(),
            "site_weights": construction.diagram.weights.tolist(),
        },
        NativeCertificationRequest(
            MeshCertificationSchedule("volume_plc"),
            source.source_id,
            source.source_revision,
            specification.limits,
            domain=construction.domain,
            cell_regions=construction.cell_regions,
        ),
        audit_policy=CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=CellMeshAuditDisposition.REJECT,
        ),
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity_entries",
            "work_units",
            "wall_seconds",
        ),
        unenforced_limits=(
            "cavity_cells",
            "geometry_queries",
            "scratch_bytes",
            "data_bytes",
        ),
        zones=zones,
        patches=patches,
        labels=labels,
        associations=associations,
        geometry=construction.geometry,
        record_phase=record_phase,
    )


def execute_polyhedral_route(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedPolyhedralVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Keep construction, association and publication in one original scope."""
    started = monotonic()
    try:
        with native_volume_execution_budget(
            specification.limits, operation_started=started
        ):
            native_volume_checkpoint(
                specification.limits,
                MeshingStageKind.SOURCE_INSPECTION,
                operation_started=started,
            )
            result = _execute_polyhedral_route(
                source,
                specification,
                prepared,
                coordinate_contract,
                provider,
                plan_id,
                record_phase=record_phase,
            )
            native_volume_checkpoint(
                specification.limits,
                MeshingStageKind.CERTIFICATION,
                operation_started=started,
            )
            return result
    except (
        MeshcoreError,
        CoordinateEnclosureResourceError,
        PolyhedralGenerationError,
    ) as error:
        raise _construction_failure(error, "native_execution") from error


__all__ = [
    "PreparedPolyhedralVolume",
    "polyhedral_support_issues",
    "execute_polyhedral_route",
]
