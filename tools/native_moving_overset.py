# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native wall cutting, moving donor refresh, Poisson update and durable restart.

Run: PYTHONPATH=. python -m tools.native_moving_overset
Interpolation uses each actual P1 FE layout and is explicitly nonconservative.
The moving square annulus has a protected solid wall and an artificial exterior.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax.discretization import (
    FiniteElementDiscretization,
    FiniteElementPlan,
    PolygonalConnectivity,
)
from phydrax.equations import CompiledFiniteElementProblem
from phydrax.lifecycle import CompositionEntry
from phydrax.lifecycle._meshing_field_records import MeshingFieldDeclaration
from phydrax.linalg import LinearSolvePolicy
from phydrax.meshing import (
    CellMeshingResult,
    MeshPart,
    NativeMeshingPhaseRecorder,
    NativeMeshingPlan,
    NativeMeshingProvider,
    OversetConnectivity,
    OversetPartSpec,
    OversetPolicy,
    OversetRegistration,
    OversetVertexStatus,
    prepare_overset_connectivity,
    prepare_overset_field_transfer,
    prepare_overset_motion_rebind,
)


jax.config.update("jax_enable_x64", True)


def _cell_result(part: MeshPart) -> CellMeshingResult:
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError("The native planar workflow requires an accepted cell mesh.")
    return carrier


def _archive_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Archived {name} must be a named scientific record mapping.")
    records: dict[str, object] = {}
    for key, record in value.items():
        if not isinstance(key, str):
            raise TypeError(f"Archived {name} must use string record names.")
        records[key] = record
    return records


def rigid_planar_successor(
    part: MeshPart,
    offset: ArrayLike,
    *,
    motion_id: str,
    plan: NativeMeshingPlan,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> tuple[MeshPart, NativeMeshingPlan]:
    """Translate actual native authority and recertify unchanged mesh topology.

    The original admitted plan is required: a split mesh boundary cannot
    reconstruct the original source or its hard physical request. No accepted
    mesh is regenerated, and no source-bound evidence is copied to moved points.
    """
    from dataclasses import replace

    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from phydrax.discretization._cell_geometry_validity import (
        certify_cell_geometry_validity,
    )
    from phydrax.geometry import PiecewiseLinearDomain, PlanarMeshRegion, SegmentMesh
    from phydrax.meshing._association import (
        GeometryAssociationKind,
        GeometrySourceEntityRole,
        PlcAssociationTransfer,
    )
    from phydrax.meshing._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
    from phydrax.meshing._certification import (
        MeshCertificationPreparedEvidence,
        MeshCertificationSchedule,
    )
    from phydrax.meshing._contracts import MeshingDerivativeMode, SurfaceMeshingSpec
    from phydrax.meshing._controls import ProtectedFeature
    from phydrax.meshing._quality import evaluate_cell_quality
    from phydrax.meshing._scope import MeshingScope
    from phydrax.meshing._sizing import UniformSizeControl
    from phydrax.meshing._trace import (
        MeshingStageKind,
        MeshingStageReport,
        MeshingStageStatus,
    )
    from phydrax.meshing.providers._native import NativeMeshingProvider
    from phydrax.meshing.providers._native_planar import (
        _planar_compliance,
        _source_arrays,
        PreparedPlanarDomain,
    )
    from phydrax.meshing.providers._native_publication import (
        NativeCertificationRequest,
        publish_native_result,
    )
    from phydrax.meshing.providers._native_sources import (
        NativePlanarSource,
        source_entity_id,
    )

    if not isinstance(plan, NativeMeshingPlan) or not isinstance(
        plan.source, NativePlanarSource
    ):
        raise TypeError(
            "Rigid planar motion requires the actual original native planar plan."
        )
    if not isinstance(plan.specification, SurfaceMeshingSpec) or not isinstance(
        plan.prepared, PreparedPlanarDomain
    ):
        raise TypeError(
            "Rigid planar motion consumes the admitted constrained planar route."
        )
    before = _cell_result(part)
    source_certificate = before.certification
    if source_certificate is None or source_certificate.request.domain is None:
        raise ValueError(
            "Rigid planar motion requires its accepted represented-domain certificate."
        )
    accepted_domain = source_certificate.request.domain
    if not isinstance(accepted_domain, PiecewiseLinearDomain):
        raise TypeError(
            "Rigid planar motion requires a represented piecewise-linear support domain."
        )
    connectivity = before.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Rigid planar motion requires the accepted polygonal incidence.")
    if before.compliance.specification_id != plan.specification.specification_id:
        raise ValueError(
            "The native request does not bind this accepted mesh publication."
        )
    shift = np.asarray(offset, dtype=np.float64)
    if shift.shape != (2,) or not np.all(np.isfinite(shift)):
        raise ValueError(
            "Rigid planar offset must be a finite two-dimensional translation."
        )
    if not motion_id:
        raise ValueError("Rigid motion must have an explicit identity.")
    source, specification = plan.source, plan.specification
    revision = canonical_fingerprint(
        {
            "kind": "native-planar-rigid-pose",
            "source": source.binding_id,
            "motion": str(motion_id),
            "offset": array_tree_fingerprint(shift),
        }
    )
    edges, offsets = (
        np.asarray(source.region.edges),
        np.asarray(source.region.loop_offsets),
    )
    loops = tuple(
        tuple(int(row) for row in edges[first:last, 0])
        for first, last in zip(offsets[:-1], offsets[1:], strict=True)
    )
    region = PlanarMeshRegion(
        np.asarray(source.region.vertices) + shift, loops, feature_id=source.source_id
    )
    embedded = (
        None
        if source.embedded is None
        else SegmentMesh(
            source.embedded.vertices + shift,
            source.embedded.edges,
            source_id=source.embedded.source_id,
        )
    )
    moved_source = NativePlanarSource(region, revision, embedded=embedded)

    def rebound(scope: MeshingScope) -> MeshingScope:
        if (scope.source_id, scope.source_revision) != (
            source.source_id,
            source.source_revision,
        ):
            raise ValueError("A physical control is bound to another source revision.")
        return MeshingScope(
            scope.source_id,
            revision,
            scope.entity_kind,
            scope.entity_dimension,
            scope.entity_set_id,
            scope.entity_ids,
        )

    controls = []
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            raise TypeError(
                "The admitted planar route must retain its real uniform size controls."
            )
        controls.append(
            UniformSizeControl(
                rebound(control.scope),
                control.target_size,
                minimum_size=control.minimum_size,
                maximum_size=control.maximum_size,
                maximum_growth_rate=control.maximum_growth_rate,
                strength=control.strength,
                priority=control.priority,
            )
        )
    features = tuple(
        ProtectedFeature(
            rebound(feature.scope),
            feature.feature_kind,
            maximum_deviation=feature.maximum_deviation,
            hard=feature.hard,
        )
        for feature in specification.protected_features
    )
    moved_specification = SurfaceMeshingSpec(
        specification.target,
        rebound(specification.scope),
        planar_embedding=specification.planar_embedding,
        size_controls=tuple(controls),
        protected_features=features,
        region_controls=specification.region_controls,
        patch_controls=specification.patch_controls,
        periodic_constraints=specification.periodic_constraints,
        layer_controls=specification.layer_controls,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        limits=specification.limits,
        quality_target=specification.quality_target,
        background_metric=specification.background_metric,
        deterministic=specification.deterministic,
    )
    moved_plan = NativeMeshingProvider(plan.options).plan(
        moved_source,
        moved_specification,
        coordinate_contract=plan.coordinate_contract,
        record_phase=record_phase,
    )
    prepared = moved_plan.prepared
    if not isinstance(prepared, PreparedPlanarDomain):
        raise TypeError("Source translation did not retain the admitted planar route.")
    mesh = before.mesh.with_coordinates(
        before.mesh.coordinates + shift, numeric_version=revision
    )
    if mesh.topology_id != before.mesh.topology_id:
        raise RuntimeError(
            "A rigid coordinate change altered scientific topology identity."
        )
    edge_set = before.mesh.entity_set(1)
    ids = np.asarray(edge_set.entity_ids)
    order = np.argsort(ids, kind="stable")
    _, source_edges, _ = _source_arrays(source)
    edge_names = {
        source_entity_id(source.source_revision, "edge", index): index
        for index in range(len(source_edges))
    }
    pairs, sources = [], []
    for association in before.associations:
        if (
            association.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
            and association.source_id == source.source_id
            and association.source_revision == source.source_revision
            and association.target_entity_set_id == edge_set.entity_set_id
        ):
            roles = association.source_entity_roles
            if roles is None:
                raise ValueError(
                    "Rigid planar motion requires explicit accepted source entity roles."
                )
            for identifier, name, dimension, role in zip(
                np.asarray(association.target_global_ids),
                association.source_entity_ids,
                np.asarray(association.source_dimensions),
                roles,
                strict=True,
            ):
                if dimension == 2 and role is GeometrySourceEntityRole.REGION:
                    continue
                if dimension != 1 or role is not GeometrySourceEntityRole.EDGE:
                    raise ValueError(
                        "The accepted edge association has an invalid planar source stratum."
                    )
                if name not in edge_names:
                    raise ValueError(
                        "The accepted edge association names an unknown source entity."
                    )
                row = order[np.searchsorted(ids[order], identifier)]
                pairs.append(np.asarray(connectivity.edges)[row])
                sources.append(edge_names[name])
    if not pairs or set(sources) != set(range(len(source_edges))):
        raise ValueError(
            "Rigid motion requires every authoritative source constraint association."
        )
    constrained, constraint_sources = (
        np.asarray(pairs, dtype=np.int64),
        np.asarray(sources, dtype=np.int64),
    )
    points = np.asarray(mesh.coordinates)
    recovered_domain, rounding, admissible = prepared.domain(
        moved_source, points, constrained, constraint_sources
    )
    domain = PiecewiseLinearDomain(
        accepted_domain.vertices + shift,
        accepted_domain.facets,
        accepted_domain.facet_regions,
        recovered_domain.region_ids,
        source_id=moved_source.source_id,
    )
    moved_geometry = phx.discretization.CellGeometrySpec(
        dict(zip(before.geometry.block_names, before.geometry.elements, strict=True)),
        dict(
            zip(before.geometry.block_names, before.geometry.geometry_dofs, strict=True)
        ),
        before.geometry.coordinates + shift,
        restriction_source=before.geometry.restriction_source,
    )

    def authority_transfer(
        authority: NativePlanarSource,
        support_domain: PiecewiseLinearDomain,
    ) -> PlcAssociationTransfer:
        authority_points, authority_edges, _ = _source_arrays(authority)
        return PlcAssociationTransfer(
            support_domain,
            plan.coordinate_contract,
            authority.source_revision,
            edge_vertices=authority_edges,
            source_vertices=authority_points,
            triangle_vertices=np.empty((0, 3), dtype=np.int64),
            triangle_facets=np.empty((0,), dtype=np.int64),
            facet_regions=np.empty((0, 2), dtype=np.int64),
        )

    predecessor_transfer = authority_transfer(source, accepted_domain)
    successor_transfer = authority_transfer(moved_source, domain)
    audit_policy = CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=CellMeshAuditDisposition.REJECT,
    )
    region_labels = np.zeros(mesh.blocks[0].cell_count, dtype=np.int64)
    request = NativeCertificationRequest(
        MeshCertificationSchedule("volume_plc"),
        moved_source.source_id,
        moved_source.source_revision,
        moved_specification.limits,
        domain=domain,
        cell_regions=region_labels,
    )
    validity = certify_cell_geometry_validity(
        moved_geometry,
        mesh=mesh,
        policy=audit_policy.validity_policy,
    )
    premises = MeshCertificationPreparedEvidence(
        mesh,
        moved_geometry,
        validity,
        schedule=request.schedule,
        domain=domain,
        cell_regions=region_labels,
        limits=request.certificate_limits,
    )
    patches, zones, labels, associations = (
        successor_transfer.transition_source_associations(
            before,
            predecessor_transfer,
            mesh,
            geometry=moved_geometry,
            translation=shift,
            embedding=premises.embedding,
            coverage=premises.coverage,
        )
    )
    measured = evaluate_cell_quality(mesh)
    minimum_angle = float(np.min(np.asarray(measured.minimum_angle)))
    retained_steiner = mesh.coordinates.shape[0] - plan.prepared.points.shape[0]
    if retained_steiner < 0:
        raise ValueError(
            "The accepted topology lost original source constraint vertices."
        )
    # This status is the observed unchanged-topology predicate; no CDT was run.
    status = "ok" if mesh.topology_id == before.mesh.topology_id else "topology_changed"
    compliance = _planar_compliance(
        moved_specification,
        prepared,
        points,
        np.asarray(mesh.blocks[0].vertices),
        (constrained, constraint_sources),
        (np.degrees(minimum_angle), retained_steiner, status),
        (rounding, admissible),
    )
    result = publish_native_result(
        mesh,
        plan.coordinate_contract,
        compliance,
        (
            MeshingStageReport(
                MeshingStageKind.GEOMETRY_ASSOCIATION,
                MeshingStageStatus.PASSED,
                input_ids=(before.result_id, source.binding_id),
                output_ids=(mesh.mesh_id, moved_source.binding_id),
            ),
        ),
        before.provider,
        {
            "kind": "native-planar-rigid-motion",
            "source": moved_source.binding_id,
            "source_result": before.result_id,
            "plan": moved_plan.plan_id,
            "motion": str(motion_id),
        },
        replace(request, prepared=premises),
        audit_policy=audit_policy,
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity",
            "data_bytes",
        ),
        unenforced_limits=("cavity_cells",),
        patches=patches,
        zones=zones,
        labels=labels,
        associations=associations,
        record_phase=record_phase,
        geometry=moved_geometry,
    )
    successor = phx.meshing.MeshPart(part.name, result)
    if (
        _cell_result(successor).geometry.geometry_layout_id
        != before.geometry.geometry_layout_id
    ):
        raise RuntimeError("Rigid motion changed the accepted coordinate layout.")
    return successor, moved_plan


def deform_planar_interior(
    part: MeshPart,
    coordinates: ArrayLike,
    *,
    motion_id: str,
    plan: NativeMeshingPlan,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> MeshPart:
    """Recertify native mesh motion inside its unchanged authoritative domain."""
    from dataclasses import replace

    from phydrax._fingerprint import canonical_fingerprint
    from phydrax.discretization._cell_geometry_validity import (
        certify_cell_geometry_validity,
    )
    from phydrax.geometry import PiecewiseLinearDomain
    from phydrax.meshing._association import (
        _retain_association_scopes,
        PlcAssociationTransfer,
    )
    from phydrax.meshing._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
    from phydrax.meshing._certification import (
        MeshCertificationPreparedEvidence,
        MeshCertificationSchedule,
    )
    from phydrax.meshing._contracts import MeshingDerivativeMode, SurfaceMeshingSpec
    from phydrax.meshing._lineage import identity_lineage, inherit_mesh_organization
    from phydrax.meshing._quality import evaluate_cell_quality
    from phydrax.meshing._trace import (
        MeshingStageKind,
        MeshingStageReport,
        MeshingStageStatus,
    )
    from phydrax.meshing.providers._native_planar import (
        _planar_compliance,
        _source_arrays,
        PreparedPlanarDomain,
    )
    from phydrax.meshing.providers._native_publication import (
        NativeCertificationRequest,
        publish_native_result,
    )
    from phydrax.meshing.providers._native_sources import NativePlanarSource

    before = _cell_result(part)
    if (
        not isinstance(plan.source, NativePlanarSource)
        or not isinstance(plan.prepared, PreparedPlanarDomain)
        or not isinstance(plan.specification, SurfaceMeshingSpec)
    ):
        raise TypeError(
            "Interior motion requires the admitted native planar source and request."
        )
    if before.compliance.specification_id != plan.specification.specification_id:
        raise ValueError("Interior motion must bind the original hard native request.")
    certificate = before.certification
    if certificate is None or not isinstance(
        certificate.request.domain, PiecewiseLinearDomain
    ):
        raise ValueError(
            "Interior motion requires its accepted represented-domain certificate."
        )
    domain = certificate.request.domain
    points = np.asarray(coordinates, dtype=np.float64)
    if (
        points.shape != before.mesh.coordinates.shape
        or not np.all(np.isfinite(points))
        or not motion_id
    ):
        raise ValueError(
            "Interior motion requires finite same-layout coordinates and an explicit motion identity."
        )
    connectivity = before.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Interior planar motion requires polygonal incidence.")
    boundary = np.unique(
        np.asarray(connectivity.edges)[np.asarray(connectivity.boundary_edges)]
    )
    if not np.array_equal(
        points[boundary], np.asarray(before.mesh.coordinates)[boundary]
    ):
        raise ValueError(
            "Interior motion cannot change the authoritative physical boundary."
        )
    if not np.array_equal(before.geometry.coordinates, before.mesh.coordinates):
        raise ValueError(
            "Interior motion requires the actual affine native coordinate map."
        )
    revision = canonical_fingerprint(
        {
            "kind": "native-planar-interior-motion",
            "source": before.result_id,
            "motion": motion_id,
        }
    )
    mesh = before.mesh.with_coordinates(points, numeric_version=revision)
    geometry = phx.discretization.CellGeometrySpec(
        dict(zip(before.geometry.block_names, before.geometry.elements, strict=True)),
        dict(
            zip(before.geometry.block_names, before.geometry.geometry_dofs, strict=True)
        ),
        points,
    )
    source_points, source_edges, _ = _source_arrays(plan.source)
    transfer = PlcAssociationTransfer(
        domain,
        plan.coordinate_contract,
        plan.source.source_revision,
        edge_vertices=source_edges,
        source_vertices=source_points,
        triangle_vertices=np.empty((0, 3), dtype=np.int64),
        triangle_facets=np.empty((0,), dtype=np.int64),
        facet_regions=np.empty((0, 2), dtype=np.int64),
        maximum_support_queries=plan.specification.limits.maximum_geometry_queries,
    )
    audit_policy = CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=CellMeshAuditDisposition.REJECT,
    )
    region_labels = np.zeros(mesh.blocks[0].cell_count, dtype=np.int64)
    request = NativeCertificationRequest(
        MeshCertificationSchedule("volume_plc"),
        plan.source.source_id,
        plan.source.source_revision,
        plan.specification.limits,
        domain=domain,
        cell_regions=region_labels,
    )
    validity = certify_cell_geometry_validity(
        geometry, mesh=mesh, policy=audit_policy.validity_policy
    )
    prepared = MeshCertificationPreparedEvidence(
        mesh,
        geometry,
        validity,
        schedule=request.schedule,
        domain=domain,
        cell_regions=region_labels,
        limits=request.certificate_limits,
    )
    lineage = identity_lineage(before.mesh, mesh)
    proved = transfer.propagate(
        before,
        lineage,
        mesh,
        geometry=geometry,
        embedding=prepared.embedding,
        coverage=prepared.coverage,
    )
    edge_set = mesh.entity_set(1)
    edge_support = next(
        value for value in proved if value.target_entity_set_id == edge_set.entity_set_id
    )
    edge_rows = edge_support.target_rows(np.asarray(edge_set.entity_ids))
    selected = np.asarray(edge_support.source_dimensions)[edge_rows] == 1
    constrained = np.asarray(connectivity.edges)[selected]
    constraint_sources = np.asarray(edge_support.source_indices)[edge_rows][selected]
    _, rounding, admissible = plan.prepared.domain(
        plan.source, points, constrained, constraint_sources
    )
    measured = evaluate_cell_quality(mesh)
    compliance = _planar_compliance(
        plan.specification,
        plan.prepared,
        points,
        np.asarray(mesh.blocks[0].vertices),
        (constrained, constraint_sources),
        (
            np.degrees(float(np.min(np.asarray(measured.minimum_angle)))),
            mesh.coordinates.shape[0] - plan.prepared.points.shape[0],
            "ok",
        ),
        (rounding, admissible),
    )
    associations = _retain_association_scopes(before.associations, proved)
    patches, zones, labels = inherit_mesh_organization(before, mesh, lineage)
    result = publish_native_result(
        mesh,
        plan.coordinate_contract,
        compliance,
        (
            MeshingStageReport(
                MeshingStageKind.GEOMETRY_ASSOCIATION,
                MeshingStageStatus.PASSED,
                input_ids=(before.result_id,),
                output_ids=(mesh.mesh_id,),
            ),
        ),
        before.provider,
        {
            "kind": "native-planar-interior-motion",
            "source_result": before.result_id,
            "plan": plan.plan_id,
            "motion": motion_id,
        },
        replace(request, prepared=prepared),
        audit_policy=audit_policy,
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity",
            "data_bytes",
        ),
        unenforced_limits=("cavity_cells",),
        patches=patches,
        zones=zones,
        labels=labels,
        associations=associations,
        geometry=geometry,
        record_phase=record_phase,
    )
    return MeshPart(part.name, result)


def native_planar_part(
    name: str,
    half: float,
    *,
    cavity: float | None = None,
    size: float = 0.4,
) -> tuple[MeshPart, NativeMeshingPlan]:
    """Author and execute the actual protected native planar physical request."""
    from phydrax._fingerprint import canonical_fingerprint

    vertices = np.asarray(
        ((-half, -half), (half, -half), (half, half), (-half, half)), dtype=np.float64
    )
    loops = ((0, 1, 2, 3),)
    if cavity is not None:
        vertices = np.concatenate(
            (
                vertices,
                np.asarray(
                    (
                        (-cavity, -cavity),
                        (-cavity, cavity),
                        (cavity, cavity),
                        (cavity, -cavity),
                    ),
                    dtype=np.float64,
                ),
            )
        )
        loops = (*loops, (4, 5, 6, 7))
    revision = canonical_fingerprint(
        {
            "kind": "native-moving-planar-authority",
            "name": name,
            "vertices": vertices.tolist(),
            "loops": loops,
        }
    )
    source = phx.meshing.NativePlanarSource(
        phx.geometry.PlanarMeshRegion(vertices, loops, feature_id=name),
        revision,
    )
    scope = phx.meshing.MeshingScope(
        name,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "fluid-region",
        np.asarray((0,), dtype=np.int64),
    )
    protected = tuple(
        phx.meshing.ProtectedFeature(
            phx.meshing.MeshingScope(
                name,
                revision,
                phx.meshing.MeshingEntityKind.GEOMETRY,
                1,
                f"source-loop:{index}",
                np.arange(4 * index, 4 * index + 4, dtype=np.int64),
            ),
            phx.meshing.FeatureKind.CURVE,
            maximum_deviation=1e-10,
            hard=True,
        )
        for index in range(len(loops))
    )
    request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 2, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                size,
                maximum_size=1.5 * size,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(
            relative_tolerance=0.5,
            target_statistics=("p50",),
        ),
        protected_features=protected,
        quality_target=phx.meshing.MeshQualityTarget(
            minimum_angle=np.radians(25.0), hard=True
        ),
    )
    plan = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("planar_constrained_delaunay"),
    ).plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
    return phx.meshing.MeshPart(name, plan.execute()), plan


def field_plans(connectivity: OversetConnectivity) -> dict[str, FiniteElementPlan]:
    """Original owned physical inputs retained for source-preserving restart."""
    plans = {}
    for part in connectivity.assembly.parts:
        carrier = _cell_result(part)
        plans[part.name] = FiniteElementPlan(
            carrier.mesh,
            phx.discretization.FiniteElementFieldSpec(
                "u", phx.discretization.lagrange_element("triangle", 1)
            ),
            coordinate_spec=carrier.geometry,
        )
    return plans


def fields(connectivity: OversetConnectivity) -> dict[str, FiniteElementDiscretization]:
    return {name: plan.prepare() for name, plan in field_plans(connectivity).items()}


def field_declarations(
    connectivity: OversetConnectivity,
    plans: Mapping[str, FiniteElementPlan],
    discretizations: Mapping[str, FiniteElementDiscretization],
    coefficients: Mapping[str, Array],
) -> dict[str, MeshingFieldDeclaration]:
    """Authored normalized-temperature state, in its actual accepted FE layout.

    ``u`` is dimensionless; coordinates are SI lengths, so the unit Poisson
    source denotes one inverse-square-meter in this normalized model. Geometry
    decisions are frozen within an accepted epoch. This steady solve has no
    time/material history. The historical ``temperature:<part>`` entry names
    remain explicit state-role identifiers, not a Kelvin-unit inference.
    """
    from phydrax._differentiation import BranchDifferentiationPolicy
    from phydrax._frozendict import frozendict
    from phydrax.discretization._views import FieldTracePolicy
    from phydrax.lifecycle._meshing_field_records import (
        MeshingFieldStateRole,
        prepare_meshing_field_index_binding,
    )

    names = {part.name for part in connectivity.assembly.parts}
    if set(plans) != names or set(discretizations) != names or set(coefficients) != names:
        raise ValueError(
            "Physical field inputs must name every accepted registration part."
        )
    declarations = {}
    specs = {spec.part_name: spec for spec in connectivity.specs}
    for part in connectivity.assembly.parts:
        name, plan, prepared = part.name, plans[part.name], discretizations[part.name]
        carrier = _cell_result(part)
        if prepared.plan_id != plan.plan_id:
            raise ValueError(
                "The retained physical plan does not bind this prepared field."
            )
        if tuple(field.name for field in plan.fields) != ("u",):
            raise ValueError(
                "This normalized scalar Poisson workflow declares exactly field u."
            )
        value = coefficients[name]
        index_binding = prepare_meshing_field_index_binding(
            carrier,
            "mesh/vertex_ids",
            field_space=prepared.field_spaces[0],
        )
        declarations[name] = MeshingFieldDeclaration(
            part_name=name,
            owner="finite_element",
            topology_id=carrier.mesh.topology_id,
            geometry_layout_id=carrier.geometry.geometry_layout_id,
            field_space_ids=frozendict(
                {space.name: space.field_space_id for space in prepared.field_spaces}
            ),
            value_units=frozendict({"u": (phx.units.ONE,)}),
            maximum_derivative_orders=frozendict({"u": 1}),
            branch_policy=BranchDifferentiationPolicy.FROZEN_DECISION,
            trace_policy=FieldTracePolicy("cell-sided"),
            state_roles=(
                MeshingFieldStateRole(
                    f"temperature:{name}",
                    "u",
                    "coefficients",
                    tuple(value.shape),
                    np.dtype(value.dtype),
                    connectivity.epoch,
                    index_binding=index_binding,
                ),
            ),
            history_policy="none",
            numeric_version=prepared.numeric_version,
            finite_element_fields=plan.fields,
            precision_policy=plan.precision_policy,
            source_wall_policy=specs[name],
        )
    return declarations


def field_entries(
    connectivity: OversetConnectivity,
    discretizations: Mapping[str, FiniteElementDiscretization],
    coefficients: Mapping[str, Array],
    revision: str,
) -> tuple[tuple[CompositionEntry, ...], tuple[CompositionEntry, ...]]:
    part_entries = {entry.entry_id: entry for entry in connectivity.composition_entries()}
    prepared, states = [], []
    for name, field in sorted(discretizations.items()):
        part = part_entries[f"overset:part:{name}"]
        artifact = phx.lifecycle.CompositionEntry(
            field,
            entry_id=f"fe:{name}",
            role="discretization",
            owner_id="native-overset-example",
            structure_id=field.dof_maps[0].dof_map_id,
            revision_id=field.prepared_id,
            semantics_id=f"temperature-space:{name}",
            dependencies=(part.binding("revision"),),
        )
        prepared.append(artifact)
        states.append(
            phx.lifecycle.CompositionEntry(
                coefficients[name],
                entry_id=f"temperature:{name}",
                role="physical-state",
                owner_id="native-overset-example",
                structure_id=artifact.structure_id,
                revision_id=f"temperature:{name}:{revision}",
                semantics_id="temperature",
                dependencies=(artifact.binding("revision"),),
            )
        )
    return tuple(prepared), tuple(states)


def prepare_poisson_update(
    field: FiniteElementDiscretization,
    carried: Array,
) -> CompiledFiniteElementProblem:
    form = phx.equations.FiniteElementForm(
        "moving-annulus-poisson",
        "u",
        (phx.equations.DiffusionAction("u", 1.0), phx.equations.SourceAction("u", 1.0)),
    )
    constraint = phx.discretization.dirichlet_constraint(field, "u")
    return phx.equations.compile_finite_element_problem(
        form,
        field,
        constraint=constraint,
        dirichlet_values=carried,
    )


def poisson_solve_policy(field: FiniteElementDiscretization) -> LinearSolvePolicy:
    """Absolute solve accuracy below the independent original-system criterion."""
    return phx.linalg.LinearSolvePolicy(
        tolerance=phx.linalg.TolerancePolicy(
            relative=0.0,
            absolute=1e-11,
            max_steps=max(
                32, 4 * field.dof_maps[field._field_index("u")].global_dof_count
            ),
        )
    )


def poisson_update(
    field: FiniteElementDiscretization,
    carried: Array,
) -> tuple[Array, float]:
    problem = prepare_poisson_update(field, carried)
    system, rhs = problem.linear_system()
    if not isinstance(rhs, Array):
        raise TypeError(
            "The scalar Poisson problem requires one array-valued right-hand side."
        )
    result = phx.linalg.solve(system, rhs, policy=poisson_solve_policy(field))
    if not bool(jnp.all(result.successful)):
        raise RuntimeError("The moving-annulus Poisson update failed.")
    if not isinstance(result.value, Array):
        raise TypeError("The scalar Poisson solve requires one array-valued solution.")
    action = system.operator.mv(result.value)
    if not isinstance(action, Array):
        raise TypeError(
            "The scalar Poisson operator must produce one array-valued residual."
        )
    residual = float(jnp.max(jnp.abs(action - rhs)))
    if residual > 1e-9:
        raise RuntimeError("The Poisson update did not satisfy its native residual.")
    expanded = problem.expand(result.value)
    if not isinstance(expanded, Array):
        raise TypeError("The scalar Poisson update must expand to one coefficient array.")
    return expanded, residual


def restart_native(
    connectivity: OversetConnectivity,
    coefficients: Mapping[str, Array],
    *,
    plans: Mapping[str, NativeMeshingPlan],
    declarations: Mapping[str, MeshingFieldDeclaration],
    path: str | PathLike[str] | None = None,
    archive_limits: ArrayArchiveLimits | None = None,
) -> tuple[
    OversetConnectivity,
    dict[str, Array],
    dict[str, NativeMeshingPlan],
    dict[str, FiniteElementDiscretization],
]:
    """Durably restore full authority, hard requests, field declarations and state.

    Only canonical source-closure codecs own serialization. Prepared spatial
    connectivity, solver plans and field query caches are rebuilt after restore.
    This steady example owns one history-free physical coefficient field per
    part; stateful consumers must pass their complete named state to lifecycle.
    """
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.lifecycle._meshing_field_records import prepare_meshing_field_owner
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    limits = DEFAULT_ARRAY_ARCHIVE_LIMITS if archive_limits is None else archive_limits

    connectivity.require_complete()
    names = {part.name for part in connectivity.assembly.parts}
    if set(coefficients) != names or set(plans) != names or set(declarations) != names:
        raise ValueError(
            "Native restart requires complete generation and field inputs for every part."
        )
    primary_part = next(
        (
            part
            for part in connectivity.assembly.parts
            if _cell_result(part).certification is not None
        ),
        None,
    )
    if primary_part is None:
        raise ValueError(
            "Native restart must retain its real source certification request."
        )
    primary = _cell_result(primary_part)
    certification = primary.certification
    if certification is None:
        raise ValueError(
            "Native restart must retain its real source certification request."
        )
    accepted = {}
    roles = {}
    for name, declaration in declarations.items():
        coefficient_roles = tuple(
            role for role in declaration.state_roles if role.role == "coefficients"
        )
        if (
            declaration.history_policy != "none"
            or len(coefficient_roles) != 1
            or len(declaration.state_roles) != 1
        ):
            raise ValueError(
                "This steady workflow cannot omit stateful/multifield owner obligations."
            )
        role = coefficient_roles[0]
        roles[name] = role.entry_name
        accepted[role.entry_name] = coefficients[name]
    records = {
        "certification_inputs": certification.request,
        "report": certification,
        "associations": primary.associations,
        "registration": connectivity.registration(),
        "generation_sources": {name: plan.source for name, plan in plans.items()},
        "generation_specifications": {
            name: plan.specification for name, plan in plans.items()
        },
        "generation_options": {name: plan.options for name, plan in plans.items()},
        "primary_generation_part": primary_part.name,
        "field_declarations": declarations,
        "accepted_data": {"fields": accepted},
    }

    def restore_at(
        target: Path,
    ) -> tuple[
        OversetConnectivity,
        dict[str, Array],
        dict[str, NativeMeshingPlan],
        dict[str, FiniteElementDiscretization],
    ]:
        from phydrax.meshing._contracts import SurfaceMeshingSpec
        from phydrax.meshing.providers._native import NativeMeshingOptions
        from phydrax.meshing.providers._native_sources import NativePlanarSource

        receipt = write_meshing_source_closure(target, records, limits=limits)
        restored = _archive_mapping(
            read_meshing_source_closure(
                receipt.path,
                expected_content_id=receipt.content_id,
                limits=limits,
            ),
            "source closure",
        )
        registration = restored["registration"]
        if not isinstance(registration, OversetRegistration):
            raise TypeError(
                "Native restart requires the canonical overset registration owner."
            )
        result = registration.prepare()
        result.require_complete()
        if result.connectivity_id != connectivity.connectivity_id:
            raise RuntimeError("Native restart changed accepted registration identity.")
        accepted_data = _archive_mapping(restored["accepted_data"], "accepted data")
        archived_values = _archive_mapping(accepted_data["fields"], "accepted fields")
        sources = _archive_mapping(restored["generation_sources"], "generation sources")
        specifications = _archive_mapping(
            restored["generation_specifications"], "generation specifications"
        )
        options = _archive_mapping(restored["generation_options"], "generation options")
        field_records = _archive_mapping(
            restored["field_declarations"], "field declarations"
        )
        restored_values: dict[str, Array] = {}
        restored_plans: dict[str, NativeMeshingPlan] = {}
        restored_fields: dict[str, FiniteElementDiscretization] = {}
        for part in result.assembly.parts:
            name = part.name
            value = archived_values[roles[name]]
            source, specification, route_options = (
                sources[name],
                specifications[name],
                options[name],
            )
            declaration = field_records[name]
            if not isinstance(value, Array):
                raise TypeError(
                    "Accepted FE coefficients must retain their declared JAX array backend."
                )
            if not isinstance(source, NativePlanarSource) or not isinstance(
                specification, SurfaceMeshingSpec
            ):
                raise TypeError(
                    "Native restart requires the original planar authority and physical request."
                )
            if not isinstance(route_options, NativeMeshingOptions):
                raise TypeError(
                    "Native restart requires the original native generation options."
                )
            if (
                not isinstance(declaration, MeshingFieldDeclaration)
                or declaration.owner != "finite_element"
            ):
                raise TypeError(
                    "Native restart requires the original finite-element field declaration."
                )
            restored_values[name] = value
            restored_plans[name] = NativeMeshingProvider(route_options).plan(
                source,
                specification,
                coordinate_contract=part.coordinate_contract,
            )
            prepared, reconstruction = prepare_meshing_field_owner(
                declaration, _cell_result(part)
            )
            if (
                not isinstance(prepared, FiniteElementDiscretization)
                or reconstruction is not None
            ):
                raise TypeError(
                    "The scalar FE declaration must rebuild its actual finite-element owner."
                )
            restored_fields[name] = prepared
        for name in names:
            np.testing.assert_array_equal(restored_values[name], coefficients[name])
            if restored_plans[name].plan_id != plans[name].plan_id:
                raise RuntimeError(
                    "Native restart changed actual source/request/route identity."
                )
        return result, restored_values, restored_plans, restored_fields

    if path is not None:
        return restore_at(Path(path))
    with TemporaryDirectory(prefix="phydrax-native-source-overset-") as directory:
        return restore_at(Path(directory) / "native-overset.source-closure.zip")


def main() -> None:
    background, background_plan = native_planar_part("background", 3.0)
    body, moving_plan = native_planar_part("moving", 2.25, cavity=0.25)
    generation_plans = {"background": background_plan, "moving": moving_plan}
    mesh = _cell_result(body).mesh
    incidence = mesh.connectivity
    if not isinstance(incidence, PolygonalConnectivity):
        raise TypeError("The moving annulus requires accepted polygonal incidence.")
    boundary = np.asarray(incidence.boundary_edges)
    corners = np.asarray(mesh.coordinates)[np.asarray(incidence.edges)]
    wall = boundary & np.all(np.abs(corners) <= 0.25, axis=(1, 2))
    identifiers = np.asarray(mesh.entity_set(1).entity_ids)
    connectivity = prepare_overset_connectivity(
        phx.meshing.MeshAssembly((background, body)),
        (
            OversetPartSpec(background.name),
            OversetPartSpec(
                body.name,
                wall=body.scope(1, identifiers[wall]),
                boundary=body.scope(1, identifiers[boundary & ~wall]),
            ),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    connectivity.require_complete()
    physical_plans = field_plans(connectivity)
    discretizations = {name: plan.prepare() for name, plan in physical_plans.items()}
    coefficients = {
        name: 5 + jnp.sum(field.dof_maps[0].dof_coordinates, axis=1)
        for name, field in discretizations.items()
    }
    prepared, states = field_entries(
        connectivity, discretizations, coefficients, "initial"
    )
    composition = phx.lifecycle.Composition(
        (*connectivity.composition_entries(), *prepared, *states),
        boundary_id="accepted-poisson-step",
    )
    residuals = []
    for step in range(2):
        moving = connectivity.assembly.part("moving")
        offset = 0.55 if step == 0 else 0.05
        successor, moving_plan = rigid_planar_successor(
            moving,
            np.asarray((offset, 0.0)),
            motion_id=f"native-motion:{step}",
            plan=generation_plans["moving"],
        )
        generation_plans["moving"] = moving_plan
        candidate = connectivity.reregister({moving.name: successor})
        candidate.require_complete()
        physical_plans = field_plans(candidate)
        next_fields = {name: plan.prepare() for name, plan in physical_plans.items()}
        route = prepare_overset_field_transfer(
            candidate, next_fields, "u", previous=connectivity
        )
        receptors = route.apply(coefficients)
        next_values = dict(
            coefficients
        )  # Material-attached active DOFs retain their values.
        for evidence in route.receptors:
            name = evidence.part_name
            next_values[name] = (
                next_values[name].at[evidence.receptor_rows].set(receptors[name])
            )
        prepared, states = field_entries(
            candidate, next_fields, next_values, f"motion-{step}"
        )
        old_states = tuple(
            composition.entry(f"temperature:{name}") for name in sorted(coefficients)
        )
        transport = phx.lifecycle.CompositionTransport(
            "physical-remap",
            tuple(entry.entry_id for entry in old_states),
            states,
            source_structure_ids=tuple(entry.structure_id for entry in old_states),
            route_id=route.transfer_id,
            successful=True,
        )
        motion = prepare_overset_motion_rebind(
            composition,
            connectivity,
            candidate,
            transports=(transport,),
            reprepare=prepared,
        )
        refused = motion.commit(accepted_boundary=False)
        if refused.published or refused.composition is not composition:
            raise RuntimeError(
                "Motion refusal did not preserve the accepted composition."
            )
        receipt = motion.commit(accepted_boundary=True)
        if not receipt.published:
            raise RuntimeError("Complete moving registration was refused.")
        connectivity, composition, discretizations = (
            candidate,
            receipt.composition,
            next_fields,
        )
        coefficients = next_values
        coefficients["moving"], residual = poisson_update(
            discretizations["moving"], coefficients["moving"]
        )
        residuals.append(residual)
        prepared, states = field_entries(
            connectivity, discretizations, coefficients, f"poisson-{step}"
        )
        composition = phx.lifecycle.Composition(
            (*connectivity.composition_entries(), *prepared, *states),
            boundary_id="accepted-poisson-step",
        )

    archive_limits = ArrayArchiveLimits(
        max_members=4096,
        max_manifest_bytes=8 * 1024 * 1024,
        max_central_directory_bytes=4 * 1024 * 1024,
        max_manifest_nesting=64,
    )
    declarations = field_declarations(
        connectivity, physical_plans, discretizations, coefficients
    )
    restored, restored_values, restored_generation, restored_fields = restart_native(
        connectivity,
        coefficients,
        plans=generation_plans,
        declarations=declarations,
        archive_limits=archive_limits,
    )
    replay, residual = poisson_update(
        restored_fields["moving"], restored_values["moving"]
    )
    uninterrupted, _ = poisson_update(discretizations["moving"], coefficients["moving"])
    np.testing.assert_array_equal(replay, uninterrupted)
    from phydrax.meshing.providers._native_sources import NativePlanarSource

    authority_preserved = True
    for name, generation_plan in generation_plans.items():
        original_source = generation_plan.source
        restored_source = restored_generation[name].source
        if not isinstance(original_source, NativePlanarSource) or not isinstance(
            restored_source, NativePlanarSource
        ):
            raise TypeError(
                "Native planar restart must retain its admitted source authority."
            )
        authority_preserved &= restored_source.binding_id == original_source.binding_id
    print(
        {
            "epochs": connectivity.epoch,
            "orphans": connectivity.orphan_count,
            "protected_wall_conflicts": connectivity.conflict_count,
            "background_holes": int(
                connectivity.blanking_of("background")
                .vertex_ids_with(OversetVertexStatus.HOLE)
                .size
            ),
            "poisson_residuals": residuals + [residual],
            "rollback_preserved": True,
            "restart_reproduced": restored.connectivity_id
            == connectivity.connectivity_id,
            "source_authority_preserved": authority_preserved,
            "interpolative_conservative": False,
        }
    )


if __name__ == "__main__":
    main()
