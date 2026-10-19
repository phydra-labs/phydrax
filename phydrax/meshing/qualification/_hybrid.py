# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from contextlib import nullcontext
from typing import final

import numpy as np

from phydrax import StrictModule
from phydrax._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosurePhaseEvidence,
)
from phydrax.geometry import MeshCertificateLimits
from phydrax.geometry._embedding_state import coordinate_certificate_request
from phydrax.lifecycle._meshing_sources import native_family_audit_policy
from phydrax.meshing import (
    CellMeshingResult,
    certify_cell_mesh,
    GeometryAssociation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshCertificationInputs,
    MeshingLimits,
    NativeLayerCoreSource,
    prepare_native_layer_association_transfer,
    VolumeMeshingSpec,
)
from phydrax.meshing._curving import straight_geometry
from phydrax.meshing._lineage import identity_lineage
from phydrax.meshing._measurements import NativeExecutionRecord
from phydrax.meshing._reference_root_composition import affine_reference_root_frame
from phydrax.meshing.providers._native_publication import (
    bind_native_execution_result,
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from phydrax.meshing.providers._native_route_policy import publication_policy
from phydrax.typing import checked


@final
class HybridHighOrderMap(StrictModule):
    """Store a lifted source map and its inherited source certification request."""

    geometry: CellGeometrySpec
    inputs: MeshCertificationInputs
    lifting_evidence: CoordinateEnclosurePhaseEvidence
    execution_evidence: NativeExecutionRecord | None


@checked
def prepare_hybrid_high_order_map(
    generated: CellMeshingResult,
    *,
    degree: int,
    maximum_nodes: int,
    maximum_entries: int,
    budget: CoordinateEnclosureBudget | None = None,
) -> HybridHighOrderMap:
    """Lift under an explicit caller ticket or a separate source-limits operation."""
    certification = generated.certification
    if certification is None:
        raise ValueError(
            "hybrid.generated.certification must retain the source request; got None"
        )
    authority = certification.request
    with (
        NativeExecutionScope(authority.limits.native_limits())
        if budget is None
        else nullcontext(None)
    ) as scope:
        with (
            scope.batch(
                authority.limits.maximum_work_units,
                queries_upper=authority.limits.maximum_distance_evaluations,
                bytes_upper=authority.limits.maximum_scratch_bytes,
            )
            if scope is not None
            else nullcontext(None)
        ) as ticket:
            if budget is None:
                if ticket is None:
                    raise RuntimeError(
                        "hybrid.lifting.ticket must bind the separate lifting operation; got None"
                    )
                budget = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=authority.limits.maximum_work_units,
                    maximum_memory_bytes=authority.limits.maximum_scratch_bytes,
                )
            with (
                coordinate_certificate_request(authority.limits, budget=budget),
                budget.measurement() as measurement,
            ):
                geometry = straight_geometry(
                    generated.mesh,
                    degree,
                    maximum_nodes=maximum_nodes,
                    maximum_entries=maximum_entries,
                    source_geometry=generated.geometry,
                    budget=budget,
                )
    lifting_evidence = CoordinateEnclosurePhaseEvidence(
        work_units=measurement.work_units,
        native_charged_work_units=measurement.native_charged_work_units,
        geometry_queries=measurement.geometry_queries,
        peak_bytes_upper=measurement.peak_bytes_upper,
    )
    execution_evidence = None
    if scope is not None:
        if scope.evidence is None:
            raise RuntimeError(
                "hybrid.lifting.execution must retain the ended lifting operation; got None"
            )
        execution_evidence = NativeExecutionRecord(
            scope.evidence,
            preparation_evidence=generated.execution_evidence,
            owner_id=(
                generated.result_id
                if generated.execution_evidence is None
                else generated.execution_evidence.owner_id
            ),
        )
    inputs = MeshCertificationInputs(
        generated.mesh,
        geometry,
        authority.schedule,
        domain=authority.domain,
        cell_regions=authority.cell_regions,
        source=authority.source,
        fidelity_tolerance=authority.fidelity_tolerance,
        fidelity_sample_order=authority.fidelity_sample_order,
        limits=authority.limits,
        junction_vertices=authority.junction_vertices,
        scoped_fidelity=authority.scoped_fidelity,
    )
    return HybridHighOrderMap(geometry, inputs, lifting_evidence, execution_evidence)


@checked
def qualify_hybrid_high_order_publication(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    generated: CellMeshingResult,
    prepared: HybridHighOrderMap,
) -> CellMeshingResult:
    """Publish the lifted map under the native source request with fresh certificates."""
    authority = prepared.inputs
    if (
        generated.certification is None
        or not generated.certification.passed
        or generated.compliance.specification_id != specification.specification_id
        or generated.trace.binding is None
        or (
            generated.trace.binding.source_id,
            generated.trace.binding.source_revision,
        )
        != (source.source_id, source.source_revision)
    ):
        raise ValueError(
            "hybrid.generated must retain its accepted native source request; "
            f"got result {generated.result_id!r}"
        )
    request = NativeCertificationRequest(
        authority.schedule,
        source.source_id,
        source.source_revision,
        specification.limits,
        domain=authority.domain,
        cell_regions=(
            None
            if authority.cell_regions is None
            else np.asarray(authority.cell_regions, dtype=np.int64)
        ),
        fidelity_source=authority.source,
        fidelity_tolerance=authority.fidelity_tolerance,
        junction_vertices=authority.junction_vertices,
        scoped_fidelity=authority.scoped_fidelity,
    )
    with (
        NativeExecutionScope(specification.limits.native_limits()) as scope,
        NativeHostStorageWorkspace(scope) as workspace,
    ):
        result = publish_native_result(
            generated.mesh,
            generated.coordinate_contract,
            generated.compliance,
            generated.trace.stages,
            generated.provider,
            {
                "kind": "native-hybrid-high-order-publication",
                "source_result_id": generated.result_id,
                "specification_id": specification.specification_id,
                "lifting_operation": (
                    "separate-source-limits-operation"
                    if prepared.execution_evidence is not None
                    else "explicit-caller-ticket"
                ),
                "lifting_work_units": prepared.lifting_evidence.work_units,
                "lifting_native_work_units": prepared.lifting_evidence.native_charged_work_units,
                "lifting_geometry_queries": prepared.lifting_evidence.geometry_queries,
                "lifting_peak_bytes_upper": prepared.lifting_evidence.peak_bytes_upper,
            },
            request,
            policy=publication_policy(
                "layer_core",
                audit_policy=native_family_audit_policy(
                    source, specification, generated.mesh
                ),
                derivative_mode=generated.derivative_mode,
            ),
            organization=NativePublicationOrganization(
                boundary=generated.boundary,
                patches=generated.patches,
                zones=generated.zones,
                labels=generated.labels,
                attributes=generated.attributes,
                associations=generated.associations,
                region_evidence=generated.region_evidence,
                region_boundary_evidence=generated.region_boundary_evidence,
                surface_source=generated.surface_source,
            ),
            execution=NativePublicationExecution(scope, workspace),
            geometry=prepared.geometry,
        )
    if scope.evidence is None:
        raise RuntimeError(
            "hybrid.publication.execution must retain completed evidence; got None"
        )
    return bind_native_execution_result(
        result,
        scope.evidence,
        specification.limits,
        scope.operation_started,
        preparation_evidence=(
            generated.execution_evidence
            if prepared.execution_evidence is None
            else prepared.execution_evidence
        ),
    )


@checked
def prepare_hybrid_adaptation_policy(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    result: CellMeshingResult,
    *,
    limits: MeshingLimits,
) -> MeshAdaptationPolicy:
    """Bind mixed adaptation to the source audit and layer association transfer."""
    return MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_MIXED,
        limits=limits,
        audit_policy=native_family_audit_policy(source, specification, result.mesh),
        association_transfer=prepare_native_layer_association_transfer(
            source, specification, result
        ),
    )


@checked
def qualify_hybrid_reference_cells(mesh: CellMesh) -> None:
    """Require an affine reference-root frame for every published hybrid cell."""
    for block in mesh.blocks:
        for vertices in np.asarray(block.vertices):
            affine_reference_root_frame(
                np.asarray(mesh.coordinates)[vertices], block.cell_kind
            )


@checked
def hybrid_source_coefficients_restored(
    generated: CellMeshingResult,
    target: CellMeshingResult,
    *,
    limits: MeshCertificateLimits,
) -> bool:
    """Compare generated and target source coefficient banks under one certificate allowance."""
    with coordinate_certificate_request(limits) as budget:
        return target.geometry.source_coordinates(
            budget=budget
        ) == generated.geometry.source_coordinates(budget=budget)


@checked
def qualify_hybrid_curved_publication(
    result: CellMeshingResult,
    geometry: CellGeometrySpec,
    association: GeometryAssociation,
    inputs: MeshCertificationInputs,
) -> CellMeshingResult:
    """Certify curved geometry while retaining publication metadata and identity lineage."""
    return certify_cell_mesh(
        result.mesh,
        result.coordinate_contract,
        geometry=geometry,
        patches=result.patches,
        zones=result.zones,
        labels=result.labels,
        attributes=result.attributes,
        associations=(association,),
        region_evidence=result.region_evidence,
        region_boundary_evidence=result.region_boundary_evidence,
        certification_inputs=inputs,
        lineage=identity_lineage(result.mesh, result.mesh),
    )
