# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from typing import final

import numpy as np

from phydrax import StrictModule
from phydrax._meshcore import (
    current_native_execution_scope,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosurePhaseEvidence,
)
from phydrax.geometry import MeshCertificateLimits
from phydrax.geometry._approximate_source_coverage import (
    ApproximateSourceCoverageCertificate,
    certify_approximate_source_coverage,
)
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._domain_planar_boundary import CadSourceDomain
from phydrax.geometry._embedding_state import coordinate_certificate_request
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._source_fidelity import require_issued_cad_boundary_closure
from phydrax.lifecycle._meshing_sources import (
    meshing_source_owner_reconstruction_scope,
    native_family_audit_policy,
)
from phydrax.meshing import (
    CellMeshingResult,
    GeometryAssociation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshCertificationInputs,
    MeshingLimits,
    NativeLayerCoreSource,
    prepare_native_layer_association_transfer,
    VolumeMeshingSpec,
)
from phydrax.meshing._audit import CellMeshAuditPolicy, required_audit_policy
from phydrax.meshing._certification import MeshCertificationPreparedEvidence
from phydrax.meshing._contracts import MeshingDerivativeMode
from phydrax.meshing._curving import CurvedGeometryEvidence, straight_geometry
from phydrax.meshing._measurements import NativeExecutionRecord
from phydrax.meshing._native_preparation import (
    native_preparation_ledger,
    stage_native_completed_source_issue,
    stage_native_source_publication,
)
from phydrax.meshing._reference_root_composition import affine_reference_root_frame
from phydrax.meshing._volume_generation import native_volume_source_execution_budget
from phydrax.meshing.providers._native_publication import (
    bind_native_execution_result,
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from phydrax.meshing.providers._native_route_policy import (
    NativePublicationPolicy,
    publication_policy,
)
from phydrax.typing import checked


def _publish_in_request(
    result: CellMeshingResult,
    provenance: dict[str, object],
    certification: NativeCertificationRequest,
    policy: NativePublicationPolicy,
    organization: NativePublicationOrganization,
    geometry: CellGeometrySpec,
    preparation_evidence: NativeExecutionRecord | None,
    /,
    *,
    prepare: Callable[[CoordinateEnclosureBudget], MeshCertificationPreparedEvidence] | None = None,
    execution: NativePublicationExecution | None = None,
    history_evidence: tuple[NativeExecutionRecord, ...] = (),
) -> CellMeshingResult:
    """Publish under one request scope whose coordinate ledger and caches it owns.

    The ledger opens on a settled ticket of the request scope; publication keeps
    its cache bytes in that scope's retained storage, so every audit and
    certificate phase draws from ``certification.limits``. ``prepare`` binds
    already-earned volume premises inside this same scope before acceptance.
    """
    limits = certification.limits
    if execution is not None:
        scope, workspace, budget = execution.scope, execution.workspace, execution.budget
        if (
            scope is not current_native_execution_scope()
            or budget is None
            or preparation_evidence is not None
        ):
            raise ValueError(
                "hybrid.publication.execution must own its active request and already-accounted preparation budget"
            )
        for history in history_evidence:
            native_preparation_ledger(scope).record_history(scope, history)
        with (
            budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size))
            if budget.storage_ticket is None else nullcontext(),
            meshing_source_owner_reconstruction_scope(budget=budget),
        ):
            if prepare is not None:
                with budget.bind_ticket(scope.ticket_view()):
                    certification = replace(certification, prepared=prepare(budget))
            return publish_native_result(
                result.mesh,
                result.coordinate_contract,
                result.compliance,
                result.trace.stages,
                result.provider,
                provenance,
                certification,
                policy=policy,
                organization=organization,
                execution=execution,
                geometry=geometry,
            )
    preparation_owner = result.result_id
    if preparation_evidence is not None:
        if preparation_evidence.owner_id is None:
            raise ValueError("hybrid.publication.preparation must retain its actual request owner")
        preparation_owner = preparation_evidence.owner_id
    with (
        NativeExecutionScope(limits.native_limits()) as scope,
        NativeHostStorageWorkspace(scope) as workspace,
        native_volume_source_execution_budget(
            limits, preparation_evidence, preparation_owner, scope=scope
        ) if preparation_evidence is not None else nullcontext(),
    ):
        with scope.batch(0) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
        published = _publish_in_request(
            result, provenance, certification, policy, organization, geometry, None,
            prepare=prepare,
            execution=NativePublicationExecution(scope, workspace, None, budget),
            history_evidence=history_evidence,
        )
        overlap = (
            native_preparation_ledger(scope).overlap(scope, preparation_evidence)
            if preparation_evidence is not None else None
        )
        publication_owner = (
            preparation_evidence.owner_id
            if preparation_evidence is not None
            else published.execution_evidence.owner_id
            if published.execution_evidence is not None
            else result.execution_evidence.owner_id
            if result.execution_evidence is not None
            else published.result_id
        )
        with budget.bind_ticket(scope.ticket_view()):
            if scope.parent is not None:
                stage_native_source_publication(scope, published, budget=budget)
            issue = (
                stage_native_completed_source_issue(
                    scope, published,
                    owner_id=publication_owner,
                    overlap=overlap, budget=budget,
                )
                if scope.parent is None else None
            )
    if scope.evidence is None:
        raise RuntimeError(
            "hybrid.publication.execution must retain completed evidence; got None"
        )
    bound = bind_native_execution_result(
        published,
        scope.evidence if overlap is None else overlap.normalize(scope.evidence),
        limits,
        scope.operation_started,
        preparation_evidence=preparation_evidence,
        history_evidence=history_evidence,
        owner_id=publication_owner
        if preparation_evidence is None and published.execution_evidence is None else None,
    )
    if issue is not None:
        issue.complete(bound)
    if scope.parent is not None:
        if bound.execution_evidence is None:
            raise RuntimeError("Hybrid publication must retain its ended execution receipt.")
        native_preparation_ledger(scope.parent).remember_receipt(
            scope.parent, bound.execution_evidence
        )
        stage_native_source_publication(scope.parent, bound)
    return bound


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
                    mapped_source_evidence=authority.mapped_source_evidence,
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
            history_evidence=()
            if generated.execution_evidence is None
            else (generated.execution_evidence,),
            owner_id=(
                generated.result_id
                if generated.execution_evidence is None
                else generated.execution_evidence.owner_id
            ),
        )
    return HybridHighOrderMap(geometry, inputs, lifting_evidence, execution_evidence)


@checked
def qualify_hybrid_high_order_publication(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    generated: CellMeshingResult,
    prepared: HybridHighOrderMap,
    *,
    execution: NativePublicationExecution | None = None,
) -> CellMeshingResult:
    """Publish the lifted map under the native source request with fresh certificates."""
    authority = prepared.inputs
    if prepared.execution_evidence is None and execution is None:
        raise ValueError(
            "hybrid.lifting.explicit_ticket must publish within its actual request execution"
        )
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
    # The lifted map republishes the plan-generated result; its provenance keeps
    # that source-bound generation plan identity for archive authentication.
    generated_provenance = json.loads(generated.provenance.content_json)
    plan_id = (
        dict(generated_provenance["items"]).get("plan")
        if generated_provenance.get("kind") == "mapping"
        else None
    )
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError(
            "hybrid.generated.provenance must retain its source-bound generation plan; "
            f"got {plan_id!r}"
        )
    return _publish_in_request(
        generated,
        {
            "kind": "native-hybrid-high-order-publication",
            "plan": plan_id,
            "source_result_id": generated.result_id,
            "specification": specification.specification_id,
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
        publication_policy(
            "layer_core",
            audit_policy=native_family_audit_policy(
                source, specification, generated.mesh
            ),
            derivative_mode=generated.derivative_mode,
        ),
        NativePublicationOrganization(
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
        prepared.geometry,
        prepared.execution_evidence,
        execution=execution,
        history_evidence=()
        if prepared.execution_evidence is not None or generated.execution_evidence is None
        else (generated.execution_evidence,),
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
    *,
    limits: MeshingLimits,
    audit_policy: CellMeshAuditPolicy,
    curving: CurvedGeometryEvidence | None = None,
    execution: NativePublicationExecution | None = None,
) -> CellMeshingResult:
    """Publish curved geometry under ``limits`` with the inputs' explicit certificate caps.

    The inputs' certificate limits are used unchanged; certification runs on
    tickets of the outer request ``limits``. The audit executes the schedule's
    required checks over the caller's explicit source-family policy. Accepted
    ``curving`` evidence for this exact geometry
    supplies its earned validity and global embedding as prepared volume
    premises. CAD publication also retains its independently certified whole
    source-region coverage and fidelity; topology and quality are still audited.
    An explicit ``execution`` keeps preparation and publication on the caller's
    actual ledger; that caller closes and binds the complete request afterward.
    """
    source = inputs.source
    if source is None:
        raise ValueError(
            "hybrid.curved.inputs.source must bind the source query; got None"
        )
    cell_regions = (
        None
        if inputs.cell_regions is None
        else np.asarray(inputs.cell_regions, dtype=np.int64)
    )
    certification = NativeCertificationRequest(
        inputs.schedule,
        source.source_id,
        source.source_revision,
        limits,
        domain=inputs.domain,
        cell_regions=cell_regions,
        fidelity_source=source,
        fidelity_tolerance=inputs.fidelity_tolerance,
        junction_vertices=inputs.junction_vertices,
        scoped_fidelity=inputs.scoped_fidelity,
        certificate_limits=inputs.limits,
    )
    audit_policy = required_audit_policy(
        audit_policy, certification.schedule.required_audit_checks
    )
    prepare: Callable[[CoordinateEnclosureBudget], MeshCertificationPreparedEvidence] | None = None
    domain = inputs.domain
    # The publication audit's local policy remains authoritative. A CAD boundary
    # theorem is independent of its local Bernstein cap, so a policy mismatch
    # re-earns local validity and embedding instead of discarding that theorem.
    reusable_cad = (
        curving is not None
        and isinstance(domain, CadSourceDomain)
        and isinstance(source, MeshingDomainBoundarySource)
        and curving.fidelity is not None
        and curving.fidelity.cad_closure is not None
        and isinstance(curving.coverage, ApproximateSourceCoverageCertificate)
    )
    if (
        curving is not None
        and curving.accepted
        and (
            curving.certificate.policy_id == audit_policy.validity_policy.policy_id
            or reusable_cad
        )
        and cell_regions is not None
        and isinstance(domain, (PiecewiseLinearDomain, MappedReferenceDomain, CadSourceDomain))
        and inputs.junction_vertices is None
    ):
        evidence, regions = curving, cell_regions

        def prepare_curving(budget: CoordinateEnclosureBudget) -> MeshCertificationPreparedEvidence:
            validity, embedding = evidence.certificate, evidence.embedding
            coverage = (
                evidence.coverage
                if isinstance(evidence.coverage, ApproximateSourceCoverageCertificate)
                else None
            )
            if reusable_cad:
                fidelity = evidence.fidelity
                if (
                    fidelity is None or fidelity.cad_closure is None
                    or not isinstance(source, MeshingDomainBoundarySource)
                    or not isinstance(domain, CadSourceDomain)
                ):
                    raise ValueError("CAD curving publication must retain its issued source premises")
                require_issued_cad_boundary_closure(
                    result.mesh, geometry, fidelity.cad_closure, inputs.limits,
                    source=source, budget=budget,
                )
                if validity.policy_id != audit_policy.validity_policy.policy_id:
                    validity = certify_cell_geometry_validity(
                        geometry, mesh=result.mesh,
                        policy=audit_policy.validity_policy, budget=budget,
                    )
                    embedding = certify_global_embedding(
                        result.mesh, geometry, validity, limits=inputs.limits,
                        cad_boundary=fidelity.cad_closure, budget=budget,
                    )
                    coverage = certify_approximate_source_coverage(
                        result.mesh, geometry, domain, regions,
                        embedding=embedding, fidelity=fidelity, source=source,
                        limits=inputs.limits, budget=budget,
                    )
            return MeshCertificationPreparedEvidence(
                result.mesh,
                geometry,
                validity,
                schedule=inputs.schedule,
                domain=domain,
                cell_regions=regions,
                embedding=embedding,
                coverage=coverage,
                limits=inputs.limits,
                source=source,
                fidelity_tolerance=inputs.fidelity_tolerance,
                fidelity_sample_order=inputs.fidelity_sample_order,
                scoped_fidelity=inputs.scoped_fidelity,
                budget=budget,
            )

        prepare = prepare_curving
    provenance: dict[str, object] = {
        "kind": "native-hybrid-curved-publication",
        "source_result_id": result.result_id,
        "certificate_limits_id": inputs.limits.limits_id,
    }
    original = json.loads(result.provenance.content_json)
    if original.get("kind") == "mapping":
        original_items = dict(original["items"])
        # Curving changes geometry, not the source-bound generation request.
        for name in ("plan", "specification"):
            if name in original_items:
                provenance[name] = original_items[name]
    return _publish_in_request(
        result,
        provenance,
        certification,
        publication_policy(
            "layer_core",
            audit_policy=audit_policy,
            derivative_mode=MeshingDerivativeMode.FIXED_TOPOLOGY_EXACT,
        ),
        NativePublicationOrganization(
            patches=result.patches,
            zones=result.zones,
            labels=result.labels,
            attributes=result.attributes,
            associations=(association,),
            region_evidence=result.region_evidence,
            region_boundary_evidence=result.region_boundary_evidence,
        ),
        geometry,
        None,
        prepare=prepare,
        execution=execution,
        history_evidence=()
        if result.execution_evidence is None else (result.execution_evidence,),
    )
