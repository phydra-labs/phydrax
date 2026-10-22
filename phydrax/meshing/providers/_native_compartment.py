#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Admit occupied-image material requests and publish certified native cells."""

from __future__ import annotations

import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from os import getpid
from typing import Any, final

import equinox as eqx
import numpy as np

from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing.providers._native_route_policy import publication_policy

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._coordinate_operands import CoordinateOperandCollector
from ...geometry._compartments import CompartmentMeshingSource
from ...geometry._embedding_state import MeshCertificateLimits
from ...geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding
from ...typing import checked
from .._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from .._certification import MeshCertificationReport
from .._compartments import (
    CompartmentConstruction,
    generate_compartment_volume,
    generate_reconstructed_image_volume,
)
from .._contracts import (
    MeshingProviderInfo,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._controls import FeatureKind
from .._measurements import measure_phase, NativeMeshingPhaseRecorder
from .._result import CellMeshingResult
from .._sizing import UniformSizeControl
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from .._volume_generation import (
    native_volume_checkpoint,
    NativeVolumeSchedule,
)
from ._native_options import NativeMeshingOptions
from ._native_publication import (
    coordinate_budget,
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from ._native_route_checks import simplex_entity_limits
from ._native_support import format_support_issues, SupportIssue
from ._native_volume import volume_compliance


type ImageVolumeSource = CompartmentMeshingSource | LabelFieldVolumeBinding

@dataclass(frozen=True, slots=True)
class _CompartmentOuterProofPayload:
    domain: object
    certificate: object


@dataclass(frozen=True, slots=True)
class _IssuedCompartmentMaterialAuthority:
    source: weakref.ReferenceType[Any]
    specification: weakref.ReferenceType[Any]
    options: weakref.ReferenceType[Any]
    report: weakref.ReferenceType[Any]
    evidence: weakref.ReferenceType[Any]
    root: weakref.ReferenceType[NativeExecutionScope]
    constructors: tuple[weakref.ReferenceType[Any], ...]
    content: str
    outer_domain_id: str
    outer_certificate_id: str
    outer_content: str
    maximum_work_units: int
    maximum_memory_bytes: int
    process_id: int


# Only the actual publication boundary below can issue these process-local
# seals. Neither the table nor its callbacks keep a numerical owner alive.
_issued_compartment_material: dict[tuple[int, ...], _IssuedCompartmentMaterialAuthority] = {}


def _issue_compartment_material_authority(
    source: CompartmentMeshingSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    result: CellMeshingResult,
    construction: CompartmentConstruction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    from ...lifecycle import _meshing_registration as registration

    root = registration._material_root(budget)
    report, evidence = result.certification, result.region_evidence
    outer, outer_certificate = construction.outer_domain, construction.outer_certificate
    if (
        root is None or report is None or evidence is None
        or evidence is not construction.region_evidence
        or outer is None or outer_certificate is None
        or outer_certificate.status != "certified"
        or outer_certificate.domain_id != outer.domain_id
        or evidence.coverage.status != "certified"
        or budget.maximum_work_units != specification.limits.maximum_work_units
        or budget.maximum_memory_bytes != specification.limits.maximum_scratch_bytes
    ):
        raise ValueError("compartment_material.authority requires actual material, outer and publication proofs")
    MeshCertificationReport.require_passed(report)
    outer_certificate.binding.require(result.mesh, result.geometry)
    evidence.require_source(source.compartments)
    evidence.require_current(result.mesh, result.zones, result.patches, geometry=result.geometry)
    content = registration._content(
        source, specification, options, result, budget, memoize=False
    )
    outer_content = registration.stored_record_content_id(
        _CompartmentOuterProofPayload(outer, outer_certificate),
        excluded="", budget=budget, memoize=False,
    )
    constructors = tuple(weakref.ref(function) for function in registration._material_constructors(
        source, specification, options, result
    ))
    key = (id(source), id(report), id(evidence), id(root))

    def forget(reference: weakref.ReferenceType[Any]) -> None:
        entry = _issued_compartment_material.get(key)
        if entry is not None and any(owner is reference for owner in (
            entry.source, entry.specification, entry.options, entry.report,
            entry.evidence, entry.root,
        )):
            del _issued_compartment_material[key]

    _issued_compartment_material[key] = _IssuedCompartmentMaterialAuthority(
        weakref.ref(source, forget), weakref.ref(specification, forget),
        weakref.ref(options, forget), weakref.ref(report, forget),
        weakref.ref(evidence, forget), weakref.ref(root, forget),
        constructors, content, outer.domain_id, outer_certificate.certificate_id,
        outer_content,
        budget.maximum_work_units, budget.maximum_memory_bytes, getpid(),
    )


def _require_issued_compartment_material_authority(
    source: CompartmentMeshingSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    result: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> object:
    from ...lifecycle import _meshing_registration as registration

    root = registration._material_root(budget)
    report, evidence = result.certification, result.region_evidence
    entry = _issued_compartment_material.get((id(source), id(report), id(evidence), id(root)))
    if (
        root is None or entry is None or entry.process_id != getpid()
        or entry.source() is not source or entry.specification() is not specification
        or entry.options() is not options or entry.report() is not report
        or entry.evidence() is not evidence or entry.root() is not root
        or budget.maximum_work_units > entry.maximum_work_units
        or budget.maximum_memory_bytes > entry.maximum_memory_bytes
    ):
        raise ValueError("compartment_material.authority requires its original live paid root and current owners")
    constructors = registration._material_constructors(source, specification, options, result)
    if len(constructors) != len(entry.constructors) or any(
        reference() is not function for reference, function in zip(
            entry.constructors, constructors, strict=True
        )
    ):
        raise ValueError("compartment_material.authority constructors changed since actual production")
    if entry.content != registration._content(
        source, specification, options, result, budget, memoize=False
    ):
        raise ValueError("compartment_material.authority current payload differs from actual production")
    return evidence


@final
class _ImageSourceFacts(StrictModule):
    binding_id: str = eqx.field(static=True)
    regions: tuple[str, ...] = eqx.field(static=True)
    interfaces: tuple[tuple[str, str, str, bool], ...] = eqx.field(static=True)


def _image_source_facts(source: ImageVolumeSource, /) -> _ImageSourceFacts:
    """Read the source's binding, regions, and interface incidence once."""
    match source:
        case CompartmentMeshingSource():
            return _ImageSourceFacts(
                source.source_id,
                tuple(value.compartment_id for value in source.compartments.compartments),
                source.interface_definitions,
            )
        case LabelFieldVolumeBinding():
            return _ImageSourceFacts(
                source.binding_id, source.domain.region_ids, source.interface_definitions
            )
        case _:
            raise TypeError(
                "compartment.source must be CompartmentMeshingSource or "
                f"LabelFieldVolumeBinding; got {type(source).__name__}."
            )


@final
class _SupportPhase(StrictModule):
    issues: tuple[SupportIssue, ...] = eqx.field(static=True)


def _image_request_support(
    source: ImageVolumeSource, specification: VolumeMeshingSpec, /
) -> _SupportPhase:
    """Check the image route's target, all boundary entities, and size request."""
    unsupported: list[SupportIssue] = []
    target = specification.target
    families = target.cell_families
    if (
        target.topological_dimension,
        target.ambient_dimension,
        target.geometry_order,
    ) != (3, 3, 1):
        unsupported.append(
            SupportIssue("geometry", "affine three-dimensional tetrahedral geometry")
        )
    if (
        set((*families.required, *families.preferred)) != {"tetrahedron"}
        or families.allow_mixed
        or families.allowed_transitions
    ):
        unsupported.append(
            SupportIssue("cell-families", "a pure tetrahedral cell-family policy")
        )
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        unsupported.append(
            SupportIssue("fill-strategy", "the simplex volume-fill strategy")
        )
    boundary = specification.boundary_scope
    if (
        boundary.source_id != source.source_id
        or boundary.source_revision != source.source_revision
        or boundary.entity_dimension != 2
        or boundary.entity_set_id != f"{source.source_id}:boundary"
        or not np.array_equal(
            np.asarray(boundary.entity_ids), np.asarray((0,), dtype=np.int64)
        )
    ):
        unsupported.append(
            SupportIssue("boundary-scope", "the complete occupied-image boundary scope")
        )
    sizes = specification.size_controls
    if (
        len(sizes) != 1
        or not isinstance(sizes[0], UniformSizeControl)
        or sizes[0].scope.scope_id != boundary.scope_id
    ):
        unsupported.append(
            SupportIssue("size-control", "one whole-image uniform size control")
        )
    return _SupportPhase(tuple(unsupported))


def _image_region_support(
    source: ImageVolumeSource,
    specification: VolumeMeshingSpec,
    facts: _ImageSourceFacts,
    /,
) -> _SupportPhase:
    """Check region scopes and consistent material semantics in request order."""
    unsupported: list[SupportIssue] = []
    regions = facts.regions
    controlled: set[str] = set()
    for control in specification.region_controls:
        indices = np.asarray(control.scope.entity_ids, dtype=np.int64)
        if control.region_name not in regions or control.region_name in controlled:
            unsupported.append(
                SupportIssue(
                    "region-identity",
                    "unique authoritative compartment region identities",
                )
            )
        elif (
            control.scope.entity_dimension != 3
            or control.scope.entity_set_id != f"{source.source_id}:regions"
            or not np.array_equal(
                indices, np.asarray((regions.index(control.region_name),), dtype=np.int64)
            )
        ):
            unsupported.append(
                SupportIssue("region-scope", "exact source-region control scopes")
            )
        if not control.meshing_enabled:
            unsupported.append(
                SupportIssue("disabled-region", "disabling a required source compartment")
            )
        controlled.add(control.region_name)
    if any(seed.region_name not in regions for seed in specification.region_seeds):
        unsupported.append(
            SupportIssue("region-seed", "region seeds naming authoritative compartments")
        )
    for region in regions:
        semantics = {
            (seed.material_id, seed.role)
            for seed in specification.region_seeds
            if seed.region_name == region
        }
        semantics.update(
            (control.material_id, control.role)
            for control in specification.region_controls
            if control.region_name == region
        )
        if len(semantics) > 1:
            unsupported.append(
                SupportIssue(
                    "region-semantics",
                    "consistent region-control and seed material semantics",
                )
            )
    return _SupportPhase(tuple(unsupported))


def _image_patch_support(
    source: ImageVolumeSource,
    specification: VolumeMeshingSpec,
    facts: _ImageSourceFacts,
    /,
) -> _SupportPhase:
    """Check interface adjacency and exterior patch ownership."""
    unsupported: list[SupportIssue] = []
    regions, interfaces = facts.regions, facts.interfaces
    patch_scopes: set[str] = set()
    for control in specification.patch_controls:
        indices = np.asarray(control.scope.entity_ids, dtype=np.int64)
        if control.scope.scope_id in patch_scopes:
            unsupported.append(
                SupportIssue("duplicate-patch", "one patch control per source interface")
            )
        patch_scopes.add(control.scope.scope_id)
        if control.scope.entity_dimension != 2 or indices.shape != (1,):
            unsupported.append(
                SupportIssue(
                    "patch-dimension",
                    "one codimension-one source entity per patch control",
                )
            )
        elif control.scope.entity_set_id == f"{source.source_id}:interfaces":
            row = indices.item()
            if (
                row < 0
                or row >= len(interfaces)
                or control.adjacent_region_names != tuple(sorted(interfaces[row][1:3]))
            ):
                unsupported.append(
                    SupportIssue(
                        "patch-adjacency", "the exact adjacency of each source interface"
                    )
                )
        elif control.scope.entity_set_id == f"{source.source_id}:boundary":
            if (
                indices.item() != 0
                or len(control.adjacent_region_names) != 1
                or control.adjacent_region_names[0] not in regions
            ):
                unsupported.append(
                    SupportIssue(
                        "exterior-patch", "an occupied-image exterior boundary patch"
                    )
                )
            # A full boundary control cannot describe just one region when
            # several materials touch the exterior; use explicit source facets.
            if len(regions) != 1:
                unsupported.append(
                    SupportIssue(
                        "exterior-regions", "region-resolved exterior patch scopes"
                    )
                )
        else:
            unsupported.append(
                SupportIssue(
                    "patch-scope", "authoritative image boundary/interface patch scopes"
                )
            )
    return _SupportPhase(tuple(unsupported))


def _image_feature_support(
    source: ImageVolumeSource,
    specification: VolumeMeshingSpec,
    facts: _ImageSourceFacts,
    /,
) -> _SupportPhase:
    """Check protected image entity scopes and material-interface identity."""
    unsupported: list[SupportIssue] = []
    interfaces = facts.interfaces
    for feature in specification.protected_features:
        if feature.scope.entity_dimension != 2 or feature.scope.entity_set_id not in (
            f"{source.source_id}:boundary",
            f"{source.source_id}:interfaces",
        ):
            unsupported.append(
                SupportIssue(
                    "feature-scope", "protected occupied-image boundary/interface scopes"
                )
            )
        else:
            identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
            interface_scope = (
                feature.scope.entity_set_id == f"{source.source_id}:interfaces"
            )
            if np.any(identifiers >= (len(interfaces) if interface_scope else 1)) or (
                feature.feature_kind is FeatureKind.MATERIAL_INTERFACE
                and not interface_scope
            ):
                unsupported.append(
                    SupportIssue(
                        "feature-identity",
                        "exact authoritative protected-interface identities",
                    )
                )
    return _SupportPhase(tuple(unsupported))


@checked
def compartment_support_issues(
    source: ImageVolumeSource, specification: VolumeMeshingSpec, /
) -> tuple[SupportIssue, ...]:
    """Return unmet image-route requirements in validation order."""
    facts = _image_source_facts(source)
    issues = (
        *_image_request_support(source, specification).issues,
        *_image_region_support(source, specification, facts).issues,
        *_image_patch_support(source, specification, facts).issues,
        *_image_feature_support(source, specification, facts).issues,
    )
    if specification.periodic_constraints:
        issues += (
            SupportIssue(
                "periodic-recovery",
                "orbit-constrained PLC recovery and boundary-Steiner scheduling",
            ),
        )
    if specification.layer_controls:
        issues += (
            SupportIssue(
                "layer-route", "the native layer/core volume route for layer controls"
            ),
        )
    # Seeds reach constrained recovery unchanged; contradictory seeds retain native evidence.
    return issues


@final
class PreparedCompartmentVolume(StrictModule, NonTrainableState):
    schedule: NativeVolumeSchedule
    source_binding_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: ImageVolumeSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        issues = compartment_support_issues(source, specification)
        if issues:
            raise ValueError(
                "PreparedCompartmentVolume.specification must be supported by the image "
                f"route; got {format_support_issues(issues)}."
            )
        self.schedule = schedule
        self.source_binding_id = _image_source_facts(source).binding_id
        self.source_revision = source.source_revision
        self.specification_id = specification.specification_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "native-compartment-volume",
                "source": self.source_binding_id,
                "revision": source.source_revision,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
            }
        )

    def validate_source_integrity(self) -> None:
        """Authenticate the retained source/request/schedule preparation binding."""
        if not isinstance(self.schedule, NativeVolumeSchedule):
            raise TypeError(
                "PreparedCompartmentVolume.schedule must be NativeVolumeSchedule; got a different restored schedule kind."
            )
        expected = canonical_fingerprint(
            {
                "kind": "native-compartment-volume",
                "source": self.source_binding_id,
                "revision": self.source_revision,
                "specification": self.specification_id,
                "schedule": self.schedule.schedule_id,
            }
        )
        if expected != self.prepared_id:
            raise ValueError(
                "PreparedCompartmentVolume.prepared_id must match its source/request/schedule binding; got stale preparation identity."
            )


@checked
def execute_compartment_route(
    source: ImageVolumeSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedCompartmentVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    options: NativeMeshingOptions,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    """Publish native material cells after independent acceptance."""
    collector = CoordinateOperandCollector(capture_affine_areas=True)
    with collector.live_affine_areas(scope):
        return _execute_compartment_route(
            source, specification, prepared, coordinate_contract, provider, plan_id,
            options=options, record_phase=record_phase, scope=scope, workspace=workspace,
        )


def _execute_compartment_route(
    source: ImageVolumeSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedCompartmentVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    options: NativeMeshingOptions,
    record_phase: NativeMeshingPhaseRecorder | None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    prepared.validate_source_integrity()
    if (
        prepared.source_binding_id != _image_source_facts(source).binding_id
        or prepared.source_revision != source.source_revision
        or prepared.specification_id != specification.specification_id
    ):
        raise ValueError(
            "PreparedCompartmentVolume.source_binding_id must match the image request together with source_revision and specification_id; got a different source or request."
        )
    if coordinate_contract.spatial_id != source.coordinate_contract.spatial_id:
        raise ValueError(
            "execute_compartment_route.coordinate_contract.spatial_id must match the image coordinate contract; got a different spatial identity."
        )
    # The native determinant-floor repair and the publication audit share one policy.
    audit_policy = CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=CellMeshAuditDisposition.REJECT,
    )
    material_limits = MeshCertificateLimits()
    match source:
        case CompartmentMeshingSource():
            construction = generate_compartment_volume(
                source,
                specification,
                prepared.schedule,
                validity_policy=audit_policy.validity_policy,
                input_id=prepared.prepared_id,
                certificate_limits=material_limits,
                record_phase=record_phase,
                scope=scope,
                workspace=workspace,
            )
        case LabelFieldVolumeBinding():
            construction = generate_reconstructed_image_volume(
                source,
                specification,
                prepared.schedule,
                validity_policy=audit_policy.validity_policy,
                input_id=prepared.prepared_id,
                certificate_limits=material_limits,
                record_phase=record_phase,
                scope=scope,
                workspace=workspace,
            )
        case _:
            raise TypeError(
                "execute_compartment_route.source must be CompartmentMeshingSource or LabelFieldVolumeBinding; got an unsupported source kind."
            )
    volume = construction.volume
    mesh = volume.mesh
    with measure_phase(record_phase, "compliance"):
        native_volume_checkpoint(
            specification.limits, MeshingStageKind.SPECIFICATION_COMPLIANCE, scope=scope
        )
        vertices = np.asarray(mesh.coordinates, dtype=np.float64)
        cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
        simplex_entity_limits(
            vertices,
            cells,
            specification.limits,
            MeshingStageKind.VOLUME_FILL,
            cell_kind=mesh.blocks[0].cell_kind,
        )
        native_volume_checkpoint(
            specification.limits, MeshingStageKind.VOLUME_FILL, scope=scope
        )
        compliance = volume_compliance(
            specification, prepared.schedule, volume, vertices, cells, scope=scope
        )
    stages = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(source.source_id,),
            output_ids=(prepared.prepared_id,),
        ),
        *volume.stages,
    )
    publication_budget = coordinate_budget(specification.limits)
    result = publish_native_result(
        mesh,
        coordinate_contract,
        compliance,
        stages,
        provider,
        {
            "kind": "native-image-material-mesh",
            "route": "image_material_tetrahedral",
            "source": source.source_id,
            "revision": source.source_revision,
            "plan": plan_id,
            "specification": specification.specification_id,
            "interpretation": source.interpretation,
            "regions": construction.region_evidence.evidence_id,
        },
        NativeCertificationRequest(
            MeshCertificationSchedule("volume_plc"),
            source.source_id,
            source.source_revision,
            specification.limits,
            domain=volume.domain,
            cell_regions=volume.cell_regions,
        ),
        policy=publication_policy("compartment", audit_policy=audit_policy),
        organization=NativePublicationOrganization(
            patches=volume.patches,
            zones=volume.zones,
            labels=volume.labels,
            associations=volume.associations,
            region_evidence=construction.region_evidence,
        ),
        execution=NativePublicationExecution(scope, workspace, record_phase, publication_budget),
        geometry=volume.geometry,
    )
    native_volume_checkpoint(
        specification.limits, MeshingStageKind.CERTIFICATION, scope=scope
    )
    if isinstance(source, CompartmentMeshingSource):
        from ...lifecycle._meshing_registration import (
            _remember_issued_compartment_material_registration,
        )

        # This is the same budget that paid for final publication, attached to
        # the same actual scope without reserving or opening another request.
        with (
            publication_budget.bind_ticket(scope.ticket_view())
            if publication_budget.ticket.settled
            else nullcontext()
        ):
            _issue_compartment_material_authority(
                source, specification, options, result, construction, budget=publication_budget
            )
            _remember_issued_compartment_material_registration(
                source, specification, options, result, budget=publication_budget
            )
    return result


__all__ = [
    "PreparedCompartmentVolume",
    "compartment_support_issues",
    "execute_compartment_route",
]
