#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared source admission and publication phases for native layer/core routes."""

from __future__ import annotations

from contextlib import ExitStack
from typing import final, Protocol

import phydrax.meshing.providers._native_sources as sources_owner

from .._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..typing import checked
from . import (
    _layer_core_periodic as periodic_owner,
    _layer_core_resources as resources_owner,
)
from ._contracts import MeshingProviderInfo, VolumeMeshingSpec
from ._measurements import NativeMeshingPhaseRecorder
from ._native_preparation import NativePreparedOwner
from ._result import CellMeshingResult, MeshingComplianceReport
from ._trace import MeshingStageKind, MeshingStageReport
from ._volume_generation import NativeVolumeSchedule
from .providers._native_publication import (
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from .providers._native_route_checks import check_deadline
from .providers._native_route_policy import NativePublicationPolicy, publication_policy


class PreparedLayerIdentity(NativePreparedOwner, Protocol):
    """The live preparation ledger owns admission of this object identity."""

    @property
    def source_binding_id(self) -> str: ...

    @property
    def specification_id(self) -> str: ...


class PreparedLayerReference(Protocol):
    @property
    def core_specification(self) -> VolumeMeshingSpec: ...


class PreparedMappedLayer(Protocol):
    """Reference fill inputs consumed without a dependency on the executor owner."""

    @property
    def reference_prepared(self) -> PreparedLayerReference | None: ...

    @property
    def source_work_units(self) -> int: ...

    @property
    def schedule(self) -> NativeVolumeSchedule: ...


@final
class LayerRoutePreparation(StrictModule, NonTrainableState):
    """Periodic request evidence and one policy shared by core repair and audit."""

    work: resources_owner.LayerCoreSourceWork
    periodic_requested: tuple[tuple[str, float], ...]
    periodic_achieved: tuple[tuple[str, float], ...]
    policy: NativePublicationPolicy
    started: float


@final
class LayerCorePublication(StrictModule, NonTrainableState):
    """Consumer evidence ready for the shared audit/certification boundary."""

    mesh: CellMesh
    compliance: MeshingComplianceReport
    stages: tuple[MeshingStageReport, ...]
    provenance: dict[str, str]
    certification: NativeCertificationRequest
    organization: NativePublicationOrganization
    geometry: CellGeometrySpec | None = None


@checked
def prepare_layer_route(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    source_work_units: int,
    scope: NativeExecutionScope,
    resources: ExitStack,
    /,
    *,
    started: float,
) -> LayerRoutePreparation:
    """Settle source seam evidence before native fill begins.

    Both affine and mapped publication require full source association and
    reject a nonwatertight boundary. Native determinant-floor repair and final
    audit consume this same policy.
    """
    limits = specification.limits
    work = resources_owner.LayerCoreSourceWork(
        limits.maximum_work_units,
        source_work_units,
        ticket=resources.enter_context(
            scope.batch(
                min(limits.maximum_work_units, scope.remaining().remaining_work_units)
                if specification.periodic_constraints
                else resources_owner.layer_source_work_upper(source, specification),
                bytes_upper=min(
                    limits.maximum_scratch_bytes,
                    scope.remaining().remaining_managed_bytes,
                ),
            )
        ),
    )
    budget = CoordinateEnclosureBudget(
        work.ticket,
        maximum_work_units=limits.maximum_work_units,
        maximum_memory_bytes=limits.maximum_scratch_bytes,
    )
    resources.enter_context(budget.activate())
    requested, achieved = periodic_owner.core_periodic_constraint_evidence(
        source, specification, work=work, budget=budget
    )
    resources.close()
    policy = publication_policy("layer_core")
    return LayerRoutePreparation(work, requested, achieved, policy, started)


@checked
def publish_layer_route(
    publication: LayerCorePublication,
    preparation: LayerRoutePreparation,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Publish the completed carrier and preserve the final deadline checkpoint."""
    result = publish_native_result(
        publication.mesh,
        coordinate_contract,
        publication.compliance,
        publication.stages,
        provider,
        publication.provenance,
        publication.certification,
        policy=preparation.policy,
        organization=publication.organization,
        execution=NativePublicationExecution(scope, workspace, record_phase),
        geometry=publication.geometry,
    )
    check_deadline(
        preparation.started,
        publication.certification.limits,
        MeshingStageKind.CERTIFICATION,
    )
    return result
