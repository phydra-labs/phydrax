#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Occupied-image material admission and canonical native publication."""

from __future__ import annotations

from time import monotonic
from typing import final

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import current_native_execution_budget, NativeExecutionBudget
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._compartments import CompartmentMeshingSource
from ...geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding
from .._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from .._certification import MeshCertificationSchedule
from .._compartments import (
    generate_compartment_volume,
    generate_reconstructed_image_volume,
)
from .._contracts import (
    MeshingDerivativeMode,
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
    _native_volume_operation_started,
    native_volume_checkpoint,
    native_volume_execution_budget,
    NativeVolumeSchedule,
)
from ._native_publication import (
    bind_native_execution_result,
    NativeCertificationRequest,
    publish_native_result,
    simplex_entity_limits,
)
from ._native_volume import _volume_compliance


type ImageVolumeSource = CompartmentMeshingSource | LabelFieldVolumeBinding


def _image_source_binding_id(source: ImageVolumeSource, /) -> str:
    match source:
        case CompartmentMeshingSource():
            return source.source_id
        case LabelFieldVolumeBinding():
            return source.binding_id
        case _:
            raise TypeError(
                "The image route requires an authoritative image volume source."
            )


def _source_regions(source: ImageVolumeSource, /) -> tuple[str, ...]:
    match source:
        case CompartmentMeshingSource():
            return tuple(
                value.compartment_id for value in source.compartments.compartments
            )
        case LabelFieldVolumeBinding():
            return source.domain.region_ids
        case _:
            raise TypeError(
                "The image route requires an authoritative image volume source."
            )


def _source_interfaces(
    source: ImageVolumeSource, /
) -> tuple[tuple[str, str, str, bool], ...]:
    match source:
        case CompartmentMeshingSource():
            return source.interface_definitions
        case LabelFieldVolumeBinding():
            return source.interface_definitions
        case _:
            raise TypeError(
                "The image route requires an authoritative image volume source."
            )


def compartment_support_issues(
    source: ImageVolumeSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    """Identify exact unenforceable requests, never replace their interpretation."""
    unsupported: list[str] = []
    target = specification.target
    families = target.cell_families
    if (
        target.topological_dimension,
        target.ambient_dimension,
        target.geometry_order,
    ) != (3, 3, 1):
        unsupported.append("affine three-dimensional tetrahedral geometry")
    if (
        set((*families.required, *families.preferred)) != {"tetrahedron"}
        or families.allow_mixed
        or families.allowed_transitions
    ):
        unsupported.append("a pure tetrahedral cell-family policy")
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        unsupported.append("the simplex volume-fill strategy")
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
        unsupported.append("the complete occupied-image boundary scope")
    sizes = specification.size_controls
    if (
        len(sizes) != 1
        or not isinstance(sizes[0], UniformSizeControl)
        or sizes[0].scope.scope_id != boundary.scope_id
    ):
        unsupported.append("one whole-image uniform size control")
    regions = _source_regions(source)
    controlled: set[str] = set()
    for control in specification.region_controls:
        indices = np.asarray(control.scope.entity_ids, dtype=np.int64)
        if control.region_name not in regions or control.region_name in controlled:
            unsupported.append("unique authoritative compartment region identities")
        elif (
            control.scope.entity_dimension != 3
            or control.scope.entity_set_id != f"{source.source_id}:regions"
            or not np.array_equal(
                indices, np.asarray((regions.index(control.region_name),), dtype=np.int64)
            )
        ):
            unsupported.append("exact source-region control scopes")
        if not control.meshing_enabled:
            unsupported.append("disabling a required source compartment")
        controlled.add(control.region_name)
    if any(seed.region_name not in regions for seed in specification.region_seeds):
        unsupported.append("region seeds naming authoritative compartments")
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
            unsupported.append("consistent region-control and seed material semantics")
    interfaces = _source_interfaces(source)
    patch_scopes: set[str] = set()
    for control in specification.patch_controls:
        indices = np.asarray(control.scope.entity_ids, dtype=np.int64)
        if control.scope.scope_id in patch_scopes:
            unsupported.append("one patch control per source interface")
        patch_scopes.add(control.scope.scope_id)
        if control.scope.entity_dimension != 2 or indices.shape != (1,):
            unsupported.append("one codimension-one source entity per patch control")
        elif control.scope.entity_set_id == f"{source.source_id}:interfaces":
            row = indices.item()
            if (
                row < 0
                or row >= len(interfaces)
                or control.adjacent_region_names != tuple(sorted(interfaces[row][1:3]))
            ):
                unsupported.append("the exact adjacency of each source interface")
        elif control.scope.entity_set_id == f"{source.source_id}:boundary":
            if (
                indices.item() != 0
                or len(control.adjacent_region_names) != 1
                or control.adjacent_region_names[0] not in regions
            ):
                unsupported.append("an occupied-image exterior boundary patch")
            # A full boundary control cannot describe just one region when
            # several materials touch the exterior; use explicit source facets.
            if len(regions) != 1:
                unsupported.append("region-resolved exterior patch scopes")
        else:
            unsupported.append("authoritative image boundary/interface patch scopes")
    for feature in specification.protected_features:
        if feature.scope.entity_dimension != 2 or feature.scope.entity_set_id not in (
            f"{source.source_id}:boundary",
            f"{source.source_id}:interfaces",
        ):
            unsupported.append("protected occupied-image boundary/interface scopes")
        else:
            identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
            interface_scope = (
                feature.scope.entity_set_id == f"{source.source_id}:interfaces"
            )
            if np.any(identifiers >= (len(interfaces) if interface_scope else 1)) or (
                feature.feature_kind is FeatureKind.MATERIAL_INTERFACE
                and not interface_scope
            ):
                unsupported.append("exact authoritative protected-interface identities")
    if specification.periodic_constraints:
        unsupported.append(
            "orbit-constrained PLC recovery and boundary-Steiner scheduling"
        )
    if specification.layer_controls:
        unsupported.append("the native layer/core volume route for layer controls")
    # Hole seeds and region seeds are handed unchanged to native constrained
    # recovery; contradictory seeds fail with their native region evidence.
    return unsupported


@final
class PreparedCompartmentVolume(StrictModule, NonTrainableState):
    schedule: NativeVolumeSchedule
    source_binding_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: ImageVolumeSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        if not isinstance(source, (CompartmentMeshingSource, LabelFieldVolumeBinding)):
            raise TypeError("source must be an authoritative image volume source.")
        if not isinstance(specification, VolumeMeshingSpec):
            raise TypeError("specification must be VolumeMeshingSpec.")
        if not isinstance(schedule, NativeVolumeSchedule):
            raise TypeError("schedule must be NativeVolumeSchedule.")
        issues = compartment_support_issues(source, specification)
        if issues:
            raise ValueError("Unsupported compartment request: " + "; ".join(issues))
        self.schedule = schedule
        self.source_binding_id = _image_source_binding_id(source)
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
            raise TypeError("A prepared material route requires its native schedule.")
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
            raise ValueError("Prepared material source/request identity is stale.")


def execute_compartment_route(
    source: ImageVolumeSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedCompartmentVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    execution_budget: NativeExecutionBudget | None = None,
    operation_started: float | None = None,
) -> CellMeshingResult:
    """Publish native material cells only after exact independent acceptance."""
    prepared.validate_source_integrity()
    if (
        prepared.source_binding_id != _image_source_binding_id(source)
        or prepared.source_revision != source.source_revision
        or prepared.specification_id != specification.specification_id
    ):
        raise ValueError(
            "The prepared material route binds a different source or request."
        )
    if coordinate_contract.spatial_id != source.coordinate_contract.spatial_id:
        raise ValueError(
            "The request and occupied image require one coordinate contract."
        )
    active = current_native_execution_budget()
    if execution_budget is not None and execution_budget is not active:
        raise RuntimeError("Image publication must borrow the actual active budget.")
    started = _native_volume_operation_started(operation_started)
    if active is None and started is None:
        started = monotonic()
    with native_volume_execution_budget(
        specification.limits,
        operation_started=started,
    ) as execution:
        result = _execute_compartment_route_bound(
            source,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            record_phase=record_phase,
            execution_budget=execution,
            operation_started=started,
        )
    if active is not None:
        return result
    evidence = execution.evidence
    if evidence is None:
        raise RuntimeError(
            "Image publication lost its completed original execution evidence."
        )
    if started is None:
        raise RuntimeError("Owning image publication lost its original host clock.")
    return bind_native_execution_result(result, evidence, specification.limits, started)


def _execute_compartment_route_bound(
    source: ImageVolumeSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedCompartmentVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None,
    execution_budget: NativeExecutionBudget,
    operation_started: float | None,
) -> CellMeshingResult:
    started = operation_started
    # The native determinant-floor repair and the publication audit share one policy.
    audit_policy = CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=CellMeshAuditDisposition.REJECT,
    )
    match source:
        case CompartmentMeshingSource():
            construction = generate_compartment_volume(
                source,
                specification,
                prepared.schedule,
                validity_policy=audit_policy.validity_policy,
                input_id=prepared.prepared_id,
                record_phase=record_phase,
                operation_started=started,
                execution_budget=execution_budget,
            )
        case LabelFieldVolumeBinding():
            construction = generate_reconstructed_image_volume(
                source,
                specification,
                prepared.schedule,
                validity_policy=audit_policy.validity_policy,
                input_id=prepared.prepared_id,
                record_phase=record_phase,
                operation_started=started,
                execution_budget=execution_budget,
            )
        case _:
            raise TypeError(
                "The image route requires an authoritative image volume source."
            )
    volume = construction.volume
    mesh = volume.mesh
    with measure_phase(record_phase, "compliance"):
        native_volume_checkpoint(
            specification.limits, MeshingStageKind.SPECIFICATION_COMPLIANCE
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
            specification.limits, MeshingStageKind.VOLUME_FILL, operation_started=started
        )
        compliance = _volume_compliance(
            specification, prepared.schedule, volume, vertices, cells
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
        audit_policy=audit_policy,
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity_entries",
            "data_bytes",
            "work_units:native_and_metered_host",
            "wall_seconds:native_and_host_boundaries",
            "cavity_cells:native",
            "geometry_queries:native_and_source_batches",
            "scratch_bytes:managed_native_and_bounded_host",
        ),
        unenforced_limits=(
            "scratch_bytes:unmanaged_host_device_compiler",
            "work_units:unmetered_host_device_compiler",
            "wall_seconds:nonpreemptible_host_device_compiler",
        ),
        patches=volume.patches,
        zones=volume.zones,
        labels=volume.labels,
        associations=volume.associations,
        region_evidence=construction.region_evidence,
        record_phase=record_phase,
        operation_started=started,
    )
    native_volume_checkpoint(
        specification.limits, MeshingStageKind.CERTIFICATION, operation_started=started
    )
    return result


__all__ = [
    "PreparedCompartmentVolume",
    "compartment_support_issues",
    "execute_compartment_route",
]
