#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._contracts import MeshingFailureCategory


class MeshingStageKind(StrEnum):
    SOURCE_INSPECTION = "source_inspection"
    SCOPE_RESOLUTION = "scope_resolution"
    CONTROL_RESOLUTION = "control_resolution"
    SIZE_FIELD_RESOLUTION = "size_field_resolution"
    FEATURE_DISCOVERY = "feature_discovery"
    TOPOLOGY_REPAIR = "topology_repair"
    SURFACE_MESHING = "surface_meshing"
    LAYER_GENERATION = "layer_generation"
    VOLUME_FILL = "volume_fill"
    OPTIMIZATION = "optimization"
    CANONICALIZATION = "canonicalization"
    GEOMETRY_ASSOCIATION = "geometry_association"
    TOPOLOGY_AUDIT = "topology_audit"
    GEOMETRY_AUDIT = "geometry_audit"
    QUALITY_EVALUATION = "quality_evaluation"
    SPECIFICATION_COMPLIANCE = "specification_compliance"
    LINEAGE_CONSTRUCTION = "lineage_construction"


class MeshingStageStatus(StrEnum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class MeshingDiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class MeshingDiagnostic(StrictModule, NonTrainableState):
    severity: MeshingDiagnosticSeverity = eqx.field(static=True)
    message: str = eqx.field(static=True)
    provider_code: str = eqx.field(static=True)
    failure_category: MeshingFailureCategory | None = eqx.field(static=True)
    entity_ids: tuple[int, ...] = eqx.field(static=True)
    locations: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    diagnostic_id: str = eqx.field(static=True)

    def __init__(
        self,
        severity: MeshingDiagnosticSeverity,
        message: str,
        /,
        *,
        provider_code: str = "",
        failure_category: MeshingFailureCategory | None = None,
        entity_ids: tuple[int, ...] = (),
        locations: tuple[tuple[float, ...], ...] = (),
    ):
        if not isinstance(severity, MeshingDiagnosticSeverity):
            raise TypeError("severity must be MeshingDiagnosticSeverity.")
        if failure_category is not None and not isinstance(
            failure_category, MeshingFailureCategory
        ):
            raise TypeError("failure_category must be MeshingFailureCategory or None.")
        text = str(message).strip()
        if not text:
            raise ValueError("Meshing diagnostic message must be non-empty.")
        identifiers = tuple(int(value) for value in entity_ids)
        points = tuple(
            tuple(float(component) for component in point) for point in locations
        )
        self.severity = severity
        self.message = text
        self.provider_code = str(provider_code)
        self.failure_category = failure_category
        self.entity_ids = identifiers
        self.locations = points
        self.diagnostic_id = canonical_fingerprint(
            {
                "kind": "meshing-diagnostic",
                "severity": severity.value,
                "message": text,
                "provider_code": str(provider_code),
                "failure_category": (
                    None if failure_category is None else failure_category.value
                ),
                "entity_ids": identifiers,
                "locations": points,
            }
        )


class MeshingStageReport(StrictModule, NonTrainableState):
    stage: MeshingStageKind = eqx.field(static=True)
    status: MeshingStageStatus = eqx.field(static=True)
    input_ids: tuple[str, ...] = eqx.field(static=True)
    output_ids: tuple[str, ...] = eqx.field(static=True)
    diagnostics: tuple[MeshingDiagnostic, ...]
    created_count: int = eqx.field(static=True)
    modified_count: int = eqx.field(static=True)
    deleted_count: int = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        stage: MeshingStageKind,
        status: MeshingStageStatus,
        /,
        *,
        input_ids: tuple[str, ...] = (),
        output_ids: tuple[str, ...] = (),
        diagnostics: tuple[MeshingDiagnostic, ...] = (),
        created_count: int = 0,
        modified_count: int = 0,
        deleted_count: int = 0,
    ):
        if not isinstance(stage, MeshingStageKind):
            raise TypeError("stage must be MeshingStageKind.")
        if not isinstance(status, MeshingStageStatus):
            raise TypeError("status must be MeshingStageStatus.")
        if not all(isinstance(value, MeshingDiagnostic) for value in diagnostics):
            raise TypeError("diagnostics must contain MeshingDiagnostic values.")
        counts = (int(created_count), int(modified_count), int(deleted_count))
        if any(value < 0 for value in counts):
            raise ValueError("Meshing stage entity counts must be non-negative.")
        if status is MeshingStageStatus.FAILED and not any(
            value.severity is MeshingDiagnosticSeverity.ERROR for value in diagnostics
        ):
            raise ValueError("Failed meshing stages require one error diagnostic.")
        self.stage = stage
        self.status = status
        self.input_ids = tuple(str(value) for value in input_ids)
        self.output_ids = tuple(str(value) for value in output_ids)
        self.diagnostics = tuple(diagnostics)
        self.created_count, self.modified_count, self.deleted_count = counts
        self.report_id = canonical_fingerprint(
            {
                "kind": "meshing-stage-report",
                "stage": stage.value,
                "status": status.value,
                "input_ids": self.input_ids,
                "output_ids": self.output_ids,
                "diagnostics": [value.diagnostic_id for value in diagnostics],
                "counts": counts,
            }
        )


class MeshingTrace(StrictModule, NonTrainableState):
    stages: tuple[MeshingStageReport, ...]
    successful: bool = eqx.field(static=True)
    trace_id: str = eqx.field(static=True)

    def __init__(self, stages: tuple[MeshingStageReport, ...], /):
        if not stages or not all(
            isinstance(stage, MeshingStageReport) for stage in stages
        ):
            raise ValueError("Meshing traces require at least one stage report.")
        failed = tuple(
            index
            for index, stage in enumerate(stages)
            if stage.status is MeshingStageStatus.FAILED
        )
        if failed and failed != (len(stages) - 1,):
            raise ValueError("A failed meshing stage must terminate the trace.")
        self.stages = tuple(stages)
        self.successful = not failed
        self.trace_id = canonical_fingerprint(
            {
                "kind": "meshing-trace",
                "stages": [stage.report_id for stage in stages],
            }
        )


__all__ = [
    "MeshingDiagnostic",
    "MeshingDiagnosticSeverity",
    "MeshingStageKind",
    "MeshingStageReport",
    "MeshingStageStatus",
    "MeshingTrace",
]
