#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ._contracts import MeshingFailureCategory


class MeshingStageKind(StrEnum):
    SOURCE_INSPECTION = "source_inspection"
    SCOPE_RESOLUTION = "scope_resolution"
    CONTROL_RESOLUTION = "control_resolution"
    SIZE_FIELD_RESOLUTION = "size_field_resolution"
    FEATURE_DISCOVERY = "feature_discovery"
    TOPOLOGY_REPAIR = "topology_repair"
    CURVE_MESHING = "curve_meshing"
    SURFACE_MESHING = "surface_meshing"
    LAYER_GENERATION = "layer_generation"
    VOLUME_FILL = "volume_fill"
    OPTIMIZATION = "optimization"
    CANONICALIZATION = "canonicalization"
    GEOMETRY_ASSOCIATION = "geometry_association"
    TOPOLOGY_AUDIT = "topology_audit"
    GEOMETRY_AUDIT = "geometry_audit"
    QUALITY_EVALUATION = "quality_evaluation"
    CERTIFICATION = "certification"
    SPECIFICATION_COMPLIANCE = "specification_compliance"
    LINEAGE_CONSTRUCTION = "lineage_construction"


class MeshingStageStatus(StrEnum):
    """Stage outcome; FAILED and UNRESOLVED are terminal and unsuccessful.

    UNRESOLVED marks a mandatory check that could not be decided (work budget,
    undecided predicate, unsupported geometry) as distinct from a disproof.
    """

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    UNRESOLVED = "unresolved"


class MeshingDiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class MeshingDiagnostic(StrictModule, NonTrainableState):
    """One stage diagnostic with its failing entities and measured quantities.

    ``quantities`` lists ``(name, requested, achieved)`` for every checked
    request the diagnostic concerns.
    """

    severity: MeshingDiagnosticSeverity = eqx.field(static=True)
    message: str = eqx.field(static=True)
    provider_code: str = eqx.field(static=True)
    failure_category: MeshingFailureCategory | None = eqx.field(static=True)
    entity_ids: tuple[int, ...] = eqx.field(static=True)
    locations: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    quantities: tuple[tuple[str, float, float], ...] = eqx.field(static=True)
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
        quantities: tuple[tuple[str, float, float], ...] = (),
    ) -> None:
        if not isinstance(severity, MeshingDiagnosticSeverity):
            raise TypeError("severity must be MeshingDiagnosticSeverity.")
        if failure_category is not None and not isinstance(
            failure_category, MeshingFailureCategory
        ):
            raise TypeError("failure_category must be MeshingFailureCategory or None.")
        text = str(message).strip()
        if not text:
            raise ValueError("Meshing diagnostic message must be non-empty.")
        identifiers = tuple(entity_ids)
        points = tuple(
            tuple(float(component) for component in point) for point in locations
        )
        measured = tuple(
            (str(name), float(requested), float(achieved))
            for name, requested, achieved in quantities
        )
        if any(not name for name, _, _ in measured):
            raise ValueError("Diagnostic quantities require non-empty names.")
        self.severity = severity
        self.message = text
        self.provider_code = str(provider_code)
        self.failure_category = failure_category
        self.entity_ids = identifiers
        self.locations = points
        self.quantities = measured
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
                "quantities": measured,
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
    ) -> None:
        if not isinstance(stage, MeshingStageKind):
            raise TypeError("stage must be MeshingStageKind.")
        if not isinstance(status, MeshingStageStatus):
            raise TypeError("status must be MeshingStageStatus.")
        if not all(isinstance(value, MeshingDiagnostic) for value in diagnostics):
            raise TypeError("diagnostics must contain MeshingDiagnostic values.")
        counts = (int(created_count), int(modified_count), int(deleted_count))
        if any(value < 0 for value in counts):
            raise ValueError("Meshing stage entity counts must be non-negative.")
        if status in (
            MeshingStageStatus.FAILED,
            MeshingStageStatus.UNRESOLVED,
        ) and not any(
            value.severity is MeshingDiagnosticSeverity.ERROR for value in diagnostics
        ):
            raise ValueError(
                "Failed or unresolved meshing stages require one error diagnostic."
            )
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


class MeshingEvidenceBinding(StrictModule, NonTrainableState):
    """Identity of the inputs a meshing trace's evidence was computed from.

    Binds the geometry source revision, mesh topology, actual coordinate array
    and coordinate-element layout, the governing policy identities and the
    runtime identity.
    """

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    policy_ids: tuple[str, ...] = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_id: str,
        source_revision: str,
        topology_id: str,
        geometry_id: str,
        geometry_layout_id: str,
        policy_ids: tuple[str, ...],
        runtime_id: str,
    ) -> None:
        values = {
            "source_id": source_id,
            "source_revision": source_revision,
            "topology_id": topology_id,
            "geometry_id": geometry_id,
            "geometry_layout_id": geometry_layout_id,
            "runtime_id": runtime_id,
        }
        for name, value in values.items():
            canonical_identifier(value, name)
        policies = tuple(
            sorted(canonical_identifier(value, "policy_ids") for value in policy_ids)
        )
        self.source_id = source_id
        self.source_revision = source_revision
        self.topology_id = topology_id
        self.geometry_id = geometry_id
        self.geometry_layout_id = geometry_layout_id
        self.policy_ids = policies
        self.runtime_id = runtime_id
        self.binding_id = canonical_fingerprint(
            {"kind": "meshing-evidence-binding", **values, "policy_ids": policies}
        )


class MeshingTrace(StrictModule, NonTrainableState):
    """Ordered stage reports; a FAILED or UNRESOLVED stage must be last.

    ``binding`` (when present) identifies the source, topology, coordinates,
    layout, policies and runtime the recorded evidence belongs to.
    """

    stages: tuple[MeshingStageReport, ...]
    binding: MeshingEvidenceBinding | None
    successful: bool = eqx.field(static=True)
    trace_id: str = eqx.field(static=True)

    def __init__(
        self,
        stages: tuple[MeshingStageReport, ...],
        /,
        *,
        binding: MeshingEvidenceBinding | None = None,
    ) -> None:
        if not stages or not all(
            isinstance(stage, MeshingStageReport) for stage in stages
        ):
            raise ValueError("Meshing traces require at least one stage report.")
        if binding is not None and not isinstance(binding, MeshingEvidenceBinding):
            raise TypeError("binding must be MeshingEvidenceBinding or None.")
        terminal = tuple(
            index
            for index, stage in enumerate(stages)
            if stage.status in (MeshingStageStatus.FAILED, MeshingStageStatus.UNRESOLVED)
        )
        if terminal and terminal != (len(stages) - 1,):
            raise ValueError("A failed meshing stage must terminate the trace.")
        self.stages = tuple(stages)
        self.binding = binding
        self.successful = not terminal
        self.trace_id = canonical_fingerprint(
            {
                "kind": "meshing-trace",
                "stages": [stage.report_id for stage in stages],
                "binding": None if binding is None else binding.binding_id,
            }
        )


__all__ = [
    "MeshingDiagnostic",
    "MeshingDiagnosticSeverity",
    "MeshingEvidenceBinding",
    "MeshingStageKind",
    "MeshingStageReport",
    "MeshingStageStatus",
    "MeshingTrace",
]
