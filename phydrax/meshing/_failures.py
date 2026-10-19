#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Meshing failure categories and structured scientific/resource evidence."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import SupportsFloat

import equinox as eqx
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._meshcore import NativeExecutionEvidence
from .._strict import StrictModule
from .._trainable import NonTrainableState


type FailureQuantity = SupportsFloat


class MeshingFailureCategory(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_SPECIFICATION = "invalid_specification"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    UNSUPPORTED_COMBINATION = "unsupported_combination"
    INVALID_SOURCE = "invalid_source"
    SCOPE_RESOLUTION_FAILED = "scope_resolution_failed"
    CONTROL_CONFLICT = "control_conflict"
    REGION_RESOLUTION_FAILED = "region_resolution_failed"
    PROVIDER_EXECUTION_FAILED = "provider_execution_failed"
    INTERRUPTED = "interrupted"
    TIMED_OUT = "timed_out"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    CONVERSION_FAILED = "conversion_failed"
    CANONICALIZATION_FAILED = "canonicalization_failed"
    ASSOCIATION_FAILED = "association_failed"
    AUDIT_FAILED = "audit_failed"
    QUALITY_REJECTED = "quality_rejected"
    COMPLIANCE_FAILED = "compliance_failed"
    LINEAGE_FAILED = "lineage_failed"
    TRANSFER_FAILED = "transfer_failed"


def quantities(
    values: tuple[tuple[str, FailureQuantity], ...], name: str, /
) -> tuple[tuple[str, float], ...]:
    entries = tuple((str(key).strip(), float(value)) for key, value in values)
    keys = tuple(key for key, _ in entries)
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError(
            f"{name}.quantities must have unique nonempty names; got duplicate or empty keys"
        )
    if any(np.isnan(value) for _, value in entries):
        raise ValueError(f"{name}.quantities must exclude NaN; got a NaN value")
    return tuple(sorted(entries))


def quantity_payload(
    entries: tuple[tuple[str, float], ...], /
) -> list[list[str | float]]:
    """Return identity quantities with explicit tokens for unbounded values."""
    return [
        [key, value if np.isfinite(value) else ("+inf" if value > 0 else "-inf")]
        for key, value in entries
    ]


class MeshingFailureEvidence(StrictModule, NonTrainableState):
    """Structured evidence of one failed meshing request.

    ``stage`` names the failing stage (a ``MeshingStageKind`` value on native
    routes), ``entity_ids`` the failing or unresolved entities, and
    ``requested``/``achieved`` the named quantities in the convention of
    ``MeshingComplianceReport``. ``checkpoint_id`` references the lifecycle
    checkpoint of the last accepted state when the failing operation has one.
    """

    category: MeshingFailureCategory = eqx.field(static=True)
    message: str = eqx.field(static=True)
    provider_code: str = eqx.field(static=True)
    stage: str = eqx.field(static=True)
    entity_ids: tuple[int, ...] = eqx.field(static=True)
    locations: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    requested: tuple[tuple[str, float], ...] = eqx.field(static=True)
    achieved: tuple[tuple[str, float], ...] = eqx.field(static=True)
    checkpoint_id: str | None = eqx.field(static=True)
    logical_findings: tuple[tuple[str, Array], ...]
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        category: MeshingFailureCategory,
        message: str,
        /,
        *,
        provider_code: str = "",
        stage: str = "",
        entity_ids: tuple[int, ...] = (),
        locations: tuple[tuple[float, ...], ...] = (),
        requested: tuple[tuple[str, FailureQuantity], ...] = (),
        achieved: tuple[tuple[str, FailureQuantity], ...] = (),
        checkpoint_id: str | None = None,
        logical_findings: tuple[tuple[str, Array], ...] = (),
    ) -> None:
        if not isinstance(category, MeshingFailureCategory):
            raise TypeError(
                f"MeshingFailureEvidence.category must be MeshingFailureCategory; got {type(category).__name__}"
            )
        text = str(message).strip()
        if not text:
            raise ValueError(
                f"MeshingFailureEvidence.message must contain nonwhitespace text; got {message!r}"
            )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in entity_ids
        ):
            raise TypeError(
                f"MeshingFailureEvidence.entity_ids must contain integers; got {entity_ids!r}"
            )
        points = tuple(
            tuple(float(component) for component in point) for point in locations
        )
        if any(not np.all(np.isfinite(point)) for point in points):
            raise ValueError(
                f"MeshingFailureEvidence.locations must contain finite coordinates; got {points!r}"
            )
        checkpoint = None if checkpoint_id is None else str(checkpoint_id).strip()
        if checkpoint == "":
            raise ValueError(
                f"MeshingFailureEvidence.checkpoint_id must be nonempty when supplied; got {checkpoint_id!r}"
            )
        findings = tuple(logical_findings)
        names = tuple(name for name, _ in findings)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError(
                "logical_findings.names must be unique and canonically ordered; got incompatible names"
            )
        if any(not isinstance(value, Array) for _, value in findings):
            raise TypeError(
                "logical_findings.values must remain JAX arrays; got another object kind"
            )
        self.category = category
        self.message = text
        self.provider_code = str(provider_code)
        self.stage = str(stage)
        self.entity_ids = tuple(int(value) for value in entity_ids)
        self.locations = points
        self.requested = quantities(requested, "requested")
        self.achieved = quantities(achieved, "achieved")
        self.checkpoint_id = checkpoint
        self.logical_findings = findings
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "meshing-failure-evidence",
                "category": category.value,
                "message": text,
                "provider_code": self.provider_code,
                "stage": self.stage,
                "entity_ids": list(self.entity_ids),
                "locations": [list(point) for point in points],
                "requested": quantity_payload(self.requested),
                "achieved": quantity_payload(self.achieved),
                "checkpoint_id": checkpoint,
                "logical_findings": None
                if not findings
                else logical_array_value_collection_digest(dict(findings)),
            }
        )


class CutCornerLinkIncidence(StrictModule):
    """Store the scientific cell, vertex, edge, and face IDs of a corner link."""

    cell_id: int = eqx.field(static=True)
    vertex_id: int = eqx.field(static=True)
    edge_ids: tuple[int, ...] = eqx.field(static=True)
    face_ids: tuple[int, ...] = eqx.field(static=True)


class CutStageRecord(StrictModule):
    """Retain execution evidence for a completed or refused cut phase."""

    stage: str = eqx.field(static=True)
    origin: tuple[Fraction, ...] | None = eqx.field(static=True)
    execution: NativeExecutionEvidence
    failure: str | None = eqx.field(static=True)
    link_incidence: CutCornerLinkIncidence | None


class CutStageError(Exception):
    """Carry the phase prefix and its unmodified failure to the request boundary."""

    cause: BaseException
    stage_records: tuple[CutStageRecord, ...]

    def __init__(
        self, cause: BaseException, stage_records: tuple[CutStageRecord, ...]
    ) -> None:
        self.cause = cause
        self.stage_records = stage_records
        super().__init__(str(cause))


class MeshingFailure(RuntimeError):
    """Failed meshing request carrying its structured ``evidence``."""

    cut_stage_records: tuple[CutStageRecord, ...]

    def __init__(
        self,
        category: MeshingFailureCategory,
        message: str,
        /,
        *,
        provider_code: str = "",
        stage: str = "",
        entity_ids: tuple[int, ...] = (),
        locations: tuple[tuple[float, ...], ...] = (),
        requested: tuple[tuple[str, FailureQuantity], ...] = (),
        achieved: tuple[tuple[str, FailureQuantity], ...] = (),
        checkpoint_id: str | None = None,
        logical_findings: tuple[tuple[str, Array], ...] = (),
        cut_stage_records: tuple[CutStageRecord, ...] = (),
    ) -> None:
        self.evidence = MeshingFailureEvidence(
            category,
            message,
            provider_code=provider_code,
            stage=stage,
            entity_ids=entity_ids,
            locations=locations,
            requested=requested,
            achieved=achieved,
            checkpoint_id=checkpoint_id,
            logical_findings=logical_findings,
        )
        self.cut_stage_records = cut_stage_records
        super().__init__(self.evidence.message)

    @property
    def category(self) -> MeshingFailureCategory:
        return self.evidence.category

    @property
    def provider_code(self) -> str:
        return self.evidence.provider_code

    @property
    def stage(self) -> str:
        return self.evidence.stage

    @property
    def entity_ids(self) -> tuple[int, ...]:
        return self.evidence.entity_ids

    @property
    def locations(self) -> tuple[tuple[float, ...], ...]:
        return self.evidence.locations

    @property
    def requested(self) -> tuple[tuple[str, float], ...]:
        return self.evidence.requested

    @property
    def achieved(self) -> tuple[tuple[str, float], ...]:
        return self.evidence.achieved
