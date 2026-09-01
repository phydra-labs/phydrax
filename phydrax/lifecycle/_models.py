#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


RunStatus: TypeAlias = Literal[
    "planned", "queued", "running", "completed", "failed", "cancelled"
]
MetadataRecord: TypeAlias = tuple[tuple[str, str], ...]
PayloadRecord: TypeAlias = tuple[str, str]
ResultFieldRecord: TypeAlias = tuple[str, str, str]


class NumericRevision(StrictModule, NonTrainableState):
    """Immutable numeric-content identity and optional direct lineage."""

    content_digest: str = eqx.field(static=True)
    label: str = eqx.field(static=True)
    parent_digest: str | None = eqx.field(static=True)
    metadata: MetadataRecord = eqx.field(static=True)
    revision_id: str = eqx.field(static=True)

    def __init__(
        self,
        content_digest: str,
        /,
        *,
        label: str = "",
        parent_digest: str | None = None,
        metadata: Mapping[str, str] | Sequence[tuple[str, str]] = (),
    ):
        digest = _digest("content_digest", content_digest)
        label_ = str(label).strip()
        parent = (
            None if parent_digest is None else _digest("parent_digest", parent_digest)
        )
        metadata_ = _metadata(metadata)
        if parent == digest:
            raise ValueError("A numeric revision cannot parent itself.")
        self.content_digest = digest
        self.label = label_
        self.parent_digest = parent
        self.metadata = metadata_
        self.revision_id = canonical_fingerprint(
            {
                "kind": "numeric-revision",
                "content_digest": digest,
                "label": label_,
                "parent_digest": parent,
                "metadata": [list(record) for record in metadata_],
            }
        )


class CheckpointShard(StrictModule, NonTrainableState):
    """Checksum and ownership record for one checkpoint payload shard."""

    shard_id: str = eqx.field(static=True)
    payload_digest: str = eqx.field(static=True)
    byte_count: int = eqx.field(static=True)
    layout_ids: tuple[str, ...] = eqx.field(static=True)
    shard_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        shard_id: str,
        payload_digest: str,
        byte_count: int,
        layout_ids: Sequence[str] = (),
        /,
    ):
        shard = _identifier("shard_id", shard_id)
        digest = _digest("payload_digest", payload_digest)
        count = int(byte_count)
        layouts = _identifiers("layout_ids", layout_ids)
        if count < 0:
            raise ValueError("Checkpoint shard byte_count must be nonnegative.")
        self.shard_id = shard
        self.payload_digest = digest
        self.byte_count = count
        self.layout_ids = layouts
        self.shard_fingerprint = canonical_fingerprint(
            {
                "kind": "checkpoint-shard",
                "shard_id": shard,
                "payload_digest": digest,
                "byte_count": count,
                "layout_ids": list(layouts),
            }
        )


class CheckpointManifest(StrictModule, NonTrainableState):
    """Canonical all-shard checkpoint identity and completion boundary."""

    checkpoint_id: str = eqx.field(static=True)
    analysis_plan_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    execution_plan_id: str = eqx.field(static=True)
    shards: tuple[CheckpointShard, ...]
    complete: bool = eqx.field(static=True)
    parent_checkpoint_id: str | None = eqx.field(static=True)
    diagnostic_ids: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        checkpoint_id: str,
        analysis_plan_id: str,
        numeric_revision_id: str,
        execution_plan_id: str,
        shards: Sequence[CheckpointShard],
        /,
        *,
        complete: bool,
        parent_checkpoint_id: str | None = None,
        diagnostic_ids: Sequence[str] = (),
    ):
        checkpoint = _identifier("checkpoint_id", checkpoint_id)
        analysis = _identifier("analysis_plan_id", analysis_plan_id)
        revision = _identifier("numeric_revision_id", numeric_revision_id)
        execution = _identifier("execution_plan_id", execution_plan_id)
        shards_ = tuple(shards)
        complete_ = bool(complete)
        parent = _optional_identifier("parent_checkpoint_id", parent_checkpoint_id)
        diagnostics = _identifiers("diagnostic_ids", diagnostic_ids)
        if not all(isinstance(shard, CheckpointShard) for shard in shards_):
            raise TypeError("Checkpoint shards must contain CheckpointShard values.")
        if len({shard.shard_id for shard in shards_}) != len(shards_):
            raise ValueError("Checkpoint shard IDs must be unique.")
        if complete_ and not shards_:
            raise ValueError("A complete checkpoint requires at least one shard.")
        if parent == checkpoint:
            raise ValueError("A checkpoint cannot parent itself.")
        self.checkpoint_id = checkpoint
        self.analysis_plan_id = analysis
        self.numeric_revision_id = revision
        self.execution_plan_id = execution
        self.shards = shards_
        self.complete = complete_
        self.parent_checkpoint_id = parent
        self.diagnostic_ids = diagnostics
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "checkpoint-manifest",
                "checkpoint_id": checkpoint,
                "analysis_plan_id": analysis,
                "numeric_revision_id": revision,
                "execution_plan_id": execution,
                "shards": [shard.shard_fingerprint for shard in shards_],
                "complete": complete_,
                "parent_checkpoint_id": parent,
                "diagnostic_ids": list(diagnostics),
            }
        )


class AnalysisPlan(StrictModule, NonTrainableState):
    """Provider-neutral envelope referencing exactly one provider analysis plan."""

    analysis_plan_id: str = eqx.field(static=True)
    provider_plan_id: str = eqx.field(static=True)
    discretization_key: str = eqx.field(static=True)
    field_layout_ids: tuple[str, ...] = eqx.field(static=True)
    material_plan_id: str | None = eqx.field(static=True)
    constraint_ids: tuple[str, ...] = eqx.field(static=True)
    capability_ids: tuple[str, ...] = eqx.field(static=True)
    model_manifest_id: str | None = eqx.field(static=True)
    plan_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        analysis_plan_id: str,
        provider_plan_id: str,
        discretization_key: str,
        field_layout_ids: Sequence[str],
        /,
        *,
        material_plan_id: str | None = None,
        constraint_ids: Sequence[str] = (),
        capability_ids: Sequence[str] = (),
        model_manifest_id: str | None = None,
    ):
        analysis = _identifier("analysis_plan_id", analysis_plan_id)
        provider = _identifier("provider_plan_id", provider_plan_id)
        discretization = _identifier("discretization_key", discretization_key)
        layouts = _identifiers("field_layout_ids", field_layout_ids)
        material = _optional_identifier("material_plan_id", material_plan_id)
        constraints = _identifiers("constraint_ids", constraint_ids)
        capabilities = _identifiers("capability_ids", capability_ids)
        model = _optional_identifier("model_manifest_id", model_manifest_id)
        if not layouts:
            raise ValueError("AnalysisPlan requires at least one field layout.")
        self.analysis_plan_id = analysis
        self.provider_plan_id = provider
        self.discretization_key = discretization
        self.field_layout_ids = layouts
        self.material_plan_id = material
        self.constraint_ids = constraints
        self.capability_ids = capabilities
        self.model_manifest_id = model
        self.plan_fingerprint = canonical_fingerprint(
            {
                "kind": "analysis-plan",
                "analysis_plan_id": analysis,
                "provider_plan_id": provider,
                "discretization_key": discretization,
                "field_layout_ids": list(layouts),
                "material_plan_id": material,
                "constraint_ids": list(constraints),
                "capability_ids": list(capabilities),
                "model_manifest_id": model,
            }
        )


class ExecutionPlan(StrictModule, NonTrainableState):
    """Backend and numerical-policy identity for an analysis execution."""

    execution_plan_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    solver_policy_id: str = eqx.field(static=True)
    device_mesh_id: str | None = eqx.field(static=True)
    reduction_policy_id: str | None = eqx.field(static=True)
    cache_key: str | None = eqx.field(static=True)
    plan_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        execution_plan_id: str,
        backend: str,
        precision_policy_id: str,
        solver_policy_id: str,
        /,
        *,
        device_mesh_id: str | None = None,
        reduction_policy_id: str | None = None,
        cache_key: str | None = None,
    ):
        execution = _identifier("execution_plan_id", execution_plan_id)
        backend_ = _identifier("backend", backend)
        precision = _identifier("precision_policy_id", precision_policy_id)
        solver = _identifier("solver_policy_id", solver_policy_id)
        device = _optional_identifier("device_mesh_id", device_mesh_id)
        reduction = _optional_identifier("reduction_policy_id", reduction_policy_id)
        cache = _optional_identifier("cache_key", cache_key)
        self.execution_plan_id = execution
        self.backend = backend_
        self.precision_policy_id = precision
        self.solver_policy_id = solver
        self.device_mesh_id = device
        self.reduction_policy_id = reduction
        self.cache_key = cache
        self.plan_fingerprint = canonical_fingerprint(
            {
                "kind": "execution-plan",
                "execution_plan_id": execution,
                "backend": backend_,
                "precision_policy_id": precision,
                "solver_policy_id": solver,
                "device_mesh_id": device,
                "reduction_policy_id": reduction,
                "cache_key": cache,
            }
        )


class RunRecord(StrictModule, NonTrainableState):
    """Immutable execution state linking plans, diagnostics, results, and restart."""

    run_id: str = eqx.field(static=True)
    analysis_plan_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    execution_plan_id: str = eqx.field(static=True)
    status: RunStatus = eqx.field(static=True)
    result_ids: tuple[str, ...] = eqx.field(static=True)
    diagnostic_ids: tuple[str, ...] = eqx.field(static=True)
    checkpoint_id: str | None = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        run_id: str,
        analysis_plan_id: str,
        numeric_revision_id: str,
        execution_plan_id: str,
        status: RunStatus,
        /,
        *,
        result_ids: Sequence[str] = (),
        diagnostic_ids: Sequence[str] = (),
        checkpoint_id: str | None = None,
    ):
        run = _identifier("run_id", run_id)
        analysis = _identifier("analysis_plan_id", analysis_plan_id)
        revision = _identifier("numeric_revision_id", numeric_revision_id)
        execution = _identifier("execution_plan_id", execution_plan_id)
        status_ = str(status).strip()
        results = _identifiers("result_ids", result_ids)
        diagnostics = _identifiers("diagnostic_ids", diagnostic_ids)
        checkpoint = _optional_identifier("checkpoint_id", checkpoint_id)
        if status_ not in (
            "planned",
            "queued",
            "running",
            "completed",
            "failed",
            "cancelled",
        ):
            raise ValueError("Unknown run status.")
        self.run_id = run
        self.analysis_plan_id = analysis
        self.numeric_revision_id = revision
        self.execution_plan_id = execution
        self.status = status_  # type: ignore[assignment]
        self.result_ids = results
        self.diagnostic_ids = diagnostics
        self.checkpoint_id = checkpoint
        self.record_id = canonical_fingerprint(
            {
                "kind": "run-record",
                "run_id": run,
                "analysis_plan_id": analysis,
                "numeric_revision_id": revision,
                "execution_plan_id": execution,
                "status": status_,
                "result_ids": list(results),
                "diagnostic_ids": list(diagnostics),
                "checkpoint_id": checkpoint,
            }
        )


class ModelManifest(StrictModule, NonTrainableState):
    """Model payload identity bound to an analysis and numeric revision."""

    model_id: str = eqx.field(static=True)
    analysis_plan_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    payloads: tuple[PayloadRecord, ...] = eqx.field(static=True)
    unit_contract_id: str | None = eqx.field(static=True)
    association_ids: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        model_id: str,
        analysis_plan_id: str,
        numeric_revision_id: str,
        payloads: Mapping[str, str] | Sequence[PayloadRecord],
        /,
        *,
        unit_contract_id: str | None = None,
        association_ids: Sequence[str] = (),
    ):
        model = _identifier("model_id", model_id)
        analysis = _identifier("analysis_plan_id", analysis_plan_id)
        revision = _identifier("numeric_revision_id", numeric_revision_id)
        payloads_ = _payloads(payloads)
        units = _optional_identifier("unit_contract_id", unit_contract_id)
        associations = _identifiers("association_ids", association_ids)
        if not payloads_:
            raise ValueError("ModelManifest requires at least one payload.")
        self.model_id = model
        self.analysis_plan_id = analysis
        self.numeric_revision_id = revision
        self.payloads = payloads_
        self.unit_contract_id = units
        self.association_ids = associations
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "model-manifest",
                "model_id": model,
                "analysis_plan_id": analysis,
                "numeric_revision_id": revision,
                "payloads": [list(record) for record in payloads_],
                "unit_contract_id": units,
                "association_ids": list(associations),
            }
        )


class ResultManifest(StrictModule, NonTrainableState):
    """Named result fields, physical units, and payload checksums for one run."""

    result_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    fields: tuple[ResultFieldRecord, ...] = eqx.field(static=True)
    payloads: tuple[PayloadRecord, ...] = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    diagnostic_ids: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        result_id: str,
        run_id: str,
        fields: Mapping[str, str]
        | Sequence[str]
        | Sequence[tuple[str, str]]
        | Sequence[ResultFieldRecord],
        payloads: Mapping[str, str] | Sequence[PayloadRecord],
        /,
        *,
        evidence_ids: Sequence[str] = (),
        diagnostic_ids: Sequence[str] = (),
    ):
        result = _identifier("result_id", result_id)
        run = _identifier("run_id", run_id)
        fields_ = _result_fields(fields)
        payloads_ = _payloads(payloads)
        evidence = _identifiers("evidence_ids", evidence_ids)
        diagnostics = _identifiers("diagnostic_ids", diagnostic_ids)
        payload_names = {record[0] for record in payloads_}
        if not fields_ or not payloads_:
            raise ValueError("ResultManifest requires fields and payloads.")
        if any(record[1] not in payload_names for record in fields_):
            raise ValueError("Every result field must reference a named payload.")
        self.result_id = result
        self.run_id = run
        self.fields = fields_
        self.payloads = payloads_
        self.evidence_ids = evidence
        self.diagnostic_ids = diagnostics
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "result-manifest",
                "result_id": result,
                "run_id": run,
                "fields": [list(record) for record in fields_],
                "payloads": [list(record) for record in payloads_],
                "evidence_ids": list(evidence),
                "diagnostic_ids": list(diagnostics),
            }
        )


class ResultRevision(StrictModule, NonTrainableState):
    """Immutable result-manifest revision with optional direct ancestry."""

    manifest: ResultManifest
    parent_result_id: str | None = eqx.field(static=True)
    revision_id: str = eqx.field(static=True)

    def __init__(
        self,
        manifest: ResultManifest,
        parent_result_id: str | None = None,
        /,
    ):
        if not isinstance(manifest, ResultManifest):
            raise TypeError("manifest must be a ResultManifest.")
        parent = _optional_identifier("parent_result_id", parent_result_id)
        if parent == manifest.result_id:
            raise ValueError("A result revision cannot parent itself.")
        self.manifest = manifest
        self.parent_result_id = parent
        self.revision_id = canonical_fingerprint(
            {
                "kind": "result-revision",
                "manifest_id": manifest.manifest_id,
                "parent_result_id": parent,
            }
        )


def _identifier(name: str, value: str, /) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _optional_identifier(name: str, value: str | None, /) -> str | None:
    return None if value is None else _identifier(name, value)


def _identifiers(name: str, values: Sequence[str], /) -> tuple[str, ...]:
    normalized = tuple(_identifier(name, value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique.")
    return normalized


def _digest(name: str, value: str, /) -> str:
    digest = _identifier(name, value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return digest


def _metadata(value: Mapping[str, str] | Sequence[tuple[str, str]], /) -> MetadataRecord:
    records = value.items() if isinstance(value, Mapping) else value
    normalized = tuple(
        sorted(
            (
                _identifier("metadata key", key),
                _identifier("metadata value", item),
            )
            for key, item in records
        )
    )
    if len({record[0] for record in normalized}) != len(normalized):
        raise ValueError("Numeric revision metadata keys must be unique.")
    return normalized


def _payloads(
    value: Mapping[str, str] | Sequence[PayloadRecord], /
) -> tuple[PayloadRecord, ...]:
    records = value.items() if isinstance(value, Mapping) else value
    normalized = tuple(
        (_identifier("payload name", name), _digest("payload digest", digest))
        for name, digest in records
    )
    if len({record[0] for record in normalized}) != len(normalized):
        raise ValueError("Payload names must be unique.")
    return normalized


def _result_fields(
    value: Mapping[str, str]
    | Sequence[str]
    | Sequence[tuple[str, str]]
    | Sequence[ResultFieldRecord],
    /,
) -> tuple[ResultFieldRecord, ...]:
    if isinstance(value, Mapping):
        records = tuple((name, name, unit) for name, unit in value.items())
    else:
        records = []
        for record in value:
            if isinstance(record, str):
                records.append((record, record, "1"))
            elif len(record) == 2:
                records.append((record[0], record[0], record[1]))
            elif len(record) == 3:
                records.append(record)
            else:
                raise ValueError("Result field records must have two or three entries.")
    normalized = tuple(
        (
            _identifier("field name", name),
            _identifier("field payload", payload),
            _identifier("field unit", unit),
        )
        for name, payload, unit in records
    )
    if len({record[0] for record in normalized}) != len(normalized):
        raise ValueError("Result field names must be unique.")
    return normalized


__all__ = [
    "AnalysisPlan",
    "CheckpointManifest",
    "CheckpointShard",
    "ExecutionPlan",
    "MetadataRecord",
    "ModelManifest",
    "NumericRevision",
    "PayloadRecord",
    "ResultFieldRecord",
    "ResultManifest",
    "ResultRevision",
    "RunRecord",
    "RunStatus",
]
