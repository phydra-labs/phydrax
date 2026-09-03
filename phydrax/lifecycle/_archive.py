#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np

from .._array_archive import (
    array_collection_digest,
    array_payload_byte_count,
    array_payload_digest,
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from .._fingerprint import canonical_fingerprint
from ..diagnostics import Diagnostic, DiagnosticError
from ._models import (
    AnalysisPlan,
    CheckpointManifest,
    CheckpointShard,
    ExecutionPlan,
    ModelManifest,
    NumericRevision,
    ResultManifest,
    ResultRevision,
    RunRecord,
)


LifecycleRecord: TypeAlias = (
    NumericRevision
    | CheckpointManifest
    | AnalysisPlan
    | ExecutionPlan
    | RunRecord
    | ModelManifest
    | ResultManifest
    | ResultRevision
)


@dataclass(frozen=True, slots=True)
class LifecycleArchive:
    """Validated lifecycle record and immutable named payload arrays."""

    path: Path
    manifest: LifecycleRecord
    arrays: Mapping[str, np.ndarray]
    archive_id: str


@dataclass(frozen=True, slots=True)
class SampledField:
    """One selected named payload and its physical unit."""

    name: str
    payload_name: str
    unit: str | None
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class LifecycleQuery:
    """Validated metadata and selected payloads for generic sampled export."""

    archive: LifecycleArchive
    fields: tuple[SampledField, ...]


SupportBundleDisclosure: TypeAlias = Literal[
    "arrays", "payloads", "paths", "identifiers", "free-text", "secrets"
]
_FULL_SUPPORT_DISCLOSURE = frozenset(
    {"arrays", "payloads", "paths", "identifiers", "free-text", "secrets"}
)


@dataclass(frozen=True, slots=True)
class SupportBundleAuthorization:
    """Auditable data-owner authorization for a full-fidelity support payload."""

    authorization_id: str
    data_owner_id: str
    source_archive_id: str
    authorized_at: int
    disclosures: frozenset[SupportBundleDisclosure]

    def __post_init__(self) -> None:
        identifiers = (
            self.authorization_id,
            self.data_owner_id,
            self.source_archive_id,
        )
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or any(ord(character) < 32 for character in value)
            for value in identifiers
        ):
            raise ValueError(
                "Support authorization identifiers must be non-empty stripped text."
            )
        if (
            isinstance(self.authorized_at, bool)
            or not isinstance(self.authorized_at, int)
            or self.authorized_at < 0
        ):
            raise ValueError("Support authorization time must be a non-negative integer.")
        disclosures = frozenset(self.disclosures)
        if disclosures != _FULL_SUPPORT_DISCLOSURE:
            raise ValueError(
                "Full-archive support authorization must explicitly grant every "
                "sensitive disclosure category."
            )
        object.__setattr__(self, "disclosures", disclosures)


class SampledExporter(Protocol):
    """Extension point for provider-neutral sampled result export."""

    def __call__(self, query: LifecycleQuery, destination: Path, /) -> Path: ...


_EXPORTERS: dict[str, SampledExporter] = {}


def payload_digest(value: Any, /) -> str:
    """Return the canonical checksum expected by lifecycle payload records."""
    return array_payload_digest(value)


def payload_byte_count(value: Any, /) -> int:
    """Return the canonical serialized byte count for one payload array."""
    return array_payload_byte_count(value)


def collection_digest(values: Mapping[str, Any], /) -> str:
    """Return one content identity for a named payload collection."""
    return array_collection_digest(values)


def create(
    path: str | Path,
    /,
    *,
    manifest: LifecycleRecord,
    arrays: Mapping[str, Any],
) -> LifecycleArchive:
    """Atomically create and reopen one checksum-bound lifecycle archive."""

    if not isinstance(
        manifest,
        (
            NumericRevision,
            CheckpointManifest,
            AnalysisPlan,
            ExecutionPlan,
            RunRecord,
            ModelManifest,
            ResultManifest,
            ResultRevision,
        ),
    ):
        raise TypeError("manifest must be a lifecycle record.")
    arrays_ = {str(name): np.asarray(value) for name, value in arrays.items()}
    if any(not name for name in arrays_):
        raise ValueError("Lifecycle payload names must be non-empty.")
    _validate_payloads(manifest, arrays_)
    record = _encode_record(manifest)
    record_digest = canonical_fingerprint(record)
    archive_id = canonical_fingerprint(
        {
            "kind": "lifecycle-archive",
            "record_digest": record_digest,
            "payload_digest": array_collection_digest(arrays_),
        }
    )
    write_array_archive(
        path,
        manifest={
            "kind": "lifecycle-archive",
            "record": record,
            "record_digest": record_digest,
            "archive_id": archive_id,
        },
        arrays=arrays_,
    )
    return open(path, allow_incomplete=True, limits=None)


def open(
    path: str | Path,
    /,
    *,
    allow_incomplete: bool = False,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> LifecycleArchive:
    """Open a canonical archive, validating every identity and payload checksum."""

    source = Path(path)
    container, arrays = read_array_archive(source, limits=limits)
    if container.get("kind") != "lifecycle-archive":
        raise ArrayArchiveCorruptionError("Archive is not a lifecycle archive.")
    record = container.get("record")
    if not isinstance(record, Mapping):
        raise ArrayArchiveCorruptionError("Lifecycle record is missing.")
    record_digest = canonical_fingerprint(record)
    if container.get("record_digest") != record_digest:
        raise ArrayArchiveCorruptionError("Lifecycle record checksum failed.")
    manifest = _decode_record(record)
    _validate_payloads(manifest, arrays)
    archive_id = canonical_fingerprint(
        {
            "kind": "lifecycle-archive",
            "record_digest": record_digest,
            "payload_digest": array_collection_digest(arrays),
        }
    )
    if container.get("archive_id") != archive_id:
        raise ArrayArchiveCorruptionError("Lifecycle archive identity mismatch.")
    if (
        isinstance(manifest, CheckpointManifest)
        and not manifest.complete
        and not allow_incomplete
    ):
        raise DiagnosticError(
            (
                Diagnostic(
                    "LIFECYCLE_CHECKPOINT_INCOMPLETE",
                    "error",
                    "open",
                    "Checkpoint manifest is not complete.",
                    entity_ids=(manifest.checkpoint_id,),
                    remediation="Resume shard creation or open the last complete checkpoint.",
                ),
            )
        )
    immutable = MappingProxyType(dict(arrays))
    return LifecycleArchive(source, manifest, immutable, archive_id)


def list_fields(source: str | Path | LifecycleArchive, /) -> tuple[str, ...]:
    """List stable result-field names or generic named payloads."""

    archive = _as_archive(source)
    manifest = _result_manifest(archive.manifest)
    if manifest is not None:
        return tuple(record[0] for record in manifest.fields)
    return tuple(sorted(archive.arrays))


def query(
    source: str | Path | LifecycleArchive,
    /,
    *,
    fields: Sequence[str] = (),
) -> LifecycleQuery:
    """Select metadata-bound sampled fields without provider-specific logic."""

    archive = _as_archive(source)
    requested = tuple(str(name).strip() for name in fields)
    if any(not name for name in requested) or len(set(requested)) != len(requested):
        raise ValueError("Queried field names must be non-empty and unique.")
    result = _result_manifest(archive.manifest)
    sampled: list[SampledField] = []
    if result is not None:
        available = {record[0]: record for record in result.fields}
        selected = tuple(available) if not requested else requested
        missing = tuple(name for name in selected if name not in available)
        if missing:
            raise KeyError(f"Unknown lifecycle result fields: {', '.join(missing)}")
        for name in selected:
            _, payload_name, unit = available[name]
            sampled.append(
                SampledField(name, payload_name, unit, archive.arrays[payload_name])
            )
    else:
        selected = tuple(sorted(archive.arrays)) if not requested else requested
        missing = tuple(name for name in selected if name not in archive.arrays)
        if missing:
            raise KeyError(f"Unknown lifecycle payloads: {', '.join(missing)}")
        sampled.extend(
            SampledField(name, name, None, archive.arrays[name]) for name in selected
        )
    return LifecycleQuery(archive, tuple(sampled))


def register_exporter(
    format: str,
    exporter: SampledExporter,
    /,
    *,
    replace: bool = False,
) -> None:
    """Register one explicit sampled-data exporter extension."""

    key = str(format).strip().lower()
    if not key or not callable(exporter):
        raise ValueError("Exporter format and callable must be valid.")
    if key in _EXPORTERS and not replace:
        raise ValueError(f"Exporter {key!r} is already registered.")
    _EXPORTERS[key] = exporter


def export(
    source: str | Path | LifecycleArchive,
    destination: str | Path,
    /,
    *,
    format: str,
    fields: Sequence[str] = (),
) -> Path:
    """Export selected sampled payloads through a registered generic exporter."""

    key = str(format).strip().lower()
    if key not in _EXPORTERS:
        raise ValueError(f"No sampled exporter is registered for {key!r}.")
    return _EXPORTERS[key](query(source, fields=fields), Path(destination))


_SUPPORT_RECORD_KINDS = frozenset(
    {
        "analysis-plan",
        "checkpoint-manifest",
        "execution-plan",
        "model-manifest",
        "numeric-revision",
        "result-manifest",
        "result-revision",
        "run-record",
    }
)
_SUPPORT_TELEMETRY_ALLOWLIST: Mapping[str, Any] = MappingProxyType(
    {
        "record": MappingProxyType({"kind": str, "complete": bool}),
        "archive": MappingProxyType({"array_count": int, "array_bytes": int}),
    }
)


def _allowlisted_support_mapping(
    source: Mapping[str, Any],
    schema: Mapping[str, Any],
    /,
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, rule in schema.items():
        if key not in source:
            continue
        value = source[key]
        if isinstance(rule, Mapping):
            if not isinstance(value, Mapping):
                raise TypeError(f"Support telemetry field {key!r} must be a mapping.")
            sanitized[key] = _allowlisted_support_mapping(value, rule)
        elif type(value) is not rule:
            raise TypeError(
                f"Support telemetry field {key!r} has an invalid scalar type."
            )
        else:
            sanitized[key] = value
    return sanitized


def _sanitized_support_telemetry(archive: LifecycleArchive, /) -> dict[str, Any]:
    record = _encode_record(archive.manifest)
    telemetry = _allowlisted_support_mapping(
        {
            "record": record,
            "archive": {
                "array_count": len(archive.arrays),
                "array_bytes": sum(value.nbytes for value in archive.arrays.values()),
            },
        },
        _SUPPORT_TELEMETRY_ALLOWLIST,
    )
    record_kind = telemetry["record"]["kind"]
    if record_kind not in _SUPPORT_RECORD_KINDS:
        raise TypeError("Lifecycle record kind is not admitted to support telemetry.")
    return telemetry


def support_bundle(
    source: str | Path | LifecycleArchive,
    destination: str | Path,
    /,
    *,
    authorization: SupportBundleAuthorization | None = None,
    archive_limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Path:
    """Create a sanitized support bundle, or an explicitly authorized full copy.

    The default bundle contains only recursively allowlisted numeric/enum
    telemetry. A full source archive can only be included with explicit,
    auditable authorization from its data owner.
    """
    if authorization is not None and not isinstance(
        authorization, SupportBundleAuthorization
    ):
        raise TypeError("authorization must be SupportBundleAuthorization or None.")
    archive = _as_archive(source, limits=archive_limits)
    telemetry = _sanitized_support_telemetry(archive)
    if authorization is None:
        return write_array_archive(
            destination,
            manifest={
                "kind": "lifecycle-support-bundle",
                "disclosure": "sanitized",
                "telemetry": telemetry,
                "audit": {"data_owner_authorized": False},
            },
            arrays={},
        )
    if authorization.source_archive_id != archive.archive_id:
        raise ValueError("Support authorization is not bound to the source archive.")

    payload = archive.path.read_bytes()
    record = _encode_record(archive.manifest)
    authorization_record = {
        "authorization_id": authorization.authorization_id,
        "data_owner_id": authorization.data_owner_id,
        "source_archive_id": authorization.source_archive_id,
        "authorized_at": authorization.authorized_at,
        "disclosures": sorted(authorization.disclosures),
    }
    authorization_record["authorization_fingerprint"] = canonical_fingerprint(
        authorization_record
    )
    return write_array_archive(
        destination,
        manifest={
            "kind": "lifecycle-support-bundle",
            "disclosure": "data-owner-authorized",
            "telemetry": telemetry,
            "audit": {
                "data_owner_authorized": True,
                **authorization_record,
            },
            "source": {
                "archive_id": archive.archive_id,
                "record_kind": record["kind"],
                "record_id": _record_id(archive.manifest),
                "diagnostic_ids": list(_diagnostic_ids(archive.manifest)),
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
            },
        },
        arrays={"archive": np.frombuffer(payload, dtype=np.uint8)},
    )


def _export_npz(query_: LifecycleQuery, destination: Path, /) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {field.name: field.values for field in query_.fields}
    metadata = {
        "archive_id": query_.archive.archive_id,
        "fields": [
            {
                "name": field.name,
                "payload_name": field.payload_name,
                "unit": field.unit,
            }
            for field in query_.fields
        ],
    }
    np.savez(
        destination,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    return destination


def _as_archive(
    source: str | Path | LifecycleArchive,
    /,
    *,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> LifecycleArchive:
    if isinstance(source, LifecycleArchive):
        return source
    return open(source, limits=limits)


def _result_manifest(record: LifecycleRecord, /) -> ResultManifest | None:
    if isinstance(record, ResultManifest):
        return record
    if isinstance(record, ResultRevision):
        return record.manifest
    return None


def _validate_payloads(
    manifest: LifecycleRecord,
    arrays: Mapping[str, Any],
    /,
) -> None:
    expected: tuple[tuple[str, str, int | None], ...]
    if isinstance(manifest, CheckpointManifest):
        expected = tuple(
            (shard.shard_id, shard.payload_digest, shard.byte_count)
            for shard in manifest.shards
        )
    elif isinstance(manifest, ModelManifest):
        expected = tuple((name, digest, None) for name, digest in manifest.payloads)
    elif isinstance(manifest, ResultManifest):
        expected = tuple((name, digest, None) for name, digest in manifest.payloads)
    elif isinstance(manifest, ResultRevision):
        expected = tuple(
            (name, digest, None) for name, digest in manifest.manifest.payloads
        )
    elif isinstance(manifest, NumericRevision):
        if arrays and array_collection_digest(arrays) != manifest.content_digest:
            raise ValueError("Numeric revision content digest does not match payloads.")
        return
    else:
        if arrays:
            raise ValueError("This lifecycle record does not own array payloads.")
        return
    names = {name for name, _, _ in expected}
    if names != set(arrays):
        raise ValueError("Lifecycle payload inventory does not match its manifest.")
    for name, digest, byte_count in expected:
        value = arrays[name]
        if array_payload_digest(value) != digest:
            raise ValueError(f"Lifecycle payload {name!r} checksum mismatch.")
        if byte_count is not None and array_payload_byte_count(value) != byte_count:
            raise ValueError(f"Lifecycle payload {name!r} byte count mismatch.")


def _encode_record(record: LifecycleRecord, /) -> dict[str, Any]:
    if isinstance(record, NumericRevision):
        return {
            "kind": "numeric-revision",
            "content_digest": record.content_digest,
            "label": record.label,
            "parent_digest": record.parent_digest,
            "metadata": [list(item) for item in record.metadata],
            "revision_id": record.revision_id,
        }
    if isinstance(record, CheckpointManifest):
        return {
            "kind": "checkpoint-manifest",
            "checkpoint_id": record.checkpoint_id,
            "analysis_plan_id": record.analysis_plan_id,
            "numeric_revision_id": record.numeric_revision_id,
            "execution_plan_id": record.execution_plan_id,
            "shards": [
                {
                    "shard_id": shard.shard_id,
                    "payload_digest": shard.payload_digest,
                    "byte_count": shard.byte_count,
                    "layout_ids": list(shard.layout_ids),
                    "shard_fingerprint": shard.shard_fingerprint,
                }
                for shard in record.shards
            ],
            "complete": record.complete,
            "parent_checkpoint_id": record.parent_checkpoint_id,
            "diagnostic_ids": list(record.diagnostic_ids),
            "manifest_id": record.manifest_id,
        }
    if isinstance(record, AnalysisPlan):
        return {
            "kind": "analysis-plan",
            "analysis_plan_id": record.analysis_plan_id,
            "provider_plan_id": record.provider_plan_id,
            "discretization_key": record.discretization_key,
            "field_layout_ids": list(record.field_layout_ids),
            "material_plan_id": record.material_plan_id,
            "constraint_ids": list(record.constraint_ids),
            "capability_ids": list(record.capability_ids),
            "model_manifest_id": record.model_manifest_id,
            "plan_fingerprint": record.plan_fingerprint,
        }
    if isinstance(record, ExecutionPlan):
        return {
            "kind": "execution-plan",
            "execution_plan_id": record.execution_plan_id,
            "backend": record.backend,
            "precision_policy_id": record.precision_policy_id,
            "solver_policy_id": record.solver_policy_id,
            "device_mesh_id": record.device_mesh_id,
            "reduction_policy_id": record.reduction_policy_id,
            "cache_key": record.cache_key,
            "plan_fingerprint": record.plan_fingerprint,
        }
    if isinstance(record, RunRecord):
        return {
            "kind": "run-record",
            "run_id": record.run_id,
            "analysis_plan_id": record.analysis_plan_id,
            "numeric_revision_id": record.numeric_revision_id,
            "execution_plan_id": record.execution_plan_id,
            "status": record.status,
            "result_ids": list(record.result_ids),
            "diagnostic_ids": list(record.diagnostic_ids),
            "checkpoint_id": record.checkpoint_id,
            "record_id": record.record_id,
        }
    if isinstance(record, ModelManifest):
        return {
            "kind": "model-manifest",
            "model_id": record.model_id,
            "analysis_plan_id": record.analysis_plan_id,
            "numeric_revision_id": record.numeric_revision_id,
            "payloads": [list(item) for item in record.payloads],
            "unit_contract_id": record.unit_contract_id,
            "association_ids": list(record.association_ids),
            "manifest_id": record.manifest_id,
        }
    if isinstance(record, ResultManifest):
        return _encode_result_manifest(record)
    if isinstance(record, ResultRevision):
        return {
            "kind": "result-revision",
            "manifest": _encode_result_manifest(record.manifest),
            "parent_result_id": record.parent_result_id,
            "revision_id": record.revision_id,
        }
    raise TypeError("Unknown lifecycle record.")


def _encode_result_manifest(record: ResultManifest, /) -> dict[str, Any]:
    return {
        "kind": "result-manifest",
        "result_id": record.result_id,
        "run_id": record.run_id,
        "fields": [list(item) for item in record.fields],
        "payloads": [list(item) for item in record.payloads],
        "evidence_ids": list(record.evidence_ids),
        "diagnostic_ids": list(record.diagnostic_ids),
        "manifest_id": record.manifest_id,
    }


def _decode_record(record: Mapping[str, Any], /) -> LifecycleRecord:
    kind = record.get("kind")
    if kind == "numeric-revision":
        value = NumericRevision(
            record["content_digest"],
            label=record["label"],
            parent_digest=record["parent_digest"],
            metadata=record["metadata"],
        )
        _identity(record, "revision_id", value.revision_id)
        return value
    if kind == "checkpoint-manifest":
        shards = tuple(_decode_shard(item) for item in record["shards"])
        value = CheckpointManifest(
            record["checkpoint_id"],
            record["analysis_plan_id"],
            record["numeric_revision_id"],
            record["execution_plan_id"],
            shards,
            complete=record["complete"],
            parent_checkpoint_id=record["parent_checkpoint_id"],
            diagnostic_ids=record["diagnostic_ids"],
        )
        _identity(record, "manifest_id", value.manifest_id)
        return value
    if kind == "analysis-plan":
        value = AnalysisPlan(
            record["analysis_plan_id"],
            record["provider_plan_id"],
            record["discretization_key"],
            record["field_layout_ids"],
            material_plan_id=record["material_plan_id"],
            constraint_ids=record["constraint_ids"],
            capability_ids=record["capability_ids"],
            model_manifest_id=record["model_manifest_id"],
        )
        _identity(record, "plan_fingerprint", value.plan_fingerprint)
        return value
    if kind == "execution-plan":
        value = ExecutionPlan(
            record["execution_plan_id"],
            record["backend"],
            record["precision_policy_id"],
            record["solver_policy_id"],
            device_mesh_id=record["device_mesh_id"],
            reduction_policy_id=record["reduction_policy_id"],
            cache_key=record["cache_key"],
        )
        _identity(record, "plan_fingerprint", value.plan_fingerprint)
        return value
    if kind == "run-record":
        value = RunRecord(
            record["run_id"],
            record["analysis_plan_id"],
            record["numeric_revision_id"],
            record["execution_plan_id"],
            record["status"],
            result_ids=record["result_ids"],
            diagnostic_ids=record["diagnostic_ids"],
            checkpoint_id=record["checkpoint_id"],
        )
        _identity(record, "record_id", value.record_id)
        return value
    if kind == "model-manifest":
        value = ModelManifest(
            record["model_id"],
            record["analysis_plan_id"],
            record["numeric_revision_id"],
            record["payloads"],
            unit_contract_id=record["unit_contract_id"],
            association_ids=record["association_ids"],
        )
        _identity(record, "manifest_id", value.manifest_id)
        return value
    if kind == "result-manifest":
        return _decode_result_manifest(record)
    if kind == "result-revision":
        nested = record["manifest"]
        if not isinstance(nested, Mapping):
            raise ArrayArchiveCorruptionError("Result revision manifest is invalid.")
        value = ResultRevision(
            _decode_result_manifest(nested),
            record["parent_result_id"],
        )
        _identity(record, "revision_id", value.revision_id)
        return value
    raise ArrayArchiveCorruptionError("Unknown lifecycle record kind.")


def _decode_shard(record: Mapping[str, Any], /) -> CheckpointShard:
    value = CheckpointShard(
        record["shard_id"],
        record["payload_digest"],
        record["byte_count"],
        record["layout_ids"],
    )
    _identity(record, "shard_fingerprint", value.shard_fingerprint)
    return value


def _decode_result_manifest(record: Mapping[str, Any], /) -> ResultManifest:
    if record.get("kind") != "result-manifest":
        raise ArrayArchiveCorruptionError("Result manifest record is invalid.")
    value = ResultManifest(
        record["result_id"],
        record["run_id"],
        record["fields"],
        record["payloads"],
        evidence_ids=record["evidence_ids"],
        diagnostic_ids=record["diagnostic_ids"],
    )
    _identity(record, "manifest_id", value.manifest_id)
    return value


def _identity(record: Mapping[str, Any], name: str, expected: str, /) -> None:
    if record.get(name) != expected:
        raise ArrayArchiveCorruptionError(f"Lifecycle {name} mismatch.")


def _record_id(record: LifecycleRecord, /) -> str:
    if isinstance(record, NumericRevision):
        return record.revision_id
    if isinstance(record, CheckpointManifest):
        return record.manifest_id
    if isinstance(record, (AnalysisPlan, ExecutionPlan)):
        return record.plan_fingerprint
    if isinstance(record, RunRecord):
        return record.record_id
    if isinstance(record, (ModelManifest, ResultManifest)):
        return record.manifest_id
    return record.revision_id


def _diagnostic_ids(record: LifecycleRecord, /) -> tuple[str, ...]:
    if isinstance(record, CheckpointManifest):
        return record.diagnostic_ids
    if isinstance(record, RunRecord):
        return record.diagnostic_ids
    if isinstance(record, ResultManifest):
        return record.diagnostic_ids
    if isinstance(record, ResultRevision):
        return record.manifest.diagnostic_ids
    return ()


register_exporter("npz", _export_npz)


__all__ = [
    "LifecycleArchive",
    "LifecycleQuery",
    "LifecycleRecord",
    "SupportBundleAuthorization",
    "SupportBundleDisclosure",
    "SampledExporter",
    "SampledField",
    "create",
    "export",
    "list_fields",
    "collection_digest",
    "payload_byte_count",
    "payload_digest",
    "open",
    "query",
    "register_exporter",
    "support_bundle",
]
