#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification._registry import SupportTuple


ChunkEncoding: TypeAlias = Literal["identity", "zlib"]
MetadataRecord: TypeAlias = tuple[tuple[str, str], ...]
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TIMESTAMP = (1 << 63) - 1


class RepositoryError(RuntimeError):
    """Base class for durable artifact repository failures."""


class RepositoryConflictError(RepositoryError):
    """Raised when an immutable or conditional repository write conflicts."""


class RepositoryCorruptionError(RepositoryError):
    """Raised when durable repository state fails validation."""


class UnsupportedRepositoryProfileError(RepositoryError):
    """Raised when a storage profile lacks required durability semantics."""


class RepositoryTransaction(StrictModule, NonTrainableState):
    """Immutable identity for one attempt-private artifact transaction."""

    provider_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    writer_id: str = eqx.field(static=True)
    attempt_id: str = eqx.field(static=True)
    base_manifest_id: str | None = eqx.field(static=True)
    base_pointer_token: str | None = eqx.field(static=True)
    started_at: int = eqx.field(static=True)
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        artifact_id: str,
        writer_id: str,
        attempt_id: str,
        /,
        *,
        base_manifest_id: str | None,
        base_pointer_token: str | None,
        started_at: int,
    ):
        provider = _identifier(provider_id, "provider_id")
        artifact = _identifier(artifact_id, "artifact_id")
        writer = _identifier(writer_id, "writer_id")
        attempt = _identifier(attempt_id, "attempt_id")
        base = _optional_digest(base_manifest_id, "base_manifest_id")
        token = (
            None
            if base_pointer_token is None
            else _opaque_token(base_pointer_token, "base_pointer_token")
        )
        started = _timestamp(started_at, "started_at")
        self.provider_id = provider
        self.artifact_id = artifact
        self.writer_id = writer
        self.attempt_id = attempt
        self.base_manifest_id = base
        self.base_pointer_token = token
        self.started_at = started
        self.transaction_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "artifact-transaction",
            "provider_id": self.provider_id,
            "artifact_id": self.artifact_id,
            "writer_id": self.writer_id,
            "attempt_id": self.attempt_id,
            "base_manifest_id": self.base_manifest_id,
            "base_pointer_token": self.base_pointer_token,
            "started_at": self.started_at,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "transaction_id": self.transaction_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> RepositoryTransaction:
        _record_kind(record, "artifact-transaction")
        value = cls(
            _required_string(record, "provider_id"),
            _required_string(record, "artifact_id"),
            _required_string(record, "writer_id"),
            _required_string(record, "attempt_id"),
            base_manifest_id=_optional_string(record.get("base_manifest_id")),
            base_pointer_token=_optional_string(record.get("base_pointer_token")),
            started_at=_required_integer(record, "started_at"),
        )
        _record_identity(record, "transaction_id", value.transaction_id)
        return value


class ChunkRecord(StrictModule, NonTrainableState):
    """Checksums, byte bounds, and immutable location of one artifact chunk."""

    transaction_id: str = eqx.field(static=True)
    logical_name: str = eqx.field(static=True)
    index: int = eqx.field(static=True)
    offset: int = eqx.field(static=True)
    plaintext_size: int = eqx.field(static=True)
    encoded_size: int = eqx.field(static=True)
    plaintext_sha256: str = eqx.field(static=True)
    encoded_sha256: str = eqx.field(static=True)
    encoding: ChunkEncoding = eqx.field(static=True)
    object_key: str = eqx.field(static=True)
    chunk_id: str = eqx.field(static=True)

    def __init__(
        self,
        transaction_id: str,
        logical_name: str,
        index: int,
        offset: int,
        plaintext_size: int,
        encoded_size: int,
        plaintext_sha256: str,
        encoded_sha256: str,
        encoding: ChunkEncoding,
        object_key: str,
        /,
    ):
        transaction = _digest(transaction_id, "transaction_id")
        logical = _identifier(logical_name, "logical_name")
        index_ = _nonnegative(index, "index")
        offset_ = _nonnegative(offset, "offset")
        plaintext_size_ = _nonnegative(plaintext_size, "plaintext_size")
        encoded_size_ = _nonnegative(encoded_size, "encoded_size")
        plaintext_digest = _digest(plaintext_sha256, "plaintext_sha256")
        encoded_digest = _digest(encoded_sha256, "encoded_sha256")
        if encoding not in ("identity", "zlib"):
            raise ValueError("Chunk encoding must be 'identity' or 'zlib'.")
        key = _object_key(object_key)
        self.transaction_id = transaction
        self.logical_name = logical
        self.index = index_
        self.offset = offset_
        self.plaintext_size = plaintext_size_
        self.encoded_size = encoded_size_
        self.plaintext_sha256 = plaintext_digest
        self.encoded_sha256 = encoded_digest
        self.encoding = encoding
        self.object_key = key
        self.chunk_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "artifact-chunk",
            "transaction_id": self.transaction_id,
            "logical_name": self.logical_name,
            "index": self.index,
            "offset": self.offset,
            "plaintext_size": self.plaintext_size,
            "encoded_size": self.encoded_size,
            "plaintext_sha256": self.plaintext_sha256,
            "encoded_sha256": self.encoded_sha256,
            "encoding": self.encoding,
            "object_key": self.object_key,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "chunk_id": self.chunk_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ChunkRecord:
        _record_kind(record, "artifact-chunk")
        encoding = _required_string(record, "encoding")
        if encoding not in ("identity", "zlib"):
            raise ValueError("Serialized chunk has an unsupported encoding.")
        value = cls(
            _required_string(record, "transaction_id"),
            _required_string(record, "logical_name"),
            _required_integer(record, "index"),
            _required_integer(record, "offset"),
            _required_integer(record, "plaintext_size"),
            _required_integer(record, "encoded_size"),
            _required_string(record, "plaintext_sha256"),
            _required_string(record, "encoded_sha256"),
            encoding,
            _required_string(record, "object_key"),
        )
        _record_identity(record, "chunk_id", value.chunk_id)
        return value


class ArtifactManifest(StrictModule, NonTrainableState):
    """Immutable complete content manifest and sole readable commit target."""

    provider_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    transaction_id: str = eqx.field(static=True)
    base_manifest_id: str | None = eqx.field(static=True)
    chunks: tuple[ChunkRecord, ...] = eqx.field(static=True)
    metadata: MetadataRecord = eqx.field(static=True)
    committed_at: int = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        artifact_id: str,
        transaction_id: str,
        base_manifest_id: str | None,
        chunks: Sequence[ChunkRecord],
        /,
        *,
        metadata: Mapping[str, str] | Sequence[tuple[str, str]] = (),
        committed_at: int,
    ):
        provider = _identifier(provider_id, "provider_id")
        artifact = _identifier(artifact_id, "artifact_id")
        transaction = _digest(transaction_id, "transaction_id")
        base = _optional_digest(base_manifest_id, "base_manifest_id")
        chunks_ = tuple(
            sorted(tuple(chunks), key=lambda item: (item.logical_name, item.index))
        )
        if not chunks_ or not all(isinstance(item, ChunkRecord) for item in chunks_):
            raise TypeError("Artifact manifests require typed, non-empty chunks.")
        if any(item.transaction_id != transaction for item in chunks_):
            raise ValueError("Every manifest chunk must belong to its transaction.")
        if len({item.chunk_id for item in chunks_}) != len(chunks_):
            raise ValueError("Artifact manifest chunk IDs must be unique.")
        _validate_chunk_partition(chunks_)
        metadata_ = _metadata(metadata)
        committed = _timestamp(committed_at, "committed_at")
        self.provider_id = provider
        self.artifact_id = artifact
        self.transaction_id = transaction
        self.base_manifest_id = base
        self.chunks = chunks_
        self.metadata = metadata_
        self.committed_at = committed
        self.complete = True
        self.manifest_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "artifact-manifest",
            "provider_id": self.provider_id,
            "artifact_id": self.artifact_id,
            "transaction_id": self.transaction_id,
            "base_manifest_id": self.base_manifest_id,
            "chunks": [chunk.to_record() for chunk in self.chunks],
            "metadata": [list(item) for item in self.metadata],
            "committed_at": self.committed_at,
            "complete": True,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "manifest_id": self.manifest_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ArtifactManifest:
        _record_kind(record, "artifact-manifest")
        if record.get("complete") is not True:
            raise ValueError("Only complete artifact manifests are readable.")
        raw_chunks = record["chunks"]
        raw_metadata = record.get("metadata", ())
        if not isinstance(raw_chunks, Sequence) or isinstance(raw_chunks, (str, bytes)):
            raise TypeError("Serialized manifest chunks must be a sequence.")
        if not isinstance(raw_metadata, (Mapping, Sequence)) or isinstance(
            raw_metadata, (str, bytes)
        ):
            raise TypeError("Serialized manifest metadata must be records.")
        value = cls(
            _required_string(record, "provider_id"),
            _required_string(record, "artifact_id"),
            _required_string(record, "transaction_id"),
            _optional_string(record.get("base_manifest_id")),
            tuple(ChunkRecord.from_record(item) for item in raw_chunks),
            metadata=raw_metadata,
            committed_at=_required_integer(record, "committed_at"),
        )
        _record_identity(record, "manifest_id", value.manifest_id)
        return value


class LeaseRecord(StrictModule, NonTrainableState):
    """Time-bounded pin preventing collection of an artifact's roots."""

    provider_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    holder_id: str = eqx.field(static=True)
    lease_id: str = eqx.field(static=True)
    issued_at: int = eqx.field(static=True)
    expires_at: int = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        artifact_id: str,
        holder_id: str,
        lease_id: str,
        issued_at: int,
        expires_at: int,
        /,
    ):
        provider = _identifier(provider_id, "provider_id")
        artifact = _identifier(artifact_id, "artifact_id")
        holder = _identifier(holder_id, "holder_id")
        lease = _identifier(lease_id, "lease_id")
        issued = _timestamp(issued_at, "issued_at")
        expires = _timestamp(expires_at, "expires_at")
        if expires <= issued:
            raise ValueError("Lease expiry must be later than issuance.")
        self.provider_id = provider
        self.artifact_id = artifact
        self.holder_id = holder
        self.lease_id = lease
        self.issued_at = issued
        self.expires_at = expires
        self.record_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "artifact-lease",
            "provider_id": self.provider_id,
            "artifact_id": self.artifact_id,
            "holder_id": self.holder_id,
            "lease_id": self.lease_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "record_id": self.record_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> LeaseRecord:
        _record_kind(record, "artifact-lease")
        value = cls(
            _required_string(record, "provider_id"),
            _required_string(record, "artifact_id"),
            _required_string(record, "holder_id"),
            _required_string(record, "lease_id"),
            _required_integer(record, "issued_at"),
            _required_integer(record, "expires_at"),
        )
        _record_identity(record, "record_id", value.record_id)
        return value


class LegalHoldRecord(StrictModule, NonTrainableState):
    """Explicit non-expiring legal pin for one artifact."""

    provider_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    hold_id: str = eqx.field(static=True)
    authority: str = eqx.field(static=True)
    placed_at: int = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        artifact_id: str,
        hold_id: str,
        authority: str,
        placed_at: int,
        /,
    ):
        provider = _identifier(provider_id, "provider_id")
        artifact = _identifier(artifact_id, "artifact_id")
        hold = _identifier(hold_id, "hold_id")
        authority_ = _identifier(authority, "authority")
        placed = _timestamp(placed_at, "placed_at")
        self.provider_id = provider
        self.artifact_id = artifact
        self.hold_id = hold
        self.authority = authority_
        self.placed_at = placed
        self.record_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "artifact-legal-hold",
            "provider_id": self.provider_id,
            "artifact_id": self.artifact_id,
            "hold_id": self.hold_id,
            "authority": self.authority,
            "placed_at": self.placed_at,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "record_id": self.record_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> LegalHoldRecord:
        _record_kind(record, "artifact-legal-hold")
        value = cls(
            _required_string(record, "provider_id"),
            _required_string(record, "artifact_id"),
            _required_string(record, "hold_id"),
            _required_string(record, "authority"),
            _required_integer(record, "placed_at"),
        )
        _record_identity(record, "record_id", value.record_id)
        return value


class RetentionPolicy(StrictModule, NonTrainableState):
    """Minimum-age and history bounds used only for unreachable roots."""

    keep_latest_commits: int = eqx.field(static=True)
    minimum_age_seconds: int = eqx.field(static=True)
    abandoned_attempt_grace_seconds: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        keep_latest_commits: int = 1,
        minimum_age_seconds: int = 0,
        abandoned_attempt_grace_seconds: int = 3600,
    ):
        keep = _nonnegative(keep_latest_commits, "keep_latest_commits")
        age = _nonnegative(minimum_age_seconds, "minimum_age_seconds")
        grace = _nonnegative(
            abandoned_attempt_grace_seconds, "abandoned_attempt_grace_seconds"
        )
        self.keep_latest_commits = keep
        self.minimum_age_seconds = age
        self.abandoned_attempt_grace_seconds = grace
        self.policy_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "artifact-retention-policy",
            "keep_latest_commits": self.keep_latest_commits,
            "minimum_age_seconds": self.minimum_age_seconds,
            "abandoned_attempt_grace_seconds": self.abandoned_attempt_grace_seconds,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "policy_id": self.policy_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> RetentionPolicy:
        _record_kind(record, "artifact-retention-policy")
        value = cls(
            keep_latest_commits=_required_integer(record, "keep_latest_commits"),
            minimum_age_seconds=_required_integer(record, "minimum_age_seconds"),
            abandoned_attempt_grace_seconds=_required_integer(
                record, "abandoned_attempt_grace_seconds"
            ),
        )
        _record_identity(record, "policy_id", value.policy_id)
        return value


class TombstoneRecord(StrictModule, NonTrainableState):
    """Immutable, delayed authorization to make one artifact unreachable."""

    provider_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    created_at: int = eqx.field(static=True)
    eligible_at: int = eqx.field(static=True)
    tombstone_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        artifact_id: str,
        reason: str,
        created_at: int,
        eligible_at: int,
        /,
    ):
        provider = _identifier(provider_id, "provider_id")
        artifact = _identifier(artifact_id, "artifact_id")
        reason_ = str(reason).strip()
        if not reason_ or len(reason_.encode("utf-8")) > 4096:
            raise ValueError("Tombstone reason must be nonempty and at most 4096 bytes.")
        created = _timestamp(created_at, "created_at")
        eligible = _timestamp(eligible_at, "eligible_at")
        if eligible < created:
            raise ValueError("Tombstone eligibility cannot precede creation.")
        self.provider_id = provider
        self.artifact_id = artifact
        self.reason = reason_
        self.created_at = created
        self.eligible_at = eligible
        self.tombstone_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "artifact-tombstone",
            "provider_id": self.provider_id,
            "artifact_id": self.artifact_id,
            "reason": self.reason,
            "created_at": self.created_at,
            "eligible_at": self.eligible_at,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "tombstone_id": self.tombstone_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> TombstoneRecord:
        _record_kind(record, "artifact-tombstone")
        value = cls(
            _required_string(record, "provider_id"),
            _required_string(record, "artifact_id"),
            _required_string(record, "reason"),
            _required_integer(record, "created_at"),
            _required_integer(record, "eligible_at"),
        )
        _record_identity(record, "tombstone_id", value.tombstone_id)
        return value


class GarbageCollectionReport(StrictModule, NonTrainableState):
    """Deterministic account of roots removed by one collection pass."""

    provider_id: str = eqx.field(static=True)
    collected_at: int = eqx.field(static=True)
    removed_attempt_ids: tuple[str, ...] = eqx.field(static=True)
    removed_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    expired_lease_ids: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        collected_at: int,
        removed_attempt_ids: Sequence[str],
        removed_artifact_ids: Sequence[str],
        expired_lease_ids: Sequence[str],
        /,
    ):
        provider = _identifier(provider_id, "provider_id")
        collected = _timestamp(collected_at, "collected_at")
        attempts = _identifiers(removed_attempt_ids, "removed_attempt_ids")
        artifacts = _identifiers(removed_artifact_ids, "removed_artifact_ids")
        leases = _identifiers(expired_lease_ids, "expired_lease_ids")
        self.provider_id = provider
        self.collected_at = collected
        self.removed_attempt_ids = attempts
        self.removed_artifact_ids = artifacts
        self.expired_lease_ids = leases
        self.report_id = canonical_fingerprint(
            {
                "kind": "artifact-garbage-collection",
                "provider_id": provider,
                "collected_at": collected,
                "removed_attempt_ids": list(attempts),
                "removed_artifact_ids": list(artifacts),
                "expired_lease_ids": list(leases),
            }
        )


class ArtifactRepository(Protocol):
    """Provider-neutral transactional repository contract."""

    provider_id: str
    support_tuple: SupportTuple
    maximum_chunk_bytes: int

    def begin(
        self,
        artifact_id: str,
        writer_id: str,
        /,
        *,
        attempt_id: str | None = None,
        started_at: int | None = None,
    ) -> RepositoryTransaction: ...

    def write_chunk(
        self,
        transaction: RepositoryTransaction,
        logical_name: str,
        index: int,
        offset: int,
        payload: bytes | bytearray | memoryview,
        /,
        *,
        encoding: ChunkEncoding = "identity",
    ) -> ChunkRecord: ...

    def commit(
        self,
        transaction: RepositoryTransaction,
        chunks: Sequence[ChunkRecord],
        /,
        *,
        metadata: Mapping[str, str] | Sequence[tuple[str, str]] = (),
        committed_at: int | None = None,
    ) -> ArtifactManifest: ...

    def get_manifest(self, artifact_id: str, /) -> ArtifactManifest: ...

    def read_chunk(
        self,
        manifest: ArtifactManifest,
        chunk: ChunkRecord,
        /,
        *,
        maximum_plaintext_bytes: int | None = None,
    ) -> bytes: ...

    def acquire_lease(
        self,
        artifact_id: str,
        holder_id: str,
        /,
        *,
        expires_at: int,
        lease_id: str | None = None,
        issued_at: int | None = None,
    ) -> LeaseRecord: ...

    def release_lease(self, lease: LeaseRecord, /) -> None: ...

    def place_legal_hold(
        self,
        artifact_id: str,
        authority: str,
        /,
        *,
        hold_id: str | None = None,
        placed_at: int | None = None,
    ) -> LegalHoldRecord: ...

    def release_legal_hold(self, hold: LegalHoldRecord, /) -> None: ...

    def set_retention(self, artifact_id: str, policy: RetentionPolicy, /) -> None: ...

    def tombstone(
        self,
        artifact_id: str,
        reason: str,
        /,
        *,
        created_at: int | None = None,
        eligible_at: int | None = None,
    ) -> TombstoneRecord: ...

    def collect_garbage(
        self,
        /,
        *,
        now: int | None = None,
        default_policy: RetentionPolicy | None = None,
    ) -> GarbageCollectionReport: ...


def _validate_chunk_partition(chunks: Sequence[ChunkRecord], /) -> None:
    names = sorted({item.logical_name for item in chunks})
    for name in names:
        records = tuple(item for item in chunks if item.logical_name == name)
        expected_offset = 0
        for expected_index, item in enumerate(records):
            if item.index != expected_index:
                raise ValueError(f"Chunk indexes for {name!r} contain a hole.")
            if item.offset != expected_offset:
                relation = "overlap" if item.offset < expected_offset else "hole"
                raise ValueError(f"Chunk byte ranges for {name!r} contain a {relation}.")
            expected_offset += item.plaintext_size


def _metadata(value: Mapping[str, str] | Sequence[tuple[str, str]], /) -> MetadataRecord:
    records = value.items() if isinstance(value, Mapping) else value
    normalized = tuple(
        sorted(
            (
                _identifier(name, "metadata key"),
                _metadata_value(item),
            )
            for name, item in records
        )
    )
    if len({name for name, _ in normalized}) != len(normalized):
        raise ValueError("Artifact metadata keys must be unique.")
    return normalized


def _metadata_value(value: object, /) -> str:
    if not isinstance(value, str):
        raise TypeError("Artifact metadata values must be strings.")
    if len(value.encode("utf-8")) > 4096:
        raise ValueError("Artifact metadata values must be at most 4096 bytes.")
    return value


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be a nonempty path-safe identifier of at most 256 characters."
        )
    return value


def _opaque_token(value: str, name: str, /) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 1024
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValueError(f"{name} must be a bounded nonempty opaque token.")
    return value


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    normalized = tuple(sorted(_identifier(item, name) for item in values))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique.")
    return normalized


def _digest(value: str, name: str, /) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _optional_digest(value: str | None, name: str, /) -> str | None:
    return None if value is None else _digest(value, name)


def _object_key(value: str, /) -> str:
    if not isinstance(value, str) or value.startswith("/") or "\\" in value:
        raise ValueError("Chunk object keys must be relative canonical paths.")
    segments = value.split("/")
    if not segments or any(
        not segment or segment in (".", "..") or _IDENTIFIER.fullmatch(segment) is None
        for segment in segments
    ):
        raise ValueError("Chunk object keys contain an unsafe path segment.")
    return value


def _timestamp(value: int, name: str, /) -> int:
    normalized = int(value)
    if normalized < 0 or normalized > _MAX_TIMESTAMP:
        raise ValueError(f"{name} must be a non-negative signed 64-bit timestamp.")
    return normalized


def _nonnegative(value: int, name: str, /) -> int:
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return normalized


def _record_kind(record: Mapping[str, object], expected: str, /) -> None:
    if not isinstance(record, Mapping) or record.get("kind") != expected:
        raise ValueError(f"Expected serialized record kind {expected!r}.")


def _record_identity(record: Mapping[str, object], name: str, expected: str, /) -> None:
    if record.get(name) != expected:
        raise ValueError(f"Serialized {name} does not match record content.")


def _required_string(record: Mapping[str, object], name: str, /) -> str:
    value = record[name]
    if not isinstance(value, str):
        raise TypeError(f"Serialized {name} must be a string.")
    return value


def _required_integer(record: Mapping[str, object], name: str, /) -> int:
    value = record[name]
    if type(value) is not int:
        raise TypeError(f"Serialized {name} must be an integer.")
    return value


def _optional_string(value: object, /) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Serialized optional value must be a string or null.")
    return value


__all__ = [
    "ArtifactManifest",
    "ArtifactRepository",
    "ChunkEncoding",
    "ChunkRecord",
    "GarbageCollectionReport",
    "LeaseRecord",
    "LegalHoldRecord",
    "RepositoryConflictError",
    "RepositoryCorruptionError",
    "RepositoryError",
    "RepositoryTransaction",
    "RetentionPolicy",
    "TombstoneRecord",
    "UnsupportedRepositoryProfileError",
]
