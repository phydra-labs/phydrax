#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import shutil
import stat
import threading
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification._registry import SupportTuple
from ._chunk_repository import (
    _identifier,
    _object_key,
    ArtifactManifest,
    ArtifactRepository,
    ChunkEncoding,
    ChunkRecord,
    GarbageCollectionReport,
    LeaseRecord,
    LegalHoldRecord,
    RepositoryConflictError,
    RepositoryCorruptionError,
    RepositoryError,
    RepositoryTransaction,
    RetentionPolicy,
    TombstoneRecord,
    UnsupportedRepositoryProfileError,
)


_METADATA_LIMIT = 8 * 1024 * 1024
_POINTER_LIMIT = 64 * 1024
_DEFAULT_CHUNK_LIMIT = 64 * 1024 * 1024
FailureInjector = Callable[[str], None]


class HPCFilesystemProfile(StrictModule, NonTrainableState):
    """Declared HPC filesystem durability semantics used for qualification."""

    provider_id: str = eqx.field(static=True)
    filesystem: str = eqx.field(static=True)
    atomic_rename_same_filesystem: bool = eqx.field(static=True)
    file_fsync: bool = eqx.field(static=True)
    directory_fsync: bool = eqx.field(static=True)
    advisory_locking: bool = eqx.field(static=True)
    attempt_private_staging: bool = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        filesystem: str,
        /,
        *,
        atomic_rename_same_filesystem: bool,
        file_fsync: bool,
        directory_fsync: bool,
        advisory_locking: bool,
        attempt_private_staging: bool,
    ):
        provider = _identifier(provider_id, "provider_id")
        filesystem_ = _identifier(filesystem, "filesystem")
        self.provider_id = provider
        self.filesystem = filesystem_
        self.atomic_rename_same_filesystem = bool(atomic_rename_same_filesystem)
        self.file_fsync = bool(file_fsync)
        self.directory_fsync = bool(directory_fsync)
        self.advisory_locking = bool(advisory_locking)
        self.attempt_private_staging = bool(attempt_private_staging)
        self.profile_id = canonical_fingerprint(
            {
                "kind": "hpc-filesystem-profile",
                "provider_id": provider,
                "filesystem": filesystem_,
                "atomic_rename_same_filesystem": self.atomic_rename_same_filesystem,
                "file_fsync": self.file_fsync,
                "directory_fsync": self.directory_fsync,
                "advisory_locking": self.advisory_locking,
                "attempt_private_staging": self.attempt_private_staging,
            }
        )

    def require_transactional_support(self) -> None:
        missing = tuple(
            name
            for name, present in (
                ("same-filesystem atomic rename", self.atomic_rename_same_filesystem),
                ("file fsync", self.file_fsync),
                ("directory fsync", self.directory_fsync),
                ("advisory locking", self.advisory_locking),
                ("attempt-private staging", self.attempt_private_staging),
            )
            if not present
        )
        if missing:
            raise UnsupportedRepositoryProfileError(
                "HPC filesystem profile lacks required semantics: " + ", ".join(missing)
            )


class POSIXRepositoryPolicy(StrictModule, NonTrainableState):
    """Qualified filesystem profile and bounded-object policy."""

    filesystem_profile: HPCFilesystemProfile
    maximum_chunk_bytes: int = eqx.field(static=True)
    maximum_metadata_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        filesystem_profile: HPCFilesystemProfile,
        /,
        *,
        maximum_chunk_bytes: int = _DEFAULT_CHUNK_LIMIT,
        maximum_metadata_bytes: int = _METADATA_LIMIT,
    ):
        if not isinstance(filesystem_profile, HPCFilesystemProfile):
            raise TypeError("filesystem_profile must be HPCFilesystemProfile.")
        filesystem_profile.require_transactional_support()
        chunk_limit = _positive(maximum_chunk_bytes, "maximum_chunk_bytes")
        metadata_limit = _positive(maximum_metadata_bytes, "maximum_metadata_bytes")
        self.filesystem_profile = filesystem_profile
        self.maximum_chunk_bytes = chunk_limit
        self.maximum_metadata_bytes = metadata_limit
        self.policy_id = canonical_fingerprint(
            {
                "kind": "posix-repository-policy",
                "filesystem_profile_id": filesystem_profile.profile_id,
                "maximum_chunk_bytes": chunk_limit,
                "maximum_metadata_bytes": metadata_limit,
            }
        )


class ObjectStoreProfile(StrictModule, NonTrainableState):
    """Exact S3-compatible semantics required by the object repository."""

    provider_id: str = eqx.field(static=True)
    conditional_create: bool = eqx.field(static=True)
    conditional_replace: bool = eqx.field(static=True)
    strongly_consistent_reads: bool = eqx.field(static=True)
    strongly_consistent_listing: bool = eqx.field(static=True)
    multipart_free_objects: bool = eqx.field(static=True)
    maximum_object_bytes: int = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        /,
        *,
        conditional_create: bool,
        conditional_replace: bool,
        strongly_consistent_reads: bool,
        strongly_consistent_listing: bool,
        multipart_free_objects: bool,
        maximum_object_bytes: int,
    ):
        provider = _identifier(provider_id, "provider_id")
        maximum = _positive(maximum_object_bytes, "maximum_object_bytes")
        self.provider_id = provider
        self.conditional_create = bool(conditional_create)
        self.conditional_replace = bool(conditional_replace)
        self.strongly_consistent_reads = bool(strongly_consistent_reads)
        self.strongly_consistent_listing = bool(strongly_consistent_listing)
        self.multipart_free_objects = bool(multipart_free_objects)
        self.maximum_object_bytes = maximum
        self.profile_id = canonical_fingerprint(
            {
                "kind": "object-store-profile",
                "provider_id": provider,
                "conditional_create": self.conditional_create,
                "conditional_replace": self.conditional_replace,
                "strongly_consistent_reads": self.strongly_consistent_reads,
                "strongly_consistent_listing": self.strongly_consistent_listing,
                "multipart_free_objects": self.multipart_free_objects,
                "maximum_object_bytes": maximum,
            }
        )

    def require_transactional_support(self) -> None:
        missing = tuple(
            name
            for name, present in (
                ("conditional create", self.conditional_create),
                ("conditional replace", self.conditional_replace),
                ("strongly consistent reads", self.strongly_consistent_reads),
                ("strongly consistent listing", self.strongly_consistent_listing),
                ("multipart-free bounded objects", self.multipart_free_objects),
            )
            if not present
        )
        if missing:
            raise UnsupportedRepositoryProfileError(
                "Object-store profile lacks required semantics: " + ", ".join(missing)
            )


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    """Opaque conditional token and byte size for one object."""

    key: str
    etag: str
    size: int


@dataclass(frozen=True, slots=True)
class ObjectValue:
    """One bounded immutable object read."""

    metadata: ObjectMetadata
    data: bytes


class ObjectNotFoundError(RepositoryError):
    """Raised when an object client key is absent."""


class ObjectPreconditionError(RepositoryConflictError):
    """Raised when an object conditional write or delete loses a race."""


class ConditionalObjectClient(Protocol):
    """Narrow whole-object API required from an S3-compatible client adapter."""

    def create_object(self, key: str, data: bytes, /) -> ObjectMetadata: ...

    def replace_object(
        self, key: str, data: bytes, expected_etag: str, /
    ) -> ObjectMetadata: ...

    def read_object(self, key: str, /, *, maximum_bytes: int) -> ObjectValue: ...

    def delete_object(self, key: str, /, *, expected_etag: str | None = None) -> None: ...

    def list_objects(self, prefix: str, /) -> tuple[ObjectMetadata, ...]: ...


class InMemoryConditionalObjectClient:
    """Standards-faithful conditional object client for qualification scenarios."""

    def __init__(self, /, *, maximum_object_bytes: int):
        self.maximum_object_bytes = _positive(
            maximum_object_bytes, "maximum_object_bytes"
        )
        self._objects: dict[str, tuple[bytes, str]] = {}
        self._generation = 0
        self._lock = threading.RLock()
        self._read_requests: list[tuple[str, int]] = []

    @property
    def read_requests(self) -> tuple[tuple[str, int], ...]:
        return tuple(self._read_requests)

    def create_object(self, key: str, data: bytes, /) -> ObjectMetadata:
        key_ = _object_key(key)
        payload = _bounded_bytes(data, self.maximum_object_bytes, "object")
        with self._lock:
            if key_ in self._objects:
                raise ObjectPreconditionError(f"Object {key_!r} already exists.")
            etag = self._next_etag(payload)
            self._objects[key_] = (payload, etag)
            return ObjectMetadata(key_, etag, len(payload))

    def replace_object(
        self, key: str, data: bytes, expected_etag: str, /
    ) -> ObjectMetadata:
        key_ = _object_key(key)
        expected = _identifier(expected_etag, "expected_etag")
        payload = _bounded_bytes(data, self.maximum_object_bytes, "object")
        with self._lock:
            current = self._objects.get(key_)
            if current is None or current[1] != expected:
                raise ObjectPreconditionError(
                    f"Object {key_!r} conditional replace failed."
                )
            etag = self._next_etag(payload)
            self._objects[key_] = (payload, etag)
            return ObjectMetadata(key_, etag, len(payload))

    def read_object(self, key: str, /, *, maximum_bytes: int) -> ObjectValue:
        key_ = _object_key(key)
        maximum = _nonnegative(maximum_bytes, "maximum_bytes")
        with self._lock:
            self._read_requests.append((key_, maximum))
            current = self._objects.get(key_)
            if current is None:
                raise ObjectNotFoundError(f"Object {key_!r} does not exist.")
            payload, etag = current
            if len(payload) > maximum:
                raise RepositoryCorruptionError(
                    f"Object {key_!r} exceeds its bounded read limit."
                )
            return ObjectValue(ObjectMetadata(key_, etag, len(payload)), payload)

    def delete_object(self, key: str, /, *, expected_etag: str | None = None) -> None:
        key_ = _object_key(key)
        with self._lock:
            current = self._objects.get(key_)
            if current is None:
                raise ObjectNotFoundError(f"Object {key_!r} does not exist.")
            if expected_etag is not None and current[1] != expected_etag:
                raise ObjectPreconditionError(
                    f"Object {key_!r} conditional delete failed."
                )
            del self._objects[key_]

    def list_objects(self, prefix: str, /) -> tuple[ObjectMetadata, ...]:
        prefix_ = _object_prefix(prefix)
        with self._lock:
            return tuple(
                ObjectMetadata(key, etag, len(payload))
                for key, (payload, etag) in sorted(self._objects.items())
                if key.startswith(prefix_)
            )

    def _next_etag(self, payload: bytes, /) -> str:
        self._generation += 1
        return f"e{self._generation:x}-{hashlib.sha256(payload).hexdigest()}"


class POSIXArtifactRepository:
    """Crash-consistent POSIX repository with attempt-private immutable roots."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        policy: POSIXRepositoryPolicy,
        /,
        *,
        failure_injector: FailureInjector | None = None,
    ):
        if not isinstance(policy, POSIXRepositoryPolicy):
            raise TypeError("policy must be POSIXRepositoryPolicy.")
        policy.filesystem_profile.require_transactional_support()
        self.root = Path(root)
        self.policy = policy
        self.provider_id = policy.filesystem_profile.provider_id
        self.maximum_chunk_bytes = policy.maximum_chunk_bytes
        self.maximum_metadata_bytes = policy.maximum_metadata_bytes
        self.failure_injector = failure_injector
        self.support_tuple = SupportTuple(
            "artifact.repository",
            {
                "provider": self.provider_id,
                "transport": "posix",
                "profile_id": policy.filesystem_profile.profile_id,
                "policy_id": policy.policy_id,
                "maximum_chunk_bytes": self.maximum_chunk_bytes,
            },
        )
        self._initialize()

    def begin(
        self,
        artifact_id: str,
        writer_id: str,
        /,
        *,
        attempt_id: str | None = None,
        started_at: int | None = None,
    ) -> RepositoryTransaction:
        artifact = _identifier(artifact_id, "artifact_id")
        writer = _identifier(writer_id, "writer_id")
        attempt = (
            secrets.token_hex(16)
            if attempt_id is None
            else _identifier(attempt_id, "attempt_id")
        )
        with self._exclusive_lock():
            pointer = self._read_pointer_optional(artifact)
            base = None if pointer is None else str(pointer["manifest_id"])
            transaction = RepositoryTransaction(
                self.provider_id,
                artifact,
                writer,
                attempt,
                base_manifest_id=base,
                base_pointer_token=base,
                started_at=_now() if started_at is None else started_at,
            )
            root = self._attempt_path(attempt)
            staging = self.root / "staging" / f"{attempt}.{secrets.token_hex(16)}.attempt"
            staging.mkdir(mode=0o700)
            try:
                (staging / "chunks").mkdir(mode=0o700)
                self._atomic_create_immutable(
                    staging / "transaction.json",
                    _json_bytes(transaction.to_record(), self.maximum_metadata_bytes),
                )
                _fsync_directory(staging / "chunks")
                _fsync_directory(staging)
                if root.exists():
                    raise RepositoryConflictError(
                        f"Repository attempt {attempt!r} already exists."
                    )
                os.rename(staging, root)
                _fsync_directory(self.root / "staging")
                _fsync_directory(self.root / "roots")
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
                    _fsync_directory(self.root / "staging")
            return transaction

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
    ) -> ChunkRecord:
        self._validate_transaction(transaction)
        logical = _identifier(logical_name, "logical_name")
        index_ = _nonnegative(index, "index")
        offset_ = _nonnegative(offset, "offset")
        plaintext = _bounded_bytes(payload, self.maximum_chunk_bytes, "chunk plaintext")
        encoded = _encode_chunk(plaintext, encoding)
        if len(encoded) > self.maximum_chunk_bytes:
            raise ValueError("Encoded chunk exceeds maximum_chunk_bytes.")
        key = self._chunk_key(transaction.attempt_id, logical, index_)
        record = ChunkRecord(
            transaction.transaction_id,
            logical,
            index_,
            offset_,
            len(plaintext),
            len(encoded),
            hashlib.sha256(plaintext).hexdigest(),
            hashlib.sha256(encoded).hexdigest(),
            encoding,
            key,
        )
        self._atomic_create_immutable(self.root / key, encoded)
        return record

    def commit(
        self,
        transaction: RepositoryTransaction,
        chunks: Sequence[ChunkRecord],
        /,
        *,
        metadata: Mapping[str, str] | Sequence[tuple[str, str]] = (),
        committed_at: int | None = None,
    ) -> ArtifactManifest:
        self._validate_transaction(transaction)
        manifest = ArtifactManifest(
            self.provider_id,
            transaction.artifact_id,
            transaction.transaction_id,
            transaction.base_manifest_id,
            chunks,
            metadata=metadata,
            committed_at=_now() if committed_at is None else committed_at,
        )
        self._validate_staged_manifest(transaction, manifest)
        manifest_payload = _json_bytes(manifest.to_record(), self.maximum_metadata_bytes)
        manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
        marker = _commit_marker(transaction, manifest, manifest_sha)
        marker_payload = _json_bytes(marker, self.maximum_metadata_bytes)
        pointer = _pointer_record(transaction, manifest, manifest_sha)
        pointer_payload = _json_bytes(pointer, _POINTER_LIMIT)
        attempt = self._attempt_path(transaction.attempt_id)

        with self._exclusive_lock():
            current = self._read_pointer_optional(transaction.artifact_id)
            if (
                current is not None
                and current.get("transaction_id") == transaction.transaction_id
            ):
                raise RepositoryConflictError("Transaction is already committed.")
            current_manifest_id = None if current is None else str(current["manifest_id"])
            if current_manifest_id != transaction.base_manifest_id:
                raise RepositoryConflictError(
                    "Artifact pointer changed after this transaction began."
                )
            self._fail("before_manifest")
            self._create_or_verify_immutable(attempt / "manifest.json", manifest_payload)
            _fsync_directory(attempt)
            self._fail("after_manifest")
            self._fail("before_commit_marker")
            self._create_or_verify_immutable(attempt / "COMMIT", marker_payload)
            _fsync_directory(attempt)
            self._fail("after_commit_marker")
            self._fail("before_pointer")
            self._replace_pointer(transaction.artifact_id, pointer_payload)
            self._fail("after_pointer")
        return manifest

    def get_manifest(self, artifact_id: str, /) -> ArtifactManifest:
        artifact = _identifier(artifact_id, "artifact_id")
        pointer = self._read_pointer_optional(artifact)
        if pointer is None:
            raise ObjectNotFoundError(f"Artifact {artifact!r} does not exist.")
        return self._manifest_from_pointer(pointer, artifact)

    def read_chunk(
        self,
        manifest: ArtifactManifest,
        chunk: ChunkRecord,
        /,
        *,
        maximum_plaintext_bytes: int | None = None,
    ) -> bytes:
        self._validate_manifest_argument(manifest, chunk)
        maximum = (
            self.maximum_chunk_bytes
            if maximum_plaintext_bytes is None
            else _positive(maximum_plaintext_bytes, "maximum_plaintext_bytes")
        )
        if chunk.plaintext_size > maximum:
            raise RepositoryCorruptionError("Chunk exceeds requested plaintext bound.")
        encoded = _read_bounded_file(
            self.root / chunk.object_key,
            min(self.maximum_chunk_bytes, chunk.encoded_size),
        )
        return _decode_and_validate_chunk(chunk, encoded, maximum)

    def read_bytes(
        self,
        artifact_id: str,
        logical_name: str,
        /,
        *,
        maximum_bytes: int,
    ) -> bytes:
        manifest = self.get_manifest(artifact_id)
        logical = _identifier(logical_name, "logical_name")
        chunks = tuple(item for item in manifest.chunks if item.logical_name == logical)
        total = sum(item.plaintext_size for item in chunks)
        maximum = _positive(maximum_bytes, "maximum_bytes")
        if not chunks:
            raise ObjectNotFoundError(f"Artifact payload {logical!r} does not exist.")
        if total > maximum:
            raise RepositoryCorruptionError(
                "Artifact payload exceeds requested byte bound."
            )
        return b"".join(
            self.read_chunk(manifest, item, maximum_plaintext_bytes=maximum)
            for item in chunks
        )

    def acquire_lease(
        self,
        artifact_id: str,
        holder_id: str,
        /,
        *,
        expires_at: int,
        lease_id: str | None = None,
        issued_at: int | None = None,
    ) -> LeaseRecord:
        with self._exclusive_lock():
            artifact = _identifier(artifact_id, "artifact_id")
            self.get_manifest(artifact)
            lease = LeaseRecord(
                self.provider_id,
                artifact,
                holder_id,
                secrets.token_hex(16) if lease_id is None else lease_id,
                _now() if issued_at is None else issued_at,
                expires_at,
            )
            directory = self.root / "leases" / artifact
            directory.mkdir(mode=0o700, exist_ok=True)
            self._atomic_create_immutable(
                directory / f"{lease.lease_id}.json",
                _json_bytes(lease.to_record(), self.maximum_metadata_bytes),
            )
            _fsync_directory(directory)
            return lease

    def release_lease(self, lease: LeaseRecord, /) -> None:
        with self._exclusive_lock():
            if (
                not isinstance(lease, LeaseRecord)
                or lease.provider_id != self.provider_id
            ):
                raise TypeError("lease must belong to this repository provider.")
            path = self.root / "leases" / lease.artifact_id / f"{lease.lease_id}.json"
            persisted = LeaseRecord.from_record(
                _read_json_file(path, self.maximum_metadata_bytes)
            )
            if persisted.record_id != lease.record_id:
                raise RepositoryConflictError("Lease release record does not match.")
            self._unlink_existing(path)

    def place_legal_hold(
        self,
        artifact_id: str,
        authority: str,
        /,
        *,
        hold_id: str | None = None,
        placed_at: int | None = None,
    ) -> LegalHoldRecord:
        with self._exclusive_lock():
            artifact = _identifier(artifact_id, "artifact_id")
            self.get_manifest(artifact)
            hold = LegalHoldRecord(
                self.provider_id,
                artifact,
                secrets.token_hex(16) if hold_id is None else hold_id,
                authority,
                _now() if placed_at is None else placed_at,
            )
            directory = self.root / "holds" / artifact
            directory.mkdir(mode=0o700, exist_ok=True)
            self._atomic_create_immutable(
                directory / f"{hold.hold_id}.json",
                _json_bytes(hold.to_record(), self.maximum_metadata_bytes),
            )
            _fsync_directory(directory)
            return hold

    def release_legal_hold(self, hold: LegalHoldRecord, /) -> None:
        with self._exclusive_lock():
            if (
                not isinstance(hold, LegalHoldRecord)
                or hold.provider_id != self.provider_id
            ):
                raise TypeError("hold must belong to this repository provider.")
            path = self.root / "holds" / hold.artifact_id / f"{hold.hold_id}.json"
            persisted = LegalHoldRecord.from_record(
                _read_json_file(path, self.maximum_metadata_bytes)
            )
            if persisted.record_id != hold.record_id:
                raise RepositoryConflictError("Legal-hold release record does not match.")
            self._unlink_existing(path)

    def set_retention(self, artifact_id: str, policy: RetentionPolicy, /) -> None:
        with self._exclusive_lock():
            artifact = _identifier(artifact_id, "artifact_id")
            if not isinstance(policy, RetentionPolicy):
                raise TypeError("policy must be RetentionPolicy.")
            self.get_manifest(artifact)
            payload = _json_bytes(
                {
                    "kind": "artifact-retention",
                    "provider_id": self.provider_id,
                    "artifact_id": artifact,
                    "policy": policy.to_record(),
                },
                self.maximum_metadata_bytes,
            )
            self._atomic_replace_file(
                self.root / "retention" / f"{artifact}.json", payload
            )

    def tombstone(
        self,
        artifact_id: str,
        reason: str,
        /,
        *,
        created_at: int | None = None,
        eligible_at: int | None = None,
    ) -> TombstoneRecord:
        with self._exclusive_lock():
            artifact = _identifier(artifact_id, "artifact_id")
            self.get_manifest(artifact)
            created = _now() if created_at is None else int(created_at)
            record = TombstoneRecord(
                self.provider_id,
                artifact,
                reason,
                created,
                created if eligible_at is None else eligible_at,
            )
            self._atomic_create_immutable(
                self.root / "tombstones" / f"{artifact}.json",
                _json_bytes(record.to_record(), self.maximum_metadata_bytes),
            )
            _fsync_directory(self.root / "tombstones")
            return record

    def collect_garbage(
        self,
        /,
        *,
        now: int | None = None,
        default_policy: RetentionPolicy | None = None,
    ) -> GarbageCollectionReport:
        with self._exclusive_lock():
            return self._collect_garbage_locked(now=now, default_policy=default_policy)

    def _collect_garbage_locked(
        self,
        /,
        *,
        now: int | None,
        default_policy: RetentionPolicy | None,
    ) -> GarbageCollectionReport:
        collected = _now() if now is None else _nonnegative(now, "now")
        default = RetentionPolicy() if default_policy is None else default_policy
        if not isinstance(default, RetentionPolicy):
            raise TypeError("default_policy must be RetentionPolicy.")
        self._cleanup_trash()
        self._cleanup_staging(collected, default.abandoned_attempt_grace_seconds)
        removed_attempts: list[str] = []
        removed_artifacts: list[str] = []
        expired_leases = self._expire_and_collect_leases(collected)
        active_leases = self._active_lease_artifacts(collected)
        held = self._held_artifacts()
        pointers = self._all_pointers()
        roots = self._all_roots()
        published = _published_manifest_ids(roots, tuple(pointers.values()))
        committed_by_artifact: dict[
            str, list[tuple[RepositoryTransaction, ArtifactManifest]]
        ] = {}
        for transaction, manifest in roots:
            if manifest is not None and manifest.manifest_id in published:
                committed_by_artifact.setdefault(transaction.artifact_id, []).append(
                    (transaction, manifest)
                )
        for values in committed_by_artifact.values():
            values.sort(
                key=lambda item: (item[1].committed_at, item[0].attempt_id), reverse=True
            )

        for transaction, manifest in roots:
            artifact = transaction.artifact_id
            if artifact in active_leases or artifact in held:
                continue
            policy = self._retention_for(artifact, default)
            pointer = pointers.get(artifact)
            is_current = (
                pointer is not None and pointer["attempt_id"] == transaction.attempt_id
            )
            tombstone = self._tombstone_optional(artifact)
            tombstoned = tombstone is not None and tombstone.eligible_at <= collected
            if manifest is None or manifest.manifest_id not in published:
                if (
                    collected - transaction.started_at
                    < policy.abandoned_attempt_grace_seconds
                ):
                    continue
            else:
                if collected - manifest.committed_at < policy.minimum_age_seconds:
                    continue
                rank = next(
                    index
                    for index, item in enumerate(committed_by_artifact[artifact])
                    if item[0].attempt_id == transaction.attempt_id
                )
                if not tombstoned and rank < policy.keep_latest_commits:
                    continue
                if is_current and not tombstoned:
                    continue
            if is_current:
                current = self._read_pointer_optional(artifact)
                if current is None or current["attempt_id"] != transaction.attempt_id:
                    continue
                self._unlink_existing(self._pointer_path(artifact))
                removed_artifacts.append(artifact)
            self._remove_root(transaction.attempt_id)
            removed_attempts.append(transaction.attempt_id)
        return GarbageCollectionReport(
            self.provider_id,
            collected,
            removed_attempts,
            removed_artifacts,
            expired_leases,
        )

    def _initialize(self) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in (
            "roots",
            "artifacts",
            "leases",
            "holds",
            "retention",
            "tombstones",
            "trash",
            "staging",
            "locks",
        ):
            (self.root / name).mkdir(mode=0o700, exist_ok=True)
        lock = self.root / "locks" / "repository.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        for name in (
            "roots",
            "artifacts",
            "leases",
            "holds",
            "retention",
            "tombstones",
            "trash",
            "staging",
            "locks",
        ):
            _fsync_directory(self.root / name)
        _fsync_directory(self.root)

    def _remove_root(self, attempt_id: str, /) -> None:
        source = self._attempt_path(attempt_id)
        destination = self.root / "trash" / attempt_id
        os.replace(source, destination)
        _fsync_directory(self.root / "roots")
        _fsync_directory(self.root / "trash")
        shutil.rmtree(destination)
        _fsync_directory(self.root / "trash")

    def _cleanup_trash(self) -> None:
        trash = self.root / "trash"
        for path in sorted(trash.iterdir()):
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise RepositoryCorruptionError(
                    "Repository trash contains a non-directory."
                )
            shutil.rmtree(path)
        _fsync_directory(trash)

    def _cleanup_staging(self, now: int, grace_seconds: int, /) -> None:
        staging = self.root / "staging"
        for path in sorted(staging.iterdir()):
            information = path.lstat()
            if not stat.S_ISDIR(information.st_mode):
                raise RepositoryCorruptionError(
                    "Repository staging contains a non-directory."
                )
            if now - int(information.st_mtime) >= grace_seconds:
                shutil.rmtree(path)
        _fsync_directory(staging)

    def _validate_transaction(self, transaction: RepositoryTransaction, /) -> None:
        if not isinstance(transaction, RepositoryTransaction):
            raise TypeError("transaction must be RepositoryTransaction.")
        if transaction.provider_id != self.provider_id:
            raise RepositoryConflictError(
                "Transaction provider does not match repository."
            )
        path = self._attempt_path(transaction.attempt_id) / "transaction.json"
        record = RepositoryTransaction.from_record(
            _read_json_file(path, self.maximum_metadata_bytes)
        )
        if record.transaction_id != transaction.transaction_id:
            raise RepositoryConflictError(
                "Transaction record does not match its attempt."
            )

    def _validate_staged_manifest(
        self, transaction: RepositoryTransaction, manifest: ArtifactManifest, /
    ) -> None:
        expected_prefix = f"roots/{transaction.attempt_id}/chunks/"
        for chunk in manifest.chunks:
            expected_key = self._chunk_key(
                transaction.attempt_id, chunk.logical_name, chunk.index
            )
            if chunk.object_key != expected_key or not chunk.object_key.startswith(
                expected_prefix
            ):
                raise RepositoryConflictError(
                    "Manifest chunk does not belong to the attempt-private root."
                )
            encoded = _read_bounded_file(
                self.root / chunk.object_key,
                min(self.maximum_chunk_bytes, chunk.encoded_size),
            )
            _decode_and_validate_chunk(chunk, encoded, self.maximum_chunk_bytes)

    def _validate_manifest_argument(
        self, manifest: ArtifactManifest, chunk: ChunkRecord, /
    ) -> None:
        if not isinstance(manifest, ArtifactManifest) or not isinstance(
            chunk, ChunkRecord
        ):
            raise TypeError("manifest and chunk must be typed repository records.")
        if manifest.provider_id != self.provider_id:
            raise RepositoryConflictError("Manifest provider does not match repository.")
        if chunk.chunk_id not in {item.chunk_id for item in manifest.chunks}:
            raise RepositoryConflictError("Chunk is not present in the manifest.")

    def _manifest_from_pointer(
        self, pointer: Mapping[str, object], artifact: str, /
    ) -> ArtifactManifest:
        _validate_pointer(pointer, self.provider_id, artifact)
        attempt = _identifier(str(pointer["attempt_id"]), "attempt_id")
        manifest_payload = _read_bounded_file(
            self._attempt_path(attempt) / "manifest.json", self.maximum_metadata_bytes
        )
        if hashlib.sha256(manifest_payload).hexdigest() != pointer["manifest_sha256"]:
            raise RepositoryCorruptionError("Artifact manifest checksum failed.")
        manifest = ArtifactManifest.from_record(_json_record(manifest_payload))
        if (
            manifest.manifest_id != pointer["manifest_id"]
            or manifest.transaction_id != pointer["transaction_id"]
            or manifest.provider_id != self.provider_id
            or manifest.artifact_id != artifact
        ):
            raise RepositoryCorruptionError("Artifact pointer and manifest disagree.")
        marker = _read_json_file(
            self._attempt_path(attempt) / "COMMIT", self.maximum_metadata_bytes
        )
        _validate_commit_marker(marker, pointer)
        expected_prefix = f"roots/{attempt}/chunks/"
        if any(
            not item.object_key.startswith(expected_prefix) for item in manifest.chunks
        ):
            raise RepositoryCorruptionError("Manifest chunk escapes its immutable root.")
        return manifest

    def _read_pointer_optional(self, artifact: str, /) -> dict[str, object] | None:
        path = self._pointer_path(artifact)
        if not path.exists():
            return None
        record = _read_json_file(path, _POINTER_LIMIT)
        _validate_pointer(record, self.provider_id, artifact)
        return record

    def _all_pointers(self) -> dict[str, dict[str, object]]:
        pointers: dict[str, dict[str, object]] = {}
        for path in sorted((self.root / "artifacts").glob("*.pointer")):
            artifact = _identifier(path.stem, "artifact_id")
            pointer = _read_json_file(path, _POINTER_LIMIT)
            _validate_pointer(pointer, self.provider_id, artifact)
            pointers[artifact] = pointer
        return pointers

    def _all_roots(
        self,
    ) -> tuple[tuple[RepositoryTransaction, ArtifactManifest | None], ...]:
        roots: list[tuple[RepositoryTransaction, ArtifactManifest | None]] = []
        for path in sorted((self.root / "roots").iterdir()):
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise RepositoryCorruptionError(
                    "Repository roots contain a non-directory."
                )
            attempt = _identifier(path.name, "attempt_id")
            transaction = RepositoryTransaction.from_record(
                _read_json_file(path / "transaction.json", self.maximum_metadata_bytes)
            )
            if (
                transaction.attempt_id != attempt
                or transaction.provider_id != self.provider_id
            ):
                raise RepositoryCorruptionError("Attempt transaction identity mismatch.")
            manifest_path = path / "manifest.json"
            marker_path = path / "COMMIT"
            if marker_path.exists() and not manifest_path.exists():
                raise RepositoryCorruptionError("Commit marker has no manifest.")
            manifest = (
                None
                if not marker_path.exists()
                else ArtifactManifest.from_record(
                    _read_json_file(manifest_path, self.maximum_metadata_bytes)
                )
            )
            if manifest is not None:
                if (
                    manifest.provider_id != transaction.provider_id
                    or manifest.artifact_id != transaction.artifact_id
                    or manifest.transaction_id != transaction.transaction_id
                    or manifest.base_manifest_id != transaction.base_manifest_id
                    or any(
                        not chunk.object_key.startswith(
                            f"roots/{transaction.attempt_id}/chunks/"
                        )
                        for chunk in manifest.chunks
                    )
                ):
                    raise RepositoryCorruptionError(
                        "Committed root manifest does not match its transaction."
                    )
                marker = _read_json_file(marker_path, self.maximum_metadata_bytes)
                expected = _pointer_record(
                    transaction,
                    manifest,
                    hashlib.sha256(
                        _read_bounded_file(manifest_path, self.maximum_metadata_bytes)
                    ).hexdigest(),
                )
                _validate_commit_marker(marker, expected)
            roots.append((transaction, manifest))
        return tuple(roots)

    def _expire_and_collect_leases(self, now: int, /) -> tuple[str, ...]:
        expired: list[str] = []
        for path in sorted((self.root / "leases").glob("*/*.json")):
            lease = LeaseRecord.from_record(
                _read_json_file(path, self.maximum_metadata_bytes)
            )
            if lease.provider_id != self.provider_id:
                raise RepositoryCorruptionError("Lease provider mismatch.")
            if lease.expires_at <= now:
                self._unlink_existing(path)
                expired.append(lease.lease_id)
        return tuple(expired)

    def _active_lease_artifacts(self, now: int, /) -> set[str]:
        active: set[str] = set()
        for path in sorted((self.root / "leases").glob("*/*.json")):
            lease = LeaseRecord.from_record(
                _read_json_file(path, self.maximum_metadata_bytes)
            )
            if lease.provider_id != self.provider_id:
                raise RepositoryCorruptionError("Lease provider mismatch.")
            if lease.expires_at > now:
                active.add(lease.artifact_id)
        return active

    def _held_artifacts(self) -> set[str]:
        held: set[str] = set()
        for path in sorted((self.root / "holds").glob("*/*.json")):
            hold = LegalHoldRecord.from_record(
                _read_json_file(path, self.maximum_metadata_bytes)
            )
            if hold.provider_id != self.provider_id:
                raise RepositoryCorruptionError("Legal-hold provider mismatch.")
            held.add(hold.artifact_id)
        return held

    def _retention_for(
        self, artifact: str, default: RetentionPolicy, /
    ) -> RetentionPolicy:
        path = self.root / "retention" / f"{artifact}.json"
        if not path.exists():
            return default
        record = _read_json_file(path, self.maximum_metadata_bytes)
        if (
            record.get("kind") != "artifact-retention"
            or record.get("provider_id") != self.provider_id
            or record.get("artifact_id") != artifact
            or not isinstance(record.get("policy"), Mapping)
        ):
            raise RepositoryCorruptionError("Retention record is invalid.")
        return RetentionPolicy.from_record(record["policy"])

    def _tombstone_optional(self, artifact: str, /) -> TombstoneRecord | None:
        path = self.root / "tombstones" / f"{artifact}.json"
        if not path.exists():
            return None
        value = TombstoneRecord.from_record(
            _read_json_file(path, self.maximum_metadata_bytes)
        )
        if value.provider_id != self.provider_id or value.artifact_id != artifact:
            raise RepositoryCorruptionError("Tombstone provider or artifact mismatch.")
        return value

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        descriptor = os.open(self.root / "locks" / "repository.lock", os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _replace_pointer(self, artifact: str, payload: bytes, /) -> None:
        self._atomic_replace_file(self._pointer_path(artifact), payload)

    def _atomic_replace_file(self, destination: Path, payload: bytes, /) -> None:
        temporary = (
            destination.parent / f".{destination.name}.{secrets.token_hex(16)}.tmp"
        )
        self._create_immutable_file(temporary, payload)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)

    def _create_immutable_file(self, path: Path, payload: bytes, /) -> None:
        try:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
        except FileExistsError as error:
            raise RepositoryConflictError(
                f"Immutable repository object {path.name!r} already exists."
            ) from error
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RepositoryError(f"Cannot write repository file {path}.")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_create_immutable(self, path: Path, payload: bytes, /) -> None:
        temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.immutable.tmp"
        self._create_immutable_file(temporary, payload)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise RepositoryConflictError(
                    f"Immutable repository object {path.name!r} already exists."
                ) from error
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _create_or_verify_immutable(self, path: Path, payload: bytes, /) -> None:
        temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.immutable.tmp"
        self._create_immutable_file(temporary, payload)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = _read_bounded_file(path, len(payload))
                if existing != payload:
                    raise RepositoryConflictError(
                        f"Immutable repository object {path.name!r} conflicts."
                    )
            else:
                _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _unlink_existing(self, path: Path, /) -> None:
        try:
            path.unlink()
        except FileNotFoundError as error:
            raise RepositoryConflictError(
                f"Repository record {path.name!r} no longer exists."
            ) from error
        _fsync_directory(path.parent)

    def _attempt_path(self, attempt: str, /) -> Path:
        return self.root / "roots" / _identifier(attempt, "attempt_id")

    def _pointer_path(self, artifact: str, /) -> Path:
        return self.root / "artifacts" / f"{_identifier(artifact, 'artifact_id')}.pointer"

    @staticmethod
    def _chunk_key(attempt: str, logical: str, index: int, /) -> str:
        return f"roots/{attempt}/chunks/{logical}.{index:08d}.bin"

    def _fail(self, point: str, /) -> None:
        if self.failure_injector is not None:
            self.failure_injector(point)


class S3ArtifactRepository:
    """Transactional S3-compatible repository over conditional whole objects."""

    def __init__(
        self,
        client: ConditionalObjectClient,
        profile: ObjectStoreProfile,
        namespace: str,
        /,
        *,
        maximum_chunk_bytes: int = _DEFAULT_CHUNK_LIMIT,
        failure_injector: FailureInjector | None = None,
    ):
        if not isinstance(profile, ObjectStoreProfile):
            raise TypeError("profile must be ObjectStoreProfile.")
        profile.require_transactional_support()
        maximum = _positive(maximum_chunk_bytes, "maximum_chunk_bytes")
        if maximum > profile.maximum_object_bytes:
            raise UnsupportedRepositoryProfileError(
                "maximum_chunk_bytes exceeds the object-store profile bound."
            )
        self.client = client
        self.profile = profile
        self.namespace = _namespace(namespace)
        self.provider_id = profile.provider_id
        self.maximum_chunk_bytes = maximum
        self.maximum_metadata_bytes = min(_METADATA_LIMIT, profile.maximum_object_bytes)
        self.failure_injector = failure_injector
        self.support_tuple = SupportTuple(
            "artifact.repository",
            {
                "provider": self.provider_id,
                "transport": "s3-compatible",
                "profile_id": profile.profile_id,
                "maximum_chunk_bytes": maximum,
            },
        )

    def begin(
        self,
        artifact_id: str,
        writer_id: str,
        /,
        *,
        attempt_id: str | None = None,
        started_at: int | None = None,
    ) -> RepositoryTransaction:
        artifact = _identifier(artifact_id, "artifact_id")
        attempt = secrets.token_hex(16) if attempt_id is None else attempt_id
        pointer_value = self._read_pointer_optional(artifact)
        pointer = None if pointer_value is None else _json_record(pointer_value.data)
        transaction = RepositoryTransaction(
            self.provider_id,
            artifact,
            writer_id,
            attempt,
            base_manifest_id=None if pointer is None else str(pointer["manifest_id"]),
            base_pointer_token=(
                None if pointer_value is None else pointer_value.metadata.etag
            ),
            started_at=_now() if started_at is None else started_at,
        )
        self.client.create_object(
            self._transaction_key(transaction.attempt_id),
            _json_bytes(transaction.to_record(), self.maximum_metadata_bytes),
        )
        return transaction

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
    ) -> ChunkRecord:
        self._validate_transaction(transaction)
        logical = _identifier(logical_name, "logical_name")
        index_ = _nonnegative(index, "index")
        offset_ = _nonnegative(offset, "offset")
        plaintext = _bounded_bytes(payload, self.maximum_chunk_bytes, "chunk plaintext")
        encoded = _encode_chunk(plaintext, encoding)
        if len(encoded) > self.maximum_chunk_bytes:
            raise ValueError("Encoded chunk exceeds maximum_chunk_bytes.")
        key = self._chunk_key(transaction.attempt_id, logical, index_)
        record = ChunkRecord(
            transaction.transaction_id,
            logical,
            index_,
            offset_,
            len(plaintext),
            len(encoded),
            hashlib.sha256(plaintext).hexdigest(),
            hashlib.sha256(encoded).hexdigest(),
            encoding,
            key,
        )
        self.client.create_object(key, encoded)
        return record

    def commit(
        self,
        transaction: RepositoryTransaction,
        chunks: Sequence[ChunkRecord],
        /,
        *,
        metadata: Mapping[str, str] | Sequence[tuple[str, str]] = (),
        committed_at: int | None = None,
    ) -> ArtifactManifest:
        self._validate_transaction(transaction)
        manifest = ArtifactManifest(
            self.provider_id,
            transaction.artifact_id,
            transaction.transaction_id,
            transaction.base_manifest_id,
            chunks,
            metadata=metadata,
            committed_at=_now() if committed_at is None else committed_at,
        )
        self._validate_staged_manifest(transaction, manifest)
        manifest_payload = _json_bytes(manifest.to_record(), self.maximum_metadata_bytes)
        manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
        marker_payload = _json_bytes(
            _commit_marker(transaction, manifest, manifest_sha),
            self.maximum_metadata_bytes,
        )
        pointer_payload = _json_bytes(
            _pointer_record(transaction, manifest, manifest_sha), _POINTER_LIMIT
        )
        self._fail("before_manifest")
        self._create_or_verify_object(
            self._manifest_key(transaction.attempt_id), manifest_payload
        )
        self._fail("after_manifest")
        self._fail("before_commit_marker")
        self._create_or_verify_object(
            self._commit_key(transaction.attempt_id), marker_payload
        )
        self._fail("after_commit_marker")
        guard = self._acquire_artifact_guard(transaction.artifact_id)
        try:
            self._fail("before_pointer")
            pointer_key = self._pointer_key(transaction.artifact_id)
            if transaction.base_pointer_token is None:
                self.client.create_object(pointer_key, pointer_payload)
            else:
                self.client.replace_object(
                    pointer_key, pointer_payload, transaction.base_pointer_token
                )
            self._fail("after_pointer")
        finally:
            self._release_artifact_guard(guard)
        return manifest

    def get_manifest(self, artifact_id: str, /) -> ArtifactManifest:
        artifact = _identifier(artifact_id, "artifact_id")
        value = self._read_pointer_optional(artifact)
        if value is None:
            raise ObjectNotFoundError(f"Artifact {artifact!r} does not exist.")
        pointer = _json_record(value.data)
        _validate_pointer(pointer, self.provider_id, artifact)
        manifest_payload = self.client.read_object(
            self._manifest_key(str(pointer["attempt_id"])),
            maximum_bytes=self.maximum_metadata_bytes,
        ).data
        if hashlib.sha256(manifest_payload).hexdigest() != pointer["manifest_sha256"]:
            raise RepositoryCorruptionError("Artifact manifest checksum failed.")
        manifest = ArtifactManifest.from_record(_json_record(manifest_payload))
        if (
            manifest.manifest_id != pointer["manifest_id"]
            or manifest.transaction_id != pointer["transaction_id"]
            or manifest.provider_id != self.provider_id
            or manifest.artifact_id != artifact
        ):
            raise RepositoryCorruptionError("Artifact pointer and manifest disagree.")
        marker = _json_record(
            self.client.read_object(
                self._commit_key(str(pointer["attempt_id"])),
                maximum_bytes=self.maximum_metadata_bytes,
            ).data
        )
        _validate_commit_marker(marker, pointer)
        expected_prefix = self._root_prefix(str(pointer["attempt_id"])) + "chunks/"
        if any(
            not item.object_key.startswith(expected_prefix) for item in manifest.chunks
        ):
            raise RepositoryCorruptionError("Manifest chunk escapes its immutable root.")
        return manifest

    def read_chunk(
        self,
        manifest: ArtifactManifest,
        chunk: ChunkRecord,
        /,
        *,
        maximum_plaintext_bytes: int | None = None,
    ) -> bytes:
        self._validate_manifest_argument(manifest, chunk)
        maximum = (
            self.maximum_chunk_bytes
            if maximum_plaintext_bytes is None
            else _positive(maximum_plaintext_bytes, "maximum_plaintext_bytes")
        )
        if chunk.plaintext_size > maximum:
            raise RepositoryCorruptionError("Chunk exceeds requested plaintext bound.")
        encoded = self.client.read_object(
            chunk.object_key,
            maximum_bytes=min(self.maximum_chunk_bytes, chunk.encoded_size),
        ).data
        return _decode_and_validate_chunk(chunk, encoded, maximum)

    def read_bytes(
        self,
        artifact_id: str,
        logical_name: str,
        /,
        *,
        maximum_bytes: int,
    ) -> bytes:
        manifest = self.get_manifest(artifact_id)
        logical = _identifier(logical_name, "logical_name")
        chunks = tuple(item for item in manifest.chunks if item.logical_name == logical)
        total = sum(item.plaintext_size for item in chunks)
        maximum = _positive(maximum_bytes, "maximum_bytes")
        if not chunks:
            raise ObjectNotFoundError(f"Artifact payload {logical!r} does not exist.")
        if total > maximum:
            raise RepositoryCorruptionError(
                "Artifact payload exceeds requested byte bound."
            )
        return b"".join(
            self.read_chunk(manifest, item, maximum_plaintext_bytes=maximum)
            for item in chunks
        )

    def acquire_lease(
        self,
        artifact_id: str,
        holder_id: str,
        /,
        *,
        expires_at: int,
        lease_id: str | None = None,
        issued_at: int | None = None,
    ) -> LeaseRecord:
        artifact = _identifier(artifact_id, "artifact_id")
        guard = self._acquire_artifact_guard(artifact)
        try:
            pointer = self._read_pointer_optional(artifact)
            if pointer is None:
                raise ObjectNotFoundError(f"Artifact {artifact!r} does not exist.")
            self.get_manifest(artifact)
            lease = LeaseRecord(
                self.provider_id,
                artifact,
                holder_id,
                secrets.token_hex(16) if lease_id is None else lease_id,
                _now() if issued_at is None else issued_at,
                expires_at,
            )
            self.client.create_object(
                self._lease_key(artifact, lease.lease_id),
                _json_bytes(lease.to_record(), self.maximum_metadata_bytes),
            )
            return lease
        finally:
            self._release_artifact_guard(guard)

    def release_lease(self, lease: LeaseRecord, /) -> None:
        if not isinstance(lease, LeaseRecord) or lease.provider_id != self.provider_id:
            raise TypeError("lease must belong to this repository provider.")
        key = self._lease_key(lease.artifact_id, lease.lease_id)
        value = self.client.read_object(key, maximum_bytes=self.maximum_metadata_bytes)
        persisted = LeaseRecord.from_record(_json_record(value.data))
        if persisted.record_id != lease.record_id:
            raise RepositoryConflictError("Lease release record does not match.")
        self.client.delete_object(key, expected_etag=value.metadata.etag)

    def place_legal_hold(
        self,
        artifact_id: str,
        authority: str,
        /,
        *,
        hold_id: str | None = None,
        placed_at: int | None = None,
    ) -> LegalHoldRecord:
        artifact = _identifier(artifact_id, "artifact_id")
        guard = self._acquire_artifact_guard(artifact)
        try:
            if self._read_pointer_optional(artifact) is None:
                raise ObjectNotFoundError(f"Artifact {artifact!r} does not exist.")
            self.get_manifest(artifact)
            hold = LegalHoldRecord(
                self.provider_id,
                artifact,
                secrets.token_hex(16) if hold_id is None else hold_id,
                authority,
                _now() if placed_at is None else placed_at,
            )
            self.client.create_object(
                self._hold_key(artifact, hold.hold_id),
                _json_bytes(hold.to_record(), self.maximum_metadata_bytes),
            )
            return hold
        finally:
            self._release_artifact_guard(guard)

    def release_legal_hold(self, hold: LegalHoldRecord, /) -> None:
        if not isinstance(hold, LegalHoldRecord) or hold.provider_id != self.provider_id:
            raise TypeError("hold must belong to this repository provider.")
        key = self._hold_key(hold.artifact_id, hold.hold_id)
        value = self.client.read_object(key, maximum_bytes=self.maximum_metadata_bytes)
        persisted = LegalHoldRecord.from_record(_json_record(value.data))
        if persisted.record_id != hold.record_id:
            raise RepositoryConflictError("Legal-hold release record does not match.")
        self.client.delete_object(key, expected_etag=value.metadata.etag)

    def set_retention(self, artifact_id: str, policy: RetentionPolicy, /) -> None:
        artifact = _identifier(artifact_id, "artifact_id")
        if not isinstance(policy, RetentionPolicy):
            raise TypeError("policy must be RetentionPolicy.")
        guard = self._acquire_artifact_guard(artifact)
        try:
            self.get_manifest(artifact)
            key = self._retention_key(artifact)
            payload = _json_bytes(
                {
                    "kind": "artifact-retention",
                    "provider_id": self.provider_id,
                    "artifact_id": artifact,
                    "policy": policy.to_record(),
                },
                self.maximum_metadata_bytes,
            )
            current = self._read_object_optional(key, self.maximum_metadata_bytes)
            if current is None:
                self.client.create_object(key, payload)
            else:
                self.client.replace_object(key, payload, current.metadata.etag)
        finally:
            self._release_artifact_guard(guard)

    def tombstone(
        self,
        artifact_id: str,
        reason: str,
        /,
        *,
        created_at: int | None = None,
        eligible_at: int | None = None,
    ) -> TombstoneRecord:
        artifact = _identifier(artifact_id, "artifact_id")
        guard = self._acquire_artifact_guard(artifact)
        try:
            self.get_manifest(artifact)
            created = _now() if created_at is None else int(created_at)
            record = TombstoneRecord(
                self.provider_id,
                artifact,
                reason,
                created,
                created if eligible_at is None else eligible_at,
            )
            self.client.create_object(
                self._tombstone_key(artifact),
                _json_bytes(record.to_record(), self.maximum_metadata_bytes),
            )
            return record
        finally:
            self._release_artifact_guard(guard)

    def collect_garbage(
        self,
        /,
        *,
        now: int | None = None,
        default_policy: RetentionPolicy | None = None,
    ) -> GarbageCollectionReport:
        collected = _now() if now is None else _nonnegative(now, "now")
        default = RetentionPolicy() if default_policy is None else default_policy
        if not isinstance(default, RetentionPolicy):
            raise TypeError("default_policy must be RetentionPolicy.")
        expired_leases = self._expire_leases(collected)
        pointers = self._all_pointers()
        roots = self._all_roots()
        pointer_records = {
            artifact: _json_record(value.data) for artifact, value in pointers.items()
        }
        published = _published_manifest_ids(roots, tuple(pointer_records.values()))
        committed_by_artifact: dict[
            str, list[tuple[RepositoryTransaction, ArtifactManifest]]
        ] = {}
        for transaction, manifest in roots:
            if manifest is not None and manifest.manifest_id in published:
                committed_by_artifact.setdefault(transaction.artifact_id, []).append(
                    (transaction, manifest)
                )
        for values in committed_by_artifact.values():
            values.sort(
                key=lambda item: (item[1].committed_at, item[0].attempt_id), reverse=True
            )
        removed_attempts: list[str] = []
        removed_artifacts: list[str] = []
        for transaction, manifest in roots:
            artifact = transaction.artifact_id
            guard = self._acquire_artifact_guard(artifact)
            try:
                if (
                    artifact in self._active_lease_artifacts(collected)
                    or artifact in self._held_artifacts()
                ):
                    continue
                policy = self._retention_for(artifact, default)
                pointer_value = self._read_pointer_optional(artifact)
                pointer = (
                    None if pointer_value is None else _json_record(pointer_value.data)
                )
                is_current = (
                    pointer is not None
                    and pointer["attempt_id"] == transaction.attempt_id
                )
                tombstone = self._tombstone_optional(artifact)
                tombstoned = tombstone is not None and tombstone.eligible_at <= collected
                if manifest is None or manifest.manifest_id not in published:
                    if (
                        collected - transaction.started_at
                        < policy.abandoned_attempt_grace_seconds
                    ):
                        continue
                else:
                    if collected - manifest.committed_at < policy.minimum_age_seconds:
                        continue
                    rank = next(
                        index
                        for index, item in enumerate(committed_by_artifact[artifact])
                        if item[0].attempt_id == transaction.attempt_id
                    )
                    if not tombstoned and rank < policy.keep_latest_commits:
                        continue
                    if is_current and not tombstoned:
                        continue
                if is_current:
                    if pointer_value is None:
                        raise RepositoryCorruptionError(
                            "Current pointer disappeared during GC."
                        )
                    self.client.delete_object(
                        self._pointer_key(artifact),
                        expected_etag=pointer_value.metadata.etag,
                    )
                    removed_artifacts.append(artifact)
                root_objects = tuple(
                    sorted(
                        self.client.list_objects(
                            self._root_prefix(transaction.attempt_id)
                        ),
                        key=lambda item: (
                            item.key.endswith("/transaction.json"),
                            item.key,
                        ),
                    )
                )
                for value in root_objects:
                    self.client.delete_object(value.key, expected_etag=value.etag)
                removed_attempts.append(transaction.attempt_id)
            finally:
                self._release_artifact_guard(guard)
        return GarbageCollectionReport(
            self.provider_id,
            collected,
            removed_attempts,
            removed_artifacts,
            expired_leases,
        )

    def _validate_transaction(self, transaction: RepositoryTransaction, /) -> None:
        if not isinstance(transaction, RepositoryTransaction):
            raise TypeError("transaction must be RepositoryTransaction.")
        if transaction.provider_id != self.provider_id:
            raise RepositoryConflictError(
                "Transaction provider does not match repository."
            )
        persisted = RepositoryTransaction.from_record(
            _json_record(
                self.client.read_object(
                    self._transaction_key(transaction.attempt_id),
                    maximum_bytes=self.maximum_metadata_bytes,
                ).data
            )
        )
        if persisted.transaction_id != transaction.transaction_id:
            raise RepositoryConflictError(
                "Transaction record does not match its attempt."
            )

    def _validate_staged_manifest(
        self, transaction: RepositoryTransaction, manifest: ArtifactManifest, /
    ) -> None:
        prefix = self._root_prefix(transaction.attempt_id) + "chunks/"
        for chunk in manifest.chunks:
            expected = self._chunk_key(
                transaction.attempt_id, chunk.logical_name, chunk.index
            )
            if chunk.object_key != expected or not expected.startswith(prefix):
                raise RepositoryConflictError(
                    "Manifest chunk does not belong to the attempt-private root."
                )
            encoded = self.client.read_object(
                chunk.object_key,
                maximum_bytes=min(self.maximum_chunk_bytes, chunk.encoded_size),
            ).data
            _decode_and_validate_chunk(chunk, encoded, self.maximum_chunk_bytes)

    def _validate_manifest_argument(
        self, manifest: ArtifactManifest, chunk: ChunkRecord, /
    ) -> None:
        if not isinstance(manifest, ArtifactManifest) or not isinstance(
            chunk, ChunkRecord
        ):
            raise TypeError("manifest and chunk must be typed repository records.")
        if manifest.provider_id != self.provider_id:
            raise RepositoryConflictError("Manifest provider does not match repository.")
        if chunk.chunk_id not in {item.chunk_id for item in manifest.chunks}:
            raise RepositoryConflictError("Chunk is not present in the manifest.")

    def _create_or_verify_object(self, key: str, payload: bytes, /) -> None:
        existing = self._read_object_optional(key, len(payload))
        if existing is None:
            self.client.create_object(key, payload)
        elif existing.data != payload:
            raise RepositoryConflictError(
                f"Immutable repository object {key!r} conflicts."
            )

    def _read_pointer_optional(self, artifact: str, /) -> ObjectValue | None:
        value = self._read_object_optional(self._pointer_key(artifact), _POINTER_LIMIT)
        if value is not None:
            _validate_pointer(_json_record(value.data), self.provider_id, artifact)
        return value

    def _read_object_optional(self, key: str, maximum: int, /) -> ObjectValue | None:
        try:
            return self.client.read_object(key, maximum_bytes=maximum)
        except ObjectNotFoundError:
            return None

    def _acquire_artifact_guard(self, artifact_id: str, /) -> ObjectMetadata:
        artifact = _identifier(artifact_id, "artifact_id")
        key = self._guard_key(artifact)
        token = secrets.token_hex(16)
        payload = _json_bytes(
            {
                "kind": "artifact-metadata-guard",
                "provider_id": self.provider_id,
                "artifact_id": artifact,
                "token": token,
                "acquired_at": _now(),
            },
            self.maximum_metadata_bytes,
        )
        current = self._read_object_optional(key, self.maximum_metadata_bytes)
        if current is not None:
            record = _json_record(current.data)
            if (
                record.get("kind") != "artifact-metadata-guard"
                or record.get("provider_id") != self.provider_id
                or record.get("artifact_id") != artifact
                or not isinstance(record.get("token"), str)
                or type(record.get("acquired_at")) is not int
            ):
                raise RepositoryCorruptionError("Artifact metadata guard is invalid.")
            raise RepositoryConflictError(
                f"Artifact {artifact!r} metadata is being modified."
            )
        return self.client.create_object(key, payload)

    def _release_artifact_guard(self, metadata: ObjectMetadata, /) -> None:
        self.client.delete_object(metadata.key, expected_etag=metadata.etag)

    def _all_pointers(self) -> dict[str, ObjectValue]:
        prefix = self._key("artifacts/")
        values: dict[str, ObjectValue] = {}
        for metadata in self.client.list_objects(prefix):
            if not metadata.key.endswith(".pointer"):
                raise RepositoryCorruptionError(
                    "Artifact namespace has an invalid object."
                )
            artifact = _identifier(
                metadata.key[len(prefix) : -len(".pointer")], "artifact_id"
            )
            value = self.client.read_object(metadata.key, maximum_bytes=_POINTER_LIMIT)
            _validate_pointer(_json_record(value.data), self.provider_id, artifact)
            values[artifact] = value
        return values

    def _all_roots(
        self,
    ) -> tuple[tuple[RepositoryTransaction, ArtifactManifest | None], ...]:
        prefix = self._key("roots/")
        attempts = sorted(
            {
                metadata.key[len(prefix) :].split("/", 1)[0]
                for metadata in self.client.list_objects(prefix)
            }
        )
        roots: list[tuple[RepositoryTransaction, ArtifactManifest | None]] = []
        for attempt in attempts:
            attempt_ = _identifier(attempt, "attempt_id")
            transaction = RepositoryTransaction.from_record(
                _json_record(
                    self.client.read_object(
                        self._transaction_key(attempt_),
                        maximum_bytes=self.maximum_metadata_bytes,
                    ).data
                )
            )
            if transaction.provider_id != self.provider_id:
                raise RepositoryCorruptionError("Attempt provider mismatch.")
            marker = self._read_object_optional(
                self._commit_key(attempt_), self.maximum_metadata_bytes
            )
            manifest_value = self._read_object_optional(
                self._manifest_key(attempt_), self.maximum_metadata_bytes
            )
            if marker is not None and manifest_value is None:
                raise RepositoryCorruptionError("Commit marker has no manifest.")
            manifest = (
                None
                if marker is None
                else ArtifactManifest.from_record(_json_record(manifest_value.data))
            )
            if manifest is not None:
                if (
                    manifest.provider_id != transaction.provider_id
                    or manifest.artifact_id != transaction.artifact_id
                    or manifest.transaction_id != transaction.transaction_id
                    or manifest.base_manifest_id != transaction.base_manifest_id
                    or any(
                        not chunk.object_key.startswith(
                            self._root_prefix(transaction.attempt_id) + "chunks/"
                        )
                        for chunk in manifest.chunks
                    )
                ):
                    raise RepositoryCorruptionError(
                        "Committed root manifest does not match its transaction."
                    )
                expected = _pointer_record(
                    transaction,
                    manifest,
                    hashlib.sha256(manifest_value.data).hexdigest(),
                )
                _validate_commit_marker(_json_record(marker.data), expected)
            roots.append((transaction, manifest))
        return tuple(roots)

    def _expire_leases(self, now: int, /) -> tuple[str, ...]:
        expired: list[str] = []
        for metadata in self.client.list_objects(self._key("leases/")):
            value = self.client.read_object(
                metadata.key, maximum_bytes=self.maximum_metadata_bytes
            )
            lease = LeaseRecord.from_record(_json_record(value.data))
            if lease.provider_id != self.provider_id:
                raise RepositoryCorruptionError("Lease provider mismatch.")
            if lease.expires_at <= now:
                self.client.delete_object(metadata.key, expected_etag=value.metadata.etag)
                expired.append(lease.lease_id)
        return tuple(expired)

    def _active_lease_artifacts(self, now: int, /) -> set[str]:
        active: set[str] = set()
        for metadata in self.client.list_objects(self._key("leases/")):
            lease = LeaseRecord.from_record(
                _json_record(
                    self.client.read_object(
                        metadata.key, maximum_bytes=self.maximum_metadata_bytes
                    ).data
                )
            )
            if lease.expires_at > now:
                active.add(lease.artifact_id)
        return active

    def _held_artifacts(self) -> set[str]:
        held: set[str] = set()
        for metadata in self.client.list_objects(self._key("holds/")):
            hold = LegalHoldRecord.from_record(
                _json_record(
                    self.client.read_object(
                        metadata.key, maximum_bytes=self.maximum_metadata_bytes
                    ).data
                )
            )
            if hold.provider_id != self.provider_id:
                raise RepositoryCorruptionError("Legal-hold provider mismatch.")
            held.add(hold.artifact_id)
        return held

    def _retention_for(
        self, artifact: str, default: RetentionPolicy, /
    ) -> RetentionPolicy:
        value = self._read_object_optional(
            self._retention_key(artifact), self.maximum_metadata_bytes
        )
        if value is None:
            return default
        record = _json_record(value.data)
        if (
            record.get("kind") != "artifact-retention"
            or record.get("provider_id") != self.provider_id
            or record.get("artifact_id") != artifact
            or not isinstance(record.get("policy"), Mapping)
        ):
            raise RepositoryCorruptionError("Retention record is invalid.")
        return RetentionPolicy.from_record(record["policy"])

    def _tombstone_optional(self, artifact: str, /) -> TombstoneRecord | None:
        value = self._read_object_optional(
            self._tombstone_key(artifact), self.maximum_metadata_bytes
        )
        if value is None:
            return None
        record = TombstoneRecord.from_record(_json_record(value.data))
        if record.provider_id != self.provider_id or record.artifact_id != artifact:
            raise RepositoryCorruptionError("Tombstone provider or artifact mismatch.")
        return record

    def _key(self, suffix: str, /) -> str:
        return f"{self.namespace}/{suffix}"

    def _root_prefix(self, attempt: str, /) -> str:
        return self._key(f"roots/{_identifier(attempt, 'attempt_id')}/")

    def _transaction_key(self, attempt: str, /) -> str:
        return self._root_prefix(attempt) + "transaction.json"

    def _manifest_key(self, attempt: str, /) -> str:
        return self._root_prefix(attempt) + "manifest.json"

    def _commit_key(self, attempt: str, /) -> str:
        return self._root_prefix(attempt) + "COMMIT"

    def _chunk_key(self, attempt: str, logical: str, index: int, /) -> str:
        return self._root_prefix(attempt) + f"chunks/{logical}.{index:08d}.bin"

    def _pointer_key(self, artifact: str, /) -> str:
        return self._key(f"artifacts/{_identifier(artifact, 'artifact_id')}.pointer")

    def _lease_key(self, artifact: str, lease: str, /) -> str:
        return self._key(
            f"leases/{_identifier(artifact, 'artifact_id')}/{_identifier(lease, 'lease_id')}.json"
        )

    def _hold_key(self, artifact: str, hold: str, /) -> str:
        return self._key(
            f"holds/{_identifier(artifact, 'artifact_id')}/{_identifier(hold, 'hold_id')}.json"
        )

    def _retention_key(self, artifact: str, /) -> str:
        return self._key(f"retention/{_identifier(artifact, 'artifact_id')}.json")

    def _tombstone_key(self, artifact: str, /) -> str:
        return self._key(f"tombstones/{_identifier(artifact, 'artifact_id')}.json")

    def _guard_key(self, artifact: str, /) -> str:
        return self._key(f"guards/{_identifier(artifact, 'artifact_id')}.json")

    def _fail(self, point: str, /) -> None:
        if self.failure_injector is not None:
            self.failure_injector(point)


def _published_manifest_ids(
    roots: Sequence[tuple[RepositoryTransaction, ArtifactManifest | None]],
    pointers: Sequence[Mapping[str, object]],
    /,
) -> frozenset[str]:
    manifests = {
        manifest.manifest_id: (transaction, manifest)
        for transaction, manifest in roots
        if manifest is not None
    }
    frontier: list[str] = []
    for pointer in pointers:
        manifest_id = str(pointer["manifest_id"])
        root = manifests.get(manifest_id)
        if root is None or root[0].attempt_id != pointer["attempt_id"]:
            raise RepositoryCorruptionError(
                "Artifact pointer references a missing immutable root."
            )
        frontier.append(manifest_id)
    published: set[str] = set()
    while frontier:
        manifest_id = frontier.pop()
        if manifest_id in published:
            continue
        published.add(manifest_id)
        parent = manifests[manifest_id][1].base_manifest_id
        if parent is not None and parent in manifests:
            frontier.append(parent)
    return frozenset(published)


def _commit_marker(
    transaction: RepositoryTransaction,
    manifest: ArtifactManifest,
    manifest_sha256: str,
    /,
) -> dict[str, object]:
    return {
        "kind": "artifact-commit",
        "provider_id": transaction.provider_id,
        "artifact_id": transaction.artifact_id,
        "attempt_id": transaction.attempt_id,
        "transaction_id": transaction.transaction_id,
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest_sha256,
    }


def _pointer_record(
    transaction: RepositoryTransaction,
    manifest: ArtifactManifest,
    manifest_sha256: str,
    /,
) -> dict[str, object]:
    return {
        "kind": "artifact-pointer",
        "provider_id": transaction.provider_id,
        "artifact_id": transaction.artifact_id,
        "attempt_id": transaction.attempt_id,
        "transaction_id": transaction.transaction_id,
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest_sha256,
    }


def _validate_pointer(
    record: Mapping[str, object], provider: str, artifact: str, /
) -> None:
    if (
        record.get("kind") != "artifact-pointer"
        or record.get("provider_id") != provider
        or record.get("artifact_id") != artifact
    ):
        raise RepositoryCorruptionError("Artifact pointer identity is invalid.")
    _identifier(str(record.get("attempt_id")), "attempt_id")
    for name in ("transaction_id", "manifest_id", "manifest_sha256"):
        value = record.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RepositoryCorruptionError(f"Artifact pointer {name} is invalid.")


def _validate_commit_marker(
    marker: Mapping[str, object], pointer: Mapping[str, object], /
) -> None:
    expected = dict(pointer)
    expected["kind"] = "artifact-commit"
    if dict(marker) != expected:
        raise RepositoryCorruptionError("Commit marker and artifact pointer disagree.")


def _encode_chunk(payload: bytes, encoding: ChunkEncoding, /) -> bytes:
    if encoding == "identity":
        return payload
    if encoding == "zlib":
        return zlib.compress(payload, level=9)
    raise ValueError("Chunk encoding must be 'identity' or 'zlib'.")


def _decode_and_validate_chunk(
    chunk: ChunkRecord, encoded: bytes, maximum_plaintext_bytes: int, /
) -> bytes:
    if len(encoded) != chunk.encoded_size:
        raise RepositoryCorruptionError("Encoded chunk size mismatch.")
    if hashlib.sha256(encoded).hexdigest() != chunk.encoded_sha256:
        raise RepositoryCorruptionError("Encoded chunk checksum mismatch.")
    if chunk.encoding == "identity":
        plaintext = encoded
    else:
        decompressor = zlib.decompressobj()
        plaintext = decompressor.decompress(encoded, maximum_plaintext_bytes + 1)
        if (
            len(plaintext) > maximum_plaintext_bytes
            or decompressor.unconsumed_tail
            or not decompressor.eof
            or decompressor.unused_data
        ):
            raise RepositoryCorruptionError(
                "Encoded chunk exceeds its bound or is not canonical zlib data."
            )
    if len(plaintext) != chunk.plaintext_size:
        raise RepositoryCorruptionError("Plaintext chunk size mismatch.")
    if hashlib.sha256(plaintext).hexdigest() != chunk.plaintext_sha256:
        raise RepositoryCorruptionError("Plaintext chunk checksum mismatch.")
    return plaintext


def _json_bytes(record: Mapping[str, object], maximum: int, /) -> bytes:
    try:
        payload = (
            json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("Repository metadata must contain finite JSON values.") from error
    if len(payload) > maximum:
        raise ValueError("Repository metadata exceeds its byte bound.")
    return payload


def _json_record(payload: bytes, /) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_json_object,
            parse_constant=reject_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as error:
        raise RepositoryCorruptionError("Repository metadata is invalid JSON.") from error
    if not isinstance(value, dict):
        raise RepositoryCorruptionError("Repository metadata must be a JSON object.")
    return value


def _unique_json_object(
    pairs: list[tuple[str, object]],
    /,
) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"Duplicate JSON member {name!r} is forbidden.")
        value[name] = item
    return value


def _read_json_file(path: Path, maximum: int, /) -> dict[str, object]:
    return _json_record(_read_bounded_file(path, maximum))


def _read_bounded_file(path: Path, maximum: int, /) -> bytes:
    maximum_ = _nonnegative(maximum, "maximum read bytes")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            information = os.fstat(descriptor)
            if not stat.S_ISREG(information.st_mode) or information.st_size > maximum_:
                raise RepositoryCorruptionError(
                    f"Repository file {path.name!r} exceeds its read bound."
                )
            parts: list[bytes] = []
            remaining = maximum_ + 1
            while remaining:
                part = os.read(descriptor, min(1024 * 1024, remaining))
                if not part:
                    break
                parts.append(part)
                remaining -= len(part)
            payload = b"".join(parts)
        finally:
            os.close(descriptor)
    except RepositoryCorruptionError:
        raise
    except (FileNotFoundError, PermissionError, OSError) as error:
        raise RepositoryCorruptionError(
            f"Cannot read repository file {path.name!r}."
        ) from error
    if len(payload) > maximum_ or len(payload) != information.st_size:
        raise RepositoryCorruptionError(
            f"Repository file {path.name!r} changed or exceeded its read bound."
        )
    return payload


def _fsync_directory(path: Path, /) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bounded_bytes(
    value: bytes | bytearray | memoryview,
    maximum: int,
    name: str,
    /,
) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like.")
    view = memoryview(value)
    if view.nbytes > maximum:
        raise ValueError(f"{name} exceeds its byte bound.")
    return bytes(view)


def _positive(value: int, name: str, /) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _nonnegative(value: int, name: str, /) -> int:
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return normalized


def _now() -> int:
    return time.time_ns() // 1_000_000_000


def _namespace(value: str, /) -> str:
    normalized = _object_key(value.rstrip("/"))
    return normalized


def _object_prefix(value: str, /) -> str:
    normalized = value.rstrip("/")
    _object_key(normalized)
    return normalized + "/"


__all__ = [
    "ArtifactManifest",
    "ArtifactRepository",
    "ChunkEncoding",
    "ChunkRecord",
    "ConditionalObjectClient",
    "HPCFilesystemProfile",
    "InMemoryConditionalObjectClient",
    "GarbageCollectionReport",
    "LeaseRecord",
    "LegalHoldRecord",
    "ObjectMetadata",
    "ObjectNotFoundError",
    "ObjectPreconditionError",
    "ObjectStoreProfile",
    "ObjectValue",
    "RepositoryConflictError",
    "RepositoryCorruptionError",
    "RepositoryError",
    "RepositoryTransaction",
    "RetentionPolicy",
    "POSIXArtifactRepository",
    "POSIXRepositoryPolicy",
    "S3ArtifactRepository",
    "TombstoneRecord",
    "UnsupportedRepositoryProfileError",
]
