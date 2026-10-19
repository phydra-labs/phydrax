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
from functools import lru_cache
from pathlib import Path
from typing import assert_never, cast, Iterator, Protocol

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._host_io import open_directory_beneath
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer, positive_integer
from ..logging import emit
from ..qualification._registry import SupportTuple
from ..typing import checked, parse
from ._chunk_repository import (
    ArtifactManifest,
    ArtifactRepository,
    ChunkEncoding,
    ChunkRecord,
    GarbageCollectionReport,
    identifier,
    LeaseRecord,
    LegalHoldRecord,
    RepositoryConflictError,
    RepositoryCorruptionError,
    RepositoryError,
    RepositoryTransaction,
    RetentionPolicy,
    TombstoneRecord,
    UnsupportedRepositoryProfileError,
    validate_object_key,
)


_METADATA_LIMIT = 8 * 1024 * 1024
_POINTER_LIMIT = 64 * 1024
_DEFAULT_CHUNK_LIMIT = 64 * 1024 * 1024
FailureInjector = Callable[[str], None]


def _parse_manifest(payload: bytes, /) -> ArtifactManifest:
    return ArtifactManifest.from_record(_json_record(payload))


class _ManifestDecoder:
    """Retain bounded decoded metadata for one repository's lifetime.

    Callers revalidate durable bytes, pointers, checksums, and commit markers
    before reusing decoded records. Eight metadata-sized entries preserve
    multi-owner restore reuse without retaining unrelated repository lifetimes.
    """

    def __init__(self) -> None:
        self._cached_parse = lru_cache(maxsize=8)(_parse_manifest)

    def decode(self, payload: bytes, /) -> ArtifactManifest:
        if len(payload) > _METADATA_LIMIT:
            return _parse_manifest(payload)
        return self._cached_parse(payload)


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
    ) -> None:
        provider = identifier(provider_id, "provider_id")
        filesystem_ = identifier(filesystem, "filesystem")
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
                "filesystem_profile.features must include the transactional filesystem semantics; got missing "
                + ", ".join(missing)
            )


class POSIXRepositoryPolicy(StrictModule, NonTrainableState):
    """Qualified filesystem profile and bounded-object policy."""

    filesystem_profile: HPCFilesystemProfile
    maximum_chunk_bytes: int = eqx.field(static=True)
    maximum_metadata_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        filesystem_profile: HPCFilesystemProfile,
        /,
        *,
        maximum_chunk_bytes: int = _DEFAULT_CHUNK_LIMIT,
        maximum_metadata_bytes: int = _METADATA_LIMIT,
    ) -> None:
        filesystem_profile.require_transactional_support()
        chunk_limit = positive_integer(maximum_chunk_bytes, "maximum_chunk_bytes")
        metadata_limit = positive_integer(
            maximum_metadata_bytes, "maximum_metadata_bytes"
        )
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
    """Declare S3-compatible semantics required by the object repository."""

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
    ) -> None:
        provider = identifier(provider_id, "provider_id")
        maximum = positive_integer(maximum_object_bytes, "maximum_object_bytes")
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
                "object_store_profile.features must include the conditional and consistency semantics; got missing "
                + ", ".join(missing)
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


class ArtifactGuardRecoveryAuthorization(StrictModule, NonTrainableState):
    """Externally fenced authorization to recover one abandoned metadata guard."""

    provider_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    guard_etag: str = eqx.field(static=True)
    authority_id: str = eqx.field(static=True)
    fencing_evidence_id: str = eqx.field(static=True)
    issued_at: int = eqx.field(static=True)
    worker_fenced: bool = eqx.field(static=True)
    authorization_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        artifact_id: str,
        guard_etag: str,
        authority_id: str,
        fencing_evidence_id: str,
        issued_at: int,
        /,
        *,
        worker_fenced: bool,
    ) -> None:
        if worker_fenced is not True:
            raise ValueError(
                "guard_authorization.worker_fenced must have external worker-fencing evidence; got False"
            )
        self.provider_id = identifier(provider_id, "provider_id")
        self.artifact_id = identifier(artifact_id, "artifact_id")
        self.guard_etag = identifier(guard_etag, "guard_etag")
        self.authority_id = identifier(authority_id, "authority_id")
        self.fencing_evidence_id = identifier(fencing_evidence_id, "fencing_evidence_id")
        self.issued_at = nonnegative_integer(issued_at, "issued_at")
        self.worker_fenced = True
        self.authorization_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "artifact-guard-recovery-authorization",
            "provider_id": self.provider_id,
            "artifact_id": self.artifact_id,
            "guard_etag": self.guard_etag,
            "authority_id": self.authority_id,
            "fencing_evidence_id": self.fencing_evidence_id,
            "issued_at": self.issued_at,
            "worker_fenced": self.worker_fenced,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self._content_record(),
            "authorization_id": self.authorization_id,
        }

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], /
    ) -> ArtifactGuardRecoveryAuthorization:
        if record.get("kind") != "artifact-guard-recovery-authorization":
            raise ValueError(
                "guard_authorization.record.kind must identify an artifact guard authorization; got a different record kind"
            )
        value = cls(
            str(record.get("provider_id", "")),
            str(record.get("artifact_id", "")),
            str(record.get("guard_etag", "")),
            str(record.get("authority_id", "")),
            str(record.get("fencing_evidence_id", "")),
            # __init__ validates and normalizes issued_at through _nonnegative.
            cast(int, record.get("issued_at")),
            worker_fenced=record.get("worker_fenced") is True,
        )
        if record.get("authorization_id") != value.authorization_id:
            raise ValueError(
                "guard_authorization.authorization_id must match the decoded fields; got authorization identity mismatch"
            )
        return value


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

    def __init__(self, /, *, maximum_object_bytes: int) -> None:
        self.maximum_object_bytes = positive_integer(
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
        key_ = validate_object_key(key)
        payload = _bounded_bytes(data, self.maximum_object_bytes, "object")
        with self._lock:
            if key_ in self._objects:
                raise ObjectPreconditionError(
                    f"object.key must be absent before conditional creation; got {key_!r} already exists"
                )
            etag = self._next_etag(payload)
            self._objects[key_] = (payload, etag)
            return ObjectMetadata(key_, etag, len(payload))

    def replace_object(
        self, key: str, data: bytes, expected_etag: str, /
    ) -> ObjectMetadata:
        key_ = validate_object_key(key)
        expected = identifier(expected_etag, "expected_etag")
        payload = _bounded_bytes(data, self.maximum_object_bytes, "object")
        with self._lock:
            current = self._objects.get(key_)
            if current is None or current[1] != expected:
                raise ObjectPreconditionError(
                    f"object.expected_etag must match the current token for replacement; got conditional replace failed for {key_!r}"
                )
            etag = self._next_etag(payload)
            self._objects[key_] = (payload, etag)
            return ObjectMetadata(key_, etag, len(payload))

    def read_object(self, key: str, /, *, maximum_bytes: int) -> ObjectValue:
        key_ = validate_object_key(key)
        maximum = nonnegative_integer(maximum_bytes, "maximum_bytes")
        with self._lock:
            self._read_requests.append((key_, maximum))
            current = self._objects.get(key_)
            if current is None:
                raise ObjectNotFoundError(
                    f"object.key must identify a stored object; got {key_!r} does not exist"
                )
            payload, etag = current
            if len(payload) > maximum:
                raise RepositoryCorruptionError(
                    f"object.size must fit its bounded read limit; got {key_!r} exceeds the read limit"
                )
            return ObjectValue(ObjectMetadata(key_, etag, len(payload)), payload)

    def delete_object(self, key: str, /, *, expected_etag: str | None = None) -> None:
        key_ = validate_object_key(key)
        with self._lock:
            current = self._objects.get(key_)
            if current is None:
                raise ObjectNotFoundError(
                    f"object.key must identify a stored object; got {key_!r} does not exist"
                )
            if expected_etag is not None and current[1] != expected_etag:
                raise ObjectPreconditionError(
                    f"object.expected_etag must match the current token for deletion; got conditional delete failed for {key_!r}"
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

    @checked
    def __init__(
        self,
        root: str | os.PathLike[str],
        policy: POSIXRepositoryPolicy,
        /,
        *,
        failure_injector: FailureInjector | None = None,
    ) -> None:
        policy.filesystem_profile.require_transactional_support()
        self.root = _admit_repository_root(root)
        self.policy = policy
        self.provider_id = policy.filesystem_profile.provider_id
        self.maximum_chunk_bytes = policy.maximum_chunk_bytes
        self.maximum_metadata_bytes = policy.maximum_metadata_bytes
        self.failure_injector = failure_injector
        self._manifest_decoder = _ManifestDecoder()
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
        with open_directory_beneath(
            self.root,
            trusted_root=self.root.parent,
            maximum_depth=1,
        ) as opened_root:
            self._root_descriptor = os.dup(opened_root.descriptor)
        self._directory_descriptors = {
            name: os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=self._root_descriptor,
            )
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
            )
        }
        self._layout_identities = self._repository_layout_identities()
        root_status = os.fstat(self._root_descriptor)
        self._pinned_layout_identities = {
            "": (root_status.st_dev, root_status.st_ino),
            **{
                name: (
                    os.fstat(descriptor).st_dev,
                    os.fstat(descriptor).st_ino,
                )
                for name, descriptor in self._directory_descriptors.items()
            },
        }
        if self._pinned_layout_identities != self._layout_identities:
            raise RepositoryCorruptionError(
                "repository.storage_layout must remain unchanged while descriptors are pinned; got storage layout changed"
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
        artifact = identifier(artifact_id, "artifact_id")
        writer = identifier(writer_id, "writer_id")
        attempt = (
            secrets.token_hex(16)
            if attempt_id is None
            else identifier(attempt_id, "attempt_id")
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
            self._make_directory(staging)
            try:
                self._make_directory(staging / "chunks")
                self._atomic_create_immutable(
                    staging / "transaction.json",
                    _json_bytes(transaction.to_record(), self.maximum_metadata_bytes),
                )
                self._fsync_repository_directory(staging / "chunks")
                self._fsync_repository_directory(staging)
                if self._repository_path_exists(root):
                    raise RepositoryConflictError(
                        f"repository.attempt_id must name an unused attempt root; got {attempt!r} already exists"
                    )
                self._rename_repository_path(staging, root)
            finally:
                if self._repository_path_exists(staging):
                    self._remove_repository_tree(staging)
                    self._fsync_repository_directory(self.root / "staging")
        emit(
            "DEBUG",
            "lifecycle.repository.transaction.started",
            "Repository transaction started",
            artifact_id=transaction.artifact_id,
            provider_id=self.provider_id,
            transaction_id=transaction.transaction_id,
        )
        return transaction

    @checked
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
        logical = identifier(logical_name, "logical_name")
        index_ = nonnegative_integer(index, "index")
        offset_ = nonnegative_integer(offset, "offset")
        plaintext = _bounded_bytes(payload, self.maximum_chunk_bytes, "chunk plaintext")
        encoded = _encode_chunk(plaintext, encoding)
        if len(encoded) > self.maximum_chunk_bytes:
            raise ValueError(
                "chunk.encoded_size must not exceed maximum_chunk_bytes; got encoded chunk byte limit exceeded"
            )
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

    @checked
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
                raise RepositoryConflictError(
                    "transaction.transaction_id must not already be published; got already committed"
                )
            current_manifest_id = None if current is None else str(current["manifest_id"])
            if current_manifest_id != transaction.base_manifest_id:
                raise RepositoryConflictError(
                    "transaction.base_manifest_id must match the current artifact pointer; got pointer changed after this transaction began"
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
        emit(
            "INFO",
            "lifecycle.repository.manifest.committed",
            "Repository manifest committed",
            artifact_id=manifest.artifact_id,
            chunk_bytes=sum(chunk.plaintext_size for chunk in manifest.chunks),
            chunk_count=len(manifest.chunks),
            manifest_id=manifest.manifest_id,
            provider_id=self.provider_id,
            transaction_id=transaction.transaction_id,
        )
        return manifest

    def get_manifest(self, artifact_id: str, /) -> ArtifactManifest:
        self._verify_repository_layout()
        artifact = identifier(artifact_id, "artifact_id")
        pointer = self._read_pointer_optional(artifact)
        if pointer is None:
            raise ObjectNotFoundError(
                f"repository.artifact_id must identify a committed artifact; got {artifact!r} does not exist"
            )
        return self._manifest_from_pointer(pointer, artifact)

    @checked
    def read_chunk(
        self,
        manifest: ArtifactManifest,
        chunk: ChunkRecord,
        /,
        *,
        maximum_plaintext_bytes: int | None = None,
    ) -> bytes:
        self._verify_repository_layout()
        self._validate_manifest_argument(manifest, chunk)
        maximum = (
            self.maximum_chunk_bytes
            if maximum_plaintext_bytes is None
            else positive_integer(maximum_plaintext_bytes, "maximum_plaintext_bytes")
        )
        if chunk.plaintext_size > maximum:
            raise RepositoryCorruptionError(
                "chunk.plaintext_size must fit the requested plaintext bound; got plaintext bound exceeded"
            )
        encoded = self._read_bounded_file(
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
        logical = identifier(logical_name, "logical_name")
        chunks = tuple(item for item in manifest.chunks if item.logical_name == logical)
        total = sum(item.plaintext_size for item in chunks)
        maximum = positive_integer(maximum_bytes, "maximum_bytes")
        if not chunks:
            raise ObjectNotFoundError(
                f"artifact.logical_name must identify a stored payload; got payload {logical!r} does not exist"
            )
        if total > maximum:
            raise RepositoryCorruptionError(
                "artifact.payload_size must fit the requested byte bound; got payload byte bound exceeded"
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
            artifact = identifier(artifact_id, "artifact_id")
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
            if not self._repository_path_exists(directory):
                self._make_directory(directory)
            self._atomic_create_immutable(
                directory / f"{lease.lease_id}.json",
                _json_bytes(lease.to_record(), self.maximum_metadata_bytes),
            )
            self._fsync_repository_directory(directory)
            return lease

    @checked
    def release_lease(self, lease: LeaseRecord, /) -> None:
        with self._exclusive_lock():
            if lease.provider_id != self.provider_id:
                raise TypeError(
                    "lease.provider_id must match the repository provider; got a foreign lease provider"
                )
            path = self.root / "leases" / lease.artifact_id / f"{lease.lease_id}.json"
            persisted = LeaseRecord.from_record(
                self._read_json_file(path, self.maximum_metadata_bytes)
            )
            if persisted.record_id != lease.record_id:
                raise RepositoryConflictError(
                    "lease.record_id must match the stored lease before release; got release record mismatch"
                )
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
            artifact = identifier(artifact_id, "artifact_id")
            self.get_manifest(artifact)
            hold = LegalHoldRecord(
                self.provider_id,
                artifact,
                secrets.token_hex(16) if hold_id is None else hold_id,
                authority,
                _now() if placed_at is None else placed_at,
            )
            directory = self.root / "holds" / artifact
            if not self._repository_path_exists(directory):
                self._make_directory(directory)
            self._atomic_create_immutable(
                directory / f"{hold.hold_id}.json",
                _json_bytes(hold.to_record(), self.maximum_metadata_bytes),
            )
            self._fsync_repository_directory(directory)
            return hold

    @checked
    def release_legal_hold(self, hold: LegalHoldRecord, /) -> None:
        with self._exclusive_lock():
            if hold.provider_id != self.provider_id:
                raise TypeError(
                    "hold.provider_id must match the repository provider; got a foreign legal-hold provider"
                )
            path = self.root / "holds" / hold.artifact_id / f"{hold.hold_id}.json"
            persisted = LegalHoldRecord.from_record(
                self._read_json_file(path, self.maximum_metadata_bytes)
            )
            if persisted.record_id != hold.record_id:
                raise RepositoryConflictError(
                    "hold.record_id must match the stored legal hold before release; got release record mismatch"
                )
            self._unlink_existing(path)

    @checked
    def set_retention(self, artifact_id: str, policy: RetentionPolicy, /) -> None:
        with self._exclusive_lock():
            artifact = identifier(artifact_id, "artifact_id")
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
            artifact = identifier(artifact_id, "artifact_id")
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
        emit(
            "WARNING",
            "lifecycle.repository.tombstone.committed",
            "Repository tombstone committed",
            artifact_id=record.artifact_id,
            provider_id=self.provider_id,
        )
        return record

    @checked
    def collect_garbage(
        self,
        /,
        *,
        now: int | None = None,
        default_policy: RetentionPolicy | None = None,
    ) -> GarbageCollectionReport:
        with self._exclusive_lock():
            report = self._collect_garbage_locked(now=now, default_policy=default_policy)
        emit(
            "INFO",
            "lifecycle.repository.garbage_collection.completed",
            "Repository garbage collection completed",
            expired_lease_count=len(report.expired_lease_ids),
            provider_id=self.provider_id,
            removed_artifact_count=len(report.removed_artifact_ids),
            removed_attempt_count=len(report.removed_attempt_ids),
        )
        return report

    def _collect_garbage_locked(
        self,
        /,
        *,
        now: int | None,
        default_policy: RetentionPolicy | None,
    ) -> GarbageCollectionReport:
        collected = _now() if now is None else nonnegative_integer(now, "now")
        default = RetentionPolicy() if default_policy is None else default_policy
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
            directory = self.root / name
            directory.mkdir(mode=0o700, exist_ok=True)
            information = directory.lstat()
            if not stat.S_ISDIR(information.st_mode) or stat.S_ISLNK(information.st_mode):
                raise RepositoryCorruptionError(
                    f"repository.storage_component.kind must be a real directory; got non-directory or symlink for {name!r}"
                )
        lock = self.root / "locks" / "repository.lock"
        descriptor = os.open(
            lock,
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RepositoryCorruptionError(
                    "repository.lock.kind must be a regular file; got a non-regular lock file"
                )
        finally:
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

    def _repository_layout_identities(self) -> dict[str, tuple[int, int]]:
        identities: dict[str, tuple[int, int]] = {}
        for name in (
            "",
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
            path = self.root if not name else self.root / name
            information = path.lstat()
            if not stat.S_ISDIR(information.st_mode) or stat.S_ISLNK(information.st_mode):
                raise RepositoryCorruptionError(
                    "repository.storage_layout.kind must contain real directories; got a symbolic link or non-directory"
                )
            identities[name] = (information.st_dev, information.st_ino)
        return identities

    def _verify_repository_layout(self) -> None:
        current = self._repository_layout_identities()
        if (
            current != self._layout_identities
            or current != self._pinned_layout_identities
        ):
            raise RepositoryCorruptionError(
                "repository.storage_layout must match its initialized and pinned identities; got storage layout changed after initialization"
            )

    def _remove_root(self, attempt_id: str, /) -> None:
        source = self._attempt_path(attempt_id)
        destination = self.root / "trash" / attempt_id
        self._rename_repository_path(source, destination, replace=True)
        self._remove_repository_tree(destination)
        self._fsync_repository_directory(self.root / "trash")

    def _cleanup_trash(self) -> None:
        trash = self.root / "trash"
        for name, information in self._repository_directory_entries(trash):
            if not stat.S_ISDIR(information.st_mode):
                raise RepositoryCorruptionError(
                    "repository.trash.entries must be directories; got a non-directory trash entry"
                )
            self._remove_repository_tree(trash / name)
        self._fsync_repository_directory(trash)

    def _cleanup_staging(self, now: int, grace_seconds: int, /) -> None:
        staging = self.root / "staging"
        for name, information in self._repository_directory_entries(staging):
            if not stat.S_ISDIR(information.st_mode):
                raise RepositoryCorruptionError(
                    "repository.staging.entries must be directories; got a non-directory staging entry"
                )
            if now - int(information.st_mtime) >= grace_seconds:
                self._remove_repository_tree(staging / name)
        self._fsync_repository_directory(staging)

    def _validate_transaction(self, transaction: RepositoryTransaction, /) -> None:
        if transaction.provider_id != self.provider_id:
            raise RepositoryConflictError(
                "transaction.provider_id must match the repository provider; got a foreign transaction provider"
            )
        path = self._attempt_path(transaction.attempt_id) / "transaction.json"
        record = RepositoryTransaction.from_record(
            self._read_json_file(path, self.maximum_metadata_bytes)
        )
        if record.transaction_id != transaction.transaction_id:
            raise RepositoryConflictError(
                "transaction.transaction_id must match its stored attempt; got transaction record mismatch"
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
                    "chunk.object_key must belong to the attempt-private root; got a foreign manifest chunk key"
                )
            encoded = self._read_bounded_file(
                self.root / chunk.object_key,
                min(self.maximum_chunk_bytes, chunk.encoded_size),
            )
            _decode_and_validate_chunk(chunk, encoded, self.maximum_chunk_bytes)

    def _validate_manifest_argument(
        self, manifest: ArtifactManifest, chunk: ChunkRecord, /
    ) -> None:
        if manifest.provider_id != self.provider_id:
            raise RepositoryConflictError(
                "manifest.provider_id must match the repository provider; got a foreign manifest provider"
            )
        if chunk.chunk_id not in {item.chunk_id for item in manifest.chunks}:
            raise RepositoryConflictError(
                "manifest.chunks must include the requested chunk ID; got a chunk not present in the manifest"
            )

    def _manifest_from_pointer(
        self, pointer: Mapping[str, object], artifact: str, /
    ) -> ArtifactManifest:
        _validate_pointer(pointer, self.provider_id, artifact)
        attempt = identifier(str(pointer["attempt_id"]), "attempt_id")
        manifest_payload = self._read_bounded_file(
            self._attempt_path(attempt) / "manifest.json", self.maximum_metadata_bytes
        )
        if hashlib.sha256(manifest_payload).hexdigest() != pointer["manifest_sha256"]:
            raise RepositoryCorruptionError(
                "artifact.manifest_sha256 must match durable manifest bytes; got manifest checksum failed"
            )
        manifest = self._manifest_decoder.decode(manifest_payload)
        if (
            manifest.manifest_id != pointer["manifest_id"]
            or manifest.transaction_id != pointer["transaction_id"]
            or manifest.provider_id != self.provider_id
            or manifest.artifact_id != artifact
        ):
            raise RepositoryCorruptionError(
                "artifact.pointer.fields must agree with the manifest; got pointer and manifest disagreement"
            )
        marker = self._read_json_file(
            self._attempt_path(attempt) / "COMMIT", self.maximum_metadata_bytes
        )
        _validate_commit_marker(marker, pointer)
        expected_prefix = f"roots/{attempt}/chunks/"
        if any(
            not item.object_key.startswith(expected_prefix) for item in manifest.chunks
        ):
            raise RepositoryCorruptionError(
                "manifest.chunks.object_key must stay inside the immutable root; got an escaping chunk key"
            )
        return manifest

    def _read_pointer_optional(self, artifact: str, /) -> dict[str, object] | None:
        path = self._pointer_path(artifact)
        if not self._repository_path_exists(path):
            return None
        record = self._read_json_file(path, _POINTER_LIMIT)
        _validate_pointer(record, self.provider_id, artifact)
        return record

    def _all_pointers(self) -> dict[str, dict[str, object]]:
        pointers: dict[str, dict[str, object]] = {}
        directory = self.root / "artifacts"
        for name, _ in self._repository_directory_entries(directory):
            if not name.endswith(".pointer"):
                continue
            artifact = identifier(name.removesuffix(".pointer"), "artifact_id")
            path = directory / name
            pointer = self._read_json_file(path, _POINTER_LIMIT)
            _validate_pointer(pointer, self.provider_id, artifact)
            pointers[artifact] = pointer
        return pointers

    def _all_roots(
        self,
    ) -> tuple[tuple[RepositoryTransaction, ArtifactManifest | None], ...]:
        roots: list[tuple[RepositoryTransaction, ArtifactManifest | None]] = []
        directory = self.root / "roots"
        for name, information in self._repository_directory_entries(directory):
            if not stat.S_ISDIR(information.st_mode):
                raise RepositoryCorruptionError(
                    "repository.roots.entries must be directories; got a non-directory root entry"
                )
            attempt = identifier(name, "attempt_id")
            path = directory / attempt
            transaction = RepositoryTransaction.from_record(
                self._read_json_file(
                    path / "transaction.json", self.maximum_metadata_bytes
                )
            )
            if (
                transaction.attempt_id != attempt
                or transaction.provider_id != self.provider_id
            ):
                raise RepositoryCorruptionError(
                    "repository.attempt.transaction_ids must match the root attempt and provider; got attempt transaction identity mismatch"
                )
            manifest_path = path / "manifest.json"
            marker_path = path / "COMMIT"
            marker_exists = self._repository_path_exists(marker_path)
            manifest_exists = self._repository_path_exists(manifest_path)
            if marker_exists and not manifest_exists:
                raise RepositoryCorruptionError(
                    "artifact.commit_marker.manifest must be present for a committed root; got commit marker with no manifest"
                )
            manifest = (
                None
                if not marker_exists
                else ArtifactManifest.from_record(
                    self._read_json_file(manifest_path, self.maximum_metadata_bytes)
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
                        "artifact.manifest.fields must match the root transaction and chunk ownership; got committed root manifest mismatch"
                    )
                marker = self._read_json_file(marker_path, self.maximum_metadata_bytes)
                expected = _pointer_record(
                    transaction,
                    manifest,
                    hashlib.sha256(
                        self._read_bounded_file(
                            manifest_path, self.maximum_metadata_bytes
                        )
                    ).hexdigest(),
                )
                _validate_commit_marker(marker, expected)
            roots.append((transaction, manifest))
        return tuple(roots)

    def _expire_and_collect_leases(self, now: int, /) -> tuple[str, ...]:
        expired: list[str] = []
        directory = self.root / "leases"
        for path in self._nested_repository_files(directory, suffix=".json"):
            lease = LeaseRecord.from_record(
                self._read_json_file(path, self.maximum_metadata_bytes)
            )
            if lease.provider_id != self.provider_id:
                raise RepositoryCorruptionError(
                    "lease.provider_id must match the repository provider; got lease provider mismatch"
                )
            if lease.expires_at <= now:
                self._unlink_existing(path)
                expired.append(lease.lease_id)
        return tuple(expired)

    def _active_lease_artifacts(self, now: int, /) -> set[str]:
        active: set[str] = set()
        directory = self.root / "leases"
        for path in self._nested_repository_files(directory, suffix=".json"):
            lease = LeaseRecord.from_record(
                self._read_json_file(path, self.maximum_metadata_bytes)
            )
            if lease.provider_id != self.provider_id:
                raise RepositoryCorruptionError(
                    "lease.provider_id must match the repository provider; got lease provider mismatch"
                )
            if lease.expires_at > now:
                active.add(lease.artifact_id)
        return active

    def _held_artifacts(self) -> set[str]:
        held: set[str] = set()
        directory = self.root / "holds"
        for path in self._nested_repository_files(directory, suffix=".json"):
            hold = LegalHoldRecord.from_record(
                self._read_json_file(path, self.maximum_metadata_bytes)
            )
            if hold.provider_id != self.provider_id:
                raise RepositoryCorruptionError(
                    "hold.provider_id must match the repository provider; got legal-hold provider mismatch"
                )
            held.add(hold.artifact_id)
        return held

    def _retention_for(
        self, artifact: str, default: RetentionPolicy, /
    ) -> RetentionPolicy:
        path = self.root / "retention" / f"{artifact}.json"
        if not self._repository_path_exists(path):
            return default
        record = self._read_json_file(path, self.maximum_metadata_bytes)
        policy = record.get("policy")
        if (
            record.get("kind") != "artifact-retention"
            or record.get("provider_id") != self.provider_id
            or record.get("artifact_id") != artifact
            or not isinstance(policy, Mapping)
        ):
            raise RepositoryCorruptionError(
                "artifact.retention.fields must match the artifact and provider with a mapping policy; got invalid retention record"
            )
        return RetentionPolicy.from_record(policy)

    def _tombstone_optional(self, artifact: str, /) -> TombstoneRecord | None:
        path = self.root / "tombstones" / f"{artifact}.json"
        if not self._repository_path_exists(path):
            return None
        value = TombstoneRecord.from_record(
            self._read_json_file(path, self.maximum_metadata_bytes)
        )
        if value.provider_id != self.provider_id or value.artifact_id != artifact:
            raise RepositoryCorruptionError(
                "artifact.tombstone.fields must match the artifact and provider; got tombstone provider or artifact mismatch"
            )
        return value

    @contextmanager
    def _open_parent_descriptor(self, path: Path, /) -> Iterator[tuple[int, str]]:
        try:
            relative = path.relative_to(self.root)
        except ValueError as error:
            raise RepositoryCorruptionError(
                "repository.path must remain inside its pinned root; got a path escaped its pinned root"
            ) from error
        if not relative.parts or any(
            component in ("", ".", "..") for component in relative.parts
        ):
            raise RepositoryCorruptionError(
                "repository.path must use nonempty relative components without dot segments; "
                f"got {relative.parts!r}"
            )
        first = relative.parts[0]
        if len(relative.parts) > 1 and first in self._directory_descriptors:
            descriptors = [os.dup(self._directory_descriptors[first])]
            traversed = relative.parts[1:-1]
        else:
            descriptors = [os.dup(self._root_descriptor)]
            traversed = relative.parts[:-1]
        try:
            parent = descriptors[-1]
            for component in traversed:
                parent = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent,
                )
                descriptors.append(parent)
            yield parent, relative.parts[-1]
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def _repository_directory_entries(
        self,
        directory: Path,
        /,
    ) -> tuple[tuple[str, os.stat_result], ...]:
        with self._open_parent_descriptor(directory) as (parent, name):
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            try:
                entries = tuple(sorted(os.listdir(descriptor)))
                return tuple(
                    (
                        entry,
                        os.stat(entry, dir_fd=descriptor, follow_symlinks=False),
                    )
                    for entry in entries
                )
            finally:
                os.close(descriptor)

    def _nested_repository_files(
        self,
        directory: Path,
        /,
        *,
        suffix: str,
    ) -> tuple[Path, ...]:
        paths: list[Path] = []
        for child_name, child_status in self._repository_directory_entries(directory):
            if not stat.S_ISDIR(child_status.st_mode):
                raise RepositoryCorruptionError(
                    f"repository.directory.children must be directories; got a non-directory child in {directory.name!r}"
                )
            child = directory / child_name
            for name, information in self._repository_directory_entries(child):
                if stat.S_ISDIR(information.st_mode):
                    raise RepositoryCorruptionError(
                        f"repository.directory.files must not contain nested directories; got a nested directory in {child.name!r}"
                    )
                if name.endswith(suffix):
                    paths.append(child / name)
        return tuple(sorted(paths))

    def _remove_repository_tree(self, path: Path, /) -> None:
        with self._open_parent_descriptor(path) as (parent, name):
            information = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(information.st_mode):
                raise RepositoryCorruptionError(
                    "repository.cleanup_target.kind must be a directory; got a non-directory cleanup target"
                )
            shutil.rmtree(name, dir_fd=parent)

    def _repository_path_exists(self, path: Path, /) -> bool:
        with self._open_parent_descriptor(path) as (parent, name):
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return False
            return True

    def _make_directory(self, path: Path, /) -> None:
        with self._open_parent_descriptor(path) as (parent, name):
            os.mkdir(name, 0o700, dir_fd=parent)
            os.fsync(parent)

    def _fsync_repository_directory(self, path: Path, /) -> None:
        with self._open_parent_descriptor(path) as (parent, name):
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _rename_repository_path(
        self, source: Path, destination: Path, /, *, replace: bool = False
    ) -> None:
        with self._open_parent_descriptor(source) as (source_parent, source_name):
            with self._open_parent_descriptor(destination) as (
                destination_parent,
                destination_name,
            ):
                operation = os.replace if replace else os.rename
                operation(
                    source_name,
                    destination_name,
                    src_dir_fd=source_parent,
                    dst_dir_fd=destination_parent,
                )
                os.fsync(source_parent)
                if destination_parent != source_parent:
                    os.fsync(destination_parent)

    def _read_bounded_file(self, path: Path, maximum: int, /) -> bytes:
        maximum_ = nonnegative_integer(maximum, "maximum read bytes")
        with self._open_parent_descriptor(path) as (parent, name):
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_:
                    raise RepositoryCorruptionError(
                        f"repository.file.size must fit its byte bound on a regular file; got {path.name!r} violates its byte bound"
                    )
                payload = bytearray()
                while len(payload) <= maximum_:
                    block = os.read(descriptor, min(1 << 20, maximum_ + 1 - len(payload)))
                    if not block:
                        break
                    payload.extend(block)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
        if (
            len(payload) > maximum_
            or len(payload) != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise RepositoryCorruptionError(
                f"repository.file.identity must remain unchanged during the read; got {path.name!r} changed while being read"
            )
        return bytes(payload)

    def _read_json_file(self, path: Path, maximum: int, /) -> dict[str, object]:
        return _json_record(self._read_bounded_file(path, maximum))

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self._verify_repository_layout()
        lock_path = self.root / "locks" / "repository.lock"
        with self._open_parent_descriptor(lock_path) as (parent, name):
            descriptor = os.open(
                name,
                os.O_RDWR | os.O_NOFOLLOW,
                dir_fd=parent,
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._verify_repository_layout()
            yield
            self._verify_repository_layout()
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _replace_pointer(self, artifact: str, payload: bytes, /) -> None:
        self._atomic_replace_file(self._pointer_path(artifact), payload)

    def _atomic_replace_file(self, destination: Path, payload: bytes, /) -> None:
        temporary_name = f".{destination.name}.{secrets.token_hex(16)}.tmp"
        with self._open_parent_descriptor(destination) as (parent, name):
            self._create_immutable_at(parent, temporary_name, payload)
            try:
                os.replace(
                    temporary_name,
                    name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                )
                os.fsync(parent)
            finally:
                try:
                    os.unlink(temporary_name, dir_fd=parent)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _create_immutable_at(parent: int, name: str, payload: bytes, /) -> None:
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
        except FileExistsError as error:
            raise RepositoryConflictError(
                f"repository.immutable_object.key must be absent before creation; got {name!r} already exists"
            ) from error
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RepositoryError(
                        f"repository.file.write_progress must be positive until payload completion; got failed write for {name}"
                    )
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _create_immutable_file(self, path: Path, payload: bytes, /) -> None:
        with self._open_parent_descriptor(path) as (parent, name):
            self._create_immutable_at(parent, name, payload)

    def _atomic_create_immutable(self, path: Path, payload: bytes, /) -> None:
        temporary_name = f".{path.name}.{secrets.token_hex(16)}.immutable.tmp"
        with self._open_parent_descriptor(path) as (parent, name):
            self._create_immutable_at(parent, temporary_name, payload)
            try:
                try:
                    os.link(
                        temporary_name,
                        name,
                        src_dir_fd=parent,
                        dst_dir_fd=parent,
                        follow_symlinks=False,
                    )
                except FileExistsError as error:
                    raise RepositoryConflictError(
                        f"repository.immutable_object.key must be absent before creation; got {name!r} already exists"
                    ) from error
                os.fsync(parent)
            finally:
                os.unlink(temporary_name, dir_fd=parent)

    def _create_or_verify_immutable(self, path: Path, payload: bytes, /) -> None:
        temporary_name = f".{path.name}.{secrets.token_hex(16)}.immutable.tmp"
        with self._open_parent_descriptor(path) as (parent, name):
            self._create_immutable_at(parent, temporary_name, payload)
            try:
                try:
                    os.link(
                        temporary_name,
                        name,
                        src_dir_fd=parent,
                        dst_dir_fd=parent,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    existing = self._read_bounded_file(path, len(payload))
                    if existing != payload:
                        raise RepositoryConflictError(
                            f"repository.immutable_object.payload must match existing immutable bytes; got conflicting object {name!r}"
                        )
                else:
                    os.fsync(parent)
            finally:
                os.unlink(temporary_name, dir_fd=parent)

    def _unlink_existing(self, path: Path, /) -> None:
        with self._open_parent_descriptor(path) as (parent, name):
            try:
                os.unlink(name, dir_fd=parent)
            except FileNotFoundError as error:
                raise RepositoryConflictError(
                    f"repository.record.key must exist until unlink; got record {name!r} no longer exists"
                ) from error
            os.fsync(parent)

    def _attempt_path(self, attempt: str, /) -> Path:
        return self.root / "roots" / identifier(attempt, "attempt_id")

    def _pointer_path(self, artifact: str, /) -> Path:
        return self.root / "artifacts" / f"{identifier(artifact, 'artifact_id')}.pointer"

    @staticmethod
    def _chunk_key(attempt: str, logical: str, index: int, /) -> str:
        return f"roots/{attempt}/chunks/{logical}.{index:08d}.bin"

    def _fail(self, point: str, /) -> None:
        if self.failure_injector is not None:
            self.failure_injector(point)


class S3ArtifactRepository:
    """Transactional S3-compatible repository over conditional whole objects."""

    @checked
    def __init__(
        self,
        client: ConditionalObjectClient,
        profile: ObjectStoreProfile,
        namespace: str,
        /,
        *,
        maximum_chunk_bytes: int = _DEFAULT_CHUNK_LIMIT,
        failure_injector: FailureInjector | None = None,
    ) -> None:
        profile.require_transactional_support()
        maximum = positive_integer(maximum_chunk_bytes, "maximum_chunk_bytes")
        if maximum > profile.maximum_object_bytes:
            raise UnsupportedRepositoryProfileError(
                "repository.maximum_chunk_bytes must fit the object-store profile bound; got a chunk limit larger than the profile bound"
            )
        self.client = client
        self.profile = profile
        self.namespace = _namespace(namespace)
        self.provider_id = profile.provider_id
        self.maximum_chunk_bytes = maximum
        self.maximum_metadata_bytes = min(_METADATA_LIMIT, profile.maximum_object_bytes)
        self.failure_injector = failure_injector
        self._manifest_decoder = _ManifestDecoder()
        self.support_tuple = SupportTuple(
            "artifact.repository",
            {
                "provider": self.provider_id,
                "transport": "s3-compatible",
                "profile_id": profile.profile_id,
                "maximum_chunk_bytes": maximum,
                "guard_recovery": "explicit-external-worker-fence",
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
        artifact = identifier(artifact_id, "artifact_id")
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
        emit(
            "DEBUG",
            "lifecycle.repository.transaction.started",
            "Repository transaction started",
            artifact_id=transaction.artifact_id,
            provider_id=self.provider_id,
            transaction_id=transaction.transaction_id,
        )
        return transaction

    @checked
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
        logical = identifier(logical_name, "logical_name")
        index_ = nonnegative_integer(index, "index")
        offset_ = nonnegative_integer(offset, "offset")
        plaintext = _bounded_bytes(payload, self.maximum_chunk_bytes, "chunk plaintext")
        encoded = _encode_chunk(plaintext, encoding)
        if len(encoded) > self.maximum_chunk_bytes:
            raise ValueError(
                "chunk.encoded_size must not exceed maximum_chunk_bytes; got encoded chunk byte limit exceeded"
            )
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

    @checked
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
        emit(
            "INFO",
            "lifecycle.repository.manifest.committed",
            "Repository manifest committed",
            artifact_id=manifest.artifact_id,
            chunk_bytes=sum(chunk.plaintext_size for chunk in manifest.chunks),
            chunk_count=len(manifest.chunks),
            manifest_id=manifest.manifest_id,
            provider_id=self.provider_id,
            transaction_id=transaction.transaction_id,
        )
        return manifest

    def get_manifest(self, artifact_id: str, /) -> ArtifactManifest:
        artifact = identifier(artifact_id, "artifact_id")
        value = self._read_pointer_optional(artifact)
        if value is None:
            raise ObjectNotFoundError(
                f"repository.artifact_id must identify a committed artifact; got {artifact!r} does not exist"
            )
        pointer = _json_record(value.data)
        _validate_pointer(pointer, self.provider_id, artifact)
        manifest_payload = self.client.read_object(
            self._manifest_key(str(pointer["attempt_id"])),
            maximum_bytes=self.maximum_metadata_bytes,
        ).data
        if hashlib.sha256(manifest_payload).hexdigest() != pointer["manifest_sha256"]:
            raise RepositoryCorruptionError(
                "artifact.manifest_sha256 must match durable manifest bytes; got manifest checksum failed"
            )
        manifest = self._manifest_decoder.decode(manifest_payload)
        if (
            manifest.manifest_id != pointer["manifest_id"]
            or manifest.transaction_id != pointer["transaction_id"]
            or manifest.provider_id != self.provider_id
            or manifest.artifact_id != artifact
        ):
            raise RepositoryCorruptionError(
                "artifact.pointer.fields must agree with the manifest; got pointer and manifest disagreement"
            )
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
            raise RepositoryCorruptionError(
                "manifest.chunks.object_key must stay inside the immutable root; got an escaping chunk key"
            )
        return manifest

    @checked
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
            else positive_integer(maximum_plaintext_bytes, "maximum_plaintext_bytes")
        )
        if chunk.plaintext_size > maximum:
            raise RepositoryCorruptionError(
                "chunk.plaintext_size must fit the requested plaintext bound; got plaintext bound exceeded"
            )
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
        logical = identifier(logical_name, "logical_name")
        chunks = tuple(item for item in manifest.chunks if item.logical_name == logical)
        total = sum(item.plaintext_size for item in chunks)
        maximum = positive_integer(maximum_bytes, "maximum_bytes")
        if not chunks:
            raise ObjectNotFoundError(
                f"artifact.logical_name must identify a stored payload; got payload {logical!r} does not exist"
            )
        if total > maximum:
            raise RepositoryCorruptionError(
                "artifact.payload_size must fit the requested byte bound; got payload byte bound exceeded"
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
        artifact = identifier(artifact_id, "artifact_id")
        guard = self._acquire_artifact_guard(artifact)
        try:
            pointer = self._read_pointer_optional(artifact)
            if pointer is None:
                raise ObjectNotFoundError(
                    f"repository.artifact_id must identify a committed artifact; got {artifact!r} does not exist"
                )
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

    @checked
    def release_lease(self, lease: LeaseRecord, /) -> None:
        if lease.provider_id != self.provider_id:
            raise TypeError(
                "lease.provider_id must match the repository provider; got a foreign lease provider"
            )
        key = self._lease_key(lease.artifact_id, lease.lease_id)
        value = self.client.read_object(key, maximum_bytes=self.maximum_metadata_bytes)
        persisted = LeaseRecord.from_record(_json_record(value.data))
        if persisted.record_id != lease.record_id:
            raise RepositoryConflictError(
                "lease.record_id must match the stored lease before release; got release record mismatch"
            )
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
        artifact = identifier(artifact_id, "artifact_id")
        guard = self._acquire_artifact_guard(artifact)
        try:
            if self._read_pointer_optional(artifact) is None:
                raise ObjectNotFoundError(
                    f"repository.artifact_id must identify a committed artifact; got {artifact!r} does not exist"
                )
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

    @checked
    def release_legal_hold(self, hold: LegalHoldRecord, /) -> None:
        if hold.provider_id != self.provider_id:
            raise TypeError(
                "hold.provider_id must match the repository provider; got a foreign legal-hold provider"
            )
        key = self._hold_key(hold.artifact_id, hold.hold_id)
        value = self.client.read_object(key, maximum_bytes=self.maximum_metadata_bytes)
        persisted = LegalHoldRecord.from_record(_json_record(value.data))
        if persisted.record_id != hold.record_id:
            raise RepositoryConflictError(
                "hold.record_id must match the stored legal hold before release; got release record mismatch"
            )
        self.client.delete_object(key, expected_etag=value.metadata.etag)

    @checked
    def set_retention(self, artifact_id: str, policy: RetentionPolicy, /) -> None:
        artifact = identifier(artifact_id, "artifact_id")
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
        artifact = identifier(artifact_id, "artifact_id")
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
        finally:
            self._release_artifact_guard(guard)
        emit(
            "WARNING",
            "lifecycle.repository.tombstone.committed",
            "Repository tombstone committed",
            artifact_id=record.artifact_id,
            provider_id=self.provider_id,
        )
        return record

    @checked
    def collect_garbage(
        self,
        /,
        *,
        now: int | None = None,
        default_policy: RetentionPolicy | None = None,
    ) -> GarbageCollectionReport:
        collected = _now() if now is None else nonnegative_integer(now, "now")
        default = RetentionPolicy() if default_policy is None else default_policy
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
                            "artifact.pointer must remain present during current-root garbage collection; got current pointer disappeared during GC"
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
        report = GarbageCollectionReport(
            self.provider_id,
            collected,
            removed_attempts,
            removed_artifacts,
            expired_leases,
        )
        emit(
            "INFO",
            "lifecycle.repository.garbage_collection.completed",
            "Repository garbage collection completed",
            expired_lease_count=len(report.expired_lease_ids),
            provider_id=self.provider_id,
            removed_artifact_count=len(report.removed_artifact_ids),
            removed_attempt_count=len(report.removed_attempt_ids),
        )
        return report

    def _validate_transaction(self, transaction: RepositoryTransaction, /) -> None:
        if transaction.provider_id != self.provider_id:
            raise RepositoryConflictError(
                "transaction.provider_id must match the repository provider; got a foreign transaction provider"
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
                "transaction.transaction_id must match its stored attempt; got transaction record mismatch"
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
                    "chunk.object_key must belong to the attempt-private root; got a foreign manifest chunk key"
                )
            encoded = self.client.read_object(
                chunk.object_key,
                maximum_bytes=min(self.maximum_chunk_bytes, chunk.encoded_size),
            ).data
            _decode_and_validate_chunk(chunk, encoded, self.maximum_chunk_bytes)

    def _validate_manifest_argument(
        self, manifest: ArtifactManifest, chunk: ChunkRecord, /
    ) -> None:
        if manifest.provider_id != self.provider_id:
            raise RepositoryConflictError(
                "manifest.provider_id must match the repository provider; got a foreign manifest provider"
            )
        if chunk.chunk_id not in {item.chunk_id for item in manifest.chunks}:
            raise RepositoryConflictError(
                "manifest.chunks must include the requested chunk ID; got a chunk not present in the manifest"
            )

    def _create_or_verify_object(self, key: str, payload: bytes, /) -> None:
        existing = self._read_object_optional(key, len(payload))
        if existing is None:
            self.client.create_object(key, payload)
        elif existing.data != payload:
            raise RepositoryConflictError(
                f"repository.immutable_object.payload must match existing immutable bytes; got conflicting object {key!r}"
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

    def artifact_guard_metadata(self, artifact_id: str, /) -> ObjectMetadata:
        """Return the conditional token for an abandoned metadata guard."""

        artifact = identifier(artifact_id, "artifact_id")
        value = self.client.read_object(
            self._guard_key(artifact),
            maximum_bytes=self.maximum_metadata_bytes,
        )
        record = _json_record(value.data)
        if (
            record.get("kind") != "artifact-metadata-guard"
            or record.get("provider_id") != self.provider_id
            or record.get("artifact_id") != artifact
            or not isinstance(record.get("token"), str)
            or type(record.get("acquired_at")) is not int
        ):
            raise RepositoryCorruptionError(
                "artifact.metadata_guard.fields must identify the artifact, provider, token, and acquisition time; got invalid metadata guard"
            )
        return value.metadata

    @checked
    def recover_artifact_guard(
        self, authorization: ArtifactGuardRecoveryAuthorization, /
    ) -> None:
        """Release an abandoned guard only after an external worker fence."""

        if authorization.provider_id != self.provider_id:
            raise RepositoryConflictError(
                "guard_authorization.provider_id must match the repository provider; got an authorization for another provider"
            )
        metadata = self.artifact_guard_metadata(authorization.artifact_id)
        if metadata.etag != authorization.guard_etag:
            raise RepositoryConflictError(
                "guard_authorization.guard_etag must match the current guard after fencing; got guard changed after external fencing evidence"
            )
        self.client.delete_object(metadata.key, expected_etag=authorization.guard_etag)

    def _acquire_artifact_guard(self, artifact_id: str, /) -> ObjectMetadata:
        artifact = identifier(artifact_id, "artifact_id")
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
                raise RepositoryCorruptionError(
                    "artifact.metadata_guard.fields must identify the artifact, provider, token, and acquisition time; got invalid metadata guard"
                )
            raise RepositoryConflictError(
                f"artifact.metadata_guard must be absent before modification; got artifact {artifact!r} metadata is being modified"
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
                    "repository.artifact_namespace.keys must name pointer objects; got an invalid artifact namespace object"
                )
            artifact = identifier(
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
            attempt_ = identifier(attempt, "attempt_id")
            transaction = RepositoryTransaction.from_record(
                _json_record(
                    self.client.read_object(
                        self._transaction_key(attempt_),
                        maximum_bytes=self.maximum_metadata_bytes,
                    ).data
                )
            )
            if transaction.provider_id != self.provider_id:
                raise RepositoryCorruptionError(
                    "transaction.provider_id must match the repository provider; got attempt provider mismatch"
                )
            marker = self._read_object_optional(
                self._commit_key(attempt_), self.maximum_metadata_bytes
            )
            manifest_value = self._read_object_optional(
                self._manifest_key(attempt_), self.maximum_metadata_bytes
            )
            manifest: ArtifactManifest | None = None
            if marker is not None:
                if manifest_value is None:
                    raise RepositoryCorruptionError(
                        "artifact.commit_marker.manifest must be present for a committed root; got commit marker with no manifest"
                    )
                manifest = ArtifactManifest.from_record(_json_record(manifest_value.data))
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
                        "artifact.manifest.fields must match the root transaction and chunk ownership; got committed root manifest mismatch"
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
                raise RepositoryCorruptionError(
                    "lease.provider_id must match the repository provider; got lease provider mismatch"
                )
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
                raise RepositoryCorruptionError(
                    "hold.provider_id must match the repository provider; got legal-hold provider mismatch"
                )
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
        policy = record.get("policy")
        if (
            record.get("kind") != "artifact-retention"
            or record.get("provider_id") != self.provider_id
            or record.get("artifact_id") != artifact
            or not isinstance(policy, Mapping)
        ):
            raise RepositoryCorruptionError(
                "artifact.retention.fields must match the artifact and provider with a mapping policy; got invalid retention record"
            )
        return RetentionPolicy.from_record(policy)

    def _tombstone_optional(self, artifact: str, /) -> TombstoneRecord | None:
        value = self._read_object_optional(
            self._tombstone_key(artifact), self.maximum_metadata_bytes
        )
        if value is None:
            return None
        record = TombstoneRecord.from_record(_json_record(value.data))
        if record.provider_id != self.provider_id or record.artifact_id != artifact:
            raise RepositoryCorruptionError(
                "artifact.tombstone.fields must match the artifact and provider; got tombstone provider or artifact mismatch"
            )
        return record

    def _key(self, suffix: str, /) -> str:
        return f"{self.namespace}/{suffix}"

    def _root_prefix(self, attempt: str, /) -> str:
        return self._key(f"roots/{identifier(attempt, 'attempt_id')}/")

    def _transaction_key(self, attempt: str, /) -> str:
        return self._root_prefix(attempt) + "transaction.json"

    def _manifest_key(self, attempt: str, /) -> str:
        return self._root_prefix(attempt) + "manifest.json"

    def _commit_key(self, attempt: str, /) -> str:
        return self._root_prefix(attempt) + "COMMIT"

    def _chunk_key(self, attempt: str, logical: str, index: int, /) -> str:
        return self._root_prefix(attempt) + f"chunks/{logical}.{index:08d}.bin"

    def _pointer_key(self, artifact: str, /) -> str:
        return self._key(f"artifacts/{identifier(artifact, 'artifact_id')}.pointer")

    def _lease_key(self, artifact: str, lease: str, /) -> str:
        return self._key(
            f"leases/{identifier(artifact, 'artifact_id')}/{identifier(lease, 'lease_id')}.json"
        )

    def _hold_key(self, artifact: str, hold: str, /) -> str:
        return self._key(
            f"holds/{identifier(artifact, 'artifact_id')}/{identifier(hold, 'hold_id')}.json"
        )

    def _retention_key(self, artifact: str, /) -> str:
        return self._key(f"retention/{identifier(artifact, 'artifact_id')}.json")

    def _tombstone_key(self, artifact: str, /) -> str:
        return self._key(f"tombstones/{identifier(artifact, 'artifact_id')}.json")

    def _guard_key(self, artifact: str, /) -> str:
        return self._key(f"guards/{identifier(artifact, 'artifact_id')}.json")

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
                "artifact.pointer.root must identify the committed attempt; got a missing or mismatched immutable root"
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
        raise RepositoryCorruptionError(
            "artifact.pointer.identity must match its record kind, artifact, and provider; got invalid pointer identity"
        )
    identifier(str(record.get("attempt_id")), "attempt_id")
    for name in ("transaction_id", "manifest_id", "manifest_sha256"):
        value = record.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RepositoryCorruptionError(
                f"artifact.pointer.{name} must be a lowercase SHA-256 digest; got {value!r}"
            )


def _validate_commit_marker(
    marker: Mapping[str, object], pointer: Mapping[str, object], /
) -> None:
    expected = dict(pointer)
    expected["kind"] = "artifact-commit"
    if dict(marker) != expected:
        raise RepositoryCorruptionError(
            "artifact.commit_marker.fields must match the artifact pointer; got commit marker and pointer disagreement"
        )


def _encode_chunk(payload: bytes, encoding: ChunkEncoding, /) -> bytes:
    selected = parse(encoding, ChunkEncoding, "chunk.encoding")
    match selected:
        case "identity":
            return payload
        case "zlib":
            return zlib.compress(payload, level=9)
        case _ as invalid:
            assert_never(invalid)


def _decode_and_validate_chunk(
    chunk: ChunkRecord, encoded: bytes, maximum_plaintext_bytes: int, /
) -> bytes:
    if len(encoded) != chunk.encoded_size:
        raise RepositoryCorruptionError(
            "chunk.encoded_size must match the stored bytes; got encoded chunk size mismatch"
        )
    if hashlib.sha256(encoded).hexdigest() != chunk.encoded_sha256:
        raise RepositoryCorruptionError(
            "chunk.encoded_sha256 must match the stored bytes; got encoded chunk checksum mismatch"
        )
    match chunk.encoding:
        case "identity":
            plaintext = encoded
        case "zlib":
            decompressor = zlib.decompressobj()
            plaintext = decompressor.decompress(encoded, maximum_plaintext_bytes + 1)
            if (
                len(plaintext) > maximum_plaintext_bytes
                or decompressor.unconsumed_tail
                or not decompressor.eof
                or decompressor.unused_data
            ):
                raise RepositoryCorruptionError(
                    "chunk.encoded_payload must contain bounded zlib data without trailing bytes; "
                    "got incomplete, trailing, or over-limit data"
                )
        case _ as invalid:
            assert_never(invalid)
    if len(plaintext) != chunk.plaintext_size:
        raise RepositoryCorruptionError(
            "chunk.plaintext_size must match decoded bytes; got plaintext chunk size mismatch"
        )
    if hashlib.sha256(plaintext).hexdigest() != chunk.plaintext_sha256:
        raise RepositoryCorruptionError(
            "chunk.plaintext_sha256 must match decoded bytes; got plaintext chunk checksum mismatch"
        )
    return plaintext


def _json_bytes(record: Mapping[str, object], maximum: int, /) -> bytes:
    try:
        payload = (
            json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError(
            "repository.metadata.values must be finite JSON values; got a non-JSON or non-finite value"
        ) from error
    if len(payload) > maximum:
        raise ValueError(
            "repository.metadata.nbytes must fit its byte bound; got metadata byte bound exceeded"
        )
    return payload


def _json_record(payload: bytes, /) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(
            f"repository.metadata.constants must be finite JSON values; got non-finite constant {value!r}"
        )

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
        raise RepositoryCorruptionError(
            "repository.metadata.json must decode as JSON without duplicate keys or non-finite constants; got invalid JSON"
        ) from error
    if not isinstance(value, dict):
        raise RepositoryCorruptionError(
            "repository.metadata.root must be a JSON object; got a non-object JSON value"
        )
    return value


def _unique_json_object(
    pairs: list[tuple[str, object]],
    /,
) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(
                f"repository.metadata.members must be unique; got duplicate member {name!r}"
            )
        value[name] = item
    return value


def _admit_repository_root(root: str | os.PathLike[str], /) -> Path:
    candidate = Path(root).absolute()
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current = current / component
        try:
            information = current.lstat()
        except FileNotFoundError:
            current.mkdir(mode=0o700)
            information = current.lstat()
        if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(information.st_mode):
            raise ValueError(
                "repository.root.components must be real directories; got a symlink or non-directory component"
            )
    return candidate


def _read_json_file(path: Path, maximum: int, /) -> dict[str, object]:
    return _json_record(_read_bounded_file(path, maximum))


def _read_bounded_file(path: Path, maximum: int, /) -> bytes:
    maximum_ = nonnegative_integer(maximum, "maximum read bytes")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            information = os.fstat(descriptor)
            if not stat.S_ISREG(information.st_mode) or information.st_size > maximum_:
                raise RepositoryCorruptionError(
                    f"repository.file.size must fit its read bound on a regular file; got {path.name!r} exceeds its read bound"
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
            f"repository.file.path must identify a readable bounded file; got cannot read {path.name!r}"
        ) from error
    if len(payload) > maximum_ or len(payload) != information.st_size:
        raise RepositoryCorruptionError(
            f"repository.file.size must remain unchanged and fit its read bound; got {path.name!r} changed or exceeded its read bound"
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
        raise TypeError(
            f"repository.payload must be bytes-like; got {type(value).__name__} for {name!r}"
        )
    view = memoryview(value)
    if view.nbytes > maximum:
        raise ValueError(
            f"repository.payload.nbytes must fit the declared byte bound; got {name!r} exceeds its byte bound"
        )
    return bytes(view)


def _now() -> int:
    return time.time_ns() // 1_000_000_000


def _namespace(value: str, /) -> str:
    normalized = validate_object_key(value.rstrip("/"))
    return normalized


def _object_prefix(value: str, /) -> str:
    normalized = value.rstrip("/")
    validate_object_key(normalized)
    return normalized + "/"


__all__ = [
    "ArtifactGuardRecoveryAuthorization",
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
