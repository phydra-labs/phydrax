#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Addressable-shard checkpoint publication and topology-neutral restore."""

from __future__ import annotations

import hashlib
import io
import json
import operator
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec, Sharding, SingleDeviceSharding

from .._array_archive import (
    _read_npy_metadata,
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
)
from .._fingerprint import canonical_fingerprint, canonical_json
from .._strict import Strict
from ._chunk_repository import (
    _MAX_METADATA_VALUE_BYTES,
    ArtifactManifest,
    ArtifactRepository,
    ChunkEncoding,
    ChunkRecord,
    RepositoryCorruptionError,
    RepositoryTransaction,
)
from ._models import CheckpointManifest, CheckpointShard
from ._restart_topology import (
    admit_topology_restart,
    MeshingArrayRole,
    MeshingCheckpointState,
    RestartAdmission,
    TopologyRestartPolicy,
    TopologyRestartRelation,
)


if TYPE_CHECKING:
    from .._model._structure import ModelRecipeArray
    from ._meshing_sources import MeshingSourceValidation


CanonicalIndex = tuple[tuple[int, int, int], ...]
_DISTRIBUTED_CHECKPOINT_LIMITS = DEFAULT_ARRAY_ARCHIVE_LIMITS
_NO_PARENT_ID = "none"
# One process artifact holds two logical streams: packed shard payloads and the
# canonical JSON shard descriptor table that metadata references by digest.
_SHARD_PAYLOADS = "shard-payloads"
_SHARD_DESCRIPTORS = "shard-descriptors"
# Shards no larger than this share packed chunks; it also bounds the plaintext
# decoded to read one small shard.
_PACKED_CHUNK_BYTES = 1 << 16
_PROCESS_OWNER_KEYS = (
    "checkpoint_id",
    "analysis_plan_id",
    "numeric_revision_id",
    "execution_plan_id",
    "process_index",
    "topology_epoch",
    "repository_id",
    "parent_checkpoint_id",
    "parent_manifest_id",
)
_PROCESS_ARTIFACT_KEYS = frozenset(
    {
        *_PROCESS_OWNER_KEYS,
        "shard_count",
        "shard_descriptor_sha256",
        "shard_descriptor_bytes",
    }
)
_SHARD_DESCRIPTOR_KEYS = frozenset(
    {
        "array_path",
        "global_shape",
        "dtype",
        "index",
        "device_id",
        "replica_id",
        "payload_digest",
        "byte_count",
    }
)
_SHARD_METADATA_KEYS = frozenset(
    {
        *_PROCESS_OWNER_KEYS,
        "artifact_id",
        "artifact_manifest_id",
        "payload_offset",
        "array_path",
        "global_shape",
        "dtype",
        "index",
        "device_id",
        "replica_id",
    }
)
_PROCESS_SHARD_TABLE_CAPACITY = 64


def _canonical_index(index: Any, shape: tuple[int, ...]) -> CanonicalIndex:
    entries = index if isinstance(index, tuple) else (index,)
    if len(entries) != len(shape):
        raise ValueError("shard index rank does not match global array rank")
    normalized: list[tuple[int, int, int]] = []
    for entry, size in zip(entries, shape, strict=True):
        if isinstance(entry, int):
            value = entry if entry >= 0 else size + entry
            if not 0 <= value < size:
                raise ValueError("integer shard index is out of bounds")
            normalized.append((value, value + 1, 1))
        elif isinstance(entry, slice):
            start, stop, step = entry.indices(size)
            if step != 1:
                raise ValueError("checkpoint shards require unit-stride indices")
            normalized.append((start, stop, step))
        else:
            raise TypeError("checkpoint shard indices must contain integers or slices")
    return tuple(normalized)


def _unique_json_object(pairs: list[tuple[str, Any]], /) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"duplicate checkpoint metadata member {name!r}")
        value[name] = item
    return value


def _metadata_json(
    value: str,
    /,
    *,
    maximum_nesting: int = 16,
    maximum_bytes: int = _MAX_METADATA_VALUE_BYTES,
) -> Any:
    if not isinstance(value, str) or len(value.encode("utf-8")) > maximum_bytes:
        raise RepositoryCorruptionError("Checkpoint metadata exceeds its byte limit.")
    depth = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > maximum_nesting:
                raise RepositoryCorruptionError(
                    "Checkpoint metadata exceeds its nesting limit."
                )
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise RepositoryCorruptionError("Checkpoint metadata JSON is invalid.")
    if depth or in_string:
        raise RepositoryCorruptionError("Checkpoint metadata JSON is invalid.")
    try:
        return json.loads(value, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise RepositoryCorruptionError("Checkpoint metadata JSON is invalid.") from error


def _admit_array_spec(
    shape_value: Any,
    dtype_value: Any,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[tuple[int, ...], np.dtype[Any], int]:
    if not isinstance(shape_value, list) or any(
        type(extent) is not int or extent < 0 for extent in shape_value
    ):
        raise RepositoryCorruptionError("Checkpoint global shape is invalid.")
    shape = tuple(shape_value)
    if len(shape) > limits.max_array_rank:
        raise RepositoryCorruptionError("Checkpoint array exceeds the rank limit.")
    try:
        dtype = np.dtype(dtype_value)
    except (TypeError, ValueError) as error:
        raise RepositoryCorruptionError("Checkpoint array dtype is invalid.") from error
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.metadata is not None
        or dtype.kind not in limits.allowed_dtype_kinds
        or dtype.itemsize > limits.max_dtype_itemsize
    ):
        raise RepositoryCorruptionError(
            "Checkpoint array dtype is not admitted by policy."
        )
    elements = 1
    for extent in shape:
        if extent > limits.max_axis_length:
            raise RepositoryCorruptionError(
                "Checkpoint array exceeds the axis-length limit."
            )
        if extent and elements > limits.max_total_array_elements // extent:
            raise RepositoryCorruptionError(
                "Checkpoint array exceeds the aggregate element limit."
            )
        elements *= extent
    if elements > limits.max_total_array_elements:
        raise RepositoryCorruptionError(
            "Checkpoint array exceeds the aggregate element limit."
        )
    byte_count = elements * dtype.itemsize
    if byte_count > limits.max_aggregate_bytes:
        raise RepositoryCorruptionError(
            "Checkpoint array exceeds the aggregate byte limit."
        )
    return shape, dtype, elements


def _parse_index(value: str, shape: tuple[int, ...], /) -> CanonicalIndex:
    raw = _metadata_json(value)
    if (
        not isinstance(raw, list)
        or len(raw) != len(shape)
        or any(
            not isinstance(entry, list)
            or len(entry) != 3
            or any(type(component) is not int for component in entry)
            for entry in raw
        )
    ):
        raise RepositoryCorruptionError("Checkpoint shard index is invalid.")
    index = tuple(tuple(entry) for entry in raw)
    if any(
        step != 1 or start < 0 or stop < start or stop > extent
        for (start, stop, step), extent in zip(index, shape, strict=True)
    ):
        raise RepositoryCorruptionError("Checkpoint shard index is out of bounds.")
    return index


def _index_size(index: CanonicalIndex, maximum: int, /) -> int:
    size = 1
    for start, stop, step in index:
        if step != 1:
            raise RepositoryCorruptionError(
                "Checkpoint shard indices require unit stride."
            )
        extent = stop - start
        if extent and size > maximum // extent:
            raise RepositoryCorruptionError("Checkpoint shard exceeds the element limit.")
        size *= extent
    return size


def _indices_overlap(left: CanonicalIndex, right: CanonicalIndex) -> bool:
    if len(left) != len(right):
        raise ValueError("checkpoint shard index ranks differ")
    return all(
        max(left_start, right_start) < min(left_stop, right_stop)
        for (left_start, left_stop, _), (right_start, right_stop, _) in zip(
            left, right, strict=True
        )
    )


def _array_payload(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    payload = np.ascontiguousarray(array) if array.ndim else array
    buffer = io.BytesIO()
    np.save(buffer, payload, allow_pickle=False)
    return buffer.getvalue()


def _layout_id(
    array_path: str,
    global_shape: Sequence[int],
    dtype: str,
    index: Sequence[Sequence[int]],
    process_index: int,
    device_id: int,
    replica_id: int,
    topology_epoch: int,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "array_path": array_path,
            "global_shape": list(global_shape),
            "dtype": dtype,
            "index": [list(entry) for entry in index],
            "process_index": process_index,
            "device_id": device_id,
            "replica_id": replica_id,
            "topology_epoch": topology_epoch,
        }
    )


def _shard_id(
    checkpoint_id: str,
    execution_plan_id: str,
    layout_id: str,
    payload_digest: str,
    parent_manifest_id: str,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "checkpoint_id": checkpoint_id,
            "execution_plan_id": execution_plan_id,
            "layout_id": layout_id,
            "payload_digest": payload_digest,
            "parent_manifest_id": parent_manifest_id,
        }
    )


@dataclass(frozen=True, slots=True)
class AddressableCheckpointShard:
    """Host snapshot of one canonical, non-redundant JAX array shard."""

    array_path: str
    global_shape: tuple[int, ...]
    dtype: str
    index: CanonicalIndex
    process_index: int
    device_id: int
    replica_id: int
    payload: np.ndarray
    layout_id: str

    @property
    def payload_bytes(self) -> bytes:
        return _array_payload(self.payload)


@dataclass(frozen=True, slots=True)
class ProcessCheckpointPublication:
    """One process's immutable artifact and checkpoint-shard acknowledgements."""

    process_index: int
    artifact_manifest: ArtifactManifest | None
    shards: tuple[CheckpointShard, ...]
    meshing_state: MeshingCheckpointState | None = None


def snapshot_addressable_arrays(
    tree: Any,
    /,
    *,
    topology_epoch: int = 0,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    host_array_owner: int = 0,
) -> tuple[AddressableCheckpointShard, ...]:
    """Snapshot addressable JAX shards and one replica of authored host arrays."""

    if type(topology_epoch) is not int or topology_epoch < 0:
        raise ValueError("topology_epoch must be a non-negative integer")
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance")
    if (
        type(host_array_owner) is not int
        or not 0 <= host_array_owner < jax.process_count()
    ):
        raise ValueError("host_array_owner must identify one current process.")
    pending: list[tuple[str, jax.Array, Any, CanonicalIndex]] = []
    snapshots: list[AddressableCheckpointShard] = []
    total_bytes = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not isinstance(leaf, (jax.Array, np.ndarray)):
            continue
        array_path = jax.tree_util.keystr(path) or "<root>"
        shape = tuple(leaf.shape)
        try:
            _, admitted_dtype, elements = _admit_array_spec(
                list(shape), np.dtype(leaf.dtype).str, limits
            )
        except RepositoryCorruptionError as error:
            raise ValueError(str(error)) from error
        byte_count = elements * admitted_dtype.itemsize
        if total_bytes > limits.max_aggregate_bytes - byte_count:
            raise ValueError("Checkpoint arrays exceed the aggregate byte limit.")
        total_bytes += byte_count
        if isinstance(leaf, np.ndarray):
            # Native authored coefficients already belong to the host. Publish
            # one canonical replica without staging the complete value on a device.
            if jax.process_index() != host_array_owner:
                continue
            if len(pending) + len(snapshots) >= limits.max_members:
                raise ValueError("Checkpoint exceeds the shard-count limit.")
            payload = (
                np.array(leaf, copy=True, order="C") if leaf.flags.writeable else leaf
            )
            payload.setflags(write=False)
            index = tuple((0, size, 1) for size in shape)
            device = next(
                device
                for device in jax.devices()
                if device.process_index == host_array_owner
            )
            layout_id = _layout_id(
                array_path,
                shape,
                np.dtype(leaf.dtype).str,
                index,
                int(device.process_index),
                int(device.id),
                0,
                topology_epoch,
            )
            snapshots.append(
                AddressableCheckpointShard(
                    array_path=array_path,
                    global_shape=shape,
                    dtype=np.dtype(leaf.dtype).str,
                    index=index,
                    process_index=int(device.process_index),
                    device_id=int(device.id),
                    replica_id=0,
                    payload=payload,
                    layout_id=layout_id,
                )
            )
            continue
        ownership: dict[CanonicalIndex, tuple[int, int]] = {}
        for device, index in leaf.sharding.devices_indices_map(shape).items():
            normalized = _canonical_index(index, shape)
            key = (int(device.process_index), int(device.id))
            previous = ownership.get(normalized)
            if previous is None or key < previous:
                ownership[normalized] = key
        for shard in leaf.addressable_shards:
            normalized = _canonical_index(shard.index, shape)
            device_key = (int(shard.device.process_index), int(shard.device.id))
            if ownership[normalized] != device_key:
                continue
            if len(pending) + len(snapshots) >= limits.max_members:
                raise ValueError("Checkpoint exceeds the shard-count limit.")
            shard.data.copy_to_host_async()
            pending.append((array_path, leaf, shard, normalized))

    for array_path, leaf, shard, index in pending:
        payload = np.asarray(shard.data)
        local_shape = tuple(stop - start for start, stop, _ in index)
        if (
            payload.shape != local_shape
            or payload.dtype != np.dtype(leaf.dtype)
            or payload.nbytes > limits.max_member_bytes
        ):
            raise ValueError("Addressable checkpoint shard payload is inconsistent.")
        layout_id = _layout_id(
            array_path,
            leaf.shape,
            np.dtype(leaf.dtype).str,
            index,
            int(shard.device.process_index),
            int(shard.device.id),
            int(shard.replica_id),
            topology_epoch,
        )
        snapshots.append(
            AddressableCheckpointShard(
                array_path=array_path,
                global_shape=tuple(leaf.shape),
                dtype=np.dtype(leaf.dtype).str,
                index=index,
                process_index=int(shard.device.process_index),
                device_id=int(shard.device.id),
                replica_id=int(shard.replica_id),
                payload=payload,
                layout_id=layout_id,
            )
        )
    return tuple(snapshots)


def _write_packed_stream(
    repository: ArtifactRepository,
    transaction: RepositoryTransaction,
    logical_name: str,
    payloads: Iterable[bytes],
    /,
    *,
    encoding: ChunkEncoding,
) -> list[ChunkRecord]:
    """Write payloads contiguously; small payloads never straddle a chunk."""

    packed_limit = min(_PACKED_CHUNK_BYTES, repository.maximum_chunk_bytes)
    records: list[ChunkRecord] = []
    pending: list[bytes] = []
    pending_bytes = 0
    offset = 0

    def write(data: bytes | memoryview) -> None:
        nonlocal offset
        records.append(
            repository.write_chunk(
                transaction, logical_name, len(records), offset, data, encoding=encoding
            )
        )
        offset += len(data)

    for payload in payloads:
        if pending and pending_bytes + len(payload) > packed_limit:
            write(b"".join(pending))
            pending.clear()
            pending_bytes = 0
        if len(payload) <= packed_limit:
            pending.append(payload)
            pending_bytes += len(payload)
            continue
        view = memoryview(payload)
        for start in range(0, len(payload), repository.maximum_chunk_bytes):
            write(view[start : start + repository.maximum_chunk_bytes])
    if pending:
        write(b"".join(pending))
    return records


def _read_artifact_range(
    repository: ArtifactRepository,
    artifact: ArtifactManifest,
    logical_name: str,
    offset: int,
    byte_count: int,
    /,
) -> bytes:
    """Read one exact plaintext range of a logical artifact stream."""

    stop = offset + byte_count
    parts = []
    for chunk in artifact.chunks:
        if (
            chunk.logical_name != logical_name
            or chunk.offset + chunk.plaintext_size <= offset
            or chunk.offset >= stop
        ):
            continue
        data = repository.read_chunk(
            artifact, chunk, maximum_plaintext_bytes=chunk.plaintext_size
        )
        if len(data) != chunk.plaintext_size:
            raise RepositoryCorruptionError("Checkpoint artifact chunk size mismatch.")
        parts.append(
            memoryview(data)[max(offset - chunk.offset, 0) : stop - chunk.offset]
        )
    payload = b"".join(parts)
    if len(payload) != byte_count:
        raise RepositoryCorruptionError(
            "Checkpoint artifact range is not covered by its chunks."
        )
    return payload


@dataclass(frozen=True, slots=True)
class _ProcessShardTable:
    """Admitted shard records of one immutable committed process artifact."""

    shards: tuple[CheckpointShard, ...]
    fingerprints: frozenset[str]


# Keyed by content-addressed manifest identity, so a hit names identical bytes.
_PROCESS_SHARD_TABLES: dict[tuple[str, str, ArrayArchiveLimits], _ProcessShardTable] = {}


def _process_artifact_metadata(
    repository: ArtifactRepository, artifact: ArtifactManifest, /
) -> dict[str, str]:
    metadata = dict(artifact.metadata)
    if (
        set(metadata) != _PROCESS_ARTIFACT_KEYS
        or artifact.provider_id != repository.provider_id
        or metadata["repository_id"] != repository.provider_id
        or artifact.artifact_id
        != f"{metadata['checkpoint_id']}.process-{metadata['process_index']}"
    ):
        raise RepositoryCorruptionError(
            "Process checkpoint artifact metadata is invalid."
        )
    return metadata


def _artifact_shards(
    artifact: ArtifactManifest,
    descriptors: Any,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[tuple[CheckpointShard, ...], int]:
    """Expand one process's compact descriptor table into admitted shard records.

    Process-level ownership lives once in artifact metadata. Layout and shard
    IDs are recomputed from content, and payload offsets follow table order in
    the packed payload stream.
    """

    metadata = dict(artifact.metadata)
    owner = {name: metadata[name] for name in _PROCESS_OWNER_KEYS}
    process_index = _metadata_integer(metadata, "process_index")
    topology_epoch = _metadata_integer(metadata, "topology_epoch")
    if (
        not isinstance(descriptors, list)
        or len(descriptors) != _metadata_integer(metadata, "shard_count")
        or len(descriptors) > limits.max_members
    ):
        raise RepositoryCorruptionError(
            "Process checkpoint shard descriptors are invalid."
        )
    shards = []
    offset = 0
    for descriptor in descriptors:
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != _SHARD_DESCRIPTOR_KEYS
            or any(
                type(descriptor[name]) is not int or descriptor[name] < 0
                for name in ("device_id", "replica_id", "byte_count")
            )
            or any(
                not isinstance(descriptor[name], str)
                for name in ("array_path", "dtype", "payload_digest")
            )
        ):
            raise RepositoryCorruptionError(
                "Process checkpoint shard descriptors are invalid."
            )
        try:
            layout_id = _layout_id(
                descriptor["array_path"],
                descriptor["global_shape"],
                descriptor["dtype"],
                descriptor["index"],
                process_index,
                descriptor["device_id"],
                descriptor["replica_id"],
                topology_epoch,
            )
            shard = CheckpointShard(
                _shard_id(
                    owner["checkpoint_id"],
                    owner["execution_plan_id"],
                    layout_id,
                    descriptor["payload_digest"],
                    owner["parent_manifest_id"],
                ),
                descriptor["payload_digest"],
                descriptor["byte_count"],
                (layout_id,),
                metadata={
                    **owner,
                    "artifact_id": artifact.artifact_id,
                    "artifact_manifest_id": artifact.manifest_id,
                    "payload_offset": str(offset),
                    "array_path": descriptor["array_path"],
                    "global_shape": canonical_json(descriptor["global_shape"]),
                    "dtype": descriptor["dtype"],
                    "index": canonical_json(descriptor["index"]),
                    "device_id": str(descriptor["device_id"]),
                    "replica_id": str(descriptor["replica_id"]),
                },
            )
        except (TypeError, ValueError) as error:
            raise RepositoryCorruptionError(
                "Process checkpoint shard descriptors are invalid."
            ) from error
        _validated_shard(shard, limits)
        if offset > limits.max_aggregate_bytes - shard.byte_count:
            raise RepositoryCorruptionError(
                "Checkpoint exceeds the aggregate payload-byte limit."
            )
        offset += shard.byte_count
        shards.append(shard)
    return tuple(shards), offset


def _process_shard_table(
    repository: ArtifactRepository,
    artifact: ArtifactManifest,
    limits: ArrayArchiveLimits,
    /,
) -> _ProcessShardTable:
    """Read the digest-bound descriptor table of one committed process artifact."""

    key = (artifact.provider_id, artifact.manifest_id, limits)
    table = _PROCESS_SHARD_TABLES.get(key)
    if table is not None:
        return table
    metadata = _process_artifact_metadata(repository, artifact)
    descriptor_bytes = _metadata_integer(metadata, "shard_descriptor_bytes")
    descriptor_chunks = tuple(
        chunk for chunk in artifact.chunks if chunk.logical_name == _SHARD_DESCRIPTORS
    )
    if (
        descriptor_bytes > limits.max_manifest_bytes
        or sum(chunk.plaintext_size for chunk in descriptor_chunks) != descriptor_bytes
    ):
        raise RepositoryCorruptionError(
            "Process checkpoint shard descriptor table exceeds its declared bounds."
        )
    payload = _read_artifact_range(
        repository, artifact, _SHARD_DESCRIPTORS, 0, descriptor_bytes
    )
    if hashlib.sha256(payload).hexdigest() != metadata["shard_descriptor_sha256"]:
        raise RepositoryCorruptionError(
            "Process checkpoint shard descriptor digest mismatch."
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise RepositoryCorruptionError(
            "Process checkpoint shard descriptors are invalid."
        ) from error
    descriptors = _metadata_json(text, maximum_bytes=limits.max_manifest_bytes)
    if canonical_json(descriptors) != text:
        raise RepositoryCorruptionError(
            "Process checkpoint shard descriptors are not canonical."
        )
    shards, payload_bytes = _artifact_shards(artifact, descriptors, limits)
    if (
        not {chunk.logical_name for chunk in artifact.chunks}
        <= {_SHARD_DESCRIPTORS, _SHARD_PAYLOADS}
        or sum(
            chunk.plaintext_size
            for chunk in artifact.chunks
            if chunk.logical_name == _SHARD_PAYLOADS
        )
        != payload_bytes
    ):
        raise RepositoryCorruptionError(
            "Process checkpoint artifact payload inventory is inconsistent."
        )
    table = _ProcessShardTable(
        shards, frozenset(shard.shard_fingerprint for shard in shards)
    )
    if len(_PROCESS_SHARD_TABLES) >= _PROCESS_SHARD_TABLE_CAPACITY:
        del _PROCESS_SHARD_TABLES[next(iter(_PROCESS_SHARD_TABLES))]
    _PROCESS_SHARD_TABLES[key] = table
    return table


def read_process_checkpoint_shards(
    repository: ArtifactRepository,
    artifact: ArtifactManifest,
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
) -> tuple[CheckpointShard, ...]:
    """Read and admit one committed process artifact's shard records."""

    return _process_shard_table(repository, artifact, limits).shards


def _parent_lineage(
    repository: ArtifactRepository,
    parent_manifest: CheckpointManifest | None,
    process_index: int,
    execution_plan_id: str,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[str, str]:
    if parent_manifest is None:
        return _NO_PARENT_ID, _NO_PARENT_ID
    if (
        not isinstance(parent_manifest, CheckpointManifest)
        or parent_manifest.complete is not True
    ):
        raise TypeError("parent_manifest must be a complete CheckpointManifest or None")
    if parent_manifest.execution_plan_id != execution_plan_id:
        raise ValueError("Parent checkpoint has the wrong execution ownership.")
    parent_artifact_id = f"{parent_manifest.checkpoint_id}.process-{process_index}"
    parent_artifact = repository.get_manifest(parent_artifact_id)
    parent_metadata = dict(parent_artifact.metadata)
    if (
        parent_artifact.provider_id != repository.provider_id
        or parent_metadata.get("repository_id") != repository.provider_id
        or parent_metadata.get("checkpoint_id") != parent_manifest.checkpoint_id
        or parent_metadata.get("analysis_plan_id") != parent_manifest.analysis_plan_id
        or parent_metadata.get("numeric_revision_id")
        != parent_manifest.numeric_revision_id
        or parent_metadata.get("execution_plan_id") != parent_manifest.execution_plan_id
        or parent_metadata.get("process_index") != str(process_index)
        or parent_metadata.get("parent_checkpoint_id")
        != (parent_manifest.parent_checkpoint_id or _NO_PARENT_ID)
        or parent_metadata.get("parent_manifest_id")
        != (parent_manifest.parent_manifest_id or _NO_PARENT_ID)
    ):
        raise RepositoryCorruptionError(
            "Parent checkpoint artifact ownership is inconsistent."
        )
    durable = _process_shard_table(repository, parent_artifact, limits)
    parent_process_fingerprints = [
        shard.shard_fingerprint
        for shard in parent_manifest.shards
        if _metadata(shard).get("process_index") == str(process_index)
    ]
    if (
        len(parent_process_fingerprints) != len(durable.shards)
        or set(parent_process_fingerprints) != durable.fingerprints
    ):
        raise RepositoryCorruptionError(
            "Parent checkpoint manifest does not match its repository artifact."
        )
    return parent_manifest.checkpoint_id, parent_manifest.manifest_id


def publish_process_checkpoint(
    repository: ArtifactRepository,
    checkpoint_id: str,
    execution_plan_id: str,
    tree: Any,
    /,
    *,
    analysis_plan_id: str,
    numeric_revision_id: str,
    writer_id: str,
    attempt_id: str | None = None,
    topology_epoch: int = 0,
    encoding: ChunkEncoding = "identity",
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
    host_array_owner: int = 0,
) -> ProcessCheckpointPublication:
    """Publish this process's canonical addressable shards transactionally.

    Shard payloads are packed into one bounded-chunk stream. Their descriptor
    table is a second content-addressed stream that artifact metadata binds by
    SHA-256 digest and byte size; ownership fields are recorded once.
    """

    process_index = jax.process_index()
    parent_checkpoint_id, parent_manifest_id = _parent_lineage(
        repository,
        parent_manifest,
        process_index,
        execution_plan_id,
        limits,
    )
    snapshots = snapshot_addressable_arrays(
        tree,
        topology_epoch=topology_epoch,
        limits=limits,
        host_array_owner=host_array_owner,
    )
    descriptors: list[dict[str, Any]] = []

    def payloads() -> Iterator[bytes]:
        for snapshot in snapshots:
            payload = snapshot.payload_bytes
            if len(payload) > limits.max_member_bytes:
                raise ValueError("Checkpoint shard payload exceeds its byte limit.")
            descriptors.append(
                {
                    "array_path": snapshot.array_path,
                    "global_shape": list(snapshot.global_shape),
                    "dtype": snapshot.dtype,
                    "index": [list(entry) for entry in snapshot.index],
                    "device_id": snapshot.device_id,
                    "replica_id": snapshot.replica_id,
                    "payload_digest": hashlib.sha256(payload).hexdigest(),
                    "byte_count": len(payload),
                }
            )
            yield payload

    artifact_id = f"{checkpoint_id}.process-{process_index}"
    transaction = repository.begin(
        artifact_id,
        writer_id,
        attempt_id=attempt_id,
    )
    chunk_records = _write_packed_stream(
        repository, transaction, _SHARD_PAYLOADS, payloads(), encoding=encoding
    )
    descriptor_payload = canonical_json(descriptors).encode("utf-8")
    if len(descriptor_payload) > limits.max_manifest_bytes:
        raise ValueError("Checkpoint shard descriptors exceed their byte limit.")
    chunk_records += _write_packed_stream(
        repository,
        transaction,
        _SHARD_DESCRIPTORS,
        (descriptor_payload,),
        encoding=encoding,
    )
    manifest = repository.commit(
        transaction,
        chunk_records,
        metadata={
            "checkpoint_id": checkpoint_id,
            "analysis_plan_id": analysis_plan_id,
            "numeric_revision_id": numeric_revision_id,
            "execution_plan_id": execution_plan_id,
            "process_index": str(process_index),
            "topology_epoch": str(topology_epoch),
            "repository_id": repository.provider_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "parent_manifest_id": parent_manifest_id,
            "shard_count": str(len(descriptors)),
            "shard_descriptor_sha256": hashlib.sha256(descriptor_payload).hexdigest(),
            "shard_descriptor_bytes": str(len(descriptor_payload)),
        },
    )
    shards, _ = _artifact_shards(manifest, descriptors, limits)
    return ProcessCheckpointPublication(process_index, manifest, shards)


def _metadata(shard: CheckpointShard) -> dict[str, str]:
    return dict(shard.metadata)


def _metadata_integer(metadata: Mapping[str, str], name: str, /) -> int:
    value = metadata[name]
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise RepositoryCorruptionError(f"Checkpoint shard {name} metadata is invalid.")
    return int(value)


def _validated_shard(
    shard: CheckpointShard,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[str, tuple[int, ...], np.dtype[Any], CanonicalIndex]:
    metadata = _metadata(shard)
    if set(metadata) != _SHARD_METADATA_KEYS or any(
        not isinstance(metadata[name], str) or not metadata[name]
        for name in (
            "artifact_manifest_id",
            "artifact_id",
            "checkpoint_id",
            "analysis_plan_id",
            "numeric_revision_id",
            "execution_plan_id",
            "array_path",
            "repository_id",
        )
    ):
        raise RepositoryCorruptionError("Checkpoint shard metadata is invalid.")
    if bool(metadata["parent_checkpoint_id"]) != bool(metadata["parent_manifest_id"]):
        raise RepositoryCorruptionError("Checkpoint shard parent lineage is incomplete.")
    for name in (
        "process_index",
        "device_id",
        "replica_id",
        "topology_epoch",
        "payload_offset",
    ):
        _metadata_integer(metadata, name)
    shape_value = _metadata_json(metadata["global_shape"])
    shape, dtype, _ = _admit_array_spec(shape_value, metadata["dtype"], limits)
    index = _parse_index(metadata["index"], shape)
    local_elements = _index_size(index, limits.max_array_elements)
    local_bytes = local_elements * dtype.itemsize
    if (
        local_bytes > limits.max_member_bytes
        or shard.byte_count > limits.max_member_bytes
        or shard.byte_count < local_bytes
    ):
        raise RepositoryCorruptionError(
            "Checkpoint shard payload exceeds its byte limit."
        )
    return metadata["array_path"], shape, dtype, index


def _validate_exact_coverage(
    shards: Sequence[CheckpointShard],
    limits: ArrayArchiveLimits,
    /,
) -> None:
    if len(shards) > limits.max_members:
        raise RepositoryCorruptionError("Checkpoint exceeds the shard-count limit.")
    by_path: dict[
        str,
        list[tuple[CheckpointShard, tuple[int, ...], np.dtype[Any], CanonicalIndex]],
    ] = {}
    total_payload_bytes = 0
    for shard in shards:
        path, shape, dtype, index = _validated_shard(shard, limits)
        if total_payload_bytes > limits.max_aggregate_bytes - shard.byte_count:
            raise RepositoryCorruptionError(
                "Checkpoint exceeds the aggregate payload-byte limit."
            )
        total_payload_bytes += shard.byte_count
        by_path.setdefault(path, []).append((shard, shape, dtype, index))

    for path, path_shards in by_path.items():
        shapes = {item[1] for item in path_shards}
        dtypes = {item[2].str for item in path_shards}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise RepositoryCorruptionError(
                f"Checkpoint array {path!r} has inconsistent metadata."
            )
        shape = next(iter(shapes))
        _, _, expected = _admit_array_spec(list(shape), next(iter(dtypes)), limits)
        indices = [item[3] for item in path_shards]
        if len(set(indices)) != len(indices):
            raise RepositoryCorruptionError(
                f"Checkpoint array {path!r} contains duplicate shards."
            )
        for left_index, left in enumerate(indices):
            for right in indices[left_index + 1 :]:
                if _indices_overlap(left, right):
                    raise RepositoryCorruptionError(
                        f"Checkpoint array {path!r} contains overlapping shards."
                    )
        covered = sum(
            _index_size(index, limits.max_total_array_elements) for index in indices
        )
        if covered != expected:
            raise RepositoryCorruptionError(
                f"Checkpoint array {path!r} covers {covered} of {expected} values."
            )


def _validate_parent_ownership(
    parent_manifest: CheckpointManifest | None,
    checkpoint_id: str,
    analysis_plan_id: str,
    numeric_revision_id: str,
    execution_plan_id: str,
    /,
) -> tuple[str, str]:
    if parent_manifest is None:
        return _NO_PARENT_ID, _NO_PARENT_ID
    if (
        not isinstance(parent_manifest, CheckpointManifest)
        or parent_manifest.complete is not True
    ):
        raise TypeError("parent_manifest must be a complete CheckpointManifest or None")
    if parent_manifest.checkpoint_id == checkpoint_id:
        raise ValueError("A distributed checkpoint cannot parent itself.")
    if (
        parent_manifest.analysis_plan_id != analysis_plan_id
        or parent_manifest.numeric_revision_id != numeric_revision_id
        or parent_manifest.execution_plan_id != execution_plan_id
    ):
        raise ValueError("Parent checkpoint has incompatible lifecycle ownership.")
    return parent_manifest.checkpoint_id, parent_manifest.manifest_id


def assemble_distributed_checkpoint_manifest(
    checkpoint_id: str,
    analysis_plan_id: str,
    numeric_revision_id: str,
    execution_plan_id: str,
    publications: Sequence[ProcessCheckpointPublication],
    /,
    *,
    expected_process_count: int,
    parent_manifest: CheckpointManifest | None = None,
    diagnostic_ids: Sequence[str] = (),
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
) -> CheckpointManifest:
    """Validate all process acknowledgements and create the global commit record."""

    if type(expected_process_count) is not int or expected_process_count <= 0:
        raise ValueError("expected_process_count must be a positive integer")
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance")
    if expected_process_count > limits.max_members:
        raise ValueError("expected_process_count exceeds the process-count limit")
    parent_checkpoint_id, parent_manifest_id = _validate_parent_ownership(
        parent_manifest,
        checkpoint_id,
        analysis_plan_id,
        numeric_revision_id,
        execution_plan_id,
    )
    publications_ = tuple(publications)
    if parent_manifest is not None and any(
        publication.artifact_manifest is None for publication in publications_
    ):
        raise ValueError(
            "Parented distributed checkpoints require repository-backed publications."
        )
    process_indices = tuple(publication.process_index for publication in publications_)
    if tuple(sorted(process_indices)) != tuple(range(expected_process_count)):
        raise ValueError(
            "checkpoint publications do not cover every process exactly once"
        )
    shards = tuple(shard for publication in publications_ for shard in publication.shards)
    if not shards:
        raise ValueError("a complete distributed checkpoint requires array shards")
    _validate_exact_coverage(shards, limits)
    expected_lineage = (parent_checkpoint_id, parent_manifest_id)
    repository_ids = set()
    for publication in publications_:
        if publication.artifact_manifest is not None:
            artifact_metadata = dict(publication.artifact_manifest.metadata)
            repository_id = artifact_metadata.get("repository_id")
            if (
                publication.artifact_manifest.provider_id != repository_id
                or artifact_metadata.get("checkpoint_id") != checkpoint_id
                or artifact_metadata.get("analysis_plan_id") != analysis_plan_id
                or artifact_metadata.get("numeric_revision_id") != numeric_revision_id
                or artifact_metadata.get("execution_plan_id") != execution_plan_id
                or artifact_metadata.get("parent_checkpoint_id") != parent_checkpoint_id
                or artifact_metadata.get("parent_manifest_id") != parent_manifest_id
            ):
                raise RepositoryCorruptionError(
                    "Process checkpoint publication lineage is inconsistent."
                )
            repository_ids.add(repository_id)
        for shard in publication.shards:
            metadata = _metadata(shard)
            if (
                (
                    metadata["parent_checkpoint_id"],
                    metadata["parent_manifest_id"],
                )
                != expected_lineage
                or metadata["checkpoint_id"] != checkpoint_id
                or metadata["analysis_plan_id"] != analysis_plan_id
                or metadata["numeric_revision_id"] != numeric_revision_id
                or metadata["execution_plan_id"] != execution_plan_id
            ):
                raise RepositoryCorruptionError(
                    "Checkpoint shard ownership or parent lineage is inconsistent."
                )
            repository_ids.add(metadata["repository_id"])
    if len(repository_ids) != 1:
        raise RepositoryCorruptionError(
            "Distributed checkpoint publications span repository identities."
        )
    _validate_meshing_manifest_admission(publications_, shards, limits=limits)
    return CheckpointManifest(
        checkpoint_id,
        analysis_plan_id,
        numeric_revision_id,
        execution_plan_id,
        shards,
        complete=True,
        parent_manifest_id=(None if parent_manifest is None else parent_manifest_id),
        parent_checkpoint_id=(None if parent_manifest is None else parent_checkpoint_id),
        diagnostic_ids=diagnostic_ids,
    )


def assemble_distributed_checkpoint_from_repository(
    repository: ArtifactRepository,
    checkpoint_id: str,
    analysis_plan_id: str,
    numeric_revision_id: str,
    execution_plan_id: str,
    /,
    *,
    expected_process_count: int,
    parent_manifest: CheckpointManifest | None = None,
    diagnostic_ids: Sequence[str] = (),
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
) -> CheckpointManifest:
    """Assemble deterministic process acknowledgements from durable artifacts."""

    if type(expected_process_count) is not int or expected_process_count <= 0:
        raise ValueError("expected_process_count must be a positive integer")
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance")
    if expected_process_count > limits.max_members:
        raise ValueError("expected_process_count exceeds the process-count limit")
    parent_checkpoint_id, parent_manifest_id = _validate_parent_ownership(
        parent_manifest,
        checkpoint_id,
        analysis_plan_id,
        numeric_revision_id,
        execution_plan_id,
    )
    publications = []
    for process_index in range(expected_process_count):
        if parent_manifest is not None:
            _parent_lineage(
                repository,
                parent_manifest,
                process_index,
                execution_plan_id,
                limits,
            )
        artifact_id = f"{checkpoint_id}.process-{process_index}"
        artifact = repository.get_manifest(artifact_id)
        artifact_metadata = _process_artifact_metadata(repository, artifact)
        if artifact_metadata["checkpoint_id"] != checkpoint_id:
            raise RepositoryCorruptionError(
                "Process checkpoint artifact has the wrong checkpoint_id."
            )
        if artifact_metadata["analysis_plan_id"] != analysis_plan_id:
            raise RepositoryCorruptionError(
                "Process checkpoint artifact has the wrong analysis plan."
            )
        if artifact_metadata["numeric_revision_id"] != numeric_revision_id:
            raise RepositoryCorruptionError(
                "Process checkpoint artifact has the wrong numeric revision."
            )
        if artifact_metadata["execution_plan_id"] != execution_plan_id:
            raise RepositoryCorruptionError(
                "Process checkpoint artifact has the wrong execution plan."
            )
        if artifact_metadata["process_index"] != str(process_index):
            raise RepositoryCorruptionError(
                "Process checkpoint artifact has the wrong process index."
            )
        if (
            artifact_metadata["parent_checkpoint_id"] != parent_checkpoint_id
            or artifact_metadata["parent_manifest_id"] != parent_manifest_id
        ):
            raise RepositoryCorruptionError(
                "Process checkpoint artifact parent lineage is inconsistent."
            )
        shards = _process_shard_table(repository, artifact, limits).shards
        publications.append(
            ProcessCheckpointPublication(
                process_index,
                artifact,
                shards,
                _read_process_meshing_state(
                    repository, shards, process_index, limits=limits
                ),
            )
        )
    return assemble_distributed_checkpoint_manifest(
        checkpoint_id,
        analysis_plan_id,
        numeric_revision_id,
        execution_plan_id,
        publications,
        expected_process_count=expected_process_count,
        parent_manifest=parent_manifest,
        diagnostic_ids=diagnostic_ids,
        limits=limits,
    )


def _validate_shard_repository_binding(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    shard: CheckpointShard,
    limits: ArrayArchiveLimits,
    /,
) -> None:
    metadata = _metadata(shard)
    if (
        metadata["repository_id"] != repository.provider_id
        or metadata["checkpoint_id"] != manifest.checkpoint_id
        or metadata["analysis_plan_id"] != manifest.analysis_plan_id
        or metadata["numeric_revision_id"] != manifest.numeric_revision_id
        or metadata["execution_plan_id"] != manifest.execution_plan_id
        or metadata["parent_checkpoint_id"]
        != (manifest.parent_checkpoint_id or _NO_PARENT_ID)
        or metadata["parent_manifest_id"]
        != (manifest.parent_manifest_id or _NO_PARENT_ID)
    ):
        raise RepositoryCorruptionError(
            "Checkpoint shard lifecycle ownership is inconsistent."
        )
    artifact = repository.get_manifest(metadata["artifact_id"])
    artifact_metadata = dict(artifact.metadata)
    if (
        artifact.provider_id != repository.provider_id
        or metadata["artifact_manifest_id"] != artifact.manifest_id
        or artifact_metadata.get("repository_id") != repository.provider_id
        or artifact_metadata.get("checkpoint_id") != manifest.checkpoint_id
        or artifact_metadata.get("analysis_plan_id") != manifest.analysis_plan_id
        or artifact_metadata.get("numeric_revision_id") != manifest.numeric_revision_id
        or artifact_metadata.get("execution_plan_id") != manifest.execution_plan_id
        or artifact_metadata.get("process_index") != metadata["process_index"]
        or artifact_metadata.get("parent_checkpoint_id")
        != (manifest.parent_checkpoint_id or _NO_PARENT_ID)
        or artifact_metadata.get("parent_manifest_id")
        != (manifest.parent_manifest_id or _NO_PARENT_ID)
    ):
        raise RepositoryCorruptionError(
            "Checkpoint shard repository artifact ownership is inconsistent."
        )
    if (
        shard.shard_fingerprint
        not in _process_shard_table(repository, artifact, limits).fingerprints
    ):
        raise RepositoryCorruptionError("Checkpoint shard descriptor was substituted.")


def _read_shard_payload(
    repository: ArtifactRepository,
    shard: CheckpointShard,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any],
    limits: ArrayArchiveLimits,
    /,
) -> np.ndarray:
    metadata = _metadata(shard)
    artifact = repository.get_manifest(metadata["artifact_id"])
    if (
        artifact.manifest_id != metadata["artifact_manifest_id"]
        or shard.byte_count > limits.max_member_bytes
    ):
        raise RepositoryCorruptionError(
            "Checkpoint shard payload binding is inconsistent."
        )
    payload = _read_artifact_range(
        repository,
        artifact,
        _SHARD_PAYLOADS,
        _metadata_integer(metadata, "payload_offset"),
        shard.byte_count,
    )
    if hashlib.sha256(payload).hexdigest() != shard.payload_digest:
        raise RepositoryCorruptionError("Checkpoint shard payload digest mismatch.")
    stream = io.BytesIO(payload)
    try:
        dtype, shape, _ = _read_npy_metadata(stream, len(payload), limits)
    except ArrayArchiveCorruptionError as error:
        raise RepositoryCorruptionError(
            "Checkpoint shard NumPy payload metadata is invalid."
        ) from error
    if shape != expected_shape or dtype != expected_dtype:
        raise RepositoryCorruptionError(
            "Checkpoint shard payload shape or dtype is inconsistent."
        )
    stream.seek(0)
    try:
        value = np.load(
            stream,
            allow_pickle=False,
            max_header_size=limits.max_npy_header_bytes,
        )
    except (EOFError, OSError, ValueError) as error:
        raise RepositoryCorruptionError(
            "Checkpoint shard NumPy payload is invalid."
        ) from error
    if value.shape != expected_shape or value.dtype != expected_dtype:
        raise RepositoryCorruptionError("Checkpoint shard payload changed while loading.")
    return np.array(value, copy=True, order="C")


def _intersection(
    source: CanonicalIndex,
    destination: CanonicalIndex,
) -> tuple[tuple[slice, ...], tuple[slice, ...]] | None:
    source_slices = []
    destination_slices = []
    for source_axis, destination_axis in zip(source, destination, strict=True):
        source_start, source_stop, _ = source_axis
        destination_start, destination_stop, _ = destination_axis
        start = max(source_start, destination_start)
        stop = min(source_stop, destination_stop)
        if start >= stop:
            return None
        source_slices.append(slice(start - source_start, stop - source_start))
        destination_slices.append(
            slice(start - destination_start, stop - destination_start)
        )
    return tuple(source_slices), tuple(destination_slices)


def _checkpoint_array_loader(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    array_path: str,
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
) -> tuple[tuple[int, ...], Callable[[Any], np.ndarray]]:
    """Admit one canonical array and prepare bounded destination range loading."""

    if not isinstance(manifest, CheckpointManifest) or manifest.complete is not True:
        raise TypeError("manifest must be a complete CheckpointManifest")
    if not isinstance(array_path, str) or not array_path:
        raise ValueError("array_path must be non-empty text")
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance")
    parent_checkpoint_id, parent_manifest_id = _validate_parent_ownership(
        parent_manifest,
        manifest.checkpoint_id,
        manifest.analysis_plan_id,
        manifest.numeric_revision_id,
        manifest.execution_plan_id,
    )
    if manifest.parent_checkpoint_id != (
        None if parent_manifest is None else parent_checkpoint_id
    ) or manifest.parent_manifest_id != (
        None if parent_manifest is None else parent_manifest_id
    ):
        raise ValueError("Checkpoint restore parent lineage is inconsistent.")
    path_shards = tuple(
        shard
        for shard in manifest.shards
        if _metadata(shard).get("array_path") == array_path
    )
    if not path_shards:
        raise KeyError(f"checkpoint has no array at path {array_path!r}")
    for process_index in {
        _metadata_integer(_metadata(shard), "process_index") for shard in path_shards
    }:
        expected_parent = _parent_lineage(
            repository,
            parent_manifest,
            process_index,
            manifest.execution_plan_id,
            limits,
        )
        if expected_parent != (parent_checkpoint_id, parent_manifest_id):
            raise RepositoryCorruptionError(
                "Checkpoint restore parent repository lineage is inconsistent."
            )
    _validate_exact_coverage(path_shards, limits)
    _, global_shape, dtype, _ = _validated_shard(path_shards[0], limits)

    source_records = []
    # Preserve full-checkpoint integrity admission without retaining the global
    # payload on each host. Source bytes are validated one shard at a time;
    # callbacks then retain only one intersecting source and one local output.
    for shard in path_shards:
        _validate_shard_repository_binding(repository, manifest, shard, limits)
        _, shape, shard_dtype, source_index = _validated_shard(shard, limits)
        if shape != global_shape or shard_dtype != dtype:
            raise RepositoryCorruptionError(
                "Checkpoint array shard metadata is inconsistent."
            )
        local_shape = tuple(stop - start for start, stop, _ in source_index)
        _read_shard_payload(
            repository,
            shard,
            local_shape,
            dtype,
            limits,
        )
        source_records.append((source_index, local_shape, shard))
    admitted_sources = tuple(source_records)

    def load_destination(index: Any) -> np.ndarray:
        destination = _canonical_index(index, global_shape)
        local_shape = tuple(stop - start for start, stop, _ in destination)
        output = np.empty(local_shape, dtype=dtype)
        covered = np.zeros(local_shape, dtype=np.bool_)
        for source_index, source_shape, shard in admitted_sources:
            overlap = _intersection(source_index, destination)
            if overlap is None:
                continue
            payload = _read_shard_payload(repository, shard, source_shape, dtype, limits)
            source_slice, destination_slice = overlap
            output[destination_slice] = payload[source_slice]
            covered[destination_slice] = True
        if not bool(np.all(covered)):
            raise RepositoryCorruptionError(
                "Destination checkpoint shard is not fully covered."
            )
        return output

    return global_shape, load_destination


@jax.jit(donate_argnums=(0,))
def _place_checkpoint_packet(
    destination: jax.Array,
    packet: jax.Array,
    starts: tuple[int, ...],
) -> jax.Array:
    return jax.lax.dynamic_update_slice(destination, packet, starts)


def _restore_checkpoint_packets(
    manifest: CheckpointManifest,
    array_path: str,
    global_shape: tuple[int, ...],
    load_destination: Callable[[Any], np.ndarray],
    sharding: Sharding,
    limits: ArrayArchiveLimits,
    maximum_host_packet_bytes: int,
) -> jax.Array:
    """Assemble admitted addressable archive packets only in device storage."""
    budget = operator.index(maximum_host_packet_bytes)
    if isinstance(maximum_host_packet_bytes, bool) or budget < 1:
        raise ValueError("maximum_host_packet_bytes must be a positive integer.")
    if isinstance(sharding, NamedSharding):
        placement = NamedSharding(sharding.mesh, PartitionSpec())
    elif isinstance(sharding, SingleDeviceSharding):
        placement = sharding
    else:
        raise TypeError(
            "Packet restoration requires an explicit named or single-device sharding."
        )
    records = tuple(
        _validated_shard(shard, limits)
        for shard in manifest.shards
        if _metadata(shard)["array_path"] == array_path
    )
    dtype = records[0][2]
    for _, _, source_dtype, index in records:
        if _index_size(index, limits.max_array_elements) * source_dtype.itemsize > budget:
            raise ValueError(
                "An archived numerical packet exceeds maximum_host_packet_bytes."
            )
    result = jnp.zeros(global_shape, dtype=dtype, device=placement)
    for _, _, _, index in records:
        destination = tuple(slice(start, stop, step) for start, stop, step in index)
        packet = jax.device_put(load_destination(destination), placement)
        result = _place_checkpoint_packet(
            result, packet, tuple(start for start, _, _ in index)
        )
        # This is the explicit cold-I/O packet barrier, not a runtime topology
        # iteration. It bounds live ingress packets while donating the device
        # assembly buffer; no complete logical host destination is allocated.
        result.block_until_ready()
    return jax.device_put(result, sharding)


def restore_global_array_from_checkpoint(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    array_path: str,
    sharding: Sharding,
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
    maximum_host_packet_bytes: int | None = None,
) -> jax.Array:
    """Restore one globally bounded array into a destination sharding.

    With ``maximum_host_packet_bytes``, restore original archive rectangles
    individually and assemble only on devices. The limit bounds each decoded
    numerical packet, not aggregate process peak memory or archive codec bytes.
    This avoids a complete global NumPy destination when owner count shrinks.
    """
    global_shape, load_destination = _checkpoint_array_loader(
        repository, manifest, array_path, limits=limits, parent_manifest=parent_manifest
    )
    if maximum_host_packet_bytes is not None:
        return _restore_checkpoint_packets(
            manifest,
            array_path,
            global_shape,
            load_destination,
            sharding,
            limits,
            maximum_host_packet_bytes,
        )
    return jax.make_array_from_callback(global_shape, sharding, load_destination)


_MESHING_SOURCE_PREFIX = "source-record"


class _AdmittedMeshingSourceArrays(Strict, Mapping[str, jax.Array]):
    """Immutable exact logical-bank values with one canonical content admission."""

    _arrays: Mapping[str, jax.Array]
    content_digest: str
    maximum_chunk_bytes: int
    source_validation: MeshingSourceValidation | None

    def __init__(
        self,
        arrays: Mapping[str, jax.Array],
        /,
        *,
        maximum_chunk_bytes: int,
        source_validation: MeshingSourceValidation | None = None,
    ) -> None:
        from types import MappingProxyType

        from .._fingerprint import logical_array_value_collection_digest

        if isinstance(arrays, _AdmittedMeshingSourceArrays):
            self._arrays = arrays._arrays
            self.content_digest = arrays.content_digest
        else:
            values = dict(sorted(arrays.items()))
            if any(
                not isinstance(name, str) or not name or not isinstance(value, jax.Array)
                for name, value in values.items()
            ):
                raise TypeError(
                    "Source logical banks require canonical names and exact JAX arrays."
                )
            self._arrays = MappingProxyType(values)
            self.content_digest = logical_array_value_collection_digest(
                values, maximum_chunk_bytes=maximum_chunk_bytes
            )
        if source_validation is not None:
            source_validation.require_binding(self._arrays, self.content_digest)
        self.maximum_chunk_bytes = maximum_chunk_bytes
        self.source_validation = source_validation

    def __getitem__(self, name: str, /) -> jax.Array:
        return self._arrays[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)


def _admit_meshing_source_logical_arrays(
    arrays: Mapping[str, jax.Array] | None,
    /,
    *,
    maximum_chunk_bytes: int = 1 << 20,
) -> Mapping[str, jax.Array] | None:
    if isinstance(arrays, _AdmittedMeshingSourceArrays):
        return arrays
    if arrays is None or not arrays:
        return None
    return _AdmittedMeshingSourceArrays(arrays, maximum_chunk_bytes=maximum_chunk_bytes)


def _source_array_name(
    name: str, owner_index: int, bindings: Mapping[str, str], /
) -> str:
    alias = bindings.get(name)
    if alias is not None:
        return _source_logical_array_name(alias)
    return f"source-owner-{owner_index}:{name.replace('/', ':')}"


def _source_logical_array_name(alias: str, /) -> str:
    return f"source-logical-{canonical_fingerprint(alias)}"


def _meshing_source_recipe(
    recipe_json: str,
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
) -> dict[str, Any]:
    from .._model._structure import validate_model_structure_recipe
    from ._meshing_sources import register_meshing_source_artifacts

    register_meshing_source_artifacts()
    # A semantic recipe edge can use a mapping object, its items list and an
    # item-pair list. The owning recipe validator still admits semantic depth.
    recipe = _metadata_json(
        recipe_json,
        maximum_nesting=4 * limits.max_manifest_nesting + 2,
        maximum_bytes=limits.max_manifest_bytes,
    )
    if not isinstance(recipe, dict) or canonical_json(recipe) != recipe_json:
        raise ValueError("Source closure recipe must be canonical JSON.")
    validate_model_structure_recipe(recipe, limits=limits)
    return recipe


def _validate_meshing_source_logical_bank(
    inventory: Sequence[ModelRecipeArray],
    array_bindings: Mapping[str, str],
    logical_arrays: Mapping[str, jax.Array] | None,
    logical_array_names: Sequence[str],
    logical_content_digest: str | None,
    /,
    *,
    maximum_chunk_bytes: int,
) -> tuple[tuple[str, ...], str | None]:
    from ._chunk_repository import _digest

    entries = {entry.name: entry for entry in inventory}
    if set(array_bindings) - set(entries):
        raise ValueError("Source logical bindings refer to undeclared recipe arrays.")
    if any(
        entries[name].backend != "jax" or not isinstance(alias, str) or not alias
        for name, alias in array_bindings.items()
    ):
        raise ValueError("Source logical bindings require JAX arrays and explicit names.")
    names = tuple(sorted(logical_array_names))
    if len(set(names)) != len(names) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("Source logical array names must be unique canonical strings.")
    admitted = _admit_meshing_source_logical_arrays(
        logical_arrays, maximum_chunk_bytes=maximum_chunk_bytes
    )
    if admitted is not None:
        actual_names = tuple(sorted(admitted))
        if names and names != actual_names:
            raise ValueError("Source logical bank inventory changed.")
        names = actual_names
        if not isinstance(admitted, _AdmittedMeshingSourceArrays):
            raise TypeError(
                "Logical bank admission must retain its exact verified mapping."
            )
        actual_digest = admitted.content_digest
        if logical_content_digest is not None and logical_content_digest != actual_digest:
            raise ValueError("Source logical numerical content changed.")
        logical_content_digest = actual_digest if names else None
    if set(array_bindings.values()) - set(names):
        raise ValueError(
            "Source recipe bindings require complete declared logical banks."
        )
    if names:
        if not isinstance(logical_content_digest, str):
            raise TypeError("Source logical banks require their complete content digest.")
        _digest(logical_content_digest, "source_logical_content_digest")
    elif logical_content_digest is not None:
        raise ValueError("An empty source logical bank cannot declare a content digest.")
    return names, logical_content_digest


def _meshing_source_content_digest(
    values: Mapping[str, jax.Array | np.ndarray],
    array_bindings: Mapping[str, str],
    logical_arrays: Mapping[str, jax.Array] | None,
    logical_content_digest: str | None,
    /,
    *,
    maximum_chunk_bytes: int,
    owner_index: int,
    array_paths: Mapping[str, str],
) -> str:
    from .._fingerprint import logical_array_value_collection_digest

    local_values: dict[str, jax.Array | np.ndarray] = {}
    for name, value in values.items():
        alias = array_bindings.get(name)
        if alias is not None:
            if logical_arrays is None or value is not logical_arrays[alias]:
                raise ValueError(
                    f"Source owner {owner_index} field {array_paths[name]} "
                    f"must retain exact logical binding {alias!r} value identity."
                )
        else:
            if isinstance(value, jax.Array) and not value.is_fully_addressable:
                raise ValueError(
                    f"Nonaddressable source owner {owner_index} field {array_paths[name]} "
                    "requires its explicit logical binding."
                )
            local_values[name] = value
    return canonical_fingerprint(
        {
            "local_content_digest": logical_array_value_collection_digest(
                local_values, maximum_chunk_bytes=maximum_chunk_bytes
            ),
            "logical_content_digest": logical_content_digest,
            "array_bindings": tuple(sorted(array_bindings.items())),
        }
    )


def _prepare_meshing_source_binding(
    closure: Mapping[str, Any] | None,
    recipe_json: str | None,
    content_digest: str | None,
    /,
    *,
    owner_index: int,
    array_bindings: Mapping[str, str],
    logical_arrays: Mapping[str, jax.Array] | None,
    logical_array_names: Sequence[str],
    logical_content_digest: str | None,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    maximum_chunk_bytes: int = 1 << 20,
) -> tuple[
    str | None,
    str | None,
    tuple[tuple[str, MeshingArrayRole], ...],
    tuple[str, ...],
    str | None,
    Mapping[str, jax.Array] | None,
]:
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be ArrayArchiveLimits.")
    if closure is None and recipe_json is None:
        if (
            content_digest is not None
            or array_bindings
            or logical_arrays
            or logical_array_names
            or logical_content_digest is not None
        ):
            raise ValueError("Source content requires its canonical recipe.")
        return None, None, (), (), None, None
    from .._model._structure import (
        model_recipe_array_inventory,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from ._chunk_repository import _digest
    from ._meshing_sources import (
        register_meshing_source_artifacts,
        validate_collective_mesh_source_epochs,
        validate_meshing_source_closure,
    )

    register_meshing_source_artifacts()
    if closure is not None:
        admitted = _admit_meshing_source_logical_arrays(
            logical_arrays, maximum_chunk_bytes=maximum_chunk_bytes
        )
        if admitted is None:
            admitted = _AdmittedMeshingSourceArrays(
                {}, maximum_chunk_bytes=maximum_chunk_bytes
            )
        if not isinstance(admitted, _AdmittedMeshingSourceArrays):
            raise TypeError(
                "Source validation requires the actual immutable bank admission."
            )
        if admitted.source_validation is None:
            proof = validate_collective_mesh_source_epochs(
                (closure,),
                source_logical_arrays=admitted,
                source_logical_content_digest=admitted.content_digest,
                maximum_chunk_bytes=maximum_chunk_bytes,
                limits=limits,
            )
            admitted = _AdmittedMeshingSourceArrays(
                admitted,
                maximum_chunk_bytes=maximum_chunk_bytes,
                source_validation=proof,
            )
        logical_arrays = admitted
        validate_meshing_source_closure(
            closure,
            limits=limits,
            source_logical_arrays=admitted,
            source_logical_content_digest=admitted.content_digest,
            maximum_chunk_bytes=maximum_chunk_bytes,
            source_validation=admitted.source_validation,
        )
        actual_recipe_json = canonical_json(model_structure_recipe(closure))
        if recipe_json is not None and actual_recipe_json != recipe_json:
            raise ValueError("Source closure records changed from their recipe.")
        recipe_json = actual_recipe_json
    if not isinstance(recipe_json, str):
        raise TypeError("Source closure recipe must be canonical JSON.")
    recipe = _meshing_source_recipe(recipe_json, limits=limits)
    inventory = model_recipe_array_inventory(
        recipe, prefix=_MESHING_SOURCE_PREFIX, limits=limits
    )
    names, logical_content_digest = _validate_meshing_source_logical_bank(
        inventory,
        array_bindings,
        logical_arrays,
        logical_array_names,
        logical_content_digest,
        maximum_chunk_bytes=maximum_chunk_bytes,
    )
    if closure is not None:
        values = model_recipe_array_values(
            closure, recipe, prefix=_MESHING_SOURCE_PREFIX, limits=limits
        )
        actual_digest = _meshing_source_content_digest(
            values,
            array_bindings,
            logical_arrays,
            logical_content_digest,
            maximum_chunk_bytes=maximum_chunk_bytes,
            owner_index=owner_index,
            array_paths={entry.name: entry.path for entry in inventory},
        )
        if content_digest is not None and content_digest != actual_digest:
            raise ValueError("Source closure numerical content changed.")
        content_digest = actual_digest
    if not isinstance(content_digest, str):
        raise TypeError("Source closure requires its numerical content digest.")
    _digest(content_digest, "source_content_digest")
    roles: tuple[tuple[str, MeshingArrayRole], ...] = tuple(
        sorted(
            {
                (
                    _source_array_name(entry.name, owner_index, array_bindings),
                    "source-record",
                )
                for entry in inventory
            }
            | {(_source_logical_array_name(name), "source-record") for name in names}
        )
    )
    return (
        recipe_json,
        content_digest,
        roles,
        names,
        logical_content_digest,
        logical_arrays,
    )


def _validate_meshing_source_references(
    closure: Mapping[str, Any],
    references: Mapping[str, str | None],
    /,
) -> None:
    from ._meshing_sources import validate_meshing_source_checkpoint_references

    validate_meshing_source_checkpoint_references(closure, references)


def _meshing_source_arrays(
    state: MeshingCheckpointState,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> dict[str, jax.Array | np.ndarray]:
    from .._model._structure import model_recipe_array_values

    if state.source_recipe_json is None:
        return {}
    if state.source_closure is None:
        raise ValueError("Publishing source closure requires the actual source records.")
    bank = state.source_logical_arrays
    if (
        not isinstance(bank, _AdmittedMeshingSourceArrays)
        or bank.source_validation is None
    ):
        raise ValueError(
            "Process publication requires its actual prior common source admission."
        )
    bank.source_validation.require_binding(bank, bank.content_digest)
    bindings = dict(state.source_array_bindings)
    _prepare_meshing_source_binding(
        state.source_closure,
        state.source_recipe_json,
        state.source_content_digest,
        owner_index=state.source_owner_index,
        array_bindings=bindings,
        logical_arrays=state.source_logical_arrays,
        logical_array_names=state.source_logical_array_names,
        logical_content_digest=state.source_logical_content_digest,
        limits=limits,
        maximum_chunk_bytes=(
            state.source_logical_arrays.maximum_chunk_bytes
            if isinstance(state.source_logical_arrays, _AdmittedMeshingSourceArrays)
            else 1 << 20
        ),
    )
    _validate_meshing_source_references(state.source_closure, state.to_payload())
    recipe = _meshing_source_recipe(state.source_recipe_json, limits=limits)
    local = {
        _source_array_name(name, state.source_owner_index, bindings): value
        for name, value in model_recipe_array_values(
            state.source_closure, recipe, prefix=_MESHING_SOURCE_PREFIX, limits=limits
        ).items()
        if name not in bindings
    }
    if state.source_logical_arrays is not None:
        local.update(
            {
                _source_logical_array_name(name): value
                for name, value in state.source_logical_arrays.items()
            }
        )
    return local


def _meshing_array_path(name: str, /) -> str:
    return jax.tree_util.keystr(
        (jax.tree_util.DictKey("arrays"), jax.tree_util.DictKey(name))
    )


def _validate_meshing_arrays(
    state: MeshingCheckpointState,
    arrays: Mapping[str, jax.Array | np.ndarray],
    /,
) -> None:
    if not isinstance(state, MeshingCheckpointState):
        raise TypeError("state must be MeshingCheckpointState.")
    if set(arrays) != {name for name, _ in state.array_roles}:
        raise ValueError("Meshing checkpoint arrays must exactly match declared roles.")
    for name, role in state.array_roles:
        array = arrays[name]
        if role == "source-record" and isinstance(array, np.ndarray):
            continue
        if not isinstance(array, jax.Array):
            raise TypeError("Meshing checkpoint scientific arrays must be JAX arrays.")
        if role == "entity-ids" and (
            array.ndim != 1 or not np.issubdtype(array.dtype, np.integer)
        ):
            raise ValueError("Stable entity IDs must be logical integer vectors.")


def publish_process_meshing_checkpoint(
    repository: ArtifactRepository,
    checkpoint_id: str,
    execution_plan_id: str,
    state: MeshingCheckpointState,
    arrays: Mapping[str, jax.Array],
    /,
    *,
    analysis_plan_id: str,
    numeric_revision_id: str,
    writer_id: str,
    attempt_id: str | None = None,
    encoding: ChunkEncoding = "identity",
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
) -> ProcessCheckpointPublication:
    """Publish accepted scientific leaves, never a prepared execution object.

    ``arrays`` are globally logical arrays, not a process's dense closure views.
    The canonical CellMesh storage owner supplies logical coordinates, stable
    IDs and global-ID incidences. Solver/history and accepted evidence owners
    supply the other declared leaves. Source-record leaves are taken from
    ``state.source_closure``; callers cannot replace them in ``arrays``. JAX
    numerical values cross to host only as addressable shards. Authored native
    NumPy coefficients are snapshotted directly on their canonical process owner.
    """

    if set(arrays) != {
        name for name, role in state.array_roles if role != "source-record"
    }:
        raise ValueError(
            "Accepted arrays must exactly cover non-source scientific roles."
        )
    source_arrays = _meshing_source_arrays(state, limits=limits)
    scientific_arrays = {**arrays, **source_arrays}
    _validate_meshing_arrays(state, scientific_arrays)
    payload = canonical_json(state.to_payload()).encode("utf-8")
    if len(payload) > limits.max_manifest_bytes:
        raise ValueError("Meshing checkpoint metadata exceeds its byte limit.")
    if (
        state.source_recipe_json is not None
        and state.source_owner_index != jax.process_index()
    ):
        raise ValueError(
            "Source publication must retain the current owner's exact record."
        )
    record_name = f"record-owner-{jax.process_index()}"
    record = np.frombuffer(payload, dtype=np.uint8)
    host_owner = jax.process_index()
    publication = publish_process_checkpoint(
        repository,
        checkpoint_id,
        execution_plan_id,
        {
            record_name: record,
            "arrays": scientific_arrays,
        },
        analysis_plan_id=analysis_plan_id,
        numeric_revision_id=numeric_revision_id,
        writer_id=writer_id,
        attempt_id=attempt_id,
        topology_epoch=state.mesh_epoch,
        encoding=encoding,
        limits=limits,
        parent_manifest=parent_manifest,
        host_array_owner=host_owner,
    )
    return replace(
        publication,
        meshing_state=MeshingCheckpointState.from_payload(
            state.to_payload(), limits=limits
        ),
    )


def _meshing_record_path(process_index: int, /) -> str:
    return jax.tree_util.keystr((jax.tree_util.DictKey(f"record-owner-{process_index}"),))


def _decode_meshing_state_record(
    record: np.ndarray,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> tuple[dict[str, Any], MeshingCheckpointState]:
    try:
        text = record.tobytes().decode("utf-8")
        payload = _metadata_json(text, maximum_bytes=limits.max_manifest_bytes)
        if not isinstance(payload, dict) or canonical_json(payload) != text:
            raise ValueError("Meshing checkpoint record must be canonical JSON.")
        state = MeshingCheckpointState.from_payload(payload, limits=limits)
    except (TypeError, ValueError, UnicodeError) as error:
        raise RepositoryCorruptionError(
            "Meshing checkpoint record is invalid."
        ) from error
    return payload, state


def _meshing_record_shard(
    shards: Sequence[CheckpointShard],
    process_index: int,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> CheckpointShard | None:
    path = _meshing_record_path(process_index)
    matching = tuple(shard for shard in shards if _metadata(shard)["array_path"] == path)
    if not matching:
        return None
    if len(matching) != 1:
        raise RepositoryCorruptionError(
            "An owner must publish exactly one canonical metadata record."
        )
    shard = matching[0]
    _, shape, dtype, index = _validated_shard(shard, limits)
    if (
        len(shape) != 1
        or shape[0] > limits.max_manifest_bytes
        or dtype != np.dtype(np.uint8)
        or index != ((0, shape[0], 1),)
        or _metadata_integer(_metadata(shard), "process_index") != process_index
    ):
        raise RepositoryCorruptionError(
            "Meshing checkpoint owner record layout is invalid."
        )
    return shard


def _read_process_meshing_state(
    repository: ArtifactRepository,
    shards: Sequence[CheckpointShard],
    process_index: int,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> MeshingCheckpointState | None:
    shard = _meshing_record_shard(shards, process_index, limits=limits)
    if shard is None:
        return None
    _, shape, dtype, _ = _validated_shard(shard, limits)
    record = _read_shard_payload(repository, shard, shape, dtype, limits)
    return _decode_meshing_state_record(record, limits=limits)[1]


def _validate_meshing_source_manifest_layouts(
    states: Sequence[MeshingCheckpointState],
    shards: Sequence[CheckpointShard],
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    from .._model._structure import model_recipe_array_inventory

    specifications = {}
    for shard in shards:
        path, shape, dtype, _ = _validated_shard(shard, limits)
        specifications[path] = (shape, dtype.str)
    for state in states:
        if state.source_recipe_json is None:
            continue
        recipe = _meshing_source_recipe(state.source_recipe_json, limits=limits)
        for entry in model_recipe_array_inventory(
            recipe, prefix=_MESHING_SOURCE_PREFIX, limits=limits
        ):
            path = _meshing_array_path(state.source_array_name(entry.name))
            if specifications[path] != (entry.shape, entry.dtype):
                raise RepositoryCorruptionError(
                    f"Source owner {state.source_owner_index} field {entry.path} "
                    "has inconsistent checkpoint shape or dtype."
                )


def _validate_meshing_manifest_admission(
    publications: Sequence[ProcessCheckpointPublication],
    shards: Sequence[CheckpointShard],
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    record_paths = {
        _metadata(shard)["array_path"]
        for shard in shards
        if _metadata(shard)["array_path"].startswith("['record-owner-")
    }
    if not record_paths and all(
        publication.meshing_state is None for publication in publications
    ):
        return
    ordered = tuple(
        sorted(publications, key=lambda publication: publication.process_index)
    )
    states = []
    for publication in ordered:
        state = publication.meshing_state
        if not isinstance(state, MeshingCheckpointState):
            raise RepositoryCorruptionError(
                "Meshing manifest requires every owner's canonical record acknowledgement."
            )
        shard = _meshing_record_shard(
            publication.shards, publication.process_index, limits=limits
        )
        if shard is None:
            raise RepositoryCorruptionError(
                "Meshing manifest lacks an acknowledged owner record."
            )
        record = canonical_json(state.to_payload()).encode("utf-8")
        payload = _array_payload(np.frombuffer(record, dtype=np.uint8))
        if len(record) > limits.max_manifest_bytes or (
            hashlib.sha256(payload).hexdigest() != shard.payload_digest
            or len(payload) != shard.byte_count
        ):
            raise RepositoryCorruptionError(
                "Meshing owner record acknowledgement changed its exact payload."
            )
        states.append(state)
    source_flags = tuple(state.source_recipe_json is not None for state in states)
    if any(source_flags) and not all(source_flags):
        raise RepositoryCorruptionError(
            "Source and no-source owner records cannot share an accepted checkpoint."
        )
    expected: str | Mapping[int, str] = (
        {index: state.checkpoint_state_id for index, state in enumerate(states)}
        if all(source_flags)
        else states[0].checkpoint_state_id
    )
    _validate_meshing_owner_states(states, expected, 0)
    expected_paths = {_meshing_record_path(index) for index in range(len(states))} | {
        _meshing_array_path(name) for state in states for name, _ in state.array_roles
    }
    if {_metadata(shard)["array_path"] for shard in shards} != expected_paths or any(
        _metadata_integer(_metadata(shard), "topology_epoch") != states[0].mesh_epoch
        for shard in shards
    ):
        raise RepositoryCorruptionError(
            "Meshing manifest scientific inventory or epoch is incomplete."
        )
    _validate_meshing_source_manifest_layouts(states, shards, limits=limits)


def _read_meshing_checkpoint_record(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    record_path: str,
    /,
    *,
    limits: ArrayArchiveLimits,
    parent_manifest: CheckpointManifest | None,
) -> tuple[dict[str, Any], MeshingCheckpointState]:
    shards = tuple(
        shard
        for shard in manifest.shards
        if _metadata(shard)["array_path"] == record_path
    )
    if not shards:
        raise RepositoryCorruptionError("Meshing checkpoint has no scientific record.")
    _, shape, dtype, _ = _validated_shard(shards[0], limits)
    if (
        len(shape) != 1
        or shape[0] > limits.max_manifest_bytes
        or dtype != np.dtype(np.uint8)
    ):
        raise RepositoryCorruptionError("Meshing checkpoint record layout is invalid.")
    record = restore_global_array_from_checkpoint(
        repository,
        manifest,
        record_path,
        SingleDeviceSharding(jax.local_devices()[0]),
        limits=limits,
        parent_manifest=parent_manifest,
    )
    payload, state = _decode_meshing_state_record(np.asarray(record), limits=limits)
    return payload, state


def _meshing_checkpoint_record_paths(manifest: CheckpointManifest, /) -> tuple[str, ...]:
    paths = {_metadata(shard)["array_path"] for shard in manifest.shards}
    processes = {
        _metadata_integer(_metadata(shard), "process_index") for shard in manifest.shards
    }
    if not processes or max(processes) >= len(manifest.shards):
        raise RepositoryCorruptionError(
            "Meshing source owner count exceeds its record coverage."
        )
    owner_paths = tuple(
        _meshing_record_path(index) for index in range(max(processes) + 1)
    )
    if not owner_paths or any(path not in paths for path in owner_paths):
        raise RepositoryCorruptionError(
            "Meshing source records do not cover every original owner."
        )
    for owner_index, path in enumerate(owner_paths):
        if any(
            _metadata_integer(_metadata(shard), "process_index") != owner_index
            for shard in manifest.shards
            if _metadata(shard)["array_path"] == path
        ):
            raise RepositoryCorruptionError("Meshing source record ownership is invalid.")
    return owner_paths


def read_meshing_checkpoint_states(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
) -> tuple[MeshingCheckpointState, ...]:
    """Read bounded canonical owner recipes without loading scientific leaves.

    Source records remain absent until full restore. The returned declarations
    let cold callers prepare placement from the accepted archive alone; callers
    still bind the archive/record identities to their accepted composition.
    """
    if not isinstance(manifest, CheckpointManifest) or manifest.complete is not True:
        raise TypeError("manifest must be a complete CheckpointManifest")
    _validate_exact_coverage(manifest.shards, limits)
    return tuple(
        _read_meshing_checkpoint_record(
            repository, manifest, path, limits=limits, parent_manifest=parent_manifest
        )[1]
        for path in _meshing_checkpoint_record_paths(manifest)
    )


def _validate_meshing_owner_states(
    states: Sequence[MeshingCheckpointState],
    expected_state_id: str | Mapping[int, str],
    source_owner_index: int,
    /,
) -> None:
    if isinstance(expected_state_id, str):
        if len(states) != 1 and any(
            state.source_recipe_json is not None for state in states
        ):
            raise ValueError(
                "Multi-owner source restart requires every original state ID."
            )
        expected_ids = dict.fromkeys(range(len(states)), expected_state_id)
    elif isinstance(expected_state_id, Mapping):
        expected_ids = dict(expected_state_id)
        if any(type(index) is not int for index in expected_ids) or set(
            expected_ids
        ) != set(range(len(states))):
            raise ValueError(
                "Expected state IDs must cover every original owner exactly once."
            )
    else:
        raise TypeError("expected_state_id must be a state ID or complete owner mapping.")
    if type(source_owner_index) is not int or not 0 <= source_owner_index < len(states):
        raise ValueError("source_owner_index must select one original source owner.")
    for index, state in enumerate(states):
        if state.checkpoint_state_id != expected_ids[index]:
            raise RepositoryCorruptionError(
                "Meshing checkpoint scientific binding changed."
            )
        if state.source_recipe_json is not None and state.source_owner_index != index:
            raise RepositoryCorruptionError(
                "Meshing source record changed its original owner."
            )
    scientific_roles = tuple(
        item for item in states[0].array_roles if item[1] != "source-record"
    )
    if any(
        tuple(item for item in state.array_roles if item[1] != "source-record")
        != scientific_roles
        or state.mesh_epoch != states[0].mesh_epoch
        or state.source_logical_array_names != states[0].source_logical_array_names
        or state.source_logical_content_digest != states[0].source_logical_content_digest
        for state in states[1:]
    ):
        raise RepositoryCorruptionError(
            "Source owners disagree on shared logical scientific state."
        )


def restore_meshing_checkpoint(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    shardings: Mapping[str, Sharding],
    relation: TopologyRestartRelation,
    policy: TopologyRestartPolicy,
    /,
    *,
    expected_state_id: str | Mapping[int, str],
    source_owner_index: int = 0,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
    maximum_host_packet_bytes: int | None = None,
) -> tuple[MeshingCheckpointState, dict[str, jax.Array | np.ndarray], RestartAdmission]:
    """Restore exact source-owner records and logical arrays into new placement.

    The owner bank retains complete historical typed source records, including
    local result shapes and original collective proofs. It is not a new mesh
    carrier. Repack/reprepare consumes ``state.source_owner_states`` and selects
    an authoritative original record explicitly; source ranks never determine
    successor scientific IDs or local slots. Multi-owner archives require the
    complete expected identity mapping, not a single partial owner hash.
    """
    admission = admit_topology_restart(relation, policy)
    if not admission.admitted:
        raise ValueError(f"Topology restart is not admitted: {admission.reason}")
    if not isinstance(manifest, CheckpointManifest) or manifest.complete is not True:
        raise TypeError("manifest must be a complete CheckpointManifest")
    _validate_exact_coverage(manifest.shards, limits)
    record_paths = _meshing_checkpoint_record_paths(manifest)
    records = tuple(
        _read_meshing_checkpoint_record(
            repository, manifest, path, limits=limits, parent_manifest=parent_manifest
        )
        for path in record_paths
    )
    states = tuple(state for _, state in records)
    _validate_meshing_owner_states(states, expected_state_id, source_owner_index)
    from .._model._structure import (
        model_from_logical_array_recipe,
        model_recipe_array_inventory,
    )

    recipes = tuple(
        None
        if state.source_recipe_json is None
        else _meshing_source_recipe(state.source_recipe_json, limits=limits)
        for state in states
    )
    inventories = tuple(
        ()
        if recipe is None
        else model_recipe_array_inventory(
            recipe, prefix=_MESHING_SOURCE_PREFIX, limits=limits
        )
        for recipe in recipes
    )
    names = {name for state in states for name, _ in state.array_roles}
    host_names = {
        state.source_array_name(entry.name)
        for state, inventory in zip(states, inventories, strict=True)
        for entry in inventory
        if entry.backend == "numpy"
    }
    if set(shardings) != names - host_names:
        raise ValueError(
            "Destination shardings must exactly cover logical JAX scientific arrays."
        )
    expected_paths = set(record_paths) | {_meshing_array_path(name) for name in names}
    if {
        _metadata(shard)["array_path"] for shard in manifest.shards
    } != expected_paths or any(
        _metadata_integer(_metadata(shard), "topology_epoch") != states[0].mesh_epoch
        for shard in manifest.shards
    ):
        raise RepositoryCorruptionError(
            "Meshing checkpoint array/epoch binding is invalid."
        )
    arrays: dict[str, jax.Array | np.ndarray] = {
        name: restore_global_array_from_checkpoint(
            repository,
            manifest,
            _meshing_array_path(name),
            shardings[name],
            limits=limits,
            parent_manifest=parent_manifest,
            maximum_host_packet_bytes=maximum_host_packet_bytes,
        )
        for name in sorted(names - host_names)
    }
    for name in sorted(host_names):
        shape, load_host = _checkpoint_array_loader(
            repository,
            manifest,
            _meshing_array_path(name),
            limits=limits,
            parent_manifest=parent_manifest,
        )
        host = load_host(tuple(slice(0, size) for size in shape))
        host.setflags(write=False)
        arrays[name] = host
    logical_arrays: dict[str, jax.Array] = {}
    for name in states[0].source_logical_array_names:
        value = arrays[_source_logical_array_name(name)]
        if not isinstance(value, jax.Array):
            raise RepositoryCorruptionError(
                "A source logical bank changed its JAX backend."
            )
        logical_arrays[name] = value
    source_chunk_bytes = (
        1 << 20 if maximum_host_packet_bytes is None else maximum_host_packet_bytes
    )
    admitted_logical_arrays = _admit_meshing_source_logical_arrays(
        logical_arrays, maximum_chunk_bytes=source_chunk_bytes
    )
    decoded_closures: list[Mapping[str, Any] | None] = []
    for (_, state), recipe, inventory in zip(records, recipes, inventories, strict=True):
        if recipe is None:
            decoded_closures.append(None)
            continue
        source_arrays = {
            entry.name: arrays[state.source_array_name(entry.name)] for entry in inventory
        }
        try:
            closure = model_from_logical_array_recipe(
                recipe, source_arrays, prefix=_MESHING_SOURCE_PREFIX, limits=limits
            )
            if not isinstance(closure, Mapping):
                raise TypeError("Source closure must be an immutable scientific mapping.")
            decoded_closures.append(closure)
        except (TypeError, ValueError) as error:
            raise RepositoryCorruptionError(
                "Meshing source closure is invalid."
            ) from error
    source_roots = tuple(closure for closure in decoded_closures if closure is not None)
    if source_roots:
        from ._meshing_sources import validate_collective_mesh_source_epochs

        if admitted_logical_arrays is None:
            admitted_logical_arrays = _AdmittedMeshingSourceArrays(
                {}, maximum_chunk_bytes=source_chunk_bytes
            )
        if not isinstance(admitted_logical_arrays, _AdmittedMeshingSourceArrays):
            raise TypeError(
                "Source restore must retain its exact immutable bank admission."
            )
        proof = validate_collective_mesh_source_epochs(
            source_roots,
            source_logical_arrays=admitted_logical_arrays,
            source_logical_content_digest=admitted_logical_arrays.content_digest,
            maximum_chunk_bytes=source_chunk_bytes,
            limits=limits,
        )
        admitted_logical_arrays = _AdmittedMeshingSourceArrays(
            admitted_logical_arrays,
            maximum_chunk_bytes=source_chunk_bytes,
            source_validation=proof,
        )
    restored_states = []
    for (payload, state), closure in zip(records, decoded_closures, strict=True):
        if closure is None:
            restored_states.append(state)
            continue
        try:
            restored = MeshingCheckpointState.from_payload(
                payload,
                source_closure=closure,
                source_logical_arrays=admitted_logical_arrays,
                limits=limits,
                maximum_source_chunk_bytes=source_chunk_bytes,
            )
        except (TypeError, ValueError) as error:
            raise RepositoryCorruptionError(
                "Meshing source closure is invalid."
            ) from error
        _validate_meshing_arrays(
            restored, {name: arrays[name] for name, _ in restored.array_roles}
        )
        restored_states.append(restored)
    selected = restored_states[source_owner_index]
    if selected.source_closure is not None:
        selected = MeshingCheckpointState.from_payload(
            records[source_owner_index][0],
            source_closure=selected.source_closure,
            source_logical_arrays=admitted_logical_arrays,
            source_owner_states=restored_states,
            limits=limits,
            maximum_source_chunk_bytes=source_chunk_bytes,
        )
    else:
        _validate_meshing_arrays(selected, arrays)
    return selected, arrays, admission


__all__ = (
    "AddressableCheckpointShard",
    "assemble_distributed_checkpoint_from_repository",
    "ProcessCheckpointPublication",
    "assemble_distributed_checkpoint_manifest",
    "publish_process_checkpoint",
    "publish_process_meshing_checkpoint",
    "read_meshing_checkpoint_states",
    "restore_meshing_checkpoint",
    "restore_global_array_from_checkpoint",
    "snapshot_addressable_arrays",
)
