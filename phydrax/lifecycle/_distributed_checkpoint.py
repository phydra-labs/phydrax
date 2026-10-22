#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Addressable-shard checkpoint publication and topology-neutral restore."""

from __future__ import annotations

import hashlib
import io
import operator
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, Sharding, SingleDeviceSharding

from .._array_archive import (
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_npy_metadata,
)
from .._fingerprint import (
    canonical_fingerprint,
    canonical_json,
)
from .._host_storage import immutable_host_array
from .._model._structure import (
    model_from_logical_array_recipe,
    model_recipe_array_inventory,
    ModelRecipeArray,
)
from .._strict import StrictModule
from ..typing import checked
from ._chunk_repository import (
    ArtifactManifest,
    ArtifactRepository,
    ChunkEncoding,
    ChunkRecord,
    RepositoryCorruptionError,
    RepositoryTransaction,
)
from ._meshing_checkpoint_binding import (
    admit_meshing_source_logical_arrays,
    AdmittedMeshingSourceArrays,
    authenticate_meshing_source_publication,
    MESHING_SOURCE_PREFIX,
    meshing_source_recipe,
    meshing_source_restore_admission,
    metadata_json,
    source_array_name,
    source_logical_array_name,
)
from ._meshing_sources import (
    meshing_source_local_restore_phase,
    meshing_source_owner_reconstruction_scope,
    validate_collective_mesh_source_epochs,
    validate_meshing_source_checkpoint_references,
)
from ._models import CheckpointManifest, CheckpointShard
from ._restart_topology import (
    admit_topology_restart,
    MeshingCheckpointState,
    RestartAdmission,
    TopologyRestartPolicy,
    TopologyRestartRelation,
)


CanonicalIndex = tuple[tuple[int, int, int], ...]
_DISTRIBUTED_CHECKPOINT_LIMITS = DEFAULT_ARRAY_ARCHIVE_LIMITS
_NO_PARENT_ID = "none"
# One process artifact holds two logical streams: packed shard payloads and the
# deterministically encoded JSON shard table that metadata references by digest.
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


def _canonical_index(index: Any, shape: tuple[int, ...]) -> CanonicalIndex:
    entries = index if isinstance(index, tuple) else (index,)
    if len(entries) != len(shape):
        raise ValueError(
            f"checkpoint.shard_index must match the global array rank; got {len(entries)} axes for rank {len(shape)}"
        )
    normalized: list[tuple[int, int, int]] = []
    for entry, size in zip(entries, shape, strict=True):
        match entry:
            case int():
                value = entry if entry >= 0 else size + entry
                if not 0 <= value < size:
                    raise ValueError(
                        f"checkpoint.shard_index must lie inside the global axis; got index {entry} for extent {size}"
                    )
                normalized.append((value, value + 1, 1))
            case slice():
                start, stop, step = entry.indices(size)
                if step != 1:
                    raise ValueError(
                        f"checkpoint.shard_index must use unit-stride slices; got stride {step}"
                    )
                normalized.append((start, stop, step))
            case _:
                raise TypeError(
                    f"checkpoint.shard_index must contain integers or slices; got {type(entry).__name__}"
                )
    return tuple(normalized)


def _admit_array_spec(
    shape_value: Any,
    dtype_value: Any,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[tuple[int, ...], np.dtype[Any], int]:
    if not isinstance(shape_value, list) or any(
        type(extent) is not int or extent < 0 for extent in shape_value
    ):
        raise RepositoryCorruptionError(
            "checkpoint.array.shape must contain nonnegative integer extents; got an invalid global shape"
        )
    shape = tuple(shape_value)
    if len(shape) > limits.max_array_rank:
        raise RepositoryCorruptionError(
            f"checkpoint.array.shape must have rank at most {limits.max_array_rank}; got rank {len(shape)}"
        )
    try:
        dtype = np.dtype(dtype_value)
    except (TypeError, ValueError) as error:
        raise RepositoryCorruptionError(
            "checkpoint.array.dtype must resolve to a NumPy dtype; got an invalid dtype"
        ) from error
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.metadata is not None
        or dtype.kind not in limits.allowed_dtype_kinds
        or dtype.itemsize > limits.max_dtype_itemsize
    ):
        raise RepositoryCorruptionError(
            "checkpoint.array.dtype must satisfy the archive dtype policy; got a dtype not admitted by policy"
        )
    elements = 1
    for extent in shape:
        if extent > limits.max_axis_length:
            raise RepositoryCorruptionError(
                f"checkpoint.array.shape must have axes at most {limits.max_axis_length}; got extent {extent}"
            )
        if extent and elements > limits.max_total_array_elements // extent:
            raise RepositoryCorruptionError(
                "checkpoint.array.shape must fit the aggregate element limit; got aggregate element limit exceeded"
            )
        elements *= extent
    if elements > limits.max_total_array_elements:
        raise RepositoryCorruptionError(
            "checkpoint.array.shape must fit the aggregate element limit; got aggregate element limit exceeded"
        )
    byte_count = elements * dtype.itemsize
    if byte_count > limits.max_aggregate_bytes:
        raise RepositoryCorruptionError(
            f"checkpoint.array.nbytes must not exceed {limits.max_aggregate_bytes}; got {byte_count} bytes"
        )
    return shape, dtype, elements


def _parse_index(value: str, shape: tuple[int, ...], /) -> CanonicalIndex:
    raw = metadata_json(value)
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
        raise RepositoryCorruptionError(
            "checkpoint.shard_index must contain integer start-stop-stride triples matching the global rank; got an invalid index encoding"
        )
    index = tuple(tuple(entry) for entry in raw)
    if any(
        step != 1 or start < 0 or stop < start or stop > extent
        for (start, stop, step), extent in zip(index, shape, strict=True)
    ):
        raise RepositoryCorruptionError(
            "checkpoint.shard_index must contain ordered unit-stride ranges inside the global shape; got out-of-bounds shard ranges"
        )
    return index


def _index_size(index: CanonicalIndex, maximum: int, /) -> int:
    size = 1
    for start, stop, step in index:
        if step != 1:
            raise RepositoryCorruptionError(
                f"checkpoint.shard_index must use unit stride; got {step}"
            )
        extent = stop - start
        if extent and size > maximum // extent:
            raise RepositoryCorruptionError(
                "checkpoint.shard_index must fit the element limit; got shard element limit exceeded"
            )
        size *= extent
    return size


def _indices_overlap(left: CanonicalIndex, right: CanonicalIndex) -> bool:
    if len(left) != len(right):
        raise ValueError(
            f"checkpoint.shard_index must have equal ranks for overlap comparison; got {len(left)} and {len(right)} axes"
        )
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
    """Snapshot one non-redundant addressable array shard on the host."""

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


class _PendingCheckpointShard(StrictModule):
    array_path: str = eqx.field(static=True)
    global_shape: tuple[int, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    index: CanonicalIndex = eqx.field(static=True)
    process_index: int = eqx.field(static=True)
    device_id: int = eqx.field(static=True)
    replica_id: int = eqx.field(static=True)
    data: jax.Array


def _snapshot_host_array(
    array_path: str,
    leaf: np.ndarray,
    host_array_owner: int,
    topology_epoch: int,
) -> AddressableCheckpointShard:
    # Authored coefficients stay on the host, with one publishing replica.
    payload = immutable_host_array(leaf)
    if payload.flags.writeable:
        payload = np.array(payload, copy=True, order="C")
    payload.setflags(write=False)
    shape = tuple(leaf.shape)
    index = tuple((0, size, 1) for size in shape)
    device = next(
        device for device in jax.devices() if device.process_index == host_array_owner
    )
    dtype = np.dtype(leaf.dtype).str
    process_index, device_id = int(device.process_index), int(device.id)
    return AddressableCheckpointShard(
        array_path,
        shape,
        dtype,
        index,
        process_index,
        device_id,
        0,
        payload,
        _layout_id(
            array_path, shape, dtype, index, process_index, device_id, 0, topology_epoch
        ),
    )


def _stage_addressable_shards(
    array_path: str,
    leaf: jax.Array,
    pending: list[_PendingCheckpointShard],
    snapshot_count: int,
    limits: ArrayArchiveLimits,
) -> None:
    shape = tuple(leaf.shape)
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
        if len(pending) + snapshot_count >= limits.max_members:
            raise ValueError(
                "checkpoint.shards must fit the shard-count limit; got shard-count limit exceeded"
            )
        shard.data.copy_to_host_async()
        pending.append(
            _PendingCheckpointShard(
                array_path,
                shape,
                np.dtype(leaf.dtype).str,
                normalized,
                device_key[0],
                device_key[1],
                int(shard.replica_id),
                shard.data,
            )
        )


def _snapshot_staged_shard(
    staged: _PendingCheckpointShard,
    topology_epoch: int,
    limits: ArrayArchiveLimits,
) -> AddressableCheckpointShard:
    payload = np.asarray(staged.data)
    local_shape = tuple(stop - start for start, stop, _ in staged.index)
    if (
        payload.shape != local_shape
        or payload.dtype != np.dtype(staged.dtype)
        or payload.nbytes > limits.max_member_bytes
    ):
        raise ValueError(
            "checkpoint.shard.payload must match its layout and byte limit; got inconsistent shape, dtype, or byte count"
        )
    return AddressableCheckpointShard(
        staged.array_path,
        staged.global_shape,
        staged.dtype,
        staged.index,
        staged.process_index,
        staged.device_id,
        staged.replica_id,
        payload,
        _layout_id(
            staged.array_path,
            staged.global_shape,
            staged.dtype,
            staged.index,
            staged.process_index,
            staged.device_id,
            staged.replica_id,
            topology_epoch,
        ),
    )


@checked
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
        raise ValueError(
            f"checkpoint.topology_epoch must be a nonnegative integer; got {topology_epoch if type(topology_epoch) is int else type(topology_epoch).__name__!r}"
        )
    if (
        type(host_array_owner) is not int
        or not 0 <= host_array_owner < jax.process_count()
    ):
        raise ValueError(
            f"checkpoint.host_array_owner must identify a current process; got {host_array_owner if type(host_array_owner) is int else type(host_array_owner).__name__!r}"
        )
    pending: list[_PendingCheckpointShard] = []
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
            raise ValueError(
                "checkpoint.arrays must fit the aggregate byte limit; got aggregate byte limit exceeded"
            )
        total_bytes += byte_count
        match leaf:
            case np.ndarray():
                if jax.process_index() != host_array_owner:
                    continue
                if len(pending) + len(snapshots) >= limits.max_members:
                    raise ValueError(
                        "checkpoint.shards must fit the shard-count limit; got shard-count limit exceeded"
                    )
                snapshots.append(
                    _snapshot_host_array(
                        array_path, leaf, host_array_owner, topology_epoch
                    )
                )
            case jax.Array():
                _stage_addressable_shards(
                    array_path, leaf, pending, len(snapshots), limits
                )
            case _:
                raise TypeError(
                    f"checkpoint.leaf must be a JAX or NumPy array; got {type(leaf).__name__}"
                )

    for staged in pending:
        snapshots.append(_snapshot_staged_shard(staged, topology_epoch, limits))
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


def _iter_artifact_range(
    repository: ArtifactRepository,
    artifact: ArtifactManifest,
    logical_name: str,
    offset: int,
    byte_count: int,
    /,
) -> Iterator[memoryview]:
    """Stream one plaintext byte range of a logical artifact in bounded chunks."""

    stop = offset + byte_count
    observed = 0
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
            raise RepositoryCorruptionError(
                "artifact.chunk.plaintext_size must match the decoded chunk; got chunk size mismatch"
            )
        payload = memoryview(data)[max(offset - chunk.offset, 0) : stop - chunk.offset]
        observed += len(payload)
        yield payload
    if observed != byte_count:
        raise RepositoryCorruptionError(
            "artifact.payload_range must be covered by its chunks; got uncovered payload bytes"
        )


def _read_artifact_range(
    repository: ArtifactRepository,
    artifact: ArtifactManifest,
    logical_name: str,
    offset: int,
    byte_count: int,
    /,
) -> bytes:
    """Read one admitted metadata range of a logical artifact stream."""

    return b"".join(
        _iter_artifact_range(repository, artifact, logical_name, offset, byte_count)
    )


@dataclass(frozen=True, slots=True)
class _ProcessShardTable:
    """Admitted shard records of one immutable committed process artifact."""

    shards: tuple[CheckpointShard, ...]
    fingerprints: frozenset[str]


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
            "artifact.metadata must bind the declared owner and streams to this repository; got invalid process artifact metadata"
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
            "artifact.shard_descriptors must contain bounded records with declared fields and ownership; got invalid shard descriptors"
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
                "artifact.shard_descriptors must contain bounded records with declared fields and ownership; got invalid shard descriptors"
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
                "artifact.shard_descriptors must contain bounded records with declared fields and ownership; got invalid shard descriptors"
            ) from error
        _validated_shard(shard, limits)
        if offset > limits.max_aggregate_bytes - shard.byte_count:
            raise RepositoryCorruptionError(
                "checkpoint.shards must fit the aggregate payload-byte limit; got aggregate payload-byte limit exceeded"
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
            "artifact.shard_descriptors must fit the manifest byte bound and declared stream size; got descriptor byte bound or size mismatch"
        )
    payload = _read_artifact_range(
        repository, artifact, _SHARD_DESCRIPTORS, 0, descriptor_bytes
    )
    if hashlib.sha256(payload).hexdigest() != metadata["shard_descriptor_sha256"]:
        raise RepositoryCorruptionError(
            "artifact.shard_descriptor_sha256 must match descriptor content; got descriptor digest mismatch"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise RepositoryCorruptionError(
            "artifact.shard_descriptors must contain bounded records with declared fields and ownership; got invalid shard descriptors"
        ) from error
    descriptors = metadata_json(text, maximum_bytes=limits.max_manifest_bytes)
    if canonical_json(descriptors) != text:
        raise RepositoryCorruptionError(
            "artifact.shard_descriptors must use deterministic JSON encoding; got different bytes"
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
            f"ProcessCheckpoint.artifact.chunks must contain only shard descriptors and {payload_bytes} shard payload bytes; got {tuple((chunk.logical_name, chunk.plaintext_size) for chunk in artifact.chunks)!r}."
        )
    table = _ProcessShardTable(
        shards, frozenset(shard.shard_fingerprint for shard in shards)
    )
    return table


@checked
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
    if parent_manifest.complete is not True:
        raise TypeError(
            "CheckpointManifest.complete must be True for checkpoint lineage; got False"
        )
    if parent_manifest.execution_plan_id != execution_plan_id:
        raise ValueError(
            "parent_manifest.execution_plan_id must match the successor execution plan; got incompatible execution ownership"
        )
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
            "parent_artifact.metadata must match its parent manifest and repository; got inconsistent artifact ownership"
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
            "parent_manifest.shards must match the committed process artifact; got inconsistent process shard inventory"
        )
    return parent_manifest.checkpoint_id, parent_manifest.manifest_id


@checked
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
    """Publish this process's non-redundant addressable shards transactionally.

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
                raise ValueError(
                    "checkpoint.shard.payload must fit its admitted layout and byte limit; got payload byte-count refusal"
                )
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
        raise ValueError(
            "checkpoint.shard_descriptors must fit the manifest byte limit; got descriptor byte limit exceeded"
        )
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
        raise RepositoryCorruptionError(
            f"checkpoint.shard_metadata.{name} must be nonnegative decimal text; got {value if type(value) is str else type(value).__name__!r}"
        )
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
        raise RepositoryCorruptionError(
            "checkpoint.shard_metadata.fields must contain the declared nonempty owner and array fields; got invalid metadata fields"
        )
    if bool(metadata["parent_checkpoint_id"]) != bool(metadata["parent_manifest_id"]):
        raise RepositoryCorruptionError(
            "checkpoint.shard_metadata.parent_ids must declare checkpoint and manifest lineage together; got incomplete parent lineage"
        )
    for name in (
        "process_index",
        "device_id",
        "replica_id",
        "topology_epoch",
        "payload_offset",
    ):
        _metadata_integer(metadata, name)
    shape_value = metadata_json(metadata["global_shape"])
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
            "checkpoint.shard.payload must fit its admitted layout and byte limit; got payload byte-count refusal"
        )
    return metadata["array_path"], shape, dtype, index


def _validate_exact_coverage(
    shards: Sequence[CheckpointShard],
    limits: ArrayArchiveLimits,
    /,
) -> None:
    if len(shards) > limits.max_members:
        raise RepositoryCorruptionError(
            "checkpoint.shards must fit the shard-count limit; got shard-count limit exceeded"
        )
    by_path: dict[
        str,
        list[tuple[CheckpointShard, tuple[int, ...], np.dtype[Any], CanonicalIndex]],
    ] = {}
    total_payload_bytes = 0
    for shard in shards:
        path, shape, dtype, index = _validated_shard(shard, limits)
        if total_payload_bytes > limits.max_aggregate_bytes - shard.byte_count:
            raise RepositoryCorruptionError(
                "checkpoint.shards must fit the aggregate payload-byte limit; got aggregate payload-byte limit exceeded"
            )
        total_payload_bytes += shard.byte_count
        by_path.setdefault(path, []).append((shard, shape, dtype, index))

    for path, path_shards in by_path.items():
        shapes = {item[1] for item in path_shards}
        dtypes = {item[2].str for item in path_shards}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise RepositoryCorruptionError(
                f"checkpoint.array.layout must agree across shards; got inconsistent metadata for {path!r}"
            )
        shape = next(iter(shapes))
        _, _, expected = _admit_array_spec(list(shape), next(iter(dtypes)), limits)
        indices = [item[3] for item in path_shards]
        if len(set(indices)) != len(indices):
            raise RepositoryCorruptionError(
                f"checkpoint.array.shards must have distinct ranges; got duplicate shards for {path!r}"
            )
        for left_index, left in enumerate(indices):
            for right in indices[left_index + 1 :]:
                if _indices_overlap(left, right):
                    raise RepositoryCorruptionError(
                        f"checkpoint.array.shards must have disjoint ranges; got overlapping shards for {path!r}"
                    )
        covered = sum(
            _index_size(index, limits.max_total_array_elements) for index in indices
        )
        if covered != expected:
            raise RepositoryCorruptionError(
                f"checkpoint.array.coverage must match the global element count; got {path!r} covers {covered} of {expected} values"
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
    if parent_manifest.complete is not True:
        raise TypeError(
            "CheckpointManifest.complete must be True for checkpoint lineage; got False"
        )
    if parent_manifest.checkpoint_id == checkpoint_id:
        raise ValueError(
            "parent_manifest.checkpoint_id must differ from the successor checkpoint ID; got a self-parented checkpoint"
        )
    if (
        parent_manifest.analysis_plan_id != analysis_plan_id
        or parent_manifest.numeric_revision_id != numeric_revision_id
        or parent_manifest.execution_plan_id != execution_plan_id
    ):
        raise ValueError(
            "parent_manifest.lifecycle_ids must match the successor analysis, revision, and execution; got incompatible lifecycle ownership"
        )
    return parent_manifest.checkpoint_id, parent_manifest.manifest_id


@checked
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
        raise ValueError(
            f"checkpoint.expected_process_count must be a positive integer; got {expected_process_count if type(expected_process_count) is int else type(expected_process_count).__name__!r}"
        )
    if expected_process_count > limits.max_members:
        raise ValueError(
            f"checkpoint.expected_process_count must not exceed {limits.max_members}; got {expected_process_count}"
        )
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
            "publications.artifact_manifest must retain repository-backed parent lineage; got a parented publication without an artifact"
        )
    process_indices = tuple(publication.process_index for publication in publications_)
    if tuple(sorted(process_indices)) != tuple(range(expected_process_count)):
        raise ValueError(
            "publications.process_index must cover every process once; got missing or duplicate process indices"
        )
    shards = tuple(shard for publication in publications_ for shard in publication.shards)
    if not shards:
        raise ValueError(
            "publications.shards must contain checkpoint arrays; got no shards"
        )
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
                    "publication.artifact_metadata must match checkpoint lifecycle and parent IDs; got inconsistent publication lineage"
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
                    "checkpoint.shard_metadata.owner_ids must match the publication lifecycle and parent IDs; got inconsistent ownership or lineage"
                )
            repository_ids.add(metadata["repository_id"])
    if len(repository_ids) != 1:
        raise RepositoryCorruptionError(
            "publications.repository_id must agree across processes; got multiple repository identities"
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


@checked
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
        raise ValueError(
            f"checkpoint.expected_process_count must be a positive integer; got {expected_process_count if type(expected_process_count) is int else type(expected_process_count).__name__!r}"
        )
    if expected_process_count > limits.max_members:
        raise ValueError(
            f"checkpoint.expected_process_count must not exceed {limits.max_members}; got {expected_process_count}"
        )
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
                "artifact.metadata.checkpoint_id must match the requested checkpoint; got the wrong checkpoint_id"
            )
        if artifact_metadata["analysis_plan_id"] != analysis_plan_id:
            raise RepositoryCorruptionError(
                "artifact.metadata.analysis_plan_id must match the requested analysis; got the wrong analysis plan"
            )
        if artifact_metadata["numeric_revision_id"] != numeric_revision_id:
            raise RepositoryCorruptionError(
                "artifact.metadata.numeric_revision_id must match the requested revision; got the wrong numeric revision"
            )
        if artifact_metadata["execution_plan_id"] != execution_plan_id:
            raise RepositoryCorruptionError(
                "artifact.metadata.execution_plan_id must match the requested execution; got the wrong execution plan"
            )
        if artifact_metadata["process_index"] != str(process_index):
            raise RepositoryCorruptionError(
                "artifact.metadata.process_index must match the requested owner; got the wrong process index"
            )
        if (
            artifact_metadata["parent_checkpoint_id"] != parent_checkpoint_id
            or artifact_metadata["parent_manifest_id"] != parent_manifest_id
        ):
            raise RepositoryCorruptionError(
                "artifact.metadata.parent_ids must match the requested parent manifest; got inconsistent parent lineage"
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
    tables: dict[str, _ProcessShardTable],
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
            "checkpoint.shard_metadata.lifecycle_ids must match the manifest; got inconsistent lifecycle ownership"
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
            "checkpoint.shard_metadata.artifact_ids must match the committed repository artifact; got inconsistent artifact ownership"
        )
    table = tables.get(artifact.manifest_id)
    if table is None:
        table = _process_shard_table(repository, artifact, limits)
        tables[artifact.manifest_id] = table
    if shard.shard_fingerprint not in table.fingerprints:
        raise RepositoryCorruptionError(
            "checkpoint.shard_descriptor must belong to the committed descriptor table; got a substituted descriptor"
        )


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
            "checkpoint.shard_metadata.payload_binding must match the artifact and byte policy; got inconsistent payload binding"
        )
    digest = hashlib.sha256()
    observed = 0
    with tempfile.SpooledTemporaryFile(max_size=1_048_576, mode="w+b") as stream:
        for payload in _iter_artifact_range(
            repository,
            artifact,
            _SHARD_PAYLOADS,
            _metadata_integer(metadata, "payload_offset"),
            shard.byte_count,
        ):
            observed += len(payload)
            if observed > shard.byte_count:
                raise RepositoryCorruptionError(
                    "checkpoint.shard.byte_count must match decoded bytes; got payload byte count mismatch"
                )
            digest.update(payload)
            stream.write(payload)
        if observed != shard.byte_count:
            raise RepositoryCorruptionError(
                "checkpoint.shard.byte_count must match decoded bytes; got payload byte count mismatch"
            )
        if digest.hexdigest() != shard.payload_digest:
            raise RepositoryCorruptionError(
                "checkpoint.shard.payload_digest must match archived bytes; got payload digest mismatch"
            )
        stream.seek(0)
        try:
            dtype, shape, _ = read_npy_metadata(stream, observed, limits)
        except ArrayArchiveCorruptionError as error:
            raise RepositoryCorruptionError(
                "checkpoint.shard.npy_metadata must satisfy the array archive policy; got invalid NumPy payload metadata"
            ) from error
        if shape != expected_shape or dtype != expected_dtype:
            raise RepositoryCorruptionError(
                "checkpoint.shard.layout must match the declared shape and dtype; got inconsistent payload shape or dtype"
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
                "checkpoint.shard.npy_payload must decode without pickles under header limits; got invalid NumPy payload"
            ) from error
    if value.shape != expected_shape or value.dtype != expected_dtype:
        raise RepositoryCorruptionError(
            "checkpoint.shard.layout must remain unchanged during decoding; got payload shape or dtype changed while loading"
        )
    value = immutable_host_array(value)
    value.setflags(write=False)
    return value


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


def _checkpoint_shards_by_path(
    manifest: CheckpointManifest,
) -> dict[str, tuple[CheckpointShard, ...]]:
    """Index immutable shard selection metadata without admitting payloads."""
    grouped: dict[str, list[CheckpointShard]] = {}
    for shard in manifest.shards:
        grouped.setdefault(_metadata(shard)["array_path"], []).append(shard)
    return {path: tuple(shards) for path, shards in grouped.items()}


def _checkpoint_array_loader(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    array_path: str,
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
    tables: dict[str, _ProcessShardTable] | None = None,
    shards_by_path: Mapping[str, tuple[CheckpointShard, ...]] | None = None,
) -> tuple[tuple[int, ...], Callable[[Any], np.ndarray], tuple[CheckpointShard, ...]]:
    """Admit an array's shard coverage and prepare bounded destination loading."""

    if manifest.complete is not True:
        raise TypeError("CheckpointManifest.complete must be True for restore; got False")
    if not isinstance(array_path, str) or not array_path:
        raise ValueError(
            f"checkpoint.array_path must be nonempty text; got {array_path if type(array_path) is str else type(array_path).__name__!r}"
        )
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
        raise ValueError(
            "checkpoint.parent_ids must match the supplied parent manifest; got inconsistent restore parent lineage"
        )
    path_shards = (
        tuple(
            shard
            for shard in manifest.shards
            if _metadata(shard).get("array_path") == array_path
        )
        if shards_by_path is None
        else shards_by_path.get(array_path, ())
    )
    if not path_shards:
        raise KeyError(
            f"checkpoint.array_path must identify an archived array; got no array at path {array_path!r}"
        )
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
                "checkpoint.parent_artifacts must match the requested repository lineage; got inconsistent parent repository lineage"
            )
    _validate_exact_coverage(path_shards, limits)
    _, global_shape, dtype, _ = _validated_shard(path_shards[0], limits)

    source_records = []
    if tables is None:
        tables = {}
    # Preserve full-checkpoint integrity admission without retaining the global
    # payload on each host. Source bytes are validated one shard at a time;
    # callbacks then retain only one intersecting source and one local output.
    for shard in path_shards:
        _validate_shard_repository_binding(repository, manifest, shard, limits, tables)
        _, shape, shard_dtype, source_index = _validated_shard(shard, limits)
        if shape != global_shape or shard_dtype != dtype:
            raise RepositoryCorruptionError(
                "checkpoint.array.layout must match the admitted global shape and dtype; got inconsistent shard metadata"
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
                "checkpoint.destination_shard.coverage must include every destination element; got a destination not fully covered"
            )
        return output

    return global_shape, load_destination, path_shards


@jax.jit(donate_argnums=(0,))
def _place_checkpoint_packet(
    destination: jax.Array,
    packet: jax.Array,
    starts: tuple[int, ...],
) -> jax.Array:
    return jax.lax.dynamic_update_slice(destination, packet, starts)


def _restore_checkpoint_packets(
    path_shards: tuple[CheckpointShard, ...],
    global_shape: tuple[int, ...],
    load_destination: Callable[[Any], np.ndarray],
    sharding: Sharding,
    limits: ArrayArchiveLimits,
    maximum_host_packet_bytes: int,
) -> jax.Array:
    """Assemble admitted archive packets only into this process's destination shards.

    Each distinct addressable destination is filled in device storage from the
    archived packets that intersect it; one ingress packet is live at a time.
    No process allocates a destination owned by another process, and a global
    replica exists only where the requested sharding replicates the array.
    """
    budget = operator.index(maximum_host_packet_bytes)
    if isinstance(maximum_host_packet_bytes, bool) or budget < 1:
        raise ValueError(
            f"restore.maximum_host_packet_bytes must be a positive integer; got {maximum_host_packet_bytes!r}"
        )
    match sharding:
        case NamedSharding() | SingleDeviceSharding():
            pass
        case _:
            raise TypeError(
                "restore.sharding must be NamedSharding or SingleDeviceSharding for packet restoration; got an unsupported sharding"
            )
    records = tuple(_validated_shard(shard, limits) for shard in path_shards)
    dtype = records[0][2]
    for _, _, source_dtype, index in records:
        if _index_size(index, limits.max_array_elements) * source_dtype.itemsize > budget:
            raise ValueError(
                "restore.maximum_host_packet_bytes must cover every archived numerical packet; got an over-limit packet"
            )
    addressable = sharding.addressable_devices_indices_map(global_shape)
    destinations: dict[CanonicalIndex, list[jax.Device]] = {}
    for device, index in addressable.items():
        destinations.setdefault(_canonical_index(index, global_shape), []).append(device)
    placed: dict[jax.Device, jax.Array] = {}
    for destination, devices in destinations.items():
        first = devices[0]
        if any(index == destination for _, _, _, index in records):
            # One archived packet is exactly this destination: adopt it rather
            # than compiling a zero-fill and a redundant copy.
            result = jax.device_put(
                load_destination(
                    tuple(slice(start, stop) for start, stop, _ in destination)
                ),
                first,
            )
        else:
            result = jnp.zeros(
                tuple(stop - start for start, stop, _ in destination),
                dtype=dtype,
                device=first,
            )
            for _, _, _, index in records:
                overlap = _intersection(index, destination)
                if overlap is None:
                    continue
                _, local = overlap
                region = tuple(
                    slice(axis[0] + part.start, axis[0] + part.stop)
                    for axis, part in zip(destination, local, strict=True)
                )
                packet = jax.device_put(load_destination(region), first)
                result = _place_checkpoint_packet(
                    result, packet, tuple(part.start for part in local)
                )
                # Explicit cold-I/O packet barrier: it bounds live ingress
                # packets while donating this destination's assembly buffer.
                result.block_until_ready()
        placed[first] = result.block_until_ready()
        for device in devices[1:]:
            placed[device] = jax.device_put(result, device)
    return jax.make_array_from_single_device_arrays(
        global_shape, sharding, [placed[device] for device in addressable]
    )


@checked
def _restore_checkpoint_array(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    array_path: str,
    sharding: Sharding,
    /,
    *,
    limits: ArrayArchiveLimits,
    parent_manifest: CheckpointManifest | None,
    maximum_host_packet_bytes: int | None,
    tables: dict[str, _ProcessShardTable],
    shards_by_path: Mapping[str, tuple[CheckpointShard, ...]] | None = None,
) -> jax.Array:
    """Restore an array while retaining this restore's admitted artifact tables."""
    global_shape, load_destination, path_shards = _checkpoint_array_loader(
        repository,
        manifest,
        array_path,
        limits=limits,
        parent_manifest=parent_manifest,
        tables=tables,
        shards_by_path=shards_by_path,
    )
    if maximum_host_packet_bytes is not None:
        return _restore_checkpoint_packets(
            path_shards,
            global_shape,
            load_destination,
            sharding,
            limits,
            maximum_host_packet_bytes,
        )
    return jax.make_array_from_callback(global_shape, sharding, load_destination)


@checked
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

    With ``maximum_host_packet_bytes``, restore archived rectangles
    individually and assemble only on devices. The limit bounds each decoded
    numerical packet, not aggregate process peak memory or archive codec bytes.
    This avoids a global NumPy destination when owner count shrinks.
    """
    return _restore_checkpoint_array(
        repository,
        manifest,
        array_path,
        sharding,
        limits=limits,
        parent_manifest=parent_manifest,
        maximum_host_packet_bytes=maximum_host_packet_bytes,
        tables={},
    )


def _meshing_source_arrays(
    state: MeshingCheckpointState,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> dict[str, jax.Array | np.ndarray]:

    if state.source_recipe_json is None:
        return {}
    if state.source_closure is None:
        raise ValueError(
            "state.source_closure must retain the source records for publication; got None"
        )
    bank = state.source_logical_arrays
    if bank is None or bank.source_validation is None:
        raise ValueError(
            "state.source_logical_arrays must retain prior collective source admission; "
            f"got {type(bank).__name__}"
        )
    bank.source_validation.require_binding(bank, bank.content_digest)
    bindings = dict(state.source_array_bindings)
    binding = authenticate_meshing_source_publication(
        state.source_closure,
        state.source_recipe_json,
        state.source_content_digest,
        owner_index=state.source_owner_index,
        array_bindings=bindings,
        logical_arrays=state.source_logical_arrays,
        logical_array_names=state.source_logical_array_names,
        logical_content_digest=state.source_logical_content_digest,
        limits=limits,
        maximum_chunk_bytes=bank.maximum_chunk_bytes,
    )
    # Immediate metadata reads need no Equinox bound method over the source DAG.
    validate_meshing_source_checkpoint_references(
        state.source_closure, MeshingCheckpointState.to_payload(state)
    )
    if binding.array_values is None:
        raise ValueError(
            "source.publication must retain authenticated live array values; got None"
        )
    local = {
        source_array_name(name, state.source_owner_index, bindings): value
        for name, value in binding.array_values.items()
        if name not in bindings
    }
    if binding.logical_arrays is not None:
        local.update(
            {
                source_logical_array_name(name): value
                for name, value in binding.logical_arrays.items()
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
    if set(arrays) != {name for name, _ in state.array_roles}:
        raise ValueError(
            "state.array_roles must match checkpoint array names; "
            f"got {tuple(sorted(arrays))}"
        )
    for name, role in state.array_roles:
        array = arrays[name]
        if role == "source-record" and isinstance(array, np.ndarray):
            continue
        if not isinstance(array, jax.Array):
            raise TypeError(
                f"checkpoint.array.backend must be JAX for scientific roles; got {type(array).__name__} for {name!r}"
            )
        if role == "entity-ids" and (
            array.ndim != 1 or not np.issubdtype(array.dtype, np.integer)
        ):
            raise ValueError(
                f"checkpoint.entity_ids must be logical integer vectors; got invalid rank or dtype for {name!r}"
            )


@checked
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
    The CellMesh storage owner supplies logical coordinates, stable
    IDs and global-ID incidences. Solver/history and accepted evidence owners
    supply the other declared leaves. Source-record leaves are taken from
    ``state.source_closure``; callers cannot replace them in ``arrays``. JAX
    numerical values cross to host only as addressable shards. Authored native
    NumPy coefficients are snapshotted directly on their publishing process.
    """

    if set(arrays) != {
        name for name, role in state.array_roles if role != "source-record"
    }:
        raise ValueError(
            "publication.arrays must cover every non-source scientific role; got missing or undeclared accepted arrays"
        )
    source_arrays = _meshing_source_arrays(state, limits=limits)
    scientific_arrays = {**arrays, **source_arrays}
    _validate_meshing_arrays(state, scientific_arrays)
    payload = canonical_json(MeshingCheckpointState.to_payload(state)).encode("utf-8")
    if len(payload) > limits.max_manifest_bytes:
        raise ValueError(
            "checkpoint_state.metadata must fit the manifest byte limit; got metadata byte limit exceeded"
        )
    if (
        state.source_recipe_json is not None
        and state.source_owner_index != jax.process_index()
    ):
        raise ValueError(
            "state.source_owner_index must identify the publishing process; "
            f"got {state.source_owner_index} for process {jax.process_index()}"
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
            MeshingCheckpointState.to_payload(state), limits=limits
        ),
    )


def _meshing_record_path(process_index: int, /) -> str:
    return jax.tree_util.keystr((jax.tree_util.DictKey(f"record-owner-{process_index}"),))


class _MeshingStateRecord(StrictModule):
    payload: dict[str, Any]
    state: MeshingCheckpointState


def _decode_meshing_state_record(
    record: np.ndarray,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> _MeshingStateRecord:
    try:
        text = record.tobytes().decode("utf-8")
        payload = metadata_json(text, maximum_bytes=limits.max_manifest_bytes)
        if not isinstance(payload, dict) or canonical_json(payload) != text:
            raise ValueError(
                "checkpoint.record must use deterministic JSON encoding; got different bytes"
            )
        state = MeshingCheckpointState.from_payload(payload, limits=limits)
    except (TypeError, ValueError, UnicodeError) as error:
        raise RepositoryCorruptionError(
            "checkpoint.record must decode into the declared scientific state; got invalid checkpoint metadata"
        ) from error
    return _MeshingStateRecord(payload, state)


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
            "process.shards must contain one metadata record; "
            f"got {len(matching)} records for owner {process_index}"
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
            "checkpoint.owner_record.layout must be a bounded uint8 vector owned by its process; got invalid owner record layout"
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
    return _decode_meshing_state_record(record, limits=limits).state


def _validate_meshing_source_manifest_layouts(
    states: Sequence[MeshingCheckpointState],
    shards: Sequence[CheckpointShard],
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:

    specifications = {}
    for shard in shards:
        path, shape, dtype, _ = _validated_shard(shard, limits)
        specifications[path] = (shape, dtype.str)
    for state in states:
        if state.source_recipe_json is None:
            continue
        recipe = meshing_source_recipe(state.source_recipe_json, limits=limits)
        for entry in model_recipe_array_inventory(
            recipe, prefix=MESHING_SOURCE_PREFIX, limits=limits
        ):
            path = _meshing_array_path(state.source_array_name(entry.name))
            if specifications[path] != (entry.shape, entry.dtype):
                raise RepositoryCorruptionError(
                    f"source_owner.{state.source_owner_index}.array_layout must match its recipe; got inconsistent checkpoint shape or dtype for {entry.path}"
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
                "publications.meshing_state must acknowledge every source owner; "
                f"got no state for owner {publication.process_index}"
            )
        shard = _meshing_record_shard(
            publication.shards, publication.process_index, limits=limits
        )
        if shard is None:
            raise RepositoryCorruptionError(
                "manifest.source_owner_records must include each acknowledged owner record; got a missing acknowledged record"
            )
        record = canonical_json(state.to_payload()).encode("utf-8")
        payload = _array_payload(np.frombuffer(record, dtype=np.uint8))
        if len(record) > limits.max_manifest_bytes or (
            hashlib.sha256(payload).hexdigest() != shard.payload_digest
            or len(payload) != shard.byte_count
        ):
            raise RepositoryCorruptionError(
                "publications.meshing_state must match its acknowledged payload; got different bytes"
            )
        states.append(state)
    source_flags = tuple(state.source_recipe_json is not None for state in states)
    if any(source_flags) and not all(source_flags):
        raise RepositoryCorruptionError(
            "manifest.source_owner_records must either all bind sources or all omit sources; got mixed source and no-source records"
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
            "manifest.scientific_inventory must match every declared array and epoch; got incomplete scientific inventory or epoch"
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
) -> _MeshingStateRecord:
    shards = tuple(
        shard
        for shard in manifest.shards
        if _metadata(shard)["array_path"] == record_path
    )
    if not shards:
        raise RepositoryCorruptionError(
            "manifest.scientific_record must identify a published owner record; got no scientific record"
        )
    _, shape, dtype, _ = _validated_shard(shards[0], limits)
    if (
        len(shape) != 1
        or shape[0] > limits.max_manifest_bytes
        or dtype != np.dtype(np.uint8)
    ):
        raise RepositoryCorruptionError(
            "checkpoint.record.layout must be a bounded uint8 vector; got invalid record layout"
        )
    record = restore_global_array_from_checkpoint(
        repository,
        manifest,
        record_path,
        SingleDeviceSharding(jax.local_devices()[0]),
        limits=limits,
        parent_manifest=parent_manifest,
    )
    return _decode_meshing_state_record(np.asarray(record), limits=limits)


def _meshing_checkpoint_record_paths(manifest: CheckpointManifest, /) -> tuple[str, ...]:
    paths = {_metadata(shard)["array_path"] for shard in manifest.shards}
    processes = {
        _metadata_integer(_metadata(shard), "process_index") for shard in manifest.shards
    }
    if not processes or max(processes) >= len(manifest.shards):
        raise RepositoryCorruptionError(
            "manifest.source_owner_count must fit its shard-record coverage; got invalid owner count or insufficient record coverage"
        )
    owner_paths = tuple(
        _meshing_record_path(index) for index in range(max(processes) + 1)
    )
    if not owner_paths or any(path not in paths for path in owner_paths):
        raise RepositoryCorruptionError(
            "manifest.source_records must cover every source owner; got missing owner records"
        )
    for owner_index, path in enumerate(owner_paths):
        if any(
            _metadata_integer(_metadata(shard), "process_index") != owner_index
            for shard in manifest.shards
            if _metadata(shard)["array_path"] == path
        ):
            raise RepositoryCorruptionError(
                "manifest.source_record.process_index must match the named record owner; got invalid source record ownership"
            )
    return owner_paths


@checked
def read_meshing_checkpoint_states(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
) -> tuple[MeshingCheckpointState, ...]:
    """Read bounded owner recipes without loading scientific leaves.

    Source records remain absent until full restore. The returned declarations
    let cold callers prepare placement from the accepted archive alone; callers
    still bind the archive/record identities to their accepted composition.
    """
    if manifest.complete is not True:
        raise TypeError("CheckpointManifest.complete must be True for restore; got False")
    _validate_exact_coverage(manifest.shards, limits)
    return tuple(
        _read_meshing_checkpoint_record(
            repository, manifest, path, limits=limits, parent_manifest=parent_manifest
        ).state
        for path in _meshing_checkpoint_record_paths(manifest)
    )


def _validate_meshing_owner_states(
    states: Sequence[MeshingCheckpointState],
    expected_state_id: str | Mapping[int, str],
    source_owner_index: int,
    /,
) -> None:
    match expected_state_id:
        case str():
            if len(states) != 1 and any(
                state.source_recipe_json is not None for state in states
            ):
                raise ValueError(
                    "restore.expected_state_id must name every source owner for multi-owner restart; "
                    f"got one ID for {len(states)} owners"
                )
            expected_ids = dict.fromkeys(range(len(states)), expected_state_id)
        case Mapping():
            expected_ids = dict(expected_state_id)
            if any(type(index) is not int for index in expected_ids) or set(
                expected_ids
            ) != set(range(len(states))):
                raise ValueError(
                    f"restore.expected_state_id must cover source owner indices once; got indices {tuple(expected_ids)}"
                )
        case _:
            raise TypeError(
                f"restore.expected_state_id must be a state ID or owner mapping; got {type(expected_state_id).__name__}"
            )
    if type(source_owner_index) is not int or not 0 <= source_owner_index < len(states):
        raise ValueError(
            f"restore.source_owner_index must select a source owner; got {source_owner_index if type(source_owner_index) is int else type(source_owner_index).__name__!r} for {len(states)} owners"
        )
    for index, state in enumerate(states):
        if state.checkpoint_state_id != expected_ids[index]:
            raise RepositoryCorruptionError(
                "checkpoint_state.checkpoint_state_id must match its accepted scientific binding; got scientific binding changed"
            )
        if state.source_recipe_json is not None and state.source_owner_index != index:
            raise RepositoryCorruptionError(
                "state.source_owner_index must match its checkpoint record owner; "
                f"got {state.source_owner_index} for owner {index}"
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
            "source_owner_states.shared_scientific_state must agree across owners; got different roles, epoch, or logical content"
        )


class MeshingCheckpointRestore(StrictModule):
    """Return the admitted state, restored arrays, and topology decision."""

    state: MeshingCheckpointState
    arrays: dict[str, jax.Array | np.ndarray]
    admission: RestartAdmission


@dataclass(frozen=True, slots=True)
class _MeshingRestorePlan:
    records: tuple[_MeshingStateRecord, ...]
    recipes: tuple[dict[str, Any] | None, ...]
    inventories: tuple[tuple[ModelRecipeArray, ...], ...]
    names: frozenset[str]
    host_names: frozenset[str]
    decoded: frozenset[int]


@dataclass(frozen=True, slots=True)
class _RestoredMeshingArrays:
    arrays: dict[str, jax.Array | np.ndarray]
    logical_arrays: AdmittedMeshingSourceArrays | None
    source_chunk_bytes: int


@dataclass(frozen=True, slots=True)
class _RestoredMeshingSources:
    states: tuple[MeshingCheckpointState, ...]
    logical_arrays: AdmittedMeshingSourceArrays | None


def _decoded_meshing_owners(
    decoded_source_owners: Sequence[int] | None,
    source_owner_index: int,
    owner_count: int,
    /,
) -> frozenset[int]:
    if decoded_source_owners is None:
        return frozenset(range(owner_count))
    owners = tuple(decoded_source_owners)
    if (
        not owners
        or any(type(owner) is not int or not 0 <= owner < owner_count for owner in owners)
        or len(set(owners)) != len(owners)
        or source_owner_index not in owners
    ):
        raise ValueError(
            "restore.decoded_source_owners must name distinct archived owners including source_owner_index; "
            f"got {owners!r} for {owner_count} owners and selected owner {source_owner_index!r}"
        )
    return frozenset(owners)


def _prepare_meshing_restore(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    shardings: Mapping[str, Sharding],
    expected_state_id: str | Mapping[int, str],
    source_owner_index: int,
    limits: ArrayArchiveLimits,
    parent_manifest: CheckpointManifest | None,
    decoded_source_owners: Sequence[int] | None,
) -> _MeshingRestorePlan:
    if manifest.complete is not True:
        raise TypeError("CheckpointManifest.complete must be True for restore; got False")
    _validate_exact_coverage(manifest.shards, limits)
    record_paths = _meshing_checkpoint_record_paths(manifest)
    records = tuple(
        _read_meshing_checkpoint_record(
            repository, manifest, path, limits=limits, parent_manifest=parent_manifest
        )
        for path in record_paths
    )
    states = tuple(record.state for record in records)
    _validate_meshing_owner_states(states, expected_state_id, source_owner_index)
    recipes = tuple(
        None
        if state.source_recipe_json is None
        else meshing_source_recipe(state.source_recipe_json, limits=limits)
        for state in states
    )
    inventories = tuple(
        ()
        if recipe is None
        else model_recipe_array_inventory(
            recipe, prefix=MESHING_SOURCE_PREFIX, limits=limits
        )
        for recipe in recipes
    )
    decoded = _decoded_meshing_owners(
        decoded_source_owners, source_owner_index, len(states)
    )
    archived_names = frozenset(name for state in states for name, _ in state.array_roles)
    # Every owner's record and identity is admitted above; only the selected
    # owners' source records are restored and decoded by this process.
    names = frozenset(
        name for owner in decoded for name, _ in states[owner].array_roles
    ) | frozenset(
        source_logical_array_name(name) for name in states[0].source_logical_array_names
    )
    host_names = frozenset(
        states[owner].source_array_name(entry.name)
        for owner in decoded
        for entry in inventories[owner]
        if entry.backend == "numpy"
    )
    if set(shardings) != names - host_names:
        raise ValueError(
            "restore.destination_shardings must cover every logical JAX scientific array; got missing or undeclared destination shardings"
        )
    expected_paths = set(record_paths) | {
        _meshing_array_path(name) for name in archived_names
    }
    if {
        _metadata(shard)["array_path"] for shard in manifest.shards
    } != expected_paths or any(
        _metadata_integer(_metadata(shard), "topology_epoch") != states[0].mesh_epoch
        for shard in manifest.shards
    ):
        raise RepositoryCorruptionError(
            "manifest.array_epoch_binding must match the restored scientific state; got invalid array or epoch binding"
        )
    return _MeshingRestorePlan(
        records, recipes, inventories, names, host_names, decoded
    )


def _load_meshing_arrays(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    shardings: Mapping[str, Sharding],
    plan: _MeshingRestorePlan,
    limits: ArrayArchiveLimits,
    parent_manifest: CheckpointManifest | None,
    maximum_host_packet_bytes: int | None,
) -> dict[str, jax.Array | np.ndarray]:
    """Admit and place this process's destinations; no collective is entered."""
    # Each committed artifact table is authenticated once for this restore.
    # Shard membership, lifecycle ownership and every payload remain checked.
    tables: dict[str, _ProcessShardTable] = {}
    # Index only the immutable manifest's selection metadata. Every selected
    # shard still undergoes coverage, repository binding and payload admission.
    shards_by_path = _checkpoint_shards_by_path(manifest)
    arrays: dict[str, jax.Array | np.ndarray] = {
        name: _restore_checkpoint_array(
            repository,
            manifest,
            _meshing_array_path(name),
            shardings[name],
            limits=limits,
            parent_manifest=parent_manifest,
            maximum_host_packet_bytes=maximum_host_packet_bytes,
            tables=tables,
            shards_by_path=shards_by_path,
        )
        for name in sorted(plan.names - plan.host_names)
    }
    for name in sorted(plan.host_names):
        shape, load_host, _ = _checkpoint_array_loader(
            repository,
            manifest,
            _meshing_array_path(name),
            limits=limits,
            parent_manifest=parent_manifest,
            tables=tables,
            shards_by_path=shards_by_path,
        )
        host = immutable_host_array(load_host(tuple(slice(0, size) for size in shape)))
        host.setflags(write=False)
        arrays[name] = host
    return arrays


def _admit_meshing_arrays(
    arrays: dict[str, jax.Array | np.ndarray],
    plan: _MeshingRestorePlan,
    maximum_host_packet_bytes: int | None,
) -> _RestoredMeshingArrays:
    logical_arrays: dict[str, jax.Array] = {}
    with meshing_source_local_restore_phase():
        for name in plan.records[0].state.source_logical_array_names:
            value = arrays[source_logical_array_name(name)]
            match value:
                case jax.Array():
                    logical_arrays[name] = value
                case _:
                    raise RepositoryCorruptionError(
                        f"source_binding.logical_arrays.backend must remain JAX during restore; got {type(value).__name__}"
                    )
        source_chunk_bytes = (
            1 << 20 if maximum_host_packet_bytes is None else maximum_host_packet_bytes
        )
    admitted = admit_meshing_source_logical_arrays(
        logical_arrays, maximum_chunk_bytes=source_chunk_bytes
    )
    return _RestoredMeshingArrays(arrays, admitted, source_chunk_bytes)


def _restore_meshing_sources(
    plan: _MeshingRestorePlan,
    restored_arrays: _RestoredMeshingArrays,
    limits: ArrayArchiveLimits,
) -> _RestoredMeshingSources:
    arrays = restored_arrays.arrays
    source_chunk_bytes = restored_arrays.source_chunk_bytes
    admitted = restored_arrays.logical_arrays
    decoded_closures: list[Mapping[str, Any] | None] = []
    # Recipe reconstruction is owner-local. A peer must not start common-epoch
    # replay while another process has refused its typed source root.
    with meshing_source_local_restore_phase():
        for owner, (record, recipe, inventory) in enumerate(
            zip(plan.records, plan.recipes, plan.inventories, strict=True)
        ):
            if recipe is None or owner not in plan.decoded:
                decoded_closures.append(None)
                continue
            source_arrays = {
                entry.name: arrays[record.state.source_array_name(entry.name)]
                for entry in inventory
            }
            try:
                closure = model_from_logical_array_recipe(
                    recipe, source_arrays, prefix=MESHING_SOURCE_PREFIX, limits=limits
                )
                if not isinstance(closure, Mapping):
                    raise TypeError(
                        "source_binding.closure must be an immutable scientific mapping; got an incompatible decoded closure"
                    )
                decoded_closures.append(closure)
            except (TypeError, ValueError) as error:
                raise RepositoryCorruptionError(
                    "source_binding.closure must reconstruct and validate its retained scientific fields; got an invalid source closure"
                ) from error
        source_roots = tuple(
            closure for closure in decoded_closures if closure is not None
        )
        if source_roots and admitted is None:
            admitted = AdmittedMeshingSourceArrays(
                {}, maximum_chunk_bytes=source_chunk_bytes
            )
    if jax.process_count() > 1:
        roots_present = np.asarray(
            multihost_utils.process_allgather(
                np.int32(bool(source_roots)), tiled=False
            )
        ).reshape(-1)
        if np.any(roots_present != roots_present[0]):
            raise ValueError(
                "restore.collective_admission must agree whether source replay is required; "
                "got decoded source roots on only some processes"
            )
    if source_roots:
        assert admitted is not None
        proof = validate_collective_mesh_source_epochs(
            source_roots,
            source_logical_arrays=admitted,
            source_logical_content_digest=admitted.content_digest,
            maximum_chunk_bytes=source_chunk_bytes,
            limits=limits,
        )
    # Common replay retains actual reconstructed numerical owners, their full
    # content authentication, and clock scans before strictly local source guards.
    with meshing_source_local_restore_phase():
        if source_roots:
            assert admitted is not None
            admitted = AdmittedMeshingSourceArrays(
                admitted, maximum_chunk_bytes=source_chunk_bytes, source_validation=proof
            )
        restored_states: list[MeshingCheckpointState] = []
        for record, closure in zip(plan.records, decoded_closures, strict=True):
            if closure is None:
                restored_states.append(record.state)
                continue
            try:
                restored = MeshingCheckpointState.from_payload(
                    record.payload,
                    source_closure=closure,
                    source_logical_arrays=admitted,
                    limits=limits,
                    maximum_source_chunk_bytes=source_chunk_bytes,
                )
            except (TypeError, ValueError) as error:
                raise RepositoryCorruptionError(
                    "source_binding.closure must reconstruct and validate its retained scientific fields; got an invalid source closure"
                ) from error
            _validate_meshing_arrays(
                restored, {name: arrays[name] for name, _ in restored.array_roles}
            )
            restored_states.append(restored)
        result = _RestoredMeshingSources(tuple(restored_states), admitted)
    return result


@checked
def meshing_checkpoint_array_layouts(
    manifest: CheckpointManifest,
    names: Sequence[str],
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Return archived global shapes and dtypes from shard metadata alone.

    Destination placement is chosen from these logical layouts before any
    payload is read; every listed name must identify an archived array.
    """
    shards = _checkpoint_shards_by_path(manifest)
    layouts: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
    for name in names:
        selected = shards.get(_meshing_array_path(name))
        if not selected:
            raise KeyError(
                f"checkpoint.array_path must identify an archived array; got no array named {name!r}"
            )
        _, shape, dtype, _ = _validated_shard(selected[0], limits)
        layouts[name] = (shape, dtype)
    return layouts


@checked
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
    decoded_source_owners: Sequence[int] | None = None,
) -> MeshingCheckpointRestore:
    """Restore source-owner records and logical arrays into new placement.

    The owner bank retains historical typed source records, local result shapes,
    and collective evidence. Repreparation explicitly selects a source owner;
    source ranks never determine successor scientific IDs or local slots.
    Multi-owner archives require the expected identity mapping for every owner.
    ``decoded_source_owners`` limits restored and decoded source records to the
    named owners, which must include ``source_owner_index``; every owner's
    record identity and the shared logical bank are still admitted. ``None``
    decodes every owner. Local decode and owner-validation phases agree refusal
    before common-epoch replay or acceptance; a failed invocation retains no
    source reconstruction or publication admission for a subsequent restore.
    """
    with meshing_source_local_restore_phase():
        admission = admit_topology_restart(relation, policy)
        if not admission.admitted:
            raise ValueError(
                f"restore_plan.admission must admit the topology restart; got {admission.reason}"
            )
        plan = _prepare_meshing_restore(
            repository,
            manifest,
            shardings,
            expected_state_id,
            source_owner_index,
            limits,
            parent_manifest,
            decoded_source_owners,
        )
        loaded = _load_meshing_arrays(
            repository,
            manifest,
            shardings,
            plan,
            limits,
            parent_manifest,
            maximum_host_packet_bytes,
        )
    # Source-bank admission has its own collective digest; the preceding phase
    # admits every local record and destination before that work begins.
    restored_arrays = _admit_meshing_arrays(loaded, plan, maximum_host_packet_bytes)
    with meshing_source_restore_admission(), meshing_source_owner_reconstruction_scope():
        restored_sources = _restore_meshing_sources(plan, restored_arrays, limits)
        with meshing_source_local_restore_phase():
            selected = restored_sources.states[source_owner_index]
            if selected.source_closure is not None:
                selected = MeshingCheckpointState.from_payload(
                    plan.records[source_owner_index].payload,
                    source_closure=selected.source_closure,
                    source_logical_arrays=selected.source_logical_arrays,
                    source_owner_states=restored_sources.states,
                    limits=limits,
                    maximum_source_chunk_bytes=restored_arrays.source_chunk_bytes,
                )
            else:
                _validate_meshing_arrays(selected, restored_arrays.arrays)
            result = MeshingCheckpointRestore(selected, restored_arrays.arrays, admission)
        return result


__all__ = (
    "AddressableCheckpointShard",
    "MeshingCheckpointRestore",
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
