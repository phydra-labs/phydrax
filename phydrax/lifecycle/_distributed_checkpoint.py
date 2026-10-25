#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Addressable-shard checkpoint publication and topology-neutral restore."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Sharding, SingleDeviceSharding

from .._array_archive import (
    _read_npy_metadata,
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
)
from .._fingerprint import canonical_fingerprint, canonical_json
from .._jax_config import default_device
from ._chunk_repository import (
    _MAX_METADATA_VALUE_BYTES,
    ArtifactManifest,
    ArtifactRepository,
    ChunkEncoding,
    ChunkRecord,
    RepositoryCorruptionError,
    RepositoryError,
    RepositoryTransaction,
)
from ._models import CheckpointManifest, CheckpointShard
from ._repository import ObjectNotFoundError


CanonicalIndex = tuple[tuple[int, int, int], ...]
_DISTRIBUTED_CHECKPOINT_LIMITS = DEFAULT_ARRAY_ARCHIVE_LIMITS
_NO_PARENT_ID = "none"
_METADATA_JSON_NESTING = 16
# Each process artifact stores its ordered shard descriptors as canonical JSON
# in ordinary authenticated chunks under this reserved logical name; the
# artifact metadata binds the SHA-256 of those bytes.
_INVENTORY_LOGICAL_NAME = "checkpoint-shard-inventory"
_INVENTORY_DIGEST_KEY = "shard_inventory_digest"
_SHARD_METADATA_KEYS = frozenset(
    {
        "artifact_id",
        "checkpoint_id",
        "analysis_plan_id",
        "numeric_revision_id",
        "execution_plan_id",
        "logical_name",
        "array_path",
        "global_shape",
        "dtype",
        "index",
        "process_index",
        "source_process_index",
        "device_id",
        "replica_id",
        "topology_epoch",
        "repository_id",
        "parent_checkpoint_id",
        "parent_manifest_id",
    }
)
_DESCRIPTOR_KEYS = _SHARD_METADATA_KEYS | {
    "shard_id",
    "payload_digest",
    "byte_count",
    "layout_id",
}
# Descriptor fields that must equal the owning artifact's metadata exactly.
_DESCRIPTOR_OWNERSHIP_KEYS = (
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
_PROCESS_METADATA_KEYS = frozenset(
    {
        "checkpoint_id",
        "analysis_plan_id",
        "numeric_revision_id",
        "execution_plan_id",
        "process_index",
        "topology_epoch",
        "shard_count",
        _INVENTORY_DIGEST_KEY,
        "repository_id",
        "parent_checkpoint_id",
        "parent_manifest_id",
    }
)


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


def _index_payload(index: CanonicalIndex) -> str:
    return canonical_json([list(entry) for entry in index])


def _unique_json_object(pairs: list[tuple[str, Any]], /) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"duplicate checkpoint metadata member {name!r}")
        value[name] = item
    return value


def _bounded_json(
    text: str,
    maximum_nesting: int,
    /,
    *,
    maximum_items: int | None = None,
) -> Any:
    """Decode JSON after a bounded scan, refusing duplicate object members.

    The scan enforces ``maximum_nesting`` before decoding. With
    ``maximum_items`` the root must be an array and its top-level elements of
    any kind are counted before decoding, so an inventory cannot allocate more
    entries than its admitted count.
    """
    depth = 0
    in_string = False
    escaped = False
    in_item = False
    items = 0
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if maximum_items is not None and character not in " \t\r\n":
            if depth == 0 and character != "[":
                raise RepositoryCorruptionError(
                    "Checkpoint shard inventory root must be a JSON array."
                )
            if depth == 1:
                if character == ",":
                    in_item = False
                elif character != "]" and not in_item:
                    in_item = True
                    items += 1
                    if items > maximum_items:
                        raise RepositoryCorruptionError(
                            "Checkpoint shard inventory exceeds its admitted entry count."
                        )
        if character == '"':
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
        return json.loads(text, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise RepositoryCorruptionError("Checkpoint metadata JSON is invalid.") from error


def _metadata_json(value: str, /) -> Any:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > _MAX_METADATA_VALUE_BYTES
    ):
        raise RepositoryCorruptionError("Checkpoint metadata exceeds its byte limit.")
    return _bounded_json(value, _METADATA_JSON_NESTING)


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


def _any_overlap(indices: Sequence[CanonicalIndex], /) -> bool:
    """Whether two boxes overlap, by a leading-axis sweep over sorted starts.

    Only boxes whose leading intervals are simultaneously open are compared, so
    row-blocked region records cost ``O(n log n)`` instead of all pairs.
    """
    if len(indices) <= 1:
        return False
    if not indices[0]:
        return True
    ordered = sorted(indices, key=lambda index: (index[0][0], index[0][1]))
    open_boxes: list[CanonicalIndex] = []
    for box in ordered:
        start = box[0][0]
        open_boxes = [other for other in open_boxes if other[0][1] > start]
        if any(_indices_overlap(box, other) for other in open_boxes):
            return True
        if box[0][1] > start:
            open_boxes.append(box)
    return False


def _array_payload(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    payload = np.ascontiguousarray(array) if array.ndim else array
    buffer = io.BytesIO()
    np.save(buffer, payload, allow_pickle=False)
    return buffer.getvalue()


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


def _local_checkpoint_buffers(
    leaf: jax.Array,
    indices: Mapping[jax.Device, CanonicalIndex],
) -> Iterator[tuple[jax.Device, int, jax.Array]]:
    local_devices = leaf.sharding.addressable_devices
    if leaf.is_fully_replicated and len(local_devices) == 1:
        # JAX 0.11.2 can expose a globally sharded self-view through
        # addressable_shards when one local buffer holds a complete replica.
        # Its public addressable_data API supplies the correct local view.
        # Global map order preserves JAX's replica numbering.
        for replica_id, device in enumerate(indices):
            if device in local_devices:
                yield device, replica_id, leaf.addressable_data(0)
                return
        raise ValueError("A local checkpoint replica has no global device index.")
    for shard in leaf.addressable_shards:
        yield shard.device, shard.replica_id, shard.data


def _region_rows(
    index: CanonicalIndex, dtype: np.dtype[Any], region_bytes: int | None, /
) -> tuple[tuple[int, int], ...]:
    """Canonical leading-axis row blocks of one shard (shard-relative bounds).

    ``region_bytes=None`` keeps the whole shard as one record. Otherwise each
    record holds at most ``region_bytes`` raw bytes and at least one row, so a
    record's digest authenticates exactly the rows a restore reads.
    """
    if not index:
        return ((0, 1),)
    rows = index[0][1] - index[0][0]
    row_bytes = dtype.itemsize
    for start, stop, _ in index[1:]:
        row_bytes *= stop - start
    if region_bytes is None or rows <= 1 or row_bytes == 0:
        return ((0, rows),)
    per_region = max(1, region_bytes // row_bytes)
    return tuple(
        (start, min(start + per_region, rows)) for start in range(0, rows, per_region)
    )


def snapshot_addressable_arrays(
    tree: Any,
    /,
    *,
    topology_epoch: int = 0,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    region_bytes: int | None = None,
) -> tuple[AddressableCheckpointShard, ...]:
    """Copy only canonical locally addressable JAX shards to host memory.

    With ``region_bytes`` every canonical shard is split into leading-axis
    region records of at most that many raw bytes (at least one row each).
    Every region is an independent authenticated record with its own index,
    layout identity and payload digest, so overlap restore reads and verifies
    only the regions that intersect a destination.
    """

    if type(topology_epoch) is not int or topology_epoch < 0:
        raise ValueError("topology_epoch must be a non-negative integer")
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance")
    if region_bytes is not None and (type(region_bytes) is not int or region_bytes <= 0):
        raise ValueError("region_bytes must be a positive integer or None")
    pending: list[tuple[str, jax.Array, jax.Device, int, jax.Array, CanonicalIndex]] = []
    total_bytes = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not isinstance(leaf, jax.Array):
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
        indices = {
            device: _canonical_index(index, shape)
            for device, index in leaf.sharding.devices_indices_map(shape).items()
        }
        ownership: dict[CanonicalIndex, tuple[int, int]] = {}
        for device, normalized in indices.items():
            key = (int(device.process_index), int(device.id))
            previous = ownership.get(normalized)
            if previous is None or key < previous:
                ownership[normalized] = key
        for device, replica_id, data in _local_checkpoint_buffers(leaf, indices):
            normalized = indices[device]
            device_key = (int(device.process_index), int(device.id))
            if ownership[normalized] != device_key:
                continue
            if len(pending) >= limits.max_members:
                raise ValueError("Checkpoint exceeds the shard-count limit.")
            data.copy_to_host_async()
            pending.append((array_path, leaf, device, replica_id, data, normalized))

    snapshots: list[AddressableCheckpointShard] = []
    for array_path, leaf, device, replica_id, data, index in pending:
        payload = np.asarray(data)
        local_shape = tuple(stop - start for start, stop, _ in index)
        dtype = np.dtype(leaf.dtype)
        if (
            payload.shape != local_shape
            or payload.dtype != dtype
            or payload.nbytes > limits.max_member_bytes
        ):
            raise ValueError("Addressable checkpoint shard payload is inconsistent.")
        for row_start, row_stop in _region_rows(index, dtype, region_bytes):
            if len(snapshots) >= limits.max_members:
                raise ValueError("Checkpoint exceeds the shard-count limit.")
            if index:
                origin = index[0][0]
                region = (
                    (origin + row_start, origin + row_stop, 1),
                    *index[1:],
                )
                region_payload = payload[row_start:row_stop]
            else:
                region = index
                region_payload = payload
            layout_id = canonical_fingerprint(
                {
                    "array_path": array_path,
                    "global_shape": list(leaf.shape),
                    "dtype": dtype.str,
                    "index": [list(entry) for entry in region],
                    "process_index": int(device.process_index),
                    "device_id": int(device.id),
                    "replica_id": int(replica_id),
                    "topology_epoch": topology_epoch,
                }
            )
            snapshots.append(
                AddressableCheckpointShard(
                    array_path=array_path,
                    global_shape=tuple(leaf.shape),
                    dtype=dtype.str,
                    index=region,
                    process_index=int(device.process_index),
                    device_id=int(device.id),
                    replica_id=int(replica_id),
                    payload=region_payload,
                    layout_id=layout_id,
                )
            )
    return tuple(snapshots)


def _write_stream(
    repository: ArtifactRepository,
    transaction: RepositoryTransaction,
    logical_name: str,
    payload: bytes,
    /,
    *,
    encoding: ChunkEncoding = "identity",
) -> list[ChunkRecord]:
    """Write one logical payload as ordered chunks of the repository bound."""
    view = memoryview(payload)
    records: list[ChunkRecord] = []
    offset = 0
    while offset < len(view):
        stop = min(offset + repository.maximum_chunk_bytes, len(view))
        records.append(
            repository.write_chunk(
                transaction,
                logical_name,
                len(records),
                offset,
                view[offset:stop],
                encoding=encoding,
            )
        )
        offset = stop
    return records


def _decimal_metadata(
    value: str,
    message: str,
    /,
    *,
    maximum: int | None = None,
    excess: str | None = None,
) -> int:
    """Convert canonical decimal metadata, refusing it as repository corruption.

    With ``maximum`` the digit count is compared before conversion. Without
    it, conversion is bounded by the interpreter's integer-string limit, whose
    refusal is reported as corruption rather than leaking ``ValueError``.
    """
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise RepositoryCorruptionError(message)
    if maximum is not None and len(value.lstrip("0")) > len(str(maximum)):
        raise RepositoryCorruptionError(excess or message)
    try:
        number = int(value)
    except ValueError as error:
        raise RepositoryCorruptionError(message) from error
    if maximum is not None and number > maximum:
        raise RepositoryCorruptionError(excess or message)
    return number


def _admit_process_header(
    artifact: ArtifactManifest,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[dict[str, str], int, tuple[ChunkRecord, ...], int]:
    """Admit a process artifact's metadata and declared inventory without reads.

    The metadata key set is exact, so retired process artifacts that embedded
    descriptors in metadata are refused. Returns the metadata, the declared
    shard count (at most ``limits.max_members``), the inventory chunks and
    their declared plaintext bytes (at most ``limits.max_manifest_bytes``).
    """
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance")
    metadata = dict(artifact.metadata)
    if (
        set(metadata) != _PROCESS_METADATA_KEYS
        or any(not value for value in metadata.values())
        or artifact.provider_id != metadata["repository_id"]
        or artifact.artifact_id
        != f"{metadata['checkpoint_id']}.process-{metadata['process_index']}"
    ):
        raise RepositoryCorruptionError(
            "Process checkpoint artifact metadata is invalid."
        )
    _decimal_metadata(
        metadata["process_index"], "Process checkpoint process index is invalid."
    )
    _decimal_metadata(
        metadata["topology_epoch"], "Process checkpoint topology epoch is invalid."
    )
    shard_count = _decimal_metadata(
        metadata["shard_count"],
        "Process checkpoint shard count is invalid.",
        maximum=limits.max_members,
        excess="Process checkpoint shard inventory exceeds the shard-count limit.",
    )
    chunks = tuple(
        chunk
        for chunk in artifact.chunks
        if chunk.logical_name == _INVENTORY_LOGICAL_NAME
    )
    declared = sum(chunk.plaintext_size for chunk in chunks)
    if not chunks or declared <= 0:
        raise RepositoryCorruptionError("Process checkpoint shard inventory is missing.")
    if declared > limits.max_manifest_bytes:
        raise RepositoryCorruptionError(
            "Process checkpoint shard inventory exceeds its byte limit."
        )
    if (2 if shard_count else 1) > limits.max_manifest_nesting:
        raise RepositoryCorruptionError(
            "Process checkpoint shard inventory exceeds its nesting limit."
        )
    return metadata, shard_count, chunks, declared


def _read_process_inventory(
    repository: ArtifactRepository,
    artifact: ArtifactManifest,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[dict[str, Any], ...]:
    """Admit one process artifact's canonical shard-descriptor inventory.

    :func:`_admit_process_header` bounds the metadata, declared shard count and
    inventory bytes before any inventory chunk is read; the joined bytes must
    match the metadata digest, ``limits.max_manifest_nesting``, the admitted
    top-level entry count (checked before decoding) and the canonical
    encoding. Every descriptor must carry its artifact's exact ownership and
    lineage, a unique ``shard_id`` and the ordinal ``array-{n}`` payload, and
    the artifact must hold exactly those payloads plus the inventory. Array
    payload chunks are not read.
    """
    metadata, shard_count, chunks, declared = _admit_process_header(artifact, limits)
    if artifact.provider_id != repository.provider_id:
        raise RepositoryCorruptionError(
            "Process checkpoint artifact metadata is invalid."
        )
    digest = hashlib.sha256()
    payload = bytearray()
    for chunk in chunks:
        piece = repository.read_chunk(artifact, chunk, maximum_plaintext_bytes=declared)
        if len(piece) > declared - len(payload):
            raise RepositoryCorruptionError(
                "Process checkpoint shard inventory byte count mismatch."
            )
        digest.update(piece)
        payload += piece
    if len(payload) != declared or digest.hexdigest() != metadata[_INVENTORY_DIGEST_KEY]:
        raise RepositoryCorruptionError(
            "Process checkpoint shard inventory digest mismatch."
        )
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise RepositoryCorruptionError(
            "Process checkpoint shard inventory is not canonical."
        ) from error
    descriptors = _bounded_json(
        text, limits.max_manifest_nesting, maximum_items=shard_count
    )
    del text
    if not isinstance(descriptors, list) or len(descriptors) != shard_count:
        raise RepositoryCorruptionError(
            "Process checkpoint shard descriptors are invalid."
        )
    try:
        canonical = canonical_json(descriptors).encode("ascii") == payload
    except (TypeError, ValueError) as error:
        raise RepositoryCorruptionError(
            "Process checkpoint shard inventory is not canonical."
        ) from error
    if not canonical:
        raise RepositoryCorruptionError(
            "Process checkpoint shard inventory is not canonical."
        )
    shard_ids: set[str] = set()
    payload_names = {_INVENTORY_LOGICAL_NAME}
    for ordinal, descriptor in enumerate(descriptors):
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != _DESCRIPTOR_KEYS
            or type(descriptor["byte_count"]) is not int
            or descriptor["byte_count"] < 0
            or any(
                not isinstance(descriptor[name], str) or not descriptor[name]
                for name in _DESCRIPTOR_KEYS - {"byte_count"}
            )
        ):
            raise RepositoryCorruptionError(
                "Process checkpoint shard descriptors are invalid."
            )
        if (
            descriptor["artifact_id"] != artifact.artifact_id
            or descriptor["logical_name"] != f"array-{ordinal}"
            or any(
                descriptor[name] != metadata[name] for name in _DESCRIPTOR_OWNERSHIP_KEYS
            )
        ):
            raise RepositoryCorruptionError(
                "Process checkpoint shard ownership metadata is inconsistent."
            )
        if descriptor["shard_id"] in shard_ids:
            raise RepositoryCorruptionError(
                "Process checkpoint shard descriptors repeat a shard ID."
            )
        shard_ids.add(descriptor["shard_id"])
        payload_names.add(descriptor["logical_name"])
    if {chunk.logical_name for chunk in artifact.chunks} != payload_names:
        raise RepositoryCorruptionError(
            "Process checkpoint artifact payload inventory is inconsistent."
        )
    return tuple(descriptors)


_ArtifactBinding = tuple[ArtifactManifest, dict[str, str], dict[str, dict[str, Any]]]


def _bind_artifact(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    artifact: ArtifactManifest,
    limits: ArrayArchiveLimits,
    /,
) -> _ArtifactBinding:
    """Bind one process artifact to a checkpoint and index its descriptors."""
    artifact_metadata = dict(artifact.metadata)
    if (
        artifact.provider_id != repository.provider_id
        or artifact_metadata.get("repository_id") != repository.provider_id
        or artifact_metadata.get("checkpoint_id") != manifest.checkpoint_id
        or artifact_metadata.get("analysis_plan_id") != manifest.analysis_plan_id
        or artifact_metadata.get("numeric_revision_id") != manifest.numeric_revision_id
        or artifact_metadata.get("execution_plan_id") != manifest.execution_plan_id
        or artifact_metadata.get("parent_checkpoint_id")
        != (manifest.parent_checkpoint_id or _NO_PARENT_ID)
        or artifact_metadata.get("parent_manifest_id")
        != (manifest.parent_manifest_id or _NO_PARENT_ID)
    ):
        raise RepositoryCorruptionError(
            "Checkpoint shard repository artifact ownership is inconsistent."
        )
    descriptors = _read_process_inventory(repository, artifact, limits)
    return (
        artifact,
        artifact_metadata,
        {descriptor["shard_id"]: descriptor for descriptor in descriptors},
    )


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
    ):
        raise RepositoryCorruptionError(
            "Parent checkpoint artifact ownership is inconsistent."
        )
    binding = _bind_artifact(repository, parent_manifest, parent_artifact, limits)
    descriptors = binding[2]
    parent_process_shards = tuple(
        shard
        for shard in parent_manifest.shards
        if _metadata(shard).get("process_index") == str(process_index)
    )
    if (
        len(parent_process_shards) != len(descriptors)
        or {shard.shard_id for shard in parent_process_shards} != set(descriptors)
        or any(
            _metadata(shard).get("artifact_id") != parent_artifact_id
            or _metadata(shard).get("repository_id") != repository.provider_id
            for shard in parent_process_shards
        )
    ):
        raise RepositoryCorruptionError(
            "Parent checkpoint manifest does not match its repository artifact."
        )
    _validate_repository_bindings(
        repository,
        parent_manifest,
        parent_process_shards,
        limits,
        {parent_artifact_id: binding},
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
    writer_index: int | None = None,
    attempt_id: str | None = None,
    topology_epoch: int = 0,
    encoding: ChunkEncoding = "identity",
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
    region_bytes: int | None = None,
) -> ProcessCheckpointPublication:
    """Publish this process's canonical addressable shards transactionally.

    ``region_bytes`` publishes each shard as canonical leading-axis region
    records (see :func:`snapshot_addressable_arrays`); every record carries its
    own payload digest and is restored independently.

    ``writer_index`` is the checkpoint writer slot of this process: the
    ``process-{slot}`` artifact, the artifact and shard ``process_index``
    metadata and the slot range ``[0, expected_process_count)`` that assembly
    requires. It defaults to ``jax.process_index()`` (a whole-world job); an
    execution subgroup passes its group-local rank. Every shard additionally
    records ``source_process_index``, the physical process that held the
    snapshot, and its layout identity keeps binding the physical device.

    The ordered shard descriptors are stored as canonical JSON chunks under
    the reserved ``checkpoint-shard-inventory`` payload, whose SHA-256 the
    artifact metadata binds as ``shard_inventory_digest``; a process without
    shards stores the empty inventory. ``limits.max_manifest_bytes``,
    ``limits.max_manifest_nesting`` and ``limits.max_members`` bound that
    inventory, and every reader must pass limits admitting it.
    """

    if writer_index is not None and (type(writer_index) is not int or writer_index < 0):
        raise ValueError("writer_index must be a nonnegative integer or None.")
    process_index = jax.process_index() if writer_index is None else writer_index
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
        region_bytes=region_bytes,
    )

    artifact_id = f"{checkpoint_id}.process-{process_index}"
    transaction = repository.begin(
        artifact_id,
        writer_id,
        attempt_id=attempt_id,
    )
    chunk_records = []
    checkpoint_shards = []
    descriptors = []
    for shard_index, snapshot in enumerate(snapshots):
        payload = snapshot.payload_bytes
        if len(payload) > limits.max_member_bytes:
            raise ValueError("Checkpoint shard payload exceeds its byte limit.")
        logical_name = f"array-{shard_index}"
        chunk_records.extend(
            _write_stream(
                repository, transaction, logical_name, payload, encoding=encoding
            )
        )
        digest = hashlib.sha256(payload).hexdigest()
        metadata = {
            "artifact_id": artifact_id,
            "checkpoint_id": checkpoint_id,
            "analysis_plan_id": analysis_plan_id,
            "numeric_revision_id": numeric_revision_id,
            "execution_plan_id": execution_plan_id,
            "logical_name": logical_name,
            "array_path": snapshot.array_path,
            "global_shape": canonical_json(list(snapshot.global_shape)),
            "dtype": snapshot.dtype,
            "index": _index_payload(snapshot.index),
            "process_index": str(process_index),
            "source_process_index": str(snapshot.process_index),
            "device_id": str(snapshot.device_id),
            "replica_id": str(snapshot.replica_id),
            "topology_epoch": str(topology_epoch),
            "repository_id": repository.provider_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "parent_manifest_id": parent_manifest_id,
        }
        shard_id = canonical_fingerprint(
            {
                "checkpoint_id": checkpoint_id,
                "execution_plan_id": execution_plan_id,
                "layout_id": snapshot.layout_id,
                "payload_digest": digest,
                "parent_manifest_id": parent_manifest_id,
            }
        )
        checkpoint_shards.append(
            CheckpointShard(
                shard_id,
                digest,
                len(payload),
                (snapshot.layout_id,),
                metadata=metadata,
            )
        )
        descriptors.append(
            {
                "shard_id": shard_id,
                "payload_digest": digest,
                "byte_count": len(payload),
                "layout_id": snapshot.layout_id,
                **metadata,
            }
        )

    inventory = canonical_json(descriptors).encode("ascii")
    if len(inventory) > limits.max_manifest_bytes:
        raise ValueError("Checkpoint shard inventory exceeds its byte limit.")
    if (2 if descriptors else 1) > limits.max_manifest_nesting:
        raise ValueError("Checkpoint shard inventory exceeds its nesting limit.")
    chunk_records.extend(
        _write_stream(repository, transaction, _INVENTORY_LOGICAL_NAME, inventory)
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
            "shard_count": str(len(checkpoint_shards)),
            _INVENTORY_DIGEST_KEY: hashlib.sha256(inventory).hexdigest(),
        },
    )
    published_shards = tuple(
        CheckpointShard(
            shard.shard_id,
            shard.payload_digest,
            shard.byte_count,
            shard.layout_ids,
            metadata={
                **_metadata(shard),
                "artifact_manifest_id": manifest.manifest_id,
            },
        )
        for shard in checkpoint_shards
    )
    return ProcessCheckpointPublication(
        process_index,
        manifest,
        published_shards,
    )


def _metadata(shard: CheckpointShard) -> dict[str, str]:
    return dict(shard.metadata)


def _metadata_integer(metadata: Mapping[str, str], name: str, /) -> int:
    return _decimal_metadata(
        metadata[name], f"Checkpoint shard {name} metadata is invalid."
    )


def _validated_shard(
    shard: CheckpointShard,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[str, tuple[int, ...], np.dtype[Any], CanonicalIndex]:
    metadata = _metadata(shard)
    if set(metadata) != _SHARD_METADATA_KEYS | {"artifact_manifest_id"} or any(
        not isinstance(metadata[name], str) or not metadata[name]
        for name in (
            "artifact_manifest_id",
            "artifact_id",
            "checkpoint_id",
            "analysis_plan_id",
            "numeric_revision_id",
            "execution_plan_id",
            "logical_name",
            "array_path",
            "repository_id",
        )
    ):
        raise RepositoryCorruptionError("Checkpoint shard metadata is invalid.")
    if bool(metadata["parent_checkpoint_id"]) != bool(metadata["parent_manifest_id"]):
        raise RepositoryCorruptionError("Checkpoint shard parent lineage is incomplete.")
    for name in (
        "process_index",
        "source_process_index",
        "device_id",
        "replica_id",
        "topology_epoch",
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
        if _any_overlap(indices):
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


# Complexity rationale: flat protocol predicates dominate the branch count.
# Keep this fail-closed ownership/coverage pass in its documented refusal order;
# bounded inventory decoding and artifact binding have separate owning helpers.
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
    for publication in publications_:
        artifact = publication.artifact_manifest
        if artifact is None:
            continue
        artifact_metadata, shard_count, _, _ = _admit_process_header(artifact, limits)
        if (
            artifact_metadata["process_index"] != str(publication.process_index)
            or shard_count != len(publication.shards)
            or {chunk.logical_name for chunk in artifact.chunks}
            != {
                _INVENTORY_LOGICAL_NAME,
                *(f"array-{ordinal}" for ordinal in range(shard_count)),
            }
            or any(
                _metadata(shard).get("logical_name") != f"array-{ordinal}"
                or _metadata(shard).get("artifact_id") != artifact.artifact_id
                or _metadata(shard).get("artifact_manifest_id") != artifact.manifest_id
                or _metadata(shard).get("process_index")
                != artifact_metadata["process_index"]
                or _metadata(shard).get("topology_epoch")
                != artifact_metadata["topology_epoch"]
                for ordinal, shard in enumerate(publication.shards)
            )
        ):
            raise RepositoryCorruptionError(
                "Process checkpoint publication does not match its artifact."
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


# Keep acknowledgement ownership, bounded metadata admission and global coverage
# in one fail-closed assembly boundary. No commit may be constructed before every
# process record has passed these ordered checks.
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
        artifact_metadata = dict(artifact.metadata)
        if (
            set(artifact_metadata) != _PROCESS_METADATA_KEYS
            or artifact.provider_id != repository.provider_id
            or artifact_metadata["repository_id"] != repository.provider_id
        ):
            raise RepositoryCorruptionError(
                "Process checkpoint artifact metadata is invalid."
            )
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
        descriptors = _read_process_inventory(repository, artifact, limits)
        try:
            shards = tuple(
                CheckpointShard(
                    descriptor["shard_id"],
                    descriptor["payload_digest"],
                    descriptor["byte_count"],
                    (descriptor["layout_id"],),
                    metadata={
                        **{key: descriptor[key] for key in _SHARD_METADATA_KEYS},
                        "artifact_manifest_id": artifact.manifest_id,
                    },
                )
                for descriptor in descriptors
            )
        except (TypeError, ValueError) as error:
            raise RepositoryCorruptionError(
                "Process checkpoint shard descriptors are invalid."
            ) from error
        publications.append(
            ProcessCheckpointPublication(
                process_index,
                artifact,
                shards,
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


def _validate_repository_bindings(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    shards: Sequence[CheckpointShard],
    limits: ArrayArchiveLimits,
    bound: Mapping[str, _ArtifactBinding] | None = None,
    /,
) -> dict[str, ArtifactManifest]:
    """Bind every shard record to its durable artifact descriptor exactly.

    Each artifact manifest is read and its descriptor inventory admitted once
    under ``limits``; every record must then match its own descriptor field for
    field. ``bound`` supplies artifacts already bound to this same ``manifest``
    by :func:`_bind_artifact`, so they are not read again.
    """
    bindings: dict[str, _ArtifactBinding] = {} if bound is None else dict(bound)
    expected_parent_checkpoint = manifest.parent_checkpoint_id or _NO_PARENT_ID
    expected_parent_manifest = manifest.parent_manifest_id or _NO_PARENT_ID
    for shard in shards:
        metadata = _metadata(shard)
        if (
            metadata["repository_id"] != repository.provider_id
            or metadata["checkpoint_id"] != manifest.checkpoint_id
            or metadata["analysis_plan_id"] != manifest.analysis_plan_id
            or metadata["numeric_revision_id"] != manifest.numeric_revision_id
            or metadata["execution_plan_id"] != manifest.execution_plan_id
            or metadata["parent_checkpoint_id"] != expected_parent_checkpoint
            or metadata["parent_manifest_id"] != expected_parent_manifest
        ):
            raise RepositoryCorruptionError(
                "Checkpoint shard lifecycle ownership is inconsistent."
            )
        artifact_id = metadata["artifact_id"]
        binding = bindings.get(artifact_id)
        if binding is None:
            binding = _bind_artifact(
                repository, manifest, repository.get_manifest(artifact_id), limits
            )
            bindings[artifact_id] = binding
        artifact, artifact_metadata, by_id = binding
        if (
            metadata["artifact_manifest_id"] != artifact.manifest_id
            or artifact_metadata.get("process_index") != metadata["process_index"]
        ):
            raise RepositoryCorruptionError(
                "Checkpoint shard repository artifact ownership is inconsistent."
            )
        expected_descriptor = {
            "shard_id": shard.shard_id,
            "payload_digest": shard.payload_digest,
            "byte_count": shard.byte_count,
            "layout_id": shard.layout_ids[0] if len(shard.layout_ids) == 1 else "",
            **{
                key: value
                for key, value in metadata.items()
                if key != "artifact_manifest_id"
            },
        }
        if by_id.get(shard.shard_id) != expected_descriptor:
            raise RepositoryCorruptionError(
                "Checkpoint shard descriptor was substituted."
            )
    return {identifier: binding[0] for identifier, binding in bindings.items()}


def _read_shard_payload(
    repository: ArtifactRepository,
    shard: CheckpointShard,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any],
    limits: ArrayArchiveLimits,
    /,
    artifact: ArtifactManifest | None = None,
) -> np.ndarray:
    """Read one record, verifying byte count, whole-record digest and schema.

    ``artifact`` reuses an already bound artifact manifest; the record's digest
    is always recomputed over every payload byte before decoding.
    """
    metadata = _metadata(shard)
    if artifact is None:
        artifact = repository.get_manifest(metadata["artifact_id"])
    elif artifact.artifact_id != metadata["artifact_id"]:
        raise RepositoryCorruptionError(
            "Checkpoint shard artifact binding is inconsistent."
        )
    logical_name = metadata["logical_name"]
    chunks = tuple(
        chunk for chunk in artifact.chunks if chunk.logical_name == logical_name
    )
    if not chunks:
        raise RepositoryCorruptionError(
            "Checkpoint shard artifact contains no matching chunks."
        )
    declared_bytes = sum(chunk.plaintext_size for chunk in chunks)
    if declared_bytes != shard.byte_count or declared_bytes > limits.max_member_bytes:
        raise RepositoryCorruptionError(
            "Checkpoint shard payload byte count is inconsistent."
        )
    digest = hashlib.sha256()
    observed = 0
    with tempfile.SpooledTemporaryFile(max_size=1_048_576, mode="w+b") as stream:
        for chunk in chunks:
            payload = repository.read_chunk(
                artifact, chunk, maximum_plaintext_bytes=shard.byte_count
            )
            observed += len(payload)
            if observed > shard.byte_count:
                raise RepositoryCorruptionError(
                    "Checkpoint shard payload byte count mismatch."
                )
            digest.update(payload)
            stream.write(payload)
        if observed != shard.byte_count:
            raise RepositoryCorruptionError(
                "Checkpoint shard payload byte count mismatch."
            )
        if digest.hexdigest() != shard.payload_digest:
            raise RepositoryCorruptionError("Checkpoint shard payload digest mismatch.")
        stream.seek(0)
        try:
            dtype, shape, _ = _read_npy_metadata(stream, observed, limits)
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


def _slice_volume(slices: tuple[slice, ...], /) -> int:
    volume = 1
    for entry in slices:
        volume *= entry.stop - entry.start
    return volume


def _index_volume(index: CanonicalIndex, /) -> int:
    volume = 1
    for start, stop, _ in index:
        volume *= stop - start
    return volume


@dataclass(frozen=True, slots=True)
class _AdmittedArray:
    """Validated region records of one array path, bound to their artifacts."""

    global_shape: tuple[int, ...]
    dtype: np.dtype[Any]
    records: tuple[tuple[CheckpointShard, CanonicalIndex], ...]
    artifacts: Mapping[str, ArtifactManifest]


def _path_records(
    manifest: CheckpointManifest,
    array_path: str,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[tuple[CheckpointShard, ...], tuple[int, ...], np.dtype[Any]]:
    if not isinstance(manifest, CheckpointManifest) or manifest.complete is not True:
        raise TypeError("manifest must be a complete CheckpointManifest")
    if not isinstance(array_path, str) or not array_path:
        raise ValueError("array_path must be non-empty text")
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance")
    path_shards = tuple(
        shard
        for shard in manifest.shards
        if _metadata(shard).get("array_path") == array_path
    )
    if not path_shards:
        raise KeyError(f"checkpoint has no array at path {array_path!r}")
    _validate_exact_coverage(path_shards, limits)
    _, global_shape, dtype, _ = _validated_shard(path_shards[0], limits)
    return path_shards, global_shape, dtype


def _admitted_array(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    array_path: str,
    limits: ArrayArchiveLimits,
    parent_manifest: CheckpointManifest | None,
    /,
) -> _AdmittedArray:
    path_shards, global_shape, dtype = _path_records(manifest, array_path, limits)
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
    artifacts = _validate_repository_bindings(repository, manifest, path_shards, limits)
    records = []
    for shard in path_shards:
        _, shape, shard_dtype, source_index = _validated_shard(shard, limits)
        if shape != global_shape or shard_dtype != dtype:
            raise RepositoryCorruptionError(
                "Checkpoint array shard metadata is inconsistent."
            )
        records.append((shard, source_index))
    return _AdmittedArray(global_shape, dtype, tuple(records), artifacts)


def _assemble_region(
    repository: ArtifactRepository,
    admitted: _AdmittedArray,
    destination: CanonicalIndex,
    limits: ArrayArchiveLimits,
    /,
) -> np.ndarray:
    """Copy only intersecting, individually authenticated records into a region.

    Memory is the destination block plus one record at a time. Exact coverage
    follows from validated non-overlap plus equal intersected volume.
    """
    output = np.empty(
        tuple(stop - start for start, stop, _ in destination), dtype=admitted.dtype
    )
    covered = 0
    for shard, source_index in admitted.records:
        overlap = _intersection(source_index, destination)
        if overlap is None:
            continue
        source_slice, destination_slice = overlap
        payload = _read_shard_payload(
            repository,
            shard,
            tuple(stop - start for start, stop, _ in source_index),
            admitted.dtype,
            limits,
            admitted.artifacts[_metadata(shard)["artifact_id"]],
        )
        output[destination_slice] = payload[source_slice]
        covered += _slice_volume(destination_slice)
    if covered != _index_volume(destination):
        raise RepositoryCorruptionError(
            "Destination checkpoint shard is not fully covered."
        )
    return output


@dataclass(frozen=True, slots=True)
class CheckpointRegionRestorePlan:
    """Authenticated region records one process reads to restore one array.

    Restore granularity is the published region record: every intersecting
    record is read and digest-verified in full, so ``read_payload_bytes``
    (encoded records, including their NumPy headers) bounds the I/O and is at
    least ``destination_bytes`` (raw destination values). Records that do not
    intersect this process's destinations are never read here.
    """

    array_path: str
    global_shape: tuple[int, ...]
    dtype: str
    destinations: tuple[CanonicalIndex, ...]
    record_count: int
    read_shard_ids: tuple[str, ...]
    read_payload_bytes: int
    destination_bytes: int


def _destination_indices(
    destination: Sharding | Sequence[Any],
    global_shape: tuple[int, ...],
    /,
) -> tuple[CanonicalIndex, ...]:
    raw = (
        tuple(destination.addressable_devices_indices_map(global_shape).values())
        if isinstance(destination, Sharding)
        else tuple(destination)
    )
    indices: list[CanonicalIndex] = []
    for index in raw:
        normalized = _canonical_index(index, global_shape)
        if normalized not in indices:
            indices.append(normalized)
    return tuple(indices)


def plan_checkpoint_region_restore(
    manifest: CheckpointManifest,
    array_path: str,
    destination: Sharding | Sequence[Any],
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
) -> CheckpointRegionRestorePlan:
    """Report exactly which authenticated records a restore reads on this process.

    ``destination`` is a destination sharding (its addressable indices) or an
    explicit sequence of index tuples. Only manifest metadata is inspected;
    :func:`restore_global_array_from_checkpoint` and
    :func:`read_checkpoint_region` read exactly these records.
    """
    path_shards, global_shape, dtype = _path_records(manifest, array_path, limits)
    destinations = _destination_indices(destination, global_shape)
    sources = tuple((shard, _validated_shard(shard, limits)[3]) for shard in path_shards)
    read: dict[str, int] = {}
    for target in destinations:
        for shard, source_index in sources:
            if _intersection(source_index, target) is not None:
                read.setdefault(shard.shard_id, shard.byte_count)
    return CheckpointRegionRestorePlan(
        array_path=array_path,
        global_shape=global_shape,
        dtype=dtype.str,
        destinations=destinations,
        record_count=len(path_shards),
        read_shard_ids=tuple(read),
        read_payload_bytes=sum(read.values()),
        destination_bytes=sum(
            _index_volume(target) * dtype.itemsize for target in destinations
        ),
    )


def restore_global_array_from_checkpoint(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    array_path: str,
    sharding: Sharding,
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
    committed: bool = True,
) -> jax.Array:
    """Restore one globally bounded array directly into a destination sharding.

    Every record's metadata and repository binding is validated, but only the
    records intersecting this process's addressable destinations are read;
    each read record is digest-verified in full. A destination index shared by
    several local replicas is assembled once.
    ``committed=False`` restores an uncommitted local single-device array, so
    compiled consumers may place it alongside their distributed arguments.
    Named/global shardings remain explicitly committed.
    """
    if type(committed) is not bool:
        raise TypeError("committed must be a bool.")
    if not committed and (
        not isinstance(sharding, SingleDeviceSharding)
        or next(iter(sharding.device_set)).process_index != jax.process_index()
    ):
        raise ValueError("Uncommitted restore requires a local SingleDeviceSharding.")
    admitted = _admitted_array(repository, manifest, array_path, limits, parent_manifest)
    assembled: dict[CanonicalIndex, np.ndarray] = {}

    def load_destination(index: Any) -> np.ndarray:
        destination = _canonical_index(index, admitted.global_shape)
        block = assembled.get(destination)
        if block is None:
            block = _assemble_region(repository, admitted, destination, limits)
            assembled[destination] = block
        return block

    if not committed:
        with default_device(next(iter(sharding.device_set))):
            value = jnp.asarray(
                load_destination(tuple(slice(None) for _ in admitted.global_shape))
            )
        if value.dtype != admitted.dtype:
            raise ValueError("The active precision cannot restore this array dtype.")
        return value

    return jax.make_array_from_callback(admitted.global_shape, sharding, load_destination)


def read_checkpoint_region(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    array_path: str,
    region: Any,
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
) -> np.ndarray:
    """Read one host region (integers/unit-stride slices per axis) of an array.

    Only intersecting authenticated records are read, each verified in full;
    the result is a read-only host block of exactly the requested region.
    """
    admitted = _admitted_array(repository, manifest, array_path, limits, parent_manifest)
    destination = _canonical_index(region, admitted.global_shape)
    if _index_volume(destination) * admitted.dtype.itemsize > limits.max_member_bytes:
        raise ValueError("Checkpoint region exceeds the member byte limit.")
    block = _assemble_region(repository, admitted, destination, limits)
    block.setflags(write=False)
    return block


class IncompleteDistributedCheckpointError(RepositoryError):
    """A distributed checkpoint has no durable global commit and is not restorable."""


_COMMIT_LOGICAL_NAME = "global-commit"
_COMMIT_RECORD_KEYS = frozenset(
    {
        "kind",
        "checkpoint_id",
        "analysis_plan_id",
        "numeric_revision_id",
        "execution_plan_id",
        "manifest_id",
        "expected_process_count",
        "parent_checkpoint_id",
        "parent_manifest_id",
        "diagnostic_ids",
        "shard_count",
        "repository_id",
    }
)
_COMMIT_METADATA_KEYS = frozenset(
    {"kind", "checkpoint_id", "manifest_id", "record_digest", "repository_id"}
)


def _commit_artifact_id(checkpoint_id: str, /) -> str:
    return f"{checkpoint_id}.commit"


@dataclass(frozen=True, slots=True)
class DistributedCheckpointCommit:
    """Durable global commit receipt of one complete distributed checkpoint."""

    checkpoint_id: str
    manifest_id: str
    expected_process_count: int
    record_digest: str
    artifact_manifest: ArtifactManifest


def commit_distributed_checkpoint(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    /,
    *,
    expected_process_count: int,
    writer_id: str,
    attempt_id: str | None = None,
    parent_manifest: CheckpointManifest | None = None,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
) -> DistributedCheckpointCommit:
    """Atomically publish the single global commit of a complete checkpoint.

    The manifest is re-derived from the durable per-process artifacts and must
    be identical; every shard record is bound to its repository descriptor.
    Until this commit exists, :func:`load_distributed_checkpoint_commit`
    refuses the checkpoint, so partial coverage never becomes restorable.
    """
    if not isinstance(manifest, CheckpointManifest) or manifest.complete is not True:
        raise TypeError("manifest must be a complete CheckpointManifest")
    durable = assemble_distributed_checkpoint_from_repository(
        repository,
        manifest.checkpoint_id,
        manifest.analysis_plan_id,
        manifest.numeric_revision_id,
        manifest.execution_plan_id,
        expected_process_count=expected_process_count,
        parent_manifest=parent_manifest,
        diagnostic_ids=manifest.diagnostic_ids,
        limits=limits,
    )
    if durable.manifest_id != manifest.manifest_id:
        raise RepositoryCorruptionError(
            "The global commit manifest differs from the durable publications."
        )
    _validate_repository_bindings(repository, durable, durable.shards, limits)
    record = {
        "kind": "distributed-checkpoint-commit",
        "checkpoint_id": durable.checkpoint_id,
        "analysis_plan_id": durable.analysis_plan_id,
        "numeric_revision_id": durable.numeric_revision_id,
        "execution_plan_id": durable.execution_plan_id,
        "manifest_id": durable.manifest_id,
        "expected_process_count": expected_process_count,
        "parent_checkpoint_id": durable.parent_checkpoint_id or _NO_PARENT_ID,
        "parent_manifest_id": durable.parent_manifest_id or _NO_PARENT_ID,
        "diagnostic_ids": list(durable.diagnostic_ids),
        "shard_count": len(durable.shards),
        "repository_id": repository.provider_id,
    }
    payload = canonical_json(record).encode("utf-8")
    if len(payload) > _MAX_METADATA_VALUE_BYTES:
        raise ValueError("The global commit record exceeds its byte limit.")
    digest = hashlib.sha256(payload).hexdigest()
    transaction = repository.begin(
        _commit_artifact_id(durable.checkpoint_id), writer_id, attempt_id=attempt_id
    )
    chunks = _write_stream(repository, transaction, _COMMIT_LOGICAL_NAME, payload)
    artifact = repository.commit(
        transaction,
        chunks,
        metadata={
            "kind": "distributed-checkpoint-commit",
            "checkpoint_id": durable.checkpoint_id,
            "manifest_id": durable.manifest_id,
            "record_digest": digest,
            "repository_id": repository.provider_id,
        },
    )
    return DistributedCheckpointCommit(
        durable.checkpoint_id,
        durable.manifest_id,
        expected_process_count,
        digest,
        artifact,
    )


def load_distributed_checkpoint_commit(
    repository: ArtifactRepository,
    checkpoint_id: str,
    /,
    *,
    parent_manifest: CheckpointManifest | None = None,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
) -> CheckpointManifest:
    """Load the globally committed manifest or refuse an uncommitted checkpoint.

    The commit record is digest-verified, the manifest re-derived from the
    durable per-process artifacts, and its identity must equal the committed
    one. A missing commit raises :class:`IncompleteDistributedCheckpointError`.
    """
    try:
        artifact = repository.get_manifest(_commit_artifact_id(checkpoint_id))
    except ObjectNotFoundError as error:
        raise IncompleteDistributedCheckpointError(
            f"Distributed checkpoint {checkpoint_id!r} has no global commit."
        ) from error
    metadata = dict(artifact.metadata)
    if (
        set(metadata) != _COMMIT_METADATA_KEYS
        or metadata["kind"] != "distributed-checkpoint-commit"
        or metadata["checkpoint_id"] != checkpoint_id
        or metadata["repository_id"] != repository.provider_id
        or artifact.provider_id != repository.provider_id
        or {chunk.logical_name for chunk in artifact.chunks} != {_COMMIT_LOGICAL_NAME}
    ):
        raise RepositoryCorruptionError("Distributed checkpoint commit is invalid.")
    declared = sum(chunk.plaintext_size for chunk in artifact.chunks)
    if declared > _MAX_METADATA_VALUE_BYTES:
        raise RepositoryCorruptionError(
            "Distributed checkpoint commit exceeds its byte limit."
        )
    digest = hashlib.sha256()
    pieces = []
    for chunk in artifact.chunks:
        piece = repository.read_chunk(artifact, chunk, maximum_plaintext_bytes=declared)
        digest.update(piece)
        pieces.append(piece)
    payload = b"".join(pieces)
    if len(payload) != declared or digest.hexdigest() != metadata["record_digest"]:
        raise RepositoryCorruptionError("Distributed checkpoint commit digest mismatch.")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RepositoryCorruptionError(
            "Distributed checkpoint commit is invalid."
        ) from error
    record = _metadata_json(text)
    if (
        not isinstance(record, dict)
        or set(record) != _COMMIT_RECORD_KEYS
        or record["kind"] != "distributed-checkpoint-commit"
        or record["checkpoint_id"] != checkpoint_id
        or record["manifest_id"] != metadata["manifest_id"]
        or record["repository_id"] != repository.provider_id
        or type(record["expected_process_count"]) is not int
        or type(record["shard_count"]) is not int
        or not isinstance(record["diagnostic_ids"], list)
        or any(
            not isinstance(record[name], str) or not record[name]
            for name in (
                "analysis_plan_id",
                "numeric_revision_id",
                "execution_plan_id",
                "parent_checkpoint_id",
                "parent_manifest_id",
            )
        )
    ):
        raise RepositoryCorruptionError("Distributed checkpoint commit is invalid.")
    expected_parent = (
        (_NO_PARENT_ID, _NO_PARENT_ID)
        if parent_manifest is None
        else (parent_manifest.checkpoint_id, parent_manifest.manifest_id)
    )
    if (record["parent_checkpoint_id"], record["parent_manifest_id"]) != expected_parent:
        raise ValueError("The committed checkpoint has another parent lineage.")
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        checkpoint_id,
        record["analysis_plan_id"],
        record["numeric_revision_id"],
        record["execution_plan_id"],
        expected_process_count=record["expected_process_count"],
        parent_manifest=parent_manifest,
        diagnostic_ids=record["diagnostic_ids"],
        limits=limits,
    )
    if (
        manifest.manifest_id != record["manifest_id"]
        or len(manifest.shards) != record["shard_count"]
    ):
        raise RepositoryCorruptionError(
            "Durable publications differ from the global commit."
        )
    return manifest


__all__ = (
    "AddressableCheckpointShard",
    "assemble_distributed_checkpoint_from_repository",
    "CheckpointRegionRestorePlan",
    "commit_distributed_checkpoint",
    "DistributedCheckpointCommit",
    "IncompleteDistributedCheckpointError",
    "load_distributed_checkpoint_commit",
    "plan_checkpoint_region_restore",
    "ProcessCheckpointPublication",
    "assemble_distributed_checkpoint_manifest",
    "publish_process_checkpoint",
    "read_checkpoint_region",
    "restore_global_array_from_checkpoint",
    "snapshot_addressable_arrays",
)
