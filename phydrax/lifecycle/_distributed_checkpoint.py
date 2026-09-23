#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Addressable-shard checkpoint publication and topology-neutral restore."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.sharding import Sharding

from .._array_archive import (
    _read_npy_metadata,
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
)
from .._fingerprint import canonical_fingerprint, canonical_json
from ._chunk_repository import (
    ArtifactManifest,
    ArtifactRepository,
    ChunkEncoding,
    RepositoryCorruptionError,
)
from ._models import CheckpointManifest, CheckpointShard


CanonicalIndex = tuple[tuple[int, int, int], ...]
_DISTRIBUTED_CHECKPOINT_LIMITS = DEFAULT_ARRAY_ARCHIVE_LIMITS
_NO_PARENT_ID = "none"


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


def _metadata_json(value: str, /) -> Any:
    if not isinstance(value, str) or len(value.encode("utf-8")) > 1024 * 1024:
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
            if depth > 16:
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


def snapshot_addressable_arrays(
    tree: Any,
    /,
    *,
    topology_epoch: int = 0,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
) -> tuple[AddressableCheckpointShard, ...]:
    """Copy only canonical locally addressable JAX shards to host memory."""

    if type(topology_epoch) is not int or topology_epoch < 0:
        raise ValueError("topology_epoch must be a non-negative integer")
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance")
    pending: list[tuple[str, jax.Array, Any, CanonicalIndex]] = []
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
            if len(pending) >= limits.max_members:
                raise ValueError("Checkpoint exceeds the shard-count limit.")
            shard.data.copy_to_host_async()
            pending.append((array_path, leaf, shard, normalized))

    snapshots: list[AddressableCheckpointShard] = []
    for array_path, leaf, shard, index in pending:
        payload = np.asarray(shard.data)
        local_shape = tuple(stop - start for start, stop, _ in index)
        if (
            payload.shape != local_shape
            or payload.dtype != np.dtype(leaf.dtype)
            or payload.nbytes > limits.max_member_bytes
        ):
            raise ValueError("Addressable checkpoint shard payload is inconsistent.")
        layout_id = canonical_fingerprint(
            {
                "array_path": array_path,
                "global_shape": list(leaf.shape),
                "dtype": np.dtype(leaf.dtype).str,
                "index": [list(entry) for entry in index],
                "process_index": int(shard.device.process_index),
                "device_id": int(shard.device.id),
                "replica_id": int(shard.replica_id),
                "topology_epoch": topology_epoch,
            }
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


def _parent_lineage(
    repository: ArtifactRepository,
    parent_manifest: CheckpointManifest | None,
    process_index: int,
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
    descriptors = _metadata_json(parent_metadata.get("shards", ""))
    if not isinstance(descriptors, list):
        raise RepositoryCorruptionError(
            "Parent checkpoint shard descriptors are invalid."
        )
    descriptor_ids = {
        descriptor.get("shard_id")
        for descriptor in descriptors
        if isinstance(descriptor, dict)
    }
    parent_process_shards = tuple(
        shard
        for shard in parent_manifest.shards
        if _metadata(shard).get("process_index") == str(process_index)
    )
    if (
        len(descriptor_ids) != len(descriptors)
        or {shard.shard_id for shard in parent_process_shards} != descriptor_ids
        or any(
            _metadata(shard).get("artifact_id") != parent_artifact_id
            or _metadata(shard).get("repository_id") != repository.provider_id
            for shard in parent_process_shards
        )
    ):
        raise RepositoryCorruptionError(
            "Parent checkpoint manifest does not match its repository artifact."
        )
    for shard in parent_process_shards:
        _validate_shard_repository_binding(repository, parent_manifest, shard)
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
) -> ProcessCheckpointPublication:
    """Publish this process's canonical addressable shards transactionally."""

    process_index = jax.process_index()
    parent_checkpoint_id, parent_manifest_id = _parent_lineage(
        repository,
        parent_manifest,
        process_index,
        execution_plan_id,
    )
    snapshots = snapshot_addressable_arrays(
        tree,
        topology_epoch=topology_epoch,
        limits=limits,
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
        offset = 0
        chunk_index = 0
        while offset < len(payload):
            stop = min(offset + repository.maximum_chunk_bytes, len(payload))
            chunk_records.append(
                repository.write_chunk(
                    transaction,
                    logical_name,
                    chunk_index,
                    offset,
                    payload[offset:stop],
                    encoding=encoding,
                )
            )
            offset = stop
            chunk_index += 1
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
            "process_index": str(snapshot.process_index),
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

    if not chunk_records:
        chunk_records.append(
            repository.write_chunk(
                transaction,
                "checkpoint-ack",
                0,
                0,
                b"{}",
                encoding="identity",
            )
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
            "shards": canonical_json(descriptors),
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
    expected = {
        "artifact_id",
        "checkpoint_id",
        "analysis_plan_id",
        "numeric_revision_id",
        "artifact_manifest_id",
        "execution_plan_id",
        "logical_name",
        "array_path",
        "global_shape",
        "dtype",
        "index",
        "process_index",
        "device_id",
        "replica_id",
        "topology_epoch",
        "repository_id",
        "parent_checkpoint_id",
        "parent_manifest_id",
    }
    if set(metadata) != expected or any(
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
    for name in ("process_index", "device_id", "replica_id", "topology_epoch"):
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
    structural_keys = frozenset({"shard_id", "payload_digest", "byte_count", "layout_id"})
    metadata_keys = {
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
        "device_id",
        "replica_id",
        "topology_epoch",
        "repository_id",
        "parent_checkpoint_id",
        "parent_manifest_id",
    }
    descriptor_keys = structural_keys | metadata_keys
    artifact_metadata_keys = {
        "checkpoint_id",
        "analysis_plan_id",
        "numeric_revision_id",
        "execution_plan_id",
        "process_index",
        "topology_epoch",
        "shard_count",
        "shards",
        "repository_id",
        "parent_checkpoint_id",
        "parent_manifest_id",
    }
    for process_index in range(expected_process_count):
        if parent_manifest is not None:
            _parent_lineage(
                repository,
                parent_manifest,
                process_index,
                execution_plan_id,
            )
        artifact_id = f"{checkpoint_id}.process-{process_index}"
        artifact = repository.get_manifest(artifact_id)
        artifact_metadata = dict(artifact.metadata)
        if (
            set(artifact_metadata) != artifact_metadata_keys
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
        descriptors = _metadata_json(artifact_metadata["shards"])
        if (
            not isinstance(descriptors, list)
            or len(descriptors) > limits.max_members
            or any(
                not isinstance(descriptor, dict)
                or set(descriptor) != descriptor_keys
                or type(descriptor["byte_count"]) is not int
                or descriptor["byte_count"] < 0
                or any(
                    not isinstance(descriptor[name], str) or not descriptor[name]
                    for name in descriptor_keys
                    - {"byte_count", "parent_checkpoint_id", "parent_manifest_id"}
                )
                or bool(descriptor["parent_checkpoint_id"])
                != bool(descriptor["parent_manifest_id"])
                for descriptor in descriptors
            )
        ):
            raise RepositoryCorruptionError(
                "Process checkpoint shard descriptors are invalid."
            )
        if any(
            descriptor["artifact_id"] != artifact_id
            or descriptor["process_index"] != str(process_index)
            or descriptor["topology_epoch"] != artifact_metadata["topology_epoch"]
            or descriptor["logical_name"] != f"array-{ordinal}"
            or descriptor["repository_id"] != repository.provider_id
            or descriptor["parent_checkpoint_id"] != parent_checkpoint_id
            or descriptor["parent_manifest_id"] != parent_manifest_id
            for ordinal, descriptor in enumerate(descriptors)
        ):
            raise RepositoryCorruptionError(
                "Process checkpoint shard ownership metadata is inconsistent."
            )
        expected_payloads = (
            {"checkpoint-ack"}
            if not descriptors
            else {descriptor["logical_name"] for descriptor in descriptors}
        )
        if {chunk.logical_name for chunk in artifact.chunks} != expected_payloads:
            raise RepositoryCorruptionError(
                "Process checkpoint artifact payload inventory is inconsistent."
            )
        try:
            shards = tuple(
                CheckpointShard(
                    descriptor["shard_id"],
                    descriptor["payload_digest"],
                    descriptor["byte_count"],
                    (descriptor["layout_id"],),
                    metadata={
                        **{key: descriptor[key] for key in metadata_keys},
                        "artifact_manifest_id": artifact.manifest_id,
                    },
                )
                for descriptor in descriptors
            )
        except (TypeError, ValueError) as error:
            raise RepositoryCorruptionError(
                "Process checkpoint shard descriptors are invalid."
            ) from error
        shard_count = artifact_metadata["shard_count"]
        if not shard_count.isascii() or not shard_count.isdecimal():
            raise RepositoryCorruptionError("Process checkpoint shard count is invalid.")
        if int(shard_count) != len(shards):
            raise RepositoryCorruptionError(
                "Process checkpoint shard count is inconsistent."
            )
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


def _validate_shard_repository_binding(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    shard: CheckpointShard,
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
    descriptors = _metadata_json(artifact_metadata.get("shards", ""))
    descriptor_metadata = {
        key: value for key, value in metadata.items() if key != "artifact_manifest_id"
    }
    expected_descriptor = {
        "shard_id": shard.shard_id,
        "payload_digest": shard.payload_digest,
        "byte_count": shard.byte_count,
        "layout_id": shard.layout_ids[0] if len(shard.layout_ids) == 1 else "",
        **descriptor_metadata,
    }
    matches = (
        [
            descriptor
            for descriptor in descriptors
            if isinstance(descriptor, dict)
            and descriptor.get("shard_id") == shard.shard_id
        ]
        if isinstance(descriptors, list)
        else []
    )
    if len(matches) != 1 or matches[0] != expected_descriptor:
        raise RepositoryCorruptionError("Checkpoint shard descriptor was substituted.")
    descriptor_names = {
        descriptor.get("logical_name")
        for descriptor in descriptors
        if isinstance(descriptor, dict)
    }
    if {chunk.logical_name for chunk in artifact.chunks} != descriptor_names:
        raise RepositoryCorruptionError(
            "Checkpoint shard repository payload inventory is inconsistent."
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
    payload = b"".join(
        repository.read_chunk(
            artifact,
            chunk,
            maximum_plaintext_bytes=shard.byte_count,
        )
        for chunk in chunks
    )
    if len(payload) != shard.byte_count:
        raise RepositoryCorruptionError("Checkpoint shard payload byte count mismatch.")
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


def restore_global_array_from_checkpoint(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    array_path: str,
    sharding: Sharding,
    /,
    *,
    limits: ArrayArchiveLimits = _DISTRIBUTED_CHECKPOINT_LIMITS,
    parent_manifest: CheckpointManifest | None = None,
) -> jax.Array:
    """Restore one globally bounded array directly into a destination sharding."""

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
        )
        if expected_parent != (parent_checkpoint_id, parent_manifest_id):
            raise RepositoryCorruptionError(
                "Checkpoint restore parent repository lineage is inconsistent."
            )
    _validate_exact_coverage(path_shards, limits)
    _, global_shape, dtype, _ = _validated_shard(path_shards[0], limits)

    source_records = []
    for shard in path_shards:
        _validate_shard_repository_binding(repository, manifest, shard)
        _, shape, shard_dtype, source_index = _validated_shard(shard, limits)
        if shape != global_shape or shard_dtype != dtype:
            raise RepositoryCorruptionError(
                "Checkpoint array shard metadata is inconsistent."
            )
        local_shape = tuple(stop - start for start, stop, _ in source_index)
        payload = _read_shard_payload(
            repository,
            shard,
            local_shape,
            dtype,
            limits,
        )
        source_records.append((source_index, payload))
    admitted_sources = tuple(source_records)

    def load_destination(index: Any) -> np.ndarray:
        destination = _canonical_index(index, global_shape)
        local_shape = tuple(stop - start for start, stop, _ in destination)
        output = np.empty(local_shape, dtype=dtype)
        covered = np.zeros(local_shape, dtype=np.bool_)
        for source_index, payload in admitted_sources:
            overlap = _intersection(source_index, destination)
            if overlap is None:
                continue
            source_slice, destination_slice = overlap
            output[destination_slice] = payload[source_slice]
            covered[destination_slice] = True
        if not bool(np.all(covered)):
            raise RepositoryCorruptionError(
                "Destination checkpoint shard is not fully covered."
            )
        return output

    return jax.make_array_from_callback(global_shape, sharding, load_destination)


__all__ = (
    "AddressableCheckpointShard",
    "assemble_distributed_checkpoint_from_repository",
    "ProcessCheckpointPublication",
    "assemble_distributed_checkpoint_manifest",
    "publish_process_checkpoint",
    "restore_global_array_from_checkpoint",
    "snapshot_addressable_arrays",
)
