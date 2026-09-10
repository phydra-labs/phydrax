#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Addressable-shard checkpoint publication and topology-neutral restore."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.sharding import Sharding

from .._fingerprint import canonical_fingerprint, canonical_json
from ._chunk_repository import ArtifactManifest, ArtifactRepository, ChunkEncoding
from ._models import CheckpointManifest, CheckpointShard


CanonicalIndex = tuple[tuple[int, int, int], ...]


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


def _parse_index(value: str) -> CanonicalIndex:
    raw = json.loads(value)
    return tuple((int(start), int(stop), int(step)) for start, stop, step in raw)


def _index_size(index: CanonicalIndex) -> int:
    size = 1
    for start, stop, step in index:
        if step != 1:
            raise ValueError("checkpoint shard indices require unit stride")
        size *= max(stop - start, 0)
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
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
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
) -> tuple[AddressableCheckpointShard, ...]:
    """Copy only canonical locally addressable JAX shards to host memory."""

    if topology_epoch < 0:
        raise ValueError("topology_epoch must be non-negative")
    pending: list[tuple[str, jax.Array, Any, CanonicalIndex]] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not isinstance(leaf, jax.Array):
            continue
        array_path = jax.tree_util.keystr(path) or "<root>"
        shape = tuple(int(size) for size in leaf.shape)
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
            shard.data.copy_to_host_async()
            pending.append((array_path, leaf, shard, normalized))

    snapshots: list[AddressableCheckpointShard] = []
    for array_path, leaf, shard, index in pending:
        payload = np.asarray(shard.data)
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
                global_shape=tuple(int(size) for size in leaf.shape),
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


def publish_process_checkpoint(
    repository: ArtifactRepository,
    checkpoint_id: str,
    execution_plan_id: str,
    tree: Any,
    /,
    *,
    writer_id: str,
    attempt_id: str | None = None,
    topology_epoch: int = 0,
    encoding: ChunkEncoding = "identity",
) -> ProcessCheckpointPublication:
    """Publish this process's canonical addressable shards transactionally."""

    process_index = jax.process_index()
    snapshots = snapshot_addressable_arrays(tree, topology_epoch=topology_epoch)

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
            "logical_name": logical_name,
            "array_path": snapshot.array_path,
            "global_shape": canonical_json(list(snapshot.global_shape)),
            "dtype": snapshot.dtype,
            "index": _index_payload(snapshot.index),
            "process_index": str(snapshot.process_index),
            "device_id": str(snapshot.device_id),
            "replica_id": str(snapshot.replica_id),
            "topology_epoch": str(topology_epoch),
        }
        shard_id = canonical_fingerprint(
            {
                "checkpoint_id": checkpoint_id,
                "execution_plan_id": execution_plan_id,
                "layout_id": snapshot.layout_id,
                "payload_digest": digest,
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
            "execution_plan_id": execution_plan_id,
            "process_index": str(process_index),
            "topology_epoch": str(topology_epoch),
            "shard_count": str(len(checkpoint_shards)),
            "shards": canonical_json(descriptors),
        },
    )
    return ProcessCheckpointPublication(
        process_index,
        manifest,
        tuple(checkpoint_shards),
    )


def _metadata(shard: CheckpointShard) -> dict[str, str]:
    return dict(shard.metadata)


def _validate_exact_coverage(shards: Sequence[CheckpointShard]) -> None:
    by_path: dict[str, list[CheckpointShard]] = {}
    for shard in shards:
        metadata = _metadata(shard)
        path = metadata.get("array_path")
        if path is None:
            continue
        by_path.setdefault(path, []).append(shard)

    for path, path_shards in by_path.items():
        metadata = [_metadata(shard) for shard in path_shards]
        shapes = {tuple(json.loads(item["global_shape"])) for item in metadata}
        dtypes = {item["dtype"] for item in metadata}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise ValueError(f"checkpoint array {path!r} has inconsistent metadata")
        shape = tuple(int(size) for size in next(iter(shapes)))
        indices = [_parse_index(item["index"]) for item in metadata]
        if len(set(indices)) != len(indices):
            raise ValueError(f"checkpoint array {path!r} contains duplicate shards")
        for left_index, left in enumerate(indices):
            for right in indices[left_index + 1 :]:
                if _indices_overlap(left, right):
                    raise ValueError(
                        f"checkpoint array {path!r} contains overlapping shards"
                    )
        covered = sum(_index_size(index) for index in indices)
        expected = int(np.prod(shape, dtype=np.int64))
        if covered != expected:
            raise ValueError(
                f"checkpoint array {path!r} covers {covered} of {expected} values"
            )


def assemble_distributed_checkpoint_manifest(
    checkpoint_id: str,
    analysis_plan_id: str,
    numeric_revision_id: str,
    execution_plan_id: str,
    publications: Sequence[ProcessCheckpointPublication],
    /,
    *,
    expected_process_count: int,
    parent_checkpoint_id: str | None = None,
    diagnostic_ids: Sequence[str] = (),
) -> CheckpointManifest:
    """Validate all process acknowledgements and create the global commit record."""

    publications_ = tuple(publications)
    process_indices = tuple(publication.process_index for publication in publications_)
    if tuple(sorted(process_indices)) != tuple(range(expected_process_count)):
        raise ValueError(
            "checkpoint publications do not cover every process exactly once"
        )
    shards = tuple(shard for publication in publications_ for shard in publication.shards)
    if not shards:
        raise ValueError("a complete distributed checkpoint requires array shards")
    _validate_exact_coverage(shards)
    return CheckpointManifest(
        checkpoint_id,
        analysis_plan_id,
        numeric_revision_id,
        execution_plan_id,
        shards,
        complete=True,
        parent_checkpoint_id=parent_checkpoint_id,
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
    parent_checkpoint_id: str | None = None,
    diagnostic_ids: Sequence[str] = (),
) -> CheckpointManifest:
    """Assemble deterministic process acknowledgements from durable artifacts."""

    publications = []
    structural_keys = frozenset({"shard_id", "payload_digest", "byte_count", "layout_id"})
    for process_index in range(expected_process_count):
        artifact_id = f"{checkpoint_id}.process-{process_index}"
        artifact = repository.get_manifest(artifact_id)
        artifact_metadata = dict(artifact.metadata)
        if artifact_metadata.get("checkpoint_id") != checkpoint_id:
            raise ValueError("process checkpoint artifact has the wrong checkpoint_id")
        if artifact_metadata.get("execution_plan_id") != execution_plan_id:
            raise ValueError("process checkpoint artifact has the wrong execution plan")
        if artifact_metadata.get("process_index") != str(process_index):
            raise ValueError("process checkpoint artifact has the wrong process index")
        descriptors = json.loads(artifact_metadata["shards"])
        if not isinstance(descriptors, list) or any(
            not isinstance(descriptor, dict) for descriptor in descriptors
        ):
            raise ValueError("process checkpoint shard descriptors are invalid")
        shards = tuple(
            CheckpointShard(
                descriptor["shard_id"],
                descriptor["payload_digest"],
                int(descriptor["byte_count"]),
                (descriptor["layout_id"],),
                metadata={
                    key: str(value)
                    for key, value in descriptor.items()
                    if key not in structural_keys
                },
            )
            for descriptor in descriptors
        )
        if int(artifact_metadata["shard_count"]) != len(shards):
            raise ValueError("process checkpoint shard count is inconsistent")
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
        parent_checkpoint_id=parent_checkpoint_id,
        diagnostic_ids=diagnostic_ids,
    )


def _read_shard_payload(
    repository: ArtifactRepository,
    shard: CheckpointShard,
) -> np.ndarray:
    metadata = _metadata(shard)
    artifact = repository.get_manifest(metadata["artifact_id"])
    logical_name = metadata["logical_name"]
    chunks = tuple(
        sorted(
            (chunk for chunk in artifact.chunks if chunk.logical_name == logical_name),
            key=lambda chunk: chunk.index,
        )
    )
    if not chunks:
        raise ValueError("checkpoint shard artifact contains no matching chunks")
    payload = b"".join(repository.read_chunk(artifact, chunk) for chunk in chunks)
    if len(payload) != shard.byte_count:
        raise ValueError("checkpoint shard payload byte count mismatch")
    if hashlib.sha256(payload).hexdigest() != shard.payload_digest:
        raise ValueError("checkpoint shard payload digest mismatch")
    return np.load(io.BytesIO(payload), allow_pickle=False)


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
) -> jax.Array:
    """Restore one global array directly into a destination sharding."""

    path_shards = tuple(
        shard
        for shard in manifest.shards
        if _metadata(shard).get("array_path") == array_path
    )
    if not path_shards:
        raise KeyError(f"checkpoint has no array at path {array_path!r}")
    _validate_exact_coverage(path_shards)
    first_metadata = _metadata(path_shards[0])
    global_shape = tuple(int(size) for size in json.loads(first_metadata["global_shape"]))
    dtype = np.dtype(first_metadata["dtype"])

    source_records = tuple(
        (
            _parse_index(_metadata(shard)["index"]),
            shard,
        )
        for shard in path_shards
    )

    def load_destination(index: Any) -> np.ndarray:
        destination = _canonical_index(index, global_shape)
        local_shape = tuple(stop - start for start, stop, _ in destination)
        output = np.empty(local_shape, dtype=dtype)
        covered = np.zeros(local_shape, dtype=bool)
        for source_index, shard in source_records:
            overlap = _intersection(source_index, destination)
            if overlap is None:
                continue
            source_slice, destination_slice = overlap
            payload = _read_shard_payload(repository, shard)
            output[destination_slice] = payload[source_slice]
            covered[destination_slice] = True
        if not bool(np.all(covered)):
            raise ValueError("destination checkpoint shard is not fully covered")
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
