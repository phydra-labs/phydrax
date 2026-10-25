#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authenticated region records, overlap restore, shard inventories and commit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from phydrax._array_archive import ArrayArchiveLimits
from phydrax._jax_config import default_device
from phydrax.lifecycle._chunk_repository import RepositoryCorruptionError
from phydrax.lifecycle._distributed_checkpoint import (
    _any_overlap,
    assemble_distributed_checkpoint_from_repository,
    assemble_distributed_checkpoint_manifest,
    commit_distributed_checkpoint,
    IncompleteDistributedCheckpointError,
    load_distributed_checkpoint_commit,
    plan_checkpoint_region_restore,
    ProcessCheckpointPublication,
    publish_process_checkpoint,
    read_checkpoint_region,
    restore_global_array_from_checkpoint,
)
from phydrax.lifecycle._models import CheckpointManifest, CheckpointShard
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)


def _repository(tmp_path: Path, maximum_chunk_bytes: int = 64) -> POSIXArtifactRepository:
    profile = HPCFilesystemProfile(
        "posix.region-test",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    policy = POSIXRepositoryPolicy(
        profile,
        maximum_chunk_bytes=maximum_chunk_bytes,
        maximum_metadata_bytes=1 << 20,
    )
    return POSIXArtifactRepository(tmp_path / "repository", policy)


def _publish(
    repository: POSIXArtifactRepository,
    checkpoint_id: str,
    value: jax.Array,
    *,
    region_bytes: int | None,
) -> CheckpointManifest:
    publish_process_checkpoint(
        repository,
        checkpoint_id,
        "execution-regions",
        {"value": value},
        analysis_plan_id="analysis-regions",
        numeric_revision_id="revision-regions",
        writer_id="writer",
        region_bytes=region_bytes,
    )
    return assemble_distributed_checkpoint_from_repository(
        repository,
        checkpoint_id,
        "analysis-regions",
        "revision-regions",
        "execution-regions",
        expected_process_count=1,
    )


def _value() -> jax.Array:
    device = jax.devices()[0]
    sharding = NamedSharding(
        Mesh(np.asarray((device,), dtype=object), ("rows",)), PartitionSpec("rows")
    )
    return jax.device_put(jnp.arange(24, dtype=jnp.float64).reshape(8, 3), sharding)


def test_region_records_restore_only_intersecting_regions(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    value = _value()
    manifest = _publish(repository, "regions", value, region_bytes=48)
    records = [
        shard
        for shard in manifest.shards
        if dict(shard.metadata)["array_path"] == "['value']"
    ]
    assert len(records) == 4
    plan = plan_checkpoint_region_restore(
        manifest, "['value']", ((slice(2, 4), slice(None)),)
    )
    assert plan.record_count == 4 and len(plan.read_shard_ids) == 1
    assert plan.read_payload_bytes >= plan.destination_bytes == 2 * 3 * 8
    block = read_checkpoint_region(
        repository, manifest, "['value']", (slice(2, 4), slice(1, 3))
    )
    np.testing.assert_array_equal(block, np.arange(24).reshape(8, 3)[2:4, 1:3])
    assert not block.flags.writeable
    spanning = plan_checkpoint_region_restore(
        manifest, "['value']", ((slice(1, 6), slice(0, 1)),)
    )
    assert len(spanning.read_shard_ids) == 3
    restored = restore_global_array_from_checkpoint(
        repository, manifest, "['value']", value.sharding
    )
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(value))
    whole = plan_checkpoint_region_restore(manifest, "['value']", value.sharding)
    assert len(whole.read_shard_ids) == 4


def test_restore_preserves_uncommitted_device_placement(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    device = jax.local_devices()[-1]
    with default_device(device):
        original = jnp.asarray([1.25, -3.5], dtype=jnp.float64)
    manifest = _publish(repository, "uncommitted-device", original, region_bytes=None)
    restored = restore_global_array_from_checkpoint(
        repository, manifest, "['value']", original.sharding, committed=False
    )
    assert not restored.committed
    assert restored.devices() == original.devices()
    np.testing.assert_array_equal(restored, original)
    placed = restore_global_array_from_checkpoint(
        repository, manifest, "['value']", SingleDeviceSharding(device)
    )
    assert placed.committed
    np.testing.assert_array_equal(placed, original)
    named = NamedSharding(Mesh(np.asarray([device]), ("owner",)), PartitionSpec())
    with pytest.raises(ValueError):
        restore_global_array_from_checkpoint(
            repository, manifest, "['value']", named, committed=False
        )


def test_default_publication_keeps_one_record_per_shard(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manifest = _publish(repository, "whole", _value(), region_bytes=None)
    assert len(manifest.shards) == 1


def test_tampered_region_is_refused_while_other_regions_restore(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manifest = _publish(repository, "tamper", _value(), region_bytes=48)
    record = next(
        shard
        for shard in manifest.shards
        if dict(shard.metadata)["index"] == "[[0,2,1],[0,3,1]]"
    )
    metadata = dict(record.metadata)
    artifact = repository.get_manifest(metadata["artifact_id"])
    chunk = next(
        chunk
        for chunk in artifact.chunks
        if chunk.logical_name == metadata["logical_name"]
    )
    target = tmp_path / "repository" / chunk.object_key
    target.chmod(0o600)
    data = bytearray(target.read_bytes())
    data[-1] ^= 0x01
    target.write_bytes(bytes(data))
    with pytest.raises(RepositoryCorruptionError):
        read_checkpoint_region(
            repository, manifest, "['value']", (slice(0, 1), slice(None))
        )
    with pytest.raises(RepositoryCorruptionError):
        restore_global_array_from_checkpoint(
            repository, manifest, "['value']", _value().sharding
        )
    untouched = read_checkpoint_region(
        repository, manifest, "['value']", (slice(4, 8), slice(None))
    )
    np.testing.assert_array_equal(untouched, np.arange(24).reshape(8, 3)[4:8])


def test_uncommitted_checkpoint_is_refused_and_commit_binds_durable_manifest(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    manifest = _publish(repository, "commit", _value(), region_bytes=48)
    with pytest.raises(IncompleteDistributedCheckpointError):
        load_distributed_checkpoint_commit(repository, "commit")
    partial = CheckpointManifest(
        manifest.checkpoint_id,
        manifest.analysis_plan_id,
        manifest.numeric_revision_id,
        manifest.execution_plan_id,
        manifest.shards[:-1],
        complete=True,
    )
    with pytest.raises(RepositoryCorruptionError, match="durable publications"):
        commit_distributed_checkpoint(
            repository, partial, expected_process_count=1, writer_id="writer"
        )
    with pytest.raises(IncompleteDistributedCheckpointError):
        load_distributed_checkpoint_commit(repository, "commit")
    receipt = commit_distributed_checkpoint(
        repository, manifest, expected_process_count=1, writer_id="writer"
    )
    assert receipt.manifest_id == manifest.manifest_id
    loaded = load_distributed_checkpoint_commit(repository, "commit")
    assert loaded.manifest_id == manifest.manifest_id


def test_overlap_sweep_detects_multidimensional_overlap() -> None:
    disjoint = (
        ((0, 2, 1), (0, 3, 1)),
        ((2, 4, 1), (0, 1, 1)),
        ((2, 4, 1), (1, 3, 1)),
    )
    assert not _any_overlap(disjoint)
    overlapping = disjoint + (((3, 5, 1), (2, 3, 1)),)
    assert _any_overlap(overlapping)
    # Touching leading intervals with overlapping trailing axes do not overlap.
    assert not _any_overlap((((0, 2, 1), (0, 3, 1)), ((2, 3, 1), (0, 3, 1))))
    assert not _any_overlap((((0, 0, 1), (0, 3, 1)), ((0, 2, 1), (0, 3, 1))))


_INVENTORY = "checkpoint-shard-inventory"
_DIGEST_KEY = "shard_inventory_digest"


class _ReadRecorder:
    """Repository proxy recording every logical payload whose chunk is read."""

    def __init__(self, repository: POSIXArtifactRepository) -> None:
        self._repository = repository
        self.reads: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._repository, name)

    def read_chunk(self, manifest: Any, chunk: Any, /, **options: Any) -> bytes:
        self.reads.append(chunk.logical_name)
        return self._repository.read_chunk(manifest, chunk, **options)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _inventory(
    repository: POSIXArtifactRepository, artifact_id: str
) -> tuple[bytes, list[dict[str, Any]]]:
    payload = repository.read_bytes(artifact_id, _INVENTORY, maximum_bytes=1 << 24)
    return payload, json.loads(payload)


def _rewrite(
    repository: POSIXArtifactRepository,
    artifact_id: str,
    *,
    inventory: bytes | None = None,
    metadata: dict[str, str] | None = None,
    drop: frozenset[str] = frozenset(),
    extra: dict[str, bytes] | None = None,
) -> None:
    """Commit a writer-authenticated replacement of one process artifact."""
    artifact = repository.get_manifest(artifact_id)
    payloads = {
        name: repository.read_bytes(artifact_id, name, maximum_bytes=1 << 24)
        for name in sorted({chunk.logical_name for chunk in artifact.chunks} - drop)
    }
    if inventory is not None:
        payloads[_INVENTORY] = inventory
    payloads.update(extra or {})
    transaction = repository.begin(artifact_id, "forger")
    step = repository.maximum_chunk_bytes
    chunks = [
        repository.write_chunk(
            transaction, name, index, offset, payload[offset : offset + step]
        )
        for name, payload in payloads.items()
        for index, offset in enumerate(range(0, len(payload), step))
    ]
    repository.commit(
        transaction,
        chunks,
        metadata=dict(artifact.metadata) if metadata is None else metadata,
    )


class _Rewrite(TypedDict, total=False):
    """Keyword arguments of one :func:`_rewrite` forgery."""

    inventory: bytes
    metadata: dict[str, str]
    drop: frozenset[str]
    extra: dict[str, bytes]


def _resigned(metadata: dict[str, str], inventory: bytes, count: int) -> _Rewrite:
    return _Rewrite(
        inventory=inventory,
        metadata={
            **metadata,
            _DIGEST_KEY: hashlib.sha256(inventory).hexdigest(),
            "shard_count": str(count),
        },
    )


def _descriptors(metadata: dict[str, str], descriptors: list[dict[str, Any]]) -> _Rewrite:
    return _resigned(metadata, _canonical(descriptors), len(descriptors))


def _duplicate_shard_id(
    repository: POSIXArtifactRepository,
    artifact_id: str,
    payload: bytes,
    descriptors: list[dict[str, Any]],
    metadata: dict[str, str],
) -> _Rewrite:
    ordinal = len(descriptors)
    duplicate = {**descriptors[0], "logical_name": f"array-{ordinal}"}
    copied = repository.read_bytes(artifact_id, "array-0", maximum_bytes=1 << 24)
    forged = _descriptors(metadata, [*descriptors, duplicate])
    forged["extra"] = {f"array-{ordinal}": copied}
    return forged


def _duplicate_member(
    repository: POSIXArtifactRepository,
    artifact_id: str,
    payload: bytes,
    descriptors: list[dict[str, Any]],
    metadata: dict[str, str],
) -> _Rewrite:
    text = payload.decode("ascii")
    forged = text.replace(
        '{"analysis_plan_id":', '{"analysis_plan_id":"forged","analysis_plan_id":', 1
    )
    assert forged != text
    return _resigned(metadata, forged.encode("ascii"), len(descriptors))


def _legacy_metadata_inventory(
    repository: POSIXArtifactRepository,
    artifact_id: str,
    payload: bytes,
    descriptors: list[dict[str, Any]],
    metadata: dict[str, str],
) -> _Rewrite:
    # The retired wire format: descriptors embedded in metadata["shards"].
    legacy = {key: value for key, value in metadata.items() if key != _DIGEST_KEY}
    return _Rewrite(
        metadata={**legacy, "shards": payload.decode("ascii")},
        drop=frozenset({_INVENTORY}),
    )


_Forgery = Callable[
    [POSIXArtifactRepository, str, bytes, list[dict[str, Any]], dict[str, str]],
    _Rewrite,
]
_FORGERIES: dict[str, _Forgery] = {
    "missing-inventory": lambda r, a, p, d, m: _Rewrite(drop=frozenset({_INVENTORY})),
    "digest-mismatch": lambda r, a, p, d, m: _Rewrite(
        inventory=_canonical([{**d[0], "byte_count": d[0]["byte_count"] + 1}, *d[1:]])
    ),
    "reordered-descriptors": lambda r, a, p, d, m: _descriptors(m, d[::-1]),
    "dropped-descriptor": lambda r, a, p, d, m: _descriptors(m, d[:-1]),
    "non-canonical-encoding": lambda r, a, p, d, m: _resigned(
        m, json.dumps(d, indent=1).encode("ascii"), len(d)
    ),
    "duplicate-member": _duplicate_member,
    "duplicate-shard-id": _duplicate_shard_id,
    "extra-payload": lambda r, a, p, d, m: _Rewrite(extra={"stray-payload": b"stray"}),
    "foreign-artifact-owner": lambda r, a, p, d, m: _descriptors(
        m, [{**d[0], "artifact_id": "other.process-0"}, *d[1:]]
    ),
    "substituted-lineage": lambda r, a, p, d, m: _descriptors(
        m,
        [
            {**d[0], "parent_checkpoint_id": "forged", "parent_manifest_id": "f" * 64},
            *d[1:],
        ],
    ),
    "shard-count": lambda r, a, p, d, m: _Rewrite(
        metadata={**m, "shard_count": str(len(d) + 1)}
    ),
    "legacy-metadata-inventory": _legacy_metadata_inventory,
}


@pytest.mark.parametrize("forgery", sorted(_FORGERIES))
def test_forged_or_retired_process_inventory_is_refused(
    tmp_path: Path, forgery: str
) -> None:
    repository = _repository(tmp_path)
    manifest = _publish(repository, "forged", _value(), region_bytes=48)
    commit_distributed_checkpoint(
        repository, manifest, expected_process_count=1, writer_id="writer"
    )
    artifact_id = "forged.process-0"
    payload, descriptors = _inventory(repository, artifact_id)
    assert len(descriptors) == 4
    metadata = dict(repository.get_manifest(artifact_id).metadata)
    _rewrite(
        repository,
        artifact_id,
        **_FORGERIES[forgery](repository, artifact_id, payload, descriptors, metadata),
    )
    with pytest.raises(RepositoryCorruptionError):
        assemble_distributed_checkpoint_from_repository(
            repository,
            "forged",
            "analysis-regions",
            "revision-regions",
            "execution-regions",
            expected_process_count=1,
        )
    with pytest.raises(RepositoryCorruptionError):
        load_distributed_checkpoint_commit(repository, "forged")
    with pytest.raises(RepositoryCorruptionError):
        read_checkpoint_region(
            repository, manifest, "['value']", (slice(4, 8), slice(None))
        )


def test_corrupted_inventory_chunk_is_refused(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manifest = _publish(repository, "corrupt", _value(), region_bytes=48)
    artifact = repository.get_manifest("corrupt.process-0")
    chunk = next(chunk for chunk in artifact.chunks if chunk.logical_name == _INVENTORY)
    target = tmp_path / "repository" / chunk.object_key
    target.chmod(0o600)
    data = bytearray(target.read_bytes())
    data[-1] ^= 0x01
    target.write_bytes(bytes(data))
    with pytest.raises(RepositoryCorruptionError):
        assemble_distributed_checkpoint_from_repository(
            repository,
            "corrupt",
            "analysis-regions",
            "revision-regions",
            "execution-regions",
            expected_process_count=1,
        )
    with pytest.raises(RepositoryCorruptionError):
        read_checkpoint_region(
            repository, manifest, "['value']", (slice(4, 8), slice(None))
        )


def test_empty_participant_stores_the_empty_inventory(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    value = _value()
    for writer_index, tree in enumerate(({"value": value}, {})):
        publish_process_checkpoint(
            repository,
            "sparse",
            "execution-regions",
            tree,
            analysis_plan_id="analysis-regions",
            numeric_revision_id="revision-regions",
            writer_id=f"writer-{writer_index}",
            writer_index=writer_index,
        )
    empty = repository.get_manifest("sparse.process-1")
    empty_metadata = dict(empty.metadata)
    assert {chunk.logical_name for chunk in empty.chunks} == {_INVENTORY}
    assert _inventory(repository, "sparse.process-1") == (b"[]", [])
    assert empty_metadata["shard_count"] == "0"
    assert empty_metadata[_DIGEST_KEY] == hashlib.sha256(b"[]").hexdigest()
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "sparse",
        "analysis-regions",
        "revision-regions",
        "execution-regions",
        expected_process_count=2,
    )
    commit_distributed_checkpoint(
        repository, manifest, expected_process_count=2, writer_id="writer"
    )
    loaded = load_distributed_checkpoint_commit(repository, "sparse")
    assert loaded.manifest_id == manifest.manifest_id
    restored = restore_global_array_from_checkpoint(
        repository, loaded, "['value']", value.sharding
    )
    assert np.asarray(restored).tobytes() == np.asarray(value).tobytes()


def test_region_read_never_reads_unrelated_corrupted_payloads(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    manifest = _publish(repository, "selective", _value(), region_bytes=48)
    artifact = repository.get_manifest("selective.process-0")
    wanted = next(
        dict(shard.metadata)["logical_name"]
        for shard in manifest.shards
        if dict(shard.metadata)["index"] == "[[4,6,1],[0,3,1]]"
    )
    for chunk in artifact.chunks:
        if chunk.logical_name in {_INVENTORY, wanted}:
            continue
        target = tmp_path / "repository" / chunk.object_key
        target.chmod(0o600)
        data = bytearray(target.read_bytes())
        data[0] ^= 0x01
        target.write_bytes(bytes(data))
    recorder = _ReadRecorder(repository)
    block = read_checkpoint_region(
        recorder,  # ty: ignore[invalid-argument-type]
        manifest,
        "['value']",
        (slice(4, 6), slice(None)),
    )
    np.testing.assert_array_equal(block, np.arange(24).reshape(8, 3)[4:6])
    assert set(recorder.reads) == {_INVENTORY, wanted}
    with pytest.raises(RepositoryCorruptionError):
        restore_global_array_from_checkpoint(
            repository, manifest, "['value']", _value().sharding
        )


def test_inventory_beyond_metadata_value_ceiling_round_trips_bitwise(
    tmp_path: Path,
) -> None:
    # Large chunks keep the 900 one-row records within the repository's own
    # chunk-count and artifact-manifest budgets; only the inventory is large.
    repository = _repository(tmp_path, maximum_chunk_bytes=1 << 20)
    rows = 900
    sufficient = ArrayArchiveLimits(max_manifest_bytes=8 << 20, max_members=1024)
    host = np.random.default_rng(7).standard_normal((rows, 1))
    host[0, 0] = -0.0
    sharding = _value().sharding
    key = "halo-owner-" + "x" * 1013
    array_path = f"['{key}']"
    tree = {key: jax.device_put(jnp.asarray(host), sharding)}
    with pytest.raises(ValueError, match="inventory"):
        publish_process_checkpoint(
            repository,
            "large-refused",
            "execution-large",
            tree,
            analysis_plan_id="analysis-large",
            numeric_revision_id="revision-large",
            writer_id="writer",
            limits=ArrayArchiveLimits(max_members=1024),
            region_bytes=8,
        )
    publish_process_checkpoint(
        repository,
        "large",
        "execution-large",
        tree,
        analysis_plan_id="analysis-large",
        numeric_revision_id="revision-large",
        writer_id="writer",
        limits=sufficient,
        region_bytes=8,
    )
    metadata = dict(repository.get_manifest("large.process-0").metadata)
    payload, descriptors = _inventory(repository, "large.process-0")
    assert len(payload) > 1 << 20
    assert len(descriptors) == rows
    assert "shards" not in metadata
    assert metadata[_DIGEST_KEY] == hashlib.sha256(payload).hexdigest()
    assert metadata["shard_count"] == str(rows)
    assert max(len(item.encode("utf-8")) for item in metadata.values()) < 1 << 10
    identity = ("large", "analysis-large", "revision-large", "execution-large")
    manifest = assemble_distributed_checkpoint_from_repository(
        repository, *identity, expected_process_count=1, limits=sufficient
    )
    commit_distributed_checkpoint(
        repository,
        manifest,
        expected_process_count=1,
        writer_id="writer",
        limits=sufficient,
    )
    loaded = load_distributed_checkpoint_commit(repository, "large", limits=sufficient)
    assert loaded.manifest_id == manifest.manifest_id
    assert len(loaded.shards) == rows
    restored = restore_global_array_from_checkpoint(
        repository, loaded, array_path, sharding, limits=sufficient
    )
    assert restored.dtype == jnp.float64
    assert np.asarray(restored).tobytes() == host.tobytes()
    block = read_checkpoint_region(
        repository, loaded, array_path, (slice(450, 452), slice(None)), limits=sufficient
    )
    assert block.tobytes() == host[450:452].tobytes()
    for too_small in (
        ArrayArchiveLimits(max_manifest_bytes=1 << 20, max_members=1024),
        ArrayArchiveLimits(max_manifest_bytes=8 << 20, max_members=rows - 1),
    ):
        recorder = _ReadRecorder(repository)
        with pytest.raises(RepositoryCorruptionError):
            assemble_distributed_checkpoint_from_repository(
                recorder,  # ty: ignore[invalid-argument-type]
                *identity,
                expected_process_count=1,
                limits=too_small,
            )
        assert recorder.reads == []
        with pytest.raises(RepositoryCorruptionError):
            load_distributed_checkpoint_commit(repository, "large", limits=too_small)
        with pytest.raises(RepositoryCorruptionError):
            read_checkpoint_region(
                repository,
                loaded,
                array_path,
                (slice(0, 1), slice(None)),
                limits=too_small,
            )


_REGION_IDENTITY = (
    "analysis-regions",
    "revision-regions",
    "execution-regions",
)


def _publication(
    repository: POSIXArtifactRepository, checkpoint_id: str, *, region_bytes: int | None
) -> ProcessCheckpointPublication:
    return publish_process_checkpoint(
        repository,
        checkpoint_id,
        "execution-regions",
        {"value": _value()},
        analysis_plan_id="analysis-regions",
        numeric_revision_id="revision-regions",
        writer_id="writer",
        region_bytes=region_bytes,
    )


def test_publication_assembly_refuses_insufficient_inventory_depth(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    publication = _publication(repository, "depth", region_bytes=48)
    with pytest.raises(RepositoryCorruptionError):
        assemble_distributed_checkpoint_manifest(
            "depth",
            *_REGION_IDENTITY,
            (publication,),
            expected_process_count=1,
            limits=ArrayArchiveLimits(max_manifest_nesting=1),
        )


def test_publication_assembly_admits_inventory_headers(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    publication = _publication(repository, "headers", region_bytes=48)
    artifact_id = "headers.process-0"
    manifest = assemble_distributed_checkpoint_manifest(
        "headers", *_REGION_IDENTITY, (publication,), expected_process_count=1
    )
    durable = assemble_distributed_checkpoint_from_repository(
        repository, "headers", *_REGION_IDENTITY, expected_process_count=1
    )
    assert manifest.manifest_id == durable.manifest_id
    payload, descriptors = _inventory(repository, artifact_id)
    with pytest.raises(RepositoryCorruptionError):
        assemble_distributed_checkpoint_manifest(
            "headers",
            *_REGION_IDENTITY,
            (publication,),
            expected_process_count=1,
            limits=ArrayArchiveLimits(max_manifest_bytes=len(payload) - 1),
        )
    reordered = ProcessCheckpointPublication(
        0, publication.artifact_manifest, publication.shards[::-1]
    )
    with pytest.raises(RepositoryCorruptionError):
        assemble_distributed_checkpoint_manifest(
            "headers", *_REGION_IDENTITY, (reordered,), expected_process_count=1
        )
    metadata = dict(repository.get_manifest(artifact_id).metadata)
    _rewrite(
        repository,
        artifact_id,
        **_legacy_metadata_inventory(
            repository, artifact_id, payload, descriptors, metadata
        ),
    )
    legacy = repository.get_manifest(artifact_id)
    rebound = tuple(
        CheckpointShard(
            shard.shard_id,
            shard.payload_digest,
            shard.byte_count,
            shard.layout_ids,
            metadata={**dict(shard.metadata), "artifact_manifest_id": legacy.manifest_id},
        )
        for shard in publication.shards
    )
    with pytest.raises(RepositoryCorruptionError):
        assemble_distributed_checkpoint_manifest(
            "headers",
            *_REGION_IDENTITY,
            (ProcessCheckpointPublication(0, legacy, rebound),),
            expected_process_count=1,
        )


@pytest.mark.parametrize(
    "inventory",
    [
        b"[" + b",".join([b"{}"] * 200_000) + b"]",
        b"[1,2,3]",
        b'{"shard_id":"x"}',
        b'"[]"',
    ],
    ids=["many-objects", "non-object-entries", "object-root", "string-root"],
)
def test_inventory_entries_beyond_declared_count_are_refused_before_decoding(
    tmp_path: Path, inventory: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path, maximum_chunk_bytes=1 << 20)
    _publication(repository, "understated", region_bytes=None)
    artifact_id = "understated.process-0"
    metadata = dict(repository.get_manifest(artifact_id).metadata)
    _rewrite(repository, artifact_id, **_resigned(metadata, inventory, 0))
    forged_text = inventory.decode("ascii")
    decoded_forgery: list[int] = []
    decode = json.loads

    def admitting_decoder(text: str | bytes | bytearray, **options: Any) -> Any:
        # Resource boundary: the forged inventory must never reach the decoder;
        # all other repository JSON decodes normally.
        forged = forged_text if isinstance(text, str) else inventory
        if text == forged:
            decoded_forgery.append(len(text))
        return decode(text, **options)

    monkeypatch.setattr(json, "loads", admitting_decoder)
    with pytest.raises(RepositoryCorruptionError):
        assemble_distributed_checkpoint_from_repository(
            repository, "understated", *_REGION_IDENTITY, expected_process_count=1
        )
    assert decoded_forgery == []


@pytest.mark.parametrize("field", ["shard_count", "topology_epoch"])
def test_oversized_decimal_metadata_is_repository_corruption(
    tmp_path: Path, field: str
) -> None:
    repository = _repository(tmp_path)
    publication = _publication(repository, "decimal", region_bytes=None)
    artifact_id = "decimal.process-0"
    metadata = dict(repository.get_manifest(artifact_id).metadata)
    _rewrite(repository, artifact_id, metadata={**metadata, field: "9" * 5000})
    forged = repository.get_manifest(artifact_id)
    rebound = tuple(
        CheckpointShard(
            shard.shard_id,
            shard.payload_digest,
            shard.byte_count,
            shard.layout_ids,
            metadata={**dict(shard.metadata), "artifact_manifest_id": forged.manifest_id},
        )
        for shard in publication.shards
    )
    with pytest.raises(RepositoryCorruptionError):
        assemble_distributed_checkpoint_from_repository(
            repository, "decimal", *_REGION_IDENTITY, expected_process_count=1
        )
    with pytest.raises(RepositoryCorruptionError):
        assemble_distributed_checkpoint_manifest(
            "decimal",
            *_REGION_IDENTITY,
            (ProcessCheckpointPublication(0, forged, rebound),),
            expected_process_count=1,
        )
