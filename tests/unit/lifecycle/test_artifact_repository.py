#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
from pathlib import Path

import pytest

from phydrax.lifecycle._chunk_repository import (
    RepositoryConflictError,
    RepositoryCorruptionError,
    RetentionPolicy,
    UnsupportedRepositoryProfileError,
)
from phydrax.lifecycle._repository import (
    ArtifactGuardRecoveryAuthorization,
    HPCFilesystemProfile,
    InMemoryConditionalObjectClient,
    ObjectNotFoundError,
    ObjectPreconditionError,
    ObjectStoreProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
    S3ArtifactRepository,
)
from phydrax.lifecycle._restart_topology import (
    admit_topology_restart,
    CanonicalRestartChunk,
    DestinationShard,
    DirectRestorePlan,
    execute_direct_restore,
    prepare_direct_restore,
    RestartChunkMapping,
    TopologyRestartPolicy,
    TopologyRestartRelation,
)
from phydrax.qualification._registry import SupportTuple


class SimulatedCrash(RuntimeError):
    pass


def _posix_policy(
    provider_id: str = "posix.qualified",
    /,
    *,
    maximum_chunk_bytes: int = 1024,
) -> POSIXRepositoryPolicy:
    profile = HPCFilesystemProfile(
        provider_id,
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXRepositoryPolicy(
        profile,
        maximum_chunk_bytes=maximum_chunk_bytes,
        maximum_metadata_bytes=64 * 1024,
    )


def _object_profile(
    provider_id: str = "s3.qualified",
    /,
) -> ObjectStoreProfile:
    return ObjectStoreProfile(
        provider_id,
        conditional_create=True,
        conditional_replace=True,
        strongly_consistent_reads=True,
        strongly_consistent_listing=True,
        multipart_free_objects=True,
        maximum_object_bytes=64 * 1024,
    )


def _stage(
    repository: POSIXArtifactRepository | S3ArtifactRepository,
    artifact: str,
    attempt: str,
    payload: bytes,
    /,
    *,
    started_at: int,
):
    transaction = repository.begin(
        artifact,
        "writer-a",
        attempt_id=attempt,
        started_at=started_at,
    )
    chunk = repository.write_chunk(
        transaction,
        "state",
        0,
        0,
        payload,
        encoding="zlib",
    )
    return transaction, chunk


@pytest.mark.parametrize(
    ("crash_point", "visible"),
    (
        ("before_manifest", False),
        ("after_manifest", False),
        ("before_commit_marker", False),
        ("after_commit_marker", False),
        ("before_pointer", False),
        ("after_pointer", True),
    ),
)
def test_posix_commit_crash_boundaries_are_recoverable_and_atomic(
    tmp_path: Path,
    crash_point: str,
    visible: bool,
) -> None:
    def fail(point: str) -> None:
        if point == crash_point:
            raise SimulatedCrash(point)

    policy = _posix_policy()
    repository = POSIXArtifactRepository(
        tmp_path / "repository",
        policy,
        failure_injector=fail,
    )
    transaction, chunk = _stage(
        repository,
        "checkpoint-a",
        f"attempt-{crash_point}",
        b"restart-state",
        started_at=10,
    )

    with pytest.raises(SimulatedCrash, match=crash_point):
        repository.commit(transaction, (chunk,), committed_at=20)

    reopened = POSIXArtifactRepository(tmp_path / "repository", policy)
    if visible:
        manifest = reopened.get_manifest("checkpoint-a")
        assert reopened.read_chunk(manifest, manifest.chunks[0]) == b"restart-state"
        with pytest.raises(RepositoryConflictError, match="already committed"):
            reopened.commit(transaction, (chunk,), committed_at=20)
    else:
        with pytest.raises(ObjectNotFoundError):
            reopened.get_manifest("checkpoint-a")
        manifest = reopened.commit(transaction, (chunk,), committed_at=20)
        assert reopened.read_chunk(manifest, chunk) == b"restart-state"


def test_posix_reader_detects_chunk_and_commit_marker_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    repository = POSIXArtifactRepository(root, _posix_policy())
    transaction, chunk = _stage(
        repository,
        "checkpoint-a",
        "attempt-a",
        b"correct-state",
        started_at=10,
    )
    manifest = repository.commit(transaction, (chunk,), committed_at=20)
    (root / chunk.object_key).write_bytes(b"corrupt")

    with pytest.raises(RepositoryCorruptionError, match="size mismatch"):
        repository.read_chunk(manifest, chunk)

    marker = root / "roots" / transaction.attempt_id / "COMMIT"
    marker.write_bytes(b"{}\n")
    with pytest.raises(RepositoryCorruptionError, match="marker"):
        repository.get_manifest("checkpoint-a")


def test_duplicate_chunk_and_stale_writer_conflicts_are_fail_closed(
    tmp_path: Path,
) -> None:
    repository = POSIXArtifactRepository(tmp_path / "repository", _posix_policy())
    first, first_chunk = _stage(
        repository,
        "checkpoint-a",
        "attempt-first",
        b"first",
        started_at=10,
    )
    stale, stale_chunk = _stage(
        repository,
        "checkpoint-a",
        "attempt-stale",
        b"stale",
        started_at=11,
    )
    with pytest.raises(RepositoryConflictError, match="already exists"):
        repository.write_chunk(first, "state", 0, 0, b"duplicate")
    repository.commit(first, (first_chunk,), committed_at=20)

    with pytest.raises(RepositoryConflictError, match="pointer changed"):
        repository.commit(stale, (stale_chunk,), committed_at=21)


def test_s3_client_conditional_writes_and_repository_stale_writer_conflict() -> None:
    client = InMemoryConditionalObjectClient(maximum_object_bytes=64 * 1024)
    created = client.create_object("qualification/value", b"one")
    with pytest.raises(ObjectPreconditionError, match="already exists"):
        client.create_object("qualification/value", b"duplicate")
    replaced = client.replace_object("qualification/value", b"two", created.etag)
    with pytest.raises(ObjectPreconditionError, match="conditional replace"):
        client.replace_object("qualification/value", b"three", created.etag)
    assert (
        client.read_object("qualification/value", maximum_bytes=3).metadata.etag
        == replaced.etag
    )

    repository = S3ArtifactRepository(
        client,
        _object_profile(),
        "qualification/repository",
        maximum_chunk_bytes=1024,
    )
    first, first_chunk = _stage(
        repository,
        "checkpoint-a",
        "attempt-first",
        b"first",
        started_at=10,
    )
    stale, stale_chunk = _stage(
        repository,
        "checkpoint-a",
        "attempt-stale",
        b"stale",
        started_at=11,
    )
    repository.commit(first, (first_chunk,), committed_at=20)
    with pytest.raises(ObjectPreconditionError):
        repository.commit(stale, (stale_chunk,), committed_at=21)


def test_s3_metadata_guard_serializes_lease_and_garbage_collection() -> None:
    client = InMemoryConditionalObjectClient(maximum_object_bytes=64 * 1024)
    repository = S3ArtifactRepository(
        client,
        _object_profile(),
        "qualification/guarded-repository",
        maximum_chunk_bytes=1024,
    )
    transaction, chunk = _stage(
        repository,
        "checkpoint-guarded",
        "attempt-guarded",
        b"guarded-state",
        started_at=5,
    )
    repository.commit(transaction, (chunk,), committed_at=10)
    repository.set_retention(
        "checkpoint-guarded",
        RetentionPolicy(
            keep_latest_commits=0,
            minimum_age_seconds=0,
            abandoned_attempt_grace_seconds=0,
        ),
    )
    repository.tombstone(
        "checkpoint-guarded",
        "eligible",
        created_at=20,
        eligible_at=20,
    )

    guard = repository._acquire_artifact_guard("checkpoint-guarded")
    with pytest.raises(RepositoryConflictError, match="metadata is being modified"):
        repository.collect_garbage(now=30)
    assert (
        repository.read_bytes("checkpoint-guarded", "state", maximum_bytes=32)
        == b"guarded-state"
    )
    metadata = repository.artifact_guard_metadata("checkpoint-guarded")
    with pytest.raises(ValueError, match="worker-fencing evidence"):
        ArtifactGuardRecoveryAuthorization(
            repository.provider_id,
            "checkpoint-guarded",
            metadata.etag,
            "operator",
            "fence-evidence",
            30,
            worker_fenced=False,
        )
    wrong = ArtifactGuardRecoveryAuthorization(
        repository.provider_id,
        "checkpoint-guarded",
        metadata.etag + "-stale",
        "operator",
        "fence-evidence",
        30,
        worker_fenced=True,
    )
    with pytest.raises(RepositoryConflictError, match="changed after external fencing"):
        repository.recover_artifact_guard(wrong)
    authorization = ArtifactGuardRecoveryAuthorization(
        repository.provider_id,
        "checkpoint-guarded",
        metadata.etag,
        "operator",
        "fence-evidence",
        30,
        worker_fenced=True,
    )
    assert (
        ArtifactGuardRecoveryAuthorization.from_record(
            authorization.to_record()
        ).authorization_id
        == authorization.authorization_id
    )
    repository.recover_artifact_guard(authorization)

    lease = repository.acquire_lease(
        "checkpoint-guarded",
        "reader",
        lease_id="lease-guarded",
        issued_at=20,
        expires_at=40,
    )
    assert repository.collect_garbage(now=30).removed_attempt_ids == ()
    repository.release_lease(lease)
    report = repository.collect_garbage(now=31)
    assert report.removed_artifact_ids == ("checkpoint-guarded",)
    assert report.removed_attempt_ids == ("attempt-guarded",)


def test_s3_reader_uses_bounded_reads_and_detects_object_corruption() -> None:
    client = InMemoryConditionalObjectClient(maximum_object_bytes=64 * 1024)
    repository = S3ArtifactRepository(
        client,
        _object_profile(),
        "qualification/repository",
        maximum_chunk_bytes=1024,
    )
    transaction, chunk = _stage(
        repository,
        "checkpoint-a",
        "attempt-a",
        b"bounded-state",
        started_at=10,
    )
    manifest = repository.commit(transaction, (chunk,), committed_at=20)
    value = client.read_object(chunk.object_key, maximum_bytes=1024)
    client.replace_object(chunk.object_key, b"bad", value.metadata.etag)

    with pytest.raises(RepositoryCorruptionError, match="size mismatch"):
        repository.read_chunk(manifest, chunk, maximum_plaintext_bytes=32)
    assert client.read_requests
    assert all(
        maximum > 0 and maximum <= 64 * 1024 for _, maximum in client.read_requests
    )
    with pytest.raises(RepositoryCorruptionError, match="plaintext bound"):
        repository.read_chunk(manifest, chunk, maximum_plaintext_bytes=4)


def test_provider_mismatch_and_unsupported_profiles_fail_closed(
    tmp_path: Path,
) -> None:
    bad_profile = HPCFilesystemProfile(
        "posix.unsupported",
        "unknown-parallel-fs",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=False,
        advisory_locking=False,
        attempt_private_staging=True,
    )
    with pytest.raises(UnsupportedRepositoryProfileError, match="directory fsync"):
        POSIXRepositoryPolicy(bad_profile)

    first = POSIXArtifactRepository(tmp_path / "one", _posix_policy("posix.one"))
    second = POSIXArtifactRepository(tmp_path / "two", _posix_policy("posix.two"))
    transaction, chunk = _stage(
        first,
        "checkpoint-a",
        "attempt-a",
        b"state",
        started_at=10,
    )
    manifest = first.commit(transaction, (chunk,), committed_at=20)
    with pytest.raises(RepositoryConflictError, match="provider"):
        second.read_chunk(manifest, chunk)

    bad_object_profile = ObjectStoreProfile(
        "s3.unsupported",
        conditional_create=True,
        conditional_replace=False,
        strongly_consistent_reads=True,
        strongly_consistent_listing=True,
        multipart_free_objects=True,
        maximum_object_bytes=1024,
    )
    with pytest.raises(UnsupportedRepositoryProfileError, match="conditional replace"):
        S3ArtifactRepository(
            InMemoryConditionalObjectClient(maximum_object_bytes=1024),
            bad_object_profile,
            "bad/repository",
        )


def test_retention_removes_only_unreachable_history(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    repository = POSIXArtifactRepository(root, _posix_policy())
    first, first_chunk = _stage(
        repository,
        "checkpoint-a",
        "attempt-first",
        b"first",
        started_at=5,
    )
    repository.commit(first, (first_chunk,), committed_at=10)
    second, second_chunk = _stage(
        repository,
        "checkpoint-a",
        "attempt-second",
        b"second",
        started_at=15,
    )
    repository.commit(second, (second_chunk,), committed_at=20)
    repository.set_retention(
        "checkpoint-a",
        RetentionPolicy(
            keep_latest_commits=1,
            minimum_age_seconds=5,
            abandoned_attempt_grace_seconds=0,
        ),
    )

    report = repository.collect_garbage(now=30)

    assert report.removed_attempt_ids == ("attempt-first",)
    assert not (root / "roots" / "attempt-first").exists()
    assert (root / "roots" / "attempt-second").exists()
    assert repository.read_bytes("checkpoint-a", "state", maximum_bytes=16) == b"second"


def test_active_lease_and_legal_hold_pin_tombstoned_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    repository = POSIXArtifactRepository(root, _posix_policy())
    transaction, chunk = _stage(
        repository,
        "checkpoint-a",
        "attempt-a",
        b"state",
        started_at=5,
    )
    repository.commit(transaction, (chunk,), committed_at=10)
    repository.set_retention(
        "checkpoint-a",
        RetentionPolicy(
            keep_latest_commits=0,
            minimum_age_seconds=0,
            abandoned_attempt_grace_seconds=0,
        ),
    )
    lease = repository.acquire_lease(
        "checkpoint-a",
        "reader-a",
        lease_id="lease-a",
        issued_at=20,
        expires_at=40,
    )
    hold = repository.place_legal_hold(
        "checkpoint-a",
        "compliance",
        hold_id="hold-a",
        placed_at=20,
    )
    repository.tombstone(
        "checkpoint-a",
        "expired campaign",
        created_at=20,
        eligible_at=20,
    )

    assert repository.collect_garbage(now=30).removed_attempt_ids == ()
    expired = repository.collect_garbage(now=50)
    assert expired.expired_lease_ids == (lease.lease_id,)
    assert expired.removed_attempt_ids == ()
    repository.release_legal_hold(hold)
    collected = repository.collect_garbage(now=51)

    assert collected.removed_artifact_ids == ("checkpoint-a",)
    assert collected.removed_attempt_ids == ("attempt-a",)
    with pytest.raises(ObjectNotFoundError):
        repository.get_manifest("checkpoint-a")


def _support(topology: str) -> SupportTuple:
    return SupportTuple(
        "artifact.restart",
        {
            "topology": topology,
            "precision": "float64",
            "layout": "canonical-bytes",
        },
    )


def test_same_and_topology_change_restart_admission_is_exact() -> None:
    source = _support("two-shards")
    target = _support("four-shards")
    same = TopologyRestartRelation(source, source, "bitwise")
    changed = TopologyRestartRelation(
        source,
        target,
        "tolerance",
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-10,
    )
    strict = TopologyRestartPolicy(allow_topology_change=False)
    qualified = TopologyRestartPolicy(
        allow_topology_change=True,
        allow_tolerance_restart=True,
        maximum_absolute_tolerance=1.0e-12,
        maximum_relative_tolerance=1.0e-10,
    )
    unsupported = TopologyRestartRelation(
        source,
        target,
        "unsupported",
        reason="layout conversion has no qualification evidence",
    )

    assert admit_topology_restart(same, strict).admitted
    assert not admit_topology_restart(changed, strict).admitted
    assert admit_topology_restart(changed, qualified).admitted
    assert not admit_topology_restart(unsupported, qualified).admitted


def test_direct_restore_repartitions_without_execution_cache_or_global_gather() -> None:
    support = _support("canonical")
    relation = TopologyRestartRelation(support, support, "bitwise")
    policy = TopologyRestartPolicy(allow_topology_change=False)
    source_payloads = (b"abcd", b"efgh")
    sources = (
        CanonicalRestartChunk(
            "state",
            0,
            4,
            hashlib.sha256(source_payloads[0]).hexdigest(),
        ),
        CanonicalRestartChunk(
            "state",
            4,
            4,
            hashlib.sha256(source_payloads[1]).hexdigest(),
        ),
        CanonicalRestartChunk(
            "compiled-cache",
            0,
            3,
            hashlib.sha256(b"jit").hexdigest(),
            payload_class="execution-cache",
        ),
    )
    destinations = (
        DestinationShard("destination-0", "state", 0, 3),
        DestinationShard("destination-1", "state", 3, 5),
    )
    plan = prepare_direct_restore(relation, policy, sources, destinations)
    source_data = {
        plan.source_chunks[0].chunk_id: source_payloads[0],
        plan.source_chunks[1].chunk_id: source_payloads[1],
    }
    output = {item.shard_id: bytearray(item.byte_count) for item in destinations}
    requested: list[int] = []

    def read_range(chunk, offset: int, count: int) -> bytes:
        requested.append(count)
        return source_data[chunk.chunk_id][offset : offset + count]

    def write_range(shard, offset: int, payload: bytes) -> None:
        output[shard.shard_id][offset : offset + len(payload)] = payload

    report = execute_direct_restore(plan, read_range, write_range)

    assert plan.excludes_execution_cache
    assert all(item.payload_class == "restart-state" for item in plan.source_chunks)
    assert bytes(output["destination-0"]) == b"abc"
    assert bytes(output["destination-1"]) == b"defgh"
    assert report.transferred_bytes == 8
    assert max(requested) <= 4


def test_restore_plan_rejects_chunk_holes_and_mapping_overlaps() -> None:
    support = _support("canonical")
    relation = TopologyRestartRelation(support, support, "bitwise")
    policy = TopologyRestartPolicy(allow_topology_change=False)
    admission = admit_topology_restart(relation, policy)
    first = CanonicalRestartChunk("state", 0, 4, hashlib.sha256(b"abcd").hexdigest())
    hole = CanonicalRestartChunk("state", 5, 3, hashlib.sha256(b"fgh").hexdigest())
    destination = DestinationShard("destination", "state", 0, 8)
    with pytest.raises(ValueError, match="hole"):
        prepare_direct_restore(
            relation,
            policy,
            (first, hole),
            (destination,),
        )

    second = CanonicalRestartChunk("state", 4, 4, hashlib.sha256(b"efgh").hexdigest())
    overlapping = (
        RestartChunkMapping(first.chunk_id, 0, "destination", 0, 4),
        RestartChunkMapping(second.chunk_id, 0, "destination", 3, 4),
    )
    with pytest.raises(ValueError, match="canonical payload coordinates|overlap"):
        DirectRestorePlan(
            relation,
            admission,
            (first, second),
            (destination,),
            overlapping,
        )
