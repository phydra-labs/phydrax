#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax.lifecycle._chunk_repository import RepositoryConflictError
from phydrax.lifecycle._event_graph_repository import (
    checkpoint_content_id,
    deterministic_commit_owner,
    EventGraphEpochManifest,
    EventGraphRepository,
    GlobalEntity,
    GlobalEvent,
    GlobalEventEdge,
    GlobalWorkItem,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)


class SimulatedCrash(RuntimeError):
    pass


def _digest(name):
    return canonical_fingerprint({"test": name})


def _repository(path, *, fail=None):
    profile = HPCFilesystemProfile(
        "dark-sector-test-posix",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    artifact_repository = POSIXArtifactRepository(
        path,
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=256,
            maximum_metadata_bytes=64 * 1024,
        ),
    )
    return EventGraphRepository(
        artifact_repository,
        maximum_record_bytes=1024 * 1024,
        failure_injector=fail,
    )


def _graph(repository):
    source = GlobalEntity(
        "particle",
        "dark-fermion",
        _digest("source-state"),
        frame_id=_digest("frame"),
        frame_realization_id=_digest("frame-realization"),
        unit_contract_id=_digest("units"),
        rights_id="native",
        provenance_ids=("generator-a",),
    )
    middle = GlobalEntity(
        "particle",
        "dark-vector",
        _digest("middle-state"),
        frame_id=_digest("frame"),
        frame_realization_id=_digest("frame-realization"),
        unit_contract_id=_digest("units"),
        rights_id="native",
        provenance_ids=("generator-a",),
    )
    target = GlobalEntity(
        "particle",
        "dark-fermion",
        _digest("target-state"),
        frame_id=_digest("frame"),
        frame_realization_id=_digest("frame-realization"),
        unit_contract_id=_digest("units"),
        rights_id="native",
        provenance_ids=("generator-a",),
    )
    for entity in (source, middle, target):
        repository.put_entity(entity, "writer-a", committed_at=1)
    first = GlobalEvent(
        "split",
        _digest("model"),
        (source.entity_id,),
        (middle.entity_id,),
        epoch_sequence=0,
    )
    repository.put_event(first, "writer-a", committed_at=1)
    second = GlobalEvent(
        "convert",
        _digest("model"),
        (middle.entity_id,),
        (target.entity_id,),
        epoch_sequence=0,
        parent_event_ids=(first.event_id,),
    )
    repository.put_event(second, "writer-a", committed_at=1)
    edge = GlobalEventEdge(
        first.event_id,
        second.event_id,
        middle.entity_id,
        "produced-consumed",
    )
    repository.put_edge(edge, "writer-a", committed_at=1)
    work = GlobalWorkItem(
        "transport",
        (target.entity_id,),
        _digest("model"),
        partition_key="shard-0",
    )
    repository.put_work(work, "writer-a", committed_at=1)
    return (source, middle, target), (first, second), edge, work


def _manifest(graph, checkpoint, *, evidence=()):
    entities, events, edge, work = graph
    return EventGraphEpochManifest(
        "run-a",
        0,
        None,
        _digest("plan"),
        _digest("compile"),
        _digest("capacity"),
        _digest("species"),
        _digest("topology"),
        entity_ids=tuple(entity.entity_id for entity in entities),
        event_ids=tuple(event.event_id for event in events),
        edge_ids=(edge.edge_id,),
        work_ids=(work.work_id,),
        deferred_work_ids=(work.work_id,),
        matrix_element_revision_id=_digest("matrix-element"),
        checkpoint_id=checkpoint_content_id(checkpoint),
        commit_owner_id="writer-a",
        conservation_status="conserved",
        evidence_ids=evidence,
    )


def test_content_addressed_graph_is_immutable_acyclic_and_exactly_once(tmp_path):
    repository = _repository(tmp_path / "repository")
    graph = _graph(repository)
    entities, events, _, _ = graph
    checkpoint = b"fixed-capacity-state"
    manifest = _manifest(graph, checkpoint)
    first = repository.append_epoch(
        manifest, checkpoint, writer_id="writer-a", committed_at=2
    )
    repeated = repository.append_epoch(
        manifest, checkpoint, writer_id="writer-a", committed_at=2
    )
    assert repeated.tip.tip_id == first.tip.tip_id
    assert (
        repository.load_epoch(manifest.epoch_manifest_id).checkpoint_payload == checkpoint
    )

    with pytest.raises(ValueError, match="flow|lineage"):
        repository.put_edge(
            GlobalEventEdge(
                events[1].event_id,
                events[0].event_id,
                entities[1].entity_id,
                "reverse",
            ),
            "writer-a",
            committed_at=3,
        )
    conflicting = _manifest(graph, b"different", evidence=("different-evidence",))
    with pytest.raises(RepositoryConflictError):
        repository.append_epoch(
            conflicting, b"different", writer_id="writer-a", committed_at=3
        )


def test_crash_before_and_after_tip_have_deterministic_recovery(tmp_path):
    points = {"after_epoch_manifest", "after_tip"}
    for point in points:
        raised = False

        def fail(actual):
            nonlocal raised
            if actual == point and not raised:
                raised = True
                raise SimulatedCrash(actual)

        repository = _repository(tmp_path / point, fail=fail)
        graph = _graph(repository)
        checkpoint = point.encode()
        manifest = _manifest(graph, checkpoint)
        with pytest.raises(SimulatedCrash):
            repository.append_epoch(
                manifest, checkpoint, writer_id="writer-a", committed_at=2
            )
        receipt = repository.append_epoch(
            manifest, checkpoint, writer_id="writer-a", committed_at=2
        )
        assert receipt.tip.epoch_manifest_id == manifest.epoch_manifest_id
        assert repository.run_tip("run-a").tip_id == receipt.tip.tip_id


def test_expired_work_lease_is_stolen_by_deterministic_owner(tmp_path):
    repository = _repository(tmp_path / "leases")
    _entities, _events, _edge, work = _graph(repository)
    workers = ("worker-a", "worker-b")
    owner = deterministic_commit_owner(work.work_id, workers)
    first = repository.acquire_work_lease(
        work,
        owner,
        eligible_worker_ids=workers,
        issued_at=1,
        expires_at=10,
    )
    assert (
        repository.acquire_work_lease(
            work,
            owner,
            eligible_worker_ids=workers,
            issued_at=5,
            expires_at=20,
        ).lease_id
        == first.lease_id
    )
    thief = "worker-b" if owner == "worker-a" else "worker-a"
    stolen = repository.acquire_work_lease(
        work,
        thief,
        eligible_worker_ids=(thief,),
        issued_at=10,
        expires_at=20,
    )
    assert stolen.generation == 1
    assert stolen.previous_lease_id == first.lease_id


def test_collection_tombstones_only_proven_unreachable_records(tmp_path):
    repository = _repository(tmp_path / "gc")
    graph = _graph(repository)
    checkpoint = b"reachable"
    manifest = _manifest(graph, checkpoint)
    repository.append_epoch(manifest, checkpoint, writer_id="writer-a", committed_at=2)
    orphan = GlobalEntity(
        "particle",
        "orphan",
        _digest("orphan-state"),
        frame_id=_digest("frame"),
        frame_realization_id=_digest("frame-realization"),
        unit_contract_id=_digest("units"),
        rights_id="native",
    )
    repository.put_entity(orphan, "writer-a", committed_at=3)
    report = repository.collect_unreachable(
        ("run-a",),
        entity_ids=(graph[0][0].entity_id, orphan.entity_id),
        reason="reachability",
        now=4,
    )
    assert report.tombstoned_artifact_ids == (f"dark-sector.entity.{orphan.entity_id}",)
