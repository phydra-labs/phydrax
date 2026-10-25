#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Root-group stage transport: credits, backpressure, atomic transfer, consensus.

Single-process cases run the identical three-collective schedule over the
live root group; the two-process probe exercises real cross-process subgroup
transport and fail-coupled rank loss through the local launcher.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

import phydrax.atomistic._stage_transport as stage_transport
from phydrax import config as pxconfig
from phydrax._execution_launch import (
    launch_local_processes,
    LocalProcessLaunchError,
    LocalProcessLaunchPlan,
)
from phydrax._execution_runtime import ExecutionRuntime
from phydrax.atomistic._campaign import AtomisticTaskPacket
from phydrax.atomistic._stage_transport import (
    _ProbeReport,
    AtomisticPacketCodec,
    AtomisticStageTransportPlan,
    AtomisticTransportCapacityError,
    AtomisticTransportClosedError,
    AtomisticTransportEdge,
    AtomisticTransportSchemaError,
    AtomisticTransportStage,
    PreparedAtomisticStageTransport,
    process_subset_group_spec,
)
from phydrax.lifecycle._distributed_checkpoint import (
    IncompleteDistributedCheckpointError,
    load_distributed_checkpoint_commit,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)


def _packet(value: float, *, child: str | None = None) -> AtomisticTaskPacket:
    return AtomisticTaskPacket(
        positions=np.arange(6, dtype=np.float64).reshape(2, 3) * value,
        species=np.asarray([1, 8], np.int32),
        atom_mask=np.ones((2,), np.bool_),
        time=value,
        child_id=child,
        family=None if child is None else "family-a",
    )


def _prepared(
    *,
    outbox: int = 3,
    inbox: int = 1,
    slots: int = 1,
    rounds: int = 32,
    sinks: tuple[str, ...] = (),
) -> PreparedAtomisticStageTransport:
    if not sys.platform.startswith("linux"):
        pytest.skip("requires native process host mapping evidence")
    root = ExecutionRuntime.current().root_group
    group = process_subset_group_spec(root.spec, (0,))
    edges = [AtomisticTransportEdge("produce", "consume", _packet(1.0))]
    edges += [AtomisticTransportEdge("consume", sink, _packet(1.0)) for sink in sinks]
    plan = AtomisticStageTransportPlan(
        (
            AtomisticTransportStage("produce", group),
            AtomisticTransportStage("consume", group),
        ),
        tuple(edges),
        sinks=sinks,
        outbox_capacity=outbox,
        inbox_capacity=inbox,
        lane_slots=slots,
        maximum_rounds=rounds,
    )
    return plan.prepare(root, maximum_round_bytes=1 << 26)


def _same(left: AtomisticTaskPacket, right: AtomisticTaskPacket) -> None:
    np.testing.assert_array_equal(left.positions, right.positions)
    np.testing.assert_array_equal(left.species, right.species)
    np.testing.assert_array_equal(left.atom_mask, right.atom_mask)
    np.testing.assert_array_equal(left.time, right.time)
    assert (left.child_id, left.family) == (right.child_id, right.family)


def test_packet_codec_roundtrips_identity_and_refuses_schema_changes() -> None:
    codec = AtomisticPacketCodec(_packet(1.0))
    packet = _packet(2.5, child="child-1")
    encoded = codec.encode(packet)
    assert encoded.shape == (codec.byte_count,)
    _same(codec.decode(encoded), packet)
    with pytest.raises(AtomisticTransportSchemaError):
        codec.encode(AtomisticTaskPacket(positions=np.zeros((3, 3)), time=0.0))
    with pytest.raises(AtomisticTransportSchemaError):
        codec.decode(encoded[:-1])


def test_credit_saturation_parks_packets_bitwise_without_loss() -> None:
    prepared = _prepared(outbox=3, inbox=1)
    state = prepared.initial_state()
    packets = [_packet(value) for value in (1.0, 2.0, 3.0)]
    for packet in packets:
        state = prepared.deposit(state, "produce", packet)
    first = prepared.round(state, drained=False)
    assert (first.evidence.granted, first.evidence.committed) == (1, 1)
    assert first.evidence.parked == 2
    state = first.state
    assert prepared.inbox_count(state, "consume") == 1
    parked_header = state.outbox_header.copy()
    parked_bytes = state.outbox_bytes.copy()
    # The consumer has no credit: nothing moves and parked packets stay intact.
    for _ in range(2):
        saturated = prepared.round(state, drained=False)
        assert saturated.evidence.credits[1] == 0
        assert (saturated.evidence.granted, saturated.evidence.parked) == (0, 2)
        state = saturated.state
        np.testing.assert_array_equal(state.outbox_header, parked_header)
        np.testing.assert_array_equal(state.outbox_bytes, parked_bytes)
    delivered = []
    while len(delivered) < 3:
        state, delivery = prepared.receive(state, "consume")
        if delivery is not None:
            delivered.append(delivery)
        state = prepared.round(state, drained=False).state
    assert [delivery.sequence for delivery in delivered] == [0, 1, 2]
    for delivery, packet in zip(delivered, packets, strict=True):
        assert delivery.source == "produce"
        _same(delivery.fragments[0], packet)
    assert prepared.outbox_free(state, "produce") == 3


def test_full_outbox_refuses_without_overwriting_parked_packets() -> None:
    prepared = _prepared(outbox=2)
    state = prepared.deposit(prepared.initial_state(), "produce", _packet(1.0))
    state = prepared.deposit(state, "produce", _packet(2.0))
    assert prepared.outbox_free(state, "produce") == 0
    with pytest.raises(AtomisticTransportCapacityError, match="parked"):
        prepared.deposit(state, "produce", _packet(3.0))
    codec = prepared.codec(0)
    _same(codec.decode(state.outbox_bytes[0, 0, : codec.byte_count]), _packet(1.0))
    _same(codec.decode(state.outbox_bytes[0, 1, : codec.byte_count]), _packet(2.0))


def test_group_transfer_is_atomic_when_any_fragment_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(outbox=2, inbox=2)
    state = prepared.deposit(prepared.initial_state(), "produce", _packet(1.0))
    before = {
        name: np.array(getattr(state, name), copy=True)
        for name in ("outbox_header", "outbox_bytes", "inbox_header", "inbox_bytes")
    }
    monkeypatch.setattr(stage_transport, "_verify_fragment", lambda *args: False)
    failed = prepared.round(state, drained=False)
    assert (failed.evidence.granted, failed.evidence.committed) == (1, 0)
    assert failed.evidence.refused == 1
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(failed.state, name), value)
    monkeypatch.undo()
    committed = prepared.round(failed.state, drained=False)
    assert committed.evidence.committed == 1
    _, delivery = prepared.receive(committed.state, "consume")
    assert delivery is not None
    _same(delivery.fragments[0], _packet(1.0))


def test_refused_first_grant_is_never_overtaken_on_its_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(outbox=2, inbox=2, slots=2)
    state = prepared.deposit(prepared.initial_state(), "produce", _packet(1.0))
    state = prepared.deposit(state, "produce", _packet(2.0))
    verify = stage_transport._verify_fragment
    # Only the first packet (sequence 0) fails verification.
    monkeypatch.setattr(
        stage_transport,
        "_verify_fragment",
        lambda header, payload, expected: (
            expected[1] != 0 and verify(header, payload, expected)
        ),
    )
    failed = prepared.round(state, drained=False)
    assert (failed.evidence.granted, failed.evidence.committed) == (2, 0)
    assert prepared.inbox_count(failed.state, "consume") == 0
    np.testing.assert_array_equal(failed.state.outbox_header, state.outbox_header)
    np.testing.assert_array_equal(failed.state.outbox_bytes, state.outbox_bytes)
    monkeypatch.undo()
    committed = prepared.round(failed.state, drained=False)
    assert committed.evidence.committed == 2
    following = committed.state
    sequences = []
    for _ in range(2):
        following, delivery = prepared.receive(following, "consume")
        assert delivery is not None
        sequences.append(delivery.sequence)
    assert sequences == [0, 1]


def test_sink_refusal_blocks_later_publications_of_the_edge() -> None:
    prepared = _prepared(outbox=2, inbox=1, slots=2, sinks=("sink",))

    class _FirstRefusingSink(_Sink):
        def publish(self, key: str, packet: AtomisticTaskPacket, /) -> str | None:
            if float(packet.time) == 1.0 and not self.accepting:
                return None
            self.records[key] = f"receipt:{float(packet.time)}"
            return self.records[key]

    sink = _FirstRefusingSink()
    state = prepared.deposit(
        prepared.initial_state(), "consume", _packet(1.0), destination="sink"
    )
    state = prepared.deposit(state, "consume", _packet(2.0), destination="sink")
    refused = prepared.round(state, drained=False, sinks={"sink": sink})
    assert (refused.evidence.granted, refused.evidence.committed) == (2, 0)
    assert sink.records == {}
    sink.accepting = True
    published = prepared.round(refused.state, drained=False, sinks={"sink": sink})
    assert published.evidence.committed == 2
    assert published.evidence.receipts == ("receipt:1.0", "receipt:2.0")


def test_empty_rounds_follow_the_schedule_and_terminate_by_consensus() -> None:
    prepared = _prepared(rounds=3)
    state = prepared.initial_state()
    idle = prepared.round(state, drained=False)
    assert (idle.evidence.granted, idle.evidence.terminated) == (0, False)
    assert int(idle.state.round_index) == 1
    assert idle.state.transfer_digest.tobytes() != state.transfer_digest.tobytes()
    done = prepared.round(idle.state, drained=True)
    assert done.evidence.terminated and done.state.terminated
    with pytest.raises(AtomisticTransportClosedError):
        prepared.round(done.state, drained=True)
    # Pending work blocks termination even when every driver is drained.
    pending = prepared.deposit(prepared.initial_state(), "produce", _packet(1.0))
    assert not prepared.round(pending, drained=True).evidence.terminated


def test_cancellation_consensus_closes_the_schedule() -> None:
    prepared = _prepared()
    state = prepared.deposit(prepared.initial_state(), "produce", _packet(1.0))
    cancelled = prepared.round(state, drained=False, cancel=True)
    assert cancelled.evidence.cancelled and cancelled.evidence.granted == 0
    np.testing.assert_array_equal(cancelled.state.outbox_bytes, state.outbox_bytes)
    with pytest.raises(AtomisticTransportClosedError):
        prepared.deposit(cancelled.state, "produce", _packet(2.0))


def test_round_bound_is_a_capacity_refusal() -> None:
    prepared = _prepared(rounds=1)
    state = prepared.round(prepared.initial_state(), drained=False).state
    with pytest.raises(AtomisticTransportCapacityError, match="round"):
        prepared.round(state, drained=False)


class _Sink:
    sink_id = "sink-test"

    def __init__(self) -> None:
        self.accepting = False
        self.records: dict[str, str] = {}

    def publish(self, key: str, packet: AtomisticTaskPacket, /) -> str | None:
        if not self.accepting:
            return None
        self.records[key] = f"receipt:{float(packet.time)}"
        return self.records[key]

    def receipt(self, key: str, /) -> str | None:
        return self.records.get(key)


def test_sink_backpressure_parks_and_durable_receipts_commit() -> None:
    prepared = _prepared(outbox=2, inbox=1, sinks=("sink",))
    sink = _Sink()
    state = prepared.deposit(
        prepared.initial_state(), "consume", _packet(4.0), destination="sink"
    )
    parked = prepared.round(state, drained=False, sinks={"sink": sink})
    assert (parked.evidence.granted, parked.evidence.committed) == (1, 0)
    np.testing.assert_array_equal(parked.state.outbox_bytes, state.outbox_bytes)
    sink.accepting = True
    published = prepared.round(parked.state, drained=False, sinks={"sink": sink})
    assert published.evidence.committed == 1
    assert published.evidence.receipts == ("receipt:4.0",)
    assert int(published.state.published[0]) == 1
    assert prepared.outbox_free(published.state, "consume") == 2
    with pytest.raises(ValueError, match="sink host"):
        prepared.round(published.state, drained=False)


def test_stage_groups_are_explicit_root_children() -> None:
    prepared = _prepared()
    assert prepared.is_member("produce")
    group = prepared.stage_group("produce")
    assert group.spec.parent_group_id == prepared.root.spec.group_id
    with pytest.raises(KeyError):
        prepared.stage_group("unknown")
    root = prepared.root.spec
    with pytest.raises(ValueError, match="subset"):
        process_subset_group_spec(root, (max(root.process_indices) + 1,))


def _plan(edge: AtomisticTransportEdge) -> AtomisticStageTransportPlan:
    root = ExecutionRuntime.current().root_group
    group = process_subset_group_spec(root.spec, (0,))
    return AtomisticStageTransportPlan(
        (
            AtomisticTransportStage("produce", group),
            AtomisticTransportStage("consume", group),
        ),
        (edge,),
        outbox_capacity=1,
        inbox_capacity=1,
        lane_slots=1,
        maximum_rounds=4,
    )


def test_declared_edge_layouts_agree_schemas_over_the_root_group() -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("requires native process host mapping evidence")
    root = ExecutionRuntime.current().root_group
    templated = AtomisticTransportEdge(
        "produce", "consume", _packet(1.0), layout_id="declared-layout"
    )
    declared = AtomisticTransportEdge(
        "produce", "consume", None, layout_id="declared-layout"
    )
    # The edge identity is the declared layout on every process, with or
    # without the endpoint's template.
    assert templated.edge_id == declared.edge_id
    with pytest.raises(ValueError, match="layout_id"):
        AtomisticTransportEdge("produce", "consume", None)
    prepared = _plan(templated).prepare(root, maximum_round_bytes=1 << 26)
    codec = templated.codec
    assert codec is not None
    assert prepared.codec(0).schema_id == codec.schema_id
    assert prepared.slot_bytes == codec.byte_count
    # This process is an endpoint: holding no schema is refused by consensus.
    with pytest.raises(AtomisticTransportSchemaError, match="hold no packet schema"):
        _plan(declared).prepare(root, maximum_round_bytes=1 << 26)


def test_domain_continuations_carry_their_task_identity_through_the_codec() -> None:
    from phydrax.atomistic._domain_runtime import AtomisticDomainPacket
    from tests._support.atomistic_domain import domain_training_fixture

    runtime, build, _, _, positions, cell = domain_training_fixture(2)
    first = build(positions, cell)
    template = runtime.to_packet(first)
    codec = AtomisticPacketCodec(AtomisticTaskPacket(continuation=template, family="f"))
    moved = runtime.with_dynamics(
        first, first.positions, first.momenta, step_index=3, time=1.5
    )
    packet = runtime.to_packet(moved)
    # Another task state of the same schema: only its identity differs.
    assert packet.header_id != template.header_id
    encoded = codec.encode(AtomisticTaskPacket(continuation=packet, family="f"))
    decoded = codec.decode(encoded).continuation
    assert isinstance(decoded, AtomisticDomainPacket)
    assert (decoded.header_id, decoded.run_id, decoded.logical_count) == (
        packet.header_id,
        packet.run_id,
        packet.logical_count,
    )
    for name, value in packet.rows.items():
        np.testing.assert_array_equal(decoded.rows[name], value)
    restored = runtime.from_packet((decoded,))
    assert int(np.asarray(restored.step_index)) == 3
    # Re-fragmenting keeps every active row exactly once (compacted).
    merged = runtime.merge_packets((packet.without_rows(), packet))
    active = packet.rows["active"]
    assert int(merged.rows["active"].sum()) == int(active.sum())
    np.testing.assert_array_equal(
        np.sort(merged.rows["stable_ids"][merged.rows["active"]]),
        np.sort(packet.rows["stable_ids"][active]),
    )
    # A non-domain packet cannot claim a continuation identity.
    forged = encoded.copy()
    header = json.dumps(
        {"child_id": None, "continuation": None, "family": "f"},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    forged[:4] = np.asarray([len(header)], dtype="<u4").view(np.uint8)
    forged[4 : 4 + len(header)] = np.frombuffer(header, dtype=np.uint8)
    with pytest.raises(AtomisticTransportSchemaError, match="identity"):
        codec.decode(forged)


def _repository(path: Path) -> POSIXArtifactRepository:
    profile = HPCFilesystemProfile(
        "posix.stage-transport-probe",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXArtifactRepository(
        path,
        POSIXRepositoryPolicy(
            profile, maximum_chunk_bytes=1 << 16, maximum_metadata_bytes=1 << 20
        ),
    )


def _launch_probe(directory: Path, scenario: str) -> list[_ProbeReport]:
    root = Path(__file__).parents[3]
    environment = pxconfig.environment(
        pxconfig.RuntimeConfig(
            platforms=("cpu",), cpu_device_count=2, cpu_collectives_implementation="gloo"
        )
    )
    environment.update(
        PYTHONPATH=str(root),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    results = launch_local_processes(
        (
            sys.executable,
            "-c",
            "import sys; from phydrax.atomistic._stage_transport import "
            "stage_transport_probe_rank; stage_transport_probe_rank(sys.argv[1], sys.argv[2])",
            str(directory),
            scenario,
        ),
        LocalProcessLaunchPlan(2, f"127.0.0.1:{port}", devices_per_process=2),
        environment=environment,
        cwd=str(root),
        timeout_seconds=240.0,
    )
    return [
        json.loads(result.stdout.decode().strip().splitlines()[-1]) for result in results
    ]


@pytest.mark.skipif(
    os.name != "posix" or not sys.platform.startswith("linux"),
    reason="requires POSIX process groups and native host mapping evidence",
)
def test_two_process_probe_transports_through_subgroups(tmp_path: Path) -> None:
    reports = _launch_probe(tmp_path, "complete")
    producer, consumer = reports
    assert producer["physics"] == [float(np.arange(12).sum() * 2 * v) for v in (1, 2, 3)]
    assert consumer["received"] == producer["physics"]
    assert producer["rounds"] == consumer["rounds"]
    parked = [entry["parked"] for entry in producer["rounds"]]
    assert parked[0] == 2 and parked[-1] == 0
    assert producer["transfer_digest"] == consumer["transfer_digest"]
    assert producer["manifest_id"] == consumer["manifest_id"]
    assert producer["restored_bitwise"] and consumer["restored_bitwise"]


@pytest.mark.skipif(
    os.name != "posix" or not sys.platform.startswith("linux"),
    reason="requires POSIX process groups and native host mapping evidence",
)
def test_killed_rank_fails_the_coupled_job_without_a_global_commit(
    tmp_path: Path,
) -> None:
    with pytest.raises(LocalProcessLaunchError):
        _launch_probe(tmp_path, "kill")
    repository = _repository(tmp_path / "repository")
    with pytest.raises(IncompleteDistributedCheckpointError):
        load_distributed_checkpoint_commit(repository, "stage-transport-probe-kill")
