#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import FrozenInstanceError

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import FiniteVolumePrecisionPolicy
from phydrax.solver import (
    FiniteVolumeConservativeContentState,
    FiniteVolumeRuntimeState,
    PreparedFiniteVolumeRuntime,
)
from phydrax.solver._finite_volume_topology_events import (
    FiniteVolumeTopologyEpoch,
    FiniteVolumeTopologyEvent,
    FiniteVolumeTopologyEventJournal,
    FiniteVolumeTopologyEventRequest,
    FiniteVolumeTopologyEventScheduler,
    TopologyEventKind,
    TopologyEventState,
    TopologyEventStatus,
)


def _epoch(
    name: str,
    *,
    parent_epoch_id: str | None = None,
) -> FiniteVolumeTopologyEpoch:
    return FiniteVolumeTopologyEpoch(
        f"prepared-{name}",
        f"topology-{name}",
        f"geometry-{name}",
        parent_epoch_id=parent_epoch_id,
        topology_artifact_id=f"topology-artifact-{name}",
        metrics_artifact_id=f"metrics-artifact-{name}",
        operators_artifact_id=f"operators-artifact-{name}",
    )


def _request(
    epoch: FiniteVolumeTopologyEpoch,
    name: str = "one",
    *,
    kind: TopologyEventKind = TopologyEventKind.REMESH,
    payload_id: str | None = "request-payload",
) -> FiniteVolumeTopologyEventRequest:
    return FiniteVolumeTopologyEventRequest(
        kind,
        epoch.epoch_id,
        f"requested-spec-{name}",
        payload_id=payload_id,
        reason=f"reason-{name}",
    )


def test_topology_event_requested_then_committed_publishes_one_epoch():
    initial = _epoch("initial")
    requested = FiniteVolumeTopologyEventJournal.allocate(
        initial, capacity=3, time=0.0
    ).append_requested(_request(initial), 4, 1.25)
    pending_event = requested.event(0)
    with pytest.raises(ValueError, match="wrong input epoch"):
        requested.commit(0, _epoch("wrong-parent"))
    assert requested.event(0) == pending_event

    result = _epoch("result", parent_epoch_id=initial.epoch_id)
    committed = requested.commit(0, result, payload_id="committed-payload")
    event = committed.event(0)
    assert pending_event.state is TopologyEventState.REQUESTED
    assert pending_event.status is TopologyEventStatus.PENDING
    assert event.sequence == 0
    assert event.state is TopologyEventState.COMMITTED
    assert event.status is TopologyEventStatus.SUCCESS
    assert event.result_id == result.epoch_id
    assert event.payload_id == "committed-payload"
    assert committed.current_epoch_id == result.epoch_id
    assert committed.epoch_table == (initial, result)
    assert int(committed.count) == 1
    assert int(committed.next_sequence) == 1
    assert requested.current_epoch_id == initial.epoch_id
    with pytest.raises(ValueError, match="no longer requested"):
        committed.fail(0)
    assert requested.event(0) == pending_event


def test_topology_event_requested_then_failed_keeps_current_epoch():
    initial = _epoch("initial")
    requested = FiniteVolumeTopologyEventJournal(initial, capacity=2).append_requested(
        _request(initial, kind=TopologyEventKind.AMR_REGRID), 2, 0.5
    )
    failed = requested.fail(
        0,
        result_id="failure-evidence",
        payload_id="failure-payload",
    )

    event = failed.event(0)
    assert event.state is TopologyEventState.FAILED
    assert event.status is TopologyEventStatus.FAILED
    assert event.result_id == "failure-evidence"
    assert event.payload_id == "failure-payload"
    assert failed.current_epoch_id == initial.epoch_id
    assert failed.epoch_table == (initial,)
    with pytest.raises(ValueError, match="no longer requested"):
        failed.commit(0, _epoch("late", parent_epoch_id=initial.epoch_id))
    with pytest.raises(ValueError, match="no longer requested"):
        failed.fail(0)
    assert requested.event(0).state is TopologyEventState.REQUESTED


def test_topology_event_schema_rejects_illegal_states_and_enum_values():
    initial = _epoch("initial")
    request = _request(initial)
    with pytest.raises(ValueError, match="valid TopologyEventKind"):
        FiniteVolumeTopologyEventRequest(
            99,
            initial.epoch_id,
            "requested-spec",
        )
    with pytest.raises(TypeError, match="must be TopologyEventKind"):
        FiniteVolumeTopologyEventRequest(
            True,
            initial.epoch_id,
            "requested-spec",
        )
    with pytest.raises(ValueError, match="committed event"):
        FiniteVolumeTopologyEvent(
            0,
            1,
            0.25,
            TopologyEventKind.REMESH,
            TopologyEventState.COMMITTED,
            TopologyEventStatus.PENDING,
            request.request_id,
            initial.epoch_id,
            "result",
            request.payload_id,
        )
    with pytest.raises(ValueError, match="requested event"):
        FiniteVolumeTopologyEvent(
            0,
            1,
            0.25,
            TopologyEventKind.REMESH,
            TopologyEventState.REQUESTED,
            TopologyEventStatus.PENDING,
            request.request_id,
            initial.epoch_id,
            "unexpected-result",
            request.payload_id,
        )


def test_topology_event_journal_capacity_overflow_is_sticky_and_nonmutating():
    initial = _epoch("initial")
    journal = FiniteVolumeTopologyEventJournal.allocate(initial, capacity=1)
    requested = journal.append_requested(_request(initial), 1, 0.25)
    full = requested.fail(0, result_id="capacity-filled")
    overflowed = full.append_requested(_request(initial, "two"), 2, 0.5)

    assert bool(overflowed.overflowed)
    assert int(overflowed.count) == 1
    assert int(overflowed.next_sequence) == 1
    assert overflowed.event(0) == full.event(0)
    assert not bool(full.overflowed)
    repeated = overflowed.append_requested(_request(initial, "three"), 3, 0.75)
    assert bool(repeated.overflowed)
    assert repeated.journal_id == overflowed.journal_id
    assert overflowed.journal_id != full.journal_id


def test_topology_event_journal_rejects_stale_and_parallel_requests():
    initial = _epoch("initial")
    journal = FiniteVolumeTopologyEventJournal.allocate(initial, capacity=3)
    stale = _epoch("stale")
    with pytest.raises(ValueError, match="input epoch is stale"):
        journal.append_requested(_request(stale), 1, 0.25)

    requested = journal.append_requested(_request(initial, "first"), 1, 0.25)
    with pytest.raises(ValueError, match="already has a pending request"):
        requested.append_requested(_request(initial, "parallel"), 1, 0.25)

    result = _epoch("result", parent_epoch_id=initial.epoch_id)
    advanced = requested.commit(0, result)
    with pytest.raises(ValueError, match="input epoch is stale"):
        advanced.append_requested(_request(initial, "stale-after-commit"), 2, 0.5)
    next_request = advanced.append_requested(_request(result, "next"), 2, 0.5)
    assert next_request.event(1).input_epoch_id == result.epoch_id


def test_topology_event_sequence_steps_and_times_are_monotone():
    initial = _epoch("initial")
    journal = FiniteVolumeTopologyEventJournal.allocate(initial, capacity=3)
    first_requested = journal.append_requested(_request(initial, "first"), 5, 1.0)
    first = first_requested.fail(0)
    second_requested = first.append_requested(_request(initial, "second"), 5, 1.0)
    second = second_requested.fail(1)

    assert first.event(0).sequence == 0
    assert second.event(1).sequence == 1
    with pytest.raises(ValueError, match="accepted steps must be monotone"):
        second.append_requested(_request(initial, "old-step"), 4, 1.5)
    with pytest.raises(ValueError, match="times must be monotone"):
        second.append_requested(_request(initial, "old-time"), 6, 0.5)
    with pytest.raises(IndexError, match="unrequested"):
        second.commit(2, _epoch("missing", parent_epoch_id=initial.epoch_id))


def test_topology_content_identities_cover_all_static_content():
    initial = _epoch("initial")
    repeated = _epoch("initial")
    changed_geometry = FiniteVolumeTopologyEpoch(
        initial.prepared_id,
        initial.topology_id,
        "different-geometry",
        topology_artifact_id=initial.topology_artifact_id,
        metrics_artifact_id=initial.metrics_artifact_id,
        operators_artifact_id=initial.operators_artifact_id,
    )
    request = _request(initial)
    repeated_request = _request(initial)
    changed_request = _request(initial, payload_id="different-payload")

    assert initial.epoch_id == repeated.epoch_id
    assert initial.epoch_id != changed_geometry.epoch_id
    assert request.request_id == repeated_request.request_id
    assert request.request_id != changed_request.request_id

    requested = FiniteVolumeTopologyEventJournal.allocate(
        initial, capacity=2
    ).append_requested(request, 1, 0.25)
    failed_a = requested.fail(0, result_id="failure-a")
    failed_b = requested.fail(0, result_id="failure-b")
    assert requested.requested_ids[0] == request.request_id
    assert requested.journal_id != failed_a.journal_id
    assert failed_a.journal_id != failed_b.journal_id
    assert requested.event(0).event_id != failed_a.event(0).event_id

    float16_journal = FiniteVolumeTopologyEventJournal.allocate(
        initial, capacity=2, time=np.asarray(0.0, dtype=np.float16)
    )
    float32_journal = FiniteVolumeTopologyEventJournal.allocate(
        initial, capacity=2, time=np.asarray(0.0, dtype=np.float32)
    )
    replayed_float16 = FiniteVolumeTopologyEventJournal.from_events(
        initial,
        (),
        capacity=2,
        time=np.asarray(0.0, dtype=np.float16),
    )
    assert float16_journal.journal_id != float32_journal.journal_id
    assert replayed_float16.journal_id == float16_journal.journal_id

    event_variants = (
        FiniteVolumeTopologyEvent(
            0,
            1,
            0.25,
            TopologyEventKind.REMESH,
            TopologyEventState.REQUESTED,
            TopologyEventStatus.PENDING,
            "request",
            initial.epoch_id,
            None,
            "payload",
        ),
        FiniteVolumeTopologyEvent(
            1,
            1,
            0.25,
            TopologyEventKind.REMESH,
            TopologyEventState.REQUESTED,
            TopologyEventStatus.PENDING,
            "request",
            initial.epoch_id,
            None,
            "payload",
        ),
        FiniteVolumeTopologyEvent(
            0,
            2,
            0.25,
            TopologyEventKind.REMESH,
            TopologyEventState.REQUESTED,
            TopologyEventStatus.PENDING,
            "request",
            initial.epoch_id,
            None,
            "payload",
        ),
        FiniteVolumeTopologyEvent(
            0,
            1,
            0.5,
            TopologyEventKind.REMESH,
            TopologyEventState.REQUESTED,
            TopologyEventStatus.PENDING,
            "request",
            initial.epoch_id,
            None,
            "payload",
        ),
        FiniteVolumeTopologyEvent(
            0,
            1,
            0.25,
            TopologyEventKind.AMR_REGRID,
            TopologyEventState.REQUESTED,
            TopologyEventStatus.PENDING,
            "request",
            initial.epoch_id,
            None,
            "payload",
        ),
        FiniteVolumeTopologyEvent(
            0,
            1,
            0.25,
            TopologyEventKind.REMESH,
            TopologyEventState.REQUESTED,
            TopologyEventStatus.PENDING,
            "other-request",
            initial.epoch_id,
            None,
            "payload",
        ),
        FiniteVolumeTopologyEvent(
            0,
            1,
            0.25,
            TopologyEventKind.REMESH,
            TopologyEventState.REQUESTED,
            TopologyEventStatus.PENDING,
            "request",
            "other-input-epoch",
            None,
            "payload",
        ),
        FiniteVolumeTopologyEvent(
            0,
            1,
            0.25,
            TopologyEventKind.REMESH,
            TopologyEventState.FAILED,
            TopologyEventStatus.FAILED,
            "request",
            initial.epoch_id,
            "failure-a",
            "payload",
        ),
        FiniteVolumeTopologyEvent(
            0,
            1,
            0.25,
            TopologyEventKind.REMESH,
            TopologyEventState.FAILED,
            TopologyEventStatus.FAILED,
            "request",
            initial.epoch_id,
            "failure-b",
            "payload",
        ),
        FiniteVolumeTopologyEvent(
            0,
            1,
            0.25,
            TopologyEventKind.REMESH,
            TopologyEventState.REQUESTED,
            TopologyEventStatus.PENDING,
            "request",
            initial.epoch_id,
            None,
            "other-payload",
        ),
    )
    assert len({event.event_id for event in event_variants}) == len(event_variants)


def test_topology_event_journal_replay_reconstructs_exact_history():
    initial = _epoch("initial")
    failed = (
        FiniteVolumeTopologyEventJournal.allocate(initial, capacity=2)
        .append_requested(_request(initial, "failed"), 1, 0.25)
        .fail(0, result_id="failure-evidence", payload_id="failure-payload")
    )
    requested = failed.append_requested(_request(initial, "committed"), 2, 0.5)
    result = _epoch("result", parent_epoch_id=initial.epoch_id)
    committed = requested.commit(1, result, payload_id="commit-payload")
    source = committed.append_requested(_request(result, "overflow"), 3, 0.75)
    records = tuple(source.event(sequence) for sequence in range(int(source.count)))

    replayed = FiniteVolumeTopologyEventJournal.from_events(
        initial,
        records,
        result_epochs=(result,),
        capacity=2,
        overflowed=True,
    )

    assert replayed.journal_id == source.journal_id
    assert replayed.current_epoch_id == source.current_epoch_id
    assert replayed.epoch_table == source.epoch_table
    assert tuple(replayed.event(index) for index in range(2)) == records
    for replayed_array, source_array in zip(
        jax.tree.leaves(eqx.filter(replayed, eqx.is_array)),
        jax.tree.leaves(eqx.filter(source, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(replayed_array, source_array)

    with pytest.raises(ValueError, match="contiguous from zero"):
        FiniteVolumeTopologyEventJournal.from_events(
            initial,
            (records[1],),
            result_epochs=(result,),
            capacity=2,
        )
    with pytest.raises(ValueError, match="Committed topology event slot"):
        FiniteVolumeTopologyEventJournal.from_events(
            initial,
            records,
            capacity=2,
            overflowed=True,
        )

    impossible_first = FiniteVolumeTopologyEvent(
        0,
        1,
        0.25,
        TopologyEventKind.REMESH,
        TopologyEventState.FAILED,
        TopologyEventStatus.FAILED,
        "impossible-request",
        result.epoch_id,
        "failure",
        None,
    )
    with pytest.raises(ValueError, match="historical tip"):
        FiniteVolumeTopologyEventJournal.from_events(
            initial,
            (impossible_first, records[1]),
            result_epochs=(result,),
            capacity=2,
        )

    unresolved = FiniteVolumeTopologyEvent(
        0,
        1,
        0.25,
        TopologyEventKind.REMESH,
        TopologyEventState.REQUESTED,
        TopologyEventStatus.PENDING,
        "unresolved-request",
        initial.epoch_id,
        None,
        None,
    )
    later = FiniteVolumeTopologyEvent(
        1,
        2,
        0.5,
        TopologyEventKind.REMESH,
        TopologyEventState.FAILED,
        TopologyEventStatus.FAILED,
        "later-request",
        initial.epoch_id,
        "failure",
        None,
    )
    with pytest.raises(ValueError, match="final journal record"):
        FiniteVolumeTopologyEventJournal.from_events(
            initial,
            (unresolved, later),
            capacity=2,
        )


def test_topology_epoch_event_request_and_journal_are_immutable():
    initial = _epoch("initial")
    request = _request(initial)
    event = FiniteVolumeTopologyEvent(
        0,
        1,
        0.25,
        TopologyEventKind.REMESH,
        TopologyEventState.REQUESTED,
        TopologyEventStatus.PENDING,
        request.request_id,
        initial.epoch_id,
        None,
        request.payload_id,
    )
    journal = FiniteVolumeTopologyEventJournal.allocate(initial, capacity=1)

    with pytest.raises(FrozenInstanceError):
        initial.topology_id = "mutated"
    with pytest.raises(FrozenInstanceError):
        request.reason = "mutated"
    with pytest.raises(FrozenInstanceError):
        event.status = TopologyEventStatus.SUCCESS
    with pytest.raises(FrozenInstanceError):
        journal.count = jnp.asarray(1)


def test_topology_event_journal_numeric_storage_is_jit_safe_arrays():
    initial = _epoch("initial")
    journal = FiniteVolumeTopologyEventJournal.allocate(
        initial, capacity=2, time=jnp.asarray(0.0, dtype=jnp.float32)
    ).append_requested(_request(initial), 3, 0.75)

    dynamic, static = eqx.partition(journal, eqx.is_array)
    leaves = jax.tree.leaves(dynamic)
    assert leaves
    assert all(eqx.is_array(leaf) for leaf in leaves)
    assert static.current_epoch_id == initial.epoch_id
    assert static.epoch_table == (initial,)

    summary = jax.jit(
        lambda value: (
            value.kinds,
            value.states,
            value.statuses,
            value.accepted_steps,
            value.times,
            value.next_sequence,
            value.count,
            value.overflowed,
        )
    )(journal)
    assert all(isinstance(value, jax.Array) for value in summary)
    np.testing.assert_array_equal(summary[0], np.asarray([0, -1], dtype=np.int32))
    assert summary[4].dtype == jnp.float32


def test_scheduler_builds_certified_remap_before_committing_event():
    source = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        quadrilaterals=np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        cell_global_ids=np.asarray((10,), dtype=np.int64),
    ).prepare()
    target = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        triangles=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        cell_global_ids=np.asarray((20, 21), dtype=np.int64),
    ).prepare()
    initial = FiniteVolumeTopologyEpoch(
        source.prepared_id,
        source.topology_id,
        source.geometry_id,
        topology_artifact_id="source-topology",
        metrics_artifact_id="source-metrics",
        operators_artifact_id="source-operators",
    )
    journal = FiniteVolumeTopologyEventJournal.allocate(initial, capacity=2, time=0.0)
    scheduler = FiniteVolumeTopologyEventScheduler(journal)
    request = _request(initial, kind=TopologyEventKind.REMESH)
    scheduler.submit(request, 1, 0.1)
    successor = FiniteVolumeTopologyEpoch(
        target.prepared_id,
        target.topology_id,
        target.geometry_id,
        parent_epoch_id=initial.epoch_id,
        topology_artifact_id="target-topology",
        metrics_artifact_id="target-metrics",
        operators_artifact_id="target-operators",
    )
    source_metrics = phx.discretization.lower_static_unstructured_stage_metrics(
        source, topology_epoch_id=initial.epoch_id
    )
    target_metrics = phx.discretization.lower_static_unstructured_stage_metrics(
        target, topology_epoch_id=successor.epoch_id
    )
    precision = FiniteVolumePrecisionPolicy("float64")
    source_content = FiniteVolumeConservativeContentState(
        jnp.ones((1, 1)),
        source.cell_volumes,
        jnp.ones((1,), dtype=bool),
        0.0,
        topology_epoch_id=initial.epoch_id,
        geometry_family_id=source_metrics.geometry_family_id,
        geometry_layout_id=source_metrics.geometry_layout_id,
        geometry_version=source_metrics.geometry_version,
        evidence_policy_id=source_metrics.evidence.policy_id,
        evidence_version=source_metrics.evidence.evidence_version,
        precision=precision,
    )

    def transfer(content, remap):
        average = remap.apply(content.cell_average())
        return FiniteVolumeConservativeContentState(
            average * target.cell_volumes[:, None],
            target.cell_volumes,
            jnp.ones((target.cell_count,), dtype=bool),
            0.1,
            topology_epoch_id=successor.epoch_id,
            geometry_family_id=target_metrics.geometry_family_id,
            geometry_layout_id=target_metrics.geometry_layout_id,
            geometry_version=target_metrics.geometry_version,
            evidence_policy_id=target_metrics.evidence.policy_id,
            evidence_version=target_metrics.evidence.evidence_version,
            precision=precision,
        )

    result = scheduler.transact(
        accepted=True,
        source_geometry=source,
        target_geometry=target,
        candidate_epoch=successor,
        remap_tolerance=1e-10,
        source_content=source_content,
        transfer=transfer,
    )
    assert result.committed
    assert result.result_epoch == successor
    assert isinstance(result.content_state, FiniteVolumeConservativeContentState)
    assert result.content_state.topology_epoch_id == successor.epoch_id
    assert result.content_state.conservative_content.shape == (2, 1)
    np.testing.assert_allclose(
        result.content_state.volume_integral(),
        source_content.volume_integral(),
        atol=1e-14,
    )
    system = phx.equations.ScalarConservationSystem(
        2,
        lambda state, axis, args: jnp.zeros_like(state),
        lambda left, right, axis, args: jnp.zeros(left.shape[:-1]),
        system_id="topology-remap-restart",
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        target.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in target.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "topology-remap-restart",
        "state",
        system,
        boundaries,
    )
    compiled = phx.equations.compile_conservation_problem(problem, target, method)
    target_runtime = PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(
            fallback_flux=phx.discretization.RusanovFluxPlan()
        ),
    ).reprepare_for_epoch(successor)
    resumed = FiniteVolumeRuntimeState(
        result.content_state,
        result.journal,
        1e-3,
    )
    assert result.journal.current_epoch_id == successor.epoch_id
    assert target_runtime.initial_topology_epoch.epoch_id == successor.epoch_id
    resumed_result = target_runtime.advance(resumed)
    assert resumed_result.accepted

    bad_target = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        triangles=np.asarray(((0, 1, 2),), dtype=np.int32),
        cell_global_ids=np.asarray((30,), dtype=np.int64),
    ).prepare()
    bad_successor = FiniteVolumeTopologyEpoch(
        bad_target.prepared_id,
        bad_target.topology_id,
        bad_target.geometry_id,
        parent_epoch_id=initial.epoch_id,
        topology_artifact_id="bad-topology",
        metrics_artifact_id="bad-metrics",
        operators_artifact_id="bad-operators",
    )
    failed_journal = FiniteVolumeTopologyEventJournal.allocate(
        initial, capacity=2, time=0.0
    )
    failed_scheduler = FiniteVolumeTopologyEventScheduler(failed_journal)
    failed_scheduler.submit(_request(initial, name="bad"), 1, 0.1)
    failed = failed_scheduler.transact(
        accepted=True,
        source_geometry=source,
        target_geometry=bad_target,
        candidate_epoch=bad_successor,
        source_content=source_content,
    )
    assert not failed.committed
    assert failed.journal.current_epoch_id == initial.epoch_id
    assert result.journal.current_epoch_id == successor.epoch_id
