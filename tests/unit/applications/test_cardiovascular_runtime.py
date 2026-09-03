#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax._numerics._checkpointed_scan import (
    AdaptiveReplayPreparationPolicy,
    prepare_replay_schedule,
)
from phydrax.applications.cardiovascular._execution import (
    admit_cardiovascular_capacity,
    cardiovascular_runtime_diagnostic,
    CardiovascularCapacityManifest,
    CardiovascularCapacityRequest,
    CardiovascularCheckpointRecord,
    CardiovascularCohortCaseCandidate,
    CardiovascularCohortExecution,
    CardiovascularDistributedCollectiveExecution,
    CardiovascularDistributedReferenceExecution,
    CardiovascularEventSpec,
    CardiovascularExecutionManifest,
    CardiovascularLifecycleCheckpointCodec,
    CardiovascularMultiratePlan,
    CardiovascularRuntimeError,
    CardiovascularRuntimeStatus,
    CardiovascularSaltationPolicy,
    CardiovascularSerialExecution,
    CardiovascularStepCandidate,
    commit_cardiovascular_schedule,
    execute_cardiovascular_cohort,
    execute_cardiovascular_distributed_collective,
    execute_cardiovascular_distributed_reference,
    observe_single_device_runtime,
    prepare_cardiovascular_cohort,
    prepare_cardiovascular_distributed_execution,
    prepare_cardiovascular_scheduler,
    read_cardiovascular_distributed_solver_checkpoint,
    replay_cardiovascular_schedule,
    require_cardiovascular_distributed_transport,
    run_cardiovascular_schedule,
    write_cardiovascular_distributed_solver_checkpoint,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization.fem._distributed import (
    lower_distributed_finite_element_phases,
    partition_cells_cost_aware,
)
from phydrax.discretization.fem._generic import FiniteElementFieldSpec, FiniteElementPlan
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.lifecycle import (
    CheckpointManifest,
    CheckpointShard,
    create as create_lifecycle_archive,
    payload_byte_count,
    payload_digest,
)
from phydrax.linalg import FailurePolicy, GMRES, LinearSolvePolicy, TolerancePolicy


def _capacity(**overrides):
    values = {
        "maximum_cohort_cases": 8,
        "maximum_state_values": 64,
        "maximum_checkpoint_arrays": 8,
        "maximum_checkpoint_bytes": 16_384,
        "maximum_macro_steps": 4,
        "maximum_scheduled_steps": 16,
        "maximum_events": 8,
        "maximum_partitions": 4,
    }
    values.update(overrides)
    return CardiovascularCapacityManifest(**values)


def _execution(route=None, *, capacity=None, **overrides):
    values = {
        "case_manifest_id": "case:adult-aorta:001",
        "analysis_plan_id": "analysis:coupled-heart:001",
        "numeric_revision_id": "revision:baseline:001",
        "topology_id": "topology:fixed:001",
        "solver_policy_id": "solver:newton-krylov:001",
        "precision_policy_id": "precision:f64:001",
        "backend": jax.default_backend(),
        "capacity": _capacity() if capacity is None else capacity,
        "route": CardiovascularSerialExecution() if route is None else route,
    }
    values.update(overrides)
    return CardiovascularExecutionManifest(**values)


def _distributed_fem(part_count=2):
    mesh = CellMesh.from_triangles(
        np.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            )
        ),
        np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int32),
    )
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("pressure", lagrange_element("triangle", 1)),
    ).prepare()
    phases = lower_distributed_finite_element_phases(
        discretization, partition_cells_cost_aware(discretization, part_count)
    )
    return discretization, phases


def _replay_schedule(step_count=4):
    return prepare_replay_schedule(
        step_count,
        8,
        AdaptiveReplayPreparationPolicy(64, 128),
    )


def _external_checkpoint(path, execution, values, checkpoint_id):
    array = np.asarray(values)
    manifest = CheckpointManifest(
        checkpoint_id,
        execution.analysis_plan_id,
        execution.numeric_revision_id,
        execution.manifest_id,
        (
            CheckpointShard(
                "state",
                payload_digest(array),
                payload_byte_count(array),
            ),
        ),
        complete=True,
    )
    return create_lifecycle_archive(path, manifest=manifest, arrays={"state": array})


def test_capacity_admission_is_atomic_and_single_device_evidence_is_observed():
    capacity = _capacity(maximum_events=2)
    admitted = admit_cardiovascular_capacity(
        capacity,
        CardiovascularCapacityRequest(cohort_cases=8, events=2),
    )
    refused = admit_cardiovascular_capacity(
        capacity,
        CardiovascularCapacityRequest(
            cohort_cases=9,
            events=3,
            checkpoint_arrays=capacity.maximum_checkpoint_arrays + 1,
        ),
    )

    assert admitted.eligible
    assert admitted.status is CardiovascularRuntimeStatus.SUCCESS
    assert not refused.eligible
    assert refused.status is CardiovascularRuntimeStatus.CAPACITY_REFUSED
    assert refused.exceeded_resources == (
        "checkpoint_arrays",
        "cohort_cases",
        "events",
    )

    evidence = observe_single_device_runtime(_execution(capacity=capacity))
    assert evidence.eligible
    assert evidence.platform == jax.default_backend()
    assert evidence.device_id >= 0
    assert evidence.visible_backend_devices >= 1


def test_lifecycle_checkpoint_serial_restart_lineage_capacity_and_corruption(tmp_path):
    execution = _execution()
    codec = CardiovascularLifecycleCheckpointCodec(execution)
    first_path = tmp_path / "accepted-0001.phx"
    first = codec.write(
        first_path,
        {
            "activation": jnp.asarray([0.25, 0.5, 0.75]),
            "pressure": jnp.asarray([11.0, 12.0]),
            "step": jnp.asarray(7, dtype=jnp.int32),
        },
        checkpoint_id="checkpoint:0001",
        committed=True,
        layout_ids={"activation": ("layout:nodes",), "pressure": ("layout:cavities",)},
    )
    restored = codec.read(first_path)

    assert isinstance(restored, CardiovascularCheckpointRecord)
    assert restored.checkpoint_id == "checkpoint:0001"
    assert restored.archive.archive_id == first.archive.archive_id
    np.testing.assert_array_equal(restored.arrays["activation"], [0.25, 0.5, 0.75])
    np.testing.assert_array_equal(restored.arrays["pressure"], [11.0, 12.0])
    assert not restored.arrays["pressure"].flags.writeable

    second = codec.write(
        tmp_path / "accepted-0002.phx",
        restored.arrays,
        checkpoint_id="checkpoint:0002",
        parent_checkpoint_id=restored.checkpoint_id,
        committed=True,
    )
    assert second.parent_checkpoint_id == "checkpoint:0001"

    with pytest.raises(CardiovascularRuntimeError) as uncommitted:
        codec.write(
            tmp_path / "uncommitted.phx",
            {"state": jnp.ones((2,))},
            checkpoint_id="checkpoint:uncommitted",
            committed=False,
        )
    assert uncommitted.value.status is CardiovascularRuntimeStatus.CHECKPOINT_REFUSED
    assert not (tmp_path / "uncommitted.phx").exists()

    mismatched = CardiovascularLifecycleCheckpointCodec(
        _execution(numeric_revision_id="revision:changed:002")
    )
    with pytest.raises(CardiovascularRuntimeError) as mismatch:
        mismatched.read(first_path)
    assert mismatch.value.status is CardiovascularRuntimeStatus.CHECKPOINT_MISMATCH

    with zipfile.ZipFile(first_path, "a") as archive:
        archive.writestr("unexpected-member", b"injected corruption")
    with pytest.raises(ArrayArchiveCorruptionError):
        codec.read(first_path)

    too_small = CardiovascularLifecycleCheckpointCodec(
        _execution(capacity=_capacity(maximum_state_values=1))
    )
    with pytest.raises(CardiovascularRuntimeError) as refusal:
        too_small.write(
            tmp_path / "too-large.phx",
            {"state": jnp.ones((2,))},
            checkpoint_id="checkpoint:too-large",
            committed=True,
        )
    assert refusal.value.status is CardiovascularRuntimeStatus.CHECKPOINT_REFUSED
    assert not (tmp_path / "too-large.phx").exists()
    oversized_execution = _execution(capacity=_capacity(maximum_state_values=1))
    oversized_path = tmp_path / "external-oversized.phx"
    _external_checkpoint(
        oversized_path,
        oversized_execution,
        np.asarray([1.0, 2.0]),
        "checkpoint:external-oversized",
    )
    with pytest.raises(ArrayArchiveCorruptionError):
        CardiovascularLifecycleCheckpointCodec(oversized_execution).read(oversized_path)

    nonfinite_path = tmp_path / "external-nonfinite.phx"
    _external_checkpoint(
        nonfinite_path,
        execution,
        np.asarray([1.0, np.nan]),
        "checkpoint:external-nonfinite",
    )
    with pytest.raises(ArrayArchiveCorruptionError, match="finite numeric"):
        codec.read(nonfinite_path)


def test_execution_pool_cohort_is_case_and_lane_deterministic_and_fail_closed():
    case_ids = ("patient-c", "patient-a", "patient-b", "patient-d")
    one_lane = prepare_cardiovascular_cohort(
        _execution(CardiovascularCohortExecution(1)), case_ids
    )
    three_lanes = prepare_cardiovascular_cohort(
        _execution(CardiovascularCohortExecution(3)), tuple(reversed(case_ids))
    )

    def execute(case_id, key):
        case_offset = sum(map(ord, case_id))
        return CardiovascularCohortCaseCandidate(jax.random.uniform(key) + case_offset)

    first = execute_cardiovascular_cohort(one_lane, jax.random.key(17), execute)
    second = execute_cardiovascular_cohort(three_lanes, jax.random.key(17), execute)
    assert first.committed and second.committed
    assert first.case_ids == second.case_ids == tuple(sorted(case_ids))
    np.testing.assert_array_equal(
        np.asarray(first.evidence.semantic_keys),
        np.asarray(second.evidence.semantic_keys),
    )
    np.testing.assert_array_equal(np.asarray(first.values), np.asarray(second.values))

    def reject_one(case_id, key):
        value = jax.random.uniform(key)
        if case_id == "patient-b":
            return CardiovascularCohortCaseCandidate(
                value,
                accepted=False,
                status=CardiovascularRuntimeStatus.STEP_REJECTED,
            )
        return CardiovascularCohortCaseCandidate(value)

    failed = execute_cardiovascular_cohort(one_lane, jax.random.key(17), reject_one)
    assert not failed.committed
    assert failed.values == ()
    assert failed.evidence.status is CardiovascularRuntimeStatus.COMMIT_REFUSED


def test_distributed_fem_operator_owned_shards_transpose_solve_and_restart(tmp_path):
    discretization, phases = _distributed_fem()
    replay = _replay_schedule()
    reference = prepare_cardiovascular_distributed_execution(
        _execution(CardiovascularDistributedReferenceExecution(2)), phases, replay
    )
    cell_values = jnp.asarray(((1.5, -0.25), (2.0, 0.75)))
    reference_evidence = execute_cardiovascular_distributed_reference(
        reference, cell_values
    )
    np.testing.assert_allclose(reference_evidence.residual_norm, 0.0)

    single_discretization, single_phases = _distributed_fem(1)
    collective_execution = _execution(
        CardiovascularDistributedCollectiveExecution(1, "cardiovascular-parts")
    )
    collective = prepare_cardiovascular_distributed_execution(
        collective_execution, single_phases, replay
    )
    require_cardiovascular_distributed_transport(collective)
    operator = single_discretization.mass
    expected_solution = jnp.asarray((0.5, -0.25, 0.75, 1.25))
    right_hand_side = operator.mv(expected_solution)
    policy = LinearSolvePolicy(
        GMRES(restart=4),
        tolerance=TolerancePolicy(relative=1.0e-7, absolute=1.0e-9, max_steps=16),
        failure=FailurePolicy("status"),
    )
    evidence = execute_cardiovascular_distributed_collective(
        collective,
        single_discretization.dof_maps[0],
        operator,
        right_hand_side,
        initial_guess=jnp.zeros_like(expected_solution),
        solver_policy=policy,
    )
    np.testing.assert_allclose(evidence.operator_residual_norm, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(evidence.halo_residual_norm, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(evidence.transpose_residual_norm, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(evidence.solver_serial_residual_norm, 0.0, atol=1.0e-6)
    assert evidence.solver_state.successful
    assert evidence.finite_element_operator_id == operator.operator_id
    assert evidence.partition_id == single_phases.partition.partition_id
    assert evidence.device_mesh_id == collective.capability.device_mesh_id
    assert evidence.transport_id == collective.capability.transport_id
    assert evidence.solver_state.owned_solution.sharding.spec[0] == "cardiovascular-parts"

    codec = CardiovascularLifecycleCheckpointCodec(collective_execution)
    checkpoint_path = tmp_path / "distributed-solver.phx"
    written = write_cardiovascular_distributed_solver_checkpoint(
        codec,
        checkpoint_path,
        evidence.solver_state,
        checkpoint_id="checkpoint:distributed:0001",
    )
    restored = read_cardiovascular_distributed_solver_checkpoint(
        codec, checkpoint_path, collective, evidence.solver_state
    )
    restarted = execute_cardiovascular_distributed_collective(
        collective,
        single_discretization.dof_maps[0],
        operator,
        None,
        solver_policy=policy,
        restart_state=restored,
    )
    assert written.checkpoint_id == restored.checkpoint_id
    assert restarted.solver_state.checkpoint_id == restored.checkpoint_id
    assert restarted.solver_state.solve_count == 2
    np.testing.assert_allclose(restarted.solver_serial_residual_norm, 0.0, atol=1.0e-6)

    unavailable_process_mesh = prepare_cardiovascular_distributed_execution(
        _execution(
            CardiovascularDistributedCollectiveExecution(
                2, "cardiovascular-hosts", process_count=2
            )
        ),
        phases,
        replay,
    )
    if jax.process_count() < 2:
        assert (
            unavailable_process_mesh.capability.reason
            == "insufficient-process-device-mesh"
        )
        with pytest.raises(CardiovascularRuntimeError):
            require_cardiovascular_distributed_transport(unavailable_process_mesh)


def test_multirate_event_localization_order_saltation_and_exact_replay():
    execution = _execution(
        capacity=_capacity(
            maximum_macro_steps=2,
            maximum_scheduled_steps=4,
            maximum_events=4,
        )
    )
    plan = CardiovascularMultiratePlan(
        ("electrophysiology",),
        (2,),
        1.0,
        events=(
            CardiovascularEventSpec(
                "valve-close",
                direction=1,
                priority=20,
                saltation_policy=CardiovascularSaltationPolicy("kPa", 0.1),
            ),
            CardiovascularEventSpec(
                "valve-open",
                direction=1,
                priority=10,
                saltation_policy=CardiovascularSaltationPolicy("kPa", 0.1),
            ),
        ),
        localization_iterations=48,
        localization_tolerance_ms=1.0e-10,
    )
    prepared = prepare_cardiovascular_scheduler(execution, plan)

    def advance(state, subsystem_id, start_ms, end_ms):
        assert subsystem_id == "electrophysiology"
        return CardiovascularStepCandidate(state + (end_ms - start_ms))

    def guards(state, time_ms):
        del time_ms
        return jnp.asarray((state - 0.75, state - 0.75))

    reset_order = []

    def reset(state, source_id, time_ms):
        del time_ms
        reset_order.append(source_id)
        return CardiovascularStepCandidate(state + 0.125)

    candidate = run_cardiovascular_schedule(
        prepared,
        jnp.asarray(0.0),
        1,
        advance,
        guards,
        reset,
    )
    committed = commit_cardiovascular_schedule(candidate)

    assert committed.committed
    assert committed.status is CardiovascularRuntimeStatus.SUCCESS
    assert reset_order[:2] == ["valve-open", "valve-close"]
    assert int(committed.evidence.event_count) == 2
    np.testing.assert_array_equal(
        np.asarray(committed.evidence.event_source_indices)[:2], [1, 0]
    )
    np.testing.assert_allclose(
        np.asarray(committed.evidence.event_times_ms)[:2], [0.75, 0.75], atol=1e-9
    )
    np.testing.assert_array_equal(
        np.asarray(committed.evidence.saltation_eligible)[:2], [True, True]
    )
    np.testing.assert_allclose(
        np.asarray(committed.evidence.event_guard_slope_per_ms)[:2],
        [1.0, 1.0],
    )

    reset_order.clear()
    replay_commit, replay_evidence = replay_cardiovascular_schedule(
        prepared,
        jnp.asarray(0.0),
        1,
        committed,
        advance,
        guards,
        reset,
    )
    assert replay_commit.committed
    assert replay_evidence.equivalent
    assert replay_evidence.status is CardiovascularRuntimeStatus.SUCCESS
    changed_prepared = prepare_cardiovascular_scheduler(
        _execution(
            capacity=execution.capacity,
            numeric_revision_id="revision:replay-mismatch",
        ),
        plan,
    )
    _, mismatched_replay = replay_cardiovascular_schedule(
        changed_prepared,
        jnp.asarray(0.0),
        1,
        committed,
        advance,
        guards,
        reset,
    )
    assert not mismatched_replay.equivalent
    assert mismatched_replay.status is CardiovascularRuntimeStatus.REPLAY_MISMATCH


def test_scheduler_failure_and_capacity_refusal_roll_back_atomically():
    execution = _execution(
        capacity=_capacity(
            maximum_macro_steps=1,
            maximum_scheduled_steps=1,
            maximum_events=1,
            maximum_state_values=2,
        )
    )
    prepared = prepare_cardiovascular_scheduler(
        execution,
        CardiovascularMultiratePlan(("mechanics",), (1,), 1.0),
    )
    initial = jnp.asarray([3.0, 4.0])

    def rejected_advance(state, subsystem_id, start_ms, end_ms):
        del subsystem_id, start_ms, end_ms
        return CardiovascularStepCandidate(
            state + 100.0,
            accepted=False,
            status=CardiovascularRuntimeStatus.STEP_REJECTED,
        )

    def no_events(state, time_ms):
        del state, time_ms
        return jnp.zeros((0,))

    def no_reset(state, source_id, time_ms):
        del source_id, time_ms
        return CardiovascularStepCandidate(state)

    rejected = commit_cardiovascular_schedule(
        run_cardiovascular_schedule(
            prepared,
            initial,
            1,
            rejected_advance,
            no_events,
            no_reset,
        )
    )
    assert not rejected.committed
    assert rejected.status is CardiovascularRuntimeStatus.STEP_REJECTED
    np.testing.assert_array_equal(rejected.state, initial)

    over_steps = commit_cardiovascular_schedule(
        run_cardiovascular_schedule(
            prepared,
            initial,
            2,
            rejected_advance,
            no_events,
            no_reset,
        )
    )
    assert not over_steps.committed
    assert over_steps.status is CardiovascularRuntimeStatus.CAPACITY_REFUSED
    np.testing.assert_array_equal(over_steps.state, initial)

    over_state = commit_cardiovascular_schedule(
        run_cardiovascular_schedule(
            prepared,
            jnp.asarray([1.0, 2.0, 3.0]),
            1,
            rejected_advance,
            no_events,
            no_reset,
        )
    )
    assert not over_state.committed
    assert over_state.status is CardiovascularRuntimeStatus.CAPACITY_REFUSED
    np.testing.assert_array_equal(over_state.state, [1.0, 2.0, 3.0])


def test_scheduler_enforces_exact_callback_leaf_contract_at_every_boundary():
    execution = _execution(
        capacity=_capacity(
            maximum_macro_steps=1,
            maximum_scheduled_steps=1,
            maximum_events=2,
        )
    )
    tolerance = 1.0e-8
    prepared = prepare_cardiovascular_scheduler(
        execution,
        CardiovascularMultiratePlan(
            ("electrophysiology",),
            (1,),
            1.0,
            events=(CardiovascularEventSpec("threshold", direction=1),),
            localization_tolerance_ms=tolerance,
        ),
    )
    initial = jnp.asarray(0.0, dtype=jnp.float32)

    def guards(state, time_ms):
        del time_ms
        return jnp.asarray((state - 0.5,))

    def normal_advance(state, subsystem_id, start_ms, end_ms):
        del subsystem_id
        return CardiovascularStepCandidate(state + end_ms - start_ms)

    def normal_reset(state, source_id, time_ms):
        del source_id, time_ms
        return CardiovascularStepCandidate(state + 0.1)

    def wrong_dtype(state, subsystem_id, start_ms, end_ms):
        del subsystem_id
        value = state + end_ms - start_ms
        return CardiovascularStepCandidate(value.astype(jnp.int32))

    with pytest.raises(ValueError, match="exact state leaf"):
        run_cardiovascular_schedule(
            prepared, initial, 1, wrong_dtype, guards, normal_reset
        )

    def wrong_localization_dtype(state, subsystem_id, start_ms, end_ms):
        del subsystem_id
        value = state + end_ms - start_ms
        if start_ms == 0.0 and end_ms < 1.0:
            value = value.astype(jnp.int32)
        return CardiovascularStepCandidate(value)

    with pytest.raises(ValueError, match="exact state leaf"):
        run_cardiovascular_schedule(
            prepared,
            initial,
            1,
            wrong_localization_dtype,
            guards,
            normal_reset,
        )

    def wrong_reset_shape(state, source_id, time_ms):
        del source_id, time_ms
        return CardiovascularStepCandidate(jnp.stack((state, state)))

    with pytest.raises(ValueError, match="exact state leaf"):
        run_cardiovascular_schedule(
            prepared, initial, 1, normal_advance, guards, wrong_reset_shape
        )

    def wrong_nudge_shape(state, subsystem_id, start_ms, end_ms):
        del subsystem_id
        value = state + end_ms - start_ms
        if start_ms > 0.0 and end_ms - start_ms <= 1.1 * tolerance:
            value = jnp.stack((value, value))
        return CardiovascularStepCandidate(value)

    with pytest.raises(ValueError, match="exact state leaf"):
        run_cardiovascular_schedule(
            prepared, initial, 1, wrong_nudge_shape, guards, normal_reset
        )


def test_sanitized_diagnostic_never_retains_injected_sensitive_detail():
    secret = "patient-name=not-for-storage"
    diagnostic = cardiovascular_runtime_diagnostic(
        CardiovascularRuntimeStatus.STEP_REJECTED,
        phase="qualification-failure-injection",
        run_id="qualification:001",
        entity_ids=("case:deidentified",),
    )
    encoded = repr(diagnostic)
    assert secret not in encoded
    assert secret not in diagnostic.message
    assert diagnostic.code == "CARDIOVASCULAR_STEP_REJECTED"
    assert diagnostic.run_id == "qualification:001"
