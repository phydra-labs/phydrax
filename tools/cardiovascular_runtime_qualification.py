#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic failure-injection campaign for cardiovascular orchestration."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax._numerics._checkpointed_scan import (
    AdaptiveReplayPreparationPolicy,
    prepare_replay_schedule,
)
from phydrax.applications.cardiovascular._execution import (
    cardiovascular_runtime_diagnostic,
    CardiovascularCapacityManifest,
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


def _capacity() -> CardiovascularCapacityManifest:
    return CardiovascularCapacityManifest(
        maximum_cohort_cases=16,
        maximum_state_values=64,
        maximum_checkpoint_arrays=8,
        maximum_checkpoint_bytes=32_768,
        maximum_macro_steps=2,
        maximum_scheduled_steps=8,
        maximum_events=4,
        maximum_partitions=2,
    )


def _execution(route) -> CardiovascularExecutionManifest:
    return CardiovascularExecutionManifest(
        case_manifest_id="qualification:case",
        analysis_plan_id="qualification:analysis",
        numeric_revision_id="qualification:revision",
        topology_id="qualification:fixed-topology",
        solver_policy_id="qualification:solver",
        precision_policy_id="qualification:f64",
        backend=jax.default_backend(),
        capacity=_capacity(),
        route=route,
    )


def _distributed_fem(part_count=2):
    mesh = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
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


def _external_checkpoint(path: Path, execution, values, checkpoint_id: str) -> None:
    array = np.asarray(values)
    manifest = CheckpointManifest(
        checkpoint_id,
        execution.analysis_plan_id,
        execution.numeric_revision_id,
        execution.manifest_id,
        (CheckpointShard("state", payload_digest(array), payload_byte_count(array)),),
        complete=True,
    )
    create_lifecycle_archive(path, manifest=manifest, arrays={"state": array})


def _checkpoint_qualification(root: Path) -> dict[str, object]:
    execution = _execution(CardiovascularSerialExecution())
    codec = CardiovascularLifecycleCheckpointCodec(execution)
    path = root / "checkpoint.phx"
    written = codec.write(
        path,
        {
            "state": jnp.asarray([1.0, 2.0, 3.0]),
            "accepted_step": jnp.asarray(11, dtype=jnp.int32),
        },
        checkpoint_id="qualification:checkpoint:0001",
        committed=True,
    )
    restored = codec.read(path)
    restart_exact = bool(
        written.archive.archive_id == restored.archive.archive_id
        and np.array_equal(restored.arrays["state"], np.asarray([1.0, 2.0, 3.0]))
        and int(restored.arrays["accepted_step"]) == 11
    )

    refused_path = root / "refused.phx"
    atomic_refusal = False
    try:
        codec.write(
            refused_path,
            {"state": jnp.ones((2,))},
            checkpoint_id="qualification:checkpoint:refused",
            committed=False,
        )
    except CardiovascularRuntimeError as error:
        atomic_refusal = bool(
            error.status is CardiovascularRuntimeStatus.CHECKPOINT_REFUSED
            and not refused_path.exists()
        )

    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("injected-corruption", b"qualification fault")
    corruption_detected = False
    try:
        codec.read(path)
    except ArrayArchiveCorruptionError:
        corruption_detected = True
    oversized_path = root / "oversized.phx"
    _external_checkpoint(
        oversized_path,
        execution,
        np.arange(execution.capacity.maximum_state_values + 1.0),
        "qualification:checkpoint:oversized",
    )
    bounded_preflight = False
    try:
        codec.read(oversized_path)
    except ArrayArchiveCorruptionError:
        bounded_preflight = True

    nonfinite_path = root / "nonfinite.phx"
    _external_checkpoint(
        nonfinite_path,
        execution,
        np.asarray([1.0, np.nan]),
        "qualification:checkpoint:nonfinite",
    )
    nonfinite_rejected = False
    try:
        codec.read(nonfinite_path)
    except ArrayArchiveCorruptionError:
        nonfinite_rejected = True
    return {
        "restart_exact": restart_exact,
        "atomic_checkpoint_refusal": atomic_refusal,
        "corruption_detected": corruption_detected,
        "bounded_preflight": bounded_preflight,
        "nonfinite_rejected": nonfinite_rejected,
    }


def _cohort_qualification() -> dict[str, object]:
    case_ids = tuple(f"case-{index:03d}" for index in range(9))
    lane_one = prepare_cardiovascular_cohort(
        _execution(CardiovascularCohortExecution(1)), case_ids
    )
    lane_four = prepare_cardiovascular_cohort(
        _execution(CardiovascularCohortExecution(4)), tuple(reversed(case_ids))
    )

    def execute(case_id, key):
        offset = sum(map(ord, case_id))
        return CardiovascularCohortCaseCandidate(jax.random.normal(key) + offset)

    first = execute_cardiovascular_cohort(lane_one, jax.random.key(401), execute)
    second = execute_cardiovascular_cohort(lane_four, jax.random.key(401), execute)
    return {
        "canonical_case_order": first.case_ids == second.case_ids,
        "semantic_keys_exact": bool(
            np.array_equal(first.evidence.semantic_keys, second.evidence.semantic_keys)
        ),
        "lane_independent_values": bool(
            np.array_equal(np.asarray(first.values), np.asarray(second.values))
        ),
    }


def _event_and_failure_qualification() -> dict[str, object]:
    execution = _execution(CardiovascularSerialExecution())
    plan = CardiovascularMultiratePlan(
        ("electrophysiology",),
        (2,),
        1.0,
        events=(
            CardiovascularEventSpec(
                "valve-secondary",
                direction=1,
                priority=20,
                saltation_policy=CardiovascularSaltationPolicy("kPa", 0.1),
            ),
            CardiovascularEventSpec(
                "valve-primary",
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
        del subsystem_id
        return CardiovascularStepCandidate(state + end_ms - start_ms)

    def guards(state, time_ms):
        del time_ms
        return jnp.asarray((state - 0.75, state - 0.75))

    ordering: list[str] = []

    def reset(state, source_id, time_ms):
        del time_ms
        ordering.append(source_id)
        return CardiovascularStepCandidate(state + 0.125)

    candidate = run_cardiovascular_schedule(
        prepared, jnp.asarray(0.0), 1, advance, guards, reset
    )
    committed = commit_cardiovascular_schedule(candidate)
    ordering.clear()
    replayed, replay = replay_cardiovascular_schedule(
        prepared,
        jnp.asarray(0.0),
        1,
        committed,
        advance,
        guards,
        reset,
    )

    def injected_failure(state, subsystem_id, start_ms, end_ms):
        del subsystem_id, start_ms, end_ms
        return CardiovascularStepCandidate(
            state + 10_000.0,
            accepted=False,
            status=CardiovascularRuntimeStatus.STEP_REJECTED,
        )

    failed = commit_cardiovascular_schedule(
        run_cardiovascular_schedule(
            prepared,
            jnp.asarray(7.0),
            1,
            injected_failure,
            guards,
            reset,
        )
    )
    diagnostic = cardiovascular_runtime_diagnostic(
        failed.status,
        phase="qualification-failure-injection",
        run_id="qualification:runtime",
    )
    serialized_diagnostic = repr(diagnostic)

    def wrong_shape(state, subsystem_id, start_ms, end_ms):
        del subsystem_id, start_ms, end_ms
        return CardiovascularStepCandidate(jnp.stack((state, state)))

    leaf_contract_enforced = False
    try:
        run_cardiovascular_schedule(
            prepared,
            jnp.asarray(0.0),
            1,
            wrong_shape,
            guards,
            reset,
        )
    except ValueError as error:
        leaf_contract_enforced = "exact state leaf" in str(error)
    return {
        "event_commit": committed.committed,
        "event_count": int(committed.evidence.event_count),
        "event_order": tuple(
            plan.events[int(index)].source_id
            for index in np.asarray(committed.evidence.event_source_indices)[
                np.asarray(committed.evidence.event_active)
            ]
        ),
        "event_time_max_error_ms": float(
            np.max(
                np.abs(
                    np.asarray(committed.evidence.event_times_ms)[
                        np.asarray(committed.evidence.event_active)
                    ]
                    - 0.75
                )
            )
        ),
        "saltation_records": int(np.sum(committed.evidence.saltation_eligible)),
        "replay_exact": bool(replay.equivalent and replayed.committed),
        "atomic_failure_rollback": bool(
            not failed.committed and float(failed.state) == 7.0
        ),
        "diagnostic_sanitized": "10_000" not in serialized_diagnostic,
        "leaf_contract_enforced": leaf_contract_enforced,
    }


def _distributed_qualification(root: Path) -> dict[str, object]:
    discretization, phases = _distributed_fem()
    replay = prepare_replay_schedule(4, 8, AdaptiveReplayPreparationPolicy(64, 128))
    cell_values = jnp.asarray(((1.5, -0.25), (2.0, 0.75)))
    reference = prepare_cardiovascular_distributed_execution(
        _execution(CardiovascularDistributedReferenceExecution(2)), phases, replay
    )
    reference_evidence = execute_cardiovascular_distributed_reference(
        reference, cell_values
    )

    single_discretization, single_phases = _distributed_fem(1)
    single_execution = _execution(
        CardiovascularDistributedCollectiveExecution(1, "cardiovascular-qualification")
    )
    single_device = prepare_cardiovascular_distributed_execution(
        single_execution, single_phases, replay
    )
    operator = single_discretization.mass
    exact_solution = jnp.asarray((0.5, -0.25, 0.75, 1.25))
    right_hand_side = operator.mv(exact_solution)
    policy = LinearSolvePolicy(
        GMRES(restart=4),
        tolerance=TolerancePolicy(relative=1.0e-7, absolute=1.0e-9, max_steps=16),
        failure=FailurePolicy("status"),
    )
    single_evidence = execute_cardiovascular_distributed_collective(
        single_device,
        single_discretization.dof_maps[0],
        operator,
        right_hand_side,
        initial_guess=jnp.zeros_like(exact_solution),
        solver_policy=policy,
    )
    codec = CardiovascularLifecycleCheckpointCodec(single_execution)
    checkpoint_path = root / "distributed-solver.phx"
    checkpoint = write_cardiovascular_distributed_solver_checkpoint(
        codec,
        checkpoint_path,
        single_evidence.solver_state,
        checkpoint_id="qualification:distributed-solver:0001",
    )
    restored = read_cardiovascular_distributed_solver_checkpoint(
        codec, checkpoint_path, single_device, single_evidence.solver_state
    )
    restarted = execute_cardiovascular_distributed_collective(
        single_device,
        single_discretization.dof_maps[0],
        operator,
        None,
        solver_policy=policy,
        restart_state=restored,
    )

    multi_device = prepare_cardiovascular_distributed_execution(
        _execution(
            CardiovascularDistributedCollectiveExecution(
                2, "cardiovascular-qualification"
            )
        ),
        phases,
        replay,
    )
    if multi_device.capability.transport_eligible:
        multi_operator = discretization.mass
        multi_evidence = execute_cardiovascular_distributed_collective(
            multi_device,
            discretization.dof_maps[0],
            multi_operator,
            multi_operator.mv(exact_solution),
            initial_guess=jnp.zeros_like(exact_solution),
            solver_policy=policy,
        )
        multi_device_support = (
            True,
            "qualified",
            multi_device.capability.requested_device_count,
            multi_device.capability.available_device_count,
        )
        multi_device_valid = float(multi_evidence.operator_residual_norm) <= 1.0e-7
    else:
        transport_refused = False
        try:
            require_cardiovascular_distributed_transport(multi_device)
        except CardiovascularRuntimeError as error:
            transport_refused = (
                error.status is CardiovascularRuntimeStatus.DISTRIBUTED_INELIGIBLE
            )
        multi_device_support = (
            False,
            multi_device.capability.reason,
            multi_device.capability.requested_device_count,
            multi_device.capability.available_device_count,
        )
        multi_device_valid = transport_refused

    multi_host = prepare_cardiovascular_distributed_execution(
        _execution(
            CardiovascularDistributedCollectiveExecution(
                2,
                "cardiovascular-qualification-hosts",
                process_count=2,
            )
        ),
        phases,
        replay,
    )
    multi_host_support = (
        multi_host.capability.transport_eligible,
        multi_host.capability.reason,
        multi_host.capability.requested_process_count,
        multi_host.capability.available_process_count,
    )
    return {
        "reference_residual": float(reference_evidence.residual_norm),
        "reference_eligible": reference.capability.reference_eligible,
        "single_device_operator_residual": float(single_evidence.operator_residual_norm),
        "single_device_halo_residual": float(single_evidence.halo_residual_norm),
        "single_device_transpose_residual": float(
            single_evidence.transpose_residual_norm
        ),
        "single_device_solver_residual": float(single_evidence.solver_residual_norm),
        "single_device_solver_serial_residual": float(
            single_evidence.solver_serial_residual_norm
        ),
        "solver_successful": single_evidence.solver_state.successful,
        "owned_array_sharding": str(
            single_evidence.solver_state.owned_solution.sharding.spec
        ),
        "identity_binding": {
            "mesh": single_evidence.device_mesh_id,
            "operator": single_evidence.finite_element_operator_id,
            "distributed_operator": single_evidence.distributed_operator_id,
            "partition": single_evidence.partition_id,
            "transport": single_evidence.transport_id,
            "solver_plan": single_evidence.solver_plan_id,
        },
        "checkpoint_restart_exact": bool(
            checkpoint.checkpoint_id == restored.checkpoint_id
            and restarted.solver_state.checkpoint_id == restored.checkpoint_id
            and restarted.solver_state.solve_count == 2
            and float(restarted.solver_serial_residual_norm) <= 1.0e-6
        ),
        "multi_device_support": multi_device_support,
        "multi_device_valid_or_blocked": multi_device_valid,
        "multi_host_support": multi_host_support,
        "multi_host_fail_closed": not multi_host.capability.transport_eligible,
    }


def qualification_report() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="phydrax-cardio-runtime-") as directory:
        root = Path(directory)
        checkpoint = _checkpoint_qualification(root)
        distributed = _distributed_qualification(root)
    cohort = _cohort_qualification()
    scheduler = _event_and_failure_qualification()
    passed = bool(
        all(checkpoint.values())
        and all(cohort.values())
        and scheduler["event_commit"]
        and scheduler["event_count"] == 2
        and scheduler["event_order"] == ("valve-primary", "valve-secondary")
        and scheduler["event_time_max_error_ms"] <= 1.0e-8
        and scheduler["saltation_records"] == 2
        and scheduler["replay_exact"]
        and scheduler["atomic_failure_rollback"]
        and scheduler["diagnostic_sanitized"]
        and scheduler["leaf_contract_enforced"]
        and distributed["reference_residual"] <= 1.0e-12
        and distributed["reference_eligible"]
        and distributed["single_device_operator_residual"] <= 1.0e-7
        and distributed["single_device_halo_residual"] <= 1.0e-7
        and distributed["single_device_transpose_residual"] <= 1.0e-7
        and distributed["single_device_solver_residual"] <= 1.0e-6
        and distributed["single_device_solver_serial_residual"] <= 1.0e-6
        and distributed["solver_successful"]
        and all(distributed["identity_binding"].values())
        and distributed["checkpoint_restart_exact"]
        and distributed["multi_device_valid_or_blocked"]
        and distributed["multi_host_fail_closed"]
    )
    return {
        "campaign": "cardiovascular-runtime",
        "passed": passed,
        "checkpoint": checkpoint,
        "cohort": cohort,
        "scheduler": scheduler,
        "distributed": distributed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualification_report()
    encoded = json.dumps(report, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
