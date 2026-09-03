#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Capacity-bound execution, restart, replay, and runtime evidence.

The cardiovascular runtime is deliberately a thin orchestration layer over
PhydraX lifecycle archives, execution-pool primitives, replay schedules, and
finite-element partition plans.  It owns no transport, solver, archive format,
or mesh representation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike

from ..._array_archive import ArrayArchiveCorruptionError, ArrayArchiveLimits
from ..._execution_pool import (
    PoolExecutionSignature,
    refill_completed_tasks,
    semantic_task_keys,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics._checkpointed_scan import checkpointed_scan, PreparedReplaySchedule
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...diagnostics import Diagnostic
from ...discretization.fem import (
    DistributedFiniteElementOperator,
    FiniteElementDistributedPhasePlan,
    FiniteElementDofMap,
    JaxCollectiveBackend,
)
from ...lifecycle import (
    CheckpointManifest,
    CheckpointShard,
    create as create_lifecycle_archive,
    LifecycleArchive,
    open as open_lifecycle_archive,
    payload_byte_count,
    payload_digest,
)
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    FailurePolicy,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    OperatorCapabilities,
    prepare as prepare_linear_solve,
    PreparedLinearSolve,
    solve as solve_linear_system,
    TolerancePolicy,
    transpose,
)


State = TypeVar("State")
Value = TypeVar("Value")


class CardiovascularRuntimeStatus(StrEnum):
    """Fail-closed terminal status for cardiovascular host orchestration."""

    SUCCESS = "success"
    CAPACITY_REFUSED = "capacity-refused"
    STEP_REJECTED = "step-rejected"
    EVENT_LOCALIZATION_FAILED = "event-localization-failed"
    EVENT_RESET_REJECTED = "event-reset-rejected"
    EVENT_CAPACITY_EXCEEDED = "event-capacity-exceeded"
    CHECKPOINT_REFUSED = "checkpoint-refused"
    CHECKPOINT_MISMATCH = "checkpoint-mismatch"
    DISTRIBUTED_INELIGIBLE = "distributed-ineligible"
    COMMIT_REFUSED = "commit-refused"
    REPLAY_MISMATCH = "replay-mismatch"


_DIAGNOSTIC_TEXT: dict[CardiovascularRuntimeStatus, tuple[str, str, str]] = {
    CardiovascularRuntimeStatus.CAPACITY_REFUSED: (
        "CARDIOVASCULAR_CAPACITY_REFUSED",
        "The prepared execution exceeds a declared fixed capacity.",
        "Prepare a larger capacity manifest before compiling or executing.",
    ),
    CardiovascularRuntimeStatus.STEP_REJECTED: (
        "CARDIOVASCULAR_STEP_REJECTED",
        "A numerical transition was rejected; no candidate state was committed.",
        "Inspect the solver evidence and revise the numerical plan.",
    ),
    CardiovascularRuntimeStatus.EVENT_LOCALIZATION_FAILED: (
        "CARDIOVASCULAR_EVENT_LOCALIZATION_FAILED",
        "An event bracket could not be replayed to a finite localized transition.",
        "Inspect event guards and the declared localization tolerance.",
    ),
    CardiovascularRuntimeStatus.EVENT_RESET_REJECTED: (
        "CARDIOVASCULAR_EVENT_RESET_REJECTED",
        "An event reset was rejected; the complete candidate was rolled back.",
        "Inspect the reset evidence and fixed event topology.",
    ),
    CardiovascularRuntimeStatus.EVENT_CAPACITY_EXCEEDED: (
        "CARDIOVASCULAR_EVENT_CAPACITY_EXCEEDED",
        "The fixed event-record capacity was exhausted before completion.",
        "Prepare a larger event capacity after checking for repeated or Zeno events.",
    ),
    CardiovascularRuntimeStatus.CHECKPOINT_REFUSED: (
        "CARDIOVASCULAR_CHECKPOINT_REFUSED",
        "The state is not eligible for an atomic lifecycle checkpoint.",
        "Checkpoint only a committed state within the declared payload capacity.",
    ),
    CardiovascularRuntimeStatus.CHECKPOINT_MISMATCH: (
        "CARDIOVASCULAR_CHECKPOINT_MISMATCH",
        "The lifecycle checkpoint does not match this execution manifest.",
        "Resume with the exact case, topology, numerical revision, and execution plan.",
    ),
    CardiovascularRuntimeStatus.DISTRIBUTED_INELIGIBLE: (
        "CARDIOVASCULAR_DISTRIBUTED_INELIGIBLE",
        "The requested distributed capability is not eligible in this runtime.",
        "Use local reference execution or bind transport through the generic FEM API.",
    ),
    CardiovascularRuntimeStatus.COMMIT_REFUSED: (
        "CARDIOVASCULAR_COMMIT_REFUSED",
        "An unsuccessful candidate cannot cross the atomic commit boundary.",
        "Resolve the recorded failure and execute a new candidate.",
    ),
    CardiovascularRuntimeStatus.REPLAY_MISMATCH: (
        "CARDIOVASCULAR_REPLAY_MISMATCH",
        "Replay did not reproduce the recorded event route and committed state.",
        "Use the original execution manifest, callbacks, and numerical revision.",
    ),
}


def cardiovascular_runtime_diagnostic(
    status: CardiovascularRuntimeStatus,
    /,
    *,
    phase: str,
    run_id: str | None = None,
    entity_ids: Sequence[str] = (),
) -> Diagnostic:
    """Build a bounded diagnostic without retaining raw exception or patient text."""

    status_ = CardiovascularRuntimeStatus(status)
    if status_ is CardiovascularRuntimeStatus.SUCCESS:
        raise ValueError("Successful execution does not require a failure diagnostic.")
    phase_ = _identifier(phase, "phase")
    entities = tuple(_identifier(value, "entity_id") for value in entity_ids)
    if len(set(entities)) != len(entities):
        raise ValueError("Diagnostic entity IDs must be unique.")
    run = None if run_id is None else _identifier(run_id, "run_id")
    code, message, remediation = _DIAGNOSTIC_TEXT[status_]
    return Diagnostic(
        code,
        "error",
        phase_,
        message,
        entity_ids=entities,
        remediation=remediation,
        run_id=run,
    )


class CardiovascularRuntimeError(RuntimeError):
    """Sanitized runtime failure carrying a structured diagnostic only."""

    status: CardiovascularRuntimeStatus
    diagnostic: Diagnostic

    def __init__(
        self,
        status: CardiovascularRuntimeStatus,
        /,
        *,
        phase: str,
        run_id: str | None = None,
        entity_ids: Sequence[str] = (),
    ):
        status_ = CardiovascularRuntimeStatus(status)
        if status_ is CardiovascularRuntimeStatus.SUCCESS:
            raise ValueError("CardiovascularRuntimeError requires a failure status.")
        diagnostic = cardiovascular_runtime_diagnostic(
            status_, phase=phase, run_id=run_id, entity_ids=entity_ids
        )
        self.status = status_
        self.diagnostic = diagnostic
        super().__init__(diagnostic.code)


class CardiovascularCapacityManifest(StrictModule, NonTrainableState):
    """Hard capacities for one prepared cardiovascular execution family."""

    maximum_cohort_cases: int = eqx.field(static=True)
    maximum_state_values: int = eqx.field(static=True)
    maximum_checkpoint_arrays: int = eqx.field(static=True)
    maximum_checkpoint_bytes: int = eqx.field(static=True)
    maximum_macro_steps: int = eqx.field(static=True)
    maximum_scheduled_steps: int = eqx.field(static=True)
    maximum_events: int = eqx.field(static=True)
    maximum_partitions: int = eqx.field(static=True)
    capacity_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_cohort_cases: int,
        maximum_state_values: int,
        maximum_checkpoint_arrays: int,
        maximum_checkpoint_bytes: int,
        maximum_macro_steps: int,
        maximum_scheduled_steps: int,
        maximum_events: int,
        maximum_partitions: int,
    ):
        positive = (
            _positive_integer(maximum_cohort_cases, "maximum_cohort_cases"),
            _positive_integer(maximum_state_values, "maximum_state_values"),
            _positive_integer(maximum_checkpoint_arrays, "maximum_checkpoint_arrays"),
            _positive_integer(maximum_checkpoint_bytes, "maximum_checkpoint_bytes"),
            _positive_integer(maximum_macro_steps, "maximum_macro_steps"),
            _positive_integer(maximum_scheduled_steps, "maximum_scheduled_steps"),
            _positive_integer(maximum_partitions, "maximum_partitions"),
        )
        events = _nonnegative_integer(maximum_events, "maximum_events")
        (
            self.maximum_cohort_cases,
            self.maximum_state_values,
            self.maximum_checkpoint_arrays,
            self.maximum_checkpoint_bytes,
            self.maximum_macro_steps,
            self.maximum_scheduled_steps,
            self.maximum_partitions,
        ) = positive
        self.maximum_events = events
        self.capacity_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-capacity-manifest",
                "maximum_cohort_cases": positive[0],
                "maximum_state_values": positive[1],
                "maximum_checkpoint_arrays": positive[2],
                "maximum_checkpoint_bytes": positive[3],
                "maximum_macro_steps": positive[4],
                "maximum_scheduled_steps": positive[5],
                "maximum_events": events,
                "maximum_partitions": positive[6],
            }
        )


class CardiovascularCapacityRequest(StrictModule, NonTrainableState):
    """One all-or-nothing request against a capacity manifest."""

    cohort_cases: int = eqx.field(static=True)
    state_values: int = eqx.field(static=True)
    checkpoint_arrays: int = eqx.field(static=True)
    checkpoint_bytes: int = eqx.field(static=True)
    macro_steps: int = eqx.field(static=True)
    scheduled_steps: int = eqx.field(static=True)
    events: int = eqx.field(static=True)
    partitions: int = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        cohort_cases: int = 0,
        state_values: int = 0,
        checkpoint_arrays: int = 0,
        checkpoint_bytes: int = 0,
        macro_steps: int = 0,
        scheduled_steps: int = 0,
        events: int = 0,
        partitions: int = 0,
    ):
        values = tuple(
            _nonnegative_integer(value, name)
            for value, name in (
                (cohort_cases, "cohort_cases"),
                (state_values, "state_values"),
                (checkpoint_arrays, "checkpoint_arrays"),
                (checkpoint_bytes, "checkpoint_bytes"),
                (macro_steps, "macro_steps"),
                (scheduled_steps, "scheduled_steps"),
                (events, "events"),
                (partitions, "partitions"),
            )
        )
        (
            self.cohort_cases,
            self.state_values,
            self.checkpoint_arrays,
            self.checkpoint_bytes,
            self.macro_steps,
            self.scheduled_steps,
            self.events,
            self.partitions,
        ) = values
        self.request_id = canonical_fingerprint(
            {"kind": "cardiovascular-capacity-request", "values": values}
        )


class CardiovascularCapacityAdmission(StrictModule, NonTrainableState):
    """Deterministic admission evidence; refusal never truncates a request."""

    capacity_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    eligible: bool = eqx.field(static=True)
    exceeded_resources: tuple[str, ...] = eqx.field(static=True)
    status: CardiovascularRuntimeStatus = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)

    def __init__(
        self,
        capacity_id: str,
        request_id: str,
        eligible: bool,
        exceeded_resources: Sequence[str],
        /,
    ):
        capacity = _identifier(capacity_id, "capacity_id")
        request = _identifier(request_id, "request_id")
        exceeded = tuple(sorted(_identifier(v, "resource") for v in exceeded_resources))
        if len(set(exceeded)) != len(exceeded):
            raise ValueError("Exceeded capacity resources must be unique.")
        eligible_ = bool(eligible)
        if eligible_ == bool(exceeded):
            raise ValueError("Capacity eligibility and exceeded resources disagree.")
        status = (
            CardiovascularRuntimeStatus.SUCCESS
            if eligible_
            else CardiovascularRuntimeStatus.CAPACITY_REFUSED
        )
        self.capacity_id = capacity
        self.request_id = request
        self.eligible = eligible_
        self.exceeded_resources = exceeded
        self.status = status
        self.admission_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-capacity-admission",
                "capacity": capacity,
                "request": request,
                "eligible": eligible_,
                "exceeded": exceeded,
            }
        )


def admit_cardiovascular_capacity(
    capacity: CardiovascularCapacityManifest,
    request: CardiovascularCapacityRequest,
    /,
) -> CardiovascularCapacityAdmission:
    """Evaluate every capacity dimension and return one atomic admission decision."""

    if not isinstance(capacity, CardiovascularCapacityManifest):
        raise TypeError("capacity must be CardiovascularCapacityManifest.")
    if not isinstance(request, CardiovascularCapacityRequest):
        raise TypeError("request must be CardiovascularCapacityRequest.")
    observed = (
        request.cohort_cases,
        request.state_values,
        request.checkpoint_arrays,
        request.checkpoint_bytes,
        request.macro_steps,
        request.scheduled_steps,
        request.events,
        request.partitions,
    )
    limits = (
        capacity.maximum_cohort_cases,
        capacity.maximum_state_values,
        capacity.maximum_checkpoint_arrays,
        capacity.maximum_checkpoint_bytes,
        capacity.maximum_macro_steps,
        capacity.maximum_scheduled_steps,
        capacity.maximum_events,
        capacity.maximum_partitions,
    )
    names = (
        "cohort_cases",
        "state_values",
        "checkpoint_arrays",
        "checkpoint_bytes",
        "macro_steps",
        "scheduled_steps",
        "events",
        "partitions",
    )
    exceeded = tuple(
        name
        for name, value, limit in zip(names, observed, limits, strict=True)
        if value > limit
    )
    return CardiovascularCapacityAdmission(
        capacity.capacity_id, request.request_id, not exceeded, exceeded
    )


class CardiovascularSerialExecution(StrictModule, NonTrainableState):
    """One explicitly selected JAX device."""

    device_index: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(self, device_index: int = 0, /):
        index = _nonnegative_integer(device_index, "device_index")
        self.device_index = index
        self.route_id = canonical_fingerprint(
            {"kind": "cardiovascular-serial-execution", "device_index": index}
        )


class CardiovascularCohortExecution(StrictModule, NonTrainableState):
    """Single-device execution-pool route for independent cases."""

    lane_count: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(self, lane_count: int, /):
        lanes = _positive_integer(lane_count, "lane_count")
        self.lane_count = lanes
        self.route_id = canonical_fingerprint(
            {"kind": "cardiovascular-cohort-execution", "lane_count": lanes}
        )


class CardiovascularDistributedReferenceExecution(StrictModule, NonTrainableState):
    """Local reference semantics for an existing distributed FEM phase plan."""

    partition_count: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(self, partition_count: int, /):
        count = _positive_integer(partition_count, "partition_count")
        self.partition_count = count
        self.route_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-distributed-reference-execution",
                "partition_count": count,
            }
        )


class CardiovascularDistributedCollectiveExecution(StrictModule, NonTrainableState):
    """Exact JAX device-mesh request for owned-array FEM execution."""

    partition_count: int = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    process_count: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        partition_count: int,
        axis_name: str,
        /,
        *,
        process_count: int = 1,
    ):
        count = _positive_integer(partition_count, "partition_count")
        axis = _identifier(axis_name, "axis_name")
        processes = _positive_integer(process_count, "process_count")
        if processes > count:
            raise ValueError("process_count cannot exceed partition_count.")
        self.partition_count = count
        self.axis_name = axis
        self.process_count = processes
        self.route_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-distributed-collective-execution",
                "partition_count": count,
                "axis_name": axis,
                "process_count": processes,
            }
        )


CardiovascularExecutionRoute: TypeAlias = (
    CardiovascularSerialExecution
    | CardiovascularCohortExecution
    | CardiovascularDistributedReferenceExecution
    | CardiovascularDistributedCollectiveExecution
)


class CardiovascularExecutionManifest(StrictModule, NonTrainableState):
    """Immutable case, topology, numeric, route, and capacity execution identity."""

    case_manifest_id: str = eqx.field(static=True)
    analysis_plan_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    solver_policy_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    capacity: CardiovascularCapacityManifest
    route: CardiovascularExecutionRoute
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        case_manifest_id: str,
        analysis_plan_id: str,
        numeric_revision_id: str,
        topology_id: str,
        solver_policy_id: str,
        precision_policy_id: str,
        backend: str,
        capacity: CardiovascularCapacityManifest,
        route: CardiovascularExecutionRoute,
    ):
        identifiers = tuple(
            _identifier(value, name)
            for value, name in (
                (case_manifest_id, "case_manifest_id"),
                (analysis_plan_id, "analysis_plan_id"),
                (numeric_revision_id, "numeric_revision_id"),
                (topology_id, "topology_id"),
                (solver_policy_id, "solver_policy_id"),
                (precision_policy_id, "precision_policy_id"),
                (backend, "backend"),
            )
        )
        if not isinstance(capacity, CardiovascularCapacityManifest):
            raise TypeError("capacity must be CardiovascularCapacityManifest.")
        if not isinstance(
            route,
            (
                CardiovascularSerialExecution,
                CardiovascularCohortExecution,
                CardiovascularDistributedReferenceExecution,
                CardiovascularDistributedCollectiveExecution,
            ),
        ):
            raise TypeError("route must be a cardiovascular execution route.")
        (
            self.case_manifest_id,
            self.analysis_plan_id,
            self.numeric_revision_id,
            self.topology_id,
            self.solver_policy_id,
            self.precision_policy_id,
            self.backend,
        ) = identifiers
        self.capacity = capacity
        self.route = route
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-execution-manifest",
                "case": identifiers[0],
                "analysis": identifiers[1],
                "numeric_revision": identifiers[2],
                "topology": identifiers[3],
                "solver": identifiers[4],
                "precision": identifiers[5],
                "backend": identifiers[6],
                "capacity": capacity.capacity_id,
                "route": route.route_id,
            }
        )


class CardiovascularSingleDeviceEvidence(StrictModule, NonTrainableState):
    """Observed eligibility of an explicitly selected single-device route."""

    execution_manifest_id: str = eqx.field(static=True)
    eligible: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    platform: str = eqx.field(static=True)
    device_kind: str = eqx.field(static=True)
    device_id: int = eqx.field(static=True)
    process_index: int = eqx.field(static=True)
    visible_backend_devices: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        execution_manifest_id: str,
        eligible: bool,
        reason: str,
        platform: str,
        device_kind: str,
        device_id: int,
        process_index: int,
        visible_backend_devices: int,
    ):
        execution = _identifier(execution_manifest_id, "execution_manifest_id")
        reason_ = _identifier(reason, "reason")
        platform_ = _identifier(platform, "platform")
        kind = _identifier(device_kind, "device_kind")
        device_id_ = int(device_id)
        process = int(process_index)
        visible = _nonnegative_integer(visible_backend_devices, "visible_backend_devices")
        if device_id_ < -1 or process < -1:
            raise ValueError(
                "Unavailable device identifiers use -1; others are nonnegative."
            )
        eligible_ = bool(eligible)
        if eligible_ and (device_id_ < 0 or process < 0 or visible < 1):
            raise ValueError(
                "Eligible single-device evidence requires an observed device."
            )
        self.execution_manifest_id = execution
        self.eligible = eligible_
        self.reason = reason_
        self.platform = platform_
        self.device_kind = kind
        self.device_id = device_id_
        self.process_index = process
        self.visible_backend_devices = visible
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-single-device-evidence",
                "execution": execution,
                "eligible": eligible_,
                "reason": reason_,
                "platform": platform_,
                "device_kind": kind,
                "device_id": device_id_,
                "process_index": process,
                "visible_backend_devices": visible,
            }
        )


def observe_single_device_runtime(
    execution: CardiovascularExecutionManifest, /
) -> CardiovascularSingleDeviceEvidence:
    """Observe a selected JAX device; distributed routes are explicitly ineligible."""

    if not isinstance(execution, CardiovascularExecutionManifest):
        raise TypeError("execution must be CardiovascularExecutionManifest.")
    all_devices = tuple(jax.devices())
    devices = tuple(
        device for device in all_devices if device.platform == execution.backend
    )
    if isinstance(
        execution.route,
        (
            CardiovascularDistributedReferenceExecution,
            CardiovascularDistributedCollectiveExecution,
        ),
    ):
        return CardiovascularSingleDeviceEvidence(
            execution_manifest_id=execution.manifest_id,
            eligible=False,
            reason="distributed-route",
            platform=execution.backend,
            device_kind="unavailable",
            device_id=-1,
            process_index=-1,
            visible_backend_devices=len(devices),
        )
    index = (
        execution.route.device_index
        if isinstance(execution.route, CardiovascularSerialExecution)
        else 0
    )
    if index >= len(devices):
        return CardiovascularSingleDeviceEvidence(
            execution_manifest_id=execution.manifest_id,
            eligible=False,
            reason=(
                "backend-unavailable" if not devices else "device-index-out-of-range"
            ),
            platform=execution.backend,
            device_kind="unavailable",
            device_id=-1,
            process_index=-1,
            visible_backend_devices=len(devices),
        )
    device = devices[index]
    return CardiovascularSingleDeviceEvidence(
        execution_manifest_id=execution.manifest_id,
        eligible=True,
        reason="eligible",
        platform=device.platform,
        device_kind=device.device_kind,
        device_id=int(device.id),
        process_index=int(device.process_index),
        visible_backend_devices=len(devices),
    )


class CardiovascularCheckpointRecord(StrictModule, NonTrainableState):
    """Validated lifecycle archive bound to one exact execution manifest."""

    archive: LifecycleArchive
    execution_manifest_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)
    parent_checkpoint_id: str | None = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        archive: LifecycleArchive,
        execution_manifest_id: str,
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        /,
    ):
        if not isinstance(archive, LifecycleArchive):
            raise TypeError("archive must be LifecycleArchive.")
        execution = _identifier(execution_manifest_id, "execution_manifest_id")
        checkpoint = _identifier(checkpoint_id, "checkpoint_id")
        parent = (
            None
            if parent_checkpoint_id is None
            else _identifier(parent_checkpoint_id, "parent_checkpoint_id")
        )
        self.archive = archive
        self.execution_manifest_id = execution
        self.checkpoint_id = checkpoint
        self.parent_checkpoint_id = parent
        self.record_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-checkpoint-record",
                "archive": archive.archive_id,
                "execution": execution,
                "checkpoint": checkpoint,
                "parent": parent,
            }
        )

    @property
    def arrays(self) -> Mapping[str, np.ndarray]:
        return self.archive.arrays


class CardiovascularLifecycleCheckpointCodec(StrictModule, NonTrainableState):
    """Atomic cardiovascular checkpoint codec over :class:`LifecycleArchive`."""

    execution: CardiovascularExecutionManifest
    codec_id: str = eqx.field(static=True)

    def __init__(self, execution: CardiovascularExecutionManifest, /):
        if not isinstance(execution, CardiovascularExecutionManifest):
            raise TypeError("execution must be CardiovascularExecutionManifest.")
        self.execution = execution
        self.codec_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-lifecycle-checkpoint-codec",
                "execution": execution.manifest_id,
            }
        )

    def write(
        self,
        path: str | Path,
        arrays: Mapping[str, ArrayLike],
        /,
        *,
        checkpoint_id: str,
        committed: bool,
        layout_ids: Mapping[str, Sequence[str]] | None = None,
        parent_checkpoint_id: str | None = None,
        diagnostic_ids: Sequence[str] = (),
    ) -> CardiovascularCheckpointRecord:
        """Atomically publish only a committed, capacity-admitted state."""

        if not committed:
            raise CardiovascularRuntimeError(
                CardiovascularRuntimeStatus.CHECKPOINT_REFUSED,
                phase="checkpoint-write",
                entity_ids=(self.execution.manifest_id,),
            )
        if not isinstance(arrays, Mapping) or not arrays:
            raise ValueError("Checkpoint arrays must be a non-empty mapping.")
        checkpoint = _identifier(checkpoint_id, "checkpoint_id")
        parent = (
            None
            if parent_checkpoint_id is None
            else _identifier(parent_checkpoint_id, "parent_checkpoint_id")
        )
        diagnostics = tuple(_identifier(v, "diagnostic_id") for v in diagnostic_ids)
        if len(set(diagnostics)) != len(diagnostics):
            raise ValueError("Checkpoint diagnostic IDs must be unique.")
        payloads = {
            _identifier(name, "payload_name"): _checkpoint_array(value)
            for name, value in arrays.items()
        }
        if len(payloads) != len(arrays):
            raise ValueError(
                "Checkpoint payload names must be unique after normalization."
            )
        total_values = sum(int(value.size) for value in payloads.values())
        total_bytes = sum(payload_byte_count(value) for value in payloads.values())
        admission = admit_cardiovascular_capacity(
            self.execution.capacity,
            CardiovascularCapacityRequest(
                state_values=total_values,
                checkpoint_arrays=len(payloads),
                checkpoint_bytes=total_bytes,
            ),
        )
        if not admission.eligible:
            raise CardiovascularRuntimeError(
                CardiovascularRuntimeStatus.CHECKPOINT_REFUSED,
                phase="checkpoint-write",
                entity_ids=(self.execution.manifest_id, admission.admission_id),
            )
        layouts = {} if layout_ids is None else dict(layout_ids)
        if set(layouts) - set(payloads):
            raise ValueError("Checkpoint layout IDs reference an unknown payload.")
        shards = tuple(
            CheckpointShard(
                name,
                payload_digest(payloads[name]),
                payload_byte_count(payloads[name]),
                tuple(_identifier(v, "layout_id") for v in layouts.get(name, ())),
            )
            for name in sorted(payloads)
        )
        manifest = CheckpointManifest(
            checkpoint,
            self.execution.analysis_plan_id,
            self.execution.numeric_revision_id,
            self.execution.manifest_id,
            shards,
            complete=True,
            parent_checkpoint_id=parent,
            diagnostic_ids=diagnostics,
        )
        archive = create_lifecycle_archive(path, manifest=manifest, arrays=payloads)
        return CardiovascularCheckpointRecord(
            archive,
            self.execution.manifest_id,
            checkpoint,
            parent,
        )

    def read(self, path: str | Path, /) -> CardiovascularCheckpointRecord:
        """Open and verify archive checksums and exact execution compatibility."""

        archive = open_lifecycle_archive(
            path, limits=_checkpoint_archive_limits(self.execution.capacity)
        )
        manifest = archive.manifest
        if not isinstance(manifest, CheckpointManifest):
            raise CardiovascularRuntimeError(
                CardiovascularRuntimeStatus.CHECKPOINT_MISMATCH,
                phase="checkpoint-read",
                entity_ids=(self.execution.manifest_id,),
            )
        expected = (
            (manifest.analysis_plan_id, self.execution.analysis_plan_id),
            (manifest.numeric_revision_id, self.execution.numeric_revision_id),
            (manifest.execution_plan_id, self.execution.manifest_id),
        )
        if any(observed != required for observed, required in expected):
            raise CardiovascularRuntimeError(
                CardiovascularRuntimeStatus.CHECKPOINT_MISMATCH,
                phase="checkpoint-read",
                entity_ids=(self.execution.manifest_id, manifest.checkpoint_id),
            )
        if any(
            value.dtype.kind not in "biufc" or bool(np.any(~np.isfinite(value)))
            for value in archive.arrays.values()
        ):
            raise ArrayArchiveCorruptionError(
                "Cardiovascular checkpoint payloads must be finite numeric arrays."
            )
        total_values = sum(int(value.size) for value in archive.arrays.values())
        total_bytes = sum(payload_byte_count(value) for value in archive.arrays.values())
        admission = admit_cardiovascular_capacity(
            self.execution.capacity,
            CardiovascularCapacityRequest(
                state_values=total_values,
                checkpoint_arrays=len(archive.arrays),
                checkpoint_bytes=total_bytes,
            ),
        )
        if not admission.eligible:
            raise CardiovascularRuntimeError(
                CardiovascularRuntimeStatus.CHECKPOINT_REFUSED,
                phase="checkpoint-read",
                entity_ids=(self.execution.manifest_id, admission.admission_id),
            )
        return CardiovascularCheckpointRecord(
            archive,
            self.execution.manifest_id,
            manifest.checkpoint_id,
            manifest.parent_checkpoint_id,
        )


class PreparedCardiovascularCohort(StrictModule, NonTrainableState):
    """Canonical case order and semantic RNG keys for an execution pool."""

    execution_manifest_id: str = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    lane_count: int = eqx.field(static=True)
    signature: PoolExecutionSignature
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        execution_manifest_id: str,
        case_ids: Sequence[str],
        lane_count: int,
        signature: PoolExecutionSignature,
        /,
    ):
        execution = _identifier(execution_manifest_id, "execution_manifest_id")
        cases = tuple(sorted(_identifier(value, "case_id") for value in case_ids))
        lanes = _positive_integer(lane_count, "lane_count")
        if not cases or len(set(cases)) != len(cases):
            raise ValueError("Cohort case IDs must be non-empty and unique.")
        if lanes > len(cases):
            lanes = len(cases)
        if not isinstance(signature, PoolExecutionSignature):
            raise TypeError("signature must be PoolExecutionSignature.")
        self.execution_manifest_id = execution
        self.case_ids = cases
        self.lane_count = lanes
        self.signature = signature
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-cohort",
                "execution": execution,
                "cases": cases,
                "lanes": lanes,
                "signature": signature.signature_id,
            }
        )


class CardiovascularCohortCaseCandidate(StrictModule, Generic[Value]):
    """One pool task result with an explicit fail-closed acceptance bit."""

    value: Value
    accepted: bool = eqx.field(static=True)
    status: CardiovascularRuntimeStatus = eqx.field(static=True)

    def __init__(
        self,
        value: Value,
        /,
        *,
        accepted: bool = True,
        status: CardiovascularRuntimeStatus = CardiovascularRuntimeStatus.SUCCESS,
    ):
        accepted_ = bool(accepted)
        status_ = CardiovascularRuntimeStatus(status)
        if accepted_ != (status_ is CardiovascularRuntimeStatus.SUCCESS):
            raise ValueError("Cohort task acceptance and status disagree.")
        self.value = value
        self.accepted = accepted_
        self.status = status_


class CardiovascularCohortEvidence(StrictModule, NonTrainableState):
    """Task-order-independent keys and canonical per-case completion evidence."""

    semantic_keys: Array
    accepted: Array
    completion_wave: Array
    execution_manifest_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    status: CardiovascularRuntimeStatus = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        semantic_keys: ArrayLike,
        accepted: ArrayLike,
        completion_wave: ArrayLike,
        execution_manifest_id: str,
        prepared_id: str,
        status: CardiovascularRuntimeStatus,
        /,
    ):
        keys = jnp.asarray(semantic_keys, dtype=jnp.uint32)
        accepted_ = jnp.asarray(accepted, dtype=bool)
        waves = jnp.asarray(completion_wave, dtype=jnp.int32)
        if keys.ndim != 2 or keys.shape[1:] != (2,):
            raise ValueError("Cohort semantic keys must have shape (case, 2).")
        if accepted_.shape != keys.shape[:1] or waves.shape != keys.shape[:1]:
            raise ValueError("Cohort evidence arrays must share the case axis.")
        execution = _identifier(execution_manifest_id, "execution_manifest_id")
        prepared = _identifier(prepared_id, "prepared_id")
        status_ = CardiovascularRuntimeStatus(status)
        all_accepted = bool(np.all(np.asarray(accepted_)))
        if all_accepted != (status_ is CardiovascularRuntimeStatus.SUCCESS):
            raise ValueError("Cohort evidence status disagrees with task acceptance.")
        self.semantic_keys = keys
        self.accepted = accepted_
        self.completion_wave = waves
        self.execution_manifest_id = execution
        self.prepared_id = prepared
        self.status = status_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-cohort-evidence",
                "execution": execution,
                "prepared": prepared,
                "status": status_.value,
                "arrays": array_tree_fingerprint((keys, accepted_, waves)),
            }
        )


class CardiovascularCohortResult(StrictModule, Generic[Value]):
    """Canonical case-indexed pool result; partial candidates are never committed."""

    case_ids: tuple[str, ...] = eqx.field(static=True)
    values: tuple[Value, ...]
    committed: bool = eqx.field(static=True)
    evidence: CardiovascularCohortEvidence


CohortExecutor: TypeAlias = Callable[
    [str, Array], CardiovascularCohortCaseCandidate[Value]
]


def prepare_cardiovascular_cohort(
    execution: CardiovascularExecutionManifest,
    case_ids: Sequence[str],
    /,
) -> PreparedCardiovascularCohort:
    """Prepare canonical case order for the generic execution-pool substrate."""

    if not isinstance(execution, CardiovascularExecutionManifest):
        raise TypeError("execution must be CardiovascularExecutionManifest.")
    if not isinstance(execution.route, CardiovascularCohortExecution):
        raise TypeError("Cohort preparation requires CardiovascularCohortExecution.")
    cases = tuple(case_ids)
    admission = admit_cardiovascular_capacity(
        execution.capacity,
        CardiovascularCapacityRequest(cohort_cases=len(cases)),
    )
    if not admission.eligible:
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.CAPACITY_REFUSED,
            phase="cohort-prepare",
            entity_ids=(execution.manifest_id, admission.admission_id),
        )
    signature = PoolExecutionSignature(
        topology_id=execution.topology_id,
        method_id=execution.solver_policy_id,
        precision_id=execution.precision_policy_id,
        backend_id=execution.backend,
        shard_count=1,
    )
    return PreparedCardiovascularCohort(
        execution.manifest_id,
        cases,
        execution.route.lane_count,
        signature,
    )


def execute_cardiovascular_cohort(
    prepared: PreparedCardiovascularCohort,
    root_key: Array,
    executor: CohortExecutor[Value],
    /,
) -> CardiovascularCohortResult[Value]:
    """Execute deterministic refill waves with semantic, lane-independent keys."""

    if not isinstance(prepared, PreparedCardiovascularCohort):
        raise TypeError("prepared must be PreparedCardiovascularCohort.")
    if not callable(executor):
        raise TypeError("executor must be callable.")
    count = len(prepared.case_ids)
    semantic_indices = jnp.arange(count, dtype=jnp.uint32)
    keys = semantic_task_keys(jnp.asarray(root_key), semantic_indices)
    key_words = jax.random.key_data(keys)
    lane_ids = np.arange(prepared.lane_count, dtype=np.int32)
    next_task = prepared.lane_count
    completed = 0
    values: list[Any] = [None] * count
    accepted = np.zeros((count,), dtype=bool)
    waves = np.zeros((count,), dtype=np.int32)
    wave = 0
    while completed < count:
        active = lane_ids < count
        for lane in range(prepared.lane_count):
            task = int(lane_ids[lane])
            if not active[lane]:
                continue
            candidate = executor(prepared.case_ids[task], keys[task])
            if not isinstance(candidate, CardiovascularCohortCaseCandidate):
                raise TypeError("executor must return CardiovascularCohortCaseCandidate.")
            values[task] = candidate.value
            accepted[task] = candidate.accepted
            waves[task] = wave
        refill = refill_completed_tasks(
            jnp.asarray(lane_ids),
            jnp.asarray(active),
            jnp.asarray(next_task, dtype=jnp.int32),
            jnp.asarray(completed, dtype=jnp.int32),
            count,
        )
        lane_ids = np.asarray(refill.task_ids, dtype=np.int32)
        next_task = int(refill.next_task)
        completed = int(refill.completed)
        wave += 1
    status = (
        CardiovascularRuntimeStatus.SUCCESS
        if bool(np.all(accepted))
        else CardiovascularRuntimeStatus.COMMIT_REFUSED
    )
    evidence = CardiovascularCohortEvidence(
        key_words,
        accepted,
        waves,
        prepared.execution_manifest_id,
        prepared.prepared_id,
        status,
    )
    return CardiovascularCohortResult(
        case_ids=prepared.case_ids,
        values=tuple(values) if status is CardiovascularRuntimeStatus.SUCCESS else (),
        committed=status is CardiovascularRuntimeStatus.SUCCESS,
        evidence=evidence,
    )


class CardiovascularDistributedCapability(StrictModule, NonTrainableState):
    """Observed process/device-mesh eligibility for distributed execution."""

    reference_eligible: bool = eqx.field(static=True)
    transport_eligible: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    requested_device_count: int = eqx.field(static=True)
    available_device_count: int = eqx.field(static=True)
    requested_process_count: int = eqx.field(static=True)
    available_process_count: int = eqx.field(static=True)
    device_ids: tuple[int, ...] = eqx.field(static=True)
    device_process_indices: tuple[int, ...] = eqx.field(static=True)
    device_mesh_id: str | None = eqx.field(static=True)
    transport_id: str | None = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reference_eligible: bool,
        transport_eligible: bool,
        reason: str,
        backend: str,
        requested_device_count: int,
        available_device_count: int,
        requested_process_count: int = 1,
        available_process_count: int = 1,
        device_ids: Sequence[int] = (),
        device_process_indices: Sequence[int] = (),
        device_mesh_id: str | None = None,
        transport_id: str | None = None,
    ):
        reference = bool(reference_eligible)
        transport = bool(transport_eligible)
        reason_ = _identifier(reason, "reason")
        backend_ = _identifier(backend, "backend")
        requested = _positive_integer(requested_device_count, "requested_device_count")
        available = _nonnegative_integer(available_device_count, "available_device_count")
        requested_processes = _positive_integer(
            requested_process_count, "requested_process_count"
        )
        available_processes = _positive_integer(
            available_process_count, "available_process_count"
        )
        identifiers = tuple(int(value) for value in device_ids)
        process_indices = tuple(int(value) for value in device_process_indices)
        mesh = (
            None
            if device_mesh_id is None
            else _identifier(device_mesh_id, "device_mesh_id")
        )
        transport_identity = (
            None if transport_id is None else _identifier(transport_id, "transport_id")
        )
        if reference and transport:
            raise ValueError(
                "Reference and collective transport eligibility are exclusive."
            )
        device_keys = tuple(zip(process_indices, identifiers, strict=True))
        if (
            len(process_indices) != len(identifiers)
            or any(value < 0 for value in (*identifiers, *process_indices))
            or len(set(device_keys)) != len(device_keys)
        ):
            raise ValueError(
                "Distributed process/device identities must be unique and nonnegative."
            )
        if transport:
            if (
                len(identifiers) != requested
                or available < requested
                or available_processes < requested_processes
                or len(set(process_indices)) != requested_processes
                or mesh is None
                or transport_identity is None
            ):
                raise ValueError(
                    "Eligible collective transport requires an exact observed "
                    "process/device mesh."
                )
        elif (
            identifiers
            or process_indices
            or mesh is not None
            or transport_identity is not None
        ):
            raise ValueError(
                "Ineligible transport cannot retain a device mesh or transport ID."
            )
        self.reference_eligible = reference
        self.transport_eligible = transport
        self.reason = reason_
        self.backend = backend_
        self.requested_device_count = requested
        self.available_device_count = available
        self.requested_process_count = requested_processes
        self.available_process_count = available_processes
        self.device_ids = identifiers
        self.device_process_indices = process_indices
        self.device_mesh_id = mesh
        self.transport_id = transport_identity
        self.capability_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-distributed-capability",
                "reference_eligible": reference,
                "transport_eligible": transport,
                "reason": reason_,
                "backend": backend_,
                "requested_device_count": requested,
                "available_device_count": available,
                "requested_process_count": requested_processes,
                "available_process_count": available_processes,
                "device_ids": identifiers,
                "device_process_indices": process_indices,
                "device_mesh_id": mesh,
                "transport_id": transport_identity,
            }
        )


class CardiovascularDistributedContract(StrictModule, NonTrainableState):
    """Existing FEM phase and replay identities under an explicit route capability."""

    execution_manifest_id: str = eqx.field(static=True)
    solver_policy_id: str = eqx.field(static=True)
    phase_plan: FiniteElementDistributedPhasePlan
    replay_schedule: PreparedReplaySchedule
    route: (
        CardiovascularDistributedReferenceExecution
        | CardiovascularDistributedCollectiveExecution
    )
    capability: CardiovascularDistributedCapability
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        execution_manifest_id: str,
        solver_policy_id: str,
        phase_plan: FiniteElementDistributedPhasePlan,
        replay_schedule: PreparedReplaySchedule,
        route: (
            CardiovascularDistributedReferenceExecution
            | CardiovascularDistributedCollectiveExecution
        ),
        capability: CardiovascularDistributedCapability,
        /,
    ):
        solver_policy = _identifier(solver_policy_id, "solver_policy_id")
        execution = _identifier(execution_manifest_id, "execution_manifest_id")
        if not isinstance(phase_plan, FiniteElementDistributedPhasePlan):
            raise TypeError("phase_plan must be FiniteElementDistributedPhasePlan.")
        if not isinstance(replay_schedule, PreparedReplaySchedule):
            raise TypeError("replay_schedule must be PreparedReplaySchedule.")
        if not isinstance(
            route,
            (
                CardiovascularDistributedReferenceExecution,
                CardiovascularDistributedCollectiveExecution,
            ),
        ):
            raise TypeError("route must be a distributed cardiovascular route.")
        if not isinstance(capability, CardiovascularDistributedCapability):
            raise TypeError("capability must be CardiovascularDistributedCapability.")
        if route.partition_count != phase_plan.partition.part_count:
            raise ValueError("Distributed route and FEM partition counts differ.")
        reference_route = isinstance(route, CardiovascularDistributedReferenceExecution)
        if capability.reference_eligible != reference_route:
            raise ValueError("Distributed route and reference capability are incoherent.")
        if capability.requested_device_count != route.partition_count:
            raise ValueError("Distributed route and capability device counts differ.")
        requested_processes = 1 if reference_route else route.process_count
        if capability.requested_process_count != requested_processes:
            raise ValueError("Distributed route and capability process counts differ.")
        if reference_route and capability.transport_eligible:
            raise ValueError("Local reference routes cannot claim transport.")
        if (
            not reference_route
            and capability.transport_eligible
            and (
                capability.device_mesh_id is None
                or capability.transport_id is None
                or len(capability.device_ids) != route.partition_count
            )
        ):
            raise ValueError("Collective capability lacks an exact device-mesh identity.")
        self.execution_manifest_id = execution
        self.solver_policy_id = solver_policy
        self.phase_plan = phase_plan
        self.replay_schedule = replay_schedule
        self.route = route
        self.capability = capability
        self.contract_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-distributed-contract",
                "execution": execution,
                "solver_policy": solver_policy,
                "phase_plan": phase_plan.plan_id,
                "replay_schedule": replay_schedule.schedule_id,
                "route": route.route_id,
                "capability": capability.capability_id,
            }
        )


def prepare_cardiovascular_distributed_execution(
    execution: CardiovascularExecutionManifest,
    phase_plan: FiniteElementDistributedPhasePlan,
    replay_schedule: PreparedReplaySchedule,
    /,
) -> CardiovascularDistributedContract:
    """Bind an exact process/device mesh to existing FEM and replay plans."""

    if not isinstance(execution, CardiovascularExecutionManifest):
        raise TypeError("execution must be CardiovascularExecutionManifest.")
    route = execution.route
    if not isinstance(
        route,
        (
            CardiovascularDistributedReferenceExecution,
            CardiovascularDistributedCollectiveExecution,
        ),
    ):
        raise TypeError("Distributed preparation requires a distributed route.")
    admission = admit_cardiovascular_capacity(
        execution.capacity,
        CardiovascularCapacityRequest(
            scheduled_steps=replay_schedule.step_count,
            partitions=route.partition_count,
        ),
    )
    if not admission.eligible:
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.CAPACITY_REFUSED,
            phase="distributed-prepare",
            entity_ids=(execution.manifest_id, admission.admission_id),
        )
    backend_devices = tuple(
        device for device in jax.devices() if device.platform == execution.backend
    )
    process_count = int(jax.process_count())
    process_index = int(jax.process_index())
    local_devices = tuple(
        device for device in backend_devices if int(device.process_index) == process_index
    )
    requested_processes = (
        1
        if isinstance(route, CardiovascularDistributedReferenceExecution)
        else route.process_count
    )
    available_devices = (
        len(local_devices) if requested_processes == 1 else len(backend_devices)
    )
    capability_arguments = {
        "backend": execution.backend,
        "requested_device_count": route.partition_count,
        "available_device_count": available_devices,
        "requested_process_count": requested_processes,
        "available_process_count": process_count,
    }
    if isinstance(route, CardiovascularDistributedReferenceExecution):
        capability = CardiovascularDistributedCapability(
            reference_eligible=True,
            transport_eligible=False,
            reason="local-reference-eligible",
            **capability_arguments,
        )
    elif route.process_count > process_count:
        capability = CardiovascularDistributedCapability(
            reference_eligible=False,
            transport_eligible=False,
            reason="insufficient-process-device-mesh",
            **capability_arguments,
        )
    elif route.process_count > 1:
        capability = CardiovascularDistributedCapability(
            reference_eligible=False,
            transport_eligible=False,
            reason="process-spanning-owned-array-mesh-unavailable",
            **capability_arguments,
        )
    elif len(local_devices) < route.partition_count:
        capability = CardiovascularDistributedCapability(
            reference_eligible=False,
            transport_eligible=False,
            reason="insufficient-local-device-mesh",
            **capability_arguments,
        )
    else:
        selected = local_devices[: route.partition_count]
        device_ids = tuple(int(device.id) for device in selected)
        process_indices = tuple(int(device.process_index) for device in selected)
        device_mesh_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-process-device-mesh",
                "backend": execution.backend,
                "axis_name": route.axis_name,
                "requested_process_count": route.process_count,
                "devices": [
                    {
                        "id": int(device.id),
                        "process_index": int(device.process_index),
                        "kind": device.device_kind,
                    }
                    for device in selected
                ],
            }
        )
        transport_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-owned-array-shard-map-transport",
                "device_mesh": device_mesh_id,
                "axis_name": route.axis_name,
                "phase_plan": phase_plan.plan_id,
                "partition": phase_plan.partition.partition_id,
                "worksets": phase_plan.worksets.plan_id,
                "operations": (
                    "owned-array-sharded-ingress",
                    "halo-psum-reconstruction",
                    "distributed-fem-operator-psum",
                    "distributed-fem-transpose-psum",
                    "global-krylov-solve",
                ),
            }
        )
        capability = CardiovascularDistributedCapability(
            reference_eligible=False,
            transport_eligible=True,
            reason="jax-owned-array-shard-map-eligible",
            device_ids=device_ids,
            device_process_indices=process_indices,
            device_mesh_id=device_mesh_id,
            transport_id=transport_id,
            **capability_arguments,
        )
    return CardiovascularDistributedContract(
        execution.manifest_id,
        execution.solver_policy_id,
        phase_plan,
        replay_schedule,
        route,
        capability,
    )


class CardiovascularDistributedReferenceEvidence(StrictModule, NonTrainableState):
    """Partition-local accumulation compared with the serial reference definition."""

    partition_values: Array
    serial_reference: Array
    distributed_reference: Array
    residual_norm: Array
    contract_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        partition_values: ArrayLike,
        serial_reference: ArrayLike,
        distributed_reference: ArrayLike,
        residual_norm: ArrayLike,
        contract_id: str,
        /,
    ):
        partitions = jnp.asarray(partition_values)
        serial = jnp.asarray(serial_reference)
        distributed = jnp.asarray(distributed_reference)
        residual = jnp.asarray(residual_norm)
        if partitions.ndim < 1 or serial.shape != distributed.shape or residual.shape:
            raise ValueError("Distributed reference evidence shapes are invalid.")
        contract = _identifier(contract_id, "contract_id")
        self.partition_values = partitions
        self.serial_reference = serial
        self.distributed_reference = distributed
        self.residual_norm = residual
        self.contract_id = contract
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-distributed-reference-evidence",
                "contract": contract,
                "arrays": array_tree_fingerprint(
                    (partitions, serial, distributed, residual)
                ),
            }
        )


def execute_cardiovascular_distributed_reference(
    contract: CardiovascularDistributedContract,
    cell_values: ArrayLike,
    /,
) -> CardiovascularDistributedReferenceEvidence:
    """Execute owned-local phase semantics and compare with a serial cell sum."""

    if not isinstance(contract, CardiovascularDistributedContract):
        raise TypeError("contract must be CardiovascularDistributedContract.")
    if not contract.capability.reference_eligible or not isinstance(
        contract.route, CardiovascularDistributedReferenceExecution
    ):
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.DISTRIBUTED_INELIGIBLE,
            phase="distributed-reference",
            entity_ids=(contract.contract_id,),
        )
    values = jnp.asarray(cell_values)
    if values.ndim < 1 or values.shape[0] != contract.phase_plan.worksets.cell_count:
        raise ValueError("cell_values must have one leading entry per FEM cell.")
    local = jnp.stack(
        tuple(
            contract.phase_plan.local_contribution(part, values)
            for part in range(contract.route.partition_count)
        ),
        axis=0,
    )
    distributed = jnp.sum(local, axis=0)
    serial = jnp.sum(values, axis=0)
    residual = jnp.sqrt(jnp.sum(jnp.abs(distributed - serial) ** 2))
    return CardiovascularDistributedReferenceEvidence(
        local, serial, distributed, residual, contract.contract_id
    )


class CardiovascularDistributedSolverState(StrictModule, NonTrainableState):
    """Checkpointable owned shards for one exactly bound distributed FEM solve."""

    owned_solution: Array
    owned_right_hand_side: Array
    owned_dof_ids: Array
    owned_valid: Array
    solve_count: int = eqx.field(static=True)
    iteration_count: int = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    execution_manifest_id: str = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)
    phase_plan_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    workset_plan_id: str = eqx.field(static=True)
    dof_map_id: str = eqx.field(static=True)
    finite_element_operator_id: str = eqx.field(static=True)
    distributed_operator_id: str = eqx.field(static=True)
    solver_policy_id: str = eqx.field(static=True)
    solver_plan_id: str = eqx.field(static=True)
    device_mesh_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    checkpoint_id: str | None = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        owned_solution: ArrayLike,
        owned_right_hand_side: ArrayLike,
        owned_dof_ids: ArrayLike,
        owned_valid: ArrayLike,
        /,
        *,
        solve_count: int,
        iteration_count: int,
        successful: bool,
        execution_manifest_id: str,
        contract_id: str,
        phase_plan_id: str,
        partition_id: str,
        workset_plan_id: str,
        dof_map_id: str,
        finite_element_operator_id: str,
        distributed_operator_id: str,
        solver_policy_id: str,
        solver_plan_id: str,
        device_mesh_id: str,
        transport_id: str,
        checkpoint_id: str | None = None,
    ):
        solution = jnp.asarray(owned_solution)
        right_hand_side = jnp.asarray(owned_right_hand_side)
        identifiers = jnp.asarray(owned_dof_ids)
        valid = jnp.asarray(owned_valid)
        if (
            solution.shape != right_hand_side.shape
            or solution.ndim < 2
            or identifiers.ndim != 2
            or valid.shape != identifiers.shape
            or solution.shape[:2] != identifiers.shape
            or identifiers.dtype.kind not in "iu"
            or valid.dtype != jnp.bool_
        ):
            raise ValueError("Distributed solver owned-shard shapes are invalid.")
        if bool(
            np.any(~np.isfinite(np.asarray(solution)))
            or np.any(~np.isfinite(np.asarray(right_hand_side)))
            or np.any(np.asarray(identifiers)[np.asarray(valid)] < 0)
        ):
            raise ValueError("Distributed solver owned shards must be finite and valid.")
        solves = _nonnegative_integer(solve_count, "solve_count")
        iterations = _nonnegative_integer(iteration_count, "iteration_count")
        binding = tuple(
            _identifier(value, name)
            for value, name in (
                (execution_manifest_id, "execution_manifest_id"),
                (contract_id, "contract_id"),
                (phase_plan_id, "phase_plan_id"),
                (partition_id, "partition_id"),
                (workset_plan_id, "workset_plan_id"),
                (dof_map_id, "dof_map_id"),
                (finite_element_operator_id, "finite_element_operator_id"),
                (distributed_operator_id, "distributed_operator_id"),
                (solver_policy_id, "solver_policy_id"),
                (solver_plan_id, "solver_plan_id"),
                (device_mesh_id, "device_mesh_id"),
                (transport_id, "transport_id"),
            )
        )
        checkpoint = (
            None if checkpoint_id is None else _identifier(checkpoint_id, "checkpoint_id")
        )
        self.owned_solution = solution
        self.owned_right_hand_side = right_hand_side
        self.owned_dof_ids = identifiers
        self.owned_valid = valid
        self.solve_count = solves
        self.iteration_count = iterations
        self.successful = bool(successful)
        (
            self.execution_manifest_id,
            self.contract_id,
            self.phase_plan_id,
            self.partition_id,
            self.workset_plan_id,
            self.dof_map_id,
            self.finite_element_operator_id,
            self.distributed_operator_id,
            self.solver_policy_id,
            self.solver_plan_id,
            self.device_mesh_id,
            self.transport_id,
        ) = binding
        self.checkpoint_id = checkpoint
        self.state_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-distributed-solver-state",
                "binding": binding,
                "solve_count": solves,
                "iteration_count": iterations,
                "successful": bool(successful),
                "checkpoint": checkpoint,
                "arrays": array_tree_fingerprint(
                    (solution, right_hand_side, identifiers, valid)
                ),
            }
        )

    @property
    def binding_ids(self) -> tuple[str, ...]:
        return (
            self.contract_id,
            self.phase_plan_id,
            self.partition_id,
            self.workset_plan_id,
            self.dof_map_id,
            self.finite_element_operator_id,
            self.distributed_operator_id,
            self.solver_policy_id,
            self.solver_plan_id,
            self.device_mesh_id,
            self.transport_id,
        )


class CardiovascularDistributedCollectiveEvidence(StrictModule, NonTrainableState):
    """Owned-shard FEM action, transpose, halo, solve, and restart evidence."""

    solver_state: CardiovascularDistributedSolverState
    serial_operator_action: Array
    distributed_operator_action: Array
    serial_transpose_action: Array
    distributed_transpose_action: Array
    operator_residual_norm: Array
    halo_residual_norm: Array
    transpose_residual_norm: Array
    solver_residual_norm: Array
    solver_serial_residual_norm: Array
    owned_value_count: int = eqx.field(static=True)
    halo_value_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        solver_state: CardiovascularDistributedSolverState,
        serial_operator_action: ArrayLike,
        distributed_operator_action: ArrayLike,
        serial_transpose_action: ArrayLike,
        distributed_transpose_action: ArrayLike,
        operator_residual_norm: ArrayLike,
        halo_residual_norm: ArrayLike,
        transpose_residual_norm: ArrayLike,
        solver_residual_norm: ArrayLike,
        solver_serial_residual_norm: ArrayLike,
        /,
        *,
        owned_value_count: int,
        halo_value_count: int,
    ):
        if not isinstance(solver_state, CardiovascularDistributedSolverState):
            raise TypeError("solver_state must be CardiovascularDistributedSolverState.")
        serial = jnp.asarray(serial_operator_action)
        distributed = jnp.asarray(distributed_operator_action)
        serial_transpose = jnp.asarray(serial_transpose_action)
        distributed_transpose = jnp.asarray(distributed_transpose_action)
        residuals = tuple(
            jnp.asarray(value)
            for value in (
                operator_residual_norm,
                halo_residual_norm,
                transpose_residual_norm,
                solver_residual_norm,
                solver_serial_residual_norm,
            )
        )
        if (
            serial.shape != distributed.shape
            or serial_transpose.shape != distributed_transpose.shape
            or any(value.shape for value in residuals)
        ):
            raise ValueError("Collective FEM execution evidence shapes are invalid.")
        if bool(np.any(~np.isfinite(np.asarray(residuals)))):
            raise ValueError("Collective FEM execution residuals must be finite.")
        owned_count = _positive_integer(owned_value_count, "owned_value_count")
        halo_count = _nonnegative_integer(halo_value_count, "halo_value_count")
        self.solver_state = solver_state
        self.serial_operator_action = serial
        self.distributed_operator_action = distributed
        self.serial_transpose_action = serial_transpose
        self.distributed_transpose_action = distributed_transpose
        (
            self.operator_residual_norm,
            self.halo_residual_norm,
            self.transpose_residual_norm,
            self.solver_residual_norm,
            self.solver_serial_residual_norm,
        ) = residuals
        self.owned_value_count = owned_count
        self.halo_value_count = halo_count
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-distributed-collective-evidence",
                "state": solver_state.state_id,
                "binding": solver_state.binding_ids,
                "owned_value_count": owned_count,
                "halo_value_count": halo_count,
                "arrays": array_tree_fingerprint(
                    (
                        serial,
                        distributed,
                        serial_transpose,
                        distributed_transpose,
                        *residuals,
                    )
                ),
            }
        )

    @property
    def contract_id(self) -> str:
        return self.solver_state.contract_id

    @property
    def device_mesh_id(self) -> str:
        return self.solver_state.device_mesh_id

    @property
    def finite_element_operator_id(self) -> str:
        return self.solver_state.finite_element_operator_id

    @property
    def distributed_operator_id(self) -> str:
        return self.solver_state.distributed_operator_id

    @property
    def partition_id(self) -> str:
        return self.solver_state.partition_id

    @property
    def transport_id(self) -> str:
        return self.solver_state.transport_id

    @property
    def solver_plan_id(self) -> str:
        return self.solver_state.solver_plan_id


class _OwnedShardedFiniteElementOperator(AbstractLinearOperator):
    """Global FEM semantics from exactly-once owned coordinate contributions."""

    finite_element_operator: AbstractLinearOperator
    forward: DistributedFiniteElementOperator
    transposed: DistributedFiniteElementOperator
    dof_owner: Array
    axis_name: str = eqx.field(static=True)

    def __init__(
        self,
        finite_element_operator: AbstractLinearOperator,
        dof_owner: ArrayLike,
        axis_name: str,
        /,
    ):
        owners = jnp.asarray(dof_owner)
        axis = _identifier(axis_name, "axis_name")
        collective = JaxCollectiveBackend(axis)
        transposed = transpose(finite_element_operator)
        self.source = finite_element_operator.source
        self.target = finite_element_operator.target
        self.properties = finite_element_operator.properties
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=False,
            materialize=False,
        )
        self.batch_shape = ()
        self.finite_element_operator = finite_element_operator
        self.forward = DistributedFiniteElementOperator(
            finite_element_operator, collective
        )
        self.transposed = DistributedFiniteElementOperator(transposed, collective)
        self.dof_owner = owners
        self.axis_name = axis
        self.operator_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-owned-sharded-fem-operator",
                "finite_element_operator": finite_element_operator.operator_id,
                "distributed_forward": self.forward.operator_id,
                "distributed_transpose": self.transposed.operator_id,
                "dof_owner": array_tree_fingerprint(owners),
                "axis_name": axis,
            }
        )

    def _owned_part(self, value: Array, /) -> Array:
        part = jax.lax.axis_index(self.axis_name)
        mask = self.dof_owner == part
        mask = mask.reshape(mask.shape + (1,) * (value.ndim - 1))
        return jnp.where(mask, value, jnp.zeros((), dtype=value.dtype))

    def mv(self, vector: Any, /) -> Any:
        value = self.source.validate(vector)
        return self.target.validate(self.forward.mv(self._owned_part(value)))

    def transpose_mv(self, vector: Any, /) -> Any:
        value = self.target.validate(vector)
        return self.source.validate(self.transposed.mv(self._owned_part(value)))

    def adjoint_mv(self, vector: Any, /) -> Any:
        del vector
        raise ValueError("Owned-sharded FEM operator does not declare an adjoint.")

    def _materialize(self, /) -> Array:
        raise ValueError("Owned-sharded FEM operators cannot be materialized.")


def execute_cardiovascular_distributed_collective(
    contract: CardiovascularDistributedContract,
    dof_map: FiniteElementDofMap,
    finite_element_operator: AbstractLinearOperator,
    right_hand_side: ArrayLike | None,
    /,
    *,
    initial_guess: ArrayLike | None = None,
    solver_policy: LinearSolvePolicy | None = None,
    restart_state: CardiovascularDistributedSolverState | None = None,
) -> CardiovascularDistributedCollectiveEvidence:
    """Apply and solve a generic FEM operator from partition-owned JAX shards."""

    if not isinstance(contract, CardiovascularDistributedContract):
        raise TypeError("contract must be CardiovascularDistributedContract.")
    if not isinstance(dof_map, FiniteElementDofMap):
        raise TypeError("dof_map must be FiniteElementDofMap.")
    if not isinstance(finite_element_operator, AbstractLinearOperator):
        raise TypeError("finite_element_operator must be AbstractLinearOperator.")
    capability = contract.capability
    if (
        not isinstance(contract.route, CardiovascularDistributedCollectiveExecution)
        or contract.route.process_count != 1
        or not capability.transport_eligible
        or capability.reference_eligible
        or capability.device_mesh_id is None
        or capability.transport_id is None
    ):
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.DISTRIBUTED_INELIGIBLE,
            phase="distributed-collective",
            entity_ids=(contract.contract_id, capability.capability_id),
        )
    _validate_finite_element_operator(dof_map, finite_element_operator)
    mesh = _cardiovascular_device_mesh(contract)
    owners, owned_ids, owned_valid, halo_ids, halo_valid = _distributed_dof_layout(
        contract, dof_map
    )
    owned_ids_array = jnp.asarray(owned_ids)
    owned_valid_array = jnp.asarray(owned_valid)
    axis_name = contract.route.axis_name
    distributed_operator = _OwnedShardedFiniteElementOperator(
        finite_element_operator, owners, axis_name
    )
    policy = (
        LinearSolvePolicy(
            GMRES(restart=min(8, finite_element_operator.source.size)),
            tolerance=TolerancePolicy(
                relative=1.0e-8,
                absolute=1.0e-10,
                max_steps=max(8, finite_element_operator.source.size),
            ),
            failure=FailurePolicy("status"),
        )
        if solver_policy is None
        else solver_policy
    )
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("solver_policy must be LinearSolvePolicy or None.")
    problem = LinearSystem(
        distributed_operator,
        problem_id=canonical_fingerprint(
            {
                "kind": "cardiovascular-distributed-fem-linear-system",
                "contract": contract.contract_id,
                "operator": distributed_operator.operator_id,
                "solver_policy": contract.solver_policy_id,
            }
        ),
    )
    prepared = prepare_linear_solve(problem, policy)
    if restart_state is None:
        if right_hand_side is None:
            raise ValueError("right_hand_side is required without restart_state.")
        rhs = finite_element_operator.target.validate(jnp.asarray(right_hand_side))
        guess = (
            finite_element_operator.source.zeros()
            if initial_guess is None
            else finite_element_operator.source.validate(jnp.asarray(initial_guess))
        )
        owned_guess = _shard_owned_values(mesh, axis_name, owned_ids, owned_valid, guess)
        owned_rhs = _shard_owned_values(mesh, axis_name, owned_ids, owned_valid, rhs)
        solve_count = 1
        checkpoint_id = None
    else:
        if right_hand_side is not None or initial_guess is not None:
            raise ValueError(
                "right_hand_side and initial_guess must be omitted for restart."
            )
        _validate_distributed_restart(
            restart_state,
            contract,
            dof_map,
            finite_element_operator,
            distributed_operator,
            prepared,
            owned_ids,
            owned_valid,
        )
        sharding = NamedSharding(
            mesh,
            PartitionSpec(
                axis_name,
                *(None for _ in range(restart_state.owned_solution.ndim - 1)),
            ),
        )
        owned_guess = jax.device_put(restart_state.owned_solution, sharding)
        owned_rhs = jax.device_put(restart_state.owned_right_hand_side, sharding)
        solve_count = restart_state.solve_count + 1
        checkpoint_id = restart_state.checkpoint_id

    replicated = PartitionSpec()
    owned_spec = PartitionSpec(axis_name, *(None for _ in range(owned_guess.ndim - 1)))
    collective = JaxCollectiveBackend(axis_name)

    def execute_local(local_guess, local_rhs):
        part = jax.lax.axis_index(axis_name)
        ids = owned_ids_array[part]
        valid = owned_valid_array[part]
        reconstructed_guess = collective.sum(
            _scatter_owned(local_guess[0], ids, valid, finite_element_operator.source)
        )
        reconstructed_rhs = collective.sum(
            _scatter_owned(local_rhs[0], ids, valid, finite_element_operator.target)
        )
        operator_action = distributed_operator.mv(reconstructed_guess)
        transpose_action = distributed_operator.transpose_mv(reconstructed_rhs)
        solved = solve_linear_system(
            prepared,
            reconstructed_rhs,
            initial_guess=reconstructed_guess,
        )
        solution = jnp.asarray(solved.value)
        gathered = solution[jnp.where(valid, ids, 0)]
        mask = valid.reshape(valid.shape + (1,) * (gathered.ndim - 1))
        owned_solution = jnp.where(mask, gathered, 0.0)[None, ...]
        return (
            operator_action,
            transpose_action,
            reconstructed_guess,
            reconstructed_rhs,
            owned_solution,
            solved.diagnostics.residual_norm,
            solved.successful,
            solved.diagnostics.iterations,
        )

    (
        distributed_action,
        distributed_transpose,
        reconstructed_guess,
        reconstructed_rhs,
        owned_solution,
        solver_residual,
        solver_successful,
        iteration_count,
    ) = jax.shard_map(
        execute_local,
        mesh=mesh,
        in_specs=(owned_spec, owned_spec),
        out_specs=(
            replicated,
            replicated,
            replicated,
            replicated,
            owned_spec,
            replicated,
            replicated,
            replicated,
        ),
    )(owned_guess, owned_rhs)
    serial_action = finite_element_operator.mv(reconstructed_guess)
    serial_transpose = finite_element_operator.transpose_mv(reconstructed_rhs)
    serial_solve = solve_linear_system(
        LinearSystem(finite_element_operator),
        reconstructed_rhs,
        policy=policy,
        initial_guess=reconstructed_guess,
    )
    reconstructed_solution = _unpack_owned_values(
        owned_solution,
        owned_ids,
        owned_valid,
        finite_element_operator.source,
    )
    original_guess = _unpack_owned_values(
        owned_guess,
        owned_ids,
        owned_valid,
        finite_element_operator.source,
    )
    solver_state = CardiovascularDistributedSolverState(
        owned_solution,
        owned_rhs,
        owned_ids,
        owned_valid,
        solve_count=solve_count,
        iteration_count=int(np.asarray(iteration_count)),
        successful=bool(np.asarray(solver_successful)),
        execution_manifest_id=contract.execution_manifest_id,
        contract_id=contract.contract_id,
        phase_plan_id=contract.phase_plan.plan_id,
        partition_id=contract.phase_plan.partition.partition_id,
        workset_plan_id=contract.phase_plan.worksets.plan_id,
        dof_map_id=dof_map.dof_map_id,
        finite_element_operator_id=finite_element_operator.operator_id,
        distributed_operator_id=distributed_operator.operator_id,
        solver_policy_id=contract.solver_policy_id,
        solver_plan_id=prepared.plan.plan_id,
        device_mesh_id=capability.device_mesh_id,
        transport_id=capability.transport_id,
        checkpoint_id=checkpoint_id,
    )
    return CardiovascularDistributedCollectiveEvidence(
        solver_state,
        serial_action,
        distributed_action,
        serial_transpose,
        distributed_transpose,
        _array_norm(distributed_action - serial_action),
        _array_norm(reconstructed_guess - original_guess),
        _array_norm(distributed_transpose - serial_transpose),
        solver_residual,
        _array_norm(
            reconstructed_solution
            - finite_element_operator.source.validate(serial_solve.value)
        ),
        owned_value_count=int(np.sum(owned_valid)),
        halo_value_count=int(np.sum(halo_valid)),
    )


def write_cardiovascular_distributed_solver_checkpoint(
    codec: CardiovascularLifecycleCheckpointCodec,
    path: str | Path,
    state: CardiovascularDistributedSolverState,
    /,
    *,
    checkpoint_id: str,
    parent_checkpoint_id: str | None = None,
) -> CardiovascularCheckpointRecord:
    """Atomically checkpoint accepted owned shards with every runtime binding."""

    if not isinstance(codec, CardiovascularLifecycleCheckpointCodec):
        raise TypeError("codec must be CardiovascularLifecycleCheckpointCodec.")
    if not isinstance(state, CardiovascularDistributedSolverState):
        raise TypeError("state must be CardiovascularDistributedSolverState.")
    if codec.execution.manifest_id != state.execution_manifest_id:
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.CHECKPOINT_MISMATCH,
            phase="distributed-checkpoint-write",
            entity_ids=(codec.execution.manifest_id, state.state_id),
        )
    arrays = {
        "solver/owned_solution": state.owned_solution,
        "solver/owned_right_hand_side": state.owned_right_hand_side,
        "solver/owned_dof_ids": state.owned_dof_ids,
        "solver/owned_valid": state.owned_valid,
        "solver/solve_count": np.asarray(state.solve_count, dtype=np.int64),
        "solver/iteration_count": np.asarray(state.iteration_count, dtype=np.int64),
        "solver/successful": np.asarray(state.successful, dtype=bool),
    }
    layouts = {name: state.binding_ids for name in arrays}
    return codec.write(
        path,
        arrays,
        checkpoint_id=checkpoint_id,
        committed=state.successful,
        layout_ids=layouts,
        parent_checkpoint_id=parent_checkpoint_id,
    )


def read_cardiovascular_distributed_solver_checkpoint(
    codec: CardiovascularLifecycleCheckpointCodec,
    path: str | Path,
    contract: CardiovascularDistributedContract,
    expected_state: CardiovascularDistributedSolverState,
    /,
) -> CardiovascularDistributedSolverState:
    """Restore owned shards only when checkpoint and runtime bindings are exact."""

    if not isinstance(codec, CardiovascularLifecycleCheckpointCodec):
        raise TypeError("codec must be CardiovascularLifecycleCheckpointCodec.")
    if not isinstance(contract, CardiovascularDistributedContract):
        raise TypeError("contract must be CardiovascularDistributedContract.")
    if not isinstance(expected_state, CardiovascularDistributedSolverState):
        raise TypeError("expected_state must be CardiovascularDistributedSolverState.")
    if (
        codec.execution.manifest_id != expected_state.execution_manifest_id
        or contract.contract_id != expected_state.contract_id
    ):
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.CHECKPOINT_MISMATCH,
            phase="distributed-checkpoint-read",
            entity_ids=(codec.execution.manifest_id, contract.contract_id),
        )
    record = codec.read(path)
    names = (
        "solver/owned_solution",
        "solver/owned_right_hand_side",
        "solver/owned_dof_ids",
        "solver/owned_valid",
        "solver/solve_count",
        "solver/iteration_count",
        "solver/successful",
    )
    manifest = record.archive.manifest
    shards = {shard.shard_id: shard for shard in manifest.shards}
    if (
        set(record.arrays) != set(names)
        or set(shards) != set(names)
        or any(shards[name].layout_ids != expected_state.binding_ids for name in names)
    ):
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.CHECKPOINT_MISMATCH,
            phase="distributed-checkpoint-read",
            entity_ids=(contract.contract_id, record.checkpoint_id),
        )
    identifiers = np.asarray(record.arrays["solver/owned_dof_ids"])
    valid = np.asarray(record.arrays["solver/owned_valid"])
    if not np.array_equal(identifiers, np.asarray(expected_state.owned_dof_ids)) or not (
        np.array_equal(valid, np.asarray(expected_state.owned_valid))
    ):
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.CHECKPOINT_MISMATCH,
            phase="distributed-checkpoint-read",
            entity_ids=(contract.contract_id, record.checkpoint_id),
        )
    mesh = _cardiovascular_device_mesh(contract)
    axis_name = contract.route.axis_name
    solution = np.asarray(record.arrays["solver/owned_solution"])
    right_hand_side = np.asarray(record.arrays["solver/owned_right_hand_side"])
    sharding = NamedSharding(
        mesh,
        PartitionSpec(axis_name, *(None for _ in range(solution.ndim - 1))),
    )
    return CardiovascularDistributedSolverState(
        jax.device_put(solution, sharding),
        jax.device_put(right_hand_side, sharding),
        identifiers,
        valid,
        solve_count=int(record.arrays["solver/solve_count"]),
        iteration_count=int(record.arrays["solver/iteration_count"]),
        successful=bool(record.arrays["solver/successful"]),
        execution_manifest_id=expected_state.execution_manifest_id,
        contract_id=expected_state.contract_id,
        phase_plan_id=expected_state.phase_plan_id,
        partition_id=expected_state.partition_id,
        workset_plan_id=expected_state.workset_plan_id,
        dof_map_id=expected_state.dof_map_id,
        finite_element_operator_id=expected_state.finite_element_operator_id,
        distributed_operator_id=expected_state.distributed_operator_id,
        solver_policy_id=expected_state.solver_policy_id,
        solver_plan_id=expected_state.solver_plan_id,
        device_mesh_id=expected_state.device_mesh_id,
        transport_id=expected_state.transport_id,
        checkpoint_id=record.checkpoint_id,
    )


def _cardiovascular_device_mesh(contract: CardiovascularDistributedContract, /) -> Mesh:
    capability = contract.capability
    devices = {
        (int(device.process_index), int(device.id)): device
        for device in jax.devices()
        if device.platform == capability.backend
    }
    keys = tuple(
        zip(
            capability.device_process_indices,
            capability.device_ids,
            strict=True,
        )
    )
    if len(keys) != contract.route.partition_count or any(
        key not in devices for key in keys
    ):
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.DISTRIBUTED_INELIGIBLE,
            phase="distributed-device-mesh",
            entity_ids=(contract.contract_id, capability.capability_id),
        )
    return Mesh(
        np.asarray(tuple(devices[key] for key in keys), dtype=object),
        (contract.route.axis_name,),
    )


def _validate_finite_element_operator(
    dof_map: FiniteElementDofMap,
    operator: AbstractLinearOperator,
    /,
) -> None:
    expected = (dof_map.global_dof_count,) + dof_map.component_shape
    if (
        not isinstance(operator.source, ArraySpace)
        or not isinstance(operator.target, ArraySpace)
        or operator.source.shape != expected
        or operator.target.shape != expected
        or operator.batch_shape
        or not operator.capabilities.transpose
    ):
        raise ValueError(
            "Distributed FEM execution requires one unbatched square ArraySpace "
            "operator matching the supplied finite-element DOF map with transpose."
        )


def _distributed_dof_layout(
    contract: CardiovascularDistributedContract,
    dof_map: FiniteElementDofMap,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cell_dofs = np.concatenate(
        tuple(np.asarray(value, dtype=np.int32) for value in dof_map.cell_dofs),
        axis=0,
    )
    cell_owner = np.asarray(contract.phase_plan.partition.cell_owner, dtype=np.int32)
    if cell_dofs.shape[0] != cell_owner.size:
        raise ValueError("FEM DOF map and distributed cell partition do not match.")
    part_count = contract.route.partition_count
    dof_owner = np.full((dof_map.global_dof_count,), part_count, dtype=np.int32)
    np.minimum.at(
        dof_owner,
        cell_dofs.reshape((-1,)),
        np.repeat(cell_owner, cell_dofs.shape[1]),
    )
    if np.any(dof_owner == part_count):
        raise ValueError("Every FEM degree of freedom must have an owning partition.")
    owned_routes = tuple(
        np.flatnonzero(dof_owner == part).astype(np.int32) for part in range(part_count)
    )
    halo_routes = []
    for part in range(part_count):
        local_dofs = np.unique(cell_dofs[cell_owner == part].reshape((-1,)))
        halo_routes.append(local_dofs[dof_owner[local_dofs] != part].astype(np.int32))
    owned_width = max(map(len, owned_routes))
    halo_width = max(map(len, halo_routes), default=0)
    owned_ids = np.zeros((part_count, owned_width), dtype=np.int32)
    owned_valid = np.zeros_like(owned_ids, dtype=bool)
    halo_ids = np.zeros((part_count, halo_width), dtype=np.int32)
    halo_valid = np.zeros_like(halo_ids, dtype=bool)
    for part, route in enumerate(owned_routes):
        owned_ids[part, : route.size] = route
        owned_valid[part, : route.size] = True
    for part, route in enumerate(halo_routes):
        halo_ids[part, : route.size] = route
        halo_valid[part, : route.size] = True
    return dof_owner, owned_ids, owned_valid, halo_ids, halo_valid


def _shard_owned_values(
    mesh: Mesh,
    axis_name: str,
    owned_ids: np.ndarray,
    owned_valid: np.ndarray,
    values: ArrayLike,
    /,
) -> Array:
    value = np.asarray(values)
    packed = np.zeros(owned_ids.shape + value.shape[1:], dtype=value.dtype)
    for part in range(owned_ids.shape[0]):
        valid = owned_valid[part]
        packed[part, valid] = value[owned_ids[part, valid]]
    return jax.device_put(
        packed,
        NamedSharding(
            mesh,
            PartitionSpec(axis_name, *(None for _ in range(packed.ndim - 1))),
        ),
    )


def _scatter_owned(
    local_values: Array,
    owned_ids: ArrayLike,
    owned_valid: ArrayLike,
    space: ArraySpace,
    /,
) -> Array:
    identifiers = jnp.asarray(owned_ids)
    valid = jnp.asarray(owned_valid)
    safe = jnp.where(valid, identifiers, 0)
    mask = valid.reshape(valid.shape + (1,) * (local_values.ndim - 1))
    return (
        jnp.zeros(space.shape, dtype=local_values.dtype)
        .at[safe]
        .add(jnp.where(mask, local_values, 0.0))
    )


def _unpack_owned_values(
    owned_values: ArrayLike,
    owned_ids: np.ndarray,
    owned_valid: np.ndarray,
    space: ArraySpace,
    /,
) -> Array:
    values = jnp.asarray(owned_values)
    result = jnp.zeros(space.shape, dtype=values.dtype)
    for part in range(owned_ids.shape[0]):
        valid = owned_valid[part]
        result = result.at[owned_ids[part, valid]].set(values[part, valid])
    return result


def _validate_distributed_restart(
    state: CardiovascularDistributedSolverState,
    contract: CardiovascularDistributedContract,
    dof_map: FiniteElementDofMap,
    finite_element_operator: AbstractLinearOperator,
    distributed_operator: _OwnedShardedFiniteElementOperator,
    prepared: PreparedLinearSolve,
    owned_ids: np.ndarray,
    owned_valid: np.ndarray,
    /,
) -> None:
    expected = (
        contract.execution_manifest_id,
        contract.contract_id,
        contract.phase_plan.plan_id,
        contract.phase_plan.partition.partition_id,
        contract.phase_plan.worksets.plan_id,
        dof_map.dof_map_id,
        finite_element_operator.operator_id,
        distributed_operator.operator_id,
        contract.solver_policy_id,
        prepared.plan.plan_id,
        contract.capability.device_mesh_id,
        contract.capability.transport_id,
    )
    observed = (
        state.execution_manifest_id,
        state.contract_id,
        state.phase_plan_id,
        state.partition_id,
        state.workset_plan_id,
        state.dof_map_id,
        state.finite_element_operator_id,
        state.distributed_operator_id,
        state.solver_policy_id,
        state.solver_plan_id,
        state.device_mesh_id,
        state.transport_id,
    )
    if (
        observed != expected
        or not state.successful
        or not np.array_equal(np.asarray(state.owned_dof_ids), owned_ids)
        or not np.array_equal(np.asarray(state.owned_valid), owned_valid)
    ):
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.CHECKPOINT_MISMATCH,
            phase="distributed-restart",
            entity_ids=(contract.contract_id, state.state_id),
        )


def _array_norm(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return jnp.sqrt(jnp.sum(jnp.abs(array) ** 2))


def execute_cardiovascular_distributed_replay(
    contract: CardiovascularDistributedContract,
    body: Callable[[Any, Any], tuple[Any, Any]],
    initial: Any,
    xs: Any,
    /,
) -> tuple[Any, Any]:
    """Run the generic scheduled replay under eligible local reference semantics."""

    if not isinstance(contract, CardiovascularDistributedContract):
        raise TypeError("contract must be CardiovascularDistributedContract.")
    if (
        not isinstance(contract.route, CardiovascularDistributedReferenceExecution)
        or not contract.capability.reference_eligible
        or contract.capability.transport_eligible
    ):
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.DISTRIBUTED_INELIGIBLE,
            phase="distributed-replay",
            entity_ids=(contract.contract_id,),
        )
    return checkpointed_scan(
        body,
        initial,
        xs,
        length=contract.replay_schedule.step_count,
        mode="scheduled",
        schedule=contract.replay_schedule,
    )


def require_cardiovascular_distributed_transport(
    contract: CardiovascularDistributedContract, /
) -> None:
    """Require an observed exact device mesh; never substitute reference execution."""

    if not isinstance(contract, CardiovascularDistributedContract):
        raise TypeError("contract must be CardiovascularDistributedContract.")
    if not contract.capability.transport_eligible:
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.DISTRIBUTED_INELIGIBLE,
            phase="distributed-transport",
            entity_ids=(contract.contract_id, contract.capability.capability_id),
        )


class CardiovascularSaltationPolicy(StrictModule, NonTrainableState):
    """Guard-unit-aware transversality threshold for saltation evidence."""

    guard_unit: str = eqx.field(static=True)
    minimum_absolute_slope_per_ms: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, guard_unit: str, minimum_absolute_slope_per_ms: float, /):
        unit = _identifier(guard_unit, "guard_unit")
        slope = float(minimum_absolute_slope_per_ms)
        if not math.isfinite(slope) or slope <= 0.0:
            raise ValueError("minimum_absolute_slope_per_ms must be finite and positive.")
        self.guard_unit = unit
        self.minimum_absolute_slope_per_ms = slope
        self.policy_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-saltation-policy",
                "guard_unit": unit,
                "minimum_absolute_slope_per_ms": slope,
            }
        )


class CardiovascularEventSpec(StrictModule, NonTrainableState):
    """Stable event identity, deterministic priority, and saltation policy."""

    source_id: str = eqx.field(static=True)
    direction: int = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    terminal: bool = eqx.field(static=True)
    saltation_policy: CardiovascularSaltationPolicy | None
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        /,
        *,
        direction: int = 0,
        priority: int = 0,
        terminal: bool = False,
        saltation_policy: CardiovascularSaltationPolicy | None = None,
    ):
        source = _identifier(source_id, "source_id")
        direction_ = int(direction)
        priority_ = int(priority)
        if isinstance(direction, (bool, np.bool_)) or direction_ not in (-1, 0, 1):
            raise ValueError("Event direction must be -1, 0, or 1.")
        if isinstance(priority, (bool, np.bool_)):
            raise TypeError("Event priority must be an integer.")
        if saltation_policy is not None and not isinstance(
            saltation_policy, CardiovascularSaltationPolicy
        ):
            raise TypeError(
                "saltation_policy must be CardiovascularSaltationPolicy or None."
            )
        self.source_id = source
        self.direction = direction_
        self.priority = priority_
        self.terminal = bool(terminal)
        self.saltation_policy = saltation_policy
        self.event_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-event-spec",
                "source": source,
                "direction": direction_,
                "priority": priority_,
                "terminal": bool(terminal),
                "saltation_policy": (
                    None if saltation_policy is None else saltation_policy.policy_id
                ),
            }
        )


class CardiovascularMultiratePlan(StrictModule, NonTrainableState):
    """Fixed-topology subsystem rates and event localization policy."""

    subsystem_ids: tuple[str, ...] = eqx.field(static=True)
    substeps_per_macro: tuple[int, ...] = eqx.field(static=True)
    macro_step_ms: float = eqx.field(static=True)
    events: tuple[CardiovascularEventSpec, ...]
    localization_iterations: int = eqx.field(static=True)
    localization_tolerance_ms: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        subsystem_ids: Sequence[str],
        substeps_per_macro: Sequence[int],
        macro_step_ms: float,
        /,
        *,
        events: Sequence[CardiovascularEventSpec] = (),
        localization_iterations: int = 40,
        localization_tolerance_ms: float = 1.0e-9,
    ):
        subsystems = tuple(_identifier(v, "subsystem_id") for v in subsystem_ids)
        rates = tuple(
            _positive_integer(v, "substeps_per_macro") for v in substeps_per_macro
        )
        step = float(macro_step_ms)
        events_ = tuple(events)
        iterations = _positive_integer(localization_iterations, "localization_iterations")
        tolerance = float(localization_tolerance_ms)
        if not subsystems or len(subsystems) != len(rates):
            raise ValueError("One positive substep count is required per subsystem.")
        if len(set(subsystems)) != len(subsystems):
            raise ValueError("Subsystem IDs must be unique.")
        if not all(isinstance(event, CardiovascularEventSpec) for event in events_):
            raise TypeError("events must contain CardiovascularEventSpec values.")
        source_ids = tuple(event.source_id for event in events_)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Event source IDs must be unique.")
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("macro_step_ms must be finite and positive.")
        if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= step:
            raise ValueError(
                "localization_tolerance_ms must be positive and below macro_step_ms."
            )
        self.subsystem_ids = subsystems
        self.substeps_per_macro = rates
        self.macro_step_ms = step
        self.events = events_
        self.localization_iterations = iterations
        self.localization_tolerance_ms = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-multirate-plan",
                "subsystems": subsystems,
                "substeps_per_macro": rates,
                "macro_step_ms": step,
                "events": [event.event_id for event in events_],
                "localization_iterations": iterations,
                "localization_tolerance_ms": tolerance,
            }
        )


class PreparedCardiovascularScheduler(StrictModule, NonTrainableState):
    """Precomputed fixed-capacity owner/time schedule for a multirate plan."""

    execution_manifest_id: str = eqx.field(static=True)
    plan: CardiovascularMultiratePlan
    owner_indices: Array
    start_times_ms: Array
    end_times_ms: Array
    steps_per_macro: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    state_value_capacity: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        execution_manifest_id: str,
        plan: CardiovascularMultiratePlan,
        owner_indices: ArrayLike,
        start_times_ms: ArrayLike,
        end_times_ms: ArrayLike,
        steps_per_macro: int,
        event_capacity: int,
        state_value_capacity: int,
        /,
    ):
        execution = _identifier(execution_manifest_id, "execution_manifest_id")
        if not isinstance(plan, CardiovascularMultiratePlan):
            raise TypeError("plan must be CardiovascularMultiratePlan.")
        owners = jnp.asarray(owner_indices, dtype=jnp.int32)
        starts = jnp.asarray(start_times_ms, dtype=float)
        ends = jnp.asarray(end_times_ms, dtype=float)
        per_macro = _positive_integer(steps_per_macro, "steps_per_macro")
        events = _nonnegative_integer(event_capacity, "event_capacity")
        state_values = _positive_integer(state_value_capacity, "state_value_capacity")
        if owners.ndim != 1 or starts.shape != owners.shape or ends.shape != owners.shape:
            raise ValueError("Prepared schedule arrays must be matching rank-one arrays.")
        if bool(
            np.any(np.asarray(owners) < 0)
            or np.any(np.asarray(owners) >= len(plan.subsystem_ids))
            or np.any(~np.isfinite(np.asarray(starts)))
            or np.any(~np.isfinite(np.asarray(ends)))
            or np.any(np.asarray(ends) <= np.asarray(starts))
        ):
            raise ValueError("Prepared schedule entries are invalid.")
        self.execution_manifest_id = execution
        self.plan = plan
        self.owner_indices = owners
        self.start_times_ms = starts
        self.end_times_ms = ends
        self.steps_per_macro = per_macro
        self.event_capacity = events
        self.state_value_capacity = state_values
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-scheduler",
                "execution": execution,
                "plan": plan.plan_id,
                "steps_per_macro": per_macro,
                "event_capacity": events,
                "state_value_capacity": state_values,
                "schedule": array_tree_fingerprint((owners, starts, ends)),
            }
        )


class CardiovascularStepCandidate(StrictModule, Generic[State]):
    """One uncommitted subsystem advance or event reset."""

    state: State
    accepted: bool = eqx.field(static=True)
    status: CardiovascularRuntimeStatus = eqx.field(static=True)

    def __init__(
        self,
        state: State,
        /,
        *,
        accepted: bool = True,
        status: CardiovascularRuntimeStatus = CardiovascularRuntimeStatus.SUCCESS,
    ):
        accepted_ = bool(accepted)
        status_ = CardiovascularRuntimeStatus(status)
        if accepted_ != (status_ is CardiovascularRuntimeStatus.SUCCESS):
            raise ValueError("Step candidate acceptance and status disagree.")
        self.state = state
        self.accepted = accepted_
        self.status = status_


class CardiovascularScheduleEvidence(StrictModule, NonTrainableState):
    """Fixed-shape route, event, rollback, and saltation eligibility evidence."""

    scheduled_owner_indices: Array
    scheduled_start_times_ms: Array
    scheduled_end_times_ms: Array
    scheduled_active: Array
    event_source_indices: Array
    event_times_ms: Array
    event_active: Array
    event_guard_before: Array
    event_guard_after: Array
    event_guard_slope_per_ms: Array
    saltation_eligible: Array
    scheduled_step_count: Array
    event_count: Array
    terminal_event: Array
    execution_manifest_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    status: CardiovascularRuntimeStatus = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return self.status is CardiovascularRuntimeStatus.SUCCESS


class CardiovascularScheduleCandidate(StrictModule, Generic[State]):
    """Whole-run transaction retaining both rollback and proposed states."""

    initial_state: State
    proposed_state: State
    evidence: CardiovascularScheduleEvidence
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_state: State,
        proposed_state: State,
        evidence: CardiovascularScheduleEvidence,
        /,
    ):
        if not isinstance(evidence, CardiovascularScheduleEvidence):
            raise TypeError("evidence must be CardiovascularScheduleEvidence.")
        if jax.tree_util.tree_structure(initial_state) != jax.tree_util.tree_structure(
            proposed_state
        ):
            raise ValueError("A schedule candidate cannot change state PyTree structure.")
        self.initial_state = initial_state
        self.proposed_state = proposed_state
        self.evidence = evidence
        self.candidate_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-schedule-candidate",
                "evidence": evidence.evidence_id,
            }
        )


class CardiovascularScheduleCommit(StrictModule, Generic[State]):
    """Atomic schedule outcome: failure returns the original rollback state."""

    state: State
    committed: bool = eqx.field(static=True)
    status: CardiovascularRuntimeStatus = eqx.field(static=True)
    evidence: CardiovascularScheduleEvidence
    commit_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: State,
        committed: bool,
        status: CardiovascularRuntimeStatus,
        evidence: CardiovascularScheduleEvidence,
        candidate_id: str,
        /,
    ):
        committed_ = bool(committed)
        status_ = CardiovascularRuntimeStatus(status)
        if committed_ != (status_ is CardiovascularRuntimeStatus.SUCCESS):
            raise ValueError("Schedule commit flag and status disagree.")
        self.state = state
        self.committed = committed_
        self.status = status_
        self.evidence = evidence
        self.commit_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-schedule-commit",
                "candidate": _identifier(candidate_id, "candidate_id"),
                "committed": committed_,
                "status": status_.value,
            }
        )


class CardiovascularReplayEvidence(StrictModule, NonTrainableState):
    """Exact route/state replay comparison for one schedule candidate."""

    equivalent: bool = eqx.field(static=True)
    reference_evidence_id: str = eqx.field(static=True)
    replay_evidence_id: str = eqx.field(static=True)
    status: CardiovascularRuntimeStatus = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        equivalent: bool,
        reference_evidence_id: str,
        replay_evidence_id: str,
        /,
    ):
        equivalent_ = bool(equivalent)
        reference = _identifier(reference_evidence_id, "reference_evidence_id")
        replay = _identifier(replay_evidence_id, "replay_evidence_id")
        status = (
            CardiovascularRuntimeStatus.SUCCESS
            if equivalent_
            else CardiovascularRuntimeStatus.REPLAY_MISMATCH
        )
        self.equivalent = equivalent_
        self.reference_evidence_id = reference
        self.replay_evidence_id = replay
        self.status = status
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-replay-evidence",
                "equivalent": equivalent_,
                "reference": reference,
                "replay": replay,
            }
        )


ScheduleAdvance: TypeAlias = Callable[
    [State, str, float, float], CardiovascularStepCandidate[State]
]
EventEvaluator: TypeAlias = Callable[[State, float], ArrayLike]
EventReset: TypeAlias = Callable[[State, str, float], CardiovascularStepCandidate[State]]


def prepare_cardiovascular_scheduler(
    execution: CardiovascularExecutionManifest,
    plan: CardiovascularMultiratePlan,
    /,
) -> PreparedCardiovascularScheduler:
    """Lower multirate owners to a deterministic fixed-capacity host schedule."""

    if not isinstance(execution, CardiovascularExecutionManifest):
        raise TypeError("execution must be CardiovascularExecutionManifest.")
    if not isinstance(plan, CardiovascularMultiratePlan):
        raise TypeError("plan must be CardiovascularMultiratePlan.")
    per_macro = sum(plan.substeps_per_macro)
    total = execution.capacity.maximum_macro_steps * per_macro
    admission = admit_cardiovascular_capacity(
        execution.capacity,
        CardiovascularCapacityRequest(
            macro_steps=execution.capacity.maximum_macro_steps,
            scheduled_steps=total,
            events=execution.capacity.maximum_events if plan.events else 0,
        ),
    )
    if not admission.eligible:
        raise CardiovascularRuntimeError(
            CardiovascularRuntimeStatus.CAPACITY_REFUSED,
            phase="scheduler-prepare",
            entity_ids=(execution.manifest_id, admission.admission_id),
        )
    records: list[tuple[float, str, int, float]] = []
    for macro in range(execution.capacity.maximum_macro_steps):
        origin = macro * plan.macro_step_ms
        for owner, (subsystem, rate) in enumerate(
            zip(plan.subsystem_ids, plan.substeps_per_macro, strict=True)
        ):
            local_step = plan.macro_step_ms / rate
            for local_index in range(rate):
                start = origin + local_index * local_step
                end = origin + (local_index + 1) * local_step
                records.append((end, subsystem, owner, start))
    records.sort(key=lambda item: (item[0], item[1]))
    owners = np.asarray([item[2] for item in records], dtype=np.int32)
    starts = np.asarray([item[3] for item in records], dtype=float)
    ends = np.asarray([item[0] for item in records], dtype=float)
    return PreparedCardiovascularScheduler(
        execution.manifest_id,
        plan,
        owners,
        starts,
        ends,
        per_macro,
        execution.capacity.maximum_events if plan.events else 0,
        execution.capacity.maximum_state_values,
    )


def run_cardiovascular_schedule(
    prepared: PreparedCardiovascularScheduler,
    initial_state: State,
    macro_steps: int,
    advance: ScheduleAdvance[State],
    event_values: EventEvaluator[State],
    reset_event: EventReset[State],
    /,
) -> CardiovascularScheduleCandidate[State]:
    """Build one event-split candidate; no state crosses commit in this function."""

    if not isinstance(prepared, PreparedCardiovascularScheduler):
        raise TypeError("prepared must be PreparedCardiovascularScheduler.")
    count = _positive_integer(macro_steps, "macro_steps")
    capacity_macro = prepared.owner_indices.shape[0] // prepared.steps_per_macro
    if count > capacity_macro:
        return _failed_schedule_candidate(
            prepared,
            initial_state,
            initial_state,
            CardiovascularRuntimeStatus.CAPACITY_REFUSED,
            0,
            (),
        )
    if _tree_value_count(initial_state) > prepared.state_value_capacity:
        return _failed_schedule_candidate(
            prepared,
            initial_state,
            initial_state,
            CardiovascularRuntimeStatus.CAPACITY_REFUSED,
            0,
            (),
        )
    if not callable(advance) or not callable(event_values) or not callable(reset_event):
        raise TypeError("Scheduler callbacks must be callable.")
    plan = prepared.plan
    maximum_steps = int(prepared.owner_indices.shape[0])
    maximum_events = _prepared_event_capacity(prepared)
    active_steps = count * prepared.steps_per_macro
    scheduled_active = np.zeros((maximum_steps,), dtype=bool)
    event_source = np.full((maximum_events,), -1, dtype=np.int32)
    event_times = np.zeros((maximum_events,), dtype=float)
    event_active = np.zeros((maximum_events,), dtype=bool)
    guard_before_record = np.zeros((maximum_events,), dtype=float)
    guard_after_record = np.zeros((maximum_events,), dtype=float)
    guard_slope_record = np.zeros((maximum_events,), dtype=float)
    saltation = np.zeros((maximum_events,), dtype=bool)
    state = initial_state
    initial_signature = _state_leaf_signature(initial_state)
    event_count = 0
    terminal = False
    status = CardiovascularRuntimeStatus.SUCCESS
    completed_steps = 0
    for scheduled_index in range(active_steps):
        if terminal or status is not CardiovascularRuntimeStatus.SUCCESS:
            break
        scheduled_active[scheduled_index] = True
        owner = int(np.asarray(prepared.owner_indices[scheduled_index]))
        subsystem = plan.subsystem_ids[owner]
        step_start = float(np.asarray(prepared.start_times_ms[scheduled_index]))
        step_end = float(np.asarray(prepared.end_times_ms[scheduled_index]))
        segment_state = state
        segment_start = step_start
        while segment_start < step_end and not terminal:
            proposed = advance(segment_state, subsystem, segment_start, step_end)
            if not isinstance(proposed, CardiovascularStepCandidate):
                raise TypeError("advance must return CardiovascularStepCandidate.")
            _require_state_signature(proposed.state, initial_signature, "advance")
            if not proposed.accepted:
                status = CardiovascularRuntimeStatus.STEP_REJECTED
                break
            if not plan.events:
                segment_state = proposed.state
                segment_start = step_end
                break
            start_guards, start_finite = _evaluate_event_values(
                event_values, segment_state, segment_start, len(plan.events)
            )
            end_guards, end_finite = _evaluate_event_values(
                event_values, proposed.state, step_end, len(plan.events)
            )
            if not start_finite or not end_finite:
                status = CardiovascularRuntimeStatus.EVENT_LOCALIZATION_FAILED
                break
            crossing = tuple(
                index
                for index, event in enumerate(plan.events)
                if _event_crossed(event, start_guards[index], end_guards[index])
            )
            if not crossing:
                segment_state = proposed.state
                segment_start = step_end
                break
            if event_count >= maximum_events:
                status = CardiovascularRuntimeStatus.EVENT_CAPACITY_EXCEEDED
                break
            localized: list[tuple[float, int, State, float]] = []
            localization_failed = False
            for event_index in crossing:
                result = _localize_event(
                    plan,
                    event_index,
                    segment_state,
                    proposed.state,
                    subsystem,
                    segment_start,
                    step_end,
                    start_guards[event_index],
                    end_guards[event_index],
                    advance,
                    event_values,
                    initial_signature,
                )
                if result is None:
                    localization_failed = True
                    break
                localized.append(result)
            if localization_failed:
                status = CardiovascularRuntimeStatus.EVENT_LOCALIZATION_FAILED
                break
            earliest = min(item[0] for item in localized)
            simultaneous = tuple(
                sorted(
                    (
                        item
                        for item in localized
                        if abs(item[0] - earliest) <= plan.localization_tolerance_ms
                    ),
                    key=lambda item: (
                        plan.events[item[1]].priority,
                        plan.events[item[1]].source_id,
                    ),
                )
            )
            localized_time = earliest
            reset_state = min(simultaneous, key=lambda item: item[0])[2]
            for _, event_index, _, root_guard in simultaneous:
                if event_count >= maximum_events:
                    status = CardiovascularRuntimeStatus.EVENT_CAPACITY_EXCEEDED
                    break
                event = plan.events[event_index]
                reset = reset_event(reset_state, event.source_id, localized_time)
                if not isinstance(reset, CardiovascularStepCandidate):
                    raise TypeError(
                        "reset_event must return CardiovascularStepCandidate."
                    )
                _require_state_signature(reset.state, initial_signature, "reset_event")
                if not reset.accepted:
                    status = CardiovascularRuntimeStatus.EVENT_RESET_REJECTED
                    break
                post_guards, post_finite = _evaluate_event_values(
                    event_values, reset.state, localized_time, len(plan.events)
                )
                if not post_finite:
                    status = CardiovascularRuntimeStatus.EVENT_RESET_REJECTED
                    break
                event_source[event_count] = event_index
                event_times[event_count] = localized_time
                event_active[event_count] = True
                guard_before_record[event_count] = root_guard
                guard_after_record[event_count] = post_guards[event_index]
                guard_slope = abs(end_guards[event_index] - start_guards[event_index]) / (
                    step_end - segment_start
                )
                guard_slope_record[event_count] = guard_slope
                saltation_policy = event.saltation_policy
                saltation[event_count] = bool(
                    saltation_policy is not None
                    and guard_slope >= saltation_policy.minimum_absolute_slope_per_ms
                    and math.isfinite(post_guards[event_index])
                )
                event_count += 1
                reset_state = reset.state
                terminal = terminal or event.terminal
            if status is not CardiovascularRuntimeStatus.SUCCESS:
                break
            segment_state = reset_state
            if terminal:
                segment_start = step_end
            else:
                next_time = min(
                    step_end,
                    localized_time + plan.localization_tolerance_ms,
                )
                if next_time > localized_time:
                    nudge = advance(
                        segment_state,
                        subsystem,
                        localized_time,
                        next_time,
                    )
                    if not isinstance(nudge, CardiovascularStepCandidate):
                        raise TypeError(
                            "advance must return CardiovascularStepCandidate."
                        )
                    _require_state_signature(
                        nudge.state, initial_signature, "advance nudge"
                    )
                    if not nudge.accepted:
                        status = CardiovascularRuntimeStatus.STEP_REJECTED
                        break
                    segment_state = nudge.state
                segment_start = next_time
        if status is CardiovascularRuntimeStatus.SUCCESS:
            state = segment_state
            completed_steps += 1
    evidence = _make_schedule_evidence(
        prepared,
        scheduled_active,
        event_source,
        event_times,
        event_active,
        guard_before_record,
        guard_after_record,
        guard_slope_record,
        saltation,
        completed_steps,
        event_count,
        terminal,
        status,
    )
    return CardiovascularScheduleCandidate(initial_state, state, evidence)


def commit_cardiovascular_schedule(
    candidate: CardiovascularScheduleCandidate[State], /
) -> CardiovascularScheduleCommit[State]:
    """Atomically commit success or return the untouched rollback state."""

    if not isinstance(candidate, CardiovascularScheduleCandidate):
        raise TypeError("candidate must be CardiovascularScheduleCandidate.")
    committed = candidate.evidence.successful
    return CardiovascularScheduleCommit(
        candidate.proposed_state if committed else candidate.initial_state,
        committed,
        (CardiovascularRuntimeStatus.SUCCESS if committed else candidate.evidence.status),
        candidate.evidence,
        candidate.candidate_id,
    )


def replay_cardiovascular_schedule(
    prepared: PreparedCardiovascularScheduler,
    initial_state: State,
    macro_steps: int,
    reference: CardiovascularScheduleCommit[State],
    advance: ScheduleAdvance[State],
    event_values: EventEvaluator[State],
    reset_event: EventReset[State],
    /,
) -> tuple[CardiovascularScheduleCommit[State], CardiovascularReplayEvidence]:
    """Replay a schedule and compare route, localized events, and committed state."""

    if not isinstance(reference, CardiovascularScheduleCommit):
        raise TypeError("reference must be CardiovascularScheduleCommit.")
    replay_candidate = run_cardiovascular_schedule(
        prepared,
        initial_state,
        macro_steps,
        advance,
        event_values,
        reset_event,
    )
    replay_commit = commit_cardiovascular_schedule(replay_candidate)
    route_equal = _schedule_routes_equal(
        reference.evidence,
        replay_commit.evidence,
        prepared.plan.localization_tolerance_ms,
    )
    state_equal = _tree_exact_equal(reference.state, replay_commit.state)
    equivalent = bool(
        route_equal
        and state_equal
        and reference.committed == replay_commit.committed
        and reference.status is replay_commit.status
    )
    return replay_commit, CardiovascularReplayEvidence(
        equivalent,
        reference.evidence.evidence_id,
        replay_commit.evidence.evidence_id,
    )


def _prepared_event_capacity(prepared: PreparedCardiovascularScheduler, /) -> int:
    return prepared.event_capacity


def _localize_event(
    plan: CardiovascularMultiratePlan,
    event_index: int,
    start_state: State,
    end_state: State,
    subsystem: str,
    start_time: float,
    end_time: float,
    start_guard: float,
    end_guard: float,
    advance: ScheduleAdvance[State],
    event_values: EventEvaluator[State],
    state_signature: tuple[Any, tuple[tuple[tuple[int, ...], str], ...]],
) -> tuple[float, int, State, float] | None:
    event = plan.events[event_index]
    low_time = start_time
    high_time = end_time
    low_guard = start_guard
    high_guard = end_guard
    high_state = end_state
    for _ in range(plan.localization_iterations):
        if high_time - low_time <= plan.localization_tolerance_ms:
            break
        mid_time = 0.5 * (low_time + high_time)
        candidate = advance(start_state, subsystem, start_time, mid_time)
        if not isinstance(candidate, CardiovascularStepCandidate):
            raise TypeError("advance must return CardiovascularStepCandidate.")
        _require_state_signature(candidate.state, state_signature, "localization replay")
        if not candidate.accepted:
            return None
        guards, finite = _evaluate_event_values(
            event_values, candidate.state, mid_time, len(plan.events)
        )
        if not finite:
            return None
        mid_guard = guards[event_index]
        if _event_crossed(event, low_guard, mid_guard):
            high_time = mid_time
            high_guard = mid_guard
            high_state = candidate.state
        else:
            low_time = mid_time
            low_guard = mid_guard
    return high_time, event_index, high_state, high_guard


def _event_crossed(event: CardiovascularEventSpec, left: float, right: float, /) -> bool:
    if not math.isfinite(left) or not math.isfinite(right) or left == 0.0:
        return False
    if event.direction > 0:
        return left < 0.0 <= right
    if event.direction < 0:
        return left > 0.0 >= right
    return (left < 0.0 <= right) or (left > 0.0 >= right)


def _evaluate_event_values(
    evaluator: EventEvaluator[Any], state: Any, time_ms: float, count: int, /
) -> tuple[np.ndarray, bool]:
    values = np.asarray(evaluator(state, time_ms), dtype=float)
    if values.shape != (count,):
        raise ValueError("event_values must return one scalar per prepared event.")
    return values, bool(np.all(np.isfinite(values)))


def _make_schedule_evidence(
    prepared: PreparedCardiovascularScheduler,
    scheduled_active: np.ndarray,
    event_source: np.ndarray,
    event_times: np.ndarray,
    event_active: np.ndarray,
    guard_before: np.ndarray,
    guard_after: np.ndarray,
    guard_slope: np.ndarray,
    saltation: np.ndarray,
    completed_steps: int,
    event_count: int,
    terminal: bool,
    status: CardiovascularRuntimeStatus,
    /,
) -> CardiovascularScheduleEvidence:
    status_ = CardiovascularRuntimeStatus(status)
    arrays = (
        prepared.owner_indices,
        prepared.start_times_ms,
        prepared.end_times_ms,
        jnp.asarray(scheduled_active),
        jnp.asarray(event_source),
        jnp.asarray(event_times),
        jnp.asarray(event_active),
        jnp.asarray(guard_before),
        jnp.asarray(guard_after),
        jnp.asarray(guard_slope),
        jnp.asarray(saltation),
        jnp.asarray(completed_steps, dtype=jnp.int32),
        jnp.asarray(event_count, dtype=jnp.int32),
        jnp.asarray(terminal),
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-schedule-evidence",
            "execution": prepared.execution_manifest_id,
            "prepared": prepared.prepared_id,
            "status": status_.value,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    return CardiovascularScheduleEvidence(
        scheduled_owner_indices=arrays[0],
        scheduled_start_times_ms=arrays[1],
        scheduled_end_times_ms=arrays[2],
        scheduled_active=arrays[3],
        event_source_indices=arrays[4],
        event_times_ms=arrays[5],
        event_active=arrays[6],
        event_guard_before=arrays[7],
        event_guard_after=arrays[8],
        event_guard_slope_per_ms=arrays[9],
        saltation_eligible=arrays[10],
        scheduled_step_count=arrays[11],
        event_count=arrays[12],
        terminal_event=arrays[13],
        execution_manifest_id=prepared.execution_manifest_id,
        prepared_id=prepared.prepared_id,
        status=status_,
        evidence_id=evidence_id,
    )


def _failed_schedule_candidate(
    prepared: PreparedCardiovascularScheduler,
    initial_state: State,
    proposed_state: State,
    status: CardiovascularRuntimeStatus,
    completed_steps: int,
    event_records: Sequence[tuple[int, float, float, float, bool]],
    /,
) -> CardiovascularScheduleCandidate[State]:
    maximum_steps = int(prepared.owner_indices.shape[0])
    maximum_events = _prepared_event_capacity(prepared)
    event_source = np.full((maximum_events,), -1, dtype=np.int32)
    event_times = np.zeros((maximum_events,), dtype=float)
    event_active = np.zeros((maximum_events,), dtype=bool)
    guard_before = np.zeros((maximum_events,), dtype=float)
    guard_after = np.zeros((maximum_events,), dtype=float)
    guard_slope = np.zeros((maximum_events,), dtype=float)
    saltation = np.zeros((maximum_events,), dtype=bool)
    for index, record in enumerate(event_records):
        source, time, before, after, eligible = record
        event_source[index] = source
        event_times[index] = time
        event_active[index] = True
        guard_before[index] = before
        guard_after[index] = after
        saltation[index] = eligible
    evidence = _make_schedule_evidence(
        prepared,
        np.arange(maximum_steps) < completed_steps,
        event_source,
        event_times,
        event_active,
        guard_before,
        guard_after,
        guard_slope,
        saltation,
        completed_steps,
        len(event_records),
        False,
        status,
    )
    return CardiovascularScheduleCandidate(initial_state, proposed_state, evidence)


def _schedule_routes_equal(
    reference: CardiovascularScheduleEvidence,
    replay: CardiovascularScheduleEvidence,
    tolerance: float,
    /,
) -> bool:
    if (
        reference.execution_manifest_id != replay.execution_manifest_id
        or reference.prepared_id != replay.prepared_id
    ):
        return False
    for reference_schedule, replay_schedule in (
        (reference.scheduled_owner_indices, replay.scheduled_owner_indices),
        (reference.scheduled_start_times_ms, replay.scheduled_start_times_ms),
        (reference.scheduled_end_times_ms, replay.scheduled_end_times_ms),
    ):
        if not np.array_equal(
            np.asarray(reference_schedule), np.asarray(replay_schedule)
        ):
            return False
    reference_active = np.asarray(reference.event_active)
    replay_active = np.asarray(replay.event_active)
    if not np.array_equal(
        np.asarray(reference.scheduled_active), np.asarray(replay.scheduled_active)
    ):
        return False
    if not np.array_equal(reference_active, replay_active):
        return False
    if not np.array_equal(
        np.asarray(reference.event_source_indices)[reference_active],
        np.asarray(replay.event_source_indices)[replay_active],
    ):
        return False
    if not np.allclose(
        np.asarray(reference.event_times_ms)[reference_active],
        np.asarray(replay.event_times_ms)[replay_active],
        rtol=0.0,
        atol=tolerance,
    ):
        return False
    return bool(
        reference.status is replay.status
        and int(reference.scheduled_step_count) == int(replay.scheduled_step_count)
        and int(reference.event_count) == int(replay.event_count)
    )


def _tree_exact_equal(left: Any, right: Any, /) -> bool:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if left_structure != right_structure or len(left_leaves) != len(right_leaves):
        return False
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _state_leaf_signature(
    value: Any, /
) -> tuple[Any, tuple[tuple[tuple[int, ...], str], ...]]:
    leaves, structure = jax.tree_util.tree_flatten(value)
    records = tuple(
        (tuple(np.asarray(leaf).shape), np.asarray(leaf).dtype.str) for leaf in leaves
    )
    return structure, records


def _require_state_signature(
    value: Any,
    expected: tuple[Any, tuple[tuple[tuple[int, ...], str], ...]],
    phase: str,
    /,
) -> None:
    observed = _state_leaf_signature(value)
    if observed != expected:
        raise ValueError(
            f"{phase} changed the exact state leaf count, shape, dtype, or PyTree."
        )


def _checkpoint_array(value: ArrayLike, /) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc" or np.any(~np.isfinite(array)):
        raise ValueError("Checkpoint payloads must be finite numeric or boolean arrays.")
    return array


def _checkpoint_archive_limits(
    capacity: CardiovascularCapacityManifest, /
) -> ArrayArchiveLimits:
    """Derive untrusted-read limits from the admitted execution capacity."""

    members = capacity.maximum_checkpoint_arrays + 1
    manifest_bytes = 65_536 + 4_096 * capacity.maximum_checkpoint_arrays
    directory_bytes = max(4_096, 256 * members)
    aggregate_bytes = capacity.maximum_checkpoint_bytes + manifest_bytes
    container_bytes = aggregate_bytes + directory_bytes + 4_096
    return ArrayArchiveLimits(
        max_container_bytes=container_bytes,
        max_aggregate_bytes=aggregate_bytes,
        max_member_bytes=max(capacity.maximum_checkpoint_bytes, manifest_bytes),
        max_manifest_bytes=manifest_bytes,
        max_members=members,
        max_central_directory_bytes=directory_bytes,
        max_npy_header_bytes=min(65_536, max(256, capacity.maximum_checkpoint_bytes)),
        max_array_rank=8,
        max_axis_length=capacity.maximum_state_values,
        max_array_elements=capacity.maximum_state_values,
        max_total_array_elements=capacity.maximum_state_values,
        max_dtype_itemsize=16,
        max_manifest_nesting=16,
    )


def _tree_value_count(value: Any, /) -> int:
    leaves = jax.tree_util.tree_leaves(value)
    return sum(int(np.asarray(leaf).size) for leaf in leaves)


def _identifier(value: object, name: str, /) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _nonnegative_integer(value: object, name: str, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _positive_integer(value: object, name: str, /) -> int:
    result = _nonnegative_integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


__all__ = [
    "CardiovascularCapacityAdmission",
    "CardiovascularCapacityManifest",
    "CardiovascularCapacityRequest",
    "CardiovascularCheckpointRecord",
    "CardiovascularCohortCaseCandidate",
    "CardiovascularCohortEvidence",
    "CardiovascularCohortExecution",
    "CardiovascularCohortResult",
    "CardiovascularDistributedCapability",
    "CardiovascularDistributedCollectiveEvidence",
    "CardiovascularDistributedCollectiveExecution",
    "CardiovascularDistributedContract",
    "CardiovascularDistributedReferenceEvidence",
    "CardiovascularDistributedReferenceExecution",
    "CardiovascularDistributedSolverState",
    "CardiovascularEventSpec",
    "CardiovascularExecutionManifest",
    "CardiovascularExecutionRoute",
    "CardiovascularLifecycleCheckpointCodec",
    "CardiovascularMultiratePlan",
    "CardiovascularReplayEvidence",
    "CardiovascularRuntimeError",
    "CardiovascularRuntimeStatus",
    "CardiovascularScheduleCandidate",
    "CardiovascularScheduleCommit",
    "CardiovascularScheduleEvidence",
    "CardiovascularSaltationPolicy",
    "CardiovascularSerialExecution",
    "CardiovascularSingleDeviceEvidence",
    "CardiovascularStepCandidate",
    "PreparedCardiovascularCohort",
    "PreparedCardiovascularScheduler",
    "admit_cardiovascular_capacity",
    "cardiovascular_runtime_diagnostic",
    "commit_cardiovascular_schedule",
    "execute_cardiovascular_cohort",
    "execute_cardiovascular_distributed_collective",
    "execute_cardiovascular_distributed_reference",
    "execute_cardiovascular_distributed_replay",
    "observe_single_device_runtime",
    "prepare_cardiovascular_cohort",
    "prepare_cardiovascular_distributed_execution",
    "prepare_cardiovascular_scheduler",
    "read_cardiovascular_distributed_solver_checkpoint",
    "replay_cardiovascular_schedule",
    "require_cardiovascular_distributed_transport",
    "run_cardiovascular_schedule",
    "write_cardiovascular_distributed_solver_checkpoint",
]
