#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._platform_support import (
    _identifier,
    _nonnegative_integer,
    _positive_integer,
    TensorNetworkExecutionManifest,
    TensorNetworkFailure,
)


if TYPE_CHECKING:
    from ..solver._runtime_lifecycle import RuntimeCheckpointEnvelope


class TensorNetworkRunStatus(StrEnum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TensorNetworkCheckpointError(RuntimeError):
    failure: TensorNetworkFailure

    def __init__(self, failure: TensorNetworkFailure, detail: str, /):
        failure_ = TensorNetworkFailure(failure)
        if failure_ not in (
            TensorNetworkFailure.CHECKPOINT_NOT_ACCEPTED,
            TensorNetworkFailure.CHECKPOINT_MISMATCH,
        ):
            raise ValueError("Checkpoint errors require a checkpoint failure category.")
        self.failure = failure_
        self.detail = _identifier(detail, "checkpoint failure detail")
        super().__init__(f"{failure_.value}: {self.detail}")


class TensorNetworkCancelledError(RuntimeError):
    failure = TensorNetworkFailure.CANCELLED

    def __init__(self, run_id: str, detail: str, /):
        self.run_id = _identifier(run_id, "run_id")
        self.detail = _identifier(detail, "cancellation detail")
        super().__init__(f"cancelled tensor-network run {self.run_id}: {self.detail}")


class TensorNetworkCheckpointRecord(StrictModule, NonTrainableState):
    execution_manifest_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)
    generation: int = eqx.field(static=True)
    accepted_step: int = eqx.field(static=True)
    accepted_time: float = eqx.field(static=True)
    artifact_name: str = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        execution_manifest_id: str,
        checkpoint_id: str,
        generation: int,
        accepted_step: int,
        accepted_time: float,
        artifact_name: str,
    ):
        identifiers = tuple(
            _identifier(value, name)
            for value, name in (
                (execution_manifest_id, "execution_manifest_id"),
                (checkpoint_id, "checkpoint_id"),
                (artifact_name, "artifact_name"),
            )
        )
        generation_ = _nonnegative_integer(generation, "generation")
        step = _nonnegative_integer(accepted_step, "accepted_step")
        time = float(accepted_time)
        if not math.isfinite(time) or time < 0.0:
            raise ValueError("accepted_time must be finite and nonnegative.")
        self.execution_manifest_id = identifiers[0]
        self.checkpoint_id = identifiers[1]
        self.generation = generation_
        self.accepted_step = step
        self.accepted_time = time
        self.artifact_name = identifiers[2]
        self.record_id = canonical_fingerprint(
            {
                "kind": "tensor-network-checkpoint-record",
                "execution_manifest": identifiers[0],
                "checkpoint": identifiers[1],
                "generation": generation_,
                "accepted_step": step,
                "accepted_time": time,
                "artifact_name": identifiers[2],
            }
        )


class TensorNetworkCheckpointPublication(StrictModule, NonTrainableState):
    published: bool = eqx.field(static=True)
    failure: TensorNetworkFailure = eqx.field(static=True)
    record: TensorNetworkCheckpointRecord | None
    execution_manifest_id: str = eqx.field(static=True)
    publication_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        published: bool,
        failure: TensorNetworkFailure,
        record: TensorNetworkCheckpointRecord | None,
        execution_manifest_id: str,
    ):
        published_ = bool(published)
        failure_ = TensorNetworkFailure(failure)
        if published_:
            if failure_ != TensorNetworkFailure.NONE or not isinstance(
                record, TensorNetworkCheckpointRecord
            ):
                raise ValueError("Published checkpoints require a successful record.")
        elif (
            failure_ != TensorNetworkFailure.CHECKPOINT_NOT_ACCEPTED or record is not None
        ):
            raise ValueError("Unpublished checkpoints must be explicitly unaccepted.")
        manifest_id = _identifier(execution_manifest_id, "execution_manifest_id")
        if record is not None and record.execution_manifest_id != manifest_id:
            raise ValueError("Checkpoint record belongs to another execution manifest.")
        self.published = published_
        self.failure = failure_
        self.record = record
        self.execution_manifest_id = manifest_id
        self.publication_id = canonical_fingerprint(
            {
                "kind": "tensor-network-checkpoint-publication",
                "execution_manifest": manifest_id,
                "published": published_,
                "failure": failure_.value,
                "record": None if record is None else record.record_id,
            }
        )


class TensorNetworkAcceptedCheckpointBoundary(NonTrainableState):
    """Host boundary that atomically publishes accepted runtime checkpoints only."""

    def __init__(
        self,
        root: str | Path,
        execution: TensorNetworkExecutionManifest,
        /,
        *,
        retention: int = 3,
    ):
        from ..solver._production_runtime import (
            CheckpointGenerationPolicy,
            DurableCheckpointStore,
            ProductionCaseManifest,
        )

        if not isinstance(execution, TensorNetworkExecutionManifest):
            raise TypeError("execution must be TensorNetworkExecutionManifest.")
        self.execution = execution
        self.case_manifest = ProductionCaseManifest(
            problem_id=execution.support.support_tuple_id,
            method_id=execution.method_id,
            precision_id=execution.precision_policy_id,
            topology_id=execution.structure_id,
            geometry_layout_id=execution.source_id,
            dtype=execution.support.dtype,
        )
        if self.case_manifest.backend != execution.support.backend:
            raise ValueError(
                "Checkpoint runtime backend differs from the exact support tuple."
            )
        self.store = DurableCheckpointStore(
            root,
            self.case_manifest,
            CheckpointGenerationPolicy(_positive_integer(retention, "retention")),
        )
        self._lock = threading.Lock()

    def _validate_envelope(self, envelope: RuntimeCheckpointEnvelope, /) -> None:
        from ..solver._runtime_lifecycle import RuntimeCheckpointEnvelope

        if not isinstance(envelope, RuntimeCheckpointEnvelope):
            raise TypeError("envelope must be RuntimeCheckpointEnvelope.")
        expected = (
            (envelope.mesh_id, self.execution.structure_id, "structure"),
            (envelope.method_id, self.execution.method_id, "method"),
            (
                envelope.precision_id,
                self.execution.precision_policy_id,
                "precision policy",
            ),
            (envelope.topology_epoch_id, self.execution.source_id, "source"),
        )
        changed = tuple(
            name for observed, required, name in expected if observed != required
        )
        if changed:
            raise TensorNetworkCheckpointError(
                TensorNetworkFailure.CHECKPOINT_MISMATCH,
                f"checkpoint changed {', '.join(changed)} identity",
            )
        step = int(np.asarray(envelope.step_index))
        time = float(np.asarray(envelope.time))
        cursor = int(np.asarray(envelope.schedule_cursor))
        if step < 0 or cursor < 0 or not math.isfinite(time) or time < 0.0:
            raise TensorNetworkCheckpointError(
                TensorNetworkFailure.CHECKPOINT_MISMATCH,
                "checkpoint accepted coordinates are invalid",
            )

    def publish(
        self,
        generation: int,
        envelope: RuntimeCheckpointEnvelope,
        /,
        *,
        accepted: bool,
    ) -> TensorNetworkCheckpointPublication:
        generation_ = _nonnegative_integer(generation, "generation")
        if not bool(accepted):
            return TensorNetworkCheckpointPublication(
                published=False,
                failure=TensorNetworkFailure.CHECKPOINT_NOT_ACCEPTED,
                record=None,
                execution_manifest_id=self.execution.manifest_id,
            )
        self._validate_envelope(envelope)
        with self._lock:
            path = self.store.commit(generation_, envelope)
        record = TensorNetworkCheckpointRecord(
            execution_manifest_id=self.execution.manifest_id,
            checkpoint_id=envelope.checkpoint_id,
            generation=generation_,
            accepted_step=int(np.asarray(envelope.step_index)),
            accepted_time=float(np.asarray(envelope.time)),
            artifact_name=path.name,
        )
        return TensorNetworkCheckpointPublication(
            published=True,
            failure=TensorNetworkFailure.NONE,
            record=record,
            execution_manifest_id=self.execution.manifest_id,
        )

    def latest(
        self,
        state_template: Any,
        /,
        *,
        controller_template: Any = (),
        observer_templates: Sequence[Any] = (),
        rng_template: Any = (),
    ) -> RuntimeCheckpointEnvelope:
        with self._lock:
            envelope = self.store.latest(
                state_template,
                controller_template=controller_template,
                observer_templates=observer_templates,
                rng_template=rng_template,
            )
        self._validate_envelope(envelope)
        return envelope


class TensorNetworkReplayRecord(StrictModule, NonTrainableState):
    execution_manifest_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    backend_evidence_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    accepted_steps: int = eqx.field(static=True)
    output_signature_id: str = eqx.field(static=True)
    output_digest: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)

    def __init__(
        self,
        execution: TensorNetworkExecutionManifest,
        output: Any,
        /,
        *,
        checkpoint_id: str,
        route_id: str,
        accepted_steps: int,
    ):
        if not isinstance(execution, TensorNetworkExecutionManifest):
            raise TypeError("execution must be TensorNetworkExecutionManifest.")
        checkpoint = _identifier(checkpoint_id, "checkpoint_id")
        route = _identifier(route_id, "route_id")
        steps = _nonnegative_integer(accepted_steps, "accepted_steps")
        leaves = tuple(np.asarray(leaf) for leaf in jax.tree.leaves(output))
        if not leaves or any(
            leaf.dtype.hasobject or leaf.dtype.kind not in "biufc" for leaf in leaves
        ):
            raise TypeError("Replay output must be a nonempty numerical array PyTree.")
        fingerprint = array_tree_fingerprint(output)
        signature_id = canonical_fingerprint(fingerprint["signature"])
        digest = str(fingerprint["sha256"])
        self.execution_manifest_id = execution.manifest_id
        self.support_tuple_id = execution.support.support_tuple_id
        self.structure_id = execution.structure_id
        self.method_id = execution.method_id
        self.precision_policy_id = execution.precision_policy_id
        self.source_id = execution.source_id
        self.backend_evidence_id = execution.backend_evidence_id
        self.checkpoint_id = checkpoint
        self.route_id = route
        self.accepted_steps = steps
        self.output_signature_id = signature_id
        self.output_digest = digest
        self.replay_id = canonical_fingerprint(
            {
                "kind": "tensor-network-replay-record",
                "execution_manifest": execution.manifest_id,
                "support": self.support_tuple_id,
                "structure": self.structure_id,
                "method": self.method_id,
                "precision_policy": self.precision_policy_id,
                "source": self.source_id,
                "backend_evidence": self.backend_evidence_id,
                "checkpoint": checkpoint,
                "route": route,
                "accepted_steps": steps,
                "output_signature": signature_id,
                "output_digest": digest,
            }
        )


class TensorNetworkReplayMismatchError(RuntimeError):
    failure = TensorNetworkFailure.REPLAY_MISMATCH

    def __init__(self, mismatches: Sequence[str], /):
        mismatches_ = tuple(_identifier(value, "replay mismatch") for value in mismatches)
        if not mismatches_:
            raise ValueError("Replay mismatch errors require mismatch coordinates.")
        self.mismatches = mismatches_
        super().__init__(f"replay-mismatch: {', '.join(mismatches_)}")


class TensorNetworkReplayCompatibility(StrictModule, NonTrainableState):
    reference_replay_id: str = eqx.field(static=True)
    candidate_replay_id: str = eqx.field(static=True)
    compatible: bool = eqx.field(static=True)
    failure: TensorNetworkFailure = eqx.field(static=True)
    mismatches: tuple[str, ...] = eqx.field(static=True)
    compatibility_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_replay_id: str,
        candidate_replay_id: str,
        compatible: bool,
        failure: TensorNetworkFailure,
        mismatches: Sequence[str],
        /,
    ):
        reference = _identifier(reference_replay_id, "reference_replay_id")
        candidate = _identifier(candidate_replay_id, "candidate_replay_id")
        compatible_ = bool(compatible)
        failure_ = TensorNetworkFailure(failure)
        mismatches_ = tuple(_identifier(value, "replay mismatch") for value in mismatches)
        if compatible_:
            if failure_ != TensorNetworkFailure.NONE or mismatches_:
                raise ValueError("Compatible replay records cannot contain mismatches.")
        elif failure_ != TensorNetworkFailure.REPLAY_MISMATCH or not mismatches_:
            raise ValueError("Incompatible replay records require typed mismatches.")
        self.reference_replay_id = reference
        self.candidate_replay_id = candidate
        self.compatible = compatible_
        self.failure = failure_
        self.mismatches = mismatches_
        self.compatibility_id = canonical_fingerprint(
            {
                "kind": "tensor-network-replay-compatibility",
                "reference": reference,
                "candidate": candidate,
                "compatible": compatible_,
                "failure": failure_.value,
                "mismatches": mismatches_,
            }
        )

    def require_compatible(self) -> str:
        if not self.compatible:
            raise TensorNetworkReplayMismatchError(self.mismatches)
        return self.compatibility_id


def compare_tensor_network_replays(
    reference: TensorNetworkReplayRecord,
    candidate: TensorNetworkReplayRecord,
    /,
) -> TensorNetworkReplayCompatibility:
    """Compare deterministic replay identities and numerical output bit-for-bit."""

    if not isinstance(reference, TensorNetworkReplayRecord) or not isinstance(
        candidate, TensorNetworkReplayRecord
    ):
        raise TypeError("Replay comparison requires two replay records.")
    coordinates = (
        ("support tuple", reference.support_tuple_id, candidate.support_tuple_id),
        ("structure", reference.structure_id, candidate.structure_id),
        ("method", reference.method_id, candidate.method_id),
        (
            "precision policy",
            reference.precision_policy_id,
            candidate.precision_policy_id,
        ),
        ("source", reference.source_id, candidate.source_id),
        (
            "backend evidence",
            reference.backend_evidence_id,
            candidate.backend_evidence_id,
        ),
        ("checkpoint", reference.checkpoint_id, candidate.checkpoint_id),
        ("route", reference.route_id, candidate.route_id),
        ("accepted steps", reference.accepted_steps, candidate.accepted_steps),
        (
            "output structure",
            reference.output_signature_id,
            candidate.output_signature_id,
        ),
        ("output digest", reference.output_digest, candidate.output_digest),
    )
    mismatches = tuple(name for name, left, right in coordinates if left != right)
    compatible = not mismatches
    failure = (
        TensorNetworkFailure.NONE if compatible else TensorNetworkFailure.REPLAY_MISMATCH
    )
    return TensorNetworkReplayCompatibility(
        reference.replay_id,
        candidate.replay_id,
        compatible,
        failure,
        mismatches,
    )


class TensorNetworkSupervisorState(StrictModule, NonTrainableState):
    run_id: str = eqx.field(static=True)
    status: TensorNetworkRunStatus = eqx.field(static=True)
    transition_index: int = eqx.field(static=True)
    failure: TensorNetworkFailure = eqx.field(static=True)
    detail: str = eqx.field(static=True)
    last_checkpoint_id: str | None = eqx.field(static=True)
    replay_id: str | None = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        run_id: str,
        status: TensorNetworkRunStatus,
        transition_index: int,
        failure: TensorNetworkFailure,
        detail: str,
        last_checkpoint_id: str | None = None,
        replay_id: str | None = None,
    ):
        run = _identifier(run_id, "run_id")
        status_ = TensorNetworkRunStatus(status)
        index = _nonnegative_integer(transition_index, "transition_index")
        failure_ = TensorNetworkFailure(failure)
        detail_ = _identifier(detail, "supervisor detail")
        checkpoint = (
            None
            if last_checkpoint_id is None
            else _identifier(last_checkpoint_id, "last_checkpoint_id")
        )
        replay = None if replay_id is None else _identifier(replay_id, "replay_id")
        if status_ == TensorNetworkRunStatus.CANCELLED:
            if failure_ != TensorNetworkFailure.CANCELLED:
                raise ValueError(
                    "Cancelled supervisor state requires cancellation failure."
                )
        elif status_ == TensorNetworkRunStatus.FAILED:
            if failure_ in (TensorNetworkFailure.NONE, TensorNetworkFailure.CANCELLED):
                raise ValueError(
                    "Failed supervisor state requires a non-cancellation failure."
                )
        elif failure_ != TensorNetworkFailure.NONE:
            raise ValueError("Non-failed supervisor state cannot contain a failure.")
        if status_ == TensorNetworkRunStatus.COMPLETED and replay is None:
            raise ValueError("Completed runs require deterministic replay evidence.")
        if status_ != TensorNetworkRunStatus.COMPLETED and replay is not None:
            raise ValueError("Only completed runs may cite replay evidence.")
        self.run_id = run
        self.status = status_
        self.transition_index = index
        self.failure = failure_
        self.detail = detail_
        self.last_checkpoint_id = checkpoint
        self.replay_id = replay
        self.state_id = canonical_fingerprint(
            {
                "kind": "tensor-network-supervisor-state",
                "run": run,
                "status": status_.value,
                "transition_index": index,
                "failure": failure_.value,
                "detail": detail_,
                "last_checkpoint": checkpoint,
                "replay": replay,
            }
        )


class TensorNetworkRunSupervisor(NonTrainableState):
    """Finite host lifecycle; numerical operations remain caller-owned and never retry."""

    def __init__(self, execution: TensorNetworkExecutionManifest, /):
        if not isinstance(execution, TensorNetworkExecutionManifest):
            raise TypeError("execution must be TensorNetworkExecutionManifest.")
        self.execution = execution
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._state = TensorNetworkSupervisorState(
            run_id=execution.manifest_id,
            status=TensorNetworkRunStatus.READY,
            transition_index=0,
            failure=TensorNetworkFailure.NONE,
            detail="admitted and ready",
        )

    @property
    def state(self) -> TensorNetworkSupervisorState:
        with self._lock:
            return self._state

    def _replace(
        self,
        status: TensorNetworkRunStatus,
        failure: TensorNetworkFailure,
        detail: str,
        /,
        *,
        checkpoint_id: str | None = None,
        replay_id: str | None = None,
    ) -> TensorNetworkSupervisorState:
        current = self._state
        self._state = TensorNetworkSupervisorState(
            run_id=current.run_id,
            status=status,
            transition_index=current.transition_index + 1,
            failure=failure,
            detail=detail,
            last_checkpoint_id=(
                current.last_checkpoint_id if checkpoint_id is None else checkpoint_id
            ),
            replay_id=replay_id,
        )
        return self._state

    def start(self) -> TensorNetworkSupervisorState:
        with self._lock:
            if self._state.status != TensorNetworkRunStatus.READY:
                raise RuntimeError("Only a ready tensor-network run can start.")
            if self._cancel.is_set():
                return self._replace(
                    TensorNetworkRunStatus.CANCELLED,
                    TensorNetworkFailure.CANCELLED,
                    "cancellation requested before execution",
                )
            return self._replace(
                TensorNetworkRunStatus.RUNNING,
                TensorNetworkFailure.NONE,
                "execution started",
            )

    def request_cancellation(self, detail: str) -> TensorNetworkSupervisorState:
        detail_ = _identifier(detail, "cancellation detail")
        self._cancel.set()
        with self._lock:
            if self._state.status in (
                TensorNetworkRunStatus.READY,
                TensorNetworkRunStatus.RUNNING,
            ):
                return self._replace(
                    TensorNetworkRunStatus.CANCELLED,
                    TensorNetworkFailure.CANCELLED,
                    detail_,
                )
            return self._state

    def raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise TensorNetworkCancelledError(
                self.execution.manifest_id, self.state.detail
            )

    def record_checkpoint(
        self, publication: TensorNetworkCheckpointPublication, /
    ) -> TensorNetworkSupervisorState:
        if not isinstance(publication, TensorNetworkCheckpointPublication):
            raise TypeError("publication must be TensorNetworkCheckpointPublication.")
        if not publication.published or publication.record is None:
            raise TensorNetworkCheckpointError(
                TensorNetworkFailure.CHECKPOINT_NOT_ACCEPTED,
                "supervisor rejects an unpublished checkpoint",
            )
        if publication.record.execution_manifest_id != self.execution.manifest_id:
            raise TensorNetworkCheckpointError(
                TensorNetworkFailure.CHECKPOINT_MISMATCH,
                "checkpoint belongs to another execution manifest",
            )
        with self._lock:
            if self._state.status != TensorNetworkRunStatus.RUNNING:
                raise RuntimeError("Only a running workflow can record a checkpoint.")
            return self._replace(
                TensorNetworkRunStatus.RUNNING,
                TensorNetworkFailure.NONE,
                "accepted checkpoint published",
                checkpoint_id=publication.record.checkpoint_id,
            )

    def complete(
        self, replay: TensorNetworkReplayRecord, /
    ) -> TensorNetworkSupervisorState:
        if not isinstance(replay, TensorNetworkReplayRecord):
            raise TypeError("replay must be TensorNetworkReplayRecord.")
        with self._lock:
            if self._state.status != TensorNetworkRunStatus.RUNNING:
                raise RuntimeError("Only a running workflow can complete.")
            if replay.execution_manifest_id != self.execution.manifest_id:
                return self._replace(
                    TensorNetworkRunStatus.FAILED,
                    TensorNetworkFailure.REPLAY_MISMATCH,
                    "replay belongs to another execution manifest",
                )
            if (
                self._state.last_checkpoint_id is not None
                and replay.checkpoint_id != self._state.last_checkpoint_id
            ):
                return self._replace(
                    TensorNetworkRunStatus.FAILED,
                    TensorNetworkFailure.REPLAY_MISMATCH,
                    "replay checkpoint differs from the last accepted checkpoint",
                )
            return self._replace(
                TensorNetworkRunStatus.COMPLETED,
                TensorNetworkFailure.NONE,
                "execution completed with replay evidence",
                replay_id=replay.replay_id,
            )

    def fail(
        self, failure: TensorNetworkFailure, detail: str, /
    ) -> TensorNetworkSupervisorState:
        failure_ = TensorNetworkFailure(failure)
        if failure_ in (TensorNetworkFailure.NONE, TensorNetworkFailure.CANCELLED):
            raise ValueError("fail requires a non-cancellation failure category.")
        with self._lock:
            if self._state.status not in (
                TensorNetworkRunStatus.READY,
                TensorNetworkRunStatus.RUNNING,
            ):
                raise RuntimeError("Terminal tensor-network runs cannot fail again.")
            return self._replace(
                TensorNetworkRunStatus.FAILED,
                failure_,
                _identifier(detail, "failure detail"),
            )


TelemetryValue: TypeAlias = str | int | float | bool


class TensorNetworkTelemetryPolicy(StrictModule, NonTrainableState):
    permitted_fields: tuple[str, ...] = eqx.field(static=True)
    redacted_fields: tuple[str, ...] = eqx.field(static=True)
    maximum_fields: int = eqx.field(static=True)
    maximum_text_characters: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        permitted_fields: Sequence[str],
        /,
        *,
        redacted_fields: Sequence[str],
        maximum_fields: int = 64,
        maximum_text_characters: int = 4096,
    ):
        permitted = tuple(
            sorted(_identifier(v, "telemetry field") for v in permitted_fields)
        )
        redacted = tuple(
            sorted(_identifier(v, "redacted field") for v in redacted_fields)
        )
        if not permitted or len(set(permitted)) != len(permitted):
            raise ValueError("Telemetry permitted fields must be nonempty and unique.")
        if len(set(redacted)) != len(redacted) or not set(redacted).issubset(permitted):
            raise ValueError("Redacted telemetry fields must be unique and permitted.")
        maximum = _positive_integer(maximum_fields, "maximum_fields")
        characters = _positive_integer(maximum_text_characters, "maximum_text_characters")
        if len(permitted) > maximum:
            raise ValueError("Telemetry permitted fields exceed maximum_fields.")
        self.permitted_fields = permitted
        self.redacted_fields = redacted
        self.maximum_fields = maximum
        self.maximum_text_characters = characters
        self.policy_id = canonical_fingerprint(
            {
                "kind": "tensor-network-telemetry-policy",
                "permitted_fields": permitted,
                "redacted_fields": redacted,
                "maximum_fields": maximum,
                "maximum_text_characters": characters,
            }
        )


class TensorNetworkTelemetryRecord(StrictModule, NonTrainableState):
    run_id: str = eqx.field(static=True)
    event: str = eqx.field(static=True)
    sequence: int = eqx.field(static=True)
    timestamp_ns: int = eqx.field(static=True)
    attributes: tuple[tuple[str, TelemetryValue], ...] = eqx.field(static=True)
    redacted_fields: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        run_id: str,
        event: str,
        sequence: int,
        timestamp_ns: int,
        attributes: Sequence[tuple[str, TelemetryValue]],
        redacted_fields: Sequence[str],
        policy_id: str,
        /,
    ):
        run = _identifier(run_id, "run_id")
        event_ = _identifier(event, "event")
        sequence_ = _nonnegative_integer(sequence, "sequence")
        timestamp = _nonnegative_integer(timestamp_ns, "timestamp_ns")
        policy = _identifier(policy_id, "policy_id")
        attributes_ = tuple(
            sorted(
                (
                    _identifier(name, "telemetry field"),
                    _telemetry_value(value, 2**31 - 1),
                )
                for name, value in attributes
            )
        )
        names = tuple(name for name, _ in attributes_)
        if len(set(names)) != len(names):
            raise ValueError("Telemetry record fields must be unique.")
        redacted = tuple(
            sorted(_identifier(name, "redacted field") for name in redacted_fields)
        )
        if len(set(redacted)) != len(redacted) or not set(redacted).issubset(names):
            raise ValueError("Telemetry record redacted fields are invalid.")
        if any(dict(attributes_)[name] != "<redacted>" for name in redacted):
            raise ValueError(
                "Sensitive telemetry values must be redacted before storage."
            )
        self.run_id = run
        self.event = event_
        self.sequence = sequence_
        self.timestamp_ns = timestamp
        self.attributes = attributes_
        self.redacted_fields = redacted
        self.policy_id = policy
        self.record_id = canonical_fingerprint(
            {
                "kind": "tensor-network-telemetry-record",
                "run": run,
                "event": event_,
                "sequence": sequence_,
                "timestamp_ns": timestamp,
                "attributes": dict(attributes_),
                "redacted_fields": redacted,
                "policy": policy,
            }
        )


def _telemetry_value(value: object, maximum_characters: int, /) -> TelemetryValue:
    if type(value) not in (str, int, float, bool):
        raise TypeError("Telemetry values must be JSON scalar values.")
    if isinstance(value, str):
        if len(value) > maximum_characters:
            raise ValueError("Telemetry text exceeds its bounded capacity.")
        return value
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Telemetry numbers must be finite.")
    return value


def redact_tensor_network_telemetry(
    policy: TensorNetworkTelemetryPolicy,
    attributes: Mapping[str, object],
    /,
    *,
    run_id: str,
    event: str,
    sequence: int,
    timestamp_ns: int,
) -> TensorNetworkTelemetryRecord:
    """Create a bounded record whose sensitive values never enter stored state."""

    if not isinstance(policy, TensorNetworkTelemetryPolicy):
        raise TypeError("policy must be TensorNetworkTelemetryPolicy.")
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping.")
    if len(attributes) > policy.maximum_fields:
        raise ValueError("Telemetry attributes exceed maximum_fields.")
    keys = tuple(_identifier(key, "telemetry field") for key in attributes)
    if any(key not in policy.permitted_fields for key in keys):
        raise ValueError("Telemetry contains a field not declared by the policy.")
    redacted = set(policy.redacted_fields)
    normalized = tuple(
        sorted(
            (
                key,
                "<redacted>"
                if key in redacted
                else _telemetry_value(attributes[key], policy.maximum_text_characters),
            )
            for key in keys
        )
    )
    sequence_ = _nonnegative_integer(sequence, "sequence")
    timestamp = _nonnegative_integer(timestamp_ns, "timestamp_ns")
    run = _identifier(run_id, "run_id")
    event_ = _identifier(event, "event")
    redacted_present = tuple(key for key, _ in normalized if key in redacted)
    return TensorNetworkTelemetryRecord(
        run,
        event_,
        sequence_,
        timestamp,
        normalized,
        redacted_present,
        policy.policy_id,
    )


__all__ = [
    "TensorNetworkAcceptedCheckpointBoundary",
    "TensorNetworkCancelledError",
    "TensorNetworkCheckpointError",
    "TensorNetworkCheckpointPublication",
    "TensorNetworkCheckpointRecord",
    "TensorNetworkReplayCompatibility",
    "TensorNetworkReplayMismatchError",
    "TensorNetworkReplayRecord",
    "TensorNetworkRunStatus",
    "TensorNetworkRunSupervisor",
    "TensorNetworkSupervisorState",
    "TensorNetworkTelemetryPolicy",
    "TensorNetworkTelemetryRecord",
    "compare_tensor_network_replays",
    "redact_tensor_network_telemetry",
]
