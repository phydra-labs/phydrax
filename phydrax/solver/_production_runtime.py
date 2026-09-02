#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._fixed_step import (
    AbstractFixedStepMethod,
    RetriedFixedStepResult,
    retry_fixed_step,
    RobustRetryPolicy,
)
from ._runtime_lifecycle import (
    AcceptedStepTriggerGraph,
    AcceptedStepTriggerGraphState,
    ByteBoundedAsyncPublisher,
    ExactTimeSchedule,
    read_runtime_checkpoint,
    RuntimeCheckpointEnvelope,
    StreamingMomentPlan,
    StreamingMomentState,
    write_runtime_checkpoint,
)


RunStatus = Literal["ready", "running", "completed", "failed", "cancelled"]


class ProductionCaseManifest(StrictModule, NonTrainableState):
    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        problem_id: str,
        method_id: str,
        precision_id: str,
        topology_id: str,
        geometry_layout_id: str,
        dtype: str,
    ):
        values = tuple(
            str(value)
            for value in (
                problem_id,
                method_id,
                precision_id,
                topology_id,
                geometry_layout_id,
                dtype,
            )
        )
        if any(not value for value in values):
            raise ValueError("Production case manifest identities are required.")
        backend = jax.default_backend()
        (
            self.problem_id,
            self.method_id,
            self.precision_id,
            self.topology_id,
            self.geometry_layout_id,
            self.dtype,
        ) = values
        self.backend = backend
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "production-case-manifest",
                "problem": self.problem_id,
                "method": self.method_id,
                "precision": self.precision_id,
                "topology": self.topology_id,
                "geometry_layout": self.geometry_layout_id,
                "backend": backend,
                "dtype": self.dtype,
            }
        )


class CheckpointGenerationPolicy(StrictModule, NonTrainableState):
    retention: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, retention: int = 3, /):
        retention_ = int(retention)
        if retention_ <= 0:
            raise ValueError("Checkpoint retention must be positive.")
        self.retention = retention_
        self.policy_id = canonical_fingerprint(
            {"kind": "checkpoint-generation-policy", "retention": retention_}
        )


class DurableCheckpointStore:
    """Crash-consistent local checkpoint generations with one committed pointer."""

    def __init__(
        self,
        root: str | Path,
        manifest: ProductionCaseManifest,
        policy: CheckpointGenerationPolicy,
        /,
    ):
        if not isinstance(manifest, ProductionCaseManifest) or not isinstance(
            policy, CheckpointGenerationPolicy
        ):
            raise TypeError("Checkpoint store requires manifest and policy.")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.policy = policy
        self._pointer = self.root / "committed.json"

    def _generation_path(self, generation: int) -> Path:
        return self.root / f"generation-{int(generation):08d}.phx"

    def commit(
        self,
        generation: int,
        envelope: RuntimeCheckpointEnvelope,
        /,
    ) -> Path:
        target = self._generation_path(generation)
        temporary = target.with_suffix(".tmp")
        write_runtime_checkpoint(temporary, envelope)
        os.replace(temporary, target)
        pointer_temporary = self._pointer.with_suffix(".tmp")
        pointer_temporary.write_text(
            json.dumps(
                {
                    "generation": int(generation),
                    "checkpoint": target.name,
                    "checkpoint_id": envelope.checkpoint_id,
                    "manifest_id": self.manifest.manifest_id,
                },
                sort_keys=True,
            )
            + "\n"
        )
        os.replace(pointer_temporary, self._pointer)
        descriptor = os.open(self.root, os.O_RDONLY)
        os.fsync(descriptor)
        os.close(descriptor)
        generations = sorted(self.root.glob("generation-*.phx"))
        for obsolete in generations[: -self.policy.retention]:
            obsolete.unlink()
        return target

    def latest(
        self,
        state_template: Any,
        /,
        *,
        controller_template: Any = (),
        observer_templates: Sequence[Any] = (),
        rng_template: Any = (),
    ) -> RuntimeCheckpointEnvelope:
        if not self._pointer.exists():
            raise FileNotFoundError("No committed checkpoint generation exists.")
        pointer = json.loads(self._pointer.read_text())
        if pointer["manifest_id"] != self.manifest.manifest_id:
            raise ValueError("Committed checkpoint belongs to another case manifest.")
        envelope = read_runtime_checkpoint(
            self.root / pointer["checkpoint"],
            state_template=state_template,
            mesh_id=self.manifest.topology_id,
            method_id=self.manifest.method_id,
            precision_id=self.manifest.precision_id,
            topology_epoch_id=self.manifest.geometry_layout_id,
            controller_template=controller_template,
            observer_templates=observer_templates,
            rng_template=rng_template,
        )
        if envelope.checkpoint_id != pointer["checkpoint_id"]:
            raise ValueError("Committed checkpoint pointer checksum is stale.")
        return envelope


class ProductionFailureRecord(StrictModule, NonTrainableState):
    step_index: Array
    time: Array
    category: str = eqx.field(static=True)
    detail: str = eqx.field(static=True)
    last_checkpoint_id: str = eqx.field(static=True)
    failure_id: str = eqx.field(static=True)

    def __init__(
        self,
        step_index: ArrayLike,
        time: ArrayLike,
        category: str,
        detail: str,
        last_checkpoint_id: str,
        /,
    ):
        self.step_index = jnp.asarray(step_index)
        self.time = jnp.asarray(time)
        self.category = str(category)
        self.detail = str(detail)
        self.last_checkpoint_id = str(last_checkpoint_id)
        self.failure_id = canonical_fingerprint(
            {
                "kind": "production-failure-record",
                "step": int(np.asarray(self.step_index)),
                "time": float(np.asarray(self.time)),
                "category": self.category,
                "detail": self.detail,
                "last_checkpoint": self.last_checkpoint_id,
            }
        )


class ProductionTerminalManifest(StrictModule, NonTrainableState):
    status: RunStatus = eqx.field(static=True)
    case_manifest_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    last_checkpoint_id: str = eqx.field(static=True)
    failure_id: str | None = eqx.field(static=True)
    terminal_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: RunStatus,
        case_manifest_id: str,
        run_id: str,
        last_checkpoint_id: str,
        failure_id: str | None,
        /,
    ):
        if status not in ("completed", "failed", "cancelled"):
            raise ValueError("Terminal manifest status is not terminal.")
        self.status = status
        self.case_manifest_id = str(case_manifest_id)
        self.run_id = str(run_id)
        self.last_checkpoint_id = str(last_checkpoint_id)
        self.failure_id = None if failure_id is None else str(failure_id)
        self.terminal_id = canonical_fingerprint(
            {
                "kind": "production-terminal-manifest",
                "status": status,
                "case": self.case_manifest_id,
                "run": self.run_id,
                "last_checkpoint": self.last_checkpoint_id,
                "failure": self.failure_id,
            }
        )

    def payload(self, /) -> dict[str, Any]:
        return {
            "status": self.status,
            "case_manifest_id": self.case_manifest_id,
            "run_id": self.run_id,
            "last_checkpoint_id": self.last_checkpoint_id,
            "failure_id": self.failure_id,
            "terminal_id": self.terminal_id,
        }


class ProductionRunState(StrictModule):
    step_index: Array
    time: Array
    accepted_state: Array
    controller_state: Any
    rng_state: Any
    schedule_cursor: Array
    moment_states: tuple[StreamingMomentState, ...]
    trigger_states: tuple[AcceptedStepTriggerGraphState, ...]
    output_cursor: Array
    status: RunStatus = eqx.field(static=True)
    last_checkpoint_id: str = eqx.field(static=True)


class ProductionRunResult(StrictModule):
    state: ProductionRunState
    successful: Array
    failure: ProductionFailureRecord | None
    run_id: str = eqx.field(static=True)


class ProductionRunPlan(StrictModule, NonTrainableState):
    method: AbstractFixedStepMethod
    retry_policy: RobustRetryPolicy
    step_size: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    checkpoint_interval: int = eqx.field(static=True)
    schedule: ExactTimeSchedule | None
    moments: tuple[StreamingMomentPlan, ...]
    triggers: tuple[AcceptedStepTriggerGraph, ...]
    validator: Callable = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractFixedStepMethod,
        retry_policy: RobustRetryPolicy,
        /,
        *,
        step_size: float,
        maximum_steps: int,
        checkpoint_interval: int,
        schedule: ExactTimeSchedule | None = None,
        moments: Sequence[StreamingMomentPlan] = (),
        triggers: Sequence[AcceptedStepTriggerGraph] = (),
        validator: Callable | None = None,
    ):
        step = float(step_size)
        steps = int(maximum_steps)
        interval = int(checkpoint_interval)
        moments_ = tuple(moments)
        triggers_ = tuple(triggers)
        validator_ = lambda state: (
            jnp.all(jnp.isfinite(state)) if validator is None else validator
        )
        if (
            not isinstance(method, AbstractFixedStepMethod)
            or not isinstance(retry_policy, RobustRetryPolicy)
            or not step > 0.0
            or steps <= 0
            or interval <= 0
            or (schedule is not None and not isinstance(schedule, ExactTimeSchedule))
            or any(not isinstance(value, StreamingMomentPlan) for value in moments_)
            or any(not isinstance(value, AcceptedStepTriggerGraph) for value in triggers_)
            or not callable(validator_)
        ):
            raise ValueError("Production run plan is invalid.")
        self.method = method
        self.retry_policy = retry_policy
        self.step_size = step
        self.maximum_steps = steps
        self.checkpoint_interval = interval
        self.schedule = schedule
        self.moments = moments_
        self.triggers = triggers_
        self.validator = validator_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "production-run-plan",
                "method": method.method_id,
                "retry": retry_policy.policy_id,
                "step_size": step,
                "maximum_steps": steps,
                "checkpoint_interval": interval,
                "schedule": None if schedule is None else schedule.schedule_id,
                "moments": tuple(value.plan_id for value in moments_),
                "triggers": tuple(value.graph_id for value in triggers_),
            }
        )


class PreparedProductionRun:
    def __init__(
        self,
        manifest: ProductionCaseManifest,
        plan: ProductionRunPlan,
        checkpoint_store: DurableCheckpointStore,
        /,
        *,
        publisher: ByteBoundedAsyncPublisher | None = None,
    ):
        if (
            not isinstance(manifest, ProductionCaseManifest)
            or not isinstance(plan, ProductionRunPlan)
            or not isinstance(checkpoint_store, DurableCheckpointStore)
            or checkpoint_store.manifest.manifest_id != manifest.manifest_id
        ):
            raise TypeError("Prepared production run inputs are incompatible.")
        self.manifest = manifest
        self.plan = plan
        self.checkpoint_store = checkpoint_store
        self.publisher = publisher
        self.run_id = canonical_fingerprint(
            {
                "kind": "prepared-production-run",
                "manifest": manifest.manifest_id,
                "plan": plan.plan_id,
            }
        )

    def initial_state(
        self,
        state: ArrayLike,
        /,
        *,
        time: ArrayLike = 0.0,
        controller_state: Any = (),
        rng_state: Any = (),
    ) -> ProductionRunState:
        value = jnp.asarray(state)
        return ProductionRunState(
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(time, dtype=value.real.dtype),
            value,
            controller_state,
            rng_state,
            jnp.asarray(0, dtype=jnp.int64),
            tuple(plan.initial_state(value.real.dtype) for plan in self.plan.moments),
            tuple(plan.initial_state(value.real.dtype) for plan in self.plan.triggers),
            jnp.asarray(0, dtype=jnp.int64),
            "ready",
            "",
        )

    def _envelope(self, state: ProductionRunState, /) -> RuntimeCheckpointEnvelope:
        return RuntimeCheckpointEnvelope(
            state.accepted_state,
            time=state.time,
            step_index=state.step_index,
            schedule_cursor=state.schedule_cursor,
            mesh_id=self.manifest.topology_id,
            method_id=self.manifest.method_id,
            precision_id=self.manifest.precision_id,
            topology_epoch_id=self.manifest.geometry_layout_id,
            controller_state=(
                state.controller_state,
                state.trigger_states,
                state.output_cursor,
            ),
            observer_states=state.moment_states,
            rng_state=state.rng_state,
        )

    def checkpoint(self, state: ProductionRunState, /) -> ProductionRunState:
        envelope = self._envelope(state)
        self.checkpoint_store.commit(int(np.asarray(state.step_index)), envelope)
        return ProductionRunState(
            state.step_index,
            state.time,
            state.accepted_state,
            state.controller_state,
            state.rng_state,
            state.schedule_cursor,
            state.moment_states,
            state.trigger_states,
            state.output_cursor,
            state.status,
            envelope.checkpoint_id,
        )

    def resume(self, template: ProductionRunState, /) -> ProductionRunState:
        envelope = self.checkpoint_store.latest(
            template.accepted_state,
            controller_template=(
                template.controller_state,
                template.trigger_states,
                template.output_cursor,
            ),
            observer_templates=template.moment_states,
            rng_template=template.rng_state,
        )
        controller, triggers, output_cursor = envelope.controller_state
        return ProductionRunState(
            envelope.step_index,
            envelope.time,
            envelope.state,
            controller,
            envelope.rng_state,
            envelope.schedule_cursor,
            envelope.observer_states,
            triggers,
            output_cursor,
            "ready",
            envelope.checkpoint_id,
        )

    def _commit_terminal(
        self, state: ProductionRunState, failure: ProductionFailureRecord | None, /
    ) -> ProductionTerminalManifest:
        status: RunStatus = "failed" if failure is not None else "completed"
        terminal = ProductionTerminalManifest(
            status,
            self.manifest.manifest_id,
            self.run_id,
            state.last_checkpoint_id,
            None if failure is None else failure.failure_id,
        )
        target = self.checkpoint_store.root / "terminal.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(terminal.payload(), sort_keys=True) + "\n")
        os.replace(temporary, target)
        return terminal

    def step(
        self,
        state: ProductionRunState,
        args: Any = None,
        /,
    ) -> tuple[ProductionRunState, RetriedFixedStepResult]:
        proposed_step = jnp.asarray(self.plan.step_size, dtype=state.time.dtype)
        if self.plan.schedule is not None:
            proposed_step = self.plan.schedule.clamp_step(
                state.time, proposed_step, state.schedule_cursor
            )
        result = retry_fixed_step(
            self.plan.method,
            self.plan.retry_policy,
            state.step_index,
            state.time,
            state.accepted_state,
            proposed_step,
            args,
        )
        accepted = result.successful & self.plan.validator(result.accepted_state)
        accepted_host = bool(np.asarray(accepted))
        next_time = state.time + jnp.where(accepted, result.accepted_step_size, 0.0)
        next_step = state.step_index + accepted.astype(state.step_index.dtype)
        next_state = jnp.where(accepted, result.accepted_state, state.accepted_state)
        if accepted_host:
            moment_states = tuple(
                plan.update(
                    next_time,
                    next_state,
                    moment_state,
                    weight=result.accepted_step_size,
                    args=args,
                )
                for plan, moment_state in zip(
                    self.plan.moments, state.moment_states, strict=True
                )
            )
            moment_values = tuple(moment.mean for moment in moment_states)
            trigger_states = []
            for graph, trigger_state in zip(
                self.plan.triggers, state.trigger_states, strict=True
            ):
                values = moment_values[: len(graph.triggers)]
                _fire, updated_trigger = graph.evaluate(
                    values, trigger_state, accepted=True
                )
                trigger_states.append(updated_trigger)
            schedule_cursor = (
                state.schedule_cursor
                if self.plan.schedule is None
                else self.plan.schedule.advance_cursor(next_time, state.schedule_cursor)
            )
        else:
            moment_states = state.moment_states
            trigger_states = list(state.trigger_states)
            schedule_cursor = state.schedule_cursor
        updated = ProductionRunState(
            next_step,
            next_time,
            next_state,
            state.controller_state,
            state.rng_state,
            schedule_cursor,
            moment_states,
            tuple(trigger_states),
            state.output_cursor,
            "running" if accepted_host else "failed",
            state.last_checkpoint_id,
        )
        if (
            accepted_host
            and int(np.asarray(next_step)) % self.plan.checkpoint_interval == 0
        ):
            updated = self.checkpoint(updated)
        if self.publisher is not None and accepted_host:
            event_id = canonical_fingerprint(
                {
                    "kind": "production-output-event",
                    "run": self.run_id,
                    "step": int(np.asarray(next_step)),
                }
            )
            self.publisher.publish(event_id, next_state)
            updated = eqx.tree_at(
                lambda value: value.output_cursor,
                updated,
                updated.output_cursor + 1,
            )
        return updated, result

    def run(
        self,
        state: ProductionRunState,
        args: Any = None,
        /,
    ) -> ProductionRunResult:
        current = state
        failure = None
        for _iteration in range(self.plan.maximum_steps):
            if current.status == "failed":
                break
            current, step_result = self.step(current, args)
            if not bool(np.asarray(step_result.successful)):
                failure = ProductionFailureRecord(
                    current.step_index,
                    current.time,
                    "step-rejected",
                    "All robust retry attempts failed.",
                    current.last_checkpoint_id,
                )
                break
        if failure is None and current.status != "failed":
            current = ProductionRunState(
                current.step_index,
                current.time,
                current.accepted_state,
                current.controller_state,
                current.rng_state,
                current.schedule_cursor,
                current.moment_states,
                current.trigger_states,
                current.output_cursor,
                "completed",
                current.last_checkpoint_id,
            )
        current = self.checkpoint(current)
        self._commit_terminal(current, failure)
        if self.publisher is not None:
            self.publisher.drain()
        return ProductionRunResult(
            current,
            jnp.asarray(failure is None),
            failure,
            self.run_id,
        )


__all__ = [
    "CheckpointGenerationPolicy",
    "ProductionCaseManifest",
    "ProductionFailureRecord",
    "ProductionRunPlan",
    "ProductionRunResult",
    "ProductionRunState",
    "ProductionTerminalManifest",
    "DurableCheckpointStore",
    "PreparedProductionRun",
]
