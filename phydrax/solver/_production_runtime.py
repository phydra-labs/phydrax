#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ._fixed_step import (
    _canonical_structured_state,
    _state_dtype,
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
    RuntimeCheckpointEncodingPlan,
    RuntimeCheckpointEnvelope,
    StreamingMomentPlan,
    StreamingMomentState,
    write_runtime_checkpoint,
)


RunStatus = Literal["ready", "running", "completed", "failed", "cancelled"]
ProductionTriggerAction = Literal["checkpoint", "publish", "stop"]


def _finite_array_tree(state: Any, /) -> Array:
    leaves = jax.tree.leaves(state)
    if not leaves:
        raise ValueError("Production state must contain array leaves.")
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))


def _canonical_auxiliary_tree(tree: Any, role: str, /) -> Any:
    leaves, treedef = jax.tree.flatten(tree)
    if any(not eqx.is_array(leaf) for leaf in leaves):
        raise TypeError(f"Production {role} must be an array-only PyTree.")
    return jax.tree.unflatten(treedef, tuple(jnp.asarray(leaf) for leaf in leaves))


def _fsync_directory(path: Path, /) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: dict[str, Any], /) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


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
    """Crash-consistent, monotone checkpoint generations and one durable pointer."""

    _POINTER_KEYS = frozenset(
        {
            "generation",
            "checkpoint",
            "checkpoint_id",
            "manifest_id",
            "runtime_id",
            "encoding_id",
        }
    )

    def __init__(
        self,
        root: str | Path,
        manifest: ProductionCaseManifest,
        policy: CheckpointGenerationPolicy,
        /,
        *,
        encoding_plan: RuntimeCheckpointEncodingPlan | None = None,
    ):
        if not isinstance(manifest, ProductionCaseManifest) or not isinstance(
            policy, CheckpointGenerationPolicy
        ):
            raise TypeError("Checkpoint store requires manifest and policy.")
        encoding = (
            RuntimeCheckpointEncodingPlan() if encoding_plan is None else encoding_plan
        )
        if not isinstance(encoding, RuntimeCheckpointEncodingPlan):
            raise TypeError(
                "encoding_plan must be RuntimeCheckpointEncodingPlan or None."
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        _fsync_directory(self.root.parent)
        self.manifest = manifest
        self.policy = policy
        self.encoding_plan = encoding
        self._pointer = self.root / "committed.json"

    def _generation_path(self, generation: int) -> Path:
        return self.root / f"generation-{int(generation):08d}.phx"

    def _read_pointer(self) -> dict[str, Any]:
        if not self._pointer.exists():
            raise FileNotFoundError("No committed checkpoint generation exists.")
        payload = json.loads(self._pointer.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != self._POINTER_KEYS:
            raise ValueError("Committed checkpoint pointer schema is corrupt.")
        generation = payload["generation"]
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("Committed checkpoint generation is corrupt.")
        expected_name = self._generation_path(generation).name
        if payload["checkpoint"] != expected_name:
            raise ValueError("Committed checkpoint pointer path is stale or unsafe.")
        for name in (
            "checkpoint_id",
            "manifest_id",
            "runtime_id",
            "encoding_id",
        ):
            if not isinstance(payload[name], str) or not payload[name]:
                raise ValueError("Committed checkpoint pointer identity is corrupt.")
        return payload

    def commit(
        self,
        generation: int,
        envelope: RuntimeCheckpointEnvelope,
        /,
    ) -> Path:
        if isinstance(generation, bool):
            raise TypeError("Checkpoint generation must be an integer.")
        generation_ = int(generation)
        if generation_ < 0 or not isinstance(envelope, RuntimeCheckpointEnvelope):
            raise ValueError("Checkpoint generation or envelope is invalid.")
        if (
            envelope.mesh_id != self.manifest.topology_id
            or envelope.method_id != self.manifest.method_id
            or envelope.precision_id != self.manifest.precision_id
            or envelope.topology_epoch_id != self.manifest.geometry_layout_id
            or envelope.encoding_plan.encoding_id != self.encoding_plan.encoding_id
            or int(np.asarray(envelope.step_index)) != generation_
        ):
            raise ValueError("Checkpoint envelope does not belong to this store.")
        if self._pointer.exists():
            current = self._read_pointer()
            if generation_ <= current["generation"]:
                raise ValueError("Checkpoint generations must increase monotonically.")
        target = self._generation_path(generation_)
        if target.exists():
            raise FileExistsError(
                "Checkpoint generation already exists and is immutable."
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=self.root
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        try:
            write_runtime_checkpoint(temporary, envelope)
            os.replace(temporary, target)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)
        pointer = {
            "generation": generation_,
            "checkpoint": target.name,
            "checkpoint_id": envelope.checkpoint_id,
            "manifest_id": self.manifest.manifest_id,
            "runtime_id": envelope.runtime_id,
            "encoding_id": self.encoding_plan.encoding_id,
        }
        _write_json_atomic(self._pointer, pointer)
        generations = sorted(self.root.glob("generation-*.phx"))
        removed = False
        for obsolete in generations[: -self.policy.retention]:
            obsolete.unlink()
            removed = True
        if removed:
            _fsync_directory(self.root)
        return target

    def latest(
        self,
        state_template: Any,
        /,
        *,
        controller_template: Any = (),
        observer_templates: Sequence[Any] = (),
        rng_template: Any = (),
        runtime_id: str | None = None,
    ) -> RuntimeCheckpointEnvelope:
        pointer = self._read_pointer()
        if pointer["manifest_id"] != self.manifest.manifest_id:
            raise ValueError("Committed checkpoint belongs to another case manifest.")
        if pointer["encoding_id"] != self.encoding_plan.encoding_id:
            raise ValueError("Committed checkpoint encoding identity changed.")
        if runtime_id is not None and pointer["runtime_id"] != str(runtime_id):
            raise ValueError("Committed checkpoint belongs to another prepared runtime.")
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
            runtime_id=pointer["runtime_id"],
            encoding_plan=self.encoding_plan,
        )
        if (
            envelope.checkpoint_id != pointer["checkpoint_id"]
            or int(np.asarray(envelope.step_index)) != pointer["generation"]
        ):
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


class ProductionTriggerBinding(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    graph: AcceptedStepTriggerGraph
    moment_indices: tuple[int, ...] = eqx.field(static=True)
    moment_components: tuple[int, ...] = eqx.field(static=True)
    action: ProductionTriggerAction = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        graph: AcceptedStepTriggerGraph,
        moment_indices: Sequence[int],
        action: ProductionTriggerAction,
        action_id: str,
        /,
        *,
        moment_components: Sequence[int] = (),
    ):
        name_ = str(name)
        indices = tuple(int(value) for value in moment_indices)
        components = tuple(int(value) for value in moment_components)
        action_id_ = str(action_id)
        if (
            not name_
            or not isinstance(graph, AcceptedStepTriggerGraph)
            or len(indices) != len(graph.triggers)
            or any(value < 0 for value in indices)
            or (components and len(components) != len(indices))
            or any(value < 0 for value in components)
            or action not in ("checkpoint", "publish", "stop")
            or not action_id_
        ):
            raise ValueError("Production trigger binding is invalid.")
        self.name = name_
        self.graph = graph
        self.moment_indices = indices
        self.moment_components = components
        self.action = action
        self.action_id = action_id_
        self.binding_id = canonical_fingerprint(
            {
                "kind": "production-trigger-binding",
                "name": name_,
                "graph": graph.graph_id,
                "moment_indices": indices,
                "moment_components": components,
                "action": action,
                "action_id": action_id_,
            }
        )


class ProductionRunState(StrictModule):
    step_index: Array
    time: Array
    accepted_state: PyTree[Array]
    controller_state: Any
    rng_state: Any
    schedule_cursor: Array
    moment_states: tuple[StreamingMomentState, ...]
    trigger_states: tuple[AcceptedStepTriggerGraphState, ...]
    output_cursor: Array
    status: RunStatus = eqx.field(static=True)
    last_checkpoint_id: str = eqx.field(static=True)


def _replace_run_metadata(
    state: ProductionRunState,
    /,
    *,
    status: RunStatus | None = None,
    last_checkpoint_id: str | None = None,
) -> ProductionRunState:
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
        state.status if status is None else status,
        state.last_checkpoint_id if last_checkpoint_id is None else last_checkpoint_id,
    )


class ProductionRunResult(StrictModule):
    state: ProductionRunState
    successful: Array
    failure: ProductionFailureRecord | None
    run_id: str = eqx.field(static=True)


class ProductionRunPlan(StrictModule, NonTrainableState):
    method: AbstractFixedStepMethod
    retry_policy: RobustRetryPolicy
    output_schedule: ExactTimeSchedule | None
    moments: tuple[StreamingMomentPlan, ...]
    trigger_bindings: tuple[ProductionTriggerBinding, ...]
    validator: Callable = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    end_time: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    checkpoint_interval: int = eqx.field(static=True)
    segment_steps: int = eqx.field(static=True)
    validator_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractFixedStepMethod,
        retry_policy: RobustRetryPolicy,
        /,
        *,
        step_size: float,
        end_time: float,
        maximum_steps: int,
        checkpoint_interval: int,
        segment_steps: int = 32,
        output_schedule: ExactTimeSchedule | None = None,
        moments: Sequence[StreamingMomentPlan] = (),
        trigger_bindings: Sequence[ProductionTriggerBinding] = (),
        validator: Callable | None = None,
        validator_id: str | None = None,
    ):
        step = float(step_size)
        end = float(end_time)
        steps = int(maximum_steps)
        interval = int(checkpoint_interval)
        segment = int(segment_steps)
        moments_ = tuple(moments)
        bindings = tuple(trigger_bindings)
        if validator is None:
            if validator_id is not None:
                raise ValueError("validator_id requires a supplied validator.")
            validator_ = _finite_array_tree
            validator_identifier = "production-validator:finite-array-tree"
        else:
            validator_ = validator
            validator_identifier = "" if validator_id is None else str(validator_id)
            if not callable(validator_) or not validator_identifier:
                raise ValueError("A supplied validator requires a stable validator_id.")
        if (
            not isinstance(method, AbstractFixedStepMethod)
            or not isinstance(retry_policy, RobustRetryPolicy)
            or not math.isfinite(step)
            or step <= 0.0
            or not math.isfinite(end)
            or steps <= 0
            or interval <= 0
            or segment <= 0
            or (
                output_schedule is not None
                and not isinstance(output_schedule, ExactTimeSchedule)
            )
            or any(not isinstance(value, StreamingMomentPlan) for value in moments_)
            or any(not isinstance(value, ProductionTriggerBinding) for value in bindings)
        ):
            raise ValueError("Production run plan is invalid.")
        if any(
            index >= len(moments_)
            for binding in bindings
            for index in binding.moment_indices
        ):
            raise ValueError(
                "Trigger bindings reference an unavailable streaming moment."
            )
        for binding in bindings:
            if binding.moment_components:
                for index, component in zip(
                    binding.moment_indices,
                    binding.moment_components,
                    strict=True,
                ):
                    value_size = int(np.prod(moments_[index].value_shape, dtype=int))
                    if component >= max(value_size, 1):
                        raise ValueError(
                            "Trigger component is outside its streaming moment."
                        )
            elif any(moments_[index].value_shape for index in binding.moment_indices):
                raise ValueError(
                    "Vector streaming moments require explicit trigger components."
                )
        if (
            output_schedule is not None
            and float(np.asarray(output_schedule.targets[-1]))
            > end + output_schedule.tolerance
        ):
            raise ValueError("Production output targets cannot exceed end_time.")
        required = method.required_step_size
        if required is not None and (
            not math.isfinite(float(required))
            or not np.isclose(step, float(required), rtol=0.0, atol=0.0)
        ):
            raise ValueError("Production step_size is incompatible with the method.")
        if retry_policy.maximum_retries and not method.allows_step_reduction:
            raise ValueError("Production retry policy requires forbidden step reduction.")
        self.method = method
        self.retry_policy = retry_policy
        self.output_schedule = output_schedule
        self.moments = moments_
        self.trigger_bindings = bindings
        self.validator = validator_
        self.step_size = step
        self.end_time = end
        self.maximum_steps = steps
        self.checkpoint_interval = interval
        self.segment_steps = segment
        self.validator_id = validator_identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "production-run-plan",
                "method": method.method_id,
                "retry": retry_policy.policy_id,
                "step_size": step,
                "end_time": end,
                "maximum_steps": steps,
                "checkpoint_interval": interval,
                "segment_steps": segment,
                "output_schedule": None
                if output_schedule is None
                else output_schedule.schedule_id,
                "moments": tuple(value.plan_id for value in moments_),
                "trigger_bindings": tuple(value.binding_id for value in bindings),
                "validator": validator_identifier,
            }
        )


class _SegmentState(StrictModule):
    step_index: Array
    time: Array
    accepted_state: PyTree[Array]
    schedule_cursor: Array
    moment_states: tuple[StreamingMomentState, ...]
    trigger_states: tuple[AcceptedStepTriggerGraphState, ...]
    output_cursor: Array
    running: Array
    stop_requested: Array


class _SegmentRecord(StrictModule):
    state: _SegmentState
    result: RetriedFixedStepResult
    attempted: Array
    accepted: Array
    method_successful: Array
    schedule_cursor_before: Array
    output_due: Array
    checkpoint_due: Array
    trigger_fires: tuple[Array, ...]


class PreparedProductionRun:
    """Bound production runtime with bounded compiled scan segments."""

    def __init__(
        self,
        manifest: ProductionCaseManifest,
        plan: ProductionRunPlan,
        checkpoint_store: DurableCheckpointStore,
        /,
        *,
        args: Any = None,
        args_id: str | None = None,
        publisher: ByteBoundedAsyncPublisher | None = None,
    ):
        if (
            not isinstance(manifest, ProductionCaseManifest)
            or not isinstance(plan, ProductionRunPlan)
            or not isinstance(checkpoint_store, DurableCheckpointStore)
            or checkpoint_store.manifest.manifest_id != manifest.manifest_id
        ):
            raise TypeError("Prepared production run inputs are incompatible.")
        if manifest.method_id != plan.method.method_id:
            raise ValueError(
                "Production manifest method identity does not match the plan."
            )
        if publisher is not None and not isinstance(publisher, ByteBoundedAsyncPublisher):
            raise TypeError("publisher must be ByteBoundedAsyncPublisher or None.")
        if publisher is None and (
            plan.output_schedule is not None
            or any(value.action == "publish" for value in plan.trigger_bindings)
        ):
            raise ValueError("Scheduled or triggered publication requires a publisher.")
        if args_id is None:
            if args is not None:
                raise ValueError("Bound production args require a stable args_id.")
            args_identifier = "production-args:none"
        else:
            args_identifier = str(args_id)
            if not args_identifier:
                raise ValueError("args_id must be nonempty when supplied.")
        self.manifest = manifest
        self.plan = plan
        self.checkpoint_store = checkpoint_store
        self.args = args
        self.args_id = args_identifier
        self.publisher = publisher
        self.run_id = canonical_fingerprint(
            {
                "kind": "prepared-production-run",
                "manifest": manifest.manifest_id,
                "plan": plan.plan_id,
                "checkpoint_policy": checkpoint_store.policy.policy_id,
                "checkpoint_encoding": checkpoint_store.encoding_plan.encoding_id,
                "args": args_identifier,
            }
        )
        self._compiled_segment = self._compile_segment(plan.segment_steps)
        self._compiled_one_step = self._compile_segment(1)

    def _compile_segment(self, length: int, /):
        plan = self.plan
        args = self.args
        retry_decision_id = canonical_fingerprint(
            {
                "kind": "retried-fixed-step-decision",
                "method": plan.method.method_id,
                "retry_policy": plan.retry_policy.policy_id,
            }
        )
        attempt_count = plan.retry_policy.maximum_retries + 1

        def scan_step(carry: _SegmentState, unused: None):
            del unused
            tolerance = jnp.asarray(
                32.0 * jnp.finfo(carry.time.dtype).eps, dtype=carry.time.dtype
            )
            active = (
                carry.running
                & ~carry.stop_requested
                & (carry.step_index < plan.maximum_steps)
                & (carry.time < plan.end_time - tolerance)
            )
            proposed_step = jnp.minimum(
                jnp.asarray(plan.step_size, dtype=carry.time.dtype),
                jnp.asarray(plan.end_time, dtype=carry.time.dtype) - carry.time,
            )
            if plan.output_schedule is not None:
                proposed_step = plan.output_schedule.clamp_step(
                    carry.time, proposed_step, carry.schedule_cursor
                )

            def advance(_: None) -> RetriedFixedStepResult:
                return retry_fixed_step(
                    plan.method,
                    plan.retry_policy,
                    carry.step_index,
                    carry.time,
                    carry.accepted_state,
                    proposed_step,
                    args,
                )

            def inactive(_: None) -> RetriedFixedStepResult:
                return RetriedFixedStepResult(
                    carry.accepted_state,
                    carry.accepted_state,
                    jnp.asarray(True),
                    jnp.zeros((), dtype=carry.time.dtype),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.zeros((attempt_count,), dtype=carry.time.dtype),
                    retry_decision_id,
                )

            result = jax.lax.cond(active, advance, inactive, operand=None)
            valid = jnp.asarray(plan.validator(result.accepted_state))
            if valid.shape != () or valid.dtype != jnp.dtype(bool):
                raise TypeError(
                    "Production validators must return a scalar Boolean array."
                )
            accepted = active & result.successful & valid
            next_time = carry.time + jnp.where(accepted, result.accepted_step_size, 0.0)
            next_step = carry.step_index + accepted.astype(carry.step_index.dtype)
            next_state = tree_where(accepted, result.accepted_state, carry.accepted_state)
            proposed_moments = tuple(
                moment.update(
                    next_time,
                    next_state,
                    moment_state,
                    previous_time=carry.time,
                    args=args,
                )
                for moment, moment_state in zip(
                    plan.moments, carry.moment_states, strict=True
                )
            )
            moment_states = tuple(
                tree_where(accepted, proposed, current)
                for proposed, current in zip(
                    proposed_moments, carry.moment_states, strict=True
                )
            )
            trigger_states = []
            trigger_fires = []
            for binding, trigger_state in zip(
                plan.trigger_bindings, carry.trigger_states, strict=True
            ):
                if binding.moment_components:
                    values = tuple(
                        moment_states[index].mean.reshape((-1,))[component]
                        for index, component in zip(
                            binding.moment_indices,
                            binding.moment_components,
                            strict=True,
                        )
                    )
                else:
                    values = tuple(
                        moment_states[index].mean.reshape(())
                        for index in binding.moment_indices
                    )
                fire, proposed_trigger = binding.graph.evaluate(
                    values, trigger_state, accepted=accepted
                )
                trigger_fires.append(fire)
                trigger_states.append(
                    tree_where(accepted, proposed_trigger, trigger_state)
                )
            if plan.output_schedule is None:
                schedule_cursor = carry.schedule_cursor
                output_due = jnp.asarray(False)
            else:
                proposed_cursor = plan.output_schedule.advance_cursor(
                    next_time, carry.schedule_cursor
                )
                schedule_cursor = jnp.where(
                    accepted, proposed_cursor, carry.schedule_cursor
                )
                output_due = accepted & (schedule_cursor > carry.schedule_cursor)
            stop_fire = jnp.asarray(False)
            publish_count = jnp.asarray(0, dtype=carry.output_cursor.dtype)
            for binding, fire in zip(plan.trigger_bindings, trigger_fires, strict=True):
                if binding.action == "stop":
                    stop_fire = stop_fire | fire
                elif binding.action == "publish":
                    publish_count = publish_count + fire.astype(publish_count.dtype)
            output_increment = (schedule_cursor - carry.schedule_cursor).astype(
                carry.output_cursor.dtype
            ) + publish_count
            running = carry.running & ~(active & ~accepted)
            next_carry = _SegmentState(
                next_step,
                next_time,
                next_state,
                schedule_cursor,
                moment_states,
                tuple(trigger_states),
                carry.output_cursor + output_increment,
                running,
                carry.stop_requested | stop_fire,
            )
            checkpoint_due = accepted & (next_step % plan.checkpoint_interval == 0)
            record = _SegmentRecord(
                next_carry,
                result,
                active,
                accepted,
                result.successful,
                carry.schedule_cursor,
                output_due,
                checkpoint_due,
                tuple(trigger_fires),
            )
            return next_carry, record

        @jax.jit
        def execute(initial: _SegmentState):
            return jax.lax.scan(scan_step, initial, xs=None, length=length)

        return execute

    def _preflight_horizon(self, time: ArrayLike, step_index: ArrayLike, /) -> None:
        start = float(np.asarray(time))
        step = int(np.asarray(step_index))
        tolerance = 32.0 * np.finfo(np.asarray(time).dtype).eps
        if not math.isfinite(start) or start > self.plan.end_time + tolerance:
            raise ValueError("Production start time exceeds the absolute end_time.")
        if step < 0 or step > self.plan.maximum_steps:
            raise ValueError("Production step index exceeds absolute step capacity.")
        remaining = max(self.plan.end_time - start, 0.0)
        nominal_steps = int(
            math.ceil(max(remaining - tolerance, 0.0) / self.plan.step_size)
        )
        if step + nominal_steps > self.plan.maximum_steps:
            raise ValueError(
                "Absolute step capacity cannot reach end_time from this state."
            )
        if not self.plan.method.allows_step_reduction:
            points = [self.plan.end_time]
            if self.plan.output_schedule is not None:
                points.extend(
                    float(value)
                    for value in np.asarray(self.plan.output_schedule.targets)
                    if float(value) > start + self.plan.output_schedule.tolerance
                )
            for point in points:
                raw = (point - start) / self.plan.step_size
                if raw < -tolerance or not np.isclose(
                    raw, round(raw), rtol=1.0e-12, atol=1.0e-12
                ):
                    raise ValueError(
                        "Method step-reduction constraints are incompatible with the runtime horizon."
                    )

    def initial_state(
        self,
        state: Any,
        /,
        *,
        time: ArrayLike = 0.0,
        controller_state: Any = (),
        rng_state: Any = (),
    ) -> ProductionRunState:
        value = _canonical_structured_state(state)
        dtype = _state_dtype(value)
        if str(jnp.dtype(dtype)) != self.manifest.dtype:
            raise ValueError("Initial state precision does not match the case manifest.")
        time_dtype = jnp.asarray(0, dtype=dtype).real.dtype
        time_ = jnp.asarray(time, dtype=time_dtype)
        if time_.shape != ():
            raise ValueError("Production start time must be scalar.")
        self._preflight_horizon(time_, jnp.asarray(0, dtype=jnp.int64))
        schedule_cursor = (
            jnp.asarray(0, dtype=jnp.int64)
            if self.plan.output_schedule is None
            else self.plan.output_schedule.initial_cursor(time_).astype(jnp.int64)
        )
        return ProductionRunState(
            jnp.asarray(0, dtype=jnp.int64),
            time_,
            value,
            _canonical_auxiliary_tree(controller_state, "controller state"),
            _canonical_auxiliary_tree(rng_state, "RNG state"),
            schedule_cursor,
            tuple(moment.initial_state(time_dtype) for moment in self.plan.moments),
            tuple(
                binding.graph.initial_state(time_dtype)
                for binding in self.plan.trigger_bindings
            ),
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
            runtime_id=self.run_id,
            encoding_plan=self.checkpoint_store.encoding_plan,
        )

    def checkpoint(self, state: ProductionRunState, /) -> ProductionRunState:
        envelope = self._envelope(state)
        if state.last_checkpoint_id == envelope.checkpoint_id:
            return state
        self.checkpoint_store.commit(int(np.asarray(state.step_index)), envelope)
        return _replace_run_metadata(state, last_checkpoint_id=envelope.checkpoint_id)

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
            runtime_id=self.run_id,
        )
        controller, triggers, output_cursor = envelope.controller_state
        self._preflight_horizon(envelope.time, envelope.step_index)
        expected_cursor = (
            0
            if self.plan.output_schedule is None
            else int(np.asarray(self.plan.output_schedule.initial_cursor(envelope.time)))
        )
        if int(np.asarray(envelope.schedule_cursor)) != expected_cursor:
            raise ValueError("Checkpoint output schedule cursor is stale.")
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
        terminal = ProductionTerminalManifest(
            state.status,
            self.manifest.manifest_id,
            self.run_id,
            state.last_checkpoint_id,
            None if failure is None else failure.failure_id,
        )
        _write_json_atomic(
            self.checkpoint_store.root / "terminal.json", terminal.payload()
        )
        return terminal

    @staticmethod
    def _index_tree(tree: Any, index: int, /) -> Any:
        return jax.tree.map(lambda leaf: np.asarray(leaf)[index], tree)

    def _segment_initial(self, state: ProductionRunState, /) -> _SegmentState:
        return _SegmentState(
            state.step_index,
            state.time,
            state.accepted_state,
            state.schedule_cursor,
            state.moment_states,
            state.trigger_states,
            state.output_cursor,
            jnp.asarray(True),
            jnp.asarray(False),
        )

    def _production_state(
        self,
        segment: _SegmentState,
        source: ProductionRunState,
        status: RunStatus,
        last_checkpoint_id: str,
        /,
    ) -> ProductionRunState:
        return ProductionRunState(
            segment.step_index,
            segment.time,
            segment.accepted_state,
            source.controller_state,
            source.rng_state,
            segment.schedule_cursor,
            segment.moment_states,
            segment.trigger_states,
            segment.output_cursor,
            status,
            last_checkpoint_id,
        )

    def _publish(self, event_id: str, state: PyTree[Array], /) -> str | None:
        if self.publisher is None:
            return "No publisher is bound."
        try:
            self.publisher.publish(event_id, state)
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return None

    def _drain_outputs(self, /) -> str | None:
        if self.publisher is None:
            return None
        try:
            self.publisher.drain()
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return None

    def _process_segment(
        self,
        source: ProductionRunState,
        final_segment: _SegmentState,
        records: _SegmentRecord,
        /,
    ) -> tuple[ProductionRunState, ProductionFailureRecord | None]:
        attempted = np.asarray(records.attempted, dtype=bool)
        accepted = np.asarray(records.accepted, dtype=bool)
        method_successful = np.asarray(records.method_successful, dtype=bool)
        output_due = np.asarray(records.output_due, dtype=bool)
        checkpoint_due = np.asarray(records.checkpoint_due, dtype=bool)
        trigger_fires = tuple(
            np.asarray(value, dtype=bool) for value in records.trigger_fires
        )
        last_checkpoint = source.last_checkpoint_id
        for index in np.flatnonzero(attempted):
            snapshot_segment = self._index_tree(records.state, int(index))
            snapshot = self._production_state(
                snapshot_segment, source, "running", last_checkpoint
            )
            if not accepted[index]:
                category = (
                    "state-invalid" if method_successful[index] else "step-rejected"
                )
                detail = (
                    "The accepted-state validator rejected the candidate."
                    if method_successful[index]
                    else "All robust retry attempts failed."
                )
                failed = _replace_run_metadata(snapshot, status="failed")
                return failed, ProductionFailureRecord(
                    failed.step_index,
                    failed.time,
                    category,
                    detail,
                    last_checkpoint,
                )
            if output_due[index]:
                before = int(np.asarray(records.schedule_cursor_before)[index])
                after = int(np.asarray(snapshot.schedule_cursor))
                for cursor in range(before, after):
                    event_id = canonical_fingerprint(
                        {
                            "kind": "scheduled-production-output",
                            "run": self.run_id,
                            "schedule": self.plan.output_schedule.schedule_id,
                            "cursor": cursor,
                        }
                    )
                    detail = self._publish(event_id, snapshot.accepted_state)
                    if detail is not None:
                        failed = _replace_run_metadata(snapshot, status="failed")
                        return failed, ProductionFailureRecord(
                            failed.step_index,
                            failed.time,
                            "output-failed",
                            detail,
                            last_checkpoint,
                        )
            trigger_checkpoint = False
            for binding_index, binding in enumerate(self.plan.trigger_bindings):
                if not trigger_fires[binding_index][index]:
                    continue
                if binding.action == "checkpoint":
                    trigger_checkpoint = True
                elif binding.action == "publish":
                    fire_count = int(
                        np.asarray(snapshot.trigger_states[binding_index].fire_count)
                    )
                    event_id = canonical_fingerprint(
                        {
                            "kind": "triggered-production-output",
                            "run": self.run_id,
                            "binding": binding.binding_id,
                            "action": binding.action_id,
                            "fire_count": fire_count,
                        }
                    )
                    detail = self._publish(event_id, snapshot.accepted_state)
                    if detail is not None:
                        failed = _replace_run_metadata(snapshot, status="failed")
                        return failed, ProductionFailureRecord(
                            failed.step_index,
                            failed.time,
                            "output-failed",
                            detail,
                            last_checkpoint,
                        )
            if checkpoint_due[index] or trigger_checkpoint:
                detail = self._drain_outputs()
                if detail is not None:
                    failed = _replace_run_metadata(snapshot, status="failed")
                    return failed, ProductionFailureRecord(
                        failed.step_index,
                        failed.time,
                        "output-failed",
                        detail,
                        last_checkpoint,
                    )
                snapshot = self.checkpoint(snapshot)
                last_checkpoint = snapshot.last_checkpoint_id
        status: RunStatus
        tolerance = 32.0 * np.finfo(np.asarray(final_segment.time).dtype).eps
        if not bool(np.asarray(final_segment.running)):
            status = "failed"
        elif bool(np.asarray(final_segment.stop_requested)):
            status = "cancelled"
        elif float(np.asarray(final_segment.time)) >= self.plan.end_time - tolerance:
            status = "completed"
        elif int(np.asarray(final_segment.step_index)) >= self.plan.maximum_steps:
            status = "failed"
        else:
            status = "running"
        current = self._production_state(final_segment, source, status, last_checkpoint)
        if status == "failed":
            return current, ProductionFailureRecord(
                current.step_index,
                current.time,
                "step-capacity-exhausted",
                "Absolute step capacity was exhausted before end_time.",
                last_checkpoint,
            )
        return current, None

    def _execute(
        self, state: ProductionRunState, *, one_step: bool
    ) -> tuple[ProductionRunState, ProductionFailureRecord | None, _SegmentRecord]:
        if state.status not in ("ready", "running"):
            raise ValueError("Only ready or running production state can advance.")
        executor = self._compiled_one_step if one_step else self._compiled_segment
        final_segment, records = executor(self._segment_initial(state))
        final_host, records_host = jax.device_get((final_segment, records))
        current, failure = self._process_segment(state, final_host, records_host)
        return current, failure, records_host

    def step(
        self, state: ProductionRunState, /
    ) -> tuple[ProductionRunState, RetriedFixedStepResult]:
        current, _failure, records = self._execute(state, one_step=True)
        attempted = np.flatnonzero(np.asarray(records.attempted, dtype=bool))
        if attempted.size != 1:
            raise ValueError("Production state has already reached its horizon.")
        return current, self._index_tree(records.result, int(attempted[0]))

    def run(self, state: ProductionRunState, /) -> ProductionRunResult:
        current = state
        failure = None
        while current.status in ("ready", "running"):
            current, failure, _records = self._execute(current, one_step=False)
            if failure is not None or current.status in (
                "completed",
                "failed",
                "cancelled",
            ):
                break
        drain_detail = self._drain_outputs()
        publication_failed = failure is not None and failure.category == "output-failed"
        if drain_detail is not None:
            prior_detail = (
                ""
                if failure is None
                else f" Prior failure {failure.category}: {failure.detail}"
            )
            current = _replace_run_metadata(current, status="failed")
            failure = ProductionFailureRecord(
                current.step_index,
                current.time,
                "output-failed",
                f"{drain_detail}{prior_detail}",
                current.last_checkpoint_id,
            )
            publication_failed = True
        if not publication_failed:
            current = self.checkpoint(current)
        self._commit_terminal(current, failure)
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
    "ProductionTriggerAction",
    "ProductionTriggerBinding",
    "DurableCheckpointStore",
    "PreparedProductionRun",
]
