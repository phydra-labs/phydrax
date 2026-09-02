#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class RuntimeCheckpointEnvelope(StrictModule):
    state: Any
    controller_state: Any
    observer_states: tuple[Any, ...]
    rng_state: Any
    time: Array
    step_index: Array
    schedule_cursor: Array
    mesh_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    topology_epoch_id: str = eqx.field(static=True)
    partition_id: str | None = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: Any,
        /,
        *,
        time: ArrayLike,
        step_index: ArrayLike,
        schedule_cursor: ArrayLike,
        mesh_id: str,
        method_id: str,
        precision_id: str,
        topology_epoch_id: str,
        controller_state: Any = (),
        observer_states: Sequence[Any] = (),
        rng_state: Any = (),
        partition_id: str | None = None,
    ):
        time_ = jnp.asarray(time)
        step = jnp.asarray(step_index)
        cursor = jnp.asarray(schedule_cursor)
        if (
            time_.shape != ()
            or not jnp.issubdtype(time_.dtype, jnp.inexact)
            or step.shape != ()
            or step.dtype.kind not in "iu"
            or cursor.shape != ()
            or cursor.dtype.kind not in "iu"
        ):
            raise ValueError("Checkpoint time, step, and schedule cursor are invalid.")
        identifiers = tuple(
            str(value) for value in (mesh_id, method_id, precision_id, topology_epoch_id)
        )
        if any(not value for value in identifiers):
            raise ValueError("Checkpoint compatibility identities must be nonempty.")
        partition = None if partition_id is None else str(partition_id)
        if partition == "":
            raise ValueError("partition_id must be nonempty when supplied.")
        observers = tuple(observer_states)
        self.state = state
        self.controller_state = controller_state
        self.observer_states = observers
        self.rng_state = rng_state
        self.time = time_
        self.step_index = step.astype(jnp.int64)
        self.schedule_cursor = cursor.astype(jnp.int64)
        self.mesh_id, self.method_id, self.precision_id, self.topology_epoch_id = (
            identifiers
        )
        self.partition_id = partition
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "runtime-checkpoint-envelope",
                "mesh": self.mesh_id,
                "method": self.method_id,
                "precision": self.precision_id,
                "topology_epoch": self.topology_epoch_id,
                "partition": partition,
                "time": float(np.asarray(time_)),
                "step": int(np.asarray(step)),
                "schedule_cursor": int(np.asarray(cursor)),
                "observer_count": len(observers),
            }
        )


def write_runtime_checkpoint(
    path: str | Path, envelope: RuntimeCheckpointEnvelope, /
) -> Path:
    if not isinstance(envelope, RuntimeCheckpointEnvelope):
        raise TypeError("envelope must be RuntimeCheckpointEnvelope.")
    arrays: dict[str, object] = {
        "runtime/time": np.asarray(envelope.time),
        "runtime/step_index": np.asarray(envelope.step_index),
        "runtime/schedule_cursor": np.asarray(envelope.schedule_cursor),
    }
    state_spec = pack_array_tree("state", envelope.state, arrays)
    controller_spec = pack_array_tree("controller", envelope.controller_state, arrays)
    rng_spec = pack_array_tree("rng", envelope.rng_state, arrays)
    observer_specs = [
        pack_array_tree(f"observer/{index:04d}", value, arrays)
        for index, value in enumerate(envelope.observer_states)
    ]
    manifest = {
        "kind": "runtime-checkpoint",
        "checkpoint_id": envelope.checkpoint_id,
        "mesh_id": envelope.mesh_id,
        "method_id": envelope.method_id,
        "precision_id": envelope.precision_id,
        "topology_epoch_id": envelope.topology_epoch_id,
        "partition_id": envelope.partition_id,
        "state": state_spec,
        "controller": controller_spec,
        "rng": rng_spec,
        "observers": observer_specs,
    }
    return write_array_archive(path, manifest=manifest, arrays=arrays)


def read_runtime_checkpoint(
    path: str | Path,
    /,
    *,
    state_template: Any,
    mesh_id: str,
    method_id: str,
    precision_id: str,
    topology_epoch_id: str,
    controller_template: Any = (),
    observer_templates: Sequence[Any] = (),
    rng_template: Any = (),
    partition_id: str | None = None,
) -> RuntimeCheckpointEnvelope:
    manifest, arrays = read_array_archive(path)
    expected = {
        "kind": "runtime-checkpoint",
        "mesh_id": str(mesh_id),
        "method_id": str(method_id),
        "precision_id": str(precision_id),
        "topology_epoch_id": str(topology_epoch_id),
        "partition_id": None if partition_id is None else str(partition_id),
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("Runtime checkpoint compatibility identities changed.")
    observer_specs = manifest.get("observers")
    templates = tuple(observer_templates)
    if not isinstance(observer_specs, list) or len(observer_specs) != len(templates):
        raise ValueError("Runtime checkpoint observer state count changed.")
    state = unpack_array_tree(manifest["state"], arrays, state_template)
    controller = unpack_array_tree(manifest["controller"], arrays, controller_template)
    rng = unpack_array_tree(manifest["rng"], arrays, rng_template)
    observers = tuple(
        unpack_array_tree(specification, arrays, template)
        for specification, template in zip(observer_specs, templates, strict=True)
    )
    return RuntimeCheckpointEnvelope(
        state,
        time=arrays["runtime/time"],
        step_index=arrays["runtime/step_index"],
        schedule_cursor=arrays["runtime/schedule_cursor"],
        mesh_id=mesh_id,
        method_id=method_id,
        precision_id=precision_id,
        topology_epoch_id=topology_epoch_id,
        controller_state=controller,
        observer_states=observers,
        rng_state=rng,
        partition_id=partition_id,
    )


class ExactTimeSchedule(StrictModule, NonTrainableState):
    targets: Array
    tolerance: float = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, targets: ArrayLike, /, *, tolerance: float = 1.0e-12):
        values = np.asarray(targets, dtype=float)
        tolerance_ = float(tolerance)
        if (
            values.ndim != 1
            or values.size == 0
            or np.any(~np.isfinite(values))
            or np.any(np.diff(values) <= 0.0)
            or not math.isfinite(tolerance_)
            or tolerance_ < 0.0
        ):
            raise ValueError("Exact-time schedule targets or tolerance are invalid.")
        self.targets = jnp.asarray(values)
        self.tolerance = tolerance_
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "exact-time-schedule",
                "targets": values.tolist(),
                "tolerance": tolerance_,
            }
        )

    def clamp_step(
        self, time: ArrayLike, step_size: ArrayLike, cursor: ArrayLike, /
    ) -> Array:
        time_ = jnp.asarray(time)
        step = jnp.asarray(step_size)
        cursor_ = jnp.asarray(cursor, dtype=jnp.int32)
        safe = jnp.minimum(cursor_, self.targets.shape[0] - 1)
        remaining = self.targets[safe] - time_
        active = cursor_ < self.targets.shape[0]
        return jnp.where(
            active & (remaining > self.tolerance), jnp.minimum(step, remaining), step
        )

    def advance_cursor(self, time: ArrayLike, cursor: ArrayLike, /) -> Array:
        time_ = jnp.asarray(time)
        cursor_ = jnp.asarray(cursor, dtype=jnp.int32)
        consumed = jnp.sum(self.targets <= time_ + self.tolerance).astype(jnp.int32)
        return jnp.maximum(cursor_, consumed)


ObservableReduction = Literal["sum", "mean", "maximum", "minimum", "last"]


class StreamingObservableState(StrictModule):
    count: Array
    total: Array
    maximum: Array
    minimum: Array
    last: Array
    last_time: Array


class StreamingObservablePlan(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    evaluator: Callable = eqx.field(static=True)
    reduction: ObservableReduction = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, name: str, evaluator: Callable, reduction: ObservableReduction, /):
        name_ = str(name)
        reduction_ = str(reduction)
        if (
            not name_
            or not callable(evaluator)
            or reduction_ not in ("sum", "mean", "maximum", "minimum", "last")
        ):
            raise ValueError("Streaming observable definition is invalid.")
        self.name = name_
        self.evaluator = evaluator
        self.reduction = reduction_
        self.plan_id = canonical_fingerprint(
            {"kind": "streaming-observable", "name": name_, "reduction": reduction_}
        )

    def initial_state(
        self, shape: tuple[int, ...], dtype=float, /
    ) -> StreamingObservableState:
        zeros = jnp.zeros(shape, dtype=dtype)
        return StreamingObservableState(
            jnp.asarray(0, dtype=jnp.int64),
            zeros,
            jnp.full(shape, -jnp.inf, dtype=dtype),
            jnp.full(shape, jnp.inf, dtype=dtype),
            zeros,
            jnp.asarray(-jnp.inf, dtype=dtype),
        )

    def update(
        self,
        time: ArrayLike,
        state: StreamingObservableState,
        simulation_state: Any,
        args: Any = None,
        /,
    ) -> StreamingObservableState:
        if not isinstance(state, StreamingObservableState):
            raise TypeError("state must be StreamingObservableState.")
        value = jnp.asarray(self.evaluator(jnp.asarray(time), simulation_state, args))
        if value.shape != state.total.shape:
            raise ValueError("Observable evaluator changed its declared shape.")
        return StreamingObservableState(
            state.count + 1,
            state.total + value,
            jnp.maximum(state.maximum, value),
            jnp.minimum(state.minimum, value),
            value,
            jnp.asarray(time),
        )

    def value(self, state: StreamingObservableState, /) -> Array:
        if self.reduction == "sum":
            return state.total
        if self.reduction == "mean":
            return state.total / jnp.maximum(state.count, 1)
        if self.reduction == "maximum":
            return state.maximum
        if self.reduction == "minimum":
            return state.minimum
        return state.last

    def merge(
        self, left: StreamingObservableState, right: StreamingObservableState, /
    ) -> StreamingObservableState:
        choose_right = right.last_time >= left.last_time
        return StreamingObservableState(
            left.count + right.count,
            left.total + right.total,
            jnp.maximum(left.maximum, right.maximum),
            jnp.minimum(left.minimum, right.minimum),
            jnp.where(choose_right, right.last, left.last),
            jnp.maximum(left.last_time, right.last_time),
        )


class AcceptedStepTriggerState(StrictModule):
    latched: Array
    fire_count: Array
    last_value: Array


class AcceptedStepTrigger(StrictModule, NonTrainableState):
    threshold: float = eqx.field(static=True)
    direction: Literal["above", "below"] = eqx.field(static=True)
    hysteresis: float = eqx.field(static=True)
    trigger_id: str = eqx.field(static=True)

    def __init__(
        self,
        threshold: float,
        /,
        *,
        direction: Literal["above", "below"] = "above",
        hysteresis: float = 0.0,
    ):
        threshold_ = float(threshold)
        hysteresis_ = float(hysteresis)
        if (
            not math.isfinite(threshold_)
            or direction not in ("above", "below")
            or not math.isfinite(hysteresis_)
            or hysteresis_ < 0.0
        ):
            raise ValueError("Accepted-step trigger parameters are invalid.")
        self.threshold = threshold_
        self.direction = direction
        self.hysteresis = hysteresis_
        self.trigger_id = canonical_fingerprint(
            {
                "kind": "accepted-step-trigger",
                "threshold": threshold_,
                "direction": direction,
                "hysteresis": hysteresis_,
            }
        )

    def initial_state(self, dtype=float, /) -> AcceptedStepTriggerState:
        return AcceptedStepTriggerState(
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(jnp.nan, dtype=dtype),
        )

    def evaluate(
        self,
        value: ArrayLike,
        state: AcceptedStepTriggerState,
        /,
        *,
        accepted: ArrayLike,
    ) -> tuple[Array, AcceptedStepTriggerState]:
        value_ = jnp.asarray(value).reshape(())
        accepted_ = jnp.asarray(accepted, dtype=bool).reshape(())
        active = (
            value_ >= self.threshold
            if self.direction == "above"
            else value_ <= self.threshold
        )
        release = (
            value_ <= self.threshold - self.hysteresis
            if self.direction == "above"
            else value_ >= self.threshold + self.hysteresis
        )
        fire = accepted_ & active & ~state.latched
        latched = jnp.where(
            accepted_, jnp.where(release, False, state.latched | active), state.latched
        )
        return fire, AcceptedStepTriggerState(
            latched,
            state.fire_count + fire.astype(jnp.int64),
            jnp.where(accepted_, value_, state.last_value),
        )


class BoundedAsyncPublisher:
    """One-worker immutable snapshot publisher with bounded backpressure."""

    def __init__(self, writer: Callable[[Any], Any], /, *, maximum_pending: int = 2):
        if not callable(writer) or int(maximum_pending) <= 0:
            raise ValueError("Async publisher requires a writer and positive capacity.")
        self._writer = writer
        self._maximum_pending = int(maximum_pending)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="phydrax-output"
        )
        self._pending: deque[Future] = deque()
        self._closed = False

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @staticmethod
    def _snapshot(value: Any, /) -> Any:
        return jax.tree.map(lambda leaf: np.array(leaf, copy=True), value)

    def publish(self, value: Any, /) -> Future:
        if self._closed:
            raise RuntimeError("Async publisher is closed.")
        while len(self._pending) >= self._maximum_pending:
            self._pending.popleft().result()
        future = self._executor.submit(self._writer, self._snapshot(value))
        self._pending.append(future)
        return future

    def drain(self) -> None:
        while self._pending:
            self._pending.popleft().result()

    def close(self) -> None:
        if not self._closed:
            self.drain()
            self._executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self) -> "BoundedAsyncPublisher":
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        del exception_type, exception, traceback
        self.close()


class ByteBoundedAsyncPublisher:
    """Immutable exactly-once publisher bounded by item count and resident bytes."""

    def __init__(
        self,
        writer: Callable[[str, Any], Any],
        /,
        *,
        maximum_pending: int = 2,
        maximum_pending_bytes: int,
    ):
        if (
            not callable(writer)
            or int(maximum_pending) <= 0
            or int(maximum_pending_bytes) <= 0
        ):
            raise ValueError("Byte-bounded publisher capacity is invalid.")
        self._writer = writer
        self._maximum_pending = int(maximum_pending)
        self._maximum_pending_bytes = int(maximum_pending_bytes)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="phydrax-output"
        )
        self._pending: deque[tuple[Future, int, str]] = deque()
        self._pending_bytes = 0
        self._submitted: set[str] = set()
        self._acknowledged: set[str] = set()
        self._closed = False

    @staticmethod
    def _snapshot(value: Any, /) -> tuple[Any, int]:
        snapshot = jax.tree.map(lambda leaf: np.array(leaf, copy=True), value)
        byte_count = sum(
            int(np.asarray(leaf).nbytes) for leaf in jax.tree.leaves(snapshot)
        )
        return snapshot, byte_count

    @property
    def pending_bytes(self) -> int:
        return self._pending_bytes

    @property
    def acknowledged_event_ids(self) -> frozenset[str]:
        return frozenset(self._acknowledged)

    def _complete_oldest(self) -> None:
        future, byte_count, event_id = self._pending.popleft()
        future.result()
        self._pending_bytes -= byte_count
        self._acknowledged.add(event_id)

    def publish(self, event_id: str, value: Any, /) -> Future:
        identifier = str(event_id)
        if self._closed:
            raise RuntimeError("Async publisher is closed.")
        if not identifier or identifier in self._submitted:
            raise ValueError("Async publication event IDs must be unique.")
        snapshot, byte_count = self._snapshot(value)
        if byte_count > self._maximum_pending_bytes:
            raise ValueError("One output snapshot exceeds the pending-byte budget.")
        while (
            len(self._pending) >= self._maximum_pending
            or self._pending_bytes + byte_count > self._maximum_pending_bytes
        ):
            self._complete_oldest()
        future = self._executor.submit(self._writer, identifier, snapshot)
        self._pending.append((future, byte_count, identifier))
        self._pending_bytes += byte_count
        self._submitted.add(identifier)
        return future

    def drain(self) -> None:
        while self._pending:
            self._complete_oldest()

    def close(self) -> None:
        if not self._closed:
            self.drain()
            self._executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self) -> "ByteBoundedAsyncPublisher":
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        del exception_type, exception, traceback
        self.close()


class StreamingMomentState(StrictModule):
    weight: Array
    mean: Array
    second_moment: Array
    minimum: Array
    maximum: Array
    histogram: Array


class StreamingMomentPlan(StrictModule, NonTrainableState):
    evaluator: Callable = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)
    histogram_edges: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: Callable,
        /,
        *,
        value_shape: tuple[int, ...] = (),
        histogram_edges: ArrayLike = (),
        plan_id: str,
    ):
        edges = jnp.asarray(histogram_edges)
        shape = tuple(int(value) for value in value_shape)
        if (
            not callable(evaluator)
            or any(value <= 0 for value in shape)
            or edges.ndim != 1
            or (edges.size not in (0, 1) and bool(jnp.any(jnp.diff(edges) <= 0.0)))
            or not str(plan_id)
        ):
            raise ValueError("Streaming moment plan is invalid.")
        self.evaluator = evaluator
        self.value_shape = shape
        self.histogram_edges = edges
        self.plan_id = canonical_fingerprint(
            {
                "kind": "streaming-moment-plan",
                "name": str(plan_id),
                "value_shape": shape,
                "histogram_bins": max(int(edges.size) - 1, 0),
            }
        )

    def initial_state(self, dtype=float, /) -> StreamingMomentState:
        return StreamingMomentState(
            jnp.asarray(0.0, dtype=dtype),
            jnp.zeros(self.value_shape, dtype=dtype),
            jnp.zeros(self.value_shape, dtype=dtype),
            jnp.full(self.value_shape, jnp.inf, dtype=dtype),
            jnp.full(self.value_shape, -jnp.inf, dtype=dtype),
            jnp.zeros((max(int(self.histogram_edges.size) - 1, 0),), dtype=jnp.int64),
        )

    def update(
        self,
        time: ArrayLike,
        simulation_state: Any,
        state: StreamingMomentState,
        /,
        *,
        weight: ArrayLike = 1.0,
        args: Any = None,
    ) -> StreamingMomentState:
        value = jnp.asarray(self.evaluator(jnp.asarray(time), simulation_state, args))
        weight_ = jnp.asarray(weight, dtype=value.real.dtype)
        if value.shape != self.value_shape or weight_.shape != ():
            raise ValueError("Streaming moment value or weight shape changed.")
        total_weight = state.weight + weight_
        delta = value - state.mean
        mean = (
            state.mean
            + jnp.where(total_weight > 0.0, weight_ / total_weight, 0.0) * delta
        )
        second = state.second_moment + weight_ * delta * (value - mean)
        histogram = state.histogram
        if self.histogram_edges.size > 1:
            if value.shape != ():
                raise ValueError("Histograms require scalar streaming moments.")
            index = jnp.clip(
                jnp.searchsorted(self.histogram_edges, value, side="right") - 1,
                0,
                histogram.size - 1,
            )
            histogram = histogram.at[index].add(1)
        return StreamingMomentState(
            total_weight,
            mean,
            second,
            jnp.minimum(state.minimum, value),
            jnp.maximum(state.maximum, value),
            histogram,
        )

    def merge(
        self, left: StreamingMomentState, right: StreamingMomentState, /
    ) -> StreamingMomentState:
        total = left.weight + right.weight
        delta = right.mean - left.mean
        mean = left.mean + jnp.where(total > 0.0, right.weight / total, 0.0) * delta
        second = (
            left.second_moment
            + right.second_moment
            + jnp.where(
                total > 0.0,
                left.weight * right.weight / total,
                0.0,
            )
            * delta**2
        )
        return StreamingMomentState(
            total,
            mean,
            second,
            jnp.minimum(left.minimum, right.minimum),
            jnp.maximum(left.maximum, right.maximum),
            left.histogram + right.histogram,
        )

    def variance(self, state: StreamingMomentState, /) -> Array:
        return state.second_moment / jnp.maximum(state.weight, 1.0)


class AcceptedStepTriggerGraphState(StrictModule):
    trigger_states: tuple[AcceptedStepTriggerState, ...]
    debounce_counter: Array
    fire_count: Array


class AcceptedStepTriggerGraph(StrictModule, NonTrainableState):
    triggers: tuple[AcceptedStepTrigger, ...]
    operation: Literal["all", "any"] = eqx.field(static=True)
    debounce_steps: int = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        triggers: tuple[AcceptedStepTrigger, ...],
        /,
        *,
        operation: Literal["all", "any"] = "all",
        debounce_steps: int = 0,
    ):
        if (
            not triggers
            or any(not isinstance(value, AcceptedStepTrigger) for value in triggers)
            or operation not in ("all", "any")
            or int(debounce_steps) < 0
        ):
            raise ValueError("Accepted-step trigger graph is invalid.")
        self.triggers = tuple(triggers)
        self.operation = operation
        self.debounce_steps = int(debounce_steps)
        self.graph_id = canonical_fingerprint(
            {
                "kind": "accepted-step-trigger-graph",
                "triggers": tuple(value.trigger_id for value in triggers),
                "operation": operation,
                "debounce_steps": int(debounce_steps),
            }
        )

    def initial_state(self, dtype=float, /) -> AcceptedStepTriggerGraphState:
        return AcceptedStepTriggerGraphState(
            tuple(value.initial_state(dtype) for value in self.triggers),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int64),
        )

    def evaluate(
        self,
        values: tuple[ArrayLike, ...],
        state: AcceptedStepTriggerGraphState,
        /,
        *,
        accepted: ArrayLike,
    ) -> tuple[Array, AcceptedStepTriggerGraphState]:
        if len(values) != len(self.triggers):
            raise ValueError("Trigger graph value count changed.")
        fires = []
        states = []
        for trigger, value, trigger_state in zip(
            self.triggers, values, state.trigger_states, strict=True
        ):
            fire, updated = trigger.evaluate(value, trigger_state, accepted=accepted)
            fires.append(fire)
            states.append(updated)
        active = (
            jnp.all(jnp.stack(tuple(fires)))
            if self.operation == "all"
            else jnp.any(jnp.stack(tuple(fires)))
        )
        counter = jnp.where(
            active,
            state.debounce_counter + 1,
            jnp.asarray(0, dtype=jnp.int32),
        )
        fire = active & (counter > self.debounce_steps)
        return fire, AcceptedStepTriggerGraphState(
            tuple(states),
            counter,
            state.fire_count + fire.astype(jnp.int64),
        )


__all__ = [
    "AcceptedStepTrigger",
    "AcceptedStepTriggerState",
    "AcceptedStepTriggerGraph",
    "AcceptedStepTriggerGraphState",
    "BoundedAsyncPublisher",
    "ExactTimeSchedule",
    "ByteBoundedAsyncPublisher",
    "RuntimeCheckpointEnvelope",
    "StreamingObservablePlan",
    "StreamingObservableState",
    "read_runtime_checkpoint",
    "StreamingMomentPlan",
    "StreamingMomentState",
    "write_runtime_checkpoint",
]
