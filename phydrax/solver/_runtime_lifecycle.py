#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._array_archive import (
    array_collection_digest,
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.spectral._coordinates import HermitianSpectralCoordinates
from ..linalg._real_coordinates import RealCoordinateEvidence


def _host_array_tree(tree: Any, role: str, /) -> Any:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    arrays = []
    for leaf in leaves:
        value = np.asarray(leaf)
        if value.dtype.hasobject:
            raise TypeError(f"Runtime checkpoint {role} must be an array-only PyTree.")
        arrays.append(value)
    return jax.tree_util.tree_unflatten(treedef, arrays)


def _tree_where(predicate: Array, proposed: Any, current: Any, /) -> Any:
    if jax.tree_util.tree_structure(proposed) != jax.tree_util.tree_structure(current):
        raise ValueError("Transactional runtime updates must preserve PyTree structure.")
    return jax.tree.map(
        lambda new, old: jnp.where(predicate, new, old), proposed, current
    )


class RuntimeCheckpointLeafBinding(StrictModule, NonTrainableState):
    """Hermitian coordinate encoding for one indexed state-tree leaf."""

    leaf_index: int = eqx.field(static=True)
    coordinates: HermitianSpectralCoordinates
    evidence: RealCoordinateEvidence
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        leaf_index: int,
        coordinates: HermitianSpectralCoordinates,
        evidence: RealCoordinateEvidence,
        /,
    ):
        index = int(leaf_index)
        if index < 0:
            raise ValueError("Checkpoint encoding leaf_index must be nonnegative.")
        if not isinstance(coordinates, HermitianSpectralCoordinates):
            raise TypeError("coordinates must be HermitianSpectralCoordinates.")
        if not isinstance(evidence, RealCoordinateEvidence):
            raise TypeError("evidence must be RealCoordinateEvidence.")
        if evidence.evidence_id != coordinates.evidence.evidence_id:
            raise ValueError(
                "Checkpoint coordinate evidence does not bind the coordinate map."
            )
        self.leaf_index = index
        self.coordinates = coordinates
        self.evidence = evidence
        self.binding_id = canonical_fingerprint(
            {
                "kind": "runtime-checkpoint-leaf-binding",
                "leaf_index": index,
                "coordinates": coordinates.coordinate_id,
                "evidence": evidence.evidence_id,
            }
        )


class RuntimeCheckpointEncodingPlan(StrictModule, NonTrainableState):
    """Fixed leafwise encoding for runtime checkpoints; unbound leaves stay native."""

    bindings: tuple[RuntimeCheckpointLeafBinding, ...]
    encoding_id: str = eqx.field(static=True)

    def __init__(self, bindings: Sequence[RuntimeCheckpointLeafBinding] = (), /):
        bindings_ = tuple(bindings)
        if any(
            not isinstance(value, RuntimeCheckpointLeafBinding) for value in bindings_
        ):
            raise TypeError("Every checkpoint encoding must be a leaf binding.")
        indices = tuple(value.leaf_index for value in bindings_)
        if len(set(indices)) != len(indices):
            raise ValueError("Checkpoint encoding leaf indices must be unique.")
        ordered = tuple(sorted(bindings_, key=lambda value: value.leaf_index))
        self.bindings = ordered
        self.encoding_id = canonical_fingerprint(
            {
                "kind": "runtime-checkpoint-encoding-plan",
                "bindings": tuple(value.binding_id for value in ordered),
            }
        )

    def binding_for(self, leaf_index: int, /) -> RuntimeCheckpointLeafBinding | None:
        for binding in self.bindings:
            if binding.leaf_index == leaf_index:
                return binding
        return None


def _pack_state_tree(
    tree: Any,
    arrays: dict[str, object],
    encoding: RuntimeCheckpointEncodingPlan,
    /,
) -> dict[str, object]:
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    if encoding.bindings and encoding.bindings[-1].leaf_index >= len(path_leaves):
        raise ValueError("Checkpoint encoding selects a state leaf that does not exist.")
    paths: list[str] = []
    names: list[str] = []
    records: list[dict[str, Any] | None] = []
    for index, (path, leaf) in enumerate(path_leaves):
        path_string = jax.tree_util.keystr(path) or "<root>"
        name = f"state/{index:06d}"
        binding = encoding.binding_for(index)
        value = np.asarray(leaf)
        if binding is None:
            stored = value
            record = None
        else:
            if value.shape != binding.evidence.source_shape or value.dtype != np.dtype(
                binding.evidence.source_dtype
            ):
                raise ValueError(
                    "Hermitian checkpoint encoding does not match the selected leaf."
                )
            stored = np.asarray(
                binding.coordinates.to_real_coordinates(jnp.asarray(value))
            )
            record = {
                "binding_id": binding.binding_id,
                "coordinate_id": binding.coordinates.coordinate_id,
                "evidence_id": binding.evidence.evidence_id,
            }
        if stored.dtype.hasobject:
            raise TypeError("Runtime checkpoint state leaves must be native arrays.")
        paths.append(path_string)
        names.append(name)
        records.append(record)
        arrays[name] = stored
    return {
        "paths": paths,
        "arrays": names,
        "num_leaves": len(names),
        "encodings": records,
    }


def _unpack_state_tree(
    specification: Mapping[str, Any],
    arrays: Mapping[str, Any],
    template: Any,
    encoding: RuntimeCheckpointEncodingPlan,
    /,
) -> Any:
    if not isinstance(specification, Mapping):
        raise ValueError("Archived state-tree specification must be a mapping.")
    template_path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    expected_paths = [
        jax.tree_util.keystr(path) or "<root>" for path, _ in template_path_leaves
    ]
    names = specification.get("arrays")
    records = specification.get("encodings")
    if (
        specification.get("paths") != expected_paths
        or specification.get("num_leaves") != len(expected_paths)
        or not isinstance(names, list)
        or len(names) != len(expected_paths)
        or not isinstance(records, list)
        or len(records) != len(expected_paths)
    ):
        raise ValueError("Archived state does not match the exact runtime template.")
    if encoding.bindings and encoding.bindings[-1].leaf_index >= len(expected_paths):
        raise ValueError("Checkpoint encoding selects a state leaf that does not exist.")
    leaves = []
    for index, (name, record, (_, template_leaf)) in enumerate(
        zip(names, records, template_path_leaves, strict=True)
    ):
        if not isinstance(name, str) or name not in arrays:
            raise ValueError("Archived state array is missing.")
        binding = encoding.binding_for(index)
        if binding is None:
            if record is not None:
                raise ValueError("Archived state unexpectedly uses coordinate encoding.")
            value = jnp.asarray(arrays[name])
        else:
            expected_record = {
                "binding_id": binding.binding_id,
                "coordinate_id": binding.coordinates.coordinate_id,
                "evidence_id": binding.evidence.evidence_id,
            }
            if record != expected_record:
                raise ValueError("Archived coordinate encoding identity changed.")
            coordinates = jnp.asarray(arrays[name])
            if (
                coordinates.shape != binding.evidence.coordinate_shape
                or coordinates.dtype != jnp.dtype(binding.evidence.coordinate_dtype)
            ):
                raise ValueError("Archived Hermitian coordinates changed shape or dtype.")
            value = binding.coordinates.from_real_coordinates(coordinates)
        expected = jnp.asarray(template_leaf)
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError("Archived state leaf shape or dtype changed.")
        leaves.append(value)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _default_runtime_id(
    mesh_id: str,
    method_id: str,
    precision_id: str,
    topology_epoch_id: str,
    partition_id: str | None,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "runtime-compatibility",
            "mesh": mesh_id,
            "method": method_id,
            "precision": precision_id,
            "topology_epoch": topology_epoch_id,
            "partition": partition_id,
        }
    )


class RuntimeCheckpointEnvelope(StrictModule):
    state: Any
    controller_state: Any
    observer_states: tuple[Any, ...]
    rng_state: Any
    time: Array
    step_index: Array
    schedule_cursor: Array
    encoding_plan: RuntimeCheckpointEncodingPlan
    archive_arrays: dict[str, Any]
    archive_specs: dict[str, Any] = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    topology_epoch_id: str = eqx.field(static=True)
    partition_id: str | None = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    content_digest: str = eqx.field(static=True)
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
        runtime_id: str | None = None,
        encoding_plan: RuntimeCheckpointEncodingPlan | None = None,
    ):
        time_ = np.asarray(time)
        step = np.asarray(step_index)
        cursor = np.asarray(schedule_cursor)
        if (
            time_.shape != ()
            or not np.issubdtype(time_.dtype, np.inexact)
            or not np.isfinite(time_)
            or step.shape != ()
            or step.dtype.kind not in "iu"
            or int(step) < 0
            or cursor.shape != ()
            or cursor.dtype.kind not in "iu"
            or int(cursor) < 0
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
        encoding = (
            RuntimeCheckpointEncodingPlan() if encoding_plan is None else encoding_plan
        )
        if not isinstance(encoding, RuntimeCheckpointEncodingPlan):
            raise TypeError(
                "encoding_plan must be RuntimeCheckpointEncodingPlan or None."
            )
        state_ = _host_array_tree(state, "state")
        controller = _host_array_tree(controller_state, "controller state")
        observers = tuple(
            _host_array_tree(value, "observer state") for value in observer_states
        )
        rng = _host_array_tree(rng_state, "RNG state")
        arrays: dict[str, object] = {
            "runtime/time": time_,
            "runtime/step_index": step.astype(np.int64),
            "runtime/schedule_cursor": cursor.astype(np.int64),
        }
        state_spec = _pack_state_tree(state_, arrays, encoding)
        controller_spec = pack_array_tree("controller", controller, arrays)
        rng_spec = pack_array_tree("rng", rng, arrays)
        observer_specs = [
            pack_array_tree(f"observer/{index:04d}", value, arrays)
            for index, value in enumerate(observers)
        ]
        content_digest = array_collection_digest(arrays)
        runtime = (
            _default_runtime_id(*identifiers, partition)
            if runtime_id is None
            else str(runtime_id)
        )
        if not runtime:
            raise ValueError("runtime_id must be nonempty when supplied.")
        specs = {
            "state": state_spec,
            "controller": controller_spec,
            "rng": rng_spec,
            "observers": observer_specs,
        }
        self.state = state_
        self.controller_state = controller
        self.observer_states = observers
        self.rng_state = rng
        self.time = time_
        self.step_index = arrays["runtime/step_index"]
        self.schedule_cursor = arrays["runtime/schedule_cursor"]
        self.encoding_plan = encoding
        self.archive_arrays = arrays
        self.archive_specs = specs
        self.mesh_id, self.method_id, self.precision_id, self.topology_epoch_id = (
            identifiers
        )
        self.partition_id = partition
        self.runtime_id = runtime
        self.content_digest = content_digest
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "runtime-checkpoint-envelope",
                "runtime": runtime,
                "mesh": self.mesh_id,
                "method": self.method_id,
                "precision": self.precision_id,
                "topology_epoch": self.topology_epoch_id,
                "partition": partition,
                "encoding": encoding.encoding_id,
                "content": content_digest,
                "trees": specs,
            }
        )


def write_runtime_checkpoint(
    path: str | Path, envelope: RuntimeCheckpointEnvelope, /
) -> Path:
    if not isinstance(envelope, RuntimeCheckpointEnvelope):
        raise TypeError("envelope must be RuntimeCheckpointEnvelope.")
    manifest = {
        "kind": "runtime-checkpoint",
        "checkpoint_id": envelope.checkpoint_id,
        "runtime_id": envelope.runtime_id,
        "content_digest": envelope.content_digest,
        "encoding_id": envelope.encoding_plan.encoding_id,
        "mesh_id": envelope.mesh_id,
        "method_id": envelope.method_id,
        "precision_id": envelope.precision_id,
        "topology_epoch_id": envelope.topology_epoch_id,
        "partition_id": envelope.partition_id,
        **envelope.archive_specs,
    }
    return write_array_archive(path, manifest=manifest, arrays=envelope.archive_arrays)


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
    runtime_id: str | None = None,
    encoding_plan: RuntimeCheckpointEncodingPlan | None = None,
) -> RuntimeCheckpointEnvelope:
    manifest, arrays = read_array_archive(path)
    partition = None if partition_id is None else str(partition_id)
    runtime = (
        _default_runtime_id(
            str(mesh_id),
            str(method_id),
            str(precision_id),
            str(topology_epoch_id),
            partition,
        )
        if runtime_id is None
        else str(runtime_id)
    )
    encoding = RuntimeCheckpointEncodingPlan() if encoding_plan is None else encoding_plan
    if not isinstance(encoding, RuntimeCheckpointEncodingPlan):
        raise TypeError("encoding_plan must be RuntimeCheckpointEncodingPlan or None.")
    expected = {
        "kind": "runtime-checkpoint",
        "runtime_id": runtime,
        "encoding_id": encoding.encoding_id,
        "mesh_id": str(mesh_id),
        "method_id": str(method_id),
        "precision_id": str(precision_id),
        "topology_epoch_id": str(topology_epoch_id),
        "partition_id": partition,
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("Runtime checkpoint compatibility identities changed.")
    observer_specs = manifest.get("observers")
    templates = tuple(observer_templates)
    if not isinstance(observer_specs, list) or len(observer_specs) != len(templates):
        raise ValueError("Runtime checkpoint observer state count changed.")
    state = _unpack_state_tree(manifest.get("state"), arrays, state_template, encoding)
    controller = unpack_array_tree(
        manifest.get("controller"), arrays, controller_template
    )
    rng = unpack_array_tree(manifest.get("rng"), arrays, rng_template)
    observers = tuple(
        unpack_array_tree(specification, arrays, template)
        for specification, template in zip(observer_specs, templates, strict=True)
    )
    envelope = RuntimeCheckpointEnvelope(
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
        runtime_id=runtime,
        encoding_plan=encoding,
    )
    if (
        manifest.get("content_digest") != envelope.content_digest
        or manifest.get("checkpoint_id") != envelope.checkpoint_id
    ):
        raise ValueError("Runtime checkpoint content identity is inconsistent.")
    return envelope


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

    def initial_cursor(self, time: ArrayLike, /) -> Array:
        return self.advance_cursor(time, jnp.asarray(0, dtype=jnp.int32))


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

    def __enter__(self) -> BoundedAsyncPublisher:
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        del exception_type, exception, traceback
        self.close()


class ByteBoundedAsyncPublisher:
    """Immutable process-exact publisher with bounded asynchronous backpressure."""

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

    def __enter__(self) -> ByteBoundedAsyncPublisher:
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        del exception_type, exception, traceback
        self.close()


MomentWeighting = Literal["sample", "time"]


class StreamingMomentState(StrictModule):
    weight: Array
    mean: Array
    second_moment: Array
    minimum: Array
    maximum: Array
    histogram: Array
    batch_weights: Array
    batch_totals: Array
    last_time: Array


class StreamingMomentPlan(StrictModule, NonTrainableState):
    evaluator: Callable
    value_shape: tuple[int, ...] = eqx.field(static=True)
    histogram_edges: Array
    weighting: MomentWeighting = eqx.field(static=True)
    window_start: float | None = eqx.field(static=True)
    window_end: float | None = eqx.field(static=True)
    batch_duration: float | None = eqx.field(static=True)
    maximum_batches: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: Callable,
        /,
        *,
        value_shape: tuple[int, ...] = (),
        histogram_edges: ArrayLike = (),
        weighting: MomentWeighting = "time",
        window_start: float | None = None,
        window_end: float | None = None,
        batch_duration: float | None = None,
        maximum_batches: int = 0,
        plan_id: str,
    ):
        edges = np.asarray(histogram_edges)
        shape = tuple(int(value) for value in value_shape)
        start = None if window_start is None else float(window_start)
        end = None if window_end is None else float(window_end)
        duration = None if batch_duration is None else float(batch_duration)
        batches = int(maximum_batches)
        if (
            not callable(evaluator)
            or any(value <= 0 for value in shape)
            or edges.ndim != 1
            or (edges.size not in (0, 1) and np.any(np.diff(edges) <= 0.0))
            or weighting not in ("sample", "time")
            or (start is not None and not math.isfinite(start))
            or (end is not None and not math.isfinite(end))
            or (start is not None and end is not None and end <= start)
            or (duration is not None and (not math.isfinite(duration) or duration <= 0.0))
            or batches < 0
            or ((duration is None) != (batches == 0))
            or (duration is not None and start is None)
            or not str(plan_id)
        ):
            raise ValueError("Streaming moment plan is invalid.")
        self.evaluator = evaluator
        self.value_shape = shape
        self.histogram_edges = jnp.asarray(edges)
        self.weighting = weighting
        self.window_start = start
        self.window_end = end
        self.batch_duration = duration
        self.maximum_batches = batches
        self.plan_id = canonical_fingerprint(
            {
                "kind": "streaming-moment-plan",
                "name": str(plan_id),
                "value_shape": shape,
                "histogram_edges": edges.tolist(),
                "weighting": weighting,
                "window_start": start,
                "window_end": end,
                "batch_duration": duration,
                "maximum_batches": batches,
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
            jnp.zeros((self.maximum_batches,), dtype=dtype),
            jnp.zeros((self.maximum_batches, *self.value_shape), dtype=dtype),
            jnp.asarray(-jnp.inf, dtype=dtype),
        )

    def _window_weight(
        self,
        previous_time: Array,
        time: Array,
        explicit_weight: ArrayLike | None,
        /,
    ) -> Array:
        if explicit_weight is not None:
            base = jnp.asarray(explicit_weight, dtype=time.dtype)
            if base.shape != ():
                raise ValueError("Streaming moment weight must be scalar.")
            included = jnp.asarray(True)
            if self.window_start is not None:
                included = included & (time >= self.window_start)
            if self.window_end is not None:
                included = included & (time <= self.window_end)
            return jnp.where(included, base, 0.0)
        if self.weighting == "sample":
            included = jnp.asarray(True)
            if self.window_start is not None:
                included = included & (time >= self.window_start)
            if self.window_end is not None:
                included = included & (time <= self.window_end)
            return included.astype(time.dtype)
        start = previous_time
        end = time
        if self.window_start is not None:
            start = jnp.maximum(start, self.window_start)
        if self.window_end is not None:
            end = jnp.minimum(end, self.window_end)
        return jnp.maximum(end - start, 0.0)

    def update(
        self,
        time: ArrayLike,
        simulation_state: Any,
        state: StreamingMomentState,
        /,
        *,
        previous_time: ArrayLike | None = None,
        weight: ArrayLike | None = None,
        args: Any = None,
    ) -> StreamingMomentState:
        time_ = jnp.asarray(time)
        previous = time_ if previous_time is None else jnp.asarray(previous_time)
        if self.weighting == "time" and previous_time is None and weight is None:
            raise ValueError("Time-weighted moments require previous_time or weight.")
        value = jnp.asarray(self.evaluator(time_, simulation_state, args))
        if value.shape != self.value_shape or jnp.issubdtype(
            value.dtype, jnp.complexfloating
        ):
            raise ValueError("Streaming moment evaluator changed shape or real dtype.")
        weight_ = self._window_weight(previous, time_, weight).astype(value.dtype)
        total_weight = state.weight + weight_
        active = weight_ > 0.0
        delta = value - state.mean
        mean = state.mean + jnp.where(active, weight_ / total_weight, 0.0) * delta
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
            histogram = histogram.at[index].add(active.astype(jnp.int64))
        batch_weights = state.batch_weights
        batch_totals = state.batch_totals
        if self.batch_duration is not None:
            starts = self.window_start + self.batch_duration * jnp.arange(
                self.maximum_batches, dtype=time_.dtype
            )
            ends = starts + self.batch_duration
            if self.weighting == "time" and weight is None:
                overlaps = jnp.maximum(
                    jnp.minimum(time_, ends) - jnp.maximum(previous, starts), 0.0
                )
            else:
                overlaps = ((time_ >= starts) & (time_ < ends)).astype(time_.dtype)
                overlaps = overlaps * jnp.where(active, weight_, 0.0)
            batch_weights = batch_weights + overlaps
            reshape = (self.maximum_batches,) + (1,) * len(self.value_shape)
            batch_totals = batch_totals + overlaps.reshape(reshape) * value
        return StreamingMomentState(
            total_weight,
            mean,
            second,
            jnp.where(active, jnp.minimum(state.minimum, value), state.minimum),
            jnp.where(active, jnp.maximum(state.maximum, value), state.maximum),
            histogram,
            batch_weights,
            batch_totals,
            jnp.maximum(state.last_time, time_),
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
            + jnp.where(total > 0.0, left.weight * right.weight / total, 0.0) * delta**2
        )
        return StreamingMomentState(
            total,
            mean,
            second,
            jnp.minimum(left.minimum, right.minimum),
            jnp.maximum(left.maximum, right.maximum),
            left.histogram + right.histogram,
            left.batch_weights + right.batch_weights,
            left.batch_totals + right.batch_totals,
            jnp.maximum(left.last_time, right.last_time),
        )

    def variance(self, state: StreamingMomentState, /) -> Array:
        return jnp.where(
            state.weight > 0.0,
            state.second_moment / state.weight,
            jnp.zeros_like(state.second_moment),
        )

    def batch_mean_standard_error(self, state: StreamingMomentState, /) -> Array:
        if self.batch_duration is None:
            raise ValueError("Batch-mean uncertainty was not configured.")
        ends = self.window_start + self.batch_duration * (
            jnp.arange(self.maximum_batches, dtype=state.last_time.dtype) + 1
        )
        completed = (ends <= state.last_time) & (state.batch_weights > 0.0)
        reshape = (self.maximum_batches,) + (1,) * len(self.value_shape)
        mask = completed.reshape(reshape)
        means = jnp.where(
            mask,
            state.batch_totals / state.batch_weights.reshape(reshape),
            0.0,
        )
        count = jnp.sum(completed)
        aggregate = jnp.sum(jnp.where(mask, means, 0.0), axis=0) / jnp.maximum(count, 1)
        squared = jnp.sum(jnp.where(mask, (means - aggregate) ** 2, 0.0), axis=0)
        sample_variance = squared / jnp.maximum(count - 1, 1)
        return jnp.where(count > 1, jnp.sqrt(sample_variance / count), jnp.nan)


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
        fire = (
            jnp.asarray(accepted, dtype=bool) & active & (counter > self.debounce_steps)
        )
        proposed = AcceptedStepTriggerGraphState(
            tuple(states),
            counter,
            state.fire_count + fire.astype(jnp.int64),
        )
        return fire, _tree_where(jnp.asarray(accepted, dtype=bool), proposed, state)


__all__ = [
    "AcceptedStepTrigger",
    "AcceptedStepTriggerState",
    "AcceptedStepTriggerGraph",
    "AcceptedStepTriggerGraphState",
    "BoundedAsyncPublisher",
    "ExactTimeSchedule",
    "ByteBoundedAsyncPublisher",
    "MomentWeighting",
    "RuntimeCheckpointEncodingPlan",
    "RuntimeCheckpointEnvelope",
    "RuntimeCheckpointLeafBinding",
    "StreamingObservablePlan",
    "StreamingObservableState",
    "read_runtime_checkpoint",
    "StreamingMomentPlan",
    "StreamingMomentState",
    "write_runtime_checkpoint",
]
