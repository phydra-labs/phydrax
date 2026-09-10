#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transform-safe iteration observation, control, and host delivery."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, cast, Literal, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ._execution_control import (
    DistributedObservationPolicy,
    global_boolean_consensus,
    ObservationScope,
)
from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


IterationGranularity = Literal[
    "terminal",
    "output",
    "segment",
    "step",
    "attempt",
    "inner-iteration",
]
_GRANULARITIES: tuple[IterationGranularity, ...] = (
    "terminal",
    "output",
    "segment",
    "step",
    "attempt",
    "inner-iteration",
)


class IterationPhase(IntEnum):
    """Portable lifecycle phase for a solver-owned iteration record."""

    START = 0
    ATTEMPT = 1
    COMMIT = 2
    VALIDATE = 3
    TERMINAL = 4


class IterationCoordinates(StrictModule):
    """Array-valued logical coordinates shared by all iteration records."""

    phase: Array
    ordinal: Array
    invocation: Array
    attempt: Array
    accepted: Array
    rejected: Array
    active: Array
    committed: Array
    terminal: Array

    def __init__(
        self,
        phase: IterationPhase | ArrayLike,
        ordinal: ArrayLike,
        /,
        *,
        invocation: ArrayLike = 0,
        attempt: ArrayLike = 0,
        accepted: ArrayLike = 0,
        rejected: ArrayLike = 0,
        active: ArrayLike = True,
        committed: ArrayLike = False,
        terminal: ArrayLike = False,
    ):
        self.phase = jnp.asarray(phase, dtype=jnp.int32)
        self.ordinal = jnp.asarray(ordinal, dtype=jnp.int32)
        self.invocation = jnp.asarray(invocation, dtype=jnp.int32)
        self.attempt = jnp.asarray(attempt, dtype=jnp.int32)
        self.accepted = jnp.asarray(accepted, dtype=jnp.int32)
        self.rejected = jnp.asarray(rejected, dtype=jnp.int32)
        self.active = jnp.asarray(active, dtype=bool)
        self.committed = jnp.asarray(committed, dtype=bool)
        self.terminal = jnp.asarray(terminal, dtype=bool)


class IterationRecord(StrictModule):
    """One typed, transform-safe record from a single logical scope."""

    coordinates: IterationCoordinates
    status: Array
    metrics: PyTree[Any]

    def __init__(
        self,
        coordinates: IterationCoordinates,
        status: ArrayLike,
        metrics: PyTree[Any],
        /,
    ):
        if not isinstance(coordinates, IterationCoordinates):
            raise TypeError("coordinates must be IterationCoordinates.")
        self.coordinates = coordinates
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.metrics = metrics


class IterationCapabilities(StrictModule, NonTrainableState):
    """Static observation and control capabilities of one iterative owner."""

    granularities: tuple[IterationGranularity, ...] = eqx.field(static=True)
    child_roles: tuple[str, ...] = eqx.field(static=True)
    device_stop: bool = eqx.field(static=True)
    host_stop: bool = eqx.field(static=True)
    host_streaming: bool = eqx.field(static=True)
    mapped_records: bool = eqx.field(static=True)
    checkpointable: bool = eqx.field(static=True)

    def __init__(
        self,
        granularities: Sequence[IterationGranularity] = ("terminal",),
        /,
        *,
        child_roles: Sequence[str] = (),
        device_stop: bool = False,
        host_stop: bool = False,
        host_streaming: bool = False,
        mapped_records: bool = True,
        checkpointable: bool = False,
    ):
        granularities_ = cast(
            tuple[IterationGranularity, ...],
            tuple(str(value) for value in granularities),
        )
        if not granularities_ or any(
            value not in _GRANULARITIES for value in granularities_
        ):
            raise ValueError("Iteration granularities are invalid.")
        if "terminal" not in granularities_:
            raise ValueError("Every iterative owner must support terminal evidence.")
        child_roles_ = tuple(str(role) for role in child_roles)
        if any(not role for role in child_roles_) or len(set(child_roles_)) != len(
            child_roles_
        ):
            raise ValueError("Iteration child roles must be unique non-empty strings.")
        self.granularities = granularities_
        self.child_roles = child_roles_
        self.device_stop = bool(device_stop)
        self.host_stop = bool(host_stop)
        self.host_streaming = bool(host_streaming)
        self.mapped_records = bool(mapped_records)
        self.checkpointable = bool(checkpointable)

    @classmethod
    def terminal_only(cls) -> IterationCapabilities:
        return cls(("terminal",))


class IterationScope(StrictModule, NonTrainableState):
    """Static deterministic identity of one logical iterative scope."""

    scope_id: str = eqx.field(static=True)
    parent_scope_id: str | None = eqx.field(static=True)
    role: str = eqx.field(static=True)
    algorithm_id: str = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        scope_id: str,
        /,
        *,
        parent_scope_id: str | None,
        role: str,
        algorithm_id: str,
        depth: int,
    ):
        scope_id_ = str(scope_id)
        role_ = str(role)
        algorithm_id_ = str(algorithm_id)
        depth_ = int(depth)
        if not scope_id_ or not role_ or not algorithm_id_ or depth_ < 0:
            raise ValueError("Iteration scope identity is invalid.")
        self.scope_id = scope_id_
        self.parent_scope_id = None if parent_scope_id is None else str(parent_scope_id)
        self.role = role_
        self.algorithm_id = algorithm_id_
        self.depth = depth_


class IterationDecision(StrictModule):
    """Pure control decision evaluated at an owner-defined safe boundary."""

    stop: Array

    def __init__(self, stop: ArrayLike = False, /):
        stop_ = jnp.asarray(stop, dtype=bool)
        if stop_.shape != ():
            raise ValueError("Iteration stop decisions must be scalar.")
        self.stop = stop_


class AbstractIterationObserver(StrictModule, NonTrainableState):
    """Pure fixed-shape observer evaluated inside transformed execution."""

    observer_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def initialize(self, initial: IterationRecord, /) -> PyTree[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, state: PyTree[Any], record: IterationRecord, /) -> PyTree[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def finalize(
        self,
        state: PyTree[Any],
        initial: IterationRecord,
        terminal: IterationRecord,
        /,
    ) -> PyTree[Any]:
        raise NotImplementedError


class AbstractIterationStopRule(StrictModule, NonTrainableState):
    """Pure fixed-shape stopping rule evaluated only at safe boundaries."""

    rule_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def initialize(self, initial: IterationRecord, /) -> PyTree[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self, state: PyTree[Any], record: IterationRecord, /
    ) -> tuple[PyTree[Any], IterationDecision]:
        raise NotImplementedError


class IterationTraceState(StrictModule):
    records: PyTree[Any]
    stored_count: Array
    seen_count: Array
    dropped_count: Array


class IterationTrace(StrictModule):
    """Bounded homogeneous record trace with lossless terminal evidence."""

    initial: IterationRecord
    records: PyTree[Any]
    valid: Array
    terminal: IterationRecord
    stored_count: Array
    seen_count: Array
    dropped_count: Array


class IterationTraceObserver(AbstractIterationObserver):
    """Retain a bounded cadence-sampled trace of domain records."""

    capacity: int = eqx.field(static=True)
    cadence: int = eqx.field(static=True)
    committed_only: bool = eqx.field(static=True)
    observer_id: str = eqx.field(static=True)

    def __init__(
        self,
        capacity: int,
        /,
        *,
        cadence: int = 1,
        committed_only: bool = False,
    ):
        capacity_ = int(capacity)
        cadence_ = int(cadence)
        if capacity_ < 0 or cadence_ <= 0:
            raise ValueError("Trace capacity and cadence are invalid.")
        self.capacity = capacity_
        self.cadence = cadence_
        self.committed_only = bool(committed_only)
        self.observer_id = canonical_fingerprint(
            {
                "kind": "iteration-trace",
                "capacity": capacity_,
                "cadence": cadence_,
                "committed_only": bool(committed_only),
            }
        )

    def initialize(self, initial: IterationRecord, /) -> IterationTraceState:
        records = jax.tree.map(
            lambda leaf: jnp.zeros(
                (self.capacity,) + jnp.asarray(leaf).shape,
                dtype=jnp.asarray(leaf).dtype,
            ),
            initial,
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        return IterationTraceState(records, zero, zero, zero)

    def update(
        self, state: IterationTraceState, record: IterationRecord, /
    ) -> IterationTraceState:
        coordinates = record.coordinates
        selected = jnp.any(coordinates.active) & (
            jnp.mod(coordinates.ordinal, self.cadence) == 0
        )
        if self.committed_only:
            selected = selected & jnp.any(coordinates.committed)
        can_store = selected & (state.stored_count < self.capacity)
        if self.capacity == 0:
            records = state.records
        else:
            index = jnp.minimum(state.stored_count, self.capacity - 1)
            candidate = jax.tree.map(
                lambda buffer, leaf: buffer.at[index].set(leaf),
                state.records,
                record,
            )
            records = _tree_select(can_store, candidate, state.records)
        return IterationTraceState(
            records,
            state.stored_count + can_store.astype(jnp.int32),
            state.seen_count + selected.astype(jnp.int32),
            state.dropped_count + (selected & ~can_store).astype(jnp.int32),
        )

    def finalize(
        self,
        state: IterationTraceState,
        initial: IterationRecord,
        terminal: IterationRecord,
        /,
    ) -> IterationTrace:
        valid = jnp.arange(self.capacity, dtype=jnp.int32) < state.stored_count
        return IterationTrace(
            initial,
            state.records,
            valid,
            terminal,
            state.stored_count,
            state.seen_count,
            state.dropped_count,
        )


class IterationCounts(StrictModule):
    attempted: Array
    committed: Array
    rejected: Array


class IterationCountObserver(AbstractIterationObserver):
    """Count active attempts, commits, and rejections in constant memory."""

    observer_id: str = eqx.field(static=True)

    def __init__(self):
        self.observer_id = canonical_fingerprint({"kind": "iteration-counts"})

    def initialize(self, initial: IterationRecord, /) -> IterationCounts:
        del initial
        zero = jnp.asarray(0, dtype=jnp.int32)
        return IterationCounts(zero, zero, zero)

    def update(
        self, state: IterationCounts, record: IterationRecord, /
    ) -> IterationCounts:
        coordinates = record.coordinates
        active = jnp.sum(
            jnp.asarray(coordinates.active, dtype=jnp.int32), dtype=jnp.int32
        )
        committed = jnp.sum(
            jnp.asarray(coordinates.active & coordinates.committed, dtype=jnp.int32),
            dtype=jnp.int32,
        )
        rejected = jnp.sum(
            jnp.asarray(
                coordinates.active
                & (coordinates.phase == int(IterationPhase.ATTEMPT))
                & ~coordinates.committed,
                dtype=jnp.int32,
            ),
            dtype=jnp.int32,
        )
        return IterationCounts(
            state.attempted + active,
            state.committed + committed,
            state.rejected + rejected,
        )

    def finalize(
        self,
        state: IterationCounts,
        initial: IterationRecord,
        terminal: IterationRecord,
        /,
    ) -> IterationCounts:
        del initial, terminal
        return state


class IterationMomentState(StrictModule):
    count: Array
    total: PyTree[Any]
    minimum: PyTree[Any]
    maximum: PyTree[Any]
    last: PyTree[Any]


class IterationMoments(StrictModule):
    count: Array
    total: PyTree[Any]
    mean: PyTree[Any]
    minimum: PyTree[Any]
    maximum: PyTree[Any]
    last: PyTree[Any]


class IterationMomentObserver(AbstractIterationObserver):
    """Compute streaming moments of a pure real-valued record projection."""

    projector: Callable = eqx.field(static=True)
    observer_id: str = eqx.field(static=True)

    def __init__(self, projector: Callable, observer_id: str, /):
        observer_id_ = str(observer_id)
        if not callable(projector) or not observer_id_:
            raise ValueError("Moment observer projector and identity are required.")
        self.projector = projector
        self.observer_id = canonical_fingerprint(
            {"kind": "iteration-moments", "observer_id": observer_id_}
        )

    def _value(self, record: IterationRecord, /) -> PyTree[Any]:
        value = jax.tree.map(jnp.asarray, self.projector(record))
        for leaf in jax.tree.leaves(value):
            if not (
                jnp.issubdtype(leaf.dtype, jnp.floating)
                or jnp.issubdtype(leaf.dtype, jnp.integer)
                or jnp.issubdtype(leaf.dtype, jnp.bool_)
            ):
                raise TypeError("Moment observer projections must be real-valued.")
        return jax.tree.map(
            lambda leaf: jnp.asarray(leaf, dtype=jnp.result_type(leaf, jnp.float32)),
            value,
        )

    def initialize(self, initial: IterationRecord, /) -> IterationMomentState:
        value = self._value(initial)
        zero = jax.tree.map(jnp.zeros_like, value)
        minimum = jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.inf), value)
        maximum = jax.tree.map(lambda leaf: jnp.full_like(leaf, -jnp.inf), value)
        return IterationMomentState(
            jnp.asarray(0, dtype=jnp.int32), zero, minimum, maximum, zero
        )

    def update(
        self, state: IterationMomentState, record: IterationRecord, /
    ) -> IterationMomentState:
        value = self._value(record)
        return IterationMomentState(
            state.count + 1,
            jax.tree.map(jnp.add, state.total, value),
            jax.tree.map(jnp.minimum, state.minimum, value),
            jax.tree.map(jnp.maximum, state.maximum, value),
            value,
        )

    def finalize(
        self,
        state: IterationMomentState,
        initial: IterationRecord,
        terminal: IterationRecord,
        /,
    ) -> IterationMoments:
        del initial, terminal
        denominator = jnp.maximum(state.count, 1)
        mean = jax.tree.map(lambda leaf: leaf / denominator, state.total)
        return IterationMoments(
            state.count,
            state.total,
            mean,
            state.minimum,
            state.maximum,
            state.last,
        )


class CallableIterationObserver(AbstractIterationObserver):
    """Adapt explicit pure initialize/update/finalize functions."""

    initializer: Callable = eqx.field(static=True)
    updater: Callable = eqx.field(static=True)
    finalizer: Callable = eqx.field(static=True)
    observer_id: str = eqx.field(static=True)

    def __init__(
        self,
        initializer: Callable,
        updater: Callable,
        finalizer: Callable,
        observer_id: str,
        /,
    ):
        observer_id_ = str(observer_id)
        if (
            not callable(initializer)
            or not callable(updater)
            or not callable(finalizer)
            or not observer_id_
        ):
            raise ValueError("Callable observer functions and identity are required.")
        self.initializer = initializer
        self.updater = updater
        self.finalizer = finalizer
        self.observer_id = canonical_fingerprint(
            {"kind": "callable-iteration-observer", "observer_id": observer_id_}
        )

    def initialize(self, initial: IterationRecord, /) -> PyTree[Any]:
        return self.initializer(initial)

    def update(self, state: PyTree[Any], record: IterationRecord, /) -> PyTree[Any]:
        return self.updater(state, record)

    def finalize(
        self,
        state: PyTree[Any],
        initial: IterationRecord,
        terminal: IterationRecord,
        /,
    ) -> PyTree[Any]:
        return self.finalizer(state, initial, terminal)


class CallableIterationStopRule(AbstractIterationStopRule):
    """Adapt explicit pure state initialization and stopping functions."""

    initializer: Callable = eqx.field(static=True)
    evaluator: Callable = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        initializer: Callable,
        evaluator: Callable,
        rule_id: str,
        /,
    ):
        rule_id_ = str(rule_id)
        if not callable(initializer) or not callable(evaluator) or not rule_id_:
            raise ValueError("Callable stop-rule functions and identity are required.")
        self.initializer = initializer
        self.evaluator = evaluator
        self.rule_id = canonical_fingerprint(
            {"kind": "callable-iteration-stop-rule", "rule_id": rule_id_}
        )

    def initialize(self, initial: IterationRecord, /) -> PyTree[Any]:
        return self.initializer(initial)

    def evaluate(
        self, state: PyTree[Any], record: IterationRecord, /
    ) -> tuple[PyTree[Any], IterationDecision]:
        next_state, stop = self.evaluator(state, record)
        return next_state, IterationDecision(stop)


class IterationChildPlan(StrictModule, NonTrainableState):
    role: str = eqx.field(static=True)
    plan: IterationPlan

    def __init__(self, role: str, plan: IterationPlan, /):
        role_ = str(role)
        if not role_ or not isinstance(plan, IterationPlan):
            raise ValueError("Iteration child plans require a role and plan.")
        self.role = role_
        self.plan = plan


class IterationPlan(StrictModule, NonTrainableState):
    """Static pure observation and device-control plan for one scope."""

    granularity: IterationGranularity = eqx.field(static=True)
    observers: tuple[AbstractIterationObserver, ...]
    stop_rule: AbstractIterationStopRule | None
    children: tuple[IterationChildPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        granularity: IterationGranularity = "step",
        observers: Sequence[AbstractIterationObserver] = (),
        stop_rule: AbstractIterationStopRule | None = None,
        children: Sequence[IterationChildPlan] = (),
    ):
        granularity_ = str(granularity)
        if granularity_ not in _GRANULARITIES:
            raise ValueError("Iteration plan granularity is invalid.")
        observers_ = tuple(observers)
        if any(not isinstance(value, AbstractIterationObserver) for value in observers_):
            raise TypeError(
                "Iteration observers must implement AbstractIterationObserver."
            )
        observer_ids = tuple(value.observer_id for value in observers_)
        if len(observer_ids) != len(set(observer_ids)):
            raise ValueError("Iteration observer identities must be unique per scope.")
        if stop_rule is not None and not isinstance(stop_rule, AbstractIterationStopRule):
            raise TypeError("stop_rule must implement AbstractIterationStopRule.")
        children_ = tuple(children)
        if any(not isinstance(value, IterationChildPlan) for value in children_):
            raise TypeError("children must contain IterationChildPlan values.")
        child_roles = tuple(value.role for value in children_)
        if len(child_roles) != len(set(child_roles)):
            raise ValueError("Iteration child roles must be unique per scope.")
        self.granularity = granularity_
        self.observers = observers_
        self.stop_rule = stop_rule
        self.children = children_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iteration-plan",
                "granularity": granularity_,
                "observer_ids": observer_ids,
                "stop_rule_id": None if stop_rule is None else stop_rule.rule_id,
                "children": [
                    {"role": value.role, "plan_id": value.plan.plan_id}
                    for value in children_
                ],
            }
        )

    def child(self, role: str, /) -> IterationPlan | None:
        role_ = str(role)
        for child in self.children:
            if child.role == role_:
                return child.plan
        return None


class IterationRuntimeState(StrictModule):
    observer_states: tuple[PyTree[Any], ...]
    stop_state: PyTree[Any] | None
    stop_requested: Array
    initial: IterationRecord
    last: IterationRecord


class IterationEvidence(StrictModule):
    """Final pure evidence from one observed iterative scope."""

    scope: IterationScope = eqx.field(static=True)
    capabilities: IterationCapabilities = eqx.field(static=True)
    observer_outputs: tuple[PyTree[Any], ...]
    terminal: IterationRecord
    stop_requested: Array
    plan_id: str = eqx.field(static=True)


def bind_iteration_scope(
    plan: IterationPlan,
    capabilities: IterationCapabilities,
    algorithm_id: str,
    /,
    *,
    role: str = "root",
    parent: IterationScope | None = None,
) -> IterationScope:
    """Validate a plan and derive a deterministic static scope identity."""
    if not isinstance(plan, IterationPlan):
        raise TypeError("plan must be IterationPlan.")
    if not isinstance(capabilities, IterationCapabilities):
        raise TypeError("capabilities must be IterationCapabilities.")
    if plan.granularity not in capabilities.granularities:
        raise ValueError(
            f"Iteration granularity {plan.granularity!r} is unsupported; "
            f"supported granularities are {capabilities.granularities}."
        )
    if plan.stop_rule is not None and not capabilities.device_stop:
        raise ValueError("This iterative owner does not support device-side stopping.")
    unsupported_children = tuple(
        child.role
        for child in plan.children
        if child.role not in capabilities.child_roles
    )
    if unsupported_children:
        raise ValueError(
            f"Unsupported iteration child roles: {unsupported_children}; "
            f"supported roles are {capabilities.child_roles}."
        )
    algorithm_id_ = str(algorithm_id)
    role_ = str(role)
    if not algorithm_id_ or not role_:
        raise ValueError("Iteration algorithm and role identities are required.")
    parent_id = None if parent is None else parent.scope_id
    depth = 0 if parent is None else parent.depth + 1
    scope_id = canonical_fingerprint(
        {
            "kind": "iteration-scope",
            "plan_id": plan.plan_id,
            "algorithm_id": algorithm_id_,
            "role": role_,
            "parent_scope_id": parent_id,
        }
    )
    return IterationScope(
        scope_id,
        parent_scope_id=parent_id,
        role=role_,
        algorithm_id=algorithm_id_,
        depth=depth,
    )


def initialize_iteration(
    plan: IterationPlan, initial: IterationRecord, /
) -> IterationRuntimeState:
    """Initialize pure observer and control state from one initial record."""
    initial_ = _stop_gradient_tree(initial)
    observer_states = tuple(observer.initialize(initial_) for observer in plan.observers)
    stop_state = None if plan.stop_rule is None else plan.stop_rule.initialize(initial_)
    return IterationRuntimeState(
        observer_states,
        stop_state,
        jnp.asarray(False),
        initial_,
        initial_,
    )


def update_iteration(
    plan: IterationPlan,
    state: IterationRuntimeState,
    record: IterationRecord,
    /,
    *,
    allow_stop: ArrayLike = True,
) -> IterationRuntimeState:
    """Update observers and optional stop rule without side effects."""
    record_ = _stop_gradient_tree(record)
    active = jnp.any(record_.coordinates.active)
    observer_states: list[PyTree[Any]] = []
    for observer, observer_state in zip(
        plan.observers, state.observer_states, strict=True
    ):
        candidate = jax.lax.cond(
            active,
            lambda operands, observer=observer: observer.update(operands[0], operands[1]),
            lambda operands: operands[0],
            (observer_state, record_),
        )
        observer_states.append(candidate)
    stop_state = state.stop_state
    stop_requested = state.stop_requested
    stop_rule = plan.stop_rule
    if stop_rule is not None:
        evaluate = active & jnp.asarray(allow_stop, dtype=bool) & ~stop_requested
        candidate_state, decision = jax.lax.cond(
            evaluate,
            lambda operands: stop_rule.evaluate(operands[0], operands[1]),
            lambda operands: (operands[0], IterationDecision(False)),
            (stop_state, record_),
        )
        stop_state = candidate_state
        stop_requested = stop_requested | decision.stop
    return IterationRuntimeState(
        tuple(observer_states),
        stop_state,
        stop_requested,
        state.initial,
        _tree_select(active, record_, state.last),
    )


def finalize_iteration(
    plan: IterationPlan,
    scope: IterationScope,
    capabilities: IterationCapabilities,
    state: IterationRuntimeState,
    terminal: IterationRecord,
    /,
) -> IterationEvidence:
    """Finalize one observed scope after its authoritative certification."""
    terminal_ = _stop_gradient_tree(terminal)
    outputs = tuple(
        observer.finalize(observer_state, state.initial, terminal_)
        for observer, observer_state in zip(
            plan.observers, state.observer_states, strict=True
        )
    )
    return IterationEvidence(
        scope,
        capabilities,
        outputs,
        terminal_,
        state.stop_requested,
        plan.plan_id,
    )


@dataclass(frozen=True, slots=True)
class HostIterationEvent:
    """Host-materialized event delivered to sinks and host control."""

    session_id: str
    event_id: str
    sequence: int
    scope: IterationScope
    record: IterationRecord
    process_index: int = 0
    process_count: int = 1
    execution_group_id: str | None = None
    observation_scope: ObservationScope = ObservationScope.LOCAL


class IterationSink(Protocol):
    sink_id: str

    def emit(self, event: HostIterationEvent, /) -> None: ...


class IterationHostControl(Protocol):
    control_id: str

    def stop(self, event: HostIterationEvent, /) -> bool: ...


@dataclass(frozen=True, slots=True)
class CallableIterationSink:
    callback: Callable[[HostIterationEvent], None]
    sink_id: str

    def __post_init__(self) -> None:
        if not callable(self.callback) or not self.sink_id:
            raise ValueError("Callable iteration sinks require a callback and identity.")

    def emit(self, event: HostIterationEvent, /) -> None:
        result = self.callback(event)
        if result is not None:
            raise TypeError("Iteration sinks must not return control values.")


@dataclass(frozen=True, slots=True)
class CallableIterationHostControl:
    callback: Callable[[HostIterationEvent], bool]
    control_id: str

    def __post_init__(self) -> None:
        if not callable(self.callback) or not self.control_id:
            raise ValueError("Host iteration control requires a callback and identity.")

    def stop(self, event: HostIterationEvent, /) -> bool:
        return bool(self.callback(event))


@dataclass(frozen=True, slots=True)
class IterationSessionState:
    session_id: str
    cursor: int
    stop_requested: bool


class IterationSession:
    """Explicit ordered host-delivery session outside transformed execution."""

    __slots__ = (
        "_control",
        "_cursor",
        "_session_id",
        "_sinks",
        "_stop_requested",
        "_observation_policy",
    )

    def __init__(
        self,
        session_id: str,
        /,
        *,
        sinks: Sequence[IterationSink] = (),
        control: IterationHostControl | None = None,
        state: IterationSessionState | None = None,
        observation_policy: DistributedObservationPolicy | None = None,
    ):
        session_id_ = str(session_id)
        sinks_ = tuple(sinks)
        if not session_id_ or any(not sink.sink_id for sink in sinks_):
            raise ValueError("Iteration sessions require stable non-empty identities.")
        if len({sink.sink_id for sink in sinks_}) != len(sinks_):
            raise ValueError("Iteration sink identities must be unique per session.")
        if state is not None and state.session_id != session_id_:
            raise ValueError("Iteration session state belongs to another session.")
        policy = (
            DistributedObservationPolicy()
            if observation_policy is None
            else observation_policy
        )
        if policy.scope not in (
            ObservationScope.LOCAL,
            ObservationScope.ALL_PROCESSES,
            ObservationScope.COORDINATOR,
        ):
            raise ValueError(
                "Iteration records support local, all-process, or coordinator delivery."
            )
        self._session_id = session_id_
        self._sinks = sinks_
        self._control = control
        self._cursor = 0 if state is None else int(state.cursor)
        self._stop_requested = False if state is None else bool(state.stop_requested)
        self._observation_policy = policy

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def sink_ids(self) -> tuple[str, ...]:
        return tuple(sink.sink_id for sink in self._sinks)

    @property
    def control_id(self) -> str | None:
        return None if self._control is None else self._control.control_id

    @property
    def observation_policy(self) -> DistributedObservationPolicy:
        return self._observation_policy

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def snapshot(self) -> IterationSessionState:
        return IterationSessionState(self._session_id, self._cursor, self._stop_requested)

    def restore(self, state: IterationSessionState, /) -> None:
        if not isinstance(state, IterationSessionState):
            raise TypeError("state must be IterationSessionState.")
        if state.session_id != self._session_id:
            raise ValueError("Iteration session state belongs to another session.")
        if self._cursor not in (0, state.cursor) or (
            self._stop_requested and not state.stop_requested
        ):
            raise ValueError("Iteration session has advanced beyond the restored state.")
        self._cursor = int(state.cursor)
        self._stop_requested = bool(state.stop_requested)

    def emit(self, scope: IterationScope, record: IterationRecord, /) -> bool:
        materialized = jax.device_get(record)
        process_index = jax.process_index()
        process_count = jax.process_count()
        policy = self._observation_policy
        deliver = (
            policy.scope in (ObservationScope.LOCAL, ObservationScope.ALL_PROCESSES)
            or process_index == policy.coordinator_process
        )
        event_id = canonical_fingerprint(
            {
                "kind": "host-iteration-event",
                "session_id": self._session_id,
                "scope_id": scope.scope_id,
                "sequence": self._cursor,
                "process_index": process_index,
                "execution_group_id": policy.execution_group_id,
                "observation_scope": policy.scope.value,
            }
        )
        requested = False
        if deliver:
            event = HostIterationEvent(
                self._session_id,
                event_id,
                self._cursor,
                scope,
                materialized,
                process_index,
                process_count,
                policy.execution_group_id,
                policy.scope,
            )
            for sink in self._sinks:
                sink.emit(event)
            if self._control is not None:
                requested = self._control.stop(event)
        if process_count > 1 and policy.scope in (
            ObservationScope.ALL_PROCESSES,
            ObservationScope.COORDINATOR,
        ):
            requested = global_boolean_consensus(requested, require_all=False)
        self._stop_requested = self._stop_requested or requested
        self._cursor += 1
        return self._stop_requested

    def emit_trace(self, scope: IterationScope, trace: IterationTrace, /) -> bool:
        materialized = jax.device_get(trace)
        self.emit(scope, materialized.initial)
        count = int(np.asarray(materialized.stored_count))
        for index in range(count):
            self.emit(
                scope,
                jax.tree.map(lambda leaf, index=index: leaf[index], materialized.records),
            )
        self.emit(scope, materialized.terminal)
        return self._stop_requested


def _stop_gradient_tree(tree: PyTree[Any], /) -> PyTree[Any]:
    return jax.tree.map(
        lambda leaf: jax.lax.stop_gradient(jnp.asarray(leaf)),
        tree,
    )


def _tree_select(
    predicate: ArrayLike,
    when_true: PyTree[Any],
    when_false: PyTree[Any],
    /,
) -> PyTree[Any]:
    predicate_ = jnp.asarray(predicate, dtype=bool)
    return jax.tree.map(
        lambda true, false: jnp.where(predicate_, true, false),
        when_true,
        when_false,
    )


__all__ = [
    "AbstractIterationObserver",
    "AbstractIterationStopRule",
    "CallableIterationHostControl",
    "CallableIterationObserver",
    "CallableIterationSink",
    "CallableIterationStopRule",
    "HostIterationEvent",
    "IterationCapabilities",
    "IterationChildPlan",
    "IterationCoordinates",
    "IterationCountObserver",
    "IterationCounts",
    "IterationDecision",
    "IterationEvidence",
    "IterationGranularity",
    "IterationHostControl",
    "IterationMomentObserver",
    "IterationMoments",
    "IterationPhase",
    "IterationPlan",
    "IterationRecord",
    "IterationRuntimeState",
    "IterationScope",
    "IterationSession",
    "IterationSessionState",
    "IterationSink",
    "IterationTrace",
    "IterationTraceObserver",
    "bind_iteration_scope",
    "finalize_iteration",
    "initialize_iteration",
    "update_iteration",
]
