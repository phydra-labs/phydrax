#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical synaptic transport, relation lifetimes, and transactional learning."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import floor, isclose, isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._units import ELECTROPHYSIOLOGY_UNITS


def _positive(value: float, name: str, /, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool.")
    resolved = float(value)
    valid = resolved >= 0.0 if allow_zero else resolved > 0.0
    if not isfinite(resolved) or not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return resolved


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


class SynapseKind(IntEnum):
    CURRENT = 0
    CONDUCTANCE = 1


class SynapseRelationEventKind(IntEnum):
    ACTIVATE = 1
    DEACTIVATE = 2


class SynapseStatus(IntEnum):
    SUCCESS = 0
    CAPACITY_EXCEEDED = 1
    SLOT_OCCUPIED = 2
    SLOT_EMPTY = 3
    INVALID_ENDPOINT = 4
    INVALID_PARAMETER = 5
    NONFINITE = 6


class CurrentSynapse(StrictModule, NonTrainableState):
    """Exponentially decaying outward-positive current synapse."""

    time_constant_ms: float = eqx.field(static=True)
    current_scale_nA: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    kind: int = eqx.field(static=True)

    def __init__(self, time_constant_ms: float, current_scale_nA: float, /):
        tau = _positive(time_constant_ms, "time_constant_ms")
        if isinstance(current_scale_nA, bool):
            raise TypeError("current_scale_nA must be a real scalar, not bool.")
        scale = float(current_scale_nA)
        if not isfinite(scale):
            raise ValueError("current_scale_nA must be finite.")
        self.time_constant_ms = tau
        self.current_scale_nA = scale
        self.model_id = canonical_fingerprint(
            {
                "kind": "current-synapse-v1",
                "time_constant_ms": tau,
                "current_scale_nA": scale,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )
        self.kind = int(SynapseKind.CURRENT)


class ConductanceSynapse(StrictModule, NonTrainableState):
    """Exponentially decaying conductance synapse with exact voltage affinity."""

    time_constant_ms: float = eqx.field(static=True)
    conductance_scale_uS: float = eqx.field(static=True)
    reversal_mV: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    kind: int = eqx.field(static=True)

    def __init__(
        self, time_constant_ms: float, conductance_scale_uS: float, reversal_mV: float, /
    ):
        tau = _positive(time_constant_ms, "time_constant_ms")
        scale = _positive(conductance_scale_uS, "conductance_scale_uS", allow_zero=True)
        if isinstance(reversal_mV, bool):
            raise TypeError("reversal_mV must be a real scalar, not bool.")
        reversal = float(reversal_mV)
        if not isfinite(reversal):
            raise ValueError("reversal_mV must be finite.")
        self.time_constant_ms = tau
        self.conductance_scale_uS = scale
        self.reversal_mV = reversal
        self.model_id = canonical_fingerprint(
            {
                "kind": "conductance-synapse-v1",
                "time_constant_ms": tau,
                "conductance_scale_uS": scale,
                "reversal_mV": reversal,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )
        self.kind = int(SynapseKind.CONDUCTANCE)


SynapseModel = CurrentSynapse | ConductanceSynapse


class SynapseConnection(StrictModule, NonTrainableState):
    """Stable logical endpoints and a canonical physical transmission delay."""

    relation_id: str = eqx.field(static=True)
    pre_cell: int = eqx.field(static=True)
    pre_compartment: int = eqx.field(static=True)
    post_cell: int = eqx.field(static=True)
    post_compartment: int = eqx.field(static=True)
    delay_ms: float = eqx.field(static=True)
    weight: float = eqx.field(static=True)
    model: SynapseModel
    connection_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation_id: str,
        pre_cell: int,
        pre_compartment: int,
        post_cell: int,
        post_compartment: int,
        model: SynapseModel,
        /,
        *,
        delay_ms: float = 0.0,
        weight: float = 1.0,
    ):
        identifier = _identifier(relation_id, "relation_id")
        indices = (pre_cell, pre_compartment, post_cell, post_compartment)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in indices
        ):
            raise TypeError("Synapse endpoints must be integers.")
        if any(value < 0 for value in indices):
            raise ValueError("Synapse endpoints must be nonnegative.")
        if not isinstance(model, (CurrentSynapse, ConductanceSynapse)):
            raise TypeError("model must be a CurrentSynapse or ConductanceSynapse.")
        self.relation_id = identifier
        self.pre_cell = pre_cell
        self.pre_compartment = pre_compartment
        self.post_cell = post_cell
        self.post_compartment = post_compartment
        self.delay_ms = _positive(delay_ms, "delay_ms", allow_zero=True)
        self.weight = _positive(weight, "weight", allow_zero=True)
        self.model = model
        self.connection_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-synapse-connection",
                "relation_id": identifier,
                "pre": [pre_cell, pre_compartment],
                "post": [post_cell, post_compartment],
                "delay_ms": self.delay_ms,
                "weight": self.weight,
                "model": model.model_id,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )


class SynapseNetworkPlan(StrictModule, NonTrainableState):
    """Fixed relation capacity over heterogeneous, flattened cell compartments.

    ``delay_ms`` is physical in both modes. Clock execution requires each
    connection delay to be an integer multiple of ``dt_ms`` (up to floating-point
    representation error); event execution has no such grid restriction.
    """

    compartment_counts: tuple[int, ...] = eqx.field(static=True)
    synapse_capacity: int = eqx.field(static=True)
    maximum_delay_ms: float = eqx.field(static=True)
    dt_ms: float = eqx.field(static=True)
    execution: str = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    connections: tuple[SynapseConnection, ...]
    slot_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        compartment_counts: tuple[int, ...],
        synapse_capacity: int,
        maximum_delay_ms: float,
        dt_ms: float,
        /,
        *,
        connections: Sequence[SynapseConnection] = (),
        execution: str = "clock",
    ):
        counts = tuple(compartment_counts)
        capacities = (*counts, synapse_capacity)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in capacities
        ):
            raise TypeError("Network capacities must be integers.")
        if not counts or any(value <= 0 for value in capacities):
            raise ValueError("Network capacities must be positive and cells nonempty.")
        if execution not in ("clock", "event"):
            raise ValueError("execution must be 'clock' or 'event'.")
        maximum = _positive(maximum_delay_ms, "maximum_delay_ms", allow_zero=True)
        step = _positive(dt_ms, "dt_ms")
        values = tuple(connections)
        if any(not isinstance(value, SynapseConnection) for value in values):
            raise TypeError("connections must contain only SynapseConnection values.")
        if len(values) > synapse_capacity:
            raise ValueError("Initial connections exceed synapse_capacity.")
        relation_ids = tuple(value.relation_id for value in values)
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError("Initial relation identifiers must be unique.")
        for value in values:
            if (
                value.pre_cell >= len(counts)
                or value.post_cell >= len(counts)
                or value.pre_compartment >= counts[value.pre_cell]
                or value.post_compartment >= counts[value.post_cell]
            ):
                raise ValueError("An initial synapse endpoint exceeds cell capacity.")
            if value.delay_ms > maximum:
                raise ValueError("An initial synapse delay exceeds maximum_delay_ms.")
            if execution == "clock" and not isclose(
                value.delay_ms / step,
                round(value.delay_ms / step),
                rel_tol=8.0 * 2.0**-52,
                abs_tol=0.0,
            ):
                raise ValueError("Clock synapse delay_ms must be a multiple of dt_ms.")
        slots = relation_ids + tuple(
            f"reserved-synapse-slot-{index}"
            for index in range(len(values), synapse_capacity)
        )
        if len(set(slots)) != len(slots):
            raise ValueError(
                "Initial relation identifiers collide with reserved slot identities."
            )
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        self.compartment_counts = counts
        self.synapse_capacity = synapse_capacity
        self.maximum_delay_ms = maximum
        self.dt_ms = step
        self.execution = execution
        self.offsets = tuple(offsets)
        self.connections = values
        self.slot_ids = slots
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-synapse-network",
                "compartment_counts": list(counts),
                "synapse_capacity": synapse_capacity,
                "maximum_delay_ms": maximum,
                "dt_ms": step,
                "execution": execution,
                "connections": [value.connection_id for value in values],
                "slot_ids": list(slots),
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )

    @property
    def cell_capacity(self) -> int:
        return len(self.compartment_counts)

    @property
    def endpoint_count(self) -> int:
        return self.offsets[-1]

    def prepare(self) -> PreparedSynapseNetwork:
        return prepare_synapse_network(self)


class PreparedSynapseNetwork(StrictModule, NonTrainableState):
    """Prepared network identity and immutable fixed-capacity dimensions."""

    plan: SynapseNetworkPlan
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: SynapseNetworkPlan, /):
        self.plan = plan
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-electrophysiology-synapse-network-v1",
                "plan": plan.plan_id,
            }
        )


class SynapseRelationState(StrictModule):
    """Physical relation state, independent of the delivery engine.

    ``generation`` changes only on activation/deactivation; ``relation_version``
    also changes on weight updates. In-flight events use the former identity.
    """

    active: Array
    pre_cell: Array
    pre_compartment: Array
    post_cell: Array
    post_compartment: Array
    pre_endpoint: Array
    post_endpoint: Array
    kind: Array
    weight: Array
    scale: Array
    reversal_mV: Array
    time_constant_ms: Array
    delay_ms: Array
    activation: Array
    relation_version: Array
    generation: Array


class SynapseTransportState(StrictModule):
    """Clock rings of weight-at-emission amplitudes and unweighted event counts."""

    delay_buffer: Array
    cursor: Array
    event_count_buffer: Array


class SynapseNetworkState(StrictModule):
    """Standalone clock state; event engines compose relations with their queue."""

    relations: SynapseRelationState
    transport: SynapseTransportState
    step_index: Array


class SynapseNetworkEvidence(StrictModule):
    """Capacity, affinity, conservation, and finiteness evidence."""

    active_count: Array
    capacity_remaining: Array
    total_arrival: Array
    arrival_counts: Array
    conductance_uS: Array
    current_offset_nA: Array
    finite: Array
    status: Array


class SynapseNetworkCandidate(StrictModule):
    """Uncommitted network propagation transition."""

    proposed: SynapseNetworkState
    evidence: SynapseNetworkEvidence
    successful: Array


class SynapseRelationEvent(StrictModule):
    """Dynamic activation/deactivation request for one fixed relation slot.

    ``slot == -1`` requests deterministic allocation of the lowest inactive slot.
    Model parameters are ignored for deactivation events.
    """

    kind: Array
    slot: Array
    pre_cell: Array
    pre_compartment: Array
    post_cell: Array
    post_compartment: Array
    synapse_kind: Array
    weight: Array
    scale: Array
    reversal_mV: Array
    time_constant_ms: Array
    delay_ms: Array
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: int,
        slot: int,
        pre_cell: int,
        pre_compartment: int,
        post_cell: int,
        post_compartment: int,
        synapse_kind: int,
        weight: float,
        scale: float,
        reversal_mV: float,
        time_constant_ms: float,
        delay_ms: float,
        /,
        *,
        event_id: str = "synapse-relation-event",
    ):
        integer_values = (
            kind,
            slot,
            pre_cell,
            pre_compartment,
            post_cell,
            post_compartment,
            synapse_kind,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_values
        ):
            raise TypeError("Relation event kinds, slot, and endpoints must be integers.")
        scalar_values = (weight, scale, reversal_mV, time_constant_ms, delay_ms)
        if any(
            isinstance(value, bool) or not isfinite(float(value))
            for value in scalar_values
        ):
            raise ValueError("Relation event parameters must be finite real scalars.")
        name = _identifier(event_id, "event_id")
        self.kind = jnp.asarray(kind, dtype=jnp.int32)
        self.slot = jnp.asarray(slot, dtype=jnp.int32)
        self.pre_cell = jnp.asarray(pre_cell, dtype=jnp.int32)
        self.pre_compartment = jnp.asarray(pre_compartment, dtype=jnp.int32)
        self.post_cell = jnp.asarray(post_cell, dtype=jnp.int32)
        self.post_compartment = jnp.asarray(post_compartment, dtype=jnp.int32)
        self.synapse_kind = jnp.asarray(synapse_kind, dtype=jnp.int32)
        self.weight = jnp.asarray(weight)
        self.scale = jnp.asarray(scale)
        self.reversal_mV = jnp.asarray(reversal_mV)
        self.time_constant_ms = jnp.asarray(time_constant_ms)
        self.delay_ms = jnp.asarray(float(delay_ms))
        self.event_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-synapse-relation-event",
                "name": name,
                "event_kind": kind,
                "slot": slot,
                "pre": [pre_cell, pre_compartment],
                "post": [post_cell, post_compartment],
                "synapse_kind": synapse_kind,
                "weight": float(weight),
                "scale": float(scale),
                "reversal_mV": float(reversal_mV),
                "time_constant_ms": float(time_constant_ms),
                "delay_ms": float(delay_ms),
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )


class SynapseRelationEventCandidate(StrictModule):
    """Uncommitted discrete relation change with explicit acceptance evidence."""

    proposed: SynapseRelationState
    resolved_slot: Array
    status: Array
    successful: Array
    active_before: Array
    active_after: Array


def _model_arrays(model: SynapseModel, /) -> tuple[int, float, float, float]:
    if isinstance(model, CurrentSynapse):
        return model.kind, model.current_scale_nA, 0.0, model.time_constant_ms
    return (
        model.kind,
        model.conductance_scale_uS,
        model.reversal_mV,
        model.time_constant_ms,
    )


def prepare_synapse_network(plan: SynapseNetworkPlan, /) -> PreparedSynapseNetwork:
    if not isinstance(plan, SynapseNetworkPlan):
        raise TypeError("plan must be a SynapseNetworkPlan.")
    return PreparedSynapseNetwork(plan)


def initialize_synapse_network(runtime: PreparedSynapseNetwork, /) -> SynapseNetworkState:
    """Initialize relations and clock transport (empty storage in event mode)."""
    plan = runtime.plan
    capacity = plan.synapse_capacity
    count = len(plan.connections)
    padding = capacity - count
    pre_cell = [value.pre_cell for value in plan.connections] + [0] * padding
    pre_compartment = [value.pre_compartment for value in plan.connections] + [
        0
    ] * padding
    post_cell = [value.post_cell for value in plan.connections] + [0] * padding
    post_compartment = [value.post_compartment for value in plan.connections] + [
        0
    ] * padding
    pre_endpoint = [
        plan.offsets[cell] + compartment
        for cell, compartment in zip(pre_cell, pre_compartment, strict=True)
    ]
    post_endpoint = [
        plan.offsets[cell] + compartment
        for cell, compartment in zip(post_cell, post_compartment, strict=True)
    ]
    model_values = [_model_arrays(value.model) for value in plan.connections]
    model_values += [(0, 0.0, 0.0, 1.0)] * padding
    dtype = jnp.asarray(0.0).dtype
    relations = SynapseRelationState(
        active=jnp.arange(capacity) < count,
        pre_cell=jnp.asarray(pre_cell, dtype=jnp.int32),
        pre_compartment=jnp.asarray(pre_compartment, dtype=jnp.int32),
        post_cell=jnp.asarray(post_cell, dtype=jnp.int32),
        post_compartment=jnp.asarray(post_compartment, dtype=jnp.int32),
        pre_endpoint=jnp.asarray(pre_endpoint, dtype=jnp.int32),
        post_endpoint=jnp.asarray(post_endpoint, dtype=jnp.int32),
        kind=jnp.asarray([value[0] for value in model_values], dtype=jnp.int32),
        weight=jnp.asarray(
            [value.weight for value in plan.connections] + [0.0] * padding, dtype=dtype
        ),
        scale=jnp.asarray([value[1] for value in model_values], dtype=dtype),
        reversal_mV=jnp.asarray([value[2] for value in model_values], dtype=dtype),
        time_constant_ms=jnp.asarray([value[3] for value in model_values], dtype=dtype),
        delay_ms=jnp.asarray(
            [value.delay_ms for value in plan.connections] + [0.0] * padding, dtype=dtype
        ),
        activation=jnp.zeros((capacity,), dtype=dtype),
        relation_version=jnp.zeros((capacity,), dtype=jnp.int32),
        generation=jnp.zeros((capacity,), dtype=jnp.int32),
    )
    # Correct only floating representation error at an integral upper bound.
    ratio = plan.maximum_delay_ms / plan.dt_ms
    integral = round(ratio)
    if isclose(ratio, integral, rel_tol=8.0 * 2.0**-52, abs_tol=0.0):
        ratio = float(integral)
    rows = floor(ratio) + 1 if plan.execution == "clock" else 0
    transport = SynapseTransportState(
        jnp.zeros((rows, capacity), dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((rows, capacity), dtype=dtype),
    )
    return SynapseNetworkState(relations, transport, jnp.asarray(0, dtype=jnp.int32))


def _clock_delay_valid(delay_ms: Array, dt_ms: float, /) -> Array:
    ratio = delay_ms / dt_ms
    tolerance = 8.0 * jnp.finfo(ratio.dtype).eps * jnp.abs(ratio)
    return jnp.abs(ratio - jnp.rint(ratio)) <= tolerance


def evaluate_synapse_relation_event(
    runtime: PreparedSynapseNetwork,
    state: SynapseRelationState,
    event: SynapseRelationEvent,
    /,
) -> SynapseRelationEventCandidate:
    """Evaluate one dynamic synaptogenesis event without mutating relation state."""
    plan = runtime.plan
    inactive = ~state.active
    has_capacity = jnp.any(inactive)
    first_inactive = jnp.argmax(inactive.astype(jnp.int32))
    requested_allocate = event.slot == -1
    slot = jnp.where(requested_allocate, first_inactive, event.slot)
    slot_in_range = (slot >= 0) & (slot < plan.synapse_capacity)
    safe_slot = jnp.clip(slot, 0, plan.synapse_capacity - 1)
    activating = event.kind == int(SynapseRelationEventKind.ACTIVATE)
    deactivating = event.kind == int(SynapseRelationEventKind.DEACTIVATE)
    counts = jnp.asarray(plan.compartment_counts, dtype=jnp.int32)
    offsets = jnp.asarray(plan.offsets, dtype=jnp.int32)
    pre_cell = jnp.clip(event.pre_cell, 0, plan.cell_capacity - 1)
    post_cell = jnp.clip(event.post_cell, 0, plan.cell_capacity - 1)
    endpoint_valid = (
        (event.pre_cell >= 0)
        & (event.pre_cell < plan.cell_capacity)
        & (event.post_cell >= 0)
        & (event.post_cell < plan.cell_capacity)
        & (event.pre_compartment >= 0)
        & (event.pre_compartment < counts[pre_cell])
        & (event.post_compartment >= 0)
        & (event.post_compartment < counts[post_cell])
    )
    kind_valid = (event.synapse_kind >= int(SynapseKind.CURRENT)) & (
        event.synapse_kind <= int(SynapseKind.CONDUCTANCE)
    )
    scale_valid = (event.synapse_kind == int(SynapseKind.CURRENT)) | (event.scale >= 0.0)
    parameters_valid = (
        kind_valid
        & (event.weight >= 0.0)
        & scale_valid
        & (event.time_constant_ms > 0.0)
        & (event.delay_ms >= 0.0)
        & (event.delay_ms <= plan.maximum_delay_ms)
        & jnp.isfinite(event.delay_ms)
        & jnp.isfinite(event.weight)
        & jnp.isfinite(event.scale)
        & jnp.isfinite(event.reversal_mV)
        & jnp.isfinite(event.time_constant_ms)
    )
    if plan.execution == "clock":
        parameters_valid = parameters_valid & _clock_delay_valid(
            event.delay_ms, plan.dt_ms
        )
    slot_active = state.active[safe_slot]
    status = jnp.asarray(int(SynapseStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        requested_allocate & ~has_capacity,
        int(SynapseStatus.CAPACITY_EXCEEDED),
        status,
    )
    status = jnp.where(
        (status == int(SynapseStatus.SUCCESS)) & ~slot_in_range,
        int(SynapseStatus.CAPACITY_EXCEEDED),
        status,
    )
    status = jnp.where(
        (status == int(SynapseStatus.SUCCESS)) & activating & slot_active,
        int(SynapseStatus.SLOT_OCCUPIED),
        status,
    )
    status = jnp.where(
        (status == int(SynapseStatus.SUCCESS)) & deactivating & ~slot_active,
        int(SynapseStatus.SLOT_EMPTY),
        status,
    )
    status = jnp.where(
        (status == int(SynapseStatus.SUCCESS)) & activating & ~endpoint_valid,
        int(SynapseStatus.INVALID_ENDPOINT),
        status,
    )
    status = jnp.where(
        (status == int(SynapseStatus.SUCCESS)) & activating & ~parameters_valid,
        int(SynapseStatus.INVALID_PARAMETER),
        status,
    )
    status = jnp.where(
        (status == int(SynapseStatus.SUCCESS))
        & (~(activating | deactivating) | (requested_allocate & deactivating)),
        int(SynapseStatus.INVALID_PARAMETER),
        status,
    )
    successful = status == int(SynapseStatus.SUCCESS)
    proposed = SynapseRelationState(
        active=state.active.at[safe_slot].set(activating),
        pre_cell=state.pre_cell.at[safe_slot].set(
            jnp.where(activating, event.pre_cell, 0)
        ),
        pre_compartment=state.pre_compartment.at[safe_slot].set(
            jnp.where(activating, event.pre_compartment, 0)
        ),
        post_cell=state.post_cell.at[safe_slot].set(
            jnp.where(activating, event.post_cell, 0)
        ),
        post_compartment=state.post_compartment.at[safe_slot].set(
            jnp.where(activating, event.post_compartment, 0)
        ),
        pre_endpoint=state.pre_endpoint.at[safe_slot].set(
            jnp.where(activating, offsets[pre_cell] + event.pre_compartment, 0)
        ),
        post_endpoint=state.post_endpoint.at[safe_slot].set(
            jnp.where(activating, offsets[post_cell] + event.post_compartment, 0)
        ),
        kind=state.kind.at[safe_slot].set(jnp.where(activating, event.synapse_kind, 0)),
        weight=state.weight.at[safe_slot].set(jnp.where(activating, event.weight, 0.0)),
        scale=state.scale.at[safe_slot].set(jnp.where(activating, event.scale, 0.0)),
        reversal_mV=state.reversal_mV.at[safe_slot].set(
            jnp.where(activating, event.reversal_mV, 0.0)
        ),
        time_constant_ms=state.time_constant_ms.at[safe_slot].set(
            jnp.where(activating, event.time_constant_ms, 1.0)
        ),
        delay_ms=state.delay_ms.at[safe_slot].set(
            jnp.where(activating, event.delay_ms, 0.0)
        ),
        activation=state.activation.at[safe_slot].set(0.0),
        relation_version=state.relation_version.at[safe_slot].add(1),
        generation=state.generation.at[safe_slot].add(1),
    )
    return SynapseRelationEventCandidate(
        proposed,
        slot,
        status,
        successful,
        jnp.sum(state.active),
        jnp.sum(proposed.active),
    )


def commit_synapse_relation_event(
    candidate: SynapseRelationEventCandidate, current: SynapseRelationState, /
) -> SynapseRelationState:
    """Commit a validated discrete relation change, otherwise preserve all state."""
    return jax.tree.map(
        lambda proposed, prior: jnp.where(candidate.successful, proposed, prior),
        candidate.proposed,
        current,
    )


def commit_synapse_network_relation_event(
    candidate: SynapseRelationEventCandidate, current: SynapseNetworkState, /
) -> SynapseNetworkState:
    """Commit a structural clock change and cancel all pending slot deliveries."""
    slot = jnp.clip(candidate.resolved_slot, 0, current.relations.active.shape[0] - 1)
    transport = SynapseTransportState(
        current.transport.delay_buffer.at[:, slot].set(0.0),
        current.transport.cursor,
        current.transport.event_count_buffer.at[:, slot].set(0.0),
    )
    proposed = SynapseNetworkState(candidate.proposed, transport, current.step_index)
    return jax.tree.map(
        lambda new, prior: jnp.where(candidate.successful, new, prior), proposed, current
    )


def decay_synapse_relations(
    relations: SynapseRelationState, elapsed_ms: Array | float, /
) -> SynapseRelationState:
    """Exactly decay activation over a nonnegative elapsed physical interval."""
    activation = jnp.where(
        relations.active,
        relations.activation * jnp.exp(-elapsed_ms / relations.time_constant_ms),
        0.0,
    )
    return eqx.tree_at(lambda value: value.activation, relations, activation)


def apply_synapse_arrivals(
    relations: SynapseRelationState, amplitudes: Array, /
) -> SynapseRelationState:
    """Add weight-at-emission amplitudes; never resample a relation's weight."""
    arrivals = jnp.asarray(amplitudes)
    if arrivals.shape != relations.activation.shape:
        raise ValueError("amplitudes must have shape (synapse_capacity,).")
    activation = jnp.where(relations.active, relations.activation + arrivals, 0.0)
    return eqx.tree_at(lambda value: value.activation, relations, activation)


def synapse_drive(
    runtime: PreparedSynapseNetwork,
    relations: SynapseRelationState,
    voltage_mV: Array,
    /,
) -> tuple[Array, Array]:
    """Return flat affine outward-current coefficients: ``I = g * V + offset``."""
    shape = (runtime.plan.endpoint_count,)
    if jnp.shape(voltage_mV) != shape:
        raise ValueError(f"voltage_mV must have shape {shape}.")
    strength = relations.activation * relations.scale
    conductance_relation = jnp.where(
        relations.active & (relations.kind == int(SynapseKind.CONDUCTANCE)),
        strength,
        0.0,
    )
    current_relation = jnp.where(
        relations.active & (relations.kind == int(SynapseKind.CURRENT)),
        strength,
        0.0,
    )
    conductance = (
        jnp.zeros(shape, dtype=strength.dtype)
        .at[relations.post_endpoint]
        .add(conductance_relation)
    )
    offset = (
        jnp.zeros(shape, dtype=strength.dtype)
        .at[relations.post_endpoint]
        .add(current_relation - conductance_relation * relations.reversal_mV)
    )
    return conductance, offset


def evaluate_synapse_network_transition(
    runtime: PreparedSynapseNetwork,
    state: SynapseNetworkState,
    presynaptic_spikes: Array,
    /,
) -> SynapseNetworkCandidate:
    """Evaluate one clock tick: schedule emissions, decay, then apply due arrivals.

    Zero-delay emissions arrive in this tick; an integral delay of ``k * dt_ms``
    arrives ``k`` ticks later. Counts are transported independently of weights,
    including for zero-weight relations, for arrival-time learning.
    """
    plan = runtime.plan
    if plan.execution != "clock":
        raise ValueError("Standalone clock transitions require execution='clock'.")
    spikes = jnp.asarray(presynaptic_spikes)
    expected = (plan.endpoint_count,)
    if spikes.shape != expected:
        raise ValueError(f"presynaptic_spikes must have shape {expected}.")
    relations, transport = state.relations, state.transport
    counts = jnp.where(relations.active, spikes[relations.pre_endpoint], 0.0)
    emitted = relations.weight * counts
    rows = transport.delay_buffer.shape[0]
    delays = jnp.rint(relations.delay_ms / plan.dt_ms).astype(jnp.int32)
    targets = (transport.cursor + delays) % rows
    indices = jnp.arange(plan.synapse_capacity)
    scheduled = transport.delay_buffer.at[targets, indices].add(emitted)
    scheduled_counts = transport.event_count_buffer.at[targets, indices].add(counts)
    arrivals = jnp.where(relations.active, scheduled[transport.cursor], 0.0)
    arrival_counts = jnp.where(relations.active, scheduled_counts[transport.cursor], 0.0)
    scheduled = scheduled.at[transport.cursor].set(0.0)
    scheduled_counts = scheduled_counts.at[transport.cursor].set(0.0)
    proposed_relations = apply_synapse_arrivals(
        decay_synapse_relations(relations, plan.dt_ms), arrivals
    )
    conductance, offset = synapse_drive(
        runtime, proposed_relations, jnp.zeros(expected, dtype=relations.activation.dtype)
    )
    finite = (
        jnp.all(jnp.isfinite(proposed_relations.activation))
        & jnp.all(jnp.isfinite(conductance))
        & jnp.all(jnp.isfinite(offset))
        & jnp.all(jnp.isfinite(spikes))
        & jnp.all(jnp.isfinite(scheduled))
        & jnp.all(jnp.isfinite(scheduled_counts))
        & jnp.all(jnp.isfinite(arrivals))
        & jnp.all(jnp.isfinite(arrival_counts))
    )
    valid = jnp.all(spikes >= 0.0) & jnp.all(
        ~relations.active
        | (
            _clock_delay_valid(relations.delay_ms, plan.dt_ms)
            & (relations.delay_ms >= 0.0)
            & (relations.delay_ms <= plan.maximum_delay_ms)
        )
    )
    successful = finite & valid
    status = jnp.where(
        ~finite,
        int(SynapseStatus.NONFINITE),
        jnp.where(
            valid, int(SynapseStatus.SUCCESS), int(SynapseStatus.INVALID_PARAMETER)
        ),
    ).astype(jnp.int32)
    proposed = SynapseNetworkState(
        proposed_relations,
        SynapseTransportState(scheduled, (transport.cursor + 1) % rows, scheduled_counts),
        state.step_index + 1,
    )
    evidence = SynapseNetworkEvidence(
        jnp.sum(relations.active),
        plan.synapse_capacity - jnp.sum(relations.active),
        jnp.sum(arrivals),
        arrival_counts,
        conductance,
        offset,
        finite,
        status,
    )
    return SynapseNetworkCandidate(proposed, evidence, successful)


def commit_synapse_network_transition(
    candidate: SynapseNetworkCandidate, current: SynapseNetworkState, /
) -> SynapseNetworkState:
    """Commit a finite propagation candidate or fail closed."""
    return jax.tree.map(
        lambda proposed, prior: jnp.where(candidate.successful, proposed, prior),
        candidate.proposed,
        current,
    )


class PairSTDPPlan(StrictModule, NonTrainableState):
    """Nearest-pair STDP with physical decay and selectable presynaptic timing.

    Simultaneous events pair only with earlier traces, never with one another.
    Soft bounds multiply potentiation/depression by the normalized distance to
    the upper/lower bound raised to ``weight_exponent``.
    """

    pre_time_constant_ms: float = eqx.field(static=True)
    post_time_constant_ms: float = eqx.field(static=True)
    potentiation: float = eqx.field(static=True)
    depression: float = eqx.field(static=True)
    minimum_weight: float = eqx.field(static=True)
    maximum_weight: float = eqx.field(static=True)
    trace_bound: float = eqx.field(static=True)
    pairing: str = eqx.field(static=True)
    weight_dependence: str = eqx.field(static=True)
    weight_exponent: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pre_time_constant_ms: float,
        post_time_constant_ms: float,
        potentiation: float,
        depression: float,
        minimum_weight: float,
        maximum_weight: float,
        /,
        *,
        trace_bound: float = 10.0,
        pairing: str = "emission",
        weight_dependence: str = "additive",
        weight_exponent: float = 1.0,
    ):
        pre_tau = _positive(pre_time_constant_ms, "pre_time_constant_ms")
        post_tau = _positive(post_time_constant_ms, "post_time_constant_ms")
        plus = _positive(potentiation, "potentiation", allow_zero=True)
        minus = _positive(depression, "depression", allow_zero=True)
        lower = _positive(minimum_weight, "minimum_weight", allow_zero=True)
        upper = _positive(maximum_weight, "maximum_weight")
        bound = _positive(trace_bound, "trace_bound")
        if upper < lower:
            raise ValueError("maximum_weight must be at least minimum_weight.")
        if pairing not in ("emission", "arrival"):
            raise ValueError("pairing must be 'emission' or 'arrival'.")
        if weight_dependence not in ("additive", "soft-bound"):
            raise ValueError("weight_dependence must be 'additive' or 'soft-bound'.")
        self.pairing = pairing
        self.weight_dependence = weight_dependence
        self.weight_exponent = _positive(weight_exponent, "weight_exponent")
        self.pre_time_constant_ms = pre_tau
        self.post_time_constant_ms = post_tau
        self.potentiation = plus
        self.depression = minus
        self.minimum_weight = lower
        self.maximum_weight = upper
        self.trace_bound = bound
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pair-stdp",
                "pre_time_constant_ms": pre_tau,
                "post_time_constant_ms": post_tau,
                "potentiation": plus,
                "depression": minus,
                "minimum_weight": lower,
                "maximum_weight": upper,
                "trace_bound": bound,
                "pairing": pairing,
                "weight_dependence": weight_dependence,
                "weight_exponent": self.weight_exponent,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )


class PairSTDPState(StrictModule):
    """Bounded nearest-pair traces tied to relation lifetime generations."""

    pre_trace: Array
    post_trace: Array
    step_index: Array
    generation: Array


class PairSTDPEvidence(StrictModule):
    """Plasticity bound, finiteness, and aggregate update evidence."""

    weight_delta: Array
    trace_bound_satisfied: Array
    weight_bound_satisfied: Array
    finite: Array
    status: Array


class PairSTDPCandidate(StrictModule):
    """Uncommitted plasticity candidate for relation and trace state."""

    relations: SynapseRelationState
    plasticity: PairSTDPState
    evidence: PairSTDPEvidence
    successful: Array


def initialize_pair_stdp(runtime: PreparedSynapseNetwork, /) -> PairSTDPState:
    capacity = runtime.plan.synapse_capacity
    dtype = jnp.asarray(0.0).dtype
    return PairSTDPState(
        jnp.zeros((capacity,), dtype=dtype),
        jnp.zeros((capacity,), dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=jnp.int32),
    )


def _pair_update(
    runtime: PreparedSynapseNetwork,
    plan: PairSTDPPlan,
    relations: SynapseRelationState,
    state: PairSTDPState,
    presynaptic_spikes: Array,
    postsynaptic_spikes: Array,
    elapsed_ms: Array | float | None,
    presynaptic_arrivals: Array | None,
    /,
) -> tuple[Array, PairSTDPState, Array, Array]:
    expected = (runtime.plan.endpoint_count,)
    pre_spikes = jnp.asarray(presynaptic_spikes)
    post_spikes = jnp.asarray(postsynaptic_spikes)
    if pre_spikes.shape != expected or post_spikes.shape != expected:
        raise ValueError(f"STDP spike arrays must have shape {expected}.")
    elapsed = jnp.asarray(runtime.plan.dt_ms if elapsed_ms is None else elapsed_ms)
    if elapsed.shape != ():
        raise ValueError("elapsed_ms must be scalar.")
    finite = (
        jnp.all(jnp.isfinite(pre_spikes))
        & jnp.all(jnp.isfinite(post_spikes))
        & jnp.isfinite(elapsed)
        & jnp.all(jnp.isfinite(state.pre_trace))
        & jnp.all(jnp.isfinite(state.post_trace))
    )
    valid = jnp.all(pre_spikes >= 0.0) & jnp.all(post_spikes >= 0.0) & (elapsed >= 0.0)
    if plan.pairing == "arrival":
        if presynaptic_arrivals is None:
            raise ValueError(
                "Arrival-time STDP requires unweighted presynaptic_arrivals."
            )
        arrivals = jnp.asarray(presynaptic_arrivals)
        if arrivals.shape != (runtime.plan.synapse_capacity,):
            raise ValueError("presynaptic_arrivals must have shape (synapse_capacity,).")
        pre_event = jnp.where(relations.active, arrivals, 0.0)
        finite = finite & jnp.all(jnp.isfinite(arrivals))
        valid = valid & jnp.all(arrivals >= 0.0)
    else:
        pre_event = jnp.where(relations.active, pre_spikes[relations.pre_endpoint], 0.0)
    post_event = jnp.where(relations.active, post_spikes[relations.post_endpoint], 0.0)
    same_lifetime = relations.active & (state.generation == relations.generation)
    decayed_pre = jnp.where(
        same_lifetime,
        state.pre_trace * jnp.exp(-elapsed / plan.pre_time_constant_ms),
        0.0,
    )
    decayed_post = jnp.where(
        same_lifetime,
        state.post_trace * jnp.exp(-elapsed / plan.post_time_constant_ms),
        0.0,
    )
    potentiation = plan.potentiation * post_event * decayed_pre
    depression = plan.depression * pre_event * decayed_post
    if plan.weight_dependence == "soft-bound":
        span = plan.maximum_weight - plan.minimum_weight
        if span == 0.0:
            potentiation = jnp.zeros_like(potentiation)
            depression = jnp.zeros_like(depression)
        else:
            fraction = jnp.clip((relations.weight - plan.minimum_weight) / span, 0.0, 1.0)
            potentiation = potentiation * (1.0 - fraction) ** plan.weight_exponent
            depression = depression * fraction**plan.weight_exponent
    delta = jnp.where(relations.active, potentiation - depression, 0.0)
    pre_trace = jnp.where(
        pre_event > 0.0, jnp.minimum(pre_event, plan.trace_bound), decayed_pre
    )
    post_trace = jnp.where(
        post_event > 0.0, jnp.minimum(post_event, plan.trace_bound), decayed_post
    )
    proposed = PairSTDPState(
        pre_trace, post_trace, state.step_index + 1, relations.generation
    )
    finite = finite & jnp.all(jnp.isfinite(delta))
    return delta, proposed, valid, finite


def _update_synapse_weights(
    plan: PairSTDPPlan, relations: SynapseRelationState, delta: Array, /
) -> SynapseRelationState:
    weight = jnp.where(
        relations.active & (delta != 0.0),
        jnp.clip(relations.weight + delta, plan.minimum_weight, plan.maximum_weight),
        relations.weight,
    )
    version = relations.relation_version + (weight != relations.weight).astype(jnp.int32)
    return eqx.tree_at(
        lambda value: (value.weight, value.relation_version), relations, (weight, version)
    )


def _pair_evidence(
    plan: PairSTDPPlan,
    prior: SynapseRelationState,
    relations: SynapseRelationState,
    state: PairSTDPState,
    valid: Array,
    finite: Array,
    /,
) -> tuple[PairSTDPEvidence, Array]:
    trace_ok = (
        jnp.all(state.pre_trace <= plan.trace_bound)
        & jnp.all(state.post_trace <= plan.trace_bound)
        & jnp.all(state.pre_trace >= 0.0)
        & jnp.all(state.post_trace >= 0.0)
    )
    weight_ok = jnp.all(
        ~relations.active
        | (
            (relations.weight >= plan.minimum_weight)
            & (relations.weight <= plan.maximum_weight)
        )
    )
    finite = (
        finite
        & jnp.all(jnp.isfinite(prior.weight))
        & jnp.all(jnp.isfinite(relations.weight))
        & jnp.all(jnp.isfinite(state.pre_trace))
        & jnp.all(jnp.isfinite(state.post_trace))
    )
    accepted = trace_ok & weight_ok & valid & finite
    status = jnp.where(
        ~finite,
        int(SynapseStatus.NONFINITE),
        jnp.where(
            accepted, int(SynapseStatus.SUCCESS), int(SynapseStatus.INVALID_PARAMETER)
        ),
    ).astype(jnp.int32)
    return PairSTDPEvidence(
        jnp.sum(relations.weight - prior.weight), trace_ok, weight_ok, finite, status
    ), accepted


def evaluate_pair_stdp(
    runtime: PreparedSynapseNetwork,
    plan: PairSTDPPlan,
    relations: SynapseRelationState,
    state: PairSTDPState,
    presynaptic_spikes: Array,
    postsynaptic_spikes: Array,
    /,
    *,
    elapsed_ms: Array | float | None = None,
    presynaptic_arrivals: Array | None = None,
) -> PairSTDPCandidate:
    """Evaluate a pair update; events occur after the supplied elapsed interval.

    Endpoint spikes are flat. Arrival-time pairing consumes per-relation event
    counts, not weighted amplitudes; this also permits learning at zero weight.
    """
    delta, plasticity, valid, finite = _pair_update(
        runtime,
        plan,
        relations,
        state,
        presynaptic_spikes,
        postsynaptic_spikes,
        elapsed_ms,
        presynaptic_arrivals,
    )
    proposed = _update_synapse_weights(plan, relations, delta)
    evidence, successful = _pair_evidence(
        plan, relations, proposed, plasticity, valid, finite
    )
    return PairSTDPCandidate(proposed, plasticity, evidence, successful)


def commit_pair_stdp(
    candidate: PairSTDPCandidate, relations: SynapseRelationState, state: PairSTDPState, /
) -> tuple[SynapseRelationState, PairSTDPState]:
    """Atomically commit a valid plasticity transition or preserve both states."""
    committed_relations = jax.tree.map(
        lambda proposed, prior: jnp.where(candidate.successful, proposed, prior),
        candidate.relations,
        relations,
    )
    committed_plasticity = jax.tree.map(
        lambda proposed, prior: jnp.where(candidate.successful, proposed, prior),
        candidate.plasticity,
        state,
    )
    return committed_relations, committed_plasticity


class EligibilitySTDPPlan(StrictModule, NonTrainableState):
    """Three-factor plasticity: decaying signed pair eligibility times reward.

    A modulation value is an impulse, not a rate; it is applied once at the
    evaluation boundary, after decay and the current pair events. Zero modulation
    leaves weights unchanged while retaining credit for a delayed reward.
    """

    pair_stdp: PairSTDPPlan
    eligibility_time_constant_ms: float = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    eligibility_bound: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pair_stdp: PairSTDPPlan,
        eligibility_time_constant_ms: float,
        /,
        *,
        learning_rate: float = 1.0,
        eligibility_bound: float = 1.0e6,
    ):
        if not isinstance(pair_stdp, PairSTDPPlan):
            raise TypeError("pair_stdp must be a PairSTDPPlan.")
        self.pair_stdp = pair_stdp
        self.eligibility_time_constant_ms = _positive(
            eligibility_time_constant_ms, "eligibility_time_constant_ms"
        )
        self.learning_rate = _positive(learning_rate, "learning_rate", allow_zero=True)
        self.eligibility_bound = _positive(eligibility_bound, "eligibility_bound")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "eligibility-stdp",
                "pair_stdp": pair_stdp.plan_id,
                "eligibility_time_constant_ms": self.eligibility_time_constant_ms,
                "learning_rate": self.learning_rate,
                "eligibility_bound": self.eligibility_bound,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )


class EligibilitySTDPState(StrictModule):
    """Pair traces and signed eligibility for the same relation generations."""

    pair_state: PairSTDPState
    eligibility: Array


class EligibilitySTDPEvidence(StrictModule):
    """Evidence for an atomic trace, eligibility, and modulated weight update."""

    weight_delta: Array
    trace_bound_satisfied: Array
    eligibility_bound_satisfied: Array
    weight_bound_satisfied: Array
    finite: Array
    status: Array


class EligibilitySTDPCandidate(StrictModule):
    relations: SynapseRelationState
    plasticity: EligibilitySTDPState
    evidence: EligibilitySTDPEvidence
    successful: Array


def initialize_eligibility_stdp(
    runtime: PreparedSynapseNetwork, /
) -> EligibilitySTDPState:
    pair = initialize_pair_stdp(runtime)
    return EligibilitySTDPState(pair, jnp.zeros_like(pair.pre_trace))


def evaluate_eligibility_stdp(
    runtime: PreparedSynapseNetwork,
    plan: EligibilitySTDPPlan,
    relations: SynapseRelationState,
    state: EligibilitySTDPState,
    presynaptic_spikes: Array,
    postsynaptic_spikes: Array,
    /,
    *,
    elapsed_ms: Array | float | None = None,
    presynaptic_arrivals: Array | None = None,
    modulation: Array | float = 0.0,
    modulation_scope: str = "global",
) -> EligibilitySTDPCandidate:
    """Evaluate delayed reward or punishment without committing any coupled state.

    ``global`` accepts a scalar, ``post`` a flat endpoint vector, and ``relation``
    a capacity vector. Values may be signed. For a reward-only event, pass zero
    endpoint spikes and the physical time since the last learning evaluation.
    Eligibility persists after modulation and is erased on relation deletion.
    """
    reward = jnp.asarray(modulation)
    if modulation_scope == "global":
        expected = ()
    elif modulation_scope == "post":
        expected = (runtime.plan.endpoint_count,)
    elif modulation_scope == "relation":
        expected = (runtime.plan.synapse_capacity,)
    else:
        raise ValueError("modulation_scope must be 'global', 'post', or 'relation'.")
    if reward.shape != expected:
        raise ValueError(f"{modulation_scope} modulation must have shape {expected}.")
    projected_reward = (
        reward[relations.post_endpoint] if modulation_scope == "post" else reward
    )
    pair_delta, pair_state, valid, finite = _pair_update(
        runtime,
        plan.pair_stdp,
        relations,
        state.pair_state,
        presynaptic_spikes,
        postsynaptic_spikes,
        elapsed_ms,
        presynaptic_arrivals,
    )
    elapsed = runtime.plan.dt_ms if elapsed_ms is None else elapsed_ms
    same_lifetime = relations.active & (
        state.pair_state.generation == relations.generation
    )
    decayed = jnp.where(
        same_lifetime,
        state.eligibility * jnp.exp(-elapsed / plan.eligibility_time_constant_ms),
        0.0,
    )
    raw_eligibility = decayed + pair_delta
    eligibility = jnp.where(
        relations.active,
        jnp.clip(raw_eligibility, -plan.eligibility_bound, plan.eligibility_bound),
        0.0,
    )
    delta = plan.learning_rate * projected_reward * eligibility
    proposed = _update_synapse_weights(plan.pair_stdp, relations, delta)
    finite = (
        finite
        & jnp.all(jnp.isfinite(reward))
        & jnp.all(jnp.isfinite(state.eligibility))
        & jnp.all(jnp.isfinite(raw_eligibility))
        & jnp.all(jnp.isfinite(delta))
    )
    eligibility_ok = jnp.all(jnp.abs(eligibility) <= plan.eligibility_bound)
    pair_evidence, successful = _pair_evidence(
        plan.pair_stdp, relations, proposed, pair_state, valid & eligibility_ok, finite
    )
    evidence = EligibilitySTDPEvidence(
        pair_evidence.weight_delta,
        pair_evidence.trace_bound_satisfied,
        eligibility_ok,
        pair_evidence.weight_bound_satisfied,
        pair_evidence.finite,
        pair_evidence.status,
    )
    return EligibilitySTDPCandidate(
        proposed, EligibilitySTDPState(pair_state, eligibility), evidence, successful
    )


def commit_eligibility_stdp(
    candidate: EligibilitySTDPCandidate,
    relations: SynapseRelationState,
    state: EligibilitySTDPState,
    /,
) -> tuple[SynapseRelationState, EligibilitySTDPState]:
    """Atomically accept weight, pair traces, and eligibility, or preserve all."""
    return jax.tree.map(
        lambda proposed, prior: jnp.where(candidate.successful, proposed, prior),
        (candidate.relations, candidate.plasticity),
        (relations, state),
    )


def commit_synapse_relation_event_with_plasticity(
    candidate: SynapseRelationEventCandidate,
    relations: SynapseRelationState,
    plasticity: PairSTDPState | EligibilitySTDPState,
    /,
) -> tuple[SynapseRelationState, PairSTDPState | EligibilitySTDPState]:
    """Commit a lifetime change and clear every learning trace for that slot."""
    slot = jnp.clip(candidate.resolved_slot, 0, relations.active.shape[0] - 1)
    pair = (
        plasticity.pair_state
        if isinstance(plasticity, EligibilitySTDPState)
        else plasticity
    )
    reset_pair = PairSTDPState(
        pair.pre_trace.at[slot].set(0.0),
        pair.post_trace.at[slot].set(0.0),
        pair.step_index,
        pair.generation.at[slot].set(candidate.proposed.generation[slot]),
    )
    reset = (
        EligibilitySTDPState(reset_pair, plasticity.eligibility.at[slot].set(0.0))
        if isinstance(plasticity, EligibilitySTDPState)
        else reset_pair
    )
    return jax.tree.map(
        lambda proposed, prior: jnp.where(candidate.successful, proposed, prior),
        (candidate.proposed, reset),
        (relations, plasticity),
    )


def commit_synapse_network_relation_event_with_plasticity(
    candidate: SynapseRelationEventCandidate,
    state: SynapseNetworkState,
    plasticity: PairSTDPState | EligibilitySTDPState,
    /,
) -> tuple[SynapseNetworkState, PairSTDPState | EligibilitySTDPState]:
    """Commit structural clock edits, clearing pending delivery and learning."""
    committed = commit_synapse_network_relation_event(candidate, state)
    _, learning = commit_synapse_relation_event_with_plasticity(
        candidate, state.relations, plasticity
    )
    return committed, learning


__all__ = [
    "ConductanceSynapse",
    "CurrentSynapse",
    "EligibilitySTDPCandidate",
    "EligibilitySTDPEvidence",
    "EligibilitySTDPPlan",
    "EligibilitySTDPState",
    "PairSTDPCandidate",
    "PairSTDPEvidence",
    "PairSTDPPlan",
    "PairSTDPState",
    "PreparedSynapseNetwork",
    "SynapseConnection",
    "SynapseKind",
    "SynapseNetworkCandidate",
    "SynapseNetworkEvidence",
    "SynapseNetworkPlan",
    "SynapseNetworkState",
    "SynapseRelationEvent",
    "SynapseRelationEventCandidate",
    "SynapseRelationEventKind",
    "SynapseRelationState",
    "SynapseStatus",
    "SynapseTransportState",
    "apply_synapse_arrivals",
    "commit_eligibility_stdp",
    "commit_pair_stdp",
    "commit_synapse_network_relation_event",
    "commit_synapse_network_relation_event_with_plasticity",
    "commit_synapse_network_transition",
    "commit_synapse_relation_event",
    "commit_synapse_relation_event_with_plasticity",
    "decay_synapse_relations",
    "evaluate_eligibility_stdp",
    "evaluate_pair_stdp",
    "evaluate_synapse_network_transition",
    "evaluate_synapse_relation_event",
    "initialize_eligibility_stdp",
    "initialize_pair_stdp",
    "initialize_synapse_network",
    "prepare_synapse_network",
    "synapse_drive",
]
