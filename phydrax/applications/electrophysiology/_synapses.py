#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity multicell synapses, relation events, and pair-based STDP."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


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
            }
        )
        self.kind = int(SynapseKind.CONDUCTANCE)


SynapseModel = CurrentSynapse | ConductanceSynapse


class SynapseConnection(StrictModule, NonTrainableState):
    """Stable initial relation between two fixed-capacity cell compartments."""

    relation_id: str = eqx.field(static=True)
    pre_cell: int = eqx.field(static=True)
    pre_compartment: int = eqx.field(static=True)
    post_cell: int = eqx.field(static=True)
    post_compartment: int = eqx.field(static=True)
    delay_steps: int = eqx.field(static=True)
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
        delay_steps: int = 0,
        weight: float = 1.0,
    ):
        identifier = _identifier(relation_id, "relation_id")
        indices = (pre_cell, pre_compartment, post_cell, post_compartment, delay_steps)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in indices
        ):
            raise TypeError("Synapse endpoints and delay_steps must be integers.")
        if any(value < 0 for value in indices):
            raise ValueError("Synapse endpoints and delay_steps must be nonnegative.")
        if not isinstance(model, (CurrentSynapse, ConductanceSynapse)):
            raise TypeError("model must be a CurrentSynapse or ConductanceSynapse.")
        weight_ = _positive(weight, "weight", allow_zero=True)
        self.relation_id = identifier
        self.pre_cell = pre_cell
        self.pre_compartment = pre_compartment
        self.post_cell = post_cell
        self.post_compartment = post_compartment
        self.delay_steps = delay_steps
        self.weight = weight_
        self.model = model
        self.connection_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-synapse-connection-v1",
                "relation_id": identifier,
                "pre": [pre_cell, pre_compartment],
                "post": [post_cell, post_compartment],
                "delay_steps": delay_steps,
                "weight": weight_,
                "model": model.model_id,
            }
        )


class SynapseNetworkPlan(StrictModule, NonTrainableState):
    """Fixed-capacity multicell relation plan with spare synaptogenesis slots."""

    cell_capacity: int = eqx.field(static=True)
    compartments_per_cell: int = eqx.field(static=True)
    synapse_capacity: int = eqx.field(static=True)
    maximum_delay_steps: int = eqx.field(static=True)
    dt_ms: float = eqx.field(static=True)
    connections: tuple[SynapseConnection, ...]
    slot_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_capacity: int,
        compartments_per_cell: int,
        synapse_capacity: int,
        maximum_delay_steps: int,
        dt_ms: float,
        /,
        *,
        connections: Sequence[SynapseConnection] = (),
    ):
        capacities = (cell_capacity, compartments_per_cell, synapse_capacity)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in capacities
        ):
            raise TypeError("Network capacities must be integers.")
        if any(value <= 0 for value in capacities):
            raise ValueError("Network capacities must be positive.")
        if isinstance(maximum_delay_steps, bool) or not isinstance(
            maximum_delay_steps, int
        ):
            raise TypeError("maximum_delay_steps must be an integer.")
        if maximum_delay_steps < 0:
            raise ValueError("maximum_delay_steps must be nonnegative.")
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
                value.pre_cell >= cell_capacity
                or value.post_cell >= cell_capacity
                or value.pre_compartment >= compartments_per_cell
                or value.post_compartment >= compartments_per_cell
            ):
                raise ValueError("An initial synapse endpoint exceeds network capacity.")
            if value.delay_steps > maximum_delay_steps:
                raise ValueError("An initial synapse delay exceeds maximum_delay_steps.")
        slots = relation_ids + tuple(
            f"reserved-synapse-slot-{index}"
            for index in range(len(values), synapse_capacity)
        )
        if len(set(slots)) != len(slots):
            raise ValueError(
                "Initial relation identifiers collide with reserved slot identities."
            )
        self.cell_capacity = cell_capacity
        self.compartments_per_cell = compartments_per_cell
        self.synapse_capacity = synapse_capacity
        self.maximum_delay_steps = maximum_delay_steps
        self.dt_ms = step
        self.connections = values
        self.slot_ids = slots
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-synapse-network-v1",
                "cell_capacity": cell_capacity,
                "compartments_per_cell": compartments_per_cell,
                "synapse_capacity": synapse_capacity,
                "maximum_delay_steps": maximum_delay_steps,
                "dt_ms": step,
                "connections": [value.connection_id for value in values],
                "slot_ids": list(slots),
            }
        )

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
    """Fixed-capacity relation, activation, delay, and provenance state."""

    active: Array
    pre_cell: Array
    pre_compartment: Array
    post_cell: Array
    post_compartment: Array
    kind: Array
    weight: Array
    scale: Array
    reversal_mV: Array
    time_constant_ms: Array
    delay_steps: Array
    activation: Array
    delay_buffer: Array
    cursor: Array
    relation_version: Array
    step_index: Array


class SynapseNetworkEvidence(StrictModule):
    """Capacity, affinity, conservation, and finiteness evidence."""

    active_count: Array
    capacity_remaining: Array
    total_arrival: Array
    conductance_uS: Array
    current_offset_nA: Array
    finite: Array
    status: Array


class SynapseNetworkCandidate(StrictModule):
    """Uncommitted network propagation transition."""

    proposed: SynapseRelationState
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
    delay_steps: Array
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
        delay_steps: int,
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
            delay_steps,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_values
        ):
            raise TypeError(
                "Relation event kinds, slot, endpoints, and delay must be integers."
            )
        scalar_values = (weight, scale, reversal_mV, time_constant_ms)
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
        self.delay_steps = jnp.asarray(delay_steps, dtype=jnp.int32)
        self.event_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-synapse-relation-event-v1",
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
                "delay_steps": delay_steps,
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


def initialize_synapse_network(
    runtime: PreparedSynapseNetwork, /
) -> SynapseRelationState:
    """Materialize initial relations and zero dynamic activation."""
    plan = runtime.plan
    capacity = plan.synapse_capacity
    count = len(plan.connections)
    active = [index < count for index in range(capacity)]
    padding = capacity - count
    pre_cell = [value.pre_cell for value in plan.connections] + [0] * padding
    pre_compartment = [value.pre_compartment for value in plan.connections] + [
        0
    ] * padding
    post_cell = [value.post_cell for value in plan.connections] + [0] * padding
    post_compartment = [value.post_compartment for value in plan.connections] + [
        0
    ] * padding
    weights = [value.weight for value in plan.connections] + [0.0] * padding
    delays = [value.delay_steps for value in plan.connections] + [0] * padding
    model_values = [_model_arrays(value.model) for value in plan.connections]
    model_values += [(0, 0.0, 0.0, 1.0)] * padding
    dtype = jnp.asarray(0.0).dtype
    return SynapseRelationState(
        jnp.asarray(active),
        jnp.asarray(pre_cell, dtype=jnp.int32),
        jnp.asarray(pre_compartment, dtype=jnp.int32),
        jnp.asarray(post_cell, dtype=jnp.int32),
        jnp.asarray(post_compartment, dtype=jnp.int32),
        jnp.asarray([value[0] for value in model_values], dtype=jnp.int32),
        jnp.asarray(weights, dtype=dtype),
        jnp.asarray([value[1] for value in model_values], dtype=dtype),
        jnp.asarray([value[2] for value in model_values], dtype=dtype),
        jnp.asarray([value[3] for value in model_values], dtype=dtype),
        jnp.asarray(delays, dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=dtype),
        jnp.zeros((plan.maximum_delay_steps + 1, capacity), dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )


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
    endpoint_valid = (
        (event.pre_cell >= 0)
        & (event.pre_cell < plan.cell_capacity)
        & (event.post_cell >= 0)
        & (event.post_cell < plan.cell_capacity)
        & (event.pre_compartment >= 0)
        & (event.pre_compartment < plan.compartments_per_cell)
        & (event.post_compartment >= 0)
        & (event.post_compartment < plan.compartments_per_cell)
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
        & (event.delay_steps >= 0)
        & (event.delay_steps <= plan.maximum_delay_steps)
        & jnp.isfinite(event.weight)
        & jnp.isfinite(event.scale)
        & jnp.isfinite(event.reversal_mV)
        & jnp.isfinite(event.time_constant_ms)
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
        (status == int(SynapseStatus.SUCCESS)) & ~(activating | deactivating),
        int(SynapseStatus.INVALID_PARAMETER),
        status,
    )
    successful = status == int(SynapseStatus.SUCCESS)
    target_active = activating
    proposed = SynapseRelationState(
        state.active.at[safe_slot].set(target_active),
        state.pre_cell.at[safe_slot].set(event.pre_cell),
        state.pre_compartment.at[safe_slot].set(event.pre_compartment),
        state.post_cell.at[safe_slot].set(event.post_cell),
        state.post_compartment.at[safe_slot].set(event.post_compartment),
        state.kind.at[safe_slot].set(event.synapse_kind),
        state.weight.at[safe_slot].set(event.weight),
        state.scale.at[safe_slot].set(event.scale),
        state.reversal_mV.at[safe_slot].set(event.reversal_mV),
        state.time_constant_ms.at[safe_slot].set(event.time_constant_ms),
        state.delay_steps.at[safe_slot].set(event.delay_steps),
        state.activation.at[safe_slot].set(0.0),
        state.delay_buffer.at[:, safe_slot].set(0.0),
        state.cursor,
        state.relation_version.at[safe_slot].add(1),
        state.step_index,
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


def evaluate_synapse_network_transition(
    runtime: PreparedSynapseNetwork,
    state: SynapseRelationState,
    presynaptic_spikes: Array,
    /,
) -> SynapseNetworkCandidate:
    """Evaluate delayed arrivals and exact current/conductance affinity."""
    plan = runtime.plan
    spikes = jnp.asarray(presynaptic_spikes)
    expected = (plan.cell_capacity, plan.compartments_per_cell)
    if spikes.shape != expected:
        raise ValueError(f"presynaptic_spikes must have shape {expected}.")
    active = state.active
    emitted = jnp.where(
        active, state.weight * spikes[state.pre_cell, state.pre_compartment], 0.0
    )
    target_slots = (state.cursor + state.delay_steps) % (plan.maximum_delay_steps + 1)
    scheduled = state.delay_buffer.at[
        target_slots, jnp.arange(plan.synapse_capacity)
    ].add(emitted)
    arrivals = scheduled[state.cursor]
    scheduled = scheduled.at[state.cursor].set(0.0)
    activation = (
        state.activation * jnp.exp(-plan.dt_ms / state.time_constant_ms) + arrivals
    )
    current_relation = jnp.where(
        active & (state.kind == int(SynapseKind.CURRENT)), activation * state.scale, 0.0
    )
    conductance_relation = jnp.where(
        active & (state.kind == int(SynapseKind.CONDUCTANCE)),
        activation * state.scale,
        0.0,
    )
    shape = (plan.cell_capacity, plan.compartments_per_cell)
    conductance = (
        jnp.zeros(shape, dtype=activation.dtype)
        .at[state.post_cell, state.post_compartment]
        .add(conductance_relation)
    )
    offset = (
        jnp.zeros(shape, dtype=activation.dtype)
        .at[state.post_cell, state.post_compartment]
        .add(current_relation - conductance_relation * state.reversal_mV)
    )
    finite = (
        jnp.all(jnp.isfinite(activation))
        & jnp.all(jnp.isfinite(conductance))
        & jnp.all(jnp.isfinite(offset))
        & jnp.all(jnp.isfinite(spikes))
    )
    status = jnp.where(
        finite, int(SynapseStatus.SUCCESS), int(SynapseStatus.NONFINITE)
    ).astype(jnp.int32)
    proposed = SynapseRelationState(
        state.active,
        state.pre_cell,
        state.pre_compartment,
        state.post_cell,
        state.post_compartment,
        state.kind,
        state.weight,
        state.scale,
        state.reversal_mV,
        state.time_constant_ms,
        state.delay_steps,
        activation,
        scheduled,
        (state.cursor + 1) % (plan.maximum_delay_steps + 1),
        state.relation_version,
        state.step_index + 1,
    )
    evidence = SynapseNetworkEvidence(
        jnp.sum(active),
        plan.synapse_capacity - jnp.sum(active),
        jnp.sum(arrivals),
        conductance,
        offset,
        finite,
        status,
    )
    return SynapseNetworkCandidate(proposed, evidence, finite)


def commit_synapse_network_transition(
    candidate: SynapseNetworkCandidate, current: SynapseRelationState, /
) -> SynapseRelationState:
    """Commit a finite propagation candidate or fail closed."""
    return jax.tree.map(
        lambda proposed, prior: jnp.where(candidate.successful, proposed, prior),
        candidate.proposed,
        current,
    )


class PairSTDPPlan(StrictModule, NonTrainableState):
    """Bounded nearest-pair trace plasticity plan."""

    pre_time_constant_ms: float = eqx.field(static=True)
    post_time_constant_ms: float = eqx.field(static=True)
    potentiation: float = eqx.field(static=True)
    depression: float = eqx.field(static=True)
    minimum_weight: float = eqx.field(static=True)
    maximum_weight: float = eqx.field(static=True)
    trace_bound: float = eqx.field(static=True)
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
        self.pre_time_constant_ms = pre_tau
        self.post_time_constant_ms = post_tau
        self.potentiation = plus
        self.depression = minus
        self.minimum_weight = lower
        self.maximum_weight = upper
        self.trace_bound = bound
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pair-stdp-v1",
                "pre_time_constant_ms": pre_tau,
                "post_time_constant_ms": post_tau,
                "potentiation": plus,
                "depression": minus,
                "minimum_weight": lower,
                "maximum_weight": upper,
                "trace_bound": bound,
            }
        )


class PairSTDPState(StrictModule):
    """Fixed-capacity bounded pre/post traces."""

    pre_trace: Array
    post_trace: Array
    step_index: Array


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
    )


def evaluate_pair_stdp(
    runtime: PreparedSynapseNetwork,
    plan: PairSTDPPlan,
    relations: SynapseRelationState,
    state: PairSTDPState,
    presynaptic_spikes: Array,
    postsynaptic_spikes: Array,
    /,
) -> PairSTDPCandidate:
    """Evaluate bounded pair STDP without committing its discrete weight change."""
    expected = (runtime.plan.cell_capacity, runtime.plan.compartments_per_cell)
    pre_spikes = jnp.asarray(presynaptic_spikes)
    post_spikes = jnp.asarray(postsynaptic_spikes)
    if pre_spikes.shape != expected or post_spikes.shape != expected:
        raise ValueError(f"STDP spike arrays must have shape {expected}.")
    pre_event = (
        relations.active * pre_spikes[relations.pre_cell, relations.pre_compartment]
    )
    post_event = (
        relations.active * post_spikes[relations.post_cell, relations.post_compartment]
    )
    decayed_pre = (
        state.pre_trace
        * jnp.exp(-runtime.plan.dt_ms / plan.pre_time_constant_ms)
        * relations.active
    )
    decayed_post = (
        state.post_trace
        * jnp.exp(-runtime.plan.dt_ms / plan.post_time_constant_ms)
        * relations.active
    )
    delta = relations.active * (
        plan.potentiation * post_event * decayed_pre
        - plan.depression * pre_event * decayed_post
    )
    active_weight = jnp.clip(
        relations.weight + delta,
        plan.minimum_weight,
        plan.maximum_weight,
    )
    weight = jnp.where(relations.active, active_weight, relations.weight)
    pre_trace = jnp.where(
        pre_event > 0.0,
        jnp.minimum(pre_event, plan.trace_bound),
        decayed_pre,
    )
    post_trace = jnp.where(
        post_event > 0.0,
        jnp.minimum(post_event, plan.trace_bound),
        decayed_post,
    )
    proposed_relations = SynapseRelationState(
        relations.active,
        relations.pre_cell,
        relations.pre_compartment,
        relations.post_cell,
        relations.post_compartment,
        relations.kind,
        weight,
        relations.scale,
        relations.reversal_mV,
        relations.time_constant_ms,
        relations.delay_steps,
        relations.activation,
        relations.delay_buffer,
        relations.cursor,
        relations.relation_version + (weight != relations.weight).astype(jnp.int32),
        relations.step_index,
    )
    proposed_plasticity = PairSTDPState(pre_trace, post_trace, state.step_index + 1)
    trace_ok = (
        jnp.all(pre_trace <= plan.trace_bound)
        & jnp.all(post_trace <= plan.trace_bound)
        & jnp.all(pre_trace >= 0.0)
        & jnp.all(post_trace >= 0.0)
    )
    weight_ok = jnp.all(
        ~relations.active
        | ((weight >= plan.minimum_weight) & (weight <= plan.maximum_weight))
    )
    spikes_valid = (
        jnp.all(jnp.isfinite(pre_spikes))
        & jnp.all(jnp.isfinite(post_spikes))
        & jnp.all(pre_spikes >= 0.0)
        & jnp.all(post_spikes >= 0.0)
    )
    finite = (
        jnp.all(jnp.isfinite(weight))
        & jnp.all(jnp.isfinite(pre_trace))
        & jnp.all(jnp.isfinite(post_trace))
        & jnp.all(jnp.isfinite(pre_spikes))
        & jnp.all(jnp.isfinite(post_spikes))
    )
    successful = trace_ok & weight_ok & finite & spikes_valid
    status = jnp.where(
        ~finite,
        int(SynapseStatus.NONFINITE),
        jnp.where(
            spikes_valid,
            int(SynapseStatus.SUCCESS),
            int(SynapseStatus.INVALID_PARAMETER),
        ),
    ).astype(jnp.int32)
    evidence = PairSTDPEvidence(
        jnp.sum(weight - relations.weight), trace_ok, weight_ok, finite, status
    )
    return PairSTDPCandidate(
        proposed_relations, proposed_plasticity, evidence, successful
    )


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


def commit_synapse_relation_event_with_plasticity(
    candidate: SynapseRelationEventCandidate,
    relations: SynapseRelationState,
    plasticity: PairSTDPState,
    /,
) -> tuple[SynapseRelationState, PairSTDPState]:
    """Commit a relation event and atomically clear that slot's STDP traces."""
    committed_relations = commit_synapse_relation_event(candidate, relations)
    safe_slot = jnp.clip(
        candidate.resolved_slot,
        0,
        relations.active.shape[0] - 1,
    )
    reset = PairSTDPState(
        plasticity.pre_trace.at[safe_slot].set(0.0),
        plasticity.post_trace.at[safe_slot].set(0.0),
        plasticity.step_index,
    )
    committed_plasticity = jax.tree.map(
        lambda proposed, prior: jnp.where(candidate.successful, proposed, prior),
        reset,
        plasticity,
    )
    return committed_relations, committed_plasticity


__all__ = [
    "ConductanceSynapse",
    "CurrentSynapse",
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
    "SynapseRelationEvent",
    "SynapseRelationEventCandidate",
    "SynapseRelationEventKind",
    "SynapseRelationState",
    "SynapseStatus",
    "commit_pair_stdp",
    "commit_synapse_network_transition",
    "commit_synapse_relation_event",
    "commit_synapse_relation_event_with_plasticity",
    "evaluate_pair_stdp",
    "evaluate_synapse_network_transition",
    "evaluate_synapse_relation_event",
    "initialize_pair_stdp",
    "initialize_synapse_network",
    "prepare_synapse_network",
]
