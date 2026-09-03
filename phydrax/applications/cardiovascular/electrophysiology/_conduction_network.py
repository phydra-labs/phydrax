#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity Purkinje propagation with explicit discrete event boundaries."""

from __future__ import annotations

import heapq
from enum import IntEnum, IntFlag
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class PurkinjeEventKind(IntEnum):
    """Semantic outcome stored in a fixed event slot."""

    INACTIVE = 0
    STIMULUS_ACTIVATION = 1
    NETWORK_ACTIVATION = 2
    REFRACTORY_REJECTION = 3
    EDGE_BLOCK = 4
    WAVE_COLLISION = 5


class PurkinjePropagationStatus(IntFlag):
    """Fail-closed propagation status."""

    SUCCESS = 0
    EVENT_CAPACITY_EXCEEDED = 1
    NONFINITE_STATE = 2
    NONMONOTONE_INPUT = 4


class PurkinjeNetworkPlan(StrictModule, NonTrainableState):
    """One immutable graph capacity with stable IDs and physical edge delays.

    ``-1`` is the only inactive stable-ID sentinel.  Inactive edge incidence is
    also ``(-1, -1)``.  This permits a larger compiled capacity than the active
    graph without changing array shapes or entity identity.
    """

    node_ids: Array
    edge_ids: Array
    edge_nodes: Array
    edge_delay_ms: Array
    refractory_period_ms: Array
    event_capacity: int = eqx.field(static=True)
    stimulus_capacity: int = eqx.field(static=True)
    collision_tolerance_ms: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_ids: ArrayLike,
        edge_ids: ArrayLike,
        edge_nodes: ArrayLike,
        edge_delay_ms: ArrayLike,
        refractory_period_ms: ArrayLike | float,
        /,
        *,
        event_capacity: int,
        stimulus_capacity: int | None = None,
        collision_tolerance_ms: float = 1.0e-9,
    ):
        nodes = np.asarray(node_ids, dtype=np.int64)
        edges = np.asarray(edge_ids, dtype=np.int64)
        incidence = np.asarray(edge_nodes, dtype=np.int32)
        delays = np.asarray(edge_delay_ms, dtype=float)
        if nodes.ndim != 1 or nodes.size == 0:
            raise ValueError("node_ids must be a non-empty fixed-capacity vector.")
        if edges.ndim != 1 or incidence.shape != (edges.size, 2):
            raise ValueError("edge_nodes must have shape [edge capacity, 2].")
        if delays.shape != edges.shape:
            raise ValueError("edge_delay_ms must have one value per edge capacity slot.")
        active_nodes = nodes >= 0
        active_edges = edges >= 0
        if np.any(nodes < -1) or np.unique(nodes[active_nodes]).size != np.sum(
            active_nodes
        ):
            raise ValueError(
                "Active node IDs must be unique and -1 is the inactive sentinel."
            )
        if np.any(edges < -1) or np.unique(edges[active_edges]).size != np.sum(
            active_edges
        ):
            raise ValueError(
                "Active edge IDs must be unique and -1 is the inactive sentinel."
            )
        if np.any(incidence[~active_edges] != -1):
            raise ValueError("Inactive edge slots must use incidence (-1, -1).")
        if np.any(incidence[active_edges] < 0) or np.any(
            incidence[active_edges] >= nodes.size
        ):
            raise ValueError("Active edge incidence lies outside node capacity.")
        if np.any(~active_nodes[incidence[active_edges]]):
            raise ValueError("Every active edge endpoint must be an active node.")
        if np.any(incidence[active_edges, 0] == incidence[active_edges, 1]):
            raise ValueError("Purkinje edges may not be self loops.")
        canonical_edges = np.sort(incidence[active_edges], axis=1)
        if np.unique(canonical_edges, axis=0).shape[0] != canonical_edges.shape[0]:
            raise ValueError("Active Purkinje edges must be unique undirected pairs.")
        if not np.all(np.isfinite(delays[active_edges])) or np.any(
            delays[active_edges] <= 0.0
        ):
            raise ValueError(
                "Active edge delays must be finite and positive in milliseconds."
            )
        if np.any(delays[~active_edges] != 0.0):
            raise ValueError("Inactive edge delays must be zero.")
        refractory = np.asarray(refractory_period_ms, dtype=float)
        if refractory.ndim == 0:
            refractory = np.full(nodes.shape, float(refractory))
            refractory[~active_nodes] = 0.0
        if refractory.shape != nodes.shape:
            raise ValueError(
                "refractory_period_ms must be scalar or match node capacity."
            )
        if (
            not np.all(np.isfinite(refractory[active_nodes]))
            or np.any(refractory[active_nodes] < 0.0)
            or np.any(refractory[~active_nodes] != 0.0)
        ):
            raise ValueError("Active refractory periods must be finite and nonnegative.")
        output_capacity = int(event_capacity)
        input_capacity = (
            output_capacity if stimulus_capacity is None else int(stimulus_capacity)
        )
        tolerance = float(collision_tolerance_ms)
        if output_capacity <= 0 or input_capacity <= 0:
            raise ValueError("Event and stimulus capacities must be positive.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("collision_tolerance_ms must be finite and nonnegative.")
        self.node_ids = jnp.asarray(nodes)
        self.edge_ids = jnp.asarray(edges)
        self.edge_nodes = jnp.asarray(incidence)
        self.edge_delay_ms = jnp.asarray(delays)
        self.refractory_period_ms = jnp.asarray(refractory)
        self.event_capacity = output_capacity
        self.stimulus_capacity = input_capacity
        self.collision_tolerance_ms = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-fixed-capacity-purkinje-network",
                "arrays": array_tree_fingerprint(
                    (nodes, edges, incidence, delays, refractory)
                ),
                "event_capacity": output_capacity,
                "stimulus_capacity": input_capacity,
                "collision_tolerance_ms": tolerance,
            }
        )


class PurkinjeNetworkState(StrictModule):
    """Complete fixed-shape state at a propagation transaction boundary."""

    latest_activation_time_ms: Array
    refractory_until_ms: Array
    activation_count: Array
    edge_blocked: Array
    last_processed_time_ms: Array
    plan_id: str = eqx.field(static=True)


class PurkinjeStimulusBatch(StrictModule):
    """Fixed-capacity external stimuli; inactive slots have ID and node ``-1``."""

    event_ids: Array
    node_indices: Array
    time_ms: Array
    active: Array
    plan_id: str = eqx.field(static=True)


class PurkinjeEventBatch(StrictModule):
    """Deterministically ordered fixed-capacity propagation outcomes."""

    event_ids: Array
    kind: Array
    time_ms: Array
    node_index: Array
    edge_index: Array
    source_node_index: Array
    parent_event_id: Array
    active: Array
    plan_id: str = eqx.field(static=True)


class PurkinjePropagationEvidence(StrictModule):
    """Capacity, collision, block, refractory, ordering, and derivative evidence."""

    event_count: Array
    activation_count: Array
    refractory_rejection_count: Array
    blocked_wave_count: Array
    collision_count: Array
    event_order_margin_ms: Array
    deterministic_order: Array
    fixed_event_sequence_derivative_valid: Array
    overflowed: Array
    finite: Array
    status: Array
    successful: Array


class PurkinjePropagationResult(StrictModule):
    """Accepted state, rejected candidate, fixed event batch, and evidence."""

    state: PurkinjeNetworkState
    candidate_state: PurkinjeNetworkState
    events: PurkinjeEventBatch
    evidence: PurkinjePropagationEvidence


def initialize_purkinje_state(plan: PurkinjeNetworkPlan, /) -> PurkinjeNetworkState:
    """Return a quiescent state with inactive topology slots blocked."""

    if not isinstance(plan, PurkinjeNetworkPlan):
        raise TypeError("plan must be a PurkinjeNetworkPlan.")
    node_count = plan.node_ids.shape[0]
    inactive_edges = plan.edge_ids < 0
    return PurkinjeNetworkState(
        jnp.full((node_count,), -jnp.inf, dtype=plan.edge_delay_ms.dtype),
        jnp.full((node_count,), -jnp.inf, dtype=plan.edge_delay_ms.dtype),
        jnp.zeros((node_count,), dtype=jnp.int32),
        inactive_edges,
        jnp.asarray(-jnp.inf, dtype=plan.edge_delay_ms.dtype),
        plan.plan_id,
    )


def with_purkinje_edge_block(
    plan: PurkinjeNetworkPlan,
    state: PurkinjeNetworkState,
    edge_blocked: ArrayLike,
    /,
) -> PurkinjeNetworkState:
    """Create a new state with an explicit block mask for the same topology epoch."""

    _validate_state(plan, state)
    blocked = np.asarray(edge_blocked, dtype=bool)
    if blocked.shape != tuple(plan.edge_ids.shape):
        raise ValueError("edge_blocked must match the fixed edge capacity.")
    blocked = blocked | (np.asarray(plan.edge_ids) < 0)
    return PurkinjeNetworkState(
        state.latest_activation_time_ms,
        state.refractory_until_ms,
        state.activation_count,
        jnp.asarray(blocked),
        state.last_processed_time_ms,
        state.plan_id,
    )


def make_purkinje_stimulus_batch(
    plan: PurkinjeNetworkPlan,
    event_ids: ArrayLike,
    node_indices: ArrayLike,
    time_ms: ArrayLike,
    /,
) -> PurkinjeStimulusBatch:
    """Pad validated external stimuli to the plan's fixed input capacity."""

    if not isinstance(plan, PurkinjeNetworkPlan):
        raise TypeError("plan must be a PurkinjeNetworkPlan.")
    identifiers = np.asarray(event_ids, dtype=np.int64)
    nodes = np.asarray(node_indices, dtype=np.int32)
    times = np.asarray(time_ms, dtype=float)
    if (
        identifiers.ndim != 1
        or nodes.shape != identifiers.shape
        or times.shape != identifiers.shape
    ):
        raise ValueError(
            "Stimulus IDs, node indices, and times must be equal-length vectors."
        )
    if identifiers.size > plan.stimulus_capacity:
        raise ValueError("Stimulus count exceeds the prepared stimulus capacity.")
    if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
        raise ValueError("Active stimulus event IDs must be unique and nonnegative.")
    active_nodes = np.asarray(plan.node_ids) >= 0
    if (
        np.any(nodes < 0)
        or np.any(nodes >= active_nodes.size)
        or np.any(~active_nodes[nodes])
    ):
        raise ValueError("Every stimulus must target an active Purkinje node.")
    if not np.all(np.isfinite(times)):
        raise ValueError("Stimulus times must be finite.")
    capacity = plan.stimulus_capacity
    padded_ids = np.full((capacity,), -1, dtype=np.int64)
    padded_nodes = np.full((capacity,), -1, dtype=np.int32)
    padded_times = np.zeros((capacity,), dtype=times.dtype)
    active = np.zeros((capacity,), dtype=bool)
    padded_ids[: identifiers.size] = identifiers
    padded_nodes[: nodes.size] = nodes
    padded_times[: times.size] = times
    active[: identifiers.size] = True
    return PurkinjeStimulusBatch(
        jnp.asarray(padded_ids),
        jnp.asarray(padded_nodes),
        jnp.asarray(padded_times),
        jnp.asarray(active),
        plan.plan_id,
    )


def _validate_state(plan: PurkinjeNetworkPlan, state: PurkinjeNetworkState, /) -> None:
    if not isinstance(state, PurkinjeNetworkState) or state.plan_id != plan.plan_id:
        raise ValueError("Purkinje state belongs to another network plan.")
    node_shape = tuple(plan.node_ids.shape)
    edge_shape = tuple(plan.edge_ids.shape)
    if (
        state.latest_activation_time_ms.shape != node_shape
        or state.refractory_until_ms.shape != node_shape
        or state.activation_count.shape != node_shape
        or state.edge_blocked.shape != edge_shape
    ):
        raise ValueError("Purkinje state changed the fixed topology capacity.")


def _validate_stimuli(plan: PurkinjeNetworkPlan, batch: PurkinjeStimulusBatch, /) -> None:
    if not isinstance(batch, PurkinjeStimulusBatch) or batch.plan_id != plan.plan_id:
        raise ValueError("Stimulus batch belongs to another network plan.")
    expected = (plan.stimulus_capacity,)
    arrays = (batch.event_ids, batch.node_indices, batch.time_ms, batch.active)
    if any(value.shape != expected for value in arrays):
        raise ValueError("Stimulus batch changed the fixed input capacity.")


def propagate_purkinje(
    plan: PurkinjeNetworkPlan,
    state: PurkinjeNetworkState,
    stimuli: PurkinjeStimulusBatch,
    /,
) -> PurkinjePropagationResult:
    """Execute one host-side event transaction with deterministic priority ordering.

    Events are ordered by ``(time, stable node ID, stable event ID, insertion)``.
    This transaction is intentionally outside automatic differentiation.  The
    evidence only qualifies a continuous timing derivative when the realized
    event sequence contains no branch events and has a positive time margin.
    """

    if not isinstance(plan, PurkinjeNetworkPlan):
        raise TypeError("plan must be a PurkinjeNetworkPlan.")
    _validate_state(plan, state)
    _validate_stimuli(plan, stimuli)

    node_ids = np.asarray(plan.node_ids)
    edge_ids = np.asarray(plan.edge_ids)
    incidence = np.asarray(plan.edge_nodes)
    delays = np.asarray(plan.edge_delay_ms)
    refractory_period = np.asarray(plan.refractory_period_ms)
    edge_blocked = np.asarray(state.edge_blocked, dtype=bool).copy()
    latest = np.asarray(state.latest_activation_time_ms).copy()
    refractory_until = np.asarray(state.refractory_until_ms).copy()
    activation_count = np.asarray(state.activation_count).copy()
    prior_last_time = float(np.asarray(state.last_processed_time_ms))
    active_stimuli = np.asarray(stimuli.active, dtype=bool)
    input_ids = np.asarray(stimuli.event_ids)
    input_nodes = np.asarray(stimuli.node_indices)
    input_times = np.asarray(stimuli.time_ms)

    queue: list[tuple[float, int, int, int, str, tuple[int, ...]]] = []
    insertion = 0
    next_event_id = int(np.max(input_ids[active_stimuli], initial=-1)) + 1
    active_tokens: dict[int, bool] = {}
    pending: dict[tuple[int, int, int], list[tuple[int, float, int]]] = {}

    def push(
        time: float, node: int, event_id: int, category: str, payload: tuple[int, ...]
    ):
        nonlocal insertion
        stable_node = int(node_ids[node]) if node >= 0 else np.iinfo(np.int64).max
        heapq.heappush(
            queue,
            (float(time), stable_node, int(event_id), insertion, category, payload),
        )
        insertion += 1

    for event_id, node, time in zip(
        input_ids[active_stimuli],
        input_nodes[active_stimuli],
        input_times[active_stimuli],
    ):
        push(float(time), int(node), int(event_id), "stimulus", ())

    adjacency: list[list[int]] = [[] for _ in range(node_ids.size)]
    for edge_index in np.flatnonzero(edge_ids >= 0):
        left, right = incidence[edge_index]
        adjacency[int(left)].append(int(edge_index))
        adjacency[int(right)].append(int(edge_index))
    for node_edges in adjacency:
        node_edges.sort(key=lambda edge_index: int(edge_ids[edge_index]))

    output_id = np.full((plan.event_capacity,), -1, dtype=np.int64)
    output_kind = np.zeros((plan.event_capacity,), dtype=np.int32)
    output_time = np.zeros((plan.event_capacity,), dtype=float)
    output_node = np.full((plan.event_capacity,), -1, dtype=np.int32)
    output_edge = np.full((plan.event_capacity,), -1, dtype=np.int32)
    output_source = np.full((plan.event_capacity,), -1, dtype=np.int32)
    output_parent = np.full((plan.event_capacity,), -1, dtype=np.int64)
    output_active = np.zeros((plan.event_capacity,), dtype=bool)
    output_count = 0
    last_processed = prior_last_time

    def record(
        event_id: int,
        kind: PurkinjeEventKind,
        time: float,
        node: int,
        edge: int,
        source: int,
        parent: int,
    ) -> bool:
        nonlocal output_count, last_processed
        if output_count >= plan.event_capacity:
            return False
        output_id[output_count] = event_id
        output_kind[output_count] = int(kind)
        output_time[output_count] = time
        output_node[output_count] = node
        output_edge[output_count] = edge
        output_source[output_count] = source
        output_parent[output_count] = parent
        output_active[output_count] = True
        output_count += 1
        last_processed = max(last_processed, time)
        return True

    def schedule_wave(depart_time: float, source: int, edge_index: int, parent: int):
        nonlocal next_event_id
        left, right = incidence[edge_index]
        target = int(right if source == left else left)
        event_id = next_event_id
        next_event_id += 1
        if edge_blocked[edge_index]:
            push(
                depart_time,
                target,
                event_id,
                "blocked",
                (target, edge_index, source, parent),
            )
            return
        opposite_key = (edge_index, target, source)
        candidates = pending.get(opposite_key, [])
        live = [item for item in candidates if active_tokens.get(item[0], False)]
        if live:
            opposite_token, opposite_departure, opposite_event_id = min(
                live, key=lambda item: (item[1], item[2], item[0])
            )
            delay = float(delays[edge_index])
            if (
                abs(depart_time - opposite_departure)
                < delay - plan.collision_tolerance_ms
            ):
                active_tokens[opposite_token] = False
                collision_time = 0.5 * (depart_time + opposite_departure + delay)
                collision_event_id = next_event_id
                next_event_id += 1
                push(
                    collision_time,
                    -1,
                    collision_event_id,
                    "collision",
                    (edge_index, source, parent),
                )
                return
        token = insertion
        active_tokens[token] = True
        pending.setdefault((edge_index, source, target), []).append(
            (token, depart_time, event_id)
        )
        arrival_time = depart_time + float(delays[edge_index])
        push(
            arrival_time,
            target,
            event_id,
            "wave",
            (target, edge_index, source, parent, token),
        )

    overflowed = False
    nonmonotone = bool(np.any(input_times[active_stimuli] < prior_last_time))
    while queue and output_count < plan.event_capacity:
        time, _, event_id, _, category, payload = heapq.heappop(queue)
        if category == "wave":
            target, edge_index, source, parent, token = payload
            if not active_tokens.get(token, False):
                continue
            active_tokens[token] = False
            accepted = time + plan.collision_tolerance_ms >= refractory_until[target]
            if not accepted:
                record(
                    event_id,
                    PurkinjeEventKind.REFRACTORY_REJECTION,
                    time,
                    target,
                    edge_index,
                    source,
                    parent,
                )
                continue
            latest[target] = time
            refractory_until[target] = time + refractory_period[target]
            activation_count[target] += 1
            record(
                event_id,
                PurkinjeEventKind.NETWORK_ACTIVATION,
                time,
                target,
                edge_index,
                source,
                parent,
            )
            for next_edge in adjacency[target]:
                if next_edge != edge_index:
                    schedule_wave(time, target, next_edge, event_id)
        elif category == "stimulus":
            node = int(input_nodes[np.flatnonzero(input_ids == event_id)[0]])
            accepted = time + plan.collision_tolerance_ms >= refractory_until[node]
            if not accepted:
                record(
                    event_id,
                    PurkinjeEventKind.REFRACTORY_REJECTION,
                    time,
                    node,
                    -1,
                    node,
                    -1,
                )
                continue
            latest[node] = time
            refractory_until[node] = time + refractory_period[node]
            activation_count[node] += 1
            record(
                event_id,
                PurkinjeEventKind.STIMULUS_ACTIVATION,
                time,
                node,
                -1,
                node,
                -1,
            )
            for edge_index in adjacency[node]:
                schedule_wave(time, node, edge_index, event_id)
        elif category == "blocked":
            target, edge_index, source, parent = payload
            record(
                event_id,
                PurkinjeEventKind.EDGE_BLOCK,
                time,
                target,
                edge_index,
                source,
                parent,
            )
        else:
            edge_index, source, parent = payload
            record(
                event_id,
                PurkinjeEventKind.WAVE_COLLISION,
                time,
                -1,
                edge_index,
                source,
                parent,
            )
    overflowed = bool(queue)
    finite = bool(
        np.all(
            np.isfinite(latest[np.asarray(plan.node_ids) >= 0])
            | np.isneginf(latest[np.asarray(plan.node_ids) >= 0])
        )
        and np.all(
            np.isfinite(refractory_until[np.asarray(plan.node_ids) >= 0])
            | np.isneginf(refractory_until[np.asarray(plan.node_ids) >= 0])
        )
    )
    candidate_state = PurkinjeNetworkState(
        jnp.asarray(latest),
        jnp.asarray(refractory_until),
        jnp.asarray(activation_count),
        jnp.asarray(edge_blocked),
        jnp.asarray(last_processed),
        plan.plan_id,
    )
    status = int(PurkinjePropagationStatus.SUCCESS)
    if overflowed:
        status |= int(PurkinjePropagationStatus.EVENT_CAPACITY_EXCEEDED)
    if not finite:
        status |= int(PurkinjePropagationStatus.NONFINITE_STATE)
    if nonmonotone:
        status |= int(PurkinjePropagationStatus.NONMONOTONE_INPUT)
    successful = status == int(PurkinjePropagationStatus.SUCCESS)
    accepted_state = candidate_state if successful else state
    active_slice = slice(0, output_count)
    kinds = output_kind[active_slice]
    ordered_times = output_time[active_slice]
    if output_count < 2:
        event_order_margin = np.inf
    else:
        event_order_margin = float(np.min(np.diff(ordered_times)))
    deterministic_order = bool(np.all(np.diff(ordered_times) >= 0.0))
    branch_count = int(
        np.sum(
            (kinds == int(PurkinjeEventKind.REFRACTORY_REJECTION))
            | (kinds == int(PurkinjeEventKind.EDGE_BLOCK))
            | (kinds == int(PurkinjeEventKind.WAVE_COLLISION))
        )
    )
    derivative_valid = bool(
        successful
        and deterministic_order
        and event_order_margin > plan.collision_tolerance_ms
        and branch_count == 0
    )
    events = PurkinjeEventBatch(
        jnp.asarray(output_id),
        jnp.asarray(output_kind),
        jnp.asarray(output_time),
        jnp.asarray(output_node),
        jnp.asarray(output_edge),
        jnp.asarray(output_source),
        jnp.asarray(output_parent),
        jnp.asarray(output_active),
        plan.plan_id,
    )
    evidence = PurkinjePropagationEvidence(
        jnp.asarray(output_count, dtype=jnp.int32),
        jnp.asarray(
            np.sum(
                (kinds == int(PurkinjeEventKind.STIMULUS_ACTIVATION))
                | (kinds == int(PurkinjeEventKind.NETWORK_ACTIVATION))
            ),
            dtype=jnp.int32,
        ),
        jnp.asarray(
            np.sum(kinds == int(PurkinjeEventKind.REFRACTORY_REJECTION)),
            dtype=jnp.int32,
        ),
        jnp.asarray(np.sum(kinds == int(PurkinjeEventKind.EDGE_BLOCK)), dtype=jnp.int32),
        jnp.asarray(
            np.sum(kinds == int(PurkinjeEventKind.WAVE_COLLISION)), dtype=jnp.int32
        ),
        jnp.asarray(event_order_margin, dtype=plan.edge_delay_ms.dtype),
        jnp.asarray(deterministic_order),
        jnp.asarray(derivative_valid),
        jnp.asarray(overflowed),
        jnp.asarray(finite),
        jnp.asarray(status, dtype=jnp.int32),
        jnp.asarray(successful),
    )
    return PurkinjePropagationResult(accepted_state, candidate_state, events, evidence)


__all__ = [
    "PurkinjeEventBatch",
    "PurkinjeEventKind",
    "PurkinjeNetworkPlan",
    "PurkinjeNetworkState",
    "PurkinjePropagationEvidence",
    "PurkinjePropagationResult",
    "PurkinjePropagationStatus",
    "PurkinjeStimulusBatch",
    "initialize_purkinje_state",
    "make_purkinje_stimulus_batch",
    "propagate_purkinje",
    "with_purkinje_edge_block",
]
