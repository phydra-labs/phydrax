#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Purkinje--muscle junction exchange and stateful pacing protocols."""

from __future__ import annotations

from enum import IntFlag
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._conduction_network import (
    make_purkinje_stimulus_batch,
    PurkinjeEventKind,
    PurkinjeNetworkPlan,
    PurkinjePropagationResult,
    PurkinjeStimulusBatch,
)


class PMJStatus(IntFlag):
    """Fail-closed PMJ exchange or scheduling status."""

    SUCCESS = 0
    NONFINITE = 1
    EVENT_CAPACITY_EXCEEDED = 2
    CAUSALITY_FAILURE = 4
    CONSERVATION_FAILURE = 8


class PMJExchangePlan(StrictModule, NonTrainableState):
    """Fixed-capacity PMJ support, delay, and ohmic exchange definition."""

    junction_ids: Array
    purkinje_node_indices: Array
    tissue_node_indices: Array
    delay_ms: Array
    coupling_conductance_mS: Array
    purkinje_plan_id: str = eqx.field(static=True)
    purkinje_node_count: int = eqx.field(static=True)
    tissue_node_count: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        junction_ids: ArrayLike,
        purkinje_node_indices: ArrayLike,
        tissue_node_indices: ArrayLike,
        delay_ms: ArrayLike,
        coupling_conductance_mS: ArrayLike,
        /,
        *,
        purkinje_plan: PurkinjeNetworkPlan,
        tissue_node_count: int,
        event_capacity: int,
    ):
        identifiers = np.asarray(junction_ids, dtype=np.int64)
        purkinje = np.asarray(purkinje_node_indices, dtype=np.int32)
        tissue = np.asarray(tissue_node_indices, dtype=np.int32)
        delays = np.asarray(delay_ms, dtype=float)
        conductance = np.asarray(coupling_conductance_mS, dtype=float)
        shape = identifiers.shape
        if identifiers.ndim != 1 or identifiers.size == 0:
            raise ValueError("junction_ids must be a non-empty fixed-capacity vector.")
        if any(value.shape != shape for value in (purkinje, tissue, delays, conductance)):
            raise ValueError("Every PMJ array must match junction capacity.")
        active = identifiers >= 0
        if np.any(identifiers < -1) or np.unique(identifiers[active]).size != np.sum(
            active
        ):
            raise ValueError(
                "Active PMJ IDs must be unique and -1 is the inactive sentinel."
            )
        if not isinstance(purkinje_plan, PurkinjeNetworkPlan):
            raise TypeError("purkinje_plan must be a PurkinjeNetworkPlan.")
        purkinje_count = int(purkinje_plan.node_ids.shape[0])
        tissue_count = int(tissue_node_count)
        capacity = int(event_capacity)
        if purkinje_count <= 0 or tissue_count <= 0 or capacity <= 0:
            raise ValueError("PMJ node counts and event capacity must be positive.")
        if (
            np.any(purkinje[active] < 0)
            or np.any(purkinje[active] >= purkinje_count)
            or np.any(tissue[active] < 0)
            or np.any(tissue[active] >= tissue_count)
        ):
            raise ValueError("Active PMJ support indices lie outside their node layouts.")
        if np.any(purkinje[~active] != -1) or np.any(tissue[~active] != -1):
            raise ValueError("Inactive PMJ support indices must be -1.")
        if (
            not np.all(np.isfinite(delays[active]))
            or np.any(delays[active] < 0.0)
            or not np.all(np.isfinite(conductance[active]))
            or np.any(conductance[active] < 0.0)
        ):
            raise ValueError(
                "Active PMJ delays and conductances must be finite and nonnegative."
            )
        if np.any(delays[~active] != 0.0) or np.any(conductance[~active] != 0.0):
            raise ValueError("Inactive PMJ delay and conductance slots must be zero.")
        self.junction_ids = jnp.asarray(identifiers)
        self.purkinje_node_indices = jnp.asarray(purkinje)
        self.tissue_node_indices = jnp.asarray(tissue)
        self.delay_ms = jnp.asarray(delays)
        self.coupling_conductance_mS = jnp.asarray(conductance)
        self.purkinje_plan_id = purkinje_plan.plan_id
        self.purkinje_node_count = purkinje_count
        self.tissue_node_count = tissue_count
        self.event_capacity = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-pmj-exchange-plan",
                "arrays": array_tree_fingerprint(
                    (identifiers, purkinje, tissue, delays, conductance)
                ),
                "purkinje_plan_id": purkinje_plan.plan_id,
                "purkinje_node_count": purkinje_count,
                "tissue_node_count": tissue_count,
                "event_capacity": capacity,
            }
        )


class PMJExchangeEvidence(StrictModule):
    """Support, finiteness, and equal-and-opposite current evidence."""

    net_exchange_current_uA: Array
    maximum_pair_balance_error_uA: Array
    active_junction_count: Array
    finite: Array
    conservative: Array
    status: Array
    successful: Array


class PMJExchangeResult(StrictModule):
    """Nodal currents use positive-inward sign on each receiving compartment."""

    junction_current_purkinje_to_tissue_uA: Array
    purkinje_current_uA: Array
    tissue_current_uA: Array
    evidence: PMJExchangeEvidence
    plan_id: str = eqx.field(static=True)


class PMJActivationBatch(StrictModule):
    """Fixed-capacity delayed tissue activations produced by Purkinje events."""

    event_ids: Array
    junction_index: Array
    tissue_node_index: Array
    activation_time_ms: Array
    parent_purkinje_event_id: Array
    accepted: Array
    active: Array
    plan_id: str = eqx.field(static=True)


class PMJActivationEvidence(StrictModule):
    """Support/timing, refractory, capacity, and causality evidence."""

    event_count: Array
    accepted_count: Array
    refractory_rejection_count: Array
    minimum_delay_ms: Array
    causal: Array
    overflowed: Array
    finite: Array
    status: Array
    successful: Array


class PMJActivationResult(StrictModule):
    activations: PMJActivationBatch
    evidence: PMJActivationEvidence


def evaluate_pmj_exchange(
    plan: PMJExchangePlan,
    purkinje_voltage_mV: ArrayLike,
    tissue_voltage_mV: ArrayLike,
    /,
) -> PMJExchangeResult:
    """Evaluate conservative ohmic exchange on the fixed PMJ support.

    Since mS times mV is uA, no hidden unit rescaling is present.  Positive
    junction current flows from the Purkinje node into the tissue node.
    """

    if not isinstance(plan, PMJExchangePlan):
        raise TypeError("plan must be a PMJExchangePlan.")
    purkinje_voltage = jnp.asarray(purkinje_voltage_mV)
    tissue_voltage = jnp.asarray(tissue_voltage_mV)
    if purkinje_voltage.shape != (plan.purkinje_node_count,):
        raise ValueError("purkinje_voltage_mV changed the prepared node layout.")
    if tissue_voltage.shape != (plan.tissue_node_count,):
        raise ValueError("tissue_voltage_mV changed the prepared node layout.")
    active = plan.junction_ids >= 0
    safe_purkinje = jnp.where(active, plan.purkinje_node_indices, 0)
    safe_tissue = jnp.where(active, plan.tissue_node_indices, 0)
    pair_current = jnp.where(
        active,
        plan.coupling_conductance_mS
        * (purkinje_voltage[safe_purkinje] - tissue_voltage[safe_tissue]),
        0.0,
    )
    purkinje_current = (
        jnp.zeros((plan.purkinje_node_count,), dtype=pair_current.dtype)
        .at[safe_purkinje]
        .add(-pair_current)
    )
    tissue_current = (
        jnp.zeros((plan.tissue_node_count,), dtype=pair_current.dtype)
        .at[safe_tissue]
        .add(pair_current)
    )
    net_current = jnp.sum(purkinje_current) + jnp.sum(tissue_current)
    pair_balance = pair_current + (-pair_current)
    finite = (
        jnp.all(jnp.isfinite(pair_current))
        & jnp.all(jnp.isfinite(purkinje_voltage))
        & jnp.all(jnp.isfinite(tissue_voltage))
    )
    conservative = jnp.abs(net_current) <= 32.0 * jnp.finfo(
        pair_current.dtype
    ).eps * jnp.maximum(jnp.sum(jnp.abs(pair_current)), 1.0)
    status = jnp.asarray(int(PMJStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(PMJStatus.NONFINITE)),
    )
    status = jnp.where(
        conservative,
        status,
        jnp.bitwise_or(status, int(PMJStatus.CONSERVATION_FAILURE)),
    )
    successful = status == int(PMJStatus.SUCCESS)
    evidence = PMJExchangeEvidence(
        net_current,
        jnp.max(jnp.abs(pair_balance), initial=0.0),
        jnp.sum(active, dtype=jnp.int32),
        finite,
        conservative,
        status,
        successful,
    )
    return PMJExchangeResult(
        pair_current,
        purkinje_current,
        tissue_current,
        evidence,
        plan.plan_id,
    )


def schedule_pmj_activations(
    plan: PMJExchangePlan,
    propagation: PurkinjePropagationResult,
    tissue_refractory_until_ms: ArrayLike,
    /,
) -> PMJActivationResult:
    """Map events from one accepted Purkinje transaction to delayed tissue support."""

    if not isinstance(plan, PMJExchangePlan):
        raise TypeError("plan must be a PMJExchangePlan.")
    if not isinstance(propagation, PurkinjePropagationResult):
        raise TypeError("propagation must be a PurkinjePropagationResult.")
    if not bool(np.asarray(propagation.evidence.successful)):
        raise ValueError("PMJ scheduling rejects unsuccessful Purkinje transactions.")
    purkinje_events = propagation.events
    if (
        purkinje_events.plan_id != plan.purkinje_plan_id
        or propagation.state.plan_id != plan.purkinje_plan_id
    ):
        raise ValueError("Purkinje propagation belongs to another PMJ-bound plan.")
    refractory = np.asarray(tissue_refractory_until_ms, dtype=float)
    if refractory.shape != (plan.tissue_node_count,):
        raise ValueError("tissue_refractory_until_ms changed the tissue layout.")
    if not np.all(np.isfinite(refractory) | np.isneginf(refractory)):
        raise ValueError("Tissue refractory times must be finite or negative infinity.")
    junction_ids = np.asarray(plan.junction_ids)
    junction_purkinje = np.asarray(plan.purkinje_node_indices)
    junction_tissue = np.asarray(plan.tissue_node_indices)
    delays = np.asarray(plan.delay_ms)
    event_active = np.asarray(purkinje_events.active, dtype=bool)
    event_kind = np.asarray(purkinje_events.kind)
    event_node = np.asarray(purkinje_events.node_index)
    event_time = np.asarray(purkinje_events.time_ms)
    event_ids = np.asarray(purkinje_events.event_ids)
    activation_kind = (event_kind == int(PurkinjeEventKind.STIMULUS_ACTIVATION)) | (
        event_kind == int(PurkinjeEventKind.NETWORK_ACTIVATION)
    )
    event_shape = event_ids.shape
    if event_ids.ndim != 1 or any(
        value.shape != event_shape
        for value in (event_active, event_kind, event_node, event_time)
    ):
        raise ValueError("Purkinje event arrays must share one fixed vector capacity.")
    activation_slots = event_active & activation_kind
    if (
        np.any(event_node[activation_slots] < 0)
        or np.any(event_node[activation_slots] >= plan.purkinje_node_count)
        or not np.all(np.isfinite(event_time[activation_slots]))
    ):
        raise ValueError("An active Purkinje activation has invalid support or time.")
    candidates: list[tuple[float, int, int, int, int]] = []
    for source_slot in np.flatnonzero(event_active & activation_kind):
        node = int(event_node[source_slot])
        for junction_index in np.flatnonzero(
            (junction_ids >= 0) & (junction_purkinje == node)
        ):
            tissue_node = int(junction_tissue[junction_index])
            time = float(event_time[source_slot] + delays[junction_index])
            candidates.append(
                (
                    time,
                    tissue_node,
                    int(junction_ids[junction_index]),
                    int(event_ids[source_slot]),
                    int(junction_index),
                )
            )
    candidates.sort()
    overflowed = len(candidates) > plan.event_capacity
    selected = candidates[: plan.event_capacity]
    identifiers = np.full((plan.event_capacity,), -1, dtype=np.int64)
    junction_index_out = np.full((plan.event_capacity,), -1, dtype=np.int32)
    tissue_out = np.full((plan.event_capacity,), -1, dtype=np.int32)
    time_out = np.zeros((plan.event_capacity,), dtype=float)
    parent_out = np.full((plan.event_capacity,), -1, dtype=np.int64)
    accepted = np.zeros((plan.event_capacity,), dtype=bool)
    active = np.zeros((plan.event_capacity,), dtype=bool)
    for output_index, (time, tissue_node, _, parent_id, junction_index) in enumerate(
        selected
    ):
        identifiers[output_index] = output_index
        junction_index_out[output_index] = junction_index
        tissue_out[output_index] = tissue_node
        time_out[output_index] = time
        parent_out[output_index] = parent_id
        accepted[output_index] = time >= refractory[tissue_node]
        active[output_index] = True
    selected_delays = (
        delays[junction_index_out[active]] if np.any(active) else np.asarray((np.inf,))
    )
    causal = bool(np.all(selected_delays >= 0.0))
    finite = bool(np.all(np.isfinite(time_out[active])))
    status = int(PMJStatus.SUCCESS)
    if overflowed:
        status |= int(PMJStatus.EVENT_CAPACITY_EXCEEDED)
    if not finite:
        status |= int(PMJStatus.NONFINITE)
    if not causal:
        status |= int(PMJStatus.CAUSALITY_FAILURE)
    successful = status == int(PMJStatus.SUCCESS)
    batch = PMJActivationBatch(
        jnp.asarray(identifiers),
        jnp.asarray(junction_index_out),
        jnp.asarray(tissue_out),
        jnp.asarray(time_out),
        jnp.asarray(parent_out),
        jnp.asarray(accepted),
        jnp.asarray(active),
        plan.plan_id,
    )
    evidence = PMJActivationEvidence(
        jnp.asarray(len(selected), dtype=jnp.int32),
        jnp.asarray(np.sum(accepted & active), dtype=jnp.int32),
        jnp.asarray(np.sum((~accepted) & active), dtype=jnp.int32),
        jnp.asarray(np.min(selected_delays)),
        jnp.asarray(causal),
        jnp.asarray(overflowed),
        jnp.asarray(finite),
        jnp.asarray(status, dtype=jnp.int32),
        jnp.asarray(successful),
    )
    return PMJActivationResult(batch, evidence)


class TissuePacingTarget(StrictModule, NonTrainableState):
    """Pulse sites on a tissue current-density field."""

    node_count: int = eqx.field(static=True)
    site_indices: Array
    target_id: str = eqx.field(static=True)

    def __init__(self, node_count: int, site_indices: ArrayLike, /):
        count = int(node_count)
        sites = np.asarray(site_indices, dtype=np.int32)
        if count <= 0 or sites.ndim != 1:
            raise ValueError(
                "Tissue pacing requires a positive node count and site vector."
            )
        active = sites >= 0
        if np.any(sites < -1) or np.any(sites[active] >= count):
            raise ValueError("A tissue pacing site lies outside node capacity.")
        self.node_count = count
        self.site_indices = jnp.asarray(sites)
        self.target_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-tissue-pacing-target",
                "node_count": count,
                "sites": array_tree_fingerprint(sites),
            }
        )


class PurkinjePacingTarget(StrictModule, NonTrainableState):
    """Pulse sites on a Purkinje event network."""

    node_count: int = eqx.field(static=True)
    site_indices: Array
    target_id: str = eqx.field(static=True)

    def __init__(self, node_count: int, site_indices: ArrayLike, /):
        count = int(node_count)
        sites = np.asarray(site_indices, dtype=np.int32)
        if count <= 0 or sites.ndim != 1:
            raise ValueError(
                "Purkinje pacing requires a positive node count and site vector."
            )
        active = sites >= 0
        if np.any(sites < -1) or np.any(sites[active] >= count):
            raise ValueError("A Purkinje pacing site lies outside node capacity.")
        self.node_count = count
        self.site_indices = jnp.asarray(sites)
        self.target_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-purkinje-pacing-target",
                "node_count": count,
                "sites": array_tree_fingerprint(sites),
            }
        )


PacingTarget = TissuePacingTarget | PurkinjePacingTarget


class PacingProtocol(StrictModule, NonTrainableState):
    """Fixed-capacity pulse train with an explicit tissue or Purkinje target route."""

    target: PacingTarget
    pulse_ids: Array
    start_time_ms: Array
    duration_ms: Array
    amplitude_uA_per_mm3: Array
    protocol_id: str = eqx.field(static=True)

    def __init__(
        self,
        target: PacingTarget,
        pulse_ids: ArrayLike,
        start_time_ms: ArrayLike,
        duration_ms: ArrayLike,
        amplitude_uA_per_mm3: ArrayLike,
        /,
    ):
        if not isinstance(target, (TissuePacingTarget, PurkinjePacingTarget)):
            raise TypeError("target must be a tissue or Purkinje pacing target.")
        identifiers = np.asarray(pulse_ids, dtype=np.int64)
        starts = np.asarray(start_time_ms, dtype=float)
        durations = np.asarray(duration_ms, dtype=float)
        amplitudes = np.asarray(amplitude_uA_per_mm3, dtype=float)
        shape = identifiers.shape
        if identifiers.ndim != 1 or identifiers.size == 0:
            raise ValueError("pulse_ids must be a non-empty fixed-capacity vector.")
        if any(value.shape != shape for value in (starts, durations, amplitudes)):
            raise ValueError("Every pacing pulse array must match pulse capacity.")
        if target.site_indices.shape != shape:
            raise ValueError("Pacing target sites must match pulse capacity.")
        active = identifiers >= 0
        if np.any(identifiers < -1) or np.unique(identifiers[active]).size != np.sum(
            active
        ):
            raise ValueError(
                "Active pulse IDs must be unique and -1 is the inactive sentinel."
            )
        if (
            not np.all(np.isfinite(starts[active]))
            or not np.all(np.isfinite(durations[active]))
            or np.any(durations[active] <= 0.0)
            or not np.all(np.isfinite(amplitudes[active]))
        ):
            raise ValueError(
                "Active pacing times, durations, and amplitudes must be finite."
            )
        if (
            np.any(starts[~active] != 0.0)
            or np.any(durations[~active] != 0.0)
            or np.any(amplitudes[~active] != 0.0)
            or np.any(np.asarray(target.site_indices)[~active] != -1)
        ):
            raise ValueError("Inactive pulse slots must contain zero values and site -1.")
        self.target = target
        self.pulse_ids = jnp.asarray(identifiers)
        self.start_time_ms = jnp.asarray(starts)
        self.duration_ms = jnp.asarray(durations)
        self.amplitude_uA_per_mm3 = jnp.asarray(amplitudes)
        self.protocol_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-pacing-protocol",
                "target": target.target_id,
                "arrays": array_tree_fingerprint(
                    (identifiers, starts, durations, amplitudes)
                ),
            }
        )


class PacingEvaluation(StrictModule):
    nodal_stimulus_uA_per_mm3: Array
    active_pulse_mask: Array
    finite: Array
    protocol_id: str = eqx.field(static=True)


def evaluate_pacing_protocol(
    protocol: PacingProtocol, time_ms: ArrayLike, /
) -> PacingEvaluation:
    """Evaluate the rectangular pulse field at one kernel time."""

    if not isinstance(protocol, PacingProtocol):
        raise TypeError("protocol must be a PacingProtocol.")
    time = jnp.asarray(time_ms)
    if time.shape != ():
        raise ValueError("time_ms must be scalar.")
    active_slots = protocol.pulse_ids >= 0
    active = (
        active_slots
        & (time >= protocol.start_time_ms)
        & (time < protocol.start_time_ms + protocol.duration_ms)
    )
    safe_sites = jnp.where(active_slots, protocol.target.site_indices, 0)
    nodal = (
        jnp.zeros(
            (protocol.target.node_count,),
            dtype=jnp.result_type(time, protocol.amplitude_uA_per_mm3),
        )
        .at[safe_sites]
        .add(jnp.where(active, protocol.amplitude_uA_per_mm3, 0.0))
    )
    finite = jnp.isfinite(time) & jnp.all(jnp.isfinite(nodal))
    return PacingEvaluation(nodal, active, finite, protocol.protocol_id)


def pacing_purkinje_stimuli(
    protocol: PacingProtocol,
    network: PurkinjeNetworkPlan,
    window_start_ms: float,
    window_end_ms: float,
    /,
) -> PurkinjeStimulusBatch:
    """Convert pulse onsets in ``(start, end]`` to a fixed network stimulus batch."""

    if not isinstance(protocol.target, PurkinjePacingTarget):
        raise TypeError("Purkinje event conversion requires a PurkinjePacingTarget.")
    if protocol.target.node_count != network.node_ids.shape[0]:
        raise ValueError("Pacing target and Purkinje network node capacities differ.")
    start = float(window_start_ms)
    end = float(window_end_ms)
    if not isfinite(start) or not isfinite(end) or end < start:
        raise ValueError("Pacing conversion requires a finite nondecreasing window.")
    pulse_ids = np.asarray(protocol.pulse_ids)
    pulse_starts = np.asarray(protocol.start_time_ms)
    sites = np.asarray(protocol.target.site_indices)
    selected = (pulse_ids >= 0) & (pulse_starts > start) & (pulse_starts <= end)
    order = np.lexsort((pulse_ids[selected], sites[selected], pulse_starts[selected]))
    return make_purkinje_stimulus_batch(
        network,
        pulse_ids[selected][order],
        sites[selected][order],
        pulse_starts[selected][order],
    )


class DemandPacingControllerPlan(StrictModule, NonTrainableState):
    """Escape-interval controller with bounded cycle-length feedback."""

    site_index: int = eqx.field(static=True)
    escape_interval_ms: float = eqx.field(static=True)
    target_sensed_interval_ms: float = eqx.field(static=True)
    minimum_cycle_length_ms: float = eqx.field(static=True)
    maximum_cycle_length_ms: float = eqx.field(static=True)
    feedback_gain: float = eqx.field(static=True)
    duration_ms: float = eqx.field(static=True)
    amplitude_uA_per_mm3: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_index: int,
        /,
        *,
        escape_interval_ms: float,
        target_sensed_interval_ms: float,
        minimum_cycle_length_ms: float,
        maximum_cycle_length_ms: float,
        feedback_gain: float,
        duration_ms: float,
        amplitude_uA_per_mm3: float,
    ):
        site = int(site_index)
        values = (
            float(escape_interval_ms),
            float(target_sensed_interval_ms),
            float(minimum_cycle_length_ms),
            float(maximum_cycle_length_ms),
            float(duration_ms),
        )
        gain = float(feedback_gain)
        amplitude = float(amplitude_uA_per_mm3)
        if site < 0:
            raise ValueError("site_index must be nonnegative.")
        if not all(isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                "Controller intervals and duration must be finite and positive."
            )
        if values[2] > values[3]:
            raise ValueError("minimum_cycle_length_ms may not exceed maximum.")
        if not isfinite(gain) or gain < 0.0 or not isfinite(amplitude):
            raise ValueError(
                "Controller gain and pulse amplitude must be finite; gain is nonnegative."
            )
        self.site_index = site
        self.escape_interval_ms = values[0]
        self.target_sensed_interval_ms = values[1]
        self.minimum_cycle_length_ms = values[2]
        self.maximum_cycle_length_ms = values[3]
        self.feedback_gain = gain
        self.duration_ms = values[4]
        self.amplitude_uA_per_mm3 = amplitude
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-demand-pacing-controller",
                "site_index": site,
                "escape_interval_ms": values[0],
                "target_sensed_interval_ms": values[1],
                "minimum_cycle_length_ms": values[2],
                "maximum_cycle_length_ms": values[3],
                "feedback_gain": gain,
                "duration_ms": values[4],
                "amplitude_uA_per_mm3": amplitude,
            }
        )


class DemandPacingControllerState(StrictModule):
    last_sensed_time_ms: Array
    next_pulse_time_ms: Array
    cycle_length_ms: Array
    command_count: Array
    last_update_time_ms: Array
    plan_id: str = eqx.field(static=True)


class PacingPulseCommand(StrictModule):
    command_id: Array
    site_index: Array
    start_time_ms: Array
    duration_ms: Array
    amplitude_uA_per_mm3: Array
    emitted: Array


class PacingControllerEvidence(StrictModule):
    sensed: Array
    emitted: Array
    monotone_time: Array
    finite: Array
    successful: Array


class PacingControllerResult(StrictModule):
    state: DemandPacingControllerState
    candidate_state: DemandPacingControllerState
    command: PacingPulseCommand
    evidence: PacingControllerEvidence


def initialize_demand_pacing_controller(
    plan: DemandPacingControllerPlan, start_time_ms: float = 0.0, /
) -> DemandPacingControllerState:
    if not isinstance(plan, DemandPacingControllerPlan):
        raise TypeError("plan must be a DemandPacingControllerPlan.")
    start = float(start_time_ms)
    if not isfinite(start):
        raise ValueError("start_time_ms must be finite.")
    dtype = jnp.asarray(start).dtype
    return DemandPacingControllerState(
        jnp.asarray(-jnp.inf, dtype=dtype),
        jnp.asarray(start + plan.escape_interval_ms, dtype=dtype),
        jnp.asarray(plan.escape_interval_ms, dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(start, dtype=dtype),
        plan.plan_id,
    )


def step_demand_pacing_controller(
    plan: DemandPacingControllerPlan,
    state: DemandPacingControllerState,
    time_ms: float,
    /,
    *,
    sensed_activation_time_ms: float | None = None,
) -> PacingControllerResult:
    """Advance sensing first, then emit at most one due escape pulse."""

    if not isinstance(plan, DemandPacingControllerPlan):
        raise TypeError("plan must be a DemandPacingControllerPlan.")
    if (
        not isinstance(state, DemandPacingControllerState)
        or state.plan_id != plan.plan_id
    ):
        raise ValueError("Controller state belongs to another plan.")
    time = float(time_ms)
    sensed = sensed_activation_time_ms is not None
    sensed_time = float(sensed_activation_time_ms) if sensed else -np.inf
    prior_update = float(np.asarray(state.last_update_time_ms))
    prior_sensed = float(np.asarray(state.last_sensed_time_ms))
    monotone = isfinite(time) and time >= prior_update
    sensed_valid = (not sensed) or (
        isfinite(sensed_time) and sensed_time >= prior_update and sensed_time <= time
    )
    cycle = float(np.asarray(state.cycle_length_ms))
    next_pulse = float(np.asarray(state.next_pulse_time_ms))
    last_sensed = prior_sensed
    if sensed and sensed_valid:
        if isfinite(prior_sensed):
            observed = sensed_time - prior_sensed
            cycle = float(
                np.clip(
                    plan.escape_interval_ms
                    + plan.feedback_gain * (observed - plan.target_sensed_interval_ms),
                    plan.minimum_cycle_length_ms,
                    plan.maximum_cycle_length_ms,
                )
            )
        last_sensed = sensed_time
        next_pulse = sensed_time + plan.escape_interval_ms
    emitted = monotone and sensed_valid and time >= next_pulse
    command_count = int(np.asarray(state.command_count))
    command_time = next_pulse if emitted else time
    if emitted:
        command_count += 1
        next_pulse = command_time + cycle
    finite = all(isfinite(value) for value in (time, cycle, next_pulse, command_time))
    successful = monotone and sensed_valid and finite
    candidate = DemandPacingControllerState(
        jnp.asarray(last_sensed),
        jnp.asarray(next_pulse),
        jnp.asarray(cycle),
        jnp.asarray(command_count, dtype=jnp.int32),
        jnp.asarray(time),
        plan.plan_id,
    )
    accepted = candidate if successful else state
    command = PacingPulseCommand(
        jnp.asarray(command_count - 1 if emitted else -1, dtype=jnp.int64),
        jnp.asarray(plan.site_index, dtype=jnp.int32),
        jnp.asarray(command_time),
        jnp.asarray(plan.duration_ms),
        jnp.asarray(plan.amplitude_uA_per_mm3),
        jnp.asarray(emitted),
    )
    evidence = PacingControllerEvidence(
        jnp.asarray(sensed),
        jnp.asarray(emitted),
        jnp.asarray(monotone and sensed_valid),
        jnp.asarray(finite),
        jnp.asarray(successful),
    )
    return PacingControllerResult(accepted, candidate, command, evidence)


__all__ = [
    "DemandPacingControllerPlan",
    "DemandPacingControllerState",
    "PMJActivationBatch",
    "PMJActivationEvidence",
    "PMJActivationResult",
    "PMJExchangeEvidence",
    "PMJExchangePlan",
    "PMJExchangeResult",
    "PMJStatus",
    "PacingControllerEvidence",
    "PacingControllerResult",
    "PacingEvaluation",
    "PacingProtocol",
    "PacingPulseCommand",
    "PacingTarget",
    "PurkinjePacingTarget",
    "TissuePacingTarget",
    "evaluate_pacing_protocol",
    "evaluate_pmj_exchange",
    "initialize_demand_pacing_controller",
    "pacing_purkinje_stimuli",
    "schedule_pmj_activations",
    "step_demand_pacing_controller",
]
