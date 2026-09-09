#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded, atomic neural evolution with endogenous events and native cell groups."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntFlag
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.internal import while_loop
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import localize_numerical_event
from ._cable import (
    CableStepInputs,
    initialize_cable_state,
    PreparedCableSolver,
    step_cable,
)
from ._events import (
    build_source_fanout,
    cancel_neural_events,
    enqueue_neural_event,
    initialize_event_queue,
    NeuralEventQueue,
    peek_neural_event_time,
    pop_neural_event,
    SourceFanout,
)
from ._ions import (
    evaluate_ion_concentration_transition,
    initialize_ion_concentrations,
    IonConcentrationState,
    PreparedIonDynamics,
)
from ._neurons import (
    AdaptiveExponentialIntegrateAndFire,
    advance_point_neuron,
    initialize_point_neuron,
    LeakyIntegrateAndFire,
    PointNeuronState,
    reset_point_neuron,
)
from ._protocol import CurrentClamp, VoltageClamp
from ._stochastic import (
    evaluate_stochastic_channel_transition,
    initialize_stochastic_channels,
    PreparedMarkovChannel,
    StochasticChannelState,
)
from ._synapses import (
    apply_synapse_arrivals,
    commit_eligibility_stdp,
    commit_pair_stdp,
    commit_synapse_relation_event,
    commit_synapse_relation_event_with_plasticity,
    decay_synapse_relations,
    EligibilitySTDPPlan,
    EligibilitySTDPState,
    evaluate_eligibility_stdp,
    evaluate_pair_stdp,
    evaluate_synapse_relation_event,
    initialize_eligibility_stdp,
    initialize_pair_stdp,
    initialize_synapse_network,
    PairSTDPPlan,
    PairSTDPState,
    PreparedSynapseNetwork,
    synapse_drive,
    SynapseNetworkPlan,
    SynapseRelationEvent,
    SynapseRelationState,
)


class NeuralStatus(IntFlag):
    SUCCESS = 0
    INVALID_INPUT = 1
    CELL_FAILURE = 2
    EVENT_CAPACITY = 4
    RECORDING_CAPACITY = 8
    EVENT_WORK_EXHAUSTED = 16
    LEARNING_FAILURE = 32
    AUXILIARY_FAILURE = 64
    ROOT_FAILURE = 128


class SpikeSource(StrictModule, NonTrainableState):
    """An externally scheduled event source, not a membrane-voltage model."""


class SpikeSourceState(StrictModule):
    time_ms: Array


class NeuralCellPlan(StrictModule):
    """Stable cell identity, physical model and one observable spike site.

    Cable thresholds report crossings without resetting the physical cell.
    SpikeSource endpoints have no physical voltage; their recording mask is false.
    """

    cell_id: str = eqx.field(static=True)
    model: (
        PreparedCableSolver
        | LeakyIntegrateAndFire
        | AdaptiveExponentialIntegrateAndFire
        | SpikeSource
    )
    detector_compartment: int = eqx.field(static=True)
    threshold_mV: Array
    rearm_mV: Array
    compartment_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        cell_id,
        model,
        /,
        *,
        detector_compartment=0,
        threshold_mV=None,
        rearm_mV=None,
    ):
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("cell_id must be a nonempty string.")
        if isinstance(model, PreparedCableSolver):
            ids = model.morphology.plan.compartment_ids
            threshold = 0.0 if threshold_mV is None else threshold_mV
        elif isinstance(
            model, (LeakyIntegrateAndFire, AdaptiveExponentialIntegrateAndFire)
        ):
            ids = ("soma",)
            if threshold_mV is not None:
                raise ValueError("Point-neuron thresholds belong to the physical model.")
            threshold = model.threshold_mV
        elif isinstance(model, SpikeSource):
            ids = ("source",)
            threshold = 0.0
        else:
            raise TypeError("Unsupported native neural cell model.")
        if isinstance(detector_compartment, bool) or not isinstance(
            detector_compartment, int
        ):
            raise TypeError("detector_compartment must be an integer.")
        if not 0 <= detector_compartment < len(ids):
            raise ValueError("Detector compartment exceeds cell layout.")
        threshold = jnp.asarray(threshold)
        rearm = threshold if rearm_mV is None else jnp.asarray(rearm_mV)
        if threshold.shape or rearm.shape:
            raise ValueError("Detector threshold and rearm voltage must be scalars.")
        if not np.isfinite(np.asarray(threshold)) or not np.isfinite(np.asarray(rearm)):
            raise ValueError("Detector voltages must be finite.")
        if float(rearm) > float(threshold):
            raise ValueError("Rearm voltage must not exceed threshold.")
        self.cell_id, self.model = cell_id, model
        self.detector_compartment = detector_compartment
        self.threshold_mV, self.rearm_mV = threshold, rearm
        self.compartment_ids = tuple(ids)


class NeuralIonCoupling(StrictModule, NonTrainableState):
    """Explicit ionic-current map: current(old_cell, new_cell) -> [species,compartment].

    The map supplies outward-positive nA consistent with the selected membrane
    mechanisms. Species attribution is never guessed from total membrane current.
    """

    cell_id: str = eqx.field(static=True)
    runtime: PreparedIonDynamics
    current: Callable = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    def __init__(self, cell_id, runtime, current, /, *, coupling_id):
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("cell_id must be a nonempty string.")
        if not isinstance(runtime, PreparedIonDynamics):
            raise TypeError("runtime must be prepared ion dynamics.")
        if not callable(current):
            raise TypeError("current must be callable.")
        if not isinstance(coupling_id, str) or not coupling_id:
            raise ValueError("coupling_id must be a nonempty stable identifier.")
        self.cell_id = cell_id
        self.runtime = runtime
        self.current = current
        self.coupling_id = coupling_id


class NeuralChannelCoupling(StrictModule):
    """Held channel conductance between fixed, independently prepared draw clocks."""

    cell_id: str = eqx.field(static=True)
    runtime: PreparedMarkovChannel
    open_state: int = eqx.field(static=True)
    single_channel_conductance_uS: Array
    reversal_mV: Array
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self, cell_id, runtime, open_state, single_channel_conductance_uS, reversal_mV, /
    ):
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("cell_id must be a nonempty string.")
        if not isinstance(runtime, PreparedMarkovChannel):
            raise TypeError("runtime must be a prepared Markov channel.")
        count = runtime.transition_probability.shape[0]
        if (
            isinstance(open_state, bool)
            or not isinstance(open_state, int)
            or not 0 <= open_state < count
        ):
            raise ValueError("open_state is outside the channel-state layout.")
        scale, reversal = float(single_channel_conductance_uS), float(reversal_mV)
        if not isfinite(scale) or scale < 0 or not isfinite(reversal):
            raise ValueError(
                "Channel conductance must be nonnegative and parameters finite."
            )
        self.cell_id, self.runtime, self.open_state = cell_id, runtime, open_state
        self.single_channel_conductance_uS = jnp.asarray(scale)
        self.reversal_mV = jnp.asarray(reversal)
        self.coupling_id = canonical_fingerprint(
            {
                "kind": "neural-channel-coupling",
                "cell_id": cell_id,
                "runtime": runtime.runtime_id,
                "open_state": open_state,
                "single_channel_conductance_uS": scale,
                "reversal_mV": reversal,
            }
        )


class NeuralNetworkPlan(StrictModule):
    """Prepared-capacity specification; dynamic morphology requires a new plan."""

    cells: tuple[NeuralCellPlan, ...]
    synapses: SynapseNetworkPlan
    learning: PairSTDPPlan | EligibilitySTDPPlan | None
    current_clamps: tuple[tuple[str, CurrentClamp], ...]
    voltage_clamps: tuple[tuple[str, VoltageClamp], ...]
    ion_couplings: tuple[NeuralIonCoupling, ...]
    channel_couplings: tuple[NeuralChannelCoupling, ...]
    external_spikes: tuple[tuple[float, str], ...] = eqx.field(static=True)
    queue_capacity: int = eqx.field(static=True)
    spike_capacity: int = eqx.field(static=True)
    recording_capacity: int = eqx.field(static=True)
    maximum_events_per_step: int = eqx.field(static=True)
    root_subdivisions: int = eqx.field(static=True)
    root_iterations: int = eqx.field(static=True)
    root_tolerance_ms: float = eqx.field(static=True)
    grazing_tolerance: float = eqx.field(static=True)
    modulation_scope: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cells: Sequence[NeuralCellPlan],
        synapses: SynapseNetworkPlan,
        /,
        *,
        queue_capacity=1024,
        spike_capacity=1024,
        recording_capacity=1024,
        maximum_events_per_step=128,
        root_subdivisions=4,
        root_iterations=48,
        root_tolerance_ms=1e-7,
        grazing_tolerance=1e-8,
        learning=None,
        modulation_scope="global",
        current_clamps=(),
        voltage_clamps=(),
        external_spikes=(),
        ion_couplings=(),
        channel_couplings=(),
    ):
        cells = tuple(cells)
        if not cells or any(not isinstance(cell, NeuralCellPlan) for cell in cells):
            raise ValueError("cells must contain native NeuralCellPlan values.")
        ids = tuple(cell.cell_id for cell in cells)
        if len(set(ids)) != len(ids):
            raise ValueError("Cell identifiers must be unique.")
        if (
            tuple(len(cell.compartment_ids) for cell in cells)
            != synapses.compartment_counts
        ):
            raise ValueError("Cell layouts and synapse endpoints disagree.")
        for name, value in (
            ("queue_capacity", queue_capacity),
            ("spike_capacity", spike_capacity),
            ("recording_capacity", recording_capacity),
            ("maximum_events_per_step", maximum_events_per_step),
            ("root_subdivisions", root_subdivisions),
            ("root_iterations", root_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if root_subdivisions > maximum_events_per_step:
            raise ValueError("Event work must admit at least all root subdivisions.")
        for value in (root_tolerance_ms, grazing_tolerance):
            if not isfinite(float(value)) or value <= 0:
                raise ValueError("Root tolerances must be positive and finite.")
        if learning is not None and not isinstance(
            learning, (PairSTDPPlan, EligibilitySTDPPlan)
        ):
            raise TypeError("learning must be a native STDP plan.")
        if modulation_scope not in ("global", "post", "relation"):
            raise ValueError("Unknown modulation scope.")
        currents, voltages = tuple(current_clamps), tuple(voltage_clamps)
        for cell_id, clamp in currents + voltages:
            if cell_id not in ids:
                raise ValueError("Clamp refers to an unknown cell.")
            cell = cells[ids.index(cell_id)]
            if clamp.compartment_id not in cell.compartment_ids or isinstance(
                cell.model, SpikeSource
            ):
                raise ValueError("Clamp refers to an unsupported physical endpoint.")
            if isinstance(clamp, VoltageClamp) and not isinstance(
                cell.model, PreparedCableSolver
            ):
                raise TypeError("Exact voltage clamps require a cable model.")
        for index, (cell_id, left) in enumerate(voltages):
            for other_id, right in voltages[index + 1 :]:
                if (
                    cell_id == other_id
                    and left.compartment_id == right.compartment_id
                    and left.start_ms < right.stop_ms
                    and right.start_ms < left.stop_ms
                ):
                    raise ValueError("Overlapping voltage commands target one endpoint.")
        spikes = []
        for time, cell_id in external_spikes:
            time = float(time)
            if not isfinite(time) or time < 0 or cell_id not in ids:
                raise ValueError(
                    "External spikes require finite nonnegative times and known cells."
                )
            if not isinstance(cells[ids.index(cell_id)].model, SpikeSource):
                raise TypeError(
                    "External spikes require a SpikeSource, not a physical reset."
                )
            if synapses.execution == "clock" and not np.isclose(
                time / synapses.dt_ms, round(time / synapses.dt_ms), rtol=0, atol=1e-9
            ):
                raise ValueError(
                    "Clock execution cannot represent an off-grid external spike."
                )
            spikes.append((time, cell_id))
        spikes.sort(key=lambda event: (event[0], ids.index(event[1])))
        ions, channels = tuple(ion_couplings), tuple(channel_couplings)
        if len({value.cell_id for value in ions}) != len(ions):
            raise ValueError("At most one ion coupling is allowed per cell.")
        for binding in ions + channels:
            if binding.cell_id not in ids:
                raise ValueError("Auxiliary dynamics refer to an unknown cell.")
            cell = cells[ids.index(binding.cell_id)]
            if isinstance(binding, NeuralIonCoupling) and not isinstance(
                cell.model, PreparedCableSolver
            ):
                raise TypeError("Ion concentration coupling requires a cable state.")
            if isinstance(cell.model, SpikeSource):
                raise TypeError("A SpikeSource cannot own membrane channels.")
        if synapses.execution == "clock":
            boundaries = [
                time
                for _, clamp in currents + voltages
                for time in (clamp.start_ms, clamp.stop_ms)
            ]
            boundaries.extend(binding.runtime.dt_ms for binding in channels)
            if any(
                not np.isclose(
                    value / synapses.dt_ms,
                    round(value / synapses.dt_ms),
                    rtol=0,
                    atol=1e-9,
                )
                for value in boundaries
            ):
                raise ValueError(
                    "Clock execution requires grid-resolved clamps and channel clocks."
                )
            if any(binding.runtime.dt_ms < synapses.dt_ms for binding in channels):
                raise ValueError(
                    "Channel update clocks cannot be finer than clock execution."
                )
        self.cells, self.synapses, self.learning = cells, synapses, learning
        self.current_clamps, self.voltage_clamps = currents, voltages
        self.external_spikes = tuple(spikes)
        self.ion_couplings, self.channel_couplings = ions, channels
        self.queue_capacity, self.spike_capacity = queue_capacity, spike_capacity
        self.recording_capacity, self.maximum_events_per_step = (
            recording_capacity,
            maximum_events_per_step,
        )
        self.root_subdivisions, self.root_iterations = root_subdivisions, root_iterations
        self.root_tolerance_ms, self.grazing_tolerance = (
            float(root_tolerance_ms),
            float(grazing_tolerance),
        )
        self.modulation_scope = modulation_scope
        self.plan_id = canonical_fingerprint(
            {
                "kind": "neural-network-plan",
                "cells": [
                    {
                        "cell_id": cell.cell_id,
                        "model": (
                            cell.model.runtime_id
                            if isinstance(cell.model, PreparedCableSolver)
                            else type(cell.model).__name__
                        ),
                        "detector_compartment": cell.detector_compartment,
                    }
                    for cell in cells
                ],
                "synapses": synapses.plan_id,
                "learning": None if learning is None else learning.plan_id,
                "current_clamps": [
                    [cell_id, clamp.stimulus_id] for cell_id, clamp in currents
                ],
                "voltage_clamps": [
                    [cell_id, clamp.stimulus_id] for cell_id, clamp in voltages
                ],
                "ion_couplings": [
                    [binding.cell_id, binding.runtime.runtime_id, binding.coupling_id]
                    for binding in ions
                ],
                "channel_couplings": [
                    [binding.cell_id, binding.coupling_id] for binding in channels
                ],
                "external_spikes": list(spikes),
                "queue_capacity": queue_capacity,
                "spike_capacity": spike_capacity,
                "recording_capacity": recording_capacity,
                "maximum_events_per_step": maximum_events_per_step,
                "root_subdivisions": root_subdivisions,
                "root_iterations": root_iterations,
                "root_tolerance_ms": float(root_tolerance_ms),
                "grazing_tolerance": float(grazing_tolerance),
                "modulation_scope": modulation_scope,
                "array_content": array_tree_fingerprint((cells, ions, channels)),
            }
        )

    def prepare(self):
        return prepare_neural_network(self)


class NeuralCellGroup(StrictModule, NonTrainableState):
    indices: tuple[int, ...] = eqx.field(static=True)
    endpoints: Array
    detector_indices: Array


class PreparedNeuralNetwork(StrictModule):
    plan: NeuralNetworkPlan
    synapses: PreparedSynapseNetwork
    groups: tuple[NeuralCellGroup, ...]
    cell_locations: tuple[tuple[int, int], ...] = eqx.field(static=True)
    physical_voltage_mask: Array
    detector_endpoints: Array
    external_times_ms: Array
    external_endpoints: Array
    runtime_id: str = eqx.field(static=True)


def prepare_neural_network(plan: NeuralNetworkPlan, /) -> PreparedNeuralNetwork:
    if not isinstance(plan, NeuralNetworkPlan):
        raise TypeError("plan must be a NeuralNetworkPlan.")
    grouped: list[list[int]] = []
    signatures = []
    for index, cell in enumerate(plan.cells):
        signature = (
            jax.tree.structure(cell.model),
            tuple((leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(cell.model)),
        )
        if signature in signatures:
            grouped[signatures.index(signature)].append(index)
        else:
            signatures.append(signature)
            grouped.append([index])
    offsets = plan.synapses.offsets
    groups, locations = [], [None] * len(plan.cells)
    for group_index, indices in enumerate(grouped):
        endpoints = [list(range(offsets[i], offsets[i + 1])) for i in indices]
        groups.append(
            NeuralCellGroup(
                tuple(indices),
                jnp.asarray(endpoints, dtype=jnp.int32),
                jnp.asarray(
                    [plan.cells[i].detector_compartment for i in indices], dtype=jnp.int32
                ),
            )
        )
        for within, index in enumerate(indices):
            locations[index] = (group_index, within)
    physical = [
        not isinstance(cell.model, SpikeSource)
        for cell in plan.cells
        for _ in cell.compartment_ids
    ]
    detectors = [
        offsets[i] + cell.detector_compartment for i, cell in enumerate(plan.cells)
    ]
    ids = tuple(cell.cell_id for cell in plan.cells)
    external_endpoints = [
        offsets[ids.index(cell_id)] for _, cell_id in plan.external_spikes
    ]
    identity = canonical_fingerprint(
        {
            "kind": "prepared-neural-network",
            "plan": plan.plan_id,
        }
    )
    return PreparedNeuralNetwork(
        plan,
        plan.synapses.prepare(),
        tuple(groups),
        tuple(locations),
        jnp.asarray(physical),
        jnp.asarray(detectors, dtype=jnp.int32),
        jnp.asarray([time for time, _ in plan.external_spikes]),
        jnp.asarray(external_endpoints, dtype=jnp.int32),
        identity,
    )


class NeuralSpikeRecording(StrictModule):
    time_ms: Array
    endpoint: Array
    count: Array


class NeuralRecording(StrictModule):
    time_ms: Array
    voltage_mV: Array
    count: Array


class NeuralNetworkState(StrictModule):
    groups: tuple
    relations: SynapseRelationState
    queue: NeuralEventQueue
    fanout: SourceFanout
    learning: PairSTDPState | EligibilitySTDPState | None
    ions: tuple[IonConcentrationState, ...]
    channels: tuple[StochasticChannelState, ...]
    next_channel_times_ms: Array
    armed: Array
    time_ms: Array
    step_index: Array
    external_cursor: Array
    spikes: NeuralSpikeRecording
    recording: NeuralRecording


class NeuralNetworkInputs(StrictModule):
    injected_current_nA: Array
    voltage_clamp_mask: Array
    voltage_clamp_target_mV: Array
    modulation: Array


class NeuralNetworkEvidence(StrictModule):
    status: Array
    successful: Array
    sensitivity_valid: Array
    event_batches: Array
    emitted_spikes: Array
    delivered_messages: Array
    maximum_queue_occupancy: Array


class NeuralNetworkResult(StrictModule):
    state: NeuralNetworkState
    evidence: NeuralNetworkEvidence


class NeuralNetworkRunResult(StrictModule):
    state: NeuralNetworkState
    voltage_mV: Array
    status: Array
    sensitivity_valid: Array
    emitted_spikes: Array
    delivered_messages: Array
    maximum_queue_occupancy: Array


class NeuralNetworkCheckpoint(StrictModule, NonTrainableState):
    state: NeuralNetworkState
    runtime_id: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)


def _stack(values):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *values)


def _group_models(runtime, group):
    return _stack([runtime.plan.cells[i].model for i in group.indices])


def _cell_state(runtime, groups, cell_index):
    group, index = runtime.cell_locations[cell_index]
    return jax.tree.map(lambda value: value[index], groups[group])


def _replace_cell(runtime, groups, cell_index, cell):
    group, index = runtime.cell_locations[cell_index]
    changed = jax.tree.map(
        lambda values, value: values.at[index].set(value), groups[group], cell
    )
    return tuple(changed if i == group else value for i, value in enumerate(groups))


def neural_voltage(runtime: PreparedNeuralNetwork, state: NeuralNetworkState, /) -> Array:
    """Flat endpoint voltages; consult physical_voltage_mask for source endpoints."""
    result = jnp.zeros((runtime.plan.synapses.endpoint_count,), dtype=state.time_ms.dtype)
    for group, values in zip(runtime.groups, state.groups, strict=True):
        if isinstance(values, SpikeSourceState):
            voltage = jnp.zeros(group.endpoints.shape, dtype=result.dtype)
        else:
            voltage = values.voltage_mV.reshape(group.endpoints.shape)
        result = result.at[group.endpoints.reshape(-1)].set(voltage.reshape(-1))
    return result


def initialize_neural_network(
    runtime: PreparedNeuralNetwork,
    voltage_mV=None,
    /,
    *,
    time_ms=0.0,
    intracellular_mM=(),
    extracellular_mM=(),
    channel_counts=(),
    key=None,
) -> NeuralNetworkState:
    plan = runtime.plan
    if not isfinite(float(time_ms)) or time_ms < 0:
        raise ValueError("Initial time must be finite and nonnegative.")
    if plan.synapses.execution == "clock" and not np.isclose(
        float(time_ms) / plan.synapses.dt_ms,
        round(float(time_ms) / plan.synapses.dt_ms),
        rtol=0,
        atol=1.0e-9,
    ):
        raise ValueError("Clock execution requires a grid-aligned initial time.")
    if len(intracellular_mM) != len(plan.ion_couplings) or len(extracellular_mM) != len(
        plan.ion_couplings
    ):
        raise ValueError("Each ion coupling requires explicit initial concentrations.")
    if len(channel_counts) != len(plan.channel_couplings) or (
        channel_counts and key is None
    ):
        raise ValueError(
            "Each channel coupling requires counts and an explicit random key."
        )
    if voltage_mV is not None:
        voltage_mV = jnp.asarray(voltage_mV)
        if voltage_mV.shape != (plan.synapses.endpoint_count,) or not np.all(
            np.isfinite(np.asarray(voltage_mV))
        ):
            raise ValueError("Initial voltage must be a finite flat endpoint vector.")
    ions = tuple(
        initialize_ion_concentrations(binding.runtime, inside, outside)
        for binding, inside, outside in zip(
            plan.ion_couplings, intracellular_mM, extracellular_mM, strict=True
        )
    )
    cells = []
    for index, cell in enumerate(plan.cells):
        start, stop = plan.synapses.offsets[index : index + 2]
        voltage = None if voltage_mV is None else voltage_mV[start:stop]
        if isinstance(cell.model, PreparedCableSolver):
            voltage = jnp.full((stop - start,), -65.0) if voltage is None else voltage
            matches = [
                i
                for i, binding in enumerate(plan.ion_couplings)
                if binding.cell_id == cell.cell_id
            ]
            ion = ions[matches[0]] if matches else None
            value = initialize_cable_state(
                cell.model,
                voltage,
                intracellular_mM=None if ion is None else ion.intracellular_mM,
                extracellular_mM=None if ion is None else ion.extracellular_mM,
            )
            value = eqx.tree_at(
                lambda s: s.time_ms, value, jnp.asarray(time_ms, dtype=voltage.dtype)
            )
        elif isinstance(cell.model, SpikeSource):
            value = SpikeSourceState(jnp.asarray(float(time_ms)))
        else:
            value = initialize_point_neuron(
                cell.model, None if voltage is None else voltage[0], time_ms=time_ms
            )
            if float(value.voltage_mV) >= float(cell.model.threshold_mV):
                raise ValueError(
                    "A point-neuron initial voltage must be below its spike cutoff."
                )
        cells.append(value)
    groups = tuple(_stack([cells[i] for i in group.indices]) for group in runtime.groups)
    dtype = jnp.result_type(*[jax.tree.leaves(value)[0] for value in cells], float)
    channels = tuple(
        initialize_stochastic_channels(
            binding.runtime, counts, jax.random.fold_in(key, index)
        )
        for index, (binding, counts) in enumerate(
            zip(plan.channel_couplings, channel_counts, strict=True)
        )
    )
    synapses = initialize_synapse_network(runtime.synapses)
    if isinstance(plan.learning, EligibilitySTDPPlan):
        learning = initialize_eligibility_stdp(runtime.synapses)
    elif isinstance(plan.learning, PairSTDPPlan):
        learning = initialize_pair_stdp(runtime.synapses)
    else:
        learning = None
    state = NeuralNetworkState(
        groups,
        synapses.relations,
        initialize_event_queue(plan.queue_capacity, dtype=dtype),
        build_source_fanout(synapses.relations, plan.synapses.endpoint_count),
        learning,
        ions,
        channels,
        jnp.asarray(
            [time_ms + binding.runtime.dt_ms for binding in plan.channel_couplings],
            dtype=dtype,
        ),
        jnp.zeros((plan.synapses.endpoint_count,), dtype=bool),
        jnp.asarray(time_ms, dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(sum(t < time_ms for t, _ in plan.external_spikes), dtype=jnp.int32),
        NeuralSpikeRecording(
            jnp.zeros((plan.spike_capacity,), dtype=dtype),
            jnp.full((plan.spike_capacity,), -1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        NeuralRecording(
            jnp.zeros((plan.recording_capacity,), dtype=dtype),
            jnp.zeros(
                (plan.recording_capacity, plan.synapses.endpoint_count), dtype=dtype
            ),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return eqx.tree_at(
        lambda value: value.armed, state, _rearm(runtime, state, state.armed)
    )


def zero_neural_inputs(
    runtime: PreparedNeuralNetwork, /, *, dtype=None
) -> NeuralNetworkInputs:
    zeros = jnp.zeros((runtime.plan.synapses.endpoint_count,), dtype=dtype)
    scope = runtime.plan.modulation_scope
    shape = (
        ()
        if scope == "global"
        else (
            (runtime.plan.synapses.endpoint_count,)
            if scope == "post"
            else (runtime.plan.synapses.synapse_capacity,)
        )
    )
    return NeuralNetworkInputs(
        zeros,
        jnp.zeros(zeros.shape, dtype=bool),
        zeros,
        jnp.zeros(shape, dtype=zeros.dtype),
    )


def _rearm(runtime, state, armed):
    voltage = neural_voltage(runtime, state)
    for index, cell in enumerate(runtime.plan.cells):
        endpoint = runtime.plan.synapses.offsets[index] + cell.detector_compartment
        if not isinstance(cell.model, SpikeSource):
            threshold = (
                cell.threshold_mV
                if isinstance(cell.model, PreparedCableSolver)
                else cell.model.threshold_mV
            )
            can_arm = (voltage[endpoint] <= cell.rearm_mV) & (
                voltage[endpoint] < threshold
            )
            armed = armed.at[endpoint].set(armed[endpoint] | can_arm)
    return armed


def _stimulus(runtime, inputs, time):
    injected, mask, target = (
        inputs.injected_current_nA,
        inputs.voltage_clamp_mask,
        inputs.voltage_clamp_target_mV,
    )
    ids = tuple(cell.cell_id for cell in runtime.plan.cells)
    for cell_id, clamp in runtime.plan.current_clamps:
        index = ids.index(cell_id)
        endpoint = runtime.plan.synapses.offsets[index] + runtime.plan.cells[
            index
        ].compartment_ids.index(clamp.compartment_id)
        active = (time >= clamp.start_ms) & (time < clamp.stop_ms)
        injected = injected.at[endpoint].add(jnp.where(active, clamp.amplitude_nA, 0.0))
    for cell_id, clamp in runtime.plan.voltage_clamps:
        index = ids.index(cell_id)
        endpoint = runtime.plan.synapses.offsets[index] + runtime.plan.cells[
            index
        ].compartment_ids.index(clamp.compartment_id)
        active = (time >= clamp.start_ms) & (time < clamp.stop_ms)
        target = target.at[endpoint].set(
            jnp.where(active, clamp.target_mV, target[endpoint])
        )
        mask = mask.at[endpoint].set(mask[endpoint] | active)
    return NeuralNetworkInputs(injected, mask, target, inputs.modulation)


def _drive(runtime, state, elapsed):
    x = elapsed / state.relations.time_constant_ms
    factor = -jnp.expm1(-x) / jnp.where(x == 0, 1.0, x)
    averaged = eqx.tree_at(
        lambda value: value.activation,
        state.relations,
        state.relations.activation * jnp.where(x == 0, 1.0, factor),
    )
    conductance, offset = synapse_drive(
        runtime.synapses, averaged, neural_voltage(runtime, state)
    )
    ids = tuple(cell.cell_id for cell in runtime.plan.cells)
    for binding, channel in zip(
        runtime.plan.channel_couplings, state.channels, strict=True
    ):
        index = ids.index(binding.cell_id)
        start, stop = runtime.plan.synapses.offsets[index : index + 2]
        values = (
            channel.counts[:, binding.open_state] * binding.single_channel_conductance_uS
        )
        conductance = conductance.at[start:stop].add(values)
        offset = offset.at[start:stop].add(-values * binding.reversal_mV)
    return conductance, offset


def _advance_one(
    model, state, elapsed, injected, conductance, offset, clamp_mask, clamp_target, time
):
    if isinstance(model, PreparedCableSolver):
        result = step_cable(
            model,
            state,
            CableStepInputs(injected, conductance, offset, clamp_mask, clamp_target),
            elapsed_ms=elapsed,
        )
        return result.state, result.evidence.successful
    if isinstance(model, SpikeSource):
        return SpikeSourceState(time + elapsed), jnp.asarray(True)
    value = advance_point_neuron(
        model, state, elapsed, injected[0], conductance[0], offset[0], time_ms=time
    )
    return value, jnp.all(jnp.isfinite(value.voltage_mV)) & jnp.all(
        jnp.isfinite(value.adaptation_nA)
    )


def _flow(runtime, state, elapsed, inputs):
    conductance, offset = _drive(runtime, state, elapsed)
    inputs = _stimulus(runtime, inputs, state.time_ms)
    groups, successful = [], jnp.asarray(True)
    for group, values in zip(runtime.groups, state.groups, strict=True):
        models = _group_models(runtime, group)
        indices = group.endpoints
        advanced, valid = eqx.filter_vmap(
            lambda model, value, current, g, o, mask, target: _advance_one(
                model, value, elapsed, current, g, o, mask, target, state.time_ms
            )
        )(
            models,
            values,
            inputs.injected_current_nA[indices],
            conductance[indices],
            offset[indices],
            inputs.voltage_clamp_mask[indices],
            inputs.voltage_clamp_target_mV[indices],
        )
        groups.append(advanced)
        successful = successful & jnp.all(valid)
    groups = tuple(groups)
    ions = []
    ids = tuple(cell.cell_id for cell in runtime.plan.cells)
    for binding, ion in zip(runtime.plan.ion_couplings, state.ions, strict=True):
        index = ids.index(binding.cell_id)
        old, new = (
            _cell_state(runtime, state.groups, index),
            _cell_state(runtime, groups, index),
        )
        candidate = evaluate_ion_concentration_transition(
            binding.runtime, ion, binding.current(old, new), elapsed
        )
        ions.append(candidate.proposed)
        successful = successful & candidate.evidence.successful
        new = eqx.tree_at(
            lambda cell: (cell.intracellular_mM, cell.extracellular_mM),
            new,
            (candidate.proposed.intracellular_mM, candidate.proposed.extracellular_mM),
        )
        groups = _replace_cell(runtime, groups, index, new)
    return eqx.tree_at(
        lambda value: (value.groups, value.ions, value.relations, value.time_ms),
        state,
        (
            groups,
            tuple(ions),
            decay_synapse_relations(state.relations, elapsed),
            state.time_ms + elapsed,
        ),
    ), successful


def _flow_or_identity(runtime, state, elapsed, inputs):
    return jax.lax.cond(
        elapsed > 0,
        lambda _: _flow(runtime, state, elapsed, inputs),
        lambda _: (state, jnp.asarray(True)),
        operand=None,
    )


def _next_boundary(runtime, state, target):
    time = state.time_ms
    end = jnp.minimum(
        target, time + runtime.plan.synapses.dt_ms / runtime.plan.root_subdivisions
    )
    end = jnp.minimum(end, peek_neural_event_time(state.queue))
    if runtime.plan.external_spikes:
        index = jnp.minimum(state.external_cursor, len(runtime.plan.external_spikes) - 1)
        end = jnp.minimum(
            end,
            jnp.where(
                state.external_cursor < len(runtime.plan.external_spikes),
                runtime.external_times_ms[index],
                jnp.inf,
            ),
        )
    if runtime.plan.channel_couplings:
        end = jnp.minimum(end, jnp.min(state.next_channel_times_ms))
    for _, clamp in runtime.plan.current_clamps + runtime.plan.voltage_clamps:
        end = jnp.minimum(end, jnp.where(clamp.start_ms > time, clamp.start_ms, jnp.inf))
        end = jnp.minimum(end, jnp.where(clamp.stop_ms > time, clamp.stop_ms, jnp.inf))
    for values in state.groups:
        if isinstance(values, PointNeuronState):
            release = jnp.min(
                jnp.where(
                    values.refractory_until_ms > time, values.refractory_until_ms, jnp.inf
                )
            )
            end = jnp.minimum(end, release)
    return end


def _thresholds(runtime):
    return jnp.stack(
        [
            cell.model.threshold_mV
            if isinstance(
                cell.model, (LeakyIntegrateAndFire, AdaptiveExponentialIntegrateAndFire)
            )
            else cell.threshold_mV
            for cell in runtime.plan.cells
        ]
    )


def _localize(runtime, state, candidate, elapsed, inputs):
    initial, final = neural_voltage(runtime, state), neural_voltage(runtime, candidate)
    threshold = _thresholds(runtime)
    endpoints = runtime.detector_endpoints
    enabled = state.armed[endpoints] & runtime.physical_voltage_mask[endpoints]
    crossed = enabled & (initial[endpoints] < threshold) & (final[endpoints] >= threshold)
    immediate = enabled & (initial[endpoints] >= threshold)
    roots = jnp.full((len(runtime.plan.cells),), jnp.inf, dtype=state.time_ms.dtype)
    derivative_valid, root_valid = jnp.asarray(True), jnp.asarray(True)
    inputs = _stimulus(runtime, inputs, state.time_ms)
    for group, values in zip(runtime.groups, state.groups, strict=True):
        indices = jnp.asarray(group.indices)
        models = _group_models(runtime, group)
        ep = group.endpoints

        # The endpoint row is an array argument; no dynamic Python cell indexing.
        def locate(
            model, value, detector, active, level, current, mask, clamp, endpoint_row
        ):
            def at_time(h):
                g, o = _drive(runtime, state, h)
                advanced, _ = _advance_one(
                    model,
                    value,
                    h,
                    current,
                    g[endpoint_row],
                    o[endpoint_row],
                    mask,
                    clamp,
                    state.time_ms,
                )
                if isinstance(advanced, SpikeSourceState):
                    return jnp.asarray(0.0, dtype=elapsed.dtype)
                return advanced.voltage_mV.reshape(-1)[detector]

            def guard(h, voltage):
                return jnp.where(active, voltage - level, h - elapsed * 0.5)

            result = localize_numerical_event(
                at_time,
                guard,
                jnp.zeros_like(elapsed),
                elapsed,
                iterations=runtime.plan.root_iterations,
                tolerance=runtime.plan.root_tolerance_ms,
                grazing_tolerance=runtime.plan.grazing_tolerance,
            )
            return (
                jnp.where(active, result.event_time, jnp.inf),
                (~active | result.successful),
                (~active | result.derivative_valid),
            )

        found, good, transverse = eqx.filter_vmap(locate)(
            models,
            values,
            group.detector_indices,
            crossed[indices],
            threshold[indices],
            inputs.injected_current_nA[ep],
            inputs.voltage_clamp_mask[ep],
            inputs.voltage_clamp_target_mV[ep],
            ep,
        )
        roots = roots.at[indices].set(found)
        derivative_valid = derivative_valid & jnp.all(transverse)
        root_valid = root_valid & jnp.all(good)
    roots = jnp.where(immediate, 0.0, roots)
    earliest = jnp.minimum(jnp.min(roots), elapsed)
    fired = (
        jnp.zeros_like(state.armed)
        .at[endpoints]
        .set(jnp.isfinite(roots) & (roots == earliest))
    )
    nearby = jnp.sum(
        jnp.isfinite(roots)
        & (jnp.abs(roots - earliest) <= runtime.plan.root_tolerance_ms)
    )
    derivative_valid = derivative_valid & (nearby <= 1) & ~jnp.any(immediate)
    return earliest, fired, root_valid, derivative_valid


def _external(runtime, state, counts):
    if not runtime.plan.external_spikes:
        return state, counts
    total = len(runtime.plan.external_spikes)

    def condition(carry):
        cursor, _ = carry
        return (cursor < total) & (
            runtime.external_times_ms[jnp.minimum(cursor, total - 1)] <= state.time_ms
        )

    def body(carry):
        cursor, values = carry
        endpoint = runtime.external_endpoints[cursor]
        one = jnp.asarray(1, dtype=values.dtype)
        return cursor + one, values.at[endpoint].add(one)

    cursor, counts = while_loop(
        condition, body, (state.external_cursor, counts), max_steps=total, kind="bounded"
    )
    return eqx.tree_at(lambda value: value.external_cursor, state, cursor), counts


def _emit(runtime, state, counts):
    source_indices = jnp.nonzero(counts, size=counts.shape[0], fill_value=0)[0]
    source_count = jnp.count_nonzero(counts)

    def condition(carry):
        source, _, _, queue_valid, recording_valid, _ = carry
        return (source < source_count) & queue_valid & recording_valid

    def body(carry):
        source, queue, recording, queue_valid, recording_valid, occupancy = carry
        endpoint = source_indices[source]
        count = counts[endpoint]
        first, stop = state.fanout.offsets[endpoint], state.fanout.offsets[endpoint + 1]

        def send_condition(inner):
            slot_index, _, okay = inner
            return (slot_index < stop) & okay

        def send(inner):
            slot_index, events, okay = inner
            slot = state.fanout.slots[slot_index]
            events, accepted = enqueue_neural_event(
                events,
                state.time_ms + state.relations.delay_ms[slot],
                slot,
                state.relations.generation[slot],
                state.relations.weight[slot] * count,
                count=count,
            )
            return slot_index + 1, events, okay & accepted

        _, queue, queue_ok = while_loop(
            send_condition,
            send,
            (first, queue, queue_valid),
            max_steps=runtime.plan.synapses.synapse_capacity,
            kind="bounded",
        )
        room = recording.count + count <= runtime.plan.spike_capacity

        def record_condition(inner):
            written, _ = inner
            return (written < count) & room

        def record(inner):
            written, value = inner
            index = value.count
            value = NeuralSpikeRecording(
                value.time_ms.at[index].set(state.time_ms),
                value.endpoint.at[index].set(endpoint.astype(value.endpoint.dtype)),
                index + 1,
            )
            return written + 1, value

        _, recording = while_loop(
            record_condition,
            record,
            (jnp.asarray(0, dtype=jnp.int32), recording),
            max_steps=runtime.plan.spike_capacity,
            kind="bounded",
        )
        return (
            source + 1,
            queue,
            recording,
            queue_valid & queue_ok,
            recording_valid & room,
            jnp.maximum(occupancy, queue.size),
        )

    _, queue, recording, queue_valid, recording_valid, occupancy = while_loop(
        condition,
        body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            state.queue,
            state.spikes,
            jnp.asarray(True),
            jnp.asarray(True),
            state.queue.size,
        ),
        max_steps=counts.shape[0],
        kind="bounded",
    )
    proposed = eqx.tree_at(
        lambda value: (value.queue, value.spikes), state, (queue, recording)
    )
    return proposed, queue_valid, recording_valid, occupancy


def _deliver(runtime, state):
    capacity = runtime.plan.synapses.synapse_capacity
    amplitudes = jnp.zeros((capacity,), dtype=state.relations.weight.dtype)
    counts = jnp.zeros((capacity,), dtype=jnp.int32)

    def condition(carry):
        queue, _, _, _ = carry
        return (queue.size > 0) & (peek_neural_event_time(queue) <= state.time_ms)

    def body(carry):
        queue, amounts, arrivals, delivered = carry
        queue, _, slot, generation, amplitude, count, popped = pop_neural_event(queue)
        valid = (
            popped
            & state.relations.active[slot]
            & (state.relations.generation[slot] == generation)
        )
        return (
            queue,
            amounts.at[slot].add(jnp.where(valid, amplitude, 0.0)),
            arrivals.at[slot].add(jnp.where(valid, count, 0)),
            delivered + valid.astype(jnp.int32),
        )

    queue, amplitudes, counts, delivered = while_loop(
        condition,
        body,
        (state.queue, amplitudes, counts, jnp.asarray(0, dtype=jnp.int32)),
        max_steps=runtime.plan.queue_capacity,
        kind="bounded",
    )
    relations = apply_synapse_arrivals(state.relations, amplitudes)
    return (
        eqx.tree_at(
            lambda value: (value.queue, value.relations), state, (queue, relations)
        ),
        counts,
        delivered,
    )


def _interventions(runtime, state, fired):
    groups = state.groups
    for group, values in zip(runtime.groups, groups, strict=True):
        if isinstance(values, PointNeuronState):
            models = _group_models(runtime, group)
            fire = fired[group.endpoints[:, 0]]
            reset = eqx.filter_vmap(
                lambda model, value: reset_point_neuron(model, value, state.time_ms)
            )(models, values)
            changed = jax.tree.map(
                lambda new, old, fire=fire: jnp.where(fire, new, old), reset, values
            )
            group_index = runtime.cell_locations[group.indices[0]][0]
            groups = tuple(
                changed if i == group_index else value for i, value in enumerate(groups)
            )
    state = eqx.tree_at(
        lambda value: (value.groups, value.armed), state, (groups, state.armed & ~fired)
    )
    return eqx.tree_at(
        lambda value: value.armed, state, _rearm(runtime, state, state.armed)
    )


def _learn(runtime, state, counts, arrivals, elapsed, modulation):
    plan = runtime.plan.learning
    if plan is None:
        return state, jnp.asarray(True)
    if isinstance(plan, EligibilitySTDPPlan):
        candidate = evaluate_eligibility_stdp(
            runtime.synapses,
            plan,
            state.relations,
            state.learning,
            counts,
            counts,
            elapsed_ms=elapsed,
            presynaptic_arrivals=arrivals,
            modulation=modulation,
            modulation_scope=runtime.plan.modulation_scope,
        )
        relations, learning = commit_eligibility_stdp(
            candidate, state.relations, state.learning
        )
    else:
        candidate = evaluate_pair_stdp(
            runtime.synapses,
            plan,
            state.relations,
            state.learning,
            counts,
            counts,
            elapsed_ms=elapsed,
            presynaptic_arrivals=arrivals,
        )
        relations, learning = commit_pair_stdp(candidate, state.relations, state.learning)
    return eqx.tree_at(
        lambda value: (value.relations, value.learning), state, (relations, learning)
    ), candidate.successful


def _channel_updates(runtime, state):
    channels, times, valid = (
        list(state.channels),
        state.next_channel_times_ms,
        jnp.asarray(True),
    )
    for index, binding in enumerate(runtime.plan.channel_couplings):
        due = times[index] <= state.time_ms

        def update(_, binding=binding, channel=channels[index]):
            candidate = evaluate_stochastic_channel_transition(binding.runtime, channel)
            return candidate.proposed, candidate.evidence.successful

        current = channels[index]
        channels[index], okay = jax.lax.cond(
            due,
            update,
            lambda _, current=current: (current, jnp.asarray(True)),
            operand=None,
        )
        times = times.at[index].add(jnp.where(due, binding.runtime.dt_ms, 0.0))
        valid = valid & okay
    return eqx.tree_at(
        lambda value: (value.channels, value.next_channel_times_ms),
        state,
        (tuple(channels), times),
    ), valid


def _event_transition(runtime, state, target, inputs):
    end = _next_boundary(runtime, state, target)
    elapsed = jnp.maximum(end - state.time_ms, 0.0)
    candidate, flow_ok = _flow_or_identity(runtime, state, elapsed, inputs)

    def localize(_):
        return _localize(runtime, state, candidate, elapsed, inputs)

    duration, fired, root_ok, derivative_valid = jax.lax.cond(
        elapsed > 0,
        localize,
        lambda _: (
            elapsed,
            jnp.zeros_like(state.armed),
            jnp.asarray(True),
            jnp.asarray(True),
        ),
        operand=None,
    )
    candidate, accepted_flow = jax.lax.cond(
        duration < elapsed,
        lambda _: _flow_or_identity(runtime, state, duration, inputs),
        lambda _: (candidate, flow_ok),
        operand=None,
    )
    candidate, counts = _external(runtime, candidate, fired.astype(jnp.int32))
    candidate, queue_ok, recording_ok, occupancy = _emit(runtime, candidate, counts)
    candidate, arrivals, delivered = _deliver(runtime, candidate)
    candidate = _interventions(runtime, candidate, fired)
    candidate, learning_ok = _learn(
        runtime,
        candidate,
        counts,
        arrivals,
        duration,
        jnp.zeros_like(inputs.modulation),
    )
    candidate, channel_ok = _channel_updates(runtime, candidate)
    status = jnp.asarray(0, dtype=jnp.int32)
    status = status | jnp.where(accepted_flow, 0, int(NeuralStatus.CELL_FAILURE))
    status = status | jnp.where(root_ok, 0, int(NeuralStatus.ROOT_FAILURE))
    status = status | jnp.where(queue_ok, 0, int(NeuralStatus.EVENT_CAPACITY))
    status = status | jnp.where(recording_ok, 0, int(NeuralStatus.RECORDING_CAPACITY))
    status = status | jnp.where(learning_ok, 0, int(NeuralStatus.LEARNING_FAILURE))
    status = status | jnp.where(channel_ok, 0, int(NeuralStatus.AUXILIARY_FAILURE))
    return (
        candidate,
        status,
        derivative_valid,
        jnp.sum(counts, dtype=jnp.int32),
        delivered,
        occupancy,
    )


def _clock_transition(runtime, state, target, inputs):
    # Sources and arrivals at the left boundary are consumed exactly once.
    zeros = jnp.zeros_like(state.armed, dtype=jnp.int32)
    state, initial_counts = _external(runtime, state, zeros)
    state, initial_queue_ok, initial_recording_ok, occupancy = _emit(
        runtime, state, initial_counts
    )
    state, initial_arrivals, delivered = _deliver(runtime, state)
    state, initial_learning_ok = _learn(
        runtime,
        state,
        initial_counts,
        initial_arrivals,
        jnp.zeros_like(state.time_ms),
        jnp.zeros_like(inputs.modulation),
    )
    prior = state
    elapsed = target - state.time_ms
    state, flow_ok = _flow_or_identity(runtime, state, elapsed, inputs)
    voltage = neural_voltage(runtime, state)
    threshold = _thresholds(runtime)
    endpoints = runtime.detector_endpoints
    fired_values = (
        prior.armed[endpoints]
        & runtime.physical_voltage_mask[endpoints]
        & (voltage[endpoints] >= threshold)
    )
    fired = jnp.zeros_like(state.armed).at[endpoints].set(fired_values)
    state, counts = _external(runtime, state, fired.astype(jnp.int32))
    state, queue_ok, recording_ok, later_occupancy = _emit(runtime, state, counts)
    state, arrivals, later_delivered = _deliver(runtime, state)
    state = _interventions(runtime, state, fired)
    state, learning_ok = _learn(
        runtime,
        state,
        counts,
        arrivals,
        elapsed,
        jnp.zeros_like(inputs.modulation),
    )
    state, channel_ok = _channel_updates(runtime, state)
    status = jnp.asarray(0, dtype=jnp.int32)
    status = status | jnp.where(flow_ok, 0, int(NeuralStatus.CELL_FAILURE))
    status = status | jnp.where(
        initial_queue_ok & queue_ok, 0, int(NeuralStatus.EVENT_CAPACITY)
    )
    status = status | jnp.where(
        initial_recording_ok & recording_ok,
        0,
        int(NeuralStatus.RECORDING_CAPACITY),
    )
    status = status | jnp.where(
        initial_learning_ok & learning_ok,
        0,
        int(NeuralStatus.LEARNING_FAILURE),
    )
    status = status | jnp.where(channel_ok, 0, int(NeuralStatus.AUXILIARY_FAILURE))
    return (
        state,
        status,
        jnp.asarray(True),
        jnp.sum(initial_counts + counts, dtype=jnp.int32),
        delivered + later_delivered,
        jnp.maximum(occupancy, later_occupancy),
    )


@jax.custom_jvp
def _sensitivity_gate(value, valid):
    return value


@_sensitivity_gate.defjvp
def _sensitivity_gate_jvp(primals, tangents):
    value, valid = primals
    tangent, _ = tangents
    return value, tangent * jnp.where(
        valid, jnp.ones_like(value), jnp.full_like(value, jnp.nan)
    )


def step_neural_network(
    runtime: PreparedNeuralNetwork,
    state: NeuralNetworkState,
    inputs: NeuralNetworkInputs | None = None,
    /,
) -> NeuralNetworkResult:
    """Advance one requested dt atomically; events and auxiliary draws are bounded.

    Event-mode gradients are conditional on a separated, transverse numerical
    event realization. Invalid sensitivities have NaN tangents without changing
    an otherwise accepted forward state. Clock gradients hold threshold choices.
    """
    if inputs is None:
        inputs = zero_neural_inputs(runtime, dtype=state.time_ms.dtype)
    count = runtime.plan.synapses.endpoint_count
    if any(
        value.shape != (count,)
        for value in (
            inputs.injected_current_nA,
            inputs.voltage_clamp_mask,
            inputs.voltage_clamp_target_mV,
        )
    ):
        raise ValueError("Neural inputs must use flat endpoint vectors.")
    expected_modulation = zero_neural_inputs(
        runtime, dtype=state.time_ms.dtype
    ).modulation.shape
    if inputs.modulation.shape != expected_modulation:
        raise ValueError("Modulation shape does not match the prepared support.")
    valid_input = (
        jnp.all(jnp.isfinite(inputs.injected_current_nA))
        & jnp.all(jnp.isfinite(inputs.voltage_clamp_target_mV))
        & jnp.all(jnp.isfinite(inputs.modulation))
    )
    for index, cell in enumerate(runtime.plan.cells):
        start, stop = runtime.plan.synapses.offsets[index : index + 2]
        if not isinstance(cell.model, PreparedCableSolver):
            valid_input = valid_input & ~jnp.any(inputs.voltage_clamp_mask[start:stop])
        if isinstance(cell.model, SpikeSource):
            valid_input = valid_input & jnp.all(
                inputs.injected_current_nA[start:stop] == 0
            )
    target = state.time_ms + runtime.plan.synapses.dt_ms
    if runtime.plan.synapses.execution == "clock":
        target = (
            jnp.rint(state.time_ms / runtime.plan.synapses.dt_ms) + 1
        ) * runtime.plan.synapses.dt_ms
        candidate, status, derivative_valid, emitted, delivered, occupancy = (
            _clock_transition(runtime, state, target, inputs)
        )
        batches = jnp.asarray(1, dtype=jnp.int32)
    else:

        def condition(carry):
            current, status, _, _, _, _, _ = carry
            return (current.time_ms < target) & (status == 0)

        def body(carry):
            current, status, valid, batches, emissions, deliveries, occupancy = carry
            changed, next_status, transverse, emitted, delivered, next_occupancy = (
                _event_transition(runtime, current, target, inputs)
            )
            return (
                changed,
                status | next_status,
                valid & transverse,
                batches + 1,
                emissions + emitted,
                deliveries + delivered,
                jnp.maximum(occupancy, next_occupancy),
            )

        initial = (
            state,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            state.queue.size,
        )
        candidate, status, derivative_valid, batches, emitted, delivered, occupancy = (
            while_loop(
                condition,
                jax.checkpoint(body),
                initial,
                max_steps=runtime.plan.maximum_events_per_step,
                kind="bounded",
            )
        )
        status = status | jnp.where(
            candidate.time_ms >= target, 0, int(NeuralStatus.EVENT_WORK_EXHAUSTED)
        )
    zeros = jnp.zeros((count,), dtype=jnp.int32)
    candidate, learned = _learn(
        runtime,
        candidate,
        zeros,
        jnp.zeros((runtime.plan.synapses.synapse_capacity,), dtype=jnp.int32),
        jnp.zeros_like(state.time_ms),
        inputs.modulation,
    )
    status = status | jnp.where(learned, 0, int(NeuralStatus.LEARNING_FAILURE))
    status = status | jnp.where(valid_input, 0, int(NeuralStatus.INVALID_INPUT))
    room = candidate.recording.count < runtime.plan.recording_capacity
    status = status | jnp.where(room, 0, int(NeuralStatus.RECORDING_CAPACITY))
    successful = status == 0
    slot = jnp.minimum(candidate.recording.count, runtime.plan.recording_capacity - 1)
    recording = NeuralRecording(
        candidate.recording.time_ms.at[slot].set(candidate.time_ms),
        candidate.recording.voltage_mV.at[slot].set(neural_voltage(runtime, candidate)),
        candidate.recording.count + 1,
    )
    candidate = eqx.tree_at(
        lambda value: (value.recording, value.step_index),
        candidate,
        (recording, state.step_index + 1),
    )
    accepted = jax.tree.map(
        lambda proposed, old: jnp.where(successful, proposed, old), candidate, state
    )
    if runtime.plan.synapses.execution == "event":
        accepted = jax.tree.map(
            lambda value: (
                _sensitivity_gate(value, derivative_valid & successful)
                if eqx.is_inexact_array(value)
                else value
            ),
            accepted,
        )
    return NeuralNetworkResult(
        accepted,
        NeuralNetworkEvidence(
            status,
            successful,
            derivative_valid & successful,
            batches,
            emitted,
            delivered,
            occupancy,
        ),
    )


def run_neural_network(
    runtime: PreparedNeuralNetwork,
    state: NeuralNetworkState,
    steps: int,
    /,
    *,
    inputs: NeuralNetworkInputs | None = None,
) -> NeuralNetworkRunResult:
    """Fixed-length continuation; after the first failure subsequent steps retain state."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")

    def advance(carry, _):
        current, failed_status = carry

        def run(_):
            result = step_neural_network(runtime, current, inputs)
            return result.state, result.evidence

        def stopped(_):
            zero = jnp.asarray(0, dtype=jnp.int32)
            return current, NeuralNetworkEvidence(
                failed_status,
                jnp.asarray(False),
                jnp.asarray(False),
                zero,
                zero,
                zero,
                current.queue.size,
            )

        current, evidence = jax.lax.cond(failed_status == 0, run, stopped, operand=None)
        return (current, failed_status | evidence.status), (
            neural_voltage(runtime, current),
            evidence.status,
            evidence.sensitivity_valid,
            evidence.emitted_spikes,
            evidence.delivered_messages,
            evidence.maximum_queue_occupancy,
        )

    (state, _), values = jax.lax.scan(
        advance, (state, jnp.asarray(0, dtype=jnp.int32)), xs=None, length=steps
    )
    return NeuralNetworkRunResult(state, *values)


def apply_neural_relation_event(
    runtime: PreparedNeuralNetwork,
    state: NeuralNetworkState,
    event: SynapseRelationEvent,
    /,
) -> NeuralNetworkResult:
    """Apply one boundary structural transaction, including pending-event cancellation."""
    candidate = evaluate_synapse_relation_event(runtime.synapses, state.relations, event)
    if state.learning is None:
        relations = commit_synapse_relation_event(candidate, state.relations)
        learning = None
    else:
        relations, learning = commit_synapse_relation_event_with_plasticity(
            candidate, state.relations, state.learning
        )
    queue = cancel_neural_events(state.queue, relations.active, relations.generation)
    fanout = build_source_fanout(relations, runtime.plan.synapses.endpoint_count)
    proposed = eqx.tree_at(
        lambda value: (value.relations, value.learning, value.queue, value.fanout),
        state,
        (relations, learning, queue, fanout),
    )
    accepted = jax.tree.map(
        lambda new, old: jnp.where(candidate.successful, new, old), proposed, state
    )
    zero = jnp.asarray(0, dtype=jnp.int32)
    status = jnp.where(candidate.successful, zero, int(NeuralStatus.INVALID_INPUT))
    return NeuralNetworkResult(
        accepted,
        NeuralNetworkEvidence(
            status, candidate.successful, jnp.asarray(False), zero, zero, zero, queue.size
        ),
    )


def checkpoint_neural_network(
    runtime: PreparedNeuralNetwork, state: NeuralNetworkState, /
) -> NeuralNetworkCheckpoint:
    return NeuralNetworkCheckpoint(
        state, runtime.runtime_id, array_tree_fingerprint(state)
    )


def restore_neural_network(
    runtime: PreparedNeuralNetwork, checkpoint: NeuralNetworkCheckpoint, /
) -> NeuralNetworkState:
    if (
        checkpoint.runtime_id != runtime.runtime_id
        or checkpoint.content_id != array_tree_fingerprint(checkpoint.state)
    ):
        raise ValueError("Neural checkpoint runtime or content identity mismatch.")
    return checkpoint.state


__all__ = [
    "NeuralCellPlan",
    "NeuralChannelCoupling",
    "NeuralIonCoupling",
    "NeuralNetworkCheckpoint",
    "NeuralNetworkEvidence",
    "NeuralNetworkInputs",
    "NeuralNetworkPlan",
    "NeuralNetworkResult",
    "NeuralNetworkRunResult",
    "NeuralNetworkState",
    "NeuralRecording",
    "NeuralSpikeRecording",
    "NeuralStatus",
    "PreparedNeuralNetwork",
    "SpikeSource",
    "apply_neural_relation_event",
    "checkpoint_neural_network",
    "initialize_neural_network",
    "neural_voltage",
    "prepare_neural_network",
    "restore_neural_network",
    "run_neural_network",
    "step_neural_network",
    "zero_neural_inputs",
]
