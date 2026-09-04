#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stimuli, exact clamps, recording, checkpointing, and replay."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._cable import (
    CableState,
    CableStepInputs,
    CableStepResult,
    PreparedCableSolver,
    step_cable,
)
from ._units import ELECTROPHYSIOLOGY_UNITS


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _finite(value: float, name: str, /) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool.")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _interval(start_ms: float, stop_ms: float, /) -> tuple[float, float]:
    start = _finite(start_ms, "start_ms")
    stop = _finite(stop_ms, "stop_ms")
    if start < 0.0 or stop <= start:
        raise ValueError("A stimulus interval requires 0 <= start_ms < stop_ms.")
    return start, stop


class RecordingStatus(IntEnum):
    """Recording transition status."""

    SUCCESS = 0
    CAPACITY_EXCEEDED = 1
    NONFINITE = 2
    REJECTED_CABLE_STEP = 3


class CurrentClamp(StrictModule, NonTrainableState):
    """Rectangular inward-positive current injection in nA."""

    clamp_id: str = eqx.field(static=True)
    compartment_id: str = eqx.field(static=True)
    amplitude_nA: float = eqx.field(static=True)
    start_ms: float = eqx.field(static=True)
    stop_ms: float = eqx.field(static=True)
    stimulus_id: str = eqx.field(static=True)

    def __init__(
        self,
        clamp_id: str,
        compartment_id: str,
        amplitude_nA: float,
        start_ms: float,
        stop_ms: float,
        /,
    ):
        identifier = _identifier(clamp_id, "clamp_id")
        compartment = _identifier(compartment_id, "compartment_id")
        amplitude = _finite(amplitude_nA, "amplitude_nA")
        start, stop = _interval(start_ms, stop_ms)
        self.clamp_id = identifier
        self.compartment_id = compartment
        self.amplitude_nA = amplitude
        self.start_ms = start
        self.stop_ms = stop
        self.stimulus_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-current-clamp-v1",
                "clamp_id": identifier,
                "compartment_id": compartment,
                "amplitude_nA": amplitude,
                "start_ms": start,
                "stop_ms": stop,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )


class VoltageClamp(StrictModule, NonTrainableState):
    """Rectangular exact Dirichlet voltage command in mV."""

    clamp_id: str = eqx.field(static=True)
    compartment_id: str = eqx.field(static=True)
    target_mV: float = eqx.field(static=True)
    start_ms: float = eqx.field(static=True)
    stop_ms: float = eqx.field(static=True)
    stimulus_id: str = eqx.field(static=True)

    def __init__(
        self,
        clamp_id: str,
        compartment_id: str,
        target_mV: float,
        start_ms: float,
        stop_ms: float,
        /,
    ):
        identifier = _identifier(clamp_id, "clamp_id")
        compartment = _identifier(compartment_id, "compartment_id")
        target = _finite(target_mV, "target_mV")
        start, stop = _interval(start_ms, stop_ms)
        self.clamp_id = identifier
        self.compartment_id = compartment
        self.target_mV = target
        self.start_ms = start
        self.stop_ms = stop
        self.stimulus_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-voltage-clamp-v1",
                "clamp_id": identifier,
                "compartment_id": compartment,
                "target_mV": target,
                "start_ms": start,
                "stop_ms": stop,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )


class RecordingPlan(StrictModule, NonTrainableState):
    """Fixed-capacity voltage recording plan resolved by stable IDs."""

    compartment_ids: tuple[str, ...] = eqx.field(static=True)
    sample_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, compartment_ids: Sequence[str], sample_capacity: int, /):
        identifiers = tuple(compartment_ids)
        if not identifiers:
            raise ValueError("Recording requires at least one compartment identifier.")
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError(
                "Recording compartment identifiers must be non-empty strings."
            )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Recording compartment identifiers must be unique.")
        if isinstance(sample_capacity, bool) or not isinstance(sample_capacity, int):
            raise TypeError("sample_capacity must be an integer.")
        if sample_capacity <= 0:
            raise ValueError("sample_capacity must be positive.")
        self.compartment_ids = identifiers
        self.sample_capacity = sample_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-recording-v1",
                "compartment_ids": list(identifiers),
                "sample_capacity": sample_capacity,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )


class ElectrophysiologyProtocol(StrictModule, NonTrainableState):
    """Host plan for current clamps, voltage clamps, and recordings."""

    current_clamps: tuple[CurrentClamp, ...]
    voltage_clamps: tuple[VoltageClamp, ...]
    recording: RecordingPlan
    protocol_id: str = eqx.field(static=True)

    def __init__(
        self,
        recording: RecordingPlan,
        /,
        *,
        current_clamps: Sequence[CurrentClamp] = (),
        voltage_clamps: Sequence[VoltageClamp] = (),
    ):
        if not isinstance(recording, RecordingPlan):
            raise TypeError("recording must be a RecordingPlan.")
        currents = tuple(current_clamps)
        voltages = tuple(voltage_clamps)
        if any(not isinstance(value, CurrentClamp) for value in currents):
            raise TypeError("current_clamps must contain only CurrentClamp values.")
        if any(not isinstance(value, VoltageClamp) for value in voltages):
            raise TypeError("voltage_clamps must contain only VoltageClamp values.")
        identities = tuple(value.clamp_id for value in currents + voltages)
        if len(set(identities)) != len(identities):
            raise ValueError("Clamp identifiers must be unique within a protocol.")
        for index, left in enumerate(voltages):
            for right in voltages[index + 1 :]:
                overlap = left.start_ms < right.stop_ms and right.start_ms < left.stop_ms
                if left.compartment_id == right.compartment_id and overlap:
                    raise ValueError("Voltage clamps on one compartment cannot overlap.")
        self.current_clamps = currents
        self.voltage_clamps = voltages
        self.recording = recording
        self.protocol_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-protocol-v1",
                "current_clamps": [value.stimulus_id for value in currents],
                "voltage_clamps": [value.stimulus_id for value in voltages],
                "recording": recording.plan_id,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )

    def prepare(self, cable: PreparedCableSolver, /) -> PreparedElectrophysiologyProtocol:
        """Resolve all stable compartment identifiers into fixed device arrays."""
        return prepare_electrophysiology_protocol(self, cable)


class PreparedElectrophysiologyProtocol(StrictModule, NonTrainableState):
    """Fixed-shape device protocol bound to one prepared cable runtime."""

    plan: ElectrophysiologyProtocol
    cable: PreparedCableSolver
    current_indices: Array
    current_amplitudes_nA: Array
    current_start_ms: Array
    current_stop_ms: Array
    voltage_indices: Array
    voltage_targets_mV: Array
    voltage_start_ms: Array
    voltage_stop_ms: Array
    recording_indices: Array
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ElectrophysiologyProtocol,
        cable: PreparedCableSolver,
        current_indices: Array,
        current_amplitudes_nA: Array,
        current_start_ms: Array,
        current_stop_ms: Array,
        voltage_indices: Array,
        voltage_targets_mV: Array,
        voltage_start_ms: Array,
        voltage_stop_ms: Array,
        recording_indices: Array,
        /,
    ):
        self.plan = plan
        self.cable = cable
        self.current_indices = current_indices
        self.current_amplitudes_nA = current_amplitudes_nA
        self.current_start_ms = current_start_ms
        self.current_stop_ms = current_stop_ms
        self.voltage_indices = voltage_indices
        self.voltage_targets_mV = voltage_targets_mV
        self.voltage_start_ms = voltage_start_ms
        self.voltage_stop_ms = voltage_stop_ms
        self.recording_indices = recording_indices
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-electrophysiology-protocol-v1",
                "protocol": plan.protocol_id,
                "cable": cable.runtime_id,
            }
        )


class RecordingState(StrictModule):
    """Fixed-capacity append-only recording state."""

    time_ms: Array
    voltage_mV: Array
    valid: Array
    count: Array
    overflowed: Array


class ExperimentState(StrictModule):
    """Cable and recording state sufficient for exact deterministic replay."""

    cable: CableState
    recording: RecordingState


class ExperimentStepResult(StrictModule):
    """One protocol transition with cable and recording evidence."""

    state: ExperimentState
    cable: CableStepResult
    recording_status: Array


class ExperimentRunResult(StrictModule):
    """Final state and fixed-length trajectory evidence."""

    state: ExperimentState
    voltage_mV: Array
    cable_status: Array
    recording_status: Array


class ExperimentCheckpoint(StrictModule, NonTrainableState):
    """Content-addressed host checkpoint for exact continuation."""

    state: ExperimentState
    protocol_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(self, state: ExperimentState, protocol_id: str, checkpoint_id: str, /):
        self.state = state
        self.protocol_id = protocol_id
        self.checkpoint_id = checkpoint_id


def prepare_electrophysiology_protocol(
    plan: ElectrophysiologyProtocol, cable: PreparedCableSolver, /
) -> PreparedElectrophysiologyProtocol:
    """Resolve protocol identifiers and reject morphology mismatches."""
    if not isinstance(plan, ElectrophysiologyProtocol):
        raise TypeError("plan must be an ElectrophysiologyProtocol.")
    if not isinstance(cable, PreparedCableSolver):
        raise TypeError("cable must be a PreparedCableSolver.")
    morphology = cable.morphology.plan
    for identifier in (
        tuple(value.compartment_id for value in plan.current_clamps)
        + tuple(value.compartment_id for value in plan.voltage_clamps)
        + plan.recording.compartment_ids
    ):
        morphology.compartment_index(identifier)
    dtype = cable.morphology.capacitance_nF.dtype
    return PreparedElectrophysiologyProtocol(
        plan,
        cable,
        jnp.asarray(
            [
                morphology.compartment_index(value.compartment_id)
                for value in plan.current_clamps
            ],
            dtype=jnp.int32,
        ),
        jnp.asarray([value.amplitude_nA for value in plan.current_clamps], dtype=dtype),
        jnp.asarray([value.start_ms for value in plan.current_clamps], dtype=dtype),
        jnp.asarray([value.stop_ms for value in plan.current_clamps], dtype=dtype),
        jnp.asarray(
            [
                morphology.compartment_index(value.compartment_id)
                for value in plan.voltage_clamps
            ],
            dtype=jnp.int32,
        ),
        jnp.asarray([value.target_mV for value in plan.voltage_clamps], dtype=dtype),
        jnp.asarray([value.start_ms for value in plan.voltage_clamps], dtype=dtype),
        jnp.asarray([value.stop_ms for value in plan.voltage_clamps], dtype=dtype),
        jnp.asarray(
            [
                morphology.compartment_index(value)
                for value in plan.recording.compartment_ids
            ],
            dtype=jnp.int32,
        ),
    )


def initialize_recording(runtime: PreparedElectrophysiologyProtocol, /) -> RecordingState:
    """Allocate the protocol's fixed recording capacity."""
    capacity = runtime.plan.recording.sample_capacity
    channels = len(runtime.plan.recording.compartment_ids)
    dtype = runtime.cable.morphology.capacitance_nF.dtype
    return RecordingState(
        jnp.zeros((capacity,), dtype=dtype),
        jnp.zeros((capacity, channels), dtype=dtype),
        jnp.zeros((capacity,), dtype=bool),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
    )


def initialize_experiment(
    runtime: PreparedElectrophysiologyProtocol, cable_state: CableState, /
) -> ExperimentState:
    """Combine a shape-compatible cable state with empty recording storage."""
    if cable_state.voltage_mV.shape != (runtime.cable.morphology.plan.compartment_count,):
        raise ValueError("cable_state does not match the prepared protocol morphology.")
    return ExperimentState(cable_state, initialize_recording(runtime))


def protocol_inputs(
    runtime: PreparedElectrophysiologyProtocol, time_ms: Array, /
) -> CableStepInputs:
    """Evaluate all rectangular stimuli with outward/inward sign conventions."""
    count = runtime.cable.morphology.plan.compartment_count
    dtype = runtime.cable.morphology.capacitance_nF.dtype
    currents_active = (time_ms >= runtime.current_start_ms) & (
        time_ms < runtime.current_stop_ms
    )
    currents = (
        jnp.zeros((count,), dtype=dtype)
        .at[runtime.current_indices]
        .add(runtime.current_amplitudes_nA * currents_active)
    )
    voltage_active = (time_ms >= runtime.voltage_start_ms) & (
        time_ms < runtime.voltage_stop_ms
    )
    active_count = (
        jnp.zeros((count,), dtype=jnp.int32)
        .at[runtime.voltage_indices]
        .add(voltage_active.astype(jnp.int32))
    )
    mask = active_count > 0
    targets = (
        jnp.zeros((count,), dtype=dtype)
        .at[runtime.voltage_indices]
        .add(runtime.voltage_targets_mV * voltage_active)
    )
    zeros = jnp.zeros((count,), dtype=dtype)
    return CableStepInputs(currents, zeros, zeros, mask, targets)


def _record(
    runtime: PreparedElectrophysiologyProtocol,
    recording: RecordingState,
    cable: CableState,
    accepted_step: Array,
    /,
) -> tuple[RecordingState, Array]:
    capacity = runtime.plan.recording.sample_capacity
    room = recording.count < capacity
    index = jnp.minimum(recording.count, capacity - 1)
    sampled = cable.voltage_mV[runtime.recording_indices]
    finite = jnp.isfinite(cable.time_ms) & jnp.all(jnp.isfinite(sampled))
    accept = accepted_step & room & finite
    time = recording.time_ms.at[index].set(
        jnp.where(accept, cable.time_ms, recording.time_ms[index])
    )
    voltage = recording.voltage_mV.at[index].set(
        jnp.where(accept, sampled, recording.voltage_mV[index])
    )
    valid = recording.valid.at[index].set(jnp.where(accept, True, recording.valid[index]))
    status = jnp.where(
        ~accepted_step,
        int(RecordingStatus.REJECTED_CABLE_STEP),
        jnp.where(
            ~finite,
            int(RecordingStatus.NONFINITE),
            jnp.where(
                room,
                int(RecordingStatus.SUCCESS),
                int(RecordingStatus.CAPACITY_EXCEEDED),
            ),
        ),
    ).astype(jnp.int32)
    return RecordingState(
        time,
        voltage,
        valid,
        recording.count + accept.astype(jnp.int32),
        recording.overflowed | (accepted_step & ~room),
    ), status


def step_experiment(
    runtime: PreparedElectrophysiologyProtocol, state: ExperimentState, /
) -> ExperimentStepResult:
    """Evaluate, solve, commit, and record one deterministic transition."""
    inputs = protocol_inputs(runtime, state.cable.time_ms)
    cable_result = step_cable(runtime.cable, state.cable, inputs)
    recording, recording_status = _record(
        runtime,
        state.recording,
        cable_result.state,
        cable_result.evidence.successful,
    )
    return ExperimentStepResult(
        ExperimentState(cable_result.state, recording), cable_result, recording_status
    )


def run_experiment(
    runtime: PreparedElectrophysiologyProtocol,
    initial_state: ExperimentState,
    steps: int,
    /,
) -> ExperimentRunResult:
    """Run a fixed-length compiled scan without changing any state shape."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer.")
    if steps < 0:
        raise ValueError("steps must be nonnegative.")

    def advance(state, _):
        result = step_experiment(runtime, state)
        return result.state, (
            result.state.cable.voltage_mV,
            result.cable.evidence.status,
            result.recording_status,
        )

    state, trajectory = jax.lax.scan(advance, initial_state, xs=None, length=steps)
    return ExperimentRunResult(state, trajectory[0], trajectory[1], trajectory[2])


def _checkpoint_identity(
    runtime: PreparedElectrophysiologyProtocol, state: ExperimentState, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "electrophysiology-checkpoint-v1",
            "protocol": runtime.runtime_id,
            "state": array_tree_fingerprint(state),
            "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
        }
    )


def checkpoint_experiment(
    runtime: PreparedElectrophysiologyProtocol, state: ExperimentState, /
) -> ExperimentCheckpoint:
    """Create a content-addressed host checkpoint."""
    return ExperimentCheckpoint(
        state, runtime.runtime_id, _checkpoint_identity(runtime, state)
    )


def restore_experiment(
    runtime: PreparedElectrophysiologyProtocol, checkpoint: ExperimentCheckpoint, /
) -> ExperimentState:
    """Validate checkpoint provenance and content before restoring it."""
    if checkpoint.protocol_id != runtime.runtime_id:
        raise ValueError("Checkpoint protocol identity does not match this runtime.")
    if checkpoint.checkpoint_id != _checkpoint_identity(runtime, checkpoint.state):
        raise ValueError("Checkpoint content identity mismatch.")
    return checkpoint.state


def replay_experiment(
    runtime: PreparedElectrophysiologyProtocol,
    checkpoint: ExperimentCheckpoint,
    steps: int,
    /,
) -> ExperimentRunResult:
    """Continue an exact checkpoint through the same deterministic scan."""
    return run_experiment(runtime, restore_experiment(runtime, checkpoint), steps)


__all__ = [
    "CurrentClamp",
    "ElectrophysiologyProtocol",
    "ExperimentCheckpoint",
    "ExperimentRunResult",
    "ExperimentState",
    "ExperimentStepResult",
    "PreparedElectrophysiologyProtocol",
    "RecordingPlan",
    "RecordingState",
    "RecordingStatus",
    "VoltageClamp",
    "checkpoint_experiment",
    "initialize_experiment",
    "initialize_recording",
    "prepare_electrophysiology_protocol",
    "protocol_inputs",
    "replay_experiment",
    "restore_experiment",
    "run_experiment",
    "step_experiment",
]
