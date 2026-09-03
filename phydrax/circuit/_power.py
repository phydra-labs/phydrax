#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._strict import StrictModule
from ._components import Capacitor, Inductor, Resistor
from ._dae import PreparedCircuitDAE
from ._elements import (
    AbstractCircuitEnergyLaw,
    CircuitElement,
    energy_law_for,
    IndependentCurrentSourceLaw,
    IndependentVoltageSourceLaw,
)
from ._mna import MNAResult, NodeId, PreparedMNA
from ._periodic import HarmonicBalanceResult


PhasorAmplitudeConvention: TypeAlias = Literal["rms"]
CircuitCurrentOrientation: TypeAlias = Literal["ports-into-circuit;elements-into-device"]
CircuitEnergyCurrentOrientation: TypeAlias = Literal[
    "ports-into-circuit;sources-and-elements-into-device"
]


class MNAPowerLedger(StrictModule):
    """Independent complex-power contributions for one solved MNA result."""

    port_real_power: Array
    port_reactive_power: Array
    source_real_power: Array
    source_reactive_power: Array
    element_real_power: Array
    element_reactive_power: Array
    real_power_residual: Array
    reactive_power_residual: Array
    finite: Array
    available: Array
    real_power_closed: Array
    reactive_power_closed: Array
    port_ids: tuple[str, ...] = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    element_ids: tuple[str, ...] = eqx.field(static=True)
    unsupported_element_ids: tuple[str, ...] = eqx.field(static=True)
    unavailable_reasons: tuple[str, ...] = eqx.field(static=True)
    phasor_amplitude_convention: PhasorAmplitudeConvention = eqx.field(static=True)
    current_orientation: CircuitCurrentOrientation = eqx.field(static=True)
    closure_tolerance: float = eqx.field(static=True)


class CircuitEnergyLedger(StrictModule):
    """Pointwise and interval energy accounting for one circuit trajectory."""

    times: Array
    element_stored_energy: Array
    element_stored_energy_rate: Array
    element_dissipated_power: Array
    port_power: Array
    source_power: Array
    balance_residual: Array
    stored_energy_change: Array
    element_dissipated_energy: Array
    port_energy: Array
    source_energy: Array
    interval_balance_defect: Array
    finite: Array
    passive_dissipation_valid: Array
    available: Array
    closed: Array
    element_ids: tuple[str, ...] = eqx.field(static=True)
    port_ids: tuple[str, ...] = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    unsupported_element_ids: tuple[str, ...] = eqx.field(static=True)
    unavailable_reasons: tuple[str, ...] = eqx.field(static=True)
    current_orientation: CircuitEnergyCurrentOrientation = eqx.field(static=True)
    closure_tolerance: float = eqx.field(static=True)

    @property
    def stored_energy(self) -> Array:
        return jnp.sum(self.element_stored_energy, axis=-1)

    @property
    def stored_energy_rate(self) -> Array:
        return jnp.sum(self.element_stored_energy_rate, axis=-1)

    @property
    def dissipated_power(self) -> Array:
        return jnp.sum(self.element_dissipated_power, axis=-1)

    @property
    def total_port_power(self) -> Array:
        return jnp.sum(self.port_power, axis=-1)

    @property
    def total_source_power(self) -> Array:
        return jnp.sum(self.source_power, axis=-1)


class CircuitPeriodicEnergyLedger(StrictModule):
    """Period-integrated energy evidence for one harmonic-balance waveform."""

    samples: CircuitEnergyLedger
    period: Array
    endpoint_energy_change: Array
    integrated_stored_energy_rate: Array
    endpoint_energy_defect: Array
    element_dissipated_energy: Array
    port_energy: Array
    source_energy: Array
    integrated_balance_residual: Array
    period_balance_defect: Array
    aliasing_tail: Array
    aliasing_tail_valid: Array
    harmonic_residual_norm: Array
    harmonic_relative_residual: Array
    harmonic_balance_successful: Array
    finite: Array
    available: Array
    closed: Array
    closure_tolerance: float = eqx.field(static=True)


def _closure_tolerance(value: float, /) -> float:
    tolerance = float(value)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("closure_tolerance must be finite and non-negative.")
    return tolerance


def _finite_contributions(*values: Array) -> Array:
    finite = jnp.asarray(True)
    for value in values:
        finite = finite & jnp.all(jnp.isfinite(value))
    return finite


def _power_scale(*contributions: Array) -> Array:
    scale = jnp.asarray(0.0)
    for contribution in contributions:
        scale = scale + jnp.sum(jnp.abs(contribution), axis=-2)
    return jnp.maximum(scale, 1.0)


def assess_mna_power_ledger(
    ledger: MNAPowerLedger,
    /,
    *,
    closure_tolerance: float | None = None,
) -> MNAPowerLedger:
    """Recompute MNA closure from the independently retained contributions."""
    if not isinstance(ledger, MNAPowerLedger):
        raise TypeError("ledger must be MNAPowerLedger.")
    tolerance = _closure_tolerance(
        ledger.closure_tolerance if closure_tolerance is None else closure_tolerance
    )
    real_residual = (
        jnp.sum(ledger.port_real_power, axis=-2)
        + jnp.sum(ledger.source_real_power, axis=-2)
        + jnp.sum(ledger.element_real_power, axis=-2)
    )
    reactive_residual = (
        jnp.sum(ledger.port_reactive_power, axis=-2)
        + jnp.sum(ledger.source_reactive_power, axis=-2)
        + jnp.sum(ledger.element_reactive_power, axis=-2)
    )
    available = jnp.asarray(not ledger.unavailable_reasons)
    real_residual = jnp.where(available, real_residual, jnp.nan)
    reactive_residual = jnp.where(available, reactive_residual, jnp.nan)
    finite = _finite_contributions(
        ledger.port_real_power,
        ledger.port_reactive_power,
        ledger.source_real_power,
        ledger.source_reactive_power,
        ledger.element_real_power,
        ledger.element_reactive_power,
    )
    real_scale = _power_scale(
        ledger.port_real_power,
        ledger.source_real_power,
        ledger.element_real_power,
    )
    reactive_scale = _power_scale(
        ledger.port_reactive_power,
        ledger.source_reactive_power,
        ledger.element_reactive_power,
    )
    return MNAPowerLedger(
        ledger.port_real_power,
        ledger.port_reactive_power,
        ledger.source_real_power,
        ledger.source_reactive_power,
        ledger.element_real_power,
        ledger.element_reactive_power,
        real_residual,
        reactive_residual,
        finite,
        available,
        available & finite & (jnp.abs(real_residual) <= tolerance * real_scale),
        available & finite & (jnp.abs(reactive_residual) <= tolerance * reactive_scale),
        ledger.port_ids,
        ledger.source_ids,
        ledger.element_ids,
        ledger.unsupported_element_ids,
        ledger.unavailable_reasons,
        ledger.phasor_amplitude_convention,
        ledger.current_orientation,
        tolerance,
    )


def _mna_node_values(
    prepared: PreparedMNA, result: MNAResult, nodes: tuple[NodeId, ...], /
) -> Array:
    zero = jnp.zeros(
        result.solution.shape[:-2] + (result.solution.shape[-1],),
        dtype=result.solution.dtype,
    )
    values = tuple(
        zero
        if node == prepared.circuit.ground
        else result.node_voltages[..., prepared.plan.node_ids.index(node), :]
        for node in nodes
    )
    return jnp.stack(values, axis=-2)


def evaluate_mna_power_ledger(
    prepared: PreparedMNA,
    result: MNAResult,
    /,
    *,
    closure_tolerance: float = 1e-9,
) -> MNAPowerLedger:
    """Evaluate RMS phasor power without reusing the wave-power identity."""
    if not isinstance(prepared, PreparedMNA):
        raise TypeError("prepared must be PreparedMNA.")
    if not isinstance(result, MNAResult):
        raise TypeError("result must be MNAResult.")
    if result.numeric_version.shape != prepared.numeric_version.shape or bool(
        jnp.any(result.numeric_version != prepared.numeric_version)
    ):
        raise ValueError("MNA result does not belong to this prepared numeric state.")
    if result.node_ids != prepared.plan.node_ids or result.port_ids != tuple(
        port.port_id for port in prepared.circuit.ports
    ):
        raise ValueError("MNA result coordinates do not match the prepared circuit.")
    tolerance = _closure_tolerance(closure_tolerance)
    port_complex = -result.port_voltages * jnp.conj(result.port_currents)
    element_complex: list[Array] = []
    element_ids: list[str] = []
    unsupported_ids: list[str] = []
    reasons: list[str] = []
    for instance, start in zip(
        prepared.circuit.instances,
        prepared.plan.instance_auxiliary_offsets,
        strict=True,
    ):
        component = instance.component
        frequency_law = (
            component.frequency_law
            if isinstance(component, CircuitElement)
            else component
        )
        if isinstance(component, CircuitElement) and isinstance(
            component.implicit_law,
            (IndependentCurrentSourceLaw, IndependentVoltageSourceLaw),
        ):
            unsupported_ids.append(instance.instance_id)
            reasons.append(
                f"{instance.instance_id}: independent sources have no explicit "
                "MNA phasor forcing law"
            )
            continue
        if not isinstance(frequency_law, (Resistor, Capacitor, Inductor)):
            unsupported_ids.append(instance.instance_id)
            reasons.append(
                f"{instance.instance_id}: {type(frequency_law).__name__} has no "
                "supported element power law"
            )
            continue
        voltages = _mna_node_values(prepared, result, instance.nodes)
        stop = start + frequency_law.auxiliary_count
        auxiliary = result.solution[..., start:stop, :]
        stamp = frequency_law.evaluate(prepared.angular_frequency)
        currents = contract("...ij,...jr->...ir", stamp.y, voltages) + contract(
            "...ij,...jr->...ir", stamp.b, auxiliary
        )
        element_complex.append(jnp.sum(voltages * jnp.conj(currents), axis=-2))
        element_ids.append(instance.instance_id)
    batch_rhs_shape = result.solution.shape[:-2] + (result.solution.shape[-1],)
    element = (
        jnp.stack(element_complex, axis=-2)
        if element_complex
        else jnp.zeros(batch_rhs_shape[:-1] + (0, batch_rhs_shape[-1]))
    )
    sources = jnp.zeros(batch_rhs_shape[:-1] + (0, batch_rhs_shape[-1]))
    provisional = MNAPowerLedger(
        jnp.real(port_complex),
        jnp.imag(port_complex),
        jnp.real(sources),
        jnp.imag(sources),
        jnp.real(element),
        jnp.imag(element),
        jnp.zeros(batch_rhs_shape),
        jnp.zeros(batch_rhs_shape),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.zeros(batch_rhs_shape, dtype=bool),
        jnp.zeros(batch_rhs_shape, dtype=bool),
        tuple(port.port_id for port in prepared.circuit.ports),
        (),
        tuple(element_ids),
        tuple(unsupported_ids),
        tuple(reasons),
        "rms",
        "ports-into-circuit;elements-into-device",
        tolerance,
    )
    return assess_mna_power_ledger(provisional)


def _terminal_values(
    nodes: tuple[NodeId, ...],
    ground: NodeId | None,
    node_ids: tuple[NodeId, ...],
    values: Array,
    /,
) -> Array:
    zero = jnp.asarray(0.0, dtype=values.dtype)
    return jnp.stack(
        tuple(zero if node == ground else values[node_ids.index(node)] for node in nodes)
    )


def _trapezoid(values: Array, times: Array, /) -> Array:
    if times.size < 2:
        return jnp.zeros(values.shape[1:], dtype=values.dtype)
    widths = jnp.diff(times).reshape((times.size - 1,) + (1,) * (values.ndim - 1))
    return jnp.sum(0.5 * (values[:-1] + values[1:]) * widths, axis=0)


def assess_circuit_energy_ledger(
    ledger: CircuitEnergyLedger,
    /,
    *,
    closure_tolerance: float | None = None,
) -> CircuitEnergyLedger:
    """Recompute transient closure and interval defect from retained terms."""
    if not isinstance(ledger, CircuitEnergyLedger):
        raise TypeError("ledger must be CircuitEnergyLedger.")
    tolerance = _closure_tolerance(
        ledger.closure_tolerance if closure_tolerance is None else closure_tolerance
    )
    stored = jnp.sum(ledger.element_stored_energy, axis=-1)
    stored_rate = jnp.sum(ledger.element_stored_energy_rate, axis=-1)
    dissipation = jnp.sum(ledger.element_dissipated_power, axis=-1)
    ports = jnp.sum(ledger.port_power, axis=-1)
    sources = jnp.sum(ledger.source_power, axis=-1)
    residual = stored_rate + dissipation + ports + sources
    available = jnp.asarray(not ledger.unavailable_reasons)
    residual = jnp.where(available, residual, jnp.nan)
    stored_change = stored[-1] - stored[0]
    dissipated_energy = _trapezoid(ledger.element_dissipated_power, ledger.times)
    port_energy = _trapezoid(ledger.port_power, ledger.times)
    source_energy = _trapezoid(ledger.source_power, ledger.times)
    interval_defect = (
        stored_change
        + jnp.sum(dissipated_energy)
        + jnp.sum(port_energy)
        + jnp.sum(source_energy)
    )
    interval_defect = jnp.where(available, interval_defect, jnp.nan)
    finite = _finite_contributions(
        ledger.times,
        ledger.element_stored_energy,
        ledger.element_stored_energy_rate,
        ledger.element_dissipated_power,
        ledger.port_power,
        ledger.source_power,
    )
    sample_scale = jnp.maximum(
        jnp.abs(stored_rate) + jnp.abs(dissipation) + jnp.abs(ports) + jnp.abs(sources),
        1.0,
    )
    energy_scale = jnp.maximum(
        jnp.abs(stored_change)
        + jnp.sum(jnp.abs(dissipated_energy))
        + jnp.sum(jnp.abs(port_energy))
        + jnp.sum(jnp.abs(source_energy)),
        1.0,
    )
    dissipation_scale = jnp.maximum(
        jnp.max(jnp.abs(ledger.element_dissipated_power), initial=0.0),
        1.0,
    )
    passive_dissipation_valid = jnp.all(
        ledger.element_dissipated_power >= -tolerance * dissipation_scale
    )
    closed = (
        available
        & finite
        & passive_dissipation_valid
        & jnp.all(jnp.abs(residual) <= tolerance * sample_scale)
        & (jnp.abs(interval_defect) <= tolerance * energy_scale)
    )
    return CircuitEnergyLedger(
        ledger.times,
        ledger.element_stored_energy,
        ledger.element_stored_energy_rate,
        ledger.element_dissipated_power,
        ledger.port_power,
        ledger.source_power,
        residual,
        stored_change,
        dissipated_energy,
        port_energy,
        source_energy,
        interval_defect,
        finite,
        passive_dissipation_valid,
        available,
        closed,
        ledger.element_ids,
        ledger.port_ids,
        ledger.source_ids,
        ledger.unsupported_element_ids,
        ledger.unavailable_reasons,
        ledger.current_orientation,
        tolerance,
    )


def evaluate_circuit_energy_ledger(
    prepared: PreparedCircuitDAE,
    times: ArrayLike,
    states: ArrayLike,
    state_rates: ArrayLike,
    /,
    *,
    args: Any = None,
    port_currents: ArrayLike | None = None,
    closure_tolerance: float = 1e-7,
) -> CircuitEnergyLedger:
    """Evaluate passive storage/dissipation and separately signed source power."""
    if not isinstance(prepared, PreparedCircuitDAE):
        raise TypeError("prepared must be PreparedCircuitDAE.")
    time_values = jnp.asarray(times, dtype=float)
    state_values = jnp.asarray(states)
    rate_values = jnp.asarray(state_rates)
    expected = (time_values.size, prepared.plan.layout.size)
    if time_values.ndim != 1 or time_values.size == 0:
        raise ValueError("times must be one nonempty vector.")
    if state_values.shape != expected or rate_values.shape != expected:
        raise ValueError(f"states and state_rates must have shape {expected}.")
    if jnp.iscomplexobj(state_values) or jnp.iscomplexobj(rate_values):
        raise TypeError("Circuit energy trajectories must be real time-domain values.")
    trajectory_dtype = jnp.result_type(state_values, rate_values, jnp.float64)
    state_values = state_values.astype(trajectory_dtype)
    rate_values = rate_values.astype(trajectory_dtype)
    if time_values.size > 1 and bool(jnp.any(jnp.diff(time_values) <= 0.0)):
        raise ValueError("times must be strictly increasing.")
    tolerance = _closure_tolerance(closure_tolerance)
    circuit = prepared.plan.circuit
    layout = prepared.plan.layout
    passive: list[tuple[int, AbstractCircuitEnergyLaw]] = []
    source_indices: list[int] = []
    unsupported_ids: list[str] = []
    reasons: list[str] = []
    for index, (instance, law) in enumerate(
        zip(circuit.instances, prepared.plan.laws, strict=True)
    ):
        if isinstance(law, (IndependentCurrentSourceLaw, IndependentVoltageSourceLaw)):
            source_indices.append(index)
            continue
        energy_law = energy_law_for(instance.component)
        if energy_law is None:
            unsupported_ids.append(instance.instance_id)
            reasons.append(
                f"{instance.instance_id}: {type(law).__name__} has no passive energy law"
            )
        else:
            passive.append((index, energy_law))
    if port_currents is None:
        current_values = jnp.full(
            (time_values.size, len(circuit.ports)), jnp.nan, dtype=state_values.dtype
        )
        reasons.append("port currents are required for terminal power accounting")
    else:
        current_values = jnp.asarray(port_currents)
        if current_values.shape != (time_values.size, len(circuit.ports)):
            raise ValueError(
                "port_currents must have shape (samples, circuit port count)."
            )
        if jnp.iscomplexobj(current_values):
            raise TypeError("Transient port currents must be real.")
    inputs = args["inputs"] if isinstance(args, dict) and "inputs" in args else None
    law_args = args["args"] if isinstance(args, dict) and "args" in args else args
    stored_samples: list[Array] = []
    stored_rate_samples: list[Array] = []
    dissipation_samples: list[Array] = []
    source_samples: list[Array] = []
    port_samples: list[Array] = []
    for time, state, state_rate, port_current in zip(
        time_values, state_values, rate_values, current_values, strict=True
    ):
        evaluations = []
        terminal_voltages = []
        terminal_voltage_rates = []
        for instance, law, (start, stop) in zip(
            circuit.instances,
            prepared.plan.laws,
            layout.auxiliary_ranges,
            strict=True,
        ):
            voltage = _terminal_values(
                instance.nodes, circuit.ground, layout.node_ids, state
            )
            voltage_rate = _terminal_values(
                instance.nodes, circuit.ground, layout.node_ids, state_rate
            )
            evaluation = law.evaluate(
                time,
                voltage,
                voltage_rate,
                state[start:stop],
                state_rate[start:stop],
                inputs,
                law_args,
            )
            terminal_voltages.append(voltage)
            terminal_voltage_rates.append(voltage_rate)
            evaluations.append(evaluation)
        sample_stored = []
        sample_stored_rate = []
        sample_dissipation = []
        for index, energy_law in passive:
            start, stop = layout.auxiliary_ranges[index]
            voltage = terminal_voltages[index]
            voltage_rate = terminal_voltage_rates[index]
            local_state = state[start:stop]
            local_state_rate = state_rate[start:stop]
            stored = jnp.asarray(
                energy_law.stored_energy(voltage, local_state, args=law_args)
            )
            dissipated = jnp.asarray(
                energy_law.dissipated_power(
                    voltage,
                    evaluations[index].terminal_currents,
                    local_state,
                    args=law_args,
                )
            )
            _, stored_rate = jax.jvp(
                lambda terminal, local: energy_law.stored_energy(
                    terminal, local, args=law_args
                ),
                (voltage, local_state),
                (voltage_rate, local_state_rate),
            )
            if stored.shape != () or dissipated.shape != () or stored_rate.shape != ():
                raise ValueError("Circuit energy laws must return scalar contributions.")
            sample_stored.append(jnp.real(stored))
            sample_stored_rate.append(jnp.real(stored_rate))
            sample_dissipation.append(jnp.real(dissipated))
        sample_sources = tuple(
            jnp.real(
                jnp.sum(
                    terminal_voltages[index]
                    * jnp.conj(evaluations[index].terminal_currents)
                )
            )
            for index in source_indices
        )
        sample_ports = []
        for port, current in zip(circuit.ports, port_current, strict=True):
            terminal_voltage = _terminal_values(
                (port.positive, port.negative),
                circuit.ground,
                layout.node_ids,
                state,
            )
            sample_ports.append(
                -jnp.real((terminal_voltage[0] - terminal_voltage[1]) * current)
            )
        stored_samples.append(
            jnp.stack(sample_stored)
            if sample_stored
            else jnp.zeros((0,), dtype=state_values.dtype)
        )
        stored_rate_samples.append(
            jnp.stack(sample_stored_rate)
            if sample_stored_rate
            else jnp.zeros((0,), dtype=state_values.dtype)
        )
        dissipation_samples.append(
            jnp.stack(sample_dissipation)
            if sample_dissipation
            else jnp.zeros((0,), dtype=state_values.dtype)
        )
        source_samples.append(
            jnp.stack(sample_sources)
            if sample_sources
            else jnp.zeros((0,), dtype=state_values.dtype)
        )
        port_samples.append(
            jnp.stack(sample_ports)
            if sample_ports
            else jnp.zeros((0,), dtype=state_values.dtype)
        )
    stored_values = jnp.stack(stored_samples)
    stored_rate_values = jnp.stack(stored_rate_samples)
    dissipation_values = jnp.stack(dissipation_samples)
    port_power = jnp.stack(port_samples)
    source_power = jnp.stack(source_samples)
    provisional = CircuitEnergyLedger(
        time_values,
        stored_values,
        stored_rate_values,
        dissipation_values,
        port_power,
        source_power,
        jnp.zeros((time_values.size,)),
        jnp.asarray(0.0),
        jnp.zeros((len(passive),)),
        jnp.zeros((len(circuit.ports),)),
        jnp.zeros((len(source_indices),)),
        jnp.asarray(0.0),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        tuple(circuit.instances[index].instance_id for index, _ in passive),
        tuple(port.port_id for port in circuit.ports),
        tuple(circuit.instances[index].instance_id for index in source_indices),
        tuple(unsupported_ids),
        tuple(reasons),
        "ports-into-circuit;sources-and-elements-into-device",
        tolerance,
    )
    return assess_circuit_energy_ledger(provisional)


def assess_circuit_periodic_energy_ledger(
    ledger: CircuitPeriodicEnergyLedger,
    /,
    *,
    closure_tolerance: float | None = None,
) -> CircuitPeriodicEnergyLedger:
    """Recompute periodic energy closure and endpoint-rate consistency."""
    if not isinstance(ledger, CircuitPeriodicEnergyLedger):
        raise TypeError("ledger must be CircuitPeriodicEnergyLedger.")
    tolerance = _closure_tolerance(
        ledger.closure_tolerance if closure_tolerance is None else closure_tolerance
    )
    samples = assess_circuit_energy_ledger(ledger.samples, closure_tolerance=tolerance)
    period = ledger.period
    integrated_stored_rate = period * jnp.mean(samples.stored_energy_rate)
    endpoint_change = jnp.asarray(0.0, dtype=integrated_stored_rate.dtype)
    endpoint_defect = integrated_stored_rate - endpoint_change
    dissipated_energy = period * jnp.mean(samples.element_dissipated_power, axis=0)
    port_energy = period * jnp.mean(samples.port_power, axis=0)
    source_energy = period * jnp.mean(samples.source_power, axis=0)
    period_defect = (
        endpoint_change
        + jnp.sum(dissipated_energy)
        + jnp.sum(port_energy)
        + jnp.sum(source_energy)
    )
    integrated_residual = period * jnp.mean(samples.balance_residual)
    available = samples.available
    endpoint_defect = jnp.where(available, endpoint_defect, jnp.nan)
    period_defect = jnp.where(available, period_defect, jnp.nan)
    integrated_residual = jnp.where(available, integrated_residual, jnp.nan)
    finite = samples.finite & _finite_contributions(
        period,
        ledger.aliasing_tail,
        integrated_stored_rate,
        ledger.harmonic_residual_norm,
        ledger.harmonic_relative_residual,
        dissipated_energy,
        port_energy,
        source_energy,
    )
    sample_scale = jnp.maximum(
        jnp.abs(samples.stored_energy_rate)
        + jnp.abs(samples.dissipated_power)
        + jnp.abs(samples.total_port_power)
        + jnp.abs(samples.total_source_power),
        1.0,
    )
    energy_scale = jnp.maximum(
        jnp.abs(endpoint_change)
        + jnp.abs(integrated_stored_rate)
        + jnp.sum(jnp.abs(dissipated_energy))
        + jnp.sum(jnp.abs(port_energy))
        + jnp.sum(jnp.abs(source_energy)),
        1.0,
    )
    closed = (
        available
        & finite
        & samples.passive_dissipation_valid
        & jnp.all(jnp.abs(samples.balance_residual) <= tolerance * sample_scale)
        & (jnp.abs(endpoint_defect) <= tolerance * energy_scale)
        & (jnp.abs(period_defect) <= tolerance * energy_scale)
        & ledger.aliasing_tail_valid
        & ledger.harmonic_balance_successful
    )
    return CircuitPeriodicEnergyLedger(
        samples,
        period,
        endpoint_change,
        integrated_stored_rate,
        endpoint_defect,
        dissipated_energy,
        port_energy,
        source_energy,
        integrated_residual,
        period_defect,
        ledger.aliasing_tail,
        ledger.aliasing_tail_valid,
        ledger.harmonic_residual_norm,
        ledger.harmonic_relative_residual,
        ledger.harmonic_balance_successful,
        finite,
        available,
        closed,
        tolerance,
    )


def evaluate_harmonic_balance_energy_ledger(
    prepared_dae: PreparedCircuitDAE,
    result: HarmonicBalanceResult,
    /,
    *,
    args: Any = None,
    port_currents: ArrayLike | None = None,
    closure_tolerance: float = 1e-7,
) -> CircuitPeriodicEnergyLedger:
    """Evaluate period-integrated energy on an existing harmonic-balance result."""
    if not isinstance(prepared_dae, PreparedCircuitDAE):
        raise TypeError("prepared_dae must be PreparedCircuitDAE.")
    if not isinstance(result, HarmonicBalanceResult):
        raise TypeError("result must be HarmonicBalanceResult.")
    if result.plan.circuit_dae_plan_id != prepared_dae.plan.plan_id:
        raise ValueError("Harmonic-balance result belongs to a different circuit DAE.")
    temporal = result.plan.temporal
    rates = temporal.derivative(result.waveform)
    samples = evaluate_circuit_energy_ledger(
        prepared_dae,
        temporal.times,
        result.waveform,
        rates,
        args=args,
        port_currents=port_currents,
        closure_tolerance=closure_tolerance,
    )
    provisional = CircuitPeriodicEnergyLedger(
        samples,
        temporal.period,
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.zeros((len(samples.element_ids),)),
        jnp.zeros((len(samples.port_ids),)),
        jnp.zeros((len(samples.source_ids),)),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        result.diagnostics.aliasing_tail,
        result.diagnostics.aliasing_tail_valid,
        result.diagnostics.residual_norm,
        result.diagnostics.relative_residual,
        result.nonlinear.successful,
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        _closure_tolerance(closure_tolerance),
    )
    return assess_circuit_periodic_energy_ledger(provisional)


__all__ = [
    "CircuitCurrentOrientation",
    "CircuitEnergyCurrentOrientation",
    "CircuitEnergyLedger",
    "CircuitPeriodicEnergyLedger",
    "MNAPowerLedger",
    "PhasorAmplitudeConvention",
    "assess_circuit_energy_ledger",
    "assess_circuit_periodic_energy_ledger",
    "assess_mna_power_ledger",
    "evaluate_circuit_energy_ledger",
    "evaluate_harmonic_balance_energy_ledger",
    "evaluate_mna_power_ledger",
]
