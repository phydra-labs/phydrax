#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import HermitianSpectrum


class CombLegSpec(StrictModule):
    system_dimension: int = eqx.field(static=True)
    memory_dimension: int = eqx.field(static=True)
    slot_count: int = eqx.field(static=True)
    vectorization: str = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        system_dimension: int,
        memory_dimension: int,
        slot_count: int,
        /,
        *,
        vectorization: str = "column",
        spec_id: str = "causal-process-v1",
    ):
        if min(system_dimension, memory_dimension, slot_count) < 1:
            raise ValueError("Process dimensions and slot count must be positive.")
        if vectorization != "column":
            raise ValueError("Only column vectorization is currently supported.")
        self.system_dimension = int(system_dimension)
        self.memory_dimension = int(memory_dimension)
        self.slot_count = int(slot_count)
        self.vectorization = vectorization
        self.spec_id = str(spec_id)


class QuantumInstrument(StrictModule):
    kraus: Array
    outcome_active: Array
    kraus_active: Array
    completeness_residual: Array
    valid: Array
    instrument_id: str = eqx.field(static=True)

    def __init__(
        self,
        kraus: ArrayLike,
        outcome_active: ArrayLike,
        kraus_active: ArrayLike,
        /,
        *,
        instrument_id: str,
    ):
        values = jnp.asarray(kraus)
        outcomes = jnp.asarray(outcome_active, dtype=bool)
        active = jnp.asarray(kraus_active, dtype=bool)
        if values.ndim != 4 or values.shape[-2] != values.shape[-1]:
            raise ValueError("Instrument Kraus array requires (outcome,kraus,d,d).")
        if outcomes.shape != values.shape[:1] or active.shape != values.shape[:2]:
            raise ValueError("Instrument masks do not match Kraus capacities.")
        dimension = values.shape[-1]
        total = jnp.zeros((dimension, dimension), dtype=values.dtype)
        for outcome in range(values.shape[0]):
            for index in range(values.shape[1]):
                operator = values[outcome, index]
                total = total + jnp.where(
                    outcomes[outcome] & active[outcome, index],
                    jnp.conj(operator.T) @ operator,
                    0.0,
                )
        residual = jnp.max(jnp.abs(total - jnp.eye(dimension, dtype=values.dtype)))
        self.kraus = values
        self.outcome_active = outcomes
        self.kraus_active = active
        self.completeness_residual = residual
        self.valid = jnp.all(jnp.isfinite(values)) & (residual <= 1e-8)
        self.instrument_id = str(instrument_id)

    @property
    def dimension(self) -> int:
        return int(self.kraus.shape[-1])


class CausalProcessResult(StrictModule):
    final_system_state: Array
    probability: Array
    valid: Array
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        final_system_state: ArrayLike,
        probability: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        process_id: str,
    ):
        self.final_system_state = jnp.asarray(final_system_state)
        self.probability = jnp.asarray(probability)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.process_id = str(process_id)


class CausalProcessTensor(StrictModule):
    spec: CombLegSpec
    initial_state: Array
    channel_kraus: tuple[Array, ...]
    channel_completeness_residuals: Array
    valid: Array
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        spec: CombLegSpec,
        initial_state: ArrayLike,
        channel_kraus: Sequence[ArrayLike],
        /,
        *,
        process_id: str,
    ):
        state = jnp.asarray(initial_state)
        composite = spec.system_dimension * spec.memory_dimension
        if state.shape != (composite, composite):
            raise ValueError("Initial system-memory state shape is invalid.")
        channels = tuple(jnp.asarray(values) for values in channel_kraus)
        if len(channels) != spec.slot_count or any(
            values.ndim != 3 or values.shape[1:] != (composite, composite)
            for values in channels
        ):
            raise ValueError("One composite Kraus channel is required per process slot.")
        residuals = []
        for values in channels:
            total = sum(jnp.conj(operator.T) @ operator for operator in values)
            residuals.append(
                jnp.max(jnp.abs(total - jnp.eye(composite, dtype=values.dtype)))
            )
        trace_residual = jnp.abs(jnp.trace(state) - 1.0)
        hermiticity_residual = jnp.max(jnp.abs(state - jnp.conj(state.T)))
        spectrum = HermitianSpectrum(0.5 * state + 0.5 * jnp.conj(state.T))
        minimum = spectrum.minimum_eigenvalue
        self.spec = spec
        self.initial_state = state
        self.channel_kraus = channels
        self.channel_completeness_residuals = jnp.stack(residuals)
        self.valid = (
            spectrum.valid
            & jnp.all(jnp.isfinite(state))
            & jnp.all(self.channel_completeness_residuals <= 1e-8)
            & (trace_residual <= 1e-8)
            & (hermiticity_residual <= 1e-8)
            & (minimum >= -1e-8)
        )
        self.process_id = str(process_id)

    def contract(
        self,
        instruments: Sequence[QuantumInstrument],
        outcomes: Sequence[int],
        /,
    ) -> CausalProcessResult:
        operations = tuple(instruments)
        selected = tuple(int(value) for value in outcomes)
        if (
            len(operations) != self.spec.slot_count
            or len(selected) != self.spec.slot_count
        ):
            raise ValueError("One instrument outcome is required per process slot.")
        state = self.initial_state
        system = self.spec.system_dimension
        memory = self.spec.memory_dimension
        memory_identity = jnp.eye(memory, dtype=state.dtype)
        valid = self.valid
        for slot, (instrument, outcome) in enumerate(
            zip(operations, selected, strict=True)
        ):
            if (
                instrument.dimension != system
                or not 0 <= outcome < instrument.kraus.shape[0]
                or not bool(instrument.outcome_active[outcome])
            ):
                raise ValueError(
                    "Instrument outcome is incompatible or inactive for the process."
                )
            updated = jnp.zeros_like(state)
            for index in range(instrument.kraus.shape[1]):
                local = jnp.kron(instrument.kraus[outcome, index], memory_identity)
                updated = updated + jnp.where(
                    instrument.kraus_active[outcome, index],
                    local @ state @ jnp.conj(local.T),
                    0.0,
                )
            state = jnp.zeros_like(updated)
            for operator in self.channel_kraus[slot]:
                state = state + operator @ updated @ jnp.conj(operator.T)
            valid = valid & instrument.valid & jnp.all(jnp.isfinite(state))
        probability = jnp.real(jnp.trace(state))
        tensor = state.reshape((system, memory, system, memory))
        reduced = jnp.trace(tensor, axis1=1, axis2=3)
        normalized = jnp.where(probability > 0.0, reduced / probability, reduced)
        valid = valid & (probability >= 0.0)
        return CausalProcessResult(
            normalized, probability, valid, process_id=self.process_id
        )


__all__ = [
    "CausalProcessResult",
    "CausalProcessTensor",
    "CombLegSpec",
    "QuantumInstrument",
]
