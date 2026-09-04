#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Composable membrane mechanisms and ordered fixed-shape programs."""

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from typing import Protocol, runtime_checkable, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import (
    conversion_factor as _unit_conversion_factor,
    derived_unit,
    MICROAMPERE_PER_SQUARE_CENTIMETER,
    MICROMETER,
    MICROSIEMENS,
    MILLISIEMENS_PER_SQUARE_CENTIMETER,
    NANOAMPERE,
)
from ._morphology import PreparedCellMorphology
from ._units import ELECTROPHYSIOLOGY_UNITS


_MAX_GATES = 3


def _conductance_density_area_to_microsiemens() -> float:
    density_area = derived_unit(
        "mS/cm2*um2",
        ((MILLISIEMENS_PER_SQUARE_CENTIMETER, 1), (MICROMETER, 2)),
    )
    return float(_unit_conversion_factor(density_area, MICROSIEMENS))


def _current_density_area_to_nanoamperes() -> float:
    density_area = derived_unit(
        "uA/cm2*um2",
        ((MICROAMPERE_PER_SQUARE_CENTIMETER, 1), (MICROMETER, 2)),
    )
    return float(_unit_conversion_factor(density_area, NANOAMPERE))


def _finite(value: float, name: str, /) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool.")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _positive(value: float, name: str, /, *, allow_zero: bool = False) -> float:
    resolved = _finite(value, name)
    if resolved < 0.0 if allow_zero else resolved <= 0.0:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}.")
    return resolved


def _mechanism_identifier(value: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("mechanism_id must be a non-empty string.")
    return value


class MechanismStatus(IntFlag):
    """Bitwise evidence emitted by membrane program evaluation."""

    SUCCESS = 0
    NONFINITE = 1
    INVALID_CONCENTRATION = 2
    NONLINEAR_ROUTED = 4


@runtime_checkable
class MembraneMechanism(Protocol):
    """Protocol implemented by mechanisms admitted to an ordered program.

    Currents use the outward-positive convention and must be represented as
    ``conductance_uS * voltage_mV + current_offset_nA``. A voltage-independent
    nonlinear current is returned separately as routing evidence, while also
    appearing exactly once in ``current_offset_nA``.
    """

    mechanism_id: str
    gate_count: int
    nonlinear: bool

    def initial_gates(self, voltage_mV: Array, /) -> Array: ...

    def affine_current(
        self,
        voltage_mV: Array,
        gates: Array,
        membrane_area_um2: Array,
        intracellular_mM: Array,
        extracellular_mM: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]: ...

    def update_gates(self, voltage_mV: Array, gates: Array, dt_ms: Array, /) -> Array: ...


class PassiveLeak(StrictModule, NonTrainableState):
    """Ohmic leak with conductance density in mS/cm²."""

    conductance_density_mS_cm2: Array
    reversal_mV: Array
    mechanism_id: str = eqx.field(static=True)
    conductance_density_area_to_uS: float = eqx.field(static=True)
    gate_count: int = eqx.field(static=True)
    nonlinear: bool = eqx.field(static=True)

    def __init__(
        self,
        conductance_density_mS_cm2: float,
        reversal_mV: float,
        /,
        *,
        mechanism_id: str = "passive-leak",
    ):
        conductance = _positive(
            conductance_density_mS_cm2,
            "conductance_density_mS_cm2",
            allow_zero=True,
        )
        reversal = _finite(reversal_mV, "reversal_mV")
        identifier = _mechanism_identifier(mechanism_id)
        self.conductance_density_mS_cm2 = jnp.asarray(conductance)
        self.reversal_mV = jnp.asarray(reversal)
        self.conductance_density_area_to_uS = _conductance_density_area_to_microsiemens()
        self.mechanism_id = canonical_fingerprint(
            {
                "kind": "passive-leak-v1",
                "name": identifier,
                "conductance_density_mS_cm2": conductance,
                "reversal_mV": reversal,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )
        self.gate_count = 0
        self.nonlinear = False

    def initial_gates(self, voltage_mV: Array, /) -> Array:
        return jnp.zeros(voltage_mV.shape + (_MAX_GATES,), dtype=voltage_mV.dtype)

    def affine_current(
        self,
        voltage_mV: Array,
        gates: Array,
        membrane_area_um2: Array,
        intracellular_mM: Array,
        extracellular_mM: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        del gates, intracellular_mM, extracellular_mM
        conductance = (
            self.conductance_density_mS_cm2
            * membrane_area_um2
            * self.conductance_density_area_to_uS
        )
        offset = -conductance * self.reversal_mV
        zeros = jnp.zeros_like(voltage_mV)
        return conductance, offset, zeros, jnp.zeros_like(voltage_mV, dtype=jnp.int32)

    def update_gates(self, voltage_mV: Array, gates: Array, dt_ms: Array, /) -> Array:
        del voltage_mV, dt_ms
        return gates


def exact_affine_gate_update(
    gate: Array,
    steady_state: Array,
    time_constant_ms: Array,
    dt_ms: Array,
    /,
) -> Array:
    """Integrate ``dx/dt = (x_inf - x) / tau`` exactly over one step."""
    decay = jnp.exp(-dt_ms / time_constant_ms)
    return steady_state + (gate - steady_state) * decay


def _vtrap(value: Array, /) -> Array:
    small = jnp.abs(value) < 1.0e-4
    denominator = jnp.where(small, jnp.ones_like(value), -jnp.expm1(-value))
    regular = value / denominator
    series = 1.0 + 0.5 * value + value * value / 12.0
    return jnp.where(small, series, regular)


def hodgkin_huxley_rates(voltage_mV: Array, /) -> tuple[Array, ...]:
    """Return alpha/beta rates per millisecond for classic squid HH gates."""
    voltage = jnp.asarray(voltage_mV)
    alpha_m = _vtrap((voltage + 40.0) / 10.0)
    beta_m = 4.0 * jnp.exp(-(voltage + 65.0) / 18.0)
    alpha_h = 0.07 * jnp.exp(-(voltage + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + jnp.exp(-(voltage + 35.0) / 10.0))
    alpha_n = 0.1 * _vtrap((voltage + 55.0) / 10.0)
    beta_n = 0.125 * jnp.exp(-(voltage + 65.0) / 80.0)
    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


def _steady_and_tau(alpha: Array, beta: Array, /) -> tuple[Array, Array]:
    rate = alpha + beta
    return alpha / rate, 1.0 / rate


class HodgkinHuxleyNaK(StrictModule, NonTrainableState):
    """Classic fast sodium and delayed-rectifier potassium currents."""

    sodium_conductance_density_mS_cm2: Array
    potassium_conductance_density_mS_cm2: Array
    sodium_reversal_mV: Array
    potassium_reversal_mV: Array
    conductance_density_area_to_uS: float = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)
    gate_count: int = eqx.field(static=True)
    nonlinear: bool = eqx.field(static=True)

    def __init__(
        self,
        sodium_conductance_density_mS_cm2: float = 120.0,
        potassium_conductance_density_mS_cm2: float = 36.0,
        sodium_reversal_mV: float = 50.0,
        potassium_reversal_mV: float = -77.0,
        /,
        *,
        mechanism_id: str = "hodgkin-huxley-na-k",
    ):
        sodium = _positive(
            sodium_conductance_density_mS_cm2,
            "sodium_conductance_density_mS_cm2",
            allow_zero=True,
        )
        potassium = _positive(
            potassium_conductance_density_mS_cm2,
            "potassium_conductance_density_mS_cm2",
            allow_zero=True,
        )
        ena = _finite(sodium_reversal_mV, "sodium_reversal_mV")
        ek = _finite(potassium_reversal_mV, "potassium_reversal_mV")
        identifier = _mechanism_identifier(mechanism_id)
        self.sodium_conductance_density_mS_cm2 = jnp.asarray(sodium)
        self.potassium_conductance_density_mS_cm2 = jnp.asarray(potassium)
        self.sodium_reversal_mV = jnp.asarray(ena)
        self.potassium_reversal_mV = jnp.asarray(ek)
        self.conductance_density_area_to_uS = _conductance_density_area_to_microsiemens()
        self.mechanism_id = canonical_fingerprint(
            {
                "kind": "hodgkin-huxley-na-k-v1",
                "name": identifier,
                "sodium_conductance_density_mS_cm2": sodium,
                "potassium_conductance_density_mS_cm2": potassium,
                "sodium_reversal_mV": ena,
                "potassium_reversal_mV": ek,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )
        self.gate_count = 3
        self.nonlinear = False

    def initial_gates(self, voltage_mV: Array, /) -> Array:
        rates = hodgkin_huxley_rates(voltage_mV)
        m_inf, _ = _steady_and_tau(rates[0], rates[1])
        h_inf, _ = _steady_and_tau(rates[2], rates[3])
        n_inf, _ = _steady_and_tau(rates[4], rates[5])
        return jnp.stack((m_inf, h_inf, n_inf), axis=-1)

    def affine_current(
        self,
        voltage_mV: Array,
        gates: Array,
        membrane_area_um2: Array,
        intracellular_mM: Array,
        extracellular_mM: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        del intracellular_mM, extracellular_mM
        m_gate, h_gate, n_gate = gates[..., 0], gates[..., 1], gates[..., 2]
        area_scale = membrane_area_um2 * self.conductance_density_area_to_uS
        sodium = self.sodium_conductance_density_mS_cm2 * area_scale * m_gate**3 * h_gate
        potassium = self.potassium_conductance_density_mS_cm2 * area_scale * n_gate**4
        conductance = sodium + potassium
        offset = (
            -sodium * self.sodium_reversal_mV - potassium * self.potassium_reversal_mV
        )
        zeros = jnp.zeros_like(voltage_mV)
        finite = jnp.isfinite(conductance) & jnp.isfinite(offset)
        status = jnp.where(
            finite,
            int(MechanismStatus.SUCCESS),
            int(MechanismStatus.NONFINITE),
        ).astype(jnp.int32)
        return conductance, offset, zeros, status

    def update_gates(self, voltage_mV: Array, gates: Array, dt_ms: Array, /) -> Array:
        rates = hodgkin_huxley_rates(voltage_mV)
        m_inf, m_tau = _steady_and_tau(rates[0], rates[1])
        h_inf, h_tau = _steady_and_tau(rates[2], rates[3])
        n_inf, n_tau = _steady_and_tau(rates[4], rates[5])
        return jnp.stack(
            (
                exact_affine_gate_update(gates[..., 0], m_inf, m_tau, dt_ms),
                exact_affine_gate_update(gates[..., 1], h_inf, h_tau, dt_ms),
                exact_affine_gate_update(gates[..., 2], n_inf, n_tau, dt_ms),
            ),
            axis=-1,
        )


class SodiumPotassiumPump(StrictModule, NonTrainableState):
    """Electrogenic 3Na⁺/2K⁺ pump routed as an ion-nonlinear affine offset."""

    maximum_current_density_uA_cm2: Array
    sodium_half_saturation_mM: Array
    potassium_half_saturation_mM: Array
    sodium_species: int = eqx.field(static=True)
    potassium_species: int = eqx.field(static=True)
    current_density_area_to_nA: float = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)
    gate_count: int = eqx.field(static=True)
    nonlinear: bool = eqx.field(static=True)

    def __init__(
        self,
        maximum_current_density_uA_cm2: float,
        sodium_half_saturation_mM: float,
        potassium_half_saturation_mM: float,
        /,
        *,
        sodium_species: int = 0,
        potassium_species: int = 1,
        mechanism_id: str = "sodium-potassium-pump",
    ):
        maximum = _positive(
            maximum_current_density_uA_cm2,
            "maximum_current_density_uA_cm2",
            allow_zero=True,
        )
        sodium_half = _positive(sodium_half_saturation_mM, "sodium_half_saturation_mM")
        potassium_half = _positive(
            potassium_half_saturation_mM, "potassium_half_saturation_mM"
        )
        if isinstance(sodium_species, bool) or not isinstance(sodium_species, int):
            raise TypeError("sodium_species must be an integer.")
        if isinstance(potassium_species, bool) or not isinstance(potassium_species, int):
            raise TypeError("potassium_species must be an integer.")
        if (
            sodium_species < 0
            or potassium_species < 0
            or sodium_species == potassium_species
        ):
            raise ValueError("Pump species indices must be distinct and nonnegative.")
        identifier = _mechanism_identifier(mechanism_id)
        self.maximum_current_density_uA_cm2 = jnp.asarray(maximum)
        self.sodium_half_saturation_mM = jnp.asarray(sodium_half)
        self.potassium_half_saturation_mM = jnp.asarray(potassium_half)
        self.sodium_species = sodium_species
        self.potassium_species = potassium_species
        self.current_density_area_to_nA = _current_density_area_to_nanoamperes()
        self.mechanism_id = canonical_fingerprint(
            {
                "kind": "sodium-potassium-pump-v1",
                "name": identifier,
                "maximum_current_density_uA_cm2": maximum,
                "sodium_half_saturation_mM": sodium_half,
                "potassium_half_saturation_mM": potassium_half,
                "sodium_species": sodium_species,
                "potassium_species": potassium_species,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )
        self.gate_count = 0
        self.nonlinear = True

    def initial_gates(self, voltage_mV: Array, /) -> Array:
        return jnp.zeros(voltage_mV.shape + (_MAX_GATES,), dtype=voltage_mV.dtype)

    def affine_current(
        self,
        voltage_mV: Array,
        gates: Array,
        membrane_area_um2: Array,
        intracellular_mM: Array,
        extracellular_mM: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        del gates
        species_count = intracellular_mM.shape[0]
        if (
            self.sodium_species >= species_count
            or self.potassium_species >= species_count
        ):
            raise ValueError(
                "Pump species indices exceed the prepared ion-state capacity."
            )
        sodium = intracellular_mM[self.sodium_species]
        potassium = extracellular_mM[self.potassium_species]
        valid = (
            (sodium > 0.0)
            & (potassium > 0.0)
            & jnp.isfinite(sodium)
            & jnp.isfinite(potassium)
        )
        sodium_factor = sodium**3 / (sodium**3 + self.sodium_half_saturation_mM**3)
        potassium_factor = potassium**2 / (
            potassium**2 + self.potassium_half_saturation_mM**2
        )
        current = (
            self.maximum_current_density_uA_cm2
            * membrane_area_um2
            * self.current_density_area_to_nA
            * sodium_factor
            * potassium_factor
        )
        conductance = jnp.zeros_like(voltage_mV)
        status = jnp.where(
            valid & jnp.isfinite(current),
            int(MechanismStatus.NONLINEAR_ROUTED),
            int(MechanismStatus.INVALID_CONCENTRATION) | int(MechanismStatus.NONFINITE),
        ).astype(jnp.int32)
        return conductance, current, current, status

    def update_gates(self, voltage_mV: Array, gates: Array, dt_ms: Array, /) -> Array:
        del voltage_mV, dt_ms
        return gates


class MembraneProgram(StrictModule, NonTrainableState):
    """Ordered immutable mechanism program with a fixed three-gate lane per entry."""

    mechanisms: tuple[MembraneMechanism, ...]
    program_id: str = eqx.field(static=True)
    has_nonlinear_mechanisms: bool = eqx.field(static=True)

    def __init__(self, mechanisms: Sequence[MembraneMechanism], /):
        values = tuple(mechanisms)
        if not values:
            raise ValueError("A membrane program requires at least one mechanism.")
        if any(not isinstance(value, MembraneMechanism) for value in values):
            raise TypeError("Every program entry must implement MembraneMechanism.")
        identifiers = tuple(value.mechanism_id for value in values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Mechanism identities must be unique within a program.")
        if any(value.gate_count < 0 or value.gate_count > _MAX_GATES for value in values):
            raise ValueError("A mechanism may use at most three fixed gate lanes.")
        self.mechanisms = values
        self.has_nonlinear_mechanisms = any(value.nonlinear for value in values)
        self.program_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-membrane-program-v1",
                "mechanisms": list(identifiers),
                "maximum_gates_per_mechanism": _MAX_GATES,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )


class MembraneProgramState(StrictModule):
    """Fixed-shape gate state ``[mechanism, compartment, gate_lane]``."""

    gates: Array


class MembraneEvaluation(StrictModule):
    """Affine outward-current coefficients and per-compartment route evidence."""

    conductance_uS: Array
    current_offset_nA: Array
    nonlinear_current_nA: Array
    status: Array
    finite: Array
    nonlinear_routed: Array


def initialize_membrane_program(
    program: MembraneProgram,
    voltage_mV: Array,
    /,
) -> MembraneProgramState:
    """Initialize every gate lane at its voltage-dependent steady state."""
    voltage = jnp.asarray(voltage_mV)
    states = []
    for mechanism in program.mechanisms:
        gates = mechanism.initial_gates(voltage)
        if gates.shape != voltage.shape + (_MAX_GATES,):
            raise ValueError("Mechanism initial_gates returned an invalid fixed shape.")
        states.append(gates)
    return MembraneProgramState(jnp.stack(states, axis=0))


def evaluate_membrane_program(
    program: MembraneProgram,
    state: MembraneProgramState,
    morphology: PreparedCellMorphology,
    voltage_mV: Array,
    intracellular_mM: Array,
    extracellular_mM: Array,
    /,
) -> MembraneEvaluation:
    """Evaluate ordered mechanisms into one exact affine voltage current."""
    voltage = jnp.asarray(voltage_mV)
    expected = (len(program.mechanisms), voltage.shape[0], _MAX_GATES)
    if state.gates.shape != expected:
        raise ValueError(f"Program gates must have shape {expected}.")
    conductance = jnp.zeros_like(voltage)
    offset = jnp.zeros_like(voltage)
    nonlinear_current = jnp.zeros_like(voltage)
    status = jnp.zeros_like(voltage, dtype=jnp.int32)
    for index, mechanism in enumerate(program.mechanisms):
        contribution = mechanism.affine_current(
            voltage,
            state.gates[index],
            morphology.membrane_area_um2,
            intracellular_mM,
            extracellular_mM,
        )
        conductance = conductance + contribution[0]
        offset = offset + contribution[1]
        nonlinear_current = nonlinear_current + contribution[2]
        status = jnp.bitwise_or(status, contribution[3])
    finite = (
        jnp.isfinite(conductance) & jnp.isfinite(offset) & jnp.isfinite(nonlinear_current)
    )
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(MechanismStatus.NONFINITE)),
    )
    return MembraneEvaluation(
        conductance,
        offset,
        nonlinear_current,
        status,
        finite,
        (status & int(MechanismStatus.NONLINEAR_ROUTED)) != 0,
    )


def update_membrane_program(
    program: MembraneProgram,
    state: MembraneProgramState,
    voltage_mV: Array,
    dt_ms: Array,
    /,
) -> MembraneProgramState:
    """Apply exact affine gate updates in immutable program order."""
    updated = []
    for index, mechanism in enumerate(program.mechanisms):
        gates = mechanism.update_gates(voltage_mV, state.gates[index], dt_ms)
        if gates.shape != state.gates[index].shape:
            raise ValueError("Mechanism update_gates changed the fixed gate shape.")
        updated.append(gates)
    return MembraneProgramState(jnp.stack(updated, axis=0))


__all__ = [
    "HodgkinHuxleyNaK",
    "MechanismStatus",
    "MembraneEvaluation",
    "MembraneMechanism",
    "MembraneProgram",
    "MembraneProgramState",
    "PassiveLeak",
    "SodiumPotassiumPump",
    "evaluate_membrane_program",
    "exact_affine_gate_update",
    "hodgkin_huxley_rates",
    "initialize_membrane_program",
    "update_membrane_program",
]
