#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Ion concentration, Nernst, and charge-conserving transition dynamics."""

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import (
    conversion_factor as _unit_conversion_factor,
    COULOMB,
    derived_unit,
    MILLIMOLAR,
    MILLISECOND,
    MOLE,
    NANOAMPERE,
    PICOLITER,
    VOLT,
)
from ._units import ELECTROPHYSIOLOGY_UNITS


FARADAY_C_PER_MOL = 96_485.33212
GAS_CONSTANT_J_PER_MOL_K = 8.314462618


class IonStatus(IntFlag):
    """Fail-closed bitwise concentration-transition status."""

    SUCCESS = 0
    NONFINITE = 1
    NONPOSITIVE_CONCENTRATION = 2
    CONSERVATION_FAILURE = 4
    CHARGE_FAILURE = 8
    INVALID_TIMESTEP = 16


def _positive(value: float, name: str, /) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool.")
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


class IonSpecies(StrictModule, NonTrainableState):
    """Stable ionic species with an integer nonzero valence."""

    species_id: str = eqx.field(static=True)
    valence: int = eqx.field(static=True)
    species_fingerprint: str = eqx.field(static=True)

    def __init__(self, species_id: str, valence: int, /):
        identifier = _identifier(species_id, "species_id")
        if isinstance(valence, bool) or not isinstance(valence, int):
            raise TypeError("valence must be an integer.")
        if valence == 0:
            raise ValueError("Ion valence must be nonzero.")
        self.species_id = identifier
        self.valence = valence
        self.species_fingerprint = canonical_fingerprint(
            {
                "kind": "electrophysiology-ion-species-v1",
                "species_id": identifier,
                "valence": valence,
            }
        )


class IonDynamicsPlan(StrictModule, NonTrainableState):
    """Fixed species, volume, temperature, and evidence-tolerance plan."""

    species: tuple[IonSpecies, ...]
    intracellular_volume_pL: tuple[float, ...] = eqx.field(static=True)
    extracellular_volume_pL: tuple[float, ...] = eqx.field(static=True)
    temperature_K: float = eqx.field(static=True)
    minimum_concentration_mM: float = eqx.field(static=True)
    conservation_tolerance_mol: float = eqx.field(static=True)
    charge_tolerance_C: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: Sequence[IonSpecies],
        intracellular_volume_pL: Sequence[float],
        extracellular_volume_pL: Sequence[float],
        /,
        *,
        temperature_K: float = 310.15,
        minimum_concentration_mM: float = 1.0e-12,
        conservation_tolerance_mol: float = 1.0e-18,
        charge_tolerance_C: float = 1.0e-15,
    ):
        species_values = tuple(species)
        if not species_values or any(
            not isinstance(value, IonSpecies) for value in species_values
        ):
            raise TypeError("species must contain one or more IonSpecies values.")
        identifiers = tuple(value.species_id for value in species_values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Ion species identifiers must be unique.")
        intracellular = tuple(
            _positive(value, "intracellular_volume_pL")
            for value in intracellular_volume_pL
        )
        extracellular = tuple(
            _positive(value, "extracellular_volume_pL")
            for value in extracellular_volume_pL
        )
        if not intracellular or len(intracellular) != len(extracellular):
            raise ValueError(
                "Intracellular and extracellular volume arrays must have equal nonzero length."
            )
        temperature = _positive(temperature_K, "temperature_K")
        minimum = _positive(minimum_concentration_mM, "minimum_concentration_mM")
        conservation = _positive(conservation_tolerance_mol, "conservation_tolerance_mol")
        charge = _positive(charge_tolerance_C, "charge_tolerance_C")
        self.species = species_values
        self.intracellular_volume_pL = intracellular
        self.extracellular_volume_pL = extracellular
        self.temperature_K = temperature
        self.minimum_concentration_mM = minimum
        self.conservation_tolerance_mol = conservation
        self.charge_tolerance_C = charge
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-ion-dynamics-v1",
                "species": [value.species_fingerprint for value in species_values],
                "intracellular_volume_pL": list(intracellular),
                "extracellular_volume_pL": list(extracellular),
                "temperature_K": temperature,
                "minimum_concentration_mM": minimum,
                "conservation_tolerance_mol": conservation,
                "charge_tolerance_C": charge,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )

    def prepare(self) -> PreparedIonDynamics:
        return prepare_ion_dynamics(self)


class PreparedIonDynamics(StrictModule, NonTrainableState):
    """Fixed-shape device volume and valence runtime."""

    plan: IonDynamicsPlan
    valence: Array
    intracellular_volume_pL: Array
    extracellular_volume_pL: Array
    thermal_voltage_to_mV: float = eqx.field(static=True)
    charge_per_nA_ms_C: float = eqx.field(static=True)
    amount_per_mM_pL_mol: float = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: IonDynamicsPlan,
        valence: Array,
        intracellular_volume_pL: Array,
        extracellular_volume_pL: Array,
        /,
        *,
        thermal_voltage_to_mV: float,
        charge_per_nA_ms_C: float,
        amount_per_mM_pL_mol: float,
    ):
        self.plan = plan
        self.valence = valence
        self.intracellular_volume_pL = intracellular_volume_pL
        self.extracellular_volume_pL = extracellular_volume_pL
        self.thermal_voltage_to_mV = thermal_voltage_to_mV
        self.charge_per_nA_ms_C = charge_per_nA_ms_C
        self.amount_per_mM_pL_mol = amount_per_mM_pL_mol
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-electrophysiology-ion-dynamics-v1", "plan": plan.plan_id}
        )


class IonConcentrationState(StrictModule):
    """Fixed-shape intra/extracellular concentrations in mM."""

    intracellular_mM: Array
    extracellular_mM: Array
    step_index: Array


class IonConcentrationEvidence(StrictModule):
    """Mole, electrical-charge, positivity, and finiteness evidence."""

    total_moles_before: Array
    total_moles_after: Array
    conservation_residual_mol: Array
    intracellular_charge_residual_C: Array
    minimum_concentration_mM: Array
    finite: Array
    status: Array
    successful: Array


class IonConcentrationCandidate(StrictModule):
    """Uncommitted concentration update and its conservation evidence."""

    proposed: IonConcentrationState
    evidence: IonConcentrationEvidence


def prepare_ion_dynamics(plan: IonDynamicsPlan, /) -> PreparedIonDynamics:
    if not isinstance(plan, IonDynamicsPlan):
        raise TypeError("plan must be an IonDynamicsPlan.")
    nA_ms = derived_unit("nA*ms", ((NANOAMPERE, 1), (MILLISECOND, 1)))
    mM_pL = derived_unit("mM*pL", ((MILLIMOLAR, 1), (PICOLITER, 1)))
    thermal_voltage_to_mV = float(
        _unit_conversion_factor(VOLT, ELECTROPHYSIOLOGY_UNITS.voltage)
    )
    charge_per_nA_ms_C = float(_unit_conversion_factor(nA_ms, COULOMB))
    amount_per_mM_pL_mol = float(_unit_conversion_factor(mM_pL, MOLE))
    dtype = jnp.asarray(0.0).dtype
    return PreparedIonDynamics(
        plan,
        jnp.asarray([value.valence for value in plan.species], dtype=dtype),
        jnp.asarray(plan.intracellular_volume_pL, dtype=dtype),
        jnp.asarray(plan.extracellular_volume_pL, dtype=dtype),
        thermal_voltage_to_mV=thermal_voltage_to_mV,
        charge_per_nA_ms_C=charge_per_nA_ms_C,
        amount_per_mM_pL_mol=amount_per_mM_pL_mol,
    )


def initialize_ion_concentrations(
    runtime: PreparedIonDynamics, intracellular_mM: Array, extracellular_mM: Array, /
) -> IonConcentrationState:
    """Validate and materialize a positive fixed-shape concentration state."""
    intracellular_input = jnp.asarray(intracellular_mM)
    extracellular_input = jnp.asarray(extracellular_mM)
    dtype = jnp.result_type(
        intracellular_input.dtype,
        extracellular_input.dtype,
        jnp.float32,
    )
    intracellular = intracellular_input.astype(dtype)
    extracellular = extracellular_input.astype(dtype)
    shape = (len(runtime.plan.species), len(runtime.plan.intracellular_volume_pL))
    if intracellular.shape != shape or extracellular.shape != shape:
        raise ValueError(f"Concentrations must have shape {shape}.")
    if not bool(jnp.all(jnp.isfinite(intracellular))) or not bool(
        jnp.all(jnp.isfinite(extracellular))
    ):
        raise ValueError("Initial concentrations must be finite.")
    if not bool(
        jnp.all(intracellular > runtime.plan.minimum_concentration_mM)
    ) or not bool(jnp.all(extracellular > runtime.plan.minimum_concentration_mM)):
        raise ValueError("Initial concentrations must exceed minimum_concentration_mM.")
    return IonConcentrationState(
        intracellular, extracellular, jnp.asarray(0, dtype=jnp.int32)
    )


def nernst_potential_mV(
    runtime: PreparedIonDynamics, state: IonConcentrationState, /
) -> Array:
    """Return each species/compartment Nernst potential in mV."""
    thermal_mV = (
        runtime.thermal_voltage_to_mV
        * GAS_CONSTANT_J_PER_MOL_K
        * runtime.plan.temperature_K
        / FARADAY_C_PER_MOL
    )
    return (
        thermal_mV
        / runtime.valence[:, None]
        * jnp.log(state.extracellular_mM / state.intracellular_mM)
    )


def evaluate_ion_concentration_transition(
    runtime: PreparedIonDynamics,
    state: IonConcentrationState,
    outward_current_nA: Array,
    dt_ms: Array,
    /,
) -> IonConcentrationCandidate:
    """Evaluate a closed two-volume ion transfer under outward-positive current."""
    dtype = jnp.result_type(state.intracellular_mM.dtype, jnp.float32)
    current = jnp.asarray(outward_current_nA, dtype=dtype)
    if current.shape != state.intracellular_mM.shape:
        raise ValueError("outward_current_nA must match the concentration state shape.")
    dt = jnp.asarray(dt_ms, dtype=dtype)
    if dt.shape != ():
        raise ValueError("dt_ms must be a scalar.")
    transfer_moles = (
        current
        * dt
        * runtime.charge_per_nA_ms_C
        / (runtime.valence[:, None] * FARADAY_C_PER_MOL)
    )
    intracellular_delta_mM = -transfer_moles / (
        runtime.intracellular_volume_pL[None, :] * runtime.amount_per_mM_pL_mol
    )
    extracellular_delta_mM = transfer_moles / (
        runtime.extracellular_volume_pL[None, :] * runtime.amount_per_mM_pL_mol
    )
    intracellular = state.intracellular_mM + intracellular_delta_mM
    extracellular = state.extracellular_mM + extracellular_delta_mM
    before = (
        state.intracellular_mM
        * runtime.intracellular_volume_pL[None, :]
        * runtime.amount_per_mM_pL_mol
        + state.extracellular_mM
        * runtime.extracellular_volume_pL[None, :]
        * runtime.amount_per_mM_pL_mol
    )
    after = (
        intracellular
        * runtime.intracellular_volume_pL[None, :]
        * runtime.amount_per_mM_pL_mol
        + extracellular
        * runtime.extracellular_volume_pL[None, :]
        * runtime.amount_per_mM_pL_mol
    )
    conservation_residual = after - before
    intracellular_charge_change = (
        intracellular_delta_mM
        * runtime.intracellular_volume_pL[None, :]
        * runtime.amount_per_mM_pL_mol
        * runtime.valence[:, None]
        * FARADAY_C_PER_MOL
    )
    charge_residual = (
        intracellular_charge_change + current * dt * runtime.charge_per_nA_ms_C
    )
    minimum = jnp.minimum(jnp.min(intracellular), jnp.min(extracellular))
    timestep_valid = jnp.isfinite(dt) & (dt > 0.0)
    finite = (
        jnp.all(jnp.isfinite(intracellular))
        & jnp.all(jnp.isfinite(extracellular))
        & jnp.all(jnp.isfinite(current))
        & jnp.isfinite(dt)
    )
    positive = minimum > runtime.plan.minimum_concentration_mM
    conservation_ok = (
        jnp.max(jnp.abs(conservation_residual)) <= runtime.plan.conservation_tolerance_mol
    )
    charge_ok = jnp.max(jnp.abs(charge_residual)) <= runtime.plan.charge_tolerance_C
    status = jnp.asarray(int(IonStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(finite, status, jnp.bitwise_or(status, int(IonStatus.NONFINITE)))
    status = jnp.where(
        positive, status, jnp.bitwise_or(status, int(IonStatus.NONPOSITIVE_CONCENTRATION))
    )
    status = jnp.where(
        conservation_ok,
        status,
        jnp.bitwise_or(status, int(IonStatus.CONSERVATION_FAILURE)),
    )
    status = jnp.where(
        charge_ok, status, jnp.bitwise_or(status, int(IonStatus.CHARGE_FAILURE))
    )
    status = jnp.where(
        timestep_valid,
        status,
        jnp.bitwise_or(status, int(IonStatus.INVALID_TIMESTEP)),
    )
    successful = finite & positive & conservation_ok & charge_ok & timestep_valid
    proposed = IonConcentrationState(intracellular, extracellular, state.step_index + 1)
    evidence = IonConcentrationEvidence(
        before,
        after,
        conservation_residual,
        charge_residual,
        minimum,
        finite,
        status,
        successful,
    )
    return IonConcentrationCandidate(proposed, evidence)


def commit_ion_concentration_transition(
    candidate: IonConcentrationCandidate, current: IonConcentrationState, /
) -> IonConcentrationState:
    """Commit a valid conservative transfer or preserve the current state."""
    return jax.tree.map(
        lambda proposed, prior: jnp.where(candidate.evidence.successful, proposed, prior),
        candidate.proposed,
        current,
    )


def sodium_potassium_pump_ion_currents(
    pump_current_nA: Array,
    sodium_species: int = 0,
    potassium_species: int = 1,
    species_count: int = 2,
    /,
) -> Array:
    """Route net pump current into 3 Na⁺ outward and 2 K⁺ inward currents."""
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (sodium_species, potassium_species, species_count)
    ):
        raise TypeError("Species indices and species_count must be integers.")
    if (
        species_count <= 0
        or sodium_species < 0
        or potassium_species < 0
        or sodium_species >= species_count
        or potassium_species >= species_count
        or sodium_species == potassium_species
    ):
        raise ValueError(
            "Pump species indices must be distinct and within species_count."
        )
    current = jnp.asarray(pump_current_nA)
    routed = jnp.zeros((species_count,) + current.shape, dtype=current.dtype)
    routed = routed.at[sodium_species].set(3.0 * current)
    return routed.at[potassium_species].set(-2.0 * current)


__all__ = [
    "FARADAY_C_PER_MOL",
    "GAS_CONSTANT_J_PER_MOL_K",
    "IonConcentrationCandidate",
    "IonConcentrationEvidence",
    "IonConcentrationState",
    "IonDynamicsPlan",
    "IonSpecies",
    "IonStatus",
    "PreparedIonDynamics",
    "commit_ion_concentration_transition",
    "evaluate_ion_concentration_transition",
    "initialize_ion_concentrations",
    "nernst_potential_mV",
    "prepare_ion_dynamics",
    "sodium_potassium_pump_ion_currents",
]
