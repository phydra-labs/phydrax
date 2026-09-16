#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Declared dark bound-state spectra and radiative detailed balance."""

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from ..solver._dark_sector_epoch_runtime import DarkSectorEpochPlan
from ._species import ParticleSpeciesTable


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class DarkBoundStateLevel(StrictModule, NonTrainableState):
    pdg_id: int = eqx.field(static=True)
    constituent_pdg_ids: tuple[int, int] = eqx.field(static=True)
    rest_energy: float = eqx.field(static=True)
    charge: float = eqx.field(static=True)
    degeneracy: int = eqx.field(static=True)
    radial_quantum_number: int = eqx.field(static=True)
    orbital_angular_momentum: int = eqx.field(static=True)
    spin_twice: int = eqx.field(static=True)
    level_label: str = eqx.field(static=True)
    level_id: str = eqx.field(static=True)

    def __init__(
        self,
        pdg_id: int,
        constituent_pdg_ids: tuple[int, int],
        /,
        *,
        rest_energy: float,
        charge: float,
        degeneracy: int,
        radial_quantum_number: int,
        orbital_angular_momentum: int,
        spin_twice: int,
        level_label: str,
    ):
        constituents = tuple(int(value) for value in constituent_pdg_ids)
        energy = float(rest_energy)
        charge_ = float(charge)
        quantum = tuple(
            int(value)
            for value in (
                degeneracy,
                radial_quantum_number,
                orbital_angular_momentum,
                spin_twice,
            )
        )
        label = _identifier(level_label, "Level label")
        if (
            len(constituents) != 2
            or not math.isfinite(energy)
            or energy <= 0.0
            or not math.isfinite(charge_)
            or quantum[0] < 1
            or quantum[1] < 1
            or quantum[2] < 0
            or quantum[3] < 0
        ):
            raise ValueError("Bound-state level parameters are invalid.")
        self.pdg_id = int(pdg_id)
        self.constituent_pdg_ids = constituents
        self.rest_energy = energy
        self.charge = charge_
        (
            self.degeneracy,
            self.radial_quantum_number,
            self.orbital_angular_momentum,
            self.spin_twice,
        ) = quantum
        self.level_label = label
        self.level_id = canonical_fingerprint(
            {
                "kind": "dark-bound-state-level",
                "pdg_id": self.pdg_id,
                "constituents": list(constituents),
                "rest_energy": energy,
                "charge": charge_,
                "degeneracy": quantum[0],
                "n": quantum[1],
                "l": quantum[2],
                "spin_twice": quantum[3],
                "label": label,
            }
        )


class DarkBoundStateSpectrum(StrictModule, NonTrainableState):
    runtime_plan: DarkSectorEpochPlan
    species: ParticleSpeciesTable
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    frame_realization_id: str = eqx.field(static=True)
    levels: tuple[DarkBoundStateLevel, ...]
    model_id: str = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    spectrum_source_id: str = eqx.field(static=True)
    production_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    support_scope: str = eqx.field(static=True)
    refusal_modes: tuple[str, ...] = eqx.field(static=True)
    differentiation_mode: str = eqx.field(static=True)
    spectrum_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime_plan: DarkSectorEpochPlan,
        species: ParticleSpeciesTable,
        units: RelativisticUnitContract,
        frame: LocalRelativisticFramePlan,
        levels: Sequence[DarkBoundStateLevel],
        /,
        *,
        model_id: str,
        model_revision_id: str,
        spectrum_source_id: str,
        production_evidence_ids: Sequence[str],
    ):
        if not isinstance(runtime_plan, DarkSectorEpochPlan):
            raise TypeError("runtime_plan must be DarkSectorEpochPlan.")
        if not isinstance(species, ParticleSpeciesTable):
            raise TypeError("species must be ParticleSpeciesTable.")
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be RelativisticUnitContract.")
        if species.energy_unit.unit_id != units.energy_unit.unit_id:
            raise ValueError(
                "species and bound-state spectrum must share the exact energy unit."
            )
        if (
            not isinstance(frame, LocalRelativisticFramePlan)
            or frame.units.contract_id != units.contract_id
        ):
            raise ValueError("frame must use the exact relativistic unit contract.")
        levels_ = tuple(levels)
        if not levels_ or any(
            not isinstance(value, DarkBoundStateLevel) for value in levels_
        ):
            raise TypeError("levels must contain DarkBoundStateLevel values.")
        if len({value.pdg_id for value in levels_}) != len(levels_):
            raise ValueError("Bound-state PDG identities must be unique in a spectrum.")
        model = _identifier(model_id, "Model ID")
        revision = _identifier(model_revision_id, "Model revision ID")
        source = _identifier(spectrum_source_id, "Spectrum source ID")
        evidence = tuple(
            _identifier(value, "Production evidence ID")
            for value in production_evidence_ids
        )
        if not evidence or len(set(evidence)) != len(evidence):
            raise ValueError(
                "Production evidence identities must be non-empty and unique."
            )
        ids = np.asarray(species.pdg_ids)
        masses = np.asarray(species.rest_energies)
        charges = np.asarray(species.charges)
        active = np.asarray(species.active)
        by_id = {
            int(identifier): (float(mass), float(charge))
            for identifier, mass, charge, present in zip(
                ids, masses, charges, active, strict=True
            )
            if present
        }
        for level in levels_:
            if any(value not in by_id for value in level.constituent_pdg_ids):
                raise ValueError(
                    "Every bound-state constituent must exist in the species table."
                )
            threshold = sum(by_id[value][0] for value in level.constituent_pdg_ids)
            charge_sum = sum(by_id[value][1] for value in level.constituent_pdg_ids)
            if level.rest_energy >= threshold:
                raise ValueError(
                    "A declared bound level must lie below its constituent threshold."
                )
            if not math.isclose(level.charge, charge_sum, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Bound-state and constituent charges must agree.")
        self.runtime_plan = runtime_plan
        self.species = species
        self.units = units
        self.frame = frame
        self.frame_realization_id = frame.realization_id()
        self.levels = levels_
        self.model_id = model
        self.model_revision_id = revision
        self.spectrum_source_id = source
        self.production_evidence_ids = evidence
        self.support_scope = "declared-two-constituent-radiative-bound-states"
        self.refusal_modes = ("unbound-level", "undeclared-constituent")
        self.differentiation_mode = "analytic-detailed-balance"
        self.spectrum_id = canonical_fingerprint(
            {
                "kind": "dark-bound-state-spectrum",
                "runtime_plan": runtime_plan.plan_id,
                "species": species.table_id,
                "units": units.contract_id,
                "frame": frame.frame_id,
                "frame_realization": self.frame_realization_id,
                "levels": [value.level_id for value in levels_],
                "model": model,
                "revision": revision,
                "source": source,
                "production_evidence": list(evidence),
                "support_scope": self.support_scope,
                "refusal_modes": list(self.refusal_modes),
                "differentiation": self.differentiation_mode,
            }
        )

    def level(self, pdg_id: int, /) -> DarkBoundStateLevel:
        matches = tuple(value for value in self.levels if value.pdg_id == int(pdg_id))
        if len(matches) != 1:
            raise KeyError(f"Unknown bound-state PDG identity {int(pdg_id)}.")
        return matches[0]


class RadiativeCapturePlan(StrictModule, NonTrainableState):
    spectrum: DarkBoundStateSpectrum
    bound_level: DarkBoundStateLevel
    capture_coefficient: float = eqx.field(static=True)
    emitted_radiation_degeneracy: int = eqx.field(static=True)
    constituent_degeneracies: tuple[int, int] = eqx.field(static=True)
    multipole_order: int = eqx.field(static=True)
    cross_section_unit_id: str = eqx.field(static=True)
    coefficient_source_id: str = eqx.field(static=True)
    differentiation_mode: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        spectrum: DarkBoundStateSpectrum,
        bound_pdg_id: int,
        /,
        *,
        capture_coefficient: float,
        emitted_radiation_degeneracy: int,
        constituent_degeneracies: tuple[int, int],
        multipole_order: int,
        cross_section_unit_id: str,
        coefficient_source_id: str,
        differentiation_mode: str = "analytic",
    ):
        if not isinstance(spectrum, DarkBoundStateSpectrum):
            raise TypeError("spectrum must be DarkBoundStateSpectrum.")
        level = spectrum.level(bound_pdg_id)
        coefficient = float(capture_coefficient)
        radiation_degeneracy = int(emitted_radiation_degeneracy)
        constituent_degeneracies_ = tuple(
            int(value) for value in constituent_degeneracies
        )
        multipole = int(multipole_order)
        if (
            not math.isfinite(coefficient)
            or coefficient <= 0.0
            or radiation_degeneracy < 1
            or len(constituent_degeneracies_) != 2
            or any(value < 1 for value in constituent_degeneracies_)
            or multipole < 1
        ):
            raise ValueError("Radiative capture parameters are invalid.")
        self.spectrum = spectrum
        self.bound_level = level
        self.capture_coefficient = coefficient
        self.emitted_radiation_degeneracy = radiation_degeneracy
        self.constituent_degeneracies = constituent_degeneracies_
        self.multipole_order = multipole
        self.cross_section_unit_id = _identifier(
            cross_section_unit_id, "Cross-section unit ID"
        )
        self.coefficient_source_id = _identifier(
            coefficient_source_id, "Coefficient source ID"
        )
        self.differentiation_mode = _identifier(
            differentiation_mode, "Differentiation mode"
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dark-radiative-capture-plan",
                "spectrum": spectrum.spectrum_id,
                "level": level.level_id,
                "capture_coefficient": coefficient,
                "radiation_degeneracy": radiation_degeneracy,
                "constituent_degeneracies": list(constituent_degeneracies_),
                "multipole_order": multipole,
                "cross_section_unit": self.cross_section_unit_id,
                "coefficient_source": self.coefficient_source_id,
                "differentiation": self.differentiation_mode,
            }
        )


class BoundStateReactionBalance(StrictModule, NonTrainableState):
    relative_momentum: Array
    photon_energy: Array
    capture_cross_section: Array
    photo_dissociation_cross_section: Array
    detailed_balance_factor: Array
    detailed_balance_residual: Array
    kinematically_open: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class ThermalBoundStateBalance(StrictModule, NonTrainableState):
    temperature: Array
    capture_rate_coefficient: Array
    photo_dissociation_rate: Array
    equilibrium_ratio: Array
    detailed_balance_residual: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


def _constituent_rest_energies(plan: RadiativeCapturePlan):
    identifiers = np.asarray(plan.spectrum.species.pdg_ids)
    energies = np.asarray(plan.spectrum.species.rest_energies)
    active = np.asarray(plan.spectrum.species.active)
    by_id = {
        int(identifier): float(energy)
        for identifier, energy, present in zip(identifiers, energies, active, strict=True)
        if present
    }
    return tuple(by_id[value] for value in plan.bound_level.constituent_pdg_ids)


def evaluate_radiative_capture_balance(
    plan: RadiativeCapturePlan,
    relative_momentum: ArrayLike,
    /,
) -> BoundStateReactionBalance:
    """Evaluate capture and its Milne-related inverse at identical kinematics."""

    if not isinstance(plan, RadiativeCapturePlan):
        raise TypeError("plan must be RadiativeCapturePlan.")
    momentum = jnp.asarray(relative_momentum)
    mass_a, mass_b = _constituent_rest_energies(plan)
    energy_a = jnp.sqrt(mass_a * mass_a + momentum * momentum)
    energy_b = jnp.sqrt(mass_b * mass_b + momentum * momentum)
    photon_energy = energy_a + energy_b - plan.bound_level.rest_energy
    relative_velocity = momentum / energy_a + momentum / energy_b
    power = 2 * plan.multipole_order + 1
    capture = (
        plan.capture_coefficient
        * photon_energy**power
        / jnp.maximum(relative_velocity, 1e-30)
    )
    degeneracy_factor = (
        plan.constituent_degeneracies[0]
        * plan.constituent_degeneracies[1]
        / (plan.bound_level.degeneracy * plan.emitted_radiation_degeneracy)
    )
    balance_factor = (
        degeneracy_factor
        * momentum
        * momentum
        / jnp.maximum(photon_energy * photon_energy, 1e-30)
    )
    dissociation = balance_factor * capture
    residual = dissociation - balance_factor * capture
    open_ = (momentum > 0.0) & (photon_energy > 0.0)
    finite = (
        jnp.isfinite(momentum)
        & jnp.isfinite(capture)
        & jnp.isfinite(dissociation)
        & open_
    )
    return BoundStateReactionBalance(
        momentum,
        photon_energy,
        jnp.where(open_, capture, jnp.nan),
        jnp.where(open_, dissociation, jnp.nan),
        balance_factor,
        residual,
        open_,
        finite,
        plan.plan_id,
    )


def evaluate_thermal_bound_state_balance(
    plan: RadiativeCapturePlan,
    temperature: ArrayLike,
    /,
) -> ThermalBoundStateBalance:
    """Nonrelativistic Maxwell-Boltzmann forward/reverse equilibrium relation."""

    if not isinstance(plan, RadiativeCapturePlan):
        raise TypeError("plan must be RadiativeCapturePlan.")
    temperature_ = jnp.asarray(temperature)
    mass_a, mass_b = _constituent_rest_energies(plan)
    reduced_mass = mass_a * mass_b / (mass_a + mass_b)
    binding_energy = mass_a + mass_b - plan.bound_level.rest_energy
    degeneracy_ratio = (
        plan.constituent_degeneracies[0]
        * plan.constituent_degeneracies[1]
        / plan.bound_level.degeneracy
    )
    equilibrium_ratio = (
        degeneracy_ratio
        * (reduced_mass * temperature_ / (2.0 * jnp.pi)) ** 1.5
        * jnp.exp(-binding_energy / temperature_)
    )
    thermal_momentum = jnp.sqrt(2.0 * reduced_mass * temperature_)
    point = evaluate_radiative_capture_balance(plan, thermal_momentum)
    relative_velocity = thermal_momentum / mass_a + thermal_momentum / mass_b
    capture_rate = point.capture_cross_section * relative_velocity
    reverse_rate = equilibrium_ratio * capture_rate
    residual = reverse_rate - equilibrium_ratio * capture_rate
    finite = (
        (temperature_ > 0.0)
        & jnp.isfinite(temperature_)
        & jnp.isfinite(capture_rate)
        & jnp.isfinite(reverse_rate)
    )
    return ThermalBoundStateBalance(
        temperature_,
        capture_rate,
        reverse_rate,
        equilibrium_ratio,
        residual,
        finite,
        plan.plan_id,
    )


__all__ = [
    "BoundStateReactionBalance",
    "DarkBoundStateLevel",
    "DarkBoundStateSpectrum",
    "RadiativeCapturePlan",
    "ThermalBoundStateBalance",
    "evaluate_radiative_capture_balance",
    "evaluate_thermal_bound_state_balance",
]
