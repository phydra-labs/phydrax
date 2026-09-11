#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-molecule ideal-gas rigid-rotor harmonic-oscillator thermochemistry."""

from __future__ import annotations

from math import isfinite
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..atomistic import (
    AtomicStructure,
    AtomisticSystemPlan,
    AtomisticUnitSystem,
    single_system_energy_to_molar_factor,
)
from ..units import AMOUNT, derived_unit, ENERGY, TEMPERATURE, UnitDefinition
from ._optimization import _require_structure_matches_system
from ._units import ChemistryPhysicalConstants
from ._vibration import StationaryPointKind, VibrationalAnalysisResult


_BOLTZMANN_J_PER_K = 1.380649e-23
_PLANCK_J_S = 6.62607015e-34


class MolecularThermochemistryResult(StrictModule, NonTrainableState):
    component_internal_energies: Array
    component_entropies: Array
    zero_point_energy: Array
    internal_energy: Array
    enthalpy: Array
    entropy: Array
    gibbs_energy: Array
    temperature: Array
    pressure: Array
    successful: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    units: AtomisticUnitSystem
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_internal_energies: ArrayLike,
        component_entropies: ArrayLike,
        zero_point_energy: ArrayLike,
        internal_energy: ArrayLike,
        enthalpy: ArrayLike,
        entropy: ArrayLike,
        gibbs_energy: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        successful: ArrayLike,
        units: AtomisticUnitSystem,
        plan_id: str,
        /,
    ):
        energies = jnp.asarray(component_internal_energies)
        entropies = jnp.asarray(component_entropies, dtype=energies.dtype)
        if energies.shape != (4,) or entropies.shape != (4,):
            raise ValueError("Thermochemistry component arrays must have four entries.")
        scalars = tuple(
            jnp.asarray(value, dtype=energies.dtype).reshape(())
            for value in (
                zero_point_energy,
                internal_energy,
                enthalpy,
                entropy,
                gibbs_energy,
                temperature,
                pressure,
            )
        )
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        successful_ = jnp.asarray(successful, dtype=bool).reshape(())
        self.component_internal_energies = energies
        self.component_entropies = entropies
        (
            self.zero_point_energy,
            self.internal_energy,
            self.enthalpy,
            self.entropy,
            self.gibbs_energy,
            self.temperature,
            self.pressure,
        ) = scalars
        self.successful = successful_
        self.component_names = ("electronic", "translation", "rotation", "vibration")
        self.units = units
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "molecular-thermochemistry-result",
                "plan": self.plan_id,
                "units": units.unit_system_id,
                "successful": bool(successful_),
                "arrays": array_tree_fingerprint(
                    {
                        "component_internal_energies": np.asarray(energies),
                        "component_entropies": np.asarray(entropies),
                        "scalars": np.asarray(scalars),
                    }
                ),
            }
        )


class MolarThermochemistryResult(StrictModule, NonTrainableState):
    component_internal_energies: Array
    component_entropies: Array
    zero_point_energy: Array
    internal_energy: Array
    enthalpy: Array
    entropy: Array
    gibbs_energy: Array
    temperature: Array
    pressure: Array
    energy_unit: UnitDefinition
    entropy_unit: UnitDefinition
    source_result_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: MolecularThermochemistryResult,
        energy_unit: UnitDefinition,
        /,
    ):
        if not isinstance(source, MolecularThermochemistryResult):
            raise TypeError("source must be MolecularThermochemistryResult.")
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY / AMOUNT:
            raise ValueError("energy_unit must have the ENERGY / AMOUNT dimension.")
        factor = single_system_energy_to_molar_factor(
            source.units.scale.energy_unit,
            energy_unit,
            constant_set_id=source.units.constant_set_id,
        )
        entropy_unit_ = derived_unit(
            f"{energy_unit.symbol}/{source.units.temperature_unit.symbol}",
            ((energy_unit, 1), (source.units.temperature_unit, -1)),
        )
        if entropy_unit_.dimension != ENERGY / AMOUNT / TEMPERATURE:
            raise RuntimeError("Derived molar entropy unit has an invalid dimension.")
        self.component_internal_energies = factor * source.component_internal_energies
        self.component_entropies = factor * source.component_entropies
        self.zero_point_energy = factor * source.zero_point_energy
        self.internal_energy = factor * source.internal_energy
        self.enthalpy = factor * source.enthalpy
        self.entropy = factor * source.entropy
        self.gibbs_energy = factor * source.gibbs_energy
        self.temperature = source.temperature
        self.pressure = source.pressure
        self.energy_unit = energy_unit
        self.entropy_unit = entropy_unit_
        self.source_result_id = source.result_id
        self.result_id = canonical_fingerprint(
            {
                "kind": "molar-thermochemistry-result",
                "source": source.result_id,
                "energy_unit": energy_unit.unit_id,
                "entropy_unit": entropy_unit_.unit_id,
            }
        )


class HarmonicThermochemistryPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    temperature: float = eqx.field(static=True)
    pressure: float = eqx.field(static=True)
    symmetry_number: int = eqx.field(static=True)
    electronic_degeneracy: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        temperature: float,
        pressure: float,
        /,
        *,
        symmetry_number: int,
        electronic_degeneracy: int,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        temperature_ = float(temperature)
        pressure_ = float(pressure)
        if any(not isfinite(value) or value <= 0.0 for value in (temperature_, pressure_)):
            raise ValueError("Thermochemistry temperature and pressure must be positive finite.")
        if isinstance(symmetry_number, bool) or not isinstance(symmetry_number, Integral):
            raise TypeError("symmetry_number must be an integer.")
        if isinstance(electronic_degeneracy, bool) or not isinstance(
            electronic_degeneracy, Integral
        ):
            raise TypeError("electronic_degeneracy must be an integer.")
        symmetry = int(symmetry_number)
        degeneracy = int(electronic_degeneracy)
        if symmetry <= 0 or degeneracy <= 0:
            raise ValueError("Symmetry number and electronic degeneracy must be positive.")
        ChemistryPhysicalConstants(system.units)
        self.system = system
        self.temperature = temperature_
        self.pressure = pressure_
        self.symmetry_number = symmetry
        self.electronic_degeneracy = degeneracy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "harmonic-thermochemistry-plan",
                "system": system.system_id,
                "temperature": temperature_,
                "pressure": pressure_,
                "symmetry_number": symmetry,
                "electronic_degeneracy": degeneracy,
                "unit_system": system.units.unit_system_id,
            }
        )

    def evaluate(
        self,
        structure: AtomicStructure,
        vibration: VibrationalAnalysisResult,
        electronic_energy: ArrayLike,
        /,
    ) -> MolecularThermochemistryResult:
        if not isinstance(structure, AtomicStructure):
            raise TypeError("structure must be AtomicStructure.")
        if not isinstance(vibration, VibrationalAnalysisResult):
            raise TypeError("vibration must be VibrationalAnalysisResult.")
        _require_structure_matches_system(structure, self.system)
        if vibration.units.unit_system_id != self.system.units.unit_system_id:
            raise ValueError("Vibration and thermochemistry unit systems differ.")
        if not bool(vibration.successful) or vibration.stationary_point not in (
            StationaryPointKind.MINIMUM,
            StationaryPointKind.FIRST_ORDER_SADDLE,
        ):
            raise ValueError("Thermochemistry requires a qualified minimum or first-order saddle.")
        imaginary_count = int(np.count_nonzero(np.asarray(vibration.imaginary_mask)))
        if vibration.stationary_point is StationaryPointKind.MINIMUM and imaginary_count:
            raise ValueError("Minimum thermochemistry cannot contain imaginary modes.")
        if vibration.stationary_point is StationaryPointKind.FIRST_ORDER_SADDLE and imaginary_count != 1:
            raise ValueError("First-order-saddle thermochemistry requires exactly one imaginary mode.")
        energy = float(np.asarray(electronic_energy))
        if not np.isfinite(energy):
            raise ValueError("electronic_energy must be finite.")
        units = self.system.units
        temperature_si = self.temperature * float(units.temperature_unit.scale_to_reference)
        pressure_si = self.pressure * float(units.pressure_unit.scale_to_reference)
        energy_scale = float(units.scale.energy_unit.scale_to_reference)
        entropy_scale = energy_scale / float(units.temperature_unit.scale_to_reference)
        active = np.asarray(self.system.active_mask, dtype=bool)
        masses_si = (
            np.asarray(self.system.masses)[active]
            * float(units.mass_unit.scale_to_reference)
        )
        positions_si = (
            np.asarray(structure.positions)[active]
            * float(units.scale.length_unit.scale_to_reference)
        )
        total_mass = float(np.sum(masses_si))
        beta = 1.0 / (_BOLTZMANN_J_PER_K * temperature_si)
        volume_per_molecule = _BOLTZMANN_J_PER_K * temperature_si / pressure_si
        q_translation = (
            (2.0 * np.pi * total_mass * _BOLTZMANN_J_PER_K * temperature_si)
            / (_PLANCK_J_S**2)
        ) ** 1.5 * volume_per_molecule
        u_translation = 1.5 * _BOLTZMANN_J_PER_K * temperature_si
        s_translation = _BOLTZMANN_J_PER_K * (np.log(q_translation) + 2.5)
        center = np.sum(masses_si[:, None] * positions_si, axis=0) / total_mass
        relative = positions_si - center
        inertia = np.zeros((3, 3), dtype=float)
        for mass, position in zip(masses_si, relative, strict=True):
            radius_squared = float(position @ position)
            inertia += mass * (radius_squared * np.eye(3) - np.outer(position, position))
        moments = np.linalg.eigvalsh(inertia)
        if vibration.external_mode_count == 3:
            u_rotation = 0.0
            s_rotation = 0.0
        elif vibration.external_mode_count == 5:
            moment = float(np.mean(moments[-2:]))
            q_rotation = (
                8.0
                * np.pi**2
                * moment
                * _BOLTZMANN_J_PER_K
                * temperature_si
                / (self.symmetry_number * _PLANCK_J_S**2)
            )
            u_rotation = _BOLTZMANN_J_PER_K * temperature_si
            s_rotation = _BOLTZMANN_J_PER_K * (np.log(q_rotation) + 1.0)
        else:
            if np.any(moments <= 0.0):
                raise ValueError("Nonlinear rotor has a non-positive principal moment.")
            q_rotation = (
                np.sqrt(np.pi)
                / self.symmetry_number
                * (8.0 * np.pi**2 * _BOLTZMANN_J_PER_K * temperature_si / _PLANCK_J_S**2)
                ** 1.5
                * np.sqrt(float(np.prod(moments)))
            )
            u_rotation = 1.5 * _BOLTZMANN_J_PER_K * temperature_si
            s_rotation = _BOLTZMANN_J_PER_K * (np.log(q_rotation) + 1.5)
        angular_si = np.abs(np.asarray(vibration.angular_frequencies)) / float(
            units.time_unit.scale_to_reference
        )
        retained = ~np.asarray(vibration.imaginary_mask)
        angular_si = angular_si[retained]
        if np.any(angular_si <= 0.0):
            raise ValueError("Strict RRHO requires every retained vibrational mode to be positive.")
        quanta = (_PLANCK_J_S / (2.0 * np.pi)) * angular_si
        scaled = beta * quanta
        occupation = 1.0 / np.expm1(scaled)
        zpe_si = float(0.5 * np.sum(quanta))
        u_vibration = float(np.sum(0.5 * quanta + quanta * occupation))
        s_vibration = float(
            _BOLTZMANN_J_PER_K
            * np.sum(scaled * occupation - np.log1p(-np.exp(-scaled)))
        )
        s_electronic = _BOLTZMANN_J_PER_K * np.log(self.electronic_degeneracy)
        component_energy_si = np.asarray(
            (energy * energy_scale, u_translation, u_rotation, u_vibration)
        )
        component_entropy_si = np.asarray((s_electronic, s_translation, s_rotation, s_vibration))
        internal_si = float(np.sum(component_energy_si))
        enthalpy_si = internal_si + _BOLTZMANN_J_PER_K * temperature_si
        entropy_si = float(np.sum(component_entropy_si))
        gibbs_si = enthalpy_si - temperature_si * entropy_si
        values = np.asarray(
            (
                *component_energy_si,
                *component_entropy_si,
                zpe_si,
                internal_si,
                enthalpy_si,
                entropy_si,
                gibbs_si,
            )
        )
        successful = bool(np.all(np.isfinite(values)) and q_translation > 0.0)
        return MolecularThermochemistryResult(
            component_energy_si / energy_scale,
            component_entropy_si / entropy_scale,
            zpe_si / energy_scale,
            internal_si / energy_scale,
            enthalpy_si / energy_scale,
            entropy_si / entropy_scale,
            gibbs_si / energy_scale,
            self.temperature,
            self.pressure,
            successful,
            units,
            self.plan_id,
        )


def to_molar_thermochemistry(
    source: MolecularThermochemistryResult,
    energy_unit: UnitDefinition,
    /,
) -> MolarThermochemistryResult:
    return MolarThermochemistryResult(source, energy_unit)


__all__ = [
    "HarmonicThermochemistryPlan",
    "MolarThermochemistryResult",
    "MolecularThermochemistryResult",
    "to_molar_thermochemistry",
]
