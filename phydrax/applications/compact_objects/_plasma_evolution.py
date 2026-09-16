#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._plasma_closures import (
    TwoTemperatureElectronIonClosure,
    TwoTemperaturePlasmaState,
)


class ElectronHeatingEvaluation(StrictModule):
    electron_fraction: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class AbstractElectronHeatingPlan(StrictModule, NonTrainableState):
    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def evaluate(
        self,
        electron_temperature: ArrayLike,
        ion_temperature: ArrayLike,
        gas_pressure: ArrayLike,
        magnetic_pressure: ArrayLike,
        /,
    ) -> ElectronHeatingEvaluation:
        raise NotImplementedError


class ConstantElectronHeatingPlan(AbstractElectronHeatingPlan):
    fraction: float = eqx.field(static=True)

    def __init__(self, fraction: float, /) -> None:
        value = float(fraction)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Electron heating fraction must lie in [0, 1].")
        self.fraction = value
        self.plan_id = canonical_fingerprint(
            {"kind": "constant-electron-heating", "fraction": value}
        )

    def evaluate(
        self,
        electron_temperature: ArrayLike,
        ion_temperature: ArrayLike,
        gas_pressure: ArrayLike,
        magnetic_pressure: ArrayLike,
        /,
    ) -> ElectronHeatingEvaluation:
        electron, ion, gas, magnetic = jnp.broadcast_arrays(
            jnp.asarray(electron_temperature),
            jnp.asarray(ion_temperature),
            jnp.asarray(gas_pressure),
            jnp.asarray(magnetic_pressure),
        )
        fraction = jnp.full_like(electron, self.fraction)
        finite = (
            jnp.isfinite(electron)
            & jnp.isfinite(ion)
            & jnp.isfinite(gas)
            & jnp.isfinite(magnetic)
        )
        physical = (
            finite & (electron > 0.0) & (ion > 0.0) & (gas >= 0.0) & (magnetic >= 0.0)
        )
        return ElectronHeatingEvaluation(
            fraction, finite, physical, physical, physical, self.plan_id
        )


class TurbulentElectronHeatingPlan(AbstractElectronHeatingPlan):
    """Smooth magnetization/temperature-ratio turbulent heating fit."""

    minimum_fraction: float = eqx.field(static=True)
    maximum_fraction: float = eqx.field(static=True)
    beta_scale: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_fraction: float = 0.02,
        maximum_fraction: float = 0.95,
        beta_scale: float = 1.0,
    ) -> None:
        values = tuple(
            float(value) for value in (minimum_fraction, maximum_fraction, beta_scale)
        )
        if (
            any(not np.isfinite(value) for value in values)
            or not 0.0 <= values[0] <= values[1] <= 1.0
            or values[2] <= 0.0
        ):
            raise ValueError("Turbulent electron heating controls are invalid.")
        self.minimum_fraction, self.maximum_fraction, self.beta_scale = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "turbulent-electron-heating",
                "fraction_support": values[:2],
                "beta_scale": values[2],
            }
        )

    def evaluate(
        self,
        electron_temperature: ArrayLike,
        ion_temperature: ArrayLike,
        gas_pressure: ArrayLike,
        magnetic_pressure: ArrayLike,
        /,
    ) -> ElectronHeatingEvaluation:
        electron, ion, gas, magnetic = jnp.broadcast_arrays(
            jnp.asarray(electron_temperature),
            jnp.asarray(ion_temperature),
            jnp.asarray(gas_pressure),
            jnp.asarray(magnetic_pressure),
        )
        beta = gas / jnp.maximum(magnetic, jnp.finfo(gas.dtype).tiny)
        ratio = electron / jnp.maximum(ion, jnp.finfo(ion.dtype).tiny)
        weight = 1.0 / (1.0 + beta / self.beta_scale)
        relativistic_bias = jnp.sqrt(jnp.clip(ratio, 0.0, 1.0))
        fraction = self.minimum_fraction + (
            self.maximum_fraction - self.minimum_fraction
        ) * weight * (0.5 + 0.5 * relativistic_bias)
        finite = (
            jnp.isfinite(electron)
            & jnp.isfinite(ion)
            & jnp.isfinite(gas)
            & jnp.isfinite(magnetic)
            & jnp.isfinite(fraction)
        )
        physical = (
            finite & (electron > 0.0) & (ion > 0.0) & (gas >= 0.0) & (magnetic >= 0.0)
        )
        qualified = physical & (fraction >= 0.0) & (fraction <= 1.0)
        return ElectronHeatingEvaluation(
            fraction,
            finite,
            physical,
            qualified,
            qualified & (magnetic > 0.0),
            self.plan_id,
        )


class ReconnectionElectronHeatingPlan(AbstractElectronHeatingPlan):
    """Bounded reconnection heating increasing with magnetic pressure fraction."""

    low_fraction: float = eqx.field(static=True)
    high_fraction: float = eqx.field(static=True)
    transition_ratio: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        low_fraction: float = 0.1,
        high_fraction: float = 0.5,
        transition_ratio: float = 1.0,
    ) -> None:
        values = tuple(
            float(value) for value in (low_fraction, high_fraction, transition_ratio)
        )
        if (
            any(not np.isfinite(value) for value in values)
            or not 0.0 <= values[0] <= values[1] <= 1.0
            or values[2] <= 0.0
        ):
            raise ValueError("Reconnection heating controls are invalid.")
        self.low_fraction, self.high_fraction, self.transition_ratio = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reconnection-electron-heating",
                "fraction_support": values[:2],
                "transition_ratio": values[2],
            }
        )

    def evaluate(
        self,
        electron_temperature: ArrayLike,
        ion_temperature: ArrayLike,
        gas_pressure: ArrayLike,
        magnetic_pressure: ArrayLike,
        /,
    ) -> ElectronHeatingEvaluation:
        electron, ion, gas, magnetic = jnp.broadcast_arrays(
            jnp.asarray(electron_temperature),
            jnp.asarray(ion_temperature),
            jnp.asarray(gas_pressure),
            jnp.asarray(magnetic_pressure),
        )
        ratio = magnetic / jnp.maximum(gas, jnp.finfo(gas.dtype).tiny)
        weight = ratio / (ratio + self.transition_ratio)
        fraction = self.low_fraction + (self.high_fraction - self.low_fraction) * weight
        finite = (
            jnp.isfinite(electron)
            & jnp.isfinite(ion)
            & jnp.isfinite(gas)
            & jnp.isfinite(magnetic)
            & jnp.isfinite(fraction)
        )
        physical = (
            finite & (electron > 0.0) & (ion > 0.0) & (gas >= 0.0) & (magnetic >= 0.0)
        )
        qualified = physical & (fraction >= 0.0) & (fraction <= 1.0)
        return ElectronHeatingEvaluation(
            fraction,
            finite,
            physical,
            qualified,
            qualified & (gas > 0.0),
            self.plan_id,
        )


class RelativisticTwoTemperatureState(StrictModule):
    electron_internal_energy: Array
    ion_internal_energy: Array
    electron_temperature: Array
    ion_temperature: Array
    rest_mass_density: Array


class RelativisticTwoTemperatureLedger(StrictModule):
    electron_adiabatic_change: Array
    ion_adiabatic_change: Array
    dissipative_heating: Array
    electron_heating: Array
    ion_heating: Array
    radiation_exchange: Array
    coulomb_electron_exchange: Array
    coulomb_ion_exchange: Array
    total_energy_defect: Array
    minimum_species_energy: Array
    finite: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class RelativisticTwoTemperatureResult(StrictModule):
    candidate: RelativisticTwoTemperatureState
    state: RelativisticTwoTemperatureState
    heating: ElectronHeatingEvaluation
    ledger: RelativisticTwoTemperatureLedger
    accepted: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class RelativisticTwoTemperaturePlan(StrictModule, NonTrainableState):
    """Adiabatic species evolution, dissipation partition, and Coulomb exchange."""

    scale: RelativityScaleContract
    electron_mass_per_particle: float = eqx.field(static=True)
    ion_mass_per_particle: float = eqx.field(static=True)
    electron_adiabatic_index: float = eqx.field(static=True)
    ion_adiabatic_index: float = eqx.field(static=True)
    heating: AbstractElectronHeatingPlan
    coulomb: TwoTemperatureElectronIonClosure
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        heating: AbstractElectronHeatingPlan,
        coulomb: TwoTemperatureElectronIonClosure,
        /,
        *,
        electron_mass_per_particle: float,
        ion_mass_per_particle: float,
        electron_adiabatic_index: float = 4.0 / 3.0,
        ion_adiabatic_index: float = 5.0 / 3.0,
        energy_tolerance: float = 1.0e-9,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        if not isinstance(heating, AbstractElectronHeatingPlan):
            raise TypeError("heating must implement AbstractElectronHeatingPlan.")
        if not isinstance(coulomb, TwoTemperatureElectronIonClosure):
            raise TypeError("coulomb must be TwoTemperatureElectronIonClosure.")
        if coulomb.scale.scale_id != scale.scale_id:
            raise ValueError("Two-temperature scale identities differ.")
        values = tuple(
            float(value)
            for value in (
                electron_mass_per_particle,
                ion_mass_per_particle,
                electron_adiabatic_index,
                ion_adiabatic_index,
                energy_tolerance,
            )
        )
        if (
            any(not np.isfinite(value) or value <= 0.0 for value in values)
            or values[2] <= 1.0
            or values[3] <= 1.0
        ):
            raise ValueError("Two-temperature evolution controls are invalid.")
        self.scale = scale
        self.heating = heating
        self.coulomb = coulomb
        (
            self.electron_mass_per_particle,
            self.ion_mass_per_particle,
            self.electron_adiabatic_index,
            self.ion_adiabatic_index,
            self.energy_tolerance,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-two-temperature-evolution",
                "scale": scale.scale_id,
                "heating": heating.plan_id,
                "coulomb": coulomb.closure_id,
                "particle_masses": values[:2],
                "adiabatic_indices": values[2:4],
                "energy_tolerance": values[4],
            }
        )

    def initialize(
        self,
        rest_mass_density: ArrayLike,
        electron_temperature: ArrayLike,
        ion_temperature: ArrayLike,
        /,
    ) -> RelativisticTwoTemperatureState:
        density, electron, ion = jnp.broadcast_arrays(
            jnp.asarray(rest_mass_density),
            jnp.asarray(electron_temperature),
            jnp.asarray(ion_temperature),
        )
        boltzmann = jnp.asarray(float(self.scale.boltzmann_constant), density.dtype)
        electron_number = density / self.electron_mass_per_particle
        ion_number = density / self.ion_mass_per_particle
        electron_energy = (
            electron_number * boltzmann * electron / (self.electron_adiabatic_index - 1.0)
        )
        ion_energy = ion_number * boltzmann * ion / (self.ion_adiabatic_index - 1.0)
        return RelativisticTwoTemperatureState(
            electron_energy, ion_energy, electron, ion, density
        )

    def advance(
        self,
        state: RelativisticTwoTemperatureState,
        rest_mass_density: ArrayLike,
        total_internal_energy: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        gas_pressure: ArrayLike,
        magnetic_pressure: ArrayLike,
        radiation_exchange: ArrayLike = 0.0,
    ) -> RelativisticTwoTemperatureResult:
        if not isinstance(state, RelativisticTwoTemperatureState):
            raise TypeError("state must be RelativisticTwoTemperatureState.")
        density, total, step, gas, magnetic, radiation = jnp.broadcast_arrays(
            jnp.asarray(rest_mass_density),
            jnp.asarray(total_internal_energy),
            jnp.asarray(step_size),
            jnp.asarray(gas_pressure),
            jnp.asarray(magnetic_pressure),
            jnp.asarray(radiation_exchange),
        )
        ratio = density / state.rest_mass_density
        electron_adiabatic = (
            state.electron_internal_energy * ratio**self.electron_adiabatic_index
        )
        ion_adiabatic = state.ion_internal_energy * ratio**self.ion_adiabatic_index
        dissipation = total - electron_adiabatic - ion_adiabatic
        tolerance = self.energy_tolerance * jnp.maximum(jnp.abs(total), 1.0)
        dissipation_valid = dissipation >= -tolerance
        dissipative_heating = jnp.maximum(dissipation, 0.0)
        heating = self.heating.evaluate(
            state.electron_temperature,
            state.ion_temperature,
            gas,
            magnetic,
        )
        electron_heating = heating.electron_fraction * dissipative_heating
        ion_heating = dissipative_heating - electron_heating
        electron_pre_coulomb = electron_adiabatic + electron_heating + radiation
        ion_pre_coulomb = ion_adiabatic + ion_heating
        boltzmann = jnp.asarray(float(self.scale.boltzmann_constant), density.dtype)
        electron_number = density / self.electron_mass_per_particle
        ion_number = density / self.ion_mass_per_particle
        electron_capacity = (
            electron_number * boltzmann / (self.electron_adiabatic_index - 1.0)
        )
        ion_capacity = ion_number * boltzmann / (self.ion_adiabatic_index - 1.0)
        electron_temperature = electron_pre_coulomb / electron_capacity
        ion_temperature = ion_pre_coulomb / ion_capacity
        coulomb = self.coulomb.advance(
            electron_number,
            ion_number,
            TwoTemperaturePlasmaState(electron_temperature, ion_temperature),
            step,
        )
        electron_candidate = electron_pre_coulomb + coulomb.electron_energy_exchange
        ion_candidate = ion_pre_coulomb + coulomb.ion_energy_exchange
        target = total + radiation
        defect = electron_candidate + ion_candidate - target
        finite = (
            jnp.isfinite(density)
            & jnp.isfinite(total)
            & jnp.isfinite(step)
            & jnp.isfinite(defect)
            & jnp.isfinite(electron_candidate)
            & jnp.isfinite(ion_candidate)
        )
        physical = (
            finite
            & (density > 0.0)
            & (total >= 0.0)
            & (step >= 0.0)
            & (electron_candidate >= 0.0)
            & (ion_candidate >= 0.0)
            & dissipation_valid
            & heating.physically_valid
            & coulomb.physically_valid
        )
        scale = jnp.maximum(jnp.abs(target), 1.0)
        balanced = jnp.abs(defect) <= self.energy_tolerance * scale
        qualified_local = physical & heating.qualified & coulomb.qualified & balanced
        accepted = jnp.all(qualified_local)
        candidate = RelativisticTwoTemperatureState(
            electron_candidate,
            ion_candidate,
            coulomb.candidate.electron_temperature,
            coulomb.candidate.ion_temperature,
            density,
        )
        accepted_state = RelativisticTwoTemperatureState(
            jnp.where(
                accepted,
                candidate.electron_internal_energy,
                state.electron_internal_energy,
            ),
            jnp.where(accepted, candidate.ion_internal_energy, state.ion_internal_energy),
            jnp.where(
                accepted, candidate.electron_temperature, state.electron_temperature
            ),
            jnp.where(accepted, candidate.ion_temperature, state.ion_temperature),
            jnp.where(accepted, candidate.rest_mass_density, state.rest_mass_density),
        )
        ledger = RelativisticTwoTemperatureLedger(
            electron_adiabatic - state.electron_internal_energy,
            ion_adiabatic - state.ion_internal_energy,
            dissipative_heating,
            electron_heating,
            ion_heating,
            radiation,
            coulomb.electron_energy_exchange,
            coulomb.ion_energy_exchange,
            defect,
            jnp.minimum(electron_candidate, ion_candidate),
            jnp.all(finite),
            accepted,
            self.plan_id,
        )
        return RelativisticTwoTemperatureResult(
            candidate,
            accepted_state,
            heating,
            ledger,
            accepted,
            jnp.all(finite),
            jnp.all(physical),
            accepted,
            accepted
            & jnp.all(heating.derivative_valid)
            & jnp.all(coulomb.derivative_valid)
            & jnp.all(dissipation > -tolerance),
        )


class RelativisticPairState(StrictModule):
    electron_number_density: Array
    positron_number_density: Array
    photon_number_density: Array
    material_energy_density: Array
    radiation_energy_density: Array


class PairReactionLedger(StrictModule):
    pair_number_change: Array
    photon_number_change: Array
    material_energy_change: Array
    radiation_energy_change: Array
    charge_defect: Array
    photon_stoichiometry_defect: Array
    energy_defect: Array
    finite: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class PairReactionResult(StrictModule):
    candidate: RelativisticPairState
    state: RelativisticPairState
    ledger: PairReactionLedger
    accepted: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class PairCreationAnnihilationPlan(StrictModule, NonTrainableState):
    """Implicit charge-conserving photon-pair creation and annihilation."""

    pair_rest_energy: float = eqx.field(static=True)
    creation_coefficient: float = eqx.field(static=True)
    annihilation_coefficient: float = eqx.field(static=True)
    threshold_temperature: float = eqx.field(static=True)
    balance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        pair_rest_energy: float,
        creation_coefficient: float,
        annihilation_coefficient: float,
        threshold_temperature: float,
        balance_tolerance: float = 1.0e-10,
    ) -> None:
        values = tuple(
            float(value)
            for value in (
                pair_rest_energy,
                creation_coefficient,
                annihilation_coefficient,
                threshold_temperature,
                balance_tolerance,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Pair-reaction controls must be finite and positive.")
        (
            self.pair_rest_energy,
            self.creation_coefficient,
            self.annihilation_coefficient,
            self.threshold_temperature,
            self.balance_tolerance,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-pair-reaction",
                "pair_rest_energy": values[0],
                "creation_coefficient": values[1],
                "annihilation_coefficient": values[2],
                "threshold_temperature": values[3],
                "balance_tolerance": values[4],
            }
        )

    def advance(
        self,
        state: RelativisticPairState,
        radiation_temperature: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> PairReactionResult:
        if not isinstance(state, RelativisticPairState):
            raise TypeError("state must be RelativisticPairState.")
        electron, positron, photons, material, radiation, temperature, step = (
            jnp.broadcast_arrays(
                jnp.asarray(state.electron_number_density),
                jnp.asarray(state.positron_number_density),
                jnp.asarray(state.photon_number_density),
                jnp.asarray(state.material_energy_density),
                jnp.asarray(state.radiation_energy_density),
                jnp.asarray(radiation_temperature),
                jnp.asarray(step_size),
            )
        )
        charge = electron - positron
        activation = jnp.maximum(temperature / self.threshold_temperature - 1.0, 0.0)
        creation = self.creation_coefficient * activation * photons**2
        available_creation = jnp.minimum(
            0.5 * photons,
            radiation / (2.0 * self.pair_rest_energy),
        ) / jnp.maximum(step, jnp.finfo(step.dtype).tiny)
        creation = jnp.minimum(creation, available_creation)
        provisional = positron + step * creation
        coefficient = step * self.annihilation_coefficient
        linear = 1.0 + coefficient * charge
        discriminant = linear**2 + 4.0 * coefficient * provisional
        positron_candidate = jnp.where(
            coefficient > 0.0,
            2.0
            * provisional
            / jnp.maximum(
                linear + jnp.sqrt(jnp.maximum(discriminant, 0.0)),
                jnp.finfo(step.dtype).tiny,
            ),
            provisional,
        )
        electron_candidate = positron_candidate + charge
        pair_change = positron_candidate - positron
        photon_change = -2.0 * pair_change
        radiation_change = -2.0 * self.pair_rest_energy * pair_change
        material_change = -radiation_change
        candidate = RelativisticPairState(
            electron_candidate,
            positron_candidate,
            photons + photon_change,
            material + material_change,
            radiation + radiation_change,
        )
        charge_defect = (
            candidate.electron_number_density - candidate.positron_number_density - charge
        )
        photon_defect = photon_change + 2.0 * pair_change
        energy_defect = material_change + radiation_change
        finite = jnp.all(
            jnp.stack(
                (
                    jnp.isfinite(candidate.electron_number_density),
                    jnp.isfinite(candidate.positron_number_density),
                    jnp.isfinite(candidate.photon_number_density),
                    jnp.isfinite(candidate.material_energy_density),
                    jnp.isfinite(candidate.radiation_energy_density),
                    jnp.isfinite(charge_defect),
                    jnp.isfinite(photon_defect),
                    jnp.isfinite(energy_defect),
                )
            ),
            axis=0,
        )
        physical = (
            finite
            & (step >= 0.0)
            & (candidate.electron_number_density >= 0.0)
            & (candidate.positron_number_density >= 0.0)
            & (candidate.photon_number_density >= 0.0)
            & (candidate.material_energy_density >= 0.0)
            & (candidate.radiation_energy_density >= 0.0)
        )
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(material_change), jnp.abs(radiation_change)), 1.0
        )
        tolerance = self.balance_tolerance * scale
        balanced = (
            (jnp.abs(charge_defect) <= tolerance)
            & (jnp.abs(photon_defect) <= tolerance)
            & (jnp.abs(energy_defect) <= tolerance)
        )
        qualified_local = physical & balanced
        accepted = jnp.all(qualified_local)
        accepted_state = RelativisticPairState(
            jnp.where(accepted, candidate.electron_number_density, electron),
            jnp.where(accepted, candidate.positron_number_density, positron),
            jnp.where(accepted, candidate.photon_number_density, photons),
            jnp.where(accepted, candidate.material_energy_density, material),
            jnp.where(accepted, candidate.radiation_energy_density, radiation),
        )
        ledger = PairReactionLedger(
            pair_change,
            photon_change,
            material_change,
            radiation_change,
            charge_defect,
            photon_defect,
            energy_defect,
            jnp.all(finite),
            accepted,
            self.plan_id,
        )
        return PairReactionResult(
            candidate,
            accepted_state,
            ledger,
            accepted,
            jnp.all(finite),
            jnp.all(physical),
            accepted,
            accepted & jnp.all(discriminant > 0.0),
        )


class GyrotropicPlasmaState(StrictModule):
    parallel_pressure: Array
    perpendicular_pressure: Array
    temperature: Array


class GyrotropicPlasmaEvaluation(StrictModule):
    stress_covariant: Array
    heat_flux: Array
    entropy_production: Array
    firehose_margin: Array
    mirror_margin: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class GyrotropicPlasmaClosurePlan(StrictModule, NonTrainableState):
    """Pressure-anisotropic stress and field-aligned heat conduction."""

    conductivity: float = eqx.field(static=True)
    firehose_factor: float = eqx.field(static=True)
    mirror_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        conductivity: float,
        firehose_factor: float = 1.0,
        mirror_factor: float = 1.0,
    ) -> None:
        values = tuple(
            float(value) for value in (conductivity, firehose_factor, mirror_factor)
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Gyrotropic plasma controls must be nonnegative.")
        self.conductivity, self.firehose_factor, self.mirror_factor = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gyrotropic-plasma-closure",
                "conductivity": values[0],
                "firehose_factor": values[1],
                "mirror_factor": values[2],
            }
        )

    def evaluate(
        self,
        state: GyrotropicPlasmaState,
        magnetic_field: ArrayLike,
        temperature_gradient_covector: ArrayLike,
        spatial_metric: ArrayLike,
        inverse_spatial_metric: ArrayLike,
        /,
    ) -> GyrotropicPlasmaEvaluation:
        if not isinstance(state, GyrotropicPlasmaState):
            raise TypeError("state must be GyrotropicPlasmaState.")
        magnetic = jnp.asarray(magnetic_field)
        gradient = jnp.asarray(temperature_gradient_covector)
        metric = jnp.asarray(spatial_metric)
        inverse = jnp.asarray(inverse_spatial_metric)
        magnetic_covector = ein.contract("...ij,...j->...i", metric, magnetic)
        magnetic_squared = jnp.sum(magnetic_covector * magnetic, axis=-1)
        safe_magnetic = jnp.maximum(magnetic_squared, jnp.finfo(magnetic.dtype).tiny)
        unit_vector = magnetic / jnp.sqrt(safe_magnetic)[..., None]
        unit_covector = magnetic_covector / jnp.sqrt(safe_magnetic)[..., None]
        parallel_gradient = jnp.sum(unit_vector * gradient, axis=-1)
        heat_flux = -self.conductivity * parallel_gradient[..., None] * unit_vector
        anisotropy = state.parallel_pressure - state.perpendicular_pressure
        firehose_margin = self.firehose_factor * magnetic_squared - anisotropy
        beta_perpendicular = 2.0 * state.perpendicular_pressure / safe_magnetic
        mirror_bound = (
            self.mirror_factor
            * state.parallel_pressure
            / jnp.maximum(beta_perpendicular, jnp.finfo(magnetic.dtype).tiny)
        )
        mirror_margin = mirror_bound - (
            state.perpendicular_pressure - state.parallel_pressure
        )
        stress = (
            state.perpendicular_pressure[..., None, None] * metric
            + anisotropy[..., None, None]
            * unit_covector[..., :, None]
            * unit_covector[..., None, :]
        )
        entropy = (
            self.conductivity
            * parallel_gradient**2
            / jnp.maximum(state.temperature**2, jnp.finfo(magnetic.dtype).tiny)
        )
        identity = ein.contract("...ik,...kj->...ij", metric, inverse)
        finite = (
            jnp.all(jnp.isfinite(stress), axis=(-2, -1))
            & jnp.all(jnp.isfinite(heat_flux), axis=-1)
            & jnp.isfinite(entropy)
            & jnp.all(jnp.isfinite(identity), axis=(-2, -1))
        )
        physical = (
            finite
            & (state.parallel_pressure >= 0.0)
            & (state.perpendicular_pressure >= 0.0)
            & (state.temperature > 0.0)
            & (magnetic_squared > 0.0)
            & (entropy >= 0.0)
        )
        qualified = physical & (firehose_margin >= 0.0) & (mirror_margin >= 0.0)
        return GyrotropicPlasmaEvaluation(
            stress,
            heat_flux,
            entropy,
            firehose_margin,
            mirror_margin,
            finite,
            physical,
            qualified,
            qualified & (firehose_margin > 0.0) & (mirror_margin > 0.0),
            self.plan_id,
        )


__all__ = [
    "AbstractElectronHeatingPlan",
    "ConstantElectronHeatingPlan",
    "ElectronHeatingEvaluation",
    "GyrotropicPlasmaClosurePlan",
    "GyrotropicPlasmaEvaluation",
    "GyrotropicPlasmaState",
    "PairCreationAnnihilationPlan",
    "PairReactionLedger",
    "PairReactionResult",
    "ReconnectionElectronHeatingPlan",
    "RelativisticPairState",
    "RelativisticTwoTemperatureLedger",
    "RelativisticTwoTemperaturePlan",
    "RelativisticTwoTemperatureResult",
    "RelativisticTwoTemperatureState",
    "TurbulentElectronHeatingPlan",
]
