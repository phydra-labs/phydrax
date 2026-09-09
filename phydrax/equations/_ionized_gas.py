#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ._electrochemistry import FARADAY_CONSTANT
from ._hyperbolic_systems import (
    AbstractAdmissibleSystem,
    AbstractEntropyDiffusionSystem,
    AbstractNormalReflectionSystem,
)
from ._nonequilibrium_gas import (
    TwoTemperatureMixtureEulerSystem,
    TwoTemperatureMixtureNavierStokesSystem,
    TwoTemperatureRecovery,
    TwoTemperatureThermodynamicsPlan,
)
from ._transport_closures import AbstractTransportClosure, TransportProperties


class PlasmaQuasiNeutralityEvidence(StrictModule):
    charge_density: Array
    positive_charge_density: Array
    negative_charge_density: Array
    relative_defect: Array
    finite: Array
    successful: Array


class IonizedThermodynamicEvaluation(StrictModule):
    base_recovery: TwoTemperatureRecovery
    total_pressure: Array
    heavy_pressure: Array
    electron_pressure: Array
    electron_temperature: Array
    electron_molar_density: Array
    charge_density: Array
    frozen_sound_speed_squared: Array
    neutrality: PlasmaQuasiNeutralityEvidence
    finite: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class IonizedMixtureThermodynamicsPlan(StrictModule, NonTrainableState):
    """Ionized multitemperature closure with an explicit electron-energy mode."""

    base: TwoTemperatureThermodynamicsPlan
    electron_species_index: int = eqx.field(static=True)
    electron_mode_index: int = eqx.field(static=True)
    neutrality_tolerance: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: TwoTemperatureThermodynamicsPlan,
        electron_species_index: int,
        electron_mode_index: int,
        /,
        *,
        neutrality_tolerance: float = 1.0e-6,
    ):
        species_index = int(electron_species_index)
        mode_index = int(electron_mode_index)
        tolerance = float(neutrality_tolerance)
        if (
            not isinstance(base, TwoTemperatureThermodynamicsPlan)
            or not 0 <= species_index < base.schema.species_count
            or not 0 <= mode_index < base.modes.mode_count
            or int(base.schema.charges[species_index]) >= 0
            or not base.modes.modes[mode_index].pressure_bearing
            or base.modes.modes[mode_index].kind != "ideal-degrees-of-freedom"
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError(
                "Ionized thermodynamics requires a negative electron species and "
                "pressure-bearing ideal electron mode."
            )
        active = np.asarray(
            base.modes.modes[mode_index].characteristic_temperatures > 0.0
        )
        if np.count_nonzero(active) != 1 or not active[species_index]:
            raise ValueError("Electron mode must belong only to the electron species.")
        self.base = base
        self.electron_species_index = species_index
        self.electron_mode_index = mode_index
        self.neutrality_tolerance = tolerance
        self.model_id = canonical_fingerprint(
            {
                "kind": "ionized-mixture-thermodynamics",
                "base": base.model_id,
                "electron_species_index": species_index,
                "electron_mode_index": mode_index,
                "neutrality_tolerance": tolerance,
            }
        )

    @property
    def schema(self):
        return self.base.schema

    @property
    def modes(self):
        return self.base.modes

    def charge_density(self, species_mass_density: ArrayLike, /) -> Array:
        density = jnp.asarray(species_mass_density)
        concentration = density / self.schema.molar_masses.astype(density.dtype)
        return FARADAY_CONSTANT * contract(
            "...s,s->...",
            concentration,
            self.schema.charges.astype(density.dtype),
            backend="jax",
        )

    def neutrality_evidence(
        self, species_mass_density: ArrayLike, /
    ) -> PlasmaQuasiNeutralityEvidence:
        density = jnp.asarray(species_mass_density)
        concentration = density / self.schema.molar_masses.astype(density.dtype)
        signed = (
            FARADAY_CONSTANT * concentration * self.schema.charges.astype(density.dtype)
        )
        positive = jnp.sum(jnp.maximum(signed, 0.0), axis=-1)
        negative = -jnp.sum(jnp.minimum(signed, 0.0), axis=-1)
        charge = positive - negative
        scale = jnp.maximum(positive + negative, jnp.finfo(density.dtype).tiny)
        relative = jnp.abs(charge) / scale
        finite = (
            jnp.all(jnp.isfinite(density), axis=-1)
            & jnp.isfinite(charge)
            & jnp.isfinite(relative)
        )
        return PlasmaQuasiNeutralityEvidence(
            charge,
            positive,
            negative,
            relative,
            finite,
            finite & (relative <= self.neutrality_tolerance),
        )

    def recover_from_conserved(
        self,
        species_mass_density: ArrayLike,
        heavy_internal_energy_density: ArrayLike,
        mode_energy_densities: ArrayLike,
        /,
    ) -> IonizedThermodynamicEvaluation:
        species = jnp.asarray(species_mass_density)
        recovery = self.base.recover(
            species, heavy_internal_energy_density, mode_energy_densities
        )
        electron_temperature = recovery.state.mode_temperatures[
            ..., self.electron_mode_index
        ]
        electron_molar_density = (
            species[..., self.electron_species_index]
            / self.schema.molar_masses[self.electron_species_index]
        )
        electron_pressure = (
            UNIVERSAL_GAS_CONSTANT * electron_molar_density * electron_temperature
        )
        total_pressure = recovery.state.pressure + electron_pressure
        electron_gamma = 5.0 / 3.0
        sound_squared = (
            recovery.state.frozen_sound_speed_squared
            + electron_gamma
            * electron_pressure
            / jnp.maximum(
                recovery.state.mass_density,
                jnp.finfo(species.dtype).tiny,
            )
        )
        neutrality = self.neutrality_evidence(species)
        finite = (
            recovery.finite
            & jnp.isfinite(electron_temperature)
            & jnp.isfinite(electron_pressure)
            & jnp.isfinite(total_pressure)
            & jnp.isfinite(sound_squared)
            & neutrality.finite
        )
        successful = (
            finite
            & recovery.successful
            & (electron_temperature > 0.0)
            & (electron_pressure >= 0.0)
            & (total_pressure > 0.0)
            & (sound_squared > 0.0)
        )
        return IonizedThermodynamicEvaluation(
            recovery,
            total_pressure,
            recovery.state.pressure,
            electron_pressure,
            electron_temperature,
            electron_molar_density,
            neutrality.charge_density,
            sound_squared,
            neutrality,
            finite,
            successful,
            self.model_id,
        )


class IonizedMultitemperatureEulerSystem(
    AbstractAdmissibleSystem, AbstractNormalReflectionSystem
):
    """Ionized Euler transport over the canonical species/mode state layout."""

    thermodynamics: IonizedMixtureThermodynamicsPlan
    base: TwoTemperatureMixtureEulerSystem

    def __init__(
        self,
        thermodynamics: IonizedMixtureThermodynamicsPlan,
        dimension: int = 1,
        /,
        *,
        density_floor: float = 1.0e-12,
        pressure_floor: float = 1.0e-12,
    ):
        if not isinstance(thermodynamics, IonizedMixtureThermodynamicsPlan):
            raise TypeError("thermodynamics must be IonizedMixtureThermodynamicsPlan.")
        base = TwoTemperatureMixtureEulerSystem(
            thermodynamics.base,
            dimension,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
        )
        self.thermodynamics = thermodynamics
        self.base = base
        self.dimension = base.dimension
        self.component_names = base.component_names
        self.system_id = canonical_fingerprint(
            {
                "kind": "ionized-multitemperature-euler",
                "thermodynamics": thermodynamics.model_id,
                "dimension": self.dimension,
                "density_floor": density_floor,
                "pressure_floor": pressure_floor,
            }
        )

    @property
    def species_count(self) -> int:
        return self.base.species_count

    @property
    def mode_count(self) -> int:
        return self.base.mode_count

    @property
    def momentum_slice(self) -> slice:
        return self.base.momentum_slice

    @property
    def energy_index(self) -> int:
        return self.base.energy_index

    @property
    def mode_slice(self) -> slice:
        return self.base.mode_slice

    @property
    def density_floor(self) -> float:
        return self.base.density_floor

    @property
    def pressure_floor(self) -> float:
        return self.base.pressure_floor

    @property
    def maximum_thermal_iterations(self) -> int:
        return self.base.maximum_thermal_iterations

    def density(self, state: ArrayLike, /) -> Array:
        return self.base.density(state)

    def recover_thermodynamics(
        self, state: ArrayLike, /
    ) -> IonizedThermodynamicEvaluation:
        value = jnp.asarray(state)
        density = self.base.density(value)
        momentum = value[..., self.momentum_slice]
        kinetic = (
            0.5
            * contract("...i,...i->...", momentum, momentum, backend="jax")
            / jnp.maximum(density, self.density_floor)
        )
        modes = value[..., self.mode_slice]
        heavy_energy = value[..., self.energy_index] - kinetic - jnp.sum(modes, axis=-1)
        return self.thermodynamics.recover_from_conserved(
            value[..., : self.species_count], heavy_energy, modes
        )

    def pressure(self, state: ArrayLike, /) -> Array:
        return self.recover_thermodynamics(state).total_pressure

    def temperature(self, state: ArrayLike, /) -> Array:
        return self.recover_thermodynamics(state).base_recovery.state.heavy_temperature

    def mode_temperatures(self, state: ArrayLike, /) -> Array:
        return self.recover_thermodynamics(state).base_recovery.state.mode_temperatures

    def electron_temperature(self, state: ArrayLike, /) -> Array:
        return self.recover_thermodynamics(state).electron_temperature

    def charge_density(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        return self.thermodynamics.charge_density(value[..., : self.species_count])

    def frozen_sound_speed(self, state: ArrayLike, /) -> Array:
        return jnp.sqrt(self.recover_thermodynamics(state).frozen_sound_speed_squared)

    def conserved_to_primitive(self, state: Array, /) -> Array:
        return self.base.conserved_to_primitive(state)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return self.base.primitive_to_conserved(primitive)

    def primitive_velocity(self, primitive: Array, /) -> Array:
        return self.base.primitive_velocity(primitive)

    def with_primitive_velocity(self, primitive: Array, velocity: Array, /) -> Array:
        return self.base.with_primitive_velocity(primitive, velocity)

    def with_primitive_temperature(
        self, primitive: Array, temperature: Array, /
    ) -> Array:
        return self.base.with_primitive_temperature(primitive, temperature)

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        flux = self.base.physical_flux(state, axis, args)
        pressure_correction = self.pressure(state) - self.base.pressure(state)
        return (
            flux.at[..., self.momentum_slice.start + int(axis)]
            .add(pressure_correction)
            .at[..., self.energy_index]
            .add(
                pressure_correction
                * self.primitive_velocity(self.conserved_to_primitive(state))[
                    ..., int(axis)
                ]
            )
        )

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        axis_ = int(axis)
        left_primitive = self.conserved_to_primitive(left)
        right_primitive = self.conserved_to_primitive(right)
        left_velocity = self.primitive_velocity(left_primitive)[..., axis_]
        right_velocity = self.primitive_velocity(right_primitive)[..., axis_]
        left_sound = self.frozen_sound_speed(left)
        right_sound = self.frozen_sound_speed(right)
        return (
            jnp.minimum(left_velocity - left_sound, right_velocity - right_sound),
            jnp.maximum(left_velocity + left_sound, right_velocity + right_sound),
        )

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        lower, upper = self.signal_bounds(left, right, axis, args)
        return jnp.maximum(jnp.abs(lower), jnp.abs(upper))

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        normal_ = jnp.asarray(normal)
        left_velocity = self.primitive_velocity(self.conserved_to_primitive(left))
        right_velocity = self.primitive_velocity(self.conserved_to_primitive(right))
        left_normal = contract("...i,...i->...", left_velocity, normal_, backend="jax")
        right_normal = contract("...i,...i->...", right_velocity, normal_, backend="jax")
        left_sound = self.frozen_sound_speed(left)
        right_sound = self.frozen_sound_speed(right)
        return (
            jnp.minimum(left_normal - left_sound, right_normal - right_sound),
            jnp.maximum(left_normal + left_sound, right_normal + right_sound),
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return self.base.reflect_state(state, axis)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        return self.base.reflect_normal_state(state, normal)

    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        recovered = self.recover_thermodynamics(value)
        return (
            self.base.admissible(value)
            & recovered.successful
            & (recovered.total_pressure > self.pressure_floor)
        )


class IonizedMultitemperatureNavierStokesSystem(
    AbstractAdmissibleSystem,
    AbstractNormalReflectionSystem,
    AbstractEntropyDiffusionSystem,
):
    """Ionized Euler pressure with molecular and modal diffusion."""

    inviscid: IonizedMultitemperatureEulerSystem
    base: TwoTemperatureMixtureNavierStokesSystem

    def __init__(
        self,
        thermodynamics: IonizedMixtureThermodynamicsPlan,
        transport: AbstractTransportClosure,
        dimension: int = 1,
        /,
        *,
        mode_diffusivities: ArrayLike | None = None,
    ):
        inviscid = IonizedMultitemperatureEulerSystem(thermodynamics, dimension)
        base = TwoTemperatureMixtureNavierStokesSystem(
            thermodynamics.base,
            transport,
            dimension,
            mode_diffusivities=mode_diffusivities,
        )
        self.inviscid = inviscid
        self.base = base
        self.dimension = dimension
        self.component_names = inviscid.component_names
        self.system_id = canonical_fingerprint(
            {
                "kind": "ionized-multitemperature-navier-stokes",
                "inviscid": inviscid.system_id,
                "transport": transport.closure_id,
                "mode_diffusivities": base.mode_diffusivities,
            }
        )

    @property
    def thermodynamics(self):
        return self.inviscid.thermodynamics

    @property
    def transport(self):
        return self.base.transport

    @property
    def species_count(self) -> int:
        return self.inviscid.species_count

    @property
    def mode_count(self) -> int:
        return self.inviscid.mode_count

    @property
    def momentum_slice(self) -> slice:
        return self.inviscid.momentum_slice

    @property
    def energy_index(self) -> int:
        return self.inviscid.energy_index

    @property
    def mode_slice(self) -> slice:
        return self.inviscid.mode_slice

    @property
    def density_floor(self) -> float:
        return self.inviscid.density_floor

    @property
    def pressure_floor(self) -> float:
        return self.inviscid.pressure_floor

    @property
    def maximum_thermal_iterations(self) -> int:
        return self.inviscid.maximum_thermal_iterations

    def density(self, state: ArrayLike, /) -> Array:
        return self.inviscid.density(state)

    def pressure(self, state: ArrayLike, /) -> Array:
        return self.inviscid.pressure(state)

    def temperature(self, state: ArrayLike, /) -> Array:
        return self.inviscid.temperature(state)

    def electron_temperature(self, state: ArrayLike, /) -> Array:
        return self.inviscid.electron_temperature(state)

    def charge_density(self, state: ArrayLike, /) -> Array:
        return self.inviscid.charge_density(state)

    def recover_thermodynamics(self, state: ArrayLike, /):
        return self.inviscid.recover_thermodynamics(state)

    def conserved_to_primitive(self, state: Array, /) -> Array:
        return self.inviscid.conserved_to_primitive(state)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return self.inviscid.primitive_to_conserved(primitive)

    def primitive_velocity(self, primitive: Array, /) -> Array:
        return self.inviscid.primitive_velocity(primitive)

    def with_primitive_velocity(self, primitive: Array, velocity: Array, /) -> Array:
        return self.inviscid.with_primitive_velocity(primitive, velocity)

    def with_primitive_temperature(
        self, primitive: Array, temperature: Array, /
    ) -> Array:
        return self.inviscid.with_primitive_temperature(primitive, temperature)

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        return self.inviscid.physical_flux(state, axis, args)

    def max_wave_speed(self, *args, **kwargs):
        return self.inviscid.max_wave_speed(*args, **kwargs)

    def signal_bounds(self, *args, **kwargs):
        return self.inviscid.signal_bounds(*args, **kwargs)

    def normal_signal_bounds(self, *args, **kwargs):
        return self.inviscid.normal_signal_bounds(*args, **kwargs)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return self.inviscid.reflect_state(state, axis)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        return self.inviscid.reflect_normal_state(state, normal)

    def admissible(self, state: Array, /) -> Array:
        return self.inviscid.admissible(state)

    def transport_properties(
        self, state: ArrayLike, args: Any = None, /
    ) -> TransportProperties:
        return self.base.transport_properties(state, args)

    def viscous_flux(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        return self.base.viscous_flux(state, conserved_gradient, args)

    def maximum_diffusivity(self, state: Array, args: Any = None, /) -> Array:
        return self.base.maximum_diffusivity(state, args)

    def entropy_viscous_production(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        return self.base.entropy_viscous_production(state, conserved_gradient, args)


__all__ = [
    "IonizedMixtureThermodynamicsPlan",
    "IonizedMultitemperatureEulerSystem",
    "IonizedMultitemperatureNavierStokesSystem",
    "IonizedThermodynamicEvaluation",
    "PlasmaQuasiNeutralityEvidence",
]
