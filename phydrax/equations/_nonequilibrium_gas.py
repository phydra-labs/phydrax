#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_thermodynamics import (
    AbstractSpeciesThermodynamicsPlan,
    UNIVERSAL_GAS_CONSTANT,
)
from ._hyperbolic_systems import (
    AbstractAdmissibleSystem,
    AbstractEntropyDiffusionSystem,
    AbstractNormalReflectionSystem,
)
from ._thermal_modes import (
    ThermalModeEvaluation,
    ThermalModeSchema,
    ThermalModeTemperatureResult,
)
from ._transport_closures import AbstractTransportClosure, TransportProperties


@jax.custom_jvp
def _implicit_heavy_temperature(
    temperature: Array,
    target_energy: Array,
    evaluated_energy: Array,
    heat_capacity: Array,
    /,
) -> Array:
    del target_energy, evaluated_energy, heat_capacity
    return temperature


@_implicit_heavy_temperature.defjvp
def _implicit_heavy_temperature_jvp(primals, tangents):
    temperature, _, _, heat_capacity = primals
    _, target_tangent, evaluated_tangent, _ = tangents
    tangent = (target_tangent - evaluated_tangent) / heat_capacity
    return temperature, tangent


class TwoTemperatureThermodynamicEvaluation(StrictModule):
    heavy_temperature: Array
    mode_temperatures: Array
    mass_density: Array
    molar_density: Array
    mole_fraction: Array
    pressure: Array
    heavy_internal_energy_density: Array
    mode_energy_densities: Array
    heavy_heat_capacity_volume: Array
    heavy_heat_capacity_pressure: Array
    frozen_sound_speed_squared: Array
    mode_evaluation: ThermalModeEvaluation
    finite: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class TwoTemperatureRecovery(StrictModule):
    state: TwoTemperatureThermodynamicEvaluation
    heavy_energy_residual: Array
    mode_energy_residual: Array
    heavy_temperature_margin: Array
    mode_temperature_margin: Array
    iteration_count: Array
    finite: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class TwoTemperatureThermodynamicsPlan(StrictModule, NonTrainableState):
    """Ideal heavy-particle calorics plus disjoint explicit thermal modes."""

    heavy_thermodynamics: AbstractSpeciesThermodynamicsPlan
    modes: ThermalModeSchema
    maximum_iterations: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        heavy_thermodynamics: AbstractSpeciesThermodynamicsPlan,
        modes: ThermalModeSchema,
        /,
        *,
        maximum_iterations: int = 80,
    ):
        iterations = int(maximum_iterations)
        if (
            not isinstance(heavy_thermodynamics, AbstractSpeciesThermodynamicsPlan)
            or not isinstance(modes, ThermalModeSchema)
            or heavy_thermodynamics.schema.schema_id != modes.species.schema_id
            or iterations <= 0
        ):
            raise ValueError("Two-temperature calorics and mode schema must match.")
        self.heavy_thermodynamics = heavy_thermodynamics
        self.modes = modes
        self.maximum_iterations = iterations
        self.model_id = canonical_fingerprint(
            {
                "kind": "two-temperature-thermodynamics",
                "heavy_thermodynamics": heavy_thermodynamics.thermodynamics_id,
                "modes": modes.schema_id,
                "maximum_iterations": iterations,
            }
        )

    @property
    def schema(self):
        return self.heavy_thermodynamics.schema

    def _heavy_state(
        self, species_mass_density: Array, temperature: Array, /
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        density = jnp.asarray(species_mass_density)
        temperature_ = jnp.asarray(temperature, dtype=density.dtype)
        species = self.heavy_thermodynamics.evaluate(temperature_)
        amount_density = density / self.schema.molar_masses.astype(density.dtype)
        molar_density = jnp.sum(amount_density, axis=-1)
        safe_molar_density = jnp.maximum(molar_density, jnp.finfo(density.dtype).tiny)
        mole_fraction = amount_density / safe_molar_density[..., None]
        mass_density = jnp.sum(density, axis=-1)
        internal_energy = contract(
            "...s,...s->...",
            amount_density,
            species.molar_internal_energy,
            backend="jax",
        )
        heat_capacity_volume = contract(
            "...s,...s->...",
            amount_density,
            species.molar_heat_capacity_volume,
            backend="jax",
        )
        heat_capacity_pressure = heat_capacity_volume + (
            UNIVERSAL_GAS_CONSTANT * molar_density
        )
        pressure = UNIVERSAL_GAS_CONSTANT * molar_density * temperature_
        successful = (
            species.successful
            & jnp.all(density >= 0.0, axis=-1)
            & (mass_density > 0.0)
            & (molar_density > 0.0)
            & (heat_capacity_volume > 0.0)
            & (heat_capacity_pressure > 0.0)
            & (pressure > 0.0)
        )
        return (
            mass_density,
            molar_density,
            mole_fraction,
            pressure,
            internal_energy,
            heat_capacity_volume,
            heat_capacity_pressure,
            successful,
        )

    def evaluate(
        self,
        species_mass_density: ArrayLike,
        heavy_temperature: ArrayLike,
        mode_temperatures: ArrayLike,
        /,
    ) -> TwoTemperatureThermodynamicEvaluation:
        density = jnp.asarray(species_mass_density)
        heavy = jnp.asarray(heavy_temperature, dtype=density.dtype)
        mode_temperature = jnp.asarray(mode_temperatures, dtype=density.dtype)
        if (
            density.ndim < 1
            or density.shape[-1] != self.schema.species_count
            or heavy.shape != density.shape[:-1]
            or mode_temperature.shape != density.shape[:-1] + (self.modes.mode_count,)
        ):
            raise ValueError("Two-temperature thermodynamic state shapes are invalid.")
        (
            mass_density,
            molar_density,
            mole_fraction,
            pressure,
            heavy_energy,
            cv,
            cp,
            heavy_successful,
        ) = self._heavy_state(density, heavy)
        mode = self.modes.evaluate(density, mode_temperature)
        sound_squared = (cp / cv) * pressure / mass_density
        finite = (
            jnp.all(jnp.isfinite(density), axis=-1)
            & jnp.isfinite(heavy)
            & jnp.all(jnp.isfinite(mode_temperature), axis=-1)
            & jnp.isfinite(sound_squared)
        )
        successful = finite & heavy_successful & mode.successful & (sound_squared > 0.0)
        return TwoTemperatureThermodynamicEvaluation(
            heavy,
            mode_temperature,
            mass_density,
            molar_density,
            mole_fraction,
            pressure,
            heavy_energy,
            mode.energy_densities,
            cv,
            cp,
            sound_squared,
            mode,
            finite,
            successful,
            self.model_id,
        )

    def solve_heavy_temperature(
        self,
        species_mass_density: ArrayLike,
        heavy_internal_energy_density: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array, Array, Array]:
        density = jnp.asarray(species_mass_density)
        target = jnp.asarray(heavy_internal_energy_density, dtype=density.dtype)
        cell_shape = density.shape[:-1]
        if target.shape != cell_shape:
            raise ValueError("Heavy internal energy must match the density cells.")
        lower = jnp.full(
            cell_shape,
            self.heavy_thermodynamics.minimum_temperature,
            dtype=density.dtype,
        )
        upper = jnp.full(
            cell_shape,
            self.heavy_thermodynamics.maximum_temperature,
            dtype=density.dtype,
        )
        lower_energy = self._heavy_state(density, lower)[4]
        upper_energy = self._heavy_state(density, upper)[4]
        bracketed = (target >= lower_energy) & (target <= upper_energy)

        def body(_, bounds):
            low, high = bounds
            midpoint = 0.5 * (low + high)
            energy = self._heavy_state(density, midpoint)[4]
            choose_lower = energy < target
            return jnp.where(choose_lower, midpoint, low), jnp.where(
                choose_lower, high, midpoint
            )

        lower, upper = jax.lax.fori_loop(0, self.maximum_iterations, body, (lower, upper))
        raw = 0.5 * (lower + upper)
        heavy_state = self._heavy_state(density, raw)
        temperature = _implicit_heavy_temperature(
            raw, target, heavy_state[4], heavy_state[5]
        )
        final = self._heavy_state(density, temperature)
        residual = final[4] - target
        margin = jnp.minimum(
            temperature - self.heavy_thermodynamics.minimum_temperature,
            self.heavy_thermodynamics.maximum_temperature - temperature,
        )
        scale = jnp.maximum(jnp.abs(target), 1.0)
        successful = (
            bracketed
            & final[7]
            & jnp.isfinite(residual)
            & (jnp.abs(residual) <= 128.0 * jnp.finfo(density.dtype).eps * scale)
        )
        return temperature, residual, margin, successful, final[5]

    def recover(
        self,
        species_mass_density: ArrayLike,
        heavy_internal_energy_density: ArrayLike,
        mode_energy_densities: ArrayLike,
        /,
    ) -> TwoTemperatureRecovery:
        density = jnp.asarray(species_mass_density)
        heavy_energy = jnp.asarray(heavy_internal_energy_density, dtype=density.dtype)
        mode_energy = jnp.asarray(mode_energy_densities, dtype=density.dtype)
        heavy_temperature, heavy_residual, heavy_margin, heavy_successful, _ = (
            self.solve_heavy_temperature(density, heavy_energy)
        )
        mode_result: ThermalModeTemperatureResult = self.modes.solve_temperatures(
            density,
            mode_energy,
            maximum_iterations=self.maximum_iterations,
        )
        state = self.evaluate(density, heavy_temperature, mode_result.temperatures)
        finite = (
            state.finite
            & jnp.isfinite(heavy_residual)
            & jnp.all(jnp.isfinite(mode_result.energy_residual), axis=-1)
        )
        successful = finite & heavy_successful & mode_result.successful & state.successful
        return TwoTemperatureRecovery(
            state,
            heavy_residual,
            mode_result.energy_residual,
            heavy_margin,
            mode_result.bracket_margin,
            jnp.asarray(self.maximum_iterations, dtype=jnp.int32),
            finite,
            successful,
            self.model_id,
        )

    def equilibrium_internal_energy_density(
        self, species_mass_density: ArrayLike, temperature: ArrayLike, /
    ) -> Array:
        density = jnp.asarray(species_mass_density)
        temperature_ = jnp.asarray(temperature, dtype=density.dtype)
        mode_temperatures = jnp.broadcast_to(
            temperature_[..., None],
            density.shape[:-1] + (self.modes.mode_count,),
        )
        evaluation = self.evaluate(density, temperature_, mode_temperatures)
        return evaluation.heavy_internal_energy_density + jnp.sum(
            evaluation.mode_energy_densities, axis=-1
        )


class TwoTemperatureMixtureEulerSystem(
    AbstractAdmissibleSystem, AbstractNormalReflectionSystem
):
    """Neutral multi-species Euler system with explicit thermal-mode energies."""

    thermodynamics: TwoTemperatureThermodynamicsPlan
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    maximum_thermal_iterations: int = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: TwoTemperatureThermodynamicsPlan,
        dimension: int = 1,
        /,
        *,
        density_floor: float = 1.0e-12,
        pressure_floor: float = 1.0e-12,
        maximum_thermal_iterations: int | None = None,
    ):
        dimension_ = int(dimension)
        density_floor_ = float(density_floor)
        pressure_floor_ = float(pressure_floor)
        iterations = (
            thermodynamics.maximum_iterations
            if maximum_thermal_iterations is None
            else int(maximum_thermal_iterations)
        )
        if (
            not isinstance(thermodynamics, TwoTemperatureThermodynamicsPlan)
            or dimension_ not in (1, 2, 3)
            or not np.isfinite(density_floor_)
            or density_floor_ <= 0.0
            or not np.isfinite(pressure_floor_)
            or pressure_floor_ <= 0.0
            or iterations <= 0
        ):
            raise ValueError("Two-temperature Euler system inputs are invalid.")
        self.thermodynamics = thermodynamics
        self.dimension = dimension_
        self.density_floor = density_floor_
        self.pressure_floor = pressure_floor_
        self.maximum_thermal_iterations = iterations
        self.component_names = (
            *(f"species_density:{name}" for name in thermodynamics.schema.species_names),
            *(f"momentum_{axis}" for axis in range(dimension_)),
            "total_energy",
            *(f"mode_energy:{name}" for name in thermodynamics.modes.mode_names),
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "two-temperature-mixture-euler",
                "thermodynamics": thermodynamics.model_id,
                "dimension": dimension_,
                "density_floor": density_floor_,
                "pressure_floor": pressure_floor_,
                "maximum_thermal_iterations": iterations,
            }
        )

    @property
    def species_count(self) -> int:
        return self.thermodynamics.schema.species_count

    @property
    def mode_count(self) -> int:
        return self.thermodynamics.modes.mode_count

    @property
    def momentum_slice(self) -> slice:
        return slice(self.species_count, self.species_count + self.dimension)

    @property
    def energy_index(self) -> int:
        return self.species_count + self.dimension

    @property
    def mode_slice(self) -> slice:
        return slice(self.energy_index + 1, self.component_count)

    def _check_state(self, state: ArrayLike, name: str, /) -> Array:
        value = jnp.asarray(state)
        if value.ndim < 1 or value.shape[-1] != self.component_count:
            raise ValueError(f"{name} must end in {self.component_count} components.")
        return value

    def density(self, state: ArrayLike, /) -> Array:
        value = self._check_state(state, "Two-temperature state")
        return jnp.sum(value[..., : self.species_count], axis=-1)

    def recover_thermodynamics(self, state: ArrayLike, /) -> TwoTemperatureRecovery:
        value = self._check_state(state, "Two-temperature state")
        species_density = value[..., : self.species_count]
        density = jnp.sum(species_density, axis=-1)
        momentum = value[..., self.momentum_slice]
        kinetic = (
            0.5
            * contract("...i,...i->...", momentum, momentum, backend="jax")
            / jnp.maximum(density, self.density_floor)
        )
        mode_energy = value[..., self.mode_slice]
        heavy_energy = (
            value[..., self.energy_index] - kinetic - jnp.sum(mode_energy, axis=-1)
        )
        return self.thermodynamics.recover(species_density, heavy_energy, mode_energy)

    def pressure(self, state: ArrayLike, /) -> Array:
        return self.recover_thermodynamics(state).state.pressure

    def temperature(self, state: ArrayLike, /) -> Array:
        return self.recover_thermodynamics(state).state.heavy_temperature

    def mode_temperatures(self, state: ArrayLike, /) -> Array:
        return self.recover_thermodynamics(state).state.mode_temperatures

    def frozen_sound_speed(self, state: ArrayLike, /) -> Array:
        return jnp.sqrt(
            jnp.maximum(
                self.recover_thermodynamics(state).state.frozen_sound_speed_squared,
                0.0,
            )
        )

    def primitive_velocity(self, primitive: Array, /) -> Array:
        value = self._check_state(primitive, "Two-temperature primitive state")
        return value[..., self.momentum_slice]

    def with_primitive_velocity(self, primitive: Array, velocity: Array, /) -> Array:
        value = self._check_state(primitive, "Two-temperature primitive state")
        return value.at[..., self.momentum_slice].set(velocity)

    def with_primitive_temperature(
        self, primitive: Array, temperature: Array, /
    ) -> Array:
        value = self._check_state(primitive, "Two-temperature primitive state")
        return value.at[..., self.energy_index].set(temperature)

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = self._check_state(state, "Two-temperature state")
        density = self.density(value)
        recovered = self.recover_thermodynamics(value)
        velocity = value[..., self.momentum_slice] / jnp.maximum(
            density[..., None], self.density_floor
        )
        return jnp.concatenate(
            (
                value[..., : self.species_count],
                velocity,
                recovered.state.heavy_temperature[..., None],
                recovered.state.mode_temperatures,
            ),
            axis=-1,
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = self._check_state(primitive, "Two-temperature primitive state")
        species_density = value[..., : self.species_count]
        velocity = value[..., self.momentum_slice]
        heavy_temperature = value[..., self.energy_index]
        mode_temperatures = value[..., self.mode_slice]
        thermodynamic = self.thermodynamics.evaluate(
            species_density, heavy_temperature, mode_temperatures
        )
        density = jnp.sum(species_density, axis=-1)
        kinetic = (
            0.5 * density * contract("...i,...i->...", velocity, velocity, backend="jax")
        )
        total_energy = (
            thermodynamic.heavy_internal_energy_density
            + jnp.sum(thermodynamic.mode_energy_densities, axis=-1)
            + kinetic
        )
        return jnp.concatenate(
            (
                species_density,
                density[..., None] * velocity,
                total_energy[..., None],
                thermodynamic.mode_energy_densities,
            ),
            axis=-1,
        )

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("Two-temperature flux axis is out of range.")
        value = self._check_state(state, "Two-temperature state")
        density = self.density(value)
        velocity = value[..., self.momentum_slice] / jnp.maximum(
            density[..., None], self.density_floor
        )
        normal_velocity = velocity[..., axis_]
        pressure = self.pressure(value)
        species_flux = value[..., : self.species_count] * normal_velocity[..., None]
        momentum_flux = value[..., self.momentum_slice] * normal_velocity[..., None]
        momentum_flux = momentum_flux.at[..., axis_].add(pressure)
        energy_flux = (value[..., self.energy_index] + pressure) * normal_velocity
        mode_flux = value[..., self.mode_slice] * normal_velocity[..., None]
        return jnp.concatenate(
            (
                species_flux,
                momentum_flux,
                energy_flux[..., None],
                mode_flux,
            ),
            axis=-1,
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
        left_value = self._check_state(left, "Left state")
        right_value = self._check_state(right, "Right state")
        left_density = self.density(left_value)
        right_density = self.density(right_value)
        left_velocity = (
            left_value[..., self.momentum_slice]
            / jnp.maximum(left_density[..., None], self.density_floor)
        )[..., axis_]
        right_velocity = (
            right_value[..., self.momentum_slice]
            / jnp.maximum(right_density[..., None], self.density_floor)
        )[..., axis_]
        left_sound = self.frozen_sound_speed(left_value)
        right_sound = self.frozen_sound_speed(right_value)
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
        if normal_.shape[-1] != self.dimension:
            raise ValueError("Normal must end in the physical dimension.")
        left_value = self._check_state(left, "Left state")
        right_value = self._check_state(right, "Right state")
        left_density = self.density(left_value)
        right_density = self.density(right_value)
        left_velocity = left_value[..., self.momentum_slice] / jnp.maximum(
            left_density[..., None], self.density_floor
        )
        right_velocity = right_value[..., self.momentum_slice] / jnp.maximum(
            right_density[..., None], self.density_floor
        )
        left_normal = contract("...i,...i->...", left_velocity, normal_, backend="jax")
        right_normal = contract("...i,...i->...", right_velocity, normal_, backend="jax")
        left_sound = self.frozen_sound_speed(left_value)
        right_sound = self.frozen_sound_speed(right_value)
        return (
            jnp.minimum(left_normal - left_sound, right_normal - right_sound),
            jnp.maximum(left_normal + left_sound, right_normal + right_sound),
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        value = self._check_state(state, "Two-temperature state")
        index = self.species_count + int(axis)
        if not self.species_count <= index < self.energy_index:
            raise ValueError("Reflection axis is out of range.")
        return value.at[..., index].multiply(-1.0)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        value = self._check_state(state, "Two-temperature state")
        normal_ = jnp.asarray(normal)
        if normal_.shape[-1] != self.dimension:
            raise ValueError("Normal must end in the physical dimension.")
        momentum = value[..., self.momentum_slice]
        reflected = (
            momentum
            - 2.0
            * contract("...i,...i->...", momentum, normal_, backend="jax")[..., None]
            * normal_
        )
        return value.at[..., self.momentum_slice].set(reflected)

    def admissible(self, state: Array, /) -> Array:
        value = self._check_state(state, "Two-temperature state")
        species = value[..., : self.species_count]
        modes = value[..., self.mode_slice]
        recovered = self.recover_thermodynamics(value)
        return (
            jnp.all(jnp.isfinite(value), axis=-1)
            & jnp.all(species >= 0.0, axis=-1)
            & (jnp.sum(species, axis=-1) > self.density_floor)
            & jnp.all(modes >= 0.0, axis=-1)
            & recovered.successful
            & (recovered.state.pressure > self.pressure_floor)
        )


class TwoTemperatureMixtureNavierStokesSystem(
    AbstractAdmissibleSystem,
    AbstractNormalReflectionSystem,
    AbstractEntropyDiffusionSystem,
):
    """Two-temperature Euler transport with plus molecular and modal diffusion."""

    inviscid: TwoTemperatureMixtureEulerSystem
    thermodynamics: TwoTemperatureThermodynamicsPlan
    transport: AbstractTransportClosure
    mode_diffusivities: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: TwoTemperatureThermodynamicsPlan,
        transport: AbstractTransportClosure,
        dimension: int = 1,
        /,
        *,
        mode_diffusivities: ArrayLike | None = None,
        density_floor: float = 1.0e-12,
        pressure_floor: float = 1.0e-12,
        maximum_thermal_iterations: int | None = None,
    ):
        if not isinstance(transport, AbstractTransportClosure):
            raise TypeError("Two-temperature transport must be AbstractTransportClosure.")
        inviscid = TwoTemperatureMixtureEulerSystem(
            thermodynamics,
            dimension,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
            maximum_thermal_iterations=maximum_thermal_iterations,
        )
        values = (
            np.zeros(inviscid.mode_count, dtype=float)
            if mode_diffusivities is None
            else np.asarray(mode_diffusivities, dtype=float)
        )
        if (
            values.shape != (inviscid.mode_count,)
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError(
                "mode_diffusivities must contain one nonnegative value per mode."
            )
        self.inviscid = inviscid
        self.thermodynamics = thermodynamics
        self.transport = transport
        self.mode_diffusivities = tuple(float(value) for value in values)
        self.dimension = inviscid.dimension
        self.component_names = inviscid.component_names
        self.system_id = canonical_fingerprint(
            {
                "kind": "two-temperature-mixture-navier-stokes",
                "inviscid": inviscid.system_id,
                "transport": transport.closure_id,
                "mode_diffusivities": self.mode_diffusivities,
            }
        )

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

    def mode_temperatures(self, state: ArrayLike, /) -> Array:
        return self.inviscid.mode_temperatures(state)

    def recover_thermodynamics(self, state: ArrayLike, /) -> TwoTemperatureRecovery:
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

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        return self.inviscid.max_wave_speed(left, right, axis, args)

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        return self.inviscid.signal_bounds(left, right, axis, args)

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        return self.inviscid.normal_signal_bounds(left, right, normal, args)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return self.inviscid.reflect_state(state, axis)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        return self.inviscid.reflect_normal_state(state, normal)

    def admissible(self, state: Array, /) -> Array:
        return self.inviscid.admissible(state)

    def transport_properties(
        self, state: ArrayLike, args: Any = None, /
    ) -> TransportProperties:
        value = jnp.asarray(state)
        return self.transport.properties(self.temperature(value), value, args)

    def primitive_gradients(
        self, state: ArrayLike, conserved_gradient: ArrayLike, /
    ) -> Array:
        value = jnp.asarray(state)
        gradient = jnp.asarray(conserved_gradient)
        if gradient.shape != value.shape + (self.dimension,):
            raise ValueError("Two-temperature gradients must append one physical axis.")
        flat_state = value.reshape((-1, self.component_count))
        jacobian = jax.vmap(jax.jacfwd(self.inviscid.conserved_to_primitive))(flat_state)
        jacobian = jacobian.reshape(
            value.shape[:-1] + (self.component_count, self.component_count)
        )
        return contract("...pc,...cd->...pd", jacobian, gradient, backend="jax")

    def viscous_flux(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        value = jnp.asarray(state)
        primitive = self.inviscid.conserved_to_primitive(value)
        primitive_gradient = self.primitive_gradients(value, conserved_gradient)
        velocity = primitive[..., self.momentum_slice]
        velocity_gradient = primitive_gradient[..., self.momentum_slice, :]
        heavy_temperature_gradient = primitive_gradient[..., self.energy_index, :]
        properties = self.transport_properties(value, args)
        divergence = jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
        identity = jnp.eye(self.dimension, dtype=value.dtype)
        deviatoric = (
            velocity_gradient
            + jnp.swapaxes(velocity_gradient, -1, -2)
            - (2.0 / 3.0) * divergence[..., None, None] * identity
        )
        stress = (
            properties.dynamic_viscosity[..., None, None] * deviatoric
            + properties.bulk_viscosity[..., None, None]
            * divergence[..., None, None]
            * identity
        )
        mode_gradient = jnp.asarray(conserved_gradient)[..., self.mode_slice, :]
        mode_diffusion = (
            jnp.asarray(self.mode_diffusivities, dtype=value.dtype)[..., None]
            * mode_gradient
        )
        energy_flux = (
            contract("...i,...ij->...j", velocity, stress, backend="jax")
            + properties.thermal_conductivity[..., None] * heavy_temperature_gradient
            + jnp.sum(mode_diffusion, axis=-2)
        )
        species_flux = jnp.zeros(
            value.shape[:-1] + (self.species_count, self.dimension),
            dtype=value.dtype,
        )
        return jnp.concatenate(
            (
                species_flux,
                stress,
                energy_flux[..., None, :],
                mode_diffusion,
            ),
            axis=-2,
        )

    def maximum_diffusivity(self, state: Array, args: Any = None, /) -> Array:
        value = jnp.asarray(state)
        recovered = self.recover_thermodynamics(value).state
        properties = self.transport_properties(value, args)
        density = jnp.maximum(recovered.mass_density, self.density_floor)
        shear = properties.dynamic_viscosity / density
        longitudinal = (
            properties.bulk_viscosity + (4.0 / 3.0) * properties.dynamic_viscosity
        ) / density
        thermal = properties.thermal_conductivity / recovered.heavy_heat_capacity_pressure
        modal = max(self.mode_diffusivities)
        return jnp.maximum(jnp.maximum(jnp.maximum(shear, longitudinal), thermal), modal)

    def entropy_viscous_production(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        flux = self.viscous_flux(state, conserved_gradient, args)
        return jnp.sum(flux * flux, axis=(-2, -1))


__all__ = [
    "TwoTemperatureMixtureEulerSystem",
    "TwoTemperatureMixtureNavierStokesSystem",
    "TwoTemperatureRecovery",
    "TwoTemperatureThermodynamicEvaluation",
    "TwoTemperatureThermodynamicsPlan",
]
