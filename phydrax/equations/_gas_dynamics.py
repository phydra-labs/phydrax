#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from ._homogeneous_thermodynamics import (
    DensityEnergyStateResult,
    HomogeneousHelmholtzPlan,
)
from ._hyperbolic_systems import (
    AbstractAdmissibleSystem,
    AbstractNormalReflectionSystem,
)


class HomogeneousMixtureEulerSystem(
    AbstractAdmissibleSystem,
    AbstractNormalReflectionSystem,
):
    """Frozen-composition Euler flow driven by homogeneous phase thermodynamics."""

    thermodynamics: HomogeneousHelmholtzPlan
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    maximum_thermal_iterations: int = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: HomogeneousHelmholtzPlan,
        dimension: int = 1,
        /,
        *,
        density_floor: float = 1.0e-12,
        pressure_floor: float = 1.0e-12,
        maximum_thermal_iterations: int = 80,
    ) -> None:
        if not isinstance(thermodynamics, HomogeneousHelmholtzPlan):
            raise TypeError("thermodynamics must be HomogeneousHelmholtzPlan.")
        dimension_value = int(dimension)
        iterations = int(maximum_thermal_iterations)
        if dimension_value not in (1, 2, 3):
            raise ValueError("Euler dimension must be one, two, or three.")
        if not np.isfinite(density_floor) or density_floor <= 0.0:
            raise ValueError("density_floor must be finite and positive.")
        if not np.isfinite(pressure_floor) or pressure_floor <= 0.0:
            raise ValueError("pressure_floor must be finite and positive.")
        if iterations <= 0:
            raise ValueError("maximum_thermal_iterations must be positive.")
        self.thermodynamics = thermodynamics
        self.dimension = dimension_value
        self.density_floor = float(density_floor)
        self.pressure_floor = float(pressure_floor)
        self.maximum_thermal_iterations = iterations
        self.component_names = (
            *(f"species_density_{name}" for name in thermodynamics.schema.species_names),
            *(f"momentum_{axis}" for axis in range(dimension_value)),
            "total_energy",
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "homogeneous-mixture-euler",
                "thermodynamics": thermodynamics.model_id,
                "dimension": dimension_value,
                "density_floor": self.density_floor,
                "pressure_floor": self.pressure_floor,
                "maximum_thermal_iterations": iterations,
            }
        )

    @property
    def species_count(self) -> int:
        return self.thermodynamics.schema.species_count

    def density(self, state: Array, /) -> Array:
        return jnp.sum(jnp.asarray(state)[..., : self.species_count], axis=-1)

    def recover_thermodynamics(self, state: Array, /) -> DensityEnergyStateResult:
        value = jnp.asarray(state)
        species_density = value[..., : self.species_count]
        density = jnp.sum(species_density, axis=-1)
        momentum = value[..., self.species_count : -1]
        safe_density = jnp.maximum(density, self.density_floor)
        kinetic_energy = (
            0.5
            * contract("...d,...d->...", momentum, momentum, backend="jax")
            / safe_density
        )
        internal_energy = value[..., -1] - kinetic_energy
        return self.thermodynamics.solve_density_energy(
            species_density,
            internal_energy,
            maximum_iterations=self.maximum_thermal_iterations,
        )

    def pressure(self, state: Array, /) -> Array:
        return self.recover_thermodynamics(state).state.pressure

    def temperature(self, state: Array, /) -> Array:
        return self.recover_thermodynamics(state).state.temperature

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        density = self.density(value)
        velocity = value[..., self.species_count : -1] / jnp.maximum(
            density[..., None], self.density_floor
        )
        temperature = self.temperature(value)
        return jnp.concatenate(
            (value[..., : self.species_count], velocity, temperature[..., None]),
            axis=-1,
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        species_density = value[..., : self.species_count]
        density = jnp.sum(species_density, axis=-1)
        velocity = value[..., self.species_count : -1]
        temperature = value[..., -1]
        thermodynamics = self.thermodynamics.evaluate_density_temperature(
            species_density, temperature
        )
        molar_density = jnp.sum(
            species_density / self.thermodynamics.schema.molar_masses.astype(value.dtype),
            axis=-1,
        )
        internal_energy = molar_density * thermodynamics.molar_internal_energy
        kinetic_energy = (
            0.5 * density * contract("...d,...d->...", velocity, velocity, backend="jax")
        )
        return jnp.concatenate(
            (
                species_density,
                density[..., None] * velocity,
                (internal_energy + kinetic_energy)[..., None],
            ),
            axis=-1,
        )

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        value = jnp.asarray(state)
        axis_value = int(axis)
        density = self.density(value)
        momentum = value[..., self.species_count : -1]
        velocity = momentum / jnp.maximum(density[..., None], self.density_floor)
        normal_velocity = velocity[..., axis_value]
        pressure = self.pressure(value)
        species_flux = value[..., : self.species_count] * normal_velocity[..., None]
        momentum_flux = momentum * normal_velocity[..., None]
        momentum_flux = momentum_flux.at[..., axis_value].add(pressure)
        energy_flux = (value[..., -1] + pressure) * normal_velocity
        return jnp.concatenate(
            (species_flux, momentum_flux, energy_flux[..., None]), axis=-1
        )

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        del args
        axis_value = int(axis)

        def speed(state):
            value = jnp.asarray(state)
            density = self.density(value)
            velocity = value[..., self.species_count + axis_value] / jnp.maximum(
                density, self.density_floor
            )
            sound = jnp.sqrt(
                jnp.maximum(
                    self.recover_thermodynamics(value).state.frozen_sound_speed_squared,
                    0.0,
                )
            )
            return jnp.abs(velocity) + sound

        return jnp.maximum(speed(left), speed(right))

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        axis_value = int(axis)

        def bounds(state):
            value = jnp.asarray(state)
            density = self.density(value)
            velocity = value[..., self.species_count + axis_value] / jnp.maximum(
                density, self.density_floor
            )
            sound = jnp.sqrt(
                jnp.maximum(
                    self.recover_thermodynamics(value).state.frozen_sound_speed_squared,
                    0.0,
                )
            )
            return velocity - sound, velocity + sound

        left_lower, left_upper = bounds(left)
        right_lower, right_upper = bounds(right)
        return jnp.minimum(left_lower, right_lower), jnp.maximum(left_upper, right_upper)

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        normal_value = jnp.asarray(normal)
        if normal_value.shape[-1] != self.dimension:
            raise ValueError("normal must end in the physical dimension.")

        def bounds(state):
            value = jnp.asarray(state)
            density = self.density(value)
            velocity = value[..., self.species_count : -1] / jnp.maximum(
                density[..., None], self.density_floor
            )
            normal_velocity = contract(
                "...d,...d->...", velocity, normal_value, backend="jax"
            )
            sound = jnp.sqrt(
                jnp.maximum(
                    self.recover_thermodynamics(value).state.frozen_sound_speed_squared,
                    0.0,
                )
            )
            normal_norm = jnp.sqrt(
                contract("...d,...d->...", normal_value, normal_value, backend="jax")
            )
            return (
                normal_velocity - sound * normal_norm,
                normal_velocity + sound * normal_norm,
            )

        left_lower, left_upper = bounds(left)
        right_lower, right_upper = bounds(right)
        return jnp.minimum(left_lower, right_lower), jnp.maximum(left_upper, right_upper)

    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        species_density = value[..., : self.species_count]
        density = jnp.sum(species_density, axis=-1)
        recovered = self.recover_thermodynamics(value)
        return (
            jnp.all(jnp.isfinite(value), axis=-1)
            & jnp.all(species_density >= 0.0, axis=-1)
            & (density >= self.density_floor)
            & recovered.successful
            & (recovered.state.pressure >= self.pressure_floor)
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        axis_value = int(axis)
        if axis_value < 0 or axis_value >= self.dimension:
            raise ValueError("axis is outside the physical dimension.")
        return jnp.asarray(state).at[..., self.species_count + axis_value].multiply(-1.0)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        value = jnp.asarray(state)
        normal_value = jnp.asarray(normal)
        if normal_value.shape[-1] != self.dimension:
            raise ValueError("normal must end in the physical dimension.")
        norm_squared = contract(
            "...d,...d->...", normal_value, normal_value, backend="jax"
        )
        unit = normal_value / jnp.sqrt(norm_squared)[..., None]
        momentum = value[..., self.species_count : -1]
        normal_momentum = contract("...d,...d->...", momentum, unit, backend="jax")
        reflected = momentum - 2.0 * normal_momentum[..., None] * unit
        return value.at[..., self.species_count : -1].set(reflected)


__all__ = ["HomogeneousMixtureEulerSystem"]
