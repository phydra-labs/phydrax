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
from ..linalg import inverse
from ._chemical_species import ChemicalPhaseKind
from ._homogeneous_thermodynamics import (
    DensityEnergyStateResult,
    HomogeneousHelmholtzPlan,
)
from ._hyperbolic_systems import (
    AbstractAdmissibleSystem,
    AbstractCharacteristicSystem,
    AbstractEntropyDiffusionSystem,
    AbstractEntropySystem,
    AbstractNormalCharacteristicSystem,
    AbstractNormalReflectionSystem,
)
from ._transport_closures import AbstractTransportClosure


@jax.custom_jvp
def _implicit_density_energy_temperature(
    temperature: Array,
    internal_energy_density: Array,
    evaluated_internal_energy_density: Array,
    volumetric_heat_capacity: Array,
    /,
) -> Array:
    del (
        internal_energy_density,
        evaluated_internal_energy_density,
        volumetric_heat_capacity,
    )
    return temperature


@_implicit_density_energy_temperature.defjvp
def _implicit_density_energy_temperature_jvp(primals, tangents):
    (
        temperature,
        internal_energy_density,
        evaluated_internal_energy_density,
        volumetric_heat_capacity,
    ) = primals
    (
        _,
        internal_energy_tangent,
        evaluated_internal_energy_tangent,
        _,
    ) = tangents
    del internal_energy_density, evaluated_internal_energy_density
    temperature_tangent = (
        internal_energy_tangent - evaluated_internal_energy_tangent
    ) / volumetric_heat_capacity
    return temperature, temperature_tangent


def _normal_frame(normal: Array, dimension: int, /) -> Array:
    value = jnp.asarray(normal)
    if value.ndim == 0 or value.shape[-1] != dimension:
        raise ValueError("normal must end in the physical dimension.")
    norm = jnp.sqrt(contract("...d,...d->...", value, value, backend="jax"))
    value = eqx.error_if(
        value,
        jnp.any(~jnp.isfinite(norm) | (norm <= 0.0)),
        "normal must be finite and nonzero.",
    )
    unit = value / norm[..., None]
    if dimension == 1:
        return unit[..., None, :]
    if dimension == 2:
        tangent = jnp.stack((-unit[..., 1], unit[..., 0]), axis=-1)
        return jnp.stack((unit, tangent), axis=-2)
    seed = jax.nn.one_hot(jnp.argmin(jnp.abs(unit), axis=-1), 3, dtype=unit.dtype)
    first = jnp.cross(seed, unit)
    first = (
        first
        / jnp.sqrt(contract("...d,...d->...", first, first, backend="jax"))[..., None]
    )
    second = jnp.cross(unit, first)
    return jnp.stack((unit, first, second), axis=-2)


class HomogeneousMixtureEulerSystem(
    AbstractCharacteristicSystem,
    AbstractAdmissibleSystem,
    AbstractEntropySystem,
    AbstractNormalReflectionSystem,
    AbstractNormalCharacteristicSystem,
):
    """Frozen-composition Euler flow driven by homogeneous gas thermodynamics.

    Conserved states are ``[rho_s..., rho*u..., rho*E]`` and primitive states
    are ``[rho_s..., u..., T]``. Formation and reference chemical energies are
    consequently part of the conserved total energy.
    """

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
        if any(
            phase is not ChemicalPhaseKind.GAS for phase in thermodynamics.schema.phases
        ):
            raise ValueError(
                "Homogeneous gas dynamics requires an all-gas species schema."
            )
        if any(
            spec.standard_pressure is None for spec in thermodynamics.schema.phase_specs
        ):
            raise ValueError(
                "Homogeneous gas dynamics requires explicit gas standard pressure."
            )
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

    @property
    def momentum_slice(self) -> slice:
        return slice(self.species_count, self.species_count + self.dimension)

    @property
    def energy_index(self) -> int:
        return self.species_count + self.dimension

    def _check_state(self, state: ArrayLike, name: str, /) -> Array:
        value = jnp.asarray(state)
        if value.ndim < 1 or value.shape[-1] != self.component_count:
            raise ValueError(f"{name} must end in {self.component_count} components.")
        return value

    def density(self, state: ArrayLike, /) -> Array:
        value = self._check_state(state, "Conserved state")
        return jnp.sum(value[..., : self.species_count], axis=-1)

    def _caloric_density_gradient(
        self, species_density: Array, temperature: Array, /
    ) -> Array:
        variables = jnp.concatenate((species_density, temperature[..., None]), axis=-1)

        def internal_energy_density(arguments):
            density = arguments[: self.species_count]
            evaluation = self.thermodynamics.evaluate_density_temperature(
                density, arguments[-1]
            )
            molar_density = jnp.sum(
                density / self.thermodynamics.schema.molar_masses.astype(arguments.dtype)
            )
            return molar_density * evaluation.molar_internal_energy

        flat = variables.reshape((-1, self.species_count + 1))
        gradient = jax.vmap(jax.grad(internal_energy_density))(flat)
        return gradient.reshape(variables.shape)

    def recover_thermodynamics(self, state: ArrayLike, /) -> DensityEnergyStateResult:
        value = self._check_state(state, "Conserved state")
        species_density = value[..., : self.species_count]
        density = jnp.sum(species_density, axis=-1)
        momentum = value[..., self.momentum_slice]
        safe_density = jnp.maximum(density, self.density_floor)
        kinetic_energy = (
            0.5
            * contract("...d,...d->...", momentum, momentum, backend="jax")
            / safe_density
        )
        internal_energy = value[..., self.energy_index] - kinetic_energy
        raw = self.thermodynamics.solve_density_energy(
            species_density,
            internal_energy,
            maximum_iterations=self.maximum_thermal_iterations,
        )
        molar_density = jnp.sum(
            species_density / self.thermodynamics.schema.molar_masses.astype(value.dtype),
            axis=-1,
        )
        evaluated_internal_energy = molar_density * raw.state.molar_internal_energy
        volumetric_heat_capacity = molar_density * raw.state.molar_heat_capacity_volume
        temperature = _implicit_density_energy_temperature(
            raw.state.temperature,
            internal_energy,
            evaluated_internal_energy,
            volumetric_heat_capacity,
        )
        thermodynamic_state = self.thermodynamics.evaluate_density_temperature(
            species_density, temperature
        )
        residual = (
            molar_density * thermodynamic_state.molar_internal_energy - internal_energy
        )
        return DensityEnergyStateResult(
            thermodynamic_state,
            species_density,
            internal_energy,
            residual,
            raw.temperature_bracket_margin,
            raw.iteration_count,
            raw.successful & thermodynamic_state.evidence.successful,
            raw.model_id,
        )

    def pressure(self, state: ArrayLike, /) -> Array:
        return self.recover_thermodynamics(state).state.pressure

    def temperature(self, state: ArrayLike, /) -> Array:
        return self.recover_thermodynamics(state).state.temperature

    def frozen_sound_speed(self, state: ArrayLike, /) -> Array:
        recovered = self.recover_thermodynamics(state)
        return jnp.sqrt(jnp.maximum(recovered.state.frozen_sound_speed_squared, 0.0))

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = self._check_state(state, "Conserved state")
        density = self.density(value)
        velocity = value[..., self.momentum_slice] / jnp.maximum(
            density[..., None], self.density_floor
        )
        return jnp.concatenate(
            (
                value[..., : self.species_count],
                velocity,
                self.temperature(value)[..., None],
            ),
            axis=-1,
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = self._check_state(primitive, "Primitive state")
        species_density = value[..., : self.species_count]
        density = jnp.sum(species_density, axis=-1)
        velocity = value[..., self.momentum_slice]
        temperature = value[..., self.energy_index]
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
        axis_value = int(axis)
        if axis_value < 0 or axis_value >= self.dimension:
            raise ValueError("axis is outside the physical dimension.")
        value = self._check_state(state, "Conserved state")
        density = self.density(value)
        momentum = value[..., self.momentum_slice]
        velocity = momentum / jnp.maximum(density[..., None], self.density_floor)
        normal_velocity = velocity[..., axis_value]
        pressure = self.pressure(value)
        species_flux = value[..., : self.species_count] * normal_velocity[..., None]
        momentum_flux = momentum * normal_velocity[..., None]
        momentum_flux = momentum_flux.at[..., axis_value].add(pressure)
        energy_flux = (value[..., self.energy_index] + pressure) * normal_velocity
        return jnp.concatenate(
            (species_flux, momentum_flux, energy_flux[..., None]), axis=-1
        )

    def _normal_state_speed(self, state: Array, normal: Array, /) -> tuple[Array, Array]:
        value = self._check_state(state, "Conserved state")
        normal_value = jnp.asarray(normal)
        density = self.density(value)
        velocity = value[..., self.momentum_slice] / jnp.maximum(
            density[..., None], self.density_floor
        )
        normal_velocity = contract(
            "...d,...d->...", velocity, normal_value, backend="jax"
        )
        normal_norm = jnp.sqrt(
            contract("...d,...d->...", normal_value, normal_value, backend="jax")
        )
        return normal_velocity, self.frozen_sound_speed(value) * normal_norm

    def max_wave_speed(
        self, left: Array, right: Array, axis: int, args: Any = None, /
    ) -> Array:
        del args
        axis_value = int(axis)
        if axis_value < 0 or axis_value >= self.dimension:
            raise ValueError("axis is outside the physical dimension.")
        left_value = self._check_state(left, "Left state")
        right_value = self._check_state(right, "Right state")
        left_speed = jnp.abs(
            left_value[..., self.species_count + axis_value]
            / jnp.maximum(self.density(left_value), self.density_floor)
        ) + self.frozen_sound_speed(left_value)
        right_speed = jnp.abs(
            right_value[..., self.species_count + axis_value]
            / jnp.maximum(self.density(right_value), self.density_floor)
        ) + self.frozen_sound_speed(right_value)
        return jnp.maximum(left_speed, right_speed)

    def signal_bounds(
        self, left: Array, right: Array, axis: int, args: Any = None, /
    ) -> tuple[Array, Array]:
        del args
        axis_value = int(axis)
        if axis_value < 0 or axis_value >= self.dimension:
            raise ValueError("axis is outside the physical dimension.")

        def bounds(state):
            value = self._check_state(state, "Interface state")
            velocity = value[..., self.species_count + axis_value] / jnp.maximum(
                self.density(value), self.density_floor
            )
            sound = self.frozen_sound_speed(value)
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
        if normal_value.ndim == 0 or normal_value.shape[-1] != self.dimension:
            raise ValueError("normal must end in the physical dimension.")
        left_velocity, left_sound = self._normal_state_speed(left, normal_value)
        right_velocity, right_sound = self._normal_state_speed(right, normal_value)
        return (
            jnp.minimum(left_velocity - left_sound, right_velocity - right_sound),
            jnp.maximum(left_velocity + left_sound, right_velocity + right_sound),
        )

    def admissible(self, state: Array, /) -> Array:
        value = self._check_state(state, "Conserved state")
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
        return (
            self._check_state(state, "Conserved state")
            .at[..., self.species_count + axis_value]
            .multiply(-1.0)
        )

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        value = self._check_state(state, "Conserved state")
        frame = _normal_frame(jnp.asarray(normal), self.dimension)
        unit = frame[..., 0, :]
        momentum = value[..., self.momentum_slice]
        normal_momentum = contract("...d,...d->...", momentum, unit, backend="jax")
        reflected = momentum - 2.0 * normal_momentum[..., None] * unit
        return value.at[..., self.momentum_slice].set(reflected)

    def _characteristic_basis_point(
        self, primitive: Array, frame: Array, normal_norm: Array, /
    ) -> tuple[Array, Array]:
        species_density = primitive[: self.species_count]
        temperature = primitive[self.energy_index]
        thermodynamic_variables = jnp.concatenate(
            (species_density, temperature[None]), axis=0
        )

        def pressure_of(arguments):
            return self.thermodynamics.evaluate_density_temperature(
                arguments[: self.species_count], arguments[-1]
            ).pressure

        pressure_gradient = jax.grad(pressure_of)(thermodynamic_variables)
        pressure_temperature = pressure_gradient[-1]
        pressure_temperature = eqx.error_if(
            pressure_temperature,
            ~jnp.isfinite(pressure_temperature)
            | (jnp.abs(pressure_temperature) <= jnp.finfo(primitive.dtype).tiny),
            "Frozen-composition characteristic basis requires nonzero dp/dT.",
        )
        thermodynamic_state = self.thermodynamics.evaluate_density_temperature(
            species_density, temperature
        )
        sound = jnp.sqrt(thermodynamic_state.frozen_sound_speed_squared)
        density = jnp.sum(species_density)
        acoustic_temperature = (
            sound**2 * density
            - contract(
                "s,s->",
                pressure_gradient[: self.species_count],
                species_density,
                backend="jax",
            )
        ) / pressure_temperature

        primitive_columns = []
        acoustic_minus = jnp.zeros_like(primitive)
        acoustic_minus = acoustic_minus.at[: self.species_count].set(species_density)
        acoustic_minus = acoustic_minus.at[self.momentum_slice].set(-sound * frame[0])
        acoustic_minus = acoustic_minus.at[self.energy_index].set(acoustic_temperature)
        primitive_columns.append(acoustic_minus)
        for species in range(self.species_count):
            contact_column = jnp.zeros_like(primitive)
            contact_column = contact_column.at[species].set(1.0)
            contact_column = contact_column.at[self.energy_index].set(
                -pressure_gradient[species] / pressure_temperature
            )
            primitive_columns.append(contact_column)
        for tangent in range(1, self.dimension):
            shear_column = jnp.zeros_like(primitive)
            shear_column = shear_column.at[self.momentum_slice].set(frame[tangent])
            primitive_columns.append(shear_column)
        acoustic_plus = jnp.zeros_like(primitive)
        acoustic_plus = acoustic_plus.at[: self.species_count].set(species_density)
        acoustic_plus = acoustic_plus.at[self.momentum_slice].set(sound * frame[0])
        acoustic_plus = acoustic_plus.at[self.energy_index].set(acoustic_temperature)
        primitive_columns.append(acoustic_plus)

        primitive_right = jnp.stack(tuple(primitive_columns), axis=-1)
        conserved_jacobian = jax.jacfwd(self.primitive_to_conserved)(primitive)
        right = contract("ij,jk->ik", conserved_jacobian, primitive_right, backend="jax")
        normal_velocity = contract(
            "d,d->", primitive[self.momentum_slice], frame[0], backend="jax"
        )
        convective = normal_norm * normal_velocity
        speeds = jnp.stack(
            (
                convective - normal_norm * sound,
                *(convective for _ in range(self.species_count + self.dimension - 1)),
                convective + normal_norm * sound,
            )
        )
        return right, speeds

    def normal_eigensystem(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        del args
        left_value = self._check_state(left, "Left state")
        right_value = self._check_state(right, "Right state")
        if left_value.shape != right_value.shape:
            raise ValueError("Characteristic states must have matching shapes.")
        normal_value = jnp.asarray(normal, dtype=left_value.dtype)
        expected_normal_shape = left_value.shape[:-1] + (self.dimension,)
        if normal_value.shape != expected_normal_shape:
            normal_value = jnp.broadcast_to(normal_value, expected_normal_shape)
        frame = _normal_frame(normal_value, self.dimension)
        normal_norm = jnp.sqrt(
            contract("...d,...d->...", normal_value, normal_value, backend="jax")
        )
        primitive = 0.5 * (
            self.conserved_to_primitive(left_value)
            + self.conserved_to_primitive(right_value)
        )
        flat_primitive = primitive.reshape((-1, self.component_count))
        flat_frame = frame.reshape((-1, self.dimension, self.dimension))
        flat_norm = normal_norm.reshape((-1,))
        right_matrix, speeds = jax.vmap(self._characteristic_basis_point)(
            flat_primitive, flat_frame, flat_norm
        )
        right_matrix = right_matrix.reshape(
            primitive.shape[:-1] + (self.component_count, self.component_count)
        )
        speeds = speeds.reshape(primitive.shape)
        inverse_result = inverse(right_matrix)
        left_matrix = eqx.error_if(
            inverse_result.value,
            jnp.any(~inverse_result.successful),
            "Homogeneous-mixture characteristic basis is singular.",
        )
        return left_matrix, right_matrix, speeds

    def eigensystem(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        axis_value = int(axis)
        if axis_value < 0 or axis_value >= self.dimension:
            raise ValueError("axis is outside the physical dimension.")
        normal = jax.nn.one_hot(axis_value, self.dimension, dtype=jnp.asarray(left).dtype)
        normal = jnp.broadcast_to(
            normal, jnp.asarray(left).shape[:-1] + (self.dimension,)
        )
        return self.normal_eigensystem(left, right, normal, args)

    def entropy(self, state: ArrayLike, /) -> Array:
        recovered = self.recover_thermodynamics(state)
        return -recovered.state.molar_density * recovered.state.molar_entropy

    def entropy_flux(self, state: ArrayLike, axis: int, /) -> Array:
        value = self._check_state(state, "Conserved state")
        velocity = value[..., self.species_count + int(axis)] / jnp.maximum(
            self.density(value), self.density_floor
        )
        return self.entropy(value) * velocity

    def normal_entropy_flux(self, state: ArrayLike, normal: ArrayLike, /) -> Array:
        value = self._check_state(state, "Conserved state")
        velocity = value[..., self.momentum_slice] / jnp.maximum(
            self.density(value)[..., None], self.density_floor
        )
        return self.entropy(value) * contract(
            "...d,...d->...", velocity, jnp.asarray(normal), backend="jax"
        )

    def entropy_evidence(self, state: ArrayLike, /) -> Array:
        recovered = self.recover_thermodynamics(state)
        chemical = self.thermodynamics.evaluate_chemical(
            recovered.state.temperature,
            recovered.state.molar_density,
            recovered.state.mole_fraction,
        )
        return (
            recovered.successful
            & chemical.successful
            & jnp.all(recovered.state.mole_fraction > 0.0, axis=-1)
        )

    def entropy_variables(self, state: Array, /) -> Array:
        value = self._check_state(state, "Conserved state")
        recovered = self.recover_thermodynamics(value)
        thermodynamic_state = recovered.state
        chemical = self.thermodynamics.evaluate_chemical(
            thermodynamic_state.temperature,
            thermodynamic_state.molar_density,
            thermodynamic_state.mole_fraction,
        )
        temperature = thermodynamic_state.temperature
        density = self.density(value)
        velocity = value[..., self.momentum_slice] / jnp.maximum(
            density[..., None], self.density_floor
        )
        speed_squared = contract("...d,...d->...", velocity, velocity, backend="jax")
        species_variables = (
            chemical.chemical_potential
            / self.thermodynamics.schema.molar_masses.astype(value.dtype)
            - 0.5 * speed_squared[..., None]
        ) / temperature[..., None]
        variables = jnp.concatenate(
            (
                species_variables,
                velocity / temperature[..., None],
                (-1.0 / temperature)[..., None],
            ),
            axis=-1,
        )
        return eqx.error_if(
            variables,
            jnp.any(
                ~(
                    recovered.successful
                    & chemical.successful
                    & jnp.all(thermodynamic_state.mole_fraction > 0.0, axis=-1)
                )
            ),
            "Entropy variables require successful stable homogeneous gas evidence.",
        )


class HomogeneousMixtureCompressibleNavierStokesSystem(
    AbstractCharacteristicSystem,
    AbstractAdmissibleSystem,
    AbstractEntropySystem,
    AbstractNormalReflectionSystem,
    AbstractNormalCharacteristicSystem,
    AbstractEntropyDiffusionSystem,
):
    """All-species compressible Navier–Stokes system with canonical calorics.

    With ``species_diffusivities=None`` species are frozen and their diffusive
    flux is exactly zero. Supplied nonnegative Fick diffusivities are corrected
    by the local mass fractions so their diffusive species flux sums exactly to
    zero; the energy flux carries the matching partial-specific-enthalpy flux.
    """

    inviscid: HomogeneousMixtureEulerSystem
    thermodynamics: HomogeneousHelmholtzPlan
    transport: AbstractTransportClosure
    species_diffusivities: tuple[float, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: HomogeneousHelmholtzPlan,
        transport: AbstractTransportClosure,
        dimension: int = 1,
        /,
        *,
        species_diffusivities: ArrayLike | None = None,
        density_floor: float = 1.0e-12,
        pressure_floor: float = 1.0e-12,
        maximum_thermal_iterations: int = 80,
    ) -> None:
        if not isinstance(transport, AbstractTransportClosure):
            raise TypeError("transport must be an AbstractTransportClosure.")
        inviscid = HomogeneousMixtureEulerSystem(
            thermodynamics,
            dimension,
            density_floor=density_floor,
            pressure_floor=pressure_floor,
            maximum_thermal_iterations=maximum_thermal_iterations,
        )
        diffusivities = None
        if species_diffusivities is not None:
            values = np.asarray(species_diffusivities, dtype=float)
            if (
                values.shape != (inviscid.species_count,)
                or np.any(~np.isfinite(values))
                or np.any(values < 0.0)
            ):
                raise ValueError(
                    "species_diffusivities must contain one finite nonnegative value per species."
                )
            diffusivities = tuple(float(value) for value in values)
        self.inviscid = inviscid
        self.thermodynamics = thermodynamics
        self.transport = transport
        self.species_diffusivities = diffusivities
        self.dimension = inviscid.dimension
        self.component_names = inviscid.component_names
        self.system_id = canonical_fingerprint(
            {
                "kind": "homogeneous-mixture-compressible-navier-stokes",
                "inviscid": inviscid.system_id,
                "transport": transport.closure_id,
                "species_diffusivities": diffusivities,
            }
        )

    @property
    def species_count(self) -> int:
        return self.inviscid.species_count

    @property
    def momentum_slice(self) -> slice:
        return self.inviscid.momentum_slice

    @property
    def energy_index(self) -> int:
        return self.inviscid.energy_index

    @property
    def density_floor(self) -> float:
        return self.inviscid.density_floor

    @property
    def pressure_floor(self) -> float:
        return self.inviscid.pressure_floor

    def density(self, state: ArrayLike, /) -> Array:
        return self.inviscid.density(state)

    def recover_thermodynamics(self, state: ArrayLike, /) -> DensityEnergyStateResult:
        return self.inviscid.recover_thermodynamics(state)

    def pressure(self, state: ArrayLike, /) -> Array:
        return self.inviscid.pressure(state)

    def temperature(self, state: ArrayLike, /) -> Array:
        return self.inviscid.temperature(state)

    def conserved_to_primitive(self, state: Array, /) -> Array:
        return self.inviscid.conserved_to_primitive(state)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return self.inviscid.primitive_to_conserved(primitive)

    def primitive_gradients(
        self, state: ArrayLike, conserved_gradient: ArrayLike, /
    ) -> tuple[Array, Array, Array]:
        value = self.inviscid._check_state(state, "Conserved state")
        gradient = jnp.asarray(conserved_gradient)
        if gradient.shape != value.shape + (self.dimension,):
            raise ValueError(
                "Conserved gradients must append one physical derivative axis."
            )
        density = self.density(value)
        momentum = value[..., self.momentum_slice]
        velocity = momentum / jnp.maximum(density[..., None], self.density_floor)
        species_gradient = gradient[..., : self.species_count, :]
        density_gradient = jnp.sum(species_gradient, axis=-2)
        momentum_gradient = gradient[..., self.momentum_slice, :]
        velocity_gradient = (
            momentum_gradient - velocity[..., :, None] * density_gradient[..., None, :]
        ) / jnp.maximum(density[..., None, None], self.density_floor)
        energy_gradient = gradient[..., self.energy_index, :]
        speed_squared = contract("...i,...i->...", velocity, velocity, backend="jax")
        kinetic_gradient = (
            contract("...i,...ij->...j", velocity, momentum_gradient, backend="jax")
            - 0.5 * speed_squared[..., None] * density_gradient
        )
        internal_gradient = energy_gradient - kinetic_gradient
        primitive = self.conserved_to_primitive(value)
        caloric_gradient = self.inviscid._caloric_density_gradient(
            primitive[..., : self.species_count], primitive[..., -1]
        )
        volumetric_cv = caloric_gradient[..., -1]
        temperature_gradient = (
            internal_gradient
            - contract(
                "...s,...sd->...d",
                caloric_gradient[..., : self.species_count],
                species_gradient,
                backend="jax",
            )
        ) / volumetric_cv[..., None]
        temperature_gradient = eqx.error_if(
            temperature_gradient,
            jnp.any(~jnp.isfinite(volumetric_cv) | (volumetric_cv <= 0.0)),
            "Primitive gradients require positive volumetric heat capacity.",
        )
        return velocity_gradient, temperature_gradient, species_gradient

    def partial_specific_enthalpies(self, state: Array, /) -> Array:
        evaluation = self.recover_thermodynamics(state).state
        flat_temperature = evaluation.temperature.reshape((-1,))
        flat_density = evaluation.molar_density.reshape((-1,))
        flat_composition = evaluation.mole_fraction.reshape((-1, self.species_count))

        def point(temperature, molar_density, composition):
            def chemical_at(temperature_value, density_value):
                return self.thermodynamics.evaluate_chemical(
                    temperature_value, density_value, composition
                ).chemical_potential

            chemical = chemical_at(temperature, molar_density)
            chemical_temperature = jax.jacfwd(chemical_at, argnums=0)(
                temperature, molar_density
            )
            chemical_density = jax.jacfwd(chemical_at, argnums=1)(
                temperature, molar_density
            )
            thermo = self.thermodynamics.evaluate(temperature, molar_density, composition)
            constant_pressure_temperature = chemical_temperature - (
                chemical_density
                * thermo.pressure_temperature_derivative
                / thermo.pressure_molar_density_derivative
            )
            partial_molar_enthalpy = (
                chemical - temperature * constant_pressure_temperature
            )
            return (
                partial_molar_enthalpy
                / self.thermodynamics.schema.molar_masses.astype(temperature.dtype)
            )

        enthalpy = jax.vmap(point)(flat_temperature, flat_density, flat_composition)
        return enthalpy.reshape(evaluation.temperature.shape + (self.species_count,))

    def _species_diffusive_flux(self, state: Array, species_gradient: Array, /) -> Array:
        shape = state.shape[:-1] + (self.species_count, self.dimension)
        if self.species_diffusivities is None:
            return jnp.zeros(shape, dtype=state.dtype)
        species_density = state[..., : self.species_count]
        density = self.density(state)
        mass_fraction = species_density / jnp.maximum(
            density[..., None], self.density_floor
        )
        density_gradient = jnp.sum(species_gradient, axis=-2)
        mass_fraction_gradient = (
            species_gradient
            - mass_fraction[..., :, None] * density_gradient[..., None, :]
        ) / jnp.maximum(density[..., None, None], self.density_floor)
        diffusivity = jnp.asarray(self.species_diffusivities, dtype=state.dtype)
        raw = density[..., None, None] * diffusivity[:, None] * mass_fraction_gradient
        return raw - mass_fraction[..., :, None] * jnp.sum(raw, axis=-2)[..., None, :]

    def viscous_flux(
        self,
        state: ArrayLike,
        conserved_gradient: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        value = self.inviscid._check_state(state, "Conserved state")
        velocity = value[..., self.momentum_slice] / jnp.maximum(
            self.density(value)[..., None], self.density_floor
        )
        velocity_gradient, temperature_gradient, species_gradient = (
            self.primitive_gradients(value, conserved_gradient)
        )
        properties = self.transport.properties(self.temperature(value), value, args)
        species_flux = self._species_diffusive_flux(value, species_gradient)
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
        energy_flux = (
            contract("...i,...ij->...j", velocity, stress, backend="jax")
            + properties.thermal_conductivity[..., None] * temperature_gradient
        )
        if self.species_diffusivities is not None:
            enthalpy = self.partial_specific_enthalpies(value)
            energy_flux = energy_flux + contract(
                "...s,...sd->...d", enthalpy, species_flux, backend="jax"
            )
        return jnp.concatenate((species_flux, stress, energy_flux[..., None, :]), axis=-2)

    def viscous_flux_from_primitive_gradients(
        self,
        velocity: ArrayLike,
        velocity_gradient: ArrayLike,
        temperature_gradient: ArrayLike,
        dynamic_viscosity: ArrayLike,
        bulk_viscosity: ArrayLike,
        thermal_conductivity: ArrayLike,
        /,
    ) -> Array:
        if self.species_diffusivities is not None:
            raise ValueError(
                "Supplied species diffusion requires conserved species gradients."
            )
        velocity_value = jnp.asarray(velocity)
        gradient = jnp.asarray(velocity_gradient)
        temperature_gradient_value = jnp.asarray(temperature_gradient)
        viscosity = jnp.asarray(dynamic_viscosity)
        bulk = jnp.asarray(bulk_viscosity)
        conductivity = jnp.asarray(thermal_conductivity)
        coefficient_shape = velocity_value.shape[:-1]
        if (
            velocity_value.shape[-1] != self.dimension
            or gradient.shape != velocity_value.shape + (self.dimension,)
            or temperature_gradient_value.shape != velocity_value.shape
            or viscosity.shape != coefficient_shape
            or bulk.shape != coefficient_shape
            or conductivity.shape != coefficient_shape
        ):
            raise ValueError(
                "Primitive gradients and transport coefficients are incompatible."
            )
        divergence = jnp.trace(gradient, axis1=-2, axis2=-1)
        identity = jnp.eye(self.dimension, dtype=velocity_value.dtype)
        deviatoric = (
            gradient
            + jnp.swapaxes(gradient, -1, -2)
            - (2.0 / 3.0) * divergence[..., None, None] * identity
        )
        stress = viscosity[..., None, None] * deviatoric + (
            bulk[..., None, None] * divergence[..., None, None] * identity
        )
        energy_flux = (
            contract("...i,...ij->...j", velocity_value, stress, backend="jax")
            + conductivity[..., None] * temperature_gradient_value
        )
        species_flux = jnp.zeros(
            coefficient_shape + (self.species_count, self.dimension),
            dtype=velocity_value.dtype,
        )
        return jnp.concatenate((species_flux, stress, energy_flux[..., None, :]), axis=-2)

    def maximum_diffusivity(self, state: Array, args: Any = None, /) -> Array:
        value = self.inviscid._check_state(state, "Conserved state")
        recovered = self.recover_thermodynamics(value)
        properties = self.transport.properties(recovered.state.temperature, value, args)
        density = self.density(value)
        volumetric_cp = (
            recovered.state.molar_density * recovered.state.molar_heat_capacity_pressure
        )
        result = jnp.maximum(
            properties.dynamic_viscosity / jnp.maximum(density, self.density_floor),
            properties.thermal_conductivity / volumetric_cp,
        )
        if self.species_diffusivities is not None:
            result = jnp.maximum(result, max(self.species_diffusivities))
        return result

    def entropy_viscous_production(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        value = jnp.asarray(state)
        gradient = jnp.asarray(conserved_gradient)
        flat_state = value.reshape((-1, value.shape[-1]))
        flat_gradient = gradient.reshape((-1, gradient.shape[-2], gradient.shape[-1]))
        hessian = jax.vmap(jax.jacfwd(self.entropy_variables))(flat_state)
        entropy_gradient = contract("nij,njd->nid", hessian, flat_gradient, backend="jax")
        viscous_flux = self.viscous_flux(value, gradient, args).reshape(
            entropy_gradient.shape
        )
        production = contract("nid,nid->n", entropy_gradient, viscous_flux, backend="jax")
        return production.reshape(value.shape[:-1])

    def viscous_normal_flux(
        self,
        state: ArrayLike,
        conserved_gradient: ArrayLike,
        normal: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        flux = self.viscous_flux(state, conserved_gradient, args)
        normal_value = jnp.asarray(normal)
        if normal_value.shape != flux.shape[:-2] + (self.dimension,):
            raise ValueError("Viscous normal shape is incompatible with the state.")
        return contract("...ij,...j->...i", flux, normal_value, backend="jax")

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        return self.inviscid.physical_flux(state, axis, args)

    def max_wave_speed(
        self, left: Array, right: Array, axis: int, args: Any = None, /
    ) -> Array:
        return self.inviscid.max_wave_speed(left, right, axis, args)

    def signal_bounds(
        self, left: Array, right: Array, axis: int, args: Any = None, /
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

    def admissible(self, state: Array, /) -> Array:
        return self.inviscid.admissible(state)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return self.inviscid.reflect_state(state, axis)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        return self.inviscid.reflect_normal_state(state, normal)

    def eigensystem(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        return self.inviscid.eigensystem(left, right, axis, args)

    def normal_eigensystem(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array]:
        return self.inviscid.normal_eigensystem(left, right, normal, args)

    def entropy(self, state: ArrayLike, /) -> Array:
        return self.inviscid.entropy(state)

    def entropy_flux(self, state: ArrayLike, axis: int, /) -> Array:
        return self.inviscid.entropy_flux(state, axis)

    def normal_entropy_flux(self, state: ArrayLike, normal: ArrayLike, /) -> Array:
        return self.inviscid.normal_entropy_flux(state, normal)

    def entropy_evidence(self, state: ArrayLike, /) -> Array:
        return self.inviscid.entropy_evidence(state)

    def entropy_variables(self, state: Array, /) -> Array:
        return self.inviscid.entropy_variables(state)


__all__ = [
    "HomogeneousMixtureCompressibleNavierStokesSystem",
    "HomogeneousMixtureEulerSystem",
]
