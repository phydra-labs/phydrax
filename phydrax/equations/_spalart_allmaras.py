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
from ._gas_dynamics import HomogeneousMixtureCompressibleNavierStokesSystem
from ._hyperbolic_systems import (
    AbstractAdmissibleSystem,
    AbstractEntropyDiffusionSystem,
    AbstractNormalReflectionSystem,
    ConservationDiffusionEvaluation,
)


def _signed_floor(value: Array, floor: Array, /) -> Array:
    sign = jnp.where(value < 0.0, -1.0, 1.0)
    return jnp.where(jnp.abs(value) < floor, sign * floor, value)


class SpalartAllmarasArguments(StrictModule):
    """Runtime wall distance and base-transport arguments."""

    wall_distance: Array
    transport_args: Any

    def __init__(self, wall_distance: ArrayLike, transport_args: Any = None, /):
        self.wall_distance = jnp.asarray(wall_distance)
        self.transport_args = transport_args


class SpalartAllmarasEvaluation(StrictModule):
    eddy_viscosity: Array
    modified_strain: Array
    production: Array
    destruction: Array
    cross_diffusion: Array
    source: Array
    diffusion_coefficient: Array
    working_variable_ratio: Array
    finite: Array
    successful: Array


class SpalartAllmarasNegativePlan(StrictModule, NonTrainableState):
    """NASA-TMR SA-neg-noft2 closure with explicit model identity."""

    sigma: float = eqx.field(static=True)
    cb1: float = eqx.field(static=True)
    cb2: float = eqx.field(static=True)
    kappa: float = eqx.field(static=True)
    cw1: float = eqx.field(static=True)
    cw2: float = eqx.field(static=True)
    cw3: float = eqx.field(static=True)
    cv1: float = eqx.field(static=True)
    cv2: float = eqx.field(static=True)
    cv3: float = eqx.field(static=True)
    ct3: float = eqx.field(static=True)
    cn1: float = eqx.field(static=True)
    turbulent_prandtl: float = eqx.field(static=True)
    minimum_working_ratio: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        sigma: float = 2.0 / 3.0,
        cb1: float = 0.1355,
        cb2: float = 0.622,
        kappa: float = 0.41,
        cw2: float = 0.3,
        cw3: float = 2.0,
        cv1: float = 7.1,
        cv2: float = 0.7,
        cv3: float = 0.9,
        ct3: float = 1.2,
        cn1: float = 16.0,
        turbulent_prandtl: float = 0.9,
        minimum_working_ratio: float = -0.9,
    ):
        values = tuple(
            float(value)
            for value in (
                sigma,
                cb1,
                cb2,
                kappa,
                cw2,
                cw3,
                cv1,
                cv2,
                cv3,
                ct3,
                cn1,
                turbulent_prandtl,
                minimum_working_ratio,
            )
        )
        if (
            any(not np.isfinite(value) for value in values)
            or min(values[:12]) <= 0.0
            or not -1.0 < values[12] < 0.0
        ):
            raise ValueError("SA-neg constants or admissibility ratio are invalid.")
        self.sigma = values[0]
        self.cb1 = values[1]
        self.cb2 = values[2]
        self.kappa = values[3]
        self.cw2 = values[4]
        self.cw3 = values[5]
        self.cv1 = values[6]
        self.cv2 = values[7]
        self.cv3 = values[8]
        self.ct3 = values[9]
        self.cn1 = values[10]
        self.turbulent_prandtl = values[11]
        self.minimum_working_ratio = values[12]
        self.cw1 = self.cb1 / self.kappa**2 + (1.0 + self.cb2) / self.sigma
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spalart-allmaras-negative-noft2",
                "sigma": self.sigma,
                "cb1": self.cb1,
                "cb2": self.cb2,
                "kappa": self.kappa,
                "cw1": self.cw1,
                "cw2": self.cw2,
                "cw3": self.cw3,
                "cv1": self.cv1,
                "cv2": self.cv2,
                "cv3": self.cv3,
                "ct3": self.ct3,
                "cn1": self.cn1,
                "turbulent_prandtl": self.turbulent_prandtl,
                "minimum_working_ratio": self.minimum_working_ratio,
            }
        )

    def diffusion_coefficient(
        self,
        molecular_kinematic_viscosity: ArrayLike,
        working_variable: ArrayLike,
        /,
    ) -> Array:
        molecular = jnp.asarray(molecular_kinematic_viscosity)
        working = jnp.asarray(working_variable, dtype=molecular.dtype)
        safe = jnp.maximum(molecular, jnp.finfo(molecular.dtype).tiny)
        ratio = working / safe
        ratio_cubed = ratio**3
        negative_factor = (self.cn1 + ratio_cubed) / _signed_floor(
            self.cn1 - ratio_cubed,
            jnp.asarray(jnp.finfo(molecular.dtype).eps, dtype=molecular.dtype),
        )
        return molecular + jnp.where(working >= 0.0, working, negative_factor * working)

    def evaluate(
        self,
        density: ArrayLike,
        molecular_kinematic_viscosity: ArrayLike,
        working_variable: ArrayLike,
        velocity_gradient: ArrayLike,
        working_variable_gradient: ArrayLike,
        wall_distance: ArrayLike,
        /,
    ) -> SpalartAllmarasEvaluation:
        density_ = jnp.asarray(density)
        molecular = jnp.asarray(molecular_kinematic_viscosity, dtype=density_.dtype)
        working = jnp.asarray(working_variable, dtype=density_.dtype)
        velocity_gradient_ = jnp.asarray(velocity_gradient, dtype=density_.dtype)
        gradient = jnp.asarray(working_variable_gradient, dtype=density_.dtype)
        distance = jnp.asarray(wall_distance, dtype=density_.dtype)
        cell_shape = density_.shape
        dimension = velocity_gradient_.shape[-1]
        if (
            molecular.shape != cell_shape
            or working.shape != cell_shape
            or distance.shape not in ((), cell_shape)
            or velocity_gradient_.shape != cell_shape + (dimension, dimension)
            or gradient.shape != cell_shape + (dimension,)
        ):
            raise ValueError("SA-neg inputs have incompatible cell shapes.")
        distance = jnp.broadcast_to(distance, cell_shape)
        tiny = jnp.asarray(jnp.finfo(density_.dtype).tiny, dtype=density_.dtype)
        epsilon = jnp.asarray(jnp.finfo(density_.dtype).eps, dtype=density_.dtype)
        safe_molecular = jnp.maximum(molecular, tiny)
        safe_distance_squared = jnp.maximum(distance * distance, tiny)
        ratio = working / safe_molecular
        ratio_cubed = ratio**3
        fv1 = ratio_cubed / _signed_floor(ratio_cubed + self.cv1**3, epsilon)
        eddy_kinematic = jnp.where(working > 0.0, working * fv1, 0.0)
        eddy_viscosity = density_ * jnp.maximum(eddy_kinematic, 0.0)
        rotation = 0.5 * (velocity_gradient_ - jnp.swapaxes(velocity_gradient_, -1, -2))
        strain = jnp.sqrt(
            jnp.maximum(2.0 * jnp.sum(rotation * rotation, axis=(-2, -1)), 0.0)
        )
        fv2 = 1.0 - ratio / _signed_floor(1.0 + ratio * fv1, epsilon)
        s_bar = working * fv2 / (self.kappa**2 * safe_distance_squared)
        ordinary = strain + s_bar
        denominator = (self.cv3 - 2.0 * self.cv2) * strain - s_bar
        corrected = strain + strain * (
            self.cv2**2 * strain + self.cv3 * s_bar
        ) / _signed_floor(denominator, epsilon)
        modified_strain = jnp.where(s_bar >= -self.cv2 * strain, ordinary, corrected)
        modified_strain = jnp.maximum(modified_strain, tiny)
        r = jnp.clip(
            working / (modified_strain * self.kappa**2 * safe_distance_squared),
            0.0,
            10.0,
        )
        g = r + self.cw2 * (r**6 - r)
        fw = g * ((1.0 + self.cw3**6) / (g**6 + self.cw3**6)) ** (1.0 / 6.0)
        positive = working >= 0.0
        production = jnp.where(
            positive,
            self.cb1 * modified_strain * working,
            self.cb1 * (1.0 - self.ct3) * strain * working,
        )
        destruction = jnp.where(
            positive,
            self.cw1 * fw * (working / jnp.maximum(distance, tiny)) ** 2,
            -self.cw1 * (working / jnp.maximum(distance, tiny)) ** 2,
        )
        cross_diffusion = self.cb2 / self.sigma * jnp.sum(gradient * gradient, axis=-1)
        source = production - destruction + cross_diffusion
        diffusion = self.diffusion_coefficient(molecular, working)
        finite = (
            jnp.isfinite(density_)
            & jnp.isfinite(molecular)
            & jnp.isfinite(working)
            & jnp.isfinite(distance)
            & jnp.isfinite(eddy_viscosity)
            & jnp.isfinite(modified_strain)
            & jnp.isfinite(source)
            & jnp.isfinite(diffusion)
            & jnp.all(jnp.isfinite(velocity_gradient_), axis=(-2, -1))
            & jnp.all(jnp.isfinite(gradient), axis=-1)
        )
        successful = (
            finite
            & (density_ > 0.0)
            & (molecular > 0.0)
            & (distance > 0.0)
            & (ratio >= self.minimum_working_ratio)
            & (diffusion > 0.0)
        )
        return SpalartAllmarasEvaluation(
            eddy_viscosity,
            modified_strain,
            production,
            destruction,
            cross_diffusion,
            source,
            diffusion,
            ratio,
            finite,
            successful,
        )


class SpalartAllmarasCompressibleSystem(
    AbstractAdmissibleSystem,
    AbstractNormalReflectionSystem,
    AbstractEntropyDiffusionSystem,
):
    """Canonical homogeneous-mixture Navier-Stokes plus one SA-neg density."""

    base: HomogeneousMixtureCompressibleNavierStokesSystem
    model: SpalartAllmarasNegativePlan

    def __init__(
        self,
        base: HomogeneousMixtureCompressibleNavierStokesSystem,
        model: SpalartAllmarasNegativePlan | None = None,
        /,
    ):
        model_ = SpalartAllmarasNegativePlan() if model is None else model
        if not isinstance(base, HomogeneousMixtureCompressibleNavierStokesSystem):
            raise TypeError("SA-neg requires canonical mixture Navier-Stokes.")
        if base.favre_les is not None:
            raise ValueError("SA-neg and Favre LES cannot share one canonical state.")
        if not isinstance(model_, SpalartAllmarasNegativePlan):
            raise TypeError("model must be SpalartAllmarasNegativePlan.")
        self.base = base
        self.model = model_
        self.dimension = base.dimension
        self.component_names = (*base.component_names, "spalart_allmaras_density")
        self.system_id = canonical_fingerprint(
            {
                "kind": "spalart-allmaras-compressible-system",
                "base": base.system_id,
                "model": model_.plan_id,
            }
        )

    @property
    def thermodynamics(self):
        return self.base.thermodynamics

    @property
    def species_count(self) -> int:
        return self.base.species_count

    @property
    def momentum_slice(self) -> slice:
        return self.base.momentum_slice

    @property
    def energy_index(self) -> int:
        return self.base.energy_index

    @property
    def turbulence_index(self) -> int:
        return self.base.component_count

    @property
    def density_floor(self) -> float:
        return self.base.density_floor

    @property
    def pressure_floor(self) -> float:
        return self.base.pressure_floor

    @property
    def maximum_thermal_iterations(self) -> int:
        return self.base.maximum_thermal_iterations

    @property
    def transports_sgs_kinetic_energy(self) -> bool:
        return False

    def _check_state(self, state: ArrayLike, name: str, /) -> Array:
        value = jnp.asarray(state)
        if value.ndim < 1 or value.shape[-1] != self.component_count:
            raise ValueError(f"{name} must end in {self.component_count} components.")
        return value

    def gas_state(self, state: ArrayLike, /) -> Array:
        return self._check_state(state, "SA-neg state")[..., : self.base.component_count]

    def density(self, state: ArrayLike, /) -> Array:
        return self.base.density(self.gas_state(state))

    def pressure(self, state: ArrayLike, /) -> Array:
        return self.base.pressure(self.gas_state(state))

    def temperature(self, state: ArrayLike, /) -> Array:
        return self.base.temperature(self.gas_state(state))

    def recover_thermodynamics(self, state: ArrayLike, /):
        return self.base.recover_thermodynamics(self.gas_state(state))

    def working_variable(self, state: ArrayLike, /) -> Array:
        value = self._check_state(state, "SA-neg state")
        return value[..., self.turbulence_index] / jnp.maximum(
            self.density(value), self.density_floor
        )

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = self._check_state(state, "SA-neg state")
        return jnp.concatenate(
            (
                self.base.conserved_to_primitive(self.gas_state(value)),
                self.working_variable(value)[..., None],
            ),
            axis=-1,
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = self._check_state(primitive, "SA-neg primitive state")
        gas = self.base.primitive_to_conserved(value[..., : self.base.component_count])
        density = self.base.density(gas)
        turbulence = density * value[..., self.turbulence_index]
        return jnp.concatenate((gas, turbulence[..., None]), axis=-1)

    def primitive_velocity(self, primitive: Array, /) -> Array:
        value = self._check_state(primitive, "SA-neg primitive state")
        return self.base.primitive_velocity(value[..., : self.base.component_count])

    def with_primitive_velocity(self, primitive: Array, velocity: Array, /) -> Array:
        value = self._check_state(primitive, "SA-neg primitive state")
        gas = self.base.with_primitive_velocity(
            value[..., : self.base.component_count], velocity
        )
        return value.at[..., : self.base.component_count].set(gas)

    def with_primitive_temperature(
        self, primitive: Array, temperature: Array, /
    ) -> Array:
        value = self._check_state(primitive, "SA-neg primitive state")
        gas = self.base.with_primitive_temperature(
            value[..., : self.base.component_count], temperature
        )
        return value.at[..., : self.base.component_count].set(gas)

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        value = self._check_state(state, "SA-neg state")
        gas = self.gas_state(value)
        base_flux = self.base.physical_flux(gas, axis, args)
        velocity = gas[..., self.momentum_slice] / jnp.maximum(
            self.density(value)[..., None], self.density_floor
        )
        scalar_flux = value[..., self.turbulence_index] * velocity[..., int(axis)]
        return jnp.concatenate((base_flux, scalar_flux[..., None]), axis=-1)

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        return self.base.max_wave_speed(
            self.gas_state(left), self.gas_state(right), axis, args
        )

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        return self.base.signal_bounds(
            self.gas_state(left), self.gas_state(right), axis, args
        )

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        return self.base.normal_signal_bounds(
            self.gas_state(left), self.gas_state(right), normal, args
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        value = self._check_state(state, "SA-neg state")
        gas = self.base.reflect_state(self.gas_state(value), axis)
        return value.at[..., : self.base.component_count].set(gas)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        value = self._check_state(state, "SA-neg state")
        gas = self.base.reflect_normal_state(self.gas_state(value), normal)
        return value.at[..., : self.base.component_count].set(gas)

    def admissible(self, state: Array, /) -> Array:
        value = self._check_state(state, "SA-neg state")
        gas = self.gas_state(value)
        density = self.density(value)
        properties = self.base.transport_properties(gas)
        molecular = properties.dynamic_viscosity / jnp.maximum(
            density, self.density_floor
        )
        ratio = self.working_variable(value) / jnp.maximum(
            molecular, jnp.finfo(value.dtype).tiny
        )
        return (
            self.base.admissible(gas)
            & jnp.isfinite(value[..., self.turbulence_index])
            & (ratio >= self.model.minimum_working_ratio)
            & (
                self.model.diffusion_coefficient(molecular, self.working_variable(value))
                > 0.0
            )
        )

    def diffusion_evaluation(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> ConservationDiffusionEvaluation:
        if not isinstance(args, SpalartAllmarasArguments):
            raise TypeError("SA-neg diffusion requires SpalartAllmarasArguments.")
        value = self._check_state(state, "SA-neg state")
        gradient = jnp.asarray(conserved_gradient)
        if gradient.shape != value.shape + (self.dimension,):
            raise ValueError("SA-neg conserved gradients have the wrong shape.")
        gas = self.gas_state(value)
        gas_gradient = gradient[..., : self.base.component_count, :]
        base = self.base.diffusion_evaluation(gas, gas_gradient, args.transport_args)
        recovered = self.base.recover_thermodynamics(gas).state
        velocity_gradient, temperature_gradient, species_gradient = (
            self.base.primitive_gradients(gas, gas_gradient)
        )
        del species_gradient
        density = self.density(value)
        density_gradient = jnp.sum(gas_gradient[..., : self.species_count, :], axis=-2)
        working = self.working_variable(value)
        working_density_gradient = gradient[..., self.turbulence_index, :]
        working_gradient = (
            working_density_gradient - working[..., None] * density_gradient
        ) / jnp.maximum(density[..., None], self.density_floor)
        properties = self.base.transport_properties(gas, args.transport_args)
        molecular = properties.dynamic_viscosity / jnp.maximum(
            density, self.density_floor
        )
        closure = self.model.evaluate(
            density,
            molecular,
            working,
            velocity_gradient,
            working_gradient,
            args.wall_distance,
        )
        divergence = jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
        identity = jnp.eye(self.dimension, dtype=value.dtype)
        deviatoric = (
            velocity_gradient
            + jnp.swapaxes(velocity_gradient, -1, -2)
            - (2.0 / 3.0) * divergence[..., None, None] * identity
        )
        turbulent_stress = closure.eddy_viscosity[..., None, None] * deviatoric
        velocity = gas[..., self.momentum_slice] / jnp.maximum(
            density[..., None], self.density_floor
        )
        specific_cp = (
            recovered.molar_density * recovered.molar_heat_capacity_pressure
        ) / jnp.maximum(density, self.density_floor)
        turbulent_conductivity = (
            closure.eddy_viscosity * specific_cp / self.model.turbulent_prandtl
        )
        gas_flux = base.flux.at[..., self.momentum_slice, :].add(turbulent_stress)
        turbulent_energy = (
            contract("...i,...ij->...j", velocity, turbulent_stress, backend="jax")
            + turbulent_conductivity[..., None] * temperature_gradient
        )
        gas_flux = gas_flux.at[..., self.energy_index, :].add(turbulent_energy)
        scalar_flux = (
            density[..., None]
            * closure.diffusion_coefficient[..., None]
            * working_gradient
            / self.model.sigma
        )
        flux = jnp.concatenate((gas_flux, scalar_flux[..., None, :]), axis=-2)
        scalar_source = density * closure.source
        source = jnp.concatenate((base.source, scalar_source[..., None]), axis=-1)
        scalar_density = value[..., self.turbulence_index]
        scalar_step = jnp.where(
            (scalar_source < 0.0) & (scalar_density > 0.0),
            0.5
            * scalar_density
            / jnp.maximum(-scalar_source, jnp.finfo(value.dtype).tiny),
            jnp.inf,
        )
        source_step = jnp.minimum(base.source_step, jnp.min(scalar_step))
        finite = (
            base.finite
            & jnp.all(closure.finite)
            & jnp.all(jnp.isfinite(flux))
            & jnp.all(jnp.isfinite(source))
        )
        successful = base.successful & jnp.all(closure.successful) & finite
        return ConservationDiffusionEvaluation(
            flux, source, source_step, finite, successful
        )

    def viscous_flux(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        return self.diffusion_evaluation(state, conserved_gradient, args).flux

    def maximum_diffusivity(self, state: Array, args: Any = None, /) -> Array:
        transport_args = (
            args.transport_args if isinstance(args, SpalartAllmarasArguments) else args
        )
        value = self._check_state(state, "SA-neg state")
        gas = self.gas_state(value)
        density = self.density(value)
        properties = self.base.transport_properties(gas, transport_args)
        molecular = properties.dynamic_viscosity / jnp.maximum(
            density, self.density_floor
        )
        working = self.working_variable(value)
        sa_diffusion = (
            self.model.diffusion_coefficient(molecular, working) / self.model.sigma
        )
        turbulent_heat = jnp.maximum(working, 0.0) / self.model.turbulent_prandtl
        return jnp.maximum(
            self.base.maximum_diffusivity(gas, transport_args),
            jnp.maximum(sa_diffusion, turbulent_heat),
        )

    def entropy_viscous_production(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        transport_args = (
            args.transport_args if isinstance(args, SpalartAllmarasArguments) else args
        )
        value = self._check_state(state, "SA-neg state")
        gradient = jnp.asarray(conserved_gradient)
        gas = self.gas_state(value)
        gas_gradient = gradient[..., : self.base.component_count, :]
        density = self.density(value)
        density_gradient = jnp.sum(gas_gradient[..., : self.species_count, :], axis=-2)
        working = self.working_variable(value)
        working_gradient = (
            gradient[..., self.turbulence_index, :]
            - working[..., None] * density_gradient
        ) / jnp.maximum(density[..., None], self.density_floor)
        properties = self.base.transport_properties(gas, transport_args)
        molecular = properties.dynamic_viscosity / jnp.maximum(
            density, self.density_floor
        )
        scalar_production = (
            density
            * self.model.diffusion_coefficient(molecular, working)
            * jnp.sum(working_gradient * working_gradient, axis=-1)
            / self.model.sigma
        )
        return (
            self.base.entropy_viscous_production(gas, gas_gradient, transport_args)
            + scalar_production
        )


__all__ = [
    "SpalartAllmarasArguments",
    "SpalartAllmarasCompressibleSystem",
    "SpalartAllmarasEvaluation",
    "SpalartAllmarasNegativePlan",
]
