#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...equations._hyperbolic_systems import (
    AbstractAdmissibleSystem,
    AbstractEntropyDiffusionSystem,
    AbstractNormalReflectionSystem,
)
from ...equations._materials import AbstractThermodynamicMaterial
from ...equations._transport_closures import AbstractTransportClosure


class MaterialEulerSystem(AbstractAdmissibleSystem, AbstractNormalReflectionSystem):
    """Euler layout for certified caloric materials on non-characteristic FV routes."""

    material: AbstractThermodynamicMaterial

    def __init__(
        self,
        material: AbstractThermodynamicMaterial,
        dimension: int = 1,
        /,
    ):
        dimension_ = int(dimension)
        if not isinstance(material, AbstractThermodynamicMaterial):
            raise TypeError("material must implement AbstractThermodynamicMaterial.")
        if dimension_ not in (1, 2, 3):
            raise ValueError("Material Euler dimension must be one, two, or three.")
        self.dimension = dimension_
        self.component_names = (
            "density",
            *(f"momentum_{axis}" for axis in range(dimension_)),
            "total_energy",
        )
        self.material = material
        self.system_id = canonical_fingerprint(
            {
                "kind": "material-euler-system",
                "dimension": dimension_,
                "material": material.material_id,
                "route_capability": "non-characteristic-finite-volume",
            }
        )

    def pressure(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum = value[..., 1 : 1 + self.dimension]
        velocity_squared = (
            oe.contract("...d,...d->...", momentum, momentum, backend="jax") / density**2
        )
        internal = value[..., -1] / density - 0.5 * velocity_squared
        return self.material.pressure(density, internal)

    def temperature(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        return self.material.temperature(value[..., 0], self.pressure(value))

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        velocity = value[..., 1 : 1 + self.dimension] / density[..., None]
        return jnp.concatenate(
            (density[..., None], velocity, self.pressure(value)[..., None]), axis=-1
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        density = value[..., 0]
        velocity = value[..., 1 : 1 + self.dimension]
        pressure = value[..., -1]
        internal = self.material.specific_internal_energy(density, pressure)
        kinetic = (
            0.5
            * density
            * oe.contract("...d,...d->...", velocity, velocity, backend="jax")
        )
        return jnp.concatenate(
            (
                density[..., None],
                density[..., None] * velocity,
                (density * internal + kinetic)[..., None],
            ),
            axis=-1,
        )

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("Material Euler flux axis is out of range.")
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum = value[..., 1 : 1 + self.dimension]
        velocity = momentum / density[..., None]
        pressure = self.pressure(value)
        normal_velocity = velocity[..., axis_]
        momentum_flux = momentum * normal_velocity[..., None]
        momentum_flux = momentum_flux.at[..., axis_].add(pressure)
        return jnp.concatenate(
            (
                momentum[..., axis_ : axis_ + 1],
                momentum_flux,
                ((value[..., -1] + pressure) * normal_velocity)[..., None],
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
        if not 0 <= axis_ < self.dimension:
            raise ValueError("Material Euler signal axis is out of range.")
        left_primitive = self.conserved_to_primitive(left)
        right_primitive = self.conserved_to_primitive(right)
        left_sound = self.material.sound_speed(
            left_primitive[..., 0], left_primitive[..., -1]
        )
        right_sound = self.material.sound_speed(
            right_primitive[..., 0], right_primitive[..., -1]
        )
        return (
            jnp.minimum(
                left_primitive[..., 1 + axis_] - left_sound,
                right_primitive[..., 1 + axis_] - right_sound,
            ),
            jnp.maximum(
                left_primitive[..., 1 + axis_] + left_sound,
                right_primitive[..., 1 + axis_] + right_sound,
            ),
        )

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
        if normal_.shape[-1:] != (self.dimension,):
            raise ValueError("Material Euler normal has the wrong dimension.")
        left_primitive = self.conserved_to_primitive(left)
        right_primitive = self.conserved_to_primitive(right)
        left_velocity = oe.contract(
            "...d,...d->...", left_primitive[..., 1:-1], normal_, backend="jax"
        )
        right_velocity = oe.contract(
            "...d,...d->...", right_primitive[..., 1:-1], normal_, backend="jax"
        )
        left_sound = self.material.sound_speed(
            left_primitive[..., 0], left_primitive[..., -1]
        )
        right_sound = self.material.sound_speed(
            right_primitive[..., 0], right_primitive[..., -1]
        )
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

    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        pressure = self.pressure(value)
        return (
            jnp.all(jnp.isfinite(value), axis=-1)
            & jnp.isfinite(pressure)
            & self.material.admissible(value[..., 0], pressure)
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("Material Euler reflection axis is out of range.")
        return jnp.asarray(state).at[..., 1 + axis_].multiply(-1.0)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        value = jnp.asarray(state)
        normal_ = jnp.asarray(normal)
        if normal_.shape[-1:] != (self.dimension,):
            raise ValueError("Material Euler normal has the wrong dimension.")
        norm = jnp.sqrt(oe.contract("...d,...d->...", normal_, normal_, backend="jax"))
        unit = normal_ / norm[..., None]
        momentum = value[..., 1 : 1 + self.dimension]
        normal_momentum = oe.contract("...d,...d->...", momentum, unit, backend="jax")
        reflected = momentum - 2.0 * normal_momentum[..., None] * unit
        return value.at[..., 1 : 1 + self.dimension].set(reflected)


class MaterialCompressibleNavierStokesSystem(
    AbstractAdmissibleSystem,
    AbstractEntropyDiffusionSystem,
    AbstractNormalReflectionSystem,
):
    """General-caloric compressible NS for non-characteristic FV routes."""

    inviscid: MaterialEulerSystem
    material: AbstractThermodynamicMaterial
    transport: AbstractTransportClosure

    def __init__(
        self,
        material: AbstractThermodynamicMaterial,
        transport: AbstractTransportClosure,
        dimension: int = 1,
        /,
    ):
        if not isinstance(transport, AbstractTransportClosure):
            raise TypeError("transport must be an AbstractTransportClosure.")
        inviscid = MaterialEulerSystem(material, dimension)
        self.dimension = inviscid.dimension
        self.component_names = inviscid.component_names
        self.inviscid = inviscid
        self.material = material
        self.transport = transport
        self.system_id = canonical_fingerprint(
            {
                "kind": "material-compressible-navier-stokes-system",
                "inviscid": inviscid.system_id,
                "transport": transport.closure_id,
                "route_capability": "non-characteristic-finite-volume",
            }
        )

    def pressure(self, state: ArrayLike, /) -> Array:
        return self.inviscid.pressure(state)

    def temperature(self, state: ArrayLike, /) -> Array:
        return self.inviscid.temperature(state)

    def conserved_to_primitive(self, state: Array, /) -> Array:
        return self.inviscid.conserved_to_primitive(state)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return self.inviscid.primitive_to_conserved(primitive)

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        return self.inviscid.physical_flux(state, axis, args)

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

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        return self.inviscid.max_wave_speed(left, right, axis, args)

    def admissible(self, state: Array, /) -> Array:
        return self.inviscid.admissible(state)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return self.inviscid.reflect_state(state, axis)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        return self.inviscid.reflect_normal_state(state, normal)

    def primitive_gradients(
        self,
        state: ArrayLike,
        conserved_gradient: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        value = jnp.asarray(state)
        gradient = jnp.asarray(conserved_gradient)
        if gradient.shape != value.shape + (self.dimension,):
            raise ValueError(
                "Conserved gradients must append one physical derivative axis."
            )
        density = value[..., 0]
        momentum = value[..., 1 : 1 + self.dimension]
        velocity = momentum / density[..., None]
        density_gradient = gradient[..., 0, :]
        momentum_gradient = gradient[..., 1 : 1 + self.dimension, :]
        velocity_gradient = (
            momentum_gradient - velocity[..., :, None] * density_gradient[..., None, :]
        ) / density[..., None, None]
        flat_state = value.reshape((-1, value.shape[-1]))
        temperature_jacobian = jax.vmap(
            jax.jacfwd(lambda point: self.temperature(point))
        )(flat_state).reshape(value.shape)
        temperature_gradient = oe.contract(
            "...i,...id->...d",
            temperature_jacobian,
            gradient,
            backend="jax",
        )
        return velocity_gradient, temperature_gradient

    def viscous_flux(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        value = jnp.asarray(state)
        velocity = value[..., 1 : 1 + self.dimension] / value[..., :1]
        velocity_gradient, temperature_gradient = self.primitive_gradients(
            value, conserved_gradient
        )
        properties = self.transport.properties(self.temperature(value), value, args)
        divergence = jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
        identity = jnp.eye(self.dimension, dtype=value.dtype)
        stress = properties.dynamic_viscosity[..., None, None] * (
            velocity_gradient
            + jnp.swapaxes(velocity_gradient, -1, -2)
            - (2.0 / 3.0) * divergence[..., None, None] * identity
        ) + properties.bulk_viscosity[..., None, None] * (
            divergence[..., None, None] * identity
        )
        energy_flux = (
            oe.contract("...i,...ij->...j", velocity, stress, backend="jax")
            + properties.thermal_conductivity[..., None] * temperature_gradient
        )
        mass_flux = jnp.zeros(value.shape[:-1] + (1, self.dimension), dtype=value.dtype)
        return jnp.concatenate((mass_flux, stress, energy_flux[..., None, :]), axis=-2)

    def maximum_diffusivity(self, state: Array, args: Any = None, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        pressure = self.pressure(value)
        temperature = self.material.temperature(density, pressure)
        properties = self.transport.properties(temperature, value, args)
        heat_capacity = self.material.specific_heat_cp(density, pressure)
        return jnp.maximum(
            properties.dynamic_viscosity / density,
            properties.thermal_conductivity / (density * heat_capacity),
        )

    def entropy_viscous_production(
        self,
        state: Array,
        conserved_gradient: Array,
        args: Any = None,
        /,
    ) -> Array:
        value = jnp.asarray(state)
        temperature = self.temperature(value)
        velocity_gradient, temperature_gradient = self.primitive_gradients(
            value, conserved_gradient
        )
        properties = self.transport.properties(temperature, value, args)
        divergence = jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
        identity = jnp.eye(self.dimension, dtype=value.dtype)
        stress = properties.dynamic_viscosity[..., None, None] * (
            velocity_gradient
            + jnp.swapaxes(velocity_gradient, -1, -2)
            - (2.0 / 3.0) * divergence[..., None, None] * identity
        ) + properties.bulk_viscosity[..., None, None] * (
            divergence[..., None, None] * identity
        )
        viscous = (
            oe.contract("...ij,...ij->...", stress, velocity_gradient, backend="jax")
            / temperature
        )
        thermal = (
            properties.thermal_conductivity
            * oe.contract(
                "...d,...d->...",
                temperature_gradient,
                temperature_gradient,
                backend="jax",
            )
            / temperature**2
        )
        return viscous + thermal


__all__ = ["MaterialCompressibleNavierStokesSystem", "MaterialEulerSystem"]
