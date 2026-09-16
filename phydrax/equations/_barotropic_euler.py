#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein

from .._fingerprint import canonical_fingerprint
from ._barotropic import AbstractBarotropicMaterial
from ._hyperbolic_systems import (
    AbstractAdmissibleSystem,
    AbstractNormalReflectionSystem,
)


class BarotropicEulerSystem(AbstractAdmissibleSystem, AbstractNormalReflectionSystem):
    """Mass and momentum conservation closed by one barotropic material."""

    material: AbstractBarotropicMaterial

    def __init__(
        self,
        dimension: int = 1,
        /,
        *,
        material: AbstractBarotropicMaterial,
    ):
        dimension_ = int(dimension)
        if dimension_ not in (1, 2, 3):
            raise ValueError("Barotropic Euler dimension must be one, two, or three.")
        if not isinstance(material, AbstractBarotropicMaterial):
            raise TypeError("material must be AbstractBarotropicMaterial.")
        self.dimension = dimension_
        self.component_names = (
            "density",
            *(f"momentum_{axis}" for axis in range(dimension_)),
        )
        self.material = material
        self.system_id = canonical_fingerprint(
            {
                "kind": "barotropic-euler-system",
                "dimension": dimension_,
                "material": material.material_id,
            }
        )

    @property
    def momentum_slice(self) -> slice:
        return slice(1, 1 + self.dimension)

    def pressure(self, state: ArrayLike, /) -> Array:
        return self.material.pressure(jnp.asarray(state)[..., 0])

    def primitive_velocity(self, primitive: Array, /) -> Array:
        return jnp.asarray(primitive)[..., 1 : 1 + self.dimension]

    def with_primitive_velocity(self, primitive: Array, velocity: Array, /) -> Array:
        return jnp.asarray(primitive).at[..., 1 : 1 + self.dimension].set(velocity)

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        velocity = value[..., self.momentum_slice] / density[..., None]
        return jnp.concatenate((density[..., None], velocity), axis=-1)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        density = value[..., 0]
        velocity = value[..., 1 : 1 + self.dimension]
        return jnp.concatenate(
            (density[..., None], density[..., None] * velocity), axis=-1
        )

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("Barotropic Euler flux axis is out of range.")
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum = value[..., self.momentum_slice]
        velocity = momentum / density[..., None]
        normal_velocity = velocity[..., axis_]
        momentum_flux = momentum * normal_velocity[..., None]
        momentum_flux = momentum_flux.at[..., axis_].add(self.material.pressure(density))
        return jnp.concatenate((momentum[..., axis_ : axis_ + 1], momentum_flux), axis=-1)

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
        left_sound = self.material.sound_speed(left_primitive[..., 0])
        right_sound = self.material.sound_speed(right_primitive[..., 0])
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
        left_primitive = self.conserved_to_primitive(left)
        right_primitive = self.conserved_to_primitive(right)
        left_normal = ein.contract(
            "...d,...d->...", left_primitive[..., 1:], normal_, backend="jax"
        )
        right_normal = ein.contract(
            "...d,...d->...", right_primitive[..., 1:], normal_, backend="jax"
        )
        left_sound = self.material.sound_speed(left_primitive[..., 0])
        right_sound = self.material.sound_speed(right_primitive[..., 0])
        return (
            jnp.minimum(left_normal - left_sound, right_normal - right_sound),
            jnp.maximum(left_normal + left_sound, right_normal + right_sound),
        )

    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return self.material.admissible(value[..., 0]) & jnp.all(
            jnp.isfinite(value), axis=-1
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        return jnp.asarray(state).at[..., 1 + int(axis)].multiply(-1.0)

    def reflect_normal_state(self, state: Array, normal: Array, /) -> Array:
        value = jnp.asarray(state)
        normal_ = jnp.asarray(normal)
        norm = jnp.sqrt(ein.contract("...d,...d->...", normal_, normal_, backend="jax"))
        unit = normal_ / norm[..., None]
        momentum = value[..., self.momentum_slice]
        normal_momentum = ein.contract("...d,...d->...", momentum, unit, backend="jax")
        reflected = momentum - 2.0 * normal_momentum[..., None] * unit
        return value.at[..., self.momentum_slice].set(reflected)

    def total_energy_density(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum = value[..., self.momentum_slice]
        kinetic = 0.5 * jnp.sum(momentum**2, axis=-1) / density
        internal = density * self.material.specific_internal_energy(density)
        return internal + kinetic


__all__ = ["BarotropicEulerSystem"]
