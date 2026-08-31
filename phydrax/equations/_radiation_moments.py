#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ._hyperbolic_systems import AbstractAdmissibleSystem


class MultigroupM1RadiationSystem(AbstractAdmissibleSystem):
    """Hyperbolic multigroup M1 radiation moments with reduced light speed."""

    group_count: int = eqx.field(static=True)
    reduced_light_speed: float = eqx.field(static=True)
    energy_floor: float = eqx.field(static=True)

    def __init__(
        self,
        group_count: int = 1,
        dimension: int = 1,
        /,
        *,
        reduced_light_speed: float = 1.0,
        energy_floor: float = 1e-12,
    ):
        groups = int(group_count)
        dimension_ = int(dimension)
        speed = float(reduced_light_speed)
        floor = float(energy_floor)
        if (
            groups <= 0
            or dimension_ not in (1, 2, 3)
            or not np.isfinite(speed)
            or speed <= 0.0
            or not np.isfinite(floor)
            or floor <= 0.0
        ):
            raise ValueError("M1 radiation system parameters are invalid.")
        self.group_count = groups
        self.dimension = dimension_
        self.reduced_light_speed = speed
        self.energy_floor = floor
        self.component_names = tuple(
            name
            for group in range(groups)
            for name in (
                f"radiation_energy_{group}",
                *(f"radiation_flux_{group}_{axis}" for axis in range(dimension_)),
            )
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "multigroup-m1-radiation-system",
                "group_count": groups,
                "dimension": dimension_,
                "reduced_light_speed": speed,
                "energy_floor": floor,
            }
        )

    @property
    def group_width(self) -> int:
        return 1 + self.dimension

    def _groups(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        if value.shape[-1] != self.group_count * self.group_width:
            raise ValueError("M1 radiation state component count is invalid.")
        return value.reshape(value.shape[:-1] + (self.group_count, self.group_width))

    def conserved_to_primitive(self, state: Array, /) -> Array:
        return jnp.asarray(state)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return jnp.asarray(primitive)

    def _eddington_tensor(self, group_state: Array, /) -> Array:
        energy = group_state[..., 0]
        flux = group_state[..., 1:]
        speed = jnp.asarray(self.reduced_light_speed, dtype=energy.dtype)
        magnitude = jnp.sqrt(jnp.sum(flux**2, axis=-1))
        reduced = jnp.clip(magnitude / (speed * energy), 0.0, 1.0)
        chi = (3.0 + 4.0 * reduced**2) / (
            5.0 + 2.0 * jnp.sqrt(jnp.maximum(4.0 - 3.0 * reduced**2, 0.0))
        )
        direction = flux / jnp.maximum(magnitude[..., None], self.energy_floor)
        identity = jnp.eye(self.dimension, dtype=energy.dtype)
        outer = direction[..., :, None] * direction[..., None, :]
        return (
            0.5 * (1.0 - chi)[..., None, None] * identity
            + 0.5 * (3.0 * chi - 1.0)[..., None, None] * outer
        )

    def physical_flux(self, state: Array, axis: int, args=None, /) -> Array:
        del args
        axis_ = int(axis)
        groups = self._groups(state)
        energy = groups[..., 0]
        flux = groups[..., 1:]
        tensor = self._eddington_tensor(groups)
        output = jnp.zeros_like(groups)
        output = output.at[..., 0].set(flux[..., axis_])
        output = output.at[..., 1:].set(
            self.reduced_light_speed**2 * energy[..., None] * tensor[..., :, axis_]
        )
        return output.reshape(state.shape)

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args=None,
        /,
    ) -> Array:
        del right, axis, args
        return jnp.full(left.shape[:-1], self.reduced_light_speed, dtype=left.dtype)

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args=None,
        /,
    ) -> tuple[Array, Array]:
        speed = self.max_wave_speed(left, right, axis, args)
        return -speed, speed

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        unit_normal: Array,
        args=None,
        /,
    ) -> tuple[Array, Array]:
        del right, unit_normal, args
        speed = jnp.full(left.shape[:-1], self.reduced_light_speed, dtype=left.dtype)
        return -speed, speed

    def admissible(self, state: Array, /) -> Array:
        groups = self._groups(state)
        energy = groups[..., 0]
        flux_norm = jnp.sqrt(jnp.sum(groups[..., 1:] ** 2, axis=-1))
        return jnp.all(
            (energy > self.energy_floor)
            & (flux_norm <= self.reduced_light_speed * energy),
            axis=-1,
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        groups = self._groups(state)
        reflected = groups.at[..., 1 + int(axis)].multiply(-1.0)
        return reflected.reshape(state.shape)


__all__ = ["MultigroupM1RadiationSystem"]
