#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization


class WeaklyCompressibleSPHStateLayout(StrictModule, NonTrainableState):
    """Packed per-particle position, velocity, and optional density state."""

    active_mask: Array
    state_geometry_id: str = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    density_evolved: bool = eqx.field(static=True)
    width: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        /,
        *,
        density_evolved: bool,
        layout_id: str | None = None,
    ):
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        capacity = particles.capacity
        dimension = particles.ambient_dimension
        evolved = bool(density_evolved)
        width = 2 * dimension + int(evolved)
        generated = canonical_fingerprint(
            {
                "kind": "weakly-compressible-sph-state-layout",
                "particles": particles.prepared_id,
                "particle_capacity": capacity,
                "ambient_dimension": dimension,
                "density_evolved": evolved,
                "width": width,
            }
        )
        identifier = generated if layout_id is None else str(layout_id)
        if not identifier:
            raise ValueError("layout_id must be non-empty.")
        self.active_mask = particles.active_mask
        self.state_geometry_id = f"state-geometry:wcsph:{identifier}"
        self.particle_capacity = capacity
        self.ambient_dimension = dimension
        self.density_evolved = evolved
        self.width = width
        self.layout_id = identifier

    @property
    def shape(self) -> tuple[int, int]:
        return self.particle_capacity, self.width

    def _vector(self, name: str, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        expected = (self.particle_capacity, self.ambient_dimension)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}.")
        active = self.active_mask[:, None]
        array = eqx.error_if(
            array,
            jnp.any(jnp.where(active, ~jnp.isfinite(array), False)),
            f"Active particle {name} values must be finite.",
        )
        return jnp.where(active, array, 0.0)

    def _density(self, value: ArrayLike, /) -> Array:
        density = jnp.asarray(value)
        expected = (self.particle_capacity,)
        if density.shape != expected:
            raise ValueError(f"density must have shape {expected}.")
        density = eqx.error_if(
            density,
            jnp.any(
                jnp.where(
                    self.active_mask,
                    ~jnp.isfinite(density) | (density <= 0.0),
                    False,
                )
            ),
            "Active particle density values must be finite and positive.",
        )
        return jnp.where(self.active_mask, density, 1.0)

    def pack(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        density: ArrayLike | None = None,
        /,
    ) -> Array:
        position_ = self._vector("position", position)
        velocity_ = self._vector("velocity", velocity)
        if self.density_evolved:
            if density is None:
                raise ValueError("Evolved-density state requires density.")
            density_ = self._density(density)
            return jnp.concatenate((position_, velocity_, density_[:, None]), axis=-1)
        if density is not None:
            raise ValueError("Summation-density state does not accept density.")
        return jnp.concatenate((position_, velocity_), axis=-1)

    def validate(self, state: ArrayLike, /) -> Array:
        position, velocity, density = self.unpack(state)
        return self.pack(position, velocity, density)

    def unpack(self, state: ArrayLike, /) -> tuple[Array, Array, Array | None]:
        value = jnp.asarray(state)
        if value.shape != self.shape:
            raise ValueError(f"WCSPH state must have shape {self.shape}.")
        dimension = self.ambient_dimension
        position = self._vector("position", value[:, :dimension])
        velocity = self._vector("velocity", value[:, dimension : 2 * dimension])
        if not self.density_evolved:
            return position, velocity, None
        return position, velocity, self._density(value[:, -1])

    def position(self, state: ArrayLike, /) -> Array:
        return self.unpack(state)[0]

    def velocity(self, state: ArrayLike, /) -> Array:
        return self.unpack(state)[1]

    def density(self, state: ArrayLike, /) -> Array:
        density = self.unpack(state)[2]
        if density is None:
            raise ValueError("Summation-density state has no density component.")
        return density

    def pack_rate(
        self,
        position_rate: ArrayLike,
        velocity_rate: ArrayLike,
        density_rate: ArrayLike | None = None,
        /,
    ) -> Array:
        position_ = jnp.asarray(position_rate)
        velocity_ = jnp.asarray(velocity_rate)
        expected_vector = (self.particle_capacity, self.ambient_dimension)
        if position_.shape != expected_vector or velocity_.shape != expected_vector:
            raise ValueError("WCSPH vector rates do not match the state layout.")
        active = self.active_mask[:, None]
        position_ = jnp.where(active, position_, 0.0)
        velocity_ = jnp.where(active, velocity_, 0.0)
        if self.density_evolved:
            if density_rate is None:
                raise ValueError("Evolved-density state requires a density rate.")
            density_ = jnp.asarray(density_rate)
            if density_.shape != (self.particle_capacity,):
                raise ValueError("WCSPH density rate does not match the state layout.")
            return jnp.concatenate(
                (
                    position_,
                    velocity_,
                    jnp.where(self.active_mask, density_, 0.0)[:, None],
                ),
                axis=-1,
            )
        if density_rate is not None:
            raise ValueError("Summation-density state does not accept a density rate.")
        return jnp.concatenate((position_, velocity_), axis=-1)

    def unpack_rate(self, rate: ArrayLike, /) -> tuple[Array, Array, Array | None]:
        value = jnp.asarray(rate)
        if value.shape != self.shape:
            raise ValueError(f"WCSPH rate must have shape {self.shape}.")
        dimension = self.ambient_dimension
        position_rate = value[:, :dimension]
        velocity_rate = value[:, dimension : 2 * dimension]
        density_rate = value[:, -1] if self.density_evolved else None
        return position_rate, velocity_rate, density_rate


__all__ = ["WeaklyCompressibleSPHStateLayout"]
