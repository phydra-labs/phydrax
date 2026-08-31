#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class VortexParticleProperties(StrictModule):
    """Physical blob data independent of material-particle mass.

    Two-dimensional temporal strength is circulation; three-dimensional strength
    is a vector. Neither is inferred from ``ParticleDiscretization.masses``.
    """

    core_radius: Array | None
    volume: Array | None
    properties_id: str = eqx.field(static=True)

    def __init__(
        self,
        core_radius: ArrayLike | None = None,
        volume: ArrayLike | None = None,
        /,
        *,
        properties_id: str | None = None,
    ):
        core = None if core_radius is None else jnp.asarray(core_radius)
        volumes = None if volume is None else jnp.asarray(volume)
        if core is not None and core.ndim != 1:
            raise ValueError("Vortex-particle core_radius must be rank one.")
        if volumes is not None and volumes.ndim != 1:
            raise ValueError("Vortex-particle volume must be rank one.")
        generated = canonical_fingerprint(
            {
                "kind": "vortex-particle-properties",
                "core_radius": (None if core is None else array_tree_fingerprint(core)),
                "volume": (None if volumes is None else array_tree_fingerprint(volumes)),
            }
        )
        identifier = generated if properties_id is None else str(properties_id)
        if not identifier:
            raise ValueError("properties_id must be non-empty.")
        self.core_radius = core
        self.volume = volumes
        self.properties_id = identifier

    def validate(
        self,
        capacity: int,
        /,
        *,
        require_core_radius: bool = False,
        require_volume: bool = False,
    ) -> None:
        capacity_ = int(capacity)
        if capacity_ <= 0:
            raise ValueError("Vortex-particle capacity must be positive.")
        if require_core_radius and self.core_radius is None:
            raise ValueError("The selected vortex backend requires core_radius.")
        if require_volume and self.volume is None:
            raise ValueError("The selected vortex backend requires particle volume.")
        if self.core_radius is not None and self.core_radius.shape != (capacity_,):
            raise ValueError(
                f"core_radius must have shape ({capacity_},), got {self.core_radius.shape}."
            )
        if self.volume is not None and self.volume.shape != (capacity_,):
            raise ValueError(
                f"volume must have shape ({capacity_},), got {self.volume.shape}."
            )

    def safe_core_radius(
        self, active_mask: ArrayLike, /, *, dtype: Any | None = None
    ) -> Array:
        if self.core_radius is None:
            raise ValueError("Vortex-particle core_radius is unavailable.")
        active = jnp.asarray(active_mask, dtype=bool)
        if active.shape != self.core_radius.shape:
            raise ValueError("active_mask must match core_radius shape.")
        value = jnp.asarray(self.core_radius, dtype=dtype)
        invalid = jnp.any(jnp.where(active, ~jnp.isfinite(value) | (value <= 0.0), False))
        value = eqx.error_if(
            value,
            invalid,
            "Active vortex core radii must be finite and strictly positive.",
        )
        return jnp.where(active, value, jnp.ones_like(value))

    def safe_volume(
        self, active_mask: ArrayLike, /, *, dtype: Any | None = None
    ) -> Array:
        if self.volume is None:
            raise ValueError("Vortex-particle volume is unavailable.")
        active = jnp.asarray(active_mask, dtype=bool)
        if active.shape != self.volume.shape:
            raise ValueError("active_mask must match volume shape.")
        value = jnp.asarray(self.volume, dtype=dtype)
        invalid = jnp.any(jnp.where(active, ~jnp.isfinite(value) | (value <= 0.0), False))
        value = eqx.error_if(
            value,
            invalid,
            "Active vortex particle volumes must be finite and strictly positive.",
        )
        return jnp.where(active, value, jnp.ones_like(value))


class VortexParticleState(StrictModule):
    position: Array
    strength: Array


class VortexParticleStateLayout(StrictModule, NonTrainableState):
    """Canonical packed layout: positions followed by scalar/vector strengths."""

    capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    position_shape: tuple[int, int] = eqx.field(static=True)
    strength_shape: tuple[int, ...] = eqx.field(static=True)
    position_size: int = eqx.field(static=True)
    strength_size: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    state_geometry_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, capacity: int, dimension: int, /, *, layout_id: str | None = None):
        capacity_, dimension_ = int(capacity), int(dimension)
        if capacity_ <= 0:
            raise ValueError("Vortex-particle state capacity must be positive.")
        if dimension_ not in (2, 3):
            raise ValueError("Vortex-particle state dimension must be 2 or 3.")
        position_shape = (capacity_, dimension_)
        strength_shape = (capacity_,) if dimension_ == 2 else (capacity_, 3)
        position_size = capacity_ * dimension_
        strength_size = capacity_ if dimension_ == 2 else capacity_ * 3
        generated = canonical_fingerprint(
            {
                "kind": "vortex-particle-state-layout",
                "capacity": capacity_,
                "dimension": dimension_,
                "packing": "position-then-strength-row-major",
            }
        )
        identifier = generated if layout_id is None else str(layout_id)
        if not identifier:
            raise ValueError("layout_id must be non-empty.")
        self.capacity, self.dimension = capacity_, dimension_
        self.position_shape, self.strength_shape = position_shape, strength_shape
        self.position_size, self.strength_size = position_size, strength_size
        self.state_size = position_size + strength_size
        self.state_geometry_id = canonical_fingerprint(
            {"kind": "euclidean-vortex-particle-state", "layout": identifier}
        )
        self.layout_id = identifier

    def pack(self, position: ArrayLike, strength: ArrayLike, /) -> Array:
        position_, strength_ = jnp.asarray(position), jnp.asarray(strength)
        if position_.shape != self.position_shape:
            raise ValueError(
                f"Vortex position must have shape {self.position_shape}, got {position_.shape}."
            )
        if strength_.shape != self.strength_shape:
            raise ValueError(
                f"Vortex strength must have shape {self.strength_shape}, got {strength_.shape}."
            )
        dtype = jnp.result_type(position_, strength_, jnp.float32)
        return jnp.concatenate(
            (
                jnp.asarray(position_, dtype=dtype).reshape((-1,)),
                jnp.asarray(strength_, dtype=dtype).reshape((-1,)),
            )
        )

    def unpack(self, state: ArrayLike, /) -> VortexParticleState:
        value = jnp.asarray(state)
        if value.shape != (self.state_size,):
            raise ValueError(
                f"Packed vortex state must have shape ({self.state_size},), got {value.shape}."
            )
        return VortexParticleState(
            value[: self.position_size].reshape(self.position_shape),
            value[self.position_size :].reshape(self.strength_shape),
        )


__all__ = ["VortexParticleProperties", "VortexParticleState", "VortexParticleStateLayout"]
