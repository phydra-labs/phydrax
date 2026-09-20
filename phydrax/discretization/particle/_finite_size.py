#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


ParticleFieldProvider = Callable[[Array, Array, Any], ArrayLike]


class FiniteParticleMotionKind(str, Enum):
    OVERDAMPED_STOKES = "overdamped-stokes"
    INERTIAL_STOKES = "inertial-stokes"
    OVERDAMPED_BROWNIAN = "overdamped-brownian"


class FiniteParticleWallPolicy(str, Enum):
    IMPERMEABLE_SLIDE = "impermeable-slide"
    RESTITUTION = "restitution"
    ABSORB = "absorb"


class FiniteParticleTransportUnits(StrictModule, NonTrainableState):
    length_unit_id: str = eqx.field(static=True)
    time_unit_id: str = eqx.field(static=True)
    mass_unit_id: str = eqx.field(static=True)
    temperature_unit_id: str = eqx.field(static=True)
    frame: str = eqx.field(static=True)
    units_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        length_unit_id: str,
        time_unit_id: str,
        mass_unit_id: str,
        temperature_unit_id: str,
        frame: str,
    ) -> None:
        values = tuple(
            str(value)
            for value in (
                length_unit_id,
                time_unit_id,
                mass_unit_id,
                temperature_unit_id,
                frame,
            )
        )
        if any(not value or value != value.strip() for value in values):
            raise ValueError("Finite-particle units and frame identifiers are required.")
        (
            self.length_unit_id,
            self.time_unit_id,
            self.mass_unit_id,
            self.temperature_unit_id,
            self.frame,
        ) = values
        self.units_id = canonical_fingerprint(
            {"kind": "finite-particle-units", "values": values}
        )


class FiniteParticleProperties(StrictModule):
    radii: Array
    relaxation_times: Array
    mobilities: Array
    diffusion_tensors: Array
    property_id: str = eqx.field(static=True)

    def __init__(
        self,
        radii: ArrayLike,
        relaxation_times: ArrayLike,
        mobilities: ArrayLike,
        diffusion_tensors: ArrayLike,
        /,
    ) -> None:
        radius = np.asarray(radii, dtype=np.float64)
        relaxation = np.asarray(relaxation_times, dtype=np.float64)
        mobility = np.asarray(mobilities, dtype=np.float64)
        diffusion = np.asarray(diffusion_tensors, dtype=np.float64)
        if (
            radius.ndim != 1
            or relaxation.shape != radius.shape
            or mobility.shape != radius.shape
            or diffusion.shape[:1] != radius.shape
            or diffusion.ndim != 3
            or diffusion.shape[1] != diffusion.shape[2]
            or np.any(~np.isfinite(radius))
            or np.any(radius <= 0.0)
            or np.any(~np.isfinite(relaxation))
            or np.any(relaxation < 0.0)
            or np.any(~np.isfinite(mobility))
            or np.any(mobility < 0.0)
            or np.any(~np.isfinite(diffusion))
            or not np.allclose(diffusion, np.swapaxes(diffusion, -1, -2))
        ):
            raise ValueError("Finite-particle properties are invalid.")
        diagonal = np.zeros_like(diffusion)
        diagonal_indices = np.arange(diffusion.shape[-1])
        diagonal[:, diagonal_indices, diagonal_indices] = diffusion[
            :, diagonal_indices, diagonal_indices
        ]
        if np.any(np.diagonal(diffusion, axis1=-2, axis2=-1) < 0.0) or not np.allclose(
            diffusion, diagonal
        ):
            raise ValueError(
                "Initial finite-particle diffusion tensors must be diagonal and "
                "positive semidefinite in the declared frame."
            )
        self.radii = jnp.asarray(radius)
        self.relaxation_times = jnp.asarray(relaxation)
        self.mobilities = jnp.asarray(mobility)
        self.diffusion_tensors = jnp.asarray(diffusion)
        self.property_id = canonical_fingerprint(
            {
                "kind": "finite-particle-properties",
                "radii": array_tree_fingerprint(radius),
                "relaxation_times": array_tree_fingerprint(relaxation),
                "mobilities": array_tree_fingerprint(mobility),
                "diffusion_tensors": array_tree_fingerprint(diffusion),
            }
        )

    @property
    def capacity(self) -> int:
        return self.radii.shape[0]

    @property
    def dimension(self) -> int:
        return self.diffusion_tensors.shape[-1]


class FiniteParticleVelocityFieldPlan(StrictModule, NonTrainableState):
    provider: ParticleFieldProvider = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    velocity_unit_id: str = eqx.field(static=True)
    frame: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider: ParticleFieldProvider,
        /,
        *,
        provider_id: str,
        velocity_unit_id: str,
        frame: str,
    ) -> None:
        identity = str(provider_id)
        unit = str(velocity_unit_id)
        frame_ = str(frame)
        if not callable(provider) or not identity or not unit or not frame_:
            raise ValueError("Particle velocity provider metadata are invalid.")
        self.provider = provider
        self.provider_id = identity
        self.velocity_unit_id = unit
        self.frame = frame_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-particle-velocity-field",
                "provider": identity,
                "unit": unit,
                "frame": frame_,
            }
        )

    def evaluate(self, time: Array, position: Array, args: Any = None, /) -> Array:
        value = jnp.asarray(self.provider(time, position, args), dtype=position.dtype)
        if value.shape != position.shape:
            raise ValueError(
                "Particle velocity provider must return one vector per slot."
            )
        return value


class FiniteParticleForcePlan(StrictModule, NonTrainableState):
    provider: ParticleFieldProvider = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    force_unit_id: str = eqx.field(static=True)
    frame: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider: ParticleFieldProvider,
        /,
        *,
        provider_id: str,
        force_unit_id: str,
        frame: str,
    ) -> None:
        identity = str(provider_id)
        unit = str(force_unit_id)
        frame_ = str(frame)
        if not callable(provider) or not identity or not unit or not frame_:
            raise ValueError("Particle force provider metadata are invalid.")
        self.provider = provider
        self.provider_id = identity
        self.force_unit_id = unit
        self.frame = frame_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-particle-force",
                "provider": identity,
                "unit": unit,
                "frame": frame_,
            }
        )

    def evaluate(self, time: Array, position: Array, args: Any = None, /) -> Array:
        value = jnp.asarray(self.provider(time, position, args), dtype=position.dtype)
        if value.shape != position.shape:
            raise ValueError("Particle force provider must return one vector per slot.")
        return value


__all__ = [
    "FiniteParticleForcePlan",
    "FiniteParticleMotionKind",
    "FiniteParticleProperties",
    "FiniteParticleTransportUnits",
    "FiniteParticleVelocityFieldPlan",
    "FiniteParticleWallPolicy",
    "ParticleFieldProvider",
]
