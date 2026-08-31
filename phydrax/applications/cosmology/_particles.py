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
from ._background import FLRWBackground


class CosmologicalParticleState(StrictModule):
    positions: Array
    canonical_momenta: Array
    masses: Array
    scale_factor: Array


class CosmologicalParticleDiagnostics(StrictModule):
    drift_factor: Array
    first_kick_factor: Array
    second_kick_factor: Array
    total_momentum: Array
    successful: Array


class CosmologicalKDKPlan(StrictModule, NonTrainableState):
    background: FLRWBackground
    box_size: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        background: FLRWBackground,
        box_size: tuple[float, ...],
        /,
    ):
        lengths = tuple(float(value) for value in box_size)
        if (
            not isinstance(background, FLRWBackground)
            or not lengths
            or any(value <= 0.0 for value in lengths)
        ):
            raise ValueError("Cosmological KDK domain is invalid.")
        self.background = background
        self.box_size = lengths
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cosmological-kdk",
                "background": background.background_id,
                "box_size": list(lengths),
            }
        )

    def initialize(
        self,
        positions: ArrayLike,
        canonical_momenta: ArrayLike,
        masses: ArrayLike,
        scale_factor: ArrayLike,
        /,
    ) -> CosmologicalParticleState:
        position = jnp.asarray(positions)
        momentum = jnp.asarray(canonical_momenta, dtype=position.dtype)
        mass = jnp.asarray(masses, dtype=position.dtype)
        if (
            position.ndim != 2
            or position.shape != momentum.shape
            or position.shape[1] != len(self.box_size)
            or mass.shape != (position.shape[0],)
        ):
            raise ValueError("Cosmological particle arrays are inconsistent.")
        return CosmologicalParticleState(
            position,
            momentum,
            mass,
            jnp.asarray(scale_factor, dtype=position.dtype).reshape(()),
        )

    def advance(
        self,
        state: CosmologicalParticleState,
        end_scale_factor: ArrayLike,
        acceleration_start: ArrayLike,
        acceleration_end: ArrayLike,
        /,
    ) -> tuple[CosmologicalParticleState, CosmologicalParticleDiagnostics]:
        end = jnp.asarray(end_scale_factor, dtype=state.scale_factor.dtype).reshape(())
        midpoint = 0.5 * (state.scale_factor + end)
        acceleration_0 = jnp.asarray(acceleration_start, dtype=state.positions.dtype)
        acceleration_1 = jnp.asarray(acceleration_end, dtype=state.positions.dtype)
        if (
            acceleration_0.shape != state.positions.shape
            or acceleration_1.shape != state.positions.shape
        ):
            raise ValueError("Cosmological accelerations must align with particles.")
        kick_0 = self.background.kick_factor(state.scale_factor, midpoint)
        kick_1 = self.background.kick_factor(midpoint, end)
        drift = self.background.drift_factor(state.scale_factor, end)
        momentum_half = (
            state.canonical_momenta + kick_0 * state.masses[:, None] * acceleration_0
        )
        positions = state.positions + drift * momentum_half / state.masses[:, None]
        box = jnp.asarray(self.box_size, dtype=positions.dtype)
        positions = jnp.mod(positions, box)
        momenta = momentum_half + kick_1 * state.masses[:, None] * acceleration_1
        successful = (
            jnp.isfinite(end)
            & (end > state.scale_factor)
            & jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(momenta))
        )
        accepted = CosmologicalParticleState(
            jnp.where(successful, positions, state.positions),
            jnp.where(successful, momenta, state.canonical_momenta),
            state.masses,
            jnp.where(successful, end, state.scale_factor),
        )
        diagnostics = CosmologicalParticleDiagnostics(
            drift_factor=drift,
            first_kick_factor=kick_0,
            second_kick_factor=kick_1,
            total_momentum=jnp.sum(accepted.canonical_momenta, axis=0),
            successful=successful,
        )
        return accepted, diagnostics


__all__ = [
    "CosmologicalKDKPlan",
    "CosmologicalParticleDiagnostics",
    "CosmologicalParticleState",
]
