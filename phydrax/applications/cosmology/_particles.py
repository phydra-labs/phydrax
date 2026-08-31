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
from ...discretization.particle import ParticleDiscretization
from ._background import FLRWBackground
from ._scales import CODE_COSMOLOGY_SCALE, CosmologyScaleContract


class CosmologicalParticleState(StrictModule):
    """Comoving positions and p = m a^2 dx/dt at one scale factor."""

    positions: Array
    canonical_momenta: Array
    scale_factor: Array


class CosmologicalParticleDiagnostics(StrictModule):
    drift_factor: Array
    first_kick_factor: Array
    second_kick_factor: Array
    total_momentum: Array
    successful: Array


class _CosmologicalKDKProposal(StrictModule):
    positions: Array
    half_momenta: Array
    end_scale_factor: Array
    drift_factor: Array
    first_kick_factor: Array
    second_kick_factor: Array
    successful: Array


class CosmologicalKDKPlan(StrictModule, NonTrainableState):
    """Scale-factor KDK for dx/da=p/(m a^3 H) and dp/da=m g_psi/(a^2 H)."""

    particles: ParticleDiscretization
    box_size: tuple[float, ...] = eqx.field(static=True)
    scale: CosmologyScaleContract
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        box_size: tuple[float, ...],
        /,
        *,
        scale: CosmologyScaleContract = CODE_COSMOLOGY_SCALE,
    ):
        lengths = tuple(float(value) for value in box_size)
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if not isinstance(scale, CosmologyScaleContract):
            raise TypeError("scale must be a CosmologyScaleContract.")
        if (
            not lengths
            or len(lengths) != particles.ambient_dimension
            or any(not jnp.isfinite(value) or value <= 0.0 for value in lengths)
        ):
            raise ValueError("Cosmological KDK domain is invalid.")
        self.particles = particles
        self.box_size = lengths
        self.scale = scale
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cosmological-kdk",
                "particles": particles.prepared_id,
                "box_size": list(lengths),
                "scale": scale.scale_id,
            }
        )

    def initialize(
        self,
        positions: ArrayLike,
        canonical_momenta: ArrayLike,
        scale_factor: ArrayLike,
        /,
    ) -> CosmologicalParticleState:
        position = jnp.asarray(positions)
        momentum = jnp.asarray(canonical_momenta, dtype=position.dtype)
        expected = (self.particles.capacity, self.particles.ambient_dimension)
        if position.shape != expected or momentum.shape != expected:
            raise ValueError(f"Cosmological particle arrays must have shape {expected}.")
        scale = jnp.asarray(scale_factor, dtype=position.dtype)
        if scale.shape != ():
            raise ValueError("Cosmological particle scale factor must be scalar.")
        active = self.particles.active_mask[:, None]
        position = jnp.where(active, position, 0.0)
        momentum = jnp.where(active, momentum, 0.0)
        scale = eqx.error_if(
            scale,
            ~jnp.isfinite(scale)
            | (scale <= 0.0)
            | jnp.any(~jnp.isfinite(position))
            | jnp.any(~jnp.isfinite(momentum)),
            "Cosmological particle initial state must be finite with positive a.",
        )
        return CosmologicalParticleState(position, momentum, scale)

    def propose(
        self,
        background: FLRWBackground,
        state: CosmologicalParticleState,
        end_scale_factor: ArrayLike,
        acceleration_start: ArrayLike,
        /,
    ) -> _CosmologicalKDKProposal:
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        if not isinstance(state, CosmologicalParticleState):
            raise TypeError("state must be CosmologicalParticleState.")
        if background.scale.scale_id != self.scale.scale_id:
            raise ValueError("Background and KDK scale contracts disagree.")
        end = jnp.asarray(end_scale_factor, dtype=state.scale_factor.dtype)
        if end.shape != ():
            raise ValueError("Cosmological end scale factor must be scalar.")
        acceleration = jnp.asarray(acceleration_start, dtype=state.positions.dtype)
        if acceleration.shape != state.positions.shape:
            raise ValueError("Cosmological acceleration must align with particles.")
        interval_valid = jnp.isfinite(end) & (end > state.scale_factor)
        safe_end = jnp.where(
            interval_valid,
            end,
            state.scale_factor * (1.0 + jnp.finfo(state.scale_factor.dtype).eps),
        )
        midpoint = 0.5 * (state.scale_factor + safe_end)
        kick_0 = background.kick_factor(state.scale_factor, midpoint)
        kick_1 = background.kick_factor(midpoint, safe_end)
        drift = background.drift_factor(state.scale_factor, safe_end)
        masses = self.particles.safe_masses.astype(state.positions.dtype)
        active = self.particles.active_mask[:, None]
        half = state.canonical_momenta + kick_0 * masses[:, None] * acceleration
        positions = state.positions + drift * half / masses[:, None]
        positions = jnp.mod(positions, jnp.asarray(self.box_size, dtype=positions.dtype))
        positions = jnp.where(active, positions, 0.0)
        half = jnp.where(active, half, 0.0)
        successful = (
            interval_valid
            & jnp.all(jnp.isfinite(acceleration) | ~active)
            & jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(half))
        )
        return _CosmologicalKDKProposal(
            positions,
            half,
            safe_end,
            drift,
            kick_0,
            kick_1,
            successful,
        )

    def complete(
        self,
        state: CosmologicalParticleState,
        proposal: _CosmologicalKDKProposal,
        acceleration_end: ArrayLike,
        /,
    ) -> tuple[CosmologicalParticleState, CosmologicalParticleDiagnostics]:
        acceleration = jnp.asarray(acceleration_end, dtype=state.positions.dtype)
        if acceleration.shape != state.positions.shape:
            raise ValueError("Cosmological acceleration must align with particles.")
        masses = self.particles.safe_masses.astype(state.positions.dtype)
        active = self.particles.active_mask[:, None]
        momenta = (
            proposal.half_momenta
            + proposal.second_kick_factor * masses[:, None] * acceleration
        )
        momenta = jnp.where(active, momenta, 0.0)
        successful = (
            proposal.successful
            & jnp.all(jnp.isfinite(acceleration) | ~active)
            & jnp.all(jnp.isfinite(momenta))
        )
        accepted = CosmologicalParticleState(
            jnp.where(successful, proposal.positions, state.positions),
            jnp.where(successful, momenta, state.canonical_momenta),
            jnp.where(successful, proposal.end_scale_factor, state.scale_factor),
        )
        diagnostics = CosmologicalParticleDiagnostics(
            drift_factor=proposal.drift_factor,
            first_kick_factor=proposal.first_kick_factor,
            second_kick_factor=proposal.second_kick_factor,
            total_momentum=jnp.sum(accepted.canonical_momenta, axis=0),
            successful=successful,
        )
        return accepted, diagnostics

    def advance(
        self,
        background: FLRWBackground,
        state: CosmologicalParticleState,
        end_scale_factor: ArrayLike,
        acceleration_start: ArrayLike,
        acceleration_end: ArrayLike,
        /,
    ) -> tuple[CosmologicalParticleState, CosmologicalParticleDiagnostics]:
        proposal = self.propose(
            background,
            state,
            end_scale_factor,
            acceleration_start,
        )
        return self.complete(state, proposal, acceleration_end)


__all__ = [
    "CosmologicalKDKPlan",
    "CosmologicalParticleDiagnostics",
    "CosmologicalParticleState",
]
