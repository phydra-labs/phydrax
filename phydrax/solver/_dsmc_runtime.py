#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.dsmc._chemistry import DSMCInternalReactionPlan
from ..discretization.dsmc._collisions import VSSVHSCollisionPlan
from ..discretization.dsmc._core import (
    DSMCParticleState,
    DSMCStreamingPlan,
)


class DSMCMomentEvaluation(StrictModule):
    number_density: Array
    mass_density: Array
    velocity: Array
    translational_temperature: Array
    rotational_energy_density: Array
    vibrational_energy_density: Array
    sample_weight: Array
    finite: Array
    successful: Array


class DSMCRuntimeState(StrictModule):
    particles: DSMCParticleState
    time: Array
    accepted_steps: Array
    key: PRNGKeyArray
    runtime_id: str = eqx.field(static=True)


class DSMCStepResult(StrictModule):
    candidate: DSMCRuntimeState
    accepted: DSMCRuntimeState
    moments: DSMCMomentEvaluation
    collision_count: Array
    reaction_count: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DSMCProductionPlan(StrictModule, NonTrainableState):
    """Fixed-capacity streaming, elastic collision, and internal-reaction epoch."""

    streaming: DSMCStreamingPlan
    collisions: VSSVHSCollisionPlan
    internal: DSMCInternalReactionPlan
    collision_candidate_capacity: int = eqx.field(static=True)
    majorant_sigma_speed: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        streaming: DSMCStreamingPlan,
        collisions: VSSVHSCollisionPlan,
        internal: DSMCInternalReactionPlan,
        /,
        *,
        collision_candidate_capacity: int,
        majorant_sigma_speed: float,
    ):
        capacity = int(collision_candidate_capacity)
        majorant = float(majorant_sigma_speed)
        if (
            not isinstance(streaming, DSMCStreamingPlan)
            or not isinstance(collisions, VSSVHSCollisionPlan)
            or not isinstance(internal, DSMCInternalReactionPlan)
            or collisions.species.plan_id != internal.species.plan_id
            or capacity <= 0
            or not np.isfinite(majorant)
            or majorant <= 0.0
        ):
            raise ValueError("DSMC production plans, capacity, or majorant are invalid.")
        self.streaming = streaming
        self.collisions = collisions
        self.internal = internal
        self.collision_candidate_capacity = capacity
        self.majorant_sigma_speed = majorant
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-production",
                "streaming": streaming.plan_id,
                "collisions": collisions.plan_id,
                "internal": internal.plan_id,
                "collision_candidate_capacity": capacity,
                "majorant_sigma_speed": majorant,
            }
        )

    def initialize(
        self, particles: DSMCParticleState, key: PRNGKeyArray, /
    ) -> DSMCRuntimeState:
        if particles.position.shape[0] <= 1 or key.shape != (2,):
            raise ValueError(
                "DSMC initialization requires particle capacity and PRNG key."
            )
        return DSMCRuntimeState(
            particles,
            jnp.asarray(0.0, dtype=particles.position.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            key,
            canonical_fingerprint(
                {
                    "kind": "dsmc-runtime",
                    "plan": self.plan_id,
                    "capacity": particles.capacity,
                }
            ),
        )

    def moments(self, particles: DSMCParticleState, /) -> DSMCMomentEvaluation:
        cell_count = self.streaming.cells.cell_count
        active = particles.active & (particles.cell_id >= 0)
        weight = jnp.where(active, particles.statistical_weight, 0.0)
        cell_weight = (
            jnp.zeros((cell_count,), dtype=weight.dtype).at[particles.cell_id].add(weight)
        )
        species_mass = self.collisions.species.molecular_masses[particles.species_index]
        particle_mass = weight * species_mass
        cell_mass = (
            jnp.zeros((cell_count,), dtype=weight.dtype)
            .at[particles.cell_id]
            .add(particle_mass)
        )
        momentum = (
            jnp.zeros(
                (cell_count, particles.velocity.shape[-1]), dtype=particles.velocity.dtype
            )
            .at[particles.cell_id]
            .add(particle_mass[:, None] * particles.velocity)
        )
        velocity = momentum / jnp.maximum(
            cell_mass[:, None], jnp.finfo(cell_mass.dtype).tiny
        )
        peculiar = particles.velocity - velocity[jnp.maximum(particles.cell_id, 0)]
        kinetic = 0.5 * particle_mass * jnp.sum(peculiar * peculiar, axis=-1)
        cell_kinetic = (
            jnp.zeros((cell_count,), dtype=weight.dtype)
            .at[particles.cell_id]
            .add(jnp.where(active, kinetic, 0.0))
        )
        dimension = particles.velocity.shape[-1]
        temperature = (
            2.0
            * cell_kinetic
            / jnp.maximum(
                dimension * 1.380649e-23 * cell_weight, jnp.finfo(weight.dtype).tiny
            )
        )
        rotational = (
            jnp.zeros((cell_count,), dtype=weight.dtype)
            .at[particles.cell_id]
            .add(jnp.where(active, weight * particles.rotational_energy, 0.0))
        )
        vibrational = (
            jnp.zeros((cell_count,), dtype=weight.dtype)
            .at[particles.cell_id]
            .add(jnp.where(active, weight * particles.vibrational_energy, 0.0))
        )
        volume = self.streaming.cells.cell_volume
        finite = jnp.all(jnp.isfinite(velocity)) & jnp.all(jnp.isfinite(temperature))
        return DSMCMomentEvaluation(
            cell_weight / volume,
            cell_mass / volume,
            velocity,
            temperature,
            rotational / volume,
            vibrational / volume,
            cell_weight,
            finite,
            finite & jnp.all(cell_weight >= 0.0),
        )

    def advance(self, state: DSMCRuntimeState, step_size: ArrayLike, /) -> DSMCStepResult:
        step = jnp.asarray(step_size, dtype=state.particles.position.dtype)
        stream = self.streaming.advance(state.particles, step)
        key_next, pair_key, collision_key, internal_key = jax.random.split(state.key, 4)
        first = jax.random.randint(
            pair_key, (self.collision_candidate_capacity,), 0, state.particles.capacity
        )
        second = jax.random.randint(
            jax.random.fold_in(pair_key, 1),
            (self.collision_candidate_capacity,),
            0,
            state.particles.capacity,
        )
        collision_uniforms = jax.random.uniform(
            collision_key,
            (self.collision_candidate_capacity, 3),
            dtype=state.particles.position.dtype,
        )
        collision = self.collisions.collide(
            stream.state, first, second, collision_uniforms, self.majorant_sigma_speed
        )
        internal_uniforms = jax.random.uniform(
            internal_key,
            (self.collision_candidate_capacity, 4),
            dtype=state.particles.position.dtype,
        )
        internal = self.internal.apply(collision.state, first, second, internal_uniforms)
        finite = stream.finite & collision.finite & internal.finite
        successful = (
            stream.successful & collision.successful & internal.successful & finite
        )
        candidate = DSMCRuntimeState(
            internal.state,
            state.time + step,
            state.accepted_steps + 1,
            key_next,
            state.runtime_id,
        )
        accepted = DSMCRuntimeState(
            jax.tree.map(
                lambda new, old: (
                    jnp.where(successful, new, old) if isinstance(new, jax.Array) else new
                ),
                candidate.particles,
                state.particles,
            ),
            jnp.where(successful, candidate.time, state.time),
            jnp.where(successful, candidate.accepted_steps, state.accepted_steps),
            jnp.where(successful, candidate.key, state.key),
            state.runtime_id,
        )
        moments = self.moments(accepted.particles)
        return DSMCStepResult(
            candidate,
            accepted,
            moments,
            jnp.sum(collision.accepted),
            jnp.sum(internal.reacted),
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "DSMCMomentEvaluation",
    "DSMCProductionPlan",
    "DSMCRuntimeState",
    "DSMCStepResult",
]
