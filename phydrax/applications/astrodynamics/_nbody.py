#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle import ParticleDiscretization
from ._context import AstrodynamicsContext
from ._status import AstrodynamicsStatus


def _norm(value: Array, /, *, axis=-1) -> Array:
    return jnp.sqrt(jnp.sum(value * value, axis=axis))


class NBodyState(StrictModule):
    position: Array
    velocity: Array
    particles: ParticleDiscretization
    context: AstrodynamicsContext

    def __init__(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        particles: ParticleDiscretization,
        context: AstrodynamicsContext,
        /,
    ):
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        if particles.ambient_dimension != 3:
            raise ValueError("Astrodynamics N-body state requires three dimensions.")
        position_ = jnp.asarray(position)
        velocity_ = jnp.asarray(velocity, dtype=position_.dtype)
        expected = (particles.capacity, 3)
        if position_.shape != expected or velocity_.shape != expected:
            raise ValueError(f"N-body position and velocity must have shape {expected}.")
        active = particles.active_mask[:, None]
        self.position = jnp.where(active, position_, 0.0)
        self.velocity = jnp.where(active, velocity_, 0.0)
        self.particles = particles
        self.context = context


class DirectNBodyEvaluation(StrictModule):
    acceleration: Array
    potential_energy: Array
    net_internal_force: Array
    minimum_pair_distance: Array
    collision: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class DirectNBodyGravityPlan(StrictModule, NonTrainableState):
    particles: ParticleDiscretization
    context: AstrodynamicsContext
    gravitational_constant: Array
    softening: Array
    collision_distance: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        context: AstrodynamicsContext,
        /,
        *,
        gravitational_constant: ArrayLike = 1.0,
        softening: ArrayLike = 0.0,
        collision_distance: ArrayLike = 0.0,
    ):
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if particles.ambient_dimension != 3:
            raise ValueError(
                "Direct N-body gravity requires three-dimensional particles."
            )
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        coupling = jnp.asarray(gravitational_constant).reshape(())
        smoothing = jnp.asarray(softening, dtype=coupling.dtype).reshape(())
        collision = jnp.asarray(collision_distance, dtype=coupling.dtype).reshape(())
        self.particles = particles
        self.context = context
        self.gravitational_constant = coupling
        self.softening = smoothing
        self.collision_distance = collision
        self.plan_id = canonical_fingerprint(
            {
                "kind": "direct-nbody-gravity",
                "particles": particles.prepared_id,
                "context": context.context_id,
            }
        )

    def evaluate(self, position: ArrayLike, /) -> DirectNBodyEvaluation:
        positions = jnp.asarray(position)
        expected = (self.particles.capacity, 3)
        if positions.shape != expected:
            raise ValueError(f"N-body positions must have shape {expected}.")
        active = self.particles.active_mask
        pair_active = active[:, None] & active[None, :]
        diagonal = jnp.eye(self.particles.capacity, dtype=bool)
        pair_active = pair_active & ~diagonal
        displacement = positions[None, :, :] - positions[:, None, :]
        distance_squared = jnp.sum(displacement * displacement, axis=-1)
        distance = jnp.sqrt(jnp.where(pair_active, distance_squared, jnp.inf))
        minimum = jnp.min(distance)
        collision = minimum <= self.collision_distance
        singular = (self.softening == 0.0) & jnp.any(
            pair_active & (distance_squared == 0.0)
        )
        finite = (
            jnp.all(jnp.where(active[:, None], jnp.isfinite(positions), True))
            & jnp.isfinite(self.gravitational_constant)
            & jnp.isfinite(self.softening)
            & jnp.isfinite(self.collision_distance)
        )
        domain = (
            finite
            & (self.gravitational_constant > 0.0)
            & (self.softening >= 0.0)
            & (self.collision_distance >= 0.0)
            & ~singular
            & ~collision
        )
        softened = distance_squared + self.softening * self.softening
        safe_softened = jnp.where(pair_active, softened, 1.0)
        inverse_cube = jnp.where(pair_active, safe_softened ** (-1.5), 0.0)
        masses = self.particles.masses.astype(positions.dtype)
        acceleration = self.gravitational_constant * jnp.sum(
            masses[None, :, None] * displacement * inverse_cube[..., None], axis=1
        )
        acceleration = jnp.where(active[:, None] & domain, acceleration, 0.0)
        upper = jnp.triu(pair_active, k=1)
        inverse_distance = jnp.where(upper, safe_softened ** (-0.5), 0.0)
        potential = -self.gravitational_constant * jnp.sum(
            masses[:, None] * masses[None, :] * inverse_distance
        )
        net_force = jnp.sum(masses[:, None] * acceleration, axis=0)
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                singular | collision,
                int(AstrodynamicsStatus.COLLISION),
                jnp.where(
                    domain,
                    int(AstrodynamicsStatus.SUCCESS),
                    int(AstrodynamicsStatus.INVALID_DOMAIN),
                ),
            ),
        ).astype(jnp.int32)
        return DirectNBodyEvaluation(
            acceleration,
            jnp.where(domain, potential, jnp.asarray(jnp.nan, dtype=positions.dtype)),
            net_force,
            minimum,
            singular | collision,
            domain,
            status,
            self.plan_id,
        )


class NBodyPropagationResult(StrictModule):
    times: Array
    positions: Array
    velocities: Array
    valid: Array
    status: Array
    energy: Array
    linear_momentum: Array
    angular_momentum: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class NBodyPropagationPlan(StrictModule, NonTrainableState):
    gravity: DirectNBodyGravityPlan
    times: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, gravity: DirectNBodyGravityPlan, times: ArrayLike, /):
        if not isinstance(gravity, DirectNBodyGravityPlan):
            raise TypeError("gravity must be a DirectNBodyGravityPlan.")
        times_host = np.asarray(times, dtype=float)
        if (
            times_host.ndim != 1
            or times_host.size < 2
            or np.any(~np.isfinite(times_host))
            or np.any(np.diff(times_host) <= 0.0)
        ):
            raise ValueError("N-body times must be finite and strictly increasing.")
        self.gravity = gravity
        self.times = jnp.asarray(times_host)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "direct-nbody-propagation",
                "gravity": gravity.plan_id,
                "num_times": int(times_host.size),
            }
        )

    def rollout(self, initial_state: NBodyState, /) -> NBodyPropagationResult:
        if not isinstance(initial_state, NBodyState):
            raise TypeError("initial_state must be an NBodyState.")
        if initial_state.particles.prepared_id != self.gravity.particles.prepared_id:
            raise ValueError("N-body state and gravity particle identities differ.")
        self.gravity.context.require_compatible(initial_state.context)
        active = self.gravity.particles.active_mask[:, None]
        initial_force = self.gravity.evaluate(initial_state.position)

        def step(carry, interval):
            position, velocity, acceleration, active_path = carry
            start, end = interval
            dt = end - start

            def advance(_):
                half_velocity = velocity + 0.5 * dt * acceleration
                next_position = jnp.where(active, position + dt * half_velocity, 0.0)
                next_force = self.gravity.evaluate(next_position)
                next_velocity = jnp.where(
                    active,
                    half_velocity + 0.5 * dt * next_force.acceleration,
                    0.0,
                )
                accepted = next_force.valid & jnp.all(jnp.isfinite(next_velocity))
                return (
                    jnp.where(accepted, next_position, position),
                    jnp.where(accepted, next_velocity, velocity),
                    jnp.where(accepted, next_force.acceleration, acceleration),
                    accepted,
                    next_force.status,
                )

            def hold(_):
                return (
                    position,
                    velocity,
                    acceleration,
                    jnp.asarray(False),
                    jnp.asarray(int(AstrodynamicsStatus.NONCONVERGED), dtype=jnp.int32),
                )

            next_position, next_velocity, next_acceleration, accepted, status = (
                jax.lax.cond(active_path, advance, hold, operand=None)
            )
            return (
                next_position,
                next_velocity,
                next_acceleration,
                active_path & accepted,
            ), (next_position, next_velocity, active_path & accepted, status)

        intervals = jnp.stack((self.times[:-1], self.times[1:]), axis=-1)
        (_, _, _, completed), outputs = jax.lax.scan(
            step,
            (
                initial_state.position,
                initial_state.velocity,
                initial_force.acceleration,
                initial_force.valid,
            ),
            intervals,
        )
        positions = jnp.concatenate((initial_state.position[None], outputs[0]), axis=0)
        velocities = jnp.concatenate((initial_state.velocity[None], outputs[1]), axis=0)
        valid = jnp.concatenate((initial_force.valid[None], outputs[2]))
        status = jnp.concatenate((initial_force.status[None], outputs[3]))
        masses = self.gravity.particles.masses.astype(positions.dtype)

        def diagnostics(position, velocity):
            force = self.gravity.evaluate(position)
            kinetic = 0.5 * jnp.sum(masses[:, None] * velocity * velocity)
            momentum = jnp.sum(masses[:, None] * velocity, axis=0)
            angular = jnp.sum(jnp.cross(position, masses[:, None] * velocity), axis=0)
            return kinetic + force.potential_energy, momentum, angular

        energy, momentum, angular = jax.vmap(diagnostics)(positions, velocities)
        return NBodyPropagationResult(
            self.times,
            positions,
            velocities,
            valid,
            status,
            energy,
            momentum,
            angular,
            completed,
            self.plan_id,
        )


__all__ = [
    "DirectNBodyEvaluation",
    "DirectNBodyGravityPlan",
    "NBodyPropagationPlan",
    "NBodyPropagationResult",
    "NBodyState",
]
