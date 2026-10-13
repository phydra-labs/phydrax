#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle import ParticleDiscretization
from ...typing import checked
from ._context import AstrodynamicsContext
from ._status import AstrodynamicsStatus


_LeapfrogCarry: TypeAlias = tuple[Array, Array, Array, Array]
_LeapfrogBranch: TypeAlias = tuple[Array, Array, Array, Array, Array]
_LeapfrogOutput: TypeAlias = tuple[Array, Array, Array, Array]


def _norm(value: Array, /, *, axis: int = -1) -> Array:
    return jnp.sqrt(jnp.sum(value * value, axis=axis))


def _gravity_workspace_policy(chunk: int | None, budget: int) -> None:
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("pair_workspace_bytes must be an integer.")
    if budget < 1:
        raise ValueError("pair_workspace_bytes must be positive.")
    if chunk is not None:
        if isinstance(chunk, bool) or not isinstance(chunk, int):
            raise TypeError("target_chunk_size must be an integer or None.")
        if chunk < 1:
            raise ValueError("target_chunk_size must be positive.")


def _direct_gravity_rows(
    positions: Array,
    masses: Array,
    active: Array,
    softening: Array,
    target_chunk_size: int | None,
    pair_workspace_bytes: int = 4 * 1024 * 1024,
) -> tuple[Array, Array, Array, Array]:
    """Exact source reductions with at most a bounded batch of target rows."""
    indices = jnp.arange(positions.shape[0])

    def row(i: Array) -> tuple[Array, Array, Array, Array]:
        pair = active[i] & active & (indices != i)
        displacement = positions - positions[i]
        squared = jnp.sum(displacement * displacement, axis=-1)
        safe = jnp.where(pair, squared + softening * softening, 1.0)
        acceleration = jnp.sum(
            masses[:, None]
            * displacement
            * jnp.where(pair, safe ** (-1.5), 0.0)[:, None],
            axis=0,
        )
        potential = jnp.sum(
            masses[i] * masses * jnp.where(pair & (indices > i), safe ** (-0.5), 0.0)
        )
        minimum = jnp.min(jnp.sqrt(jnp.where(pair, squared, jnp.inf)))
        singular = jnp.any(pair & (squared == 0.0))
        return acceleration, potential, minimum, singular

    # Six floating source-sized intermediates per target row, as in the
    # qualification workspace model; an explicit chunk overrides admission.
    extent = positions.shape[0]
    chunk = min(
        extent,
        max(1, pair_workspace_bytes // (6 * extent * positions.dtype.itemsize))
        if target_chunk_size is None
        else target_chunk_size,
    )
    if chunk == extent:
        acceleration, potential, minimum, singular = jax.vmap(row)(indices)
    else:
        acceleration, potential, minimum, singular = jax.lax.map(
            row, indices, batch_size=chunk
        )
    return acceleration, jnp.sum(potential), jnp.min(minimum), jnp.any(singular)


class NBodyState(StrictModule):
    position: Array
    velocity: Array
    particles: ParticleDiscretization
    context: AstrodynamicsContext

    @checked
    def __init__(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        particles: ParticleDiscretization,
        context: AstrodynamicsContext,
        /,
    ) -> None:
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
    target_chunk_size: int | None = eqx.field(static=True)
    pair_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        particles: ParticleDiscretization,
        context: AstrodynamicsContext,
        /,
        *,
        gravitational_constant: ArrayLike = 1.0,
        softening: ArrayLike = 0.0,
        collision_distance: ArrayLike = 0.0,
        target_chunk_size: int | None = None,
        pair_workspace_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if particles.ambient_dimension != 3:
            raise ValueError(
                "Direct N-body gravity requires three-dimensional particles."
            )
        _gravity_workspace_policy(target_chunk_size, pair_workspace_bytes)
        coupling = jnp.asarray(gravitational_constant).reshape(())
        smoothing = jnp.asarray(softening, dtype=coupling.dtype).reshape(())
        collision = jnp.asarray(collision_distance, dtype=coupling.dtype).reshape(())
        self.particles = particles
        self.context = context
        self.gravitational_constant = coupling
        self.softening = smoothing
        self.collision_distance = collision
        self.target_chunk_size = target_chunk_size
        self.pair_workspace_bytes = pair_workspace_bytes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "direct-nbody-gravity",
                "particles": particles.prepared_id,
                "context": context.context_id,
                "gravitational_constant": coupling,
                "softening": smoothing,
                "collision_distance": collision,
                "target_chunk_size": target_chunk_size,
                "pair_workspace_bytes": pair_workspace_bytes,
            }
        )

    def evaluate(self, position: ArrayLike, /) -> DirectNBodyEvaluation:
        positions = jnp.asarray(position)
        expected = (self.particles.capacity, 3)
        if positions.shape != expected:
            raise ValueError(f"N-body positions must have shape {expected}.")
        active = self.particles.active_mask
        masses = self.particles.masses.astype(positions.dtype)
        raw_acceleration, pair_potential, minimum, coincident = _direct_gravity_rows(
            positions,
            masses,
            active,
            self.softening,
            self.target_chunk_size,
            self.pair_workspace_bytes,
        )
        collision = minimum <= self.collision_distance
        singular = (self.softening == 0.0) & coincident
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
        acceleration = self.gravitational_constant * raw_acceleration
        acceleration = jnp.where(active[:, None] & domain, acceleration, 0.0)
        potential = -self.gravitational_constant * pair_potential
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

    @checked
    def __init__(self, gravity: DirectNBodyGravityPlan, times: ArrayLike, /) -> None:
        times_host = np.asarray(times, dtype=np.float64)
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
                "num_times": times_host.size,
            }
        )

    @checked
    def rollout(self, initial_state: NBodyState, /) -> NBodyPropagationResult:
        if initial_state.particles.prepared_id != self.gravity.particles.prepared_id:
            raise ValueError("N-body state and gravity particle identities differ.")
        self.gravity.context.require_compatible(initial_state.context)
        active = self.gravity.particles.active_mask[:, None]
        initial_force = self.gravity.evaluate(initial_state.position)

        def step(
            carry: _LeapfrogCarry, interval: Array
        ) -> tuple[_LeapfrogCarry, _LeapfrogOutput]:
            position, velocity, acceleration, active_path = carry
            start, end = interval
            dt = end - start

            def advance(_: None) -> _LeapfrogBranch:
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

            def hold(_: None) -> _LeapfrogBranch:
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

        def diagnostics(position: Array, velocity: Array) -> tuple[Array, Array, Array]:
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
