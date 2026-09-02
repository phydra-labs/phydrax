#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._types import FLIPParticleState


class FLIPSolidBoundaryResult(StrictModule):
    candidate_particles: FLIPParticleState
    accepted_particles: FLIPParticleState
    collided: Array
    hit_fraction: Array
    impulse: Array
    wall_work: Array
    penetration: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class FLIPSolidBoundaryPlan(StrictModule, NonTrainableState):
    signed_distance: Callable[[Array, Array, Any], ArrayLike] = eqx.field(static=True)
    wall_velocity_provider: Callable[[Array, Array, Any], ArrayLike] = eqx.field(
        static=True
    )
    no_slip: bool = eqx.field(static=True)
    bisection_steps: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        signed_distance: Callable[[Array, Array, Any], ArrayLike],
        wall_velocity: Callable[[Array, Array, Any], ArrayLike],
        /,
        *,
        no_slip: bool,
        bisection_steps: int = 12,
        field_id: str,
    ):
        if not callable(signed_distance) or not callable(wall_velocity):
            raise TypeError("Solid boundary providers must be callable.")
        steps = int(bisection_steps)
        if steps <= 0 or not str(field_id):
            raise ValueError("Solid boundary preparation is invalid.")
        self.signed_distance = signed_distance
        self.wall_velocity_provider = wall_velocity
        self.no_slip = bool(no_slip)
        self.bisection_steps = steps
        self.source_id = str(field_id)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "flip-solid-boundary",
                "field_id": self.source_id,
                "no_slip": bool(no_slip),
                "steps": steps,
            }
        )

    def apply(
        self,
        particles: FLIPParticleState,
        proposed_position: ArrayLike,
        mass: ArrayLike,
        active_mask: ArrayLike,
        time: ArrayLike,
        /,
        *,
        args: Any = None,
    ) -> FLIPSolidBoundaryResult:
        start = particles.position
        end = jnp.asarray(proposed_position, dtype=start.dtype)
        masses = jnp.asarray(mass, dtype=start.dtype)
        active = jnp.asarray(active_mask, dtype=bool)
        time_ = jnp.asarray(time, dtype=start.dtype)
        phi_start = jnp.asarray(self.signed_distance(start, time_, args))
        phi_end = jnp.asarray(self.signed_distance(end, time_, args))
        if phi_start.shape != active.shape or phi_end.shape != active.shape:
            raise ValueError("Solid SDF must return one value per particle.")
        collided = active & (phi_start >= 0.0) & (phi_end < 0.0)
        lower = jnp.zeros_like(phi_start)
        upper = jnp.ones_like(phi_start)
        for _ in range(self.bisection_steps):
            midpoint = 0.5 * (lower + upper)
            point = start + midpoint[:, None] * (end - start)
            phi = jnp.asarray(self.signed_distance(point, time_, args))
            outside = phi >= 0.0
            lower = jnp.where(collided & outside, midpoint, lower)
            upper = jnp.where(collided & ~outside, midpoint, upper)
        fraction = jnp.where(collided, lower, 1.0)
        hit = start + fraction[:, None] * (end - start)
        epsilon = jnp.sqrt(jnp.finfo(start.dtype).eps)
        gradients = []
        for axis in range(start.shape[1]):
            direction = (
                jnp.zeros((start.shape[1],), dtype=start.dtype).at[axis].set(epsilon)
            )
            plus = jnp.asarray(self.signed_distance(hit + direction, time_, args))
            minus = jnp.asarray(self.signed_distance(hit - direction, time_, args))
            gradients.append((plus - minus) / (2.0 * epsilon))
        normal = jnp.stack(tuple(gradients), axis=-1)
        normal = normal / jnp.maximum(
            jnp.sqrt(jnp.sum(normal**2, axis=-1))[:, None], 1.0e-30
        )
        wall = jnp.asarray(self.wall_velocity_provider(hit, time_, args))
        relative = particles.velocity - wall
        reflected = (
            wall
            if self.no_slip
            else particles.velocity
            - 2.0 * jnp.sum(relative * normal, axis=-1)[:, None] * normal
        )
        remaining = (1.0 - fraction)[:, None] * (end - start)
        candidate_position = jnp.where(
            collided[:, None],
            hit
            + remaining
            - 2.0 * jnp.sum(remaining * normal, axis=-1)[:, None] * normal,
            end,
        )
        candidate_velocity = jnp.where(collided[:, None], reflected, particles.velocity)
        impulse = masses[:, None] * (candidate_velocity - particles.velocity)
        wall_work = jnp.sum(impulse * wall)
        penetration = jnp.maximum(
            -jnp.asarray(self.signed_distance(candidate_position, time_, args)), 0.0
        )
        finite = (
            jnp.all(jnp.isfinite(candidate_position))
            & jnp.all(jnp.isfinite(candidate_velocity))
            & jnp.isfinite(wall_work)
        )
        successful = finite & (
            jnp.max(jnp.where(active, penetration, 0.0), initial=0.0) <= 10.0 * epsilon
        )
        candidate = FLIPParticleState(candidate_position, candidate_velocity)
        accepted = FLIPParticleState(
            jnp.where(successful, candidate.position, particles.position),
            jnp.where(successful, candidate.velocity, particles.velocity),
        )
        return FLIPSolidBoundaryResult(
            candidate,
            accepted,
            collided,
            fraction,
            impulse,
            wall_work,
            penetration,
            finite,
            successful,
            self.plan_id,
        )


__all__ = ["FLIPSolidBoundaryPlan", "FLIPSolidBoundaryResult"]
