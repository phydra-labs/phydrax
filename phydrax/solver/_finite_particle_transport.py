#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from phydrax.ein import contract

from .._admissibility import (
    AdmissibilityHeader,
    AdmissibilityReason,
    DerivativeAvailability,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle._finite_size import (
    FiniteParticleForcePlan,
    FiniteParticleMotionKind,
    FiniteParticleProperties,
    FiniteParticleTransportUnits,
    FiniteParticleVelocityFieldPlan,
    FiniteParticleWallPolicy,
)
from ..discretization.particle._population import (
    ParticlePopulationPlan,
    ParticlePopulationState,
    ParticleSlotReusePolicy,
)
from ..geometry._eroded_domain import AbstractFiniteRadiusWallPlan


class FiniteParticleTransportReason(IntFlag):
    INITIAL_CLEARANCE_INVALID = 1 << 8
    FINAL_CLEARANCE_INVALID = 1 << 9
    WALL_EVENT_UNRESOLVED = 1 << 10
    MOTION_MODEL_INVALID = 1 << 11


class FiniteParticleTransportState(StrictModule):
    population: ParticlePopulationState
    position: Array
    velocity: Array
    terminal_code: Array
    event_count: Array
    time: Array
    accepted_steps: Array
    key: Array
    runtime_id: str = eqx.field(static=True)


class FiniteParticleStepResult(StrictModule):
    candidate: FiniteParticleTransportState
    accepted: FiniteParticleTransportState
    free_position: Array
    contact_fraction: Array
    contact_point: Array
    contact_normal: Array
    contacted: Array
    absorbed: Array
    header: AdmissibilityHeader
    derivative_availability: DerivativeAvailability = eqx.field(static=True)
    successful: Array
    plan_id: str = eqx.field(static=True)


class FiniteParticleTransportPlan(StrictModule, NonTrainableState):
    """One-way finite-radius transport with bounded exact wall events."""

    population: ParticlePopulationPlan
    properties: FiniteParticleProperties
    units: FiniteParticleTransportUnits
    velocity_field: FiniteParticleVelocityFieldPlan
    force: FiniteParticleForcePlan | None
    wall: AbstractFiniteRadiusWallPlan
    motion: FiniteParticleMotionKind = eqx.field(static=True)
    wall_policy: FiniteParticleWallPolicy = eqx.field(static=True)
    restitution: float = eqx.field(static=True)
    bisection_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        population: ParticlePopulationPlan,
        properties: FiniteParticleProperties,
        units: FiniteParticleTransportUnits,
        velocity_field: FiniteParticleVelocityFieldPlan,
        wall: AbstractFiniteRadiusWallPlan,
        /,
        *,
        motion: FiniteParticleMotionKind,
        wall_policy: FiniteParticleWallPolicy = FiniteParticleWallPolicy.IMPERMEABLE_SLIDE,
        force: FiniteParticleForcePlan | None = None,
        restitution: float = 1.0,
        bisection_steps: int = 24,
    ) -> None:
        motion_ = FiniteParticleMotionKind(motion)
        wall_policy_ = FiniteParticleWallPolicy(wall_policy)
        restitution_ = float(restitution)
        bisection = int(bisection_steps)
        if (
            not isinstance(population, ParticlePopulationPlan)
            or not isinstance(properties, FiniteParticleProperties)
            or not isinstance(units, FiniteParticleTransportUnits)
            or not isinstance(velocity_field, FiniteParticleVelocityFieldPlan)
            or not isinstance(wall, AbstractFiniteRadiusWallPlan)
            or (force is not None and not isinstance(force, FiniteParticleForcePlan))
            or properties.capacity != population.particles.capacity
            or properties.dimension != population.particles.ambient_dimension
            or wall.dimension != properties.dimension
            or velocity_field.frame != units.frame
            or (force is not None and force.frame != units.frame)
            or not np.isfinite(restitution_)
            or not 0.0 <= restitution_ <= 1.0
            or bisection <= 0
        ):
            raise ValueError("Finite-particle transport plans are incompatible.")
        structural = np.asarray(population.particles.active_mask)
        if motion_ is FiniteParticleMotionKind.INERTIAL_STOKES and np.any(
            np.asarray(properties.relaxation_times)[structural] <= 0.0
        ):
            raise ValueError(
                "Inertial Stokes transport requires positive relaxation time."
            )
        if motion_ is FiniteParticleMotionKind.OVERDAMPED_BROWNIAN and np.any(
            np.sum(
                np.diagonal(np.asarray(properties.diffusion_tensors), axis1=-2, axis2=-1),
                axis=-1,
            )[structural]
            <= 0.0
        ):
            raise ValueError("Brownian transport requires positive active diffusion.")
        self.population = population
        self.properties = properties
        self.units = units
        self.velocity_field = velocity_field
        self.force = force
        self.wall = wall
        self.motion = motion_
        self.wall_policy = wall_policy_
        self.restitution = restitution_
        self.bisection_steps = bisection
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-particle-transport",
                "population": population.plan_id,
                "properties": properties.property_id,
                "units": units.units_id,
                "velocity_field": velocity_field.plan_id,
                "force": None if force is None else force.plan_id,
                "wall": wall.plan_id,
                "motion": motion_.value,
                "wall_policy": wall_policy_.value,
                "restitution": restitution_,
                "bisection_steps": bisection,
            }
        )

    @property
    def runtime_id(self) -> str:
        return self._runtime_id()

    def _runtime_id(self, /) -> str:
        return canonical_fingerprint(
            {
                "kind": "finite-particle-runtime",
                "plan": self.plan_id,
                "capacity": self.properties.capacity,
            }
        )

    def initialize(
        self,
        population: ParticlePopulationState,
        position: ArrayLike,
        velocity: ArrayLike,
        key: PRNGKeyArray,
        /,
    ) -> FiniteParticleTransportState:
        position_ = jnp.asarray(position)
        velocity_ = jnp.asarray(velocity, dtype=position_.dtype)
        shape = (self.properties.capacity, self.properties.dimension)
        if (
            position_.shape != shape
            or velocity_.shape != shape
            or population.active.shape != (self.properties.capacity,)
            or jax.random.key_data(key).shape != (2,)
        ):
            raise ValueError("Finite-particle initial state has incompatible shape.")
        active = np.asarray(population.active)
        finite = np.all(np.isfinite(np.asarray(position_)[active])) and np.all(
            np.isfinite(np.asarray(velocity_)[active])
        )
        erosion = self.wall.evaluate(position_, self.properties.radii)
        if not finite or not np.all(np.asarray(erosion.header.eligible)[active]):
            raise ValueError("Active finite particles must start in the eroded domain.")
        return FiniteParticleTransportState(
            population,
            jnp.where(population.active[:, None], position_, 0.0),
            jnp.where(population.active[:, None], velocity_, 0.0),
            jnp.where(population.active, 0, -1).astype(jnp.int32),
            jnp.zeros((self.properties.capacity,), dtype=jnp.int32),
            jnp.asarray(0.0, dtype=position_.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jax.random.key_data(key),
            self._runtime_id(),
        )

    def _free_motion(
        self,
        state: FiniteParticleTransportState,
        step: Array,
        key: PRNGKeyArray,
        args: Any,
        /,
    ) -> tuple[Array, Array, Array]:
        fluid_velocity = self.velocity_field.evaluate(state.time, state.position, args)
        force = (
            jnp.zeros_like(state.position)
            if self.force is None
            else self.force.evaluate(state.time, state.position, args)
        )
        mass = jnp.maximum(state.population.mass, jnp.finfo(state.position.dtype).tiny)
        mobility = self.properties.mobilities.astype(state.position.dtype)
        drift = fluid_velocity + mobility[:, None] * force
        if self.motion is FiniteParticleMotionKind.INERTIAL_STOKES:
            relaxation = self.properties.relaxation_times.astype(state.position.dtype)
            target = fluid_velocity + relaxation[:, None] * force / mass[:, None]
            decay = jnp.exp(-step / relaxation)
            velocity = target + (state.velocity - target) * decay[:, None]
            position = (
                state.position
                + target * step
                + relaxation[:, None] * (state.velocity - target) * (1.0 - decay)[:, None]
            )
        else:
            velocity = drift
            position = state.position + step * drift
            if self.motion is FiniteParticleMotionKind.OVERDAMPED_BROWNIAN:
                normal = jax.random.normal(
                    key, state.position.shape, dtype=state.position.dtype
                )
                diagonal = jnp.diagonal(
                    self.properties.diffusion_tensors.astype(state.position.dtype),
                    axis1=-2,
                    axis2=-1,
                )
                position = position + jnp.sqrt(2.0 * step * diagonal) * normal
        position = jnp.where(state.population.active[:, None], position, state.position)
        velocity = jnp.where(state.population.active[:, None], velocity, 0.0)
        finite = jnp.all(jnp.isfinite(fluid_velocity)) & jnp.all(jnp.isfinite(force))
        return position, velocity, finite

    def advance(
        self,
        state: FiniteParticleTransportState,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> FiniteParticleStepResult:
        if state.runtime_id != self._runtime_id():
            raise ValueError("Finite-particle state does not belong to this plan.")
        step = jnp.asarray(step_size, dtype=state.position.dtype)
        if step.shape != ():
            raise ValueError("Finite-particle step size must be scalar.")
        root = jax.random.wrap_key_data(state.key)
        next_key, motion_key = jax.random.split(root)
        free_position, free_velocity, field_finite = self._free_motion(
            state, step, motion_key, args
        )
        start = self.wall.evaluate(state.position, self.properties.radii)
        end = self.wall.evaluate(free_position, self.properties.radii)
        active = state.population.active
        contacted = active & (start.clearance >= 0.0) & (end.clearance < 0.0)
        low = jnp.zeros((self.properties.capacity,), dtype=state.position.dtype)
        high = jnp.ones_like(low)
        segment = free_position - state.position
        regular = jnp.ones_like(active)

        def bisect(_, carry):
            lower, upper, regular_ = carry
            middle = 0.5 * (lower + upper)
            point = state.position + middle[:, None] * segment
            query = self.wall.evaluate(point, self.properties.radii)
            inside = query.clearance >= 0.0
            query_regular = jnp.all(jnp.isfinite(query.inward_normal), axis=-1) & (
                query.feature_margin > 0.0
            )
            lower = jnp.where(contacted & inside, middle, lower)
            upper = jnp.where(contacted & ~inside, middle, upper)
            regular_ = regular_ & (~contacted | query_regular)
            return lower, upper, regular_

        low, high, regular = jax.lax.fori_loop(
            0, self.bisection_steps, bisect, (low, high, regular)
        )
        del high
        contact_point = state.position + low[:, None] * segment
        contact = self.wall.evaluate(contact_point, self.properties.radii)
        contact_normal = contact.inward_normal
        remaining = (1.0 - low)[:, None] * segment
        normal_displacement = contract(
            "pi,pi->p", remaining, contact_normal, backend="jax"
        )
        inward_component = jnp.minimum(normal_displacement, 0.0)
        if self.wall_policy is FiniteParticleWallPolicy.IMPERMEABLE_SLIDE:
            response_displacement = remaining - inward_component[:, None] * contact_normal
            response_velocity = (
                free_velocity
                - jnp.minimum(
                    contract("pi,pi->p", free_velocity, contact_normal, backend="jax"),
                    0.0,
                )[:, None]
                * contact_normal
            )
            absorbed = jnp.zeros_like(active)
        elif self.wall_policy is FiniteParticleWallPolicy.RESTITUTION:
            response_displacement = (
                remaining
                - (1.0 + self.restitution) * inward_component[:, None] * contact_normal
            )
            velocity_normal = contract(
                "pi,pi->p", free_velocity, contact_normal, backend="jax"
            )
            response_velocity = (
                free_velocity
                - (1.0 + self.restitution)
                * jnp.minimum(velocity_normal, 0.0)[:, None]
                * contact_normal
            )
            absorbed = jnp.zeros_like(active)
        else:
            response_displacement = jnp.zeros_like(remaining)
            response_velocity = jnp.zeros_like(free_velocity)
            absorbed = contacted
        response_position = contact_point + response_displacement
        final_position = jnp.where(contacted[:, None], response_position, free_position)
        final_velocity = jnp.where(contacted[:, None], response_velocity, free_velocity)
        final = self.wall.evaluate(final_position, self.properties.radii)
        final_active = active & ~absorbed
        population = ParticlePopulationState(
            final_active,
            jnp.where(final_active, state.population.mass, 0.0),
            state.population.incarnation,
            state.population.ever_occupied,
            state.population.retired
            | (
                absorbed
                & (self.population.reuse_policy is ParticleSlotReusePolicy.NEVER_REUSE)
            ),
        )
        final_position = jnp.where(final_active[:, None], final_position, 0.0)
        final_velocity = jnp.where(final_active[:, None], final_velocity, 0.0)
        finite = (
            field_finite
            & jnp.isfinite(step)
            & jnp.all(jnp.isfinite(final_position))
            & jnp.all(jnp.isfinite(final_velocity))
        )
        initial_ok = jnp.all(~active | start.header.eligible)
        final_ok = jnp.all(~final_active | final.header.eligible)
        event_ok = jnp.all(~contacted | (regular & contact.header.eligible))
        supported = finite & (step > 0.0) & initial_ok & final_ok & event_ok
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            initial_ok,
            reasons,
            reasons
            | jnp.asarray(
                int(FiniteParticleTransportReason.INITIAL_CLEARANCE_INVALID), jnp.uint32
            ),
        )
        reasons = jnp.where(
            final_ok,
            reasons,
            reasons
            | jnp.asarray(
                int(FiniteParticleTransportReason.FINAL_CLEARANCE_INVALID), jnp.uint32
            ),
        )
        reasons = jnp.where(
            event_ok,
            reasons,
            reasons
            | jnp.asarray(
                int(FiniteParticleTransportReason.WALL_EVENT_UNRESOLVED), jnp.uint32
            ),
        )
        reasons = jnp.where(
            step > 0.0,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), jnp.uint32),
        )
        minimum_clearance = jnp.where(
            jnp.any(final_active),
            jnp.min(jnp.where(final_active, final.clearance, jnp.inf)),
            1.0,
        )
        header = AdmissibilityHeader(
            jnp.where(supported, minimum_clearance, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "finite-particle-step-evidence", "plan": self.plan_id}
            ),
        )
        successful = supported & header.globally_eligible
        candidate = FiniteParticleTransportState(
            population,
            final_position,
            final_velocity,
            jnp.where(absorbed, -2, state.terminal_code),
            state.event_count + contacted.astype(jnp.int32),
            state.time + step,
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
            jax.random.key_data(next_key),
            state.runtime_id,
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old),
            candidate,
            state,
        )
        derivative = DerivativeAvailability.WITHIN_FIXED_MODEL
        return FiniteParticleStepResult(
            candidate,
            accepted,
            free_position,
            low,
            contact_point,
            contact_normal,
            contacted,
            absorbed,
            header,
            derivative,
            successful,
            self.plan_id,
        )


__all__ = [
    "FiniteParticleStepResult",
    "FiniteParticleTransportPlan",
    "FiniteParticleTransportReason",
    "FiniteParticleTransportState",
]
