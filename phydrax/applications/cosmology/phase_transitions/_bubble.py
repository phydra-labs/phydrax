#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class ThinWallBubblePlan(StrictModule, NonTrainableState):
    surface_tension: Array
    vacuum_energy_difference: Array
    inside_mass: Array
    outside_mass: Array
    time_step: Array
    maximum_steps: int = eqx.field(static=True)
    minimum_radius: float = eqx.field(static=True)
    maximum_energy_residual: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        surface_tension: float,
        vacuum_energy_difference: float,
        inside_mass: float,
        outside_mass: float,
        time_step: float,
        maximum_steps: int,
        minimum_radius: float = 1.0e-8,
        maximum_energy_residual: float = 1.0e-6,
    ):
        values = tuple(
            map(
                float,
                (
                    surface_tension,
                    vacuum_energy_difference,
                    inside_mass,
                    outside_mass,
                    time_step,
                    minimum_radius,
                    maximum_energy_residual,
                ),
            )
        )
        steps = int(maximum_steps)
        if (
            any(not math.isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[2] < 0.0
            or values[3] < 0.0
            or values[4] <= 0.0
            or values[5] <= 0.0
            or values[6] < 0.0
            or steps < 1
        ):
            raise ValueError("Thin-wall bubble physical/resource policy is invalid.")
        (
            surface_tension_,
            vacuum_energy_difference_,
            inside_mass_,
            outside_mass_,
            time_step_,
            minimum_radius_,
            maximum_energy_residual_,
        ) = values
        self.surface_tension = jnp.asarray(surface_tension_)
        self.vacuum_energy_difference = jnp.asarray(vacuum_energy_difference_)
        self.inside_mass = jnp.asarray(inside_mass_)
        self.outside_mass = jnp.asarray(outside_mass_)
        self.time_step = jnp.asarray(time_step_)
        self.maximum_steps = steps
        self.minimum_radius = minimum_radius_
        self.maximum_energy_residual = maximum_energy_residual_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "trapped-particle-thin-wall-plan",
                "values": list(values),
                "maximum_steps": steps,
            }
        )


class BubbleParticleEnsemble(StrictModule, NonTrainableState):
    positions: Array
    momenta: Array
    weights: Array
    inside: Array
    active: Array
    valid: Array
    capacity: int = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        momenta: ArrayLike,
        weights: ArrayLike,
        inside: ArrayLike,
        active: ArrayLike,
        /,
    ):
        positions_ = jnp.asarray(positions)
        momenta_ = jnp.asarray(momenta, dtype=positions_.dtype)
        weights_ = jnp.asarray(weights, dtype=positions_.dtype)
        inside_ = jnp.asarray(inside, dtype=jnp.bool_)
        active_ = jnp.asarray(active, dtype=jnp.bool_)
        if (
            positions_.ndim != 2
            or positions_.shape[1] != 3
            or momenta_.shape != positions_.shape
            or weights_.shape != positions_.shape[:1]
            or inside_.shape != weights_.shape
            or active_.shape != weights_.shape
        ):
            raise ValueError(
                "Bubble particle arrays must align with (particle, 3) support."
            )
        valid = (
            jnp.all(jnp.isfinite(positions_), axis=-1)
            & jnp.all(jnp.isfinite(momenta_), axis=-1)
            & jnp.isfinite(weights_)
            & (weights_ >= 0.0)
        )
        self.positions = positions_
        self.momenta = momenta_
        self.weights = jnp.where(active_, weights_, 0.0)
        self.inside = inside_
        self.active = active_
        self.valid = jnp.where(active_, valid, True)
        self.capacity = weights_.shape[0]


class ThinWallBubbleState(StrictModule, NonTrainableState):
    radius: Array
    velocity: Array
    time: Array
    particles: BubbleParticleEnsemble
    initial_total_energy: Array
    accepted: Array
    step_index: Array
    plan_id: str = eqx.field(static=True)


def _wall_energy(plan: ThinWallBubblePlan, radius, velocity):
    gamma = 1.0 / jnp.sqrt(
        jnp.maximum(1.0 - velocity * velocity, jnp.finfo(radius.dtype).tiny)
    )
    area = 4.0 * jnp.pi * radius * radius
    volume = 4.0 * jnp.pi * radius**3 / 3.0
    return area * plan.surface_tension * gamma - volume * plan.vacuum_energy_difference


def _particle_energy(plan: ThinWallBubblePlan, particles: BubbleParticleEnsemble):
    masses = jnp.where(particles.inside, plan.inside_mass, plan.outside_mass)
    energy = jnp.sqrt(
        jnp.sum(particles.momenta * particles.momenta, axis=-1) + masses * masses
    )
    return energy, jnp.sum(
        jnp.where(particles.active & particles.valid, particles.weights * energy, 0.0)
    )


def prepare_thin_wall_bubble(
    plan: ThinWallBubblePlan,
    radius: float,
    velocity: float,
    particles: BubbleParticleEnsemble,
    /,
) -> ThinWallBubbleState:
    if not isinstance(plan, ThinWallBubblePlan) or not isinstance(
        particles, BubbleParticleEnsemble
    ):
        raise TypeError("plan and particles must use phase-transition types.")
    radius_ = jnp.asarray(radius, dtype=particles.positions.dtype)
    velocity_ = jnp.asarray(velocity, dtype=particles.positions.dtype)
    particle_energy, particle_total = _particle_energy(plan, particles)
    del particle_energy
    total = _wall_energy(plan, radius_, velocity_) + particle_total
    accepted = (
        (radius_ > plan.minimum_radius)
        & (jnp.abs(velocity_) < 1.0)
        & jnp.all(particles.valid)
    )
    return ThinWallBubbleState(
        radius_,
        velocity_,
        jnp.asarray(0.0, dtype=radius_.dtype),
        particles,
        total,
        accepted,
        jnp.asarray(0, dtype=jnp.int32),
        plan.plan_id,
    )


class BubbleStepResult(StrictModule, NonTrainableState):
    state: ThinWallBubbleState
    reflected: Array
    transmitted: Array
    pressure: Array
    energy_residual: Array
    derivative_valid: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


def step_thin_wall_bubble(
    plan: ThinWallBubblePlan, state: ThinWallBubbleState, /
) -> BubbleStepResult:
    """Advance one spherical thin-wall step with relativistic reflection/transmission."""
    if (
        not isinstance(plan, ThinWallBubblePlan)
        or not isinstance(state, ThinWallBubbleState)
        or state.plan_id != plan.plan_id
    ):
        raise TypeError("plan and state must describe the same thin-wall model.")
    particles = state.particles
    mass = jnp.where(particles.inside, plan.inside_mass, plan.outside_mass)
    energy = jnp.sqrt(jnp.sum(particles.momenta**2, axis=-1) + mass**2)
    particle_velocity = particles.momenta / jnp.maximum(
        energy[:, None], jnp.finfo(energy.dtype).tiny
    )
    dt = plan.time_step
    a = jnp.sum(particle_velocity**2, axis=-1) - state.velocity**2
    b = 2.0 * (
        jnp.sum(particles.positions * particle_velocity, axis=-1)
        - state.radius * state.velocity
    )
    c = jnp.sum(particles.positions**2, axis=-1) - state.radius**2
    discriminant = b * b - 4.0 * a * c
    linear_time = -c / jnp.where(jnp.abs(b) > jnp.finfo(b.dtype).tiny, b, jnp.inf)
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    quadratic_time_1 = (-b - root) / jnp.where(
        jnp.abs(2.0 * a) > jnp.finfo(a.dtype).tiny, 2.0 * a, jnp.inf
    )
    quadratic_time_2 = (-b + root) / jnp.where(
        jnp.abs(2.0 * a) > jnp.finfo(a.dtype).tiny, 2.0 * a, jnp.inf
    )
    candidate_time = jnp.where(
        jnp.abs(a) <= jnp.finfo(a.dtype).eps,
        linear_time,
        jnp.minimum(
            jnp.where(quadratic_time_1 > 0.0, quadratic_time_1, jnp.inf),
            jnp.where(quadratic_time_2 > 0.0, quadratic_time_2, jnp.inf),
        ),
    )
    crossing = (
        particles.active
        & particles.valid
        & (discriminant >= 0.0)
        & (candidate_time > 0.0)
        & (candidate_time <= dt)
    )
    collision_position = particles.positions + particle_velocity * candidate_time[:, None]
    collision_radius = state.radius + state.velocity * candidate_time
    normal = collision_position / jnp.maximum(
        collision_radius[:, None], jnp.finfo(energy.dtype).tiny
    )
    normal_momentum = jnp.sum(particles.momenta * normal, axis=-1)
    tangential_momentum = particles.momenta - normal_momentum[:, None] * normal
    tangential_squared = jnp.sum(tangential_momentum**2, axis=-1)
    gamma_wall = 1.0 / jnp.sqrt(
        jnp.maximum(1.0 - state.velocity**2, jnp.finfo(energy.dtype).tiny)
    )
    wall_energy = gamma_wall * (energy - state.velocity * normal_momentum)
    wall_normal_momentum = gamma_wall * (normal_momentum - state.velocity * energy)
    destination_inside = ~particles.inside
    destination_mass = jnp.where(destination_inside, plan.inside_mass, plan.outside_mass)
    available_normal_squared = wall_energy**2 - tangential_squared - destination_mass**2
    transmitted = crossing & (available_normal_squared >= 0.0)
    reflected = crossing & ~transmitted
    transmitted_normal = jnp.sign(wall_normal_momentum) * jnp.sqrt(
        jnp.maximum(available_normal_squared, 0.0)
    )
    final_wall_normal = jnp.where(transmitted, transmitted_normal, -wall_normal_momentum)
    final_mass = jnp.where(transmitted, destination_mass, mass)
    final_wall_energy = jnp.sqrt(
        jnp.maximum(final_wall_normal**2 + tangential_squared + final_mass**2, 0.0)
    )
    final_normal = gamma_wall * (final_wall_normal + state.velocity * final_wall_energy)
    final_energy = gamma_wall * (final_wall_energy + state.velocity * final_wall_normal)
    final_momentum = tangential_momentum + final_normal[:, None] * normal
    before_collision = collision_position
    remaining = dt - candidate_time
    final_velocity = final_momentum / jnp.maximum(
        final_energy[:, None], jnp.finfo(final_energy.dtype).tiny
    )
    crossed_position = before_collision + final_velocity * remaining[:, None]
    free_position = particles.positions + particle_velocity * dt
    next_position = jnp.where(crossing[:, None], crossed_position, free_position)
    next_momentum = jnp.where(crossing[:, None], final_momentum, particles.momenta)
    next_inside = jnp.where(transmitted, destination_inside, particles.inside)
    momentum_transfer = jnp.where(crossing, normal_momentum - final_normal, 0.0)
    area = 4.0 * jnp.pi * state.radius**2
    pressure = -jnp.sum(particles.weights * momentum_transfer) / jnp.maximum(
        area, jnp.finfo(area.dtype).tiny
    )
    velocity_element = jnp.maximum(1.0 - state.velocity**2, 0.0)
    acceleration = (
        velocity_element**1.5
        * (plan.vacuum_energy_difference + pressure)
        / plan.surface_tension
        - 2.0 * velocity_element / state.radius
    )
    next_velocity = state.velocity + dt * acceleration
    next_radius = state.radius + dt * state.velocity
    next_particles = BubbleParticleEnsemble(
        next_position, next_momentum, particles.weights, next_inside, particles.active
    )
    _, particle_total = _particle_energy(plan, next_particles)
    total_energy = _wall_energy(plan, next_radius, next_velocity) + particle_total
    residual = (total_energy - state.initial_total_energy) / jnp.maximum(
        jnp.abs(state.initial_total_energy), jnp.finfo(total_energy.dtype).tiny
    )
    accepted = (
        state.accepted
        & (next_radius > plan.minimum_radius)
        & (jnp.abs(next_velocity) < 1.0)
        & jnp.isfinite(residual)
        & (jnp.abs(residual) <= plan.maximum_energy_residual)
    )
    accepted_particles = BubbleParticleEnsemble(
        jnp.where(accepted, next_particles.positions, particles.positions),
        jnp.where(accepted, next_particles.momenta, particles.momenta),
        particles.weights,
        jnp.where(accepted, next_particles.inside, particles.inside),
        particles.active,
    )
    next_state = ThinWallBubbleState(
        jnp.where(accepted, next_radius, state.radius),
        jnp.where(accepted, next_velocity, state.velocity),
        jnp.where(accepted, state.time + dt, state.time),
        accepted_particles,
        state.initial_total_energy,
        accepted,
        state.step_index + accepted.astype(jnp.int32),
        plan.plan_id,
    )
    derivative_valid = particles.active & ~crossing
    return BubbleStepResult(
        next_state,
        reflected,
        transmitted,
        pressure,
        residual,
        derivative_valid,
        accepted,
        plan.plan_id,
    )


class BubbleSimulationResult(StrictModule, NonTrainableState):
    state: ThinWallBubbleState
    radius_history: Array
    velocity_history: Array
    energy_residual_history: Array
    accepted_history: Array
    reflected_count: Array
    transmitted_count: Array
    plan_id: str = eqx.field(static=True)


def simulate_thin_wall_bubble(
    plan: ThinWallBubblePlan, initial: ThinWallBubbleState, /
) -> BubbleSimulationResult:
    if not isinstance(plan, ThinWallBubblePlan) or not isinstance(
        initial, ThinWallBubbleState
    ):
        raise TypeError("plan and initial must use phase-transition types.")

    def step(state, _):
        result = step_thin_wall_bubble(plan, state)
        summary = (
            result.state.radius,
            result.state.velocity,
            result.energy_residual,
            result.accepted,
            jnp.sum(result.reflected, dtype=jnp.int32),
            jnp.sum(result.transmitted, dtype=jnp.int32),
        )
        return result.state, summary

    state, history = jax.lax.scan(step, initial, xs=None, length=plan.maximum_steps)
    radius, velocity, residual, accepted, reflected, transmitted = history
    return BubbleSimulationResult(
        state, radius, velocity, residual, accepted, reflected, transmitted, plan.plan_id
    )


__all__ = [
    "BubbleParticleEnsemble",
    "BubbleSimulationResult",
    "BubbleStepResult",
    "ThinWallBubblePlan",
    "ThinWallBubbleState",
    "prepare_thin_wall_bubble",
    "simulate_thin_wall_bubble",
    "step_thin_wall_bubble",
]
