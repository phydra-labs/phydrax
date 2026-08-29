#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dem import DEMEvaluation
from ._pairwise import scatter_pair_exchange, scatter_pair_sum
from ._precision import ParticleAccumulation
from ._rigid_sphere import PreparedRigidSphereSet


class DEMTemperatureState(StrictModule):
    temperature: Array
    cumulative_boundary_heat: Array
    cumulative_mechanical_heat: Array


class DEMContactThermalResponse(StrictModule):
    temperature_rate: Array
    pair_heat_to_left: Array
    boundary_heat_to_particles: Array
    total_internal_exchange: Array
    entropy_production: Array
    step_restriction: Array
    successful: Array


class LumpedContactThermalPlan(StrictModule, NonTrainableState):
    heat_capacity: Array
    conductance: Array
    wall_temperature: Array
    mechanical_heat_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        heat_capacity: ArrayLike,
        conductance: ArrayLike,
        /,
        *,
        wall_temperature: ArrayLike = (),
        mechanical_heat_fraction: float = 0.0,
        plan_id: str | None = None,
    ):
        capacity = np.asarray(heat_capacity)
        conductance_ = np.asarray(conductance)
        wall = np.asarray(wall_temperature)
        fraction = float(mechanical_heat_fraction)
        if capacity.ndim != 1 or capacity.size == 0:
            raise ValueError("heat_capacity must be a nonempty rank-1 array.")
        if conductance_.shape != (capacity.size, capacity.size):
            raise ValueError("conductance must be a square material-pair table.")
        if wall.ndim != 1:
            raise ValueError("wall_temperature must be rank-1.")
        if (
            np.any(~np.isfinite(capacity))
            or np.any(capacity <= 0.0)
            or np.any(~np.isfinite(conductance_))
            or np.any(conductance_ < 0.0)
            or not np.array_equal(conductance_, conductance_.T)
            or np.any(~np.isfinite(wall))
            or not np.isfinite(fraction)
            or fraction < 0.0
            or fraction > 1.0
        ):
            raise ValueError(
                "Thermal capacities, conductance, walls, or heat fraction are invalid."
            )
        generated = canonical_fingerprint(
            {
                "kind": "lumped-contact-thermal-plan",
                "values": array_tree_fingerprint(
                    {
                        "heat_capacity": capacity,
                        "conductance": conductance_,
                        "wall_temperature": wall,
                    }
                ),
                "mechanical_heat_fraction": fraction,
            }
        )
        self.heat_capacity = jnp.asarray(capacity)
        self.conductance = jnp.asarray(conductance_)
        self.wall_temperature = jnp.asarray(wall)
        self.mechanical_heat_fraction = fraction
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def initialize(
        self, bodies: PreparedRigidSphereSet, temperature: ArrayLike, /
    ) -> DEMTemperatureState:
        value = jnp.asarray(temperature, dtype=bodies.radii.dtype)
        if value.shape != (bodies.capacity,):
            raise ValueError("temperature must have particle-capacity shape.")
        value = eqx.error_if(
            value,
            jnp.any(
                bodies.particles.active_mask & (~jnp.isfinite(value) | (value <= 0.0))
            ),
            "Active particle temperatures must be finite and positive.",
        )
        return DEMTemperatureState(
            jnp.where(bodies.particles.active_mask, value, 1.0),
            jnp.zeros((self.wall_temperature.shape[0],), dtype=value.dtype),
            jnp.zeros((), dtype=value.dtype),
        )

    def evaluate(
        self,
        bodies: PreparedRigidSphereSet,
        evaluation: DEMEvaluation,
        state: DEMTemperatureState,
        /,
        *,
        accumulation: ParticleAccumulation = "deterministic",
    ) -> DEMContactThermalResponse:
        if not isinstance(bodies, PreparedRigidSphereSet):
            raise TypeError("bodies must be PreparedRigidSphereSet.")
        if not isinstance(evaluation, DEMEvaluation):
            raise TypeError("evaluation must be DEMEvaluation.")
        if not isinstance(state, DEMTemperatureState):
            raise TypeError("state must be DEMTemperatureState.")
        pairs = evaluation.neighborhood.pair_relation
        left = pairs.left_indices
        right = pairs.right_indices
        material_left = bodies.material_ids[left]
        material_right = bodies.material_ids[right]
        conductance = self.conductance[material_left, material_right]
        active = evaluation.particle_contact.active
        heat_left = conductance * (state.temperature[right] - state.temperature[left])
        heat_left = jnp.where(active, heat_left, 0.0)
        particle_heat = scatter_pair_exchange(
            pairs,
            heat_left,
            size=bodies.capacity,
            accumulation=accumulation,
            valid=active,
        )
        degree = scatter_pair_sum(
            pairs,
            conductance,
            conductance,
            size=bodies.capacity,
            accumulation=accumulation,
            valid=active,
        )
        boundary_heat = jnp.zeros(
            (len(evaluation.boundaries),), dtype=state.temperature.dtype
        )
        for index, response in enumerate(evaluation.boundaries):
            if index >= self.wall_temperature.shape[0]:
                raise ValueError("Thermal wall temperatures do not cover DEM barriers.")
            material_wall = jnp.full(
                (bodies.capacity,), response.material_id, dtype=jnp.int32
            )
            wall_conductance = self.conductance[bodies.material_ids, material_wall]
            wall_heat = wall_conductance * (
                self.wall_temperature[index] - state.temperature
            )
            wall_heat = jnp.where(response.contact.active, wall_heat, 0.0)
            particle_heat = particle_heat + wall_heat
            degree = degree + jnp.where(response.contact.active, wall_conductance, 0.0)
            boundary_heat = boundary_heat.at[index].set(jnp.sum(wall_heat))
        particle_capacity = (
            bodies.particles.safe_masses * self.heat_capacity[bodies.material_ids]
        )
        rate = jnp.where(
            bodies.particles.active_mask,
            particle_heat / particle_capacity,
            0.0,
        )
        restriction_values = jnp.where(degree > 0.0, particle_capacity / degree, jnp.inf)
        restriction = jnp.min(
            jnp.where(bodies.particles.active_mask, restriction_values, jnp.inf)
        )
        entropy = jnp.sum(
            jnp.where(
                active,
                conductance
                * (state.temperature[right] - state.temperature[left]) ** 2
                / (state.temperature[right] * state.temperature[left]),
                0.0,
            )
        )
        internal = jnp.sum(
            scatter_pair_exchange(
                pairs,
                heat_left,
                size=bodies.capacity,
                accumulation=accumulation,
                valid=active,
            )
        )
        successful = (
            evaluation.successful
            & jnp.all(jnp.isfinite(rate))
            & jnp.isfinite(entropy)
            & (entropy >= 0.0)
            & ~jnp.isnan(restriction)
            & (restriction > 0.0)
        )
        return DEMContactThermalResponse(
            rate,
            heat_left,
            boundary_heat,
            internal,
            entropy,
            restriction,
            successful,
        )

    def step(
        self,
        bodies: PreparedRigidSphereSet,
        evaluation: DEMEvaluation,
        state: DEMTemperatureState,
        step_size: Array,
        /,
        *,
        accumulation: ParticleAccumulation = "deterministic",
    ) -> DEMTemperatureState:
        response = self.evaluate(bodies, evaluation, state, accumulation=accumulation)
        candidate = state.temperature + step_size * response.temperature_rate
        candidate = eqx.error_if(
            candidate,
            ~response.successful
            | jnp.any(
                bodies.particles.active_mask
                & (~jnp.isfinite(candidate) | (candidate <= 0.0))
            )
            | (step_size > response.step_restriction),
            "DEM thermal contact step is not admissible.",
        )
        return DEMTemperatureState(
            jnp.where(bodies.particles.active_mask, candidate, 1.0),
            state.cumulative_boundary_heat
            + step_size * response.boundary_heat_to_particles,
            state.cumulative_mechanical_heat,
        )


__all__ = [
    "DEMContactThermalResponse",
    "DEMTemperatureState",
    "LumpedContactThermalPlan",
]
