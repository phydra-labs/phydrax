#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._cell_list import CellListParticleNeighborhoodPlan
from ._core import ParticleSetPlan
from ._dem import DEMResolvedLoad, DEMRuntimeState, PreparedSoftSphereDEMDynamics
from ._dem_contact_state import remap_dem_contact_history
from ._dem_liquid import DEMLiquidState
from ._neighborhood import DenseParticleNeighborhoodPlan
from ._pair_state import (
    IMPLICIT_BARRIER_INTERACTION,
    match_particle_pair_keys,
    particle_wall_interaction_keys,
    ParticlePairKeySpace,
)
from ._particle_morphology import ParticleDynamicBodyProperties
from ._population import ParticlePopulationState
from ._rigid_sphere import RigidSphereKinematics, RigidSphereSetPlan
from ._verlet import VerletParticleNeighborhoodPlan


class ParticleCapacityStatus(IntEnum):
    SUCCESS = 0
    CAPACITY_LIMIT_REACHED = 1
    MIGRATION_FAILED = 2


class ParticleCapacityGrowthPolicy(StrictModule, NonTrainableState):
    growth_factor: float = eqx.field(static=True)
    minimum_increment: int = eqx.field(static=True)
    maximum_capacity: int = eqx.field(static=True)
    free_slot_trigger: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        growth_factor: float = 2.0,
        minimum_increment: int = 8,
        maximum_capacity: int = 1_000_000,
        free_slot_trigger: int = 0,
    ):
        factor = float(growth_factor)
        increment = int(minimum_increment)
        maximum = int(maximum_capacity)
        trigger = int(free_slot_trigger)
        if (
            not np.isfinite(factor)
            or factor <= 1.0
            or increment <= 0
            or maximum <= 0
            or trigger < 0
        ):
            raise ValueError("Particle capacity growth controls are invalid.")
        self.growth_factor = factor
        self.minimum_increment = increment
        self.maximum_capacity = maximum
        self.free_slot_trigger = trigger
        self.policy_id = canonical_fingerprint(
            {
                "kind": "particle-capacity-growth-policy",
                "growth_factor": factor,
                "minimum_increment": increment,
                "maximum_capacity": maximum,
                "free_slot_trigger": trigger,
            }
        )

    def target_capacity(self, current: int, required_slots: int, /) -> int:
        required = int(required_slots)
        if current <= 0 or required < 0:
            raise ValueError("Current capacity and required slots are invalid.")
        minimum = current + max(required, self.minimum_increment)
        geometric = int(np.ceil(self.growth_factor * current))
        return min(max(minimum, geometric), self.maximum_capacity)


class ParticleCapacityRequest(StrictModule, NonTrainableState):
    required_particle_slots: int = eqx.field(static=True)
    required_pair_capacity: int = eqx.field(static=True)
    required_internal_cells: int = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        required_particle_slots: int,
        /,
        *,
        required_pair_capacity: int = 0,
        required_internal_cells: int = 0,
        reason: str = "particle_event",
    ):
        particles = int(required_particle_slots)
        pairs = int(required_pair_capacity)
        cells = int(required_internal_cells)
        reason_ = str(reason)
        if particles < 0 or pairs < 0 or cells < 0 or not reason_:
            raise ValueError("Particle capacity request is invalid.")
        self.required_particle_slots = particles
        self.required_pair_capacity = pairs
        self.required_internal_cells = cells
        self.reason = reason_
        self.request_id = canonical_fingerprint(
            {
                "kind": "particle-capacity-request",
                "particles": particles,
                "pairs": pairs,
                "cells": cells,
                "reason": reason_,
            }
        )


class ParticleExecutionEpoch(StrictModule):
    dynamics: PreparedSoftSphereDEMDynamics
    state: DEMRuntimeState
    ever_occupied: Array
    retired: Array
    epoch_index: Array
    epoch_id: str = eqx.field(static=True)


class ParticleEpochTransition(StrictModule):
    source_epoch: ParticleExecutionEpoch
    candidate_epoch: ParticleExecutionEpoch
    accepted_epoch: ParticleExecutionEpoch
    old_to_new: Array
    appended_particle_ids: Array
    mass_residual: Array
    momentum_residual: Array
    successful: Array
    status: Array
    transition_id: str = eqx.field(static=True)


def initialize_particle_execution_epoch(
    dynamics: PreparedSoftSphereDEMDynamics,
    state: DEMRuntimeState,
    /,
) -> ParticleExecutionEpoch:
    if not isinstance(dynamics, PreparedSoftSphereDEMDynamics):
        raise TypeError("dynamics must be PreparedSoftSphereDEMDynamics.")
    if not isinstance(state, DEMRuntimeState):
        raise TypeError("state must be DEMRuntimeState.")
    capacity = dynamics.bodies.capacity
    if state.body_properties.active.shape != (capacity,):
        raise ValueError("DEM state does not match dynamics capacity.")
    occupied = jnp.asarray(state.body_properties.active, dtype=bool)
    return ParticleExecutionEpoch(
        dynamics,
        state,
        occupied,
        jnp.zeros_like(occupied),
        jnp.zeros((), dtype=jnp.int32),
        canonical_fingerprint(
            {
                "kind": "particle-execution-epoch",
                "dynamics": dynamics.prepared_id,
                "epoch": 0,
            }
        ),
    )


def _grown_neighborhood_plan(plan, capacity: int, /):
    pair_capacity = capacity * (capacity - 1) // 2
    if isinstance(plan, DenseParticleNeighborhoodPlan):
        return DenseParticleNeighborhoodPlan(pair_capacity, box=plan.box)
    if isinstance(plan, CellListParticleNeighborhoodPlan):
        return CellListParticleNeighborhoodPlan(
            plan.search_radius,
            max(plan.maximum_particles_per_cell, capacity),
            max(plan.maximum_pairs, pair_capacity),
            plan.box,
            maximum_candidate_slots=max(
                plan.maximum_candidate_slots,
                capacity * max(plan.maximum_particles_per_cell, capacity),
            ),
        )
    if isinstance(plan, VerletParticleNeighborhoodPlan):
        return VerletParticleNeighborhoodPlan(
            _grown_neighborhood_plan(plan.base, capacity),
            plan.interaction_radius,
            plan.skin,
        )
    raise TypeError("Automatic epoch growth does not support this neighborhood plan.")


def _pad(array: Array, capacity: int, fill: Any = 0, /) -> Array:
    old = int(array.shape[0])
    if capacity < old:
        raise ValueError("Growth capacity cannot shrink an array.")
    shape = (capacity - old,) + array.shape[1:]
    padding = jnp.full(shape, fill, dtype=array.dtype)
    return jnp.concatenate((array, padding), axis=0)


def _zero_resolved_load(dynamics: PreparedSoftSphereDEMDynamics, /) -> DEMResolvedLoad:
    zero = dynamics._zero_load()
    return DEMResolvedLoad(
        zero,
        tuple(dynamics._zero_load() for _ in dynamics.barriers),
        dynamics._zero_load(),
        dynamics._zero_load(),
        dynamics._zero_load(),
    )


def _remap_boundary_history(
    old_history,
    new_empty,
    old_ids,
    new_ids,
    active,
    /,
):
    old_zeros = jnp.zeros((old_ids.shape[0],), dtype=jnp.int64)
    new_zeros = jnp.zeros((new_ids.shape[0],), dtype=jnp.int64)
    old_keys = particle_wall_interaction_keys(
        old_ids,
        old_zeros,
        old_zeros,
        old_zeros,
        old_history.valid,
        interaction_kind=IMPLICIT_BARRIER_INTERACTION,
    )
    new_keys = particle_wall_interaction_keys(
        new_ids,
        new_zeros,
        new_zeros,
        new_zeros,
        active,
        interaction_kind=IMPLICIT_BARRIER_INTERACTION,
    )
    remap = match_particle_pair_keys(
        old_keys,
        old_history.valid,
        new_keys,
        active,
    )
    remapped = remap_dem_contact_history(old_history, remap, new_keys, active)
    return remapped, remap.successful


def grow_particle_execution_epoch(
    epoch: ParticleExecutionEpoch,
    policy: ParticleCapacityGrowthPolicy,
    request: ParticleCapacityRequest,
    time: ArrayLike,
    /,
    *,
    args: Any = None,
) -> ParticleEpochTransition:
    if not isinstance(epoch, ParticleExecutionEpoch):
        raise TypeError("epoch must be ParticleExecutionEpoch.")
    if not isinstance(policy, ParticleCapacityGrowthPolicy):
        raise TypeError("policy must be ParticleCapacityGrowthPolicy.")
    if not isinstance(request, ParticleCapacityRequest):
        raise TypeError("request must be ParticleCapacityRequest.")
    old_dynamics = epoch.dynamics
    old_state = epoch.state
    old_capacity = old_dynamics.bodies.capacity
    available = int(np.count_nonzero(~np.asarray(epoch.ever_occupied)))
    needed = max(request.required_particle_slots - available, 0)
    target = policy.target_capacity(old_capacity, needed)
    limit_ok = target >= old_capacity + needed and target > old_capacity
    if not limit_ok:
        transition_id = canonical_fingerprint(
            {
                "kind": "particle-epoch-transition",
                "old": epoch.epoch_id,
                "request": request.request_id,
                "status": int(ParticleCapacityStatus.CAPACITY_LIMIT_REACHED),
            }
        )
        return ParticleEpochTransition(
            epoch,
            epoch,
            epoch,
            jnp.arange(old_capacity, dtype=jnp.int32),
            jnp.zeros((0,), dtype=jnp.int64),
            jnp.zeros((), dtype=old_state.kinematics.position.dtype),
            jnp.zeros(
                (old_dynamics.bodies.ambient_dimension,),
                dtype=old_state.kinematics.position.dtype,
            ),
            jnp.asarray(False),
            jnp.asarray(
                int(ParticleCapacityStatus.CAPACITY_LIMIT_REACHED), dtype=jnp.int32
            ),
            transition_id,
        )
    old_ids = old_dynamics.bodies.particles.particle_ids
    first_id = int(np.max(np.asarray(old_ids))) + 1
    appended = jnp.arange(first_id, first_id + target - old_capacity, dtype=jnp.int64)
    new_ids = jnp.concatenate((old_ids, appended))
    static_mass = jnp.where(
        old_state.body_properties.masses > 0.0, old_state.body_properties.masses, 1.0
    )
    new_mass = _pad(static_mass, target, 1.0)
    particles = ParticleSetPlan(
        new_ids,
        new_mass,
        ambient_dimension=old_dynamics.bodies.ambient_dimension,
        active_mask=jnp.ones((target,), dtype=bool),
        coordinate_dtype=old_dynamics.bodies.particles.plan.coordinate_dtype,
    ).prepare(numeric_version=str(int(np.asarray(epoch.epoch_index)) + 1))
    old_body_plan = old_dynamics.bodies.plan
    static_radii = jnp.where(
        old_state.body_properties.radii > 0.0, old_state.body_properties.radii, 1.0
    )
    body_plan = RigidSphereSetPlan(
        _pad(static_radii, target, 1.0),
        _pad(old_body_plan.material_ids, target, 0),
        fixed_mask=_pad(old_body_plan.fixed_mask, target, False),
    )
    bodies = body_plan.prepare(particles)
    neighborhood_plan = _grown_neighborhood_plan(old_dynamics.neighborhood.plan, target)
    neighborhood = neighborhood_plan.prepare(particles)
    new_dynamics = PreparedSoftSphereDEMDynamics(
        bodies,
        neighborhood,
        ParticlePairKeySpace(particles),
        old_dynamics.method.contact.prepare(
            old_dynamics.materials, old_dynamics.bodies.ambient_dimension
        ),
        old_dynamics.method,
        old_dynamics.materials,
        barriers=old_dynamics.barriers,
        gravity=old_dynamics.gravity,
        external_load=old_dynamics.external_load,
        external_load_id=old_dynamics.external_load_id,
        execution=old_dynamics.execution,
        precision=old_dynamics.precision,
    )
    position = _pad(old_state.kinematics.position, target, 0.0)
    velocity = _pad(old_state.kinematics.velocity, target, 0.0)
    angular = _pad(old_state.kinematics.angular_velocity, target, 0.0)
    active = _pad(old_state.body_properties.active, target, False)
    population = ParticlePopulationState(
        active,
        _pad(old_state.body_properties.masses, target, 0.0),
        _pad(old_state.body_properties.population.incarnation, target, 0),
        _pad(old_state.body_properties.population.ever_occupied, target, False),
        _pad(old_state.body_properties.population.retired, target, False),
    )
    properties = ParticleDynamicBodyProperties(
        population,
        _pad(old_state.body_properties.inverse_masses, target, 0.0),
        _pad(old_state.body_properties.radii, target, 0.0),
        _pad(old_state.body_properties.inertias, target, 1.0),
        _pad(old_state.body_properties.inverse_inertias, target, 0.0),
    )
    neighborhood_state = new_dynamics.neighborhood.build(position, active_mask=active)
    new_keys = new_dynamics.pair_key_space.keys(neighborhood_state.pair_relation)
    remap = match_particle_pair_keys(
        old_state.particle_history.pair_keys,
        old_state.particle_history.valid,
        new_keys.keys,
        new_keys.valid,
    )
    particle_history = remap_dem_contact_history(
        old_state.particle_history,
        remap,
        new_keys.keys,
        new_keys.valid,
    )
    boundary_histories = []
    boundary_success = jnp.asarray(True)
    for old_history, new_empty in zip(
        old_state.boundary_histories,
        new_dynamics.empty_boundary_histories(),
        strict=True,
    ):
        migrated, successful = _remap_boundary_history(
            old_history,
            new_empty,
            old_ids,
            new_ids,
            active,
        )
        boundary_histories.append(migrated)
        boundary_success = boundary_success & successful
    cache = (
        new_dynamics.neighborhood.initialize(position, active_mask=active)
        if isinstance(new_dynamics.neighborhood.plan, VerletParticleNeighborhoodPlan)
        else None
    )
    liquid = (
        None
        if old_state.liquid is None
        else DEMLiquidState(
            _pad(old_state.liquid.film_volume, target, 0.0),
            old_state.liquid.cumulative_evaporated_volume,
            old_state.liquid.initial_total_volume,
            old_state.liquid.balance_residual,
            old_state.liquid.successful,
        )
    )
    staged = DEMRuntimeState(
        RigidSphereKinematics(position, velocity, angular),
        properties,
        particle_history,
        tuple(boundary_histories),
        cache,
        _zero_resolved_load(new_dynamics),
        old_state.energy,
        old_state.periodic_cell,
        liquid,
    )
    evaluation = new_dynamics.evaluate(jnp.asarray(time), staged, jnp.asarray(0.0), args)
    candidate_state = DEMRuntimeState(
        staged.kinematics,
        staged.body_properties,
        evaluation.particle_contact.next_history,
        tuple(value.contact.next_history for value in evaluation.boundaries),
        evaluation.neighborhood_cache,
        evaluation.loads,
        staged.energy,
        staged.periodic_cell,
        staged.liquid if evaluation.liquid is None else evaluation.liquid.next_state,
    )
    old_mass = jnp.sum(
        jnp.where(old_state.body_properties.active, old_state.body_properties.masses, 0.0)
    )
    new_mass_total = jnp.sum(jnp.where(active, properties.masses, 0.0))
    mass_residual = new_mass_total - old_mass
    old_momentum = jnp.sum(
        jnp.where(
            old_state.body_properties.active[:, None],
            old_state.body_properties.masses[:, None] * old_state.kinematics.velocity,
            0.0,
        ),
        axis=0,
    )
    new_momentum = jnp.sum(
        jnp.where(active[:, None], properties.masses[:, None] * velocity, 0.0), axis=0
    )
    momentum_residual = new_momentum - old_momentum
    tolerance = 256.0 * jnp.finfo(position.dtype).eps
    successful = (
        remap.successful
        & boundary_success
        & evaluation.successful
        & (jnp.abs(mass_residual) <= tolerance * jnp.maximum(jnp.abs(old_mass), 1.0))
        & jnp.all(
            jnp.abs(momentum_residual)
            <= tolerance * jnp.maximum(jnp.linalg.norm(old_momentum), 1.0)
        )
    )
    next_index = epoch.epoch_index + jnp.asarray(1, dtype=jnp.int32)
    next_epoch_id = canonical_fingerprint(
        {
            "kind": "particle-execution-epoch",
            "dynamics": new_dynamics.prepared_id,
            "parent": epoch.epoch_id,
            "request": request.request_id,
            "epoch": int(np.asarray(next_index)),
        }
    )
    candidate_epoch = ParticleExecutionEpoch(
        new_dynamics,
        candidate_state,
        _pad(epoch.ever_occupied, target, False),
        _pad(epoch.retired, target, False),
        next_index,
        next_epoch_id,
    )
    accepted_epoch = candidate_epoch if bool(np.asarray(successful)) else epoch
    status = jnp.where(
        successful,
        int(ParticleCapacityStatus.SUCCESS),
        int(ParticleCapacityStatus.MIGRATION_FAILED),
    ).astype(jnp.int32)
    transition_id = canonical_fingerprint(
        {
            "kind": "particle-epoch-transition",
            "old": epoch.epoch_id,
            "new": next_epoch_id,
            "request": request.request_id,
            "target_capacity": target,
        }
    )
    return ParticleEpochTransition(
        epoch,
        candidate_epoch,
        accepted_epoch,
        jnp.arange(old_capacity, dtype=jnp.int32),
        appended,
        mass_residual,
        momentum_residual,
        successful,
        status,
        transition_id,
    )


__all__ = [
    "ParticleCapacityGrowthPolicy",
    "ParticleCapacityRequest",
    "ParticleCapacityStatus",
    "ParticleEpochTransition",
    "ParticleExecutionEpoch",
    "grow_particle_execution_epoch",
    "initialize_particle_execution_epoch",
]
