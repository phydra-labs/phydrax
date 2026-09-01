#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._numerics._compensated import compensated_sum
from ..._tree_math import tree_allfinite, tree_where
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ._boundary import PrescribedGridVelocityResult
from ._contact import MPMGridConstraintResult
from ._phases import advance_grid_velocity, normalize_grid_momentum, update_deformation
from ._schedule import (
    AffineMUSLMPMSchedule,
    PostAdvectionMUSLMPMSchedule,
    USFMPMSchedule,
)
from ._transfer import (
    apic_particle_angular_momentum,
    apic_particle_kinetic_energy,
    build_apic_route_payload,
    gather_apic,
    grid_angular_momentum,
)
from ._types import (
    MPMDiagnostics,
    MPMEnergyLedger,
    MPMGridState,
    MPMParticleState,
    MPMRejectionReason,
    MPMRunStatus,
    MPMRuntimeState,
    MPMScheduleEvidence,
    MPMStepRestriction,
    MPMStepResult,
    MPMTransferEvidence,
)
from ._velocity_transfer import apply_velocity_transfer


def _relative(left: Array, right: Array, /) -> Array:
    scale = jnp.maximum(1.0, jnp.maximum(jnp.linalg.norm(left), jnp.linalg.norm(right)))
    return jnp.linalg.norm(left - right) / scale


def _route_digest(state) -> Array:
    slots = jnp.arange(state.stencil.indices.shape[1], dtype=jnp.int64)[None, :]
    values = jnp.where(
        state.stencil.valid, state.stencil.indices.astype(jnp.int64) + 1, 0
    )
    return jnp.sum(values * (slots + 17))


def _zero_constraint(velocity, mass, dimension):
    return MPMGridConstraintResult(
        velocity,
        jnp.zeros((dimension,), dtype=velocity.dtype),
        jnp.zeros((), dtype=velocity.dtype),
        jnp.zeros((), dtype=velocity.dtype),
        jnp.asarray(jnp.inf, dtype=velocity.dtype),
        jnp.zeros(mass.shape, dtype=bool),
        jnp.zeros(mass.shape, dtype=jnp.int32),
        jnp.asarray(True),
    )


def multifield_step_detailed(dynamics, state, dt, arguments, routes):
    """USL-minus attempt for a fixed field axis; current contact supports K=2."""
    particle = state.particles
    active = (
        dynamics.particles.active_mask
        if state.lifecycle_state is None
        else state.lifecycle_state.active
    )
    dimension = dynamics.dimension
    field_count = dynamics.nodal_fields.field_count
    slots = state.velocity_field_slots
    mass = (
        dynamics.particles.safe_masses.astype(particle.position.dtype)
        if state.lifecycle_state is None
        else state.lifecycle_state.masses.astype(particle.position.dtype)
    )
    external, external_ok = dynamics._external(state.time, particle, arguments)
    p2g_affine = (
        particle.affine_velocity
        if dynamics.method.transfer.requires_affine_state
        else jnp.zeros_like(particle.affine_velocity)
    )
    storage_state = (
        None
        if dynamics.active_blocks is None
        else dynamics.active_blocks.build(routes, state.storage_state)
    )
    mass_results = []
    momenta = []
    internal_forces = []
    external_forces = []
    mass_gradients = []
    normalized = []
    updates = []
    for field in range(field_count):
        owned = active & (slots == field)
        field_mass = jnp.where(owned, mass, 0.0)
        mass_result = dynamics.splat.deposit_content(routes, field_mass)
        payload = build_apic_route_payload(
            routes,
            mass,
            particle.velocity,
            p2g_affine,
            particle.reference_volume,
            particle.first_piola,
            particle.deformation_gradient,
            external,
            owned,
        )
        scattered = dynamics.splat.scatter_route_payload(routes, payload)
        momentum = scattered.values[..., :dimension]
        internal = scattered.values[..., dimension : 2 * dimension]
        field_external = scattered.values[..., 2 * dimension :]
        gradient_payload = (
            mass[:, None, None] * routes.weight_gradients * owned[:, None, None]
        )
        gradient = dynamics.splat.scatter_route_payload(routes, gradient_payload).values
        normalized_field = normalize_grid_momentum(
            mass_result.content,
            momentum,
            mass_tolerance_factor=dynamics.method.mass_tolerance_factor,
        )
        update = advance_grid_velocity(normalized_field, internal, field_external, dt)
        mass_results.append(mass_result)
        momenta.append(momentum)
        internal_forces.append(internal)
        external_forces.append(field_external)
        mass_gradients.append(gradient)
        normalized.append(normalized_field)
        updates.append(update)
    density = mass / jnp.where(active, particle.reference_volume, 1.0)
    scheduled_deformation = particle.deformation_gradient
    schedule_pre_successful = jnp.asarray(True)
    scheduled_material = None
    if isinstance(dynamics.method.schedule, USFMPMSchedule):
        pre_gradient = jnp.zeros_like(particle.deformation_gradient)
        pre_successful = jnp.asarray(True)
        for field in range(field_count):
            owned = active & (slots == field)
            gathered = gather_apic(
                routes,
                normalized[field].velocity.reshape(
                    (dynamics.splat.target_size, dimension)
                ),
                owned,
                (
                    dynamics.method.transfer.maximum_condition
                    if np.isfinite(dynamics.method.transfer.maximum_condition)
                    else 1.0e30
                ),
            )
            pre_gradient = jnp.where(
                owned[:, None, None],
                gathered.velocity_gradient,
                pre_gradient,
            )
            pre_successful = pre_successful & gathered.successful
        scheduled_deformation = update_deformation(
            particle.deformation_gradient, pre_gradient, dt
        )
        scheduled_material = dynamics.material.evaluate(
            scheduled_deformation,
            particle.material_state,
            density,
            arguments.material_parameters,
            state.time + dt,
            dt,
        )
        schedule_pre_successful = pre_successful
        internal_forces = []
        updates = []
        for field in range(field_count):
            owned = active & (slots == field)
            payload = build_apic_route_payload(
                routes,
                mass,
                particle.velocity,
                p2g_affine,
                particle.reference_volume,
                scheduled_material.first_piola,
                scheduled_deformation,
                external,
                owned,
            )
            scattered = dynamics.splat.scatter_route_payload(routes, payload)
            internal = scattered.values[..., dimension : 2 * dimension]
            internal_forces.append(internal)
            updates.append(
                advance_grid_velocity(
                    normalized[field],
                    internal,
                    external_forces[field],
                    dt,
                )
            )
    grid_mass = jnp.stack(tuple(result.content for result in mass_results))
    grid_momentum = jnp.stack(tuple(momenta))
    internal_force = jnp.stack(tuple(internal_forces))
    external_force = jnp.stack(tuple(external_forces))
    mass_gradient = jnp.stack(tuple(mass_gradients))
    velocity_before = jnp.stack(tuple(value.velocity for value in normalized))
    grid_active = jnp.stack(tuple(value.active for value in normalized))
    grid_acceleration = jnp.stack(tuple(value.acceleration for value in updates))
    velocity_trial = jnp.stack(tuple(value.velocity for value in updates))
    if storage_state is not None:
        node_mask = storage_state.active_node_mask
        grid_active = grid_active & node_mask[None, ...]
        velocity_before = jnp.where(grid_active[..., None], velocity_before, 0.0)
        grid_acceleration = jnp.where(grid_active[..., None], grid_acceleration, 0.0)
        velocity_trial = jnp.where(grid_active[..., None], velocity_trial, 0.0)

    rigid_results = []
    rigid_velocity = []
    for field in range(field_count):
        result = (
            dynamics._apply_contact(
                velocity_trial[field], grid_mass[field], state.time, dt, arguments
            )
            if dynamics.contact is not None
            else _zero_constraint(velocity_trial[field], grid_mass[field], dimension)
        )
        rigid_results.append(result)
        rigid_velocity.append(result.velocity)
    constrained = jnp.stack(tuple(rigid_velocity))
    contact_plan = dynamics.nodal_fields.contact_plan
    if contact_plan is not None:
        essential_mask = (
            None
            if dynamics.boundary is None
            else jnp.broadcast_to(dynamics.boundary.mask, constrained.shape)
        )
        essential_values = (
            None
            if dynamics.boundary is None
            else jnp.broadcast_to(
                dynamics.boundary.values.astype(constrained.dtype),
                constrained.shape,
            )
        )
        graph = contact_plan.build_graph(grid_mass, mass_gradient)
        field_contact = contact_plan.solve(
            grid_mass,
            constrained,
            graph,
            dt,
            essential_mask=essential_mask,
            essential_values=essential_values,
        )
        grid_after = field_contact.velocity
        field_contact_successful = field_contact.successful
        field_action_reaction = field_contact.action_reaction_defect
        field_contact_dissipation = field_contact.dissipation
        boundary_results = []
        for field in range(field_count):
            delta = grid_mass[field][..., None] * (grid_after[field] - constrained[field])
            impulse = compensated_sum(delta.reshape((-1, dimension)), axis=0)
            work = compensated_sum(
                0.5
                * grid_mass[field]
                * (
                    jnp.sum(grid_after[field] ** 2, axis=-1)
                    - jnp.sum(constrained[field] ** 2, axis=-1)
                )
            )
            boundary_results.append(
                PrescribedGridVelocityResult(
                    grid_after[field], impulse, work, field_contact.successful
                )
            )
    else:
        field_contact_successful = jnp.asarray(True)
        field_action_reaction = jnp.zeros((), dtype=grid_mass.dtype)
        field_contact_dissipation = jnp.zeros((), dtype=grid_mass.dtype)
        boundary_results = []
        boundary_velocity = []
        for field in range(field_count):
            if dynamics.boundary is None:
                result = PrescribedGridVelocityResult(
                    constrained[field],
                    jnp.zeros((dimension,), dtype=grid_mass.dtype),
                    jnp.zeros((), dtype=grid_mass.dtype),
                    jnp.asarray(True),
                )
            else:
                result = dynamics.boundary.apply(constrained[field], grid_mass[field], dt)
            boundary_results.append(result)
            boundary_velocity.append(result.velocity)
        grid_after = jnp.stack(tuple(boundary_velocity))

    gathers = tuple(
        apply_velocity_transfer(
            dynamics.method.transfer,
            dynamics.method.advection,
            routes,
            velocity_before[field],
            grid_after[field],
            particle.velocity,
            active & (slots == field),
        )
        for field in range(field_count)
    )
    next_velocity = jnp.zeros_like(particle.velocity)
    next_gradient = jnp.zeros_like(particle.deformation_gradient)
    next_advection = jnp.zeros_like(particle.velocity)
    next_affine = jnp.zeros_like(particle.affine_velocity)
    gather_successful = jnp.asarray(True)
    maximum_condition = jnp.zeros((), dtype=grid_mass.dtype)
    for field, gathered in enumerate(gathers):
        owned = active & (slots == field)
        next_velocity = jnp.where(owned[:, None], gathered.velocity, next_velocity)
        next_advection = jnp.where(
            owned[:, None], gathered.advection_velocity, next_advection
        )
        next_gradient = jnp.where(
            owned[:, None, None], gathered.velocity_gradient, next_gradient
        )
        next_affine = jnp.where(
            owned[:, None, None], gathered.affine_velocity, next_affine
        )
        gather_successful = gather_successful & gathered.successful
        maximum_condition = jnp.maximum(
            maximum_condition,
            jnp.max(jnp.where(owned, gathered.condition_estimate, 0.0)),
        )
    second_mass_defect = jnp.zeros((), dtype=grid_mass.dtype)
    second_momentum_defect = jnp.zeros((), dtype=grid_mass.dtype)
    second_constraint_work = jnp.zeros((), dtype=grid_mass.dtype)
    second_successful = jnp.asarray(True)
    second_route_digest = _route_digest(routes)
    if dynamics.method.schedule.second_momentum_extrapolation:
        second_routes = routes
        if isinstance(dynamics.method.schedule, PostAdvectionMUSLMPMSchedule):
            trial_position = particle.position + dt * next_advection
            second_input = dynamics.splat.plan.assignment.update_input(
                trial_position,
                particle.deformation_gradient,
                state.assignment_input,
            )
            second_routes = dynamics.splat.build(
                trial_position, assignment_input=second_input
            )
        second_masses = []
        second_momenta = []
        second_normalized = []
        second_route_digest = _route_digest(second_routes)
        for field in range(field_count):
            owned = active & (slots == field)
            field_mass = jnp.where(owned, mass, 0.0)
            mass_result = dynamics.splat.deposit_content(second_routes, field_mass)
            if (
                isinstance(
                    dynamics.method.schedule,
                    (AffineMUSLMPMSchedule, PostAdvectionMUSLMPMSchedule),
                )
                and dynamics.method.schedule.second_transfer_mode
                == "apic-affine-momentum"
            ):
                payload = build_apic_route_payload(
                    second_routes,
                    mass,
                    next_velocity,
                    next_affine,
                    particle.reference_volume,
                    jnp.zeros_like(particle.first_piola),
                    particle.deformation_gradient,
                    jnp.zeros_like(particle.velocity),
                    owned,
                )
                momentum_result = dynamics.splat.scatter_route_payload(
                    second_routes, payload
                )
                momentum = momentum_result.values[..., :dimension]
                momentum_successful = momentum_result.successful
            else:
                source_momentum = mass[:, None] * next_velocity
                momentum_result = dynamics.splat.deposit_content(
                    second_routes,
                    jnp.where(owned[:, None], source_momentum, 0.0),
                )
                momentum = momentum_result.content
                momentum_successful = momentum_result.successful
            second_masses.append(mass_result)
            second_momenta.append(momentum)
            second_normalized.append(
                normalize_grid_momentum(
                    mass_result.content,
                    momentum,
                    mass_tolerance_factor=dynamics.method.mass_tolerance_factor,
                )
            )
            second_successful = (
                second_successful & mass_result.successful & momentum_successful
            )
        second_mass_grid = jnp.stack(tuple(value.content for value in second_masses))
        second_velocity_grid = jnp.stack(
            tuple(value.velocity for value in second_normalized)
        )
        second_rigid_results = []
        second_rigid_velocity = []
        for field in range(field_count):
            result = dynamics._apply_contact(
                second_velocity_grid[field],
                second_mass_grid[field],
                state.time,
                dt,
                arguments,
            )
            second_rigid_results.append(result)
            second_rigid_velocity.append(result.velocity)
            second_successful = second_successful & result.successful
        second_constrained = jnp.stack(tuple(second_rigid_velocity))
        if dynamics.nodal_fields.contact_plan is not None:
            second_graph = dynamics.nodal_fields.contact_plan.build_graph(
                second_mass_grid, mass_gradient
            )
            essential_mask = (
                None
                if dynamics.boundary is None
                else jnp.broadcast_to(dynamics.boundary.mask, second_constrained.shape)
            )
            essential_values = (
                None
                if dynamics.boundary is None
                else jnp.broadcast_to(
                    dynamics.boundary.values.astype(second_constrained.dtype),
                    second_constrained.shape,
                )
            )
            second_field_contact = dynamics.nodal_fields.contact_plan.solve(
                second_mass_grid,
                second_constrained,
                second_graph,
                dt,
                essential_mask=essential_mask,
                essential_values=essential_values,
            )
            second_after = second_field_contact.velocity
            second_successful = second_successful & second_field_contact.successful
            second_constraint_work = (
                second_constraint_work + second_field_contact.dissipation
            )
        else:
            second_values = []
            for field in range(field_count):
                if dynamics.boundary is None:
                    second_values.append(second_constrained[field])
                else:
                    boundary = dynamics.boundary.apply(
                        second_constrained[field], second_mass_grid[field], dt
                    )
                    second_values.append(boundary.velocity)
                    second_constraint_work = second_constraint_work + boundary.work
                    second_successful = second_successful & boundary.successful
            second_after = jnp.stack(tuple(second_values))
        second_gradient = jnp.zeros_like(next_gradient)
        for field in range(field_count):
            owned = active & (slots == field)
            gathered = gather_apic(
                second_routes,
                second_after[field].reshape((dynamics.splat.target_size, dimension)),
                owned,
                (
                    dynamics.method.transfer.maximum_condition
                    if np.isfinite(dynamics.method.transfer.maximum_condition)
                    else 1.0e30
                ),
            )
            second_gradient = jnp.where(
                owned[:, None, None],
                gathered.velocity_gradient,
                second_gradient,
            )
            second_successful = second_successful & gathered.successful
        next_gradient = second_gradient
        total_particle_mass = jnp.sum(jnp.where(active, mass, 0.0))
        total_second_mass = jnp.sum(second_mass_grid)
        second_mass_defect = jnp.abs(
            total_second_mass - total_particle_mass
        ) / jnp.maximum(1.0, total_particle_mass)
        source_momentum = jnp.sum(
            jnp.where(active[:, None], mass[:, None] * next_velocity, 0.0),
            axis=0,
        )
        target_momentum = jnp.sum(
            jnp.stack(tuple(second_momenta)).reshape((-1, dimension)), axis=0
        )
        second_momentum_defect = _relative(source_momentum, target_momentum)
    candidate_position = particle.position + dt * next_advection
    if isinstance(dynamics.method.schedule, USFMPMSchedule):
        candidate_deformation = scheduled_deformation
        material = scheduled_material
    else:
        candidate_deformation = update_deformation(
            particle.deformation_gradient, next_gradient, dt
        )
        material = dynamics.material.evaluate(
            candidate_deformation,
            particle.material_state,
            density,
            arguments.material_parameters,
            state.time + dt,
            dt,
        )
    identity = jnp.broadcast_to(
        jnp.eye(dimension, dtype=grid_mass.dtype), candidate_deformation.shape
    )
    determinant = solve_small_linear(
        SmallLinearSolvePlan(dimension), candidate_deformation, identity
    ).determinant
    material_ok = jnp.all((~active) | (material.successful & material.admissible))
    jacobian_ok = jnp.all((~active) | (jnp.isfinite(determinant) & (determinant > 0.0)))
    maximum_wave = jnp.max(
        jnp.where(active, particle.maximum_wave_speed, 0.0), initial=0.0
    )
    maximum_velocity = jnp.max(
        jnp.where(active, jnp.linalg.norm(particle.velocity, axis=-1), 0.0), initial=0.0
    )
    maximum_acceleration = jnp.max(
        jnp.where(grid_active, jnp.linalg.norm(grid_acceleration, axis=-1), 0.0),
        initial=0.0,
    )
    tiny = jnp.finfo(grid_mass.dtype).tiny
    acoustic = (
        dynamics.method.acoustic_cfl
        * dynamics.minimum_spacing
        / jnp.maximum(maximum_wave, tiny)
    )
    advective = jnp.where(
        maximum_velocity > 0.0,
        dynamics.method.advective_cfl
        * dynamics.minimum_spacing
        / jnp.maximum(maximum_velocity, tiny),
        jnp.inf,
    )
    force = jnp.where(
        maximum_acceleration > 0.0,
        dynamics.method.force_cfl
        * jnp.sqrt(dynamics.minimum_spacing / jnp.maximum(maximum_acceleration, tiny)),
        jnp.inf,
    )
    contact_limit = jnp.min(
        jnp.stack(tuple(result.contact_step_limit for result in rigid_results))
    )
    material_limit = jnp.min(jnp.where(active, material.suggested_step, jnp.inf))
    limits = jnp.stack((acoustic, advective, force, contact_limit, material_limit))
    selected = jnp.min(limits)
    limiting = jnp.argmin(limits).astype(jnp.int32) + 1
    restriction = MPMStepRestriction(
        acoustic,
        advective,
        force,
        contact_limit,
        material_limit,
        jnp.inf,
        jnp.inf,
        selected,
        limiting,
        selected,
    )
    contact_successful = (
        field_contact_successful
        & jnp.all(jnp.stack(tuple(result.successful for result in rigid_results)))
        & jnp.all(jnp.stack(tuple(result.successful for result in boundary_results)))
    )
    candidate = MPMParticleState(
        candidate_position,
        next_velocity,
        candidate_deformation,
        next_affine,
        particle.reference_volume,
        material.first_piola,
        material.reference_energy_density,
        material.maximum_wave_speed,
        material.trial_state,
    )
    finite = tree_allfinite(candidate)
    successful = (
        external_ok
        & gather_successful
        & schedule_pre_successful
        & second_successful
        & material_ok
        & jacobian_ok
        & contact_successful
        & finite
        & (dt <= selected)
        & jnp.all(jnp.stack(tuple(result.successful for result in mass_results)))
    )
    accepted_particle = tree_where(successful, candidate, particle)
    candidate_input = dynamics.splat.plan.assignment.update_input(
        candidate_position, candidate_deformation, state.assignment_input
    )
    accepted_input = tree_where(successful, candidate_input, state.assignment_input)
    accepted_storage = tree_where(successful, storage_state, state.storage_state)
    status = jnp.where(
        successful,
        int(MPMRunStatus.SUCCESS),
        jnp.where(
            ~contact_successful,
            int(MPMRunStatus.CONTACT_REJECTED),
            jnp.where(
                ~material_ok | ~jacobian_ok,
                int(MPMRunStatus.MATERIAL_REJECTED),
                jnp.where(
                    dt > selected,
                    int(MPMRunStatus.STABILITY_LIMIT_EXCEEDED),
                    int(MPMRunStatus.APIC_MOMENT_FAILED),
                ),
            ),
        ),
    ).astype(jnp.int32)
    accepted_state = MPMRuntimeState(
        accepted_particle,
        jnp.where(successful, state.time + dt, state.time),
        jnp.where(successful, state.accepted_step + 1, state.accepted_step),
        status,
        state.topology_generation,
        accepted_input,
        state.material_slots,
        state.body_ids,
        state.velocity_field_slots,
        accepted_storage,
        state.lifecycle_state,
    )
    candidate_state = MPMRuntimeState(
        candidate,
        state.time + dt,
        state.accepted_step + 1,
        status,
        state.topology_generation,
        candidate_input,
        state.material_slots,
        state.body_ids,
        state.velocity_field_slots,
        storage_state,
        state.lifecycle_state,
    )
    particle_mass = compensated_sum(jnp.where(active, mass, 0.0))
    target_mass = compensated_sum(grid_mass)
    particle_momentum = compensated_sum(
        jnp.where(active[:, None], mass[:, None] * particle.velocity, 0.0), axis=0
    )
    target_momentum = compensated_sum(grid_momentum.reshape((-1, dimension)), axis=0)
    aggregate_momentum = jnp.sum(grid_momentum, axis=0)
    aggregate_active = jnp.any(grid_active, axis=0)
    particle_angular = apic_particle_angular_momentum(
        particle.position,
        particle.velocity,
        particle.affine_velocity,
        mass,
        routes,
        active,
    )
    target_angular = grid_angular_momentum(
        dynamics.grid_coordinates,
        aggregate_momentum.reshape((-1, dimension)),
        aggregate_active.reshape((-1,)),
    )
    angular_valid = jnp.asarray(
        dimension > 1 and not any(dynamics.particle_domain.periodic)
    )
    angular_defect = jnp.where(
        angular_valid, _relative(particle_angular, target_angular), 0.0
    )
    net_internal = compensated_sum(internal_force.reshape((-1, dimension)), axis=0)
    transfer_successful = (_relative(particle_momentum, target_momentum) <= 1.0e-10) & (
        (~angular_valid) | (angular_defect <= 1.0e-10)
    )
    transfer = MPMTransferEvidence(
        particle_mass,
        target_mass,
        jnp.abs(particle_mass - target_mass) / jnp.maximum(1.0, particle_mass),
        particle_momentum,
        target_momentum,
        _relative(particle_momentum, target_momentum),
        particle_angular,
        target_angular,
        angular_valid,
        angular_defect,
        net_internal,
        jnp.max(jnp.abs(routes.partition_sums - 1.0)),
        jnp.max(jnp.abs(routes.gradient_sums)),
        jnp.max(jnp.abs(routes.first_moments)),
        maximum_condition,
        jnp.sum(grid_active, dtype=jnp.int32),
        field_action_reaction,
        field_contact_successful,
        routes.valid_route_count,
        _route_digest(routes),
        transfer_successful,
    )
    particle_kinetic_before = apic_particle_kinetic_energy(
        mass,
        particle.velocity,
        particle.affine_velocity,
        routes.second_moments,
        active,
    )
    particle_kinetic_after = apic_particle_kinetic_energy(
        mass, next_velocity, next_affine, routes.second_moments, active
    )
    grid_kinetic_before = compensated_sum(
        0.5 * grid_mass * jnp.sum(velocity_before**2, axis=-1)
    )
    grid_kinetic_after = compensated_sum(
        0.5 * grid_mass * jnp.sum(grid_after**2, axis=-1)
    )
    material_before = compensated_sum(
        jnp.where(
            active, particle.reference_volume * particle.reference_energy_density, 0.0
        )
    )
    material_after = compensated_sum(
        jnp.where(
            active, particle.reference_volume * material.reference_energy_density, 0.0
        )
    )
    boundary_work = compensated_sum(
        jnp.stack(tuple(result.work for result in boundary_results))
    )
    rigid_work = compensated_sum(
        jnp.stack(tuple(result.work for result in rigid_results))
    )
    rigid_dissipation = compensated_sum(
        jnp.stack(tuple(result.dissipation for result in rigid_results))
    )
    plastic_dissipation = compensated_sum(
        jnp.where(active, particle.reference_volume * material.dissipation_increment, 0.0)
    )
    energy = MPMEnergyLedger(
        particle_kinetic_before,
        grid_kinetic_before,
        grid_kinetic_after,
        particle_kinetic_after,
        material_before,
        material_after,
        jnp.zeros((), dtype=grid_mass.dtype),
        boundary_work,
        rigid_work,
        rigid_dissipation + field_contact_dissipation,
        plastic_dissipation,
        jnp.zeros((), dtype=grid_mass.dtype),
        particle_kinetic_after
        + material_after
        + plastic_dissipation
        - particle_kinetic_before
        - material_before
        - boundary_work
        - rigid_work,
    )
    schedule = MPMScheduleEvidence(
        jnp.asarray(dynamics.method.schedule.schedule_code, dtype=jnp.int32),
        jnp.asarray(isinstance(dynamics.method.schedule, USFMPMSchedule)),
        jnp.asarray(dynamics.method.schedule.second_momentum_extrapolation),
        second_mass_defect,
        second_momentum_defect,
        second_constraint_work,
        second_route_digest
        + jnp.asarray(dynamics.method.schedule.schedule_code * 104729, dtype=jnp.int64),
        schedule_pre_successful & second_successful,
    )
    diagnostics = MPMDiagnostics(
        transfer,
        schedule,
        energy,
        jnp.min(jnp.where(active, determinant, jnp.inf)),
        jnp.max(jnp.where(active, determinant, 0.0)),
        material_ok,
        finite,
    )
    reasons = jnp.where(successful, 0, int(MPMRejectionReason.NONFINITE)).astype(
        jnp.int32
    )
    return MPMStepResult(
        candidate_state,
        accepted_state,
        MPMGridState(
            grid_mass,
            grid_momentum,
            velocity_before,
            internal_force,
            external_force,
            grid_after,
            grid_active,
        ),
        restriction,
        diagnostics,
        successful,
        reasons,
        dt,
        selected - dt,
        ~successful,
        jnp.where(successful, selected, jnp.minimum(selected, 0.5 * dt)),
    )


__all__ = ["multifield_step_detailed"]
