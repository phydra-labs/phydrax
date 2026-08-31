#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ..._numerics._compensated import compensated_sum
from ..._tree_math import tree_allfinite, tree_where
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ._boundary import PrescribedGridVelocityResult
from ._contact import MPMGridConstraintResult
from ._fields import project_two_field_contact
from ._phases import advance_grid_velocity, normalize_grid_momentum, update_deformation
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
    active = dynamics.particles.active_mask
    dimension = dynamics.dimension
    field_count = dynamics.nodal_fields.field_count
    slots = state.velocity_field_slots
    mass = dynamics.particles.safe_masses.astype(particle.position.dtype)
    external, external_ok = dynamics._external(state.time, particle, arguments)
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
            particle.affine_velocity,
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
    if field_count == 2 and dynamics.nodal_fields.contact_friction is not None:
        field_contact = project_two_field_contact(
            grid_mass,
            constrained,
            mass_gradient,
            friction=dynamics.nodal_fields.contact_friction,
        )
        constrained = field_contact.velocity
    else:
        from ._fields import MPMMultifieldContactEvidence

        field_contact = MPMMultifieldContactEvidence(
            constrained,
            jnp.zeros_like(constrained[0]),
            jnp.zeros(grid_mass.shape[1:], dtype=bool),
            jnp.zeros_like(constrained[0]),
            jnp.zeros((), dtype=grid_mass.dtype),
            jnp.zeros((), dtype=grid_mass.dtype),
            jnp.asarray(True),
        )
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
        gather_apic(
            routes,
            grid_after[field].reshape((dynamics.splat.target_size, dimension)),
            active & (slots == field),
            dynamics.method.transfer.maximum_condition,
        )
        for field in range(field_count)
    )
    next_velocity = jnp.zeros_like(particle.velocity)
    next_gradient = jnp.zeros_like(particle.deformation_gradient)
    next_affine = jnp.zeros_like(particle.affine_velocity)
    gather_successful = jnp.asarray(True)
    maximum_condition = jnp.zeros((), dtype=grid_mass.dtype)
    for field, gathered in enumerate(gathers):
        owned = active & (slots == field)
        next_velocity = jnp.where(owned[:, None], gathered.velocity, next_velocity)
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
    candidate_position = particle.position + dt * next_velocity
    candidate_deformation = update_deformation(
        particle.deformation_gradient, next_gradient, dt
    )
    density = mass / jnp.where(active, particle.reference_volume, 1.0)
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
        field_contact.successful
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
        storage_state,
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
    angular_valid = jnp.asarray(not any(dynamics.particle_domain.periodic))
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
        field_contact.action_reaction_defect,
        field_contact.successful,
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
        rigid_dissipation + field_contact.dissipation,
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
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.zeros((), dtype=grid_mass.dtype),
        jnp.zeros((), dtype=grid_mass.dtype),
        jnp.zeros((), dtype=grid_mass.dtype),
        _route_digest(routes),
        jnp.asarray(True),
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
