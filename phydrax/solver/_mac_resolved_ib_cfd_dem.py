#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite, tree_where
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.particle import DEMRuntimeState, RigidSphereKinematics
from ..equations._mac_ib_cfd_dem import (
    evaluate_resolved_mac_ib_cfd_dem,
    ResolvedMACIBCFDEMCouplingPlan,
    ResolvedMACIBEvaluation,
)
from ._structured_incompressible import MACRateProjectionResult


class MACResolvedIBWindowStatus(IntFlag):
    SUCCESS = 0
    INVALID_TIME_STEP = 1
    IB_EVALUATION_FAILED = 2
    DEM_STEP_FAILED = 4
    PRESSURE_PROJECTION_FAILED = 8
    NONFINITE = 16
    MOMENTUM_IDENTITY_FAILED = 32
    WORK_IDENTITY_FAILED = 64


class MACResolvedIBCouplingSchedulePlan(StrictModule, NonTrainableState):
    """Fixed-ratio DEM subcycling inside one atomic MAC macro window."""

    dem_substeps: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, dem_substeps: int, /):
        count = int(dem_substeps)
        if count <= 0:
            raise ValueError("dem_substeps must be positive.")
        self.dem_substeps = count
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "mac-resolved-ib-coupling-schedule",
                "dem_substeps": count,
            }
        )


class MACResolvedIBCouplingState(StrictModule):
    """Atomic resolved MAC--DEM state with accepted cumulative ledgers."""

    dem_state: DEMRuntimeState
    fluid_state: Array
    cumulative_body_impulse: Array
    cumulative_body_angular_impulse: Array
    cumulative_fluid_impulse: Array
    cumulative_fixed_body_reaction_impulse: Array
    cumulative_fixed_body_reaction_angular_impulse: Array
    cumulative_wall_reaction_impulse: Array
    cumulative_projection_reaction_impulse: Array
    cumulative_hydrodynamic_body_work: Array
    cumulative_hydrodynamic_fluid_work: Array
    cumulative_penalty_dissipation: Array
    cumulative_particle_contact_impulse: Array
    cumulative_particle_contact_angular_impulse: Array
    cumulative_boundary_contact_impulse: Array
    cumulative_boundary_contact_angular_impulse: Array
    cumulative_particle_contact_work: Array
    cumulative_boundary_contact_work: Array
    cumulative_prescribed_wall_work: Array
    cumulative_contact_balance_loss: Array
    accepted_windows: Array

    @classmethod
    def initialize(
        cls,
        coupling: ResolvedMACIBCFDEMCouplingPlan,
        dem_state: DEMRuntimeState,
        fluid_state: ArrayLike,
        /,
    ) -> MACResolvedIBCouplingState:
        if not isinstance(coupling, ResolvedMACIBCFDEMCouplingPlan):
            raise TypeError("coupling must be ResolvedMACIBCFDEMCouplingPlan.")
        if not isinstance(dem_state, DEMRuntimeState):
            raise TypeError("dem_state must be DEMRuntimeState.")
        fluid = coupling.fluid.validate_state(fluid_state)
        bodies = coupling.dynamics.bodies
        dtype = fluid.dtype
        body_vector = jnp.zeros((bodies.capacity, bodies.ambient_dimension), dtype=dtype)
        body_angular = jnp.zeros((bodies.capacity, bodies.angular_dimension), dtype=dtype)
        resultant = jnp.zeros((bodies.ambient_dimension,), dtype=dtype)
        scalar = jnp.zeros((), dtype=dtype)
        return cls(
            dem_state=dem_state,
            fluid_state=fluid,
            cumulative_body_impulse=body_vector,
            cumulative_body_angular_impulse=body_angular,
            cumulative_fluid_impulse=resultant,
            cumulative_fixed_body_reaction_impulse=body_vector,
            cumulative_fixed_body_reaction_angular_impulse=body_angular,
            cumulative_wall_reaction_impulse=resultant,
            cumulative_projection_reaction_impulse=resultant,
            cumulative_hydrodynamic_body_work=scalar,
            cumulative_hydrodynamic_fluid_work=scalar,
            cumulative_penalty_dissipation=scalar,
            cumulative_particle_contact_impulse=body_vector,
            cumulative_particle_contact_angular_impulse=body_angular,
            cumulative_boundary_contact_impulse=body_vector,
            cumulative_boundary_contact_angular_impulse=body_angular,
            cumulative_particle_contact_work=scalar,
            cumulative_boundary_contact_work=scalar,
            cumulative_prescribed_wall_work=scalar,
            cumulative_contact_balance_loss=scalar,
            accepted_windows=jnp.zeros((), dtype=jnp.int32),
        )


class MACResolvedIBMacroStepResult(StrictModule):
    """Candidate, rollback, and ledger evidence for one atomic macro window."""

    candidate_state: MACResolvedIBCouplingState
    accepted_state: MACResolvedIBCouplingState
    last_evaluation: ResolvedMACIBEvaluation
    pressure_projection: MACRateProjectionResult
    fluid_source_rate: FaceVelocity
    unprojected_rate: FaceVelocity
    wall_constrained_rate: FaceVelocity
    projected_rate: FaceVelocity
    body_impulse: Array
    body_angular_impulse: Array
    fluid_impulse: Array
    fixed_body_reaction_impulse: Array
    fixed_body_reaction_angular_impulse: Array
    wall_reaction_impulse: Array
    projection_reaction_impulse: Array
    particle_contact_impulse: Array
    particle_contact_angular_impulse: Array
    boundary_contact_impulse: Array
    boundary_contact_angular_impulse: Array
    hydrodynamic_body_work: Array
    hydrodynamic_fluid_work: Array
    penalty_dissipation: Array
    hydrodynamic_work_residual: Array
    particle_contact_work: Array
    boundary_contact_work: Array
    prescribed_wall_work: Array
    contact_balance_loss: Array
    pre_reaction_momentum_residual: Array
    status: Array
    successful: Array
    schedule_id: str = eqx.field(static=True)


def _hydrodynamic_kick(
    coupling: ResolvedMACIBCFDEMCouplingPlan,
    state: DEMRuntimeState,
    force: Array,
    torque: Array,
    scale: Array,
    /,
) -> DEMRuntimeState:
    bodies = coupling.dynamics.bodies
    mobile = (bodies.particles.active_mask & ~bodies.fixed_mask)[:, None]
    velocity = state.kinematics.velocity + scale * (
        bodies.inverse_masses[:, None] * force
    )
    angular_velocity = state.kinematics.angular_velocity + scale * (
        bodies.inverse_inertias[:, None] * torque
    )
    velocity = jnp.where(mobile, velocity, 0.0)
    angular_velocity = jnp.where(mobile, angular_velocity, 0.0)
    kinematics = RigidSphereKinematics(
        state.kinematics.position,
        velocity,
        angular_velocity,
    )
    kinetic_energy = 0.5 * jnp.sum(
        bodies.particles.safe_masses[:, None] * velocity**2
    ) + 0.5 * jnp.sum(bodies.inertias[:, None] * angular_velocity**2)
    updated = eqx.tree_at(lambda value: value.kinematics, state, kinematics)
    return eqx.tree_at(lambda value: value.energy.kinetic_energy, updated, kinetic_energy)


def _face_resultant(
    coupling: ResolvedMACIBCFDEMCouplingPlan,
    value: FaceVelocity,
    /,
) -> Array:
    return jnp.stack(
        tuple(
            jnp.sum(measure * component)
            for measure, component in zip(
                coupling.transfer.operators.face_dual_measures, value, strict=True
            )
        )
    )


def _boundary_loads(state: DEMRuntimeState, /) -> tuple[Array, Array]:
    force = jnp.zeros_like(state.kinematics.position)
    torque = jnp.zeros_like(state.kinematics.angular_velocity)
    for load in state.loads.boundaries:
        force = force + load.force
        torque = torque + load.torque
    return force, torque


def advance_mac_resolved_ib_window(
    coupling: ResolvedMACIBCFDEMCouplingPlan,
    schedule: MACResolvedIBCouplingSchedulePlan,
    state: MACResolvedIBCouplingState,
    time: ArrayLike,
    fluid_step_size: ArrayLike,
    /,
    *,
    args: Any = None,
) -> MACResolvedIBMacroStepResult:
    """Advance one frozen-fluid, fixed-subcycle, atomic MAC--DEM window."""

    if not isinstance(coupling, ResolvedMACIBCFDEMCouplingPlan):
        raise TypeError("coupling must be ResolvedMACIBCFDEMCouplingPlan.")
    if not isinstance(schedule, MACResolvedIBCouplingSchedulePlan):
        raise TypeError("schedule must be MACResolvedIBCouplingSchedulePlan.")
    if not isinstance(state, MACResolvedIBCouplingState):
        raise TypeError("state must be MACResolvedIBCouplingState.")
    fluid_state = coupling.fluid.validate_state(state.fluid_state)
    dtype = fluid_state.dtype
    macro_dt = jnp.asarray(fluid_step_size, dtype=dtype)
    start_time = jnp.asarray(time, dtype=dtype)
    if macro_dt.shape != () or start_time.shape != ():
        raise ValueError("time and fluid_step_size must be scalar.")
    valid_window = jnp.isfinite(macro_dt) & (macro_dt > 0.0) & jnp.isfinite(start_time)
    safe_macro_dt = jnp.where(valid_window, macro_dt, 1.0)
    safe_start_time = jnp.where(jnp.isfinite(start_time), start_time, 0.0)
    dem_dt = safe_macro_dt / schedule.dem_substeps
    indices = jnp.arange(schedule.dem_substeps, dtype=jnp.int32)
    fluid_velocity = coupling.fluid.unpack_velocity(fluid_state)
    bodies = coupling.dynamics.bodies
    zero_face = tuple(jnp.zeros_like(value) for value in fluid_velocity)
    zero_body = jnp.zeros_like(state.dem_state.kinematics.position)
    zero_angular = jnp.zeros_like(state.dem_state.kinematics.angular_velocity)
    zero_scalar = jnp.zeros((), dtype=dtype)
    initial_status = jnp.where(
        valid_window, 0, int(MACResolvedIBWindowStatus.INVALID_TIME_STEP)
    ).astype(jnp.int32)

    def substep(carry, index):
        (
            dem_state,
            fluid_increment,
            body_impulse,
            body_angular_impulse,
            fixed_reaction_impulse,
            fixed_reaction_angular_impulse,
            particle_contact_impulse,
            particle_contact_angular_impulse,
            boundary_contact_impulse,
            boundary_contact_angular_impulse,
            body_work,
            fluid_work,
            dissipation,
            particle_contact_work,
            boundary_contact_work,
            prescribed_wall_work,
            contact_balance_loss,
            prior_success,
            status,
        ) = carry
        subtime = safe_start_time + index.astype(dtype) * dem_dt
        first = evaluate_resolved_mac_ib_cfd_dem(
            coupling, dem_state.kinematics, fluid_velocity, dem_dt
        )
        pre = _hydrodynamic_kick(
            coupling,
            dem_state,
            first.body_force,
            first.body_torque,
            0.5 * dem_dt,
        )
        detail = coupling.dynamics.step_detailed(index, subtime, pre, dem_dt, args)
        second = evaluate_resolved_mac_ib_cfd_dem(
            coupling, detail.accepted_state.kinematics, fluid_velocity, dem_dt
        )
        post = _hydrodynamic_kick(
            coupling,
            detail.accepted_state,
            second.body_force,
            second.body_torque,
            0.5 * dem_dt,
        )
        local_success = first.successful & detail.successful & second.successful
        successful = prior_success & local_success
        accepted_dem = tree_where(successful, post, dem_state)
        take = successful.astype(dtype)

        body_delta = 0.5 * dem_dt * (first.body_force + second.body_force)
        angular_delta = 0.5 * dem_dt * (first.body_torque + second.body_torque)
        fixed_delta = (
            0.5
            * dem_dt
            * (first.fixed_body_reaction_force + second.fixed_body_reaction_force)
        )
        fixed_angular_delta = (
            0.5
            * dem_dt
            * (first.fixed_body_reaction_torque + second.fixed_body_reaction_torque)
        )
        source_delta = tuple(
            0.5 * dem_dt * (left + right)
            for left, right in zip(
                first.fluid_velocity_source_rate,
                second.fluid_velocity_source_rate,
                strict=True,
            )
        )
        next_fluid_increment = tuple(
            current + take * delta
            for current, delta in zip(fluid_increment, source_delta, strict=True)
        )

        prior_particle = dem_state.loads.particle_contact
        next_particle = detail.accepted_state.loads.particle_contact
        particle_delta = 0.5 * dem_dt * (prior_particle.force + next_particle.force)
        particle_angular_delta = (
            0.5 * dem_dt * (prior_particle.torque + next_particle.torque)
        )
        prior_boundary_force, prior_boundary_torque = _boundary_loads(dem_state)
        next_boundary_force, next_boundary_torque = _boundary_loads(detail.accepted_state)
        boundary_delta = 0.5 * dem_dt * (prior_boundary_force + next_boundary_force)
        boundary_angular_delta = (
            0.5 * dem_dt * (prior_boundary_torque + next_boundary_torque)
        )
        body_work_delta = 0.5 * dem_dt * (first.body_work_rate + second.body_work_rate)
        fluid_work_delta = 0.5 * dem_dt * (first.fluid_work_rate + second.fluid_work_rate)
        dissipation_delta = (
            0.5
            * dem_dt
            * (first.penalty_dissipation_rate + second.penalty_dissipation_rate)
        )
        particle_work_delta = detail.energy.particle_contact_work
        boundary_work_delta = jnp.sum(detail.energy.boundary_contact_work)
        wall_work_delta = jnp.sum(detail.energy.prescribed_wall_work)
        balance_delta = detail.energy.contact_balance_loss

        status = status | jnp.where(
            detail.successful, 0, int(MACResolvedIBWindowStatus.DEM_STEP_FAILED)
        ).astype(jnp.int32)
        status = status | jnp.where(
            first.successful & second.successful,
            0,
            int(MACResolvedIBWindowStatus.IB_EVALUATION_FAILED),
        ).astype(jnp.int32)
        next_carry = (
            accepted_dem,
            next_fluid_increment,
            body_impulse + take * body_delta,
            body_angular_impulse + take * angular_delta,
            fixed_reaction_impulse + take * fixed_delta,
            fixed_reaction_angular_impulse + take * fixed_angular_delta,
            particle_contact_impulse + take * particle_delta,
            particle_contact_angular_impulse + take * particle_angular_delta,
            boundary_contact_impulse + take * boundary_delta,
            boundary_contact_angular_impulse + take * boundary_angular_delta,
            body_work + take * body_work_delta,
            fluid_work + take * fluid_work_delta,
            dissipation + take * dissipation_delta,
            particle_contact_work + take * particle_work_delta,
            boundary_contact_work + take * boundary_work_delta,
            prescribed_wall_work + take * wall_work_delta,
            contact_balance_loss + take * balance_delta,
            successful,
            status,
        )
        return next_carry, second

    initial_carry = (
        state.dem_state,
        zero_face,
        zero_body,
        zero_angular,
        zero_body,
        zero_angular,
        zero_body,
        zero_angular,
        zero_body,
        zero_angular,
        zero_scalar,
        zero_scalar,
        zero_scalar,
        zero_scalar,
        zero_scalar,
        zero_scalar,
        zero_scalar,
        valid_window,
        initial_status,
    )
    (
        (
            dem_candidate,
            fluid_increment,
            body_impulse,
            body_angular_impulse,
            fixed_reaction_impulse,
            fixed_reaction_angular_impulse,
            particle_contact_impulse,
            particle_contact_angular_impulse,
            boundary_contact_impulse,
            boundary_contact_angular_impulse,
            body_work,
            fluid_work,
            dissipation,
            particle_contact_work,
            boundary_contact_work,
            prescribed_wall_work,
            contact_balance_loss,
            substeps_successful,
            status,
        ),
        evaluations,
    ) = jax.lax.scan(substep, initial_carry, indices)

    fluid_source_rate = tuple(value / safe_macro_dt for value in fluid_increment)
    base_rate = coupling.fluid.unconstrained_rate(safe_start_time, fluid_state, args)
    unprojected_rate = tuple(
        base + source for base, source in zip(base_rate, fluid_source_rate, strict=True)
    )
    wall_constrained_rate = coupling.fluid.momentum.boundaries.homogeneous_rate(
        unprojected_rate
    )
    projection = coupling.fluid.projection.project_rate(wall_constrained_rate)
    projected_rate = projection.rate
    candidate_velocity = tuple(
        value + safe_macro_dt * rate
        for value, rate in zip(fluid_velocity, projected_rate, strict=True)
    )
    fluid_candidate = coupling.fluid.momentum.operators.velocity_space.flatten(
        candidate_velocity
    )

    fluid_impulse = _face_resultant(coupling, fluid_increment)
    wall_delta_rate = tuple(
        bounded - raw
        for bounded, raw in zip(wall_constrained_rate, unprojected_rate, strict=True)
    )
    projection_delta_rate = tuple(
        projected - bounded
        for projected, bounded in zip(projected_rate, wall_constrained_rate, strict=True)
    )
    wall_reaction_impulse = safe_macro_dt * _face_resultant(coupling, wall_delta_rate)
    projection_reaction_impulse = safe_macro_dt * _face_resultant(
        coupling, projection_delta_rate
    )
    pre_reaction_residual = jnp.sum(body_impulse, axis=0) + fluid_impulse
    hydrodynamic_work_residual = body_work + fluid_work + dissipation
    scale = jnp.maximum(
        1.0,
        jnp.max(
            jnp.stack(
                (
                    jnp.max(jnp.abs(body_impulse)),
                    jnp.max(jnp.abs(fluid_impulse)),
                    jnp.abs(body_work),
                    jnp.abs(fluid_work),
                    jnp.abs(dissipation),
                )
            )
        ),
    )
    tolerance = 4096.0 * jnp.finfo(dtype).eps * scale
    momentum_identity = jnp.max(jnp.abs(pre_reaction_residual)) <= tolerance
    work_identity = jnp.abs(hydrodynamic_work_residual) <= tolerance

    candidate = MACResolvedIBCouplingState(
        dem_state=dem_candidate,
        fluid_state=fluid_candidate,
        cumulative_body_impulse=state.cumulative_body_impulse + body_impulse,
        cumulative_body_angular_impulse=(
            state.cumulative_body_angular_impulse + body_angular_impulse
        ),
        cumulative_fluid_impulse=state.cumulative_fluid_impulse + fluid_impulse,
        cumulative_fixed_body_reaction_impulse=(
            state.cumulative_fixed_body_reaction_impulse + fixed_reaction_impulse
        ),
        cumulative_fixed_body_reaction_angular_impulse=(
            state.cumulative_fixed_body_reaction_angular_impulse
            + fixed_reaction_angular_impulse
        ),
        cumulative_wall_reaction_impulse=(
            state.cumulative_wall_reaction_impulse + wall_reaction_impulse
        ),
        cumulative_projection_reaction_impulse=(
            state.cumulative_projection_reaction_impulse + projection_reaction_impulse
        ),
        cumulative_hydrodynamic_body_work=(
            state.cumulative_hydrodynamic_body_work + body_work
        ),
        cumulative_hydrodynamic_fluid_work=(
            state.cumulative_hydrodynamic_fluid_work + fluid_work
        ),
        cumulative_penalty_dissipation=(
            state.cumulative_penalty_dissipation + dissipation
        ),
        cumulative_particle_contact_impulse=(
            state.cumulative_particle_contact_impulse + particle_contact_impulse
        ),
        cumulative_particle_contact_angular_impulse=(
            state.cumulative_particle_contact_angular_impulse
            + particle_contact_angular_impulse
        ),
        cumulative_boundary_contact_impulse=(
            state.cumulative_boundary_contact_impulse + boundary_contact_impulse
        ),
        cumulative_boundary_contact_angular_impulse=(
            state.cumulative_boundary_contact_angular_impulse
            + boundary_contact_angular_impulse
        ),
        cumulative_particle_contact_work=(
            state.cumulative_particle_contact_work + particle_contact_work
        ),
        cumulative_boundary_contact_work=(
            state.cumulative_boundary_contact_work + boundary_contact_work
        ),
        cumulative_prescribed_wall_work=(
            state.cumulative_prescribed_wall_work + prescribed_wall_work
        ),
        cumulative_contact_balance_loss=(
            state.cumulative_contact_balance_loss + contact_balance_loss
        ),
        accepted_windows=state.accepted_windows + jnp.asarray(1, dtype=jnp.int32),
    )
    finite = tree_allfinite(candidate) & tree_allfinite(projection)
    status = status | jnp.where(
        projection.converged,
        0,
        int(MACResolvedIBWindowStatus.PRESSURE_PROJECTION_FAILED),
    ).astype(jnp.int32)
    status = status | jnp.where(
        finite, 0, int(MACResolvedIBWindowStatus.NONFINITE)
    ).astype(jnp.int32)
    status = status | jnp.where(
        momentum_identity,
        0,
        int(MACResolvedIBWindowStatus.MOMENTUM_IDENTITY_FAILED),
    ).astype(jnp.int32)
    status = status | jnp.where(
        work_identity,
        0,
        int(MACResolvedIBWindowStatus.WORK_IDENTITY_FAILED),
    ).astype(jnp.int32)
    successful = (
        substeps_successful
        & projection.converged
        & finite
        & momentum_identity
        & work_identity
        & (status == int(MACResolvedIBWindowStatus.SUCCESS))
    )
    accepted = tree_where(successful, candidate, state)
    last_evaluation = jax.tree.map(lambda value: value[-1], evaluations)
    return MACResolvedIBMacroStepResult(
        candidate_state=candidate,
        accepted_state=accepted,
        last_evaluation=last_evaluation,
        pressure_projection=projection,
        fluid_source_rate=fluid_source_rate,
        unprojected_rate=unprojected_rate,
        wall_constrained_rate=wall_constrained_rate,
        projected_rate=projected_rate,
        body_impulse=body_impulse,
        body_angular_impulse=body_angular_impulse,
        fluid_impulse=fluid_impulse,
        fixed_body_reaction_impulse=fixed_reaction_impulse,
        fixed_body_reaction_angular_impulse=fixed_reaction_angular_impulse,
        wall_reaction_impulse=wall_reaction_impulse,
        projection_reaction_impulse=projection_reaction_impulse,
        particle_contact_impulse=particle_contact_impulse,
        particle_contact_angular_impulse=particle_contact_angular_impulse,
        boundary_contact_impulse=boundary_contact_impulse,
        boundary_contact_angular_impulse=boundary_contact_angular_impulse,
        hydrodynamic_body_work=body_work,
        hydrodynamic_fluid_work=fluid_work,
        penalty_dissipation=dissipation,
        hydrodynamic_work_residual=hydrodynamic_work_residual,
        particle_contact_work=particle_contact_work,
        boundary_contact_work=boundary_contact_work,
        prescribed_wall_work=prescribed_wall_work,
        contact_balance_loss=contact_balance_loss,
        pre_reaction_momentum_residual=pre_reaction_residual,
        status=status,
        successful=successful,
        schedule_id=schedule.schedule_id,
    )


__all__ = [
    "MACResolvedIBCouplingSchedulePlan",
    "MACResolvedIBCouplingState",
    "MACResolvedIBMacroStepResult",
    "MACResolvedIBWindowStatus",
    "advance_mac_resolved_ib_window",
]
