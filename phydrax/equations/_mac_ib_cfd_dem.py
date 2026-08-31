#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_marker_transfer import (
    MACMarkerRelation,
    MACMarkerTransferDiagnostics,
    PreparedMACMarkerTransfer,
)
from ..discretization.particle import (
    PreparedSoftSphereDEMDynamics,
    RigidSphereKinematics,
    sphere_lever_torque,
    sphere_spin_velocity,
)
from ._ib_cfd_dem import IBConstraintPlan, ResolvedIBGeometryPlan
from ._mac_incompressible import CompiledMACIncompressibleDynamics


class ResolvedMACIBStatus(IntFlag):
    SUCCESS = 0
    INVALID_STEP_SIZE = 1
    TRANSFER_FAILED = 2
    NONFINITE = 4
    WORK_IDENTITY_FAILED = 8
    FORCE_IDENTITY_FAILED = 16
    SLIP_TOLERANCE_EXCEEDED = 32


class ResolvedMACIBCFDEMCouplingPlan(StrictModule, NonTrainableState):
    """Prepared unit-density resolved IB coupling between MAC flow and DEM."""

    fluid: CompiledMACIncompressibleDynamics
    dynamics: PreparedSoftSphereDEMDynamics
    geometry: ResolvedIBGeometryPlan
    constraint: IBConstraintPlan
    transfer: PreparedMACMarkerTransfer
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fluid: CompiledMACIncompressibleDynamics,
        dynamics: PreparedSoftSphereDEMDynamics,
        geometry: ResolvedIBGeometryPlan,
        constraint: IBConstraintPlan,
        transfer: PreparedMACMarkerTransfer,
        /,
    ):
        if not isinstance(fluid, CompiledMACIncompressibleDynamics):
            raise TypeError("fluid must be CompiledMACIncompressibleDynamics.")
        if not isinstance(dynamics, PreparedSoftSphereDEMDynamics):
            raise TypeError("dynamics must be PreparedSoftSphereDEMDynamics.")
        if not isinstance(geometry, ResolvedIBGeometryPlan):
            raise TypeError("geometry must be ResolvedIBGeometryPlan.")
        if not isinstance(constraint, IBConstraintPlan):
            raise TypeError("constraint must be IBConstraintPlan.")
        if not isinstance(transfer, PreparedMACMarkerTransfer):
            raise TypeError("transfer must be PreparedMACMarkerTransfer.")
        bodies = dynamics.bodies
        if geometry.marker_offset.shape[1] != bodies.ambient_dimension:
            raise ValueError("IB marker and DEM dimensions differ.")
        if int(jnp.max(geometry.marker_owner)) >= bodies.capacity:
            raise ValueError("IB marker owner exceeds the DEM body capacity.")
        if transfer.dimension != bodies.ambient_dimension:
            raise ValueError("MAC transfer and DEM dimensions differ.")
        if transfer.operators.prepared_id != fluid.momentum.operators.prepared_id:
            raise ValueError("MAC transfer and fluid dynamics must share operators.")
        self.fluid = fluid
        self.dynamics = dynamics
        self.geometry = geometry
        self.constraint = constraint
        self.transfer = transfer
        self.plan_id = canonical_fingerprint(
            {
                "kind": "resolved-mac-ib-cfd-dem-coupling-plan",
                "fluid": fluid.compilation_id,
                "dynamics": dynamics.prepared_id,
                "geometry": geometry.plan_id,
                "constraint": constraint.plan_id,
                "transfer": transfer.prepared_id,
            }
        )


class ResolvedMACIBEvaluation(StrictModule):
    """One fail-closed resolved MAC penalty-IB evaluation."""

    relation: MACMarkerRelation
    transfer_diagnostics: MACMarkerTransferDiagnostics
    marker_position: Array
    marker_velocity: Array
    marker_fluid_velocity: Array
    marker_slip: Array
    marker_force: Array
    body_force: Array
    body_torque: Array
    fluid_velocity_source_rate: FaceVelocity
    marker_impulse: Array
    body_impulse: Array
    body_angular_impulse: Array
    fluid_velocity_increment: FaceVelocity
    fluid_impulse: Array
    fixed_body_reaction_force: Array
    fixed_body_reaction_torque: Array
    fixed_body_reaction_impulse: Array
    fixed_body_reaction_angular_impulse: Array
    body_work_rate: Array
    fluid_work_rate: Array
    penalty_dissipation_rate: Array
    work_adjoint_residual: Array
    rigid_work_residual: Array
    penalty_work_residual: Array
    momentum_residual: Array
    maximum_slip: Array
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _face_resultant(
    plan: ResolvedMACIBCFDEMCouplingPlan,
    value: FaceVelocity,
    /,
) -> Array:
    return jnp.stack(
        tuple(
            jnp.sum(measure * component)
            for measure, component in zip(
                plan.transfer.operators.face_dual_measures, value, strict=True
            )
        )
    )


def evaluate_resolved_mac_ib_cfd_dem(
    plan: ResolvedMACIBCFDEMCouplingPlan,
    kinematics: RigidSphereKinematics,
    fluid_velocity: FaceVelocity,
    step_size: ArrayLike,
    /,
) -> ResolvedMACIBEvaluation:
    """Evaluate componentwise MAC penalty coupling and exact opposite impulses."""

    if not isinstance(plan, ResolvedMACIBCFDEMCouplingPlan):
        raise TypeError("plan must be ResolvedMACIBCFDEMCouplingPlan.")
    if not isinstance(kinematics, RigidSphereKinematics):
        raise TypeError("kinematics must be RigidSphereKinematics.")
    bodies = plan.dynamics.bodies
    expected_position = (bodies.capacity, bodies.ambient_dimension)
    expected_angular = (bodies.capacity, bodies.angular_dimension)
    if (
        kinematics.position.shape != expected_position
        or kinematics.velocity.shape != expected_position
        or kinematics.angular_velocity.shape != expected_angular
    ):
        raise ValueError("Rigid-sphere kinematics do not match the coupling bodies.")
    velocity = plan.transfer.operators.validate_velocity(fluid_velocity)
    dtype = plan.transfer.operators.pressure_space.dtype
    step = jnp.asarray(step_size, dtype=dtype)
    if step.shape != ():
        raise ValueError("step_size must be scalar.")
    valid_step = jnp.isfinite(step) & (step > 0.0)
    accepted_step = jnp.where(valid_step, step, 0.0)

    owner = plan.geometry.marker_owner
    offset = plan.geometry.marker_offset.astype(dtype)
    marker_active = bodies.particles.active_mask[owner]
    marker_position = kinematics.position[owner] + offset
    marker_velocity = kinematics.velocity[owner] + sphere_spin_velocity(
        kinematics.angular_velocity[owner], offset, bodies.ambient_dimension
    )
    marker_velocity = jnp.where(marker_active[:, None], marker_velocity, 0.0)
    relation = plan.transfer.relation(marker_position, active_mask=marker_active)
    marker_fluid_velocity = plan.transfer.gather(relation, velocity)
    marker_slip = jnp.where(
        marker_active[:, None], marker_fluid_velocity - marker_velocity, 0.0
    )
    marker_force = (
        plan.constraint.penalty
        * plan.geometry.marker_weight.astype(dtype)[:, None]
        * marker_slip
    )
    marker_force = jnp.where(marker_active[:, None], marker_force, 0.0)

    transfer_diagnostics = plan.transfer.diagnostics(relation, velocity, marker_force)
    fluid_source = plan.transfer.spread(relation, -marker_force)
    body_force = jnp.zeros_like(kinematics.position).at[owner].add(marker_force)
    marker_torque = sphere_lever_torque(offset, marker_force, bodies.ambient_dimension)
    body_torque = jnp.zeros_like(kinematics.angular_velocity).at[owner].add(marker_torque)
    body_force = jnp.where(bodies.particles.active_mask[:, None], body_force, 0.0)
    body_torque = jnp.where(bodies.particles.active_mask[:, None], body_torque, 0.0)

    fixed = bodies.fixed_mask[:, None]
    fixed_reaction_force = jnp.where(fixed, -body_force, 0.0)
    fixed_reaction_torque = jnp.where(fixed, -body_torque, 0.0)
    marker_fluid_work = jnp.sum(marker_fluid_velocity * marker_force)
    fluid_work = jnp.real(
        plan.transfer.operators.velocity_space.inner(velocity, fluid_source)
    )
    marker_body_work = jnp.sum(marker_velocity * marker_force)
    body_work = jnp.sum(kinematics.velocity * body_force) + jnp.sum(
        kinematics.angular_velocity * body_torque
    )
    work_adjoint_residual = fluid_work + marker_fluid_work
    rigid_work_residual = body_work - marker_body_work
    dissipation = plan.constraint.penalty * jnp.sum(
        plan.geometry.marker_weight.astype(dtype)
        * jnp.sum(marker_slip * marker_slip, axis=-1)
    )
    penalty_work_residual = body_work + fluid_work + dissipation

    marker_impulse = accepted_step * marker_force
    body_impulse = accepted_step * body_force
    body_angular_impulse = accepted_step * body_torque
    fluid_increment = tuple(accepted_step * value for value in fluid_source)
    fluid_impulse = _face_resultant(plan, fluid_increment)
    fixed_reaction_impulse = accepted_step * fixed_reaction_force
    fixed_reaction_angular_impulse = accepted_step * fixed_reaction_torque
    momentum_residual = jnp.sum(body_impulse, axis=0) + fluid_impulse
    maximum_slip = jnp.max(jnp.linalg.norm(marker_slip, axis=-1))

    finite = (
        jnp.all(jnp.isfinite(marker_position) | ~marker_active[:, None])
        & jnp.all(jnp.isfinite(marker_velocity))
        & jnp.all(jnp.isfinite(marker_force))
        & jnp.all(jnp.isfinite(body_force))
        & jnp.all(jnp.isfinite(body_torque))
        & jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(component)) for component in fluid_source)
            )
        )
        & jnp.isfinite(body_work)
        & jnp.isfinite(fluid_work)
        & jnp.isfinite(dissipation)
    )
    scale = jnp.maximum(
        1.0,
        jnp.max(
            jnp.stack(
                (
                    jnp.abs(body_work),
                    jnp.abs(fluid_work),
                    jnp.abs(dissipation),
                    jnp.max(jnp.abs(body_impulse)),
                    jnp.max(jnp.abs(fluid_impulse)),
                )
            )
        ),
    )
    tolerance = 2048.0 * jnp.finfo(dtype).eps * scale
    work_identity = (
        (jnp.abs(work_adjoint_residual) <= tolerance)
        & (jnp.abs(rigid_work_residual) <= tolerance)
        & (jnp.abs(penalty_work_residual) <= tolerance)
    )
    force_identity = (
        jnp.max(jnp.abs(transfer_diagnostics.force_residual)) <= tolerance
    ) & (jnp.max(jnp.abs(momentum_residual)) <= tolerance)

    status = jnp.asarray(int(ResolvedMACIBStatus.SUCCESS), dtype=jnp.int32)
    status = status | jnp.where(
        valid_step, 0, int(ResolvedMACIBStatus.INVALID_STEP_SIZE)
    ).astype(jnp.int32)
    status = status | jnp.where(
        relation.successful & transfer_diagnostics.successful,
        0,
        int(ResolvedMACIBStatus.TRANSFER_FAILED),
    ).astype(jnp.int32)
    status = status | jnp.where(finite, 0, int(ResolvedMACIBStatus.NONFINITE)).astype(
        jnp.int32
    )
    status = status | jnp.where(
        work_identity, 0, int(ResolvedMACIBStatus.WORK_IDENTITY_FAILED)
    ).astype(jnp.int32)
    status = status | jnp.where(
        force_identity, 0, int(ResolvedMACIBStatus.FORCE_IDENTITY_FAILED)
    ).astype(jnp.int32)
    status = status | jnp.where(
        maximum_slip <= plan.constraint.slip_tolerance,
        0,
        int(ResolvedMACIBStatus.SLIP_TOLERANCE_EXCEEDED),
    ).astype(jnp.int32)
    successful = status == int(ResolvedMACIBStatus.SUCCESS)
    return ResolvedMACIBEvaluation(
        relation=relation,
        transfer_diagnostics=transfer_diagnostics,
        marker_position=marker_position,
        marker_velocity=marker_velocity,
        marker_fluid_velocity=marker_fluid_velocity,
        marker_slip=marker_slip,
        marker_force=marker_force,
        body_force=body_force,
        body_torque=body_torque,
        fluid_velocity_source_rate=fluid_source,
        marker_impulse=marker_impulse,
        body_impulse=body_impulse,
        body_angular_impulse=body_angular_impulse,
        fluid_velocity_increment=fluid_increment,
        fluid_impulse=fluid_impulse,
        fixed_body_reaction_force=fixed_reaction_force,
        fixed_body_reaction_torque=fixed_reaction_torque,
        fixed_body_reaction_impulse=fixed_reaction_impulse,
        fixed_body_reaction_angular_impulse=fixed_reaction_angular_impulse,
        body_work_rate=body_work,
        fluid_work_rate=fluid_work,
        penalty_dissipation_rate=dissipation,
        work_adjoint_residual=work_adjoint_residual,
        rigid_work_residual=rigid_work_residual,
        penalty_work_residual=penalty_work_residual,
        momentum_residual=momentum_residual,
        maximum_slip=maximum_slip,
        status=status,
        successful=successful,
        plan_id=plan.plan_id,
    )


__all__ = [
    "ResolvedMACIBCFDEMCouplingPlan",
    "ResolvedMACIBEvaluation",
    "ResolvedMACIBStatus",
    "evaluate_resolved_mac_ib_cfd_dem",
]
