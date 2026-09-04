#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
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
from ._mac_incompressible import CompiledMACIncompressibleDynamics


class IBPenaltyPlan(StrictModule, NonTrainableState):
    """Linear slip penalty with explicit qualification/acceptance semantics."""

    penalty: float = eqx.field(static=True)
    slip_tolerance: float = eqx.field(static=True)
    require_slip_for_acceptance: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        penalty: float,
        /,
        *,
        slip_tolerance: float = 1.0e-6,
        require_slip_for_acceptance: bool = True,
    ):
        penalty_ = float(penalty)
        tolerance = float(slip_tolerance)
        if (
            not np.isfinite(penalty_)
            or penalty_ <= 0.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Penalty and slip tolerance must be finite and positive.")
        self.penalty = penalty_
        self.slip_tolerance = tolerance
        self.require_slip_for_acceptance = bool(require_slip_for_acceptance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ib-penalty-plan",
                "penalty": penalty_,
                "slip_tolerance": tolerance,
                "require_slip_for_acceptance": bool(require_slip_for_acceptance),
            }
        )


class MACPenaltyIBStatus(IntFlag):
    SUCCESS = 0
    INVALID_STEP_SIZE = 1
    TRANSFER_FAILED = 2
    NONFINITE = 4
    WORK_IDENTITY_FAILED = 8
    FORCE_IDENTITY_FAILED = 16
    SLIP_TOLERANCE_EXCEEDED = 32


class MACPenaltyIBCFDEMCouplingPlan(StrictModule, NonTrainableState):
    """Prepared unit-density MAC penalty coupling to spherical DEM bodies."""

    fluid: CompiledMACIncompressibleDynamics
    dynamics: PreparedSoftSphereDEMDynamics
    marker_owner: Array
    penalty: IBPenaltyPlan
    transfer: PreparedMACMarkerTransfer
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fluid: CompiledMACIncompressibleDynamics,
        dynamics: PreparedSoftSphereDEMDynamics,
        marker_owner: ArrayLike,
        penalty: IBPenaltyPlan,
        transfer: PreparedMACMarkerTransfer,
        /,
    ):
        if not isinstance(fluid, CompiledMACIncompressibleDynamics):
            raise TypeError("fluid must be CompiledMACIncompressibleDynamics.")
        if fluid.algebraic_les is not None:
            raise ValueError(
                "MAC penalty immersed coupling does not support active algebraic LES."
            )
        if not isinstance(dynamics, PreparedSoftSphereDEMDynamics):
            raise TypeError("dynamics must be PreparedSoftSphereDEMDynamics.")
        if not isinstance(penalty, IBPenaltyPlan):
            raise TypeError("penalty must be IBPenaltyPlan.")
        if not isinstance(transfer, PreparedMACMarkerTransfer):
            raise TypeError("transfer must be PreparedMACMarkerTransfer.")
        owner = np.asarray(marker_owner)
        if owner.shape != (transfer.markers.capacity,) or not np.issubdtype(
            owner.dtype, np.integer
        ):
            raise TypeError("marker_owner must be an integer marker-capacity vector.")
        bodies = dynamics.bodies
        active_marker = np.asarray(transfer.markers.active_mask)
        if np.any(owner[active_marker] < 0) or np.any(
            owner[active_marker] >= bodies.capacity
        ):
            raise ValueError("Active marker owners must name valid DEM bodies.")
        if transfer.dimension != bodies.ambient_dimension:
            raise ValueError("MAC transfer and DEM dimensions differ.")
        if transfer.operators.prepared_id != fluid.momentum.operators.prepared_id:
            raise ValueError("MAC transfer and fluid dynamics must share operators.")
        body_active = np.asarray(bodies.particles.active_mask)
        if np.any(~body_active[owner[active_marker]]):
            raise ValueError("Active markers cannot belong to inactive DEM bodies.")
        self.fluid = fluid
        self.dynamics = dynamics
        self.marker_owner = jnp.asarray(owner, dtype=jnp.int32)
        self.penalty = penalty
        self.transfer = transfer
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-penalty-ib-cfd-dem-coupling-plan",
                "fluid": fluid.compilation_id,
                "dynamics": dynamics.prepared_id,
                "marker_owner": array_tree_fingerprint(owner),
                "penalty": penalty.plan_id,
                "transfer": transfer.prepared_id,
            }
        )


class MACPenaltyIBEvaluation(StrictModule):
    """One measure-consistent MAC penalty coupling evaluation."""

    relation: MACMarkerRelation
    transfer_diagnostics: MACMarkerTransferDiagnostics
    marker_position: Array
    marker_velocity: Array
    marker_fluid_velocity: Array
    marker_slip: Array
    fluid_marker_force_density: Array
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
    numerically_valid: Array
    slip_qualified: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _face_resultant(plan: MACPenaltyIBCFDEMCouplingPlan, value: FaceVelocity, /) -> Array:
    return jnp.stack(
        tuple(
            jnp.sum(measure * component)
            for measure, component in zip(
                plan.transfer.operators.face_dual_measures, value, strict=True
            )
        )
    )


def evaluate_mac_penalty_ib_cfd_dem(
    plan: MACPenaltyIBCFDEMCouplingPlan,
    kinematics: RigidSphereKinematics,
    fluid_velocity: FaceVelocity,
    step_size: ArrayLike,
    /,
) -> MACPenaltyIBEvaluation:
    """Evaluate penalty slip, reciprocal loads, and measure-aware transfer."""

    if not isinstance(plan, MACPenaltyIBCFDEMCouplingPlan):
        raise TypeError("plan must be MACPenaltyIBCFDEMCouplingPlan.")
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

    markers = plan.transfer.markers
    active_indices = markers.active_indices
    owner_full = plan.marker_owner
    owner = owner_full[active_indices]
    offset_full = markers.reference_position.astype(dtype)
    offset = offset_full[active_indices]
    marker_position_full = kinematics.position[owner_full] + offset_full
    marker_position = marker_position_full[active_indices]
    marker_velocity = kinematics.velocity[owner] + sphere_spin_velocity(
        kinematics.angular_velocity[owner], offset, bodies.ambient_dimension
    )
    relation = plan.transfer.relation(marker_position_full)
    marker_fluid_velocity = plan.transfer.gather(relation, velocity)
    marker_slip = marker_fluid_velocity - marker_velocity

    fluid_force_density = -plan.penalty.penalty * marker_slip
    marker_weight = markers.material_measure.weights[active_indices].astype(dtype)
    marker_force = -marker_weight[:, None] * fluid_force_density
    fluid_source = plan.transfer.spread(relation, fluid_force_density)
    transfer_diagnostics = plan.transfer.diagnostics(
        relation, velocity, fluid_force_density
    )
    body_force = jnp.zeros_like(kinematics.position).at[owner].add(marker_force)
    marker_torque = sphere_lever_torque(offset, marker_force, bodies.ambient_dimension)
    body_torque = jnp.zeros_like(kinematics.angular_velocity).at[owner].add(marker_torque)
    body_force = jnp.where(bodies.particles.active_mask[:, None], body_force, 0.0)
    body_torque = jnp.where(bodies.particles.active_mask[:, None], body_torque, 0.0)

    fixed = bodies.fixed_mask[:, None]
    fixed_reaction_force = jnp.where(fixed, -body_force, 0.0)
    fixed_reaction_torque = jnp.where(fixed, -body_torque, 0.0)
    marker_fluid_work = jnp.real(
        markers.active_velocity_space.inner(marker_fluid_velocity, fluid_force_density)
    )
    fluid_work = jnp.real(
        plan.transfer.operators.velocity_space.inner(velocity, fluid_source)
    )
    marker_body_work = jnp.sum(marker_velocity * marker_force)
    body_work = jnp.sum(kinematics.velocity * body_force) + jnp.sum(
        kinematics.angular_velocity * body_torque
    )
    work_adjoint_residual = fluid_work - marker_fluid_work
    rigid_work_residual = body_work - marker_body_work
    dissipation = plan.penalty.penalty * jnp.sum(
        marker_weight * jnp.sum(marker_slip * marker_slip, axis=-1)
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
        jnp.all(jnp.isfinite(marker_position))
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
    tolerance = 4096.0 * jnp.finfo(dtype).eps * scale
    work_identity = (
        (jnp.abs(work_adjoint_residual) <= tolerance)
        & (jnp.abs(rigid_work_residual) <= tolerance)
        & (jnp.abs(penalty_work_residual) <= tolerance)
    )
    force_identity = (
        jnp.max(jnp.abs(transfer_diagnostics.force_residual)) <= tolerance
    ) & (jnp.max(jnp.abs(momentum_residual)) <= tolerance)
    numerically_valid = (
        valid_step
        & relation.successful
        & transfer_diagnostics.successful
        & finite
        & work_identity
        & force_identity
    )
    slip_qualified = maximum_slip <= plan.penalty.slip_tolerance

    status = jnp.asarray(int(MACPenaltyIBStatus.SUCCESS), dtype=jnp.int32)
    status = status | jnp.where(
        valid_step, 0, int(MACPenaltyIBStatus.INVALID_STEP_SIZE)
    ).astype(jnp.int32)
    status = status | jnp.where(
        relation.successful & transfer_diagnostics.successful,
        0,
        int(MACPenaltyIBStatus.TRANSFER_FAILED),
    ).astype(jnp.int32)
    status = status | jnp.where(finite, 0, int(MACPenaltyIBStatus.NONFINITE)).astype(
        jnp.int32
    )
    status = status | jnp.where(
        work_identity, 0, int(MACPenaltyIBStatus.WORK_IDENTITY_FAILED)
    ).astype(jnp.int32)
    status = status | jnp.where(
        force_identity, 0, int(MACPenaltyIBStatus.FORCE_IDENTITY_FAILED)
    ).astype(jnp.int32)
    status = status | jnp.where(
        slip_qualified, 0, int(MACPenaltyIBStatus.SLIP_TOLERANCE_EXCEEDED)
    ).astype(jnp.int32)
    accepted_slip = (
        slip_qualified if plan.penalty.require_slip_for_acceptance else jnp.asarray(True)
    )
    successful = numerically_valid & accepted_slip
    return MACPenaltyIBEvaluation(
        relation=relation,
        transfer_diagnostics=transfer_diagnostics,
        marker_position=marker_position,
        marker_velocity=marker_velocity,
        marker_fluid_velocity=marker_fluid_velocity,
        marker_slip=marker_slip,
        fluid_marker_force_density=fluid_force_density,
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
        numerically_valid=numerically_valid,
        slip_qualified=slip_qualified,
        successful=successful,
        plan_id=plan.plan_id,
    )


__all__ = [
    "IBPenaltyPlan",
    "MACPenaltyIBCFDEMCouplingPlan",
    "MACPenaltyIBEvaluation",
    "MACPenaltyIBStatus",
    "evaluate_mac_penalty_ib_cfd_dem",
]
