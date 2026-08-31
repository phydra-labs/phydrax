#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle import PreparedRigidSphereSet, RigidSphereKinematics


class ResolvedIBGeometryPlan(StrictModule, NonTrainableState):
    marker_offset: Array
    marker_owner: Array
    marker_weight: Array
    cell_centers: Array
    support_radius: float = eqx.field(static=True)
    maximum_cells_per_marker: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        marker_offset: ArrayLike,
        marker_owner: ArrayLike,
        marker_weight: ArrayLike,
        cell_centers: ArrayLike,
        support_radius: float,
        maximum_cells_per_marker: int,
        /,
    ):
        offset = np.asarray(marker_offset)
        owner = np.asarray(marker_owner)
        weight = np.asarray(marker_weight)
        cells = np.asarray(cell_centers)
        radius = float(support_radius)
        capacity = int(maximum_cells_per_marker)
        if offset.ndim != 2 or offset.shape[1] not in (2, 3) or offset.shape[0] == 0:
            raise ValueError("marker_offset must have shape (markers,2|3).")
        if owner.shape != (offset.shape[0],) or not np.issubdtype(
            owner.dtype, np.integer
        ):
            raise TypeError("marker_owner must be an integer marker vector.")
        if (
            weight.shape != owner.shape
            or cells.ndim != 2
            or cells.shape[1] != offset.shape[1]
        ):
            raise ValueError("Marker weights and cell centers have incompatible shapes.")
        if (
            np.any(~np.isfinite(offset))
            or np.any(owner < 0)
            or np.any(~np.isfinite(weight))
            or np.any(weight <= 0.0)
            or np.any(~np.isfinite(cells))
            or not np.isfinite(radius)
            or radius <= 0.0
            or capacity <= 0
            or capacity > cells.shape[0]
        ):
            raise ValueError("Resolved-IB marker/grid data or capacities are invalid.")
        self.marker_offset = jnp.asarray(offset)
        self.marker_owner = jnp.asarray(owner, dtype=jnp.int32)
        self.marker_weight = jnp.asarray(weight)
        self.cell_centers = jnp.asarray(cells)
        self.support_radius = radius
        self.maximum_cells_per_marker = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "resolved-ib-geometry-plan",
                "values": array_tree_fingerprint(
                    {
                        "offset": offset,
                        "owner": owner,
                        "weight": weight,
                        "cells": cells,
                    }
                ),
                "support_radius": radius,
                "maximum_cells_per_marker": capacity,
            }
        )


class IBConstraintPlan(StrictModule, NonTrainableState):
    penalty: float = eqx.field(static=True)
    slip_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, penalty: float, /, *, slip_tolerance: float = 1.0e-6):
        penalty_ = float(penalty)
        tolerance = float(slip_tolerance)
        if (
            not np.isfinite(penalty_)
            or penalty_ <= 0.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("IB penalty and slip tolerance must be finite and positive.")
        self.penalty = penalty_
        self.slip_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ib-constraint-plan",
                "penalty": penalty_,
                "slip_tolerance": tolerance,
            }
        )


class ResolvedIBCFDEMCouplingPlan(StrictModule, NonTrainableState):
    bodies: PreparedRigidSphereSet
    geometry: ResolvedIBGeometryPlan
    constraint: IBConstraintPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bodies: PreparedRigidSphereSet,
        geometry: ResolvedIBGeometryPlan,
        constraint: IBConstraintPlan,
        /,
    ):
        if not isinstance(bodies, PreparedRigidSphereSet):
            raise TypeError("bodies must be PreparedRigidSphereSet.")
        if not isinstance(geometry, ResolvedIBGeometryPlan):
            raise TypeError("geometry must be ResolvedIBGeometryPlan.")
        if not isinstance(constraint, IBConstraintPlan):
            raise TypeError("constraint must be IBConstraintPlan.")
        if (
            geometry.marker_offset.shape[1] != bodies.ambient_dimension
            or int(jnp.max(geometry.marker_owner)) >= bodies.capacity
        ):
            raise ValueError("IB marker ownership/dimension does not match bodies.")
        self.bodies = bodies
        self.geometry = geometry
        self.constraint = constraint
        self.plan_id = canonical_fingerprint(
            {
                "kind": "resolved-ib-cfd-dem-plan",
                "bodies": bodies.prepared_id,
                "geometry": geometry.plan_id,
                "constraint": constraint.plan_id,
            }
        )


class ResolvedIBEvaluation(StrictModule):
    body_force: Array
    body_torque: Array
    fluid_momentum_source_rate: Array
    marker_force: Array
    marker_slip: Array
    work_adjoint_residual: Array
    maximum_slip: Array
    capacity_overflow: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def evaluate_resolved_ib_cfd_dem(
    plan: ResolvedIBCFDEMCouplingPlan,
    kinematics: RigidSphereKinematics,
    fluid_velocity: ArrayLike,
    /,
) -> ResolvedIBEvaluation:
    if not isinstance(plan, ResolvedIBCFDEMCouplingPlan):
        raise TypeError("plan must be ResolvedIBCFDEMCouplingPlan.")
    velocity = jnp.asarray(fluid_velocity, dtype=kinematics.position.dtype)
    cells = plan.geometry.cell_centers
    if velocity.shape != cells.shape:
        raise ValueError("fluid_velocity must match IB cell centers.")
    owner = plan.geometry.marker_owner
    offset = plan.geometry.marker_offset
    marker_position = kinematics.position[owner] + offset
    if plan.bodies.ambient_dimension == 2:
        omega = kinematics.angular_velocity[owner, 0]
        spin = jnp.stack((-omega * offset[:, 1], omega * offset[:, 0]), axis=-1)
    else:
        spin = jnp.cross(kinematics.angular_velocity[owner], offset)
    marker_velocity = kinematics.velocity[owner] + spin
    displacement = marker_position[:, None, :] - cells[None, :, :]
    distance = jnp.linalg.norm(displacement, axis=-1)
    q = distance / plan.geometry.support_radius
    raw = jnp.where(q < 1.0, (1.0 - q) ** 4 * (1.0 + 4.0 * q), 0.0)
    support_count = jnp.sum(raw > 0.0, axis=-1, dtype=jnp.int32)
    values, indices = jax.lax.top_k(raw, plan.geometry.maximum_cells_per_marker)
    weight_sum = jnp.sum(values, axis=-1)
    valid = values > 0.0
    weights = jnp.where(
        valid, values / jnp.where(weight_sum > 0.0, weight_sum, 1.0)[:, None], 0.0
    )
    marker_fluid_velocity = jnp.sum(weights[:, :, None] * velocity[indices], axis=1)
    slip = marker_fluid_velocity - marker_velocity
    marker_force = plan.constraint.penalty * plan.geometry.marker_weight[:, None] * slip
    fluid_source = jnp.zeros_like(velocity)
    fluid_payload = -weights[:, :, None] * marker_force[:, None, :]
    fluid_source = fluid_source.at[indices.reshape(-1)].add(
        fluid_payload.reshape((-1, velocity.shape[1]))
    )
    body_force = jnp.zeros_like(kinematics.position)
    body_force = body_force.at[owner].add(marker_force)
    if plan.bodies.ambient_dimension == 2:
        marker_torque = (
            offset[:, 0] * marker_force[:, 1] - offset[:, 1] * marker_force[:, 0]
        )[:, None]
    else:
        marker_torque = jnp.cross(offset, marker_force)
    body_torque = jnp.zeros_like(kinematics.angular_velocity)
    body_torque = body_torque.at[owner].add(marker_torque)
    interpolation_work = jnp.sum(marker_fluid_velocity * marker_force)
    spreading_work = jnp.sum(velocity * fluid_source)
    work_residual = interpolation_work + spreading_work
    maximum_slip = jnp.max(jnp.linalg.norm(slip, axis=-1))
    overflow = support_count > plan.geometry.maximum_cells_per_marker
    successful = (
        ~jnp.any(overflow)
        & jnp.all(weight_sum > 0.0)
        & jnp.all(jnp.isfinite(marker_force))
        & jnp.all(jnp.isfinite(fluid_source))
        & (jnp.abs(work_residual) <= 1.0e-10)
        & (maximum_slip <= plan.constraint.slip_tolerance)
    )
    return ResolvedIBEvaluation(
        body_force,
        body_torque,
        fluid_source,
        marker_force,
        slip,
        work_residual,
        maximum_slip,
        overflow,
        successful,
        plan.plan_id,
    )


__all__ = [
    "IBConstraintPlan",
    "ResolvedIBCFDEMCouplingPlan",
    "ResolvedIBEvaluation",
    "ResolvedIBGeometryPlan",
    "evaluate_resolved_ib_cfd_dem",
]
