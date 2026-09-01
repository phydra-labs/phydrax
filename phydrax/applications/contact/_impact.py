#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._kinematics import ContactKinematicsEpoch
from ._cone import ContactConeResult


class RollingSpinningResistancePlan(StrictModule, NonTrainableState):
    rolling_coefficient: float = eqx.field(static=True)
    spinning_coefficient: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        rolling_coefficient: float,
        spinning_coefficient: float,
        regularization: float = 1.0e-10,
    ):
        rolling = float(rolling_coefficient)
        spinning = float(spinning_coefficient)
        regularization_ = float(regularization)
        if (
            any(not np.isfinite(value) or value < 0.0 for value in (rolling, spinning))
            or not np.isfinite(regularization_)
            or regularization_ <= 0.0
        ):
            raise ValueError("Rolling/spinning resistance parameters are invalid.")
        self.rolling_coefficient = rolling
        self.spinning_coefficient = spinning
        self.regularization = regularization_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rolling-spinning-resistance-plan",
                "rolling": rolling.hex(),
                "spinning": spinning.hex(),
                "regularization": regularization_.hex(),
            }
        )


class RollingSpinningResistance(StrictModule):
    rolling_impulse: Array
    spinning_impulse: Array
    dissipated_work: Array
    cone_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def evaluate_rolling_spinning_resistance(
    plan: RollingSpinningResistancePlan,
    normal_impulse: ArrayLike,
    relative_angular_velocity: ArrayLike,
    normal: ArrayLike,
    effective_radius: ArrayLike,
    /,
) -> RollingSpinningResistance:
    if not isinstance(plan, RollingSpinningResistancePlan):
        raise TypeError("plan must be RollingSpinningResistancePlan.")
    impulse = jnp.asarray(normal_impulse)
    angular = jnp.asarray(relative_angular_velocity, dtype=impulse.dtype)
    normal_ = jnp.asarray(normal, dtype=impulse.dtype)
    radius = jnp.asarray(effective_radius, dtype=impulse.dtype)
    if angular.shape != normal_.shape or angular.shape[:-1] != impulse.shape:
        raise ValueError("Rolling/spinning kinematic shapes are invalid.")
    spinning_speed = jnp.sum(angular * normal_, axis=-1)
    rolling_velocity = angular - spinning_speed[..., None] * normal_
    rolling_norm = jnp.sqrt(
        jnp.sum(rolling_velocity * rolling_velocity, axis=-1) + plan.regularization**2
    )
    rolling_limit = plan.rolling_coefficient * jnp.maximum(impulse, 0.0) * radius
    spinning_limit = plan.spinning_coefficient * jnp.maximum(impulse, 0.0) * radius
    rolling_impulse = (
        -rolling_limit[..., None] * rolling_velocity / rolling_norm[..., None]
    )
    spinning_impulse = -spinning_limit * jnp.tanh(spinning_speed / plan.regularization)
    dissipated = -(
        jnp.sum(rolling_impulse * rolling_velocity, axis=-1)
        + spinning_impulse * spinning_speed
    )
    rolling_defect = jnp.maximum(
        jnp.sqrt(jnp.sum(rolling_impulse * rolling_impulse, axis=-1)) - rolling_limit,
        0.0,
    )
    spinning_defect = jnp.maximum(jnp.abs(spinning_impulse) - spinning_limit, 0.0)
    cone_defect = jnp.maximum(rolling_defect, spinning_defect)
    finite = (
        jnp.all(jnp.isfinite(rolling_impulse))
        & jnp.all(jnp.isfinite(spinning_impulse))
        & jnp.all(jnp.isfinite(dissipated))
    )
    successful = finite & jnp.all(dissipated >= -64.0 * jnp.finfo(impulse.dtype).eps)
    return RollingSpinningResistance(
        rolling_impulse,
        spinning_impulse,
        dissipated,
        cone_defect,
        finite,
        successful,
        plan.plan_id,
    )


class ContactImpulseAssembly(StrictModule):
    surface_impulse: Array
    surface_force: Array
    action_reaction_residual: Array
    dissipated_work: Array
    finite: Array
    successful: Array


def assemble_contact_impulses(
    kinematics: ContactKinematicsEpoch,
    result: ContactConeResult,
    positions: ArrayLike,
    step_size: ArrayLike,
    /,
) -> ContactImpulseAssembly:
    if not isinstance(kinematics, ContactKinematicsEpoch):
        raise TypeError("kinematics must be ContactKinematicsEpoch.")
    if not isinstance(result, ContactConeResult):
        raise TypeError("result must be ContactConeResult.")
    current = jnp.asarray(positions)
    dt = jnp.asarray(step_size, dtype=current.dtype)
    if dt.shape != () or bool(dt <= 0.0):
        raise ValueError("step_size must be positive and scalar.")
    surface_impulse = jnp.zeros_like(current)
    offset = 0
    dissipated = jnp.asarray(0.0, dtype=current.dtype)
    for batch in kinematics.batches:
        stop = offset + batch.capacity
        impulse = result.impulse[offset:stop]
        tangent_world = jnp.sum(batch.tangent_basis * impulse[:, None, 1:], axis=-1)
        route_impulse = impulse[:, :1] * batch.normal + tangent_world
        route_impulse = jnp.where(batch.valid[:, None], route_impulse, 0.0)
        safe = jnp.clip(batch.vertex_indices, 0, current.shape[0] - 1)
        local = batch.coefficients[..., None] * route_impulse[:, None, :]
        local = jnp.where((batch.vertex_indices >= 0)[..., None], local, 0.0)
        surface_impulse = surface_impulse.at[safe.reshape((-1,))].add(
            local.reshape((-1, current.shape[1]))
        )
        dissipated = dissipated - jnp.sum(impulse[:, 1:] * batch.tangential_velocity)
        offset = stop
    force = surface_impulse / dt
    balance = jnp.sum(surface_impulse, axis=0)
    finite = (
        jnp.all(jnp.isfinite(surface_impulse))
        & jnp.all(jnp.isfinite(force))
        & jnp.all(jnp.isfinite(balance))
        & jnp.isfinite(dissipated)
    )
    return ContactImpulseAssembly(
        surface_impulse,
        force,
        balance,
        dissipated,
        finite,
        result.evidence.successful & finite,
    )


__all__ = [
    "ContactImpulseAssembly",
    "RollingSpinningResistance",
    "RollingSpinningResistancePlan",
    "assemble_contact_impulses",
    "evaluate_rolling_spinning_resistance",
]
