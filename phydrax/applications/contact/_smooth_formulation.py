#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...discretization.contact._kinematics import ContactKinematicsEpoch
from ._closure import ContactClosureEvaluation


class SmoothContactAssembly(StrictModule):
    surface_force: Array
    total_potential: Array
    dissipated_power: Array
    action_reaction_residual: Array
    moment_residual: Array
    finite: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


def assemble_smooth_contact(
    kinematics: ContactKinematicsEpoch,
    closure: ContactClosureEvaluation,
    positions: ArrayLike,
    /,
) -> SmoothContactAssembly:
    if not isinstance(kinematics, ContactKinematicsEpoch):
        raise TypeError("kinematics must be ContactKinematicsEpoch.")
    if not isinstance(closure, ContactClosureEvaluation):
        raise TypeError("closure must be ContactClosureEvaluation.")
    if len(kinematics.batches) != len(closure.batches):
        raise ValueError("Contact kinematics and closure batch counts differ.")
    current = jnp.asarray(positions)
    if current.ndim != 2 or current.shape[1] not in (2, 3):
        raise ValueError("Contact assembly positions require dimension two or three.")
    surface_force = jnp.zeros_like(current)
    for batch, response in zip(kinematics.batches, closure.batches, strict=True):
        tangential_world = jnp.sum(
            batch.tangent_basis * response.tangential.traction[:, None, :],
            axis=-1,
        )
        route_force = (
            response.normal.traction[:, None] * batch.normal + tangential_world
        ) * batch.quadrature_weight[:, None]
        route_force = jnp.where(batch.valid[:, None], route_force, 0.0)
        safe = jnp.clip(batch.vertex_indices, 0, current.shape[0] - 1)
        local_force = batch.coefficients[..., None] * route_force[:, None, :]
        valid_endpoint = batch.vertex_indices >= 0
        local_force = jnp.where(valid_endpoint[..., None], local_force, 0.0)
        surface_force = surface_force.at[safe.reshape((-1,))].add(
            local_force.reshape((-1, current.shape[1]))
        )
    balance = jnp.sum(surface_force, axis=0)
    if current.shape[1] == 3:
        moment = jnp.sum(jnp.cross(current, surface_force), axis=0)
    else:
        moment = jnp.sum(
            current[:, 0] * surface_force[:, 1] - current[:, 1] * surface_force[:, 0]
        )[None]
    finite = (
        jnp.all(jnp.isfinite(surface_force))
        & jnp.all(jnp.isfinite(balance))
        & jnp.all(jnp.isfinite(moment))
        & jnp.isfinite(closure.evidence.total_potential)
        & jnp.isfinite(closure.evidence.total_dissipated_power)
    )
    return SmoothContactAssembly(
        surface_force,
        closure.evidence.total_potential,
        closure.evidence.total_dissipated_power,
        balance,
        moment,
        finite,
        closure.evidence.successful & finite,
        closure.evidence.closure_id,
    )


__all__ = ["SmoothContactAssembly", "assemble_smooth_contact"]
